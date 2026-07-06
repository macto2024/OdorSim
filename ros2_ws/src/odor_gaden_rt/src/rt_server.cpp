/*-----------------------------------------------------------------------------
 * odor_gaden_rt: real-time GADEN server with moving gas sources.
 *
 * Wraps a live gaden::Scene (RunningSimulation instances) instead of the
 * pre-baked snapshots used by gaden_player, so source positions can be
 * mutated while the dispersion simulation runs. Designed to run in lockstep
 * with an external physics sim (robosuite):
 *
 *   sub  /gaden/source_poses  geometry_msgs/PoseArray  pose i -> source i
 *   sub  /gaden/step          std_msgs/Empty           one AdvanceTimestep()
 *   pub  /gaden/sim_time      std_msgs/Float32         after every step
 *   srv  /odor_value          gaden_msgs/GasPosition   per-gas ppm at points
 *   srv  /wind_value          gaden_msgs/WindPosition  wind vector at points
 *
 * Parameters:
 *   scenarioPath (string, required): environment configuration directory
 *       (contains config.yaml, simulations/, scenes/). STL geometry and
 *       uniform wind are preprocessed in-memory at startup.
 *   sceneID (string, "scene1"): which scene (source list) to run.
 *   stepOnTimer (bool, false): advance on a wall timer at 1/deltaTime Hz
 *       instead of waiting for /gaden/step (standalone / RViz mode).
 *   publishMarkers (bool, true): RViz filament + source markers.
 *---------------------------------------------------------------------------*/
#define GADEN_LOGGER_ID "OdorGadenRT"

#include <gaden/EnvironmentConfigMetadata.hpp>
#include <gaden/Preprocessing.hpp>
#include <gaden/RunningSimulation.hpp>
#include <gaden/Scene.hpp>
#include <gaden_common/Utils.hpp>
#include <gaden_common/Visualization.hpp>

#include <gaden_msgs/srv/gas_position.hpp>
#include <gaden_msgs/srv/wind_position.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/empty.hpp>
#include <std_msgs/msg/float32.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <optional>

class RTServer : public rclcpp::Node
{
public:
    RTServer()
        : rclcpp::Node("odor_gaden_rt")
    {}

    void Init()
    {
        std::filesystem::path scenarioPath = declare_parameter<std::string>("scenarioPath", "");
        std::string sceneID = declare_parameter<std::string>("sceneID", "scene1");
        stepOnTimer = declare_parameter<bool>("stepOnTimer", false);
        publishMarkers = declare_parameter<bool>("publishMarkers", true);

        GADEN_VERIFY(scenarioPath != "", "Parameter 'scenarioPath' is required "
                                         "(environment configuration directory with config.yaml)");

        // ---- 1. parse scenario metadata (config.yaml + simulations/ + scenes/) ----
        gaden::EnvironmentConfigMetadata metadata(scenarioPath);
        GADEN_CHECK_RESULT(metadata.ReadDirectory());

        // ---- 2. preprocess geometry + wind in-memory (STL -> occupancy grid) ----
        envConfig = gaden::Preprocessing::Preprocess(metadata);
        GADEN_VERIFY(envConfig != nullptr, "Preprocessing failed; check the scenario config");

        auto const& desc = envConfig->environment.description;
        GADEN_INFO("Environment ready: {}x{}x{} cells, cellSize {} m, bounds ({}, {}, {}) - ({}, {}, {})",
                   desc.dimensions.x, desc.dimensions.y, desc.dimensions.z, desc.cellSize,
                   desc.minCoord.x, desc.minCoord.y, desc.minCoord.z,
                   desc.maxCoord.x, desc.maxCoord.y, desc.maxCoord.z);

        // ---- 3. build the live scene (one RunningSimulation per source) ----
        gaden::RunningSceneMetadata sceneMeta = metadata.GetRunningScene(sceneID);
        for (auto& params : sceneMeta.params)
            params.saveResults = false; // real-time only; never write results to disk

        scene.emplace(sceneMeta, envConfig);
        deltaTime = sceneMeta.params.at(0).deltaTime;

        for (size_t i = 0; i < scene->GetSimulations().size(); i++)
        {
            auto const& source = scene->GetSimulations()[i]->simulationMetadata.source;
            GADEN_INFO("Source {}: gas '{}' at ({:.2f}, {:.2f}, {:.2f})",
                       i, gaden::to_string(source->gasType),
                       source->sourcePosition.x, source->sourcePosition.y, source->sourcePosition.z);
        }

        // ---- 4. ROS interfaces ----
        sourcePosesSub = create_subscription<geometry_msgs::msg::PoseArray>(
            "/gaden/source_poses", 5,
            std::bind(&RTServer::SourcePosesCallback, this, std::placeholders::_1));

        stepSub = create_subscription<std_msgs::msg::Empty>(
            "/gaden/step", 20,
            [this](std_msgs::msg::Empty::SharedPtr) { Step(); });

        simTimePub = create_publisher<std_msgs::msg::Float32>("/gaden/sim_time", 5);

        gasService = create_service<gaden_msgs::srv::GasPosition>(
            "/odor_value",
            std::bind(&RTServer::GasCallback, this, std::placeholders::_1, std::placeholders::_2));

        windService = create_service<gaden_msgs::srv::WindPosition>(
            "/wind_value",
            std::bind(&RTServer::WindCallback, this, std::placeholders::_1, std::placeholders::_2));

        if (publishMarkers)
        {
            filamentMarkerPub = create_publisher<visualization_msgs::msg::Marker>("/gaden/filament_visualization", 1);
            sourceMarkerPub = create_publisher<visualization_msgs::msg::MarkerArray>("/gaden/source_visualization", 1);
        }

        if (stepOnTimer)
        {
            stepTimer = create_wall_timer(std::chrono::duration<float>(deltaTime),
                                          [this]() { Step(); });
            GADEN_INFO("Stepping on wall timer every {:.3f} s", deltaTime);
        }
        else
            GADEN_INFO("Waiting for /gaden/step messages (lockstep mode), dt = {:.3f} s", deltaTime);

        GADEN_INFO_COLOR(fmt::terminal_color::blue, "odor_gaden_rt ready.");
    }

private:
    void SourcePosesCallback(geometry_msgs::msg::PoseArray::SharedPtr msg)
    {
        auto const& simulations = scene->GetSimulations();
        if (msg->poses.size() != simulations.size())
        {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                                 "/gaden/source_poses has %zu poses but the scene has %zu sources; "
                                 "ignoring message",
                                 msg->poses.size(), simulations.size());
            return;
        }

        for (size_t i = 0; i < simulations.size(); i++)
        {
            auto const& p = msg->poses[i].position;
            gaden::Vector3 newPos{(float)p.x, (float)p.y, (float)p.z};
            if (!envConfig->environment.IsInBounds(newPos))
            {
                RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                                     "Source %zu pose (%.2f, %.2f, %.2f) is out of bounds; keeping previous",
                                     i, p.x, p.y, p.z);
                continue;
            }
            simulations[i]->simulationMetadata.source->sourcePosition = newPos;
        }
    }

    void Step()
    {
        scene->AdvanceTimestep();
        simTime += deltaTime;

        std_msgs::msg::Float32 msg;
        msg.data = simTime;
        simTimePub->publish(msg);

        if (publishMarkers)
            PublishMarkers();
    }

    void GasCallback(gaden_msgs::srv::GasPosition::Request::SharedPtr req,
                     gaden_msgs::srv::GasPosition::Response::SharedPtr res)
    {
        std::vector<gaden::GasType> gasTypes = scene->GetGasTypes();

        // de-duplicate: multiple sources can emit the same gas type
        std::vector<gaden::GasType> uniqueTypes;
        for (auto type : gasTypes)
            if (std::find(uniqueTypes.begin(), uniqueTypes.end(), type) == uniqueTypes.end())
                uniqueTypes.push_back(type);

        for (auto type : uniqueTypes)
            res->gas_type.push_back(gaden::to_string(type));

        for (size_t i = 0; i < req->x.size(); i++)
        {
            std::map<gaden::GasType, float> concentrations =
                scene->SampleConcentrations(gaden::Vector3(req->x[i], req->y[i], req->z[i]));

            gaden_msgs::msg::GasInCell cell;
            for (auto type : uniqueTypes)
                cell.concentration.push_back(concentrations.at(type));
            res->positions.push_back(cell);
        }
    }

    void WindCallback(gaden_msgs::srv::WindPosition::Request::SharedPtr req,
                      gaden_msgs::srv::WindPosition::Response::SharedPtr res)
    {
        for (size_t i = 0; i < req->x.size(); i++)
        {
            gaden::Vector3 wind = scene->SampleWind(gaden::Vector3{(float)req->x[i], (float)req->y[i], (float)req->z[i]});
            res->u.push_back(wind.x);
            res->v.push_back(wind.y);
            res->w.push_back(wind.z);
        }
    }

    void PublishMarkers()
    {
        visualization_msgs::msg::Marker filamentMarker;
        filamentMarker.header.frame_id = "map";
        filamentMarker.header.stamp = now();
        filamentMarker.ns = "filaments";
        filamentMarker.action = visualization_msgs::msg::Marker::ADD;
        filamentMarker.type = visualization_msgs::msg::Marker::POINTS;
        filamentMarker.scale.x = 0.025;
        filamentMarker.scale.y = 0.025;
        filamentMarker.scale.z = 0.025;

        visualization_msgs::msg::MarkerArray sourceMarkers;

        auto const& simulations = scene->GetSimulations();
        for (size_t i = 0; i < simulations.size(); i++)
        {
            for (auto const& filament : simulations[i]->GetFilaments())
            {
                geometry_msgs::msg::Point p;
                p.x = filament.position.x;
                p.y = filament.position.y;
                p.z = filament.position.z;
                filamentMarker.points.push_back(p);
                filamentMarker.colors.push_back(GadenUtils::toRosColor(scene->GetColors().at(i)));
            }

            visualization_msgs::msg::Marker sourceMarker = GadenUtils::MarkerSourcePosition(this, *simulations[i]);
            sourceMarker.ns = "sources";
            sourceMarker.id = i;
            sourceMarkers.markers.push_back(sourceMarker);
        }

        filamentMarkerPub->publish(filamentMarker);
        sourceMarkerPub->publish(sourceMarkers);
    }

private:
    std::shared_ptr<gaden::EnvironmentConfiguration> envConfig;
    std::optional<gaden::Scene> scene;
    float deltaTime = 0.05f;
    float simTime = 0.0f;
    bool stepOnTimer = false;
    bool publishMarkers = true;

    rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr sourcePosesSub;
    rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr stepSub;
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr simTimePub;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr filamentMarkerPub;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr sourceMarkerPub;
    rclcpp::Service<gaden_msgs::srv::GasPosition>::SharedPtr gasService;
    rclcpp::Service<gaden_msgs::srv::WindPosition>::SharedPtr windService;
    rclcpp::TimerBase::SharedPtr stepTimer;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<RTServer>();
    node->Init();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
