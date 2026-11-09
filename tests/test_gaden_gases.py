"""Integration checks for OdorSim's tracked gaden_core gas extension."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "gaden" / "gaden_common" / "third_party" / "gaden_core"
HEADER = CORE / "include" / "gaden" / "datatypes" / "GasTypes.hpp"
PATCH = ROOT / "patches" / "gaden_core_add_excel_gases.patch"
BUOYANCY_PATCH = ROOT / "patches" / "gaden_core_concentration_buoyancy.patch"
RUNNING_SIMULATION = CORE / "src" / "RunningSimulation.cpp"
BUOYANCY_HEADER = CORE / "include" / "gaden" / "internal" / "Buoyancy.hpp"


class GadenGasIntegrationTests(unittest.TestCase):
    def test_tracked_patch_defines_extended_gases(self):
        text = PATCH.read_text(encoding="utf-8")
        for gas_id, gas in enumerate(
            ("methanol", "acetic_acid", "ethyl_acetate", "toluene"),
            start=14,
        ):
            self.assertIn(f"{gas}", text)
            self.assertIn(f"= {gas_id}", text)

    @unittest.skipUnless(HEADER.is_file(), "local gaden_core clone is not installed")
    def test_tracked_patch_matches_local_gaden_core(self):
        self.assertTrue(PATCH.is_file())
        completed = subprocess.run(
            ["git", "-C", str(CORE), "apply", "--reverse", "--check", str(PATCH)],
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            completed.returncode,
            0,
            completed.stderr or "tracked gas patch does not match local gaden_core",
        )

    @unittest.skipUnless(RUNNING_SIMULATION.is_file(), "local gaden_core clone is not installed")
    def test_concentration_buoyancy_patch_matches_local_gaden_core(self):
        text = BUOYANCY_PATCH.read_text(encoding="utf-8")
        self.assertIn("BuoyancyDriftVelocity", text)
        self.assertIn("buoyancyTau", text)
        runtime_text = RUNNING_SIMULATION.read_text()
        self.assertEqual(runtime_text.count("ConcentrationAtCenter(filament)"), 1)
        self.assertIn("originalTerminalVelocity", runtime_text)
        self.assertNotIn("terminal_buoyancy_velocity", runtime_text)
        completed = subprocess.run(
            [
                "git", "-C", str(CORE), "apply", "--reverse", "--check",
                str(BUOYANCY_PATCH),
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            completed.returncode,
            0,
            completed.stderr
            or "tracked concentration-buoyancy patch does not match local gaden_core",
        )

    @unittest.skipUnless(shutil.which("g++"), "g++ is required for the C++ smoke test")
    @unittest.skipUnless(BUOYANCY_HEADER.is_file(), "local gaden_core clone is not installed")
    def test_concentration_buoyancy_math(self):
        source = r"""
#include <cmath>
#include <gaden/internal/Buoyancy.hpp>

int main()
{
    const float dilute = gaden::BuoyancyDriftVelocity(4.f, 3.f, 1.f);
    const float concentrated = gaden::BuoyancyDriftVelocity(40.f, 3.f, 1.f);
    const float tauTwo = gaden::BuoyancyDriftVelocity(40.f, 3.f, 2.f);
    if (!(dilute < 0.f && concentrated < dilute))
        return 1;
    if (std::abs(tauTwo - 2.f * concentrated) > 1e-8f)
        return 2;
    if (gaden::BuoyancyDriftVelocity(40.f, 3.f, 0.f) != 0.f)
        return 3;
    if (gaden::BuoyancyDriftVelocity(0.f, 3.f, 1.f) != 0.f)
        return 4;
    if (!(gaden::BuoyancyDriftVelocity(40.f, 0.55f, 1.f) > 0.f))
        return 5;
    return 0;
}
"""
        with tempfile.TemporaryDirectory(prefix="odorsim_buoyancy_cpp_") as temp:
            temp_path = Path(temp)
            cpp = temp_path / "buoyancy_smoke.cpp"
            binary = temp_path / "buoyancy_smoke"
            cpp.write_text(source, encoding="ascii")
            subprocess.run(
                ["g++", "-std=c++17", "-I", str(CORE / "include"), str(cpp), "-o", str(binary)],
                check=True,
            )
            subprocess.run([str(binary)], check=True)

    @unittest.skipUnless(shutil.which("g++"), "g++ is required for the C++ smoke test")
    @unittest.skipUnless(HEADER.is_file(), "local gaden_core clone is not installed")
    def test_cpp_enum_names_and_specific_gravity(self):
        source = r"""
#include <array>
#include <cmath>
#include <iostream>
#include <gaden/datatypes/GasTypes.hpp>

int main()
{
    constexpr std::array<gaden::GasType, 6> types = {
        gaden::GasType::ethanol,
        gaden::GasType::methanol,
        gaden::GasType::acetone,
        gaden::GasType::acetic_acid,
        gaden::GasType::ethyl_acetate,
        gaden::GasType::toluene,
    };
    constexpr std::array<float, 6> expected = {
        46.0684f / 28.96546f,
        32.0419f / 28.96546f,
        58.0791f / 28.96546f,
        60.0520f / 28.96546f,
        88.1051f / 28.96546f,
        92.1384f / 28.96546f,
    };
    static_assert(static_cast<int>(types[1]) == 14);
    static_assert(static_cast<int>(types[3]) == 15);
    static_assert(static_cast<int>(types[4]) == 16);
    static_assert(static_cast<int>(types[5]) == 17);
    static_assert(gaden::SpecificGravity.size() == 18);
    for (std::size_t i = 0; i < types.size(); ++i)
    {
        const auto type = types[i];
        const auto index = static_cast<std::size_t>(type);
        if (gaden::to_string(type).empty()
            || std::abs(gaden::SpecificGravity.at(index) - expected[i]) > 0.00001f)
            return 1;
        std::cout << gaden::to_string(type) << " "
                  << gaden::SpecificGravity.at(index) << "\n";
    }
    return 0;
}
"""
        with tempfile.TemporaryDirectory(prefix="odorsim_gas_cpp_") as temp:
            temp_path = Path(temp)
            cpp = temp_path / "gas_smoke.cpp"
            binary = temp_path / "gas_smoke"
            cpp.write_text(source, encoding="ascii")
            subprocess.run(
                ["g++", "-std=c++17", "-I", str(CORE / "include"), str(cpp), "-lfmt", "-o", str(binary)],
                check=True,
            )
            completed = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
        self.assertEqual(
            [line.split()[0] for line in completed.stdout.splitlines()],
            ["ethanol", "methanol", "acetone", "acetic_acid", "ethyl_acetate", "toluene"],
        )
