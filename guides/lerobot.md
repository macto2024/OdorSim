# Convert data to LeRobot

Pipeline: optional MOX feature synthesis → LeRobot export. Skip synthesis when teleop already wrote live voltage into `episode.npz`.

```bash
source setup/activate.sh
```

## Optional synthesize

Writes `features.npz` next to each episode (useful for the other stream / other MOX models):

```bash
python -m odor_sim.recording.synthesize datasets/teleop
```

Replays each episode's ppm(t) through the MOX model twice:

- **Continuous** → `enose_voltage_continuous`
- **Sniffing** → `enose_voltage_sampling` (only while `enose_state == 1`)

By default **every** MOX model is synthesized. Each model writes `features_<MODEL>.npz` / `features_meta_<MODEL>.json`; primary `TGS2620` is also mirrored into canonical `features.npz` / `features_meta.json` for LeRobot export.

```bash
# One episode, single sensor only
python -m odor_sim.recording.synthesize datasets/teleop/episode_<timestamp> \
  --mox-model TGS2600 --no-overwrite
```

Useful options: `--mox-model`, `--load-resistance`, `--vcc`, `--rate`, `--no-overwrite`.

## LeRobot env + convert

Install LeRobot in a separate environment:

```bash
python3 -m venv .venv-lerobot
source .venv-lerobot/bin/activate
pip install -U pip wheel
pip install -e .
pip install "lerobot[dataset]>=0.6"
```

```bash
# All raw episodes
python -m odor_sim.recording.convert \
  --input datasets/teleop \
  --output odorsim/odorlift

# Selected episodes
python -m odor_sim.recording.convert \
  --input datasets/teleop/episode_20260708_173353 \
  --output odorsim/odorlift
```

Useful options: `--input`, `--output`, `--root` (default `datasets/lerobot`), `--fps`, `--no-videos`, `--robot-type`.

Default output:

```text
datasets/lerobot/odorsim/odorlift/
```
