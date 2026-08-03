# Scripted OdorPlace collection (OpenVLA-OFT)

Headless scripted pick-and-place on the **`OdorPlace`** task for vision-only OpenVLA-OFT demos. Uses the same episode layout as teleop (`episode.npz`, `meta.json`, `frames/`).

Full pipeline (collect → RLDS → fine-tune → eval): **Odorbot** `docs/odorplace-oft.md`
(sibling `/path/to/Odorbot/docs/odorplace-oft.md`, or `Odorbot/docs/…` when this tree is
`Odorbot/third_party/OdorSim`).

**Cameras / grasp:** `agentview` is pulled back (pos `(0.575, 0, 1.45)`, FOV 52.5° — midpoint of stock and a farther framing) so cups stay in frame. Scripted grasp dwells **0.75 s** at cup height before closing and again after close (`--grasp-dwell`). On a failed grab, the same episode **re-reaches** (up to `--max-grasp-attempts`, default 8) instead of discarding; retries stay in the trajectory. Frames: `agentview` + wrist at 256×256.

## Quick collect

From an activated OdorSim / Odorbot sim env:

```bash
# Dry-run plan
python -m odor_sim.bridge.collect_odorplace_oft \
  --out-dir /path/to/Odorbot/datasets/odorplace_oft \
  --num-per-scene 150 \
  --dry-run

# Parallel collection (resume-safe via manifest.csv)
python -m odor_sim.bridge.collect_odorplace_oft \
  --out-dir /path/to/Odorbot/datasets/odorplace_oft \
  --num-per-scene 150 \
  --num-workers 6 \
  --quiet
```

Single-episode debug (no batch plan):

```bash
python -m odor_sim.bridge.scripted_place \
  --out-dir datasets/teleop \
  --cups porcelain_mug \
  --instruction "pick up the porcelain mug and put it on the plate"
```

Scenes A/B/C, instruction variants, RLDS build, and `finetune.py` commands are documented in Odorbot’s odorplace-oft guide.
