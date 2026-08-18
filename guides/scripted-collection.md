# Scripted pick-place collection (OpenVLA-OFT)

Two headless collectors share the teleop episode layout (`episode.npz`, `meta.json`, `frames/`) but are **not** the same task:

| Collector | Env | Sensing | Prefer module |
|-----------|-----|---------|---------------|
| Vision pick-place | `OdorPlace` | RGB only (no GADEN odor) | `collect_place_oft` |
| Odor pick-place | `ClassifyLiquidPlace` | Sniff → decide → place | `collect_odor_place_oft` |

Odor-conditioned demos: [scripted-odor-place-collection.md](scripted-odor-place-collection.md).

---

# Vision-only: `OdorPlace` → `collect_place_oft`

Headless scripted pick-and-place of catalog mugs onto a plate for vision-only OpenVLA-OFT demos.

Full pipeline (collect → RLDS → fine-tune → eval): **Odorbot** `docs/odorplace-oft.md`
(sibling `/path/to/Odorbot/docs/odorplace-oft.md`, or `Odorbot/docs/…` when this tree is
`Odorbot/third_party/OdorSim`).

**Cameras / grasp:** `agentview` is pulled back (pos `(0.575, 0, 1.45)`, FOV 52.5°) so cups stay in frame. Scripted grasp dwells **0.75 s** at cup height before closing and again after close (`--grasp-dwell`). On a failed grab, the same episode **re-reaches** (up to `--max-grasp-attempts`, default 8) instead of discarding; retries stay in the trajectory. Frames: `agentview` + wrist at 256×256.

## Quick collect

From an activated OdorSim / Odorbot sim env:

```bash
# Dry-run plan
python -m odor_sim.bridge.collect_place_oft \
  --out-dir /path/to/Odorbot/datasets/place_oft \
  --num-per-scene 150 \
  --dry-run

# Parallel collection (resume-safe via manifest.csv)
python -m odor_sim.bridge.collect_place_oft \
  --out-dir /path/to/Odorbot/datasets/place_oft \
  --num-per-scene 150 \
  --num-workers 6 \
  --quiet
```

Default plan: **450** successful demos (`--num-per-scene 150` × scenes A/B/C). Multi-cup scenes use **2** mugs by default (`--max-cups`); 3 mugs do not fit the spawn band reliably.

Single-episode debug (no batch plan):

```bash
python -m odor_sim.bridge.scripted_place \
  --out-dir datasets/teleop \
  --cups porcelain_mug \
  --instruction "pick up the porcelain mug and put it on the plate"
```

Scenes A/B/C, instruction variants, RLDS build (`build_rlds_odorplace`), and `finetune.py` commands are documented in Odorbot’s odorplace-oft guide.
