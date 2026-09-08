# Baseline Evaluation

Fire3D ships thin data, alignment, and metric adapters for the baselines used
in the paper. Upstream repositories and model weights are not vendored. Their
exact revisions are pinned in `registry.json` and can be fetched into the
Git-ignored `baselines/_upstream/` directory:

```bash
python -m baselines.fetch efm3d shaper
python -m baselines.doctor --require-checkouts
```

| Adapter | Evaluation scope |
|---|---|
| EFM3D | 3D scene perception on iTHOR and Imaginarium |
| EFM3D + ShapeR | Inferred perception followed by object reconstruction |
| ShapeR | Object reconstruction from ground-truth perception |
| SAM3D Objects | Object reconstruction from ground-truth perception |
| Boxer + SAM2 + TRELLIS.2 | Composed end-to-end pipeline |
| SceneScript | SceneScript OBB export to the common perception schema |
| SimRecon | Scene-perception metrics |
| HoloScene | Scene/object geometry and foreground-only appearance metrics |
| LiteReality | Object-reconstruction geometry metrics |

Each adapter's `--help` documents its required upstream checkout, checkpoint,
input, and output paths. The common world-coordinate contract is implemented
in `baselines/common/fire3d_baseline_align.py`. Evaluation data for iTHOR,
Imaginarium, and ShapeR is distributed from
`datasets/hongchi/Fire3D`; source-dataset terms continue to apply.
