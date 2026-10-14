# Matcher evaluation (W3)

Ground-truth-free comparison of feature-matching / dense-correspondence backends
on the project's real cloud frames. The question is not "which matcher is best in
general" but "which one localises the horizontal disparity best on **these**
images", because the geometry is disparity-starved (18.3 px at 1500 m, so 1 px is
5.5 % of height at 1.5 km).

- **[PREREGISTRATION.md](PREREGISTRATION.md)** — metric, data, decision rule and
  failure condition, fixed before the first number (guardrail G2).
- **REPORT.md** — the ranked recommendation, the table, the reasons, the rejected
  candidates and the limits.
- `harness.py` — frame selection, the pipeline's exact rectification, and the
  criteria (plane-fit residual, epipolar split, radial profile, perturbation,
  gradient strata).
- `evaluate.py` — runs one backend over a fixed frame grid and writes
  `results/<backend><tag>_<date>.json`.
- `report.py` — aggregates the JSON into `report.json` and figures.
- `benchmark_throughput.py` — matcher-only ms/frame (the evaluate loop times the
  metric computation too, so its `ms` field is not a clean matcher timing).
- `verify_loftr_baseline.py` — proves the kornia LoFTR path still reproduces the
  frozen W1 correspondence cache **byte-for-byte**.

Run order:

```bash
source /home/iustimov/miniconda3/etc/profile.d/conda.sh && conda activate clouds
cd /home/iustimov/clouds
python experiments/matchers/verify_loftr_baseline.py     # 38/38 byte-identical
bash experiments/matchers/_run_all.sh                    # all candidates, both sessions
python experiments/matchers/benchmark_throughput.py
python experiments/matchers/report.py
```

## Install and version evidence

Everything below was executed in the `clouds` conda env (Python 3.12.15,
torch 2.14.1+cu130, kornia 0.8.3, RTX 3090 24 GB). Weights are cached under
`~/.cache/torch/hub/checkpoints/` unless a path is given.

| backend | install | weights | tested version |
|---|---|---|---|
| `kornia_loftr` | already installed | `loftr_outdoor.ckpt` (auto) | kornia 0.8.3 |
| `efficient_loftr` | `git clone https://github.com/zju3dv/EfficientLoFTR` + `EFFLOFTR_DIR` | `huggingface.co/kornia/Efficient_LOFTR` → `eloftr_outdoor.ckpt` | repo `07e9c140`, einops 0.8.2, yacs 0.1.8 |
| `roma` | `pip install romatch==0.1.2` | `roma_outdoor.pth` (auto, `Parskatt/storage` release) | romatch 0.1.2 |
| `dkm` | `pip install "git+https://github.com/Parskatt/DKM.git"` | `DKMv3_outdoor.pth` (auto) | dkm 0.3.0 |
| `mast3r` | `git clone --recursive https://github.com/naver/mast3r` + `MAST3R_DIR` | `naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric` | repo `f5209afc` |
| `da3` | `pip install --no-deps depth-anything-3==0.1.1` (+ `omegaconf addict evo moviepy==1.0.3`) | `depth-anything/DA3-BASE` | 0.1.1 |
| `disk_lightglue` | kornia 0.8.3 + cached LightGlue weights | `disk_lightglue_v0-1_arxiv-pth`, `depth-save.pth` | 0.8.3 |
| `cv2_dis_flow`, `cv2_sgbm` | OpenCV (already installed) | — | — |

Two environment facts are part of the result:

- **No CUDA toolchain.** `nvcc` is not on `PATH` and `CUDA_HOME=/usr/local/cuda-12.1`
  does not exist, so RoMa's optional `local_corr` CUDA extension cannot be built and
  RoMa ran on its pure-PyTorch correlation fallback (`use_custom_corr=False`). This
  is the one place a candidate could not be run in its reference configuration.
- **`pip install roma` is the wrong package.** PyPI `roma` is a rigid-motion
  library, not the image matcher; the matcher is `romatch`. `roma` was installed,
  identified, and uninstalled.

Everything is read-only with respect to the measurement data: no file under
`~/clouds_data/` is written, and the calibration and `camera.roll_correction_deg`
(0.0) are unchanged. The shared calibration roll is fitted as a nuisance parameter
inside the metric when needed, never applied to the config.
