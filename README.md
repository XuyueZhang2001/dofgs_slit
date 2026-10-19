# DOF-GS with a Rotating Slit Aperture

This repository extends **[DOF-GS](https://github.com/rongduo/DOFGS_implementation)** (*Adjustable Depth-of-Field 3D Gaussian Splatting for Refocusing, Defocus Rendering and Blur Removal*, CVPR 2025) from a circular aperture to a **centered, rotatable slit (rectangular) aperture**. Each pose can be captured at several slit orientations. The defocus blur of every capture is then elongated along that capture's slit direction, and the model jointly recovers the all-in-focus (AIF) scene and the optical parameters from all orientations.

The original circular-aperture model is still available and is the default (`aperture_type: disk`).

---

## What changed relative to DOF-GS

### Defocus model

DOF-GS adds an isotropic circle of confusion to the 2D covariance of every Gaussian. Here that term is replaced by an anisotropic slit PSF with orientation θ:

```
d      = |1/z - 1/F|                      defocus of a Gaussian at view-space depth z
r_L    = 0.5 * K_L * d                    half-length along the slit long axis (px)
r_W    = 0.5 * K_W * d                    half-length along the slit short axis (px)
ΔΣ     = c * (r_L² · u uᵀ + r_W² · v vᵀ)  u = (cos θ, sin θ), v = (-sin θ, cos θ)
```

- `c = 1/3` matches the variance of a uniform box, i.e. a Gaussian approximation of the rectangular PSF.
- With `K_W = K_L` and `c = 1/(2 ln 4)`, the model reduces exactly to the original DOF-GS.
- θ is measured in image space: x points right, y points down, and θ runs from +x towards +y.
- The slit rotates about the optical axis, so the PSF is centrally symmetric. Points in front of and behind the focal plane therefore give the same PSF, and θ and θ + 180° are equivalent.

### Code changes

| Part | Change |
|---|---|
| `submodules/diff-gaussian-rasterization-dof/` | The forward pass adds the rotated slit covariance, including the off-diagonal term. The backward pass gives analytic gradients w.r.t. `K_L`, `K_W`, θ, `F` and depth. The new inputs (`Kw`, `theta`, `aperture_coeff`) are passed through the C++/Python bindings. In debug mode, kernel launch errors are reported. |
| `src/model_dofgs/gaussian_model.py` | Slit parameters: `F` and `K_L` (shared or per view), a short/long ratio `K_W / K_L` in (0, 1), the known per-view angles, and an optional global angle offset. All of these are saved in checkpoints. |
| `src/model_dofgs/renderer/__init__.py` | Passes `K_w` and `slit_theta` to the rasterizer, so any slit angle can be rendered. |
| `src/model_dofgs/training.py`, `arguments/` | Loads the slit angles and adds the slit configuration options. |
| `src/scene/dataset_readers.py` | Optional `split.json` for an explicit train/test split. |
| `render.py` | `--slit_sweep` re-renders training views at arbitrary slit angles. |
| `tools/` | Simulation of slit and equal-area circular-aperture datasets, AIF recovery evaluation, and comparison of recovered optics with ground truth. |
| `tests/` | GPU smoke test for the rasterizer. |

---

## Installation

Follow the original environment setup:

```shell
git clone https://github.com/XuyueZhang2001/dofgs_slit.git
cd dofgs_slit

conda env create -f environment.yml
conda activate dofgs
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
pip install torchmetrics==1.3.0 torchtyping==0.1.4 torch-efficient-distloss==0.1.3
```

Then build the CUDA extensions **for your GPU architecture**. If they are built for the wrong architecture, the kernels never launch and you get a misleading "CUDA out of memory" error that tries to allocate an absurd amount of memory. Common values are `8.0` for A100, `8.6` for RTX 30xx / A6000, `8.9` for RTX 40xx and `9.0` for H100.

```shell
export TORCH_CUDA_ARCH_LIST="8.0"          # set to your GPU; use "8.0;8.6" for several
pip install --no-build-isolation submodules/diff-gaussian-rasterization-dof
pip install --no-build-isolation submodules/simple-knn
```

When rebuilding after code changes, uninstall the old extension and delete its `build/` folder first:

```shell
pip uninstall -y diff_dof_gs_rasterizer
rm -rf submodules/diff-gaussian-rasterization-dof/build
```

### Smoke test (GPU, no dataset needed)

```shell
python tests/test_slit_rasterizer.py              # add DEBUG=1 in front to surface CUDA errors
```

It checks three things:
1. With `K_W = K_L`, the slit path reproduces the original isotropic result.
2. A defocused Gaussian is elongated along the slit direction.
3. The analytic gradients w.r.t. `K_L`, `K_W`, θ and `F` match finite differences.

---

## Data and pre-trained DOF-GS scenes

The datasets and checkpoints are those of the original DOF-GS:

- **Real defocus dataset** ([link](https://drive.google.com/drive/folders/1_TkpcJnw504ZOWmgVTD7vWqPdzbk9Wx_)): put it under `./data/DeblurNeRF_data/real_defocus_blur/`.
- **Synthetic defocus dataset with COLMAP** and **pre-optimized DOF-GS scenes** ([link](https://drive.google.com/drive/folders/1E3ZDelbCZn2ypUtho9C2k0PqCruse8dP?usp=sharing)): put the datasets under `./data/DeblurNeRF_data/` and the checkpoints under `./output/model_dofgs/`, e.g. `./output/model_dofgs/real_results/real_cake/`.

Each scene folder must contain `sparse/0/` (COLMAP) and `images/`. The data paths are listed in `config_files/scene_info.yaml`.

---

## Usage

### Original DOF-GS (circular aperture)

The commands are unchanged from the original repository:

```shell
python train.py --config ./config_files/config_real_cake.yaml
python render.py --config ./config_files/config_real_cake.yaml --scene real_cake --iteration 15000 --skip_train
```

To select a GPU, prefix the command with `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<id>`.

### Slit aperture on your own data

1. **Scene folder.** Use a standard COLMAP scene (`images/`, `sparse/0/`) plus `slit_angles.json`, which maps each image name (without extension) to the slit angle in degrees, using the image-space convention above:

   ```json
   { "IMG_0001": 0.0, "IMG_0002": 45.0, "IMG_0003": 90.0, "IMG_0004": 135.0 }
   ```

   Alternatively, set `slit_angle_regex` to read the angle from the file name.

2. **Register the scene** in `config_files/scene_info.yaml`:

   ```yaml
   my_slit_scene:
     source_path: ./data/my_slit_scene
     holdout: 8
   ```

3. **Config.** Copy an existing config, set `scene_name: [my_slit_scene]`, and add:

   ```yaml
   aperture_type: slit
   slit_angle_file: ''              # '' -> <source_path>/slit_angles.json
   slit_width_ratio: 0.1            # short/long ratio of the physical slit (initial value)
   slit_ratio_lr: 0.01              # set to 0 to keep the ratio fixed at slit_width_ratio
   slit_share_F: true               # one focus distance for all views
   slit_share_K: true               # one aperture scale for all views
   slit_learn_theta_offset: true    # learn a global offset between nominal angles and image axes
   slit_theta_lr: 0.001
   coc_max: 15                      # initial max blur (px) used to initialise K
   ```

4. **Train and render:**

   ```shell
   python train.py --config ./config_files/config_my_slit_scene.yaml
   python render.py --config ./config_files/config_my_slit_scene.yaml --scene my_slit_scene \
       --iteration 30000 --slit_sweep 0 45 90 135
   ```

   At the start of training, the log prints the per-view angle table. At every save it prints the learned width ratio and angle offset. `--slit_sweep` writes `*_slitXXX.XX.png` renders at the requested angles.

---

## Simulation and evaluation

When no slit captures are available, a scene reconstructed by the original DOF-GS can serve as ground truth. Slit captures with known optics are rendered from it, and the recovered AIF image and parameters are then compared with that ground truth.

### 1. Simulate a slit dataset

```shell
python tools/simulate_slit_dataset.py --config ./config_files/config_real_cake.yaml \
    --scene real_cake --iteration 15000 \
    --n_angles 4 --max_blur_px 20 --width_ratio 0.1
```

This writes `./data/slit_sim/slitsim_real_cake/`, which contains:
- `images/`: slit-blurred training views and AIF test views;
- `aif/`: AIF ground truth of the training poses;
- `defocus/*.npy`: per-pixel ground-truth blur length in px;
- `sparse/0/`, `slit_angles.json`, `split.json`;
- `sim_gt.json`: all ground-truth optics.

The script also writes `config_files/config_slitsim_real_cake.yaml` and registers the scene. Useful options:

| Option | Meaning |
|---|---|
| `--focus_quantile`, `--focus_depth` | focus distance, as a quantile of inverse depth or as an explicit value |
| `--theta_bias_deg` | rendered angle = recorded angle + bias, to test learning of the angle offset |
| `--noise_std` | additive Gaussian noise (std on [0, 1] images) on the defocused training captures; default 0 |
| `--full_circle` | spread the angles over 360° instead of 180° |
| `--defocus_only` | only (re)write `defocus/*.npy` for an existing simulated dataset |

`K` is parametrised as `K = clamp(Ks, 0.01, 200) * 100 + 100`, so `K_L < 101` cannot be represented. The script warns when the requested blur is too small.

### 2. Equal-area circular-aperture baseline

```shell
python tools/simulate_slit_dataset.py --config ./config_files/config_real_cake.yaml \
    --scene real_cake --iteration 15000 \
    --aperture disk --match_area ./data/slit_sim/slitsim_real_cake
```

This uses the same scene, poses and focus distance, with a disk of the same area as the slit: `K_D = 2·sqrt(K_L·K_W/π)`. It writes `./data/slit_sim/disksim_real_cake/` and a config with `aperture_type: disk`, i.e. the original DOF-GS model. With noise, use the same `--noise_std` as the slit dataset and `--disk_copies <n_angles>`, so both setups receive the same light per image and the same number of captures.

### Fixed-area aperture design (n angles, width ratio)

`tools/aperture_theory.py` predicts, under the Gaussian PSF model, how a rotating slit compares with a disk of the same area. It reports the worst-direction blur variance, the depth-of-field gain at equal light, and, with `--area_px2`, the multi-capture SNR gain per frequency. For n evenly spaced angles over 180°, the optimal width ratio is `a* = tan(90°/n)`.

```shell
python tools/aperture_theory.py --area_px2 40          # table, CSV and plots in ./output/aperture_theory/
```

To simulate a design at the same area and focus as an existing dataset, use `--match_area`. Only the width ratio changes. The default design is **n = 4, a = 0.4**, close to `a* = tan(22.5°) = 0.414`. Use `--width_ratio` to set the ratio, or `--optimal_ratio` to use `a*`. The scene name gets a `_n<n>_a<a>` suffix, and the training config fixes the ratio at its true value unless `--learn_ratio` is passed:

```shell
python tools/simulate_slit_dataset.py --config ./config_files/config_real_cake.yaml \
    --scene real_cake --iteration 15000 --n_angles 4 --width_ratio 0.4 \
    --match_area ./data/slit_sim/slitsim_real_cake
```

**Batch sweep.** `tools/run_design_sweep.py` runs simulate → train → evaluate for several designs and the equal-area disk baseline. Finished steps are skipped, so an interrupted sweep can be restarted. At the end it collects the results:

```shell
python tools/run_design_sweep.py --config ./config_files/config_real_cake.yaml \
    --scene real_cake --iteration 15000 --gpu 1 --designs 4:0.4 8:0.2      # add --dry_run to preview
python tools/collect_design_sweep.py --source_scene real_cake              # table + measured-vs-theory plot
```

| Option | Meaning |
|---|---|
| `--train_mode per_view` (default) | one image per iteration, as in DOF-GS; for n = 4 the budget equals the earlier runs |
| `--train_mode per_pose` | all images of one pose per iteration (`slit_group_by_pose: true`), with the AiF render shared across the angles |
| `--scale_iterations` | multiply the schedule of slit designs by n / 4, so every image gets the same number of updates (off by default) |
| `--iterations N` | stretch the whole schedule to N iterations |
| `--noise_std s` | additive Gaussian noise with std `s` on every defocused training capture (images in [0, 1]; default 0 = noise-free) |
| `--disk_copies` | captures per pose of the disk; default: the slit n when `--noise_std > 0`, otherwise 1 |

**Noisy comparison at an equal number of captures.** With `--noise_std > 0`, every training capture gets its own noise. The all-in-focus references and the test views stay clean. The disk is captured n times per pose by default, so slit (n = 4, a = 0.4) and disk both record 4 noisy images per pose:

```shell
python tools/run_design_sweep.py --config ./config_files/config_real_cake.yaml \
    --scene real_cake --iteration 15000 --gpu 1 --noise_std 0.01
# -> slitsim_real_cake_n4_a0.40_noise0.01  and  disksim_real_cake_c4_noise0.01
```

Noisy runs have their own scene names, so they never overwrite the noise-free runs. Without noise the disk keeps one image per pose and the existing `disksim_real_cake` is reused. `collect_design_sweep.py` compares every slit run with the disk at the same noise level. The simulator warns that the noise level differs from the `--match_area` reference. You can ignore this warning, because the reference only provides the focus distance and the aperture area.

Each variant trains into its own `model_folder` (`slit_sim`, `slit_sim_perpose`, `slit_sim_scaled4`, ...). `collect_design_sweep.py` writes `./output/design_sweep/<scene>/design_sweep.{md,csv}` and `measured_vs_theory.png`.

`render.py --one_angle_per_pose` renders only the first image of every training pose. This is off by default.

### 3. Train

```shell
python train.py --config ./config_files/config_slitsim_real_cake.yaml
python train.py --config ./config_files/config_disksim_real_cake.yaml
```

The outputs go to `./output/model_dofgs/<model_folder>/<scene>/`. To run a variant on the same data, such as a fixed width ratio, copy the config, change the parameters, and set a different `model_folder` so that runs do not overwrite each other.

### 4. Evaluate image and 3D recovery

```shell
python tools/eval_slit_recovery.py --config ./config_files/config_slitsim_real_cake_n4_a0.40.yaml \
    --scene slitsim_real_cake_n4_a0.40 --iteration 40000 --save_images
```

The same command works for the disk baseline. The ground truth is the simulated dataset plus, for the 3D metrics, the source DOF-GS scene it was simulated from. That scene is found from `sim_gt.json` (`source_scene`, `source_iteration`) and `config_files/config_<source_scene>.yaml`; use `--source_config` to point elsewhere.

**Image metrics.** PSNR and SSIM use the same implementation as DOF-GS. LPIPS is added if `pip install lpips` is available.

| Row | Compares | Purpose |
|---|---|---|
| train recovered AIF vs AIF GT | AIF render of the trained model at training poses vs ground truth | deblurring at captured poses |
| train blurred input vs AIF GT | the blurred captures themselves, averaged over angles | baseline: how blurry the data is |
| train mean of angles vs AIF GT | pixel-wise mean of the images of a pose | baseline: naive multi-angle fusion |
| train re-rendered blur vs input | blurred images re-rendered with the learned optics vs the inputs | data fit (not recovery) |
| **test recovered AIF vs AIF GT** | AIF render at held-out poses vs ground truth | **main result** (novel view + deblurring) |

The script also prints the AIF PSNR gain over the blurred inputs.

**3D metrics (A: depth).** The AIF depth of the trained model is compared with that of the source scene at every training pose and test pose. Depth is the alpha-normalised camera distance from the rasterizer.

| Metric | Meaning | Better |
|---|---|---|
| `abs_rel` | mean \|d − d\*\| / d\* | lower |
| `rmse` | depth RMSE (scene units) | lower |
| `rmse_inv` | RMSE of inverse depth 1/d (directly related to defocus) | lower |
| `delta1` | fraction of pixels with max(d/d\*, d\*/d) < 1.25 | higher |
| `coverage` | fraction of valid GT pixels that are also valid in the reconstruction | higher |

AbsRel is also reported per relative-defocus bin, `|1/z − 1/F| / d_max` in `[0,0.25)`, `[0.25,0.5)`, `[0.5,1)` and `≥1`, together with the pixel fraction of each bin. The bins depend only on scene depth, so slit and disk can be compared bin by bin.

**3D metrics (B: point cloud).** The centres of opaque (opacity > 0.5), visible Gaussians of the trained model are compared with those of the source scene. Both models share the same COLMAP frame, so no alignment is needed.

| Metric | Meaning | Better |
|---|---|---|
| `accuracy` | mean distance from reconstructed points to the nearest GT point | lower |
| `completeness` | mean distance from GT points to the nearest reconstructed point | lower |
| `chamfer` | mean of accuracy and completeness | lower |
| `precision` / `recall` / `F-score` @ τ | fraction of points within τ, for τ = 0.2 %, 0.5 % and 1 % of the GT bounding-box diagonal | higher |

| Option | Meaning |
|---|---|
| `--no_3d` | skip the depth and point-cloud evaluation |
| `--source_config` | config of the source DOF-GS scene (ground truth for 3D) |
| `--opacity_thresh` | opacity threshold for the point cloud (default 0.5) |
| `--pc_max_points` | random subsample per point cloud for the nearest-neighbour search (default 500k) |
| `--fscore_tau` | F-score thresholds as fractions of the GT bounding-box diagonal |

Results are written to `<model_path>/eval_slit_recovery/iteration_<N>/`:
- `summary.json`: the means of all metrics, under `summary.{train,test}.{recovered,input,mean,refit,depth,depth_by_defocus}` and `summary.geometry`, together with the design (`design`) and theory prediction (`theory`);
- `per_view.csv` and `per_view_depth.csv`: per-view image and depth metrics;
- `strips/`: GT | input | mean | recovered side by side (only with `--save_images`).

`run_design_sweep.py` skips evaluations whose `summary.json` already exists. To add the 3D metrics to an older evaluation, run the command above directly; no retraining is needed.

DOF-GS itself reports only PSNR and SSIM of AIF renders. The depth and point-cloud metrics are an addition of this repository.

### 5. Compare recovered optics (slit models)

```shell
python tools/compare_slit_params.py --sim ./data/slit_sim/slitsim_real_cake \
    --params ./output/model_dofgs/slit_sim/slitsim_real_cake/defocus_params/iteration_40000/defocus_params.pth
```

This prints the estimated and ground-truth `F`, `K_L`, `K_W`, width ratio and angle offset.

---

## Notes and limitations

- **Gaussian PSF approximation.** The rectangular PSF is approximated by an anisotropic Gaussian with matched variance. Sharp edges, sinc zeros of the OTF, diffraction from a narrow slit, and partial occlusion at depth boundaries are not modelled. The simulator uses the same approximation, so simulation results test the optimisation (identifiability, convergence), not model mismatch.
- **Width identifiability.** The slit covariance splits into `s_W·I + (s_L − s_W)·uuᵀ`. The rotating rank-1 term constrains `F`, `K_L` and θ well. The isotropic term `s_W·I` is the same in every orientation and can be absorbed by slightly larger Gaussians, so a learned width ratio may collapse towards 0, especially when the short-axis blur is only a few pixels. When the slit geometry is known, fix the ratio with `slit_ratio_lr: 0`.
- **Simulated vs. paper metrics.** Simulated datasets use DOF-GS renders as ground truth. Absolute PSNR/SSIM values are therefore not comparable to the numbers in the DOF-GS paper; compare configurations on the same simulated data instead.
- The random seed is fixed in `src/model_dofgs/training.py` (`3467`). Use several seeds before drawing conclusions from small metric differences.

---

## Repository layout (additions)

```
tools/
  simulate_slit_dataset.py   slit / equal-area disk dataset simulation from a trained DOF-GS scene
  eval_slit_recovery.py      image (AIF), depth and point-cloud metrics with baselines
  compare_slit_params.py     recovered vs ground-truth optics
  aperture_theory.py         equal-area slit vs disk theory (optimal ratio, DOF gain, SNR gain)
  run_design_sweep.py        simulate -> train -> evaluate for several (n, a) designs
  collect_design_sweep.py    results table and measured-vs-theory plot
tests/
  test_slit_rasterizer.py    rasterizer smoke test (regression, anisotropy, gradients)
src/utils/slit_utils.py      slit angle loading, grouping of images by camera pose
```

---

## Citation

This work builds on DOF-GS. If you use this code, please cite the original paper:

```bibtex
@article{wang2024dofgs,
  title={DOF-GS: Adjustable Depth-of-Field 3D Gaussian Splatting for Refocusing, Defocus Rendering and Blur Removal},
  author={Wang, Yujie and Chakravarthula, Praneeth and Chen, Baoquan},
  journal={The IEEE / CVF Computer Vision and Pattern Recognition Conference},
  year={2025}
}
```

## Acknowledgement and license

This code is based on the [DOF-GS implementation](https://github.com/rongduo/DOFGS_implementation), which in turn builds on [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) and [Mip-Splatting](https://github.com/autonomousvision/mip-splatting). The 3DGS license (non-commercial research and evaluation use) applies; see `LICENSE.md` in the rasterizer submodule.
