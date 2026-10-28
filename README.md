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
| `src/model_dofgs/training.py`, `arguments/` | Loads the slit angles and adds the slit configuration options. Adds the directional all-in-focus loss for slit captures (see [Directional all-in-focus loss](#directional-all-in-focus-loss-slit)). |
| `src/utils/directional_loss.py` | Directional all-in-focus loss: smoothing along the long slit axis, derivative across the short axis, short-axis blur mask. |
| `src/scene/dataset_readers.py` | Optional `split.json` for an explicit train/test split. |
| `render.py` | `--slit_sweep` re-renders training views at arbitrary slit angles. |
| `tools/` | Simulation of slit and equal-area circular-aperture datasets (Gaussian PSF, or layered box / disk PSF with optional diffraction), AIF recovery evaluation, and comparison of recovered optics with ground truth. |
| `tests/` | GPU smoke test for the rasterizer; CPU checks of the directional loss. |

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
| `--noise_std` | additive Gaussian noise (std on [0, 1] images, at full exposure) on the defocused training captures; default 0 |
| `--exposure`, `--noise_scaling` | relative exposure time of every capture (default 1) and how the noise grows when it is shorter: `shot` → σ/√e (default), `read` → σ/e |
| `--full_circle` | spread the angles over 360° instead of 180° |
| `--psf` | PSF model of the captures: `gaussian` (default, **recommended**: the rasterizer's model), `gaussian_layered` or `box` (experimental, see [Realistic PSF simulation](#realistic-psf-simulation-layered)) |
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

Noisy runs have their own scene names, so they never overwrite the noise-free runs. Without noise the disk keeps one image per pose and the existing `disksim_real_cake` is reused. `collect_design_sweep.py` compares every slit run with the disk at the same noise level. The simulator prints a note that the noise level differs from the `--match_area` reference. This is expected, because the reference only provides the focus distance and the aperture area.

**Equal total exposure time.** A capture with relative exposure `e < 1` records `e·I` and is brought back to full brightness by a gain `1/e`. The image therefore keeps its brightness, and only the noise grows:

| `--noise_scaling` | effective std | regime |
|---|---|---|
| `shot` (default) | σ / √e | photon-noise limited (Gaussian approximation); n captures at 1/n equal one full capture |
| `read` | σ / e | read-noise limited; every extra capture pays the read noise again (worst case for the slit) |

`--equal_total_exposure` captures every slit design n times at exposure 1/n, and the disk once at exposure 1, so both have the same total exposure time per pose. Under either scaling, a single full exposure is the strongest disk baseline at that time budget.

```shell
python tools/run_design_sweep.py --config ./config_files/config_real_cake.yaml \
    --scene real_cake --iteration 15000 --gpu 1 --noise_std 0.01 --equal_total_exposure --noise_scaling read
# -> slitsim_real_cake_n4_a0.40_noise0.01_e0.25_read  and  disksim_real_cake_noise0.01
python tools/run_design_sweep.py --config ./config_files/config_real_cake.yaml \
    --scene real_cake --iteration 15000 --gpu 1 --noise_std 0.01 --equal_total_exposure --noise_scaling shot
# -> slitsim_real_cake_n4_a0.40_noise0.01_e0.25_shot (the disk run above is reused)
```

`--slit_exposure` and `--disk_exposure` set the exposures directly. `collect_design_sweep.py` pairs each slit run with the disk that has the same noise σ and the same total exposure per pose. It also lists the captures, the exposure and the effective noise of every run.

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

## Directional all-in-focus loss (slit)

DOF-GS supervises the all-in-focus (AiF) render directly only where a capture is in focus. Its in-focus mask is computed from the long-axis CoC `r_L`, so for a slit only pixels that are sharp in every direction are used. A slit capture at angle θ, however, is blurred strongly only along the long axis `u = (cos θ, sin θ)`. Along the short axis `v = (−sin θ, cos θ)`, the blur is `r_W = a·r_L`, so the capture still carries the scene detail across `v` over a depth range about `1/a` times larger.

The directional loss uses this detail:

```
L_dir = || M_W ⊙ ∂_v( G_u * I_AiF − G_u * I_θ ) ||_1
```

- `G_u` is a 1D Gaussian smoothing along `u`, wider than the long-axis blur. Smoothing both images with it removes their difference along `u` (`G_u * h_u ≈ G_u`). Without it, the loss would push the AiF render to be blurred along `u`. In the frequency domain, only frequencies with small `k_u` (and any `k_v`) are compared. These are the frequencies that the capture reproduces faithfully, whatever the exact long-axis blur.
- `∂_v` is the derivative across the slit (Sobel, normalised to a per-pixel difference).
- `M_W` is a soft mask from the rendered CoC map. It keeps pixels with a short-axis blur std `σ_W = a·√c·r_L` below `slit_dir_sigma_w_max`, and a long-axis blur std `σ_L = √c·r_L` at most `slit_dir_sigma_l_max`, so that `G_u` can absorb it. An image border of `3·std(G_u)` is excluded.
- The capture and the CoC map are treated as constants, so the gradients reach only the AiF render. The loss is averaged over the masked pixels, so its weight does not depend on the mask coverage. Over the n slit angles, it constrains the AiF render along n directions.

The loss is added to the in-focus AiF loss of every slit image after `infocus_start_iter`, with a linear warm-up. It is only used for `aperture_type: slit`; disk training is unchanged. **The default weight is 0.03**, used together with the direction-aware densification (`slit_densify_stat: max_angle`). Alone, the loss gave only a limited benefit; combined with `max_angle`, it gave the best all-in-focus image (see [Combined results](#combined-results-directional-loss--direction-aware-densification)).

| Option (`OptimizationParams`) | Default | Meaning |
|---|---|---|
| `slit_dir_loss` | 0.03 | weight; 0 disables the loss (original model) |
| `slit_dir_start_iter` | -1 | start iteration; -1 (or anything earlier) means `infocus_start_iter` |
| `slit_dir_warmup` | 2000 | iterations of the linear weight ramp |
| `slit_dir_sigma_w_max` | 1.0 | short-axis blur std threshold of the mask (px) |
| `slit_dir_sigma_l_max` | 4.0 | largest long-axis blur std absorbed by `G_u` (px) |
| `slit_dir_smooth_mult` | 2.5 | std of `G_u` = mult × `slit_dir_sigma_l_max` |

During training, the progress bar shows `Ldir`. At every save, the training prints the loss, the mask coverage, and the ratio of the weighted term to the AiF loss. TensorBoard logs the same values.

**Weight sweep on an existing dataset** (no new simulation, no disk retraining):

```shell
python tools/run_design_sweep.py --config ./config_files/config_real_cake.yaml \
    --scene real_cake --iteration 15000 --gpu 1 --no_disk --dir_loss 0 0.005 0.03
```

Each weight `w > 0` is trained into `./output/model_dofgs/slit_sim_dir<w>/<scene>/`. `--dir_loss 0` reuses or reproduces the original model in `slit_sim`. Without `--dir_loss`, the sweep uses the code default (0.03). `collect_design_sweep.py` reads the weight from each run's `config_runtime.yaml` and shows it in the `λ_dir` column. A `_dir` run is compared with the disk of the same folder without the suffix. Runs trained before this option existed count as `λ_dir = 0`. `python tools/list_runs.py --filter <text>` lists the image, depth and point-cloud metrics of all matching runs in one table.

With `train.py` directly, slit configs without `slit_dir_loss` use the default 0.03. Set `slit_dir_loss: 0` (and `slit_densify_stat: mean`) for the original model, and give each variant its own `model_folder`, so that results do not mix.

### Results and recommendation

The tests used the `real_cake` scene, simulated with n = 4 and a = 0.4 (10 × 4 px slit at the largest defocus), 40 000 iterations and one image per iteration. All numbers are from the test split.

Without noise:

| λ_dir | 0 (run 1) | 0 (run 2) | 0.001 | 0.005 | 0.01 | 0.03 | 0.05 | 0.1 | 0.3 |
|---|---|---|---|---|---|---|---|---|---|
| AiF PSNR | 31.98 | 32.17 | 32.23 | 32.26 | 31.95 | 32.32 | 31.95 | 31.93 | 31.89 |
| AiF LPIPS ↓ | 0.039 | 0.039 | 0.039 | 0.038 | 0.040 | 0.039 | 0.042 | 0.042 | 0.041 |

With noise σ = 0.01 (4 captures per pose):

| λ_dir | AiF PSNR | LPIPS ↓ | depth abs_rel ↓ | depth rmse ↓ | point-cloud chamfer ↓ | F@0.005 ↑ | Gaussians |
|---|---|---|---|---|---|---|---|
| 0 | 31.88 | 0.0399 | **0.0627** | **11.93** | **3.607** | 0.686 | 106 990 |
| 0.03 | **31.99** | **0.0395** | 0.0639 | 14.05 | 3.653 | 0.685 | 112 919 |

Findings:

- **Run-to-run variation is about 0.2 dB.** Two identical λ_dir = 0 runs differ by 0.19 dB (CUDA non-determinism, even with a fixed seed). For λ_dir between 0.001 and 0.03, the AiF PSNR lies within this range and does not change monotonically with the weight. Any gain is therefore not significant.
- **Weights of 0.05 and above are slightly worse.** Three runs agree: AiF PSNR is about 31.9 and LPIPS rises to 0.041–0.042.
- **Geometry does not improve.** With noise, depth rmse gets worse (11.93 → 14.05), and the point-cloud metrics stay about the same. The loss constrains only the appearance of the AiF render, and the number of Gaussians grows by 2–6 %. Some of the added Gaussians do not lie on the surface.
- **Why the effect is small.** In this simulation the forward model matches the data exactly and the angles are known, so the defocus loss already uses the directional information of the slit captures. `L_dir` adds little new information. It may matter more with model mismatch (real optics, box PSF), larger blur, or wrong `K` / depth estimates. This has not been tested.

**Recommendation (loss alone).** On its own, the loss gives at most a small gain, so a weight between 0.005 and 0.03 is safe. Together with the direction-aware densification, 0.03 gave the best result and is now the default; see [Combined results](#combined-results-directional-loss--direction-aware-densification).

CPU checks (axis convention, invariance to the long-axis blur, mask, gradients): `python tests/test_directional_loss.py`.

**Limitations.** `G_u * h_u ≈ G_u` is an approximation: the effective widths are `√(σ_u² + σ_L²)` and `σ_u`. Near occlusion boundaries the blur is not shift-invariant. The mask depends on the current depth and `K` estimates.

---

## Direction-aware densification (slit)

3DGS splits or clones a Gaussian when its mean view-space gradient norm, averaged over all views where it is visible, exceeds `densify_grad_threshold` (default 0.0002). With a rotating slit, detail is visible mainly in the views whose short axis `v` crosses it. Views whose long axis blurs the detail away give almost no gradient. Averaged over the n angles, such detail is diluted to roughly 1/n–1/2 of its strength, so it may never reach the threshold. A disk blurs every view the same way, so it does not have this problem.

`slit_densify_stat: max_angle` (in `OptimizationParams`, **default for slits**) accumulates the statistic per slit angle and densifies on the strongest angle:

```
ḡ_i = max_θ ( Σ_{views at θ} ‖∇μ_i‖ / #views at θ )
```

A Gaussian is refined if *any* slit orientation shows that it is under-resolved. Angles θ and θ + 180° form one group. A disk has a single group, so the option does not change it. `mean` is the original statistic. Only Python code changes (the statistics become one column per angle); the rasterizer is unchanged.

The statistic usually produces more Gaussians, and more Gaussians alone can raise PSNR. Compare it with a **threshold-matched control**, which uses the original mean with a lower threshold so that it ends with about as many Gaussians (`#pts` in `list_runs.py`):

```shell
# defaults: max over angles + directional loss 0.03 (existing noise-free dataset, slit only)
python tools/run_design_sweep.py --config ./config_files/config_real_cake.yaml \
    --scene real_cake --iteration 15000 --gpu 1 --no_disk
# control: same loss, original mean statistic, lower threshold (tune until #pts matches the run above)
python tools/run_design_sweep.py --config ./config_files/config_real_cake.yaml \
    --scene real_cake --iteration 15000 --gpu 1 --no_disk --densify_stat mean --densify_grad_threshold 0.00015
# original model: no directional loss, mean statistic (folder slit_sim)
python tools/run_design_sweep.py --config ./config_files/config_real_cake.yaml \
    --scene real_cake --iteration 15000 --gpu 1 --no_disk --dir_loss 0 --densify_stat mean
python tools/list_runs.py --filter slitsim_real_cake_n4_a0.40 | grep -E "slit_sim |densmax|thr|model_folder|---"
```

| Option | Default | Model folder suffix |
|---|---|---|
| `--densify_stat mean\|max_angle` (`slit_densify_stat`) | `max_angle` | `_densmax` (slit runs only) |
| `--densify_grad_threshold t` | config / 0.0002 | `_thr<t>` |

The training log prints the statistic and the threshold in use. `list_runs.py` shows them in the `densify` and `thr` columns, and `collect_design_sweep.py` compares `_densmax` / `_thr` slit runs with the disk of the folder without these suffixes.

### Combined results (directional loss + direction-aware densification)

The tests used `real_cake` simulated with n = 4 and a = 0.4, noise-free, the Gaussian PSF (`--psf gaussian`), 40 000 iterations and the default threshold 0.0002. All numbers are from the test split. Three identical runs of the original model give the run-to-run spread.

| λ_dir | densify | AiF PSNR | SSIM | LPIPS ↓ | depth abs_rel ↓ | chamfer ↓ | F@0.005 ↑ | Gaussians |
|---|---|---|---|---|---|---|---|---|
| 0 | mean (3 runs) | 31.98 / 32.17 / 32.05 | 0.929–0.930 | 0.038–0.039 | 0.055–0.059 | 3.35–6.63 | 0.68–0.70 | 117–123 k |
| 0.03 | mean | 32.32 | 0.932 | 0.039 | 0.056 | 3.74 | 0.689 | 120 k |
| 0 | max_angle | 32.22 | 0.931 | 0.039 | 0.060 | 3.67 | 0.704 | 152 k |
| 0.003 | max_angle | 32.38 | 0.932 | 0.038 | 0.063 | 4.09 | 0.696 | 148 k |
| **0.03** | **max_angle** | **32.56** | **0.935** | **0.037** | 0.060 | 4.58 | 0.690 | 148 k |

- **Images.** λ_dir = 0.03 with `max_angle` gives the best AiF image: +0.5 dB over the mean of the three baseline runs, above their 0.19 dB spread, with the best SSIM and LPIPS. Each option alone gives about +0.15–0.25 dB, within or near the spread.
- **Geometry.** Rendered depth stays about the same. The point-cloud metrics vary strongly between identical baseline runs (chamfer 3.35–6.63), and the combined run lies within that range. Its accuracy (rec→GT) and chamfer distance are worse than in two of the three baseline runs. `max_angle` adds about 25 % more Gaussians. The F-scores stay about the same.
- **Not yet controlled.** The gain has not been compared with a threshold-matched control (`--densify_stat mean` with a lower threshold and as many Gaussians), and comes from single runs. Repeat runs and the control are needed before claiming it.

**Defaults.** For slit apertures, `slit_dir_loss: 0.03` and `slit_densify_stat: max_angle` are the defaults. Use `slit_dir_loss: 0` and `slit_densify_stat: mean` (sweep: `--dir_loss 0 --densify_stat mean`) for the original model. Disk training is not affected by either option.

---

## Training with known optics (simulated datasets)

By default, the focus distance `F` and the aperture `K` are initialised from the scene depth and learned together with the scene. Under model mismatch (for example, layered `--psf gaussian_layered` data reconstructed with the rasterizer's splat blur), the learned optics can drift to compensate for the blur model. In one test they drifted by 7.6 % (F) and 6.2 % (K), placing the focal plane wrongly. This option tests whether correct optics recover a sharp all-in-focus image.

`optics_from_sim_gt` (in `ModelParams`) uses the ground truth stored in `<source_path>/sim_gt.json`. For a slit, this is `F`, `K_L` and the width ratio; for a disk, `F` and `K_D`:

| Value | Behaviour |
|---|---|
| `none` (default) | depth-based initial values, learned (original behaviour) |
| `init` | ground truth as initial values, still learned |
| `fixed` | ground truth, frozen; the slit angle offset is fixed at 0 |

`init` shows whether the drift comes from the initialisation or from the optimisation itself. `fixed` shows the result with correct optics. A real system can reach this regime by calibrating the focus distance and the aperture. The training log prints the values used.

```shell
python tools/run_design_sweep.py --config ./config_files/config_real_cake.yaml \
    --scene real_cake --iteration 15000 --gpu 1 --psf gaussian_layered --optics fixed
python tools/run_design_sweep.py --config ./config_files/config_real_cake.yaml \
    --scene real_cake --iteration 15000 --gpu 1 --psf gaussian_layered --optics init
```

**Result.** On layered (`gaussian_layered`) data, `init` and `fixed` did not improve the all-in-focus image. For the disk, `fixed` was 3.6 dB worse than `none`. The drift of the learned optics is a consequence of the blur-model mismatch, not its cause (see [Results and recommendation](#results-and-recommendation-1) under the layered simulation). The default `none` is recommended.

`--optics init|fixed` applies to slit and disk alike. It reuses the existing datasets and trains into `slit_sim_optinit` / `slit_sim_optfixed`. To use the option with `train.py` directly, set `optics_from_sim_gt` in the config and give the run its own `model_folder`. `tools/list_runs.py` shows the mode in the `optics` column; runs trained before this option existed count as `none`. Use `tools/compare_slit_params.py` to check how far `init` runs drift from the ground truth.

---

## Realistic PSF simulation (layered)

> **Recommended: keep the default `--psf gaussian`.** The layered PSFs are experimental. With them, the reconstruction does not recover a sharper all-in-focus image than its inputs (see [Results and recommendation](#results-and-recommendation-1) below).

By default, the simulator renders the defocused captures with the rasterizer's Gaussian PSF, which is the same model the reconstruction uses. `--psf box` instead renders them with the real aperture shape, while the reconstruction still uses the Gaussian. This tests the method under model mismatch. The renderer is in `tools/psf_render.py`:

1. The scene is split into `--n_layers` (default 48) layers, uniform in inverse depth, by Gaussian centre. The layer range is the 0.1–99.9 % range of the visible centres of each view.
2. Each layer is rendered on its own, without defocus, as premultiplied colour and alpha. The alpha comes from two renders, on black and on white backgrounds. Because the full 3D model is available, content hidden behind foreground layers is also rendered.
3. Each layer is convolved, by FFT with replicate padding, with the PSF of its defocus `d = |1/z − 1/F|`:
   - slit: a uniform box of `K_L·d × K_W·d` along θ;
   - disk: a uniform disk of diameter `K_D·d`;
   - `gaussian_layered`: the rasterizer's Gaussian, used to check the pipeline.

   Kernels are supersampled (`--supersample`, default 8) and normalised.
4. The layers are composited back to front, `I ← k*(αC) + (1 − k*α)·I`, which approximates foreground blur spreading over the background. Tone mapping is applied afterwards.

Sizes follow the existing definitions (full length `K·d`), so equal-area comparisons and `--match_area` are unchanged. The box has the same variance as the Gaussian it replaces (`L²/12 = r²/3`), so the two simulations differ only in the PSF shape. All-in-focus references and test views stay ideal.

**Diffraction (optional, `--diffraction`).** The in-focus Fraunhofer PSF of the physical aperture is convolved with every layer kernel, per colour channel: `sinc²` along both slit axes, or an Airy pattern for the disk. The physical aperture has the shape of the simulated one and the area `--aperture_area_mm2`. The default optics are **assumed values, not a measured system**:

| Option | Default | Meaning |
|---|---|---|
| `--aperture_area_mm2` | 40 | aperture area: 10 × 4 mm slit at a = 0.4, 20 × 2 mm at a = 0.1, 7.1 mm disk |
| `--focal_mm` | 50 | focal length |
| `--pixel_um` | 4.0 | effective pixel pitch of the simulated images (sensor pitch × downsampling) |
| `--wavelengths_nm` | 610 550 465 | R, G, B wavelengths |

With these defaults, the first zero across a 4 mm slit lies at `λf/w` ≈ 1.5–1.9 px (B to R), and the Airy first zero of the disk at about 1.0–1.3 px. Across a 2 mm slit (a = 0.1), the diffraction blur is twice as large. The simulator prints these values and stores them in `sim_gt.json`. Note that the effective pixel pitch scales the result directly: images downsampled by 4× have a 4× larger pitch and 4× less diffraction in pixels.

```shell
# check the layered pipeline against the default simulation (should be close)
python tools/simulate_slit_dataset.py --config ./config_files/config_real_cake.yaml \
    --scene real_cake --iteration 15000 --n_angles 4 --width_ratio 0.4 \
    --match_area ./data/slit_sim/slitsim_real_cake --psf gaussian_layered
# box PSF, slit and equal-area disk, then train and evaluate with the Gaussian model
python tools/run_design_sweep.py --config ./config_files/config_real_cake.yaml \
    --scene real_cake --iteration 15000 --gpu 1 --psf box
# box PSF with diffraction
python tools/run_design_sweep.py --config ./config_files/config_real_cake.yaml \
    --scene real_cake --iteration 15000 --gpu 1 --psf box --diffraction
```

Non-default PSFs add `_glayer`, `_box` and `_diff` to the scene names, for example `slitsim_real_cake_n4_a0.40_box` and `disksim_real_cake_box`. `collect_design_sweep.py` and `list_runs.py` show the PSF model, and each slit run is compared only with the disk of the same PSF model. In the evaluation, the row *re-rendered blur vs input (fit)* shows how well the Gaussian model fits the captures.

CPU checks of the kernels and the compositing: `python tests/test_psf_render.py`.

**Cost and limitations.** Each training pose needs `2 × n_layers` renders plus one FFT convolution per layer and angle. This adds minutes to the simulation, but nothing to training. Gaussians are assigned to layers by their centres, so large Gaussians that span several layers are blurred with a single kernel. Layered compositing is an approximation of occlusion-aware defocus, not ray tracing. Diffraction is added as an in-focus PSF, an approximation of the full defocused Fresnel PSF that holds when either geometric blur or diffraction dominates. Aberrations and vignetting are not modelled.

### Results and recommendation

The tests used `real_cake`, with n = 4 and a = 0.4 (10 × 4 px at the largest defocus) and the equal-area disk (7.14 px). There was no noise, 40 000 iterations and one image per iteration. All numbers are from the test split.

**The two simulations blur differently.** The inputs share the same optics and the same PSF variance. The all-in-focus references of both datasets are identical, because they come from the same renderer. The layered inputs, however, are much closer to the all-in-focus ground truth. Between the two simulations, the slit inputs differ by 31.96 dB on average, and the mean brightness stays the same (blurred / AiF = 1.0065 vs 0.998).

| Input vs AiF GT (PSNR) | `--psf gaussian` | `--psf gaussian_layered` |
|---|---|---|
| slit | 29.04 | 32.85 |
| disk | 30.24 | 34.18 |

**Reconstruction of the layered data with the rasterizer model.** The table shows the test AiF PSNR for each `--optics` mode:

| | input | `none` | `init` | `fixed` | fit (re-rendered vs input, train) |
|---|---|---|---|---|---|
| slit, layered | 32.85 | 29.23 | 29.43 | 29.40 | 39.07 (45.20 for `--psf gaussian`) |
| disk, layered | 34.18 | 33.62 | 32.23 | 30.02 | 42.17 (44.04 for `--psf gaussian`) |

With the default `--psf gaussian`, the slit reaches 31.98 dB (+2.8 dB over its inputs). With layered data, the recovered all-in-focus image is **worse than the blurred inputs**. It is blurred overall, with no blotches. In the `none` mode, the learned optics also drift (F −7.6 %, K +6.2 %). However, correct optics (`init`, `fixed`) do not help the slit and make the disk worse. The learned optics compensate for the error; they do not cause it.

**Cause: the rasterizer's defocus is not a convolution.** The rasterizer adds the CoC covariance to each 2D splat, but keeps the splat's peak opacity. Its opacity coefficient only compensates the Mip-Splatting 2D filter. This is the same as in the original DOF-GS code. A physical blur of a splat lowers the peak by `sqrt(det Σ / det(Σ + ΔΣ))`, which can be a factor of tens for small splats. Without it, the blurred splats stay nearly opaque. Each pixel then mostly takes the colour of the front-most enlarged splats, instead of averaging its neighbourhood. When the data are blurred physically, the true scene is not a solution of the fitting problem. The optimiser dims and spreads the fine details, which bakes the blur into the scene, so the all-in-focus render comes out blurred. Different views and slit angles require different compensations. The slit suffers more than the disk, also because the slit shares F and K across all views (`slit_share_F/K: true`), while the disk, like the original DOF-GS, learns them per view.

**Why DOF-GS still works on real photos.** On the real Deblur-NeRF defocus scenes, its reported AiF PSNR (24.12 dB) is comparable to other deblurring methods and 1.4–2.9 dB above methods that do not model blur. It is not a near-perfect deblurring. Per-view learnable focus and aperture can absorb part of the model error, and the in-focus localisation supervises the all-in-focus render directly where the captures are sharp.

**Recommendation.**

- **Use the default `--psf gaussian`** for aperture-design studies (n, a, noise, exposure). Results then compare slit and disk *within the DOF-GS blur model*. They test the optimisation and the aperture design, but not physical accuracy. State this when reporting them.
- Keep `gaussian_layered` / `box` for model-mismatch studies.
- For real slit data, consider per-view optics (`slit_share_F: false`, `slit_share_K: false`), which gives the slit the same compensation as the original DOF-GS.
- A physically consistent fix would scale each splat's opacity by `sqrt(det Σ / det(Σ + ΔΣ))` (energy-conserving CoC). This is not implemented.

---

## Notes and limitations

- **Gaussian PSF approximation.** The reconstruction approximates the rectangular PSF by an anisotropic Gaussian with matched variance. Sharp edges, sinc zeros of the OTF, diffraction from a narrow slit, and partial occlusion at depth boundaries are not modelled. The default and recommended simulation (`--psf gaussian`) uses the same approximation, so it tests the optimisation (identifiability, convergence) and the aperture design, not model mismatch. Use `--psf gaussian_layered` / `box` (optionally with `--diffraction`) to test model mismatch; see [Realistic PSF simulation](#realistic-psf-simulation-layered).
- **Splat blur is not a convolution.** Like the original DOF-GS, the rasterizer enlarges each splat by the CoC but keeps its peak opacity, so its defocus does not conserve energy. On physically blurred (layered) data, the recovered all-in-focus image is worse than the inputs. See [Results and recommendation](#results-and-recommendation-1) under the layered simulation.
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
  list_runs.py               image, depth and point-cloud metrics of all evaluated runs
  psf_render.py              layered rendering with box / disk / Gaussian PSFs and diffraction
tests/
  test_slit_rasterizer.py    rasterizer smoke test (regression, anisotropy, gradients)
  test_directional_loss.py   CPU checks of the directional all-in-focus loss
  test_psf_render.py         CPU checks of the PSF kernels and layered compositing
src/utils/slit_utils.py      slit angle loading, grouping of images by camera pose
src/utils/directional_loss.py  directional all-in-focus loss for slit captures
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
