  <h1 align="center">DOF-GS: Adjustable Depth-of-Field 3D Gaussian Splatting for Refocusing, Defocus Rendering and Blur Removal.</h1>
  <h2 align="center">CVPR 2025</h2>
  <p align="center">
    <a href="https://yujiewang.info/">Yujie Wang</a>
    ·
    <a href="https://www.cs.unc.edu/~cpk/">Praneeth Chakravarthula</a>
    ·
    <a href="https://baoquanchen.info/">Baoquan Chen</a>
  </p>
  <h3 align="center"><a href="https://arxiv.org/pdf/2405.17351">Paper</a> | <a href="https://arxiv.org/abs/2405.17351"> Arxiv </a> | <a href="https://dof-gs.github.io/">Project Page</a> </h3>
  <div align="center"></div>


### Installation
Clone the repository and create an anaconda environment using
```shell
git clone --recursive https://github.com/rongduo/DOFGS_implementation.git
cd DOFGS_implementation

conda env create -f environment.yml
conda activate dofgs
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118  torchaudio==2.0.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.0.1+cu118.html  
pip install torchmetrics==1.3.0 torchtyping==0.1.4 torch-efficient-distloss==0.1.3
pip install submodules/diff-gaussian-rasterization-dof
pip install submodules/simple-knn
```

### Download dataset
Please download the real defocus dataset from [this url](https://drive.google.com/drive/folders/1_TkpcJnw504ZOWmgVTD7vWqPdzbk9Wx_) and put the downloaded data under the folder ```./data/DeblurNeRF_data/```.

Plase download the synthetic defocus dataset with COLMAP data from [this url](https://drive.google.com/drive/folders/1E3ZDelbCZn2ypUtho9C2k0PqCruse8dP?usp=sharing) and put the downloaded data under the folder ```./data/DeblurNeRF_data/```

The file structure of the `./data/` folder should be:

```
data/
├── DeblurNeRF_data/
│   ├── real_defocus_blur/
│   │   ├── defocuscake/
│   │   ├── defocuscaps/
│   │   └── ...
│   └── synthetic_defocus_blur/
│       ├── defocuscozy2room/
│       ├── defocusfactory/
│       └── ...
```

### Run Optimization

To run the optimizer, simply use

```shell
# run on real defocus scenes
sh scripts/train_real.sh
# fun on synthetic defocus scenes:
sh scripts/train_syn.sh
```

### Render and Evaluation
To render and evaluate the reconstructed scene for all scenes, run the following command:
```shell
# render and evaluate on real defocus scenes
sh scripts/val_real.sh
# render and evaluate on synthetic defocus scenes
sh scripts/val_syn.sh
```
After running, the images and numerical results are available in corresponding scene folders.

### Download pre-optimized scenes
You can also directly download the reconstructed scenes for different scenes in the real defocus dataset and synthetic dataset from [this url](https://drive.google.com/drive/folders/1E3ZDelbCZn2ypUtho9C2k0PqCruse8dP?usp=sharing) and put the decompressed folder in `./output/` folder. Then the `./output/model_dofgs/` folder should be like:

The file structure of the `./output/model_dofgs/` folder should be:

```
output/
└── model_dofgs/
    ├── real_results/           # Results on real defocus dataset at 600×400 resolution
    │   ├── real_cake/
    │   ├── real_caps/
    │   └── ...
    └── synthetic_results/      # Results on synthetic defocus dataset
        ├── synthetic_cozy2room/
        ├── synthetic_factory/
        └── ...
```


Note: The original checkpoints were unfortunately lost during server migration. We have reproduced the models, and the performance may slightly differ from the results reported in the paper.

All of the parameters can be easily specified by editing `config_files/config_xxx.yaml`.
<details>
<summary><strong> Key Configuration Parameters (Click to expand)</strong></summary>

- `model_folder`: Output folder name for the current experiment (e.g., `real_results`)
- `scene_name`: List of scene(s) to be trained or evaluated (e.g., `real_cake`)
- `model_version`: Specifies the directory under `src/` that contains the core module implementation
- `output_path`: The path where output results are saved. This does not need to be specified manually—it will be automatically derived from `model_folder`, `model_version` and `scene_name`
`source_path`: The path to the data corresponding to `scene_name`.  It does not need to be provided explicitly, as it will be automatically retrieved from `config_files\scene_info.yaml`
- `num_extra_points`: Number of extra points to be added during densification
- `iterations`: Total number of training iterations
- `test_iterations`, `save_iterations`: Iterations at which evaluation and checkpoint saving occur
- `fdist_lr`, `position_lr_init`, `position_lr_final`, `scaling_lr`: Learning rates for focal distance, gaussian attributes.
- `lambda_infocus`, `lambda_dssim`, `lambda_mask_loss`: Loss weights for different supervision signals
- `use_mask_loss`: Whether to apply in-focus mask supervision
- `use_depth_estimator`: Whether to use a depth estimator for supervision (off by default)
- `band_limit`: Whether to apply band-limiting within the gaussian rasterizer
- `kernel_size`: Kernel size used additionally within the gaussian rasterizer 
- `tonemap_type`: Type of tone mapping to apply (e.g., `none`, `gamma`, etc.)
</details>
<br>


### Bibtex
```
@article{wang2024dofgs,
  title={DOF-GS: Adjustable Depth-of-Field 3D Gaussian Splatting for Refocusing, Defocus Rendering and Blur Removal},
  author={Wang, Yujie and Chakravarthula, Praneeth and Chen, Baoquan},
  journal={The IEEE / CVF Computer Vision and Pattern Recognition Conference},
  year={2025}
}
```

### Acknowledgement
Our implementation is primarily based on the official codebase for [3D Gaussian Splatting (3DGS)](https://github.com/graphdeco-inria/gaussian-splatting). Please follow the license of 3DGS. And we have also drawn inspiration from parts of the [Mip-Splatting](https://github.com/autonomousvision/mip-splatting) implementation. We sincerely thank the authors of both projects for their excellent work and for sharing their codebases with the community.

