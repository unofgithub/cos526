# COS526 Final Project Report
### Nobline Yoo

---

This code release accompanies the following paper:

### Perceptual Losses for "Differentiable Point-Based Radiance Fields for Efficient View Synthesis" and Application to Real-World Scenes
Nobline Yoo

 [PDF](https://www.overleaf.com/read/tcdbjcthrrpx)
**Abstract:** 
Novel view synthesis is a challenging task, wherein the goal is to take sparse, unstructured photographs of a scene and render novel views of the same. Recent, successful methods build off of NeRF (Neural Radiance Fields), which uses a volumetric function to represent a scene. While these methods have yielded state-of-the-art metrics, the present issue is that take very long to train and render views. In one recent work, Zhang, Baek et al. seek to address this issue of computational efficiency by building network-free, end-to-end differentiable, point-based radiance fields that use splat rendering for image synthesis. While this method is up to 300x faster than NeRF in both training and inference, it has a few limitations, two of which we address in our work. Firstly, upon closer analysis of the qualitative results, we notice spot-like artifacts in images synthesized on the Blender dataset. Secondly, the authors note that the proposed method requires a foreground object mask, which indicates that in real-world datasets that contains no such masks, the method, as in, would not be appropriate.

In our work, we experiment with new formulations of the training objective (using MS-SSIM for denoising and Canny edge-based loss for accurate super-resolution) to reduce artifacts. Furthermore, we modify the point-cloud refinement process proposed by Zhang, Baek et al. Specifically, we show that by skipping the step where outlier points are removed, the model achieve a 4.68 point increase in PSNR.

Our main contributions are as follows:


1. We propose a new formulation of loss that employs a Canny edge-based term to reduce spot-artifacts, while maintaining high levels of detail reconstruction, setting a new baseline in PSNR from Zhang, Baek et al.
2. Through ablation studies, we identify the successes (denoising) and failure modes (over-smoothing) that come with using the MS-SSIM term.
3. We apply Zhang, Baek et al. to a new context: real-world scenes with no object segmentation masks (LLFF).


## Installation

We recommend using a [`conda`](https://docs.conda.io/en/latest/miniconda.html) environment for this codebase. The following commands will set up a new conda environment with the correct requirements:

```bash
# Create and activate new conda env
conda create -n pytorch3d python=3.9
conda activate pytorch3d

# Install pytorch3D
conda install pytorch=1.13.0 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
conda install pytorch3d -c pytorch3d

# Install other libraries
conda install numpy matplotlib tqdm imageio
pip install pytorch-msssim
pip install DISTS-pytorch
pip install kornia

# Note
If using DISTS (not required in most of our experiments), the SLURM node must have internet capabilities to download a pretrained VGG. To bypass this, download the pretrained VGG in advance and modify the local DISTS package to use the version you downloaded.

## Reproduce
You can train the model on NeRF synthetic dataset within 3 minutes. Here datadir is the dataset folder path. Dataname is the scene name. Basedir is the log folder path. Data_r is the ratio between the used point number and the initialized point number. Splatting_r is the radius for the splatting.

If you are using SLURM, then you can try running, once you adjust the job details (e.g. email to notify, GPU).

```bash
sbatch train_hotdog_1.slurm
```

`train_hotdog_1.slurm` all the way through `train_hotdog_45.slurm` are different experiments with different parameters or losses. `train_fern_remoutlier{True/False}.slurm` experiments with skipping the "remove outliers" step in the point cloud refinement process.

After around 15 - 30 minutes, you can see the following output:

```
Training time: ###.## s
Rendering quality: ##.## dB
Rendering speed: ###.## fps
Model size: #.## MB
```

## Citation

Our work builds off of:

```
@article{zhang2022differentiable,
  title={Differentiable Point-Based Radiance Fields for Efficient View Synthesis},
  author={Zhang, Qiang and Baek, Seung-Hwan and Rusinkiewicz, Szymon and Heide, Felix},
  journal={arXiv preprint arXiv:2205.14330},
  year={2022}
}
```
