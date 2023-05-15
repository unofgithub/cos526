# point-radiance

---

This code release accompanies the following paper:

### Differentiable Point-Based Radiance Fields for Efficient View Synthesis
Qiang Zhang, Seung-Hwan Baek, Szymon Rusinkiweicz, Felix Heide

*Siggraph Asia*, 2022

 [PDF](https://arxiv.org/pdf/2205.14330.pdf) | [arXiv](https://arxiv.org/abs/2205.14330) 
**Abstract:** 
We propose a differentiable rendering algorithm for efficient novel
view synthesis. By departing from volume-based representations
in favor of a learned point representation, we improve on existing
methods more than an order of magnitude in memory and run-
time, both in training and inference. The method begins with a
uniformly-sampled random point cloud and learns per-point posi-
tion and view-dependent appearance, using a differentiable splat-
based renderer to train the model to reproduce a set of input train-
ing images with the given pose. Our method is up to 300 × faster
than NeRF in both training and inference, with only a marginal
sacrifice in quality, while using less than 10 MB of memory for a
static scene. For dynamic scenes, our method trains two orders of
magnitude faster than STNeRF and renders at a near interactive
rate, while maintaining high image quality and temporal coherence
even without imposing any temporal-coherency regularizers.


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
