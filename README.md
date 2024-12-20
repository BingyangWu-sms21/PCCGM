# DiffusionPDE

## Dependencies
See the list of dependencies in `environment.yml`
Install to a new conda environement called `PCCGM` using `conda env create -f environment.yml`. Activate the environment using `conda activate PCCGM`

## Install GPU Support for PyTorch (Optional)
If your machine has an NVIDIA GPU and you want to use GPU acceleration with PyTorch, ensure that you install a GPU-compatible version of PyTorch. Depending on your CUDA version, follow these steps:

1. **Check your CUDA version**:
   Run the following command to verify your CUDA version:
   ```bash
   nvidia-smi
   ```
   You will see the CUDA version (e.g., `11.8`) displayed in the output.

2. **Install GPU-compatible PyTorch**:
   Replace the default CPU-only PyTorch in the Conda environment with a GPU-enabled version. Use the appropriate CUDA version from the [PyTorch Get Started page](https://pytorch.org/get-started/locally/).

   Example for CUDA 11.8:
   ```bash
   conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
   ```

## Install torch-fenics (Optional)
To generate data or run experiments of 2D Representative Volume Element (2D-RVE), additional torch-fenics package need to be installed from the [torch-fenics repository](https://github.com/barkm/torch-fenics).

## Dataset generation
I have written a highly paralelized solver for 2D darcy flow in which we can control the underlying dimension of the parameterization. Generate datasets using `data_generation/darcy_flow/generate_darcy.py`

## Training
To train a diffusion model, see the configuration file in `configs/darcy_flow.yaml`
I have designed the code to be very flexible. Any SDE formulations can be easily and simply adapted along with any denoising model architectures. This is very useful for rapidly prototyping and experimenting.
New datasets can be easily implemented, but they require a pytorch-lighting data module to be created
The training utilizes parallelized strategies in pytorch-lightning, utilizing many efficiency improvements such as automatic mixed precision (AMP), DDP for small models, and DeepSpeed for large models which must be sharded across gpus.

Train the model by creating a configuration file and running `python train.py --logdir /path/to/logs --config /path/to/config.yaml0 --name experiment_name`

To view training, use tensorboard via `tensorboard --logdir /path/to/logdir`. This shows all training losses as well as samples from the reverse sde during training
