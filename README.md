# PyOptiX

Python bindings for OptiX 7.

## Installation

### OptiX SDK

Install the [OptiX 7.2.0
SDK](https://developer.nvidia.com/optix/downloads/7.2.0/linux64).


### Conda environment

Create an environment containing pre-requisites:

```
conda create -n pyoptix python numpy conda-forge::cupy pybind11 pillow cmake
```

Activate the environment:

```
conda activate pyoptix
```

### PyOptiX installation

Build and install PyOptiX into the environment with:

```
export PYOPTIX_CMAKE_ARGS="-DOptiX_INSTALL_DIR=<optix install dir>"
pip3 install --global-option build --global-option --debug .
```

`<optix install dir>` should be the OptiX 7.2.0 install location - for example,
`/home/gmarkall/numbadev/NVIDIA-OptiX-SDK-7.2.0-linux64-x86_64`.


## Running the example

The example can be run from the examples directory with:

```
cd examples
python hello.py
```

If the example runs successfully, a green square will be rendered.
