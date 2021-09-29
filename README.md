# PyOptiX

Python bindings for OptiX 7.

## Installation

### OptiX SDK

Install any [OptiX 7.x SDK](https://developer.nvidia.com/optix/downloads/7.3.0/linux64). 


### Conda environment

Create an environment containing pre-requisites:

```
conda create -n pyoptix python numpy conda-forge::cupy pybind11 pillow cmake
```

Activate the environment:

```
conda activate pyoptix
```

### PyOptiX Installation

Build and install PyOptiX into the environment with:

```
export PYOPTIX_CMAKE_ARGS="-DOptiX_INSTALL_DIR=<optix install dir>"
pip3 install --global-option build --global-option --debug .
```
`<optix install dir>` should be the OptiX 7.3.0 install location - for example,
`/home/user/NVIDIA-OptiX-SDK-7.3.0-linux64-x86_64`.

When compiling against an Optix 7.0 SDK build also set a path variable pointing
to the system's stddef.h location. E.g.
```
export PYOPTIX_STDDEF_DIR="/usr/include/linux"
```

## Running the Examples

The example can be run from the examples directory with:

```
cd examples
python hello.py
```

If the example runs successfully, a green square will be rendered.

## Running the Test Suite

Test tests are using `pytest` and can be run from the test directory like this:
```
cd test
python -m pytest
```
