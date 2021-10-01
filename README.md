# PyOptiX

Python bindings for OptiX 7.

## Installation

### OptiX SDK
Install any [OptiX 7.x SDK](https://developer.nvidia.com/optix/downloads/7.3.0/linux64). 

### Clone Repository
```
git clone --recurse-submodules git@github.com:keithroe/PyOptiX.git
```

### Dependencies
Building the optix wrapper module, running examples or tests depend on the modules mentioned in
`requirements.txt` in the project's base directory:
```
cmake, cupy, fastrlock, numpy, Pillow, pybind11, pynvrtc, pytest
```
In most cases, it makes sense to setup a python environment.

#### `venv` Virtual Environment
Create and activate a new virtual environment:
```
python3 -m venv env
source env/bin/activate
```
Install all dependencies:
```
pip install -r requirements.txt
```

#### Conda Environment
Create an environment containing pre-requisites:
```
conda create -n pyoptix python numpy conda-forge::cupy pybind11 pillow cmake
```
Activate the environment:
```
conda activate pyoptix
```
The `pynvrtc` dependency, necessary for running the examples, needs to be installed via pip:
```
pip install pynvrtc
```

### Building the `optix` Module
Point `setuptools/CMake` to Optix by setting the following environment variable.

Linux:
```
export PYOPTIX_CMAKE_ARGS="-DOptiX_INSTALL_DIR=<optix install dir>"
```
Windows:
```
set PYOPTIX_CMAKE_ARGS=-DOptix_INSTALL_DIR=C:\ProgramData\NVIDIA Corporation\OptiX SDK 7.0.0
```

Build and install using `setuptools`:
```
cd optix
python setup.py install
```

When compiling against an Optix 7.0 SDK build also set a path variable pointing
to the system's stddef.h location. E.g.
```
export PYOPTIX_STDDEF_DIR="/usr/include/linux"
```

## Running the Examples

Examples can be run from the examples directory with:
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
