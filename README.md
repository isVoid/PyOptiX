# PyOptiX

Python bindings for OptiX 7 - this branch also contains an experimental
implementation of an OptiX kernel written in Python, compiled with
[Numba](https://numba.pydata.org).


## Installation

### OptiX SDK
Install any [OptiX 7.x SDK](https://developer.nvidia.com/optix/downloads/7.3.0/linux64). 

### Clone Repository
via ssh:
```
git clone --recurse-submodules git@github.com:keithroe/PyOptiX.git
```
or htmls:
```
git clone --recurse-submodules https://github.com/keithroe/PyOptiX.git 
```


### Dependencies
OptiX python module build requirements:
* [cmake](https://cmake.org/)
* [pip](https://pypi.org/project/pip/)

To run the PyOptiX examples or tests, the python modules specified in 
`PyOptiX/requirements.txt` must be installed:
* pytest
* cupy
* numpy
* Pillow
* pynvrtc

### Virtual Environment
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
conda create -n pyoptix python numpy conda-forge::cupy pybind11 pillow cmake numba pytest
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

When compiling against an Optix 7.0 SDK an additional environment variable needs to be set
containing a path to the system's stddef.h location. E.g.
```
export PYOPTIX_STDDEF_DIR="/usr/include/linux"
```

## Running the Examples

Run the examples:
```bash
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


## Explanation

The Python implementation of the OptiX kernel and Numba extensions consists of
three parts:

- Generic OptiX extention types for Numba. These include new types introduced in
the OptiX SDK. They can be vector math types such as `float3`, `uint4` etc. Or it
could be OptiX intrinsic methods such as `GetSbtDataPointer`. These are included in
examples/numba_support.py. We intend to build more examples by reusing these extensions.
- The second part are the user code. These are the ray tracing kernels that user
of PyOptiX will write. They are in each of the example files, such as `hello.py`,
`triangle.py`.
- Code that should be generated from the user's code - these tell Numba how to
  support the data structures that the user declared, and how to create them
  from the `SbtDataPointer`, etc. I've handwritten these for this example, to
  understand what a code generator should generate, and because it would have
  taken too long and been too risky to write something to generate this off the
  bat. The correspondence between the user's code and the "hand-written
  generated" code is mechanical - there is a clear path to write a generator for
  these based on the example code.
