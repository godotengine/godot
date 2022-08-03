#! /bin/bash

# Run this in //sw/gpgpu/thrust/thrust/system/cuda/detail/cub to add a THRUST_
# prefix to CUB's namespace macro.

sed -i -e 's/CUB_NS_P/THRUST_CUB_NS_P/g' `find . -type f`

