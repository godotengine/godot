#! /bin/bash

# Run this in //sw/gpgpu/thrust/thrust/system/cuda/detail/cub to undo the
# renaming of CUB's namespace macro.

sed -i -e 's|THRUST_CUB_NS_P|CUB_NS_P|g' `find . -type f`

