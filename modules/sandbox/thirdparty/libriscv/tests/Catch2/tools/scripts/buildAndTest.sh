#!/usr/bin/env sh

# Start at the root of the Catch project directory, for example:
# cd Catch2

# begin-snippet: catch2-build-and-test
# 1. Regenerate the amalgamated distribution
./tools/scripts/generateAmalgamatedFiles.py

# 2. Configure the full test build
cmake -B debug-build -S . -DCMAKE_BUILD_TYPE=Debug --preset all-tests

# 3. Run the actual build
cmake --build debug-build

# 4. Run the tests using CTest
ctest -j 4 --output-on-failure -C Debug --test-dir debug-build
# end-snippet
