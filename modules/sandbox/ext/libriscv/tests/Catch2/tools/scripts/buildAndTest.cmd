rem Start at the root of the Catch project directory, for example:
rem cd Catch2

rem begin-snippet: catch2-build-and-test-win
rem 1. Regenerate the amalgamated distribution
python tools\scripts\generateAmalgamatedFiles.py

rem 2. Configure the full test build
cmake -B debug-build -S . -DCMAKE_BUILD_TYPE=Debug --preset all-tests

rem 3. Run the actual build
cmake --build debug-build

rem 4. Run the tests using CTest
ctest -j 4 --output-on-failure -C Debug --test-dir debug-build
rem end-snippet
