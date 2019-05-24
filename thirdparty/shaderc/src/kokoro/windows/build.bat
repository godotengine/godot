:: Copyright (C) 2017 Google Inc.
::
:: Licensed under the Apache License, Version 2.0 (the "License");
:: you may not use this file except in compliance with the License.
:: You may obtain a copy of the License at
::
::     http://www.apache.org/licenses/LICENSE-2.0
::
:: Unless required by applicable law or agreed to in writing, software
:: distributed under the License is distributed on an "AS IS" BASIS,
:: WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
:: See the License for the specific language governing permissions and
:: limitations under the License.
::
:: Windows Build Script.

@echo on

set BUILD_ROOT=%cd%
set SRC=%cd%\github\shaderc
set BUILD_TYPE=%1
set VS_VERSION=%2

:: Force usage of python 3.6.
set PATH=C:\python36;%PATH%

cd %SRC%
python utils\git-sync-deps

cmake --version

mkdir build
cd %SRC%\build

:: #########################################
:: set up msvc build env
:: #########################################
if %VS_VERSION% == 2017 (
  call "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvarsall.bat" amd64
  echo "Using VS 2017..."
) else if %VS_VERSION% == 2015 (
  call "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat" amd64
  echo "Using VS 2015..."
) else if %VS_VERSION% == 2013 (
  call "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\vcvarsall.bat" amd64
  echo "Using VS 2013..."
)

:: #########################################
:: Start building.
:: #########################################
echo "Starting build... %DATE% %TIME%"
if "%KOKORO_GITHUB_COMMIT%." == "." (
  set BUILD_SHA=%KOKORO_GITHUB_PULL_REQUEST_COMMIT%
) else (
  set BUILD_SHA=%KOKORO_GITHUB_COMMIT%
)

set CMAKE_FLAGS=-DSHADERC_ENABLE_SPVC=ON -DRE2_BUILD_TESTING=OFF -GNinja -DCMAKE_BUILD_TYPE=%BUILD_TYPE% -DCMAKE_C_COMPILER=cl.exe -DCMAKE_CXX_COMPILER=cl.exe

:: Skip building SPIRV-Tools tests for VS2013
if %VS_VERSION% == 2013 (
  set CMAKE_FLAGS=%CMAKE_FLAGS% -DSHADERC_SKIP_TESTS=ON -DSPIRV_SKIP_TESTS=ON
)

cmake %CMAKE_FLAGS% ..
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

echo "Build glslang... %DATE% %TIME%"
ninja glslangValidator
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

echo "Build everything... %DATE% %TIME%"
ninja
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

echo "Check Shaderc for copyright notices... %DATE% %TIME%"
ninja check-copyright
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
echo "Build Completed %DATE% %TIME%"

:: This lets us use !ERRORLEVEL! inside an IF ... () and get the actual error at that point.
setlocal ENABLEDELAYEDEXPANSION

:: ################################################
:: Run the tests (We no longer run tests on VS2013)
:: ################################################
echo "Running tests... %DATE% %TIME%"
if %VS_VERSION% NEQ 2013 (
  ctest -C %BUILD_TYPE% --output-on-failure -j4
  if !ERRORLEVEL! NEQ 0 exit /b !ERRORLEVEL!
)
echo "Tests passed %DATE% %TIME%"

:: Clean up some directories.
rm -rf %SRC%\build
rm -rf %SRC%\third_party

exit /b 0
