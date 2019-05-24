:: Copyright (c) 2018 Google LLC.
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
set SRC=%cd%\github\SPIRV-Tools
set BUILD_TYPE=%1
set VS_VERSION=%2

:: Force usage of python 3.6
set PATH=C:\python36;%PATH%

cd %SRC%
git clone --depth=1 https://github.com/KhronosGroup/SPIRV-Headers external/spirv-headers
git clone --depth=1 https://github.com/google/googletest          external/googletest
git clone --depth=1 https://github.com/google/effcee              external/effcee
git clone --depth=1 https://github.com/google/re2                 external/re2

:: #########################################
:: set up msvc build env
:: #########################################
if %VS_VERSION% == 2017 (
  call "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
  echo "Using VS 2017..."
) else if %VS_VERSION% == 2015 (
  call "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat" x64
  echo "Using VS 2015..."
) else if %VS_VERSION% == 2013 (
  call "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\vcvarsall.bat" x64
  echo "Using VS 2013..."
)

cd %SRC%
mkdir build
cd build

:: #########################################
:: Start building.
:: #########################################
echo "Starting build... %DATE% %TIME%"
if "%KOKORO_GITHUB_COMMIT%." == "." (
  set BUILD_SHA=%KOKORO_GITHUB_PULL_REQUEST_COMMIT%
) else (
  set BUILD_SHA=%KOKORO_GITHUB_COMMIT%
)

set CMAKE_FLAGS=-GNinja -DSPIRV_BUILD_COMPRESSION=ON -DCMAKE_BUILD_TYPE=%BUILD_TYPE% -DCMAKE_INSTALL_PREFIX=install -DRE2_BUILD_TESTING=OFF -DCMAKE_C_COMPILER=cl.exe -DCMAKE_CXX_COMPILER=cl.exe

:: Skip building tests for VS2013
if %VS_VERSION% == 2013 (
  set CMAKE_FLAGS=%CMAKE_FLAGS% -DSPIRV_SKIP_TESTS=ON
)

cmake %CMAKE_FLAGS% ..

if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

echo "Build everything... %DATE% %TIME%"
ninja
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
echo "Build Completed %DATE% %TIME%"

:: This lets us use !ERRORLEVEL! inside an IF ... () and get the actual error at that point.
setlocal ENABLEDELAYEDEXPANSION

:: ################################################
:: Run the tests (We no longer run tests on VS2013)
:: ################################################
echo "Running Tests... %DATE% %TIME%"
if %VS_VERSION% NEQ 2013 (
  ctest -C %BUILD_TYPE% --output-on-failure --timeout 300
  if !ERRORLEVEL! NEQ 0 exit /b !ERRORLEVEL!
)
echo "Tests Completed %DATE% %TIME%"

:: Clean up some directories.
rm -rf %SRC%\build
rm -rf %SRC%\external

exit /b 0

