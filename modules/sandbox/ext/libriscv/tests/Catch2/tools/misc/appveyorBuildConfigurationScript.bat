SETLOCAL EnableDelayedExpansion

@REM  # Possibilities:
@REM  # Debug build + coverage
@REM  # Debug build + examples
@REM  # Debug build +   ---
@REM  # Release build
if "%CONFIGURATION%"=="Debug" (
  if "%coverage%"=="1" (
    @REM # coverage needs to build the special helper as well as the main
    cmake -Stools/misc -Bbuild-misc -A%PLATFORM% || exit /b !ERRORLEVEL!
    cmake --build build-misc || exit /b !ERRORLEVEL!
    cmake -S. -BBuild -A%PLATFORM% -DCATCH_TEST_USE_WMAIN=%wmain% -DMEMORYCHECK_COMMAND=build-misc\Debug\CoverageHelper.exe -DMEMORYCHECK_COMMAND_OPTIONS=--sep-- -DMEMORYCHECK_TYPE=Valgrind -DCATCH_BUILD_EXAMPLES=%examples% -DCATCH_BUILD_EXTRA_TESTS=%examples% -DCATCH_ENABLE_CONFIGURE_TESTS=%configure_tests% -DCATCH_DEVELOPMENT_BUILD=ON || exit /b !ERRORLEVEL!
  ) else (
    @REM # We know that coverage is 0
    cmake -S. -BBuild -A%PLATFORM% -DCATCH_TEST_USE_WMAIN=%wmain% -DCATCH_BUILD_EXAMPLES=%examples% -DCATCH_BUILD_EXTRA_TESTS=%examples% -DCATCH_BUILD_SURROGATES=%surrogates% -DCATCH_DEVELOPMENT_BUILD=ON -DCATCH_ENABLE_CONFIGURE_TESTS=%configure_tests% || exit /b !ERRORLEVEL!
  )
)
if "%CONFIGURATION%"=="Release" (
  cmake -S. -BBuild -A%PLATFORM% -DCATCH_TEST_USE_WMAIN=%wmain% -DCATCH_DEVELOPMENT_BUILD=ON || exit /b !ERRORLEVEL!
)
