# MaterialX Unit Tests

MaterialX unit tests are run via the `MaterialXTest` executable.  This can be performed using the `ctest` command in CMake, as is done on each commit through [GitHub Actions](https://github.com/AcademySoftwareFoundation/MaterialX/blob/main/.github/workflows/main.yml), or by directly running the executable with a set of test tags.  By default, all tests that are supported by the set of built libraries are run.

## 1. Core Tests

The tests in the MaterialXCore folder validate the behavior of each element and value class in MaterialX, as well as core behaviors such as document traversal and graph operations:

- Document.cpp
- Element.cpp
- Material.cpp
- Traversal.cpp
- Types.cpp
- Value.cpp

## 2. I/O Tests

The tests in the MaterialXFormat folder validate the behavior of basic system operations and supported I/O formats for MaterialX content:

- Environment.cpp
- File.cpp
- XmlIo.cpp

## 3. Shader Generation and Render Test Suite

### 3.1 Test Inputs

Refer to the [test suite documentation](../../resources/Materials/TestSuite/README.md) for more information about the organization of the test suite data used for these tests.

### 3.2 Shader Generation Tests

- GenShader.cpp : Core shader generation tests which are run when the test tag `[genshader]` is specified.
- GenGlsl.cpp : GLSL shader generation tests which are run when the test tag `[genglsl]` is specified.
- GenOsl.cpp : OSL shader generation tests which are run when the test tag `[genosl]` is specified.
- GenMdl.cpp : MDL shader generation tests which are run when the test tag `[genmdl]` is specified.
- GenMsl.cpp : MSL shader generation tests which are run when the test tage `[genmsl]` is specified. 

Per-language tests will scan MaterialX files in the test suite for input materials.

#### Test Outputs
Depending on which tests are executed log files are produced at the location that MaterialXTest was executed.

- `gen<language>_<target>_generatetest.txt`: Contains a log of generation for a give language and target pair.
- `gen<language>_<target>_implementation_check.txt`: Contains a log of whether implementations exist for all nodedefs for a given language and target pair.

### 3.3 Render Tests

- Render.cpp : Core render tests which are run when the test tag `[rendercore]` is specified.
- RenderGlsl.cpp : GLSL render tests which are run when the test tag `[renderglsl]` is specified.
- RenderOsl.cpp : OSL render tests which are run when the test tag `[renderosl]` is specified.
- RenderMsl.mm: MSL render tests which are run when the test tage `[rendermsl]` is specified.

Per language tests will scan MaterialX files in the test suite for input materials.

#### Per-Language Render Setup

When rendering tests are enabled through the `MATERIALX_TEST_RENDER` option, the test suite will generate shader code for each test material and supported language.  Rendering will also be performed in languages for which support libraries have been provided:
- `GLSL`:
    - OpenGL version 4.0 and later are supported.
- `OSL`:
    - Set the following build options to enable OSL support:
        - `MATERIALX_OSL_BINARY_OSLC`: Path to the OSL compiler binary (e.g. `oslc.exe`).
        - `MATERIALX_OSL_BINARY_TESTRENDER`: Path to the OSL test render binary (e.g. `testrender.exe`).
        - `MATERIALX_OSL_INCLUDE_PATH`: Path to the OSL shader include folder, which contains headers such as `stdosl.h`.
    - OSL versions 1.9.10 and later are supported.
- `MDL` :
    - Set the following build options to enable MDL support:
        - `MATERIALX_MDLC_EXECUTABLE`: Full path to the MDL compiler binary (e.g. `mdlc.exe').
        - `MATERIALX_MDL_RENDER_EXECUTABLE`: Full path to the binary for render testing.
    - Optionally, `MATERIALX_MDL_RENDER_ARGUMENTS` can be set to provide command line arguments for non-interactive rendering.
    - MDL versions 1.6 and later are supported.
- `MSL`:
    - Metal Shading Language (MSL) 2.0 and later are supported.
    - Minimum tested operating system version is macOS Catalina 10.15.7

#### Test Outputs

- `gen<language>_<target>_render_doc_validation_log.txt`: Contains a log of whether input document validation check errors for a language and target pair.
- `gen<language>_<target>_render_profiling_log.txt`: Contains a log of execution times for a give language and target pair.
- `gen<language>_<target>_render_log.txt`: Contains a log of compilation and rendering checks for a language and target pair.  Note that if an error occurred a reference to a per-material error file will be given.

#### HTML Render Comparisons
- A `tests_to_html` Python script is provided in the [`python/MaterialXTest`](../../python/MaterialXTest) folder, which can be run to generate an HTML file comparing the rendered results in each shading language.
- Example render comparisons may be found in [commits to the MaterialX repository](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1164), and we encourage developers to post their own results when making changes that have the potential to impact generated shaders.
