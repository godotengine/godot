# Shader Generation and Render Test Suite

The sub-folders in the test suite contain a set of input MaterialX documents. During execution of the test suite, each file is parsed to determine renderable elements.  For each element the appropriate shader generator is used to produced source code for ShaderGen tests. For Render validation tests the code is compiled, and/or rendered.

## Folder layout

- The main grouping of input files is by library: ([stdlib](stdlib) and [pbrlib](pbrlib)).
- Additional sub-folders group documents based on Element group or category. For example math tests are found in [stdlib/math](stdlib/math)), with:
    - `math.mtlx`
    - `math_operators.mtlx`
    - `transform.mtlx`
    - `trig.mtlx`, and
    - `vector_math.mtlx`

  documents containing the various Elements to test.
- It is possible to add additional tests by simply adding new MaterialX documents under the TestSuite sub-folder.
- The [Geometry](../../Geometry) and [Images](../../Images) folders provide stock input geometry and images for usage by the test suite.
- The [Utilities folder](Utilities) provides utilities used for rendering with `testrender` as well as the light configuration specification for hardware rendering.

At the top level, the [options file (_options.mtlx)](_options.mtlx) is a MaterialX document that defines the set of execution options for Render tests. The values may be edited locally as desired.

For details on how to build the unit test module to use this test suite see the [source documentation](../../../source/MaterialXTest/README.md).
