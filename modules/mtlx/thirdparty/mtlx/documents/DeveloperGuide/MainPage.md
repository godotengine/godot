# MaterialX Overview

MaterialX is an open standard for representing rich material and look-development content in computer graphics, enabling its platform-independent description and exchange across applications and renderers.  Launched at [Industrial Light & Magic](https://www.ilm.com/) in 2012, MaterialX has been a key technology in their feature films and real-time experiences since _Star Wars: The Force Awakens_ and _Millennium Falcon: Smugglers Run_.  The project was released as open source in 2017, with companies including Sony Pictures Imageworks, Pixar, Autodesk, Adobe, and SideFX contributing to its ongoing development.  In 2021, MaterialX became the seventh hosted project of the [Academy Software Foundation](https://www.aswf.io/).

### Quick Start for Developers

- Download the latest version of the [CMake](https://cmake.org/) build system.
- Point CMake to the root of the MaterialX library and generate C++ projects for your platform and compiler.
- Select the `MATERIALX_BUILD_PYTHON` option to build Python bindings.
- Select the `MATERIALX_BUILD_VIEWER` option to build the MaterialX viewer.

### Supported Platforms

The MaterialX codebase requires a compiler with support for C++14, and can be built with any of the following:

- Microsoft Visual Studio 2017 or newer
- GCC 6 or newer
- Clang 6 or newer

The Python bindings for MaterialX are based on [PyBind11](https://github.com/pybind/pybind11), and support Python versions 3.6 and greater.

### Building MaterialX

#### Building MaterialX C++

The MaterialX C++ libraries are automatically included when building MaterialX through CMake.

To enable OpenImageIO support in MaterialX builds, the following additional options may be used:

- `MATERIALX_BUILD_OIIO`: Requests that MaterialXRender be built with OpenImageIO in addition to stb_image, extending the set of supported image formats.
- `MATERIALX_OIIO_DIR`: Path to the root folder of an OpenImageIO installation.  If MATERIALX_BUILD_OIIO has been enabled, then this option may be used to select which installation is used.

See the [MaterialX Unit Tests](https://github.com/AcademySoftwareFoundation/MaterialX/tree/main/source/MaterialXTest) page for documentation on shader generation and render testing in GLSL, OSL, and MDL.

#### Building MaterialX Python

By default, the `MATERIALX_BUILD_PYTHON` option will use the active version of Python in the developer's path.  To select a specific version of Python, use one or more of the following advanced options:

- `MATERIALX_PYTHON_VERSION`: Python version to be used in building the MaterialX Python package (e.g. `3.9`)
- `MATERIALX_PYTHON_EXECUTABLE`: Python executable to be used in building the MaterialX Python package (e.g. `C:/Python39/python.exe`)

Additional options for the generation of MaterialX Python include the following:

- `MATERIALX_PYTHON_OCIO_DIR`: Path to a folder containing the default OCIO configuration to be packaged with MaterialX Python. The recommended OpenColorIO configuration for MaterialX is [ACES 1.2](https://github.com/colour-science/OpenColorIO-Configs/tree/feature/aces-1.2-config/aces_1.2).
- `MATERIALX_PYTHON_PYBIND11_DIR`: Path to a folder containing the PyBind11 source to be used in building MaterialX Python. Defaults to the included PyBind11 source.

#### Building The MaterialX Viewer

Select the `MATERIALX_BUILD_VIEWER` option to build the MaterialX Viewer.  Installation will copy the **MaterialXView** executable to a `bin/` directory within the selected install folder.

#### Building API Documentation

To generate HTML documentation for the MaterialX C++ API, make sure a version of [Doxygen](https://www.doxygen.org/) is on your path, and select the advanced option `MATERIALX_BUILD_DOCS` in CMake.  This option will add a target named `MaterialXDocs` to your project, which can be built as an independent step from your development environment.

### Installing MaterialX

Building the `install` target of your project will install the MaterialX C++ and Python libraries to the folder specified by the `CMAKE_INSTALL_PREFIX` setting, and will install MaterialX Python as a third-party library in your Python environment.  Installation of MaterialX Python as a third-party library can be disabled by setting `MATERIALX_INSTALL_PYTHON` to `OFF`.

### MaterialX Versioning

The MaterialX codebase uses a modified semantic versioning system where the *major* and *minor* versions match that of the corresponding MaterialX [specification](https://github.com/AcademySoftwareFoundation/MaterialX/blob/main/documents/Specification/MaterialX.Specification.md), and the *build* version represents engineering advances within that specification version.  MaterialX documents are similarly marked with the specification version they were authored in, and they are valid to load into any MaterialX codebase with an equal or higher specification version.

Upgrading of MaterialX documents from earlier versions is handled at import time by the `Document::upgradeVersion` method, which applies the syntax and node interface upgrades that have occurred in previous specification revisions.  This allows the syntax conventions of MaterialX and the names and interfaces of nodes to evolve over time, without invalidating documents from earlier versions.

#### MaterialX API Changes

The following rules describe the categories of changes to the [MaterialX API](https://materialx.org/docs/api/classes.html) that are allowed in version upgrades:

- In *build* version upgrades, only non-breaking changes to the MaterialX API are allowed.  For any API call that is modified in a build version upgrade, backwards compatibility should be maintained using deprecated C++ and Python wrappers for the original API call.
- In *minor* and *major* version upgrades, breaking changes to the MaterialX API are allowed, though their benefit should be carefully weighed against their cost.  Any breaking changes to API calls should be highlighted in the release notes for the new version.

#### MaterialX Data Library Changes

The following rules describe the categories of changes to the [MaterialX Data Libraries](https://github.com/AcademySoftwareFoundation/MaterialX/tree/main/libraries) that are allowed in version upgrades:

- In *build* version upgrades, only additive changes and fixes to the MaterialX data libraries are allowed.  Additive changes are allowed to introduce new nodes, node versions, and node inputs with backwards-compatible default values.  Data library fixes are allowed to update a node implementation to improve its alignment with the specification, without making any changes to its name or interface.
- In *minor* version upgrades, changes to the names and interfaces of MaterialX nodes are allowed, with the requirement that version upgrade logic be used to maintain the validity and visual interpretation of documents from earlier versions.
- In *major* version upgrades, changes to the syntax rules of MaterialX documents are allowed, with the requirement that version upgrade logic be used to maintain the validity and visual interpretation of documents from earlier versions.  These changes usually require synchronized updates to both the MaterialX API and data libraries.

### Additional Links

- The main [MaterialX website](http://www.materialx.org) provides background on the project's history, industry collaborations, and recent presentations.
- The [Python Scripts](https://github.com/materialx/MaterialX/tree/main/python/Scripts) folder contains standalone examples of MaterialX Python code.
- The [MaterialX Unit Tests](https://github.com/materialx/MaterialX/tree/main/source/MaterialXTest) folder contains examples of useful patterns for MaterialX C++.
- The [MaterialX Viewer](https://github.com/materialx/MaterialX/blob/main/documents/DeveloperGuide/Viewer.md) is a complete, cross-platform C++ application based upon [MaterialX Shader Generation](https://github.com/materialx/MaterialX/blob/main/documents/DeveloperGuide/ShaderGeneration.md)
