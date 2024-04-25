# Change Log

## [1.38.9] - 2024-02-26

### Added

- Added an initial NPR (non-photorealistic rendering) data library to MaterialX, supporting the [View Direction](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1631), [Facing Ratio](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1671), and [Gooch Shading](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1674) nodes.
- Added new nodes to the standard data library, including [Reflect](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1690), [Refract](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1698), [Safe Power](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1689), [Create Matrix](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1553), and [Round](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1678).
- Added support for the generation of [pre-filtered environment maps](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1420) in MaterialX GLSL and MSL.
- Added support for [geometry drag & drop](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1663), [frame capture](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1636), [UI ranges](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1623) and [enumerated values](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1632) in the MaterialX Web Viewer.
- Added [floating popups](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1565) for hovered pins in the MaterialX Graph Editor.
- Added [UI ranges](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1646) to the 'useSpecularWorkflow' and 'normal' inputs of the UsdPreviewSurface shading model.
- Added [versioning rules](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1664) for the MaterialX API and data libraries to the developer guide.
- Added initial C++ [fuzz tests](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1605) and [coverage tests](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1548) to GitHub Actions CI.
- Added [GCC 13, Clang 15](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1602), and [Python 3.12](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1588) builds to GitHub Actions CI.

### Changed

- Enabled the [new OSL closures](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1702) by default in shader generation, with the MATERIALX_OSL_LEGACY_CLOSURES flag used to request legacy closures.
- Updated the MaterialX Web Viewer to [three.js r152](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1615).
- Switched to a more efficient representation of [HDR images](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1616) in the MaterialX Web Viewer.
- Improved the logic for [connecting pins](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1569) in the MaterialX Graph Editor.
- Improved the handling of [filename inputs](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1547) in OSL shader generation.
- Reduced the size of the MaterialX data libraries, improving the use of [graph definitions](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1653) and merging [duplicate implementations](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1642).
- Raised the minimum CMake version to [CMake 3.16](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1607).
- Updated the C++ unit test library to [Catch 2.13.10](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1566).

### Fixed

- Fixed the attenuation of [coated emission](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1534) in the Standard Surface shading model.
- Fixed the implementation of the [overlay node](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1539) in shader generation.
- Fixed an edge case for [type pointer comparisons](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1665) in shader generation.
- Fixed an edge case for [transform nodes](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1560) in GLSL and MSL shader generation.
- Fixed the implementation of [mx_hsvtorgb](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1584) in MDL shader generation.
- Fixed [orphaned links](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1667) when deleting nodes in the MaterialX Graph Editor.
- Fixed [scroll wheel interactions](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1578) across windows of the MaterialX Graph Editor.
- Fixed the generation of unused [imgui.ini files](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1570) in the MaterialX Graph Editor.
- Fixed a dependency on [module import order](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1595) in MaterialX Python.
- Fixed an [off-by-one index check](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1582) in Element::setChildIndex.
- Fixed a [missing null check](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1601) in Graph::propertyEditor.
- Fixed cases where [absolute paths](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1603) were stored in MaterialXConfig.cmake.

## [1.38.8] - 2023-09-08

### Added
- Added a broad set of new pattern nodes to MaterialX, including [Circle, Hexagon, Cloverleaf, Line, Grid, Crosshatch](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1411), [Checkerboard](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1328), [Random Color, Random Float](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1330), [Triangle Wave](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1334), [Integer Floor, Integer Ceiling](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1362), and [Distance](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1333).
- Added support for [MaterialX builds on iOS](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1435).
- Added support for [drag-and-drop import](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1482) of MaterialX files in the [Web Viewer](https://academysoftwarefoundation.github.io/MaterialX/).
- Added generation of [MaterialX Python wheels](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1317) in GitHub Actions, enabling the distribution of MaterialX Python packages through PyPI.
- Added support for the [lin_displayp3 and srgb_displayp3](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1368) colorspaces in shader generation.
- Added support for the [blackbody PBR node](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1367) in shader generation.
- Added support for [displacement](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1396) in MDL generation.
- Added [blend](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1350) and [up-axis](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1489) controls to the triplanar projection node.
- Added version details to [shared libraries](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1447) on Windows.
- Added a [MacOS 13](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1375) build to GitHub Actions.

### Changed
- Raised the minimum C++ version for MaterialX builds to [C++14](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1340).
- Upgraded the [PyBind11 library](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1343) to version 2.10.4, raising the minimum Python version to 3.6, and enabling support for Python versions 3.11 and beyond.
- Improved the performance and convergence of [GGX importance sampling](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1390) in GLSL generation, leveraging insights from the HPG 2023 paper by Jonathan Dupuy and Anis Benyoub.
- Improved [property panel display](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1346) in the MaterialX Graph Editor.
- Improved [node spacing](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1476) in the MaterialX Graph Editor.
- Improved the robustness of [MaterialX unit tests](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1370) with respect to the current working directory.
- Simplified the handling of [default colors](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1452) in GLSL generation, removing dynamic branches on texture size.
- Simplified the definitions of the [default color transforms](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1352), implementing them as language-independent MaterialX graphs.
- Simplified the interface of [ShaderGenerator::emitFunctionCall](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1344), marking its original interface as deprecated.
- Marked legacy interfaces for [findRenderableElements and findRenderableMaterialNodes](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1340) as deprecated, making their usage visible to clients as compiler warnings.
- Moved the MaterialX specification to [public Markdown files in GitHub](https://github.com/AcademySoftwareFoundation/MaterialX/tree/main/documents/Specification), enabling direct contributions from the community.

### Fixed
- Fixed brightness artifacts in the [triplanar projection node](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1350).
- Aligned default values for [conductor_bsdf](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1379) with the MaterialX specification.
- Fixed [volume mixing](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1395) in MDL generation.
- Fixed a bug to improve [shader generation determinism](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1376).
- Fixed a bug to improve the [consistency of auto layout](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1389) in the MaterialX Graph Editor.
- Fixed a bug to enable [multi-output connection edits](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1506) in the MaterialX Graph Editor.
- Fixed a bug in [dot node optimizations](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1522) for shader generation.

## [1.38.7] - 2023-04-21

### Added
- Added the [MaterialX Graph Editor](https://github.com/AcademySoftwareFoundation/MaterialX/blob/main/documents/DeveloperGuide/GraphEditor.md), an example application for visualizing, creating, and editing MaterialX graphs.
- Added support for the [Metal Shading Language](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1258) in MaterialX shader generation and rendering.
- Added support for the [generalized_schlick_edf node](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1262), enabling the expression of coated emissive surfaces.
- Added support for the [adobergb and lin_adobergb](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1118) color spaces in shader generation.
- Added uisoftmin and uisoftmax attributes to [mix nodes](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1168) and [IOR inputs](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1151).
- Added support for [authored bitangent vectors](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1156) in GLSL, selected by the hwImplicitBitangents generator option.
- Added a [tangent input](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1157) to the glTF PBR shading model.
- Added a [Clang Format](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1174) step to GitHub Actions builds.
- Added support for [Xcode 14](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1326).

### Changed
- Included the [standard data libraries](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1237) in MaterialX Python packages.
- Improved the [support library](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1184) and [node implementations](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1185) for OSL generation.
- Updated MDL shader generation to support [MDL 1.7](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1273).
- Improved the handling of [functional graphs](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1243) in MDL generation.
- Upgraded the [NanoGUI version](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1195) in the viewer to support Apple M1 builds.
- Upgraded the [Catch library](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1310) to version 2.13.9.

### Fixed
- Fixed logic for [tangent basis orthogonalization](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1177) in generated GLSL.
- Fixed logic for [metallic F90](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1187) and an [opacity edge case](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1186) in UsdPreviewSurface.
- Fixed parsing of [inline source code variables](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1191) in node implementations.

## [1.38.6] - 2022-11-04

### Added
- Added new [Unified Noise](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1087), [Color Correct](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1086), and [Bump](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1085) nodes, providing artistic interfaces over the standard procedural, adjustment, and geometric primitives.
- Added shader generation support for the [MaterialX closures in OSL](https://github.com/AcademySoftwareFoundation/OpenShadingLanguage/releases/tag/v1.12.6.2), selected by a new [CMake build option](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1039).
- Added a shader translation graph from [Autodesk Standard Surface to glTF PBR](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1064).
- Added support for the [transmission_extra_roughness](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1079) input in the Standard Surface shading model.
- Added support for the [iridescence](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1055) inputs in the glTF PBR shading model.
- Added support for the [subsurface_bsdf](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1053) node in MDL.
- Added an [operationorder](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1027) input to the 'place2d' node.
- Extended the [mix](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1077) node to support multi-channel interpolators.
- Extended the [flattenSubgraphs](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1101) method to handle a broader range of graph structures.

### Changed
- Improved the alignment of the [UsdPreviewSurface shading graph](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1082) with reference implementations.
- Improved the accuracy of the [thin-film BSDF](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1055) implementation in GLSL.
- Improved code generation for the [mix](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1104) node of 'surfaceshader' type.
- Removed [add and multiply](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1089) nodes for 'surfaceshader', 'volumeshader', and 'displacementshader' types.
- Refactored and extended the [Advanced Options](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1093) panel in MaterialXView.

### Fixed
- Fixed the implementation of the [screen](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1111) compositing node in GLSL, OSL, and MDL.
- Fixed parent/child precedence in the [getGeometryBindings](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1113) function.
- Fixed [OpenGL errors](https://github.com/AcademySoftwareFoundation/MaterialX/pull/1126) when unbinding geometry in MaterialXView on MacOS.

## [1.38.5] - 2022-07-09

### Added
- Added a [refraction approximation](https://github.com/AcademySoftwareFoundation/MaterialX/pull/918) for transmissive surfaces in GLSL, selected by the hwTransmissionRenderMethod generator option.
- Added support for generalized Schlick transmission in GLSL and OSL.
- Added support for code generation from material nodes.
- Added a specialization of GlslShaderGenerator for Vulkan GLSL generation.
- Added the [Chess Set](https://github.com/AcademySoftwareFoundation/MaterialX/pull/982) example from the [Karma: A Beautiful Game](https://www.sidefx.com/tutorials/karma-a-beautiful-game/) tutorial.  Contributed to the MaterialX project by SideFX, original artwork by Moeen and Mujtaba Sayed.
- Added static and dynamic analysis tests to GitHub Actions builds.
- Added support for GCC 12, Clang 13, and Clang 14.

### Changed
- Switched to [relative shader includes](https://github.com/AcademySoftwareFoundation/MaterialX/pull/926) within the MaterialX data libraries, enabling custom folder names in distributions.
- Improved and extended the sample [Web Viewer](https://academysoftwarefoundation.github.io/MaterialX/).

### Fixed
- Fixed math for normalizing normal and tangent vectors in GLSL.
- Fixed math for transforming a vector4 by a matrix in OSL.
- Fixed compatibility of OSL Worley noise with OSL 1.9.10.
- Fixed layering of thin-film effects in the Standard Surface shading model.
- Fixed input default values in the glTF PBR shading model.

## [1.38.4] - 2022-04-06

### Added
- Added [JavaScript bindings](https://github.com/AcademySoftwareFoundation/MaterialX/tree/main/javascript) for MaterialXCore, MaterialXFormat, and MaterialXGenGlsl.
- Added a sample [Web Viewer](https://academysoftwarefoundation.github.io/MaterialX/), built and deployed through GitHub Actions.
- Added a MaterialX graph for the [glTF PBR](libraries/bxdf/gltf_pbr.mtlx) shading model.
- Added new 'worleynoise2d' and 'worleynoise3d' nodes, with implementations in GLSL, OSL, and MDL.
- Added new 'surface_unlit' node, with implementations in GLSL, OSL, and MDL.
- Added support for the glTF geometry format in MaterialXRender and MaterialXView.

### Changed
- Moved the MaterialX project to the [Academy Software Foundation GitHub](https://github.com/AcademySoftwareFoundation/MaterialX).
- Removed hardcoded references to "libraries" in calls to GenContext::registerSourceCodeSearchPath.  (See Developer Notes below for additional details.)
- Improved the accuracy of mx_ggx_dir_albedo_analytic and mx_fresnel_conductor in GLSL.
- Updated the PyBind11 library to version 2.9.0.

### Fixed
- Aligned GLSL and MDL implementations of 'fractal3d' with OSL.
- Fixed MDL implementations of 'sheen_bsdf' and 'thin_film_bsdf'.
- Fixed an error in code generation from multi-output node graphs.

### Developer Notes
- This release removes hardcoded references to "libraries" in calls to GenContext::registerSourceCodeSearchPath within the MaterialX codebase.  Applications with their own custom code generators should make the same change, removing hardcoded references to "libraries" in calls to GenContext::registerSourceCodeSearchPath.  See pull request [877](https://github.com/AcademySoftwareFoundation/MaterialX/pull/877) for coding details.

## [1.38.3] - 2021-12-14

### Added
- Added an inheritance structure for versions of Autodesk Standard Surface.
- Added versioning and customization support to MaterialX namespaces in C++.
- Added preprocessor definitions for the API version to MaterialXCore.
- Added color transform methods to the Image class.
- Added an initial ClangFormat file for the MaterialX codebase.
- Added initial support for FreeBSD.
- Added support for Xcode 13.

### Changed
- Refactored BSDF handling in shader generation, allowing for more flexible and efficient vertical layering.
- Optimized GLSL implementations for GGX specular, moving common computations to tangent space.
- Refactored the TextureBaker API for clarity and flexibility.
- Merged the ViewHandler and viewer camera into a new Camera class in MaterialXRender.
- Updated CMake configuration generation logic, adding handling for shared library builds.
- Updated the PyBind11 library to version 2.7.1

### Fixed
- Fixed a performance regression in MaterialXView for multi-UDIM assets.
- Fixed a bug that caused shader inputs to be skipped in createUIPropertyGroups.
- Fixed the value of transmission roughness in UsdPreviewSurface.
- Fixed Vulkan compatibility for generated GLSL code.

## [1.38.2] - 2021-10-07

### Added
- Added an initial shader translation graph from Autodesk Standard Surface to UsdPreviewSurface.
- Added example script translateshader.py with validation in GitHub Actions.
- Added support for filename templates in texture baking.
- Added graph definitions for the MaterialX Lama node set.
- Added UI properties to the interface of UsdPreviewSurface.
- Added an initial ESSL shader generator.
- Added support for GCC 11.

### Changed
- Improved the accuracy of directional albedo computations for GGX specular and Imageworks sheen.
- Updated default color space names to follow ACES 1.2 conventions (e.g. g22_rec709), while maintaining compatibility with legacy names (e.g. gamma22).

### Fixed
- Fixed the default value of the roughness input of UsdPreviewSurface.
- Fixed the assignment of channel names in EXR files written through OpenImageIO.

## [1.38.1] - 2021-06-18

### Added
- Added support for shared library builds on Windows.
- Added support for 16-bit unsigned integer images in MaterialXRender.
- Added support for compound nodegraphs with user interfaces in shader and UI generation.
- Added headers for newly proposed MaterialX closures in OSL.
- Added a shader translation command to the viewer, assigned to the 'T' hotkey.

### Changed
- Improved the memory efficiency of texture baking operations.
- Improved the compatibility of generated MDL code with Omniverse.
- Refactored image resolution logic into new methods ImageHandler\:\:getReferencedImages and MaterialX\:\:getMaxDimensions.
- Moved the viewer hotkey for GLSL code generation from 'S' to 'G' for consistency with other languages.

### Fixed
- Fixed the Color3.asTuple and Color4.asTuple methods in Python

## [1.38.0] - 2021-03-02

Updated the MaterialX library to the v1.38 specification.  See the [v1.38 changelist](http://www.materialx.org/assets/MaterialX.v1.38.Changelist.pdf) for full details.

### Added
- Added support for the generalized 'layer' node in Physically Based Shading.
- Added user controls for texture baking and wedge rendering in the [MaterialX Viewer](https://github.com/materialx/MaterialX/blob/main/documents/DeveloperGuide/Viewer.md).
- Added support for Nvidia's Material Definition Language (MDL) in MaterialX code generation.
- Added support for inline source code in Implementation elements.
- Added support for TargetDef elements.
- Added viewer rendering to cloud-based tests in GitHub Actions.
- Added support for Xcode 12.

### Changed
- Updated the set of standard nodes to match the v1.38 specification, including significant improvements to the [Physically Based Shading](http://www.materialx.org/assets/MaterialX.v1.38.PBRSpec.pdf) nodes.
- Replaced specialized Material elements with material nodes, allowing more flexible material definitions and more consistent traversal.
- Unified the Input and Parameter element classes, simplifying the MaterialX API and client code.
- Updated the MaterialX viewer to use native classes for GLSL rendering and camera controls, opening the door to additional render frameworks in the future.
- Updated the prefiltered path for specular environment rendering in GLSL, providing a closer match with the Filtered Importance Sampling path.
- Updated the definition of Autodesk Standard Surface to version 1.0.1.
- Updated the definition of UsdPreviewSurface to version 2.3.
- Renamed the default branch from master to main.

### Removed
- Removed support for the 'complex_ior', 'backfacing', 'viewdirection' and 'fresnel' nodes in Physically Based Shading.
- Removed support for the Color2 type.

## [1.37.4] - 2020-12-18

### Added
- Added software rendering tests for MaterialXRenderGlsl to GitHub Actions.

### Changed
- Improved the robustness of context management in MaterialXRenderGlsl.

### Fixed
- Added a missing VAO binding to GlslRenderer\:\:drawScreenSpaceQuad.

## [1.37.3] - 2020-11-24

### Added
- Added Render Transparency and Render Double-Sided options to the Advanced Settings panel of the viewer.
- Added viewer support for partially-transparent mesh partitions.
- Added a subsurface scattering approximation to MaterialX GLSL.
- Added a CMake option for building shared libraries on Linux and MacOS.
- Added support for the latest OpenImageIO interface.

### Changed
- Improved the robustness of texture baking and shader translation.
- Unified the handling of missing images in generated GLSL.
- Moved CI builds from Travis and Appveyor to GitHub Actions.

### Fixed
- Fixed a bug in code generation for custom BSDF/EDF graphs.
- Fixed rendering of single-channel textures in MaterialXRenderGlsl.

## [1.37.2] - 2020-09-06

### Added
- Added support for texture baking from Python, including new example script [baketextures.py](python/Scripts/baketextures.py).
- Added support for texture baking of materials with multiple UDIMs.
- Added support for floating-point render targets in ShaderRenderer and its subclasses, allowing for HDR texture baking.
- Added support for displacement shaders in generated OSL.
- Added the ShaderTranslator class, laying the groundwork for support of shader translation graphs.
- Added Python bindings for the Image class.

### Fixed
- Fixed the alignment of environment backgrounds in the viewer.

### Removed
- Removed the CopyOptions class, making it the default behavior of Document\:\:importLibrary to skip conflicting elements.

## [1.37.1] - 2020-06-04

### Added
- Added command-line options for mesh, light, and camera transforms to the viewer.
- Added command-line options for screen dimensions and background color to the viewer.
- Added a Light Rotation slider to the Advanced Settings panel of the viewer.
- Added utility methods Backdrop\:\:setContainsElements and Backdrop\:\:getContainsElements.
- Added backwards compatibility for OpenImageIO 1.x.
- Added support for GCC 10.

### Changed
- Improved energy conservation and preservation computations in generated GLSL.
- Upgraded Smith masking-shadowing to height-correlated form in generated GLSL.
- Improved the robustness of tangent frame computations in MaterialXRender.
- Renamed Backdrop\:\:setContains and getContains to Backdrop\:\:setContainsString and getContainsString for consistency.

### Fixed
- Fixed the GLSL implementation of Burley diffuse for punctual lights.
- Fixed the upgrade path for compare nodes in v1.36 documents.

## [1.37.0] - 2020-03-20

Updated the MaterialX library to the v1.37 specification.  See the [v1.37 changelist](http://www.materialx.org/assets/MaterialX.v1.37REV2.Changelist.pdf) for full details.

### Added
- Added a Shadow Map option to the viewer, supported by shadowing functionality in GLSL code generation.
- Added support for the 'uisoftmin', 'uisoftmax', and 'uistep' attributes, updating Autodesk Standard Surface to leverage these features.
- Added support for LookGroup elements.
- Added support for Clang 9.

### Changed
- Updated the set of standard nodes to match the v1.37 specification.
- Unified the rules for NodeDef outputs, with all NodeDefs defining their output set through Output child elements rather than 'type' attributes.
- Replaced GeomAttr elements with GeomProp elements.
- Replaced backdrop nodes with Backdrop elements.
- Aligned Matrix33 and Matrix44 with the row-vector convention, for improved consistency with Imath, USD, and other libraries.
- Updated the stb_image library to version 2.23.

## [1.36.5] - 2020-01-11

### Added
- Added a Load Environment option to the viewer, allowing arbitrary latitude-longitude HDR environments to be loaded at run-time.
- Added an initial TextureBaker class, supporting baking of procedural content to textures.
- Added initial support for units, including the MaterialX\:\:Unit, MaterialX\:\:UnitDef, and MaterialX\:\:UnitTypeDef classes.
- Added support for unit conversion in shader code generation.
- Added support for Visual Studio 2019.

### Changed
- Updated Autodesk Standard Surface to the latest interface and graph.
- Updated the PyBind11 library to version 2.4.3.

## [1.36.4] - 2019-09-26

### Added
- Added a Save Material option to the viewer.
- Added property accessors to PropertyAssign and PropertySetAssign
- Added Python bindings for TypeDesc and array-type Values.
- Added Python functions getTypeString, getValueString, and createValueFromStrings.
- Added support for GCC 9 and Clang 8.

### Changed
- Updated the interface of readFromXmlFile and writeToXmlFile to support FilePath and FileSearchPath arguments.
- Extended Python bindings for FilePath and FileSearchPath.

### Removed
- Deprecated Python functions typeToName, valueToString, and stringToValue.
- Removed deprecated Python functions objectToString and stringToObject.

### Fixed
- Fixed the OSL implementation of roughness_dual.

## [1.36.3] - 2019-08-02

Merged shader code generation and physically-based shading nodes from Autodesk's ShaderX extensions.  Added a default MaterialX viewer based on GLSL shader generation.

### Added
- Added the MaterialXGenShader library, supporting shader code generation in GLSL and OSL.
- Added the MaterialXRender library, providing helper functionality for rendering MaterialX content.
- Added the MaterialXView library, providing a default MaterialX viewer.
- Added the physically-based shading node library (libraries/pbrlib).
- Added a root-level 'cmake' folder, including a standard FindMaterialX module.
- Added a root-level 'resources' folder, including example materials and meshes.
- Added documents for the 1.37 specification.

### Changed
- Moved the MaterialX data libraries from 'documents/Libraries' to 'libraries'.
- Updated MaterialX node definitions to the 1.37 specification.
- Updated the PyBind11 library to version 2.2.4.

### Removed
- Removed customizations of PyBind11 to support Python 2.6.  Only Python versions 2.7 and 3.x are now supported.

## [1.36.2] - 2019-03-05

### Added
- Added support for 'nodedef' attributes on MaterialX\:\:Node, integrating this usage into GraphElement\:\:addNodeInstance.
- Added the MaterialX\:\:GeomPropDef class for geometric input declarations.
- Added the Document\:\:getGeomAttrValue method.
- Added the ValueElement\:\:getResolvedValue method.
- Added support for the MATERIALX_SEARCH_PATH environment variable.
- Added support for GCC 8 and Clang 7.

### Changed
- Added callbacks Observer\:\:onCopyContent and Observer\:\:onClearContent, and removed callback Observer::onInitialize.
- Moved the standard document library to the 'documents/Libraries/stdlib' folder.

## [1.36.1] - 2018-12-18

### Added
- Added support for interface tokens, including the MaterialX\:\:BindToken class and '[TOKEN]' syntax in filenames.
- Added support for Clang 6.

### Changed
- Updated geometry token syntax from '%TOKEN' to '\<TOKEN\>'.
- Replaced readXIncludes boolean with a readXIncludeFunction callback in the XmlReadOptions structure.
- Combined individual options into an XmlWriteOptions argument for the XML write functions.
- Extended functionality of the vector and matrix classes.
- Updated the PyBind11 library to version 2.2.3.
- Updated the PugiXML library to version 1.9.

### Fixed
- Fixed graph implementations of range, extract, tiledimage, and ramp4 nodes.

## [1.36.0] - 2018-07-23

Updated the MaterialX library to the v1.36 specification.  See the [v1.36 changelist](http://www.materialx.org/assets/MaterialX.v1.36.Changelist.pdf) for full details.

### Added
- Added support for Element namespaces.
- Added support for NodeDef inheritance.
- Added support for root-level node elements.
- Added support for inheritance attributes on MaterialX\:\:Material and MaterialX\:\:Look.
- Added support for include and exclude attributes on MaterialX\:\:Collection.
- Added the MaterialX\:\:Token class for string substitutions.
- Added the MaterialX\:\:Variant, MaterialX\:\:VariantSet, and MaterialX\:\:VariantAssign classes.
- Added the MaterialX\:\:GeomPath class for geometry name comparisons.
- Added the Collection\:\:matchesGeomString method, for testing matches between collections and geometries.
- Added the Material\:\:getGeometryBindings method, for finding the bindings of a material to specific geometries.

### Removed
- Removed the MaterialX\:\:MaterialInherit and MaterialX\:\:LookInherit classes.
- Removed the MaterialX\:\:CollectionAdd and MaterialX\:\:CollectionRemove classes.
- Removed the MaterialX\:\:Override class and support for public names.
- Removed the 'channels' attribute from MaterialX\:\:InterfaceElement.
- Removed the Material::getReferencingMaterialAssigns method (deprecated in Python).

## [1.35.5] - 2018-05-07

### Added
- Added material inheritance support to graph traversal and the high-level Material API.
- Added Material methods getActiveShaderRefs and getActiveOverrides.
- Added PropertySet methods setPropertyValue and getPropertyValue.
- Added Element methods setInheritsFrom, getInheritsFrom, traverseInheritance, hasInheritanceCycle, and getDescendant.
- Added function templates MaterialX\:\:fromValueString and MaterialX\:\:toValueString.
- Added math functionality to the vector and matrix classes.
- Added support for Visual Studio 2017, GCC 7, and Clang 5.

### Changed
- Renamed Matrix3x3 to Matrix33 and Matrix4x4 to Matrix44.
- Renamed VectorN\:\:length to VectorN\:\:numElements.
- Updated the PyBind11 library to version 2.2.1.

## [1.35.4] - 2017-12-18

### Added
- Added high-level Material API, including getPrimaryShaderParameters, getPrimaryShaderInputs, getBoundGeomStrings, and getBoundGeomCollections.
- Added methods ValueElement\:\:getBoundValue and ValueElement\:\:getDefaultValue.
- Added support for multi-output nodes.
- Added support for TypeDef members.
- Added StringResolver class, for applying substring modifiers to data values.
- Added example interfaces for the Disney BRDF, Disney BSDF, and alSurface shaders.

### Changed
- Renamed method Material\:\:getReferencedShaderDefs to Material\:\:getShaderNodeDefs.
- Renamed method ShaderRef\:\:getReferencedShaderDef to ShaderRef\:\:getNodeDef.
- Renamed method Node\:\:getReferencedNodeDef to Node\:\:getNodeDef.
- Added a 'string' suffix to all accessors for 'node', 'nodedef', and 'collection' strings.
- Combined individual booleans into an XmlReadOptions argument for the XML read functions.

### Removed
- Removed method Document\:\:applyStringSubstitutions (deprecated in Python).
- Removed method InterfaceElement\:\:getParameterValueString (deprecated in Python).

## [1.35.3] - 2017-10-11

### Added
- Added support for Python 3.
- Added support for standard TypeDef attributes.
- Added support for values of type 'stringarray'.
- Added method Element\:\:setName.
- Extended Python bindings for Document, NodeGraph, MaterialAssign, and Collection.

### Changed
- Modified NodeGraph\:\:topologicalSort to return elements in a more intuitive top-down order, with upstream elements preceding downstream elements.
- Removed special cases for string return values in MaterialX Python, with all strings now returned as 'unicode' in Python 2 and 'str' in Python 3.
- Updated OSL reference implementations.

### Fixed
- Fixed handling of empty names in Element\:\:addChildOfCategory.
- Fixed an edge case in Document\:\:upgradeVersion.

## [1.35.2] - 2017-07-03

### Added
- Added OSL source files for the standard nodes.
- Added example document 'PostShaderComposite.mtlx'.
- Added method MaterialX\:\:prependXInclude.

### Changed
- Argument 'writeXIncludes' defaults to true in MaterialX\:\:writeToXmlStream and MaterialX\:\:writeToXmlString.

### Fixed
- Fixed handling of BindInput elements with missing connections.

## [1.35.1] - 2017-06-23

### Added
- Added a 'viewercollection' attribute to MaterialX\:\:Visibility.
- Added Python support for visibility and source URI methods.

### Changed
- Changed naming convention from 'ColorSpace' to 'ColorManagement' in Document methods.
- Split library document 'mx_stdlib.mtlx' into 'mx_stdlib_defs.mtlx' and 'mx_stdlib_osl_impl.mtlx'.

## [1.35.0] - 2017-06-20

Updated the MaterialX library to the v1.35 specification.  See the [v1.35 changelist](http://www.materialx.org/assets/MaterialX.v1.35.Changelist.pdf) for full details.

### Added
- Added the MaterialX\:\:Visibility class.
- Added 'file', 'function', and 'language' attributes to MaterialX\:\:Implementation.
- Added 'node' and 'nodedef' attributes to MaterialX\:\:ShaderRef.  In v1.35, these attributes define which NodeDef is referenced by a ShaderRef.
- Added a 'material' attribute to MaterialX\:\:MaterialAssign.  In v1.35, this attribute defines which Material is referenced by a MaterialAssign.

### Changed
- Removed the MaterialX\:\:LightAssign and MaterialX\:\:Light classes.  In v1.35, this functionality is now handled by the MaterialX\:\:Visibility class.
- Removed the 'default' attribute from MaterialX\:\:ValueElement.  In v1.35, this functionality is now handled by the 'value' attribute.
- Replaced the 'matrix' type with 'matrix33' and 'matrix44', and replaced the MaterialX\:\:Matrix16 class with MaterialX\:\:Matrix3x3 and MaterialX\:\:Matrix4x4.
- Renamed Material\:\:getMaterialAssigns to Material\:\:getReferencingMaterialAssigns.
- Changed the argument type for MaterialAssign\:\:setExclusive and MaterialAssign\:\:getExclusive to boolean.

## [1.34.4] - 2017-06-09

### Added
- Added support for graph-based implementations of nodes.
- Added support for subtree/subgraph pruning in traversals.
- Added NodeGraph\:\:topologicalSort and MaterialX\:\:printGraphDot methods.
- Added a File module to MaterialXFormat and MaterialXTest.

### Changed
- Extended NodeGraph::flattenSubgraphs to support subgraph recursion.
- Added a searchPath argument to MaterialX\:\:readFromXmlFile.

### Fixed
- Fixed an issue where connecting elements were not returned in graph traversal edges.

## [1.34.3] - 2017-05-16

### Added
- Added support for document validation, including the Document\:\:validate and Element\:\:validate methods.
- Added helper methods ValueElement\:\:getResolvedValueString and Element\:\:getNamePath.
- Added standard library document.
