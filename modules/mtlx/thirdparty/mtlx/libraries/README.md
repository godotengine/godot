# MaterialX Data Libraries

This folder contains the standard data libraries for MaterialX, providing declarations and graph definitions for the MaterialX nodes, and source code for all supported shader generators.

## Standard Pattern Library
- [stdlib](stdlib)
    - [stdlib_defs.mtlx](stdlib/stdlib_defs.mtlx) : Nodedef declarations.
    - [stdlib_ng.mtlx](stdlib/stdlib_ng.mtlx) : Nodegraph definitions.
    - [genglsl](stdlib/genglsl): GLSL language support.
        - [lib](stdlib/genglsl/lib) : Shader utility files.
        - [stdlib_genglsl_impl.mtlx](stdlib/genglsl/stdlib_genglsl_impl.mtlx) : Mapping from declarations to implementations.
    - [genosl](stdlib/genosl): OSL language support.
        - [lib](stdlib/genosl/lib) : Shader utility files.
        - [stdlib_genosl_impl.mtlx](stdlib/genosl/stdlib_genosl_impl.mtlx) : Mapping from declarations to implementations.
    - [genmdl](stdlib/genmdl): MDL language support.
        - [stdlib_genmdl_impl.mtlx](stdlib/genmdl/stdlib_genmdl_impl.mtlx) : Mapping from declarations to implementations.
        - Additional MaterialX support libraries for MDL are located in the [source/MaterialXGenMdl/mdl/materialx](../source/MaterialXGenMdl/mdl/materialx) package folder
    - [genmsl](stdlib/genmsl): MSL language support.
        - [lib](stdlib/genmsl/lib) : Shader utility files.
        - [stdlib_genmsl_impl.mtlx](stdlib/genmsl/stdlib_genmsl_impl.mtlx) : Mapping from declarations to implementations.

## Physically Based Shading Library
- [pbrlib](pbrlib)
    - [pbrlib_defs.mtlx](pbrlib/pbrlib_defs.mtlx) : Nodedef declarations.
    - [pbrlib_ng.mtlx](pbrlib/pbrlib_ng.mtlx) : Nodegraph definitions.
    - [genglsl](pbrlib/genglsl) : GLSL language support
        - [lib](pbrlib/genglsl/lib) : Shader utility files.
        - [pbrlib_genglsl_impl.mtlx](pbrlib/genglsl/pbrlib_genglsl_impl.mtlx) : Mapping from declarations to implementations.
    - [genosl](pbrlib/genosl) : OSL language support
        - [lib](pbrlib/genosl/lib) : Shader utility files.
        - [pbrlib_genosl_impl.mtlx](pbrlib/genosl/pbrlib_genosl_impl.mtlx) : Mapping from declarations to implementations.
    - [genmdl](pbrlib/genmdl) : MDL language support
        - [pbrlib_genmdl_impl.mtlx](pbrlib/genmdl/pbrlib_genmdl_impl.mtlx) : Mapping from declarations to implementations.
    - [genmsl](pbrlib/genmsl) : MSL language support
        - [pbrlib_genmsl_impl.mtlx](pbrlib/genmsl/pbrlib_genmsl_impl.mtlx) : Mapping from declarations to implementations.

## BxDF Graph Library
- [bxdf](bxdf)
    - [standard_surface.mtlx](bxdf/standard_surface.mtlx) : Graph definition of the [Autodesk Standard Surface](https://autodesk.github.io/standard-surface/) shading model.
    - [gltf_pbr.mtlx](bxdf/gltf_pbr.mtlx) : Graph definition of the [glTF PBR](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#appendix-b-brdf-implementation) shading model.
    - [usd_preview_surface.mtlx](bxdf/usd_preview_surface.mtlx) : Graph definition of the [UsdPreviewSurface](https://openusd.org/release/spec_usdpreviewsurface.html) shading model.
    - [lama](bxdf/lama) : Graph definitions of the [MaterialX Lama](https://rmanwiki.pixar.com/display/REN24/MaterialX+Lama) node set.

## Color Management Library
- MaterialX shader generation natively supports a small set of common spaces for input colors, with all color transforms implemented as language-independent MaterialX graphs.The canonical definitions of these color transforms may be found in the OpenColorIO configuration for [ACES 1.2](https://github.com/colour-science/OpenColorIO-Configs/tree/feature/aces-1.2-config/aces_1.2).
    - lin_rec709
    - g18_rec709
    - g22_rec709
    - rec709_display
    - acescg (lin_ap1)
    - g22_ap1
    - srgb_texture
    - lin_adobergb
    - adobergb
    - srgb_displayp3
    - lin_displayp3
- [cmlib](cmlib)
    - [cmlib_defs.mtlx](cmlib/cmlib_defs.mtlx) : Nodedef declarations.
    - [cmlib_ng.mtlx](cmlib/cmlib_ng.mtlx) : Nodegraph definitions.

## Target Definitions
- Each target implementation requires a target definition for declaration / implementation correspondence to work.
- The [targets](targets) folder contains definition files for the following core targets:
  - GLSL : `genglsl`
  - OSL : `genosl`
  - MDL : `genmdl`
  - MSL : `genmsl`
- Any additional target files should be added under this folder and loaded in as required.

### Target Support
- GLSL target support is for version 4.0 or higher.
- OSL target support is for version 1.9.10 or higher.
- MDL target support is for version 1.7.
- Basic GLSL and MSL `lightshader` node definitions and implementations are provided for the following light types:
    - point, directional, spot
- Shader generation does not currently support:
    - `ambientocclusion` node.
    - `arrayappend` node.
    - `curveadjust` node.
    - `displacementshader` and `volumeshader` nodes for hardware shading targets (GLSL, MSL).
