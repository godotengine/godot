# libshaderc

A library for compiling shader strings into SPIR-V.

## Build Artifacts

There are two main shaderc libraries that are created during a CMake
compilation. The first is `libshaderc`, which is a static library
containing just the functionality exposed by libshaderc. It depends
on other compilation targets `glslang`, `OSDependent`, `OGLCompiler`,
`shaderc_util`, `SPIRV`, `HLSL`, `SPIRV-Tools`, and `SPIRV-Tools-opt`.

The other is `libshaderc_combined`, which is a static library containing
libshaderc and all of its dependencies.


## Integrating libshaderc

There are several ways of integrating libshaderc into external projects.

1. If the external project uses CMake, then `shaderc/CMakeLists.txt` can be
included into the external project's CMake configuration and shaderc can be used
as a link target.
This is the simplest way to use libshaderc in an external project.

2. If the external project uses CMake and is building for Linux or Android,
`target_link_libraries(shaderc_combined)` can instead be specified. This is
functionally identical to the previous option.

3. If the external project does not use CMake, then the external project can
instead directly use the generated libraries.  `shaderc/libshaderc/include`
should be added to the include path, and
`build/libshaderc/libshaderc_combined.a` should be linked. Note that on some
platforms `-lpthread` should also be specified.

4. If the external project does not use CMake and cannot use
`libshaderc_combined`, the following libraries or their platform-dependent
counterparts should be linked in the order specified.
  * `build/libshaderc/libshaderc.a`
  * `build/third_party/glslang/glslang/glslang.a`
  * `build/third_party/glslang/glslang/OSDependent/{Platform}/libOSDependent.a`
  * `build/third_party/glslang/OGLCompilersDLL/libOGLCompiler.a`
  * `build/third_party/glslang/libglslang.a`
  * `build/shaderc_util/libshaderc_util.a`
  * `build/third_party/glslang/SPIRV/libSPIRV.a`
  * `build/third_party/glslang/hlsl/libHLSL.a`
  * `build/third_party/spirv-tools/libSPIRV-Tools-opt.a`
  * `build/third_party/spirv-tools/libSPIRV-Tools.a`

5. If building for Android using the Android NDK, `shaderc/Android.mk` can be
included in the application's `Android.mk` and `LOCAL_STATIC_LIBRARIES:=shaderc`
can be specified. See `shaderc/android_test` for an example.
