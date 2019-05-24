# libshaderc_spvc

A library for validating and translating SPIR-V, using SPIRV-Tools validation and the SPIRV-Cross compiler.

## Build Artifacts

There are two main shaderc libraries that are created during a CMake
compilation. The first is `libshaderc_spvc`, which is a static library
containing just the functionality exposed by libshaderc_spvc. It depends
on other compilation targets `shaderc_util`, `SPIRV-Tools`, and `SPIRV-Cross`.

The other is `libshaderc_spvc_combined`, which is a static library containing
libshaderc_spvc and all of its dependencies.


## Integrating libshaderc_spvc

There are several ways of integrating libshaderc_spvc into external projects.

1. If the external project uses CMake, then `shaderc/CMakeLists.txt` can be
included into the external project's CMake configuration and shaderc_spvc can be used
as a link target.
This is the simplest way to use libshaderc_spvc in an external project.

2. If the external project uses CMake and is building for Linux or Android,
`target_link_libraries(shaderc_spvc_combined)` can instead be specified. This is
functionally identical to the previous option.

3. If the external project does not use CMake, then the external project can
instead directly use the generated libraries.  `shaderc/libshaderc_spvc/include`
should be added to the include path, and
`build/libshaderc_spvc/libshaderc_spvc_combined.a` should be linked. Note that on some
platforms `-lpthread` should also be specified.

4. If the external project does not use CMake and cannot use
`libshaderc_spvc_combined`, the following libraries or their platform-dependent
counterparts should be linked in the order specified.
  * `build/libshaderc_spvc/libshaderc_spvc.a`
  * `build/libshaderc_util/libshaderc_util.a`
  * `build/third_party/spirv-tools/libSPIRV-Tools-opt.a`
  * `build/third_party/spirv-tools/libSPIRV-Tools.a`
  * `build/third_party/SPIRV-Cross/libspirv-cross-core.a`
  * `build/third_party/SPIRV-Cross/libspirv-cross-glsl.a` etc.

5. If building for Android using the Android NDK, `shaderc/Android.mk` can be
included in the application's `Android.mk` and `LOCAL_STATIC_LIBRARIES:=shaderc_spvc`
can be specified. See `shaderc/android_test` for an example.
