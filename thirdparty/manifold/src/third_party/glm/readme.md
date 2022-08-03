![glm](/doc/manual/logo-mini.png)

[OpenGL Mathematics](http://glm.g-truc.net/) (*GLM*) is a header only C++ mathematics library for graphics software based on the [OpenGL Shading Language (GLSL) specifications](https://www.opengl.org/registry/doc/GLSLangSpec.4.50.diff.pdf).

*GLM* provides classes and functions designed and implemented with the same naming conventions and functionality than *GLSL* so that anyone who knows *GLSL*, can use *GLM* as well in C++.

This project isn't limited to *GLSL* features. An extension system, based on the *GLSL* extension conventions, provides extended capabilities: matrix transformations, quaternions, data packing, random numbers, noise, etc...

This library works perfectly with *[OpenGL](https://www.opengl.org)* but it also ensures interoperability with other third party libraries and SDK. It is a good candidate for software rendering (raytracing / rasterisation), image processing, physics simulations and any development context that requires a simple and convenient mathematics library.

*GLM* is written in C++98 but can take advantage of C++11 when supported by the compiler. It is a platform independent library with no dependence and it officially supports the following compilers:
- [*GCC*](http://gcc.gnu.org/) 4.7 and higher
- [*Intel C++ Compose*](https://software.intel.com/en-us/intel-compilers) XE 2013 and higher
- [*Clang*](http://llvm.org/) 3.4 and higher
- [*Apple Clang 6.0*](https://developer.apple.com/library/mac/documentation/CompilerTools/Conceptual/LLVMCompilerOverview/index.html) and higher
- [*Visual C++*](http://www.visualstudio.com/) 2013 and higher
- [*CUDA*](https://developer.nvidia.com/about-cuda) 9.0 and higher (experimental)
- [*SYCL*](https://www.khronos.org/sycl/) (experimental: only [ComputeCpp](https://codeplay.com/products/computesuite/computecpp) implementation has been tested).
- Any C++11 compiler

For more information about *GLM*, please have a look at the [manual](manual.md) and the [API reference documentation](http://glm.g-truc.net/0.9.8/api/index.html).
The source code and the documentation are licensed under either the [Happy Bunny License (Modified MIT) or the MIT License](manual.md#section0).

Thanks for contributing to the project by [submitting pull requests](https://github.com/g-truc/glm/pulls).

```cpp
#include <glm/vec3.hpp> // glm::vec3
#include <glm/vec4.hpp> // glm::vec4
#include <glm/mat4x4.hpp> // glm::mat4
#include <glm/ext/matrix_transform.hpp> // glm::translate, glm::rotate, glm::scale
#include <glm/ext/matrix_clip_space.hpp> // glm::perspective
#include <glm/ext/scalar_constants.hpp> // glm::pi

glm::mat4 camera(float Translate, glm::vec2 const& Rotate)
{
	glm::mat4 Projection = glm::perspective(glm::pi<float>() * 0.25f, 4.0f / 3.0f, 0.1f, 100.f);
	glm::mat4 View = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, -Translate));
	View = glm::rotate(View, Rotate.y, glm::vec3(-1.0f, 0.0f, 0.0f));
	View = glm::rotate(View, Rotate.x, glm::vec3(0.0f, 1.0f, 0.0f));
	glm::mat4 Model = glm::scale(glm::mat4(1.0f), glm::vec3(0.5f));
	return Projection * View * Model;
}
```

## [Lastest release](https://github.com/g-truc/glm/releases/latest)

## Project Health

| Service | System | Compiler | Status |
| ------- | ------ | -------- | ------ |
| [Travis CI](https://travis-ci.org/g-truc/glm)| MacOSX, Linux 64 bits | Clang 3.6, Clang 5.0, GCC 4.9, GCC 7.3 | [![Travis CI](https://travis-ci.org/g-truc/glm.svg?branch=master)](https://travis-ci.org/g-truc/glm)
| [AppVeyor](https://ci.appveyor.com/project/Groovounet/glm)| Windows 32 and 64 | Visual Studio 2013, Visual Studio 2015, Visual Studio 2017 | [![AppVeyor](https://ci.appveyor.com/api/projects/status/32r7s2skrgm9ubva?svg=true)](https://ci.appveyor.com/project/Groovounet/glm)

## Release notes

### [GLM 0.9.9.9](https://github.com/g-truc/glm/releases/tag/0.9.9.9) - 2020-XX-XX
#### Features:
- Added *GLM_EXT_scalar_reciprocal* with tests
- Added *GLM_EXT_vector_reciprocal* with tests
- Added `glm::iround` and `glm::uround` to *GLM_EXT_scalar_common* and *GLM_EXT_vector_common*
- Added *GLM_EXT_matrix_integer* with tests

#### Improvements:
- Added `constexpr` qualifier for `cross` product #1040
- Added `constexpr` qualifier for `dot` product #1040

#### Fixes:
- Fixed incorrect assertion for `glm::min` and `glm::max` #1009
- Fixed quaternion orientation in `glm::decompose` #1012
- Fixed singularity in quaternion to euler angle roll conversion #1019
- Fixed `quat` `glm::pow` handling of small magnitude quaternions #1022
- Fixed `glm::fastNormalize` build error #1033
- Fixed `glm::isMultiple` build error #1034
- Fixed `glm::adjugate` calculation #1035
- Fixed `glm::angle` discards the sign of result for angles in range (2*pi-1, 2*pi) #1038
- Removed ban on using `glm::string_cast` with *CUDA* host code #1041

### [GLM 0.9.9.8](https://github.com/g-truc/glm/releases/tag/0.9.9.8) - 2020-04-13
#### Features:
- Added *GLM_EXT_vector_intX* and *GLM_EXT_vector_uintX* extensions
- Added *GLM_EXT_matrix_intX* and *GLM_EXT_matrix_uintX* extensions

#### Improvements:
- Added `glm::clamp`, `glm::repeat`, `glm::mirrorClamp` and `glm::mirrorRepeat` function to `GLM_EXT_scalar_commond` and `GLM_EXT_vector_commond` extensions with tests

#### Fixes:
- Fixed unnecessary warnings from `matrix_projection.inl` #995
- Fixed quaternion `glm::slerp` overload which interpolates with extra spins #996
- Fixed for `glm::length` using arch64 #992
- Fixed singularity check for `glm::quatLookAt` #770

### [GLM 0.9.9.7](https://github.com/g-truc/glm/releases/tag/0.9.9.7) - 2020-01-05
#### Improvements:
- Improved *Neon* support with more functions optimized #950
- Added *CMake* *GLM* interface #963
- Added `glm::fma` implementation based on `std::fma` #969
- Added missing quat constexpr #955
- Added `GLM_FORCE_QUAT_DATA_WXYZ` to store quat data as w,x,y,z instead of x,y,z,w #983

#### Fixes:
- Fixed equal *ULP* variation when using negative sign #965
- Fixed for intersection ray/plane and added related tests #953
- Fixed ARM 64bit detection #949
- Fixed *GLM_EXT_matrix_clip_space* warnings #980
- Fixed Wimplicit-int-float-conversion warnings with clang 10+ #986
- Fixed *GLM_EXT_matrix_clip_space* `perspectiveFov`

### [GLM 0.9.9.6](https://github.com/g-truc/glm/releases/tag/0.9.9.6) - 2019-09-08
#### Features:
- Added *Neon* support #945
- Added *SYCL* support #914
- Added *GLM_EXT_scalar_integer* extension with power of two and multiple scalar functions
- Added *GLM_EXT_vector_integer* extension with power of two and multiple vector functions

#### Improvements:
- Added *Visual C++ 2019* detection
- Added *Visual C++ 2017* 15.8 and 15.9 detection
- Added missing genType check for `glm::bitCount` and `glm::bitfieldReverse` #893

#### Fixes:
- Fixed for g++6 where -std=c++1z sets __cplusplus to 201500 instead of 201402 #921
- Fixed hash hashes qua instead of tquat #919
- Fixed `.natvis` as structs renamed #915
- Fixed `glm::ldexp` and `glm::frexp` declaration #895
- Fixed missing const to quaternion conversion operators #890
- Fixed *GLM_EXT_scalar_ulp* and *GLM_EXT_vector_ulp* API coding style
- Fixed quaternion componant order: `w, {x, y, z}` #916
- Fixed `GLM_HAS_CXX11_STL` broken on Clang with Linux #926
- Fixed *Clang* or *GCC* build due to wrong `GLM_HAS_IF_CONSTEXPR` definition #907
- Fixed *CUDA* 9 build #910

#### Deprecation:
 - Removed CMake install and uninstall scripts

### [GLM 0.9.9.5](https://github.com/g-truc/glm/releases/tag/0.9.9.5) - 2019-04-01
#### Fixes:
- Fixed build errors when defining `GLM_ENABLE_EXPERIMENTAL` #884 #883
- Fixed `if constexpr` warning #887
- Fixed missing declarations for `glm::frexp` and `glm::ldexp` #886

### [GLM 0.9.9.4](https://github.com/g-truc/glm/releases/tag/0.9.9.4) - 2019-03-19
#### Features:
- Added `glm::mix` implementation for matrices in *GLM_EXT_matrix_common/ #842
- Added *CMake* `BUILD_SHARED_LIBS` and `BUILD_STATIC_LIBS` build options #871

#### Improvements:
- Added GLM_FORCE_INTRINSICS to enable SIMD instruction code path. By default, it's disabled allowing constexpr support by default. #865
- Optimized inverseTransform #867

#### Fixes:
- Fixed in `glm::mat4x3` conversion #829
- Fixed `constexpr` issue on GCC #832 #865
- Fixed `glm::mix` implementation to improve GLSL conformance #866
- Fixed `glm::int8` being defined as unsigned char with some compiler #839
- Fixed `glm::vec1` include #856
- Ignore `.vscode` #848

### [GLM 0.9.9.3](https://github.com/g-truc/glm/releases/tag/0.9.9.3) - 2018-10-31
#### Features:
- Added `glm::equal` and `glm::notEqual` overload with max ULPs parameters for scalar numbers #121
- Added `GLM_FORCE_SILENT_WARNINGS` to silent *GLM* warnings when using language extensions but using W4 or Wpedantic warnings #814 #775
- Added adjugate functions to `GLM_GTX_matrix_operation` #151
- Added `GLM_FORCE_ALIGNED_GENTYPES` to enable aligned types and SIMD instruction are not enabled. This disable `constexpr` #816

#### Improvements:
- Added constant time ULP distance between float #121
- Added `GLM_FORCE_SILENT_WARNINGS` to suppress *GLM* warnings #822

#### Fixes:
- Fixed `glm::simplex` noise build with double #734
- Fixed `glm::bitfieldInsert` according to GLSL spec #818
- Fixed `glm::refract` for negative 'k' #808

### [GLM 0.9.9.2](https://github.com/g-truc/glm/releases/tag/0.9.9.2) - 2018-09-14
#### Fixes:
- Fixed `GLM_FORCE_CXX**` section in the manual
- Fixed default initialization with vector and quaternion types using `GLM_FORCE_CTOR_INIT` #812

### [GLM 0.9.9.1](https://github.com/g-truc/glm/releases/tag/0.9.9.1) - 2018-09-03
#### Features:
- Added `bitfieldDeinterleave` to *GLM_GTC_bitfield*
- Added missing `glm::equal` and `glm::notEqual` with epsilon for quaternion types to *GLM_GTC_quaternion*
- Added *GLM_EXT_matrix_relational*: `glm::equal` and `glm::notEqual` with epsilon for matrix types
- Added missing aligned matrix types to *GLM_GTC_type_aligned*
- Added C++17 detection
- Added *Visual C++* language standard version detection
- Added PDF manual build from markdown

#### Improvements:
- Added a section to the manual for contributing to *GLM*
- Refactor manual, lists all configuration defines
- Added missing `glm::vec1` based constructors
- Redesigned constexpr support which excludes both SIMD and `constexpr` #783
- Added detection of *Visual C++ 2017* toolsets
- Added identity functions #765
- Splitted headers into EXT extensions to improve compilation time #670
- Added separated performance tests
- Clarified refract valid range of the indices of refraction, between -1 and 1 inclusively #806

#### Fixes:
- Fixed SIMD detection on *Clang* and *GCC*
- Fixed build problems due to `std::printf` and `std::clock_t` #778
- Fixed int mod
- Anonymous unions require C++ language extensions
- Fixed `glm::ortho` #790
- Fixed *Visual C++* 2013 warnings in vector relational code #782
- Fixed *ICC* build errors with constexpr #704
- Fixed defaulted operator= and constructors #791
- Fixed invalid conversion from int scalar with vec4 constructor when using SSE instruction
- Fixed infinite loop in random functions when using negative radius values using an assert #739

### [GLM 0.9.9.0](https://github.com/g-truc/glm/releases/tag/0.9.9.0) - 2018-05-22
#### Features:
- Added *RGBM* encoding in *GLM_GTC_packing* #420
- Added *GLM_GTX_color_encoding* extension
- Added *GLM_GTX_vec_swizzle*, faster compile time swizzling then swizzle operator #558
- Added *GLM_GTX_exterior_product* with a `vec2` `glm::cross` implementation #621
- Added *GLM_GTX_matrix_factorisation* to factor matrices in various forms #654
- Added [`GLM_ENABLE_EXPERIMENTAL`](manual.md#section7_4) to enable experimental features.
- Added packing functions for integer vectors #639
- Added conan packaging configuration #643 #641
- Added `glm::quatLookAt` to *GLM_GTX_quaternion* #659
- Added `glm::fmin`, `glm::fmax` and `glm::fclamp` to *GLM_GTX_extended_min_max* #372
- Added *GLM_EXT_vector_relational*: extend `glm::equal` and `glm::notEqual` to take an epsilon argument
- Added *GLM_EXT_vector_relational*: `glm::openBounded` and `glm::closeBounded`
- Added *GLM_EXT_vec1*: `*vec1` types
- Added *GLM_GTX_texture*: `levels` function
- Added spearate functions to use both nagative one and zero near clip plans #680
- Added `GLM_FORCE_SINGLE_ONLY` to use *GLM* on platforms that don't support double #627
- Added *GLM_GTX_easing* for interpolation functions #761

#### Improvements:
- No more default initialization of vector, matrix and quaternion types
- Added lowp variant of GTC_color_space convertLinearToSRGB #419
- Replaced the manual by a markdown version #458
- Improved API documentation #668
- Optimized GTC_packing implementation
- Optimized GTC_noise functions
- Optimized GTC_color_space HSV to RGB conversions
- Optimised GTX_color_space_YCoCg YCoCgR conversions
- Optimized GTX_matrix_interpolation axisAngle function
- Added FAQ 12: Windows headers cause build errors... #557
- Removed GCC shadow warnings #595
- Added error for including of different versions of GLM #619
- Added GLM_FORCE_IGNORE_VERSION to ignore error caused by including different version of GLM #619
- Reduced warnings when using very strict compilation flags #646
- length() member functions are constexpr #657
- Added support of -Weverything with Clang #646
- Improved exponential function test coverage
- Enabled warnings as error with Clang unit tests
- Conan package is an external repository: https://github.com/bincrafters/conan-glm
- Clarify quat_cast documentation, applying on pure rotation matrices #759

#### Fixes:
- Removed doxygen references to *GLM_GTC_half_float* which was removed in 0.9.4
- Fixed `glm::decompose` #448
- Fixed `glm::intersectRayTriangle` #6
- Fixed dual quaternion != operator #629
- Fixed usused variable warning in *GLM_GTX_spline* #618
- Fixed references to `GLM_FORCE_RADIANS` which was removed #642
- Fixed `glm::fastInverseSqrt` to use fast inverse square #640
- Fixed `glm::axisAngle` NaN #638
- Fixed integer pow from *GLM_GTX_integer* with null exponent #658
- Fixed `quat` `normalize` build error #656
- Fixed *Visual C++ 2017.2* warning regarding `__has_feature` definision #655
- Fixed documentation warnings
- Fixed `GLM_HAS_OPENMP` when *OpenMP* is not enabled
- Fixed Better follow GLSL `min` and `max` specification #372
- Fixed quaternion constructor from two vectors special cases #469
- Fixed `glm::to_string` on quaternions wrong components order #681
- Fixed `glm::acsch` #698
- Fixed `glm::isnan` on *CUDA* #727

#### Deprecation:
- Requires *Visual Studio 2013*, *GCC 4.7*, *Clang 3.4*, *Cuda 7*, *ICC 2013* or a C++11 compiler
- Removed *GLM_GTX_simd_vec4* extension
- Removed *GLM_GTX_simd_mat4* extension
- Removed *GLM_GTX_simd_quat* extension
- Removed `GLM_SWIZZLE`, use `GLM_FORCE_SWIZZLE` instead
- Removed `GLM_MESSAGES`, use `GLM_FORCE_MESSAGES` instead
- Removed `GLM_DEPTH_ZERO_TO_ONE`, use `GLM_FORCE_DEPTH_ZERO_TO_ONE` instead
- Removed `GLM_LEFT_HANDED`, use `GLM_FORCE_LEFT_HANDED` instead
- Removed `GLM_FORCE_NO_CTOR_INIT`
- Removed `glm::uninitialize`

---
### [GLM 0.9.8.5](https://github.com/g-truc/glm/releases/tag/0.9.8.5) - 2017-08-16
#### Features:
- Added *Conan* package support #647

#### Fixes:
- Fixed *Clang* version detection from source #608
- Fixed `glm::packF3x9_E1x5` exponent packing #614
- Fixed build error `min` and `max` specializations with integer #616
- Fixed `simd_mat4` build error #652

---
### [GLM 0.9.8.4](https://github.com/g-truc/glm/releases/tag/0.9.8.4) - 2017-01-22
#### Fixes:
- Fixed *GLM_GTC_packing* test failing on *GCC* x86 due to denorms #212 #577
- Fixed `POPCNT` optimization build in *Clang* #512
- Fixed `glm::intersectRayPlane` returns true in parallel case #578
- Fixed *GCC* 6.2 compiler warnings #580
- Fixed *GLM_GTX_matrix_decompose* `glm::decompose` #582 #448
- Fixed *GCC* 4.5 and older build #566
- Fixed *Visual C++* internal error when declaring a global vec type with siwzzle expression enabled #594
- Fixed `GLM_FORCE_CXX11` with Clang and libstlc++ which wasn't using C++11 STL features. #604

---
### [GLM 0.9.8.3](https://github.com/g-truc/glm/releases/tag/0.9.8.3) - 2016-11-12
#### Improvements:
- Broader support of `GLM_FORCE_UNRESTRICTED_GENTYPE` #378

#### Fixes:
- Fixed Android build error with C++11 compiler but C++98 STL #284 #564
- Fixed *GLM_GTX_transform2* shear* functions #403
- Fixed interaction between `GLM_FORCE_UNRESTRICTED_GENTYPE` and `glm::ortho` function #568
- Fixed `glm::bitCount` with AVX on 32 bit builds #567
- Fixed *CMake* `find_package` with version specification #572 #573

---
### [GLM 0.9.8.2](https://github.com/g-truc/glm/releases/tag/0.9.8.2) - 2016-11-01
#### Improvements:
- Added *Visual C++* 15 detection
- Added *Clang* 4.0 detection
- Added warning messages when using `GLM_FORCE_CXX**` but the compiler
  is known to not fully support the requested C++ version #555
- Refactored `GLM_COMPILER_VC` values
- Made quat, vec, mat type component `length()` static #565

#### Fixes:
- Fixed *Visual C++* `constexpr` build error #555, #556

---
### [GLM 0.9.8.1](https://github.com/g-truc/glm/releases/tag/0.9.8.1) - 2016-09-25
#### Improvements:
- Optimized quaternion `glm::log` function #554

#### Fixes:
- Fixed *GCC* warning filtering, replaced -pedantic by -Wpedantic
- Fixed SIMD faceforward bug. #549
- Fixed *GCC* 4.8 with C++11 compilation option #550
- Fixed *Visual Studio* aligned type W4 warning #548
- Fixed packing/unpacking function fixed for 5_6_5 and 5_5_5_1 #552

---
### [GLM 0.9.8.0](https://github.com/g-truc/glm/releases/tag/0.9.8.0) - 2016-09-11
#### Features:
- Added right and left handed projection and clip control support #447 #415 #119
- Added `glm::compNormalize` and `glm::compScale` functions to *GLM_GTX_component_wise*
- Added `glm::packF3x9_E1x5` and `glm::unpackF3x9_E1x5` to *GLM_GTC_packing* for RGB9E5 #416
- Added `(un)packHalf` to *GLM_GTC_packing*
- Added `(un)packUnorm` and `(un)packSnorm` to *GLM_GTC_packing*
- Added 16bit pack and unpack to *GLM_GTC_packing*
- Added 8bit pack and unpack to *GLM_GTC_packing*
- Added missing `bvec*` && and || operators
- Added `glm::iround` and `glm::uround` to *GLM_GTC_integer*, fast round on positive values
- Added raw SIMD API
- Added 'aligned' qualifiers
- Added *GLM_GTC_type_aligned* with aligned *vec* types
- Added *GLM_GTC_functions* extension
- Added quaternion version of `glm::isnan` and `glm::isinf` #521
- Added `glm::lowestBitValue` to *GLM_GTX_bit* #536
- Added `GLM_FORCE_UNRESTRICTED_GENTYPE` allowing non basic `genType` #543

#### Improvements:
- Improved SIMD and swizzle operators interactions with *GCC* and *Clang* #474
- Improved *GLM_GTC_random* `linearRand` documentation
- Improved *GLM_GTC_reciprocal* documentation
- Improved `GLM_FORCE_EXPLICIT_CTOR` coverage #481
- Improved *OpenMP* support detection for *Clang*, *GCC*, *ICC* and *VC*
- Improved *GLM_GTX_wrap* for SIMD friendliness
- Added `constexpr` for `*vec*`, `*mat*`, `*quat*` and `*dual_quat*` types #493
- Added *NEON* instruction set detection
- Added *MIPS* CPUs detection
- Added *PowerPC* CPUs detection
- Use *Cuda* built-in function for abs function implementation with Cuda compiler
- Factorized `GLM_COMPILER_LLVM` and `GLM_COMPILER_APPLE_CLANG` into `GLM_COMPILER_CLANG`
- No more warnings for use of long long
- Added more information to build messages

#### Fixes:
- Fixed *GLM_GTX_extended_min_max* filename typo #386
- Fixed `glm::intersectRayTriangle` to not do any unintentional backface culling
- Fixed long long warnings when using C++98 on *GCC* and *Clang* #482
- Fixed sign with signed integer function on non-x86 architecture
- Fixed strict aliasing warnings #473
- Fixed missing `glm::vec1` overload to `glm::length2` and `glm::distance2` functions #431
- Fixed *GLM* test '/fp:fast' and '/Za' command-line options are incompatible
- Fixed quaterion to mat3 cast function `glm::mat3_cast` from *GLM_GTC_quaternion* #542
- Fixed *GLM_GTX_io* for *Cuda* #547 #546

#### Deprecation:
- Removed `GLM_FORCE_SIZE_FUNC` define
- Deprecated *GLM_GTX_simd_vec4* extension
- Deprecated *GLM_GTX_simd_mat4* extension
- Deprecated *GLM_GTX_simd_quat* extension
- Deprecated `GLM_SWIZZLE`, use `GLM_FORCE_SWIZZLE` instead
- Deprecated `GLM_MESSAGES`, use `GLM_FORCE_MESSAGES` instead

---
### [GLM 0.9.7.6](https://github.com/g-truc/glm/releases/tag/0.9.7.6) - 2016-07-16
#### Improvements:
- Added pkg-config file #509
- Updated list of compiler versions detected
- Improved C++ 11 STL detection #523

#### Fixes:
- Fixed STL for C++11 detection on ICC #510
- Fixed missing vec1 overload to length2 and distance2 functions #431
- Fixed long long warnings when using C++98 on GCC and Clang #482
- Fixed scalar reciprocal functions (GTC_reciprocal) #520

---
### [GLM 0.9.7.5](https://github.com/g-truc/glm/releases/tag/0.9.7.5) - 2016-05-24
#### Improvements:
- Added Visual C++ Clang toolset detection

#### Fixes:
- Fixed uaddCarry warning #497
- Fixed roundPowerOfTwo and floorPowerOfTwo #503
- Fixed Visual C++ SIMD instruction set automatic detection in 64 bits
- Fixed to_string when used with GLM_FORCE_INLINE #506
- Fixed GLM_FORCE_INLINE with binary vec4 operators
- Fixed GTX_extended_min_max filename typo #386
- Fixed intersectRayTriangle to not do any unintentional backface culling

---
### [GLM 0.9.7.4](https://github.com/g-truc/glm/releases/tag/0.9.7.4) - 2016-03-19
#### Fixes:
- Fixed asinh and atanh warning with C++98 STL #484
- Fixed polar coordinates function latitude #485
- Fixed outerProduct defintions and operator signatures for mat2x4 and vec4 #475
- Fixed eulerAngles precision error, returns NaN  #451
- Fixed undefined reference errors #489
- Fixed missing GLM_PLATFORM_CYGWIN declaration #495
- Fixed various undefined reference errors #490

---
### [GLM 0.9.7.3](https://github.com/g-truc/glm/releases/tag/0.9.7.3) - 2016-02-21
#### Improvements:
- Added AVX512 detection

#### Fixes:
- Fixed CMake policy warning
- Fixed GCC 6.0 detection #477
- Fixed Clang build on Windows #479
- Fixed 64 bits constants warnings on GCC #463

---
### [GLM 0.9.7.2](https://github.com/g-truc/glm/releases/tag/0.9.7.2) - 2016-01-03
#### Fixes:
- Fixed GTC_round floorMultiple/ceilMultiple #412
- Fixed GTC_packing unpackUnorm3x10_1x2 #414
- Fixed GTC_matrix_inverse affineInverse #192
- Fixed ICC on Linux build errors #449
- Fixed ldexp and frexp compilation errors
- Fixed "Declaration shadows a field" warning #468
- Fixed 'GLM_COMPILER_VC2005 is not defined' warning #468
- Fixed various 'X is not defined' warnings #468
- Fixed missing unary + operator #435
- Fixed Cygwin build errors when using C++11 #405

---
### [GLM 0.9.7.1](https://github.com/g-truc/glm/releases/tag/0.9.7.1) - 2015-09-07
#### Improvements:
- Improved constexpr for constant functions coverage #198
- Added to_string for quat and dual_quat in GTX_string_cast #375
- Improved overall execution time of unit tests #396

#### Fixes:
- Fixed strict alignment warnings #235 #370
- Fixed link errors on compilers not supported default function #377
- Fixed compilation warnings in vec4
- Fixed non-identity quaternions for equal vectors #234
- Fixed excessive GTX_fast_trigonometry execution time #396
- Fixed Visual Studio 2015 'hides class member' warnings #394
- Fixed builtin bitscan never being used #392
- Removed unused func_noise.* files #398

---
### [GLM 0.9.7.0](https://github.com/g-truc/glm/releases/tag/0.9.7.0) - 2015-08-02
#### Features:
- Added GTC_color_space: convertLinearToSRGB and convertSRGBToLinear functions
- Added 'fmod' overload to GTX_common with tests #308
- Left handed perspective and lookAt functions #314
- Added functions eulerAngleXYZ and extractEulerAngleXYZ #311
- Added <glm/gtx/hash.hpp> to perform std::hash on GLM types #320 #367
- Added <glm/gtx/wrap.hpp> for texcoord wrapping
- Added static components and precision members to all vector and quat types #350
- Added .gitignore #349
- Added support of defaulted functions to GLM types, to use them in unions #366

#### Improvements:
- Changed usage of __has_include to support Intel compiler #307
- Specialized integer implementation of YCoCg-R #310
- Don't show status message in 'FindGLM' if 'QUIET' option is set. #317
- Added master branch continuous integration service on Linux 64 #332
- Clarified manual regarding angle unit in GLM, added FAQ 11 #326
- Updated list of compiler versions

#### Fixes:
- Fixed default precision for quat and dual_quat type #312
- Fixed (u)int64 MSB/LSB handling on BE archs #306
- Fixed multi-line comment warning in g++. #315
- Fixed specifier removal by 'std::make_pair<>' #333
- Fixed perspective fovy argument documentation #327
- Removed -m64 causing build issues on Linux 32 #331
- Fixed isfinite with C++98 compilers #343
- Fixed Intel compiler build error on Linux #354
- Fixed use of libstdc++ with Clang #351
- Fixed quaternion pow #346
- Fixed decompose warnings #373
- Fixed matrix conversions #371

#### Deprecation:
- Removed integer specification for 'mod' in GTC_integer #308
- Removed GTX_multiple, replaced by GTC_round

---
### [GLM 0.9.6.3](https://github.com/g-truc/glm/releases/tag/0.9.6.3) - 2015-02-15
- Fixed Android doesn't have C++ 11 STL #284

---
### [GLM 0.9.6.2](https://github.com/g-truc/glm/releases/tag/0.9.6.2) - 2015-02-15
#### Features:
- Added display of GLM version with other GLM_MESSAGES
- Added ARM instruction set detection

#### Improvements:
- Removed assert for perspective with zFar < zNear #298
- Added Visual Studio natvis support for vec1, quat and dualqual types
- Cleaned up C++11 feature detections
- Clarify GLM licensing

#### Fixes:
- Fixed faceforward build #289
- Fixed conflict with Xlib #define True 1 #293
- Fixed decompose function VS2010 templating issues #294
- Fixed mat4x3 = mat2x3 * mat4x2 operator #297
- Fixed warnings in F2x11_1x10 packing function in GTC_packing #295
- Fixed Visual Studio natvis support for vec4 #288
- Fixed GTC_packing *pack*norm*x* build and added tests #292
- Disabled GTX_scalar_multiplication for GCC, failing to build tests #242
- Fixed Visual C++ 2015 constexpr errors: Disabled only partial support
- Fixed functions not inlined with Clang #302
- Fixed memory corruption (undefined behaviour) #303

---
### [GLM 0.9.6.1](https://github.com/g-truc/glm/releases/tag/0.9.6.1) - 2014-12-10
#### Features:
- Added GLM_LANG_CXX14_FLAG and GLM_LANG_CXX1Z_FLAG language feature flags
- Added C++14 detection

#### Improvements:
- Clean up GLM_MESSAGES compilation log to report only detected capabilities

#### Fixes:
- Fixed scalar uaddCarry build error with Cuda #276
- Fixed C++11 explicit conversion operators detection #282
- Fixed missing explicit conversion when using integer log2 with *vec1 types
- Fixed 64 bits integer GTX_string_cast to_string on VC 32 bit compiler
- Fixed Android build issue, STL C++11 is not supported by the NDK #284
- Fixed unsupported _BitScanForward64 and _BitScanReverse64 in VC10
- Fixed Visual C++ 32 bit build #283
- Fixed GLM_FORCE_SIZE_FUNC pragma message
- Fixed C++98 only build
- Fixed conflict between GTX_compatibility and GTC_quaternion #286
- Fixed C++ language restriction using GLM_FORCE_CXX**

---
### [GLM 0.9.6.0](https://github.com/g-truc/glm/releases/tag/0.9.6.0) - 2014-11-30
#### Features:
- Exposed template vector and matrix types in 'glm' namespace #239, #244
- Added GTX_scalar_multiplication for C++ 11 compiler only #242
- Added GTX_range for C++ 11 compiler only #240
- Added closestPointOnLine function for tvec2 to GTX_closest_point #238
- Added GTC_vec1 extension, *vec1 support to *vec* types
- Updated GTX_associated_min_max with vec1 support
- Added support of precision and integers to linearRand #230
- Added Integer types support to GTX_string_cast #249
- Added vec3 slerp #237
- Added GTX_common with isdenomal #223
- Added GLM_FORCE_SIZE_FUNC to replace .length() by .size() #245
- Added GLM_FORCE_NO_CTOR_INIT
- Added 'uninitialize' to explicitly not initialize a GLM type
- Added GTC_bitfield extension, promoted GTX_bit
- Added GTC_integer extension, promoted GTX_bit and GTX_integer
- Added GTC_round extension, promoted GTX_bit
- Added GLM_FORCE_EXPLICIT_CTOR to require explicit type conversions #269
- Added GTX_type_aligned for aligned vector, matrix and quaternion types

#### Improvements:
- Rely on C++11 to implement isinf and isnan
- Removed GLM_FORCE_CUDA, Cuda is implicitly detected
- Separated Apple Clang and LLVM compiler detection
- Used pragma once
- Undetected C++ compiler automatically compile with GLM_FORCE_CXX98 and 
  GLM_FORCE_PURE
- Added not function (from GLSL specification) on VC12
- Optimized bitfieldReverse and bitCount functions
- Optimized findLSB and findMSB functions.
- Optimized matrix-vector multiple performance with Cuda #257, #258
- Reduced integer type redifinitions #233
- Rewrited of GTX_fast_trigonometry #264 #265
- Made types trivially copyable #263
- Removed <iostream> in GLM tests
- Used std features within GLM without redeclaring
- Optimized cot function #272
- Optimized sign function #272
- Added explicit cast from quat to mat3 and mat4 #275

#### Fixes:
- Fixed std::nextafter not supported with C++11 on Android #217
- Fixed missing value_type for dual quaternion
- Fixed return type of dual quaternion length
- Fixed infinite loop in isfinite function with GCC #221
- Fixed Visual Studio 14 compiler warnings
- Fixed implicit conversion from another tvec2 type to another tvec2 #241
- Fixed lack of consistency of quat and dualquat constructors
- Fixed uaddCarray #253
- Fixed float comparison warnings #270

#### Deprecation:
- Requires Visual Studio 2010, GCC 4.2, Apple Clang 4.0, LLVM 3.0, Cuda 4, ICC 2013 or a C++98 compiler
- Removed degrees for function parameters
- Removed GLM_FORCE_RADIANS, active by default
- Removed VC 2005 / 8 and 2008 / 9 support
- Removed GCC 3.4 to 4.3 support
- Removed LLVM GCC support
- Removed LLVM 2.6 to 3.1 support
- Removed CUDA 3.0 to 3.2 support

---
### [GLM 0.9.5.4 - 2014-06-21](https://github.com/g-truc/glm/releases/tag/0.9.5.4)
- Fixed non-utf8 character #196
- Added FindGLM install for CMake #189
- Fixed GTX_color_space - saturation #195
- Fixed glm::isinf and glm::isnan for with Android NDK 9d #191
- Fixed builtin GLM_ARCH_SSE4 #204
- Optimized Quaternion vector rotation #205
- Fixed missing doxygen @endcond tag #211
- Fixed instruction set detection with Clang #158
- Fixed orientate3 function #207
- Fixed lerp when cosTheta is close to 1 in quaternion slerp #210
- Added GTX_io for io with <iostream> #144
- Fixed fastDistance ambiguity #215
- Fixed tweakedInfinitePerspective #208 and added user-defined epsilon to
  tweakedInfinitePerspective
- Fixed std::copy and std::vector with GLM types #214
- Fixed strict aliasing issues #212, #152
- Fixed std::nextafter not supported with C++11 on Android #213
- Fixed corner cases in exp and log functions for quaternions #199

---
### GLM 0.9.5.3 - 2014-04-02
- Added instruction set auto detection with Visual C++ using _M_IX86_FP - /arch
  compiler argument
- Fixed GTX_raw_data code dependency
- Fixed GCC instruction set detection
- Added GLM_GTX_matrix_transform_2d extension (#178, #176)
- Fixed CUDA issues (#169, #168, #183, #182)
- Added support for all extensions but GTX_string_cast to CUDA
- Fixed strict aliasing warnings in GCC 4.8.1 / Android NDK 9c (#152)
- Fixed missing bitfieldInterleave definisions
- Fixed usubBorrow (#171)
- Fixed eulerAngle*** not consistent for right-handed coordinate system (#173)
- Added full tests for eulerAngle*** functions (#173)
- Added workaround for a CUDA compiler bug (#186, #185)

---
### GLM 0.9.5.2 - 2014-02-08
- Fixed initializer list ambiguity (#159, #160)
- Fixed warnings with the Android NDK 9c
- Fixed non power of two matrix products
- Fixed mix function link error
- Fixed SSE code included in GLM tests on "pure" platforms
- Fixed undefined reference to fastInverseSqrt (#161)
- Fixed GLM_FORCE_RADIANS with <glm/ext.hpp> build error (#165)
- Fix dot product clamp range for vector angle functions. (#163)
- Tentative fix for strict aliasing warning in GCC 4.8.1 / Android NDK 9c (#152)
- Fixed GLM_GTC_constants description brief (#162)

---
### GLM 0.9.5.1 - 2014-01-11
- Fixed angle and orientedAngle that sometimes return NaN values (#145)
- Deprecated degrees for function parameters and display a message
- Added possible static_cast conversion of GLM types (#72)
- Fixed error 'inverse' is not a member of 'glm' from glm::unProject (#146)
- Fixed mismatch between some declarations and definitions
- Fixed inverse link error when using namespace glm; (#147)
- Optimized matrix inverse and division code (#149)
- Added intersectRayPlane function (#153)
- Fixed outerProduct return type (#155)

---
### GLM 0.9.5.0 - 2013-12-25
- Added forward declarations (glm/fwd.hpp) for faster compilations
- Added per feature headers
- Minimized GLM internal dependencies
- Improved Intel Compiler detection
- Added bitfieldInterleave and _mm_bit_interleave_si128 functions
- Added GTX_scalar_relational
- Added GTX_dual_quaternion
- Added rotation function to GTX_quaternion (#22)
- Added precision variation of each type
- Added quaternion comparison functions
- Fixed GTX_multiple for negative value
- Removed GTX_ocl_type extension
- Fixed post increment and decrement operators
- Fixed perspective with zNear == 0 (#71)
- Removed l-value swizzle operators
- Cleaned up compiler detection code for unsupported compilers
- Replaced C cast by C++ casts
- Fixed .length() that should return a int and not a size_t
- Added GLM_FORCE_SIZE_T_LENGTH and glm::length_t
- Removed unnecessary conversions
- Optimized packing and unpacking functions
- Removed the normalization of the up argument of lookAt function (#114)
- Added low precision specializations of inversesqrt
- Fixed ldexp and frexp implementations
- Increased assert coverage
- Increased static_assert coverage
- Replaced GLM traits by STL traits when possible
- Allowed including individual core feature
- Increased unit tests completness
- Added creating of a quaternion from two vectors
- Added C++11 initializer lists
- Fixed umulExtended and imulExtended implementations for vector types (#76)
- Fixed CUDA coverage for GTC extensions
- Added GTX_io extension
- Improved GLM messages enabled when defining GLM_MESSAGES
- Hidden matrix _inverse function implementation detail into private section

---
### [GLM 0.9.4.6](https://github.com/g-truc/glm/releases/tag/0.9.4.6) - 2013-09-20
- Fixed detection to select the last known compiler if newer version #106
- Fixed is_int and is_uint code duplication with GCC and C++11 #107 
- Fixed test suite build while using Clang in C++11 mode
- Added c++1y mode support in CMake test suite
- Removed ms extension mode to CMake when no using Visual C++
- Added pedantic mode to CMake test suite for Clang and GCC
- Added use of GCC frontend on Unix for ICC and Visual C++ fronted on Windows
  for ICC
- Added compilation errors for unsupported compiler versions
- Fixed glm::orientation with GLM_FORCE_RADIANS defined #112
- Fixed const ref issue on assignment operator taking a scalar parameter #116
- Fixed glm::eulerAngleY implementation #117

---
### GLM 0.9.4.5 - 2013-08-12
- Fixed CUDA support
- Fixed inclusion of intrinsics in "pure" mode #92
- Fixed language detection on GCC when the C++0x mode isn't enabled #95
- Fixed issue #97: register is deprecated in C++11
- Fixed issue #96: CUDA issues
- Added Windows CE detection #92
- Added missing value_ptr for quaternions #99

---
### GLM 0.9.4.4 - 2013-05-29
- Fixed slerp when costheta is close to 1 #65
- Fixed mat4x2 value_type constructor #70
- Fixed glm.natvis for Visual C++ 12 #82
- Added assert in inversesqrt to detect division by zero #61
- Fixed missing swizzle operators #86
- Fixed CUDA warnings #86
- Fixed GLM natvis for VC11 #82
- Fixed GLM_GTX_multiple with negative values #79
- Fixed glm::perspective when zNear is zero #71

---
### GLM 0.9.4.3 - 2013-03-20
- Detected qualifier for Clang
- Fixed C++11 mode for GCC, couldn't be enabled without MS extensions
- Fixed squad, intermediate and exp quaternion functions
- Fixed GTX_polar_coordinates euclidean function, takes a vec2 instead of a vec3
- Clarify the license applying on the manual
- Added a docx copy of the manual
- Fixed GLM_GTX_matrix_interpolation
- Fixed isnan and isinf on Android with Clang
- Autodetected C++ version using __cplusplus value
- Fixed mix for bool and bvec* third parameter

---
### GLM 0.9.4.2 - 2013-02-14
- Fixed compAdd from GTX_component_wise
- Fixed SIMD support for Intel compiler on Windows
- Fixed isnan and isinf for CUDA compiler
- Fixed GLM_FORCE_RADIANS on glm::perspective
- Fixed GCC warnings
- Fixed packDouble2x32 on Xcode
- Fixed mix for vec4 SSE implementation
- Fixed 0x2013 dash character in comments that cause issue in Windows 
  Japanese mode
- Fixed documentation warnings
- Fixed CUDA warnings

---
### GLM 0.9.4.1 - 2012-12-22
- Improved half support: -0.0 case and implicit conversions
- Fixed Intel Composer Compiler support on Linux
- Fixed interaction between quaternion and euler angles
- Fixed GTC_constants build
- Fixed GTX_multiple
- Fixed quat slerp using mix function when cosTheta close to 1
- Improved fvec4SIMD and fmat4x4SIMD implementations
- Fixed assert messages
- Added slerp and lerp quaternion functions and tests

---
### GLM 0.9.4.0 - 2012-11-18
- Added Intel Composer Compiler support
- Promoted GTC_espilon extension
- Promoted GTC_ulp extension
- Removed GLM website from the source repository
- Added GLM_FORCE_RADIANS so that all functions takes radians for arguments
- Fixed detection of Clang and LLVM GCC on MacOS X
- Added debugger visualizers for Visual C++ 2012
- Requires Visual Studio 2005, GCC 4.2, Clang 2.6, Cuda 3, ICC 2013 or a C++98 compiler

---
### [GLM 0.9.3.4](https://github.com/g-truc/glm/releases/tag/0.9.3.4) - 2012-06-30
- Added SSE4 and AVX2 detection.
- Removed VIRTREV_xstream and the incompatibility generated with GCC
- Fixed C++11 compiler option for GCC
- Removed MS language extension option for GCC (not fonctionnal)
- Fixed bitfieldExtract for vector types
- Fixed warnings
- Fixed SSE includes

---
### GLM 0.9.3.3 - 2012-05-10
- Fixed isinf and isnan
- Improved compatibility with Intel compiler
- Added CMake test build options: SIMD, C++11, fast math and MS land ext
- Fixed SIMD mat4 test on GCC
- Fixed perspectiveFov implementation
- Fixed matrixCompMult for none-square matrices
- Fixed namespace issue on stream operators
- Fixed various warnings
- Added VC11 support

---
### GLM 0.9.3.2 - 2012-03-15
- Fixed doxygen documentation
- Fixed Clang version detection
- Fixed simd mat4 /= operator

---
### GLM 0.9.3.1 - 2012-01-25
- Fixed platform detection
- Fixed warnings
- Removed detail code from Doxygen doc

---
### GLM 0.9.3.0 - 2012-01-09
- Added CPP Check project
- Fixed conflict with Windows headers
- Fixed isinf implementation
- Fixed Boost conflict
- Fixed warnings

---
### GLM 0.9.3.B - 2011-12-12
- Added support for Chrone Native Client
- Added epsilon constant
- Removed value_size function from vector types
- Fixed roundEven on GCC
- Improved API documentation
- Fixed modf implementation
- Fixed step function accuracy
- Fixed outerProduct

---
### GLM 0.9.3.A - 2011-11-11
- Improved doxygen documentation
- Added new swizzle operators for C++11 compilers
- Added new swizzle operators declared as functions
- Added GLSL 4.20 length for vector and matrix types
- Promoted GLM_GTC_noise extension: simplex, perlin, periodic noise functions
- Promoted GLM_GTC_random extension: linear, gaussian and various random number 
generation distribution
- Added GLM_GTX_constants: provides useful constants
- Added extension versioning
- Removed many unused namespaces
- Fixed half based type contructors
- Added GLSL core noise functions

---
### [GLM 0.9.2.7](https://github.com/g-truc/glm/releases/tag/0.9.2.7) - 2011-10-24
- Added more swizzling constructors
- Added missing none-squared matrix products

---
### [GLM 0.9.2.6](https://github.com/g-truc/glm/releases/tag/0.9.2.6) - 2011-10-01
- Fixed half based type build on old GCC
- Fixed /W4 warnings on Visual C++
- Fixed some missing l-value swizzle operators

---
### GLM 0.9.2.5 - 2011-09-20
- Fixed floatBitToXint functions
- Fixed pack and unpack functions
- Fixed round functions

---
### GLM 0.9.2.4 - 2011-09-03
- Fixed extensions bugs

---
### GLM 0.9.2.3 - 2011-06-08
- Fixed build issues

---
### GLM 0.9.2.2 - 2011-06-02
- Expend matrix constructors flexibility
- Improved quaternion implementation
- Fixed many warnings across platforms and compilers

---
### GLM 0.9.2.1 - 2011-05-24
- Automatically detect CUDA support
- Improved compiler detection
- Fixed errors and warnings in VC with C++ extensions disabled
- Fixed and tested GLM_GTX_vector_angle
- Fixed and tested GLM_GTX_rotate_vector

---
### GLM 0.9.2.0 - 2011-05-09
- Added CUDA support
- Added CTest test suite
- Added GLM_GTX_ulp extension
- Added GLM_GTX_noise extension
- Added GLM_GTX_matrix_interpolation extension
- Updated quaternion slerp interpolation

---
### [GLM 0.9.1.3](https://github.com/g-truc/glm/releases/tag/0.9.1.3) - 2011-05-07
- Fixed bugs

---
### GLM 0.9.1.2 - 2011-04-15
- Fixed bugs

---
### GLM 0.9.1.1 - 2011-03-17
- Fixed bugs

---
### GLM 0.9.1.0 - 2011-03-03
- Fixed bugs

---
### GLM 0.9.1.B - 2011-02-13
- Updated API documentation
- Improved SIMD implementation
- Fixed Linux build

---
### [GLM 0.9.0.8](https://github.com/g-truc/glm/releases/tag/0.9.0.8) - 2011-02-13
- Added quaternion product operator.
- Clarify that GLM is a header only library.

---
### GLM 0.9.1.A - 2011-01-31
- Added SIMD support
- Added new swizzle functions
- Improved static assert error message with C++0x static_assert
- New setup system
- Reduced branching
- Fixed trunc implementation

---
### [GLM 0.9.0.7](https://github.com/g-truc/glm/releases/tag/0.9.0.7) - 2011-01-30
- Added GLSL 4.10 packing functions
- Added == and != operators for every types.

---
### GLM 0.9.0.6 - 2010-12-21
- Many matrices bugs fixed

---
### GLM 0.9.0.5 - 2010-11-01
- Improved Clang support
- Fixed bugs

---
### GLM 0.9.0.4 - 2010-10-04
- Added autoexp for GLM
- Fixed bugs

---
### GLM 0.9.0.3 - 2010-08-26
- Fixed non-squared matrix operators

---
### GLM 0.9.0.2 - 2010-07-08
- Added GLM_GTX_int_10_10_10_2
- Fixed bugs

---
### GLM 0.9.0.1 - 2010-06-21
- Fixed extensions errors

---
### GLM 0.9.0.0 - 2010-05-25
- Objective-C support
- Fixed warnings
- Updated documentation

---
### GLM 0.9.B.2 - 2010-04-30
- Git transition
- Removed experimental code from releases
- Fixed bugs

---
### GLM 0.9.B.1 - 2010-04-03
- Based on GLSL 4.00 specification
- Added the new core functions
- Added some implicit conversion support

---
### GLM 0.9.A.2 - 2010-02-20
- Improved some possible errors messages
- Improved declarations and definitions match

---
### GLM 0.9.A.1 - 2010-02-09
- Removed deprecated features
- Internal redesign

---
### GLM 0.8.4.4 final - 2010-01-25
- Fixed warnings

---
### GLM 0.8.4.3 final - 2009-11-16
- Fixed Half float arithmetic
- Fixed setup defines

---
### GLM 0.8.4.2 final - 2009-10-19
- Fixed Half float adds

---
### GLM 0.8.4.1 final - 2009-10-05
- Updated documentation
- Fixed MacOS X build

---
### GLM 0.8.4.0 final - 2009-09-16
- Added GCC 4.4 and VC2010 support
- Added matrix optimizations

---
### GLM 0.8.3.5 final - 2009-08-11
- Fixed bugs

---
### GLM 0.8.3.4 final - 2009-08-10
- Updated GLM according GLSL 1.5 spec
- Fixed bugs

---
### GLM 0.8.3.3 final - 2009-06-25
- Fixed bugs

---
### GLM 0.8.3.2 final - 2009-06-04
- Added GLM_GTC_quaternion
- Added GLM_GTC_type_precision

---
### GLM 0.8.3.1 final - 2009-05-21
- Fixed old extension system.

---
### GLM 0.8.3.0 final - 2009-05-06
- Added stable extensions.
- Added new extension system.

---
### GLM 0.8.2.3 final - 2009-04-01
- Fixed bugs.

---
### GLM 0.8.2.2 final - 2009-02-24
- Fixed bugs.

---
### GLM 0.8.2.1 final - 2009-02-13
- Fixed bugs.

---
### GLM 0.8.2 final - 2009-01-21
- Fixed bugs.

---
### GLM 0.8.1 final - 2008-10-30
- Fixed bugs.

---
### GLM 0.8.0 final - 2008-10-23
- New method to use extension.

---
### GLM 0.8.0 beta3 - 2008-10-10
- Added CMake support for GLM tests.

---
### GLM 0.8.0 beta2 - 2008-10-04
- Improved half scalars and vectors support.

---
### GLM 0.8.0 beta1 - 2008-09-26
- Improved GLSL conformance
- Added GLSL 1.30 support
- Improved API documentation

---
### GLM 0.7.6 final - 2008-08-08
- Improved C++ standard comformance
- Added Static assert for types checking

---
### GLM 0.7.5 final - 2008-07-05
- Added build message system with Visual Studio
- Pedantic build with GCC

---
### GLM 0.7.4 final - 2008-06-01
- Added external dependencies system.

---
### GLM 0.7.3 final - 2008-05-24
- Fixed bugs
- Added new extension group

---
### GLM 0.7.2 final - 2008-04-27
- Updated documentation
- Added preprocessor options

---
### GLM 0.7.1 final - 2008-03-24
- Disabled half on GCC
- Fixed extensions

---
### GLM 0.7.0 final - 2008-03-22
- Changed to MIT license
- Added new documentation

---
### GLM 0.6.4 - 2007-12-10
- Fixed swizzle operators

---
### GLM 0.6.3 - 2007-11-05
- Fixed type data accesses
- Fixed 3DSMax sdk conflict

---
### GLM 0.6.2 - 2007-10-08
- Fixed extension

---
### GLM 0.6.1 - 2007-10-07
- Fixed a namespace error
- Added extensions

---
### GLM 0.6.0 : 2007-09-16
- Added new extension namespace mecanium
- Added Automatic compiler detection

---
### GLM 0.5.1 - 2007-02-19
- Fixed swizzle operators

---
### GLM 0.5.0 - 2007-01-06
- Upgrated to GLSL 1.2
- Added swizzle operators
- Added setup settings

---
### GLM 0.4.1 - 2006-05-22
- Added OpenGL examples

---
### GLM 0.4.0 - 2006-05-17
- Added missing operators to vec* and mat*
- Added first GLSL 1.2 features
- Fixed windows.h before glm.h when windows.h required

---
### GLM 0.3.2 - 2006-04-21
- Fixed texcoord components access.
- Fixed mat4 and imat4 division operators.

---
### GLM 0.3.1 - 2006-03-28
- Added GCC 4.0 support under MacOS X.
- Added GCC 4.0 and 4.1 support under Linux.
- Added code optimisations.

---
### GLM 0.3 - 2006-02-19
- Improved GLSL type conversion and construction compliance.
- Added experimental extensions.
- Added Doxygen Documentation.
- Added code optimisations.
- Fixed bugs.

---
### GLM 0.2 - 2005-05-05
- Improve adaptative from GLSL.
- Add experimental extensions based on OpenGL extension process.
- Fixe bugs.

---
### GLM 0.1 - 2005-02-21
- Add vec2, vec3, vec4 GLSL types
- Add ivec2, ivec3, ivec4 GLSL types
- Add bvec2, bvec3, bvec4 GLSL types
- Add mat2, mat3, mat4 GLSL types
- Add almost all functions

