//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
#ifndef GLSLANG_SHADERLANG_H_
#define GLSLANG_SHADERLANG_H_

#include <stddef.h>

#include "KHR/khrplatform.h"

#include <array>
#include <map>
#include <set>
#include <string>
#include <vector>

//
// This is the platform independent interface between an OGL driver
// and the shading language compiler.
//

// Note: make sure to increment ANGLE_SH_VERSION when changing ShaderVars.h
#include "ShaderVars.h"

// Version number for shader translation API.
// It is incremented every time the API changes.
#define ANGLE_SH_VERSION 217

enum ShShaderSpec
{
    SH_GLES2_SPEC,
    SH_WEBGL_SPEC,

    SH_GLES3_SPEC,
    SH_WEBGL2_SPEC,

    SH_GLES3_1_SPEC,
    SH_WEBGL3_SPEC,

    SH_GL_CORE_SPEC,
    SH_GL_COMPATIBILITY_SPEC,
};

enum ShShaderOutput
{
    // ESSL output only supported in some configurations.
    SH_ESSL_OUTPUT = 0x8B45,

    // GLSL output only supported in some configurations.
    SH_GLSL_COMPATIBILITY_OUTPUT = 0x8B46,
    // Note: GL introduced core profiles in 1.5.
    SH_GLSL_130_OUTPUT      = 0x8B47,
    SH_GLSL_140_OUTPUT      = 0x8B80,
    SH_GLSL_150_CORE_OUTPUT = 0x8B81,
    SH_GLSL_330_CORE_OUTPUT = 0x8B82,
    SH_GLSL_400_CORE_OUTPUT = 0x8B83,
    SH_GLSL_410_CORE_OUTPUT = 0x8B84,
    SH_GLSL_420_CORE_OUTPUT = 0x8B85,
    SH_GLSL_430_CORE_OUTPUT = 0x8B86,
    SH_GLSL_440_CORE_OUTPUT = 0x8B87,
    SH_GLSL_450_CORE_OUTPUT = 0x8B88,

    // Prefer using these to specify HLSL output type:
    SH_HLSL_3_0_OUTPUT       = 0x8B48,  // D3D 9
    SH_HLSL_4_1_OUTPUT       = 0x8B49,  // D3D 11
    SH_HLSL_4_0_FL9_3_OUTPUT = 0x8B4A,  // D3D 11 feature level 9_3

    // Output specialized GLSL to be fed to glslang for Vulkan SPIR.
    SH_GLSL_VULKAN_OUTPUT = 0x8B4B,

    // Output specialized GLSL to be fed to glslang for Vulkan SPIR to be cross compiled to Metal
    // later.
    SH_GLSL_METAL_OUTPUT = 0x8B4C,
};

// Compile options.
// The Compile options type is defined in ShaderVars.h, to allow ANGLE to import the ShaderVars
// header without needing the ShaderLang header. This avoids some conflicts with glslang.

const ShCompileOptions SH_VALIDATE               = 0;
const ShCompileOptions SH_VALIDATE_LOOP_INDEXING = UINT64_C(1) << 0;
const ShCompileOptions SH_INTERMEDIATE_TREE      = UINT64_C(1) << 1;
const ShCompileOptions SH_OBJECT_CODE            = UINT64_C(1) << 2;
const ShCompileOptions SH_VARIABLES              = UINT64_C(1) << 3;
const ShCompileOptions SH_LINE_DIRECTIVES        = UINT64_C(1) << 4;
const ShCompileOptions SH_SOURCE_PATH            = UINT64_C(1) << 5;

// This flag will keep invariant declaration for input in fragment shader for GLSL >=4.20 on AMD.
// From GLSL >= 4.20, it's optional to add invariant for fragment input, but GPU vendors have
// different implementations about this. Some drivers forbid invariant in fragment for GLSL>= 4.20,
// e.g. Linux Mesa, some drivers treat that as optional, e.g. NVIDIA, some drivers require invariant
// must match between vertex and fragment shader, e.g. AMD. The behavior on AMD is obviously wrong.
// Remove invariant for input in fragment shader to workaround the restriction on Intel Mesa.
// But don't remove on AMD Linux to avoid triggering the bug on AMD.
const ShCompileOptions SH_DONT_REMOVE_INVARIANT_FOR_FRAGMENT_INPUT = UINT64_C(1) << 6;

// Due to spec difference between GLSL 4.1 or lower and ESSL3, some platforms (for example, Mac OSX
// core profile) require a variable's "invariant"/"centroid" qualifiers to match between vertex and
// fragment shader. A simple solution to allow such shaders to link is to omit the two qualifiers.
// AMD driver in Linux requires invariant qualifier to match between vertex and fragment shaders,
// while ESSL3 disallows invariant qualifier in fragment shader and GLSL >= 4.2 doesn't require
// invariant qualifier to match between shaders. Remove invariant qualifier from vertex shader to
// workaround AMD driver bug.
// Note that the two flags take effect on ESSL3 input shaders translated to GLSL 4.1 or lower and to
// GLSL 4.2 or newer on Linux AMD.
// TODO(zmo): This is not a good long-term solution. Simply dropping these qualifiers may break some
// developers' content. A more complex workaround of dynamically generating, compiling, and
// re-linking shaders that use these qualifiers should be implemented.
const ShCompileOptions SH_REMOVE_INVARIANT_AND_CENTROID_FOR_ESSL3 = UINT64_C(1) << 7;

// This flag works around bug in Intel Mac drivers related to abs(i) where
// i is an integer.
const ShCompileOptions SH_EMULATE_ABS_INT_FUNCTION = UINT64_C(1) << 8;

// Enforce the GLSL 1.017 Appendix A section 7 packing restrictions.
// This flag only enforces (and can only enforce) the packing
// restrictions for uniform variables in both vertex and fragment
// shaders. ShCheckVariablesWithinPackingLimits() lets embedders
// enforce the packing restrictions for varying variables during
// program link time.
const ShCompileOptions SH_ENFORCE_PACKING_RESTRICTIONS = UINT64_C(1) << 9;

// This flag ensures all indirect (expression-based) array indexing
// is clamped to the bounds of the array. This ensures, for example,
// that you cannot read off the end of a uniform, whether an array
// vec234, or mat234 type. The ShArrayIndexClampingStrategy enum,
// specified in the ShBuiltInResources when constructing the
// compiler, selects the strategy for the clamping implementation.
const ShCompileOptions SH_CLAMP_INDIRECT_ARRAY_BOUNDS = UINT64_C(1) << 10;

// This flag limits the complexity of an expression.
const ShCompileOptions SH_LIMIT_EXPRESSION_COMPLEXITY = UINT64_C(1) << 11;

// This flag limits the depth of the call stack.
const ShCompileOptions SH_LIMIT_CALL_STACK_DEPTH = UINT64_C(1) << 12;

// This flag initializes gl_Position to vec4(0,0,0,0) at the
// beginning of the vertex shader's main(), and has no effect in the
// fragment shader. It is intended as a workaround for drivers which
// incorrectly fail to link programs if gl_Position is not written.
const ShCompileOptions SH_INIT_GL_POSITION = UINT64_C(1) << 13;

// This flag replaces
//   "a && b" with "a ? b : false",
//   "a || b" with "a ? true : b".
// This is to work around a MacOSX driver bug that |b| is executed
// independent of |a|'s value.
const ShCompileOptions SH_UNFOLD_SHORT_CIRCUIT = UINT64_C(1) << 14;

// This flag initializes output variables to 0 at the beginning of main().
// It is to avoid undefined behaviors.
const ShCompileOptions SH_INIT_OUTPUT_VARIABLES = UINT64_C(1) << 15;

// This flag scalarizes vec/ivec/bvec/mat constructor args.
// It is intended as a workaround for Linux/Mac driver bugs.
const ShCompileOptions SH_SCALARIZE_VEC_AND_MAT_CONSTRUCTOR_ARGS = UINT64_C(1) << 16;

// This flag overwrites a struct name with a unique prefix.
// It is intended as a workaround for drivers that do not handle
// struct scopes correctly, including all Mac drivers and Linux AMD.
const ShCompileOptions SH_REGENERATE_STRUCT_NAMES = UINT64_C(1) << 17;

// This flag makes the compiler not prune unused function early in the
// compilation process. Pruning coupled with SH_LIMIT_CALL_STACK_DEPTH
// helps avoid bad shaders causing stack overflows.
const ShCompileOptions SH_DONT_PRUNE_UNUSED_FUNCTIONS = UINT64_C(1) << 18;

// This flag works around a bug in NVIDIA 331 series drivers related
// to pow(x, y) where y is a constant vector.
const ShCompileOptions SH_REMOVE_POW_WITH_CONSTANT_EXPONENT = UINT64_C(1) << 19;

// This flag works around bugs in Mac drivers related to do-while by
// transforming them into an other construct.
const ShCompileOptions SH_REWRITE_DO_WHILE_LOOPS = UINT64_C(1) << 20;

// This flag works around a bug in the HLSL compiler optimizer that folds certain
// constant pow expressions incorrectly. Only applies to the HLSL back-end. It works
// by expanding the integer pow expressions into a series of multiplies.
const ShCompileOptions SH_EXPAND_SELECT_HLSL_INTEGER_POW_EXPRESSIONS = UINT64_C(1) << 21;

// Flatten "#pragma STDGL invariant(all)" into the declarations of
// varying variables and built-in GLSL variables. This compiler
// option is enabled automatically when needed.
const ShCompileOptions SH_FLATTEN_PRAGMA_STDGL_INVARIANT_ALL = UINT64_C(1) << 22;

// Some drivers do not take into account the base level of the texture in the results of the
// HLSL GetDimensions builtin.  This flag instructs the compiler to manually add the base level
// offsetting.
const ShCompileOptions SH_HLSL_GET_DIMENSIONS_IGNORES_BASE_LEVEL = UINT64_C(1) << 23;

// This flag works around an issue in translating GLSL function texelFetchOffset on
// INTEL drivers. It works by translating texelFetchOffset into texelFetch.
const ShCompileOptions SH_REWRITE_TEXELFETCHOFFSET_TO_TEXELFETCH = UINT64_C(1) << 24;

// This flag works around condition bug of for and while loops in Intel Mac OSX drivers.
// Condition calculation is not correct. Rewrite it from "CONDITION" to "CONDITION && true".
const ShCompileOptions SH_ADD_AND_TRUE_TO_LOOP_CONDITION = UINT64_C(1) << 25;

// This flag works around a bug in evaluating unary minus operator on integer on some INTEL
// drivers. It works by translating -(int) into ~(int) + 1.
const ShCompileOptions SH_REWRITE_INTEGER_UNARY_MINUS_OPERATOR = UINT64_C(1) << 26;

// This flag works around a bug in evaluating isnan() on some INTEL D3D and Mac OSX drivers.
// It works by using an expression to emulate this function.
const ShCompileOptions SH_EMULATE_ISNAN_FLOAT_FUNCTION = UINT64_C(1) << 27;

// This flag will use all uniforms of unused std140 and shared uniform blocks at the
// beginning of the vertex/fragment shader's main(). It is intended as a workaround for Mac
// drivers with shader version 4.10. In those drivers, they will treat unused
// std140 and shared uniform blocks' members as inactive. However, WebGL2.0 based on
// OpenGL ES3.0.4 requires all members of a named uniform block declared with a shared or std140
// layout qualifier to be considered active. The uniform block itself is also considered active.
const ShCompileOptions SH_USE_UNUSED_STANDARD_SHARED_BLOCKS = UINT64_C(1) << 28;

// This flag works around a bug in unary minus operator on float numbers on Intel
// Mac OSX 10.11 drivers. It works by translating -float into 0.0 - float.
const ShCompileOptions SH_REWRITE_FLOAT_UNARY_MINUS_OPERATOR = UINT64_C(1) << 29;

// This flag works around a bug in evaluating atan(y, x) on some NVIDIA OpenGL drivers.
// It works by using an expression to emulate this function.
const ShCompileOptions SH_EMULATE_ATAN2_FLOAT_FUNCTION = UINT64_C(1) << 30;

// Set to initialize uninitialized local and global temporary variables. Should only be used with
// GLSL output. In HLSL output variables are initialized regardless of if this flag is set.
const ShCompileOptions SH_INITIALIZE_UNINITIALIZED_LOCALS = UINT64_C(1) << 31;

// The flag modifies the shader in the following way:
// Every occurrence of gl_InstanceID is replaced by the global temporary variable InstanceID.
// Every occurrence of gl_ViewID_OVR is replaced by the varying variable ViewID_OVR.
// At the beginning of the body of main() in a vertex shader the following initializers are added:
// ViewID_OVR = uint(gl_InstanceID) % num_views;
// InstanceID = gl_InstanceID / num_views;
// ViewID_OVR is added as a varying variable to both the vertex and fragment shaders.
const ShCompileOptions SH_INITIALIZE_BUILTINS_FOR_INSTANCED_MULTIVIEW = UINT64_C(1) << 32;

// With the flag enabled the GLSL/ESSL vertex shader is modified to include code for viewport
// selection in the following way:
// - Code to enable the extension ARB_shader_viewport_layer_array/NV_viewport_array2 is included.
// - Code to select the viewport index or layer is inserted at the beginning of main after
// ViewID_OVR's initialization.
// - A declaration of the uniform multiviewBaseViewLayerIndex.
// Note: The SH_INITIALIZE_BUILTINS_FOR_INSTANCED_MULTIVIEW flag also has to be enabled to have the
// temporary variable ViewID_OVR declared and initialized.
const ShCompileOptions SH_SELECT_VIEW_IN_NV_GLSL_VERTEX_SHADER = UINT64_C(1) << 33;

// If the flag is enabled, gl_PointSize is clamped to the maximum point size specified in
// ShBuiltInResources in vertex shaders.
const ShCompileOptions SH_CLAMP_POINT_SIZE = UINT64_C(1) << 34;

// Turn some arithmetic operations that operate on a float vector-scalar pair into vector-vector
// operations. This is done recursively. Some scalar binary operations inside vector constructors
// are also turned into vector operations.
//
// This is targeted to work around a bug in NVIDIA OpenGL drivers that was reproducible on NVIDIA
// driver version 387.92. It works around the most common occurrences of the bug.
const ShCompileOptions SH_REWRITE_VECTOR_SCALAR_ARITHMETIC = UINT64_C(1) << 35;

// Don't use loops to initialize uninitialized variables. Only has an effect if some kind of
// variable initialization is turned on.
const ShCompileOptions SH_DONT_USE_LOOPS_TO_INITIALIZE_VARIABLES = UINT64_C(1) << 36;

// Don't use D3D constant register zero when allocating space for uniforms. This is targeted to work
// around a bug in NVIDIA D3D driver version 388.59 where in very specific cases the driver would
// not handle constant register zero correctly. Only has an effect on HLSL translation.
const ShCompileOptions SH_SKIP_D3D_CONSTANT_REGISTER_ZERO = UINT64_C(1) << 37;

// Clamp gl_FragDepth to the range [0.0, 1.0] in case it is statically used.
const ShCompileOptions SH_CLAMP_FRAG_DEPTH = UINT64_C(1) << 38;

// Rewrite expressions like "v.x = z = expression;". Works around a bug in NVIDIA OpenGL drivers
// prior to version 397.31.
const ShCompileOptions SH_REWRITE_REPEATED_ASSIGN_TO_SWIZZLED = UINT64_C(1) << 39;

// Rewrite gl_DrawID as a uniform int
const ShCompileOptions SH_EMULATE_GL_DRAW_ID = UINT64_C(1) << 40;

// This flag initializes shared variables to 0.
// It is to avoid ompute shaders being able to read undefined values that could be coming from
// another webpage/application.
const ShCompileOptions SH_INIT_SHARED_VARIABLES = UINT64_C(1) << 41;

// Forces the value returned from an atomic operations to be always be resolved. This is targeted to
// workaround a bug in NVIDIA D3D driver where the return value from
// RWByteAddressBuffer.InterlockedAdd does not get resolved when used in the .yzw components of a
// RWByteAddressBuffer.Store operation. Only has an effect on HLSL translation.
// http://anglebug.com/3246
const ShCompileOptions SH_FORCE_ATOMIC_VALUE_RESOLUTION = UINT64_C(1) << 42;

// Rewrite gl_BaseVertex and gl_BaseInstance as uniform int
const ShCompileOptions SH_EMULATE_GL_BASE_VERTEX_BASE_INSTANCE = UINT64_C(1) << 43;

// Emulate seamful cube map sampling for OpenGL ES2.0.  Currently only applies to the Vulkan
// backend, as is done after samplers are moved out of structs.  Can likely be made to work on
// the other backends as well.
const ShCompileOptions SH_EMULATE_SEAMFUL_CUBE_MAP_SAMPLING = UINT64_C(1) << 44;

// If requested, validates the AST after every transformation.  Useful for debugging.
const ShCompileOptions SH_VALIDATE_AST = UINT64_C(1) << 46;

// Use old version of RewriteStructSamplers, which doesn't produce as many
// sampler arrays in parameters. This causes a few tests to pass on Android.
const ShCompileOptions SH_USE_OLD_REWRITE_STRUCT_SAMPLERS = UINT64_C(1) << 47;

// This flag works around a inconsistent behavior in Mac AMD driver where gl_VertexID doesn't
// include base vertex value. It replaces gl_VertexID with (gl_VertexID + angle_BaseVertex)
// when angle_BaseVertex is available.
const ShCompileOptions SH_ADD_BASE_VERTEX_TO_VERTEX_ID = UINT64_C(1) << 48;

// This flag adds linear depth range mapping to shader on Metal back-end. It is used when Metal's
// auto MTLViewport's depth range mapping on old GPUs is simply a clamp instead of linear map. Thus,
// an emulation in shader is needed.
const ShCompileOptions SH_METAL_EMULATE_LINEAR_DEPTH_RANGE_MAP = UINT64_C(1) << 49;

// Defines alternate strategies for implementing array index clamping.
enum ShArrayIndexClampingStrategy
{
    // Use the clamp intrinsic for array index clamping.
    SH_CLAMP_WITH_CLAMP_INTRINSIC = 1,

    // Use a user-defined function for array index clamping.
    SH_CLAMP_WITH_USER_DEFINED_INT_CLAMP_FUNCTION
};

// The 64 bits hash function. The first parameter is the input string; the
// second parameter is the string length.
using ShHashFunction64 = khronos_uint64_t (*)(const char *, size_t);

//
// Implementation dependent built-in resources (constants and extensions).
// The names for these resources has been obtained by stripping gl_/GL_.
//
struct ShBuiltInResources
{
    // Constants.
    int MaxVertexAttribs;
    int MaxVertexUniformVectors;
    int MaxVaryingVectors;
    int MaxVertexTextureImageUnits;
    int MaxCombinedTextureImageUnits;
    int MaxTextureImageUnits;
    int MaxFragmentUniformVectors;
    int MaxDrawBuffers;

    // Extensions.
    // Set to 1 to enable the extension, else 0.
    int OES_standard_derivatives;
    int OES_EGL_image_external;
    int OES_EGL_image_external_essl3;
    int NV_EGL_stream_consumer_external;
    int ARB_texture_rectangle;
    int EXT_blend_func_extended;
    int EXT_draw_buffers;
    int EXT_frag_depth;
    int EXT_shader_texture_lod;
    int WEBGL_debug_shader_precision;
    int EXT_shader_framebuffer_fetch;
    int NV_shader_framebuffer_fetch;
    int ARM_shader_framebuffer_fetch;
    int OVR_multiview;
    int OVR_multiview2;
    int EXT_multisampled_render_to_texture;
    int EXT_YUV_target;
    int EXT_geometry_shader;
    int OES_texture_storage_multisample_2d_array;
    int OES_texture_3D;
    int ANGLE_texture_multisample;
    int ANGLE_multi_draw;
    int ANGLE_base_vertex_base_instance;
    int APPLE_clip_distance;

    // Set to 1 to enable replacing GL_EXT_draw_buffers #extension directives
    // with GL_NV_draw_buffers in ESSL output. This flag can be used to emulate
    // EXT_draw_buffers by using it in combination with GLES3.0 glDrawBuffers
    // function. This applies to Tegra K1 devices.
    int NV_draw_buffers;

    // Set to 1 if highp precision is supported in the ESSL 1.00 version of the
    // fragment language. Does not affect versions of the language where highp
    // support is mandatory.
    // Default is 0.
    int FragmentPrecisionHigh;

    // GLSL ES 3.0 constants.
    int MaxVertexOutputVectors;
    int MaxFragmentInputVectors;
    int MinProgramTexelOffset;
    int MaxProgramTexelOffset;

    // Extension constants.

    // Value of GL_MAX_DUAL_SOURCE_DRAW_BUFFERS_EXT for OpenGL ES output context.
    // Value of GL_MAX_DUAL_SOURCE_DRAW_BUFFERS for OpenGL output context.
    // GLES SL version 100 gl_MaxDualSourceDrawBuffersEXT value for EXT_blend_func_extended.
    int MaxDualSourceDrawBuffers;

    // Value of GL_MAX_VIEWS_OVR.
    int MaxViewsOVR;

    // Name Hashing.
    // Set a 64 bit hash function to enable user-defined name hashing.
    // Default is NULL.
    ShHashFunction64 HashFunction;

    // Selects a strategy to use when implementing array index clamping.
    // Default is SH_CLAMP_WITH_CLAMP_INTRINSIC.
    ShArrayIndexClampingStrategy ArrayIndexClampingStrategy;

    // The maximum complexity an expression can be when SH_LIMIT_EXPRESSION_COMPLEXITY is turned on.
    int MaxExpressionComplexity;

    // The maximum depth a call stack can be.
    int MaxCallStackDepth;

    // The maximum number of parameters a function can have when SH_LIMIT_EXPRESSION_COMPLEXITY is
    // turned on.
    int MaxFunctionParameters;

    // GLES 3.1 constants

    // texture gather offset constraints.
    int MinProgramTextureGatherOffset;
    int MaxProgramTextureGatherOffset;

    // maximum number of available image units
    int MaxImageUnits;

    // maximum number of image uniforms in a vertex shader
    int MaxVertexImageUniforms;

    // maximum number of image uniforms in a fragment shader
    int MaxFragmentImageUniforms;

    // maximum number of image uniforms in a compute shader
    int MaxComputeImageUniforms;

    // maximum total number of image uniforms in a program
    int MaxCombinedImageUniforms;

    // maximum number of uniform locations
    int MaxUniformLocations;

    // maximum number of ssbos and images in a shader
    int MaxCombinedShaderOutputResources;

    // maximum number of groups in each dimension
    std::array<int, 3> MaxComputeWorkGroupCount;
    // maximum number of threads per work group in each dimension
    std::array<int, 3> MaxComputeWorkGroupSize;

    // maximum number of total uniform components
    int MaxComputeUniformComponents;

    // maximum number of texture image units in a compute shader
    int MaxComputeTextureImageUnits;

    // maximum number of atomic counters in a compute shader
    int MaxComputeAtomicCounters;

    // maximum number of atomic counter buffers in a compute shader
    int MaxComputeAtomicCounterBuffers;

    // maximum number of atomic counters in a vertex shader
    int MaxVertexAtomicCounters;

    // maximum number of atomic counters in a fragment shader
    int MaxFragmentAtomicCounters;

    // maximum number of atomic counters in a program
    int MaxCombinedAtomicCounters;

    // maximum binding for an atomic counter
    int MaxAtomicCounterBindings;

    // maximum number of atomic counter buffers in a vertex shader
    int MaxVertexAtomicCounterBuffers;

    // maximum number of atomic counter buffers in a fragment shader
    int MaxFragmentAtomicCounterBuffers;

    // maximum number of atomic counter buffers in a program
    int MaxCombinedAtomicCounterBuffers;

    // maximum number of buffer object storage in machine units
    int MaxAtomicCounterBufferSize;

    // maximum number of uniform block bindings
    int MaxUniformBufferBindings;

    // maximum number of shader storage buffer bindings
    int MaxShaderStorageBufferBindings;

    // maximum point size (higher limit from ALIASED_POINT_SIZE_RANGE)
    float MaxPointSize;

    // EXT_geometry_shader constants
    int MaxGeometryUniformComponents;
    int MaxGeometryUniformBlocks;
    int MaxGeometryInputComponents;
    int MaxGeometryOutputComponents;
    int MaxGeometryOutputVertices;
    int MaxGeometryTotalOutputComponents;
    int MaxGeometryTextureImageUnits;
    int MaxGeometryAtomicCounterBuffers;
    int MaxGeometryAtomicCounters;
    int MaxGeometryShaderStorageBlocks;
    int MaxGeometryShaderInvocations;
    int MaxGeometryImageUniforms;

    // APPLE_clip_distance/EXT_clip_cull_distance constant
    int MaxClipDistances;
};

//
// ShHandle held by but opaque to the driver.  It is allocated,
// managed, and de-allocated by the compiler. Its contents
// are defined by and used by the compiler.
//
// If handle creation fails, 0 will be returned.
//
using ShHandle = void *;

namespace sh
{

//
// Driver must call this first, once, before doing any other compiler operations.
// If the function succeeds, the return value is true, else false.
//
bool Initialize();
//
// Driver should call this at shutdown.
// If the function succeeds, the return value is true, else false.
//
bool Finalize();

//
// Initialize built-in resources with minimum expected values.
// Parameters:
// resources: The object to initialize. Will be comparable with memcmp.
//
void InitBuiltInResources(ShBuiltInResources *resources);

//
// Returns the a concatenated list of the items in ShBuiltInResources as a null-terminated string.
// This function must be updated whenever ShBuiltInResources is changed.
// Parameters:
// handle: Specifies the handle of the compiler to be used.
const std::string &GetBuiltInResourcesString(const ShHandle handle);

//
// Driver calls these to create and destroy compiler objects.
//
// Returns the handle of constructed compiler, null if the requested compiler is not supported.
// Parameters:
// type: Specifies the type of shader - GL_FRAGMENT_SHADER or GL_VERTEX_SHADER.
// spec: Specifies the language spec the compiler must conform to - SH_GLES2_SPEC or SH_WEBGL_SPEC.
// output: Specifies the output code type - for example SH_ESSL_OUTPUT, SH_GLSL_OUTPUT,
//         SH_HLSL_3_0_OUTPUT or SH_HLSL_4_1_OUTPUT. Note: Each output type may only
//         be supported in some configurations.
// resources: Specifies the built-in resources.
ShHandle ConstructCompiler(sh::GLenum type,
                           ShShaderSpec spec,
                           ShShaderOutput output,
                           const ShBuiltInResources *resources);
void Destruct(ShHandle handle);

//
// Compiles the given shader source.
// If the function succeeds, the return value is true, else false.
// Parameters:
// handle: Specifies the handle of compiler to be used.
// shaderStrings: Specifies an array of pointers to null-terminated strings containing the shader
// source code.
// numStrings: Specifies the number of elements in shaderStrings array.
// compileOptions: A mask containing the following parameters:
// SH_VALIDATE: Validates shader to ensure that it conforms to the spec
//              specified during compiler construction.
// SH_VALIDATE_LOOP_INDEXING: Validates loop and indexing in the shader to
//                            ensure that they do not exceed the minimum
//                            functionality mandated in GLSL 1.0 spec,
//                            Appendix A, Section 4 and 5.
//                            There is no need to specify this parameter when
//                            compiling for WebGL - it is implied.
// SH_INTERMEDIATE_TREE: Writes intermediate tree to info log.
//                       Can be queried by calling sh::GetInfoLog().
// SH_OBJECT_CODE: Translates intermediate tree to glsl or hlsl shader.
//                 Can be queried by calling sh::GetObjectCode().
// SH_VARIABLES: Extracts attributes, uniforms, and varyings.
//               Can be queried by calling ShGetVariableInfo().
//
bool Compile(const ShHandle handle,
             const char *const shaderStrings[],
             size_t numStrings,
             ShCompileOptions compileOptions);

// Clears the results from the previous compilation.
void ClearResults(const ShHandle handle);

// Return the version of the shader language.
int GetShaderVersion(const ShHandle handle);

// Return the currently set language output type.
ShShaderOutput GetShaderOutputType(const ShHandle handle);

// Returns null-terminated information log for a compiled shader.
// Parameters:
// handle: Specifies the compiler
const std::string &GetInfoLog(const ShHandle handle);

// Returns null-terminated object code for a compiled shader.
// Parameters:
// handle: Specifies the compiler
const std::string &GetObjectCode(const ShHandle handle);

// Returns a (original_name, hash) map containing all the user defined names in the shader,
// including variable names, function names, struct names, and struct field names.
// Parameters:
// handle: Specifies the compiler
const std::map<std::string, std::string> *GetNameHashingMap(const ShHandle handle);

// Shader variable inspection.
// Returns a pointer to a list of variables of the designated type.
// (See ShaderVars.h for type definitions, included above)
// Returns NULL on failure.
// Parameters:
// handle: Specifies the compiler
const std::vector<sh::ShaderVariable> *GetUniforms(const ShHandle handle);
const std::vector<sh::ShaderVariable> *GetVaryings(const ShHandle handle);
const std::vector<sh::ShaderVariable> *GetInputVaryings(const ShHandle handle);
const std::vector<sh::ShaderVariable> *GetOutputVaryings(const ShHandle handle);
const std::vector<sh::ShaderVariable> *GetAttributes(const ShHandle handle);
const std::vector<sh::ShaderVariable> *GetOutputVariables(const ShHandle handle);
const std::vector<sh::InterfaceBlock> *GetInterfaceBlocks(const ShHandle handle);
const std::vector<sh::InterfaceBlock> *GetUniformBlocks(const ShHandle handle);
const std::vector<sh::InterfaceBlock> *GetShaderStorageBlocks(const ShHandle handle);
sh::WorkGroupSize GetComputeShaderLocalGroupSize(const ShHandle handle);
// Returns the number of views specified through the num_views layout qualifier. If num_views is
// not set, the function returns -1.
int GetVertexShaderNumViews(const ShHandle handle);

// Returns true if the passed in variables pack in maxVectors followingthe packing rules from the
// GLSL 1.017 spec, Appendix A, section 7.
// Returns false otherwise. Also look at the SH_ENFORCE_PACKING_RESTRICTIONS
// flag above.
// Parameters:
// maxVectors: the available rows of registers.
// variables: an array of variables.
bool CheckVariablesWithinPackingLimits(int maxVectors,
                                       const std::vector<sh::ShaderVariable> &variables);

// Gives the compiler-assigned register for a shader storage block.
// The method writes the value to the output variable "indexOut".
// Returns true if it found a valid shader storage block, false otherwise.
// Parameters:
// handle: Specifies the compiler
// shaderStorageBlockName: Specifies the shader storage block
// indexOut: output variable that stores the assigned register
bool GetShaderStorageBlockRegister(const ShHandle handle,
                                   const std::string &shaderStorageBlockName,
                                   unsigned int *indexOut);

// Gives the compiler-assigned register for a uniform block.
// The method writes the value to the output variable "indexOut".
// Returns true if it found a valid uniform block, false otherwise.
// Parameters:
// handle: Specifies the compiler
// uniformBlockName: Specifies the uniform block
// indexOut: output variable that stores the assigned register
bool GetUniformBlockRegister(const ShHandle handle,
                             const std::string &uniformBlockName,
                             unsigned int *indexOut);

// Gives a map from uniform names to compiler-assigned registers in the default uniform block.
// Note that the map contains also registers of samplers that have been extracted from structs.
const std::map<std::string, unsigned int> *GetUniformRegisterMap(const ShHandle handle);

// Sampler, image and atomic counters share registers(t type and u type),
// GetReadonlyImage2DRegisterIndex and GetImage2DRegisterIndex return the first index into
// a range of reserved registers for image2D/iimage2D/uimage2D variables.
// Parameters: handle: Specifies the compiler
unsigned int GetReadonlyImage2DRegisterIndex(const ShHandle handle);
unsigned int GetImage2DRegisterIndex(const ShHandle handle);

// The method records these used function names related with image2D/iimage2D/uimage2D, these
// functions will be dynamically generated.
// Parameters:
// handle: Specifies the compiler
const std::set<std::string> *GetUsedImage2DFunctionNames(const ShHandle handle);

bool HasValidGeometryShaderInputPrimitiveType(const ShHandle handle);
bool HasValidGeometryShaderOutputPrimitiveType(const ShHandle handle);
bool HasValidGeometryShaderMaxVertices(const ShHandle handle);
GLenum GetGeometryShaderInputPrimitiveType(const ShHandle handle);
GLenum GetGeometryShaderOutputPrimitiveType(const ShHandle handle);
int GetGeometryShaderInvocations(const ShHandle handle);
int GetGeometryShaderMaxVertices(const ShHandle handle);

//
// Helper function to identify specs that are based on the WebGL spec.
//
inline bool IsWebGLBasedSpec(ShShaderSpec spec)
{
    return (spec == SH_WEBGL_SPEC || spec == SH_WEBGL2_SPEC || spec == SH_WEBGL3_SPEC);
}

//
// Helper function to identify DesktopGL specs
//
inline bool IsDesktopGLSpec(ShShaderSpec spec)
{
    return spec == SH_GL_CORE_SPEC || spec == SH_GL_COMPATIBILITY_SPEC;
}
}  // namespace sh

#endif  // GLSLANG_SHADERLANG_H_
