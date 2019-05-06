/*
 * Copyright 2019 Hans-Kristian Arntzen
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef SPIRV_CROSS_C_API_H
#define SPIRV_CROSS_C_API_H

#include <stddef.h>
#include "spirv.h"

/*
 * C89-compatible wrapper for SPIRV-Cross' API.
 * Documentation here is sparse unless the behavior does not map 1:1 with C++ API.
 * It is recommended to look at the canonical C++ API for more detailed information.
 */

#ifdef __cplusplus
extern "C" {
#endif

/* Bumped if ABI or API breaks backwards compatibility. */
#define SPVC_C_API_VERSION_MAJOR 0
/* Bumped if APIs or enumerations are added in a backwards compatible way. */
#define SPVC_C_API_VERSION_MINOR 9
/* Bumped if internal implementation details change. */
#define SPVC_C_API_VERSION_PATCH 0

#if !defined(SPVC_PUBLIC_API)
#if defined(SPVC_EXPORT_SYMBOLS)
/* Exports symbols. Standard C calling convention is used. */
#if defined(__GNUC__)
#define SPVC_PUBLIC_API __attribute__((visibility("default")))
#elif defined(_MSC_VER)
#define SPVC_PUBLIC_API __declspec(dllexport)
#else
#define SPVC_PUBLIC_API
#endif
#else
#define SPVC_PUBLIC_API
#endif
#endif

/*
 * Gets the SPVC_C_API_VERSION_* used to build this library.
 * Can be used to check for ABI mismatch if so-versioning did not catch it.
 */
SPVC_PUBLIC_API void spvc_get_version(unsigned *major, unsigned *minor, unsigned *patch);

/* These types are opaque to the user. */
typedef struct spvc_context_s *spvc_context;
typedef struct spvc_parsed_ir_s *spvc_parsed_ir;
typedef struct spvc_compiler_s *spvc_compiler;
typedef struct spvc_compiler_options_s *spvc_compiler_options;
typedef struct spvc_resources_s *spvc_resources;
struct spvc_type_s;
typedef const struct spvc_type_s *spvc_type;
typedef struct spvc_constant_s *spvc_constant;
struct spvc_set_s;
typedef const struct spvc_set_s *spvc_set;

/*
 * Shallow typedefs. All SPIR-V IDs are plain 32-bit numbers, but this helps communicate which data is used.
 * Maps to a SPIRType.
 */
typedef SpvId spvc_type_id;
/* Maps to a SPIRVariable. */
typedef SpvId spvc_variable_id;
/* Maps to a SPIRConstant. */
typedef SpvId spvc_constant_id;

/* See C++ API. */
typedef struct spvc_reflected_resource
{
	spvc_variable_id id;
	spvc_type_id base_type_id;
	spvc_type_id type_id;
	const char *name;
} spvc_reflected_resource;

/* See C++ API. */
typedef struct spvc_entry_point
{
	SpvExecutionModel execution_model;
	const char *name;
} spvc_entry_point;

/* See C++ API. */
typedef struct spvc_combined_image_sampler
{
	spvc_variable_id combined_id;
	spvc_variable_id image_id;
	spvc_variable_id sampler_id;
} spvc_combined_image_sampler;

/* See C++ API. */
typedef struct spvc_specialization_constant
{
	spvc_constant_id id;
	unsigned constant_id;
} spvc_specialization_constant;

/* See C++ API. */
typedef struct spvc_buffer_range
{
	unsigned index;
	size_t offset;
	size_t range;
} spvc_buffer_range;

/* See C++ API. */
typedef struct spvc_hlsl_root_constants
{
	unsigned start;
	unsigned end;
	unsigned binding;
	unsigned space;
} spvc_hlsl_root_constants;

/* See C++ API. */
typedef struct spvc_hlsl_vertex_attribute_remap
{
	unsigned location;
	const char *semantic;
} spvc_hlsl_vertex_attribute_remap;

/*
 * Be compatible with non-C99 compilers, which do not have stdbool.
 * Only recent MSVC compilers supports this for example, and ideally SPIRV-Cross should be linkable
 * from a wide range of compilers in its C wrapper.
 */
typedef unsigned char spvc_bool;
#define SPVC_TRUE ((spvc_bool)1)
#define SPVC_FALSE ((spvc_bool)0)

typedef enum spvc_result
{
	/* Success. */
	SPVC_SUCCESS = 0,

	/* The SPIR-V is invalid. Should have been caught by validation ideally. */
	SPVC_ERROR_INVALID_SPIRV = -1,

	/* The SPIR-V might be valid or invalid, but SPIRV-Cross currently cannot correctly translate this to your target language. */
	SPVC_ERROR_UNSUPPORTED_SPIRV = -2,

	/* If for some reason we hit this, new or malloc failed. */
	SPVC_ERROR_OUT_OF_MEMORY = -3,

	/* Invalid API argument. */
	SPVC_ERROR_INVALID_ARGUMENT = -4,

	SPVC_ERROR_INT_MAX = 0x7fffffff
} spvc_result;

typedef enum spvc_capture_mode
{
	/* The Parsed IR payload will be copied, and the handle can be reused to create other compiler instances. */
	SPVC_CAPTURE_MODE_COPY = 0,

	/*
	 * The payload will now be owned by the compiler.
	 * parsed_ir should now be considered a dead blob and must not be used further.
	 * This is optimal for performance and should be the go-to option.
	 */
	SPVC_CAPTURE_MODE_TAKE_OWNERSHIP = 1,

	SPVC_CAPTURE_MODE_INT_MAX = 0x7fffffff
} spvc_capture_mode;

typedef enum spvc_backend
{
	/* This backend can only perform reflection, no compiler options are supported. Maps to spirv_cross::Compiler. */
	SPVC_BACKEND_NONE = 0,
	SPVC_BACKEND_GLSL = 1, /* spirv_cross::CompilerGLSL */
	SPVC_BACKEND_HLSL = 2, /* CompilerHLSL */
	SPVC_BACKEND_MSL = 3, /* CompilerMSL */
	SPVC_BACKEND_CPP = 4, /* CompilerCPP */
	SPVC_BACKEND_JSON = 5, /* CompilerReflection w/ JSON backend */
	SPVC_BACKEND_INT_MAX = 0x7fffffff
} spvc_backend;

/* Maps to C++ API. */
typedef enum spvc_resource_type
{
	SPVC_RESOURCE_TYPE_UNKNOWN = 0,
	SPVC_RESOURCE_TYPE_UNIFORM_BUFFER = 1,
	SPVC_RESOURCE_TYPE_STORAGE_BUFFER = 2,
	SPVC_RESOURCE_TYPE_STAGE_INPUT = 3,
	SPVC_RESOURCE_TYPE_STAGE_OUTPUT = 4,
	SPVC_RESOURCE_TYPE_SUBPASS_INPUT = 5,
	SPVC_RESOURCE_TYPE_STORAGE_IMAGE = 6,
	SPVC_RESOURCE_TYPE_SAMPLED_IMAGE = 7,
	SPVC_RESOURCE_TYPE_ATOMIC_COUNTER = 8,
	SPVC_RESOURCE_TYPE_PUSH_CONSTANT = 9,
	SPVC_RESOURCE_TYPE_SEPARATE_IMAGE = 10,
	SPVC_RESOURCE_TYPE_SEPARATE_SAMPLERS = 11,
	SPVC_RESOURCE_TYPE_ACCELERATION_STRUCTURE = 12,
	SPVC_RESOURCE_TYPE_INT_MAX = 0x7fffffff
} spvc_resource_type;

/* Maps to spirv_cross::SPIRType::BaseType. */
typedef enum spvc_basetype
{
	SPVC_BASETYPE_UNKNOWN = 0,
	SPVC_BASETYPE_VOID = 1,
	SPVC_BASETYPE_BOOLEAN = 2,
	SPVC_BASETYPE_INT8 = 3,
	SPVC_BASETYPE_UINT8 = 4,
	SPVC_BASETYPE_INT16 = 5,
	SPVC_BASETYPE_UINT16 = 6,
	SPVC_BASETYPE_INT32 = 7,
	SPVC_BASETYPE_UINT32 = 8,
	SPVC_BASETYPE_INT64 = 9,
	SPVC_BASETYPE_UINT64 = 10,
	SPVC_BASETYPE_ATOMIC_COUNTER = 11,
	SPVC_BASETYPE_FP16 = 12,
	SPVC_BASETYPE_FP32 = 13,
	SPVC_BASETYPE_FP64 = 14,
	SPVC_BASETYPE_STRUCT = 15,
	SPVC_BASETYPE_IMAGE = 16,
	SPVC_BASETYPE_SAMPLED_IMAGE = 17,
	SPVC_BASETYPE_SAMPLER = 18,
	SPVC_BASETYPE_ACCELERATION_STRUCTURE = 19,

	SPVC_BASETYPE_INT_MAX = 0x7fffffff
} spvc_basetype;

#define SPVC_COMPILER_OPTION_COMMON_BIT 0x1000000
#define SPVC_COMPILER_OPTION_GLSL_BIT 0x2000000
#define SPVC_COMPILER_OPTION_HLSL_BIT 0x4000000
#define SPVC_COMPILER_OPTION_MSL_BIT 0x8000000
#define SPVC_COMPILER_OPTION_LANG_BITS 0x0f000000
#define SPVC_COMPILER_OPTION_ENUM_BITS 0xffffff

#define SPVC_MAKE_MSL_VERSION(major, minor, patch) ((major) * 10000 + (minor) * 100 + (patch))

/* Maps to C++ API. */
typedef enum spvc_msl_platform
{
	SPVC_MSL_PLATFORM_IOS = 0,
	SPVC_MSL_PLATFORM_MACOS = 1,
	SPVC_MSL_PLATFORM_MAX_INT = 0x7fffffff
} spvc_msl_platform;

/* Maps to C++ API. */
typedef enum spvc_msl_vertex_format
{
	SPVC_MSL_VERTEX_FORMAT_OTHER = 0,
	SPVC_MSL_VERTEX_FORMAT_UINT8 = 1,
	SPVC_MSL_VERTEX_FORMAT_UINT16 = 2
} spvc_msl_vertex_format;

/* Maps to C++ API. */
typedef struct spvc_msl_vertex_attribute
{
	unsigned location;
	unsigned msl_buffer;
	unsigned msl_offset;
	unsigned msl_stride;
	spvc_bool per_instance;
	spvc_msl_vertex_format format;
	SpvBuiltIn builtin;
} spvc_msl_vertex_attribute;

/*
 * Initializes the vertex attribute struct.
 */
SPVC_PUBLIC_API void spvc_msl_vertex_attribute_init(spvc_msl_vertex_attribute *attr);

/* Maps to C++ API. */
typedef struct spvc_msl_resource_binding
{
	SpvExecutionModel stage;
	unsigned desc_set;
	unsigned binding;
	unsigned msl_buffer;
	unsigned msl_texture;
	unsigned msl_sampler;
} spvc_msl_resource_binding;

/*
 * Initializes the resource binding struct.
 * The defaults are non-zero.
 */
SPVC_PUBLIC_API void spvc_msl_resource_binding_init(spvc_msl_resource_binding *binding);

#define SPVC_MSL_PUSH_CONSTANT_DESC_SET (~(0u))
#define SPVC_MSL_PUSH_CONSTANT_BINDING (0)
#define SPVC_MSL_SWIZZLE_BUFFER_BINDING (~(1u))

/* Obsolete. Sticks around for backwards compatibility. */
#define SPVC_MSL_AUX_BUFFER_STRUCT_VERSION 1

/* Runtime check for incompatibility. Obsolete. */
SPVC_PUBLIC_API unsigned spvc_msl_get_aux_buffer_struct_version(void);

/* Maps to C++ API. */
typedef enum spvc_msl_sampler_coord
{
	SPVC_MSL_SAMPLER_COORD_NORMALIZED = 0,
	SPVC_MSL_SAMPLER_COORD_PIXEL = 1,
	SPVC_MSL_SAMPLER_INT_MAX = 0x7fffffff
} spvc_msl_sampler_coord;

/* Maps to C++ API. */
typedef enum spvc_msl_sampler_filter
{
	SPVC_MSL_SAMPLER_FILTER_NEAREST = 0,
	SPVC_MSL_SAMPLER_FILTER_LINEAR = 1,
	SPVC_MSL_SAMPLER_FILTER_INT_MAX = 0x7fffffff
} spvc_msl_sampler_filter;

/* Maps to C++ API. */
typedef enum spvc_msl_sampler_mip_filter
{
	SPVC_MSL_SAMPLER_MIP_FILTER_NONE = 0,
	SPVC_MSL_SAMPLER_MIP_FILTER_NEAREST = 1,
	SPVC_MSL_SAMPLER_MIP_FILTER_LINEAR = 2,
	SPVC_MSL_SAMPLER_MIP_FILTER_INT_MAX = 0x7fffffff
} spvc_msl_sampler_mip_filter;

/* Maps to C++ API. */
typedef enum spvc_msl_sampler_address
{
	SPVC_MSL_SAMPLER_ADDRESS_CLAMP_TO_ZERO = 0,
	SPVC_MSL_SAMPLER_ADDRESS_CLAMP_TO_EDGE = 1,
	SPVC_MSL_SAMPLER_ADDRESS_CLAMP_TO_BORDER = 2,
	SPVC_MSL_SAMPLER_ADDRESS_REPEAT = 3,
	SPVC_MSL_SAMPLER_ADDRESS_MIRRORED_REPEAT = 4,
	SPVC_MSL_SAMPLER_ADDRESS_INT_MAX = 0x7fffffff
} spvc_msl_sampler_address;

/* Maps to C++ API. */
typedef enum spvc_msl_sampler_compare_func
{
	SPVC_MSL_SAMPLER_COMPARE_FUNC_NEVER = 0,
	SPVC_MSL_SAMPLER_COMPARE_FUNC_LESS = 1,
	SPVC_MSL_SAMPLER_COMPARE_FUNC_LESS_EQUAL = 2,
	SPVC_MSL_SAMPLER_COMPARE_FUNC_GREATER = 3,
	SPVC_MSL_SAMPLER_COMPARE_FUNC_GREATER_EQUAL = 4,
	SPVC_MSL_SAMPLER_COMPARE_FUNC_EQUAL = 5,
	SPVC_MSL_SAMPLER_COMPARE_FUNC_NOT_EQUAL = 6,
	SPVC_MSL_SAMPLER_COMPARE_FUNC_ALWAYS = 7,
	SPVC_MSL_SAMPLER_COMPARE_FUNC_INT_MAX = 0x7fffffff
} spvc_msl_sampler_compare_func;

/* Maps to C++ API. */
typedef enum spvc_msl_sampler_border_color
{
	SPVC_MSL_SAMPLER_BORDER_COLOR_TRANSPARENT_BLACK = 0,
	SPVC_MSL_SAMPLER_BORDER_COLOR_OPAQUE_BLACK = 1,
	SPVC_MSL_SAMPLER_BORDER_COLOR_OPAQUE_WHITE = 2,
	SPVC_MSL_SAMPLER_BORDER_COLOR_INT_MAX = 0x7fffffff
} spvc_msl_sampler_border_color;

/* Maps to C++ API. */
typedef struct spvc_msl_constexpr_sampler
{
	spvc_msl_sampler_coord coord;
	spvc_msl_sampler_filter min_filter;
	spvc_msl_sampler_filter mag_filter;
	spvc_msl_sampler_mip_filter mip_filter;
	spvc_msl_sampler_address s_address;
	spvc_msl_sampler_address t_address;
	spvc_msl_sampler_address r_address;
	spvc_msl_sampler_compare_func compare_func;
	spvc_msl_sampler_border_color border_color;
	float lod_clamp_min;
	float lod_clamp_max;
	int max_anisotropy;

	spvc_bool compare_enable;
	spvc_bool lod_clamp_enable;
	spvc_bool anisotropy_enable;
} spvc_msl_constexpr_sampler;

/*
 * Initializes the constexpr sampler struct.
 * The defaults are non-zero.
 */
SPVC_PUBLIC_API void spvc_msl_constexpr_sampler_init(spvc_msl_constexpr_sampler *sampler);

/* Maps to the various spirv_cross::Compiler*::Option structures. See C++ API for defaults and details. */
typedef enum spvc_compiler_option
{
	SPVC_COMPILER_OPTION_UNKNOWN = 0,

	SPVC_COMPILER_OPTION_FORCE_TEMPORARY = 1 | SPVC_COMPILER_OPTION_COMMON_BIT,
	SPVC_COMPILER_OPTION_FLATTEN_MULTIDIMENSIONAL_ARRAYS = 2 | SPVC_COMPILER_OPTION_COMMON_BIT,
	SPVC_COMPILER_OPTION_FIXUP_DEPTH_CONVENTION = 3 | SPVC_COMPILER_OPTION_COMMON_BIT,
	SPVC_COMPILER_OPTION_FLIP_VERTEX_Y = 4 | SPVC_COMPILER_OPTION_COMMON_BIT,

	SPVC_COMPILER_OPTION_GLSL_SUPPORT_NONZERO_BASE_INSTANCE = 5 | SPVC_COMPILER_OPTION_GLSL_BIT,
	SPVC_COMPILER_OPTION_GLSL_SEPARATE_SHADER_OBJECTS = 6 | SPVC_COMPILER_OPTION_GLSL_BIT,
	SPVC_COMPILER_OPTION_GLSL_ENABLE_420PACK_EXTENSION = 7 | SPVC_COMPILER_OPTION_GLSL_BIT,
	SPVC_COMPILER_OPTION_GLSL_VERSION = 8 | SPVC_COMPILER_OPTION_GLSL_BIT,
	SPVC_COMPILER_OPTION_GLSL_ES = 9 | SPVC_COMPILER_OPTION_GLSL_BIT,
	SPVC_COMPILER_OPTION_GLSL_VULKAN_SEMANTICS = 10 | SPVC_COMPILER_OPTION_GLSL_BIT,
	SPVC_COMPILER_OPTION_GLSL_ES_DEFAULT_FLOAT_PRECISION_HIGHP = 11 | SPVC_COMPILER_OPTION_GLSL_BIT,
	SPVC_COMPILER_OPTION_GLSL_ES_DEFAULT_INT_PRECISION_HIGHP = 12 | SPVC_COMPILER_OPTION_GLSL_BIT,

	SPVC_COMPILER_OPTION_HLSL_SHADER_MODEL = 13 | SPVC_COMPILER_OPTION_HLSL_BIT,
	SPVC_COMPILER_OPTION_HLSL_POINT_SIZE_COMPAT = 14 | SPVC_COMPILER_OPTION_HLSL_BIT,
	SPVC_COMPILER_OPTION_HLSL_POINT_COORD_COMPAT = 15 | SPVC_COMPILER_OPTION_HLSL_BIT,
	SPVC_COMPILER_OPTION_HLSL_SUPPORT_NONZERO_BASE_VERTEX_BASE_INSTANCE = 16 | SPVC_COMPILER_OPTION_HLSL_BIT,

	SPVC_COMPILER_OPTION_MSL_VERSION = 17 | SPVC_COMPILER_OPTION_MSL_BIT,
	SPVC_COMPILER_OPTION_MSL_TEXEL_BUFFER_TEXTURE_WIDTH = 18 | SPVC_COMPILER_OPTION_MSL_BIT,

	/* Obsolete, use SWIZZLE_BUFFER_INDEX instead. */
	SPVC_COMPILER_OPTION_MSL_AUX_BUFFER_INDEX = 19 | SPVC_COMPILER_OPTION_MSL_BIT,
	SPVC_COMPILER_OPTION_MSL_SWIZZLE_BUFFER_INDEX = 19 | SPVC_COMPILER_OPTION_MSL_BIT,

	SPVC_COMPILER_OPTION_MSL_INDIRECT_PARAMS_BUFFER_INDEX = 20 | SPVC_COMPILER_OPTION_MSL_BIT,
	SPVC_COMPILER_OPTION_MSL_SHADER_OUTPUT_BUFFER_INDEX = 21 | SPVC_COMPILER_OPTION_MSL_BIT,
	SPVC_COMPILER_OPTION_MSL_SHADER_PATCH_OUTPUT_BUFFER_INDEX = 22 | SPVC_COMPILER_OPTION_MSL_BIT,
	SPVC_COMPILER_OPTION_MSL_SHADER_TESS_FACTOR_OUTPUT_BUFFER_INDEX = 23 | SPVC_COMPILER_OPTION_MSL_BIT,
	SPVC_COMPILER_OPTION_MSL_SHADER_INPUT_WORKGROUP_INDEX = 24 | SPVC_COMPILER_OPTION_MSL_BIT,
	SPVC_COMPILER_OPTION_MSL_ENABLE_POINT_SIZE_BUILTIN = 25 | SPVC_COMPILER_OPTION_MSL_BIT,
	SPVC_COMPILER_OPTION_MSL_DISABLE_RASTERIZATION = 26 | SPVC_COMPILER_OPTION_MSL_BIT,
	SPVC_COMPILER_OPTION_MSL_CAPTURE_OUTPUT_TO_BUFFER = 27 | SPVC_COMPILER_OPTION_MSL_BIT,
	SPVC_COMPILER_OPTION_MSL_SWIZZLE_TEXTURE_SAMPLES = 28 | SPVC_COMPILER_OPTION_MSL_BIT,
	SPVC_COMPILER_OPTION_MSL_PAD_FRAGMENT_OUTPUT_COMPONENTS = 29 | SPVC_COMPILER_OPTION_MSL_BIT,
	SPVC_COMPILER_OPTION_MSL_TESS_DOMAIN_ORIGIN_LOWER_LEFT = 30 | SPVC_COMPILER_OPTION_MSL_BIT,
	SPVC_COMPILER_OPTION_MSL_PLATFORM = 31 | SPVC_COMPILER_OPTION_MSL_BIT,
	SPVC_COMPILER_OPTION_MSL_ARGUMENT_BUFFERS = 32 | SPVC_COMPILER_OPTION_MSL_BIT,

	SPVC_COMPILER_OPTION_GLSL_EMIT_PUSH_CONSTANT_AS_UNIFORM_BUFFER = 33 | SPVC_COMPILER_OPTION_GLSL_BIT,

	SPVC_COMPILER_OPTION_MSL_TEXTURE_BUFFER_NATIVE = 34 | SPVC_COMPILER_OPTION_MSL_BIT,

	SPVC_COMPILER_OPTION_GLSL_EMIT_UNIFORM_BUFFER_AS_PLAIN_UNIFORMS = 35 | SPVC_COMPILER_OPTION_GLSL_BIT,

	SPVC_COMPILER_OPTION_INT_MAX = 0x7fffffff
} spvc_compiler_option;

/*
 * Context is the highest-level API construct.
 * The context owns all memory allocations made by its child object hierarchy, including various non-opaque structs and strings.
 * This means that the API user only has to care about one "destroy" call ever when using the C API.
 * All pointers handed out by the APIs are only valid as long as the context
 * is alive and spvc_context_release_allocations has not been called.
 */
SPVC_PUBLIC_API spvc_result spvc_context_create(spvc_context *context);

/* Frees all memory allocations and objects associated with the context and its child objects. */
SPVC_PUBLIC_API void spvc_context_destroy(spvc_context context);

/* Frees all memory allocations and objects associated with the context and its child objects, but keeps the context alive. */
SPVC_PUBLIC_API void spvc_context_release_allocations(spvc_context context);

/* Get the string for the last error which was logged. */
SPVC_PUBLIC_API const char *spvc_context_get_last_error_string(spvc_context context);

/* Get notified in a callback when an error triggers. Useful for debugging. */
typedef void (*spvc_error_callback)(void *userdata, const char *error);
SPVC_PUBLIC_API void spvc_context_set_error_callback(spvc_context context, spvc_error_callback cb, void *userdata);

/* SPIR-V parsing interface. Maps to Parser which then creates a ParsedIR, and that IR is extracted into the handle. */
SPVC_PUBLIC_API spvc_result spvc_context_parse_spirv(spvc_context context, const SpvId *spirv, size_t word_count,
                                                     spvc_parsed_ir *parsed_ir);

/*
 * Create a compiler backend. Capture mode controls if we construct by copy or move semantics.
 * It is always recommended to use SPVC_CAPTURE_MODE_TAKE_OWNERSHIP if you only intend to cross-compile the IR once.
 */
SPVC_PUBLIC_API spvc_result spvc_context_create_compiler(spvc_context context, spvc_backend backend,
                                                         spvc_parsed_ir parsed_ir, spvc_capture_mode mode,
                                                         spvc_compiler *compiler);

/* Maps directly to C++ API. */
SPVC_PUBLIC_API unsigned spvc_compiler_get_current_id_bound(spvc_compiler compiler);

/* Create compiler options, which will initialize defaults. */
SPVC_PUBLIC_API spvc_result spvc_compiler_create_compiler_options(spvc_compiler compiler,
                                                                  spvc_compiler_options *options);
/* Override options. Will return error if e.g. MSL options are used for the HLSL backend, etc. */
SPVC_PUBLIC_API spvc_result spvc_compiler_options_set_bool(spvc_compiler_options options,
                                                           spvc_compiler_option option, spvc_bool value);
SPVC_PUBLIC_API spvc_result spvc_compiler_options_set_uint(spvc_compiler_options options,
                                                           spvc_compiler_option option, unsigned value);
/* Set compiler options. */
SPVC_PUBLIC_API spvc_result spvc_compiler_install_compiler_options(spvc_compiler compiler,
                                                                   spvc_compiler_options options);

/* Compile IR into a string. *source is owned by the context, and caller must not free it themselves. */
SPVC_PUBLIC_API spvc_result spvc_compiler_compile(spvc_compiler compiler, const char **source);

/* Maps to C++ API. */
SPVC_PUBLIC_API spvc_result spvc_compiler_add_header_line(spvc_compiler compiler, const char *line);
SPVC_PUBLIC_API spvc_result spvc_compiler_require_extension(spvc_compiler compiler, const char *ext);
SPVC_PUBLIC_API spvc_result spvc_compiler_flatten_buffer_block(spvc_compiler compiler, spvc_variable_id id);

/*
 * HLSL specifics.
 * Maps to C++ API.
 */
SPVC_PUBLIC_API spvc_result spvc_compiler_hlsl_set_root_constants_layout(spvc_compiler compiler,
                                                                         const spvc_hlsl_root_constants *constant_info,
                                                                         size_t count);
SPVC_PUBLIC_API spvc_result spvc_compiler_hlsl_add_vertex_attribute_remap(spvc_compiler compiler,
                                                                          const spvc_hlsl_vertex_attribute_remap *remap,
                                                                          size_t remaps);
SPVC_PUBLIC_API spvc_variable_id spvc_compiler_hlsl_remap_num_workgroups_builtin(spvc_compiler compiler);

/*
 * MSL specifics.
 * Maps to C++ API.
 */
SPVC_PUBLIC_API spvc_bool spvc_compiler_msl_is_rasterization_disabled(spvc_compiler compiler);

/* Obsolete. Renamed to needs_swizzle_buffer. */
SPVC_PUBLIC_API spvc_bool spvc_compiler_msl_needs_aux_buffer(spvc_compiler compiler);
SPVC_PUBLIC_API spvc_bool spvc_compiler_msl_needs_swizzle_buffer(spvc_compiler compiler);

SPVC_PUBLIC_API spvc_bool spvc_compiler_msl_needs_output_buffer(spvc_compiler compiler);
SPVC_PUBLIC_API spvc_bool spvc_compiler_msl_needs_patch_output_buffer(spvc_compiler compiler);
SPVC_PUBLIC_API spvc_bool spvc_compiler_msl_needs_input_threadgroup_mem(spvc_compiler compiler);
SPVC_PUBLIC_API spvc_result spvc_compiler_msl_add_vertex_attribute(spvc_compiler compiler,
                                                                   const spvc_msl_vertex_attribute *attrs);
SPVC_PUBLIC_API spvc_result spvc_compiler_msl_add_resource_binding(spvc_compiler compiler,
                                                                   const spvc_msl_resource_binding *binding);
SPVC_PUBLIC_API spvc_result spvc_compiler_msl_add_discrete_descriptor_set(spvc_compiler compiler, unsigned desc_set);
SPVC_PUBLIC_API spvc_bool spvc_compiler_msl_is_vertex_attribute_used(spvc_compiler compiler, unsigned location);
SPVC_PUBLIC_API spvc_bool spvc_compiler_msl_is_resource_used(spvc_compiler compiler,
                                                             SpvExecutionModel model,
                                                             unsigned set,
                                                             unsigned binding);
SPVC_PUBLIC_API spvc_result spvc_compiler_msl_remap_constexpr_sampler(spvc_compiler compiler, spvc_variable_id id, const spvc_msl_constexpr_sampler *sampler);
SPVC_PUBLIC_API spvc_result spvc_compiler_msl_set_fragment_output_components(spvc_compiler compiler, unsigned location, unsigned components);

/*
 * Reflect resources.
 * Maps almost 1:1 to C++ API.
 */
SPVC_PUBLIC_API spvc_result spvc_compiler_get_active_interface_variables(spvc_compiler compiler, spvc_set *set);
SPVC_PUBLIC_API spvc_result spvc_compiler_set_enabled_interface_variables(spvc_compiler compiler, spvc_set set);
SPVC_PUBLIC_API spvc_result spvc_compiler_create_shader_resources(spvc_compiler compiler, spvc_resources *resources);
SPVC_PUBLIC_API spvc_result spvc_compiler_create_shader_resources_for_active_variables(spvc_compiler compiler,
                                                                                       spvc_resources *resources,
                                                                                       spvc_set active);
SPVC_PUBLIC_API spvc_result spvc_resources_get_resource_list_for_type(spvc_resources resources, spvc_resource_type type,
                                                                      const spvc_reflected_resource **resource_list,
                                                                      size_t *resource_size);

/*
 * Decorations.
 * Maps to C++ API.
 */
SPVC_PUBLIC_API void spvc_compiler_set_decoration(spvc_compiler compiler, SpvId id, SpvDecoration decoration,
                                                  unsigned argument);
SPVC_PUBLIC_API void spvc_compiler_set_decoration_string(spvc_compiler compiler, SpvId id, SpvDecoration decoration,
                                                         const char *argument);
SPVC_PUBLIC_API void spvc_compiler_set_name(spvc_compiler compiler, SpvId id, const char *argument);
SPVC_PUBLIC_API void spvc_compiler_set_member_decoration(spvc_compiler compiler, spvc_type_id id, unsigned member_index,
                                                         SpvDecoration decoration, unsigned argument);
SPVC_PUBLIC_API void spvc_compiler_set_member_decoration_string(spvc_compiler compiler, spvc_type_id id,
                                                                unsigned member_index, SpvDecoration decoration,
                                                                const char *argument);
SPVC_PUBLIC_API void spvc_compiler_set_member_name(spvc_compiler compiler, spvc_type_id id, unsigned member_index,
                                                   const char *argument);
SPVC_PUBLIC_API void spvc_compiler_unset_decoration(spvc_compiler compiler, SpvId id, SpvDecoration decoration);
SPVC_PUBLIC_API void spvc_compiler_unset_member_decoration(spvc_compiler compiler, spvc_type_id id,
                                                           unsigned member_index, SpvDecoration decoration);

SPVC_PUBLIC_API spvc_bool spvc_compiler_has_decoration(spvc_compiler compiler, SpvId id, SpvDecoration decoration);
SPVC_PUBLIC_API spvc_bool spvc_compiler_has_member_decoration(spvc_compiler compiler, spvc_type_id id,
                                                              unsigned member_index, SpvDecoration decoration);
SPVC_PUBLIC_API const char *spvc_compiler_get_name(spvc_compiler compiler, SpvId id);
SPVC_PUBLIC_API unsigned spvc_compiler_get_decoration(spvc_compiler compiler, SpvId id, SpvDecoration decoration);
SPVC_PUBLIC_API const char *spvc_compiler_get_decoration_string(spvc_compiler compiler, SpvId id,
                                                                SpvDecoration decoration);
SPVC_PUBLIC_API unsigned spvc_compiler_get_member_decoration(spvc_compiler compiler, spvc_type_id id,
                                                             unsigned member_index, SpvDecoration decoration);
SPVC_PUBLIC_API const char *spvc_compiler_get_member_decoration_string(spvc_compiler compiler, spvc_type_id id,
                                                                       unsigned member_index, SpvDecoration decoration);
SPVC_PUBLIC_API const char *spvc_compiler_get_member_name(spvc_compiler compiler, spvc_type_id id, unsigned member_index);

/*
 * Entry points.
 * Maps to C++ API.
 */
SPVC_PUBLIC_API spvc_result spvc_compiler_get_entry_points(spvc_compiler compiler,
                                                           const spvc_entry_point **entry_points,
                                                           size_t *num_entry_points);
SPVC_PUBLIC_API spvc_result spvc_compiler_set_entry_point(spvc_compiler compiler, const char *name,
                                                          SpvExecutionModel model);
SPVC_PUBLIC_API spvc_result spvc_compiler_rename_entry_point(spvc_compiler compiler, const char *old_name,
                                                             const char *new_name, SpvExecutionModel model);
SPVC_PUBLIC_API const char *spvc_compiler_get_cleansed_entry_point_name(spvc_compiler compiler, const char *name,
                                                                        SpvExecutionModel model);
SPVC_PUBLIC_API void spvc_compiler_set_execution_mode(spvc_compiler compiler, SpvExecutionMode mode);
SPVC_PUBLIC_API void spvc_compiler_unset_execution_mode(spvc_compiler compiler, SpvExecutionMode mode);
SPVC_PUBLIC_API void spvc_compiler_set_execution_mode_with_arguments(spvc_compiler compiler, SpvExecutionMode mode,
                                                                     unsigned arg0, unsigned arg1, unsigned arg2);
SPVC_PUBLIC_API spvc_result spvc_compiler_get_execution_modes(spvc_compiler compiler, const SpvExecutionMode **modes,
                                                              size_t *num_modes);
SPVC_PUBLIC_API unsigned spvc_compiler_get_execution_mode_argument(spvc_compiler compiler, SpvExecutionMode mode);
SPVC_PUBLIC_API unsigned spvc_compiler_get_execution_mode_argument_by_index(spvc_compiler compiler,
                                                                            SpvExecutionMode mode, unsigned index);
SPVC_PUBLIC_API SpvExecutionModel spvc_compiler_get_execution_model(spvc_compiler compiler);

/*
 * Type query interface.
 * Maps to C++ API, except it's read-only.
 */
SPVC_PUBLIC_API spvc_type spvc_compiler_get_type_handle(spvc_compiler compiler, spvc_type_id id);

SPVC_PUBLIC_API spvc_basetype spvc_type_get_basetype(spvc_type type);
SPVC_PUBLIC_API unsigned spvc_type_get_bit_width(spvc_type type);
SPVC_PUBLIC_API unsigned spvc_type_get_vector_size(spvc_type type);
SPVC_PUBLIC_API unsigned spvc_type_get_columns(spvc_type type);
SPVC_PUBLIC_API unsigned spvc_type_get_num_array_dimensions(spvc_type type);
SPVC_PUBLIC_API spvc_bool spvc_type_array_dimension_is_literal(spvc_type type, unsigned dimension);
SPVC_PUBLIC_API SpvId spvc_type_get_array_dimension(spvc_type type, unsigned dimension);
SPVC_PUBLIC_API unsigned spvc_type_get_num_member_types(spvc_type type);
SPVC_PUBLIC_API spvc_type_id spvc_type_get_member_type(spvc_type type, unsigned index);
SPVC_PUBLIC_API SpvStorageClass spvc_type_get_storage_class(spvc_type type);

/* Image type query. */
SPVC_PUBLIC_API spvc_type_id spvc_type_get_image_sampled_type(spvc_type type);
SPVC_PUBLIC_API SpvDim spvc_type_get_image_dimension(spvc_type type);
SPVC_PUBLIC_API spvc_bool spvc_type_get_image_is_depth(spvc_type type);
SPVC_PUBLIC_API spvc_bool spvc_type_get_image_arrayed(spvc_type type);
SPVC_PUBLIC_API spvc_bool spvc_type_get_image_multisampled(spvc_type type);
SPVC_PUBLIC_API spvc_bool spvc_type_get_image_is_storage(spvc_type type);
SPVC_PUBLIC_API SpvImageFormat spvc_type_get_image_storage_format(spvc_type type);
SPVC_PUBLIC_API SpvAccessQualifier spvc_type_get_image_access_qualifier(spvc_type type);

/*
 * Buffer layout query.
 * Maps to C++ API.
 */
SPVC_PUBLIC_API spvc_result spvc_compiler_get_declared_struct_size(spvc_compiler compiler, spvc_type struct_type, size_t *size);
SPVC_PUBLIC_API spvc_result spvc_compiler_get_declared_struct_size_runtime_array(spvc_compiler compiler,
                                                                                 spvc_type struct_type, size_t array_size, size_t *size);

SPVC_PUBLIC_API spvc_result spvc_compiler_type_struct_member_offset(spvc_compiler compiler,
                                                                    spvc_type type, unsigned index, unsigned *offset);
SPVC_PUBLIC_API spvc_result spvc_compiler_type_struct_member_array_stride(spvc_compiler compiler,
                                                                          spvc_type type, unsigned index, unsigned *stride);
SPVC_PUBLIC_API spvc_result spvc_compiler_type_struct_member_matrix_stride(spvc_compiler compiler,
                                                                           spvc_type type, unsigned index, unsigned *stride);

/*
 * Workaround helper functions.
 * Maps to C++ API.
 */
SPVC_PUBLIC_API spvc_result spvc_compiler_build_dummy_sampler_for_combined_images(spvc_compiler compiler, spvc_variable_id *id);
SPVC_PUBLIC_API spvc_result spvc_compiler_build_combined_image_samplers(spvc_compiler compiler);
SPVC_PUBLIC_API spvc_result spvc_compiler_get_combined_image_samplers(spvc_compiler compiler,
                                                                      const spvc_combined_image_sampler **samplers,
                                                                      size_t *num_samplers);

/*
 * Constants
 * Maps to C++ API.
 */
SPVC_PUBLIC_API spvc_result spvc_compiler_get_specialization_constants(spvc_compiler compiler,
                                                                       const spvc_specialization_constant **constants,
                                                                       size_t *num_constants);
SPVC_PUBLIC_API spvc_constant spvc_compiler_get_constant_handle(spvc_compiler compiler,
                                                                spvc_constant_id id);

SPVC_PUBLIC_API spvc_constant_id spvc_compiler_get_work_group_size_specialization_constants(spvc_compiler compiler,
                                                                                            spvc_specialization_constant *x,
                                                                                            spvc_specialization_constant *y,
                                                                                            spvc_specialization_constant *z);

/*
 * Buffer ranges
 * Maps to C++ API.
 */
SPVC_PUBLIC_API spvc_result spvc_compiler_get_active_buffer_ranges(spvc_compiler compiler,
                                                                   spvc_variable_id id,
                                                                   const spvc_buffer_range **ranges,
                                                                   size_t *num_ranges);

/*
 * No stdint.h until C99, sigh :(
 * For smaller types, the result is sign or zero-extended as appropriate.
 * Maps to C++ API.
 * TODO: The SPIRConstant query interface and modification interface is not quite complete.
 */
SPVC_PUBLIC_API float spvc_constant_get_scalar_fp16(spvc_constant constant, unsigned column, unsigned row);
SPVC_PUBLIC_API float spvc_constant_get_scalar_fp32(spvc_constant constant, unsigned column, unsigned row);
SPVC_PUBLIC_API double spvc_constant_get_scalar_fp64(spvc_constant constant, unsigned column, unsigned row);
SPVC_PUBLIC_API unsigned spvc_constant_get_scalar_u32(spvc_constant constant, unsigned column, unsigned row);
SPVC_PUBLIC_API int spvc_constant_get_scalar_i32(spvc_constant constant, unsigned column, unsigned row);
SPVC_PUBLIC_API unsigned spvc_constant_get_scalar_u16(spvc_constant constant, unsigned column, unsigned row);
SPVC_PUBLIC_API int spvc_constant_get_scalar_i16(spvc_constant constant, unsigned column, unsigned row);
SPVC_PUBLIC_API unsigned spvc_constant_get_scalar_u8(spvc_constant constant, unsigned column, unsigned row);
SPVC_PUBLIC_API int spvc_constant_get_scalar_i8(spvc_constant constant, unsigned column, unsigned row);
SPVC_PUBLIC_API void spvc_constant_get_subconstants(spvc_constant constant, const spvc_constant_id **constituents, size_t *count);
SPVC_PUBLIC_API spvc_type_id spvc_constant_get_type(spvc_constant constant);

/*
 * Misc reflection
 * Maps to C++ API.
 */
SPVC_PUBLIC_API spvc_bool spvc_compiler_get_binary_offset_for_decoration(spvc_compiler compiler,
                                                                         spvc_variable_id id,
                                                                         SpvDecoration decoration,
                                                                         unsigned *word_offset);

SPVC_PUBLIC_API spvc_bool spvc_compiler_buffer_is_hlsl_counter_buffer(spvc_compiler compiler, spvc_variable_id id);
SPVC_PUBLIC_API spvc_bool spvc_compiler_buffer_get_hlsl_counter_buffer(spvc_compiler compiler, spvc_variable_id id,
                                                                       spvc_variable_id *counter_id);

SPVC_PUBLIC_API spvc_result spvc_compiler_get_declared_capabilities(spvc_compiler compiler,
                                                                    const SpvCapability **capabilities,
                                                                    size_t *num_capabilities);
SPVC_PUBLIC_API spvc_result spvc_compiler_get_declared_extensions(spvc_compiler compiler, const char ***extensions,
                                                                  size_t *num_extensions);

SPVC_PUBLIC_API const char *spvc_compiler_get_remapped_declared_block_name(spvc_compiler compiler, spvc_variable_id id);
SPVC_PUBLIC_API spvc_result spvc_compiler_get_buffer_block_decorations(spvc_compiler compiler, spvc_variable_id id,
                                                                       const SpvDecoration **decorations,
                                                                       size_t *num_decorations);

#ifdef __cplusplus
}
#endif
#endif
