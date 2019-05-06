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

#include "spirv_cross_c.h"

#if SPIRV_CROSS_C_API_CPP
#include "spirv_cpp.hpp"
#endif
#if SPIRV_CROSS_C_API_GLSL
#include "spirv_glsl.hpp"
#else
#include "spirv_cross.hpp"
#endif
#if SPIRV_CROSS_C_API_HLSL
#include "spirv_hlsl.hpp"
#endif
#if SPIRV_CROSS_C_API_MSL
#include "spirv_msl.hpp"
#endif
#if SPIRV_CROSS_C_API_REFLECT
#include "spirv_reflect.hpp"
#endif
#include "spirv_parser.hpp"
#include <memory>
#include <new>
#include <string.h>

// clang-format off

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996)
#endif

#ifndef SPIRV_CROSS_EXCEPTIONS_TO_ASSERTIONS
#define SPVC_BEGIN_SAFE_SCOPE try
#else
#define SPVC_BEGIN_SAFE_SCOPE
#endif

#ifndef SPIRV_CROSS_EXCEPTIONS_TO_ASSERTIONS
#define SPVC_END_SAFE_SCOPE(context, error) \
	catch (const std::exception &e)         \
	{                                       \
		(context)->report_error(e.what());  \
		return (error);                     \
	}
#else
#define SPVC_END_SAFE_SCOPE(context, error)
#endif

using namespace std;
using namespace SPIRV_CROSS_NAMESPACE;

struct ScratchMemoryAllocation
{
	virtual ~ScratchMemoryAllocation() = default;
};

struct StringAllocation : ScratchMemoryAllocation
{
	explicit StringAllocation(const char *name)
	    : str(name)
	{
	}

	explicit StringAllocation(std::string name)
	    : str(std::move(name))
	{
	}

	std::string str;
};

template <typename T>
struct TemporaryBuffer : ScratchMemoryAllocation
{
	SmallVector<T> buffer;
};

template <typename T, typename... Ts>
static inline std::unique_ptr<T> spvc_allocate(Ts &&... ts)
{
	return std::unique_ptr<T>(new T(std::forward<Ts>(ts)...));
}

struct spvc_context_s
{
	string last_error;
	SmallVector<unique_ptr<ScratchMemoryAllocation>> allocations;
	const char *allocate_name(const std::string &name);

	spvc_error_callback callback = nullptr;
	void *callback_userdata = nullptr;
	void report_error(std::string msg);
};

void spvc_context_s::report_error(std::string msg)
{
	last_error = std::move(msg);
	if (callback)
		callback(callback_userdata, last_error.c_str());
}

const char *spvc_context_s::allocate_name(const std::string &name)
{
	SPVC_BEGIN_SAFE_SCOPE
	{
		auto alloc = spvc_allocate<StringAllocation>(name);
		auto *ret = alloc->str.c_str();
		allocations.emplace_back(std::move(alloc));
		return ret;
	}
	SPVC_END_SAFE_SCOPE(this, nullptr)
}

struct spvc_parsed_ir_s : ScratchMemoryAllocation
{
	spvc_context context = nullptr;
	ParsedIR parsed;
};

struct spvc_compiler_s : ScratchMemoryAllocation
{
	spvc_context context = nullptr;
	unique_ptr<Compiler> compiler;
	spvc_backend backend = SPVC_BACKEND_NONE;
};

struct spvc_compiler_options_s : ScratchMemoryAllocation
{
	spvc_context context = nullptr;
	uint32_t backend_flags = 0;
#if SPIRV_CROSS_C_API_GLSL
	CompilerGLSL::Options glsl;
#endif
#if SPIRV_CROSS_C_API_MSL
	CompilerMSL::Options msl;
#endif
#if SPIRV_CROSS_C_API_HLSL
	CompilerHLSL::Options hlsl;
#endif
};

struct spvc_set_s : ScratchMemoryAllocation
{
	std::unordered_set<uint32_t> set;
};

// Dummy-inherit to we can keep our opaque type handle type safe in C-land as well,
// and avoid just throwing void * around.
struct spvc_type_s : SPIRType
{
};

struct spvc_constant_s : SPIRConstant
{
};

struct spvc_resources_s : ScratchMemoryAllocation
{
	spvc_context context = nullptr;
	SmallVector<spvc_reflected_resource> uniform_buffers;
	SmallVector<spvc_reflected_resource> storage_buffers;
	SmallVector<spvc_reflected_resource> stage_inputs;
	SmallVector<spvc_reflected_resource> stage_outputs;
	SmallVector<spvc_reflected_resource> subpass_inputs;
	SmallVector<spvc_reflected_resource> storage_images;
	SmallVector<spvc_reflected_resource> sampled_images;
	SmallVector<spvc_reflected_resource> atomic_counters;
	SmallVector<spvc_reflected_resource> push_constant_buffers;
	SmallVector<spvc_reflected_resource> separate_images;
	SmallVector<spvc_reflected_resource> separate_samplers;
	SmallVector<spvc_reflected_resource> acceleration_structures;

	bool copy_resources(SmallVector<spvc_reflected_resource> &outputs, const SmallVector<Resource> &inputs);
	bool copy_resources(const ShaderResources &resources);
};

spvc_result spvc_context_create(spvc_context *context)
{
	auto *ctx = new (std::nothrow) spvc_context_s;
	if (!ctx)
		return SPVC_ERROR_OUT_OF_MEMORY;

	*context = ctx;
	return SPVC_SUCCESS;
}

void spvc_context_destroy(spvc_context context)
{
	delete context;
}

void spvc_context_release_allocations(spvc_context context)
{
	context->allocations.clear();
}

const char *spvc_context_get_last_error_string(spvc_context context)
{
	return context->last_error.c_str();
}

SPVC_PUBLIC_API void spvc_context_set_error_callback(spvc_context context, spvc_error_callback cb, void *userdata)
{
	context->callback = cb;
	context->callback_userdata = userdata;
}

spvc_result spvc_context_parse_spirv(spvc_context context, const SpvId *spirv, size_t word_count,
                                     spvc_parsed_ir *parsed_ir)
{
	SPVC_BEGIN_SAFE_SCOPE
	{
		std::unique_ptr<spvc_parsed_ir_s> pir(new (std::nothrow) spvc_parsed_ir_s);
		if (!pir)
		{
			context->report_error("Out of memory.");
			return SPVC_ERROR_OUT_OF_MEMORY;
		}

		pir->context = context;
		Parser parser(spirv, word_count);
		parser.parse();
		pir->parsed = move(parser.get_parsed_ir());
		*parsed_ir = pir.get();
		context->allocations.push_back(std::move(pir));
	}
	SPVC_END_SAFE_SCOPE(context, SPVC_ERROR_INVALID_SPIRV)
	return SPVC_SUCCESS;
}

spvc_result spvc_context_create_compiler(spvc_context context, spvc_backend backend, spvc_parsed_ir parsed_ir,
                                         spvc_capture_mode mode, spvc_compiler *compiler)
{
	SPVC_BEGIN_SAFE_SCOPE
	{
		std::unique_ptr<spvc_compiler_s> comp(new (std::nothrow) spvc_compiler_s);
		if (!comp)
		{
			context->report_error("Out of memory.");
			return SPVC_ERROR_OUT_OF_MEMORY;
		}
		comp->backend = backend;
		comp->context = context;

		if (mode != SPVC_CAPTURE_MODE_COPY && mode != SPVC_CAPTURE_MODE_TAKE_OWNERSHIP)
		{
			context->report_error("Invalid argument for capture mode.");
			return SPVC_ERROR_INVALID_ARGUMENT;
		}

		switch (backend)
		{
		case SPVC_BACKEND_NONE:
			if (mode == SPVC_CAPTURE_MODE_TAKE_OWNERSHIP)
				comp->compiler.reset(new Compiler(move(parsed_ir->parsed)));
			else if (mode == SPVC_CAPTURE_MODE_COPY)
				comp->compiler.reset(new Compiler(parsed_ir->parsed));
			break;

#if SPIRV_CROSS_C_API_GLSL
		case SPVC_BACKEND_GLSL:
			if (mode == SPVC_CAPTURE_MODE_TAKE_OWNERSHIP)
				comp->compiler.reset(new CompilerGLSL(move(parsed_ir->parsed)));
			else if (mode == SPVC_CAPTURE_MODE_COPY)
				comp->compiler.reset(new CompilerGLSL(parsed_ir->parsed));
			break;
#endif

#if SPIRV_CROSS_C_API_HLSL
		case SPVC_BACKEND_HLSL:
			if (mode == SPVC_CAPTURE_MODE_TAKE_OWNERSHIP)
				comp->compiler.reset(new CompilerHLSL(move(parsed_ir->parsed)));
			else if (mode == SPVC_CAPTURE_MODE_COPY)
				comp->compiler.reset(new CompilerHLSL(parsed_ir->parsed));
			break;
#endif

#if SPIRV_CROSS_C_API_MSL
		case SPVC_BACKEND_MSL:
			if (mode == SPVC_CAPTURE_MODE_TAKE_OWNERSHIP)
				comp->compiler.reset(new CompilerMSL(move(parsed_ir->parsed)));
			else if (mode == SPVC_CAPTURE_MODE_COPY)
				comp->compiler.reset(new CompilerMSL(parsed_ir->parsed));
			break;
#endif

#if SPIRV_CROSS_C_API_CPP
		case SPVC_BACKEND_CPP:
			if (mode == SPVC_CAPTURE_MODE_TAKE_OWNERSHIP)
				comp->compiler.reset(new CompilerCPP(move(parsed_ir->parsed)));
			else if (mode == SPVC_CAPTURE_MODE_COPY)
				comp->compiler.reset(new CompilerCPP(parsed_ir->parsed));
			break;
#endif

#if SPIRV_CROSS_C_API_REFLECT
		case SPVC_BACKEND_JSON:
			if (mode == SPVC_CAPTURE_MODE_TAKE_OWNERSHIP)
				comp->compiler.reset(new CompilerReflection(move(parsed_ir->parsed)));
			else if (mode == SPVC_CAPTURE_MODE_COPY)
				comp->compiler.reset(new CompilerReflection(parsed_ir->parsed));
			break;
#endif

		default:
			context->report_error("Invalid backend.");
			return SPVC_ERROR_INVALID_ARGUMENT;
		}

		*compiler = comp.get();
		context->allocations.push_back(std::move(comp));
	}
	SPVC_END_SAFE_SCOPE(context, SPVC_ERROR_OUT_OF_MEMORY)
	return SPVC_SUCCESS;
}

spvc_result spvc_compiler_create_compiler_options(spvc_compiler compiler, spvc_compiler_options *options)
{
	SPVC_BEGIN_SAFE_SCOPE
	{
		std::unique_ptr<spvc_compiler_options_s> opt(new (std::nothrow) spvc_compiler_options_s);
		if (!opt)
		{
			compiler->context->report_error("Out of memory.");
			return SPVC_ERROR_OUT_OF_MEMORY;
		}

		opt->context = compiler->context;
		opt->backend_flags = 0;
		switch (compiler->backend)
		{
#if SPIRV_CROSS_C_API_MSL
		case SPVC_BACKEND_MSL:
			opt->backend_flags |= SPVC_COMPILER_OPTION_MSL_BIT | SPVC_COMPILER_OPTION_COMMON_BIT;
			opt->glsl = static_cast<CompilerMSL *>(compiler->compiler.get())->get_common_options();
			opt->msl = static_cast<CompilerMSL *>(compiler->compiler.get())->get_msl_options();
			break;
#endif

#if SPIRV_CROSS_C_API_HLSL
		case SPVC_BACKEND_HLSL:
			opt->backend_flags |= SPVC_COMPILER_OPTION_HLSL_BIT | SPVC_COMPILER_OPTION_COMMON_BIT;
			opt->glsl = static_cast<CompilerHLSL *>(compiler->compiler.get())->get_common_options();
			opt->hlsl = static_cast<CompilerHLSL *>(compiler->compiler.get())->get_hlsl_options();
			break;
#endif

#if SPIRV_CROSS_C_API_GLSL
		case SPVC_BACKEND_GLSL:
			opt->backend_flags |= SPVC_COMPILER_OPTION_GLSL_BIT | SPVC_COMPILER_OPTION_COMMON_BIT;
			opt->glsl = static_cast<CompilerGLSL *>(compiler->compiler.get())->get_common_options();
			break;
#endif

		default:
			break;
		}

		*options = opt.get();
		compiler->context->allocations.push_back(std::move(opt));
	}
	SPVC_END_SAFE_SCOPE(compiler->context, SPVC_ERROR_OUT_OF_MEMORY)
	return SPVC_SUCCESS;
}

spvc_result spvc_compiler_options_set_bool(spvc_compiler_options options, spvc_compiler_option option,
                                           spvc_bool value)
{
	return spvc_compiler_options_set_uint(options, option, value ? 1 : 0);
}

spvc_result spvc_compiler_options_set_uint(spvc_compiler_options options, spvc_compiler_option option, unsigned value)
{
	(void)value;
	(void)option;
	uint32_t supported_mask = options->backend_flags;
	uint32_t required_mask = option & SPVC_COMPILER_OPTION_LANG_BITS;
	if ((required_mask | supported_mask) != supported_mask)
	{
		options->context->report_error("Option is not supported by current backend.");
		return SPVC_ERROR_INVALID_ARGUMENT;
	}

	switch (option)
	{
#if SPIRV_CROSS_C_API_GLSL
	case SPVC_COMPILER_OPTION_FORCE_TEMPORARY:
		options->glsl.force_temporary = value != 0;
		break;
	case SPVC_COMPILER_OPTION_FLATTEN_MULTIDIMENSIONAL_ARRAYS:
		options->glsl.flatten_multidimensional_arrays = value != 0;
		break;
	case SPVC_COMPILER_OPTION_FIXUP_DEPTH_CONVENTION:
		options->glsl.vertex.fixup_clipspace = value != 0;
		break;
	case SPVC_COMPILER_OPTION_FLIP_VERTEX_Y:
		options->glsl.vertex.flip_vert_y = value != 0;
		break;

	case SPVC_COMPILER_OPTION_GLSL_SUPPORT_NONZERO_BASE_INSTANCE:
		options->glsl.vertex.support_nonzero_base_instance = value != 0;
		break;
	case SPVC_COMPILER_OPTION_GLSL_SEPARATE_SHADER_OBJECTS:
		options->glsl.separate_shader_objects = value != 0;
		break;
	case SPVC_COMPILER_OPTION_GLSL_ENABLE_420PACK_EXTENSION:
		options->glsl.enable_420pack_extension = value != 0;
		break;
	case SPVC_COMPILER_OPTION_GLSL_VERSION:
		options->glsl.version = value;
		break;
	case SPVC_COMPILER_OPTION_GLSL_ES:
		options->glsl.es = value != 0;
		break;
	case SPVC_COMPILER_OPTION_GLSL_VULKAN_SEMANTICS:
		options->glsl.vulkan_semantics = value != 0;
		break;
	case SPVC_COMPILER_OPTION_GLSL_ES_DEFAULT_FLOAT_PRECISION_HIGHP:
		options->glsl.fragment.default_float_precision =
		    value != 0 ? CompilerGLSL::Options::Precision::Highp : CompilerGLSL::Options::Precision::Mediump;
		break;
	case SPVC_COMPILER_OPTION_GLSL_ES_DEFAULT_INT_PRECISION_HIGHP:
		options->glsl.fragment.default_int_precision =
		    value != 0 ? CompilerGLSL::Options::Precision::Highp : CompilerGLSL::Options::Precision::Mediump;
		break;
	case SPVC_COMPILER_OPTION_GLSL_EMIT_PUSH_CONSTANT_AS_UNIFORM_BUFFER:
		options->glsl.emit_push_constant_as_uniform_buffer = value != 0;
		break;
	case SPVC_COMPILER_OPTION_GLSL_EMIT_UNIFORM_BUFFER_AS_PLAIN_UNIFORMS:
		options->glsl.emit_uniform_buffer_as_plain_uniforms = value != 0;
		break;
#endif

#if SPIRV_CROSS_C_API_HLSL
	case SPVC_COMPILER_OPTION_HLSL_SHADER_MODEL:
		options->hlsl.shader_model = value;
		break;

	case SPVC_COMPILER_OPTION_HLSL_POINT_SIZE_COMPAT:
		options->hlsl.point_size_compat = value != 0;
		break;

	case SPVC_COMPILER_OPTION_HLSL_POINT_COORD_COMPAT:
		options->hlsl.point_coord_compat = value != 0;
		break;

	case SPVC_COMPILER_OPTION_HLSL_SUPPORT_NONZERO_BASE_VERTEX_BASE_INSTANCE:
		options->hlsl.support_nonzero_base_vertex_base_instance = value != 0;
		break;
#endif

#if SPIRV_CROSS_C_API_MSL
	case SPVC_COMPILER_OPTION_MSL_VERSION:
		options->msl.msl_version = value;
		break;

	case SPVC_COMPILER_OPTION_MSL_TEXEL_BUFFER_TEXTURE_WIDTH:
		options->msl.texel_buffer_texture_width = value;
		break;

	case SPVC_COMPILER_OPTION_MSL_SWIZZLE_BUFFER_INDEX:
		options->msl.swizzle_buffer_index = value;
		break;

	case SPVC_COMPILER_OPTION_MSL_INDIRECT_PARAMS_BUFFER_INDEX:
		options->msl.indirect_params_buffer_index = value;
		break;

	case SPVC_COMPILER_OPTION_MSL_SHADER_OUTPUT_BUFFER_INDEX:
		options->msl.shader_output_buffer_index = value;
		break;

	case SPVC_COMPILER_OPTION_MSL_SHADER_PATCH_OUTPUT_BUFFER_INDEX:
		options->msl.shader_patch_output_buffer_index = value;
		break;

	case SPVC_COMPILER_OPTION_MSL_SHADER_TESS_FACTOR_OUTPUT_BUFFER_INDEX:
		options->msl.shader_tess_factor_buffer_index = value;
		break;

	case SPVC_COMPILER_OPTION_MSL_SHADER_INPUT_WORKGROUP_INDEX:
		options->msl.shader_input_wg_index = value;
		break;

	case SPVC_COMPILER_OPTION_MSL_ENABLE_POINT_SIZE_BUILTIN:
		options->msl.enable_point_size_builtin = value != 0;
		break;

	case SPVC_COMPILER_OPTION_MSL_DISABLE_RASTERIZATION:
		options->msl.disable_rasterization = value != 0;
		break;

	case SPVC_COMPILER_OPTION_MSL_CAPTURE_OUTPUT_TO_BUFFER:
		options->msl.capture_output_to_buffer = value != 0;
		break;

	case SPVC_COMPILER_OPTION_MSL_SWIZZLE_TEXTURE_SAMPLES:
		options->msl.swizzle_texture_samples = value != 0;
		break;

	case SPVC_COMPILER_OPTION_MSL_PAD_FRAGMENT_OUTPUT_COMPONENTS:
		options->msl.pad_fragment_output_components = value != 0;
		break;

	case SPVC_COMPILER_OPTION_MSL_TESS_DOMAIN_ORIGIN_LOWER_LEFT:
		options->msl.tess_domain_origin_lower_left = value != 0;
		break;

	case SPVC_COMPILER_OPTION_MSL_PLATFORM:
		options->msl.platform = static_cast<CompilerMSL::Options::Platform>(value);
		break;

	case SPVC_COMPILER_OPTION_MSL_ARGUMENT_BUFFERS:
		options->msl.argument_buffers = value != 0;
		break;

	case SPVC_COMPILER_OPTION_MSL_TEXTURE_BUFFER_NATIVE:
		options->msl.texture_buffer_native = value != 0;
		break;
#endif

	default:
		options->context->report_error("Unknown option.");
		return SPVC_ERROR_INVALID_ARGUMENT;
	}

	return SPVC_SUCCESS;
}

spvc_result spvc_compiler_install_compiler_options(spvc_compiler compiler, spvc_compiler_options options)
{
	(void)options;
	switch (compiler->backend)
	{
#if SPIRV_CROSS_C_API_GLSL
	case SPVC_BACKEND_GLSL:
		static_cast<CompilerGLSL &>(*compiler->compiler).set_common_options(options->glsl);
		break;
#endif

#if SPIRV_CROSS_C_API_HLSL
	case SPVC_BACKEND_HLSL:
		static_cast<CompilerHLSL &>(*compiler->compiler).set_common_options(options->glsl);
		static_cast<CompilerHLSL &>(*compiler->compiler).set_hlsl_options(options->hlsl);
		break;
#endif

#if SPIRV_CROSS_C_API_MSL
	case SPVC_BACKEND_MSL:
		static_cast<CompilerMSL &>(*compiler->compiler).set_common_options(options->glsl);
		static_cast<CompilerMSL &>(*compiler->compiler).set_msl_options(options->msl);
		break;
#endif

	default:
		break;
	}

	return SPVC_SUCCESS;
}

spvc_result spvc_compiler_add_header_line(spvc_compiler compiler, const char *line)
{
#if SPIRV_CROSS_C_API_GLSL
	if (compiler->backend == SPVC_BACKEND_NONE)
	{
		compiler->context->report_error("Cross-compilation related option used on NONE backend which only supports reflection.");
		return SPVC_ERROR_INVALID_ARGUMENT;
	}

	static_cast<CompilerGLSL *>(compiler->compiler.get())->add_header_line(line);
	return SPVC_SUCCESS;
#else
	(void)line;
	compiler->context->report_error("Cross-compilation related option used on NONE backend which only supports reflection.");
	return SPVC_ERROR_INVALID_ARGUMENT;
#endif
}

spvc_result spvc_compiler_require_extension(spvc_compiler compiler, const char *line)
{
#if SPIRV_CROSS_C_API_GLSL
	if (compiler->backend == SPVC_BACKEND_NONE)
	{
		compiler->context->report_error("Cross-compilation related option used on NONE backend which only supports reflection.");
		return SPVC_ERROR_INVALID_ARGUMENT;
	}

	static_cast<CompilerGLSL *>(compiler->compiler.get())->require_extension(line);
	return SPVC_SUCCESS;
#else
	(void)line;
	compiler->context->report_error("Cross-compilation related option used on NONE backend which only supports reflection.");
	return SPVC_ERROR_INVALID_ARGUMENT;
#endif
}

spvc_result spvc_compiler_flatten_buffer_block(spvc_compiler compiler, spvc_variable_id id)
{
#if SPIRV_CROSS_C_API_GLSL
	if (compiler->backend == SPVC_BACKEND_NONE)
	{
		compiler->context->report_error("Cross-compilation related option used on NONE backend which only supports reflection.");
		return SPVC_ERROR_INVALID_ARGUMENT;
	}

	static_cast<CompilerGLSL *>(compiler->compiler.get())->flatten_buffer_block(id);
	return SPVC_SUCCESS;
#else
	(void)id;
	compiler->context->report_error("Cross-compilation related option used on NONE backend which only supports reflection.");
	return SPVC_ERROR_INVALID_ARGUMENT;
#endif
}

spvc_result spvc_compiler_hlsl_set_root_constants_layout(spvc_compiler compiler,
                                                         const spvc_hlsl_root_constants *constant_info,
                                                         size_t count)
{
#if SPIRV_CROSS_C_API_HLSL
	if (compiler->backend != SPVC_BACKEND_HLSL)
	{
		compiler->context->report_error("HLSL function used on a non-HLSL backend.");
		return SPVC_ERROR_INVALID_ARGUMENT;
	}

	auto &hlsl = *static_cast<CompilerHLSL *>(compiler->compiler.get());
	vector<RootConstants> roots;
	roots.reserve(count);
	for (size_t i = 0; i < count; i++)
	{
		RootConstants root;
		root.binding = constant_info[i].binding;
		root.space = constant_info[i].space;
		root.start = constant_info[i].start;
		root.end = constant_info[i].end;
		roots.push_back(root);
	}

	hlsl.set_root_constant_layouts(std::move(roots));
	return SPVC_SUCCESS;
#else
	(void)constant_info;
	(void)count;
	compiler->context->report_error("HLSL function used on a non-HLSL backend.");
	return SPVC_ERROR_INVALID_ARGUMENT;
#endif
}

spvc_result spvc_compiler_hlsl_add_vertex_attribute_remap(spvc_compiler compiler,
                                                          const spvc_hlsl_vertex_attribute_remap *remap,
                                                          size_t count)
{
#if SPIRV_CROSS_C_API_HLSL
	if (compiler->backend != SPVC_BACKEND_HLSL)
	{
		compiler->context->report_error("HLSL function used on a non-HLSL backend.");
		return SPVC_ERROR_INVALID_ARGUMENT;
	}

	HLSLVertexAttributeRemap re;
	auto &hlsl = *static_cast<CompilerHLSL *>(compiler->compiler.get());
	for (size_t i = 0; i < count; i++)
	{
		re.location = remap[i].location;
		re.semantic = remap[i].semantic;
		hlsl.add_vertex_attribute_remap(re);
	}

	return SPVC_SUCCESS;
#else
	(void)remap;
	(void)count;
	compiler->context->report_error("HLSL function used on a non-HLSL backend.");
	return SPVC_ERROR_INVALID_ARGUMENT;
#endif
}

spvc_variable_id spvc_compiler_hlsl_remap_num_workgroups_builtin(spvc_compiler compiler)
{
#if SPIRV_CROSS_C_API_HLSL
	if (compiler->backend != SPVC_BACKEND_HLSL)
	{
		compiler->context->report_error("HLSL function used on a non-HLSL backend.");
		return 0;
	}

	auto &hlsl = *static_cast<CompilerHLSL *>(compiler->compiler.get());
	return hlsl.remap_num_workgroups_builtin();
#else
	compiler->context->report_error("HLSL function used on a non-HLSL backend.");
	return 0;
#endif
}

spvc_bool spvc_compiler_msl_is_rasterization_disabled(spvc_compiler compiler)
{
#if SPIRV_CROSS_C_API_MSL
	if (compiler->backend != SPVC_BACKEND_MSL)
	{
		compiler->context->report_error("MSL function used on a non-MSL backend.");
		return SPVC_FALSE;
	}

	auto &msl = *static_cast<CompilerMSL *>(compiler->compiler.get());
	return msl.get_is_rasterization_disabled() ? SPVC_TRUE : SPVC_FALSE;
#else
	compiler->context->report_error("MSL function used on a non-MSL backend.");
	return SPVC_FALSE;
#endif
}

spvc_bool spvc_compiler_msl_needs_swizzle_buffer(spvc_compiler compiler)
{
#if SPIRV_CROSS_C_API_MSL
	if (compiler->backend != SPVC_BACKEND_MSL)
	{
		compiler->context->report_error("MSL function used on a non-MSL backend.");
		return SPVC_FALSE;
	}

	auto &msl = *static_cast<CompilerMSL *>(compiler->compiler.get());
	return msl.needs_swizzle_buffer() ? SPVC_TRUE : SPVC_FALSE;
#else
	compiler->context->report_error("MSL function used on a non-MSL backend.");
	return SPVC_FALSE;
#endif
}

spvc_bool spvc_compiler_msl_needs_aux_buffer(spvc_compiler compiler)
{
	return spvc_compiler_msl_needs_swizzle_buffer(compiler);
}

spvc_bool spvc_compiler_msl_needs_output_buffer(spvc_compiler compiler)
{
#if SPIRV_CROSS_C_API_MSL
	if (compiler->backend != SPVC_BACKEND_MSL)
	{
		compiler->context->report_error("MSL function used on a non-MSL backend.");
		return SPVC_FALSE;
	}

	auto &msl = *static_cast<CompilerMSL *>(compiler->compiler.get());
	return msl.needs_output_buffer() ? SPVC_TRUE : SPVC_FALSE;
#else
	compiler->context->report_error("MSL function used on a non-MSL backend.");
	return SPVC_FALSE;
#endif
}

spvc_bool spvc_compiler_msl_needs_patch_output_buffer(spvc_compiler compiler)
{
#if SPIRV_CROSS_C_API_MSL
	if (compiler->backend != SPVC_BACKEND_MSL)
	{
		compiler->context->report_error("MSL function used on a non-MSL backend.");
		return SPVC_FALSE;
	}

	auto &msl = *static_cast<CompilerMSL *>(compiler->compiler.get());
	return msl.needs_patch_output_buffer() ? SPVC_TRUE : SPVC_FALSE;
#else
	compiler->context->report_error("MSL function used on a non-MSL backend.");
	return SPVC_FALSE;
#endif
}

spvc_bool spvc_compiler_msl_needs_input_threadgroup_mem(spvc_compiler compiler)
{
#if SPIRV_CROSS_C_API_MSL
	if (compiler->backend != SPVC_BACKEND_MSL)
	{
		compiler->context->report_error("MSL function used on a non-MSL backend.");
		return SPVC_FALSE;
	}

	auto &msl = *static_cast<CompilerMSL *>(compiler->compiler.get());
	return msl.needs_input_threadgroup_mem() ? SPVC_TRUE : SPVC_FALSE;
#else
	compiler->context->report_error("MSL function used on a non-MSL backend.");
	return SPVC_FALSE;
#endif
}

spvc_result spvc_compiler_msl_add_vertex_attribute(spvc_compiler compiler, const spvc_msl_vertex_attribute *va)
{
#if SPIRV_CROSS_C_API_MSL
	if (compiler->backend != SPVC_BACKEND_MSL)
	{
		compiler->context->report_error("MSL function used on a non-MSL backend.");
		return SPVC_ERROR_INVALID_ARGUMENT;
	}

	auto &msl = *static_cast<CompilerMSL *>(compiler->compiler.get());
	MSLVertexAttr attr;
	attr.location = va->location;
	attr.msl_buffer = va->msl_buffer;
	attr.msl_offset = va->msl_offset;
	attr.msl_stride = va->msl_stride;
	attr.format = static_cast<MSLVertexFormat>(va->format);
	attr.builtin = static_cast<spv::BuiltIn>(va->builtin);
	attr.per_instance = va->per_instance;
	msl.add_msl_vertex_attribute(attr);
	return SPVC_SUCCESS;
#else
	(void)va;
	compiler->context->report_error("MSL function used on a non-MSL backend.");
	return SPVC_ERROR_INVALID_ARGUMENT;
#endif
}

spvc_result spvc_compiler_msl_add_resource_binding(spvc_compiler compiler,
                                                   const spvc_msl_resource_binding *binding)
{
#if SPIRV_CROSS_C_API_MSL
	if (compiler->backend != SPVC_BACKEND_MSL)
	{
		compiler->context->report_error("MSL function used on a non-MSL backend.");
		return SPVC_ERROR_INVALID_ARGUMENT;
	}

	auto &msl = *static_cast<CompilerMSL *>(compiler->compiler.get());
	MSLResourceBinding bind;
	bind.binding = binding->binding;
	bind.desc_set = binding->desc_set;
	bind.stage = static_cast<spv::ExecutionModel>(binding->stage);
	bind.msl_buffer = binding->msl_buffer;
	bind.msl_texture = binding->msl_texture;
	bind.msl_sampler = binding->msl_sampler;
	msl.add_msl_resource_binding(bind);
	return SPVC_SUCCESS;
#else
	(void)binding;
	compiler->context->report_error("MSL function used on a non-MSL backend.");
	return SPVC_ERROR_INVALID_ARGUMENT;
#endif
}

spvc_result spvc_compiler_msl_add_discrete_descriptor_set(spvc_compiler compiler, unsigned desc_set)
{
#if SPIRV_CROSS_C_API_MSL
	if (compiler->backend != SPVC_BACKEND_MSL)
	{
		compiler->context->report_error("MSL function used on a non-MSL backend.");
		return SPVC_ERROR_INVALID_ARGUMENT;
	}

	auto &msl = *static_cast<CompilerMSL *>(compiler->compiler.get());
	msl.add_discrete_descriptor_set(desc_set);
	return SPVC_SUCCESS;
#else
	(void)desc_set;
	compiler->context->report_error("MSL function used on a non-MSL backend.");
	return SPVC_ERROR_INVALID_ARGUMENT;
#endif
}

spvc_bool spvc_compiler_msl_is_vertex_attribute_used(spvc_compiler compiler, unsigned location)
{
#if SPIRV_CROSS_C_API_MSL
	if (compiler->backend != SPVC_BACKEND_MSL)
	{
		compiler->context->report_error("MSL function used on a non-MSL backend.");
		return SPVC_FALSE;
	}

	auto &msl = *static_cast<CompilerMSL *>(compiler->compiler.get());
	return msl.is_msl_vertex_attribute_used(location) ? SPVC_TRUE : SPVC_FALSE;
#else
	(void)location;
	compiler->context->report_error("MSL function used on a non-MSL backend.");
	return SPVC_FALSE;
#endif
}

spvc_bool spvc_compiler_msl_is_resource_used(spvc_compiler compiler, SpvExecutionModel model, unsigned set,
                                             unsigned binding)
{
#if SPIRV_CROSS_C_API_MSL
	if (compiler->backend != SPVC_BACKEND_MSL)
	{
		compiler->context->report_error("MSL function used on a non-MSL backend.");
		return SPVC_FALSE;
	}

	auto &msl = *static_cast<CompilerMSL *>(compiler->compiler.get());
	return msl.is_msl_resource_binding_used(static_cast<spv::ExecutionModel>(model), set, binding) ? SPVC_TRUE :
	                                                                                                 SPVC_FALSE;
#else
	(void)model;
	(void)set;
	(void)binding;
	compiler->context->report_error("MSL function used on a non-MSL backend.");
	return SPVC_FALSE;
#endif
}

spvc_result spvc_compiler_msl_remap_constexpr_sampler(spvc_compiler compiler, spvc_variable_id id,
                                                      const spvc_msl_constexpr_sampler *sampler)
{
#if SPIRV_CROSS_C_API_MSL
	if (compiler->backend != SPVC_BACKEND_MSL)
	{
		compiler->context->report_error("MSL function used on a non-MSL backend.");
		return SPVC_ERROR_INVALID_ARGUMENT;
	}

	auto &msl = *static_cast<CompilerMSL *>(compiler->compiler.get());
	MSLConstexprSampler samp;
	samp.s_address = static_cast<MSLSamplerAddress>(sampler->s_address);
	samp.t_address = static_cast<MSLSamplerAddress>(sampler->t_address);
	samp.r_address = static_cast<MSLSamplerAddress>(sampler->r_address);
	samp.lod_clamp_min = sampler->lod_clamp_min;
	samp.lod_clamp_max = sampler->lod_clamp_max;
	samp.lod_clamp_enable = sampler->lod_clamp_enable;
	samp.min_filter = static_cast<MSLSamplerFilter>(sampler->min_filter);
	samp.mag_filter = static_cast<MSLSamplerFilter>(sampler->mag_filter);
	samp.mip_filter = static_cast<MSLSamplerMipFilter>(sampler->mip_filter);
	samp.compare_enable = sampler->compare_enable;
	samp.anisotropy_enable = sampler->anisotropy_enable;
	samp.max_anisotropy = sampler->max_anisotropy;
	samp.compare_func = static_cast<MSLSamplerCompareFunc>(sampler->compare_func);
	samp.coord = static_cast<MSLSamplerCoord>(sampler->coord);
	samp.border_color = static_cast<MSLSamplerBorderColor>(sampler->border_color);
	msl.remap_constexpr_sampler(id, samp);
	return SPVC_SUCCESS;
#else
	(void)id;
	(void)sampler;
	compiler->context->report_error("MSL function used on a non-MSL backend.");
	return SPVC_ERROR_INVALID_ARGUMENT;
#endif
}

spvc_result spvc_compiler_msl_set_fragment_output_components(spvc_compiler compiler, unsigned location,
                                                             unsigned components)
{
#if SPIRV_CROSS_C_API_MSL
	if (compiler->backend != SPVC_BACKEND_MSL)
	{
		compiler->context->report_error("MSL function used on a non-MSL backend.");
		return SPVC_ERROR_INVALID_ARGUMENT;
	}

	auto &msl = *static_cast<CompilerMSL *>(compiler->compiler.get());
	msl.set_fragment_output_components(location, components);
	return SPVC_SUCCESS;
#else
	(void)location;
	(void)components;
	compiler->context->report_error("MSL function used on a non-MSL backend.");
	return SPVC_ERROR_INVALID_ARGUMENT;
#endif
}

spvc_result spvc_compiler_compile(spvc_compiler compiler, const char **source)
{
	SPVC_BEGIN_SAFE_SCOPE
	{
		auto result = compiler->compiler->compile();
		if (result.empty())
		{
			compiler->context->report_error("Unsupported SPIR-V.");
			return SPVC_ERROR_UNSUPPORTED_SPIRV;
		}

		*source = compiler->context->allocate_name(result);
		if (!*source)
		{
			compiler->context->report_error("Out of memory.");
			return SPVC_ERROR_OUT_OF_MEMORY;
		}
		return SPVC_SUCCESS;
	}
	SPVC_END_SAFE_SCOPE(compiler->context, SPVC_ERROR_UNSUPPORTED_SPIRV)
}

bool spvc_resources_s::copy_resources(SmallVector<spvc_reflected_resource> &outputs,
                                      const SmallVector<Resource> &inputs)
{
	for (auto &i : inputs)
	{
		spvc_reflected_resource r;
		r.base_type_id = i.base_type_id;
		r.type_id = i.type_id;
		r.id = i.id;
		r.name = context->allocate_name(i.name);
		if (!r.name)
			return false;

		outputs.push_back(r);
	}

	return true;
}

bool spvc_resources_s::copy_resources(const ShaderResources &resources)
{
	if (!copy_resources(uniform_buffers, resources.uniform_buffers))
		return false;
	if (!copy_resources(storage_buffers, resources.storage_buffers))
		return false;
	if (!copy_resources(stage_inputs, resources.stage_inputs))
		return false;
	if (!copy_resources(stage_outputs, resources.stage_outputs))
		return false;
	if (!copy_resources(subpass_inputs, resources.subpass_inputs))
		return false;
	if (!copy_resources(storage_images, resources.storage_images))
		return false;
	if (!copy_resources(sampled_images, resources.sampled_images))
		return false;
	if (!copy_resources(atomic_counters, resources.atomic_counters))
		return false;
	if (!copy_resources(push_constant_buffers, resources.push_constant_buffers))
		return false;
	if (!copy_resources(separate_images, resources.separate_images))
		return false;
	if (!copy_resources(separate_samplers, resources.separate_samplers))
		return false;
	if (!copy_resources(acceleration_structures, resources.acceleration_structures))
		return false;

	return true;
}

spvc_result spvc_compiler_get_active_interface_variables(spvc_compiler compiler, spvc_set *set)
{
	SPVC_BEGIN_SAFE_SCOPE
	{
		std::unique_ptr<spvc_set_s> ptr(new (std::nothrow) spvc_set_s);
		if (!ptr)
		{
			compiler->context->report_error("Out of memory.");
			return SPVC_ERROR_OUT_OF_MEMORY;
		}

		auto active = compiler->compiler->get_active_interface_variables();
		ptr->set = std::move(active);
		*set = ptr.get();
		compiler->context->allocations.push_back(std::move(ptr));
	}
	SPVC_END_SAFE_SCOPE(compiler->context, SPVC_ERROR_INVALID_ARGUMENT)
	return SPVC_SUCCESS;
}

spvc_result spvc_compiler_set_enabled_interface_variables(spvc_compiler compiler, spvc_set set)
{
	SPVC_BEGIN_SAFE_SCOPE
	{
		compiler->compiler->set_enabled_interface_variables(set->set);
	}
	SPVC_END_SAFE_SCOPE(compiler->context, SPVC_ERROR_INVALID_ARGUMENT)
	return SPVC_SUCCESS;
}

spvc_result spvc_compiler_create_shader_resources_for_active_variables(spvc_compiler compiler, spvc_resources *resources,
                                                                       spvc_set set)
{
	SPVC_BEGIN_SAFE_SCOPE
	{
		std::unique_ptr<spvc_resources_s> res(new (std::nothrow) spvc_resources_s);
		if (!res)
		{
			compiler->context->report_error("Out of memory.");
			return SPVC_ERROR_OUT_OF_MEMORY;
		}

		res->context = compiler->context;
		auto accessed_resources = compiler->compiler->get_shader_resources(set->set);

		if (!res->copy_resources(accessed_resources))
		{
			res->context->report_error("Out of memory.");
			return SPVC_ERROR_OUT_OF_MEMORY;
		}
		*resources = res.get();
		compiler->context->allocations.push_back(std::move(res));
	}
	SPVC_END_SAFE_SCOPE(compiler->context, SPVC_ERROR_OUT_OF_MEMORY)
	return SPVC_SUCCESS;
}

spvc_result spvc_compiler_create_shader_resources(spvc_compiler compiler, spvc_resources *resources)
{
	SPVC_BEGIN_SAFE_SCOPE
	{
		std::unique_ptr<spvc_resources_s> res(new (std::nothrow) spvc_resources_s);
		if (!res)
		{
			compiler->context->report_error("Out of memory.");
			return SPVC_ERROR_OUT_OF_MEMORY;
		}

		res->context = compiler->context;
		auto accessed_resources = compiler->compiler->get_shader_resources();

		if (!res->copy_resources(accessed_resources))
		{
			res->context->report_error("Out of memory.");
			return SPVC_ERROR_OUT_OF_MEMORY;
		}

		*resources = res.get();
		compiler->context->allocations.push_back(std::move(res));
	}
	SPVC_END_SAFE_SCOPE(compiler->context, SPVC_ERROR_OUT_OF_MEMORY)
	return SPVC_SUCCESS;
}

spvc_result spvc_resources_get_resource_list_for_type(spvc_resources resources, spvc_resource_type type,
                                                      const spvc_reflected_resource **resource_list,
                                                      size_t *resource_size)
{
	const SmallVector<spvc_reflected_resource> *list = nullptr;
	switch (type)
	{
	case SPVC_RESOURCE_TYPE_UNIFORM_BUFFER:
		list = &resources->uniform_buffers;
		break;

	case SPVC_RESOURCE_TYPE_STORAGE_BUFFER:
		list = &resources->storage_buffers;
		break;

	case SPVC_RESOURCE_TYPE_STAGE_INPUT:
		list = &resources->stage_inputs;
		break;

	case SPVC_RESOURCE_TYPE_STAGE_OUTPUT:
		list = &resources->stage_outputs;
		break;

	case SPVC_RESOURCE_TYPE_SUBPASS_INPUT:
		list = &resources->subpass_inputs;
		break;

	case SPVC_RESOURCE_TYPE_STORAGE_IMAGE:
		list = &resources->storage_images;
		break;

	case SPVC_RESOURCE_TYPE_SAMPLED_IMAGE:
		list = &resources->sampled_images;
		break;

	case SPVC_RESOURCE_TYPE_ATOMIC_COUNTER:
		list = &resources->atomic_counters;
		break;

	case SPVC_RESOURCE_TYPE_PUSH_CONSTANT:
		list = &resources->push_constant_buffers;
		break;

	case SPVC_RESOURCE_TYPE_SEPARATE_IMAGE:
		list = &resources->separate_images;
		break;

	case SPVC_RESOURCE_TYPE_SEPARATE_SAMPLERS:
		list = &resources->separate_samplers;
		break;

	case SPVC_RESOURCE_TYPE_ACCELERATION_STRUCTURE:
		list = &resources->acceleration_structures;
		break;

	default:
		break;
	}

	if (!list)
	{
		resources->context->report_error("Invalid argument.");
		return SPVC_ERROR_INVALID_ARGUMENT;
	}

	*resource_size = list->size();
	*resource_list = list->data();
	return SPVC_SUCCESS;
}

void spvc_compiler_set_decoration(spvc_compiler compiler, SpvId id, SpvDecoration decoration, unsigned argument)
{
	compiler->compiler->set_decoration(id, static_cast<spv::Decoration>(decoration), argument);
}

void spvc_compiler_set_decoration_string(spvc_compiler compiler, SpvId id, SpvDecoration decoration,
                                         const char *argument)
{
	compiler->compiler->set_decoration_string(id, static_cast<spv::Decoration>(decoration), argument);
}

void spvc_compiler_set_name(spvc_compiler compiler, SpvId id, const char *argument)
{
	compiler->compiler->set_name(id, argument);
}

void spvc_compiler_set_member_decoration(spvc_compiler compiler, spvc_type_id id, unsigned member_index,
                                         SpvDecoration decoration, unsigned argument)
{
	compiler->compiler->set_member_decoration(id, member_index, static_cast<spv::Decoration>(decoration), argument);
}

void spvc_compiler_set_member_decoration_string(spvc_compiler compiler, spvc_type_id id, unsigned member_index,
                                                SpvDecoration decoration, const char *argument)
{
	compiler->compiler->set_member_decoration_string(id, member_index, static_cast<spv::Decoration>(decoration),
	                                                 argument);
}

void spvc_compiler_set_member_name(spvc_compiler compiler, spvc_type_id id, unsigned member_index, const char *argument)
{
	compiler->compiler->set_member_name(id, member_index, argument);
}

void spvc_compiler_unset_decoration(spvc_compiler compiler, SpvId id, SpvDecoration decoration)
{
	compiler->compiler->unset_decoration(id, static_cast<spv::Decoration>(decoration));
}

void spvc_compiler_unset_member_decoration(spvc_compiler compiler, spvc_type_id id, unsigned member_index,
                                           SpvDecoration decoration)
{
	compiler->compiler->unset_member_decoration(id, member_index, static_cast<spv::Decoration>(decoration));
}

spvc_bool spvc_compiler_has_decoration(spvc_compiler compiler, SpvId id, SpvDecoration decoration)
{
	return compiler->compiler->has_decoration(id, static_cast<spv::Decoration>(decoration)) ? SPVC_TRUE : SPVC_FALSE;
}

spvc_bool spvc_compiler_has_member_decoration(spvc_compiler compiler, spvc_type_id id, unsigned member_index,
                                              SpvDecoration decoration)
{
	return compiler->compiler->has_member_decoration(id, member_index, static_cast<spv::Decoration>(decoration)) ?
	           SPVC_TRUE :
	           SPVC_FALSE;
}

const char *spvc_compiler_get_name(spvc_compiler compiler, SpvId id)
{
	return compiler->compiler->get_name(id).c_str();
}

unsigned spvc_compiler_get_decoration(spvc_compiler compiler, SpvId id, SpvDecoration decoration)
{
	return compiler->compiler->get_decoration(id, static_cast<spv::Decoration>(decoration));
}

const char *spvc_compiler_get_decoration_string(spvc_compiler compiler, SpvId id, SpvDecoration decoration)
{
	return compiler->compiler->get_decoration_string(id, static_cast<spv::Decoration>(decoration)).c_str();
}

unsigned spvc_compiler_get_member_decoration(spvc_compiler compiler, spvc_type_id id, unsigned member_index,
                                             SpvDecoration decoration)
{
	return compiler->compiler->get_member_decoration(id, member_index, static_cast<spv::Decoration>(decoration));
}

const char *spvc_compiler_get_member_decoration_string(spvc_compiler compiler, spvc_type_id id, unsigned member_index,
                                                       SpvDecoration decoration)
{
	return compiler->compiler->get_member_decoration_string(id, member_index, static_cast<spv::Decoration>(decoration))
	    .c_str();
}

const char *spvc_compiler_get_member_name(spvc_compiler compiler, spvc_type_id id, unsigned member_index)
{
	return compiler->compiler->get_member_name(id, member_index).c_str();
}

spvc_result spvc_compiler_get_entry_points(spvc_compiler compiler, const spvc_entry_point **entry_points,
                                           size_t *num_entry_points)
{
	SPVC_BEGIN_SAFE_SCOPE
	{
		auto entries = compiler->compiler->get_entry_points_and_stages();
		SmallVector<spvc_entry_point> translated;
		translated.reserve(entries.size());

		for (auto &entry : entries)
		{
			spvc_entry_point new_entry;
			new_entry.execution_model = static_cast<SpvExecutionModel>(entry.execution_model);
			new_entry.name = compiler->context->allocate_name(entry.name);
			if (!new_entry.name)
			{
				compiler->context->report_error("Out of memory.");
				return SPVC_ERROR_OUT_OF_MEMORY;
			}
			translated.push_back(new_entry);
		}

		auto ptr = spvc_allocate<TemporaryBuffer<spvc_entry_point>>();
		ptr->buffer = std::move(translated);
		*entry_points = ptr->buffer.data();
		*num_entry_points = ptr->buffer.size();
		compiler->context->allocations.push_back(std::move(ptr));
	}
	SPVC_END_SAFE_SCOPE(compiler->context, SPVC_ERROR_OUT_OF_MEMORY)
	return SPVC_SUCCESS;
}

spvc_result spvc_compiler_set_entry_point(spvc_compiler compiler, const char *name, SpvExecutionModel model)
{
	compiler->compiler->set_entry_point(name, static_cast<spv::ExecutionModel>(model));
	return SPVC_SUCCESS;
}

spvc_result spvc_compiler_rename_entry_point(spvc_compiler compiler, const char *old_name, const char *new_name,
                                             SpvExecutionModel model)
{
	SPVC_BEGIN_SAFE_SCOPE
	{
		compiler->compiler->rename_entry_point(old_name, new_name, static_cast<spv::ExecutionModel>(model));
	}
	SPVC_END_SAFE_SCOPE(compiler->context, SPVC_ERROR_INVALID_ARGUMENT)
	return SPVC_SUCCESS;
}

const char *spvc_compiler_get_cleansed_entry_point_name(spvc_compiler compiler, const char *name,
                                                        SpvExecutionModel model)
{
	SPVC_BEGIN_SAFE_SCOPE
	{
		auto cleansed_name =
		    compiler->compiler->get_cleansed_entry_point_name(name, static_cast<spv::ExecutionModel>(model));
		return compiler->context->allocate_name(cleansed_name);
	}
	SPVC_END_SAFE_SCOPE(compiler->context, nullptr)
}

void spvc_compiler_set_execution_mode(spvc_compiler compiler, SpvExecutionMode mode)
{
	compiler->compiler->set_execution_mode(static_cast<spv::ExecutionMode>(mode));
}

void spvc_compiler_set_execution_mode_with_arguments(spvc_compiler compiler, SpvExecutionMode mode, unsigned arg0,
                                                     unsigned arg1,
                                                     unsigned arg2)
{
	compiler->compiler->set_execution_mode(static_cast<spv::ExecutionMode>(mode), arg0, arg1, arg2);
}

void spvc_compiler_unset_execution_mode(spvc_compiler compiler, SpvExecutionMode mode)
{
	compiler->compiler->unset_execution_mode(static_cast<spv::ExecutionMode>(mode));
}

spvc_result spvc_compiler_get_execution_modes(spvc_compiler compiler, const SpvExecutionMode **modes, size_t *num_modes)
{
	SPVC_BEGIN_SAFE_SCOPE
	{
		auto ptr = spvc_allocate<TemporaryBuffer<SpvExecutionMode>>();

		compiler->compiler->get_execution_mode_bitset().for_each_bit(
		    [&](uint32_t bit) { ptr->buffer.push_back(static_cast<SpvExecutionMode>(bit)); });

		*modes = ptr->buffer.data();
		*num_modes = ptr->buffer.size();
		compiler->context->allocations.push_back(std::move(ptr));
	}
	SPVC_END_SAFE_SCOPE(compiler->context, SPVC_ERROR_OUT_OF_MEMORY)
	return SPVC_SUCCESS;
}

unsigned spvc_compiler_get_execution_mode_argument(spvc_compiler compiler, SpvExecutionMode mode)
{
	return compiler->compiler->get_execution_mode_argument(static_cast<spv::ExecutionMode>(mode));
}

unsigned spvc_compiler_get_execution_mode_argument_by_index(spvc_compiler compiler, SpvExecutionMode mode,
                                                            unsigned index)
{
	return compiler->compiler->get_execution_mode_argument(static_cast<spv::ExecutionMode>(mode), index);
}

SpvExecutionModel spvc_compiler_get_execution_model(spvc_compiler compiler)
{
	return static_cast<SpvExecutionModel>(compiler->compiler->get_execution_model());
}

spvc_type spvc_compiler_get_type_handle(spvc_compiler compiler, spvc_type_id id)
{
	// Should only throw if an intentionally garbage ID is passed, but the IDs are not type-safe.
	SPVC_BEGIN_SAFE_SCOPE
	{
		return static_cast<spvc_type>(&compiler->compiler->get_type(id));
	}
	SPVC_END_SAFE_SCOPE(compiler->context, nullptr)
}

static spvc_basetype convert_basetype(SPIRType::BaseType type)
{
	// For now the enums match up.
	return static_cast<spvc_basetype>(type);
}

spvc_basetype spvc_type_get_basetype(spvc_type type)
{
	return convert_basetype(type->basetype);
}

unsigned spvc_type_get_bit_width(spvc_type type)
{
	return type->width;
}

unsigned spvc_type_get_vector_size(spvc_type type)
{
	return type->vecsize;
}

unsigned spvc_type_get_columns(spvc_type type)
{
	return type->columns;
}

unsigned spvc_type_get_num_array_dimensions(spvc_type type)
{
	return unsigned(type->array.size());
}

spvc_bool spvc_type_array_dimension_is_literal(spvc_type type, unsigned dimension)
{
	return type->array_size_literal[dimension] ? SPVC_TRUE : SPVC_FALSE;
}

SpvId spvc_type_get_array_dimension(spvc_type type, unsigned dimension)
{
	return type->array[dimension];
}

unsigned spvc_type_get_num_member_types(spvc_type type)
{
	return unsigned(type->member_types.size());
}

spvc_type_id spvc_type_get_member_type(spvc_type type, unsigned index)
{
	return type->member_types[index];
}

SpvStorageClass spvc_type_get_storage_class(spvc_type type)
{
	return static_cast<SpvStorageClass>(type->storage);
}

// Image type query.
spvc_type_id spvc_type_get_image_sampled_type(spvc_type type)
{
	return type->image.type;
}

SpvDim spvc_type_get_image_dimension(spvc_type type)
{
	return static_cast<SpvDim>(type->image.dim);
}

spvc_bool spvc_type_get_image_is_depth(spvc_type type)
{
	return type->image.depth ? SPVC_TRUE : SPVC_FALSE;
}

spvc_bool spvc_type_get_image_arrayed(spvc_type type)
{
	return type->image.arrayed ? SPVC_TRUE : SPVC_FALSE;
}

spvc_bool spvc_type_get_image_multisampled(spvc_type type)
{
	return type->image.ms ? SPVC_TRUE : SPVC_FALSE;
}

spvc_bool spvc_type_get_image_is_storage(spvc_type type)
{
	return type->image.sampled == 2 ? SPVC_TRUE : SPVC_FALSE;
}

SpvImageFormat spvc_type_get_image_storage_format(spvc_type type)
{
	return static_cast<SpvImageFormat>(static_cast<const SPIRType *>(type)->image.format);
}

SpvAccessQualifier spvc_type_get_image_access_qualifier(spvc_type type)
{
	return static_cast<SpvAccessQualifier>(static_cast<const SPIRType *>(type)->image.access);
}

spvc_result spvc_compiler_get_declared_struct_size(spvc_compiler compiler, spvc_type struct_type, size_t *size)
{
	SPVC_BEGIN_SAFE_SCOPE
	{
		*size = compiler->compiler->get_declared_struct_size(*static_cast<const SPIRType *>(struct_type));
	}
	SPVC_END_SAFE_SCOPE(compiler->context, SPVC_ERROR_INVALID_ARGUMENT)
	return SPVC_SUCCESS;
}

spvc_result spvc_compiler_get_declared_struct_size_runtime_array(spvc_compiler compiler, spvc_type struct_type,
                                                                 size_t array_size, size_t *size)
{
	SPVC_BEGIN_SAFE_SCOPE
	{
		*size = compiler->compiler->get_declared_struct_size_runtime_array(*static_cast<const SPIRType *>(struct_type),
		                                                                   array_size);
	}
	SPVC_END_SAFE_SCOPE(compiler->context, SPVC_ERROR_INVALID_ARGUMENT)
	return SPVC_SUCCESS;
}

spvc_result spvc_compiler_type_struct_member_offset(spvc_compiler compiler, spvc_type type, unsigned index, unsigned *offset)
{
	SPVC_BEGIN_SAFE_SCOPE
	{
		*offset = compiler->compiler->type_struct_member_offset(*static_cast<const SPIRType *>(type), index);
	}
	SPVC_END_SAFE_SCOPE(compiler->context, SPVC_ERROR_INVALID_ARGUMENT)
	return SPVC_SUCCESS;
}

spvc_result spvc_compiler_type_struct_member_array_stride(spvc_compiler compiler, spvc_type type, unsigned index, unsigned *stride)
{
	SPVC_BEGIN_SAFE_SCOPE
	{
		*stride = compiler->compiler->type_struct_member_array_stride(*static_cast<const SPIRType *>(type), index);
	}
	SPVC_END_SAFE_SCOPE(compiler->context, SPVC_ERROR_INVALID_ARGUMENT)
	return SPVC_SUCCESS;
}

spvc_result spvc_compiler_type_struct_member_matrix_stride(spvc_compiler compiler, spvc_type type, unsigned index, unsigned *stride)
{
	SPVC_BEGIN_SAFE_SCOPE
	{
		*stride = compiler->compiler->type_struct_member_matrix_stride(*static_cast<const SPIRType *>(type), index);
	}
	SPVC_END_SAFE_SCOPE(compiler->context, SPVC_ERROR_INVALID_ARGUMENT)
	return SPVC_SUCCESS;
}

spvc_result spvc_compiler_build_dummy_sampler_for_combined_images(spvc_compiler compiler, spvc_variable_id *id)
{
	SPVC_BEGIN_SAFE_SCOPE
	{
		*id = compiler->compiler->build_dummy_sampler_for_combined_images();
	}
	SPVC_END_SAFE_SCOPE(compiler->context, SPVC_ERROR_INVALID_ARGUMENT)
	return SPVC_SUCCESS;
}

spvc_result spvc_compiler_build_combined_image_samplers(spvc_compiler compiler)
{
	SPVC_BEGIN_SAFE_SCOPE
	{
		compiler->compiler->build_combined_image_samplers();
	}
	SPVC_END_SAFE_SCOPE(compiler->context, SPVC_ERROR_UNSUPPORTED_SPIRV)
	return SPVC_SUCCESS;
}

spvc_result spvc_compiler_get_combined_image_samplers(spvc_compiler compiler,
                                                      const spvc_combined_image_sampler **samplers,
                                                      size_t *num_samplers)
{
	SPVC_BEGIN_SAFE_SCOPE
	{
		auto combined = compiler->compiler->get_combined_image_samplers();
		SmallVector<spvc_combined_image_sampler> translated;
		translated.reserve(combined.size());
		for (auto &c : combined)
		{
			spvc_combined_image_sampler trans = { c.combined_id, c.image_id, c.sampler_id };
			translated.push_back(trans);
		}

		auto ptr = spvc_allocate<TemporaryBuffer<spvc_combined_image_sampler>>();
		ptr->buffer = std::move(translated);
		*samplers = ptr->buffer.data();
		*num_samplers = ptr->buffer.size();
		compiler->context->allocations.push_back(std::move(ptr));
	}
	SPVC_END_SAFE_SCOPE(compiler->context, SPVC_ERROR_OUT_OF_MEMORY)
	return SPVC_SUCCESS;
}

spvc_result spvc_compiler_get_specialization_constants(spvc_compiler compiler,
                                                       const spvc_specialization_constant **constants,
                                                       size_t *num_constants)
{
	SPVC_BEGIN_SAFE_SCOPE
	{
		auto spec_constants = compiler->compiler->get_specialization_constants();
		SmallVector<spvc_specialization_constant> translated;
		translated.reserve(spec_constants.size());
		for (auto &c : spec_constants)
		{
			spvc_specialization_constant trans = { c.id, c.constant_id };
			translated.push_back(trans);
		}

		auto ptr = spvc_allocate<TemporaryBuffer<spvc_specialization_constant>>();
		ptr->buffer = std::move(translated);
		*constants = ptr->buffer.data();
		*num_constants = ptr->buffer.size();
		compiler->context->allocations.push_back(std::move(ptr));
	}
	SPVC_END_SAFE_SCOPE(compiler->context, SPVC_ERROR_OUT_OF_MEMORY)
	return SPVC_SUCCESS;
}

spvc_constant spvc_compiler_get_constant_handle(spvc_compiler compiler, spvc_variable_id id)
{
	SPVC_BEGIN_SAFE_SCOPE
	{
		return static_cast<spvc_constant>(&compiler->compiler->get_constant(id));
	}
	SPVC_END_SAFE_SCOPE(compiler->context, nullptr)
}

spvc_constant_id spvc_compiler_get_work_group_size_specialization_constants(spvc_compiler compiler,
                                                                            spvc_specialization_constant *x,
                                                                            spvc_specialization_constant *y,
                                                                            spvc_specialization_constant *z)
{
	SpecializationConstant tmpx;
	SpecializationConstant tmpy;
	SpecializationConstant tmpz;
	spvc_constant_id ret = compiler->compiler->get_work_group_size_specialization_constants(tmpx, tmpy, tmpz);
	x->id = tmpx.id;
	x->constant_id = tmpx.constant_id;
	y->id = tmpy.id;
	y->constant_id = tmpy.constant_id;
	z->id = tmpz.id;
	z->constant_id = tmpz.constant_id;
	return ret;
}

spvc_result spvc_compiler_get_active_buffer_ranges(spvc_compiler compiler,
                                                   spvc_variable_id id,
                                                   const spvc_buffer_range **ranges,
                                                   size_t *num_ranges)
{
	SPVC_BEGIN_SAFE_SCOPE
	{
		auto active_ranges = compiler->compiler->get_active_buffer_ranges(id);
		SmallVector<spvc_buffer_range> translated;
		translated.reserve(active_ranges.size());
		for (auto &r : active_ranges)
		{
			spvc_buffer_range trans = { r.index, r.offset, r.range };
			translated.push_back(trans);
		}

		auto ptr = spvc_allocate<TemporaryBuffer<spvc_buffer_range>>();
		ptr->buffer = std::move(translated);
		*ranges = ptr->buffer.data();
		*num_ranges = ptr->buffer.size();
		compiler->context->allocations.push_back(std::move(ptr));
	}
	SPVC_END_SAFE_SCOPE(compiler->context, SPVC_ERROR_OUT_OF_MEMORY)
	return SPVC_SUCCESS;
}

float spvc_constant_get_scalar_fp16(spvc_constant constant, unsigned column, unsigned row)
{
	return constant->scalar_f16(column, row);
}

float spvc_constant_get_scalar_fp32(spvc_constant constant, unsigned column, unsigned row)
{
	return constant->scalar_f32(column, row);
}

double spvc_constant_get_scalar_fp64(spvc_constant constant, unsigned column, unsigned row)
{
	return constant->scalar_f64(column, row);
}

unsigned spvc_constant_get_scalar_u32(spvc_constant constant, unsigned column, unsigned row)
{
	return constant->scalar(column, row);
}

int spvc_constant_get_scalar_i32(spvc_constant constant, unsigned column, unsigned row)
{
	return constant->scalar_i32(column, row);
}

unsigned spvc_constant_get_scalar_u16(spvc_constant constant, unsigned column, unsigned row)
{
	return constant->scalar_u16(column, row);
}

int spvc_constant_get_scalar_i16(spvc_constant constant, unsigned column, unsigned row)
{
	return constant->scalar_i16(column, row);
}

unsigned spvc_constant_get_scalar_u8(spvc_constant constant, unsigned column, unsigned row)
{
	return constant->scalar_u8(column, row);
}

int spvc_constant_get_scalar_i8(spvc_constant constant, unsigned column, unsigned row)
{
	return constant->scalar_i8(column, row);
}

void spvc_constant_get_subconstants(spvc_constant constant, const spvc_constant_id **constituents, size_t *count)
{
	static_assert(sizeof(spvc_constant_id) == sizeof(constant->subconstants.front()), "ID size is not consistent.");
	*constituents = reinterpret_cast<spvc_constant_id *>(constant->subconstants.data());
	*count = constant->subconstants.size();
}

spvc_type_id spvc_constant_get_type(spvc_constant constant)
{
	return constant->constant_type;
}

spvc_bool spvc_compiler_get_binary_offset_for_decoration(spvc_compiler compiler, spvc_variable_id id,
                                                         SpvDecoration decoration,
                                                         unsigned *word_offset)
{
	uint32_t off = 0;
	bool ret = compiler->compiler->get_binary_offset_for_decoration(id, static_cast<spv::Decoration>(decoration), off);
	if (ret)
	{
		*word_offset = off;
		return SPVC_TRUE;
	}
	else
		return SPVC_FALSE;
}

spvc_bool spvc_compiler_buffer_is_hlsl_counter_buffer(spvc_compiler compiler, spvc_variable_id id)
{
	return compiler->compiler->buffer_is_hlsl_counter_buffer(id) ? SPVC_TRUE : SPVC_FALSE;
}

spvc_bool spvc_compiler_buffer_get_hlsl_counter_buffer(spvc_compiler compiler, spvc_variable_id id,
                                                       spvc_variable_id *counter_id)
{
	uint32_t buffer;
	bool ret = compiler->compiler->buffer_get_hlsl_counter_buffer(id, buffer);
	if (ret)
	{
		*counter_id = buffer;
		return SPVC_TRUE;
	}
	else
		return SPVC_FALSE;
}

spvc_result spvc_compiler_get_declared_capabilities(spvc_compiler compiler, const SpvCapability **capabilities,
                                                    size_t *num_capabilities)
{
	auto &caps = compiler->compiler->get_declared_capabilities();
	static_assert(sizeof(SpvCapability) == sizeof(spv::Capability), "Enum size mismatch.");
	*capabilities = reinterpret_cast<const SpvCapability *>(caps.data());
	*num_capabilities = caps.size();
	return SPVC_SUCCESS;
}

spvc_result spvc_compiler_get_declared_extensions(spvc_compiler compiler, const char ***extensions,
                                                  size_t *num_extensions)
{
	SPVC_BEGIN_SAFE_SCOPE
	{
		auto &exts = compiler->compiler->get_declared_extensions();
		SmallVector<const char *> duped;
		duped.reserve(exts.size());
		for (auto &ext : exts)
			duped.push_back(compiler->context->allocate_name(ext));

		auto ptr = spvc_allocate<TemporaryBuffer<const char *>>();
		ptr->buffer = std::move(duped);
		*extensions = ptr->buffer.data();
		*num_extensions = ptr->buffer.size();
		compiler->context->allocations.push_back(std::move(ptr));
	}
	SPVC_END_SAFE_SCOPE(compiler->context, SPVC_ERROR_OUT_OF_MEMORY)
	return SPVC_SUCCESS;
}

const char *spvc_compiler_get_remapped_declared_block_name(spvc_compiler compiler, spvc_variable_id id)
{
	SPVC_BEGIN_SAFE_SCOPE
	{
		auto name = compiler->compiler->get_remapped_declared_block_name(id);
		return compiler->context->allocate_name(name);
	}
	SPVC_END_SAFE_SCOPE(compiler->context, nullptr)
}

spvc_result spvc_compiler_get_buffer_block_decorations(spvc_compiler compiler, spvc_variable_id id,
                                                       const SpvDecoration **decorations, size_t *num_decorations)
{
	SPVC_BEGIN_SAFE_SCOPE
	{
		auto flags = compiler->compiler->get_buffer_block_flags(id);
		auto bitset = spvc_allocate<TemporaryBuffer<SpvDecoration>>();

		flags.for_each_bit([&](uint32_t bit) { bitset->buffer.push_back(static_cast<SpvDecoration>(bit)); });

		*decorations = bitset->buffer.data();
		*num_decorations = bitset->buffer.size();
		compiler->context->allocations.push_back(std::move(bitset));
	}
	SPVC_END_SAFE_SCOPE(compiler->context, SPVC_ERROR_INVALID_ARGUMENT)
	return SPVC_SUCCESS;
}

unsigned spvc_msl_get_aux_buffer_struct_version(void)
{
	return SPVC_MSL_AUX_BUFFER_STRUCT_VERSION;
}

void spvc_msl_vertex_attribute_init(spvc_msl_vertex_attribute *attr)
{
#if SPIRV_CROSS_C_API_MSL
	// Crude, but works.
	MSLVertexAttr attr_default;
	attr->location = attr_default.location;
	attr->per_instance = attr_default.per_instance ? SPVC_TRUE : SPVC_FALSE;
	attr->format = static_cast<spvc_msl_vertex_format>(attr_default.format);
	attr->builtin = static_cast<SpvBuiltIn>(attr_default.builtin);
	attr->msl_buffer = attr_default.msl_buffer;
	attr->msl_offset = attr_default.msl_offset;
	attr->msl_stride = attr_default.msl_stride;
#else
	memset(attr, 0, sizeof(*attr));
#endif
}

void spvc_msl_resource_binding_init(spvc_msl_resource_binding *binding)
{
#if SPIRV_CROSS_C_API_MSL
	MSLResourceBinding binding_default;
	binding->desc_set = binding_default.desc_set;
	binding->binding = binding_default.binding;
	binding->msl_buffer = binding_default.msl_buffer;
	binding->msl_texture = binding_default.msl_texture;
	binding->msl_sampler = binding_default.msl_sampler;
	binding->stage = static_cast<SpvExecutionModel>(binding_default.stage);
#else
	memset(binding, 0, sizeof(*binding));
#endif
}

void spvc_msl_constexpr_sampler_init(spvc_msl_constexpr_sampler *sampler)
{
#if SPIRV_CROSS_C_API_MSL
	MSLConstexprSampler defaults;
	sampler->anisotropy_enable = defaults.anisotropy_enable ? SPVC_TRUE : SPVC_FALSE;
	sampler->border_color = static_cast<spvc_msl_sampler_border_color>(defaults.border_color);
	sampler->compare_enable = defaults.compare_enable ? SPVC_TRUE : SPVC_FALSE;
	sampler->coord = static_cast<spvc_msl_sampler_coord>(defaults.coord);
	sampler->compare_func = static_cast<spvc_msl_sampler_compare_func>(defaults.compare_func);
	sampler->lod_clamp_enable = defaults.lod_clamp_enable ? SPVC_TRUE : SPVC_FALSE;
	sampler->lod_clamp_max = defaults.lod_clamp_max;
	sampler->lod_clamp_min = defaults.lod_clamp_min;
	sampler->mag_filter = static_cast<spvc_msl_sampler_filter>(defaults.mag_filter);
	sampler->min_filter = static_cast<spvc_msl_sampler_filter>(defaults.min_filter);
	sampler->mip_filter = static_cast<spvc_msl_sampler_mip_filter>(defaults.mip_filter);
	sampler->max_anisotropy = defaults.max_anisotropy;
	sampler->s_address = static_cast<spvc_msl_sampler_address>(defaults.s_address);
	sampler->t_address = static_cast<spvc_msl_sampler_address>(defaults.t_address);
	sampler->r_address = static_cast<spvc_msl_sampler_address>(defaults.r_address);
#else
	memset(sampler, 0, sizeof(*sampler));
#endif
}

unsigned spvc_compiler_get_current_id_bound(spvc_compiler compiler)
{
	return compiler->compiler->get_current_id_bound();
}

void spvc_get_version(unsigned *major, unsigned *minor, unsigned *patch)
{
	*major = SPVC_C_API_VERSION_MAJOR;
	*minor = SPVC_C_API_VERSION_MINOR;
	*patch = SPVC_C_API_VERSION_PATCH;
}

#ifdef _MSC_VER
#pragma warning(pop)
#endif
