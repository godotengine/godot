/*
 * Copyright 2015-2020 Arm Limited
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

#include "spirv_glsl.hpp"
#include "GLSL.std.450.h"
#include "spirv_common.hpp"
#include <algorithm>
#include <assert.h>
#include <cmath>
#include <limits>
#include <locale.h>
#include <utility>

#ifndef _WIN32
#include <langinfo.h>
#endif
#include <locale.h>

using namespace spv;
using namespace SPIRV_CROSS_NAMESPACE;
using namespace std;

static bool is_unsigned_opcode(Op op)
{
	// Don't have to be exhaustive, only relevant for legacy target checking ...
	switch (op)
	{
	case OpShiftRightLogical:
	case OpUGreaterThan:
	case OpUGreaterThanEqual:
	case OpULessThan:
	case OpULessThanEqual:
	case OpUConvert:
	case OpUDiv:
	case OpUMod:
	case OpUMulExtended:
	case OpConvertUToF:
	case OpConvertFToU:
		return true;

	default:
		return false;
	}
}

static bool is_unsigned_glsl_opcode(GLSLstd450 op)
{
	// Don't have to be exhaustive, only relevant for legacy target checking ...
	switch (op)
	{
	case GLSLstd450UClamp:
	case GLSLstd450UMin:
	case GLSLstd450UMax:
	case GLSLstd450FindUMsb:
		return true;

	default:
		return false;
	}
}

static bool packing_is_vec4_padded(BufferPackingStandard packing)
{
	switch (packing)
	{
	case BufferPackingHLSLCbuffer:
	case BufferPackingHLSLCbufferPackOffset:
	case BufferPackingStd140:
	case BufferPackingStd140EnhancedLayout:
		return true;

	default:
		return false;
	}
}

static bool packing_is_hlsl(BufferPackingStandard packing)
{
	switch (packing)
	{
	case BufferPackingHLSLCbuffer:
	case BufferPackingHLSLCbufferPackOffset:
		return true;

	default:
		return false;
	}
}

static bool packing_has_flexible_offset(BufferPackingStandard packing)
{
	switch (packing)
	{
	case BufferPackingStd140:
	case BufferPackingStd430:
	case BufferPackingScalar:
	case BufferPackingHLSLCbuffer:
		return false;

	default:
		return true;
	}
}

static bool packing_is_scalar(BufferPackingStandard packing)
{
	switch (packing)
	{
	case BufferPackingScalar:
	case BufferPackingScalarEnhancedLayout:
		return true;

	default:
		return false;
	}
}

static BufferPackingStandard packing_to_substruct_packing(BufferPackingStandard packing)
{
	switch (packing)
	{
	case BufferPackingStd140EnhancedLayout:
		return BufferPackingStd140;
	case BufferPackingStd430EnhancedLayout:
		return BufferPackingStd430;
	case BufferPackingHLSLCbufferPackOffset:
		return BufferPackingHLSLCbuffer;
	case BufferPackingScalarEnhancedLayout:
		return BufferPackingScalar;
	default:
		return packing;
	}
}

// Sanitizes underscores for GLSL where multiple underscores in a row are not allowed.
string CompilerGLSL::sanitize_underscores(const string &str)
{
	string res;
	res.reserve(str.size());

	bool last_underscore = false;
	for (auto c : str)
	{
		if (c == '_')
		{
			if (last_underscore)
				continue;

			res += c;
			last_underscore = true;
		}
		else
		{
			res += c;
			last_underscore = false;
		}
	}
	return res;
}

void CompilerGLSL::init()
{
	if (ir.source.known)
	{
		options.es = ir.source.es;
		options.version = ir.source.version;
	}

	// Query the locale to see what the decimal point is.
	// We'll rely on fixing it up ourselves in the rare case we have a comma-as-decimal locale
	// rather than setting locales ourselves. Settings locales in a safe and isolated way is rather
	// tricky.
#ifdef _WIN32
	// On Windows, localeconv uses thread-local storage, so it should be fine.
	const struct lconv *conv = localeconv();
	if (conv && conv->decimal_point)
		current_locale_radix_character = *conv->decimal_point;
#elif defined(__ANDROID__) && __ANDROID_API__ < 26
	// nl_langinfo is not supported on this platform, fall back to the worse alternative.
	const struct lconv *conv = localeconv();
	if (conv && conv->decimal_point)
		current_locale_radix_character = *conv->decimal_point;
#else
	// localeconv, the portable function is not MT safe ...
	const char *decimal_point = nl_langinfo(RADIXCHAR);
	if (decimal_point && *decimal_point != '\0')
		current_locale_radix_character = *decimal_point;
#endif
}

static const char *to_pls_layout(PlsFormat format)
{
	switch (format)
	{
	case PlsR11FG11FB10F:
		return "layout(r11f_g11f_b10f) ";
	case PlsR32F:
		return "layout(r32f) ";
	case PlsRG16F:
		return "layout(rg16f) ";
	case PlsRGB10A2:
		return "layout(rgb10_a2) ";
	case PlsRGBA8:
		return "layout(rgba8) ";
	case PlsRG16:
		return "layout(rg16) ";
	case PlsRGBA8I:
		return "layout(rgba8i)";
	case PlsRG16I:
		return "layout(rg16i) ";
	case PlsRGB10A2UI:
		return "layout(rgb10_a2ui) ";
	case PlsRGBA8UI:
		return "layout(rgba8ui) ";
	case PlsRG16UI:
		return "layout(rg16ui) ";
	case PlsR32UI:
		return "layout(r32ui) ";
	default:
		return "";
	}
}

static SPIRType::BaseType pls_format_to_basetype(PlsFormat format)
{
	switch (format)
	{
	default:
	case PlsR11FG11FB10F:
	case PlsR32F:
	case PlsRG16F:
	case PlsRGB10A2:
	case PlsRGBA8:
	case PlsRG16:
		return SPIRType::Float;

	case PlsRGBA8I:
	case PlsRG16I:
		return SPIRType::Int;

	case PlsRGB10A2UI:
	case PlsRGBA8UI:
	case PlsRG16UI:
	case PlsR32UI:
		return SPIRType::UInt;
	}
}

static uint32_t pls_format_to_components(PlsFormat format)
{
	switch (format)
	{
	default:
	case PlsR32F:
	case PlsR32UI:
		return 1;

	case PlsRG16F:
	case PlsRG16:
	case PlsRG16UI:
	case PlsRG16I:
		return 2;

	case PlsR11FG11FB10F:
		return 3;

	case PlsRGB10A2:
	case PlsRGBA8:
	case PlsRGBA8I:
	case PlsRGB10A2UI:
	case PlsRGBA8UI:
		return 4;
	}
}

const char *CompilerGLSL::vector_swizzle(int vecsize, int index)
{
	static const char *const swizzle[4][4] = {
		{ ".x", ".y", ".z", ".w" },
		{ ".xy", ".yz", ".zw", nullptr },
		{ ".xyz", ".yzw", nullptr, nullptr },
#if defined(__GNUC__) && (__GNUC__ == 9)
		// This works around a GCC 9 bug, see details in https://gcc.gnu.org/bugzilla/show_bug.cgi?id=90947.
		// This array ends up being compiled as all nullptrs, tripping the assertions below.
		{ "", nullptr, nullptr, "$" },
#else
		{ "", nullptr, nullptr, nullptr },
#endif
	};

	assert(vecsize >= 1 && vecsize <= 4);
	assert(index >= 0 && index < 4);
	assert(swizzle[vecsize - 1][index]);

	return swizzle[vecsize - 1][index];
}

void CompilerGLSL::reset()
{
	// We do some speculative optimizations which should pretty much always work out,
	// but just in case the SPIR-V is rather weird, recompile until it's happy.
	// This typically only means one extra pass.
	clear_force_recompile();

	// Clear invalid expression tracking.
	invalid_expressions.clear();
	current_function = nullptr;

	// Clear temporary usage tracking.
	expression_usage_counts.clear();
	forwarded_temporaries.clear();
	suppressed_usage_tracking.clear();

	// Ensure that we declare phi-variable copies even if the original declaration isn't deferred
	flushed_phi_variables.clear();

	reset_name_caches();

	ir.for_each_typed_id<SPIRFunction>([&](uint32_t, SPIRFunction &func) {
		func.active = false;
		func.flush_undeclared = true;
	});

	ir.for_each_typed_id<SPIRVariable>([&](uint32_t, SPIRVariable &var) { var.dependees.clear(); });

	ir.reset_all_of_type<SPIRExpression>();
	ir.reset_all_of_type<SPIRAccessChain>();

	statement_count = 0;
	indent = 0;
}

void CompilerGLSL::remap_pls_variables()
{
	for (auto &input : pls_inputs)
	{
		auto &var = get<SPIRVariable>(input.id);

		bool input_is_target = false;
		if (var.storage == StorageClassUniformConstant)
		{
			auto &type = get<SPIRType>(var.basetype);
			input_is_target = type.image.dim == DimSubpassData;
		}

		if (var.storage != StorageClassInput && !input_is_target)
			SPIRV_CROSS_THROW("Can only use in and target variables for PLS inputs.");
		var.remapped_variable = true;
	}

	for (auto &output : pls_outputs)
	{
		auto &var = get<SPIRVariable>(output.id);
		if (var.storage != StorageClassOutput)
			SPIRV_CROSS_THROW("Can only use out variables for PLS outputs.");
		var.remapped_variable = true;
	}
}

void CompilerGLSL::remap_ext_framebuffer_fetch(uint32_t input_attachment_index, uint32_t color_location)
{
	subpass_to_framebuffer_fetch_attachment.push_back({ input_attachment_index, color_location });
	inout_color_attachments.insert(color_location);
}

void CompilerGLSL::find_static_extensions()
{
	ir.for_each_typed_id<SPIRType>([&](uint32_t, const SPIRType &type) {
		if (type.basetype == SPIRType::Double)
		{
			if (options.es)
				SPIRV_CROSS_THROW("FP64 not supported in ES profile.");
			if (!options.es && options.version < 400)
				require_extension_internal("GL_ARB_gpu_shader_fp64");
		}
		else if (type.basetype == SPIRType::Int64 || type.basetype == SPIRType::UInt64)
		{
			if (options.es)
				SPIRV_CROSS_THROW("64-bit integers not supported in ES profile.");
			if (!options.es)
				require_extension_internal("GL_ARB_gpu_shader_int64");
		}
		else if (type.basetype == SPIRType::Half)
		{
			require_extension_internal("GL_EXT_shader_explicit_arithmetic_types_float16");
			if (options.vulkan_semantics)
				require_extension_internal("GL_EXT_shader_16bit_storage");
		}
		else if (type.basetype == SPIRType::SByte || type.basetype == SPIRType::UByte)
		{
			require_extension_internal("GL_EXT_shader_explicit_arithmetic_types_int8");
			if (options.vulkan_semantics)
				require_extension_internal("GL_EXT_shader_8bit_storage");
		}
		else if (type.basetype == SPIRType::Short || type.basetype == SPIRType::UShort)
		{
			require_extension_internal("GL_EXT_shader_explicit_arithmetic_types_int16");
			if (options.vulkan_semantics)
				require_extension_internal("GL_EXT_shader_16bit_storage");
		}
	});

	auto &execution = get_entry_point();
	switch (execution.model)
	{
	case ExecutionModelGLCompute:
		if (!options.es && options.version < 430)
			require_extension_internal("GL_ARB_compute_shader");
		if (options.es && options.version < 310)
			SPIRV_CROSS_THROW("At least ESSL 3.10 required for compute shaders.");
		break;

	case ExecutionModelGeometry:
		if (options.es && options.version < 320)
			require_extension_internal("GL_EXT_geometry_shader");
		if (!options.es && options.version < 150)
			require_extension_internal("GL_ARB_geometry_shader4");

		if (execution.flags.get(ExecutionModeInvocations) && execution.invocations != 1)
		{
			// Instanced GS is part of 400 core or this extension.
			if (!options.es && options.version < 400)
				require_extension_internal("GL_ARB_gpu_shader5");
		}
		break;

	case ExecutionModelTessellationEvaluation:
	case ExecutionModelTessellationControl:
		if (options.es && options.version < 320)
			require_extension_internal("GL_EXT_tessellation_shader");
		if (!options.es && options.version < 400)
			require_extension_internal("GL_ARB_tessellation_shader");
		break;

	case ExecutionModelRayGenerationNV:
	case ExecutionModelIntersectionNV:
	case ExecutionModelAnyHitNV:
	case ExecutionModelClosestHitNV:
	case ExecutionModelMissNV:
	case ExecutionModelCallableNV:
		if (options.es || options.version < 460)
			SPIRV_CROSS_THROW("Ray tracing shaders require non-es profile with version 460 or above.");
		require_extension_internal("GL_NV_ray_tracing");
		break;

	default:
		break;
	}

	if (!pls_inputs.empty() || !pls_outputs.empty())
	{
		if (execution.model != ExecutionModelFragment)
			SPIRV_CROSS_THROW("Can only use GL_EXT_shader_pixel_local_storage in fragment shaders.");
		require_extension_internal("GL_EXT_shader_pixel_local_storage");
	}

	if (!inout_color_attachments.empty())
	{
		if (execution.model != ExecutionModelFragment)
			SPIRV_CROSS_THROW("Can only use GL_EXT_shader_framebuffer_fetch in fragment shaders.");
		if (options.vulkan_semantics)
			SPIRV_CROSS_THROW("Cannot use EXT_shader_framebuffer_fetch in Vulkan GLSL.");
		require_extension_internal("GL_EXT_shader_framebuffer_fetch");
	}

	if (options.separate_shader_objects && !options.es && options.version < 410)
		require_extension_internal("GL_ARB_separate_shader_objects");

	if (ir.addressing_model == AddressingModelPhysicalStorageBuffer64EXT)
	{
		if (!options.vulkan_semantics)
			SPIRV_CROSS_THROW("GL_EXT_buffer_reference is only supported in Vulkan GLSL.");
		if (options.es && options.version < 320)
			SPIRV_CROSS_THROW("GL_EXT_buffer_reference requires ESSL 320.");
		else if (!options.es && options.version < 450)
			SPIRV_CROSS_THROW("GL_EXT_buffer_reference requires GLSL 450.");
		require_extension_internal("GL_EXT_buffer_reference");
	}
	else if (ir.addressing_model != AddressingModelLogical)
	{
		SPIRV_CROSS_THROW("Only Logical and PhysicalStorageBuffer64EXT addressing models are supported.");
	}

	// Check for nonuniform qualifier and passthrough.
	// Instead of looping over all decorations to find this, just look at capabilities.
	for (auto &cap : ir.declared_capabilities)
	{
		switch (cap)
		{
		case CapabilityShaderNonUniformEXT:
		case CapabilityRuntimeDescriptorArrayEXT:
			if (!options.vulkan_semantics)
				SPIRV_CROSS_THROW("GL_EXT_nonuniform_qualifier is only supported in Vulkan GLSL.");
			require_extension_internal("GL_EXT_nonuniform_qualifier");
			break;

		case CapabilityGeometryShaderPassthroughNV:
			if (execution.model == ExecutionModelGeometry)
			{
				require_extension_internal("GL_NV_geometry_shader_passthrough");
				execution.geometry_passthrough = true;
			}
			break;

		default:
			break;
		}
	}
}

string CompilerGLSL::compile()
{
	if (options.vulkan_semantics)
		backend.allow_precision_qualifiers = true;
	backend.force_gl_in_out_block = true;
	backend.supports_extensions = true;
	backend.use_array_constructor = true;

	// Scan the SPIR-V to find trivial uses of extensions.
	fixup_type_alias();
	reorder_type_alias();
	build_function_control_flow_graphs_and_analyze();
	find_static_extensions();
	fixup_image_load_store_access();
	update_active_builtins();
	analyze_image_and_sampler_usage();
	analyze_interlocked_resource_usage();
	if (!inout_color_attachments.empty())
		emit_inout_fragment_outputs_copy_to_subpass_inputs();

	// Shaders might cast unrelated data to pointers of non-block types.
	// Find all such instances and make sure we can cast the pointers to a synthesized block type.
	if (ir.addressing_model == AddressingModelPhysicalStorageBuffer64EXT)
		analyze_non_block_pointer_types();

	uint32_t pass_count = 0;
	do
	{
		if (pass_count >= 3)
			SPIRV_CROSS_THROW("Over 3 compilation loops detected. Must be a bug!");

		reset();

		buffer.reset();

		emit_header();
		emit_resources();

		emit_function(get<SPIRFunction>(ir.default_entry_point), Bitset());

		pass_count++;
	} while (is_forcing_recompilation());

	// Implement the interlocked wrapper function at the end.
	// The body was implemented in lieu of main().
	if (interlocked_is_complex)
	{
		statement("void main()");
		begin_scope();
		statement("// Interlocks were used in a way not compatible with GLSL, this is very slow.");
		if (options.es)
			statement("beginInvocationInterlockNV();");
		else
			statement("beginInvocationInterlockARB();");
		statement("spvMainInterlockedBody();");
		if (options.es)
			statement("endInvocationInterlockNV();");
		else
			statement("endInvocationInterlockARB();");
		end_scope();
	}

	// Entry point in GLSL is always main().
	get_entry_point().name = "main";

	return buffer.str();
}

std::string CompilerGLSL::get_partial_source()
{
	return buffer.str();
}

void CompilerGLSL::build_workgroup_size(SmallVector<string> &arguments, const SpecializationConstant &wg_x,
                                        const SpecializationConstant &wg_y, const SpecializationConstant &wg_z)
{
	auto &execution = get_entry_point();

	if (wg_x.id)
	{
		if (options.vulkan_semantics)
			arguments.push_back(join("local_size_x_id = ", wg_x.constant_id));
		else
			arguments.push_back(join("local_size_x = ", get<SPIRConstant>(wg_x.id).specialization_constant_macro_name));
	}
	else
		arguments.push_back(join("local_size_x = ", execution.workgroup_size.x));

	if (wg_y.id)
	{
		if (options.vulkan_semantics)
			arguments.push_back(join("local_size_y_id = ", wg_y.constant_id));
		else
			arguments.push_back(join("local_size_y = ", get<SPIRConstant>(wg_y.id).specialization_constant_macro_name));
	}
	else
		arguments.push_back(join("local_size_y = ", execution.workgroup_size.y));

	if (wg_z.id)
	{
		if (options.vulkan_semantics)
			arguments.push_back(join("local_size_z_id = ", wg_z.constant_id));
		else
			arguments.push_back(join("local_size_z = ", get<SPIRConstant>(wg_z.id).specialization_constant_macro_name));
	}
	else
		arguments.push_back(join("local_size_z = ", execution.workgroup_size.z));
}

void CompilerGLSL::emit_header()
{
	auto &execution = get_entry_point();
	statement("#version ", options.version, options.es && options.version > 100 ? " es" : "");

	if (!options.es && options.version < 420)
	{
		// Needed for binding = # on UBOs, etc.
		if (options.enable_420pack_extension)
		{
			statement("#ifdef GL_ARB_shading_language_420pack");
			statement("#extension GL_ARB_shading_language_420pack : require");
			statement("#endif");
		}
		// Needed for: layout(early_fragment_tests) in;
		if (execution.flags.get(ExecutionModeEarlyFragmentTests))
			require_extension_internal("GL_ARB_shader_image_load_store");
	}

	// Needed for: layout(post_depth_coverage) in;
	if (execution.flags.get(ExecutionModePostDepthCoverage))
		require_extension_internal("GL_ARB_post_depth_coverage");

	// Needed for: layout({pixel,sample}_interlock_[un]ordered) in;
	if (execution.flags.get(ExecutionModePixelInterlockOrderedEXT) ||
	    execution.flags.get(ExecutionModePixelInterlockUnorderedEXT) ||
	    execution.flags.get(ExecutionModeSampleInterlockOrderedEXT) ||
	    execution.flags.get(ExecutionModeSampleInterlockUnorderedEXT))
	{
		if (options.es)
		{
			if (options.version < 310)
				SPIRV_CROSS_THROW("At least ESSL 3.10 required for fragment shader interlock.");
			require_extension_internal("GL_NV_fragment_shader_interlock");
		}
		else
		{
			if (options.version < 420)
				require_extension_internal("GL_ARB_shader_image_load_store");
			require_extension_internal("GL_ARB_fragment_shader_interlock");
		}
	}

	for (auto &ext : forced_extensions)
	{
		if (ext == "GL_EXT_shader_explicit_arithmetic_types_float16")
		{
			// Special case, this extension has a potential fallback to another vendor extension in normal GLSL.
			// GL_AMD_gpu_shader_half_float is a superset, so try that first.
			statement("#if defined(GL_AMD_gpu_shader_half_float)");
			statement("#extension GL_AMD_gpu_shader_half_float : require");
			if (!options.vulkan_semantics)
			{
				statement("#elif defined(GL_NV_gpu_shader5)");
				statement("#extension GL_NV_gpu_shader5 : require");
			}
			else
			{
				statement("#elif defined(GL_EXT_shader_explicit_arithmetic_types_float16)");
				statement("#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require");
			}
			statement("#else");
			statement("#error No extension available for FP16.");
			statement("#endif");
		}
		else if (ext == "GL_EXT_shader_explicit_arithmetic_types_int16")
		{
			if (options.vulkan_semantics)
				statement("#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require");
			else
			{
				statement("#if defined(GL_AMD_gpu_shader_int16)");
				statement("#extension GL_AMD_gpu_shader_int16 : require");
				statement("#else");
				statement("#error No extension available for Int16.");
				statement("#endif");
			}
		}
		else if (ext == "GL_ARB_post_depth_coverage")
		{
			if (options.es)
				statement("#extension GL_EXT_post_depth_coverage : require");
			else
			{
				statement("#if defined(GL_ARB_post_depth_coverge)");
				statement("#extension GL_ARB_post_depth_coverage : require");
				statement("#else");
				statement("#extension GL_EXT_post_depth_coverage : require");
				statement("#endif");
			}
		}
		else
			statement("#extension ", ext, " : require");
	}

	for (auto &header : header_lines)
		statement(header);

	SmallVector<string> inputs;
	SmallVector<string> outputs;

	switch (execution.model)
	{
	case ExecutionModelGeometry:
		if ((execution.flags.get(ExecutionModeInvocations)) && execution.invocations != 1)
			inputs.push_back(join("invocations = ", execution.invocations));
		if (execution.flags.get(ExecutionModeInputPoints))
			inputs.push_back("points");
		if (execution.flags.get(ExecutionModeInputLines))
			inputs.push_back("lines");
		if (execution.flags.get(ExecutionModeInputLinesAdjacency))
			inputs.push_back("lines_adjacency");
		if (execution.flags.get(ExecutionModeTriangles))
			inputs.push_back("triangles");
		if (execution.flags.get(ExecutionModeInputTrianglesAdjacency))
			inputs.push_back("triangles_adjacency");

		if (!execution.geometry_passthrough)
		{
			// For passthrough, these are implies and cannot be declared in shader.
			outputs.push_back(join("max_vertices = ", execution.output_vertices));
			if (execution.flags.get(ExecutionModeOutputTriangleStrip))
				outputs.push_back("triangle_strip");
			if (execution.flags.get(ExecutionModeOutputPoints))
				outputs.push_back("points");
			if (execution.flags.get(ExecutionModeOutputLineStrip))
				outputs.push_back("line_strip");
		}
		break;

	case ExecutionModelTessellationControl:
		if (execution.flags.get(ExecutionModeOutputVertices))
			outputs.push_back(join("vertices = ", execution.output_vertices));
		break;

	case ExecutionModelTessellationEvaluation:
		if (execution.flags.get(ExecutionModeQuads))
			inputs.push_back("quads");
		if (execution.flags.get(ExecutionModeTriangles))
			inputs.push_back("triangles");
		if (execution.flags.get(ExecutionModeIsolines))
			inputs.push_back("isolines");
		if (execution.flags.get(ExecutionModePointMode))
			inputs.push_back("point_mode");

		if (!execution.flags.get(ExecutionModeIsolines))
		{
			if (execution.flags.get(ExecutionModeVertexOrderCw))
				inputs.push_back("cw");
			if (execution.flags.get(ExecutionModeVertexOrderCcw))
				inputs.push_back("ccw");
		}

		if (execution.flags.get(ExecutionModeSpacingFractionalEven))
			inputs.push_back("fractional_even_spacing");
		if (execution.flags.get(ExecutionModeSpacingFractionalOdd))
			inputs.push_back("fractional_odd_spacing");
		if (execution.flags.get(ExecutionModeSpacingEqual))
			inputs.push_back("equal_spacing");
		break;

	case ExecutionModelGLCompute:
	{
		if (execution.workgroup_size.constant != 0)
		{
			SpecializationConstant wg_x, wg_y, wg_z;
			get_work_group_size_specialization_constants(wg_x, wg_y, wg_z);

			// If there are any spec constants on legacy GLSL, defer declaration, we need to set up macro
			// declarations before we can emit the work group size.
			if (options.vulkan_semantics ||
			    ((wg_x.id == ConstantID(0)) && (wg_y.id == ConstantID(0)) && (wg_z.id == ConstantID(0))))
				build_workgroup_size(inputs, wg_x, wg_y, wg_z);
		}
		else
		{
			inputs.push_back(join("local_size_x = ", execution.workgroup_size.x));
			inputs.push_back(join("local_size_y = ", execution.workgroup_size.y));
			inputs.push_back(join("local_size_z = ", execution.workgroup_size.z));
		}
		break;
	}

	case ExecutionModelFragment:
		if (options.es)
		{
			switch (options.fragment.default_float_precision)
			{
			case Options::Lowp:
				statement("precision lowp float;");
				break;

			case Options::Mediump:
				statement("precision mediump float;");
				break;

			case Options::Highp:
				statement("precision highp float;");
				break;

			default:
				break;
			}

			switch (options.fragment.default_int_precision)
			{
			case Options::Lowp:
				statement("precision lowp int;");
				break;

			case Options::Mediump:
				statement("precision mediump int;");
				break;

			case Options::Highp:
				statement("precision highp int;");
				break;

			default:
				break;
			}
		}

		if (execution.flags.get(ExecutionModeEarlyFragmentTests))
			inputs.push_back("early_fragment_tests");
		if (execution.flags.get(ExecutionModePostDepthCoverage))
			inputs.push_back("post_depth_coverage");

		if (execution.flags.get(ExecutionModePixelInterlockOrderedEXT))
			inputs.push_back("pixel_interlock_ordered");
		else if (execution.flags.get(ExecutionModePixelInterlockUnorderedEXT))
			inputs.push_back("pixel_interlock_unordered");
		else if (execution.flags.get(ExecutionModeSampleInterlockOrderedEXT))
			inputs.push_back("sample_interlock_ordered");
		else if (execution.flags.get(ExecutionModeSampleInterlockUnorderedEXT))
			inputs.push_back("sample_interlock_unordered");

		if (!options.es && execution.flags.get(ExecutionModeDepthGreater))
			statement("layout(depth_greater) out float gl_FragDepth;");
		else if (!options.es && execution.flags.get(ExecutionModeDepthLess))
			statement("layout(depth_less) out float gl_FragDepth;");

		break;

	default:
		break;
	}

	if (!inputs.empty())
		statement("layout(", merge(inputs), ") in;");
	if (!outputs.empty())
		statement("layout(", merge(outputs), ") out;");

	statement("");
}

bool CompilerGLSL::type_is_empty(const SPIRType &type)
{
	return type.basetype == SPIRType::Struct && type.member_types.empty();
}

void CompilerGLSL::emit_struct(SPIRType &type)
{
	// Struct types can be stamped out multiple times
	// with just different offsets, matrix layouts, etc ...
	// Type-punning with these types is legal, which complicates things
	// when we are storing struct and array types in an SSBO for example.
	// If the type master is packed however, we can no longer assume that the struct declaration will be redundant.
	if (type.type_alias != TypeID(0) &&
	    !has_extended_decoration(type.type_alias, SPIRVCrossDecorationBufferBlockRepacked))
		return;

	add_resource_name(type.self);
	auto name = type_to_glsl(type);

	statement(!backend.explicit_struct_type ? "struct " : "", name);
	begin_scope();

	type.member_name_cache.clear();

	uint32_t i = 0;
	bool emitted = false;
	for (auto &member : type.member_types)
	{
		add_member_name(type, i);
		emit_struct_member(type, member, i);
		i++;
		emitted = true;
	}

	// Don't declare empty structs in GLSL, this is not allowed.
	if (type_is_empty(type) && !backend.supports_empty_struct)
	{
		statement("int empty_struct_member;");
		emitted = true;
	}

	if (has_extended_decoration(type.self, SPIRVCrossDecorationPaddingTarget))
		emit_struct_padding_target(type);

	end_scope_decl();

	if (emitted)
		statement("");
}

string CompilerGLSL::to_interpolation_qualifiers(const Bitset &flags)
{
	string res;
	//if (flags & (1ull << DecorationSmooth))
	//    res += "smooth ";
	if (flags.get(DecorationFlat))
		res += "flat ";
	if (flags.get(DecorationNoPerspective))
		res += "noperspective ";
	if (flags.get(DecorationCentroid))
		res += "centroid ";
	if (flags.get(DecorationPatch))
		res += "patch ";
	if (flags.get(DecorationSample))
		res += "sample ";
	if (flags.get(DecorationInvariant))
		res += "invariant ";
	if (flags.get(DecorationExplicitInterpAMD))
		res += "__explicitInterpAMD ";

	return res;
}

string CompilerGLSL::layout_for_member(const SPIRType &type, uint32_t index)
{
	if (is_legacy())
		return "";

	bool is_block = ir.meta[type.self].decoration.decoration_flags.get(DecorationBlock) ||
	                ir.meta[type.self].decoration.decoration_flags.get(DecorationBufferBlock);
	if (!is_block)
		return "";

	auto &memb = ir.meta[type.self].members;
	if (index >= memb.size())
		return "";
	auto &dec = memb[index];

	SmallVector<string> attr;

	if (has_member_decoration(type.self, index, DecorationPassthroughNV))
		attr.push_back("passthrough");

	// We can only apply layouts on members in block interfaces.
	// This is a bit problematic because in SPIR-V decorations are applied on the struct types directly.
	// This is not supported on GLSL, so we have to make the assumption that if a struct within our buffer block struct
	// has a decoration, it was originally caused by a top-level layout() qualifier in GLSL.
	//
	// We would like to go from (SPIR-V style):
	//
	// struct Foo { layout(row_major) mat4 matrix; };
	// buffer UBO { Foo foo; };
	//
	// to
	//
	// struct Foo { mat4 matrix; }; // GLSL doesn't support any layout shenanigans in raw struct declarations.
	// buffer UBO { layout(row_major) Foo foo; }; // Apply the layout on top-level.
	auto flags = combined_decoration_for_member(type, index);

	if (flags.get(DecorationRowMajor))
		attr.push_back("row_major");
	// We don't emit any global layouts, so column_major is default.
	//if (flags & (1ull << DecorationColMajor))
	//    attr.push_back("column_major");

	if (dec.decoration_flags.get(DecorationLocation) && can_use_io_location(type.storage, true))
		attr.push_back(join("location = ", dec.location));

	// Can only declare component if we can declare location.
	if (dec.decoration_flags.get(DecorationComponent) && can_use_io_location(type.storage, true))
	{
		if (!options.es)
		{
			if (options.version < 440 && options.version >= 140)
				require_extension_internal("GL_ARB_enhanced_layouts");
			else if (options.version < 140)
				SPIRV_CROSS_THROW("Component decoration is not supported in targets below GLSL 1.40.");
			attr.push_back(join("component = ", dec.component));
		}
		else
			SPIRV_CROSS_THROW("Component decoration is not supported in ES targets.");
	}

	// SPIRVCrossDecorationPacked is set by layout_for_variable earlier to mark that we need to emit offset qualifiers.
	// This is only done selectively in GLSL as needed.
	if (has_extended_decoration(type.self, SPIRVCrossDecorationExplicitOffset) &&
	    dec.decoration_flags.get(DecorationOffset))
		attr.push_back(join("offset = ", dec.offset));
	else if (type.storage == StorageClassOutput && dec.decoration_flags.get(DecorationOffset))
		attr.push_back(join("xfb_offset = ", dec.offset));

	if (attr.empty())
		return "";

	string res = "layout(";
	res += merge(attr);
	res += ") ";
	return res;
}

const char *CompilerGLSL::format_to_glsl(spv::ImageFormat format)
{
	if (options.es && is_desktop_only_format(format))
		SPIRV_CROSS_THROW("Attempting to use image format not supported in ES profile.");

	switch (format)
	{
	case ImageFormatRgba32f:
		return "rgba32f";
	case ImageFormatRgba16f:
		return "rgba16f";
	case ImageFormatR32f:
		return "r32f";
	case ImageFormatRgba8:
		return "rgba8";
	case ImageFormatRgba8Snorm:
		return "rgba8_snorm";
	case ImageFormatRg32f:
		return "rg32f";
	case ImageFormatRg16f:
		return "rg16f";
	case ImageFormatRgba32i:
		return "rgba32i";
	case ImageFormatRgba16i:
		return "rgba16i";
	case ImageFormatR32i:
		return "r32i";
	case ImageFormatRgba8i:
		return "rgba8i";
	case ImageFormatRg32i:
		return "rg32i";
	case ImageFormatRg16i:
		return "rg16i";
	case ImageFormatRgba32ui:
		return "rgba32ui";
	case ImageFormatRgba16ui:
		return "rgba16ui";
	case ImageFormatR32ui:
		return "r32ui";
	case ImageFormatRgba8ui:
		return "rgba8ui";
	case ImageFormatRg32ui:
		return "rg32ui";
	case ImageFormatRg16ui:
		return "rg16ui";
	case ImageFormatR11fG11fB10f:
		return "r11f_g11f_b10f";
	case ImageFormatR16f:
		return "r16f";
	case ImageFormatRgb10A2:
		return "rgb10_a2";
	case ImageFormatR8:
		return "r8";
	case ImageFormatRg8:
		return "rg8";
	case ImageFormatR16:
		return "r16";
	case ImageFormatRg16:
		return "rg16";
	case ImageFormatRgba16:
		return "rgba16";
	case ImageFormatR16Snorm:
		return "r16_snorm";
	case ImageFormatRg16Snorm:
		return "rg16_snorm";
	case ImageFormatRgba16Snorm:
		return "rgba16_snorm";
	case ImageFormatR8Snorm:
		return "r8_snorm";
	case ImageFormatRg8Snorm:
		return "rg8_snorm";
	case ImageFormatR8ui:
		return "r8ui";
	case ImageFormatRg8ui:
		return "rg8ui";
	case ImageFormatR16ui:
		return "r16ui";
	case ImageFormatRgb10a2ui:
		return "rgb10_a2ui";
	case ImageFormatR8i:
		return "r8i";
	case ImageFormatRg8i:
		return "rg8i";
	case ImageFormatR16i:
		return "r16i";
	default:
	case ImageFormatUnknown:
		return nullptr;
	}
}

uint32_t CompilerGLSL::type_to_packed_base_size(const SPIRType &type, BufferPackingStandard)
{
	switch (type.basetype)
	{
	case SPIRType::Double:
	case SPIRType::Int64:
	case SPIRType::UInt64:
		return 8;
	case SPIRType::Float:
	case SPIRType::Int:
	case SPIRType::UInt:
		return 4;
	case SPIRType::Half:
	case SPIRType::Short:
	case SPIRType::UShort:
		return 2;
	case SPIRType::SByte:
	case SPIRType::UByte:
		return 1;

	default:
		SPIRV_CROSS_THROW("Unrecognized type in type_to_packed_base_size.");
	}
}

uint32_t CompilerGLSL::type_to_packed_alignment(const SPIRType &type, const Bitset &flags,
                                                BufferPackingStandard packing)
{
	// If using PhysicalStorageBufferEXT storage class, this is a pointer,
	// and is 64-bit.
	if (type.storage == StorageClassPhysicalStorageBufferEXT)
	{
		if (!type.pointer)
			SPIRV_CROSS_THROW("Types in PhysicalStorageBufferEXT must be pointers.");

		if (ir.addressing_model == AddressingModelPhysicalStorageBuffer64EXT)
		{
			if (packing_is_vec4_padded(packing) && type_is_array_of_pointers(type))
				return 16;
			else
				return 8;
		}
		else
			SPIRV_CROSS_THROW("AddressingModelPhysicalStorageBuffer64EXT must be used for PhysicalStorageBufferEXT.");
	}

	if (!type.array.empty())
	{
		uint32_t minimum_alignment = 1;
		if (packing_is_vec4_padded(packing))
			minimum_alignment = 16;

		auto *tmp = &get<SPIRType>(type.parent_type);
		while (!tmp->array.empty())
			tmp = &get<SPIRType>(tmp->parent_type);

		// Get the alignment of the base type, then maybe round up.
		return max(minimum_alignment, type_to_packed_alignment(*tmp, flags, packing));
	}

	if (type.basetype == SPIRType::Struct)
	{
		// Rule 9. Structs alignments are maximum alignment of its members.
		uint32_t alignment = 1;
		for (uint32_t i = 0; i < type.member_types.size(); i++)
		{
			auto member_flags = ir.meta[type.self].members[i].decoration_flags;
			alignment =
			    max(alignment, type_to_packed_alignment(get<SPIRType>(type.member_types[i]), member_flags, packing));
		}

		// In std140, struct alignment is rounded up to 16.
		if (packing_is_vec4_padded(packing))
			alignment = max(alignment, 16u);

		return alignment;
	}
	else
	{
		const uint32_t base_alignment = type_to_packed_base_size(type, packing);

		// Alignment requirement for scalar block layout is always the alignment for the most basic component.
		if (packing_is_scalar(packing))
			return base_alignment;

		// Vectors are *not* aligned in HLSL, but there's an extra rule where vectors cannot straddle
		// a vec4, this is handled outside since that part knows our current offset.
		if (type.columns == 1 && packing_is_hlsl(packing))
			return base_alignment;

		// From 7.6.2.2 in GL 4.5 core spec.
		// Rule 1
		if (type.vecsize == 1 && type.columns == 1)
			return base_alignment;

		// Rule 2
		if ((type.vecsize == 2 || type.vecsize == 4) && type.columns == 1)
			return type.vecsize * base_alignment;

		// Rule 3
		if (type.vecsize == 3 && type.columns == 1)
			return 4 * base_alignment;

		// Rule 4 implied. Alignment does not change in std430.

		// Rule 5. Column-major matrices are stored as arrays of
		// vectors.
		if (flags.get(DecorationColMajor) && type.columns > 1)
		{
			if (packing_is_vec4_padded(packing))
				return 4 * base_alignment;
			else if (type.vecsize == 3)
				return 4 * base_alignment;
			else
				return type.vecsize * base_alignment;
		}

		// Rule 6 implied.

		// Rule 7.
		if (flags.get(DecorationRowMajor) && type.vecsize > 1)
		{
			if (packing_is_vec4_padded(packing))
				return 4 * base_alignment;
			else if (type.columns == 3)
				return 4 * base_alignment;
			else
				return type.columns * base_alignment;
		}

		// Rule 8 implied.
	}

	SPIRV_CROSS_THROW("Did not find suitable rule for type. Bogus decorations?");
}

uint32_t CompilerGLSL::type_to_packed_array_stride(const SPIRType &type, const Bitset &flags,
                                                   BufferPackingStandard packing)
{
	// Array stride is equal to aligned size of the underlying type.
	uint32_t parent = type.parent_type;
	assert(parent);

	auto &tmp = get<SPIRType>(parent);

	uint32_t size = type_to_packed_size(tmp, flags, packing);
	if (tmp.array.empty())
	{
		uint32_t alignment = type_to_packed_alignment(type, flags, packing);
		return (size + alignment - 1) & ~(alignment - 1);
	}
	else
	{
		// For multidimensional arrays, array stride always matches size of subtype.
		// The alignment cannot change because multidimensional arrays are basically N * M array elements.
		return size;
	}
}

uint32_t CompilerGLSL::type_to_packed_size(const SPIRType &type, const Bitset &flags, BufferPackingStandard packing)
{
	if (!type.array.empty())
	{
		return to_array_size_literal(type) * type_to_packed_array_stride(type, flags, packing);
	}

	// If using PhysicalStorageBufferEXT storage class, this is a pointer,
	// and is 64-bit.
	if (type.storage == StorageClassPhysicalStorageBufferEXT)
	{
		if (!type.pointer)
			SPIRV_CROSS_THROW("Types in PhysicalStorageBufferEXT must be pointers.");

		if (ir.addressing_model == AddressingModelPhysicalStorageBuffer64EXT)
			return 8;
		else
			SPIRV_CROSS_THROW("AddressingModelPhysicalStorageBuffer64EXT must be used for PhysicalStorageBufferEXT.");
	}

	uint32_t size = 0;

	if (type.basetype == SPIRType::Struct)
	{
		uint32_t pad_alignment = 1;

		for (uint32_t i = 0; i < type.member_types.size(); i++)
		{
			auto member_flags = ir.meta[type.self].members[i].decoration_flags;
			auto &member_type = get<SPIRType>(type.member_types[i]);

			uint32_t packed_alignment = type_to_packed_alignment(member_type, member_flags, packing);
			uint32_t alignment = max(packed_alignment, pad_alignment);

			// The next member following a struct member is aligned to the base alignment of the struct that came before.
			// GL 4.5 spec, 7.6.2.2.
			if (member_type.basetype == SPIRType::Struct)
				pad_alignment = packed_alignment;
			else
				pad_alignment = 1;

			size = (size + alignment - 1) & ~(alignment - 1);
			size += type_to_packed_size(member_type, member_flags, packing);
		}
	}
	else
	{
		const uint32_t base_alignment = type_to_packed_base_size(type, packing);

		if (packing_is_scalar(packing))
		{
			size = type.vecsize * type.columns * base_alignment;
		}
		else
		{
			if (type.columns == 1)
				size = type.vecsize * base_alignment;

			if (flags.get(DecorationColMajor) && type.columns > 1)
			{
				if (packing_is_vec4_padded(packing))
					size = type.columns * 4 * base_alignment;
				else if (type.vecsize == 3)
					size = type.columns * 4 * base_alignment;
				else
					size = type.columns * type.vecsize * base_alignment;
			}

			if (flags.get(DecorationRowMajor) && type.vecsize > 1)
			{
				if (packing_is_vec4_padded(packing))
					size = type.vecsize * 4 * base_alignment;
				else if (type.columns == 3)
					size = type.vecsize * 4 * base_alignment;
				else
					size = type.vecsize * type.columns * base_alignment;
			}
		}
	}

	return size;
}

bool CompilerGLSL::buffer_is_packing_standard(const SPIRType &type, BufferPackingStandard packing,
                                              uint32_t *failed_validation_index, uint32_t start_offset,
                                              uint32_t end_offset)
{
	// This is very tricky and error prone, but try to be exhaustive and correct here.
	// SPIR-V doesn't directly say if we're using std430 or std140.
	// SPIR-V communicates this using Offset and ArrayStride decorations (which is what really matters),
	// so we have to try to infer whether or not the original GLSL source was std140 or std430 based on this information.
	// We do not have to consider shared or packed since these layouts are not allowed in Vulkan SPIR-V (they are useless anyways, and custom offsets would do the same thing).
	//
	// It is almost certain that we're using std430, but it gets tricky with arrays in particular.
	// We will assume std430, but infer std140 if we can prove the struct is not compliant with std430.
	//
	// The only two differences between std140 and std430 are related to padding alignment/array stride
	// in arrays and structs. In std140 they take minimum vec4 alignment.
	// std430 only removes the vec4 requirement.

	uint32_t offset = 0;
	uint32_t pad_alignment = 1;

	bool is_top_level_block =
	    has_decoration(type.self, DecorationBlock) || has_decoration(type.self, DecorationBufferBlock);

	for (uint32_t i = 0; i < type.member_types.size(); i++)
	{
		auto &memb_type = get<SPIRType>(type.member_types[i]);
		auto member_flags = ir.meta[type.self].members[i].decoration_flags;

		// Verify alignment rules.
		uint32_t packed_alignment = type_to_packed_alignment(memb_type, member_flags, packing);

		// This is a rather dirty workaround to deal with some cases of OpSpecConstantOp used as array size, e.g:
		// layout(constant_id = 0) const int s = 10;
		// const int S = s + 5; // SpecConstantOp
		// buffer Foo { int data[S]; }; // <-- Very hard for us to deduce a fixed value here,
		// we would need full implementation of compile-time constant folding. :(
		// If we are the last member of a struct, there might be cases where the actual size of that member is irrelevant
		// for our analysis (e.g. unsized arrays).
		// This lets us simply ignore that there are spec constant op sized arrays in our buffers.
		// Querying size of this member will fail, so just don't call it unless we have to.
		//
		// This is likely "best effort" we can support without going into unacceptably complicated workarounds.
		bool member_can_be_unsized =
		    is_top_level_block && size_t(i + 1) == type.member_types.size() && !memb_type.array.empty();

		uint32_t packed_size = 0;
		if (!member_can_be_unsized)
			packed_size = type_to_packed_size(memb_type, member_flags, packing);

		// We only need to care about this if we have non-array types which can straddle the vec4 boundary.
		if (packing_is_hlsl(packing))
		{
			// If a member straddles across a vec4 boundary, alignment is actually vec4.
			uint32_t begin_word = offset / 16;
			uint32_t end_word = (offset + packed_size - 1) / 16;
			if (begin_word != end_word)
				packed_alignment = max(packed_alignment, 16u);
		}

		uint32_t alignment = max(packed_alignment, pad_alignment);
		offset = (offset + alignment - 1) & ~(alignment - 1);

		// Field is not in the specified range anymore and we can ignore any further fields.
		if (offset >= end_offset)
			break;

		// The next member following a struct member is aligned to the base alignment of the struct that came before.
		// GL 4.5 spec, 7.6.2.2.
		if (memb_type.basetype == SPIRType::Struct && !memb_type.pointer)
			pad_alignment = packed_alignment;
		else
			pad_alignment = 1;

		// Only care about packing if we are in the given range
		if (offset >= start_offset)
		{
			uint32_t actual_offset = type_struct_member_offset(type, i);

			// We only care about offsets in std140, std430, etc ...
			// For EnhancedLayout variants, we have the flexibility to choose our own offsets.
			if (!packing_has_flexible_offset(packing))
			{
				if (actual_offset != offset) // This cannot be the packing we're looking for.
				{
					if (failed_validation_index)
						*failed_validation_index = i;
					return false;
				}
			}
			else if ((actual_offset & (alignment - 1)) != 0)
			{
				// We still need to verify that alignment rules are observed, even if we have explicit offset.
				if (failed_validation_index)
					*failed_validation_index = i;
				return false;
			}

			// Verify array stride rules.
			if (!memb_type.array.empty() && type_to_packed_array_stride(memb_type, member_flags, packing) !=
			                                    type_struct_member_array_stride(type, i))
			{
				if (failed_validation_index)
					*failed_validation_index = i;
				return false;
			}

			// Verify that sub-structs also follow packing rules.
			// We cannot use enhanced layouts on substructs, so they better be up to spec.
			auto substruct_packing = packing_to_substruct_packing(packing);

			if (!memb_type.pointer && !memb_type.member_types.empty() &&
			    !buffer_is_packing_standard(memb_type, substruct_packing))
			{
				if (failed_validation_index)
					*failed_validation_index = i;
				return false;
			}
		}

		// Bump size.
		offset += packed_size;
	}

	return true;
}

bool CompilerGLSL::can_use_io_location(StorageClass storage, bool block)
{
	// Location specifiers are must have in SPIR-V, but they aren't really supported in earlier versions of GLSL.
	// Be very explicit here about how to solve the issue.
	if ((get_execution_model() != ExecutionModelVertex && storage == StorageClassInput) ||
	    (get_execution_model() != ExecutionModelFragment && storage == StorageClassOutput))
	{
		uint32_t minimum_desktop_version = block ? 440 : 410;
		// ARB_enhanced_layouts vs ARB_separate_shader_objects ...

		if (!options.es && options.version < minimum_desktop_version && !options.separate_shader_objects)
			return false;
		else if (options.es && options.version < 310)
			return false;
	}

	if ((get_execution_model() == ExecutionModelVertex && storage == StorageClassInput) ||
	    (get_execution_model() == ExecutionModelFragment && storage == StorageClassOutput))
	{
		if (options.es && options.version < 300)
			return false;
		else if (!options.es && options.version < 330)
			return false;
	}

	if (storage == StorageClassUniform || storage == StorageClassUniformConstant || storage == StorageClassPushConstant)
	{
		if (options.es && options.version < 310)
			return false;
		else if (!options.es && options.version < 430)
			return false;
	}

	return true;
}

string CompilerGLSL::layout_for_variable(const SPIRVariable &var)
{
	// FIXME: Come up with a better solution for when to disable layouts.
	// Having layouts depend on extensions as well as which types
	// of layouts are used. For now, the simple solution is to just disable
	// layouts for legacy versions.
	if (is_legacy())
		return "";

	if (subpass_input_is_framebuffer_fetch(var.self))
		return "";

	SmallVector<string> attr;

	auto &type = get<SPIRType>(var.basetype);
	auto &flags = get_decoration_bitset(var.self);
	auto &typeflags = get_decoration_bitset(type.self);

	if (flags.get(DecorationPassthroughNV))
		attr.push_back("passthrough");

	if (options.vulkan_semantics && var.storage == StorageClassPushConstant)
		attr.push_back("push_constant");
	else if (var.storage == StorageClassShaderRecordBufferNV)
		attr.push_back("shaderRecordNV");

	if (flags.get(DecorationRowMajor))
		attr.push_back("row_major");
	if (flags.get(DecorationColMajor))
		attr.push_back("column_major");

	if (options.vulkan_semantics)
	{
		if (flags.get(DecorationInputAttachmentIndex))
			attr.push_back(join("input_attachment_index = ", get_decoration(var.self, DecorationInputAttachmentIndex)));
	}

	bool is_block = has_decoration(type.self, DecorationBlock);
	if (flags.get(DecorationLocation) && can_use_io_location(var.storage, is_block))
	{
		Bitset combined_decoration;
		for (uint32_t i = 0; i < ir.meta[type.self].members.size(); i++)
			combined_decoration.merge_or(combined_decoration_for_member(type, i));

		// If our members have location decorations, we don't need to
		// emit location decorations at the top as well (looks weird).
		if (!combined_decoration.get(DecorationLocation))
			attr.push_back(join("location = ", get_decoration(var.self, DecorationLocation)));
	}

	// Transform feedback
	bool uses_enhanced_layouts = false;
	if (is_block && var.storage == StorageClassOutput)
	{
		// For blocks, there is a restriction where xfb_stride/xfb_buffer must only be declared on the block itself,
		// since all members must match the same xfb_buffer. The only thing we will declare for members of the block
		// is the xfb_offset.
		uint32_t member_count = uint32_t(type.member_types.size());
		bool have_xfb_buffer_stride = false;
		bool have_any_xfb_offset = false;
		uint32_t xfb_stride = 0, xfb_buffer = 0;

		if (flags.get(DecorationXfbBuffer) && flags.get(DecorationXfbStride))
		{
			have_xfb_buffer_stride = true;
			xfb_buffer = get_decoration(var.self, DecorationXfbBuffer);
			xfb_stride = get_decoration(var.self, DecorationXfbStride);
		}

		// Verify that none of the members violate our assumption.
		for (uint32_t i = 0; i < member_count; i++)
		{
			// Only members with an Offset decoration participate in XFB.
			if (!has_member_decoration(type.self, i, DecorationOffset))
				continue;
			have_any_xfb_offset = true;

			if (has_member_decoration(type.self, i, DecorationXfbBuffer))
			{
				uint32_t buffer_index = get_member_decoration(type.self, i, DecorationXfbBuffer);
				if (have_xfb_buffer_stride && buffer_index != xfb_buffer)
					SPIRV_CROSS_THROW("IO block member XfbBuffer mismatch.");
				have_xfb_buffer_stride = true;
				xfb_buffer = buffer_index;
			}

			if (has_member_decoration(type.self, i, DecorationXfbStride))
			{
				uint32_t stride = get_member_decoration(type.self, i, DecorationXfbStride);
				if (have_xfb_buffer_stride && stride != xfb_stride)
					SPIRV_CROSS_THROW("IO block member XfbStride mismatch.");
				have_xfb_buffer_stride = true;
				xfb_stride = stride;
			}
		}

		if (have_xfb_buffer_stride && have_any_xfb_offset)
		{
			attr.push_back(join("xfb_buffer = ", xfb_buffer));
			attr.push_back(join("xfb_stride = ", xfb_stride));
			uses_enhanced_layouts = true;
		}
	}
	else if (var.storage == StorageClassOutput && flags.get(DecorationXfbBuffer) && flags.get(DecorationXfbStride) &&
	         flags.get(DecorationOffset))
	{
		// XFB for standalone variables, we can emit all decorations.
		attr.push_back(join("xfb_buffer = ", get_decoration(var.self, DecorationXfbBuffer)));
		attr.push_back(join("xfb_stride = ", get_decoration(var.self, DecorationXfbStride)));
		attr.push_back(join("xfb_offset = ", get_decoration(var.self, DecorationOffset)));
		uses_enhanced_layouts = true;
	}

	// Can only declare Component if we can declare location.
	if (flags.get(DecorationComponent) && can_use_io_location(var.storage, is_block))
	{
		uses_enhanced_layouts = true;
		attr.push_back(join("component = ", get_decoration(var.self, DecorationComponent)));
	}

	if (uses_enhanced_layouts)
	{
		if (!options.es)
		{
			if (options.version < 440 && options.version >= 140)
				require_extension_internal("GL_ARB_enhanced_layouts");
			else if (options.version < 140)
				SPIRV_CROSS_THROW("GL_ARB_enhanced_layouts is not supported in targets below GLSL 1.40.");
			if (!options.es && options.version < 440)
				require_extension_internal("GL_ARB_enhanced_layouts");
		}
		else if (options.es)
			SPIRV_CROSS_THROW("GL_ARB_enhanced_layouts is not supported in ESSL.");
	}

	if (flags.get(DecorationIndex))
		attr.push_back(join("index = ", get_decoration(var.self, DecorationIndex)));

	// Do not emit set = decoration in regular GLSL output, but
	// we need to preserve it in Vulkan GLSL mode.
	if (var.storage != StorageClassPushConstant && var.storage != StorageClassShaderRecordBufferNV)
	{
		if (flags.get(DecorationDescriptorSet) && options.vulkan_semantics)
			attr.push_back(join("set = ", get_decoration(var.self, DecorationDescriptorSet)));
	}

	bool push_constant_block = options.vulkan_semantics && var.storage == StorageClassPushConstant;
	bool ssbo_block = var.storage == StorageClassStorageBuffer || var.storage == StorageClassShaderRecordBufferNV ||
	                  (var.storage == StorageClassUniform && typeflags.get(DecorationBufferBlock));
	bool emulated_ubo = var.storage == StorageClassPushConstant && options.emit_push_constant_as_uniform_buffer;
	bool ubo_block = var.storage == StorageClassUniform && typeflags.get(DecorationBlock);

	// GL 3.0/GLSL 1.30 is not considered legacy, but it doesn't have UBOs ...
	bool can_use_buffer_blocks = (options.es && options.version >= 300) || (!options.es && options.version >= 140);

	// pretend no UBOs when options say so
	if (ubo_block && options.emit_uniform_buffer_as_plain_uniforms)
		can_use_buffer_blocks = false;

	bool can_use_binding;
	if (options.es)
		can_use_binding = options.version >= 310;
	else
		can_use_binding = options.enable_420pack_extension || (options.version >= 420);

	// Make sure we don't emit binding layout for a classic uniform on GLSL 1.30.
	if (!can_use_buffer_blocks && var.storage == StorageClassUniform)
		can_use_binding = false;

	if (var.storage == StorageClassShaderRecordBufferNV)
		can_use_binding = false;

	if (can_use_binding && flags.get(DecorationBinding))
		attr.push_back(join("binding = ", get_decoration(var.self, DecorationBinding)));

	if (var.storage != StorageClassOutput && flags.get(DecorationOffset))
		attr.push_back(join("offset = ", get_decoration(var.self, DecorationOffset)));

	// Instead of adding explicit offsets for every element here, just assume we're using std140 or std430.
	// If SPIR-V does not comply with either layout, we cannot really work around it.
	if (can_use_buffer_blocks && (ubo_block || emulated_ubo))
	{
		attr.push_back(buffer_to_packing_standard(type, false));
	}
	else if (can_use_buffer_blocks && (push_constant_block || ssbo_block))
	{
		attr.push_back(buffer_to_packing_standard(type, true));
	}

	// For images, the type itself adds a layout qualifer.
	// Only emit the format for storage images.
	if (type.basetype == SPIRType::Image && type.image.sampled == 2)
	{
		const char *fmt = format_to_glsl(type.image.format);
		if (fmt)
			attr.push_back(fmt);
	}

	if (attr.empty())
		return "";

	string res = "layout(";
	res += merge(attr);
	res += ") ";
	return res;
}

string CompilerGLSL::buffer_to_packing_standard(const SPIRType &type, bool support_std430_without_scalar_layout)
{
	if (support_std430_without_scalar_layout && buffer_is_packing_standard(type, BufferPackingStd430))
		return "std430";
	else if (buffer_is_packing_standard(type, BufferPackingStd140))
		return "std140";
	else if (options.vulkan_semantics && buffer_is_packing_standard(type, BufferPackingScalar))
	{
		require_extension_internal("GL_EXT_scalar_block_layout");
		return "scalar";
	}
	else if (support_std430_without_scalar_layout &&
	         buffer_is_packing_standard(type, BufferPackingStd430EnhancedLayout))
	{
		if (options.es && !options.vulkan_semantics)
			SPIRV_CROSS_THROW("Push constant block cannot be expressed as neither std430 nor std140. ES-targets do "
			                  "not support GL_ARB_enhanced_layouts.");
		if (!options.es && !options.vulkan_semantics && options.version < 440)
			require_extension_internal("GL_ARB_enhanced_layouts");

		set_extended_decoration(type.self, SPIRVCrossDecorationExplicitOffset);
		return "std430";
	}
	else if (buffer_is_packing_standard(type, BufferPackingStd140EnhancedLayout))
	{
		// Fallback time. We might be able to use the ARB_enhanced_layouts to deal with this difference,
		// however, we can only use layout(offset) on the block itself, not any substructs, so the substructs better be the appropriate layout.
		// Enhanced layouts seem to always work in Vulkan GLSL, so no need for extensions there.
		if (options.es && !options.vulkan_semantics)
			SPIRV_CROSS_THROW("Push constant block cannot be expressed as neither std430 nor std140. ES-targets do "
			                  "not support GL_ARB_enhanced_layouts.");
		if (!options.es && !options.vulkan_semantics && options.version < 440)
			require_extension_internal("GL_ARB_enhanced_layouts");

		set_extended_decoration(type.self, SPIRVCrossDecorationExplicitOffset);
		return "std140";
	}
	else if (options.vulkan_semantics && buffer_is_packing_standard(type, BufferPackingScalarEnhancedLayout))
	{
		set_extended_decoration(type.self, SPIRVCrossDecorationExplicitOffset);
		require_extension_internal("GL_EXT_scalar_block_layout");
		return "scalar";
	}
	else if (!support_std430_without_scalar_layout && options.vulkan_semantics &&
	         buffer_is_packing_standard(type, BufferPackingStd430))
	{
		// UBOs can support std430 with GL_EXT_scalar_block_layout.
		require_extension_internal("GL_EXT_scalar_block_layout");
		return "std430";
	}
	else if (!support_std430_without_scalar_layout && options.vulkan_semantics &&
	         buffer_is_packing_standard(type, BufferPackingStd430EnhancedLayout))
	{
		// UBOs can support std430 with GL_EXT_scalar_block_layout.
		set_extended_decoration(type.self, SPIRVCrossDecorationExplicitOffset);
		require_extension_internal("GL_EXT_scalar_block_layout");
		return "std430";
	}
	else
	{
		SPIRV_CROSS_THROW("Buffer block cannot be expressed as any of std430, std140, scalar, even with enhanced "
		                  "layouts. You can try flattening this block to support a more flexible layout.");
	}
}

void CompilerGLSL::emit_push_constant_block(const SPIRVariable &var)
{
	if (flattened_buffer_blocks.count(var.self))
		emit_buffer_block_flattened(var);
	else if (options.vulkan_semantics)
		emit_push_constant_block_vulkan(var);
	else if (options.emit_push_constant_as_uniform_buffer)
		emit_buffer_block_native(var);
	else
		emit_push_constant_block_glsl(var);
}

void CompilerGLSL::emit_push_constant_block_vulkan(const SPIRVariable &var)
{
	emit_buffer_block(var);
}

void CompilerGLSL::emit_push_constant_block_glsl(const SPIRVariable &var)
{
	// OpenGL has no concept of push constant blocks, implement it as a uniform struct.
	auto &type = get<SPIRType>(var.basetype);

	auto &flags = ir.meta[var.self].decoration.decoration_flags;
	flags.clear(DecorationBinding);
	flags.clear(DecorationDescriptorSet);

#if 0
    if (flags & ((1ull << DecorationBinding) | (1ull << DecorationDescriptorSet)))
        SPIRV_CROSS_THROW("Push constant blocks cannot be compiled to GLSL with Binding or Set syntax. "
                            "Remap to location with reflection API first or disable these decorations.");
#endif

	// We're emitting the push constant block as a regular struct, so disable the block qualifier temporarily.
	// Otherwise, we will end up emitting layout() qualifiers on naked structs which is not allowed.
	auto &block_flags = ir.meta[type.self].decoration.decoration_flags;
	bool block_flag = block_flags.get(DecorationBlock);
	block_flags.clear(DecorationBlock);

	emit_struct(type);

	if (block_flag)
		block_flags.set(DecorationBlock);

	emit_uniform(var);
	statement("");
}

void CompilerGLSL::emit_buffer_block(const SPIRVariable &var)
{
	auto &type = get<SPIRType>(var.basetype);
	bool ubo_block = var.storage == StorageClassUniform && has_decoration(type.self, DecorationBlock);

	if (flattened_buffer_blocks.count(var.self))
		emit_buffer_block_flattened(var);
	else if (is_legacy() || (!options.es && options.version == 130) ||
	         (ubo_block && options.emit_uniform_buffer_as_plain_uniforms))
		emit_buffer_block_legacy(var);
	else
		emit_buffer_block_native(var);
}

void CompilerGLSL::emit_buffer_block_legacy(const SPIRVariable &var)
{
	auto &type = get<SPIRType>(var.basetype);
	bool ssbo = var.storage == StorageClassStorageBuffer ||
	            ir.meta[type.self].decoration.decoration_flags.get(DecorationBufferBlock);
	if (ssbo)
		SPIRV_CROSS_THROW("SSBOs not supported in legacy targets.");

	// We're emitting the push constant block as a regular struct, so disable the block qualifier temporarily.
	// Otherwise, we will end up emitting layout() qualifiers on naked structs which is not allowed.
	auto &block_flags = ir.meta[type.self].decoration.decoration_flags;
	bool block_flag = block_flags.get(DecorationBlock);
	block_flags.clear(DecorationBlock);
	emit_struct(type);
	if (block_flag)
		block_flags.set(DecorationBlock);
	emit_uniform(var);
	statement("");
}

void CompilerGLSL::emit_buffer_reference_block(SPIRType &type, bool forward_declaration)
{
	string buffer_name;

	if (forward_declaration)
	{
		// Block names should never alias, but from HLSL input they kind of can because block types are reused for UAVs ...
		// Allow aliased name since we might be declaring the block twice. Once with buffer reference (forward declared) and one proper declaration.
		// The names must match up.
		buffer_name = to_name(type.self, false);

		// Shaders never use the block by interface name, so we don't
		// have to track this other than updating name caches.
		// If we have a collision for any reason, just fallback immediately.
		if (ir.meta[type.self].decoration.alias.empty() ||
		    block_ssbo_names.find(buffer_name) != end(block_ssbo_names) ||
		    resource_names.find(buffer_name) != end(resource_names))
		{
			buffer_name = join("_", type.self);
		}

		// Make sure we get something unique for both global name scope and block name scope.
		// See GLSL 4.5 spec: section 4.3.9 for details.
		add_variable(block_ssbo_names, resource_names, buffer_name);

		// If for some reason buffer_name is an illegal name, make a final fallback to a workaround name.
		// This cannot conflict with anything else, so we're safe now.
		// We cannot reuse this fallback name in neither global scope (blocked by block_names) nor block name scope.
		if (buffer_name.empty())
			buffer_name = join("_", type.self);

		block_names.insert(buffer_name);
		block_ssbo_names.insert(buffer_name);
	}
	else if (type.basetype != SPIRType::Struct)
		buffer_name = type_to_glsl(type);
	else
		buffer_name = to_name(type.self, false);

	if (!forward_declaration)
	{
		if (type.basetype == SPIRType::Struct)
			statement("layout(buffer_reference, ", buffer_to_packing_standard(type, true), ") buffer ", buffer_name);
		else
			statement("layout(buffer_reference) buffer ", buffer_name);

		begin_scope();

		if (type.basetype == SPIRType::Struct)
		{
			type.member_name_cache.clear();

			uint32_t i = 0;
			for (auto &member : type.member_types)
			{
				add_member_name(type, i);
				emit_struct_member(type, member, i);
				i++;
			}
		}
		else
		{
			auto &pointee_type = get_pointee_type(type);
			statement(type_to_glsl(pointee_type), " value", type_to_array_glsl(pointee_type), ";");
		}

		end_scope_decl();
		statement("");
	}
	else
	{
		statement("layout(buffer_reference) buffer ", buffer_name, ";");
	}
}

void CompilerGLSL::emit_buffer_block_native(const SPIRVariable &var)
{
	auto &type = get<SPIRType>(var.basetype);

	Bitset flags = ir.get_buffer_block_flags(var);
	bool ssbo = var.storage == StorageClassStorageBuffer || var.storage == StorageClassShaderRecordBufferNV ||
	            ir.meta[type.self].decoration.decoration_flags.get(DecorationBufferBlock);
	bool is_restrict = ssbo && flags.get(DecorationRestrict);
	bool is_writeonly = ssbo && flags.get(DecorationNonReadable);
	bool is_readonly = ssbo && flags.get(DecorationNonWritable);
	bool is_coherent = ssbo && flags.get(DecorationCoherent);

	// Block names should never alias, but from HLSL input they kind of can because block types are reused for UAVs ...
	auto buffer_name = to_name(type.self, false);

	auto &block_namespace = ssbo ? block_ssbo_names : block_ubo_names;

	// Shaders never use the block by interface name, so we don't
	// have to track this other than updating name caches.
	// If we have a collision for any reason, just fallback immediately.
	if (ir.meta[type.self].decoration.alias.empty() || block_namespace.find(buffer_name) != end(block_namespace) ||
	    resource_names.find(buffer_name) != end(resource_names))
	{
		buffer_name = get_block_fallback_name(var.self);
	}

	// Make sure we get something unique for both global name scope and block name scope.
	// See GLSL 4.5 spec: section 4.3.9 for details.
	add_variable(block_namespace, resource_names, buffer_name);

	// If for some reason buffer_name is an illegal name, make a final fallback to a workaround name.
	// This cannot conflict with anything else, so we're safe now.
	// We cannot reuse this fallback name in neither global scope (blocked by block_names) nor block name scope.
	if (buffer_name.empty())
		buffer_name = join("_", get<SPIRType>(var.basetype).self, "_", var.self);

	block_names.insert(buffer_name);
	block_namespace.insert(buffer_name);

	// Save for post-reflection later.
	declared_block_names[var.self] = buffer_name;

	statement(layout_for_variable(var), is_coherent ? "coherent " : "", is_restrict ? "restrict " : "",
	          is_writeonly ? "writeonly " : "", is_readonly ? "readonly " : "", ssbo ? "buffer " : "uniform ",
	          buffer_name);

	begin_scope();

	type.member_name_cache.clear();

	uint32_t i = 0;
	for (auto &member : type.member_types)
	{
		add_member_name(type, i);
		emit_struct_member(type, member, i);
		i++;
	}

	// var.self can be used as a backup name for the block name,
	// so we need to make sure we don't disturb the name here on a recompile.
	// It will need to be reset if we have to recompile.
	preserve_alias_on_reset(var.self);
	add_resource_name(var.self);
	end_scope_decl(to_name(var.self) + type_to_array_glsl(type));
	statement("");
}

void CompilerGLSL::emit_buffer_block_flattened(const SPIRVariable &var)
{
	auto &type = get<SPIRType>(var.basetype);

	// Block names should never alias.
	auto buffer_name = to_name(type.self, false);
	size_t buffer_size = (get_declared_struct_size(type) + 15) / 16;

	SPIRType::BaseType basic_type;
	if (get_common_basic_type(type, basic_type))
	{
		SPIRType tmp;
		tmp.basetype = basic_type;
		tmp.vecsize = 4;
		if (basic_type != SPIRType::Float && basic_type != SPIRType::Int && basic_type != SPIRType::UInt)
			SPIRV_CROSS_THROW("Basic types in a flattened UBO must be float, int or uint.");

		auto flags = ir.get_buffer_block_flags(var);
		statement("uniform ", flags_to_qualifiers_glsl(tmp, flags), type_to_glsl(tmp), " ", buffer_name, "[",
		          buffer_size, "];");
	}
	else
		SPIRV_CROSS_THROW("All basic types in a flattened block must be the same.");
}

const char *CompilerGLSL::to_storage_qualifiers_glsl(const SPIRVariable &var)
{
	auto &execution = get_entry_point();

	if (subpass_input_is_framebuffer_fetch(var.self))
		return "";

	if (var.storage == StorageClassInput || var.storage == StorageClassOutput)
	{
		if (is_legacy() && execution.model == ExecutionModelVertex)
			return var.storage == StorageClassInput ? "attribute " : "varying ";
		else if (is_legacy() && execution.model == ExecutionModelFragment)
			return "varying "; // Fragment outputs are renamed so they never hit this case.
		else if (execution.model == ExecutionModelFragment && var.storage == StorageClassOutput)
		{
			if (inout_color_attachments.count(get_decoration(var.self, DecorationLocation)) != 0)
				return "inout ";
			else
				return "out ";
		}
		else
			return var.storage == StorageClassInput ? "in " : "out ";
	}
	else if (var.storage == StorageClassUniformConstant || var.storage == StorageClassUniform ||
	         var.storage == StorageClassPushConstant)
	{
		return "uniform ";
	}
	else if (var.storage == StorageClassRayPayloadNV)
	{
		return "rayPayloadNV ";
	}
	else if (var.storage == StorageClassIncomingRayPayloadNV)
	{
		return "rayPayloadInNV ";
	}
	else if (var.storage == StorageClassHitAttributeNV)
	{
		return "hitAttributeNV ";
	}
	else if (var.storage == StorageClassCallableDataNV)
	{
		return "callableDataNV ";
	}
	else if (var.storage == StorageClassIncomingCallableDataNV)
	{
		return "callableDataInNV ";
	}

	return "";
}

void CompilerGLSL::emit_flattened_io_block(const SPIRVariable &var, const char *qual)
{
	auto &type = get<SPIRType>(var.basetype);
	if (!type.array.empty())
		SPIRV_CROSS_THROW("Array of varying structs cannot be flattened to legacy-compatible varyings.");

	auto old_flags = ir.meta[type.self].decoration.decoration_flags;
	// Emit the members as if they are part of a block to get all qualifiers.
	ir.meta[type.self].decoration.decoration_flags.set(DecorationBlock);

	type.member_name_cache.clear();

	uint32_t i = 0;
	for (auto &member : type.member_types)
	{
		add_member_name(type, i);
		auto &membertype = get<SPIRType>(member);

		if (membertype.basetype == SPIRType::Struct)
			SPIRV_CROSS_THROW("Cannot flatten struct inside structs in I/O variables.");

		// Pass in the varying qualifier here so it will appear in the correct declaration order.
		// Replace member name while emitting it so it encodes both struct name and member name.
		// Sanitize underscores because joining the two identifiers might create more than 1 underscore in a row,
		// which is not allowed.
		auto backup_name = get_member_name(type.self, i);
		auto member_name = to_member_name(type, i);
		set_member_name(type.self, i, sanitize_underscores(join(to_name(var.self), "_", member_name)));
		emit_struct_member(type, member, i, qual);
		// Restore member name.
		set_member_name(type.self, i, member_name);
		i++;
	}

	ir.meta[type.self].decoration.decoration_flags = old_flags;

	// Treat this variable as flattened from now on.
	flattened_structs.insert(var.self);
}

void CompilerGLSL::emit_interface_block(const SPIRVariable &var)
{
	auto &type = get<SPIRType>(var.basetype);

	// Either make it plain in/out or in/out blocks depending on what shader is doing ...
	bool block = ir.meta[type.self].decoration.decoration_flags.get(DecorationBlock);
	const char *qual = to_storage_qualifiers_glsl(var);

	if (block)
	{
		// ESSL earlier than 310 and GLSL earlier than 150 did not support
		// I/O variables which are struct types.
		// To support this, flatten the struct into separate varyings instead.
		if ((options.es && options.version < 310) || (!options.es && options.version < 150))
		{
			// I/O blocks on ES require version 310 with Android Extension Pack extensions, or core version 320.
			// On desktop, I/O blocks were introduced with geometry shaders in GL 3.2 (GLSL 150).
			emit_flattened_io_block(var, qual);
		}
		else
		{
			if (options.es && options.version < 320)
			{
				// Geometry and tessellation extensions imply this extension.
				if (!has_extension("GL_EXT_geometry_shader") && !has_extension("GL_EXT_tessellation_shader"))
					require_extension_internal("GL_EXT_shader_io_blocks");
			}

			// Block names should never alias.
			auto block_name = to_name(type.self, false);

			// The namespace for I/O blocks is separate from other variables in GLSL.
			auto &block_namespace = type.storage == StorageClassInput ? block_input_names : block_output_names;

			// Shaders never use the block by interface name, so we don't
			// have to track this other than updating name caches.
			if (block_name.empty() || block_namespace.find(block_name) != end(block_namespace))
				block_name = get_fallback_name(type.self);
			else
				block_namespace.insert(block_name);

			// If for some reason buffer_name is an illegal name, make a final fallback to a workaround name.
			// This cannot conflict with anything else, so we're safe now.
			if (block_name.empty())
				block_name = join("_", get<SPIRType>(var.basetype).self, "_", var.self);

			// Instance names cannot alias block names.
			resource_names.insert(block_name);

			statement(layout_for_variable(var), qual, block_name);
			begin_scope();

			type.member_name_cache.clear();

			uint32_t i = 0;
			for (auto &member : type.member_types)
			{
				add_member_name(type, i);
				emit_struct_member(type, member, i);
				i++;
			}

			add_resource_name(var.self);
			end_scope_decl(join(to_name(var.self), type_to_array_glsl(type)));
			statement("");
		}
	}
	else
	{
		// ESSL earlier than 310 and GLSL earlier than 150 did not support
		// I/O variables which are struct types.
		// To support this, flatten the struct into separate varyings instead.
		if (type.basetype == SPIRType::Struct &&
		    ((options.es && options.version < 310) || (!options.es && options.version < 150)))
		{
			emit_flattened_io_block(var, qual);
		}
		else
		{
			add_resource_name(var.self);

			// Tessellation control and evaluation shaders must have either gl_MaxPatchVertices or unsized arrays for input arrays.
			// Opt for unsized as it's the more "correct" variant to use.
			bool control_point_input_array = type.storage == StorageClassInput && !type.array.empty() &&
			                                 !has_decoration(var.self, DecorationPatch) &&
			                                 (get_entry_point().model == ExecutionModelTessellationControl ||
			                                  get_entry_point().model == ExecutionModelTessellationEvaluation);

			uint32_t old_array_size = 0;
			bool old_array_size_literal = true;

			if (control_point_input_array)
			{
				swap(type.array.back(), old_array_size);
				swap(type.array_size_literal.back(), old_array_size_literal);
			}

			statement(layout_for_variable(var), to_qualifiers_glsl(var.self),
			          variable_decl(type, to_name(var.self), var.self), ";");

			if (control_point_input_array)
			{
				swap(type.array.back(), old_array_size);
				swap(type.array_size_literal.back(), old_array_size_literal);
			}

			// If a StorageClassOutput variable has an initializer, we need to initialize it in main().
			if (var.storage == StorageClassOutput && var.initializer)
			{
				auto &entry_func = this->get<SPIRFunction>(ir.default_entry_point);
				entry_func.fixup_hooks_in.push_back(
				    [&]() { statement(to_name(var.self), " = ", to_expression(var.initializer), ";"); });
			}
		}
	}
}

void CompilerGLSL::emit_uniform(const SPIRVariable &var)
{
	auto &type = get<SPIRType>(var.basetype);
	if (type.basetype == SPIRType::Image && type.image.sampled == 2 && type.image.dim != DimSubpassData)
	{
		if (!options.es && options.version < 420)
			require_extension_internal("GL_ARB_shader_image_load_store");
		else if (options.es && options.version < 310)
			SPIRV_CROSS_THROW("At least ESSL 3.10 required for shader image load store.");
	}

	add_resource_name(var.self);
	statement(layout_for_variable(var), variable_decl(var), ";");
}

string CompilerGLSL::constant_value_macro_name(uint32_t id)
{
	return join("SPIRV_CROSS_CONSTANT_ID_", id);
}

void CompilerGLSL::emit_specialization_constant_op(const SPIRConstantOp &constant)
{
	auto &type = get<SPIRType>(constant.basetype);
	auto name = to_name(constant.self);
	statement("const ", variable_decl(type, name), " = ", constant_op_expression(constant), ";");
}

void CompilerGLSL::emit_constant(const SPIRConstant &constant)
{
	auto &type = get<SPIRType>(constant.constant_type);
	auto name = to_name(constant.self);

	SpecializationConstant wg_x, wg_y, wg_z;
	ID workgroup_size_id = get_work_group_size_specialization_constants(wg_x, wg_y, wg_z);

	// This specialization constant is implicitly declared by emitting layout() in;
	if (constant.self == workgroup_size_id)
		return;

	// These specialization constants are implicitly declared by emitting layout() in;
	// In legacy GLSL, we will still need to emit macros for these, so a layout() in; declaration
	// later can use macro overrides for work group size.
	bool is_workgroup_size_constant = ConstantID(constant.self) == wg_x.id || ConstantID(constant.self) == wg_y.id ||
	                                  ConstantID(constant.self) == wg_z.id;

	if (options.vulkan_semantics && is_workgroup_size_constant)
	{
		// Vulkan GLSL does not need to declare workgroup spec constants explicitly, it is handled in layout().
		return;
	}
	else if (!options.vulkan_semantics && is_workgroup_size_constant &&
	         !has_decoration(constant.self, DecorationSpecId))
	{
		// Only bother declaring a workgroup size if it is actually a specialization constant, because we need macros.
		return;
	}

	// Only scalars have constant IDs.
	if (has_decoration(constant.self, DecorationSpecId))
	{
		if (options.vulkan_semantics)
		{
			statement("layout(constant_id = ", get_decoration(constant.self, DecorationSpecId), ") const ",
			          variable_decl(type, name), " = ", constant_expression(constant), ";");
		}
		else
		{
			const string &macro_name = constant.specialization_constant_macro_name;
			statement("#ifndef ", macro_name);
			statement("#define ", macro_name, " ", constant_expression(constant));
			statement("#endif");

			// For workgroup size constants, only emit the macros.
			if (!is_workgroup_size_constant)
				statement("const ", variable_decl(type, name), " = ", macro_name, ";");
		}
	}
	else
	{
		statement("const ", variable_decl(type, name), " = ", constant_expression(constant), ";");
	}
}

void CompilerGLSL::emit_entry_point_declarations()
{
}

void CompilerGLSL::replace_illegal_names(const unordered_set<string> &keywords)
{
	ir.for_each_typed_id<SPIRVariable>([&](uint32_t, const SPIRVariable &var) {
		if (is_hidden_variable(var))
			return;

		auto *meta = ir.find_meta(var.self);
		if (!meta)
			return;

		auto &m = meta->decoration;
		if (m.alias.compare(0, 3, "gl_") == 0 || keywords.find(m.alias) != end(keywords))
			m.alias = join("_", m.alias);
	});

	ir.for_each_typed_id<SPIRType>([&](uint32_t, const SPIRType &type) {
		auto *meta = ir.find_meta(type.self);
		if (!meta)
			return;

		auto &m = meta->decoration;
		if (m.alias.compare(0, 3, "gl_") == 0 || keywords.find(m.alias) != end(keywords))
			m.alias = join("_", m.alias);

		for (auto &memb : meta->members)
			if (memb.alias.compare(0, 3, "gl_") == 0 || keywords.find(memb.alias) != end(keywords))
				memb.alias = join("_", memb.alias);
	});
}

void CompilerGLSL::replace_illegal_names()
{
	// clang-format off
	static const unordered_set<string> keywords = {
		"abs", "acos", "acosh", "all", "any", "asin", "asinh", "atan", "atanh",
		"atomicAdd", "atomicCompSwap", "atomicCounter", "atomicCounterDecrement", "atomicCounterIncrement",
		"atomicExchange", "atomicMax", "atomicMin", "atomicOr", "atomicXor",
		"bitCount", "bitfieldExtract", "bitfieldInsert", "bitfieldReverse",
		"ceil", "cos", "cosh", "cross", "degrees",
		"dFdx", "dFdxCoarse", "dFdxFine",
		"dFdy", "dFdyCoarse", "dFdyFine",
		"distance", "dot", "EmitStreamVertex", "EmitVertex", "EndPrimitive", "EndStreamPrimitive", "equal", "exp", "exp2",
		"faceforward", "findLSB", "findMSB", "float16BitsToInt16", "float16BitsToUint16", "floatBitsToInt", "floatBitsToUint", "floor", "fma", "fract",
		"frexp", "fwidth", "fwidthCoarse", "fwidthFine",
		"greaterThan", "greaterThanEqual", "groupMemoryBarrier",
		"imageAtomicAdd", "imageAtomicAnd", "imageAtomicCompSwap", "imageAtomicExchange", "imageAtomicMax", "imageAtomicMin", "imageAtomicOr", "imageAtomicXor",
		"imageLoad", "imageSamples", "imageSize", "imageStore", "imulExtended", "int16BitsToFloat16", "intBitsToFloat", "interpolateAtOffset", "interpolateAtCentroid", "interpolateAtSample",
		"inverse", "inversesqrt", "isinf", "isnan", "ldexp", "length", "lessThan", "lessThanEqual", "log", "log2",
		"matrixCompMult", "max", "memoryBarrier", "memoryBarrierAtomicCounter", "memoryBarrierBuffer", "memoryBarrierImage", "memoryBarrierShared",
		"min", "mix", "mod", "modf", "noise", "noise1", "noise2", "noise3", "noise4", "normalize", "not", "notEqual",
		"outerProduct", "packDouble2x32", "packHalf2x16", "packInt2x16", "packInt4x16", "packSnorm2x16", "packSnorm4x8",
		"packUint2x16", "packUint4x16", "packUnorm2x16", "packUnorm4x8", "pow",
		"radians", "reflect", "refract", "round", "roundEven", "sign", "sin", "sinh", "smoothstep", "sqrt", "step",
		"tan", "tanh", "texelFetch", "texelFetchOffset", "texture", "textureGather", "textureGatherOffset", "textureGatherOffsets",
		"textureGrad", "textureGradOffset", "textureLod", "textureLodOffset", "textureOffset", "textureProj", "textureProjGrad",
		"textureProjGradOffset", "textureProjLod", "textureProjLodOffset", "textureProjOffset", "textureQueryLevels", "textureQueryLod", "textureSamples", "textureSize",
		"transpose", "trunc", "uaddCarry", "uint16BitsToFloat16", "uintBitsToFloat", "umulExtended", "unpackDouble2x32", "unpackHalf2x16", "unpackInt2x16", "unpackInt4x16",
		"unpackSnorm2x16", "unpackSnorm4x8", "unpackUint2x16", "unpackUint4x16", "unpackUnorm2x16", "unpackUnorm4x8", "usubBorrow",

		"active", "asm", "atomic_uint", "attribute", "bool", "break", "buffer",
		"bvec2", "bvec3", "bvec4", "case", "cast", "centroid", "class", "coherent", "common", "const", "continue", "default", "discard",
		"dmat2", "dmat2x2", "dmat2x3", "dmat2x4", "dmat3", "dmat3x2", "dmat3x3", "dmat3x4", "dmat4", "dmat4x2", "dmat4x3", "dmat4x4",
		"do", "double", "dvec2", "dvec3", "dvec4", "else", "enum", "extern", "external", "false", "filter", "fixed", "flat", "float",
		"for", "fvec2", "fvec3", "fvec4", "goto", "half", "highp", "hvec2", "hvec3", "hvec4", "if", "iimage1D", "iimage1DArray",
		"iimage2D", "iimage2DArray", "iimage2DMS", "iimage2DMSArray", "iimage2DRect", "iimage3D", "iimageBuffer", "iimageCube",
		"iimageCubeArray", "image1D", "image1DArray", "image2D", "image2DArray", "image2DMS", "image2DMSArray", "image2DRect",
		"image3D", "imageBuffer", "imageCube", "imageCubeArray", "in", "inline", "inout", "input", "int", "interface", "invariant",
		"isampler1D", "isampler1DArray", "isampler2D", "isampler2DArray", "isampler2DMS", "isampler2DMSArray", "isampler2DRect",
		"isampler3D", "isamplerBuffer", "isamplerCube", "isamplerCubeArray", "ivec2", "ivec3", "ivec4", "layout", "long", "lowp",
		"mat2", "mat2x2", "mat2x3", "mat2x4", "mat3", "mat3x2", "mat3x3", "mat3x4", "mat4", "mat4x2", "mat4x3", "mat4x4", "mediump",
		"namespace", "noinline", "noperspective", "out", "output", "packed", "partition", "patch", "precise", "precision", "public", "readonly",
		"resource", "restrict", "return", "sample", "sampler1D", "sampler1DArray", "sampler1DArrayShadow",
		"sampler1DShadow", "sampler2D", "sampler2DArray", "sampler2DArrayShadow", "sampler2DMS", "sampler2DMSArray",
		"sampler2DRect", "sampler2DRectShadow", "sampler2DShadow", "sampler3D", "sampler3DRect", "samplerBuffer",
		"samplerCube", "samplerCubeArray", "samplerCubeArrayShadow", "samplerCubeShadow", "shared", "short", "sizeof", "smooth", "static",
		"struct", "subroutine", "superp", "switch", "template", "this", "true", "typedef", "uimage1D", "uimage1DArray", "uimage2D",
		"uimage2DArray", "uimage2DMS", "uimage2DMSArray", "uimage2DRect", "uimage3D", "uimageBuffer", "uimageCube",
		"uimageCubeArray", "uint", "uniform", "union", "unsigned", "usampler1D", "usampler1DArray", "usampler2D", "usampler2DArray",
		"usampler2DMS", "usampler2DMSArray", "usampler2DRect", "usampler3D", "usamplerBuffer", "usamplerCube",
		"usamplerCubeArray", "using", "uvec2", "uvec3", "uvec4", "varying", "vec2", "vec3", "vec4", "void", "volatile",
		"while", "writeonly",
	};
	// clang-format on

	replace_illegal_names(keywords);
}

void CompilerGLSL::replace_fragment_output(SPIRVariable &var)
{
	auto &m = ir.meta[var.self].decoration;
	uint32_t location = 0;
	if (m.decoration_flags.get(DecorationLocation))
		location = m.location;

	// If our variable is arrayed, we must not emit the array part of this as the SPIR-V will
	// do the access chain part of this for us.
	auto &type = get<SPIRType>(var.basetype);

	if (type.array.empty())
	{
		// Redirect the write to a specific render target in legacy GLSL.
		m.alias = join("gl_FragData[", location, "]");

		if (is_legacy_es() && location != 0)
			require_extension_internal("GL_EXT_draw_buffers");
	}
	else if (type.array.size() == 1)
	{
		// If location is non-zero, we probably have to add an offset.
		// This gets really tricky since we'd have to inject an offset in the access chain.
		// FIXME: This seems like an extremely odd-ball case, so it's probably fine to leave it like this for now.
		m.alias = "gl_FragData";
		if (location != 0)
			SPIRV_CROSS_THROW("Arrayed output variable used, but location is not 0. "
			                  "This is unimplemented in SPIRV-Cross.");

		if (is_legacy_es())
			require_extension_internal("GL_EXT_draw_buffers");
	}
	else
		SPIRV_CROSS_THROW("Array-of-array output variable used. This cannot be implemented in legacy GLSL.");

	var.compat_builtin = true; // We don't want to declare this variable, but use the name as-is.
}

void CompilerGLSL::replace_fragment_outputs()
{
	ir.for_each_typed_id<SPIRVariable>([&](uint32_t, SPIRVariable &var) {
		auto &type = this->get<SPIRType>(var.basetype);

		if (!is_builtin_variable(var) && !var.remapped_variable && type.pointer && var.storage == StorageClassOutput)
			replace_fragment_output(var);
	});
}

string CompilerGLSL::remap_swizzle(const SPIRType &out_type, uint32_t input_components, const string &expr)
{
	if (out_type.vecsize == input_components)
		return expr;
	else if (input_components == 1 && !backend.can_swizzle_scalar)
		return join(type_to_glsl(out_type), "(", expr, ")");
	else
	{
		// FIXME: This will not work with packed expressions.
		auto e = enclose_expression(expr) + ".";
		// Just clamp the swizzle index if we have more outputs than inputs.
		for (uint32_t c = 0; c < out_type.vecsize; c++)
			e += index_to_swizzle(min(c, input_components - 1));
		if (backend.swizzle_is_function && out_type.vecsize > 1)
			e += "()";

		remove_duplicate_swizzle(e);
		return e;
	}
}

void CompilerGLSL::emit_pls()
{
	auto &execution = get_entry_point();
	if (execution.model != ExecutionModelFragment)
		SPIRV_CROSS_THROW("Pixel local storage only supported in fragment shaders.");

	if (!options.es)
		SPIRV_CROSS_THROW("Pixel local storage only supported in OpenGL ES.");

	if (options.version < 300)
		SPIRV_CROSS_THROW("Pixel local storage only supported in ESSL 3.0 and above.");

	if (!pls_inputs.empty())
	{
		statement("__pixel_local_inEXT _PLSIn");
		begin_scope();
		for (auto &input : pls_inputs)
			statement(pls_decl(input), ";");
		end_scope_decl();
		statement("");
	}

	if (!pls_outputs.empty())
	{
		statement("__pixel_local_outEXT _PLSOut");
		begin_scope();
		for (auto &output : pls_outputs)
			statement(pls_decl(output), ";");
		end_scope_decl();
		statement("");
	}
}

void CompilerGLSL::fixup_image_load_store_access()
{
	if (!options.enable_storage_image_qualifier_deduction)
		return;

	ir.for_each_typed_id<SPIRVariable>([&](uint32_t var, const SPIRVariable &) {
		auto &vartype = expression_type(var);
		if (vartype.basetype == SPIRType::Image && vartype.image.sampled == 2)
		{
			// Very old glslangValidator and HLSL compilers do not emit required qualifiers here.
			// Solve this by making the image access as restricted as possible and loosen up if we need to.
			// If any no-read/no-write flags are actually set, assume that the compiler knows what it's doing.

			auto &flags = ir.meta[var].decoration.decoration_flags;
			if (!flags.get(DecorationNonWritable) && !flags.get(DecorationNonReadable))
			{
				flags.set(DecorationNonWritable);
				flags.set(DecorationNonReadable);
			}
		}
	});
}

static bool is_block_builtin(BuiltIn builtin)
{
	return builtin == BuiltInPosition || builtin == BuiltInPointSize || builtin == BuiltInClipDistance ||
	       builtin == BuiltInCullDistance;
}

bool CompilerGLSL::should_force_emit_builtin_block(StorageClass storage)
{
	// If the builtin block uses XFB, we need to force explicit redeclaration of the builtin block.

	if (storage != StorageClassOutput)
		return false;
	bool should_force = false;

	ir.for_each_typed_id<SPIRVariable>([&](uint32_t, SPIRVariable &var) {
		if (should_force)
			return;

		auto &type = this->get<SPIRType>(var.basetype);
		bool block = has_decoration(type.self, DecorationBlock);
		if (var.storage == storage && block && is_builtin_variable(var))
		{
			uint32_t member_count = uint32_t(type.member_types.size());
			for (uint32_t i = 0; i < member_count; i++)
			{
				if (has_member_decoration(type.self, i, DecorationBuiltIn) &&
				    is_block_builtin(BuiltIn(get_member_decoration(type.self, i, DecorationBuiltIn))) &&
				    has_member_decoration(type.self, i, DecorationOffset))
				{
					should_force = true;
				}
			}
		}
		else if (var.storage == storage && !block && is_builtin_variable(var))
		{
			if (is_block_builtin(BuiltIn(get_decoration(type.self, DecorationBuiltIn))) &&
			    has_decoration(var.self, DecorationOffset))
			{
				should_force = true;
			}
		}
	});

	return should_force;
}

void CompilerGLSL::emit_declared_builtin_block(StorageClass storage, ExecutionModel model)
{
	Bitset emitted_builtins;
	Bitset global_builtins;
	const SPIRVariable *block_var = nullptr;
	bool emitted_block = false;
	bool builtin_array = false;

	// Need to use declared size in the type.
	// These variables might have been declared, but not statically used, so we haven't deduced their size yet.
	uint32_t cull_distance_size = 0;
	uint32_t clip_distance_size = 0;

	bool have_xfb_buffer_stride = false;
	bool have_any_xfb_offset = false;
	uint32_t xfb_stride = 0, xfb_buffer = 0;
	std::unordered_map<uint32_t, uint32_t> builtin_xfb_offsets;

	ir.for_each_typed_id<SPIRVariable>([&](uint32_t, SPIRVariable &var) {
		auto &type = this->get<SPIRType>(var.basetype);
		bool block = has_decoration(type.self, DecorationBlock);
		Bitset builtins;

		if (var.storage == storage && block && is_builtin_variable(var))
		{
			uint32_t index = 0;
			for (auto &m : ir.meta[type.self].members)
			{
				if (m.builtin)
				{
					builtins.set(m.builtin_type);
					if (m.builtin_type == BuiltInCullDistance)
						cull_distance_size = this->get<SPIRType>(type.member_types[index]).array.front();
					else if (m.builtin_type == BuiltInClipDistance)
						clip_distance_size = this->get<SPIRType>(type.member_types[index]).array.front();

					if (is_block_builtin(m.builtin_type) && m.decoration_flags.get(DecorationOffset))
					{
						have_any_xfb_offset = true;
						builtin_xfb_offsets[m.builtin_type] = m.offset;
					}
				}
				index++;
			}

			if (storage == StorageClassOutput && has_decoration(var.self, DecorationXfbBuffer) &&
			    has_decoration(var.self, DecorationXfbStride))
			{
				uint32_t buffer_index = get_decoration(var.self, DecorationXfbBuffer);
				uint32_t stride = get_decoration(var.self, DecorationXfbStride);
				if (have_xfb_buffer_stride && buffer_index != xfb_buffer)
					SPIRV_CROSS_THROW("IO block member XfbBuffer mismatch.");
				if (have_xfb_buffer_stride && stride != xfb_stride)
					SPIRV_CROSS_THROW("IO block member XfbBuffer mismatch.");
				have_xfb_buffer_stride = true;
				xfb_buffer = buffer_index;
				xfb_stride = stride;
			}
		}
		else if (var.storage == storage && !block && is_builtin_variable(var))
		{
			// While we're at it, collect all declared global builtins (HLSL mostly ...).
			auto &m = ir.meta[var.self].decoration;
			if (m.builtin)
			{
				global_builtins.set(m.builtin_type);
				if (m.builtin_type == BuiltInCullDistance)
					cull_distance_size = type.array.front();
				else if (m.builtin_type == BuiltInClipDistance)
					clip_distance_size = type.array.front();

				if (is_block_builtin(m.builtin_type) && m.decoration_flags.get(DecorationXfbStride) &&
				    m.decoration_flags.get(DecorationXfbBuffer) && m.decoration_flags.get(DecorationOffset))
				{
					have_any_xfb_offset = true;
					builtin_xfb_offsets[m.builtin_type] = m.offset;
					uint32_t buffer_index = m.xfb_buffer;
					uint32_t stride = m.xfb_stride;
					if (have_xfb_buffer_stride && buffer_index != xfb_buffer)
						SPIRV_CROSS_THROW("IO block member XfbBuffer mismatch.");
					if (have_xfb_buffer_stride && stride != xfb_stride)
						SPIRV_CROSS_THROW("IO block member XfbBuffer mismatch.");
					have_xfb_buffer_stride = true;
					xfb_buffer = buffer_index;
					xfb_stride = stride;
				}
			}
		}

		if (builtins.empty())
			return;

		if (emitted_block)
			SPIRV_CROSS_THROW("Cannot use more than one builtin I/O block.");

		emitted_builtins = builtins;
		emitted_block = true;
		builtin_array = !type.array.empty();
		block_var = &var;
	});

	global_builtins =
	    Bitset(global_builtins.get_lower() & ((1ull << BuiltInPosition) | (1ull << BuiltInPointSize) |
	                                          (1ull << BuiltInClipDistance) | (1ull << BuiltInCullDistance)));

	// Try to collect all other declared builtins.
	if (!emitted_block)
		emitted_builtins = global_builtins;

	// Can't declare an empty interface block.
	if (emitted_builtins.empty())
		return;

	if (storage == StorageClassOutput)
	{
		if (have_xfb_buffer_stride && have_any_xfb_offset)
		{
			statement("layout(xfb_buffer = ", xfb_buffer, ", xfb_stride = ", xfb_stride, ") out gl_PerVertex");
			if (!options.es)
			{
				if (options.version < 440 && options.version >= 140)
					require_extension_internal("GL_ARB_enhanced_layouts");
				else if (options.version < 140)
					SPIRV_CROSS_THROW("Component decoration is not supported in targets below GLSL 1.40.");
				if (!options.es && options.version < 440)
					require_extension_internal("GL_ARB_enhanced_layouts");
			}
			else if (options.es)
				SPIRV_CROSS_THROW("Need GL_ARB_enhanced_layouts for xfb_stride or xfb_buffer.");
		}
		else
			statement("out gl_PerVertex");
	}
	else
	{
		// If we have passthrough, there is no way PerVertex cannot be passthrough.
		if (get_entry_point().geometry_passthrough)
			statement("layout(passthrough) in gl_PerVertex");
		else
			statement("in gl_PerVertex");
	}

	begin_scope();
	if (emitted_builtins.get(BuiltInPosition))
	{
		auto itr = builtin_xfb_offsets.find(BuiltInPosition);
		if (itr != end(builtin_xfb_offsets))
			statement("layout(xfb_offset = ", itr->second, ") vec4 gl_Position;");
		else
			statement("vec4 gl_Position;");
	}

	if (emitted_builtins.get(BuiltInPointSize))
	{
		auto itr = builtin_xfb_offsets.find(BuiltInPointSize);
		if (itr != end(builtin_xfb_offsets))
			statement("layout(xfb_offset = ", itr->second, ") float gl_PointSize;");
		else
			statement("float gl_PointSize;");
	}

	if (emitted_builtins.get(BuiltInClipDistance))
	{
		auto itr = builtin_xfb_offsets.find(BuiltInClipDistance);
		if (itr != end(builtin_xfb_offsets))
			statement("layout(xfb_offset = ", itr->second, ") float gl_ClipDistance[", clip_distance_size, "];");
		else
			statement("float gl_ClipDistance[", clip_distance_size, "];");
	}

	if (emitted_builtins.get(BuiltInCullDistance))
	{
		auto itr = builtin_xfb_offsets.find(BuiltInCullDistance);
		if (itr != end(builtin_xfb_offsets))
			statement("layout(xfb_offset = ", itr->second, ") float gl_CullDistance[", cull_distance_size, "];");
		else
			statement("float gl_CullDistance[", cull_distance_size, "];");
	}

	if (builtin_array)
	{
		// Make sure the array has a supported name in the code.
		if (storage == StorageClassOutput)
			set_name(block_var->self, "gl_out");
		else if (storage == StorageClassInput)
			set_name(block_var->self, "gl_in");

		if (model == ExecutionModelTessellationControl && storage == StorageClassOutput)
			end_scope_decl(join(to_name(block_var->self), "[", get_entry_point().output_vertices, "]"));
		else
			end_scope_decl(join(to_name(block_var->self), "[]"));
	}
	else
		end_scope_decl();
	statement("");
}

void CompilerGLSL::declare_undefined_values()
{
	bool emitted = false;
	ir.for_each_typed_id<SPIRUndef>([&](uint32_t, const SPIRUndef &undef) {
		string initializer;
		if (options.force_zero_initialized_variables && type_can_zero_initialize(this->get<SPIRType>(undef.basetype)))
			initializer = join(" = ", to_zero_initialized_expression(undef.basetype));

		statement(variable_decl(this->get<SPIRType>(undef.basetype), to_name(undef.self), undef.self), initializer,
		          ";");
		emitted = true;
	});

	if (emitted)
		statement("");
}

bool CompilerGLSL::variable_is_lut(const SPIRVariable &var) const
{
	bool statically_assigned = var.statically_assigned && var.static_expression != ID(0) && var.remapped_variable;

	if (statically_assigned)
	{
		auto *constant = maybe_get<SPIRConstant>(var.static_expression);
		if (constant && constant->is_used_as_lut)
			return true;
	}

	return false;
}

void CompilerGLSL::emit_resources()
{
	auto &execution = get_entry_point();

	replace_illegal_names();

	// Legacy GL uses gl_FragData[], redeclare all fragment outputs
	// with builtins.
	if (execution.model == ExecutionModelFragment && is_legacy())
		replace_fragment_outputs();

	// Emit PLS blocks if we have such variables.
	if (!pls_inputs.empty() || !pls_outputs.empty())
		emit_pls();

	// Emit custom gl_PerVertex for SSO compatibility.
	if (options.separate_shader_objects && !options.es && execution.model != ExecutionModelFragment)
	{
		switch (execution.model)
		{
		case ExecutionModelGeometry:
		case ExecutionModelTessellationControl:
		case ExecutionModelTessellationEvaluation:
			emit_declared_builtin_block(StorageClassInput, execution.model);
			emit_declared_builtin_block(StorageClassOutput, execution.model);
			break;

		case ExecutionModelVertex:
			emit_declared_builtin_block(StorageClassOutput, execution.model);
			break;

		default:
			break;
		}
	}
	else if (should_force_emit_builtin_block(StorageClassOutput))
	{
		emit_declared_builtin_block(StorageClassOutput, execution.model);
	}
	else if (execution.geometry_passthrough)
	{
		// Need to declare gl_in with Passthrough.
		// If we're doing passthrough, we cannot emit an output block, so the output block test above will never pass.
		emit_declared_builtin_block(StorageClassInput, execution.model);
	}
	else
	{
		// Need to redeclare clip/cull distance with explicit size to use them.
		// SPIR-V mandates these builtins have a size declared.
		const char *storage = execution.model == ExecutionModelFragment ? "in" : "out";
		if (clip_distance_count != 0)
			statement(storage, " float gl_ClipDistance[", clip_distance_count, "];");
		if (cull_distance_count != 0)
			statement(storage, " float gl_CullDistance[", cull_distance_count, "];");
		if (clip_distance_count != 0 || cull_distance_count != 0)
			statement("");
	}

	if (position_invariant)
	{
		statement("invariant gl_Position;");
		statement("");
	}

	bool emitted = false;

	// If emitted Vulkan GLSL,
	// emit specialization constants as actual floats,
	// spec op expressions will redirect to the constant name.
	//
	{
		auto loop_lock = ir.create_loop_hard_lock();
		for (auto &id_ : ir.ids_for_constant_or_type)
		{
			auto &id = ir.ids[id_];

			if (id.get_type() == TypeConstant)
			{
				auto &c = id.get<SPIRConstant>();

				bool needs_declaration = c.specialization || c.is_used_as_lut;

				if (needs_declaration)
				{
					if (!options.vulkan_semantics && c.specialization)
					{
						c.specialization_constant_macro_name =
						    constant_value_macro_name(get_decoration(c.self, DecorationSpecId));
					}
					emit_constant(c);
					emitted = true;
				}
			}
			else if (id.get_type() == TypeConstantOp)
			{
				emit_specialization_constant_op(id.get<SPIRConstantOp>());
				emitted = true;
			}
			else if (id.get_type() == TypeType)
			{
				auto &type = id.get<SPIRType>();
				if (type.basetype == SPIRType::Struct && type.array.empty() && !type.pointer &&
				    (!ir.meta[type.self].decoration.decoration_flags.get(DecorationBlock) &&
				     !ir.meta[type.self].decoration.decoration_flags.get(DecorationBufferBlock)))
				{
					if (emitted)
						statement("");
					emitted = false;

					emit_struct(type);
				}
			}
		}
	}

	if (emitted)
		statement("");

	// If we needed to declare work group size late, check here.
	// If the work group size depends on a specialization constant, we need to declare the layout() block
	// after constants (and their macros) have been declared.
	if (execution.model == ExecutionModelGLCompute && !options.vulkan_semantics &&
	    execution.workgroup_size.constant != 0)
	{
		SpecializationConstant wg_x, wg_y, wg_z;
		get_work_group_size_specialization_constants(wg_x, wg_y, wg_z);

		if ((wg_x.id != ConstantID(0)) || (wg_y.id != ConstantID(0)) || (wg_z.id != ConstantID(0)))
		{
			SmallVector<string> inputs;
			build_workgroup_size(inputs, wg_x, wg_y, wg_z);
			statement("layout(", merge(inputs), ") in;");
			statement("");
		}
	}

	emitted = false;

	if (ir.addressing_model == AddressingModelPhysicalStorageBuffer64EXT)
	{
		for (auto type : physical_storage_non_block_pointer_types)
		{
			emit_buffer_reference_block(get<SPIRType>(type), false);
		}

		// Output buffer reference blocks.
		// Do this in two stages, one with forward declaration,
		// and one without. Buffer reference blocks can reference themselves
		// to support things like linked lists.
		ir.for_each_typed_id<SPIRType>([&](uint32_t, SPIRType &type) {
			bool has_block_flags = has_decoration(type.self, DecorationBlock);
			if (has_block_flags && type.pointer && type.pointer_depth == 1 && !type_is_array_of_pointers(type) &&
			    type.storage == StorageClassPhysicalStorageBufferEXT)
			{
				emit_buffer_reference_block(type, true);
			}
		});

		ir.for_each_typed_id<SPIRType>([&](uint32_t, SPIRType &type) {
			bool has_block_flags = has_decoration(type.self, DecorationBlock);
			if (has_block_flags && type.pointer && type.pointer_depth == 1 && !type_is_array_of_pointers(type) &&
			    type.storage == StorageClassPhysicalStorageBufferEXT)
			{
				emit_buffer_reference_block(type, false);
			}
		});
	}

	// Output UBOs and SSBOs
	ir.for_each_typed_id<SPIRVariable>([&](uint32_t, SPIRVariable &var) {
		auto &type = this->get<SPIRType>(var.basetype);

		bool is_block_storage = type.storage == StorageClassStorageBuffer || type.storage == StorageClassUniform ||
		                        type.storage == StorageClassShaderRecordBufferNV;
		bool has_block_flags = ir.meta[type.self].decoration.decoration_flags.get(DecorationBlock) ||
		                       ir.meta[type.self].decoration.decoration_flags.get(DecorationBufferBlock);

		if (var.storage != StorageClassFunction && type.pointer && is_block_storage && !is_hidden_variable(var) &&
		    has_block_flags)
		{
			emit_buffer_block(var);
		}
	});

	// Output push constant blocks
	ir.for_each_typed_id<SPIRVariable>([&](uint32_t, SPIRVariable &var) {
		auto &type = this->get<SPIRType>(var.basetype);
		if (var.storage != StorageClassFunction && type.pointer && type.storage == StorageClassPushConstant &&
		    !is_hidden_variable(var))
		{
			emit_push_constant_block(var);
		}
	});

	bool skip_separate_image_sampler = !combined_image_samplers.empty() || !options.vulkan_semantics;

	// Output Uniform Constants (values, samplers, images, etc).
	ir.for_each_typed_id<SPIRVariable>([&](uint32_t, SPIRVariable &var) {
		auto &type = this->get<SPIRType>(var.basetype);

		// If we're remapping separate samplers and images, only emit the combined samplers.
		if (skip_separate_image_sampler)
		{
			// Sampler buffers are always used without a sampler, and they will also work in regular GL.
			bool sampler_buffer = type.basetype == SPIRType::Image && type.image.dim == DimBuffer;
			bool separate_image = type.basetype == SPIRType::Image && type.image.sampled == 1;
			bool separate_sampler = type.basetype == SPIRType::Sampler;
			if (!sampler_buffer && (separate_image || separate_sampler))
				return;
		}

		if (var.storage != StorageClassFunction && type.pointer &&
		    (type.storage == StorageClassUniformConstant || type.storage == StorageClassAtomicCounter ||
		     type.storage == StorageClassRayPayloadNV || type.storage == StorageClassIncomingRayPayloadNV ||
		     type.storage == StorageClassCallableDataNV || type.storage == StorageClassIncomingCallableDataNV ||
		     type.storage == StorageClassHitAttributeNV) &&
		    !is_hidden_variable(var))
		{
			emit_uniform(var);
			emitted = true;
		}
	});

	if (emitted)
		statement("");
	emitted = false;

	// Output in/out interfaces.
	ir.for_each_typed_id<SPIRVariable>([&](uint32_t, SPIRVariable &var) {
		auto &type = this->get<SPIRType>(var.basetype);

		bool is_hidden = is_hidden_variable(var);

		// Unused output I/O variables might still be required to implement framebuffer fetch.
		if (var.storage == StorageClassOutput && !is_legacy() &&
		    inout_color_attachments.count(get_decoration(var.self, DecorationLocation)) != 0)
		{
			is_hidden = false;
		}

		if (var.storage != StorageClassFunction && type.pointer &&
		    (var.storage == StorageClassInput || var.storage == StorageClassOutput) &&
		    interface_variable_exists_in_entry_point(var.self) && !is_hidden)
		{
			emit_interface_block(var);
			emitted = true;
		}
		else if (is_builtin_variable(var))
		{
			// For gl_InstanceIndex emulation on GLES, the API user needs to
			// supply this uniform.
			if (options.vertex.support_nonzero_base_instance &&
			    ir.meta[var.self].decoration.builtin_type == BuiltInInstanceIndex && !options.vulkan_semantics)
			{
				statement("uniform int SPIRV_Cross_BaseInstance;");
				emitted = true;
			}
		}
	});

	// Global variables.
	for (auto global : global_variables)
	{
		auto &var = get<SPIRVariable>(global);
		if (var.storage != StorageClassOutput)
		{
			if (!variable_is_lut(var))
			{
				add_resource_name(var.self);

				string initializer;
				if (options.force_zero_initialized_variables && var.storage == StorageClassPrivate &&
				    !var.initializer && !var.static_expression && type_can_zero_initialize(get_variable_data_type(var)))
				{
					initializer = join(" = ", to_zero_initialized_expression(get_variable_data_type_id(var)));
				}

				statement(variable_decl(var), initializer, ";");
				emitted = true;
			}
		}
	}

	if (emitted)
		statement("");

	declare_undefined_values();
}

// Returns a string representation of the ID, usable as a function arg.
// Default is to simply return the expression representation fo the arg ID.
// Subclasses may override to modify the return value.
string CompilerGLSL::to_func_call_arg(const SPIRFunction::Parameter &, uint32_t id)
{
	// Make sure that we use the name of the original variable, and not the parameter alias.
	uint32_t name_id = id;
	auto *var = maybe_get<SPIRVariable>(id);
	if (var && var->basevariable)
		name_id = var->basevariable;
	return to_expression(name_id);
}

void CompilerGLSL::handle_invalid_expression(uint32_t id)
{
	// We tried to read an invalidated expression.
	// This means we need another pass at compilation, but next time, force temporary variables so that they cannot be invalidated.
	forced_temporaries.insert(id);
	force_recompile();
}

// Converts the format of the current expression from packed to unpacked,
// by wrapping the expression in a constructor of the appropriate type.
// GLSL does not support packed formats, so simply return the expression.
// Subclasses that do will override.
string CompilerGLSL::unpack_expression_type(string expr_str, const SPIRType &, uint32_t, bool, bool)
{
	return expr_str;
}

// Sometimes we proactively enclosed an expression where it turns out we might have not needed it after all.
void CompilerGLSL::strip_enclosed_expression(string &expr)
{
	if (expr.size() < 2 || expr.front() != '(' || expr.back() != ')')
		return;

	// Have to make sure that our first and last parens actually enclose everything inside it.
	uint32_t paren_count = 0;
	for (auto &c : expr)
	{
		if (c == '(')
			paren_count++;
		else if (c == ')')
		{
			paren_count--;

			// If we hit 0 and this is not the final char, our first and final parens actually don't
			// enclose the expression, and we cannot strip, e.g.: (a + b) * (c + d).
			if (paren_count == 0 && &c != &expr.back())
				return;
		}
	}
	expr.erase(expr.size() - 1, 1);
	expr.erase(begin(expr));
}

string CompilerGLSL::enclose_expression(const string &expr)
{
	bool need_parens = false;

	// If the expression starts with a unary we need to enclose to deal with cases where we have back-to-back
	// unary expressions.
	if (!expr.empty())
	{
		auto c = expr.front();
		if (c == '-' || c == '+' || c == '!' || c == '~' || c == '&' || c == '*')
			need_parens = true;
	}

	if (!need_parens)
	{
		uint32_t paren_count = 0;
		for (auto c : expr)
		{
			if (c == '(' || c == '[')
				paren_count++;
			else if (c == ')' || c == ']')
			{
				assert(paren_count);
				paren_count--;
			}
			else if (c == ' ' && paren_count == 0)
			{
				need_parens = true;
				break;
			}
		}
		assert(paren_count == 0);
	}

	// If this expression contains any spaces which are not enclosed by parentheses,
	// we need to enclose it so we can treat the whole string as an expression.
	// This happens when two expressions have been part of a binary op earlier.
	if (need_parens)
		return join('(', expr, ')');
	else
		return expr;
}

string CompilerGLSL::dereference_expression(const SPIRType &expr_type, const std::string &expr)
{
	// If this expression starts with an address-of operator ('&'), then
	// just return the part after the operator.
	// TODO: Strip parens if unnecessary?
	if (expr.front() == '&')
		return expr.substr(1);
	else if (backend.native_pointers)
		return join('*', expr);
	else if (expr_type.storage == StorageClassPhysicalStorageBufferEXT && expr_type.basetype != SPIRType::Struct &&
	         expr_type.pointer_depth == 1)
	{
		return join(enclose_expression(expr), ".value");
	}
	else
		return expr;
}

string CompilerGLSL::address_of_expression(const std::string &expr)
{
	if (expr.size() > 3 && expr[0] == '(' && expr[1] == '*' && expr.back() == ')')
	{
		// If we have an expression which looks like (*foo), taking the address of it is the same as stripping
		// the first two and last characters. We might have to enclose the expression.
		// This doesn't work for cases like (*foo + 10),
		// but this is an r-value expression which we cannot take the address of anyways.
		return enclose_expression(expr.substr(2, expr.size() - 3));
	}
	else if (expr.front() == '*')
	{
		// If this expression starts with a dereference operator ('*'), then
		// just return the part after the operator.
		return expr.substr(1);
	}
	else
		return join('&', enclose_expression(expr));
}

// Just like to_expression except that we enclose the expression inside parentheses if needed.
string CompilerGLSL::to_enclosed_expression(uint32_t id, bool register_expression_read)
{
	return enclose_expression(to_expression(id, register_expression_read));
}

// Used explicitly when we want to read a row-major expression, but without any transpose shenanigans.
// need_transpose must be forced to false.
string CompilerGLSL::to_unpacked_row_major_matrix_expression(uint32_t id)
{
	return unpack_expression_type(to_expression(id), expression_type(id),
	                              get_extended_decoration(id, SPIRVCrossDecorationPhysicalTypeID),
	                              has_extended_decoration(id, SPIRVCrossDecorationPhysicalTypePacked), true);
}

string CompilerGLSL::to_unpacked_expression(uint32_t id, bool register_expression_read)
{
	// If we need to transpose, it will also take care of unpacking rules.
	auto *e = maybe_get<SPIRExpression>(id);
	bool need_transpose = e && e->need_transpose;
	bool is_remapped = has_extended_decoration(id, SPIRVCrossDecorationPhysicalTypeID);
	bool is_packed = has_extended_decoration(id, SPIRVCrossDecorationPhysicalTypePacked);

	if (!need_transpose && (is_remapped || is_packed))
	{
		return unpack_expression_type(to_expression(id, register_expression_read),
		                              get_pointee_type(expression_type_id(id)),
		                              get_extended_decoration(id, SPIRVCrossDecorationPhysicalTypeID),
		                              has_extended_decoration(id, SPIRVCrossDecorationPhysicalTypePacked), false);
	}
	else
		return to_expression(id, register_expression_read);
}

string CompilerGLSL::to_enclosed_unpacked_expression(uint32_t id, bool register_expression_read)
{
	// If we need to transpose, it will also take care of unpacking rules.
	auto *e = maybe_get<SPIRExpression>(id);
	bool need_transpose = e && e->need_transpose;
	bool is_remapped = has_extended_decoration(id, SPIRVCrossDecorationPhysicalTypeID);
	bool is_packed = has_extended_decoration(id, SPIRVCrossDecorationPhysicalTypePacked);
	if (!need_transpose && (is_remapped || is_packed))
	{
		return unpack_expression_type(to_expression(id, register_expression_read), expression_type(id),
		                              get_extended_decoration(id, SPIRVCrossDecorationPhysicalTypeID),
		                              has_extended_decoration(id, SPIRVCrossDecorationPhysicalTypePacked), false);
	}
	else
		return to_enclosed_expression(id, register_expression_read);
}

string CompilerGLSL::to_dereferenced_expression(uint32_t id, bool register_expression_read)
{
	auto &type = expression_type(id);
	if (type.pointer && should_dereference(id))
		return dereference_expression(type, to_enclosed_expression(id, register_expression_read));
	else
		return to_expression(id, register_expression_read);
}

string CompilerGLSL::to_pointer_expression(uint32_t id, bool register_expression_read)
{
	auto &type = expression_type(id);
	if (type.pointer && expression_is_lvalue(id) && !should_dereference(id))
		return address_of_expression(to_enclosed_expression(id, register_expression_read));
	else
		return to_unpacked_expression(id, register_expression_read);
}

string CompilerGLSL::to_enclosed_pointer_expression(uint32_t id, bool register_expression_read)
{
	auto &type = expression_type(id);
	if (type.pointer && expression_is_lvalue(id) && !should_dereference(id))
		return address_of_expression(to_enclosed_expression(id, register_expression_read));
	else
		return to_enclosed_unpacked_expression(id, register_expression_read);
}

string CompilerGLSL::to_extract_component_expression(uint32_t id, uint32_t index)
{
	auto expr = to_enclosed_expression(id);
	if (has_extended_decoration(id, SPIRVCrossDecorationPhysicalTypePacked))
		return join(expr, "[", index, "]");
	else
		return join(expr, ".", index_to_swizzle(index));
}

string CompilerGLSL::to_rerolled_array_expression(const string &base_expr, const SPIRType &type)
{
	uint32_t size = to_array_size_literal(type);
	auto &parent = get<SPIRType>(type.parent_type);
	string expr = "{ ";

	for (uint32_t i = 0; i < size; i++)
	{
		auto subexpr = join(base_expr, "[", convert_to_string(i), "]");
		if (parent.array.empty())
			expr += subexpr;
		else
			expr += to_rerolled_array_expression(subexpr, parent);

		if (i + 1 < size)
			expr += ", ";
	}

	expr += " }";
	return expr;
}

string CompilerGLSL::to_composite_constructor_expression(uint32_t id)
{
	auto &type = expression_type(id);
	if (!backend.array_is_value_type && !type.array.empty())
	{
		// For this case, we need to "re-roll" an array initializer from a temporary.
		// We cannot simply pass the array directly, since it decays to a pointer and it cannot
		// participate in a struct initializer. E.g.
		// float arr[2] = { 1.0, 2.0 };
		// Foo foo = { arr }; must be transformed to
		// Foo foo = { { arr[0], arr[1] } };
		// The array sizes cannot be deduced from specialization constants since we cannot use any loops.

		// We're only triggering one read of the array expression, but this is fine since arrays have to be declared
		// as temporaries anyways.
		return to_rerolled_array_expression(to_enclosed_expression(id), type);
	}
	else
		return to_unpacked_expression(id);
}

string CompilerGLSL::to_expression(uint32_t id, bool register_expression_read)
{
	auto itr = invalid_expressions.find(id);
	if (itr != end(invalid_expressions))
		handle_invalid_expression(id);

	if (ir.ids[id].get_type() == TypeExpression)
	{
		// We might have a more complex chain of dependencies.
		// A possible scenario is that we
		//
		// %1 = OpLoad
		// %2 = OpDoSomething %1 %1. here %2 will have a dependency on %1.
		// %3 = OpDoSomethingAgain %2 %2. Here %3 will lose the link to %1 since we don't propagate the dependencies like that.
		// OpStore %1 %foo // Here we can invalidate %1, and hence all expressions which depend on %1. Only %2 will know since it's part of invalid_expressions.
		// %4 = OpDoSomethingAnotherTime %3 %3 // If we forward all expressions we will see %1 expression after store, not before.
		//
		// However, we can propagate up a list of depended expressions when we used %2, so we can check if %2 is invalid when reading %3 after the store,
		// and see that we should not forward reads of the original variable.
		auto &expr = get<SPIRExpression>(id);
		for (uint32_t dep : expr.expression_dependencies)
			if (invalid_expressions.find(dep) != end(invalid_expressions))
				handle_invalid_expression(dep);
	}

	if (register_expression_read)
		track_expression_read(id);

	switch (ir.ids[id].get_type())
	{
	case TypeExpression:
	{
		auto &e = get<SPIRExpression>(id);
		if (e.base_expression)
			return to_enclosed_expression(e.base_expression) + e.expression;
		else if (e.need_transpose)
		{
			// This should not be reached for access chains, since we always deal explicitly with transpose state
			// when consuming an access chain expression.
			uint32_t physical_type_id = get_extended_decoration(id, SPIRVCrossDecorationPhysicalTypeID);
			bool is_packed = has_extended_decoration(id, SPIRVCrossDecorationPhysicalTypePacked);
			return convert_row_major_matrix(e.expression, get<SPIRType>(e.expression_type), physical_type_id,
			                                is_packed);
		}
		else
		{
			if (is_forcing_recompilation())
			{
				// During first compilation phase, certain expression patterns can trigger exponential growth of memory.
				// Avoid this by returning dummy expressions during this phase.
				// Do not use empty expressions here, because those are sentinels for other cases.
				return "_";
			}
			else
				return e.expression;
		}
	}

	case TypeConstant:
	{
		auto &c = get<SPIRConstant>(id);
		auto &type = get<SPIRType>(c.constant_type);

		// WorkGroupSize may be a constant.
		auto &dec = ir.meta[c.self].decoration;
		if (dec.builtin)
			return builtin_to_glsl(dec.builtin_type, StorageClassGeneric);
		else if (c.specialization)
			return to_name(id);
		else if (c.is_used_as_lut)
			return to_name(id);
		else if (type.basetype == SPIRType::Struct && !backend.can_declare_struct_inline)
			return to_name(id);
		else if (!type.array.empty() && !backend.can_declare_arrays_inline)
			return to_name(id);
		else
			return constant_expression(c);
	}

	case TypeConstantOp:
		return to_name(id);

	case TypeVariable:
	{
		auto &var = get<SPIRVariable>(id);
		// If we try to use a loop variable before the loop header, we have to redirect it to the static expression,
		// the variable has not been declared yet.
		if (var.statically_assigned || (var.loop_variable && !var.loop_variable_enable))
			return to_expression(var.static_expression);
		else if (var.deferred_declaration)
		{
			var.deferred_declaration = false;
			return variable_decl(var);
		}
		else if (flattened_structs.count(id))
		{
			return load_flattened_struct(var);
		}
		else
		{
			auto &dec = ir.meta[var.self].decoration;
			if (dec.builtin)
				return builtin_to_glsl(dec.builtin_type, var.storage);
			else
				return to_name(id);
		}
	}

	case TypeCombinedImageSampler:
		// This type should never be taken the expression of directly.
		// The intention is that texture sampling functions will extract the image and samplers
		// separately and take their expressions as needed.
		// GLSL does not use this type because OpSampledImage immediately creates a combined image sampler
		// expression ala sampler2D(texture, sampler).
		SPIRV_CROSS_THROW("Combined image samplers have no default expression representation.");

	case TypeAccessChain:
		// We cannot express this type. They only have meaning in other OpAccessChains, OpStore or OpLoad.
		SPIRV_CROSS_THROW("Access chains have no default expression representation.");

	default:
		return to_name(id);
	}
}

string CompilerGLSL::constant_op_expression(const SPIRConstantOp &cop)
{
	auto &type = get<SPIRType>(cop.basetype);
	bool binary = false;
	bool unary = false;
	string op;

	if (is_legacy() && is_unsigned_opcode(cop.opcode))
		SPIRV_CROSS_THROW("Unsigned integers are not supported on legacy targets.");

	// TODO: Find a clean way to reuse emit_instruction.
	switch (cop.opcode)
	{
	case OpSConvert:
	case OpUConvert:
	case OpFConvert:
		op = type_to_glsl_constructor(type);
		break;

#define GLSL_BOP(opname, x) \
	case Op##opname:        \
		binary = true;      \
		op = x;             \
		break

#define GLSL_UOP(opname, x) \
	case Op##opname:        \
		unary = true;       \
		op = x;             \
		break

		GLSL_UOP(SNegate, "-");
		GLSL_UOP(Not, "~");
		GLSL_BOP(IAdd, "+");
		GLSL_BOP(ISub, "-");
		GLSL_BOP(IMul, "*");
		GLSL_BOP(SDiv, "/");
		GLSL_BOP(UDiv, "/");
		GLSL_BOP(UMod, "%");
		GLSL_BOP(SMod, "%");
		GLSL_BOP(ShiftRightLogical, ">>");
		GLSL_BOP(ShiftRightArithmetic, ">>");
		GLSL_BOP(ShiftLeftLogical, "<<");
		GLSL_BOP(BitwiseOr, "|");
		GLSL_BOP(BitwiseXor, "^");
		GLSL_BOP(BitwiseAnd, "&");
		GLSL_BOP(LogicalOr, "||");
		GLSL_BOP(LogicalAnd, "&&");
		GLSL_UOP(LogicalNot, "!");
		GLSL_BOP(LogicalEqual, "==");
		GLSL_BOP(LogicalNotEqual, "!=");
		GLSL_BOP(IEqual, "==");
		GLSL_BOP(INotEqual, "!=");
		GLSL_BOP(ULessThan, "<");
		GLSL_BOP(SLessThan, "<");
		GLSL_BOP(ULessThanEqual, "<=");
		GLSL_BOP(SLessThanEqual, "<=");
		GLSL_BOP(UGreaterThan, ">");
		GLSL_BOP(SGreaterThan, ">");
		GLSL_BOP(UGreaterThanEqual, ">=");
		GLSL_BOP(SGreaterThanEqual, ">=");

	case OpSelect:
	{
		if (cop.arguments.size() < 3)
			SPIRV_CROSS_THROW("Not enough arguments to OpSpecConstantOp.");

		// This one is pretty annoying. It's triggered from
		// uint(bool), int(bool) from spec constants.
		// In order to preserve its compile-time constness in Vulkan GLSL,
		// we need to reduce the OpSelect expression back to this simplified model.
		// If we cannot, fail.
		if (to_trivial_mix_op(type, op, cop.arguments[2], cop.arguments[1], cop.arguments[0]))
		{
			// Implement as a simple cast down below.
		}
		else
		{
			// Implement a ternary and pray the compiler understands it :)
			return to_ternary_expression(type, cop.arguments[0], cop.arguments[1], cop.arguments[2]);
		}
		break;
	}

	case OpVectorShuffle:
	{
		string expr = type_to_glsl_constructor(type);
		expr += "(";

		uint32_t left_components = expression_type(cop.arguments[0]).vecsize;
		string left_arg = to_enclosed_expression(cop.arguments[0]);
		string right_arg = to_enclosed_expression(cop.arguments[1]);

		for (uint32_t i = 2; i < uint32_t(cop.arguments.size()); i++)
		{
			uint32_t index = cop.arguments[i];
			if (index >= left_components)
				expr += right_arg + "." + "xyzw"[index - left_components];
			else
				expr += left_arg + "." + "xyzw"[index];

			if (i + 1 < uint32_t(cop.arguments.size()))
				expr += ", ";
		}

		expr += ")";
		return expr;
	}

	case OpCompositeExtract:
	{
		auto expr = access_chain_internal(cop.arguments[0], &cop.arguments[1], uint32_t(cop.arguments.size() - 1),
		                                  ACCESS_CHAIN_INDEX_IS_LITERAL_BIT, nullptr);
		return expr;
	}

	case OpCompositeInsert:
		SPIRV_CROSS_THROW("OpCompositeInsert spec constant op is not supported.");

	default:
		// Some opcodes are unimplemented here, these are currently not possible to test from glslang.
		SPIRV_CROSS_THROW("Unimplemented spec constant op.");
	}

	uint32_t bit_width = 0;
	if (unary || binary || cop.opcode == OpSConvert || cop.opcode == OpUConvert)
		bit_width = expression_type(cop.arguments[0]).width;

	SPIRType::BaseType input_type;
	bool skip_cast_if_equal_type = opcode_is_sign_invariant(cop.opcode);

	switch (cop.opcode)
	{
	case OpIEqual:
	case OpINotEqual:
		input_type = to_signed_basetype(bit_width);
		break;

	case OpSLessThan:
	case OpSLessThanEqual:
	case OpSGreaterThan:
	case OpSGreaterThanEqual:
	case OpSMod:
	case OpSDiv:
	case OpShiftRightArithmetic:
	case OpSConvert:
	case OpSNegate:
		input_type = to_signed_basetype(bit_width);
		break;

	case OpULessThan:
	case OpULessThanEqual:
	case OpUGreaterThan:
	case OpUGreaterThanEqual:
	case OpUMod:
	case OpUDiv:
	case OpShiftRightLogical:
	case OpUConvert:
		input_type = to_unsigned_basetype(bit_width);
		break;

	default:
		input_type = type.basetype;
		break;
	}

#undef GLSL_BOP
#undef GLSL_UOP
	if (binary)
	{
		if (cop.arguments.size() < 2)
			SPIRV_CROSS_THROW("Not enough arguments to OpSpecConstantOp.");

		string cast_op0;
		string cast_op1;
		auto expected_type = binary_op_bitcast_helper(cast_op0, cast_op1, input_type, cop.arguments[0],
		                                              cop.arguments[1], skip_cast_if_equal_type);

		if (type.basetype != input_type && type.basetype != SPIRType::Boolean)
		{
			expected_type.basetype = input_type;
			auto expr = bitcast_glsl_op(type, expected_type);
			expr += '(';
			expr += join(cast_op0, " ", op, " ", cast_op1);
			expr += ')';
			return expr;
		}
		else
			return join("(", cast_op0, " ", op, " ", cast_op1, ")");
	}
	else if (unary)
	{
		if (cop.arguments.size() < 1)
			SPIRV_CROSS_THROW("Not enough arguments to OpSpecConstantOp.");

		// Auto-bitcast to result type as needed.
		// Works around various casting scenarios in glslang as there is no OpBitcast for specialization constants.
		return join("(", op, bitcast_glsl(type, cop.arguments[0]), ")");
	}
	else if (cop.opcode == OpSConvert || cop.opcode == OpUConvert)
	{
		if (cop.arguments.size() < 1)
			SPIRV_CROSS_THROW("Not enough arguments to OpSpecConstantOp.");

		auto &arg_type = expression_type(cop.arguments[0]);
		if (arg_type.width < type.width && input_type != arg_type.basetype)
		{
			auto expected = arg_type;
			expected.basetype = input_type;
			return join(op, "(", bitcast_glsl(expected, cop.arguments[0]), ")");
		}
		else
			return join(op, "(", to_expression(cop.arguments[0]), ")");
	}
	else
	{
		if (cop.arguments.size() < 1)
			SPIRV_CROSS_THROW("Not enough arguments to OpSpecConstantOp.");
		return join(op, "(", to_expression(cop.arguments[0]), ")");
	}
}

string CompilerGLSL::constant_expression(const SPIRConstant &c)
{
	auto &type = get<SPIRType>(c.constant_type);

	if (type.pointer)
	{
		return backend.null_pointer_literal;
	}
	else if (!c.subconstants.empty())
	{
		// Handles Arrays and structures.
		string res;

		// Allow Metal to use the array<T> template to make arrays a value type
		bool needs_trailing_tracket = false;
		if (backend.use_initializer_list && backend.use_typed_initializer_list && type.basetype == SPIRType::Struct &&
		    type.array.empty())
		{
			res = type_to_glsl_constructor(type) + "{ ";
		}
		else if (backend.use_initializer_list && backend.use_typed_initializer_list && backend.array_is_value_type &&
		         !type.array.empty())
		{
			res = type_to_glsl_constructor(type) + "({ ";
			needs_trailing_tracket = true;
		}
		else if (backend.use_initializer_list)
		{
			res = "{ ";
		}
		else
		{
			res = type_to_glsl_constructor(type) + "(";
		}

		for (auto &elem : c.subconstants)
		{
			auto &subc = get<SPIRConstant>(elem);
			if (subc.specialization)
				res += to_name(elem);
			else
				res += constant_expression(subc);

			if (&elem != &c.subconstants.back())
				res += ", ";
		}

		res += backend.use_initializer_list ? " }" : ")";
		if (needs_trailing_tracket)
			res += ")";

		return res;
	}
	else if (type.basetype == SPIRType::Struct && type.member_types.size() == 0)
	{
		// Metal tessellation likes empty structs which are then constant expressions.
		if (backend.supports_empty_struct)
			return "{ }";
		else if (backend.use_typed_initializer_list)
			return join(type_to_glsl(get<SPIRType>(c.constant_type)), "{ 0 }");
		else if (backend.use_initializer_list)
			return "{ 0 }";
		else
			return join(type_to_glsl(get<SPIRType>(c.constant_type)), "(0)");
	}
	else if (c.columns() == 1)
	{
		return constant_expression_vector(c, 0);
	}
	else
	{
		string res = type_to_glsl(get<SPIRType>(c.constant_type)) + "(";
		for (uint32_t col = 0; col < c.columns(); col++)
		{
			if (c.specialization_constant_id(col) != 0)
				res += to_name(c.specialization_constant_id(col));
			else
				res += constant_expression_vector(c, col);

			if (col + 1 < c.columns())
				res += ", ";
		}
		res += ")";
		return res;
	}
}

#ifdef _MSC_VER
// sprintf warning.
// We cannot rely on snprintf existing because, ..., MSVC.
#pragma warning(push)
#pragma warning(disable : 4996)
#endif

string CompilerGLSL::convert_half_to_string(const SPIRConstant &c, uint32_t col, uint32_t row)
{
	string res;
	float float_value = c.scalar_f16(col, row);

	// There is no literal "hf" in GL_NV_gpu_shader5, so to avoid lots
	// of complicated workarounds, just value-cast to the half type always.
	if (std::isnan(float_value) || std::isinf(float_value))
	{
		SPIRType type;
		type.basetype = SPIRType::Half;
		type.vecsize = 1;
		type.columns = 1;

		if (float_value == numeric_limits<float>::infinity())
			res = join(type_to_glsl(type), "(1.0 / 0.0)");
		else if (float_value == -numeric_limits<float>::infinity())
			res = join(type_to_glsl(type), "(-1.0 / 0.0)");
		else if (std::isnan(float_value))
			res = join(type_to_glsl(type), "(0.0 / 0.0)");
		else
			SPIRV_CROSS_THROW("Cannot represent non-finite floating point constant.");
	}
	else
	{
		SPIRType type;
		type.basetype = SPIRType::Half;
		type.vecsize = 1;
		type.columns = 1;
		res = join(type_to_glsl(type), "(", convert_to_string(float_value, current_locale_radix_character), ")");
	}

	return res;
}

string CompilerGLSL::convert_float_to_string(const SPIRConstant &c, uint32_t col, uint32_t row)
{
	string res;
	float float_value = c.scalar_f32(col, row);

	if (std::isnan(float_value) || std::isinf(float_value))
	{
		// Use special representation.
		if (!is_legacy())
		{
			SPIRType out_type;
			SPIRType in_type;
			out_type.basetype = SPIRType::Float;
			in_type.basetype = SPIRType::UInt;
			out_type.vecsize = 1;
			in_type.vecsize = 1;
			out_type.width = 32;
			in_type.width = 32;

			char print_buffer[32];
			sprintf(print_buffer, "0x%xu", c.scalar(col, row));
			res = join(bitcast_glsl_op(out_type, in_type), "(", print_buffer, ")");
		}
		else
		{
			if (float_value == numeric_limits<float>::infinity())
			{
				if (backend.float_literal_suffix)
					res = "(1.0f / 0.0f)";
				else
					res = "(1.0 / 0.0)";
			}
			else if (float_value == -numeric_limits<float>::infinity())
			{
				if (backend.float_literal_suffix)
					res = "(-1.0f / 0.0f)";
				else
					res = "(-1.0 / 0.0)";
			}
			else if (std::isnan(float_value))
			{
				if (backend.float_literal_suffix)
					res = "(0.0f / 0.0f)";
				else
					res = "(0.0 / 0.0)";
			}
			else
				SPIRV_CROSS_THROW("Cannot represent non-finite floating point constant.");
		}
	}
	else
	{
		res = convert_to_string(float_value, current_locale_radix_character);
		if (backend.float_literal_suffix)
			res += "f";
	}

	return res;
}

std::string CompilerGLSL::convert_double_to_string(const SPIRConstant &c, uint32_t col, uint32_t row)
{
	string res;
	double double_value = c.scalar_f64(col, row);

	if (std::isnan(double_value) || std::isinf(double_value))
	{
		// Use special representation.
		if (!is_legacy())
		{
			SPIRType out_type;
			SPIRType in_type;
			out_type.basetype = SPIRType::Double;
			in_type.basetype = SPIRType::UInt64;
			out_type.vecsize = 1;
			in_type.vecsize = 1;
			out_type.width = 64;
			in_type.width = 64;

			uint64_t u64_value = c.scalar_u64(col, row);

			if (options.es)
				SPIRV_CROSS_THROW("64-bit integers/float not supported in ES profile.");
			require_extension_internal("GL_ARB_gpu_shader_int64");

			char print_buffer[64];
			sprintf(print_buffer, "0x%llx%s", static_cast<unsigned long long>(u64_value),
			        backend.long_long_literal_suffix ? "ull" : "ul");
			res = join(bitcast_glsl_op(out_type, in_type), "(", print_buffer, ")");
		}
		else
		{
			if (options.es)
				SPIRV_CROSS_THROW("FP64 not supported in ES profile.");
			if (options.version < 400)
				require_extension_internal("GL_ARB_gpu_shader_fp64");

			if (double_value == numeric_limits<double>::infinity())
			{
				if (backend.double_literal_suffix)
					res = "(1.0lf / 0.0lf)";
				else
					res = "(1.0 / 0.0)";
			}
			else if (double_value == -numeric_limits<double>::infinity())
			{
				if (backend.double_literal_suffix)
					res = "(-1.0lf / 0.0lf)";
				else
					res = "(-1.0 / 0.0)";
			}
			else if (std::isnan(double_value))
			{
				if (backend.double_literal_suffix)
					res = "(0.0lf / 0.0lf)";
				else
					res = "(0.0 / 0.0)";
			}
			else
				SPIRV_CROSS_THROW("Cannot represent non-finite floating point constant.");
		}
	}
	else
	{
		res = convert_to_string(double_value, current_locale_radix_character);
		if (backend.double_literal_suffix)
			res += "lf";
	}

	return res;
}

#ifdef _MSC_VER
#pragma warning(pop)
#endif

string CompilerGLSL::constant_expression_vector(const SPIRConstant &c, uint32_t vector)
{
	auto type = get<SPIRType>(c.constant_type);
	type.columns = 1;

	auto scalar_type = type;
	scalar_type.vecsize = 1;

	string res;
	bool splat = backend.use_constructor_splatting && c.vector_size() > 1;
	bool swizzle_splat = backend.can_swizzle_scalar && c.vector_size() > 1;

	if (!type_is_floating_point(type))
	{
		// Cannot swizzle literal integers as a special case.
		swizzle_splat = false;
	}

	if (splat || swizzle_splat)
	{
		// Cannot use constant splatting if we have specialization constants somewhere in the vector.
		for (uint32_t i = 0; i < c.vector_size(); i++)
		{
			if (c.specialization_constant_id(vector, i) != 0)
			{
				splat = false;
				swizzle_splat = false;
				break;
			}
		}
	}

	if (splat || swizzle_splat)
	{
		if (type.width == 64)
		{
			uint64_t ident = c.scalar_u64(vector, 0);
			for (uint32_t i = 1; i < c.vector_size(); i++)
			{
				if (ident != c.scalar_u64(vector, i))
				{
					splat = false;
					swizzle_splat = false;
					break;
				}
			}
		}
		else
		{
			uint32_t ident = c.scalar(vector, 0);
			for (uint32_t i = 1; i < c.vector_size(); i++)
			{
				if (ident != c.scalar(vector, i))
				{
					splat = false;
					swizzle_splat = false;
				}
			}
		}
	}

	if (c.vector_size() > 1 && !swizzle_splat)
		res += type_to_glsl(type) + "(";

	switch (type.basetype)
	{
	case SPIRType::Half:
		if (splat || swizzle_splat)
		{
			res += convert_half_to_string(c, vector, 0);
			if (swizzle_splat)
				res = remap_swizzle(get<SPIRType>(c.constant_type), 1, res);
		}
		else
		{
			for (uint32_t i = 0; i < c.vector_size(); i++)
			{
				if (c.vector_size() > 1 && c.specialization_constant_id(vector, i) != 0)
					res += to_name(c.specialization_constant_id(vector, i));
				else
					res += convert_half_to_string(c, vector, i);

				if (i + 1 < c.vector_size())
					res += ", ";
			}
		}
		break;

	case SPIRType::Float:
		if (splat || swizzle_splat)
		{
			res += convert_float_to_string(c, vector, 0);
			if (swizzle_splat)
				res = remap_swizzle(get<SPIRType>(c.constant_type), 1, res);
		}
		else
		{
			for (uint32_t i = 0; i < c.vector_size(); i++)
			{
				if (c.vector_size() > 1 && c.specialization_constant_id(vector, i) != 0)
					res += to_name(c.specialization_constant_id(vector, i));
				else
					res += convert_float_to_string(c, vector, i);

				if (i + 1 < c.vector_size())
					res += ", ";
			}
		}
		break;

	case SPIRType::Double:
		if (splat || swizzle_splat)
		{
			res += convert_double_to_string(c, vector, 0);
			if (swizzle_splat)
				res = remap_swizzle(get<SPIRType>(c.constant_type), 1, res);
		}
		else
		{
			for (uint32_t i = 0; i < c.vector_size(); i++)
			{
				if (c.vector_size() > 1 && c.specialization_constant_id(vector, i) != 0)
					res += to_name(c.specialization_constant_id(vector, i));
				else
					res += convert_double_to_string(c, vector, i);

				if (i + 1 < c.vector_size())
					res += ", ";
			}
		}
		break;

	case SPIRType::Int64:
		if (splat)
		{
			res += convert_to_string(c.scalar_i64(vector, 0));
			if (backend.long_long_literal_suffix)
				res += "ll";
			else
				res += "l";
		}
		else
		{
			for (uint32_t i = 0; i < c.vector_size(); i++)
			{
				if (c.vector_size() > 1 && c.specialization_constant_id(vector, i) != 0)
					res += to_name(c.specialization_constant_id(vector, i));
				else
				{
					res += convert_to_string(c.scalar_i64(vector, i));
					if (backend.long_long_literal_suffix)
						res += "ll";
					else
						res += "l";
				}

				if (i + 1 < c.vector_size())
					res += ", ";
			}
		}
		break;

	case SPIRType::UInt64:
		if (splat)
		{
			res += convert_to_string(c.scalar_u64(vector, 0));
			if (backend.long_long_literal_suffix)
				res += "ull";
			else
				res += "ul";
		}
		else
		{
			for (uint32_t i = 0; i < c.vector_size(); i++)
			{
				if (c.vector_size() > 1 && c.specialization_constant_id(vector, i) != 0)
					res += to_name(c.specialization_constant_id(vector, i));
				else
				{
					res += convert_to_string(c.scalar_u64(vector, i));
					if (backend.long_long_literal_suffix)
						res += "ull";
					else
						res += "ul";
				}

				if (i + 1 < c.vector_size())
					res += ", ";
			}
		}
		break;

	case SPIRType::UInt:
		if (splat)
		{
			res += convert_to_string(c.scalar(vector, 0));
			if (is_legacy())
			{
				// Fake unsigned constant literals with signed ones if possible.
				// Things like array sizes, etc, tend to be unsigned even though they could just as easily be signed.
				if (c.scalar_i32(vector, 0) < 0)
					SPIRV_CROSS_THROW("Tried to convert uint literal into int, but this made the literal negative.");
			}
			else if (backend.uint32_t_literal_suffix)
				res += "u";
		}
		else
		{
			for (uint32_t i = 0; i < c.vector_size(); i++)
			{
				if (c.vector_size() > 1 && c.specialization_constant_id(vector, i) != 0)
					res += to_name(c.specialization_constant_id(vector, i));
				else
				{
					res += convert_to_string(c.scalar(vector, i));
					if (is_legacy())
					{
						// Fake unsigned constant literals with signed ones if possible.
						// Things like array sizes, etc, tend to be unsigned even though they could just as easily be signed.
						if (c.scalar_i32(vector, i) < 0)
							SPIRV_CROSS_THROW(
							    "Tried to convert uint literal into int, but this made the literal negative.");
					}
					else if (backend.uint32_t_literal_suffix)
						res += "u";
				}

				if (i + 1 < c.vector_size())
					res += ", ";
			}
		}
		break;

	case SPIRType::Int:
		if (splat)
			res += convert_to_string(c.scalar_i32(vector, 0));
		else
		{
			for (uint32_t i = 0; i < c.vector_size(); i++)
			{
				if (c.vector_size() > 1 && c.specialization_constant_id(vector, i) != 0)
					res += to_name(c.specialization_constant_id(vector, i));
				else
					res += convert_to_string(c.scalar_i32(vector, i));
				if (i + 1 < c.vector_size())
					res += ", ";
			}
		}
		break;

	case SPIRType::UShort:
		if (splat)
		{
			res += convert_to_string(c.scalar(vector, 0));
		}
		else
		{
			for (uint32_t i = 0; i < c.vector_size(); i++)
			{
				if (c.vector_size() > 1 && c.specialization_constant_id(vector, i) != 0)
					res += to_name(c.specialization_constant_id(vector, i));
				else
				{
					if (*backend.uint16_t_literal_suffix)
					{
						res += convert_to_string(c.scalar_u16(vector, i));
						res += backend.uint16_t_literal_suffix;
					}
					else
					{
						// If backend doesn't have a literal suffix, we need to value cast.
						res += type_to_glsl(scalar_type);
						res += "(";
						res += convert_to_string(c.scalar_u16(vector, i));
						res += ")";
					}
				}

				if (i + 1 < c.vector_size())
					res += ", ";
			}
		}
		break;

	case SPIRType::Short:
		if (splat)
		{
			res += convert_to_string(c.scalar_i16(vector, 0));
		}
		else
		{
			for (uint32_t i = 0; i < c.vector_size(); i++)
			{
				if (c.vector_size() > 1 && c.specialization_constant_id(vector, i) != 0)
					res += to_name(c.specialization_constant_id(vector, i));
				else
				{
					if (*backend.int16_t_literal_suffix)
					{
						res += convert_to_string(c.scalar_i16(vector, i));
						res += backend.int16_t_literal_suffix;
					}
					else
					{
						// If backend doesn't have a literal suffix, we need to value cast.
						res += type_to_glsl(scalar_type);
						res += "(";
						res += convert_to_string(c.scalar_i16(vector, i));
						res += ")";
					}
				}

				if (i + 1 < c.vector_size())
					res += ", ";
			}
		}
		break;

	case SPIRType::UByte:
		if (splat)
		{
			res += convert_to_string(c.scalar_u8(vector, 0));
		}
		else
		{
			for (uint32_t i = 0; i < c.vector_size(); i++)
			{
				if (c.vector_size() > 1 && c.specialization_constant_id(vector, i) != 0)
					res += to_name(c.specialization_constant_id(vector, i));
				else
				{
					res += type_to_glsl(scalar_type);
					res += "(";
					res += convert_to_string(c.scalar_u8(vector, i));
					res += ")";
				}

				if (i + 1 < c.vector_size())
					res += ", ";
			}
		}
		break;

	case SPIRType::SByte:
		if (splat)
		{
			res += convert_to_string(c.scalar_i8(vector, 0));
		}
		else
		{
			for (uint32_t i = 0; i < c.vector_size(); i++)
			{
				if (c.vector_size() > 1 && c.specialization_constant_id(vector, i) != 0)
					res += to_name(c.specialization_constant_id(vector, i));
				else
				{
					res += type_to_glsl(scalar_type);
					res += "(";
					res += convert_to_string(c.scalar_i8(vector, i));
					res += ")";
				}

				if (i + 1 < c.vector_size())
					res += ", ";
			}
		}
		break;

	case SPIRType::Boolean:
		if (splat)
			res += c.scalar(vector, 0) ? "true" : "false";
		else
		{
			for (uint32_t i = 0; i < c.vector_size(); i++)
			{
				if (c.vector_size() > 1 && c.specialization_constant_id(vector, i) != 0)
					res += to_name(c.specialization_constant_id(vector, i));
				else
					res += c.scalar(vector, i) ? "true" : "false";

				if (i + 1 < c.vector_size())
					res += ", ";
			}
		}
		break;

	default:
		SPIRV_CROSS_THROW("Invalid constant expression basetype.");
	}

	if (c.vector_size() > 1 && !swizzle_splat)
		res += ")";

	return res;
}

SPIRExpression &CompilerGLSL::emit_uninitialized_temporary_expression(uint32_t type, uint32_t id)
{
	forced_temporaries.insert(id);
	emit_uninitialized_temporary(type, id);
	return set<SPIRExpression>(id, to_name(id), type, true);
}

void CompilerGLSL::emit_uninitialized_temporary(uint32_t result_type, uint32_t result_id)
{
	// If we're declaring temporaries inside continue blocks,
	// we must declare the temporary in the loop header so that the continue block can avoid declaring new variables.
	if (current_continue_block && !hoisted_temporaries.count(result_id))
	{
		auto &header = get<SPIRBlock>(current_continue_block->loop_dominator);
		if (find_if(begin(header.declare_temporary), end(header.declare_temporary),
		            [result_type, result_id](const pair<uint32_t, uint32_t> &tmp) {
			            return tmp.first == result_type && tmp.second == result_id;
		            }) == end(header.declare_temporary))
		{
			header.declare_temporary.emplace_back(result_type, result_id);
			hoisted_temporaries.insert(result_id);
			force_recompile();
		}
	}
	else if (hoisted_temporaries.count(result_id) == 0)
	{
		auto &type = get<SPIRType>(result_type);
		auto &flags = ir.meta[result_id].decoration.decoration_flags;

		// The result_id has not been made into an expression yet, so use flags interface.
		add_local_variable_name(result_id);

		string initializer;
		if (options.force_zero_initialized_variables && type_can_zero_initialize(type))
			initializer = join(" = ", to_zero_initialized_expression(result_type));

		statement(flags_to_qualifiers_glsl(type, flags), variable_decl(type, to_name(result_id)), initializer, ";");
	}
}

string CompilerGLSL::declare_temporary(uint32_t result_type, uint32_t result_id)
{
	auto &type = get<SPIRType>(result_type);
	auto &flags = ir.meta[result_id].decoration.decoration_flags;

	// If we're declaring temporaries inside continue blocks,
	// we must declare the temporary in the loop header so that the continue block can avoid declaring new variables.
	if (current_continue_block && !hoisted_temporaries.count(result_id))
	{
		auto &header = get<SPIRBlock>(current_continue_block->loop_dominator);
		if (find_if(begin(header.declare_temporary), end(header.declare_temporary),
		            [result_type, result_id](const pair<uint32_t, uint32_t> &tmp) {
			            return tmp.first == result_type && tmp.second == result_id;
		            }) == end(header.declare_temporary))
		{
			header.declare_temporary.emplace_back(result_type, result_id);
			hoisted_temporaries.insert(result_id);
			force_recompile();
		}

		return join(to_name(result_id), " = ");
	}
	else if (hoisted_temporaries.count(result_id))
	{
		// The temporary has already been declared earlier, so just "declare" the temporary by writing to it.
		return join(to_name(result_id), " = ");
	}
	else
	{
		// The result_id has not been made into an expression yet, so use flags interface.
		add_local_variable_name(result_id);
		return join(flags_to_qualifiers_glsl(type, flags), variable_decl(type, to_name(result_id)), " = ");
	}
}

bool CompilerGLSL::expression_is_forwarded(uint32_t id) const
{
	return forwarded_temporaries.count(id) != 0;
}

bool CompilerGLSL::expression_suppresses_usage_tracking(uint32_t id) const
{
	return suppressed_usage_tracking.count(id) != 0;
}

SPIRExpression &CompilerGLSL::emit_op(uint32_t result_type, uint32_t result_id, const string &rhs, bool forwarding,
                                      bool suppress_usage_tracking)
{
	if (forwarding && (forced_temporaries.find(result_id) == end(forced_temporaries)))
	{
		// Just forward it without temporary.
		// If the forward is trivial, we do not force flushing to temporary for this expression.
		forwarded_temporaries.insert(result_id);
		if (suppress_usage_tracking)
			suppressed_usage_tracking.insert(result_id);

		return set<SPIRExpression>(result_id, rhs, result_type, true);
	}
	else
	{
		// If expression isn't immutable, bind it to a temporary and make the new temporary immutable (they always are).
		statement(declare_temporary(result_type, result_id), rhs, ";");
		return set<SPIRExpression>(result_id, to_name(result_id), result_type, true);
	}
}

void CompilerGLSL::emit_unary_op(uint32_t result_type, uint32_t result_id, uint32_t op0, const char *op)
{
	bool forward = should_forward(op0);
	emit_op(result_type, result_id, join(op, to_enclosed_unpacked_expression(op0)), forward);
	inherit_expression_dependencies(result_id, op0);
}

void CompilerGLSL::emit_binary_op(uint32_t result_type, uint32_t result_id, uint32_t op0, uint32_t op1, const char *op)
{
	bool forward = should_forward(op0) && should_forward(op1);
	emit_op(result_type, result_id,
	        join(to_enclosed_unpacked_expression(op0), " ", op, " ", to_enclosed_unpacked_expression(op1)), forward);

	inherit_expression_dependencies(result_id, op0);
	inherit_expression_dependencies(result_id, op1);
}

void CompilerGLSL::emit_unrolled_unary_op(uint32_t result_type, uint32_t result_id, uint32_t operand, const char *op)
{
	auto &type = get<SPIRType>(result_type);
	auto expr = type_to_glsl_constructor(type);
	expr += '(';
	for (uint32_t i = 0; i < type.vecsize; i++)
	{
		// Make sure to call to_expression multiple times to ensure
		// that these expressions are properly flushed to temporaries if needed.
		expr += op;
		expr += to_extract_component_expression(operand, i);

		if (i + 1 < type.vecsize)
			expr += ", ";
	}
	expr += ')';
	emit_op(result_type, result_id, expr, should_forward(operand));

	inherit_expression_dependencies(result_id, operand);
}

void CompilerGLSL::emit_unrolled_binary_op(uint32_t result_type, uint32_t result_id, uint32_t op0, uint32_t op1,
                                           const char *op, bool negate, SPIRType::BaseType expected_type)
{
	auto &type0 = expression_type(op0);
	auto &type1 = expression_type(op1);

	SPIRType target_type0 = type0;
	SPIRType target_type1 = type1;
	target_type0.basetype = expected_type;
	target_type1.basetype = expected_type;
	target_type0.vecsize = 1;
	target_type1.vecsize = 1;

	auto &type = get<SPIRType>(result_type);
	auto expr = type_to_glsl_constructor(type);
	expr += '(';
	for (uint32_t i = 0; i < type.vecsize; i++)
	{
		// Make sure to call to_expression multiple times to ensure
		// that these expressions are properly flushed to temporaries if needed.
		if (negate)
			expr += "!(";

		if (expected_type != SPIRType::Unknown && type0.basetype != expected_type)
			expr += bitcast_expression(target_type0, type0.basetype, to_extract_component_expression(op0, i));
		else
			expr += to_extract_component_expression(op0, i);

		expr += ' ';
		expr += op;
		expr += ' ';

		if (expected_type != SPIRType::Unknown && type1.basetype != expected_type)
			expr += bitcast_expression(target_type1, type1.basetype, to_extract_component_expression(op1, i));
		else
			expr += to_extract_component_expression(op1, i);

		if (negate)
			expr += ")";

		if (i + 1 < type.vecsize)
			expr += ", ";
	}
	expr += ')';
	emit_op(result_type, result_id, expr, should_forward(op0) && should_forward(op1));

	inherit_expression_dependencies(result_id, op0);
	inherit_expression_dependencies(result_id, op1);
}

SPIRType CompilerGLSL::binary_op_bitcast_helper(string &cast_op0, string &cast_op1, SPIRType::BaseType &input_type,
                                                uint32_t op0, uint32_t op1, bool skip_cast_if_equal_type)
{
	auto &type0 = expression_type(op0);
	auto &type1 = expression_type(op1);

	// We have to bitcast if our inputs are of different type, or if our types are not equal to expected inputs.
	// For some functions like OpIEqual and INotEqual, we don't care if inputs are of different types than expected
	// since equality test is exactly the same.
	bool cast = (type0.basetype != type1.basetype) || (!skip_cast_if_equal_type && type0.basetype != input_type);

	// Create a fake type so we can bitcast to it.
	// We only deal with regular arithmetic types here like int, uints and so on.
	SPIRType expected_type;
	expected_type.basetype = input_type;
	expected_type.vecsize = type0.vecsize;
	expected_type.columns = type0.columns;
	expected_type.width = type0.width;

	if (cast)
	{
		cast_op0 = bitcast_glsl(expected_type, op0);
		cast_op1 = bitcast_glsl(expected_type, op1);
	}
	else
	{
		// If we don't cast, our actual input type is that of the first (or second) argument.
		cast_op0 = to_enclosed_unpacked_expression(op0);
		cast_op1 = to_enclosed_unpacked_expression(op1);
		input_type = type0.basetype;
	}

	return expected_type;
}

void CompilerGLSL::emit_binary_op_cast(uint32_t result_type, uint32_t result_id, uint32_t op0, uint32_t op1,
                                       const char *op, SPIRType::BaseType input_type, bool skip_cast_if_equal_type)
{
	string cast_op0, cast_op1;
	auto expected_type = binary_op_bitcast_helper(cast_op0, cast_op1, input_type, op0, op1, skip_cast_if_equal_type);
	auto &out_type = get<SPIRType>(result_type);

	// We might have casted away from the result type, so bitcast again.
	// For example, arithmetic right shift with uint inputs.
	// Special case boolean outputs since relational opcodes output booleans instead of int/uint.
	string expr;
	if (out_type.basetype != input_type && out_type.basetype != SPIRType::Boolean)
	{
		expected_type.basetype = input_type;
		expr = bitcast_glsl_op(out_type, expected_type);
		expr += '(';
		expr += join(cast_op0, " ", op, " ", cast_op1);
		expr += ')';
	}
	else
		expr += join(cast_op0, " ", op, " ", cast_op1);

	emit_op(result_type, result_id, expr, should_forward(op0) && should_forward(op1));
	inherit_expression_dependencies(result_id, op0);
	inherit_expression_dependencies(result_id, op1);
}

void CompilerGLSL::emit_unary_func_op(uint32_t result_type, uint32_t result_id, uint32_t op0, const char *op)
{
	bool forward = should_forward(op0);
	emit_op(result_type, result_id, join(op, "(", to_unpacked_expression(op0), ")"), forward);
	inherit_expression_dependencies(result_id, op0);
}

void CompilerGLSL::emit_binary_func_op(uint32_t result_type, uint32_t result_id, uint32_t op0, uint32_t op1,
                                       const char *op)
{
	bool forward = should_forward(op0) && should_forward(op1);
	emit_op(result_type, result_id, join(op, "(", to_unpacked_expression(op0), ", ", to_unpacked_expression(op1), ")"),
	        forward);
	inherit_expression_dependencies(result_id, op0);
	inherit_expression_dependencies(result_id, op1);
}

void CompilerGLSL::emit_unary_func_op_cast(uint32_t result_type, uint32_t result_id, uint32_t op0, const char *op,
                                           SPIRType::BaseType input_type, SPIRType::BaseType expected_result_type)
{
	auto &out_type = get<SPIRType>(result_type);
	auto &expr_type = expression_type(op0);
	auto expected_type = out_type;

	// Bit-widths might be different in unary cases because we use it for SConvert/UConvert and friends.
	expected_type.basetype = input_type;
	expected_type.width = expr_type.width;
	string cast_op = expr_type.basetype != input_type ? bitcast_glsl(expected_type, op0) : to_unpacked_expression(op0);

	string expr;
	if (out_type.basetype != expected_result_type)
	{
		expected_type.basetype = expected_result_type;
		expected_type.width = out_type.width;
		expr = bitcast_glsl_op(out_type, expected_type);
		expr += '(';
		expr += join(op, "(", cast_op, ")");
		expr += ')';
	}
	else
	{
		expr += join(op, "(", cast_op, ")");
	}

	emit_op(result_type, result_id, expr, should_forward(op0));
	inherit_expression_dependencies(result_id, op0);
}

// Very special case. Handling bitfieldExtract requires us to deal with different bitcasts of different signs
// and different vector sizes all at once. Need a special purpose method here.
void CompilerGLSL::emit_trinary_func_op_bitextract(uint32_t result_type, uint32_t result_id, uint32_t op0, uint32_t op1,
                                                   uint32_t op2, const char *op,
                                                   SPIRType::BaseType expected_result_type,
                                                   SPIRType::BaseType input_type0, SPIRType::BaseType input_type1,
                                                   SPIRType::BaseType input_type2)
{
	auto &out_type = get<SPIRType>(result_type);
	auto expected_type = out_type;
	expected_type.basetype = input_type0;

	string cast_op0 =
	    expression_type(op0).basetype != input_type0 ? bitcast_glsl(expected_type, op0) : to_unpacked_expression(op0);

	auto op1_expr = to_unpacked_expression(op1);
	auto op2_expr = to_unpacked_expression(op2);

	// Use value casts here instead. Input must be exactly int or uint, but SPIR-V might be 16-bit.
	expected_type.basetype = input_type1;
	expected_type.vecsize = 1;
	string cast_op1 = expression_type(op1).basetype != input_type1 ?
	                      join(type_to_glsl_constructor(expected_type), "(", op1_expr, ")") :
	                      op1_expr;

	expected_type.basetype = input_type2;
	expected_type.vecsize = 1;
	string cast_op2 = expression_type(op2).basetype != input_type2 ?
	                      join(type_to_glsl_constructor(expected_type), "(", op2_expr, ")") :
	                      op2_expr;

	string expr;
	if (out_type.basetype != expected_result_type)
	{
		expected_type.vecsize = out_type.vecsize;
		expected_type.basetype = expected_result_type;
		expr = bitcast_glsl_op(out_type, expected_type);
		expr += '(';
		expr += join(op, "(", cast_op0, ", ", cast_op1, ", ", cast_op2, ")");
		expr += ')';
	}
	else
	{
		expr += join(op, "(", cast_op0, ", ", cast_op1, ", ", cast_op2, ")");
	}

	emit_op(result_type, result_id, expr, should_forward(op0) && should_forward(op1) && should_forward(op2));
	inherit_expression_dependencies(result_id, op0);
	inherit_expression_dependencies(result_id, op1);
	inherit_expression_dependencies(result_id, op2);
}

void CompilerGLSL::emit_trinary_func_op_cast(uint32_t result_type, uint32_t result_id, uint32_t op0, uint32_t op1,
                                             uint32_t op2, const char *op, SPIRType::BaseType input_type)
{
	auto &out_type = get<SPIRType>(result_type);
	auto expected_type = out_type;
	expected_type.basetype = input_type;
	string cast_op0 =
	    expression_type(op0).basetype != input_type ? bitcast_glsl(expected_type, op0) : to_unpacked_expression(op0);
	string cast_op1 =
	    expression_type(op1).basetype != input_type ? bitcast_glsl(expected_type, op1) : to_unpacked_expression(op1);
	string cast_op2 =
	    expression_type(op2).basetype != input_type ? bitcast_glsl(expected_type, op2) : to_unpacked_expression(op2);

	string expr;
	if (out_type.basetype != input_type)
	{
		expr = bitcast_glsl_op(out_type, expected_type);
		expr += '(';
		expr += join(op, "(", cast_op0, ", ", cast_op1, ", ", cast_op2, ")");
		expr += ')';
	}
	else
	{
		expr += join(op, "(", cast_op0, ", ", cast_op1, ", ", cast_op2, ")");
	}

	emit_op(result_type, result_id, expr, should_forward(op0) && should_forward(op1) && should_forward(op2));
	inherit_expression_dependencies(result_id, op0);
	inherit_expression_dependencies(result_id, op1);
	inherit_expression_dependencies(result_id, op2);
}

void CompilerGLSL::emit_binary_func_op_cast_clustered(uint32_t result_type, uint32_t result_id, uint32_t op0,
                                                      uint32_t op1, const char *op, SPIRType::BaseType input_type)
{
	// Special purpose method for implementing clustered subgroup opcodes.
	// Main difference is that op1 does not participate in any casting, it needs to be a literal.
	auto &out_type = get<SPIRType>(result_type);
	auto expected_type = out_type;
	expected_type.basetype = input_type;
	string cast_op0 =
	    expression_type(op0).basetype != input_type ? bitcast_glsl(expected_type, op0) : to_unpacked_expression(op0);

	string expr;
	if (out_type.basetype != input_type)
	{
		expr = bitcast_glsl_op(out_type, expected_type);
		expr += '(';
		expr += join(op, "(", cast_op0, ", ", to_expression(op1), ")");
		expr += ')';
	}
	else
	{
		expr += join(op, "(", cast_op0, ", ", to_expression(op1), ")");
	}

	emit_op(result_type, result_id, expr, should_forward(op0));
	inherit_expression_dependencies(result_id, op0);
}

void CompilerGLSL::emit_binary_func_op_cast(uint32_t result_type, uint32_t result_id, uint32_t op0, uint32_t op1,
                                            const char *op, SPIRType::BaseType input_type, bool skip_cast_if_equal_type)
{
	string cast_op0, cast_op1;
	auto expected_type = binary_op_bitcast_helper(cast_op0, cast_op1, input_type, op0, op1, skip_cast_if_equal_type);
	auto &out_type = get<SPIRType>(result_type);

	// Special case boolean outputs since relational opcodes output booleans instead of int/uint.
	string expr;
	if (out_type.basetype != input_type && out_type.basetype != SPIRType::Boolean)
	{
		expected_type.basetype = input_type;
		expr = bitcast_glsl_op(out_type, expected_type);
		expr += '(';
		expr += join(op, "(", cast_op0, ", ", cast_op1, ")");
		expr += ')';
	}
	else
	{
		expr += join(op, "(", cast_op0, ", ", cast_op1, ")");
	}

	emit_op(result_type, result_id, expr, should_forward(op0) && should_forward(op1));
	inherit_expression_dependencies(result_id, op0);
	inherit_expression_dependencies(result_id, op1);
}

void CompilerGLSL::emit_trinary_func_op(uint32_t result_type, uint32_t result_id, uint32_t op0, uint32_t op1,
                                        uint32_t op2, const char *op)
{
	bool forward = should_forward(op0) && should_forward(op1) && should_forward(op2);
	emit_op(result_type, result_id,
	        join(op, "(", to_unpacked_expression(op0), ", ", to_unpacked_expression(op1), ", ",
	             to_unpacked_expression(op2), ")"),
	        forward);

	inherit_expression_dependencies(result_id, op0);
	inherit_expression_dependencies(result_id, op1);
	inherit_expression_dependencies(result_id, op2);
}

void CompilerGLSL::emit_quaternary_func_op(uint32_t result_type, uint32_t result_id, uint32_t op0, uint32_t op1,
                                           uint32_t op2, uint32_t op3, const char *op)
{
	bool forward = should_forward(op0) && should_forward(op1) && should_forward(op2) && should_forward(op3);
	emit_op(result_type, result_id,
	        join(op, "(", to_unpacked_expression(op0), ", ", to_unpacked_expression(op1), ", ",
	             to_unpacked_expression(op2), ", ", to_unpacked_expression(op3), ")"),
	        forward);

	inherit_expression_dependencies(result_id, op0);
	inherit_expression_dependencies(result_id, op1);
	inherit_expression_dependencies(result_id, op2);
	inherit_expression_dependencies(result_id, op3);
}

void CompilerGLSL::emit_bitfield_insert_op(uint32_t result_type, uint32_t result_id, uint32_t op0, uint32_t op1,
                                           uint32_t op2, uint32_t op3, const char *op,
                                           SPIRType::BaseType offset_count_type)
{
	// Only need to cast offset/count arguments. Types of base/insert must be same as result type,
	// and bitfieldInsert is sign invariant.
	bool forward = should_forward(op0) && should_forward(op1) && should_forward(op2) && should_forward(op3);

	auto op0_expr = to_unpacked_expression(op0);
	auto op1_expr = to_unpacked_expression(op1);
	auto op2_expr = to_unpacked_expression(op2);
	auto op3_expr = to_unpacked_expression(op3);

	SPIRType target_type;
	target_type.vecsize = 1;
	target_type.basetype = offset_count_type;

	if (expression_type(op2).basetype != offset_count_type)
	{
		// Value-cast here. Input might be 16-bit. GLSL requires int.
		op2_expr = join(type_to_glsl_constructor(target_type), "(", op2_expr, ")");
	}

	if (expression_type(op3).basetype != offset_count_type)
	{
		// Value-cast here. Input might be 16-bit. GLSL requires int.
		op3_expr = join(type_to_glsl_constructor(target_type), "(", op3_expr, ")");
	}

	emit_op(result_type, result_id, join(op, "(", op0_expr, ", ", op1_expr, ", ", op2_expr, ", ", op3_expr, ")"),
	        forward);

	inherit_expression_dependencies(result_id, op0);
	inherit_expression_dependencies(result_id, op1);
	inherit_expression_dependencies(result_id, op2);
	inherit_expression_dependencies(result_id, op3);
}

// EXT_shader_texture_lod only concerns fragment shaders so lod tex functions
// are not allowed in ES 2 vertex shaders. But SPIR-V only supports lod tex
// functions in vertex shaders so we revert those back to plain calls when
// the lod is a constant value of zero.
bool CompilerGLSL::check_explicit_lod_allowed(uint32_t lod)
{
	auto &execution = get_entry_point();
	bool allowed = !is_legacy_es() || execution.model == ExecutionModelFragment;
	if (!allowed && lod != 0)
	{
		auto *lod_constant = maybe_get<SPIRConstant>(lod);
		if (!lod_constant || lod_constant->scalar_f32() != 0.0f)
		{
			SPIRV_CROSS_THROW("Explicit lod not allowed in legacy ES non-fragment shaders.");
		}
	}
	return allowed;
}

string CompilerGLSL::legacy_tex_op(const std::string &op, const SPIRType &imgtype, uint32_t lod, uint32_t tex)
{
	const char *type;
	switch (imgtype.image.dim)
	{
	case spv::Dim1D:
		type = (imgtype.image.arrayed && !options.es) ? "1DArray" : "1D";
		break;
	case spv::Dim2D:
		type = (imgtype.image.arrayed && !options.es) ? "2DArray" : "2D";
		break;
	case spv::Dim3D:
		type = "3D";
		break;
	case spv::DimCube:
		type = "Cube";
		break;
	case spv::DimRect:
		type = "2DRect";
		break;
	case spv::DimBuffer:
		type = "Buffer";
		break;
	case spv::DimSubpassData:
		type = "2D";
		break;
	default:
		type = "";
		break;
	}

	bool use_explicit_lod = check_explicit_lod_allowed(lod);

	if (op == "textureLod" || op == "textureProjLod" || op == "textureGrad" || op == "textureProjGrad")
	{
		if (is_legacy_es())
		{
			if (use_explicit_lod)
				require_extension_internal("GL_EXT_shader_texture_lod");
		}
		else if (is_legacy())
			require_extension_internal("GL_ARB_shader_texture_lod");
	}

	if (op == "textureLodOffset" || op == "textureProjLodOffset")
	{
		if (is_legacy_es())
			SPIRV_CROSS_THROW(join(op, " not allowed in legacy ES"));

		require_extension_internal("GL_EXT_gpu_shader4");
	}

	// GLES has very limited support for shadow samplers.
	// Basically shadow2D and shadow2DProj work through EXT_shadow_samplers,
	// everything else can just throw
	if (image_is_comparison(imgtype, tex) && is_legacy_es())
	{
		if (op == "texture" || op == "textureProj")
			require_extension_internal("GL_EXT_shadow_samplers");
		else
			SPIRV_CROSS_THROW(join(op, " not allowed on depth samplers in legacy ES"));
	}

	bool is_es_and_depth = is_legacy_es() && image_is_comparison(imgtype, tex);
	std::string type_prefix = image_is_comparison(imgtype, tex) ? "shadow" : "texture";

	if (op == "texture")
		return is_es_and_depth ? join(type_prefix, type, "EXT") : join(type_prefix, type);
	else if (op == "textureLod")
	{
		if (use_explicit_lod)
			return join(type_prefix, type, is_legacy_es() ? "LodEXT" : "Lod");
		else
			return join(type_prefix, type);
	}
	else if (op == "textureProj")
		return join(type_prefix, type, is_es_and_depth ? "ProjEXT" : "Proj");
	else if (op == "textureGrad")
		return join(type_prefix, type, is_legacy_es() ? "GradEXT" : is_legacy_desktop() ? "GradARB" : "Grad");
	else if (op == "textureProjLod")
	{
		if (use_explicit_lod)
			return join(type_prefix, type, is_legacy_es() ? "ProjLodEXT" : "ProjLod");
		else
			return join(type_prefix, type, "Proj");
	}
	else if (op == "textureLodOffset")
	{
		if (use_explicit_lod)
			return join(type_prefix, type, "LodOffset");
		else
			return join(type_prefix, type);
	}
	else if (op == "textureProjGrad")
		return join(type_prefix, type,
		            is_legacy_es() ? "ProjGradEXT" : is_legacy_desktop() ? "ProjGradARB" : "ProjGrad");
	else if (op == "textureProjLodOffset")
	{
		if (use_explicit_lod)
			return join(type_prefix, type, "ProjLodOffset");
		else
			return join(type_prefix, type, "ProjOffset");
	}
	else
	{
		SPIRV_CROSS_THROW(join("Unsupported legacy texture op: ", op));
	}
}

bool CompilerGLSL::to_trivial_mix_op(const SPIRType &type, string &op, uint32_t left, uint32_t right, uint32_t lerp)
{
	auto *cleft = maybe_get<SPIRConstant>(left);
	auto *cright = maybe_get<SPIRConstant>(right);
	auto &lerptype = expression_type(lerp);

	// If our targets aren't constants, we cannot use construction.
	if (!cleft || !cright)
		return false;

	// If our targets are spec constants, we cannot use construction.
	if (cleft->specialization || cright->specialization)
		return false;

	// We can only use trivial construction if we have a scalar
	// (should be possible to do it for vectors as well, but that is overkill for now).
	if (lerptype.basetype != SPIRType::Boolean || lerptype.vecsize > 1)
		return false;

	// If our bool selects between 0 and 1, we can cast from bool instead, making our trivial constructor.
	bool ret = false;
	switch (type.basetype)
	{
	case SPIRType::Short:
	case SPIRType::UShort:
		ret = cleft->scalar_u16() == 0 && cright->scalar_u16() == 1;
		break;

	case SPIRType::Int:
	case SPIRType::UInt:
		ret = cleft->scalar() == 0 && cright->scalar() == 1;
		break;

	case SPIRType::Half:
		ret = cleft->scalar_f16() == 0.0f && cright->scalar_f16() == 1.0f;
		break;

	case SPIRType::Float:
		ret = cleft->scalar_f32() == 0.0f && cright->scalar_f32() == 1.0f;
		break;

	case SPIRType::Double:
		ret = cleft->scalar_f64() == 0.0 && cright->scalar_f64() == 1.0;
		break;

	case SPIRType::Int64:
	case SPIRType::UInt64:
		ret = cleft->scalar_u64() == 0 && cright->scalar_u64() == 1;
		break;

	default:
		break;
	}

	if (ret)
		op = type_to_glsl_constructor(type);
	return ret;
}

string CompilerGLSL::to_ternary_expression(const SPIRType &restype, uint32_t select, uint32_t true_value,
                                           uint32_t false_value)
{
	string expr;
	auto &lerptype = expression_type(select);

	if (lerptype.vecsize == 1)
		expr = join(to_enclosed_expression(select), " ? ", to_enclosed_pointer_expression(true_value), " : ",
		            to_enclosed_pointer_expression(false_value));
	else
	{
		auto swiz = [this](uint32_t expression, uint32_t i) { return to_extract_component_expression(expression, i); };

		expr = type_to_glsl_constructor(restype);
		expr += "(";
		for (uint32_t i = 0; i < restype.vecsize; i++)
		{
			expr += swiz(select, i);
			expr += " ? ";
			expr += swiz(true_value, i);
			expr += " : ";
			expr += swiz(false_value, i);
			if (i + 1 < restype.vecsize)
				expr += ", ";
		}
		expr += ")";
	}

	return expr;
}

void CompilerGLSL::emit_mix_op(uint32_t result_type, uint32_t id, uint32_t left, uint32_t right, uint32_t lerp)
{
	auto &lerptype = expression_type(lerp);
	auto &restype = get<SPIRType>(result_type);

	// If this results in a variable pointer, assume it may be written through.
	if (restype.pointer)
	{
		register_write(left);
		register_write(right);
	}

	string mix_op;
	bool has_boolean_mix = *backend.boolean_mix_function &&
	                       ((options.es && options.version >= 310) || (!options.es && options.version >= 450));
	bool trivial_mix = to_trivial_mix_op(restype, mix_op, left, right, lerp);

	// Cannot use boolean mix when the lerp argument is just one boolean,
	// fall back to regular trinary statements.
	if (lerptype.vecsize == 1)
		has_boolean_mix = false;

	// If we can reduce the mix to a simple cast, do so.
	// This helps for cases like int(bool), uint(bool) which is implemented with
	// OpSelect bool 1 0.
	if (trivial_mix)
	{
		emit_unary_func_op(result_type, id, lerp, mix_op.c_str());
	}
	else if (!has_boolean_mix && lerptype.basetype == SPIRType::Boolean)
	{
		// Boolean mix not supported on desktop without extension.
		// Was added in OpenGL 4.5 with ES 3.1 compat.
		//
		// Could use GL_EXT_shader_integer_mix on desktop at least,
		// but Apple doesn't support it. :(
		// Just implement it as ternary expressions.
		auto expr = to_ternary_expression(get<SPIRType>(result_type), lerp, right, left);
		emit_op(result_type, id, expr, should_forward(left) && should_forward(right) && should_forward(lerp));
		inherit_expression_dependencies(id, left);
		inherit_expression_dependencies(id, right);
		inherit_expression_dependencies(id, lerp);
	}
	else if (lerptype.basetype == SPIRType::Boolean)
		emit_trinary_func_op(result_type, id, left, right, lerp, backend.boolean_mix_function);
	else
		emit_trinary_func_op(result_type, id, left, right, lerp, "mix");
}

string CompilerGLSL::to_combined_image_sampler(VariableID image_id, VariableID samp_id)
{
	// Keep track of the array indices we have used to load the image.
	// We'll need to use the same array index into the combined image sampler array.
	auto image_expr = to_expression(image_id);
	string array_expr;
	auto array_index = image_expr.find_first_of('[');
	if (array_index != string::npos)
		array_expr = image_expr.substr(array_index, string::npos);

	auto &args = current_function->arguments;

	// For GLSL and ESSL targets, we must enumerate all possible combinations for sampler2D(texture2D, sampler) and redirect
	// all possible combinations into new sampler2D uniforms.
	auto *image = maybe_get_backing_variable(image_id);
	auto *samp = maybe_get_backing_variable(samp_id);
	if (image)
		image_id = image->self;
	if (samp)
		samp_id = samp->self;

	auto image_itr = find_if(begin(args), end(args),
	                         [image_id](const SPIRFunction::Parameter &param) { return image_id == param.id; });

	auto sampler_itr = find_if(begin(args), end(args),
	                           [samp_id](const SPIRFunction::Parameter &param) { return samp_id == param.id; });

	if (image_itr != end(args) || sampler_itr != end(args))
	{
		// If any parameter originates from a parameter, we will find it in our argument list.
		bool global_image = image_itr == end(args);
		bool global_sampler = sampler_itr == end(args);
		VariableID iid = global_image ? image_id : VariableID(uint32_t(image_itr - begin(args)));
		VariableID sid = global_sampler ? samp_id : VariableID(uint32_t(sampler_itr - begin(args)));

		auto &combined = current_function->combined_parameters;
		auto itr = find_if(begin(combined), end(combined), [=](const SPIRFunction::CombinedImageSamplerParameter &p) {
			return p.global_image == global_image && p.global_sampler == global_sampler && p.image_id == iid &&
			       p.sampler_id == sid;
		});

		if (itr != end(combined))
			return to_expression(itr->id) + array_expr;
		else
		{
			SPIRV_CROSS_THROW(
			    "Cannot find mapping for combined sampler parameter, was build_combined_image_samplers() used "
			    "before compile() was called?");
		}
	}
	else
	{
		// For global sampler2D, look directly at the global remapping table.
		auto &mapping = combined_image_samplers;
		auto itr = find_if(begin(mapping), end(mapping), [image_id, samp_id](const CombinedImageSampler &combined) {
			return combined.image_id == image_id && combined.sampler_id == samp_id;
		});

		if (itr != end(combined_image_samplers))
			return to_expression(itr->combined_id) + array_expr;
		else
		{
			SPIRV_CROSS_THROW("Cannot find mapping for combined sampler, was build_combined_image_samplers() used "
			                  "before compile() was called?");
		}
	}
}

void CompilerGLSL::emit_sampled_image_op(uint32_t result_type, uint32_t result_id, uint32_t image_id, uint32_t samp_id)
{
	if (options.vulkan_semantics && combined_image_samplers.empty())
	{
		emit_binary_func_op(result_type, result_id, image_id, samp_id,
		                    type_to_glsl(get<SPIRType>(result_type), result_id).c_str());
	}
	else
	{
		// Make sure to suppress usage tracking. It is illegal to create temporaries of opaque types.
		emit_op(result_type, result_id, to_combined_image_sampler(image_id, samp_id), true, true);
	}

	// Make sure to suppress usage tracking and any expression invalidation.
	// It is illegal to create temporaries of opaque types.
	forwarded_temporaries.erase(result_id);
}

static inline bool image_opcode_is_sample_no_dref(Op op)
{
	switch (op)
	{
	case OpImageSampleExplicitLod:
	case OpImageSampleImplicitLod:
	case OpImageSampleProjExplicitLod:
	case OpImageSampleProjImplicitLod:
	case OpImageFetch:
	case OpImageRead:
	case OpImageSparseSampleExplicitLod:
	case OpImageSparseSampleImplicitLod:
	case OpImageSparseSampleProjExplicitLod:
	case OpImageSparseSampleProjImplicitLod:
	case OpImageSparseFetch:
	case OpImageSparseRead:
		return true;

	default:
		return false;
	}
}

void CompilerGLSL::emit_texture_op(const Instruction &i)
{
	auto *ops = stream(i);
	auto op = static_cast<Op>(i.op);

	SmallVector<uint32_t> inherited_expressions;

	uint32_t result_type_id = ops[0];
	uint32_t id = ops[1];

	bool forward = false;
	string expr = to_texture_op(i, &forward, inherited_expressions);
	emit_op(result_type_id, id, expr, forward);
	for (auto &inherit : inherited_expressions)
		inherit_expression_dependencies(id, inherit);

	switch (op)
	{
	case OpImageSampleDrefImplicitLod:
	case OpImageSampleImplicitLod:
	case OpImageSampleProjImplicitLod:
	case OpImageSampleProjDrefImplicitLod:
		register_control_dependent_expression(id);
		break;

	default:
		break;
	}
}

std::string CompilerGLSL::to_texture_op(const Instruction &i, bool *forward,
                                        SmallVector<uint32_t> &inherited_expressions)
{
	auto *ops = stream(i);
	auto op = static_cast<Op>(i.op);
	uint32_t length = i.length;

	uint32_t result_type_id = ops[0];
	VariableID img = ops[2];
	uint32_t coord = ops[3];
	uint32_t dref = 0;
	uint32_t comp = 0;
	bool gather = false;
	bool proj = false;
	bool fetch = false;
	const uint32_t *opt = nullptr;

	auto &result_type = get<SPIRType>(result_type_id);

	inherited_expressions.push_back(coord);

	// Make sure non-uniform decoration is back-propagated to where it needs to be.
	if (has_decoration(img, DecorationNonUniformEXT))
		propagate_nonuniform_qualifier(img);

	switch (op)
	{
	case OpImageSampleDrefImplicitLod:
	case OpImageSampleDrefExplicitLod:
		dref = ops[4];
		opt = &ops[5];
		length -= 5;
		break;

	case OpImageSampleProjDrefImplicitLod:
	case OpImageSampleProjDrefExplicitLod:
		dref = ops[4];
		opt = &ops[5];
		length -= 5;
		proj = true;
		break;

	case OpImageDrefGather:
		dref = ops[4];
		opt = &ops[5];
		length -= 5;
		gather = true;
		break;

	case OpImageGather:
		comp = ops[4];
		opt = &ops[5];
		length -= 5;
		gather = true;
		break;

	case OpImageFetch:
	case OpImageRead: // Reads == fetches in Metal (other langs will not get here)
		opt = &ops[4];
		length -= 4;
		fetch = true;
		break;

	case OpImageSampleProjImplicitLod:
	case OpImageSampleProjExplicitLod:
		opt = &ops[4];
		length -= 4;
		proj = true;
		break;

	default:
		opt = &ops[4];
		length -= 4;
		break;
	}

	// Bypass pointers because we need the real image struct
	auto &type = expression_type(img);
	auto &imgtype = get<SPIRType>(type.self);

	uint32_t coord_components = 0;
	switch (imgtype.image.dim)
	{
	case spv::Dim1D:
		coord_components = 1;
		break;
	case spv::Dim2D:
		coord_components = 2;
		break;
	case spv::Dim3D:
		coord_components = 3;
		break;
	case spv::DimCube:
		coord_components = 3;
		break;
	case spv::DimBuffer:
		coord_components = 1;
		break;
	default:
		coord_components = 2;
		break;
	}

	if (dref)
		inherited_expressions.push_back(dref);

	if (proj)
		coord_components++;
	if (imgtype.image.arrayed)
		coord_components++;

	uint32_t bias = 0;
	uint32_t lod = 0;
	uint32_t grad_x = 0;
	uint32_t grad_y = 0;
	uint32_t coffset = 0;
	uint32_t offset = 0;
	uint32_t coffsets = 0;
	uint32_t sample = 0;
	uint32_t minlod = 0;
	uint32_t flags = 0;

	if (length)
	{
		flags = *opt++;
		length--;
	}

	auto test = [&](uint32_t &v, uint32_t flag) {
		if (length && (flags & flag))
		{
			v = *opt++;
			inherited_expressions.push_back(v);
			length--;
		}
	};

	test(bias, ImageOperandsBiasMask);
	test(lod, ImageOperandsLodMask);
	test(grad_x, ImageOperandsGradMask);
	test(grad_y, ImageOperandsGradMask);
	test(coffset, ImageOperandsConstOffsetMask);
	test(offset, ImageOperandsOffsetMask);
	test(coffsets, ImageOperandsConstOffsetsMask);
	test(sample, ImageOperandsSampleMask);
	test(minlod, ImageOperandsMinLodMask);

	string expr;
	expr += to_function_name(img, imgtype, !!fetch, !!gather, !!proj, !!coffsets, (!!coffset || !!offset),
	                         (!!grad_x || !!grad_y), !!dref, lod, minlod);
	expr += "(";
	expr += to_function_args(img, imgtype, fetch, gather, proj, coord, coord_components, dref, grad_x, grad_y, lod,
	                         coffset, offset, bias, comp, sample, minlod, forward);
	expr += ")";

	// texture(samplerXShadow) returns float. shadowX() returns vec4. Swizzle here.
	if (is_legacy() && image_is_comparison(imgtype, img))
		expr += ".r";

	// Sampling from a texture which was deduced to be a depth image, might actually return 1 component here.
	// Remap back to 4 components as sampling opcodes expect.
	if (backend.comparison_image_samples_scalar && image_opcode_is_sample_no_dref(op))
	{
		bool image_is_depth = false;
		const auto *combined = maybe_get<SPIRCombinedImageSampler>(img);
		VariableID image_id = combined ? combined->image : img;

		if (combined && image_is_comparison(imgtype, combined->image))
			image_is_depth = true;
		else if (image_is_comparison(imgtype, img))
			image_is_depth = true;

		// We must also check the backing variable for the image.
		// We might have loaded an OpImage, and used that handle for two different purposes.
		// Once with comparison, once without.
		auto *image_variable = maybe_get_backing_variable(image_id);
		if (image_variable && image_is_comparison(get<SPIRType>(image_variable->basetype), image_variable->self))
			image_is_depth = true;

		if (image_is_depth)
			expr = remap_swizzle(result_type, 1, expr);
	}

	if (!backend.support_small_type_sampling_result && result_type.width < 32)
	{
		// Just value cast (narrowing) to expected type since we cannot rely on narrowing to work automatically.
		// Hopefully compiler picks this up and converts the texturing instruction to the appropriate precision.
		expr = join(type_to_glsl_constructor(result_type), "(", expr, ")");
	}

	// Deals with reads from MSL. We might need to downconvert to fewer components.
	if (op == OpImageRead)
		expr = remap_swizzle(result_type, 4, expr);

	return expr;
}

bool CompilerGLSL::expression_is_constant_null(uint32_t id) const
{
	auto *c = maybe_get<SPIRConstant>(id);
	if (!c)
		return false;
	return c->constant_is_null();
}

// Returns the function name for a texture sampling function for the specified image and sampling characteristics.
// For some subclasses, the function is a method on the specified image.
string CompilerGLSL::to_function_name(VariableID tex, const SPIRType &imgtype, bool is_fetch, bool is_gather,
                                      bool is_proj, bool has_array_offsets, bool has_offset, bool has_grad, bool,
                                      uint32_t lod, uint32_t minlod)
{
	if (minlod != 0)
		SPIRV_CROSS_THROW("Sparse texturing not yet supported.");

	string fname;

	// textureLod on sampler2DArrayShadow and samplerCubeShadow does not exist in GLSL for some reason.
	// To emulate this, we will have to use textureGrad with a constant gradient of 0.
	// The workaround will assert that the LOD is in fact constant 0, or we cannot emit correct code.
	// This happens for HLSL SampleCmpLevelZero on Texture2DArray and TextureCube.
	bool workaround_lod_array_shadow_as_grad = false;
	if (((imgtype.image.arrayed && imgtype.image.dim == Dim2D) || imgtype.image.dim == DimCube) &&
	    image_is_comparison(imgtype, tex) && lod)
	{
		if (!expression_is_constant_null(lod))
		{
			SPIRV_CROSS_THROW(
			    "textureLod on sampler2DArrayShadow is not constant 0.0. This cannot be expressed in GLSL.");
		}
		workaround_lod_array_shadow_as_grad = true;
	}

	if (is_fetch)
		fname += "texelFetch";
	else
	{
		fname += "texture";

		if (is_gather)
			fname += "Gather";
		if (has_array_offsets)
			fname += "Offsets";
		if (is_proj)
			fname += "Proj";
		if (has_grad || workaround_lod_array_shadow_as_grad)
			fname += "Grad";
		if (!!lod && !workaround_lod_array_shadow_as_grad)
			fname += "Lod";
	}

	if (has_offset)
		fname += "Offset";

	return is_legacy() ? legacy_tex_op(fname, imgtype, lod, tex) : fname;
}

std::string CompilerGLSL::convert_separate_image_to_expression(uint32_t id)
{
	auto *var = maybe_get_backing_variable(id);

	// If we are fetching from a plain OpTypeImage, we must combine with a dummy sampler in GLSL.
	// In Vulkan GLSL, we can make use of the newer GL_EXT_samplerless_texture_functions.
	if (var)
	{
		auto &type = get<SPIRType>(var->basetype);
		if (type.basetype == SPIRType::Image && type.image.sampled == 1 && type.image.dim != DimBuffer)
		{
			if (options.vulkan_semantics)
			{
				if (dummy_sampler_id)
				{
					// Don't need to consider Shadow state since the dummy sampler is always non-shadow.
					auto sampled_type = type;
					sampled_type.basetype = SPIRType::SampledImage;
					return join(type_to_glsl(sampled_type), "(", to_expression(id), ", ",
					            to_expression(dummy_sampler_id), ")");
				}
				else
				{
					// Newer glslang supports this extension to deal with texture2D as argument to texture functions.
					require_extension_internal("GL_EXT_samplerless_texture_functions");
				}
			}
			else
			{
				if (!dummy_sampler_id)
					SPIRV_CROSS_THROW(
					    "Cannot find dummy sampler ID. Was build_dummy_sampler_for_combined_images() called?");

				return to_combined_image_sampler(id, dummy_sampler_id);
			}
		}
	}

	return to_expression(id);
}

// Returns the function args for a texture sampling function for the specified image and sampling characteristics.
string CompilerGLSL::to_function_args(VariableID img, const SPIRType &imgtype, bool is_fetch, bool is_gather,
                                      bool is_proj, uint32_t coord, uint32_t coord_components, uint32_t dref,
                                      uint32_t grad_x, uint32_t grad_y, uint32_t lod, uint32_t coffset, uint32_t offset,
                                      uint32_t bias, uint32_t comp, uint32_t sample, uint32_t /*minlod*/,
                                      bool *p_forward)
{
	string farg_str;
	if (is_fetch)
		farg_str = convert_separate_image_to_expression(img);
	else
		farg_str = to_expression(img);

	bool swizz_func = backend.swizzle_is_function;
	auto swizzle = [swizz_func](uint32_t comps, uint32_t in_comps) -> const char * {
		if (comps == in_comps)
			return "";

		switch (comps)
		{
		case 1:
			return ".x";
		case 2:
			return swizz_func ? ".xy()" : ".xy";
		case 3:
			return swizz_func ? ".xyz()" : ".xyz";
		default:
			return "";
		}
	};

	bool forward = should_forward(coord);

	// The IR can give us more components than we need, so chop them off as needed.
	auto swizzle_expr = swizzle(coord_components, expression_type(coord).vecsize);
	// Only enclose the UV expression if needed.
	auto coord_expr = (*swizzle_expr == '\0') ? to_expression(coord) : (to_enclosed_expression(coord) + swizzle_expr);

	// texelFetch only takes int, not uint.
	auto &coord_type = expression_type(coord);
	if (coord_type.basetype == SPIRType::UInt)
	{
		auto expected_type = coord_type;
		expected_type.vecsize = coord_components;
		expected_type.basetype = SPIRType::Int;
		coord_expr = bitcast_expression(expected_type, coord_type.basetype, coord_expr);
	}

	// textureLod on sampler2DArrayShadow and samplerCubeShadow does not exist in GLSL for some reason.
	// To emulate this, we will have to use textureGrad with a constant gradient of 0.
	// The workaround will assert that the LOD is in fact constant 0, or we cannot emit correct code.
	// This happens for HLSL SampleCmpLevelZero on Texture2DArray and TextureCube.
	bool workaround_lod_array_shadow_as_grad =
	    ((imgtype.image.arrayed && imgtype.image.dim == Dim2D) || imgtype.image.dim == DimCube) &&
	    image_is_comparison(imgtype, img) && lod;

	if (dref)
	{
		forward = forward && should_forward(dref);

		// SPIR-V splits dref and coordinate.
		if (is_gather || coord_components == 4) // GLSL also splits the arguments in two. Same for textureGather.
		{
			farg_str += ", ";
			farg_str += to_expression(coord);
			farg_str += ", ";
			farg_str += to_expression(dref);
		}
		else if (is_proj)
		{
			// Have to reshuffle so we get vec4(coord, dref, proj), special case.
			// Other shading languages splits up the arguments for coord and compare value like SPIR-V.
			// The coordinate type for textureProj shadow is always vec4 even for sampler1DShadow.
			farg_str += ", vec4(";

			if (imgtype.image.dim == Dim1D)
			{
				// Could reuse coord_expr, but we will mess up the temporary usage checking.
				farg_str += to_enclosed_expression(coord) + ".x";
				farg_str += ", ";
				farg_str += "0.0, ";
				farg_str += to_expression(dref);
				farg_str += ", ";
				farg_str += to_enclosed_expression(coord) + ".y)";
			}
			else if (imgtype.image.dim == Dim2D)
			{
				// Could reuse coord_expr, but we will mess up the temporary usage checking.
				farg_str += to_enclosed_expression(coord) + (swizz_func ? ".xy()" : ".xy");
				farg_str += ", ";
				farg_str += to_expression(dref);
				farg_str += ", ";
				farg_str += to_enclosed_expression(coord) + ".z)";
			}
			else
				SPIRV_CROSS_THROW("Invalid type for textureProj with shadow.");
		}
		else
		{
			// Create a composite which merges coord/dref into a single vector.
			auto type = expression_type(coord);
			type.vecsize = coord_components + 1;
			farg_str += ", ";
			farg_str += type_to_glsl_constructor(type);
			farg_str += "(";
			farg_str += coord_expr;
			farg_str += ", ";
			farg_str += to_expression(dref);
			farg_str += ")";
		}
	}
	else
	{
		farg_str += ", ";
		farg_str += coord_expr;
	}

	if (grad_x || grad_y)
	{
		forward = forward && should_forward(grad_x);
		forward = forward && should_forward(grad_y);
		farg_str += ", ";
		farg_str += to_expression(grad_x);
		farg_str += ", ";
		farg_str += to_expression(grad_y);
	}

	if (lod)
	{
		if (workaround_lod_array_shadow_as_grad)
		{
			// Implement textureGrad() instead. LOD == 0.0 is implemented as gradient of 0.0.
			// Implementing this as plain texture() is not safe on some implementations.
			if (imgtype.image.dim == Dim2D)
				farg_str += ", vec2(0.0), vec2(0.0)";
			else if (imgtype.image.dim == DimCube)
				farg_str += ", vec3(0.0), vec3(0.0)";
		}
		else
		{
			if (check_explicit_lod_allowed(lod))
			{
				forward = forward && should_forward(lod);
				farg_str += ", ";

				auto &lod_expr_type = expression_type(lod);

				// Lod expression for TexelFetch in GLSL must be int, and only int.
				if (is_fetch && imgtype.image.dim != DimBuffer && !imgtype.image.ms &&
				    lod_expr_type.basetype != SPIRType::Int)
				{
					farg_str += join("int(", to_expression(lod), ")");
				}
				else
				{
					farg_str += to_expression(lod);
				}
			}
		}
	}
	else if (is_fetch && imgtype.image.dim != DimBuffer && !imgtype.image.ms)
	{
		// Lod argument is optional in OpImageFetch, but we require a LOD value, pick 0 as the default.
		farg_str += ", 0";
	}

	if (coffset)
	{
		forward = forward && should_forward(coffset);
		farg_str += ", ";
		farg_str += to_expression(coffset);
	}
	else if (offset)
	{
		forward = forward && should_forward(offset);
		farg_str += ", ";
		farg_str += to_expression(offset);
	}

	if (bias)
	{
		forward = forward && should_forward(bias);
		farg_str += ", ";
		farg_str += to_expression(bias);
	}

	if (comp)
	{
		forward = forward && should_forward(comp);
		farg_str += ", ";
		farg_str += to_expression(comp);
	}

	if (sample)
	{
		farg_str += ", ";
		farg_str += to_expression(sample);
	}

	*p_forward = forward;

	return farg_str;
}

void CompilerGLSL::emit_glsl_op(uint32_t result_type, uint32_t id, uint32_t eop, const uint32_t *args, uint32_t length)
{
	auto op = static_cast<GLSLstd450>(eop);

	if (is_legacy() && is_unsigned_glsl_opcode(op))
		SPIRV_CROSS_THROW("Unsigned integers are not supported on legacy GLSL targets.");

	// If we need to do implicit bitcasts, make sure we do it with the correct type.
	uint32_t integer_width = get_integer_width_for_glsl_instruction(op, args, length);
	auto int_type = to_signed_basetype(integer_width);
	auto uint_type = to_unsigned_basetype(integer_width);

	switch (op)
	{
	// FP fiddling
	case GLSLstd450Round:
		emit_unary_func_op(result_type, id, args[0], "round");
		break;

	case GLSLstd450RoundEven:
		if ((options.es && options.version >= 300) || (!options.es && options.version >= 130))
			emit_unary_func_op(result_type, id, args[0], "roundEven");
		else
			SPIRV_CROSS_THROW("roundEven supported only in ESSL 300 and GLSL 130 and up.");
		break;

	case GLSLstd450Trunc:
		emit_unary_func_op(result_type, id, args[0], "trunc");
		break;
	case GLSLstd450SAbs:
		emit_unary_func_op_cast(result_type, id, args[0], "abs", int_type, int_type);
		break;
	case GLSLstd450FAbs:
		emit_unary_func_op(result_type, id, args[0], "abs");
		break;
	case GLSLstd450SSign:
		emit_unary_func_op_cast(result_type, id, args[0], "sign", int_type, int_type);
		break;
	case GLSLstd450FSign:
		emit_unary_func_op(result_type, id, args[0], "sign");
		break;
	case GLSLstd450Floor:
		emit_unary_func_op(result_type, id, args[0], "floor");
		break;
	case GLSLstd450Ceil:
		emit_unary_func_op(result_type, id, args[0], "ceil");
		break;
	case GLSLstd450Fract:
		emit_unary_func_op(result_type, id, args[0], "fract");
		break;
	case GLSLstd450Radians:
		emit_unary_func_op(result_type, id, args[0], "radians");
		break;
	case GLSLstd450Degrees:
		emit_unary_func_op(result_type, id, args[0], "degrees");
		break;
	case GLSLstd450Fma:
		if ((!options.es && options.version < 400) || (options.es && options.version < 320))
		{
			auto expr = join(to_enclosed_expression(args[0]), " * ", to_enclosed_expression(args[1]), " + ",
			                 to_enclosed_expression(args[2]));

			emit_op(result_type, id, expr,
			        should_forward(args[0]) && should_forward(args[1]) && should_forward(args[2]));
			for (uint32_t i = 0; i < 3; i++)
				inherit_expression_dependencies(id, args[i]);
		}
		else
			emit_trinary_func_op(result_type, id, args[0], args[1], args[2], "fma");
		break;
	case GLSLstd450Modf:
		register_call_out_argument(args[1]);
		forced_temporaries.insert(id);
		emit_binary_func_op(result_type, id, args[0], args[1], "modf");
		break;

	case GLSLstd450ModfStruct:
	{
		auto &type = get<SPIRType>(result_type);
		emit_uninitialized_temporary_expression(result_type, id);
		statement(to_expression(id), ".", to_member_name(type, 0), " = ", "modf(", to_expression(args[0]), ", ",
		          to_expression(id), ".", to_member_name(type, 1), ");");
		break;
	}

	// Minmax
	case GLSLstd450UMin:
		emit_binary_func_op_cast(result_type, id, args[0], args[1], "min", uint_type, false);
		break;

	case GLSLstd450SMin:
		emit_binary_func_op_cast(result_type, id, args[0], args[1], "min", int_type, false);
		break;

	case GLSLstd450FMin:
		emit_binary_func_op(result_type, id, args[0], args[1], "min");
		break;

	case GLSLstd450FMax:
		emit_binary_func_op(result_type, id, args[0], args[1], "max");
		break;

	case GLSLstd450UMax:
		emit_binary_func_op_cast(result_type, id, args[0], args[1], "max", uint_type, false);
		break;

	case GLSLstd450SMax:
		emit_binary_func_op_cast(result_type, id, args[0], args[1], "max", int_type, false);
		break;

	case GLSLstd450FClamp:
		emit_trinary_func_op(result_type, id, args[0], args[1], args[2], "clamp");
		break;

	case GLSLstd450UClamp:
		emit_trinary_func_op_cast(result_type, id, args[0], args[1], args[2], "clamp", uint_type);
		break;

	case GLSLstd450SClamp:
		emit_trinary_func_op_cast(result_type, id, args[0], args[1], args[2], "clamp", int_type);
		break;

	// Trig
	case GLSLstd450Sin:
		emit_unary_func_op(result_type, id, args[0], "sin");
		break;
	case GLSLstd450Cos:
		emit_unary_func_op(result_type, id, args[0], "cos");
		break;
	case GLSLstd450Tan:
		emit_unary_func_op(result_type, id, args[0], "tan");
		break;
	case GLSLstd450Asin:
		emit_unary_func_op(result_type, id, args[0], "asin");
		break;
	case GLSLstd450Acos:
		emit_unary_func_op(result_type, id, args[0], "acos");
		break;
	case GLSLstd450Atan:
		emit_unary_func_op(result_type, id, args[0], "atan");
		break;
	case GLSLstd450Sinh:
		emit_unary_func_op(result_type, id, args[0], "sinh");
		break;
	case GLSLstd450Cosh:
		emit_unary_func_op(result_type, id, args[0], "cosh");
		break;
	case GLSLstd450Tanh:
		emit_unary_func_op(result_type, id, args[0], "tanh");
		break;
	case GLSLstd450Asinh:
		emit_unary_func_op(result_type, id, args[0], "asinh");
		break;
	case GLSLstd450Acosh:
		emit_unary_func_op(result_type, id, args[0], "acosh");
		break;
	case GLSLstd450Atanh:
		emit_unary_func_op(result_type, id, args[0], "atanh");
		break;
	case GLSLstd450Atan2:
		emit_binary_func_op(result_type, id, args[0], args[1], "atan");
		break;

	// Exponentials
	case GLSLstd450Pow:
		emit_binary_func_op(result_type, id, args[0], args[1], "pow");
		break;
	case GLSLstd450Exp:
		emit_unary_func_op(result_type, id, args[0], "exp");
		break;
	case GLSLstd450Log:
		emit_unary_func_op(result_type, id, args[0], "log");
		break;
	case GLSLstd450Exp2:
		emit_unary_func_op(result_type, id, args[0], "exp2");
		break;
	case GLSLstd450Log2:
		emit_unary_func_op(result_type, id, args[0], "log2");
		break;
	case GLSLstd450Sqrt:
		emit_unary_func_op(result_type, id, args[0], "sqrt");
		break;
	case GLSLstd450InverseSqrt:
		emit_unary_func_op(result_type, id, args[0], "inversesqrt");
		break;

	// Matrix math
	case GLSLstd450Determinant:
		emit_unary_func_op(result_type, id, args[0], "determinant");
		break;
	case GLSLstd450MatrixInverse:
		emit_unary_func_op(result_type, id, args[0], "inverse");
		break;

	// Lerping
	case GLSLstd450FMix:
	case GLSLstd450IMix:
	{
		emit_mix_op(result_type, id, args[0], args[1], args[2]);
		break;
	}
	case GLSLstd450Step:
		emit_binary_func_op(result_type, id, args[0], args[1], "step");
		break;
	case GLSLstd450SmoothStep:
		emit_trinary_func_op(result_type, id, args[0], args[1], args[2], "smoothstep");
		break;

	// Packing
	case GLSLstd450Frexp:
		register_call_out_argument(args[1]);
		forced_temporaries.insert(id);
		emit_binary_func_op(result_type, id, args[0], args[1], "frexp");
		break;

	case GLSLstd450FrexpStruct:
	{
		auto &type = get<SPIRType>(result_type);
		emit_uninitialized_temporary_expression(result_type, id);
		statement(to_expression(id), ".", to_member_name(type, 0), " = ", "frexp(", to_expression(args[0]), ", ",
		          to_expression(id), ".", to_member_name(type, 1), ");");
		break;
	}

	case GLSLstd450Ldexp:
	{
		bool forward = should_forward(args[0]) && should_forward(args[1]);

		auto op0 = to_unpacked_expression(args[0]);
		auto op1 = to_unpacked_expression(args[1]);
		auto &op1_type = expression_type(args[1]);
		if (op1_type.basetype != SPIRType::Int)
		{
			// Need a value cast here.
			auto target_type = op1_type;
			target_type.basetype = SPIRType::Int;
			op1 = join(type_to_glsl_constructor(target_type), "(", op1, ")");
		}

		auto expr = join("ldexp(", op0, ", ", op1, ")");

		emit_op(result_type, id, expr, forward);
		inherit_expression_dependencies(id, args[0]);
		inherit_expression_dependencies(id, args[1]);
		break;
	}

	case GLSLstd450PackSnorm4x8:
		emit_unary_func_op(result_type, id, args[0], "packSnorm4x8");
		break;
	case GLSLstd450PackUnorm4x8:
		emit_unary_func_op(result_type, id, args[0], "packUnorm4x8");
		break;
	case GLSLstd450PackSnorm2x16:
		emit_unary_func_op(result_type, id, args[0], "packSnorm2x16");
		break;
	case GLSLstd450PackUnorm2x16:
		emit_unary_func_op(result_type, id, args[0], "packUnorm2x16");
		break;
	case GLSLstd450PackHalf2x16:
		emit_unary_func_op(result_type, id, args[0], "packHalf2x16");
		break;
	case GLSLstd450UnpackSnorm4x8:
		emit_unary_func_op(result_type, id, args[0], "unpackSnorm4x8");
		break;
	case GLSLstd450UnpackUnorm4x8:
		emit_unary_func_op(result_type, id, args[0], "unpackUnorm4x8");
		break;
	case GLSLstd450UnpackSnorm2x16:
		emit_unary_func_op(result_type, id, args[0], "unpackSnorm2x16");
		break;
	case GLSLstd450UnpackUnorm2x16:
		emit_unary_func_op(result_type, id, args[0], "unpackUnorm2x16");
		break;
	case GLSLstd450UnpackHalf2x16:
		emit_unary_func_op(result_type, id, args[0], "unpackHalf2x16");
		break;

	case GLSLstd450PackDouble2x32:
		emit_unary_func_op(result_type, id, args[0], "packDouble2x32");
		break;
	case GLSLstd450UnpackDouble2x32:
		emit_unary_func_op(result_type, id, args[0], "unpackDouble2x32");
		break;

	// Vector math
	case GLSLstd450Length:
		emit_unary_func_op(result_type, id, args[0], "length");
		break;
	case GLSLstd450Distance:
		emit_binary_func_op(result_type, id, args[0], args[1], "distance");
		break;
	case GLSLstd450Cross:
		emit_binary_func_op(result_type, id, args[0], args[1], "cross");
		break;
	case GLSLstd450Normalize:
		emit_unary_func_op(result_type, id, args[0], "normalize");
		break;
	case GLSLstd450FaceForward:
		emit_trinary_func_op(result_type, id, args[0], args[1], args[2], "faceforward");
		break;
	case GLSLstd450Reflect:
		emit_binary_func_op(result_type, id, args[0], args[1], "reflect");
		break;
	case GLSLstd450Refract:
		emit_trinary_func_op(result_type, id, args[0], args[1], args[2], "refract");
		break;

	// Bit-fiddling
	case GLSLstd450FindILsb:
		// findLSB always returns int.
		emit_unary_func_op_cast(result_type, id, args[0], "findLSB", expression_type(args[0]).basetype, int_type);
		break;

	case GLSLstd450FindSMsb:
		emit_unary_func_op_cast(result_type, id, args[0], "findMSB", int_type, int_type);
		break;

	case GLSLstd450FindUMsb:
		emit_unary_func_op_cast(result_type, id, args[0], "findMSB", uint_type,
		                        int_type); // findMSB always returns int.
		break;

	// Multisampled varying
	case GLSLstd450InterpolateAtCentroid:
		emit_unary_func_op(result_type, id, args[0], "interpolateAtCentroid");
		break;
	case GLSLstd450InterpolateAtSample:
		emit_binary_func_op(result_type, id, args[0], args[1], "interpolateAtSample");
		break;
	case GLSLstd450InterpolateAtOffset:
		emit_binary_func_op(result_type, id, args[0], args[1], "interpolateAtOffset");
		break;

	case GLSLstd450NMin:
	case GLSLstd450NMax:
	{
		emit_nminmax_op(result_type, id, args[0], args[1], op);
		break;
	}

	case GLSLstd450NClamp:
	{
		// Make sure we have a unique ID here to avoid aliasing the extra sub-expressions between clamp and NMin sub-op.
		// IDs cannot exceed 24 bits, so we can make use of the higher bits for some unique flags.
		uint32_t &max_id = extra_sub_expressions[id | 0x80000000u];
		if (!max_id)
			max_id = ir.increase_bound_by(1);

		// Inherit precision qualifiers.
		ir.meta[max_id] = ir.meta[id];

		emit_nminmax_op(result_type, max_id, args[0], args[1], GLSLstd450NMax);
		emit_nminmax_op(result_type, id, max_id, args[2], GLSLstd450NMin);
		break;
	}

	default:
		statement("// unimplemented GLSL op ", eop);
		break;
	}
}

void CompilerGLSL::emit_nminmax_op(uint32_t result_type, uint32_t id, uint32_t op0, uint32_t op1, GLSLstd450 op)
{
	// Need to emulate this call.
	uint32_t &ids = extra_sub_expressions[id];
	if (!ids)
	{
		ids = ir.increase_bound_by(5);
		auto btype = get<SPIRType>(result_type);
		btype.basetype = SPIRType::Boolean;
		set<SPIRType>(ids, btype);
	}

	uint32_t btype_id = ids + 0;
	uint32_t left_nan_id = ids + 1;
	uint32_t right_nan_id = ids + 2;
	uint32_t tmp_id = ids + 3;
	uint32_t mixed_first_id = ids + 4;

	// Inherit precision qualifiers.
	ir.meta[tmp_id] = ir.meta[id];
	ir.meta[mixed_first_id] = ir.meta[id];

	emit_unary_func_op(btype_id, left_nan_id, op0, "isnan");
	emit_unary_func_op(btype_id, right_nan_id, op1, "isnan");
	emit_binary_func_op(result_type, tmp_id, op0, op1, op == GLSLstd450NMin ? "min" : "max");
	emit_mix_op(result_type, mixed_first_id, tmp_id, op1, left_nan_id);
	emit_mix_op(result_type, id, mixed_first_id, op0, right_nan_id);
}

void CompilerGLSL::emit_spv_amd_shader_ballot_op(uint32_t result_type, uint32_t id, uint32_t eop, const uint32_t *args,
                                                 uint32_t)
{
	require_extension_internal("GL_AMD_shader_ballot");

	enum AMDShaderBallot
	{
		SwizzleInvocationsAMD = 1,
		SwizzleInvocationsMaskedAMD = 2,
		WriteInvocationAMD = 3,
		MbcntAMD = 4
	};

	auto op = static_cast<AMDShaderBallot>(eop);

	switch (op)
	{
	case SwizzleInvocationsAMD:
		emit_binary_func_op(result_type, id, args[0], args[1], "swizzleInvocationsAMD");
		register_control_dependent_expression(id);
		break;

	case SwizzleInvocationsMaskedAMD:
		emit_binary_func_op(result_type, id, args[0], args[1], "swizzleInvocationsMaskedAMD");
		register_control_dependent_expression(id);
		break;

	case WriteInvocationAMD:
		emit_trinary_func_op(result_type, id, args[0], args[1], args[2], "writeInvocationAMD");
		register_control_dependent_expression(id);
		break;

	case MbcntAMD:
		emit_unary_func_op(result_type, id, args[0], "mbcntAMD");
		register_control_dependent_expression(id);
		break;

	default:
		statement("// unimplemented SPV AMD shader ballot op ", eop);
		break;
	}
}

void CompilerGLSL::emit_spv_amd_shader_explicit_vertex_parameter_op(uint32_t result_type, uint32_t id, uint32_t eop,
                                                                    const uint32_t *args, uint32_t)
{
	require_extension_internal("GL_AMD_shader_explicit_vertex_parameter");

	enum AMDShaderExplicitVertexParameter
	{
		InterpolateAtVertexAMD = 1
	};

	auto op = static_cast<AMDShaderExplicitVertexParameter>(eop);

	switch (op)
	{
	case InterpolateAtVertexAMD:
		emit_binary_func_op(result_type, id, args[0], args[1], "interpolateAtVertexAMD");
		break;

	default:
		statement("// unimplemented SPV AMD shader explicit vertex parameter op ", eop);
		break;
	}
}

void CompilerGLSL::emit_spv_amd_shader_trinary_minmax_op(uint32_t result_type, uint32_t id, uint32_t eop,
                                                         const uint32_t *args, uint32_t)
{
	require_extension_internal("GL_AMD_shader_trinary_minmax");

	enum AMDShaderTrinaryMinMax
	{
		FMin3AMD = 1,
		UMin3AMD = 2,
		SMin3AMD = 3,
		FMax3AMD = 4,
		UMax3AMD = 5,
		SMax3AMD = 6,
		FMid3AMD = 7,
		UMid3AMD = 8,
		SMid3AMD = 9
	};

	auto op = static_cast<AMDShaderTrinaryMinMax>(eop);

	switch (op)
	{
	case FMin3AMD:
	case UMin3AMD:
	case SMin3AMD:
		emit_trinary_func_op(result_type, id, args[0], args[1], args[2], "min3");
		break;

	case FMax3AMD:
	case UMax3AMD:
	case SMax3AMD:
		emit_trinary_func_op(result_type, id, args[0], args[1], args[2], "max3");
		break;

	case FMid3AMD:
	case UMid3AMD:
	case SMid3AMD:
		emit_trinary_func_op(result_type, id, args[0], args[1], args[2], "mid3");
		break;

	default:
		statement("// unimplemented SPV AMD shader trinary minmax op ", eop);
		break;
	}
}

void CompilerGLSL::emit_spv_amd_gcn_shader_op(uint32_t result_type, uint32_t id, uint32_t eop, const uint32_t *args,
                                              uint32_t)
{
	require_extension_internal("GL_AMD_gcn_shader");

	enum AMDGCNShader
	{
		CubeFaceIndexAMD = 1,
		CubeFaceCoordAMD = 2,
		TimeAMD = 3
	};

	auto op = static_cast<AMDGCNShader>(eop);

	switch (op)
	{
	case CubeFaceIndexAMD:
		emit_unary_func_op(result_type, id, args[0], "cubeFaceIndexAMD");
		break;
	case CubeFaceCoordAMD:
		emit_unary_func_op(result_type, id, args[0], "cubeFaceCoordAMD");
		break;
	case TimeAMD:
	{
		string expr = "timeAMD()";
		emit_op(result_type, id, expr, true);
		register_control_dependent_expression(id);
		break;
	}

	default:
		statement("// unimplemented SPV AMD gcn shader op ", eop);
		break;
	}
}

void CompilerGLSL::emit_subgroup_op(const Instruction &i)
{
	const uint32_t *ops = stream(i);
	auto op = static_cast<Op>(i.op);

	if (!options.vulkan_semantics)
		SPIRV_CROSS_THROW("Can only use subgroup operations in Vulkan semantics.");

	// If we need to do implicit bitcasts, make sure we do it with the correct type.
	uint32_t integer_width = get_integer_width_for_instruction(i);
	auto int_type = to_signed_basetype(integer_width);
	auto uint_type = to_unsigned_basetype(integer_width);

	switch (op)
	{
	case OpGroupNonUniformElect:
		require_extension_internal("GL_KHR_shader_subgroup_basic");
		break;

	case OpGroupNonUniformBroadcast:
	case OpGroupNonUniformBroadcastFirst:
	case OpGroupNonUniformBallot:
	case OpGroupNonUniformInverseBallot:
	case OpGroupNonUniformBallotBitExtract:
	case OpGroupNonUniformBallotBitCount:
	case OpGroupNonUniformBallotFindLSB:
	case OpGroupNonUniformBallotFindMSB:
		require_extension_internal("GL_KHR_shader_subgroup_ballot");
		break;

	case OpGroupNonUniformShuffle:
	case OpGroupNonUniformShuffleXor:
		require_extension_internal("GL_KHR_shader_subgroup_shuffle");
		break;

	case OpGroupNonUniformShuffleUp:
	case OpGroupNonUniformShuffleDown:
		require_extension_internal("GL_KHR_shader_subgroup_shuffle_relative");
		break;

	case OpGroupNonUniformAll:
	case OpGroupNonUniformAny:
	case OpGroupNonUniformAllEqual:
		require_extension_internal("GL_KHR_shader_subgroup_vote");
		break;

	case OpGroupNonUniformFAdd:
	case OpGroupNonUniformFMul:
	case OpGroupNonUniformFMin:
	case OpGroupNonUniformFMax:
	case OpGroupNonUniformIAdd:
	case OpGroupNonUniformIMul:
	case OpGroupNonUniformSMin:
	case OpGroupNonUniformSMax:
	case OpGroupNonUniformUMin:
	case OpGroupNonUniformUMax:
	case OpGroupNonUniformBitwiseAnd:
	case OpGroupNonUniformBitwiseOr:
	case OpGroupNonUniformBitwiseXor:
	{
		auto operation = static_cast<GroupOperation>(ops[3]);
		if (operation == GroupOperationClusteredReduce)
		{
			require_extension_internal("GL_KHR_shader_subgroup_clustered");
		}
		else if (operation == GroupOperationExclusiveScan || operation == GroupOperationInclusiveScan ||
		         operation == GroupOperationReduce)
		{
			require_extension_internal("GL_KHR_shader_subgroup_arithmetic");
		}
		else
			SPIRV_CROSS_THROW("Invalid group operation.");
		break;
	}

	case OpGroupNonUniformQuadSwap:
	case OpGroupNonUniformQuadBroadcast:
		require_extension_internal("GL_KHR_shader_subgroup_quad");
		break;

	default:
		SPIRV_CROSS_THROW("Invalid opcode for subgroup.");
	}

	uint32_t result_type = ops[0];
	uint32_t id = ops[1];

	auto scope = static_cast<Scope>(get<SPIRConstant>(ops[2]).scalar());
	if (scope != ScopeSubgroup)
		SPIRV_CROSS_THROW("Only subgroup scope is supported.");

	switch (op)
	{
	case OpGroupNonUniformElect:
		emit_op(result_type, id, "subgroupElect()", true);
		break;

	case OpGroupNonUniformBroadcast:
		emit_binary_func_op(result_type, id, ops[3], ops[4], "subgroupBroadcast");
		break;

	case OpGroupNonUniformBroadcastFirst:
		emit_unary_func_op(result_type, id, ops[3], "subgroupBroadcastFirst");
		break;

	case OpGroupNonUniformBallot:
		emit_unary_func_op(result_type, id, ops[3], "subgroupBallot");
		break;

	case OpGroupNonUniformInverseBallot:
		emit_unary_func_op(result_type, id, ops[3], "subgroupInverseBallot");
		break;

	case OpGroupNonUniformBallotBitExtract:
		emit_binary_func_op(result_type, id, ops[3], ops[4], "subgroupBallotBitExtract");
		break;

	case OpGroupNonUniformBallotFindLSB:
		emit_unary_func_op(result_type, id, ops[3], "subgroupBallotFindLSB");
		break;

	case OpGroupNonUniformBallotFindMSB:
		emit_unary_func_op(result_type, id, ops[3], "subgroupBallotFindMSB");
		break;

	case OpGroupNonUniformBallotBitCount:
	{
		auto operation = static_cast<GroupOperation>(ops[3]);
		if (operation == GroupOperationReduce)
			emit_unary_func_op(result_type, id, ops[4], "subgroupBallotBitCount");
		else if (operation == GroupOperationInclusiveScan)
			emit_unary_func_op(result_type, id, ops[4], "subgroupBallotInclusiveBitCount");
		else if (operation == GroupOperationExclusiveScan)
			emit_unary_func_op(result_type, id, ops[4], "subgroupBallotExclusiveBitCount");
		else
			SPIRV_CROSS_THROW("Invalid BitCount operation.");
		break;
	}

	case OpGroupNonUniformShuffle:
		emit_binary_func_op(result_type, id, ops[3], ops[4], "subgroupShuffle");
		break;

	case OpGroupNonUniformShuffleXor:
		emit_binary_func_op(result_type, id, ops[3], ops[4], "subgroupShuffleXor");
		break;

	case OpGroupNonUniformShuffleUp:
		emit_binary_func_op(result_type, id, ops[3], ops[4], "subgroupShuffleUp");
		break;

	case OpGroupNonUniformShuffleDown:
		emit_binary_func_op(result_type, id, ops[3], ops[4], "subgroupShuffleDown");
		break;

	case OpGroupNonUniformAll:
		emit_unary_func_op(result_type, id, ops[3], "subgroupAll");
		break;

	case OpGroupNonUniformAny:
		emit_unary_func_op(result_type, id, ops[3], "subgroupAny");
		break;

	case OpGroupNonUniformAllEqual:
		emit_unary_func_op(result_type, id, ops[3], "subgroupAllEqual");
		break;

		// clang-format off
#define GLSL_GROUP_OP(op, glsl_op) \
case OpGroupNonUniform##op: \
	{ \
		auto operation = static_cast<GroupOperation>(ops[3]); \
		if (operation == GroupOperationReduce) \
			emit_unary_func_op(result_type, id, ops[4], "subgroup" #glsl_op); \
		else if (operation == GroupOperationInclusiveScan) \
			emit_unary_func_op(result_type, id, ops[4], "subgroupInclusive" #glsl_op); \
		else if (operation == GroupOperationExclusiveScan) \
			emit_unary_func_op(result_type, id, ops[4], "subgroupExclusive" #glsl_op); \
		else if (operation == GroupOperationClusteredReduce) \
			emit_binary_func_op(result_type, id, ops[4], ops[5], "subgroupClustered" #glsl_op); \
		else \
			SPIRV_CROSS_THROW("Invalid group operation."); \
		break; \
	}

#define GLSL_GROUP_OP_CAST(op, glsl_op, type) \
case OpGroupNonUniform##op: \
	{ \
		auto operation = static_cast<GroupOperation>(ops[3]); \
		if (operation == GroupOperationReduce) \
			emit_unary_func_op_cast(result_type, id, ops[4], "subgroup" #glsl_op, type, type); \
		else if (operation == GroupOperationInclusiveScan) \
			emit_unary_func_op_cast(result_type, id, ops[4], "subgroupInclusive" #glsl_op, type, type); \
		else if (operation == GroupOperationExclusiveScan) \
			emit_unary_func_op_cast(result_type, id, ops[4], "subgroupExclusive" #glsl_op, type, type); \
		else if (operation == GroupOperationClusteredReduce) \
			emit_binary_func_op_cast_clustered(result_type, id, ops[4], ops[5], "subgroupClustered" #glsl_op, type); \
		else \
			SPIRV_CROSS_THROW("Invalid group operation."); \
		break; \
	}

	GLSL_GROUP_OP(FAdd, Add)
	GLSL_GROUP_OP(FMul, Mul)
	GLSL_GROUP_OP(FMin, Min)
	GLSL_GROUP_OP(FMax, Max)
	GLSL_GROUP_OP(IAdd, Add)
	GLSL_GROUP_OP(IMul, Mul)
	GLSL_GROUP_OP_CAST(SMin, Min, int_type)
	GLSL_GROUP_OP_CAST(SMax, Max, int_type)
	GLSL_GROUP_OP_CAST(UMin, Min, uint_type)
	GLSL_GROUP_OP_CAST(UMax, Max, uint_type)
	GLSL_GROUP_OP(BitwiseAnd, And)
	GLSL_GROUP_OP(BitwiseOr, Or)
	GLSL_GROUP_OP(BitwiseXor, Xor)
#undef GLSL_GROUP_OP
#undef GLSL_GROUP_OP_CAST
		// clang-format on

	case OpGroupNonUniformQuadSwap:
	{
		uint32_t direction = get<SPIRConstant>(ops[4]).scalar();
		if (direction == 0)
			emit_unary_func_op(result_type, id, ops[3], "subgroupQuadSwapHorizontal");
		else if (direction == 1)
			emit_unary_func_op(result_type, id, ops[3], "subgroupQuadSwapVertical");
		else if (direction == 2)
			emit_unary_func_op(result_type, id, ops[3], "subgroupQuadSwapDiagonal");
		else
			SPIRV_CROSS_THROW("Invalid quad swap direction.");
		break;
	}

	case OpGroupNonUniformQuadBroadcast:
	{
		emit_binary_func_op(result_type, id, ops[3], ops[4], "subgroupQuadBroadcast");
		break;
	}

	default:
		SPIRV_CROSS_THROW("Invalid opcode for subgroup.");
	}

	register_control_dependent_expression(id);
}

string CompilerGLSL::bitcast_glsl_op(const SPIRType &out_type, const SPIRType &in_type)
{
	// OpBitcast can deal with pointers.
	if (out_type.pointer || in_type.pointer)
		return type_to_glsl(out_type);

	if (out_type.basetype == in_type.basetype)
		return "";

	assert(out_type.basetype != SPIRType::Boolean);
	assert(in_type.basetype != SPIRType::Boolean);

	bool integral_cast = type_is_integral(out_type) && type_is_integral(in_type);
	bool same_size_cast = out_type.width == in_type.width;

	// Trivial bitcast case, casts between integers.
	if (integral_cast && same_size_cast)
		return type_to_glsl(out_type);

	// Catch-all 8-bit arithmetic casts (GL_EXT_shader_explicit_arithmetic_types).
	if (out_type.width == 8 && in_type.width >= 16 && integral_cast && in_type.vecsize == 1)
		return "unpack8";
	else if (in_type.width == 8 && out_type.width == 16 && integral_cast && out_type.vecsize == 1)
		return "pack16";
	else if (in_type.width == 8 && out_type.width == 32 && integral_cast && out_type.vecsize == 1)
		return "pack32";

	// Floating <-> Integer special casts. Just have to enumerate all cases. :(
	// 16-bit, 32-bit and 64-bit floats.
	if (out_type.basetype == SPIRType::UInt && in_type.basetype == SPIRType::Float)
	{
		if (is_legacy_es())
			SPIRV_CROSS_THROW("Float -> Uint bitcast not supported on legacy ESSL.");
		else if (!options.es && options.version < 330)
			require_extension_internal("GL_ARB_shader_bit_encoding");
		return "floatBitsToUint";
	}
	else if (out_type.basetype == SPIRType::Int && in_type.basetype == SPIRType::Float)
	{
		if (is_legacy_es())
			SPIRV_CROSS_THROW("Float -> Int bitcast not supported on legacy ESSL.");
		else if (!options.es && options.version < 330)
			require_extension_internal("GL_ARB_shader_bit_encoding");
		return "floatBitsToInt";
	}
	else if (out_type.basetype == SPIRType::Float && in_type.basetype == SPIRType::UInt)
	{
		if (is_legacy_es())
			SPIRV_CROSS_THROW("Uint -> Float bitcast not supported on legacy ESSL.");
		else if (!options.es && options.version < 330)
			require_extension_internal("GL_ARB_shader_bit_encoding");
		return "uintBitsToFloat";
	}
	else if (out_type.basetype == SPIRType::Float && in_type.basetype == SPIRType::Int)
	{
		if (is_legacy_es())
			SPIRV_CROSS_THROW("Int -> Float bitcast not supported on legacy ESSL.");
		else if (!options.es && options.version < 330)
			require_extension_internal("GL_ARB_shader_bit_encoding");
		return "intBitsToFloat";
	}

	else if (out_type.basetype == SPIRType::Int64 && in_type.basetype == SPIRType::Double)
		return "doubleBitsToInt64";
	else if (out_type.basetype == SPIRType::UInt64 && in_type.basetype == SPIRType::Double)
		return "doubleBitsToUint64";
	else if (out_type.basetype == SPIRType::Double && in_type.basetype == SPIRType::Int64)
		return "int64BitsToDouble";
	else if (out_type.basetype == SPIRType::Double && in_type.basetype == SPIRType::UInt64)
		return "uint64BitsToDouble";
	else if (out_type.basetype == SPIRType::Short && in_type.basetype == SPIRType::Half)
		return "float16BitsToInt16";
	else if (out_type.basetype == SPIRType::UShort && in_type.basetype == SPIRType::Half)
		return "float16BitsToUint16";
	else if (out_type.basetype == SPIRType::Half && in_type.basetype == SPIRType::Short)
		return "int16BitsToFloat16";
	else if (out_type.basetype == SPIRType::Half && in_type.basetype == SPIRType::UShort)
		return "uint16BitsToFloat16";

	// And finally, some even more special purpose casts.
	if (out_type.basetype == SPIRType::UInt64 && in_type.basetype == SPIRType::UInt && in_type.vecsize == 2)
		return "packUint2x32";
	else if (out_type.basetype == SPIRType::Half && in_type.basetype == SPIRType::UInt && in_type.vecsize == 1)
		return "unpackFloat2x16";
	else if (out_type.basetype == SPIRType::UInt && in_type.basetype == SPIRType::Half && in_type.vecsize == 2)
		return "packFloat2x16";
	else if (out_type.basetype == SPIRType::Int && in_type.basetype == SPIRType::Short && in_type.vecsize == 2)
		return "packInt2x16";
	else if (out_type.basetype == SPIRType::Short && in_type.basetype == SPIRType::Int && in_type.vecsize == 1)
		return "unpackInt2x16";
	else if (out_type.basetype == SPIRType::UInt && in_type.basetype == SPIRType::UShort && in_type.vecsize == 2)
		return "packUint2x16";
	else if (out_type.basetype == SPIRType::UShort && in_type.basetype == SPIRType::UInt && in_type.vecsize == 1)
		return "unpackUint2x16";
	else if (out_type.basetype == SPIRType::Int64 && in_type.basetype == SPIRType::Short && in_type.vecsize == 4)
		return "packInt4x16";
	else if (out_type.basetype == SPIRType::Short && in_type.basetype == SPIRType::Int64 && in_type.vecsize == 1)
		return "unpackInt4x16";
	else if (out_type.basetype == SPIRType::UInt64 && in_type.basetype == SPIRType::UShort && in_type.vecsize == 4)
		return "packUint4x16";
	else if (out_type.basetype == SPIRType::UShort && in_type.basetype == SPIRType::UInt64 && in_type.vecsize == 1)
		return "unpackUint4x16";

	return "";
}

string CompilerGLSL::bitcast_glsl(const SPIRType &result_type, uint32_t argument)
{
	auto op = bitcast_glsl_op(result_type, expression_type(argument));
	if (op.empty())
		return to_enclosed_unpacked_expression(argument);
	else
		return join(op, "(", to_unpacked_expression(argument), ")");
}

std::string CompilerGLSL::bitcast_expression(SPIRType::BaseType target_type, uint32_t arg)
{
	auto expr = to_expression(arg);
	auto &src_type = expression_type(arg);
	if (src_type.basetype != target_type)
	{
		auto target = src_type;
		target.basetype = target_type;
		expr = join(bitcast_glsl_op(target, src_type), "(", expr, ")");
	}

	return expr;
}

std::string CompilerGLSL::bitcast_expression(const SPIRType &target_type, SPIRType::BaseType expr_type,
                                             const std::string &expr)
{
	if (target_type.basetype == expr_type)
		return expr;

	auto src_type = target_type;
	src_type.basetype = expr_type;
	return join(bitcast_glsl_op(target_type, src_type), "(", expr, ")");
}

string CompilerGLSL::builtin_to_glsl(BuiltIn builtin, StorageClass storage)
{
	switch (builtin)
	{
	case BuiltInPosition:
		return "gl_Position";
	case BuiltInPointSize:
		return "gl_PointSize";
	case BuiltInClipDistance:
		return "gl_ClipDistance";
	case BuiltInCullDistance:
		return "gl_CullDistance";
	case BuiltInVertexId:
		if (options.vulkan_semantics)
			SPIRV_CROSS_THROW(
			    "Cannot implement gl_VertexID in Vulkan GLSL. This shader was created with GL semantics.");
		return "gl_VertexID";
	case BuiltInInstanceId:
		if (options.vulkan_semantics)
			SPIRV_CROSS_THROW(
			    "Cannot implement gl_InstanceID in Vulkan GLSL. This shader was created with GL semantics.");
		return "gl_InstanceID";
	case BuiltInVertexIndex:
		if (options.vulkan_semantics)
			return "gl_VertexIndex";
		else
			return "gl_VertexID"; // gl_VertexID already has the base offset applied.
	case BuiltInInstanceIndex:
		if (options.vulkan_semantics)
			return "gl_InstanceIndex";
		else if (options.vertex.support_nonzero_base_instance)
			return "(gl_InstanceID + SPIRV_Cross_BaseInstance)"; // ... but not gl_InstanceID.
		else
			return "gl_InstanceID";
	case BuiltInPrimitiveId:
		if (storage == StorageClassInput && get_entry_point().model == ExecutionModelGeometry)
			return "gl_PrimitiveIDIn";
		else
			return "gl_PrimitiveID";
	case BuiltInInvocationId:
		return "gl_InvocationID";
	case BuiltInLayer:
		return "gl_Layer";
	case BuiltInViewportIndex:
		return "gl_ViewportIndex";
	case BuiltInTessLevelOuter:
		return "gl_TessLevelOuter";
	case BuiltInTessLevelInner:
		return "gl_TessLevelInner";
	case BuiltInTessCoord:
		return "gl_TessCoord";
	case BuiltInFragCoord:
		return "gl_FragCoord";
	case BuiltInPointCoord:
		return "gl_PointCoord";
	case BuiltInFrontFacing:
		return "gl_FrontFacing";
	case BuiltInFragDepth:
		return "gl_FragDepth";
	case BuiltInNumWorkgroups:
		return "gl_NumWorkGroups";
	case BuiltInWorkgroupSize:
		return "gl_WorkGroupSize";
	case BuiltInWorkgroupId:
		return "gl_WorkGroupID";
	case BuiltInLocalInvocationId:
		return "gl_LocalInvocationID";
	case BuiltInGlobalInvocationId:
		return "gl_GlobalInvocationID";
	case BuiltInLocalInvocationIndex:
		return "gl_LocalInvocationIndex";
	case BuiltInHelperInvocation:
		return "gl_HelperInvocation";
	case BuiltInBaseVertex:
		if (options.es)
			SPIRV_CROSS_THROW("BaseVertex not supported in ES profile.");
		if (options.version < 460)
		{
			require_extension_internal("GL_ARB_shader_draw_parameters");
			return "gl_BaseVertexARB";
		}
		return "gl_BaseVertex";
	case BuiltInBaseInstance:
		if (options.es)
			SPIRV_CROSS_THROW("BaseInstance not supported in ES profile.");
		if (options.version < 460)
		{
			require_extension_internal("GL_ARB_shader_draw_parameters");
			return "gl_BaseInstanceARB";
		}
		return "gl_BaseInstance";
	case BuiltInDrawIndex:
		if (options.es)
			SPIRV_CROSS_THROW("DrawIndex not supported in ES profile.");
		if (options.version < 460)
		{
			require_extension_internal("GL_ARB_shader_draw_parameters");
			return "gl_DrawIDARB";
		}
		return "gl_DrawID";

	case BuiltInSampleId:
		if (options.es && options.version < 320)
			require_extension_internal("GL_OES_sample_variables");
		if (!options.es && options.version < 400)
			SPIRV_CROSS_THROW("gl_SampleID not supported before GLSL 400.");
		return "gl_SampleID";

	case BuiltInSampleMask:
		if (options.es && options.version < 320)
			require_extension_internal("GL_OES_sample_variables");
		if (!options.es && options.version < 400)
			SPIRV_CROSS_THROW("gl_SampleMask/gl_SampleMaskIn not supported before GLSL 400.");

		if (storage == StorageClassInput)
			return "gl_SampleMaskIn";
		else
			return "gl_SampleMask";

	case BuiltInSamplePosition:
		if (options.es && options.version < 320)
			require_extension_internal("GL_OES_sample_variables");
		if (!options.es && options.version < 400)
			SPIRV_CROSS_THROW("gl_SamplePosition not supported before GLSL 400.");
		return "gl_SamplePosition";

	case BuiltInViewIndex:
		if (options.vulkan_semantics)
		{
			require_extension_internal("GL_EXT_multiview");
			return "gl_ViewIndex";
		}
		else
		{
			require_extension_internal("GL_OVR_multiview2");
			return "gl_ViewID_OVR";
		}

	case BuiltInNumSubgroups:
		if (!options.vulkan_semantics)
			SPIRV_CROSS_THROW("Need Vulkan semantics for subgroup.");
		require_extension_internal("GL_KHR_shader_subgroup_basic");
		return "gl_NumSubgroups";

	case BuiltInSubgroupId:
		if (!options.vulkan_semantics)
			SPIRV_CROSS_THROW("Need Vulkan semantics for subgroup.");
		require_extension_internal("GL_KHR_shader_subgroup_basic");
		return "gl_SubgroupID";

	case BuiltInSubgroupSize:
		if (!options.vulkan_semantics)
			SPIRV_CROSS_THROW("Need Vulkan semantics for subgroup.");
		require_extension_internal("GL_KHR_shader_subgroup_basic");
		return "gl_SubgroupSize";

	case BuiltInSubgroupLocalInvocationId:
		if (!options.vulkan_semantics)
			SPIRV_CROSS_THROW("Need Vulkan semantics for subgroup.");
		require_extension_internal("GL_KHR_shader_subgroup_basic");
		return "gl_SubgroupInvocationID";

	case BuiltInSubgroupEqMask:
		if (!options.vulkan_semantics)
			SPIRV_CROSS_THROW("Need Vulkan semantics for subgroup.");
		require_extension_internal("GL_KHR_shader_subgroup_ballot");
		return "gl_SubgroupEqMask";

	case BuiltInSubgroupGeMask:
		if (!options.vulkan_semantics)
			SPIRV_CROSS_THROW("Need Vulkan semantics for subgroup.");
		require_extension_internal("GL_KHR_shader_subgroup_ballot");
		return "gl_SubgroupGeMask";

	case BuiltInSubgroupGtMask:
		if (!options.vulkan_semantics)
			SPIRV_CROSS_THROW("Need Vulkan semantics for subgroup.");
		require_extension_internal("GL_KHR_shader_subgroup_ballot");
		return "gl_SubgroupGtMask";

	case BuiltInSubgroupLeMask:
		if (!options.vulkan_semantics)
			SPIRV_CROSS_THROW("Need Vulkan semantics for subgroup.");
		require_extension_internal("GL_KHR_shader_subgroup_ballot");
		return "gl_SubgroupLeMask";

	case BuiltInSubgroupLtMask:
		if (!options.vulkan_semantics)
			SPIRV_CROSS_THROW("Need Vulkan semantics for subgroup.");
		require_extension_internal("GL_KHR_shader_subgroup_ballot");
		return "gl_SubgroupLtMask";

	case BuiltInLaunchIdNV:
		return "gl_LaunchIDNV";
	case BuiltInLaunchSizeNV:
		return "gl_LaunchSizeNV";
	case BuiltInWorldRayOriginNV:
		return "gl_WorldRayOriginNV";
	case BuiltInWorldRayDirectionNV:
		return "gl_WorldRayDirectionNV";
	case BuiltInObjectRayOriginNV:
		return "gl_ObjectRayOriginNV";
	case BuiltInObjectRayDirectionNV:
		return "gl_ObjectRayDirectionNV";
	case BuiltInRayTminNV:
		return "gl_RayTminNV";
	case BuiltInRayTmaxNV:
		return "gl_RayTmaxNV";
	case BuiltInInstanceCustomIndexNV:
		return "gl_InstanceCustomIndexNV";
	case BuiltInObjectToWorldNV:
		return "gl_ObjectToWorldNV";
	case BuiltInWorldToObjectNV:
		return "gl_WorldToObjectNV";
	case BuiltInHitTNV:
		return "gl_HitTNV";
	case BuiltInHitKindNV:
		return "gl_HitKindNV";
	case BuiltInIncomingRayFlagsNV:
		return "gl_IncomingRayFlagsNV";

	case BuiltInBaryCoordNV:
	{
		if (options.es && options.version < 320)
			SPIRV_CROSS_THROW("gl_BaryCoordNV requires ESSL 320.");
		else if (!options.es && options.version < 450)
			SPIRV_CROSS_THROW("gl_BaryCoordNV requires GLSL 450.");
		require_extension_internal("GL_NV_fragment_shader_barycentric");
		return "gl_BaryCoordNV";
	}

	case BuiltInBaryCoordNoPerspNV:
	{
		if (options.es && options.version < 320)
			SPIRV_CROSS_THROW("gl_BaryCoordNoPerspNV requires ESSL 320.");
		else if (!options.es && options.version < 450)
			SPIRV_CROSS_THROW("gl_BaryCoordNoPerspNV requires GLSL 450.");
		require_extension_internal("GL_NV_fragment_shader_barycentric");
		return "gl_BaryCoordNoPerspNV";
	}

	case BuiltInFragStencilRefEXT:
	{
		if (!options.es)
		{
			require_extension_internal("GL_ARB_shader_stencil_export");
			return "gl_FragStencilRefARB";
		}
		else
			SPIRV_CROSS_THROW("Stencil export not supported in GLES.");
	}

	case BuiltInDeviceIndex:
		if (!options.vulkan_semantics)
			SPIRV_CROSS_THROW("Need Vulkan semantics for device group support.");
		require_extension_internal("GL_EXT_device_group");
		return "gl_DeviceIndex";

	default:
		return join("gl_BuiltIn_", convert_to_string(builtin));
	}
}

const char *CompilerGLSL::index_to_swizzle(uint32_t index)
{
	switch (index)
	{
	case 0:
		return "x";
	case 1:
		return "y";
	case 2:
		return "z";
	case 3:
		return "w";
	default:
		SPIRV_CROSS_THROW("Swizzle index out of range");
	}
}

void CompilerGLSL::access_chain_internal_append_index(std::string &expr, uint32_t /*base*/, const SPIRType *type,
                                                      AccessChainFlags flags, bool & /*access_chain_is_arrayed*/,
                                                      uint32_t index)
{
	bool index_is_literal = (flags & ACCESS_CHAIN_INDEX_IS_LITERAL_BIT) != 0;
	bool register_expression_read = (flags & ACCESS_CHAIN_SKIP_REGISTER_EXPRESSION_READ_BIT) == 0;

	expr += "[";

	// If we are indexing into an array of SSBOs or UBOs, we need to index it with a non-uniform qualifier.
	bool nonuniform_index =
	    has_decoration(index, DecorationNonUniformEXT) &&
	    (has_decoration(type->self, DecorationBlock) || has_decoration(type->self, DecorationBufferBlock));
	if (nonuniform_index)
	{
		expr += backend.nonuniform_qualifier;
		expr += "(";
	}

	if (index_is_literal)
		expr += convert_to_string(index);
	else
		expr += to_expression(index, register_expression_read);

	if (nonuniform_index)
		expr += ")";

	expr += "]";
}

string CompilerGLSL::access_chain_internal(uint32_t base, const uint32_t *indices, uint32_t count,
                                           AccessChainFlags flags, AccessChainMeta *meta)
{
	string expr;

	bool index_is_literal = (flags & ACCESS_CHAIN_INDEX_IS_LITERAL_BIT) != 0;
	bool msb_is_id = (flags & ACCESS_CHAIN_LITERAL_MSB_FORCE_ID) != 0;
	bool chain_only = (flags & ACCESS_CHAIN_CHAIN_ONLY_BIT) != 0;
	bool ptr_chain = (flags & ACCESS_CHAIN_PTR_CHAIN_BIT) != 0;
	bool register_expression_read = (flags & ACCESS_CHAIN_SKIP_REGISTER_EXPRESSION_READ_BIT) == 0;

	if (!chain_only)
	{
		// We handle transpose explicitly, so don't resolve that here.
		auto *e = maybe_get<SPIRExpression>(base);
		bool old_transpose = e && e->need_transpose;
		if (e)
			e->need_transpose = false;
		expr = to_enclosed_expression(base, register_expression_read);
		if (e)
			e->need_transpose = old_transpose;
	}

	// Start traversing type hierarchy at the proper non-pointer types,
	// but keep type_id referencing the original pointer for use below.
	uint32_t type_id = expression_type_id(base);

	if (!backend.native_pointers)
	{
		if (ptr_chain)
			SPIRV_CROSS_THROW("Backend does not support native pointers and does not support OpPtrAccessChain.");

		// Wrapped buffer reference pointer types will need to poke into the internal "value" member before
		// continuing the access chain.
		if (should_dereference(base))
		{
			auto &type = get<SPIRType>(type_id);
			expr = dereference_expression(type, expr);
		}
	}

	const auto *type = &get_pointee_type(type_id);

	bool access_chain_is_arrayed = expr.find_first_of('[') != string::npos;
	bool row_major_matrix_needs_conversion = is_non_native_row_major_matrix(base);
	bool is_packed = has_extended_decoration(base, SPIRVCrossDecorationPhysicalTypePacked);
	uint32_t physical_type = get_extended_decoration(base, SPIRVCrossDecorationPhysicalTypeID);
	bool is_invariant = has_decoration(base, DecorationInvariant);
	bool pending_array_enclose = false;
	bool dimension_flatten = false;

	const auto append_index = [&](uint32_t index, bool is_literal) {
		AccessChainFlags mod_flags = flags;
		if (!is_literal)
			mod_flags &= ~ACCESS_CHAIN_INDEX_IS_LITERAL_BIT;
		access_chain_internal_append_index(expr, base, type, mod_flags, access_chain_is_arrayed, index);
	};

	for (uint32_t i = 0; i < count; i++)
	{
		uint32_t index = indices[i];

		bool is_literal = index_is_literal;
		if (is_literal && msb_is_id && (index >> 31u) != 0u)
		{
			is_literal = false;
			index &= 0x7fffffffu;
		}

		// Pointer chains
		if (ptr_chain && i == 0)
		{
			// If we are flattening multidimensional arrays, only create opening bracket on first
			// array index.
			if (options.flatten_multidimensional_arrays)
			{
				dimension_flatten = type->array.size() >= 1;
				pending_array_enclose = dimension_flatten;
				if (pending_array_enclose)
					expr += "[";
			}

			if (options.flatten_multidimensional_arrays && dimension_flatten)
			{
				// If we are flattening multidimensional arrays, do manual stride computation.
				if (is_literal)
					expr += convert_to_string(index);
				else
					expr += to_enclosed_expression(index, register_expression_read);

				for (auto j = uint32_t(type->array.size()); j; j--)
				{
					expr += " * ";
					expr += enclose_expression(to_array_size(*type, j - 1));
				}

				if (type->array.empty())
					pending_array_enclose = false;
				else
					expr += " + ";

				if (!pending_array_enclose)
					expr += "]";
			}
			else
			{
				append_index(index, is_literal);
			}

			if (type->basetype == SPIRType::ControlPointArray)
			{
				type_id = type->parent_type;
				type = &get<SPIRType>(type_id);
			}

			access_chain_is_arrayed = true;
		}
		// Arrays
		else if (!type->array.empty())
		{
			// If we are flattening multidimensional arrays, only create opening bracket on first
			// array index.
			if (options.flatten_multidimensional_arrays && !pending_array_enclose)
			{
				dimension_flatten = type->array.size() > 1;
				pending_array_enclose = dimension_flatten;
				if (pending_array_enclose)
					expr += "[";
			}

			assert(type->parent_type);

			auto *var = maybe_get<SPIRVariable>(base);
			if (backend.force_gl_in_out_block && i == 0 && var && is_builtin_variable(*var) &&
			    !has_decoration(type->self, DecorationBlock))
			{
				// This deals with scenarios for tesc/geom where arrays of gl_Position[] are declared.
				// Normally, these variables live in blocks when compiled from GLSL,
				// but HLSL seems to just emit straight arrays here.
				// We must pretend this access goes through gl_in/gl_out arrays
				// to be able to access certain builtins as arrays.
				auto builtin = ir.meta[base].decoration.builtin_type;
				switch (builtin)
				{
				// case BuiltInCullDistance: // These are already arrays, need to figure out rules for these in tess/geom.
				// case BuiltInClipDistance:
				case BuiltInPosition:
				case BuiltInPointSize:
					if (var->storage == StorageClassInput)
						expr = join("gl_in[", to_expression(index, register_expression_read), "].", expr);
					else if (var->storage == StorageClassOutput)
						expr = join("gl_out[", to_expression(index, register_expression_read), "].", expr);
					else
						append_index(index, is_literal);
					break;

				default:
					append_index(index, is_literal);
					break;
				}
			}
			else if (options.flatten_multidimensional_arrays && dimension_flatten)
			{
				// If we are flattening multidimensional arrays, do manual stride computation.
				auto &parent_type = get<SPIRType>(type->parent_type);

				if (is_literal)
					expr += convert_to_string(index);
				else
					expr += to_enclosed_expression(index, register_expression_read);

				for (auto j = uint32_t(parent_type.array.size()); j; j--)
				{
					expr += " * ";
					expr += enclose_expression(to_array_size(parent_type, j - 1));
				}

				if (parent_type.array.empty())
					pending_array_enclose = false;
				else
					expr += " + ";

				if (!pending_array_enclose)
					expr += "]";
			}
			// Some builtins are arrays in SPIR-V but not in other languages, e.g. gl_SampleMask[] is an array in SPIR-V but not in Metal.
			// By throwing away the index, we imply the index was 0, which it must be for gl_SampleMask.
			else if (!builtin_translates_to_nonarray(BuiltIn(get_decoration(base, DecorationBuiltIn))))
			{
				append_index(index, is_literal);
			}

			type_id = type->parent_type;
			type = &get<SPIRType>(type_id);

			access_chain_is_arrayed = true;
		}
		// For structs, the index refers to a constant, which indexes into the members.
		// We also check if this member is a builtin, since we then replace the entire expression with the builtin one.
		else if (type->basetype == SPIRType::Struct)
		{
			if (!is_literal)
				index = get<SPIRConstant>(index).scalar();

			if (index >= type->member_types.size())
				SPIRV_CROSS_THROW("Member index is out of bounds!");

			BuiltIn builtin;
			if (is_member_builtin(*type, index, &builtin))
			{
				if (access_chain_is_arrayed)
				{
					expr += ".";
					expr += builtin_to_glsl(builtin, type->storage);
				}
				else
					expr = builtin_to_glsl(builtin, type->storage);
			}
			else
			{
				// If the member has a qualified name, use it as the entire chain
				string qual_mbr_name = get_member_qualified_name(type_id, index);
				if (!qual_mbr_name.empty())
					expr = qual_mbr_name;
				else
					expr += to_member_reference(base, *type, index, ptr_chain);
			}

			if (has_member_decoration(type->self, index, DecorationInvariant))
				is_invariant = true;

			is_packed = member_is_packed_physical_type(*type, index);
			if (member_is_remapped_physical_type(*type, index))
				physical_type = get_extended_member_decoration(type->self, index, SPIRVCrossDecorationPhysicalTypeID);
			else
				physical_type = 0;

			row_major_matrix_needs_conversion = member_is_non_native_row_major_matrix(*type, index);
			type = &get<SPIRType>(type->member_types[index]);
		}
		// Matrix -> Vector
		else if (type->columns > 1)
		{
			// If we have a row-major matrix here, we need to defer any transpose in case this access chain
			// is used to store a column. We can resolve it right here and now if we access a scalar directly,
			// by flipping indexing order of the matrix.

			expr += "[";
			if (is_literal)
				expr += convert_to_string(index);
			else
				expr += to_expression(index, register_expression_read);
			expr += "]";

			type_id = type->parent_type;
			type = &get<SPIRType>(type_id);
		}
		// Vector -> Scalar
		else if (type->vecsize > 1)
		{
			string deferred_index;
			if (row_major_matrix_needs_conversion)
			{
				// Flip indexing order.
				auto column_index = expr.find_last_of('[');
				if (column_index != string::npos)
				{
					deferred_index = expr.substr(column_index);
					expr.resize(column_index);
				}
			}

			if (is_literal && !is_packed && !row_major_matrix_needs_conversion)
			{
				expr += ".";
				expr += index_to_swizzle(index);
			}
			else if (ir.ids[index].get_type() == TypeConstant && !is_packed && !row_major_matrix_needs_conversion)
			{
				auto &c = get<SPIRConstant>(index);
				if (c.specialization)
				{
					// If the index is a spec constant, we cannot turn extract into a swizzle.
					expr += join("[", to_expression(index), "]");
				}
				else
				{
					expr += ".";
					expr += index_to_swizzle(c.scalar());
				}
			}
			else if (is_literal)
			{
				// For packed vectors, we can only access them as an array, not by swizzle.
				expr += join("[", index, "]");
			}
			else
			{
				expr += "[";
				expr += to_expression(index, register_expression_read);
				expr += "]";
			}

			expr += deferred_index;
			row_major_matrix_needs_conversion = false;

			is_packed = false;
			physical_type = 0;
			type_id = type->parent_type;
			type = &get<SPIRType>(type_id);
		}
		else if (!backend.allow_truncated_access_chain)
			SPIRV_CROSS_THROW("Cannot subdivide a scalar value!");
	}

	if (pending_array_enclose)
	{
		SPIRV_CROSS_THROW("Flattening of multidimensional arrays were enabled, "
		                  "but the access chain was terminated in the middle of a multidimensional array. "
		                  "This is not supported.");
	}

	if (meta)
	{
		meta->need_transpose = row_major_matrix_needs_conversion;
		meta->storage_is_packed = is_packed;
		meta->storage_is_invariant = is_invariant;
		meta->storage_physical_type = physical_type;
	}

	return expr;
}

string CompilerGLSL::to_flattened_struct_member(const SPIRVariable &var, uint32_t index)
{
	auto &type = get<SPIRType>(var.basetype);
	return sanitize_underscores(join(to_name(var.self), "_", to_member_name(type, index)));
}

string CompilerGLSL::access_chain(uint32_t base, const uint32_t *indices, uint32_t count, const SPIRType &target_type,
                                  AccessChainMeta *meta, bool ptr_chain)
{
	if (flattened_buffer_blocks.count(base))
	{
		uint32_t matrix_stride = 0;
		uint32_t array_stride = 0;
		bool need_transpose = false;
		flattened_access_chain_offset(expression_type(base), indices, count, 0, 16, &need_transpose, &matrix_stride,
		                              &array_stride, ptr_chain);

		if (meta)
		{
			meta->need_transpose = target_type.columns > 1 && need_transpose;
			meta->storage_is_packed = false;
		}

		return flattened_access_chain(base, indices, count, target_type, 0, matrix_stride, array_stride,
		                              need_transpose);
	}
	else if (flattened_structs.count(base) && count > 0)
	{
		AccessChainFlags flags = ACCESS_CHAIN_CHAIN_ONLY_BIT | ACCESS_CHAIN_SKIP_REGISTER_EXPRESSION_READ_BIT;
		if (ptr_chain)
			flags |= ACCESS_CHAIN_PTR_CHAIN_BIT;

		auto chain = access_chain_internal(base, indices, count, flags, nullptr).substr(1);
		if (meta)
		{
			meta->need_transpose = false;
			meta->storage_is_packed = false;
		}
		return sanitize_underscores(join(to_name(base), "_", chain));
	}
	else
	{
		AccessChainFlags flags = ACCESS_CHAIN_SKIP_REGISTER_EXPRESSION_READ_BIT;
		if (ptr_chain)
			flags |= ACCESS_CHAIN_PTR_CHAIN_BIT;
		return access_chain_internal(base, indices, count, flags, meta);
	}
}

string CompilerGLSL::load_flattened_struct(SPIRVariable &var)
{
	auto expr = type_to_glsl_constructor(get<SPIRType>(var.basetype));
	expr += '(';

	auto &type = get<SPIRType>(var.basetype);
	for (uint32_t i = 0; i < uint32_t(type.member_types.size()); i++)
	{
		if (i)
			expr += ", ";

		// Flatten the varyings.
		// Apply name transformation for flattened I/O blocks.
		expr += to_flattened_struct_member(var, i);
	}
	expr += ')';
	return expr;
}

void CompilerGLSL::store_flattened_struct(SPIRVariable &var, uint32_t value)
{
	// We're trying to store a structure which has been flattened.
	// Need to copy members one by one.
	auto rhs = to_expression(value);

	// Store result locally.
	// Since we're declaring a variable potentially multiple times here,
	// store the variable in an isolated scope.
	begin_scope();
	statement(variable_decl_function_local(var), " = ", rhs, ";");

	auto &type = get<SPIRType>(var.basetype);
	for (uint32_t i = 0; i < uint32_t(type.member_types.size()); i++)
	{
		// Flatten the varyings.
		// Apply name transformation for flattened I/O blocks.

		auto lhs = sanitize_underscores(join(to_name(var.self), "_", to_member_name(type, i)));
		rhs = join(to_name(var.self), ".", to_member_name(type, i));
		statement(lhs, " = ", rhs, ";");
	}
	end_scope();
}

std::string CompilerGLSL::flattened_access_chain(uint32_t base, const uint32_t *indices, uint32_t count,
                                                 const SPIRType &target_type, uint32_t offset, uint32_t matrix_stride,
                                                 uint32_t /* array_stride */, bool need_transpose)
{
	if (!target_type.array.empty())
		SPIRV_CROSS_THROW("Access chains that result in an array can not be flattened");
	else if (target_type.basetype == SPIRType::Struct)
		return flattened_access_chain_struct(base, indices, count, target_type, offset);
	else if (target_type.columns > 1)
		return flattened_access_chain_matrix(base, indices, count, target_type, offset, matrix_stride, need_transpose);
	else
		return flattened_access_chain_vector(base, indices, count, target_type, offset, matrix_stride, need_transpose);
}

std::string CompilerGLSL::flattened_access_chain_struct(uint32_t base, const uint32_t *indices, uint32_t count,
                                                        const SPIRType &target_type, uint32_t offset)
{
	std::string expr;

	expr += type_to_glsl_constructor(target_type);
	expr += "(";

	for (uint32_t i = 0; i < uint32_t(target_type.member_types.size()); ++i)
	{
		if (i != 0)
			expr += ", ";

		const SPIRType &member_type = get<SPIRType>(target_type.member_types[i]);
		uint32_t member_offset = type_struct_member_offset(target_type, i);

		// The access chain terminates at the struct, so we need to find matrix strides and row-major information
		// ahead of time.
		bool need_transpose = false;
		uint32_t matrix_stride = 0;
		if (member_type.columns > 1)
		{
			need_transpose = combined_decoration_for_member(target_type, i).get(DecorationRowMajor);
			matrix_stride = type_struct_member_matrix_stride(target_type, i);
		}

		auto tmp = flattened_access_chain(base, indices, count, member_type, offset + member_offset, matrix_stride,
		                                  0 /* array_stride */, need_transpose);

		// Cannot forward transpositions, so resolve them here.
		if (need_transpose)
			expr += convert_row_major_matrix(tmp, member_type, 0, false);
		else
			expr += tmp;
	}

	expr += ")";

	return expr;
}

std::string CompilerGLSL::flattened_access_chain_matrix(uint32_t base, const uint32_t *indices, uint32_t count,
                                                        const SPIRType &target_type, uint32_t offset,
                                                        uint32_t matrix_stride, bool need_transpose)
{
	assert(matrix_stride);
	SPIRType tmp_type = target_type;
	if (need_transpose)
		swap(tmp_type.vecsize, tmp_type.columns);

	std::string expr;

	expr += type_to_glsl_constructor(tmp_type);
	expr += "(";

	for (uint32_t i = 0; i < tmp_type.columns; i++)
	{
		if (i != 0)
			expr += ", ";

		expr += flattened_access_chain_vector(base, indices, count, tmp_type, offset + i * matrix_stride, matrix_stride,
		                                      /* need_transpose= */ false);
	}

	expr += ")";

	return expr;
}

std::string CompilerGLSL::flattened_access_chain_vector(uint32_t base, const uint32_t *indices, uint32_t count,
                                                        const SPIRType &target_type, uint32_t offset,
                                                        uint32_t matrix_stride, bool need_transpose)
{
	auto result = flattened_access_chain_offset(expression_type(base), indices, count, offset, 16);

	auto buffer_name = to_name(expression_type(base).self);

	if (need_transpose)
	{
		std::string expr;

		if (target_type.vecsize > 1)
		{
			expr += type_to_glsl_constructor(target_type);
			expr += "(";
		}

		for (uint32_t i = 0; i < target_type.vecsize; ++i)
		{
			if (i != 0)
				expr += ", ";

			uint32_t component_offset = result.second + i * matrix_stride;

			assert(component_offset % (target_type.width / 8) == 0);
			uint32_t index = component_offset / (target_type.width / 8);

			expr += buffer_name;
			expr += "[";
			expr += result.first; // this is a series of N1 * k1 + N2 * k2 + ... that is either empty or ends with a +
			expr += convert_to_string(index / 4);
			expr += "]";

			expr += vector_swizzle(1, index % 4);
		}

		if (target_type.vecsize > 1)
		{
			expr += ")";
		}

		return expr;
	}
	else
	{
		assert(result.second % (target_type.width / 8) == 0);
		uint32_t index = result.second / (target_type.width / 8);

		std::string expr;

		expr += buffer_name;
		expr += "[";
		expr += result.first; // this is a series of N1 * k1 + N2 * k2 + ... that is either empty or ends with a +
		expr += convert_to_string(index / 4);
		expr += "]";

		expr += vector_swizzle(target_type.vecsize, index % 4);

		return expr;
	}
}

std::pair<std::string, uint32_t> CompilerGLSL::flattened_access_chain_offset(
    const SPIRType &basetype, const uint32_t *indices, uint32_t count, uint32_t offset, uint32_t word_stride,
    bool *need_transpose, uint32_t *out_matrix_stride, uint32_t *out_array_stride, bool ptr_chain)
{
	// Start traversing type hierarchy at the proper non-pointer types.
	const auto *type = &get_pointee_type(basetype);

	std::string expr;

	// Inherit matrix information in case we are access chaining a vector which might have come from a row major layout.
	bool row_major_matrix_needs_conversion = need_transpose ? *need_transpose : false;
	uint32_t matrix_stride = out_matrix_stride ? *out_matrix_stride : 0;
	uint32_t array_stride = out_array_stride ? *out_array_stride : 0;

	for (uint32_t i = 0; i < count; i++)
	{
		uint32_t index = indices[i];

		// Pointers
		if (ptr_chain && i == 0)
		{
			// Here, the pointer type will be decorated with an array stride.
			array_stride = get_decoration(basetype.self, DecorationArrayStride);
			if (!array_stride)
				SPIRV_CROSS_THROW("SPIR-V does not define ArrayStride for buffer block.");

			auto *constant = maybe_get<SPIRConstant>(index);
			if (constant)
			{
				// Constant array access.
				offset += constant->scalar() * array_stride;
			}
			else
			{
				// Dynamic array access.
				if (array_stride % word_stride)
				{
					SPIRV_CROSS_THROW(
					    "Array stride for dynamic indexing must be divisible by the size of a 4-component vector. "
					    "Likely culprit here is a float or vec2 array inside a push constant block which is std430. "
					    "This cannot be flattened. Try using std140 layout instead.");
				}

				expr += to_enclosed_expression(index);
				expr += " * ";
				expr += convert_to_string(array_stride / word_stride);
				expr += " + ";
			}
		}
		// Arrays
		else if (!type->array.empty())
		{
			auto *constant = maybe_get<SPIRConstant>(index);
			if (constant)
			{
				// Constant array access.
				offset += constant->scalar() * array_stride;
			}
			else
			{
				// Dynamic array access.
				if (array_stride % word_stride)
				{
					SPIRV_CROSS_THROW(
					    "Array stride for dynamic indexing must be divisible by the size of a 4-component vector. "
					    "Likely culprit here is a float or vec2 array inside a push constant block which is std430. "
					    "This cannot be flattened. Try using std140 layout instead.");
				}

				expr += to_enclosed_expression(index, false);
				expr += " * ";
				expr += convert_to_string(array_stride / word_stride);
				expr += " + ";
			}

			uint32_t parent_type = type->parent_type;
			type = &get<SPIRType>(parent_type);

			if (!type->array.empty())
				array_stride = get_decoration(parent_type, DecorationArrayStride);
		}
		// For structs, the index refers to a constant, which indexes into the members.
		// We also check if this member is a builtin, since we then replace the entire expression with the builtin one.
		else if (type->basetype == SPIRType::Struct)
		{
			index = get<SPIRConstant>(index).scalar();

			if (index >= type->member_types.size())
				SPIRV_CROSS_THROW("Member index is out of bounds!");

			offset += type_struct_member_offset(*type, index);

			auto &struct_type = *type;
			type = &get<SPIRType>(type->member_types[index]);

			if (type->columns > 1)
			{
				matrix_stride = type_struct_member_matrix_stride(struct_type, index);
				row_major_matrix_needs_conversion =
				    combined_decoration_for_member(struct_type, index).get(DecorationRowMajor);
			}
			else
				row_major_matrix_needs_conversion = false;

			if (!type->array.empty())
				array_stride = type_struct_member_array_stride(struct_type, index);
		}
		// Matrix -> Vector
		else if (type->columns > 1)
		{
			auto *constant = maybe_get<SPIRConstant>(index);
			if (constant)
			{
				index = get<SPIRConstant>(index).scalar();
				offset += index * (row_major_matrix_needs_conversion ? (type->width / 8) : matrix_stride);
			}
			else
			{
				uint32_t indexing_stride = row_major_matrix_needs_conversion ? (type->width / 8) : matrix_stride;
				// Dynamic array access.
				if (indexing_stride % word_stride)
				{
					SPIRV_CROSS_THROW(
					    "Matrix stride for dynamic indexing must be divisible by the size of a 4-component vector. "
					    "Likely culprit here is a row-major matrix being accessed dynamically. "
					    "This cannot be flattened. Try using std140 layout instead.");
				}

				expr += to_enclosed_expression(index, false);
				expr += " * ";
				expr += convert_to_string(indexing_stride / word_stride);
				expr += " + ";
			}

			type = &get<SPIRType>(type->parent_type);
		}
		// Vector -> Scalar
		else if (type->vecsize > 1)
		{
			auto *constant = maybe_get<SPIRConstant>(index);
			if (constant)
			{
				index = get<SPIRConstant>(index).scalar();
				offset += index * (row_major_matrix_needs_conversion ? matrix_stride : (type->width / 8));
			}
			else
			{
				uint32_t indexing_stride = row_major_matrix_needs_conversion ? matrix_stride : (type->width / 8);

				// Dynamic array access.
				if (indexing_stride % word_stride)
				{
					SPIRV_CROSS_THROW(
					    "Stride for dynamic vector indexing must be divisible by the size of a 4-component vector. "
					    "This cannot be flattened in legacy targets.");
				}

				expr += to_enclosed_expression(index, false);
				expr += " * ";
				expr += convert_to_string(indexing_stride / word_stride);
				expr += " + ";
			}

			type = &get<SPIRType>(type->parent_type);
		}
		else
			SPIRV_CROSS_THROW("Cannot subdivide a scalar value!");
	}

	if (need_transpose)
		*need_transpose = row_major_matrix_needs_conversion;
	if (out_matrix_stride)
		*out_matrix_stride = matrix_stride;
	if (out_array_stride)
		*out_array_stride = array_stride;

	return std::make_pair(expr, offset);
}

bool CompilerGLSL::should_dereference(uint32_t id)
{
	const auto &type = expression_type(id);
	// Non-pointer expressions don't need to be dereferenced.
	if (!type.pointer)
		return false;

	// Handles shouldn't be dereferenced either.
	if (!expression_is_lvalue(id))
		return false;

	// If id is a variable but not a phi variable, we should not dereference it.
	if (auto *var = maybe_get<SPIRVariable>(id))
		return var->phi_variable;

	// If id is an access chain, we should not dereference it.
	if (auto *expr = maybe_get<SPIRExpression>(id))
		return !expr->access_chain;

	// Otherwise, we should dereference this pointer expression.
	return true;
}

bool CompilerGLSL::should_forward(uint32_t id) const
{
	// If id is a variable we will try to forward it regardless of force_temporary check below
	// This is important because otherwise we'll get local sampler copies (highp sampler2D foo = bar) that are invalid in OpenGL GLSL
	auto *var = maybe_get<SPIRVariable>(id);
	if (var && var->forwardable)
		return true;

	// For debugging emit temporary variables for all expressions
	if (options.force_temporary)
		return false;

	// Immutable expression can always be forwarded.
	if (is_immutable(id))
		return true;

	return false;
}

bool CompilerGLSL::should_suppress_usage_tracking(uint32_t id) const
{
	// Used only by opcodes which don't do any real "work", they just swizzle data in some fashion.
	return !expression_is_forwarded(id) || expression_suppresses_usage_tracking(id);
}

void CompilerGLSL::track_expression_read(uint32_t id)
{
	switch (ir.ids[id].get_type())
	{
	case TypeExpression:
	{
		auto &e = get<SPIRExpression>(id);
		for (auto implied_read : e.implied_read_expressions)
			track_expression_read(implied_read);
		break;
	}

	case TypeAccessChain:
	{
		auto &e = get<SPIRAccessChain>(id);
		for (auto implied_read : e.implied_read_expressions)
			track_expression_read(implied_read);
		break;
	}

	default:
		break;
	}

	// If we try to read a forwarded temporary more than once we will stamp out possibly complex code twice.
	// In this case, it's better to just bind the complex expression to the temporary and read that temporary twice.
	if (expression_is_forwarded(id) && !expression_suppresses_usage_tracking(id))
	{
		auto &v = expression_usage_counts[id];
		v++;

		if (v >= 2)
		{
			//if (v == 2)
			//    fprintf(stderr, "ID %u was forced to temporary due to more than 1 expression use!\n", id);

			forced_temporaries.insert(id);
			// Force a recompile after this pass to avoid forwarding this variable.
			force_recompile();
		}
	}
}

bool CompilerGLSL::args_will_forward(uint32_t id, const uint32_t *args, uint32_t num_args, bool pure)
{
	if (forced_temporaries.find(id) != end(forced_temporaries))
		return false;

	for (uint32_t i = 0; i < num_args; i++)
		if (!should_forward(args[i]))
			return false;

	// We need to forward globals as well.
	if (!pure)
	{
		for (auto global : global_variables)
			if (!should_forward(global))
				return false;
		for (auto aliased : aliased_variables)
			if (!should_forward(aliased))
				return false;
	}

	return true;
}

void CompilerGLSL::register_impure_function_call()
{
	// Impure functions can modify globals and aliased variables, so invalidate them as well.
	for (auto global : global_variables)
		flush_dependees(get<SPIRVariable>(global));
	for (auto aliased : aliased_variables)
		flush_dependees(get<SPIRVariable>(aliased));
}

void CompilerGLSL::register_call_out_argument(uint32_t id)
{
	register_write(id);

	auto *var = maybe_get<SPIRVariable>(id);
	if (var)
		flush_variable_declaration(var->self);
}

string CompilerGLSL::variable_decl_function_local(SPIRVariable &var)
{
	// These variables are always function local,
	// so make sure we emit the variable without storage qualifiers.
	// Some backends will inject custom variables locally in a function
	// with a storage qualifier which is not function-local.
	auto old_storage = var.storage;
	var.storage = StorageClassFunction;
	auto expr = variable_decl(var);
	var.storage = old_storage;
	return expr;
}

void CompilerGLSL::emit_variable_temporary_copies(const SPIRVariable &var)
{
	// Ensure that we declare phi-variable copies even if the original declaration isn't deferred
	if (var.allocate_temporary_copy && !flushed_phi_variables.count(var.self))
	{
		auto &type = get<SPIRType>(var.basetype);
		auto &flags = get_decoration_bitset(var.self);
		statement(flags_to_qualifiers_glsl(type, flags), variable_decl(type, join("_", var.self, "_copy")), ";");
		flushed_phi_variables.insert(var.self);
	}
}

void CompilerGLSL::flush_variable_declaration(uint32_t id)
{
	// Ensure that we declare phi-variable copies even if the original declaration isn't deferred
	auto *var = maybe_get<SPIRVariable>(id);
	if (var && var->deferred_declaration)
	{
		string initializer;
		if (options.force_zero_initialized_variables &&
		    (var->storage == StorageClassFunction || var->storage == StorageClassGeneric ||
		     var->storage == StorageClassPrivate) &&
		    !var->initializer && type_can_zero_initialize(get_variable_data_type(*var)))
		{
			initializer = join(" = ", to_zero_initialized_expression(get_variable_data_type_id(*var)));
		}

		statement(variable_decl_function_local(*var), initializer, ";");
		var->deferred_declaration = false;
	}
	if (var)
	{
		emit_variable_temporary_copies(*var);
	}
}

bool CompilerGLSL::remove_duplicate_swizzle(string &op)
{
	auto pos = op.find_last_of('.');
	if (pos == string::npos || pos == 0)
		return false;

	string final_swiz = op.substr(pos + 1, string::npos);

	if (backend.swizzle_is_function)
	{
		if (final_swiz.size() < 2)
			return false;

		if (final_swiz.substr(final_swiz.size() - 2, string::npos) == "()")
			final_swiz.erase(final_swiz.size() - 2, string::npos);
		else
			return false;
	}

	// Check if final swizzle is of form .x, .xy, .xyz, .xyzw or similar.
	// If so, and previous swizzle is of same length,
	// we can drop the final swizzle altogether.
	for (uint32_t i = 0; i < final_swiz.size(); i++)
	{
		static const char expected[] = { 'x', 'y', 'z', 'w' };
		if (i >= 4 || final_swiz[i] != expected[i])
			return false;
	}

	auto prevpos = op.find_last_of('.', pos - 1);
	if (prevpos == string::npos)
		return false;

	prevpos++;

	// Make sure there are only swizzles here ...
	for (auto i = prevpos; i < pos; i++)
	{
		if (op[i] < 'w' || op[i] > 'z')
		{
			// If swizzles are foo.xyz() like in C++ backend for example, check for that.
			if (backend.swizzle_is_function && i + 2 == pos && op[i] == '(' && op[i + 1] == ')')
				break;
			return false;
		}
	}

	// If original swizzle is large enough, just carve out the components we need.
	// E.g. foobar.wyx.xy will turn into foobar.wy.
	if (pos - prevpos >= final_swiz.size())
	{
		op.erase(prevpos + final_swiz.size(), string::npos);

		// Add back the function call ...
		if (backend.swizzle_is_function)
			op += "()";
	}
	return true;
}

// Optimizes away vector swizzles where we have something like
// vec3 foo;
// foo.xyz <-- swizzle expression does nothing.
// This is a very common pattern after OpCompositeCombine.
bool CompilerGLSL::remove_unity_swizzle(uint32_t base, string &op)
{
	auto pos = op.find_last_of('.');
	if (pos == string::npos || pos == 0)
		return false;

	string final_swiz = op.substr(pos + 1, string::npos);

	if (backend.swizzle_is_function)
	{
		if (final_swiz.size() < 2)
			return false;

		if (final_swiz.substr(final_swiz.size() - 2, string::npos) == "()")
			final_swiz.erase(final_swiz.size() - 2, string::npos);
		else
			return false;
	}

	// Check if final swizzle is of form .x, .xy, .xyz, .xyzw or similar.
	// If so, and previous swizzle is of same length,
	// we can drop the final swizzle altogether.
	for (uint32_t i = 0; i < final_swiz.size(); i++)
	{
		static const char expected[] = { 'x', 'y', 'z', 'w' };
		if (i >= 4 || final_swiz[i] != expected[i])
			return false;
	}

	auto &type = expression_type(base);

	// Sanity checking ...
	assert(type.columns == 1 && type.array.empty());

	if (type.vecsize == final_swiz.size())
		op.erase(pos, string::npos);
	return true;
}

string CompilerGLSL::build_composite_combiner(uint32_t return_type, const uint32_t *elems, uint32_t length)
{
	ID base = 0;
	string op;
	string subop;

	// Can only merge swizzles for vectors.
	auto &type = get<SPIRType>(return_type);
	bool can_apply_swizzle_opt = type.basetype != SPIRType::Struct && type.array.empty() && type.columns == 1;
	bool swizzle_optimization = false;

	for (uint32_t i = 0; i < length; i++)
	{
		auto *e = maybe_get<SPIRExpression>(elems[i]);

		// If we're merging another scalar which belongs to the same base
		// object, just merge the swizzles to avoid triggering more than 1 expression read as much as possible!
		if (can_apply_swizzle_opt && e && e->base_expression && e->base_expression == base)
		{
			// Only supposed to be used for vector swizzle -> scalar.
			assert(!e->expression.empty() && e->expression.front() == '.');
			subop += e->expression.substr(1, string::npos);
			swizzle_optimization = true;
		}
		else
		{
			// We'll likely end up with duplicated swizzles, e.g.
			// foobar.xyz.xyz from patterns like
			// OpVectorShuffle
			// OpCompositeExtract x 3
			// OpCompositeConstruct 3x + other scalar.
			// Just modify op in-place.
			if (swizzle_optimization)
			{
				if (backend.swizzle_is_function)
					subop += "()";

				// Don't attempt to remove unity swizzling if we managed to remove duplicate swizzles.
				// The base "foo" might be vec4, while foo.xyz is vec3 (OpVectorShuffle) and looks like a vec3 due to the .xyz tacked on.
				// We only want to remove the swizzles if we're certain that the resulting base will be the same vecsize.
				// Essentially, we can only remove one set of swizzles, since that's what we have control over ...
				// Case 1:
				//  foo.yxz.xyz: Duplicate swizzle kicks in, giving foo.yxz, we are done.
				//               foo.yxz was the result of OpVectorShuffle and we don't know the type of foo.
				// Case 2:
				//  foo.xyz: Duplicate swizzle won't kick in.
				//           If foo is vec3, we can remove xyz, giving just foo.
				if (!remove_duplicate_swizzle(subop))
					remove_unity_swizzle(base, subop);

				// Strips away redundant parens if we created them during component extraction.
				strip_enclosed_expression(subop);
				swizzle_optimization = false;
				op += subop;
			}
			else
				op += subop;

			if (i)
				op += ", ";
			subop = to_composite_constructor_expression(elems[i]);
		}

		base = e ? e->base_expression : ID(0);
	}

	if (swizzle_optimization)
	{
		if (backend.swizzle_is_function)
			subop += "()";

		if (!remove_duplicate_swizzle(subop))
			remove_unity_swizzle(base, subop);
		// Strips away redundant parens if we created them during component extraction.
		strip_enclosed_expression(subop);
	}

	op += subop;
	return op;
}

bool CompilerGLSL::skip_argument(uint32_t id) const
{
	if (!combined_image_samplers.empty() || !options.vulkan_semantics)
	{
		auto &type = expression_type(id);
		if (type.basetype == SPIRType::Sampler || (type.basetype == SPIRType::Image && type.image.sampled == 1))
			return true;
	}
	return false;
}

bool CompilerGLSL::optimize_read_modify_write(const SPIRType &type, const string &lhs, const string &rhs)
{
	// Do this with strings because we have a very clear pattern we can check for and it avoids
	// adding lots of special cases to the code emission.
	if (rhs.size() < lhs.size() + 3)
		return false;

	// Do not optimize matrices. They are a bit awkward to reason about in general
	// (in which order does operation happen?), and it does not work on MSL anyways.
	if (type.vecsize > 1 && type.columns > 1)
		return false;

	auto index = rhs.find(lhs);
	if (index != 0)
		return false;

	// TODO: Shift operators, but it's not important for now.
	auto op = rhs.find_first_of("+-/*%|&^", lhs.size() + 1);
	if (op != lhs.size() + 1)
		return false;

	// Check that the op is followed by space. This excludes && and ||.
	if (rhs[op + 1] != ' ')
		return false;

	char bop = rhs[op];
	auto expr = rhs.substr(lhs.size() + 3);
	// Try to find increments and decrements. Makes it look neater as += 1, -= 1 is fairly rare to see in real code.
	// Find some common patterns which are equivalent.
	if ((bop == '+' || bop == '-') && (expr == "1" || expr == "uint(1)" || expr == "1u" || expr == "int(1u)"))
		statement(lhs, bop, bop, ";");
	else
		statement(lhs, " ", bop, "= ", expr, ";");
	return true;
}

void CompilerGLSL::register_control_dependent_expression(uint32_t expr)
{
	if (forwarded_temporaries.find(expr) == end(forwarded_temporaries))
		return;

	assert(current_emitting_block);
	current_emitting_block->invalidate_expressions.push_back(expr);
}

void CompilerGLSL::emit_block_instructions(SPIRBlock &block)
{
	current_emitting_block = &block;
	for (auto &op : block.ops)
		emit_instruction(op);
	current_emitting_block = nullptr;
}

void CompilerGLSL::disallow_forwarding_in_expression_chain(const SPIRExpression &expr)
{
	// Allow trivially forwarded expressions like OpLoad or trivial shuffles,
	// these will be marked as having suppressed usage tracking.
	// Our only concern is to make sure arithmetic operations are done in similar ways.
	if (expression_is_forwarded(expr.self) && !expression_suppresses_usage_tracking(expr.self) &&
	    forced_invariant_temporaries.count(expr.self) == 0)
	{
		forced_temporaries.insert(expr.self);
		forced_invariant_temporaries.insert(expr.self);
		force_recompile();

		for (auto &dependent : expr.expression_dependencies)
			disallow_forwarding_in_expression_chain(get<SPIRExpression>(dependent));
	}
}

void CompilerGLSL::handle_store_to_invariant_variable(uint32_t store_id, uint32_t value_id)
{
	// Variables or access chains marked invariant are complicated. We will need to make sure the code-gen leading up to
	// this variable is consistent. The failure case for SPIRV-Cross is when an expression is forced to a temporary
	// in one translation unit, but not another, e.g. due to multiple use of an expression.
	// This causes variance despite the output variable being marked invariant, so the solution here is to force all dependent
	// expressions to be temporaries.
	// It is uncertain if this is enough to support invariant in all possible cases, but it should be good enough
	// for all reasonable uses of invariant.
	if (!has_decoration(store_id, DecorationInvariant))
		return;

	auto *expr = maybe_get<SPIRExpression>(value_id);
	if (!expr)
		return;

	disallow_forwarding_in_expression_chain(*expr);
}

void CompilerGLSL::emit_store_statement(uint32_t lhs_expression, uint32_t rhs_expression)
{
	auto rhs = to_pointer_expression(rhs_expression);

	// Statements to OpStore may be empty if it is a struct with zero members. Just forward the store to /dev/null.
	if (!rhs.empty())
	{
		handle_store_to_invariant_variable(lhs_expression, rhs_expression);

		auto lhs = to_dereferenced_expression(lhs_expression);

		// We might need to bitcast in order to store to a builtin.
		bitcast_to_builtin_store(lhs_expression, rhs, expression_type(rhs_expression));

		// Tries to optimize assignments like "<lhs> = <lhs> op expr".
		// While this is purely cosmetic, this is important for legacy ESSL where loop
		// variable increments must be in either i++ or i += const-expr.
		// Without this, we end up with i = i + 1, which is correct GLSL, but not correct GLES 2.0.
		if (!optimize_read_modify_write(expression_type(rhs_expression), lhs, rhs))
			statement(lhs, " = ", rhs, ";");
		register_write(lhs_expression);
	}
}

uint32_t CompilerGLSL::get_integer_width_for_instruction(const Instruction &instr) const
{
	if (instr.length < 3)
		return 32;

	auto *ops = stream(instr);

	switch (instr.op)
	{
	case OpSConvert:
	case OpConvertSToF:
	case OpUConvert:
	case OpConvertUToF:
	case OpIEqual:
	case OpINotEqual:
	case OpSLessThan:
	case OpSLessThanEqual:
	case OpSGreaterThan:
	case OpSGreaterThanEqual:
	case OpULessThan:
	case OpULessThanEqual:
	case OpUGreaterThan:
	case OpUGreaterThanEqual:
		return expression_type(ops[2]).width;

	default:
	{
		// We can look at result type which is more robust.
		auto *type = maybe_get<SPIRType>(ops[0]);
		if (type && type_is_integral(*type))
			return type->width;
		else
			return 32;
	}
	}
}

uint32_t CompilerGLSL::get_integer_width_for_glsl_instruction(GLSLstd450 op, const uint32_t *ops, uint32_t length) const
{
	if (length < 1)
		return 32;

	switch (op)
	{
	case GLSLstd450SAbs:
	case GLSLstd450SSign:
	case GLSLstd450UMin:
	case GLSLstd450SMin:
	case GLSLstd450UMax:
	case GLSLstd450SMax:
	case GLSLstd450UClamp:
	case GLSLstd450SClamp:
	case GLSLstd450FindSMsb:
	case GLSLstd450FindUMsb:
		return expression_type(ops[0]).width;

	default:
	{
		// We don't need to care about other opcodes, just return 32.
		return 32;
	}
	}
}

void CompilerGLSL::emit_instruction(const Instruction &instruction)
{
	auto ops = stream(instruction);
	auto opcode = static_cast<Op>(instruction.op);
	uint32_t length = instruction.length;

#define GLSL_BOP(op) emit_binary_op(ops[0], ops[1], ops[2], ops[3], #op)
#define GLSL_BOP_CAST(op, type) \
	emit_binary_op_cast(ops[0], ops[1], ops[2], ops[3], #op, type, opcode_is_sign_invariant(opcode))
#define GLSL_UOP(op) emit_unary_op(ops[0], ops[1], ops[2], #op)
#define GLSL_QFOP(op) emit_quaternary_func_op(ops[0], ops[1], ops[2], ops[3], ops[4], ops[5], #op)
#define GLSL_TFOP(op) emit_trinary_func_op(ops[0], ops[1], ops[2], ops[3], ops[4], #op)
#define GLSL_BFOP(op) emit_binary_func_op(ops[0], ops[1], ops[2], ops[3], #op)
#define GLSL_BFOP_CAST(op, type) \
	emit_binary_func_op_cast(ops[0], ops[1], ops[2], ops[3], #op, type, opcode_is_sign_invariant(opcode))
#define GLSL_BFOP(op) emit_binary_func_op(ops[0], ops[1], ops[2], ops[3], #op)
#define GLSL_UFOP(op) emit_unary_func_op(ops[0], ops[1], ops[2], #op)

	// If we need to do implicit bitcasts, make sure we do it with the correct type.
	uint32_t integer_width = get_integer_width_for_instruction(instruction);
	auto int_type = to_signed_basetype(integer_width);
	auto uint_type = to_unsigned_basetype(integer_width);

	switch (opcode)
	{
	// Dealing with memory
	case OpLoad:
	{
		uint32_t result_type = ops[0];
		uint32_t id = ops[1];
		uint32_t ptr = ops[2];

		flush_variable_declaration(ptr);

		// If we're loading from memory that cannot be changed by the shader,
		// just forward the expression directly to avoid needless temporaries.
		// If an expression is mutable and forwardable, we speculate that it is immutable.
		bool forward = should_forward(ptr) && forced_temporaries.find(id) == end(forced_temporaries);

		// If loading a non-native row-major matrix, mark the expression as need_transpose.
		bool need_transpose = false;
		bool old_need_transpose = false;

		auto *ptr_expression = maybe_get<SPIRExpression>(ptr);

		if (forward)
		{
			// If we're forwarding the load, we're also going to forward transpose state, so don't transpose while
			// taking the expression.
			if (ptr_expression && ptr_expression->need_transpose)
			{
				old_need_transpose = true;
				ptr_expression->need_transpose = false;
				need_transpose = true;
			}
			else if (is_non_native_row_major_matrix(ptr))
				need_transpose = true;
		}

		// If we are forwarding this load,
		// don't register the read to access chain here, defer that to when we actually use the expression,
		// using the add_implied_read_expression mechanism.
		string expr;

		bool is_packed = has_extended_decoration(ptr, SPIRVCrossDecorationPhysicalTypePacked);
		bool is_remapped = has_extended_decoration(ptr, SPIRVCrossDecorationPhysicalTypeID);
		if (forward || (!is_packed && !is_remapped))
		{
			// For the simple case, we do not need to deal with repacking.
			expr = to_dereferenced_expression(ptr, false);
		}
		else
		{
			// If we are not forwarding the expression, we need to unpack and resolve any physical type remapping here before
			// storing the expression to a temporary.
			expr = to_unpacked_expression(ptr);
		}

		// We might need to bitcast in order to load from a builtin.
		bitcast_from_builtin_load(ptr, expr, get<SPIRType>(result_type));

		// We might be trying to load a gl_Position[N], where we should be
		// doing float4[](gl_in[i].gl_Position, ...) instead.
		// Similar workarounds are required for input arrays in tessellation.
		unroll_array_from_complex_load(id, ptr, expr);

		auto &type = get<SPIRType>(result_type);
		// Shouldn't need to check for ID, but current glslang codegen requires it in some cases
		// when loading Image/Sampler descriptors. It does not hurt to check ID as well.
		if (has_decoration(id, DecorationNonUniformEXT) || has_decoration(ptr, DecorationNonUniformEXT))
		{
			propagate_nonuniform_qualifier(ptr);
			convert_non_uniform_expression(type, expr);
		}

		if (forward && ptr_expression)
			ptr_expression->need_transpose = old_need_transpose;

		// By default, suppress usage tracking since using same expression multiple times does not imply any extra work.
		// However, if we try to load a complex, composite object from a flattened buffer,
		// we should avoid emitting the same code over and over and lower the result to a temporary.
		bool usage_tracking = ptr_expression && flattened_buffer_blocks.count(ptr_expression->loaded_from) != 0 &&
		                      (type.basetype == SPIRType::Struct || (type.columns > 1));

		SPIRExpression *e = nullptr;
		if (!backend.array_is_value_type && !type.array.empty() && !forward)
		{
			// Complicated load case where we need to make a copy of ptr, but we cannot, because
			// it is an array, and our backend does not support arrays as value types.
			// Emit the temporary, and copy it explicitly.
			e = &emit_uninitialized_temporary_expression(result_type, id);
			emit_array_copy(to_expression(id), ptr, StorageClassFunction, get_backing_variable_storage(ptr));
		}
		else
			e = &emit_op(result_type, id, expr, forward, !usage_tracking);

		e->need_transpose = need_transpose;
		register_read(id, ptr, forward);

		if (forward)
		{
			// Pass through whether the result is of a packed type and the physical type ID.
			if (has_extended_decoration(ptr, SPIRVCrossDecorationPhysicalTypePacked))
				set_extended_decoration(id, SPIRVCrossDecorationPhysicalTypePacked);
			if (has_extended_decoration(ptr, SPIRVCrossDecorationPhysicalTypeID))
			{
				set_extended_decoration(id, SPIRVCrossDecorationPhysicalTypeID,
				                        get_extended_decoration(ptr, SPIRVCrossDecorationPhysicalTypeID));
			}
		}
		else
		{
			// This might have been set on an earlier compilation iteration, force it to be unset.
			unset_extended_decoration(id, SPIRVCrossDecorationPhysicalTypePacked);
			unset_extended_decoration(id, SPIRVCrossDecorationPhysicalTypeID);
		}

		inherit_expression_dependencies(id, ptr);
		if (forward)
			add_implied_read_expression(*e, ptr);
		break;
	}

	case OpInBoundsAccessChain:
	case OpAccessChain:
	case OpPtrAccessChain:
	{
		auto *var = maybe_get<SPIRVariable>(ops[2]);
		if (var)
			flush_variable_declaration(var->self);

		// If the base is immutable, the access chain pointer must also be.
		// If an expression is mutable and forwardable, we speculate that it is immutable.
		AccessChainMeta meta;
		bool ptr_chain = opcode == OpPtrAccessChain;
		auto e = access_chain(ops[2], &ops[3], length - 3, get<SPIRType>(ops[0]), &meta, ptr_chain);

		auto &expr = set<SPIRExpression>(ops[1], move(e), ops[0], should_forward(ops[2]));

		auto *backing_variable = maybe_get_backing_variable(ops[2]);
		expr.loaded_from = backing_variable ? backing_variable->self : ID(ops[2]);
		expr.need_transpose = meta.need_transpose;
		expr.access_chain = true;

		// Mark the result as being packed. Some platforms handled packed vectors differently than non-packed.
		if (meta.storage_is_packed)
			set_extended_decoration(ops[1], SPIRVCrossDecorationPhysicalTypePacked);
		if (meta.storage_physical_type != 0)
			set_extended_decoration(ops[1], SPIRVCrossDecorationPhysicalTypeID, meta.storage_physical_type);
		if (meta.storage_is_invariant)
			set_decoration(ops[1], DecorationInvariant);

		// If we have some expression dependencies in our access chain, this access chain is technically a forwarded
		// temporary which could be subject to invalidation.
		// Need to assume we're forwarded while calling inherit_expression_depdendencies.
		forwarded_temporaries.insert(ops[1]);
		// The access chain itself is never forced to a temporary, but its dependencies might.
		suppressed_usage_tracking.insert(ops[1]);

		for (uint32_t i = 2; i < length; i++)
		{
			inherit_expression_dependencies(ops[1], ops[i]);
			add_implied_read_expression(expr, ops[i]);
		}

		// If we have no dependencies after all, i.e., all indices in the access chain are immutable temporaries,
		// we're not forwarded after all.
		if (expr.expression_dependencies.empty())
			forwarded_temporaries.erase(ops[1]);

		if (has_decoration(ops[1], DecorationNonUniformEXT) || has_decoration(ops[2], DecorationNonUniformEXT))
			propagate_nonuniform_qualifier(ops[1]);

		break;
	}

	case OpStore:
	{
		auto *var = maybe_get<SPIRVariable>(ops[0]);

		if (var && var->statically_assigned)
			var->static_expression = ops[1];
		else if (var && var->loop_variable && !var->loop_variable_enable)
			var->static_expression = ops[1];
		else if (var && var->remapped_variable)
		{
			// Skip the write.
		}
		else if (var && flattened_structs.count(ops[0]))
		{
			store_flattened_struct(*var, ops[1]);
			register_write(ops[0]);
		}
		else
		{
			emit_store_statement(ops[0], ops[1]);
		}

		// Storing a pointer results in a variable pointer, so we must conservatively assume
		// we can write through it.
		if (expression_type(ops[1]).pointer)
			register_write(ops[1]);
		break;
	}

	case OpArrayLength:
	{
		uint32_t result_type = ops[0];
		uint32_t id = ops[1];
		auto e = access_chain_internal(ops[2], &ops[3], length - 3, ACCESS_CHAIN_INDEX_IS_LITERAL_BIT, nullptr);
		set<SPIRExpression>(id, join(type_to_glsl(get<SPIRType>(result_type)), "(", e, ".length())"), result_type,
		                    true);
		break;
	}

	// Function calls
	case OpFunctionCall:
	{
		uint32_t result_type = ops[0];
		uint32_t id = ops[1];
		uint32_t func = ops[2];
		const auto *arg = &ops[3];
		length -= 3;

		auto &callee = get<SPIRFunction>(func);
		auto &return_type = get<SPIRType>(callee.return_type);
		bool pure = function_is_pure(callee);

		bool callee_has_out_variables = false;
		bool emit_return_value_as_argument = false;

		// Invalidate out variables passed to functions since they can be OpStore'd to.
		for (uint32_t i = 0; i < length; i++)
		{
			if (callee.arguments[i].write_count)
			{
				register_call_out_argument(arg[i]);
				callee_has_out_variables = true;
			}

			flush_variable_declaration(arg[i]);
		}

		if (!return_type.array.empty() && !backend.can_return_array)
		{
			callee_has_out_variables = true;
			emit_return_value_as_argument = true;
		}

		if (!pure)
			register_impure_function_call();

		string funexpr;
		SmallVector<string> arglist;
		funexpr += to_name(func) + "(";

		if (emit_return_value_as_argument)
		{
			statement(type_to_glsl(return_type), " ", to_name(id), type_to_array_glsl(return_type), ";");
			arglist.push_back(to_name(id));
		}

		for (uint32_t i = 0; i < length; i++)
		{
			// Do not pass in separate images or samplers if we're remapping
			// to combined image samplers.
			if (skip_argument(arg[i]))
				continue;

			arglist.push_back(to_func_call_arg(callee.arguments[i], arg[i]));
		}

		for (auto &combined : callee.combined_parameters)
		{
			auto image_id = combined.global_image ? combined.image_id : VariableID(arg[combined.image_id]);
			auto sampler_id = combined.global_sampler ? combined.sampler_id : VariableID(arg[combined.sampler_id]);
			arglist.push_back(to_combined_image_sampler(image_id, sampler_id));
		}

		append_global_func_args(callee, length, arglist);

		funexpr += merge(arglist);
		funexpr += ")";

		// Check for function call constraints.
		check_function_call_constraints(arg, length);

		if (return_type.basetype != SPIRType::Void)
		{
			// If the function actually writes to an out variable,
			// take the conservative route and do not forward.
			// The problem is that we might not read the function
			// result (and emit the function) before an out variable
			// is read (common case when return value is ignored!
			// In order to avoid start tracking invalid variables,
			// just avoid the forwarding problem altogether.
			bool forward = args_will_forward(id, arg, length, pure) && !callee_has_out_variables && pure &&
			               (forced_temporaries.find(id) == end(forced_temporaries));

			if (emit_return_value_as_argument)
			{
				statement(funexpr, ";");
				set<SPIRExpression>(id, to_name(id), result_type, true);
			}
			else
				emit_op(result_type, id, funexpr, forward);

			// Function calls are implicit loads from all variables in question.
			// Set dependencies for them.
			for (uint32_t i = 0; i < length; i++)
				register_read(id, arg[i], forward);

			// If we're going to forward the temporary result,
			// put dependencies on every variable that must not change.
			if (forward)
				register_global_read_dependencies(callee, id);
		}
		else
			statement(funexpr, ";");

		break;
	}

	// Composite munging
	case OpCompositeConstruct:
	{
		uint32_t result_type = ops[0];
		uint32_t id = ops[1];
		const auto *const elems = &ops[2];
		length -= 2;

		bool forward = true;
		for (uint32_t i = 0; i < length; i++)
			forward = forward && should_forward(elems[i]);

		auto &out_type = get<SPIRType>(result_type);
		auto *in_type = length > 0 ? &expression_type(elems[0]) : nullptr;

		// Only splat if we have vector constructors.
		// Arrays and structs must be initialized properly in full.
		bool composite = !out_type.array.empty() || out_type.basetype == SPIRType::Struct;

		bool splat = false;
		bool swizzle_splat = false;

		if (in_type)
		{
			splat = in_type->vecsize == 1 && in_type->columns == 1 && !composite && backend.use_constructor_splatting;
			swizzle_splat = in_type->vecsize == 1 && in_type->columns == 1 && backend.can_swizzle_scalar;

			if (ir.ids[elems[0]].get_type() == TypeConstant && !type_is_floating_point(*in_type))
			{
				// Cannot swizzle literal integers as a special case.
				swizzle_splat = false;
			}
		}

		if (splat || swizzle_splat)
		{
			uint32_t input = elems[0];
			for (uint32_t i = 0; i < length; i++)
			{
				if (input != elems[i])
				{
					splat = false;
					swizzle_splat = false;
				}
			}
		}

		if (out_type.basetype == SPIRType::Struct && !backend.can_declare_struct_inline)
			forward = false;
		if (!out_type.array.empty() && !backend.can_declare_arrays_inline)
			forward = false;
		if (type_is_empty(out_type) && !backend.supports_empty_struct)
			forward = false;

		string constructor_op;
		if (backend.use_initializer_list && composite)
		{
			bool needs_trailing_tracket = false;
			// Only use this path if we are building composites.
			// This path cannot be used for arithmetic.
			if (backend.use_typed_initializer_list && out_type.basetype == SPIRType::Struct && out_type.array.empty())
				constructor_op += type_to_glsl_constructor(get<SPIRType>(result_type));
			else if (backend.use_typed_initializer_list && backend.array_is_value_type && !out_type.array.empty())
			{
				// MSL path. Array constructor is baked into type here, do not use _constructor variant.
				constructor_op += type_to_glsl_constructor(get<SPIRType>(result_type)) + "(";
				needs_trailing_tracket = true;
			}
			constructor_op += "{ ";

			if (type_is_empty(out_type) && !backend.supports_empty_struct)
				constructor_op += "0";
			else if (splat)
				constructor_op += to_unpacked_expression(elems[0]);
			else
				constructor_op += build_composite_combiner(result_type, elems, length);
			constructor_op += " }";
			if (needs_trailing_tracket)
				constructor_op += ")";
		}
		else if (swizzle_splat && !composite)
		{
			constructor_op = remap_swizzle(get<SPIRType>(result_type), 1, to_unpacked_expression(elems[0]));
		}
		else
		{
			constructor_op = type_to_glsl_constructor(get<SPIRType>(result_type)) + "(";
			if (type_is_empty(out_type) && !backend.supports_empty_struct)
				constructor_op += "0";
			else if (splat)
				constructor_op += to_unpacked_expression(elems[0]);
			else
				constructor_op += build_composite_combiner(result_type, elems, length);
			constructor_op += ")";
		}

		if (!constructor_op.empty())
		{
			emit_op(result_type, id, constructor_op, forward);
			for (uint32_t i = 0; i < length; i++)
				inherit_expression_dependencies(id, elems[i]);
		}
		break;
	}

	case OpVectorInsertDynamic:
	{
		uint32_t result_type = ops[0];
		uint32_t id = ops[1];
		uint32_t vec = ops[2];
		uint32_t comp = ops[3];
		uint32_t index = ops[4];

		flush_variable_declaration(vec);

		// Make a copy, then use access chain to store the variable.
		statement(declare_temporary(result_type, id), to_expression(vec), ";");
		set<SPIRExpression>(id, to_name(id), result_type, true);
		auto chain = access_chain_internal(id, &index, 1, 0, nullptr);
		statement(chain, " = ", to_expression(comp), ";");
		break;
	}

	case OpVectorExtractDynamic:
	{
		uint32_t result_type = ops[0];
		uint32_t id = ops[1];

		auto expr = access_chain_internal(ops[2], &ops[3], 1, 0, nullptr);
		emit_op(result_type, id, expr, should_forward(ops[2]));
		inherit_expression_dependencies(id, ops[2]);
		inherit_expression_dependencies(id, ops[3]);
		break;
	}

	case OpCompositeExtract:
	{
		uint32_t result_type = ops[0];
		uint32_t id = ops[1];
		length -= 3;

		auto &type = get<SPIRType>(result_type);

		// We can only split the expression here if our expression is forwarded as a temporary.
		bool allow_base_expression = forced_temporaries.find(id) == end(forced_temporaries);

		// Do not allow base expression for struct members. We risk doing "swizzle" optimizations in this case.
		auto &composite_type = expression_type(ops[2]);
		if (composite_type.basetype == SPIRType::Struct || !composite_type.array.empty())
			allow_base_expression = false;

		// Packed expressions or physical ID mapped expressions cannot be split up.
		if (has_extended_decoration(ops[2], SPIRVCrossDecorationPhysicalTypePacked) ||
		    has_extended_decoration(ops[2], SPIRVCrossDecorationPhysicalTypeID))
			allow_base_expression = false;

		// Cannot use base expression for row-major matrix row-extraction since we need to interleave access pattern
		// into the base expression.
		if (is_non_native_row_major_matrix(ops[2]))
			allow_base_expression = false;

		AccessChainMeta meta;
		SPIRExpression *e = nullptr;

		// Only apply this optimization if result is scalar.
		if (allow_base_expression && should_forward(ops[2]) && type.vecsize == 1 && type.columns == 1 && length == 1)
		{
			// We want to split the access chain from the base.
			// This is so we can later combine different CompositeExtract results
			// with CompositeConstruct without emitting code like
			//
			// vec3 temp = texture(...).xyz
			// vec4(temp.x, temp.y, temp.z, 1.0).
			//
			// when we actually wanted to emit this
			// vec4(texture(...).xyz, 1.0).
			//
			// Including the base will prevent this and would trigger multiple reads
			// from expression causing it to be forced to an actual temporary in GLSL.
			auto expr = access_chain_internal(ops[2], &ops[3], length,
			                                  ACCESS_CHAIN_INDEX_IS_LITERAL_BIT | ACCESS_CHAIN_CHAIN_ONLY_BIT, &meta);
			e = &emit_op(result_type, id, expr, true, should_suppress_usage_tracking(ops[2]));
			inherit_expression_dependencies(id, ops[2]);
			e->base_expression = ops[2];
		}
		else
		{
			auto expr = access_chain_internal(ops[2], &ops[3], length, ACCESS_CHAIN_INDEX_IS_LITERAL_BIT, &meta);
			e = &emit_op(result_type, id, expr, should_forward(ops[2]), should_suppress_usage_tracking(ops[2]));
			inherit_expression_dependencies(id, ops[2]);
		}

		// Pass through some meta information to the loaded expression.
		// We can still end up loading a buffer type to a variable, then CompositeExtract from it
		// instead of loading everything through an access chain.
		e->need_transpose = meta.need_transpose;
		if (meta.storage_is_packed)
			set_extended_decoration(id, SPIRVCrossDecorationPhysicalTypePacked);
		if (meta.storage_physical_type != 0)
			set_extended_decoration(id, SPIRVCrossDecorationPhysicalTypeID, meta.storage_physical_type);
		if (meta.storage_is_invariant)
			set_decoration(id, DecorationInvariant);

		break;
	}

	case OpCompositeInsert:
	{
		uint32_t result_type = ops[0];
		uint32_t id = ops[1];
		uint32_t obj = ops[2];
		uint32_t composite = ops[3];
		const auto *elems = &ops[4];
		length -= 4;

		flush_variable_declaration(composite);

		// Make a copy, then use access chain to store the variable.
		statement(declare_temporary(result_type, id), to_expression(composite), ";");
		set<SPIRExpression>(id, to_name(id), result_type, true);
		auto chain = access_chain_internal(id, elems, length, ACCESS_CHAIN_INDEX_IS_LITERAL_BIT, nullptr);
		statement(chain, " = ", to_expression(obj), ";");

		break;
	}

	case OpCopyMemory:
	{
		uint32_t lhs = ops[0];
		uint32_t rhs = ops[1];
		if (lhs != rhs)
		{
			flush_variable_declaration(lhs);
			flush_variable_declaration(rhs);
			statement(to_expression(lhs), " = ", to_expression(rhs), ";");
			register_write(lhs);
		}
		break;
	}

	case OpCopyLogical:
	{
		// This is used for copying object of different types, arrays and structs.
		// We need to unroll the copy, element-by-element.
		uint32_t result_type = ops[0];
		uint32_t id = ops[1];
		uint32_t rhs = ops[2];

		emit_uninitialized_temporary_expression(result_type, id);
		emit_copy_logical_type(id, result_type, rhs, expression_type_id(rhs), {});
		break;
	}

	case OpCopyObject:
	{
		uint32_t result_type = ops[0];
		uint32_t id = ops[1];
		uint32_t rhs = ops[2];
		bool pointer = get<SPIRType>(result_type).pointer;

		auto *chain = maybe_get<SPIRAccessChain>(rhs);
		if (chain)
		{
			// Cannot lower to a SPIRExpression, just copy the object.
			auto &e = set<SPIRAccessChain>(id, *chain);
			e.self = id;
		}
		else if (expression_is_lvalue(rhs) && !pointer)
		{
			// Need a copy.
			// For pointer types, we copy the pointer itself.
			statement(declare_temporary(result_type, id), to_unpacked_expression(rhs), ";");
			set<SPIRExpression>(id, to_name(id), result_type, true);
		}
		else
		{
			// RHS expression is immutable, so just forward it.
			// Copying these things really make no sense, but
			// seems to be allowed anyways.
			auto &e = set<SPIRExpression>(id, to_expression(rhs), result_type, true);
			if (pointer)
			{
				auto *var = maybe_get_backing_variable(rhs);
				e.loaded_from = var ? var->self : ID(0);
			}

			// If we're copying an access chain, need to inherit the read expressions.
			auto *rhs_expr = maybe_get<SPIRExpression>(rhs);
			if (rhs_expr)
			{
				e.implied_read_expressions = rhs_expr->implied_read_expressions;
				e.expression_dependencies = rhs_expr->expression_dependencies;
			}
		}
		break;
	}

	case OpVectorShuffle:
	{
		uint32_t result_type = ops[0];
		uint32_t id = ops[1];
		uint32_t vec0 = ops[2];
		uint32_t vec1 = ops[3];
		const auto *elems = &ops[4];
		length -= 4;

		auto &type0 = expression_type(vec0);

		// If we have the undefined swizzle index -1, we need to swizzle in undefined data,
		// or in our case, T(0).
		bool shuffle = false;
		for (uint32_t i = 0; i < length; i++)
			if (elems[i] >= type0.vecsize || elems[i] == 0xffffffffu)
				shuffle = true;

		// Cannot use swizzles with packed expressions, force shuffle path.
		if (!shuffle && has_extended_decoration(vec0, SPIRVCrossDecorationPhysicalTypePacked))
			shuffle = true;

		string expr;
		bool should_fwd, trivial_forward;

		if (shuffle)
		{
			should_fwd = should_forward(vec0) && should_forward(vec1);
			trivial_forward = should_suppress_usage_tracking(vec0) && should_suppress_usage_tracking(vec1);

			// Constructor style and shuffling from two different vectors.
			SmallVector<string> args;
			for (uint32_t i = 0; i < length; i++)
			{
				if (elems[i] == 0xffffffffu)
				{
					// Use a constant 0 here.
					// We could use the first component or similar, but then we risk propagating
					// a value we might not need, and bog down codegen.
					SPIRConstant c;
					c.constant_type = type0.parent_type;
					assert(type0.parent_type != ID(0));
					args.push_back(constant_expression(c));
				}
				else if (elems[i] >= type0.vecsize)
					args.push_back(to_extract_component_expression(vec1, elems[i] - type0.vecsize));
				else
					args.push_back(to_extract_component_expression(vec0, elems[i]));
			}
			expr += join(type_to_glsl_constructor(get<SPIRType>(result_type)), "(", merge(args), ")");
		}
		else
		{
			should_fwd = should_forward(vec0);
			trivial_forward = should_suppress_usage_tracking(vec0);

			// We only source from first vector, so can use swizzle.
			// If the vector is packed, unpack it before applying a swizzle (needed for MSL)
			expr += to_enclosed_unpacked_expression(vec0);
			expr += ".";
			for (uint32_t i = 0; i < length; i++)
			{
				assert(elems[i] != 0xffffffffu);
				expr += index_to_swizzle(elems[i]);
			}

			if (backend.swizzle_is_function && length > 1)
				expr += "()";
		}

		// A shuffle is trivial in that it doesn't actually *do* anything.
		// We inherit the forwardedness from our arguments to avoid flushing out to temporaries when it's not really needed.

		emit_op(result_type, id, expr, should_fwd, trivial_forward);

		inherit_expression_dependencies(id, vec0);
		if (vec0 != vec1)
			inherit_expression_dependencies(id, vec1);
		break;
	}

	// ALU
	case OpIsNan:
		GLSL_UFOP(isnan);
		break;

	case OpIsInf:
		GLSL_UFOP(isinf);
		break;

	case OpSNegate:
	case OpFNegate:
		GLSL_UOP(-);
		break;

	case OpIAdd:
	{
		// For simple arith ops, prefer the output type if there's a mismatch to avoid extra bitcasts.
		auto type = get<SPIRType>(ops[0]).basetype;
		GLSL_BOP_CAST(+, type);
		break;
	}

	case OpFAdd:
		GLSL_BOP(+);
		break;

	case OpISub:
	{
		auto type = get<SPIRType>(ops[0]).basetype;
		GLSL_BOP_CAST(-, type);
		break;
	}

	case OpFSub:
		GLSL_BOP(-);
		break;

	case OpIMul:
	{
		auto type = get<SPIRType>(ops[0]).basetype;
		GLSL_BOP_CAST(*, type);
		break;
	}

	case OpVectorTimesMatrix:
	case OpMatrixTimesVector:
	{
		// If the matrix needs transpose, just flip the multiply order.
		auto *e = maybe_get<SPIRExpression>(ops[opcode == OpMatrixTimesVector ? 2 : 3]);
		if (e && e->need_transpose)
		{
			e->need_transpose = false;
			string expr;

			if (opcode == OpMatrixTimesVector)
				expr = join(to_enclosed_unpacked_expression(ops[3]), " * ",
				            enclose_expression(to_unpacked_row_major_matrix_expression(ops[2])));
			else
				expr = join(enclose_expression(to_unpacked_row_major_matrix_expression(ops[3])), " * ",
				            to_enclosed_unpacked_expression(ops[2]));

			bool forward = should_forward(ops[2]) && should_forward(ops[3]);
			emit_op(ops[0], ops[1], expr, forward);
			e->need_transpose = true;
			inherit_expression_dependencies(ops[1], ops[2]);
			inherit_expression_dependencies(ops[1], ops[3]);
		}
		else
			GLSL_BOP(*);
		break;
	}

	case OpMatrixTimesMatrix:
	{
		auto *a = maybe_get<SPIRExpression>(ops[2]);
		auto *b = maybe_get<SPIRExpression>(ops[3]);

		// If both matrices need transpose, we can multiply in flipped order and tag the expression as transposed.
		// a^T * b^T = (b * a)^T.
		if (a && b && a->need_transpose && b->need_transpose)
		{
			a->need_transpose = false;
			b->need_transpose = false;
			auto expr = join(enclose_expression(to_unpacked_row_major_matrix_expression(ops[3])), " * ",
			                 enclose_expression(to_unpacked_row_major_matrix_expression(ops[2])));
			bool forward = should_forward(ops[2]) && should_forward(ops[3]);
			auto &e = emit_op(ops[0], ops[1], expr, forward);
			e.need_transpose = true;
			a->need_transpose = true;
			b->need_transpose = true;
			inherit_expression_dependencies(ops[1], ops[2]);
			inherit_expression_dependencies(ops[1], ops[3]);
		}
		else
			GLSL_BOP(*);

		break;
	}

	case OpFMul:
	case OpMatrixTimesScalar:
	case OpVectorTimesScalar:
		GLSL_BOP(*);
		break;

	case OpOuterProduct:
		GLSL_BFOP(outerProduct);
		break;

	case OpDot:
		GLSL_BFOP(dot);
		break;

	case OpTranspose:
		GLSL_UFOP(transpose);
		break;

	case OpSRem:
	{
		uint32_t result_type = ops[0];
		uint32_t result_id = ops[1];
		uint32_t op0 = ops[2];
		uint32_t op1 = ops[3];

		// Needs special handling.
		bool forward = should_forward(op0) && should_forward(op1);
		auto expr = join(to_enclosed_expression(op0), " - ", to_enclosed_expression(op1), " * ", "(",
		                 to_enclosed_expression(op0), " / ", to_enclosed_expression(op1), ")");

		emit_op(result_type, result_id, expr, forward);
		inherit_expression_dependencies(result_id, op0);
		inherit_expression_dependencies(result_id, op1);
		break;
	}

	case OpSDiv:
		GLSL_BOP_CAST(/, int_type);
		break;

	case OpUDiv:
		GLSL_BOP_CAST(/, uint_type);
		break;

	case OpIAddCarry:
	case OpISubBorrow:
	{
		if (options.es && options.version < 310)
			SPIRV_CROSS_THROW("Extended arithmetic is only available from ESSL 310.");
		else if (!options.es && options.version < 400)
			SPIRV_CROSS_THROW("Extended arithmetic is only available from GLSL 400.");

		uint32_t result_type = ops[0];
		uint32_t result_id = ops[1];
		uint32_t op0 = ops[2];
		uint32_t op1 = ops[3];
		auto &type = get<SPIRType>(result_type);
		emit_uninitialized_temporary_expression(result_type, result_id);
		const char *op = opcode == OpIAddCarry ? "uaddCarry" : "usubBorrow";

		statement(to_expression(result_id), ".", to_member_name(type, 0), " = ", op, "(", to_expression(op0), ", ",
		          to_expression(op1), ", ", to_expression(result_id), ".", to_member_name(type, 1), ");");
		break;
	}

	case OpUMulExtended:
	case OpSMulExtended:
	{
		if (options.es && options.version < 310)
			SPIRV_CROSS_THROW("Extended arithmetic is only available from ESSL 310.");
		else if (!options.es && options.version < 400)
			SPIRV_CROSS_THROW("Extended arithmetic is only available from GLSL 4000.");

		uint32_t result_type = ops[0];
		uint32_t result_id = ops[1];
		uint32_t op0 = ops[2];
		uint32_t op1 = ops[3];
		auto &type = get<SPIRType>(result_type);
		emit_uninitialized_temporary_expression(result_type, result_id);
		const char *op = opcode == OpUMulExtended ? "umulExtended" : "imulExtended";

		statement(op, "(", to_expression(op0), ", ", to_expression(op1), ", ", to_expression(result_id), ".",
		          to_member_name(type, 1), ", ", to_expression(result_id), ".", to_member_name(type, 0), ");");
		break;
	}

	case OpFDiv:
		GLSL_BOP(/);
		break;

	case OpShiftRightLogical:
		GLSL_BOP_CAST(>>, uint_type);
		break;

	case OpShiftRightArithmetic:
		GLSL_BOP_CAST(>>, int_type);
		break;

	case OpShiftLeftLogical:
	{
		auto type = get<SPIRType>(ops[0]).basetype;
		GLSL_BOP_CAST(<<, type);
		break;
	}

	case OpBitwiseOr:
	{
		auto type = get<SPIRType>(ops[0]).basetype;
		GLSL_BOP_CAST(|, type);
		break;
	}

	case OpBitwiseXor:
	{
		auto type = get<SPIRType>(ops[0]).basetype;
		GLSL_BOP_CAST(^, type);
		break;
	}

	case OpBitwiseAnd:
	{
		auto type = get<SPIRType>(ops[0]).basetype;
		GLSL_BOP_CAST(&, type);
		break;
	}

	case OpNot:
		GLSL_UOP(~);
		break;

	case OpUMod:
		GLSL_BOP_CAST(%, uint_type);
		break;

	case OpSMod:
		GLSL_BOP_CAST(%, int_type);
		break;

	case OpFMod:
		GLSL_BFOP(mod);
		break;

	case OpFRem:
	{
		if (is_legacy())
			SPIRV_CROSS_THROW("OpFRem requires trunc() and is only supported on non-legacy targets. A workaround is "
			                  "needed for legacy.");

		uint32_t result_type = ops[0];
		uint32_t result_id = ops[1];
		uint32_t op0 = ops[2];
		uint32_t op1 = ops[3];

		// Needs special handling.
		bool forward = should_forward(op0) && should_forward(op1);
		auto expr = join(to_enclosed_expression(op0), " - ", to_enclosed_expression(op1), " * ", "trunc(",
		                 to_enclosed_expression(op0), " / ", to_enclosed_expression(op1), ")");

		emit_op(result_type, result_id, expr, forward);
		inherit_expression_dependencies(result_id, op0);
		inherit_expression_dependencies(result_id, op1);
		break;
	}

	// Relational
	case OpAny:
		GLSL_UFOP(any);
		break;

	case OpAll:
		GLSL_UFOP(all);
		break;

	case OpSelect:
		emit_mix_op(ops[0], ops[1], ops[4], ops[3], ops[2]);
		break;

	case OpLogicalOr:
	{
		// No vector variant in GLSL for logical OR.
		auto result_type = ops[0];
		auto id = ops[1];
		auto &type = get<SPIRType>(result_type);

		if (type.vecsize > 1)
			emit_unrolled_binary_op(result_type, id, ops[2], ops[3], "||", false, SPIRType::Unknown);
		else
			GLSL_BOP(||);
		break;
	}

	case OpLogicalAnd:
	{
		// No vector variant in GLSL for logical AND.
		auto result_type = ops[0];
		auto id = ops[1];
		auto &type = get<SPIRType>(result_type);

		if (type.vecsize > 1)
			emit_unrolled_binary_op(result_type, id, ops[2], ops[3], "&&", false, SPIRType::Unknown);
		else
			GLSL_BOP(&&);
		break;
	}

	case OpLogicalNot:
	{
		auto &type = get<SPIRType>(ops[0]);
		if (type.vecsize > 1)
			GLSL_UFOP(not);
		else
			GLSL_UOP(!);
		break;
	}

	case OpIEqual:
	{
		if (expression_type(ops[2]).vecsize > 1)
			GLSL_BFOP_CAST(equal, int_type);
		else
			GLSL_BOP_CAST(==, int_type);
		break;
	}

	case OpLogicalEqual:
	case OpFOrdEqual:
	{
		if (expression_type(ops[2]).vecsize > 1)
			GLSL_BFOP(equal);
		else
			GLSL_BOP(==);
		break;
	}

	case OpINotEqual:
	{
		if (expression_type(ops[2]).vecsize > 1)
			GLSL_BFOP_CAST(notEqual, int_type);
		else
			GLSL_BOP_CAST(!=, int_type);
		break;
	}

	case OpLogicalNotEqual:
	case OpFOrdNotEqual:
	{
		if (expression_type(ops[2]).vecsize > 1)
			GLSL_BFOP(notEqual);
		else
			GLSL_BOP(!=);
		break;
	}

	case OpUGreaterThan:
	case OpSGreaterThan:
	{
		auto type = opcode == OpUGreaterThan ? uint_type : int_type;
		if (expression_type(ops[2]).vecsize > 1)
			GLSL_BFOP_CAST(greaterThan, type);
		else
			GLSL_BOP_CAST(>, type);
		break;
	}

	case OpFOrdGreaterThan:
	{
		if (expression_type(ops[2]).vecsize > 1)
			GLSL_BFOP(greaterThan);
		else
			GLSL_BOP(>);
		break;
	}

	case OpUGreaterThanEqual:
	case OpSGreaterThanEqual:
	{
		auto type = opcode == OpUGreaterThanEqual ? uint_type : int_type;
		if (expression_type(ops[2]).vecsize > 1)
			GLSL_BFOP_CAST(greaterThanEqual, type);
		else
			GLSL_BOP_CAST(>=, type);
		break;
	}

	case OpFOrdGreaterThanEqual:
	{
		if (expression_type(ops[2]).vecsize > 1)
			GLSL_BFOP(greaterThanEqual);
		else
			GLSL_BOP(>=);
		break;
	}

	case OpULessThan:
	case OpSLessThan:
	{
		auto type = opcode == OpULessThan ? uint_type : int_type;
		if (expression_type(ops[2]).vecsize > 1)
			GLSL_BFOP_CAST(lessThan, type);
		else
			GLSL_BOP_CAST(<, type);
		break;
	}

	case OpFOrdLessThan:
	{
		if (expression_type(ops[2]).vecsize > 1)
			GLSL_BFOP(lessThan);
		else
			GLSL_BOP(<);
		break;
	}

	case OpULessThanEqual:
	case OpSLessThanEqual:
	{
		auto type = opcode == OpULessThanEqual ? uint_type : int_type;
		if (expression_type(ops[2]).vecsize > 1)
			GLSL_BFOP_CAST(lessThanEqual, type);
		else
			GLSL_BOP_CAST(<=, type);
		break;
	}

	case OpFOrdLessThanEqual:
	{
		if (expression_type(ops[2]).vecsize > 1)
			GLSL_BFOP(lessThanEqual);
		else
			GLSL_BOP(<=);
		break;
	}

	// Conversion
	case OpSConvert:
	case OpConvertSToF:
	case OpUConvert:
	case OpConvertUToF:
	{
		auto input_type = opcode == OpSConvert || opcode == OpConvertSToF ? int_type : uint_type;
		uint32_t result_type = ops[0];
		uint32_t id = ops[1];

		auto &type = get<SPIRType>(result_type);
		auto &arg_type = expression_type(ops[2]);
		auto func = type_to_glsl_constructor(type);

		if (arg_type.width < type.width || type_is_floating_point(type))
			emit_unary_func_op_cast(result_type, id, ops[2], func.c_str(), input_type, type.basetype);
		else
			emit_unary_func_op(result_type, id, ops[2], func.c_str());
		break;
	}

	case OpConvertFToU:
	case OpConvertFToS:
	{
		// Cast to expected arithmetic type, then potentially bitcast away to desired signedness.
		uint32_t result_type = ops[0];
		uint32_t id = ops[1];
		auto &type = get<SPIRType>(result_type);
		auto expected_type = type;
		auto &float_type = expression_type(ops[2]);
		expected_type.basetype =
		    opcode == OpConvertFToS ? to_signed_basetype(type.width) : to_unsigned_basetype(type.width);

		auto func = type_to_glsl_constructor(expected_type);
		emit_unary_func_op_cast(result_type, id, ops[2], func.c_str(), float_type.basetype, expected_type.basetype);
		break;
	}

	case OpFConvert:
	{
		uint32_t result_type = ops[0];
		uint32_t id = ops[1];

		auto func = type_to_glsl_constructor(get<SPIRType>(result_type));
		emit_unary_func_op(result_type, id, ops[2], func.c_str());
		break;
	}

	case OpBitcast:
	{
		uint32_t result_type = ops[0];
		uint32_t id = ops[1];
		uint32_t arg = ops[2];

		auto op = bitcast_glsl_op(get<SPIRType>(result_type), expression_type(arg));
		emit_unary_func_op(result_type, id, arg, op.c_str());
		break;
	}

	case OpQuantizeToF16:
	{
		uint32_t result_type = ops[0];
		uint32_t id = ops[1];
		uint32_t arg = ops[2];

		string op;
		auto &type = get<SPIRType>(result_type);

		switch (type.vecsize)
		{
		case 1:
			op = join("unpackHalf2x16(packHalf2x16(vec2(", to_expression(arg), "))).x");
			break;
		case 2:
			op = join("unpackHalf2x16(packHalf2x16(", to_expression(arg), "))");
			break;
		case 3:
		{
			auto op0 = join("unpackHalf2x16(packHalf2x16(", to_expression(arg), ".xy))");
			auto op1 = join("unpackHalf2x16(packHalf2x16(", to_expression(arg), ".zz)).x");
			op = join("vec3(", op0, ", ", op1, ")");
			break;
		}
		case 4:
		{
			auto op0 = join("unpackHalf2x16(packHalf2x16(", to_expression(arg), ".xy))");
			auto op1 = join("unpackHalf2x16(packHalf2x16(", to_expression(arg), ".zw))");
			op = join("vec4(", op0, ", ", op1, ")");
			break;
		}
		default:
			SPIRV_CROSS_THROW("Illegal argument to OpQuantizeToF16.");
		}

		emit_op(result_type, id, op, should_forward(arg));
		inherit_expression_dependencies(id, arg);
		break;
	}

	// Derivatives
	case OpDPdx:
		GLSL_UFOP(dFdx);
		if (is_legacy_es())
			require_extension_internal("GL_OES_standard_derivatives");
		register_control_dependent_expression(ops[1]);
		break;

	case OpDPdy:
		GLSL_UFOP(dFdy);
		if (is_legacy_es())
			require_extension_internal("GL_OES_standard_derivatives");
		register_control_dependent_expression(ops[1]);
		break;

	case OpDPdxFine:
		GLSL_UFOP(dFdxFine);
		if (options.es)
		{
			SPIRV_CROSS_THROW("GL_ARB_derivative_control is unavailable in OpenGL ES.");
		}
		if (options.version < 450)
			require_extension_internal("GL_ARB_derivative_control");
		register_control_dependent_expression(ops[1]);
		break;

	case OpDPdyFine:
		GLSL_UFOP(dFdyFine);
		if (options.es)
		{
			SPIRV_CROSS_THROW("GL_ARB_derivative_control is unavailable in OpenGL ES.");
		}
		if (options.version < 450)
			require_extension_internal("GL_ARB_derivative_control");
		register_control_dependent_expression(ops[1]);
		break;

	case OpDPdxCoarse:
		if (options.es)
		{
			SPIRV_CROSS_THROW("GL_ARB_derivative_control is unavailable in OpenGL ES.");
		}
		GLSL_UFOP(dFdxCoarse);
		if (options.version < 450)
			require_extension_internal("GL_ARB_derivative_control");
		register_control_dependent_expression(ops[1]);
		break;

	case OpDPdyCoarse:
		GLSL_UFOP(dFdyCoarse);
		if (options.es)
		{
			SPIRV_CROSS_THROW("GL_ARB_derivative_control is unavailable in OpenGL ES.");
		}
		if (options.version < 450)
			require_extension_internal("GL_ARB_derivative_control");
		register_control_dependent_expression(ops[1]);
		break;

	case OpFwidth:
		GLSL_UFOP(fwidth);
		if (is_legacy_es())
			require_extension_internal("GL_OES_standard_derivatives");
		register_control_dependent_expression(ops[1]);
		break;

	case OpFwidthCoarse:
		GLSL_UFOP(fwidthCoarse);
		if (options.es)
		{
			SPIRV_CROSS_THROW("GL_ARB_derivative_control is unavailable in OpenGL ES.");
		}
		if (options.version < 450)
			require_extension_internal("GL_ARB_derivative_control");
		register_control_dependent_expression(ops[1]);
		break;

	case OpFwidthFine:
		GLSL_UFOP(fwidthFine);
		if (options.es)
		{
			SPIRV_CROSS_THROW("GL_ARB_derivative_control is unavailable in OpenGL ES.");
		}
		if (options.version < 450)
			require_extension_internal("GL_ARB_derivative_control");
		register_control_dependent_expression(ops[1]);
		break;

	// Bitfield
	case OpBitFieldInsert:
	{
		emit_bitfield_insert_op(ops[0], ops[1], ops[2], ops[3], ops[4], ops[5], "bitfieldInsert", SPIRType::Int);
		break;
	}

	case OpBitFieldSExtract:
	{
		emit_trinary_func_op_bitextract(ops[0], ops[1], ops[2], ops[3], ops[4], "bitfieldExtract", int_type, int_type,
		                                SPIRType::Int, SPIRType::Int);
		break;
	}

	case OpBitFieldUExtract:
	{
		emit_trinary_func_op_bitextract(ops[0], ops[1], ops[2], ops[3], ops[4], "bitfieldExtract", uint_type, uint_type,
		                                SPIRType::Int, SPIRType::Int);
		break;
	}

	case OpBitReverse:
		// BitReverse does not have issues with sign since result type must match input type.
		GLSL_UFOP(bitfieldReverse);
		break;

	case OpBitCount:
	{
		auto basetype = expression_type(ops[2]).basetype;
		emit_unary_func_op_cast(ops[0], ops[1], ops[2], "bitCount", basetype, int_type);
		break;
	}

	// Atomics
	case OpAtomicExchange:
	{
		uint32_t result_type = ops[0];
		uint32_t id = ops[1];
		uint32_t ptr = ops[2];
		// Ignore semantics for now, probably only relevant to CL.
		uint32_t val = ops[5];
		const char *op = check_atomic_image(ptr) ? "imageAtomicExchange" : "atomicExchange";
		forced_temporaries.insert(id);
		emit_binary_func_op(result_type, id, ptr, val, op);
		flush_all_atomic_capable_variables();
		break;
	}

	case OpAtomicCompareExchange:
	{
		uint32_t result_type = ops[0];
		uint32_t id = ops[1];
		uint32_t ptr = ops[2];
		uint32_t val = ops[6];
		uint32_t comp = ops[7];
		const char *op = check_atomic_image(ptr) ? "imageAtomicCompSwap" : "atomicCompSwap";

		forced_temporaries.insert(id);
		emit_trinary_func_op(result_type, id, ptr, comp, val, op);
		flush_all_atomic_capable_variables();
		break;
	}

	case OpAtomicLoad:
		flush_all_atomic_capable_variables();
		// FIXME: Image?
		// OpAtomicLoad seems to only be relevant for atomic counters.
		forced_temporaries.insert(ops[1]);
		GLSL_UFOP(atomicCounter);
		break;

	case OpAtomicStore:
		SPIRV_CROSS_THROW("Unsupported opcode OpAtomicStore.");

	case OpAtomicIIncrement:
	case OpAtomicIDecrement:
	{
		forced_temporaries.insert(ops[1]);
		auto &type = expression_type(ops[2]);
		if (type.storage == StorageClassAtomicCounter)
		{
			// Legacy GLSL stuff, not sure if this is relevant to support.
			if (opcode == OpAtomicIIncrement)
				GLSL_UFOP(atomicCounterIncrement);
			else
				GLSL_UFOP(atomicCounterDecrement);
		}
		else
		{
			bool atomic_image = check_atomic_image(ops[2]);
			bool unsigned_type = (type.basetype == SPIRType::UInt) ||
			                     (atomic_image && get<SPIRType>(type.image.type).basetype == SPIRType::UInt);
			const char *op = atomic_image ? "imageAtomicAdd" : "atomicAdd";

			const char *increment = nullptr;
			if (opcode == OpAtomicIIncrement && unsigned_type)
				increment = "1u";
			else if (opcode == OpAtomicIIncrement)
				increment = "1";
			else if (unsigned_type)
				increment = "uint(-1)";
			else
				increment = "-1";

			emit_op(ops[0], ops[1], join(op, "(", to_expression(ops[2]), ", ", increment, ")"), false);
		}

		flush_all_atomic_capable_variables();
		break;
	}

	case OpAtomicIAdd:
	{
		const char *op = check_atomic_image(ops[2]) ? "imageAtomicAdd" : "atomicAdd";
		forced_temporaries.insert(ops[1]);
		emit_binary_func_op(ops[0], ops[1], ops[2], ops[5], op);
		flush_all_atomic_capable_variables();
		break;
	}

	case OpAtomicISub:
	{
		const char *op = check_atomic_image(ops[2]) ? "imageAtomicAdd" : "atomicAdd";
		forced_temporaries.insert(ops[1]);
		auto expr = join(op, "(", to_expression(ops[2]), ", -", to_enclosed_expression(ops[5]), ")");
		emit_op(ops[0], ops[1], expr, should_forward(ops[2]) && should_forward(ops[5]));
		flush_all_atomic_capable_variables();
		break;
	}

	case OpAtomicSMin:
	case OpAtomicUMin:
	{
		const char *op = check_atomic_image(ops[2]) ? "imageAtomicMin" : "atomicMin";
		forced_temporaries.insert(ops[1]);
		emit_binary_func_op(ops[0], ops[1], ops[2], ops[5], op);
		flush_all_atomic_capable_variables();
		break;
	}

	case OpAtomicSMax:
	case OpAtomicUMax:
	{
		const char *op = check_atomic_image(ops[2]) ? "imageAtomicMax" : "atomicMax";
		forced_temporaries.insert(ops[1]);
		emit_binary_func_op(ops[0], ops[1], ops[2], ops[5], op);
		flush_all_atomic_capable_variables();
		break;
	}

	case OpAtomicAnd:
	{
		const char *op = check_atomic_image(ops[2]) ? "imageAtomicAnd" : "atomicAnd";
		forced_temporaries.insert(ops[1]);
		emit_binary_func_op(ops[0], ops[1], ops[2], ops[5], op);
		flush_all_atomic_capable_variables();
		break;
	}

	case OpAtomicOr:
	{
		const char *op = check_atomic_image(ops[2]) ? "imageAtomicOr" : "atomicOr";
		forced_temporaries.insert(ops[1]);
		emit_binary_func_op(ops[0], ops[1], ops[2], ops[5], op);
		flush_all_atomic_capable_variables();
		break;
	}

	case OpAtomicXor:
	{
		const char *op = check_atomic_image(ops[2]) ? "imageAtomicXor" : "atomicXor";
		forced_temporaries.insert(ops[1]);
		emit_binary_func_op(ops[0], ops[1], ops[2], ops[5], op);
		flush_all_atomic_capable_variables();
		break;
	}

	// Geometry shaders
	case OpEmitVertex:
		statement("EmitVertex();");
		break;

	case OpEndPrimitive:
		statement("EndPrimitive();");
		break;

	case OpEmitStreamVertex:
	{
		if (options.es)
			SPIRV_CROSS_THROW("Multi-stream geometry shaders not supported in ES.");
		else if (!options.es && options.version < 400)
			SPIRV_CROSS_THROW("Multi-stream geometry shaders only supported in GLSL 400.");

		auto stream_expr = to_expression(ops[0]);
		if (expression_type(ops[0]).basetype != SPIRType::Int)
			stream_expr = join("int(", stream_expr, ")");
		statement("EmitStreamVertex(", stream_expr, ");");
		break;
	}

	case OpEndStreamPrimitive:
	{
		if (options.es)
			SPIRV_CROSS_THROW("Multi-stream geometry shaders not supported in ES.");
		else if (!options.es && options.version < 400)
			SPIRV_CROSS_THROW("Multi-stream geometry shaders only supported in GLSL 400.");

		auto stream_expr = to_expression(ops[0]);
		if (expression_type(ops[0]).basetype != SPIRType::Int)
			stream_expr = join("int(", stream_expr, ")");
		statement("EndStreamPrimitive(", stream_expr, ");");
		break;
	}

	// Textures
	case OpImageSampleExplicitLod:
	case OpImageSampleProjExplicitLod:
	case OpImageSampleDrefExplicitLod:
	case OpImageSampleProjDrefExplicitLod:
	case OpImageSampleImplicitLod:
	case OpImageSampleProjImplicitLod:
	case OpImageSampleDrefImplicitLod:
	case OpImageSampleProjDrefImplicitLod:
	case OpImageFetch:
	case OpImageGather:
	case OpImageDrefGather:
		// Gets a bit hairy, so move this to a separate instruction.
		emit_texture_op(instruction);
		break;

	case OpImage:
	{
		uint32_t result_type = ops[0];
		uint32_t id = ops[1];

		// Suppress usage tracking.
		auto &e = emit_op(result_type, id, to_expression(ops[2]), true, true);

		// When using the image, we need to know which variable it is actually loaded from.
		auto *var = maybe_get_backing_variable(ops[2]);
		e.loaded_from = var ? var->self : ID(0);
		break;
	}

	case OpImageQueryLod:
	{
		if (!options.es && options.version < 400)
		{
			require_extension_internal("GL_ARB_texture_query_lod");
			// For some reason, the ARB spec is all-caps.
			GLSL_BFOP(textureQueryLOD);
		}
		else if (options.es)
			SPIRV_CROSS_THROW("textureQueryLod not supported in ES profile.");
		else
			GLSL_BFOP(textureQueryLod);
		register_control_dependent_expression(ops[1]);
		break;
	}

	case OpImageQueryLevels:
	{
		uint32_t result_type = ops[0];
		uint32_t id = ops[1];

		if (!options.es && options.version < 430)
			require_extension_internal("GL_ARB_texture_query_levels");
		if (options.es)
			SPIRV_CROSS_THROW("textureQueryLevels not supported in ES profile.");

		auto expr = join("textureQueryLevels(", convert_separate_image_to_expression(ops[2]), ")");
		auto &restype = get<SPIRType>(ops[0]);
		expr = bitcast_expression(restype, SPIRType::Int, expr);
		emit_op(result_type, id, expr, true);
		break;
	}

	case OpImageQuerySamples:
	{
		auto &type = expression_type(ops[2]);
		uint32_t result_type = ops[0];
		uint32_t id = ops[1];

		string expr;
		if (type.image.sampled == 2)
			expr = join("imageSamples(", to_expression(ops[2]), ")");
		else
			expr = join("textureSamples(", convert_separate_image_to_expression(ops[2]), ")");

		auto &restype = get<SPIRType>(ops[0]);
		expr = bitcast_expression(restype, SPIRType::Int, expr);
		emit_op(result_type, id, expr, true);
		break;
	}

	case OpSampledImage:
	{
		uint32_t result_type = ops[0];
		uint32_t id = ops[1];
		emit_sampled_image_op(result_type, id, ops[2], ops[3]);
		inherit_expression_dependencies(id, ops[2]);
		inherit_expression_dependencies(id, ops[3]);
		break;
	}

	case OpImageQuerySizeLod:
	{
		uint32_t result_type = ops[0];
		uint32_t id = ops[1];

		auto expr = join("textureSize(", convert_separate_image_to_expression(ops[2]), ", ",
		                 bitcast_expression(SPIRType::Int, ops[3]), ")");
		auto &restype = get<SPIRType>(ops[0]);
		expr = bitcast_expression(restype, SPIRType::Int, expr);
		emit_op(result_type, id, expr, true);
		break;
	}

	// Image load/store
	case OpImageRead:
	{
		// We added Nonreadable speculatively to the OpImage variable due to glslangValidator
		// not adding the proper qualifiers.
		// If it turns out we need to read the image after all, remove the qualifier and recompile.
		auto *var = maybe_get_backing_variable(ops[2]);
		if (var)
		{
			auto &flags = ir.meta[var->self].decoration.decoration_flags;
			if (flags.get(DecorationNonReadable))
			{
				flags.clear(DecorationNonReadable);
				force_recompile();
			}
		}

		uint32_t result_type = ops[0];
		uint32_t id = ops[1];

		bool pure;
		string imgexpr;
		auto &type = expression_type(ops[2]);

		if (var && var->remapped_variable) // Remapped input, just read as-is without any op-code
		{
			if (type.image.ms)
				SPIRV_CROSS_THROW("Trying to remap multisampled image to variable, this is not possible.");

			auto itr =
			    find_if(begin(pls_inputs), end(pls_inputs), [var](const PlsRemap &pls) { return pls.id == var->self; });

			if (itr == end(pls_inputs))
			{
				// For non-PLS inputs, we rely on subpass type remapping information to get it right
				// since ImageRead always returns 4-component vectors and the backing type is opaque.
				if (!var->remapped_components)
					SPIRV_CROSS_THROW("subpassInput was remapped, but remap_components is not set correctly.");
				imgexpr = remap_swizzle(get<SPIRType>(result_type), var->remapped_components, to_expression(ops[2]));
			}
			else
			{
				// PLS input could have different number of components than what the SPIR expects, swizzle to
				// the appropriate vector size.
				uint32_t components = pls_format_to_components(itr->format);
				imgexpr = remap_swizzle(get<SPIRType>(result_type), components, to_expression(ops[2]));
			}
			pure = true;
		}
		else if (type.image.dim == DimSubpassData)
		{
			if (var && subpass_input_is_framebuffer_fetch(var->self))
			{
				imgexpr = to_expression(var->self);
			}
			else if (options.vulkan_semantics)
			{
				// With Vulkan semantics, use the proper Vulkan GLSL construct.
				if (type.image.ms)
				{
					uint32_t operands = ops[4];
					if (operands != ImageOperandsSampleMask || length != 6)
						SPIRV_CROSS_THROW(
						    "Multisampled image used in OpImageRead, but unexpected operand mask was used.");

					uint32_t samples = ops[5];
					imgexpr = join("subpassLoad(", to_expression(ops[2]), ", ", to_expression(samples), ")");
				}
				else
					imgexpr = join("subpassLoad(", to_expression(ops[2]), ")");
			}
			else
			{
				if (type.image.ms)
				{
					uint32_t operands = ops[4];
					if (operands != ImageOperandsSampleMask || length != 6)
						SPIRV_CROSS_THROW(
						    "Multisampled image used in OpImageRead, but unexpected operand mask was used.");

					uint32_t samples = ops[5];
					imgexpr = join("texelFetch(", to_expression(ops[2]), ", ivec2(gl_FragCoord.xy), ",
					               to_expression(samples), ")");
				}
				else
				{
					// Implement subpass loads via texture barrier style sampling.
					imgexpr = join("texelFetch(", to_expression(ops[2]), ", ivec2(gl_FragCoord.xy), 0)");
				}
			}
			imgexpr = remap_swizzle(get<SPIRType>(result_type), 4, imgexpr);
			pure = true;
		}
		else
		{
			// imageLoad only accepts int coords, not uint.
			auto coord_expr = to_expression(ops[3]);
			auto target_coord_type = expression_type(ops[3]);
			target_coord_type.basetype = SPIRType::Int;
			coord_expr = bitcast_expression(target_coord_type, expression_type(ops[3]).basetype, coord_expr);

			// Plain image load/store.
			if (type.image.ms)
			{
				uint32_t operands = ops[4];
				if (operands != ImageOperandsSampleMask || length != 6)
					SPIRV_CROSS_THROW("Multisampled image used in OpImageRead, but unexpected operand mask was used.");

				uint32_t samples = ops[5];
				imgexpr =
				    join("imageLoad(", to_expression(ops[2]), ", ", coord_expr, ", ", to_expression(samples), ")");
			}
			else
				imgexpr = join("imageLoad(", to_expression(ops[2]), ", ", coord_expr, ")");

			imgexpr = remap_swizzle(get<SPIRType>(result_type), 4, imgexpr);
			pure = false;
		}

		if (var && var->forwardable)
		{
			bool forward = forced_temporaries.find(id) == end(forced_temporaries);
			auto &e = emit_op(result_type, id, imgexpr, forward);

			// We only need to track dependencies if we're reading from image load/store.
			if (!pure)
			{
				e.loaded_from = var->self;
				if (forward)
					var->dependees.push_back(id);
			}
		}
		else
			emit_op(result_type, id, imgexpr, false);

		inherit_expression_dependencies(id, ops[2]);
		if (type.image.ms)
			inherit_expression_dependencies(id, ops[5]);
		break;
	}

	case OpImageTexelPointer:
	{
		uint32_t result_type = ops[0];
		uint32_t id = ops[1];

		auto coord_expr = to_expression(ops[3]);
		auto target_coord_type = expression_type(ops[3]);
		target_coord_type.basetype = SPIRType::Int;
		coord_expr = bitcast_expression(target_coord_type, expression_type(ops[3]).basetype, coord_expr);

		auto expr = join(to_expression(ops[2]), ", ", coord_expr);
		if (has_decoration(id, DecorationNonUniformEXT) || has_decoration(ops[2], DecorationNonUniformEXT))
			convert_non_uniform_expression(expression_type(ops[2]), expr);

		auto &e = set<SPIRExpression>(id, expr, result_type, true);

		// When using the pointer, we need to know which variable it is actually loaded from.
		auto *var = maybe_get_backing_variable(ops[2]);
		e.loaded_from = var ? var->self : ID(0);
		inherit_expression_dependencies(id, ops[3]);
		break;
	}

	case OpImageWrite:
	{
		// We added Nonwritable speculatively to the OpImage variable due to glslangValidator
		// not adding the proper qualifiers.
		// If it turns out we need to write to the image after all, remove the qualifier and recompile.
		auto *var = maybe_get_backing_variable(ops[0]);
		if (var)
		{
			auto &flags = ir.meta[var->self].decoration.decoration_flags;
			if (flags.get(DecorationNonWritable))
			{
				flags.clear(DecorationNonWritable);
				force_recompile();
			}
		}

		auto &type = expression_type(ops[0]);
		auto &value_type = expression_type(ops[2]);
		auto store_type = value_type;
		store_type.vecsize = 4;

		// imageStore only accepts int coords, not uint.
		auto coord_expr = to_expression(ops[1]);
		auto target_coord_type = expression_type(ops[1]);
		target_coord_type.basetype = SPIRType::Int;
		coord_expr = bitcast_expression(target_coord_type, expression_type(ops[1]).basetype, coord_expr);

		if (type.image.ms)
		{
			uint32_t operands = ops[3];
			if (operands != ImageOperandsSampleMask || length != 5)
				SPIRV_CROSS_THROW("Multisampled image used in OpImageWrite, but unexpected operand mask was used.");
			uint32_t samples = ops[4];
			statement("imageStore(", to_expression(ops[0]), ", ", coord_expr, ", ", to_expression(samples), ", ",
			          remap_swizzle(store_type, value_type.vecsize, to_expression(ops[2])), ");");
		}
		else
			statement("imageStore(", to_expression(ops[0]), ", ", coord_expr, ", ",
			          remap_swizzle(store_type, value_type.vecsize, to_expression(ops[2])), ");");

		if (var && variable_storage_is_aliased(*var))
			flush_all_aliased_variables();
		break;
	}

	case OpImageQuerySize:
	{
		auto &type = expression_type(ops[2]);
		uint32_t result_type = ops[0];
		uint32_t id = ops[1];

		if (type.basetype == SPIRType::Image)
		{
			string expr;
			if (type.image.sampled == 2)
			{
				// The size of an image is always constant.
				expr = join("imageSize(", to_expression(ops[2]), ")");
			}
			else
			{
				// This path is hit for samplerBuffers and multisampled images which do not have LOD.
				expr = join("textureSize(", convert_separate_image_to_expression(ops[2]), ")");
			}

			auto &restype = get<SPIRType>(ops[0]);
			expr = bitcast_expression(restype, SPIRType::Int, expr);
			emit_op(result_type, id, expr, true);
		}
		else
			SPIRV_CROSS_THROW("Invalid type for OpImageQuerySize.");
		break;
	}

	// Compute
	case OpControlBarrier:
	case OpMemoryBarrier:
	{
		uint32_t execution_scope = 0;
		uint32_t memory;
		uint32_t semantics;

		if (opcode == OpMemoryBarrier)
		{
			memory = get<SPIRConstant>(ops[0]).scalar();
			semantics = get<SPIRConstant>(ops[1]).scalar();
		}
		else
		{
			execution_scope = get<SPIRConstant>(ops[0]).scalar();
			memory = get<SPIRConstant>(ops[1]).scalar();
			semantics = get<SPIRConstant>(ops[2]).scalar();
		}

		if (execution_scope == ScopeSubgroup || memory == ScopeSubgroup)
		{
			if (!options.vulkan_semantics)
				SPIRV_CROSS_THROW("Can only use subgroup operations in Vulkan semantics.");
			require_extension_internal("GL_KHR_shader_subgroup_basic");
		}

		if (execution_scope != ScopeSubgroup && get_entry_point().model == ExecutionModelTessellationControl)
		{
			// Control shaders only have barriers, and it implies memory barriers.
			if (opcode == OpControlBarrier)
				statement("barrier();");
			break;
		}

		// We only care about these flags, acquire/release and friends are not relevant to GLSL.
		semantics = mask_relevant_memory_semantics(semantics);

		if (opcode == OpMemoryBarrier)
		{
			// If we are a memory barrier, and the next instruction is a control barrier, check if that memory barrier
			// does what we need, so we avoid redundant barriers.
			const Instruction *next = get_next_instruction_in_block(instruction);
			if (next && next->op == OpControlBarrier)
			{
				auto *next_ops = stream(*next);
				uint32_t next_memory = get<SPIRConstant>(next_ops[1]).scalar();
				uint32_t next_semantics = get<SPIRConstant>(next_ops[2]).scalar();
				next_semantics = mask_relevant_memory_semantics(next_semantics);

				bool memory_scope_covered = false;
				if (next_memory == memory)
					memory_scope_covered = true;
				else if (next_semantics == MemorySemanticsWorkgroupMemoryMask)
				{
					// If we only care about workgroup memory, either Device or Workgroup scope is fine,
					// scope does not have to match.
					if ((next_memory == ScopeDevice || next_memory == ScopeWorkgroup) &&
					    (memory == ScopeDevice || memory == ScopeWorkgroup))
					{
						memory_scope_covered = true;
					}
				}
				else if (memory == ScopeWorkgroup && next_memory == ScopeDevice)
				{
					// The control barrier has device scope, but the memory barrier just has workgroup scope.
					memory_scope_covered = true;
				}

				// If we have the same memory scope, and all memory types are covered, we're good.
				if (memory_scope_covered && (semantics & next_semantics) == semantics)
					break;
			}
		}

		// We are synchronizing some memory or syncing execution,
		// so we cannot forward any loads beyond the memory barrier.
		if (semantics || opcode == OpControlBarrier)
		{
			assert(current_emitting_block);
			flush_control_dependent_expressions(current_emitting_block->self);
			flush_all_active_variables();
		}

		if (memory == ScopeWorkgroup) // Only need to consider memory within a group
		{
			if (semantics == MemorySemanticsWorkgroupMemoryMask)
			{
				// OpControlBarrier implies a memory barrier for shared memory as well.
				bool implies_shared_barrier = opcode == OpControlBarrier && execution_scope == ScopeWorkgroup;
				if (!implies_shared_barrier)
					statement("memoryBarrierShared();");
			}
			else if (semantics != 0)
				statement("groupMemoryBarrier();");
		}
		else if (memory == ScopeSubgroup)
		{
			const uint32_t all_barriers =
			    MemorySemanticsWorkgroupMemoryMask | MemorySemanticsUniformMemoryMask | MemorySemanticsImageMemoryMask;

			if (semantics & (MemorySemanticsCrossWorkgroupMemoryMask | MemorySemanticsSubgroupMemoryMask))
			{
				// These are not relevant for GLSL, but assume it means memoryBarrier().
				// memoryBarrier() does everything, so no need to test anything else.
				statement("subgroupMemoryBarrier();");
			}
			else if ((semantics & all_barriers) == all_barriers)
			{
				// Short-hand instead of emitting 3 barriers.
				statement("subgroupMemoryBarrier();");
			}
			else
			{
				// Pick out individual barriers.
				if (semantics & MemorySemanticsWorkgroupMemoryMask)
					statement("subgroupMemoryBarrierShared();");
				if (semantics & MemorySemanticsUniformMemoryMask)
					statement("subgroupMemoryBarrierBuffer();");
				if (semantics & MemorySemanticsImageMemoryMask)
					statement("subgroupMemoryBarrierImage();");
			}
		}
		else
		{
			const uint32_t all_barriers =
			    MemorySemanticsWorkgroupMemoryMask | MemorySemanticsUniformMemoryMask | MemorySemanticsImageMemoryMask;

			if (semantics & (MemorySemanticsCrossWorkgroupMemoryMask | MemorySemanticsSubgroupMemoryMask))
			{
				// These are not relevant for GLSL, but assume it means memoryBarrier().
				// memoryBarrier() does everything, so no need to test anything else.
				statement("memoryBarrier();");
			}
			else if ((semantics & all_barriers) == all_barriers)
			{
				// Short-hand instead of emitting 4 barriers.
				statement("memoryBarrier();");
			}
			else
			{
				// Pick out individual barriers.
				if (semantics & MemorySemanticsWorkgroupMemoryMask)
					statement("memoryBarrierShared();");
				if (semantics & MemorySemanticsUniformMemoryMask)
					statement("memoryBarrierBuffer();");
				if (semantics & MemorySemanticsImageMemoryMask)
					statement("memoryBarrierImage();");
			}
		}

		if (opcode == OpControlBarrier)
		{
			if (execution_scope == ScopeSubgroup)
				statement("subgroupBarrier();");
			else
				statement("barrier();");
		}
		break;
	}

	case OpExtInst:
	{
		uint32_t extension_set = ops[2];

		if (get<SPIRExtension>(extension_set).ext == SPIRExtension::GLSL)
		{
			emit_glsl_op(ops[0], ops[1], ops[3], &ops[4], length - 4);
		}
		else if (get<SPIRExtension>(extension_set).ext == SPIRExtension::SPV_AMD_shader_ballot)
		{
			emit_spv_amd_shader_ballot_op(ops[0], ops[1], ops[3], &ops[4], length - 4);
		}
		else if (get<SPIRExtension>(extension_set).ext == SPIRExtension::SPV_AMD_shader_explicit_vertex_parameter)
		{
			emit_spv_amd_shader_explicit_vertex_parameter_op(ops[0], ops[1], ops[3], &ops[4], length - 4);
		}
		else if (get<SPIRExtension>(extension_set).ext == SPIRExtension::SPV_AMD_shader_trinary_minmax)
		{
			emit_spv_amd_shader_trinary_minmax_op(ops[0], ops[1], ops[3], &ops[4], length - 4);
		}
		else if (get<SPIRExtension>(extension_set).ext == SPIRExtension::SPV_AMD_gcn_shader)
		{
			emit_spv_amd_gcn_shader_op(ops[0], ops[1], ops[3], &ops[4], length - 4);
		}
		else if (get<SPIRExtension>(extension_set).ext == SPIRExtension::SPV_debug_info)
		{
			break; // Ignore SPIR-V debug information extended instructions.
		}
		else
		{
			statement("// unimplemented ext op ", instruction.op);
			break;
		}

		break;
	}

	// Legacy sub-group stuff ...
	case OpSubgroupBallotKHR:
	{
		uint32_t result_type = ops[0];
		uint32_t id = ops[1];
		string expr;
		expr = join("uvec4(unpackUint2x32(ballotARB(" + to_expression(ops[2]) + ")), 0u, 0u)");
		emit_op(result_type, id, expr, should_forward(ops[2]));

		require_extension_internal("GL_ARB_shader_ballot");
		inherit_expression_dependencies(id, ops[2]);
		register_control_dependent_expression(ops[1]);
		break;
	}

	case OpSubgroupFirstInvocationKHR:
	{
		uint32_t result_type = ops[0];
		uint32_t id = ops[1];
		emit_unary_func_op(result_type, id, ops[2], "readFirstInvocationARB");

		require_extension_internal("GL_ARB_shader_ballot");
		register_control_dependent_expression(ops[1]);
		break;
	}

	case OpSubgroupReadInvocationKHR:
	{
		uint32_t result_type = ops[0];
		uint32_t id = ops[1];
		emit_binary_func_op(result_type, id, ops[2], ops[3], "readInvocationARB");

		require_extension_internal("GL_ARB_shader_ballot");
		register_control_dependent_expression(ops[1]);
		break;
	}

	case OpSubgroupAllKHR:
	{
		uint32_t result_type = ops[0];
		uint32_t id = ops[1];
		emit_unary_func_op(result_type, id, ops[2], "allInvocationsARB");

		require_extension_internal("GL_ARB_shader_group_vote");
		register_control_dependent_expression(ops[1]);
		break;
	}

	case OpSubgroupAnyKHR:
	{
		uint32_t result_type = ops[0];
		uint32_t id = ops[1];
		emit_unary_func_op(result_type, id, ops[2], "anyInvocationARB");

		require_extension_internal("GL_ARB_shader_group_vote");
		register_control_dependent_expression(ops[1]);
		break;
	}

	case OpSubgroupAllEqualKHR:
	{
		uint32_t result_type = ops[0];
		uint32_t id = ops[1];
		emit_unary_func_op(result_type, id, ops[2], "allInvocationsEqualARB");

		require_extension_internal("GL_ARB_shader_group_vote");
		register_control_dependent_expression(ops[1]);
		break;
	}

	case OpGroupIAddNonUniformAMD:
	case OpGroupFAddNonUniformAMD:
	{
		uint32_t result_type = ops[0];
		uint32_t id = ops[1];
		emit_unary_func_op(result_type, id, ops[4], "addInvocationsNonUniformAMD");

		require_extension_internal("GL_AMD_shader_ballot");
		register_control_dependent_expression(ops[1]);
		break;
	}

	case OpGroupFMinNonUniformAMD:
	case OpGroupUMinNonUniformAMD:
	case OpGroupSMinNonUniformAMD:
	{
		uint32_t result_type = ops[0];
		uint32_t id = ops[1];
		emit_unary_func_op(result_type, id, ops[4], "minInvocationsNonUniformAMD");

		require_extension_internal("GL_AMD_shader_ballot");
		register_control_dependent_expression(ops[1]);
		break;
	}

	case OpGroupFMaxNonUniformAMD:
	case OpGroupUMaxNonUniformAMD:
	case OpGroupSMaxNonUniformAMD:
	{
		uint32_t result_type = ops[0];
		uint32_t id = ops[1];
		emit_unary_func_op(result_type, id, ops[4], "maxInvocationsNonUniformAMD");

		require_extension_internal("GL_AMD_shader_ballot");
		register_control_dependent_expression(ops[1]);
		break;
	}

	case OpFragmentMaskFetchAMD:
	{
		auto &type = expression_type(ops[2]);
		uint32_t result_type = ops[0];
		uint32_t id = ops[1];

		if (type.image.dim == spv::DimSubpassData)
		{
			emit_unary_func_op(result_type, id, ops[2], "fragmentMaskFetchAMD");
		}
		else
		{
			emit_binary_func_op(result_type, id, ops[2], ops[3], "fragmentMaskFetchAMD");
		}

		require_extension_internal("GL_AMD_shader_fragment_mask");
		break;
	}

	case OpFragmentFetchAMD:
	{
		auto &type = expression_type(ops[2]);
		uint32_t result_type = ops[0];
		uint32_t id = ops[1];

		if (type.image.dim == spv::DimSubpassData)
		{
			emit_binary_func_op(result_type, id, ops[2], ops[4], "fragmentFetchAMD");
		}
		else
		{
			emit_trinary_func_op(result_type, id, ops[2], ops[3], ops[4], "fragmentFetchAMD");
		}

		require_extension_internal("GL_AMD_shader_fragment_mask");
		break;
	}

	// Vulkan 1.1 sub-group stuff ...
	case OpGroupNonUniformElect:
	case OpGroupNonUniformBroadcast:
	case OpGroupNonUniformBroadcastFirst:
	case OpGroupNonUniformBallot:
	case OpGroupNonUniformInverseBallot:
	case OpGroupNonUniformBallotBitExtract:
	case OpGroupNonUniformBallotBitCount:
	case OpGroupNonUniformBallotFindLSB:
	case OpGroupNonUniformBallotFindMSB:
	case OpGroupNonUniformShuffle:
	case OpGroupNonUniformShuffleXor:
	case OpGroupNonUniformShuffleUp:
	case OpGroupNonUniformShuffleDown:
	case OpGroupNonUniformAll:
	case OpGroupNonUniformAny:
	case OpGroupNonUniformAllEqual:
	case OpGroupNonUniformFAdd:
	case OpGroupNonUniformIAdd:
	case OpGroupNonUniformFMul:
	case OpGroupNonUniformIMul:
	case OpGroupNonUniformFMin:
	case OpGroupNonUniformFMax:
	case OpGroupNonUniformSMin:
	case OpGroupNonUniformSMax:
	case OpGroupNonUniformUMin:
	case OpGroupNonUniformUMax:
	case OpGroupNonUniformBitwiseAnd:
	case OpGroupNonUniformBitwiseOr:
	case OpGroupNonUniformBitwiseXor:
	case OpGroupNonUniformQuadSwap:
	case OpGroupNonUniformQuadBroadcast:
		emit_subgroup_op(instruction);
		break;

	case OpFUnordEqual:
	case OpFUnordNotEqual:
	case OpFUnordLessThan:
	case OpFUnordGreaterThan:
	case OpFUnordLessThanEqual:
	case OpFUnordGreaterThanEqual:
	{
		// GLSL doesn't specify if floating point comparisons are ordered or unordered,
		// but glslang always emits ordered floating point compares for GLSL.
		// To get unordered compares, we can test the opposite thing and invert the result.
		// This way, we force true when there is any NaN present.
		uint32_t op0 = ops[2];
		uint32_t op1 = ops[3];

		string expr;
		if (expression_type(op0).vecsize > 1)
		{
			const char *comp_op = nullptr;
			switch (opcode)
			{
			case OpFUnordEqual:
				comp_op = "notEqual";
				break;

			case OpFUnordNotEqual:
				comp_op = "equal";
				break;

			case OpFUnordLessThan:
				comp_op = "greaterThanEqual";
				break;

			case OpFUnordLessThanEqual:
				comp_op = "greaterThan";
				break;

			case OpFUnordGreaterThan:
				comp_op = "lessThanEqual";
				break;

			case OpFUnordGreaterThanEqual:
				comp_op = "lessThan";
				break;

			default:
				assert(0);
				break;
			}

			expr = join("not(", comp_op, "(", to_unpacked_expression(op0), ", ", to_unpacked_expression(op1), "))");
		}
		else
		{
			const char *comp_op = nullptr;
			switch (opcode)
			{
			case OpFUnordEqual:
				comp_op = " != ";
				break;

			case OpFUnordNotEqual:
				comp_op = " == ";
				break;

			case OpFUnordLessThan:
				comp_op = " >= ";
				break;

			case OpFUnordLessThanEqual:
				comp_op = " > ";
				break;

			case OpFUnordGreaterThan:
				comp_op = " <= ";
				break;

			case OpFUnordGreaterThanEqual:
				comp_op = " < ";
				break;

			default:
				assert(0);
				break;
			}

			expr = join("!(", to_enclosed_unpacked_expression(op0), comp_op, to_enclosed_unpacked_expression(op1), ")");
		}

		emit_op(ops[0], ops[1], expr, should_forward(op0) && should_forward(op1));
		inherit_expression_dependencies(ops[1], op0);
		inherit_expression_dependencies(ops[1], op1);
		break;
	}

	case OpReportIntersectionNV:
		statement("reportIntersectionNV(", to_expression(ops[0]), ", ", to_expression(ops[1]), ");");
		break;
	case OpIgnoreIntersectionNV:
		statement("ignoreIntersectionNV();");
		break;
	case OpTerminateRayNV:
		statement("terminateRayNV();");
		break;
	case OpTraceNV:
		statement("traceNV(", to_expression(ops[0]), ", ", to_expression(ops[1]), ", ", to_expression(ops[2]), ", ",
		          to_expression(ops[3]), ", ", to_expression(ops[4]), ", ", to_expression(ops[5]), ", ",
		          to_expression(ops[6]), ", ", to_expression(ops[7]), ", ", to_expression(ops[8]), ", ",
		          to_expression(ops[9]), ", ", to_expression(ops[10]), ");");
		break;
	case OpExecuteCallableNV:
		statement("executeCallableNV(", to_expression(ops[0]), ", ", to_expression(ops[1]), ");");
		break;

	case OpConvertUToPtr:
	{
		auto &type = get<SPIRType>(ops[0]);
		if (type.storage != StorageClassPhysicalStorageBufferEXT)
			SPIRV_CROSS_THROW("Only StorageClassPhysicalStorageBufferEXT is supported by OpConvertUToPtr.");

		auto op = type_to_glsl(type);
		emit_unary_func_op(ops[0], ops[1], ops[2], op.c_str());
		break;
	}

	case OpConvertPtrToU:
	{
		auto &type = get<SPIRType>(ops[0]);
		auto &ptr_type = expression_type(ops[2]);
		if (ptr_type.storage != StorageClassPhysicalStorageBufferEXT)
			SPIRV_CROSS_THROW("Only StorageClassPhysicalStorageBufferEXT is supported by OpConvertPtrToU.");

		auto op = type_to_glsl(type);
		emit_unary_func_op(ops[0], ops[1], ops[2], op.c_str());
		break;
	}

	case OpUndef:
		// Undefined value has been declared.
		break;

	case OpLine:
	{
		emit_line_directive(ops[0], ops[1]);
		break;
	}

	case OpNoLine:
		break;

	case OpDemoteToHelperInvocationEXT:
		if (!options.vulkan_semantics)
			SPIRV_CROSS_THROW("GL_EXT_demote_to_helper_invocation is only supported in Vulkan GLSL.");
		require_extension_internal("GL_EXT_demote_to_helper_invocation");
		statement(backend.demote_literal, ";");
		break;

	case OpIsHelperInvocationEXT:
		if (!options.vulkan_semantics)
			SPIRV_CROSS_THROW("GL_EXT_demote_to_helper_invocation is only supported in Vulkan GLSL.");
		require_extension_internal("GL_EXT_demote_to_helper_invocation");
		emit_op(ops[0], ops[1], "helperInvocationEXT()", false);
		break;

	case OpBeginInvocationInterlockEXT:
		// If the interlock is complex, we emit this elsewhere.
		if (!interlocked_is_complex)
		{
			if (options.es)
				statement("beginInvocationInterlockNV();");
			else
				statement("beginInvocationInterlockARB();");

			flush_all_active_variables();
			// Make sure forwarding doesn't propagate outside interlock region.
		}
		break;

	case OpEndInvocationInterlockEXT:
		// If the interlock is complex, we emit this elsewhere.
		if (!interlocked_is_complex)
		{
			if (options.es)
				statement("endInvocationInterlockNV();");
			else
				statement("endInvocationInterlockARB();");

			flush_all_active_variables();
			// Make sure forwarding doesn't propagate outside interlock region.
		}
		break;

	default:
		statement("// unimplemented op ", instruction.op);
		break;
	}
}

// Appends function arguments, mapped from global variables, beyond the specified arg index.
// This is used when a function call uses fewer arguments than the function defines.
// This situation may occur if the function signature has been dynamically modified to
// extract global variables referenced from within the function, and convert them to
// function arguments. This is necessary for shader languages that do not support global
// access to shader input content from within a function (eg. Metal). Each additional
// function args uses the name of the global variable. Function nesting will modify the
// functions and function calls all the way up the nesting chain.
void CompilerGLSL::append_global_func_args(const SPIRFunction &func, uint32_t index, SmallVector<string> &arglist)
{
	auto &args = func.arguments;
	uint32_t arg_cnt = uint32_t(args.size());
	for (uint32_t arg_idx = index; arg_idx < arg_cnt; arg_idx++)
	{
		auto &arg = args[arg_idx];
		assert(arg.alias_global_variable);

		// If the underlying variable needs to be declared
		// (ie. a local variable with deferred declaration), do so now.
		uint32_t var_id = get<SPIRVariable>(arg.id).basevariable;
		if (var_id)
			flush_variable_declaration(var_id);

		arglist.push_back(to_func_call_arg(arg, arg.id));
	}
}

string CompilerGLSL::to_member_name(const SPIRType &type, uint32_t index)
{
	if (type.type_alias != TypeID(0) &&
	    !has_extended_decoration(type.type_alias, SPIRVCrossDecorationBufferBlockRepacked))
	{
		return to_member_name(get<SPIRType>(type.type_alias), index);
	}

	auto &memb = ir.meta[type.self].members;
	if (index < memb.size() && !memb[index].alias.empty())
		return memb[index].alias;
	else
		return join("_m", index);
}

string CompilerGLSL::to_member_reference(uint32_t, const SPIRType &type, uint32_t index, bool)
{
	return join(".", to_member_name(type, index));
}

void CompilerGLSL::add_member_name(SPIRType &type, uint32_t index)
{
	auto &memb = ir.meta[type.self].members;
	if (index < memb.size() && !memb[index].alias.empty())
	{
		auto &name = memb[index].alias;
		if (name.empty())
			return;

		// Reserved for temporaries.
		if (name[0] == '_' && name.size() >= 2 && isdigit(name[1]))
		{
			name.clear();
			return;
		}

		update_name_cache(type.member_name_cache, name);
	}
}

// Checks whether the ID is a row_major matrix that requires conversion before use
bool CompilerGLSL::is_non_native_row_major_matrix(uint32_t id)
{
	// Natively supported row-major matrices do not need to be converted.
	// Legacy targets do not support row major.
	if (backend.native_row_major_matrix && !is_legacy())
		return false;

	// Non-matrix or column-major matrix types do not need to be converted.
	if (!has_decoration(id, DecorationRowMajor))
		return false;

	// Only square row-major matrices can be converted at this time.
	// Converting non-square matrices will require defining custom GLSL function that
	// swaps matrix elements while retaining the original dimensional form of the matrix.
	const auto type = expression_type(id);
	if (type.columns != type.vecsize)
		SPIRV_CROSS_THROW("Row-major matrices must be square on this platform.");

	return true;
}

// Checks whether the member is a row_major matrix that requires conversion before use
bool CompilerGLSL::member_is_non_native_row_major_matrix(const SPIRType &type, uint32_t index)
{
	// Natively supported row-major matrices do not need to be converted.
	if (backend.native_row_major_matrix && !is_legacy())
		return false;

	// Non-matrix or column-major matrix types do not need to be converted.
	if (!has_member_decoration(type.self, index, DecorationRowMajor))
		return false;

	// Only square row-major matrices can be converted at this time.
	// Converting non-square matrices will require defining custom GLSL function that
	// swaps matrix elements while retaining the original dimensional form of the matrix.
	const auto mbr_type = get<SPIRType>(type.member_types[index]);
	if (mbr_type.columns != mbr_type.vecsize)
		SPIRV_CROSS_THROW("Row-major matrices must be square on this platform.");

	return true;
}

// Checks if we need to remap physical type IDs when declaring the type in a buffer.
bool CompilerGLSL::member_is_remapped_physical_type(const SPIRType &type, uint32_t index) const
{
	return has_extended_member_decoration(type.self, index, SPIRVCrossDecorationPhysicalTypeID);
}

// Checks whether the member is in packed data type, that might need to be unpacked.
bool CompilerGLSL::member_is_packed_physical_type(const SPIRType &type, uint32_t index) const
{
	return has_extended_member_decoration(type.self, index, SPIRVCrossDecorationPhysicalTypePacked);
}

// Wraps the expression string in a function call that converts the
// row_major matrix result of the expression to a column_major matrix.
// Base implementation uses the standard library transpose() function.
// Subclasses may override to use a different function.
string CompilerGLSL::convert_row_major_matrix(string exp_str, const SPIRType &exp_type, uint32_t /* physical_type_id */,
                                              bool /*is_packed*/)
{
	strip_enclosed_expression(exp_str);
	if (!is_matrix(exp_type))
	{
		auto column_index = exp_str.find_last_of('[');
		if (column_index == string::npos)
			return exp_str;

		auto column_expr = exp_str.substr(column_index);
		exp_str.resize(column_index);

		auto transposed_expr = type_to_glsl_constructor(exp_type) + "(";

		// Loading a column from a row-major matrix. Unroll the load.
		for (uint32_t c = 0; c < exp_type.vecsize; c++)
		{
			transposed_expr += join(exp_str, '[', c, ']', column_expr);
			if (c + 1 < exp_type.vecsize)
				transposed_expr += ", ";
		}

		transposed_expr += ")";
		return transposed_expr;
	}
	else
		return join("transpose(", exp_str, ")");
}

string CompilerGLSL::variable_decl(const SPIRType &type, const string &name, uint32_t id)
{
	string type_name = type_to_glsl(type, id);
	remap_variable_type_name(type, name, type_name);
	return join(type_name, " ", name, type_to_array_glsl(type));
}

// Emit a structure member. Subclasses may override to modify output,
// or to dynamically add a padding member if needed.
void CompilerGLSL::emit_struct_member(const SPIRType &type, uint32_t member_type_id, uint32_t index,
                                      const string &qualifier, uint32_t)
{
	auto &membertype = get<SPIRType>(member_type_id);

	Bitset memberflags;
	auto &memb = ir.meta[type.self].members;
	if (index < memb.size())
		memberflags = memb[index].decoration_flags;

	string qualifiers;
	bool is_block = ir.meta[type.self].decoration.decoration_flags.get(DecorationBlock) ||
	                ir.meta[type.self].decoration.decoration_flags.get(DecorationBufferBlock);

	if (is_block)
		qualifiers = to_interpolation_qualifiers(memberflags);

	statement(layout_for_member(type, index), qualifiers, qualifier, flags_to_qualifiers_glsl(membertype, memberflags),
	          variable_decl(membertype, to_member_name(type, index)), ";");
}

void CompilerGLSL::emit_struct_padding_target(const SPIRType &)
{
}

const char *CompilerGLSL::flags_to_qualifiers_glsl(const SPIRType &type, const Bitset &flags)
{
	// GL_EXT_buffer_reference variables can be marked as restrict.
	if (flags.get(DecorationRestrictPointerEXT))
		return "restrict ";

	// Structs do not have precision qualifiers, neither do doubles (desktop only anyways, so no mediump/highp).
	if (type.basetype != SPIRType::Float && type.basetype != SPIRType::Int && type.basetype != SPIRType::UInt &&
	    type.basetype != SPIRType::Image && type.basetype != SPIRType::SampledImage &&
	    type.basetype != SPIRType::Sampler)
		return "";

	if (options.es)
	{
		auto &execution = get_entry_point();

		if (flags.get(DecorationRelaxedPrecision))
		{
			bool implied_fmediump = type.basetype == SPIRType::Float &&
			                        options.fragment.default_float_precision == Options::Mediump &&
			                        execution.model == ExecutionModelFragment;

			bool implied_imediump = (type.basetype == SPIRType::Int || type.basetype == SPIRType::UInt) &&
			                        options.fragment.default_int_precision == Options::Mediump &&
			                        execution.model == ExecutionModelFragment;

			return implied_fmediump || implied_imediump ? "" : "mediump ";
		}
		else
		{
			bool implied_fhighp =
			    type.basetype == SPIRType::Float && ((options.fragment.default_float_precision == Options::Highp &&
			                                          execution.model == ExecutionModelFragment) ||
			                                         (execution.model != ExecutionModelFragment));

			bool implied_ihighp = (type.basetype == SPIRType::Int || type.basetype == SPIRType::UInt) &&
			                      ((options.fragment.default_int_precision == Options::Highp &&
			                        execution.model == ExecutionModelFragment) ||
			                       (execution.model != ExecutionModelFragment));

			return implied_fhighp || implied_ihighp ? "" : "highp ";
		}
	}
	else if (backend.allow_precision_qualifiers)
	{
		// Vulkan GLSL supports precision qualifiers, even in desktop profiles, which is convenient.
		// The default is highp however, so only emit mediump in the rare case that a shader has these.
		if (flags.get(DecorationRelaxedPrecision))
			return "mediump ";
		else
			return "";
	}
	else
		return "";
}

const char *CompilerGLSL::to_precision_qualifiers_glsl(uint32_t id)
{
	auto &type = expression_type(id);
	bool use_precision_qualifiers = backend.allow_precision_qualifiers || options.es;
	if (use_precision_qualifiers && (type.basetype == SPIRType::Image || type.basetype == SPIRType::SampledImage))
	{
		// Force mediump for the sampler type. We cannot declare 16-bit or smaller image types.
		auto &result_type = get<SPIRType>(type.image.type);
		if (result_type.width < 32)
			return "mediump ";
	}
	return flags_to_qualifiers_glsl(type, ir.meta[id].decoration.decoration_flags);
}

string CompilerGLSL::to_qualifiers_glsl(uint32_t id)
{
	auto &flags = ir.meta[id].decoration.decoration_flags;
	string res;

	auto *var = maybe_get<SPIRVariable>(id);

	if (var && var->storage == StorageClassWorkgroup && !backend.shared_is_implied)
		res += "shared ";

	res += to_interpolation_qualifiers(flags);
	if (var)
		res += to_storage_qualifiers_glsl(*var);

	auto &type = expression_type(id);
	if (type.image.dim != DimSubpassData && type.image.sampled == 2)
	{
		if (flags.get(DecorationCoherent))
			res += "coherent ";
		if (flags.get(DecorationRestrict))
			res += "restrict ";
		if (flags.get(DecorationNonWritable))
			res += "readonly ";
		if (flags.get(DecorationNonReadable))
			res += "writeonly ";
	}

	res += to_precision_qualifiers_glsl(id);

	return res;
}

string CompilerGLSL::argument_decl(const SPIRFunction::Parameter &arg)
{
	// glslangValidator seems to make all arguments pointer no matter what which is rather bizarre ...
	auto &type = expression_type(arg.id);
	const char *direction = "";

	if (type.pointer)
	{
		if (arg.write_count && arg.read_count)
			direction = "inout ";
		else if (arg.write_count)
			direction = "out ";
	}

	return join(direction, to_qualifiers_glsl(arg.id), variable_decl(type, to_name(arg.id), arg.id));
}

string CompilerGLSL::to_initializer_expression(const SPIRVariable &var)
{
	return to_expression(var.initializer);
}

string CompilerGLSL::to_zero_initialized_expression(uint32_t type_id)
{
#ifndef NDEBUG
	auto &type = get<SPIRType>(type_id);
	assert(type.storage == StorageClassPrivate || type.storage == StorageClassFunction ||
	       type.storage == StorageClassGeneric);
#endif
	uint32_t id = ir.increase_bound_by(1);
	ir.make_constant_null(id, type_id, false);
	return constant_expression(get<SPIRConstant>(id));
}

bool CompilerGLSL::type_can_zero_initialize(const SPIRType &type) const
{
	if (type.pointer)
		return false;

	if (!type.array.empty() && options.flatten_multidimensional_arrays)
		return false;

	for (auto &literal : type.array_size_literal)
		if (!literal)
			return false;

	for (auto &memb : type.member_types)
		if (!type_can_zero_initialize(get<SPIRType>(memb)))
			return false;

	return true;
}

string CompilerGLSL::variable_decl(const SPIRVariable &variable)
{
	// Ignore the pointer type since GLSL doesn't have pointers.
	auto &type = get_variable_data_type(variable);

	if (type.pointer_depth > 1)
		SPIRV_CROSS_THROW("Cannot declare pointer-to-pointer types.");

	auto res = join(to_qualifiers_glsl(variable.self), variable_decl(type, to_name(variable.self), variable.self));

	if (variable.loop_variable && variable.static_expression)
	{
		uint32_t expr = variable.static_expression;
		if (ir.ids[expr].get_type() != TypeUndef)
			res += join(" = ", to_expression(variable.static_expression));
		else if (options.force_zero_initialized_variables && type_can_zero_initialize(type))
			res += join(" = ", to_zero_initialized_expression(get_variable_data_type_id(variable)));
	}
	else if (variable.initializer)
	{
		uint32_t expr = variable.initializer;
		if (ir.ids[expr].get_type() != TypeUndef)
			res += join(" = ", to_initializer_expression(variable));
		else if (options.force_zero_initialized_variables && type_can_zero_initialize(type))
			res += join(" = ", to_zero_initialized_expression(get_variable_data_type_id(variable)));
	}

	return res;
}

const char *CompilerGLSL::to_pls_qualifiers_glsl(const SPIRVariable &variable)
{
	auto &flags = ir.meta[variable.self].decoration.decoration_flags;
	if (flags.get(DecorationRelaxedPrecision))
		return "mediump ";
	else
		return "highp ";
}

string CompilerGLSL::pls_decl(const PlsRemap &var)
{
	auto &variable = get<SPIRVariable>(var.id);

	SPIRType type;
	type.vecsize = pls_format_to_components(var.format);
	type.basetype = pls_format_to_basetype(var.format);

	return join(to_pls_layout(var.format), to_pls_qualifiers_glsl(variable), type_to_glsl(type), " ",
	            to_name(variable.self));
}

uint32_t CompilerGLSL::to_array_size_literal(const SPIRType &type) const
{
	return to_array_size_literal(type, uint32_t(type.array.size() - 1));
}

uint32_t CompilerGLSL::to_array_size_literal(const SPIRType &type, uint32_t index) const
{
	assert(type.array.size() == type.array_size_literal.size());

	if (type.array_size_literal[index])
	{
		return type.array[index];
	}
	else
	{
		// Use the default spec constant value.
		// This is the best we can do.
		uint32_t array_size_id = type.array[index];

		// Explicitly check for this case. The error message you would get (bad cast) makes no sense otherwise.
		if (ir.ids[array_size_id].get_type() == TypeConstantOp)
			SPIRV_CROSS_THROW("An array size was found to be an OpSpecConstantOp. This is not supported since "
			                  "SPIRV-Cross cannot deduce the actual size here.");

		uint32_t array_size = get<SPIRConstant>(array_size_id).scalar();
		return array_size;
	}
}

string CompilerGLSL::to_array_size(const SPIRType &type, uint32_t index)
{
	assert(type.array.size() == type.array_size_literal.size());

	auto &size = type.array[index];
	if (!type.array_size_literal[index])
		return to_expression(size);
	else if (size)
		return convert_to_string(size);
	else if (!backend.unsized_array_supported)
	{
		// For runtime-sized arrays, we can work around
		// lack of standard support for this by simply having
		// a single element array.
		//
		// Runtime length arrays must always be the last element
		// in an interface block.
		return "1";
	}
	else
		return "";
}

string CompilerGLSL::type_to_array_glsl(const SPIRType &type)
{
	if (type.pointer && type.storage == StorageClassPhysicalStorageBufferEXT && type.basetype != SPIRType::Struct)
	{
		// We are using a wrapped pointer type, and we should not emit any array declarations here.
		return "";
	}

	if (type.array.empty())
		return "";

	if (options.flatten_multidimensional_arrays)
	{
		string res;
		res += "[";
		for (auto i = uint32_t(type.array.size()); i; i--)
		{
			res += enclose_expression(to_array_size(type, i - 1));
			if (i > 1)
				res += " * ";
		}
		res += "]";
		return res;
	}
	else
	{
		if (type.array.size() > 1)
		{
			if (!options.es && options.version < 430)
				require_extension_internal("GL_ARB_arrays_of_arrays");
			else if (options.es && options.version < 310)
				SPIRV_CROSS_THROW("Arrays of arrays not supported before ESSL version 310. "
				                  "Try using --flatten-multidimensional-arrays or set "
				                  "options.flatten_multidimensional_arrays to true.");
		}

		string res;
		for (auto i = uint32_t(type.array.size()); i; i--)
		{
			res += "[";
			res += to_array_size(type, i - 1);
			res += "]";
		}
		return res;
	}
}

string CompilerGLSL::image_type_glsl(const SPIRType &type, uint32_t id)
{
	auto &imagetype = get<SPIRType>(type.image.type);
	string res;

	switch (imagetype.basetype)
	{
	case SPIRType::Int:
	case SPIRType::Short:
	case SPIRType::SByte:
		res = "i";
		break;
	case SPIRType::UInt:
	case SPIRType::UShort:
	case SPIRType::UByte:
		res = "u";
		break;
	default:
		break;
	}

	// For half image types, we will force mediump for the sampler, and cast to f16 after any sampling operation.
	// We cannot express a true half texture type in GLSL. Neither for short integer formats for that matter.

	if (type.basetype == SPIRType::Image && type.image.dim == DimSubpassData && options.vulkan_semantics)
		return res + "subpassInput" + (type.image.ms ? "MS" : "");
	else if (type.basetype == SPIRType::Image && type.image.dim == DimSubpassData &&
	         subpass_input_is_framebuffer_fetch(id))
	{
		SPIRType sampled_type = get<SPIRType>(type.image.type);
		sampled_type.vecsize = 4;
		return type_to_glsl(sampled_type);
	}

	// If we're emulating subpassInput with samplers, force sampler2D
	// so we don't have to specify format.
	if (type.basetype == SPIRType::Image && type.image.dim != DimSubpassData)
	{
		// Sampler buffers are always declared as samplerBuffer even though they might be separate images in the SPIR-V.
		if (type.image.dim == DimBuffer && type.image.sampled == 1)
			res += "sampler";
		else
			res += type.image.sampled == 2 ? "image" : "texture";
	}
	else
		res += "sampler";

	switch (type.image.dim)
	{
	case Dim1D:
		res += "1D";
		break;
	case Dim2D:
		res += "2D";
		break;
	case Dim3D:
		res += "3D";
		break;
	case DimCube:
		res += "Cube";
		break;
	case DimRect:
		if (options.es)
			SPIRV_CROSS_THROW("Rectangle textures are not supported on OpenGL ES.");

		if (is_legacy_desktop())
			require_extension_internal("GL_ARB_texture_rectangle");

		res += "2DRect";
		break;

	case DimBuffer:
		if (options.es && options.version < 320)
			require_extension_internal("GL_OES_texture_buffer");
		else if (!options.es && options.version < 300)
			require_extension_internal("GL_EXT_texture_buffer_object");
		res += "Buffer";
		break;

	case DimSubpassData:
		res += "2D";
		break;
	default:
		SPIRV_CROSS_THROW("Only 1D, 2D, 2DRect, 3D, Buffer, InputTarget and Cube textures supported.");
	}

	if (type.image.ms)
		res += "MS";
	if (type.image.arrayed)
	{
		if (is_legacy_desktop())
			require_extension_internal("GL_EXT_texture_array");
		res += "Array";
	}

	// "Shadow" state in GLSL only exists for samplers and combined image samplers.
	if (((type.basetype == SPIRType::SampledImage) || (type.basetype == SPIRType::Sampler)) &&
	    image_is_comparison(type, id))
	{
		res += "Shadow";
	}

	return res;
}

string CompilerGLSL::type_to_glsl_constructor(const SPIRType &type)
{
	if (backend.use_array_constructor && type.array.size() > 1)
	{
		if (options.flatten_multidimensional_arrays)
			SPIRV_CROSS_THROW("Cannot flatten constructors of multidimensional array constructors, e.g. float[][]().");
		else if (!options.es && options.version < 430)
			require_extension_internal("GL_ARB_arrays_of_arrays");
		else if (options.es && options.version < 310)
			SPIRV_CROSS_THROW("Arrays of arrays not supported before ESSL version 310.");
	}

	auto e = type_to_glsl(type);
	if (backend.use_array_constructor)
	{
		for (uint32_t i = 0; i < type.array.size(); i++)
			e += "[]";
	}
	return e;
}

// The optional id parameter indicates the object whose type we are trying
// to find the description for. It is optional. Most type descriptions do not
// depend on a specific object's use of that type.
string CompilerGLSL::type_to_glsl(const SPIRType &type, uint32_t id)
{
	if (type.pointer && type.storage == StorageClassPhysicalStorageBufferEXT && type.basetype != SPIRType::Struct)
	{
		// Need to create a magic type name which compacts the entire type information.
		string name = type_to_glsl(get_pointee_type(type));
		for (size_t i = 0; i < type.array.size(); i++)
		{
			if (type.array_size_literal[i])
				name += join(type.array[i], "_");
			else
				name += join("id", type.array[i], "_");
		}
		name += "Pointer";
		return name;
	}

	switch (type.basetype)
	{
	case SPIRType::Struct:
		// Need OpName lookup here to get a "sensible" name for a struct.
		if (backend.explicit_struct_type)
			return join("struct ", to_name(type.self));
		else
			return to_name(type.self);

	case SPIRType::Image:
	case SPIRType::SampledImage:
		return image_type_glsl(type, id);

	case SPIRType::Sampler:
		// The depth field is set by calling code based on the variable ID of the sampler, effectively reintroducing
		// this distinction into the type system.
		return comparison_ids.count(id) ? "samplerShadow" : "sampler";

	case SPIRType::AccelerationStructureNV:
		return "accelerationStructureNV";

	case SPIRType::Void:
		return "void";

	default:
		break;
	}

	if (type.basetype == SPIRType::UInt && is_legacy())
		SPIRV_CROSS_THROW("Unsigned integers are not supported on legacy targets.");

	if (type.vecsize == 1 && type.columns == 1) // Scalar builtin
	{
		switch (type.basetype)
		{
		case SPIRType::Boolean:
			return "bool";
		case SPIRType::SByte:
			return backend.basic_int8_type;
		case SPIRType::UByte:
			return backend.basic_uint8_type;
		case SPIRType::Short:
			return backend.basic_int16_type;
		case SPIRType::UShort:
			return backend.basic_uint16_type;
		case SPIRType::Int:
			return backend.basic_int_type;
		case SPIRType::UInt:
			return backend.basic_uint_type;
		case SPIRType::AtomicCounter:
			return "atomic_uint";
		case SPIRType::Half:
			return "float16_t";
		case SPIRType::Float:
			return "float";
		case SPIRType::Double:
			return "double";
		case SPIRType::Int64:
			return "int64_t";
		case SPIRType::UInt64:
			return "uint64_t";
		default:
			return "???";
		}
	}
	else if (type.vecsize > 1 && type.columns == 1) // Vector builtin
	{
		switch (type.basetype)
		{
		case SPIRType::Boolean:
			return join("bvec", type.vecsize);
		case SPIRType::SByte:
			return join("i8vec", type.vecsize);
		case SPIRType::UByte:
			return join("u8vec", type.vecsize);
		case SPIRType::Short:
			return join("i16vec", type.vecsize);
		case SPIRType::UShort:
			return join("u16vec", type.vecsize);
		case SPIRType::Int:
			return join("ivec", type.vecsize);
		case SPIRType::UInt:
			return join("uvec", type.vecsize);
		case SPIRType::Half:
			return join("f16vec", type.vecsize);
		case SPIRType::Float:
			return join("vec", type.vecsize);
		case SPIRType::Double:
			return join("dvec", type.vecsize);
		case SPIRType::Int64:
			return join("i64vec", type.vecsize);
		case SPIRType::UInt64:
			return join("u64vec", type.vecsize);
		default:
			return "???";
		}
	}
	else if (type.vecsize == type.columns) // Simple Matrix builtin
	{
		switch (type.basetype)
		{
		case SPIRType::Boolean:
			return join("bmat", type.vecsize);
		case SPIRType::Int:
			return join("imat", type.vecsize);
		case SPIRType::UInt:
			return join("umat", type.vecsize);
		case SPIRType::Half:
			return join("f16mat", type.vecsize);
		case SPIRType::Float:
			return join("mat", type.vecsize);
		case SPIRType::Double:
			return join("dmat", type.vecsize);
		// Matrix types not supported for int64/uint64.
		default:
			return "???";
		}
	}
	else
	{
		switch (type.basetype)
		{
		case SPIRType::Boolean:
			return join("bmat", type.columns, "x", type.vecsize);
		case SPIRType::Int:
			return join("imat", type.columns, "x", type.vecsize);
		case SPIRType::UInt:
			return join("umat", type.columns, "x", type.vecsize);
		case SPIRType::Half:
			return join("f16mat", type.columns, "x", type.vecsize);
		case SPIRType::Float:
			return join("mat", type.columns, "x", type.vecsize);
		case SPIRType::Double:
			return join("dmat", type.columns, "x", type.vecsize);
		// Matrix types not supported for int64/uint64.
		default:
			return "???";
		}
	}
}

void CompilerGLSL::add_variable(unordered_set<string> &variables_primary,
                                const unordered_set<string> &variables_secondary, string &name)
{
	if (name.empty())
		return;

	// Reserved for temporaries.
	if (name[0] == '_' && name.size() >= 2 && isdigit(name[1]))
	{
		name.clear();
		return;
	}

	// Avoid double underscores.
	name = sanitize_underscores(name);

	update_name_cache(variables_primary, variables_secondary, name);
}

void CompilerGLSL::add_local_variable_name(uint32_t id)
{
	add_variable(local_variable_names, block_names, ir.meta[id].decoration.alias);
}

void CompilerGLSL::add_resource_name(uint32_t id)
{
	add_variable(resource_names, block_names, ir.meta[id].decoration.alias);
}

void CompilerGLSL::add_header_line(const std::string &line)
{
	header_lines.push_back(line);
}

bool CompilerGLSL::has_extension(const std::string &ext) const
{
	auto itr = find(begin(forced_extensions), end(forced_extensions), ext);
	return itr != end(forced_extensions);
}

void CompilerGLSL::require_extension(const std::string &ext)
{
	if (!has_extension(ext))
		forced_extensions.push_back(ext);
}

void CompilerGLSL::require_extension_internal(const string &ext)
{
	if (backend.supports_extensions && !has_extension(ext))
	{
		forced_extensions.push_back(ext);
		force_recompile();
	}
}

void CompilerGLSL::flatten_buffer_block(VariableID id)
{
	auto &var = get<SPIRVariable>(id);
	auto &type = get<SPIRType>(var.basetype);
	auto name = to_name(type.self, false);
	auto &flags = ir.meta[type.self].decoration.decoration_flags;

	if (!type.array.empty())
		SPIRV_CROSS_THROW(name + " is an array of UBOs.");
	if (type.basetype != SPIRType::Struct)
		SPIRV_CROSS_THROW(name + " is not a struct.");
	if (!flags.get(DecorationBlock))
		SPIRV_CROSS_THROW(name + " is not a block.");
	if (type.member_types.empty())
		SPIRV_CROSS_THROW(name + " is an empty struct.");

	flattened_buffer_blocks.insert(id);
}

bool CompilerGLSL::builtin_translates_to_nonarray(spv::BuiltIn /*builtin*/) const
{
	return false; // GLSL itself does not need to translate array builtin types to non-array builtin types
}

bool CompilerGLSL::check_atomic_image(uint32_t id)
{
	auto &type = expression_type(id);
	if (type.storage == StorageClassImage)
	{
		if (options.es && options.version < 320)
			require_extension_internal("GL_OES_shader_image_atomic");

		auto *var = maybe_get_backing_variable(id);
		if (var)
		{
			auto &flags = ir.meta[var->self].decoration.decoration_flags;
			if (flags.get(DecorationNonWritable) || flags.get(DecorationNonReadable))
			{
				flags.clear(DecorationNonWritable);
				flags.clear(DecorationNonReadable);
				force_recompile();
			}
		}
		return true;
	}
	else
		return false;
}

void CompilerGLSL::add_function_overload(const SPIRFunction &func)
{
	Hasher hasher;
	for (auto &arg : func.arguments)
	{
		// Parameters can vary with pointer type or not,
		// but that will not change the signature in GLSL/HLSL,
		// so strip the pointer type before hashing.
		uint32_t type_id = get_pointee_type_id(arg.type);
		auto &type = get<SPIRType>(type_id);

		if (!combined_image_samplers.empty())
		{
			// If we have combined image samplers, we cannot really trust the image and sampler arguments
			// we pass down to callees, because they may be shuffled around.
			// Ignore these arguments, to make sure that functions need to differ in some other way
			// to be considered different overloads.
			if (type.basetype == SPIRType::SampledImage ||
			    (type.basetype == SPIRType::Image && type.image.sampled == 1) || type.basetype == SPIRType::Sampler)
			{
				continue;
			}
		}

		hasher.u32(type_id);
	}
	uint64_t types_hash = hasher.get();

	auto function_name = to_name(func.self);
	auto itr = function_overloads.find(function_name);
	if (itr != end(function_overloads))
	{
		// There exists a function with this name already.
		auto &overloads = itr->second;
		if (overloads.count(types_hash) != 0)
		{
			// Overload conflict, assign a new name.
			add_resource_name(func.self);
			function_overloads[to_name(func.self)].insert(types_hash);
		}
		else
		{
			// Can reuse the name.
			overloads.insert(types_hash);
		}
	}
	else
	{
		// First time we see this function name.
		add_resource_name(func.self);
		function_overloads[to_name(func.self)].insert(types_hash);
	}
}

void CompilerGLSL::emit_function_prototype(SPIRFunction &func, const Bitset &return_flags)
{
	if (func.self != ir.default_entry_point)
		add_function_overload(func);

	// Avoid shadow declarations.
	local_variable_names = resource_names;

	string decl;

	auto &type = get<SPIRType>(func.return_type);
	decl += flags_to_qualifiers_glsl(type, return_flags);
	decl += type_to_glsl(type);
	decl += type_to_array_glsl(type);
	decl += " ";

	if (func.self == ir.default_entry_point)
	{
		// If we need complex fallback in GLSL, we just wrap main() in a function
		// and interlock the entire shader ...
		if (interlocked_is_complex)
			decl += "spvMainInterlockedBody";
		else
			decl += "main";

		processing_entry_point = true;
	}
	else
		decl += to_name(func.self);

	decl += "(";
	SmallVector<string> arglist;
	for (auto &arg : func.arguments)
	{
		// Do not pass in separate images or samplers if we're remapping
		// to combined image samplers.
		if (skip_argument(arg.id))
			continue;

		// Might change the variable name if it already exists in this function.
		// SPIRV OpName doesn't have any semantic effect, so it's valid for an implementation
		// to use same name for variables.
		// Since we want to make the GLSL debuggable and somewhat sane, use fallback names for variables which are duplicates.
		add_local_variable_name(arg.id);

		arglist.push_back(argument_decl(arg));

		// Hold a pointer to the parameter so we can invalidate the readonly field if needed.
		auto *var = maybe_get<SPIRVariable>(arg.id);
		if (var)
			var->parameter = &arg;
	}

	for (auto &arg : func.shadow_arguments)
	{
		// Might change the variable name if it already exists in this function.
		// SPIRV OpName doesn't have any semantic effect, so it's valid for an implementation
		// to use same name for variables.
		// Since we want to make the GLSL debuggable and somewhat sane, use fallback names for variables which are duplicates.
		add_local_variable_name(arg.id);

		arglist.push_back(argument_decl(arg));

		// Hold a pointer to the parameter so we can invalidate the readonly field if needed.
		auto *var = maybe_get<SPIRVariable>(arg.id);
		if (var)
			var->parameter = &arg;
	}

	decl += merge(arglist);
	decl += ")";
	statement(decl);
}

void CompilerGLSL::emit_function(SPIRFunction &func, const Bitset &return_flags)
{
	// Avoid potential cycles.
	if (func.active)
		return;
	func.active = true;

	// If we depend on a function, emit that function before we emit our own function.
	for (auto block : func.blocks)
	{
		auto &b = get<SPIRBlock>(block);
		for (auto &i : b.ops)
		{
			auto ops = stream(i);
			auto op = static_cast<Op>(i.op);

			if (op == OpFunctionCall)
			{
				// Recursively emit functions which are called.
				uint32_t id = ops[2];
				emit_function(get<SPIRFunction>(id), ir.meta[ops[1]].decoration.decoration_flags);
			}
		}
	}

	if (func.entry_line.file_id != 0)
		emit_line_directive(func.entry_line.file_id, func.entry_line.line_literal);
	emit_function_prototype(func, return_flags);
	begin_scope();

	if (func.self == ir.default_entry_point)
		emit_entry_point_declarations();

	current_function = &func;
	auto &entry_block = get<SPIRBlock>(func.entry_block);

	sort(begin(func.constant_arrays_needed_on_stack), end(func.constant_arrays_needed_on_stack));
	for (auto &array : func.constant_arrays_needed_on_stack)
	{
		auto &c = get<SPIRConstant>(array);
		auto &type = get<SPIRType>(c.constant_type);
		statement(variable_decl(type, join("_", array, "_array_copy")), " = ", constant_expression(c), ";");
	}

	for (auto &v : func.local_variables)
	{
		auto &var = get<SPIRVariable>(v);
		var.deferred_declaration = false;

		if (var.storage == StorageClassWorkgroup)
		{
			// Special variable type which cannot have initializer,
			// need to be declared as standalone variables.
			// Comes from MSL which can push global variables as local variables in main function.
			add_local_variable_name(var.self);
			statement(variable_decl(var), ";");
			var.deferred_declaration = false;
		}
		else if (var.storage == StorageClassPrivate)
		{
			// These variables will not have had their CFG usage analyzed, so move it to the entry block.
			// Comes from MSL which can push global variables as local variables in main function.
			// We could just declare them right now, but we would miss out on an important initialization case which is
			// LUT declaration in MSL.
			// If we don't declare the variable when it is assigned we're forced to go through a helper function
			// which copies elements one by one.
			add_local_variable_name(var.self);
			auto &dominated = entry_block.dominated_variables;
			if (find(begin(dominated), end(dominated), var.self) == end(dominated))
				entry_block.dominated_variables.push_back(var.self);
			var.deferred_declaration = true;
		}
		else if (var.storage == StorageClassFunction && var.remapped_variable && var.static_expression)
		{
			// No need to declare this variable, it has a static expression.
			var.deferred_declaration = false;
		}
		else if (expression_is_lvalue(v))
		{
			add_local_variable_name(var.self);

			if (var.initializer)
				statement(variable_decl_function_local(var), ";");
			else
			{
				// Don't declare variable until first use to declutter the GLSL output quite a lot.
				// If we don't touch the variable before first branch,
				// declare it then since we need variable declaration to be in top scope.
				var.deferred_declaration = true;
			}
		}
		else
		{
			// HACK: SPIR-V in older glslang output likes to use samplers and images as local variables, but GLSL does not allow this.
			// For these types (non-lvalue), we enforce forwarding through a shadowed variable.
			// This means that when we OpStore to these variables, we just write in the expression ID directly.
			// This breaks any kind of branching, since the variable must be statically assigned.
			// Branching on samplers and images would be pretty much impossible to fake in GLSL.
			var.statically_assigned = true;
		}

		var.loop_variable_enable = false;

		// Loop variables are never declared outside their for-loop, so block any implicit declaration.
		if (var.loop_variable)
			var.deferred_declaration = false;
	}

	// Enforce declaration order for regression testing purposes.
	for (auto &block_id : func.blocks)
	{
		auto &block = get<SPIRBlock>(block_id);
		sort(begin(block.dominated_variables), end(block.dominated_variables));
	}

	for (auto &line : current_function->fixup_hooks_in)
		line();

	emit_block_chain(entry_block);

	end_scope();
	processing_entry_point = false;
	statement("");

	// Make sure deferred declaration state for local variables is cleared when we are done with function.
	// We risk declaring Private/Workgroup variables in places we are not supposed to otherwise.
	for (auto &v : func.local_variables)
	{
		auto &var = get<SPIRVariable>(v);
		var.deferred_declaration = false;
	}
}

void CompilerGLSL::emit_fixup()
{
	auto &execution = get_entry_point();
	if (execution.model == ExecutionModelVertex)
	{
		if (options.vertex.fixup_clipspace)
		{
			const char *suffix = backend.float_literal_suffix ? "f" : "";
			statement("gl_Position.z = 2.0", suffix, " * gl_Position.z - gl_Position.w;");
		}

		if (options.vertex.flip_vert_y)
			statement("gl_Position.y = -gl_Position.y;");
	}
}

void CompilerGLSL::flush_phi(BlockID from, BlockID to)
{
	auto &child = get<SPIRBlock>(to);
	if (child.ignore_phi_from_block == from)
		return;

	unordered_set<uint32_t> temporary_phi_variables;

	for (auto itr = begin(child.phi_variables); itr != end(child.phi_variables); ++itr)
	{
		auto &phi = *itr;

		if (phi.parent == from)
		{
			auto &var = get<SPIRVariable>(phi.function_variable);

			// A Phi variable might be a loop variable, so flush to static expression.
			if (var.loop_variable && !var.loop_variable_enable)
				var.static_expression = phi.local_variable;
			else
			{
				flush_variable_declaration(phi.function_variable);

				// Check if we are going to write to a Phi variable that another statement will read from
				// as part of another Phi node in our target block.
				// For this case, we will need to copy phi.function_variable to a temporary, and use that for future reads.
				// This is judged to be extremely rare, so deal with it here using a simple, but suboptimal algorithm.
				bool need_saved_temporary =
				    find_if(itr + 1, end(child.phi_variables), [&](const SPIRBlock::Phi &future_phi) -> bool {
					    return future_phi.local_variable == ID(phi.function_variable) && future_phi.parent == from;
				    }) != end(child.phi_variables);

				if (need_saved_temporary)
				{
					// Need to make sure we declare the phi variable with a copy at the right scope.
					// We cannot safely declare a temporary here since we might be inside a continue block.
					if (!var.allocate_temporary_copy)
					{
						var.allocate_temporary_copy = true;
						force_recompile();
					}
					statement("_", phi.function_variable, "_copy", " = ", to_name(phi.function_variable), ";");
					temporary_phi_variables.insert(phi.function_variable);
				}

				// This might be called in continue block, so make sure we
				// use this to emit ESSL 1.0 compliant increments/decrements.
				auto lhs = to_expression(phi.function_variable);

				string rhs;
				if (temporary_phi_variables.count(phi.local_variable))
					rhs = join("_", phi.local_variable, "_copy");
				else
					rhs = to_pointer_expression(phi.local_variable);

				if (!optimize_read_modify_write(get<SPIRType>(var.basetype), lhs, rhs))
					statement(lhs, " = ", rhs, ";");
			}

			register_write(phi.function_variable);
		}
	}
}

void CompilerGLSL::branch_to_continue(BlockID from, BlockID to)
{
	auto &to_block = get<SPIRBlock>(to);
	if (from == to)
		return;

	assert(is_continue(to));
	if (to_block.complex_continue)
	{
		// Just emit the whole block chain as is.
		auto usage_counts = expression_usage_counts;

		emit_block_chain(to_block);

		// Expression usage counts are moot after returning from the continue block.
		expression_usage_counts = usage_counts;
	}
	else
	{
		auto &from_block = get<SPIRBlock>(from);
		bool outside_control_flow = false;
		uint32_t loop_dominator = 0;

		// FIXME: Refactor this to not use the old loop_dominator tracking.
		if (from_block.merge_block)
		{
			// If we are a loop header, we don't set the loop dominator,
			// so just use "self" here.
			loop_dominator = from;
		}
		else if (from_block.loop_dominator != BlockID(SPIRBlock::NoDominator))
		{
			loop_dominator = from_block.loop_dominator;
		}

		if (loop_dominator != 0)
		{
			auto &cfg = get_cfg_for_current_function();

			// For non-complex continue blocks, we implicitly branch to the continue block
			// by having the continue block be part of the loop header in for (; ; continue-block).
			outside_control_flow = cfg.node_terminates_control_flow_in_sub_graph(loop_dominator, from);
		}

		// Some simplification for for-loops. We always end up with a useless continue;
		// statement since we branch to a loop block.
		// Walk the CFG, if we unconditionally execute the block calling continue assuming we're in the loop block,
		// we can avoid writing out an explicit continue statement.
		// Similar optimization to return statements if we know we're outside flow control.
		if (!outside_control_flow)
			statement("continue;");
	}
}

void CompilerGLSL::branch(BlockID from, BlockID to)
{
	flush_phi(from, to);
	flush_control_dependent_expressions(from);

	bool to_is_continue = is_continue(to);

	// This is only a continue if we branch to our loop dominator.
	if ((ir.block_meta[to] & ParsedIR::BLOCK_META_LOOP_HEADER_BIT) != 0 && get<SPIRBlock>(from).loop_dominator == to)
	{
		// This can happen if we had a complex continue block which was emitted.
		// Once the continue block tries to branch to the loop header, just emit continue;
		// and end the chain here.
		statement("continue;");
	}
	else if (is_break(to))
	{
		// Very dirty workaround.
		// Switch constructs are able to break, but they cannot break out of a loop at the same time.
		// Only sensible solution is to make a ladder variable, which we declare at the top of the switch block,
		// write to the ladder here, and defer the break.
		// The loop we're breaking out of must dominate the switch block, or there is no ladder breaking case.
		if (current_emitting_switch && is_loop_break(to) &&
		    current_emitting_switch->loop_dominator != BlockID(SPIRBlock::NoDominator) &&
		    get<SPIRBlock>(current_emitting_switch->loop_dominator).merge_block == to)
		{
			if (!current_emitting_switch->need_ladder_break)
			{
				force_recompile();
				current_emitting_switch->need_ladder_break = true;
			}

			statement("_", current_emitting_switch->self, "_ladder_break = true;");
		}
		statement("break;");
	}
	else if (to_is_continue || from == to)
	{
		// For from == to case can happen for a do-while loop which branches into itself.
		// We don't mark these cases as continue blocks, but the only possible way to branch into
		// ourselves is through means of continue blocks.

		// If we are merging to a continue block, there is no need to emit the block chain for continue here.
		// We can branch to the continue block after we merge execution.

		// Here we make use of structured control flow rules from spec:
		// 2.11: - the merge block declared by a header block cannot be a merge block declared by any other header block
		//       - each header block must strictly dominate its merge block, unless the merge block is unreachable in the CFG
		// If we are branching to a merge block, we must be inside a construct which dominates the merge block.
		auto &block_meta = ir.block_meta[to];
		bool branching_to_merge =
		    (block_meta & (ParsedIR::BLOCK_META_SELECTION_MERGE_BIT | ParsedIR::BLOCK_META_MULTISELECT_MERGE_BIT |
		                   ParsedIR::BLOCK_META_LOOP_MERGE_BIT)) != 0;
		if (!to_is_continue || !branching_to_merge)
			branch_to_continue(from, to);
	}
	else if (!is_conditional(to))
		emit_block_chain(get<SPIRBlock>(to));

	// It is important that we check for break before continue.
	// A block might serve two purposes, a break block for the inner scope, and
	// a continue block in the outer scope.
	// Inner scope always takes precedence.
}

void CompilerGLSL::branch(BlockID from, uint32_t cond, BlockID true_block, BlockID false_block)
{
	auto &from_block = get<SPIRBlock>(from);
	BlockID merge_block = from_block.merge == SPIRBlock::MergeSelection ? from_block.next_block : BlockID(0);

	// If we branch directly to a selection merge target, we don't need a code path.
	// This covers both merge out of if () / else () as well as a break for switch blocks.
	bool true_sub = !is_conditional(true_block);
	bool false_sub = !is_conditional(false_block);

	bool true_block_is_selection_merge = true_block == merge_block;
	bool false_block_is_selection_merge = false_block == merge_block;

	if (true_sub)
	{
		emit_block_hints(get<SPIRBlock>(from));
		statement("if (", to_expression(cond), ")");
		begin_scope();
		branch(from, true_block);
		end_scope();

		// If we merge to continue, we handle that explicitly in emit_block_chain(),
		// so there is no need to branch to it directly here.
		// break; is required to handle ladder fallthrough cases, so keep that in for now, even
		// if we could potentially handle it in emit_block_chain().
		if (false_sub || (!false_block_is_selection_merge && is_continue(false_block)) || is_break(false_block))
		{
			statement("else");
			begin_scope();
			branch(from, false_block);
			end_scope();
		}
		else if (flush_phi_required(from, false_block))
		{
			statement("else");
			begin_scope();
			flush_phi(from, false_block);
			end_scope();
		}
	}
	else if (false_sub)
	{
		// Only need false path, use negative conditional.
		emit_block_hints(get<SPIRBlock>(from));
		statement("if (!", to_enclosed_expression(cond), ")");
		begin_scope();
		branch(from, false_block);
		end_scope();

		if ((!true_block_is_selection_merge && is_continue(true_block)) || is_break(true_block))
		{
			statement("else");
			begin_scope();
			branch(from, true_block);
			end_scope();
		}
		else if (flush_phi_required(from, true_block))
		{
			statement("else");
			begin_scope();
			flush_phi(from, true_block);
			end_scope();
		}
	}
}

// FIXME: This currently cannot handle complex continue blocks
// as in do-while.
// This should be seen as a "trivial" continue block.
string CompilerGLSL::emit_continue_block(uint32_t continue_block, bool follow_true_block, bool follow_false_block)
{
	auto *block = &get<SPIRBlock>(continue_block);

	// While emitting the continue block, declare_temporary will check this
	// if we have to emit temporaries.
	current_continue_block = block;

	SmallVector<string> statements;

	// Capture all statements into our list.
	auto *old = redirect_statement;
	redirect_statement = &statements;

	// Stamp out all blocks one after each other.
	while ((ir.block_meta[block->self] & ParsedIR::BLOCK_META_LOOP_HEADER_BIT) == 0)
	{
		// Write out all instructions we have in this block.
		emit_block_instructions(*block);

		// For plain branchless for/while continue blocks.
		if (block->next_block)
		{
			flush_phi(continue_block, block->next_block);
			block = &get<SPIRBlock>(block->next_block);
		}
		// For do while blocks. The last block will be a select block.
		else if (block->true_block && follow_true_block)
		{
			flush_phi(continue_block, block->true_block);
			block = &get<SPIRBlock>(block->true_block);
		}
		else if (block->false_block && follow_false_block)
		{
			flush_phi(continue_block, block->false_block);
			block = &get<SPIRBlock>(block->false_block);
		}
		else
		{
			SPIRV_CROSS_THROW("Invalid continue block detected!");
		}
	}

	// Restore old pointer.
	redirect_statement = old;

	// Somewhat ugly, strip off the last ';' since we use ',' instead.
	// Ideally, we should select this behavior in statement().
	for (auto &s : statements)
	{
		if (!s.empty() && s.back() == ';')
			s.erase(s.size() - 1, 1);
	}

	current_continue_block = nullptr;
	return merge(statements);
}

void CompilerGLSL::emit_while_loop_initializers(const SPIRBlock &block)
{
	// While loops do not take initializers, so declare all of them outside.
	for (auto &loop_var : block.loop_variables)
	{
		auto &var = get<SPIRVariable>(loop_var);
		statement(variable_decl(var), ";");
	}
}

string CompilerGLSL::emit_for_loop_initializers(const SPIRBlock &block)
{
	if (block.loop_variables.empty())
		return "";

	bool same_types = for_loop_initializers_are_same_type(block);
	// We can only declare for loop initializers if all variables are of same type.
	// If we cannot do this, declare individual variables before the loop header.

	// We might have a loop variable candidate which was not assigned to for some reason.
	uint32_t missing_initializers = 0;
	for (auto &variable : block.loop_variables)
	{
		uint32_t expr = get<SPIRVariable>(variable).static_expression;

		// Sometimes loop variables are initialized with OpUndef, but we can just declare
		// a plain variable without initializer in this case.
		if (expr == 0 || ir.ids[expr].get_type() == TypeUndef)
			missing_initializers++;
	}

	if (block.loop_variables.size() == 1 && missing_initializers == 0)
	{
		return variable_decl(get<SPIRVariable>(block.loop_variables.front()));
	}
	else if (!same_types || missing_initializers == uint32_t(block.loop_variables.size()))
	{
		for (auto &loop_var : block.loop_variables)
			statement(variable_decl(get<SPIRVariable>(loop_var)), ";");
		return "";
	}
	else
	{
		// We have a mix of loop variables, either ones with a clear initializer, or ones without.
		// Separate the two streams.
		string expr;

		for (auto &loop_var : block.loop_variables)
		{
			uint32_t static_expr = get<SPIRVariable>(loop_var).static_expression;
			if (static_expr == 0 || ir.ids[static_expr].get_type() == TypeUndef)
			{
				statement(variable_decl(get<SPIRVariable>(loop_var)), ";");
			}
			else
			{
				auto &var = get<SPIRVariable>(loop_var);
				auto &type = get_variable_data_type(var);
				if (expr.empty())
				{
					// For loop initializers are of the form <type id = value, id = value, id = value, etc ...
					expr = join(to_qualifiers_glsl(var.self), type_to_glsl(type), " ");
				}
				else
				{
					expr += ", ";
					// In MSL, being based on C++, the asterisk marking a pointer
					// binds to the identifier, not the type.
					if (type.pointer)
						expr += "* ";
				}

				expr += join(to_name(loop_var), " = ", to_pointer_expression(var.static_expression));
			}
		}
		return expr;
	}
}

bool CompilerGLSL::for_loop_initializers_are_same_type(const SPIRBlock &block)
{
	if (block.loop_variables.size() <= 1)
		return true;

	uint32_t expected = 0;
	Bitset expected_flags;
	for (auto &var : block.loop_variables)
	{
		// Don't care about uninitialized variables as they will not be part of the initializers.
		uint32_t expr = get<SPIRVariable>(var).static_expression;
		if (expr == 0 || ir.ids[expr].get_type() == TypeUndef)
			continue;

		if (expected == 0)
		{
			expected = get<SPIRVariable>(var).basetype;
			expected_flags = get_decoration_bitset(var);
		}
		else if (expected != get<SPIRVariable>(var).basetype)
			return false;

		// Precision flags and things like that must also match.
		if (expected_flags != get_decoration_bitset(var))
			return false;
	}

	return true;
}

bool CompilerGLSL::attempt_emit_loop_header(SPIRBlock &block, SPIRBlock::Method method)
{
	SPIRBlock::ContinueBlockType continue_type = continue_block_type(get<SPIRBlock>(block.continue_block));

	if (method == SPIRBlock::MergeToSelectForLoop || method == SPIRBlock::MergeToSelectContinueForLoop)
	{
		uint32_t current_count = statement_count;
		// If we're trying to create a true for loop,
		// we need to make sure that all opcodes before branch statement do not actually emit any code.
		// We can then take the condition expression and create a for (; cond ; ) { body; } structure instead.
		emit_block_instructions(block);

		bool condition_is_temporary = forced_temporaries.find(block.condition) == end(forced_temporaries);

		// This can work! We only did trivial things which could be forwarded in block body!
		if (current_count == statement_count && condition_is_temporary)
		{
			switch (continue_type)
			{
			case SPIRBlock::ForLoop:
			{
				// This block may be a dominating block, so make sure we flush undeclared variables before building the for loop header.
				flush_undeclared_variables(block);

				// Important that we do this in this order because
				// emitting the continue block can invalidate the condition expression.
				auto initializer = emit_for_loop_initializers(block);
				auto condition = to_expression(block.condition);

				// Condition might have to be inverted.
				if (execution_is_noop(get<SPIRBlock>(block.true_block), get<SPIRBlock>(block.merge_block)))
					condition = join("!", enclose_expression(condition));

				emit_block_hints(block);
				if (method != SPIRBlock::MergeToSelectContinueForLoop)
				{
					auto continue_block = emit_continue_block(block.continue_block, false, false);
					statement("for (", initializer, "; ", condition, "; ", continue_block, ")");
				}
				else
					statement("for (", initializer, "; ", condition, "; )");
				break;
			}

			case SPIRBlock::WhileLoop:
			{
				// This block may be a dominating block, so make sure we flush undeclared variables before building the while loop header.
				flush_undeclared_variables(block);
				emit_while_loop_initializers(block);
				emit_block_hints(block);

				auto condition = to_expression(block.condition);
				// Condition might have to be inverted.
				if (execution_is_noop(get<SPIRBlock>(block.true_block), get<SPIRBlock>(block.merge_block)))
					condition = join("!", enclose_expression(condition));

				statement("while (", condition, ")");
				break;
			}

			default:
				block.disable_block_optimization = true;
				force_recompile();
				begin_scope(); // We'll see an end_scope() later.
				return false;
			}

			begin_scope();
			return true;
		}
		else
		{
			block.disable_block_optimization = true;
			force_recompile();
			begin_scope(); // We'll see an end_scope() later.
			return false;
		}
	}
	else if (method == SPIRBlock::MergeToDirectForLoop)
	{
		auto &child = get<SPIRBlock>(block.next_block);

		// This block may be a dominating block, so make sure we flush undeclared variables before building the for loop header.
		flush_undeclared_variables(child);

		uint32_t current_count = statement_count;

		// If we're trying to create a true for loop,
		// we need to make sure that all opcodes before branch statement do not actually emit any code.
		// We can then take the condition expression and create a for (; cond ; ) { body; } structure instead.
		emit_block_instructions(child);

		bool condition_is_temporary = forced_temporaries.find(child.condition) == end(forced_temporaries);

		if (current_count == statement_count && condition_is_temporary)
		{
			uint32_t target_block = child.true_block;

			switch (continue_type)
			{
			case SPIRBlock::ForLoop:
			{
				// Important that we do this in this order because
				// emitting the continue block can invalidate the condition expression.
				auto initializer = emit_for_loop_initializers(block);
				auto condition = to_expression(child.condition);

				// Condition might have to be inverted.
				if (execution_is_noop(get<SPIRBlock>(child.true_block), get<SPIRBlock>(block.merge_block)))
				{
					condition = join("!", enclose_expression(condition));
					target_block = child.false_block;
				}

				auto continue_block = emit_continue_block(block.continue_block, false, false);
				emit_block_hints(block);
				statement("for (", initializer, "; ", condition, "; ", continue_block, ")");
				break;
			}

			case SPIRBlock::WhileLoop:
			{
				emit_while_loop_initializers(block);
				emit_block_hints(block);

				auto condition = to_expression(child.condition);
				// Condition might have to be inverted.
				if (execution_is_noop(get<SPIRBlock>(child.true_block), get<SPIRBlock>(block.merge_block)))
				{
					condition = join("!", enclose_expression(condition));
					target_block = child.false_block;
				}

				statement("while (", condition, ")");
				break;
			}

			default:
				block.disable_block_optimization = true;
				force_recompile();
				begin_scope(); // We'll see an end_scope() later.
				return false;
			}

			begin_scope();
			branch(child.self, target_block);
			return true;
		}
		else
		{
			block.disable_block_optimization = true;
			force_recompile();
			begin_scope(); // We'll see an end_scope() later.
			return false;
		}
	}
	else
		return false;
}

void CompilerGLSL::flush_undeclared_variables(SPIRBlock &block)
{
	for (auto &v : block.dominated_variables)
		flush_variable_declaration(v);
}

void CompilerGLSL::emit_hoisted_temporaries(SmallVector<pair<TypeID, ID>> &temporaries)
{
	// If we need to force temporaries for certain IDs due to continue blocks, do it before starting loop header.
	// Need to sort these to ensure that reference output is stable.
	sort(begin(temporaries), end(temporaries),
	     [](const pair<TypeID, ID> &a, const pair<TypeID, ID> &b) { return a.second < b.second; });

	for (auto &tmp : temporaries)
	{
		add_local_variable_name(tmp.second);
		auto &flags = ir.meta[tmp.second].decoration.decoration_flags;
		auto &type = get<SPIRType>(tmp.first);

		// Not all targets support pointer literals, so don't bother with that case.
		string initializer;
		if (options.force_zero_initialized_variables && type_can_zero_initialize(type))
			initializer = join(" = ", to_zero_initialized_expression(tmp.first));

		statement(flags_to_qualifiers_glsl(type, flags), variable_decl(type, to_name(tmp.second)), initializer, ";");

		hoisted_temporaries.insert(tmp.second);
		forced_temporaries.insert(tmp.second);

		// The temporary might be read from before it's assigned, set up the expression now.
		set<SPIRExpression>(tmp.second, to_name(tmp.second), tmp.first, true);
	}
}

void CompilerGLSL::emit_block_chain(SPIRBlock &block)
{
	bool select_branch_to_true_block = false;
	bool select_branch_to_false_block = false;
	bool skip_direct_branch = false;
	bool emitted_loop_header_variables = false;
	bool force_complex_continue_block = false;

	emit_hoisted_temporaries(block.declare_temporary);

	SPIRBlock::ContinueBlockType continue_type = SPIRBlock::ContinueNone;
	if (block.continue_block)
	{
		continue_type = continue_block_type(get<SPIRBlock>(block.continue_block));
		// If we know we cannot emit a loop, mark the block early as a complex loop so we don't force unnecessary recompiles.
		if (continue_type == SPIRBlock::ComplexLoop)
			block.complex_continue = true;
	}

	// If we have loop variables, stop masking out access to the variable now.
	for (auto var_id : block.loop_variables)
	{
		auto &var = get<SPIRVariable>(var_id);
		var.loop_variable_enable = true;
		// We're not going to declare the variable directly, so emit a copy here.
		emit_variable_temporary_copies(var);
	}

	// Remember deferred declaration state. We will restore it before returning.
	SmallVector<bool, 64> rearm_dominated_variables(block.dominated_variables.size());
	for (size_t i = 0; i < block.dominated_variables.size(); i++)
	{
		uint32_t var_id = block.dominated_variables[i];
		auto &var = get<SPIRVariable>(var_id);
		rearm_dominated_variables[i] = var.deferred_declaration;
	}

	// This is the method often used by spirv-opt to implement loops.
	// The loop header goes straight into the continue block.
	// However, don't attempt this on ESSL 1.0, because if a loop variable is used in a continue block,
	// it *MUST* be used in the continue block. This loop method will not work.
	if (!is_legacy_es() && block_is_loop_candidate(block, SPIRBlock::MergeToSelectContinueForLoop))
	{
		flush_undeclared_variables(block);
		if (attempt_emit_loop_header(block, SPIRBlock::MergeToSelectContinueForLoop))
		{
			if (execution_is_noop(get<SPIRBlock>(block.true_block), get<SPIRBlock>(block.merge_block)))
				select_branch_to_false_block = true;
			else
				select_branch_to_true_block = true;

			emitted_loop_header_variables = true;
			force_complex_continue_block = true;
		}
	}
	// This is the older loop behavior in glslang which branches to loop body directly from the loop header.
	else if (block_is_loop_candidate(block, SPIRBlock::MergeToSelectForLoop))
	{
		flush_undeclared_variables(block);
		if (attempt_emit_loop_header(block, SPIRBlock::MergeToSelectForLoop))
		{
			// The body of while, is actually just the true (or false) block, so always branch there unconditionally.
			if (execution_is_noop(get<SPIRBlock>(block.true_block), get<SPIRBlock>(block.merge_block)))
				select_branch_to_false_block = true;
			else
				select_branch_to_true_block = true;

			emitted_loop_header_variables = true;
		}
	}
	// This is the newer loop behavior in glslang which branches from Loop header directly to
	// a new block, which in turn has a OpBranchSelection without a selection merge.
	else if (block_is_loop_candidate(block, SPIRBlock::MergeToDirectForLoop))
	{
		flush_undeclared_variables(block);
		if (attempt_emit_loop_header(block, SPIRBlock::MergeToDirectForLoop))
		{
			skip_direct_branch = true;
			emitted_loop_header_variables = true;
		}
	}
	else if (continue_type == SPIRBlock::DoWhileLoop)
	{
		flush_undeclared_variables(block);
		emit_while_loop_initializers(block);
		emitted_loop_header_variables = true;
		// We have some temporaries where the loop header is the dominator.
		// We risk a case where we have code like:
		// for (;;) { create-temporary; break; } consume-temporary;
		// so force-declare temporaries here.
		emit_hoisted_temporaries(block.potential_declare_temporary);
		statement("do");
		begin_scope();

		emit_block_instructions(block);
	}
	else if (block.merge == SPIRBlock::MergeLoop)
	{
		flush_undeclared_variables(block);
		emit_while_loop_initializers(block);
		emitted_loop_header_variables = true;

		// We have a generic loop without any distinguishable pattern like for, while or do while.
		get<SPIRBlock>(block.continue_block).complex_continue = true;
		continue_type = SPIRBlock::ComplexLoop;

		// We have some temporaries where the loop header is the dominator.
		// We risk a case where we have code like:
		// for (;;) { create-temporary; break; } consume-temporary;
		// so force-declare temporaries here.
		emit_hoisted_temporaries(block.potential_declare_temporary);
		statement("for (;;)");
		begin_scope();

		emit_block_instructions(block);
	}
	else
	{
		emit_block_instructions(block);
	}

	// If we didn't successfully emit a loop header and we had loop variable candidates, we have a problem
	// as writes to said loop variables might have been masked out, we need a recompile.
	if (!emitted_loop_header_variables && !block.loop_variables.empty())
	{
		force_recompile();
		for (auto var : block.loop_variables)
			get<SPIRVariable>(var).loop_variable = false;
		block.loop_variables.clear();
	}

	flush_undeclared_variables(block);
	bool emit_next_block = true;

	// Handle end of block.
	switch (block.terminator)
	{
	case SPIRBlock::Direct:
		// True when emitting complex continue block.
		if (block.loop_dominator == block.next_block)
		{
			branch(block.self, block.next_block);
			emit_next_block = false;
		}
		// True if MergeToDirectForLoop succeeded.
		else if (skip_direct_branch)
			emit_next_block = false;
		else if (is_continue(block.next_block) || is_break(block.next_block) || is_conditional(block.next_block))
		{
			branch(block.self, block.next_block);
			emit_next_block = false;
		}
		break;

	case SPIRBlock::Select:
		// True if MergeToSelectForLoop or MergeToSelectContinueForLoop succeeded.
		if (select_branch_to_true_block)
		{
			if (force_complex_continue_block)
			{
				assert(block.true_block == block.continue_block);

				// We're going to emit a continue block directly here, so make sure it's marked as complex.
				auto &complex_continue = get<SPIRBlock>(block.continue_block).complex_continue;
				bool old_complex = complex_continue;
				complex_continue = true;
				branch(block.self, block.true_block);
				complex_continue = old_complex;
			}
			else
				branch(block.self, block.true_block);
		}
		else if (select_branch_to_false_block)
		{
			if (force_complex_continue_block)
			{
				assert(block.false_block == block.continue_block);

				// We're going to emit a continue block directly here, so make sure it's marked as complex.
				auto &complex_continue = get<SPIRBlock>(block.continue_block).complex_continue;
				bool old_complex = complex_continue;
				complex_continue = true;
				branch(block.self, block.false_block);
				complex_continue = old_complex;
			}
			else
				branch(block.self, block.false_block);
		}
		else
			branch(block.self, block.condition, block.true_block, block.false_block);
		break;

	case SPIRBlock::MultiSelect:
	{
		auto &type = expression_type(block.condition);
		bool unsigned_case =
		    type.basetype == SPIRType::UInt || type.basetype == SPIRType::UShort || type.basetype == SPIRType::UByte;

		if (block.merge == SPIRBlock::MergeNone)
			SPIRV_CROSS_THROW("Switch statement is not structured");

		if (type.basetype == SPIRType::UInt64 || type.basetype == SPIRType::Int64)
		{
			// SPIR-V spec suggests this is allowed, but we cannot support it in higher level languages.
			SPIRV_CROSS_THROW("Cannot use 64-bit switch selectors.");
		}

		const char *label_suffix = "";
		if (type.basetype == SPIRType::UInt && backend.uint32_t_literal_suffix)
			label_suffix = "u";
		else if (type.basetype == SPIRType::UShort)
			label_suffix = backend.uint16_t_literal_suffix;
		else if (type.basetype == SPIRType::Short)
			label_suffix = backend.int16_t_literal_suffix;

		SPIRBlock *old_emitting_switch = current_emitting_switch;
		current_emitting_switch = &block;

		if (block.need_ladder_break)
			statement("bool _", block.self, "_ladder_break = false;");

		// Find all unique case constructs.
		unordered_map<uint32_t, SmallVector<uint32_t>> case_constructs;
		SmallVector<uint32_t> block_declaration_order;
		SmallVector<uint32_t> literals_to_merge;

		// If a switch case branches to the default block for some reason, we can just remove that literal from consideration
		// and let the default: block handle it.
		// 2.11 in SPIR-V spec states that for fall-through cases, there is a very strict declaration order which we can take advantage of here.
		// We only need to consider possible fallthrough if order[i] branches to order[i + 1].
		for (auto &c : block.cases)
		{
			if (c.block != block.next_block && c.block != block.default_block)
			{
				if (!case_constructs.count(c.block))
					block_declaration_order.push_back(c.block);
				case_constructs[c.block].push_back(c.value);
			}
			else if (c.block == block.next_block && block.default_block != block.next_block)
			{
				// We might have to flush phi inside specific case labels.
				// If we can piggyback on default:, do so instead.
				literals_to_merge.push_back(c.value);
			}
		}

		// Empty literal array -> default.
		if (block.default_block != block.next_block)
		{
			auto &default_block = get<SPIRBlock>(block.default_block);

			// We need to slide in the default block somewhere in this chain
			// if there are fall-through scenarios since the default is declared separately in OpSwitch.
			// Only consider trivial fall-through cases here.
			size_t num_blocks = block_declaration_order.size();
			bool injected_block = false;

			for (size_t i = 0; i < num_blocks; i++)
			{
				auto &case_block = get<SPIRBlock>(block_declaration_order[i]);
				if (execution_is_direct_branch(case_block, default_block))
				{
					// Fallthrough to default block, we must inject the default block here.
					block_declaration_order.insert(begin(block_declaration_order) + i + 1, block.default_block);
					injected_block = true;
					break;
				}
				else if (execution_is_direct_branch(default_block, case_block))
				{
					// Default case is falling through to another case label, we must inject the default block here.
					block_declaration_order.insert(begin(block_declaration_order) + i, block.default_block);
					injected_block = true;
					break;
				}
			}

			// Order does not matter.
			if (!injected_block)
				block_declaration_order.push_back(block.default_block);

			case_constructs[block.default_block] = {};
		}

		size_t num_blocks = block_declaration_order.size();

		const auto to_case_label = [](uint32_t literal, bool is_unsigned_case) -> string {
			return is_unsigned_case ? convert_to_string(literal) : convert_to_string(int32_t(literal));
		};

		// We need to deal with a complex scenario for OpPhi. If we have case-fallthrough and Phi in the picture,
		// we need to flush phi nodes outside the switch block in a branch,
		// and skip any Phi handling inside the case label to make fall-through work as expected.
		// This kind of code-gen is super awkward and it's a last resort. Normally we would want to handle this
		// inside the case label if at all possible.
		for (size_t i = 1; i < num_blocks; i++)
		{
			if (flush_phi_required(block.self, block_declaration_order[i]) &&
			    flush_phi_required(block_declaration_order[i - 1], block_declaration_order[i]))
			{
				uint32_t target_block = block_declaration_order[i];

				// Make sure we flush Phi, it might have been marked to be ignored earlier.
				get<SPIRBlock>(target_block).ignore_phi_from_block = 0;

				auto &literals = case_constructs[target_block];

				if (literals.empty())
				{
					// Oh boy, gotta make a complete negative test instead! o.o
					// Find all possible literals that would *not* make us enter the default block.
					// If none of those literals match, we flush Phi ...
					SmallVector<string> conditions;
					for (size_t j = 0; j < num_blocks; j++)
					{
						auto &negative_literals = case_constructs[block_declaration_order[j]];
						for (auto &case_label : negative_literals)
							conditions.push_back(join(to_enclosed_expression(block.condition),
							                          " != ", to_case_label(case_label, unsigned_case)));
					}

					statement("if (", merge(conditions, " && "), ")");
					begin_scope();
					flush_phi(block.self, target_block);
					end_scope();
				}
				else
				{
					SmallVector<string> conditions;
					conditions.reserve(literals.size());
					for (auto &case_label : literals)
						conditions.push_back(join(to_enclosed_expression(block.condition),
						                          " == ", to_case_label(case_label, unsigned_case)));
					statement("if (", merge(conditions, " || "), ")");
					begin_scope();
					flush_phi(block.self, target_block);
					end_scope();
				}

				// Mark the block so that we don't flush Phi from header to case label.
				get<SPIRBlock>(target_block).ignore_phi_from_block = block.self;
			}
		}

		emit_block_hints(block);
		statement("switch (", to_expression(block.condition), ")");
		begin_scope();

		for (size_t i = 0; i < num_blocks; i++)
		{
			uint32_t target_block = block_declaration_order[i];
			auto &literals = case_constructs[target_block];

			if (literals.empty())
			{
				// Default case.
				statement("default:");
			}
			else
			{
				for (auto &case_literal : literals)
				{
					// The case label value must be sign-extended properly in SPIR-V, so we can assume 32-bit values here.
					statement("case ", to_case_label(case_literal, unsigned_case), label_suffix, ":");
				}
			}

			auto &case_block = get<SPIRBlock>(target_block);
			if (backend.support_case_fallthrough && i + 1 < num_blocks &&
			    execution_is_direct_branch(case_block, get<SPIRBlock>(block_declaration_order[i + 1])))
			{
				// We will fall through here, so just terminate the block chain early.
				// We still need to deal with Phi potentially.
				// No need for a stack-like thing here since we only do fall-through when there is a
				// single trivial branch to fall-through target..
				current_emitting_switch_fallthrough = true;
			}
			else
				current_emitting_switch_fallthrough = false;

			begin_scope();
			branch(block.self, target_block);
			end_scope();

			current_emitting_switch_fallthrough = false;
		}

		// Might still have to flush phi variables if we branch from loop header directly to merge target.
		if (flush_phi_required(block.self, block.next_block))
		{
			if (block.default_block == block.next_block || !literals_to_merge.empty())
			{
				for (auto &case_literal : literals_to_merge)
					statement("case ", to_case_label(case_literal, unsigned_case), label_suffix, ":");

				if (block.default_block == block.next_block)
					statement("default:");

				begin_scope();
				flush_phi(block.self, block.next_block);
				statement("break;");
				end_scope();
			}
		}

		end_scope();

		if (block.need_ladder_break)
		{
			statement("if (_", block.self, "_ladder_break)");
			begin_scope();
			statement("break;");
			end_scope();
		}

		current_emitting_switch = old_emitting_switch;
		break;
	}

	case SPIRBlock::Return:
	{
		for (auto &line : current_function->fixup_hooks_out)
			line();

		if (processing_entry_point)
			emit_fixup();

		auto &cfg = get_cfg_for_current_function();

		if (block.return_value)
		{
			auto &type = expression_type(block.return_value);
			if (!type.array.empty() && !backend.can_return_array)
			{
				// If we cannot return arrays, we will have a special out argument we can write to instead.
				// The backend is responsible for setting this up, and redirection the return values as appropriate.
				if (ir.ids[block.return_value].get_type() != TypeUndef)
				{
					emit_array_copy("SPIRV_Cross_return_value", block.return_value, StorageClassFunction,
					                get_backing_variable_storage(block.return_value));
				}

				if (!cfg.node_terminates_control_flow_in_sub_graph(current_function->entry_block, block.self) ||
				    block.loop_dominator != BlockID(SPIRBlock::NoDominator))
				{
					statement("return;");
				}
			}
			else
			{
				// OpReturnValue can return Undef, so don't emit anything for this case.
				if (ir.ids[block.return_value].get_type() != TypeUndef)
					statement("return ", to_expression(block.return_value), ";");
			}
		}
		else if (!cfg.node_terminates_control_flow_in_sub_graph(current_function->entry_block, block.self) ||
		         block.loop_dominator != BlockID(SPIRBlock::NoDominator))
		{
			// If this block is the very final block and not called from control flow,
			// we do not need an explicit return which looks out of place. Just end the function here.
			// In the very weird case of for(;;) { return; } executing return is unconditional,
			// but we actually need a return here ...
			statement("return;");
		}
		break;
	}

	case SPIRBlock::Kill:
		statement(backend.discard_literal, ";");
		break;

	case SPIRBlock::Unreachable:
		emit_next_block = false;
		break;

	default:
		SPIRV_CROSS_THROW("Unimplemented block terminator.");
	}

	if (block.next_block && emit_next_block)
	{
		// If we hit this case, we're dealing with an unconditional branch, which means we will output
		// that block after this. If we had selection merge, we already flushed phi variables.
		if (block.merge != SPIRBlock::MergeSelection)
			flush_phi(block.self, block.next_block);

		// For switch fallthrough cases, we terminate the chain here, but we still need to handle Phi.
		if (!current_emitting_switch_fallthrough)
		{
			// For merge selects we might have ignored the fact that a merge target
			// could have been a break; or continue;
			// We will need to deal with it here.
			if (is_loop_break(block.next_block))
			{
				// Cannot check for just break, because switch statements will also use break.
				assert(block.merge == SPIRBlock::MergeSelection);
				statement("break;");
			}
			else if (is_continue(block.next_block))
			{
				assert(block.merge == SPIRBlock::MergeSelection);
				branch_to_continue(block.self, block.next_block);
			}
			else if (BlockID(block.self) != block.next_block)
				emit_block_chain(get<SPIRBlock>(block.next_block));
		}
	}

	if (block.merge == SPIRBlock::MergeLoop)
	{
		if (continue_type == SPIRBlock::DoWhileLoop)
		{
			// Make sure that we run the continue block to get the expressions set, but this
			// should become an empty string.
			// We have no fallbacks if we cannot forward everything to temporaries ...
			const auto &continue_block = get<SPIRBlock>(block.continue_block);
			bool positive_test = execution_is_noop(get<SPIRBlock>(continue_block.true_block),
			                                       get<SPIRBlock>(continue_block.loop_dominator));

			uint32_t current_count = statement_count;
			auto statements = emit_continue_block(block.continue_block, positive_test, !positive_test);
			if (statement_count != current_count)
			{
				// The DoWhile block has side effects, force ComplexLoop pattern next pass.
				get<SPIRBlock>(block.continue_block).complex_continue = true;
				force_recompile();
			}

			// Might have to invert the do-while test here.
			auto condition = to_expression(continue_block.condition);
			if (!positive_test)
				condition = join("!", enclose_expression(condition));

			end_scope_decl(join("while (", condition, ")"));
		}
		else
			end_scope();

		// We cannot break out of two loops at once, so don't check for break; here.
		// Using block.self as the "from" block isn't quite right, but it has the same scope
		// and dominance structure, so it's fine.
		if (is_continue(block.merge_block))
			branch_to_continue(block.self, block.merge_block);
		else
			emit_block_chain(get<SPIRBlock>(block.merge_block));
	}

	// Forget about control dependent expressions now.
	block.invalidate_expressions.clear();

	// After we return, we must be out of scope, so if we somehow have to re-emit this function,
	// re-declare variables if necessary.
	assert(rearm_dominated_variables.size() == block.dominated_variables.size());
	for (size_t i = 0; i < block.dominated_variables.size(); i++)
	{
		uint32_t var = block.dominated_variables[i];
		get<SPIRVariable>(var).deferred_declaration = rearm_dominated_variables[i];
	}

	// Just like for deferred declaration, we need to forget about loop variable enable
	// if our block chain is reinstantiated later.
	for (auto &var_id : block.loop_variables)
		get<SPIRVariable>(var_id).loop_variable_enable = false;
}

void CompilerGLSL::begin_scope()
{
	statement("{");
	indent++;
}

void CompilerGLSL::end_scope()
{
	if (!indent)
		SPIRV_CROSS_THROW("Popping empty indent stack.");
	indent--;
	statement("}");
}

void CompilerGLSL::end_scope(const string &trailer)
{
	if (!indent)
		SPIRV_CROSS_THROW("Popping empty indent stack.");
	indent--;
	statement("}", trailer);
}

void CompilerGLSL::end_scope_decl()
{
	if (!indent)
		SPIRV_CROSS_THROW("Popping empty indent stack.");
	indent--;
	statement("};");
}

void CompilerGLSL::end_scope_decl(const string &decl)
{
	if (!indent)
		SPIRV_CROSS_THROW("Popping empty indent stack.");
	indent--;
	statement("} ", decl, ";");
}

void CompilerGLSL::check_function_call_constraints(const uint32_t *args, uint32_t length)
{
	// If our variable is remapped, and we rely on type-remapping information as
	// well, then we cannot pass the variable as a function parameter.
	// Fixing this is non-trivial without stamping out variants of the same function,
	// so for now warn about this and suggest workarounds instead.
	for (uint32_t i = 0; i < length; i++)
	{
		auto *var = maybe_get<SPIRVariable>(args[i]);
		if (!var || !var->remapped_variable)
			continue;

		auto &type = get<SPIRType>(var->basetype);
		if (type.basetype == SPIRType::Image && type.image.dim == DimSubpassData)
		{
			SPIRV_CROSS_THROW("Tried passing a remapped subpassInput variable to a function. "
			                  "This will not work correctly because type-remapping information is lost. "
			                  "To workaround, please consider not passing the subpass input as a function parameter, "
			                  "or use in/out variables instead which do not need type remapping information.");
		}
	}
}

const Instruction *CompilerGLSL::get_next_instruction_in_block(const Instruction &instr)
{
	// FIXME: This is kind of hacky. There should be a cleaner way.
	auto offset = uint32_t(&instr - current_emitting_block->ops.data());
	if ((offset + 1) < current_emitting_block->ops.size())
		return &current_emitting_block->ops[offset + 1];
	else
		return nullptr;
}

uint32_t CompilerGLSL::mask_relevant_memory_semantics(uint32_t semantics)
{
	return semantics & (MemorySemanticsAtomicCounterMemoryMask | MemorySemanticsImageMemoryMask |
	                    MemorySemanticsWorkgroupMemoryMask | MemorySemanticsUniformMemoryMask |
	                    MemorySemanticsCrossWorkgroupMemoryMask | MemorySemanticsSubgroupMemoryMask);
}

void CompilerGLSL::emit_array_copy(const string &lhs, uint32_t rhs_id, StorageClass, StorageClass)
{
	statement(lhs, " = ", to_expression(rhs_id), ";");
}

void CompilerGLSL::unroll_array_from_complex_load(uint32_t target_id, uint32_t source_id, std::string &expr)
{
	if (!backend.force_gl_in_out_block)
		return;
	// This path is only relevant for GL backends.

	auto *var = maybe_get<SPIRVariable>(source_id);
	if (!var)
		return;

	if (var->storage != StorageClassInput)
		return;

	auto &type = get_variable_data_type(*var);
	if (type.array.empty())
		return;

	auto builtin = BuiltIn(get_decoration(var->self, DecorationBuiltIn));
	bool is_builtin = is_builtin_variable(*var) && (builtin == BuiltInPointSize || builtin == BuiltInPosition);
	bool is_tess = is_tessellation_shader();
	bool is_patch = has_decoration(var->self, DecorationPatch);

	// Tessellation input arrays are special in that they are unsized, so we cannot directly copy from it.
	// We must unroll the array load.
	// For builtins, we couldn't catch this case normally,
	// because this is resolved in the OpAccessChain in most cases.
	// If we load the entire array, we have no choice but to unroll here.
	if (!is_patch && (is_builtin || is_tess))
	{
		auto new_expr = join("_", target_id, "_unrolled");
		statement(variable_decl(type, new_expr, target_id), ";");
		string array_expr;
		if (type.array_size_literal.back())
		{
			array_expr = convert_to_string(type.array.back());
			if (type.array.back() == 0)
				SPIRV_CROSS_THROW("Cannot unroll an array copy from unsized array.");
		}
		else
			array_expr = to_expression(type.array.back());

		// The array size might be a specialization constant, so use a for-loop instead.
		statement("for (int i = 0; i < int(", array_expr, "); i++)");
		begin_scope();
		if (is_builtin)
			statement(new_expr, "[i] = gl_in[i].", expr, ";");
		else
			statement(new_expr, "[i] = ", expr, "[i];");
		end_scope();

		expr = move(new_expr);
	}
}

void CompilerGLSL::bitcast_from_builtin_load(uint32_t source_id, std::string &expr, const SPIRType &expr_type)
{
	auto *var = maybe_get_backing_variable(source_id);
	if (var)
		source_id = var->self;

	// Only interested in standalone builtin variables.
	if (!has_decoration(source_id, DecorationBuiltIn))
		return;

	auto builtin = static_cast<BuiltIn>(get_decoration(source_id, DecorationBuiltIn));
	auto expected_type = expr_type.basetype;

	// TODO: Fill in for more builtins.
	switch (builtin)
	{
	case BuiltInLayer:
	case BuiltInPrimitiveId:
	case BuiltInViewportIndex:
	case BuiltInInstanceId:
	case BuiltInInstanceIndex:
	case BuiltInVertexId:
	case BuiltInVertexIndex:
	case BuiltInSampleId:
	case BuiltInBaseVertex:
	case BuiltInBaseInstance:
	case BuiltInDrawIndex:
	case BuiltInFragStencilRefEXT:
		expected_type = SPIRType::Int;
		break;

	case BuiltInGlobalInvocationId:
	case BuiltInLocalInvocationId:
	case BuiltInWorkgroupId:
	case BuiltInLocalInvocationIndex:
	case BuiltInWorkgroupSize:
	case BuiltInNumWorkgroups:
		expected_type = SPIRType::UInt;
		break;

	default:
		break;
	}

	if (expected_type != expr_type.basetype)
		expr = bitcast_expression(expr_type, expected_type, expr);
}

void CompilerGLSL::bitcast_to_builtin_store(uint32_t target_id, std::string &expr, const SPIRType &expr_type)
{
	// Only interested in standalone builtin variables.
	if (!has_decoration(target_id, DecorationBuiltIn))
		return;

	auto builtin = static_cast<BuiltIn>(get_decoration(target_id, DecorationBuiltIn));
	auto expected_type = expr_type.basetype;

	// TODO: Fill in for more builtins.
	switch (builtin)
	{
	case BuiltInLayer:
	case BuiltInPrimitiveId:
	case BuiltInViewportIndex:
	case BuiltInFragStencilRefEXT:
		expected_type = SPIRType::Int;
		break;

	default:
		break;
	}

	if (expected_type != expr_type.basetype)
	{
		auto type = expr_type;
		type.basetype = expected_type;
		expr = bitcast_expression(type, expr_type.basetype, expr);
	}
}

void CompilerGLSL::convert_non_uniform_expression(const SPIRType &type, std::string &expr)
{
	if (*backend.nonuniform_qualifier == '\0')
		return;

	// Handle SPV_EXT_descriptor_indexing.
	if (type.basetype == SPIRType::Sampler || type.basetype == SPIRType::SampledImage ||
	    type.basetype == SPIRType::Image)
	{
		// The image/sampler ID must be declared as non-uniform.
		// However, it is not legal GLSL to have
		// nonuniformEXT(samplers[index]), so we must move the nonuniform qualifier
		// to the array indexing, like
		// samplers[nonuniformEXT(index)].
		// While the access chain will generally be nonuniformEXT, it's not necessarily so,
		// so we might have to fixup the OpLoad-ed expression late.

		auto start_array_index = expr.find_first_of('[');
		auto end_array_index = expr.find_last_of(']');
		// Doesn't really make sense to declare a non-arrayed image with nonuniformEXT, but there's
		// nothing we can do here to express that.
		if (start_array_index == string::npos || end_array_index == string::npos || end_array_index < start_array_index)
			return;

		start_array_index++;

		expr = join(expr.substr(0, start_array_index), backend.nonuniform_qualifier, "(",
		            expr.substr(start_array_index, end_array_index - start_array_index), ")",
		            expr.substr(end_array_index, string::npos));
	}
}

void CompilerGLSL::emit_block_hints(const SPIRBlock &)
{
}

void CompilerGLSL::preserve_alias_on_reset(uint32_t id)
{
	preserved_aliases[id] = get_name(id);
}

void CompilerGLSL::reset_name_caches()
{
	for (auto &preserved : preserved_aliases)
		set_name(preserved.first, preserved.second);

	preserved_aliases.clear();
	resource_names.clear();
	block_input_names.clear();
	block_output_names.clear();
	block_ubo_names.clear();
	block_ssbo_names.clear();
	block_names.clear();
	function_overloads.clear();
}

void CompilerGLSL::fixup_type_alias()
{
	// Due to how some backends work, the "master" type of type_alias must be a block-like type if it exists.
	// FIXME: Multiple alias types which are both block-like will be awkward, for now, it's best to just drop the type
	// alias if the slave type is a block type.
	ir.for_each_typed_id<SPIRType>([&](uint32_t self, SPIRType &type) {
		if (type.type_alias && type_is_block_like(type))
		{
			// Become the master.
			ir.for_each_typed_id<SPIRType>([&](uint32_t other_id, SPIRType &other_type) {
				if (other_id == type.self)
					return;

				if (other_type.type_alias == type.type_alias)
					other_type.type_alias = type.self;
			});

			this->get<SPIRType>(type.type_alias).type_alias = self;
			type.type_alias = 0;
		}
	});

	ir.for_each_typed_id<SPIRType>([&](uint32_t, SPIRType &type) {
		if (type.type_alias && type_is_block_like(type))
		{
			// This is not allowed, drop the type_alias.
			type.type_alias = 0;
		}
		else if (type.type_alias && !type_is_block_like(this->get<SPIRType>(type.type_alias)))
		{
			// If the alias master is not a block-like type, there is no reason to use type aliasing.
			// This case can happen if two structs are declared with the same name, but they are unrelated.
			// Aliases are only used to deal with aliased types for structs which are used in different buffer types
			// which all create a variant of the same struct with different DecorationOffset values.
			type.type_alias = 0;
		}
	});
}

void CompilerGLSL::reorder_type_alias()
{
	// Reorder declaration of types so that the master of the type alias is always emitted first.
	// We need this in case a type B depends on type A (A must come before in the vector), but A is an alias of a type Abuffer, which
	// means declaration of A doesn't happen (yet), and order would be B, ABuffer and not ABuffer, B. Fix this up here.
	auto loop_lock = ir.create_loop_hard_lock();

	auto &type_ids = ir.ids_for_type[TypeType];
	for (auto alias_itr = begin(type_ids); alias_itr != end(type_ids); ++alias_itr)
	{
		auto &type = get<SPIRType>(*alias_itr);
		if (type.type_alias != TypeID(0) &&
		    !has_extended_decoration(type.type_alias, SPIRVCrossDecorationBufferBlockRepacked))
		{
			// We will skip declaring this type, so make sure the type_alias type comes before.
			auto master_itr = find(begin(type_ids), end(type_ids), ID(type.type_alias));
			assert(master_itr != end(type_ids));

			if (alias_itr < master_itr)
			{
				// Must also swap the type order for the constant-type joined array.
				auto &joined_types = ir.ids_for_constant_or_type;
				auto alt_alias_itr = find(begin(joined_types), end(joined_types), *alias_itr);
				auto alt_master_itr = find(begin(joined_types), end(joined_types), *master_itr);
				assert(alt_alias_itr != end(joined_types));
				assert(alt_master_itr != end(joined_types));

				swap(*alias_itr, *master_itr);
				swap(*alt_alias_itr, *alt_master_itr);
			}
		}
	}
}

void CompilerGLSL::emit_line_directive(uint32_t file_id, uint32_t line_literal)
{
	// If we are redirecting statements, ignore the line directive.
	// Common case here is continue blocks.
	if (redirect_statement)
		return;

	if (options.emit_line_directives)
	{
		require_extension_internal("GL_GOOGLE_cpp_style_line_directive");
		statement_no_indent("#line ", line_literal, " \"", get<SPIRString>(file_id).str, "\"");
	}
}

void CompilerGLSL::propagate_nonuniform_qualifier(uint32_t id)
{
	// SPIR-V might only tag the very last ID with NonUniformEXT, but for codegen,
	// we need to know NonUniformEXT a little earlier, when the resource is actually loaded.
	// Back-propagate the qualifier based on the expression dependency chain.

	if (!has_decoration(id, DecorationNonUniformEXT))
	{
		set_decoration(id, DecorationNonUniformEXT);
		force_recompile();
	}

	auto *e = maybe_get<SPIRExpression>(id);
	auto *combined = maybe_get<SPIRCombinedImageSampler>(id);
	auto *chain = maybe_get<SPIRAccessChain>(id);
	if (e)
	{
		for (auto &expr : e->expression_dependencies)
			propagate_nonuniform_qualifier(expr);
		for (auto &expr : e->implied_read_expressions)
			propagate_nonuniform_qualifier(expr);
	}
	else if (combined)
	{
		propagate_nonuniform_qualifier(combined->image);
		propagate_nonuniform_qualifier(combined->sampler);
	}
	else if (chain)
	{
		for (auto &expr : chain->implied_read_expressions)
			propagate_nonuniform_qualifier(expr);
	}
}

void CompilerGLSL::emit_copy_logical_type(uint32_t lhs_id, uint32_t lhs_type_id, uint32_t rhs_id, uint32_t rhs_type_id,
                                          SmallVector<uint32_t> chain)
{
	// Fully unroll all member/array indices one by one.

	auto &lhs_type = get<SPIRType>(lhs_type_id);
	auto &rhs_type = get<SPIRType>(rhs_type_id);

	if (!lhs_type.array.empty())
	{
		// Could use a loop here to support specialization constants, but it gets rather complicated with nested array types,
		// and this is a rather obscure opcode anyways, keep it simple unless we are forced to.
		uint32_t array_size = to_array_size_literal(lhs_type);
		chain.push_back(0);

		for (uint32_t i = 0; i < array_size; i++)
		{
			chain.back() = i;
			emit_copy_logical_type(lhs_id, lhs_type.parent_type, rhs_id, rhs_type.parent_type, chain);
		}
	}
	else if (lhs_type.basetype == SPIRType::Struct)
	{
		chain.push_back(0);
		uint32_t member_count = uint32_t(lhs_type.member_types.size());
		for (uint32_t i = 0; i < member_count; i++)
		{
			chain.back() = i;
			emit_copy_logical_type(lhs_id, lhs_type.member_types[i], rhs_id, rhs_type.member_types[i], chain);
		}
	}
	else
	{
		// Need to handle unpack/packing fixups since this can differ wildly between the logical types,
		// particularly in MSL.
		// To deal with this, we emit access chains and go through emit_store_statement
		// to deal with all the special cases we can encounter.

		AccessChainMeta lhs_meta, rhs_meta;
		auto lhs = access_chain_internal(lhs_id, chain.data(), uint32_t(chain.size()),
		                                 ACCESS_CHAIN_INDEX_IS_LITERAL_BIT, &lhs_meta);
		auto rhs = access_chain_internal(rhs_id, chain.data(), uint32_t(chain.size()),
		                                 ACCESS_CHAIN_INDEX_IS_LITERAL_BIT, &rhs_meta);

		uint32_t id = ir.increase_bound_by(2);
		lhs_id = id;
		rhs_id = id + 1;

		{
			auto &lhs_expr = set<SPIRExpression>(lhs_id, move(lhs), lhs_type_id, true);
			lhs_expr.need_transpose = lhs_meta.need_transpose;

			if (lhs_meta.storage_is_packed)
				set_extended_decoration(lhs_id, SPIRVCrossDecorationPhysicalTypePacked);
			if (lhs_meta.storage_physical_type != 0)
				set_extended_decoration(lhs_id, SPIRVCrossDecorationPhysicalTypeID, lhs_meta.storage_physical_type);

			forwarded_temporaries.insert(lhs_id);
			suppressed_usage_tracking.insert(lhs_id);
		}

		{
			auto &rhs_expr = set<SPIRExpression>(rhs_id, move(rhs), rhs_type_id, true);
			rhs_expr.need_transpose = rhs_meta.need_transpose;

			if (rhs_meta.storage_is_packed)
				set_extended_decoration(rhs_id, SPIRVCrossDecorationPhysicalTypePacked);
			if (rhs_meta.storage_physical_type != 0)
				set_extended_decoration(rhs_id, SPIRVCrossDecorationPhysicalTypeID, rhs_meta.storage_physical_type);

			forwarded_temporaries.insert(rhs_id);
			suppressed_usage_tracking.insert(rhs_id);
		}

		emit_store_statement(lhs_id, rhs_id);
	}
}

bool CompilerGLSL::subpass_input_is_framebuffer_fetch(uint32_t id) const
{
	if (!has_decoration(id, DecorationInputAttachmentIndex))
		return false;

	uint32_t input_attachment_index = get_decoration(id, DecorationInputAttachmentIndex);
	for (auto &remap : subpass_to_framebuffer_fetch_attachment)
		if (remap.first == input_attachment_index)
			return true;

	return false;
}

const SPIRVariable *CompilerGLSL::find_subpass_input_by_attachment_index(uint32_t index) const
{
	const SPIRVariable *ret = nullptr;
	ir.for_each_typed_id<SPIRVariable>([&](uint32_t, const SPIRVariable &var) {
		if (has_decoration(var.self, DecorationInputAttachmentIndex) &&
		    get_decoration(var.self, DecorationInputAttachmentIndex) == index)
		{
			ret = &var;
		}
	});
	return ret;
}

const SPIRVariable *CompilerGLSL::find_color_output_by_location(uint32_t location) const
{
	const SPIRVariable *ret = nullptr;
	ir.for_each_typed_id<SPIRVariable>([&](uint32_t, const SPIRVariable &var) {
		if (var.storage == StorageClassOutput && get_decoration(var.self, DecorationLocation) == location)
			ret = &var;
	});
	return ret;
}

void CompilerGLSL::emit_inout_fragment_outputs_copy_to_subpass_inputs()
{
	for (auto &remap : subpass_to_framebuffer_fetch_attachment)
	{
		auto *subpass_var = find_subpass_input_by_attachment_index(remap.first);
		auto *output_var = find_color_output_by_location(remap.second);
		if (!subpass_var)
			continue;
		if (!output_var)
			SPIRV_CROSS_THROW("Need to declare the corresponding fragment output variable to be able to read from it.");
		if (is_array(get<SPIRType>(output_var->basetype)))
			SPIRV_CROSS_THROW("Cannot use GL_EXT_shader_framebuffer_fetch with arrays of color outputs.");

		auto &func = get<SPIRFunction>(get_entry_point().self);
		func.fixup_hooks_in.push_back([=]() {
			if (is_legacy())
			{
				statement(to_expression(subpass_var->self), " = ", "gl_LastFragData[",
				          get_decoration(output_var->self, DecorationLocation), "];");
			}
			else
			{
				uint32_t num_rt_components = this->get<SPIRType>(output_var->basetype).vecsize;
				statement(to_expression(subpass_var->self), vector_swizzle(num_rt_components, 0), " = ",
				          to_expression(output_var->self), ";");
			}
		});
	}
}

bool CompilerGLSL::variable_is_depth_or_compare(VariableID id) const
{
	return image_is_comparison(get<SPIRType>(get<SPIRVariable>(id).basetype), id);
}
