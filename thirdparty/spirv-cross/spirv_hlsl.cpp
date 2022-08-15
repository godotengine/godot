/*
 * Copyright 2016-2021 Robert Konrad
 * SPDX-License-Identifier: Apache-2.0 OR MIT
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
 *
 */

/*
 * At your option, you may choose to accept this material under either:
 *  1. The Apache License, Version 2.0, found at <http://www.apache.org/licenses/LICENSE-2.0>, or
 *  2. The MIT License, found at <http://opensource.org/licenses/MIT>.
 */

#include "spirv_hlsl.hpp"
#include "GLSL.std.450.h"
#include <algorithm>
#include <assert.h>

using namespace spv;
using namespace SPIRV_CROSS_NAMESPACE;
using namespace std;

enum class ImageFormatNormalizedState
{
	None = 0,
	Unorm = 1,
	Snorm = 2
};

static ImageFormatNormalizedState image_format_to_normalized_state(ImageFormat fmt)
{
	switch (fmt)
	{
	case ImageFormatR8:
	case ImageFormatR16:
	case ImageFormatRg8:
	case ImageFormatRg16:
	case ImageFormatRgba8:
	case ImageFormatRgba16:
	case ImageFormatRgb10A2:
		return ImageFormatNormalizedState::Unorm;

	case ImageFormatR8Snorm:
	case ImageFormatR16Snorm:
	case ImageFormatRg8Snorm:
	case ImageFormatRg16Snorm:
	case ImageFormatRgba8Snorm:
	case ImageFormatRgba16Snorm:
		return ImageFormatNormalizedState::Snorm;

	default:
		break;
	}

	return ImageFormatNormalizedState::None;
}

static unsigned image_format_to_components(ImageFormat fmt)
{
	switch (fmt)
	{
	case ImageFormatR8:
	case ImageFormatR16:
	case ImageFormatR8Snorm:
	case ImageFormatR16Snorm:
	case ImageFormatR16f:
	case ImageFormatR32f:
	case ImageFormatR8i:
	case ImageFormatR16i:
	case ImageFormatR32i:
	case ImageFormatR8ui:
	case ImageFormatR16ui:
	case ImageFormatR32ui:
		return 1;

	case ImageFormatRg8:
	case ImageFormatRg16:
	case ImageFormatRg8Snorm:
	case ImageFormatRg16Snorm:
	case ImageFormatRg16f:
	case ImageFormatRg32f:
	case ImageFormatRg8i:
	case ImageFormatRg16i:
	case ImageFormatRg32i:
	case ImageFormatRg8ui:
	case ImageFormatRg16ui:
	case ImageFormatRg32ui:
		return 2;

	case ImageFormatR11fG11fB10f:
		return 3;

	case ImageFormatRgba8:
	case ImageFormatRgba16:
	case ImageFormatRgb10A2:
	case ImageFormatRgba8Snorm:
	case ImageFormatRgba16Snorm:
	case ImageFormatRgba16f:
	case ImageFormatRgba32f:
	case ImageFormatRgba8i:
	case ImageFormatRgba16i:
	case ImageFormatRgba32i:
	case ImageFormatRgba8ui:
	case ImageFormatRgba16ui:
	case ImageFormatRgba32ui:
	case ImageFormatRgb10a2ui:
		return 4;

	case ImageFormatUnknown:
		return 4; // Assume 4.

	default:
		SPIRV_CROSS_THROW("Unrecognized typed image format.");
	}
}

static string image_format_to_type(ImageFormat fmt, SPIRType::BaseType basetype)
{
	switch (fmt)
	{
	case ImageFormatR8:
	case ImageFormatR16:
		if (basetype != SPIRType::Float)
			SPIRV_CROSS_THROW("Mismatch in image type and base type of image.");
		return "unorm float";
	case ImageFormatRg8:
	case ImageFormatRg16:
		if (basetype != SPIRType::Float)
			SPIRV_CROSS_THROW("Mismatch in image type and base type of image.");
		return "unorm float2";
	case ImageFormatRgba8:
	case ImageFormatRgba16:
		if (basetype != SPIRType::Float)
			SPIRV_CROSS_THROW("Mismatch in image type and base type of image.");
		return "unorm float4";
	case ImageFormatRgb10A2:
		if (basetype != SPIRType::Float)
			SPIRV_CROSS_THROW("Mismatch in image type and base type of image.");
		return "unorm float4";

	case ImageFormatR8Snorm:
	case ImageFormatR16Snorm:
		if (basetype != SPIRType::Float)
			SPIRV_CROSS_THROW("Mismatch in image type and base type of image.");
		return "snorm float";
	case ImageFormatRg8Snorm:
	case ImageFormatRg16Snorm:
		if (basetype != SPIRType::Float)
			SPIRV_CROSS_THROW("Mismatch in image type and base type of image.");
		return "snorm float2";
	case ImageFormatRgba8Snorm:
	case ImageFormatRgba16Snorm:
		if (basetype != SPIRType::Float)
			SPIRV_CROSS_THROW("Mismatch in image type and base type of image.");
		return "snorm float4";

	case ImageFormatR16f:
	case ImageFormatR32f:
		if (basetype != SPIRType::Float)
			SPIRV_CROSS_THROW("Mismatch in image type and base type of image.");
		return "float";
	case ImageFormatRg16f:
	case ImageFormatRg32f:
		if (basetype != SPIRType::Float)
			SPIRV_CROSS_THROW("Mismatch in image type and base type of image.");
		return "float2";
	case ImageFormatRgba16f:
	case ImageFormatRgba32f:
		if (basetype != SPIRType::Float)
			SPIRV_CROSS_THROW("Mismatch in image type and base type of image.");
		return "float4";

	case ImageFormatR11fG11fB10f:
		if (basetype != SPIRType::Float)
			SPIRV_CROSS_THROW("Mismatch in image type and base type of image.");
		return "float3";

	case ImageFormatR8i:
	case ImageFormatR16i:
	case ImageFormatR32i:
		if (basetype != SPIRType::Int)
			SPIRV_CROSS_THROW("Mismatch in image type and base type of image.");
		return "int";
	case ImageFormatRg8i:
	case ImageFormatRg16i:
	case ImageFormatRg32i:
		if (basetype != SPIRType::Int)
			SPIRV_CROSS_THROW("Mismatch in image type and base type of image.");
		return "int2";
	case ImageFormatRgba8i:
	case ImageFormatRgba16i:
	case ImageFormatRgba32i:
		if (basetype != SPIRType::Int)
			SPIRV_CROSS_THROW("Mismatch in image type and base type of image.");
		return "int4";

	case ImageFormatR8ui:
	case ImageFormatR16ui:
	case ImageFormatR32ui:
		if (basetype != SPIRType::UInt)
			SPIRV_CROSS_THROW("Mismatch in image type and base type of image.");
		return "uint";
	case ImageFormatRg8ui:
	case ImageFormatRg16ui:
	case ImageFormatRg32ui:
		if (basetype != SPIRType::UInt)
			SPIRV_CROSS_THROW("Mismatch in image type and base type of image.");
		return "uint2";
	case ImageFormatRgba8ui:
	case ImageFormatRgba16ui:
	case ImageFormatRgba32ui:
		if (basetype != SPIRType::UInt)
			SPIRV_CROSS_THROW("Mismatch in image type and base type of image.");
		return "uint4";
	case ImageFormatRgb10a2ui:
		if (basetype != SPIRType::UInt)
			SPIRV_CROSS_THROW("Mismatch in image type and base type of image.");
		return "uint4";

	case ImageFormatUnknown:
		switch (basetype)
		{
		case SPIRType::Float:
			return "float4";
		case SPIRType::Int:
			return "int4";
		case SPIRType::UInt:
			return "uint4";
		default:
			SPIRV_CROSS_THROW("Unsupported base type for image.");
		}

	default:
		SPIRV_CROSS_THROW("Unrecognized typed image format.");
	}
}

string CompilerHLSL::image_type_hlsl_modern(const SPIRType &type, uint32_t id)
{
	auto &imagetype = get<SPIRType>(type.image.type);
	const char *dim = nullptr;
	bool typed_load = false;
	uint32_t components = 4;

	bool force_image_srv = hlsl_options.nonwritable_uav_texture_as_srv && has_decoration(id, DecorationNonWritable);

	switch (type.image.dim)
	{
	case Dim1D:
		typed_load = type.image.sampled == 2;
		dim = "1D";
		break;
	case Dim2D:
		typed_load = type.image.sampled == 2;
		dim = "2D";
		break;
	case Dim3D:
		typed_load = type.image.sampled == 2;
		dim = "3D";
		break;
	case DimCube:
		if (type.image.sampled == 2)
			SPIRV_CROSS_THROW("RWTextureCube does not exist in HLSL.");
		dim = "Cube";
		break;
	case DimRect:
		SPIRV_CROSS_THROW("Rectangle texture support is not yet implemented for HLSL."); // TODO
	case DimBuffer:
		if (type.image.sampled == 1)
			return join("Buffer<", type_to_glsl(imagetype), components, ">");
		else if (type.image.sampled == 2)
		{
			if (interlocked_resources.count(id))
				return join("RasterizerOrderedBuffer<", image_format_to_type(type.image.format, imagetype.basetype),
				            ">");

			typed_load = !force_image_srv && type.image.sampled == 2;

			const char *rw = force_image_srv ? "" : "RW";
			return join(rw, "Buffer<",
			            typed_load ? image_format_to_type(type.image.format, imagetype.basetype) :
			                         join(type_to_glsl(imagetype), components),
			            ">");
		}
		else
			SPIRV_CROSS_THROW("Sampler buffers must be either sampled or unsampled. Cannot deduce in runtime.");
	case DimSubpassData:
		dim = "2D";
		typed_load = false;
		break;
	default:
		SPIRV_CROSS_THROW("Invalid dimension.");
	}
	const char *arrayed = type.image.arrayed ? "Array" : "";
	const char *ms = type.image.ms ? "MS" : "";
	const char *rw = typed_load && !force_image_srv ? "RW" : "";

	if (force_image_srv)
		typed_load = false;

	if (typed_load && interlocked_resources.count(id))
		rw = "RasterizerOrdered";

	return join(rw, "Texture", dim, ms, arrayed, "<",
	            typed_load ? image_format_to_type(type.image.format, imagetype.basetype) :
	                         join(type_to_glsl(imagetype), components),
	            ">");
}

string CompilerHLSL::image_type_hlsl_legacy(const SPIRType &type, uint32_t /*id*/)
{
	auto &imagetype = get<SPIRType>(type.image.type);
	string res;

	switch (imagetype.basetype)
	{
	case SPIRType::Int:
		res = "i";
		break;
	case SPIRType::UInt:
		res = "u";
		break;
	default:
		break;
	}

	if (type.basetype == SPIRType::Image && type.image.dim == DimSubpassData)
		return res + "subpassInput" + (type.image.ms ? "MS" : "");

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
		res += "CUBE";
		break;

	case DimBuffer:
		res += "Buffer";
		break;

	case DimSubpassData:
		res += "2D";
		break;
	default:
		SPIRV_CROSS_THROW("Only 1D, 2D, 3D, Buffer, InputTarget and Cube textures supported.");
	}

	if (type.image.ms)
		res += "MS";
	if (type.image.arrayed)
		res += "Array";

	return res;
}

string CompilerHLSL::image_type_hlsl(const SPIRType &type, uint32_t id)
{
	if (hlsl_options.shader_model <= 30)
		return image_type_hlsl_legacy(type, id);
	else
		return image_type_hlsl_modern(type, id);
}

// The optional id parameter indicates the object whose type we are trying
// to find the description for. It is optional. Most type descriptions do not
// depend on a specific object's use of that type.
string CompilerHLSL::type_to_glsl(const SPIRType &type, uint32_t id)
{
	// Ignore the pointer type since GLSL doesn't have pointers.

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
		return image_type_hlsl(type, id);

	case SPIRType::Sampler:
		return comparison_ids.count(id) ? "SamplerComparisonState" : "SamplerState";

	case SPIRType::Void:
		return "void";

	default:
		break;
	}

	if (type.vecsize == 1 && type.columns == 1) // Scalar builtin
	{
		switch (type.basetype)
		{
		case SPIRType::Boolean:
			return "bool";
		case SPIRType::Int:
			return backend.basic_int_type;
		case SPIRType::UInt:
			return backend.basic_uint_type;
		case SPIRType::AtomicCounter:
			return "atomic_uint";
		case SPIRType::Half:
			if (hlsl_options.enable_16bit_types)
				return "half";
			else
				return "min16float";
		case SPIRType::Short:
			if (hlsl_options.enable_16bit_types)
				return "int16_t";
			else
				return "min16int";
		case SPIRType::UShort:
			if (hlsl_options.enable_16bit_types)
				return "uint16_t";
			else
				return "min16uint";
		case SPIRType::Float:
			return "float";
		case SPIRType::Double:
			return "double";
		case SPIRType::Int64:
			if (hlsl_options.shader_model < 60)
				SPIRV_CROSS_THROW("64-bit integers only supported in SM 6.0.");
			return "int64_t";
		case SPIRType::UInt64:
			if (hlsl_options.shader_model < 60)
				SPIRV_CROSS_THROW("64-bit integers only supported in SM 6.0.");
			return "uint64_t";
		case SPIRType::AccelerationStructure:
			return "RaytracingAccelerationStructure";
		case SPIRType::RayQuery:
			return "RayQuery<RAY_FLAG_NONE>";
		default:
			return "???";
		}
	}
	else if (type.vecsize > 1 && type.columns == 1) // Vector builtin
	{
		switch (type.basetype)
		{
		case SPIRType::Boolean:
			return join("bool", type.vecsize);
		case SPIRType::Int:
			return join("int", type.vecsize);
		case SPIRType::UInt:
			return join("uint", type.vecsize);
		case SPIRType::Half:
			return join(hlsl_options.enable_16bit_types ? "half" : "min16float", type.vecsize);
		case SPIRType::Short:
			return join(hlsl_options.enable_16bit_types ? "int16_t" : "min16int", type.vecsize);
		case SPIRType::UShort:
			return join(hlsl_options.enable_16bit_types ? "uint16_t" : "min16uint", type.vecsize);
		case SPIRType::Float:
			return join("float", type.vecsize);
		case SPIRType::Double:
			return join("double", type.vecsize);
		case SPIRType::Int64:
			return join("i64vec", type.vecsize);
		case SPIRType::UInt64:
			return join("u64vec", type.vecsize);
		default:
			return "???";
		}
	}
	else
	{
		switch (type.basetype)
		{
		case SPIRType::Boolean:
			return join("bool", type.columns, "x", type.vecsize);
		case SPIRType::Int:
			return join("int", type.columns, "x", type.vecsize);
		case SPIRType::UInt:
			return join("uint", type.columns, "x", type.vecsize);
		case SPIRType::Half:
			return join(hlsl_options.enable_16bit_types ? "half" : "min16float", type.columns, "x", type.vecsize);
		case SPIRType::Short:
			return join(hlsl_options.enable_16bit_types ? "int16_t" : "min16int", type.columns, "x", type.vecsize);
		case SPIRType::UShort:
			return join(hlsl_options.enable_16bit_types ? "uint16_t" : "min16uint", type.columns, "x", type.vecsize);
		case SPIRType::Float:
			return join("float", type.columns, "x", type.vecsize);
		case SPIRType::Double:
			return join("double", type.columns, "x", type.vecsize);
		// Matrix types not supported for int64/uint64.
		default:
			return "???";
		}
	}
}

void CompilerHLSL::emit_header()
{
	for (auto &header : header_lines)
		statement(header);

	if (header_lines.size() > 0)
	{
		statement("");
	}
}

void CompilerHLSL::emit_interface_block_globally(const SPIRVariable &var)
{
	add_resource_name(var.self);

	// The global copies of I/O variables should not contain interpolation qualifiers.
	// These are emitted inside the interface structs.
	auto &flags = ir.meta[var.self].decoration.decoration_flags;
	auto old_flags = flags;
	flags.reset();
	statement("static ", variable_decl(var), ";");
	flags = old_flags;
}

const char *CompilerHLSL::to_storage_qualifiers_glsl(const SPIRVariable &var)
{
	// Input and output variables are handled specially in HLSL backend.
	// The variables are declared as global, private variables, and do not need any qualifiers.
	if (var.storage == StorageClassUniformConstant || var.storage == StorageClassUniform ||
	    var.storage == StorageClassPushConstant)
	{
		return "uniform ";
	}

	return "";
}

void CompilerHLSL::emit_builtin_outputs_in_struct()
{
	auto &execution = get_entry_point();

	bool legacy = hlsl_options.shader_model <= 30;
	active_output_builtins.for_each_bit([&](uint32_t i) {
		const char *type = nullptr;
		const char *semantic = nullptr;
		auto builtin = static_cast<BuiltIn>(i);
		switch (builtin)
		{
		case BuiltInPosition:
			type = is_position_invariant() && backend.support_precise_qualifier ? "precise float4" : "float4";
			semantic = legacy ? "POSITION" : "SV_Position";
			break;

		case BuiltInSampleMask:
			if (hlsl_options.shader_model < 41 || execution.model != ExecutionModelFragment)
				SPIRV_CROSS_THROW("Sample Mask output is only supported in PS 4.1 or higher.");
			type = "uint";
			semantic = "SV_Coverage";
			break;

		case BuiltInFragDepth:
			type = "float";
			if (legacy)
			{
				semantic = "DEPTH";
			}
			else
			{
				if (hlsl_options.shader_model >= 50 && execution.flags.get(ExecutionModeDepthGreater))
					semantic = "SV_DepthGreaterEqual";
				else if (hlsl_options.shader_model >= 50 && execution.flags.get(ExecutionModeDepthLess))
					semantic = "SV_DepthLessEqual";
				else
					semantic = "SV_Depth";
			}
			break;

		case BuiltInClipDistance:
			// HLSL is a bit weird here, use SV_ClipDistance0, SV_ClipDistance1 and so on with vectors.
			for (uint32_t clip = 0; clip < clip_distance_count; clip += 4)
			{
				uint32_t to_declare = clip_distance_count - clip;
				if (to_declare > 4)
					to_declare = 4;

				uint32_t semantic_index = clip / 4;

				static const char *types[] = { "float", "float2", "float3", "float4" };
				statement(types[to_declare - 1], " ", builtin_to_glsl(builtin, StorageClassOutput), semantic_index,
				          " : SV_ClipDistance", semantic_index, ";");
			}
			break;

		case BuiltInCullDistance:
			// HLSL is a bit weird here, use SV_CullDistance0, SV_CullDistance1 and so on with vectors.
			for (uint32_t cull = 0; cull < cull_distance_count; cull += 4)
			{
				uint32_t to_declare = cull_distance_count - cull;
				if (to_declare > 4)
					to_declare = 4;

				uint32_t semantic_index = cull / 4;

				static const char *types[] = { "float", "float2", "float3", "float4" };
				statement(types[to_declare - 1], " ", builtin_to_glsl(builtin, StorageClassOutput), semantic_index,
				          " : SV_CullDistance", semantic_index, ";");
			}
			break;

		case BuiltInPointSize:
			// If point_size_compat is enabled, just ignore PointSize.
			// PointSize does not exist in HLSL, but some code bases might want to be able to use these shaders,
			// even if it means working around the missing feature.
			if (hlsl_options.point_size_compat)
				break;
			else
				SPIRV_CROSS_THROW("Unsupported builtin in HLSL.");

		case BuiltInLayer:
			if (hlsl_options.shader_model < 50 || get_entry_point().model != ExecutionModelGeometry)
				SPIRV_CROSS_THROW("Render target array index output is only supported in GS 5.0 or higher.");
			type = "uint";
			semantic = "SV_RenderTargetArrayIndex";
			break;

		default:
			SPIRV_CROSS_THROW("Unsupported builtin in HLSL.");
		}

		if (type && semantic)
			statement(type, " ", builtin_to_glsl(builtin, StorageClassOutput), " : ", semantic, ";");
	});
}

void CompilerHLSL::emit_builtin_inputs_in_struct()
{
	bool legacy = hlsl_options.shader_model <= 30;
	active_input_builtins.for_each_bit([&](uint32_t i) {
		const char *type = nullptr;
		const char *semantic = nullptr;
		auto builtin = static_cast<BuiltIn>(i);
		switch (builtin)
		{
		case BuiltInFragCoord:
			type = "float4";
			semantic = legacy ? "VPOS" : "SV_Position";
			break;

		case BuiltInVertexId:
		case BuiltInVertexIndex:
			if (legacy)
				SPIRV_CROSS_THROW("Vertex index not supported in SM 3.0 or lower.");
			type = "uint";
			semantic = "SV_VertexID";
			break;

		case BuiltInPrimitiveId:
			type = "uint";
			semantic = "SV_PrimitiveID";
			break;

		case BuiltInInstanceId:
		case BuiltInInstanceIndex:
			if (legacy)
				SPIRV_CROSS_THROW("Instance index not supported in SM 3.0 or lower.");
			type = "uint";
			semantic = "SV_InstanceID";
			break;

		case BuiltInSampleId:
			if (legacy)
				SPIRV_CROSS_THROW("Sample ID not supported in SM 3.0 or lower.");
			type = "uint";
			semantic = "SV_SampleIndex";
			break;

		case BuiltInSampleMask:
			if (hlsl_options.shader_model < 50 || get_entry_point().model != ExecutionModelFragment)
				SPIRV_CROSS_THROW("Sample Mask input is only supported in PS 5.0 or higher.");
			type = "uint";
			semantic = "SV_Coverage";
			break;

		case BuiltInGlobalInvocationId:
			type = "uint3";
			semantic = "SV_DispatchThreadID";
			break;

		case BuiltInLocalInvocationId:
			type = "uint3";
			semantic = "SV_GroupThreadID";
			break;

		case BuiltInLocalInvocationIndex:
			type = "uint";
			semantic = "SV_GroupIndex";
			break;

		case BuiltInWorkgroupId:
			type = "uint3";
			semantic = "SV_GroupID";
			break;

		case BuiltInFrontFacing:
			type = "bool";
			semantic = "SV_IsFrontFace";
			break;

		case BuiltInViewIndex:
			if (hlsl_options.shader_model < 61 || (get_entry_point().model != ExecutionModelVertex && get_entry_point().model != ExecutionModelFragment))
				SPIRV_CROSS_THROW("View Index input is only supported in VS and PS 6.1 or higher.");
			type = "uint";
			semantic = "SV_ViewID";
			break;

		case BuiltInNumWorkgroups:
		case BuiltInSubgroupSize:
		case BuiltInSubgroupLocalInvocationId:
		case BuiltInSubgroupEqMask:
		case BuiltInSubgroupLtMask:
		case BuiltInSubgroupLeMask:
		case BuiltInSubgroupGtMask:
		case BuiltInSubgroupGeMask:
			// Handled specially.
			break;

		case BuiltInHelperInvocation:
			if (hlsl_options.shader_model < 50 || get_entry_point().model != ExecutionModelFragment)
				SPIRV_CROSS_THROW("Helper Invocation input is only supported in PS 5.0 or higher.");
			break;

		case BuiltInClipDistance:
			// HLSL is a bit weird here, use SV_ClipDistance0, SV_ClipDistance1 and so on with vectors.
			for (uint32_t clip = 0; clip < clip_distance_count; clip += 4)
			{
				uint32_t to_declare = clip_distance_count - clip;
				if (to_declare > 4)
					to_declare = 4;

				uint32_t semantic_index = clip / 4;

				static const char *types[] = { "float", "float2", "float3", "float4" };
				statement(types[to_declare - 1], " ", builtin_to_glsl(builtin, StorageClassInput), semantic_index,
				          " : SV_ClipDistance", semantic_index, ";");
			}
			break;

		case BuiltInCullDistance:
			// HLSL is a bit weird here, use SV_CullDistance0, SV_CullDistance1 and so on with vectors.
			for (uint32_t cull = 0; cull < cull_distance_count; cull += 4)
			{
				uint32_t to_declare = cull_distance_count - cull;
				if (to_declare > 4)
					to_declare = 4;

				uint32_t semantic_index = cull / 4;

				static const char *types[] = { "float", "float2", "float3", "float4" };
				statement(types[to_declare - 1], " ", builtin_to_glsl(builtin, StorageClassInput), semantic_index,
				          " : SV_CullDistance", semantic_index, ";");
			}
			break;

		case BuiltInPointCoord:
			// PointCoord is not supported, but provide a way to just ignore that, similar to PointSize.
			if (hlsl_options.point_coord_compat)
				break;
			else
				SPIRV_CROSS_THROW("Unsupported builtin in HLSL.");

		case BuiltInLayer:
			if (hlsl_options.shader_model < 50 || get_entry_point().model != ExecutionModelFragment)
				SPIRV_CROSS_THROW("Render target array index input is only supported in PS 5.0 or higher.");
			type = "uint";
			semantic = "SV_RenderTargetArrayIndex";
			break;

		default:
			SPIRV_CROSS_THROW("Unsupported builtin in HLSL.");
		}

		if (type && semantic)
			statement(type, " ", builtin_to_glsl(builtin, StorageClassInput), " : ", semantic, ";");
	});
}

uint32_t CompilerHLSL::type_to_consumed_locations(const SPIRType &type) const
{
	// TODO: Need to verify correctness.
	uint32_t elements = 0;

	if (type.basetype == SPIRType::Struct)
	{
		for (uint32_t i = 0; i < uint32_t(type.member_types.size()); i++)
			elements += type_to_consumed_locations(get<SPIRType>(type.member_types[i]));
	}
	else
	{
		uint32_t array_multiplier = 1;
		for (uint32_t i = 0; i < uint32_t(type.array.size()); i++)
		{
			if (type.array_size_literal[i])
				array_multiplier *= type.array[i];
			else
				array_multiplier *= evaluate_constant_u32(type.array[i]);
		}
		elements += array_multiplier * type.columns;
	}
	return elements;
}

string CompilerHLSL::to_interpolation_qualifiers(const Bitset &flags)
{
	string res;
	//if (flags & (1ull << DecorationSmooth))
	//    res += "linear ";
	if (flags.get(DecorationFlat))
		res += "nointerpolation ";
	if (flags.get(DecorationNoPerspective))
		res += "noperspective ";
	if (flags.get(DecorationCentroid))
		res += "centroid ";
	if (flags.get(DecorationPatch))
		res += "patch "; // Seems to be different in actual HLSL.
	if (flags.get(DecorationSample))
		res += "sample ";
	if (flags.get(DecorationInvariant) && backend.support_precise_qualifier)
		res += "precise "; // Not supported?

	return res;
}

std::string CompilerHLSL::to_semantic(uint32_t location, ExecutionModel em, StorageClass sc)
{
	if (em == ExecutionModelVertex && sc == StorageClassInput)
	{
		// We have a vertex attribute - we should look at remapping it if the user provided
		// vertex attribute hints.
		for (auto &attribute : remap_vertex_attributes)
			if (attribute.location == location)
				return attribute.semantic;
	}

	// Not a vertex attribute, or no remap_vertex_attributes entry.
	return join("TEXCOORD", location);
}

std::string CompilerHLSL::to_initializer_expression(const SPIRVariable &var)
{
	// We cannot emit static const initializer for block constants for practical reasons,
	// so just inline the initializer.
	// FIXME: There is a theoretical problem here if someone tries to composite extract
	// into this initializer since we don't declare it properly, but that is somewhat non-sensical.
	auto &type = get<SPIRType>(var.basetype);
	bool is_block = has_decoration(type.self, DecorationBlock);
	auto *c = maybe_get<SPIRConstant>(var.initializer);
	if (is_block && c)
		return constant_expression(*c);
	else
		return CompilerGLSL::to_initializer_expression(var);
}

void CompilerHLSL::emit_interface_block_member_in_struct(const SPIRVariable &var, uint32_t member_index,
                                                         uint32_t location,
                                                         std::unordered_set<uint32_t> &active_locations)
{
	auto &execution = get_entry_point();
	auto type = get<SPIRType>(var.basetype);
	auto semantic = to_semantic(location, execution.model, var.storage);
	auto mbr_name = join(to_name(type.self), "_", to_member_name(type, member_index));
	auto &mbr_type = get<SPIRType>(type.member_types[member_index]);

	statement(to_interpolation_qualifiers(get_member_decoration_bitset(type.self, member_index)),
	          type_to_glsl(mbr_type),
	          " ", mbr_name, type_to_array_glsl(mbr_type),
	          " : ", semantic, ";");

	// Structs and arrays should consume more locations.
	uint32_t consumed_locations = type_to_consumed_locations(mbr_type);
	for (uint32_t i = 0; i < consumed_locations; i++)
		active_locations.insert(location + i);
}

void CompilerHLSL::emit_interface_block_in_struct(const SPIRVariable &var, unordered_set<uint32_t> &active_locations)
{
	auto &execution = get_entry_point();
	auto type = get<SPIRType>(var.basetype);

	string binding;
	bool use_location_number = true;
	bool legacy = hlsl_options.shader_model <= 30;
	if (execution.model == ExecutionModelFragment && var.storage == StorageClassOutput)
	{
		// Dual-source blending is achieved in HLSL by emitting to SV_Target0 and 1.
		uint32_t index = get_decoration(var.self, DecorationIndex);
		uint32_t location = get_decoration(var.self, DecorationLocation);

		if (index != 0 && location != 0)
			SPIRV_CROSS_THROW("Dual-source blending is only supported on MRT #0 in HLSL.");

		binding = join(legacy ? "COLOR" : "SV_Target", location + index);
		use_location_number = false;
		if (legacy) // COLOR must be a four-component vector on legacy shader model targets (HLSL ERR_COLOR_4COMP)
			type.vecsize = 4;
	}

	const auto get_vacant_location = [&]() -> uint32_t {
		for (uint32_t i = 0; i < 64; i++)
			if (!active_locations.count(i))
				return i;
		SPIRV_CROSS_THROW("All locations from 0 to 63 are exhausted.");
	};

	bool need_matrix_unroll = var.storage == StorageClassInput && execution.model == ExecutionModelVertex;

	auto name = to_name(var.self);
	if (use_location_number)
	{
		uint32_t location_number;

		// If an explicit location exists, use it with TEXCOORD[N] semantic.
		// Otherwise, pick a vacant location.
		if (has_decoration(var.self, DecorationLocation))
			location_number = get_decoration(var.self, DecorationLocation);
		else
			location_number = get_vacant_location();

		// Allow semantic remap if specified.
		auto semantic = to_semantic(location_number, execution.model, var.storage);

		if (need_matrix_unroll && type.columns > 1)
		{
			if (!type.array.empty())
				SPIRV_CROSS_THROW("Arrays of matrices used as input/output. This is not supported.");

			// Unroll matrices.
			for (uint32_t i = 0; i < type.columns; i++)
			{
				SPIRType newtype = type;
				newtype.columns = 1;

				string effective_semantic;
				if (hlsl_options.flatten_matrix_vertex_input_semantics)
					effective_semantic = to_semantic(location_number, execution.model, var.storage);
				else
					effective_semantic = join(semantic, "_", i);

				statement(to_interpolation_qualifiers(get_decoration_bitset(var.self)),
				          variable_decl(newtype, join(name, "_", i)), " : ", effective_semantic, ";");
				active_locations.insert(location_number++);
			}
		}
		else
		{
			statement(to_interpolation_qualifiers(get_decoration_bitset(var.self)), variable_decl(type, name), " : ",
			          semantic, ";");

			// Structs and arrays should consume more locations.
			uint32_t consumed_locations = type_to_consumed_locations(type);
			for (uint32_t i = 0; i < consumed_locations; i++)
				active_locations.insert(location_number + i);
		}
	}
	else
		statement(variable_decl(type, name), " : ", binding, ";");
}

std::string CompilerHLSL::builtin_to_glsl(spv::BuiltIn builtin, spv::StorageClass storage)
{
	switch (builtin)
	{
	case BuiltInVertexId:
		return "gl_VertexID";
	case BuiltInInstanceId:
		return "gl_InstanceID";
	case BuiltInNumWorkgroups:
	{
		if (!num_workgroups_builtin)
			SPIRV_CROSS_THROW("NumWorkgroups builtin is used, but remap_num_workgroups_builtin() was not called. "
			                  "Cannot emit code for this builtin.");

		auto &var = get<SPIRVariable>(num_workgroups_builtin);
		auto &type = get<SPIRType>(var.basetype);
		auto ret = join(to_name(num_workgroups_builtin), "_", get_member_name(type.self, 0));
		ParsedIR::sanitize_underscores(ret);
		return ret;
	}
	case BuiltInPointCoord:
		// Crude hack, but there is no real alternative. This path is only enabled if point_coord_compat is set.
		return "float2(0.5f, 0.5f)";
	case BuiltInSubgroupLocalInvocationId:
		return "WaveGetLaneIndex()";
	case BuiltInSubgroupSize:
		return "WaveGetLaneCount()";
	case BuiltInHelperInvocation:
		return "IsHelperLane()";

	default:
		return CompilerGLSL::builtin_to_glsl(builtin, storage);
	}
}

void CompilerHLSL::emit_builtin_variables()
{
	Bitset builtins = active_input_builtins;
	builtins.merge_or(active_output_builtins);

	bool need_base_vertex_info = false;

	std::unordered_map<uint32_t, ID> builtin_to_initializer;
	ir.for_each_typed_id<SPIRVariable>([&](uint32_t, SPIRVariable &var) {
		if (!is_builtin_variable(var) || var.storage != StorageClassOutput || !var.initializer)
			return;

		auto *c = this->maybe_get<SPIRConstant>(var.initializer);
		if (!c)
			return;

		auto &type = this->get<SPIRType>(var.basetype);
		if (type.basetype == SPIRType::Struct)
		{
			uint32_t member_count = uint32_t(type.member_types.size());
			for (uint32_t i = 0; i < member_count; i++)
			{
				if (has_member_decoration(type.self, i, DecorationBuiltIn))
				{
					builtin_to_initializer[get_member_decoration(type.self, i, DecorationBuiltIn)] =
						c->subconstants[i];
				}
			}
		}
		else if (has_decoration(var.self, DecorationBuiltIn))
			builtin_to_initializer[get_decoration(var.self, DecorationBuiltIn)] = var.initializer;
	});

	// Emit global variables for the interface variables which are statically used by the shader.
	builtins.for_each_bit([&](uint32_t i) {
		const char *type = nullptr;
		auto builtin = static_cast<BuiltIn>(i);
		uint32_t array_size = 0;

		string init_expr;
		auto init_itr = builtin_to_initializer.find(builtin);
		if (init_itr != builtin_to_initializer.end())
			init_expr = join(" = ", to_expression(init_itr->second));

		switch (builtin)
		{
		case BuiltInFragCoord:
		case BuiltInPosition:
			type = "float4";
			break;

		case BuiltInFragDepth:
			type = "float";
			break;

		case BuiltInVertexId:
		case BuiltInVertexIndex:
		case BuiltInInstanceIndex:
			type = "int";
			if (hlsl_options.support_nonzero_base_vertex_base_instance)
				need_base_vertex_info = true;
			break;

		case BuiltInInstanceId:
		case BuiltInSampleId:
			type = "int";
			break;

		case BuiltInPointSize:
			if (hlsl_options.point_size_compat)
			{
				// Just emit the global variable, it will be ignored.
				type = "float";
				break;
			}
			else
				SPIRV_CROSS_THROW(join("Unsupported builtin in HLSL: ", unsigned(builtin)));

		case BuiltInGlobalInvocationId:
		case BuiltInLocalInvocationId:
		case BuiltInWorkgroupId:
			type = "uint3";
			break;

		case BuiltInLocalInvocationIndex:
			type = "uint";
			break;

		case BuiltInFrontFacing:
			type = "bool";
			break;

		case BuiltInNumWorkgroups:
		case BuiltInPointCoord:
			// Handled specially.
			break;

		case BuiltInSubgroupLocalInvocationId:
		case BuiltInSubgroupSize:
			if (hlsl_options.shader_model < 60)
				SPIRV_CROSS_THROW("Need SM 6.0 for Wave ops.");
			break;

		case BuiltInSubgroupEqMask:
		case BuiltInSubgroupLtMask:
		case BuiltInSubgroupLeMask:
		case BuiltInSubgroupGtMask:
		case BuiltInSubgroupGeMask:
			if (hlsl_options.shader_model < 60)
				SPIRV_CROSS_THROW("Need SM 6.0 for Wave ops.");
			type = "uint4";
			break;

		case BuiltInHelperInvocation:
			if (hlsl_options.shader_model < 50)
				SPIRV_CROSS_THROW("Need SM 5.0 for Helper Invocation.");
			break;

		case BuiltInClipDistance:
			array_size = clip_distance_count;
			type = "float";
			break;

		case BuiltInCullDistance:
			array_size = cull_distance_count;
			type = "float";
			break;

		case BuiltInSampleMask:
			type = "int";
			break;

		case BuiltInPrimitiveId:
		case BuiltInViewIndex:
		case BuiltInLayer:
			type = "uint";
			break;

		default:
			SPIRV_CROSS_THROW(join("Unsupported builtin in HLSL: ", unsigned(builtin)));
		}

		StorageClass storage = active_input_builtins.get(i) ? StorageClassInput : StorageClassOutput;

		if (type)
		{
			if (array_size)
				statement("static ", type, " ", builtin_to_glsl(builtin, storage), "[", array_size, "]", init_expr, ";");
			else
				statement("static ", type, " ", builtin_to_glsl(builtin, storage), init_expr, ";");
		}

		// SampleMask can be both in and out with sample builtin, in this case we have already
		// declared the input variable and we need to add the output one now.
		if (builtin == BuiltInSampleMask && storage == StorageClassInput && this->active_output_builtins.get(i))
		{
			statement("static ", type, " ", this->builtin_to_glsl(builtin, StorageClassOutput), init_expr, ";");
		}
	});

	if (need_base_vertex_info)
	{
		statement("cbuffer SPIRV_Cross_VertexInfo");
		begin_scope();
		statement("int SPIRV_Cross_BaseVertex;");
		statement("int SPIRV_Cross_BaseInstance;");
		end_scope_decl();
		statement("");
	}
}

void CompilerHLSL::emit_composite_constants()
{
	// HLSL cannot declare structs or arrays inline, so we must move them out to
	// global constants directly.
	bool emitted = false;

	ir.for_each_typed_id<SPIRConstant>([&](uint32_t, SPIRConstant &c) {
		if (c.specialization)
			return;

		auto &type = this->get<SPIRType>(c.constant_type);

		if (type.basetype == SPIRType::Struct && is_builtin_type(type))
			return;

		if (type.basetype == SPIRType::Struct || !type.array.empty())
		{
			add_resource_name(c.self);
			auto name = to_name(c.self);
			statement("static const ", variable_decl(type, name), " = ", constant_expression(c), ";");
			emitted = true;
		}
	});

	if (emitted)
		statement("");
}

void CompilerHLSL::emit_specialization_constants_and_structs()
{
	bool emitted = false;
	SpecializationConstant wg_x, wg_y, wg_z;
	ID workgroup_size_id = get_work_group_size_specialization_constants(wg_x, wg_y, wg_z);

	std::unordered_set<TypeID> io_block_types;
	ir.for_each_typed_id<SPIRVariable>([&](uint32_t, const SPIRVariable &var) {
		auto &type = this->get<SPIRType>(var.basetype);
		if ((var.storage == StorageClassInput || var.storage == StorageClassOutput) &&
		    !var.remapped_variable && type.pointer && !is_builtin_variable(var) &&
		    interface_variable_exists_in_entry_point(var.self) &&
		    has_decoration(type.self, DecorationBlock))
		{
			io_block_types.insert(type.self);
		}
	});

	auto loop_lock = ir.create_loop_hard_lock();
	for (auto &id_ : ir.ids_for_constant_or_type)
	{
		auto &id = ir.ids[id_];

		if (id.get_type() == TypeConstant)
		{
			auto &c = id.get<SPIRConstant>();

			if (c.self == workgroup_size_id)
			{
				statement("static const uint3 gl_WorkGroupSize = ",
				          constant_expression(get<SPIRConstant>(workgroup_size_id)), ";");
				emitted = true;
			}
			else if (c.specialization)
			{
				auto &type = get<SPIRType>(c.constant_type);
				add_resource_name(c.self);
				auto name = to_name(c.self);

				if (has_decoration(c.self, DecorationSpecId))
				{
					// HLSL does not support specialization constants, so fallback to macros.
					c.specialization_constant_macro_name =
							constant_value_macro_name(get_decoration(c.self, DecorationSpecId));

					statement("#ifndef ", c.specialization_constant_macro_name);
					statement("#define ", c.specialization_constant_macro_name, " ", constant_expression(c));
					statement("#endif");
					statement("static const ", variable_decl(type, name), " = ", c.specialization_constant_macro_name, ";");
				}
				else
					statement("static const ", variable_decl(type, name), " = ", constant_expression(c), ";");

				emitted = true;
			}
		}
		else if (id.get_type() == TypeConstantOp)
		{
			auto &c = id.get<SPIRConstantOp>();
			auto &type = get<SPIRType>(c.basetype);
			add_resource_name(c.self);
			auto name = to_name(c.self);
			statement("static const ", variable_decl(type, name), " = ", constant_op_expression(c), ";");
			emitted = true;
		}
		else if (id.get_type() == TypeType)
		{
			auto &type = id.get<SPIRType>();
			bool is_non_io_block = has_decoration(type.self, DecorationBlock) &&
			                       io_block_types.count(type.self) == 0;
			bool is_buffer_block = has_decoration(type.self, DecorationBufferBlock);
			if (type.basetype == SPIRType::Struct && type.array.empty() &&
			    !type.pointer && !is_non_io_block && !is_buffer_block)
			{
				if (emitted)
					statement("");
				emitted = false;

				emit_struct(type);
			}
		}
	}

	if (emitted)
		statement("");
}

void CompilerHLSL::replace_illegal_names()
{
	static const unordered_set<string> keywords = {
		// Additional HLSL specific keywords.
		// From https://docs.microsoft.com/en-US/windows/win32/direct3dhlsl/dx-graphics-hlsl-appendix-keywords
		"AppendStructuredBuffer", "asm", "asm_fragment",
		"BlendState", "bool", "break", "Buffer", "ByteAddressBuffer",
		"case", "cbuffer", "centroid", "class", "column_major", "compile",
		"compile_fragment", "CompileShader", "const", "continue", "ComputeShader",
		"ConsumeStructuredBuffer",
		"default", "DepthStencilState", "DepthStencilView", "discard", "do",
		"double", "DomainShader", "dword",
		"else", "export", "false", "float", "for", "fxgroup",
		"GeometryShader", "groupshared", "half", "HullShader",
		"if", "in", "inline", "inout", "InputPatch", "int", "interface",
		"line", "lineadj", "linear", "LineStream",
		"matrix", "min16float", "min10float", "min16int", "min16uint",
		"namespace", "nointerpolation", "noperspective", "NULL",
		"out", "OutputPatch",
		"packoffset", "pass", "pixelfragment", "PixelShader", "point",
		"PointStream", "precise", "RasterizerState", "RenderTargetView",
		"return", "register", "row_major", "RWBuffer", "RWByteAddressBuffer",
		"RWStructuredBuffer", "RWTexture1D", "RWTexture1DArray", "RWTexture2D",
		"RWTexture2DArray", "RWTexture3D", "sample", "sampler", "SamplerState",
		"SamplerComparisonState", "shared", "snorm", "stateblock", "stateblock_state",
		"static", "string", "struct", "switch", "StructuredBuffer", "tbuffer",
		"technique", "technique10", "technique11", "texture", "Texture1D",
		"Texture1DArray", "Texture2D", "Texture2DArray", "Texture2DMS", "Texture2DMSArray",
		"Texture3D", "TextureCube", "TextureCubeArray", "true", "typedef", "triangle",
		"triangleadj", "TriangleStream", "uint", "uniform", "unorm", "unsigned",
		"vector", "vertexfragment", "VertexShader", "void", "volatile", "while",
	};

	CompilerGLSL::replace_illegal_names(keywords);
	CompilerGLSL::replace_illegal_names();
}

void CompilerHLSL::declare_undefined_values()
{
	bool emitted = false;
	ir.for_each_typed_id<SPIRUndef>([&](uint32_t, const SPIRUndef &undef) {
		auto &type = this->get<SPIRType>(undef.basetype);
		// OpUndef can be void for some reason ...
		if (type.basetype == SPIRType::Void)
			return;

		string initializer;
		if (options.force_zero_initialized_variables && type_can_zero_initialize(type))
			initializer = join(" = ", to_zero_initialized_expression(undef.basetype));

		statement("static ", variable_decl(type, to_name(undef.self), undef.self), initializer, ";");
		emitted = true;
	});

	if (emitted)
		statement("");
}

void CompilerHLSL::emit_resources()
{
	auto &execution = get_entry_point();

	replace_illegal_names();

	emit_specialization_constants_and_structs();
	emit_composite_constants();

	bool emitted = false;

	// Output UBOs and SSBOs
	ir.for_each_typed_id<SPIRVariable>([&](uint32_t, SPIRVariable &var) {
		auto &type = this->get<SPIRType>(var.basetype);

		bool is_block_storage = type.storage == StorageClassStorageBuffer || type.storage == StorageClassUniform;
		bool has_block_flags = ir.meta[type.self].decoration.decoration_flags.get(DecorationBlock) ||
		                       ir.meta[type.self].decoration.decoration_flags.get(DecorationBufferBlock);

		if (var.storage != StorageClassFunction && type.pointer && is_block_storage && !is_hidden_variable(var) &&
		    has_block_flags)
		{
			emit_buffer_block(var);
			emitted = true;
		}
	});

	// Output push constant blocks
	ir.for_each_typed_id<SPIRVariable>([&](uint32_t, SPIRVariable &var) {
		auto &type = this->get<SPIRType>(var.basetype);
		if (var.storage != StorageClassFunction && type.pointer && type.storage == StorageClassPushConstant &&
		    !is_hidden_variable(var))
		{
			emit_push_constant_block(var);
			emitted = true;
		}
	});

	if (execution.model == ExecutionModelVertex && hlsl_options.shader_model <= 30 &&
	    active_output_builtins.get(BuiltInPosition))
	{
		statement("uniform float4 gl_HalfPixel;");
		emitted = true;
	}

	bool skip_separate_image_sampler = !combined_image_samplers.empty() || hlsl_options.shader_model <= 30;

	// Output Uniform Constants (values, samplers, images, etc).
	ir.for_each_typed_id<SPIRVariable>([&](uint32_t, SPIRVariable &var) {
		auto &type = this->get<SPIRType>(var.basetype);

		// If we're remapping separate samplers and images, only emit the combined samplers.
		if (skip_separate_image_sampler)
		{
			// Sampler buffers are always used without a sampler, and they will also work in regular D3D.
			bool sampler_buffer = type.basetype == SPIRType::Image && type.image.dim == DimBuffer;
			bool separate_image = type.basetype == SPIRType::Image && type.image.sampled == 1;
			bool separate_sampler = type.basetype == SPIRType::Sampler;
			if (!sampler_buffer && (separate_image || separate_sampler))
				return;
		}

		if (var.storage != StorageClassFunction && !is_builtin_variable(var) && !var.remapped_variable &&
		    type.pointer && (type.storage == StorageClassUniformConstant || type.storage == StorageClassAtomicCounter) &&
		    !is_hidden_variable(var))
		{
			emit_uniform(var);
			emitted = true;
		}
	});

	if (emitted)
		statement("");
	emitted = false;

	// Emit builtin input and output variables here.
	emit_builtin_variables();

	ir.for_each_typed_id<SPIRVariable>([&](uint32_t, SPIRVariable &var) {
		auto &type = this->get<SPIRType>(var.basetype);

		if (var.storage != StorageClassFunction && !var.remapped_variable && type.pointer &&
		    (var.storage == StorageClassInput || var.storage == StorageClassOutput) && !is_builtin_variable(var) &&
		    interface_variable_exists_in_entry_point(var.self))
		{
			// Builtin variables are handled separately.
			emit_interface_block_globally(var);
			emitted = true;
		}
	});

	if (emitted)
		statement("");
	emitted = false;

	require_input = false;
	require_output = false;
	unordered_set<uint32_t> active_inputs;
	unordered_set<uint32_t> active_outputs;

	struct IOVariable
	{
		const SPIRVariable *var;
		uint32_t location;
		uint32_t block_member_index;
		bool block;
	};

	SmallVector<IOVariable> input_variables;
	SmallVector<IOVariable> output_variables;

	ir.for_each_typed_id<SPIRVariable>([&](uint32_t, SPIRVariable &var) {
		auto &type = this->get<SPIRType>(var.basetype);
		bool block = has_decoration(type.self, DecorationBlock);

		if (var.storage != StorageClassInput && var.storage != StorageClassOutput)
			return;

		if (!var.remapped_variable && type.pointer && !is_builtin_variable(var) &&
		    interface_variable_exists_in_entry_point(var.self))
		{
			if (block)
			{
				for (uint32_t i = 0; i < uint32_t(type.member_types.size()); i++)
				{
					uint32_t location = get_declared_member_location(var, i, false);
					if (var.storage == StorageClassInput)
						input_variables.push_back({ &var, location, i, true });
					else
						output_variables.push_back({ &var, location, i, true });
				}
			}
			else
			{
				uint32_t location = get_decoration(var.self, DecorationLocation);
				if (var.storage == StorageClassInput)
					input_variables.push_back({ &var, location, 0, false });
				else
					output_variables.push_back({ &var, location, 0, false });
			}
		}
	});

	const auto variable_compare = [&](const IOVariable &a, const IOVariable &b) -> bool {
		// Sort input and output variables based on, from more robust to less robust:
		// - Location
		// - Variable has a location
		// - Name comparison
		// - Variable has a name
		// - Fallback: ID
		bool has_location_a = a.block || has_decoration(a.var->self, DecorationLocation);
		bool has_location_b = b.block || has_decoration(b.var->self, DecorationLocation);

		if (has_location_a && has_location_b)
			return a.location < b.location;
		else if (has_location_a && !has_location_b)
			return true;
		else if (!has_location_a && has_location_b)
			return false;

		const auto &name1 = to_name(a.var->self);
		const auto &name2 = to_name(b.var->self);

		if (name1.empty() && name2.empty())
			return a.var->self < b.var->self;
		else if (name1.empty())
			return true;
		else if (name2.empty())
			return false;

		return name1.compare(name2) < 0;
	};

	auto input_builtins = active_input_builtins;
	input_builtins.clear(BuiltInNumWorkgroups);
	input_builtins.clear(BuiltInPointCoord);
	input_builtins.clear(BuiltInSubgroupSize);
	input_builtins.clear(BuiltInSubgroupLocalInvocationId);
	input_builtins.clear(BuiltInSubgroupEqMask);
	input_builtins.clear(BuiltInSubgroupLtMask);
	input_builtins.clear(BuiltInSubgroupLeMask);
	input_builtins.clear(BuiltInSubgroupGtMask);
	input_builtins.clear(BuiltInSubgroupGeMask);

	if (!input_variables.empty() || !input_builtins.empty())
	{
		require_input = true;
		statement("struct SPIRV_Cross_Input");

		begin_scope();
		sort(input_variables.begin(), input_variables.end(), variable_compare);
		for (auto &var : input_variables)
		{
			if (var.block)
				emit_interface_block_member_in_struct(*var.var, var.block_member_index, var.location, active_inputs);
			else
				emit_interface_block_in_struct(*var.var, active_inputs);
		}
		emit_builtin_inputs_in_struct();
		end_scope_decl();
		statement("");
	}

	if (!output_variables.empty() || !active_output_builtins.empty())
	{
		require_output = true;
		statement("struct SPIRV_Cross_Output");

		begin_scope();
		sort(output_variables.begin(), output_variables.end(), variable_compare);
		for (auto &var : output_variables)
		{
			if (var.block)
				emit_interface_block_member_in_struct(*var.var, var.block_member_index, var.location, active_outputs);
			else
				emit_interface_block_in_struct(*var.var, active_outputs);
		}
		emit_builtin_outputs_in_struct();
		end_scope_decl();
		statement("");
	}

	// Global variables.
	for (auto global : global_variables)
	{
		auto &var = get<SPIRVariable>(global);
		if (is_hidden_variable(var, true))
			continue;

		if (var.storage != StorageClassOutput)
		{
			if (!variable_is_lut(var))
			{
				add_resource_name(var.self);

				const char *storage = nullptr;
				switch (var.storage)
				{
				case StorageClassWorkgroup:
					storage = "groupshared";
					break;

				default:
					storage = "static";
					break;
				}

				string initializer;
				if (options.force_zero_initialized_variables && var.storage == StorageClassPrivate &&
				    !var.initializer && !var.static_expression && type_can_zero_initialize(get_variable_data_type(var)))
				{
					initializer = join(" = ", to_zero_initialized_expression(get_variable_data_type_id(var)));
				}
				statement(storage, " ", variable_decl(var), initializer, ";");

				emitted = true;
			}
		}
	}

	if (emitted)
		statement("");

	declare_undefined_values();

	if (requires_op_fmod)
	{
		static const char *types[] = {
			"float",
			"float2",
			"float3",
			"float4",
		};

		for (auto &type : types)
		{
			statement(type, " mod(", type, " x, ", type, " y)");
			begin_scope();
			statement("return x - y * floor(x / y);");
			end_scope();
			statement("");
		}
	}

	emit_texture_size_variants(required_texture_size_variants.srv, "4", false, "");
	for (uint32_t norm = 0; norm < 3; norm++)
	{
		for (uint32_t comp = 0; comp < 4; comp++)
		{
			static const char *qualifiers[] = { "", "unorm ", "snorm " };
			static const char *vecsizes[] = { "", "2", "3", "4" };
			emit_texture_size_variants(required_texture_size_variants.uav[norm][comp], vecsizes[comp], true,
			                           qualifiers[norm]);
		}
	}

	if (requires_fp16_packing)
	{
		// HLSL does not pack into a single word sadly :(
		statement("uint spvPackHalf2x16(float2 value)");
		begin_scope();
		statement("uint2 Packed = f32tof16(value);");
		statement("return Packed.x | (Packed.y << 16);");
		end_scope();
		statement("");

		statement("float2 spvUnpackHalf2x16(uint value)");
		begin_scope();
		statement("return f16tof32(uint2(value & 0xffff, value >> 16));");
		end_scope();
		statement("");
	}

	if (requires_uint2_packing)
	{
		statement("uint64_t spvPackUint2x32(uint2 value)");
		begin_scope();
		statement("return (uint64_t(value.y) << 32) | uint64_t(value.x);");
		end_scope();
		statement("");

		statement("uint2 spvUnpackUint2x32(uint64_t value)");
		begin_scope();
		statement("uint2 Unpacked;");
		statement("Unpacked.x = uint(value & 0xffffffff);");
		statement("Unpacked.y = uint(value >> 32);");
		statement("return Unpacked;");
		end_scope();
		statement("");
	}

	if (requires_explicit_fp16_packing)
	{
		// HLSL does not pack into a single word sadly :(
		statement("uint spvPackFloat2x16(min16float2 value)");
		begin_scope();
		statement("uint2 Packed = f32tof16(value);");
		statement("return Packed.x | (Packed.y << 16);");
		end_scope();
		statement("");

		statement("min16float2 spvUnpackFloat2x16(uint value)");
		begin_scope();
		statement("return min16float2(f16tof32(uint2(value & 0xffff, value >> 16)));");
		end_scope();
		statement("");
	}

	// HLSL does not seem to have builtins for these operation, so roll them by hand ...
	if (requires_unorm8_packing)
	{
		statement("uint spvPackUnorm4x8(float4 value)");
		begin_scope();
		statement("uint4 Packed = uint4(round(saturate(value) * 255.0));");
		statement("return Packed.x | (Packed.y << 8) | (Packed.z << 16) | (Packed.w << 24);");
		end_scope();
		statement("");

		statement("float4 spvUnpackUnorm4x8(uint value)");
		begin_scope();
		statement("uint4 Packed = uint4(value & 0xff, (value >> 8) & 0xff, (value >> 16) & 0xff, value >> 24);");
		statement("return float4(Packed) / 255.0;");
		end_scope();
		statement("");
	}

	if (requires_snorm8_packing)
	{
		statement("uint spvPackSnorm4x8(float4 value)");
		begin_scope();
		statement("int4 Packed = int4(round(clamp(value, -1.0, 1.0) * 127.0)) & 0xff;");
		statement("return uint(Packed.x | (Packed.y << 8) | (Packed.z << 16) | (Packed.w << 24));");
		end_scope();
		statement("");

		statement("float4 spvUnpackSnorm4x8(uint value)");
		begin_scope();
		statement("int SignedValue = int(value);");
		statement("int4 Packed = int4(SignedValue << 24, SignedValue << 16, SignedValue << 8, SignedValue) >> 24;");
		statement("return clamp(float4(Packed) / 127.0, -1.0, 1.0);");
		end_scope();
		statement("");
	}

	if (requires_unorm16_packing)
	{
		statement("uint spvPackUnorm2x16(float2 value)");
		begin_scope();
		statement("uint2 Packed = uint2(round(saturate(value) * 65535.0));");
		statement("return Packed.x | (Packed.y << 16);");
		end_scope();
		statement("");

		statement("float2 spvUnpackUnorm2x16(uint value)");
		begin_scope();
		statement("uint2 Packed = uint2(value & 0xffff, value >> 16);");
		statement("return float2(Packed) / 65535.0;");
		end_scope();
		statement("");
	}

	if (requires_snorm16_packing)
	{
		statement("uint spvPackSnorm2x16(float2 value)");
		begin_scope();
		statement("int2 Packed = int2(round(clamp(value, -1.0, 1.0) * 32767.0)) & 0xffff;");
		statement("return uint(Packed.x | (Packed.y << 16));");
		end_scope();
		statement("");

		statement("float2 spvUnpackSnorm2x16(uint value)");
		begin_scope();
		statement("int SignedValue = int(value);");
		statement("int2 Packed = int2(SignedValue << 16, SignedValue) >> 16;");
		statement("return clamp(float2(Packed) / 32767.0, -1.0, 1.0);");
		end_scope();
		statement("");
	}

	if (requires_bitfield_insert)
	{
		static const char *types[] = { "uint", "uint2", "uint3", "uint4" };
		for (auto &type : types)
		{
			statement(type, " spvBitfieldInsert(", type, " Base, ", type, " Insert, uint Offset, uint Count)");
			begin_scope();
			statement("uint Mask = Count == 32 ? 0xffffffff : (((1u << Count) - 1) << (Offset & 31));");
			statement("return (Base & ~Mask) | ((Insert << Offset) & Mask);");
			end_scope();
			statement("");
		}
	}

	if (requires_bitfield_extract)
	{
		static const char *unsigned_types[] = { "uint", "uint2", "uint3", "uint4" };
		for (auto &type : unsigned_types)
		{
			statement(type, " spvBitfieldUExtract(", type, " Base, uint Offset, uint Count)");
			begin_scope();
			statement("uint Mask = Count == 32 ? 0xffffffff : ((1 << Count) - 1);");
			statement("return (Base >> Offset) & Mask;");
			end_scope();
			statement("");
		}

		// In this overload, we will have to do sign-extension, which we will emulate by shifting up and down.
		static const char *signed_types[] = { "int", "int2", "int3", "int4" };
		for (auto &type : signed_types)
		{
			statement(type, " spvBitfieldSExtract(", type, " Base, int Offset, int Count)");
			begin_scope();
			statement("int Mask = Count == 32 ? -1 : ((1 << Count) - 1);");
			statement(type, " Masked = (Base >> Offset) & Mask;");
			statement("int ExtendShift = (32 - Count) & 31;");
			statement("return (Masked << ExtendShift) >> ExtendShift;");
			end_scope();
			statement("");
		}
	}

	if (requires_inverse_2x2)
	{
		statement("// Returns the inverse of a matrix, by using the algorithm of calculating the classical");
		statement("// adjoint and dividing by the determinant. The contents of the matrix are changed.");
		statement("float2x2 spvInverse(float2x2 m)");
		begin_scope();
		statement("float2x2 adj;	// The adjoint matrix (inverse after dividing by determinant)");
		statement_no_indent("");
		statement("// Create the transpose of the cofactors, as the classical adjoint of the matrix.");
		statement("adj[0][0] =  m[1][1];");
		statement("adj[0][1] = -m[0][1];");
		statement_no_indent("");
		statement("adj[1][0] = -m[1][0];");
		statement("adj[1][1] =  m[0][0];");
		statement_no_indent("");
		statement("// Calculate the determinant as a combination of the cofactors of the first row.");
		statement("float det = (adj[0][0] * m[0][0]) + (adj[0][1] * m[1][0]);");
		statement_no_indent("");
		statement("// Divide the classical adjoint matrix by the determinant.");
		statement("// If determinant is zero, matrix is not invertable, so leave it unchanged.");
		statement("return (det != 0.0f) ? (adj * (1.0f / det)) : m;");
		end_scope();
		statement("");
	}

	if (requires_inverse_3x3)
	{
		statement("// Returns the determinant of a 2x2 matrix.");
		statement("float spvDet2x2(float a1, float a2, float b1, float b2)");
		begin_scope();
		statement("return a1 * b2 - b1 * a2;");
		end_scope();
		statement_no_indent("");
		statement("// Returns the inverse of a matrix, by using the algorithm of calculating the classical");
		statement("// adjoint and dividing by the determinant. The contents of the matrix are changed.");
		statement("float3x3 spvInverse(float3x3 m)");
		begin_scope();
		statement("float3x3 adj;	// The adjoint matrix (inverse after dividing by determinant)");
		statement_no_indent("");
		statement("// Create the transpose of the cofactors, as the classical adjoint of the matrix.");
		statement("adj[0][0] =  spvDet2x2(m[1][1], m[1][2], m[2][1], m[2][2]);");
		statement("adj[0][1] = -spvDet2x2(m[0][1], m[0][2], m[2][1], m[2][2]);");
		statement("adj[0][2] =  spvDet2x2(m[0][1], m[0][2], m[1][1], m[1][2]);");
		statement_no_indent("");
		statement("adj[1][0] = -spvDet2x2(m[1][0], m[1][2], m[2][0], m[2][2]);");
		statement("adj[1][1] =  spvDet2x2(m[0][0], m[0][2], m[2][0], m[2][2]);");
		statement("adj[1][2] = -spvDet2x2(m[0][0], m[0][2], m[1][0], m[1][2]);");
		statement_no_indent("");
		statement("adj[2][0] =  spvDet2x2(m[1][0], m[1][1], m[2][0], m[2][1]);");
		statement("adj[2][1] = -spvDet2x2(m[0][0], m[0][1], m[2][0], m[2][1]);");
		statement("adj[2][2] =  spvDet2x2(m[0][0], m[0][1], m[1][0], m[1][1]);");
		statement_no_indent("");
		statement("// Calculate the determinant as a combination of the cofactors of the first row.");
		statement("float det = (adj[0][0] * m[0][0]) + (adj[0][1] * m[1][0]) + (adj[0][2] * m[2][0]);");
		statement_no_indent("");
		statement("// Divide the classical adjoint matrix by the determinant.");
		statement("// If determinant is zero, matrix is not invertable, so leave it unchanged.");
		statement("return (det != 0.0f) ? (adj * (1.0f / det)) : m;");
		end_scope();
		statement("");
	}

	if (requires_inverse_4x4)
	{
		if (!requires_inverse_3x3)
		{
			statement("// Returns the determinant of a 2x2 matrix.");
			statement("float spvDet2x2(float a1, float a2, float b1, float b2)");
			begin_scope();
			statement("return a1 * b2 - b1 * a2;");
			end_scope();
			statement("");
		}

		statement("// Returns the determinant of a 3x3 matrix.");
		statement("float spvDet3x3(float a1, float a2, float a3, float b1, float b2, float b3, float c1, "
		          "float c2, float c3)");
		begin_scope();
		statement("return a1 * spvDet2x2(b2, b3, c2, c3) - b1 * spvDet2x2(a2, a3, c2, c3) + c1 * "
		          "spvDet2x2(a2, a3, "
		          "b2, b3);");
		end_scope();
		statement_no_indent("");
		statement("// Returns the inverse of a matrix, by using the algorithm of calculating the classical");
		statement("// adjoint and dividing by the determinant. The contents of the matrix are changed.");
		statement("float4x4 spvInverse(float4x4 m)");
		begin_scope();
		statement("float4x4 adj;	// The adjoint matrix (inverse after dividing by determinant)");
		statement_no_indent("");
		statement("// Create the transpose of the cofactors, as the classical adjoint of the matrix.");
		statement(
		    "adj[0][0] =  spvDet3x3(m[1][1], m[1][2], m[1][3], m[2][1], m[2][2], m[2][3], m[3][1], m[3][2], "
		    "m[3][3]);");
		statement(
		    "adj[0][1] = -spvDet3x3(m[0][1], m[0][2], m[0][3], m[2][1], m[2][2], m[2][3], m[3][1], m[3][2], "
		    "m[3][3]);");
		statement(
		    "adj[0][2] =  spvDet3x3(m[0][1], m[0][2], m[0][3], m[1][1], m[1][2], m[1][3], m[3][1], m[3][2], "
		    "m[3][3]);");
		statement(
		    "adj[0][3] = -spvDet3x3(m[0][1], m[0][2], m[0][3], m[1][1], m[1][2], m[1][3], m[2][1], m[2][2], "
		    "m[2][3]);");
		statement_no_indent("");
		statement(
		    "adj[1][0] = -spvDet3x3(m[1][0], m[1][2], m[1][3], m[2][0], m[2][2], m[2][3], m[3][0], m[3][2], "
		    "m[3][3]);");
		statement(
		    "adj[1][1] =  spvDet3x3(m[0][0], m[0][2], m[0][3], m[2][0], m[2][2], m[2][3], m[3][0], m[3][2], "
		    "m[3][3]);");
		statement(
		    "adj[1][2] = -spvDet3x3(m[0][0], m[0][2], m[0][3], m[1][0], m[1][2], m[1][3], m[3][0], m[3][2], "
		    "m[3][3]);");
		statement(
		    "adj[1][3] =  spvDet3x3(m[0][0], m[0][2], m[0][3], m[1][0], m[1][2], m[1][3], m[2][0], m[2][2], "
		    "m[2][3]);");
		statement_no_indent("");
		statement(
		    "adj[2][0] =  spvDet3x3(m[1][0], m[1][1], m[1][3], m[2][0], m[2][1], m[2][3], m[3][0], m[3][1], "
		    "m[3][3]);");
		statement(
		    "adj[2][1] = -spvDet3x3(m[0][0], m[0][1], m[0][3], m[2][0], m[2][1], m[2][3], m[3][0], m[3][1], "
		    "m[3][3]);");
		statement(
		    "adj[2][2] =  spvDet3x3(m[0][0], m[0][1], m[0][3], m[1][0], m[1][1], m[1][3], m[3][0], m[3][1], "
		    "m[3][3]);");
		statement(
		    "adj[2][3] = -spvDet3x3(m[0][0], m[0][1], m[0][3], m[1][0], m[1][1], m[1][3], m[2][0], m[2][1], "
		    "m[2][3]);");
		statement_no_indent("");
		statement(
		    "adj[3][0] = -spvDet3x3(m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2], m[3][0], m[3][1], "
		    "m[3][2]);");
		statement(
		    "adj[3][1] =  spvDet3x3(m[0][0], m[0][1], m[0][2], m[2][0], m[2][1], m[2][2], m[3][0], m[3][1], "
		    "m[3][2]);");
		statement(
		    "adj[3][2] = -spvDet3x3(m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[3][0], m[3][1], "
		    "m[3][2]);");
		statement(
		    "adj[3][3] =  spvDet3x3(m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], "
		    "m[2][2]);");
		statement_no_indent("");
		statement("// Calculate the determinant as a combination of the cofactors of the first row.");
		statement("float det = (adj[0][0] * m[0][0]) + (adj[0][1] * m[1][0]) + (adj[0][2] * m[2][0]) + (adj[0][3] "
		          "* m[3][0]);");
		statement_no_indent("");
		statement("// Divide the classical adjoint matrix by the determinant.");
		statement("// If determinant is zero, matrix is not invertable, so leave it unchanged.");
		statement("return (det != 0.0f) ? (adj * (1.0f / det)) : m;");
		end_scope();
		statement("");
	}

	if (requires_scalar_reflect)
	{
		// FP16/FP64? No templates in HLSL.
		statement("float spvReflect(float i, float n)");
		begin_scope();
		statement("return i - 2.0 * dot(n, i) * n;");
		end_scope();
		statement("");
	}

	if (requires_scalar_refract)
	{
		// FP16/FP64? No templates in HLSL.
		statement("float spvRefract(float i, float n, float eta)");
		begin_scope();
		statement("float NoI = n * i;");
		statement("float NoI2 = NoI * NoI;");
		statement("float k = 1.0 - eta * eta * (1.0 - NoI2);");
		statement("if (k < 0.0)");
		begin_scope();
		statement("return 0.0;");
		end_scope();
		statement("else");
		begin_scope();
		statement("return eta * i - (eta * NoI + sqrt(k)) * n;");
		end_scope();
		end_scope();
		statement("");
	}

	if (requires_scalar_faceforward)
	{
		// FP16/FP64? No templates in HLSL.
		statement("float spvFaceForward(float n, float i, float nref)");
		begin_scope();
		statement("return i * nref < 0.0 ? n : -n;");
		end_scope();
		statement("");
	}

	for (TypeID type_id : composite_selection_workaround_types)
	{
		// Need out variable since HLSL does not support returning arrays.
		auto &type = get<SPIRType>(type_id);
		auto type_str = type_to_glsl(type);
		auto type_arr_str = type_to_array_glsl(type);
		statement("void spvSelectComposite(out ", type_str, " out_value", type_arr_str, ", bool cond, ",
		          type_str, " true_val", type_arr_str, ", ",
		          type_str, " false_val", type_arr_str, ")");
		begin_scope();
		statement("if (cond)");
		begin_scope();
		statement("out_value = true_val;");
		end_scope();
		statement("else");
		begin_scope();
		statement("out_value = false_val;");
		end_scope();
		end_scope();
		statement("");
	}
}

void CompilerHLSL::emit_texture_size_variants(uint64_t variant_mask, const char *vecsize_qualifier, bool uav,
                                              const char *type_qualifier)
{
	if (variant_mask == 0)
		return;

	static const char *types[QueryTypeCount] = { "float", "int", "uint" };
	static const char *dims[QueryDimCount] = { "Texture1D",   "Texture1DArray",  "Texture2D",   "Texture2DArray",
		                                       "Texture3D",   "Buffer",          "TextureCube", "TextureCubeArray",
		                                       "Texture2DMS", "Texture2DMSArray" };

	static const bool has_lod[QueryDimCount] = { true, true, true, true, true, false, true, true, false, false };

	static const char *ret_types[QueryDimCount] = {
		"uint", "uint2", "uint2", "uint3", "uint3", "uint", "uint2", "uint3", "uint2", "uint3",
	};

	static const uint32_t return_arguments[QueryDimCount] = {
		1, 2, 2, 3, 3, 1, 2, 3, 2, 3,
	};

	for (uint32_t index = 0; index < QueryDimCount; index++)
	{
		for (uint32_t type_index = 0; type_index < QueryTypeCount; type_index++)
		{
			uint32_t bit = 16 * type_index + index;
			uint64_t mask = 1ull << bit;

			if ((variant_mask & mask) == 0)
				continue;

			statement(ret_types[index], " spv", (uav ? "Image" : "Texture"), "Size(", (uav ? "RW" : ""),
			          dims[index], "<", type_qualifier, types[type_index], vecsize_qualifier, "> Tex, ",
			          (uav ? "" : "uint Level, "), "out uint Param)");
			begin_scope();
			statement(ret_types[index], " ret;");
			switch (return_arguments[index])
			{
			case 1:
				if (has_lod[index] && !uav)
					statement("Tex.GetDimensions(Level, ret.x, Param);");
				else
				{
					statement("Tex.GetDimensions(ret.x);");
					statement("Param = 0u;");
				}
				break;
			case 2:
				if (has_lod[index] && !uav)
					statement("Tex.GetDimensions(Level, ret.x, ret.y, Param);");
				else if (!uav)
					statement("Tex.GetDimensions(ret.x, ret.y, Param);");
				else
				{
					statement("Tex.GetDimensions(ret.x, ret.y);");
					statement("Param = 0u;");
				}
				break;
			case 3:
				if (has_lod[index] && !uav)
					statement("Tex.GetDimensions(Level, ret.x, ret.y, ret.z, Param);");
				else if (!uav)
					statement("Tex.GetDimensions(ret.x, ret.y, ret.z, Param);");
				else
				{
					statement("Tex.GetDimensions(ret.x, ret.y, ret.z);");
					statement("Param = 0u;");
				}
				break;
			}

			statement("return ret;");
			end_scope();
			statement("");
		}
	}
}

string CompilerHLSL::layout_for_member(const SPIRType &type, uint32_t index)
{
	auto &flags = get_member_decoration_bitset(type.self, index);

	// HLSL can emit row_major or column_major decoration in any struct.
	// Do not try to merge combined decorations for children like in GLSL.

	// Flip the convention. HLSL is a bit odd in that the memory layout is column major ... but the language API is "row-major".
	// The way to deal with this is to multiply everything in inverse order, and reverse the memory layout.
	if (flags.get(DecorationColMajor))
		return "row_major ";
	else if (flags.get(DecorationRowMajor))
		return "column_major ";

	return "";
}

void CompilerHLSL::emit_struct_member(const SPIRType &type, uint32_t member_type_id, uint32_t index,
                                      const string &qualifier, uint32_t base_offset)
{
	auto &membertype = get<SPIRType>(member_type_id);

	Bitset memberflags;
	auto &memb = ir.meta[type.self].members;
	if (index < memb.size())
		memberflags = memb[index].decoration_flags;

	string packing_offset;
	bool is_push_constant = type.storage == StorageClassPushConstant;

	if ((has_extended_decoration(type.self, SPIRVCrossDecorationExplicitOffset) || is_push_constant) &&
	    has_member_decoration(type.self, index, DecorationOffset))
	{
		uint32_t offset = memb[index].offset - base_offset;
		if (offset & 3)
			SPIRV_CROSS_THROW("Cannot pack on tighter bounds than 4 bytes in HLSL.");

		static const char *packing_swizzle[] = { "", ".y", ".z", ".w" };
		packing_offset = join(" : packoffset(c", offset / 16, packing_swizzle[(offset & 15) >> 2], ")");
	}

	statement(layout_for_member(type, index), qualifier,
	          variable_decl(membertype, to_member_name(type, index)), packing_offset, ";");
}

void CompilerHLSL::emit_rayquery_function(const char *commited, const char *candidate, const uint32_t *ops)
{
	flush_variable_declaration(ops[0]);
	uint32_t is_commited = evaluate_constant_u32(ops[3]);
	emit_op(ops[0], ops[1], join(to_expression(ops[2]), is_commited ? commited : candidate), false);
}

void CompilerHLSL::emit_buffer_block(const SPIRVariable &var)
{
	auto &type = get<SPIRType>(var.basetype);

	bool is_uav = var.storage == StorageClassStorageBuffer || has_decoration(type.self, DecorationBufferBlock);

	if (flattened_buffer_blocks.count(var.self))
	{
		emit_buffer_block_flattened(var);
	}
	else if (is_uav)
	{
		Bitset flags = ir.get_buffer_block_flags(var);
		bool is_readonly = flags.get(DecorationNonWritable) && !is_hlsl_force_storage_buffer_as_uav(var.self);
		bool is_coherent = flags.get(DecorationCoherent) && !is_readonly;
		bool is_interlocked = interlocked_resources.count(var.self) > 0;
		const char *type_name = "ByteAddressBuffer ";
		if (!is_readonly)
			type_name = is_interlocked ? "RasterizerOrderedByteAddressBuffer " : "RWByteAddressBuffer ";
		add_resource_name(var.self);
		statement(is_coherent ? "globallycoherent " : "", type_name, to_name(var.self), type_to_array_glsl(type),
		          to_resource_binding(var), ";");
	}
	else
	{
		if (type.array.empty())
		{
			// Flatten the top-level struct so we can use packoffset,
			// this restriction is similar to GLSL where layout(offset) is not possible on sub-structs.
			flattened_structs[var.self] = false;

			// Prefer the block name if possible.
			auto buffer_name = to_name(type.self, false);
			if (ir.meta[type.self].decoration.alias.empty() ||
			    resource_names.find(buffer_name) != end(resource_names) ||
			    block_names.find(buffer_name) != end(block_names))
			{
				buffer_name = get_block_fallback_name(var.self);
			}

			add_variable(block_names, resource_names, buffer_name);

			// If for some reason buffer_name is an illegal name, make a final fallback to a workaround name.
			// This cannot conflict with anything else, so we're safe now.
			if (buffer_name.empty())
				buffer_name = join("_", get<SPIRType>(var.basetype).self, "_", var.self);

			uint32_t failed_index = 0;
			if (buffer_is_packing_standard(type, BufferPackingHLSLCbufferPackOffset, &failed_index))
				set_extended_decoration(type.self, SPIRVCrossDecorationExplicitOffset);
			else
			{
				SPIRV_CROSS_THROW(join("cbuffer ID ", var.self, " (name: ", buffer_name, "), member index ",
				                       failed_index, " (name: ", to_member_name(type, failed_index),
				                       ") cannot be expressed with either HLSL packing layout or packoffset."));
			}

			block_names.insert(buffer_name);

			// Save for post-reflection later.
			declared_block_names[var.self] = buffer_name;

			type.member_name_cache.clear();
			// var.self can be used as a backup name for the block name,
			// so we need to make sure we don't disturb the name here on a recompile.
			// It will need to be reset if we have to recompile.
			preserve_alias_on_reset(var.self);
			add_resource_name(var.self);
			statement("cbuffer ", buffer_name, to_resource_binding(var));
			begin_scope();

			uint32_t i = 0;
			for (auto &member : type.member_types)
			{
				add_member_name(type, i);
				auto backup_name = get_member_name(type.self, i);
				auto member_name = to_member_name(type, i);
				member_name = join(to_name(var.self), "_", member_name);
				ParsedIR::sanitize_underscores(member_name);
				set_member_name(type.self, i, member_name);
				emit_struct_member(type, member, i, "");
				set_member_name(type.self, i, backup_name);
				i++;
			}

			end_scope_decl();
			statement("");
		}
		else
		{
			if (hlsl_options.shader_model < 51)
				SPIRV_CROSS_THROW(
				    "Need ConstantBuffer<T> to use arrays of UBOs, but this is only supported in SM 5.1.");

			add_resource_name(type.self);
			add_resource_name(var.self);

			// ConstantBuffer<T> does not support packoffset, so it is unuseable unless everything aligns as we expect.
			uint32_t failed_index = 0;
			if (!buffer_is_packing_standard(type, BufferPackingHLSLCbuffer, &failed_index))
			{
				SPIRV_CROSS_THROW(join("HLSL ConstantBuffer<T> ID ", var.self, " (name: ", to_name(type.self),
				                       "), member index ", failed_index, " (name: ", to_member_name(type, failed_index),
				                       ") cannot be expressed with normal HLSL packing rules."));
			}

			emit_struct(get<SPIRType>(type.self));
			statement("ConstantBuffer<", to_name(type.self), "> ", to_name(var.self), type_to_array_glsl(type),
			          to_resource_binding(var), ";");
		}
	}
}

void CompilerHLSL::emit_push_constant_block(const SPIRVariable &var)
{
	if (flattened_buffer_blocks.count(var.self))
	{
		emit_buffer_block_flattened(var);
	}
	else if (root_constants_layout.empty())
	{
		emit_buffer_block(var);
	}
	else
	{
		for (const auto &layout : root_constants_layout)
		{
			auto &type = get<SPIRType>(var.basetype);

			uint32_t failed_index = 0;
			if (buffer_is_packing_standard(type, BufferPackingHLSLCbufferPackOffset, &failed_index, layout.start,
			                               layout.end))
				set_extended_decoration(type.self, SPIRVCrossDecorationExplicitOffset);
			else
			{
				SPIRV_CROSS_THROW(join("Root constant cbuffer ID ", var.self, " (name: ", to_name(type.self), ")",
				                       ", member index ", failed_index, " (name: ", to_member_name(type, failed_index),
				                       ") cannot be expressed with either HLSL packing layout or packoffset."));
			}

			flattened_structs[var.self] = false;
			type.member_name_cache.clear();
			add_resource_name(var.self);
			auto &memb = ir.meta[type.self].members;

			statement("cbuffer SPIRV_CROSS_RootConstant_", to_name(var.self),
			          to_resource_register(HLSL_BINDING_AUTO_PUSH_CONSTANT_BIT, 'b', layout.binding, layout.space));
			begin_scope();

			// Index of the next field in the generated root constant constant buffer
			auto constant_index = 0u;

			// Iterate over all member of the push constant and check which of the fields
			// fit into the given root constant layout.
			for (auto i = 0u; i < memb.size(); i++)
			{
				const auto offset = memb[i].offset;
				if (layout.start <= offset && offset < layout.end)
				{
					const auto &member = type.member_types[i];

					add_member_name(type, constant_index);
					auto backup_name = get_member_name(type.self, i);
					auto member_name = to_member_name(type, i);
					member_name = join(to_name(var.self), "_", member_name);
					ParsedIR::sanitize_underscores(member_name);
					set_member_name(type.self, constant_index, member_name);
					emit_struct_member(type, member, i, "", layout.start);
					set_member_name(type.self, constant_index, backup_name);

					constant_index++;
				}
			}

			end_scope_decl();
		}
	}
}

string CompilerHLSL::to_sampler_expression(uint32_t id)
{
	auto expr = join("_", to_non_uniform_aware_expression(id));
	auto index = expr.find_first_of('[');
	if (index == string::npos)
	{
		return expr + "_sampler";
	}
	else
	{
		// We have an expression like _ident[array], so we cannot tack on _sampler, insert it inside the string instead.
		return expr.insert(index, "_sampler");
	}
}

void CompilerHLSL::emit_sampled_image_op(uint32_t result_type, uint32_t result_id, uint32_t image_id, uint32_t samp_id)
{
	if (hlsl_options.shader_model >= 40 && combined_image_samplers.empty())
	{
		set<SPIRCombinedImageSampler>(result_id, result_type, image_id, samp_id);
	}
	else
	{
		// Make sure to suppress usage tracking. It is illegal to create temporaries of opaque types.
		emit_op(result_type, result_id, to_combined_image_sampler(image_id, samp_id), true, true);
	}
}

string CompilerHLSL::to_func_call_arg(const SPIRFunction::Parameter &arg, uint32_t id)
{
	string arg_str = CompilerGLSL::to_func_call_arg(arg, id);

	if (hlsl_options.shader_model <= 30)
		return arg_str;

	// Manufacture automatic sampler arg if the arg is a SampledImage texture and we're in modern HLSL.
	auto &type = expression_type(id);

	// We don't have to consider combined image samplers here via OpSampledImage because
	// those variables cannot be passed as arguments to functions.
	// Only global SampledImage variables may be used as arguments.
	if (type.basetype == SPIRType::SampledImage && type.image.dim != DimBuffer)
		arg_str += ", " + to_sampler_expression(id);

	return arg_str;
}

string CompilerHLSL::get_inner_entry_point_name() const
{
	auto &execution = get_entry_point();

	if (hlsl_options.use_entry_point_name)
	{
		auto name = join(execution.name, "_inner");
		ParsedIR::sanitize_underscores(name);
		return name;
	}

	if (execution.model == ExecutionModelVertex)
		return "vert_main";
	else if (execution.model == ExecutionModelFragment)
		return "frag_main";
	else if (execution.model == ExecutionModelGLCompute)
		return "comp_main";
	else
		SPIRV_CROSS_THROW("Unsupported execution model.");
}

void CompilerHLSL::emit_function_prototype(SPIRFunction &func, const Bitset &return_flags)
{
	if (func.self != ir.default_entry_point)
		add_function_overload(func);

	// Avoid shadow declarations.
	local_variable_names = resource_names;

	string decl;

	auto &type = get<SPIRType>(func.return_type);
	if (type.array.empty())
	{
		decl += flags_to_qualifiers_glsl(type, return_flags);
		decl += type_to_glsl(type);
		decl += " ";
	}
	else
	{
		// We cannot return arrays in HLSL, so "return" through an out variable.
		decl = "void ";
	}

	if (func.self == ir.default_entry_point)
	{
		decl += get_inner_entry_point_name();
		processing_entry_point = true;
	}
	else
		decl += to_name(func.self);

	decl += "(";
	SmallVector<string> arglist;

	if (!type.array.empty())
	{
		// Fake array returns by writing to an out array instead.
		string out_argument;
		out_argument += "out ";
		out_argument += type_to_glsl(type);
		out_argument += " ";
		out_argument += "spvReturnValue";
		out_argument += type_to_array_glsl(type);
		arglist.push_back(std::move(out_argument));
	}

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

		// Flatten a combined sampler to two separate arguments in modern HLSL.
		auto &arg_type = get<SPIRType>(arg.type);
		if (hlsl_options.shader_model > 30 && arg_type.basetype == SPIRType::SampledImage &&
		    arg_type.image.dim != DimBuffer)
		{
			// Manufacture automatic sampler arg for SampledImage texture
			arglist.push_back(join(is_depth_image(arg_type, arg.id) ? "SamplerComparisonState " : "SamplerState ",
			                       to_sampler_expression(arg.id), type_to_array_glsl(arg_type)));
		}

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

void CompilerHLSL::emit_hlsl_entry_point()
{
	SmallVector<string> arguments;

	if (require_input)
		arguments.push_back("SPIRV_Cross_Input stage_input");

	auto &execution = get_entry_point();

	switch (execution.model)
	{
	case ExecutionModelGLCompute:
	{
		SpecializationConstant wg_x, wg_y, wg_z;
		get_work_group_size_specialization_constants(wg_x, wg_y, wg_z);

		uint32_t x = execution.workgroup_size.x;
		uint32_t y = execution.workgroup_size.y;
		uint32_t z = execution.workgroup_size.z;

		if (!execution.workgroup_size.constant && execution.flags.get(ExecutionModeLocalSizeId))
		{
			if (execution.workgroup_size.id_x)
				x = get<SPIRConstant>(execution.workgroup_size.id_x).scalar();
			if (execution.workgroup_size.id_y)
				y = get<SPIRConstant>(execution.workgroup_size.id_y).scalar();
			if (execution.workgroup_size.id_z)
				z = get<SPIRConstant>(execution.workgroup_size.id_z).scalar();
		}

		auto x_expr = wg_x.id ? get<SPIRConstant>(wg_x.id).specialization_constant_macro_name : to_string(x);
		auto y_expr = wg_y.id ? get<SPIRConstant>(wg_y.id).specialization_constant_macro_name : to_string(y);
		auto z_expr = wg_z.id ? get<SPIRConstant>(wg_z.id).specialization_constant_macro_name : to_string(z);

		statement("[numthreads(", x_expr, ", ", y_expr, ", ", z_expr, ")]");
		break;
	}
	case ExecutionModelFragment:
		if (execution.flags.get(ExecutionModeEarlyFragmentTests))
			statement("[earlydepthstencil]");
		break;
	default:
		break;
	}

	const char *entry_point_name;
	if (hlsl_options.use_entry_point_name)
		entry_point_name = get_entry_point().name.c_str();
	else
		entry_point_name = "main";

	statement(require_output ? "SPIRV_Cross_Output " : "void ", entry_point_name, "(", merge(arguments), ")");
	begin_scope();
	bool legacy = hlsl_options.shader_model <= 30;

	// Copy builtins from entry point arguments to globals.
	active_input_builtins.for_each_bit([&](uint32_t i) {
		auto builtin = builtin_to_glsl(static_cast<BuiltIn>(i), StorageClassInput);
		switch (static_cast<BuiltIn>(i))
		{
		case BuiltInFragCoord:
			// VPOS in D3D9 is sampled at integer locations, apply half-pixel offset to be consistent.
			// TODO: Do we need an option here? Any reason why a D3D9 shader would be used
			// on a D3D10+ system with a different rasterization config?
			if (legacy)
				statement(builtin, " = stage_input.", builtin, " + float4(0.5f, 0.5f, 0.0f, 0.0f);");
			else
			{
				statement(builtin, " = stage_input.", builtin, ";");
				// ZW are undefined in D3D9, only do this fixup here.
				statement(builtin, ".w = 1.0 / ", builtin, ".w;");
			}
			break;

		case BuiltInVertexId:
		case BuiltInVertexIndex:
		case BuiltInInstanceIndex:
			// D3D semantics are uint, but shader wants int.
			if (hlsl_options.support_nonzero_base_vertex_base_instance)
			{
				if (static_cast<BuiltIn>(i) == BuiltInInstanceIndex)
					statement(builtin, " = int(stage_input.", builtin, ") + SPIRV_Cross_BaseInstance;");
				else
					statement(builtin, " = int(stage_input.", builtin, ") + SPIRV_Cross_BaseVertex;");
			}
			else
				statement(builtin, " = int(stage_input.", builtin, ");");
			break;

		case BuiltInInstanceId:
			// D3D semantics are uint, but shader wants int.
			statement(builtin, " = int(stage_input.", builtin, ");");
			break;

		case BuiltInNumWorkgroups:
		case BuiltInPointCoord:
		case BuiltInSubgroupSize:
		case BuiltInSubgroupLocalInvocationId:
		case BuiltInHelperInvocation:
			break;

		case BuiltInSubgroupEqMask:
			// Emulate these ...
			// No 64-bit in HLSL, so have to do it in 32-bit and unroll.
			statement("gl_SubgroupEqMask = 1u << (WaveGetLaneIndex() - uint4(0, 32, 64, 96));");
			statement("if (WaveGetLaneIndex() >= 32) gl_SubgroupEqMask.x = 0;");
			statement("if (WaveGetLaneIndex() >= 64 || WaveGetLaneIndex() < 32) gl_SubgroupEqMask.y = 0;");
			statement("if (WaveGetLaneIndex() >= 96 || WaveGetLaneIndex() < 64) gl_SubgroupEqMask.z = 0;");
			statement("if (WaveGetLaneIndex() < 96) gl_SubgroupEqMask.w = 0;");
			break;

		case BuiltInSubgroupGeMask:
			// Emulate these ...
			// No 64-bit in HLSL, so have to do it in 32-bit and unroll.
			statement("gl_SubgroupGeMask = ~((1u << (WaveGetLaneIndex() - uint4(0, 32, 64, 96))) - 1u);");
			statement("if (WaveGetLaneIndex() >= 32) gl_SubgroupGeMask.x = 0u;");
			statement("if (WaveGetLaneIndex() >= 64) gl_SubgroupGeMask.y = 0u;");
			statement("if (WaveGetLaneIndex() >= 96) gl_SubgroupGeMask.z = 0u;");
			statement("if (WaveGetLaneIndex() < 32) gl_SubgroupGeMask.y = ~0u;");
			statement("if (WaveGetLaneIndex() < 64) gl_SubgroupGeMask.z = ~0u;");
			statement("if (WaveGetLaneIndex() < 96) gl_SubgroupGeMask.w = ~0u;");
			break;

		case BuiltInSubgroupGtMask:
			// Emulate these ...
			// No 64-bit in HLSL, so have to do it in 32-bit and unroll.
			statement("uint gt_lane_index = WaveGetLaneIndex() + 1;");
			statement("gl_SubgroupGtMask = ~((1u << (gt_lane_index - uint4(0, 32, 64, 96))) - 1u);");
			statement("if (gt_lane_index >= 32) gl_SubgroupGtMask.x = 0u;");
			statement("if (gt_lane_index >= 64) gl_SubgroupGtMask.y = 0u;");
			statement("if (gt_lane_index >= 96) gl_SubgroupGtMask.z = 0u;");
			statement("if (gt_lane_index >= 128) gl_SubgroupGtMask.w = 0u;");
			statement("if (gt_lane_index < 32) gl_SubgroupGtMask.y = ~0u;");
			statement("if (gt_lane_index < 64) gl_SubgroupGtMask.z = ~0u;");
			statement("if (gt_lane_index < 96) gl_SubgroupGtMask.w = ~0u;");
			break;

		case BuiltInSubgroupLeMask:
			// Emulate these ...
			// No 64-bit in HLSL, so have to do it in 32-bit and unroll.
			statement("uint le_lane_index = WaveGetLaneIndex() + 1;");
			statement("gl_SubgroupLeMask = (1u << (le_lane_index - uint4(0, 32, 64, 96))) - 1u;");
			statement("if (le_lane_index >= 32) gl_SubgroupLeMask.x = ~0u;");
			statement("if (le_lane_index >= 64) gl_SubgroupLeMask.y = ~0u;");
			statement("if (le_lane_index >= 96) gl_SubgroupLeMask.z = ~0u;");
			statement("if (le_lane_index >= 128) gl_SubgroupLeMask.w = ~0u;");
			statement("if (le_lane_index < 32) gl_SubgroupLeMask.y = 0u;");
			statement("if (le_lane_index < 64) gl_SubgroupLeMask.z = 0u;");
			statement("if (le_lane_index < 96) gl_SubgroupLeMask.w = 0u;");
			break;

		case BuiltInSubgroupLtMask:
			// Emulate these ...
			// No 64-bit in HLSL, so have to do it in 32-bit and unroll.
			statement("gl_SubgroupLtMask = (1u << (WaveGetLaneIndex() - uint4(0, 32, 64, 96))) - 1u;");
			statement("if (WaveGetLaneIndex() >= 32) gl_SubgroupLtMask.x = ~0u;");
			statement("if (WaveGetLaneIndex() >= 64) gl_SubgroupLtMask.y = ~0u;");
			statement("if (WaveGetLaneIndex() >= 96) gl_SubgroupLtMask.z = ~0u;");
			statement("if (WaveGetLaneIndex() < 32) gl_SubgroupLtMask.y = 0u;");
			statement("if (WaveGetLaneIndex() < 64) gl_SubgroupLtMask.z = 0u;");
			statement("if (WaveGetLaneIndex() < 96) gl_SubgroupLtMask.w = 0u;");
			break;

		case BuiltInClipDistance:
			for (uint32_t clip = 0; clip < clip_distance_count; clip++)
				statement("gl_ClipDistance[", clip, "] = stage_input.gl_ClipDistance", clip / 4, ".", "xyzw"[clip & 3],
				          ";");
			break;

		case BuiltInCullDistance:
			for (uint32_t cull = 0; cull < cull_distance_count; cull++)
				statement("gl_CullDistance[", cull, "] = stage_input.gl_CullDistance", cull / 4, ".", "xyzw"[cull & 3],
				          ";");
			break;

		default:
			statement(builtin, " = stage_input.", builtin, ";");
			break;
		}
	});

	// Copy from stage input struct to globals.
	ir.for_each_typed_id<SPIRVariable>([&](uint32_t, SPIRVariable &var) {
		auto &type = this->get<SPIRType>(var.basetype);
		bool block = has_decoration(type.self, DecorationBlock);

		if (var.storage != StorageClassInput)
			return;

		bool need_matrix_unroll = var.storage == StorageClassInput && execution.model == ExecutionModelVertex;

		if (!var.remapped_variable && type.pointer && !is_builtin_variable(var) &&
		    interface_variable_exists_in_entry_point(var.self))
		{
			if (block)
			{
				auto type_name = to_name(type.self);
				auto var_name = to_name(var.self);
				for (uint32_t mbr_idx = 0; mbr_idx < uint32_t(type.member_types.size()); mbr_idx++)
				{
					auto mbr_name = to_member_name(type, mbr_idx);
					auto flat_name = join(type_name, "_", mbr_name);
					statement(var_name, ".", mbr_name, " = stage_input.", flat_name, ";");
				}
			}
			else
			{
				auto name = to_name(var.self);
				auto &mtype = this->get<SPIRType>(var.basetype);
				if (need_matrix_unroll && mtype.columns > 1)
				{
					// Unroll matrices.
					for (uint32_t col = 0; col < mtype.columns; col++)
						statement(name, "[", col, "] = stage_input.", name, "_", col, ";");
				}
				else
				{
					statement(name, " = stage_input.", name, ";");
				}
			}
		}
	});

	// Run the shader.
	if (execution.model == ExecutionModelVertex ||
	    execution.model == ExecutionModelFragment ||
	    execution.model == ExecutionModelGLCompute)
	{
		statement(get_inner_entry_point_name(), "();");
	}
	else
		SPIRV_CROSS_THROW("Unsupported shader stage.");

	// Copy stage outputs.
	if (require_output)
	{
		statement("SPIRV_Cross_Output stage_output;");

		// Copy builtins from globals to return struct.
		active_output_builtins.for_each_bit([&](uint32_t i) {
			// PointSize doesn't exist in HLSL.
			if (i == BuiltInPointSize)
				return;

			switch (static_cast<BuiltIn>(i))
			{
			case BuiltInClipDistance:
				for (uint32_t clip = 0; clip < clip_distance_count; clip++)
					statement("stage_output.gl_ClipDistance", clip / 4, ".", "xyzw"[clip & 3], " = gl_ClipDistance[",
					          clip, "];");
				break;

			case BuiltInCullDistance:
				for (uint32_t cull = 0; cull < cull_distance_count; cull++)
					statement("stage_output.gl_CullDistance", cull / 4, ".", "xyzw"[cull & 3], " = gl_CullDistance[",
					          cull, "];");
				break;

			default:
			{
				auto builtin_expr = builtin_to_glsl(static_cast<BuiltIn>(i), StorageClassOutput);
				statement("stage_output.", builtin_expr, " = ", builtin_expr, ";");
				break;
			}
			}
		});

		ir.for_each_typed_id<SPIRVariable>([&](uint32_t, SPIRVariable &var) {
			auto &type = this->get<SPIRType>(var.basetype);
			bool block = has_decoration(type.self, DecorationBlock);

			if (var.storage != StorageClassOutput)
				return;

			if (!var.remapped_variable && type.pointer &&
			    !is_builtin_variable(var) &&
			    interface_variable_exists_in_entry_point(var.self))
			{
				if (block)
				{
					// I/O blocks need to flatten output.
					auto type_name = to_name(type.self);
					auto var_name = to_name(var.self);
					for (uint32_t mbr_idx = 0; mbr_idx < uint32_t(type.member_types.size()); mbr_idx++)
					{
						auto mbr_name = to_member_name(type, mbr_idx);
						auto flat_name = join(type_name, "_", mbr_name);
						statement("stage_output.", flat_name, " = ", var_name, ".", mbr_name, ";");
					}
				}
				else
				{
					auto name = to_name(var.self);

					if (legacy && execution.model == ExecutionModelFragment)
					{
						string output_filler;
						for (uint32_t size = type.vecsize; size < 4; ++size)
							output_filler += ", 0.0";

						statement("stage_output.", name, " = float4(", name, output_filler, ");");
					}
					else
					{
						statement("stage_output.", name, " = ", name, ";");
					}
				}
			}
		});

		statement("return stage_output;");
	}

	end_scope();
}

void CompilerHLSL::emit_fixup()
{
	if (is_vertex_like_shader() && active_output_builtins.get(BuiltInPosition))
	{
		// Do various mangling on the gl_Position.
		if (hlsl_options.shader_model <= 30)
		{
			statement("gl_Position.x = gl_Position.x - gl_HalfPixel.x * "
			          "gl_Position.w;");
			statement("gl_Position.y = gl_Position.y + gl_HalfPixel.y * "
			          "gl_Position.w;");
		}

		if (options.vertex.flip_vert_y)
			statement("gl_Position.y = -gl_Position.y;");
		if (options.vertex.fixup_clipspace)
			statement("gl_Position.z = (gl_Position.z + gl_Position.w) * 0.5;");
	}
}

void CompilerHLSL::emit_texture_op(const Instruction &i, bool sparse)
{
	if (sparse)
		SPIRV_CROSS_THROW("Sparse feedback not yet supported in HLSL.");

	auto *ops = stream(i);
	auto op = static_cast<Op>(i.op);
	uint32_t length = i.length;

	SmallVector<uint32_t> inherited_expressions;

	uint32_t result_type = ops[0];
	uint32_t id = ops[1];
	VariableID img = ops[2];
	uint32_t coord = ops[3];
	uint32_t dref = 0;
	uint32_t comp = 0;
	bool gather = false;
	bool proj = false;
	const uint32_t *opt = nullptr;
	auto *combined_image = maybe_get<SPIRCombinedImageSampler>(img);

	if (combined_image && has_decoration(img, DecorationNonUniform))
	{
		set_decoration(combined_image->image, DecorationNonUniform);
		set_decoration(combined_image->sampler, DecorationNonUniform);
	}

	auto img_expr = to_non_uniform_aware_expression(combined_image ? combined_image->image : img);

	inherited_expressions.push_back(coord);

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
		proj = true;
		opt = &ops[5];
		length -= 5;
		break;

	case OpImageDrefGather:
		dref = ops[4];
		opt = &ops[5];
		gather = true;
		length -= 5;
		break;

	case OpImageGather:
		comp = ops[4];
		opt = &ops[5];
		gather = true;
		length -= 5;
		break;

	case OpImageSampleProjImplicitLod:
	case OpImageSampleProjExplicitLod:
		opt = &ops[4];
		length -= 4;
		proj = true;
		break;

	case OpImageQueryLod:
		opt = &ops[4];
		length -= 4;
		break;

	default:
		opt = &ops[4];
		length -= 4;
		break;
	}

	auto &imgtype = expression_type(img);
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
		flags = opt[0];
		opt++;
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
	string texop;

	if (minlod != 0)
		SPIRV_CROSS_THROW("MinLod texture operand not supported in HLSL.");

	if (op == OpImageFetch)
	{
		if (hlsl_options.shader_model < 40)
		{
			SPIRV_CROSS_THROW("texelFetch is not supported in HLSL shader model 2/3.");
		}
		texop += img_expr;
		texop += ".Load";
	}
	else if (op == OpImageQueryLod)
	{
		texop += img_expr;
		texop += ".CalculateLevelOfDetail";
	}
	else
	{
		auto &imgformat = get<SPIRType>(imgtype.image.type);
		if (imgformat.basetype != SPIRType::Float)
		{
			SPIRV_CROSS_THROW("Sampling non-float textures is not supported in HLSL.");
		}

		if (hlsl_options.shader_model >= 40)
		{
			texop += img_expr;

			if (is_depth_image(imgtype, img))
			{
				if (gather)
				{
					SPIRV_CROSS_THROW("GatherCmp does not exist in HLSL.");
				}
				else if (lod || grad_x || grad_y)
				{
					// Assume we want a fixed level, and the only thing we can get in HLSL is SampleCmpLevelZero.
					texop += ".SampleCmpLevelZero";
				}
				else
					texop += ".SampleCmp";
			}
			else if (gather)
			{
				uint32_t comp_num = evaluate_constant_u32(comp);
				if (hlsl_options.shader_model >= 50)
				{
					switch (comp_num)
					{
					case 0:
						texop += ".GatherRed";
						break;
					case 1:
						texop += ".GatherGreen";
						break;
					case 2:
						texop += ".GatherBlue";
						break;
					case 3:
						texop += ".GatherAlpha";
						break;
					default:
						SPIRV_CROSS_THROW("Invalid component.");
					}
				}
				else
				{
					if (comp_num == 0)
						texop += ".Gather";
					else
						SPIRV_CROSS_THROW("HLSL shader model 4 can only gather from the red component.");
				}
			}
			else if (bias)
				texop += ".SampleBias";
			else if (grad_x || grad_y)
				texop += ".SampleGrad";
			else if (lod)
				texop += ".SampleLevel";
			else
				texop += ".Sample";
		}
		else
		{
			switch (imgtype.image.dim)
			{
			case Dim1D:
				texop += "tex1D";
				break;
			case Dim2D:
				texop += "tex2D";
				break;
			case Dim3D:
				texop += "tex3D";
				break;
			case DimCube:
				texop += "texCUBE";
				break;
			case DimRect:
			case DimBuffer:
			case DimSubpassData:
				SPIRV_CROSS_THROW("Buffer texture support is not yet implemented for HLSL"); // TODO
			default:
				SPIRV_CROSS_THROW("Invalid dimension.");
			}

			if (gather)
				SPIRV_CROSS_THROW("textureGather is not supported in HLSL shader model 2/3.");
			if (offset || coffset)
				SPIRV_CROSS_THROW("textureOffset is not supported in HLSL shader model 2/3.");

			if (grad_x || grad_y)
				texop += "grad";
			else if (lod)
				texop += "lod";
			else if (bias)
				texop += "bias";
			else if (proj || dref)
				texop += "proj";
		}
	}

	expr += texop;
	expr += "(";
	if (hlsl_options.shader_model < 40)
	{
		if (combined_image)
			SPIRV_CROSS_THROW("Separate images/samplers are not supported in HLSL shader model 2/3.");
		expr += to_expression(img);
	}
	else if (op != OpImageFetch)
	{
		string sampler_expr;
		if (combined_image)
			sampler_expr = to_non_uniform_aware_expression(combined_image->sampler);
		else
			sampler_expr = to_sampler_expression(img);
		expr += sampler_expr;
	}

	auto swizzle = [](uint32_t comps, uint32_t in_comps) -> const char * {
		if (comps == in_comps)
			return "";

		switch (comps)
		{
		case 1:
			return ".x";
		case 2:
			return ".xy";
		case 3:
			return ".xyz";
		default:
			return "";
		}
	};

	bool forward = should_forward(coord);

	// The IR can give us more components than we need, so chop them off as needed.
	string coord_expr;
	auto &coord_type = expression_type(coord);
	if (coord_components != coord_type.vecsize)
		coord_expr = to_enclosed_expression(coord) + swizzle(coord_components, expression_type(coord).vecsize);
	else
		coord_expr = to_expression(coord);

	if (proj && hlsl_options.shader_model >= 40) // Legacy HLSL has "proj" operations which do this for us.
		coord_expr = coord_expr + " / " + to_extract_component_expression(coord, coord_components);

	if (hlsl_options.shader_model < 40)
	{
		if (dref)
		{
			if (imgtype.image.dim != spv::Dim1D && imgtype.image.dim != spv::Dim2D)
			{
				SPIRV_CROSS_THROW(
				    "Depth comparison is only supported for 1D and 2D textures in HLSL shader model 2/3.");
			}

			if (grad_x || grad_y)
				SPIRV_CROSS_THROW("Depth comparison is not supported for grad sampling in HLSL shader model 2/3.");

			for (uint32_t size = coord_components; size < 2; ++size)
				coord_expr += ", 0.0";

			forward = forward && should_forward(dref);
			coord_expr += ", " + to_expression(dref);
		}
		else if (lod || bias || proj)
		{
			for (uint32_t size = coord_components; size < 3; ++size)
				coord_expr += ", 0.0";
		}

		if (lod)
		{
			coord_expr = "float4(" + coord_expr + ", " + to_expression(lod) + ")";
		}
		else if (bias)
		{
			coord_expr = "float4(" + coord_expr + ", " + to_expression(bias) + ")";
		}
		else if (proj)
		{
			coord_expr = "float4(" + coord_expr + ", " + to_extract_component_expression(coord, coord_components) + ")";
		}
		else if (dref)
		{
			// A "normal" sample gets fed into tex2Dproj as well, because the
			// regular tex2D accepts only two coordinates.
			coord_expr = "float4(" + coord_expr + ", 1.0)";
		}

		if (!!lod + !!bias + !!proj > 1)
			SPIRV_CROSS_THROW("Legacy HLSL can only use one of lod/bias/proj modifiers.");
	}

	if (op == OpImageFetch)
	{
		if (imgtype.image.dim != DimBuffer && !imgtype.image.ms)
			coord_expr =
			    join("int", coord_components + 1, "(", coord_expr, ", ", lod ? to_expression(lod) : string("0"), ")");
	}
	else
		expr += ", ";
	expr += coord_expr;

	if (dref && hlsl_options.shader_model >= 40)
	{
		forward = forward && should_forward(dref);
		expr += ", ";

		if (proj)
			expr += to_enclosed_expression(dref) + " / " + to_extract_component_expression(coord, coord_components);
		else
			expr += to_expression(dref);
	}

	if (!dref && (grad_x || grad_y))
	{
		forward = forward && should_forward(grad_x);
		forward = forward && should_forward(grad_y);
		expr += ", ";
		expr += to_expression(grad_x);
		expr += ", ";
		expr += to_expression(grad_y);
	}

	if (!dref && lod && hlsl_options.shader_model >= 40 && op != OpImageFetch)
	{
		forward = forward && should_forward(lod);
		expr += ", ";
		expr += to_expression(lod);
	}

	if (!dref && bias && hlsl_options.shader_model >= 40)
	{
		forward = forward && should_forward(bias);
		expr += ", ";
		expr += to_expression(bias);
	}

	if (coffset)
	{
		forward = forward && should_forward(coffset);
		expr += ", ";
		expr += to_expression(coffset);
	}
	else if (offset)
	{
		forward = forward && should_forward(offset);
		expr += ", ";
		expr += to_expression(offset);
	}

	if (sample)
	{
		expr += ", ";
		expr += to_expression(sample);
	}

	expr += ")";

	if (dref && hlsl_options.shader_model < 40)
		expr += ".x";

	if (op == OpImageQueryLod)
	{
		// This is rather awkward.
		// textureQueryLod returns two values, the "accessed level",
		// as well as the actual LOD lambda.
		// As far as I can tell, there is no way to get the .x component
		// according to GLSL spec, and it depends on the sampler itself.
		// Just assume X == Y, so we will need to splat the result to a float2.
		statement("float _", id, "_tmp = ", expr, ";");
		statement("float2 _", id, " = _", id, "_tmp.xx;");
		set<SPIRExpression>(id, join("_", id), result_type, true);
	}
	else
	{
		emit_op(result_type, id, expr, forward, false);
	}

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

string CompilerHLSL::to_resource_binding(const SPIRVariable &var)
{
	const auto &type = get<SPIRType>(var.basetype);

	// We can remap push constant blocks, even if they don't have any binding decoration.
	if (type.storage != StorageClassPushConstant && !has_decoration(var.self, DecorationBinding))
		return "";

	char space = '\0';

	HLSLBindingFlagBits resource_flags = HLSL_BINDING_AUTO_NONE_BIT;

	switch (type.basetype)
	{
	case SPIRType::SampledImage:
		space = 't'; // SRV
		resource_flags = HLSL_BINDING_AUTO_SRV_BIT;
		break;

	case SPIRType::Image:
		if (type.image.sampled == 2 && type.image.dim != DimSubpassData)
		{
			if (has_decoration(var.self, DecorationNonWritable) && hlsl_options.nonwritable_uav_texture_as_srv)
			{
				space = 't'; // SRV
				resource_flags = HLSL_BINDING_AUTO_SRV_BIT;
			}
			else
			{
				space = 'u'; // UAV
				resource_flags = HLSL_BINDING_AUTO_UAV_BIT;
			}
		}
		else
		{
			space = 't'; // SRV
			resource_flags = HLSL_BINDING_AUTO_SRV_BIT;
		}
		break;

	case SPIRType::Sampler:
		space = 's';
		resource_flags = HLSL_BINDING_AUTO_SAMPLER_BIT;
		break;

	case SPIRType::AccelerationStructure:
		space = 't'; // SRV
		resource_flags = HLSL_BINDING_AUTO_SRV_BIT;
		break;

	case SPIRType::Struct:
	{
		auto storage = type.storage;
		if (storage == StorageClassUniform)
		{
			if (has_decoration(type.self, DecorationBufferBlock))
			{
				Bitset flags = ir.get_buffer_block_flags(var);
				bool is_readonly = flags.get(DecorationNonWritable) && !is_hlsl_force_storage_buffer_as_uav(var.self);
				space = is_readonly ? 't' : 'u'; // UAV
				resource_flags = is_readonly ? HLSL_BINDING_AUTO_SRV_BIT : HLSL_BINDING_AUTO_UAV_BIT;
			}
			else if (has_decoration(type.self, DecorationBlock))
			{
				space = 'b'; // Constant buffers
				resource_flags = HLSL_BINDING_AUTO_CBV_BIT;
			}
		}
		else if (storage == StorageClassPushConstant)
		{
			space = 'b'; // Constant buffers
			resource_flags = HLSL_BINDING_AUTO_PUSH_CONSTANT_BIT;
		}
		else if (storage == StorageClassStorageBuffer)
		{
			// UAV or SRV depending on readonly flag.
			Bitset flags = ir.get_buffer_block_flags(var);
			bool is_readonly = flags.get(DecorationNonWritable) && !is_hlsl_force_storage_buffer_as_uav(var.self);
			space = is_readonly ? 't' : 'u';
			resource_flags = is_readonly ? HLSL_BINDING_AUTO_SRV_BIT : HLSL_BINDING_AUTO_UAV_BIT;
		}

		break;
	}
	default:
		break;
	}

	if (!space)
		return "";

	uint32_t desc_set =
	    resource_flags == HLSL_BINDING_AUTO_PUSH_CONSTANT_BIT ? ResourceBindingPushConstantDescriptorSet : 0u;
	uint32_t binding = resource_flags == HLSL_BINDING_AUTO_PUSH_CONSTANT_BIT ? ResourceBindingPushConstantBinding : 0u;

	if (has_decoration(var.self, DecorationBinding))
		binding = get_decoration(var.self, DecorationBinding);
	if (has_decoration(var.self, DecorationDescriptorSet))
		desc_set = get_decoration(var.self, DecorationDescriptorSet);

	return to_resource_register(resource_flags, space, binding, desc_set);
}

string CompilerHLSL::to_resource_binding_sampler(const SPIRVariable &var)
{
	// For combined image samplers.
	if (!has_decoration(var.self, DecorationBinding))
		return "";

	return to_resource_register(HLSL_BINDING_AUTO_SAMPLER_BIT, 's', get_decoration(var.self, DecorationBinding),
	                            get_decoration(var.self, DecorationDescriptorSet));
}

void CompilerHLSL::remap_hlsl_resource_binding(HLSLBindingFlagBits type, uint32_t &desc_set, uint32_t &binding)
{
	auto itr = resource_bindings.find({ get_execution_model(), desc_set, binding });
	if (itr != end(resource_bindings))
	{
		auto &remap = itr->second;
		remap.second = true;

		switch (type)
		{
		case HLSL_BINDING_AUTO_PUSH_CONSTANT_BIT:
		case HLSL_BINDING_AUTO_CBV_BIT:
			desc_set = remap.first.cbv.register_space;
			binding = remap.first.cbv.register_binding;
			break;

		case HLSL_BINDING_AUTO_SRV_BIT:
			desc_set = remap.first.srv.register_space;
			binding = remap.first.srv.register_binding;
			break;

		case HLSL_BINDING_AUTO_SAMPLER_BIT:
			desc_set = remap.first.sampler.register_space;
			binding = remap.first.sampler.register_binding;
			break;

		case HLSL_BINDING_AUTO_UAV_BIT:
			desc_set = remap.first.uav.register_space;
			binding = remap.first.uav.register_binding;
			break;

		default:
			break;
		}
	}
}

string CompilerHLSL::to_resource_register(HLSLBindingFlagBits flag, char space, uint32_t binding, uint32_t space_set)
{
	if ((flag & resource_binding_flags) == 0)
	{
		remap_hlsl_resource_binding(flag, space_set, binding);

		// The push constant block did not have a binding, and there were no remap for it,
		// so, declare without register binding.
		if (flag == HLSL_BINDING_AUTO_PUSH_CONSTANT_BIT && space_set == ResourceBindingPushConstantDescriptorSet)
			return "";

		if (hlsl_options.shader_model >= 51)
			return join(" : register(", space, binding, ", space", space_set, ")");
		else
			return join(" : register(", space, binding, ")");
	}
	else
		return "";
}

void CompilerHLSL::emit_modern_uniform(const SPIRVariable &var)
{
	auto &type = get<SPIRType>(var.basetype);
	switch (type.basetype)
	{
	case SPIRType::SampledImage:
	case SPIRType::Image:
	{
		bool is_coherent = false;
		if (type.basetype == SPIRType::Image && type.image.sampled == 2)
			is_coherent = has_decoration(var.self, DecorationCoherent);

		statement(is_coherent ? "globallycoherent " : "", image_type_hlsl_modern(type, var.self), " ",
		          to_name(var.self), type_to_array_glsl(type), to_resource_binding(var), ";");

		if (type.basetype == SPIRType::SampledImage && type.image.dim != DimBuffer)
		{
			// For combined image samplers, also emit a combined image sampler.
			if (is_depth_image(type, var.self))
				statement("SamplerComparisonState ", to_sampler_expression(var.self), type_to_array_glsl(type),
				          to_resource_binding_sampler(var), ";");
			else
				statement("SamplerState ", to_sampler_expression(var.self), type_to_array_glsl(type),
				          to_resource_binding_sampler(var), ";");
		}
		break;
	}

	case SPIRType::Sampler:
		if (comparison_ids.count(var.self))
			statement("SamplerComparisonState ", to_name(var.self), type_to_array_glsl(type), to_resource_binding(var),
			          ";");
		else
			statement("SamplerState ", to_name(var.self), type_to_array_glsl(type), to_resource_binding(var), ";");
		break;

	default:
		statement(variable_decl(var), to_resource_binding(var), ";");
		break;
	}
}

void CompilerHLSL::emit_legacy_uniform(const SPIRVariable &var)
{
	auto &type = get<SPIRType>(var.basetype);
	switch (type.basetype)
	{
	case SPIRType::Sampler:
	case SPIRType::Image:
		SPIRV_CROSS_THROW("Separate image and samplers not supported in legacy HLSL.");

	default:
		statement(variable_decl(var), ";");
		break;
	}
}

void CompilerHLSL::emit_uniform(const SPIRVariable &var)
{
	add_resource_name(var.self);
	if (hlsl_options.shader_model >= 40)
		emit_modern_uniform(var);
	else
		emit_legacy_uniform(var);
}

bool CompilerHLSL::emit_complex_bitcast(uint32_t, uint32_t, uint32_t)
{
	return false;
}

string CompilerHLSL::bitcast_glsl_op(const SPIRType &out_type, const SPIRType &in_type)
{
	if (out_type.basetype == SPIRType::UInt && in_type.basetype == SPIRType::Int)
		return type_to_glsl(out_type);
	else if (out_type.basetype == SPIRType::UInt64 && in_type.basetype == SPIRType::Int64)
		return type_to_glsl(out_type);
	else if (out_type.basetype == SPIRType::UInt && in_type.basetype == SPIRType::Float)
		return "asuint";
	else if (out_type.basetype == SPIRType::Int && in_type.basetype == SPIRType::UInt)
		return type_to_glsl(out_type);
	else if (out_type.basetype == SPIRType::Int64 && in_type.basetype == SPIRType::UInt64)
		return type_to_glsl(out_type);
	else if (out_type.basetype == SPIRType::Int && in_type.basetype == SPIRType::Float)
		return "asint";
	else if (out_type.basetype == SPIRType::Float && in_type.basetype == SPIRType::UInt)
		return "asfloat";
	else if (out_type.basetype == SPIRType::Float && in_type.basetype == SPIRType::Int)
		return "asfloat";
	else if (out_type.basetype == SPIRType::Int64 && in_type.basetype == SPIRType::Double)
		SPIRV_CROSS_THROW("Double to Int64 is not supported in HLSL.");
	else if (out_type.basetype == SPIRType::UInt64 && in_type.basetype == SPIRType::Double)
		SPIRV_CROSS_THROW("Double to UInt64 is not supported in HLSL.");
	else if (out_type.basetype == SPIRType::Double && in_type.basetype == SPIRType::Int64)
		return "asdouble";
	else if (out_type.basetype == SPIRType::Double && in_type.basetype == SPIRType::UInt64)
		return "asdouble";
	else if (out_type.basetype == SPIRType::Half && in_type.basetype == SPIRType::UInt && in_type.vecsize == 1)
	{
		if (!requires_explicit_fp16_packing)
		{
			requires_explicit_fp16_packing = true;
			force_recompile();
		}
		return "spvUnpackFloat2x16";
	}
	else if (out_type.basetype == SPIRType::UInt && in_type.basetype == SPIRType::Half && in_type.vecsize == 2)
	{
		if (!requires_explicit_fp16_packing)
		{
			requires_explicit_fp16_packing = true;
			force_recompile();
		}
		return "spvPackFloat2x16";
	}
	else if (out_type.basetype == SPIRType::UShort && in_type.basetype == SPIRType::Half)
	{
		if (hlsl_options.shader_model < 40)
			SPIRV_CROSS_THROW("Half to UShort requires Shader Model 4.");
		return "(" + type_to_glsl(out_type) + ")f32tof16";
	}
	else if (out_type.basetype == SPIRType::Half && in_type.basetype == SPIRType::UShort)
	{
		if (hlsl_options.shader_model < 40)
			SPIRV_CROSS_THROW("UShort to Half requires Shader Model 4.");
		return "(" + type_to_glsl(out_type) + ")f16tof32";
	}
	else
		return "";
}

void CompilerHLSL::emit_glsl_op(uint32_t result_type, uint32_t id, uint32_t eop, const uint32_t *args, uint32_t count)
{
	auto op = static_cast<GLSLstd450>(eop);

	// If we need to do implicit bitcasts, make sure we do it with the correct type.
	uint32_t integer_width = get_integer_width_for_glsl_instruction(op, args, count);
	auto int_type = to_signed_basetype(integer_width);
	auto uint_type = to_unsigned_basetype(integer_width);

	op = get_remapped_glsl_op(op);

	switch (op)
	{
	case GLSLstd450InverseSqrt:
		emit_unary_func_op(result_type, id, args[0], "rsqrt");
		break;

	case GLSLstd450Fract:
		emit_unary_func_op(result_type, id, args[0], "frac");
		break;

	case GLSLstd450RoundEven:
		if (hlsl_options.shader_model < 40)
			SPIRV_CROSS_THROW("roundEven is not supported in HLSL shader model 2/3.");
		emit_unary_func_op(result_type, id, args[0], "round");
		break;

	case GLSLstd450Acosh:
	case GLSLstd450Asinh:
	case GLSLstd450Atanh:
		SPIRV_CROSS_THROW("Inverse hyperbolics are not supported on HLSL.");

	case GLSLstd450FMix:
	case GLSLstd450IMix:
		emit_trinary_func_op(result_type, id, args[0], args[1], args[2], "lerp");
		break;

	case GLSLstd450Atan2:
		emit_binary_func_op(result_type, id, args[0], args[1], "atan2");
		break;

	case GLSLstd450Fma:
		emit_trinary_func_op(result_type, id, args[0], args[1], args[2], "mad");
		break;

	case GLSLstd450InterpolateAtCentroid:
		emit_unary_func_op(result_type, id, args[0], "EvaluateAttributeAtCentroid");
		break;
	case GLSLstd450InterpolateAtSample:
		emit_binary_func_op(result_type, id, args[0], args[1], "EvaluateAttributeAtSample");
		break;
	case GLSLstd450InterpolateAtOffset:
		emit_binary_func_op(result_type, id, args[0], args[1], "EvaluateAttributeSnapped");
		break;

	case GLSLstd450PackHalf2x16:
		if (!requires_fp16_packing)
		{
			requires_fp16_packing = true;
			force_recompile();
		}
		emit_unary_func_op(result_type, id, args[0], "spvPackHalf2x16");
		break;

	case GLSLstd450UnpackHalf2x16:
		if (!requires_fp16_packing)
		{
			requires_fp16_packing = true;
			force_recompile();
		}
		emit_unary_func_op(result_type, id, args[0], "spvUnpackHalf2x16");
		break;

	case GLSLstd450PackSnorm4x8:
		if (!requires_snorm8_packing)
		{
			requires_snorm8_packing = true;
			force_recompile();
		}
		emit_unary_func_op(result_type, id, args[0], "spvPackSnorm4x8");
		break;

	case GLSLstd450UnpackSnorm4x8:
		if (!requires_snorm8_packing)
		{
			requires_snorm8_packing = true;
			force_recompile();
		}
		emit_unary_func_op(result_type, id, args[0], "spvUnpackSnorm4x8");
		break;

	case GLSLstd450PackUnorm4x8:
		if (!requires_unorm8_packing)
		{
			requires_unorm8_packing = true;
			force_recompile();
		}
		emit_unary_func_op(result_type, id, args[0], "spvPackUnorm4x8");
		break;

	case GLSLstd450UnpackUnorm4x8:
		if (!requires_unorm8_packing)
		{
			requires_unorm8_packing = true;
			force_recompile();
		}
		emit_unary_func_op(result_type, id, args[0], "spvUnpackUnorm4x8");
		break;

	case GLSLstd450PackSnorm2x16:
		if (!requires_snorm16_packing)
		{
			requires_snorm16_packing = true;
			force_recompile();
		}
		emit_unary_func_op(result_type, id, args[0], "spvPackSnorm2x16");
		break;

	case GLSLstd450UnpackSnorm2x16:
		if (!requires_snorm16_packing)
		{
			requires_snorm16_packing = true;
			force_recompile();
		}
		emit_unary_func_op(result_type, id, args[0], "spvUnpackSnorm2x16");
		break;

	case GLSLstd450PackUnorm2x16:
		if (!requires_unorm16_packing)
		{
			requires_unorm16_packing = true;
			force_recompile();
		}
		emit_unary_func_op(result_type, id, args[0], "spvPackUnorm2x16");
		break;

	case GLSLstd450UnpackUnorm2x16:
		if (!requires_unorm16_packing)
		{
			requires_unorm16_packing = true;
			force_recompile();
		}
		emit_unary_func_op(result_type, id, args[0], "spvUnpackUnorm2x16");
		break;

	case GLSLstd450PackDouble2x32:
	case GLSLstd450UnpackDouble2x32:
		SPIRV_CROSS_THROW("packDouble2x32/unpackDouble2x32 not supported in HLSL.");

	case GLSLstd450FindILsb:
	{
		auto basetype = expression_type(args[0]).basetype;
		emit_unary_func_op_cast(result_type, id, args[0], "firstbitlow", basetype, basetype);
		break;
	}

	case GLSLstd450FindSMsb:
		emit_unary_func_op_cast(result_type, id, args[0], "firstbithigh", int_type, int_type);
		break;

	case GLSLstd450FindUMsb:
		emit_unary_func_op_cast(result_type, id, args[0], "firstbithigh", uint_type, uint_type);
		break;

	case GLSLstd450MatrixInverse:
	{
		auto &type = get<SPIRType>(result_type);
		if (type.vecsize == 2 && type.columns == 2)
		{
			if (!requires_inverse_2x2)
			{
				requires_inverse_2x2 = true;
				force_recompile();
			}
		}
		else if (type.vecsize == 3 && type.columns == 3)
		{
			if (!requires_inverse_3x3)
			{
				requires_inverse_3x3 = true;
				force_recompile();
			}
		}
		else if (type.vecsize == 4 && type.columns == 4)
		{
			if (!requires_inverse_4x4)
			{
				requires_inverse_4x4 = true;
				force_recompile();
			}
		}
		emit_unary_func_op(result_type, id, args[0], "spvInverse");
		break;
	}

	case GLSLstd450Normalize:
		// HLSL does not support scalar versions here.
		if (expression_type(args[0]).vecsize == 1)
		{
			// Returns -1 or 1 for valid input, sign() does the job.
			emit_unary_func_op(result_type, id, args[0], "sign");
		}
		else
			CompilerGLSL::emit_glsl_op(result_type, id, eop, args, count);
		break;

	case GLSLstd450Reflect:
		if (get<SPIRType>(result_type).vecsize == 1)
		{
			if (!requires_scalar_reflect)
			{
				requires_scalar_reflect = true;
				force_recompile();
			}
			emit_binary_func_op(result_type, id, args[0], args[1], "spvReflect");
		}
		else
			CompilerGLSL::emit_glsl_op(result_type, id, eop, args, count);
		break;

	case GLSLstd450Refract:
		if (get<SPIRType>(result_type).vecsize == 1)
		{
			if (!requires_scalar_refract)
			{
				requires_scalar_refract = true;
				force_recompile();
			}
			emit_trinary_func_op(result_type, id, args[0], args[1], args[2], "spvRefract");
		}
		else
			CompilerGLSL::emit_glsl_op(result_type, id, eop, args, count);
		break;

	case GLSLstd450FaceForward:
		if (get<SPIRType>(result_type).vecsize == 1)
		{
			if (!requires_scalar_faceforward)
			{
				requires_scalar_faceforward = true;
				force_recompile();
			}
			emit_trinary_func_op(result_type, id, args[0], args[1], args[2], "spvFaceForward");
		}
		else
			CompilerGLSL::emit_glsl_op(result_type, id, eop, args, count);
		break;

	default:
		CompilerGLSL::emit_glsl_op(result_type, id, eop, args, count);
		break;
	}
}

void CompilerHLSL::read_access_chain_array(const string &lhs, const SPIRAccessChain &chain)
{
	auto &type = get<SPIRType>(chain.basetype);

	// Need to use a reserved identifier here since it might shadow an identifier in the access chain input or other loops.
	auto ident = get_unique_identifier();

	statement("[unroll]");
	statement("for (int ", ident, " = 0; ", ident, " < ", to_array_size(type, uint32_t(type.array.size() - 1)), "; ",
	          ident, "++)");
	begin_scope();
	auto subchain = chain;
	subchain.dynamic_index = join(ident, " * ", chain.array_stride, " + ", chain.dynamic_index);
	subchain.basetype = type.parent_type;
	if (!get<SPIRType>(subchain.basetype).array.empty())
		subchain.array_stride = get_decoration(subchain.basetype, DecorationArrayStride);
	read_access_chain(nullptr, join(lhs, "[", ident, "]"), subchain);
	end_scope();
}

void CompilerHLSL::read_access_chain_struct(const string &lhs, const SPIRAccessChain &chain)
{
	auto &type = get<SPIRType>(chain.basetype);
	auto subchain = chain;
	uint32_t member_count = uint32_t(type.member_types.size());

	for (uint32_t i = 0; i < member_count; i++)
	{
		uint32_t offset = type_struct_member_offset(type, i);
		subchain.static_index = chain.static_index + offset;
		subchain.basetype = type.member_types[i];

		subchain.matrix_stride = 0;
		subchain.array_stride = 0;
		subchain.row_major_matrix = false;

		auto &member_type = get<SPIRType>(subchain.basetype);
		if (member_type.columns > 1)
		{
			subchain.matrix_stride = type_struct_member_matrix_stride(type, i);
			subchain.row_major_matrix = has_member_decoration(type.self, i, DecorationRowMajor);
		}

		if (!member_type.array.empty())
			subchain.array_stride = type_struct_member_array_stride(type, i);

		read_access_chain(nullptr, join(lhs, ".", to_member_name(type, i)), subchain);
	}
}

void CompilerHLSL::read_access_chain(string *expr, const string &lhs, const SPIRAccessChain &chain)
{
	auto &type = get<SPIRType>(chain.basetype);

	SPIRType target_type;
	target_type.basetype = SPIRType::UInt;
	target_type.vecsize = type.vecsize;
	target_type.columns = type.columns;

	if (!type.array.empty())
	{
		read_access_chain_array(lhs, chain);
		return;
	}
	else if (type.basetype == SPIRType::Struct)
	{
		read_access_chain_struct(lhs, chain);
		return;
	}
	else if (type.width != 32 && !hlsl_options.enable_16bit_types)
		SPIRV_CROSS_THROW("Reading types other than 32-bit from ByteAddressBuffer not yet supported, unless SM 6.2 and "
		                  "native 16-bit types are enabled.");

	string base = chain.base;
	if (has_decoration(chain.self, DecorationNonUniform))
		convert_non_uniform_expression(base, chain.self);

	bool templated_load = hlsl_options.shader_model >= 62;
	string load_expr;

	string template_expr;
	if (templated_load)
		template_expr = join("<", type_to_glsl(type), ">");

	// Load a vector or scalar.
	if (type.columns == 1 && !chain.row_major_matrix)
	{
		const char *load_op = nullptr;
		switch (type.vecsize)
		{
		case 1:
			load_op = "Load";
			break;
		case 2:
			load_op = "Load2";
			break;
		case 3:
			load_op = "Load3";
			break;
		case 4:
			load_op = "Load4";
			break;
		default:
			SPIRV_CROSS_THROW("Unknown vector size.");
		}

		if (templated_load)
			load_op = "Load";

		load_expr = join(base, ".", load_op, template_expr, "(", chain.dynamic_index, chain.static_index, ")");
	}
	else if (type.columns == 1)
	{
		// Strided load since we are loading a column from a row-major matrix.
		if (templated_load)
		{
			auto scalar_type = type;
			scalar_type.vecsize = 1;
			scalar_type.columns = 1;
			template_expr = join("<", type_to_glsl(scalar_type), ">");
			if (type.vecsize > 1)
				load_expr += type_to_glsl(type) + "(";
		}
		else if (type.vecsize > 1)
		{
			load_expr = type_to_glsl(target_type);
			load_expr += "(";
		}

		for (uint32_t r = 0; r < type.vecsize; r++)
		{
			load_expr += join(base, ".Load", template_expr, "(", chain.dynamic_index,
			                  chain.static_index + r * chain.matrix_stride, ")");
			if (r + 1 < type.vecsize)
				load_expr += ", ";
		}

		if (type.vecsize > 1)
			load_expr += ")";
	}
	else if (!chain.row_major_matrix)
	{
		// Load a matrix, column-major, the easy case.
		const char *load_op = nullptr;
		switch (type.vecsize)
		{
		case 1:
			load_op = "Load";
			break;
		case 2:
			load_op = "Load2";
			break;
		case 3:
			load_op = "Load3";
			break;
		case 4:
			load_op = "Load4";
			break;
		default:
			SPIRV_CROSS_THROW("Unknown vector size.");
		}

		if (templated_load)
		{
			auto vector_type = type;
			vector_type.columns = 1;
			template_expr = join("<", type_to_glsl(vector_type), ">");
			load_expr = type_to_glsl(type);
			load_op = "Load";
		}
		else
		{
			// Note, this loading style in HLSL is *actually* row-major, but we always treat matrices as transposed in this backend,
			// so row-major is technically column-major ...
			load_expr = type_to_glsl(target_type);
		}
		load_expr += "(";

		for (uint32_t c = 0; c < type.columns; c++)
		{
			load_expr += join(base, ".", load_op, template_expr, "(", chain.dynamic_index,
			                  chain.static_index + c * chain.matrix_stride, ")");
			if (c + 1 < type.columns)
				load_expr += ", ";
		}
		load_expr += ")";
	}
	else
	{
		// Pick out elements one by one ... Hopefully compilers are smart enough to recognize this pattern
		// considering HLSL is "row-major decl", but "column-major" memory layout (basically implicit transpose model, ugh) ...

		if (templated_load)
		{
			load_expr = type_to_glsl(type);
			auto scalar_type = type;
			scalar_type.vecsize = 1;
			scalar_type.columns = 1;
			template_expr = join("<", type_to_glsl(scalar_type), ">");
		}
		else
			load_expr = type_to_glsl(target_type);

		load_expr += "(";

		for (uint32_t c = 0; c < type.columns; c++)
		{
			for (uint32_t r = 0; r < type.vecsize; r++)
			{
				load_expr += join(base, ".Load", template_expr, "(", chain.dynamic_index,
				                  chain.static_index + c * (type.width / 8) + r * chain.matrix_stride, ")");

				if ((r + 1 < type.vecsize) || (c + 1 < type.columns))
					load_expr += ", ";
			}
		}
		load_expr += ")";
	}

	if (!templated_load)
	{
		auto bitcast_op = bitcast_glsl_op(type, target_type);
		if (!bitcast_op.empty())
			load_expr = join(bitcast_op, "(", load_expr, ")");
	}

	if (lhs.empty())
	{
		assert(expr);
		*expr = std::move(load_expr);
	}
	else
		statement(lhs, " = ", load_expr, ";");
}

void CompilerHLSL::emit_load(const Instruction &instruction)
{
	auto ops = stream(instruction);

	auto *chain = maybe_get<SPIRAccessChain>(ops[2]);
	if (chain)
	{
		uint32_t result_type = ops[0];
		uint32_t id = ops[1];
		uint32_t ptr = ops[2];

		auto &type = get<SPIRType>(result_type);
		bool composite_load = !type.array.empty() || type.basetype == SPIRType::Struct;

		if (composite_load)
		{
			// We cannot make this work in one single expression as we might have nested structures and arrays,
			// so unroll the load to an uninitialized temporary.
			emit_uninitialized_temporary_expression(result_type, id);
			read_access_chain(nullptr, to_expression(id), *chain);
			track_expression_read(chain->self);
		}
		else
		{
			string load_expr;
			read_access_chain(&load_expr, "", *chain);

			bool forward = should_forward(ptr) && forced_temporaries.find(id) == end(forced_temporaries);

			// If we are forwarding this load,
			// don't register the read to access chain here, defer that to when we actually use the expression,
			// using the add_implied_read_expression mechanism.
			if (!forward)
				track_expression_read(chain->self);

			// Do not forward complex load sequences like matrices, structs and arrays.
			if (type.columns > 1)
				forward = false;

			auto &e = emit_op(result_type, id, load_expr, forward, true);
			e.need_transpose = false;
			register_read(id, ptr, forward);
			inherit_expression_dependencies(id, ptr);
			if (forward)
				add_implied_read_expression(e, chain->self);
		}
	}
	else
		CompilerGLSL::emit_instruction(instruction);
}

void CompilerHLSL::write_access_chain_array(const SPIRAccessChain &chain, uint32_t value,
                                            const SmallVector<uint32_t> &composite_chain)
{
	auto &type = get<SPIRType>(chain.basetype);

	// Need to use a reserved identifier here since it might shadow an identifier in the access chain input or other loops.
	auto ident = get_unique_identifier();

	uint32_t id = ir.increase_bound_by(2);
	uint32_t int_type_id = id + 1;
	SPIRType int_type;
	int_type.basetype = SPIRType::Int;
	int_type.width = 32;
	set<SPIRType>(int_type_id, int_type);
	set<SPIRExpression>(id, ident, int_type_id, true);
	set_name(id, ident);
	suppressed_usage_tracking.insert(id);

	statement("[unroll]");
	statement("for (int ", ident, " = 0; ", ident, " < ", to_array_size(type, uint32_t(type.array.size() - 1)), "; ",
	          ident, "++)");
	begin_scope();
	auto subchain = chain;
	subchain.dynamic_index = join(ident, " * ", chain.array_stride, " + ", chain.dynamic_index);
	subchain.basetype = type.parent_type;

	// Forcefully allow us to use an ID here by setting MSB.
	auto subcomposite_chain = composite_chain;
	subcomposite_chain.push_back(0x80000000u | id);

	if (!get<SPIRType>(subchain.basetype).array.empty())
		subchain.array_stride = get_decoration(subchain.basetype, DecorationArrayStride);

	write_access_chain(subchain, value, subcomposite_chain);
	end_scope();
}

void CompilerHLSL::write_access_chain_struct(const SPIRAccessChain &chain, uint32_t value,
                                             const SmallVector<uint32_t> &composite_chain)
{
	auto &type = get<SPIRType>(chain.basetype);
	uint32_t member_count = uint32_t(type.member_types.size());
	auto subchain = chain;

	auto subcomposite_chain = composite_chain;
	subcomposite_chain.push_back(0);

	for (uint32_t i = 0; i < member_count; i++)
	{
		uint32_t offset = type_struct_member_offset(type, i);
		subchain.static_index = chain.static_index + offset;
		subchain.basetype = type.member_types[i];

		subchain.matrix_stride = 0;
		subchain.array_stride = 0;
		subchain.row_major_matrix = false;

		auto &member_type = get<SPIRType>(subchain.basetype);
		if (member_type.columns > 1)
		{
			subchain.matrix_stride = type_struct_member_matrix_stride(type, i);
			subchain.row_major_matrix = has_member_decoration(type.self, i, DecorationRowMajor);
		}

		if (!member_type.array.empty())
			subchain.array_stride = type_struct_member_array_stride(type, i);

		subcomposite_chain.back() = i;
		write_access_chain(subchain, value, subcomposite_chain);
	}
}

string CompilerHLSL::write_access_chain_value(uint32_t value, const SmallVector<uint32_t> &composite_chain,
                                              bool enclose)
{
	string ret;
	if (composite_chain.empty())
		ret = to_expression(value);
	else
	{
		AccessChainMeta meta;
		ret = access_chain_internal(value, composite_chain.data(), uint32_t(composite_chain.size()),
		                            ACCESS_CHAIN_INDEX_IS_LITERAL_BIT | ACCESS_CHAIN_LITERAL_MSB_FORCE_ID, &meta);
	}

	if (enclose)
		ret = enclose_expression(ret);
	return ret;
}

void CompilerHLSL::write_access_chain(const SPIRAccessChain &chain, uint32_t value,
                                      const SmallVector<uint32_t> &composite_chain)
{
	auto &type = get<SPIRType>(chain.basetype);

	// Make sure we trigger a read of the constituents in the access chain.
	track_expression_read(chain.self);

	SPIRType target_type;
	target_type.basetype = SPIRType::UInt;
	target_type.vecsize = type.vecsize;
	target_type.columns = type.columns;

	if (!type.array.empty())
	{
		write_access_chain_array(chain, value, composite_chain);
		register_write(chain.self);
		return;
	}
	else if (type.basetype == SPIRType::Struct)
	{
		write_access_chain_struct(chain, value, composite_chain);
		register_write(chain.self);
		return;
	}
	else if (type.width != 32 && !hlsl_options.enable_16bit_types)
		SPIRV_CROSS_THROW("Writing types other than 32-bit to RWByteAddressBuffer not yet supported, unless SM 6.2 and "
		                  "native 16-bit types are enabled.");

	bool templated_store = hlsl_options.shader_model >= 62;

	auto base = chain.base;
	if (has_decoration(chain.self, DecorationNonUniform))
		convert_non_uniform_expression(base, chain.self);

	string template_expr;
	if (templated_store)
		template_expr = join("<", type_to_glsl(type), ">");

	if (type.columns == 1 && !chain.row_major_matrix)
	{
		const char *store_op = nullptr;
		switch (type.vecsize)
		{
		case 1:
			store_op = "Store";
			break;
		case 2:
			store_op = "Store2";
			break;
		case 3:
			store_op = "Store3";
			break;
		case 4:
			store_op = "Store4";
			break;
		default:
			SPIRV_CROSS_THROW("Unknown vector size.");
		}

		auto store_expr = write_access_chain_value(value, composite_chain, false);

		if (!templated_store)
		{
			auto bitcast_op = bitcast_glsl_op(target_type, type);
			if (!bitcast_op.empty())
				store_expr = join(bitcast_op, "(", store_expr, ")");
		}
		else
			store_op = "Store";
		statement(base, ".", store_op, template_expr, "(", chain.dynamic_index, chain.static_index, ", ",
		          store_expr, ");");
	}
	else if (type.columns == 1)
	{
		if (templated_store)
		{
			auto scalar_type = type;
			scalar_type.vecsize = 1;
			scalar_type.columns = 1;
			template_expr = join("<", type_to_glsl(scalar_type), ">");
		}

		// Strided store.
		for (uint32_t r = 0; r < type.vecsize; r++)
		{
			auto store_expr = write_access_chain_value(value, composite_chain, true);
			if (type.vecsize > 1)
			{
				store_expr += ".";
				store_expr += index_to_swizzle(r);
			}
			remove_duplicate_swizzle(store_expr);

			if (!templated_store)
			{
				auto bitcast_op = bitcast_glsl_op(target_type, type);
				if (!bitcast_op.empty())
					store_expr = join(bitcast_op, "(", store_expr, ")");
			}

			statement(base, ".Store", template_expr, "(", chain.dynamic_index,
			          chain.static_index + chain.matrix_stride * r, ", ", store_expr, ");");
		}
	}
	else if (!chain.row_major_matrix)
	{
		const char *store_op = nullptr;
		switch (type.vecsize)
		{
		case 1:
			store_op = "Store";
			break;
		case 2:
			store_op = "Store2";
			break;
		case 3:
			store_op = "Store3";
			break;
		case 4:
			store_op = "Store4";
			break;
		default:
			SPIRV_CROSS_THROW("Unknown vector size.");
		}

		if (templated_store)
		{
			store_op = "Store";
			auto vector_type = type;
			vector_type.columns = 1;
			template_expr = join("<", type_to_glsl(vector_type), ">");
		}

		for (uint32_t c = 0; c < type.columns; c++)
		{
			auto store_expr = join(write_access_chain_value(value, composite_chain, true), "[", c, "]");

			if (!templated_store)
			{
				auto bitcast_op = bitcast_glsl_op(target_type, type);
				if (!bitcast_op.empty())
					store_expr = join(bitcast_op, "(", store_expr, ")");
			}

			statement(base, ".", store_op, template_expr, "(", chain.dynamic_index,
			          chain.static_index + c * chain.matrix_stride, ", ", store_expr, ");");
		}
	}
	else
	{
		if (templated_store)
		{
			auto scalar_type = type;
			scalar_type.vecsize = 1;
			scalar_type.columns = 1;
			template_expr = join("<", type_to_glsl(scalar_type), ">");
		}

		for (uint32_t r = 0; r < type.vecsize; r++)
		{
			for (uint32_t c = 0; c < type.columns; c++)
			{
				auto store_expr =
				    join(write_access_chain_value(value, composite_chain, true), "[", c, "].", index_to_swizzle(r));
				remove_duplicate_swizzle(store_expr);
				auto bitcast_op = bitcast_glsl_op(target_type, type);
				if (!bitcast_op.empty())
					store_expr = join(bitcast_op, "(", store_expr, ")");
				statement(base, ".Store", template_expr, "(", chain.dynamic_index,
				          chain.static_index + c * (type.width / 8) + r * chain.matrix_stride, ", ", store_expr, ");");
			}
		}
	}

	register_write(chain.self);
}

void CompilerHLSL::emit_store(const Instruction &instruction)
{
	auto ops = stream(instruction);
	auto *chain = maybe_get<SPIRAccessChain>(ops[0]);
	if (chain)
		write_access_chain(*chain, ops[1], {});
	else
		CompilerGLSL::emit_instruction(instruction);
}

void CompilerHLSL::emit_access_chain(const Instruction &instruction)
{
	auto ops = stream(instruction);
	uint32_t length = instruction.length;

	bool need_byte_access_chain = false;
	auto &type = expression_type(ops[2]);
	const auto *chain = maybe_get<SPIRAccessChain>(ops[2]);

	if (chain)
	{
		// Keep tacking on an existing access chain.
		need_byte_access_chain = true;
	}
	else if (type.storage == StorageClassStorageBuffer || has_decoration(type.self, DecorationBufferBlock))
	{
		// If we are starting to poke into an SSBO, we are dealing with ByteAddressBuffers, and we need
		// to emit SPIRAccessChain rather than a plain SPIRExpression.
		uint32_t chain_arguments = length - 3;
		if (chain_arguments > type.array.size())
			need_byte_access_chain = true;
	}

	if (need_byte_access_chain)
	{
		// If we have a chain variable, we are already inside the SSBO, and any array type will refer to arrays within a block,
		// and not array of SSBO.
		uint32_t to_plain_buffer_length = chain ? 0u : static_cast<uint32_t>(type.array.size());

		auto *backing_variable = maybe_get_backing_variable(ops[2]);

		string base;
		if (to_plain_buffer_length != 0)
			base = access_chain(ops[2], &ops[3], to_plain_buffer_length, get<SPIRType>(ops[0]));
		else if (chain)
			base = chain->base;
		else
			base = to_expression(ops[2]);

		// Start traversing type hierarchy at the proper non-pointer types.
		auto *basetype = &get_pointee_type(type);

		// Traverse the type hierarchy down to the actual buffer types.
		for (uint32_t i = 0; i < to_plain_buffer_length; i++)
		{
			assert(basetype->parent_type);
			basetype = &get<SPIRType>(basetype->parent_type);
		}

		uint32_t matrix_stride = 0;
		uint32_t array_stride = 0;
		bool row_major_matrix = false;

		// Inherit matrix information.
		if (chain)
		{
			matrix_stride = chain->matrix_stride;
			row_major_matrix = chain->row_major_matrix;
			array_stride = chain->array_stride;
		}

		auto offsets = flattened_access_chain_offset(*basetype, &ops[3 + to_plain_buffer_length],
		                                             length - 3 - to_plain_buffer_length, 0, 1, &row_major_matrix,
		                                             &matrix_stride, &array_stride);

		auto &e = set<SPIRAccessChain>(ops[1], ops[0], type.storage, base, offsets.first, offsets.second);
		e.row_major_matrix = row_major_matrix;
		e.matrix_stride = matrix_stride;
		e.array_stride = array_stride;
		e.immutable = should_forward(ops[2]);
		e.loaded_from = backing_variable ? backing_variable->self : ID(0);

		if (chain)
		{
			e.dynamic_index += chain->dynamic_index;
			e.static_index += chain->static_index;
		}

		for (uint32_t i = 2; i < length; i++)
		{
			inherit_expression_dependencies(ops[1], ops[i]);
			add_implied_read_expression(e, ops[i]);
		}
	}
	else
	{
		CompilerGLSL::emit_instruction(instruction);
	}
}

void CompilerHLSL::emit_atomic(const uint32_t *ops, uint32_t length, spv::Op op)
{
	const char *atomic_op = nullptr;

	string value_expr;
	if (op != OpAtomicIDecrement && op != OpAtomicIIncrement && op != OpAtomicLoad && op != OpAtomicStore)
		value_expr = to_expression(ops[op == OpAtomicCompareExchange ? 6 : 5]);

	bool is_atomic_store = false;

	switch (op)
	{
	case OpAtomicIIncrement:
		atomic_op = "InterlockedAdd";
		value_expr = "1";
		break;

	case OpAtomicIDecrement:
		atomic_op = "InterlockedAdd";
		value_expr = "-1";
		break;

	case OpAtomicLoad:
		atomic_op = "InterlockedAdd";
		value_expr = "0";
		break;

	case OpAtomicISub:
		atomic_op = "InterlockedAdd";
		value_expr = join("-", enclose_expression(value_expr));
		break;

	case OpAtomicSMin:
	case OpAtomicUMin:
		atomic_op = "InterlockedMin";
		break;

	case OpAtomicSMax:
	case OpAtomicUMax:
		atomic_op = "InterlockedMax";
		break;

	case OpAtomicAnd:
		atomic_op = "InterlockedAnd";
		break;

	case OpAtomicOr:
		atomic_op = "InterlockedOr";
		break;

	case OpAtomicXor:
		atomic_op = "InterlockedXor";
		break;

	case OpAtomicIAdd:
		atomic_op = "InterlockedAdd";
		break;

	case OpAtomicExchange:
		atomic_op = "InterlockedExchange";
		break;

	case OpAtomicStore:
		atomic_op = "InterlockedExchange";
		is_atomic_store = true;
		break;

	case OpAtomicCompareExchange:
		if (length < 8)
			SPIRV_CROSS_THROW("Not enough data for opcode.");
		atomic_op = "InterlockedCompareExchange";
		value_expr = join(to_expression(ops[7]), ", ", value_expr);
		break;

	default:
		SPIRV_CROSS_THROW("Unknown atomic opcode.");
	}

	if (is_atomic_store)
	{
		auto &data_type = expression_type(ops[0]);
		auto *chain = maybe_get<SPIRAccessChain>(ops[0]);

		auto &tmp_id = extra_sub_expressions[ops[0]];
		if (!tmp_id)
		{
			tmp_id = ir.increase_bound_by(1);
			emit_uninitialized_temporary_expression(get_pointee_type(data_type).self, tmp_id);
		}

		if (data_type.storage == StorageClassImage || !chain)
		{
			statement(atomic_op, "(", to_non_uniform_aware_expression(ops[0]), ", ",
			          to_expression(ops[3]), ", ", to_expression(tmp_id), ");");
		}
		else
		{
			string base = chain->base;
			if (has_decoration(chain->self, DecorationNonUniform))
				convert_non_uniform_expression(base, chain->self);
			// RWByteAddress buffer is always uint in its underlying type.
			statement(base, ".", atomic_op, "(", chain->dynamic_index, chain->static_index, ", ",
			          to_expression(ops[3]), ", ", to_expression(tmp_id), ");");
		}
	}
	else
	{
		uint32_t result_type = ops[0];
		uint32_t id = ops[1];
		forced_temporaries.insert(ops[1]);

		auto &type = get<SPIRType>(result_type);
		statement(variable_decl(type, to_name(id)), ";");

		auto &data_type = expression_type(ops[2]);
		auto *chain = maybe_get<SPIRAccessChain>(ops[2]);
		SPIRType::BaseType expr_type;
		if (data_type.storage == StorageClassImage || !chain)
		{
			statement(atomic_op, "(", to_non_uniform_aware_expression(ops[2]), ", ", value_expr, ", ", to_name(id), ");");
			expr_type = data_type.basetype;
		}
		else
		{
			// RWByteAddress buffer is always uint in its underlying type.
			string base = chain->base;
			if (has_decoration(chain->self, DecorationNonUniform))
				convert_non_uniform_expression(base, chain->self);
			expr_type = SPIRType::UInt;
			statement(base, ".", atomic_op, "(", chain->dynamic_index, chain->static_index, ", ", value_expr,
			          ", ", to_name(id), ");");
		}

		auto expr = bitcast_expression(type, expr_type, to_name(id));
		set<SPIRExpression>(id, expr, result_type, true);
	}
	flush_all_atomic_capable_variables();
}

void CompilerHLSL::emit_subgroup_op(const Instruction &i)
{
	if (hlsl_options.shader_model < 60)
		SPIRV_CROSS_THROW("Wave ops requires SM 6.0 or higher.");

	const uint32_t *ops = stream(i);
	auto op = static_cast<Op>(i.op);

	uint32_t result_type = ops[0];
	uint32_t id = ops[1];

	auto scope = static_cast<Scope>(evaluate_constant_u32(ops[2]));
	if (scope != ScopeSubgroup)
		SPIRV_CROSS_THROW("Only subgroup scope is supported.");

	const auto make_inclusive_Sum = [&](const string &expr) -> string {
		return join(expr, " + ", to_expression(ops[4]));
	};

	const auto make_inclusive_Product = [&](const string &expr) -> string {
		return join(expr, " * ", to_expression(ops[4]));
	};

	// If we need to do implicit bitcasts, make sure we do it with the correct type.
	uint32_t integer_width = get_integer_width_for_instruction(i);
	auto int_type = to_signed_basetype(integer_width);
	auto uint_type = to_unsigned_basetype(integer_width);

#define make_inclusive_BitAnd(expr) ""
#define make_inclusive_BitOr(expr) ""
#define make_inclusive_BitXor(expr) ""
#define make_inclusive_Min(expr) ""
#define make_inclusive_Max(expr) ""

	switch (op)
	{
	case OpGroupNonUniformElect:
		emit_op(result_type, id, "WaveIsFirstLane()", true);
		break;

	case OpGroupNonUniformBroadcast:
		emit_binary_func_op(result_type, id, ops[3], ops[4], "WaveReadLaneAt");
		break;

	case OpGroupNonUniformBroadcastFirst:
		emit_unary_func_op(result_type, id, ops[3], "WaveReadLaneFirst");
		break;

	case OpGroupNonUniformBallot:
		emit_unary_func_op(result_type, id, ops[3], "WaveActiveBallot");
		break;

	case OpGroupNonUniformInverseBallot:
		SPIRV_CROSS_THROW("Cannot trivially implement InverseBallot in HLSL.");

	case OpGroupNonUniformBallotBitExtract:
		SPIRV_CROSS_THROW("Cannot trivially implement BallotBitExtract in HLSL.");

	case OpGroupNonUniformBallotFindLSB:
		SPIRV_CROSS_THROW("Cannot trivially implement BallotFindLSB in HLSL.");

	case OpGroupNonUniformBallotFindMSB:
		SPIRV_CROSS_THROW("Cannot trivially implement BallotFindMSB in HLSL.");

	case OpGroupNonUniformBallotBitCount:
	{
		auto operation = static_cast<GroupOperation>(ops[3]);
		bool forward = should_forward(ops[4]);
		if (operation == GroupOperationReduce)
		{
			auto left = join("countbits(", to_enclosed_expression(ops[4]), ".x) + countbits(",
			                 to_enclosed_expression(ops[4]), ".y)");
			auto right = join("countbits(", to_enclosed_expression(ops[4]), ".z) + countbits(",
			                  to_enclosed_expression(ops[4]), ".w)");
			emit_op(result_type, id, join(left, " + ", right), forward);
			inherit_expression_dependencies(id, ops[4]);
		}
		else if (operation == GroupOperationInclusiveScan)
		{
			auto left = join("countbits(", to_enclosed_expression(ops[4]), ".x & gl_SubgroupLeMask.x) + countbits(",
			                 to_enclosed_expression(ops[4]), ".y & gl_SubgroupLeMask.y)");
			auto right = join("countbits(", to_enclosed_expression(ops[4]), ".z & gl_SubgroupLeMask.z) + countbits(",
			                  to_enclosed_expression(ops[4]), ".w & gl_SubgroupLeMask.w)");
			emit_op(result_type, id, join(left, " + ", right), forward);
			if (!active_input_builtins.get(BuiltInSubgroupLeMask))
			{
				active_input_builtins.set(BuiltInSubgroupLeMask);
				force_recompile_guarantee_forward_progress();
			}
		}
		else if (operation == GroupOperationExclusiveScan)
		{
			auto left = join("countbits(", to_enclosed_expression(ops[4]), ".x & gl_SubgroupLtMask.x) + countbits(",
			                 to_enclosed_expression(ops[4]), ".y & gl_SubgroupLtMask.y)");
			auto right = join("countbits(", to_enclosed_expression(ops[4]), ".z & gl_SubgroupLtMask.z) + countbits(",
			                  to_enclosed_expression(ops[4]), ".w & gl_SubgroupLtMask.w)");
			emit_op(result_type, id, join(left, " + ", right), forward);
			if (!active_input_builtins.get(BuiltInSubgroupLtMask))
			{
				active_input_builtins.set(BuiltInSubgroupLtMask);
				force_recompile_guarantee_forward_progress();
			}
		}
		else
			SPIRV_CROSS_THROW("Invalid BitCount operation.");
		break;
	}

	case OpGroupNonUniformShuffle:
		emit_binary_func_op(result_type, id, ops[3], ops[4], "WaveReadLaneAt");
		break;
	case OpGroupNonUniformShuffleXor:
	{
		bool forward = should_forward(ops[3]);
		emit_op(ops[0], ops[1],
		        join("WaveReadLaneAt(", to_unpacked_expression(ops[3]), ", ",
		             "WaveGetLaneIndex() ^ ", to_enclosed_expression(ops[4]), ")"), forward);
		inherit_expression_dependencies(ops[1], ops[3]);
		break;
	}
	case OpGroupNonUniformShuffleUp:
	{
		bool forward = should_forward(ops[3]);
		emit_op(ops[0], ops[1],
		        join("WaveReadLaneAt(", to_unpacked_expression(ops[3]), ", ",
		             "WaveGetLaneIndex() - ", to_enclosed_expression(ops[4]), ")"), forward);
		inherit_expression_dependencies(ops[1], ops[3]);
		break;
	}
	case OpGroupNonUniformShuffleDown:
	{
		bool forward = should_forward(ops[3]);
		emit_op(ops[0], ops[1],
		        join("WaveReadLaneAt(", to_unpacked_expression(ops[3]), ", ",
		             "WaveGetLaneIndex() + ", to_enclosed_expression(ops[4]), ")"), forward);
		inherit_expression_dependencies(ops[1], ops[3]);
		break;
	}

	case OpGroupNonUniformAll:
		emit_unary_func_op(result_type, id, ops[3], "WaveActiveAllTrue");
		break;

	case OpGroupNonUniformAny:
		emit_unary_func_op(result_type, id, ops[3], "WaveActiveAnyTrue");
		break;

	case OpGroupNonUniformAllEqual:
		emit_unary_func_op(result_type, id, ops[3], "WaveActiveAllEqual");
		break;

	// clang-format off
#define HLSL_GROUP_OP(op, hlsl_op, supports_scan) \
case OpGroupNonUniform##op: \
	{ \
		auto operation = static_cast<GroupOperation>(ops[3]); \
		if (operation == GroupOperationReduce) \
			emit_unary_func_op(result_type, id, ops[4], "WaveActive" #hlsl_op); \
		else if (operation == GroupOperationInclusiveScan && supports_scan) \
        { \
			bool forward = should_forward(ops[4]); \
			emit_op(result_type, id, make_inclusive_##hlsl_op (join("WavePrefix" #hlsl_op, "(", to_expression(ops[4]), ")")), forward); \
			inherit_expression_dependencies(id, ops[4]); \
        } \
		else if (operation == GroupOperationExclusiveScan && supports_scan) \
			emit_unary_func_op(result_type, id, ops[4], "WavePrefix" #hlsl_op); \
		else if (operation == GroupOperationClusteredReduce) \
			SPIRV_CROSS_THROW("Cannot trivially implement ClusteredReduce in HLSL."); \
		else \
			SPIRV_CROSS_THROW("Invalid group operation."); \
		break; \
	}

#define HLSL_GROUP_OP_CAST(op, hlsl_op, type) \
case OpGroupNonUniform##op: \
	{ \
		auto operation = static_cast<GroupOperation>(ops[3]); \
		if (operation == GroupOperationReduce) \
			emit_unary_func_op_cast(result_type, id, ops[4], "WaveActive" #hlsl_op, type, type); \
		else \
			SPIRV_CROSS_THROW("Invalid group operation."); \
		break; \
	}

	HLSL_GROUP_OP(FAdd, Sum, true)
	HLSL_GROUP_OP(FMul, Product, true)
	HLSL_GROUP_OP(FMin, Min, false)
	HLSL_GROUP_OP(FMax, Max, false)
	HLSL_GROUP_OP(IAdd, Sum, true)
	HLSL_GROUP_OP(IMul, Product, true)
	HLSL_GROUP_OP_CAST(SMin, Min, int_type)
	HLSL_GROUP_OP_CAST(SMax, Max, int_type)
	HLSL_GROUP_OP_CAST(UMin, Min, uint_type)
	HLSL_GROUP_OP_CAST(UMax, Max, uint_type)
	HLSL_GROUP_OP(BitwiseAnd, BitAnd, false)
	HLSL_GROUP_OP(BitwiseOr, BitOr, false)
	HLSL_GROUP_OP(BitwiseXor, BitXor, false)
	HLSL_GROUP_OP_CAST(LogicalAnd, BitAnd, uint_type)
	HLSL_GROUP_OP_CAST(LogicalOr, BitOr, uint_type)
	HLSL_GROUP_OP_CAST(LogicalXor, BitXor, uint_type)

#undef HLSL_GROUP_OP
#undef HLSL_GROUP_OP_CAST
		// clang-format on

	case OpGroupNonUniformQuadSwap:
	{
		uint32_t direction = evaluate_constant_u32(ops[4]);
		if (direction == 0)
			emit_unary_func_op(result_type, id, ops[3], "QuadReadAcrossX");
		else if (direction == 1)
			emit_unary_func_op(result_type, id, ops[3], "QuadReadAcrossY");
		else if (direction == 2)
			emit_unary_func_op(result_type, id, ops[3], "QuadReadAcrossDiagonal");
		else
			SPIRV_CROSS_THROW("Invalid quad swap direction.");
		break;
	}

	case OpGroupNonUniformQuadBroadcast:
	{
		emit_binary_func_op(result_type, id, ops[3], ops[4], "QuadReadLaneAt");
		break;
	}

	default:
		SPIRV_CROSS_THROW("Invalid opcode for subgroup.");
	}

	register_control_dependent_expression(id);
}

void CompilerHLSL::emit_instruction(const Instruction &instruction)
{
	auto ops = stream(instruction);
	auto opcode = static_cast<Op>(instruction.op);

#define HLSL_BOP(op) emit_binary_op(ops[0], ops[1], ops[2], ops[3], #op)
#define HLSL_BOP_CAST(op, type) \
	emit_binary_op_cast(ops[0], ops[1], ops[2], ops[3], #op, type, opcode_is_sign_invariant(opcode))
#define HLSL_UOP(op) emit_unary_op(ops[0], ops[1], ops[2], #op)
#define HLSL_QFOP(op) emit_quaternary_func_op(ops[0], ops[1], ops[2], ops[3], ops[4], ops[5], #op)
#define HLSL_TFOP(op) emit_trinary_func_op(ops[0], ops[1], ops[2], ops[3], ops[4], #op)
#define HLSL_BFOP(op) emit_binary_func_op(ops[0], ops[1], ops[2], ops[3], #op)
#define HLSL_BFOP_CAST(op, type) \
	emit_binary_func_op_cast(ops[0], ops[1], ops[2], ops[3], #op, type, opcode_is_sign_invariant(opcode))
#define HLSL_BFOP(op) emit_binary_func_op(ops[0], ops[1], ops[2], ops[3], #op)
#define HLSL_UFOP(op) emit_unary_func_op(ops[0], ops[1], ops[2], #op)

	// If we need to do implicit bitcasts, make sure we do it with the correct type.
	uint32_t integer_width = get_integer_width_for_instruction(instruction);
	auto int_type = to_signed_basetype(integer_width);
	auto uint_type = to_unsigned_basetype(integer_width);

	opcode = get_remapped_spirv_op(opcode);

	switch (opcode)
	{
	case OpAccessChain:
	case OpInBoundsAccessChain:
	{
		emit_access_chain(instruction);
		break;
	}
	case OpBitcast:
	{
		auto bitcast_type = get_bitcast_type(ops[0], ops[2]);
		if (bitcast_type == CompilerHLSL::TypeNormal)
			CompilerGLSL::emit_instruction(instruction);
		else
		{
			if (!requires_uint2_packing)
			{
				requires_uint2_packing = true;
				force_recompile();
			}

			if (bitcast_type == CompilerHLSL::TypePackUint2x32)
				emit_unary_func_op(ops[0], ops[1], ops[2], "spvPackUint2x32");
			else
				emit_unary_func_op(ops[0], ops[1], ops[2], "spvUnpackUint2x32");
		}

		break;
	}

	case OpSelect:
	{
		auto &value_type = expression_type(ops[3]);
		if (value_type.basetype == SPIRType::Struct || is_array(value_type))
		{
			// HLSL does not support ternary expressions on composites.
			// Cannot use branches, since we might be in a continue block
			// where explicit control flow is prohibited.
			// Emit a helper function where we can use control flow.
			TypeID value_type_id = expression_type_id(ops[3]);
			auto itr = std::find(composite_selection_workaround_types.begin(),
			                     composite_selection_workaround_types.end(),
			                     value_type_id);
			if (itr == composite_selection_workaround_types.end())
			{
				composite_selection_workaround_types.push_back(value_type_id);
				force_recompile();
			}
			emit_uninitialized_temporary_expression(ops[0], ops[1]);
			statement("spvSelectComposite(",
					  to_expression(ops[1]), ", ", to_expression(ops[2]), ", ",
					  to_expression(ops[3]), ", ", to_expression(ops[4]), ");");
		}
		else
			CompilerGLSL::emit_instruction(instruction);
		break;
	}

	case OpStore:
	{
		emit_store(instruction);
		break;
	}

	case OpLoad:
	{
		emit_load(instruction);
		break;
	}

	case OpMatrixTimesVector:
	{
		// Matrices are kept in a transposed state all the time, flip multiplication order always.
		emit_binary_func_op(ops[0], ops[1], ops[3], ops[2], "mul");
		break;
	}

	case OpVectorTimesMatrix:
	{
		// Matrices are kept in a transposed state all the time, flip multiplication order always.
		emit_binary_func_op(ops[0], ops[1], ops[3], ops[2], "mul");
		break;
	}

	case OpMatrixTimesMatrix:
	{
		// Matrices are kept in a transposed state all the time, flip multiplication order always.
		emit_binary_func_op(ops[0], ops[1], ops[3], ops[2], "mul");
		break;
	}

	case OpOuterProduct:
	{
		uint32_t result_type = ops[0];
		uint32_t id = ops[1];
		uint32_t a = ops[2];
		uint32_t b = ops[3];

		auto &type = get<SPIRType>(result_type);
		string expr = type_to_glsl_constructor(type);
		expr += "(";
		for (uint32_t col = 0; col < type.columns; col++)
		{
			expr += to_enclosed_expression(a);
			expr += " * ";
			expr += to_extract_component_expression(b, col);
			if (col + 1 < type.columns)
				expr += ", ";
		}
		expr += ")";
		emit_op(result_type, id, expr, should_forward(a) && should_forward(b));
		inherit_expression_dependencies(id, a);
		inherit_expression_dependencies(id, b);
		break;
	}

	case OpFMod:
	{
		if (!requires_op_fmod)
		{
			requires_op_fmod = true;
			force_recompile();
		}
		CompilerGLSL::emit_instruction(instruction);
		break;
	}

	case OpFRem:
		emit_binary_func_op(ops[0], ops[1], ops[2], ops[3], "fmod");
		break;

	case OpImage:
	{
		uint32_t result_type = ops[0];
		uint32_t id = ops[1];
		auto *combined = maybe_get<SPIRCombinedImageSampler>(ops[2]);

		if (combined)
		{
			auto &e = emit_op(result_type, id, to_expression(combined->image), true, true);
			auto *var = maybe_get_backing_variable(combined->image);
			if (var)
				e.loaded_from = var->self;
		}
		else
		{
			auto &e = emit_op(result_type, id, to_expression(ops[2]), true, true);
			auto *var = maybe_get_backing_variable(ops[2]);
			if (var)
				e.loaded_from = var->self;
		}
		break;
	}

	case OpDPdx:
		HLSL_UFOP(ddx);
		register_control_dependent_expression(ops[1]);
		break;

	case OpDPdy:
		HLSL_UFOP(ddy);
		register_control_dependent_expression(ops[1]);
		break;

	case OpDPdxFine:
		HLSL_UFOP(ddx_fine);
		register_control_dependent_expression(ops[1]);
		break;

	case OpDPdyFine:
		HLSL_UFOP(ddy_fine);
		register_control_dependent_expression(ops[1]);
		break;

	case OpDPdxCoarse:
		HLSL_UFOP(ddx_coarse);
		register_control_dependent_expression(ops[1]);
		break;

	case OpDPdyCoarse:
		HLSL_UFOP(ddy_coarse);
		register_control_dependent_expression(ops[1]);
		break;

	case OpFwidth:
	case OpFwidthCoarse:
	case OpFwidthFine:
		HLSL_UFOP(fwidth);
		register_control_dependent_expression(ops[1]);
		break;

	case OpLogicalNot:
	{
		auto result_type = ops[0];
		auto id = ops[1];
		auto &type = get<SPIRType>(result_type);

		if (type.vecsize > 1)
			emit_unrolled_unary_op(result_type, id, ops[2], "!");
		else
			HLSL_UOP(!);
		break;
	}

	case OpIEqual:
	{
		auto result_type = ops[0];
		auto id = ops[1];

		if (expression_type(ops[2]).vecsize > 1)
			emit_unrolled_binary_op(result_type, id, ops[2], ops[3], "==", false, SPIRType::Unknown);
		else
			HLSL_BOP_CAST(==, int_type);
		break;
	}

	case OpLogicalEqual:
	case OpFOrdEqual:
	case OpFUnordEqual:
	{
		// HLSL != operator is unordered.
		// https://docs.microsoft.com/en-us/windows/win32/direct3d10/d3d10-graphics-programming-guide-resources-float-rules.
		// isnan() is apparently implemented as x != x as well.
		// We cannot implement UnordEqual as !(OrdNotEqual), as HLSL cannot express OrdNotEqual.
		// HACK: FUnordEqual will be implemented as FOrdEqual.

		auto result_type = ops[0];
		auto id = ops[1];

		if (expression_type(ops[2]).vecsize > 1)
			emit_unrolled_binary_op(result_type, id, ops[2], ops[3], "==", false, SPIRType::Unknown);
		else
			HLSL_BOP(==);
		break;
	}

	case OpINotEqual:
	{
		auto result_type = ops[0];
		auto id = ops[1];

		if (expression_type(ops[2]).vecsize > 1)
			emit_unrolled_binary_op(result_type, id, ops[2], ops[3], "!=", false, SPIRType::Unknown);
		else
			HLSL_BOP_CAST(!=, int_type);
		break;
	}

	case OpLogicalNotEqual:
	case OpFOrdNotEqual:
	case OpFUnordNotEqual:
	{
		// HLSL != operator is unordered.
		// https://docs.microsoft.com/en-us/windows/win32/direct3d10/d3d10-graphics-programming-guide-resources-float-rules.
		// isnan() is apparently implemented as x != x as well.

		// FIXME: FOrdNotEqual cannot be implemented in a crisp and simple way here.
		// We would need to do something like not(UnordEqual), but that cannot be expressed either.
		// Adding a lot of NaN checks would be a breaking change from perspective of performance.
		// SPIR-V will generally use isnan() checks when this even matters.
		// HACK: FOrdNotEqual will be implemented as FUnordEqual.

		auto result_type = ops[0];
		auto id = ops[1];

		if (expression_type(ops[2]).vecsize > 1)
			emit_unrolled_binary_op(result_type, id, ops[2], ops[3], "!=", false, SPIRType::Unknown);
		else
			HLSL_BOP(!=);
		break;
	}

	case OpUGreaterThan:
	case OpSGreaterThan:
	{
		auto result_type = ops[0];
		auto id = ops[1];
		auto type = opcode == OpUGreaterThan ? uint_type : int_type;

		if (expression_type(ops[2]).vecsize > 1)
			emit_unrolled_binary_op(result_type, id, ops[2], ops[3], ">", false, type);
		else
			HLSL_BOP_CAST(>, type);
		break;
	}

	case OpFOrdGreaterThan:
	{
		auto result_type = ops[0];
		auto id = ops[1];

		if (expression_type(ops[2]).vecsize > 1)
			emit_unrolled_binary_op(result_type, id, ops[2], ops[3], ">", false, SPIRType::Unknown);
		else
			HLSL_BOP(>);
		break;
	}

	case OpFUnordGreaterThan:
	{
		auto result_type = ops[0];
		auto id = ops[1];

		if (expression_type(ops[2]).vecsize > 1)
			emit_unrolled_binary_op(result_type, id, ops[2], ops[3], "<=", true, SPIRType::Unknown);
		else
			CompilerGLSL::emit_instruction(instruction);
		break;
	}

	case OpUGreaterThanEqual:
	case OpSGreaterThanEqual:
	{
		auto result_type = ops[0];
		auto id = ops[1];

		auto type = opcode == OpUGreaterThanEqual ? uint_type : int_type;
		if (expression_type(ops[2]).vecsize > 1)
			emit_unrolled_binary_op(result_type, id, ops[2], ops[3], ">=", false, type);
		else
			HLSL_BOP_CAST(>=, type);
		break;
	}

	case OpFOrdGreaterThanEqual:
	{
		auto result_type = ops[0];
		auto id = ops[1];

		if (expression_type(ops[2]).vecsize > 1)
			emit_unrolled_binary_op(result_type, id, ops[2], ops[3], ">=", false, SPIRType::Unknown);
		else
			HLSL_BOP(>=);
		break;
	}

	case OpFUnordGreaterThanEqual:
	{
		auto result_type = ops[0];
		auto id = ops[1];

		if (expression_type(ops[2]).vecsize > 1)
			emit_unrolled_binary_op(result_type, id, ops[2], ops[3], "<", true, SPIRType::Unknown);
		else
			CompilerGLSL::emit_instruction(instruction);
		break;
	}

	case OpULessThan:
	case OpSLessThan:
	{
		auto result_type = ops[0];
		auto id = ops[1];

		auto type = opcode == OpULessThan ? uint_type : int_type;
		if (expression_type(ops[2]).vecsize > 1)
			emit_unrolled_binary_op(result_type, id, ops[2], ops[3], "<", false, type);
		else
			HLSL_BOP_CAST(<, type);
		break;
	}

	case OpFOrdLessThan:
	{
		auto result_type = ops[0];
		auto id = ops[1];

		if (expression_type(ops[2]).vecsize > 1)
			emit_unrolled_binary_op(result_type, id, ops[2], ops[3], "<", false, SPIRType::Unknown);
		else
			HLSL_BOP(<);
		break;
	}

	case OpFUnordLessThan:
	{
		auto result_type = ops[0];
		auto id = ops[1];

		if (expression_type(ops[2]).vecsize > 1)
			emit_unrolled_binary_op(result_type, id, ops[2], ops[3], ">=", true, SPIRType::Unknown);
		else
			CompilerGLSL::emit_instruction(instruction);
		break;
	}

	case OpULessThanEqual:
	case OpSLessThanEqual:
	{
		auto result_type = ops[0];
		auto id = ops[1];

		auto type = opcode == OpULessThanEqual ? uint_type : int_type;
		if (expression_type(ops[2]).vecsize > 1)
			emit_unrolled_binary_op(result_type, id, ops[2], ops[3], "<=", false, type);
		else
			HLSL_BOP_CAST(<=, type);
		break;
	}

	case OpFOrdLessThanEqual:
	{
		auto result_type = ops[0];
		auto id = ops[1];

		if (expression_type(ops[2]).vecsize > 1)
			emit_unrolled_binary_op(result_type, id, ops[2], ops[3], "<=", false, SPIRType::Unknown);
		else
			HLSL_BOP(<=);
		break;
	}

	case OpFUnordLessThanEqual:
	{
		auto result_type = ops[0];
		auto id = ops[1];

		if (expression_type(ops[2]).vecsize > 1)
			emit_unrolled_binary_op(result_type, id, ops[2], ops[3], ">", true, SPIRType::Unknown);
		else
			CompilerGLSL::emit_instruction(instruction);
		break;
	}

	case OpImageQueryLod:
		emit_texture_op(instruction, false);
		break;

	case OpImageQuerySizeLod:
	{
		auto result_type = ops[0];
		auto id = ops[1];

		require_texture_query_variant(ops[2]);
		auto dummy_samples_levels = join(get_fallback_name(id), "_dummy_parameter");
		statement("uint ", dummy_samples_levels, ";");

		auto expr = join("spvTextureSize(", to_non_uniform_aware_expression(ops[2]), ", ",
		                 bitcast_expression(SPIRType::UInt, ops[3]), ", ", dummy_samples_levels, ")");

		auto &restype = get<SPIRType>(ops[0]);
		expr = bitcast_expression(restype, SPIRType::UInt, expr);
		emit_op(result_type, id, expr, true);
		break;
	}

	case OpImageQuerySize:
	{
		auto result_type = ops[0];
		auto id = ops[1];

		require_texture_query_variant(ops[2]);
		bool uav = expression_type(ops[2]).image.sampled == 2;

		if (const auto *var = maybe_get_backing_variable(ops[2]))
			if (hlsl_options.nonwritable_uav_texture_as_srv && has_decoration(var->self, DecorationNonWritable))
				uav = false;

		auto dummy_samples_levels = join(get_fallback_name(id), "_dummy_parameter");
		statement("uint ", dummy_samples_levels, ";");

		string expr;
		if (uav)
			expr = join("spvImageSize(", to_non_uniform_aware_expression(ops[2]), ", ", dummy_samples_levels, ")");
		else
			expr = join("spvTextureSize(", to_non_uniform_aware_expression(ops[2]), ", 0u, ", dummy_samples_levels, ")");

		auto &restype = get<SPIRType>(ops[0]);
		expr = bitcast_expression(restype, SPIRType::UInt, expr);
		emit_op(result_type, id, expr, true);
		break;
	}

	case OpImageQuerySamples:
	case OpImageQueryLevels:
	{
		auto result_type = ops[0];
		auto id = ops[1];

		require_texture_query_variant(ops[2]);
		bool uav = expression_type(ops[2]).image.sampled == 2;
		if (opcode == OpImageQueryLevels && uav)
			SPIRV_CROSS_THROW("Cannot query levels for UAV images.");

		if (const auto *var = maybe_get_backing_variable(ops[2]))
			if (hlsl_options.nonwritable_uav_texture_as_srv && has_decoration(var->self, DecorationNonWritable))
				uav = false;

		// Keep it simple and do not emit special variants to make this look nicer ...
		// This stuff is barely, if ever, used.
		forced_temporaries.insert(id);
		auto &type = get<SPIRType>(result_type);
		statement(variable_decl(type, to_name(id)), ";");

		if (uav)
			statement("spvImageSize(", to_non_uniform_aware_expression(ops[2]), ", ", to_name(id), ");");
		else
			statement("spvTextureSize(", to_non_uniform_aware_expression(ops[2]), ", 0u, ", to_name(id), ");");

		auto &restype = get<SPIRType>(ops[0]);
		auto expr = bitcast_expression(restype, SPIRType::UInt, to_name(id));
		set<SPIRExpression>(id, expr, result_type, true);
		break;
	}

	case OpImageRead:
	{
		uint32_t result_type = ops[0];
		uint32_t id = ops[1];
		auto *var = maybe_get_backing_variable(ops[2]);
		auto &type = expression_type(ops[2]);
		bool subpass_data = type.image.dim == DimSubpassData;
		bool pure = false;

		string imgexpr;

		if (subpass_data)
		{
			if (hlsl_options.shader_model < 40)
				SPIRV_CROSS_THROW("Subpass loads are not supported in HLSL shader model 2/3.");

			// Similar to GLSL, implement subpass loads using texelFetch.
			if (type.image.ms)
			{
				uint32_t operands = ops[4];
				if (operands != ImageOperandsSampleMask || instruction.length != 6)
					SPIRV_CROSS_THROW("Multisampled image used in OpImageRead, but unexpected operand mask was used.");
				uint32_t sample = ops[5];
				imgexpr = join(to_non_uniform_aware_expression(ops[2]), ".Load(int2(gl_FragCoord.xy), ", to_expression(sample), ")");
			}
			else
				imgexpr = join(to_non_uniform_aware_expression(ops[2]), ".Load(int3(int2(gl_FragCoord.xy), 0))");

			pure = true;
		}
		else
		{
			imgexpr = join(to_non_uniform_aware_expression(ops[2]), "[", to_expression(ops[3]), "]");
			// The underlying image type in HLSL depends on the image format, unlike GLSL, where all images are "vec4",
			// except that the underlying type changes how the data is interpreted.

			bool force_srv =
			    hlsl_options.nonwritable_uav_texture_as_srv && var && has_decoration(var->self, DecorationNonWritable);
			pure = force_srv;

			if (var && !subpass_data && !force_srv)
				imgexpr = remap_swizzle(get<SPIRType>(result_type),
				                        image_format_to_components(get<SPIRType>(var->basetype).image.format), imgexpr);
		}

		if (var)
		{
			bool forward = forced_temporaries.find(id) == end(forced_temporaries);
			auto &e = emit_op(result_type, id, imgexpr, forward);

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

	case OpImageWrite:
	{
		auto *var = maybe_get_backing_variable(ops[0]);

		// The underlying image type in HLSL depends on the image format, unlike GLSL, where all images are "vec4",
		// except that the underlying type changes how the data is interpreted.
		auto value_expr = to_expression(ops[2]);
		if (var)
		{
			auto &type = get<SPIRType>(var->basetype);
			auto narrowed_type = get<SPIRType>(type.image.type);
			narrowed_type.vecsize = image_format_to_components(type.image.format);
			value_expr = remap_swizzle(narrowed_type, expression_type(ops[2]).vecsize, value_expr);
		}

		statement(to_non_uniform_aware_expression(ops[0]), "[", to_expression(ops[1]), "] = ", value_expr, ";");
		if (var && variable_storage_is_aliased(*var))
			flush_all_aliased_variables();
		break;
	}

	case OpImageTexelPointer:
	{
		uint32_t result_type = ops[0];
		uint32_t id = ops[1];

		auto expr = to_expression(ops[2]);
		expr += join("[", to_expression(ops[3]), "]");
		auto &e = set<SPIRExpression>(id, expr, result_type, true);

		// When using the pointer, we need to know which variable it is actually loaded from.
		auto *var = maybe_get_backing_variable(ops[2]);
		e.loaded_from = var ? var->self : ID(0);
		inherit_expression_dependencies(id, ops[3]);
		break;
	}

	case OpAtomicCompareExchange:
	case OpAtomicExchange:
	case OpAtomicISub:
	case OpAtomicSMin:
	case OpAtomicUMin:
	case OpAtomicSMax:
	case OpAtomicUMax:
	case OpAtomicAnd:
	case OpAtomicOr:
	case OpAtomicXor:
	case OpAtomicIAdd:
	case OpAtomicIIncrement:
	case OpAtomicIDecrement:
	case OpAtomicLoad:
	case OpAtomicStore:
	{
		emit_atomic(ops, instruction.length, opcode);
		break;
	}

	case OpControlBarrier:
	case OpMemoryBarrier:
	{
		uint32_t memory;
		uint32_t semantics;

		if (opcode == OpMemoryBarrier)
		{
			memory = evaluate_constant_u32(ops[0]);
			semantics = evaluate_constant_u32(ops[1]);
		}
		else
		{
			memory = evaluate_constant_u32(ops[1]);
			semantics = evaluate_constant_u32(ops[2]);
		}

		if (memory == ScopeSubgroup)
		{
			// No Wave-barriers in HLSL.
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
				uint32_t next_memory = evaluate_constant_u32(next_ops[1]);
				uint32_t next_semantics = evaluate_constant_u32(next_ops[2]);
				next_semantics = mask_relevant_memory_semantics(next_semantics);

				// There is no "just execution barrier" in HLSL.
				// If there are no memory semantics for next instruction, we will imply group shared memory is synced.
				if (next_semantics == 0)
					next_semantics = MemorySemanticsWorkgroupMemoryMask;

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

		if (opcode == OpControlBarrier)
		{
			// We cannot emit just execution barrier, for no memory semantics pick the cheapest option.
			if (semantics == MemorySemanticsWorkgroupMemoryMask || semantics == 0)
				statement("GroupMemoryBarrierWithGroupSync();");
			else if (semantics != 0 && (semantics & MemorySemanticsWorkgroupMemoryMask) == 0)
				statement("DeviceMemoryBarrierWithGroupSync();");
			else
				statement("AllMemoryBarrierWithGroupSync();");
		}
		else
		{
			if (semantics == MemorySemanticsWorkgroupMemoryMask)
				statement("GroupMemoryBarrier();");
			else if (semantics != 0 && (semantics & MemorySemanticsWorkgroupMemoryMask) == 0)
				statement("DeviceMemoryBarrier();");
			else
				statement("AllMemoryBarrier();");
		}
		break;
	}

	case OpBitFieldInsert:
	{
		if (!requires_bitfield_insert)
		{
			requires_bitfield_insert = true;
			force_recompile();
		}

		auto expr = join("spvBitfieldInsert(", to_expression(ops[2]), ", ", to_expression(ops[3]), ", ",
		                 to_expression(ops[4]), ", ", to_expression(ops[5]), ")");

		bool forward =
		    should_forward(ops[2]) && should_forward(ops[3]) && should_forward(ops[4]) && should_forward(ops[5]);

		auto &restype = get<SPIRType>(ops[0]);
		expr = bitcast_expression(restype, SPIRType::UInt, expr);
		emit_op(ops[0], ops[1], expr, forward);
		break;
	}

	case OpBitFieldSExtract:
	case OpBitFieldUExtract:
	{
		if (!requires_bitfield_extract)
		{
			requires_bitfield_extract = true;
			force_recompile();
		}

		if (opcode == OpBitFieldSExtract)
			HLSL_TFOP(spvBitfieldSExtract);
		else
			HLSL_TFOP(spvBitfieldUExtract);
		break;
	}

	case OpBitCount:
	{
		auto basetype = expression_type(ops[2]).basetype;
		emit_unary_func_op_cast(ops[0], ops[1], ops[2], "countbits", basetype, basetype);
		break;
	}

	case OpBitReverse:
		HLSL_UFOP(reversebits);
		break;

	case OpArrayLength:
	{
		auto *var = maybe_get_backing_variable(ops[2]);
		if (!var)
			SPIRV_CROSS_THROW("Array length must point directly to an SSBO block.");

		auto &type = get<SPIRType>(var->basetype);
		if (!has_decoration(type.self, DecorationBlock) && !has_decoration(type.self, DecorationBufferBlock))
			SPIRV_CROSS_THROW("Array length expression must point to a block type.");

		// This must be 32-bit uint, so we're good to go.
		emit_uninitialized_temporary_expression(ops[0], ops[1]);
		statement(to_non_uniform_aware_expression(ops[2]), ".GetDimensions(", to_expression(ops[1]), ");");
		uint32_t offset = type_struct_member_offset(type, ops[3]);
		uint32_t stride = type_struct_member_array_stride(type, ops[3]);
		statement(to_expression(ops[1]), " = (", to_expression(ops[1]), " - ", offset, ") / ", stride, ";");
		break;
	}

	case OpIsHelperInvocationEXT:
		if (hlsl_options.shader_model < 50 || get_entry_point().model != ExecutionModelFragment)
			SPIRV_CROSS_THROW("Helper Invocation input is only supported in PS 5.0 or higher.");
		// Helper lane state with demote is volatile by nature.
		// Do not forward this.
		emit_op(ops[0], ops[1], "IsHelperLane()", false);
		break;

	case OpBeginInvocationInterlockEXT:
	case OpEndInvocationInterlockEXT:
		if (hlsl_options.shader_model < 51)
			SPIRV_CROSS_THROW("Rasterizer order views require Shader Model 5.1.");
		break; // Nothing to do in the body

	case OpRayQueryInitializeKHR:
	{
		flush_variable_declaration(ops[0]);

		std::string ray_desc_name = get_unique_identifier();
		statement("RayDesc ", ray_desc_name, " = {", to_expression(ops[4]), ", ", to_expression(ops[5]), ", ",
			to_expression(ops[6]), ", ", to_expression(ops[7]), "};");

		statement(to_expression(ops[0]), ".TraceRayInline(", 
			to_expression(ops[1]), ", ", // acc structure
			to_expression(ops[2]), ", ", // ray flags
			to_expression(ops[3]), ", ", // mask
			ray_desc_name, ");"); // ray
		break;
	}
	case OpRayQueryProceedKHR:
	{
		flush_variable_declaration(ops[0]);
		emit_op(ops[0], ops[1], join(to_expression(ops[2]), ".Proceed()"), false);
		break;
	}	
	case OpRayQueryTerminateKHR:
	{
		flush_variable_declaration(ops[0]);
		statement(to_expression(ops[0]), ".Abort();");
		break;
	}
	case OpRayQueryGenerateIntersectionKHR:
	{
		flush_variable_declaration(ops[0]);
		statement(to_expression(ops[0]), ".CommitProceduralPrimitiveHit(", ops[1], ");");
		break;
	}
	case OpRayQueryConfirmIntersectionKHR:
	{
		flush_variable_declaration(ops[0]);
		statement(to_expression(ops[0]), ".CommitNonOpaqueTriangleHit();");
		break;
	}
	case OpRayQueryGetIntersectionTypeKHR:
	{
		emit_rayquery_function(".CommittedStatus()", ".CandidateType()", ops);
		break;
	}
	case OpRayQueryGetIntersectionTKHR:
	{
		emit_rayquery_function(".CommittedRayT()", ".CandidateTriangleRayT()", ops);
		break;
	}
	case OpRayQueryGetIntersectionInstanceCustomIndexKHR:
	{
		emit_rayquery_function(".CommittedInstanceID()", ".CandidateInstanceID()", ops);
		break;
	}
	case OpRayQueryGetIntersectionInstanceIdKHR:
	{
		emit_rayquery_function(".CommittedInstanceIndex()", ".CandidateInstanceIndex()", ops);
		break;
	}
	case OpRayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetKHR:
	{
		emit_rayquery_function(".CommittedInstanceContributionToHitGroupIndex()", 
			".CandidateInstanceContributionToHitGroupIndex()", ops);
		break;
	}
	case OpRayQueryGetIntersectionGeometryIndexKHR:
	{
		emit_rayquery_function(".CommittedGeometryIndex()",
				".CandidateGeometryIndex()", ops);
		break;
	}
	case OpRayQueryGetIntersectionPrimitiveIndexKHR:
	{
		emit_rayquery_function(".CommittedPrimitiveIndex()", ".CandidatePrimitiveIndex()", ops);
		break;
	}
	case OpRayQueryGetIntersectionBarycentricsKHR:
	{
		emit_rayquery_function(".CommittedTriangleBarycentrics()", ".CandidateTriangleBarycentrics()", ops);
		break;
	}
	case OpRayQueryGetIntersectionFrontFaceKHR:
	{
		emit_rayquery_function(".CommittedTriangleFrontFace()", ".CandidateTriangleFrontFace()", ops);
		break;
	}
	case OpRayQueryGetIntersectionCandidateAABBOpaqueKHR:
	{
		flush_variable_declaration(ops[0]);
		emit_op(ops[0], ops[1], join(to_expression(ops[2]), ".CandidateProceduralPrimitiveNonOpaque()"), false);
		break;
	}
	case OpRayQueryGetIntersectionObjectRayDirectionKHR:
	{
		emit_rayquery_function(".CommittedObjectRayDirection()", ".CandidateObjectRayDirection()", ops);
		break;
	}
	case OpRayQueryGetIntersectionObjectRayOriginKHR:
	{
		flush_variable_declaration(ops[0]);
		emit_rayquery_function(".CommittedObjectRayOrigin()", ".CandidateObjectRayOrigin()", ops);
		break;
	}
	case OpRayQueryGetIntersectionObjectToWorldKHR:
	{
		emit_rayquery_function(".CommittedObjectToWorld4x3()", ".CandidateObjectToWorld4x3()", ops);
		break;
	}
	case OpRayQueryGetIntersectionWorldToObjectKHR:
	{
		emit_rayquery_function(".CommittedWorldToObject4x3()", ".CandidateWorldToObject4x3()", ops);
		break;
	}
	case OpRayQueryGetRayFlagsKHR:
	{
		flush_variable_declaration(ops[0]);
		emit_op(ops[0], ops[1], join(to_expression(ops[2]), ".RayFlags()"), false);
		break;
	}
	case OpRayQueryGetRayTMinKHR:
	{
		flush_variable_declaration(ops[0]);
		emit_op(ops[0], ops[1], join(to_expression(ops[2]), ".RayTMin()"), false);
		break;
	}
	case OpRayQueryGetWorldRayOriginKHR:
	{
		flush_variable_declaration(ops[0]);
		emit_op(ops[0], ops[1], join(to_expression(ops[2]), ".WorldRayOrigin()"), false);
		break;
	}
	case OpRayQueryGetWorldRayDirectionKHR:
	{
		flush_variable_declaration(ops[0]);
		emit_op(ops[0], ops[1], join(to_expression(ops[2]), ".WorldRayDirection()"), false);
		break;
	}
	default:
		CompilerGLSL::emit_instruction(instruction);
		break;
	}
}

void CompilerHLSL::require_texture_query_variant(uint32_t var_id)
{
	if (const auto *var = maybe_get_backing_variable(var_id))
		var_id = var->self;

	auto &type = expression_type(var_id);
	bool uav = type.image.sampled == 2;
	if (hlsl_options.nonwritable_uav_texture_as_srv && has_decoration(var_id, DecorationNonWritable))
		uav = false;

	uint32_t bit = 0;
	switch (type.image.dim)
	{
	case Dim1D:
		bit = type.image.arrayed ? Query1DArray : Query1D;
		break;

	case Dim2D:
		if (type.image.ms)
			bit = type.image.arrayed ? Query2DMSArray : Query2DMS;
		else
			bit = type.image.arrayed ? Query2DArray : Query2D;
		break;

	case Dim3D:
		bit = Query3D;
		break;

	case DimCube:
		bit = type.image.arrayed ? QueryCubeArray : QueryCube;
		break;

	case DimBuffer:
		bit = QueryBuffer;
		break;

	default:
		SPIRV_CROSS_THROW("Unsupported query type.");
	}

	switch (get<SPIRType>(type.image.type).basetype)
	{
	case SPIRType::Float:
		bit += QueryTypeFloat;
		break;

	case SPIRType::Int:
		bit += QueryTypeInt;
		break;

	case SPIRType::UInt:
		bit += QueryTypeUInt;
		break;

	default:
		SPIRV_CROSS_THROW("Unsupported query type.");
	}

	auto norm_state = image_format_to_normalized_state(type.image.format);
	auto &variant = uav ? required_texture_size_variants
	                          .uav[uint32_t(norm_state)][image_format_to_components(type.image.format) - 1] :
	                      required_texture_size_variants.srv;

	uint64_t mask = 1ull << bit;
	if ((variant & mask) == 0)
	{
		force_recompile();
		variant |= mask;
	}
}

void CompilerHLSL::set_root_constant_layouts(std::vector<RootConstants> layout)
{
	root_constants_layout = std::move(layout);
}

void CompilerHLSL::add_vertex_attribute_remap(const HLSLVertexAttributeRemap &vertex_attributes)
{
	remap_vertex_attributes.push_back(vertex_attributes);
}

VariableID CompilerHLSL::remap_num_workgroups_builtin()
{
	update_active_builtins();

	if (!active_input_builtins.get(BuiltInNumWorkgroups))
		return 0;

	// Create a new, fake UBO.
	uint32_t offset = ir.increase_bound_by(4);

	uint32_t uint_type_id = offset;
	uint32_t block_type_id = offset + 1;
	uint32_t block_pointer_type_id = offset + 2;
	uint32_t variable_id = offset + 3;

	SPIRType uint_type;
	uint_type.basetype = SPIRType::UInt;
	uint_type.width = 32;
	uint_type.vecsize = 3;
	uint_type.columns = 1;
	set<SPIRType>(uint_type_id, uint_type);

	SPIRType block_type;
	block_type.basetype = SPIRType::Struct;
	block_type.member_types.push_back(uint_type_id);
	set<SPIRType>(block_type_id, block_type);
	set_decoration(block_type_id, DecorationBlock);
	set_member_name(block_type_id, 0, "count");
	set_member_decoration(block_type_id, 0, DecorationOffset, 0);

	SPIRType block_pointer_type = block_type;
	block_pointer_type.pointer = true;
	block_pointer_type.storage = StorageClassUniform;
	block_pointer_type.parent_type = block_type_id;
	auto &ptr_type = set<SPIRType>(block_pointer_type_id, block_pointer_type);

	// Preserve self.
	ptr_type.self = block_type_id;

	set<SPIRVariable>(variable_id, block_pointer_type_id, StorageClassUniform);
	ir.meta[variable_id].decoration.alias = "SPIRV_Cross_NumWorkgroups";

	num_workgroups_builtin = variable_id;
	get_entry_point().interface_variables.push_back(num_workgroups_builtin);
	return variable_id;
}

void CompilerHLSL::set_resource_binding_flags(HLSLBindingFlags flags)
{
	resource_binding_flags = flags;
}

void CompilerHLSL::validate_shader_model()
{
	// Check for nonuniform qualifier.
	// Instead of looping over all decorations to find this, just look at capabilities.
	for (auto &cap : ir.declared_capabilities)
	{
		switch (cap)
		{
		case CapabilityShaderNonUniformEXT:
		case CapabilityRuntimeDescriptorArrayEXT:
			if (hlsl_options.shader_model < 51)
				SPIRV_CROSS_THROW(
				    "Shader model 5.1 or higher is required to use bindless resources or NonUniformResourceIndex.");
			break;

		case CapabilityVariablePointers:
		case CapabilityVariablePointersStorageBuffer:
			SPIRV_CROSS_THROW("VariablePointers capability is not supported in HLSL.");

		default:
			break;
		}
	}

	if (ir.addressing_model != AddressingModelLogical)
		SPIRV_CROSS_THROW("Only Logical addressing model can be used with HLSL.");

	if (hlsl_options.enable_16bit_types && hlsl_options.shader_model < 62)
		SPIRV_CROSS_THROW("Need at least shader model 6.2 when enabling native 16-bit type support.");
}

string CompilerHLSL::compile()
{
	ir.fixup_reserved_names();

	// Do not deal with ES-isms like precision, older extensions and such.
	options.es = false;
	options.version = 450;
	options.vulkan_semantics = true;
	backend.float_literal_suffix = true;
	backend.double_literal_suffix = false;
	backend.long_long_literal_suffix = true;
	backend.uint32_t_literal_suffix = true;
	backend.int16_t_literal_suffix = "";
	backend.uint16_t_literal_suffix = "u";
	backend.basic_int_type = "int";
	backend.basic_uint_type = "uint";
	backend.demote_literal = "discard";
	backend.boolean_mix_function = "";
	backend.swizzle_is_function = false;
	backend.shared_is_implied = true;
	backend.unsized_array_supported = true;
	backend.explicit_struct_type = false;
	backend.use_initializer_list = true;
	backend.use_constructor_splatting = false;
	backend.can_swizzle_scalar = true;
	backend.can_declare_struct_inline = false;
	backend.can_declare_arrays_inline = false;
	backend.can_return_array = false;
	backend.nonuniform_qualifier = "NonUniformResourceIndex";
	backend.support_case_fallthrough = false;

	// SM 4.1 does not support precise for some reason.
	backend.support_precise_qualifier = hlsl_options.shader_model >= 50 || hlsl_options.shader_model == 40;

	fixup_anonymous_struct_names();
	fixup_type_alias();
	reorder_type_alias();
	build_function_control_flow_graphs_and_analyze();
	validate_shader_model();
	update_active_builtins();
	analyze_image_and_sampler_usage();
	analyze_interlocked_resource_usage();

	// Subpass input needs SV_Position.
	if (need_subpass_input)
		active_input_builtins.set(BuiltInFragCoord);

	uint32_t pass_count = 0;
	do
	{
		reset(pass_count);

		// Move constructor for this type is broken on GCC 4.9 ...
		buffer.reset();

		emit_header();
		emit_resources();

		emit_function(get<SPIRFunction>(ir.default_entry_point), Bitset());
		emit_hlsl_entry_point();

		pass_count++;
	} while (is_forcing_recompilation());

	// Entry point in HLSL is always main() for the time being.
	get_entry_point().name = "main";

	return buffer.str();
}

void CompilerHLSL::emit_block_hints(const SPIRBlock &block)
{
	switch (block.hint)
	{
	case SPIRBlock::HintFlatten:
		statement("[flatten]");
		break;
	case SPIRBlock::HintDontFlatten:
		statement("[branch]");
		break;
	case SPIRBlock::HintUnroll:
		statement("[unroll]");
		break;
	case SPIRBlock::HintDontUnroll:
		statement("[loop]");
		break;
	default:
		break;
	}
}

string CompilerHLSL::get_unique_identifier()
{
	return join("_", unique_identifier_count++, "ident");
}

void CompilerHLSL::add_hlsl_resource_binding(const HLSLResourceBinding &binding)
{
	StageSetBinding tuple = { binding.stage, binding.desc_set, binding.binding };
	resource_bindings[tuple] = { binding, false };
}

bool CompilerHLSL::is_hlsl_resource_binding_used(ExecutionModel model, uint32_t desc_set, uint32_t binding) const
{
	StageSetBinding tuple = { model, desc_set, binding };
	auto itr = resource_bindings.find(tuple);
	return itr != end(resource_bindings) && itr->second.second;
}

CompilerHLSL::BitcastType CompilerHLSL::get_bitcast_type(uint32_t result_type, uint32_t op0)
{
	auto &rslt_type = get<SPIRType>(result_type);
	auto &expr_type = expression_type(op0);

	if (rslt_type.basetype == SPIRType::BaseType::UInt64 && expr_type.basetype == SPIRType::BaseType::UInt &&
	    expr_type.vecsize == 2)
		return BitcastType::TypePackUint2x32;
	else if (rslt_type.basetype == SPIRType::BaseType::UInt && rslt_type.vecsize == 2 &&
	         expr_type.basetype == SPIRType::BaseType::UInt64)
		return BitcastType::TypeUnpackUint64;

	return BitcastType::TypeNormal;
}

bool CompilerHLSL::is_hlsl_force_storage_buffer_as_uav(ID id) const
{
	if (hlsl_options.force_storage_buffer_as_uav)
	{
		return true;
	}

	const uint32_t desc_set = get_decoration(id, spv::DecorationDescriptorSet);
	const uint32_t binding = get_decoration(id, spv::DecorationBinding);

	return (force_uav_buffer_bindings.find({ desc_set, binding }) != force_uav_buffer_bindings.end());
}

void CompilerHLSL::set_hlsl_force_storage_buffer_as_uav(uint32_t desc_set, uint32_t binding)
{
	SetBindingPair pair = { desc_set, binding };
	force_uav_buffer_bindings.insert(pair);
}

bool CompilerHLSL::builtin_translates_to_nonarray(spv::BuiltIn builtin) const
{
	return (builtin == BuiltInSampleMask);
}
