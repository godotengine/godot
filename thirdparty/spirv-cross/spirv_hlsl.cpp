/*
 * Copyright 2016-2019 Robert Konrad
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

#include "spirv_hlsl.hpp"
#include "GLSL.std.450.h"
#include <algorithm>
#include <assert.h>

using namespace spv;
using namespace SPIRV_CROSS_NAMESPACE;
using namespace std;

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

string CompilerHLSL::image_type_hlsl_modern(const SPIRType &type, uint32_t)
{
	auto &imagetype = get<SPIRType>(type.image.type);
	const char *dim = nullptr;
	bool typed_load = false;
	uint32_t components = 4;

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
			return join("RWBuffer<", image_format_to_type(type.image.format, imagetype.basetype), ">");
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
	const char *rw = typed_load ? "RW" : "";
	return join(rw, "Texture", dim, ms, arrayed, "<",
	            typed_load ? image_format_to_type(type.image.format, imagetype.basetype) :
	                         join(type_to_glsl(imagetype), components),
	            ">");
}

string CompilerHLSL::image_type_hlsl_legacy(const SPIRType &type, uint32_t id)
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
	if (image_is_comparison(type, id))
		res += "Shadow";

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
			return "min16float";
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
			return join("bool", type.vecsize);
		case SPIRType::Int:
			return join("int", type.vecsize);
		case SPIRType::UInt:
			return join("uint", type.vecsize);
		case SPIRType::Half:
			return join("min16float", type.vecsize);
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
			return join("min16float", type.columns, "x", type.vecsize);
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
			type = "float4";
			semantic = legacy ? "POSITION" : "SV_Position";
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

		default:
			SPIRV_CROSS_THROW("Unsupported builtin in HLSL.");
			break;
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

		default:
			SPIRV_CROSS_THROW("Unsupported builtin in HLSL.");
			break;
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
				array_multiplier *= get<SPIRConstant>(type.array[i]).scalar();
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
	if (flags.get(DecorationInvariant))
		res += "invariant "; // Not supported?

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

void CompilerHLSL::emit_io_block(const SPIRVariable &var)
{
	auto &execution = get_entry_point();

	auto &type = get<SPIRType>(var.basetype);
	add_resource_name(type.self);

	statement("struct ", to_name(type.self));
	begin_scope();
	type.member_name_cache.clear();

	uint32_t base_location = get_decoration(var.self, DecorationLocation);

	for (uint32_t i = 0; i < uint32_t(type.member_types.size()); i++)
	{
		string semantic;
		if (has_member_decoration(type.self, i, DecorationLocation))
		{
			uint32_t location = get_member_decoration(type.self, i, DecorationLocation);
			semantic = join(" : ", to_semantic(location, execution.model, var.storage));
		}
		else
		{
			// If the block itself has a location, but not its members, use the implicit location.
			// There could be a conflict if the block members partially specialize the locations.
			// It is unclear how SPIR-V deals with this. Assume this does not happen for now.
			uint32_t location = base_location + i;
			semantic = join(" : ", to_semantic(location, execution.model, var.storage));
		}

		add_member_name(type, i);

		auto &membertype = get<SPIRType>(type.member_types[i]);
		statement(to_interpolation_qualifiers(get_member_decoration_bitset(type.self, i)),
		          variable_decl(membertype, to_member_name(type, i)), semantic, ";");
	}

	end_scope_decl();
	statement("");

	statement("static ", variable_decl(var), ";");
	statement("");
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

	auto &m = ir.meta[var.self].decoration;
	auto name = to_name(var.self);
	if (use_location_number)
	{
		uint32_t location_number;

		// If an explicit location exists, use it with TEXCOORD[N] semantic.
		// Otherwise, pick a vacant location.
		if (m.decoration_flags.get(DecorationLocation))
			location_number = m.location;
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
				statement(to_interpolation_qualifiers(get_decoration_bitset(var.self)),
				          variable_decl(newtype, join(name, "_", i)), " : ", semantic, "_", i, ";");
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
		return sanitize_underscores(join(to_name(num_workgroups_builtin), "_", get_member_name(type.self, 0)));
	}
	case BuiltInPointCoord:
		// Crude hack, but there is no real alternative. This path is only enabled if point_coord_compat is set.
		return "float2(0.5f, 0.5f)";
	case BuiltInSubgroupLocalInvocationId:
		return "WaveGetLaneIndex()";
	case BuiltInSubgroupSize:
		return "WaveGetLaneCount()";

	default:
		return CompilerGLSL::builtin_to_glsl(builtin, storage);
	}
}

void CompilerHLSL::emit_builtin_variables()
{
	Bitset builtins = active_input_builtins;
	builtins.merge_or(active_output_builtins);

	bool need_base_vertex_info = false;

	// Emit global variables for the interface variables which are statically used by the shader.
	builtins.for_each_bit([&](uint32_t i) {
		const char *type = nullptr;
		auto builtin = static_cast<BuiltIn>(i);
		uint32_t array_size = 0;

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

		case BuiltInClipDistance:
			array_size = clip_distance_count;
			type = "float";
			break;

		case BuiltInCullDistance:
			array_size = cull_distance_count;
			type = "float";
			break;

		default:
			SPIRV_CROSS_THROW(join("Unsupported builtin in HLSL: ", unsigned(builtin)));
		}

		StorageClass storage = active_input_builtins.get(i) ? StorageClassInput : StorageClassOutput;
		// FIXME: SampleMask can be both in and out with sample builtin,
		// need to distinguish that when we add support for that.

		if (type)
		{
			if (array_size)
				statement("static ", type, " ", builtin_to_glsl(builtin, storage), "[", array_size, "];");
			else
				statement("static ", type, " ", builtin_to_glsl(builtin, storage), ";");
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
		if (type.basetype == SPIRType::Struct || !type.array.empty())
		{
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
	uint32_t workgroup_size_id = get_work_group_size_specialization_constants(wg_x, wg_y, wg_z);

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
				auto name = to_name(c.self);

				// HLSL does not support specialization constants, so fallback to macros.
				c.specialization_constant_macro_name =
				    constant_value_macro_name(get_decoration(c.self, DecorationSpecId));

				statement("#ifndef ", c.specialization_constant_macro_name);
				statement("#define ", c.specialization_constant_macro_name, " ", constant_expression(c));
				statement("#endif");
				statement("static const ", variable_decl(type, name), " = ", c.specialization_constant_macro_name, ";");
				emitted = true;
			}
		}
		else if (id.get_type() == TypeConstantOp)
		{
			auto &c = id.get<SPIRConstantOp>();
			auto &type = get<SPIRType>(c.basetype);
			auto name = to_name(c.self);
			statement("static const ", variable_decl(type, name), " = ", constant_op_expression(c), ";");
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

	if (emitted)
		statement("");
}

void CompilerHLSL::replace_illegal_names()
{
	static const unordered_set<string> keywords = {
		// Additional HLSL specific keywords.
		"line", "linear", "matrix", "point", "row_major", "sampler",
	};

	ir.for_each_typed_id<SPIRVariable>([&](uint32_t, SPIRVariable &var) {
		if (!is_hidden_variable(var))
		{
			auto &m = ir.meta[var.self].decoration;
			if (keywords.find(m.alias) != end(keywords))
				m.alias = join("_", m.alias);
		}
	});

	CompilerGLSL::replace_illegal_names();
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

	if (execution.model == ExecutionModelVertex && hlsl_options.shader_model <= 30)
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
		    type.pointer && (type.storage == StorageClassUniformConstant || type.storage == StorageClassAtomicCounter))
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
		bool block = ir.meta[type.self].decoration.decoration_flags.get(DecorationBlock);

		// Do not emit I/O blocks here.
		// I/O blocks can be arrayed, so we must deal with them separately to support geometry shaders
		// and tessellation down the line.
		if (!block && var.storage != StorageClassFunction && !var.remapped_variable && type.pointer &&
		    (var.storage == StorageClassInput || var.storage == StorageClassOutput) && !is_builtin_variable(var) &&
		    interface_variable_exists_in_entry_point(var.self))
		{
			// Only emit non-builtins which are not blocks here. Builtin variables are handled separately.
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
	SmallVector<SPIRVariable *> input_variables;
	SmallVector<SPIRVariable *> output_variables;
	ir.for_each_typed_id<SPIRVariable>([&](uint32_t, SPIRVariable &var) {
		auto &type = this->get<SPIRType>(var.basetype);
		bool block = ir.meta[type.self].decoration.decoration_flags.get(DecorationBlock);

		if (var.storage != StorageClassInput && var.storage != StorageClassOutput)
			return;

		// Do not emit I/O blocks here.
		// I/O blocks can be arrayed, so we must deal with them separately to support geometry shaders
		// and tessellation down the line.
		if (!block && !var.remapped_variable && type.pointer && !is_builtin_variable(var) &&
		    interface_variable_exists_in_entry_point(var.self))
		{
			if (var.storage == StorageClassInput)
				input_variables.push_back(&var);
			else
				output_variables.push_back(&var);
		}

		// Reserve input and output locations for block variables as necessary.
		if (block && !is_builtin_variable(var) && interface_variable_exists_in_entry_point(var.self))
		{
			auto &active = var.storage == StorageClassInput ? active_inputs : active_outputs;
			for (uint32_t i = 0; i < uint32_t(type.member_types.size()); i++)
			{
				if (has_member_decoration(type.self, i, DecorationLocation))
				{
					uint32_t location = get_member_decoration(type.self, i, DecorationLocation);
					active.insert(location);
				}
			}

			// Emit the block struct and a global variable here.
			emit_io_block(var);
		}
	});

	const auto variable_compare = [&](const SPIRVariable *a, const SPIRVariable *b) -> bool {
		// Sort input and output variables based on, from more robust to less robust:
		// - Location
		// - Variable has a location
		// - Name comparison
		// - Variable has a name
		// - Fallback: ID
		bool has_location_a = has_decoration(a->self, DecorationLocation);
		bool has_location_b = has_decoration(b->self, DecorationLocation);

		if (has_location_a && has_location_b)
		{
			return get_decoration(a->self, DecorationLocation) < get_decoration(b->self, DecorationLocation);
		}
		else if (has_location_a && !has_location_b)
			return true;
		else if (!has_location_a && has_location_b)
			return false;

		const auto &name1 = to_name(a->self);
		const auto &name2 = to_name(b->self);

		if (name1.empty() && name2.empty())
			return a->self < b->self;
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
		for (auto var : input_variables)
			emit_interface_block_in_struct(*var, active_inputs);
		emit_builtin_inputs_in_struct();
		end_scope_decl();
		statement("");
	}

	if (!output_variables.empty() || !active_output_builtins.empty())
	{
		require_output = true;
		statement("struct SPIRV_Cross_Output");

		begin_scope();
		// FIXME: Use locations properly if they exist.
		sort(output_variables.begin(), output_variables.end(), variable_compare);
		for (auto var : output_variables)
			emit_interface_block_in_struct(*var, active_outputs);
		emit_builtin_outputs_in_struct();
		end_scope_decl();
		statement("");
	}

	// Global variables.
	for (auto global : global_variables)
	{
		auto &var = get<SPIRVariable>(global);
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
				statement(storage, " ", variable_decl(var), ";");
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

	if (required_textureSizeVariants != 0)
	{
		static const char *types[QueryTypeCount] = { "float4", "int4", "uint4" };
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

				if ((required_textureSizeVariants & mask) == 0)
					continue;

				statement(ret_types[index], " SPIRV_Cross_textureSize(", dims[index], "<", types[type_index],
				          "> Tex, uint Level, out uint Param)");
				begin_scope();
				statement(ret_types[index], " ret;");
				switch (return_arguments[index])
				{
				case 1:
					if (has_lod[index])
						statement("Tex.GetDimensions(Level, ret.x, Param);");
					else
					{
						statement("Tex.GetDimensions(ret.x);");
						statement("Param = 0u;");
					}
					break;
				case 2:
					if (has_lod[index])
						statement("Tex.GetDimensions(Level, ret.x, ret.y, Param);");
					else
						statement("Tex.GetDimensions(ret.x, ret.y, Param);");
					break;
				case 3:
					if (has_lod[index])
						statement("Tex.GetDimensions(Level, ret.x, ret.y, ret.z, Param);");
					else
						statement("Tex.GetDimensions(ret.x, ret.y, ret.z, Param);");
					break;
				}

				statement("return ret;");
				end_scope();
				statement("");
			}
		}
	}

	if (requires_fp16_packing)
	{
		// HLSL does not pack into a single word sadly :(
		statement("uint SPIRV_Cross_packHalf2x16(float2 value)");
		begin_scope();
		statement("uint2 Packed = f32tof16(value);");
		statement("return Packed.x | (Packed.y << 16);");
		end_scope();
		statement("");

		statement("float2 SPIRV_Cross_unpackHalf2x16(uint value)");
		begin_scope();
		statement("return f16tof32(uint2(value & 0xffff, value >> 16));");
		end_scope();
		statement("");
	}

	if (requires_explicit_fp16_packing)
	{
		// HLSL does not pack into a single word sadly :(
		statement("uint SPIRV_Cross_packFloat2x16(min16float2 value)");
		begin_scope();
		statement("uint2 Packed = f32tof16(value);");
		statement("return Packed.x | (Packed.y << 16);");
		end_scope();
		statement("");

		statement("min16float2 SPIRV_Cross_unpackFloat2x16(uint value)");
		begin_scope();
		statement("return min16float2(f16tof32(uint2(value & 0xffff, value >> 16)));");
		end_scope();
		statement("");
	}

	// HLSL does not seem to have builtins for these operation, so roll them by hand ...
	if (requires_unorm8_packing)
	{
		statement("uint SPIRV_Cross_packUnorm4x8(float4 value)");
		begin_scope();
		statement("uint4 Packed = uint4(round(saturate(value) * 255.0));");
		statement("return Packed.x | (Packed.y << 8) | (Packed.z << 16) | (Packed.w << 24);");
		end_scope();
		statement("");

		statement("float4 SPIRV_Cross_unpackUnorm4x8(uint value)");
		begin_scope();
		statement("uint4 Packed = uint4(value & 0xff, (value >> 8) & 0xff, (value >> 16) & 0xff, value >> 24);");
		statement("return float4(Packed) / 255.0;");
		end_scope();
		statement("");
	}

	if (requires_snorm8_packing)
	{
		statement("uint SPIRV_Cross_packSnorm4x8(float4 value)");
		begin_scope();
		statement("int4 Packed = int4(round(clamp(value, -1.0, 1.0) * 127.0)) & 0xff;");
		statement("return uint(Packed.x | (Packed.y << 8) | (Packed.z << 16) | (Packed.w << 24));");
		end_scope();
		statement("");

		statement("float4 SPIRV_Cross_unpackSnorm4x8(uint value)");
		begin_scope();
		statement("int SignedValue = int(value);");
		statement("int4 Packed = int4(SignedValue << 24, SignedValue << 16, SignedValue << 8, SignedValue) >> 24;");
		statement("return clamp(float4(Packed) / 127.0, -1.0, 1.0);");
		end_scope();
		statement("");
	}

	if (requires_unorm16_packing)
	{
		statement("uint SPIRV_Cross_packUnorm2x16(float2 value)");
		begin_scope();
		statement("uint2 Packed = uint2(round(saturate(value) * 65535.0));");
		statement("return Packed.x | (Packed.y << 16);");
		end_scope();
		statement("");

		statement("float2 SPIRV_Cross_unpackUnorm2x16(uint value)");
		begin_scope();
		statement("uint2 Packed = uint2(value & 0xffff, value >> 16);");
		statement("return float2(Packed) / 65535.0;");
		end_scope();
		statement("");
	}

	if (requires_snorm16_packing)
	{
		statement("uint SPIRV_Cross_packSnorm2x16(float2 value)");
		begin_scope();
		statement("int2 Packed = int2(round(clamp(value, -1.0, 1.0) * 32767.0)) & 0xffff;");
		statement("return uint(Packed.x | (Packed.y << 16));");
		end_scope();
		statement("");

		statement("float2 SPIRV_Cross_unpackSnorm2x16(uint value)");
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
			statement(type, " SPIRV_Cross_bitfieldInsert(", type, " Base, ", type, " Insert, uint Offset, uint Count)");
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
			statement(type, " SPIRV_Cross_bitfieldUExtract(", type, " Base, uint Offset, uint Count)");
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
			statement(type, " SPIRV_Cross_bitfieldSExtract(", type, " Base, int Offset, int Count)");
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
		statement("float2x2 SPIRV_Cross_Inverse(float2x2 m)");
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
		statement("float SPIRV_Cross_Det2x2(float a1, float a2, float b1, float b2)");
		begin_scope();
		statement("return a1 * b2 - b1 * a2;");
		end_scope();
		statement_no_indent("");
		statement("// Returns the inverse of a matrix, by using the algorithm of calculating the classical");
		statement("// adjoint and dividing by the determinant. The contents of the matrix are changed.");
		statement("float3x3 SPIRV_Cross_Inverse(float3x3 m)");
		begin_scope();
		statement("float3x3 adj;	// The adjoint matrix (inverse after dividing by determinant)");
		statement_no_indent("");
		statement("// Create the transpose of the cofactors, as the classical adjoint of the matrix.");
		statement("adj[0][0] =  SPIRV_Cross_Det2x2(m[1][1], m[1][2], m[2][1], m[2][2]);");
		statement("adj[0][1] = -SPIRV_Cross_Det2x2(m[0][1], m[0][2], m[2][1], m[2][2]);");
		statement("adj[0][2] =  SPIRV_Cross_Det2x2(m[0][1], m[0][2], m[1][1], m[1][2]);");
		statement_no_indent("");
		statement("adj[1][0] = -SPIRV_Cross_Det2x2(m[1][0], m[1][2], m[2][0], m[2][2]);");
		statement("adj[1][1] =  SPIRV_Cross_Det2x2(m[0][0], m[0][2], m[2][0], m[2][2]);");
		statement("adj[1][2] = -SPIRV_Cross_Det2x2(m[0][0], m[0][2], m[1][0], m[1][2]);");
		statement_no_indent("");
		statement("adj[2][0] =  SPIRV_Cross_Det2x2(m[1][0], m[1][1], m[2][0], m[2][1]);");
		statement("adj[2][1] = -SPIRV_Cross_Det2x2(m[0][0], m[0][1], m[2][0], m[2][1]);");
		statement("adj[2][2] =  SPIRV_Cross_Det2x2(m[0][0], m[0][1], m[1][0], m[1][1]);");
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
			statement("float SPIRV_Cross_Det2x2(float a1, float a2, float b1, float b2)");
			begin_scope();
			statement("return a1 * b2 - b1 * a2;");
			end_scope();
			statement("");
		}

		statement("// Returns the determinant of a 3x3 matrix.");
		statement("float SPIRV_Cross_Det3x3(float a1, float a2, float a3, float b1, float b2, float b3, float c1, "
		          "float c2, float c3)");
		begin_scope();
		statement("return a1 * SPIRV_Cross_Det2x2(b2, b3, c2, c3) - b1 * SPIRV_Cross_Det2x2(a2, a3, c2, c3) + c1 * "
		          "SPIRV_Cross_Det2x2(a2, a3, "
		          "b2, b3);");
		end_scope();
		statement_no_indent("");
		statement("// Returns the inverse of a matrix, by using the algorithm of calculating the classical");
		statement("// adjoint and dividing by the determinant. The contents of the matrix are changed.");
		statement("float4x4 SPIRV_Cross_Inverse(float4x4 m)");
		begin_scope();
		statement("float4x4 adj;	// The adjoint matrix (inverse after dividing by determinant)");
		statement_no_indent("");
		statement("// Create the transpose of the cofactors, as the classical adjoint of the matrix.");
		statement(
		    "adj[0][0] =  SPIRV_Cross_Det3x3(m[1][1], m[1][2], m[1][3], m[2][1], m[2][2], m[2][3], m[3][1], m[3][2], "
		    "m[3][3]);");
		statement(
		    "adj[0][1] = -SPIRV_Cross_Det3x3(m[0][1], m[0][2], m[0][3], m[2][1], m[2][2], m[2][3], m[3][1], m[3][2], "
		    "m[3][3]);");
		statement(
		    "adj[0][2] =  SPIRV_Cross_Det3x3(m[0][1], m[0][2], m[0][3], m[1][1], m[1][2], m[1][3], m[3][1], m[3][2], "
		    "m[3][3]);");
		statement(
		    "adj[0][3] = -SPIRV_Cross_Det3x3(m[0][1], m[0][2], m[0][3], m[1][1], m[1][2], m[1][3], m[2][1], m[2][2], "
		    "m[2][3]);");
		statement_no_indent("");
		statement(
		    "adj[1][0] = -SPIRV_Cross_Det3x3(m[1][0], m[1][2], m[1][3], m[2][0], m[2][2], m[2][3], m[3][0], m[3][2], "
		    "m[3][3]);");
		statement(
		    "adj[1][1] =  SPIRV_Cross_Det3x3(m[0][0], m[0][2], m[0][3], m[2][0], m[2][2], m[2][3], m[3][0], m[3][2], "
		    "m[3][3]);");
		statement(
		    "adj[1][2] = -SPIRV_Cross_Det3x3(m[0][0], m[0][2], m[0][3], m[1][0], m[1][2], m[1][3], m[3][0], m[3][2], "
		    "m[3][3]);");
		statement(
		    "adj[1][3] =  SPIRV_Cross_Det3x3(m[0][0], m[0][2], m[0][3], m[1][0], m[1][2], m[1][3], m[2][0], m[2][2], "
		    "m[2][3]);");
		statement_no_indent("");
		statement(
		    "adj[2][0] =  SPIRV_Cross_Det3x3(m[1][0], m[1][1], m[1][3], m[2][0], m[2][1], m[2][3], m[3][0], m[3][1], "
		    "m[3][3]);");
		statement(
		    "adj[2][1] = -SPIRV_Cross_Det3x3(m[0][0], m[0][1], m[0][3], m[2][0], m[2][1], m[2][3], m[3][0], m[3][1], "
		    "m[3][3]);");
		statement(
		    "adj[2][2] =  SPIRV_Cross_Det3x3(m[0][0], m[0][1], m[0][3], m[1][0], m[1][1], m[1][3], m[3][0], m[3][1], "
		    "m[3][3]);");
		statement(
		    "adj[2][3] = -SPIRV_Cross_Det3x3(m[0][0], m[0][1], m[0][3], m[1][0], m[1][1], m[1][3], m[2][0], m[2][1], "
		    "m[2][3]);");
		statement_no_indent("");
		statement(
		    "adj[3][0] = -SPIRV_Cross_Det3x3(m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2], m[3][0], m[3][1], "
		    "m[3][2]);");
		statement(
		    "adj[3][1] =  SPIRV_Cross_Det3x3(m[0][0], m[0][1], m[0][2], m[2][0], m[2][1], m[2][2], m[3][0], m[3][1], "
		    "m[3][2]);");
		statement(
		    "adj[3][2] = -SPIRV_Cross_Det3x3(m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[3][0], m[3][1], "
		    "m[3][2]);");
		statement(
		    "adj[3][3] =  SPIRV_Cross_Det3x3(m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], "
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

	string qualifiers;
	bool is_block = ir.meta[type.self].decoration.decoration_flags.get(DecorationBlock) ||
	                ir.meta[type.self].decoration.decoration_flags.get(DecorationBufferBlock);

	if (is_block)
		qualifiers = to_interpolation_qualifiers(memberflags);

	string packing_offset;
	bool is_push_constant = type.storage == StorageClassPushConstant;

	if ((has_extended_decoration(type.self, SPIRVCrossDecorationPacked) || is_push_constant) &&
	    has_member_decoration(type.self, index, DecorationOffset))
	{
		uint32_t offset = memb[index].offset - base_offset;
		if (offset & 3)
			SPIRV_CROSS_THROW("Cannot pack on tighter bounds than 4 bytes in HLSL.");

		static const char *packing_swizzle[] = { "", ".y", ".z", ".w" };
		packing_offset = join(" : packoffset(c", offset / 16, packing_swizzle[(offset & 15) >> 2], ")");
	}

	statement(layout_for_member(type, index), qualifiers, qualifier,
	          variable_decl(membertype, to_member_name(type, index)), packing_offset, ";");
}

void CompilerHLSL::emit_buffer_block(const SPIRVariable &var)
{
	auto &type = get<SPIRType>(var.basetype);

	bool is_uav = var.storage == StorageClassStorageBuffer || has_decoration(type.self, DecorationBufferBlock);

	if (is_uav)
	{
		Bitset flags = ir.get_buffer_block_flags(var);
		bool is_readonly = flags.get(DecorationNonWritable);
		bool is_coherent = flags.get(DecorationCoherent);
		add_resource_name(var.self);
		statement(is_coherent ? "globallycoherent " : "", is_readonly ? "ByteAddressBuffer " : "RWByteAddressBuffer ",
		          to_name(var.self), type_to_array_glsl(type), to_resource_binding(var), ";");
	}
	else
	{
		if (type.array.empty())
		{
			if (buffer_is_packing_standard(type, BufferPackingHLSLCbufferPackOffset))
				set_extended_decoration(type.self, SPIRVCrossDecorationPacked);
			else
				SPIRV_CROSS_THROW("cbuffer cannot be expressed with either HLSL packing layout or packoffset.");

			// Flatten the top-level struct so we can use packoffset,
			// this restriction is similar to GLSL where layout(offset) is not possible on sub-structs.
			flattened_structs.insert(var.self);

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
				set_member_name(type.self, i, sanitize_underscores(join(to_name(var.self), "_", member_name)));
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

			// ConstantBuffer<T> does not support packoffset, so it is unuseable unless everything aligns as we expect.
			if (!buffer_is_packing_standard(type, BufferPackingHLSLCbuffer))
				SPIRV_CROSS_THROW("HLSL ConstantBuffer<T> cannot be expressed with normal HLSL packing rules.");

			add_resource_name(type.self);
			add_resource_name(var.self);

			emit_struct(get<SPIRType>(type.self));
			statement("ConstantBuffer<", to_name(type.self), "> ", to_name(var.self), type_to_array_glsl(type),
			          to_resource_binding(var), ";");
		}
	}
}

void CompilerHLSL::emit_push_constant_block(const SPIRVariable &var)
{
	if (root_constants_layout.empty())
	{
		emit_buffer_block(var);
	}
	else
	{
		for (const auto &layout : root_constants_layout)
		{
			auto &type = get<SPIRType>(var.basetype);

			if (buffer_is_packing_standard(type, BufferPackingHLSLCbufferPackOffset, layout.start, layout.end))
				set_extended_decoration(type.self, SPIRVCrossDecorationPacked);
			else
				SPIRV_CROSS_THROW(
				    "root constant cbuffer cannot be expressed with either HLSL packing layout or packoffset.");

			flattened_structs.insert(var.self);
			type.member_name_cache.clear();
			add_resource_name(var.self);
			auto &memb = ir.meta[type.self].members;

			statement("cbuffer SPIRV_CROSS_RootConstant_", to_name(var.self),
			          to_resource_register('b', layout.binding, layout.space));
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
					set_member_name(type.self, constant_index,
					                sanitize_underscores(join(to_name(var.self), "_", member_name)));
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
	auto expr = join("_", to_expression(id));
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

string CompilerHLSL::to_func_call_arg(uint32_t id)
{
	string arg_str = CompilerGLSL::to_func_call_arg(id);

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

void CompilerHLSL::emit_function_prototype(SPIRFunction &func, const Bitset &return_flags)
{
	if (func.self != ir.default_entry_point)
		add_function_overload(func);

	auto &execution = get_entry_point();
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
		if (execution.model == ExecutionModelVertex)
			decl += "vert_main";
		else if (execution.model == ExecutionModelFragment)
			decl += "frag_main";
		else if (execution.model == ExecutionModelGLCompute)
			decl += "comp_main";
		else
			SPIRV_CROSS_THROW("Unsupported execution model.");
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
		out_argument += "SPIRV_Cross_return_value";
		out_argument += type_to_array_glsl(type);
		arglist.push_back(move(out_argument));
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
			arglist.push_back(join(image_is_comparison(arg_type, arg.id) ? "SamplerComparisonState " : "SamplerState ",
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

	// Add I/O blocks as separate arguments with appropriate storage qualifier.
	ir.for_each_typed_id<SPIRVariable>([&](uint32_t, SPIRVariable &var) {
		auto &type = this->get<SPIRType>(var.basetype);
		bool block = ir.meta[type.self].decoration.decoration_flags.get(DecorationBlock);

		if (var.storage != StorageClassInput && var.storage != StorageClassOutput)
			return;

		if (block && !is_builtin_variable(var) && interface_variable_exists_in_entry_point(var.self))
		{
			if (var.storage == StorageClassInput)
			{
				arguments.push_back(join("in ", variable_decl(type, join("stage_input", to_name(var.self)))));
			}
			else if (var.storage == StorageClassOutput)
			{
				arguments.push_back(join("out ", variable_decl(type, join("stage_output", to_name(var.self)))));
			}
		}
	});

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

	statement(require_output ? "SPIRV_Cross_Output " : "void ", "main(", merge(arguments), ")");
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
				statement(builtin, " = stage_input.", builtin, ";");
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
		bool block = ir.meta[type.self].decoration.decoration_flags.get(DecorationBlock);

		if (var.storage != StorageClassInput)
			return;

		bool need_matrix_unroll = var.storage == StorageClassInput && execution.model == ExecutionModelVertex;

		if (!block && !var.remapped_variable && type.pointer && !is_builtin_variable(var) &&
		    interface_variable_exists_in_entry_point(var.self))
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

		// I/O blocks don't use the common stage input/output struct, but separate outputs.
		if (block && !is_builtin_variable(var) && interface_variable_exists_in_entry_point(var.self))
		{
			auto name = to_name(var.self);
			statement(name, " = stage_input", name, ";");
		}
	});

	// Run the shader.
	if (execution.model == ExecutionModelVertex)
		statement("vert_main();");
	else if (execution.model == ExecutionModelFragment)
		statement("frag_main();");
	else if (execution.model == ExecutionModelGLCompute)
		statement("comp_main();");
	else
		SPIRV_CROSS_THROW("Unsupported shader stage.");

	// Copy block outputs.
	ir.for_each_typed_id<SPIRVariable>([&](uint32_t, SPIRVariable &var) {
		auto &type = this->get<SPIRType>(var.basetype);
		bool block = ir.meta[type.self].decoration.decoration_flags.get(DecorationBlock);

		if (var.storage != StorageClassOutput)
			return;

		// I/O blocks don't use the common stage input/output struct, but separate outputs.
		if (block && !is_builtin_variable(var) && interface_variable_exists_in_entry_point(var.self))
		{
			auto name = to_name(var.self);
			statement("stage_output", name, " = ", name, ";");
		}
	});

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
			bool block = ir.meta[type.self].decoration.decoration_flags.get(DecorationBlock);

			if (var.storage != StorageClassOutput)
				return;

			if (!block && var.storage != StorageClassFunction && !var.remapped_variable && type.pointer &&
			    !is_builtin_variable(var) && interface_variable_exists_in_entry_point(var.self))
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
		});

		statement("return stage_output;");
	}

	end_scope();
}

void CompilerHLSL::emit_fixup()
{
	if (get_entry_point().model == ExecutionModelVertex)
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

void CompilerHLSL::emit_texture_op(const Instruction &i)
{
	auto *ops = stream(i);
	auto op = static_cast<Op>(i.op);
	uint32_t length = i.length;

	SmallVector<uint32_t> inherited_expressions;

	uint32_t result_type = ops[0];
	uint32_t id = ops[1];
	uint32_t img = ops[2];
	uint32_t coord = ops[3];
	uint32_t dref = 0;
	uint32_t comp = 0;
	bool gather = false;
	bool proj = false;
	const uint32_t *opt = nullptr;
	auto *combined_image = maybe_get<SPIRCombinedImageSampler>(img);
	auto img_expr = to_expression(combined_image ? combined_image->image : img);

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

	string expr;
	string texop;

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

			if (image_is_comparison(imgtype, img))
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
				uint32_t comp_num = get<SPIRConstant>(comp).scalar();
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
			if (proj)
				texop += "proj";
			if (grad_x || grad_y)
				texop += "grad";
			if (lod)
				texop += "lod";
			if (bias)
				texop += "bias";
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
			sampler_expr = to_expression(combined_image->sampler);
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

	if (hlsl_options.shader_model < 40 && lod)
	{
		string coord_filler;
		for (uint32_t size = coord_components; size < 3; ++size)
		{
			coord_filler += ", 0.0";
		}
		coord_expr = "float4(" + coord_expr + coord_filler + ", " + to_expression(lod) + ")";
	}

	if (hlsl_options.shader_model < 40 && bias)
	{
		string coord_filler;
		for (uint32_t size = coord_components; size < 3; ++size)
		{
			coord_filler += ", 0.0";
		}
		coord_expr = "float4(" + coord_expr + coord_filler + ", " + to_expression(bias) + ")";
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

	if (dref)
	{
		if (hlsl_options.shader_model < 40)
			SPIRV_CROSS_THROW("Legacy HLSL does not support comparison sampling.");

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

	if (op == OpImageQueryLod)
	{
		// This is rather awkward.
		// textureQueryLod returns two values, the "accessed level",
		// as well as the actual LOD lambda.
		// As far as I can tell, there is no way to get the .x component
		// according to GLSL spec, and it depends on the sampler itself.
		// Just assume X == Y, so we will need to splat the result to a float2.
		statement("float _", id, "_tmp = ", expr, ";");
		emit_op(result_type, id, join("float2(_", id, "_tmp, _", id, "_tmp)"), true, true);
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
	case OpImageQueryLod:
		register_control_dependent_expression(id);
		break;

	default:
		break;
	}
}

string CompilerHLSL::to_resource_binding(const SPIRVariable &var)
{
	// TODO: Basic implementation, might need special consideration for RW/RO structured buffers,
	// RW/RO images, and so on.

	if (!has_decoration(var.self, DecorationBinding))
		return "";

	const auto &type = get<SPIRType>(var.basetype);
	char space = '\0';

	switch (type.basetype)
	{
	case SPIRType::SampledImage:
		space = 't'; // SRV
		break;

	case SPIRType::Image:
		if (type.image.sampled == 2 && type.image.dim != DimSubpassData)
			space = 'u'; // UAV
		else
			space = 't'; // SRV
		break;

	case SPIRType::Sampler:
		space = 's';
		break;

	case SPIRType::Struct:
	{
		auto storage = type.storage;
		if (storage == StorageClassUniform)
		{
			if (has_decoration(type.self, DecorationBufferBlock))
			{
				Bitset flags = ir.get_buffer_block_flags(var);
				bool is_readonly = flags.get(DecorationNonWritable);
				space = is_readonly ? 't' : 'u'; // UAV
			}
			else if (has_decoration(type.self, DecorationBlock))
				space = 'b'; // Constant buffers
		}
		else if (storage == StorageClassPushConstant)
			space = 'b'; // Constant buffers
		else if (storage == StorageClassStorageBuffer)
		{
			// UAV or SRV depending on readonly flag.
			Bitset flags = ir.get_buffer_block_flags(var);
			bool is_readonly = flags.get(DecorationNonWritable);
			space = is_readonly ? 't' : 'u';
		}

		break;
	}
	default:
		break;
	}

	if (!space)
		return "";

	return to_resource_register(space, get_decoration(var.self, DecorationBinding),
	                            get_decoration(var.self, DecorationDescriptorSet));
}

string CompilerHLSL::to_resource_binding_sampler(const SPIRVariable &var)
{
	// For combined image samplers.
	if (!has_decoration(var.self, DecorationBinding))
		return "";

	return to_resource_register('s', get_decoration(var.self, DecorationBinding),
	                            get_decoration(var.self, DecorationDescriptorSet));
}

string CompilerHLSL::to_resource_register(char space, uint32_t binding, uint32_t space_set)
{
	if (hlsl_options.shader_model >= 51)
		return join(" : register(", space, binding, ", space", space_set, ")");
	else
		return join(" : register(", space, binding, ")");
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
			if (image_is_comparison(type, var.self))
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
		return "SPIRV_Cross_unpackFloat2x16";
	}
	else if (out_type.basetype == SPIRType::UInt && in_type.basetype == SPIRType::Half && in_type.vecsize == 2)
	{
		if (!requires_explicit_fp16_packing)
		{
			requires_explicit_fp16_packing = true;
			force_recompile();
		}
		return "SPIRV_Cross_packFloat2x16";
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

	switch (op)
	{
	case GLSLstd450InverseSqrt:
		emit_unary_func_op(result_type, id, args[0], "rsqrt");
		break;

	case GLSLstd450Fract:
		emit_unary_func_op(result_type, id, args[0], "frac");
		break;

	case GLSLstd450RoundEven:
		SPIRV_CROSS_THROW("roundEven is not supported on HLSL.");

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
		emit_unary_func_op(result_type, id, args[0], "SPIRV_Cross_packHalf2x16");
		break;

	case GLSLstd450UnpackHalf2x16:
		if (!requires_fp16_packing)
		{
			requires_fp16_packing = true;
			force_recompile();
		}
		emit_unary_func_op(result_type, id, args[0], "SPIRV_Cross_unpackHalf2x16");
		break;

	case GLSLstd450PackSnorm4x8:
		if (!requires_snorm8_packing)
		{
			requires_snorm8_packing = true;
			force_recompile();
		}
		emit_unary_func_op(result_type, id, args[0], "SPIRV_Cross_packSnorm4x8");
		break;

	case GLSLstd450UnpackSnorm4x8:
		if (!requires_snorm8_packing)
		{
			requires_snorm8_packing = true;
			force_recompile();
		}
		emit_unary_func_op(result_type, id, args[0], "SPIRV_Cross_unpackSnorm4x8");
		break;

	case GLSLstd450PackUnorm4x8:
		if (!requires_unorm8_packing)
		{
			requires_unorm8_packing = true;
			force_recompile();
		}
		emit_unary_func_op(result_type, id, args[0], "SPIRV_Cross_packUnorm4x8");
		break;

	case GLSLstd450UnpackUnorm4x8:
		if (!requires_unorm8_packing)
		{
			requires_unorm8_packing = true;
			force_recompile();
		}
		emit_unary_func_op(result_type, id, args[0], "SPIRV_Cross_unpackUnorm4x8");
		break;

	case GLSLstd450PackSnorm2x16:
		if (!requires_snorm16_packing)
		{
			requires_snorm16_packing = true;
			force_recompile();
		}
		emit_unary_func_op(result_type, id, args[0], "SPIRV_Cross_packSnorm2x16");
		break;

	case GLSLstd450UnpackSnorm2x16:
		if (!requires_snorm16_packing)
		{
			requires_snorm16_packing = true;
			force_recompile();
		}
		emit_unary_func_op(result_type, id, args[0], "SPIRV_Cross_unpackSnorm2x16");
		break;

	case GLSLstd450PackUnorm2x16:
		if (!requires_unorm16_packing)
		{
			requires_unorm16_packing = true;
			force_recompile();
		}
		emit_unary_func_op(result_type, id, args[0], "SPIRV_Cross_packUnorm2x16");
		break;

	case GLSLstd450UnpackUnorm2x16:
		if (!requires_unorm16_packing)
		{
			requires_unorm16_packing = true;
			force_recompile();
		}
		emit_unary_func_op(result_type, id, args[0], "SPIRV_Cross_unpackUnorm2x16");
		break;

	case GLSLstd450PackDouble2x32:
	case GLSLstd450UnpackDouble2x32:
		SPIRV_CROSS_THROW("packDouble2x32/unpackDouble2x32 not supported in HLSL.");

	case GLSLstd450FindILsb:
		emit_unary_func_op(result_type, id, args[0], "firstbitlow");
		break;

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
		emit_unary_func_op(result_type, id, args[0], "SPIRV_Cross_Inverse");
		break;
	}

	default:
		CompilerGLSL::emit_glsl_op(result_type, id, eop, args, count);
		break;
	}
}

string CompilerHLSL::read_access_chain(const SPIRAccessChain &chain)
{
	auto &type = get<SPIRType>(chain.basetype);

	SPIRType target_type;
	target_type.basetype = SPIRType::UInt;
	target_type.vecsize = type.vecsize;
	target_type.columns = type.columns;

	if (type.basetype == SPIRType::Struct)
		SPIRV_CROSS_THROW("Reading structs from ByteAddressBuffer not yet supported.");

	if (type.width != 32)
		SPIRV_CROSS_THROW("Reading types other than 32-bit from ByteAddressBuffer not yet supported.");

	if (!type.array.empty())
		SPIRV_CROSS_THROW("Reading arrays from ByteAddressBuffer not yet supported.");

	string load_expr;

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

		load_expr = join(chain.base, ".", load_op, "(", chain.dynamic_index, chain.static_index, ")");
	}
	else if (type.columns == 1)
	{
		// Strided load since we are loading a column from a row-major matrix.
		if (type.vecsize > 1)
		{
			load_expr = type_to_glsl(target_type);
			load_expr += "(";
		}

		for (uint32_t r = 0; r < type.vecsize; r++)
		{
			load_expr +=
			    join(chain.base, ".Load(", chain.dynamic_index, chain.static_index + r * chain.matrix_stride, ")");
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

		// Note, this loading style in HLSL is *actually* row-major, but we always treat matrices as transposed in this backend,
		// so row-major is technically column-major ...
		load_expr = type_to_glsl(target_type);
		load_expr += "(";
		for (uint32_t c = 0; c < type.columns; c++)
		{
			load_expr += join(chain.base, ".", load_op, "(", chain.dynamic_index,
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

		load_expr = type_to_glsl(target_type);
		load_expr += "(";
		for (uint32_t c = 0; c < type.columns; c++)
		{
			for (uint32_t r = 0; r < type.vecsize; r++)
			{
				load_expr += join(chain.base, ".Load(", chain.dynamic_index,
				                  chain.static_index + c * (type.width / 8) + r * chain.matrix_stride, ")");

				if ((r + 1 < type.vecsize) || (c + 1 < type.columns))
					load_expr += ", ";
			}
		}
		load_expr += ")";
	}

	auto bitcast_op = bitcast_glsl_op(type, target_type);
	if (!bitcast_op.empty())
		load_expr = join(bitcast_op, "(", load_expr, ")");

	return load_expr;
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

		auto load_expr = read_access_chain(*chain);

		bool forward = should_forward(ptr) && forced_temporaries.find(id) == end(forced_temporaries);

		// If we are forwarding this load,
		// don't register the read to access chain here, defer that to when we actually use the expression,
		// using the add_implied_read_expression mechanism.
		if (!forward)
			track_expression_read(chain->self);

		// Do not forward complex load sequences like matrices, structs and arrays.
		auto &type = get<SPIRType>(result_type);
		if (type.columns > 1 || !type.array.empty() || type.basetype == SPIRType::Struct)
			forward = false;

		auto &e = emit_op(result_type, id, load_expr, forward, true);
		e.need_transpose = false;
		register_read(id, ptr, forward);
		inherit_expression_dependencies(id, ptr);
		if (forward)
			add_implied_read_expression(e, chain->self);
	}
	else
		CompilerGLSL::emit_instruction(instruction);
}

void CompilerHLSL::write_access_chain(const SPIRAccessChain &chain, uint32_t value)
{
	auto &type = get<SPIRType>(chain.basetype);

	// Make sure we trigger a read of the constituents in the access chain.
	track_expression_read(chain.self);

	SPIRType target_type;
	target_type.basetype = SPIRType::UInt;
	target_type.vecsize = type.vecsize;
	target_type.columns = type.columns;

	if (type.basetype == SPIRType::Struct)
		SPIRV_CROSS_THROW("Writing structs to RWByteAddressBuffer not yet supported.");
	if (type.width != 32)
		SPIRV_CROSS_THROW("Writing types other than 32-bit to RWByteAddressBuffer not yet supported.");
	if (!type.array.empty())
		SPIRV_CROSS_THROW("Reading arrays from ByteAddressBuffer not yet supported.");

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

		auto store_expr = to_expression(value);
		auto bitcast_op = bitcast_glsl_op(target_type, type);
		if (!bitcast_op.empty())
			store_expr = join(bitcast_op, "(", store_expr, ")");
		statement(chain.base, ".", store_op, "(", chain.dynamic_index, chain.static_index, ", ", store_expr, ");");
	}
	else if (type.columns == 1)
	{
		// Strided store.
		for (uint32_t r = 0; r < type.vecsize; r++)
		{
			auto store_expr = to_enclosed_expression(value);
			if (type.vecsize > 1)
			{
				store_expr += ".";
				store_expr += index_to_swizzle(r);
			}
			remove_duplicate_swizzle(store_expr);

			auto bitcast_op = bitcast_glsl_op(target_type, type);
			if (!bitcast_op.empty())
				store_expr = join(bitcast_op, "(", store_expr, ")");
			statement(chain.base, ".Store(", chain.dynamic_index, chain.static_index + chain.matrix_stride * r, ", ",
			          store_expr, ");");
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

		for (uint32_t c = 0; c < type.columns; c++)
		{
			auto store_expr = join(to_enclosed_expression(value), "[", c, "]");
			auto bitcast_op = bitcast_glsl_op(target_type, type);
			if (!bitcast_op.empty())
				store_expr = join(bitcast_op, "(", store_expr, ")");
			statement(chain.base, ".", store_op, "(", chain.dynamic_index, chain.static_index + c * chain.matrix_stride,
			          ", ", store_expr, ");");
		}
	}
	else
	{
		for (uint32_t r = 0; r < type.vecsize; r++)
		{
			for (uint32_t c = 0; c < type.columns; c++)
			{
				auto store_expr = join(to_enclosed_expression(value), "[", c, "].", index_to_swizzle(r));
				remove_duplicate_swizzle(store_expr);
				auto bitcast_op = bitcast_glsl_op(target_type, type);
				if (!bitcast_op.empty())
					store_expr = join(bitcast_op, "(", store_expr, ")");
				statement(chain.base, ".Store(", chain.dynamic_index,
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
		write_access_chain(*chain, ops[1]);
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
		uint32_t to_plain_buffer_length = static_cast<uint32_t>(type.array.size());
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
		bool row_major_matrix = false;

		// Inherit matrix information.
		if (chain)
		{
			matrix_stride = chain->matrix_stride;
			row_major_matrix = chain->row_major_matrix;
		}

		auto offsets =
		    flattened_access_chain_offset(*basetype, &ops[3 + to_plain_buffer_length],
		                                  length - 3 - to_plain_buffer_length, 0, 1, &row_major_matrix, &matrix_stride);

		auto &e = set<SPIRAccessChain>(ops[1], ops[0], type.storage, base, offsets.first, offsets.second);
		e.row_major_matrix = row_major_matrix;
		e.matrix_stride = matrix_stride;
		e.immutable = should_forward(ops[2]);
		e.loaded_from = backing_variable ? backing_variable->self : 0;

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
	if (op != OpAtomicIDecrement && op != OpAtomicIIncrement)
		value_expr = to_expression(ops[op == OpAtomicCompareExchange ? 6 : 5]);

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

	case OpAtomicCompareExchange:
		if (length < 8)
			SPIRV_CROSS_THROW("Not enough data for opcode.");
		atomic_op = "InterlockedCompareExchange";
		value_expr = join(to_expression(ops[7]), ", ", value_expr);
		break;

	default:
		SPIRV_CROSS_THROW("Unknown atomic opcode.");
	}

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
		statement(atomic_op, "(", to_expression(ops[2]), ", ", value_expr, ", ", to_name(id), ");");
		expr_type = data_type.basetype;
	}
	else
	{
		// RWByteAddress buffer is always uint in its underlying type.
		expr_type = SPIRType::UInt;
		statement(chain->base, ".", atomic_op, "(", chain->dynamic_index, chain->static_index, ", ", value_expr, ", ",
		          to_name(id), ");");
	}

	auto expr = bitcast_expression(type, expr_type, to_name(id));
	set<SPIRExpression>(id, expr, result_type, true);
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

	auto scope = static_cast<Scope>(get<SPIRConstant>(ops[2]).scalar());
	if (scope != ScopeSubgroup)
		SPIRV_CROSS_THROW("Only subgroup scope is supported.");

	const auto make_inclusive_Sum = [&](const string &expr) -> string {
		return join(expr, " + ", to_expression(ops[4]));
	};

	const auto make_inclusive_Product = [&](const string &expr) -> string {
		return join(expr, " * ", to_expression(ops[4]));
	};

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
		break;

	case OpGroupNonUniformBallotBitExtract:
		SPIRV_CROSS_THROW("Cannot trivially implement BallotBitExtract in HLSL.");
		break;

	case OpGroupNonUniformBallotFindLSB:
		SPIRV_CROSS_THROW("Cannot trivially implement BallotFindLSB in HLSL.");
		break;

	case OpGroupNonUniformBallotFindMSB:
		SPIRV_CROSS_THROW("Cannot trivially implement BallotFindMSB in HLSL.");
		break;

	case OpGroupNonUniformBallotBitCount:
	{
		auto operation = static_cast<GroupOperation>(ops[3]);
		if (operation == GroupOperationReduce)
		{
			bool forward = should_forward(ops[4]);
			auto left = join("countbits(", to_enclosed_expression(ops[4]), ".x) + countbits(",
			                 to_enclosed_expression(ops[4]), ".y)");
			auto right = join("countbits(", to_enclosed_expression(ops[4]), ".z) + countbits(",
			                  to_enclosed_expression(ops[4]), ".w)");
			emit_op(result_type, id, join(left, " + ", right), forward);
			inherit_expression_dependencies(id, ops[4]);
		}
		else if (operation == GroupOperationInclusiveScan)
			SPIRV_CROSS_THROW("Cannot trivially implement BallotBitCount Inclusive Scan in HLSL.");
		else if (operation == GroupOperationExclusiveScan)
			SPIRV_CROSS_THROW("Cannot trivially implement BallotBitCount Exclusive Scan in HLSL.");
		else
			SPIRV_CROSS_THROW("Invalid BitCount operation.");
		break;
	}

	case OpGroupNonUniformShuffle:
		SPIRV_CROSS_THROW("Cannot trivially implement Shuffle in HLSL.");
	case OpGroupNonUniformShuffleXor:
		SPIRV_CROSS_THROW("Cannot trivially implement ShuffleXor in HLSL.");
	case OpGroupNonUniformShuffleUp:
		SPIRV_CROSS_THROW("Cannot trivially implement ShuffleUp in HLSL.");
	case OpGroupNonUniformShuffleDown:
		SPIRV_CROSS_THROW("Cannot trivially implement ShuffleDown in HLSL.");

	case OpGroupNonUniformAll:
		emit_unary_func_op(result_type, id, ops[3], "WaveActiveAllTrue");
		break;

	case OpGroupNonUniformAny:
		emit_unary_func_op(result_type, id, ops[3], "WaveActiveAnyTrue");
		break;

	case OpGroupNonUniformAllEqual:
	{
		auto &type = get<SPIRType>(result_type);
		emit_unary_func_op(result_type, id, ops[3],
		                   type.basetype == SPIRType::Boolean ? "WaveActiveAllEqualBool" : "WaveActiveAllEqual");
		break;
	}

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
	HLSL_GROUP_OP(FAdd, Sum, true)
	HLSL_GROUP_OP(FMul, Product, true)
	HLSL_GROUP_OP(FMin, Min, false)
	HLSL_GROUP_OP(FMax, Max, false)
	HLSL_GROUP_OP(IAdd, Sum, true)
	HLSL_GROUP_OP(IMul, Product, true)
	HLSL_GROUP_OP(SMin, Min, false)
	HLSL_GROUP_OP(SMax, Max, false)
	HLSL_GROUP_OP(UMin, Min, false)
	HLSL_GROUP_OP(UMax, Max, false)
	HLSL_GROUP_OP(BitwiseAnd, BitAnd, false)
	HLSL_GROUP_OP(BitwiseOr, BitOr, false)
	HLSL_GROUP_OP(BitwiseXor, BitXor, false)
#undef HLSL_GROUP_OP
		// clang-format on

	case OpGroupNonUniformQuadSwap:
	{
		uint32_t direction = get<SPIRConstant>(ops[4]).scalar();
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

	switch (opcode)
	{
	case OpAccessChain:
	case OpInBoundsAccessChain:
	{
		emit_access_chain(instruction);
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
		emit_binary_func_op(ops[0], ops[1], ops[3], ops[2], "mul");
		break;
	}

	case OpVectorTimesMatrix:
	{
		emit_binary_func_op(ops[0], ops[1], ops[3], ops[2], "mul");
		break;
	}

	case OpMatrixTimesMatrix:
	{
		emit_binary_func_op(ops[0], ops[1], ops[3], ops[2], "mul");
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
			emit_unrolled_binary_op(result_type, id, ops[2], ops[3], "==");
		else
			HLSL_BOP_CAST(==, int_type);
		break;
	}

	case OpLogicalEqual:
	case OpFOrdEqual:
	{
		auto result_type = ops[0];
		auto id = ops[1];

		if (expression_type(ops[2]).vecsize > 1)
			emit_unrolled_binary_op(result_type, id, ops[2], ops[3], "==");
		else
			HLSL_BOP(==);
		break;
	}

	case OpINotEqual:
	{
		auto result_type = ops[0];
		auto id = ops[1];

		if (expression_type(ops[2]).vecsize > 1)
			emit_unrolled_binary_op(result_type, id, ops[2], ops[3], "!=");
		else
			HLSL_BOP_CAST(!=, int_type);
		break;
	}

	case OpLogicalNotEqual:
	case OpFOrdNotEqual:
	{
		auto result_type = ops[0];
		auto id = ops[1];

		if (expression_type(ops[2]).vecsize > 1)
			emit_unrolled_binary_op(result_type, id, ops[2], ops[3], "!=");
		else
			HLSL_BOP(!=);
		break;
	}

	case OpUGreaterThan:
	case OpSGreaterThan:
	{
		auto result_type = ops[0];
		auto id = ops[1];
		auto type = opcode == OpUGreaterThan ? SPIRType::UInt : SPIRType::Int;

		if (expression_type(ops[2]).vecsize > 1)
			emit_unrolled_binary_op(result_type, id, ops[2], ops[3], ">");
		else
			HLSL_BOP_CAST(>, type);
		break;
	}

	case OpFOrdGreaterThan:
	{
		auto result_type = ops[0];
		auto id = ops[1];

		if (expression_type(ops[2]).vecsize > 1)
			emit_unrolled_binary_op(result_type, id, ops[2], ops[3], ">");
		else
			HLSL_BOP(>);
		break;
	}

	case OpUGreaterThanEqual:
	case OpSGreaterThanEqual:
	{
		auto result_type = ops[0];
		auto id = ops[1];

		auto type = opcode == OpUGreaterThanEqual ? SPIRType::UInt : SPIRType::Int;
		if (expression_type(ops[2]).vecsize > 1)
			emit_unrolled_binary_op(result_type, id, ops[2], ops[3], ">=");
		else
			HLSL_BOP_CAST(>=, type);
		break;
	}

	case OpFOrdGreaterThanEqual:
	{
		auto result_type = ops[0];
		auto id = ops[1];

		if (expression_type(ops[2]).vecsize > 1)
			emit_unrolled_binary_op(result_type, id, ops[2], ops[3], ">=");
		else
			HLSL_BOP(>=);
		break;
	}

	case OpULessThan:
	case OpSLessThan:
	{
		auto result_type = ops[0];
		auto id = ops[1];

		auto type = opcode == OpULessThan ? SPIRType::UInt : SPIRType::Int;
		if (expression_type(ops[2]).vecsize > 1)
			emit_unrolled_binary_op(result_type, id, ops[2], ops[3], "<");
		else
			HLSL_BOP_CAST(<, type);
		break;
	}

	case OpFOrdLessThan:
	{
		auto result_type = ops[0];
		auto id = ops[1];

		if (expression_type(ops[2]).vecsize > 1)
			emit_unrolled_binary_op(result_type, id, ops[2], ops[3], "<");
		else
			HLSL_BOP(<);
		break;
	}

	case OpULessThanEqual:
	case OpSLessThanEqual:
	{
		auto result_type = ops[0];
		auto id = ops[1];

		auto type = opcode == OpULessThanEqual ? SPIRType::UInt : SPIRType::Int;
		if (expression_type(ops[2]).vecsize > 1)
			emit_unrolled_binary_op(result_type, id, ops[2], ops[3], "<=");
		else
			HLSL_BOP_CAST(<=, type);
		break;
	}

	case OpFOrdLessThanEqual:
	{
		auto result_type = ops[0];
		auto id = ops[1];

		if (expression_type(ops[2]).vecsize > 1)
			emit_unrolled_binary_op(result_type, id, ops[2], ops[3], "<=");
		else
			HLSL_BOP(<=);
		break;
	}

	case OpImageQueryLod:
		emit_texture_op(instruction);
		break;

	case OpImageQuerySizeLod:
	{
		auto result_type = ops[0];
		auto id = ops[1];

		require_texture_query_variant(expression_type(ops[2]));

		auto dummy_samples_levels = join(get_fallback_name(id), "_dummy_parameter");
		statement("uint ", dummy_samples_levels, ";");

		auto expr = join("SPIRV_Cross_textureSize(", to_expression(ops[2]), ", ",
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

		require_texture_query_variant(expression_type(ops[2]));

		auto dummy_samples_levels = join(get_fallback_name(id), "_dummy_parameter");
		statement("uint ", dummy_samples_levels, ";");

		auto expr = join("SPIRV_Cross_textureSize(", to_expression(ops[2]), ", 0u, ", dummy_samples_levels, ")");
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

		require_texture_query_variant(expression_type(ops[2]));

		// Keep it simple and do not emit special variants to make this look nicer ...
		// This stuff is barely, if ever, used.
		forced_temporaries.insert(id);
		auto &type = get<SPIRType>(result_type);
		statement(variable_decl(type, to_name(id)), ";");
		statement("SPIRV_Cross_textureSize(", to_expression(ops[2]), ", 0u, ", to_name(id), ");");

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
				imgexpr = join(to_expression(ops[2]), ".Load(int2(gl_FragCoord.xy), ", to_expression(sample), ")");
			}
			else
				imgexpr = join(to_expression(ops[2]), ".Load(int3(int2(gl_FragCoord.xy), 0))");

			pure = true;
		}
		else
		{
			imgexpr = join(to_expression(ops[2]), "[", to_expression(ops[3]), "]");
			// The underlying image type in HLSL depends on the image format, unlike GLSL, where all images are "vec4",
			// except that the underlying type changes how the data is interpreted.
			if (var && !subpass_data)
				imgexpr = remap_swizzle(get<SPIRType>(result_type),
				                        image_format_to_components(get<SPIRType>(var->basetype).image.format), imgexpr);
		}

		if (var && var->forwardable)
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

		statement(to_expression(ops[0]), "[", to_expression(ops[1]), "] = ", value_expr, ";");
		if (var && variable_storage_is_aliased(*var))
			flush_all_aliased_variables();
		break;
	}

	case OpImageTexelPointer:
	{
		uint32_t result_type = ops[0];
		uint32_t id = ops[1];
		auto &e =
		    set<SPIRExpression>(id, join(to_expression(ops[2]), "[", to_expression(ops[3]), "]"), result_type, true);

		// When using the pointer, we need to know which variable it is actually loaded from.
		auto *var = maybe_get_backing_variable(ops[2]);
		e.loaded_from = var ? var->self : 0;
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
			memory = get<SPIRConstant>(ops[0]).scalar();
			semantics = get<SPIRConstant>(ops[1]).scalar();
		}
		else
		{
			memory = get<SPIRConstant>(ops[1]).scalar();
			semantics = get<SPIRConstant>(ops[2]).scalar();
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
				uint32_t next_memory = get<SPIRConstant>(next_ops[1]).scalar();
				uint32_t next_semantics = get<SPIRConstant>(next_ops[2]).scalar();
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

		auto expr = join("SPIRV_Cross_bitfieldInsert(", to_expression(ops[2]), ", ", to_expression(ops[3]), ", ",
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
			HLSL_TFOP(SPIRV_Cross_bitfieldSExtract);
		else
			HLSL_TFOP(SPIRV_Cross_bitfieldUExtract);
		break;
	}

	case OpBitCount:
		HLSL_UFOP(countbits);
		break;

	case OpBitReverse:
		HLSL_UFOP(reversebits);
		break;

	case OpArrayLength:
	{
		auto *var = maybe_get<SPIRVariable>(ops[2]);
		if (!var)
			SPIRV_CROSS_THROW("Array length must point directly to an SSBO block.");

		auto &type = get<SPIRType>(var->basetype);
		if (!has_decoration(type.self, DecorationBlock) && !has_decoration(type.self, DecorationBufferBlock))
			SPIRV_CROSS_THROW("Array length expression must point to a block type.");

		// This must be 32-bit uint, so we're good to go.
		emit_uninitialized_temporary_expression(ops[0], ops[1]);
		statement(to_expression(ops[2]), ".GetDimensions(", to_expression(ops[1]), ");");
		uint32_t offset = type_struct_member_offset(type, ops[3]);
		uint32_t stride = type_struct_member_array_stride(type, ops[3]);
		statement(to_expression(ops[1]), " = (", to_expression(ops[1]), " - ", offset, ") / ", stride, ";");
		break;
	}

	default:
		CompilerGLSL::emit_instruction(instruction);
		break;
	}
}

void CompilerHLSL::require_texture_query_variant(const SPIRType &type)
{
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

	uint64_t mask = 1ull << bit;
	if ((required_textureSizeVariants & mask) == 0)
	{
		force_recompile();
		required_textureSizeVariants |= mask;
	}
}

void CompilerHLSL::set_root_constant_layouts(std::vector<RootConstants> layout)
{
	root_constants_layout = move(layout);
}

void CompilerHLSL::add_vertex_attribute_remap(const HLSLVertexAttributeRemap &vertex_attributes)
{
	remap_vertex_attributes.push_back(vertex_attributes);
}

uint32_t CompilerHLSL::remap_num_workgroups_builtin()
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
	return variable_id;
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
		default:
			break;
		}
	}

	if (ir.addressing_model != AddressingModelLogical)
		SPIRV_CROSS_THROW("Only Logical addressing model can be used with HLSL.");
}

string CompilerHLSL::compile()
{
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
	backend.swizzle_is_function = false;
	backend.shared_is_implied = true;
	backend.unsized_array_supported = true;
	backend.explicit_struct_type = false;
	backend.use_initializer_list = true;
	backend.use_constructor_splatting = false;
	backend.boolean_mix_support = false;
	backend.can_swizzle_scalar = true;
	backend.can_declare_struct_inline = false;
	backend.can_declare_arrays_inline = false;
	backend.can_return_array = false;
	backend.nonuniform_qualifier = "NonUniformResourceIndex";

	fixup_type_alias();
	reorder_type_alias();
	build_function_control_flow_graphs_and_analyze();
	validate_shader_model();
	update_active_builtins();
	analyze_image_and_sampler_usage();

	// Subpass input needs SV_Position.
	if (need_subpass_input)
		active_input_builtins.set(BuiltInFragCoord);

	uint32_t pass_count = 0;
	do
	{
		if (pass_count >= 3)
			SPIRV_CROSS_THROW("Over 3 compilation loops detected. Must be a bug!");

		reset();

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
