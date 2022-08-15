/*
 * Copyright 2018-2021 Arm Limited
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
 */

/*
 * At your option, you may choose to accept this material under either:
 *  1. The Apache License, Version 2.0, found at <http://www.apache.org/licenses/LICENSE-2.0>, or
 *  2. The MIT License, found at <http://opensource.org/licenses/MIT>.
 */

#include "spirv_parser.hpp"
#include <assert.h>

using namespace std;
using namespace spv;

namespace SPIRV_CROSS_NAMESPACE
{
Parser::Parser(vector<uint32_t> spirv)
{
	ir.spirv = std::move(spirv);
}

Parser::Parser(const uint32_t *spirv_data, size_t word_count)
{
	ir.spirv = vector<uint32_t>(spirv_data, spirv_data + word_count);
}

static bool decoration_is_string(Decoration decoration)
{
	switch (decoration)
	{
	case DecorationHlslSemanticGOOGLE:
		return true;

	default:
		return false;
	}
}

static inline uint32_t swap_endian(uint32_t v)
{
	return ((v >> 24) & 0x000000ffu) | ((v >> 8) & 0x0000ff00u) | ((v << 8) & 0x00ff0000u) | ((v << 24) & 0xff000000u);
}

static bool is_valid_spirv_version(uint32_t version)
{
	switch (version)
	{
	// Allow v99 since it tends to just work.
	case 99:
	case 0x10000: // SPIR-V 1.0
	case 0x10100: // SPIR-V 1.1
	case 0x10200: // SPIR-V 1.2
	case 0x10300: // SPIR-V 1.3
	case 0x10400: // SPIR-V 1.4
	case 0x10500: // SPIR-V 1.5
	case 0x10600: // SPIR-V 1.6
		return true;

	default:
		return false;
	}
}

void Parser::parse()
{
	auto &spirv = ir.spirv;

	auto len = spirv.size();
	if (len < 5)
		SPIRV_CROSS_THROW("SPIRV file too small.");

	auto s = spirv.data();

	// Endian-swap if we need to.
	if (s[0] == swap_endian(MagicNumber))
		transform(begin(spirv), end(spirv), begin(spirv), [](uint32_t c) { return swap_endian(c); });

	if (s[0] != MagicNumber || !is_valid_spirv_version(s[1]))
		SPIRV_CROSS_THROW("Invalid SPIRV format.");

	uint32_t bound = s[3];

	const uint32_t MaximumNumberOfIDs = 0x3fffff;
	if (bound > MaximumNumberOfIDs)
		SPIRV_CROSS_THROW("ID bound exceeds limit of 0x3fffff.\n");

	ir.set_id_bounds(bound);

	uint32_t offset = 5;

	SmallVector<Instruction> instructions;
	while (offset < len)
	{
		Instruction instr = {};
		instr.op = spirv[offset] & 0xffff;
		instr.count = (spirv[offset] >> 16) & 0xffff;

		if (instr.count == 0)
			SPIRV_CROSS_THROW("SPIR-V instructions cannot consume 0 words. Invalid SPIR-V file.");

		instr.offset = offset + 1;
		instr.length = instr.count - 1;

		offset += instr.count;

		if (offset > spirv.size())
			SPIRV_CROSS_THROW("SPIR-V instruction goes out of bounds.");

		instructions.push_back(instr);
	}

	for (auto &i : instructions)
		parse(i);

	for (auto &fixup : forward_pointer_fixups)
	{
		auto &target = get<SPIRType>(fixup.first);
		auto &source = get<SPIRType>(fixup.second);
		target.member_types = source.member_types;
		target.basetype = source.basetype;
		target.self = source.self;
	}
	forward_pointer_fixups.clear();

	if (current_function)
		SPIRV_CROSS_THROW("Function was not terminated.");
	if (current_block)
		SPIRV_CROSS_THROW("Block was not terminated.");
	if (ir.default_entry_point == 0)
		SPIRV_CROSS_THROW("There is no entry point in the SPIR-V module.");
}

const uint32_t *Parser::stream(const Instruction &instr) const
{
	// If we're not going to use any arguments, just return nullptr.
	// We want to avoid case where we return an out of range pointer
	// that trips debug assertions on some platforms.
	if (!instr.length)
		return nullptr;

	if (instr.offset + instr.length > ir.spirv.size())
		SPIRV_CROSS_THROW("Compiler::stream() out of range.");
	return &ir.spirv[instr.offset];
}

static string extract_string(const vector<uint32_t> &spirv, uint32_t offset)
{
	string ret;
	for (uint32_t i = offset; i < spirv.size(); i++)
	{
		uint32_t w = spirv[i];

		for (uint32_t j = 0; j < 4; j++, w >>= 8)
		{
			char c = w & 0xff;
			if (c == '\0')
				return ret;
			ret += c;
		}
	}

	SPIRV_CROSS_THROW("String was not terminated before EOF");
}

void Parser::parse(const Instruction &instruction)
{
	auto *ops = stream(instruction);
	auto op = static_cast<Op>(instruction.op);
	uint32_t length = instruction.length;

	switch (op)
	{
	case OpSourceContinued:
	case OpSourceExtension:
	case OpNop:
	case OpModuleProcessed:
		break;

	case OpString:
	{
		set<SPIRString>(ops[0], extract_string(ir.spirv, instruction.offset + 1));
		break;
	}

	case OpMemoryModel:
		ir.addressing_model = static_cast<AddressingModel>(ops[0]);
		ir.memory_model = static_cast<MemoryModel>(ops[1]);
		break;

	case OpSource:
	{
		auto lang = static_cast<SourceLanguage>(ops[0]);
		switch (lang)
		{
		case SourceLanguageESSL:
			ir.source.es = true;
			ir.source.version = ops[1];
			ir.source.known = true;
			ir.source.hlsl = false;
			break;

		case SourceLanguageGLSL:
			ir.source.es = false;
			ir.source.version = ops[1];
			ir.source.known = true;
			ir.source.hlsl = false;
			break;

		case SourceLanguageHLSL:
			// For purposes of cross-compiling, this is GLSL 450.
			ir.source.es = false;
			ir.source.version = 450;
			ir.source.known = true;
			ir.source.hlsl = true;
			break;

		default:
			ir.source.known = false;
			break;
		}
		break;
	}

	case OpUndef:
	{
		uint32_t result_type = ops[0];
		uint32_t id = ops[1];
		set<SPIRUndef>(id, result_type);
		if (current_block)
			current_block->ops.push_back(instruction);
		break;
	}

	case OpCapability:
	{
		uint32_t cap = ops[0];
		if (cap == CapabilityKernel)
			SPIRV_CROSS_THROW("Kernel capability not supported.");

		ir.declared_capabilities.push_back(static_cast<Capability>(ops[0]));
		break;
	}

	case OpExtension:
	{
		auto ext = extract_string(ir.spirv, instruction.offset);
		ir.declared_extensions.push_back(std::move(ext));
		break;
	}

	case OpExtInstImport:
	{
		uint32_t id = ops[0];
		auto ext = extract_string(ir.spirv, instruction.offset + 1);
		if (ext == "GLSL.std.450")
			set<SPIRExtension>(id, SPIRExtension::GLSL);
		else if (ext == "DebugInfo")
			set<SPIRExtension>(id, SPIRExtension::SPV_debug_info);
		else if (ext == "SPV_AMD_shader_ballot")
			set<SPIRExtension>(id, SPIRExtension::SPV_AMD_shader_ballot);
		else if (ext == "SPV_AMD_shader_explicit_vertex_parameter")
			set<SPIRExtension>(id, SPIRExtension::SPV_AMD_shader_explicit_vertex_parameter);
		else if (ext == "SPV_AMD_shader_trinary_minmax")
			set<SPIRExtension>(id, SPIRExtension::SPV_AMD_shader_trinary_minmax);
		else if (ext == "SPV_AMD_gcn_shader")
			set<SPIRExtension>(id, SPIRExtension::SPV_AMD_gcn_shader);
		else if (ext == "NonSemantic.DebugPrintf")
			set<SPIRExtension>(id, SPIRExtension::NonSemanticDebugPrintf);
		else
			set<SPIRExtension>(id, SPIRExtension::Unsupported);

		// Other SPIR-V extensions which have ExtInstrs are currently not supported.

		break;
	}

	case OpExtInst:
	{
		// The SPIR-V debug information extended instructions might come at global scope.
		if (current_block)
		{
			current_block->ops.push_back(instruction);
			if (length >= 2)
			{
				const auto *type = maybe_get<SPIRType>(ops[0]);
				if (type)
					ir.load_type_width.insert({ ops[1], type->width });
			}
		}
		break;
	}

	case OpEntryPoint:
	{
		auto itr =
		    ir.entry_points.insert(make_pair(ops[1], SPIREntryPoint(ops[1], static_cast<ExecutionModel>(ops[0]),
		                                                            extract_string(ir.spirv, instruction.offset + 2))));
		auto &e = itr.first->second;

		// Strings need nul-terminator and consume the whole word.
		uint32_t strlen_words = uint32_t((e.name.size() + 1 + 3) >> 2);

		for (uint32_t i = strlen_words + 2; i < instruction.length; i++)
			e.interface_variables.push_back(ops[i]);

		// Set the name of the entry point in case OpName is not provided later.
		ir.set_name(ops[1], e.name);

		// If we don't have an entry, make the first one our "default".
		if (!ir.default_entry_point)
			ir.default_entry_point = ops[1];
		break;
	}

	case OpExecutionMode:
	{
		auto &execution = ir.entry_points[ops[0]];
		auto mode = static_cast<ExecutionMode>(ops[1]);
		execution.flags.set(mode);

		switch (mode)
		{
		case ExecutionModeInvocations:
			execution.invocations = ops[2];
			break;

		case ExecutionModeLocalSize:
			execution.workgroup_size.x = ops[2];
			execution.workgroup_size.y = ops[3];
			execution.workgroup_size.z = ops[4];
			break;

		case ExecutionModeOutputVertices:
			execution.output_vertices = ops[2];
			break;

		default:
			break;
		}
		break;
	}

	case OpExecutionModeId:
	{
		auto &execution = ir.entry_points[ops[0]];
		auto mode = static_cast<ExecutionMode>(ops[1]);
		execution.flags.set(mode);

		if (mode == ExecutionModeLocalSizeId)
		{
			execution.workgroup_size.id_x = ops[2];
			execution.workgroup_size.id_y = ops[3];
			execution.workgroup_size.id_z = ops[4];
		}

		break;
	}

	case OpName:
	{
		uint32_t id = ops[0];
		ir.set_name(id, extract_string(ir.spirv, instruction.offset + 1));
		break;
	}

	case OpMemberName:
	{
		uint32_t id = ops[0];
		uint32_t member = ops[1];
		ir.set_member_name(id, member, extract_string(ir.spirv, instruction.offset + 2));
		break;
	}

	case OpDecorationGroup:
	{
		// Noop, this simply means an ID should be a collector of decorations.
		// The meta array is already a flat array of decorations which will contain the relevant decorations.
		break;
	}

	case OpGroupDecorate:
	{
		uint32_t group_id = ops[0];
		auto &decorations = ir.meta[group_id].decoration;
		auto &flags = decorations.decoration_flags;

		// Copies decorations from one ID to another. Only copy decorations which are set in the group,
		// i.e., we cannot just copy the meta structure directly.
		for (uint32_t i = 1; i < length; i++)
		{
			uint32_t target = ops[i];
			flags.for_each_bit([&](uint32_t bit) {
				auto decoration = static_cast<Decoration>(bit);

				if (decoration_is_string(decoration))
				{
					ir.set_decoration_string(target, decoration, ir.get_decoration_string(group_id, decoration));
				}
				else
				{
					ir.meta[target].decoration_word_offset[decoration] =
					    ir.meta[group_id].decoration_word_offset[decoration];
					ir.set_decoration(target, decoration, ir.get_decoration(group_id, decoration));
				}
			});
		}
		break;
	}

	case OpGroupMemberDecorate:
	{
		uint32_t group_id = ops[0];
		auto &flags = ir.meta[group_id].decoration.decoration_flags;

		// Copies decorations from one ID to another. Only copy decorations which are set in the group,
		// i.e., we cannot just copy the meta structure directly.
		for (uint32_t i = 1; i + 1 < length; i += 2)
		{
			uint32_t target = ops[i + 0];
			uint32_t index = ops[i + 1];
			flags.for_each_bit([&](uint32_t bit) {
				auto decoration = static_cast<Decoration>(bit);

				if (decoration_is_string(decoration))
					ir.set_member_decoration_string(target, index, decoration,
					                                ir.get_decoration_string(group_id, decoration));
				else
					ir.set_member_decoration(target, index, decoration, ir.get_decoration(group_id, decoration));
			});
		}
		break;
	}

	case OpDecorate:
	case OpDecorateId:
	{
		// OpDecorateId technically supports an array of arguments, but our only supported decorations are single uint,
		// so merge decorate and decorate-id here.
		uint32_t id = ops[0];

		auto decoration = static_cast<Decoration>(ops[1]);
		if (length >= 3)
		{
			ir.meta[id].decoration_word_offset[decoration] = uint32_t(&ops[2] - ir.spirv.data());
			ir.set_decoration(id, decoration, ops[2]);
		}
		else
			ir.set_decoration(id, decoration);

		break;
	}

	case OpDecorateStringGOOGLE:
	{
		uint32_t id = ops[0];
		auto decoration = static_cast<Decoration>(ops[1]);
		ir.set_decoration_string(id, decoration, extract_string(ir.spirv, instruction.offset + 2));
		break;
	}

	case OpMemberDecorate:
	{
		uint32_t id = ops[0];
		uint32_t member = ops[1];
		auto decoration = static_cast<Decoration>(ops[2]);
		if (length >= 4)
			ir.set_member_decoration(id, member, decoration, ops[3]);
		else
			ir.set_member_decoration(id, member, decoration);
		break;
	}

	case OpMemberDecorateStringGOOGLE:
	{
		uint32_t id = ops[0];
		uint32_t member = ops[1];
		auto decoration = static_cast<Decoration>(ops[2]);
		ir.set_member_decoration_string(id, member, decoration, extract_string(ir.spirv, instruction.offset + 3));
		break;
	}

	// Build up basic types.
	case OpTypeVoid:
	{
		uint32_t id = ops[0];
		auto &type = set<SPIRType>(id);
		type.basetype = SPIRType::Void;
		break;
	}

	case OpTypeBool:
	{
		uint32_t id = ops[0];
		auto &type = set<SPIRType>(id);
		type.basetype = SPIRType::Boolean;
		type.width = 1;
		break;
	}

	case OpTypeFloat:
	{
		uint32_t id = ops[0];
		uint32_t width = ops[1];
		auto &type = set<SPIRType>(id);
		if (width == 64)
			type.basetype = SPIRType::Double;
		else if (width == 32)
			type.basetype = SPIRType::Float;
		else if (width == 16)
			type.basetype = SPIRType::Half;
		else
			SPIRV_CROSS_THROW("Unrecognized bit-width of floating point type.");
		type.width = width;
		break;
	}

	case OpTypeInt:
	{
		uint32_t id = ops[0];
		uint32_t width = ops[1];
		bool signedness = ops[2] != 0;
		auto &type = set<SPIRType>(id);
		type.basetype = signedness ? to_signed_basetype(width) : to_unsigned_basetype(width);
		type.width = width;
		break;
	}

	// Build composite types by "inheriting".
	// NOTE: The self member is also copied! For pointers and array modifiers this is a good thing
	// since we can refer to decorations on pointee classes which is needed for UBO/SSBO, I/O blocks in geometry/tess etc.
	case OpTypeVector:
	{
		uint32_t id = ops[0];
		uint32_t vecsize = ops[2];

		auto &base = get<SPIRType>(ops[1]);
		auto &vecbase = set<SPIRType>(id);

		vecbase = base;
		vecbase.vecsize = vecsize;
		vecbase.self = id;
		vecbase.parent_type = ops[1];
		break;
	}

	case OpTypeMatrix:
	{
		uint32_t id = ops[0];
		uint32_t colcount = ops[2];

		auto &base = get<SPIRType>(ops[1]);
		auto &matrixbase = set<SPIRType>(id);

		matrixbase = base;
		matrixbase.columns = colcount;
		matrixbase.self = id;
		matrixbase.parent_type = ops[1];
		break;
	}

	case OpTypeArray:
	{
		uint32_t id = ops[0];
		auto &arraybase = set<SPIRType>(id);

		uint32_t tid = ops[1];
		auto &base = get<SPIRType>(tid);

		arraybase = base;
		arraybase.parent_type = tid;

		uint32_t cid = ops[2];
		ir.mark_used_as_array_length(cid);
		auto *c = maybe_get<SPIRConstant>(cid);
		bool literal = c && !c->specialization;

		// We're copying type information into Array types, so we'll need a fixup for any physical pointer
		// references.
		if (base.forward_pointer)
			forward_pointer_fixups.push_back({ id, tid });

		arraybase.array_size_literal.push_back(literal);
		arraybase.array.push_back(literal ? c->scalar() : cid);
		// Do NOT set arraybase.self!
		break;
	}

	case OpTypeRuntimeArray:
	{
		uint32_t id = ops[0];

		auto &base = get<SPIRType>(ops[1]);
		auto &arraybase = set<SPIRType>(id);

		// We're copying type information into Array types, so we'll need a fixup for any physical pointer
		// references.
		if (base.forward_pointer)
			forward_pointer_fixups.push_back({ id, ops[1] });

		arraybase = base;
		arraybase.array.push_back(0);
		arraybase.array_size_literal.push_back(true);
		arraybase.parent_type = ops[1];
		// Do NOT set arraybase.self!
		break;
	}

	case OpTypeImage:
	{
		uint32_t id = ops[0];
		auto &type = set<SPIRType>(id);
		type.basetype = SPIRType::Image;
		type.image.type = ops[1];
		type.image.dim = static_cast<Dim>(ops[2]);
		type.image.depth = ops[3] == 1;
		type.image.arrayed = ops[4] != 0;
		type.image.ms = ops[5] != 0;
		type.image.sampled = ops[6];
		type.image.format = static_cast<ImageFormat>(ops[7]);
		type.image.access = (length >= 9) ? static_cast<AccessQualifier>(ops[8]) : AccessQualifierMax;
		break;
	}

	case OpTypeSampledImage:
	{
		uint32_t id = ops[0];
		uint32_t imagetype = ops[1];
		auto &type = set<SPIRType>(id);
		type = get<SPIRType>(imagetype);
		type.basetype = SPIRType::SampledImage;
		type.self = id;
		break;
	}

	case OpTypeSampler:
	{
		uint32_t id = ops[0];
		auto &type = set<SPIRType>(id);
		type.basetype = SPIRType::Sampler;
		break;
	}

	case OpTypePointer:
	{
		uint32_t id = ops[0];

		// Very rarely, we might receive a FunctionPrototype here.
		// We won't be able to compile it, but we shouldn't crash when parsing.
		// We should be able to reflect.
		auto *base = maybe_get<SPIRType>(ops[2]);
		auto &ptrbase = set<SPIRType>(id);

		if (base)
			ptrbase = *base;

		ptrbase.pointer = true;
		ptrbase.pointer_depth++;
		ptrbase.storage = static_cast<StorageClass>(ops[1]);

		if (ptrbase.storage == StorageClassAtomicCounter)
			ptrbase.basetype = SPIRType::AtomicCounter;

		if (base && base->forward_pointer)
			forward_pointer_fixups.push_back({ id, ops[2] });

		ptrbase.parent_type = ops[2];

		// Do NOT set ptrbase.self!
		break;
	}

	case OpTypeForwardPointer:
	{
		uint32_t id = ops[0];
		auto &ptrbase = set<SPIRType>(id);
		ptrbase.pointer = true;
		ptrbase.pointer_depth++;
		ptrbase.storage = static_cast<StorageClass>(ops[1]);
		ptrbase.forward_pointer = true;

		if (ptrbase.storage == StorageClassAtomicCounter)
			ptrbase.basetype = SPIRType::AtomicCounter;

		break;
	}

	case OpTypeStruct:
	{
		uint32_t id = ops[0];
		auto &type = set<SPIRType>(id);
		type.basetype = SPIRType::Struct;
		for (uint32_t i = 1; i < length; i++)
			type.member_types.push_back(ops[i]);

		// Check if we have seen this struct type before, with just different
		// decorations.
		//
		// Add workaround for issue #17 as well by looking at OpName for the struct
		// types, which we shouldn't normally do.
		// We should not normally have to consider type aliases like this to begin with
		// however ... glslang issues #304, #307 cover this.

		// For stripped names, never consider struct type aliasing.
		// We risk declaring the same struct multiple times, but type-punning is not allowed
		// so this is safe.
		bool consider_aliasing = !ir.get_name(type.self).empty();
		if (consider_aliasing)
		{
			for (auto &other : global_struct_cache)
			{
				if (ir.get_name(type.self) == ir.get_name(other) &&
				    types_are_logically_equivalent(type, get<SPIRType>(other)))
				{
					type.type_alias = other;
					break;
				}
			}

			if (type.type_alias == TypeID(0))
				global_struct_cache.push_back(id);
		}
		break;
	}

	case OpTypeFunction:
	{
		uint32_t id = ops[0];
		uint32_t ret = ops[1];

		auto &func = set<SPIRFunctionPrototype>(id, ret);
		for (uint32_t i = 2; i < length; i++)
			func.parameter_types.push_back(ops[i]);
		break;
	}

	case OpTypeAccelerationStructureKHR:
	{
		uint32_t id = ops[0];
		auto &type = set<SPIRType>(id);
		type.basetype = SPIRType::AccelerationStructure;
		break;
	}

	case OpTypeRayQueryKHR:
	{
		uint32_t id = ops[0];
		auto &type = set<SPIRType>(id);
		type.basetype = SPIRType::RayQuery;
		break;
	}

	// Variable declaration
	// All variables are essentially pointers with a storage qualifier.
	case OpVariable:
	{
		uint32_t type = ops[0];
		uint32_t id = ops[1];
		auto storage = static_cast<StorageClass>(ops[2]);
		uint32_t initializer = length == 4 ? ops[3] : 0;

		if (storage == StorageClassFunction)
		{
			if (!current_function)
				SPIRV_CROSS_THROW("No function currently in scope");
			current_function->add_local_variable(id);
		}

		set<SPIRVariable>(id, type, storage, initializer);
		break;
	}

	// OpPhi
	// OpPhi is a fairly magical opcode.
	// It selects temporary variables based on which parent block we *came from*.
	// In high-level languages we can "de-SSA" by creating a function local, and flush out temporaries to this function-local
	// variable to emulate SSA Phi.
	case OpPhi:
	{
		if (!current_function)
			SPIRV_CROSS_THROW("No function currently in scope");
		if (!current_block)
			SPIRV_CROSS_THROW("No block currently in scope");

		uint32_t result_type = ops[0];
		uint32_t id = ops[1];

		// Instead of a temporary, create a new function-wide temporary with this ID instead.
		auto &var = set<SPIRVariable>(id, result_type, spv::StorageClassFunction);
		var.phi_variable = true;

		current_function->add_local_variable(id);

		for (uint32_t i = 2; i + 2 <= length; i += 2)
			current_block->phi_variables.push_back({ ops[i], ops[i + 1], id });
		break;
	}

		// Constants
	case OpSpecConstant:
	case OpConstant:
	{
		uint32_t id = ops[1];
		auto &type = get<SPIRType>(ops[0]);

		if (type.width > 32)
			set<SPIRConstant>(id, ops[0], ops[2] | (uint64_t(ops[3]) << 32), op == OpSpecConstant);
		else
			set<SPIRConstant>(id, ops[0], ops[2], op == OpSpecConstant);
		break;
	}

	case OpSpecConstantFalse:
	case OpConstantFalse:
	{
		uint32_t id = ops[1];
		set<SPIRConstant>(id, ops[0], uint32_t(0), op == OpSpecConstantFalse);
		break;
	}

	case OpSpecConstantTrue:
	case OpConstantTrue:
	{
		uint32_t id = ops[1];
		set<SPIRConstant>(id, ops[0], uint32_t(1), op == OpSpecConstantTrue);
		break;
	}

	case OpConstantNull:
	{
		uint32_t id = ops[1];
		uint32_t type = ops[0];
		ir.make_constant_null(id, type, true);
		break;
	}

	case OpSpecConstantComposite:
	case OpConstantComposite:
	{
		uint32_t id = ops[1];
		uint32_t type = ops[0];

		auto &ctype = get<SPIRType>(type);

		// We can have constants which are structs and arrays.
		// In this case, our SPIRConstant will be a list of other SPIRConstant ids which we
		// can refer to.
		if (ctype.basetype == SPIRType::Struct || !ctype.array.empty())
		{
			set<SPIRConstant>(id, type, ops + 2, length - 2, op == OpSpecConstantComposite);
		}
		else
		{
			uint32_t elements = length - 2;
			if (elements > 4)
				SPIRV_CROSS_THROW("OpConstantComposite only supports 1, 2, 3 and 4 elements.");

			SPIRConstant remapped_constant_ops[4];
			const SPIRConstant *c[4];
			for (uint32_t i = 0; i < elements; i++)
			{
				// Specialization constants operations can also be part of this.
				// We do not know their value, so any attempt to query SPIRConstant later
				// will fail. We can only propagate the ID of the expression and use to_expression on it.
				auto *constant_op = maybe_get<SPIRConstantOp>(ops[2 + i]);
				auto *undef_op = maybe_get<SPIRUndef>(ops[2 + i]);
				if (constant_op)
				{
					if (op == OpConstantComposite)
						SPIRV_CROSS_THROW("Specialization constant operation used in OpConstantComposite.");

					remapped_constant_ops[i].make_null(get<SPIRType>(constant_op->basetype));
					remapped_constant_ops[i].self = constant_op->self;
					remapped_constant_ops[i].constant_type = constant_op->basetype;
					remapped_constant_ops[i].specialization = true;
					c[i] = &remapped_constant_ops[i];
				}
				else if (undef_op)
				{
					// Undefined, just pick 0.
					remapped_constant_ops[i].make_null(get<SPIRType>(undef_op->basetype));
					remapped_constant_ops[i].constant_type = undef_op->basetype;
					c[i] = &remapped_constant_ops[i];
				}
				else
					c[i] = &get<SPIRConstant>(ops[2 + i]);
			}
			set<SPIRConstant>(id, type, c, elements, op == OpSpecConstantComposite);
		}
		break;
	}

	// Functions
	case OpFunction:
	{
		uint32_t res = ops[0];
		uint32_t id = ops[1];
		// Control
		uint32_t type = ops[3];

		if (current_function)
			SPIRV_CROSS_THROW("Must end a function before starting a new one!");

		current_function = &set<SPIRFunction>(id, res, type);
		break;
	}

	case OpFunctionParameter:
	{
		uint32_t type = ops[0];
		uint32_t id = ops[1];

		if (!current_function)
			SPIRV_CROSS_THROW("Must be in a function!");

		current_function->add_parameter(type, id);
		set<SPIRVariable>(id, type, StorageClassFunction);
		break;
	}

	case OpFunctionEnd:
	{
		if (current_block)
		{
			// Very specific error message, but seems to come up quite often.
			SPIRV_CROSS_THROW(
			    "Cannot end a function before ending the current block.\n"
			    "Likely cause: If this SPIR-V was created from glslang HLSL, make sure the entry point is valid.");
		}
		current_function = nullptr;
		break;
	}

	// Blocks
	case OpLabel:
	{
		// OpLabel always starts a block.
		if (!current_function)
			SPIRV_CROSS_THROW("Blocks cannot exist outside functions!");

		uint32_t id = ops[0];

		current_function->blocks.push_back(id);
		if (!current_function->entry_block)
			current_function->entry_block = id;

		if (current_block)
			SPIRV_CROSS_THROW("Cannot start a block before ending the current block.");

		current_block = &set<SPIRBlock>(id);
		break;
	}

	// Branch instructions end blocks.
	case OpBranch:
	{
		if (!current_block)
			SPIRV_CROSS_THROW("Trying to end a non-existing block.");

		uint32_t target = ops[0];
		current_block->terminator = SPIRBlock::Direct;
		current_block->next_block = target;
		current_block = nullptr;
		break;
	}

	case OpBranchConditional:
	{
		if (!current_block)
			SPIRV_CROSS_THROW("Trying to end a non-existing block.");

		current_block->condition = ops[0];
		current_block->true_block = ops[1];
		current_block->false_block = ops[2];

		current_block->terminator = SPIRBlock::Select;

		if (current_block->true_block == current_block->false_block)
		{
			// Bogus conditional, translate to a direct branch.
			// Avoids some ugly edge cases later when analyzing CFGs.

			// There are some super jank cases where the merge block is different from the true/false,
			// and later branches can "break" out of the selection construct this way.
			// This is complete nonsense, but CTS hits this case.
			// In this scenario, we should see the selection construct as more of a Switch with one default case.
			// The problem here is that this breaks any attempt to break out of outer switch statements,
			// but it's theoretically solvable if this ever comes up using the ladder breaking system ...

			if (current_block->true_block != current_block->next_block &&
			    current_block->merge == SPIRBlock::MergeSelection)
			{
				uint32_t ids = ir.increase_bound_by(2);

				SPIRType type;
				type.basetype = SPIRType::Int;
				type.width = 32;
				set<SPIRType>(ids, type);
				auto &c = set<SPIRConstant>(ids + 1, ids);

				current_block->condition = c.self;
				current_block->default_block = current_block->true_block;
				current_block->terminator = SPIRBlock::MultiSelect;
				ir.block_meta[current_block->next_block] &= ~ParsedIR::BLOCK_META_SELECTION_MERGE_BIT;
				ir.block_meta[current_block->next_block] |= ParsedIR::BLOCK_META_MULTISELECT_MERGE_BIT;
			}
			else
			{
				ir.block_meta[current_block->next_block] &= ~ParsedIR::BLOCK_META_SELECTION_MERGE_BIT;
				current_block->next_block = current_block->true_block;
				current_block->condition = 0;
				current_block->true_block = 0;
				current_block->false_block = 0;
				current_block->merge_block = 0;
				current_block->merge = SPIRBlock::MergeNone;
				current_block->terminator = SPIRBlock::Direct;
			}
		}

		current_block = nullptr;
		break;
	}

	case OpSwitch:
	{
		if (!current_block)
			SPIRV_CROSS_THROW("Trying to end a non-existing block.");

		current_block->terminator = SPIRBlock::MultiSelect;

		current_block->condition = ops[0];
		current_block->default_block = ops[1];

		uint32_t remaining_ops = length - 2;
		if ((remaining_ops % 2) == 0)
		{
			for (uint32_t i = 2; i + 2 <= length; i += 2)
				current_block->cases_32bit.push_back({ ops[i], ops[i + 1] });
		}

		if ((remaining_ops % 3) == 0)
		{
			for (uint32_t i = 2; i + 3 <= length; i += 3)
			{
				uint64_t value = (static_cast<uint64_t>(ops[i + 1]) << 32) | ops[i];
				current_block->cases_64bit.push_back({ value, ops[i + 2] });
			}
		}

		// If we jump to next block, make it break instead since we're inside a switch case block at that point.
		ir.block_meta[current_block->next_block] |= ParsedIR::BLOCK_META_MULTISELECT_MERGE_BIT;

		current_block = nullptr;
		break;
	}

	case OpKill:
	case OpTerminateInvocation:
	{
		if (!current_block)
			SPIRV_CROSS_THROW("Trying to end a non-existing block.");
		current_block->terminator = SPIRBlock::Kill;
		current_block = nullptr;
		break;
	}

	case OpTerminateRayKHR:
		// NV variant is not a terminator.
		if (!current_block)
			SPIRV_CROSS_THROW("Trying to end a non-existing block.");
		current_block->terminator = SPIRBlock::TerminateRay;
		current_block = nullptr;
		break;

	case OpIgnoreIntersectionKHR:
		// NV variant is not a terminator.
		if (!current_block)
			SPIRV_CROSS_THROW("Trying to end a non-existing block.");
		current_block->terminator = SPIRBlock::IgnoreIntersection;
		current_block = nullptr;
		break;

	case OpReturn:
	{
		if (!current_block)
			SPIRV_CROSS_THROW("Trying to end a non-existing block.");
		current_block->terminator = SPIRBlock::Return;
		current_block = nullptr;
		break;
	}

	case OpReturnValue:
	{
		if (!current_block)
			SPIRV_CROSS_THROW("Trying to end a non-existing block.");
		current_block->terminator = SPIRBlock::Return;
		current_block->return_value = ops[0];
		current_block = nullptr;
		break;
	}

	case OpUnreachable:
	{
		if (!current_block)
			SPIRV_CROSS_THROW("Trying to end a non-existing block.");
		current_block->terminator = SPIRBlock::Unreachable;
		current_block = nullptr;
		break;
	}

	case OpSelectionMerge:
	{
		if (!current_block)
			SPIRV_CROSS_THROW("Trying to modify a non-existing block.");

		current_block->next_block = ops[0];
		current_block->merge = SPIRBlock::MergeSelection;
		ir.block_meta[current_block->next_block] |= ParsedIR::BLOCK_META_SELECTION_MERGE_BIT;

		if (length >= 2)
		{
			if (ops[1] & SelectionControlFlattenMask)
				current_block->hint = SPIRBlock::HintFlatten;
			else if (ops[1] & SelectionControlDontFlattenMask)
				current_block->hint = SPIRBlock::HintDontFlatten;
		}
		break;
	}

	case OpLoopMerge:
	{
		if (!current_block)
			SPIRV_CROSS_THROW("Trying to modify a non-existing block.");

		current_block->merge_block = ops[0];
		current_block->continue_block = ops[1];
		current_block->merge = SPIRBlock::MergeLoop;

		ir.block_meta[current_block->self] |= ParsedIR::BLOCK_META_LOOP_HEADER_BIT;
		ir.block_meta[current_block->merge_block] |= ParsedIR::BLOCK_META_LOOP_MERGE_BIT;

		ir.continue_block_to_loop_header[current_block->continue_block] = BlockID(current_block->self);

		// Don't add loop headers to continue blocks,
		// which would make it impossible branch into the loop header since
		// they are treated as continues.
		if (current_block->continue_block != BlockID(current_block->self))
			ir.block_meta[current_block->continue_block] |= ParsedIR::BLOCK_META_CONTINUE_BIT;

		if (length >= 3)
		{
			if (ops[2] & LoopControlUnrollMask)
				current_block->hint = SPIRBlock::HintUnroll;
			else if (ops[2] & LoopControlDontUnrollMask)
				current_block->hint = SPIRBlock::HintDontUnroll;
		}
		break;
	}

	case OpSpecConstantOp:
	{
		if (length < 3)
			SPIRV_CROSS_THROW("OpSpecConstantOp not enough arguments.");

		uint32_t result_type = ops[0];
		uint32_t id = ops[1];
		auto spec_op = static_cast<Op>(ops[2]);

		set<SPIRConstantOp>(id, result_type, spec_op, ops + 3, length - 3);
		break;
	}

	case OpLine:
	{
		// OpLine might come at global scope, but we don't care about those since they will not be declared in any
		// meaningful correct order.
		// Ignore all OpLine directives which live outside a function.
		if (current_block)
			current_block->ops.push_back(instruction);

		// Line directives may arrive before first OpLabel.
		// Treat this as the line of the function declaration,
		// so warnings for arguments can propagate properly.
		if (current_function)
		{
			// Store the first one we find and emit it before creating the function prototype.
			if (current_function->entry_line.file_id == 0)
			{
				current_function->entry_line.file_id = ops[0];
				current_function->entry_line.line_literal = ops[1];
			}
		}
		break;
	}

	case OpNoLine:
	{
		// OpNoLine might come at global scope.
		if (current_block)
			current_block->ops.push_back(instruction);
		break;
	}

	// Actual opcodes.
	default:
	{
		if (length >= 2)
		{
			const auto *type = maybe_get<SPIRType>(ops[0]);
			if (type)
				ir.load_type_width.insert({ ops[1], type->width });
		}

		if (!current_block)
			SPIRV_CROSS_THROW("Currently no block to insert opcode.");

		current_block->ops.push_back(instruction);
		break;
	}
	}
}

bool Parser::types_are_logically_equivalent(const SPIRType &a, const SPIRType &b) const
{
	if (a.basetype != b.basetype)
		return false;
	if (a.width != b.width)
		return false;
	if (a.vecsize != b.vecsize)
		return false;
	if (a.columns != b.columns)
		return false;
	if (a.array.size() != b.array.size())
		return false;

	size_t array_count = a.array.size();
	if (array_count && memcmp(a.array.data(), b.array.data(), array_count * sizeof(uint32_t)) != 0)
		return false;

	if (a.basetype == SPIRType::Image || a.basetype == SPIRType::SampledImage)
	{
		if (memcmp(&a.image, &b.image, sizeof(SPIRType::Image)) != 0)
			return false;
	}

	if (a.member_types.size() != b.member_types.size())
		return false;

	size_t member_types = a.member_types.size();
	for (size_t i = 0; i < member_types; i++)
	{
		if (!types_are_logically_equivalent(get<SPIRType>(a.member_types[i]), get<SPIRType>(b.member_types[i])))
			return false;
	}

	return true;
}

bool Parser::variable_storage_is_aliased(const SPIRVariable &v) const
{
	auto &type = get<SPIRType>(v.basetype);

	auto *type_meta = ir.find_meta(type.self);

	bool ssbo = v.storage == StorageClassStorageBuffer ||
	            (type_meta && type_meta->decoration.decoration_flags.get(DecorationBufferBlock));
	bool image = type.basetype == SPIRType::Image;
	bool counter = type.basetype == SPIRType::AtomicCounter;

	bool is_restrict;
	if (ssbo)
		is_restrict = ir.get_buffer_block_flags(v).get(DecorationRestrict);
	else
		is_restrict = ir.has_decoration(v.self, DecorationRestrict);

	return !is_restrict && (ssbo || image || counter);
}
} // namespace SPIRV_CROSS_NAMESPACE
