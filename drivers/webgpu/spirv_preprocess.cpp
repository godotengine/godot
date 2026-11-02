/**************************************************************************/
/*  spirv_preprocess.cpp                                                  */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "spirv_preprocess.h"

#include "core/error/error_macros.h"
#include "core/templates/hash_map.h"
#include "core/templates/hash_set.h"

#include <spirv-tools/libspirv.h>
#include <spirv-tools/optimizer.hpp>
#include <spirv/unified1/spirv.hpp>

#include <cfloat>
#include <cmath>
#include <cstring>
#include <initializer_list>
#include <map>
#include <vector>

namespace spirv_preprocess {

static constexpr const char *MALFORMED_SPIRV = "Malformed SPIR-V module; leaving it unchanged.";

static inline uint32_t read_word(const uint8_t *p_data, uint32_t p_index) {
	uint32_t word;
	memcpy(&word, p_data + p_index * 4, 4); // Little-endian hosts only, like the rest of Godot.
	return word;
}

static inline void write_word(uint8_t *p_data, uint32_t p_index, uint32_t p_word) {
	memcpy(p_data + p_index * 4, &p_word, 4);
}

static inline void push_word(Vector<uint8_t> &r_out, uint32_t p_word) {
	const int64_t size = r_out.size();
	r_out.resize(size + 4);
	memcpy(r_out.ptrw() + size, &p_word, 4);
}

static void push_inst(Vector<uint8_t> &r_out, spv::Op p_op, std::initializer_list<uint32_t> p_operands) {
	push_word(r_out, (uint32_t(p_operands.size() + 1) << 16) | p_op);
	for (uint32_t operand : p_operands) {
		push_word(r_out, operand);
	}
}

static inline void append_bytes(Vector<uint8_t> &r_out, const uint8_t *p_src, int64_t p_offset, int64_t p_count) {
	const int64_t size = r_out.size();
	r_out.resize(size + p_count);
	memcpy(r_out.ptrw() + size, p_src + p_offset, p_count);
}

// Header, alignment and instruction word counts; every pass checks this before walking a module.
static bool is_well_formed(const Vector<uint8_t> &p_bytes) {
	if (p_bytes.size() < 20 || p_bytes.size() % 4 != 0 || read_word(p_bytes.ptr(), 0) != spv::MagicNumber) {
		return false;
	}
	const uint32_t total = uint32_t(p_bytes.size() / 4);
	for (uint32_t pos = 5; pos < total;) {
		const uint32_t count = read_word(p_bytes.ptr(), pos) >> 16;
		if (count == 0 || count > total - pos) {
			return false;
		}
		pos += count;
	}
	return true;
}

// Iterates the instructions of a module that passed is_well_formed.
struct SpirvView {
	const uint8_t *data;
	uint32_t word_count;

	explicit SpirvView(const Vector<uint8_t> &p_bytes) :
			data(p_bytes.ptr()), word_count(uint32_t(p_bytes.size() / 4)) {}
	uint32_t word(uint32_t p_index) const { return read_word(data, p_index); }

	struct Inst {
		const SpirvView *view;
		uint32_t pos;
		uint32_t count() const { return view->word(pos) >> 16; }
		uint16_t op() const { return view->word(pos) & 0xffff; }
		// Operand words, 0 past the end of the instruction.
		uint32_t operator[](uint32_t p_index) const { return p_index < count() ? view->word(pos + p_index) : 0; }
	};
	struct Iterator {
		Inst inst;
		Inst operator*() const { return inst; }
		Iterator &operator++() {
			inst.pos += inst.count();
			return *this;
		}
		bool operator!=(const Iterator &p_other) const { return inst.pos != p_other.inst.pos; }
	};
	Iterator begin() const { return { { this, 5 } }; }
	Iterator end() const { return { { this, word_count } }; }
};
using Inst = SpirvView::Inst;

static inline void append_inst(Vector<uint8_t> &r_out, const Inst &p_inst) {
	append_bytes(r_out, p_inst.view->data, p_inst.pos * 4, p_inst.count() * 4);
}

// Instructions that precede annotations; injected decorations go before the first other instruction.
static bool is_before_annotations(uint16_t p_op) {
	switch (p_op) {
		case spv::OpSourceContinued:
		case spv::OpSource:
		case spv::OpSourceExtension:
		case spv::OpName:
		case spv::OpMemberName:
		case spv::OpString:
		case spv::OpExtension:
		case spv::OpExtInstImport:
		case spv::OpMemoryModel:
		case spv::OpEntryPoint:
		case spv::OpExecutionMode:
		case spv::OpCapability:
		case spv::OpModuleProcessed:
		case spv::OpExecutionModeId:
			return true;
		default:
			return false;
	}
}

// Rewrites one opcode to another with the same operand layout.
static Vector<uint8_t> replace_opcode(const Vector<uint8_t> &p_bytes, spv::Op p_from, spv::Op p_to) {
	ERR_FAIL_COND_V_MSG(!is_well_formed(p_bytes), p_bytes, MALFORMED_SPIRV);
	Vector<uint8_t> out = p_bytes; // Copied on the first write.
	for (const Inst inst : SpirvView(p_bytes)) {
		if (inst.op() == p_from) {
			write_word(out.ptrw(), inst.pos, (inst.count() << 16) | p_to);
		}
	}
	return out;
}

// Copies the module without the instructions p_strip selects, or returns it unchanged when none match.
template <typename F>
static Vector<uint8_t> strip_instructions(const Vector<uint8_t> &p_bytes, F p_strip) {
	Vector<uint8_t> out;
	bool stripped = false;
	for (const Inst inst : SpirvView(p_bytes)) {
		if (p_strip(inst)) {
			if (!stripped) {
				append_bytes(out, p_bytes.ptr(), 0, inst.pos * 4);
				stripped = true;
			}
		} else if (stripped) {
			append_inst(out, inst);
		}
	}
	return stripped ? out : p_bytes;
}

Vector<uint8_t> run_pipeline(const Vector<uint8_t> &p_bytes) {
	ERR_FAIL_COND_V_MSG(!is_well_formed(p_bytes), p_bytes, MALFORMED_SPIRV);
	Vector<uint8_t> code = freeze_and_inline(p_bytes);
	code = rewrite_copy_logical(code);
	code = rewrite_terminate_invocation(code);
	code = rewrite_float_classification(code);
	// Flattening first lets the split treat arrays of combined samplers as plain combined samplers.
	code = flatten_binding_arrays(code);
	code = split_combined_samplers(code);
	code = convert_push_constants_to_storage_buffers(code);
	code = fix_depth2_images(code);
	code = negate_position_y(code);
	code = strip_restrict_decoration(code);
	code = strip_memory_barrier(code);
	code = fix_nonfinite_literals(code);
	code = normalize_storage_buffer_access(code);
	return infer_readonly_storage(code);
}

Vector<uint8_t> freeze_and_inline(const Vector<uint8_t> &p_bytes) {
	ERR_FAIL_COND_V_MSG(!is_well_formed(p_bytes), p_bytes, MALFORMED_SPIRV);
	spvtools::Optimizer optimizer(SPV_ENV_UNIVERSAL_1_6);
	optimizer.SetMessageConsumer([](spv_message_level_t p_level, const char *, const spv_position_t &, const char *p_message) {
		if (p_level <= SPV_MSG_ERROR) {
			ERR_PRINT(p_message);
		}
	});
	// The driver patches specialization values into the defaults first. Merged returns let helpers inline.
	// DXC miscompiles row-major matrices in struct inout arguments; inline before Tint emits those calls.
	// Preserve scalar SpecIds as WGSL overrides; only unsupported composite operations need freezing.
	for (const Inst inst : SpirvView(p_bytes)) {
		if (inst.op() == spv::OpSpecConstantComposite || (inst.op() == spv::OpSpecConstantOp && (inst[3] == spv::OpCompositeInsert || inst[3] == spv::OpVectorShuffle || inst[3] == spv::OpSelect))) {
			optimizer.RegisterPass(spvtools::CreateFreezeSpecConstantValuePass());
			break;
		}
	}
	optimizer.RegisterPass(spvtools::CreateFoldSpecConstantOpAndCompositePass())
			.RegisterPass(spvtools::CreateMergeReturnPass())
			.RegisterPass(spvtools::CreateInlineExhaustivePass())
			.RegisterPass(spvtools::CreateDeadBranchElimPass())
			.RegisterPass(spvtools::CreateCFGCleanupPass())
			.RegisterPass(spvtools::CreateEliminateDeadFunctionsPass());
	std::vector<uint32_t> words;
	const bool optimized = optimizer.Run(reinterpret_cast<const uint32_t *>(p_bytes.ptr()), p_bytes.size() / 4, &words);
	ERR_FAIL_COND_V_MSG(!optimized, p_bytes, "SPIRV-Tools could not freeze specialization constants or inline shader calls.");
	Vector<uint8_t> out;
	out.resize(words.size() * 4);
	memcpy(out.ptrw(), words.data(), out.size());
	return out;
}

Vector<uint8_t> rewrite_copy_logical(const Vector<uint8_t> &p_bytes) {
	return replace_opcode(p_bytes, spv::OpCopyLogical, spv::OpCopyObject);
}

Vector<uint8_t> rewrite_terminate_invocation(const Vector<uint8_t> &p_bytes) {
	return replace_opcode(p_bytes, spv::OpTerminateInvocation, spv::OpKill);
}

Vector<uint8_t> rewrite_float_classification(const Vector<uint8_t> &p_bytes) {
	ERR_FAIL_COND_V_MSG(!is_well_formed(p_bytes), p_bytes, MALFORMED_SPIRV);
	HashMap<uint32_t, uint32_t> float_widths, vector_elements, vector_lengths, value_types;
	bool needed = false;
	uint32_t uint_type = 0;
	for (const Inst inst : SpirvView(p_bytes)) {
		if (inst.op() == spv::OpTypeFloat) {
			float_widths[inst[1]] = inst[2];
		} else if (inst.op() == spv::OpTypeVector) {
			vector_elements[inst[1]] = inst[2];
			vector_lengths[inst[1]] = inst[3];
		} else if (inst.op() == spv::OpTypeInt && inst[2] == 32 && inst[3] == 0) {
			uint_type = inst[1];
		}
		needed |= inst.op() == spv::OpIsNan || inst.op() == spv::OpIsInf;
	}
	if (!needed) {
		return p_bytes;
	}
	for (const Inst inst : SpirvView(p_bytes)) {
		if (inst.count() >= 3 && (float_widths.has(inst[1]) || vector_elements.has(inst[1]))) {
			value_types[inst[2]] = inst[1];
		}
	}
	uint32_t next_id = read_word(p_bytes.ptr(), 3);
	Vector<uint8_t> declarations;
	if (!uint_type) {
		uint_type = next_id++;
		push_inst(declarations, spv::OpTypeInt, { uint_type, 32, 0 });
	}
	const uint32_t abs_mask = next_id++, infinity = next_id++;
	push_inst(declarations, spv::OpConstant, { uint_type, abs_mask, 0x7fffffffu });
	push_inst(declarations, spv::OpConstant, { uint_type, infinity, 0x7f800000u });
	HashMap<uint32_t, uint32_t> uint_vectors, masks, infinities;
	uint_vectors[1] = uint_type;
	masks[1] = abs_mask;
	infinities[1] = infinity;
	for (const KeyValue<uint32_t, uint32_t> &element : vector_elements) {
		if (element.value == uint_type) {
			uint_vectors[vector_lengths[element.key]] = element.key;
		}
	}
	Vector<uint8_t> body;
	bool in_function = false;
	for (const Inst inst : SpirvView(p_bytes)) {
		in_function |= inst.op() == spv::OpFunction;
		if (!in_function) {
			continue;
		}
		if (inst.op() != spv::OpIsNan && inst.op() != spv::OpIsInf) {
			append_inst(body, inst);
			continue;
		}
		const uint32_t type = value_types[inst[3]];
		const uint32_t count = vector_lengths.has(type) ? vector_lengths[type] : 1;
		const uint32_t scalar = vector_elements.has(type) ? vector_elements[type] : type;
		ERR_FAIL_COND_V_MSG(float_widths[scalar] != 32, p_bytes, "WebGPU float classification requires 32-bit floats.");
		if (!uint_vectors.has(count)) {
			uint_vectors[count] = next_id++;
			push_inst(declarations, spv::OpTypeVector, { uint_vectors[count], uint_type, count });
		}
		if (!masks.has(count)) {
			for (uint32_t constant : { abs_mask, infinity }) {
				const uint32_t id = next_id++;
				(constant == abs_mask ? masks : infinities)[count] = id;
				push_word(declarations, ((count + 3) << 16) | spv::OpConstantComposite);
				push_word(declarations, uint_vectors[count]);
				push_word(declarations, id);
				for (uint32_t component = 0; component < count; component++) {
					push_word(declarations, constant);
				}
			}
		}
		const uint32_t bits = next_id++, magnitude = next_id++;
		push_inst(body, spv::OpBitcast, { uint_vectors[count], bits, inst[3] });
		push_inst(body, spv::OpBitwiseAnd, { uint_vectors[count], magnitude, bits, masks[count] });
		push_inst(body, inst.op() == spv::OpIsNan ? spv::OpUGreaterThan : spv::OpIEqual, { inst[1], inst[2], magnitude, infinities[count] });
	}
	Vector<uint8_t> out;
	append_bytes(out, p_bytes.ptr(), 0, 20);
	write_word(out.ptrw(), 3, next_id);
	for (const Inst inst : SpirvView(p_bytes)) {
		if (inst.op() == spv::OpFunction) {
			break;
		}
		append_inst(out, inst);
	}
	append_bytes(out, declarations.ptr(), 0, declarations.size());
	append_bytes(out, body.ptr(), 0, body.size());
	return out;
}

Vector<uint8_t> remap_bindings(const Vector<uint8_t> &p_bytes, const BindingMap &p_bindings) {
	ERR_FAIL_COND_V_MSG(!is_well_formed(p_bytes), p_bytes, MALFORMED_SPIRV);
	HashMap<uint32_t, uint32_t> sets;
	for (const Inst inst : SpirvView(p_bytes)) {
		if (inst.op() == spv::OpDecorate && inst[2] == spv::DecorationDescriptorSet) {
			sets[inst[1]] = inst[3];
		}
	}
	Vector<uint8_t> out = p_bytes;
	for (const Inst inst : SpirvView(p_bytes)) {
		if (inst.op() == spv::OpDecorate && inst[2] == spv::DecorationBinding && sets.has(inst[1])) {
			const uint32_t *binding = p_bindings.getptr((uint64_t(sets[inst[1]]) << 32) | inst[3]);
			if (binding) {
				write_word(out.ptrw(), inst.pos + 3, *binding);
			}
		}
	}
	return out;
}

Vector<uint8_t> flatten_binding_arrays(const Vector<uint8_t> &p_bytes) {
	ERR_FAIL_COND_V_MSG(!is_well_formed(p_bytes), p_bytes, MALFORMED_SPIRV);
	HashSet<uint32_t> handles;
	HashMap<uint32_t, uint32_t> constants, lengths, pointees, variables, sets, bindings;
	for (const Inst inst : SpirvView(p_bytes)) {
		switch (inst.op()) {
			case spv::OpTypeImage:
			case spv::OpTypeSampler:
			case spv::OpTypeSampledImage:
				handles.insert(inst[1]);
				break;
			case spv::OpConstant:
				constants[inst[2]] = inst[3];
				break;
			case spv::OpTypeArray:
				if (handles.has(inst[2])) {
					lengths[inst[1]] = constants[inst[3]];
				}
				break;
			case spv::OpTypeRuntimeArray:
				ERR_FAIL_COND_V_MSG(handles.has(inst[2]), p_bytes, "WebGPU requires fixed-size texture and sampler arrays.");
				break;
			case spv::OpTypePointer:
				pointees[inst[1]] = inst[3];
				break;
			case spv::OpVariable:
				variables[inst[2]] = inst[1];
				break;
			case spv::OpDecorate:
				if (inst[2] == spv::DecorationDescriptorSet) {
					sets[inst[1]] = inst[3];
				} else if (inst[2] == spv::DecorationBinding) {
					bindings[inst[1]] = inst[3];
				}
				break;
			default:
				break;
		}
	}
	if (lengths.is_empty()) {
		return p_bytes;
	}
	// Reserve consecutive bindings before SPIRV-Tools expands each descriptor array.
	// This is idempotent when the driver already reserved space using both stages' reflection.
	std::map<uint32_t, std::map<uint32_t, uint32_t>> ranges;
	for (const KeyValue<uint32_t, uint32_t> &binding : bindings) {
		if (!sets.has(binding.key) || !variables.has(binding.key)) {
			continue;
		}
		const uint32_t *length = lengths.getptr(pointees[variables[binding.key]]);
		ranges[sets[binding.key]][binding.value] = length ? *length : 1;
	}
	BindingMap mapping;
	for (const auto &set : ranges) {
		uint32_t next = 0;
		for (const auto &binding : set.second) {
			next = std::max(next, binding.first);
			mapping[(uint64_t(set.first) << 32) | binding.first] = next;
			next += binding.second;
			ERR_FAIL_COND_V_MSG(next * 2 >= PUSH_CONSTANT_BINDING, p_bytes, "WebGPU descriptor arrays exceed the binding range.");
		}
	}
	Vector<uint8_t> input = remap_bindings(p_bytes, mapping);
	spvtools::Optimizer optimizer(SPV_ENV_UNIVERSAL_1_6);
	optimizer.SetMessageConsumer([](spv_message_level_t p_level, const char *, const spv_position_t &, const char *p_message) {
		if (p_level <= SPV_MSG_ERROR) {
			ERR_PRINT(p_message);
		}
	});
	optimizer.RegisterPass(spvtools::CreateReplaceDescArrayAccessUsingVarIndexPass())
			.RegisterPass(spvtools::CreateAggressiveDCEPass())
			.RegisterPass(spvtools::CreateDescriptorArrayScalarReplacementPass())
			.RegisterPass(spvtools::CreateCFGCleanupPass());
	std::vector<uint32_t> words;
	const bool optimized = optimizer.Run(reinterpret_cast<const uint32_t *>(input.ptr()), input.size() / 4, &words);
	ERR_FAIL_COND_V_MSG(!optimized, p_bytes, "SPIRV-Tools could not lower descriptor arrays to individual bindings.");
	Vector<uint8_t> out;
	out.resize(words.size() * 4);
	memcpy(out.ptrw(), words.data(), out.size());
	// Scalar replacement leaves unused handle-array types, which Tint still rejects.
	HashSet<uint32_t> removed_types;
	for (const KeyValue<uint32_t, uint32_t> &length : lengths) {
		removed_types.insert(length.key);
	}
	for (const KeyValue<uint32_t, uint32_t> &pointee : pointees) {
		if (lengths.has(pointee.value)) {
			removed_types.insert(pointee.key);
		}
	}
	return strip_instructions(out, [&](const Inst &inst) {
		return ((inst.op() == spv::OpTypeArray || inst.op() == spv::OpTypePointer) && removed_types.has(inst[1])) ||
				((inst.op() == spv::OpName || inst.op() == spv::OpDecorate) && removed_types.has(inst[1]));
	});
}

Vector<uint8_t> split_combined_samplers(const Vector<uint8_t> &p_bytes) {
	ERR_FAIL_COND_V_MSG(!is_well_formed(p_bytes), p_bytes, MALFORMED_SPIRV);
	const SpirvView code(p_bytes);

	struct Split {
		uint32_t var;
		uint32_t image_type;
		uint32_t sampled_image_type;
		uint32_t set;
		uint32_t binding;
		bool multisampled;
		uint32_t image_ptr_type = 0;
		uint32_t sampler_var = 0;
	};
	HashSet<uint32_t> multisampled_images;
	HashMap<uint32_t, uint32_t> sampled_image_types; // Sampled image type -> image type.
	HashMap<uint32_t, uint32_t> uniform_constant_ptrs; // Pointer type -> pointee.
	Vector<uint32_t> uniform_constant_vars;
	HashMap<uint32_t, uint32_t> var_types;
	HashMap<uint32_t, uint32_t> sets;
	HashMap<uint32_t, uint32_t> bindings;
	HashSet<uint32_t> call_arguments;
	uint32_t sampler_type = 0;
	for (const Inst inst : code) {
		switch (inst.op()) {
			case spv::OpTypeImage:
				if (inst[6] == 1) {
					multisampled_images.insert(inst[1]);
				}
				break;
			case spv::OpTypeSampledImage:
				sampled_image_types.insert(inst[1], inst[2]);
				break;
			case spv::OpTypeSampler:
				sampler_type = inst[1];
				break;
			case spv::OpTypePointer:
				if (inst[2] == spv::StorageClassUniformConstant) {
					uniform_constant_ptrs.insert(inst[1], inst[3]);
				}
				break;
			case spv::OpVariable:
				if (inst[3] == spv::StorageClassUniformConstant) {
					uniform_constant_vars.push_back(inst[2]);
					var_types.insert(inst[2], inst[1]);
				}
				break;
			case spv::OpDecorate:
				if (inst.count() >= 4 && inst[2] == spv::DecorationDescriptorSet) {
					sets.insert(inst[1], inst[3]);
				} else if (inst.count() >= 4 && inst[2] == spv::DecorationBinding) {
					ERR_FAIL_COND_V_MSG(inst[3] >= PUSH_CONSTANT_BINDING / 2, p_bytes, "Binding too high to double below the WebGPU push-constant binding.");
					bindings.insert(inst[1], inst[3]);
				}
				break;
			case spv::OpFunctionCall:
				for (uint32_t i = 4; i < inst.count(); i++) {
					call_arguments.insert(inst[i]);
				}
				break;
			default:
				break;
		}
	}

	uint32_t next_id = code.word(3);
	Vector<Split> splits;
	HashMap<uint32_t, int> split_of_var;
	HashMap<uint32_t, uint32_t> image_ptr_types; // Image type -> new UniformConstant pointer.
	bool needs_sampler = false;
	for (uint32_t var : uniform_constant_vars) {
		const uint32_t *pointee = uniform_constant_ptrs.getptr(var_types[var]);
		const uint32_t *image = pointee ? sampled_image_types.getptr(*pointee) : nullptr;
		if (!image || !sets.has(var) || !bindings.has(var)) {
			continue;
		}
		ERR_FAIL_COND_V_MSG(call_arguments.has(var), p_bytes, "A combined image sampler is passed to a function that was not inlined.");
		Split split = { var, *image, *pointee, sets[var], bindings[var], multisampled_images.has(*image) };
		needs_sampler = needs_sampler || !split.multisampled;
		split_of_var.insert(var, int(splits.size()));
		splits.push_back(split);
	}
	const bool new_sampler_type = needs_sampler && sampler_type == 0;
	if (new_sampler_type) {
		sampler_type = next_id++;
	}
	const uint32_t sampler_ptr_type = needs_sampler ? next_id++ : 0;
	for (Split &split : splits) {
		if (!image_ptr_types.has(split.image_type)) {
			image_ptr_types.insert(split.image_type, next_id++);
		}
		split.image_ptr_type = image_ptr_types[split.image_type];
		if (!split.multisampled) {
			split.sampler_var = next_id++;
		}
	}

	// Combined-sampler loads -> split index, for OpImage on multisampled loads.
	HashMap<uint32_t, int> split_of_load;
	struct SampledLoad {
		uint32_t block;
		uint32_t image;
		uint32_t sampler;
		uint32_t type;
	};
	HashMap<uint32_t, SampledLoad> sampled_loads;
	uint32_t block = 0;
	Vector<uint8_t> out;
	append_bytes(out, code.data, 0, 20);
	bool decorations_injected = false;
	bool variables_injected = false;
	for (const Inst inst : code) {
		const uint16_t op = inst.op();
		if (!decorations_injected && !is_before_annotations(op)) {
			decorations_injected = true;
			for (const Split &split : splits) {
				if (split.sampler_var) {
					push_inst(out, spv::OpDecorate, { split.sampler_var, spv::DecorationDescriptorSet, split.set });
					push_inst(out, spv::OpDecorate, { split.sampler_var, spv::DecorationBinding, split.binding * 2 });
				}
			}
		}
		if (op == spv::OpFunction && !variables_injected) {
			variables_injected = true;
			if (new_sampler_type) {
				push_inst(out, spv::OpTypeSampler, { sampler_type });
			}
			if (sampler_ptr_type) {
				push_inst(out, spv::OpTypePointer, { sampler_ptr_type, spv::StorageClassUniformConstant, sampler_type });
			}
			for (const Split &split : splits) {
				if (split.sampler_var) {
					push_inst(out, spv::OpVariable, { sampler_ptr_type, split.sampler_var, spv::StorageClassUniformConstant });
				}
			}
		}

		if (op == spv::OpLabel) {
			block = inst[1];
		}
		const int *split_index = nullptr;
		switch (op) {
			case spv::OpEntryPoint: {
				// SPIR-V 1.4+ interfaces list every global: add the sampler next to its image.
				push_word(out, 0); // Patched below.
				const int64_t start = out.size() / 4 - 1;
				uint32_t i = 1;
				for (; i < inst.count(); i++) {
					push_word(out, inst[i]);
					if (i >= 3 && (inst[i] >> 24) == 0) {
						break; // Last word of the name.
					}
				}
				for (i++; i < inst.count(); i++) {
					push_word(out, inst[i]);
					split_index = split_of_var.getptr(inst[i]);
					if (split_index && splits[*split_index].sampler_var) {
						push_word(out, splits[*split_index].sampler_var);
					}
				}
				write_word(out.ptrw(), uint32_t(start), (uint32_t(out.size() / 4 - start) << 16) | spv::OpEntryPoint);
				continue;
			}
			case spv::OpDecorate:
				if (inst.count() >= 4 && inst[2] == spv::DecorationBinding) {
					split_index = split_of_var.getptr(inst[1]);
					const bool image_of_split = split_index && splits[*split_index].sampler_var;
					push_inst(out, spv::OpDecorate, { inst[1], spv::DecorationBinding, inst[3] * 2 + (image_of_split ? 1 : 0) });
					continue;
				}
				if (inst[2] == spv::DecorationNonWritable && split_of_var.has(inst[1])) {
					continue;
				}
				break;
			case spv::OpVariable:
				split_index = split_of_var.getptr(inst[2]);
				if (split_index) {
					// A pointer to the image, so Tint does not split it again into bindings the layout lacks.
					push_inst(out, spv::OpVariable, { splits[*split_index].image_ptr_type, inst[2], inst[3] });
					continue;
				}
				break;
			case spv::OpLoad:
				split_index = split_of_var.getptr(inst[3]);
				if (split_index) {
					const Split &split = splits[*split_index];
					if (split.multisampled) {
						push_inst(out, spv::OpLoad, { split.image_type, inst[2], split.var });
					} else {
						const uint32_t image = next_id++;
						const uint32_t sampler = next_id++;
						push_inst(out, spv::OpLoad, { split.image_type, image, split.var });
						push_inst(out, spv::OpLoad, { sampler_type, sampler, split.sampler_var });
						push_inst(out, spv::OpSampledImage, { split.sampled_image_type, inst[2], image, sampler });
						sampled_loads.insert(inst[2], { block, image, sampler, split.sampled_image_type });
					}
					split_of_load.insert(inst[2], *split_index);
					continue;
				}
				break;
			case spv::OpImage:
				split_index = split_of_load.getptr(inst[3]);
				if (split_index && splits[*split_index].multisampled) {
					// The load already produced the image.
					push_inst(out, spv::OpCopyObject, { inst[1], inst[2], inst[3] });
					continue;
				}
				break;
			default:
				break;
		}
		// Combined OpLoad values may cross blocks; OpSampledImage values must be consumed in their own block.
		if ((op >= spv::OpImageSampleImplicitLod && op <= spv::OpImageDrefGather) || op == spv::OpImage || op == spv::OpImageQueryLod) {
			const SampledLoad *load = sampled_loads.getptr(inst[3]);
			if (load && load->block != block) {
				const uint32_t sampled = next_id++;
				push_inst(out, spv::OpSampledImage, { load->type, sampled, load->image, load->sampler });
				const int64_t start = out.size();
				append_inst(out, inst);
				write_word(out.ptrw() + start, 3, sampled);
				continue;
			}
		}
		append_inst(out, inst);
		// Image pointers must follow their image type and precede the variables using them.
		if (op == spv::OpTypeImage) {
			const uint32_t *image_ptr_type = image_ptr_types.getptr(inst[1]);
			if (image_ptr_type) {
				push_inst(out, spv::OpTypePointer, { *image_ptr_type, spv::StorageClassUniformConstant, inst[1] });
			}
		}
	}
	write_word(out.ptrw(), 3, next_id);
	return out;
}

Vector<uint8_t> convert_push_constants_to_storage_buffers(const Vector<uint8_t> &p_bytes) {
	ERR_FAIL_COND_V_MSG(!is_well_formed(p_bytes), p_bytes, MALFORMED_SPIRV);
	const SpirvView code(p_bytes);
	Vector<uint32_t> push_constants;
	for (const Inst inst : code) {
		if (inst.op() == spv::OpVariable && inst[3] == spv::StorageClassPushConstant) {
			push_constants.push_back(inst[2]);
		}
	}
	if (push_constants.is_empty()) {
		return p_bytes;
	}

	Vector<uint8_t> out;
	append_bytes(out, code.data, 0, 20);
	bool decorations_injected = false;
	for (const Inst inst : code) {
		const uint16_t op = inst.op();
		if (!decorations_injected && !is_before_annotations(op)) {
			decorations_injected = true;
			for (uint32_t var : push_constants) {
				push_inst(out, spv::OpDecorate, { var, spv::DecorationDescriptorSet, PUSH_CONSTANT_SET });
				push_inst(out, spv::OpDecorate, { var, spv::DecorationBinding, PUSH_CONSTANT_BINDING });
				push_inst(out, spv::OpDecorate, { var, spv::DecorationNonWritable });
			}
		}
		if (op == spv::OpDecorate && (inst[2] == spv::DecorationDescriptorSet || inst[2] == spv::DecorationBinding) &&
				push_constants.has(inst[1])) {
			continue;
		}
		// Pointer types also cover access-chain results into the push constants.
		if ((op == spv::OpTypePointer && inst[2] == spv::StorageClassPushConstant) ||
				(op == spv::OpVariable && inst[3] == spv::StorageClassPushConstant)) {
			const uint32_t storage_class_index = op == spv::OpTypePointer ? 2 : 3;
			const int64_t start = out.size();
			append_inst(out, inst);
			write_word(out.ptrw() + start, storage_class_index, spv::StorageClassStorageBuffer);
			continue;
		}
		append_inst(out, inst);
	}
	return out;
}

const char *unsupported_feature(const Vector<uint8_t> &p_bytes) {
	ERR_FAIL_COND_V_MSG(!is_well_formed(p_bytes), nullptr, MALFORMED_SPIRV);
	for (const Inst inst : SpirvView(p_bytes)) {
		if (inst.op() == spv::OpImageTexelPointer) {
			return "image atomics";
		}
		// A declared storage image without a format fails even if this variant never accesses it.
		if (inst.op() == spv::OpTypeImage && inst.count() >= 9 && inst[7] == 2 && inst[8] == spv::ImageFormatUnknown && inst[3] != spv::DimSubpassData) {
			return "storage images without a format";
		}
		if (inst.op() != spv::OpCapability) {
			continue;
		}
		switch (inst[1]) {
			case spv::CapabilityFloat16:
			case spv::CapabilityInt16:
			case spv::CapabilityInt8:
			case spv::CapabilityStorageBuffer16BitAccess:
			case spv::CapabilityUniformAndStorageBuffer16BitAccess:
			case spv::CapabilityStoragePushConstant16:
			case spv::CapabilityStorageInputOutput16:
				return "8- or 16-bit types";
			default:
				break;
		}
	}
	return nullptr;
}

Vector<uint8_t> fix_depth2_images(const Vector<uint8_t> &p_bytes) {
	ERR_FAIL_COND_V_MSG(!is_well_formed(p_bytes), p_bytes, MALFORMED_SPIRV);
	Vector<uint8_t> out = p_bytes; // Copied on the first write.
	for (const Inst inst : SpirvView(p_bytes)) {
		if (inst.op() == spv::OpTypeImage && inst.count() >= 9 && inst[4] == 2) {
			write_word(out.ptrw(), inst.pos + 4, 1);
		}
	}
	return out;
}

Vector<uint8_t> negate_position_y(const Vector<uint8_t> &p_bytes) {
	ERR_FAIL_COND_V_MSG(!is_well_formed(p_bytes), p_bytes, MALFORMED_SPIRV);
	const SpirvView code(p_bytes);

	// gl_Position is either a decorated variable or a member of a decorated output block.
	uint32_t vertex_function = 0;
	uint32_t position_var = 0;
	uint32_t position_struct = 0;
	uint32_t position_member = 0;
	uint32_t output_var = 0;
	uint32_t float_type = 0;
	uint32_t int_type = 0;
	HashSet<uint32_t> int_types;
	HashMap<uint32_t, uint32_t> small_int_constants; // Value -> constant.
	HashMap<uint32_t, uint32_t> output_ptr_pointees;
	for (const Inst inst : code) {
		switch (inst.op()) {
			case spv::OpEntryPoint:
				if (inst[1] == spv::ExecutionModelVertex) {
					vertex_function = inst[2];
				}
				break;
			case spv::OpDecorate:
				if (inst[2] == spv::DecorationBuiltIn && inst[3] == spv::BuiltInPosition) {
					position_var = inst[1];
				}
				break;
			case spv::OpMemberDecorate:
				if (inst[3] == spv::DecorationBuiltIn && inst[4] == spv::BuiltInPosition) {
					position_struct = inst[1];
					position_member = inst[2];
				}
				break;
			case spv::OpTypeInt:
				int_types.insert(inst[1]);
				if (!int_type && inst[2] == 32) {
					int_type = inst[1];
				}
				break;
			case spv::OpTypeFloat:
				if (inst[2] == 32) {
					float_type = inst[1];
				}
				break;
			case spv::OpTypePointer:
				if (inst[2] == spv::StorageClassOutput) {
					output_ptr_pointees.insert(inst[1], inst[3]);
				}
				break;
			case spv::OpVariable:
				if (!output_var && !position_var && inst[3] == spv::StorageClassOutput) {
					const uint32_t *pointee = output_ptr_pointees.getptr(inst[1]);
					if (pointee && *pointee == position_struct) {
						output_var = inst[2];
					}
				}
				break;
			case spv::OpConstant:
				// Integer type check: 0.0f has the same bits as 0.
				if (inst[3] <= 4 && int_types.has(inst[1])) {
					small_int_constants.insert(inst[3], inst[2]);
				}
				break;
			default:
				break;
		}
	}
	if (position_var) {
		output_var = position_var;
	}
	if (!vertex_function || !output_var || !float_type) {
		return p_bytes;
	}

	uint32_t next_id = code.word(3);
	const bool new_int_type = !int_type;
	if (new_int_type) {
		int_type = next_id++;
	}
	const uint32_t ptr_float = next_id++;
	const bool member = !position_var;
	const uint32_t *existing_member = small_int_constants.getptr(position_member);
	const uint32_t member_constant = !member ? 0 : existing_member ? *existing_member
																   : next_id++;
	const uint32_t *existing_one = small_int_constants.getptr(1);
	const uint32_t one_constant = existing_one ? *existing_one : next_id++;

	Vector<uint8_t> out;
	append_bytes(out, code.data, 0, 20);
	bool types_injected = false;
	bool in_vertex_function = false;
	for (const Inst inst : code) {
		const uint16_t op = inst.op();
		if (op == spv::OpFunction) {
			if (!types_injected) {
				types_injected = true;
				if (new_int_type) {
					push_inst(out, spv::OpTypeInt, { int_type, 32, 0 });
				}
				push_inst(out, spv::OpTypePointer, { ptr_float, spv::StorageClassOutput, float_type });
				if (member && !existing_member) {
					push_inst(out, spv::OpConstant, { int_type, member_constant, position_member });
				}
				if (!existing_one) {
					push_inst(out, spv::OpConstant, { int_type, one_constant, 1 });
				}
			}
			in_vertex_function = inst[2] == vertex_function;
		} else if (op == spv::OpFunctionEnd) {
			in_vertex_function = false;
		} else if (in_vertex_function && (op == spv::OpReturn || op == spv::OpReturnValue)) {
			const uint32_t y = next_id++;
			const uint32_t loaded = next_id++;
			const uint32_t negated = next_id++;
			if (member) {
				push_inst(out, spv::OpAccessChain, { ptr_float, y, output_var, member_constant, one_constant });
			} else {
				push_inst(out, spv::OpAccessChain, { ptr_float, y, output_var, one_constant });
			}
			push_inst(out, spv::OpLoad, { float_type, loaded, y });
			push_inst(out, spv::OpFNegate, { float_type, negated, loaded });
			push_inst(out, spv::OpStore, { y, negated });
		}
		append_inst(out, inst);
	}
	write_word(out.ptrw(), 3, next_id);
	return out;
}

Vector<uint8_t> strip_restrict_decoration(const Vector<uint8_t> &p_bytes) {
	ERR_FAIL_COND_V_MSG(!is_well_formed(p_bytes), p_bytes, MALFORMED_SPIRV);
	return strip_instructions(p_bytes, [](const Inst &p_inst) {
		const uint32_t decoration = p_inst.op() == spv::OpDecorate ? p_inst[2] : p_inst.op() == spv::OpMemberDecorate ? p_inst[3]
																													  : UINT32_MAX;
		return decoration == spv::DecorationRestrict || decoration == spv::DecorationVolatile || decoration == spv::DecorationInputAttachmentIndex;
	});
}

Vector<uint8_t> strip_memory_barrier(const Vector<uint8_t> &p_bytes) {
	ERR_FAIL_COND_V_MSG(!is_well_formed(p_bytes), p_bytes, MALFORMED_SPIRV);
	return strip_instructions(p_bytes, [](const Inst &p_inst) { return p_inst.op() == spv::OpMemoryBarrier; });
}

Vector<uint8_t> fix_nonfinite_literals(const Vector<uint8_t> &p_bytes) {
	ERR_FAIL_COND_V_MSG(!is_well_formed(p_bytes), p_bytes, MALFORMED_SPIRV);
	HashMap<uint32_t, uint32_t> float_widths;
	Vector<uint8_t> out = p_bytes; // Copied on the first write.
	for (const Inst inst : SpirvView(p_bytes)) {
		if (inst.op() == spv::OpTypeFloat) {
			float_widths.insert(inst[1], inst[2]);
			continue;
		}
		if ((inst.op() != spv::OpConstant && inst.op() != spv::OpSpecConstant) || inst.count() < 4) {
			continue;
		}
		const uint32_t *width = float_widths.getptr(inst[1]);
		if (width && *width == 32) {
			float value;
			const uint32_t bits = inst[3];
			memcpy(&value, &bits, 4);
			if (!std::isfinite(value)) {
				const float replacement = std::signbit(value) ? -FLT_MAX : FLT_MAX;
				uint32_t replacement_bits;
				memcpy(&replacement_bits, &replacement, 4);
				write_word(out.ptrw(), inst.pos + 3, replacement_bits);
			}
		} else if (width && *width == 64 && inst.count() >= 5) {
			double value;
			const uint64_t bits = inst[3] | (uint64_t(inst[4]) << 32);
			memcpy(&value, &bits, 8);
			if (!std::isfinite(value)) {
				const double replacement = std::signbit(value) ? -DBL_MAX : DBL_MAX;
				uint64_t replacement_bits;
				memcpy(&replacement_bits, &replacement, 8);
				write_word(out.ptrw(), inst.pos + 3, uint32_t(replacement_bits));
				write_word(out.ptrw(), inst.pos + 4, uint32_t(replacement_bits >> 32));
			}
		}
	}
	return out;
}

Vector<uint8_t> normalize_storage_buffer_access(const Vector<uint8_t> &p_bytes) {
	ERR_FAIL_COND_V_MSG(!is_well_formed(p_bytes), p_bytes, MALFORMED_SPIRV);
	HashSet<uint32_t> blocks;
	HashSet<uint32_t> variables;
	for (const Inst inst : SpirvView(p_bytes)) {
		if (inst.op() == spv::OpTypePointer && inst[2] == spv::StorageClassStorageBuffer) {
			blocks.insert(inst[3]);
		} else if (inst.op() == spv::OpVariable && inst[3] == spv::StorageClassStorageBuffer) {
			variables.insert(inst[2]);
		}
	}
	return strip_instructions(p_bytes, [&](const Inst &p_inst) {
		return (p_inst.op() == spv::OpDecorate && p_inst[2] == spv::DecorationNonReadable && variables.has(p_inst[1])) ||
				(p_inst.op() == spv::OpMemberDecorate && p_inst[3] == spv::DecorationNonReadable && blocks.has(p_inst[1]));
	});
}

// Follows each written pointer back through the instructions that derive pointers to the StorageBuffer
// variable it came from. Function arguments count as writes.
Vector<uint8_t> infer_readonly_storage(const Vector<uint8_t> &p_bytes) {
	ERR_FAIL_COND_V_MSG(!is_well_formed(p_bytes), p_bytes, MALFORMED_SPIRV);
	const SpirvView code(p_bytes);

	Vector<uint32_t> storage_vars;
	HashSet<uint32_t> non_writable;
	HashMap<uint32_t, Vector<uint32_t>> sources; // Derived value -> values it may come from.
	Vector<uint32_t> written;
	for (const Inst inst : code) {
		switch (inst.op()) {
			case spv::OpVariable:
				if (inst[3] == spv::StorageClassStorageBuffer) {
					storage_vars.push_back(inst[2]);
				}
				break;
			case spv::OpDecorate:
				if (inst[2] == spv::DecorationNonWritable) {
					non_writable.insert(inst[1]);
				}
				break;
			case spv::OpAccessChain:
			case spv::OpInBoundsAccessChain:
			case spv::OpPtrAccessChain:
			case spv::OpInBoundsPtrAccessChain:
			case spv::OpCopyObject:
				sources[inst[2]].push_back(inst[3]);
				break;
			case spv::OpSelect:
				sources[inst[2]].push_back(inst[4]);
				sources[inst[2]].push_back(inst[5]);
				break;
			case spv::OpPhi:
				for (uint32_t i = 3; i < inst.count(); i += 2) {
					sources[inst[2]].push_back(inst[i]);
				}
				break;
			case spv::OpStore:
			case spv::OpCopyMemory:
			case spv::OpCopyMemorySized:
			case spv::OpAtomicStore:
			case spv::OpAtomicFlagClear:
				written.push_back(inst[1]);
				break;
			case spv::OpAtomicExchange:
			case spv::OpAtomicCompareExchange:
			case spv::OpAtomicCompareExchangeWeak:
			case spv::OpAtomicIIncrement:
			case spv::OpAtomicIDecrement:
			case spv::OpAtomicIAdd:
			case spv::OpAtomicISub:
			case spv::OpAtomicSMin:
			case spv::OpAtomicUMin:
			case spv::OpAtomicSMax:
			case spv::OpAtomicUMax:
			case spv::OpAtomicAnd:
			case spv::OpAtomicOr:
			case spv::OpAtomicXor:
			case spv::OpAtomicFlagTestAndSet:
			case spv::OpAtomicFMinEXT:
			case spv::OpAtomicFMaxEXT:
			case spv::OpAtomicFAddEXT:
				written.push_back(inst[3]);
				break;
			case spv::OpFunctionCall:
				for (uint32_t i = 4; i < inst.count(); i++) {
					written.push_back(inst[i]);
				}
				break;
			default:
				break;
		}
	}

	HashSet<uint32_t> visited;
	while (!written.is_empty()) {
		const uint32_t value = written[written.size() - 1];
		written.resize(written.size() - 1);
		if (visited.has(value)) {
			continue;
		}
		visited.insert(value);
		if (const Vector<uint32_t> *from = sources.getptr(value)) {
			for (uint32_t source : *from) {
				written.push_back(source);
			}
		}
	}
	Vector<uint32_t> read_only;
	for (uint32_t var : storage_vars) {
		if (!visited.has(var) && !non_writable.has(var)) {
			read_only.push_back(var);
		}
	}
	if (read_only.is_empty()) {
		return p_bytes;
	}

	Vector<uint8_t> out;
	append_bytes(out, code.data, 0, 20);
	bool inserted = false;
	for (const Inst inst : code) {
		if (!inserted && !is_before_annotations(inst.op())) {
			inserted = true;
			for (uint32_t var : read_only) {
				push_inst(out, spv::OpDecorate, { var, spv::DecorationNonWritable });
			}
		}
		append_inst(out, inst);
	}
	return out;
}

} // namespace spirv_preprocess
