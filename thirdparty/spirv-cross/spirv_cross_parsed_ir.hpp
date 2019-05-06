/*
 * Copyright 2018-2019 Arm Limited
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

#ifndef SPIRV_CROSS_PARSED_IR_HPP
#define SPIRV_CROSS_PARSED_IR_HPP

#include "spirv_common.hpp"
#include <stdint.h>
#include <unordered_map>

namespace SPIRV_CROSS_NAMESPACE
{

// This data structure holds all information needed to perform cross-compilation and reflection.
// It is the output of the Parser, but any implementation could create this structure.
// It is intentionally very "open" and struct-like with some helper functions to deal with decorations.
// Parser is the reference implementation of how this data structure should be filled in.

class ParsedIR
{
private:
	// This must be destroyed after the "ids" vector.
	std::unique_ptr<ObjectPoolGroup> pool_group;

public:
	ParsedIR();

	// Due to custom allocations from object pools, we cannot use a default copy constructor.
	ParsedIR(const ParsedIR &other);
	ParsedIR &operator=(const ParsedIR &other);

	// Moves are unproblematic, but we need to implement it anyways, since MSVC 2013 does not understand
	// how to default-implement these.
	ParsedIR(ParsedIR &&other) SPIRV_CROSS_NOEXCEPT;
	ParsedIR &operator=(ParsedIR &&other) SPIRV_CROSS_NOEXCEPT;

	// Resizes ids, meta and block_meta.
	void set_id_bounds(uint32_t bounds);

	// The raw SPIR-V, instructions and opcodes refer to this by offset + count.
	std::vector<uint32_t> spirv;

	// Holds various data structures which inherit from IVariant.
	SmallVector<Variant> ids;

	// Various meta data for IDs, decorations, names, etc.
	std::unordered_map<uint32_t, Meta> meta;

	// Holds all IDs which have a certain type.
	// This is needed so we can iterate through a specific kind of resource quickly,
	// and in-order of module declaration.
	SmallVector<uint32_t> ids_for_type[TypeCount];

	// Special purpose lists which contain a union of types.
	// This is needed so we can declare specialization constants and structs in an interleaved fashion,
	// among other things.
	// Constants can be of struct type, and struct array sizes can use specialization constants.
	SmallVector<uint32_t> ids_for_constant_or_type;
	SmallVector<uint32_t> ids_for_constant_or_variable;

	// Declared capabilities and extensions in the SPIR-V module.
	// Not really used except for reflection at the moment.
	SmallVector<spv::Capability> declared_capabilities;
	SmallVector<std::string> declared_extensions;

	// Meta data about blocks. The cross-compiler needs to query if a block is either of these types.
	// It is a bitset as there can be more than one tag per block.
	enum BlockMetaFlagBits
	{
		BLOCK_META_LOOP_HEADER_BIT = 1 << 0,
		BLOCK_META_CONTINUE_BIT = 1 << 1,
		BLOCK_META_LOOP_MERGE_BIT = 1 << 2,
		BLOCK_META_SELECTION_MERGE_BIT = 1 << 3,
		BLOCK_META_MULTISELECT_MERGE_BIT = 1 << 4
	};
	using BlockMetaFlags = uint8_t;
	SmallVector<BlockMetaFlags> block_meta;
	std::unordered_map<uint32_t, uint32_t> continue_block_to_loop_header;

	// Normally, we'd stick SPIREntryPoint in ids array, but it conflicts with SPIRFunction.
	// Entry points can therefore be seen as some sort of meta structure.
	std::unordered_map<uint32_t, SPIREntryPoint> entry_points;
	uint32_t default_entry_point = 0;

	struct Source
	{
		uint32_t version = 0;
		bool es = false;
		bool known = false;
		bool hlsl = false;

		Source() = default;
	};

	Source source;

	spv::AddressingModel addressing_model = spv::AddressingModelMax;
	spv::MemoryModel memory_model = spv::MemoryModelMax;

	// Decoration handling methods.
	// Can be useful for simple "raw" reflection.
	// However, most members are here because the Parser needs most of these,
	// and might as well just have the whole suite of decoration/name handling in one place.
	void set_name(uint32_t id, const std::string &name);
	const std::string &get_name(uint32_t id) const;
	void set_decoration(uint32_t id, spv::Decoration decoration, uint32_t argument = 0);
	void set_decoration_string(uint32_t id, spv::Decoration decoration, const std::string &argument);
	bool has_decoration(uint32_t id, spv::Decoration decoration) const;
	uint32_t get_decoration(uint32_t id, spv::Decoration decoration) const;
	const std::string &get_decoration_string(uint32_t id, spv::Decoration decoration) const;
	const Bitset &get_decoration_bitset(uint32_t id) const;
	void unset_decoration(uint32_t id, spv::Decoration decoration);

	// Decoration handling methods (for members of a struct).
	void set_member_name(uint32_t id, uint32_t index, const std::string &name);
	const std::string &get_member_name(uint32_t id, uint32_t index) const;
	void set_member_decoration(uint32_t id, uint32_t index, spv::Decoration decoration, uint32_t argument = 0);
	void set_member_decoration_string(uint32_t id, uint32_t index, spv::Decoration decoration,
	                                  const std::string &argument);
	uint32_t get_member_decoration(uint32_t id, uint32_t index, spv::Decoration decoration) const;
	const std::string &get_member_decoration_string(uint32_t id, uint32_t index, spv::Decoration decoration) const;
	bool has_member_decoration(uint32_t id, uint32_t index, spv::Decoration decoration) const;
	const Bitset &get_member_decoration_bitset(uint32_t id, uint32_t index) const;
	void unset_member_decoration(uint32_t id, uint32_t index, spv::Decoration decoration);

	void mark_used_as_array_length(uint32_t id);
	uint32_t increase_bound_by(uint32_t count);
	Bitset get_buffer_block_flags(const SPIRVariable &var) const;

	void add_typed_id(Types type, uint32_t id);
	void remove_typed_id(Types type, uint32_t id);

	template <typename T, typename Op>
	void for_each_typed_id(const Op &op)
	{
		loop_iteration_depth++;
		for (auto &id : ids_for_type[T::type])
		{
			if (ids[id].get_type() == static_cast<Types>(T::type))
				op(id, get<T>(id));
		}
		loop_iteration_depth--;
	}

	template <typename T, typename Op>
	void for_each_typed_id(const Op &op) const
	{
		for (auto &id : ids_for_type[T::type])
		{
			if (ids[id].get_type() == static_cast<Types>(T::type))
				op(id, get<T>(id));
		}
	}

	template <typename T>
	void reset_all_of_type()
	{
		reset_all_of_type(static_cast<Types>(T::type));
	}

	void reset_all_of_type(Types type);

	Meta *find_meta(uint32_t id);
	const Meta *find_meta(uint32_t id) const;

	const std::string &get_empty_string() const
	{
		return empty_string;
	}

private:
	template <typename T>
	T &get(uint32_t id)
	{
		return variant_get<T>(ids[id]);
	}

	template <typename T>
	const T &get(uint32_t id) const
	{
		return variant_get<T>(ids[id]);
	}

	uint32_t loop_iteration_depth = 0;
	std::string empty_string;
	Bitset cleared_bitset;
};
} // namespace SPIRV_CROSS_NAMESPACE

#endif
