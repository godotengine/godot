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

#ifndef SPIRV_CROSS_PARSER_HPP
#define SPIRV_CROSS_PARSER_HPP

#include "spirv_cross_parsed_ir.hpp"
#include <stdint.h>

namespace SPIRV_CROSS_NAMESPACE
{
class Parser
{
public:
	Parser(const uint32_t *spirv_data, size_t word_count);
	Parser(std::vector<uint32_t> spirv);

	void parse();

	ParsedIR &get_parsed_ir()
	{
		return ir;
	}

private:
	ParsedIR ir;
	SPIRFunction *current_function = nullptr;
	SPIRBlock *current_block = nullptr;

	void parse(const Instruction &instr);
	const uint32_t *stream(const Instruction &instr) const;

	template <typename T, typename... P>
	T &set(uint32_t id, P &&... args)
	{
		ir.add_typed_id(static_cast<Types>(T::type), id);
		auto &var = variant_set<T>(ir.ids[id], std::forward<P>(args)...);
		var.self = id;
		return var;
	}

	template <typename T>
	T &get(uint32_t id)
	{
		return variant_get<T>(ir.ids[id]);
	}

	template <typename T>
	T *maybe_get(uint32_t id)
	{
		if (ir.ids[id].get_type() == static_cast<Types>(T::type))
			return &get<T>(id);
		else
			return nullptr;
	}

	template <typename T>
	const T &get(uint32_t id) const
	{
		return variant_get<T>(ir.ids[id]);
	}

	template <typename T>
	const T *maybe_get(uint32_t id) const
	{
		if (ir.ids[id].get_type() == T::type)
			return &get<T>(id);
		else
			return nullptr;
	}

	// This must be an ordered data structure so we always pick the same type aliases.
	SmallVector<uint32_t> global_struct_cache;
	SmallVector<std::pair<uint32_t, uint32_t>> forward_pointer_fixups;

	bool types_are_logically_equivalent(const SPIRType &a, const SPIRType &b) const;
	bool variable_storage_is_aliased(const SPIRVariable &v) const;
};
} // namespace SPIRV_CROSS_NAMESPACE

#endif
