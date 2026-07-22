/*
 * Copyright 2016-2021 Arm Limited
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

#ifndef SPIRV_CROSS_CFG_HPP
#define SPIRV_CROSS_CFG_HPP

#include "spirv_common.hpp"
#include <assert.h>

namespace SPIRV_CROSS_NAMESPACE
{
class Compiler;
class CFG
{
public:
	CFG(Compiler &compiler, const SPIRFunction &function);

	Compiler &get_compiler()
	{
		return compiler;
	}

	const Compiler &get_compiler() const
	{
		return compiler;
	}

	const SPIRFunction &get_function() const
	{
		return func;
	}

	uint32_t get_immediate_dominator(uint32_t block) const
	{
		auto itr = immediate_dominators.find(block);
		if (itr != std::end(immediate_dominators))
			return itr->second;
		else
			return 0;
	}

	bool is_reachable(uint32_t block) const
	{
		return visit_order.count(block) != 0;
	}

	uint32_t get_visit_order(uint32_t block) const
	{
		auto itr = visit_order.find(block);
		assert(itr != std::end(visit_order));
		int v = itr->second.order;
		assert(v > 0);
		return uint32_t(v);
	}

	uint32_t find_common_dominator(uint32_t a, uint32_t b) const;

	const SmallVector<uint32_t> &get_preceding_edges(uint32_t block) const
	{
		auto itr = preceding_edges.find(block);
		if (itr != std::end(preceding_edges))
			return itr->second;
		else
			return empty_vector;
	}

	const SmallVector<uint32_t> &get_succeeding_edges(uint32_t block) const
	{
		auto itr = succeeding_edges.find(block);
		if (itr != std::end(succeeding_edges))
			return itr->second;
		else
			return empty_vector;
	}

	template <typename Op>
	void walk_from(std::unordered_set<uint32_t> &seen_blocks, uint32_t block, const Op &op) const
	{
		if (seen_blocks.count(block))
			return;
		seen_blocks.insert(block);

		if (op(block))
		{
			for (auto b : get_succeeding_edges(block))
				walk_from(seen_blocks, b, op);
		}
	}

	uint32_t find_loop_dominator(uint32_t block) const;

	bool node_terminates_control_flow_in_sub_graph(BlockID from, BlockID to) const;

private:
	struct VisitOrder
	{
		int order = -1;
		bool visited_resolve = false;
		bool visited_branches = false;
	};

	Compiler &compiler;
	const SPIRFunction &func;
	std::unordered_map<uint32_t, SmallVector<uint32_t>> preceding_edges;
	std::unordered_map<uint32_t, SmallVector<uint32_t>> succeeding_edges;
	std::unordered_map<uint32_t, uint32_t> immediate_dominators;
	std::unordered_map<uint32_t, VisitOrder> visit_order;
	SmallVector<uint32_t> post_order;
	SmallVector<uint32_t> empty_vector;

	void add_branch(uint32_t from, uint32_t to);
	void build_post_order_visit_order();
	void build_immediate_dominators();
	void post_order_visit_branches(uint32_t block);
	void post_order_visit_resolve(uint32_t block);
	void post_order_visit_entry(uint32_t block);
	uint32_t visit_count = 0;

	bool is_back_edge(uint32_t to) const;
	bool has_visited_branch(uint32_t to) const;
	void visit_branch(uint32_t block_id);

	SmallVector<uint32_t> visit_stack;
	size_t last_visited_size = 0;
};

class DominatorBuilder
{
public:
	DominatorBuilder(const CFG &cfg);

	void add_block(uint32_t block);
	uint32_t get_dominator() const
	{
		return dominator;
	}

	void lift_continue_block_dominator();

private:
	const CFG &cfg;
	uint32_t dominator = 0;
};
} // namespace SPIRV_CROSS_NAMESPACE

#endif
