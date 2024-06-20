/**************************************************************************/
/*  bvh_tree.h                                                            */
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

#ifndef BVH_TREE_H
#define BVH_TREE_H

// BVH Tree
// This is an implementation of a dynamic BVH with templated leaf size.
// This differs from most dynamic BVH in that it can handle more than 1 object
// in leaf nodes. This can make it far more efficient in certain circumstances.
// It also means that the splitting logic etc have to be completely different
// to a simpler tree.
// Note that MAX_CHILDREN should be fixed at 2 for now.

#include "core/math/aabb.h"
#include "core/math/bvh_abb.h"
#include "core/math/geometry_3d.h"
#include "core/math/vector3.h"
#include "core/templates/local_vector.h"
#include "core/templates/pooled_list.h"
#include <limits.h>

#define BVHABB_CLASS BVH_ABB<BOUNDS, POINT>

// not sure if this is better yet so making optional
#define BVH_EXPAND_LEAF_AABBS

// never do these checks in release
#ifdef DEV_ENABLED
//#define BVH_VERBOSE
//#define BVH_VERBOSE_TREE
//#define BVH_VERBOSE_PAIRING
//#define BVH_VERBOSE_MOVES

//#define BVH_VERBOSE_FRAME
//#define BVH_CHECKS
//#define BVH_INTEGRITY_CHECKS
#endif

// debug only assert
#ifdef BVH_CHECKS
#define BVH_ASSERT(a) CRASH_COND((a) == false)
#else
#define BVH_ASSERT(a)
#endif

#ifdef BVH_VERBOSE
#define VERBOSE_PRINT print_line
#else
#define VERBOSE_PRINT(a)
#endif

// really just a namespace
struct BVHCommon {
	// these could possibly also be the same constant,
	// although this may be useful for debugging.
	// or use zero for invalid and +1 based indices.
	static const uint32_t INVALID = (0xffffffff);
	static const uint32_t INACTIVE = (0xfffffffe);
};

// really a handle, can be anything
// note that zero is a valid reference for the BVH .. this may involve using
// a plus one based ID for clients that expect 0 to be invalid.
struct BVHHandle {
	// conversion operator
	operator uint32_t() const { return _data; }
	void set(uint32_t p_value) { _data = p_value; }

	uint32_t _data;

	void set_invalid() { _data = BVHCommon::INVALID; }
	bool is_invalid() const { return _data == BVHCommon::INVALID; }
	uint32_t id() const { return _data; }
	void set_id(uint32_t p_id) { _data = p_id; }

	bool operator==(const BVHHandle &p_h) const { return _data == p_h._data; }
	bool operator!=(const BVHHandle &p_h) const { return (*this == p_h) == false; }
};

// helper class to make iterative versions of recursive functions
template <typename T>
class BVH_IterativeInfo {
public:
	enum {
		ALLOCA_STACK_SIZE = 128
	};

	int32_t depth = 1;
	int32_t threshold = ALLOCA_STACK_SIZE - 2;
	T *stack;
	//only used in rare occasions when you run out of alloca memory
	// because tree is too unbalanced.
	LocalVector<T> aux_stack;
	int32_t get_alloca_stacksize() const { return ALLOCA_STACK_SIZE * sizeof(T); }

	T *get_first() const {
		return &stack[0];
	}

	// pop the last member of the stack, or return false
	bool pop(T &r_value) {
		if (!depth) {
			return false;
		}

		depth--;
		r_value = stack[depth];
		return true;
	}

	// request new addition to stack
	T *request() {
		if (depth > threshold) {
			if (aux_stack.is_empty()) {
				aux_stack.resize(ALLOCA_STACK_SIZE * 2);
				memcpy(aux_stack.ptr(), stack, get_alloca_stacksize());
			} else {
				aux_stack.resize(aux_stack.size() * 2);
			}
			stack = aux_stack.ptr();
			threshold = aux_stack.size() - 2;
		}
		return &stack[depth++];
	}
};

template <typename T>
class BVH_DummyPairTestFunction {
public:
	static bool user_collision_check(T *p_a, T *p_b) {
		// return false if no collision, decided by masks etc
		return true;
	}
};

template <typename T>
class BVH_DummyCullTestFunction {
public:
	static bool user_cull_check(T *p_a, T *p_b) {
		// return false if no collision
		return true;
	}
};

template <typename T, int NUM_TREES, int MAX_CHILDREN, int MAX_ITEMS, typename USER_PAIR_TEST_FUNCTION = BVH_DummyPairTestFunction<T>, typename USER_CULL_TEST_FUNCTION = BVH_DummyCullTestFunction<T>, bool USE_PAIRS = false, typename BOUNDS = AABB, typename POINT = Vector3>
class BVH_Tree {
	friend class BVH;

public:
#pragma region Pair
	// note .. maybe this can be attached to another node structure?
	// depends which works best for cache.
	struct ItemPairs {
		struct Link {
			void set(BVHHandle h, void *ud) {
				handle = h;
				userdata = ud;
			}
			BVHHandle handle;
			void *userdata;
		};

		void clear() {
			num_pairs = 0;
			extended_pairs.reset();
			expanded_aabb = BOUNDS();
		}

		BOUNDS expanded_aabb;

		// maybe we can just use the number in the vector TODO
		int32_t num_pairs;
		LocalVector<Link> extended_pairs;

		void add_pair_to(BVHHandle h, void *p_userdata) {
			Link temp;
			temp.set(h, p_userdata);

			extended_pairs.push_back(temp);
			num_pairs++;
		}

		uint32_t find_pair_to(BVHHandle h) const {
			for (int n = 0; n < num_pairs; n++) {
				if (extended_pairs[n].handle == h) {
					return n;
				}
			}
			return -1;
		}

		bool contains_pair_to(BVHHandle h) const {
			return find_pair_to(h) != BVHCommon::INVALID;
		}

		// return success
		void *remove_pair_to(BVHHandle h) {
			void *userdata = nullptr;

			for (int n = 0; n < num_pairs; n++) {
				if (extended_pairs[n].handle == h) {
					userdata = extended_pairs[n].userdata;
					extended_pairs.remove_at_unordered(n);
					num_pairs--;
					break;
				}
			}

			return userdata;
		}

		// experiment : scale the pairing expansion by the number of pairs.
		// when the number of pairs is high, the density is high and a lower collision margin is better.
		// when there are few local pairs, a larger margin is more optimal.
		real_t scale_expansion_margin(real_t p_margin) const {
			real_t x = real_t(num_pairs) * (1.0 / 9.0);
			x = MIN(x, 1.0);
			x = 1.0 - x;
			return p_margin * x;
		}
	};
#pragma endregion Pair

#pragma region Structs
	struct ItemRef {
		uint32_t tnode_id; // -1 is invalid
		uint32_t item_id; // in the leaf

		bool is_active() const { return tnode_id != BVHCommon::INACTIVE; }
		void set_inactive() {
			tnode_id = BVHCommon::INACTIVE;
			item_id = BVHCommon::INACTIVE;
		}
	};

	// extra info kept in separate parallel list to the references,
	// as this is less used as keeps cache better
	struct ItemExtra {
		// Before doing user defined pairing checks (especially in the find_leavers function),
		// we may want to check that two items have compatible tree ids and tree masks,
		// as if they are incompatible they should not pair / collide.
		bool are_item_trees_compatible(const ItemExtra &p_other) const {
			uint32_t other_type = 1 << p_other.tree_id;
			if (tree_collision_mask & other_type) {
				return true;
			}
			uint32_t our_type = 1 << tree_id;
			if (p_other.tree_collision_mask & our_type) {
				return true;
			}
			return false;
		}

		// There can be multiple user defined trees
		uint32_t tree_id;

		// Defines which trees this item should collision check against.
		// 1 << tree_id, and normally items would collide against there own
		// tree (but not always).
		uint32_t tree_collision_mask;

		uint32_t last_updated_tick;
		int32_t subindex;

		T *userdata;

		// the active reference is a separate list of which references
		// are active so that we can slowly iterate through it over many frames for
		// slow optimize.
		uint32_t active_ref_id;
	};

	// tree leaf
	struct TLeaf {
		uint16_t num_items;

	private:
		uint16_t dirty;
		// separate data orientated lists for faster SIMD traversal
		uint32_t item_ref_ids[MAX_ITEMS];
		BVHABB_CLASS aabbs[MAX_ITEMS];

	public:
		// accessors
		BVHABB_CLASS &get_aabb(uint32_t p_id) {
			BVH_ASSERT(p_id < MAX_ITEMS);
			return aabbs[p_id];
		}
		const BVHABB_CLASS &get_aabb(uint32_t p_id) const {
			BVH_ASSERT(p_id < MAX_ITEMS);
			return aabbs[p_id];
		}

		uint32_t &get_item_ref_id(uint32_t p_id) {
			BVH_ASSERT(p_id < MAX_ITEMS);
			return item_ref_ids[p_id];
		}
		const uint32_t &get_item_ref_id(uint32_t p_id) const {
			BVH_ASSERT(p_id < MAX_ITEMS);
			return item_ref_ids[p_id];
		}

		bool is_dirty() const { return dirty; }
		void set_dirty(bool p) { dirty = p; }

		void clear() {
			num_items = 0;
			set_dirty(false);
		}
		bool is_full() const { return num_items >= MAX_ITEMS; }

		void remove_item_unordered(uint32_t p_id) {
			BVH_ASSERT(p_id < num_items);
			num_items--;
			aabbs[p_id] = aabbs[num_items];
			item_ref_ids[p_id] = item_ref_ids[num_items];
		}

		uint32_t request_item() {
			if (num_items < MAX_ITEMS) {
				uint32_t id = num_items;
				num_items++;
				return id;
			}
#ifdef DEV_ENABLED
			return -1;
#else
			ERR_FAIL_V_MSG(0, "BVH request_item error.");
#endif
		}
	};

	// tree node
	struct TNode {
		BVHABB_CLASS aabb;
		// either number of children if positive
		// or leaf id if negative (leaf id 0 is disallowed)
		union {
			int32_t num_children;
			int32_t neg_leaf_id;
		};
		uint32_t parent_id; // or -1
		uint16_t children[MAX_CHILDREN];

		// height in the tree, where leaves are 0, and all above are 1+
		// (or the highest where there is a tie off)
		int32_t height;

		bool is_leaf() const { return num_children < 0; }
		void set_leaf_id(int id) { neg_leaf_id = -id; }
		int get_leaf_id() const { return -neg_leaf_id; }

		void clear() {
			num_children = 0;
			parent_id = BVHCommon::INVALID;
			height = 0; // or -1 for testing

			// for safety set to improbable value
			aabb.set_to_max_opposite_extents();

			// other members are not blanked for speed .. they may be uninitialized
		}

		bool is_full_of_children() const { return num_children >= MAX_CHILDREN; }

		void remove_child_internal(uint32_t child_num) {
			children[child_num] = children[num_children - 1];
			num_children--;
		}

		int find_child(uint32_t p_child_node_id) {
			BVH_ASSERT(!is_leaf());

			for (int n = 0; n < num_children; n++) {
				if (children[n] == p_child_node_id) {
					return n;
				}
			}

			// not found
			return -1;
		}
	};

	// instead of using linked list we maintain
	// item references (for quick lookup)
	PooledList<ItemRef, uint32_t, true> _refs;
	PooledList<ItemExtra, uint32_t, true> _extra;
	PooledList<ItemPairs> _pairs;

	// these 2 are not in sync .. nodes != leaves!
	PooledList<TNode, uint32_t, true> _nodes;
	PooledList<TLeaf, uint32_t, true> _leaves;

	// we can maintain an un-ordered list of which references are active,
	// in order to do a slow incremental optimize of the tree over each frame.
	// This will work best if dynamic objects and static objects are in a different tree.
	LocalVector<uint32_t, uint32_t, true> _active_refs;
	uint32_t _current_active_ref = 0;

	// instead of translating directly to the userdata output,
	// we keep an intermediate list of hits as reference IDs, which can be used
	// for pairing collision detection
	LocalVector<uint32_t, uint32_t, true> _cull_hits;

	// We can now have a user definable number of trees.
	// This allows using e.g. a non-pairable and pairable tree,
	// which can be more efficient for example, if we only need check non pairable against the pairable tree.
	// It also may be more efficient in terms of separating static from dynamic objects, by reducing housekeeping.
	// However this is a trade off, as there is a cost of traversing two trees.
	uint32_t _root_node_id[NUM_TREES];

	// these values may need tweaking according to the project
	// the bound of the world, and the average velocities of the objects

	// node expansion is important in the rendering tree
	// larger values give less re-insertion as items move...
	// but on the other hand over estimates the bounding box of nodes.
	// we can either use auto mode, where the expansion is based on the root node size, or specify manually
	real_t _node_expansion = 0.5;
	bool _auto_node_expansion = true;

	// pairing expansion important for physics pairing
	// larger values gives more 'sticky' pairing, and is less likely to exhibit tunneling
	// we can either use auto mode, where the expansion is based on the root node size, or specify manually
	real_t _pairing_expansion = 0.1;

#ifdef BVH_ALLOW_AUTO_EXPANSION
	bool _auto_pairing_expansion = true;
#endif

	// when using an expanded bound, we must detect the condition where a new AABB
	// is significantly smaller than the expanded bound, as this is a special case where we
	// should override the optimization and create a new expanded bound.
	// This threshold is derived from the _pairing_expansion, and should be recalculated
	// if _pairing_expansion is changed.
	real_t _aabb_shrinkage_threshold = 0.0;
#pragma endregion Structs

	BVH_Tree() {
		for (int n = 0; n < NUM_TREES; n++) {
			_root_node_id[n] = BVHCommon::INVALID;
		}

		// disallow zero leaf ids
		// (as these ids are stored as negative numbers in the node)
		uint32_t dummy_leaf_id;
		_leaves.request(dummy_leaf_id);

		// In many cases you may want to change this default in the client code,
		// or expose this value to the user.
		// This default may make sense for a typically scaled 3d game, but maybe not for 2d on a pixel scale.
		params_set_pairing_expansion(0.1);
	}

private:
	bool node_add_child(uint32_t p_node_id, uint32_t p_child_node_id) {
		TNode &tnode = _nodes[p_node_id];
		if (tnode.is_full_of_children()) {
			return false;
		}

		tnode.children[tnode.num_children] = p_child_node_id;
		tnode.num_children += 1;

		// back link in the child to the parent
		TNode &tnode_child = _nodes[p_child_node_id];
		tnode_child.parent_id = p_node_id;

		return true;
	}

	void node_replace_child(uint32_t p_parent_id, uint32_t p_old_child_id, uint32_t p_new_child_id) {
		TNode &parent = _nodes[p_parent_id];
		BVH_ASSERT(!parent.is_leaf());

		int child_num = parent.find_child(p_old_child_id);
		BVH_ASSERT(child_num != -1);
		parent.children[child_num] = p_new_child_id;

		TNode &new_child = _nodes[p_new_child_id];
		new_child.parent_id = p_parent_id;
	}

	void node_remove_child(uint32_t p_parent_id, uint32_t p_child_id, uint32_t p_tree_id, bool p_prevent_sibling = false) {
		TNode &parent = _nodes[p_parent_id];
		BVH_ASSERT(!parent.is_leaf());

		int child_num = parent.find_child(p_child_id);
		BVH_ASSERT(child_num != -1);

		parent.remove_child_internal(child_num);

		// no need to keep back references for children at the moment

		uint32_t sibling_id = 0; // always a node id, as tnode is never a leaf
		bool sibling_present = false;

		// if there are more children, or this is the root node, don't try and delete
		if (parent.num_children > 1) {
			return;
		}

		// if there is 1 sibling, it can be moved to be a child of the
		if (parent.num_children == 1) {
			// else there is now a redundant node with one child, which can be removed
			sibling_id = parent.children[0];
			sibling_present = true;
		}

		// now there may be no children in this node .. in which case it can be deleted
		// remove node if empty
		// remove link from parent
		uint32_t grandparent_id = parent.parent_id;

		// special case for root node
		if (grandparent_id == BVHCommon::INVALID) {
			if (sibling_present) {
				// change the root node
				change_root_node(sibling_id, p_tree_id);

				// delete the old root node as no longer needed
				node_free_node_and_leaf(p_parent_id);
			}

			return;
		}

		if (sibling_present) {
			node_replace_child(grandparent_id, p_parent_id, sibling_id);
		} else {
			node_remove_child(grandparent_id, p_parent_id, p_tree_id, true);
		}

		// put the node on the free list to recycle
		node_free_node_and_leaf(p_parent_id);
	}

	// A node can either be a node, or a node AND a leaf combo.
	// Both must be deleted to prevent a leak.
	void node_free_node_and_leaf(uint32_t p_node_id) {
		TNode &node = _nodes[p_node_id];
		if (node.is_leaf()) {
			int leaf_id = node.get_leaf_id();
			_leaves.free(leaf_id);
		}

		_nodes.free(p_node_id);
	}

	void change_root_node(uint32_t p_new_root_id, uint32_t p_tree_id) {
		_root_node_id[p_tree_id] = p_new_root_id;
		TNode &root = _nodes[p_new_root_id];

		// mark no parent
		root.parent_id = BVHCommon::INVALID;
	}

	void node_make_leaf(uint32_t p_node_id) {
		uint32_t child_leaf_id;
		TLeaf *child_leaf = _leaves.request(child_leaf_id);
		child_leaf->clear();

		// zero is reserved at startup, to prevent this id being used
		// (as they are stored as negative values in the node, and zero is already taken)
		BVH_ASSERT(child_leaf_id != 0);

		TNode &node = _nodes[p_node_id];
		node.neg_leaf_id = -(int)child_leaf_id;
	}

	void node_remove_item(uint32_t p_ref_id, uint32_t p_tree_id, BVHABB_CLASS *r_old_aabb = nullptr) {
		// get the reference
		ItemRef &ref = _refs[p_ref_id];
		uint32_t owner_node_id = ref.tnode_id;

		// debug draw special
		// This may not be needed
		if (owner_node_id == BVHCommon::INVALID) {
			return;
		}

		TNode &tnode = _nodes[owner_node_id];
		CRASH_COND(!tnode.is_leaf());

		TLeaf &leaf = _node_get_leaf(tnode);

		// if the aabb is not determining the corner size, then there is no need to refit!
		// (optimization, as merging AABBs takes a lot of time)
		const BVHABB_CLASS &old_aabb = leaf.get_aabb(ref.item_id);

		// shrink a little to prevent using corner aabbs
		// in order to miss the corners first we shrink by node_expansion
		// (which is added to the overall bound of the leaf), then we also
		// shrink by an epsilon, in order to miss out the very corner aabbs
		// which are important in determining the bound. Any other aabb
		// within this can be removed and not affect the overall bound.
		BVHABB_CLASS node_bound = tnode.aabb;
		node_bound.expand(-_node_expansion - 0.001f);
		bool refit = true;

		if (node_bound.is_other_within(old_aabb)) {
			refit = false;
		}

		// record the old aabb if required (for incremental remove_and_reinsert)
		if (r_old_aabb) {
			*r_old_aabb = old_aabb;
		}

		leaf.remove_item_unordered(ref.item_id);

		if (leaf.num_items) {
			// the swapped item has to have its reference changed to, to point to the new item id
			uint32_t swapped_ref_id = leaf.get_item_ref_id(ref.item_id);

			ItemRef &swapped_ref = _refs[swapped_ref_id];

			swapped_ref.item_id = ref.item_id;

			// only have to refit if it is an edge item
			// This is a VERY EXPENSIVE STEP
			// we defer the refit updates until the update function is called once per frame
			if (refit) {
				leaf.set_dirty(true);
			}
		} else {
			// remove node if empty
			// remove link from parent
			if (tnode.parent_id != BVHCommon::INVALID) {
				// DANGER .. this can potentially end up with root node with 1 child ...
				// we don't want this and must check for it

				uint32_t parent_id = tnode.parent_id;

				node_remove_child(parent_id, owner_node_id, p_tree_id);
				refit_upward(parent_id);

				// put the node on the free list to recycle
				node_free_node_and_leaf(owner_node_id);
			}

			// else if no parent, it is the root node. Do not delete
		}

		ref.tnode_id = BVHCommon::INVALID;
		ref.item_id = BVHCommon::INVALID; // unset
	}

	// returns true if needs refit of PARENT tree only, the node itself AABB is calculated
	// within this routine
	bool _node_add_item(uint32_t p_node_id, uint32_t p_ref_id, const BVHABB_CLASS &p_aabb) {
		ItemRef &ref = _refs[p_ref_id];
		ref.tnode_id = p_node_id;

		TNode &node = _nodes[p_node_id];
		BVH_ASSERT(node.is_leaf());
		TLeaf &leaf = _node_get_leaf(node);

		// optimization - we only need to do a refit
		// if the added item is changing the AABB of the node.
		// in most cases it won't.
		bool needs_refit = true;

		// expand bound now
		BVHABB_CLASS expanded = p_aabb;
		expanded.expand(_node_expansion);

		// the bound will only be valid if there is an item in there already
		if (leaf.num_items) {
			if (node.aabb.is_other_within(expanded)) {
				// no change to node AABBs
				needs_refit = false;
			} else {
				node.aabb.merge(expanded);
			}
		} else {
			// bound of the node = the new aabb
			node.aabb = expanded;
		}

		ref.item_id = leaf.request_item();
		BVH_ASSERT(ref.item_id != BVHCommon::INVALID);

		// set the aabb of the new item
		leaf.get_aabb(ref.item_id) = p_aabb;

		// back reference on the item back to the item reference
		leaf.get_item_ref_id(ref.item_id) = p_ref_id;

		return needs_refit;
	}

	uint32_t _node_create_another_child(uint32_t p_node_id, const BVHABB_CLASS &p_aabb) {
		uint32_t child_node_id;
		TNode *child_node = _nodes.request(child_node_id);
		child_node->clear();

		// may not be necessary
		child_node->aabb = p_aabb;

		node_add_child(p_node_id, child_node_id);

		return child_node_id;
	}

public:
#pragma region Cull
	// cull parameters is a convenient way of passing a bunch
	// of arguments through the culling functions without
	// writing loads of code. Not all members are used for some cull checks
	struct CullParams {
		int result_count_overall; // both trees
		int result_count; // this tree only
		int result_max;
		T **result_array;
		int *subindex_array;

		// We now process masks etc in a user template function,
		// and these for simplicity assume even for cull tests there is a
		// testing object (which has masks etc) for the user cull checks.
		// This means for cull tests on their own, the client will usually
		// want to create a dummy object, just in order to specify masks etc.
		const T *tester;

		// optional components for different tests
		POINT point;
		BVHABB_CLASS abb;
		typename BVHABB_CLASS::ConvexHull hull;
		typename BVHABB_CLASS::Segment segment;

		// When collision testing, we can specify which tree ids
		// to collide test against with the tree_collision_mask.
		uint32_t tree_collision_mask;
	};

private:
	void _cull_translate_hits(CullParams &p) {
		int num_hits = _cull_hits.size();
		int left = p.result_max - p.result_count_overall;

		if (num_hits > left) {
			num_hits = left;
		}

		int out_n = p.result_count_overall;

		for (int n = 0; n < num_hits; n++) {
			uint32_t ref_id = _cull_hits[n];

			const ItemExtra &ex = _extra[ref_id];
			p.result_array[out_n] = ex.userdata;

			if (p.subindex_array) {
				p.subindex_array[out_n] = ex.subindex;
			}

			out_n++;
		}

		p.result_count = num_hits;
		p.result_count_overall += num_hits;
	}

public:
	int cull_convex(CullParams &r_params, bool p_translate_hits = true) {
		_cull_hits.clear();
		r_params.result_count = 0;

		uint32_t tree_test_mask = 0;

		for (int n = 0; n < NUM_TREES; n++) {
			tree_test_mask <<= 1;
			if (!tree_test_mask) {
				tree_test_mask = 1;
			}

			if (_root_node_id[n] == BVHCommon::INVALID) {
				continue;
			}

			if (!(r_params.tree_collision_mask & tree_test_mask)) {
				continue;
			}

			_cull_convex_iterative(_root_node_id[n], r_params);
		}

		if (p_translate_hits) {
			_cull_translate_hits(r_params);
		}

		return r_params.result_count;
	}

	int cull_segment(CullParams &r_params, bool p_translate_hits = true) {
		_cull_hits.clear();
		r_params.result_count = 0;

		uint32_t tree_test_mask = 0;

		for (int n = 0; n < NUM_TREES; n++) {
			tree_test_mask <<= 1;
			if (!tree_test_mask) {
				tree_test_mask = 1;
			}

			if (_root_node_id[n] == BVHCommon::INVALID) {
				continue;
			}

			if (!(r_params.tree_collision_mask & tree_test_mask)) {
				continue;
			}

			_cull_segment_iterative(_root_node_id[n], r_params);
		}

		if (p_translate_hits) {
			_cull_translate_hits(r_params);
		}

		return r_params.result_count;
	}

	int cull_point(CullParams &r_params, bool p_translate_hits = true) {
		_cull_hits.clear();
		r_params.result_count = 0;

		uint32_t tree_test_mask = 0;

		for (int n = 0; n < NUM_TREES; n++) {
			tree_test_mask <<= 1;
			if (!tree_test_mask) {
				tree_test_mask = 1;
			}

			if (_root_node_id[n] == BVHCommon::INVALID) {
				continue;
			}

			if (!(r_params.tree_collision_mask & tree_test_mask)) {
				continue;
			}

			_cull_point_iterative(_root_node_id[n], r_params);
		}

		if (p_translate_hits) {
			_cull_translate_hits(r_params);
		}

		return r_params.result_count;
	}

	int cull_aabb(CullParams &r_params, bool p_translate_hits = true) {
		_cull_hits.clear();
		r_params.result_count = 0;

		uint32_t tree_test_mask = 0;

		for (int n = 0; n < NUM_TREES; n++) {
			tree_test_mask <<= 1;
			if (!tree_test_mask) {
				tree_test_mask = 1;
			}

			if (_root_node_id[n] == BVHCommon::INVALID) {
				continue;
			}

			// the tree collision mask determines which trees to collide test against
			if (!(r_params.tree_collision_mask & tree_test_mask)) {
				continue;
			}

			_cull_aabb_iterative(_root_node_id[n], r_params);
		}

		if (p_translate_hits) {
			_cull_translate_hits(r_params);
		}

		return r_params.result_count;
	}

	bool _cull_hits_full(const CullParams &p) {
		// instead of checking every hit, we can do a lazy check for this condition.
		// it isn't a problem if we write too much _cull_hits because they only the
		// result_max amount will be translated and outputted. But we might as
		// well stop our cull checks after the maximum has been reached.
		return (int)_cull_hits.size() >= p.result_max;
	}

	void _cull_hit(uint32_t p_ref_id, CullParams &p) {
		// take into account masks etc
		// this would be more efficient to do before plane checks,
		// but done here for ease to get started
		if (USE_PAIRS) {
			const ItemExtra &ex = _extra[p_ref_id];

			// user supplied function (for e.g. pairable types and pairable masks in the render tree)
			if (!USER_CULL_TEST_FUNCTION::user_cull_check(p.tester, ex.userdata)) {
				return;
			}
		}

		_cull_hits.push_back(p_ref_id);
	}

	bool _cull_segment_iterative(uint32_t p_node_id, CullParams &r_params) {
		// our function parameters to keep on a stack
		struct CullSegParams {
			uint32_t node_id;
		};

		// most of the iterative functionality is contained in this helper class
		BVH_IterativeInfo<CullSegParams> ii;

		// alloca must allocate the stack from this function, it cannot be allocated in the
		// helper class
		ii.stack = (CullSegParams *)alloca(ii.get_alloca_stacksize());

		// seed the stack
		ii.get_first()->node_id = p_node_id;

		CullSegParams csp;

		// while there are still more nodes on the stack
		while (ii.pop(csp)) {
			TNode &tnode = _nodes[csp.node_id];

			if (tnode.is_leaf()) {
				// lazy check for hits full up condition
				if (_cull_hits_full(r_params)) {
					return false;
				}

				TLeaf &leaf = _node_get_leaf(tnode);

				// test children individually
				for (int n = 0; n < leaf.num_items; n++) {
					const BVHABB_CLASS &aabb = leaf.get_aabb(n);

					if (aabb.intersects_segment(r_params.segment)) {
						uint32_t child_id = leaf.get_item_ref_id(n);

						// register hit
						_cull_hit(child_id, r_params);
					}
				}
			} else {
				// test children individually
				for (int n = 0; n < tnode.num_children; n++) {
					uint32_t child_id = tnode.children[n];
					const BVHABB_CLASS &child_abb = _nodes[child_id].aabb;

					if (child_abb.intersects_segment(r_params.segment)) {
						// add to the stack
						CullSegParams *child = ii.request();
						child->node_id = child_id;
					}
				}
			}

		} // while more nodes to pop

		// true indicates results are not full
		return true;
	}

	bool _cull_point_iterative(uint32_t p_node_id, CullParams &r_params) {
		// our function parameters to keep on a stack
		struct CullPointParams {
			uint32_t node_id;
		};

		// most of the iterative functionality is contained in this helper class
		BVH_IterativeInfo<CullPointParams> ii;

		// alloca must allocate the stack from this function, it cannot be allocated in the
		// helper class
		ii.stack = (CullPointParams *)alloca(ii.get_alloca_stacksize());

		// seed the stack
		ii.get_first()->node_id = p_node_id;

		CullPointParams cpp;

		// while there are still more nodes on the stack
		while (ii.pop(cpp)) {
			TNode &tnode = _nodes[cpp.node_id];
			// no hit with this node?
			if (!tnode.aabb.intersects_point(r_params.point)) {
				continue;
			}

			if (tnode.is_leaf()) {
				// lazy check for hits full up condition
				if (_cull_hits_full(r_params)) {
					return false;
				}

				TLeaf &leaf = _node_get_leaf(tnode);

				// test children individually
				for (int n = 0; n < leaf.num_items; n++) {
					if (leaf.get_aabb(n).intersects_point(r_params.point)) {
						uint32_t child_id = leaf.get_item_ref_id(n);

						// register hit
						_cull_hit(child_id, r_params);
					}
				}
			} else {
				// test children individually
				for (int n = 0; n < tnode.num_children; n++) {
					uint32_t child_id = tnode.children[n];

					// add to the stack
					CullPointParams *child = ii.request();
					child->node_id = child_id;
				}
			}

		} // while more nodes to pop

		// true indicates results are not full
		return true;
	}

	// Note: This is a very hot loop profiling wise. Take care when changing this and profile.
	bool _cull_aabb_iterative(uint32_t p_node_id, CullParams &r_params, bool p_fully_within = false) {
		// our function parameters to keep on a stack
		struct CullAABBParams {
			uint32_t node_id;
			bool fully_within;
		};

		// most of the iterative functionality is contained in this helper class
		BVH_IterativeInfo<CullAABBParams> ii;

		// alloca must allocate the stack from this function, it cannot be allocated in the
		// helper class
		ii.stack = (CullAABBParams *)alloca(ii.get_alloca_stacksize());

		// seed the stack
		ii.get_first()->node_id = p_node_id;
		ii.get_first()->fully_within = p_fully_within;

		CullAABBParams cap;

		// while there are still more nodes on the stack
		while (ii.pop(cap)) {
			TNode &tnode = _nodes[cap.node_id];

			if (tnode.is_leaf()) {
				// lazy check for hits full up condition
				if (_cull_hits_full(r_params)) {
					return false;
				}

				TLeaf &leaf = _node_get_leaf(tnode);

				// if fully within we can just add all items
				// as long as they pass mask checks
				if (cap.fully_within) {
					for (int n = 0; n < leaf.num_items; n++) {
						uint32_t child_id = leaf.get_item_ref_id(n);

						// register hit
						_cull_hit(child_id, r_params);
					}
				} else {
					// This section is the hottest area in profiling, so
					// is optimized highly
					// get this into a local register and preconverted to correct type
					int leaf_num_items = leaf.num_items;

					BVHABB_CLASS swizzled_tester;
					swizzled_tester.min = -r_params.abb.neg_max;
					swizzled_tester.neg_max = -r_params.abb.min;

					for (int n = 0; n < leaf_num_items; n++) {
						const BVHABB_CLASS &aabb = leaf.get_aabb(n);

						if (swizzled_tester.intersects_swizzled(aabb)) {
							uint32_t child_id = leaf.get_item_ref_id(n);

							// register hit
							_cull_hit(child_id, r_params);
						}
					}

				} // not fully within
			} else {
				if (!cap.fully_within) {
					// test children individually
					for (int n = 0; n < tnode.num_children; n++) {
						uint32_t child_id = tnode.children[n];
						const BVHABB_CLASS &child_abb = _nodes[child_id].aabb;

						if (child_abb.intersects(r_params.abb)) {
							// is the node totally within the aabb?
							bool fully_within = r_params.abb.is_other_within(child_abb);

							// add to the stack
							CullAABBParams *child = ii.request();

							// should always return valid child
							child->node_id = child_id;
							child->fully_within = fully_within;
						}
					}
				} else {
					for (int n = 0; n < tnode.num_children; n++) {
						uint32_t child_id = tnode.children[n];

						// add to the stack
						CullAABBParams *child = ii.request();

						// should always return valid child
						child->node_id = child_id;
						child->fully_within = true;
					}
				}
			}

		} // while more nodes to pop

		// true indicates results are not full
		return true;
	}

	// returns full up with results
	bool _cull_convex_iterative(uint32_t p_node_id, CullParams &r_params, bool p_fully_within = false) {
		// our function parameters to keep on a stack
		struct CullConvexParams {
			uint32_t node_id;
			bool fully_within;
		};

		// most of the iterative functionality is contained in this helper class
		BVH_IterativeInfo<CullConvexParams> ii;

		// alloca must allocate the stack from this function, it cannot be allocated in the
		// helper class
		ii.stack = (CullConvexParams *)alloca(ii.get_alloca_stacksize());

		// seed the stack
		ii.get_first()->node_id = p_node_id;
		ii.get_first()->fully_within = p_fully_within;

		// preallocate these as a once off to be reused
		uint32_t max_planes = r_params.hull.num_planes;
		uint32_t *plane_ids = (uint32_t *)alloca(sizeof(uint32_t) * max_planes);

		CullConvexParams ccp;

		// while there are still more nodes on the stack
		while (ii.pop(ccp)) {
			const TNode &tnode = _nodes[ccp.node_id];

			if (!ccp.fully_within) {
				typename BVHABB_CLASS::IntersectResult res = tnode.aabb.intersects_convex(r_params.hull);

				switch (res) {
					default: {
						continue; // miss, just move on to the next node in the stack
					} break;
					case BVHABB_CLASS::IR_PARTIAL: {
					} break;
					case BVHABB_CLASS::IR_FULL: {
						ccp.fully_within = true;
					} break;
				}

			} // if not fully within already

			if (tnode.is_leaf()) {
				// lazy check for hits full up condition
				if (_cull_hits_full(r_params)) {
					return false;
				}

				const TLeaf &leaf = _node_get_leaf(tnode);

				// if fully within, simply add all items to the result
				// (taking into account masks)
				if (ccp.fully_within) {
					for (int n = 0; n < leaf.num_items; n++) {
						uint32_t child_id = leaf.get_item_ref_id(n);

						// register hit
						_cull_hit(child_id, r_params);
					}

				} else {
					// we can either use a naive check of all the planes against the AABB,
					// or an optimized check, which finds in advance which of the planes can possibly
					// cut the AABB, and only tests those. This can be much faster.
#define BVH_CONVEX_CULL_OPTIMIZED
#ifdef BVH_CONVEX_CULL_OPTIMIZED
					// first find which planes cut the aabb
					uint32_t num_planes = tnode.aabb.find_cutting_planes(r_params.hull, plane_ids);
					BVH_ASSERT(num_planes <= max_planes);

//#define BVH_CONVEX_CULL_OPTIMIZED_RIGOR_CHECK
#ifdef BVH_CONVEX_CULL_OPTIMIZED_RIGOR_CHECK
					// rigorous check
					uint32_t results[MAX_ITEMS];
					uint32_t num_results = 0;
#endif

					// test children individually
					for (int n = 0; n < leaf.num_items; n++) {
						//const Item &item = leaf.get_item(n);
						const BVHABB_CLASS &aabb = leaf.get_aabb(n);

						if (aabb.intersects_convex_optimized(r_params.hull, plane_ids, num_planes)) {
							uint32_t child_id = leaf.get_item_ref_id(n);

#ifdef BVH_CONVEX_CULL_OPTIMIZED_RIGOR_CHECK
							results[num_results++] = child_id;
#endif

							// register hit
							_cull_hit(child_id, r_params);
						}
					}

#ifdef BVH_CONVEX_CULL_OPTIMIZED_RIGOR_CHECK
					uint32_t test_count = 0;

					for (int n = 0; n < leaf.num_items; n++) {
						const BVHABB_CLASS &aabb = leaf.get_aabb(n);

						if (aabb.intersects_convex_partial(r_params.hull)) {
							uint32_t child_id = leaf.get_item_ref_id(n);

							CRASH_COND(child_id != results[test_count++]);
							CRASH_COND(test_count > num_results);
						}
					}
#endif

#else
					// not BVH_CONVEX_CULL_OPTIMIZED
					// test children individually
					for (int n = 0; n < leaf.num_items; n++) {
						const BVHABB_CLASS &aabb = leaf.get_aabb(n);

						if (aabb.intersects_convex_partial(r_params.hull)) {
							uint32_t child_id = leaf.get_item_ref_id(n);

							// full up with results? exit early, no point in further testing
							if (!_cull_hit(child_id, r_params)) {
								return false;
							}
						}
					}
#endif // BVH_CONVEX_CULL_OPTIMIZED
				} // if not fully within
			} else {
				for (int n = 0; n < tnode.num_children; n++) {
					uint32_t child_id = tnode.children[n];

					// add to the stack
					CullConvexParams *child = ii.request();

					// should always return valid child
					child->node_id = child_id;
					child->fully_within = ccp.fully_within;
				}
			}

		} // while more nodes to pop

		// true indicates results are not full
		return true;
	}
#pragma endregion Cull

#pragma region Debug
#ifdef BVH_VERBOSE
	void _debug_recursive_print_tree(int p_tree_id) const {
		if (_root_node_id[p_tree_id] != BVHCommon::INVALID) {
			_debug_recursive_print_tree_node(_root_node_id[p_tree_id]);
		}
	}

	String _debug_aabb_to_string(const BVHABB_CLASS &aabb) const {
		POINT size = aabb.calculate_size();

		String sz;
		real_t vol = 0.0;

		for (int i = 0; i < POINT::AXIS_COUNT; ++i) {
			sz += "(";
			sz += itos(aabb.min[i]);
			sz += " ~ ";
			sz += itos(-aabb.neg_max[i]);
			sz += ") ";

			vol += size[i];
		}

		sz += "vol " + itos(vol);

		return sz;
	}

	void _debug_recursive_print_tree_node(uint32_t p_node_id, int depth = 0) const {
		const TNode &tnode = _nodes[p_node_id];

		String sz = String("\t").repeat(depth) + itos(p_node_id);

		if (tnode.is_leaf()) {
			sz += " L";
			sz += itos(tnode.height) + " ";
			const TLeaf &leaf = _node_get_leaf(tnode);

			sz += "[";
			for (int n = 0; n < leaf.num_items; n++) {
				if (n) {
					sz += ", ";
				}
				sz += "r";
				sz += itos(leaf.get_item_ref_id(n));
			}
			sz += "]  ";
		} else {
			sz += " N";
			sz += itos(tnode.height) + " ";
		}

		sz += _debug_aabb_to_string(tnode.aabb);
		print_line(sz);

		if (!tnode.is_leaf()) {
			for (int n = 0; n < tnode.num_children; n++) {
				_debug_recursive_print_tree_node(tnode.children[n], depth + 1);
			}
		}
	}
#endif
#pragma endregion Debug

#pragma region Integrity
	void _integrity_check_all() {
#ifdef BVH_INTEGRITY_CHECKS
		for (int n = 0; n < NUM_TREES; n++) {
			uint32_t root = _root_node_id[n];
			if (root != BVHCommon::INVALID) {
				_integrity_check_down(root);
			}
		}
#endif
	}

	void _integrity_check_up(uint32_t p_node_id) {
		TNode &node = _nodes[p_node_id];

		BVHABB_CLASS abb = node.aabb;
		node_update_aabb(node);

		BVHABB_CLASS abb2 = node.aabb;
		abb2.expand(-_node_expansion);

		CRASH_COND(!abb.is_other_within(abb2));
	}

	void _integrity_check_down(uint32_t p_node_id) {
		const TNode &node = _nodes[p_node_id];

		if (node.is_leaf()) {
			_integrity_check_up(p_node_id);
		} else {
			CRASH_COND(node.num_children != 2);

			for (int n = 0; n < node.num_children; n++) {
				uint32_t child_id = node.children[n];

				// check the children parent pointers are correct
				TNode &child = _nodes[child_id];
				CRASH_COND(child.parent_id != p_node_id);

				_integrity_check_down(child_id);
			}
		}
	}
#pragma endregion Integrity

#pragma region Logic
	// for slow incremental optimization, we will periodically remove each
	// item from the tree and reinsert, to give it a chance to find a better position
	void _logic_item_remove_and_reinsert(uint32_t p_ref_id) {
		// get the reference
		ItemRef &ref = _refs[p_ref_id];

		// no need to optimize inactive items
		if (!ref.is_active()) {
			return;
		}

		// special case of debug draw
		if (ref.item_id == BVHCommon::INVALID) {
			return;
		}

		BVH_ASSERT(ref.tnode_id != BVHCommon::INVALID);

		// some overlay elaborate way to find out which tree the node is in!
		BVHHandle temp_handle;
		temp_handle.set_id(p_ref_id);
		uint32_t tree_id = _handle_get_tree_id(temp_handle);

		// remove and reinsert
		BVHABB_CLASS abb;
		node_remove_item(p_ref_id, tree_id, &abb);

		// we must choose where to add to tree
		ref.tnode_id = _logic_choose_item_add_node(_root_node_id[tree_id], abb);
		_node_add_item(ref.tnode_id, p_ref_id, abb);

		refit_upward_and_balance(ref.tnode_id, tree_id);
	}

	// from randy gaul balance function
	BVHABB_CLASS _logic_abb_merge(const BVHABB_CLASS &a, const BVHABB_CLASS &b) {
		BVHABB_CLASS c = a;
		c.merge(b);
		return c;
	}

	//--------------------------------------------------------------------------------------------------
	/**
	 * @file    q3DynamicAABBTree.h
	 * @author  Randy Gaul
	 * @date    10/10/2014
	 *  Copyright (c) 2014 Randy Gaul http://www.randygaul.net
	 *  This software is provided 'as-is', without any express or implied
	 *  warranty. In no event will the authors be held liable for any damages
	 *  arising from the use of this software.
	 *  Permission is granted to anyone to use this software for any purpose,
	 *  including commercial applications, and to alter it and redistribute it
	 *  freely, subject to the following restrictions:
	 *    1. The origin of this software must not be misrepresented; you must not
	 *       claim that you wrote the original software. If you use this software
	 *       in a product, an acknowledgment in the product documentation would be
	 *       appreciated but is not required.
	 *    2. Altered source versions must be plainly marked as such, and must not
	 *       be misrepresented as being the original software.
	 *    3. This notice may not be removed or altered from any source distribution.
	 */
	//--------------------------------------------------------------------------------------------------

	// This function is based on the 'Balance' function from Randy Gaul's qu3e
	// https://github.com/RandyGaul/qu3e
	// It is MODIFIED from qu3e version.
	// This is the only function used (and _logic_abb_merge helper function).
	int32_t _logic_balance(int32_t iA, uint32_t p_tree_id) {
		//return iA; // uncomment this to bypass balance

		TNode *A = &_nodes[iA];

		if (A->is_leaf() || A->height == 1) {
			return iA;
		}

		/*        A
		 *      /   \
		 *     B     C
		 *    / \   / \
		 *   D   E F   G
		 */

		CRASH_COND(A->num_children != 2);
		int32_t iB = A->children[0];
		int32_t iC = A->children[1];
		TNode *B = &_nodes[iB];
		TNode *C = &_nodes[iC];

		int32_t balance = C->height - B->height;

		// C is higher, promote C
		if (balance > 1) {
			int32_t iF = C->children[0];
			int32_t iG = C->children[1];
			TNode *F = &_nodes[iF];
			TNode *G = &_nodes[iG];

			// grandParent point to C
			if (A->parent_id != BVHCommon::INVALID) {
				if (_nodes[A->parent_id].children[0] == iA) {
					_nodes[A->parent_id].children[0] = iC;

				} else {
					_nodes[A->parent_id].children[1] = iC;
				}
			} else {
				// check this .. seems dodgy
				change_root_node(iC, p_tree_id);
			}

			// Swap A and C
			C->children[0] = iA;
			C->parent_id = A->parent_id;
			A->parent_id = iC;

			// Finish rotation
			if (F->height > G->height) {
				C->children[1] = iF;
				A->children[1] = iG;
				G->parent_id = iA;
				A->aabb = _logic_abb_merge(B->aabb, G->aabb);
				C->aabb = _logic_abb_merge(A->aabb, F->aabb);

				A->height = 1 + MAX(B->height, G->height);
				C->height = 1 + MAX(A->height, F->height);
			}

			else {
				C->children[1] = iG;
				A->children[1] = iF;
				F->parent_id = iA;
				A->aabb = _logic_abb_merge(B->aabb, F->aabb);
				C->aabb = _logic_abb_merge(A->aabb, G->aabb);

				A->height = 1 + MAX(B->height, F->height);
				C->height = 1 + MAX(A->height, G->height);
			}

			return iC;
		}

		// B is higher, promote B
		else if (balance < -1) {
			int32_t iD = B->children[0];
			int32_t iE = B->children[1];
			TNode *D = &_nodes[iD];
			TNode *E = &_nodes[iE];

			// grandParent point to B
			if (A->parent_id != BVHCommon::INVALID) {
				if (_nodes[A->parent_id].children[0] == iA) {
					_nodes[A->parent_id].children[0] = iB;
				} else {
					_nodes[A->parent_id].children[1] = iB;
				}
			}

			else {
				// check this .. seems dodgy
				change_root_node(iB, p_tree_id);
			}

			// Swap A and B
			B->children[1] = iA;
			B->parent_id = A->parent_id;
			A->parent_id = iB;

			// Finish rotation
			if (D->height > E->height) {
				B->children[0] = iD;
				A->children[0] = iE;
				E->parent_id = iA;
				A->aabb = _logic_abb_merge(C->aabb, E->aabb);
				B->aabb = _logic_abb_merge(A->aabb, D->aabb);

				A->height = 1 + MAX(C->height, E->height);
				B->height = 1 + MAX(A->height, D->height);
			}

			else {
				B->children[0] = iE;
				A->children[0] = iD;
				D->parent_id = iA;
				A->aabb = _logic_abb_merge(C->aabb, D->aabb);
				B->aabb = _logic_abb_merge(A->aabb, E->aabb);

				A->height = 1 + MAX(C->height, D->height);
				B->height = 1 + MAX(A->height, E->height);
			}

			return iB;
		}

		return iA;
	}

	// either choose an existing node to add item to, or create a new node and return this
	uint32_t _logic_choose_item_add_node(uint32_t p_node_id, const BVHABB_CLASS &p_aabb) {
		while (true) {
			BVH_ASSERT(p_node_id != BVHCommon::INVALID);
			TNode &tnode = _nodes[p_node_id];

			if (tnode.is_leaf()) {
				// if a leaf, and non full, use this to add to
				if (!node_is_leaf_full(tnode)) {
					return p_node_id;
				}

				// else split the leaf, and use one of the children to add to
				return split_leaf(p_node_id, p_aabb);
			}

			// this should not happen???
			// is still happening, need to debug and find circumstances. Is not that serious
			// but would be nice to prevent. I think it only happens with the root node.
			if (tnode.num_children == 1) {
				WARN_PRINT_ONCE("BVH::recursive_choose_item_add_node, node with 1 child, recovering");
				p_node_id = tnode.children[0];
			} else {
				BVH_ASSERT(tnode.num_children == 2);
				TNode &childA = _nodes[tnode.children[0]];
				TNode &childB = _nodes[tnode.children[1]];
				int which = p_aabb.select_by_proximity(childA.aabb, childB.aabb);

				p_node_id = tnode.children[which];
			}
		}
	}
#pragma endregion Logic

#pragma region Misc
	int _handle_get_tree_id(BVHHandle p_handle) const {
		if (USE_PAIRS) {
			return _extra[p_handle.id()].tree_id;
		}
		return 0;
	}

	void _handle_sort(BVHHandle &p_ha, BVHHandle &p_hb) const {
		if (p_ha.id() > p_hb.id()) {
			BVHHandle temp = p_hb;
			p_hb = p_ha;
			p_ha = temp;
		}
	}

private:
	void create_root_node(int p_tree) {
		// if there is no root node, create one
		if (_root_node_id[p_tree] == BVHCommon::INVALID) {
			uint32_t root_node_id;
			TNode *node = _nodes.request(root_node_id);
			node->clear();
			_root_node_id[p_tree] = root_node_id;

			// make the root node a leaf
			uint32_t leaf_id;
			TLeaf *leaf = _leaves.request(leaf_id);
			leaf->clear();
			node->neg_leaf_id = -(int)leaf_id;
		}
	}

	bool node_is_leaf_full(TNode &tnode) const {
		const TLeaf &leaf = _node_get_leaf(tnode);
		return leaf.is_full();
	}

public:
	TLeaf &_node_get_leaf(TNode &tnode) {
		BVH_ASSERT(tnode.is_leaf());
		return _leaves[tnode.get_leaf_id()];
	}

	const TLeaf &_node_get_leaf(const TNode &tnode) const {
		BVH_ASSERT(tnode.is_leaf());
		return _leaves[tnode.get_leaf_id()];
	}
#pragma endregion Misc

#pragma region Public
	BVHHandle item_add(T *p_userdata, bool p_active, const BOUNDS &p_aabb, int32_t p_subindex, uint32_t p_tree_id, uint32_t p_tree_collision_mask, bool p_invisible = false) {
#ifdef BVH_VERBOSE_TREE
		VERBOSE_PRINT("\nitem_add BEFORE");
		_debug_recursive_print_tree(p_tree_id);
		VERBOSE_PRINT("\n");
#endif

		BVHABB_CLASS abb;
		abb.from(p_aabb);

		// NOTE that we do not expand the AABB for the first create even if
		// leaf expansion is switched on. This is for two reasons:
		// (1) We don't know if this object will move in future, in which case a non-expanded
		// bound would be better...
		// (2) We don't yet know how many objects will be paired, which is used to modify
		// the expansion margin.

		// handle to be filled with the new item ref
		BVHHandle handle;

		// ref id easier to pass around than handle
		uint32_t ref_id;

		// this should never fail
		ItemRef *ref = _refs.request(ref_id);

		// the extra data should be parallel list to the references
		uint32_t extra_id;
		ItemExtra *extra = _extra.request(extra_id);
		BVH_ASSERT(extra_id == ref_id);

		// pairs info
		if (USE_PAIRS) {
			uint32_t pairs_id;
			ItemPairs *pairs = _pairs.request(pairs_id);
			pairs->clear();
			BVH_ASSERT(pairs_id == ref_id);
		}

		extra->subindex = p_subindex;
		extra->userdata = p_userdata;
		extra->last_updated_tick = 0;

		// add an active reference to the list for slow incremental optimize
		// this list must be kept in sync with the references as they are added or removed.
		extra->active_ref_id = _active_refs.size();
		_active_refs.push_back(ref_id);

		extra->tree_id = p_tree_id;
		extra->tree_collision_mask = p_tree_collision_mask;

		// assign to handle to return
		handle.set_id(ref_id);

		create_root_node(p_tree_id);

		// we must choose where to add to tree
		if (p_active) {
			ref->tnode_id = _logic_choose_item_add_node(_root_node_id[p_tree_id], abb);

			bool refit = _node_add_item(ref->tnode_id, ref_id, abb);

			if (refit) {
				// only need to refit from the parent
				const TNode &add_node = _nodes[ref->tnode_id];
				if (add_node.parent_id != BVHCommon::INVALID) {
					refit_upward_and_balance(add_node.parent_id, p_tree_id);
				}
			}
		} else {
			ref->set_inactive();
		}

#ifdef BVH_VERBOSE
		// memory use
		int mem = _refs.estimate_memory_use();
		mem += _nodes.estimate_memory_use();

		String sz = _debug_aabb_to_string(abb);
		VERBOSE_PRINT("\titem_add [" + itos(ref_id) + "] " + itos(_refs.used_size()) + " refs,\t" + itos(_nodes.used_size()) + " nodes " + sz);
		VERBOSE_PRINT("mem use : " + itos(mem) + ", num nodes reserved : " + itos(_nodes.reserved_size()));

#endif

		return handle;
	}

	void _debug_print_refs() {
#ifdef BVH_VERBOSE_TREE
		print_line("refs.....");
		for (int n = 0; n < _refs.size(); n++) {
			const ItemRef &ref = _refs[n];
			print_line("tnode_id " + itos(ref.tnode_id) + ", item_id " + itos(ref.item_id));
		}

#endif
	}

	// returns false if noop
	bool item_move(BVHHandle p_handle, const BOUNDS &p_aabb) {
		uint32_t ref_id = p_handle.id();

		// get the reference
		ItemRef &ref = _refs[ref_id];
		if (!ref.is_active()) {
			return false;
		}

		BVHABB_CLASS abb;
		abb.from(p_aabb);

#ifdef BVH_EXPAND_LEAF_AABBS
		if (USE_PAIRS) {
			// scale the pairing expansion by the number of pairs.
			abb.expand(_pairs[ref_id].scale_expansion_margin(_pairing_expansion));
		} else {
			abb.expand(_pairing_expansion);
		}
#endif

		BVH_ASSERT(ref.tnode_id != BVHCommon::INVALID);
		TNode &tnode = _nodes[ref.tnode_id];

		// does it fit within the current leaf aabb?
		if (tnode.aabb.is_other_within(abb)) {
			// do nothing .. fast path .. not moved enough to need refit

			// however we WILL update the exact aabb in the leaf, as this will be needed
			// for accurate collision detection
			TLeaf &leaf = _node_get_leaf(tnode);

			BVHABB_CLASS &leaf_abb = leaf.get_aabb(ref.item_id);

			// no change?
#ifdef BVH_EXPAND_LEAF_AABBS
			BOUNDS leaf_aabb;
			leaf_abb.to(leaf_aabb);

			// This test should pass in a lot of cases, and by returning false we can avoid
			// collision pairing checks later, which greatly reduces processing.
			if (expanded_aabb_encloses_not_shrink(leaf_aabb, p_aabb)) {
				return false;
			}
#else
			if (leaf_abb == abb) {
				return false;
			}
#endif

#ifdef BVH_VERBOSE_MOVES
			print_line("item_move " + itos(p_handle.id()) + "(within tnode aabb) : " + _debug_aabb_to_string(abb));
#endif

			leaf_abb = abb;
			_integrity_check_all();

			return true;
		}

#ifdef BVH_VERBOSE_MOVES
		print_line("item_move " + itos(p_handle.id()) + "(outside tnode aabb) : " + _debug_aabb_to_string(abb));
#endif

		uint32_t tree_id = _handle_get_tree_id(p_handle);

		// remove and reinsert
		node_remove_item(ref_id, tree_id);

		// we must choose where to add to tree
		ref.tnode_id = _logic_choose_item_add_node(_root_node_id[tree_id], abb);

		// add to the tree
		bool needs_refit = _node_add_item(ref.tnode_id, ref_id, abb);

		// only need to refit from the PARENT
		if (needs_refit) {
			// only need to refit from the parent
			const TNode &add_node = _nodes[ref.tnode_id];
			if (add_node.parent_id != BVHCommon::INVALID) {
				// not sure we need to rebalance all the time, this can be done less often
				refit_upward(add_node.parent_id);
			}
			//refit_upward_and_balance(add_node.parent_id);
		}

		return true;
	}

	void item_remove(BVHHandle p_handle) {
		uint32_t ref_id = p_handle.id();

		uint32_t tree_id = _handle_get_tree_id(p_handle);

		VERBOSE_PRINT("item_remove [" + itos(ref_id) + "] ");

		////////////////////////////////////////
		// remove the active reference from the list for slow incremental optimize
		// this list must be kept in sync with the references as they are added or removed.
		uint32_t active_ref_id = _extra[ref_id].active_ref_id;
		uint32_t ref_id_moved_back = _active_refs[_active_refs.size() - 1];

		// swap back and decrement for fast unordered remove
		_active_refs[active_ref_id] = ref_id_moved_back;
		_active_refs.resize(_active_refs.size() - 1);

		// keep the moved active reference up to date
		_extra[ref_id_moved_back].active_ref_id = active_ref_id;
		////////////////////////////////////////

		// remove the item from the node (only if active)
		if (_refs[ref_id].is_active()) {
			node_remove_item(ref_id, tree_id);
		}

		// remove the item reference
		_refs.free(ref_id);
		_extra.free(ref_id);
		if (USE_PAIRS) {
			_pairs.free(ref_id);
		}

		// don't think refit_all is necessary?
		//refit_all(_tree_id);

#ifdef BVH_VERBOSE_TREE
		_debug_recursive_print_tree(tree_id);
#endif
	}

	// returns success
	bool item_activate(BVHHandle p_handle, const BOUNDS &p_aabb) {
		uint32_t ref_id = p_handle.id();
		ItemRef &ref = _refs[ref_id];
		if (ref.is_active()) {
			// noop
			return false;
		}

		// add to tree
		BVHABB_CLASS abb;
		abb.from(p_aabb);

		uint32_t tree_id = _handle_get_tree_id(p_handle);

		// we must choose where to add to tree
		ref.tnode_id = _logic_choose_item_add_node(_root_node_id[tree_id], abb);
		_node_add_item(ref.tnode_id, ref_id, abb);

		refit_upward_and_balance(ref.tnode_id, tree_id);

		return true;
	}

	// returns success
	bool item_deactivate(BVHHandle p_handle) {
		uint32_t ref_id = p_handle.id();
		ItemRef &ref = _refs[ref_id];
		if (!ref.is_active()) {
			// noop
			return false;
		}

		uint32_t tree_id = _handle_get_tree_id(p_handle);

		// remove from tree
		BVHABB_CLASS abb;
		node_remove_item(ref_id, tree_id, &abb);

		// mark as inactive
		ref.set_inactive();
		return true;
	}

	bool item_get_active(BVHHandle p_handle) const {
		uint32_t ref_id = p_handle.id();
		const ItemRef &ref = _refs[ref_id];
		return ref.is_active();
	}

	// during collision testing, we want to set the mask and whether pairable for the item testing from
	void item_fill_cullparams(BVHHandle p_handle, CullParams &r_params) const {
		uint32_t ref_id = p_handle.id();
		const ItemExtra &extra = _extra[ref_id];

		// which trees does this item want to collide detect against?
		r_params.tree_collision_mask = extra.tree_collision_mask;

		// The testing user defined object is passed to the user defined cull check function
		// for masks etc. This is usually a dummy object of type T with masks set.
		// However, if not using the cull_check callback (i.e. returning true), you can pass
		// a nullptr instead of dummy object, as it will not be used.
		r_params.tester = extra.userdata;
	}

	bool item_is_pairable(const BVHHandle &p_handle) {
		uint32_t ref_id = p_handle.id();
		const ItemExtra &extra = _extra[ref_id];
		return extra.pairable != 0;
	}

	void item_get_ABB(const BVHHandle &p_handle, BVHABB_CLASS &r_abb) {
		// change tree?
		uint32_t ref_id = p_handle.id();
		const ItemRef &ref = _refs[ref_id];

		TNode &tnode = _nodes[ref.tnode_id];
		TLeaf &leaf = _node_get_leaf(tnode);

		r_abb = leaf.get_aabb(ref.item_id);
	}

	bool item_set_tree(const BVHHandle &p_handle, uint32_t p_tree_id, uint32_t p_tree_collision_mask) {
		// change tree?
		uint32_t ref_id = p_handle.id();

		ItemExtra &ex = _extra[ref_id];
		ItemRef &ref = _refs[ref_id];

		bool active = ref.is_active();
		bool tree_changed = ex.tree_id != p_tree_id;
		bool mask_changed = ex.tree_collision_mask != p_tree_collision_mask;
		bool state_changed = tree_changed | mask_changed;

		// Keep an eye on this for bugs of not noticing changes to objects,
		// especially when changing client user masks that will not be detected as a change
		// in the BVH. You may need to force a collision check in this case with recheck_pairs().

		if (active && (tree_changed | mask_changed)) {
			// record abb
			TNode &tnode = _nodes[ref.tnode_id];
			TLeaf &leaf = _node_get_leaf(tnode);
			BVHABB_CLASS abb = leaf.get_aabb(ref.item_id);

			// make sure current tree is correct prior to changing
			uint32_t tree_id = _handle_get_tree_id(p_handle);

			// remove from old tree
			node_remove_item(ref_id, tree_id);

			// we must set the pairable AFTER getting the current tree
			// because the pairable status determines which tree
			ex.tree_id = p_tree_id;
			ex.tree_collision_mask = p_tree_collision_mask;

			// add to new tree
			tree_id = _handle_get_tree_id(p_handle);
			create_root_node(tree_id);

			// we must choose where to add to tree
			ref.tnode_id = _logic_choose_item_add_node(_root_node_id[tree_id], abb);
			bool needs_refit = _node_add_item(ref.tnode_id, ref_id, abb);

			// only need to refit from the PARENT
			if (needs_refit) {
				// only need to refit from the parent
				const TNode &add_node = _nodes[ref.tnode_id];
				if (add_node.parent_id != BVHCommon::INVALID) {
					refit_upward_and_balance(add_node.parent_id, tree_id);
				}
			}
		} else {
			// always keep this up to date
			ex.tree_id = p_tree_id;
			ex.tree_collision_mask = p_tree_collision_mask;
		}

		return state_changed;
	}

	void incremental_optimize() {
		// first update all aabbs as one off step..
		// this is cheaper than doing it on each move as each leaf may get touched multiple times
		// in a frame.
		for (int n = 0; n < NUM_TREES; n++) {
			if (_root_node_id[n] != BVHCommon::INVALID) {
				refit_branch(_root_node_id[n]);
			}
		}

		// now do small section reinserting to get things moving
		// gradually, and keep items in the right leaf
		if (_current_active_ref >= _active_refs.size()) {
			_current_active_ref = 0;
		}

		// special case
		if (!_active_refs.size()) {
			return;
		}

		uint32_t ref_id = _active_refs[_current_active_ref++];

		_logic_item_remove_and_reinsert(ref_id);

#ifdef BVH_VERBOSE
		/*
		// memory use
		int mem_refs = _refs.estimate_memory_use();
		int mem_nodes = _nodes.estimate_memory_use();
		int mem_leaves = _leaves.estimate_memory_use();

		String sz;
		sz += "mem_refs : " + itos(mem_refs) + " ";
		sz += "mem_nodes : " + itos(mem_nodes) + " ";
		sz += "mem_leaves : " + itos(mem_leaves) + " ";
		sz += ", num nodes : " + itos(_nodes.size());
		print_line(sz);
		*/
#endif
	}

	void update() {
		incremental_optimize();

		// keep the expansion values up to date with the world bound
//#define BVH_ALLOW_AUTO_EXPANSION
#ifdef BVH_ALLOW_AUTO_EXPANSION
		if (_auto_node_expansion || _auto_pairing_expansion) {
			BVHABB_CLASS world_bound;
			world_bound.set_to_max_opposite_extents();

			bool bound_valid = false;

			for (int n = 0; n < NUM_TREES; n++) {
				uint32_t node_id = _root_node_id[n];
				if (node_id != BVHCommon::INVALID) {
					world_bound.merge(_nodes[node_id].aabb);
					bound_valid = true;
				}
			}

			// if there are no nodes, do nothing, but if there are...
			if (bound_valid) {
				BOUNDS bb;
				world_bound.to(bb);
				real_t size = bb.get_longest_axis_size();

				// automatic AI decision for best parameters.
				// These can be overridden in project settings.

				// these magic numbers are determined by experiment
				if (_auto_node_expansion) {
					_node_expansion = size * 0.025;
				}
				if (_auto_pairing_expansion) {
					_pairing_expansion = size * 0.009;
				}
			}
		}
#endif
	}

	void params_set_pairing_expansion(real_t p_value) {
		if (p_value < 0.0) {
#ifdef BVH_ALLOW_AUTO_EXPANSION
			_auto_pairing_expansion = true;
#endif
			return;
		}
#ifdef BVH_ALLOW_AUTO_EXPANSION
		_auto_pairing_expansion = false;
#endif

		_pairing_expansion = p_value;

		// calculate shrinking threshold
		const real_t fudge_factor = 1.1;
		_aabb_shrinkage_threshold = _pairing_expansion * POINT::AXIS_COUNT * 2.0 * fudge_factor;
	}

	// This routine is not just an enclose check, it also checks for special case of shrinkage
	bool expanded_aabb_encloses_not_shrink(const BOUNDS &p_expanded_aabb, const BOUNDS &p_aabb) const {
		if (!p_expanded_aabb.encloses(p_aabb)) {
			return false;
		}

		// Check for special case of shrinkage. If the aabb has shrunk
		// significantly we want to create a new expanded bound, because
		// the previous expanded bound will have diverged significantly.
		const POINT &exp_size = p_expanded_aabb.size;
		const POINT &new_size = p_aabb.size;

		real_t exp_l = 0.0;
		real_t new_l = 0.0;

		for (int i = 0; i < POINT::AXIS_COUNT; ++i) {
			exp_l += exp_size[i];
			new_l += new_size[i];
		}

		// is difference above some metric
		real_t diff = exp_l - new_l;
		if (diff < _aabb_shrinkage_threshold) {
			return true;
		}

		return false;
	}
#pragma endregion Public

#pragma region Refit
	void _debug_node_verify_bound(uint32_t p_node_id) {
		TNode &node = _nodes[p_node_id];
		BVHABB_CLASS abb_before = node.aabb;

		node_update_aabb(node);

		BVHABB_CLASS abb_after = node.aabb;
		CRASH_COND(abb_before != abb_after);
	}

	void node_update_aabb(TNode &tnode) {
		tnode.aabb.set_to_max_opposite_extents();
		tnode.height = 0;

		if (!tnode.is_leaf()) {
			for (int n = 0; n < tnode.num_children; n++) {
				uint32_t child_node_id = tnode.children[n];

				// merge with child aabb
				const TNode &tchild = _nodes[child_node_id];
				tnode.aabb.merge(tchild.aabb);

				// do heights at the same time
				if (tchild.height > tnode.height) {
					tnode.height = tchild.height;
				}
			}

			// the height of a non leaf is always 1 bigger than the biggest child
			tnode.height++;

#ifdef BVH_CHECKS
			if (!tnode.num_children) {
				// the 'blank' aabb will screw up parent aabbs
				WARN_PRINT("BVH_Tree::TNode no children, AABB is undefined");
			}
#endif
		} else {
			// leaf
			const TLeaf &leaf = _node_get_leaf(tnode);

			for (int n = 0; n < leaf.num_items; n++) {
				tnode.aabb.merge(leaf.get_aabb(n));
			}

			// now the leaf items are unexpanded, we expand only in the node AABB
			tnode.aabb.expand(_node_expansion);
#ifdef BVH_CHECKS
			if (!leaf.num_items) {
				// the 'blank' aabb will screw up parent aabbs
				WARN_PRINT("BVH_Tree::TLeaf no items, AABB is undefined");
			}
#endif
		}
	}

	void refit_all(int p_tree_id) {
		refit_downward(_root_node_id[p_tree_id]);
	}

	void refit_upward(uint32_t p_node_id) {
		while (p_node_id != BVHCommon::INVALID) {
			TNode &tnode = _nodes[p_node_id];
			node_update_aabb(tnode);
			p_node_id = tnode.parent_id;
		}
	}

	void refit_upward_and_balance(uint32_t p_node_id, uint32_t p_tree_id) {
		while (p_node_id != BVHCommon::INVALID) {
			uint32_t before = p_node_id;
			p_node_id = _logic_balance(p_node_id, p_tree_id);

			if (before != p_node_id) {
				VERBOSE_PRINT("REBALANCED!");
			}

			TNode &tnode = _nodes[p_node_id];

			// update overall aabb from the children
			node_update_aabb(tnode);

			p_node_id = tnode.parent_id;
		}
	}

	void refit_downward(uint32_t p_node_id) {
		TNode &tnode = _nodes[p_node_id];

		// do children first
		if (!tnode.is_leaf()) {
			for (int n = 0; n < tnode.num_children; n++) {
				refit_downward(tnode.children[n]);
			}
		}

		node_update_aabb(tnode);
	}

	// go down to the leaves, then refit upward
	void refit_branch(uint32_t p_node_id) {
		// our function parameters to keep on a stack
		struct RefitParams {
			uint32_t node_id;
		};

		// most of the iterative functionality is contained in this helper class
		BVH_IterativeInfo<RefitParams> ii;

		// alloca must allocate the stack from this function, it cannot be allocated in the
		// helper class
		ii.stack = (RefitParams *)alloca(ii.get_alloca_stacksize());

		// seed the stack
		ii.get_first()->node_id = p_node_id;

		RefitParams rp;

		// while there are still more nodes on the stack
		while (ii.pop(rp)) {
			TNode &tnode = _nodes[rp.node_id];

			// do children first
			if (!tnode.is_leaf()) {
				for (int n = 0; n < tnode.num_children; n++) {
					uint32_t child_id = tnode.children[n];

					// add to the stack
					RefitParams *child = ii.request();
					child->node_id = child_id;
				}
			} else {
				// leaf .. only refit upward if dirty
				TLeaf &leaf = _node_get_leaf(tnode);
				if (leaf.is_dirty()) {
					leaf.set_dirty(false);
					refit_upward(rp.node_id);
				}
			}
		} // while more nodes to pop
	}
#pragma endregion Refit

#pragma region Split
	void _split_inform_references(uint32_t p_node_id) {
		TNode &node = _nodes[p_node_id];
		TLeaf &leaf = _node_get_leaf(node);

		for (int n = 0; n < leaf.num_items; n++) {
			uint32_t ref_id = leaf.get_item_ref_id(n);

			ItemRef &ref = _refs[ref_id];
			ref.tnode_id = p_node_id;
			ref.item_id = n;
		}
	}

	void _split_leaf_sort_groups_simple(int &num_a, int &num_b, uint16_t *group_a, uint16_t *group_b, const BVHABB_CLASS *temp_bounds, const BVHABB_CLASS full_bound) {
		// special case for low leaf sizes .. should static compile out
		if constexpr (MAX_ITEMS < 4) {
			uint32_t ind = group_a[0];

			// add to b
			group_b[num_b++] = ind;

			// remove from a
			num_a--;
			group_a[0] = group_a[num_a];
			return;
		}

		POINT center = full_bound.calculate_center();
		POINT size = full_bound.calculate_size();

		int order[POINT::AXIS_COUNT];

		order[0] = size.max_axis_index(); // The longest axis.
		order[POINT::AXIS_COUNT - 1] = size.min_axis_index(); // The shortest axis.

		static_assert(POINT::AXIS_COUNT <= 3, "BVH POINT::AXIS_COUNT has unexpected size");
		if constexpr (POINT::AXIS_COUNT == 3) {
			order[1] = 3 - (order[0] + order[2]);
		}

		// Simplest case, split on the longest axis.
		int split_axis = order[0];
		for (int a = 0; a < num_a; a++) {
			uint32_t ind = group_a[a];

			if (temp_bounds[ind].min.coord[split_axis] > center.coord[split_axis]) {
				// add to b
				group_b[num_b++] = ind;

				// remove from a
				num_a--;
				group_a[a] = group_a[num_a];

				// do this one again, as it has been replaced
				a--;
			}
		}

		// detect when split on longest axis failed
		int min_threshold = MAX_ITEMS / 4;
		int min_group_size[POINT::AXIS_COUNT];
		min_group_size[0] = MIN(num_a, num_b);
		if (min_group_size[0] < min_threshold) {
			// slow but sure .. first move everything back into a
			for (int b = 0; b < num_b; b++) {
				group_a[num_a++] = group_b[b];
			}
			num_b = 0;

			// Now calculate the best split.
			for (int axis = 1; axis < POINT::AXIS_COUNT; axis++) {
				split_axis = order[axis];
				int count = 0;

				for (int a = 0; a < num_a; a++) {
					uint32_t ind = group_a[a];

					if (temp_bounds[ind].min.coord[split_axis] > center.coord[split_axis]) {
						count++;
					}
				}

				min_group_size[axis] = MIN(count, num_a - count);
			} // for axis

			// best axis
			int best_axis = 0;
			int best_min = min_group_size[0];
			for (int axis = 1; axis < POINT::AXIS_COUNT; axis++) {
				if (min_group_size[axis] > best_min) {
					best_min = min_group_size[axis];
					best_axis = axis;
				}
			}

			// now finally do the split
			if (best_min > 0) {
				split_axis = order[best_axis];

				for (int a = 0; a < num_a; a++) {
					uint32_t ind = group_a[a];

					if (temp_bounds[ind].min.coord[split_axis] > center.coord[split_axis]) {
						// add to b
						group_b[num_b++] = ind;

						// remove from a
						num_a--;
						group_a[a] = group_a[num_a];

						// do this one again, as it has been replaced
						a--;
					}
				}
			} // if there was a split!
		} // if the longest axis wasn't a good split

		// special case, none crossed threshold
		if (!num_b) {
			uint32_t ind = group_a[0];

			// add to b
			group_b[num_b++] = ind;

			// remove from a
			num_a--;
			group_a[0] = group_a[num_a];
		}
		// opposite problem! :)
		if (!num_a) {
			uint32_t ind = group_b[0];

			// add to a
			group_a[num_a++] = ind;

			// remove from b
			num_b--;
			group_b[0] = group_b[num_b];
		}
	}

	void _split_leaf_sort_groups(int &num_a, int &num_b, uint16_t *group_a, uint16_t *group_b, const BVHABB_CLASS *temp_bounds) {
		BVHABB_CLASS groupb_aabb;
		groupb_aabb.set_to_max_opposite_extents();
		for (int n = 0; n < num_b; n++) {
			int which = group_b[n];
			groupb_aabb.merge(temp_bounds[which]);
		}
		BVHABB_CLASS groupb_aabb_new;

		BVHABB_CLASS rest_aabb;

		real_t best_size = FLT_MAX;
		int best_candidate = -1;

		// find most likely from a to move into b
		for (int check = 0; check < num_a; check++) {
			rest_aabb.set_to_max_opposite_extents();
			groupb_aabb_new = groupb_aabb;

			// find aabb of all the rest
			for (int rest = 0; rest < num_a; rest++) {
				if (rest == check) {
					continue;
				}

				int which = group_a[rest];
				rest_aabb.merge(temp_bounds[which]);
			}

			groupb_aabb_new.merge(temp_bounds[group_a[check]]);

			// now compare the sizes
			real_t size = groupb_aabb_new.get_area() + rest_aabb.get_area();
			if (size < best_size) {
				best_size = size;
				best_candidate = check;
			}
		}

		// we should now have the best, move it from group a to group b
		group_b[num_b++] = group_a[best_candidate];

		// remove best candidate from group a
		num_a--;
		group_a[best_candidate] = group_a[num_a];
	}

	uint32_t split_leaf(uint32_t p_node_id, const BVHABB_CLASS &p_added_item_aabb) {
		return split_leaf_complex(p_node_id, p_added_item_aabb);
	}

	// aabb is the new inserted node
	uint32_t split_leaf_complex(uint32_t p_node_id, const BVHABB_CLASS &p_added_item_aabb) {
		VERBOSE_PRINT("split_leaf");

		// note the tnode before and AFTER splitting may be a different address
		// in memory because the vector could get relocated. So we need to reget
		// the tnode after the split
		BVH_ASSERT(_nodes[p_node_id].is_leaf());

		// first create child leaf nodes
		uint32_t *child_ids = (uint32_t *)alloca(sizeof(uint32_t) * MAX_CHILDREN);

		for (int n = 0; n < MAX_CHILDREN; n++) {
			// create node children
			TNode *child_node = _nodes.request(child_ids[n]);

			child_node->clear();

			// back link to parent
			child_node->parent_id = p_node_id;

			// make each child a leaf node
			node_make_leaf(child_ids[n]);
		}

		// don't get any leaves or nodes till AFTER the split
		TNode &tnode = _nodes[p_node_id];
		uint32_t orig_leaf_id = tnode.get_leaf_id();
		const TLeaf &orig_leaf = _node_get_leaf(tnode);

		// store the final child ids
		for (int n = 0; n < MAX_CHILDREN; n++) {
			tnode.children[n] = child_ids[n];
		}

		// mark as no longer a leaf node
		tnode.num_children = MAX_CHILDREN;

		// 2 groups, A and B, and assign children to each to split equally
		int max_children = orig_leaf.num_items + 1; // plus 1 for the wildcard .. the item being added
		//CRASH_COND(max_children > MAX_CHILDREN);

		uint16_t *group_a = (uint16_t *)alloca(sizeof(uint16_t) * max_children);
		uint16_t *group_b = (uint16_t *)alloca(sizeof(uint16_t) * max_children);

		// we are copying the ABBs. This is ugly, but we need one extra for the inserted item...
		BVHABB_CLASS *temp_bounds = (BVHABB_CLASS *)alloca(sizeof(BVHABB_CLASS) * max_children);

		int num_a = max_children;
		int num_b = 0;

		// setup - start with all in group a
		for (int n = 0; n < orig_leaf.num_items; n++) {
			group_a[n] = n;
			temp_bounds[n] = orig_leaf.get_aabb(n);
		}
		// wildcard
		int wildcard = orig_leaf.num_items;

		group_a[wildcard] = wildcard;
		temp_bounds[wildcard] = p_added_item_aabb;

		// we can choose here either an equal split, or just 1 in the new leaf
		_split_leaf_sort_groups_simple(num_a, num_b, group_a, group_b, temp_bounds, tnode.aabb);

		uint32_t wildcard_node = BVHCommon::INVALID;

		// now there should be equal numbers in both groups
		for (int n = 0; n < num_a; n++) {
			int which = group_a[n];

			if (which != wildcard) {
				const BVHABB_CLASS &source_item_aabb = orig_leaf.get_aabb(which);
				uint32_t source_item_ref_id = orig_leaf.get_item_ref_id(which);
				//const Item &source_item = orig_leaf.get_item(which);
				_node_add_item(tnode.children[0], source_item_ref_id, source_item_aabb);
			} else {
				wildcard_node = tnode.children[0];
			}
		}
		for (int n = 0; n < num_b; n++) {
			int which = group_b[n];

			if (which != wildcard) {
				const BVHABB_CLASS &source_item_aabb = orig_leaf.get_aabb(which);
				uint32_t source_item_ref_id = orig_leaf.get_item_ref_id(which);
				//const Item &source_item = orig_leaf.get_item(which);
				_node_add_item(tnode.children[1], source_item_ref_id, source_item_aabb);
			} else {
				wildcard_node = tnode.children[1];
			}
		}

		// now remove all items from the parent and replace with the child nodes
		_leaves.free(orig_leaf_id);

		// we should keep the references up to date!
		for (int n = 0; n < MAX_CHILDREN; n++) {
			_split_inform_references(tnode.children[n]);
		}

		refit_upward(p_node_id);

		BVH_ASSERT(wildcard_node != BVHCommon::INVALID);
		return wildcard_node;
	}
#pragma endregion Split
};

#undef VERBOSE_PRINT

#endif // BVH_TREE_H
