public:
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
	if (p_value < 0) {
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
	const real_t fudge_factor = 1.1f;
	_aabb_shrinkage_threshold = _pairing_expansion * POINT::AXIS_COUNT * 2 * fudge_factor;
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

	real_t exp_l = 0;
	real_t new_l = 0;

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
