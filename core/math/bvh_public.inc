public:
BVHHandle item_add(T *p_userdata, bool p_active, const Bounds &p_aabb, int32_t p_subindex, bool p_pairable, uint32_t p_pairable_type, uint32_t p_pairable_mask, bool p_invisible = false) {
#ifdef BVH_VERBOSE_TREE
	VERBOSE_PRINT("\nitem_add BEFORE");
	_debug_recursive_print_tree(0);
	VERBOSE_PRINT("\n");
#endif

	BVHABB_CLASS abb;
	abb.from(p_aabb);

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

	if (USE_PAIRS) {
		extra->pairable_mask = p_pairable_mask;
		extra->pairable_type = p_pairable_type;
		extra->pairable = p_pairable;
	} else {
		// just for safety, in case this gets queried etc
		extra->pairable = 0;
		p_pairable = false;
	}

	// assign to handle to return
	handle.set_id(ref_id);

	_current_tree = 0;
	if (p_pairable) {
		_current_tree = 1;
	}

	create_root_node(_current_tree);

	// we must choose where to add to tree
	if (p_active) {
		ref->tnode_id = _logic_choose_item_add_node(_root_node_id[_current_tree], abb);

		bool refit = _node_add_item(ref->tnode_id, ref_id, abb);

		if (refit) {
			// only need to refit from the parent
			const TNode &add_node = _nodes[ref->tnode_id];
			if (add_node.parent_id != BVHCommon::INVALID) {
				refit_upward_and_balance(add_node.parent_id);
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
	VERBOSE_PRINT("\titem_add [" + itos(ref_id) + "] " + itos(_refs.size()) + " refs,\t" + itos(_nodes.size()) + " nodes " + sz);
	VERBOSE_PRINT("mem use : " + itos(mem) + ", num nodes : " + itos(_nodes.size()));

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
bool item_move(BVHHandle p_handle, const Bounds &p_aabb) {
	uint32_t ref_id = p_handle.id();

	// get the reference
	ItemRef &ref = _refs[ref_id];
	if (!ref.is_active()) {
		return false;
	}

	BVHABB_CLASS abb;
	abb.from(p_aabb);

	BVH_ASSERT(ref.tnode_id != BVHCommon::INVALID);
	TNode &tnode = _nodes[ref.tnode_id];

	// does it fit within the current aabb?
	if (tnode.aabb.is_other_within(abb)) {
		// do nothing .. fast path .. not moved enough to need refit

		// however we WILL update the exact aabb in the leaf, as this will be needed
		// for accurate collision detection
		TLeaf &leaf = _node_get_leaf(tnode);

		BVHABB_CLASS &leaf_abb = leaf.get_aabb(ref.item_id);

		// no change?
		if (leaf_abb == abb) {
			return false;
		}

		leaf_abb = abb;
		_integrity_check_all();

		return true;
	}

	_current_tree = _handle_get_tree_id(p_handle);

	// remove and reinsert
	node_remove_item(ref_id);

	// we must choose where to add to tree
	ref.tnode_id = _logic_choose_item_add_node(_root_node_id[_current_tree], abb);

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

	_current_tree = _handle_get_tree_id(p_handle);

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
		node_remove_item(ref_id);
	}

	// remove the item reference
	_refs.free(ref_id);
	_extra.free(ref_id);
	if (USE_PAIRS) {
		_pairs.free(ref_id);
	}

	// don't think refit_all is necessary?
	//refit_all(_current_tree);

#ifdef BVH_VERBOSE_TREE
	_debug_recursive_print_tree(_current_tree);
#endif
}

// returns success
bool item_activate(BVHHandle p_handle, const Bounds &p_aabb) {
	uint32_t ref_id = p_handle.id();
	ItemRef &ref = _refs[ref_id];
	if (ref.is_active()) {
		// noop
		return false;
	}

	// add to tree
	BVHABB_CLASS abb;
	abb.from(p_aabb);

	_current_tree = _handle_get_tree_id(p_handle);

	// we must choose where to add to tree
	ref.tnode_id = _logic_choose_item_add_node(_root_node_id[_current_tree], abb);
	_node_add_item(ref.tnode_id, ref_id, abb);

	refit_upward_and_balance(ref.tnode_id);

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

	// remove from tree
	BVHABB_CLASS abb;
	node_remove_item(ref_id, &abb);

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

	// testing from a non pairable item, we only want to test pairable items
	r_params.test_pairable_only = extra.pairable == 0;

	// we take into account the mask of the item testing from
	r_params.mask = extra.pairable_mask;
	r_params.pairable_type = extra.pairable_type;
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

bool item_set_pairable(const BVHHandle &p_handle, bool p_pairable, uint32_t p_pairable_type, uint32_t p_pairable_mask) {
	// change tree?
	uint32_t ref_id = p_handle.id();

	ItemExtra &ex = _extra[ref_id];
	ItemRef &ref = _refs[ref_id];

	bool active = ref.is_active();
	bool pairable_changed = (ex.pairable != 0) != p_pairable;
	bool state_changed = pairable_changed || (ex.pairable_type != p_pairable_type) || (ex.pairable_mask != p_pairable_mask);

	ex.pairable_type = p_pairable_type;
	ex.pairable_mask = p_pairable_mask;

	if (active && pairable_changed) {
		// record abb
		TNode &tnode = _nodes[ref.tnode_id];
		TLeaf &leaf = _node_get_leaf(tnode);
		BVHABB_CLASS abb = leaf.get_aabb(ref.item_id);

		// make sure current tree is correct prior to changing
		_current_tree = _handle_get_tree_id(p_handle);

		// remove from old tree
		node_remove_item(ref_id);

		// we must set the pairable AFTER getting the current tree
		// because the pairable status determines which tree
		ex.pairable = p_pairable;

		// add to new tree
		_current_tree = _handle_get_tree_id(p_handle);
		create_root_node(_current_tree);

		// we must choose where to add to tree
		ref.tnode_id = _logic_choose_item_add_node(_root_node_id[_current_tree], abb);
		bool needs_refit = _node_add_item(ref.tnode_id, ref_id, abb);

		// only need to refit from the PARENT
		if (needs_refit) {
			// only need to refit from the parent
			const TNode &add_node = _nodes[ref.tnode_id];
			if (add_node.parent_id != BVHCommon::INVALID) {
				refit_upward_and_balance(add_node.parent_id);
			}
		}
	} else {
		// always keep this up to date
		ex.pairable = p_pairable;
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
			Bounds bb;
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
