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

void refit_upward_and_balance(uint32_t p_node_id) {
	while (p_node_id != BVHCommon::INVALID) {
		uint32_t before = p_node_id;
		p_node_id = _logic_balance(p_node_id);

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
				refit_upward(p_node_id);
			}
		}
	} // while more nodes to pop
}
