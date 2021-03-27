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
