public:
#ifdef BVH_VERBOSE
void _debug_recursive_print_tree(int p_tree_id) const {
	if (_root_node_id[p_tree_id] != BVHCommon::INVALID) {
		_debug_recursive_print_tree_node(_root_node_id[p_tree_id]);
	}
}

String _debug_aabb_to_string(const BVHABB_CLASS &aabb) const {
	POINT size = aabb.calculate_size();

	String sz;
	float vol = 0.0;

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
