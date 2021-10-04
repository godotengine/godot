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
	if (MAX_ITEMS < 4) {
		uint32_t ind = group_a[0];

		// add to b
		group_b[num_b++] = ind;

		// remove from a
		group_a[0] = group_a[num_a - 1];
		num_a--;
		return;
	}

	Point centre = full_bound.calculate_centre();
	Point size = full_bound.calculate_size();

	int order[Point::AXIS_COUNT];

	order[0] = size.min_axis();
	order[Point::AXIS_COUNT - 1] = size.max_axis();

	static_assert(Point::AXIS_COUNT <= 3, "BVH Point::AXIS_COUNT has unexpected size");
	if (Point::AXIS_COUNT == 3) {
		order[1] = 3 - (order[0] + order[2]);
	}

	// simplest case, split on the longest axis
	int split_axis = order[0];
	for (int a = 0; a < num_a; a++) {
		uint32_t ind = group_a[a];

		if (temp_bounds[ind].min.coord[split_axis] > centre.coord[split_axis]) {
			// add to b
			group_b[num_b++] = ind;

			// remove from a
			group_a[a] = group_a[num_a - 1];
			num_a--;

			// do this one again, as it has been replaced
			a--;
		}
	}

	// detect when split on longest axis failed
	int min_threshold = MAX_ITEMS / 4;
	int min_group_size[Point::AXIS_COUNT];
	min_group_size[0] = MIN(num_a, num_b);
	if (min_group_size[0] < min_threshold) {
		// slow but sure .. first move everything back into a
		for (int b = 0; b < num_b; b++) {
			group_a[num_a++] = group_b[b];
		}
		num_b = 0;

		// now calculate the best split
		for (int axis = 1; axis < Point::AXIS_COUNT; axis++) {
			split_axis = order[axis];
			int count = 0;

			for (int a = 0; a < num_a; a++) {
				uint32_t ind = group_a[a];

				if (temp_bounds[ind].min.coord[split_axis] > centre.coord[split_axis]) {
					count++;
				}
			}

			min_group_size[axis] = MIN(count, num_a - count);
		} // for axis

		// best axis
		int best_axis = 0;
		int best_min = min_group_size[0];
		for (int axis = 1; axis < Point::AXIS_COUNT; axis++) {
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

				if (temp_bounds[ind].min.coord[split_axis] > centre.coord[split_axis]) {
					// add to b
					group_b[num_b++] = ind;

					// remove from a
					group_a[a] = group_a[num_a - 1];
					num_a--;

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
		group_a[0] = group_a[num_a - 1];
		num_a--;
	}
	// opposite problem! :)
	if (!num_a) {
		uint32_t ind = group_b[0];

		// add to a
		group_a[num_a++] = ind;

		// remove from b
		group_b[0] = group_b[num_b - 1];
		num_b--;
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

	float best_size = FLT_MAX;
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
		float size = groupb_aabb_new.get_area() + rest_aabb.get_area();
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
