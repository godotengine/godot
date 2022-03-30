public:
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
