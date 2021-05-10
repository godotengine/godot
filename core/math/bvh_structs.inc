
public:
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
	uint32_t last_updated_tick;
	uint32_t pairable;
	uint32_t pairable_mask;
	uint32_t pairable_type;

	int32_t subindex;

	// the active reference is a separate list of which references
	// are active so that we can slowly iterate through it over many frames for
	// slow optimize.
	uint32_t active_ref_id;

	T *userdata;
};

// this is an item OR a child node depending on whether a leaf node
struct Item {
	BVHABB_CLASS aabb;
	uint32_t item_ref_id;
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
	BVHABB_CLASS &get_aabb(uint32_t p_id) { return aabbs[p_id]; }
	const BVHABB_CLASS &get_aabb(uint32_t p_id) const { return aabbs[p_id]; }

	uint32_t &get_item_ref_id(uint32_t p_id) { return item_ref_ids[p_id]; }
	const uint32_t &get_item_ref_id(uint32_t p_id) const { return item_ref_ids[p_id]; }

	bool is_dirty() const { return dirty; }
	void set_dirty(bool p) { dirty = p; }

	void clear() {
		num_items = 0;
		set_dirty(true);
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
		return -1;
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
PooledList<ItemRef, true> _refs;
PooledList<ItemExtra, true> _extra;
PooledList<ItemPairs> _pairs;

// these 2 are not in sync .. nodes != leaves!
PooledList<TNode, true> _nodes;
PooledList<TLeaf, true> _leaves;

// we can maintain an un-ordered list of which references are active,
// in order to do a slow incremental optimize of the tree over each frame.
// This will work best if dynamic objects and static objects are in a different tree.
LocalVector<uint32_t, uint32_t, true> _active_refs;
uint32_t _current_active_ref = 0;

// instead of translating directly to the userdata output,
// we keep an intermediate list of hits as reference IDs, which can be used
// for pairing collision detection
LocalVector<uint32_t, uint32_t, true> _cull_hits;

// we now have multiple root nodes, allowing us to store
// more than 1 tree. This can be more efficient, while sharing the same
// common lists
enum { NUM_TREES = 2,
};

// Tree 0 - Non pairable
// Tree 1 - Pairable
// This is more efficient because in physics we only need check non pairable against the pairable tree.
uint32_t _root_node_id[NUM_TREES];
int _current_tree = 0;

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
bool _auto_pairing_expansion = true;
