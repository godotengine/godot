/*************************************************************************/
/*  bvh.h                                                                */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#ifndef BVH_H
#define BVH_H

// BVH
// This class provides a wrapper around BVH tree, which contains most of the functionality
// for a dynamic BVH with templated leaf size.
// However BVH also adds facilities for pairing, to maintain compatibility with Godot 3.2.
// Pairing is a collision pairing system, on top of the basic BVH.

// Some notes on the use of BVH / Octree from Godot 3.2.
// This is not well explained elsewhere.
// The rendering tree mask and types that are sent to the BVH are NOT layer masks.
// They are INSTANCE_TYPES (defined in visual_server.h), e.g. MESH, MULTIMESH, PARTICLES etc.
// Thus the lights do no cull by layer mask in the BVH.

// Layer masks are implemented in the renderers as a later step, and light_cull_mask appears to be
// implemented in GLES3 but not GLES2. Layer masks are not yet implemented for directional lights.

#include "bvh_tree.h"

#define BVHTREE_CLASS BVH_Tree<T, 2, MAX_ITEMS, USE_PAIRS>

template <class T, bool USE_PAIRS = false, int MAX_ITEMS = 32>
class BVH_Manager {

public:
	// note we are using uint32_t instead of BVHHandle, losing type safety, but this
	// is for compatibility with octree
	typedef void *(*PairCallback)(void *, uint32_t, T *, int, uint32_t, T *, int);
	typedef void (*UnpairCallback)(void *, uint32_t, T *, int, uint32_t, T *, int, void *);

	// these 2 are crucial for fine tuning, and can be applied manually
	// see the variable declarations for more info.
	void params_set_node_expansion(real_t p_value) {
		if (p_value >= 0.0) {
			tree._node_expansion = p_value;
			tree._auto_node_expansion = false;
		} else {
			tree._auto_node_expansion = true;
		}
	}

	void params_set_pairing_expansion(real_t p_value) {
		if (p_value >= 0.0) {
			tree._pairing_expansion = p_value;
			tree._auto_pairing_expansion = false;
		} else {
			tree._auto_pairing_expansion = true;
		}
	}

	void set_pair_callback(PairCallback p_callback, void *p_userdata) {
		pair_callback = p_callback;
		pair_callback_userdata = p_userdata;
	}
	void set_unpair_callback(UnpairCallback p_callback, void *p_userdata) {
		unpair_callback = p_callback;
		unpair_callback_userdata = p_userdata;
	}

	BVHHandle create(T *p_userdata, bool p_active, const AABB &p_aabb = AABB(), int p_subindex = 0, bool p_pairable = false, uint32_t p_pairable_type = 0, uint32_t p_pairable_mask = 1) {

		// not sure if absolutely necessary to flush collisions here. It will cost performance to, instead
		// of waiting for update, so only uncomment this if there are bugs.
		if (USE_PAIRS) {
			//_check_for_collisions();
		}

#ifdef TOOLS_ENABLED
		if (!USE_PAIRS) {
			if (p_pairable) {
				WARN_PRINT_ONCE("creating pairable item in BVH with USE_PAIRS set to false");
			}
		}
#endif

		BVHHandle h = tree.item_add(p_userdata, p_active, p_aabb, p_subindex, p_pairable, p_pairable_type, p_pairable_mask);

		if (USE_PAIRS) {
			// for safety initialize the expanded AABB
			AABB &expanded_aabb = tree._pairs[h.id()].expanded_aabb;
			expanded_aabb = p_aabb;
			expanded_aabb.grow_by(tree._pairing_expansion);

			// force a collision check no matter the AABB
			if (p_active) {
				_add_changed_item(h, p_aabb, false);
				_check_for_collisions(true);
			}
		}

		return h;
	}

	////////////////////////////////////////////////////
	// wrapper versions that use uint32_t instead of handle
	// for backward compatibility. Less type safe
	void move(uint32_t p_handle, const AABB &p_aabb) {
		BVHHandle h;
		h.set(p_handle);
		move(h, p_aabb);
	}

	void erase(uint32_t p_handle) {
		BVHHandle h;
		h.set(p_handle);
		erase(h);
	}

	bool activate(uint32_t p_handle, const AABB &p_aabb, bool p_delay_collision_check = false) {
		BVHHandle h;
		h.set(p_handle);
		return activate(h, p_aabb, p_delay_collision_check);
	}

	bool deactivate(uint32_t p_handle) {
		BVHHandle h;
		h.set(p_handle);
		return deactivate(h);
	}

	void set_pairable(uint32_t p_handle, bool p_pairable, uint32_t p_pairable_type, uint32_t p_pairable_mask) {
		BVHHandle h;
		h.set(p_handle);
		set_pairable(h, p_pairable, p_pairable_type, p_pairable_mask);
	}

	bool is_pairable(uint32_t p_handle) const {
		BVHHandle h;
		h.set(p_handle);
		return item_is_pairable(h);
	}
	int get_subindex(uint32_t p_handle) const {
		BVHHandle h;
		h.set(p_handle);
		return item_get_subindex(h);
	}

	T *get(uint32_t p_handle) const {
		BVHHandle h;
		h.set(p_handle);
		return item_get_userdata(h);
	}

	////////////////////////////////////////////////////

	void move(BVHHandle p_handle, const AABB &p_aabb) {

		if (tree.item_move(p_handle, p_aabb)) {
			if (USE_PAIRS) {
				_add_changed_item(p_handle, p_aabb);
			}
		}
	}

	void erase(BVHHandle p_handle) {
		// call unpair and remove all references to the item
		// before deleting from the tree
		if (USE_PAIRS) {
			_remove_changed_item(p_handle);
		}

		tree.item_remove(p_handle);

		_check_for_collisions(true);
	}

	// these should be read as set_visible for render trees,
	// but generically this makes items add or remove from the
	// tree internally, to speed things up by ignoring inactive items
	bool activate(BVHHandle p_handle, const AABB &p_aabb, bool p_delay_collision_check = false) {
		// sending the aabb here prevents the need for the BVH to maintain
		// a redundant copy of the aabb.
		// returns success
		if (tree.item_activate(p_handle, p_aabb)) {
			if (USE_PAIRS) {

				// in the special case of the render tree, when setting visibility we are using the combination of
				// activate then set_pairable. This would case 2 sets of collision checks. For efficiency here we allow
				// deferring to have a single collision check at the set_pairable call.
				// Watch for bugs! This may cause bugs if set_pairable is not called.
				if (!p_delay_collision_check) {
					_add_changed_item(p_handle, p_aabb, false);

					// force an immediate collision check, much like calls to set_pairable
					_check_for_collisions(true);
				}
			}
			return true;
		}

		return false;
	}

	bool deactivate(BVHHandle p_handle) {
		// returns success
		if (tree.item_deactivate(p_handle)) {
			// call unpair and remove all references to the item
			// before deleting from the tree
			if (USE_PAIRS) {
				_remove_changed_item(p_handle);

				// force check for collisions, much like an erase was called
				_check_for_collisions(true);
			}
			return true;
		}

		return false;
	}

	bool get_active(BVHHandle p_handle) const {
		return tree.item_get_active(p_handle);
	}

	// call e.g. once per frame (this does a trickle optimize)
	void update() {
		tree.update();
		_check_for_collisions();
#ifdef BVH_INTEGRITY_CHECKS
		tree.integrity_check_all();
#endif
	}

	// this can be called more frequently than per frame if necessary
	void update_collisions() {
		_check_for_collisions();
	}

	// prefer calling this directly as type safe
	void set_pairable(const BVHHandle &p_handle, bool p_pairable, uint32_t p_pairable_type, uint32_t p_pairable_mask) {
		tree.item_set_pairable(p_handle, p_pairable, p_pairable_type, p_pairable_mask);

		if (USE_PAIRS) {

			// not sure if absolutely necessary to flush collisions here. It will cost performance to, instead
			// of waiting for update, so only uncomment this if there are bugs.
			//_check_for_collisions();

			if (get_active(p_handle)) {
				// when the pairable state changes, we need to force a collision check because newly pairable
				// items may be in collision, and unpairable items might move out of collision.
				// We cannot depend on waiting for the next update, because that may come much later.
				AABB aabb;
				item_get_AABB(p_handle, aabb);

				// passing false disables the optimization which prevents collision checks if
				// the aabb hasn't changed
				_add_changed_item(p_handle, aabb, false);

				// force an immediate collision check (probably just for this one item)
				// but it must be a FULL collision check, also checking pairable state and masks.
				// This is because AABB intersecting objects may have changed pairable state / mask
				// such that they should no longer be paired. E.g. lights.
				_check_for_collisions(true);
			} // only if active
		}
	}

	// cull tests
	int cull_aabb(const AABB &p_aabb, T **p_result_array, int p_result_max, int *p_subindex_array = nullptr, uint32_t p_mask = 0xFFFFFFFF) {
		typename BVHTREE_CLASS::CullParams params;

		params.result_count_overall = 0;
		params.result_max = p_result_max;
		params.result_array = p_result_array;
		params.subindex_array = p_subindex_array;
		params.mask = p_mask;
		params.pairable_type = 0;
		params.test_pairable_only = false;
		params.abb.from(p_aabb);

		tree.cull_aabb(params);

		return params.result_count_overall;
	}

	int cull_segment(const Vector3 &p_from, const Vector3 &p_to, T **p_result_array, int p_result_max, int *p_subindex_array = nullptr, uint32_t p_mask = 0xFFFFFFFF) {
		typename BVHTREE_CLASS::CullParams params;

		params.result_count_overall = 0;
		params.result_max = p_result_max;
		params.result_array = p_result_array;
		params.subindex_array = p_subindex_array;
		params.mask = p_mask;
		params.pairable_type = 0;

		params.segment.from = p_from;
		params.segment.to = p_to;

		tree.cull_segment(params);

		return params.result_count_overall;
	}

	int cull_point(const Vector3 &p_point, T **p_result_array, int p_result_max, int *p_subindex_array = nullptr, uint32_t p_mask = 0xFFFFFFFF) {
		typename BVHTREE_CLASS::CullParams params;

		params.result_count_overall = 0;
		params.result_max = p_result_max;
		params.result_array = p_result_array;
		params.subindex_array = p_subindex_array;
		params.mask = p_mask;
		params.pairable_type = 0;

		params.point = p_point;

		tree.cull_point(params);
		return params.result_count_overall;
	}

	int cull_convex(const Vector<Plane> &p_convex, T **p_result_array, int p_result_max, uint32_t p_mask = 0xFFFFFFFF) {
		if (!p_convex.size())
			return 0;

		Vector<Vector3> convex_points = Geometry::compute_convex_mesh_points(&p_convex[0], p_convex.size());
		if (convex_points.size() == 0)
			return 0;

		typename BVHTREE_CLASS::CullParams params;
		params.result_count_overall = 0;
		params.result_max = p_result_max;
		params.result_array = p_result_array;
		params.subindex_array = nullptr;
		params.mask = p_mask;
		params.pairable_type = 0;

		params.hull.planes = &p_convex[0];
		params.hull.num_planes = p_convex.size();
		params.hull.points = &convex_points[0];
		params.hull.num_points = convex_points.size();

		tree.cull_convex(params);

		return params.result_count_overall;
	}

private:
	// do this after moving etc.
	void _check_for_collisions(bool p_full_check = false) {
		if (!changed_items.size()) {
			// noop
			return;
		}

		AABB bb;

		typename BVHTREE_CLASS::CullParams params;

		params.result_count_overall = 0;
		params.result_max = INT_MAX;
		params.result_array = nullptr;
		params.subindex_array = nullptr;
		params.mask = 0xFFFFFFFF;
		params.pairable_type = 0;

		for (unsigned int n = 0; n < changed_items.size(); n++) {
			const BVHHandle &h = changed_items[n];

			// use the expanded aabb for pairing
			const AABB &expanded_aabb = tree._pairs[h.id()].expanded_aabb;
			BVH_ABB abb;
			abb.from(expanded_aabb);

			// find all the existing paired aabbs that are no longer
			// paired, and send callbacks
			_find_leavers(h, abb, p_full_check);

			uint32_t changed_item_ref_id = h.id();

			// set up the test from this item.
			// this includes whether to test the non pairable tree,
			// and the item mask.
			tree.item_fill_cullparams(h, params);

			params.abb = abb;

			params.result_count_overall = 0; // might not be needed
			tree.cull_aabb(params, false);

			for (unsigned int i = 0; i < tree._cull_hits.size(); i++) {
				uint32_t ref_id = tree._cull_hits[i];

				// don't collide against ourself
				if (ref_id == changed_item_ref_id)
					continue;

#ifdef BVH_CHECKS
				// if neither are pairable, they should ignore each other
				// THIS SHOULD NEVER HAPPEN .. now we only test the pairable tree
				// if the changed item is not pairable
				CRASH_COND(params.test_pairable_only && !tree._extra[ref_id].pairable);
#endif

				// checkmasks is already done in the cull routine.
				BVHHandle h_collidee;
				h_collidee.set_id(ref_id);

				// find NEW enterers, and send callbacks for them only
				_collide(h, h_collidee);
			}
		}
		_reset();
	}

public:
	void item_get_AABB(BVHHandle p_handle, AABB &r_aabb) {
		BVH_ABB abb;
		tree.item_get_ABB(p_handle, abb);
		abb.to(r_aabb);
	}

private:
	// supplemental funcs
	bool item_is_pairable(BVHHandle p_handle) const { return _get_extra(p_handle).pairable; }
	T *item_get_userdata(BVHHandle p_handle) const { return _get_extra(p_handle).userdata; }
	int item_get_subindex(BVHHandle p_handle) const { return _get_extra(p_handle).subindex; }

	void _unpair(BVHHandle p_from, BVHHandle p_to) {
		tree._handle_sort(p_from, p_to);

		typename BVHTREE_CLASS::ItemPairs &pairs_from = tree._pairs[p_from.id()];
		typename BVHTREE_CLASS::ItemPairs &pairs_to = tree._pairs[p_to.id()];

		void *ud_from = pairs_from.remove_pair_to(p_to);
		pairs_to.remove_pair_to(p_from);

		// callback
		if (unpair_callback) {

			typename BVHTREE_CLASS::ItemExtra &exa = tree._extra[p_from.id()];
			typename BVHTREE_CLASS::ItemExtra &exb = tree._extra[p_to.id()];

			unpair_callback(pair_callback_userdata, p_from, exa.userdata, exa.subindex, p_to, exb.userdata, exb.subindex, ud_from);
		}
	}

	// returns true if unpair
	bool _find_leavers_process_pair(typename BVHTREE_CLASS::ItemPairs &p_pairs_from, const BVH_ABB &p_abb_from, BVHHandle p_from, BVHHandle p_to, bool p_full_check) {
		BVH_ABB abb_to;
		tree.item_get_ABB(p_to, abb_to);

		// do they overlap?
		if (p_abb_from.intersects(abb_to)) {
			// the full check for pairable / non pairable and mask changes is extra expense
			// this need not be done in most cases (for speed) except in the case where set_pairable is called
			// where the masks etc of the objects in question may have changed
			if (!p_full_check) {
				return false;
			}
			const typename BVHTREE_CLASS::ItemExtra &exa = _get_extra(p_from);
			const typename BVHTREE_CLASS::ItemExtra &exb = _get_extra(p_to);

			// one of the two must be pairable to still pair
			// if neither are pairable, we always unpair
			if (exa.pairable || exb.pairable) {
				// the masks must still be compatible to pair
				// i.e. if there is a hit between the two, then they should stay paired
				if (tree._cull_pairing_mask_test_hit(exa.pairable_mask, exa.pairable_type, exb.pairable_mask, exb.pairable_type)) {
					return false;
				}
			}
		}

		_unpair(p_from, p_to);
		return true;
	}

	// find all the existing paired aabbs that are no longer
	// paired, and send callbacks
	void _find_leavers(BVHHandle p_handle, const BVH_ABB &expanded_abb_from, bool p_full_check) {
		typename BVHTREE_CLASS::ItemPairs &p_from = tree._pairs[p_handle.id()];

		BVH_ABB abb_from = expanded_abb_from;

		// remove from pairing list for every partner
		for (unsigned int n = 0; n < p_from.extended_pairs.size(); n++) {
			BVHHandle h_to = p_from.extended_pairs[n].handle;
			if (_find_leavers_process_pair(p_from, abb_from, p_handle, h_to, p_full_check)) {
				// we need to keep the counter n up to date if we deleted a pair
				// as the number of items in p_from.extended_pairs will have decreased by 1
				// and we don't want to miss an item
				n--;
			}
		}
	}

	// find NEW enterers, and send callbacks for them only
	// handle a and b
	void _collide(BVHHandle p_ha, BVHHandle p_hb) {
		// only have to do this oneway, lower ID then higher ID
		tree._handle_sort(p_ha, p_hb);

		typename BVHTREE_CLASS::ItemPairs &p_from = tree._pairs[p_ha.id()];
		typename BVHTREE_CLASS::ItemPairs &p_to = tree._pairs[p_hb.id()];

		// does this pair exist already?
		// or only check the one with lower number of pairs for greater speed
		if (p_from.num_pairs <= p_to.num_pairs) {
			if (p_from.contains_pair_to(p_hb))
				return;
		} else {
			if (p_to.contains_pair_to(p_ha))
				return;
		}

		// callback
		void *callback_userdata = nullptr;

		if (pair_callback) {
			const typename BVHTREE_CLASS::ItemExtra &exa = _get_extra(p_ha);
			const typename BVHTREE_CLASS::ItemExtra &exb = _get_extra(p_hb);

			callback_userdata = pair_callback(pair_callback_userdata, p_ha, exa.userdata, exa.subindex, p_hb, exb.userdata, exb.subindex);
		}

		// new pair! .. only really need to store the userdata on the lower handle, but both have storage so...
		p_from.add_pair_to(p_hb, callback_userdata);
		p_to.add_pair_to(p_ha, callback_userdata);
	}

	// if we remove an item, we need to immediately remove the pairs, to prevent reading the pair after deletion
	void _remove_pairs_containing(BVHHandle p_handle) {

		typename BVHTREE_CLASS::ItemPairs &p_from = tree._pairs[p_handle.id()];

		// remove from pairing list for every partner.
		// can't easily use a for loop here, because removing changes the size of the list
		while (p_from.extended_pairs.size()) {
			BVHHandle h_to = p_from.extended_pairs[0].handle;
			_unpair(p_handle, h_to);
		}
	}

private:
	const typename BVHTREE_CLASS::ItemExtra &_get_extra(BVHHandle p_handle) const {
		return tree._extra[p_handle.id()];
	}
	const typename BVHTREE_CLASS::ItemRef &_get_ref(BVHHandle p_handle) const {
		return tree._refs[p_handle.id()];
	}

	void _reset() {
		changed_items.clear();
		_tick++;
	}

	void _add_changed_item(BVHHandle p_handle, const AABB &aabb, bool p_check_aabb = true) {

		// Note that non pairable items can pair with pairable,
		// so all types must be added to the list

		// aabb check with expanded aabb. This greatly decreases processing
		// at the cost of slightly less accurate pairing checks
		// Note this pairing AABB is separate from the AABB in the actual tree
		AABB &expanded_aabb = tree._pairs[p_handle.id()].expanded_aabb;

		// passing p_check_aabb false disables the optimization which prevents collision checks if
		// the aabb hasn't changed. This is needed where set_pairable has been called, but the position
		// has not changed.
		if (p_check_aabb && expanded_aabb.encloses(aabb))
			return;

		// ALWAYS update the new expanded aabb, even if already changed once
		// this tick, because it is vital that the AABB is kept up to date
		expanded_aabb = aabb;
		expanded_aabb.grow_by(tree._pairing_expansion);

		// this code is to ensure that changed items only appear once on the updated list
		// collision checking them multiple times is not needed, and repeats the same thing
		uint32_t &last_updated_tick = tree._extra[p_handle.id()].last_updated_tick;

		if (last_updated_tick == _tick)
			return; // already on changed list

		// mark as on list
		last_updated_tick = _tick;

		// add to the list
		changed_items.push_back(p_handle);
	}

	void _remove_changed_item(BVHHandle p_handle) {

		// Care has to be taken here for items that are deleted. The ref ID
		// could be reused on the same tick for new items. This is probably
		// rare but should be taken into consideration

		// callbacks
		_remove_pairs_containing(p_handle);

		// remove from changed items (not very efficient yet)
		for (int n = 0; n < (int)changed_items.size(); n++) {
			if (changed_items[n] == p_handle) {
				changed_items.remove_unordered(n);

				// because we are using an unordered remove,
				// the last changed item will now be at spot 'n',
				// and we need to redo it, so we prevent moving on to
				// the next n at the next for iteration.
				n--;
			}
		}

		// reset the last updated tick (may not be necessary but just in case)
		tree._extra[p_handle.id()].last_updated_tick = 0;
	}

	PairCallback pair_callback;
	UnpairCallback unpair_callback;
	void *pair_callback_userdata;
	void *unpair_callback_userdata;

	BVHTREE_CLASS tree;

	// for collision pairing,
	// maintain a list of all items moved etc on each frame / tick
	LocalVector<BVHHandle, uint32_t, true> changed_items;
	uint32_t _tick;

public:
	BVH_Manager() {
		_tick = 1; // start from 1 so items with 0 indicate never updated
		pair_callback = nullptr;
		unpair_callback = nullptr;
		pair_callback_userdata = nullptr;
		unpair_callback_userdata = nullptr;
	}
};

#undef BVHTREE_CLASS

#endif // BVH_H
