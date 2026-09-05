/**************************************************************************/
/*  cycle_collector.cpp                                                   */
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

#include "cycle_collector.h"

#include "core/object/ref_counted.h"
#include "core/object/script_instance.h"
#include "core/templates/hash_map.h"
#include "core/templates/local_vector.h"
#include "core/typedefs.h"
#include "core/variant/array.h"
#include "core/variant/dictionary.h"

Mutex CycleCollector::mutex;
HashSet<RefCounted *> CycleCollector::candidates;

void CycleCollector::possible_root(RefCounted *p_object) {
	MutexLock lock(mutex);
	if (!p_object->_gc_buffered) {
		p_object->_gc_buffered = true;
		candidates.insert(p_object);
	}
}

void CycleCollector::remove_candidate(RefCounted *p_object) {
	MutexLock lock(mutex);
	if (p_object->_gc_buffered) {
		p_object->_gc_buffered = false;
		candidates.erase(p_object);
	}
}

namespace {

enum class GCColor : uint8_t {
	BLACK, // Assumed live: either not yet visited, or proven reachable from outside the traversed subgraph.
	GRAY, // Currently being visited; its scratch refcount has been trial-decremented along every internal edge found so far.
	WHITE, // Provisionally garbage; freed at the end unless a later scan reaches it and proves it live.
};

struct GCNode {
	GCColor color = GCColor::BLACK;
	int rc = 0;
};

// All state for a single collect_cycles() pass. Keyed by raw pointer: every object referenced
// here is kept alive by the ordinary reference-counting mechanism for the pass's duration
// (either it's a root in `candidates`, which by definition has an outstanding real reference,
// or it was reached by following a live Ref-typed edge from one).
using GCState = HashMap<RefCounted *, GCNode>;

// Appends every RefCounted directly or indirectly (through Array/Dictionary nesting) referenced
// by p_value. Mirrors the recursion-depth guard `MAX_RECURSION` (core/typedefs.h) already used
// elsewhere in core for the same self-referencing-container hazard (see Variant::stringify).
void _collect_edges(const Variant &p_value, LocalVector<RefCounted *> &r_edges, int p_recursion = 0) {
	if (p_recursion > MAX_RECURSION) {
		return;
	}
	switch (p_value.get_type()) {
		case Variant::OBJECT: {
			RefCounted *rc = Object::cast_to<RefCounted>(p_value.get_validated_object());
			if (rc) {
				r_edges.push_back(rc);
			}
		} break;
		case Variant::ARRAY: {
			Array arr = p_value;
			for (int i = 0; i < arr.size(); i++) {
				_collect_edges(arr[i], r_edges, p_recursion + 1);
			}
		} break;
		case Variant::DICTIONARY: {
			Dictionary dict = p_value;
			for (const KeyValue<Variant, Variant> &kv : dict) {
				_collect_edges(kv.key, r_edges, p_recursion + 1);
				_collect_edges(kv.value, r_edges, p_recursion + 1);
			}
		} break;
		default:
			break;
	}
}

// Outgoing edges of a candidate are only followed through its script instance's traversable
// members (see ScriptInstance::get_gc_member_count/get_gc_member) -- never through arbitrary
// exposed properties/getters, which could re-enter user code mid-collection.
LocalVector<RefCounted *> _outgoing_edges(RefCounted *p_object) {
	LocalVector<RefCounted *> edges;
	ScriptInstance *si = p_object->get_script_instance();
	if (!si) {
		return edges;
	}
	int count = si->get_gc_member_count();
	for (int i = 0; i < count; i++) {
		_collect_edges(si->get_gc_member(i), edges);
	}
	return edges;
}

GCNode &_get_or_init_node(GCState &r_state, RefCounted *p_object) {
	HashMap<RefCounted *, GCNode>::Iterator E = r_state.find(p_object);
	if (E) {
		return E->value;
	}
	GCNode node;
	node.color = GCColor::BLACK;
	node.rc = p_object->get_reference_count();
	return r_state.insert(p_object, node)->value;
}

void _mark_gray(RefCounted *p_object, GCState &r_state) {
	GCNode &node = _get_or_init_node(r_state, p_object);
	if (node.color == GCColor::GRAY) {
		return;
	}
	node.color = GCColor::GRAY;
	for (RefCounted *next : _outgoing_edges(p_object)) {
		GCNode &next_node = _get_or_init_node(r_state, next);
		next_node.rc -= 1;
		_mark_gray(next, r_state);
	}
}

void _scan_black(RefCounted *p_object, GCState &r_state) {
	GCNode &node = _get_or_init_node(r_state, p_object);
	node.color = GCColor::BLACK;
	for (RefCounted *next : _outgoing_edges(p_object)) {
		GCNode &next_node = _get_or_init_node(r_state, next);
		next_node.rc += 1;
		if (next_node.color != GCColor::BLACK) {
			_scan_black(next, r_state);
		}
	}
}

void _scan(RefCounted *p_object, GCState &r_state) {
	GCNode &node = _get_or_init_node(r_state, p_object);
	if (node.color != GCColor::GRAY) {
		return;
	}
	if (node.rc > 0) {
		_scan_black(p_object, r_state);
	} else {
		node.color = GCColor::WHITE;
		for (RefCounted *next : _outgoing_edges(p_object)) {
			_scan(next, r_state);
		}
	}
}

// Identifies every WHITE object reachable from p_object, appending it to r_freed. Does not
// free anything itself -- see the big comment on the finalization step in collect_cycles()
// for why a naive memdelete() here would be unsafe (double-free via cross-member cascading).
void _collect_white(RefCounted *p_object, GCState &r_state, LocalVector<RefCounted *> &r_freed) {
	HashMap<RefCounted *, GCNode>::Iterator E = r_state.find(p_object);
	if (!E || E->value.color != GCColor::WHITE) {
		return;
	}
	E->value.color = GCColor::BLACK; // Mark first so a re-entrant visit (shared edge) is a no-op.
	for (RefCounted *next : _outgoing_edges(p_object)) {
		_collect_white(next, r_state, r_freed);
	}
	r_freed.push_back(p_object);
}

} // namespace

int CycleCollector::collect_cycles() {
	LocalVector<RefCounted *> roots;
	{
		MutexLock lock(mutex);
		roots.reserve(candidates.size());
		for (RefCounted *c : candidates) {
			c->_gc_buffered = false;
			roots.push_back(c);
		}
		candidates.clear();
	}

	GCState state;

	for (RefCounted *root : roots) {
		_mark_gray(root, state);
	}
	for (RefCounted *root : roots) {
		_scan(root, state);
	}

	LocalVector<RefCounted *> freed;
	for (RefCounted *root : roots) {
		_collect_white(root, state, freed);
	}

	// Every object in `freed` is proven reachable only from within this same garbage
	// subgraph. Freeing them isn't as simple as calling memdelete() on each in turn, though:
	// they still hold live Ref-equivalent edges *to each other* (that's the cycle), so
	// destroying one in the ordinary way would tear down its members, which would
	// unreference() -- and possibly memdelete() -- another member of the same set that we
	// haven't gotten to yet, which we would then try to memdelete() again ourselves. That's a
	// double free.
	//
	// The safe sequence (mirroring how CPython's gc breaks cycles via tp_clear before
	// deallocating):
	//   1. Pad every freed object's real refcount by +1, so no legitimate internal edge can
	//      bring it to zero while we're still severing edges below.
	//   2. Sever every outgoing edge of every freed object via clear_gc_member(), which goes
	//      through the ordinary safe Variant-release path -- this correctly unreferences
	//      whatever used to be there (be it another freed object or something still alive)
	//      without ever reaching zero early, thanks to the padding.
	//   3. Remove the padding with a real unreference(). By now every edge *within* the freed
	//      set has been severed, so this is the last real reference on each of them, and it's
	//      safe to memdelete() right here: nothing else in the set (or anywhere else, per the
	//      WHITE proof) still points to it.
	for (RefCounted *obj : freed) {
		obj->reference();
	}
	for (RefCounted *obj : freed) {
		ScriptInstance *si = obj->get_script_instance();
		if (si) {
			int count = si->get_gc_member_count();
			for (int i = 0; i < count; i++) {
				si->clear_gc_member(i);
			}
		}
	}
	for (RefCounted *obj : freed) {
		if (obj->unreference()) {
			memdelete(obj);
		}
	}

	return freed.size();
}
