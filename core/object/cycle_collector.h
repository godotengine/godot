/**************************************************************************/
/*  cycle_collector.h                                                     */
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

#pragma once

#include "core/os/mutex.h"
#include "core/templates/hash_set.h"

class RefCounted;

// Detects and frees reference cycles among RefCounted objects, which would otherwise leak
// forever: RefCounted (see ref_counted.h) is backed by a bare atomic reference count with no
// cycle detection, so an object cycle (A holds a Ref to B, B holds a Ref back to A, with
// nothing external referencing either) never reaches zero on its own (GH-7038).
//
// This implements Bacon & Rajan's synchronous trial-deletion algorithm, the same one used by
// CPython's `gc` module: on every unreference() that decrements without reaching zero, the
// object is buffered as a possible cycle root. collect_cycles() then walks the buffered set,
// trial-decrementing a *local scratch* refcount (never the real atomic) along every reachable
// edge; anything left with a scratch count of zero is proven reachable only from within the
// traversed subgraph and is freed.
//
// Scope (deliberately limited, see the implementation plan this was built against):
// - Only edges reachable via a script instance's traversable members (see
//   ScriptInstance::get_gc_member_count/get_gc_member in script_instance.h) and/or Array/
//   Dictionary contents are followed. A RefCounted with no script instance (e.g. a plain
//   Resource) is not traversable and can't be the source of a detected cycle.
// - Collection is manual only: nothing calls collect_cycles() automatically. Call it
//   yourself (e.g. bound as Engine.collect_cycles()) at a point where you know candidate
//   objects aren't being concurrently mutated by another thread.
// - Must be called from the main thread.
class CycleCollector {
	static Mutex mutex;
	static HashSet<RefCounted *> candidates;

	friend class RefCounted;

	// Called from RefCounted::unreference() when a decrement doesn't reach zero and the
	// object has a script instance (and is therefore potentially traversable). Buffers the
	// object as a possible cycle root to be examined by the next collect_cycles() call.
	static void possible_root(RefCounted *p_object);

	// Called from RefCounted's destructor. Removes the object from the candidate set, since
	// it is being freed for real via the ordinary reference-counting path and must never be
	// looked at again.
	static void remove_candidate(RefCounted *p_object);

public:
	// Examines all buffered candidates and frees any that are only reachable from within a
	// reference cycle. Must be called from the main thread. Returns the number of objects
	// freed.
	static int collect_cycles();
};
