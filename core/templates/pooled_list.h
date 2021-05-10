/*************************************************************************/
/*  pooled_list.h                                                        */
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

#pragma once

// Simple template to provide a pool with O(1) allocate and free.
// The freelist could alternatively be a linked list placed within the unused elements
// to use less memory, however a separate freelist is probably more cache friendly.

// NOTE : Take great care when using this with non POD types. The construction and destruction
// is done in the LocalVector, NOT as part of the pool. So requesting a new item does not guarantee
// a constructor is run, and free does not guarantee a destructor.
// You should generally handle clearing
// an item explicitly after a request, as it may contain 'leftovers'.
// This is by design for fastest use in the BVH. If you want a more general pool
// that does call constructors / destructors on request / free, this should probably be
// a separate template.

#include "core/templates/local_vector.h"

template <class T, bool force_trivial = false>
class PooledList {
	LocalVector<T, uint32_t, force_trivial> list;
	LocalVector<uint32_t, uint32_t, true> freelist;

	// not all list members are necessarily used
	int _used_size;

public:
	PooledList() {
		_used_size = 0;
	}

	int estimate_memory_use() const {
		return (list.size() * sizeof(T)) + (freelist.size() * sizeof(uint32_t));
	}

	const T &operator[](uint32_t p_index) const {
		return list[p_index];
	}
	T &operator[](uint32_t p_index) {
		return list[p_index];
	}

	int size() const { return _used_size; }

	T *request(uint32_t &r_id) {
		_used_size++;

		if (freelist.size()) {
			// pop from freelist
			int new_size = freelist.size() - 1;
			r_id = freelist[new_size];
			freelist.resize(new_size);
			return &list[r_id];
		}

		r_id = list.size();
		list.resize(r_id + 1);
		return &list[r_id];
	}
	void free(const uint32_t &p_id) {
		// should not be on free list already
		CRASH_COND(p_id >= list.size());
		freelist.push_back(p_id);
		_used_size--;
	}
};
