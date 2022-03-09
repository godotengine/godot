/*************************************************************************/
/*  fixed_pool_allocator.h                                               */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef FIXED_POOL_ALLOCATOR_H
#define FIXED_POOL_ALLOCATOR_H

#include "core/os/mutex.h"

// Uncomment this to replace FixedPoolAllocator with malloc for testing
// #define GODOT_FIXED_POOL_ALLOCATOR_FALLBACK_TO_MALLOC

// Prevent the user from creating pools that aren't at least aligned reasonably
#define GODOT_INTEGER_ROUND_UP(N, S) ((((N) + (S)-1) / (S)) * (S))

// Non-relocatable, will use internal members until full, then revert to using malloc.
// You can add padding to align by setting UNIT_SIZE (when sizeof(T) is not aligned).
template <class T, uint32_t MAX_UNITS = 1024, bool THREAD_SAFE = true, bool FORCE_TRIVIAL = false, uint32_t UNIT_SIZE = GODOT_INTEGER_ROUND_UP(sizeof(T), 4), class U = uint32_t>
class FixedPoolAllocator {
// No need to store this data if we are using fallback
#ifndef GODOT_FIXED_POOL_ALLOCATOR_FALLBACK_TO_MALLOC

	// Might as well align these for greater access speed if possible,
	// there won't be many pools so padding won't waste memory.
	alignas(8) uint8_t _data[MAX_UNITS * UNIT_SIZE];
	alignas(8) U _freelist[MAX_UNITS];

	alignas(8) U _freelist_size = 0;
	U _used_size = 0;

	// address of first and last elements of the list
	uintptr_t _addr_first = 0;
	uintptr_t _addr_last = 0;

#ifdef DEV_ENABLED
	// We can optionally check for double deletes.
	// This is handy in DEV builds.
	uint8_t _bitfield_used[(MAX_UNITS / 8) + 1] = { 0 };

	bool get_used(U p_id) {
		uint32_t byte = p_id / 8;
		uint32_t bit = p_id % 8;
		return (_bitfield_used[byte] & (1 << bit)) != 0;
	}

	bool set_used(U p_id, bool p_set_or_clear) {
		uint32_t byte = p_id / 8;
		uint32_t bit = p_id % 8;
		uint8_t old = _bitfield_used[byte] & (1 << bit);
		if (p_set_or_clear) {
			_bitfield_used[byte] |= (1 << bit);
			return old == 0;
		}
		_bitfield_used[byte] &= ~(1 << bit);
		return old != 0;
	}
#endif // DEV_ENABLED

	Mutex _mutex;

	void lock() {
		if (THREAD_SAFE) {
			_mutex.lock();
		}
	}
	void unlock() {
		if (THREAD_SAFE) {
			_mutex.unlock();
		}
	}

#endif // #ifndef GODOT_FIXED_POOL_ALLOCATOR_FALLBACK_TO_MALLOC

#ifdef DEV_ENABLED
	int32_t _total_allocs = 0;
#endif

public:
	// To be explicit in a pool there is a distinction
	// between the number of elements that are currently
	// in use, and the number of elements that have been reserved.
	// Using size() would be vague.
	U used_size() const { return _used_size; }
	U reserved_size() const { return MAX_UNITS; }

#ifdef GODOT_FIXED_POOL_ALLOCATOR_FALLBACK_TO_MALLOC
	T *raw_alloc() {
#ifdef DEV_ENABLED
		_total_allocs++;
#endif
		return (T *)malloc(UNIT_SIZE);
	}

	void free(T *p_obj) {
#ifdef DEV_ENABLED
		_total_allocs--;
#endif
		if (!__has_trivial_destructor(T) && !FORCE_TRIVIAL) {
			p_obj->~T();
		}
		::free(p_obj);
	}
#else

	FixedPoolAllocator() {
		static_assert(MAX_UNITS, "Cannot be zero size");
		static_assert(UNIT_SIZE, "Units cannot be zero size");

		_freelist_size = MAX_UNITS;
		for (U n = 0; n < (U)MAX_UNITS; n++) {
			_freelist[n] = n;
		}
		_addr_first = (uintptr_t)&_data[0];
		_addr_last = (uintptr_t)&_data[(MAX_UNITS - 1) * UNIT_SIZE];
	}

	bool integrity_check() const {
		return _freelist_size == (MAX_UNITS - _used_size);
	}

#ifdef DEV_ENABLED
	T *get_if_used(U p_id) {
		ERR_FAIL_COND_V(p_id >= MAX_UNITS, nullptr);
		if (get_used(p_id)) {
			return (T *)&_data[p_id * UNIT_SIZE];
		}
		return nullptr;
	}
#endif

	T *raw_alloc() {
#ifdef DEV_ENABLED
		_total_allocs++;
#endif

		lock();

		// full already?
		// revert to dynamic allocation
		if (_used_size >= MAX_UNITS) {
			unlock();
			T *obj = (T *)malloc(UNIT_SIZE);
			return obj;
		}

		if (unlikely(!integrity_check())) {
			unlock();
			ERR_FAIL_V_MSG(nullptr, String("Failed integrity check, possible double free : ") + String(typeid(T).name()));
		}

		_used_size++;

		// pop from freelist
		int new_size = _freelist_size - 1;
		U id = _freelist[new_size];
		_freelist_size = new_size;
		unlock();

#ifdef DEV_ENABLED
		CRASH_COND_MSG(!set_used(id, true), "ID allocated twice.");
#endif
		return (T *)&_data[id * UNIT_SIZE];
	}

	void free(T *p_obj) {
#ifdef DEV_ENABLED
		_total_allocs--;
#endif

		if (!__has_trivial_destructor(T) && !FORCE_TRIVIAL) {
			p_obj->~T();
		}

		// If not within the pool, use malloc / free
		uintptr_t p = (uintptr_t)p_obj;
		if ((p < _addr_first) || (p > _addr_last)) {
			::free(p_obj);
			return;
		}

		uintptr_t which = (uintptr_t)p_obj - _addr_first;
		which /= UNIT_SIZE;

		U id = which;

		// This should only hit if within the address range but out of sync
		// with a unit. This would be a pretty big error, so worth flagging.
		DEV_ASSERT(id < MAX_UNITS);
		DEV_ASSERT((T *)&_data[id * UNIT_SIZE] == p_obj);

#ifdef DEV_ENABLED
		CRASH_COND_MSG(!set_used(id, false), "Freeing previously freed address.");
#endif

		lock();

		if (unlikely(!integrity_check())) {
			unlock();
			ERR_FAIL_MSG(String("Failed integrity check, possible double free : ") + String(typeid(T).name()));
		}

		_freelist[_freelist_size++] = id;
		_used_size--;
		unlock();

		// If this assert gets hit, the _used_size has got out of sync,
		// most likely by double deleting an item.
		DEV_ASSERT(_used_size != (U)-1);
	}

#endif // #ndef GODOT_FIXED_POOL_ALLOCATOR_FALLBACK_TO_MALLOC

	template <class... Args>
	T *alloc(const Args &&...p_args) {
		T *alloc = raw_alloc();
		if (!__has_trivial_constructor(T) && !FORCE_TRIVIAL) {
			memnew_placement(alloc, T(p_args...));
		}
		return alloc;
	}

#ifdef DEV_ENABLED
	// Only measured in DEV_ENABLED builds.
	int32_t get_total_allocs() const { return _total_allocs; }
#endif

	uint32_t get_item_size() const { return sizeof(T); }
	uint32_t get_unit_size() const { return UNIT_SIZE; }

	uint64_t estimate_memory_use() const {
		return ((uint64_t)MAX_UNITS * UNIT_SIZE);
	}
};

#endif // FIXED_POOL_ALLOCATOR_H
