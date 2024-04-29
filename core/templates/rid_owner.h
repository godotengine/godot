/**************************************************************************/
/*  rid_owner.h                                                           */
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

#ifndef RID_OWNER_H
#define RID_OWNER_H

#include "core/os/memory.h"
#include "core/os/spin_lock.h"
#include "core/string/print_string.h"
#include "core/templates/hash_set.h"
#include "core/templates/list.h"
#include "core/templates/oa_hash_map.h"
#include "core/templates/rid.h"
#include "core/templates/safe_refcount.h"

#include <stdio.h>
#include <typeinfo>

class RID_AllocBase {
	static SafeNumeric<uint64_t> base_id;

protected:
	static RID _make_from_id(uint64_t p_id) {
		RID rid;
		rid._id = p_id;
		return rid;
	}

	static RID _gen_rid() {
		return _make_from_id(_gen_id());
	}

	friend struct VariantUtilityFunctions;

	static uint64_t _gen_id() {
		return base_id.increment();
	}

public:
	virtual ~RID_AllocBase() {}
};

template <class T, bool THREAD_SAFE = false>
class RID_Alloc : public RID_AllocBase {
	T **chunks = nullptr;
	uint32_t **free_list_chunks = nullptr;
	uint32_t **validator_chunks = nullptr;

	uint32_t elements_in_chunk;
	uint32_t max_alloc = 0;
	uint32_t alloc_count = 0;

	const char *description = nullptr;

	mutable SpinLock spin_lock;

	_FORCE_INLINE_ RID _allocate_rid() {
		if (THREAD_SAFE) {
			spin_lock.lock();
		}

		if (alloc_count == max_alloc) {
			//allocate a new chunk
			uint32_t chunk_count = alloc_count == 0 ? 0 : (max_alloc / elements_in_chunk);

			//grow chunks
			chunks = (T **)memrealloc(chunks, sizeof(T *) * (chunk_count + 1));
			chunks[chunk_count] = (T *)memalloc(sizeof(T) * elements_in_chunk); //but don't initialize

			//grow validators
			validator_chunks = (uint32_t **)memrealloc(validator_chunks, sizeof(uint32_t *) * (chunk_count + 1));
			validator_chunks[chunk_count] = (uint32_t *)memalloc(sizeof(uint32_t) * elements_in_chunk);
			//grow free lists
			free_list_chunks = (uint32_t **)memrealloc(free_list_chunks, sizeof(uint32_t *) * (chunk_count + 1));
			free_list_chunks[chunk_count] = (uint32_t *)memalloc(sizeof(uint32_t) * elements_in_chunk);

			//initialize
			for (uint32_t i = 0; i < elements_in_chunk; i++) {
				// Don't initialize chunk.
				validator_chunks[chunk_count][i] = 0xFFFFFFFF;
				free_list_chunks[chunk_count][i] = alloc_count + i;
			}

			max_alloc += elements_in_chunk;
		}

		uint32_t free_index = free_list_chunks[alloc_count / elements_in_chunk][alloc_count % elements_in_chunk];

		uint32_t free_chunk = free_index / elements_in_chunk;
		uint32_t free_element = free_index % elements_in_chunk;

		uint32_t validator = (uint32_t)(_gen_id() & 0x7FFFFFFF);
		CRASH_COND_MSG(validator == 0x7FFFFFFF, "Overflow in RID validator");
		uint64_t id = validator;
		id <<= 32;
		id |= free_index;

		validator_chunks[free_chunk][free_element] = validator;

		validator_chunks[free_chunk][free_element] |= 0x80000000; //mark uninitialized bit

		alloc_count++;

		if (THREAD_SAFE) {
			spin_lock.unlock();
		}

		return _make_from_id(id);
	}

public:
	RID make_rid() {
		RID rid = _allocate_rid();
		initialize_rid(rid);
		return rid;
	}
	RID make_rid(const T &p_value) {
		RID rid = _allocate_rid();
		initialize_rid(rid, p_value);
		return rid;
	}

	//allocate but don't initialize, use initialize_rid afterwards
	RID allocate_rid() {
		return _allocate_rid();
	}

	_FORCE_INLINE_ T *get_or_null(const RID &p_rid, bool p_initialize = false) {
		if (p_rid == RID()) {
			return nullptr;
		}
		if (THREAD_SAFE) {
			spin_lock.lock();
		}

		uint64_t id = p_rid.get_id();
		uint32_t idx = uint32_t(id & 0xFFFFFFFF);
		if (unlikely(idx >= max_alloc)) {
			if (THREAD_SAFE) {
				spin_lock.unlock();
			}
			return nullptr;
		}

		uint32_t idx_chunk = idx / elements_in_chunk;
		uint32_t idx_element = idx % elements_in_chunk;

		uint32_t validator = uint32_t(id >> 32);

		if (unlikely(p_initialize)) {
			if (unlikely(!(validator_chunks[idx_chunk][idx_element] & 0x80000000))) {
				if (THREAD_SAFE) {
					spin_lock.unlock();
				}
				ERR_FAIL_V_MSG(nullptr, "Initializing already initialized RID");
			}

			if (unlikely((validator_chunks[idx_chunk][idx_element] & 0x7FFFFFFF) != validator)) {
				if (THREAD_SAFE) {
					spin_lock.unlock();
				}
				ERR_FAIL_V_MSG(nullptr, "Attempting to initialize the wrong RID");
			}

			validator_chunks[idx_chunk][idx_element] &= 0x7FFFFFFF; //initialized

		} else if (unlikely(validator_chunks[idx_chunk][idx_element] != validator)) {
			if (THREAD_SAFE) {
				spin_lock.unlock();
			}
			if ((validator_chunks[idx_chunk][idx_element] & 0x80000000) && validator_chunks[idx_chunk][idx_element] != 0xFFFFFFFF) {
				ERR_FAIL_V_MSG(nullptr, "Attempting to use an uninitialized RID");
			}
			return nullptr;
		}

		T *ptr = &chunks[idx_chunk][idx_element];

		if (THREAD_SAFE) {
			spin_lock.unlock();
		}

		return ptr;
	}
	void initialize_rid(RID p_rid) {
		T *mem = get_or_null(p_rid, true);
		ERR_FAIL_NULL(mem);
		memnew_placement(mem, T);
	}
	void initialize_rid(RID p_rid, const T &p_value) {
		T *mem = get_or_null(p_rid, true);
		ERR_FAIL_NULL(mem);
		memnew_placement(mem, T(p_value));
	}

	_FORCE_INLINE_ bool owns(const RID &p_rid) const {
		if (THREAD_SAFE) {
			spin_lock.lock();
		}

		uint64_t id = p_rid.get_id();
		uint32_t idx = uint32_t(id & 0xFFFFFFFF);
		if (unlikely(idx >= max_alloc)) {
			if (THREAD_SAFE) {
				spin_lock.unlock();
			}
			return false;
		}

		uint32_t idx_chunk = idx / elements_in_chunk;
		uint32_t idx_element = idx % elements_in_chunk;

		uint32_t validator = uint32_t(id >> 32);

		bool owned = (validator != 0x7FFFFFFF) && (validator_chunks[idx_chunk][idx_element] & 0x7FFFFFFF) == validator;

		if (THREAD_SAFE) {
			spin_lock.unlock();
		}

		return owned;
	}

	_FORCE_INLINE_ void free(const RID &p_rid) {
		if (THREAD_SAFE) {
			spin_lock.lock();
		}

		uint64_t id = p_rid.get_id();
		uint32_t idx = uint32_t(id & 0xFFFFFFFF);
		if (unlikely(idx >= max_alloc)) {
			if (THREAD_SAFE) {
				spin_lock.unlock();
			}
			ERR_FAIL();
		}

		uint32_t idx_chunk = idx / elements_in_chunk;
		uint32_t idx_element = idx % elements_in_chunk;

		uint32_t validator = uint32_t(id >> 32);
		if (unlikely(validator_chunks[idx_chunk][idx_element] & 0x80000000)) {
			if (THREAD_SAFE) {
				spin_lock.unlock();
			}
			ERR_FAIL_MSG("Attempted to free an uninitialized or invalid RID");
		} else if (unlikely(validator_chunks[idx_chunk][idx_element] != validator)) {
			if (THREAD_SAFE) {
				spin_lock.unlock();
			}
			ERR_FAIL();
		}

		chunks[idx_chunk][idx_element].~T();
		validator_chunks[idx_chunk][idx_element] = 0xFFFFFFFF; // go invalid

		alloc_count--;
		free_list_chunks[alloc_count / elements_in_chunk][alloc_count % elements_in_chunk] = idx;

		if (THREAD_SAFE) {
			spin_lock.unlock();
		}
	}

	_FORCE_INLINE_ uint32_t get_rid_count() const {
		return alloc_count;
	}
	void get_owned_list(List<RID> *p_owned) const {
		if (THREAD_SAFE) {
			spin_lock.lock();
		}
		for (size_t i = 0; i < max_alloc; i++) {
			uint64_t validator = validator_chunks[i / elements_in_chunk][i % elements_in_chunk];
			if (validator != 0xFFFFFFFF) {
				p_owned->push_back(_make_from_id((validator << 32) | i));
			}
		}
		if (THREAD_SAFE) {
			spin_lock.unlock();
		}
	}

	//used for fast iteration in the elements or RIDs
	void fill_owned_buffer(RID *p_rid_buffer) const {
		if (THREAD_SAFE) {
			spin_lock.lock();
		}
		uint32_t idx = 0;
		for (size_t i = 0; i < max_alloc; i++) {
			uint64_t validator = validator_chunks[i / elements_in_chunk][i % elements_in_chunk];
			if (validator != 0xFFFFFFFF) {
				p_rid_buffer[idx] = _make_from_id((validator << 32) | i);
				idx++;
			}
		}
		if (THREAD_SAFE) {
			spin_lock.unlock();
		}
	}

	void set_description(const char *p_descrption) {
		description = p_descrption;
	}

	RID_Alloc(uint32_t p_target_chunk_byte_size = 65536) {
		elements_in_chunk = sizeof(T) > p_target_chunk_byte_size ? 1 : (p_target_chunk_byte_size / sizeof(T));
	}

	~RID_Alloc() {
		if (alloc_count) {
			print_error(vformat("ERROR: %d RID allocations of type '%s' were leaked at exit.",
					alloc_count, description ? description : typeid(T).name()));

			for (size_t i = 0; i < max_alloc; i++) {
				uint64_t validator = validator_chunks[i / elements_in_chunk][i % elements_in_chunk];
				if (validator & 0x80000000) {
					continue; //uninitialized
				}
				if (validator != 0xFFFFFFFF) {
					chunks[i / elements_in_chunk][i % elements_in_chunk].~T();
				}
			}
		}

		uint32_t chunk_count = max_alloc / elements_in_chunk;
		for (uint32_t i = 0; i < chunk_count; i++) {
			memfree(chunks[i]);
			memfree(validator_chunks[i]);
			memfree(free_list_chunks[i]);
		}

		if (chunks) {
			memfree(chunks);
			memfree(free_list_chunks);
			memfree(validator_chunks);
		}
	}
};

template <class T, bool THREAD_SAFE = false>
class RID_PtrOwner {
	RID_Alloc<T *, THREAD_SAFE> alloc;

public:
	_FORCE_INLINE_ RID make_rid(T *p_ptr) {
		return alloc.make_rid(p_ptr);
	}

	_FORCE_INLINE_ RID allocate_rid() {
		return alloc.allocate_rid();
	}

	_FORCE_INLINE_ void initialize_rid(RID p_rid, T *p_ptr) {
		alloc.initialize_rid(p_rid, p_ptr);
	}

	_FORCE_INLINE_ T *get_or_null(const RID &p_rid) {
		T **ptr = alloc.get_or_null(p_rid);
		if (unlikely(!ptr)) {
			return nullptr;
		}
		return *ptr;
	}

	_FORCE_INLINE_ void replace(const RID &p_rid, T *p_new_ptr) {
		T **ptr = alloc.get_or_null(p_rid);
		ERR_FAIL_NULL(ptr);
		*ptr = p_new_ptr;
	}

	_FORCE_INLINE_ bool owns(const RID &p_rid) const {
		return alloc.owns(p_rid);
	}

	_FORCE_INLINE_ void free(const RID &p_rid) {
		alloc.free(p_rid);
	}

	_FORCE_INLINE_ uint32_t get_rid_count() const {
		return alloc.get_rid_count();
	}

	_FORCE_INLINE_ void get_owned_list(List<RID> *p_owned) const {
		return alloc.get_owned_list(p_owned);
	}

	void fill_owned_buffer(RID *p_rid_buffer) const {
		alloc.fill_owned_buffer(p_rid_buffer);
	}

	void set_description(const char *p_descrption) {
		alloc.set_description(p_descrption);
	}

	RID_PtrOwner(uint32_t p_target_chunk_byte_size = 65536) :
			alloc(p_target_chunk_byte_size) {}
};

template <class T, bool THREAD_SAFE = false>
class RID_Owner {
	RID_Alloc<T, THREAD_SAFE> alloc;

public:
	_FORCE_INLINE_ RID make_rid() {
		return alloc.make_rid();
	}
	_FORCE_INLINE_ RID make_rid(const T &p_ptr) {
		return alloc.make_rid(p_ptr);
	}

	_FORCE_INLINE_ RID allocate_rid() {
		return alloc.allocate_rid();
	}

	_FORCE_INLINE_ void initialize_rid(RID p_rid) {
		alloc.initialize_rid(p_rid);
	}

	_FORCE_INLINE_ void initialize_rid(RID p_rid, const T &p_ptr) {
		alloc.initialize_rid(p_rid, p_ptr);
	}

	_FORCE_INLINE_ T *get_or_null(const RID &p_rid) {
		return alloc.get_or_null(p_rid);
	}

	_FORCE_INLINE_ bool owns(const RID &p_rid) const {
		return alloc.owns(p_rid);
	}

	_FORCE_INLINE_ void free(const RID &p_rid) {
		alloc.free(p_rid);
	}

	_FORCE_INLINE_ uint32_t get_rid_count() const {
		return alloc.get_rid_count();
	}

	_FORCE_INLINE_ void get_owned_list(List<RID> *p_owned) const {
		return alloc.get_owned_list(p_owned);
	}
	void fill_owned_buffer(RID *p_rid_buffer) const {
		alloc.fill_owned_buffer(p_rid_buffer);
	}

	void set_description(const char *p_descrption) {
		alloc.set_description(p_descrption);
	}
	RID_Owner(uint32_t p_target_chunk_byte_size = 65536) :
			alloc(p_target_chunk_byte_size) {}
};

#endif // RID_OWNER_H
