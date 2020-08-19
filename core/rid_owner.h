/*************************************************************************/
/*  rid_owner.h                                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef RID_OWNER_H
#define RID_OWNER_H

#include "core/list.h"
#include "core/oa_hash_map.h"
#include "core/os/memory.h"
#include "core/print_string.h"
#include "core/rid.h"
#include "core/safe_refcount.h"
#include "core/set.h"
#include "core/spin_lock.h"

#include <stdio.h>
#include <typeinfo>

class RID_AllocBase {
	static volatile uint64_t base_id;

protected:
	static RID _make_from_id(uint64_t p_id) {
		RID rid;
		rid._id = p_id;
		return rid;
	}

	static uint64_t _gen_id() {
		return atomic_increment(&base_id);
	}

	static RID _gen_rid() {
		return _make_from_id(_gen_id());
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

	SpinLock spin_lock;

public:
	RID make_rid(const T &p_value) {
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
				//dont initialize chunk
				validator_chunks[chunk_count][i] = 0xFFFFFFFF;
				free_list_chunks[chunk_count][i] = alloc_count + i;
			}

			max_alloc += elements_in_chunk;
		}

		uint32_t free_index = free_list_chunks[alloc_count / elements_in_chunk][alloc_count % elements_in_chunk];

		uint32_t free_chunk = free_index / elements_in_chunk;
		uint32_t free_element = free_index % elements_in_chunk;

		T *ptr = &chunks[free_chunk][free_element];
		memnew_placement(ptr, T(p_value));

		uint32_t validator = (uint32_t)(_gen_id() & 0xFFFFFFFF);
		uint64_t id = validator;
		id <<= 32;
		id |= free_index;

		validator_chunks[free_chunk][free_element] = validator;
		alloc_count++;

		if (THREAD_SAFE) {
			spin_lock.unlock();
		}

		return _make_from_id(id);
	}

	_FORCE_INLINE_ T *getornull(const RID &p_rid) {
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
		if (unlikely(validator_chunks[idx_chunk][idx_element] != validator)) {
			if (THREAD_SAFE) {
				spin_lock.unlock();
			}
			return nullptr;
		}

		T *ptr = &chunks[idx_chunk][idx_element];

		if (THREAD_SAFE) {
			spin_lock.unlock();
		}

		return ptr;
	}

	_FORCE_INLINE_ bool owns(const RID &p_rid) {
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

		bool owned = validator_chunks[idx_chunk][idx_element] == validator;

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
		if (unlikely(validator_chunks[idx_chunk][idx_element] != validator)) {
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

	_FORCE_INLINE_ T *get_ptr_by_index(uint32_t p_index) {
		ERR_FAIL_UNSIGNED_INDEX_V(p_index, alloc_count, nullptr);
		if (THREAD_SAFE) {
			spin_lock.lock();
		}
		uint64_t idx = free_list_chunks[p_index / elements_in_chunk][p_index % elements_in_chunk];
		T *ptr = &chunks[idx / elements_in_chunk][idx % elements_in_chunk];
		if (THREAD_SAFE) {
			spin_lock.unlock();
		}
		return ptr;
	}

	_FORCE_INLINE_ RID get_rid_by_index(uint32_t p_index) {
		ERR_FAIL_INDEX_V(p_index, alloc_count, RID());
		if (THREAD_SAFE) {
			spin_lock.lock();
		}
		uint64_t idx = free_list_chunks[p_index / elements_in_chunk][p_index % elements_in_chunk];
		uint64_t validator = validator_chunks[idx / elements_in_chunk][idx % elements_in_chunk];

		RID rid = _make_from_id((validator << 32) | idx);
		if (THREAD_SAFE) {
			spin_lock.unlock();
		}
		return rid;
	}

	void get_owned_list(List<RID> *p_owned) {
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

	void set_description(const char *p_descrption) {
		description = p_descrption;
	}

	RID_Alloc(uint32_t p_target_chunk_byte_size = 4096) {
		elements_in_chunk = sizeof(T) > p_target_chunk_byte_size ? 1 : (p_target_chunk_byte_size / sizeof(T));
	}

	~RID_Alloc() {
		if (alloc_count) {
			if (description) {
				print_error("ERROR: " + itos(alloc_count) + " RID allocations of type '" + description + "' were leaked at exit.");
			} else {
#ifdef NO_SAFE_CAST
				print_error("ERROR: " + itos(alloc_count) + " RID allocations of type 'unknown' were leaked at exit.");
#else
				print_error("ERROR: " + itos(alloc_count) + " RID allocations of type '" + typeid(T).name() + "' were leaked at exit.");
#endif
			}

			for (size_t i = 0; i < max_alloc; i++) {
				uint64_t validator = validator_chunks[i / elements_in_chunk][i % elements_in_chunk];
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

	_FORCE_INLINE_ T *getornull(const RID &p_rid) {
		T **ptr = alloc.getornull(p_rid);
		if (unlikely(!ptr)) {
			return nullptr;
		}
		return *ptr;
	}

	_FORCE_INLINE_ bool owns(const RID &p_rid) {
		return alloc.owns(p_rid);
	}

	_FORCE_INLINE_ void free(const RID &p_rid) {
		alloc.free(p_rid);
	}

	_FORCE_INLINE_ void get_owned_list(List<RID> *p_owned) {
		return alloc.get_owned_list(p_owned);
	}

	void set_description(const char *p_descrption) {
		alloc.set_description(p_descrption);
	}
	RID_PtrOwner(uint32_t p_target_chunk_byte_size = 4096) :
			alloc(p_target_chunk_byte_size) {}
};

template <class T, bool THREAD_SAFE = false>
class RID_Owner {
	RID_Alloc<T, THREAD_SAFE> alloc;

public:
	_FORCE_INLINE_ RID make_rid(const T &p_ptr) {
		return alloc.make_rid(p_ptr);
	}

	_FORCE_INLINE_ T *getornull(const RID &p_rid) {
		return alloc.getornull(p_rid);
	}

	_FORCE_INLINE_ bool owns(const RID &p_rid) {
		return alloc.owns(p_rid);
	}

	_FORCE_INLINE_ void free(const RID &p_rid) {
		alloc.free(p_rid);
	}

	_FORCE_INLINE_ uint32_t get_rid_count() const {
		return alloc.get_rid_count();
	}

	_FORCE_INLINE_ RID get_rid_by_index(uint32_t p_index) {
		return alloc.get_rid_by_index(p_index);
	}

	_FORCE_INLINE_ T *get_ptr_by_index(uint32_t p_index) {
		return alloc.get_ptr_by_index(p_index);
	}

	_FORCE_INLINE_ void get_owned_list(List<RID> *p_owned) {
		return alloc.get_owned_list(p_owned);
	}

	void set_description(const char *p_descrption) {
		alloc.set_description(p_descrption);
	}
	RID_Owner(uint32_t p_target_chunk_byte_size = 4096) :
			alloc(p_target_chunk_byte_size) {}
};

#endif // RID_OWNER_H
