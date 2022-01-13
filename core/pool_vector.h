/*************************************************************************/
/*  pool_vector.h                                                        */
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

#ifndef POOL_VECTOR_H
#define POOL_VECTOR_H

#include "core/os/memory.h"
#include "core/os/mutex.h"
#include "core/os/rw_lock.h"
#include "core/pool_allocator.h"
#include "core/safe_refcount.h"
#include "core/ustring.h"

struct MemoryPool {
	//avoid accessing these directly, must be public for template access

	static PoolAllocator *memory_pool;
	static uint8_t *pool_memory;
	static size_t *pool_size;

	struct Alloc {
		SafeRefCount refcount;
		SafeNumeric<uint32_t> lock;
		void *mem;
		PoolAllocator::ID pool_id;
		size_t size;

		Alloc *free_list;

		Alloc() :
				lock(0),
				mem(nullptr),
				pool_id(POOL_ALLOCATOR_INVALID_ID),
				size(0),
				free_list(nullptr) {
		}
	};

	static Alloc *allocs;
	static Alloc *free_list;
	static uint32_t alloc_count;
	static uint32_t allocs_used;
	static Mutex alloc_mutex;
	static size_t total_memory;
	static size_t max_memory;

	static void setup(uint32_t p_max_allocs = (1 << 16));
	static void cleanup();
};

template <class T>
class PoolVector {
	MemoryPool::Alloc *alloc;

	void _copy_on_write() {
		if (!alloc) {
			return;
		}

		//		ERR_FAIL_COND(alloc->lock>0); should not be illegal to lock this for copy on write, as it's a copy on write after all

		// Refcount should not be zero, otherwise it's a misuse of COW
		if (alloc->refcount.get() == 1) {
			return; //nothing to do
		}

		//must allocate something

		MemoryPool::alloc_mutex.lock();
		if (MemoryPool::allocs_used == MemoryPool::alloc_count) {
			MemoryPool::alloc_mutex.unlock();
			ERR_FAIL_MSG("All memory pool allocations are in use, can't COW.");
		}

		MemoryPool::Alloc *old_alloc = alloc;

		//take one from the free list
		alloc = MemoryPool::free_list;
		MemoryPool::free_list = alloc->free_list;
		//increment the used counter
		MemoryPool::allocs_used++;

		//copy the alloc data
		alloc->size = old_alloc->size;
		alloc->refcount.init();
		alloc->pool_id = POOL_ALLOCATOR_INVALID_ID;
		alloc->lock.set(0);

#ifdef DEBUG_ENABLED
		MemoryPool::total_memory += alloc->size;
		if (MemoryPool::total_memory > MemoryPool::max_memory) {
			MemoryPool::max_memory = MemoryPool::total_memory;
		}
#endif

		MemoryPool::alloc_mutex.unlock();

		if (MemoryPool::memory_pool) {
		} else {
			alloc->mem = memalloc(alloc->size);
		}

		{
			Write w;
			w._ref(alloc);
			Read r;
			r._ref(old_alloc);

			int cur_elements = alloc->size / sizeof(T);
			T *dst = (T *)w.ptr();
			const T *src = (const T *)r.ptr();
			for (int i = 0; i < cur_elements; i++) {
				memnew_placement(&dst[i], T(src[i]));
			}
		}

		if (old_alloc->refcount.unref()) {
			//this should never happen but..

#ifdef DEBUG_ENABLED
			MemoryPool::alloc_mutex.lock();
			MemoryPool::total_memory -= old_alloc->size;
			MemoryPool::alloc_mutex.unlock();
#endif

			{
				Write w;
				w._ref(old_alloc);

				int cur_elements = old_alloc->size / sizeof(T);
				T *elems = (T *)w.ptr();
				for (int i = 0; i < cur_elements; i++) {
					elems[i].~T();
				}
			}

			if (MemoryPool::memory_pool) {
				//resize memory pool
				//if none, create
				//if some resize
			} else {
				memfree(old_alloc->mem);
				old_alloc->mem = nullptr;
				old_alloc->size = 0;

				MemoryPool::alloc_mutex.lock();
				old_alloc->free_list = MemoryPool::free_list;
				MemoryPool::free_list = old_alloc;
				MemoryPool::allocs_used--;
				MemoryPool::alloc_mutex.unlock();
			}
		}
	}

	void _reference(const PoolVector &p_pool_vector) {
		if (alloc == p_pool_vector.alloc) {
			return;
		}

		_unreference();

		if (!p_pool_vector.alloc) {
			return;
		}

		if (p_pool_vector.alloc->refcount.ref()) {
			alloc = p_pool_vector.alloc;
		}
	}

	void _unreference() {
		if (!alloc) {
			return;
		}

		if (!alloc->refcount.unref()) {
			alloc = nullptr;
			return;
		}

		//must be disposed!

		{
			int cur_elements = alloc->size / sizeof(T);

			// Don't use write() here because it could otherwise provoke COW,
			// which is not desirable here because we are destroying the last reference anyways
			Write w;
			// Reference to still prevent other threads from touching the alloc
			w._ref(alloc);

			for (int i = 0; i < cur_elements; i++) {
				w[i].~T();
			}
		}

#ifdef DEBUG_ENABLED
		MemoryPool::alloc_mutex.lock();
		MemoryPool::total_memory -= alloc->size;
		MemoryPool::alloc_mutex.unlock();
#endif

		if (MemoryPool::memory_pool) {
			//resize memory pool
			//if none, create
			//if some resize
		} else {
			memfree(alloc->mem);
			alloc->mem = nullptr;
			alloc->size = 0;

			MemoryPool::alloc_mutex.lock();
			alloc->free_list = MemoryPool::free_list;
			MemoryPool::free_list = alloc;
			MemoryPool::allocs_used--;
			MemoryPool::alloc_mutex.unlock();
		}

		alloc = nullptr;
	}

public:
	class Access {
		friend class PoolVector;

	protected:
		MemoryPool::Alloc *alloc;
		T *mem;

		_FORCE_INLINE_ void _ref(MemoryPool::Alloc *p_alloc) {
			alloc = p_alloc;
			if (alloc) {
				if (alloc->lock.increment() == 1) {
					if (MemoryPool::memory_pool) {
						//lock it and get mem
					}
				}

				mem = (T *)alloc->mem;
			}
		}

		_FORCE_INLINE_ void _unref() {
			if (alloc) {
				if (alloc->lock.decrement() == 0) {
					if (MemoryPool::memory_pool) {
						//put mem back
					}
				}

				mem = nullptr;
				alloc = nullptr;
			}
		}

		Access() {
			alloc = nullptr;
			mem = nullptr;
		}

	public:
		virtual ~Access() {
			_unref();
		}

		void release() {
			_unref();
		}
	};

	class Read : public Access {
	public:
		_FORCE_INLINE_ const T &operator[](int p_index) const { return this->mem[p_index]; }
		_FORCE_INLINE_ const T *ptr() const { return this->mem; }

		void operator=(const Read &p_read) {
			if (this->alloc == p_read.alloc) {
				return;
			}
			this->_unref();
			this->_ref(p_read.alloc);
		}

		Read(const Read &p_read) {
			this->_ref(p_read.alloc);
		}

		Read() {}
	};

	class Write : public Access {
	public:
		_FORCE_INLINE_ T &operator[](int p_index) const { return this->mem[p_index]; }
		_FORCE_INLINE_ T *ptr() const { return this->mem; }

		void operator=(const Write &p_read) {
			if (this->alloc == p_read.alloc) {
				return;
			}
			this->_unref();
			this->_ref(p_read.alloc);
		}

		Write(const Write &p_read) {
			this->_ref(p_read.alloc);
		}

		Write() {}
	};

	Read read() const {
		Read r;
		if (alloc) {
			r._ref(alloc);
		}
		return r;
	}
	Write write() {
		Write w;
		if (alloc) {
			_copy_on_write(); //make sure there is only one being accessed
			w._ref(alloc);
		}
		return w;
	}

	template <class MC>
	void fill_with(const MC &p_mc) {
		int c = p_mc.size();
		resize(c);
		Write w = write();
		int idx = 0;
		for (const typename MC::Element *E = p_mc.front(); E; E = E->next()) {
			w[idx++] = E->get();
		}
	}

	void remove(int p_index) {
		int s = size();
		ERR_FAIL_INDEX(p_index, s);
		Write w = write();
		for (int i = p_index; i < s - 1; i++) {
			w[i] = w[i + 1];
		};
		w = Write();
		resize(s - 1);
	}

	inline int size() const;
	inline bool empty() const;
	T get(int p_index) const;
	void set(int p_index, const T &p_val);
	void push_back(const T &p_val);
	void append(const T &p_val) { push_back(p_val); }
	void append_array(const PoolVector<T> &p_arr) {
		int ds = p_arr.size();
		if (ds == 0) {
			return;
		}
		int bs = size();
		resize(bs + ds);
		Write w = write();
		Read r = p_arr.read();
		for (int i = 0; i < ds; i++) {
			w[bs + i] = r[i];
		}
	}

	PoolVector<T> subarray(int p_from, int p_to) {
		if (p_from < 0) {
			p_from = size() + p_from;
		}
		if (p_to < 0) {
			p_to = size() + p_to;
		}

		ERR_FAIL_INDEX_V(p_from, size(), PoolVector<T>());
		ERR_FAIL_INDEX_V(p_to, size(), PoolVector<T>());

		PoolVector<T> slice;
		int span = 1 + p_to - p_from;
		slice.resize(span);
		Read r = read();
		Write w = slice.write();
		for (int i = 0; i < span; ++i) {
			w[i] = r[p_from + i];
		}

		return slice;
	}

	Error insert(int p_pos, const T &p_val) {
		int s = size();
		ERR_FAIL_INDEX_V(p_pos, s + 1, ERR_INVALID_PARAMETER);
		resize(s + 1);
		{
			Write w = write();
			for (int i = s; i > p_pos; i--) {
				w[i] = w[i - 1];
			}
			w[p_pos] = p_val;
		}

		return OK;
	}

	String join(String delimiter) {
		String rs = "";
		int s = size();
		Read r = read();
		for (int i = 0; i < s; i++) {
			rs += r[i] + delimiter;
		}
		rs.erase(rs.length() - delimiter.length(), delimiter.length());
		return rs;
	}

	bool is_locked() const { return alloc && alloc->lock.get() > 0; }

	inline T operator[](int p_index) const;

	Error resize(int p_size);

	void invert();

	void operator=(const PoolVector &p_pool_vector) { _reference(p_pool_vector); }
	PoolVector() { alloc = nullptr; }
	PoolVector(const PoolVector &p_pool_vector) {
		alloc = nullptr;
		_reference(p_pool_vector);
	}
	~PoolVector() { _unreference(); }
};

template <class T>
int PoolVector<T>::size() const {
	return alloc ? alloc->size / sizeof(T) : 0;
}

template <class T>
bool PoolVector<T>::empty() const {
	return alloc ? alloc->size == 0 : true;
}

template <class T>
T PoolVector<T>::get(int p_index) const {
	return operator[](p_index);
}

template <class T>
void PoolVector<T>::set(int p_index, const T &p_val) {
	ERR_FAIL_INDEX(p_index, size());

	Write w = write();
	w[p_index] = p_val;
}

template <class T>
void PoolVector<T>::push_back(const T &p_val) {
	resize(size() + 1);
	set(size() - 1, p_val);
}

template <class T>
T PoolVector<T>::operator[](int p_index) const {
	CRASH_BAD_INDEX(p_index, size());

	Read r = read();
	return r[p_index];
}

template <class T>
Error PoolVector<T>::resize(int p_size) {
	ERR_FAIL_COND_V_MSG(p_size < 0, ERR_INVALID_PARAMETER, "Size of PoolVector cannot be negative.");

	if (alloc == nullptr) {
		if (p_size == 0) {
			return OK; //nothing to do here
		}

		//must allocate something
		MemoryPool::alloc_mutex.lock();
		if (MemoryPool::allocs_used == MemoryPool::alloc_count) {
			MemoryPool::alloc_mutex.unlock();
			ERR_FAIL_V_MSG(ERR_OUT_OF_MEMORY, "All memory pool allocations are in use.");
		}

		//take one from the free list
		alloc = MemoryPool::free_list;
		MemoryPool::free_list = alloc->free_list;
		//increment the used counter
		MemoryPool::allocs_used++;

		//cleanup the alloc
		alloc->size = 0;
		alloc->refcount.init();
		alloc->pool_id = POOL_ALLOCATOR_INVALID_ID;
		MemoryPool::alloc_mutex.unlock();

	} else {
		ERR_FAIL_COND_V_MSG(alloc->lock.get() > 0, ERR_LOCKED, "Can't resize PoolVector if locked."); //can't resize if locked!
	}

	size_t new_size = sizeof(T) * p_size;

	if (alloc->size == new_size) {
		return OK; //nothing to do
	}

	if (p_size == 0) {
		_unreference();
		return OK;
	}

	_copy_on_write(); // make it unique

#ifdef DEBUG_ENABLED
	MemoryPool::alloc_mutex.lock();
	MemoryPool::total_memory -= alloc->size;
	MemoryPool::total_memory += new_size;
	if (MemoryPool::total_memory > MemoryPool::max_memory) {
		MemoryPool::max_memory = MemoryPool::total_memory;
	}
	MemoryPool::alloc_mutex.unlock();
#endif

	int cur_elements = alloc->size / sizeof(T);

	if (p_size > cur_elements) {
		if (MemoryPool::memory_pool) {
			//resize memory pool
			//if none, create
			//if some resize
		} else {
			if (alloc->size == 0) {
				alloc->mem = memalloc(new_size);
			} else {
				alloc->mem = memrealloc(alloc->mem, new_size);
			}
		}

		alloc->size = new_size;

		Write w = write();

		for (int i = cur_elements; i < p_size; i++) {
			memnew_placement(&w[i], T);
		}

	} else {
		{
			Write w = write();
			for (int i = p_size; i < cur_elements; i++) {
				w[i].~T();
			}
		}

		if (MemoryPool::memory_pool) {
			//resize memory pool
			//if none, create
			//if some resize
		} else {
			if (new_size == 0) {
				memfree(alloc->mem);
				alloc->mem = nullptr;
				alloc->size = 0;

				MemoryPool::alloc_mutex.lock();
				alloc->free_list = MemoryPool::free_list;
				MemoryPool::free_list = alloc;
				MemoryPool::allocs_used--;
				MemoryPool::alloc_mutex.unlock();

			} else {
				alloc->mem = memrealloc(alloc->mem, new_size);
				alloc->size = new_size;
			}
		}
	}

	return OK;
}

template <class T>
void PoolVector<T>::invert() {
	T temp;
	Write w = write();
	int s = size();
	int half_s = s / 2;

	for (int i = 0; i < half_s; i++) {
		temp = w[i];
		w[i] = w[s - i - 1];
		w[s - i - 1] = temp;
	}
}

#endif // POOL_VECTOR_H
