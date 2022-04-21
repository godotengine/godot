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

#include "core/local_vector.h"
#include "core/os/memory.h"
#include "core/os/mutex.h"
#include "core/safe_refcount.h"
#include "core/ustring.h"

// Notes on use.

///////////////////////////////////////////////////////////////////////////////////////////
// Client code should strongly prefer to create either a Read or Write (depending on access),
// and call functions (push back etc) through that rather than through PoolVector directly.
// This is so that:
// 1) A single lock and COW is done, for a section of client code (the scope of the Read or Write) that uses the PoolVector.
// 2) The enclosing lock makes it impossible for race conditions to occur within the section.

// Uncomment this to turn on leak reporting.
// Only intended for internal testing, it is not very efficient..
// and once the reference counting is working correctly, it should
// just work (TM), unless you for instance create a PoolVector and forget to delete it.
// #define GODOT_POOL_VECTOR_REPORT_LEAKS

// These are used purely for debug statistics
struct MemoryPool {
#ifdef GODOT_POOL_VECTOR_REPORT_LEAKS
	static void report_alloc(void *p_alloc, int p_line);
	static void report_free(void *p_alloc);
#endif
	static void report_leaks();

	// These can alternatively be atomics, without the counter_mutex
	static uint32_t allocs_used;
	static size_t total_memory;
	static size_t max_memory;
	static Mutex counter_mutex;

	static void counter_add_alloc(size_t p_bytes, void *p_alloc, uint32_t p_line) {
#ifdef DEBUG_ENABLED
		counter_mutex.lock();
		allocs_used++;
		total_memory += p_bytes;
		if (total_memory > max_memory) {
			max_memory = total_memory;
		}
		counter_mutex.unlock();
#endif
#ifdef GODOT_POOL_VECTOR_REPORT_LEAKS
		report_alloc(p_alloc, p_line);
#endif
	}

	static void counter_resize_alloc(size_t p_change_bytes) {
#ifdef DEBUG_ENABLED
		counter_mutex.lock();
		total_memory += p_change_bytes;
		if (total_memory > max_memory) {
			max_memory = total_memory;
		}
		counter_mutex.unlock();
#endif
	}

	static void counter_remove_alloc(size_t p_bytes, void *p_alloc) {
#ifdef DEBUG_ENABLED
		counter_mutex.lock();
		allocs_used--;
		total_memory -= p_bytes;
		counter_mutex.unlock();
#endif
#ifdef GODOT_POOL_VECTOR_REPORT_LEAKS
		report_free(p_alloc);
#endif
	}
};

template <class T>
class PoolVector {
	///////////////////////////////////////////////////////////////////////////////////////////
	// The actual allocation which contains the vector
	// is separate from the PoolVector - as it may be shared between
	// multiple PoolVectors using COW.
	// It is thus reference counted and can be locked.
	struct Alloc {
		Mutex mutex; // may be 80 bytes or more
		uint32_t refcount = 1;
		LocalVector<T> vector;

		void lock() {
#ifdef DEV_ENABLED
			if (mutex.try_lock() == OK) {
				return;
			}
			// Put a breakpoint here to investigate contended situations.
			// Contention here is not necessarily a problem but there is great possibility for
			// unintended behaviour if two threads are attempting to access an Alloc at once.
			// This should preferably be prevented by e.g. COW, or sync points.
			WARN_PRINT_ONCE("Contended PoolVector lock detected (possibly benign).");
#endif
			mutex.lock();
		}

		void unlock() {
			mutex.unlock();
		}
	};
	///////////////////////////////////////////////////////////////////////////////////////////

	Alloc *_alloc = nullptr;

	// Should always have a lock guard on _alloc entering this routine
	void _copy_on_write() {
		DEV_ASSERT(_alloc);

		if (_alloc->refcount == 1) {
			return;
		}

		// Refcount should not be zero
		DEV_ASSERT(_alloc->refcount);

		Alloc *alloc = memnew(Alloc);

		// copy contents
		alloc->vector = _alloc->vector;

		MemoryPool::counter_add_alloc((alloc->vector.size() * sizeof(T)) + sizeof(Alloc), alloc, __LINE__);

		// release old alloc
		_unreference();

		// we are the new owner of this new allocation copy
		_alloc = alloc;
	}

	void _reference(const PoolVector &p_pool_vector) {
		ERR_FAIL_NULL_MSG(p_pool_vector._alloc, "Attempting to reference from an empty PoolVector");

		// p_pool_vector._alloc could still be changed between the above check and this.
		// It is not perfect.
		LockGuard<Mutex> guard(p_pool_vector._alloc->mutex);

		if (_alloc == p_pool_vector._alloc) {
			return;
		}

		_unreference();

		// This should ideally always pass, providing the mutex logic works okay.
		if (p_pool_vector._alloc->refcount) {
			p_pool_vector._alloc->refcount++;
			_alloc = p_pool_vector._alloc;
			return;
		}

		ERR_PRINT_ONCE("Referencing from PoolVector with zero refcount, recovering.");

		// Create a dummy, or else this may assert / crash later,
		// we are assuming that all PoolVectors have a non-NULL alloc,
		// for efficiency in accessors.
		Alloc *alloc = memnew(Alloc);
		MemoryPool::counter_add_alloc(sizeof(Alloc), alloc, __LINE__);

		_alloc = alloc;
	}

	void _unreference() {
		if (!_alloc) {
			return;
		}

		bool destroy = false;
		{
			LockGuard<Mutex> guard(_alloc->mutex);
			_alloc->refcount--;
			if (_alloc->refcount == 0) {
				destroy = true;
			}
		}

		// Perhaps the greatest multithread danger here of PoolVector is that something else
		// gets a lock and tries to use the Alloc just prior to the memdelete call.
		// However we cannot keep the lock as deleting a locked mutex is undefined behaviour.
		// This may not logically happen in practice, but this area could possibly be improved in future.

		// We must prevent any other threads from increasing the ref count if it is at zero,
		// (or perhaps even prevent them managing to lock the alloc to read or write)
		// as the previous lock will have been relinquished by this point.
		if (destroy) {
			size_t vec_size = _alloc->vector.size();
			memdelete(_alloc);
			MemoryPool::counter_remove_alloc((vec_size * sizeof(T)) + sizeof(Alloc), _alloc);
		}

		_alloc = nullptr;
	}

	///////////////////////////////////////////////////////////////////////////////////////////
	class Access {
		friend class PoolVector;

	protected:
		Alloc *_alloc;

		void _ref(Alloc *p_alloc) {
			_alloc = p_alloc;
			if (_alloc) {
				_alloc->lock();
			}
		}

		_FORCE_INLINE_ void _unref() {
			if (_alloc) {
				_alloc->unlock();
				_alloc = nullptr;
			}
		}

		Access() {
			_alloc = nullptr;
		}

	public:
		inline int size() const {
			DEV_ASSERT(_alloc);
			return _alloc->vector.size();
		}

		~Access() {
			_unref();
		}

		void release() {
			_unref();
		}
	};

public:
	class Read : public Access {
	public:
		// Note : this will go down if _alloc is NULL, (e.g. after calling release).
		// This is probably for speed, as the index is checked in PoolVector get and set.
		_FORCE_INLINE_ const T &operator[](int p_index) const { return this->_alloc->vector.ptr()[p_index]; }
		_FORCE_INLINE_ const T *ptr() const {
			return this->_alloc ? this->_alloc->vector.ptr() : nullptr;
		}

		inline T get(int p_index) const {
			DEV_ASSERT(this->_alloc);
			CRASH_BAD_INDEX(p_index, this->size());
			return (*this)[p_index];
		}

		String join(String delimiter) {
			DEV_ASSERT(this->_alloc);
			String rs = "";
			int s = this->size();
			for (int i = 0; i < s; i++) {
				rs += this->_alloc->vector[i] + delimiter;
			}
			rs.erase(rs.length() - delimiter.length(), delimiter.length());
			return rs;
		}

		PoolVector<T> subarray(int p_from, int p_to) {
			DEV_ASSERT(this->_alloc);
			if (p_from < 0) {
				p_from = this->size() + p_from;
			}
			if (p_to < 0) {
				p_to = this->size() + p_to;
			}

			ERR_FAIL_INDEX_V(p_from, this->size(), PoolVector<T>());
			ERR_FAIL_INDEX_V(p_to, this->size(), PoolVector<T>());

			PoolVector<T> slice;
			int span = 1 + p_to - p_from;
			slice.resize(span);
			Write w = slice.write();
			for (int i = 0; i < span; ++i) {
				w[i] = this->_alloc->vector[p_from + i];
			}

			return slice;
		}

		void operator=(const Read &p_read) {
			if (this->_alloc == p_read._alloc) {
				return;
			}
			this->_unref();
			this->_ref(p_read._alloc);
		}

		Read(const Read &p_read) {
			this->_ref(p_read._alloc);
		}

		Read() {}
	};

	// COW and locking is done by definition when getting the write,
	// so we can have cheap lockless versions of accessors here.
	// These should be used in preference to the PoolVector versions,
	// by getting a Write. This is because the write will lock once,
	// and it is also less susceptible to race conditions in between function
	// calls to the PoolVector.
	class Write : public Access {
	public:
		// Note : this will go down if _alloc is NULL, (e.g. after calling release).
		// This is probably for speed, as the index is checked in PoolVector get and set.
		_FORCE_INLINE_ T &operator[](int p_index) const { return this->_alloc->vector.ptr()[p_index]; }
		_FORCE_INLINE_ T *ptr() const {
			return this->_alloc ? this->_alloc->vector.ptr() : nullptr;
		}

		void set(int p_index, const T &p_val) {
			DEV_ASSERT(this->_alloc);
			ERR_FAIL_INDEX(p_index, this->size());
			this->_alloc->vector.ptr()[p_index] = p_val;
		}

		void remove(int p_index) {
			DEV_ASSERT(this->_alloc);
			this->_alloc->vector.remove(p_index);
		}

		void remove_unordered(int p_index) {
			DEV_ASSERT(this->_alloc);
			this->_alloc->vector.remove_unordered(p_index);
		}

		void push_back(const T &p_val) {
			DEV_ASSERT(this->_alloc);
			this->_alloc->vector.push_back(p_val);
		}

		void invert() {
			DEV_ASSERT(this->_alloc);
			this->_alloc->vector.invert();
		}

		Error insert(int p_pos, const T &p_val) {
			DEV_ASSERT(this->_alloc);
			int s = this->size();
			ERR_FAIL_INDEX_V(p_pos, s + 1, ERR_INVALID_PARAMETER);
			resize(s + 1);
			{
				for (int i = s; i > p_pos; i--) {
					this->_alloc->vector[i] = this->_alloc->vector[i - 1];
				}
				this->_alloc->vector[p_pos] = p_val;
			}

			return OK;
		}

		template <class MC>
		void fill_with(const MC &p_mc) {
			DEV_ASSERT(this->_alloc);
			int c = p_mc.size();
			resize(c);
			int idx = 0;
			for (const typename MC::Element *E = p_mc.front(); E; E = E->next()) {
				this->_alloc->vector[idx++] = E->get();
			}
		}

		void append_array(const PoolVector<T> &p_arr) {
			DEV_ASSERT(this->_alloc);
			Read r = p_arr.read();
			ERR_FAIL_COND(p_arr._alloc == this->_alloc);

			int ds = p_arr.size();
			if (ds == 0) {
				return;
			}
			int bs = this->size();
			this->resize(bs + ds);
			for (int i = 0; i < ds; i++) {
				this->_alloc->vector[bs + i] = r[i];
			}
		}

		Error resize(int p_size) {
			DEV_ASSERT(this->_alloc);

			int old_size = this->size();

			// no change
			if (old_size == p_size) {
				return OK;
			}

			ERR_FAIL_COND_V_MSG(p_size < 0, ERR_INVALID_PARAMETER, "Size of PoolVector cannot be negative.");

			MemoryPool::counter_resize_alloc((p_size - old_size) * sizeof(T));
			this->_alloc->vector.resize(p_size);
			return OK;
		}

		void operator=(const Write &p_read) {
			if (this->_alloc == p_read._alloc) {
				return;
			}
			this->_unref();
			this->_ref(p_read._alloc);
		}

		Write(const Write &p_read) {
			this->_ref(p_read._alloc);
		}

		Write() {}
	};
	///////////////////////////////////////////////////////////////////////////////////////////

	Read read() const {
		Read r;
		if (_alloc) {
			r._ref(this->_alloc);
		}
		return r;
	}

	Write write() {
		Write w;
		if (_alloc) {
			LockGuard<Mutex> guard(_alloc->mutex);
			_copy_on_write(); //make sure there is only one being accessed
			w._ref(_alloc);
		}
		return w;
	}

	template <class MC>
	void fill_with(const MC &p_mc) {
		Write w = write();
		w.fill_with(p_mc);
	}

	void remove(int p_index) {
		Write w = write();
		w.remove(p_index);
	}

	void remove_unordered(int p_index) {
		Write w = write();
		w.remove_unordered(p_index);
	}

	inline int size() const {
		Read r = read();
		return r.size();
	}

	inline bool empty() const {
		return size() == 0;
	}

	T get(int p_index) const {
		return operator[](p_index);
	}

	void set(int p_index, const T &p_val) {
		Write w = write();
		w.set(p_index, p_val);
	}

	void push_back(const T &p_val) {
		Write w = write();
		w.push_back(p_val);
	}

	void append(const T &p_val) { push_back(p_val); }

	void append_array(const PoolVector<T> &p_arr) {
		Write w = write();
		w.append_array(p_arr);
	}

	PoolVector<T> subarray(int p_from, int p_to) {
		Read r = read();
		return r.subarray(p_from, p_to);
	}

	Error insert(int p_pos, const T &p_val) {
		Write w = write();
		return w.insert(p_pos, p_val);
	}

	String join(String delimiter) {
		Read r = read();
		return r.join(delimiter);
	}

	inline T operator[](int p_index) const {
		Read r = read();
		return r.get(p_index);
	}

	Error resize(int p_size) {
		Write w = write();
		return w.resize(p_size);
	}

	void invert() {
		Write w = write();
		w.invert();
	}

	void operator=(const PoolVector &p_pool_vector) { _reference(p_pool_vector); }

	PoolVector() {
		_alloc = memnew(Alloc);
		MemoryPool::counter_add_alloc(sizeof(Alloc), _alloc, __LINE__);
	}

	PoolVector(const PoolVector &p_pool_vector) {
		_alloc = nullptr;
		_reference(p_pool_vector);
	}

	~PoolVector() {
		// This reduces the reference count but may not
		// delete the Alloc, if more than one PoolVector
		// references it.
		_unreference();
	}
};

#endif // POOL_VECTOR_H
