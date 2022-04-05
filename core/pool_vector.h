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

// These are used purely for debug statistics
struct MemoryPool {
#ifdef GODOT_POOL_VECTOR_REPORT_LEAKS
	static void report_alloc(void *p_alloc, int p_line);
	static void report_free(void *p_alloc);
	static void report_leaks();
#endif

#ifdef GODOT_POOL_VECTOR_USE_GLOBAL_LOCK
	static Mutex global_mutex;
#endif

	static Mutex counter_mutex;
	static uint32_t allocs_used;
	static size_t total_memory;
	static size_t max_memory;
};

template <class T>
class PoolVector {
	struct Alloc {
		uint32_t refcount = 1;
		Mutex mutex;
		LocalVector<T> vector;
	};

	Alloc *_alloc = nullptr;

	// Returns true if makes a copy.
	bool _copy_on_write() {
		lock_guard<Mutex> guard(_alloc->mutex);

		// Refcount should not be zero, otherwise it's a misuse of COW
		if (_alloc->refcount == 1) {
			return false; //nothing to do
		}
		DEV_ASSERT(_alloc->refcount);
		_alloc->refcount -= 1;

		Alloc *alloc = memnew(Alloc);
		alloc->vector = _alloc->vector;

		// we are the new owner of this new allocation copy
		_alloc = alloc;

		return true;
	}

	void _ref(Alloc *a) {
		lock_guard<Mutex> guard(a->mutex);
		a->refcount += 1;
	}

	void _unref(Alloc *a) {
		bool destroy = false;
		{
			lock_guard<Mutex> guard(a->mutex);
			a->refcount -= 1;
			if (a->refcount == 0) {
				destroy = true;
			}
		}

		if (destroy) {
			memdelete(a);
		}
	}

public:
	class Access {
		friend class PoolVector;

	protected:
		Alloc *_alloc;

		void _ref(Alloc *p_alloc) {
			_alloc = p_alloc;
			if (_alloc) {
				_alloc->mutex.lock();
			}
		}

		_FORCE_INLINE_ void _unref() {
			if (_alloc) {
				_alloc->mutex.unlock();
				_alloc = nullptr;
			}
		}

		Access(Alloc *p_alloc) {
			_ref(p_alloc);
		}

		Access() {
			_alloc = nullptr;
		}

	public:
		~Access() {
			_unref();
		}

		void release() {
			_unref();
		}
	};

	class Read : public Access {
	public:
		// This will go down if _alloc is NULL, (e.g. after calling release).
		// There is no protection (perhaps for speed?)
		// This was the same in the old PoolVector.. we could check for this.
		_FORCE_INLINE_ const T &operator[](int p_index) const { return this->_alloc->vector.ptr()[p_index]; }
		_FORCE_INLINE_ const T *ptr() const {
			return this->_alloc ? this->_alloc->vector.ptr() : nullptr;
		}

		Read(Alloc *alloc) :
				Access(alloc) {}

		Read() :
				Read(nullptr) {}
	};

	class Write : public Access {
	public:
		// This will go down if _alloc is NULL, (e.g. after calling release).
		// There is no protection (perhaps for speed?)
		// This was the same in the old PoolVector.. we could check for this.
		_FORCE_INLINE_ T &operator[](int p_index) const { return this->_alloc->vector.ptr()[p_index]; }
		_FORCE_INLINE_ T *ptr() const {
			return this->_alloc ? this->_alloc->vector.ptr() : nullptr;
		}

		Write(Alloc *alloc) :
				Access(alloc) {}

		Write() :
				Write(nullptr) {}
	};

	Read read() const {
		return Read(_alloc);
	}

	Write write() {
		_copy_on_write();
		return Write(_alloc);
	}

	PoolVector() {
		_alloc = memnew(Alloc);
	}

	void operator=(const PoolVector &p_pool_vector) {
		Alloc *cur_alloc = _alloc;
		_ref(p_pool_vector._alloc);
		_alloc = p_pool_vector._alloc;
		_unref(cur_alloc);
	}

	PoolVector(const PoolVector &p_pool_vector) :
			PoolVector() {
		*this = p_pool_vector;
	}

	~PoolVector() {
		_unref(_alloc);
	}

	int size() const {
		return _alloc->vector.size();
	}

	bool empty() const {
		return size() == 0;
	}

	void set(int p_index, const T &p_val) {
		Write w = write(); // lock
		ERR_FAIL_INDEX(p_index, size());
		w[p_index] = p_val;
	}

	inline T operator[](int p_index) const {
		CRASH_BAD_INDEX(p_index, size());
		Read r = read(); // lock
		return r[p_index];
	}

	void fill_with(const T &p_mc) {
		Write w = write(); // lock
		int c = p_mc.size();
		resize(c);
		int idx = 0;
		for (const typename T::Element *E = p_mc.front(); E; E = E->next()) {
			w[idx++] = E->get();
		}
	}

	Error resize(int p_size) {
		ERR_FAIL_COND_V_MSG(p_size < 0, ERR_INVALID_PARAMETER, "Size of PoolVector cannot be negative.");
		lock_guard<Mutex> guard(_alloc->mutex); // lock

		// no change
		if (size() == p_size) {
			return OK;
		}

		_copy_on_write(); // make it unique
		_alloc->vector.resize(p_size);
		return OK;
	}

	void push_back(const T &p_val) {
		lock_guard<Mutex> guard(_alloc->mutex); // lock
		resize(size() + 1);
		set(size() - 1, p_val);
	}

	void append(const T &p_val) {
		push_back(p_val);
	}

	void append_array(const PoolVector<T> &p_arr) {
		Write w = write(); // lock
		Read r = p_arr.read(); // acquire their lock

		ERR_FAIL_COND(p_arr._alloc == _alloc);

		int ds = p_arr.size();
		if (ds == 0) {
			return;
		}
		int bs = size();
		resize(bs + ds);
		for (int i = 0; i < ds; i++) {
			w[bs + i] = r[i];
		}
	}

	PoolVector<T> subarray(int p_from, int p_to) {
		lock_guard<Mutex> guard(_alloc->mutex); // lock
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
		lock_guard<Mutex> guard(_alloc->mutex); // lock
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
		Read r = read(); // lock
		String rs = "";
		int s = size();
		for (int i = 0; i < s; i++) {
			rs += r[i] + delimiter;
		}
		rs.erase(rs.length() - delimiter.length(), delimiter.length());
		return rs;
	}

	void invert() {
		Write w = write(); // lock
		T temp;
		int s = size();
		int half_s = s / 2;

		for (int i = 0; i < half_s; i++) {
			temp = w[i];
			w[i] = w[s - i - 1];
			w[s - i - 1] = temp;
		}
	}

	T get(int p_index) const {
		return operator[](p_index);
	}

	void remove(int p_index) {
		Write w = write(); // lock
		int s = size();
		ERR_FAIL_INDEX(p_index, s);
		for (int i = p_index; i < s - 1; i++) {
			w[i] = w[i + 1];
		};
		resize(s - 1);
	}
};

#endif // POOL_VECTOR_H
