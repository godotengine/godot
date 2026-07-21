/**************************************************************************/
/*  pool_vector.h                                                         */
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

#ifndef POOL_VECTOR_H
#define POOL_VECTOR_H

#include "core/local_vector.h"
#include "core/os/memory.h"
#include "core/safe_refcount.h"
#include "core/ustring.h"

#ifdef DEV_ENABLED
#include "core/os/thread.h"
#endif

// Kept purely for backward compatibility.
struct MemoryPool {
	static uint32_t allocs_used;
	static size_t total_memory;
	static size_t max_memory;
};

// Tracks which Shared blocks are locked by the current thread.
// Locking is now on a per thread basis, if a different thread mutates,
// it always COWs.
struct PoolVectorLockTracker {
	static const uint32_t MAX_CONCURRENT_LOCKS = 128;

	struct ThreadStorage {
		const void *ptrs[MAX_CONCURRENT_LOCKS];
		uint32_t count = 0;
	};

	_FORCE_INLINE_ static ThreadStorage &get_storage() {
		static thread_local ThreadStorage storage;
		return storage;
	}

	_FORCE_INLINE_ static void register_lock(const void *p_shared) {
		ThreadStorage &s = get_storage();

		if (unlikely(s.count >= MAX_CONCURRENT_LOCKS)) {
			CRASH_NOW_MSG("Exceeded maximum concurrent PoolVector locks on this thread!");
			return;
		}

		s.ptrs[s.count++] = p_shared;
	}

	_FORCE_INLINE_ static void unregister_lock(const void *p_shared) {
		ThreadStorage &s = get_storage();
		for (uint32_t i = 0; i < s.count; i++) {
			if (s.ptrs[i] == p_shared) {
				// Unordered remove.
				s.count--;
				if (i < s.count) {
					s.ptrs[i] = s.ptrs[s.count];
				}
				return;
			}
		}
	}

	_FORCE_INLINE_ static bool is_locally_locked(const void *p_shared) {
		const ThreadStorage &s = get_storage();
		for (uint32_t i = 0; i < s.count; i++) {
			if (s.ptrs[i] == p_shared) {
				return true;
			}
		}
		return false;
	}
};

template <class T>
class PoolVector {
	struct SharedData {
		// How many PoolVector instances share this data?
		// This now reflects ALL live owners, including Read / Writes.
		SafeRefCount refcount;
		LocalVector<T> v;

#ifdef DEV_ENABLED
		// Debug-only: which thread last mutated this data. Used to catch a
		// PoolVector (or a copy still aliasing its data) being mutated from
		// more than one thread without a clean hand-off.
		Thread::ID debug_thread_id = 0; // Unassigned.
#endif
		SharedData() {
			refcount.init();
#ifdef DEV_ENABLED
			debug_thread_id = Thread::get_caller_id();
#endif
		}
	};

	SharedData *shared_data = nullptr;

	void _copy_on_write() {
		if (!shared_data) {
			return;
		}

#ifdef DEV_ENABLED
		_debug_check_thread_safety();
#endif

		// Active shared pointer must have at least one owner.
		DEV_ASSERT(shared_data->refcount.get() > 0);

		if (shared_data->refcount.get() == 1) {
			return;
		}

		SharedData *old_shared = shared_data;
		shared_data = memnew(SharedData);

		// This pointer swap is NOT thread safe and atomic.
		// This means that two pointers (or references) to the same
		// `PoolVector` instance should not be used from different threads.
		// Always make a copy when moving to a different thread.
		shared_data->v = old_shared->v;

		// This is the atomic operation, for thread safety
		// with multiple PoolVectors sharing a `SharedData`.
		if (old_shared->refcount.unref()) {
			memdelete(old_shared);
		}
	}

#ifdef DEV_ENABLED
	// Catch the common mistake of a PoolVector instance being reused across
	// threads without transferring ownership cleanly (i.e. making a copy).
	void _debug_check_thread_safety() {
		if (!shared_data) {
			return;
		}
		Thread::ID caller = Thread::get_caller_id();
		if (shared_data->debug_thread_id == caller) {
			return;
		}
		if (shared_data->refcount.get() == 1) {
			// Sole owner: this is a valid hand-off, claim it for the new thread.
			shared_data->debug_thread_id = caller;
			return;
		}
		CRASH_NOW_MSG("PoolVector mutated from thread " + itos(caller) + ", but its data is shared and was last mutated from thread " + itos(shared_data->debug_thread_id));
	}
#endif

	// Increments the refcount of another PoolVector's SharedData.
	void _reference(const PoolVector &p_pool_vector) {
		if (shared_data == p_pool_vector.shared_data) {
			return;
		}

		_unreference();

		if (!p_pool_vector.shared_data) {
			return;
		}

		// Target must be valid.
		DEV_ASSERT(p_pool_vector.shared_data->refcount.get() > 0);

		if (p_pool_vector.shared_data->refcount.ref()) {
			shared_data = p_pool_vector.shared_data;
		}
	}

	// Decrements the refcount and deletes SharedData if it hits 0.
	void _unreference() {
		if (!shared_data) {
			return;
		}

		if (shared_data->refcount.unref()) {
			memdelete(shared_data);
		}

		shared_data = nullptr;
	}

	bool _is_locally_locked() const {
		if (!shared_data) {
			return false;
		}
		return PoolVectorLockTracker::is_locally_locked(shared_data);
	}

	_FORCE_INLINE_ static const char *_err_locked_string() { return "Cannot modify locked PoolVector."; }

public:
	// Manage locking for temporary access.
	// Accesses CO-OWN shared_data, and a lock blocks mutation .. but only on the same thread.
	// (Mutation will always result in COW for a PoolVector on another thread.)
	class Access {
		friend class PoolVector;

	protected:
		SharedData *shared = nullptr;
		T *mem = nullptr;

		_FORCE_INLINE_ void _ref(SharedData *p_shared) {
			if (!p_shared) {
				return;
			}

			// Access now also co-owns SharedData.
			if (p_shared->refcount.ref()) {
				shared = p_shared;

				// Lock prevents resizes / relocations of the buffer while the Access exists.
				PoolVectorLockTracker::register_lock(p_shared);

				if (shared->v.size() > 0) {
					mem = shared->v.ptr();

					// If size is > 0, the pointer must be valid.
					DEV_ASSERT(mem != nullptr);
				} else {
					mem = nullptr;
					DEV_ASSERT(shared->v.ptr() == nullptr || shared->v.get_capacity() > 0);
				}
			}
		}

		_FORCE_INLINE_ void _unref() {
			if (shared) {
				PoolVectorLockTracker::unregister_lock(shared);

				if (shared->refcount.unref()) {
					memdelete(shared);
				}
				shared = nullptr;
				mem = nullptr;
			}
		}

		// Prevent any shallow copy defaults here,
		// in case of future refactoring.
		Access(const Access &) = delete;
		Access &operator=(const Access &) = delete;

		Access() = default;

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

		Read &operator=(const Read &p_read) {
			if (this->shared == p_read.shared) {
				return *this;
			}
			this->_unref();
			this->_ref(p_read.shared);
			return *this;
		}

		Read(const Read &p_read) {
			this->_ref(p_read.shared);
		}

		// Move semantics.
		Read(Read &&p_read) {
			this->shared = p_read.shared;
			this->mem = p_read.mem;
			p_read.shared = nullptr;
			p_read.mem = nullptr;
		}

		Read &operator=(Read &&p_read) {
			if (this == &p_read) {
				return *this;
			}
			this->_unref();
			this->shared = p_read.shared;
			this->mem = p_read.mem;
			p_read.shared = nullptr;
			p_read.mem = nullptr;
			return *this;
		}

		Read() {}
	};

	class Write : public Access {
	public:
		_FORCE_INLINE_ T &operator[](int p_index) const { return this->mem[p_index]; }
		_FORCE_INLINE_ T *ptr() const { return this->mem; }

		Write &operator=(const Write &p_write) {
			if (this->shared == p_write.shared) {
				return *this;
			}
			this->_unref();
			this->_ref(p_write.shared);
			return *this;
		}

		Write(const Write &p_write) {
			this->_ref(p_write.shared);
		}

		// Move semantics.
		Write(Write &&p_write) {
			this->shared = p_write.shared;
			this->mem = p_write.mem;
			p_write.shared = nullptr;
			p_write.mem = nullptr;
		}

		Write &operator=(Write &&p_write) {
			if (this == &p_write) {
				return *this;
			}
			this->_unref();
			this->shared = p_write.shared;
			this->mem = p_write.mem;
			p_write.shared = nullptr;
			p_write.mem = nullptr;
			return *this;
		}

		Write() {}
	};

	// Locks the array for read access.
	Read read() const _LIFETIME_BOUND_ {
		Read r;
		if (shared_data) {
			r._ref(shared_data);
		}
		return r;
	}

	// Locks the array for write access (and COWs if shared).
	Write write() _LIFETIME_BOUND_ {
		ERR_FAIL_COND_V_MSG(_is_locally_locked(), Write(), _err_locked_string());
		Write w;
		if (shared_data) {
			_copy_on_write();
			w._ref(shared_data);
		}
		return w;
	}

	T operator[](int p_index) const {
		CRASH_BAD_INDEX(p_index, size());
		return shared_data->v[p_index];
	}

	bool is_locked() const { return _is_locally_locked(); }

	_FORCE_INLINE_ Span<T> span() const _LIFETIME_BOUND_ { return shared_data ? shared_data->v.span() : Span<T>(); }
	_FORCE_INLINE_ operator Span<T>() const _LIFETIME_BOUND_ { return span(); }

	// Basic funcs.
	int size() const { return shared_data ? (int)shared_data->v.size() : 0; }
	bool empty() const { return shared_data ? shared_data->v.empty() : true; }
	T get(int p_index) const { return operator[](p_index); }
	void set(int p_index, const T &p_val);
	void push_back(const T &p_val);
	void remove(int p_index);
	bool has(const T &p_val) const {
		return find(p_val) != -1;
	}

	// Utility funcs.
	int find(const T &p_val, int p_from = 0) const;
	int rfind(const T &p_val, int p_from = -1) const;
	int count(const T &p_val) const;

	template <class MC>
	void fill_with(const MC &p_mc);
	void fill(const T &p_val);
	void append(const T &p_val) { push_back(p_val); }
	void append_array(const PoolVector<T> &p_arr);
	PoolVector<T> subarray(int p_from, int p_to) const;
	Error insert(int p_pos, const T &p_val);
	String join(String delimiter) const;
	void invert();
	void sort();

	// Resizing.
	Error resize(int p_size);
	Error clear() { return resize(0); }

	// Constructors.
	PoolVector() = default;
	PoolVector(const PoolVector &p_pool_vector) {
		_reference(p_pool_vector);
	}

	explicit PoolVector(const Span<T> &p_span) {
		if (p_span.size() > 0) {
			shared_data = memnew(SharedData);
			shared_data->v = p_span;
		}
	}

	// Copy assignment.
	PoolVector &operator=(const PoolVector &p_pool_vector) {
		_reference(p_pool_vector);
		return *this;
	}

	PoolVector &operator=(const Span<T> &p_span) {
		ERR_FAIL_COND_V_MSG(_is_locally_locked(), *this, _err_locked_string());
		if (p_span.size() == 0) {
			_unreference();
			return *this;
		}
		if (!shared_data) {
			shared_data = memnew(SharedData);
		} else {
			_copy_on_write();
		}
		shared_data->v = p_span;
		return *this;
	}

	// Move semantics.
	_FORCE_INLINE_ PoolVector(PoolVector &&p_pool_vector) {
		shared_data = p_pool_vector.shared_data;
		p_pool_vector.shared_data = nullptr;
	}

	_FORCE_INLINE_ PoolVector &operator=(PoolVector &&p_pool_vector) {
		if (this == &p_pool_vector) {
			return *this;
		}
		_unreference();
		shared_data = p_pool_vector.shared_data;
		p_pool_vector.shared_data = nullptr;
		return *this;
	}

	~PoolVector() { _unreference(); }
};

template <class T>
void PoolVector<T>::remove(int p_index) {
	int s = size();
	ERR_FAIL_INDEX(p_index, s);
	ERR_FAIL_COND_MSG(_is_locally_locked(), _err_locked_string());
	_copy_on_write();
	shared_data->v.remove(p_index);

	// The local vector remove can result in zero length result,
	// which would lead to more than one NULL state for `PoolVector`.
	// This isn't a problem (and indeed would be faster in the case of a remove
	// followed by a push_back), but for consistency with the old version,
	// we will clear the `PoolVector` out completely in this state.
	if (empty()) {
		clear();
	}
}

template <class T>
template <class MC>
void PoolVector<T>::fill_with(const MC &p_mc) {
	ERR_FAIL_COND_MSG(_is_locally_locked(), _err_locked_string());
	int c = p_mc.size();
	resize(c);
	Write w = write();
	int idx = 0;
	for (const typename MC::Element *E = p_mc.front(); E; E = E->next()) {
		w[idx++] = E->get();
	}
}

template <class T>
int PoolVector<T>::count(const T &p_val) const {
	return (int)span().count(p_val);
}

template <class T>
int PoolVector<T>::find(const T &p_val, int p_from) const {
	if (p_from < 0) {
		return -1;
	}
	return (int)span().find(p_val, p_from);
}

template <class T>
int PoolVector<T>::rfind(const T &p_val, int p_from) const {
	const int s = size();
	if (s == 0) {
		return -1;
	}
	if (p_from < 0) {
		p_from = s + p_from;
	}
	if (p_from < 0 || p_from >= s) {
		p_from = s - 1;
	}
	return (int)span().rfind(p_val, p_from);
}

template <class T>
void PoolVector<T>::append_array(const PoolVector<T> &p_arr) {
	int ds = p_arr.size();
	if (ds == 0) {
		return;
	}
	// Intentional check AFTER checking for ds == 0 (i.e. no change).
	ERR_FAIL_COND_MSG(_is_locally_locked(), _err_locked_string());

	if (!shared_data) {
		shared_data = memnew(SharedData);
	} else {
		_copy_on_write();
	}

	if (p_arr.shared_data) {
		int current_size = shared_data->v.size();

		if (p_arr.shared_data == shared_data) {
			LocalVector<T> temp_copy = p_arr.shared_data->v;
			shared_data->v.resize(current_size + ds);

			for (int i = 0; i < ds; i++) {
				shared_data->v[current_size + i] = temp_copy[i];
			}
		} else {
			shared_data->v.resize(current_size + ds);

			for (int i = 0; i < ds; i++) {
				shared_data->v[current_size + i] = p_arr.shared_data->v[i];
			}
		}
	}
}

template <class T>
PoolVector<T> PoolVector<T>::subarray(int p_from, int p_to) const {
	int s = size();
	if (s == 0) {
		return PoolVector<T>();
	}
	if (p_from < 0) {
		p_from = s + p_from;
	}
	if (p_to < 0) {
		p_to = s + p_to;
	}

	ERR_FAIL_INDEX_V(p_from, s, PoolVector<T>());
	ERR_FAIL_INDEX_V(p_to, s, PoolVector<T>());
	ERR_FAIL_COND_V_MSG(p_from > p_to, PoolVector<T>(), "Invalid subarray range.");

	PoolVector<T> slice;
	if (shared_data) {
		slice.shared_data = memnew(SharedData);
		int slice_size = p_to - p_from + 1;
		slice.shared_data->v.resize(slice_size);
		for (int i = 0; i < slice_size; i++) {
			slice.shared_data->v[i] = shared_data->v[p_from + i];
		}
	}

	return slice;
}

template <class T>
String PoolVector<T>::join(String delimiter) const {
	int s = size();
	if (s == 0) {
		return String();
	}
	String rs = "";
	Read r = read();
	for (int i = 0; i < s; i++) {
		if (i > 0) {
			rs += delimiter;
		}
		rs += r[i];
	}
	return rs;
}

template <class T>
Error PoolVector<T>::insert(int p_pos, const T &p_val) {
	int s = size();
	ERR_FAIL_INDEX_V(p_pos, s + 1, ERR_INVALID_PARAMETER);
	ERR_FAIL_COND_V_MSG(_is_locally_locked(), ERR_LOCKED, _err_locked_string());
	if (!shared_data) {
		shared_data = memnew(SharedData);
	} else {
		_copy_on_write();
	}
	shared_data->v.insert(p_pos, p_val);
	return OK;
}

template <class T>
void PoolVector<T>::set(int p_index, const T &p_val) {
	CRASH_BAD_INDEX(p_index, size());
	ERR_FAIL_COND_MSG(_is_locally_locked(), _err_locked_string());
	_copy_on_write();
	shared_data->v[p_index] = p_val;
}

template <class T>
void PoolVector<T>::fill(const T &p_val) {
	if (size() > 0) {
		ERR_FAIL_COND_MSG(_is_locally_locked(), _err_locked_string());
		_copy_on_write();
		shared_data->v.fill(p_val);
	}
}

template <class T>
void PoolVector<T>::push_back(const T &p_val) {
	ERR_FAIL_COND_MSG(_is_locally_locked(), _err_locked_string());
	if (!shared_data) {
		shared_data = memnew(SharedData);
	} else {
		_copy_on_write();
	}
	shared_data->v.push_back(p_val);
}

template <class T>
Error PoolVector<T>::resize(int p_size) {
	ERR_FAIL_COND_V_MSG(p_size < 0, ERR_INVALID_PARAMETER, "Size of PoolVector cannot be negative.");

	// Not checking _is_locally_locked in this case is intentional, as there's no actual change.
	if (p_size == size()) {
		return OK;
	}

	if (p_size == 0) {
		if (shared_data) {
			ERR_FAIL_COND_V_MSG(_is_locally_locked(), ERR_LOCKED, "Cannot resize locked PoolVector.");
			_unreference();
		}
		return OK;
	}

	if (shared_data == nullptr) {
		shared_data = memnew(SharedData);
	} else {
		ERR_FAIL_COND_V_MSG(_is_locally_locked(), ERR_LOCKED, "Cannot resize locked PoolVector.");
		_copy_on_write();
	}

	shared_data->v.resize(p_size);
	return OK;
}

template <class T>
void PoolVector<T>::invert() {
	if (size() > 0) {
		ERR_FAIL_COND_MSG(_is_locally_locked(), _err_locked_string());
		_copy_on_write();
		shared_data->v.invert();
	}
}

template <class T>
void PoolVector<T>::sort() {
	if (size() > 0) {
		ERR_FAIL_COND_MSG(_is_locally_locked(), _err_locked_string());
		_copy_on_write();
		shared_data->v.sort();
	}
}

#endif // POOL_VECTOR_H
