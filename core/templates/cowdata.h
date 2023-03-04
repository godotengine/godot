/**************************************************************************/
/*  cowdata.h                                                             */
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

#ifndef COWDATA_H
#define COWDATA_H

#include "core/error/error_macros.h"
#include "core/os/memory.h"
#include "core/templates/safe_refcount.h"

#include <limits.h>
#include <string.h>
#include <type_traits>

template <class T>
class Vector;
class String;
class Char16String;
class CharString;
template <class T, class V>
class VMap;

struct CowDataPrefix {
	SafeNumeric<uint32_t> refcount;
	int capacity;
	int size;
};
static_assert(std::is_trivially_destructible<CowDataPrefix>::value, "");

#define ERR_PROPAGATE(m_expr) \
	if (true) {               \
		Error err = m_expr;   \
		if (unlikely(err)) {  \
			return err;       \
		}                     \
	} else                    \
		((void)0)

template <class T>
class CowData {
	template <class TV>
	friend class Vector;
	friend class String;
	friend class Char16String;
	friend class CharString;
	template <class TV, class VV>
	friend class VMap;

private:
	mutable T *_ptr = nullptr;

	// The values below are modeled as functions instead of constants
	// because otherwise compilation would fail in cases where T is
	// forward-declared (and therefore of unknown size and alignment).

	// Offset in bytes from the start of the prefix to the start of the first element.
	static constexpr size_t prefix_offset() {
		return ALIGN_AT(sizeof(CowDataPrefix), alignof(T));
	}

	// Maximum number of elements, so that:
	// - Can be allocated (not bigger than half the address space).
	// - Indices stay within the range of signed integers.
	// - There's space for the allocator pre-padding and the prefix.
	static constexpr int max_size() {
		return MIN((size_t)INT_MAX, (((SIZE_MAX >> 1) - PAD_ALIGN - prefix_offset()) / sizeof(T)));
	}

	// Minimum capacity to allocate.
	static constexpr int min_capacity() {
		// TODO: Gather some data about typical usage patterns with small arrays and improve this.
		// Current guess is preallocate a 32 byte buffer, but no more than 8 elements.
		return CLAMP(32 / (int)sizeof(T), 1, 8);
	}

	_FORCE_INLINE_ CowDataPrefix *_get_prefix() const {
		if (!_ptr) {
			return nullptr;
		}

		return reinterpret_cast<CowDataPrefix *>(reinterpret_cast<char *>(_ptr) - prefix_offset());
	}

	static _FORCE_INLINE_ T *_get_data(CowDataPrefix *prefix) {
		return reinterpret_cast<T *>(reinterpret_cast<char *>(prefix) + prefix_offset());
	}

	void _unref(CowDataPrefix *p_data);
	void _ref(const CowData *p_from);
	void _ref(const CowData &p_from);
	Error _copy_from(CowDataPrefix *p_from, int p_count, int p_capacity);
	Error _realloc(int p_new_capacity);
	Error _copy_on_write();
	Error _reserve(int p_capacity);
	bool _trim_capacity(int p_size, int &r_capacity);

public:
	void operator=(const CowData<T> &p_from) { _ref(p_from); }

	_FORCE_INLINE_ T *ptrw() {
		ERR_FAIL_COND_V(_copy_on_write(), nullptr);
		return _ptr;
	}

	_FORCE_INLINE_ const T *ptr() const {
		return _ptr;
	}

	_FORCE_INLINE_ int size() const {
		CowDataPrefix *data = _get_prefix();
		if (data) {
			return data->size;
		} else {
			return 0;
		}
	}

	_FORCE_INLINE_ void clear() { resize(0); }
	_FORCE_INLINE_ bool is_empty() const { return _ptr == nullptr; }

	_FORCE_INLINE_ void set(int p_index, const T &p_elem) {
		ERR_FAIL_INDEX(p_index, size());
		ERR_FAIL_COND(_copy_on_write());
		_ptr[p_index] = p_elem;
	}

	_FORCE_INLINE_ T &get_m(int p_index) {
		CRASH_BAD_INDEX(p_index, size());
		CRASH_COND(_copy_on_write());
		return _ptr[p_index];
	}

	_FORCE_INLINE_ const T &get(int p_index) const {
		CRASH_BAD_INDEX(p_index, size());

		return _ptr[p_index];
	}

	template <bool p_ensure_zero = false>
	Error resize(int p_size);

	Error remove_at(int p_index) {
		ERR_FAIL_INDEX_V(p_index, size(), ERR_INVALID_PARAMETER);

		int len = size();

		if (len == p_index + 1) {
			// Remove the last element by simply resizing.
			return resize(len - 1);
		}

		T last = _ptr[len - 1];

		ERR_PROPAGATE(resize(len - 1));
		// After the resize (which changes the size) we will always be the only
		// reference and no more copying is required.

		T *p = _ptr;
		for (int i = p_index; i < len - 2; i++) {
			p[i] = p[i + 1];
		}
		p[len - 2] = last;

		return OK;
	}

	Error insert(int p_pos, const T &p_val) {
		ERR_FAIL_INDEX_V(p_pos, size() + 1, ERR_INVALID_PARAMETER);
		ERR_PROPAGATE(resize(size() + 1));
		// After the resize (which changes the size) we will always be the only
		// reference and no more copying is required.

		T *p = _ptr;
		for (int i = (size() - 1); i > p_pos; i--) {
			p[i] = p[i - 1];
		}
		p[p_pos] = p_val;

		return OK;
	}

	int find(const T &p_val, int p_from = 0) const;
	int rfind(const T &p_val, int p_from = -1) const;
	int count(const T &p_val) const;

	_FORCE_INLINE_ CowData() {}
	_FORCE_INLINE_ ~CowData();
	_FORCE_INLINE_ CowData(CowData<T> &p_from) { _ref(p_from); };
};

template <class T>
void CowData<T>::_unref(CowDataPrefix *p_prefix) {
	if (!p_prefix) {
		return;
	}

	// Check, if the data is still in use or if it should be cleaned up now.
	if (p_prefix->refcount.decrement() > 0) {
		return;
	}

	// Call destructors, if necessary.
	if constexpr (!std::is_trivially_destructible<T>::value) {
		int count = p_prefix->size;
		T *ptr = _get_data(p_prefix);

		for (int i = 0; i < count; ++i) {
			ptr[i].~T();
		}
	}

	// Free the underlying memory.
	Memory::free_static(p_prefix);
}

template <class T>
Error CowData<T>::_copy_on_write() {
	CowDataPrefix *prefix = _get_prefix();

	if (!prefix) {
		return OK;
	}

	if (unlikely(prefix->refcount.get() > 1)) {
		return _copy_from(prefix, prefix->size, prefix->capacity);
	}

	return OK;
}

// Allocates a new backing memory with the given capacity and size and
// initialized with the given data (if any).
template <class T>
Error CowData<T>::_copy_from(CowDataPrefix *p_from, int p_size, int p_capacity) {
	ERR_FAIL_COND_V(p_size < 0, ERR_INVALID_PARAMETER);
	ERR_FAIL_COND_V(p_capacity < 0, ERR_INVALID_PARAMETER);

	CowDataPrefix *mem_new = (CowDataPrefix *)Memory::alloc_static(size_t(p_capacity) * sizeof(T) + prefix_offset());
	ERR_FAIL_COND_V_MSG(!mem_new, ERR_OUT_OF_MEMORY, "Insufficient memory to allocate CowData");

	new (&mem_new->refcount) SafeNumeric<uint32_t>(1);
	mem_new->size = p_size;
	mem_new->capacity = p_capacity;

	T *_data = _get_data(mem_new);

	if (p_from) {
		DEV_ASSERT(p_from->size >= p_size);

		T *ptr = _get_data(p_from);

		// Copy the existing elements.
		if constexpr (std::is_trivially_copyable<T>::value) {
			memcpy(_data, ptr, p_size * sizeof(T));
		} else {
			for (int i = 0; i < p_size; i++) {
				memnew_placement(&_data[i], T(ptr[i]));
			}
		}
	}

	_unref(_get_prefix());
	_ptr = _data;
	return OK;
}

// Resize the backing allocation to the provided capacity.
template <class T>
Error CowData<T>::_realloc(int p_new_capacity) {
	ERR_FAIL_COND_V(p_new_capacity < 0, ERR_INVALID_PARAMETER);

	CowDataPrefix *prefix = _get_prefix();

	// realloc cannot be used if
	// - copying the elements require using the copy constructor
	// - no allocation exists yet (we would have to do more initializations here)
	// - the existing allocation is shared with other CowData
	if (!std::is_trivially_destructible_v<T> || !std::is_trivially_copy_constructible_v<T> || !prefix || prefix->refcount.get() > 1) {
		return _copy_from(prefix, prefix ? prefix->size : 0, p_new_capacity);
	}

	CowDataPrefix *mem_new = (CowDataPrefix *)Memory::realloc_static(prefix, size_t(p_new_capacity) * sizeof(T) + prefix_offset());
	ERR_FAIL_COND_V_MSG(!mem_new, ERR_OUT_OF_MEMORY, "Insufficient memory to allocate CowData");
	mem_new->capacity = p_new_capacity;

	_ptr = _get_data(mem_new);
	return OK;
}

// Ensures that the backing memory has at least the given capacity.
// This does not guarantee that the backing memory is exclusively owned when returning.
template <class T>
Error CowData<T>::_reserve(int p_capacity) {
	ERR_FAIL_INDEX_V(p_capacity, max_size(), ERR_INVALID_PARAMETER);

	CowDataPrefix *prefix = _get_prefix();

	if (!prefix) {
		return _copy_from(nullptr, 0, MAX(min_capacity(), p_capacity));
	}

	int capacity = prefix->capacity;

	if (capacity >= p_capacity) {
		return OK;
	}

	int new_capacity = CLAMP(prefix->capacity * 2, p_capacity, max_size());
	return _copy_from(prefix, prefix->size, new_capacity);
}

// Returns true, if the backing memory should be trimmed when reducing to the
// given size. r_capacity is set to the capacity that it should be reduced to or
// the current capacity if it should not be reduced.
template <class T>
bool CowData<T>::_trim_capacity(int p_size, int &r_capacity) {
	CowDataPrefix *prefix = _get_prefix();
	if (!prefix) {
		r_capacity = 0;
		return false;
	}

	int new_capacity = prefix->capacity;

	while (p_size < new_capacity / 4 && new_capacity > min_capacity()) {
		new_capacity /= 2;
	}

	new_capacity = MAX(new_capacity, min_capacity());

	r_capacity = new_capacity;

	return new_capacity != prefix->capacity;
}

template <class T>
template <bool p_ensure_zero>
Error CowData<T>::resize(int p_size) {
	ERR_FAIL_INDEX_V(p_size, max_size(), ERR_INVALID_PARAMETER);

	CowDataPrefix *prefix = _get_prefix();

	// Clean up the referenced memory when resizing to zero.
	if (p_size == 0) {
		_unref(prefix);
		_ptr = nullptr;
		return OK;
	}

	int current_size = prefix ? prefix->size : 0;

	if (p_size == current_size) {
		return OK;
	}

	if (p_size > current_size) {
		ERR_PROPAGATE(_reserve(p_size));
		ERR_PROPAGATE(_copy_on_write());

		// The backing memory may have been reallocated, so fetch the pointer again.
		prefix = _get_prefix();

		// Construct the newly created elements.
		if constexpr (!std::is_trivially_constructible<T>::value) {
			for (int i = prefix->size; i < p_size; i++) {
				memnew_placement(&_ptr[i], T);
			}
		} else if (p_ensure_zero) {
			memset((void *)(_ptr + current_size), 0, (p_size - current_size) * sizeof(T));
		}

		prefix->size = p_size;
		return OK;
	} else {
		DEV_ASSERT(p_size < current_size);

		int capacity = 0;

		if (_trim_capacity(p_size, capacity) || prefix->refcount.get() > 1) {
			return _copy_from(prefix, p_size, capacity);
		} else {
			// Deinitialize no longer needed elements.
			if constexpr (!std::is_trivially_destructible<T>::value) {
				for (int i = p_size; i < prefix->size; i++) {
					T *t = &_ptr[i];
					t->~T();
				}
			}
			prefix->size = p_size;

			return OK;
		}
	}
}

template <class T>
int CowData<T>::find(const T &p_val, int p_from) const {
	int ret = -1;

	if (p_from < 0 || size() == 0) {
		return ret;
	}

	for (int i = p_from; i < size(); i++) {
		if (get(i) == p_val) {
			ret = i;
			break;
		}
	}

	return ret;
}

template <class T>
int CowData<T>::rfind(const T &p_val, int p_from) const {
	const int s = size();

	if (p_from < 0) {
		p_from = s + p_from;
	}
	if (p_from < 0 || p_from >= s) {
		p_from = s - 1;
	}

	for (int i = p_from; i >= 0; i--) {
		if (get(i) == p_val) {
			return i;
		}
	}
	return -1;
}

template <class T>
int CowData<T>::count(const T &p_val) const {
	int amount = 0;
	for (int i = 0; i < size(); i++) {
		if (get(i) == p_val) {
			amount++;
		}
	}
	return amount;
}

template <class T>
void CowData<T>::_ref(const CowData *p_from) {
	_ref(*p_from);
}

template <class T>
void CowData<T>::_ref(const CowData &p_from) {
	if (_ptr == p_from._ptr) {
		return; // Self assign, do nothing.
	}

	_unref(_get_prefix());
	_ptr = nullptr;

	if (!p_from._ptr) {
		return; // Nothing to do.
	}

	CowDataPrefix *prefix = p_from._get_prefix();
	if (prefix->refcount.conditional_increment() > 0) {
		// Only update _ptr if the refcount of that object was successfully incremented.
		_ptr = _get_data(prefix);
	}
}

template <class T>
CowData<T>::~CowData() {
	_unref(_get_prefix());
}

#undef ERR_PROPAGATE

#endif // COWDATA_H
