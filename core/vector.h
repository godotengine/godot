/*************************************************************************/
/*  vector.h                                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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
#ifndef VECTOR_H
#define VECTOR_H

/**
 * @class Vector
 * @author Juan Linietsky
 * Vector container. Regular Vector Container. Use with care and for smaller arrays when possible. Use DVector for large arrays.
*/
#include "error_macros.h"
#include "os/memory.h"
#include "safe_refcount.h"
#include "sort.h"

template <class T>
class Vector {

	mutable T *_ptr;

	// internal helpers

	_FORCE_INLINE_ SafeRefCount *_get_refcount() const {

		if (!_ptr)
			return NULL;

		return reinterpret_cast<SafeRefCount *>((uint8_t *)_ptr - sizeof(int) - sizeof(SafeRefCount));
	}

	_FORCE_INLINE_ int *_get_size() const {

		if (!_ptr)
			return NULL;
		return reinterpret_cast<int *>((uint8_t *)_ptr - sizeof(int));
	}
	_FORCE_INLINE_ T *_get_data() const {

		if (!_ptr)
			return NULL;
		return reinterpret_cast<T *>(_ptr);
	}

	_FORCE_INLINE_ size_t _get_alloc_size(size_t p_elements) const {
		//return nearest_power_of_2_templated(p_elements*sizeof(T)+sizeof(SafeRefCount)+sizeof(int));
		return next_power_of_2(p_elements * sizeof(T) + sizeof(SafeRefCount) + sizeof(int));
	}

	_FORCE_INLINE_ bool _get_alloc_size_checked(size_t p_elements, size_t *out) const {
#if defined(_add_overflow) && defined(_mul_overflow)
		size_t o;
		size_t p;
		if (_mul_overflow(p_elements, sizeof(T), &o)) return false;
		if (_add_overflow(o, sizeof(SafeRefCount) + sizeof(int), &p)) return false;
		*out = next_power_of_2(p);
		return true;
#else
		// Speed is more important than correctness here, do the operations unchecked
		// and hope the best
		*out = _get_alloc_size(p_elements);
		return true;
#endif
	}

	void _unref(void *p_data);

	void _copy_from(const Vector &p_from);
	void _copy_on_write();

public:
	_FORCE_INLINE_ T *ptr() {
		if (!_ptr) return NULL;
		_copy_on_write();
		return (T *)_get_data();
	}
	_FORCE_INLINE_ const T *ptr() const {
		if (!_ptr) return NULL;
		return _get_data();
	}

	_FORCE_INLINE_ void clear() { resize(0); }

	_FORCE_INLINE_ int size() const {
		int *size = _get_size();
		if (size)
			return *size;
		else
			return 0;
	}
	_FORCE_INLINE_ bool empty() const { return _ptr == 0; }
	Error resize(int p_size);
	bool push_back(T p_elem);

	void remove(int p_index);
	void erase(const T &p_val) {
		int idx = find(p_val);
		if (idx >= 0) remove(idx);
	};
	void invert();

	template <class T_val>
	int find(const T_val &p_val, int p_from = 0) const;

	void set(int p_index, T p_elem);
	T get(int p_index) const;

	inline T &operator[](int p_index) {

		PRAY_BAD_INDEX(p_index, size(), T);

		_copy_on_write(); // wants to write, so copy on write.

		return _get_data()[p_index];
	}

	inline const T &operator[](int p_index) const {

		PRAY_BAD_INDEX(p_index, size(), T);

		// no cow needed, since it's reading
		return _get_data()[p_index];
	}

	Error insert(int p_pos, const T &p_val);

	template <class C>
	void sort_custom() {

		int len = size();
		if (len == 0)
			return;
		T *data = &operator[](0);
		SortArray<T, C> sorter;
		sorter.sort(data, len);
	}

	void sort() {

		sort_custom<_DefaultComparator<T> >();
	}

	void ordered_insert(const T &p_val) {
		int i;
		for (i = 0; i < size(); i++) {

			if (p_val < operator[](i)) {
				break;
			};
		};
		insert(i, p_val);
	}

	void operator=(const Vector &p_from);
	Vector(const Vector &p_from);

	_FORCE_INLINE_ Vector();
	_FORCE_INLINE_ ~Vector();
};

template <class T>
void Vector<T>::_unref(void *p_data) {

	if (!p_data)
		return;

	SafeRefCount *src = reinterpret_cast<SafeRefCount *>((uint8_t *)p_data - sizeof(int) - sizeof(SafeRefCount));

	if (!src->unref())
		return; // still in use
	// clean up

	int *count = (int *)(src + 1);
	T *data = (T *)(count + 1);

	for (int i = 0; i < *count; i++) {
		// call destructors
		data[i].~T();
	}

	// free mem
	memfree((uint8_t *)p_data - sizeof(int) - sizeof(SafeRefCount));
}

template <class T>
void Vector<T>::_copy_on_write() {

	if (!_ptr)
		return;

	if (_get_refcount()->get() > 1) {
		/* in use by more than me */
		void *mem_new = memalloc(_get_alloc_size(*_get_size()));
		SafeRefCount *src_new = (SafeRefCount *)mem_new;
		src_new->init();
		int *_size = (int *)(src_new + 1);
		*_size = *_get_size();

		T *_data = (T *)(_size + 1);

		// initialize new elements
		for (int i = 0; i < *_size; i++) {

			memnew_placement(&_data[i], T(_get_data()[i]));
		}

		_unref(_ptr);
		_ptr = _data;
	}
}

template <class T>
template <class T_val>
int Vector<T>::find(const T_val &p_val, int p_from) const {

	int ret = -1;
	if (p_from < 0 || size() == 0)
		return ret;

	for (int i = p_from; i < size(); i++) {

		if (operator[](i) == p_val) {
			ret = i;
			break;
		};
	};

	return ret;
}

template <class T>
Error Vector<T>::resize(int p_size) {

	ERR_FAIL_COND_V(p_size < 0, ERR_INVALID_PARAMETER);

	if (p_size == size())
		return OK;

	if (p_size == 0) {
		// wants to clean up
		_unref(_ptr);
		_ptr = NULL;
		return OK;
	}

	// possibly changing size, copy on write
	_copy_on_write();

	size_t alloc_size;
	ERR_FAIL_COND_V(!_get_alloc_size_checked(p_size, &alloc_size), ERR_OUT_OF_MEMORY);

	if (p_size > size()) {

		if (size() == 0) {
			// alloc from scratch
			void *ptr = memalloc(alloc_size);
			ERR_FAIL_COND_V(!ptr, ERR_OUT_OF_MEMORY);
			_ptr = (T *)((uint8_t *)ptr + sizeof(int) + sizeof(SafeRefCount));
			_get_refcount()->init(); // init refcount
			*_get_size() = 0; // init size (currently, none)

		} else {
			void *_ptrnew = (T *)memrealloc((uint8_t *)_ptr - sizeof(int) - sizeof(SafeRefCount), alloc_size);
			ERR_FAIL_COND_V(!_ptrnew, ERR_OUT_OF_MEMORY);
			_ptr = (T *)((uint8_t *)_ptrnew + sizeof(int) + sizeof(SafeRefCount));
		}

		// construct the newly created elements
		T *elems = _get_data();

		for (int i = *_get_size(); i < p_size; i++) {

			memnew_placement(&elems[i], T);
		}

		*_get_size() = p_size;

	} else if (p_size < size()) {

		// deinitialize no longer needed elements
		for (int i = p_size; i < *_get_size(); i++) {

			T *t = &_get_data()[i];
			t->~T();
		}

		void *_ptrnew = (T *)memrealloc((uint8_t *)_ptr - sizeof(int) - sizeof(SafeRefCount), alloc_size);
		ERR_FAIL_COND_V(!_ptrnew, ERR_OUT_OF_MEMORY);

		_ptr = (T *)((uint8_t *)_ptrnew + sizeof(int) + sizeof(SafeRefCount));

		*_get_size() = p_size;
	}

	return OK;
}

template <class T>
void Vector<T>::invert() {

	for (int i = 0; i < size() / 2; i++) {

		SWAP(operator[](i), operator[](size() - i - 1));
	}
}

template <class T>
void Vector<T>::set(int p_index, T p_elem) {

	operator[](p_index) = p_elem;
}

template <class T>
T Vector<T>::get(int p_index) const {

	return operator[](p_index);
}

template <class T>
bool Vector<T>::push_back(T p_elem) {

	Error err = resize(size() + 1);
	ERR_FAIL_COND_V(err, true)
	set(size() - 1, p_elem);

	return false;
}

template <class T>
void Vector<T>::remove(int p_index) {

	ERR_FAIL_INDEX(p_index, size());
	T *p = ptr();
	int len = size();
	for (int i = p_index; i < len - 1; i++) {

		p[i] = p[i + 1];
	};

	resize(len - 1);
};

template <class T>
void Vector<T>::_copy_from(const Vector &p_from) {

	if (_ptr == p_from._ptr)
		return; // self assign, do nothing.

	_unref(_ptr);
	_ptr = NULL;

	if (!p_from._ptr)
		return; //nothing to do

	if (p_from._get_refcount()->ref()) // could reference
		_ptr = p_from._ptr;
}

template <class T>
void Vector<T>::operator=(const Vector &p_from) {

	_copy_from(p_from);
}

template <class T>
Error Vector<T>::insert(int p_pos, const T &p_val) {

	ERR_FAIL_INDEX_V(p_pos, size() + 1, ERR_INVALID_PARAMETER);
	resize(size() + 1);
	for (int i = (size() - 1); i > p_pos; i--)
		set(i, get(i - 1));
	set(p_pos, p_val);

	return OK;
}

template <class T>
Vector<T>::Vector(const Vector &p_from) {

	_ptr = NULL;
	_copy_from(p_from);
}

template <class T>
Vector<T>::Vector() {

	_ptr = NULL;
}

template <class T>
Vector<T>::~Vector() {

	_unref(_ptr);
}

#endif
