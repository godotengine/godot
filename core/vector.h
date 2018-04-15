/*************************************************************************/
/*  vector.h                                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2018 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2018 Godot Engine contributors (cf. AUTHORS.md)    */
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
 * Vector container. Regular Vector Container. Use with care and for smaller arrays when possible. Use PoolVector for large arrays.
*/
#include "error_macros.h"
#include "os/memory.h"
#include "safe_refcount.h"
#include "sort.h"

template<typename T>
class Vector;

template<typename T>
class Slice;

template<typename T>
class SliceImpl;

template<typename T>
class Block {
    T *_ptr;

    static _FORCE_INLINE_ size_t _get_alloc_size(size_t p_elements) {
        //return nearest_power_of_2_templated(p_elements*sizeof(T)+sizeof(SafeRefCount)+sizeof(int));
        return next_power_of_2(p_elements * sizeof(T));
    }

    static _FORCE_INLINE_ bool _get_alloc_size_checked(size_t p_elements, size_t *out) {
#if defined(_add_overflow) && defined(_mul_overflow)
        size_t o;
		size_t p;
		if (_mul_overflow(p_elements, sizeof(T), &o)) return false;
		*out = next_power_of_2(o);
		if (_add_overflow(o, static_cast<size_t>(32), &p)) return false; //no longer allocated here
		return true;
#else
        // Speed is more important than correctness here, do the operations unchecked
        // and hope the best
        *out = _get_alloc_size(p_elements);
        return true;
#endif
    }

public:
    // internal helpers

    _FORCE_INLINE_ uint32_t *_get_refcount() const {

        if (!_ptr)
            return NULL;

        return reinterpret_cast<uint32_t *>(_ptr) - 2;
    }

    _FORCE_INLINE_ uint32_t *_get_size() const {

        if (!_ptr)
            return NULL;

        return reinterpret_cast<uint32_t *>(_ptr) - 1;
    }

    _FORCE_INLINE_ T *get_data() const {

        if (!_ptr)
            return NULL;
        return reinterpret_cast<T *>(_ptr);
    }

    _FORCE_INLINE_ int size() const {

        uint32_t *size = (uint32_t *)_get_size();
        if (size)
            return *size;
        else
            return 0;
    }

    Error resize(int p_size);
    bool copy_on_write(const int p_offset, int p_size);
    void unref();

    _FORCE_INLINE_ void operator=(const Block &p_from);
    _FORCE_INLINE_ Block(const Block &p_block);
    _FORCE_INLINE_ Block();
    _FORCE_INLINE_ ~Block();
};

template <class T>
Error Block<T>::resize(int p_size) {

    size_t alloc_size;
    ERR_FAIL_COND_V(!_get_alloc_size_checked(p_size, &alloc_size), ERR_OUT_OF_MEMORY);

    if (p_size > size()) {

        if (size() == 0) {
            // alloc from scratch
            uint32_t *ptr = (uint32_t *)Memory::alloc_static(alloc_size, true);
            ERR_FAIL_COND_V(!ptr, ERR_OUT_OF_MEMORY);
            *(ptr - 1) = 0; //size, currently none
            *(ptr - 2) = 1; //refcount

            _ptr = (T *)ptr;

        } else {
            void *_ptrnew = (T *)Memory::realloc_static(_ptr, alloc_size, true);
            ERR_FAIL_COND_V(!_ptrnew, ERR_OUT_OF_MEMORY);
            _ptr = (T *)(_ptrnew);
        }

        // construct the newly created elements
        T *elems = get_data();

        for (int i = *_get_size(); i < p_size; i++) {

            memnew_placement(&elems[i], T);
        }

        *_get_size() = p_size;

    } else if (p_size < size()) {

        // deinitialize no longer needed elements
        for (uint32_t i = p_size; i < *_get_size(); i++) {

            T *t = &get_data()[i];
            t->~T();
        }

        void *_ptrnew = (T *)Memory::realloc_static(_ptr, alloc_size, true);
        ERR_FAIL_COND_V(!_ptrnew, ERR_OUT_OF_MEMORY);

        _ptr = (T *)(_ptrnew);

        *_get_size() = p_size;
    }

    return OK;
}

template <class T>
bool Block<T>::copy_on_write(int p_offset, int p_size) {

    if (!_ptr)
        return false;

    uint32_t *refc = _get_refcount();

    if (*refc > 1) {
        /* in use by more than me */
        uint32_t *mem_new = (uint32_t *)Memory::alloc_static(_get_alloc_size(p_size), true);

        *(mem_new - 2) = 1; //refcount
        *(mem_new - 1) = p_size; //size

        T *_data = (T *)(mem_new);

        // initialize new elements
        for (uint32_t i = 0; i < p_size; i++) {

            memnew_placement(&_data[i], T(get_data()[p_offset + i]));
        }

        unref();
        _ptr = _data;
        return true;
    }

    return false;
}

template <class T>
void Block<T>::unref() {

    if (!_ptr)
        return;

    uint32_t *refc = _get_refcount();

    if (atomic_decrement(refc) > 0) {
        _ptr = NULL;
        return; // still in use
    }

    // clean up

    uint32_t *count = _get_size();
    T *data = (T *)(count + 1);

    for (uint32_t i = 0; i < *count; i++) {
        // call destructors
        data[i].~T();
    }

    // free mem
    Memory::free_static((uint8_t *)_ptr, true);

    _ptr = NULL;
}

template <class T>
void Block<T>::operator=(const Block &p_from) {

    if (_ptr == p_from._ptr)
        return; // self assign, do nothing.

    unref();

    if (!p_from._ptr)
        return; //nothing to do

    if (atomic_conditional_increment(p_from._get_refcount()) > 0) { // could reference
        _ptr = p_from._ptr;
        return;
    }

    // FIXME: error
}

template <class T>
Block<T>::Block(const Block &p_block) {

    _ptr = NULL;
    *this = p_block;
}

template <class T>
Block<T>::Block() {

    _ptr = NULL;
}

template <class T>
Block<T>::~Block() {

    unref();
}

// =====================================================================================================================
// An implementation agnostic vector.

template <typename T, template<typename> class Impl>
class _Vector {
protected:
	Impl<T> _impl;

public:
	_FORCE_INLINE_ int size() const {
		return _impl.size();
	}
	_FORCE_INLINE_ bool empty() const {
		return _impl.size() == 0;
	}
	Error resize(int p_size) {
		return _impl.resize(p_size);
	}

	_FORCE_INLINE_ T *ptrw() {
		if (empty()) return NULL;
		_impl.copy_on_write();
		return (T *)_impl.get_data();
	}
	_FORCE_INLINE_ const T *ptr() const {
		if (empty()) return NULL;
		return _impl.get_data();
	}

	_FORCE_INLINE_ void clear() {
		_impl.resize(0);
	}

	void erase(const T &p_val) {
		int idx = find(p_val);
		if (idx >= 0) remove(idx);
	}

	_FORCE_INLINE_ T &operator[](int p_index) {

		CRASH_BAD_INDEX(p_index, _impl.size());

		_impl.copy_on_write(); // wants to write, so copy on write.

		return _impl.get_data()[p_index];
	}

	_FORCE_INLINE_ const T &operator[](int p_index) const {

		CRASH_BAD_INDEX(p_index, size());

		// no cow needed, since it's reading
		return _impl.get_data()[p_index];
	}

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

	template <class T_val>
	int find(const T_val &p_val, int p_from = 0) const;

	bool push_back(const T &p_elem);

	void remove(int p_index);
	void invert();

	void set(int p_index, const T &p_elem);
	T get(int p_index) const;

	Error insert(int p_pos, const T &p_val);

    Slice<T> slice(int p_offset, int p_size) const;

	void operator=(const _Vector &p_from);

    template <template<typename> class Impl2>
    _Vector(const _Vector<T, Impl2> &p_from);

    _FORCE_INLINE_ _Vector() { }
};

template <typename T, template<typename> class Impl>
template <class T_val>
int _Vector<T, Impl>::find(const T_val &p_val, int p_from) const {

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

template <typename T, template<typename> class Impl>
void _Vector<T, Impl>::invert() {

	for (int i = 0; i < size() / 2; i++) {

		SWAP(operator[](i), operator[](size() - i - 1));
	}
}

template <typename T, template<typename> class Impl>
void _Vector<T, Impl>::set(int p_index, const T &p_elem) {

	operator[](p_index) = p_elem;
}

template <typename T, template<typename> class Impl>
T _Vector<T, Impl>::get(int p_index) const {

	return operator[](p_index);
}

template <typename T, template<typename> class Impl>
bool _Vector<T, Impl>::push_back(const T &p_elem) {

	Error err = resize(size() + 1);
	ERR_FAIL_COND_V(err, true)
	set(size() - 1, p_elem);

	return false;
}

template <typename T, template<typename> class Impl>
void _Vector<T, Impl>::remove(int p_index) {

	ERR_FAIL_INDEX(p_index, size());
	T *p = ptrw();
	int len = size();
	for (int i = p_index; i < len - 1; i++) {

		p[i] = p[i + 1];
	};

	resize(len - 1);
}

template <typename T, template<typename> class Impl>
Error _Vector<T, Impl>::insert(int p_pos, const T &p_val) {

	ERR_FAIL_INDEX_V(p_pos, size() + 1, ERR_INVALID_PARAMETER);
	resize(size() + 1);
	for (int i = (size() - 1); i > p_pos; i--)
		set(i, get(i - 1));
	set(p_pos, p_val);

	return OK;
}

template <typename T, template<typename> class Impl>
void _Vector<T, Impl>::operator=(const _Vector &p_from) {

	_impl = p_from._impl;
}

template <typename T, template<typename> class Impl>
template <template<typename> class Impl2>
_Vector<T, Impl>::_Vector(const _Vector<T, Impl2> &p_from) {

	_impl = p_from._impl;
}

// =====================================================================================================================

template <typename T>
class VectorImpl {

    Block<T> _block;

public:
    const Block<T> &_get_block() const {
        return _block;
    }

    _FORCE_INLINE_ T *get_data() const {
        return _block.get_data();
    }

    _FORCE_INLINE_ int size() const {
        return _block.size();
    }

    void copy_on_write();
    Error resize(int p_size);

    void slice(int p_offset, int p_size, SliceImpl<T> &r_impl) const;

    void operator=(const VectorImpl &p_from);
};

template <class T>
void VectorImpl<T>::copy_on_write() {

    _block.copy_on_write(0, size());
}

template <class T>
Error VectorImpl<T>::resize(int p_size) {

    ERR_FAIL_COND_V(p_size < 0, ERR_INVALID_PARAMETER);

    if (p_size == size())
        return OK;

    if (p_size == 0) {
        // wants to clean up
        _block.unref();
        return OK;
    }

    // possibly changing size, copy on write
    copy_on_write();

    return _block.resize(p_size);
}

template <class T>
void VectorImpl<T>::operator=(const VectorImpl &p_from) {

    _block = p_from._block;
}

// =====================================================================================================================

template <typename T>
class SliceImpl {

    Block<T> _block;
	uint32_t _offset;
	uint32_t _size;

public:
	_FORCE_INLINE_ T *get_data() const {
        T *ptr = _block.get_data();
		if (!ptr)
			return NULL;
		return ptr + _offset;
	}

	_FORCE_INLINE_ int size() const {
		return _size;
	}

    void copy_on_write();
    Error resize(int p_size);

    void slice(int p_offset, int p_size, SliceImpl<T> &r_impl) const;

    _FORCE_INLINE_ bool is_denormalized() const {
        return _offset + _size < _block.size();
    }

    void operator=(const SliceImpl &p_from);
    void operator=(const VectorImpl<T> &p_from);

	_FORCE_INLINE_ SliceImpl();
};

template <class T>
void SliceImpl<T>::copy_on_write() {

    const int keep_denormalized = is_denormalized() ? 1 : 0;
    if (_block.copy_on_write(_offset, _size + keep_denormalized)) {
        _offset = 0;
    }
    // size always stays the same
}

template <class T>
Error SliceImpl<T>::resize(int p_size) {

	ERR_FAIL_COND_V(p_size < 0, ERR_INVALID_PARAMETER);

	if (p_size == size())
		return OK;

	if (p_size == 0) {
		// wants to clean up
		_block.unref();
		_size = 0;
        _offset = 0;
		return OK;
	}

	// possibly changing size, copy on write
    copy_on_write();

    _size = p_size;
    return _block.resize(_offset + p_size);
}

template <class T>
void SliceImpl<T>::operator=(const SliceImpl &p_from) {

    _block = p_from._block;
    _offset = p_from._offset;
    _size = p_from._size;
}

template <class T>
void SliceImpl<T>::operator=(const VectorImpl<T> &p_from) {

    _block = p_from._get_block();
    _offset = 0;
    _size = p_from.size();
}

template <class T>
SliceImpl<T>::SliceImpl() {

	_size = 0;
    _offset = 0;
}

// =====================================================================================================================

template <typename T>
class Vector : public _Vector<T, VectorImpl> {
public:
    Vector() : _Vector<T, VectorImpl>() {
    }

    Vector(const Vector &p_from) : _Vector<T, VectorImpl>(p_from) {
    }
};

template <typename T>
class Slice : public _Vector<T, SliceImpl> {
protected:
    _FORCE_INLINE_ bool _is_denormalized() const {
        return _Vector<T, SliceImpl>::_impl.is_denormalized();
    }

public:
    Slice() : _Vector<T, SliceImpl>() {
    }

    Slice(const Slice &p_from) : _Vector<T, SliceImpl>(p_from) {
	}

    Slice(const Vector<T> &p_from) : _Vector<T, SliceImpl>(p_from) {
    }
};

// =====================================================================================================================

template <class T>
void VectorImpl<T>::slice(int p_offset, int p_size, SliceImpl<T> &r_impl) const {

    SliceImpl<T> temp;
    temp = *this;
    temp.slice(p_offset, p_size, r_impl);
}

template <class T>
void SliceImpl<T>::slice(int p_offset, int p_size, SliceImpl<T> &r_impl) const {

    if (p_offset < 0)
        p_offset = 0;
    if (p_offset >= size())
        p_offset = size();
    p_size = MIN(p_size, size() - p_offset);

    if (p_size == 0)
        return; //nothing to do

    r_impl._block = _block;
    r_impl._offset = _offset + p_offset;
    r_impl._size = p_size;
}

template <typename T, template<typename> class Impl>
Slice<T> _Vector<T, Impl>::slice(int p_offset, int p_size) const {

    Slice<T> v;
    _impl.slice(p_offset, p_size, v._impl);
    return v;
}

#endif
