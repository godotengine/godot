/**************************************************************************/
/*  packed_array_ref.h                                                    */
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

#ifndef PACKED_ARRAY_REF_H
#define PACKED_ARRAY_REF_H

#include "core/os/memory.h"
#include "core/templates/vector.h"

struct PackedArrayRefBase {
	SafeRefCount refcount;
	_FORCE_INLINE_ PackedArrayRefBase *reference() {
		if (refcount.ref()) {
			return this;
		} else {
			return nullptr;
		}
	}
	static _FORCE_INLINE_ PackedArrayRefBase *reference_from(PackedArrayRefBase *p_base, PackedArrayRefBase *p_from) {
		if (p_base == p_from) {
			return p_base; //same thing, do nothing
		}

		if (p_from->reference()) {
			if (p_base->refcount.unref()) {
				memdelete(p_base);
			}
			return p_from;
		} else {
			return p_base; //keep, could not reference new
		}
	}
	static _FORCE_INLINE_ void destroy(PackedArrayRefBase *p_array) {
		if (p_array->refcount.unref()) {
			memdelete(p_array);
		}
	}
	_FORCE_INLINE_ virtual ~PackedArrayRefBase() {} //needs virtual destructor, but make inline
};

template <typename T>
struct PackedArrayRef : public PackedArrayRefBase {
	Vector<T> array;
	static _FORCE_INLINE_ PackedArrayRef<T> *create() {
		return memnew(PackedArrayRef<T>);
	}
	static _FORCE_INLINE_ PackedArrayRef<T> *create(const Vector<T> &p_from) {
		return memnew(PackedArrayRef<T>(p_from));
	}

	static _FORCE_INLINE_ const Vector<T> &get_array(PackedArrayRefBase *p_base) {
		return static_cast<PackedArrayRef<T> *>(p_base)->array;
	}
	static _FORCE_INLINE_ Vector<T> *get_array_ptr(const PackedArrayRefBase *p_base) {
		return &const_cast<PackedArrayRef<T> *>(static_cast<const PackedArrayRef<T> *>(p_base))->array;
	}

	_FORCE_INLINE_ PackedArrayRef(const Vector<T> &p_from) {
		array = p_from;
		refcount.init();
	}
	_FORCE_INLINE_ PackedArrayRef(const PackedArrayRef<T> &p_from) {
		array = p_from.array;
		refcount.init();
	}
	_FORCE_INLINE_ PackedArrayRef() {
		refcount.init();
	}
};

template <typename T>
struct PackedArrayRefRAII {
	PackedArrayRef<T> *ref = nullptr;

	PackedArrayRefRAII() {
		ref = PackedArrayRef<T>::create();
	}

	PackedArrayRefRAII(PackedArrayRefRAII<T> &p_from) {
		if (p_from.ref) {
			ref = dynamic_cast<PackedArrayRef<T> *>(p_from.ref->reference());
		}
	}

	explicit PackedArrayRefRAII(const Vector<T> &p_from) {
		ref = PackedArrayRef<T>::create(p_from);
	}

	explicit PackedArrayRefRAII(PackedArrayRef<T> &p_from) {
		ref = dynamic_cast<PackedArrayRef<T> *>(p_from.reference());
	}

	explicit PackedArrayRefRAII(PackedArrayRef<T> *p_from) {
		if (p_from) {
			ref = dynamic_cast<PackedArrayRef<T> *>(p_from->reference());
		}
	}

	~PackedArrayRefRAII() {
		if (ref) {
			PackedArrayRefBase::destroy(ref);
		}
	}

	Vector<T> *operator->() {
		return &ref->get_array_ptr(*this);
	}

	Vector<T> &operator*() {
		return ref->get_array_ptr(*this);
	}
};

#endif // PACKED_ARRAY_REF_H
