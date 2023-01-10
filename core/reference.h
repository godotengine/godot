/**************************************************************************/
/*  reference.h                                                           */
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

#ifndef REFERENCE_H
#define REFERENCE_H

#include "core/class_db.h"
#include "core/object.h"
#include "core/ref_ptr.h"
#include "core/safe_refcount.h"

class Reference : public Object {
	GDCLASS(Reference, Object);
	friend class RefBase;
	SafeRefCount refcount;
	SafeRefCount refcount_init;

protected:
	static void _bind_methods();

public:
	_FORCE_INLINE_ bool is_referenced() const { return refcount_init.get() != 1; }
	bool init_ref();
	bool reference(); // returns false if refcount is at zero and didn't get increased
	bool unreference();
	int reference_get_count() const;

	Reference();
	~Reference();
};

template <class T>
class Ref {
	T *reference;

	void ref(const Ref &p_from) {
		if (p_from.reference == reference) {
			return;
		}

		unref();

		reference = p_from.reference;
		if (reference) {
			reference->reference();
		}
	}

	void ref_pointer(T *p_ref) {
		ERR_FAIL_COND(!p_ref);

		if (p_ref->init_ref()) {
			reference = p_ref;
		}
	}

	//virtual Reference * get_reference() const { return reference; }
public:
	_FORCE_INLINE_ bool operator==(const T *p_ptr) const {
		return reference == p_ptr;
	}
	_FORCE_INLINE_ bool operator!=(const T *p_ptr) const {
		return reference != p_ptr;
	}

	_FORCE_INLINE_ bool operator<(const Ref<T> &p_r) const {
		return reference < p_r.reference;
	}
	_FORCE_INLINE_ bool operator==(const Ref<T> &p_r) const {
		return reference == p_r.reference;
	}
	_FORCE_INLINE_ bool operator!=(const Ref<T> &p_r) const {
		return reference != p_r.reference;
	}

	_FORCE_INLINE_ T *operator->() {
		return reference;
	}

	_FORCE_INLINE_ T *operator*() {
		return reference;
	}

	_FORCE_INLINE_ const T *operator->() const {
		return reference;
	}

	_FORCE_INLINE_ const T *ptr() const {
		return reference;
	}
	_FORCE_INLINE_ T *ptr() {
		return reference;
	}

	_FORCE_INLINE_ const T *operator*() const {
		return reference;
	}

	RefPtr get_ref_ptr() const {
		RefPtr refptr;
		Ref<Reference> *irr = reinterpret_cast<Ref<Reference> *>(refptr.get_data());
		*irr = *this;
		return refptr;
	};

	operator Variant() const {
		return Variant(get_ref_ptr());
	}

	void operator=(const Ref &p_from) {
		ref(p_from);
	}

	template <class T_Other>
	void operator=(const Ref<T_Other> &p_from) {
		Reference *refb = const_cast<Reference *>(static_cast<const Reference *>(p_from.ptr()));
		if (!refb) {
			unref();
			return;
		}
		Ref r;
		r.reference = Object::cast_to<T>(refb);
		ref(r);
		r.reference = nullptr;
	}

	void operator=(const RefPtr &p_refptr) {
		Ref<Reference> *irr = reinterpret_cast<Ref<Reference> *>(p_refptr.get_data());
		Reference *refb = irr->ptr();
		if (!refb) {
			unref();
			return;
		}
		Ref r;
		r.reference = Object::cast_to<T>(refb);
		ref(r);
		r.reference = nullptr;
	}

	void operator=(const Variant &p_variant) {
		RefPtr refptr = p_variant;
		Ref<Reference> *irr = reinterpret_cast<Ref<Reference> *>(refptr.get_data());
		Reference *refb = irr->ptr();
		if (!refb) {
			unref();
			return;
		}
		Ref r;
		r.reference = Object::cast_to<T>(refb);
		ref(r);
		r.reference = nullptr;
	}

	template <class T_Other>
	void reference_ptr(T_Other *p_ptr) {
		if (reference == p_ptr) {
			return;
		}
		unref();

		T *r = Object::cast_to<T>(p_ptr);
		if (r) {
			ref_pointer(r);
		}
	}

	Ref(const Ref &p_from) {
		reference = nullptr;
		ref(p_from);
	}

	template <class T_Other>
	Ref(const Ref<T_Other> &p_from) {
		reference = nullptr;
		Reference *refb = const_cast<Reference *>(static_cast<const Reference *>(p_from.ptr()));
		if (!refb) {
			unref();
			return;
		}
		Ref r;
		r.reference = Object::cast_to<T>(refb);
		ref(r);
		r.reference = nullptr;
	}

	Ref(T *p_reference) {
		reference = nullptr;
		if (p_reference) {
			ref_pointer(p_reference);
		}
	}

	Ref(const Variant &p_variant) {
		RefPtr refptr = p_variant;
		Ref<Reference> *irr = reinterpret_cast<Ref<Reference> *>(refptr.get_data());
		reference = nullptr;
		Reference *refb = irr->ptr();
		if (!refb) {
			unref();
			return;
		}
		Ref r;
		r.reference = Object::cast_to<T>(refb);
		ref(r);
		r.reference = nullptr;
	}

	Ref(const RefPtr &p_refptr) {
		Ref<Reference> *irr = reinterpret_cast<Ref<Reference> *>(p_refptr.get_data());
		reference = nullptr;
		Reference *refb = irr->ptr();
		if (!refb) {
			unref();
			return;
		}
		Ref r;
		r.reference = Object::cast_to<T>(refb);
		ref(r);
		r.reference = nullptr;
	}

	inline bool is_valid() const { return reference != nullptr; }
	inline bool is_null() const { return reference == nullptr; }

	void unref() {
		//TODO this should be moved to mutexes, since this engine does not really
		// do a lot of referencing on references and stuff
		// mutexes will avoid more crashes?

		if (reference && reference->unreference()) {
			memdelete(reference);
		}
		reference = nullptr;
	}

	void instance() {
		ref(memnew(T));
	}

	Ref() {
		reference = nullptr;
	}

	~Ref() {
		unref();
	}
};

typedef Ref<Reference> REF;

class WeakRef : public Reference {
	GDCLASS(WeakRef, Reference);

	ObjectID ref;

protected:
	static void _bind_methods();

public:
	Variant get_ref() const;
	void set_obj(Object *p_object);
	void set_ref(const REF &p_ref);

	WeakRef();
};

#ifdef PTRCALL_ENABLED

template <class T>
struct PtrToArg<Ref<T>> {
	_FORCE_INLINE_ static Ref<T> convert(const void *p_ptr) {
		return Ref<T>(const_cast<T *>(reinterpret_cast<const T *>(p_ptr)));
	}

	_FORCE_INLINE_ static void encode(Ref<T> p_val, const void *p_ptr) {
		*(Ref<Reference> *)p_ptr = p_val;
	}
};

template <class T>
struct PtrToArg<const Ref<T> &> {
	_FORCE_INLINE_ static Ref<T> convert(const void *p_ptr) {
		return Ref<T>((T *)p_ptr);
	}
};

//this is for RefPtr

template <>
struct PtrToArg<RefPtr> {
	_FORCE_INLINE_ static RefPtr convert(const void *p_ptr) {
		return Ref<Reference>(const_cast<Reference *>(reinterpret_cast<const Reference *>(p_ptr))).get_ref_ptr();
	}

	_FORCE_INLINE_ static void encode(RefPtr p_val, const void *p_ptr) {
		Ref<Reference> r = p_val;
		*(Ref<Reference> *)p_ptr = r;
	}
};

template <>
struct PtrToArg<const RefPtr &> {
	_FORCE_INLINE_ static RefPtr convert(const void *p_ptr) {
		return Ref<Reference>(const_cast<Reference *>(reinterpret_cast<const Reference *>(p_ptr))).get_ref_ptr();
	}
};

#endif // PTRCALL_ENABLED

template <class T>
struct GetTypeInfo<Ref<T>> {
	static const Variant::Type VARIANT_TYPE = Variant::OBJECT;
	static const GodotTypeInfo::Metadata METADATA = GodotTypeInfo::METADATA_NONE;

	static inline PropertyInfo get_class_info() {
		return PropertyInfo(Variant::OBJECT, String(), PROPERTY_HINT_RESOURCE_TYPE, T::get_class_static());
	}
};

template <class T>
struct GetTypeInfo<const Ref<T> &> {
	static const Variant::Type VARIANT_TYPE = Variant::OBJECT;
	static const GodotTypeInfo::Metadata METADATA = GodotTypeInfo::METADATA_NONE;

	static inline PropertyInfo get_class_info() {
		return PropertyInfo(Variant::OBJECT, String(), PROPERTY_HINT_RESOURCE_TYPE, T::get_class_static());
	}
};

#endif // REFERENCE_H
