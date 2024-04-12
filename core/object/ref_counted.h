/**************************************************************************/
/*  ref_counted.h                                                         */
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

#ifndef REF_COUNTED_H
#define REF_COUNTED_H

#include "core/object/class_db.h"
#include "core/templates/safe_refcount.h"

class RefCounted : public Object {
	GDCLASS(RefCounted, Object);
	SafeRefCount refcount;
	SafeRefCount refcount_init;

protected:
	static void _bind_methods();

public:
	_FORCE_INLINE_ bool is_referenced() const { return refcount_init.get() != 1; }
	bool init_ref();
	bool reference(); // returns false if refcount is at zero and didn't get increased
	bool unreference();
	int get_reference_count() const;

	RefCounted();
	~RefCounted() {}
};

template <typename T>
class Ref {
	T *reference = nullptr;

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
		ERR_FAIL_NULL(p_ref);

		if (p_ref->init_ref()) {
			reference = p_ref;
		}
	}

	//virtual RefCounted * get_reference() const { return reference; }
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

	_FORCE_INLINE_ T *operator*() const {
		return reference;
	}

	_FORCE_INLINE_ T *operator->() const {
		return reference;
	}

	_FORCE_INLINE_ T *ptr() const {
		return reference;
	}

	operator Variant() const {
		return Variant(reference);
	}

	void operator=(const Ref &p_from) {
		ref(p_from);
	}

	template <typename T_Other>
	void operator=(const Ref<T_Other> &p_from) {
		RefCounted *refb = const_cast<RefCounted *>(static_cast<const RefCounted *>(p_from.ptr()));
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
		Object *object = p_variant.get_validated_object();

		if (object == reference) {
			return;
		}

		unref();

		if (!object) {
			return;
		}

		T *r = Object::cast_to<T>(object);
		if (r && r->reference()) {
			reference = r;
		}
	}

	template <typename T_Other>
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
		ref(p_from);
	}

	template <typename T_Other>
	Ref(const Ref<T_Other> &p_from) {
		RefCounted *refb = const_cast<RefCounted *>(static_cast<const RefCounted *>(p_from.ptr()));
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
		if (p_reference) {
			ref_pointer(p_reference);
		}
	}

	Ref(const Variant &p_variant) {
		Object *object = p_variant.get_validated_object();

		if (!object) {
			return;
		}

		T *r = Object::cast_to<T>(object);
		if (r && r->reference()) {
			reference = r;
		}
	}

	inline bool is_valid() const { return reference != nullptr; }
	inline bool is_null() const { return reference == nullptr; }

	void unref() {
		// TODO: this should be moved to mutexes, since this engine does not really
		// do a lot of referencing on references and stuff
		// mutexes will avoid more crashes?

		if (reference && reference->unreference()) {
			memdelete(reference);
		}
		reference = nullptr;
	}

	template <typename... VarArgs>
	void instantiate(VarArgs... p_params) {
		ref(memnew(T(p_params...)));
	}

	Ref() {}

	~Ref() {
		unref();
	}
};

class WeakRef : public RefCounted {
	GDCLASS(WeakRef, RefCounted);

	ObjectID ref;

protected:
	static void _bind_methods();

public:
	Variant get_ref() const;
	void set_obj(Object *p_object);
	void set_ref(const Ref<RefCounted> &p_ref);

	WeakRef() {}
};

template <typename T>
struct PtrToArg<Ref<T>> {
	_FORCE_INLINE_ static Ref<T> convert(const void *p_ptr) {
		if (p_ptr == nullptr) {
			return Ref<T>();
		}
		// p_ptr points to a RefCounted object
		return Ref<T>(const_cast<T *>(*reinterpret_cast<T *const *>(p_ptr)));
	}

	typedef Ref<T> EncodeT;

	_FORCE_INLINE_ static void encode(Ref<T> p_val, const void *p_ptr) {
		// p_ptr points to an EncodeT object which is a Ref<T> object.
		*(const_cast<Ref<RefCounted> *>(reinterpret_cast<const Ref<RefCounted> *>(p_ptr))) = p_val;
	}
};

template <typename T>
struct PtrToArg<const Ref<T> &> {
	typedef Ref<T> EncodeT;

	_FORCE_INLINE_ static Ref<T> convert(const void *p_ptr) {
		if (p_ptr == nullptr) {
			return Ref<T>();
		}
		// p_ptr points to a RefCounted object
		return Ref<T>(*((T *const *)p_ptr));
	}
};

template <typename T>
struct GetTypeInfo<Ref<T>> {
	static const Variant::Type VARIANT_TYPE = Variant::OBJECT;
	static const GodotTypeInfo::Metadata METADATA = GodotTypeInfo::METADATA_NONE;

	static inline PropertyInfo get_class_info() {
		return PropertyInfo(Variant::OBJECT, String(), PROPERTY_HINT_RESOURCE_TYPE, T::get_class_static());
	}
};

template <typename T>
struct GetTypeInfo<const Ref<T> &> {
	static const Variant::Type VARIANT_TYPE = Variant::OBJECT;
	static const GodotTypeInfo::Metadata METADATA = GodotTypeInfo::METADATA_NONE;

	static inline PropertyInfo get_class_info() {
		return PropertyInfo(Variant::OBJECT, String(), PROPERTY_HINT_RESOURCE_TYPE, T::get_class_static());
	}
};

template <typename T>
struct VariantInternalAccessor<Ref<T>> {
	static _FORCE_INLINE_ Ref<T> get(const Variant *v) { return Ref<T>(*VariantInternal::get_object(v)); }
	static _FORCE_INLINE_ void set(Variant *v, const Ref<T> &p_ref) { VariantInternal::refcounted_object_assign(v, p_ref.ptr()); }
};

template <typename T>
struct VariantInternalAccessor<const Ref<T> &> {
	static _FORCE_INLINE_ Ref<T> get(const Variant *v) { return Ref<T>(*VariantInternal::get_object(v)); }
	static _FORCE_INLINE_ void set(Variant *v, const Ref<T> &p_ref) { VariantInternal::refcounted_object_assign(v, p_ref.ptr()); }
};

#endif // REF_COUNTED_H
