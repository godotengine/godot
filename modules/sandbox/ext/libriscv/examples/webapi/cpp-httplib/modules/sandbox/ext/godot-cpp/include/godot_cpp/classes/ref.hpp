/**************************************************************************/
/*  ref.hpp                                                               */
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

#pragma once

#include <godot_cpp/core/defs.hpp>

#include <godot_cpp/classes/object.hpp>
#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/core/binder_common.hpp>
#include <godot_cpp/core/memory.hpp>
#include <godot_cpp/variant/variant.hpp>

namespace godot {

// Helper class for RefCounted objects, same as Godot one.

class RefCounted;

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

public:
	static _FORCE_INLINE_ String get_class_static() {
		return T::get_class_static();
	}

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
		// Needs testing, Variant has a cast to Object * here.

		// Object *object = p_variant.get_validated_object();
		Object *object = p_variant;

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
		// Needs testing, Variant has a cast to Object * here.

		// Object *object = p_variant.get_validated_object();
		Object *object = p_variant;

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
		if (reference && reference->unreference()) {
			memdelete(reference);
		}
		reference = nullptr;
	}

	void instantiate() {
		ref(memnew(T()));
	}

	Ref() {}

	~Ref() {
		unref();
	}

	// Used exclusively in the bindings to recreate the Ref Godot encapsulates in return values,
	// without adding to the refcount.
	inline static Ref<T> _gde_internal_constructor(Object *obj) {
		Ref<T> r;
		r.reference = (T *)obj;
		return r;
	}
};

template <typename T>
struct PtrToArg<Ref<T>> {
	_FORCE_INLINE_ static Ref<T> convert(const void *p_ptr) {
		GDExtensionRefPtr ref = (GDExtensionRefPtr)p_ptr;
		if (unlikely(!p_ptr)) {
			return Ref<T>();
		}
		return Ref<T>(reinterpret_cast<T *>(godot::internal::get_object_instance_binding(godot::internal::gdextension_interface_ref_get_object(ref))));
	}

	typedef Ref<T> EncodeT;

	_FORCE_INLINE_ static void encode(Ref<T> p_val, void *p_ptr) {
		GDExtensionRefPtr ref = (GDExtensionRefPtr)p_ptr;
		ERR_FAIL_NULL(ref);

		// This code assumes that p_ptr points to an unset Ref<T> variable on the Godot side
		// so we only set it if we have an object to set.
		if (p_val.is_valid()) {
			godot::internal::gdextension_interface_ref_set_object(ref, p_val->_owner);
		}
	}
};

template <typename T>
struct PtrToArg<const Ref<T> &> {
	typedef Ref<T> EncodeT;

	_FORCE_INLINE_ static Ref<T> convert(const void *p_ptr) {
		GDExtensionRefPtr ref = const_cast<GDExtensionRefPtr>(p_ptr);
		if (unlikely(!p_ptr)) {
			return Ref<T>();
		}
		return Ref<T>(reinterpret_cast<T *>(godot::internal::get_object_instance_binding(godot::internal::gdextension_interface_ref_get_object(ref))));
	}
};

template <typename T>
struct GetTypeInfo<Ref<T>, typename EnableIf<TypeInherits<RefCounted, T>::value>::type> {
	static constexpr GDExtensionVariantType VARIANT_TYPE = GDEXTENSION_VARIANT_TYPE_OBJECT;
	static constexpr GDExtensionClassMethodArgumentMetadata METADATA = GDEXTENSION_METHOD_ARGUMENT_METADATA_NONE;

	static inline PropertyInfo get_class_info() {
		return make_property_info(Variant::Type::OBJECT, "", PROPERTY_HINT_RESOURCE_TYPE, T::get_class_static());
	}
};

template <typename T>
struct GetTypeInfo<const Ref<T> &, typename EnableIf<TypeInherits<RefCounted, T>::value>::type> {
	static constexpr GDExtensionVariantType VARIANT_TYPE = GDEXTENSION_VARIANT_TYPE_OBJECT;
	static constexpr GDExtensionClassMethodArgumentMetadata METADATA = GDEXTENSION_METHOD_ARGUMENT_METADATA_NONE;

	static inline PropertyInfo get_class_info() {
		return make_property_info(Variant::Type::OBJECT, "", PROPERTY_HINT_RESOURCE_TYPE, T::get_class_static());
	}
};

} // namespace godot
