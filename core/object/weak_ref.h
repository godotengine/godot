/**************************************************************************/
/*  weak_ref.h                                                            */
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

#include "core/object/object.h"

/**
 * Holds a reference to an Object without owning it.
 * Use `get_validated_object()` to access the object.
 *
 * For documentation, see:
 * https://docs.godotengine.org/en/latest/engine_details/architecture/object_class.html#object-ownership-and-casting
 */
template <typename T>
class WeakRef {
	ObjectID _object_id;

public:
	/// Safely get the validated object. Returns `nullptr` if the WeakRef was null, or the object was
	/// already freed.
	/// Note: This function is not free, so if you need to access the object many times, cache it in a
	///       T * temporarily. If there's any risk of the object being freed, use `get_validated_object()`
	///       again.
	_ALWAYS_INLINE_ T *get_validated_object() {
		return _object_id.is_valid() ? Object::cast_to<T>(ObjectDB::get_instance(_object_id)) : nullptr;
	}

	_ALWAYS_INLINE_ bool is_null() const { return _object_id.is_null(); }

	_ALWAYS_INLINE_ ObjectID get_object_id() const { return _object_id; }

	bool operator==(const Object *p_object) const { return p_object ? p_object->get_instance_id() == _object_id : _object_id.is_null(); }
	bool operator!=(const Object *p_object) const { return p_object ? p_object->get_instance_id() != _object_id : _object_id.is_valid(); }

	void operator=(T *p_object) {
		_object_id = p_object ? p_object->get_instance_id() : ObjectID();
	}

	WeakRef() = default;
	WeakRef(const WeakRef<T> &p_ref) = default;

	WeakRef(ObjectID p_object_id) :
			_object_id(p_object_id) {}

	WeakRef(T *p_object) :
			_object_id(p_object ? p_object->get_instance_id() : ObjectID()) {}
};

template <typename T>
bool operator==(const Object *p_object, WeakRef<T> p_ref) {
	return p_ref == p_object;
}
template <typename T>
bool operator!=(const Object *p_object, WeakRef<T> p_ref) {
	return p_ref != p_object;
}
