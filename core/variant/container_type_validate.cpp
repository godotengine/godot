/**************************************************************************/
/*  container_type_validate.cpp                                           */
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

#include "core/variant/container_type_validate.h"

#include "core/variant/type_helpers.h"

const Variant *ContainerTypeValidate::_internal_convert_variant(const Variant &p_variant, Variant &r_tmp_variant) const {
	if (Variant::can_convert_strict(p_variant.get_type(), variant_type)) {
		const Variant *converted_from = &p_variant;
		Callable::CallError call_error;
		Variant::construct(variant_type, r_tmp_variant, &converted_from, 1, call_error);

		if (call_error.error == Callable::CallError::CALL_OK) {
			return &r_tmp_variant;
		}
	}

	return nullptr;
}

bool ContainerTypeValidate::_internal_validate_object(const Variant &p_variant) const {
	ERR_FAIL_COND_V(p_variant.get_type() != Variant::OBJECT, false);

#ifdef DEBUG_ENABLED
	ObjectID object_id = p_variant;
	if (object_id == ObjectID()) {
		return true; // This is fine, it's null.
	}
	Object *object = ObjectDB::get_instance(object_id);
	if (object == nullptr) {
		return false;
	}
#else
	Object *object = p_variant;
	if (object == nullptr) {
		return true; //fine
	}
#endif
	if (class_name == StringName()) {
		return true; // All good, no class type requested.
	}

	const StringName &obj_class = object->get_class_name();
	if (obj_class != class_name && !object->is_class(class_name)) {
		return false;
	}

	if (script.is_null()) {
		return true; // All good, no script requested.
	}

	Script *other_script = object->get_script_instance() != nullptr ? object->get_script_instance()->get_script() : nullptr;

	// Check base script..
	if (other_script == nullptr) {
		return false;
	}
	if (!other_script->inherits_script(script)) {
		return false;
	}

	return true;
}

String ContainerTypeValidate::error_string(const Variant &p_variant, const char *p_slot) const {
	return vformat("Cannot convert %s of type '%s' into '%s'.", p_slot, VariantTypeHelpers::get_type_display_name(p_variant), VariantTypeHelpers::get_display_name(*this));
}
