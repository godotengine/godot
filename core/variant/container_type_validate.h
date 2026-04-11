/**************************************************************************/
/*  container_type_validate.h                                             */
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

#include "core/object/class_db.h"
#include "core/object/script_language.h"
#include "core/templates/vector.h"
#include "core/variant/variant.h"

// Maximum allowed nesting depth for typed containers (e.g. `Array[Array[Array[int]]]` has depth 3).
// Shared by the parser, analyzer, Array/Dictionary runtime, and ContainerTypeValidate.
static constexpr int MAX_CONTAINER_NESTING_DEPTH = 8;

struct ContainerType {
	Variant::Type builtin_type = Variant::NIL;
	StringName class_name;
	Ref<Script> script;
	Vector<ContainerType> nested_types;

	bool is_nested() const {
		return !nested_types.is_empty();
	}

	int get_depth() const {
		if (nested_types.is_empty()) {
			return 0;
		}
		int max_depth = 0;
		for (const ContainerType &nested : nested_types) {
			max_depth = MAX(max_depth, nested.get_depth());
		}
		return max_depth + 1;
	}
};

struct ContainerTypeValidate {
	Variant::Type type = Variant::NIL;
	StringName class_name;
	Ref<Script> script;
	const char *where = "container";

	Vector<ContainerTypeValidate> nested_types;

	bool is_nested() const {
		return !nested_types.is_empty();
	}

	int get_depth() const {
		if (nested_types.is_empty()) {
			return 0;
		}
		int max_depth = 0;
		for (const ContainerTypeValidate &nested : nested_types) {
			max_depth = MAX(max_depth, nested.get_depth());
		}
		return max_depth + 1;
	}

private:
	_FORCE_INLINE_ bool _internal_validate(Variant &inout_variant, const char *p_operation, bool p_output_errors) const {
		if (type == Variant::NIL) {
			return true;
		}

		if (type != inout_variant.get_type()) {
			if (inout_variant.get_type() == Variant::NIL && type == Variant::OBJECT) {
				return true;
			}

			if (Variant::can_convert_strict(inout_variant.get_type(), type)) {
				Variant converted_to;
				const Variant *converted_from = &inout_variant;
				Callable::CallError call_error;
				Variant::construct(type, converted_to, &converted_from, 1, call_error);

				if (call_error.error == Callable::CallError::CALL_OK) {
					inout_variant = converted_to;
					return true;
				}
			}

			if (p_output_errors) {
				ERR_FAIL_V_MSG(false, vformat("Attempted to %s a variable of type '%s' into a %s of type '%s'.", String(p_operation), Variant::get_type_name(inout_variant.get_type()), where, Variant::get_type_name(type)));
			} else {
				return false;
			}
		}

		if (type != Variant::OBJECT) {
			return true;
		}

		return _internal_validate_object(inout_variant, p_operation, p_output_errors);
	}

	_FORCE_INLINE_ bool _internal_validate_object(const Variant &p_variant, const char *p_operation, bool p_output_errors) const {
		ERR_FAIL_COND_V(p_variant.get_type() != Variant::OBJECT, false);

#ifdef DEBUG_ENABLED
		ObjectID object_id = p_variant;
		if (object_id == ObjectID()) {
			return true; // This is fine, it's null.
		}
		Object *object = ObjectDB::get_instance(object_id);
		if (object == nullptr) {
			if (p_output_errors) {
				ERR_FAIL_V_MSG(false, vformat("Attempted to %s an invalid (previously freed?) object instance into a '%s'.", String(p_operation), String(where)));
			} else {
				return false;
			}
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
		if (obj_class != class_name && !ClassDB::is_parent_class(obj_class, class_name)) {
			if (p_output_errors) {
				ERR_FAIL_V_MSG(false, vformat("Attempted to %s an object of type '%s' into a %s, which does not inherit from '%s'.", String(p_operation), object->get_class(), where, String(class_name)));
			} else {
				return false;
			}
		}

		if (script.is_null()) {
			return true; // All good, no script requested.
		}

		Ref<Script> other_script = object->get_script();

		// Check base script..
		if (other_script.is_null()) {
			if (p_output_errors) {
				ERR_FAIL_V_MSG(false, vformat("Attempted to %s an object into a %s, that does not inherit from '%s'.", String(p_operation), String(where), String(script->get_class_name())));
			} else {
				return false;
			}
		}
		if (!other_script->inherits_script(script)) {
			if (p_output_errors) {
				ERR_FAIL_V_MSG(false, vformat("Attempted to %s an object into a %s, that does not inherit from '%s'.", String(p_operation), String(where), String(script->get_class_name())));
			} else {
				return false;
			}
		}

		return true;
	}

public:
	_FORCE_INLINE_ bool validate(Variant &inout_variant, const char *p_operation = "use") const {
		if (!_internal_validate(inout_variant, p_operation, true)) {
			return false;
		}

		if (is_nested() && (type == Variant::ARRAY || type == Variant::DICTIONARY)) {
			return validate_nested_container(inout_variant, p_operation);
		}

		return true;
	}

	_FORCE_INLINE_ bool validate_object(const Variant &p_variant, const char *p_operation = "use") const {
		return _internal_validate_object(p_variant, p_operation, true);
	}

	_FORCE_INLINE_ bool test_validate(const Variant &p_variant) const {
		Variant tmp = p_variant;
		return _internal_validate(tmp, "", false);
	}

	_FORCE_INLINE_ bool can_reference(const ContainerTypeValidate &p_type) const {
		if (type != p_type.type) {
			return false;
		} else if (type != Variant::OBJECT) {
			return can_reference_nested(p_type);
		}

		if (class_name == StringName()) {
			return can_reference_nested(p_type);
		} else if (p_type.class_name == StringName()) {
			return false;
		} else if (class_name != p_type.class_name && !ClassDB::is_parent_class(p_type.class_name, class_name)) {
			return false;
		}

		if (script.is_null()) {
			return can_reference_nested(p_type);
		} else if (p_type.script.is_null()) {
			return false;
		} else if (script != p_type.script && !p_type.script->inherits_script(script)) {
			return false;
		}

		return can_reference_nested(p_type);
	}

	bool can_reference_nested(const ContainerTypeValidate &p_type) const {
		if (nested_types.size() != p_type.nested_types.size()) {
			return false;
		}
		for (int i = 0; i < nested_types.size(); i++) {
			if (!nested_types[i].can_reference(p_type.nested_types[i])) {
				return false;
			}
		}
		return true;
	}

	bool operator==(const ContainerTypeValidate &p_type) const {
		if (type != p_type.type || class_name != p_type.class_name || script != p_type.script) {
			return false;
		}
		if (nested_types.size() != p_type.nested_types.size()) {
			return false;
		}
		for (int i = 0; i < nested_types.size(); i++) {
			if (nested_types[i] != p_type.nested_types[i]) {
				return false;
			}
		}
		return true;
	}

	bool operator!=(const ContainerTypeValidate &p_type) const {
		return !(*this == p_type);
	}

	bool validate_nested_container(Variant &inout_variant, const char *p_operation = "use") const {
		if (get_depth() > MAX_CONTAINER_NESTING_DEPTH) {
			ERR_FAIL_V_MSG(false, vformat("Nested container depth exceeds maximum allowed (%d levels).", MAX_CONTAINER_NESTING_DEPTH));
		}

		if (type == Variant::ARRAY) {
			return validate_nested_array(inout_variant, p_operation);
		} else if (type == Variant::DICTIONARY) {
			return validate_nested_dictionary(inout_variant, p_operation);
		}

		return true;
	}

	bool validate_nested_array(Variant &inout_variant, const char *p_operation = "use") const {
		if (nested_types.is_empty()) {
			return true;
		}

		Array array = inout_variant;
		const ContainerTypeValidate &element_type = nested_types[0];

		for (int i = 0; i < array.size(); i++) {
			Variant element = array[i];
			if (!element_type.validate(element, p_operation)) {
				ERR_FAIL_V_MSG(false, vformat("Array element at index %d failed type validation.", i));
			}
			// Only write back if the element was actually coerced, to preserve COW sharing.
			if (element != array[i]) {
				array[i] = element;
			}
		}

		inout_variant = array;
		return true;
	}

	bool validate_nested_dictionary(Variant &inout_variant, const char *p_operation = "use") const {
		if (nested_types.size() < 2) {
			return true;
		}

		const Dictionary source = inout_variant;
		const ContainerTypeValidate &key_type = nested_types[0];
		const ContainerTypeValidate &value_type = nested_types[1];

		// Build a fresh dictionary so validation never mutates the caller's dict mid-iteration,
		// and keeps COW intact when nothing needs coercion.
		Dictionary result;
		bool any_coerced = false;
		for (const KeyValue<Variant, Variant> &kv : source) {
			Variant key = kv.key;
			Variant value = kv.value;

			if (!key_type.validate(key, p_operation)) {
				ERR_FAIL_V_MSG(false, "Dictionary key failed type validation.");
			}
			if (!value_type.validate(value, p_operation)) {
				ERR_FAIL_V_MSG(false, "Dictionary value failed type validation.");
			}

			if (!any_coerced && (key != kv.key || value != kv.value)) {
				any_coerced = true;
			}
			if (result.has(key)) {
				ERR_FAIL_V_MSG(false, "Dictionary key coercion would create a duplicate key.");
			}
			result[key] = value;
		}

		if (any_coerced) {
			inout_variant = result;
		}
		return true;
	}
};
