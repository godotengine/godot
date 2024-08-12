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

#ifndef CONTAINER_TYPE_VALIDATE_H
#define CONTAINER_TYPE_VALIDATE_H

#include "core/object/script_language.h"
#include "core/variant/variant.h"

struct ContainerTypeValidate {
	Variant::Type type = Variant::NIL;
	GodotTypeInfo::Metadata meta = GodotTypeInfo::METADATA_NONE;
	StringName class_name;
	Ref<Script> script;
	const char *where = "container";

	_FORCE_INLINE_ bool can_reference(const ContainerTypeValidate &p_type) const {
		if (type != p_type.type) {
			return false;
		} else if (type != Variant::OBJECT) {
			return true;
		}

		if (class_name == StringName()) {
			return true;
		} else if (p_type.class_name == StringName()) {
			return false;
		} else if (class_name != p_type.class_name && !ClassDB::is_parent_class(p_type.class_name, class_name)) {
			return false;
		}

		if (script.is_null()) {
			return true;
		} else if (p_type.script.is_null()) {
			return false;
		} else if (script != p_type.script && !p_type.script->inherits_script(script)) {
			return false;
		}

		return true;
	}

	_FORCE_INLINE_ bool operator==(const ContainerTypeValidate &p_type) const {
		return type == p_type.type && class_name == p_type.class_name && script == p_type.script;
	}
	_FORCE_INLINE_ bool operator!=(const ContainerTypeValidate &p_type) const {
		return type != p_type.type || class_name != p_type.class_name || script != p_type.script;
	}

	// Coerces String and StringName into each other and int into float when needed.
	_FORCE_INLINE_ bool validate(Variant &inout_variant, const char *p_operation = "use") const {
		if (type == Variant::NIL) {
			return true;
		}

		if (type != inout_variant.get_type()) {
			if (inout_variant.get_type() == Variant::NIL && type == Variant::OBJECT) {
				return true;
			}
			if (type == Variant::STRING && inout_variant.get_type() == Variant::STRING_NAME) {
				inout_variant = String(inout_variant);
				return true;
			} else if (type == Variant::STRING_NAME && inout_variant.get_type() == Variant::STRING) {
				inout_variant = StringName(inout_variant);
				return true;
			} else if (type == Variant::FLOAT && inout_variant.get_type() == Variant::INT) {
				inout_variant = (float)inout_variant;
				return true;
			}

			ERR_FAIL_V_MSG(false, "Attempted to " + String(p_operation) + " a variable of type '" + Variant::get_type_name(inout_variant.get_type()) + "' into a " + where + " of type '" + Variant::get_type_name(type) + "'.");
		}

		if (type == Variant::INT) {
			return validate_int(inout_variant, p_operation);
		}

		if (type == Variant::FLOAT) {
			return validate_float(inout_variant, p_operation);
		}

		if (type != Variant::OBJECT) {
			return true;
		}

		return validate_object(inout_variant, p_operation);
	}

	_FORCE_INLINE_ bool validate_int(const Variant &p_variant, const char *p_operation = "use") const {
		ERR_FAIL_COND_V(p_variant.get_type() != Variant::INT, false);

		int64_t value = p_variant;

#define CHECK_INT_BOUNDS(min, max, meta_name)                                                                                                                                               \
	{                                                                                                                                                                                       \
		if (value < min || value > max) {                                                                                                                                                   \
			ERR_FAIL_V_MSG(false, "Attempted to " + String(p_operation) + " a variable of type 'int' into a " + String(where) + ", that does not match the metadata '" + meta_name + "'."); \
		}                                                                                                                                                                                   \
	}

		if (meta == GodotTypeInfo::METADATA_INT_IS_UINT8) {
			CHECK_INT_BOUNDS(0, UINT8_MAX, "uint8");
		} else if (meta == GodotTypeInfo::METADATA_INT_IS_INT8) {
			CHECK_INT_BOUNDS(INT8_MIN, INT8_MAX, "int8");
		} else if (meta == GodotTypeInfo::METADATA_INT_IS_UINT16) {
			CHECK_INT_BOUNDS(0, UINT16_MAX, "uint16");
		} else if (meta == GodotTypeInfo::METADATA_INT_IS_INT16) {
			CHECK_INT_BOUNDS(INT16_MIN, INT16_MAX, "int16");
		} else if (meta == GodotTypeInfo::METADATA_INT_IS_UINT32) {
			CHECK_INT_BOUNDS(0, UINT32_MAX, "uint32");
		} else if (meta == GodotTypeInfo::METADATA_INT_IS_INT32) {
			CHECK_INT_BOUNDS(INT32_MIN, INT32_MAX, "int32");
		}

		return true;

#undef CHECK_INT_BOUNDS
	}

	_FORCE_INLINE_ bool validate_float(const Variant &p_variant, const char *p_operation = "use") const {
		ERR_FAIL_COND_V(p_variant.get_type() != Variant::FLOAT, false);

		if (meta == GodotTypeInfo::METADATA_REAL_IS_FLOAT) {
			// TODO: Check if p_variant fits in a 32-bit float.
		}

		return true;
	}

	_FORCE_INLINE_ bool validate_object(const Variant &p_variant, const char *p_operation = "use") const {
		ERR_FAIL_COND_V(p_variant.get_type() != Variant::OBJECT, false);

#ifdef DEBUG_ENABLED
		ObjectID object_id = p_variant;
		if (object_id == ObjectID()) {
			return true; // This is fine, it's null.
		}
		Object *object = ObjectDB::get_instance(object_id);
		ERR_FAIL_NULL_V_MSG(object, false, "Attempted to " + String(p_operation) + " an invalid (previously freed?) object instance into a '" + String(where) + ".");
#else
		Object *object = p_variant;
		if (object == nullptr) {
			return true; //fine
		}
#endif
		if (class_name == StringName()) {
			return true; // All good, no class type requested.
		}

		StringName obj_class = object->get_class_name();
		if (obj_class != class_name) {
			ERR_FAIL_COND_V_MSG(!ClassDB::is_parent_class(object->get_class_name(), class_name), false, "Attempted to " + String(p_operation) + " an object of type '" + object->get_class() + "' into a " + where + ", which does not inherit from '" + String(class_name) + "'.");
		}

		if (script.is_null()) {
			return true; // All good, no script requested.
		}

		Ref<Script> other_script = object->get_script();

		// Check base script..
		ERR_FAIL_COND_V_MSG(other_script.is_null(), false, "Attempted to " + String(p_operation) + " an object into a " + String(where) + ", that does not inherit from '" + String(script->get_class_name()) + "'.");
		ERR_FAIL_COND_V_MSG(!other_script->inherits_script(script), false, "Attempted to " + String(p_operation) + " an object into a " + String(where) + ", that does not inherit from '" + String(script->get_class_name()) + "'.");

		return true;
	}
};

#endif // CONTAINER_TYPE_VALIDATE_H
