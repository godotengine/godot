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
#include "core/variant/struct_generator.h"
#include "core/variant/variant.h"

struct ValidatedVariant {
	Variant value;
	bool valid;

	ValidatedVariant(const Variant &p_value, const bool p_valid) {
		value = p_value;
		valid = p_valid;
	}
};

struct ContainerTypeValidate {
	Variant::Type type = Variant::NIL;
	StringName class_name;
	Ref<Script> script;

	bool is_array_of_structs = false;
	const StructInfo *struct_info = nullptr; // TODO: if is_array_of_structs == true, then require struct_info != nullptr, but not sure the best way to enforce this.
	const char *where = "container";

	ContainerTypeValidate() {};
	ContainerTypeValidate(const Variant::Type p_type, const StringName &p_class_name, const Ref<Script> &p_script, const char *p_where = "container") {
		type = p_type;
		class_name = p_class_name;
		script = p_script;
		struct_info = nullptr;
		where = p_where;
	}
	ContainerTypeValidate(const StructInfo &p_struct_info, bool p_is_array_of_structs = false) {
		type = Variant::ARRAY;
		class_name = p_struct_info.name;
		script = Ref<Script>();
		is_array_of_structs = p_is_array_of_structs;
		struct_info = &p_struct_info;
		where = p_is_array_of_structs ? "TypedArray" : "Struct";
	}

	_FORCE_INLINE_ bool can_reference(const ContainerTypeValidate &p_type) const {
		if (type == Variant::NIL) {
			return true;
		}
		if (type != p_type.type) {
			return false;
		}
		if (is_array_of_structs != p_type.is_array_of_structs) {
			return false;
		}
		if (!StructInfo::is_compatible(struct_info, p_type.struct_info)) {
			return false;
		}
		if (type != Variant::OBJECT) {
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
		return type == p_type.type && class_name == p_type.class_name && script == p_type.script && is_array_of_structs == p_type.is_array_of_structs && StructInfo::is_compatible(struct_info, p_type.struct_info);
	}
	_FORCE_INLINE_ bool operator!=(const ContainerTypeValidate &p_type) const {
		return type != p_type.type || class_name != p_type.class_name || script != p_type.script || is_array_of_structs != p_type.is_array_of_structs || !StructInfo::is_compatible(struct_info, p_type.struct_info);
	}

	_FORCE_INLINE_ static ValidatedVariant validate_variant_type(const Variant::Type p_type, const Variant &p_variant, const char *p_where, const char *p_operation = "use") {
		if (p_type == Variant::NIL) {
			return ValidatedVariant(p_variant, true);
		}
		if (p_type == p_variant.get_type()) {
			return ValidatedVariant(p_variant, true);
		}
		if (p_type == Variant::OBJECT && p_variant.get_type() == Variant::NIL) {
			return ValidatedVariant(p_variant, true);
		}
		if (p_type == Variant::STRING && p_variant.get_type() == Variant::STRING_NAME) {
			return ValidatedVariant(String(p_variant), true);
		}
		if (p_type == Variant::STRING_NAME && p_variant.get_type() == Variant::STRING) {
			return ValidatedVariant(StringName(p_variant), true);
		}
		if (p_type == Variant::FLOAT && p_variant.get_type() == Variant::INT) {
			return ValidatedVariant((float)p_variant, true);
		}
		ERR_FAIL_V_MSG(ValidatedVariant(p_variant, false), "Attempted to " + String(p_operation) + " a variable of type '" + Variant::get_type_name(p_variant.get_type()) + "' into a " + p_where + " of type '" + Variant::get_type_name(p_type) + "'.");
	}

	_FORCE_INLINE_ bool validate_object(const Variant &p_variant, const char *p_operation = "use") const {
		return validate_object(class_name, script, p_variant, where, p_operation);
	}

	_FORCE_INLINE_ static bool validate_object(const StringName &p_class_name, const Ref<Script> &p_script, const Variant &p_variant, const char *p_where, const char *p_operation = "use") {
		ERR_FAIL_COND_V(p_variant.get_type() != Variant::OBJECT, false);

#ifdef DEBUG_ENABLED
		ObjectID object_id = p_variant;
		if (object_id == ObjectID()) {
			return true; // This is fine, it's null.
		}
		Object *object = ObjectDB::get_instance(object_id);
		ERR_FAIL_NULL_V_MSG(object, false, "Attempted to " + String(p_operation) + " an invalid (previously freed?) object instance into a '" + String(p_where) + ".");
#else
		Object *object = p_variant;
		if (object == nullptr) {
			return true; //fine
		}
#endif
		if (p_class_name == StringName()) {
			return true; // All good, no class type requested.
		}

		StringName obj_class = object->get_class_name();
		if (obj_class != p_class_name) {
			ERR_FAIL_COND_V_MSG(!ClassDB::is_parent_class(object->get_class_name(), p_class_name), false, "Attempted to " + String(p_operation) + " an object of type '" + object->get_class() + "' into a " + p_where + ", which does not inherit from '" + String(p_class_name) + "'.");
		}

		if (p_script.is_null()) {
			return true; // All good, no script requested.
		}

		Ref<Script> other_script = object->get_script();

		// Check base script..
		ERR_FAIL_COND_V_MSG(other_script.is_null(), false, "Attempted to " + String(p_operation) + " an object into a " + String(p_where) + ", that does not inherit from '" + String(p_script->get_class_name()) + "'.");
		ERR_FAIL_COND_V_MSG(!other_script->inherits_script(p_script), false, "Attempted to " + String(p_operation) + " an object into a " + String(p_where) + ", that does not inherit from '" + String(p_script->get_class_name()) + "'.");

		return true;
	}

	_FORCE_INLINE_ ValidatedVariant validate(const Variant &p_variant, const char *p_operation = "use") const {
		// TODO: Ensure !is_struct ?
		// Coerces String and StringName into each other and int into float when needed.
		ValidatedVariant ret = ContainerTypeValidate::validate_variant_type(type, p_variant, where, p_operation);
		if (!ret.valid) {
			return ret;
		}

		// Variant types match
		if (type == Variant::ARRAY) {
			const Array array = p_variant;
			if (array.is_struct()) { // validating a struct into a typed array of structs
				ret.valid = StructInfo::is_compatible(struct_info, array.get_struct_info());
			} else { // validating a typed array into a typed array of typed arrays (which is currently not supported)
				ret.valid = class_name == array.get_typed_class_name();
			}
		} else if (type == Variant::OBJECT) {
			ret.valid = validate_object(p_variant, p_operation);
		}
		return ret;
	}

	_FORCE_INLINE_ ValidatedVariant validate_struct_member(const Variant &p_variant, const int p_struct_index, const char *p_operation = "use") const {
		// TODO: Ensure is_struct and struct_info != nullptr ?
		CRASH_BAD_INDEX_MSG(p_struct_index, struct_info->count, "Struct tried validation for a non-existent member");
		const Variant::Type variant_type = struct_info->types[p_struct_index];
		// Coerces String and StringName into each other and int into float when needed.
		ValidatedVariant ret = ContainerTypeValidate::validate_variant_type(variant_type, p_variant, where, p_operation);
		if (!ret.valid) {
			return ret;
		}

		// Variant types match
		if (variant_type == Variant::ARRAY) {
			const Array array = p_variant;
			// Valid if (the struct member is itself a struct and the other array is a compatible struct) or (neither the struct member nor the other array are a struct).
			ret.valid = StructInfo::is_compatible(struct_info->struct_member_infos[p_struct_index], array.get_struct_info());
		} else if (variant_type == Variant::OBJECT) {
			ret.valid = validate_object(struct_info->class_names[p_struct_index], Ref<Script>(), p_variant, where, p_operation);
		}
		return ret;
	}
};

#endif // CONTAINER_TYPE_VALIDATE_H
