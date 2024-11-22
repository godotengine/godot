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

	ValidatedVariant(const Variant &p_value, const bool p_valid) :
			value(p_value), valid(p_valid) {
	}
};

struct ContainerTypeValidate {
	Variant::Type type = Variant::NIL;
	StringName class_name;
	Ref<Script> script;

	const char *where = "container";

private:
	// is_struct must be false if struct_info == nullptr
	// if is_struct is false and struct_info != nullptr, the container is a TypedArray of Structs.
	const StructInfo *struct_info = nullptr;
	bool is_struct = false;

public:
	ContainerTypeValidate() {}
	ContainerTypeValidate(const Variant::Type p_type, const StringName &p_class_name, const Ref<Script> &p_script, const char *p_where = "container") :
			type(p_type), class_name(p_class_name), script(p_script), where(p_where) {
	}
	ContainerTypeValidate(const StructInfo &p_struct_info, bool p_is_struct = true) :
			type(Variant::ARRAY), class_name(p_struct_info.name), script(Ref<Script>()), where(p_is_struct ? "Struct" : "TypedArray"), struct_info(&p_struct_info), is_struct(p_is_struct) {
	}

	_FORCE_INLINE_ void set_struct_info(const StructInfo *p_struct_info) {
		struct_info = p_struct_info;
		if (!struct_info) {
			is_struct = false;
		}
	}
	_FORCE_INLINE_ const StructInfo *get_struct_info() const {
		return struct_info;
	}
	_FORCE_INLINE_ bool get_is_struct() const {
		return is_struct;
	}
	_FORCE_INLINE_ bool is_array_of_structs() const {
		return !is_struct && struct_info != nullptr;
	}

	_FORCE_INLINE_ bool can_reference(const ContainerTypeValidate &p_type) const {
		if (type == Variant::NIL) {
			return true;
		}
		if (type != p_type.type) {
			return false;
		}
		if (is_struct != p_type.is_struct) {
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
		return type == p_type.type && class_name == p_type.class_name && script == p_type.script && is_struct == p_type.is_struct && StructInfo::is_compatible(struct_info, p_type.struct_info);
	}
	_FORCE_INLINE_ bool operator!=(const ContainerTypeValidate &p_type) const {
		return type != p_type.type || class_name != p_type.class_name || script != p_type.script || is_struct != p_type.is_struct || !StructInfo::is_compatible(struct_info, p_type.struct_info);
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
		ERR_FAIL_V_MSG(ValidatedVariant(p_variant, false), vformat("Attempted to %s a variable of type '%s' into a %s of type '%s'.", String(p_operation), Variant::get_type_name(p_variant.get_type()), p_where, Variant::get_type_name(p_type)));
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
		ERR_FAIL_NULL_V_MSG(object, false, vformat("Attempted to %s an invalid (previously freed?) object instance into a '%s'.", String(p_operation), String(p_where)));
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
			ERR_FAIL_COND_V_MSG(!ClassDB::is_parent_class(object->get_class_name(), p_class_name), false, vformat("Attempted to %s an object of type '%s' into a %s, which does not inherit from '%s'.", String(p_operation), object->get_class(), p_where, String(p_class_name)));
		}

		if (p_script.is_null()) {
			return true; // All good, no script requested.
		}

		Ref<Script> other_script = object->get_script();

		// Check base script..
		ERR_FAIL_COND_V_MSG(other_script.is_null(), false, vformat("Attempted to %s an object into a %s, that does not inherit from '%s'.", String(p_operation), String(p_where), String(p_script->get_class_name())));
		ERR_FAIL_COND_V_MSG(!other_script->inherits_script(p_script), false, vformat("Attempted to %s an object into a %s, that does not inherit from '%s'.", String(p_operation), String(p_where), String(p_script->get_class_name())));

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

		switch (variant_type) {
			case Variant::ARRAY: {
				const Array &default_array = struct_info->default_values[p_struct_index];
				ret.valid = default_array.can_reference(p_variant);
				return ret;
			}
			case Variant::DICTIONARY: {
				const Dictionary &default_dict = struct_info->default_values[p_struct_index];
				ret.valid = default_dict.can_reference(p_variant);
				return ret;
			}
			case Variant::OBJECT: {
				ret.valid = validate_object(struct_info->type_names[p_struct_index], static_cast<Ref<Script>>(struct_info->scripts[p_struct_index]), p_variant, where, p_operation);
				return ret;
			}
			default:
				return ret;
		}
	}
};

#endif // CONTAINER_TYPE_VALIDATE_H
