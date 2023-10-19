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
#include "core/variant/struct_generator.h"

struct ContainerTypeValidate {
	Variant::Type type = Variant::NIL;
	StringName class_name;
	Ref<Script> script;

	const StructInfo2 *struct_info;
	const char *where = "container";

	ContainerTypeValidate(){};
//	ContainerTypeValidate(const Variant::Type p_type, const StringName &p_class_name, const Ref<Script> &p_script = Ref<Script>(), const char *p_where = "container", const StructInfo2 *p_struct_info = nullptr) {
//		type = p_type;
//		class_name = p_class_name;
//		script = p_script;
//		struct_info = p_struct_info;
//		where = p_where;
//	}
	ContainerTypeValidate(const Variant::Type p_type, const char *p_where = "container") {
		type = p_type;
		class_name = StringName();
		script = Ref<Script>();
		struct_info = nullptr;
		where = p_where;
	}
	ContainerTypeValidate(const StringName &p_class_name, const Ref<Script> &p_script = Ref<Script>(), const char *p_where = "container") {
		type = Variant::OBJECT;
		class_name = p_class_name;
		script = p_script;
		struct_info = nullptr;
		where = p_where;
	}
	ContainerTypeValidate(const StructInfo2 *p_struct_info) {
		type = Variant::ARRAY;
		class_name = StringName();
		script = Ref<Script>();
		struct_info = p_struct_info;
		where = "Struct";
	}
//	ContainerTypeValidate(const Variant::Type p_type, const StringName &p_class_name, const uint32_t p_member_count, const StructMember *p_members, const uint32_t p_recursion_max = MAX_RECURSION, const uint32_t p_recursion_count = 0) {
//		type = p_type;
//		class_name = p_class_name;
//		where = "Struct";
//		struct_members.resize(p_member_count);
//		struct_member_names.resize(p_member_count);
//		if (!recursion_finished(p_recursion_max, p_recursion_count)) {
//			for (uint32_t i = 0; i < p_member_count; i++) {
//				StructMember member = p_members[i];
//				struct_members[i] = ContainerTypeValidate(member.type, member.class_name, member.sub_member_count, member.sub_members, p_recursion_max, p_recursion_count + 1);
//				struct_member_names[i] = member.name;
//			}
//		}
//	}

	_FORCE_INLINE_ bool is_struct() const {
		return struct_info != nullptr;
	}

	_FORCE_INLINE_ bool can_reference(const ContainerTypeValidate &p_type) const {
		if (!can_reference_base(p_type)) {
			return false;
		}

		if (recursion_finished(p_recursion_max, p_recursion_count)) {
			return true;
		}

		for (uint32_t i = 0; i < struct_info->count; i++) {
			if (!struct_members[i].can_reference(p_type.struct_members[i], p_recursion_max, p_recursion_count + 1)) {
				return false;
			}
		}
		return true;
	}

//	_FORCE_INLINE_ bool can_reference(const ContainerTypeValidate &p_type, const uint32_t p_recursion_max = 1, const uint32_t p_recursion_count = 0) const {
//		if (!can_reference_base(p_type)) {
//			return false;
//		}
//
//		if (recursion_finished(p_recursion_max, p_recursion_count)) {
//			return true;
//		}
//
//		for (uint32_t i = 0; i < struct_info->count; i++) {
//			if (!struct_members[i].can_reference(p_type.struct_members[i], p_recursion_max, p_recursion_count + 1)) {
//				return false;
//			}
//		}
//		return true;
//	}

	_FORCE_INLINE_ bool operator==(const ContainerTypeValidate &p_type) const {
		return equal_recursive(p_type);
	}
	_FORCE_INLINE_ bool operator!=(const ContainerTypeValidate &p_type) const {
		return !equal_recursive(p_type);
	}

	_FORCE_INLINE_ bool validate(Variant &inout_variant, const char *p_operation = "use") const {
		// Coerces String and StringName into each other and int into float when needed.
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

		if (type == Variant::ARRAY) {
			const Array array = inout_variant;
			if (is_struct()) {
				return operator==(array.get_type_validator());
			}
			return class_name == array.get_typed_class_name(); // TypedArray of structs
		} else if (type == Variant::OBJECT) {
			return validate_object(inout_variant, p_operation);
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

private:
	_FORCE_INLINE_ bool recursion_finished(const uint32_t p_recursion_max, const uint32_t p_recursion_count) const {
		if (p_recursion_count >= p_recursion_max) {
			return true;
		}
		if (p_recursion_count > MAX_RECURSION) {
			ERR_PRINT("Max recursion reached");
			return true;
		}
		return false;
	}

	_FORCE_INLINE_ bool can_reference_base(const ContainerTypeValidate &p_type) const {
		if (type != p_type.type) {
			return false;
		} else if (is_struct()) {
			return equal_base(p_type);
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

	_FORCE_INLINE_ bool equal_recursive(const ContainerTypeValidate &p_type, const uint32_t p_recursion_max = 1, const uint32_t p_recursion_count = 0) const {
		if (!equal_base(p_type)) {
			return false;
		}

		if (recursion_finished(p_recursion_max, p_recursion_count)) {
			return true;
		}

		for (uint32_t i = 0; i < struct_members.size(); i++) {
			if (struct_member_names[i] != p_type.struct_member_names[i]) {
				return false;
			}
			if (!struct_members[i].equal_recursive(p_type.struct_members[i], p_recursion_max, p_recursion_count + 1)) {
				return false;
			}
		}
		return true;
	}

	_FORCE_INLINE_ bool equal_base(const ContainerTypeValidate &p_type) const {
		return type == p_type.type && class_name == p_type.class_name && script == p_type.script && struct_members.size() == p_type.struct_members.size();
	}
};

#endif // CONTAINER_TYPE_VALIDATE_H
