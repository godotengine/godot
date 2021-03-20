/*************************************************************************/
/*  container_type_validate.h                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#ifndef CONTAINER_TYPE_VALIDATE_H
#define CONTAINER_TYPE_VALIDATE_H

#include "core/object/script_language.h"
#include "core/variant/variant.h"

struct ContainerTypeValidate {
	Variant::Type type = Variant::NIL;
	StringName class_name;
	Ref<Script> script;
	const char *where = "container";

	_FORCE_INLINE_ bool can_reference(const ContainerTypeValidate &p_type) const {
		if (type == p_type.type) {
			if (type != Variant::OBJECT) {
				return true; //nothing else to check
			}
		} else {
			return false;
		}

		//both are object

		if ((class_name != StringName()) != (p_type.class_name != StringName())) {
			return false; //both need to have class or none
		}

		if (class_name != p_type.class_name) {
			if (!ClassDB::is_parent_class(p_type.class_name, class_name)) {
				return false;
			}
		}

		if (script.is_null() != p_type.script.is_null()) {
			return false;
		}

		if (script != p_type.script) {
			if (!p_type.script->inherits_script(script)) {
				return false;
			}
		}

		return true;
	}

	_FORCE_INLINE_ bool validate(const Variant &p_variant, const char *p_operation = "use") {
		if (type == Variant::NIL) {
			return true;
		}

		ERR_FAIL_COND_V_MSG(type != p_variant.get_type(), false, "Attempted to " + String(p_operation) + " a variable of type '" + Variant::get_type_name(p_variant.get_type()) + "' into a " + where + " of type '" + Variant::get_type_name(type) + "'.");
		if (type != p_variant.get_type()) {
			return false;
		}

		if (type != Variant::OBJECT) {
			return true;
		}
#ifdef DEBUG_ENABLED
		ObjectID object_id = p_variant;
		if (object_id == ObjectID()) {
			return true; //fine its null;
		}
		Object *object = ObjectDB::get_instance(object_id);
		ERR_FAIL_COND_V_MSG(object == nullptr, false, "Attempted to " + String(p_operation) + " an invalid (previously freed?) object instance into a '" + String(where) + ".");
#else
		Object *object = p_variant;
		if (object == nullptr) {
			return true; //fine
		}
#endif
		if (class_name == StringName()) {
			return true; //all good, no class type requested
		}

		StringName obj_class = object->get_class_name();
		if (obj_class != class_name) {
			ERR_FAIL_COND_V_MSG(!ClassDB::is_parent_class(object->get_class_name(), class_name), false, "Attempted to " + String(p_operation) + " an object of type '" + object->get_class() + "' into a " + where + ", which does not inherit from '" + String(class_name) + "'.");
		}

		if (script.is_null()) {
			return true; //all good
		}

		Ref<Script> other_script = object->get_script();

		//check base script..
		ERR_FAIL_COND_V_MSG(other_script.is_null(), false, "Attempted to " + String(p_operation) + " an object into a " + String(where) + ", that does not inherit from '" + String(script->get_class_name()) + "'.");
		ERR_FAIL_COND_V_MSG(!other_script->inherits_script(script), false, "Attempted to " + String(p_operation) + " an object into a " + String(where) + ", that does not inherit from '" + String(script->get_class_name()) + "'.");

		return true;
	}
};

#endif // CONTAINER_TYPE_VALIDATE_H
