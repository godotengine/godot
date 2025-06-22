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

#include "core/object/script_language.h"
#include "core/templates/rb_set.h"
#include "core/templates/vector.h"
#include "core/variant/variant.h"

struct ContainerType {
	Variant::Type builtin_type = Variant::NIL;
	StringName class_name;
	Ref<Script> script;
	Vector<ContainerType> nested_types;
};

struct ContainerTypeValidate {
	static constexpr int MAX_NESTING_DEPTH = 8;

	Variant::Type type = Variant::NIL;
	StringName class_name;
	Ref<Script> script;
	const char *where = "container";
	Vector<ContainerTypeValidate> nested_types;

	_FORCE_INLINE_ bool is_nested() const {
		return !nested_types.is_empty();
	}

	_FORCE_INLINE_ int get_depth() const {
		RBSet<const ContainerTypeValidate *> visited;
		return get_depth_internal(visited);
	}

private:
	_FORCE_INLINE_ int get_depth_internal(RBSet<const ContainerTypeValidate *> &visited) const {
		if (visited.has(this)) {
			return MAX_NESTING_DEPTH + 1; // Force depth validation to fail
		}
		visited.insert(this);

		int depth = 0;
		for (const ContainerTypeValidate &nested : nested_types) {
			depth = MAX(depth, nested.get_depth_internal(visited));
		}

		visited.erase(this);
		return depth + 1; // +1 for the current level
	}

public:
	_FORCE_INLINE_ bool is_depth_valid() const {
		return get_depth() <= MAX_NESTING_DEPTH;
	}

	bool can_reference(const ContainerTypeValidate &p_type) const {
		if (type != p_type.type) {
			return false;
		} else if (type != Variant::OBJECT) {
			// For non-object types, check nested compatibility if both have nested types
			if (!nested_types.is_empty() && !p_type.nested_types.is_empty()) {
				if (nested_types.size() != p_type.nested_types.size()) {
					return false;
				}
				for (int i = 0; i < nested_types.size(); i++) {
					if (!nested_types[i].can_reference(p_type.nested_types[i])) {
						return false;
					}
				}
			}
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
		return type == p_type.type && class_name == p_type.class_name && script == p_type.script && nested_types == p_type.nested_types;
	}
	_FORCE_INLINE_ bool operator!=(const ContainerTypeValidate &p_type) const {
		return type != p_type.type || class_name != p_type.class_name || script != p_type.script || nested_types != p_type.nested_types;
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

			ERR_FAIL_V_MSG(false, vformat("Attempted to %s a variable of type '%s' into a %s of type '%s'.", String(p_operation), Variant::get_type_name(inout_variant.get_type()), where, Variant::get_type_name(type)));
		}

		if (type != Variant::OBJECT) {
			return true;
		}

		return validate_object(inout_variant, p_operation);
	}

	// Original validate_object method (unchanged)
	_FORCE_INLINE_ bool validate_object(const Variant &p_variant, const char *p_operation = "use") const {
		ERR_FAIL_COND_V(p_variant.get_type() != Variant::OBJECT, false);

#ifdef DEBUG_ENABLED
		ObjectID object_id = p_variant;
		if (object_id == ObjectID()) {
			return true; // This is fine, it's null.
		}
		Object *object = ObjectDB::get_instance(object_id);
		ERR_FAIL_NULL_V_MSG(object, false, vformat("Attempted to %s an invalid (previously freed?) object instance into a '%s'.", String(p_operation), String(where)));
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
		if (obj_class != class_name) {
			ERR_FAIL_COND_V_MSG(!ClassDB::is_parent_class(obj_class, class_name), false, vformat("Attempted to %s an object of type '%s' into a %s, which does not inherit from '%s'.", String(p_operation), object->get_class(), where, String(class_name)));
		}

		if (script.is_null()) {
			return true; // All good, no script requested.
		}

		Ref<Script> other_script = object->get_script();

		// Check base script..
		ERR_FAIL_COND_V_MSG(other_script.is_null(), false, vformat("Attempted to %s an object into a %s, that does not inherit from '%s'.", String(p_operation), String(where), String(script->get_class_name())));
		ERR_FAIL_COND_V_MSG(!other_script->inherits_script(script), false, vformat("Attempted to %s an object into a %s, that does not inherit from '%s'.", String(p_operation), String(where), String(script->get_class_name())));

		return true;
	}
};
