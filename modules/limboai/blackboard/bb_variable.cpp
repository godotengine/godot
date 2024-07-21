/**
 * bb_variable.cpp
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#include "bb_variable.h"

#include "../util/limbo_compat.h"

void BBVariable::unref() {
	if (data && data->refcount.unref()) {
		memdelete(data);
	}
	data = nullptr;
}

void BBVariable::set_value(const Variant &p_value) {
	data->value = p_value; // Setting value even when bound as a fallback in case the binding fails.
	data->value_changed = true;

	if (is_bound()) {
		Object *obj = ObjectDB::get_instance(ObjectID(data->bound_object));
		ERR_FAIL_COND_MSG(!obj, "Blackboard: Failed to get bound object.");
#ifdef LIMBOAI_MODULE
		bool r_valid;
		obj->set(data->bound_property, p_value, &r_valid);
		ERR_FAIL_COND_MSG(!r_valid, vformat("Blackboard: Failed to set bound property `%s` on %s", data->bound_property, obj));
#elif LIMBOAI_GDEXTENSION
		obj->set(data->bound_property, p_value);
#endif
	}
}

Variant BBVariable::get_value() const {
	if (is_bound()) {
		Object *obj = ObjectDB::get_instance(ObjectID(data->bound_object));
		ERR_FAIL_COND_V_MSG(!obj, data->value, "Blackboard: Failed to get bound object.");
#ifdef LIMBOAI_MODULE
		bool r_valid;
		Variant ret = obj->get(data->bound_property, &r_valid);
		ERR_FAIL_COND_V_MSG(!r_valid, data->value, vformat("Blackboard: Failed to get bound property `%s` on %s", data->bound_property, obj));
#elif LIMBOAI_GDEXTENSION
		Variant ret = obj->get(data->bound_property);
#endif
		return ret;
	}
	return data->value;
}

void BBVariable::set_type(Variant::Type p_type) {
	data->type = p_type;
	data->value = VARIANT_DEFAULT(p_type);
}

Variant::Type BBVariable::get_type() const {
	return data->type;
}

void BBVariable::set_hint(PropertyHint p_hint) {
	data->hint = p_hint;
}

PropertyHint BBVariable::get_hint() const {
	return data->hint;
}

void BBVariable::set_hint_string(const String &p_hint_string) {
	data->hint_string = p_hint_string;
}

String BBVariable::get_hint_string() const {
	return data->hint_string;
}

BBVariable BBVariable::duplicate(bool p_deep) const {
	BBVariable var;
	var.data->hint = data->hint;
	var.data->hint_string = data->hint_string;
	var.data->type = data->type;
	if (p_deep) {
		var.data->value = data->value.duplicate(p_deep);
	} else {
		var.data->value = data->value;
	}
	var.data->binding_path = data->binding_path;
	var.data->bound_object = data->bound_object;
	var.data->bound_property = data->bound_property;
	return var;
}

bool BBVariable::is_same_prop_info(const BBVariable &p_other) const {
	if (data->type != p_other.data->type) {
		return false;
	}
	if (data->hint != p_other.data->hint) {
		return false;
	}
	if (data->hint_string != p_other.data->hint_string) {
		return false;
	}
	return true;
}

void BBVariable::copy_prop_info(const BBVariable &p_other) {
	data->type = p_other.data->type;
	data->hint = p_other.data->hint;
	data->hint_string = p_other.data->hint_string;
}

void BBVariable::bind(Object *p_object, const StringName &p_property) {
	ERR_FAIL_NULL_MSG(p_object, "Blackboard: Binding failed - object is null.");
	ERR_FAIL_COND_MSG(p_property == StringName(), "Blackboard: Binding failed - property name is empty.");
	ERR_FAIL_COND_MSG(!OBJECT_HAS_PROPERTY(p_object, p_property), vformat("Blackboard: Binding failed - %s has no property `%s`.", p_object, p_property));
	data->bound_object = p_object->get_instance_id();
	data->bound_property = p_property;
}

void BBVariable::unbind() {
	data->bound_object = 0;
	data->bound_property = StringName();
}

bool BBVariable::operator==(const BBVariable &p_var) const {
	if (data == p_var.data) {
		return true;
	}

	if (!data || !p_var.data) {
		return false;
	}

	if (data->type != p_var.data->type) {
		return false;
	}

	if (data->hint != p_var.data->hint) {
		return false;
	}

	if (data->hint_string != p_var.data->hint_string) {
		return false;
	}

	if (get_value() != p_var.get_value()) {
		return false;
	}

	return true;
}

bool BBVariable::operator!=(const BBVariable &p_var) const {
	return !(*this == p_var);
}

void BBVariable::operator=(const BBVariable &p_var) {
	if (this == &p_var) {
		return;
	}

	unref();

	if (p_var.data && p_var.data->refcount.ref()) {
		data = p_var.data;
	}
}

BBVariable::BBVariable(const BBVariable &p_var) {
	if (p_var.data && p_var.data->refcount.ref()) {
		data = p_var.data;
	}
}

BBVariable::BBVariable(Variant::Type p_type, const Variant &p_value,PropertyHint p_hint, const String &p_hint_string) {
	data = memnew(Data);
	data->refcount.init();

	set_type(p_type);
	data->hint = p_hint;
	data->hint_string = p_hint_string;
	if(p_value.get_type() != Variant::NIL) {
		data->value = p_value;
	}
}

BBVariable::~BBVariable() {
	unref();
}
