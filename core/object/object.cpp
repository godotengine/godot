/**************************************************************************/
/*  object.cpp                                                            */
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

#include "object.h"

#include "core/core_string_names.h"
#include "core/extension/gdextension_manager.h"
#include "core/io/resource.h"
#include "core/object/class_db.h"
#include "core/object/message_queue.h"
#include "core/object/script_language.h"
#include "core/os/os.h"
#include "core/string/print_string.h"
#include "core/string/translation.h"
#include "core/templates/local_vector.h"
#include "core/variant/typed_array.h"

#ifdef DEBUG_ENABLED

struct _ObjectDebugLock {
	Object *obj;

	_ObjectDebugLock(Object *p_obj) {
		obj = p_obj;
		obj->_lock_index.ref();
	}
	~_ObjectDebugLock() {
		obj->_lock_index.unref();
	}
};

#define OBJ_DEBUG_LOCK _ObjectDebugLock _debug_lock(this);

#else

#define OBJ_DEBUG_LOCK

#endif

PropertyInfo::operator Dictionary() const {
	Dictionary d;
	d["name"] = name;
	d["class_name"] = class_name;
	d["type"] = type;
	d["hint"] = hint;
	d["hint_string"] = hint_string;
	d["usage"] = usage;
	return d;
}

PropertyInfo PropertyInfo::from_dict(const Dictionary &p_dict) {
	PropertyInfo pi;

	if (p_dict.has("type")) {
		pi.type = Variant::Type(int(p_dict["type"]));
	}

	if (p_dict.has("name")) {
		pi.name = p_dict["name"];
	}

	if (p_dict.has("class_name")) {
		pi.class_name = p_dict["class_name"];
	}

	if (p_dict.has("hint")) {
		pi.hint = PropertyHint(int(p_dict["hint"]));
	}

	if (p_dict.has("hint_string")) {
		pi.hint_string = p_dict["hint_string"];
	}

	if (p_dict.has("usage")) {
		pi.usage = p_dict["usage"];
	}

	return pi;
}

TypedArray<Dictionary> convert_property_list(const List<PropertyInfo> *p_list) {
	TypedArray<Dictionary> va;
	for (const List<PropertyInfo>::Element *E = p_list->front(); E; E = E->next()) {
		va.push_back(Dictionary(E->get()));
	}

	return va;
}

MethodInfo::operator Dictionary() const {
	Dictionary d;
	d["name"] = name;
	d["args"] = convert_property_list(&arguments);
	Array da;
	for (int i = 0; i < default_arguments.size(); i++) {
		da.push_back(default_arguments[i]);
	}
	d["default_args"] = da;
	d["flags"] = flags;
	d["id"] = id;
	Dictionary r = return_val;
	d["return"] = r;
	return d;
}

MethodInfo MethodInfo::from_dict(const Dictionary &p_dict) {
	MethodInfo mi;

	if (p_dict.has("name")) {
		mi.name = p_dict["name"];
	}
	Array args;
	if (p_dict.has("args")) {
		args = p_dict["args"];
	}

	for (int i = 0; i < args.size(); i++) {
		Dictionary d = args[i];
		mi.arguments.push_back(PropertyInfo::from_dict(d));
	}
	Array defargs;
	if (p_dict.has("default_args")) {
		defargs = p_dict["default_args"];
	}
	for (int i = 0; i < defargs.size(); i++) {
		mi.default_arguments.push_back(defargs[i]);
	}

	if (p_dict.has("return")) {
		mi.return_val = PropertyInfo::from_dict(p_dict["return"]);
	}

	if (p_dict.has("flags")) {
		mi.flags = p_dict["flags"];
	}

	return mi;
}

Object::Connection::operator Variant() const {
	Dictionary d;
	d["signal"] = signal;
	d["callable"] = callable;
	d["flags"] = flags;
	return d;
}

bool Object::Connection::operator<(const Connection &p_conn) const {
	if (signal == p_conn.signal) {
		return callable < p_conn.callable;
	} else {
		return signal < p_conn.signal;
	}
}

Object::Connection::Connection(const Variant &p_variant) {
	Dictionary d = p_variant;
	if (d.has("signal")) {
		signal = d["signal"];
	}
	if (d.has("callable")) {
		callable = d["callable"];
	}
	if (d.has("flags")) {
		flags = d["flags"];
	}
}

bool Object::_predelete() {
	_predelete_ok = 1;
	notification(NOTIFICATION_PREDELETE, true);
	if (_predelete_ok) {
		_class_name_ptr = nullptr; // Must restore, so constructors/destructors have proper class name access at each stage.
		notification(NOTIFICATION_PREDELETE_CLEANUP, true);
	}
	return _predelete_ok;
}

void Object::cancel_free() {
	_predelete_ok = false;
}

void Object::_postinitialize() {
	_class_name_ptr = _get_class_namev(); // Set the direct pointer, which is much faster to obtain, but can only happen after postinitialize.
	_initialize_classv();
	_class_name_ptr = nullptr; // May have been called from a constructor.
	notification(NOTIFICATION_POSTINITIALIZE);
}

void Object::get_valid_parents_static(List<String> *p_parents) {
}

void Object::_get_valid_parents_static(List<String> *p_parents) {
}

void Object::set(const StringName &p_name, const Variant &p_value, bool *r_valid) {
#ifdef TOOLS_ENABLED

	_edited = true;
#endif

	if (script_instance) {
		if (script_instance->set(p_name, p_value)) {
			if (r_valid) {
				*r_valid = true;
			}
			return;
		}
	}

	if (_extension && _extension->set) {
// C style pointer casts should never trigger a compiler warning because the risk is assumed by the user, so GCC should keep quiet about it.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-qualifiers"
#endif
		if (_extension->set(_extension_instance, (const GDExtensionStringNamePtr)&p_name, (const GDExtensionVariantPtr)&p_value)) {
			if (r_valid) {
				*r_valid = true;
			}
			return;
		}
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
	}

	// Try built-in setter.
	{
		if (ClassDB::set_property(this, p_name, p_value, r_valid)) {
			return;
		}
	}

	if (p_name == CoreStringNames::get_singleton()->_script) {
		set_script(p_value);
		if (r_valid) {
			*r_valid = true;
		}
		return;

	} else {
		Variant **V = metadata_properties.getptr(p_name);
		if (V) {
			**V = p_value;
			if (r_valid) {
				*r_valid = true;
			}
			return;
		} else if (p_name.operator String().begins_with("metadata/")) {
			// Must exist, otherwise duplicate() will not work.
			set_meta(p_name.operator String().replace_first("metadata/", ""), p_value);
			if (r_valid) {
				*r_valid = true;
			}
			return;
		}
	}

#ifdef TOOLS_ENABLED
	if (script_instance) {
		bool valid;
		script_instance->property_set_fallback(p_name, p_value, &valid);
		if (valid) {
			if (r_valid) {
				*r_valid = true;
			}
			return;
		}
	}
#endif

	// Something inside the object... :|
	bool success = _setv(p_name, p_value);
	if (success) {
		if (r_valid) {
			*r_valid = true;
		}
		return;
	}

	if (r_valid) {
		*r_valid = false;
	}
}

Variant Object::get(const StringName &p_name, bool *r_valid) const {
	Variant ret;

	if (script_instance) {
		if (script_instance->get(p_name, ret)) {
			if (r_valid) {
				*r_valid = true;
			}
			return ret;
		}
	}
	if (_extension && _extension->get) {
// C style pointer casts should never trigger a compiler warning because the risk is assumed by the user, so GCC should keep quiet about it.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-qualifiers"
#endif

		if (_extension->get(_extension_instance, (const GDExtensionStringNamePtr)&p_name, (GDExtensionVariantPtr)&ret)) {
			if (r_valid) {
				*r_valid = true;
			}
			return ret;
		}
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
	}

	// Try built-in getter.
	{
		if (ClassDB::get_property(const_cast<Object *>(this), p_name, ret)) {
			if (r_valid) {
				*r_valid = true;
			}
			return ret;
		}
	}

	if (p_name == CoreStringNames::get_singleton()->_script) {
		ret = get_script();
		if (r_valid) {
			*r_valid = true;
		}
		return ret;
	}

	const Variant *const *V = metadata_properties.getptr(p_name);

	if (V) {
		ret = **V;
		if (r_valid) {
			*r_valid = true;
		}
		return ret;

	} else {
#ifdef TOOLS_ENABLED
		if (script_instance) {
			bool valid;
			ret = script_instance->property_get_fallback(p_name, &valid);
			if (valid) {
				if (r_valid) {
					*r_valid = true;
				}
				return ret;
			}
		}
#endif
		// Something inside the object... :|
		bool success = _getv(p_name, ret);
		if (success) {
			if (r_valid) {
				*r_valid = true;
			}
			return ret;
		}

		if (r_valid) {
			*r_valid = false;
		}
		return Variant();
	}
}

void Object::set_indexed(const Vector<StringName> &p_names, const Variant &p_value, bool *r_valid) {
	if (p_names.is_empty()) {
		if (r_valid) {
			*r_valid = false;
		}
		return;
	}
	if (p_names.size() == 1) {
		set(p_names[0], p_value, r_valid);
		return;
	}

	bool valid = false;
	if (!r_valid) {
		r_valid = &valid;
	}

	List<Variant> value_stack;

	value_stack.push_back(get(p_names[0], r_valid));

	if (!*r_valid) {
		value_stack.clear();
		return;
	}

	for (int i = 1; i < p_names.size() - 1; i++) {
		value_stack.push_back(value_stack.back()->get().get_named(p_names[i], valid));
		if (r_valid) {
			*r_valid = valid;
		}

		if (!valid) {
			value_stack.clear();
			return;
		}
	}

	value_stack.push_back(p_value); // p_names[p_names.size() - 1]

	for (int i = p_names.size() - 1; i > 0; i--) {
		value_stack.back()->prev()->get().set_named(p_names[i], value_stack.back()->get(), valid);
		value_stack.pop_back();

		if (r_valid) {
			*r_valid = valid;
		}
		if (!valid) {
			value_stack.clear();
			return;
		}
	}

	set(p_names[0], value_stack.back()->get(), r_valid);
	value_stack.pop_back();

	ERR_FAIL_COND(!value_stack.is_empty());
}

Variant Object::get_indexed(const Vector<StringName> &p_names, bool *r_valid) const {
	if (p_names.is_empty()) {
		if (r_valid) {
			*r_valid = false;
		}
		return Variant();
	}
	bool valid = false;

	Variant current_value = get(p_names[0], &valid);
	for (int i = 1; i < p_names.size(); i++) {
		current_value = current_value.get_named(p_names[i], valid);

		if (!valid) {
			break;
		}
	}
	if (r_valid) {
		*r_valid = valid;
	}

	return current_value;
}

void Object::get_property_list(List<PropertyInfo> *p_list, bool p_reversed) const {
	if (script_instance && p_reversed) {
		script_instance->get_property_list(p_list);
	}

	if (_extension) {
		const ObjectGDExtension *current_extension = _extension;
		while (current_extension) {
			p_list->push_back(PropertyInfo(Variant::NIL, current_extension->class_name, PROPERTY_HINT_NONE, current_extension->class_name, PROPERTY_USAGE_CATEGORY));

			ClassDB::get_property_list(current_extension->class_name, p_list, true, this);

			if (current_extension->get_property_list) {
				uint32_t pcount;
				const GDExtensionPropertyInfo *pinfo = current_extension->get_property_list(_extension_instance, &pcount);
				for (uint32_t i = 0; i < pcount; i++) {
					p_list->push_back(PropertyInfo(pinfo[i]));
				}
				if (current_extension->free_property_list) {
					current_extension->free_property_list(_extension_instance, pinfo);
				}
			}

			current_extension = current_extension->parent;
		}
	}

	_get_property_listv(p_list, p_reversed);

	if (!is_class("Script")) { // can still be set, but this is for user-friendliness
		p_list->push_back(PropertyInfo(Variant::OBJECT, "script", PROPERTY_HINT_RESOURCE_TYPE, "Script", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_NEVER_DUPLICATE));
	}

	if (script_instance && !p_reversed) {
		script_instance->get_property_list(p_list);
	}

	for (const KeyValue<StringName, Variant> &K : metadata) {
		PropertyInfo pi = PropertyInfo(K.value.get_type(), "metadata/" + K.key.operator String());
		if (K.value.get_type() == Variant::OBJECT) {
			pi.hint = PROPERTY_HINT_RESOURCE_TYPE;
			pi.hint_string = "Resource";
		}
		p_list->push_back(pi);
	}
}

void Object::validate_property(PropertyInfo &p_property) const {
	_validate_propertyv(p_property);

	if (_extension && _extension->validate_property) {
		// GDExtension uses a StringName rather than a String for property name.
		StringName prop_name = p_property.name;
		GDExtensionPropertyInfo gdext_prop = {
			(GDExtensionVariantType)p_property.type,
			&prop_name,
			&p_property.class_name,
			(uint32_t)p_property.hint,
			&p_property.hint_string,
			p_property.usage,
		};
		if (_extension->validate_property(_extension_instance, &gdext_prop)) {
			p_property.type = (Variant::Type)gdext_prop.type;
			p_property.name = *reinterpret_cast<StringName *>(gdext_prop.name);
			p_property.class_name = *reinterpret_cast<StringName *>(gdext_prop.class_name);
			p_property.hint = (PropertyHint)gdext_prop.hint;
			p_property.hint_string = *reinterpret_cast<String *>(gdext_prop.hint_string);
			p_property.usage = gdext_prop.usage;
		};
	}

	if (script_instance) { // Call it last to allow user altering already validated properties.
		script_instance->validate_property(p_property);
	}
}

bool Object::property_can_revert(const StringName &p_name) const {
	if (script_instance) {
		if (script_instance->property_can_revert(p_name)) {
			return true;
		}
	}

// C style pointer casts should never trigger a compiler warning because the risk is assumed by the user, so GCC should keep quiet about it.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-qualifiers"
#endif
	if (_extension && _extension->property_can_revert) {
		if (_extension->property_can_revert(_extension_instance, (const GDExtensionStringNamePtr)&p_name)) {
			return true;
		}
	}
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

	return _property_can_revertv(p_name);
}

Variant Object::property_get_revert(const StringName &p_name) const {
	Variant ret;

	if (script_instance) {
		if (script_instance->property_get_revert(p_name, ret)) {
			return ret;
		}
	}

// C style pointer casts should never trigger a compiler warning because the risk is assumed by the user, so GCC should keep quiet about it.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-qualifiers"
#endif
	if (_extension && _extension->property_get_revert) {
		if (_extension->property_get_revert(_extension_instance, (const GDExtensionStringNamePtr)&p_name, (GDExtensionVariantPtr)&ret)) {
			return ret;
		}
	}
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

	if (_property_get_revertv(p_name, ret)) {
		return ret;
	}
	return Variant();
}

void Object::get_method_list(List<MethodInfo> *p_list) const {
	ClassDB::get_method_list(get_class_name(), p_list);
	if (script_instance) {
		script_instance->get_method_list(p_list);
	}
}

Variant Object::_call_bind(const Variant **p_args, int p_argcount, Callable::CallError &r_error) {
	if (p_argcount < 1) {
		r_error.error = Callable::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS;
		r_error.expected = 1;
		return Variant();
	}

	if (p_args[0]->get_type() != Variant::STRING_NAME && p_args[0]->get_type() != Variant::STRING) {
		r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
		r_error.argument = 0;
		r_error.expected = Variant::STRING_NAME;
		return Variant();
	}

	StringName method = *p_args[0];

	return callp(method, &p_args[1], p_argcount - 1, r_error);
}

Variant Object::_call_deferred_bind(const Variant **p_args, int p_argcount, Callable::CallError &r_error) {
	if (p_argcount < 1) {
		r_error.error = Callable::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS;
		r_error.expected = 1;
		return Variant();
	}

	if (p_args[0]->get_type() != Variant::STRING_NAME && p_args[0]->get_type() != Variant::STRING) {
		r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
		r_error.argument = 0;
		r_error.expected = Variant::STRING_NAME;
		return Variant();
	}

	r_error.error = Callable::CallError::CALL_OK;

	StringName method = *p_args[0];

	MessageQueue::get_singleton()->push_callp(get_instance_id(), method, &p_args[1], p_argcount - 1, true);

	return Variant();
}

bool Object::has_method(const StringName &p_method) const {
	if (p_method == CoreStringNames::get_singleton()->_free) {
		return true;
	}

	if (script_instance && script_instance->has_method(p_method)) {
		return true;
	}

	MethodBind *method = ClassDB::get_method(get_class_name(), p_method);
	if (method != nullptr) {
		return true;
	}

	const Script *scr = Object::cast_to<Script>(this);
	if (scr != nullptr) {
		return scr->has_static_method(p_method);
	}

	return false;
}

Variant Object::getvar(const Variant &p_key, bool *r_valid) const {
	if (r_valid) {
		*r_valid = false;
	}

	if (p_key.get_type() == Variant::STRING_NAME || p_key.get_type() == Variant::STRING) {
		return get(p_key, r_valid);
	}
	return Variant();
}

void Object::setvar(const Variant &p_key, const Variant &p_value, bool *r_valid) {
	if (r_valid) {
		*r_valid = false;
	}
	if (p_key.get_type() == Variant::STRING_NAME || p_key.get_type() == Variant::STRING) {
		return set(p_key, p_value, r_valid);
	}
}

Variant Object::callv(const StringName &p_method, const Array &p_args) {
	const Variant **argptrs = nullptr;

	if (p_args.size() > 0) {
		argptrs = (const Variant **)alloca(sizeof(Variant *) * p_args.size());
		for (int i = 0; i < p_args.size(); i++) {
			argptrs[i] = &p_args[i];
		}
	}

	Callable::CallError ce;
	Variant ret = callp(p_method, argptrs, p_args.size(), ce);
	if (ce.error != Callable::CallError::CALL_OK) {
		ERR_FAIL_V_MSG(Variant(), "Error calling method from 'callv': " + Variant::get_call_error_text(this, p_method, argptrs, p_args.size(), ce) + ".");
	}
	return ret;
}

Variant Object::callp(const StringName &p_method, const Variant **p_args, int p_argcount, Callable::CallError &r_error) {
	r_error.error = Callable::CallError::CALL_OK;

	if (p_method == CoreStringNames::get_singleton()->_free) {
//free must be here, before anything, always ready
#ifdef DEBUG_ENABLED
		if (p_argcount != 0) {
			r_error.error = Callable::CallError::CALL_ERROR_TOO_MANY_ARGUMENTS;
			r_error.expected = 0;
			return Variant();
		}
		if (Object::cast_to<RefCounted>(this)) {
			r_error.error = Callable::CallError::CALL_ERROR_INVALID_METHOD;
			ERR_FAIL_V_MSG(Variant(), "Can't 'free' a reference.");
		}

		if (_lock_index.get() > 1) {
			r_error.argument = 0;
			r_error.error = Callable::CallError::CALL_ERROR_INVALID_METHOD;
			ERR_FAIL_V_MSG(Variant(), "Object is locked and can't be freed.");
		}

#endif
		//must be here, must be before everything,
		memdelete(this);
		r_error.error = Callable::CallError::CALL_OK;
		return Variant();
	}

	Variant ret;
	OBJ_DEBUG_LOCK

	if (script_instance) {
		ret = script_instance->callp(p_method, p_args, p_argcount, r_error);
		//force jumptable
		switch (r_error.error) {
			case Callable::CallError::CALL_OK:
				return ret;
			case Callable::CallError::CALL_ERROR_INVALID_METHOD:
				break;
			case Callable::CallError::CALL_ERROR_INVALID_ARGUMENT:
			case Callable::CallError::CALL_ERROR_TOO_MANY_ARGUMENTS:
			case Callable::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS:
			case Callable::CallError::CALL_ERROR_METHOD_NOT_CONST:
				return ret;
			case Callable::CallError::CALL_ERROR_INSTANCE_IS_NULL: {
			}
		}
	}

	//extension does not need this, because all methods are registered in MethodBind

	MethodBind *method = ClassDB::get_method(get_class_name(), p_method);

	if (method) {
		ret = method->call(this, p_args, p_argcount, r_error);
	} else {
		r_error.error = Callable::CallError::CALL_ERROR_INVALID_METHOD;
	}

	return ret;
}

Variant Object::call_const(const StringName &p_method, const Variant **p_args, int p_argcount, Callable::CallError &r_error) {
	r_error.error = Callable::CallError::CALL_OK;

	if (p_method == CoreStringNames::get_singleton()->_free) {
		// Free is not const, so fail.
		r_error.error = Callable::CallError::CALL_ERROR_METHOD_NOT_CONST;
		return Variant();
	}

	Variant ret;
	OBJ_DEBUG_LOCK

	if (script_instance) {
		ret = script_instance->call_const(p_method, p_args, p_argcount, r_error);
		//force jumptable
		switch (r_error.error) {
			case Callable::CallError::CALL_OK:
				return ret;
			case Callable::CallError::CALL_ERROR_INVALID_METHOD:
				break;
			case Callable::CallError::CALL_ERROR_METHOD_NOT_CONST:
				break;
			case Callable::CallError::CALL_ERROR_INVALID_ARGUMENT:
			case Callable::CallError::CALL_ERROR_TOO_MANY_ARGUMENTS:
			case Callable::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS:
				return ret;
			case Callable::CallError::CALL_ERROR_INSTANCE_IS_NULL: {
			}
		}
	}

	//extension does not need this, because all methods are registered in MethodBind

	MethodBind *method = ClassDB::get_method(get_class_name(), p_method);

	if (method) {
		if (!method->is_const()) {
			r_error.error = Callable::CallError::CALL_ERROR_METHOD_NOT_CONST;
			return ret;
		}
		ret = method->call(this, p_args, p_argcount, r_error);
	} else {
		r_error.error = Callable::CallError::CALL_ERROR_INVALID_METHOD;
	}

	return ret;
}

void Object::notification(int p_notification, bool p_reversed) {
	if (p_reversed) {
		if (script_instance) {
			script_instance->notification(p_notification, p_reversed);
		}
	} else {
		_notificationv(p_notification, p_reversed);
	}

	if (_extension) {
		if (_extension->notification2) {
			_extension->notification2(_extension_instance, p_notification, static_cast<GDExtensionBool>(p_reversed));
#ifndef DISABLE_DEPRECATED
		} else if (_extension->notification) {
			_extension->notification(_extension_instance, p_notification);
#endif // DISABLE_DEPRECATED
		}
	}

	if (p_reversed) {
		_notificationv(p_notification, p_reversed);
	} else {
		if (script_instance) {
			script_instance->notification(p_notification, p_reversed);
		}
	}
}

String Object::to_string() {
	if (script_instance) {
		bool valid;
		String ret = script_instance->to_string(&valid);
		if (valid) {
			return ret;
		}
	}
	if (_extension && _extension->to_string) {
		String ret;
		GDExtensionBool is_valid;
		_extension->to_string(_extension_instance, &is_valid, &ret);
		return ret;
	}
	return "<" + get_class() + "#" + itos(get_instance_id()) + ">";
}

void Object::set_script_and_instance(const Variant &p_script, ScriptInstance *p_instance) {
	//this function is not meant to be used in any of these ways
	ERR_FAIL_COND(p_script.is_null());
	ERR_FAIL_NULL(p_instance);
	ERR_FAIL_COND(script_instance != nullptr || !script.is_null());

	script = p_script;
	script_instance = p_instance;
}

void Object::set_script(const Variant &p_script) {
	if (script == p_script) {
		return;
	}

	Ref<Script> s = p_script;
	if (!p_script.is_null()) {
		ERR_FAIL_COND_MSG(s.is_null(), "Cannot set object script. Parameter should be null or a reference to a valid script.");
		ERR_FAIL_COND_MSG(s->is_abstract(), vformat("Cannot set object script. Script '%s' should not be abstract.", s->get_path()));
	}

	script = p_script;

	if (script_instance) {
		memdelete(script_instance);
		script_instance = nullptr;
	}

	if (!s.is_null()) {
		if (s->can_instantiate()) {
			OBJ_DEBUG_LOCK
			script_instance = s->instance_create(this);
		} else if (Engine::get_singleton()->is_editor_hint()) {
			OBJ_DEBUG_LOCK
			script_instance = s->placeholder_instance_create(this);
		}
	}

	notify_property_list_changed(); //scripts may add variables, so refresh is desired
	emit_signal(CoreStringNames::get_singleton()->script_changed);
}

void Object::set_script_instance(ScriptInstance *p_instance) {
	if (script_instance == p_instance) {
		return;
	}

	if (script_instance) {
		memdelete(script_instance);
	}

	script_instance = p_instance;

	if (p_instance) {
		script = p_instance->get_script();
	} else {
		script = Variant();
	}
}

Variant Object::get_script() const {
	return script;
}

bool Object::has_meta(const StringName &p_name) const {
	return metadata.has(p_name);
}

void Object::set_meta(const StringName &p_name, const Variant &p_value) {
	if (p_value.get_type() == Variant::NIL) {
		if (metadata.has(p_name)) {
			metadata.erase(p_name);

			const String &sname = p_name;
			metadata_properties.erase("metadata/" + sname);
			if (!sname.begins_with("_")) {
				// Metadata starting with _ don't show up in the inspector, so no need to update.
				notify_property_list_changed();
			}
		}
		return;
	}

	HashMap<StringName, Variant>::Iterator E = metadata.find(p_name);
	if (E) {
		E->value = p_value;
	} else {
		ERR_FAIL_COND_MSG(!p_name.operator String().is_valid_identifier(), "Invalid metadata identifier: '" + p_name + "'.");
		Variant *V = &metadata.insert(p_name, p_value)->value;

		const String &sname = p_name;
		metadata_properties["metadata/" + sname] = V;
		if (!sname.begins_with("_")) {
			notify_property_list_changed();
		}
	}
}

Variant Object::get_meta(const StringName &p_name, const Variant &p_default) const {
	if (!metadata.has(p_name)) {
		if (p_default != Variant()) {
			return p_default;
		} else {
			ERR_FAIL_V_MSG(Variant(), "The object does not have any 'meta' values with the key '" + p_name + "'.");
		}
	}
	return metadata[p_name];
}

void Object::remove_meta(const StringName &p_name) {
	set_meta(p_name, Variant());
}

TypedArray<Dictionary> Object::_get_property_list_bind() const {
	List<PropertyInfo> lpi;
	get_property_list(&lpi);
	return convert_property_list(&lpi);
}

TypedArray<Dictionary> Object::_get_method_list_bind() const {
	List<MethodInfo> ml;
	get_method_list(&ml);
	TypedArray<Dictionary> ret;

	for (List<MethodInfo>::Element *E = ml.front(); E; E = E->next()) {
		Dictionary d = E->get();
		//va.push_back(d);
		ret.push_back(d);
	}

	return ret;
}

TypedArray<StringName> Object::_get_meta_list_bind() const {
	TypedArray<StringName> _metaret;

	for (const KeyValue<StringName, Variant> &K : metadata) {
		_metaret.push_back(K.key);
	}

	return _metaret;
}

void Object::get_meta_list(List<StringName> *p_list) const {
	for (const KeyValue<StringName, Variant> &K : metadata) {
		p_list->push_back(K.key);
	}
}

void Object::add_user_signal(const MethodInfo &p_signal) {
	ERR_FAIL_COND_MSG(p_signal.name.is_empty(), "Signal name cannot be empty.");
	ERR_FAIL_COND_MSG(ClassDB::has_signal(get_class_name(), p_signal.name), "User signal's name conflicts with a built-in signal of '" + get_class_name() + "'.");
	ERR_FAIL_COND_MSG(signal_map.has(p_signal.name), "Trying to add already existing signal '" + p_signal.name + "'.");
	SignalData s;
	s.user = p_signal;
	signal_map[p_signal.name] = s;
}

bool Object::_has_user_signal(const StringName &p_name) const {
	if (!signal_map.has(p_name)) {
		return false;
	}
	return signal_map[p_name].user.name.length() > 0;
}

struct _ObjectSignalDisconnectData {
	StringName signal;
	Callable callable;
};

Error Object::_emit_signal(const Variant **p_args, int p_argcount, Callable::CallError &r_error) {
	if (unlikely(p_argcount < 1)) {
		r_error.error = Callable::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS;
		r_error.expected = 1;
		ERR_FAIL_V(Error::ERR_INVALID_PARAMETER);
	}

	if (unlikely(p_args[0]->get_type() != Variant::STRING_NAME && p_args[0]->get_type() != Variant::STRING)) {
		r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
		r_error.argument = 0;
		r_error.expected = Variant::STRING_NAME;
		ERR_FAIL_V(Error::ERR_INVALID_PARAMETER);
	}

	r_error.error = Callable::CallError::CALL_OK;

	StringName signal = *p_args[0];

	const Variant **args = nullptr;

	int argc = p_argcount - 1;
	if (argc) {
		args = &p_args[1];
	}

	return emit_signalp(signal, args, argc);
}

Error Object::emit_signalp(const StringName &p_name, const Variant **p_args, int p_argcount) {
	if (_block_signals) {
		return ERR_CANT_ACQUIRE_RESOURCE; //no emit, signals blocked
	}

	SignalData *s = signal_map.getptr(p_name);
	if (!s) {
#ifdef DEBUG_ENABLED
		bool signal_is_valid = ClassDB::has_signal(get_class_name(), p_name);
		//check in script
		ERR_FAIL_COND_V_MSG(!signal_is_valid && !script.is_null() && !Ref<Script>(script)->has_script_signal(p_name), ERR_UNAVAILABLE, "Can't emit non-existing signal " + String("\"") + p_name + "\".");
#endif
		//not connected? just return
		return ERR_UNAVAILABLE;
	}

	// If this is a ref-counted object, prevent it from being destroyed during signal emission,
	// which is needed in certain edge cases; e.g., https://github.com/godotengine/godot/issues/73889.
	Ref<RefCounted> rc = Ref<RefCounted>(Object::cast_to<RefCounted>(this));

	List<_ObjectSignalDisconnectData> disconnect_data;

	// Ensure that disconnecting the signal or even deleting the object
	// will not affect the signal calling.
	LocalVector<Connection> slot_conns;
	slot_conns.resize(s->slot_map.size());
	{
		uint32_t idx = 0;
		for (const KeyValue<Callable, SignalData::Slot> &slot_kv : s->slot_map) {
			slot_conns[idx++] = slot_kv.value.conn;
		}
		DEV_ASSERT(idx == s->slot_map.size());
	}

	OBJ_DEBUG_LOCK

	Error err = OK;

	for (const Connection &c : slot_conns) {
		if (!c.callable.is_valid()) {
			// Target might have been deleted during signal callback, this is expected and OK.
			continue;
		}

		const Variant **args = p_args;
		int argc = p_argcount;

		if (c.flags & CONNECT_DEFERRED) {
			MessageQueue::get_singleton()->push_callablep(c.callable, args, argc, true);
		} else {
			Callable::CallError ce;
			_emitting = true;
			Variant ret;
			c.callable.callp(args, argc, ret, ce);
			_emitting = false;

			if (ce.error != Callable::CallError::CALL_OK) {
#ifdef DEBUG_ENABLED
				if (c.flags & CONNECT_PERSIST && Engine::get_singleton()->is_editor_hint() && (script.is_null() || !Ref<Script>(script)->is_tool())) {
					continue;
				}
#endif
				Object *target = c.callable.get_object();
				if (ce.error == Callable::CallError::CALL_ERROR_INVALID_METHOD && target && !ClassDB::class_exists(target->get_class_name())) {
					//most likely object is not initialized yet, do not throw error.
				} else {
					ERR_PRINT("Error calling from signal '" + String(p_name) + "' to callable: " + Variant::get_callable_error_text(c.callable, args, argc, ce) + ".");
					err = ERR_METHOD_NOT_FOUND;
				}
			}
		}

		bool disconnect = c.flags & CONNECT_ONE_SHOT;
#ifdef TOOLS_ENABLED
		if (disconnect && (c.flags & CONNECT_PERSIST) && Engine::get_singleton()->is_editor_hint()) {
			//this signal was connected from the editor, and is being edited. just don't disconnect for now
			disconnect = false;
		}
#endif
		if (disconnect) {
			_ObjectSignalDisconnectData dd;
			dd.signal = p_name;
			dd.callable = c.callable;
			disconnect_data.push_back(dd);
		}
	}

	while (!disconnect_data.is_empty()) {
		const _ObjectSignalDisconnectData &dd = disconnect_data.front()->get();

		_disconnect(dd.signal, dd.callable);
		disconnect_data.pop_front();
	}

	return err;
}

void Object::_add_user_signal(const String &p_name, const Array &p_args) {
	// this version of add_user_signal is meant to be used from scripts or external apis
	// without access to ADD_SIGNAL in bind_methods
	// added events are per instance, as opposed to the other ones, which are global

	MethodInfo mi;
	mi.name = p_name;

	for (int i = 0; i < p_args.size(); i++) {
		Dictionary d = p_args[i];
		PropertyInfo param;

		if (d.has("name")) {
			param.name = d["name"];
		}
		if (d.has("type")) {
			param.type = (Variant::Type)(int)d["type"];
		}

		mi.arguments.push_back(param);
	}

	add_user_signal(mi);
}

TypedArray<Dictionary> Object::_get_signal_list() const {
	List<MethodInfo> signal_list;
	get_signal_list(&signal_list);

	TypedArray<Dictionary> ret;
	for (const MethodInfo &E : signal_list) {
		ret.push_back(Dictionary(E));
	}

	return ret;
}

TypedArray<Dictionary> Object::_get_signal_connection_list(const StringName &p_signal) const {
	List<Connection> conns;
	get_all_signal_connections(&conns);

	TypedArray<Dictionary> ret;

	for (const Connection &c : conns) {
		if (c.signal.get_name() == p_signal) {
			ret.push_back(c);
		}
	}

	return ret;
}

TypedArray<Dictionary> Object::_get_incoming_connections() const {
	TypedArray<Dictionary> ret;
	int connections_amount = connections.size();
	for (int idx_conn = 0; idx_conn < connections_amount; idx_conn++) {
		ret.push_back(connections[idx_conn]);
	}

	return ret;
}

bool Object::has_signal(const StringName &p_name) const {
	if (!script.is_null()) {
		Ref<Script> scr = script;
		if (scr.is_valid() && scr->has_script_signal(p_name)) {
			return true;
		}
	}

	if (ClassDB::has_signal(get_class_name(), p_name)) {
		return true;
	}

	if (_has_user_signal(p_name)) {
		return true;
	}

	return false;
}

void Object::get_signal_list(List<MethodInfo> *p_signals) const {
	if (!script.is_null()) {
		Ref<Script> scr = script;
		if (scr.is_valid()) {
			scr->get_script_signal_list(p_signals);
		}
	}

	ClassDB::get_signal_list(get_class_name(), p_signals);
	//find maybe usersignals?

	for (const KeyValue<StringName, SignalData> &E : signal_map) {
		if (!E.value.user.name.is_empty()) {
			//user signal
			p_signals->push_back(E.value.user);
		}
	}
}

void Object::get_all_signal_connections(List<Connection> *p_connections) const {
	for (const KeyValue<StringName, SignalData> &E : signal_map) {
		const SignalData *s = &E.value;

		for (const KeyValue<Callable, SignalData::Slot> &slot_kv : s->slot_map) {
			p_connections->push_back(slot_kv.value.conn);
		}
	}
}

void Object::get_signal_connection_list(const StringName &p_signal, List<Connection> *p_connections) const {
	const SignalData *s = signal_map.getptr(p_signal);
	if (!s) {
		return; //nothing
	}

	for (const KeyValue<Callable, SignalData::Slot> &slot_kv : s->slot_map) {
		p_connections->push_back(slot_kv.value.conn);
	}
}

int Object::get_persistent_signal_connection_count() const {
	int count = 0;

	for (const KeyValue<StringName, SignalData> &E : signal_map) {
		const SignalData *s = &E.value;

		for (const KeyValue<Callable, SignalData::Slot> &slot_kv : s->slot_map) {
			if (slot_kv.value.conn.flags & CONNECT_PERSIST) {
				count += 1;
			}
		}
	}

	return count;
}

void Object::get_signals_connected_to_this(List<Connection> *p_connections) const {
	for (const Connection &E : connections) {
		p_connections->push_back(E);
	}
}

Error Object::connect(const StringName &p_signal, const Callable &p_callable, uint32_t p_flags) {
	ERR_FAIL_COND_V_MSG(p_callable.is_null(), ERR_INVALID_PARAMETER, "Cannot connect to '" + p_signal + "': the provided callable is null.");

	if (p_callable.is_standard()) {
		// FIXME: This branch should probably removed in favor of the `is_valid()` branch, but there exist some classes
		// that call `connect()` before they are fully registered with ClassDB. Until all such classes can be found
		// and registered soon enough this branch is needed to allow `connect()` to succeed.
		ERR_FAIL_NULL_V_MSG(p_callable.get_object(), ERR_INVALID_PARAMETER, "Cannot connect to '" + p_signal + "' to callable '" + p_callable + "': the callable object is null.");
	} else {
		ERR_FAIL_COND_V_MSG(!p_callable.is_valid(), ERR_INVALID_PARAMETER, "Cannot connect to '" + p_signal + "': the provided callable is not valid: " + p_callable);
	}

	SignalData *s = signal_map.getptr(p_signal);
	if (!s) {
		bool signal_is_valid = ClassDB::has_signal(get_class_name(), p_signal);
		//check in script
		if (!signal_is_valid && !script.is_null()) {
			if (Ref<Script>(script)->has_script_signal(p_signal)) {
				signal_is_valid = true;
			}
#ifdef TOOLS_ENABLED
			else {
				//allow connecting signals anyway if script is invalid, see issue #17070
				if (!Ref<Script>(script)->is_valid()) {
					signal_is_valid = true;
				}
			}
#endif
		}

		ERR_FAIL_COND_V_MSG(!signal_is_valid, ERR_INVALID_PARAMETER, "In Object of type '" + String(get_class()) + "': Attempt to connect nonexistent signal '" + p_signal + "' to callable '" + p_callable + "'.");

		signal_map[p_signal] = SignalData();
		s = &signal_map[p_signal];
	}

	Callable target = p_callable;

	//compare with the base callable, so binds can be ignored
	if (s->slot_map.has(*target.get_base_comparator())) {
		if (p_flags & CONNECT_REFERENCE_COUNTED) {
			s->slot_map[*target.get_base_comparator()].reference_count++;
			return OK;
		} else {
			ERR_FAIL_V_MSG(ERR_INVALID_PARAMETER, "Signal '" + p_signal + "' is already connected to given callable '" + p_callable + "' in that object.");
		}
	}

	Object *target_object = p_callable.get_object();

	SignalData::Slot slot;

	Connection conn;
	conn.callable = target;
	conn.signal = ::Signal(this, p_signal);
	conn.flags = p_flags;
	slot.conn = conn;
	if (target_object) {
		slot.cE = target_object->connections.push_back(conn);
	}
	if (p_flags & CONNECT_REFERENCE_COUNTED) {
		slot.reference_count = 1;
	}

	//use callable version as key, so binds can be ignored
	s->slot_map[*target.get_base_comparator()] = slot;

	return OK;
}

bool Object::is_connected(const StringName &p_signal, const Callable &p_callable) const {
	ERR_FAIL_COND_V_MSG(p_callable.is_null(), false, "Cannot determine if connected to '" + p_signal + "': the provided callable is null.");
	const SignalData *s = signal_map.getptr(p_signal);
	if (!s) {
		bool signal_is_valid = ClassDB::has_signal(get_class_name(), p_signal);
		if (signal_is_valid) {
			return false;
		}

		if (!script.is_null() && Ref<Script>(script)->has_script_signal(p_signal)) {
			return false;
		}

		ERR_FAIL_V_MSG(false, "Nonexistent signal: " + p_signal + ".");
	}

	Callable target = p_callable;

	return s->slot_map.has(*target.get_base_comparator());
}

void Object::disconnect(const StringName &p_signal, const Callable &p_callable) {
	_disconnect(p_signal, p_callable);
}

bool Object::_disconnect(const StringName &p_signal, const Callable &p_callable, bool p_force) {
	ERR_FAIL_COND_V_MSG(p_callable.is_null(), false, "Cannot disconnect from '" + p_signal + "': the provided callable is null.");

	SignalData *s = signal_map.getptr(p_signal);
	if (!s) {
		bool signal_is_valid = ClassDB::has_signal(get_class_name(), p_signal) ||
				(!script.is_null() && Ref<Script>(script)->has_script_signal(p_signal));
		ERR_FAIL_COND_V_MSG(signal_is_valid, false, "Attempt to disconnect a nonexistent connection from '" + to_string() + "'. Signal: '" + p_signal + "', callable: '" + p_callable + "'.");
	}
	ERR_FAIL_NULL_V_MSG(s, false, vformat("Disconnecting nonexistent signal '%s' in %s.", p_signal, to_string()));

	ERR_FAIL_COND_V_MSG(!s->slot_map.has(*p_callable.get_base_comparator()), false, "Attempt to disconnect a nonexistent connection from '" + to_string() + "'. Signal: '" + p_signal + "', callable: '" + p_callable + "'.");

	SignalData::Slot *slot = &s->slot_map[*p_callable.get_base_comparator()];

	if (!p_force) {
		slot->reference_count--; // by default is zero, if it was not referenced it will go below it
		if (slot->reference_count > 0) {
			return false;
		}
	}

	if (slot->cE) {
		Object *target_object = p_callable.get_object();
		if (target_object) {
			target_object->connections.erase(slot->cE);
		}
	}

	s->slot_map.erase(*p_callable.get_base_comparator());

	if (s->slot_map.is_empty() && ClassDB::has_signal(get_class_name(), p_signal)) {
		//not user signal, delete
		signal_map.erase(p_signal);
	}

	return true;
}

void Object::_set_bind(const StringName &p_set, const Variant &p_value) {
	set(p_set, p_value);
}

Variant Object::_get_bind(const StringName &p_name) const {
	return get(p_name);
}

void Object::_set_indexed_bind(const NodePath &p_name, const Variant &p_value) {
	set_indexed(p_name.get_as_property_path().get_subnames(), p_value);
}

Variant Object::_get_indexed_bind(const NodePath &p_name) const {
	return get_indexed(p_name.get_as_property_path().get_subnames());
}

void Object::initialize_class() {
	static bool initialized = false;
	if (initialized) {
		return;
	}
	ClassDB::_add_class<Object>();
	_bind_methods();
	initialized = true;
}

String Object::tr(const StringName &p_message, const StringName &p_context) const {
	if (!_can_translate || !TranslationServer::get_singleton()) {
		return p_message;
	}

	if (Engine::get_singleton()->is_editor_hint()) {
		return TranslationServer::get_singleton()->tool_translate(p_message, p_context);
	} else {
		return TranslationServer::get_singleton()->translate(p_message, p_context);
	}
}

String Object::tr_n(const StringName &p_message, const StringName &p_message_plural, int p_n, const StringName &p_context) const {
	if (!_can_translate || !TranslationServer::get_singleton()) {
		// Return message based on English plural rule if translation is not possible.
		if (p_n == 1) {
			return p_message;
		}
		return p_message_plural;
	}

	if (Engine::get_singleton()->is_editor_hint()) {
		return TranslationServer::get_singleton()->tool_translate_plural(p_message, p_message_plural, p_n, p_context);
	} else {
		return TranslationServer::get_singleton()->translate_plural(p_message, p_message_plural, p_n, p_context);
	}
}

void Object::_clear_internal_resource_paths(const Variant &p_var) {
	switch (p_var.get_type()) {
		case Variant::OBJECT: {
			Ref<Resource> r = p_var;
			if (!r.is_valid()) {
				return;
			}

			if (!r->is_built_in()) {
				return; //not an internal resource
			}

			Object *object = p_var;
			if (!object) {
				return;
			}

			r->set_path("");
			r->clear_internal_resource_paths();
		} break;
		case Variant::ARRAY: {
			Array a = p_var;
			for (int i = 0; i < a.size(); i++) {
				_clear_internal_resource_paths(a[i]);
			}

		} break;
		case Variant::DICTIONARY: {
			Dictionary d = p_var;
			List<Variant> keys;
			d.get_key_list(&keys);

			for (const Variant &E : keys) {
				_clear_internal_resource_paths(E);
				_clear_internal_resource_paths(d[E]);
			}
		} break;
		default: {
		}
	}
}

#ifdef TOOLS_ENABLED
void Object::editor_set_section_unfold(const String &p_section, bool p_unfolded) {
	set_edited(true);
	if (p_unfolded) {
		editor_section_folding.insert(p_section);
	} else {
		editor_section_folding.erase(p_section);
	}
}

bool Object::editor_is_section_unfolded(const String &p_section) {
	return editor_section_folding.has(p_section);
}

#endif

void Object::clear_internal_resource_paths() {
	List<PropertyInfo> pinfo;

	get_property_list(&pinfo);

	for (const PropertyInfo &E : pinfo) {
		_clear_internal_resource_paths(get(E.name));
	}
}

void Object::notify_property_list_changed() {
	emit_signal(CoreStringNames::get_singleton()->property_list_changed);
}

void Object::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_class"), &Object::get_class);
	ClassDB::bind_method(D_METHOD("is_class", "class"), &Object::is_class);
	ClassDB::bind_method(D_METHOD("set", "property", "value"), &Object::_set_bind);
	ClassDB::bind_method(D_METHOD("get", "property"), &Object::_get_bind);
	ClassDB::bind_method(D_METHOD("set_indexed", "property_path", "value"), &Object::_set_indexed_bind);
	ClassDB::bind_method(D_METHOD("get_indexed", "property_path"), &Object::_get_indexed_bind);
	ClassDB::bind_method(D_METHOD("get_property_list"), &Object::_get_property_list_bind);
	ClassDB::bind_method(D_METHOD("get_method_list"), &Object::_get_method_list_bind);
	ClassDB::bind_method(D_METHOD("property_can_revert", "property"), &Object::property_can_revert);
	ClassDB::bind_method(D_METHOD("property_get_revert", "property"), &Object::property_get_revert);
	ClassDB::bind_method(D_METHOD("notification", "what", "reversed"), &Object::notification, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("to_string"), &Object::to_string);
	ClassDB::bind_method(D_METHOD("get_instance_id"), &Object::get_instance_id);

	ClassDB::bind_method(D_METHOD("set_script", "script"), &Object::set_script);
	ClassDB::bind_method(D_METHOD("get_script"), &Object::get_script);

	ClassDB::bind_method(D_METHOD("set_meta", "name", "value"), &Object::set_meta);
	ClassDB::bind_method(D_METHOD("remove_meta", "name"), &Object::remove_meta);
	ClassDB::bind_method(D_METHOD("get_meta", "name", "default"), &Object::get_meta, DEFVAL(Variant()));
	ClassDB::bind_method(D_METHOD("has_meta", "name"), &Object::has_meta);
	ClassDB::bind_method(D_METHOD("get_meta_list"), &Object::_get_meta_list_bind);

	ClassDB::bind_method(D_METHOD("add_user_signal", "signal", "arguments"), &Object::_add_user_signal, DEFVAL(Array()));
	ClassDB::bind_method(D_METHOD("has_user_signal", "signal"), &Object::_has_user_signal);

	{
		MethodInfo mi;
		mi.name = "emit_signal";
		mi.arguments.push_back(PropertyInfo(Variant::STRING_NAME, "signal"));

		ClassDB::bind_vararg_method(METHOD_FLAGS_DEFAULT, "emit_signal", &Object::_emit_signal, mi, varray(), false);
	}

	{
		MethodInfo mi;
		mi.name = "call";
		mi.arguments.push_back(PropertyInfo(Variant::STRING_NAME, "method"));

		ClassDB::bind_vararg_method(METHOD_FLAGS_DEFAULT, "call", &Object::_call_bind, mi);
	}

	{
		MethodInfo mi;
		mi.name = "call_deferred";
		mi.arguments.push_back(PropertyInfo(Variant::STRING_NAME, "method"));

		ClassDB::bind_vararg_method(METHOD_FLAGS_DEFAULT, "call_deferred", &Object::_call_deferred_bind, mi, varray(), false);
	}

	ClassDB::bind_method(D_METHOD("set_deferred", "property", "value"), &Object::set_deferred);

	ClassDB::bind_method(D_METHOD("callv", "method", "arg_array"), &Object::callv);

	ClassDB::bind_method(D_METHOD("has_method", "method"), &Object::has_method);

	ClassDB::bind_method(D_METHOD("has_signal", "signal"), &Object::has_signal);
	ClassDB::bind_method(D_METHOD("get_signal_list"), &Object::_get_signal_list);
	ClassDB::bind_method(D_METHOD("get_signal_connection_list", "signal"), &Object::_get_signal_connection_list);
	ClassDB::bind_method(D_METHOD("get_incoming_connections"), &Object::_get_incoming_connections);

	ClassDB::bind_method(D_METHOD("connect", "signal", "callable", "flags"), &Object::connect, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("disconnect", "signal", "callable"), &Object::disconnect);
	ClassDB::bind_method(D_METHOD("is_connected", "signal", "callable"), &Object::is_connected);

	ClassDB::bind_method(D_METHOD("set_block_signals", "enable"), &Object::set_block_signals);
	ClassDB::bind_method(D_METHOD("is_blocking_signals"), &Object::is_blocking_signals);
	ClassDB::bind_method(D_METHOD("notify_property_list_changed"), &Object::notify_property_list_changed);

	ClassDB::bind_method(D_METHOD("set_message_translation", "enable"), &Object::set_message_translation);
	ClassDB::bind_method(D_METHOD("can_translate_messages"), &Object::can_translate_messages);
	ClassDB::bind_method(D_METHOD("tr", "message", "context"), &Object::tr, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("tr_n", "message", "plural_message", "n", "context"), &Object::tr_n, DEFVAL(""));

	ClassDB::bind_method(D_METHOD("is_queued_for_deletion"), &Object::is_queued_for_deletion);
	ClassDB::bind_method(D_METHOD("cancel_free"), &Object::cancel_free);

	ClassDB::add_virtual_method("Object", MethodInfo("free"), false);

	ADD_SIGNAL(MethodInfo("script_changed"));
	ADD_SIGNAL(MethodInfo("property_list_changed"));

#define BIND_OBJ_CORE_METHOD(m_method) \
	::ClassDB::add_virtual_method(get_class_static(), m_method, true, Vector<String>(), true);

	MethodInfo notification_mi("_notification", PropertyInfo(Variant::INT, "what"));
	notification_mi.arguments_metadata.push_back(GodotTypeInfo::Metadata::METADATA_INT_IS_INT32);
	BIND_OBJ_CORE_METHOD(notification_mi);
	BIND_OBJ_CORE_METHOD(MethodInfo(Variant::BOOL, "_set", PropertyInfo(Variant::STRING_NAME, "property"), PropertyInfo(Variant::NIL, "value")));
#ifdef TOOLS_ENABLED
	MethodInfo miget("_get", PropertyInfo(Variant::STRING_NAME, "property"));
	miget.return_val.name = "Variant";
	miget.return_val.usage |= PROPERTY_USAGE_NIL_IS_VARIANT;
	BIND_OBJ_CORE_METHOD(miget);

	MethodInfo plget("_get_property_list");
	plget.return_val.type = Variant::ARRAY;
	plget.return_val.hint = PROPERTY_HINT_ARRAY_TYPE;
	plget.return_val.hint_string = "Dictionary";
	BIND_OBJ_CORE_METHOD(plget);

	BIND_OBJ_CORE_METHOD(MethodInfo(Variant::NIL, "_validate_property", PropertyInfo(Variant::DICTIONARY, "property")));

	BIND_OBJ_CORE_METHOD(MethodInfo(Variant::BOOL, "_property_can_revert", PropertyInfo(Variant::STRING_NAME, "property")));
	MethodInfo mipgr("_property_get_revert", PropertyInfo(Variant::STRING_NAME, "property"));
	mipgr.return_val.name = "Variant";
	mipgr.return_val.usage |= PROPERTY_USAGE_NIL_IS_VARIANT;
	BIND_OBJ_CORE_METHOD(mipgr);

#endif
	BIND_OBJ_CORE_METHOD(MethodInfo("_init"));
	BIND_OBJ_CORE_METHOD(MethodInfo(Variant::STRING, "_to_string"));

	BIND_CONSTANT(NOTIFICATION_POSTINITIALIZE);
	BIND_CONSTANT(NOTIFICATION_PREDELETE);

	BIND_ENUM_CONSTANT(CONNECT_DEFERRED);
	BIND_ENUM_CONSTANT(CONNECT_PERSIST);
	BIND_ENUM_CONSTANT(CONNECT_ONE_SHOT);
	BIND_ENUM_CONSTANT(CONNECT_REFERENCE_COUNTED);
}

void Object::set_deferred(const StringName &p_property, const Variant &p_value) {
	MessageQueue::get_singleton()->push_set(this, p_property, p_value);
}

void Object::set_block_signals(bool p_block) {
	_block_signals = p_block;
}

bool Object::is_blocking_signals() const {
	return _block_signals;
}

Variant::Type Object::get_static_property_type(const StringName &p_property, bool *r_valid) const {
	bool valid;
	Variant::Type t = ClassDB::get_property_type(get_class_name(), p_property, &valid);
	if (valid) {
		if (r_valid) {
			*r_valid = true;
		}
		return t;
	}

	if (get_script_instance()) {
		return get_script_instance()->get_property_type(p_property, r_valid);
	}
	if (r_valid) {
		*r_valid = false;
	}

	return Variant::NIL;
}

Variant::Type Object::get_static_property_type_indexed(const Vector<StringName> &p_path, bool *r_valid) const {
	if (p_path.size() == 0) {
		if (r_valid) {
			*r_valid = false;
		}

		return Variant::NIL;
	}

	bool valid = false;
	Variant::Type t = get_static_property_type(p_path[0], &valid);
	if (!valid) {
		if (r_valid) {
			*r_valid = false;
		}

		return Variant::NIL;
	}

	Callable::CallError ce;
	Variant check;
	Variant::construct(t, check, nullptr, 0, ce);

	for (int i = 1; i < p_path.size(); i++) {
		if (check.get_type() == Variant::OBJECT || check.get_type() == Variant::DICTIONARY || check.get_type() == Variant::ARRAY) {
			// We cannot be sure about the type of properties this type can have
			if (r_valid) {
				*r_valid = false;
			}
			return Variant::NIL;
		}

		check = check.get_named(p_path[i], valid);

		if (!valid) {
			if (r_valid) {
				*r_valid = false;
			}
			return Variant::NIL;
		}
	}

	if (r_valid) {
		*r_valid = true;
	}

	return check.get_type();
}

bool Object::is_queued_for_deletion() const {
	return _is_queued_for_deletion;
}

#ifdef TOOLS_ENABLED
void Object::set_edited(bool p_edited) {
	_edited = p_edited;
	_edited_version++;
}

bool Object::is_edited() const {
	return _edited;
}

uint32_t Object::get_edited_version() const {
	return _edited_version;
}
#endif

StringName Object::get_class_name_for_extension(const GDExtension *p_library) const {
	// Only return the class name per the extension if it matches the given p_library.
	if (_extension && _extension->library == p_library) {
		return _extension->class_name;
	}

	// Extensions only have wrapper classes for classes exposed in ClassDB.
	const StringName *class_name = _get_class_namev();
	if (ClassDB::is_class_exposed(*class_name)) {
		return *class_name;
	}

	// Find the nearest parent class that's exposed.
	StringName parent_class = ClassDB::get_parent_class(*class_name);
	while (parent_class != StringName()) {
		if (ClassDB::is_class_exposed(parent_class)) {
			return parent_class;
		}
		parent_class = ClassDB::get_parent_class(parent_class);
	}

	return SNAME("Object");
}

void Object::set_instance_binding(void *p_token, void *p_binding, const GDExtensionInstanceBindingCallbacks *p_callbacks) {
	// This is only meant to be used on creation by the binder, but we also
	// need to account for reloading (where the 'binding' will be cleared).
	ERR_FAIL_COND(_instance_bindings != nullptr && _instance_bindings[0].binding != nullptr);
	if (_instance_bindings == nullptr) {
		_instance_bindings = (InstanceBinding *)memalloc(sizeof(InstanceBinding));
		_instance_binding_count = 1;
	}
	_instance_bindings[0].binding = p_binding;
	_instance_bindings[0].free_callback = p_callbacks->free_callback;
	_instance_bindings[0].reference_callback = p_callbacks->reference_callback;
	_instance_bindings[0].token = p_token;
}

void *Object::get_instance_binding(void *p_token, const GDExtensionInstanceBindingCallbacks *p_callbacks) {
	void *binding = nullptr;
	_instance_binding_mutex.lock();
	for (uint32_t i = 0; i < _instance_binding_count; i++) {
		if (_instance_bindings[i].token == p_token) {
			binding = _instance_bindings[i].binding;
			break;
		}
	}
	if (unlikely(!binding && p_callbacks)) {
		uint32_t current_size = next_power_of_2(_instance_binding_count);
		uint32_t new_size = next_power_of_2(_instance_binding_count + 1);

		if (current_size == 0 || new_size > current_size) {
			_instance_bindings = (InstanceBinding *)memrealloc(_instance_bindings, new_size * sizeof(InstanceBinding));
		}

		_instance_bindings[_instance_binding_count].free_callback = p_callbacks->free_callback;
		_instance_bindings[_instance_binding_count].reference_callback = p_callbacks->reference_callback;
		_instance_bindings[_instance_binding_count].token = p_token;

		binding = p_callbacks->create_callback(p_token, this);
		_instance_bindings[_instance_binding_count].binding = binding;

#ifdef TOOLS_ENABLED
		if (!_extension && Engine::get_singleton()->is_extension_reloading_enabled()) {
			GDExtensionManager::get_singleton()->track_instance_binding(p_token, this);
		}
#endif

		_instance_binding_count++;
	}

	_instance_binding_mutex.unlock();

	return binding;
}

bool Object::has_instance_binding(void *p_token) {
	bool found = false;
	_instance_binding_mutex.lock();
	for (uint32_t i = 0; i < _instance_binding_count; i++) {
		if (_instance_bindings[i].token == p_token) {
			found = true;
			break;
		}
	}

	_instance_binding_mutex.unlock();

	return found;
}

void Object::free_instance_binding(void *p_token) {
	bool found = false;
	_instance_binding_mutex.lock();
	for (uint32_t i = 0; i < _instance_binding_count; i++) {
		if (!found && _instance_bindings[i].token == p_token) {
			if (_instance_bindings[i].free_callback) {
				_instance_bindings[i].free_callback(_instance_bindings[i].token, this, _instance_bindings[i].binding);
			}
			found = true;
		}
		if (found) {
			if (i + 1 < _instance_binding_count) {
				_instance_bindings[i] = _instance_bindings[i + 1];
			} else {
				_instance_bindings[i] = { nullptr };
			}
		}
	}
	if (found) {
		_instance_binding_count--;
	}
	_instance_binding_mutex.unlock();
}

#ifdef TOOLS_ENABLED
void Object::clear_internal_extension() {
	ERR_FAIL_NULL(_extension);

	// Free the instance inside the GDExtension.
	if (_extension->free_instance) {
		_extension->free_instance(_extension->class_userdata, _extension_instance);
	}
	_extension = nullptr;
	_extension_instance = nullptr;

	// Clear the instance bindings.
	_instance_binding_mutex.lock();
	if (_instance_bindings[0].free_callback) {
		_instance_bindings[0].free_callback(_instance_bindings[0].token, this, _instance_bindings[0].binding);
	}
	_instance_bindings[0].binding = nullptr;
	_instance_bindings[0].token = nullptr;
	_instance_bindings[0].free_callback = nullptr;
	_instance_bindings[0].reference_callback = nullptr;
	_instance_binding_mutex.unlock();

	// Clear the virtual methods.
	while (virtual_method_list) {
		(*virtual_method_list->method) = nullptr;
		(*virtual_method_list->initialized) = false;
		virtual_method_list = virtual_method_list->next;
	}
}

void Object::reset_internal_extension(ObjectGDExtension *p_extension) {
	ERR_FAIL_COND(_extension != nullptr);

	if (p_extension) {
		_extension_instance = p_extension->recreate_instance ? p_extension->recreate_instance(p_extension->class_userdata, (GDExtensionObjectPtr)this) : nullptr;
		ERR_FAIL_NULL_MSG(_extension_instance, "Unable to recreate GDExtension instance - does this extension support hot reloading?");
		_extension = p_extension;
	}
}
#endif

void Object::_construct_object(bool p_reference) {
	type_is_reference = p_reference;
	_instance_id = ObjectDB::add_instance(this);

#ifdef DEBUG_ENABLED
	_lock_index.init(1);
#endif
}

Object::Object(bool p_reference) {
	_construct_object(p_reference);
}

Object::Object() {
	_construct_object(false);
}

void Object::detach_from_objectdb() {
	if (_instance_id != ObjectID()) {
		ObjectDB::remove_instance(this);
		_instance_id = ObjectID();
	}
}

Object::~Object() {
	if (script_instance) {
		memdelete(script_instance);
	}
	script_instance = nullptr;

	if (_extension) {
#ifdef TOOLS_ENABLED
		if (_extension->untrack_instance) {
			_extension->untrack_instance(_extension->tracking_userdata, this);
		}
#endif
		if (_extension->free_instance) {
			_extension->free_instance(_extension->class_userdata, _extension_instance);
		}
		_extension = nullptr;
		_extension_instance = nullptr;
	}
#ifdef TOOLS_ENABLED
	else if (_instance_bindings != nullptr && Engine::get_singleton()->is_extension_reloading_enabled()) {
		for (uint32_t i = 0; i < _instance_binding_count; i++) {
			GDExtensionManager::get_singleton()->untrack_instance_binding(_instance_bindings[i].token, this);
		}
	}
#endif

	if (_emitting) {
		//@todo this may need to actually reach the debugger prioritarily somehow because it may crash before
		ERR_PRINT("Object " + to_string() + " was freed or unreferenced while a signal is being emitted from it. Try connecting to the signal using 'CONNECT_DEFERRED' flag, or use queue_free() to free the object (if this object is a Node) to avoid this error and potential crashes.");
	}

	// Drop all connections to the signals of this object.
	while (signal_map.size()) {
		// Avoid regular iteration so erasing is safe.
		KeyValue<StringName, SignalData> &E = *signal_map.begin();
		SignalData *s = &E.value;

		for (const KeyValue<Callable, SignalData::Slot> &slot_kv : s->slot_map) {
			Object *target = slot_kv.value.conn.callable.get_object();
			if (likely(target)) {
				target->connections.erase(slot_kv.value.cE);
			}
		}

		signal_map.erase(E.key);
	}

	// Disconnect signals that connect to this object.
	while (connections.size()) {
		Connection c = connections.front()->get();
		bool disconnected = c.signal.get_object()->_disconnect(c.signal.get_name(), c.callable, true);
		if (unlikely(!disconnected)) {
			// If the disconnect has failed, abandon the connection to avoid getting trapped in an infinite loop here.
			connections.pop_front();
		}
	}

	if (_instance_id != ObjectID()) {
		ObjectDB::remove_instance(this);
		_instance_id = ObjectID();
	}
	_predelete_ok = 2;

	if (_instance_bindings != nullptr) {
		for (uint32_t i = 0; i < _instance_binding_count; i++) {
			if (_instance_bindings[i].free_callback) {
				_instance_bindings[i].free_callback(_instance_bindings[i].token, this, _instance_bindings[i].binding);
			}
		}
		memfree(_instance_bindings);
	}
}

bool predelete_handler(Object *p_object) {
	return p_object->_predelete();
}

void postinitialize_handler(Object *p_object) {
	p_object->_postinitialize();
}

void ObjectDB::debug_objects(DebugFunc p_func) {
	spin_lock.lock();

	for (uint32_t i = 0, count = slot_count; i < slot_max && count != 0; i++) {
		if (object_slots[i].validator) {
			p_func(object_slots[i].object);
			count--;
		}
	}
	spin_lock.unlock();
}

void Object::get_argument_options(const StringName &p_function, int p_idx, List<String> *r_options) const {
	if (p_idx == 0) {
		if (p_function == "connect" || p_function == "is_connected" || p_function == "disconnect" || p_function == "emit_signal" || p_function == "has_signal") {
			List<MethodInfo> signals;
			get_signal_list(&signals);
			for (const MethodInfo &E : signals) {
				r_options->push_back(E.name.quote());
			}
		} else if (p_function == "call" || p_function == "call_deferred" || p_function == "callv" || p_function == "has_method") {
			List<MethodInfo> methods;
			get_method_list(&methods);
			for (const MethodInfo &E : methods) {
				if (E.name.begins_with("_") && !(E.flags & METHOD_FLAG_VIRTUAL)) {
					continue;
				}
				r_options->push_back(E.name.quote());
			}
		} else if (p_function == "set" || p_function == "set_deferred" || p_function == "get") {
			List<PropertyInfo> properties;
			get_property_list(&properties);
			for (const PropertyInfo &E : properties) {
				if (E.usage & PROPERTY_USAGE_DEFAULT && !(E.usage & PROPERTY_USAGE_INTERNAL)) {
					r_options->push_back(E.name.quote());
				}
			}
		} else if (p_function == "set_meta" || p_function == "get_meta" || p_function == "has_meta" || p_function == "remove_meta") {
			for (const KeyValue<StringName, Variant> &K : metadata) {
				r_options->push_back(String(K.key).quote());
			}
		}
	} else if (p_idx == 2) {
		if (p_function == "connect") {
			// Ideally, the constants should be inferred by the parameter.
			// But a parameter's PropertyInfo does not store the enum they come from, so this will do for now.
			List<StringName> constants;
			ClassDB::get_enum_constants("Object", "ConnectFlags", &constants);
			for (const StringName &E : constants) {
				r_options->push_back(String(E));
			}
		}
	}
}

SpinLock ObjectDB::spin_lock;
uint32_t ObjectDB::slot_count = 0;
uint32_t ObjectDB::slot_max = 0;
ObjectDB::ObjectSlot *ObjectDB::object_slots = nullptr;
uint64_t ObjectDB::validator_counter = 0;

int ObjectDB::get_object_count() {
	return slot_count;
}

ObjectID ObjectDB::add_instance(Object *p_object) {
	spin_lock.lock();
	if (unlikely(slot_count == slot_max)) {
		CRASH_COND(slot_count == (1 << OBJECTDB_SLOT_MAX_COUNT_BITS));

		uint32_t new_slot_max = slot_max > 0 ? slot_max * 2 : 1;
		object_slots = (ObjectSlot *)memrealloc(object_slots, sizeof(ObjectSlot) * new_slot_max);
		for (uint32_t i = slot_max; i < new_slot_max; i++) {
			object_slots[i].object = nullptr;
			object_slots[i].is_ref_counted = false;
			object_slots[i].next_free = i;
			object_slots[i].validator = 0;
		}
		slot_max = new_slot_max;
	}

	uint32_t slot = object_slots[slot_count].next_free;
	if (object_slots[slot].object != nullptr) {
		spin_lock.unlock();
		ERR_FAIL_COND_V(object_slots[slot].object != nullptr, ObjectID());
	}
	object_slots[slot].object = p_object;
	object_slots[slot].is_ref_counted = p_object->is_ref_counted();
	validator_counter = (validator_counter + 1) & OBJECTDB_VALIDATOR_MASK;
	if (unlikely(validator_counter == 0)) {
		validator_counter = 1;
	}
	object_slots[slot].validator = validator_counter;

	uint64_t id = validator_counter;
	id <<= OBJECTDB_SLOT_MAX_COUNT_BITS;
	id |= uint64_t(slot);

	if (p_object->is_ref_counted()) {
		id |= OBJECTDB_REFERENCE_BIT;
	}

	slot_count++;

	spin_lock.unlock();

	return ObjectID(id);
}

void ObjectDB::remove_instance(Object *p_object) {
	uint64_t t = p_object->get_instance_id();
	uint32_t slot = t & OBJECTDB_SLOT_MAX_COUNT_MASK; //slot is always valid on valid object

	spin_lock.lock();

#ifdef DEBUG_ENABLED

	if (object_slots[slot].object != p_object) {
		spin_lock.unlock();
		ERR_FAIL_COND(object_slots[slot].object != p_object);
	}
	{
		uint64_t validator = (t >> OBJECTDB_SLOT_MAX_COUNT_BITS) & OBJECTDB_VALIDATOR_MASK;
		if (object_slots[slot].validator != validator) {
			spin_lock.unlock();
			ERR_FAIL_COND(object_slots[slot].validator != validator);
		}
	}

#endif
	//decrease slot count
	slot_count--;
	//set the free slot properly
	object_slots[slot_count].next_free = slot;
	//invalidate, so checks against it fail
	object_slots[slot].validator = 0;
	object_slots[slot].is_ref_counted = false;
	object_slots[slot].object = nullptr;

	spin_lock.unlock();
}

void ObjectDB::setup() {
	//nothing to do now
}

void ObjectDB::cleanup() {
	if (slot_count > 0) {
		spin_lock.lock();

		WARN_PRINT("ObjectDB instances leaked at exit (run with --verbose for details).");
		if (OS::get_singleton()->is_stdout_verbose()) {
			// Ensure calling the native classes because if a leaked instance has a script
			// that overrides any of those methods, it'd not be OK to call them at this point,
			// now the scripting languages have already been terminated.
			MethodBind *node_get_name = ClassDB::get_method("Node", "get_name");
			MethodBind *resource_get_path = ClassDB::get_method("Resource", "get_path");
			Callable::CallError call_error;

			for (uint32_t i = 0, count = slot_count; i < slot_max && count != 0; i++) {
				if (object_slots[i].validator) {
					Object *obj = object_slots[i].object;

					String extra_info;
					if (obj->is_class("Node")) {
						extra_info = " - Node name: " + String(node_get_name->call(obj, nullptr, 0, call_error));
					}
					if (obj->is_class("Resource")) {
						extra_info = " - Resource path: " + String(resource_get_path->call(obj, nullptr, 0, call_error));
					}

					uint64_t id = uint64_t(i) | (uint64_t(object_slots[i].validator) << OBJECTDB_VALIDATOR_BITS) | (object_slots[i].is_ref_counted ? OBJECTDB_REFERENCE_BIT : 0);
					print_line("Leaked instance: " + String(obj->get_class()) + ":" + itos(id) + extra_info);

					count--;
				}
			}
			print_line("Hint: Leaked instances typically happen when nodes are removed from the scene tree (with `remove_child()`) but not freed (with `free()` or `queue_free()`).");
		}
		spin_lock.unlock();
	}

	if (object_slots) {
		memfree(object_slots);
	}
}
