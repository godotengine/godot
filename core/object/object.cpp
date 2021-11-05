/*************************************************************************/
/*  object.cpp                                                           */
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

#include "object.h"

#include "core/core_string_names.h"
#include "core/io/resource.h"
#include "core/object/class_db.h"
#include "core/object/message_queue.h"
#include "core/object/script_language.h"
#include "core/os/os.h"
#include "core/string/print_string.h"
#include "core/string/translation.h"

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

Array convert_property_list(const List<PropertyInfo> *p_list) {
	Array va;
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

MethodInfo::MethodInfo() :
		flags(METHOD_FLAG_NORMAL) {}

MethodInfo::MethodInfo(const String &p_name) :
		name(p_name),
		flags(METHOD_FLAG_NORMAL) {
}

MethodInfo::MethodInfo(const String &p_name, const PropertyInfo &p_param1) :
		name(p_name),
		flags(METHOD_FLAG_NORMAL) {
	arguments.push_back(p_param1);
}

MethodInfo::MethodInfo(const String &p_name, const PropertyInfo &p_param1, const PropertyInfo &p_param2) :
		name(p_name),
		flags(METHOD_FLAG_NORMAL) {
	arguments.push_back(p_param1);
	arguments.push_back(p_param2);
}

MethodInfo::MethodInfo(const String &p_name, const PropertyInfo &p_param1, const PropertyInfo &p_param2, const PropertyInfo &p_param3) :
		name(p_name),
		flags(METHOD_FLAG_NORMAL) {
	arguments.push_back(p_param1);
	arguments.push_back(p_param2);
	arguments.push_back(p_param3);
}

MethodInfo::MethodInfo(const String &p_name, const PropertyInfo &p_param1, const PropertyInfo &p_param2, const PropertyInfo &p_param3, const PropertyInfo &p_param4) :
		name(p_name),
		flags(METHOD_FLAG_NORMAL) {
	arguments.push_back(p_param1);
	arguments.push_back(p_param2);
	arguments.push_back(p_param3);
	arguments.push_back(p_param4);
}

MethodInfo::MethodInfo(const String &p_name, const PropertyInfo &p_param1, const PropertyInfo &p_param2, const PropertyInfo &p_param3, const PropertyInfo &p_param4, const PropertyInfo &p_param5) :
		name(p_name),
		flags(METHOD_FLAG_NORMAL) {
	arguments.push_back(p_param1);
	arguments.push_back(p_param2);
	arguments.push_back(p_param3);
	arguments.push_back(p_param4);
	arguments.push_back(p_param5);
}

MethodInfo::MethodInfo(Variant::Type ret) :
		flags(METHOD_FLAG_NORMAL) {
	return_val.type = ret;
}

MethodInfo::MethodInfo(Variant::Type ret, const String &p_name) :
		name(p_name),
		flags(METHOD_FLAG_NORMAL) {
	return_val.type = ret;
}

MethodInfo::MethodInfo(Variant::Type ret, const String &p_name, const PropertyInfo &p_param1) :
		name(p_name),
		flags(METHOD_FLAG_NORMAL) {
	return_val.type = ret;
	arguments.push_back(p_param1);
}

MethodInfo::MethodInfo(Variant::Type ret, const String &p_name, const PropertyInfo &p_param1, const PropertyInfo &p_param2) :
		name(p_name),
		flags(METHOD_FLAG_NORMAL) {
	return_val.type = ret;
	arguments.push_back(p_param1);
	arguments.push_back(p_param2);
}

MethodInfo::MethodInfo(Variant::Type ret, const String &p_name, const PropertyInfo &p_param1, const PropertyInfo &p_param2, const PropertyInfo &p_param3) :
		name(p_name),
		flags(METHOD_FLAG_NORMAL) {
	return_val.type = ret;
	arguments.push_back(p_param1);
	arguments.push_back(p_param2);
	arguments.push_back(p_param3);
}

MethodInfo::MethodInfo(Variant::Type ret, const String &p_name, const PropertyInfo &p_param1, const PropertyInfo &p_param2, const PropertyInfo &p_param3, const PropertyInfo &p_param4) :
		name(p_name),
		flags(METHOD_FLAG_NORMAL) {
	return_val.type = ret;
	arguments.push_back(p_param1);
	arguments.push_back(p_param2);
	arguments.push_back(p_param3);
	arguments.push_back(p_param4);
}

MethodInfo::MethodInfo(Variant::Type ret, const String &p_name, const PropertyInfo &p_param1, const PropertyInfo &p_param2, const PropertyInfo &p_param3, const PropertyInfo &p_param4, const PropertyInfo &p_param5) :
		name(p_name),
		flags(METHOD_FLAG_NORMAL) {
	return_val.type = ret;
	arguments.push_back(p_param1);
	arguments.push_back(p_param2);
	arguments.push_back(p_param3);
	arguments.push_back(p_param4);
	arguments.push_back(p_param5);
}

MethodInfo::MethodInfo(const PropertyInfo &p_ret, const String &p_name) :
		name(p_name),
		return_val(p_ret),
		flags(METHOD_FLAG_NORMAL) {
}

MethodInfo::MethodInfo(const PropertyInfo &p_ret, const String &p_name, const PropertyInfo &p_param1) :
		name(p_name),
		return_val(p_ret),
		flags(METHOD_FLAG_NORMAL) {
	arguments.push_back(p_param1);
}

MethodInfo::MethodInfo(const PropertyInfo &p_ret, const String &p_name, const PropertyInfo &p_param1, const PropertyInfo &p_param2) :
		name(p_name),
		return_val(p_ret),
		flags(METHOD_FLAG_NORMAL) {
	arguments.push_back(p_param1);
	arguments.push_back(p_param2);
}

MethodInfo::MethodInfo(const PropertyInfo &p_ret, const String &p_name, const PropertyInfo &p_param1, const PropertyInfo &p_param2, const PropertyInfo &p_param3) :
		name(p_name),
		return_val(p_ret),
		flags(METHOD_FLAG_NORMAL) {
	arguments.push_back(p_param1);
	arguments.push_back(p_param2);
	arguments.push_back(p_param3);
}

MethodInfo::MethodInfo(const PropertyInfo &p_ret, const String &p_name, const PropertyInfo &p_param1, const PropertyInfo &p_param2, const PropertyInfo &p_param3, const PropertyInfo &p_param4) :
		name(p_name),
		return_val(p_ret),
		flags(METHOD_FLAG_NORMAL) {
	arguments.push_back(p_param1);
	arguments.push_back(p_param2);
	arguments.push_back(p_param3);
	arguments.push_back(p_param4);
}

MethodInfo::MethodInfo(const PropertyInfo &p_ret, const String &p_name, const PropertyInfo &p_param1, const PropertyInfo &p_param2, const PropertyInfo &p_param3, const PropertyInfo &p_param4, const PropertyInfo &p_param5) :
		name(p_name),
		return_val(p_ret),
		flags(METHOD_FLAG_NORMAL) {
	arguments.push_back(p_param1);
	arguments.push_back(p_param2);
	arguments.push_back(p_param3);
	arguments.push_back(p_param4);
	arguments.push_back(p_param5);
}

Object::Connection::operator Variant() const {
	Dictionary d;
	d["signal"] = signal;
	d["callable"] = callable;
	d["flags"] = flags;
	d["binds"] = binds;
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
	if (d.has("binds")) {
		binds = d["binds"];
	}
}

bool Object::_predelete() {
	_predelete_ok = 1;
	notification(NOTIFICATION_PREDELETE, true);
	if (_predelete_ok) {
		_class_ptr = nullptr; //must restore so destructors can access class ptr correctly
	}
	return _predelete_ok;
}

void Object::_postinitialize() {
	_class_ptr = _get_class_namev();
	_initialize_classv();
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
		if (_extension->set(_extension_instance, (const GDNativeStringNamePtr)&p_name, (const GDNativeVariantPtr)&p_value)) {
			if (r_valid) {
				*r_valid = true;
			}
			return;
		}
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
	}

	//try built-in setgetter
	{
		if (ClassDB::set_property(this, p_name, p_value, r_valid)) {
			/*
			if (r_valid)
				*r_valid=true;
			*/
			return;
		}
	}

	if (p_name == CoreStringNames::get_singleton()->_script) {
		set_script(p_value);
		if (r_valid) {
			*r_valid = true;
		}
		return;

	} else if (p_name == CoreStringNames::get_singleton()->_meta) {
		//set_meta(p_name,p_value);
		metadata = p_value.duplicate();
		if (r_valid) {
			*r_valid = true;
		}
		return;
	}

	//something inside the object... :|
	bool success = _setv(p_name, p_value);
	if (success) {
		if (r_valid) {
			*r_valid = true;
		}
		return;
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

		if (_extension->get(_extension_instance, (const GDNativeStringNamePtr)&p_name, (GDNativeVariantPtr)&ret)) {
			if (r_valid) {
				*r_valid = true;
			}
			return ret;
		}
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
	}

	//try built-in setgetter
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

	} else if (p_name == CoreStringNames::get_singleton()->_meta) {
		ret = metadata;
		if (r_valid) {
			*r_valid = true;
		}
		return ret;

	} else {
		//something inside the object... :|
		bool success = _getv(p_name, ret);
		if (success) {
			if (r_valid) {
				*r_valid = true;
			}
			return ret;
		}

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
		p_list->push_back(PropertyInfo(Variant::NIL, "Script Variables", PROPERTY_HINT_NONE, String(), PROPERTY_USAGE_CATEGORY));
		script_instance->get_property_list(p_list);
	}

	_get_property_listv(p_list, p_reversed);

	if (_extension && _extension->get_property_list) {
		uint32_t pcount;
		const GDNativePropertyInfo *pinfo = _extension->get_property_list(_extension_instance, &pcount);
		for (uint32_t i = 0; i < pcount; i++) {
			p_list->push_back(PropertyInfo(Variant::Type(pinfo[i].type), pinfo[i].class_name, PropertyHint(pinfo[i].hint), pinfo[i].hint_string, pinfo[i].usage, pinfo[i].class_name));
		}
		if (_extension->free_property_list) {
			_extension->free_property_list(_extension_instance, pinfo);
		}
	}

	if (!is_class("Script")) { // can still be set, but this is for user-friendliness
		p_list->push_back(PropertyInfo(Variant::OBJECT, "script", PROPERTY_HINT_RESOURCE_TYPE, "Script", PROPERTY_USAGE_DEFAULT));
	}
	if (!metadata.is_empty()) {
		p_list->push_back(PropertyInfo(Variant::DICTIONARY, "__meta__", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR | PROPERTY_USAGE_INTERNAL));
	}
	if (script_instance && !p_reversed) {
		p_list->push_back(PropertyInfo(Variant::NIL, "Script Variables", PROPERTY_HINT_NONE, String(), PROPERTY_USAGE_CATEGORY));
		script_instance->get_property_list(p_list);
	}
}

void Object::_validate_property(PropertyInfo &property) const {
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
		r_error.argument = 0;
		return Variant();
	}

	if (p_args[0]->get_type() != Variant::STRING_NAME && p_args[0]->get_type() != Variant::STRING) {
		r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
		r_error.argument = 0;
		r_error.expected = Variant::STRING_NAME;
		return Variant();
	}

	StringName method = *p_args[0];

	return call(method, &p_args[1], p_argcount - 1, r_error);
}

Variant Object::_call_deferred_bind(const Variant **p_args, int p_argcount, Callable::CallError &r_error) {
	if (p_argcount < 1) {
		r_error.error = Callable::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS;
		r_error.argument = 0;
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

	MessageQueue::get_singleton()->push_call(get_instance_id(), method, &p_args[1], p_argcount - 1, true);

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

	return method != nullptr;
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
	Variant ret = call(p_method, argptrs, p_args.size(), ce);
	if (ce.error != Callable::CallError::CALL_OK) {
		ERR_FAIL_V_MSG(Variant(), "Error calling method from 'callv': " + Variant::get_call_error_text(this, p_method, argptrs, p_args.size(), ce) + ".");
	}
	return ret;
}

Variant Object::call(const StringName &p_name, VARIANT_ARG_DECLARE) {
	VARIANT_ARGPTRS;

	int argc = 0;
	for (int i = 0; i < VARIANT_ARG_MAX; i++) {
		if (argptr[i]->get_type() == Variant::NIL) {
			break;
		}
		argc++;
	}

	Callable::CallError error;

	Variant ret = call(p_name, argptr, argc, error);
	return ret;
}

Variant Object::call(const StringName &p_method, const Variant **p_args, int p_argcount, Callable::CallError &r_error) {
	r_error.error = Callable::CallError::CALL_OK;

	if (p_method == CoreStringNames::get_singleton()->_free) {
//free must be here, before anything, always ready
#ifdef DEBUG_ENABLED
		if (p_argcount != 0) {
			r_error.argument = 0;
			r_error.error = Callable::CallError::CALL_ERROR_TOO_MANY_ARGUMENTS;
			return Variant();
		}
		if (Object::cast_to<RefCounted>(this)) {
			r_error.argument = 0;
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
		ret = script_instance->call(p_method, p_args, p_argcount, r_error);
		//force jumptable
		switch (r_error.error) {
			case Callable::CallError::CALL_OK:
				return ret;
			case Callable::CallError::CALL_ERROR_INVALID_METHOD:
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
		ret = method->call(this, p_args, p_argcount, r_error);
	} else {
		r_error.error = Callable::CallError::CALL_ERROR_INVALID_METHOD;
	}

	return ret;
}

void Object::notification(int p_notification, bool p_reversed) {
	_notificationv(p_notification, p_reversed);

	if (script_instance) {
		script_instance->notification(p_notification);
	}

	if (_extension && _extension->notification) {
		_extension->notification(_extension_instance, p_notification);
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
		return _extension->to_string(_extension_instance);
	}
	return "[" + get_class() + ":" + itos(get_instance_id()) + "]";
}

void Object::set_script_and_instance(const Variant &p_script, ScriptInstance *p_instance) {
	//this function is not meant to be used in any of these ways
	ERR_FAIL_COND(p_script.is_null());
	ERR_FAIL_COND(!p_instance);
	ERR_FAIL_COND(script_instance != nullptr || !script.is_null());

	script = p_script;
	script_instance = p_instance;
}

void Object::set_script(const Variant &p_script) {
	if (script == p_script) {
		return;
	}

	if (script_instance) {
		memdelete(script_instance);
		script_instance = nullptr;
	}

	script = p_script;
	Ref<Script> s = script;

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
		metadata.erase(p_name);
		return;
	}

	metadata[p_name] = p_value;
}

Variant Object::get_meta(const StringName &p_name) const {
	ERR_FAIL_COND_V_MSG(!metadata.has(p_name), Variant(), "The object does not have any 'meta' values with the key '" + p_name + "'.");
	return metadata[p_name];
}

void Object::remove_meta(const StringName &p_name) {
	metadata.erase(p_name);
}

Array Object::_get_property_list_bind() const {
	List<PropertyInfo> lpi;
	get_property_list(&lpi);
	return convert_property_list(&lpi);
}

Array Object::_get_method_list_bind() const {
	List<MethodInfo> ml;
	get_method_list(&ml);
	Array ret;

	for (List<MethodInfo>::Element *E = ml.front(); E; E = E->next()) {
		Dictionary d = E->get();
		//va.push_back(d);
		ret.push_back(d);
	}

	return ret;
}

Vector<StringName> Object::_get_meta_list_bind() const {
	Vector<StringName> _metaret;

	List<Variant> keys;
	metadata.get_key_list(&keys);
	for (const Variant &E : keys) {
		_metaret.push_back(E);
	}

	return _metaret;
}

void Object::get_meta_list(List<StringName> *p_list) const {
	List<Variant> keys;
	metadata.get_key_list(&keys);
	for (const Variant &E : keys) {
		p_list->push_back(E);
	}
}

void Object::add_user_signal(const MethodInfo &p_signal) {
	ERR_FAIL_COND_MSG(p_signal.name == "", "Signal name cannot be empty.");
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

Variant Object::_emit_signal(const Variant **p_args, int p_argcount, Callable::CallError &r_error) {
	r_error.error = Callable::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS;

	ERR_FAIL_COND_V(p_argcount < 1, Variant());
	if (p_args[0]->get_type() != Variant::STRING_NAME && p_args[0]->get_type() != Variant::STRING) {
		r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
		r_error.argument = 0;
		r_error.expected = Variant::STRING_NAME;
		ERR_FAIL_COND_V(p_args[0]->get_type() != Variant::STRING_NAME && p_args[0]->get_type() != Variant::STRING, Variant());
	}

	r_error.error = Callable::CallError::CALL_OK;

	StringName signal = *p_args[0];

	const Variant **args = nullptr;

	int argc = p_argcount - 1;
	if (argc) {
		args = &p_args[1];
	}

	emit_signal(signal, args, argc);

	return Variant();
}

Error Object::emit_signal(const StringName &p_name, const Variant **p_args, int p_argcount) {
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

	List<_ObjectSignalDisconnectData> disconnect_data;

	//copy on write will ensure that disconnecting the signal or even deleting the object will not affect the signal calling.
	//this happens automatically and will not change the performance of calling.
	//awesome, isn't it?
	VMap<Callable, SignalData::Slot> slot_map = s->slot_map;

	int ssize = slot_map.size();

	OBJ_DEBUG_LOCK

	Vector<const Variant *> bind_mem;

	Error err = OK;

	for (int i = 0; i < ssize; i++) {
		const Connection &c = slot_map.getv(i).conn;

		Object *target = c.callable.get_object();
		if (!target) {
			// Target might have been deleted during signal callback, this is expected and OK.
			continue;
		}

		const Variant **args = p_args;
		int argc = p_argcount;

		if (c.binds.size()) {
			//handle binds
			bind_mem.resize(p_argcount + c.binds.size());

			for (int j = 0; j < p_argcount; j++) {
				bind_mem.write[j] = p_args[j];
			}
			for (int j = 0; j < c.binds.size(); j++) {
				bind_mem.write[p_argcount + j] = &c.binds[j];
			}

			args = (const Variant **)bind_mem.ptr();
			argc = bind_mem.size();
		}

		if (c.flags & CONNECT_DEFERRED) {
			MessageQueue::get_singleton()->push_callable(c.callable, args, argc, true);
		} else {
			Callable::CallError ce;
			_emitting = true;
			Variant ret;
			c.callable.call(args, argc, ret, ce);
			_emitting = false;

			if (ce.error != Callable::CallError::CALL_OK) {
#ifdef DEBUG_ENABLED
				if (c.flags & CONNECT_PERSIST && Engine::get_singleton()->is_editor_hint() && (script.is_null() || !Ref<Script>(script)->is_tool())) {
					continue;
				}
#endif
				if (ce.error == Callable::CallError::CALL_ERROR_INVALID_METHOD && !ClassDB::class_exists(target->get_class_name())) {
					//most likely object is not initialized yet, do not throw error.
				} else {
					ERR_PRINT("Error calling from signal '" + String(p_name) + "' to callable: " + Variant::get_callable_error_text(c.callable, args, argc, ce) + ".");
					err = ERR_METHOD_NOT_FOUND;
				}
			}
		}

		bool disconnect = c.flags & CONNECT_ONESHOT;
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

Error Object::emit_signal(const StringName &p_name, VARIANT_ARG_DECLARE) {
	VARIANT_ARGPTRS;

	int argc = 0;

	for (int i = 0; i < VARIANT_ARG_MAX; i++) {
		if (argptr[i]->get_type() == Variant::NIL) {
			break;
		}
		argc++;
	}

	return emit_signal(p_name, argptr, argc);
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

Array Object::_get_signal_list() const {
	List<MethodInfo> signal_list;
	get_signal_list(&signal_list);

	Array ret;
	for (const MethodInfo &E : signal_list) {
		ret.push_back(Dictionary(E));
	}

	return ret;
}

Array Object::_get_signal_connection_list(const String &p_signal) const {
	List<Connection> conns;
	get_all_signal_connections(&conns);

	Array ret;

	for (const Connection &c : conns) {
		if (c.signal.get_name() == p_signal) {
			ret.push_back(c);
		}
	}

	return ret;
}

Array Object::_get_incoming_connections() const {
	Array ret;
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
	const StringName *S = nullptr;

	while ((S = signal_map.next(S))) {
		if (signal_map[*S].user.name != "") {
			//user signal
			p_signals->push_back(signal_map[*S].user);
		}
	}
}

void Object::get_all_signal_connections(List<Connection> *p_connections) const {
	const StringName *S = nullptr;

	while ((S = signal_map.next(S))) {
		const SignalData *s = &signal_map[*S];

		for (int i = 0; i < s->slot_map.size(); i++) {
			p_connections->push_back(s->slot_map.getv(i).conn);
		}
	}
}

void Object::get_signal_connection_list(const StringName &p_signal, List<Connection> *p_connections) const {
	const SignalData *s = signal_map.getptr(p_signal);
	if (!s) {
		return; //nothing
	}

	for (int i = 0; i < s->slot_map.size(); i++) {
		p_connections->push_back(s->slot_map.getv(i).conn);
	}
}

int Object::get_persistent_signal_connection_count() const {
	int count = 0;
	const StringName *S = nullptr;

	while ((S = signal_map.next(S))) {
		const SignalData *s = &signal_map[*S];

		for (int i = 0; i < s->slot_map.size(); i++) {
			if (s->slot_map.getv(i).conn.flags & CONNECT_PERSIST) {
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

Error Object::connect(const StringName &p_signal, const Callable &p_callable, const Vector<Variant> &p_binds, uint32_t p_flags) {
	ERR_FAIL_COND_V_MSG(p_callable.is_null(), ERR_INVALID_PARAMETER, "Cannot connect to '" + p_signal + "': the provided callable is null.");

	Object *target_object = p_callable.get_object();
	ERR_FAIL_COND_V_MSG(!target_object, ERR_INVALID_PARAMETER, "Cannot connect to '" + p_signal + "' to callable '" + p_callable + "': the callable object is null.");

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

	SignalData::Slot slot;

	Connection conn;
	conn.callable = target;
	conn.signal = ::Signal(this, p_signal);
	conn.flags = p_flags;
	conn.binds = p_binds;
	slot.conn = conn;
	slot.cE = target_object->connections.push_back(conn);
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
	//const Map<Signal::Target,Signal::Slot>::Element *E = s->slot_map.find(target);
	//return (E!=nullptr );
}

void Object::disconnect(const StringName &p_signal, const Callable &p_callable) {
	_disconnect(p_signal, p_callable);
}

void Object::_disconnect(const StringName &p_signal, const Callable &p_callable, bool p_force) {
	ERR_FAIL_COND_MSG(p_callable.is_null(), "Cannot disconnect from '" + p_signal + "': the provided callable is null.");

	Object *target_object = p_callable.get_object();
	ERR_FAIL_COND_MSG(!target_object, "Cannot disconnect '" + p_signal + "' from callable '" + p_callable + "': the callable object is null.");

	SignalData *s = signal_map.getptr(p_signal);
	if (!s) {
		bool signal_is_valid = ClassDB::has_signal(get_class_name(), p_signal) ||
				(!script.is_null() && Ref<Script>(script)->has_script_signal(p_signal));
		ERR_FAIL_COND_MSG(signal_is_valid, "Attempt to disconnect a nonexistent connection from '" + to_string() + "'. Signal: '" + p_signal + "', callable: '" + p_callable + "'.");
	}
	ERR_FAIL_COND_MSG(!s, vformat("Disconnecting nonexistent signal '%s' in %s.", p_signal, to_string()));

	ERR_FAIL_COND_MSG(!s->slot_map.has(*p_callable.get_base_comparator()), "Disconnecting nonexistent signal '" + p_signal + "', callable: " + p_callable + ".");

	SignalData::Slot *slot = &s->slot_map[p_callable];

	if (!p_force) {
		slot->reference_count--; // by default is zero, if it was not referenced it will go below it
		if (slot->reference_count > 0) {
			return;
		}
	}

	target_object->connections.erase(slot->cE);
	s->slot_map.erase(*p_callable.get_base_comparator());

	if (s->slot_map.is_empty() && ClassDB::has_signal(get_class_name(), p_signal)) {
		//not user signal, delete
		signal_map.erase(p_signal);
	}
}

void Object::_set_bind(const String &p_set, const Variant &p_value) {
	set(p_set, p_value);
}

Variant Object::_get_bind(const String &p_name) const {
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
	return TranslationServer::get_singleton()->translate(p_message, p_context);
}

String Object::tr_n(const StringName &p_message, const StringName &p_message_plural, int p_n, const StringName &p_context) const {
	if (!_can_translate || !TranslationServer::get_singleton()) {
		// Return message based on English plural rule if translation is not possible.
		if (p_n == 1) {
			return p_message;
		}
		return p_message_plural;
	}
	return TranslationServer::get_singleton()->translate_plural(p_message, p_message_plural, p_n, p_context);
}

void Object::_clear_internal_resource_paths(const Variant &p_var) {
	switch (p_var.get_type()) {
		case Variant::OBJECT: {
			RES r = p_var;
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
	ClassDB::bind_method(D_METHOD("set_indexed", "property", "value"), &Object::_set_indexed_bind);
	ClassDB::bind_method(D_METHOD("get_indexed", "property"), &Object::_get_indexed_bind);
	ClassDB::bind_method(D_METHOD("get_property_list"), &Object::_get_property_list_bind);
	ClassDB::bind_method(D_METHOD("get_method_list"), &Object::_get_method_list_bind);
	ClassDB::bind_method(D_METHOD("notification", "what", "reversed"), &Object::notification, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("to_string"), &Object::to_string);
	ClassDB::bind_method(D_METHOD("get_instance_id"), &Object::get_instance_id);

	ClassDB::bind_method(D_METHOD("set_script", "script"), &Object::set_script);
	ClassDB::bind_method(D_METHOD("get_script"), &Object::get_script);

	ClassDB::bind_method(D_METHOD("set_meta", "name", "value"), &Object::set_meta);
	ClassDB::bind_method(D_METHOD("remove_meta", "name"), &Object::remove_meta);
	ClassDB::bind_method(D_METHOD("get_meta", "name"), &Object::get_meta);
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

	ClassDB::bind_method(D_METHOD("connect", "signal", "callable", "binds", "flags"), &Object::connect, DEFVAL(Array()), DEFVAL(0));
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

	ClassDB::add_virtual_method("Object", MethodInfo("free"), false);

	ADD_SIGNAL(MethodInfo("script_changed"));
	ADD_SIGNAL(MethodInfo("property_list_changed"));

#define BIND_OBJ_CORE_METHOD(m_method) \
	::ClassDB::add_virtual_method(get_class_static(), m_method, true, Vector<String>(), true);

	BIND_OBJ_CORE_METHOD(MethodInfo("_notification", PropertyInfo(Variant::INT, "what")));
	BIND_OBJ_CORE_METHOD(MethodInfo(Variant::BOOL, "_set", PropertyInfo(Variant::STRING_NAME, "property"), PropertyInfo(Variant::NIL, "value")));
#ifdef TOOLS_ENABLED
	MethodInfo miget("_get", PropertyInfo(Variant::STRING_NAME, "property"));
	miget.return_val.name = "Variant";
	miget.return_val.usage |= PROPERTY_USAGE_NIL_IS_VARIANT;
	BIND_OBJ_CORE_METHOD(miget);

	MethodInfo plget("_get_property_list");

	plget.return_val.type = Variant::ARRAY;
	BIND_OBJ_CORE_METHOD(plget);

#endif
	BIND_OBJ_CORE_METHOD(MethodInfo("_init"));
	BIND_OBJ_CORE_METHOD(MethodInfo(Variant::STRING, "_to_string"));

	BIND_CONSTANT(NOTIFICATION_POSTINITIALIZE);
	BIND_CONSTANT(NOTIFICATION_PREDELETE);

	BIND_ENUM_CONSTANT(CONNECT_DEFERRED);
	BIND_ENUM_CONSTANT(CONNECT_PERSIST);
	BIND_ENUM_CONSTANT(CONNECT_ONESHOT);
	BIND_ENUM_CONSTANT(CONNECT_REFERENCE_COUNTED);
}

void Object::call_deferred(const StringName &p_method, VARIANT_ARG_DECLARE) {
	MessageQueue::get_singleton()->push_call(this, p_method, VARIANT_ARG_PASS);
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

void Object::get_translatable_strings(List<String> *p_strings) const {
	List<PropertyInfo> plist;
	get_property_list(&plist);

	for (const PropertyInfo &E : plist) {
		if (!(E.usage & PROPERTY_USAGE_INTERNATIONALIZED)) {
			continue;
		}

		String text = get(E.name);

		if (text == "") {
			continue;
		}

		p_strings->push_back(text);
	}
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

void Object::set_instance_binding(void *p_token, void *p_binding, const GDNativeInstanceBindingCallbacks *p_callbacks) {
	// This is only meant to be used on creation by the binder.
	ERR_FAIL_COND(_instance_bindings != nullptr);
	_instance_bindings = (InstanceBinding *)memalloc(sizeof(InstanceBinding));
	_instance_bindings[0].binding = p_binding;
	_instance_bindings[0].free_callback = p_callbacks->free_callback;
	_instance_bindings[0].reference_callback = p_callbacks->reference_callback;
	_instance_bindings[0].token = p_token;
	_instance_binding_count = 1;
}

void *Object::get_instance_binding(void *p_token, const GDNativeInstanceBindingCallbacks *p_callbacks) {
	void *binding = nullptr;
	_instance_binding_mutex.lock();
	for (uint32_t i = 0; i < _instance_binding_count; i++) {
		if (_instance_bindings[i].token == p_token) {
			binding = _instance_bindings[i].binding;
			break;
		}
	}
	if (unlikely(!binding)) {
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

void Object::_construct_object(bool p_reference) {
	type_is_reference = p_reference;
	_instance_id = ObjectDB::add_instance(this);

	ClassDB::instance_get_native_extension_data(&_extension, &_extension_instance, this);

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

Object::~Object() {
	if (script_instance) {
		memdelete(script_instance);
	}
	script_instance = nullptr;

	if (_extension && _extension->free_instance) {
		_extension->free_instance(_extension->class_userdata, _extension_instance);
		_extension = nullptr;
		_extension_instance = nullptr;
	}

	const StringName *S = nullptr;

	if (_emitting) {
		//@todo this may need to actually reach the debugger prioritarily somehow because it may crash before
		ERR_PRINT("Object " + to_string() + " was freed or unreferenced while a signal is being emitted from it. Try connecting to the signal using 'CONNECT_DEFERRED' flag, or use queue_free() to free the object (if this object is a Node) to avoid this error and potential crashes.");
	}

	while ((S = signal_map.next(nullptr))) {
		SignalData *s = &signal_map[*S];

		//brute force disconnect for performance
		int slot_count = s->slot_map.size();
		const VMap<Callable, SignalData::Slot>::Pair *slot_list = s->slot_map.get_array();

		for (int i = 0; i < slot_count; i++) {
			slot_list[i].value.conn.callable.get_object()->connections.erase(slot_list[i].value.cE);
		}

		signal_map.erase(*S);
	}

	//signals from nodes that connect to this node
	while (connections.size()) {
		Connection c = connections.front()->get();
		c.signal.get_object()->_disconnect(c.signal.get_name(), c.callable, true);
	}

	ObjectDB::remove_instance(this);
	_instance_id = ObjectID();
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
