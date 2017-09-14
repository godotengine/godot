/*************************************************************************/
/*  object.cpp                                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "class_db.h"
#include "core_string_names.h"
#include "message_queue.h"
#include "os/os.h"
#include "print_string.h"
#include "resource.h"
#include "script_language.h"
#include "translation.h"

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

	if (p_dict.has("type"))
		pi.type = Variant::Type(int(p_dict["type"]));

	if (p_dict.has("name"))
		pi.name = p_dict["name"];

	if (p_dict.has("class_name"))
		pi.class_name = p_dict["class_name"];

	if (p_dict.has("hint"))
		pi.hint = PropertyHint(int(p_dict["hint"]));

	if (p_dict.has("hint_string"))

		pi.hint_string = p_dict["hint_string"];

	if (p_dict.has("usage"))
		pi.usage = p_dict["usage"];

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
	for (int i = 0; i < default_arguments.size(); i++)
		da.push_back(default_arguments[i]);
	d["default_args"] = da;
	d["flags"] = flags;
	d["id"] = id;
	Dictionary r = return_val;
	d["return"] = r;
	return d;
}

MethodInfo::MethodInfo()
	: flags(METHOD_FLAG_NORMAL),
	  id(0) {
}

MethodInfo MethodInfo::from_dict(const Dictionary &p_dict) {

	MethodInfo mi;

	if (p_dict.has("name"))
		mi.name = p_dict["name"];
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

	if (p_dict.has("flags"))
		mi.flags = p_dict["flags"];

	return mi;
}

MethodInfo::MethodInfo(const String &p_name)
	: name(p_name),
	  flags(METHOD_FLAG_NORMAL),
	  id(0) {
}
MethodInfo::MethodInfo(const String &p_name, const PropertyInfo &p_param1)
	: name(p_name),
	  flags(METHOD_FLAG_NORMAL),
	  id(0) {
	arguments.push_back(p_param1);
}
MethodInfo::MethodInfo(const String &p_name, const PropertyInfo &p_param1, const PropertyInfo &p_param2)
	: name(p_name),
	  flags(METHOD_FLAG_NORMAL),
	  id(0) {
	arguments.push_back(p_param1);
	arguments.push_back(p_param2);
}

MethodInfo::MethodInfo(const String &p_name, const PropertyInfo &p_param1, const PropertyInfo &p_param2, const PropertyInfo &p_param3)
	: name(p_name),
	  flags(METHOD_FLAG_NORMAL),
	  id(0) {
	arguments.push_back(p_param1);
	arguments.push_back(p_param2);
	arguments.push_back(p_param3);
}

MethodInfo::MethodInfo(const String &p_name, const PropertyInfo &p_param1, const PropertyInfo &p_param2, const PropertyInfo &p_param3, const PropertyInfo &p_param4)
	: name(p_name),
	  flags(METHOD_FLAG_NORMAL),
	  id(0) {
	arguments.push_back(p_param1);
	arguments.push_back(p_param2);
	arguments.push_back(p_param3);
	arguments.push_back(p_param4);
}

MethodInfo::MethodInfo(const String &p_name, const PropertyInfo &p_param1, const PropertyInfo &p_param2, const PropertyInfo &p_param3, const PropertyInfo &p_param4, const PropertyInfo &p_param5)
	: name(p_name),
	  flags(METHOD_FLAG_NORMAL),
	  id(0) {
	arguments.push_back(p_param1);
	arguments.push_back(p_param2);
	arguments.push_back(p_param3);
	arguments.push_back(p_param4);
	arguments.push_back(p_param5);
}

MethodInfo::MethodInfo(Variant::Type ret)
	: flags(METHOD_FLAG_NORMAL),
	  id(0) {
	return_val.type = ret;
}

MethodInfo::MethodInfo(Variant::Type ret, const String &p_name)
	: name(p_name),
	  flags(METHOD_FLAG_NORMAL),
	  id(0) {
	return_val.type = ret;
}
MethodInfo::MethodInfo(Variant::Type ret, const String &p_name, const PropertyInfo &p_param1)
	: name(p_name),
	  flags(METHOD_FLAG_NORMAL),
	  id(0) {
	return_val.type = ret;
	arguments.push_back(p_param1);
}
MethodInfo::MethodInfo(Variant::Type ret, const String &p_name, const PropertyInfo &p_param1, const PropertyInfo &p_param2)
	: name(p_name),
	  flags(METHOD_FLAG_NORMAL),
	  id(0) {
	return_val.type = ret;
	arguments.push_back(p_param1);
	arguments.push_back(p_param2);
}

MethodInfo::MethodInfo(Variant::Type ret, const String &p_name, const PropertyInfo &p_param1, const PropertyInfo &p_param2, const PropertyInfo &p_param3)
	: name(p_name),
	  flags(METHOD_FLAG_NORMAL),
	  id(0) {
	return_val.type = ret;
	arguments.push_back(p_param1);
	arguments.push_back(p_param2);
	arguments.push_back(p_param3);
}

MethodInfo::MethodInfo(Variant::Type ret, const String &p_name, const PropertyInfo &p_param1, const PropertyInfo &p_param2, const PropertyInfo &p_param3, const PropertyInfo &p_param4)
	: name(p_name),
	  flags(METHOD_FLAG_NORMAL),
	  id(0) {
	return_val.type = ret;
	arguments.push_back(p_param1);
	arguments.push_back(p_param2);
	arguments.push_back(p_param3);
	arguments.push_back(p_param4);
}

MethodInfo::MethodInfo(Variant::Type ret, const String &p_name, const PropertyInfo &p_param1, const PropertyInfo &p_param2, const PropertyInfo &p_param3, const PropertyInfo &p_param4, const PropertyInfo &p_param5)
	: name(p_name),
	  flags(METHOD_FLAG_NORMAL),
	  id(0) {
	return_val.type = ret;
	arguments.push_back(p_param1);
	arguments.push_back(p_param2);
	arguments.push_back(p_param3);
	arguments.push_back(p_param4);
	arguments.push_back(p_param5);
}

MethodInfo::MethodInfo(const PropertyInfo &p_ret, const String &p_name)
	: name(p_name),
	  flags(METHOD_FLAG_NORMAL),
	  return_val(p_ret),
	  id(0) {
}

MethodInfo::MethodInfo(const PropertyInfo &p_ret, const String &p_name, const PropertyInfo &p_param1)
	: name(p_name),
	  return_val(p_ret),
	  flags(METHOD_FLAG_NORMAL),
	  id(0) {
	arguments.push_back(p_param1);
}

MethodInfo::MethodInfo(const PropertyInfo &p_ret, const String &p_name, const PropertyInfo &p_param1, const PropertyInfo &p_param2)
	: name(p_name),
	  return_val(p_ret),
	  flags(METHOD_FLAG_NORMAL),
	  id(0) {
	arguments.push_back(p_param1);
	arguments.push_back(p_param2);
}

MethodInfo::MethodInfo(const PropertyInfo &p_ret, const String &p_name, const PropertyInfo &p_param1, const PropertyInfo &p_param2, const PropertyInfo &p_param3)
	: name(p_name),
	  return_val(p_ret),
	  flags(METHOD_FLAG_NORMAL),
	  id(0) {
	arguments.push_back(p_param1);
	arguments.push_back(p_param2);
	arguments.push_back(p_param3);
}

MethodInfo::MethodInfo(const PropertyInfo &p_ret, const String &p_name, const PropertyInfo &p_param1, const PropertyInfo &p_param2, const PropertyInfo &p_param3, const PropertyInfo &p_param4)
	: name(p_name),
	  return_val(p_ret),
	  flags(METHOD_FLAG_NORMAL),
	  id(0) {
	arguments.push_back(p_param1);
	arguments.push_back(p_param2);
	arguments.push_back(p_param3);
	arguments.push_back(p_param4);
}

MethodInfo::MethodInfo(const PropertyInfo &p_ret, const String &p_name, const PropertyInfo &p_param1, const PropertyInfo &p_param2, const PropertyInfo &p_param3, const PropertyInfo &p_param4, const PropertyInfo &p_param5)
	: name(p_name),
	  return_val(p_ret),
	  flags(METHOD_FLAG_NORMAL),
	  id(0) {
	arguments.push_back(p_param1);
	arguments.push_back(p_param2);
	arguments.push_back(p_param3);
	arguments.push_back(p_param4);
	arguments.push_back(p_param5);
}

Object::Connection::operator Variant() const {

	Dictionary d;
	d["source"] = source;
	d["signal"] = signal;
	d["target"] = target;
	d["method"] = method;
	d["flags"] = flags;
	d["binds"] = binds;
	return d;
}

bool Object::Connection::operator<(const Connection &p_conn) const {

	if (source == p_conn.source) {

		if (signal == p_conn.signal) {

			if (target == p_conn.target) {

				return method < p_conn.method;
			} else {

				return target < p_conn.target;
			}
		} else
			return signal < p_conn.signal;
	} else {
		return source < p_conn.source;
	}
}
Object::Connection::Connection(const Variant &p_variant) {

	Dictionary d = p_variant;
	if (d.has("source"))
		source = d["source"];
	if (d.has("signal"))
		signal = d["signal"];
	if (d.has("target"))
		target = d["target"];
	if (d.has("method"))
		method = d["method"];
	if (d.has("flags"))
		flags = d["flags"];
	if (d.has("binds"))
		binds = d["binds"];
}

bool Object::_predelete() {

	_predelete_ok = 1;
	notification(NOTIFICATION_PREDELETE, true);
	if (_predelete_ok) {
		_class_ptr = NULL; //must restore so destructors can access class ptr correctly
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
			if (r_valid)
				*r_valid = true;
			return;
		}
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
		if (r_valid)
			*r_valid = true;
		return;

	} else if (p_name == CoreStringNames::get_singleton()->_meta) {
		//set_meta(p_name,p_value);
		metadata = p_value;
		if (r_valid)
			*r_valid = true;
		return;
#ifdef TOOLS_ENABLED
	} else if (p_name == CoreStringNames::get_singleton()->_sections_unfolded) {
		Array arr = p_value;
		for (int i = 0; i < arr.size(); i++) {
			editor_section_folding.insert(arr[i]);
		}
		if (r_valid)
			*r_valid = true;
		return;
#endif
	} else {
		//something inside the object... :|
		bool success = _setv(p_name, p_value);
		if (success) {
			if (r_valid)
				*r_valid = true;
			return;
		}
		setvar(p_name, p_value, r_valid);
	}
}

Variant Object::get(const StringName &p_name, bool *r_valid) const {

	Variant ret;

	if (script_instance) {

		if (script_instance->get(p_name, ret)) {
			if (r_valid)
				*r_valid = true;
			return ret;
		}
	}

	//try built-in setgetter
	{
		if (ClassDB::get_property(const_cast<Object *>(this), p_name, ret)) {
			if (r_valid)
				*r_valid = true;
			return ret;
		}
	}

	if (p_name == CoreStringNames::get_singleton()->_script) {
		ret = get_script();
		if (r_valid)
			*r_valid = true;
		return ret;

	} else if (p_name == CoreStringNames::get_singleton()->_meta) {
		ret = metadata;
		if (r_valid)
			*r_valid = true;
		return ret;
#ifdef TOOLS_ENABLED
	} else if (p_name == CoreStringNames::get_singleton()->_sections_unfolded) {
		Array array;
		for (Set<String>::Element *E = editor_section_folding.front(); E; E = E->next()) {
			array.push_back(E->get());
		}
		if (r_valid)
			*r_valid = true;
		return array;
#endif
	} else {
		//something inside the object... :|
		bool success = _getv(p_name, ret);
		if (success) {
			if (r_valid)
				*r_valid = true;
			return ret;
		}
		//if nothing else, use getvar
		return getvar(p_name, r_valid);
	}
}

void Object::get_property_list(List<PropertyInfo> *p_list, bool p_reversed) const {

	if (script_instance && p_reversed) {
		p_list->push_back(PropertyInfo(Variant::NIL, "Script Variables", PROPERTY_HINT_NONE, String(), PROPERTY_USAGE_CATEGORY));
		script_instance->get_property_list(p_list);
	}

	_get_property_listv(p_list, p_reversed);

	if (!is_class("Script")) // can still be set, but this is for userfriendlyness
		p_list->push_back(PropertyInfo(Variant::OBJECT, "script", PROPERTY_HINT_RESOURCE_TYPE, "Script", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_STORE_IF_NONZERO));
#ifdef TOOLS_ENABLED
	if (editor_section_folding.size()) {
		p_list->push_back(PropertyInfo(Variant::ARRAY, CoreStringNames::get_singleton()->_sections_unfolded, PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR));
	}
#endif
	if (!metadata.empty())
		p_list->push_back(PropertyInfo(Variant::DICTIONARY, "__meta__", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR | PROPERTY_USAGE_STORE_IF_NONZERO));
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

Variant Object::_call_bind(const Variant **p_args, int p_argcount, Variant::CallError &r_error) {

	if (p_argcount < 1) {
		r_error.error = Variant::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS;
		r_error.argument = 0;
		return Variant();
	}

	if (p_args[0]->get_type() != Variant::STRING) {
		r_error.error = Variant::CallError::CALL_ERROR_INVALID_ARGUMENT;
		r_error.argument = 0;
		r_error.expected = Variant::STRING;
		return Variant();
	}

	StringName method = *p_args[0];

	return call(method, &p_args[1], p_argcount - 1, r_error);
}

Variant Object::_call_deferred_bind(const Variant **p_args, int p_argcount, Variant::CallError &r_error) {

	if (p_argcount < 1) {
		r_error.error = Variant::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS;
		r_error.argument = 0;
		return Variant();
	}

	if (p_args[0]->get_type() != Variant::STRING) {
		r_error.error = Variant::CallError::CALL_ERROR_INVALID_ARGUMENT;
		r_error.argument = 0;
		r_error.expected = Variant::STRING;
		return Variant();
	}

	r_error.error = Variant::CallError::CALL_OK;

	StringName method = *p_args[0];

	MessageQueue::get_singleton()->push_call(get_instance_id(), method, &p_args[1], p_argcount - 1);

	return Variant();
}

#ifdef DEBUG_ENABLED
static bool _test_call_error(const StringName &p_func, const Variant::CallError &error) {

	switch (error.error) {

		case Variant::CallError::CALL_OK:
			return true;
		case Variant::CallError::CALL_ERROR_INVALID_METHOD:
			return false;
		case Variant::CallError::CALL_ERROR_INVALID_ARGUMENT: {

			ERR_EXPLAIN("Error Calling Function: " + String(p_func) + " - Invalid type for argument " + itos(error.argument) + ", expected " + Variant::get_type_name(error.expected));
			ERR_FAIL_V(true);
		} break;
		case Variant::CallError::CALL_ERROR_TOO_MANY_ARGUMENTS: {

			ERR_EXPLAIN("Error Calling Function: " + String(p_func) + " - Too many arguments, expected " + itos(error.argument));
			ERR_FAIL_V(true);

		} break;
		case Variant::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS: {

			ERR_EXPLAIN("Error Calling Function: " + String(p_func) + " - Too few arguments, expected " + itos(error.argument));
			ERR_FAIL_V(true);

		} break;
		case Variant::CallError::CALL_ERROR_INSTANCE_IS_NULL: {
		} //?
	}

	return true;
}
#else

#define _test_call_error(m_str, m_err) ((m_err.error == Variant::CallError::CALL_ERROR_INVALID_METHOD) ? false : true)

#endif

void Object::call_multilevel(const StringName &p_method, const Variant **p_args, int p_argcount) {

	if (p_method == CoreStringNames::get_singleton()->_free) {
#ifdef DEBUG_ENABLED
		if (Object::cast_to<Reference>(this)) {
			ERR_EXPLAIN("Can't 'free' a reference.");
			ERR_FAIL();
			return;
		}

		if (_lock_index.get() > 1) {
			ERR_EXPLAIN("Object is locked and can't be freed.");
			ERR_FAIL();
			return;
		}
#endif

		//must be here, must be before everything,
		memdelete(this);
		return;
	}

	//Variant ret;
	OBJ_DEBUG_LOCK

	Variant::CallError error;

	if (script_instance) {
		script_instance->call_multilevel(p_method, p_args, p_argcount);
		//_test_call_error(p_method,error);
	}

	MethodBind *method = ClassDB::get_method(get_class_name(), p_method);

	if (method) {

		method->call(this, p_args, p_argcount, error);
		_test_call_error(p_method, error);
	}
}

void Object::call_multilevel_reversed(const StringName &p_method, const Variant **p_args, int p_argcount) {

	MethodBind *method = ClassDB::get_method(get_class_name(), p_method);

	Variant::CallError error;
	OBJ_DEBUG_LOCK

	if (method) {

		method->call(this, p_args, p_argcount, error);
		_test_call_error(p_method, error);
	}

	//Variant ret;

	if (script_instance) {
		script_instance->call_multilevel_reversed(p_method, p_args, p_argcount);
		//_test_call_error(p_method,error);
	}
}

bool Object::has_method(const StringName &p_method) const {

	if (p_method == CoreStringNames::get_singleton()->_free) {
		return true;
	}

	if (script_instance && script_instance->has_method(p_method)) {
		return true;
	}

	MethodBind *method = ClassDB::get_method(get_class_name(), p_method);

	if (method) {
		return true;
	}

	return false;
}

Variant Object::getvar(const Variant &p_key, bool *r_valid) const {

	if (r_valid)
		*r_valid = false;
	return Variant();
}
void Object::setvar(const Variant &p_key, const Variant &p_value, bool *r_valid) {

	if (r_valid)
		*r_valid = false;
}

Variant Object::callv(const StringName &p_method, const Array &p_args) {

	if (p_args.size() == 0) {
		return call(p_method);
	}

	Vector<Variant> args;
	args.resize(p_args.size());
	Vector<const Variant *> argptrs;
	argptrs.resize(p_args.size());

	for (int i = 0; i < p_args.size(); i++) {
		args[i] = p_args[i];
		argptrs[i] = &args[i];
	}

	Variant::CallError ce;
	return call(p_method, argptrs.ptr(), p_args.size(), ce);
}

Variant Object::call(const StringName &p_name, VARIANT_ARG_DECLARE) {

	VARIANT_ARGPTRS;

	int argc = 0;
	for (int i = 0; i < VARIANT_ARG_MAX; i++) {
		if (argptr[i]->get_type() == Variant::NIL)
			break;
		argc++;
	}

	Variant::CallError error;

	Variant ret = call(p_name, argptr, argc, error);
	return ret;
}

void Object::call_multilevel(const StringName &p_name, VARIANT_ARG_DECLARE) {

	VARIANT_ARGPTRS;

	int argc = 0;
	for (int i = 0; i < VARIANT_ARG_MAX; i++) {
		if (argptr[i]->get_type() == Variant::NIL)
			break;
		argc++;
	}

	//Variant::CallError error;
	call_multilevel(p_name, argptr, argc);
}

Variant Object::call(const StringName &p_method, const Variant **p_args, int p_argcount, Variant::CallError &r_error) {

	r_error.error = Variant::CallError::CALL_OK;

	if (p_method == CoreStringNames::get_singleton()->_free) {
//free must be here, before anything, always ready
#ifdef DEBUG_ENABLED
		if (p_argcount != 0) {
			r_error.argument = 0;
			r_error.error = Variant::CallError::CALL_ERROR_TOO_MANY_ARGUMENTS;
			return Variant();
		}
		if (Object::cast_to<Reference>(this)) {
			r_error.argument = 0;
			r_error.error = Variant::CallError::CALL_ERROR_INVALID_METHOD;
			ERR_EXPLAIN("Can't 'free' a reference.");
			ERR_FAIL_V(Variant());
		}

		if (_lock_index.get() > 1) {
			r_error.argument = 0;
			r_error.error = Variant::CallError::CALL_ERROR_INVALID_METHOD;
			ERR_EXPLAIN("Object is locked and can't be freed.");
			ERR_FAIL_V(Variant());
		}

#endif
		//must be here, must be before everything,
		memdelete(this);
		r_error.error = Variant::CallError::CALL_OK;
		return Variant();
	}

	Variant ret;
	OBJ_DEBUG_LOCK
	if (script_instance) {
		ret = script_instance->call(p_method, p_args, p_argcount, r_error);
		//force jumptable
		switch (r_error.error) {

			case Variant::CallError::CALL_OK:
				return ret;
			case Variant::CallError::CALL_ERROR_INVALID_METHOD:
				break;
			case Variant::CallError::CALL_ERROR_INVALID_ARGUMENT:
			case Variant::CallError::CALL_ERROR_TOO_MANY_ARGUMENTS:
			case Variant::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS:
				return ret;
			case Variant::CallError::CALL_ERROR_INSTANCE_IS_NULL: {
			}
		}
	}

	MethodBind *method = ClassDB::get_method(get_class_name(), p_method);

	if (method) {

		ret = method->call(this, p_args, p_argcount, r_error);
	} else {
		r_error.error = Variant::CallError::CALL_ERROR_INVALID_METHOD;
	}

	return ret;
}

void Object::notification(int p_notification, bool p_reversed) {

	_notificationv(p_notification, p_reversed);

	if (script_instance) {
		script_instance->notification(p_notification);
	}
}

void Object::_changed_callback(Object *p_changed, const char *p_prop) {
}

void Object::add_change_receptor(Object *p_receptor) {

	change_receptors.insert(p_receptor);
}

void Object::remove_change_receptor(Object *p_receptor) {

	change_receptors.erase(p_receptor);
}

void Object::property_list_changed_notify() {

	_change_notify();
}

void Object::cancel_delete() {

	_predelete_ok = true;
}

void Object::set_script_and_instance(const RefPtr &p_script, ScriptInstance *p_instance) {

	//this function is not meant to be used in any of these ways
	ERR_FAIL_COND(p_script.is_null());
	ERR_FAIL_COND(!p_instance);
	ERR_FAIL_COND(script_instance != NULL || !script.is_null());

	script = p_script;
	script_instance = p_instance;
}

void Object::set_script(const RefPtr &p_script) {

	if (script == p_script)
		return;

	if (script_instance) {
		memdelete(script_instance);
		script_instance = NULL;
	}

	script = p_script;
	Ref<Script> s(script);

	if (!s.is_null() && s->can_instance()) {
		OBJ_DEBUG_LOCK
		script_instance = s->instance_create(this);
	}

	_change_notify("script");
	emit_signal(CoreStringNames::get_singleton()->script_changed);
}

void Object::set_script_instance(ScriptInstance *p_instance) {

	if (script_instance == p_instance)
		return;

	if (script_instance)
		memdelete(script_instance);

	script_instance = p_instance;

	if (p_instance)
		script = p_instance->get_script().get_ref_ptr();
	else
		script = RefPtr();
}

RefPtr Object::get_script() const {

	return script;
}

bool Object::has_meta(const String &p_name) const {

	return metadata.has(p_name);
}

void Object::set_meta(const String &p_name, const Variant &p_value) {

	if (p_value.get_type() == Variant::NIL) {
		metadata.erase(p_name);
		return;
	};

	metadata[p_name] = p_value;
}

Variant Object::get_meta(const String &p_name) const {

	ERR_FAIL_COND_V(!metadata.has(p_name), Variant());
	return metadata[p_name];
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

PoolVector<String> Object::_get_meta_list_bind() const {

	PoolVector<String> _metaret;

	List<Variant> keys;
	metadata.get_key_list(&keys);
	for (List<Variant>::Element *E = keys.front(); E; E = E->next()) {

		_metaret.push_back(E->get());
	}

	return _metaret;
}
void Object::get_meta_list(List<String> *p_list) const {

	List<Variant> keys;
	metadata.get_key_list(&keys);
	for (List<Variant>::Element *E = keys.front(); E; E = E->next()) {

		p_list->push_back(E->get());
	}
}

void Object::add_user_signal(const MethodInfo &p_signal) {

	ERR_FAIL_COND(p_signal.name == "");
	ERR_FAIL_COND(ClassDB::has_signal(get_class_name(), p_signal.name));
	ERR_FAIL_COND(signal_map.has(p_signal.name));
	Signal s;
	s.user = p_signal;
	signal_map[p_signal.name] = s;
}

bool Object::_has_user_signal(const StringName &p_name) const {

	if (!signal_map.has(p_name))
		return false;
	return signal_map[p_name].user.name.length() > 0;
}

struct _ObjectSignalDisconnectData {

	StringName signal;
	Object *target;
	StringName method;
};

Variant Object::_emit_signal(const Variant **p_args, int p_argcount, Variant::CallError &r_error) {

	r_error.error = Variant::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS;

	ERR_FAIL_COND_V(p_argcount < 1, Variant());
	if (p_args[0]->get_type() != Variant::STRING) {
		r_error.error = Variant::CallError::CALL_ERROR_INVALID_ARGUMENT;
		r_error.argument = 0;
		r_error.expected = Variant::STRING;
		ERR_FAIL_COND_V(p_args[0]->get_type() != Variant::STRING, Variant());
	}

	r_error.error = Variant::CallError::CALL_OK;

	StringName signal = *p_args[0];

	const Variant **args = NULL;

	int argc = p_argcount - 1;
	if (argc) {
		args = &p_args[1];
	}

	emit_signal(signal, args, argc);

	return Variant();
}

Error Object::emit_signal(const StringName &p_name, const Variant **p_args, int p_argcount) {

	if (_block_signals)
		return ERR_CANT_AQUIRE_RESOURCE; //no emit, signals blocked

	Signal *s = signal_map.getptr(p_name);
	if (!s) {
#ifdef DEBUG_ENABLED
		bool signal_is_valid = ClassDB::has_signal(get_class_name(), p_name);
		//check in script
		if (!signal_is_valid && !script.is_null() && !Ref<Script>(script)->has_script_signal(p_name)) {
			ERR_EXPLAIN("Can't emit non-existing signal " + String("\"") + p_name + "\".");
			ERR_FAIL_V(ERR_UNAVAILABLE);
		}
#endif
		//not connected? just return
		return ERR_UNAVAILABLE;
	}

	List<_ObjectSignalDisconnectData> disconnect_data;

	//copy on write will ensure that disconnecting the signal or even deleting the object will not affect the signal calling.
	//this happens automatically and will not change the performance of calling.
	//awesome, isn't it?
	VMap<Signal::Target, Signal::Slot> slot_map = s->slot_map;

	int ssize = slot_map.size();

	OBJ_DEBUG_LOCK

	Vector<const Variant *> bind_mem;

	Error err = OK;

	for (int i = 0; i < ssize; i++) {

		const Connection &c = slot_map.getv(i).conn;

		Object *target;
#ifdef DEBUG_ENABLED
		target = ObjectDB::get_instance(slot_map.getk(i)._id);
		ERR_CONTINUE(!target);
#else
		target = c.target;
#endif

		const Variant **args = p_args;
		int argc = p_argcount;

		if (c.binds.size()) {
			//handle binds
			bind_mem.resize(p_argcount + c.binds.size());

			for (int j = 0; j < p_argcount; j++) {
				bind_mem[j] = p_args[j];
			}
			for (int j = 0; j < c.binds.size(); j++) {
				bind_mem[p_argcount + j] = &c.binds[j];
			}

			args = bind_mem.ptr();
			argc = bind_mem.size();
		}

		if (c.flags & CONNECT_DEFERRED) {
			MessageQueue::get_singleton()->push_call(target->get_instance_id(), c.method, args, argc, true);
		} else {
			Variant::CallError ce;
			target->call(c.method, args, argc, ce);

			if (ce.error != Variant::CallError::CALL_OK) {

				if (ce.error == Variant::CallError::CALL_ERROR_INVALID_METHOD && !ClassDB::class_exists(target->get_class_name())) {
					//most likely object is not initialized yet, do not throw error.
				} else {
					ERR_PRINTS("Error calling method from signal '" + String(p_name) + "': " + Variant::get_call_error_text(target, c.method, args, argc, ce));
					err = ERR_METHOD_NOT_FOUND;
				}
			}
		}

		if (c.flags & CONNECT_ONESHOT) {
			_ObjectSignalDisconnectData dd;
			dd.signal = p_name;
			dd.target = target;
			dd.method = c.method;
			disconnect_data.push_back(dd);
		}
	}

	while (!disconnect_data.empty()) {

		const _ObjectSignalDisconnectData &dd = disconnect_data.front()->get();
		disconnect(dd.signal, dd.target, dd.method);
		disconnect_data.pop_front();
	}

	return err;
}

Error Object::emit_signal(const StringName &p_name, VARIANT_ARG_DECLARE) {

	VARIANT_ARGPTRS;

	int argc = 0;

	for (int i = 0; i < VARIANT_ARG_MAX; i++) {

		if (argptr[i]->get_type() == Variant::NIL)
			break;
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

		if (d.has("name"))
			param.name = d["name"];
		if (d.has("type"))
			param.type = (Variant::Type)(int)d["type"];

		mi.arguments.push_back(param);
	}

	add_user_signal(mi);
}

Array Object::_get_signal_list() const {

	List<MethodInfo> signal_list;
	get_signal_list(&signal_list);

	Array ret;
	for (List<MethodInfo>::Element *E = signal_list.front(); E; E = E->next()) {

		ret.push_back(Dictionary(E->get()));
	}

	return ret;
}

Array Object::_get_signal_connection_list(const String &p_signal) const {

	List<Connection> conns;
	get_all_signal_connections(&conns);

	Array ret;

	for (List<Connection>::Element *E = conns.front(); E; E = E->next()) {

		Connection &c = E->get();
		if (c.signal == p_signal) {
			Dictionary rc;
			rc["signal"] = c.signal;
			rc["method"] = c.method;
			rc["source"] = c.source;
			rc["target"] = c.target;
			rc["binds"] = c.binds;
			rc["flags"] = c.flags;
			ret.push_back(rc);
		}
	}

	return ret;
}

Array Object::_get_incoming_connections() const {

	Array ret;
	int connections_amount = connections.size();
	for (int idx_conn = 0; idx_conn < connections_amount; idx_conn++) {
		Dictionary conn_data;
		conn_data["source"] = connections[idx_conn].source;
		conn_data["signal_name"] = connections[idx_conn].signal;
		conn_data["method_name"] = connections[idx_conn].method;
		ret.push_back(conn_data);
	}

	return ret;
}

void Object::get_signal_list(List<MethodInfo> *p_signals) const {

	if (!script.is_null()) {
		Ref<Script>(script)->get_script_signal_list(p_signals);
	}

	ClassDB::get_signal_list(get_class_name(), p_signals);
	//find maybe usersignals?
	const StringName *S = NULL;

	while ((S = signal_map.next(S))) {

		if (signal_map[*S].user.name != "") {
			//user signal
			p_signals->push_back(signal_map[*S].user);
		}
	}
}

void Object::get_all_signal_connections(List<Connection> *p_connections) const {

	const StringName *S = NULL;

	while ((S = signal_map.next(S))) {

		const Signal *s = &signal_map[*S];

		for (int i = 0; i < s->slot_map.size(); i++) {

			p_connections->push_back(s->slot_map.getv(i).conn);
		}
	}
}

void Object::get_signal_connection_list(const StringName &p_signal, List<Connection> *p_connections) const {

	const Signal *s = signal_map.getptr(p_signal);
	if (!s)
		return; //nothing

	for (int i = 0; i < s->slot_map.size(); i++)
		p_connections->push_back(s->slot_map.getv(i).conn);
}

bool Object::has_persistent_signal_connections() const {

	const StringName *S = NULL;

	while ((S = signal_map.next(S))) {

		const Signal *s = &signal_map[*S];

		for (int i = 0; i < s->slot_map.size(); i++) {

			if (s->slot_map.getv(i).conn.flags & CONNECT_PERSIST)
				return true;
		}
	}

	return false;
}

void Object::get_signals_connected_to_this(List<Connection> *p_connections) const {

	for (const List<Connection>::Element *E = connections.front(); E; E = E->next()) {
		p_connections->push_back(E->get());
	}
}

Error Object::connect(const StringName &p_signal, Object *p_to_object, const StringName &p_to_method, const Vector<Variant> &p_binds, uint32_t p_flags) {

	ERR_FAIL_NULL_V(p_to_object, ERR_INVALID_PARAMETER);

	Signal *s = signal_map.getptr(p_signal);
	if (!s) {
		bool signal_is_valid = ClassDB::has_signal(get_class_name(), p_signal);
		//check in script
		if (!signal_is_valid && !script.is_null() && Ref<Script>(script)->has_script_signal(p_signal))
			signal_is_valid = true;

		if (!signal_is_valid) {
			ERR_EXPLAIN("In Object of type '" + String(get_class()) + "': Attempt to connect nonexistent signal '" + p_signal + "' to method '" + p_to_object->get_class() + "." + p_to_method + "'");
			ERR_FAIL_COND_V(!signal_is_valid, ERR_INVALID_PARAMETER);
		}
		signal_map[p_signal] = Signal();
		s = &signal_map[p_signal];
	}

	Signal::Target target(p_to_object->get_instance_id(), p_to_method);
	if (s->slot_map.has(target)) {
		ERR_EXPLAIN("Signal '" + p_signal + "' is already connected to given method '" + p_to_method + "' in that object.");
		ERR_FAIL_COND_V(s->slot_map.has(target), ERR_INVALID_PARAMETER);
	}

	Signal::Slot slot;

	Connection conn;
	conn.source = this;
	conn.target = p_to_object;
	conn.method = p_to_method;
	conn.signal = p_signal;
	conn.flags = p_flags;
	conn.binds = p_binds;
	slot.conn = conn;
	slot.cE = p_to_object->connections.push_back(conn);
	s->slot_map[target] = slot;

	return OK;
}

bool Object::is_connected(const StringName &p_signal, Object *p_to_object, const StringName &p_to_method) const {

	ERR_FAIL_NULL_V(p_to_object, false);
	const Signal *s = signal_map.getptr(p_signal);
	if (!s) {
		bool signal_is_valid = ClassDB::has_signal(get_class_name(), p_signal);
		if (signal_is_valid)
			return false;

		if (!script.is_null() && Ref<Script>(script)->has_script_signal(p_signal))
			return false;

		ERR_EXPLAIN("Nonexistent signal: " + p_signal);
		ERR_FAIL_COND_V(!s, false);
	}

	Signal::Target target(p_to_object->get_instance_id(), p_to_method);

	return s->slot_map.has(target);
	//const Map<Signal::Target,Signal::Slot>::Element *E = s->slot_map.find(target);
	//return (E!=NULL);
}

void Object::disconnect(const StringName &p_signal, Object *p_to_object, const StringName &p_to_method) {

	ERR_FAIL_NULL(p_to_object);
	Signal *s = signal_map.getptr(p_signal);
	if (!s) {
		ERR_EXPLAIN("Nonexistent signal: " + p_signal);
		ERR_FAIL_COND(!s);
	}
	if (s->lock > 0) {
		ERR_EXPLAIN("Attempt to disconnect signal '" + p_signal + "' while emitting (locks: " + itos(s->lock) + ")");
		ERR_FAIL_COND(s->lock > 0);
	}

	Signal::Target target(p_to_object->get_instance_id(), p_to_method);

	if (!s->slot_map.has(target)) {
		ERR_EXPLAIN("Disconnecting nonexistent signal '" + p_signal + "', slot: " + itos(target._id) + ":" + target.method);
		ERR_FAIL();
	}

	p_to_object->connections.erase(s->slot_map[target].cE);
	s->slot_map.erase(target);

	if (s->slot_map.empty() && ClassDB::has_signal(get_class_name(), p_signal)) {
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

void Object::initialize_class() {

	static bool initialized = false;
	if (initialized)
		return;
	ClassDB::_add_class<Object>();
	_bind_methods();
	initialized = true;
}

StringName Object::tr(const StringName &p_message) const {

	if (!_can_translate || !TranslationServer::get_singleton())
		return p_message;

	return TranslationServer::get_singleton()->translate(p_message);
}

void Object::_clear_internal_resource_paths(const Variant &p_var) {

	switch (p_var.get_type()) {

		case Variant::OBJECT: {

			RES r = p_var;
			if (!r.is_valid())
				return;

			if (!r->get_path().begins_with("res://") || r->get_path().find("::") == -1)
				return; //not an internal resource

			Object *object = p_var;
			if (!object)
				return;

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

			for (List<Variant>::Element *E = keys.front(); E; E = E->next()) {

				_clear_internal_resource_paths(E->get());
				_clear_internal_resource_paths(d[E->get()]);
			}
		} break;
		default: {}
	}
}

#ifdef TOOLS_ENABLED
void Object::editor_set_section_unfold(const String &p_section, bool p_unfolded) {

	set_edited(true);
	if (p_unfolded)
		editor_section_folding.insert(p_section);
	else
		editor_section_folding.erase(p_section);
}

bool Object::editor_is_section_unfolded(const String &p_section) {

	return editor_section_folding.has(p_section);
}

#endif

void Object::clear_internal_resource_paths() {

	List<PropertyInfo> pinfo;

	get_property_list(&pinfo);

	for (List<PropertyInfo>::Element *E = pinfo.front(); E; E = E->next()) {

		_clear_internal_resource_paths(get(E->get().name));
	}
}

void Object::_bind_methods() {

	ClassDB::bind_method(D_METHOD("get_class"), &Object::get_class);
	ClassDB::bind_method(D_METHOD("is_class", "type"), &Object::is_class);
	ClassDB::bind_method(D_METHOD("set", "property", "value"), &Object::_set_bind);
	ClassDB::bind_method(D_METHOD("get", "property"), &Object::_get_bind);
	ClassDB::bind_method(D_METHOD("get_property_list"), &Object::_get_property_list_bind);
	ClassDB::bind_method(D_METHOD("get_method_list"), &Object::_get_method_list_bind);
	ClassDB::bind_method(D_METHOD("notification", "what", "reversed"), &Object::notification, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("get_instance_id"), &Object::get_instance_id);

	ClassDB::bind_method(D_METHOD("set_script", "script"), &Object::set_script);
	ClassDB::bind_method(D_METHOD("get_script"), &Object::get_script);

	ClassDB::bind_method(D_METHOD("set_meta", "name", "value"), &Object::set_meta);
	ClassDB::bind_method(D_METHOD("get_meta", "name"), &Object::get_meta);
	ClassDB::bind_method(D_METHOD("has_meta", "name"), &Object::has_meta);
	ClassDB::bind_method(D_METHOD("get_meta_list"), &Object::_get_meta_list_bind);

	//todo reimplement this per language so all 5 arguments can be called

	//ClassDB::bind_method(D_METHOD("call","method","arg1","arg2","arg3","arg4"),&Object::_call_bind,DEFVAL(Variant()),DEFVAL(Variant()),DEFVAL(Variant()),DEFVAL(Variant()));
	//ClassDB::bind_method(D_METHOD("call_deferred","method","arg1","arg2","arg3","arg4"),&Object::_call_deferred_bind,DEFVAL(Variant()),DEFVAL(Variant()),DEFVAL(Variant()),DEFVAL(Variant()));

	ClassDB::bind_method(D_METHOD("add_user_signal", "signal", "arguments"), &Object::_add_user_signal, DEFVAL(Array()));
	ClassDB::bind_method(D_METHOD("has_user_signal", "signal"), &Object::_has_user_signal);
	//ClassDB::bind_method(D_METHOD("emit_signal","signal","arguments"),&Object::_emit_signal,DEFVAL(Array()));

	{
		MethodInfo mi;
		mi.name = "emit_signal";
		mi.arguments.push_back(PropertyInfo(Variant::STRING, "signal"));

		ClassDB::bind_vararg_method(METHOD_FLAGS_DEFAULT, "emit_signal", &Object::_emit_signal, mi);
	}

	{
		MethodInfo mi;
		mi.name = "call";
		mi.arguments.push_back(PropertyInfo(Variant::STRING, "method"));

		ClassDB::bind_vararg_method(METHOD_FLAGS_DEFAULT, "call", &Object::_call_bind, mi);
	}

	{
		MethodInfo mi;
		mi.name = "call_deferred";
		mi.arguments.push_back(PropertyInfo(Variant::STRING, "method"));

		ClassDB::bind_vararg_method(METHOD_FLAGS_DEFAULT, "call_deferred", &Object::_call_deferred_bind, mi);
	}

	ClassDB::bind_method(D_METHOD("callv", "method", "arg_array"), &Object::callv);

	ClassDB::bind_method(D_METHOD("has_method", "method"), &Object::has_method);

	ClassDB::bind_method(D_METHOD("get_signal_list"), &Object::_get_signal_list);
	ClassDB::bind_method(D_METHOD("get_signal_connection_list", "signal"), &Object::_get_signal_connection_list);
	ClassDB::bind_method(D_METHOD("get_incoming_connections"), &Object::_get_incoming_connections);

	ClassDB::bind_method(D_METHOD("connect", "signal", "target", "method", "binds", "flags"), &Object::connect, DEFVAL(Array()), DEFVAL(0));
	ClassDB::bind_method(D_METHOD("disconnect", "signal", "target", "method"), &Object::disconnect);
	ClassDB::bind_method(D_METHOD("is_connected", "signal", "target", "method"), &Object::is_connected);

	ClassDB::bind_method(D_METHOD("set_block_signals", "enable"), &Object::set_block_signals);
	ClassDB::bind_method(D_METHOD("is_blocking_signals"), &Object::is_blocking_signals);
	ClassDB::bind_method(D_METHOD("property_list_changed_notify"), &Object::property_list_changed_notify);

	ClassDB::bind_method(D_METHOD("set_message_translation", "enable"), &Object::set_message_translation);
	ClassDB::bind_method(D_METHOD("can_translate_messages"), &Object::can_translate_messages);
	ClassDB::bind_method(D_METHOD("tr", "message"), &Object::tr);

	ClassDB::bind_method(D_METHOD("is_queued_for_deletion"), &Object::is_queued_for_deletion);

	ClassDB::add_virtual_method("Object", MethodInfo("free"), false);

	ADD_SIGNAL(MethodInfo("script_changed"));

	BIND_VMETHOD(MethodInfo("_notification", PropertyInfo(Variant::INT, "what")));
	BIND_VMETHOD(MethodInfo(Variant::BOOL, "_set", PropertyInfo(Variant::STRING, "property"), PropertyInfo(Variant::NIL, "value")));
#ifdef TOOLS_ENABLED
	MethodInfo miget("_get", PropertyInfo(Variant::STRING, "property"));
	miget.return_val.name = "Variant";
	BIND_VMETHOD(miget);

	MethodInfo plget("_get_property_list");

	plget.return_val.type = Variant::ARRAY;
	BIND_VMETHOD(plget);

#endif
	BIND_VMETHOD(MethodInfo("_init"));

	BIND_CONSTANT(NOTIFICATION_POSTINITIALIZE);
	BIND_CONSTANT(NOTIFICATION_PREDELETE);

	BIND_ENUM_CONSTANT(CONNECT_DEFERRED);
	BIND_ENUM_CONSTANT(CONNECT_PERSIST);
	BIND_ENUM_CONSTANT(CONNECT_ONESHOT);
}

void Object::call_deferred(const StringName &p_method, VARIANT_ARG_DECLARE) {

	MessageQueue::get_singleton()->push_call(this, p_method, VARIANT_ARG_PASS);
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

	for (List<PropertyInfo>::Element *E = plist.front(); E; E = E->next()) {

		if (!(E->get().usage & PROPERTY_USAGE_INTERNATIONALIZED))
			continue;

		String text = get(E->get().name);

		if (text == "")
			continue;

		p_strings->push_back(text);
	}
}

Variant::Type Object::get_static_property_type(const StringName &p_property, bool *r_valid) const {

	bool valid;
	Variant::Type t = ClassDB::get_property_type(get_class_name(), p_property, &valid);
	if (valid) {
		if (r_valid)
			*r_valid = true;
		return t;
	}

	if (get_script_instance()) {
		return get_script_instance()->get_property_type(p_property, r_valid);
	}
	if (r_valid)
		*r_valid = false;

	return Variant::NIL;
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

void *Object::get_script_instance_binding(int p_script_language_index) {
#ifdef DEBUG_ENABLED
	ERR_FAIL_INDEX_V(p_script_language_index, MAX_SCRIPT_INSTANCE_BINDINGS, NULL);
#endif

	//it's up to the script language to make this thread safe, if the function is called twice due to threads being out of syncro
	//just return the same pointer.
	//if you want to put a big lock in the entire function and keep allocated pointers in a map or something, feel free to do it
	//as it should not really affect performance much (won't be called too often), as in far most caes the condition below will be false afterwards

	if (!_script_instance_bindings[p_script_language_index]) {
		_script_instance_bindings[p_script_language_index] = ScriptServer::get_language(p_script_language_index)->alloc_instance_binding_data(this);
	}

	return _script_instance_bindings[p_script_language_index];
}

Object::Object() {

	_class_ptr = NULL;
	_block_signals = false;
	_predelete_ok = 0;
	_instance_ID = 0;
	_instance_ID = ObjectDB::add_instance(this);
	_can_translate = true;
	_is_queued_for_deletion = false;
	memset(_script_instance_bindings, 0, sizeof(void *) * MAX_SCRIPT_INSTANCE_BINDINGS);
	script_instance = NULL;
#ifdef TOOLS_ENABLED

	_edited = false;
	_edited_version = 0;
#endif

#ifdef DEBUG_ENABLED
	_lock_index.init(1);
#endif
}

Object::~Object() {

	if (script_instance)
		memdelete(script_instance);
	script_instance = NULL;

	List<Connection> sconnections;
	const StringName *S = NULL;

	while ((S = signal_map.next(S))) {

		Signal *s = &signal_map[*S];

		ERR_EXPLAIN("Attempt to delete an object in the middle of a signal emission from it");
		ERR_CONTINUE(s->lock > 0);

		for (int i = 0; i < s->slot_map.size(); i++) {

			sconnections.push_back(s->slot_map.getv(i).conn);
		}
	}

	for (List<Connection>::Element *E = sconnections.front(); E; E = E->next()) {

		Connection &c = E->get();
		ERR_CONTINUE(c.source != this); //bug?

		this->disconnect(c.signal, c.target, c.method);
	}

	while (connections.size()) {

		Connection c = connections.front()->get();
		c.source->disconnect(c.signal, c.target, c.method);
	}

	ObjectDB::remove_instance(this);
	_instance_ID = 0;
	_predelete_ok = 2;

	for (int i = 0; i < MAX_SCRIPT_INSTANCE_BINDINGS; i++) {
		if (_script_instance_bindings[i]) {
			ScriptServer::get_language(i)->free_instance_binding_data(_script_instance_bindings[i]);
		}
	}
}

bool predelete_handler(Object *p_object) {

	return p_object->_predelete();
}

void postinitialize_handler(Object *p_object) {

	p_object->_postinitialize();
}

HashMap<ObjectID, Object *> ObjectDB::instances;
ObjectID ObjectDB::instance_counter = 1;
HashMap<Object *, ObjectID, ObjectDB::ObjectPtrHash> ObjectDB::instance_checks;
ObjectID ObjectDB::add_instance(Object *p_object) {

	ERR_FAIL_COND_V(p_object->get_instance_id() != 0, 0);

	rw_lock->write_lock();
	instances[++instance_counter] = p_object;
#ifdef DEBUG_ENABLED
	instance_checks[p_object] = instance_counter;
#endif
	rw_lock->write_unlock();

	return instance_counter;
}

void ObjectDB::remove_instance(Object *p_object) {

	rw_lock->write_lock();

	instances.erase(p_object->get_instance_id());
#ifdef DEBUG_ENABLED
	instance_checks.erase(p_object);
#endif

	rw_lock->write_unlock();
}
Object *ObjectDB::get_instance(ObjectID p_instance_ID) {

	rw_lock->read_lock();
	Object **obj = instances.getptr(p_instance_ID);
	rw_lock->read_unlock();

	if (!obj)
		return NULL;
	return *obj;
}

void ObjectDB::debug_objects(DebugFunc p_func) {

	rw_lock->read_lock();

	const ObjectID *K = NULL;
	while ((K = instances.next(K))) {

		p_func(instances[*K]);
	}

	rw_lock->read_unlock();
}

void Object::get_argument_options(const StringName &p_function, int p_idx, List<String> *r_options) const {
}

int ObjectDB::get_object_count() {

	rw_lock->read_lock();
	int count = instances.size();
	rw_lock->read_unlock();

	return count;
}

RWLock *ObjectDB::rw_lock = NULL;

void ObjectDB::setup() {

	rw_lock = RWLock::create();
}

void ObjectDB::cleanup() {

	rw_lock->write_lock();
	if (instances.size()) {

		WARN_PRINT("ObjectDB Instances still exist!");
		if (OS::get_singleton()->is_stdout_verbose()) {
			const ObjectID *K = NULL;
			while ((K = instances.next(K))) {

				String node_name;
				if (instances[*K]->is_class("Node"))
					node_name = " - Node Name: " + String(instances[*K]->call("get_name"));
				if (instances[*K]->is_class("Resource"))
					node_name = " - Resource Name: " + String(instances[*K]->call("get_name")) + " Path: " + String(instances[*K]->call("get_path"));
				print_line("Leaked Instance: " + String(instances[*K]->get_class()) + ":" + itos(*K) + node_name);
			}
		}
	}
	instances.clear();
	instance_checks.clear();
	rw_lock->write_unlock();

	memdelete(rw_lock);
}
