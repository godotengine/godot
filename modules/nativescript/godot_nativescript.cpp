/*************************************************************************/
/*  godot_nativescript.cpp                                               */
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
#include "godot_nativescript.h"

#include "nativescript.h"

#include "class_db.h"
#include "error_macros.h"
#include "gdnative.h"
#include "global_constants.h"
#include "project_settings.h"
#include "variant.h"

#ifdef __cplusplus
extern "C" {
#endif

extern "C" void _native_script_hook() {
}

#define NSL NativeScriptLanguage::get_singleton()

// Script API

void GDAPI godot_nativescript_register_class(void *p_gdnative_handle, const char *p_name, const char *p_base, godot_instance_create_func p_create_func, godot_instance_destroy_func p_destroy_func) {

	String *s = (String *)p_gdnative_handle;

	Map<StringName, NativeScriptDesc> *classes = &NSL->library_classes[*s];

	NativeScriptDesc desc;

	desc.create_func = p_create_func;
	desc.destroy_func = p_destroy_func;
	desc.is_tool = false;

	desc.base = p_base;

	if (classes->has(p_base)) {
		desc.base_data = &(*classes)[p_base];
		desc.base_native_type = desc.base_data->base_native_type;
	} else {
		desc.base_data = NULL;
		desc.base_native_type = p_base;
	}

	classes->insert(p_name, desc);
}

void GDAPI godot_nativescript_register_tool_class(void *p_gdnative_handle, const char *p_name, const char *p_base, godot_instance_create_func p_create_func, godot_instance_destroy_func p_destroy_func) {

	String *s = (String *)p_gdnative_handle;

	Map<StringName, NativeScriptDesc> *classes = &NSL->library_classes[*s];

	NativeScriptDesc desc;

	desc.create_func = p_create_func;
	desc.destroy_func = p_destroy_func;
	desc.is_tool = true;
	desc.base = p_base;

	if (classes->has(p_base)) {
		desc.base_data = &(*classes)[p_base];
		desc.base_native_type = desc.base_data->base_native_type;
	} else {
		desc.base_data = NULL;
		desc.base_native_type = p_base;
	}

	classes->insert(p_name, desc);
}

void GDAPI godot_nativescript_register_method(void *p_gdnative_handle, const char *p_name, const char *p_function_name, godot_method_attributes p_attr, godot_instance_method p_method) {

	String *s = (String *)p_gdnative_handle;

	Map<StringName, NativeScriptDesc>::Element *E = NSL->library_classes[*s].find(p_name);

	if (!E) {
		ERR_EXPLAIN("Attempt to register method on non-existant class!");
		ERR_FAIL();
	}

	NativeScriptDesc::Method method;
	method.method = p_method;
	method.rpc_mode = p_attr.rpc_type;
	method.info = MethodInfo(p_function_name);

	E->get().methods.insert(p_function_name, method);
}

void GDAPI godot_nativescript_register_property(void *p_gdnative_handle, const char *p_name, const char *p_path, godot_property_attributes *p_attr, godot_property_set_func p_set_func, godot_property_get_func p_get_func) {

	String *s = (String *)p_gdnative_handle;

	Map<StringName, NativeScriptDesc>::Element *E = NSL->library_classes[*s].find(p_name);

	if (!E) {
		ERR_EXPLAIN("Attempt to register method on non-existant class!");
		ERR_FAIL();
	}

	NativeScriptDesc::Property property;
	property.default_value = *(Variant *)&p_attr->default_value;
	property.getter = p_get_func;
	property.rset_mode = p_attr->rset_type;
	property.setter = p_set_func;
	property.info = PropertyInfo((Variant::Type)p_attr->type,
			p_path,
			(PropertyHint)p_attr->hint,
			*(String *)&p_attr->hint_string,
			(PropertyUsageFlags)p_attr->usage);

	E->get().properties.insert(p_path, property);
}

void GDAPI godot_nativescript_register_signal(void *p_gdnative_handle, const char *p_name, const godot_signal *p_signal) {

	String *s = (String *)p_gdnative_handle;

	Map<StringName, NativeScriptDesc>::Element *E = NSL->library_classes[*s].find(p_name);

	if (!E) {
		ERR_EXPLAIN("Attempt to register method on non-existant class!");
		ERR_FAIL();
	}

	List<PropertyInfo> args;
	Vector<Variant> default_args;

	for (int i = 0; i < p_signal->num_args; i++) {
		PropertyInfo info;

		godot_signal_argument arg = p_signal->args[i];

		info.hint = (PropertyHint)arg.hint;
		info.hint_string = *(String *)&arg.hint_string;
		info.name = *(String *)&arg.name;
		info.type = (Variant::Type)arg.type;
		info.usage = (PropertyUsageFlags)arg.usage;

		args.push_back(info);
	}

	for (int i = 0; i < p_signal->num_default_args; i++) {
		Variant *v;
		godot_signal_argument attrib = p_signal->args[i];

		v = (Variant *)&attrib.default_value;

		default_args.push_back(*v);
	}

	MethodInfo method_info;
	method_info.name = *(String *)&p_signal->name;
	method_info.arguments = args;
	method_info.default_arguments = default_args;

	NativeScriptDesc::Signal signal;
	signal.signal = method_info;

	E->get().signals_.insert(*(String *)&p_signal->name, signal);
}

void GDAPI *godot_nativescript_get_userdata(godot_object *p_instance) {
	Object *instance = (Object *)p_instance;
	if (!instance)
		return NULL;
	if (instance->get_script_instance() && instance->get_script_instance()->get_language() == NativeScriptLanguage::get_singleton()) {
		return ((NativeScriptInstance *)instance->get_script_instance())->userdata;
	}
	return NULL;
}

#ifdef __cplusplus
}
#endif
