/*************************************************************************/
/*  godot.cpp                                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
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
#include "godot.h"

#include "class_db.h"
#include "error_macros.h"
#include "gdnative.h"
#include "global_config.h"
#include "global_constants.h"
#include "variant.h"

#ifdef __cplusplus
extern "C" {
#endif

extern "C" void _string_api_anchor();
extern "C" void _vector2_api_anchor();
extern "C" void _rect2_api_anchor();
extern "C" void _vector3_api_anchor();
extern "C" void _transform2d_api_anchor();
extern "C" void _plane_api_anchor();
extern "C" void _quat_api_anchor();
extern "C" void _basis_api_anchor();
extern "C" void _rect3_api_anchor();
extern "C" void _transform_api_anchor();
extern "C" void _color_api_anchor();
extern "C" void _node_path_api_anchor();
extern "C" void _rid_api_anchor();
extern "C" void _dictionary_api_anchor();
extern "C" void _array_api_anchor();
extern "C" void _pool_arrays_api_anchor();
extern "C" void _variant_api_anchor();

void _api_anchor() {

	_string_api_anchor();
	_vector2_api_anchor();
	_rect2_api_anchor();
	_vector3_api_anchor();
	_transform2d_api_anchor();
	_plane_api_anchor();
	_quat_api_anchor();
	_rect3_api_anchor();
	_basis_api_anchor();
	_transform_api_anchor();
	_color_api_anchor();
	_node_path_api_anchor();
	_rid_api_anchor();
	_dictionary_api_anchor();
	_array_api_anchor();
	_pool_arrays_api_anchor();
	_variant_api_anchor();
}

extern "C++" {
template <class a, class b>
_FORCE_INLINE_ a memcast(b v) {
	return *((a *)&v);
}
}

void GDAPI godot_object_destroy(godot_object *p_o) {
	memdelete((Object *)p_o);
}

// Singleton API

godot_object GDAPI *godot_global_get_singleton(char *p_name) {
	return (godot_object *)GlobalConfig::get_singleton()->get_singleton_object(String(p_name));
} // result shouldn't be freed

// MethodBind API

godot_method_bind GDAPI *godot_method_bind_get_method(const char *p_classname, const char *p_methodname) {

	MethodBind *mb = ClassDB::get_method(StringName(p_classname), StringName(p_methodname));
	// MethodBind *mb = ClassDB::get_method("Node", "get_name");
	return (godot_method_bind *)mb;
}

void GDAPI godot_method_bind_ptrcall(godot_method_bind *p_method_bind, godot_object *p_instance, const void **p_args, void *p_ret) {

	MethodBind *mb = (MethodBind *)p_method_bind;
	Object *o = (Object *)p_instance;
	mb->ptrcall(o, p_args, p_ret);
}

godot_variant GDAPI godot_method_bind_call(godot_method_bind *p_method_bind, godot_object *p_instance, const godot_variant **p_args, const int p_arg_count, godot_variant_call_error *p_call_error) {
	MethodBind *mb = (MethodBind *)p_method_bind;
	Object *o = (Object *)p_instance;
	const Variant **args = (const Variant **)p_args;

	godot_variant ret;
	godot_variant_new_nil(&ret);

	Variant *ret_val = (Variant *)&ret;

	Variant::CallError r_error;
	*ret_val = mb->call(o, args, p_arg_count, r_error);

	if (p_call_error) {
		p_call_error->error = (godot_variant_call_error_error)r_error.error;
		p_call_error->argument = r_error.argument;
		p_call_error->expected = (godot_variant_type)r_error.expected;
	}

	return ret;
}

// @Todo
/*
void GDAPI godot_method_bind_varcall(godot_method_bind *p_method_bind)
{

}
*/

// Script API

void GDAPI godot_script_register_class(const char *p_name, const char *p_base, godot_instance_create_func p_create_func, godot_instance_destroy_func p_destroy_func) {
	GDNativeLibrary *library = GDNativeLibrary::get_currently_initialized_library();
	if (!library) {
		ERR_EXPLAIN("Attempt to register script after initializing library!");
		ERR_FAIL();
	}
	library->_register_script(p_name, p_base, p_create_func, p_destroy_func);
}

void GDAPI godot_script_register_tool_class(const char *p_name, const char *p_base, godot_instance_create_func p_create_func, godot_instance_destroy_func p_destroy_func) {
	GDNativeLibrary *library = GDNativeLibrary::get_currently_initialized_library();
	if (!library) {
		ERR_EXPLAIN("Attempt to register script after initializing library!");
		ERR_FAIL();
	}
	library->_register_tool_script(p_name, p_base, p_create_func, p_destroy_func);
}

void GDAPI godot_script_register_method(const char *p_name, const char *p_function_name, godot_method_attributes p_attr, godot_instance_method p_method) {
	GDNativeLibrary *library = GDNativeLibrary::get_currently_initialized_library();
	if (!library) {
		ERR_EXPLAIN("Attempt to register script after initializing library!");
		ERR_FAIL();
	}
	library->_register_script_method(p_name, p_function_name, p_attr, p_method, MethodInfo());
}

void GDAPI godot_script_register_property(const char *p_name, const char *p_path, godot_property_attributes *p_attr, godot_property_set_func p_set_func, godot_property_get_func p_get_func) {
	GDNativeLibrary *library = GDNativeLibrary::get_currently_initialized_library();
	if (!library) {
		ERR_EXPLAIN("Attempt to register script after initializing library!");
		ERR_FAIL();
	}

	library->_register_script_property(p_name, p_path, p_attr, p_set_func, p_get_func);
}

void GDAPI godot_script_register_signal(const char *p_name, const godot_signal *p_signal) {
	GDNativeLibrary *library = GDNativeLibrary::get_currently_initialized_library();
	if (!library) {
		ERR_EXPLAIN("Attempt to register script after initializing library!");
		ERR_FAIL();
	}

	library->_register_script_signal(p_name, p_signal);
}

void GDAPI *godot_native_get_userdata(godot_object *p_instance) {
	Object *instance = (Object *)p_instance;
	if (!instance)
		return NULL;
	if (instance->get_script_instance() && instance->get_script_instance()->get_language() == GDNativeScriptLanguage::get_singleton()) {
		return ((GDNativeInstance *)instance->get_script_instance())->get_userdata();
	}
	return NULL;
}

godot_class_constructor GDAPI godot_get_class_constructor(const char *p_classname) {
	ClassDB::ClassInfo *class_info = ClassDB::classes.getptr(StringName(p_classname));
	if (class_info)
		return (godot_class_constructor)class_info->creation_func;
	return NULL;
}

godot_dictionary GDAPI godot_get_global_constants() {
	godot_dictionary constants;
	godot_dictionary_new(&constants);
	Dictionary *p_constants = (Dictionary *)&constants;
	const int constants_count = GlobalConstants::get_global_constant_count();
	for (int i = 0; i < constants_count; ++i) {
		const char *name = GlobalConstants::get_global_constant_name(i);
		int value = GlobalConstants::get_global_constant_value(i);
		(*p_constants)[name] = value;
	}
	return constants;
}

// System functions
void GDAPI *godot_alloc(int p_bytes) {
	return memalloc(p_bytes);
}

void GDAPI *godot_realloc(void *p_ptr, int p_bytes) {
	return memrealloc(p_ptr, p_bytes);
}

void GDAPI godot_free(void *p_ptr) {
	memfree(p_ptr);
}

void GDAPI godot_print_error(const char *p_description, const char *p_function, const char *p_file, int p_line) {
	_err_print_error(p_function, p_file, p_line, p_description, ERR_HANDLER_ERROR);
}

void GDAPI godot_print_warning(const char *p_description, const char *p_function, const char *p_file, int p_line) {
	_err_print_error(p_function, p_file, p_line, p_description, ERR_HANDLER_WARNING);
}

void GDAPI godot_print(const godot_string *p_message) {
	print_line(*(String *)p_message);
}

#ifdef __cplusplus
}
#endif
