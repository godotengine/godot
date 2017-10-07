/*************************************************************************/
/*  register_types.cpp                                                   */
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

#include "register_types.h"

#include "arvr_interface_gdnative.h"
#include "core/os/os.h"

void arvr_call_constructor(
		void *p_handle,
		godot_string *p_proc_name,
		void *p_data,
		int p_num_args,
		void **p_args,
		void *r_return) {
	if (p_handle == NULL) {
		ERR_PRINT("No valid library handle, can't call standard varcall procedure");
		return;
	}

	void *library_proc;
	Error err = OS::get_singleton()->get_dynamic_library_symbol_handle(
			p_handle,
			*(String *)p_proc_name,
			library_proc,
			true); // we roll our own message
	if (err != OK) {
		ERR_PRINT((String("GDNative procedure \"" + *(String *)p_proc_name) + "\" does not exists and can't be called").utf8().get_data());
		return;
	}

	void *(*proc)(godot_object *);
	proc = (void *(*)(godot_object *))library_proc;

	godot_object *this_object = (godot_object *)p_args[0];
	void *p = proc(this_object);

	void **return_ptr = (void **)r_return;

	*return_ptr = p;
};

void arvr_call_destructor(
		void *p_handle,
		godot_string *p_proc_name,
		void *p_data,
		int p_num_args,
		void **p_args,
		void *r_return) {
	if (p_handle == NULL) {
		ERR_PRINT("No valid library handle, can't call standard varcall procedure");
		return;
	}

	void *library_proc;
	Error err = OS::get_singleton()->get_dynamic_library_symbol_handle(
			p_handle,
			*(String *)p_proc_name,
			library_proc,
			true); // we roll our own message
	if (err != OK) {
		ERR_PRINT((String("GDNative procedure \"" + *(String *)p_proc_name) + "\" does not exists and can't be called").utf8().get_data());
		return;
	}

	void (*proc)(void *);
	proc = (void (*)(void *))library_proc;

	proc(p_data);
};

void arvr_return_string(
		void *p_handle,
		godot_string *p_proc_name,
		void *p_data,
		int p_num_args,
		void **p_args,
		void *r_return) {
	if (p_handle == NULL) {
		ERR_PRINT("No valid library handle, can't call standard varcall procedure");
		return;
	}

	void *library_proc;
	Error err = OS::get_singleton()->get_dynamic_library_symbol_handle(
			p_handle,
			*(String *)p_proc_name,
			library_proc,
			true); // we roll our own message
	if (err != OK) {
		ERR_PRINT((String("GDNative procedure \"" + *(String *)p_proc_name) + "\" does not exists and can't be called").utf8().get_data());
		return;
	}

	godot_string (*proc)(void *);
	proc = (godot_string(*)(void *))library_proc;

	godot_string s = proc(p_data);

	StringName *return_ptr = (StringName *)r_return;

	String *returned_string = (String *)&s;

	*return_ptr = *returned_string;

	godot_string_destroy(&s);
};

void arvr_return_int(
		void *p_handle,
		godot_string *p_proc_name,
		void *p_data,
		int p_num_args,
		void **p_args,
		void *r_return) {
	if (p_handle == NULL) {
		ERR_PRINT("No valid library handle, can't call standard varcall procedure");
		return;
	}

	void *library_proc;
	Error err = OS::get_singleton()->get_dynamic_library_symbol_handle(
			p_handle,
			*(String *)p_proc_name,
			library_proc,
			true); // we roll our own message
	if (err != OK) {
		ERR_PRINT((String("GDNative procedure \"" + *(String *)p_proc_name) + "\" does not exists and can't be called").utf8().get_data());
		return;
	}

	godot_int (*proc)(void *);
	proc = (godot_int(*)(void *))library_proc;

	godot_int i = proc(p_data);

	int *return_ptr = (int *)r_return;

	*return_ptr = i;
};

void arvr_return_bool(
		void *p_handle,
		godot_string *p_proc_name,
		void *p_data,
		int p_num_args,
		void **p_args,
		void *r_return) {
	if (p_handle == NULL) {
		ERR_PRINT("No valid library handle, can't call standard varcall procedure");
		return;
	}

	void *library_proc;
	Error err = OS::get_singleton()->get_dynamic_library_symbol_handle(
			p_handle,
			*(String *)p_proc_name,
			library_proc,
			true); // we roll our own message
	if (err != OK) {
		ERR_PRINT((String("GDNative procedure \"" + *(String *)p_proc_name) + "\" does not exists and can't be called").utf8().get_data());
		return;
	}

	godot_bool (*proc)(void *);
	proc = (godot_bool(*)(void *))library_proc;

	godot_bool b = proc(p_data);

	int *return_ptr = (int *)r_return;

	*return_ptr = b;
};

void arvr_set_bool(
		void *p_handle,
		godot_string *p_proc_name,
		void *p_data,
		int p_num_args,
		void **p_args,
		void *r_return) {
	if (p_handle == NULL) {
		ERR_PRINT("No valid library handle, can't call standard varcall procedure");
		return;
	}

	void *library_proc;
	Error err = OS::get_singleton()->get_dynamic_library_symbol_handle(
			p_handle,
			*(String *)p_proc_name,
			library_proc,
			true); // we roll our own message
	if (err != OK) {
		ERR_PRINT((String("GDNative procedure \"" + *(String *)p_proc_name) + "\" does not exists and can't be called").utf8().get_data());
		return;
	}

	void (*proc)(void *, bool);
	proc = (void (*)(void *, bool))library_proc;

	bool *set_bool = (bool *)p_args[0];
	proc(p_data, *set_bool);
};

void arvr_call_method(
		void *p_handle,
		godot_string *p_proc_name,
		void *p_data,
		int p_num_args,
		void **p_args,
		void *r_return) {
	if (p_handle == NULL) {
		ERR_PRINT("No valid library handle, can't call standard varcall procedure");
		return;
	}

	void *library_proc;
	Error err = OS::get_singleton()->get_dynamic_library_symbol_handle(
			p_handle,
			*(String *)p_proc_name,
			library_proc,
			true); // we roll our own message
	if (err != OK) {
		ERR_PRINT((String("GDNative procedure \"" + *(String *)p_proc_name) + "\" does not exists and can't be called").utf8().get_data());
		return;
	}

	godot_bool (*proc)(void *);
	proc = (godot_bool(*)(void *))library_proc;

	proc(p_data);
};

void arvr_return_vector2(
		void *p_handle,
		godot_string *p_proc_name,
		void *p_data,
		int p_num_args,
		void **p_args,
		void *r_return) {
	if (p_handle == NULL) {
		ERR_PRINT("No valid library handle, can't call standard varcall procedure");
		return;
	}

	void *library_proc;
	Error err = OS::get_singleton()->get_dynamic_library_symbol_handle(
			p_handle,
			*(String *)p_proc_name,
			library_proc,
			true); // we roll our own message
	if (err != OK) {
		ERR_PRINT((String("GDNative procedure \"" + *(String *)p_proc_name) + "\" does not exists and can't be called").utf8().get_data());
		return;
	}

	godot_vector2 (*proc)(void *);
	proc = (godot_vector2(*)(void *))library_proc;

	godot_vector2 v = proc(p_data);

	godot_vector2 *return_ptr = (godot_vector2 *)r_return;

	*return_ptr = v;
};

void arvr_return_transform_for_eye(
		void *p_handle,
		godot_string *p_proc_name,
		void *p_data,
		int p_num_args,
		void **p_args,
		void *r_return) {
	if (p_handle == NULL) {
		ERR_PRINT("No valid library handle, can't call standard varcall procedure");
		return;
	}

	void *library_proc;
	Error err = OS::get_singleton()->get_dynamic_library_symbol_handle(
			p_handle,
			*(String *)p_proc_name,
			library_proc,
			true); // we roll our own message
	if (err != OK) {
		ERR_PRINT((String("GDNative procedure \"" + *(String *)p_proc_name) + "\" does not exists and can't be called").utf8().get_data());
		return;
	}

	godot_transform (*proc)(void *, int, godot_transform *);
	proc = (godot_transform(*)(void *, int, godot_transform *))library_proc;

	int *eye = (int *)p_args[0];
	godot_transform *camera_transform = (godot_transform *)p_args[1];
	godot_transform t = proc(p_data, *eye, camera_transform);

	godot_transform *return_ptr = (godot_transform *)r_return;

	*return_ptr = t;
};

void arvr_call_fill_projection_for_eye(
		void *p_handle,
		godot_string *p_proc_name,
		void *p_data,
		int p_num_args,
		void **p_args,
		void *r_return) {
	if (p_handle == NULL) {
		ERR_PRINT("No valid library handle, can't call standard varcall procedure");
		return;
	}

	void *library_proc;
	Error err = OS::get_singleton()->get_dynamic_library_symbol_handle(
			p_handle,
			*(String *)p_proc_name,
			library_proc,
			true); // we roll our own message
	if (err != OK) {
		ERR_PRINT((String("GDNative procedure \"" + *(String *)p_proc_name) + "\" does not exists and can't be called").utf8().get_data());
		return;
	}

	void (*proc)(void *, real_t *, int, real_t, real_t, real_t);
	proc = (void (*)(void *, real_t *, int, real_t, real_t, real_t))library_proc;

	real_t *projection = (real_t *)p_args[0]; // <-- we'll be writing into this buffer, must have enough space for 16 floats!
	int *eye = (int *)p_args[1];
	real_t *aspect = (real_t *)p_args[2];
	real_t *zn = (real_t *)p_args[3];
	real_t *zf = (real_t *)p_args[4];

	proc(p_data, projection, *eye, *aspect, *zn, *zf);
};

void arvr_call_commit_for_eye(
		void *p_handle,
		godot_string *p_proc_name,
		void *p_data,
		int p_num_args,
		void **p_args,
		void *r_return) {
	if (p_handle == NULL) {
		ERR_PRINT("No valid library handle, can't call standard varcall procedure");
		return;
	}

	void *library_proc;
	Error err = OS::get_singleton()->get_dynamic_library_symbol_handle(
			p_handle,
			*(String *)p_proc_name,
			library_proc,
			true); // we roll our own message
	if (err != OK) {
		ERR_PRINT((String("GDNative procedure \"" + *(String *)p_proc_name) + "\" does not exists and can't be called").utf8().get_data());
		return;
	}

	void (*proc)(void *, int, godot_rid *, godot_rect2 *);
	proc = (void (*)(void *, int, godot_rid *, godot_rect2 *))library_proc;

	int *eye = (int *)p_args[0];
	godot_rid *rid = (godot_rid *)p_args[1];
	godot_rect2 *screen_rect = (godot_rect2 *)p_args[2];

	proc(p_data, *eye, rid, screen_rect);
};

void register_nativearvr_types() {
	GDNativeCallRegistry::singleton->register_native_raw_call_type("arvr_call_constructor", arvr_call_constructor);
	GDNativeCallRegistry::singleton->register_native_raw_call_type("arvr_call_destructor", arvr_call_destructor);
	GDNativeCallRegistry::singleton->register_native_raw_call_type("arvr_return_string", arvr_return_string);
	GDNativeCallRegistry::singleton->register_native_raw_call_type("arvr_return_int", arvr_return_int);
	GDNativeCallRegistry::singleton->register_native_raw_call_type("arvr_return_bool", arvr_return_bool);
	GDNativeCallRegistry::singleton->register_native_raw_call_type("arvr_set_bool", arvr_set_bool);
	GDNativeCallRegistry::singleton->register_native_raw_call_type("arvr_call_method", arvr_call_method);
	GDNativeCallRegistry::singleton->register_native_raw_call_type("arvr_return_vector2", arvr_return_vector2);
	GDNativeCallRegistry::singleton->register_native_raw_call_type("arvr_return_transform_for_eye", arvr_return_transform_for_eye);
	GDNativeCallRegistry::singleton->register_native_raw_call_type("arvr_call_fill_projection_for_eye", arvr_call_fill_projection_for_eye);
	GDNativeCallRegistry::singleton->register_native_raw_call_type("arvr_call_commit_for_eye", arvr_call_commit_for_eye);

	ClassDB::register_class<ARVRInterfaceGDNative>();
}

void unregister_nativearvr_types() {
}
