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
#include "gdnative.h"

#include "io/resource_loader.h"
#include "io/resource_saver.h"

#include "core/os/os.h"

godot_variant cb_standard_varcall(void *handle, godot_string *p_procedure, godot_array *p_args) {
	if (handle == NULL) {
		ERR_PRINT("No valid library handle, can't call standard varcall procedure");
		godot_variant ret;
		godot_variant_new_nil(&ret);
		return ret;
	}

	void *library_proc;
	Error err = OS::get_singleton()->get_dynamic_library_symbol_handle(
			handle,
			*(String *)p_procedure,
			library_proc,
			true); // we roll our own message
	if (err != OK) {
		ERR_PRINT((String("GDNative procedure \"" + *(String *)p_procedure) + "\" does not exists and can't be called").utf8().get_data());
		godot_variant ret;
		godot_variant_new_nil(&ret);
		return ret;
	}

	godot_gdnative_procedure_fn proc;
	proc = (godot_gdnative_procedure_fn)library_proc;

	return proc(NULL, p_args);
}

GDNativeCallRegistry *GDNativeCallRegistry::singleton;

void register_gdnative_types() {

	ClassDB::register_class<GDNativeLibrary>();
	ClassDB::register_class<GDNative>();

	GDNativeCallRegistry::singleton = memnew(GDNativeCallRegistry);

	GDNativeCallRegistry::singleton->register_native_call_type("standard_varcall", cb_standard_varcall);
}

void unregister_gdnative_types() {
	memdelete(GDNativeCallRegistry::singleton);

	// This is for printing out the sizes of the core types

	/*
	print_line(String("array:\t")     + itos(sizeof(Array)));
	print_line(String("basis:\t")     + itos(sizeof(Basis)));
	print_line(String("color:\t")     + itos(sizeof(Color)));
	print_line(String("dict:\t" )     + itos(sizeof(Dictionary)));
	print_line(String("node_path:\t") + itos(sizeof(NodePath)));
	print_line(String("plane:\t")     + itos(sizeof(Plane)));
	print_line(String("poolarray:\t") + itos(sizeof(PoolByteArray)));
	print_line(String("quat:\t")      + itos(sizeof(Quat)));
	print_line(String("rect2:\t")     + itos(sizeof(Rect2)));
	print_line(String("rect3:\t")     + itos(sizeof(Rect3)));
	print_line(String("rid:\t")       + itos(sizeof(RID)));
	print_line(String("string:\t")    + itos(sizeof(String)));
	print_line(String("transform:\t") + itos(sizeof(Transform)));
	print_line(String("transfo2D:\t") + itos(sizeof(Transform2D)));
	print_line(String("variant:\t")   + itos(sizeof(Variant)));
	print_line(String("vector2:\t")   + itos(sizeof(Vector2)));
	print_line(String("vector3:\t")   + itos(sizeof(Vector3)));
	*/
}
