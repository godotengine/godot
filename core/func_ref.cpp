/*************************************************************************/
/*  func_ref.cpp                                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "func_ref.h"

Variant FuncRef::call_func(const Variant **p_args, int p_argcount, Variant::CallError &r_error) {

	if (id == 0) {
		r_error.error = Variant::CallError::CALL_ERROR_INSTANCE_IS_NULL;
		return Variant();
	}
	Object *obj = ObjectDB::get_instance(id);

	if (!obj) {
		r_error.error = Variant::CallError::CALL_ERROR_INSTANCE_IS_NULL;
		return Variant();
	}

	return obj->call(function, p_args, p_argcount, r_error);
}

void FuncRef::set_instance(Object *p_obj) {

	ERR_FAIL_NULL(p_obj);
	id = p_obj->get_instance_ID();
}
void FuncRef::set_function(const StringName &p_func) {

	function = p_func;
}

void FuncRef::_bind_methods() {

	{
		MethodInfo mi;
		mi.name = "call_func";
		Vector<Variant> defargs;
		for (int i = 0; i < 10; i++) {
			mi.arguments.push_back(PropertyInfo(Variant::NIL, "arg" + itos(i)));
			defargs.push_back(Variant());
		}
		ObjectTypeDB::bind_native_method(METHOD_FLAGS_DEFAULT, "call_func", &FuncRef::call_func, mi, defargs);
	}

	ObjectTypeDB::bind_method(_MD("set_instance", "instance"), &FuncRef::set_instance);
	ObjectTypeDB::bind_method(_MD("set_function", "name"), &FuncRef::set_function);
}

FuncRef::FuncRef() {

	id = 0;
}
