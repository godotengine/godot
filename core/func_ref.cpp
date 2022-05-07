/*************************************************************************/
/*  func_ref.cpp                                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

Variant FuncRef::call_funcv(const Array &p_args) {
	ERR_FAIL_COND_V(id == 0, Variant());

	Object *obj = ObjectDB::get_instance(id);

	ERR_FAIL_COND_V(!obj, Variant());

	return obj->callv(function, p_args);
}

void FuncRef::set_instance(Object *p_obj) {
	ERR_FAIL_NULL(p_obj);
	id = p_obj->get_instance_id();
}

void FuncRef::set_function(const StringName &p_func) {
	function = p_func;
}

StringName FuncRef::get_function() {
	return function;
}

bool FuncRef::is_valid() const {
	if (id == 0) {
		return false;
	}

	Object *obj = ObjectDB::get_instance(id);
	if (!obj) {
		return false;
	}

	return obj->has_method(function);
}

void FuncRef::_bind_methods() {
	{
		MethodInfo mi;
		mi.name = "call_func";
		Vector<Variant> defargs;
		ClassDB::bind_vararg_method(METHOD_FLAGS_DEFAULT, "call_func", &FuncRef::call_func, mi, defargs);
	}

	ClassDB::bind_method(D_METHOD("call_funcv", "arg_array"), &FuncRef::call_funcv);

	ClassDB::bind_method(D_METHOD("set_instance", "instance"), &FuncRef::set_instance);
	ClassDB::bind_method(D_METHOD("is_valid"), &FuncRef::is_valid);

	ClassDB::bind_method(D_METHOD("set_function", "name"), &FuncRef::set_function);
	ClassDB::bind_method(D_METHOD("get_function"), &FuncRef::get_function);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "function"), "set_function", "get_function");
}

FuncRef::FuncRef() :
		id(0) {
}
