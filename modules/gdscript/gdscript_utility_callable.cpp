/**************************************************************************/
/*  gdscript_utility_callable.cpp                                         */
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

#include "gdscript_utility_callable.h"

#include "core/object/method_info.h"

#include <core/error/error_macros.h>
#include <core/string/print_string.h>
#include <core/variant/variant.h>

bool GDScriptUtilityCallable::compare_equal(const CallableCustom *p_a, const CallableCustom *p_b) {
	return p_a->hash() == p_b->hash();
}

bool GDScriptUtilityCallable::compare_less(const CallableCustom *p_a, const CallableCustom *p_b) {
	return p_a->hash() < p_b->hash();
}

uint32_t GDScriptUtilityCallable::hash() const {
	return h;
}

String GDScriptUtilityCallable::get_as_text() const {
	String scope;
	switch (type) {
		case TYPE_INVALID:
			scope = "<invalid scope>";
			break;
		case TYPE_GLOBAL:
			scope = "@GlobalScope";
			break;
		case TYPE_GDSCRIPT:
			scope = "@GDScript";
			break;
	}
	return vformat("%s::%s", scope, function_name);
}

CallableCustom::CompareEqualFunc GDScriptUtilityCallable::get_compare_equal_func() const {
	return compare_equal;
}

CallableCustom::CompareLessFunc GDScriptUtilityCallable::get_compare_less_func() const {
	return compare_less;
}

bool GDScriptUtilityCallable::is_valid() const {
	return type != TYPE_INVALID;
}

StringName GDScriptUtilityCallable::get_method() const {
	return function_name;
}

ObjectID GDScriptUtilityCallable::get_object() const {
	return ObjectID();
}

int GDScriptUtilityCallable::get_argument_count(bool &r_is_valid) const {
	switch (type) {
		case TYPE_INVALID:
			r_is_valid = false;
			return 0;
		case TYPE_GLOBAL:
			r_is_valid = true;
			return Variant::get_utility_function_argument_count(function_name);
		case TYPE_GDSCRIPT:
			r_is_valid = true;
			return GDScriptUtilityFunctions::get_function_argument_count(function_name);
	}
	ERR_FAIL_V_MSG(0, "Invalid type.");
}

void GDScriptUtilityCallable::get_arguments(Vector<Variant> &r_arguments) const {
	print_line("GDScriptUtilityCallable::get_arguments is called!");
	r_arguments.clear();
	MethodInfo method_info = MethodInfo();
	switch (type) {
		case TYPE_INVALID:
			break;
		case TYPE_GLOBAL:
			method_info = Variant::get_utility_function_info(function_name);
			break;
		case TYPE_GDSCRIPT:
			method_info = GDScriptUtilityFunctions::get_function_info(function_name);
			break;
	}

	ERR_FAIL_COND_MSG(method_info == MethodInfo(), vformat("Failed to get method info of \"%s\"", get_method()));
	Vector<PropertyInfo> arguments = method_info.arguments;
	for (PropertyInfo E : arguments) {
		r_arguments.push_back(E.operator Dictionary());
	}
}

void GDScriptUtilityCallable::call(const Variant **p_arguments, int p_argcount, Variant &r_return_value, Callable::CallError &r_call_error) const {
	switch (type) {
		case TYPE_INVALID:
			r_return_value = vformat(R"(Trying to call invalid utility function "%s".)", function_name);
			r_call_error.error = Callable::CallError::CALL_ERROR_INVALID_METHOD;
			r_call_error.argument = 0;
			r_call_error.expected = 0;
			break;
		case TYPE_GLOBAL:
			Variant::call_utility_function(function_name, &r_return_value, p_arguments, p_argcount, r_call_error);
			break;
		case TYPE_GDSCRIPT:
			gdscript_function(&r_return_value, p_arguments, p_argcount, r_call_error);
			break;
	}
}

GDScriptUtilityCallable::GDScriptUtilityCallable(const StringName &p_function_name) {
	function_name = p_function_name;
	if (GDScriptUtilityFunctions::function_exists(p_function_name)) {
		type = TYPE_GDSCRIPT;
		gdscript_function = GDScriptUtilityFunctions::get_function(p_function_name);
	} else if (Variant::has_utility_function(p_function_name)) {
		type = TYPE_GLOBAL;
	} else {
		ERR_PRINT(vformat(R"(Unknown utility function "%s".)", p_function_name));
	}
	h = p_function_name.hash();
}
