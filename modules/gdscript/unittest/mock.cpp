/*************************************************************************/
/*  mock.cpp                                                             */
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

#include "mock.h"

#include "core/script_language.h"

Mock::Mock() {
}

void Mock::bind_method(const String &name) {
}

Variant Mock::getvar(const Variant &p_key, bool *r_valid) const {
	if (r_valid)
		*r_valid = true;
	print_line(p_key);
	return "Hi";
}

void Mock::setvar(const Variant &p_key, const Variant &p_value, bool *r_valid) {
	if (r_valid)
		*r_valid = true;
	print_line(p_key);
}

Variant Mock::call(const StringName &p_method, const Variant **p_args, int p_argcount, Variant::CallError &r_error) {
	Variant result = Reference::call(p_method, p_args, p_argcount, r_error);
	if (r_error.error == Variant::CallError::CALL_OK) {
		return result;
	}
	r_error.error = Variant::CallError::CALL_OK;
	print_line(p_method);
	return 10;
}

void Mock::_bind_methods() {
	ClassDB::bind_method(D_METHOD("bind_method", "name"), &Mock::bind_method);
}
