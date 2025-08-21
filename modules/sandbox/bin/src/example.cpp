/**************************************************************************/
/*  example.cpp                                                           */
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

#include "api.hpp"

static Variant my_function(Vector4 v) {
	print("Arg: ", v);
	return 123;
}

static Variant my_function2(String s, Array a) {
	print("Args: ", s, a);
	return s;
}

static Variant _ready() {
	print("_ready called!");
	return Nil;
}

static Variant _process() {
	static int counter = 0;
	if (++counter % 100 == 0) {
		print("Process called " + std::to_string(counter) + " times");
	}
	return Nil;
}

static Vector4 my_vector4(1.0f, 2.0f, 3.0f, 4.0f);
static String my_string("Hello, World!");
int main() {
	ADD_PROPERTY(my_vector4, Variant::VECTOR4);
	ADD_PROPERTY(my_string, Variant::STRING);

	ADD_API_FUNCTION(my_function, "int", "Vector4 v");
	ADD_API_FUNCTION(my_function2, "Dictionary", "String s, Array a");

	ADD_API_FUNCTION(_ready, "void");
	ADD_API_FUNCTION(_process, "void");
}
