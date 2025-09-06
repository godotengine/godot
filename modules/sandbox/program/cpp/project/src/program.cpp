/**************************************************************************/
/*  program.cpp                                                           */
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

#include <api.hpp>

static String last_string = "Hello, world!";
static Variant last_printed_string() {
	return last_string;
}
static Variant print_string(String str) {
	printf("String: %s\n", str.utf8().c_str());
	fflush(stdout);
	last_string = str;
	return Nil;
}

static long fib(long n, long acc, long prev) {
	if (n == 0)
		return acc;
	else
		return fib(n - 1, prev + acc, acc);
}

static Variant fibonacci(int n) {
	return fib(n, 0, 1);
}

int main() {
	print("Hello, world!");

	// The entire Godot API is available
	Sandbox sandbox = get_node<Sandbox>();
	print(sandbox.is_binary_translated()
					? "The current program is accelerated by binary translation."
					: "The current program is running in interpreter mode.");

	// Add public API
	ADD_API_FUNCTION(last_printed_string, "String", "", "Returns the last printed string");
	ADD_API_FUNCTION(print_string, "void", "String str", "Prints a string to the console");
	ADD_API_FUNCTION(fibonacci, "long", "int n", "Calculates the nth Fibonacci number");

	// Add a sandboxed property
	static int meaning_of_life = 42;
	add_property("meaning_of_life", Variant::Type::INT, 42, []() -> Variant { return meaning_of_life; }, [](Variant value) -> Variant { meaning_of_life = value; print("Set to: ", meaning_of_life); return Nil; });

	halt();
}
