/*************************************************************************/
/*  test_sort.cpp                                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2018 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2018 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "test_sort.h"

#include "core/array.h"
#include "core/os/os.h"
#include "core/sort_array.h"
#include "core/typedefs.h"
#include "core/ustring.h"

namespace TestSort {

enum Order {
	sorted,
	inverted,
	random
};

// Adjustable Test Parameters

int test_size = 10000;
Order initial_order = random;

// Helper Functions

void print_array(void *arr_ptr, String test_type) {
	if (OS::get_singleton()->is_stdout_verbose()) {
		if (test_type == "int") {
			int *test_arr = (int *)arr_ptr;
			for (int i = 0; i < test_size; i++) {
				OS::get_singleton()->print("%d\n", test_arr[i]);
			}
		} else if (test_type == "String") {
			String *test_arr = (String *)arr_ptr;
			for (int i = 0; i < test_size; i++) {
				OS::get_singleton()->print("%ls\n", test_arr[i].c_str());
			}
		}

		OS::get_singleton()->print("\n");
	}
}

String *letter_array() {
	String *test_values = memnew_arr(String, 26);
	test_values[0] = "a";
	test_values[1] = "b";
	test_values[2] = "c";
	test_values[3] = "d";
	test_values[4] = "e";
	test_values[5] = "f";
	test_values[6] = "g";
	test_values[7] = "h";
	test_values[8] = "i";
	test_values[9] = "j";
	test_values[10] = "k";
	test_values[11] = "l";
	test_values[12] = "m";
	test_values[13] = "n";
	test_values[14] = "o";
	test_values[15] = "p";
	test_values[16] = "q";
	test_values[17] = "r";
	test_values[18] = "s";
	test_values[19] = "t";
	test_values[20] = "u";
	test_values[21] = "v";
	test_values[22] = "w";
	test_values[23] = "x";
	test_values[24] = "y";
	test_values[25] = "z";

	return test_values;
}

int *create_test_array_int() {
	int *test_arr = memnew_arr(int, test_size);

	if (initial_order == sorted) {
		for (int i = 0; i < test_size; i++) {
			test_arr[i] = i;
		}
	} else if (initial_order == inverted) {
		for (int i = 0; i < test_size; i++) {
			test_arr[i] = test_size - 1 - i;
		}
	} else if (initial_order == random) {
		for (int i = 0; i < test_size; i++) {
			test_arr[i] = i;
		}

		for (int i = 0; i < test_size; i++) {
			int swap_index = ((i + 17) * 13) % test_size;
			SWAP(test_arr[i], test_arr[swap_index]);
		}
	}

	print_array(test_arr, "int");
	return test_arr;
}

String *create_test_array_string() {
	String *test_arr = memnew_arr(String, test_size);

	String *test_values = letter_array();

	if (initial_order == sorted) {
		test_arr[0] = test_values[0];

		int max_element_length = (test_size / 26) + 1;
		int index = 0;
		for (int i = 1; i < test_size; i++) {
			if (i % max_element_length == 0) {
				index++;
				test_arr[i] = test_values[index];
			} else {
				test_arr[i] = test_arr[i - 1] + test_values[index];
			}
		}
	} else if (initial_order == inverted) {
		test_arr[test_size - 1] = test_values[0];

		int max_element_length = (test_size / 26) + 1;
		int index = 0;
		for (int i = test_size - 2; i >= 0; i--) {
			int letter_check = (i - (test_size - 1)) % max_element_length;
			if (letter_check == 0) {
				index++;
				test_arr[i] = test_values[index];
			} else {
				test_arr[i] = test_arr[i + 1] + test_values[index];
			}
		}
	} else if (initial_order == random) {
		test_arr[0] = test_values[0];

		int max_element_length = (test_size / 26) + 1;
		int index = 0;
		for (int i = 1; i < test_size; i++) {
			if (i % max_element_length == 0) {
				index++;
				test_arr[i] = test_values[index];
			} else {
				test_arr[i] = test_arr[i - 1] + test_values[index];
			}
		}

		for (int i = 0; i < test_size; i++) {
			int swap_index = ((i + 17) * 13) % test_size;
			SWAP(test_arr[i], test_arr[swap_index]);
		}
	}

	print_array(test_arr, "String");
	return test_arr;
}

// Test Functions

bool test_sort_int() {
	OS::get_singleton()->print("Test Sorting Ints:\n");

	int *sort_arr = create_test_array_int();

	SortArray<int> sorter;
	sorter.sort(sort_arr, test_size);

	print_array(sort_arr, "int");

	OS::get_singleton()->print("Size: %d\n", test_size);
	OS::get_singleton()->print("End\n");

	for (int i = 0; i < (test_size - 1); i++) {
		if (sort_arr[i + 1] < sort_arr[i])
			return 0;
	}

	return 1;
}

bool test_sort_string() {
	OS::get_singleton()->print("Test Sorting Strings:\n");

	String *sort_arr = create_test_array_string();

	SortArray<String> sorter;
	sorter.sort(sort_arr, test_size);

	print_array(sort_arr, "String");

	OS::get_singleton()->print("Size: %d\n", test_size);
	OS::get_singleton()->print("End\n");

	for (int i = 0; i < (test_size - 1); i++) {
		if (sort_arr[i + 1] < sort_arr[i])
			return 0;
	}

	return 1;
}

bool test_select_int() {
	OS::get_singleton()->print("Test Select Int:\n");

	int select_index = test_size / 19;
	int *test_arr = create_test_array_int();

	SortArray<int> selecter;
	selecter.nth_element(test_arr, test_size, select_index);

	print_array(test_arr, "int");

	OS::get_singleton()->print("Size: %d\n", test_size);
	OS::get_singleton()->print("End\n");

	if (test_arr[select_index] == select_index)
		return 1;

	return 0;
}

bool test_select_string() {
	OS::get_singleton()->print("Test Select String:\n");

	int select_index = test_size / 19;
	String *test_arr = create_test_array_string();

	SortArray<String> selecter;
	selecter.nth_element(test_arr, test_size, select_index);

	print_array(test_arr, "String");

	OS::get_singleton()->print("Size: %d\n", test_size);
	OS::get_singleton()->print("End\n");

	int max_element_length = (test_size / 26) + 1;
	int letter_index = 0 + (select_index / max_element_length);
	int len = (select_index % max_element_length) + 1;
	String letter = letter_array()[letter_index];
	String correct_value = "";
	for (int i = 0; i < len; i++) {
		correct_value = correct_value + letter;
	}

	if (test_arr[select_index] == correct_value)
		return 1;

	return 0;
}

// Main Test Call

typedef bool (*TestFunc)(void);

TestFunc test_funcs[] = {
	test_sort_int,
	test_sort_string,
	test_select_int,
	test_select_string,
	0
};

MainLoop *test() {
	int count = 0;
	int passed = 0;

	while (test_funcs[count]) {
		bool pass = test_funcs[count]();
		if (pass)
			passed++;

		OS::get_singleton()->print("\t%s\n\n", pass ? "PASS" : "FAILED");
		count++;
	}

	OS::get_singleton()->print("Passed %i of %i tests\n", passed, count);

	return NULL;
}

} // namespace TestSort
