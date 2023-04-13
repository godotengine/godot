/**************************************************************************/
/*  test_input_map.h                                                      */
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

#ifndef TEST_INPUT_MAP_H
#define TEST_INPUT_MAP_H

#include "tests/test_macros.h"

namespace TestInputMap {

TEST_CASE("[InputMap] Suggestions should return sane and expected results") {
	HashMap<String, LocalVector<String>> test_cases;
	test_cases.insert("ui_letf", { "ui_left" });
	test_cases.insert("ui_eltf", { "ui_left" });
	test_cases.insert("ui_pu", { "ui_up" });
	test_cases.insert("ui_donw", { "ui_down" });
	test_cases.insert("ui_do", { "ui_down", "ui_undo", "ui_redo" });

	InputMap *input_map = memnew(InputMap);
	input_map->load_default();
	for (const KeyValue<String, LocalVector<String>> &test : test_cases) {
		String error_message = input_map->suggest_actions(test.key);
		for (const String &suggestion : test.value) {
			CHECK(suggestion.is_subsequence_of(error_message));
		}
	}
}

} //namespace TestInputMap

#endif // TEST_INPUT_MAP_H
