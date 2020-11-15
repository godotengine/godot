/*************************************************************************/
/*  test_object.h                                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef TEST_OBJECT_H
#define TEST_OBJECT_H

#include "core/object/object.h"

#include "thirdparty/doctest/doctest.h"

namespace TestObject {

TEST_CASE("[Object] Core getters") {
	Object object;

	CHECK_MESSAGE(
			object.is_class("Object"),
			"is_class() should return the expected value.");
	CHECK_MESSAGE(
			object.get_class() == "Object",
			"The returned class should match the expected value.");
	CHECK_MESSAGE(
			object.get_class_name() == "Object",
			"The returned class name should match the expected value.");
	CHECK_MESSAGE(
			object.get_class_static() == "Object",
			"The returned static class should match the expected value.");
	CHECK_MESSAGE(
			object.get_save_class() == "Object",
			"The returned save class should match the expected value.");
}

TEST_CASE("[Object] Metadata") {
	const String meta_path = "hello/world complex métadata\n\n\t\tpath";
	Object object;

	object.set_meta(meta_path, Color(0, 1, 0));
	CHECK_MESSAGE(
			Color(object.get_meta(meta_path)).is_equal_approx(Color(0, 1, 0)),
			"The returned object metadata after setting should match the expected value.");

	List<String> meta_list;
	object.get_meta_list(&meta_list);
	CHECK_MESSAGE(
			meta_list.size() == 1,
			"The metadata list should only contain 1 item after adding one metadata item.");

	object.remove_meta(meta_path);
	// Also try removing nonexistent metadata (it should do nothing, without printing an error message).
	object.remove_meta("I don't exist");
	ERR_PRINT_OFF;
	CHECK_MESSAGE(
			object.get_meta(meta_path) == Variant(),
			"The returned object metadata after removing should match the expected value.");
	ERR_PRINT_ON;

	List<String> meta_list2;
	object.get_meta_list(&meta_list2);
	CHECK_MESSAGE(
			meta_list2.size() == 0,
			"The metadata list should contain 0 items after removing all metadata items.");
}
} // namespace TestObject

#endif // TEST_OBJECT_H
