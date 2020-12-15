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

TEST_CASE("[Object] IAPI methods") {
	Object object;
    bool *rvalid = memnew(bool(false));

    object.set(StringName("script"), "script_source", rvalid);
	CHECK_MESSAGE(
			*rvalid == true,
			"set() should update the passed bool to the expected value.");

    *rvalid = false;
	CHECK_MESSAGE(
			object.get(StringName("script"), rvalid) == "script_source",
			"get() should return the expected value.");
	CHECK_MESSAGE(
			*rvalid == true,
			"get() should update the passed bool to the expected value.");
    
    Vector<StringName> p_names;
    object.set_indexed(p_names, "value", rvalid);
    CHECK_MESSAGE(
		*rvalid == false,
		"set_indexed() should update the passed bool to the expected value for empty Vector<StringName>.");

    *rvalid = true;
    CHECK_MESSAGE(
		object.get_indexed(p_names, rvalid) == Variant(),
		"get_indexed() should return the expected value for empty Vector<StringName>.");
    CHECK_MESSAGE(
		*rvalid == false,
		"get_indexed() should update the passed bool to the expected value for empty Vector<StringName>.");

    p_names.push_back(StringName("script"));
    object.set_indexed(p_names, "indexed_script_source", rvalid);
    CHECK_MESSAGE(
		*rvalid == true,
		"set_indexed() should update the passed bool to the expected value for valid Vector<StringName> of size 1.");

    *rvalid = false;
    CHECK_MESSAGE(
		object.get_indexed(p_names, rvalid) == "indexed_script_source",
		"get_indexed() should return the expected value for valid Vector<StringName> of size 1.");
    CHECK_MESSAGE(
		*rvalid == true,
		"get_indexed() should update the passed bool to the expected value for valid Vector<StringName> of size 1.");

    CHECK_MESSAGE(
		object.has_method(StringName("free")) == true,
		"has_method should return the expected value for StringName('free')");

    CHECK_MESSAGE(
		object.has_method(StringName("_to_string")) == false,
		"has_method should return the expected value for non-method");

    CHECK_MESSAGE(
		object.to_string() == "[Object:" + itos(object.get_instance_id()) + "]",
		"to_string should return the expected value.");

    *rvalid = false;
    CHECK_MESSAGE(
		object.getvar(StringName("script"), rvalid) == "indexed_script_source",
		"getvar should return the expected value.");
    CHECK_MESSAGE(
		*rvalid == true,
		"getvar should update the passed bool to the expected value.");

    *rvalid = false;
    object.setvar(StringName("script"), "var_script", rvalid);
    CHECK_MESSAGE(
		*rvalid == true,
		"setvar should update the passed bool to the expected value.");
    CHECK_MESSAGE(
		object.getvar(StringName("script"), rvalid) == "var_script",
		"getvar should return the expected value.");

    object.setvar(StringName("non-CoreStringName"), "nonvalid_script", rvalid);
    CHECK_MESSAGE(
		*rvalid == false,
		"setvar should update the passed bool to the expected value.");

	memdelete(rvalid);
}

TEST_CASE("[Object] Script setter and getter") {
	Object object;

    object.set_script("script_source");
	CHECK_MESSAGE(
			object.get_script() == "script_source",
			"get_script() should return the expected value.");
}
} // namespace TestObject

#endif // TEST_OBJECT_H
