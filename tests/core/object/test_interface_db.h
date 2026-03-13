/**************************************************************************/
/*  test_interface_db.h                                                   */
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

#pragma once

#include "core/object/interface_db.h"
#include "core/object/object.h"
#include "scene/main/node.h"

#include "tests/test_macros.h"

namespace TestInterfaceDB {

TEST_CASE("[InterfaceDB] Register and query interface") {
	InterfaceInfo info;
	info.name = SNAME("ITestInterface");
	info.source_language = SNAME("Test");

	MethodInfo mi;
	mi.name = SNAME("do_thing");
	mi.arguments.push_back(PropertyInfo(Variant::FLOAT, "amount"));
	info.required_methods.push_back(mi);

	PropertyInfo pi(Variant::INT, "health");
	info.required_properties.push_back(pi);

	InterfaceDB::register_interface(info);

	CHECK(InterfaceDB::interface_exists(SNAME("ITestInterface")));
	CHECK_FALSE(InterfaceDB::interface_exists(SNAME("INonExistent")));

	InterfaceInfo retrieved = InterfaceDB::get_interface_info(SNAME("ITestInterface"));
	CHECK(retrieved.name == SNAME("ITestInterface"));
	CHECK(retrieved.required_methods.size() == 1);
	CHECK(retrieved.required_methods[0].name == SNAME("do_thing"));
	CHECK(retrieved.required_properties.size() == 1);
	CHECK(retrieved.required_properties[0].name == SNAME("health"));

	InterfaceDB::unregister_interface(SNAME("ITestInterface"));
	CHECK_FALSE(InterfaceDB::interface_exists(SNAME("ITestInterface")));
}

TEST_CASE("[InterfaceDB] Get interface list") {
	InterfaceInfo info_a;
	info_a.name = SNAME("IInterfaceA");
	InterfaceDB::register_interface(info_a);

	InterfaceInfo info_b;
	info_b.name = SNAME("IInterfaceB");
	InterfaceDB::register_interface(info_b);

	List<StringName> interfaces;
	InterfaceDB::get_interface_list(&interfaces);
	CHECK(interfaces.size() >= 2);

	bool found_a = false;
	bool found_b = false;
	for (const StringName &name : interfaces) {
		if (name == SNAME("IInterfaceA")) {
			found_a = true;
		}
		if (name == SNAME("IInterfaceB")) {
			found_b = true;
		}
	}
	CHECK(found_a);
	CHECK(found_b);

	InterfaceDB::unregister_interface(SNAME("IInterfaceA"));
	InterfaceDB::unregister_interface(SNAME("IInterfaceB"));
}

TEST_CASE("[InterfaceDB] Generation counter increments on registration changes") {
	uint64_t gen_before = InterfaceDB::get_generation();

	InterfaceInfo info;
	info.name = SNAME("IGenTest");
	InterfaceDB::register_interface(info);
	CHECK(InterfaceDB::get_generation() > gen_before);

	uint64_t gen_after_register = InterfaceDB::get_generation();
	InterfaceDB::unregister_interface(SNAME("IGenTest"));
	CHECK(InterfaceDB::get_generation() > gen_after_register);
}

TEST_CASE("[InterfaceDB] Structural matching with Object") {
	// Register an interface that requires has_method("get_class") — which every Object has.
	InterfaceInfo info;
	info.name = SNAME("IStructuralTest");

	MethodInfo mi;
	mi.name = SNAME("get_class");
	info.required_methods.push_back(mi);

	InterfaceDB::register_interface(info);

	// Create a Node (which inherits Object and has get_class).
	Node *node = memnew(Node);
	CHECK(InterfaceDB::object_structurally_satisfies(node, SNAME("IStructuralTest")));
	CHECK(InterfaceDB::object_implements_interface(node, SNAME("IStructuralTest")));

	memdelete(node);
	InterfaceDB::unregister_interface(SNAME("IStructuralTest"));
}

TEST_CASE("[InterfaceDB] Structural matching fails for missing method") {
	InterfaceInfo info;
	info.name = SNAME("IStructuralFail");

	MethodInfo mi;
	mi.name = SNAME("nonexistent_method_xyz_12345");
	info.required_methods.push_back(mi);

	InterfaceDB::register_interface(info);

	Node *node = memnew(Node);
	CHECK_FALSE(InterfaceDB::object_structurally_satisfies(node, SNAME("IStructuralFail")));
	CHECK_FALSE(InterfaceDB::object_implements_interface(node, SNAME("IStructuralFail")));

	memdelete(node);
	InterfaceDB::unregister_interface(SNAME("IStructuralFail"));
}

TEST_CASE("[InterfaceDB] Object with null returns false") {
	InterfaceInfo info;
	info.name = SNAME("INullTest");
	InterfaceDB::register_interface(info);

	CHECK_FALSE(InterfaceDB::object_implements_interface(nullptr, SNAME("INullTest")));

	InterfaceDB::unregister_interface(SNAME("INullTest"));
}

TEST_CASE("[InterfaceDB] Non-existent interface returns false") {
	Node *node = memnew(Node);
	CHECK_FALSE(InterfaceDB::object_implements_interface(node, SNAME("IDoesNotExist")));
	memdelete(node);
}

TEST_CASE("[InterfaceDB] Cache invalidation") {
	InterfaceInfo info;
	info.name = SNAME("ICacheTest");

	MethodInfo mi;
	mi.name = SNAME("nonexistent_cache_method");
	info.required_methods.push_back(mi);

	InterfaceDB::register_interface(info);

	Node *node = memnew(Node);
	// First check — should be false (method doesn't exist).
	CHECK_FALSE(InterfaceDB::object_implements_interface(node, SNAME("ICacheTest")));

	// Re-register with no required methods — now anything matches.
	InterfaceInfo info2;
	info2.name = SNAME("ICacheTest");
	InterfaceDB::register_interface(info2);

	// Cache should be invalidated; now it should match.
	CHECK(InterfaceDB::object_implements_interface(node, SNAME("ICacheTest")));

	memdelete(node);
	InterfaceDB::unregister_interface(SNAME("ICacheTest"));
}

} // namespace TestInterfaceDB
