/*************************************************************************/
/*  test_object.h                                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "core/core_string_names.h"
#include "core/object/class_db.h"
#include "core/object/object.h"
#include "core/object/script_language.h"

#include "tests/test_macros.h"

// Declared in global namespace because of GDCLASS macro warning (Windows):
// "Unqualified friend declaration referring to type outside of the nearest enclosing namespace
// is a Microsoft extension; add a nested name specifier".
class _TestDerivedObject : public Object {
	GDCLASS(_TestDerivedObject, Object);

	int property_value;

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("set_property", "property"), &_TestDerivedObject::set_property);
		ClassDB::bind_method(D_METHOD("get_property"), &_TestDerivedObject::get_property);
		ADD_PROPERTY(PropertyInfo(Variant::INT, "property"), "set_property", "get_property");
	}

public:
	void set_property(int value) { property_value = value; }
	int get_property() const { return property_value; }
};

namespace TestObject {

class _MockScriptInstance : public ScriptInstance {
	StringName property_name = "NO_NAME";
	Variant property_value;

public:
	bool set(const StringName &p_name, const Variant &p_value) override {
		property_name = p_name;
		property_value = p_value;
		return true;
	}
	bool get(const StringName &p_name, Variant &r_ret) const override {
		if (property_name == p_name) {
			r_ret = property_value;
			return true;
		}
		return false;
	}
	void get_property_list(List<PropertyInfo> *p_properties) const override {
	}
	Variant::Type get_property_type(const StringName &p_name, bool *r_is_valid) const override {
		return Variant::PACKED_FLOAT32_ARRAY;
	}
	void get_method_list(List<MethodInfo> *p_list) const override {
	}
	bool has_method(const StringName &p_method) const override {
		return false;
	}
	Variant call(const StringName &p_method, const Variant **p_args, int p_argcount, Callable::CallError &r_error) override {
		return Variant();
	}
	void notification(int p_notification) override {
	}
	Ref<Script> get_script() const override {
		return Ref<Script>();
	}
	const Vector<Multiplayer::RPCConfig> get_rpc_methods() const override {
		return Vector<Multiplayer::RPCConfig>();
	}
	ScriptLanguage *get_language() override {
		return nullptr;
	}
};

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

	List<String> inheritance_list;
	object.get_inheritance_list_static(&inheritance_list);
	CHECK_MESSAGE(
			inheritance_list.size() == 1,
			"The inheritance list should consist of Object only");
	CHECK_MESSAGE(
			inheritance_list[0] == "Object",
			"The inheritance list should consist of Object only");
}

TEST_CASE("[Object] Metadata") {
	const String meta_path = "hello/world complex métadata\n\n\t\tpath";
	Object object;

	object.set_meta(meta_path, Color(0, 1, 0));
	CHECK_MESSAGE(
			Color(object.get_meta(meta_path)).is_equal_approx(Color(0, 1, 0)),
			"The returned object metadata after setting should match the expected value.");

	List<StringName> meta_list;
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

	List<StringName> meta_list2;
	object.get_meta_list(&meta_list2);
	CHECK_MESSAGE(
			meta_list2.size() == 0,
			"The metadata list should contain 0 items after removing all metadata items.");
}

TEST_CASE("[Object] Construction") {
	Object object;

	CHECK_MESSAGE(
			!object.is_ref_counted(),
			"Object is not a RefCounted.");

	Object *p_db = ObjectDB::get_instance(object.get_instance_id());
	CHECK_MESSAGE(
			p_db == &object,
			"The database pointer returned by the object id should reference same object.");
}

TEST_CASE("[Object] Script instance property setter") {
	Object object;
	_MockScriptInstance *script_instance = memnew(_MockScriptInstance);
	object.set_script_instance(script_instance);

	bool valid = false;
	object.set("some_name", 100, &valid);
	CHECK(valid);
	Variant actual_value;
	CHECK_MESSAGE(
			script_instance->get("some_name", actual_value),
			"The assigned script instance should successfully retrieve value by name.");
	CHECK_MESSAGE(
			actual_value == Variant(100),
			"The returned value should equal the one which was set by the object.");
}

TEST_CASE("[Object] Script instance property getter") {
	Object object;
	_MockScriptInstance *script_instance = memnew(_MockScriptInstance);
	script_instance->set("some_name", 100); // Make sure script instance has the property
	object.set_script_instance(script_instance);

	bool valid = false;
	const Variant &actual_value = object.get("some_name", &valid);
	CHECK(valid);
	CHECK_MESSAGE(
			actual_value == Variant(100),
			"The returned value should equal the one which was set by the script instance.");
}

TEST_CASE("[Object] Built-in property setter") {
	GDREGISTER_CLASS(_TestDerivedObject);
	_TestDerivedObject derived_object;

	bool valid = false;
	derived_object.set("property", 100, &valid);
	CHECK(valid);
	CHECK_MESSAGE(
			derived_object.get_property() == 100,
			"The property value should equal the one which was set with built-in setter.");
}

TEST_CASE("[Object] Built-in property getter") {
	GDREGISTER_CLASS(_TestDerivedObject);
	_TestDerivedObject derived_object;
	derived_object.set_property(100);

	bool valid = false;
	const Variant &actual_value = derived_object.get("property", &valid);
	CHECK(valid);
	CHECK_MESSAGE(
			actual_value == Variant(100),
			"The returned value should equal the one which was set with built-in setter.");
}

TEST_CASE("[Object] Script property setter") {
	Object object;
	Variant script;

	bool valid = false;
	object.set(CoreStringNames::get_singleton()->_script, script, &valid);
	CHECK(valid);
	CHECK_MESSAGE(
			object.get_script() == script,
			"The object script should be equal to the assigned one.");
}

TEST_CASE("[Object] Script property getter") {
	Object object;
	Variant script;
	object.set_script(script);

	bool valid = false;
	const Variant &actual_value = object.get(CoreStringNames::get_singleton()->_script, &valid);
	CHECK(valid);
	CHECK_MESSAGE(
			actual_value == script,
			"The returned value should be equal to the assigned script.");
}

TEST_CASE("[Object] Absent name setter") {
	Object object;

	bool valid = true;
	object.set("absent_name", 100, &valid);
	CHECK(!valid);
}

TEST_CASE("[Object] Absent name getter") {
	Object object;

	bool valid = true;
	const Variant &actual_value = object.get("absent_name", &valid);
	CHECK(!valid);
	CHECK_MESSAGE(
			actual_value == Variant(),
			"The returned value should equal nil variant.");
}
} // namespace TestObject

#endif // TEST_OBJECT_H
