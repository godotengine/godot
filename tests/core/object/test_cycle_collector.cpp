/**************************************************************************/
/*  test_cycle_collector.cpp                                              */
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

#include "tests/test_macros.h"

TEST_FORCE_LINK(test_cycle_collector)

#include "core/object/cycle_collector.h"
#include "core/object/ref_counted.h"
#include "core/object/script_instance.h"
#include "core/variant/array.h"

namespace TestCycleCollector {

// A minimal ScriptInstance that stands in for GDScriptInstance: it exposes a plain, directly
// settable member list through the same get_gc_member_count/get_gc_member/clear_gc_member
// interface GDScriptInstance implements, without needing the gdscript module as a test
// dependency.
class MockScriptInstance : public ScriptInstance {
public:
	Vector<Variant> members;

	bool set(const StringName &p_name, const Variant &p_value) override { return false; }
	bool get(const StringName &p_name, Variant &r_ret) const override { return false; }
	void get_property_list(List<PropertyInfo> *r_properties) const override {}
	Variant::Type get_property_type(const StringName &p_name, bool *r_is_valid) const override {
		if (r_is_valid) {
			*r_is_valid = false;
		}
		return Variant::NIL;
	}
	void validate_property(PropertyInfo &p_property) const override {}
	bool property_can_revert(const StringName &p_name) const override { return false; }
	bool property_get_revert(const StringName &p_name, Variant &r_ret) const override { return false; }
	void get_method_list(List<MethodInfo> *r_list) const override {}
	bool has_method(const StringName &p_method) const override { return false; }
	Variant callp(const StringName &p_method, const Variant **p_args, int p_argcount, Callable::CallError &r_error) override { return Variant(); }
	void notification(int p_notification, bool p_reversed = false) override {}
	Ref<Script> get_script() const override { return Ref<Script>(); }
	ScriptLanguage *get_language() override { return nullptr; }

	int get_gc_member_count() const override { return members.size(); }
	Variant get_gc_member(int p_index) const override { return members[p_index]; }
	void clear_gc_member(int p_index) override { members.write[p_index] = Variant(); }
};

// A RefCounted that always carries a MockScriptInstance, mimicking a GDScript object instance
// closely enough to exercise the real CycleCollector traversal/finalization code paths.
class MockRefCounted : public RefCounted {
	GDCLASS(MockRefCounted, RefCounted);

protected:
	static void _bind_methods() {}

public:
	static int alive_count;

	MockScriptInstance *script = nullptr;

	MockRefCounted() {
		script = memnew(MockScriptInstance);
		set_script_instance(script);
		alive_count++;
	}

	~MockRefCounted() {
		alive_count--;
	}
};

int MockRefCounted::alive_count = 0;

TEST_CASE("[CycleCollector] A two-object reference cycle is collected") {
	MockRefCounted::alive_count = 0;
	{
		Ref<MockRefCounted> a = memnew(MockRefCounted);
		Ref<MockRefCounted> b = memnew(MockRefCounted);

		a->script->members.push_back(Variant(b.ptr()));
		b->script->members.push_back(Variant(a.ptr()));
	}
	// Both local Refs just went out of scope, but each object is still held alive by the
	// other's member -- ordinary unreferencing alone must not have freed them.
	CHECK_MESSAGE(MockRefCounted::alive_count == 2, "Cyclic objects should not be freed by ordinary unreferencing.");

	int freed = CycleCollector::collect_cycles();

	CHECK(freed == 2);
	CHECK_MESSAGE(MockRefCounted::alive_count == 0, "collect_cycles() should free both objects in the cycle.");
}

TEST_CASE("[CycleCollector] A cycle routed through an Array is collected") {
	MockRefCounted::alive_count = 0;
	{
		Ref<MockRefCounted> a = memnew(MockRefCounted);
		Ref<MockRefCounted> b = memnew(MockRefCounted);

		Array arr;
		arr.push_back(b.ptr());
		a->script->members.push_back(arr);
		b->script->members.push_back(Variant(a.ptr()));
	}
	CHECK_MESSAGE(MockRefCounted::alive_count == 2, "Cyclic objects should not be freed by ordinary unreferencing.");

	int freed = CycleCollector::collect_cycles();

	CHECK(freed == 2);
	CHECK_MESSAGE(MockRefCounted::alive_count == 0, "A cycle routed through an Array should still be detected and collected.");
}

TEST_CASE("[CycleCollector] A live object referenced more than once is not falsely collected") {
	MockRefCounted::alive_count = 0;
	Ref<MockRefCounted> c = memnew(MockRefCounted);
	{
		// A second, unrelated external reference to the same object. Dropping it decrements
		// without reaching zero, exactly like an internal cycle edge would -- this must not
		// be mistaken for one.
		Ref<MockRefCounted> c2 = c;
	}
	CHECK(MockRefCounted::alive_count == 1);

	int freed = CycleCollector::collect_cycles();

	CHECK(freed == 0);
	CHECK_MESSAGE(MockRefCounted::alive_count == 1, "An object with a genuine remaining external reference must not be collected.");
}

} // namespace TestCycleCollector
