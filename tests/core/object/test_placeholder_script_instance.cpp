/**************************************************************************/
/*  test_placeholder_script_instance.cpp                                  */
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

TEST_FORCE_LINK(test_placeholder_script_instance)

// Objects do not fully integrate with placeholder instances, outside of editor builds.
// Placeholder instances should only be used for editor purposes anyway.
#ifdef TOOLS_ENABLED

#include "core/object/method_info.h"
#include "core/object/object.h"
#include "core/object/property_info.h"
#include "core/object/ref_counted.h"
#include "core/object/script_language.h"
#include "core/os/memory.h"
#include "core/string/string_name.h"
#include "core/templates/pair.h"
#include "core/variant/variant.h"

namespace TestPlaceholderScriptInstance {

class _MockScript final : public Script {
public:
	List<PropertyInfo> property_infos;
	HashMap<StringName, Variant> default_values;
	HashMap<StringName, Variant> constants;
	bool valid = true;

	virtual bool can_instantiate() const override {
		return false;
	}

	virtual Ref<Script> get_base_script() const override {
		return Ref<Script>();
	}

	virtual StringName get_global_name() const override {
		return StringName();
	}

	virtual bool inherits_script(const Ref<Script> &p_script) const override {
		return false;
	}

	virtual StringName get_instance_base_type() const override {
		return SNAME("RefCounted");
	}

	virtual ScriptInstance *instance_create(Object *p_this) override {
		return nullptr;
	}

	virtual bool has_source_code() const override {
		return false;
	}

	virtual String get_source_code() const override {
		return String();
	}

	virtual void set_source_code(const String &p_code) override {}

	virtual Error reload(bool p_keep_state = false) override { return OK; }

	virtual StringName get_doc_class_name() const override {
		return StringName();
	}

	virtual Vector<DocData::ClassDoc> get_documentation() const override {
		return Vector<DocData::ClassDoc>();
	}

	virtual String get_class_icon_path() const override {
		return String();
	}

	virtual bool has_method(const StringName &p_method) const override {
		return false;
	}

	virtual MethodInfo get_method_info(const StringName &p_method) const override {
		return MethodInfo();
	}

	virtual bool is_tool() const override {
		return false;
	}

	virtual bool is_script_valid() const override {
		return valid;
	}

	virtual bool is_abstract() const override {
		return false;
	}

	virtual ScriptLanguage *get_language() const override {
		return nullptr;
	}

	virtual bool has_script_signal(const StringName &p_signal) const override {
		return false;
	}

	virtual void get_script_signal_list(List<MethodInfo> *r_signals) const override {}

	virtual bool get_property_default_value(const StringName &p_property, Variant &r_value) const override {
		const Variant *res = default_values.getptr(p_property);
		if (res == nullptr) {
			return false;
		}
		r_value = *res;
		return true;
	}

	virtual void get_script_method_list(List<MethodInfo> *p_list) const override {}

	virtual void get_script_property_list(List<PropertyInfo> *p_list) const override {
		for (const PropertyInfo &E : property_infos) {
			p_list->push_back(E);
		}
	}

	virtual const Variant get_rpc_config() const override {
		return Variant();
	}

	virtual void get_constants(HashMap<StringName, Variant> *p_constants) override {
		for (const KeyValue<StringName, Variant> &E : constants) {
			p_constants->insert(E.key, E.value);
		}
	}

	virtual bool is_placeholder_fallback_enabled() const override {
		return !valid;
	}
};

#define MAKE_INSTANCE(m_script) \
	Ref<RefCounted> obj = memnew(RefCounted); \
	PlaceHolderScriptInstance *inst = memnew(PlaceHolderScriptInstance(nullptr, m_script, *obj)); \
	obj->set_script_instance(inst); \
	inst->update(scr->property_infos, scr->default_values);

TEST_SUITE("[PlaceholderScriptInstance]") {
	TEST_CASE("Setting a property of a valid script must change its value.") {
		Ref<_MockScript> scr = memnew(_MockScript);
		scr->property_infos.push_back(PropertyInfo(Variant::INT, "prop_a"));
		scr->default_values.insert("prop_a", 0);
		MAKE_INSTANCE(scr);

		CHECK_EQ(obj->get("prop_a"), Variant(0));
		CHECK_EQ(inst->set("prop_a", 1), true);
		CHECK_EQ(obj->get("prop_a"), Variant(1));
		CHECK_EQ(inst->set("prop_a", 2), true);
		CHECK_EQ(obj->get("prop_a"), Variant(2));
	}

	TEST_CASE("Constants of a valid script must be retrievable.") {
		Ref<_MockScript> scr = memnew(_MockScript);
		scr->constants.insert("const_a", 1);
		MAKE_INSTANCE(scr);

		CHECK_EQ(obj->get("const_a"), Variant(1));
	}

	TEST_CASE("Setting an absent property of a valid script must fail.") {
		Ref<_MockScript> scr = memnew(_MockScript);
		scr->property_infos.push_back(PropertyInfo(Variant::INT, "prop_a"));
		scr->default_values.insert("prop_a", 0);
		MAKE_INSTANCE(scr);

		{
			bool r_valid = true;
			obj->set("prop_b", 1, &r_valid);
			CHECK_EQ(r_valid, false);
		}

		{
			bool r_valid = true;
			obj->get("prop_b", &r_valid);
			CHECK_EQ(r_valid, false);
		}
	}

	TEST_CASE("Setting a property of an invalid script must change its value.") {
		Ref<_MockScript> scr = memnew(_MockScript);
		scr->valid = false;
		MAKE_INSTANCE(scr);

		obj->set("prop_a", 0);
		CHECK_EQ(obj->get("prop_a"), Variant(0));

		obj->set("prop_a", 1);
		CHECK_EQ(obj->get("prop_a"), Variant(1));
	}

	TEST_CASE("Updating the scripts default values.") {
		Ref<_MockScript> scr = memnew(_MockScript);
		scr->property_infos.push_back(PropertyInfo(Variant::INT, "prop_a"));
		scr->default_values.insert("prop_a", 0);
		MAKE_INSTANCE(scr);

		REQUIRE(inst->set("prop_a", 1));

		SUBCASE("Updating must not change modified properties.") {
			scr->default_values["prop_a"] = 2;
			inst->update(scr->property_infos, scr->default_values);

			CHECK_EQ(obj->get("prop_a"), Variant(1));
		}

		SUBCASE("After being reset a property must be updated correctly.") {
			CHECK(inst->set("prop_a", 0));
			CHECK_EQ(obj->get("prop_a"), Variant(0));

			scr->default_values["prop_a"] = 2;
			inst->update(scr->property_infos, scr->default_values);

			CHECK_EQ(obj->get("prop_a"), Variant(2));
		}

		SUBCASE("After the default value changes to the properties' current value, the property must be updated correctly.") {
			// Note: This test case is based on the current behavior at the time of writing. It does not make any claims about whether this is good behavior from a UX point of view.
			scr->default_values["prop_a"] = 1;
			inst->update(scr->property_infos, scr->default_values);
			scr->default_values["prop_a"] = 2;
			inst->update(scr->property_infos, scr->default_values);

			CHECK_EQ(obj->get("prop_a"), Variant(2));
		}
	}
}

#undef MAKE_INSTANCE

} //namespace TestPlaceholderScriptInstance
#endif // TOOLS_ENABLED
