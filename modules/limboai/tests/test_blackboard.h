#ifndef TEST_BLACKBOARD_H
#define TEST_BLACKBOARD_H

#include "core/variant/variant.h"
#include "limbo_test.h"

#include "modules/limboai/blackboard/blackboard.h"

namespace TestBlackboard {

class TestPropertyHolder : public RefCounted {
	GDCLASS(TestPropertyHolder, RefCounted);

public:
	int property = 0;

	void set_property(int p_property) { property = p_property; }
	int get_property() { return property; }

	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("set_property", "property"), &TestPropertyHolder::set_property);
		ClassDB::bind_method(D_METHOD("get_property"), &TestPropertyHolder::get_property);

		ADD_PROPERTY(PropertyInfo(Variant::INT, "property"), "set_property", "get_property");
	}
};

TEST_CASE("[Modules][LimboAI] Test Blackboard") {
	Ref<Blackboard> blackboard = memnew(Blackboard);

	blackboard->set_var("a", 1);
	blackboard->set_var("b", Vector2(2, 2));
	blackboard->set_var("c", String("3"));

	Variant not_found("not_found");

	SUBCASE("Test getter") {
		CHECK_EQ(blackboard->get_var("a", not_found), Variant(1));
		CHECK_EQ(blackboard->get_var("b", not_found), Variant(Vector2(2, 2)));
		CHECK_EQ(blackboard->get_var("c", not_found), Variant("3"));
	}

	SUBCASE("Test has_var()") {
		CHECK(blackboard->has_var("a"));
		CHECK(blackboard->has_var("b"));
		CHECK(blackboard->has_var("c"));
		CHECK_FALSE(blackboard->has_var("d"));
	}

	SUBCASE("Test erase_var()") {
		blackboard->erase_var("b");
		CHECK_FALSE(blackboard->has_var("b"));
		CHECK(blackboard->has_var("a"));
		CHECK(blackboard->has_var("c"));
	}

	SUBCASE("Test clear()") {
		blackboard->clear();
		CHECK_FALSE(blackboard->has_var("a"));
		CHECK_FALSE(blackboard->has_var("b"));
		CHECK_FALSE(blackboard->has_var("c"));
	}

	SUBCASE("Test list_vars()") {
		TypedArray<StringName> vars = blackboard->list_vars();
		CHECK_EQ(vars.size(), 3);
		CHECK(vars.has("a"));
		CHECK(vars.has("b"));
		CHECK(vars.has("c"));
	}

	SUBCASE("Test get_vars_as_dict()") {
		Dictionary dict = blackboard->get_vars_as_dict();
		CHECK_EQ(dict.size(), 3);
		CHECK(dict.has("a"));
		CHECK(dict.has("b"));
		CHECK(dict.has("c"));
		CHECK_EQ(dict["a"], Variant(1));
		CHECK_EQ(dict["b"], Variant(Vector2(2, 2)));
		CHECK_EQ(dict["c"], Variant("3"));

		// * Should not include parent scope values
		Ref<Blackboard> parent_scope = memnew(Blackboard);
		blackboard->set_parent(parent_scope);
		parent_scope->set_var("d", 1);
		dict = blackboard->get_vars_as_dict();
		CHECK_EQ(dict.size(), 3);
		CHECK_FALSE(dict.has("d"));
	}

	SUBCASE("Test populate_from_dict() overriding values") {
		Dictionary dict;
		dict["a"] = "value_a";
		dict["b"] = "value_b";
		dict["c"] = "value_c";
		blackboard->populate_from_dict(dict);

		CHECK_EQ(blackboard->list_vars().size(), 3);
		CHECK_EQ(blackboard->get_var("a", not_found), Variant("value_a"));
		CHECK_EQ(blackboard->get_var("b", not_found), Variant("value_b"));
		CHECK_EQ(blackboard->get_var("c", not_found), Variant("value_c"));
	}

	SUBCASE("Test populate_from_dict() creating new values") {
		Dictionary dict;
		dict["d"] = "value_d";
		dict["e"] = "value_e";
		blackboard->populate_from_dict(dict);

		CHECK_EQ(blackboard->list_vars().size(), 5);
		CHECK_EQ(blackboard->get_var("d", not_found), Variant("value_d"));
		CHECK_EQ(blackboard->get_var("e", not_found), Variant("value_e"));
	}

	SUBCASE("Test scoping") {
		CHECK_EQ(blackboard->top(), blackboard);

		Ref<Blackboard> parent_scope = memnew(Blackboard);
		blackboard->set_parent(parent_scope);
		CHECK_EQ(blackboard->get_parent(), parent_scope);
		CHECK_EQ(blackboard->top(), parent_scope);

		parent_scope->set_var("a", 5);
		CHECK_EQ(blackboard->get_var("a", not_found), Variant(1)); // * should return current scope value
		CHECK_EQ(parent_scope->get_var("a", not_found), Variant(5)); // * should return parent scope value
		CHECK_EQ(parent_scope->get_var("b", not_found, false), not_found); // * should not return current scope value

		parent_scope->set_var("d", 123);
		CHECK_EQ(blackboard->get_var("d", not_found), Variant(123)); // * should return parent scope value

		blackboard->set_var("d", 456);
		CHECK_EQ(blackboard->get_var("d", not_found), Variant(456)); // * should return new value
		CHECK_EQ(parent_scope->get_var("d", not_found), Variant(123)); // * should return parent scope value

		Ref<Blackboard> grand_parent_scope = memnew(Blackboard);
		parent_scope->set_parent(grand_parent_scope);
		CHECK_EQ(blackboard->top(), grand_parent_scope);

		grand_parent_scope->set_var("a", "value_found");
		CHECK_EQ(grand_parent_scope->get_var("a", not_found), Variant("value_found"));
		CHECK_EQ(grand_parent_scope->get_var("b", not_found, false), not_found);
		CHECK_EQ(grand_parent_scope->get_var("c", not_found, false), not_found);
		CHECK_EQ(grand_parent_scope->get_var("d", not_found, false), not_found);
		CHECK_EQ(parent_scope->get_var("a", not_found), Variant(5));
		CHECK_EQ(blackboard->get_var("a", not_found), Variant(1));
	}

	SUBCASE("Test binding") {
		Ref<TestPropertyHolder> holder = memnew(TestPropertyHolder);
		blackboard->bind_var_to_property("a", holder.ptr(), "property");

		holder->set_property(5);
		CHECK_EQ(blackboard->get_var("a", not_found), Variant(5));
		blackboard->set_var("a", Variant(6));
		CHECK_EQ(holder->get_property(), 6);

		blackboard->unbind_var("a");
		blackboard->set_var("a", Variant(7));
		CHECK_EQ(holder->get_property(), 6);
		CHECK_EQ(blackboard->get_var("a", not_found), Variant(7));
	}

	SUBCASE("Test linking") {
		Ref<Blackboard> target_blackboard = memnew(Blackboard);

		target_blackboard->set_var("aa", Variant(111));
		blackboard->link_var("a", target_blackboard, "aa");
		CHECK(target_blackboard->has_var("aa"));
		CHECK_FALSE(target_blackboard->has_var("a"));

		CHECK_EQ(blackboard->get_var("a", not_found), Variant(111));
		CHECK_EQ(target_blackboard->get_var("aa", not_found), Variant(111));

		blackboard->set_var("a", Variant(222));
		CHECK_EQ(blackboard->get_var("a", not_found), Variant(222));
		CHECK_EQ(target_blackboard->get_var("aa", not_found), Variant(222));

		target_blackboard->set_var("aa", Variant(333));
		CHECK_EQ(blackboard->get_var("a", not_found), Variant(333));
		CHECK_EQ(target_blackboard->get_var("aa", not_found), Variant(333));
	}
}

} //namespace TestBlackboard

#endif // TEST_BLACKBOARD_H
