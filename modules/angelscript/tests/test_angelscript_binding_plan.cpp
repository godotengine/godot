/**************************************************************************/
/*  test_angelscript_binding_plan.cpp                                     */
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

#include "../binding/as_binding_plan.h"

#define ANGELSCRIPT_BINDING_PLAN_TESTS_IMPL
#include "test_angelscript_binding_plan.h"

void as_binding_plan_default_scope_is_ordered() {
	ASBindingScope scope; // 空白/黑名单 = 放行全部。
	ASBindingPlan plan;
	plan.build(scope);
	REQUIRE(!plan.get_classes().is_empty());
	if (plan.get_classes().is_empty()) {
		return;
	}
	// Object 必须最先（引用类型体系根）。
	CHECK(plan.get_classes()[0].name == StringName("Object"));

	// 每个类的父必须出现在它之前（拓扑序）。
	HashMap<StringName, int> index;
	for (int i = 0; i < plan.get_classes().size(); i++) {
		index[plan.get_classes()[i].name] = i;
	}
	for (const ASBindingClass &c : plan.get_classes()) {
		if (c.parent == StringName()) {
			continue;
		}
		if (!index.has(c.parent)) {
			continue;
		}
		CHECK(index[c.parent] < index[c.name]);
	}

	// 34 个内建值类型 + Variant 自身。
	CHECK(plan.get_value_types().size() == 35);
}

void as_binding_plan_collects_unbound_vararg() {
	ASBindingPlan plan;
	plan.build(ASBindingScope());
	bool has_vararg = false;
	for (const ASUnboundEntry &e : plan.get_unbound()) {
		if (e.owner == StringName("Object") && e.member == "call" && e.reason.begins_with("vararg")) {
			has_vararg = true;
		}
	}
	CHECK(has_vararg); // Object::call 是 vararg，必须进未绑定清单。
}

void as_binding_plan_binds_static_methods() {
	ASBindingPlan plan;
	plan.build(ASBindingScope());

	// 静态方法改为注册为全局函数后，不应再以 "static-method" 理由出现在未绑定清单。
	// 未绑定清单很大，这里聚合成一个计数断言，避免制造海量断言噪声。
	int static_reasons = 0;
	for (const ASUnboundEntry &e : plan.get_unbound()) {
		if (e.reason == "static-method") {
			static_reasons++;
		}
	}
	CHECK(static_reasons == 0);

	// Node::print_orphan_nodes 是 ClassDB 静态方法：声明名称为全局函数 <类名>_<成员名>。
	bool found = false;
	for (const ASBindingClass &c : plan.get_classes()) {
		if (c.name != StringName("Node")) {
			continue;
		}
		for (const ASBindingMethod &m : c.methods) {
			if (m.name == StringName("print_orphan_nodes")) {
				found = true;
				CHECK(m.is_static);
				CHECK(m.as_decl == "void Node_print_orphan_nodes()");
			}
		}
	}
	CHECK(found);
}

void as_binding_plan_whitelist_keeps_ancestors() {
	ASBindingScope scope;
	scope.whitelist.push_back("Sprite2D"); // 期望连带保留 Node2D/CanvasItem/Node/Object。
	ASBindingPlan plan;
	plan.build(scope);
	HashSet<StringName> names;
	for (const ASBindingClass &c : plan.get_classes()) {
		names.insert(c.name);
	}
	CHECK(names.has("Sprite2D"));
	CHECK(names.has("Node2D"));
	CHECK(names.has("CanvasItem"));
	CHECK(names.has("Node"));
	CHECK(names.has("Object"));
	CHECK_FALSE(names.has("Timer")); // 非祖先、非白名单 => 不可见。
}

void as_binding_plan_blacklist_wins() {
	ASBindingScope scope;
	scope.blacklist.push_back("Timer");
	ASBindingPlan plan;
	plan.build(scope);
	for (const ASBindingClass &c : plan.get_classes()) {
		CHECK(c.name != StringName("Timer"));
	}
	// Object 无条件保留，即使被拉黑。
	CHECK_FALSE(plan.get_classes().is_empty());
}
