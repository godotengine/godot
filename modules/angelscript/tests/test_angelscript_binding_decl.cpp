/**************************************************************************/
/*  test_angelscript_binding_decl.cpp                                     */
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

#include "../binding/as_binding_decl.h"

// 必须先定义该宏再包含测试头：本文件与 tests/test_main.cpp（经 modules_tests.gen.h）
// 都会展开该头，不定义宏就会把 TEST_CASE 注册两遍、用例被跑两次。
#define ANGELSCRIPT_BINDING_DECL_TESTS_IMPL
#include "test_angelscript_binding_decl.h"

// 本仓 doctest 无异常：每个前置条件 REQUIRE 后都显式 return 守卫，避免退化时空指针崩溃。
static PropertyInfo make_pi(Variant::Type p_type, const StringName &p_class = StringName(), PropertyHint p_hint = PROPERTY_HINT_NONE) {
	PropertyInfo pi;
	pi.type = p_type;
	pi.class_name = p_class;
	pi.hint = p_hint;
	return pi;
}

void as_binding_decl_scalar_mapping() {
	CHECK(ASBindingDecl::variant_type_to_as(Variant::BOOL) == "bool");
	CHECK(ASBindingDecl::variant_type_to_as(Variant::INT) == "int64"); // 64 位，不能退化为 int
	CHECK(ASBindingDecl::variant_type_to_as(Variant::FLOAT) == "double");
	CHECK(ASBindingDecl::variant_type_to_as(Variant::STRING) == "String");
	CHECK(ASBindingDecl::variant_type_to_as(Variant::VECTOR2) == "Vector2");
	CHECK(ASBindingDecl::variant_type_to_as(Variant::NIL).is_empty());

	// kind 供跳板决定如何编组，Task 3/4 共用。
	CHECK(ASBindingDecl::resolve(make_pi(Variant::INT)).kind == AS_KIND_INT64);
	CHECK(ASBindingDecl::resolve(make_pi(Variant::FLOAT)).kind == AS_KIND_DOUBLE);
	CHECK(ASBindingDecl::resolve(make_pi(Variant::VECTOR2)).kind == AS_KIND_VALUE);
	CHECK(ASBindingDecl::resolve(make_pi(Variant::OBJECT, "Node")).kind == AS_KIND_OBJECT_NONOWNING);
}

void as_binding_decl_object_and_enum() {
	ASBindingType o = ASBindingDecl::resolve(make_pi(Variant::OBJECT, "Node"));
	REQUIRE(o.valid);
	if (!o.valid) {
		return;
	}
	CHECK(o.as_name == "Node@");

	ASBindingType bare = ASBindingDecl::resolve(make_pi(Variant::OBJECT));
	CHECK(bare.valid);
	CHECK(bare.as_name == "Object@");

	// 枚举型属性 M2 退化为整数（spec §3.4）。
	ASBindingType e = ASBindingDecl::resolve(make_pi(Variant::INT, StringName(), PROPERTY_HINT_ENUM));
	CHECK(e.valid);
	CHECK(e.as_name == "int64");
}

void as_binding_decl_method_forms() {
	MethodInfo mi;
	mi.name = "get_name";
	mi.return_val = make_pi(Variant::STRING);
	mi.flags = METHOD_FLAG_CONST;
	String decl, reason;
	REQUIRE(ASBindingDecl::method_to_decl(mi, &decl, &reason));
	CHECK(decl == "String get_name() const");

	MethodInfo add;
	add.name = "add_child";
	add.return_val = make_pi(Variant::NIL);
	add.arguments.push_back(make_pi(Variant::OBJECT, "Node"));
	add.flags = METHOD_FLAG_NORMAL;
	CHECK(ASBindingDecl::method_to_decl(add, &decl, &reason));
	// 对象句柄形参带 `+`（auto handle）：见 as_binding_render_param 的 Ruling C 说明。
	CHECK(decl == "void add_child(Node@+)");

	// AS 2.38 无 static 关键字：静态方法必须整体归为 unbound，且声明里不得出现 "static"。
	MethodInfo stat;
	stat.name = "get_singleton";
	stat.return_val = make_pi(Variant::OBJECT, "Node");
	stat.flags = METHOD_FLAG_STATIC;
	CHECK_FALSE(ASBindingDecl::method_to_decl(stat, &decl, &reason));
	CHECK(reason == "static-method");

	// 真正返回 Variant 的方法（usage 带 NIL_IS_VARIANT）：AS 侧声明为 Variant，
	// 由通用跳板用已注册的 Variant 拷贝构造写回返回值（B4）。
	MethodInfo vari;
	vari.name = "get_indexed";
	vari.return_val = make_pi(Variant::NIL);
	vari.return_val.usage = PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_NIL_IS_VARIANT;
	REQUIRE(ASBindingDecl::method_to_decl(vari, &decl, &reason));
	CHECK(decl == "Variant get_indexed()");

	// 真 void 返回仍然必须可注册。
	MethodInfo no_ret;
	no_ret.name = "queue_free";
	no_ret.return_val = make_pi(Variant::NIL);
	CHECK(ASBindingDecl::method_to_decl(no_ret, &decl, &reason));
	CHECK(decl == "void queue_free()");
}

void as_binding_decl_static_method_forms() {
	// 静态方法 -> AS 全局函数声明：名字 `<类名>_<成员名>`，无 self、无 const。
	MethodInfo stat;
	stat.name = "is_main_thread";
	stat.return_val = make_pi(Variant::BOOL);
	stat.flags = METHOD_FLAG_STATIC;
	String decl, reason;
	REQUIRE(ASBindingDecl::static_method_to_decl(stat, StringName("Thread"), &decl, &reason));
	CHECK(decl == "bool Thread_is_main_thread()");
	CHECK(reason.is_empty());

	// 带参的静态方法：形参照样按通用规则渲染（String 走 const 引用）。
	MethodInfo stat_arg;
	stat_arg.name = "set_default_interface";
	stat_arg.return_val = make_pi(Variant::NIL);
	stat_arg.arguments.push_back(make_pi(Variant::STRING));
	stat_arg.flags = METHOD_FLAG_STATIC;
	CHECK(ASBindingDecl::static_method_to_decl(stat_arg, StringName("MultiplayerAPI"), &decl, &reason));
	CHECK(decl == "void MultiplayerAPI_set_default_interface(const String &in)");

	// 非静态方法不得走静态分类（否则实例方法会被误注册成全局函数）。
	MethodInfo inst;
	inst.name = "get_name";
	inst.return_val = make_pi(Variant::STRING);
	inst.flags = METHOD_FLAG_CONST;
	CHECK_FALSE(ASBindingDecl::static_method_to_decl(inst, StringName("Node"), &decl, &reason));
	CHECK(reason == "not-static-method");

	// vararg 静态方法同样拒绝。
	MethodInfo vstat;
	vstat.name = "log";
	vstat.return_val = make_pi(Variant::NIL);
	vstat.flags = METHOD_FLAG_STATIC | METHOD_FLAG_VARARG;
	CHECK_FALSE(ASBindingDecl::static_method_to_decl(vstat, StringName("Thread"), &decl, &reason));
	CHECK(reason.begins_with("vararg"));

	// method_to_decl 对静态方法仍拒绝：既有契约不变，静态方法必须走 static_method_to_decl。
	CHECK_FALSE(ASBindingDecl::method_to_decl(stat, &decl, &reason));
	CHECK(reason == "static-method");
}

void as_binding_decl_variant_abi() {
	// Variant 形参：PropertyInfo.type == NIL。上游约定参数不可能是 void，
	// 因此 NIL 形参唯一含义就是 Variant，渲染为 `const Variant &in`。
	MethodInfo set_meta;
	set_meta.name = "set_meta";
	set_meta.return_val = make_pi(Variant::NIL); // 真 void
	set_meta.arguments.push_back(make_pi(Variant::STRING_NAME));
	set_meta.arguments.push_back(make_pi(Variant::NIL));
	String decl, reason;
	REQUIRE(ASBindingDecl::method_to_decl(set_meta, &decl, &reason));
	CHECK(decl == "void set_meta(const StringName &in, const Variant &in)");

	// resolve() 对 NIL 给出合法的 Variant 分类，供跳板按 AS_KIND_VALUE 编组。
	ASBindingType t = ASBindingDecl::resolve(make_pi(Variant::NIL));
	CHECK(t.valid);
	CHECK(t.as_name == "Variant");
	CHECK(t.kind == AS_KIND_VALUE);

	// 无 NIL_IS_VARIANT 的 NIL 返回仍是 void：不能把所有 NIL 都当成 Variant。
	MethodInfo no_ret;
	no_ret.name = "queue_free";
	no_ret.return_val = make_pi(Variant::NIL);
	REQUIRE(ASBindingDecl::method_to_decl(no_ret, &decl, &reason));
	CHECK(decl == "void queue_free()");

	// 静态方法（注册为全局函数）的 Variant 形参/返回同样按通用规则渲染。
	MethodInfo stat;
	stat.name = "interpolate_value";
	stat.return_val = make_pi(Variant::NIL);
	stat.return_val.usage = PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_NIL_IS_VARIANT;
	stat.arguments.push_back(make_pi(Variant::NIL));
	stat.flags = METHOD_FLAG_STATIC;
	REQUIRE(ASBindingDecl::static_method_to_decl(stat, StringName("Tween"), &decl, &reason));
	CHECK(decl == "Variant Tween_interpolate_value(const Variant &in)");
}

void as_binding_decl_rejects_vararg_and_unknown() {
	MethodInfo vararg;
	vararg.name = "call";
	vararg.return_val = make_pi(Variant::NIL);
	vararg.flags = METHOD_FLAG_VARARG;
	String decl, reason;
	CHECK_FALSE(ASBindingDecl::method_to_decl(vararg, &decl, &reason));
	CHECK(reason.begins_with("vararg"));
}
