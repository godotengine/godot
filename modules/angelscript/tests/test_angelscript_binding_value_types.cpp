/**************************************************************************/
/*  test_angelscript_binding_value_types.cpp                              */
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

#include "../as_engine.h"
#include "../binding/as_binding_value_types.h"

#define ANGELSCRIPT_BINDING_VALUE_TYPES_TESTS_IMPL
#include "test_angelscript_binding_value_types.h"

#include "core/string/print_string.h"

#include <angelscript.h>

static const char *VALUE_TYPE_NAMES[] = {
	"String", "StringName", "NodePath", "RID",
	"Vector2", "Vector2i", "Vector3", "Vector3i",
	"Vector4", "Vector4i", "Rect2", "Rect2i",
	"Transform2D", "Plane", "Quaternion", "AABB",
	"Basis", "Transform3D", "Projection", "Color",
	"PackedByteArray", "PackedInt32Array", "PackedInt64Array",
	"PackedFloat32Array", "PackedFloat64Array", "PackedStringArray",
	"PackedVector2Array", "PackedVector3Array", "PackedVector4Array",
	"PackedColorArray", "Array", "Dictionary", "Callable", "Signal",
	"Variant",
};
static const int VALUE_TYPE_NAME_COUNT = 35;

static bool nearly(double p_a, double p_b) {
	return p_a > p_b - 0.001 && p_a < p_b + 0.001;
}

// 编译并执行 `double main()`，取回其 double 返回值。
// register_all 幂等，放在这里保证本文件用例间的执行顺序不影响结果。
static bool run_double(const String &p_source, double *r_out) {
	ASEngine *as = ASEngine::get_singleton();
	if (!as->ensure_initialized() || !as->is_initialized()) {
		REQUIRE(false);
		return false;
	}
	asIScriptEngine *engine = as->get_engine();
	if (ASBindingValueTypes::register_all(engine) != OK) {
		REQUIRE(false);
		return false;
	}
	String err;
	if (!as->compile_module("as_binding_vt_run", p_source, &err)) {
		print_line("compile error: " + err);
		REQUIRE(false);
		return false;
	}
	asIScriptModule *mod = engine->GetModule("as_binding_vt_run");
	if (!mod) {
		REQUIRE(false);
		return false;
	}
	asIScriptFunction *func = mod->GetFunctionByName("main");
	if (!func) {
		REQUIRE(false);
		return false;
	}
	asIScriptContext *ctx = engine->CreateContext();
	if (!ctx) {
		REQUIRE(false);
		return false;
	}
	if (ctx->Prepare(func) < 0) {
		ctx->Release();
		REQUIRE(false);
		return false;
	}
	int rc = ctx->Execute();
	double out = ctx->GetReturnDouble();
	const char *exc = ctx->GetExceptionString();
	int exc_line = ctx->GetExceptionLineNumber();
	ctx->Release();
	if (rc != asEXECUTION_FINISHED) {
		print_line(vformat("runtime error (line %d): %s", exc_line, exc ? exc : "<none>"));
		REQUIRE(false);
		return false;
	}
	if (r_out) {
		*r_out = out;
	}
	return true;
}

void as_binding_metadata_auxiliary() {
	ASEngine *as = ASEngine::get_singleton();
	if (!as->ensure_initialized()) {
		REQUIRE(false);
		return;
	}
	asIScriptEngine *engine = as->get_engine();
	CHECK(ASBindingValueTypes::register_all(engine) == OK);
	int methods = 0;
	int behaviours = 0;
	int globals = 0;
	bool matched = true;
	auto inspect = [&matched](asIScriptFunction *f, int &count) {
		if (f != nullptr && f->GetUserData() != nullptr) {
			count++;
			matched = matched && f->GetAuxiliary() == f->GetUserData();
		}
	};
	for (asUINT i = 0; i < engine->GetObjectTypeCount(); i++) {
		asITypeInfo *type = engine->GetObjectTypeByIndex(i);
		for (asUINT j = 0; j < type->GetMethodCount(); j++) {
			inspect(type->GetMethodByIndex(j), methods);
		}
		for (asUINT j = 0; j < type->GetBehaviourCount(); j++) {
			inspect(type->GetBehaviourByIndex(j, nullptr), behaviours);
		}
		for (asUINT j = 0; j < type->GetFactoryCount(); j++) {
			inspect(type->GetFactoryByIndex(j), behaviours);
		}
	}
	for (asUINT i = 0; i < engine->GetGlobalFunctionCount(); i++) {
		inspect(engine->GetGlobalFunctionByIndex(i), globals);
	}
	CHECK(methods > 0);
	CHECK(behaviours > 0);
	CHECK(globals > 0);
	CHECK(matched);
}

void as_binding_value_types_register_all() {
	ASEngine *as = ASEngine::get_singleton();
	REQUIRE(as->ensure_initialized());
	if (!as->is_initialized()) {
		return;
	}
	asIScriptEngine *engine = as->get_engine();
	CHECK(ASBindingValueTypes::register_all(engine) == OK);
	for (int i = 0; i < VALUE_TYPE_NAME_COUNT; i++) {
		CHECK(engine->GetTypeInfoByName(VALUE_TYPE_NAMES[i]) != nullptr);
	}
	// string 由字符串工厂注册（spec §3.8.2）。
	CHECK(engine->GetTypeInfoByName("string") != nullptr);

	// 原生调用约定前提：注册对象类型必须带 asOBJ_APP_CLASS（无子标志）。
	{
		asITypeInfo *vi = engine->GetTypeInfoByName("Vector2");
		REQUIRE(vi != nullptr);
		CHECK((vi->GetFlags() & asOBJ_APP_CLASS) != 0);
		CHECK((vi->GetFlags() & (asOBJ_APP_PRIMITIVE | asOBJ_APP_FLOAT | asOBJ_APP_ARRAY)) == 0);
	}
	// 标志只应加在内建 34 个值类型上；Variant / string 仍走纯 generic，不应带 asOBJ_APP_CLASS。
	{
		asITypeInfo *si = engine->GetTypeInfoByName("string");
		REQUIRE(si != nullptr);
		CHECK((si->GetFlags() & asOBJ_APP_CLASS) == 0);
	}
}

void as_binding_value_types_vector2_roundtrip() {
	double out = 0.0;
	if (!run_double("double main() { Vector2 v(3, 4); return v.length(); }", &out)) {
		return;
	}
	CHECK(nearly(out, 5.0)); // 内省构造器 + 内建方法 + 值类型返回，全链路。
}

void as_binding_value_types_members_and_ops() {
	double out = 0.0;
	if (!run_double("double main() { Vector2 v(3, 4); v.x = 5.0; return v.x; }", &out)) {
		return;
	}
	CHECK(nearly(out, 5.0)); // 成员 virtual property 的 get/set。

	if (!run_double("double main() { Vector2 a(1, 2); Vector2 b(3, 4); Vector2 c = a + b; return c.y; }", &out)) {
		return;
	}
	CHECK(nearly(out, 6.0)); // 运算符跳板 + 值为返回。
}

void as_binding_value_types_indexing() {
	double out = 0.0;
	if (!run_double("double main() { PackedInt64Array a; a.push_back(7); return double(a[0]); }", &out)) {
		return;
	}
	CHECK(nearly(out, 7.0)); // 索引读取。
}

void as_binding_value_types_variant_array() {
	double out = 0.0;
	if (!run_double("double main() { Array a; a.append(Variant(7)); return double(a[0]); }", &out)) {
		return;
	}
	CHECK(nearly(out, 7.0)); // Variant 构造 + 数组索引 + Variant->double 转换。
}

void as_binding_value_types_variant_dictionary() {
	double out = 0.0;
	if (!run_double("double main() { Dictionary d; d[Variant(\"k\")] = Variant(9); return double(d[Variant(\"k\")]); }", &out)) {
		return;
	}
	CHECK(nearly(out, 9.0)); // keyed 读写 + String->Variant 构造。
}

void as_binding_value_types_string_interop() {
	double out = 0.0;
	if (!run_double("double main() { String s = \"abc\"; string t = s; return double(t.length()); }", &out)) {
		return;
	}
	CHECK(nearly(out, 3.0)); // String -> string 隐式构造 + string.length()。
}

void as_binding_value_types_string_variable_forms() {
	double out = 0.0;
	// 变量 → 变量的拷贝构造（`string(const string &in)`）：验证 string 实参的指针约定。
	if (!run_double("double main() { string a = \"abc\"; string b = a; return double(b.length()); }", &out)) {
		return;
	}
	CHECK(nearly(out, 3.0));

	// opAssign 的变量实参形态。
	if (!run_double("double main() { string a = \"abc\"; string b; b = a; return double(b.length()); }", &out)) {
		return;
	}
	CHECK(nearly(out, 3.0));

	// opAssign 的字面量实参形态。
	if (!run_double("double main() { string b; b = \"xyz\"; return double(b.length()); }", &out)) {
		return;
	}
	CHECK(nearly(out, 3.0));

	// 字面量构造后再自赋值，确保存储不是指向临时量。
	if (!run_double("double main() { string a = \"abcd\"; string b = a; b = a; return double(b.length()); }", &out)) {
		return;
	}
	CHECK(nearly(out, 4.0));

	// Godot String 的拷贝构造与赋值（值类型存储为 Variant，走 VT_CTOR_COPY/VT_ASSIGN）。
	if (!run_double("double main() { String s = \"abc\"; String t = s; t = s; return double(t.length()); }", &out)) {
		return;
	}
	CHECK(nearly(out, 3.0));
}

void as_binding_value_types_builtin_variant_return() {
	double out = 0.0;
	// Array.pop_back 返回 Variant（NIL + PROPERTY_USAGE_NIL_IS_VARIANT）：必须真正带回返回值，
	// 而不是被声明成 void 后静默丢弃。
	if (!run_double("double main() { Array a; a.append(Variant(7)); Variant r = a.pop_back(); return double(r); }", &out)) {
		return;
	}
	CHECK(nearly(out, 7.0));

	// 容器确实被修改（pop_back 的副作用）。
	if (!run_double("double main() { Array a; a.append(Variant(7)); a.pop_back(); return double(a.size()); }", &out)) {
		return;
	}
	CHECK(nearly(out, 0.0));

	// Dictionary.get 同样返回 Variant。
	if (!run_double("double main() { Dictionary d; d[Variant(\"k\")] = Variant(5); Variant r = d.get(Variant(\"k\"), Variant(0)); return double(r); }", &out)) {
		return;
	}
	CHECK(nearly(out, 5.0));
}

void as_binding_value_types_static_builtin_methods() {
	double out = 0.0;
	// 值类型静态内建方法没有对象 this，作为 <类型>_<方法> 全局函数注册
	// （与 ClassDB 静态方法注册为全局函数的约定一致）。
	if (!run_double("double main() { Vector2 v = Vector2_from_angle(0.0); return v.x; }", &out)) {
		return;
	}
	CHECK(nearly(out, 1.0));

	// 整型参 + String 返回的静态方法（chr 无默认参数）。
	if (!run_double("double main() { String s = String_chr(65); return double(s.length()); }", &out)) {
		return;
	}
	CHECK(nearly(out, 1.0));
}
