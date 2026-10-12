/**************************************************************************/
/*  test_angelscript_binding_visibility.cpp                               */
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
#include "../binding/as_binding_plan.h"

#define ANGELSCRIPT_BINDING_VISIBILITY_TESTS_IMPL
#include "test_angelscript_binding_visibility.h"

#include "core/config/project_settings.h"
#include "core/string/print_string.h"
#include "core/templates/hash_set.h"

#include <angelscript.h>

static bool nearly(double p_a, double p_b) {
	return p_a > p_b - 0.001 && p_a < p_b + 0.001;
}

// 用共享引擎（ASEngine 单例）编译并执行 `double main()`：ensure_initialized()
// 现在会一次性接入全量绑定计划与 @GlobalScope 工具函数，因此这里也顺带覆盖接线本身。
static bool run_double(const String &p_source, double *r_out) {
	ASEngine *as = ASEngine::get_singleton();
	if (!as->ensure_initialized() || !as->is_initialized()) {
		REQUIRE(false);
		return false;
	}
	asIScriptEngine *engine = as->get_engine();

	String err;
	if (!as->compile_module("as_binding_vis_run", p_source, &err)) {
		print_line("compile error: " + err);
		REQUIRE(false);
		return false;
	}
	asIScriptModule *mod = engine->GetModule("as_binding_vis_run");
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
	ctx->Release();
	if (rc != asEXECUTION_FINISHED) {
		print_line(vformat("runtime error (line %d): %s", rc, exc ? exc : "<none>"));
		REQUIRE(false);
		return false;
	}
	*r_out = out;
	return true;
}

void as_binding_visibility_project_setting_applies() {
	ProjectSettings *ps = ProjectSettings::get_singleton();

	// PackedStringArray 没有 initializer_list 构造，必须逐个 push_back。
	PackedStringArray whitelist;
	whitelist.push_back("Sprite2D");
	ps->set_setting("angel_script/class_whitelist", whitelist);

	ASBindingScope scope = ASBindingScope::from_project_settings();
	ASBindingPlan plan;
	plan.build(scope);

	HashSet<StringName> names;
	for (const ASBindingClass &c : plan.get_classes()) {
		names.insert(c.name);
	}

	CHECK(names.has("Sprite2D"));
	CHECK(names.has("Node")); // 白名单必须回填祖先链，否则子类签名引用的父类型不存在。
	CHECK_FALSE(names.has("Timer"));

	ps->set_setting("angel_script/class_whitelist", PackedStringArray());
}

void as_binding_visibility_utility_function_clampf() {
	// 具体标量签名的 clampf 验证 @GlobalScope 工具函数接线。
	double out = 0.0;
	REQUIRE(run_double("double main() { return clampf(5.0, 0.0, 1.0); }", &out));
	CHECK(nearly(out, 1.0));
}

void as_binding_visibility_utility_variant_function() {
	// B4 解禁 Variant 形参/返回后，全 Variant 签名的工具函数（clamp）也随之外接
	// （此前因签名含 Variant 被跳过）。脚本侧没有 double→Variant 隐式转换，必须显式构造 Variant。
	double out = 0.0;
	REQUIRE(run_double(
			"double main() {"
			" Variant v = clamp(Variant(5.0), Variant(0.0), Variant(1.0));"
			" double d = v;"
			" return d;"
			"}",
			&out));
	CHECK(nearly(out, 1.0));
}

void as_binding_visibility_static_method_global() {
	// ClassDB 静态方法注册为 AS 全局函数：Thread::is_main_thread -> Thread_is_main_thread()。
	double out = 0.0;
	REQUIRE(run_double("double main() { return Thread_is_main_thread() ? 1.0 : 0.0; }", &out));
	CHECK(nearly(out, 1.0));
}

void as_binding_visibility_variant_abi() {
	// Variant 形参/返回 ABI 的端到端验证：Object::set_meta / get_meta 是常用的
	// 带 Variant 形参且不依赖其它新绑定的方法。Resource 是 RefCounted，拥有句柄
	// 在作用域结束后释放，不会在退出时泄漏。
	double out = 0.0;
	REQUIRE(run_double(
			"double main() {"
			" Resource r;"
			" r.set_meta(StringName(\"k\"), Variant(3.0));"
			" Variant v = r.get_meta(StringName(\"k\"), Variant(0.0));"
			" double d = v;"
			" return d;"
			"}",
			&out));
	CHECK(nearly(out, 3.0));
}
