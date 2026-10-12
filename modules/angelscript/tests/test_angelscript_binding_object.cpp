/**************************************************************************/
/*  test_angelscript_binding_object.cpp                                   */
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
#include "../binding/as_binding_decl.h"
#include "../binding/as_binding_native_value_ops.h"
#include "../binding/as_binding_object.h"
#include "../binding/as_binding_plan.h"
#include "../binding/as_binding_value_types.h"

#define ANGELSCRIPT_BINDING_OBJECT_TESTS_IMPL
#include "test_angelscript_binding_object.h"

#include "core/object/object.h"
#include "core/os/memory.h"
#include "core/string/print_string.h"
#include "scene/2d/node_2d.h"
#include "scene/2d/sprite_2d.h"
#include "scene/main/node.h"

#include <angelscript.h>

static bool g_object_bound = false;

// 测试用宿主拥有节点：AS 侧无法 free 非 RefCounted 对象，脚本只能拿到非拥有句柄，
// 回收责任留在宿主，避免脚本创建的节点残留到进程退出污染 ObjectDB。
namespace {

Node *g_test_parent = nullptr;
Node *g_test_child = nullptr;
Node2D *g_test_node2d = nullptr;
Sprite2D *g_test_sprite = nullptr;

void probe_test_parent(asIScriptGeneric *p_gen) {
	if (g_test_parent == nullptr) {
		g_test_parent = memnew(Node);
	}
	p_gen->SetReturnObject(as_handle_encode(g_test_parent, AS_KIND_OBJECT_NONOWNING));
}

void probe_test_child(asIScriptGeneric *p_gen) {
	if (g_test_child == nullptr) {
		g_test_child = memnew(Node);
	}
	p_gen->SetReturnObject(as_handle_encode(g_test_child, AS_KIND_OBJECT_NONOWNING));
}

void probe_test_node2d(asIScriptGeneric *p_gen) {
	if (g_test_node2d == nullptr) {
		g_test_node2d = memnew(Node2D);
	}
	p_gen->SetReturnObject(as_handle_encode(g_test_node2d, AS_KIND_OBJECT_NONOWNING));
}

void probe_test_sprite(asIScriptGeneric *p_gen) {
	if (g_test_sprite == nullptr) {
		g_test_sprite = memnew(Sprite2D);
	}
	p_gen->SetReturnObject(as_handle_encode(g_test_sprite, AS_KIND_OBJECT_NONOWNING));
}

void release_test_nodes() {
	if (g_test_child != nullptr) {
		memdelete(g_test_child);
		g_test_child = nullptr;
	}
	if (g_test_parent != nullptr) {
		memdelete(g_test_parent);
		g_test_parent = nullptr;
	}
	if (g_test_node2d != nullptr) {
		memdelete(g_test_node2d);
		g_test_node2d = nullptr;
	}
	if (g_test_sprite != nullptr) {
		memdelete(g_test_sprite);
		g_test_sprite = nullptr;
	}
}

} // namespace

// 只绑定测试需要的那条继承链，避免把整个 ClassDB 都挂上去（默认全量留给 T6 的 e2e 验证）。
// 注册必须两遍：先全部骨架，再挂成员，最后枚举；否则跨类引用的签名会解析失败，
// 而 AS 里任何一次 Register* 失败都会永久污染引擎。
static void ensure_object_binding(asIScriptEngine *p_engine) {
	if (g_object_bound) {
		return;
	}
	g_object_bound = true;

	ASBindingScope scope;
	scope.whitelist.push_back("Object");
	scope.whitelist.push_back("RefCounted");
	scope.whitelist.push_back("Resource");
	scope.whitelist.push_back("Node");
	scope.whitelist.push_back("CanvasItem");
	scope.whitelist.push_back("Node2D");
	scope.whitelist.push_back("Sprite2D");

	ASBindingPlan plan;
	plan.build(scope);
	for (const ASBindingClass &c : plan.get_classes()) {
		ASBindingObject::register_skeleton(c, p_engine);
	}
	for (const ASBindingClass &c : plan.get_classes()) {
		ASBindingObject::register_class(c, p_engine);
	}
	ASBindingObject::register_enums(plan.get_enums(), p_engine);

	p_engine->RegisterGlobalFunction("Node@ probe_test_parent()", asFUNCTION(probe_test_parent), asCALL_GENERIC);
	p_engine->RegisterGlobalFunction("Node@ probe_test_child()", asFUNCTION(probe_test_child), asCALL_GENERIC);
	p_engine->RegisterGlobalFunction("Node2D@ probe_test_node2d()", asFUNCTION(probe_test_node2d), asCALL_GENERIC);
	p_engine->RegisterGlobalFunction("Sprite2D@ probe_test_sprite()", asFUNCTION(probe_test_sprite), asCALL_GENERIC);
}

static bool nearly(double p_a, double p_b) {
	return p_a > p_b - 0.001 && p_a < p_b + 0.001;
}

// 编译并执行 `double main()`，取回其 double 返回值。
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
	ensure_object_binding(engine);

	String err;
	if (!as->compile_module("as_binding_obj_run", p_source, &err)) {
		print_line("compile error: " + err);
		REQUIRE(false);
		return false;
	}
	asIScriptModule *mod = engine->GetModule("as_binding_obj_run");
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

void as_binding_object_register_all() {
	ASEngine *as = ASEngine::get_singleton();
	REQUIRE(as->ensure_initialized());
	if (!as->is_initialized()) {
		return;
	}
	asIScriptEngine *engine = as->get_engine();
	REQUIRE(ASBindingValueTypes::register_all(engine) == OK);
	ensure_object_binding(engine);

	for (const char *name : { "Object", "RefCounted", "Resource", "Node", "CanvasItem", "Node2D", "Sprite2D" }) {
		CHECK(engine->GetTypeInfoByName(name) != nullptr);
	}
	asITypeInfo *node = engine->GetTypeInfoByName("Node");
	REQUIRE(node != nullptr);
	CHECK(node->GetMethodByName("add_child") != nullptr);
	CHECK(node->GetMethodByName("get_child_count") != nullptr);
	// `name` 是属性而非方法：其访问器被 MethodInfo 的属性排除标志滤掉，只以 virtual property 出现。
	CHECK(node->GetMethodByName("get_name") != nullptr);
	asITypeInfo *node2d = engine->GetTypeInfoByName("Node2D");
	REQUIRE(node2d != nullptr);
	CHECK(node2d->GetMethodByName("get_position") != nullptr);
	CHECK(node2d->GetMethodByName("set_position") != nullptr);
	// 枚举名是类内裸名（如 "ProcessMode"），AS 侧以声明类做前缀：Node_ProcessMode。
	CHECK(engine->GetTypeInfoByName("Node_ProcessMode") != nullptr);
	CHECK(engine->GetGlobalPropertyIndexByName("Object_NOTIFICATION_PREDELETE") >= 0);
}

void as_binding_object_method_call() {
	double out = 0.0;
	if (!run_double(
				"double main() {"
				"  Node @parent = probe_test_parent();"
				"  Node @child = probe_test_child();"
				"  parent.add_child(child, false, 0);"
				"  int64 count = parent.get_child_count(false);"
				"  return count == 1 ? 1.0 : 0.0;"
				"}",
				&out)) {
		return;
	}
	CHECK(nearly(out, 1.0)); // 方法跳板：实参编组 + Object 方法调用 + int64 返回。
	// 节点由宿主拥有，测试结束回收，避免污染 ObjectDB（AS 侧无法 free 非 RefCounted）。
	release_test_nodes();
}

void as_binding_object_property_access() {
	double out = 0.0;
	if (!run_double(
				"double main() {"
				"  Node2D @n = probe_test_node2d();"
				"  Vector2 p(3, 4);"
				"  n.position = p;"
				"  Vector2 q = n.position;"
				"  return (q.x == 3.0 && q.y == 4.0) ? 1.0 : 0.0;"
				"}",
				&out)) {
		return;
	}
	CHECK(nearly(out, 1.0)); // virtual property 读写 + 值类型编组。
	release_test_nodes();
}

void as_binding_object_upcast() {
	double out = 0.0;
	if (!run_double(
				"double main() {"
				"  Node @n = probe_test_sprite();"
				"  Object @o = probe_test_node2d();"
				"  bool ok = (n !is null) && (o !is null);"
				"  return ok ? 1.0 : 0.0;"
				"}",
				&out)) {
		return;
	}
	CHECK(nearly(out, 1.0)); // opImplCast：对每个祖先各注册一条。
	release_test_nodes();
}

void as_binding_object_refcount_lifetime() {
	double out = 0.0;
	if (!run_double(
				"double main() {"
				"  Resource @r = Resource();"
				"  double a = double(r.get_reference_count());"
				"  Resource @d = r.duplicate(false);"
				"  double b = double(d.get_reference_count());"
				"  @d = null;"
				"  @r = null;"
				"  return (a == 1.0 && b == 1.0) ? 1.0 : 0.0;"
				"}",
				&out)) {
		return;
	}
	// 工厂与“返回 RefCounted 的方法”都必须让脚本恰好持有一份引用：多一份是泄漏，少一份是释放后使用。
	CHECK(nearly(out, 1.0));
}

// B1：锁定 M3 非拥有（弱）句柄语义 —— `Object@`/`Node@` 槽里存 ObjectID，脚本持有既
// 不参与引用计数、也不保活对象；需要保活必须声明具体 RefCounted 类型（如 `Resource@`）。
// 固化为回归可防止再次把它“修复”成补引用而破坏 `ObjectDB` 生命周期契约。
void as_binding_object_nonowning_assign() {
	double out = 0.0;
	// 1) 把 RefCounted 赋给非拥有祖先句柄，不得改变引用计数（弱引用）。
	if (!run_double(
				"double main() {"
				"  Resource @r = Resource();"
				"  double before = double(r.get_reference_count());"
				"  Object @o = r;"
				"  double after = double(r.get_reference_count());"
				"  return before * 10.0 + after;"
				"}",
				&out)) {
		return;
	}
	CHECK(nearly(out, 11.0)); // 赋值前后都恰好 1 份：弱句柄不补引用

	// 2) 同一对象赋给另一个弱句柄同样不改变引用计数。
	if (!run_double(
				"double main() {"
				"  Resource @r = Resource();"
				"  Object @a = r;"
				"  Object @b = a;"
				"  return double(r.get_reference_count());"
				"}",
				&out)) {
		return;
	}
	CHECK(nearly(out, 1.0));

	// 3) 拥有句柄仍存活时，弱句柄可正常使用。
	if (!run_double(
				"double main() {"
				"  Resource @r = Resource();"
				"  Object @o = r;"
				"  return (o !is null) ? 1.0 : 0.0;"
				"}",
				&out)) {
		return;
	}
	CHECK(nearly(out, 1.0));

	// 4) 拥有句柄释放后对象随之销毁；`is null` 只比较槽原值（ObjectID），弱句柄不会
	//    变为 null —— 失效只在解引用时经 ObjectDB 反查暴露异常。
	if (!run_double(
				"double main() {"
				"  Resource @r = Resource();"
				"  Object @o = r;"
				"  @r = null;"
				"  return (o !is null) ? 1.0 : 0.0;"
				"}",
				&out)) {
		return;
	}
	CHECK(nearly(out, 1.0));

	// 5) 空句柄安全。
	if (!run_double(
				"double main() {"
				"  Object @n = null;"
				"  return (n is null) ? 1.0 : 0.0;"
				"}",
				&out)) {
		return;
	}
	CHECK(nearly(out, 1.0));
}

// B2：丢弃拥有句柄返回值不得泄漏 ObjectDB 实例 —— 返回值 transfer 必须恰好一次。
// 覆盖多种“返回值生命周期结束后”的形态：丢弃工厂结果、丢弃方法返回值、弱上转后作用域结束、
// 接收变量后作用域结束。每段脚本前后比较 ObjectDB 实例数，差值必须为 0。
void as_binding_object_discarded_return() {
	double out = 0.0;
	// 预热一次，完成值类型与对象绑定注册，避免把首次注册的副作用计入差值。
	if (!run_double("double main() { return 0.0; }", &out)) {
		return;
	}

	struct DiscardProbe {
		const char *label;
		const char *source;
	};
	const DiscardProbe probes[] = {
		{ "discard factory result",
				"double main() { for (int i = 0; i < 8; i++) { Resource(); } return 1.0; }" },
		{ "discard method return",
				"double main() { for (int i = 0; i < 8; i++) { Resource @r = Resource(); r.duplicate(false); } return 1.0; }" },
		{ "weak upcast then scope end",
				"double main() { for (int i = 0; i < 8; i++) { Resource @r = Resource(); Object @o = r; } return 1.0; }" },
		{ "receive then scope end",
				"double main() { for (int i = 0; i < 8; i++) { Resource @r = Resource(); Resource @d = r.duplicate(false); } return 1.0; }" },
	};

	for (const DiscardProbe &p : probes) {
		const int before = ObjectDB::get_object_count();
		if (!run_double(p.source, &out)) {
			return;
		}
		const int after = ObjectDB::get_object_count();
		INFO("probe: ", p.label, " before=", before, " after=", after);
		CHECK(after == before);
	}
}

void as_binding_object_enums_and_constants() {
	double out = 0.0;
	if (!run_double(
				"double main() {"
				"  if (Node_ProcessMode::PROCESS_MODE_ALWAYS != 3) return 0.0;"
				"  if (Object_NOTIFICATION_PREDELETE != 1) return 0.0;"
				"  return 1.0;"
				"}",
				&out)) {
		return;
	}
	CHECK(nearly(out, 1.0)); // ClassDB 枚举（名带点号需净化）+ BIND_CONSTANT 全局常量。
}

void as_binding_object_native_hot_method() {
#if AS_NATIVE_VALUE_OPS_ENABLED
	// Node::get_child_count 是 P1 热点；命中原生 thunk 时计数器必须递增，
	// 未命中则走 generic，计数为 0（本用例即用于锁定该路径）。
	ASNativeValueOps::reset_native_thunk_calls();
	double out = 0.0;
	if (!run_double(
				"double main() {"
				"  Node @p = probe_test_parent();"
				"  return double(p.get_child_count(false));"
				"}",
				&out)) {
		return;
	}
	CHECK(nearly(out, 0.0)); // 宿主父节点未挂子节点。
	CHECK(ASNativeValueOps::native_thunk_calls() > 0); // 证明走了原生 thunk 而非 generic。
	release_test_nodes();
#else
	MESSAGE("native value ops disabled on this platform; native hot method path is inactive");
#endif
}
