/**************************************************************************/
/*  test_angelscript_debugger.cpp                                         */
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

#include "../as_debugger.h"
#include "../as_engine.h"
#include "../as_script_language.h"
#include "../binding/as_binding_object.h"

#include "core/debugger/engine_debugger.h"
#include "core/debugger/script_debugger.h"
#include "core/object/object.h"

#include <angelscript.h>

#define ANGELSCRIPT_DEBUGGER_TESTS_IMPL
#include "test_angelscript_debugger.h"

namespace TestAngelScriptDebugger {

namespace {

class LineDebugger : public EngineDebugger {
	EngineDebugger *previous_engine = nullptr;
	ScriptDebugger *previous_script = nullptr;

public:
	Vector<int> lines;
	Vector<String> functions;
	bool use_step = false;
	bool manual_pause = false;
	bool saw_globals = false;
	bool saw_locals = false;
	bool saw_typed_globals = false;
	int command_mode = 0; // 0=next, 1=step/out, 2=continue, 3=out
	Vector<int> stack_sizes;

	LineDebugger() {
		previous_engine = singleton;
		previous_script = script_debugger;
		singleton = this;
		script_debugger = memnew(ScriptDebugger);
		script_debugger->set_lines_left(-1);
		script_debugger->set_depth(-1);
	}
	~LineDebugger() {
		memdelete(script_debugger);
		// EngineDebugger destructor deletes the current static ScriptDebugger.
		script_debugger = nullptr;
		singleton = nullptr;
	}
	static void restore_previous(EngineDebugger *p_engine, ScriptDebugger *p_script) {
		singleton = p_engine;
		script_debugger = p_script;
	}
	EngineDebugger *get_previous_engine() const { return previous_engine; }
	ScriptDebugger *get_previous_script() const { return previous_script; }
	void send_message(const String &, const Array &) override {}
	void send_error(const String &, const String &, int, const String &, const String &, bool, ErrorHandlerType) override {}
	void debug(bool p_can_continue, bool p_is_error) override {
		if (script_debugger->is_skipping_breakpoints() && !p_is_error) return;
		CHECK(p_can_continue);
		CHECK_FALSE(p_is_error);
		CHECK(script_debugger->get_break_language() == ASScriptLanguage::get_singleton());
		CHECK(ASDebugger::get_break_context() != nullptr);
		stack_sizes.push_back(ASDebugger::get_stack_level_count());
		lines.push_back(ASDebugger::get_stack_level_line(0));
		functions.push_back(ASDebugger::get_stack_level_function(0));
		List<String> names;
		List<Variant> values;
		ASDebugger::get_globals(&names, &values, -1, -1);
		const List<Variant>::Element *global_value = values.front();
		bool saw_label = false;
		bool saw_namespaced = false;
		for (const String &name : names) {
			if (name == "total") saw_globals = true;
			if (name == "label" && global_value != nullptr) saw_label = String(global_value->get()) == "debug";
			if (name == "settings::limit" && global_value != nullptr) saw_namespaced = (int64_t)global_value->get() == 42;
			if (global_value != nullptr) global_value = global_value->next();
		}
		saw_typed_globals = saw_typed_globals || (saw_label && saw_namespaced);
		names.clear();
		values.clear();
		ASDebugger::get_stack_level_locals(0, &names, &values, -1, -1);
		for (const String &name : names) {
			if (name == "x") saw_locals = true;
		}
		if (lines.size() == 1 && !manual_pause && command_mode != 2) {
			script_debugger->set_depth(command_mode == 3 ? 1 : (use_step ? -1 : 0));
			script_debugger->set_lines_left(1);
		} else if (lines.size() == 2 && use_step) {
			script_debugger->set_depth(1);
			script_debugger->set_lines_left(1);
		} else {
			script_debugger->set_depth(-1);
			script_debugger->set_lines_left(-1);
		}
	}
	void poll_events(bool) override {
		if (manual_pause && lines.is_empty()) {
			script_debugger->debug(script_debugger->get_break_language());
		}
	}
};

// 每个用例用独立模块名，避免 AS 引擎里重名模块互相覆盖。
int g_module_seq = 0;

String next_module_name() {
	return "as_debugger_test_" + itos(g_module_seq++);
}

// 测试专用异常回调的落点：headless 下 EngineDebugger 不活跃，无法走 debug_break，
// 所以测试自己装回调，在「AS 栈仍然活着」的同一时机做断言——与生产路径时机完全一致。
using ProbeFn = void (*)(asIScriptContext *);
ProbeFn g_probe = nullptr;

void probe_callback(asIScriptContext *p_ctx, void *p_user) {
	if (g_probe != nullptr) {
		g_probe(p_ctx);
	}
}

// 编译 p_source，执行其中的 p_func_name，并在异常回调里调用 p_probe。
// 返回 false 表示编译或准备阶段就失败了（调用方应 REQUIRE 后 return）。
bool run_until_exception(const String &p_source, const char *p_func_name, ProbeFn p_probe, String *r_module_name) {
	ASEngine *as = ASEngine::get_singleton();
	if (as == nullptr || !as->ensure_initialized()) {
		return false;
	}

	const String module_name = next_module_name();
	String err;
	if (!as->compile_module(module_name, p_source, &err)) {
		return false;
	}

	asIScriptModule *mod = as->get_engine()->GetModule(module_name.utf8().get_data());
	if (mod == nullptr) {
		return false;
	}
	asIScriptFunction *func = mod->GetFunctionByName(p_func_name);
	if (func == nullptr) {
		return false;
	}

	asIScriptContext *ctx = as->get_engine()->CreateContext();
	if (ctx == nullptr) {
		return false;
	}

	g_probe = p_probe;
	ctx->SetExceptionCallback(asFUNCTION(probe_callback), nullptr, asCALL_CDECL);
	ctx->Prepare(func);
	ctx->Execute();
	g_probe = nullptr;
	ctx->Release();

	if (r_module_name != nullptr) {
		*r_module_name = module_name;
	}
	return true;
}

asIScriptContext *g_nested_context = nullptr;
asIScriptFunction *g_nested_function = nullptr;

void nested_host_call() {
	CHECK(ASEngine::get_singleton()->call_function(g_nested_context, g_nested_function, nullptr, nullptr, 0, nullptr) == OK);
}

Vector<ScriptLanguage::StackInfo> g_captured_stack;

void capture_stack(asIScriptContext *p_ctx) {
	g_captured_stack = ASDebugger::build_stack_info(p_ctx);
}

bool g_unhandled_seen = false;

void probe_unhandled_flag(asIScriptContext *p_ctx) {
	g_unhandled_seen = ASDebugger::is_unhandled_exception(p_ctx);
}

List<String> g_local_names;
List<Variant> g_local_values;

void capture_locals(asIScriptContext *p_ctx) {
	g_local_names.clear();
	g_local_values.clear();
	ASDebugger::get_stack_level_locals(p_ctx, 0, &g_local_names, &g_local_values, -1, -1);
}

// 在 names/values 两张平行表里按名字取值；找不到返回空 Variant。
Variant find_local(const String &p_name) {
	const List<String>::Element *n = g_local_names.front();
	const List<Variant>::Element *v = g_local_values.front();
	while (n != nullptr && v != nullptr) {
		if (n->get() == p_name) {
			return v->get();
		}
		n = n->next();
		v = v->next();
	}
	return Variant();
}

// 名字是否出现在局部变量表里。与 find_local 不同：它能区分「没有这个名字」
// 和「有这个名字但值恰好为空 Variant」。
bool has_local(const String &p_name) {
	for (const List<String>::Element *n = g_local_names.front(); n != nullptr; n = n->next()) {
		if (n->get() == p_name) {
			return true;
		}
	}
	return false;
}

int g_locals_max_depth = -1;

void capture_locals_with_depth(asIScriptContext *p_ctx) {
	g_local_names.clear();
	g_local_values.clear();
	ASDebugger::get_stack_level_locals(p_ctx, 0, &g_local_names, &g_local_values, -1, g_locals_max_depth);
}

List<String> g_member_names;
List<Variant> g_member_values;

void capture_members(asIScriptContext *p_ctx) {
	g_member_names.clear();
	g_member_values.clear();
	ASDebugger::get_stack_level_members(p_ctx, 0, &g_member_names, &g_member_values, -1, -1);
}

Variant find_member(const String &p_name) {
	const List<String>::Element *n = g_member_names.front();
	const List<Variant>::Element *v = g_member_values.front();
	while (n != nullptr && v != nullptr) {
		if (n->get() == p_name) {
			return v->get();
		}
		n = n->next();
		v = v->next();
	}
	return Variant();
}

// 表达式求值的第二趟（查 this 成员）只有在类方法帧里才会命中，单独捕获。
String g_member_expression;

void capture_member_expression(asIScriptContext *p_ctx) {
	capture_members(p_ctx);
	g_member_expression = ASDebugger::parse_stack_level_expression(p_ctx, 0, "tag");
}

// 非空时在解码之前把它 free 掉，用来模拟「调试器查看变量时对象已经没了」。
Object *g_free_before_decode = nullptr;

void probe_object_handle(asIScriptContext *p_ctx) {
	if (g_free_before_decode != nullptr) {
		memdelete(g_free_before_decode);
		g_free_before_decode = nullptr;
	}
	capture_locals(p_ctx);
}

// 带一个 Object@ 实参调用脚本函数。ASEngine::call_function 目前只编组标量实参
// （as_engine.cpp 的 default 分支直接返回 ERR_INVALID_PARAMETER），所以这里自己
// 按绑定层的槽语义编码句柄再 SetArgAddress——与绑定层产出的槽完全同构。
// 返回 false 表示编译/查找阶段就失败了。
bool run_with_object_arg(const String &p_source, Object *p_arg, ProbeFn p_probe) {
	ASEngine *as = ASEngine::get_singleton();
	if (as == nullptr || !as->ensure_initialized()) {
		return false;
	}
	const String module_name = next_module_name();
	String err;
	if (!as->compile_module(module_name, p_source, &err)) {
		return false;
	}
	asIScriptModule *mod = as->get_engine()->GetModule(module_name.utf8().get_data());
	if (mod == nullptr) {
		return false;
	}
	asIScriptFunction *func = mod->GetFunctionByName("probe");
	if (func == nullptr) {
		return false;
	}
	asIScriptContext *ctx = as->get_engine()->CreateContext();
	if (ctx == nullptr) {
		return false;
	}

	g_probe = p_probe;
	ctx->SetExceptionCallback(asFUNCTION(probe_callback), nullptr, asCALL_CDECL);
	ctx->Prepare(func);
	// Object 不派生自 RefCounted，槽里放的是 ObjectID 而不是裸指针。
	ctx->SetArgAddress(0, as_handle_encode(p_arg, AS_KIND_OBJECT_NONOWNING));
	ctx->Execute();

	g_probe = nullptr;
	ctx->Release();
	return true;
}

// 对一个给定的上下文，用越界的层号把全部查询跑一遍，断言都返回安全默认值。
void expect_safe_defaults(asIScriptContext *p_ctx, int p_level) {
	CHECK(ASDebugger::get_stack_level_line(p_ctx, p_level) == -1);
	CHECK(ASDebugger::get_stack_level_function(p_ctx, p_level) == String());
	CHECK(ASDebugger::get_stack_level_source(p_ctx, p_level) == String());
	CHECK(ASDebugger::parse_stack_level_expression(p_ctx, p_level, "anything") == String());

	List<String> names;
	List<Variant> values;
	ASDebugger::get_stack_level_locals(p_ctx, p_level, &names, &values, -1, -1);
	CHECK(names.is_empty());
	CHECK(values.is_empty());

	ASDebugger::get_stack_level_members(p_ctx, p_level, &names, &values, -1, -1);
	CHECK(names.is_empty());
	CHECK(values.is_empty());
}

void probe_invalid_levels(asIScriptContext *p_ctx) {
	expect_safe_defaults(p_ctx, -1);
	expect_safe_defaults(p_ctx, 999);
	// 合法层上的未知表达式同样要返回空串，而不是瞎猜。
	CHECK(ASDebugger::parse_stack_level_expression(p_ctx, 0, "no_such_name") == String());
	CHECK(ASDebugger::parse_stack_level_expression(p_ctx, 0, "a") == String("42"));
}

} // namespace

void test_callback_activation() {
	ASEngine *as = ASEngine::get_singleton();
	if (as == nullptr || !as->ensure_initialized()) {
		REQUIRE(false);
		return;
	}
	CHECK_FALSE(EngineDebugger::is_active());
	asIScriptContext *context = as->create_context();
	if (context == nullptr) {
		REQUIRE(false);
		return;
	}
	CHECK(ASDebugger::is_attached(context));
	CHECK_FALSE(ASDebugger::is_line_enabled(context));
	LineDebugger *debugger = memnew(LineDebugger);
	EngineDebugger *previous_engine = debugger->get_previous_engine();
	ScriptDebugger *previous_script = debugger->get_previous_script();
	{
		ASDebugger::ExecutionScope scope(context);
		CHECK(ASDebugger::is_line_enabled(context));
	}
	ASDebugger::sync_line_callback(context); // idempotent enable
	CHECK(ASDebugger::is_line_enabled(context));
	asIScriptContext *second = as->create_context();
	CHECK(ASDebugger::is_line_enabled(second));
	if (second != nullptr) {
		second->Release(); // cleanup balances the live count
	}
	memdelete(debugger);
	LineDebugger::restore_previous(previous_engine, previous_script);
	{
		ASDebugger::ExecutionScope scope(context);
		CHECK_FALSE(ASDebugger::is_line_enabled(context));
		CHECK(ASDebugger::is_attached(context));
	}
	ASDebugger::sync_line_callback(context); // idempotent disable
	CHECK_FALSE(ASDebugger::is_line_enabled(context));
	context->Release();
}

void test_line_stepping() {
	ASEngine *as = ASEngine::get_singleton();
	REQUIRE(as != nullptr);
	if (as == nullptr || !as->ensure_initialized()) return;
	const String source = "int total = 0;\n"
			"void child() {\n"
			"  total += 2;\n"
			"}\n"
			"int main() {\n"
			"  int x = 1;\n"
			"  child();\n"
			"  x += 3;\n"
			"  return x + total;\n"
			"}\n"
			"String label = \"debug\";\n"
			"namespace settings { int limit = 42; }\n";
	const String module_name = next_module_name();
	String error;
	REQUIRE(as->compile_module(module_name, source, &error));
	asIScriptModule *module = as->get_engine()->GetModule(module_name.utf8().get_data());
	if (module == nullptr) return;
	asIScriptFunction *function = module->GetFunctionByName("main");
	if (function == nullptr) return;
	for (int mode = 0; mode < 3; mode++) {
		LineDebugger *debugger = memnew(LineDebugger);
		EngineDebugger *previous_engine = debugger->get_previous_engine();
		ScriptDebugger *previous_script = debugger->get_previous_script();
		debugger->use_step = mode == 1;
		debugger->command_mode = mode == 1 ? 1 : 0;
		debugger->manual_pause = mode == 2;
		if (mode != 2) EngineDebugger::get_script_debugger()->insert_breakpoint(7, module_name);
		asIScriptContext *context = as->create_context();
		Variant result;
		CHECK(as->call_function(context, function, nullptr, nullptr, 0, &result) == OK);
		CHECK((int64_t)result == 6 + mode * 2);
		CHECK_FALSE(debugger->lines.is_empty());
		if (mode == 0 && debugger->lines.size() >= 2) {
			CHECK(debugger->lines[0] == 7);
			CHECK(debugger->lines[1] == 8);
			CHECK(debugger->functions[1] == "main");
		}
		if (mode == 1 && debugger->lines.size() >= 3) {
			CHECK(debugger->functions[1] == "child");
			CHECK(debugger->functions[2] == "main");
		}
		CHECK(debugger->saw_globals);
		CHECK(debugger->saw_typed_globals);
		if (mode != 2) CHECK(debugger->saw_locals);
		CHECK(ASDebugger::get_break_context() == nullptr);
		CHECK(EngineDebugger::get_script_debugger()->get_break_language() == nullptr);
		context->Release();
		memdelete(debugger);
		LineDebugger::restore_previous(previous_engine, previous_script);
	}
	as->get_engine()->DiscardModule(module_name.utf8().get_data());

	// Repeated hits at the same line are real loop iterations, not duplicate
	// debugger notifications. Removed and disabled breakpoints must not stop.
	const String loop_name = next_module_name();
	REQUIRE(as->compile_module(loop_name, "int main() {\n int x = 0;\n for (int i=0; i<3; i++) {\n x++;\n }\n return x;\n}\n", &error));
	asIScriptFunction *loop = as->get_engine()->GetModule(loop_name.utf8().get_data())->GetFunctionByName("main");
	for (int mode = 0; mode < 3; mode++) {
		LineDebugger *debugger = memnew(LineDebugger);
		EngineDebugger *previous_engine = debugger->get_previous_engine();
		ScriptDebugger *previous_script = debugger->get_previous_script();
		debugger->command_mode = 2;
		ScriptDebugger *sd = EngineDebugger::get_script_debugger();
		sd->insert_breakpoint(4, loop_name);
		if (mode == 1) sd->remove_breakpoint(4, loop_name);
		if (mode == 2) sd->set_skip_breakpoints(true);
		asIScriptContext *context = as->create_context();
		Variant result;
		CHECK(as->call_function(context, loop, nullptr, nullptr, 0, &result) == OK);
		CHECK((int64_t)result == 3);
		CHECK(debugger->lines.size() == (mode == 0 ? 3 : 0));
		context->Release();
		memdelete(debugger);
		LineDebugger::restore_previous(previous_engine, previous_script);
	}
	as->get_engine()->DiscardModule(loop_name.utf8().get_data());

	// Recursion has identical function/source names but distinct stack depths.
	const String recursive_name = next_module_name();
	REQUIRE(as->compile_module(recursive_name, "int recurse(int n) {\n if (n > 0) {\n return recurse(n-1);\n }\n return n;\n}\nint main() {\n return recurse(2);\n}\n", &error));
	LineDebugger *debugger = memnew(LineDebugger);
	EngineDebugger *previous_engine = debugger->get_previous_engine();
	ScriptDebugger *previous_script = debugger->get_previous_script();
	debugger->command_mode = 3;
	EngineDebugger::get_script_debugger()->insert_breakpoint(5, recursive_name);
	asIScriptContext *context = as->create_context();
	Variant result;
	asIScriptFunction *recursive = as->get_engine()->GetModule(recursive_name.utf8().get_data())->GetFunctionByName("main");
	CHECK(as->call_function(context, recursive, nullptr, nullptr, 0, &result) == OK);
	CHECK(debugger->stack_sizes.size() >= 2);
	if (debugger->stack_sizes.size() >= 2) CHECK(debugger->stack_sizes[1] < debugger->stack_sizes[0]);
	CHECK(ASDebugger::get_break_context() == nullptr);
	context->Release();
	memdelete(debugger);
	LineDebugger::restore_previous(previous_engine, previous_script);
	as->get_engine()->DiscardModule(recursive_name.utf8().get_data());

	// next must skip a script re-entry through a host function, including a
	// different source section, and then stop back in the original caller.
	REQUIRE(as->get_engine()->RegisterGlobalFunction("void debugger_nested_host()", asFUNCTION(nested_host_call), asCALL_CDECL) >= 0);
	const String outer_name = next_module_name();
	const String inner_name = next_module_name();
	REQUIRE(as->compile_module(inner_name, "void inner() {\n int value = 1;\n value++;\n}\n", &error));
	REQUIRE(as->compile_module(outer_name, "int main() {\n int x = 0;\n debugger_nested_host();\n x++;\n return x;\n}\n", &error));
	g_nested_context = as->create_context();
	g_nested_function = as->get_engine()->GetModule(inner_name.utf8().get_data())->GetFunctionByName("inner");
	debugger = memnew(LineDebugger);
	previous_engine = debugger->get_previous_engine();
	previous_script = debugger->get_previous_script();
	EngineDebugger::get_script_debugger()->insert_breakpoint(3, outer_name);
	context = as->create_context();
	asIScriptFunction *outer = as->get_engine()->GetModule(outer_name.utf8().get_data())->GetFunctionByName("main");
	CHECK(as->call_function(context, outer, nullptr, nullptr, 0, &result) == OK);
	CHECK((int64_t)result == 1);
	CHECK(debugger->lines.size() == 2);
	if (debugger->lines.size() == 2) CHECK(debugger->lines[1] == 4);
	CHECK(ASDebugger::get_break_context() == nullptr);
	context->Release();
	g_nested_context->Release();
	g_nested_context = nullptr;
	g_nested_function = nullptr;
	memdelete(debugger);
	LineDebugger::restore_previous(previous_engine, previous_script);
	as->get_engine()->DiscardModule(outer_name.utf8().get_data());
	as->get_engine()->DiscardModule(inner_name.utf8().get_data());

	List<String> names;
	List<Variant> values;
	names.push_back("sentinel");
	values.push_back(7);
	ASDebugger::get_globals(&names, &values, -1, -1);
	CHECK(names.size() == 1);
	CHECK(values.size() == 1);
}

void test_builds_stack_info() {
	const String source =
			"void inner() { int d = 0; int x = 1 / d; }\n"
			"void outer() { inner(); }\n";

	g_captured_stack.clear();
	String module_name;
	const bool ran = run_until_exception(source, "outer", capture_stack, &module_name);
	REQUIRE(ran);
	if (!ran) {
		return;
	}

	REQUIRE(g_captured_stack.size() >= 2);
	if (g_captured_stack.size() < 2) {
		return;
	}
	CHECK(g_captured_stack[0].func == "inner");
	CHECK(g_captured_stack[1].func == "outer");
	CHECK(g_captured_stack[0].file == module_name);
	CHECK(g_captured_stack[0].line >= 1);
}

void test_skips_caught_exception() {
	// 脚本自己 catch 掉的异常不该惊动调试器——否则用户每写一个 try/catch 都会被断下。
	const String guarded_source =
			"void guarded() {\n"
			"	try { int d = 0; int x = 1 / d; } catch { }\n"
			"}\n";

	g_unhandled_seen = true;
	String module_name;
	bool ran = run_until_exception(guarded_source, "guarded", probe_unhandled_flag, &module_name);
	REQUIRE(ran);
	if (!ran) {
		return;
	}
	CHECK_FALSE(g_unhandled_seen);

	// 对照组：没有 try/catch 的同一个异常必须被判为「未处理」，否则上面那条
	// CHECK_FALSE 可能只是因为谓词恒假而碰巧通过。
	const String bare_source =
			"void bare() { int d = 0; int x = 1 / d; }\n";

	g_unhandled_seen = false;
	ran = run_until_exception(bare_source, "bare", probe_unhandled_flag, &module_name);
	REQUIRE(ran);
	if (!ran) {
		return;
	}
	CHECK(g_unhandled_seen);
}

void test_reads_locals() {
	// late 在抛异常时还没执行到声明处。AS 对标量的 GetAddressOfVar 照样返回有效栈地址，
	// 所以必须靠 IsVarInScope 拦住，否则面板上会出现一个带垃圾值的 late。
	const String source =
			"void probe() {\n"
			"	int a = 42;\n"
			"	double b = 1.5;\n"
			"	bool c = true;\n"
			"	int d = 0;\n"
			"	int boom = 1 / d;\n"
			"	int late = 7;\n"
			"}\n";

	String module_name;
	const bool ran = run_until_exception(source, "probe", capture_locals, &module_name);
	REQUIRE(ran);
	if (!ran) {
		return;
	}

	CHECK(find_local("a") == Variant(42));
	CHECK(find_local("b") == Variant(1.5));
	CHECK(find_local("c") == Variant(true));
	CHECK(g_local_names.size() == g_local_values.size());

	// 未进入作用域的变量不得出现在列表里。
	CHECK_FALSE(has_local("late"));
	// 编译器临时变量是无名的，它们不是用户写的变量，不该进面板。
	CHECK_FALSE(has_local(String()));
}

void test_decodes_bound_value_type() {
	const String source =
			"void probe() {\n"
			"	Vector2 v = Vector2(3, 4);\n"
			"	Vector3 w = Vector3(5, 6, 7);\n"
			"	Color c = Color(1, 2, 3, 4);\n"
			"	int d = 0;\n"
			"	int boom = 1 / d;\n"
			"}\n";

	String module_name;
	const bool ran = run_until_exception(source, "probe", capture_locals, &module_name);
	REQUIRE(ran);
	if (!ran) {
		return;
	}

	const Variant got = find_local("v");
	REQUIRE(got.get_type() == Variant::VECTOR2);
	if (got.get_type() != Variant::VECTOR2) {
		return;
	}
	CHECK(Vector2(got) == Vector2(3, 4));
	CHECK(find_local("w") == Variant(Vector3(5, 6, 7)));
	CHECK(find_local("c") == Variant(Color(1, 2, 3, 4)));
}

void test_decodes_native_string() {
	// AS 原生 string 是唯一不以 Variant 为存储的绑定值类型：槽里装的是驻留工厂对象的
	// 指针。走通用值类型路径会把 8 字节指针当 Variant 读，必须有专项分支。
	const String source =
			"void probe() {\n"
			"	string s = \"hi\";\n"
			"	int d = 0;\n"
			"	int boom = 1 / d;\n"
			"}\n";

	String module_name;
	const bool ran = run_until_exception(source, "probe", capture_locals, &module_name);
	REQUIRE(ran);
	if (!ran) {
		return;
	}

	CHECK(find_local("s") == Variant("hi"));
}

void test_expands_script_class() {
	// node.next 指回 node 自身，构造一个环。解码器必须在第二次遇到它时返回 "<cycle>"，
	// 而不是无限递归把栈撑爆。
	const String source =
			"class Node2 {\n"
			"	int id;\n"
			"	double weight;\n"
			"	Node2 @next;\n"
			"}\n"
			"void probe() {\n"
			"	Node2 node;\n"
			"	node.id = 7;\n"
			"	node.weight = 2.5;\n"
			"	@node.next = node;\n"
			"	int d = 0;\n"
			"	int boom = 1 / d;\n"
			"}\n";

	g_locals_max_depth = -1;
	String module_name;
	bool ran = run_until_exception(source, "probe", capture_locals_with_depth, &module_name);
	REQUIRE(ran);
	if (!ran) {
		return;
	}

	const Variant got = find_local("node");
	REQUIRE(got.get_type() == Variant::DICTIONARY);
	if (got.get_type() != Variant::DICTIONARY) {
		return;
	}
	const Dictionary dict = got;
	CHECK(dict.has("<class>"));
	CHECK(String(dict["<class>"]) == "Node2");
	CHECK(dict["id"] == Variant(7));
	CHECK(dict["weight"] == Variant(2.5));

	REQUIRE(dict.has("next"));
	if (!dict.has("next")) {
		return;
	}
	// 环上的那一跳：next 指回 node，必须被环检测拦住。
	CHECK(dict["next"] == Variant("<cycle>"));

	// 深度上限：max_depth = 1 时顶层标量仍是真值，但不再往下展开一层对象。
	g_locals_max_depth = 1;
	g_local_names.clear();
	g_local_values.clear();
	ran = run_until_exception(source, "probe", capture_locals_with_depth, &module_name);
	REQUIRE(ran);
	if (!ran) {
		return;
	}
	const Variant shallow = find_local("node");
	REQUIRE(shallow.get_type() == Variant::DICTIONARY);
	if (shallow.get_type() != Variant::DICTIONARY) {
		return;
	}
	const Dictionary shallow_dict = shallow;
	CHECK(shallow_dict["id"] == Variant(7));
	CHECK(shallow_dict["next"] == Variant("<...>"));
}

void test_decodes_object_handle() {
	// Object 不派生自 RefCounted，所以句柄槽里放的是 ObjectID 而不是裸指针。
	// 这条用例同时钉住两件事：槽只能解一层引用、且必须经 as_handle_decode。
	const String source =
			"void probe(Object @o) {\n"
			"	int d = 0;\n"
			"	int boom = 1 / d;\n"
			"}\n";

	Object *alive = memnew(Object);
	g_free_before_decode = nullptr;
	bool ran = run_with_object_arg(source, alive, probe_object_handle);
	REQUIRE(ran);
	if (!ran) {
		memdelete(alive);
		return;
	}
	const Variant got = find_local("o");
	CHECK(got.get_type() == Variant::OBJECT);
	CHECK(got.get_validated_object() == alive);
	memdelete(alive);

	// 对象在解码之前被销毁：必须给空 Variant，绝不能二次崩溃毁掉诊断现场。
	Object *doomed = memnew(Object);
	g_free_before_decode = doomed;
	ran = run_with_object_arg(source, doomed, probe_object_handle);
	REQUIRE(ran);
	if (!ran) {
		return;
	}
	CHECK(find_local("o").get_validated_object() == nullptr);
}

void test_handles_invalid_level() {
	// 没有断点时（break_context == nullptr），所有无参查询必须给安全默认值。
	CHECK(ASDebugger::get_break_context() == nullptr);
	CHECK(ASDebugger::get_stack_level_count() == 0);
	CHECK(ASDebugger::get_stack_level_line(0) == -1);
	CHECK(ASDebugger::get_stack_level_function(0) == String());
	CHECK(ASDebugger::get_stack_level_source(0) == String());
	CHECK(ASDebugger::parse_stack_level_expression(0, "a") == String());

	List<String> names;
	List<Variant> values;
	ASDebugger::get_stack_level_locals(0, &names, &values, -1, -1);
	CHECK(names.is_empty());
	ASDebugger::get_stack_level_members(0, &names, &values, -1, -1);
	CHECK(names.is_empty());

	expect_safe_defaults(nullptr, 0);

	const String source =
			"void probe() {\n"
			"	int a = 42;\n"
			"	int d = 0;\n"
			"	int boom = 1 / d;\n"
			"}\n";

	String module_name;
	const bool ran = run_until_exception(source, "probe", probe_invalid_levels, &module_name);
	REQUIRE(ran);
	if (!ran) {
		return;
	}
}

void test_reads_this_members() {
	// 异常抛在类方法里，所以层 0 就是那个方法帧，GetThisPointer 能拿到 this。
	// holder.self 指回自身，验证 seen.insert(self) 真的挡住了自环。
	const String source =
			"class Holder {\n"
			"	int count;\n"
			"	double ratio;\n"
			"	int tag;\n"
			"	Holder @self;\n"
			"	void boom() {\n"
			"		int d = 0;\n"
			"		int x = 1 / d;\n"
			"	}\n"
			"}\n"
			"void probe() {\n"
			"	Holder h;\n"
			"	h.count = 3;\n"
			"	h.ratio = 0.25;\n"
			"	h.tag = 99;\n"
			"	@h.self = h;\n"
			"	h.boom();\n"
			"}\n";

	g_member_expression = String();
	String module_name;
	const bool ran = run_until_exception(source, "probe", capture_member_expression, &module_name);
	REQUIRE(ran);
	if (!ran) {
		return;
	}

	CHECK(g_member_names.size() == g_member_values.size());
	CHECK(find_member("count") == Variant(3));
	CHECK(find_member("ratio") == Variant(0.25));
	CHECK(find_member("self") == Variant("<cycle>"));
	// 表达式求值的第二趟：局部变量里没有 tag，必须从 this 成员里查到。
	CHECK(g_member_expression == "99");
}

void test_exception_flag_is_per_context() {
	// 复现「脏标志跨上下文泄漏」：ASScriptInstance 的构造期异常不经 call_function，
	// 标志置位后无人消费；若消费端不比对上下文，下一次别的上下文执行失败时
	// 就会被这面陈旧的旗子吃掉本该打印的 ERR_PRINT。
	ASEngine *as = ASEngine::get_singleton();
	REQUIRE(as != nullptr);
	REQUIRE(as->ensure_initialized());
	if (as == nullptr || !as->ensure_initialized()) {
		return;
	}

	asIScriptContext *producer = as->get_engine()->CreateContext();
	asIScriptContext *other = as->get_engine()->CreateContext();
	REQUIRE(producer != nullptr);
	REQUIRE(other != nullptr);
	if (producer == nullptr || other == nullptr) {
		if (producer != nullptr) {
			producer->Release();
		}
		if (other != nullptr) {
			other->Release();
		}
		return;
	}

	ASDebugger::mark_exception_reported(producer);
	// 别的上下文不得认领这面旗子，也不得把它清掉。
	CHECK_FALSE(ASDebugger::consume_exception_reported(other));
	// 产生它的上下文认领得到，且只认领一次。
	CHECK(ASDebugger::consume_exception_reported(producer));
	CHECK_FALSE(ASDebugger::consume_exception_reported(producer));

	producer->Release();
	other->Release();
}

void test_attaches_engine_internal_contexts() {
	// AS 内部有一整类上下文不经 ASEngine::create_context()：脚本全局变量初始化
	// （asCModule::InitGlobalProp）与脚本对象的拷贝构造 / opEquals / GC 枚举都走
	// asIScriptEngine::RequestContext()。不接管它，这些路径上的异常对调试器完全不可见。
	ASEngine *as = ASEngine::get_singleton();
	REQUIRE(as != nullptr);
	REQUIRE(as->ensure_initialized());
	if (as == nullptr || !as->ensure_initialized()) {
		return;
	}

	asIScriptEngine *engine = as->get_engine();
	REQUIRE(engine != nullptr);
	if (engine == nullptr) {
		return;
	}

	asIScriptContext *ctx = engine->RequestContext();
	REQUIRE(ctx != nullptr);
	if (ctx == nullptr) {
		return;
	}
	CHECK(ASDebugger::is_attached(ctx));
	engine->ReturnContext(ctx);

	// 自建上下文没有经过任何装配入口，必须是未装配的——否则上面那条 CHECK
	// 可能只是因为谓词恒真而碰巧通过。
	asIScriptContext *bare = engine->CreateContext();
	REQUIRE(bare != nullptr);
	if (bare == nullptr) {
		return;
	}
	CHECK_FALSE(ASDebugger::is_attached(bare));
	bare->Release();
}

} // namespace TestAngelScriptDebugger
