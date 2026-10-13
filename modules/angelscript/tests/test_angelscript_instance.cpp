/**************************************************************************/
/*  test_angelscript_instance.cpp                                         */
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

#include "../as_script.h"
#include "../as_script_instance.h"

#include "core/os/os.h"
#include "scene/main/node.h"

#define ANGELSCRIPT_INSTANCE_TESTS_IMPL
#include "test_angelscript_instance.h"

static Ref<ASScript> _make_script(const String &p_class, const String &p_body) {
	Ref<ASScript> script;
	script.instantiate();
	String error;
	const String source = "// godot_base: Node\nclass " + p_class + " {\n" + p_body + "\n}\n";
	script->compile_source(source, "res://" + p_class + ".as", &error);
	CHECK_MESSAGE(script->is_script_valid(), error);
	return script;
}

// 建实例 → 挂到 Node 上 → 返回实例指针；失败返回 nullptr（调用方负责守卫）。
static ScriptInstance *_attach_instance(const Ref<ASScript> &p_script, Node *p_node) {
	ScriptInstance *instance = p_script->instance_create(p_node);
	if (instance == nullptr) {
		return nullptr;
	}
	p_node->set_script_instance(instance);
	return instance;
}

void as_instance_ready_is_dispatched_by_name() {
	Ref<ASScript> script = _make_script("hot_node",
			"\tint ready_count = 0;\n"
			"\tvoid _ready() { ready_count = 7; }\n"
			"\tint get_ready_count() const { return ready_count; }");
	REQUIRE(script.is_valid());
	if (!script.is_valid()) {
		return;
	}

	Node *node = memnew(Node);
	ScriptInstance *instance = _attach_instance(script, node);
	REQUIRE(instance != nullptr);
	if (instance == nullptr) {
		memdelete(node);
		return;
	}

	CHECK(node->get_script_instance() == instance);
	CHECK(instance->has_method("_ready"));
	CHECK_FALSE(instance->has_method("_not_defined"));
	CHECK(instance->call("_ready") == Variant());

	Variant call_result;
	Callable::CallError error;
	call_result = instance->callp("get_ready_count", nullptr, 0, error);
	CHECK(error.error == Callable::CallError::CALL_OK);
	CHECK(call_result == Variant(7));

	Variant property_value;
	CHECK(instance->get("ready_count", property_value));
	CHECK(property_value == Variant(7));

	memdelete(node);
}

void as_instance_notification_reaches_notification() {
	Ref<ASScript> script = _make_script("notified_node",
			"\tint last_notification = 0;\n"
			"\tvoid _notification(int what) { last_notification = what; }\n"
			"\tint get_last_notification() const { return last_notification; }");
	REQUIRE(script.is_valid());
	if (!script.is_valid()) {
		return;
	}

	Node *node = memnew(Node);
	ScriptInstance *instance = _attach_instance(script, node);
	REQUIRE(instance != nullptr);
	if (instance == nullptr) {
		memdelete(node);
		return;
	}

	CHECK(instance->has_method("_notification"));
	instance->notification(42);

	Variant call_result;
	Callable::CallError error;
	call_result = instance->callp("get_last_notification", nullptr, 0, error);
	CHECK(error.error == Callable::CallError::CALL_OK);
	CHECK(call_result == Variant(42));

	memdelete(node);
}

void as_instance_process_receives_delta() {
	Ref<ASScript> script = _make_script("processing_node",
			"\tdouble last_delta = 0.0;\n"
			"\tvoid _process(double delta) { last_delta = delta; }\n"
			"\tdouble get_last_delta() const { return last_delta; }");
	REQUIRE(script.is_valid());
	if (!script.is_valid()) {
		return;
	}

	Node *node = memnew(Node);
	ScriptInstance *instance = _attach_instance(script, node);
	REQUIRE(instance != nullptr);
	if (instance == nullptr) {
		memdelete(node);
		return;
	}

	CHECK(instance->has_method("_process"));
	CHECK(instance->call("_process", 0.25) == Variant());

	Variant call_result;
	Callable::CallError error;
	call_result = instance->callp("get_last_delta", nullptr, 0, error);
	CHECK(error.error == Callable::CallError::CALL_OK);
	CHECK(call_result == Variant(0.25));

	memdelete(node);
}

void as_instance_signature_mismatch_is_not_exposed() {
	Ref<ASScript> script = _make_script("mismatched_node",
			"\tvoid _process() { }");
	REQUIRE(script.is_valid());
	if (!script.is_valid()) {
		return;
	}

	Node *node = memnew(Node);
	ScriptInstance *instance = _attach_instance(script, node);
	REQUIRE(instance != nullptr);
	if (instance == nullptr) {
		memdelete(node);
		return;
	}

	// _process 是引擎虚拟方法，签名不匹配（0 参而非 1 参）就当它不存在。
	CHECK_FALSE(instance->has_method("_process"));

	memdelete(node);
}

void as_instance_requires_default_constructor() {
	// 只声明带参构造时 AngelScript 不会自动生成默认构造；instance_create() 必须返回 nullptr，
	// 而不是交出一个后续任何方法调用都会失败的"哑实例"。
	Ref<ASScript> script = _make_script("no_default_ctor",
			"\tint value = 0;\n"
			"\tno_default_ctor(int v) { value = v; }");
	REQUIRE(script.is_valid());
	if (!script.is_valid()) {
		return;
	}
	REQUIRE(script->is_script_valid());
	CHECK(script->instance_create(nullptr) == nullptr);
}

void as_instance_scalar_properties_round_trip() {
	Ref<ASScript> script = _make_script("scalar_holder",
			"\tbool flag = false;\n"
			"\tint32 count = 0;\n"
			"\tint64 big = 0;\n"
			"\tfloat ratio = 0.0f;\n"
			"\tdouble precise = 0.0;\n");
	REQUIRE(script.is_valid());
	if (!script.is_valid()) {
		return;
	}
	REQUIRE(script->is_script_valid());
	if (!script->is_script_valid()) {
		return;
	}

	Node *node = memnew(Node);
	ScriptInstance *instance = _attach_instance(script, node);
	REQUIRE(instance != nullptr);
	if (instance == nullptr) {
		memdelete(node);
		return;
	}

	CHECK(instance->set("flag", Variant(true)));
	CHECK(instance->set("count", Variant((int64_t)-7)));
	CHECK(instance->set("big", Variant((int64_t)9000000000LL)));
	CHECK(instance->set("ratio", Variant(0.5)));
	CHECK(instance->set("precise", Variant(1.25)));

	Variant value;
	REQUIRE(instance->get("flag", value));
	CHECK(value == Variant(true));
	REQUIRE(instance->get("count", value));
	CHECK(value == Variant((int64_t)-7));
	REQUIRE(instance->get("big", value));
	CHECK(value == Variant((int64_t)9000000000LL));
	REQUIRE(instance->get("ratio", value));
	CHECK(value == Variant(0.5));
	REQUIRE(instance->get("precise", value));
	CHECK(value == Variant(1.25));

	// 未知属性既没有类型，也不应被报成"有效"。
	bool is_valid = true;
	CHECK(instance->get_property_type("does_not_exist", &is_valid) == Variant::NIL);
	CHECK_FALSE(is_valid);

	memdelete(node);
}

void as_instance_scalar_call_plan_is_cached() {
	Ref<ASScript> script = _make_script("cached_plan_node", "\tint64 add4(int64 a, int64 b, int64 c, int64 d) const { return a+b+c+d; }");
	REQUIRE(script.is_valid()); if (!script.is_valid()) return;
	Node *node=memnew(Node); ScriptInstance *instance=_attach_instance(script,node); REQUIRE(instance!=nullptr); if(!instance){memdelete(node);return;}
	ASScriptInstance::reset_method_resolution_count(); Callable::CallError error;
	Variant a=1,b=2,c=3,d=4; const Variant *args[4]={&a,&b,&c,&d};
	for(int i=0;i<5;i++){Variant result=instance->callp("add4",args,4,error);CHECK(error.error==Callable::CallError::CALL_OK);CHECK(result==Variant((int64_t)10));}
	CHECK(ASScriptInstance::method_resolution_count()==1); CHECK(ASScriptInstance::call_plan_analysis_count()==1); memdelete(node);
}

// Simulate a plan-cache relocation at exactly the native -> script reentry edge.
static ASEngine::ASCallPlan *invalidated_plan = nullptr;
static void invalidate_call_plan() {
	if (invalidated_plan) {
		*invalidated_plan = ASEngine::ASCallPlan();
	}
}

void as_instance_call_plan_safety() {
	ASEngine *as = ASEngine::get_singleton();
	if (!as->ensure_initialized()) {
		REQUIRE(false);
		return;
	}
	asIScriptEngine *engine = as->get_engine();
	if (!engine->GetGlobalFunctionByDecl("void invalidate_call_plan()")) {
		REQUIRE(engine->RegisterGlobalFunction("void invalidate_call_plan()", asFUNCTION(invalidate_call_plan), asCALL_CDECL) >= 0);
	}
	String error;
	if (!as->compile_module("call_plan_safety", R"(
		int64 ref_in(const int64 &in a) { return a; }
		void ref_out(int64 &out a) { a=9; }
		int64 four(int64 a,int64 b,int64 c,int64 d) { return a+b+c+d; }
		float scalar_float(float a) { return a+0.5f; }
		bool scalar_bool(bool a) { return !a; }
		int64 reenter() { invalidate_call_plan(); return 42; }
	)",
				&error)) {
		REQUIRE_MESSAGE(false, error);
		return;
	}
	asIScriptModule *mod = engine->GetModule("call_plan_safety");
	asIScriptContext *ctx = as->create_context();
	if (!ctx) {
		REQUIRE(false);
		return;
	}
	Variant value = int64_t(3), ret;
	const Variant *args[4] = { &value, &value, &value, &value };
	for (const char *name : { "ref_in", "ref_out" }) {
		auto plan = ASEngine::make_call_plan(mod->GetFunctionByName(name));
		CHECK_FALSE(plan.fast_scalar_eligible);
		CHECK_EQ(ASEngine::call_function(ctx, plan.function, nullptr, args, 1, &ret, &plan), ERR_INVALID_PARAMETER);
		CHECK_EQ(ctx->GetState(), asEXECUTION_UNINITIALIZED);
	}
	auto plan = ASEngine::make_call_plan(mod->GetFunctionByName("four"));
	CHECK(plan.fast_scalar_eligible);
	CHECK_EQ(ASEngine::call_function(ctx, plan.function, nullptr, args, 4, &ret, &plan), OK);
	CHECK_EQ(int64_t(ret), 12);
	value = 1.25;
	plan = ASEngine::make_call_plan(mod->GetFunctionByName("scalar_float"));
	CHECK_EQ(ASEngine::call_function(ctx, plan.function, nullptr, args, 1, &ret, &plan), OK);
	CHECK_EQ(double(ret), 1.75);
	value = true;
	plan = ASEngine::make_call_plan(mod->GetFunctionByName("scalar_bool"));
	CHECK_EQ(ASEngine::call_function(ctx, plan.function, nullptr, args, 1, &ret, &plan), OK);
	CHECK_FALSE(bool(ret));
	plan = ASEngine::make_call_plan(mod->GetFunctionByName("reenter"));
	invalidated_plan = &plan;
	CHECK_EQ(ASEngine::call_function(ctx, plan.function, nullptr, nullptr, 0, &ret, &plan), OK);
	invalidated_plan = nullptr;
	CHECK_EQ(int64_t(ret), 42);
	CHECK_EQ(ctx->GetState(), asEXECUTION_UNINITIALIZED);
	ctx->Release();
	engine->DiscardModule("call_plan_safety");
}

void as_instance_retained_context_probe() {
	ASEngine *as = ASEngine::get_singleton();
	if (!as->ensure_initialized()) {
		REQUIRE(false);
		return;
	}
	String error;
	if (!as->compile_module("retained_context", R"(
		int destroyed=0;
		class Item { ~Item() { destroyed++; } int value() { return 7; } }
		int ping() { return 1; }
		int fail() { int z=0; return 1/z; }
	)",
				&error)) {
		REQUIRE_MESSAGE(false, error);
		return;
	}
	asIScriptEngine *engine = as->get_engine();
	asIScriptModule *mod = engine->GetModule("retained_context");
	asIScriptContext *ctx = as->create_context();
	if (!ctx) {
		REQUIRE(false);
		return;
	}
	asIScriptFunction *ping = mod->GetFunctionByName("ping");
	const bool timing = OS::get_singleton()->get_environment("AS_BOUNDARY_TIMING") == "1";
	const int n = timing ? 200000 : 3;
	for (bool retain : { false, true }) {
		Vector<uint64_t> times;
		for (int rep = 0; rep < (timing ? 8 : 1); ++rep) {
			int64_t sum = 0;
			const uint64_t start = OS::get_singleton()->get_ticks_usec();
			for (int i = 0; i < n; i++) {
				if (ctx->Prepare(ping) < 0 || ctx->Execute() != asEXECUTION_FINISHED) {
					REQUIRE(false);
					ctx->Release();
					return;
				}
				sum += ctx->GetReturnDWord();
				if (!retain) {
					ctx->Unprepare();
				}
			}
			const uint64_t elapsed = OS::get_singleton()->get_ticks_usec() - start;
			CHECK_EQ(sum, n);
			CHECK_EQ(ctx->GetState(), retain ? asEXECUTION_FINISHED : asEXECUTION_UNINITIALIZED);
			ctx->Unprepare();
			if (rep > 0) {
				times.push_back(elapsed);
			}
		}
		if (timing) {
			times.sort();
			print_line(vformat("RETAIN|%s|%d|%d|%d|%d", retain ? "retained" : "unprepared", n, times[0], times[3], times[6]));
		}
	}
	CHECK_EQ(ctx->Prepare(mod->GetFunctionByName("fail")), asSUCCESS);
	CHECK_EQ(ctx->Execute(), asEXECUTION_EXCEPTION);
	CHECK_EQ(ctx->Prepare(ping), asSUCCESS);
	CHECK_EQ(ctx->Execute(), asEXECUTION_FINISHED);
	CHECK_EQ(ctx->GetReturnDWord(), 1u);
	ctx->Unprepare();
	// A finished method still owns this. Retaining it changes destruction timing.
	asITypeInfo *type = mod->GetTypeInfoByName("Item");
	auto *object = static_cast<asIScriptObject *>(engine->CreateScriptObject(type));
	if (!object) {
		REQUIRE(false);
		ctx->Release();
		return;
	}
	CHECK_EQ(ctx->Prepare(type->GetMethodByName("value")), asSUCCESS);
	CHECK_EQ(ctx->SetObject(object), asSUCCESS);
	CHECK_EQ(ctx->Execute(), asEXECUTION_FINISHED);
	object->Release();
	int *destroyed = static_cast<int *>(mod->GetAddressOfGlobalVar(mod->GetGlobalVarIndexByName("destroyed")));
	CHECK_EQ(*destroyed, 0);
	ctx->Unprepare();
	CHECK_EQ(*destroyed, 1);
	ctx->Release();
	engine->DiscardModule("retained_context");
}

void as_instance_repeated_dispatch_is_cached() {
	Ref<ASScript> script = _make_script("cached_dispatch_node",
			"\tint ping() const { return 5; }");
	REQUIRE(script.is_valid());
	if (!script.is_valid()) {
		return;
	}

	Node *node = memnew(Node);
	ScriptInstance *instance = _attach_instance(script, node);
	REQUIRE(instance != nullptr);
	if (instance == nullptr) {
		memdelete(node);
		return;
	}

	ASScriptInstance::reset_method_resolution_count();
	Callable::CallError error;
	for (int i = 0; i < 5; i++) {
		const Variant result = instance->callp("ping", nullptr, 0, error);
		CHECK(error.error == Callable::CallError::CALL_OK);
		CHECK(result == Variant(5));
	}
	// 5 次同名派发只应触发 1 次昂贵的方法名解析，其余命中每实例缓存。
	CHECK(ASScriptInstance::method_resolution_count() == 1);

	memdelete(node);
}
