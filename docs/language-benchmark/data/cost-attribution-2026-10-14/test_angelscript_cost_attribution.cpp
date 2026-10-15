#include "../as_debugger.h"
#include "../as_engine.h"
#include "../as_script.h"
#include "../as_script_instance.h"
#include "../binding/as_binding_native_storage.h"
#include "../binding/as_binding_native_value_ops.h"

#include "core/os/os.h"
#include "scene/main/node.h"

#define ANGELSCRIPT_COST_ATTRIBUTION_IMPL
#include "test_angelscript_cost_attribution.h"

namespace {
// Diagnostics only: keep the production operators, ABI and lifetimes untouched.
void cost_add2(Vector2 *p_dst, const Vector2 *p_src) {
	*p_dst += *p_src;
}
void cost_add_color(Color *p_dst, const Color *p_src) {
	*p_dst += *p_src;
}
void cost_add2_generic(asIScriptGeneric *p_gen) {
	Vector2 dst = ASNativeValueStorage::native_to_variant(Variant::VECTOR2, p_gen->GetObject());
	Vector2 src = ASNativeValueStorage::native_to_variant(Variant::VECTOR2, p_gen->GetArgObject(0));
	ASNativeValueStorage::variant_to_native(Variant::VECTOR2, Variant(dst + src), p_gen->GetObject());
}
void cost_add_color_generic(asIScriptGeneric *p_gen) {
	Color dst = ASNativeValueStorage::native_to_variant(Variant::COLOR, p_gen->GetObject());
	Color src = ASNativeValueStorage::native_to_variant(Variant::COLOR, p_gen->GetArgObject(0));
	ASNativeValueStorage::variant_to_native(Variant::COLOR, Variant(dst + src), p_gen->GetObject());
}

void emit_cost(const String &p_name, int p_rep, int p_n, uint64_t p_us, int64_t p_checksum) {
	print_line(vformat("COST|%s|%d|%d|%d|%d", p_name, p_rep, p_n, p_us, p_checksum));
}

void dump_bytecode(asIScriptEngine *p_engine, asIScriptFunction *p_func) {
	asUINT length = 0;
	asDWORD *code = p_func->GetByteCode(&length);
	for (asUINT offset = 0; offset < length;) {
		const asBYTE opcode = *(asBYTE *)(code + offset);
		const int size = asBCTypeSize[asBCInfo[opcode].type];
		REQUIRE(size > 0);
		REQUIRE(offset + size <= length);
		String target;
		if (opcode == asBC_CALLSYS || opcode == asBC_CALL) {
			asIScriptFunction *called = p_engine->GetFunctionById((int)code[offset + 1]);
			if (called) {
				target = called->GetDeclaration();
			}
		}
		print_line(vformat("BYTECODE|%s|%d|%s|%s", p_func->GetName(), offset, asBCInfo[opcode].name, target));
		offset += size;
	}
}

// Batch phases change the context working set; these are not additive estimates
// of the interleaved single-context hot path. No per-invocation clock reads.
void phase_costs(ASEngine *p_as, asIScriptFunction *p_ping, bool p_timing) {
	const int width = p_timing ? 256 : 4;
	const int batches = p_timing ? 800 : 1;
	Vector<asIScriptContext *> contexts;
	for (int i = 0; i < width; ++i) {
		contexts.push_back(p_as->create_context());
		REQUIRE(contexts[i] != nullptr);
	}
	for (int rep = 0; rep < (p_timing ? 8 : 1); ++rep) {
		uint64_t prepare_us = 0, execute_us = 0, unprepare_us = 0, clock_us = 0;
		int64_t sum = 0;
		bool ok = true;
		for (int batch = 0; batch < batches; ++batch) {
			uint64_t start = OS::get_singleton()->get_ticks_usec();
			clock_us += OS::get_singleton()->get_ticks_usec() - start;
			start = OS::get_singleton()->get_ticks_usec();
			for (asIScriptContext *ctx : contexts) {
				ok = (ctx->Prepare(p_ping) == asSUCCESS) && ok;
			}
			prepare_us += OS::get_singleton()->get_ticks_usec() - start;
			start = OS::get_singleton()->get_ticks_usec();
			for (asIScriptContext *ctx : contexts) {
				ok = (ctx->Execute() == asEXECUTION_FINISHED) && ok;
				sum += ctx->GetReturnDWord();
			}
			execute_us += OS::get_singleton()->get_ticks_usec() - start;
			start = OS::get_singleton()->get_ticks_usec();
			for (asIScriptContext *ctx : contexts) {
				ok = (ctx->Unprepare() == asSUCCESS) && ok;
			}
			unprepare_us += OS::get_singleton()->get_ticks_usec() - start;
		}
		CHECK(ok);
		CHECK_EQ(sum, width * batches);
		if (p_timing && rep > 0) {
			emit_cost("phase_prepare", rep, width * batches, prepare_us, sum);
			emit_cost("phase_execute_read", rep, width * batches, execute_us, sum);
			emit_cost("phase_unprepare", rep, width * batches, unprepare_us, sum);
			print_line(vformat("CLOCK|%d|%d|%d", rep, batches, clock_us));
		}
	}
	for (asIScriptContext *ctx : contexts) {
		ctx->Release();
	}
}
} // namespace

void as_cost_attribution() {
	fprintf(stderr, "COST_CHECKPOINT|start\n"); fflush(stderr);
	ASEngine *as = ASEngine::get_singleton();
	REQUIRE(as->ensure_initialized());
	asIScriptEngine *engine = as->get_engine();
	fprintf(stderr, "COST_CHECKPOINT|initialized\n"); fflush(stderr);
	// This diagnostic is specifically about the verified native storage path.
	if (!ASNativeValueStorage::is_native_storage_type(Variant::VECTOR2) || !ASNativeValueOps::native_ops_enabled()) {
		print_line("COST_UNSUPPORTED|native storage required");
		return;
	}
	REQUIRE_FALSE(ASNativeValueStorage::is_trivial_probe_type(Variant::VECTOR2));
	if (!engine->GetTypeInfoByName("Vector2")->GetMethodByName("cost_add")) {
		REQUIRE(engine->RegisterObjectMethod("Vector2", "void cost_add(const Vector2 &in)", asFUNCTION(cost_add2), asCALL_CDECL_OBJFIRST) >= 0);
		REQUIRE(engine->RegisterObjectMethod("Color", "void cost_add(const Color &in)", asFUNCTION(cost_add_color), asCALL_CDECL_OBJFIRST) >= 0);
		REQUIRE(engine->RegisterObjectMethod("Vector2", "void cost_generic_add(const Vector2 &in)", asFUNCTION(cost_add2_generic), asCALL_GENERIC) >= 0);
		REQUIRE(engine->RegisterObjectMethod("Color", "void cost_generic_add(const Color &in)", asFUNCTION(cost_add_color_generic), asCALL_GENERIC) >= 0);
	}
	String error;
	const String source = R"(
		int ping() { return 1; }
		int sum4(int a, int b, int c, int d) { return a+b+c+d; }
		int64 vec_scalar(int n) { float x=0,y=0; for(int i=0;i<n;++i) { x+=1; y+=2; } return int64(x+y); }
		int64 vec_operator(int n) { Vector2 a; Vector2 v(1,2); for(int i=0;i<n;++i) a=a+v; return int64(a.x+a.y); }
		int64 vec_inplace(int n) { Vector2 a; Vector2 v(1,2); for(int i=0;i<n;++i) a.cost_add(v); return int64(a.x+a.y); }
		int64 vec_generic_inplace(int n) { Vector2 a; Vector2 v(1,2); for(int i=0;i<n;++i) a.cost_generic_add(v); return int64(a.x+a.y); }
		int64 vec_read(int n) { Vector2 v(1,2); int64 s=0; for(int i=0;i<n;++i) s+=int64(v.x+v.y); return s; }
		int64 vec_copy(int n) { Vector2 v(1,2); int64 s=0; for(int i=0;i<n;++i) { Vector2 a(v); s+=int64(a.x+a.y); } return s; }
		int64 vec_assign(int n) { Vector2 v(1,2); Vector2 a; int64 s=0; for(int i=0;i<n;++i) { a=v; s+=int64(a.x+a.y); } return s; }
		int64 vec_default(int n) { int64 s=0; for(int i=0;i<n;++i) { Vector2 a; s+=int64(a.x+a.y)+1; } return s; }
		int64 color_scalar(int n) { float r=0,g=0,b=0,a=0; for(int i=0;i<n;++i) { r+=1; g+=2; b+=3; a+=4; } return int64(r+g+b+a); }
		int64 color_operator(int n) { Color a(0,0,0,0); Color v(1,2,3,4); for(int i=0;i<n;++i) a=a+v; return int64(a.r+a.g+a.b+a.a); }
		int64 color_inplace(int n) { Color a(0,0,0,0); Color v(1,2,3,4); for(int i=0;i<n;++i) a.cost_add(v); return int64(a.r+a.g+a.b+a.a); }
		int64 color_generic_inplace(int n) { Color a(0,0,0,0); Color v(1,2,3,4); for(int i=0;i<n;++i) a.cost_generic_add(v); return int64(a.r+a.g+a.b+a.a); }
		int64 color_read(int n) { Color v(1,2,3,4); int64 s=0; for(int i=0;i<n;++i) s+=int64(v.r+v.g+v.b+v.a); return s; }
		int64 color_copy(int n) { Color v(1,2,3,4); int64 s=0; for(int i=0;i<n;++i) { Color a(v); s+=int64(a.r+a.g+a.b+a.a); } return s; }
		int64 color_assign(int n) { Color v(1,2,3,4); Color a; int64 s=0; for(int i=0;i<n;++i) { a=v; s+=int64(a.r+a.g+a.b+a.a); } return s; }
		int64 color_default(int n) { int64 s=0; for(int i=0;i<n;++i) { Color a; s+=int64(a.a); } return s; }
		int semantics() { Vector2 v(1,2); Vector2 b(v); b.x=99; Color c; Color d(c); d.a=7; return v.x==1 && c.a==1 && d.a==7 ? 1 : 0; }
	)";
	fprintf(stderr, "COST_CHECKPOINT|compile\n"); fflush(stderr);
	Vector<ASEngine::ASMessage> messages;
	const bool diagnostics_ok = as->compile_for_diagnostics("cost_preflight", source, &messages, nullptr);
	for (const auto &message : messages) {
		fprintf(stderr, "COST_DIAGNOSTIC|%s\n", message.message.utf8().get_data()); fflush(stderr);
	}
	if (!diagnostics_ok) {
		CHECK(diagnostics_ok);
		return;
	}
	if (!as->compile_module("cost_attribution", source, &error)) {
		CHECK_MESSAGE(false, error);
		return;
	}
	fprintf(stderr, "COST_CHECKPOINT|compiled\n"); fflush(stderr);
	asIScriptModule *module = engine->GetModule("cost_attribution");
	asIScriptContext *ctx = as->create_context();
	REQUIRE(ctx != nullptr);
	CHECK_FALSE(ASDebugger::is_line_enabled(ctx));
	REQUIRE(ctx->Prepare(module->GetFunctionByName("semantics")) == asSUCCESS);
	REQUIRE(ctx->Execute() == asEXECUTION_FINISHED);
	CHECK_EQ(ctx->GetReturnDWord(), 1u);
	ctx->Unprepare();

	const bool timing = OS::get_singleton()->get_environment("AS_COST_TIMING") == "1";
	const bool reverse = OS::get_singleton()->get_environment("AS_COST_REVERSE") == "1";
	const int n = timing ? 200000 : 32;
	Ref<ASScript> script;
	script.instantiate();
	REQUIRE_MESSAGE(script->compile_source("// godot_base: Node\nclass cost_node { int ping() { return 1; } }", "res://cost_node.as", &error), error);
	Node *node = memnew(Node);
	ScriptInstance *instance = script->instance_create(node);
	REQUIRE(instance != nullptr);
	node->set_script_instance(instance);
	const StringName ping_name("ping");
	Callable::CallError ce;
	CHECK_EQ(int64_t(instance->callp(ping_name, nullptr, 0, ce)), 1);
	ASScriptInstance::reset_method_resolution_count();
	asIScriptFunction *ping = module->GetFunctionByName("ping");
	asIScriptFunction *sum4 = module->GetFunctionByName("sum4");
	const ASEngine::ASCallPlan ping_plan = ASEngine::make_call_plan(ping);
	const ASEngine::ASCallPlan sum_plan = ASEngine::make_call_plan(sum4);
	Variant args[4] = { 1, 2, 3, 4 };
	const Variant *argv[4] = { &args[0], &args[1], &args[2], &args[3] };
	const char *paths[] = { "raw0", "scoped0", "bridge_plan0", "bridge_unplanned0", "raw4", "bridge_plan4", "bridge_unplanned4", "instance0", "object0" };
	for (int index = 0; index < 9; ++index) {
		const int path = reverse ? 8-index : index;
		fprintf(stderr, "COST_CHECKPOINT|dispatch|%s\n", paths[path]); fflush(stderr);
		const bool four = path >= 4 && path <= 6;
		for (int rep = 0; rep < (timing ? 8 : 1); ++rep) {
			int64_t sum = 0;
			bool ok = true;
			const uint64_t start = OS::get_singleton()->get_ticks_usec();
			for (int i = 0; i < n; ++i) {
				if (path == 0 || path == 1 || path == 4) {
					ok = (ctx->Prepare(four ? sum4 : ping) == asSUCCESS) && ok;
					if (four) {
						for (int a = 0; a < 4; ++a) ok = (ctx->SetArgDWord(a, a+1) == asSUCCESS) && ok;
					}
					int executed;
					if (path == 1) { ASDebugger::ExecutionScope scope(ctx); executed = ctx->Execute(); }
					else executed = ctx->Execute();
					ok = (executed == asEXECUTION_FINISHED) && ok;
					sum += ctx->GetReturnDWord();
					ok = (ctx->Unprepare() == asSUCCESS) && ok;
				} else if (path < 7) {
					Variant returned;
					const auto *plan = (path == 2 || path == 5) ? (four ? &sum_plan : &ping_plan) : nullptr;
					ok = (ASEngine::call_function(ctx, four ? sum4 : ping, nullptr, four ? argv : nullptr, four ? 4 : 0, &returned, plan) == OK) && ok;
					sum += int64_t(returned);
				} else {
					Variant returned = path == 7 ? instance->callp(ping_name, nullptr, 0, ce) : node->callp(ping_name, nullptr, 0, ce);
					ok = (ce.error == Callable::CallError::CALL_OK) && ok;
					sum += int64_t(returned);
				}
			}
			const uint64_t elapsed = OS::get_singleton()->get_ticks_usec() - start;
			CHECK(ok);
			CHECK_EQ(sum, int64_t(n) * (four ? 10 : 1));
			CHECK_EQ(ctx->GetState(), asEXECUTION_UNINITIALIZED);
			if (timing && rep > 0) emit_cost(paths[path], rep, n, elapsed, sum);
		}
	}
	CHECK_EQ(ASScriptInstance::method_resolution_count(), 0u);
	memdelete(node);
	phase_costs(as, ping, timing);

	const char *geometry[] = { "vec_scalar", "vec_operator", "vec_inplace", "vec_generic_inplace", "vec_read", "vec_copy", "vec_assign", "vec_default", "color_scalar", "color_operator", "color_inplace", "color_generic_inplace", "color_read", "color_copy", "color_assign", "color_default" };
	for (int index = 0; index < 16; ++index) {
		const String name(geometry[reverse ? 15-index : index]);
		fprintf(stderr, "COST_CHECKPOINT|geometry|%s\n", name.utf8().get_data()); fflush(stderr);
		asIScriptFunction *func = module->GetFunctionByName(name.utf8().get_data());
		REQUIRE(func != nullptr);
		const int factor = name.ends_with("default") ? 1 : name.begins_with("vec") ? 3 : 10;
		if (timing) dump_bytecode(engine, func);
		for (int rep = -1; rep < (timing ? 8 : 1); ++rep) {
			const int count = rep < 0 ? 32 : n;
			ASNativeValueOps::reset_native_thunk_calls();
			const uint64_t start = OS::get_singleton()->get_ticks_usec();
			int rc = ctx->Prepare(func);
			rc |= ctx->SetArgDWord(0, count);
			int executed = ctx->Execute();
			int64_t sum = ctx->GetReturnQWord();
			rc |= ctx->Unprepare();
			const uint64_t elapsed = OS::get_singleton()->get_ticks_usec() - start;
			CHECK_EQ(rc, asSUCCESS);
			CHECK_EQ(executed, asEXECUTION_FINISHED);
			CHECK_EQ(sum, int64_t(count) * factor);
			if (timing && rep > 0) emit_cost(name, rep, count, elapsed, sum);
			if (timing && rep < 0) print_line(vformat("COUNTER|%s|%d|%d", name, count, ASNativeValueOps::native_thunk_calls()));
		}
	}
	ctx->Release();
	engine->DiscardModule("cost_attribution");
}
