#include "../as_engine.h"
#include "../binding/as_binding_native_value_ops.h"
#include "../binding/as_binding_value_types.h"

#include "core/os/os.h"

#define ANGELSCRIPT_BOUNDARY_PROBES_IMPL
#include "test_angelscript_boundary_probes.h"

// Diagnostic workload, NOT a replacement for the unchanged language benchmark.
static int64_t probe_identity(int64_t p_value) {
	return p_value;
}

void as_boundary_probes() {
	ASEngine *as = ASEngine::get_singleton();
	if (!as->ensure_initialized()) {
		REQUIRE(false);
		return;
	}
	asIScriptEngine *engine = as->get_engine();
	if (engine->GetGlobalFunctionByDecl("int64 boundary_identity(int64)") == nullptr) {
		REQUIRE(engine->RegisterGlobalFunction("int64 boundary_identity(int64)", asFUNCTION(probe_identity), asCALL_CDECL) >= 0);
	}
	String error;
	const String source = R"(
		Array a;
		Dictionary d;
		PackedInt64Array p;
		Int64Array@ t = Int64Array();
		Variant key("value");
		Variant one(int64(1));
		void setup() {
			a.clear(); p.clear(); t.clear(); t.reserve(4096);
			for (int i=0; i<4096; ++i) { a.push_back(one); p.push_back(1); t.push_back(1); }
			d[key] = one;
		}
		int64 empty_native(int n) { int64 x=0; for(int i=0;i<n;++i) x+=boundary_identity(1); return x; }
		int64 arithmetic(int n) { int64 x=0; for(int i=0;i<n;++i) x+=1; return x; }
		int64 array_size(int n) { int64 x=0; for(int i=0;i<n;++i) x+=a.size(); return x; }
		int64 array_has(int n) { int64 x=0; for(int i=0;i<n;++i) if(a.has(one)) ++x; return x; }
		int64 array_get(int n) { int64 x=0; for(int i=0;i<n;++i) x+=int64(a[i&4095]); return x; }
		int64 array_push(int n) { Array v; for(int i=0;i<n;++i) v.push_back(one); return v.size(); }
		int64 literal_key(int n) { int64 x=0; for(int i=0;i<n;++i) { Variant v("value"); if(v==key) ++x; } return x; }
		int64 copy_key(int n) { int64 x=0; for(int i=0;i<n;++i) { Variant v(key); if(v==key) ++x; } return x; }
		int64 dict_cached(int n) { int64 x=0; for(int i=0;i<n;++i) x+=int64(d[key]); return x; }
		int64 dict_literal(int n) { int64 x=0; for(int i=0;i<n;++i) x+=int64(d[Variant("value")]); return x; }
		int64 string_literal(int n) { String x; for(int i=0;i<n;++i) x=x+"x"; return x.length(); }
		int64 packed_get(int n) { int64 x=0; for(int i=0;i<n;++i) x+=p[i&4095]; return x; }
		int64 packed_push(int n) { PackedInt64Array v; for(int i=0;i<n;++i) v.push_back(1); return v.size(); }
		int64 typed_get(int n) { int64 x=0; for(int i=0;i<n;++i) x+=t.get(i&4095); return x; }
		int64 typed_push(int n) { Int64Array@ v=Int64Array(); v.reserve(n); for(int i=0;i<n;++i) v.push_back(1); return v.size(); }
		int64 vector_add(int n) { Vector2 x; Vector2 v(1,2); for(int i=0;i<n;++i) x=x+v; return int64(x.x); }
		int64 vector_assign(int n) { Vector2 x; Vector2 v(1,2); for(int i=0;i<n;++i) x=v; return int64(x.x); }
		int64 color_add(int n) { Color x(0,0,0,0); Color v(1,2,3,4); for(int i=0;i<n;++i) x=x+v; return int64(x.r); }
		int64 dispatch() { return 1; }
	)";
	if (!as->compile_module("boundary_probes", source, &error)) {
		print_line(error);
		REQUIRE(false);
		return;
	}
	asIScriptModule *module = engine->GetModule("boundary_probes");
	asIScriptContext *ctx = as->create_context();
	if (!ctx || ctx->Prepare(module->GetFunctionByName("setup")) < 0) {
		if (ctx) {
			ctx->Release();
		}
		REQUIRE(false);
		return;
	}
	int result = ctx->Execute();
	ctx->Unprepare();
	if (result != asEXECUTION_FINISHED) {
		ctx->Release();
		REQUIRE(false);
		return;
	}
	const bool timing = OS::get_singleton()->get_environment("AS_BOUNDARY_TIMING") == "1";
	const char *names[] = { "empty_native", "arithmetic", "array_size", "array_has", "array_get", "array_push", "literal_key", "copy_key", "dict_cached", "dict_literal", "string_literal", "packed_get", "packed_push", "typed_get", "typed_push", "vector_add", "vector_assign", "color_add", "dispatch" };
	for (const char *name : names) {
		asIScriptFunction *func = module->GetFunctionByName(name);
		const bool dispatch = String(name) == "dispatch";
		const int n = !timing ? 32 : String(name) == "string_literal" ? 8000
																	  : 200000;
		const int64_t expected = String(name) == "vector_assign" ? 1 : String(name) == "array_size" ? int64_t(n) * 4096
																									: n;
		Vector<uint64_t> samples;
		for (int rep = 0; rep < (timing ? 8 : 1); ++rep) {
			int64_t checksum = 0;
			uint64_t start = OS::get_singleton()->get_ticks_usec();
			if (dispatch) {
				ASEngine::ASCallPlan plan = ASEngine::make_call_plan(func);
				for (int i = 0; i < n; ++i) {
					Variant returned;
					if (ASEngine::call_function(ctx, func, nullptr, nullptr, 0, &returned, &plan) != OK) {
						result = asEXECUTION_ERROR;
						break;
					}
					checksum += int64_t(returned);
				}
			} else {
				ctx->Prepare(func);
				ctx->SetArgDWord(0, n);
				result = ctx->Execute();
				checksum = ctx->GetReturnQWord();
				ctx->Unprepare();
			}
			uint64_t elapsed = OS::get_singleton()->get_ticks_usec() - start;
			CHECK_EQ(result, asEXECUTION_FINISHED);
			CHECK_EQ(checksum, expected);
			if (rep > 0) {
				samples.push_back(elapsed);
			}
		}
		if (timing) {
			samples.sort();
			print_line(vformat("PROBE|%s|%d|%d|%d|%d|%d", name, n, samples[0], samples[3], samples[6], expected));
		}
	}
	ctx->Release();
	engine->DiscardModule("boundary_probes");
}
