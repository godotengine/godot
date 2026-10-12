/**************************************************************************/
/*  test_angelscript_typed_containers.cpp                                 */
/**************************************************************************/
#include "../as_engine.h"
#include "../binding/as_typed_containers.h"

#define ANGELSCRIPT_TYPED_CONTAINERS_TESTS_IMPL
#include "test_angelscript_typed_containers.h"

#include "core/string/print_string.h"
#include "scene/main/node.h"

#include <angelscript.h>

static bool run_typed_i64(const String &p_name, const String &p_source, int64_t *r_out, String *r_error = nullptr) {
	ASEngine *as = ASEngine::get_singleton();
	if (!as->ensure_initialized() || ASTypedContainers::register_all(as->get_engine()) != OK) {
		return false;
	}
	String error;
	if (!as->compile_module(p_name, p_source, &error)) {
		if (r_error) *r_error = error;
		return false;
	}
	asIScriptFunction *func = as->get_engine()->GetModule(p_name.utf8().get_data())->GetFunctionByName("main");
	if (!func) return false;
	asIScriptContext *ctx = as->create_context();
	if (!ctx || ctx->Prepare(func) < 0) {
		if (ctx) ctx->Release();
		return false;
	}
	const int rc = ctx->Execute();
	if (rc == asEXECUTION_FINISHED && r_out) *r_out = (int64_t)ctx->GetReturnQWord();
	if (rc != asEXECUTION_FINISHED && r_error) {
		*r_error = ctx->GetExceptionString() ? String(ctx->GetExceptionString()) : String("execution failed");
	}
	ctx->Release();
	return rc == asEXECUTION_FINISHED;
}

void as_typed_containers_register_all() {
	ASEngine *as = ASEngine::get_singleton();
	REQUIRE(as->ensure_initialized());
	REQUIRE(ASTypedContainers::register_all(as->get_engine()) == OK);
	const char *names[] = { "Int64Array", "Float64Array", "Vector2Array", "Vector3Array", "ColorArray", "StringNameArray", "ObjectArray", "Int64Int64Map", "StringNameInt64Map", "StringNameFloat64Map", "StringNameVector2Map", "StringNameColorMap", "StringNameObjectMap" };
	for (const char *name : names) CHECK(as->get_engine()->GetTypeInfoByName(name) != nullptr);
}

void as_typed_containers_numeric_arrays() {
	int64_t out = 0;
	REQUIRE(run_typed_i64("typed_numeric", R"(
int64 main() {
    Int64Array@ a = Int64Array();
    a.reserve(4); a.push_back(1); a.push_back(2); a.push_back(3);
    a.add_scalar(2); a.multiply_scalar(3);
    return a.sum() + a.min_value() + a.max_value();
})", &out));
	CHECK(out == 60); // [9,12,15]: sum 36 + min 9 + max 15.
}

void as_typed_containers_vector_arrays() {
	int64_t out = 0;
	REQUIRE(run_typed_i64("typed_vectors", R"(
int64 main() {
    Vector2Array@ a = Vector2Array(); Vector2Array@ b = Vector2Array();
    a.resize(2); b.resize(2);
    a.set(0, Vector2(1,2)); a.set(1, Vector2(3,4));
    b.fill(Vector2(2,1)); a.add_scaled_in_place(b, 2.0);
    return int64(a.sum().x + a.sum().y);
})", &out));
	CHECK(out == 22);
}

void as_typed_containers_maps() {
	int64_t out = 0;
	REQUIRE(run_typed_i64("typed_maps", R"(
int64 main() {
    StringNameInt64Map@ m = StringNameInt64Map();
    StringName k("score"); m.set(k, 41); m.set(k, m.get_or(k, 0) + 1);
    return m.has(k) ? m.get_or(k, 0) : -1;
})", &out));
	CHECK(out == 42);
}

void as_typed_containers_bounds_exception() {
	int64_t out = 0;
	String error;
	CHECK_FALSE(run_typed_i64("typed_bounds", "int64 main() { Int64Array@ a = Int64Array(); return a.get(0); }", &out, &error));
	CHECK(error.contains("index out of bounds"));
}

void as_typed_containers_object_ids() {
	// null 句柄也以空 ObjectID 安全保存；不会制造由非拥有 Node 句柄导致的测试泄漏。
	int64_t out = 0;
	String error;
	const bool ok = run_typed_i64("typed_objects", R"(
int64 main() {
    ObjectArray@ a = ObjectArray(); Node@ n = null; a.push_back(n);
    return (a.get(0) is null) ? a.remove_invalid() : -1;
})", &out, &error);
	if (!ok) MESSAGE("typed object error: ", error);
	REQUIRE(ok);
	CHECK(out == 1);
}

void as_typed_containers_bulk_conversion() {
	int64_t out = 0;
	REQUIRE(run_typed_i64("typed_bulk", R"(
int64 main() {
    PackedInt64Array p; p.push_back(4); p.push_back(5);
    Int64Array@ a = Int64Array(); a.assign_packed(p);
    PackedInt64Array q = a.to_packed();
    return q[0] + q[1] + a.size();
})", &out));
	CHECK(out == 11);
}
