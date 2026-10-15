#include <angelscript.h>
#include "core/os/os.h"
#define ANGELSCRIPT_PROPERTY_REGISTRATION_IMPL
#include "test_angelscript_property_registration.h"

namespace {
void unused_property(asIScriptGeneric *) {}
void ignore_message(const asSMessageInfo *, void *) {}
}

void as_property_registration_regression() {
	for (bool setter_first : { false, true }) {
		for (bool mismatch : { false, true }) {
			asIScriptEngine *engine = asCreateScriptEngine();
			engine->SetMessageCallback(asFUNCTION(ignore_message), nullptr, asCALL_CDECL);
			CHECK(engine->RegisterObjectType("P", 0, asOBJ_REF | asOBJ_NOCOUNT) >= 0);
			CHECK(engine->RegisterObjectType("Other", 0, asOBJ_REF | asOBJ_NOCOUNT) >= 0);
			CHECK(engine->RegisterObjectMethod("Other", "float get_value() const property", asFUNCTION(unused_property), asCALL_GENERIC) >= 0);
			const char *getter = "int get_value() const property";
			const char *setter = mismatch ? "void set_value(float) property" : "void set_value(int) property";
			CHECK(engine->RegisterObjectMethod("P", setter_first ? setter : getter, asFUNCTION(unused_property), asCALL_GENERIC) >= 0);
			const int result = engine->RegisterObjectMethod("P", setter_first ? getter : setter, asFUNCTION(unused_property), asCALL_GENERIC);
			CHECK_EQ(result >= 0, !mismatch);
			if (mismatch) CHECK_EQ(result, asINVALID_DECLARATION);
			engine->ShutDownAndRelease();
		}
	}
	asIScriptEngine *engine = asCreateScriptEngine();
	engine->SetMessageCallback(asFUNCTION(ignore_message), nullptr, asCALL_CDECL);
	CHECK(engine->RegisterObjectType("Indexed", 0, asOBJ_REF | asOBJ_NOCOUNT) >= 0);
	CHECK(engine->RegisterObjectMethod("Indexed", "int get_item(int) const property", asFUNCTION(unused_property), asCALL_GENERIC) >= 0);
	CHECK(engine->RegisterObjectMethod("Indexed", "void set_item(int, int) property", asFUNCTION(unused_property), asCALL_GENERIC) >= 0);
	CHECK(engine->RegisterGlobalFunction("int get_global_value() property", asFUNCTION(unused_property), asCALL_GENERIC) >= 0);
	CHECK(engine->RegisterGlobalFunction("void set_global_value(int) property", asFUNCTION(unused_property), asCALL_GENERIC) >= 0);
	asIScriptModule *module = engine->GetModule("props", asGM_ALWAYS_CREATE);
	module->AddScriptSection("props", "class S { int v; int get_value() const property { return v; } void set_value(int x) property { v=x; } } int main() { S s; s.value=7; return s.value; }");
	CHECK_EQ(module->Build(), asSUCCESS);
	asIScriptContext *ctx = engine->CreateContext();
	CHECK_EQ(ctx->Prepare(module->GetFunctionByName("main")), asSUCCESS);
	CHECK_EQ(ctx->Execute(), asEXECUTION_FINISHED);
	CHECK_EQ(ctx->GetReturnDWord(), 7u);
	ctx->Release();
	// Many unrelated registered functions must not change member-pair semantics.
	for (int i = 0; i < 256; ++i) {
		const String name = "PropertyScale" + itos(i);
		const CharString cname = name.utf8();
		CHECK(engine->RegisterObjectType(cname.get_data(), 0, asOBJ_REF | asOBJ_NOCOUNT) >= 0);
		CHECK(engine->RegisterObjectMethod(cname.get_data(), "int get_value() const property", asFUNCTION(unused_property), asCALL_GENERIC) >= 0);
		CHECK(engine->RegisterObjectMethod(cname.get_data(), "void set_value(int) property", asFUNCTION(unused_property), asCALL_GENERIC) >= 0);
	}
	engine->ShutDownAndRelease();
	engine = asCreateScriptEngine();
	engine->SetMessageCallback(asFUNCTION(ignore_message), nullptr, asCALL_CDECL);
	CHECK(engine->RegisterGlobalFunction("int get_bad() property", asFUNCTION(unused_property), asCALL_GENERIC) >= 0);
	CHECK_EQ(engine->RegisterGlobalFunction("void set_bad(float) property", asFUNCTION(unused_property), asCALL_GENERIC), asINVALID_DECLARATION);
	engine->ShutDownAndRelease();
}
