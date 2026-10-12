#include "tests/test_macros.h"

#ifdef TOOLS_ENABLED

#include "../as_engine.h"
#include "../as_script_language.h"
#include "core/object/editor_language.h"

#define ANGELSCRIPT_EDITOR_LANGUAGE_TESTS_IMPL
#include "test_angelscript_editor_language.h"

static EditorLanguage *editor_language() {
	ASEngine::get_singleton()->ensure_initialized();
	return ASScriptLanguage::get_singleton()->get_editor_language();
}

void as_editor_language_validate_syntax_error() {
	EditorLanguage *lang = editor_language();

	List<EditorLanguage::ScriptError> errors;
	const bool ok = lang->validate("double main() { return 1.0 ", "res://x.as", &errors, nullptr, nullptr, nullptr);
	CHECK_FALSE(ok);
	CHECK(errors.size() >= 1);
	if (!errors.is_empty()) {
		REQUIRE(errors.front() != nullptr);
		CHECK(errors.front()->get().start_line >= 1);
		CHECK_FALSE(errors.front()->get().message.is_empty());
	}
}

void as_editor_language_validate_valid_script() {
	EditorLanguage *lang = editor_language();

	List<EditorLanguage::ScriptError> errors;
	List<String> functions;
	const bool ok = lang->validate("double main() { return 42.0; }\nint helper() { return 1; }\n", "res://x.as", &errors, nullptr, &functions, nullptr);
	CHECK(ok);
	CHECK(errors.is_empty());

	bool has_main = false;
	bool has_helper = false;
	for (const String &f : functions) {
		if (f == "main") {
			has_main = true;
		}
		if (f == "helper") {
			has_helper = true;
		}
	}
	CHECK(has_main);
	CHECK(has_helper);
}

void as_editor_language_complete_code() {
	EditorLanguage *lang = editor_language();

	String code = "Vector";
	code += String::chr(0xFFFF);
	code += "();";
	List<EditorLanguage::CompletionOption> options;
	bool force = false;
	String call_hint;
	REQUIRE(lang->complete_code(code, "res://x.as", nullptr, &options, force, call_hint) == OK);
	CHECK_FALSE(force);

	bool has_vector2 = false;
	bool has_vector3 = false;
	for (const EditorLanguage::CompletionOption &o : options) {
		if (o.display == "Vector2") {
			has_vector2 = true;
		}
		if (o.display == "Vector3") {
			has_vector3 = true;
		}
	}
	CHECK(has_vector2);
	CHECK(has_vector3);

	// 未知前缀不返回任何候选。
	String unknown = "Zzz";
	unknown += String::chr(0xFFFF);
	unknown += "();";
	List<EditorLanguage::CompletionOption> unknown_options;
	REQUIRE(lang->complete_code(unknown, "res://x.as", nullptr, &unknown_options, force, call_hint) == OK);
	CHECK(unknown_options.is_empty());
}

void as_editor_language_find_function() {
	EditorLanguage *lang = editor_language();

	const String code = "int helper() {\n\treturn 1;\n}\ndouble main() {\n\treturn 2.0;\n}\n";
	CHECK(lang->find_function("helper", code) == 0);
	CHECK(lang->find_function("main", code) == 3);
	CHECK(lang->find_function("missing", code) == -1);
}

#endif // TOOLS_ENABLED
