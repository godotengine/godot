#pragma once

#include "tests/test_macros.h"

// 用例注册头：不得 include 任何 AngelScript 头；实现见 test_angelscript_editor_language.cpp。
// EditorLanguage 仅在 TOOLS_ENABLED 构建中存在。

#ifdef TOOLS_ENABLED

void as_editor_language_validate_syntax_error();
void as_editor_language_validate_valid_script();
void as_editor_language_complete_code();
void as_editor_language_find_function();

#ifndef ANGELSCRIPT_EDITOR_LANGUAGE_TESTS_IMPL

TEST_CASE("[AngelScript] editor language validate reports syntax errors") {
	as_editor_language_validate_syntax_error();
}
TEST_CASE("[AngelScript] editor language validate accepts valid script") {
	as_editor_language_validate_valid_script();
}
TEST_CASE("[AngelScript] editor language complete code") {
	as_editor_language_complete_code();
}
TEST_CASE("[AngelScript] editor language find function") {
	as_editor_language_find_function();
}

#endif // ANGELSCRIPT_EDITOR_LANGUAGE_TESTS_IMPL

#endif // TOOLS_ENABLED
