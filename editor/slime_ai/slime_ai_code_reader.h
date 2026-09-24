/**************************************************************************/
/*  slime_ai_code_reader.h                                                */
/**************************************************************************/

#pragma once

#include "core/variant/dictionary.h"

namespace SlimeAI {

// The root is supplied by trusted native code, never by a model argument.
class CodeReader {
public:
	static bool resolve_path(const String &p_project_root, const String &p_relative_path, String &r_absolute_path, String &r_error);
	static Dictionary load_document(const String &p_project_root, const String &p_relative_path);
	static Dictionary read(const String &p_relative_path, int p_start_line = 1, int p_line_limit = 80);
	static Dictionary read(const String &p_project_root, const String &p_relative_path, int p_start_line = 1, int p_line_limit = 80);
};

} // namespace SlimeAI
