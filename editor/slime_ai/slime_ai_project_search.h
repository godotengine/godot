/**************************************************************************/
/*  slime_ai_project_search.h                                             */
/**************************************************************************/

#pragma once

#include "core/variant/dictionary.h"

namespace SlimeAI {

class ProjectSearch {
public:
	static Dictionary search(const String &p_query, int p_offset = 0, int p_limit = 20);
	// Only trusted native code supplies the root. Query arguments never control it.
	static Dictionary search(const String &p_project_root, const String &p_query, int p_offset, int p_limit);
};

} // namespace SlimeAI
