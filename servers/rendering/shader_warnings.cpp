/*************************************************************************/
/*  shader_warnings.cpp                                                  */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#include "shader_warnings.h"
#include "core/variant/variant.h"

#ifdef DEBUG_ENABLED

ShaderWarning::Code ShaderWarning::get_code() const {
	return code;
}

int ShaderWarning::get_line() const {
	return line;
}

const StringName &ShaderWarning::get_subject() const {
	return subject;
}

String ShaderWarning::get_message() const {
	switch (code) {
		case FLOAT_COMPARISON:
			return vformat("Direct floating-point comparison (this may not evaluate to `true` as you expect). Instead, use `abs(a - b) < 0.0001` for an approximate but predictable comparison.");
		case UNUSED_CONSTANT:
			return vformat("The const '%s' is declared but never used.", subject);
		case UNUSED_FUNCTION:
			return vformat("The function '%s' is declared but never used.", subject);
		case UNUSED_STRUCT:
			return vformat("The struct '%s' is declared but never used.", subject);
		case UNUSED_UNIFORM:
			return vformat("The uniform '%s' is declared but never used.", subject);
		case UNUSED_VARYING:
			return vformat("The varying '%s' is declared but never used.", subject);
		default:
			break;
	}
	return String();
}

String ShaderWarning::get_name() const {
	return get_name_from_code(code);
}

String ShaderWarning::get_name_from_code(Code p_code) {
	ERR_FAIL_INDEX_V(p_code, WARNING_MAX, String());

	static const char *names[] = {
		"FLOAT_COMPARISON",
		"UNUSED_CONSTANT",
		"UNUSED_FUNCTION",
		"UNUSED_STRUCT",
		"UNUSED_UNIFORM",
		"UNUSED_VARYING",
	};

	static_assert((sizeof(names) / sizeof(*names)) == WARNING_MAX, "Amount of warning types don't match the amount of warning names.");

	return names[(int)p_code];
}

ShaderWarning::Code ShaderWarning::get_code_from_name(const String &p_name) {
	for (int i = 0; i < WARNING_MAX; i++) {
		if (get_name_from_code((Code)i) == p_name) {
			return (Code)i;
		}
	}

	ERR_FAIL_V_MSG(WARNING_MAX, "Invalid shader warning name: " + p_name);
}

static Map<int, uint32_t> *code_to_flags_map = nullptr;

static void init_code_to_flags_map() {
	code_to_flags_map = memnew((Map<int, uint32_t>));
	code_to_flags_map->insert(ShaderWarning::FLOAT_COMPARISON, ShaderWarning::FLOAT_COMPARISON_FLAG);
	code_to_flags_map->insert(ShaderWarning::UNUSED_CONSTANT, ShaderWarning::UNUSED_CONSTANT_FLAG);
	code_to_flags_map->insert(ShaderWarning::UNUSED_FUNCTION, ShaderWarning::UNUSED_FUNCTION_FLAG);
	code_to_flags_map->insert(ShaderWarning::UNUSED_STRUCT, ShaderWarning::UNUSED_STRUCT_FLAG);
	code_to_flags_map->insert(ShaderWarning::UNUSED_UNIFORM, ShaderWarning::UNUSED_UNIFORM_FLAG);
	code_to_flags_map->insert(ShaderWarning::UNUSED_VARYING, ShaderWarning::UNUSED_VARYING_FLAG);
}

ShaderWarning::CodeFlags ShaderWarning::get_flags_from_codemap(const Map<Code, bool> &p_map) {
	uint32_t result = 0U;

	if (code_to_flags_map == nullptr) {
		init_code_to_flags_map();
	}

	for (Map<Code, bool>::Element *E = p_map.front(); E; E = E->next()) {
		if (E->get()) {
			ERR_FAIL_COND_V(!code_to_flags_map->has((int)E->key()), ShaderWarning::NONE_FLAG);
			result |= (*code_to_flags_map)[(int)E->key()];
		}
	}
	return (CodeFlags)result;
}

ShaderWarning::ShaderWarning(Code p_code, int p_line, const StringName &p_subject) :
		code(p_code), line(p_line), subject(p_subject) {
}

#endif // DEBUG_ENABLED
