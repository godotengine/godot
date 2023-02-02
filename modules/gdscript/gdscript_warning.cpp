/**************************************************************************/
/*  gdscript_warning.cpp                                                  */
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

#include "gdscript_warning.h"

#include "core/variant/variant.h"

#ifdef DEBUG_ENABLED

String GDScriptWarning::get_message() const {
#define CHECK_SYMBOLS(m_amount) ERR_FAIL_COND_V(symbols.size() < m_amount, String());

	switch (code) {
		case UNASSIGNED_VARIABLE_OP_ASSIGN: {
			CHECK_SYMBOLS(1);
			return "Using assignment with operation but the variable '" + symbols[0] + "' was not previously assigned a value.";
		} break;
		case UNASSIGNED_VARIABLE: {
			CHECK_SYMBOLS(1);
			return "The variable '" + symbols[0] + "' was used but never assigned a value.";
		} break;
		case UNUSED_VARIABLE: {
			CHECK_SYMBOLS(1);
			return "The local variable '" + symbols[0] + "' is declared but never used in the block. If this is intended, prefix it with an underscore: '_" + symbols[0] + "'";
		} break;
		case UNUSED_LOCAL_CONSTANT: {
			CHECK_SYMBOLS(1);
			return "The local constant '" + symbols[0] + "' is declared but never used in the block. If this is intended, prefix it with an underscore: '_" + symbols[0] + "'";
		} break;
		case SHADOWED_VARIABLE: {
			CHECK_SYMBOLS(4);
			return vformat(R"(The local %s "%s" is shadowing an already-declared %s at line %s.)", symbols[0], symbols[1], symbols[2], symbols[3]);
		} break;
		case SHADOWED_VARIABLE_BASE_CLASS: {
			CHECK_SYMBOLS(4);
			return vformat(R"(The local %s "%s" is shadowing an already-declared %s at the base class "%s".)", symbols[0], symbols[1], symbols[2], symbols[3]);
		} break;
		case UNUSED_PRIVATE_CLASS_VARIABLE: {
			CHECK_SYMBOLS(1);
			return "The class variable '" + symbols[0] + "' is declared but never used in the script.";
		} break;
		case UNUSED_PARAMETER: {
			CHECK_SYMBOLS(2);
			return "The parameter '" + symbols[1] + "' is never used in the function '" + symbols[0] + "'. If this is intended, prefix it with an underscore: '_" + symbols[1] + "'";
		} break;
		case UNREACHABLE_CODE: {
			CHECK_SYMBOLS(1);
			return "Unreachable code (statement after return) in function '" + symbols[0] + "()'.";
		} break;
		case UNREACHABLE_PATTERN: {
			return "Unreachable pattern (pattern after wildcard or bind).";
		} break;
		case STANDALONE_EXPRESSION: {
			return "Standalone expression (the line has no effect).";
		} break;
		case NARROWING_CONVERSION: {
			return "Narrowing conversion (float is converted to int and loses precision).";
		} break;
		case INCOMPATIBLE_TERNARY: {
			return "Values of the ternary conditional are not mutually compatible.";
		} break;
		case UNUSED_SIGNAL: {
			CHECK_SYMBOLS(1);
			return "The signal '" + symbols[0] + "' is declared but never emitted.";
		} break;
		case RETURN_VALUE_DISCARDED: {
			CHECK_SYMBOLS(1);
			return "The function '" + symbols[0] + "()' returns a value that will be discarded if not used.";
		} break;
		case PROPERTY_USED_AS_FUNCTION: {
			CHECK_SYMBOLS(2);
			return "The method '" + symbols[0] + "()' was not found in base '" + symbols[1] + "' but there's a property with the same name. Did you mean to access it?";
		} break;
		case CONSTANT_USED_AS_FUNCTION: {
			CHECK_SYMBOLS(2);
			return "The method '" + symbols[0] + "()' was not found in base '" + symbols[1] + "' but there's a constant with the same name. Did you mean to access it?";
		} break;
		case FUNCTION_USED_AS_PROPERTY: {
			CHECK_SYMBOLS(2);
			return "The property '" + symbols[0] + "' was not found in base '" + symbols[1] + "' but there's a method with the same name. Did you mean to call it?";
		} break;
		case INTEGER_DIVISION: {
			return "Integer division, decimal part will be discarded.";
		} break;
		case UNSAFE_PROPERTY_ACCESS: {
			CHECK_SYMBOLS(2);
			return "The property '" + symbols[0] + "' is not present on the inferred type '" + symbols[1] + "' (but may be present on a subtype).";
		} break;
		case UNSAFE_METHOD_ACCESS: {
			CHECK_SYMBOLS(2);
			return "The method '" + symbols[0] + "' is not present on the inferred type '" + symbols[1] + "' (but may be present on a subtype).";
		} break;
		case UNSAFE_CAST: {
			CHECK_SYMBOLS(1);
			return "The value is cast to '" + symbols[0] + "' but has an unknown type.";
		} break;
		case UNSAFE_CALL_ARGUMENT: {
			CHECK_SYMBOLS(4);
			return "The argument '" + symbols[0] + "' of the function '" + symbols[1] + "' requires a the subtype '" + symbols[2] + "' but the supertype '" + symbols[3] + "' was provided";
		} break;
		case UNSAFE_VOID_RETURN: {
			CHECK_SYMBOLS(2);
			return "The method '" + symbols[0] + "()' returns 'void' but it's trying to return a call to '" + symbols[1] + "()' that can't be ensured to also be 'void'.";
		} break;
		case DEPRECATED_KEYWORD: {
			CHECK_SYMBOLS(2);
			return "The '" + symbols[0] + "' keyword is deprecated and will be removed in a future release, please replace its uses by '" + symbols[1] + "'.";
		} break;
		case STANDALONE_TERNARY: {
			return "Standalone ternary conditional operator: the return value is being discarded.";
		}
		case ASSERT_ALWAYS_TRUE: {
			return "Assert statement is redundant because the expression is always true.";
		}
		case ASSERT_ALWAYS_FALSE: {
			return "Assert statement will raise an error because the expression is always false.";
		}
		case REDUNDANT_AWAIT: {
			return R"("await" keyword not needed in this case, because the expression isn't a coroutine nor a signal.)";
		}
		case EMPTY_FILE: {
			return "Empty script file.";
		}
		case SHADOWED_GLOBAL_IDENTIFIER: {
			CHECK_SYMBOLS(3);
			return vformat(R"(The %s '%s' has the same name as a %s.)", symbols[0], symbols[1], symbols[2]);
		}
		case INT_AS_ENUM_WITHOUT_CAST: {
			return "Integer used when an enum value is expected. If this is intended cast the integer to the enum type.";
		}
		case INT_AS_ENUM_WITHOUT_MATCH: {
			CHECK_SYMBOLS(3);
			return vformat(R"(Cannot %s %s as Enum "%s": no enum member has matching value.)", symbols[0], symbols[1], symbols[2]);
		} break;
		case STATIC_CALLED_ON_INSTANCE: {
			CHECK_SYMBOLS(2);
			return vformat(R"(The function '%s()' is a static function but was called from an instance. Instead, it should be directly called from the type: '%s.%s()'.)", symbols[0], symbols[1], symbols[0]);
		}
		case CONFUSABLE_IDENTIFIER: {
			CHECK_SYMBOLS(1);
			return vformat(R"(The identifier "%s" has misleading characters and might be confused with something else.)", symbols[0]);
		}
		case RENAMED_IN_GD4_HINT: {
			break; // Renamed identifier hint is taken care of by the GDScriptAnalyzer. No message needed here.
		}
		case INFERENCE_ON_VARIANT: {
			CHECK_SYMBOLS(1);
			return vformat("The %s type is being inferred from a Variant value, so it will be typed as Variant.", symbols[0]);
		}
		case NATIVE_METHOD_OVERRIDE: {
			CHECK_SYMBOLS(2);
			return vformat(R"(The method "%s" overrides a method from native class "%s". This won't be called by the engine and may not work as expected.)", symbols[0], symbols[1]);
		}
		case GET_NODE_DEFAULT_WITHOUT_ONREADY: {
			CHECK_SYMBOLS(1);
			return vformat(R"*(The default value is using "%s" which won't return nodes in the scene tree before "_ready()" is called. Use the "@onready" annotation to solve this.)*", symbols[0]);
		}
		case ONREADY_WITH_EXPORT: {
			return R"(The "@onready" annotation will make the default value to be set after the "@export" takes effect and will override it.)";
		}
		case WARNING_MAX:
			break; // Can't happen, but silences warning
	}
	ERR_FAIL_V_MSG(String(), "Invalid GDScript warning code: " + get_name_from_code(code) + ".");

#undef CHECK_SYMBOLS
}

int GDScriptWarning::get_default_value(Code p_code) {
	ERR_FAIL_INDEX_V_MSG(p_code, WARNING_MAX, WarnLevel::IGNORE, "Getting default value of invalid warning code.");
	return default_warning_levels[p_code];
}

PropertyInfo GDScriptWarning::get_property_info(Code p_code) {
	// Making this a separate function in case a warning needs different PropertyInfo in the future.
	if (p_code == Code::RENAMED_IN_GD4_HINT) {
		return PropertyInfo(Variant::BOOL, get_settings_path_from_code(p_code));
	}
	return PropertyInfo(Variant::INT, get_settings_path_from_code(p_code), PROPERTY_HINT_ENUM, "Ignore,Warn,Error");
}

String GDScriptWarning::get_name() const {
	return get_name_from_code(code);
}

String GDScriptWarning::get_name_from_code(Code p_code) {
	ERR_FAIL_COND_V(p_code < 0 || p_code >= WARNING_MAX, String());

	static const char *names[] = {
		"UNASSIGNED_VARIABLE",
		"UNASSIGNED_VARIABLE_OP_ASSIGN",
		"UNUSED_VARIABLE",
		"UNUSED_LOCAL_CONSTANT",
		"SHADOWED_VARIABLE",
		"SHADOWED_VARIABLE_BASE_CLASS",
		"UNUSED_PRIVATE_CLASS_VARIABLE",
		"UNUSED_PARAMETER",
		"UNREACHABLE_CODE",
		"UNREACHABLE_PATTERN",
		"STANDALONE_EXPRESSION",
		"NARROWING_CONVERSION",
		"INCOMPATIBLE_TERNARY",
		"UNUSED_SIGNAL",
		"RETURN_VALUE_DISCARDED",
		"PROPERTY_USED_AS_FUNCTION",
		"CONSTANT_USED_AS_FUNCTION",
		"FUNCTION_USED_AS_PROPERTY",
		"INTEGER_DIVISION",
		"UNSAFE_PROPERTY_ACCESS",
		"UNSAFE_METHOD_ACCESS",
		"UNSAFE_CAST",
		"UNSAFE_CALL_ARGUMENT",
		"UNSAFE_VOID_RETURN",
		"DEPRECATED_KEYWORD",
		"STANDALONE_TERNARY",
		"ASSERT_ALWAYS_TRUE",
		"ASSERT_ALWAYS_FALSE",
		"REDUNDANT_AWAIT",
		"EMPTY_FILE",
		"SHADOWED_GLOBAL_IDENTIFIER",
		"INT_AS_ENUM_WITHOUT_CAST",
		"INT_AS_ENUM_WITHOUT_MATCH",
		"STATIC_CALLED_ON_INSTANCE",
		"CONFUSABLE_IDENTIFIER",
		"RENAMED_IN_GODOT_4_HINT",
		"INFERENCE_ON_VARIANT",
		"NATIVE_METHOD_OVERRIDE",
		"GET_NODE_DEFAULT_WITHOUT_ONREADY",
		"ONREADY_WITH_EXPORT",
	};

	static_assert((sizeof(names) / sizeof(*names)) == WARNING_MAX, "Amount of warning types don't match the amount of warning names.");

	return names[(int)p_code];
}

String GDScriptWarning::get_settings_path_from_code(Code p_code) {
	return "debug/gdscript/warnings/" + get_name_from_code(p_code).to_lower();
}

GDScriptWarning::Code GDScriptWarning::get_code_from_name(const String &p_name) {
	for (int i = 0; i < WARNING_MAX; i++) {
		if (get_name_from_code((Code)i) == p_name) {
			return (Code)i;
		}
	}

	return WARNING_MAX;
}

#endif // DEBUG_ENABLED
