/*************************************************************************/
/*  visual_script_builtin_funcs.cpp                                      */
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

#include "visual_script_builtin_funcs.h"

#include "core/io/marshalls.h"
#include "core/math/math_funcs.h"
#include "core/object/class_db.h"
#include "core/object/ref_counted.h"
#include "core/os/os.h"
#include "core/variant/variant_parser.h"

const char *VisualScriptBuiltinFunc::func_name[VisualScriptBuiltinFunc::FUNC_MAX] = {
	"sin",
	"cos",
	"tan",
	"sinh",
	"cosh",
	"tanh",
	"asin",
	"acos",
	"atan",
	"atan2",
	"sqrt",
	"fmod",
	"fposmod",
	"floor",
	"ceil",
	"round",
	"abs",
	"sign",
	"pow",
	"log",
	"exp",
	"is_nan",
	"is_inf",
	"ease",
	"step_decimals",
	"snapped",
	"lerp",
	"inverse_lerp",
	"range_lerp",
	"move_toward",
	"randomize",
	"randi",
	"randf",
	"randi_range",
	"randf_range",
	"randfn",
	"seed",
	"rand_seed",
	"deg2rad",
	"rad2deg",
	"linear2db",
	"db2linear",
	"wrapi",
	"wrapf",
	"pingpong",
	"max",
	"min",
	"clamp",
	"nearest_po2",
	"weakref",
	"convert",
	"typeof",
	"type_exists",
	"char",
	"str",
	"print",
	"printerr",
	"printraw",
	"print_verbose",
	"var2str",
	"str2var",
	"var2bytes",
	"bytes2var",
	"smoothstep",
	"posmod",
	"lerp_angle",
	"ord",
};

VisualScriptBuiltinFunc::BuiltinFunc VisualScriptBuiltinFunc::find_function(const String &p_string) {
	for (int i = 0; i < FUNC_MAX; i++) {
		if (p_string == func_name[i]) {
			return BuiltinFunc(i);
		}
	}

	return FUNC_MAX;
}

String VisualScriptBuiltinFunc::get_func_name(BuiltinFunc p_func) {
	ERR_FAIL_INDEX_V(p_func, FUNC_MAX, String());
	return func_name[p_func];
}

int VisualScriptBuiltinFunc::get_output_sequence_port_count() const {
	return has_input_sequence_port() ? 1 : 0;
}

bool VisualScriptBuiltinFunc::has_input_sequence_port() const {
	switch (func) {
		case MATH_RANDOMIZE:
		case TEXT_PRINT:
		case TEXT_PRINTERR:
		case TEXT_PRINTRAW:
		case TEXT_PRINT_VERBOSE:
		case MATH_SEED:
			return true;
		default:
			return false;
	}
}

int VisualScriptBuiltinFunc::get_func_argument_count(BuiltinFunc p_func) {
	switch (p_func) {
		case MATH_RANDOMIZE:
		case MATH_RANDI:
		case MATH_RANDF:
			return 0;
		case MATH_SIN:
		case MATH_COS:
		case MATH_TAN:
		case MATH_SINH:
		case MATH_COSH:
		case MATH_TANH:
		case MATH_ASIN:
		case MATH_ACOS:
		case MATH_ATAN:
		case MATH_SQRT:
		case MATH_FLOOR:
		case MATH_CEIL:
		case MATH_ROUND:
		case MATH_ABS:
		case MATH_SIGN:
		case MATH_LOG:
		case MATH_EXP:
		case MATH_ISNAN:
		case MATH_ISINF:
		case MATH_STEP_DECIMALS:
		case MATH_SEED:
		case MATH_RANDSEED:
		case MATH_DEG2RAD:
		case MATH_RAD2DEG:
		case MATH_LINEAR2DB:
		case MATH_DB2LINEAR:
		case LOGIC_NEAREST_PO2:
		case OBJ_WEAKREF:
		case TYPE_OF:
		case TEXT_CHAR:
		case TEXT_ORD:
		case TEXT_STR:
		case TEXT_PRINT:
		case TEXT_PRINTERR:
		case TEXT_PRINTRAW:
		case TEXT_PRINT_VERBOSE:
		case VAR_TO_STR:
		case STR_TO_VAR:
		case TYPE_EXISTS:
			return 1;
		case VAR_TO_BYTES:
		case BYTES_TO_VAR:
		case MATH_ATAN2:
		case MATH_FMOD:
		case MATH_FPOSMOD:
		case MATH_POSMOD:
		case MATH_PINGPONG:
		case MATH_POW:
		case MATH_EASE:
		case MATH_SNAPPED:
		case MATH_RANDI_RANGE:
		case MATH_RANDF_RANGE:
		case MATH_RANDFN:
		case LOGIC_MAX:
		case LOGIC_MIN:
		case TYPE_CONVERT:
			return 2;
		case MATH_LERP:
		case MATH_LERP_ANGLE:
		case MATH_INVERSE_LERP:
		case MATH_SMOOTHSTEP:
		case MATH_MOVE_TOWARD:
		case MATH_WRAP:
		case MATH_WRAPF:
		case LOGIC_CLAMP:
			return 3;
		case MATH_RANGE_LERP:
			return 5;
		case FUNC_MAX: {
		}
	}
	return 0;
}

int VisualScriptBuiltinFunc::get_input_value_port_count() const {
	return get_func_argument_count(func);
}

int VisualScriptBuiltinFunc::get_output_value_port_count() const {
	switch (func) {
		case MATH_RANDOMIZE:
		case TEXT_PRINT:
		case TEXT_PRINTERR:
		case TEXT_PRINTRAW:
		case TEXT_PRINT_VERBOSE:
		case MATH_SEED:
			return 0;
		case MATH_RANDSEED:
			return 2;
		default:
			return 1;
	}

	return 1;
}

String VisualScriptBuiltinFunc::get_output_sequence_port_text(int p_port) const {
	return String();
}

PropertyInfo VisualScriptBuiltinFunc::get_input_value_port_info(int p_idx) const {
	switch (func) {
		case MATH_SIN:
		case MATH_COS:
		case MATH_TAN:
		case MATH_SINH:
		case MATH_COSH:
		case MATH_TANH:
		case MATH_ASIN:
		case MATH_ACOS:
		case MATH_ATAN:
		case MATH_SQRT:
		case MATH_FLOOR:
		case MATH_CEIL:
		case MATH_ROUND:
		case MATH_ABS:
		case MATH_SIGN:
		case MATH_LOG:
		case MATH_EXP:
		case MATH_ISNAN:
		case MATH_ISINF: {
			return PropertyInfo(Variant::FLOAT, "s");
		} break;
		case MATH_ATAN2: {
			if (p_idx == 0) {
				return PropertyInfo(Variant::FLOAT, "y");
			} else {
				return PropertyInfo(Variant::FLOAT, "x");
			}
		} break;
		case MATH_FMOD:
		case MATH_FPOSMOD:
		case LOGIC_MAX:
		case LOGIC_MIN: {
			if (p_idx == 0) {
				return PropertyInfo(Variant::FLOAT, "a");
			} else {
				return PropertyInfo(Variant::FLOAT, "b");
			}
		} break;
		case MATH_POSMOD: {
			if (p_idx == 0) {
				return PropertyInfo(Variant::INT, "a");
			} else {
				return PropertyInfo(Variant::INT, "b");
			}
		} break;
		case MATH_POW: {
			if (p_idx == 0) {
				return PropertyInfo(Variant::FLOAT, "base");
			} else {
				return PropertyInfo(Variant::FLOAT, "exp");
			}
		} break;
		case MATH_EASE: {
			if (p_idx == 0) {
				return PropertyInfo(Variant::FLOAT, "s");
			} else {
				return PropertyInfo(Variant::FLOAT, "curve");
			}
		} break;
		case MATH_STEP_DECIMALS: {
			return PropertyInfo(Variant::FLOAT, "step");
		} break;
		case MATH_SNAPPED: {
			if (p_idx == 0) {
				return PropertyInfo(Variant::FLOAT, "s");
			} else {
				return PropertyInfo(Variant::FLOAT, "steps");
			}
		} break;
		case MATH_LERP:
		case MATH_LERP_ANGLE:
		case MATH_INVERSE_LERP:
		case MATH_SMOOTHSTEP: {
			if (p_idx == 0) {
				return PropertyInfo(Variant::FLOAT, "from");
			} else if (p_idx == 1) {
				return PropertyInfo(Variant::FLOAT, "to");
			} else {
				return PropertyInfo(Variant::FLOAT, "weight");
			}
		} break;
		case MATH_RANGE_LERP: {
			if (p_idx == 0) {
				return PropertyInfo(Variant::FLOAT, "value");
			} else if (p_idx == 1) {
				return PropertyInfo(Variant::FLOAT, "istart");
			} else if (p_idx == 2) {
				return PropertyInfo(Variant::FLOAT, "istop");
			} else if (p_idx == 3) {
				return PropertyInfo(Variant::FLOAT, "ostart");
			} else {
				return PropertyInfo(Variant::FLOAT, "ostop");
			}
		} break;
		case MATH_MOVE_TOWARD: {
			if (p_idx == 0) {
				return PropertyInfo(Variant::FLOAT, "from");
			} else if (p_idx == 1) {
				return PropertyInfo(Variant::FLOAT, "to");
			} else {
				return PropertyInfo(Variant::FLOAT, "delta");
			}
		} break;
		case MATH_RANDOMIZE:
		case MATH_RANDI:
		case MATH_RANDF: {
		} break;
		case MATH_RANDI_RANGE: {
			if (p_idx == 0) {
				return PropertyInfo(Variant::INT, "from");
			} else {
				return PropertyInfo(Variant::INT, "to");
			}
		} break;
		case MATH_RANDF_RANGE: {
			if (p_idx == 0) {
				return PropertyInfo(Variant::FLOAT, "from");
			} else {
				return PropertyInfo(Variant::FLOAT, "to");
			}
		} break;
		case MATH_RANDFN: {
			if (p_idx == 0) {
				return PropertyInfo(Variant::FLOAT, "mean");
			} else {
				return PropertyInfo(Variant::FLOAT, "deviation");
			}
		} break;
		case MATH_SEED:
		case MATH_RANDSEED: {
			return PropertyInfo(Variant::INT, "seed");
		} break;
		case MATH_DEG2RAD: {
			return PropertyInfo(Variant::FLOAT, "deg");
		} break;
		case MATH_RAD2DEG: {
			return PropertyInfo(Variant::FLOAT, "rad");
		} break;
		case MATH_LINEAR2DB: {
			return PropertyInfo(Variant::FLOAT, "nrg");
		} break;
		case MATH_DB2LINEAR: {
			return PropertyInfo(Variant::FLOAT, "db");
		} break;
		case MATH_PINGPONG: {
			if (p_idx == 0) {
				return PropertyInfo(Variant::FLOAT, "value");
			} else {
				return PropertyInfo(Variant::FLOAT, "length");
			}
		} break;
		case MATH_WRAP: {
			if (p_idx == 0) {
				return PropertyInfo(Variant::INT, "value");
			} else if (p_idx == 1) {
				return PropertyInfo(Variant::INT, "min");
			} else {
				return PropertyInfo(Variant::INT, "max");
			}
		} break;
		case MATH_WRAPF:
		case LOGIC_CLAMP: {
			if (p_idx == 0) {
				return PropertyInfo(Variant::FLOAT, "value");
			} else if (p_idx == 1) {
				return PropertyInfo(Variant::FLOAT, "min");
			} else {
				return PropertyInfo(Variant::FLOAT, "max");
			}
		} break;
		case LOGIC_NEAREST_PO2: {
			return PropertyInfo(Variant::INT, "value");
		} break;
		case OBJ_WEAKREF: {
			return PropertyInfo(Variant::OBJECT, "source");
		} break;
		case TYPE_CONVERT: {
			if (p_idx == 0) {
				return PropertyInfo(Variant::NIL, "what");
			} else {
				return PropertyInfo(Variant::STRING, "type");
			}
		} break;
		case TYPE_OF: {
			return PropertyInfo(Variant::NIL, "what");
		} break;
		case TYPE_EXISTS: {
			return PropertyInfo(Variant::STRING, "type");
		} break;
		case TEXT_ORD: {
			return PropertyInfo(Variant::STRING, "character");
		} break;
		case TEXT_CHAR: {
			return PropertyInfo(Variant::INT, "ascii");
		} break;
		case TEXT_STR:
		case TEXT_PRINT:
		case TEXT_PRINTERR:
		case TEXT_PRINTRAW:
		case TEXT_PRINT_VERBOSE: {
			return PropertyInfo(Variant::NIL, "value");
		} break;
		case STR_TO_VAR: {
			return PropertyInfo(Variant::STRING, "string");
		} break;
		case VAR_TO_STR:
		case VAR_TO_BYTES: {
			if (p_idx == 0) {
				return PropertyInfo(Variant::NIL, "var");
			} else {
				return PropertyInfo(Variant::BOOL, "full_objects");
			}

		} break;
		case BYTES_TO_VAR: {
			if (p_idx == 0) {
				return PropertyInfo(Variant::PACKED_BYTE_ARRAY, "bytes");
			} else {
				return PropertyInfo(Variant::BOOL, "allow_objects");
			}
		} break;
		case FUNC_MAX: {
		}
	}

	return PropertyInfo();
}

PropertyInfo VisualScriptBuiltinFunc::get_output_value_port_info(int p_idx) const {
	Variant::Type t = Variant::NIL;
	switch (func) {
		case MATH_SIN:
		case MATH_COS:
		case MATH_TAN:
		case MATH_SINH:
		case MATH_COSH:
		case MATH_TANH:
		case MATH_ASIN:
		case MATH_ACOS:
		case MATH_ATAN:
		case MATH_ATAN2:
		case MATH_SQRT:
		case MATH_FMOD:
		case MATH_FPOSMOD:
		case MATH_FLOOR:
		case MATH_CEIL: {
			t = Variant::FLOAT;
		} break;
		case MATH_POSMOD: {
			t = Variant::INT;
		} break;
		case MATH_ROUND: {
			t = Variant::FLOAT;
		} break;
		case MATH_ABS: {
			t = Variant::NIL;
		} break;
		case MATH_SIGN: {
			t = Variant::NIL;
		} break;
		case MATH_POW:
		case MATH_LOG:
		case MATH_EXP: {
			t = Variant::FLOAT;
		} break;
		case MATH_ISNAN:
		case MATH_ISINF: {
			t = Variant::BOOL;
		} break;
		case MATH_EASE: {
			t = Variant::FLOAT;
		} break;
		case MATH_STEP_DECIMALS: {
			t = Variant::INT;
		} break;
		case MATH_SNAPPED:
		case MATH_LERP:
		case MATH_LERP_ANGLE:
		case MATH_INVERSE_LERP:
		case MATH_RANGE_LERP:
		case MATH_SMOOTHSTEP:
		case MATH_MOVE_TOWARD:
		case MATH_RANDOMIZE: {
		} break;
		case MATH_RANDI: {
			t = Variant::INT;
		} break;
		case MATH_RANDF:
		case MATH_RANDFN:
		case MATH_RANDF_RANGE: {
			t = Variant::FLOAT;
		} break;
		case MATH_RANDI_RANGE: {
			t = Variant::INT;
		} break;
		case MATH_SEED: {
		} break;
		case MATH_RANDSEED: {
			if (p_idx == 0) {
				return PropertyInfo(Variant::INT, "rnd");
			} else {
				return PropertyInfo(Variant::INT, "seed");
			}
		} break;
		case MATH_DEG2RAD:
		case MATH_RAD2DEG:
		case MATH_LINEAR2DB:
		case MATH_WRAPF:
		case MATH_PINGPONG:
		case MATH_DB2LINEAR: {
			t = Variant::FLOAT;
		} break;
		case MATH_WRAP: {
			t = Variant::INT;
		} break;
		case LOGIC_MAX:
		case LOGIC_MIN:
		case LOGIC_CLAMP: {
		} break;

		case LOGIC_NEAREST_PO2: {
			t = Variant::NIL;
		} break;
		case OBJ_WEAKREF: {
			t = Variant::OBJECT;

		} break;
		case TYPE_CONVERT: {
		} break;
		case TEXT_ORD:
		case TYPE_OF: {
			t = Variant::INT;

		} break;
		case TYPE_EXISTS: {
			t = Variant::BOOL;

		} break;
		case TEXT_CHAR:
		case TEXT_STR: {
			t = Variant::STRING;

		} break;
		case TEXT_PRINT: {
		} break;
		case TEXT_PRINTERR: {
		} break;
		case TEXT_PRINTRAW: {
		} break;
		case TEXT_PRINT_VERBOSE: {
		} break;
		case VAR_TO_STR: {
			t = Variant::STRING;
		} break;
		case STR_TO_VAR: {
		} break;
		case VAR_TO_BYTES: {
			if (p_idx == 0) {
				t = Variant::PACKED_BYTE_ARRAY;
			} else {
				t = Variant::BOOL;
			}

		} break;
		case BYTES_TO_VAR: {
			if (p_idx == 1) {
				t = Variant::BOOL;
			}
		} break;
		case FUNC_MAX: {
		}
	}

	return PropertyInfo(t, "");
}

/*
String VisualScriptBuiltinFunc::get_caption() const {
	return "BuiltinFunc";
}

*/

String VisualScriptBuiltinFunc::get_caption() const {
	return func_name[func];
}

void VisualScriptBuiltinFunc::set_func(BuiltinFunc p_which) {
	ERR_FAIL_INDEX(p_which, FUNC_MAX);
	func = p_which;
	notify_property_list_changed();
	ports_changed_notify();
}

VisualScriptBuiltinFunc::BuiltinFunc VisualScriptBuiltinFunc::get_func() {
	return func;
}

#define VALIDATE_ARG_NUM(m_arg)                                           \
	if (!p_inputs[m_arg]->is_num()) {                                     \
		r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT; \
		r_error.argument = m_arg;                                         \
		r_error.expected = Variant::FLOAT;                                \
		return;                                                           \
	}

void VisualScriptBuiltinFunc::exec_func(BuiltinFunc p_func, const Variant **p_inputs, Variant *r_return, Callable::CallError &r_error, String &r_error_str) {
	switch (p_func) {
		case VisualScriptBuiltinFunc::MATH_SIN: {
			VALIDATE_ARG_NUM(0);
			*r_return = Math::sin((double)*p_inputs[0]);
		} break;
		case VisualScriptBuiltinFunc::MATH_COS: {
			VALIDATE_ARG_NUM(0);
			*r_return = Math::cos((double)*p_inputs[0]);
		} break;
		case VisualScriptBuiltinFunc::MATH_TAN: {
			VALIDATE_ARG_NUM(0);
			*r_return = Math::tan((double)*p_inputs[0]);
		} break;
		case VisualScriptBuiltinFunc::MATH_SINH: {
			VALIDATE_ARG_NUM(0);
			*r_return = Math::sinh((double)*p_inputs[0]);
		} break;
		case VisualScriptBuiltinFunc::MATH_COSH: {
			VALIDATE_ARG_NUM(0);
			*r_return = Math::cosh((double)*p_inputs[0]);
		} break;
		case VisualScriptBuiltinFunc::MATH_TANH: {
			VALIDATE_ARG_NUM(0);
			*r_return = Math::tanh((double)*p_inputs[0]);
		} break;
		case VisualScriptBuiltinFunc::MATH_ASIN: {
			VALIDATE_ARG_NUM(0);
			*r_return = Math::asin((double)*p_inputs[0]);
		} break;
		case VisualScriptBuiltinFunc::MATH_ACOS: {
			VALIDATE_ARG_NUM(0);
			*r_return = Math::acos((double)*p_inputs[0]);
		} break;
		case VisualScriptBuiltinFunc::MATH_ATAN: {
			VALIDATE_ARG_NUM(0);
			*r_return = Math::atan((double)*p_inputs[0]);
		} break;
		case VisualScriptBuiltinFunc::MATH_ATAN2: {
			VALIDATE_ARG_NUM(0);
			VALIDATE_ARG_NUM(1);
			*r_return = Math::atan2((double)*p_inputs[0], (double)*p_inputs[1]);
		} break;
		case VisualScriptBuiltinFunc::MATH_SQRT: {
			VALIDATE_ARG_NUM(0);
			*r_return = Math::sqrt((double)*p_inputs[0]);
		} break;
		case VisualScriptBuiltinFunc::MATH_FMOD: {
			VALIDATE_ARG_NUM(0);
			VALIDATE_ARG_NUM(1);
			*r_return = Math::fmod((double)*p_inputs[0], (double)*p_inputs[1]);
		} break;
		case VisualScriptBuiltinFunc::MATH_FPOSMOD: {
			VALIDATE_ARG_NUM(0);
			VALIDATE_ARG_NUM(1);
			*r_return = Math::fposmod((double)*p_inputs[0], (double)*p_inputs[1]);
		} break;
		case VisualScriptBuiltinFunc::MATH_POSMOD: {
			VALIDATE_ARG_NUM(0);
			VALIDATE_ARG_NUM(1);
			*r_return = Math::posmod((int64_t)*p_inputs[0], (int64_t)*p_inputs[1]);
		} break;
		case VisualScriptBuiltinFunc::MATH_FLOOR: {
			VALIDATE_ARG_NUM(0);
			*r_return = Math::floor((double)*p_inputs[0]);
		} break;
		case VisualScriptBuiltinFunc::MATH_CEIL: {
			VALIDATE_ARG_NUM(0);
			*r_return = Math::ceil((double)*p_inputs[0]);
		} break;
		case VisualScriptBuiltinFunc::MATH_ROUND: {
			VALIDATE_ARG_NUM(0);
			*r_return = Math::round((double)*p_inputs[0]);
		} break;
		case VisualScriptBuiltinFunc::MATH_ABS: {
			if (p_inputs[0]->get_type() == Variant::INT) {
				int64_t i = *p_inputs[0];
				*r_return = ABS(i);
			} else if (p_inputs[0]->get_type() == Variant::FLOAT) {
				real_t r = *p_inputs[0];
				*r_return = Math::abs(r);
			} else {
				r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
				r_error.argument = 0;
				r_error.expected = Variant::FLOAT;
			}
		} break;
		case VisualScriptBuiltinFunc::MATH_SIGN: {
			if (p_inputs[0]->get_type() == Variant::INT) {
				int64_t i = *p_inputs[0];
				*r_return = i < 0 ? -1 : (i > 0 ? +1 : 0);
			} else if (p_inputs[0]->get_type() == Variant::FLOAT) {
				real_t r = *p_inputs[0];
				*r_return = r < 0.0 ? -1.0 : (r > 0.0 ? +1.0 : 0.0);
			} else {
				r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
				r_error.argument = 0;
				r_error.expected = Variant::FLOAT;
			}
		} break;
		case VisualScriptBuiltinFunc::MATH_POW: {
			VALIDATE_ARG_NUM(0);
			VALIDATE_ARG_NUM(1);
			*r_return = Math::pow((double)*p_inputs[0], (double)*p_inputs[1]);
		} break;
		case VisualScriptBuiltinFunc::MATH_LOG: {
			VALIDATE_ARG_NUM(0);
			*r_return = Math::log((double)*p_inputs[0]);
		} break;
		case VisualScriptBuiltinFunc::MATH_EXP: {
			VALIDATE_ARG_NUM(0);
			*r_return = Math::exp((double)*p_inputs[0]);
		} break;
		case VisualScriptBuiltinFunc::MATH_ISNAN: {
			VALIDATE_ARG_NUM(0);
			*r_return = Math::is_nan((double)*p_inputs[0]);
		} break;
		case VisualScriptBuiltinFunc::MATH_ISINF: {
			VALIDATE_ARG_NUM(0);
			*r_return = Math::is_inf((double)*p_inputs[0]);
		} break;
		case VisualScriptBuiltinFunc::MATH_EASE: {
			VALIDATE_ARG_NUM(0);
			VALIDATE_ARG_NUM(1);
			*r_return = Math::ease((double)*p_inputs[0], (double)*p_inputs[1]);
		} break;
		case VisualScriptBuiltinFunc::MATH_STEP_DECIMALS: {
			VALIDATE_ARG_NUM(0);
			*r_return = Math::step_decimals((double)*p_inputs[0]);
		} break;
		case VisualScriptBuiltinFunc::MATH_SNAPPED: {
			VALIDATE_ARG_NUM(0);
			VALIDATE_ARG_NUM(1);
			*r_return = Math::snapped((double)*p_inputs[0], (double)*p_inputs[1]);
		} break;
		case VisualScriptBuiltinFunc::MATH_LERP: {
			VALIDATE_ARG_NUM(0);
			VALIDATE_ARG_NUM(1);
			VALIDATE_ARG_NUM(2);
			*r_return = Math::lerp((double)*p_inputs[0], (double)*p_inputs[1], (double)*p_inputs[2]);
		} break;
		case VisualScriptBuiltinFunc::MATH_LERP_ANGLE: {
			VALIDATE_ARG_NUM(0);
			VALIDATE_ARG_NUM(1);
			VALIDATE_ARG_NUM(2);
			*r_return = Math::lerp_angle((double)*p_inputs[0], (double)*p_inputs[1], (double)*p_inputs[2]);
		} break;
		case VisualScriptBuiltinFunc::MATH_INVERSE_LERP: {
			VALIDATE_ARG_NUM(0);
			VALIDATE_ARG_NUM(1);
			VALIDATE_ARG_NUM(2);
			*r_return = Math::inverse_lerp((double)*p_inputs[0], (double)*p_inputs[1], (double)*p_inputs[2]);
		} break;
		case VisualScriptBuiltinFunc::MATH_RANGE_LERP: {
			VALIDATE_ARG_NUM(0);
			VALIDATE_ARG_NUM(1);
			VALIDATE_ARG_NUM(2);
			VALIDATE_ARG_NUM(3);
			VALIDATE_ARG_NUM(4);
			*r_return = Math::range_lerp((double)*p_inputs[0], (double)*p_inputs[1], (double)*p_inputs[2], (double)*p_inputs[3], (double)*p_inputs[4]);
		} break;
		case VisualScriptBuiltinFunc::MATH_SMOOTHSTEP: {
			VALIDATE_ARG_NUM(0);
			VALIDATE_ARG_NUM(1);
			VALIDATE_ARG_NUM(2);
			*r_return = Math::smoothstep((double)*p_inputs[0], (double)*p_inputs[1], (double)*p_inputs[2]);
		} break;
		case VisualScriptBuiltinFunc::MATH_MOVE_TOWARD: {
			VALIDATE_ARG_NUM(0);
			VALIDATE_ARG_NUM(1);
			VALIDATE_ARG_NUM(2);
			*r_return = Math::move_toward((double)*p_inputs[0], (double)*p_inputs[1], (double)*p_inputs[2]);
		} break;
		case VisualScriptBuiltinFunc::MATH_RANDOMIZE: {
			Math::randomize();

		} break;
		case VisualScriptBuiltinFunc::MATH_RANDI: {
			*r_return = Math::rand();
		} break;
		case VisualScriptBuiltinFunc::MATH_RANDF: {
			*r_return = Math::randf();
		} break;
		case VisualScriptBuiltinFunc::MATH_RANDF_RANGE: {
			VALIDATE_ARG_NUM(0);
			VALIDATE_ARG_NUM(1);
			*r_return = Math::random((double)*p_inputs[0], (double)*p_inputs[1]);
		} break;
		case VisualScriptBuiltinFunc::MATH_RANDI_RANGE: {
			VALIDATE_ARG_NUM(0);
			VALIDATE_ARG_NUM(1);
			*r_return = Math::random((int)*p_inputs[0], (int)*p_inputs[1]);
		} break;
		case VisualScriptBuiltinFunc::MATH_SEED: {
			VALIDATE_ARG_NUM(0);
			uint64_t seed = *p_inputs[0];
			Math::seed(seed);

		} break;
		case VisualScriptBuiltinFunc::MATH_RANDSEED: {
			VALIDATE_ARG_NUM(0);
			uint64_t seed = *p_inputs[0];
			int ret = Math::rand_from_seed(&seed);
			Array reta;
			reta.push_back(ret);
			reta.push_back(seed);
			*r_return = reta;

		} break;
		case VisualScriptBuiltinFunc::MATH_DEG2RAD: {
			VALIDATE_ARG_NUM(0);
			*r_return = Math::deg2rad((double)*p_inputs[0]);
		} break;
		case VisualScriptBuiltinFunc::MATH_RAD2DEG: {
			VALIDATE_ARG_NUM(0);
			*r_return = Math::rad2deg((double)*p_inputs[0]);
		} break;
		case VisualScriptBuiltinFunc::MATH_LINEAR2DB: {
			VALIDATE_ARG_NUM(0);
			*r_return = Math::linear2db((double)*p_inputs[0]);
		} break;
		case VisualScriptBuiltinFunc::MATH_DB2LINEAR: {
			VALIDATE_ARG_NUM(0);
			*r_return = Math::db2linear((double)*p_inputs[0]);
		} break;
		case VisualScriptBuiltinFunc::MATH_PINGPONG: {
			VALIDATE_ARG_NUM(0);
			VALIDATE_ARG_NUM(1);
			*r_return = Math::pingpong((double)*p_inputs[0], (double)*p_inputs[1]);
		} break;
		case VisualScriptBuiltinFunc::MATH_WRAP: {
			VALIDATE_ARG_NUM(0);
			VALIDATE_ARG_NUM(1);
			VALIDATE_ARG_NUM(2);
			*r_return = Math::wrapi((int64_t)*p_inputs[0], (int64_t)*p_inputs[1], (int64_t)*p_inputs[2]);
		} break;
		case VisualScriptBuiltinFunc::MATH_WRAPF: {
			VALIDATE_ARG_NUM(0);
			VALIDATE_ARG_NUM(1);
			VALIDATE_ARG_NUM(2);
			*r_return = Math::wrapf((double)*p_inputs[0], (double)*p_inputs[1], (double)*p_inputs[2]);
		} break;
		case VisualScriptBuiltinFunc::LOGIC_MAX: {
			if (p_inputs[0]->get_type() == Variant::INT && p_inputs[1]->get_type() == Variant::INT) {
				int64_t a = *p_inputs[0];
				int64_t b = *p_inputs[1];
				*r_return = MAX(a, b);
			} else {
				VALIDATE_ARG_NUM(0);
				VALIDATE_ARG_NUM(1);

				real_t a = *p_inputs[0];
				real_t b = *p_inputs[1];

				*r_return = MAX(a, b);
			}

		} break;
		case VisualScriptBuiltinFunc::LOGIC_MIN: {
			if (p_inputs[0]->get_type() == Variant::INT && p_inputs[1]->get_type() == Variant::INT) {
				int64_t a = *p_inputs[0];
				int64_t b = *p_inputs[1];
				*r_return = MIN(a, b);
			} else {
				VALIDATE_ARG_NUM(0);
				VALIDATE_ARG_NUM(1);

				real_t a = *p_inputs[0];
				real_t b = *p_inputs[1];

				*r_return = MIN(a, b);
			}
		} break;
		case VisualScriptBuiltinFunc::LOGIC_CLAMP: {
			if (p_inputs[0]->get_type() == Variant::INT && p_inputs[1]->get_type() == Variant::INT && p_inputs[2]->get_type() == Variant::INT) {
				int64_t a = *p_inputs[0];
				int64_t b = *p_inputs[1];
				int64_t c = *p_inputs[2];
				*r_return = CLAMP(a, b, c);
			} else {
				VALIDATE_ARG_NUM(0);
				VALIDATE_ARG_NUM(1);
				VALIDATE_ARG_NUM(2);

				real_t a = *p_inputs[0];
				real_t b = *p_inputs[1];
				real_t c = *p_inputs[2];

				*r_return = CLAMP(a, b, c);
			}
		} break;
		case VisualScriptBuiltinFunc::LOGIC_NEAREST_PO2: {
			VALIDATE_ARG_NUM(0);
			int64_t num = *p_inputs[0];
			*r_return = next_power_of_2(num);
		} break;
		case VisualScriptBuiltinFunc::OBJ_WEAKREF: {
			if (p_inputs[0]->get_type() != Variant::OBJECT) {
				r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
				r_error.argument = 0;
				r_error.expected = Variant::OBJECT;

				return;
			}

			if (p_inputs[0]->is_ref()) {
				REF r = *p_inputs[0];
				if (!r.is_valid()) {
					return;
				}

				Ref<WeakRef> wref = memnew(WeakRef);
				wref->set_ref(r);
				*r_return = wref;
			} else {
				Object *obj = *p_inputs[0];
				if (!obj) {
					return;
				}
				Ref<WeakRef> wref = memnew(WeakRef);
				wref->set_obj(obj);
				*r_return = wref;
			}

		} break;
		case VisualScriptBuiltinFunc::TYPE_CONVERT: {
			VALIDATE_ARG_NUM(1);
			int type = *p_inputs[1];
			if (type < 0 || type >= Variant::VARIANT_MAX) {
				r_error_str = RTR("Invalid type argument to convert(), use TYPE_* constants.");
				r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
				r_error.argument = 0;
				r_error.expected = Variant::INT;
				return;

			} else {
				Variant::construct(Variant::Type(type), *r_return, p_inputs, 1, r_error);
			}
		} break;
		case VisualScriptBuiltinFunc::TYPE_OF: {
			*r_return = p_inputs[0]->get_type();

		} break;
		case VisualScriptBuiltinFunc::TYPE_EXISTS: {
			*r_return = ClassDB::class_exists(*p_inputs[0]);

		} break;
		case VisualScriptBuiltinFunc::TEXT_CHAR: {
			char32_t result[2] = { *p_inputs[0], 0 };

			*r_return = String(result);

		} break;
		case VisualScriptBuiltinFunc::TEXT_ORD: {
			if (p_inputs[0]->get_type() != Variant::STRING) {
				r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
				r_error.argument = 0;
				r_error.expected = Variant::STRING;

				return;
			}

			String str = p_inputs[0]->operator String();

			if (str.length() != 1) {
				r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
				r_error.argument = 0;
				r_error.expected = Variant::STRING;
				*r_return = "Expected a string of length 1 (a character).";

				return;
			}

			*r_return = str.get(0);

		} break;
		case VisualScriptBuiltinFunc::TEXT_STR: {
			String str = *p_inputs[0];

			*r_return = str;

		} break;
		case VisualScriptBuiltinFunc::TEXT_PRINT: {
			String str = *p_inputs[0];
			print_line(str);

		} break;

		case VisualScriptBuiltinFunc::TEXT_PRINTERR: {
			String str = *p_inputs[0];
			print_error(str);

		} break;
		case VisualScriptBuiltinFunc::TEXT_PRINTRAW: {
			String str = *p_inputs[0];
			OS::get_singleton()->print("%s", str.utf8().get_data());

		} break;
		case VisualScriptBuiltinFunc::TEXT_PRINT_VERBOSE: {
			String str = *p_inputs[0];
			print_verbose(str);
		} break;
		case VisualScriptBuiltinFunc::VAR_TO_STR: {
			String vars;
			VariantWriter::write_to_string(*p_inputs[0], vars);
			*r_return = vars;
		} break;
		case VisualScriptBuiltinFunc::STR_TO_VAR: {
			if (p_inputs[0]->get_type() != Variant::STRING) {
				r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
				r_error.argument = 0;
				r_error.expected = Variant::STRING;

				return;
			}

			VariantParser::StreamString ss;
			ss.s = *p_inputs[0];

			String errs;
			int line;
			Error err = VariantParser::parse(&ss, *r_return, errs, line);

			if (err != OK) {
				r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
				r_error.argument = 0;
				r_error.expected = Variant::STRING;
				*r_return = "Parse error at line " + itos(line) + ": " + errs;
				return;
			}

		} break;
		case VisualScriptBuiltinFunc::VAR_TO_BYTES: {
			if (p_inputs[1]->get_type() != Variant::BOOL) {
				r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
				r_error.argument = 1;
				r_error.expected = Variant::BOOL;
				return;
			}
			PackedByteArray barr;
			int len;
			bool full_objects = *p_inputs[1];
			Error err = encode_variant(*p_inputs[0], nullptr, len, full_objects);
			if (err) {
				r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
				r_error.argument = 0;
				r_error.expected = Variant::NIL;
				r_error_str = "Unexpected error encoding variable to bytes, likely unserializable type found (Object or RID).";
				return;
			}

			barr.resize(len);
			{
				uint8_t *w = barr.ptrw();
				encode_variant(*p_inputs[0], w, len, full_objects);
			}
			*r_return = barr;
		} break;
		case VisualScriptBuiltinFunc::BYTES_TO_VAR: {
			if (p_inputs[0]->get_type() != Variant::PACKED_BYTE_ARRAY) {
				r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
				r_error.argument = 0;
				r_error.expected = Variant::PACKED_BYTE_ARRAY;
				return;
			}
			if (p_inputs[1]->get_type() != Variant::BOOL) {
				r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
				r_error.argument = 1;
				r_error.expected = Variant::BOOL;
				return;
			}

			PackedByteArray varr = *p_inputs[0];
			bool allow_objects = *p_inputs[1];
			Variant ret;
			{
				const uint8_t *r = varr.ptr();
				Error err = decode_variant(ret, r, varr.size(), nullptr, allow_objects);
				if (err != OK) {
					r_error_str = RTR("Not enough bytes for decoding bytes, or invalid format.");
					r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
					r_error.argument = 0;
					r_error.expected = Variant::PACKED_BYTE_ARRAY;
					return;
				}
			}

			*r_return = ret;

		} break;
		default: {
		}
	}
}

class VisualScriptNodeInstanceBuiltinFunc : public VisualScriptNodeInstance {
public:
	VisualScriptBuiltinFunc *node;
	VisualScriptInstance *instance;

	VisualScriptBuiltinFunc::BuiltinFunc func;

	//virtual int get_working_memory_size() const { return 0; }
	//virtual bool is_output_port_unsequenced(int p_idx) const { return false; }
	//virtual bool get_output_port_unsequenced(int p_idx,Variant* r_value,Variant* p_working_mem,String &r_error) const { return true; }

	virtual int step(const Variant **p_inputs, Variant **p_outputs, StartMode p_start_mode, Variant *p_working_mem, Callable::CallError &r_error, String &r_error_str) {
		VisualScriptBuiltinFunc::exec_func(func, p_inputs, p_outputs[0], r_error, r_error_str);
		return 0;
	}
};

VisualScriptNodeInstance *VisualScriptBuiltinFunc::instantiate(VisualScriptInstance *p_instance) {
	VisualScriptNodeInstanceBuiltinFunc *instance = memnew(VisualScriptNodeInstanceBuiltinFunc);
	instance->node = this;
	instance->instance = p_instance;
	instance->func = func;
	return instance;
}

void VisualScriptBuiltinFunc::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_func", "which"), &VisualScriptBuiltinFunc::set_func);
	ClassDB::bind_method(D_METHOD("get_func"), &VisualScriptBuiltinFunc::get_func);

	String cc;

	for (int i = 0; i < FUNC_MAX; i++) {
		if (i > 0) {
			cc += ",";
		}
		cc += func_name[i];
	}
	ADD_PROPERTY(PropertyInfo(Variant::INT, "function", PROPERTY_HINT_ENUM, cc), "set_func", "get_func");

	BIND_ENUM_CONSTANT(MATH_SIN);
	BIND_ENUM_CONSTANT(MATH_COS);
	BIND_ENUM_CONSTANT(MATH_TAN);
	BIND_ENUM_CONSTANT(MATH_SINH);
	BIND_ENUM_CONSTANT(MATH_COSH);
	BIND_ENUM_CONSTANT(MATH_TANH);
	BIND_ENUM_CONSTANT(MATH_ASIN);
	BIND_ENUM_CONSTANT(MATH_ACOS);
	BIND_ENUM_CONSTANT(MATH_ATAN);
	BIND_ENUM_CONSTANT(MATH_ATAN2);
	BIND_ENUM_CONSTANT(MATH_SQRT);
	BIND_ENUM_CONSTANT(MATH_FMOD);
	BIND_ENUM_CONSTANT(MATH_FPOSMOD);
	BIND_ENUM_CONSTANT(MATH_FLOOR);
	BIND_ENUM_CONSTANT(MATH_CEIL);
	BIND_ENUM_CONSTANT(MATH_ROUND);
	BIND_ENUM_CONSTANT(MATH_ABS);
	BIND_ENUM_CONSTANT(MATH_SIGN);
	BIND_ENUM_CONSTANT(MATH_POW);
	BIND_ENUM_CONSTANT(MATH_LOG);
	BIND_ENUM_CONSTANT(MATH_EXP);
	BIND_ENUM_CONSTANT(MATH_ISNAN);
	BIND_ENUM_CONSTANT(MATH_ISINF);
	BIND_ENUM_CONSTANT(MATH_EASE);
	BIND_ENUM_CONSTANT(MATH_STEP_DECIMALS);
	BIND_ENUM_CONSTANT(MATH_SNAPPED);
	BIND_ENUM_CONSTANT(MATH_LERP);
	BIND_ENUM_CONSTANT(MATH_INVERSE_LERP);
	BIND_ENUM_CONSTANT(MATH_RANGE_LERP);
	BIND_ENUM_CONSTANT(MATH_MOVE_TOWARD);
	BIND_ENUM_CONSTANT(MATH_RANDOMIZE);
	BIND_ENUM_CONSTANT(MATH_RANDI);
	BIND_ENUM_CONSTANT(MATH_RANDF);
	BIND_ENUM_CONSTANT(MATH_RANDI_RANGE);
	BIND_ENUM_CONSTANT(MATH_RANDF_RANGE);
	BIND_ENUM_CONSTANT(MATH_RANDFN);
	BIND_ENUM_CONSTANT(MATH_SEED);
	BIND_ENUM_CONSTANT(MATH_RANDSEED);
	BIND_ENUM_CONSTANT(MATH_DEG2RAD);
	BIND_ENUM_CONSTANT(MATH_RAD2DEG);
	BIND_ENUM_CONSTANT(MATH_LINEAR2DB);
	BIND_ENUM_CONSTANT(MATH_DB2LINEAR);
	BIND_ENUM_CONSTANT(MATH_WRAP);
	BIND_ENUM_CONSTANT(MATH_WRAPF);
	BIND_ENUM_CONSTANT(MATH_PINGPONG);
	BIND_ENUM_CONSTANT(LOGIC_MAX);
	BIND_ENUM_CONSTANT(LOGIC_MIN);
	BIND_ENUM_CONSTANT(LOGIC_CLAMP);
	BIND_ENUM_CONSTANT(LOGIC_NEAREST_PO2);
	BIND_ENUM_CONSTANT(OBJ_WEAKREF);
	BIND_ENUM_CONSTANT(TYPE_CONVERT);
	BIND_ENUM_CONSTANT(TYPE_OF);
	BIND_ENUM_CONSTANT(TYPE_EXISTS);
	BIND_ENUM_CONSTANT(TEXT_CHAR);
	BIND_ENUM_CONSTANT(TEXT_STR);
	BIND_ENUM_CONSTANT(TEXT_PRINT);
	BIND_ENUM_CONSTANT(TEXT_PRINTERR);
	BIND_ENUM_CONSTANT(TEXT_PRINTRAW);
	BIND_ENUM_CONSTANT(TEXT_PRINT_VERBOSE);
	BIND_ENUM_CONSTANT(VAR_TO_STR);
	BIND_ENUM_CONSTANT(STR_TO_VAR);
	BIND_ENUM_CONSTANT(VAR_TO_BYTES);
	BIND_ENUM_CONSTANT(BYTES_TO_VAR);
	BIND_ENUM_CONSTANT(MATH_SMOOTHSTEP);
	BIND_ENUM_CONSTANT(MATH_POSMOD);
	BIND_ENUM_CONSTANT(MATH_LERP_ANGLE);
	BIND_ENUM_CONSTANT(TEXT_ORD);
	BIND_ENUM_CONSTANT(FUNC_MAX);
}

VisualScriptBuiltinFunc::VisualScriptBuiltinFunc(VisualScriptBuiltinFunc::BuiltinFunc func) {
	this->func = func;
}

VisualScriptBuiltinFunc::VisualScriptBuiltinFunc() {
	func = MATH_SIN;
}

template <VisualScriptBuiltinFunc::BuiltinFunc func>
static Ref<VisualScriptNode> create_builtin_func_node(const String &p_name) {
	Ref<VisualScriptBuiltinFunc> node = memnew(VisualScriptBuiltinFunc(func));
	return node;
}

void register_visual_script_builtin_func_node() {
	VisualScriptLanguage::singleton->add_register_func("functions/built_in/sin", create_builtin_func_node<VisualScriptBuiltinFunc::MATH_SIN>);
	VisualScriptLanguage::singleton->add_register_func("functions/built_in/cos", create_builtin_func_node<VisualScriptBuiltinFunc::MATH_COS>);
	VisualScriptLanguage::singleton->add_register_func("functions/built_in/tan", create_builtin_func_node<VisualScriptBuiltinFunc::MATH_TAN>);
	VisualScriptLanguage::singleton->add_register_func("functions/built_in/sinh", create_builtin_func_node<VisualScriptBuiltinFunc::MATH_SINH>);
	VisualScriptLanguage::singleton->add_register_func("functions/built_in/cosh", create_builtin_func_node<VisualScriptBuiltinFunc::MATH_COSH>);
	VisualScriptLanguage::singleton->add_register_func("functions/built_in/tanh", create_builtin_func_node<VisualScriptBuiltinFunc::MATH_TANH>);

	VisualScriptLanguage::singleton->add_register_func("functions/built_in/asin", create_builtin_func_node<VisualScriptBuiltinFunc::MATH_ASIN>);
	VisualScriptLanguage::singleton->add_register_func("functions/built_in/acos", create_builtin_func_node<VisualScriptBuiltinFunc::MATH_ACOS>);
	VisualScriptLanguage::singleton->add_register_func("functions/built_in/atan", create_builtin_func_node<VisualScriptBuiltinFunc::MATH_ATAN>);
	VisualScriptLanguage::singleton->add_register_func("functions/built_in/atan2", create_builtin_func_node<VisualScriptBuiltinFunc::MATH_ATAN2>);

	VisualScriptLanguage::singleton->add_register_func("functions/built_in/sqrt", create_builtin_func_node<VisualScriptBuiltinFunc::MATH_SQRT>);
	VisualScriptLanguage::singleton->add_register_func("functions/built_in/fmod", create_builtin_func_node<VisualScriptBuiltinFunc::MATH_FMOD>);
	VisualScriptLanguage::singleton->add_register_func("functions/built_in/fposmod", create_builtin_func_node<VisualScriptBuiltinFunc::MATH_FPOSMOD>);
	VisualScriptLanguage::singleton->add_register_func("functions/built_in/posmod", create_builtin_func_node<VisualScriptBuiltinFunc::MATH_POSMOD>);
	VisualScriptLanguage::singleton->add_register_func("functions/built_in/floor", create_builtin_func_node<VisualScriptBuiltinFunc::MATH_FLOOR>);
	VisualScriptLanguage::singleton->add_register_func("functions/built_in/ceil", create_builtin_func_node<VisualScriptBuiltinFunc::MATH_CEIL>);
	VisualScriptLanguage::singleton->add_register_func("functions/built_in/round", create_builtin_func_node<VisualScriptBuiltinFunc::MATH_ROUND>);
	VisualScriptLanguage::singleton->add_register_func("functions/built_in/abs", create_builtin_func_node<VisualScriptBuiltinFunc::MATH_ABS>);
	VisualScriptLanguage::singleton->add_register_func("functions/built_in/sign", create_builtin_func_node<VisualScriptBuiltinFunc::MATH_SIGN>);
	VisualScriptLanguage::singleton->add_register_func("functions/built_in/pow", create_builtin_func_node<VisualScriptBuiltinFunc::MATH_POW>);
	VisualScriptLanguage::singleton->add_register_func("functions/built_in/log", create_builtin_func_node<VisualScriptBuiltinFunc::MATH_LOG>);
	VisualScriptLanguage::singleton->add_register_func("functions/built_in/exp", create_builtin_func_node<VisualScriptBuiltinFunc::MATH_EXP>);
	VisualScriptLanguage::singleton->add_register_func("functions/built_in/isnan", create_builtin_func_node<VisualScriptBuiltinFunc::MATH_ISNAN>);
	VisualScriptLanguage::singleton->add_register_func("functions/built_in/isinf", create_builtin_func_node<VisualScriptBuiltinFunc::MATH_ISINF>);

	VisualScriptLanguage::singleton->add_register_func("functions/built_in/ease", create_builtin_func_node<VisualScriptBuiltinFunc::MATH_EASE>);
	VisualScriptLanguage::singleton->add_register_func("functions/built_in/step_decimals", create_builtin_func_node<VisualScriptBuiltinFunc::MATH_STEP_DECIMALS>);
	VisualScriptLanguage::singleton->add_register_func("functions/built_in/snapped", create_builtin_func_node<VisualScriptBuiltinFunc::MATH_SNAPPED>);
	VisualScriptLanguage::singleton->add_register_func("functions/built_in/lerp", create_builtin_func_node<VisualScriptBuiltinFunc::MATH_LERP>);
	VisualScriptLanguage::singleton->add_register_func("functions/built_in/lerp_angle", create_builtin_func_node<VisualScriptBuiltinFunc::MATH_LERP_ANGLE>);
	VisualScriptLanguage::singleton->add_register_func("functions/built_in/inverse_lerp", create_builtin_func_node<VisualScriptBuiltinFunc::MATH_INVERSE_LERP>);
	VisualScriptLanguage::singleton->add_register_func("functions/built_in/range_lerp", create_builtin_func_node<VisualScriptBuiltinFunc::MATH_RANGE_LERP>);
	VisualScriptLanguage::singleton->add_register_func("functions/built_in/smoothstep", create_builtin_func_node<VisualScriptBuiltinFunc::MATH_SMOOTHSTEP>);
	VisualScriptLanguage::singleton->add_register_func("functions/built_in/move_toward", create_builtin_func_node<VisualScriptBuiltinFunc::MATH_MOVE_TOWARD>);
	VisualScriptLanguage::singleton->add_register_func("functions/built_in/randomize", create_builtin_func_node<VisualScriptBuiltinFunc::MATH_RANDOMIZE>);
	VisualScriptLanguage::singleton->add_register_func("functions/built_in/randi", create_builtin_func_node<VisualScriptBuiltinFunc::MATH_RANDI>);
	VisualScriptLanguage::singleton->add_register_func("functions/built_in/randf", create_builtin_func_node<VisualScriptBuiltinFunc::MATH_RANDF>);
	VisualScriptLanguage::singleton->add_register_func("functions/built_in/randi_range", create_builtin_func_node<VisualScriptBuiltinFunc::MATH_RANDI_RANGE>);
	VisualScriptLanguage::singleton->add_register_func("functions/built_in/randf_range", create_builtin_func_node<VisualScriptBuiltinFunc::MATH_RANDF_RANGE>);
	VisualScriptLanguage::singleton->add_register_func("functions/built_in/randfn", create_builtin_func_node<VisualScriptBuiltinFunc::MATH_RANDFN>);

	VisualScriptLanguage::singleton->add_register_func("functions/built_in/seed", create_builtin_func_node<VisualScriptBuiltinFunc::MATH_SEED>);
	VisualScriptLanguage::singleton->add_register_func("functions/built_in/randseed", create_builtin_func_node<VisualScriptBuiltinFunc::MATH_RANDSEED>);
	VisualScriptLanguage::singleton->add_register_func("functions/built_in/deg2rad", create_builtin_func_node<VisualScriptBuiltinFunc::MATH_DEG2RAD>);
	VisualScriptLanguage::singleton->add_register_func("functions/built_in/rad2deg", create_builtin_func_node<VisualScriptBuiltinFunc::MATH_RAD2DEG>);
	VisualScriptLanguage::singleton->add_register_func("functions/built_in/linear2db", create_builtin_func_node<VisualScriptBuiltinFunc::MATH_LINEAR2DB>);
	VisualScriptLanguage::singleton->add_register_func("functions/built_in/db2linear", create_builtin_func_node<VisualScriptBuiltinFunc::MATH_DB2LINEAR>);
	VisualScriptLanguage::singleton->add_register_func("functions/built_in/wrapi", create_builtin_func_node<VisualScriptBuiltinFunc::MATH_WRAP>);
	VisualScriptLanguage::singleton->add_register_func("functions/built_in/wrapf", create_builtin_func_node<VisualScriptBuiltinFunc::MATH_WRAPF>);
	VisualScriptLanguage::singleton->add_register_func("functions/built_in/pingpong", create_builtin_func_node<VisualScriptBuiltinFunc::MATH_PINGPONG>);

	VisualScriptLanguage::singleton->add_register_func("functions/built_in/max", create_builtin_func_node<VisualScriptBuiltinFunc::LOGIC_MAX>);
	VisualScriptLanguage::singleton->add_register_func("functions/built_in/min", create_builtin_func_node<VisualScriptBuiltinFunc::LOGIC_MIN>);
	VisualScriptLanguage::singleton->add_register_func("functions/built_in/clamp", create_builtin_func_node<VisualScriptBuiltinFunc::LOGIC_CLAMP>);
	VisualScriptLanguage::singleton->add_register_func("functions/built_in/nearest_po2", create_builtin_func_node<VisualScriptBuiltinFunc::LOGIC_NEAREST_PO2>);

	VisualScriptLanguage::singleton->add_register_func("functions/built_in/weakref", create_builtin_func_node<VisualScriptBuiltinFunc::OBJ_WEAKREF>);
	VisualScriptLanguage::singleton->add_register_func("functions/built_in/convert", create_builtin_func_node<VisualScriptBuiltinFunc::TYPE_CONVERT>);
	VisualScriptLanguage::singleton->add_register_func("functions/built_in/typeof", create_builtin_func_node<VisualScriptBuiltinFunc::TYPE_OF>);
	VisualScriptLanguage::singleton->add_register_func("functions/built_in/type_exists", create_builtin_func_node<VisualScriptBuiltinFunc::TYPE_EXISTS>);
	VisualScriptLanguage::singleton->add_register_func("functions/built_in/char", create_builtin_func_node<VisualScriptBuiltinFunc::TEXT_CHAR>);
	VisualScriptLanguage::singleton->add_register_func("functions/built_in/ord", create_builtin_func_node<VisualScriptBuiltinFunc::TEXT_ORD>);
	VisualScriptLanguage::singleton->add_register_func("functions/built_in/str", create_builtin_func_node<VisualScriptBuiltinFunc::TEXT_STR>);
	VisualScriptLanguage::singleton->add_register_func("functions/built_in/print", create_builtin_func_node<VisualScriptBuiltinFunc::TEXT_PRINT>);
	VisualScriptLanguage::singleton->add_register_func("functions/built_in/printerr", create_builtin_func_node<VisualScriptBuiltinFunc::TEXT_PRINTERR>);
	VisualScriptLanguage::singleton->add_register_func("functions/built_in/printraw", create_builtin_func_node<VisualScriptBuiltinFunc::TEXT_PRINTRAW>);
	VisualScriptLanguage::singleton->add_register_func("functions/built_in/print_verbose", create_builtin_func_node<VisualScriptBuiltinFunc::TEXT_PRINT_VERBOSE>);
	VisualScriptLanguage::singleton->add_register_func("functions/built_in/var2str", create_builtin_func_node<VisualScriptBuiltinFunc::VAR_TO_STR>);
	VisualScriptLanguage::singleton->add_register_func("functions/built_in/str2var", create_builtin_func_node<VisualScriptBuiltinFunc::STR_TO_VAR>);
	VisualScriptLanguage::singleton->add_register_func("functions/built_in/var2bytes", create_builtin_func_node<VisualScriptBuiltinFunc::VAR_TO_BYTES>);
	VisualScriptLanguage::singleton->add_register_func("functions/built_in/bytes2var", create_builtin_func_node<VisualScriptBuiltinFunc::BYTES_TO_VAR>);
}
