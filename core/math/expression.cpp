/*************************************************************************/
/*  expression.cpp                                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "expression.h"

#include "core/class_db.h"
#include "core/func_ref.h"
#include "core/io/marshalls.h"
#include "core/math/math_funcs.h"
#include "core/os/os.h"
#include "core/reference.h"
#include "core/variant_parser.h"

const char *Expression::func_name[Expression::FUNC_MAX] = {
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
	"posmod",
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
	"stepify",
	"lerp",
	"lerp_angle",
	"inverse_lerp",
	"range_lerp",
	"smoothstep",
	"move_toward",
	"dectime",
	"randomize",
	"randi",
	"randf",
	"rand_range",
	"seed",
	"rand_seed",
	"deg2rad",
	"rad2deg",
	"linear2db",
	"db2linear",
	"polar2cartesian",
	"cartesian2polar",
	"wrapi",
	"wrapf",
	"max",
	"min",
	"clamp",
	"nearest_po2",
	"weakref",
	"funcref",
	"convert",
	"typeof",
	"type_exists",
	"char",
	"ord",
	"str",
	"print",
	"printerr",
	"printraw",
	"var2str",
	"str2var",
	"var2bytes",
	"bytes2var",
	"color_named",
};

Expression::BuiltinFunc Expression::find_function(const String &p_string) {
	for (int i = 0; i < FUNC_MAX; i++) {
		if (p_string == func_name[i]) {
			return BuiltinFunc(i);
		}
	}

	return FUNC_MAX;
}

String Expression::get_func_name(BuiltinFunc p_func) {
	ERR_FAIL_INDEX_V(p_func, FUNC_MAX, String());
	return func_name[p_func];
}

int Expression::get_func_argument_count(BuiltinFunc p_func) {
	switch (p_func) {
		case MATH_RANDOMIZE:
		case MATH_RAND:
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
		case MATH_POW:
		case MATH_EASE:
		case MATH_STEPIFY:
		case MATH_RANDOM:
		case MATH_POLAR2CARTESIAN:
		case MATH_CARTESIAN2POLAR:
		case LOGIC_MAX:
		case LOGIC_MIN:
		case FUNC_FUNCREF:
		case TYPE_CONVERT:
		case COLORN:
			return 2;
		case MATH_LERP:
		case MATH_LERP_ANGLE:
		case MATH_INVERSE_LERP:
		case MATH_SMOOTHSTEP:
		case MATH_MOVE_TOWARD:
		case MATH_DECTIME:
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

#define VALIDATE_ARG_NUM(m_arg)                                           \
	if (!p_inputs[m_arg]->is_num()) {                                     \
		r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT; \
		r_error.argument = m_arg;                                         \
		r_error.expected = Variant::FLOAT;                                \
		return;                                                           \
	}

void Expression::exec_func(BuiltinFunc p_func, const Variant **p_inputs, Variant *r_return, Callable::CallError &r_error, String &r_error_str) {
	r_error.error = Callable::CallError::CALL_OK;
	switch (p_func) {
		case MATH_SIN: {
			VALIDATE_ARG_NUM(0);
			*r_return = Math::sin((double)*p_inputs[0]);
		} break;
		case MATH_COS: {
			VALIDATE_ARG_NUM(0);
			*r_return = Math::cos((double)*p_inputs[0]);
		} break;
		case MATH_TAN: {
			VALIDATE_ARG_NUM(0);
			*r_return = Math::tan((double)*p_inputs[0]);
		} break;
		case MATH_SINH: {
			VALIDATE_ARG_NUM(0);
			*r_return = Math::sinh((double)*p_inputs[0]);
		} break;
		case MATH_COSH: {
			VALIDATE_ARG_NUM(0);
			*r_return = Math::cosh((double)*p_inputs[0]);
		} break;
		case MATH_TANH: {
			VALIDATE_ARG_NUM(0);
			*r_return = Math::tanh((double)*p_inputs[0]);
		} break;
		case MATH_ASIN: {
			VALIDATE_ARG_NUM(0);
			*r_return = Math::asin((double)*p_inputs[0]);
		} break;
		case MATH_ACOS: {
			VALIDATE_ARG_NUM(0);
			*r_return = Math::acos((double)*p_inputs[0]);
		} break;
		case MATH_ATAN: {
			VALIDATE_ARG_NUM(0);
			*r_return = Math::atan((double)*p_inputs[0]);
		} break;
		case MATH_ATAN2: {
			VALIDATE_ARG_NUM(0);
			VALIDATE_ARG_NUM(1);
			*r_return = Math::atan2((double)*p_inputs[0], (double)*p_inputs[1]);
		} break;
		case MATH_SQRT: {
			VALIDATE_ARG_NUM(0);
			*r_return = Math::sqrt((double)*p_inputs[0]);
		} break;
		case MATH_FMOD: {
			VALIDATE_ARG_NUM(0);
			VALIDATE_ARG_NUM(1);
			*r_return = Math::fmod((double)*p_inputs[0], (double)*p_inputs[1]);
		} break;
		case MATH_FPOSMOD: {
			VALIDATE_ARG_NUM(0);
			VALIDATE_ARG_NUM(1);
			*r_return = Math::fposmod((double)*p_inputs[0], (double)*p_inputs[1]);
		} break;
		case MATH_POSMOD: {
			VALIDATE_ARG_NUM(0);
			VALIDATE_ARG_NUM(1);
			*r_return = Math::posmod((int)*p_inputs[0], (int)*p_inputs[1]);
		} break;
		case MATH_FLOOR: {
			VALIDATE_ARG_NUM(0);
			*r_return = Math::floor((double)*p_inputs[0]);
		} break;
		case MATH_CEIL: {
			VALIDATE_ARG_NUM(0);
			*r_return = Math::ceil((double)*p_inputs[0]);
		} break;
		case MATH_ROUND: {
			VALIDATE_ARG_NUM(0);
			*r_return = Math::round((double)*p_inputs[0]);
		} break;
		case MATH_ABS: {
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
		case MATH_SIGN: {
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
		case MATH_POW: {
			VALIDATE_ARG_NUM(0);
			VALIDATE_ARG_NUM(1);
			*r_return = Math::pow((double)*p_inputs[0], (double)*p_inputs[1]);
		} break;
		case MATH_LOG: {
			VALIDATE_ARG_NUM(0);
			*r_return = Math::log((double)*p_inputs[0]);
		} break;
		case MATH_EXP: {
			VALIDATE_ARG_NUM(0);
			*r_return = Math::exp((double)*p_inputs[0]);
		} break;
		case MATH_ISNAN: {
			VALIDATE_ARG_NUM(0);
			*r_return = Math::is_nan((double)*p_inputs[0]);
		} break;
		case MATH_ISINF: {
			VALIDATE_ARG_NUM(0);
			*r_return = Math::is_inf((double)*p_inputs[0]);
		} break;
		case MATH_EASE: {
			VALIDATE_ARG_NUM(0);
			VALIDATE_ARG_NUM(1);
			*r_return = Math::ease((double)*p_inputs[0], (double)*p_inputs[1]);
		} break;
		case MATH_STEP_DECIMALS: {
			VALIDATE_ARG_NUM(0);
			*r_return = Math::step_decimals((double)*p_inputs[0]);
		} break;
		case MATH_STEPIFY: {
			VALIDATE_ARG_NUM(0);
			VALIDATE_ARG_NUM(1);
			*r_return = Math::stepify((double)*p_inputs[0], (double)*p_inputs[1]);
		} break;
		case MATH_LERP: {
			VALIDATE_ARG_NUM(0);
			VALIDATE_ARG_NUM(1);
			VALIDATE_ARG_NUM(2);
			*r_return = Math::lerp((double)*p_inputs[0], (double)*p_inputs[1], (double)*p_inputs[2]);
		} break;
		case MATH_LERP_ANGLE: {
			VALIDATE_ARG_NUM(0);
			VALIDATE_ARG_NUM(1);
			VALIDATE_ARG_NUM(2);
			*r_return = Math::lerp_angle((double)*p_inputs[0], (double)*p_inputs[1], (double)*p_inputs[2]);
		} break;
		case MATH_INVERSE_LERP: {
			VALIDATE_ARG_NUM(0);
			VALIDATE_ARG_NUM(1);
			VALIDATE_ARG_NUM(2);
			*r_return = Math::inverse_lerp((double)*p_inputs[0], (double)*p_inputs[1], (double)*p_inputs[2]);
		} break;
		case MATH_RANGE_LERP: {
			VALIDATE_ARG_NUM(0);
			VALIDATE_ARG_NUM(1);
			VALIDATE_ARG_NUM(2);
			VALIDATE_ARG_NUM(3);
			VALIDATE_ARG_NUM(4);
			*r_return = Math::range_lerp((double)*p_inputs[0], (double)*p_inputs[1], (double)*p_inputs[2], (double)*p_inputs[3], (double)*p_inputs[4]);
		} break;
		case MATH_SMOOTHSTEP: {
			VALIDATE_ARG_NUM(0);
			VALIDATE_ARG_NUM(1);
			VALIDATE_ARG_NUM(2);
			*r_return = Math::smoothstep((double)*p_inputs[0], (double)*p_inputs[1], (double)*p_inputs[2]);
		} break;
		case MATH_MOVE_TOWARD: {
			VALIDATE_ARG_NUM(0);
			VALIDATE_ARG_NUM(1);
			VALIDATE_ARG_NUM(2);
			*r_return = Math::move_toward((double)*p_inputs[0], (double)*p_inputs[1], (double)*p_inputs[2]);
		} break;
		case MATH_DECTIME: {
			VALIDATE_ARG_NUM(0);
			VALIDATE_ARG_NUM(1);
			VALIDATE_ARG_NUM(2);
			*r_return = Math::dectime((double)*p_inputs[0], (double)*p_inputs[1], (double)*p_inputs[2]);
		} break;
		case MATH_RANDOMIZE: {
			Math::randomize();

		} break;
		case MATH_RAND: {
			*r_return = Math::rand();
		} break;
		case MATH_RANDF: {
			*r_return = Math::randf();
		} break;
		case MATH_RANDOM: {
			VALIDATE_ARG_NUM(0);
			VALIDATE_ARG_NUM(1);
			*r_return = Math::random((double)*p_inputs[0], (double)*p_inputs[1]);
		} break;
		case MATH_SEED: {
			VALIDATE_ARG_NUM(0);
			uint64_t seed = *p_inputs[0];
			Math::seed(seed);

		} break;
		case MATH_RANDSEED: {
			VALIDATE_ARG_NUM(0);
			uint64_t seed = *p_inputs[0];
			int ret = Math::rand_from_seed(&seed);
			Array reta;
			reta.push_back(ret);
			reta.push_back(seed);
			*r_return = reta;

		} break;
		case MATH_DEG2RAD: {
			VALIDATE_ARG_NUM(0);
			*r_return = Math::deg2rad((double)*p_inputs[0]);
		} break;
		case MATH_RAD2DEG: {
			VALIDATE_ARG_NUM(0);
			*r_return = Math::rad2deg((double)*p_inputs[0]);
		} break;
		case MATH_LINEAR2DB: {
			VALIDATE_ARG_NUM(0);
			*r_return = Math::linear2db((double)*p_inputs[0]);
		} break;
		case MATH_DB2LINEAR: {
			VALIDATE_ARG_NUM(0);
			*r_return = Math::db2linear((double)*p_inputs[0]);
		} break;
		case MATH_POLAR2CARTESIAN: {
			VALIDATE_ARG_NUM(0);
			VALIDATE_ARG_NUM(1);
			double r = *p_inputs[0];
			double th = *p_inputs[1];
			*r_return = Vector2(r * Math::cos(th), r * Math::sin(th));
		} break;
		case MATH_CARTESIAN2POLAR: {
			VALIDATE_ARG_NUM(0);
			VALIDATE_ARG_NUM(1);
			double x = *p_inputs[0];
			double y = *p_inputs[1];
			*r_return = Vector2(Math::sqrt(x * x + y * y), Math::atan2(y, x));
		} break;
		case MATH_WRAP: {
			VALIDATE_ARG_NUM(0);
			VALIDATE_ARG_NUM(1);
			VALIDATE_ARG_NUM(2);
			*r_return = Math::wrapi((int64_t)*p_inputs[0], (int64_t)*p_inputs[1], (int64_t)*p_inputs[2]);
		} break;
		case MATH_WRAPF: {
			VALIDATE_ARG_NUM(0);
			VALIDATE_ARG_NUM(1);
			VALIDATE_ARG_NUM(2);
			*r_return = Math::wrapf((double)*p_inputs[0], (double)*p_inputs[1], (double)*p_inputs[2]);
		} break;
		case LOGIC_MAX: {
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
		case LOGIC_MIN: {
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
		case LOGIC_CLAMP: {
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
		case LOGIC_NEAREST_PO2: {
			VALIDATE_ARG_NUM(0);
			int64_t num = *p_inputs[0];
			*r_return = next_power_of_2(num);
		} break;
		case OBJ_WEAKREF: {
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
		case FUNC_FUNCREF: {
			if (p_inputs[0]->get_type() != Variant::OBJECT) {
				r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
				r_error.argument = 0;
				r_error.expected = Variant::OBJECT;

				return;
			}
			if (p_inputs[1]->get_type() != Variant::STRING && p_inputs[1]->get_type() != Variant::NODE_PATH) {
				r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
				r_error.argument = 1;
				r_error.expected = Variant::STRING;

				return;
			}

			Ref<FuncRef> fr = memnew(FuncRef);

			fr->set_instance(*p_inputs[0]);
			fr->set_function(*p_inputs[1]);

			*r_return = fr;

		} break;
		case TYPE_CONVERT: {
			VALIDATE_ARG_NUM(1);
			int type = *p_inputs[1];
			if (type < 0 || type >= Variant::VARIANT_MAX) {
				r_error_str = RTR("Invalid type argument to convert(), use TYPE_* constants.");
				r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
				r_error.argument = 0;
				r_error.expected = Variant::INT;
				return;

			} else {
				*r_return = Variant::construct(Variant::Type(type), p_inputs, 1, r_error);
			}
		} break;
		case TYPE_OF: {
			*r_return = p_inputs[0]->get_type();

		} break;
		case TYPE_EXISTS: {
			*r_return = ClassDB::class_exists(*p_inputs[0]);

		} break;
		case TEXT_CHAR: {
			char32_t result[2] = { *p_inputs[0], 0 };

			*r_return = String(result);

		} break;
		case TEXT_ORD: {
			if (p_inputs[0]->get_type() != Variant::STRING) {
				r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
				r_error.argument = 0;
				r_error.expected = Variant::STRING;

				return;
			}

			String str = *p_inputs[0];

			if (str.length() != 1) {
				r_error_str = RTR("Expected a string of length 1 (a character).");
				r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
				r_error.argument = 0;
				r_error.expected = Variant::STRING;

				return;
			}

			*r_return = str.get(0);

		} break;
		case TEXT_STR: {
			String str = *p_inputs[0];

			*r_return = str;

		} break;
		case TEXT_PRINT: {
			String str = *p_inputs[0];
			print_line(str);

		} break;

		case TEXT_PRINTERR: {
			String str = *p_inputs[0];
			print_error(str);

		} break;
		case TEXT_PRINTRAW: {
			String str = *p_inputs[0];
			OS::get_singleton()->print("%s", str.utf8().get_data());

		} break;
		case VAR_TO_STR: {
			String vars;
			VariantWriter::write_to_string(*p_inputs[0], vars);
			*r_return = vars;
		} break;
		case STR_TO_VAR: {
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
		case VAR_TO_BYTES: {
			PackedByteArray barr;
			bool full_objects = *p_inputs[1];
			int len;
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
		case BYTES_TO_VAR: {
			if (p_inputs[0]->get_type() != Variant::PACKED_BYTE_ARRAY) {
				r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
				r_error.argument = 0;
				r_error.expected = Variant::PACKED_BYTE_ARRAY;

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
		case COLORN: {
			VALIDATE_ARG_NUM(1);

			Color color = Color::named(*p_inputs[0]);
			color.a = *p_inputs[1];

			*r_return = String(color);

		} break;
		default: {
		}
	}
}

////////

static bool _is_number(char32_t c) {
	return (c >= '0' && c <= '9');
}

Error Expression::_get_token(Token &r_token) {
	while (true) {
#define GET_CHAR() (str_ofs >= expression.length() ? 0 : expression[str_ofs++])

		char32_t cchar = GET_CHAR();

		switch (cchar) {
			case 0: {
				r_token.type = TK_EOF;
				return OK;
			}
			case '{': {
				r_token.type = TK_CURLY_BRACKET_OPEN;
				return OK;
			}
			case '}': {
				r_token.type = TK_CURLY_BRACKET_CLOSE;
				return OK;
			}
			case '[': {
				r_token.type = TK_BRACKET_OPEN;
				return OK;
			}
			case ']': {
				r_token.type = TK_BRACKET_CLOSE;
				return OK;
			}
			case '(': {
				r_token.type = TK_PARENTHESIS_OPEN;
				return OK;
			}
			case ')': {
				r_token.type = TK_PARENTHESIS_CLOSE;
				return OK;
			}
			case ',': {
				r_token.type = TK_COMMA;
				return OK;
			}
			case ':': {
				r_token.type = TK_COLON;
				return OK;
			}
			case '$': {
				r_token.type = TK_INPUT;
				int index = 0;
				do {
					if (!_is_number(expression[str_ofs])) {
						_set_error("Expected number after '$'");
						r_token.type = TK_ERROR;
						return ERR_PARSE_ERROR;
					}
					index *= 10;
					index += expression[str_ofs] - '0';
					str_ofs++;

				} while (_is_number(expression[str_ofs]));

				r_token.value = index;
				return OK;
			}
			case '=': {
				cchar = GET_CHAR();
				if (cchar == '=') {
					r_token.type = TK_OP_EQUAL;
				} else {
					_set_error("Expected '='");
					r_token.type = TK_ERROR;
					return ERR_PARSE_ERROR;
				}
				return OK;
			}
			case '!': {
				if (expression[str_ofs] == '=') {
					r_token.type = TK_OP_NOT_EQUAL;
					str_ofs++;
				} else {
					r_token.type = TK_OP_NOT;
				}
				return OK;
			}
			case '>': {
				if (expression[str_ofs] == '=') {
					r_token.type = TK_OP_GREATER_EQUAL;
					str_ofs++;
				} else if (expression[str_ofs] == '>') {
					r_token.type = TK_OP_SHIFT_RIGHT;
					str_ofs++;
				} else {
					r_token.type = TK_OP_GREATER;
				}
				return OK;
			}
			case '<': {
				if (expression[str_ofs] == '=') {
					r_token.type = TK_OP_LESS_EQUAL;
					str_ofs++;
				} else if (expression[str_ofs] == '<') {
					r_token.type = TK_OP_SHIFT_LEFT;
					str_ofs++;
				} else {
					r_token.type = TK_OP_LESS;
				}
				return OK;
			}
			case '+': {
				r_token.type = TK_OP_ADD;
				return OK;
			}
			case '-': {
				r_token.type = TK_OP_SUB;
				return OK;
			}
			case '/': {
				r_token.type = TK_OP_DIV;
				return OK;
			}
			case '*': {
				r_token.type = TK_OP_MUL;
				return OK;
			}
			case '%': {
				r_token.type = TK_OP_MOD;
				return OK;
			}
			case '&': {
				if (expression[str_ofs] == '&') {
					r_token.type = TK_OP_AND;
					str_ofs++;
				} else {
					r_token.type = TK_OP_BIT_AND;
				}
				return OK;
			}
			case '|': {
				if (expression[str_ofs] == '|') {
					r_token.type = TK_OP_OR;
					str_ofs++;
				} else {
					r_token.type = TK_OP_BIT_OR;
				}
				return OK;
			}
			case '^': {
				r_token.type = TK_OP_BIT_XOR;

				return OK;
			}
			case '~': {
				r_token.type = TK_OP_BIT_INVERT;

				return OK;
			}
			case '\'':
			case '"': {
				String str;
				while (true) {
					char32_t ch = GET_CHAR();

					if (ch == 0) {
						_set_error("Unterminated String");
						r_token.type = TK_ERROR;
						return ERR_PARSE_ERROR;
					} else if (ch == cchar) {
						// cchar contain a corresponding quote symbol
						break;
					} else if (ch == '\\') {
						//escaped characters...

						char32_t next = GET_CHAR();
						if (next == 0) {
							_set_error("Unterminated String");
							r_token.type = TK_ERROR;
							return ERR_PARSE_ERROR;
						}
						char32_t res = 0;

						switch (next) {
							case 'b':
								res = 8;
								break;
							case 't':
								res = 9;
								break;
							case 'n':
								res = 10;
								break;
							case 'f':
								res = 12;
								break;
							case 'r':
								res = 13;
								break;
							case 'u': {
								// hex number
								for (int j = 0; j < 4; j++) {
									char32_t c = GET_CHAR();

									if (c == 0) {
										_set_error("Unterminated String");
										r_token.type = TK_ERROR;
										return ERR_PARSE_ERROR;
									}
									if (!(_is_number(c) || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F'))) {
										_set_error("Malformed hex constant in string");
										r_token.type = TK_ERROR;
										return ERR_PARSE_ERROR;
									}
									char32_t v;
									if (_is_number(c)) {
										v = c - '0';
									} else if (c >= 'a' && c <= 'f') {
										v = c - 'a';
										v += 10;
									} else if (c >= 'A' && c <= 'F') {
										v = c - 'A';
										v += 10;
									} else {
										ERR_PRINT("Bug parsing hex constant.");
										v = 0;
									}

									res <<= 4;
									res |= v;
								}

							} break;
							default: {
								res = next;
							} break;
						}

						str += res;

					} else {
						str += ch;
					}
				}

				r_token.type = TK_CONSTANT;
				r_token.value = str;
				return OK;

			} break;
			default: {
				if (cchar <= 32) {
					break;
				}

				char32_t next_char = (str_ofs >= expression.length()) ? 0 : expression[str_ofs];
				if (_is_number(cchar) || (cchar == '.' && _is_number(next_char))) {
					//a number

					String num;
#define READING_SIGN 0
#define READING_INT 1
#define READING_DEC 2
#define READING_EXP 3
#define READING_DONE 4
					int reading = READING_INT;

					char32_t c = cchar;
					bool exp_sign = false;
					bool exp_beg = false;
					bool is_float = false;

					while (true) {
						switch (reading) {
							case READING_INT: {
								if (_is_number(c)) {
									//pass
								} else if (c == '.') {
									reading = READING_DEC;
									is_float = true;
								} else if (c == 'e') {
									reading = READING_EXP;
								} else {
									reading = READING_DONE;
								}

							} break;
							case READING_DEC: {
								if (_is_number(c)) {
								} else if (c == 'e') {
									reading = READING_EXP;

								} else {
									reading = READING_DONE;
								}

							} break;
							case READING_EXP: {
								if (_is_number(c)) {
									exp_beg = true;

								} else if ((c == '-' || c == '+') && !exp_sign && !exp_beg) {
									if (c == '-') {
										is_float = true;
									}
									exp_sign = true;

								} else {
									reading = READING_DONE;
								}
							} break;
						}

						if (reading == READING_DONE) {
							break;
						}
						num += String::chr(c);
						c = GET_CHAR();
					}

					str_ofs--;

					r_token.type = TK_CONSTANT;

					if (is_float) {
						r_token.value = num.to_float();
					} else {
						r_token.value = num.to_int();
					}
					return OK;

				} else if ((cchar >= 'A' && cchar <= 'Z') || (cchar >= 'a' && cchar <= 'z') || cchar == '_') {
					String id;
					bool first = true;

					while ((cchar >= 'A' && cchar <= 'Z') || (cchar >= 'a' && cchar <= 'z') || cchar == '_' || (!first && _is_number(cchar))) {
						id += String::chr(cchar);
						cchar = GET_CHAR();
						first = false;
					}

					str_ofs--; //go back one

					if (id == "in") {
						r_token.type = TK_OP_IN;
					} else if (id == "null") {
						r_token.type = TK_CONSTANT;
						r_token.value = Variant();
					} else if (id == "true") {
						r_token.type = TK_CONSTANT;
						r_token.value = true;
					} else if (id == "false") {
						r_token.type = TK_CONSTANT;
						r_token.value = false;
					} else if (id == "PI") {
						r_token.type = TK_CONSTANT;
						r_token.value = Math_PI;
					} else if (id == "TAU") {
						r_token.type = TK_CONSTANT;
						r_token.value = Math_TAU;
					} else if (id == "INF") {
						r_token.type = TK_CONSTANT;
						r_token.value = Math_INF;
					} else if (id == "NAN") {
						r_token.type = TK_CONSTANT;
						r_token.value = Math_NAN;
					} else if (id == "not") {
						r_token.type = TK_OP_NOT;
					} else if (id == "or") {
						r_token.type = TK_OP_OR;
					} else if (id == "and") {
						r_token.type = TK_OP_AND;
					} else if (id == "self") {
						r_token.type = TK_SELF;
					} else {
						for (int i = 0; i < Variant::VARIANT_MAX; i++) {
							if (id == Variant::get_type_name(Variant::Type(i))) {
								r_token.type = TK_BASIC_TYPE;
								r_token.value = i;
								return OK;
							}
						}

						BuiltinFunc bifunc = find_function(id);
						if (bifunc != FUNC_MAX) {
							r_token.type = TK_BUILTIN_FUNC;
							r_token.value = bifunc;
							return OK;
						}

						r_token.type = TK_IDENTIFIER;
						r_token.value = id;
					}

					return OK;

				} else if (cchar == '.') {
					// Handled down there as we support '.[0-9]' as numbers above
					r_token.type = TK_PERIOD;
					return OK;

				} else {
					_set_error("Unexpected character.");
					r_token.type = TK_ERROR;
					return ERR_PARSE_ERROR;
				}
			}
		}
#undef GET_CHAR
	}

	r_token.type = TK_ERROR;
	return ERR_PARSE_ERROR;
}

const char *Expression::token_name[TK_MAX] = {
	"CURLY BRACKET OPEN",
	"CURLY BRACKET CLOSE",
	"BRACKET OPEN",
	"BRACKET CLOSE",
	"PARENTHESIS OPEN",
	"PARENTHESIS CLOSE",
	"IDENTIFIER",
	"BUILTIN FUNC",
	"SELF",
	"CONSTANT",
	"BASIC TYPE",
	"COLON",
	"COMMA",
	"PERIOD",
	"OP IN",
	"OP EQUAL",
	"OP NOT EQUAL",
	"OP LESS",
	"OP LESS EQUAL",
	"OP GREATER",
	"OP GREATER EQUAL",
	"OP AND",
	"OP OR",
	"OP NOT",
	"OP ADD",
	"OP SUB",
	"OP MUL",
	"OP DIV",
	"OP MOD",
	"OP SHIFT LEFT",
	"OP SHIFT RIGHT",
	"OP BIT AND",
	"OP BIT OR",
	"OP BIT XOR",
	"OP BIT INVERT",
	"OP INPUT",
	"EOF",
	"ERROR"
};

Expression::ENode *Expression::_parse_expression() {
	Vector<ExpressionNode> expression;

	while (true) {
		//keep appending stuff to expression
		ENode *expr = nullptr;

		Token tk;
		_get_token(tk);
		if (error_set) {
			return nullptr;
		}

		switch (tk.type) {
			case TK_CURLY_BRACKET_OPEN: {
				//a dictionary
				DictionaryNode *dn = alloc_node<DictionaryNode>();

				while (true) {
					int cofs = str_ofs;
					_get_token(tk);
					if (tk.type == TK_CURLY_BRACKET_CLOSE) {
						break;
					}
					str_ofs = cofs; //revert
					//parse an expression
					ENode *subexpr = _parse_expression();
					if (!subexpr) {
						return nullptr;
					}
					dn->dict.push_back(subexpr);

					_get_token(tk);
					if (tk.type != TK_COLON) {
						_set_error("Expected ':'");
						return nullptr;
					}

					subexpr = _parse_expression();
					if (!subexpr) {
						return nullptr;
					}

					dn->dict.push_back(subexpr);

					cofs = str_ofs;
					_get_token(tk);
					if (tk.type == TK_COMMA) {
						//all good
					} else if (tk.type == TK_CURLY_BRACKET_CLOSE) {
						str_ofs = cofs;
					} else {
						_set_error("Expected ',' or '}'");
					}
				}

				expr = dn;
			} break;
			case TK_BRACKET_OPEN: {
				//an array

				ArrayNode *an = alloc_node<ArrayNode>();

				while (true) {
					int cofs = str_ofs;
					_get_token(tk);
					if (tk.type == TK_BRACKET_CLOSE) {
						break;
					}
					str_ofs = cofs; //revert
					//parse an expression
					ENode *subexpr = _parse_expression();
					if (!subexpr) {
						return nullptr;
					}
					an->array.push_back(subexpr);

					cofs = str_ofs;
					_get_token(tk);
					if (tk.type == TK_COMMA) {
						//all good
					} else if (tk.type == TK_BRACKET_CLOSE) {
						str_ofs = cofs;
					} else {
						_set_error("Expected ',' or ']'");
					}
				}

				expr = an;
			} break;
			case TK_PARENTHESIS_OPEN: {
				//a suexpression
				ENode *e = _parse_expression();
				if (error_set) {
					return nullptr;
				}
				_get_token(tk);
				if (tk.type != TK_PARENTHESIS_CLOSE) {
					_set_error("Expected ')'");
					return nullptr;
				}

				expr = e;

			} break;
			case TK_IDENTIFIER: {
				String identifier = tk.value;

				int cofs = str_ofs;
				_get_token(tk);
				if (tk.type == TK_PARENTHESIS_OPEN) {
					//function call
					CallNode *func_call = alloc_node<CallNode>();
					func_call->method = identifier;
					SelfNode *self_node = alloc_node<SelfNode>();
					func_call->base = self_node;

					while (true) {
						int cofs2 = str_ofs;
						_get_token(tk);
						if (tk.type == TK_PARENTHESIS_CLOSE) {
							break;
						}
						str_ofs = cofs2; //revert
						//parse an expression
						ENode *subexpr = _parse_expression();
						if (!subexpr) {
							return nullptr;
						}

						func_call->arguments.push_back(subexpr);

						cofs2 = str_ofs;
						_get_token(tk);
						if (tk.type == TK_COMMA) {
							//all good
						} else if (tk.type == TK_PARENTHESIS_CLOSE) {
							str_ofs = cofs2;
						} else {
							_set_error("Expected ',' or ')'");
						}
					}

					expr = func_call;
				} else {
					//named indexing
					str_ofs = cofs;

					int input_index = -1;
					for (int i = 0; i < input_names.size(); i++) {
						if (input_names[i] == identifier) {
							input_index = i;
							break;
						}
					}

					if (input_index != -1) {
						InputNode *input = alloc_node<InputNode>();
						input->index = input_index;
						expr = input;
					} else {
						NamedIndexNode *index = alloc_node<NamedIndexNode>();
						SelfNode *self_node = alloc_node<SelfNode>();
						index->base = self_node;
						index->name = identifier;
						expr = index;
					}
				}
			} break;
			case TK_INPUT: {
				InputNode *input = alloc_node<InputNode>();
				input->index = tk.value;
				expr = input;
			} break;
			case TK_SELF: {
				SelfNode *self = alloc_node<SelfNode>();
				expr = self;
			} break;
			case TK_CONSTANT: {
				ConstantNode *constant = alloc_node<ConstantNode>();
				constant->value = tk.value;
				expr = constant;
			} break;
			case TK_BASIC_TYPE: {
				//constructor..

				Variant::Type bt = Variant::Type(int(tk.value));
				_get_token(tk);
				if (tk.type != TK_PARENTHESIS_OPEN) {
					_set_error("Expected '('");
					return nullptr;
				}

				ConstructorNode *constructor = alloc_node<ConstructorNode>();
				constructor->data_type = bt;

				while (true) {
					int cofs = str_ofs;
					_get_token(tk);
					if (tk.type == TK_PARENTHESIS_CLOSE) {
						break;
					}
					str_ofs = cofs; //revert
					//parse an expression
					ENode *subexpr = _parse_expression();
					if (!subexpr) {
						return nullptr;
					}

					constructor->arguments.push_back(subexpr);

					cofs = str_ofs;
					_get_token(tk);
					if (tk.type == TK_COMMA) {
						//all good
					} else if (tk.type == TK_PARENTHESIS_CLOSE) {
						str_ofs = cofs;
					} else {
						_set_error("Expected ',' or ')'");
					}
				}

				expr = constructor;

			} break;
			case TK_BUILTIN_FUNC: {
				//builtin function

				_get_token(tk);
				if (tk.type != TK_PARENTHESIS_OPEN) {
					_set_error("Expected '('");
					return nullptr;
				}

				BuiltinFuncNode *bifunc = alloc_node<BuiltinFuncNode>();
				bifunc->func = BuiltinFunc(int(tk.value));

				while (true) {
					int cofs = str_ofs;
					_get_token(tk);
					if (tk.type == TK_PARENTHESIS_CLOSE) {
						break;
					}
					str_ofs = cofs; //revert
					//parse an expression
					ENode *subexpr = _parse_expression();
					if (!subexpr) {
						return nullptr;
					}

					bifunc->arguments.push_back(subexpr);

					cofs = str_ofs;
					_get_token(tk);
					if (tk.type == TK_COMMA) {
						//all good
					} else if (tk.type == TK_PARENTHESIS_CLOSE) {
						str_ofs = cofs;
					} else {
						_set_error("Expected ',' or ')'");
					}
				}

				int expected_args = get_func_argument_count(bifunc->func);
				if (bifunc->arguments.size() != expected_args) {
					_set_error("Builtin func '" + get_func_name(bifunc->func) + "' expects " + itos(expected_args) + " arguments.");
				}

				expr = bifunc;

			} break;
			case TK_OP_SUB: {
				ExpressionNode e;
				e.is_op = true;
				e.op = Variant::OP_NEGATE;
				expression.push_back(e);
				continue;
			} break;
			case TK_OP_NOT: {
				ExpressionNode e;
				e.is_op = true;
				e.op = Variant::OP_NOT;
				expression.push_back(e);
				continue;
			} break;

			default: {
				_set_error("Expected expression.");
				return nullptr;
			} break;
		}

		//before going to operators, must check indexing!

		while (true) {
			int cofs2 = str_ofs;
			_get_token(tk);
			if (error_set) {
				return nullptr;
			}

			bool done = false;

			switch (tk.type) {
				case TK_BRACKET_OPEN: {
					//value indexing

					IndexNode *index = alloc_node<IndexNode>();
					index->base = expr;

					ENode *what = _parse_expression();
					if (!what) {
						return nullptr;
					}

					index->index = what;

					_get_token(tk);
					if (tk.type != TK_BRACKET_CLOSE) {
						_set_error("Expected ']' at end of index.");
						return nullptr;
					}
					expr = index;

				} break;
				case TK_PERIOD: {
					//named indexing or function call
					_get_token(tk);
					if (tk.type != TK_IDENTIFIER) {
						_set_error("Expected identifier after '.'");
						return nullptr;
					}

					StringName identifier = tk.value;

					int cofs = str_ofs;
					_get_token(tk);
					if (tk.type == TK_PARENTHESIS_OPEN) {
						//function call
						CallNode *func_call = alloc_node<CallNode>();
						func_call->method = identifier;
						func_call->base = expr;

						while (true) {
							int cofs3 = str_ofs;
							_get_token(tk);
							if (tk.type == TK_PARENTHESIS_CLOSE) {
								break;
							}
							str_ofs = cofs3; //revert
							//parse an expression
							ENode *subexpr = _parse_expression();
							if (!subexpr) {
								return nullptr;
							}

							func_call->arguments.push_back(subexpr);

							cofs3 = str_ofs;
							_get_token(tk);
							if (tk.type == TK_COMMA) {
								//all good
							} else if (tk.type == TK_PARENTHESIS_CLOSE) {
								str_ofs = cofs3;
							} else {
								_set_error("Expected ',' or ')'");
							}
						}

						expr = func_call;
					} else {
						//named indexing
						str_ofs = cofs;

						NamedIndexNode *index = alloc_node<NamedIndexNode>();
						index->base = expr;
						index->name = identifier;
						expr = index;
					}

				} break;
				default: {
					str_ofs = cofs2;
					done = true;
				} break;
			}

			if (done) {
				break;
			}
		}

		//push expression
		{
			ExpressionNode e;
			e.is_op = false;
			e.node = expr;
			expression.push_back(e);
		}

		//ok finally look for an operator

		int cofs = str_ofs;
		_get_token(tk);
		if (error_set) {
			return nullptr;
		}

		Variant::Operator op = Variant::OP_MAX;

		switch (tk.type) {
			case TK_OP_IN:
				op = Variant::OP_IN;
				break;
			case TK_OP_EQUAL:
				op = Variant::OP_EQUAL;
				break;
			case TK_OP_NOT_EQUAL:
				op = Variant::OP_NOT_EQUAL;
				break;
			case TK_OP_LESS:
				op = Variant::OP_LESS;
				break;
			case TK_OP_LESS_EQUAL:
				op = Variant::OP_LESS_EQUAL;
				break;
			case TK_OP_GREATER:
				op = Variant::OP_GREATER;
				break;
			case TK_OP_GREATER_EQUAL:
				op = Variant::OP_GREATER_EQUAL;
				break;
			case TK_OP_AND:
				op = Variant::OP_AND;
				break;
			case TK_OP_OR:
				op = Variant::OP_OR;
				break;
			case TK_OP_NOT:
				op = Variant::OP_NOT;
				break;
			case TK_OP_ADD:
				op = Variant::OP_ADD;
				break;
			case TK_OP_SUB:
				op = Variant::OP_SUBTRACT;
				break;
			case TK_OP_MUL:
				op = Variant::OP_MULTIPLY;
				break;
			case TK_OP_DIV:
				op = Variant::OP_DIVIDE;
				break;
			case TK_OP_MOD:
				op = Variant::OP_MODULE;
				break;
			case TK_OP_SHIFT_LEFT:
				op = Variant::OP_SHIFT_LEFT;
				break;
			case TK_OP_SHIFT_RIGHT:
				op = Variant::OP_SHIFT_RIGHT;
				break;
			case TK_OP_BIT_AND:
				op = Variant::OP_BIT_AND;
				break;
			case TK_OP_BIT_OR:
				op = Variant::OP_BIT_OR;
				break;
			case TK_OP_BIT_XOR:
				op = Variant::OP_BIT_XOR;
				break;
			case TK_OP_BIT_INVERT:
				op = Variant::OP_BIT_NEGATE;
				break;
			default: {
			}
		}

		if (op == Variant::OP_MAX) { //stop appending stuff
			str_ofs = cofs;
			break;
		}

		//push operator and go on
		{
			ExpressionNode e;
			e.is_op = true;
			e.op = op;
			expression.push_back(e);
		}
	}

	/* Reduce the set set of expressions and place them in an operator tree, respecting precedence */

	while (expression.size() > 1) {
		int next_op = -1;
		int min_priority = 0xFFFFF;
		bool is_unary = false;

		for (int i = 0; i < expression.size(); i++) {
			if (!expression[i].is_op) {
				continue;
			}

			int priority;

			bool unary = false;

			switch (expression[i].op) {
				case Variant::OP_BIT_NEGATE:
					priority = 0;
					unary = true;
					break;
				case Variant::OP_NEGATE:
					priority = 1;
					unary = true;
					break;

				case Variant::OP_MULTIPLY:
					priority = 2;
					break;
				case Variant::OP_DIVIDE:
					priority = 2;
					break;
				case Variant::OP_MODULE:
					priority = 2;
					break;

				case Variant::OP_ADD:
					priority = 3;
					break;
				case Variant::OP_SUBTRACT:
					priority = 3;
					break;

				case Variant::OP_SHIFT_LEFT:
					priority = 4;
					break;
				case Variant::OP_SHIFT_RIGHT:
					priority = 4;
					break;

				case Variant::OP_BIT_AND:
					priority = 5;
					break;
				case Variant::OP_BIT_XOR:
					priority = 6;
					break;
				case Variant::OP_BIT_OR:
					priority = 7;
					break;

				case Variant::OP_LESS:
					priority = 8;
					break;
				case Variant::OP_LESS_EQUAL:
					priority = 8;
					break;
				case Variant::OP_GREATER:
					priority = 8;
					break;
				case Variant::OP_GREATER_EQUAL:
					priority = 8;
					break;

				case Variant::OP_EQUAL:
					priority = 8;
					break;
				case Variant::OP_NOT_EQUAL:
					priority = 8;
					break;

				case Variant::OP_IN:
					priority = 10;
					break;

				case Variant::OP_NOT:
					priority = 11;
					unary = true;
					break;
				case Variant::OP_AND:
					priority = 12;
					break;
				case Variant::OP_OR:
					priority = 13;
					break;

				default: {
					_set_error("Parser bug, invalid operator in expression: " + itos(expression[i].op));
					return nullptr;
				}
			}

			if (priority < min_priority) {
				// < is used for left to right (default)
				// <= is used for right to left

				next_op = i;
				min_priority = priority;
				is_unary = unary;
			}
		}

		if (next_op == -1) {
			_set_error("Yet another parser bug....");
			ERR_FAIL_V(nullptr);
		}

		// OK! create operator..
		if (is_unary) {
			int expr_pos = next_op;
			while (expression[expr_pos].is_op) {
				expr_pos++;
				if (expr_pos == expression.size()) {
					//can happen..
					_set_error("Unexpected end of expression...");
					return nullptr;
				}
			}

			//consecutively do unary operators
			for (int i = expr_pos - 1; i >= next_op; i--) {
				OperatorNode *op = alloc_node<OperatorNode>();
				op->op = expression[i].op;
				op->nodes[0] = expression[i + 1].node;
				op->nodes[1] = nullptr;
				expression.write[i].is_op = false;
				expression.write[i].node = op;
				expression.remove(i + 1);
			}

		} else {
			if (next_op < 1 || next_op >= (expression.size() - 1)) {
				_set_error("Parser bug...");
				ERR_FAIL_V(nullptr);
			}

			OperatorNode *op = alloc_node<OperatorNode>();
			op->op = expression[next_op].op;

			if (expression[next_op - 1].is_op) {
				_set_error("Parser bug...");
				ERR_FAIL_V(nullptr);
			}

			if (expression[next_op + 1].is_op) {
				// this is not invalid and can really appear
				// but it becomes invalid anyway because no binary op
				// can be followed by a unary op in a valid combination,
				// due to how precedence works, unaries will always disappear first

				_set_error("Unexpected two consecutive operators.");
				return nullptr;
			}

			op->nodes[0] = expression[next_op - 1].node; //expression goes as left
			op->nodes[1] = expression[next_op + 1].node; //next expression goes as right

			//replace all 3 nodes by this operator and make it an expression
			expression.write[next_op - 1].node = op;
			expression.remove(next_op);
			expression.remove(next_op);
		}
	}

	return expression[0].node;
}

bool Expression::_compile_expression() {
	if (!expression_dirty) {
		return error_set;
	}

	if (nodes) {
		memdelete(nodes);
		nodes = nullptr;
		root = nullptr;
	}

	error_str = String();
	error_set = false;
	str_ofs = 0;

	root = _parse_expression();

	if (error_set) {
		root = nullptr;
		if (nodes) {
			memdelete(nodes);
		}
		nodes = nullptr;
		return true;
	}

	expression_dirty = false;
	return false;
}

bool Expression::_execute(const Array &p_inputs, Object *p_instance, Expression::ENode *p_node, Variant &r_ret, String &r_error_str) {
	switch (p_node->type) {
		case Expression::ENode::TYPE_INPUT: {
			const Expression::InputNode *in = static_cast<const Expression::InputNode *>(p_node);
			if (in->index < 0 || in->index >= p_inputs.size()) {
				r_error_str = vformat(RTR("Invalid input %i (not passed) in expression"), in->index);
				return true;
			}
			r_ret = p_inputs[in->index];
		} break;
		case Expression::ENode::TYPE_CONSTANT: {
			const Expression::ConstantNode *c = static_cast<const Expression::ConstantNode *>(p_node);
			r_ret = c->value;

		} break;
		case Expression::ENode::TYPE_SELF: {
			if (!p_instance) {
				r_error_str = RTR("self can't be used because instance is null (not passed)");
				return true;
			}
			r_ret = p_instance;
		} break;
		case Expression::ENode::TYPE_OPERATOR: {
			const Expression::OperatorNode *op = static_cast<const Expression::OperatorNode *>(p_node);

			Variant a;
			bool ret = _execute(p_inputs, p_instance, op->nodes[0], a, r_error_str);
			if (ret) {
				return true;
			}

			Variant b;

			if (op->nodes[1]) {
				ret = _execute(p_inputs, p_instance, op->nodes[1], b, r_error_str);
				if (ret) {
					return true;
				}
			}

			bool valid = true;
			Variant::evaluate(op->op, a, b, r_ret, valid);
			if (!valid) {
				r_error_str = vformat(RTR("Invalid operands to operator %s, %s and %s."), Variant::get_operator_name(op->op), Variant::get_type_name(a.get_type()), Variant::get_type_name(b.get_type()));
				return true;
			}

		} break;
		case Expression::ENode::TYPE_INDEX: {
			const Expression::IndexNode *index = static_cast<const Expression::IndexNode *>(p_node);

			Variant base;
			bool ret = _execute(p_inputs, p_instance, index->base, base, r_error_str);
			if (ret) {
				return true;
			}

			Variant idx;

			ret = _execute(p_inputs, p_instance, index->index, idx, r_error_str);
			if (ret) {
				return true;
			}

			bool valid;
			r_ret = base.get(idx, &valid);
			if (!valid) {
				r_error_str = vformat(RTR("Invalid index of type %s for base type %s"), Variant::get_type_name(idx.get_type()), Variant::get_type_name(base.get_type()));
				return true;
			}

		} break;
		case Expression::ENode::TYPE_NAMED_INDEX: {
			const Expression::NamedIndexNode *index = static_cast<const Expression::NamedIndexNode *>(p_node);

			Variant base;
			bool ret = _execute(p_inputs, p_instance, index->base, base, r_error_str);
			if (ret) {
				return true;
			}

			bool valid;
			r_ret = base.get_named(index->name, &valid);
			if (!valid) {
				r_error_str = vformat(RTR("Invalid named index '%s' for base type %s"), String(index->name), Variant::get_type_name(base.get_type()));
				return true;
			}

		} break;
		case Expression::ENode::TYPE_ARRAY: {
			const Expression::ArrayNode *array = static_cast<const Expression::ArrayNode *>(p_node);

			Array arr;
			arr.resize(array->array.size());
			for (int i = 0; i < array->array.size(); i++) {
				Variant value;
				bool ret = _execute(p_inputs, p_instance, array->array[i], value, r_error_str);

				if (ret) {
					return true;
				}
				arr[i] = value;
			}

			r_ret = arr;

		} break;
		case Expression::ENode::TYPE_DICTIONARY: {
			const Expression::DictionaryNode *dictionary = static_cast<const Expression::DictionaryNode *>(p_node);

			Dictionary d;
			for (int i = 0; i < dictionary->dict.size(); i += 2) {
				Variant key;
				bool ret = _execute(p_inputs, p_instance, dictionary->dict[i + 0], key, r_error_str);

				if (ret) {
					return true;
				}

				Variant value;
				ret = _execute(p_inputs, p_instance, dictionary->dict[i + 1], value, r_error_str);
				if (ret) {
					return true;
				}

				d[key] = value;
			}

			r_ret = d;
		} break;
		case Expression::ENode::TYPE_CONSTRUCTOR: {
			const Expression::ConstructorNode *constructor = static_cast<const Expression::ConstructorNode *>(p_node);

			Vector<Variant> arr;
			Vector<const Variant *> argp;
			arr.resize(constructor->arguments.size());
			argp.resize(constructor->arguments.size());

			for (int i = 0; i < constructor->arguments.size(); i++) {
				Variant value;
				bool ret = _execute(p_inputs, p_instance, constructor->arguments[i], value, r_error_str);

				if (ret) {
					return true;
				}
				arr.write[i] = value;
				argp.write[i] = &arr[i];
			}

			Callable::CallError ce;
			r_ret = Variant::construct(constructor->data_type, (const Variant **)argp.ptr(), argp.size(), ce);

			if (ce.error != Callable::CallError::CALL_OK) {
				r_error_str = vformat(RTR("Invalid arguments to construct '%s'"), Variant::get_type_name(constructor->data_type));
				return true;
			}

		} break;
		case Expression::ENode::TYPE_BUILTIN_FUNC: {
			const Expression::BuiltinFuncNode *bifunc = static_cast<const Expression::BuiltinFuncNode *>(p_node);

			Vector<Variant> arr;
			Vector<const Variant *> argp;
			arr.resize(bifunc->arguments.size());
			argp.resize(bifunc->arguments.size());

			for (int i = 0; i < bifunc->arguments.size(); i++) {
				Variant value;
				bool ret = _execute(p_inputs, p_instance, bifunc->arguments[i], value, r_error_str);
				if (ret) {
					return true;
				}
				arr.write[i] = value;
				argp.write[i] = &arr[i];
			}

			Callable::CallError ce;
			exec_func(bifunc->func, (const Variant **)argp.ptr(), &r_ret, ce, r_error_str);

			if (ce.error != Callable::CallError::CALL_OK) {
				r_error_str = "Builtin Call Failed. " + r_error_str;
				return true;
			}

		} break;
		case Expression::ENode::TYPE_CALL: {
			const Expression::CallNode *call = static_cast<const Expression::CallNode *>(p_node);

			Variant base;
			bool ret = _execute(p_inputs, p_instance, call->base, base, r_error_str);

			if (ret) {
				return true;
			}

			Vector<Variant> arr;
			Vector<const Variant *> argp;
			arr.resize(call->arguments.size());
			argp.resize(call->arguments.size());

			for (int i = 0; i < call->arguments.size(); i++) {
				Variant value;
				ret = _execute(p_inputs, p_instance, call->arguments[i], value, r_error_str);

				if (ret) {
					return true;
				}
				arr.write[i] = value;
				argp.write[i] = &arr[i];
			}

			Callable::CallError ce;
			r_ret = base.call(call->method, (const Variant **)argp.ptr(), argp.size(), ce);

			if (ce.error != Callable::CallError::CALL_OK) {
				r_error_str = vformat(RTR("On call to '%s':"), String(call->method));
				return true;
			}

		} break;
	}
	return false;
}

Error Expression::parse(const String &p_expression, const Vector<String> &p_input_names) {
	if (nodes) {
		memdelete(nodes);
		nodes = nullptr;
		root = nullptr;
	}

	error_str = String();
	error_set = false;
	str_ofs = 0;
	input_names = p_input_names;

	expression = p_expression;
	root = _parse_expression();

	if (error_set) {
		root = nullptr;
		if (nodes) {
			memdelete(nodes);
		}
		nodes = nullptr;
		return ERR_INVALID_PARAMETER;
	}

	return OK;
}

Variant Expression::execute(Array p_inputs, Object *p_base, bool p_show_error) {
	ERR_FAIL_COND_V_MSG(error_set, Variant(), "There was previously a parse error: " + error_str + ".");

	execution_error = false;
	Variant output;
	String error_txt;
	bool err = _execute(p_inputs, p_base, root, output, error_txt);
	if (err) {
		execution_error = true;
		error_str = error_txt;
		ERR_FAIL_COND_V_MSG(p_show_error, Variant(), error_str);
	}

	return output;
}

bool Expression::has_execute_failed() const {
	return execution_error;
}

String Expression::get_error_text() const {
	return error_str;
}

void Expression::_bind_methods() {
	ClassDB::bind_method(D_METHOD("parse", "expression", "input_names"), &Expression::parse, DEFVAL(Vector<String>()));
	ClassDB::bind_method(D_METHOD("execute", "inputs", "base_instance", "show_error"), &Expression::execute, DEFVAL(Array()), DEFVAL(Variant()), DEFVAL(true));
	ClassDB::bind_method(D_METHOD("has_execute_failed"), &Expression::has_execute_failed);
	ClassDB::bind_method(D_METHOD("get_error_text"), &Expression::get_error_text);
}

Expression::~Expression() {
	if (nodes) {
		memdelete(nodes);
	}
}
