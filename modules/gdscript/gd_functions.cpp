/*************************************************************************/
/*  gd_functions.cpp                                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "gd_functions.h"
#include "class_db.h"
#include "func_ref.h"
#include "gd_script.h"
#include "io/json.h"
#include "io/marshalls.h"
#include "math_funcs.h"
#include "os/os.h"
#include "reference.h"
#include "variant_parser.h"

const char *GDFunctions::get_func_name(Function p_func) {

	ERR_FAIL_INDEX_V(p_func, FUNC_MAX, "");

	static const char *_names[FUNC_MAX] = {
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
		"decimals",
		"stepify",
		"lerp",
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
		"str",
		"print",
		"printt",
		"prints",
		"printerr",
		"printraw",
		"var2str",
		"str2var",
		"var2bytes",
		"bytes2var",
		"range",
		"load",
		"inst2dict",
		"dict2inst",
		"validate_json",
		"parse_json",
		"to_json",
		"hash",
		"Color8",
		"ColorN",
		"print_stack",
		"instance_from_id",
	};

	return _names[p_func];
}

void GDFunctions::call(Function p_func, const Variant **p_args, int p_arg_count, Variant &r_ret, Variant::CallError &r_error) {

	r_error.error = Variant::CallError::CALL_OK;
#ifdef DEBUG_ENABLED

#define VALIDATE_ARG_COUNT(m_count)                                        \
	if (p_arg_count < m_count) {                                           \
		r_error.error = Variant::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS;  \
		r_error.argument = m_count;                                        \
		r_ret = Variant();                                                 \
		return;                                                            \
	}                                                                      \
	if (p_arg_count > m_count) {                                           \
		r_error.error = Variant::CallError::CALL_ERROR_TOO_MANY_ARGUMENTS; \
		r_error.argument = m_count;                                        \
		r_ret = Variant();                                                 \
		return;                                                            \
	}

#define VALIDATE_ARG_NUM(m_arg)                                          \
	if (!p_args[m_arg]->is_num()) {                                      \
		r_error.error = Variant::CallError::CALL_ERROR_INVALID_ARGUMENT; \
		r_error.argument = m_arg;                                        \
		r_error.expected = Variant::REAL;                                \
		r_ret = Variant();                                               \
		return;                                                          \
	}

#else

#define VALIDATE_ARG_COUNT(m_count)
#define VALIDATE_ARG_NUM(m_arg)
#endif

	//using a switch, so the compiler generates a jumptable

	switch (p_func) {

		case MATH_SIN: {
			VALIDATE_ARG_COUNT(1);
			VALIDATE_ARG_NUM(0);
			r_ret = Math::sin((double)*p_args[0]);
		} break;
		case MATH_COS: {
			VALIDATE_ARG_COUNT(1);
			VALIDATE_ARG_NUM(0);
			r_ret = Math::cos((double)*p_args[0]);
		} break;
		case MATH_TAN: {
			VALIDATE_ARG_COUNT(1);
			VALIDATE_ARG_NUM(0);
			r_ret = Math::tan((double)*p_args[0]);
		} break;
		case MATH_SINH: {
			VALIDATE_ARG_COUNT(1);
			VALIDATE_ARG_NUM(0);
			r_ret = Math::sinh((double)*p_args[0]);
		} break;
		case MATH_COSH: {
			VALIDATE_ARG_COUNT(1);
			VALIDATE_ARG_NUM(0);
			r_ret = Math::cosh((double)*p_args[0]);
		} break;
		case MATH_TANH: {
			VALIDATE_ARG_COUNT(1);
			VALIDATE_ARG_NUM(0);
			r_ret = Math::tanh((double)*p_args[0]);
		} break;
		case MATH_ASIN: {
			VALIDATE_ARG_COUNT(1);
			VALIDATE_ARG_NUM(0);
			r_ret = Math::asin((double)*p_args[0]);
		} break;
		case MATH_ACOS: {
			VALIDATE_ARG_COUNT(1);
			VALIDATE_ARG_NUM(0);
			r_ret = Math::acos((double)*p_args[0]);
		} break;
		case MATH_ATAN: {
			VALIDATE_ARG_COUNT(1);
			VALIDATE_ARG_NUM(0);
			r_ret = Math::atan((double)*p_args[0]);
		} break;
		case MATH_ATAN2: {
			VALIDATE_ARG_COUNT(2);
			VALIDATE_ARG_NUM(0);
			VALIDATE_ARG_NUM(1);
			r_ret = Math::atan2((double)*p_args[0], (double)*p_args[1]);
		} break;
		case MATH_SQRT: {
			VALIDATE_ARG_COUNT(1);
			VALIDATE_ARG_NUM(0);
			r_ret = Math::sqrt((double)*p_args[0]);
		} break;
		case MATH_FMOD: {
			VALIDATE_ARG_COUNT(2);
			VALIDATE_ARG_NUM(0);
			VALIDATE_ARG_NUM(1);
			r_ret = Math::fmod((double)*p_args[0], (double)*p_args[1]);
		} break;
		case MATH_FPOSMOD: {
			VALIDATE_ARG_COUNT(2);
			VALIDATE_ARG_NUM(0);
			VALIDATE_ARG_NUM(1);
			r_ret = Math::fposmod((double)*p_args[0], (double)*p_args[1]);
		} break;
		case MATH_FLOOR: {
			VALIDATE_ARG_COUNT(1);
			VALIDATE_ARG_NUM(0);
			r_ret = Math::floor((double)*p_args[0]);
		} break;
		case MATH_CEIL: {
			VALIDATE_ARG_COUNT(1);
			VALIDATE_ARG_NUM(0);
			r_ret = Math::ceil((double)*p_args[0]);
		} break;
		case MATH_ROUND: {
			VALIDATE_ARG_COUNT(1);
			VALIDATE_ARG_NUM(0);
			r_ret = Math::round((double)*p_args[0]);
		} break;
		case MATH_ABS: {
			VALIDATE_ARG_COUNT(1);
			if (p_args[0]->get_type() == Variant::INT) {

				int64_t i = *p_args[0];
				r_ret = ABS(i);
			} else if (p_args[0]->get_type() == Variant::REAL) {

				double r = *p_args[0];
				r_ret = Math::abs(r);
			} else {

				r_error.error = Variant::CallError::CALL_ERROR_INVALID_ARGUMENT;
				r_error.argument = 0;
				r_error.expected = Variant::REAL;
				r_ret = Variant();
			}
		} break;
		case MATH_SIGN: {
			VALIDATE_ARG_COUNT(1);
			if (p_args[0]->get_type() == Variant::INT) {

				int64_t i = *p_args[0];
				r_ret = i < 0 ? -1 : (i > 0 ? +1 : 0);
			} else if (p_args[0]->get_type() == Variant::REAL) {

				real_t r = *p_args[0];
				r_ret = r < 0.0 ? -1.0 : (r > 0.0 ? +1.0 : 0.0);
			} else {

				r_error.error = Variant::CallError::CALL_ERROR_INVALID_ARGUMENT;
				r_error.argument = 0;
				r_error.expected = Variant::REAL;
				r_ret = Variant();
			}
		} break;
		case MATH_POW: {
			VALIDATE_ARG_COUNT(2);
			VALIDATE_ARG_NUM(0);
			VALIDATE_ARG_NUM(1);
			r_ret = Math::pow((double)*p_args[0], (double)*p_args[1]);
		} break;
		case MATH_LOG: {
			VALIDATE_ARG_COUNT(1);
			VALIDATE_ARG_NUM(0);
			r_ret = Math::log((double)*p_args[0]);
		} break;
		case MATH_EXP: {
			VALIDATE_ARG_COUNT(1);
			VALIDATE_ARG_NUM(0);
			r_ret = Math::exp((double)*p_args[0]);
		} break;
		case MATH_ISNAN: {
			VALIDATE_ARG_COUNT(1);
			VALIDATE_ARG_NUM(0);
			r_ret = Math::is_nan((double)*p_args[0]);
		} break;
		case MATH_ISINF: {
			VALIDATE_ARG_COUNT(1);
			VALIDATE_ARG_NUM(0);
			r_ret = Math::is_inf((double)*p_args[0]);
		} break;
		case MATH_EASE: {
			VALIDATE_ARG_COUNT(2);
			VALIDATE_ARG_NUM(0);
			VALIDATE_ARG_NUM(1);
			r_ret = Math::ease((double)*p_args[0], (double)*p_args[1]);
		} break;
		case MATH_DECIMALS: {
			VALIDATE_ARG_COUNT(1);
			VALIDATE_ARG_NUM(0);
			r_ret = Math::step_decimals((double)*p_args[0]);
		} break;
		case MATH_STEPIFY: {
			VALIDATE_ARG_COUNT(2);
			VALIDATE_ARG_NUM(0);
			VALIDATE_ARG_NUM(1);
			r_ret = Math::stepify((double)*p_args[0], (double)*p_args[1]);
		} break;
		case MATH_LERP: {
			VALIDATE_ARG_COUNT(3);
			VALIDATE_ARG_NUM(0);
			VALIDATE_ARG_NUM(1);
			VALIDATE_ARG_NUM(2);
			r_ret = Math::lerp((double)*p_args[0], (double)*p_args[1], (double)*p_args[2]);
		} break;
		case MATH_DECTIME: {
			VALIDATE_ARG_COUNT(3);
			VALIDATE_ARG_NUM(0);
			VALIDATE_ARG_NUM(1);
			VALIDATE_ARG_NUM(2);
			r_ret = Math::dectime((double)*p_args[0], (double)*p_args[1], (double)*p_args[2]);
		} break;
		case MATH_RANDOMIZE: {
			Math::randomize();
			r_ret = Variant();
		} break;
		case MATH_RAND: {
			r_ret = Math::rand();
		} break;
		case MATH_RANDF: {
			r_ret = Math::randf();
		} break;
		case MATH_RANDOM: {
			VALIDATE_ARG_COUNT(2);
			VALIDATE_ARG_NUM(0);
			VALIDATE_ARG_NUM(1);
			r_ret = Math::random((double)*p_args[0], (double)*p_args[1]);
		} break;
		case MATH_SEED: {
			VALIDATE_ARG_COUNT(1);
			VALIDATE_ARG_NUM(0);
			uint64_t seed = *p_args[0];
			Math::seed(seed);
			r_ret = Variant();
		} break;
		case MATH_RANDSEED: {
			VALIDATE_ARG_COUNT(1);
			VALIDATE_ARG_NUM(0);
			uint64_t seed = *p_args[0];
			int ret = Math::rand_from_seed(&seed);
			Array reta;
			reta.push_back(ret);
			reta.push_back(seed);
			r_ret = reta;

		} break;
		case MATH_DEG2RAD: {
			VALIDATE_ARG_COUNT(1);
			VALIDATE_ARG_NUM(0);
			r_ret = Math::deg2rad((double)*p_args[0]);
		} break;
		case MATH_RAD2DEG: {
			VALIDATE_ARG_COUNT(1);
			VALIDATE_ARG_NUM(0);
			r_ret = Math::rad2deg((double)*p_args[0]);
		} break;
		case MATH_LINEAR2DB: {
			VALIDATE_ARG_COUNT(1);
			VALIDATE_ARG_NUM(0);
			r_ret = Math::linear2db((double)*p_args[0]);
		} break;
		case MATH_DB2LINEAR: {
			VALIDATE_ARG_COUNT(1);
			VALIDATE_ARG_NUM(0);
			r_ret = Math::db2linear((double)*p_args[0]);
		} break;
		case LOGIC_MAX: {
			VALIDATE_ARG_COUNT(2);
			if (p_args[0]->get_type() == Variant::INT && p_args[1]->get_type() == Variant::INT) {

				int64_t a = *p_args[0];
				int64_t b = *p_args[1];
				r_ret = MAX(a, b);
			} else {
				VALIDATE_ARG_NUM(0);
				VALIDATE_ARG_NUM(1);

				real_t a = *p_args[0];
				real_t b = *p_args[1];

				r_ret = MAX(a, b);
			}

		} break;
		case LOGIC_MIN: {
			VALIDATE_ARG_COUNT(2);
			if (p_args[0]->get_type() == Variant::INT && p_args[1]->get_type() == Variant::INT) {

				int64_t a = *p_args[0];
				int64_t b = *p_args[1];
				r_ret = MIN(a, b);
			} else {
				VALIDATE_ARG_NUM(0);
				VALIDATE_ARG_NUM(1);

				real_t a = *p_args[0];
				real_t b = *p_args[1];

				r_ret = MIN(a, b);
			}
		} break;
		case LOGIC_CLAMP: {
			VALIDATE_ARG_COUNT(3);
			if (p_args[0]->get_type() == Variant::INT && p_args[1]->get_type() == Variant::INT && p_args[2]->get_type() == Variant::INT) {

				int64_t a = *p_args[0];
				int64_t b = *p_args[1];
				int64_t c = *p_args[2];
				r_ret = CLAMP(a, b, c);
			} else {
				VALIDATE_ARG_NUM(0);
				VALIDATE_ARG_NUM(1);
				VALIDATE_ARG_NUM(2);

				real_t a = *p_args[0];
				real_t b = *p_args[1];
				real_t c = *p_args[2];

				r_ret = CLAMP(a, b, c);
			}
		} break;
		case LOGIC_NEAREST_PO2: {
			VALIDATE_ARG_COUNT(1);
			VALIDATE_ARG_NUM(0);
			int64_t num = *p_args[0];
			r_ret = nearest_power_of_2(num);
		} break;
		case OBJ_WEAKREF: {
			VALIDATE_ARG_COUNT(1);
			if (p_args[0]->get_type() != Variant::OBJECT) {

				r_error.error = Variant::CallError::CALL_ERROR_INVALID_ARGUMENT;
				r_error.argument = 0;
				r_error.expected = Variant::OBJECT;
				r_ret = Variant();
				return;
			}

			if (p_args[0]->is_ref()) {

				REF r = *p_args[0];
				if (!r.is_valid()) {
					r_ret = Variant();
					return;
				}

				Ref<WeakRef> wref = memnew(WeakRef);
				wref->set_ref(r);
				r_ret = wref;
			} else {
				Object *obj = *p_args[0];
				if (!obj) {
					r_ret = Variant();
					return;
				}
				Ref<WeakRef> wref = memnew(WeakRef);
				wref->set_obj(obj);
				r_ret = wref;
			}

		} break;
		case FUNC_FUNCREF: {
			VALIDATE_ARG_COUNT(2);
			if (p_args[0]->get_type() != Variant::OBJECT) {

				r_error.error = Variant::CallError::CALL_ERROR_INVALID_ARGUMENT;
				r_error.argument = 0;
				r_error.expected = Variant::OBJECT;
				r_ret = Variant();
				return;
			}
			if (p_args[1]->get_type() != Variant::STRING && p_args[1]->get_type() != Variant::NODE_PATH) {

				r_error.error = Variant::CallError::CALL_ERROR_INVALID_ARGUMENT;
				r_error.argument = 1;
				r_error.expected = Variant::STRING;
				r_ret = Variant();
				return;
			}

			Ref<FuncRef> fr = memnew(FuncRef);

			fr->set_instance(*p_args[0]);
			fr->set_function(*p_args[1]);

			r_ret = fr;

		} break;
		case TYPE_CONVERT: {
			VALIDATE_ARG_COUNT(2);
			VALIDATE_ARG_NUM(1);
			int type = *p_args[1];
			if (type < 0 || type >= Variant::VARIANT_MAX) {

				r_ret = RTR("Invalid type argument to convert(), use TYPE_* constants.");
				r_error.error = Variant::CallError::CALL_ERROR_INVALID_ARGUMENT;
				r_error.argument = 0;
				r_error.expected = Variant::INT;
				return;

			} else {

				r_ret = Variant::construct(Variant::Type(type), p_args, 1, r_error);
			}
		} break;
		case TYPE_OF: {

			VALIDATE_ARG_COUNT(1);
			r_ret = p_args[0]->get_type();

		} break;
		case TYPE_EXISTS: {

			VALIDATE_ARG_COUNT(1);
			r_ret = ClassDB::class_exists(*p_args[0]);

		} break;
		case TEXT_CHAR: {
			VALIDATE_ARG_COUNT(1);
			VALIDATE_ARG_NUM(0);
			CharType result[2] = { *p_args[0], 0 };
			r_ret = String(result);
		} break;
		case TEXT_STR: {

			String str;
			for (int i = 0; i < p_arg_count; i++) {

				String os = p_args[i]->operator String();

				if (i == 0)
					str = os;
				else
					str += os;
			}

			r_ret = str;

		} break;
		case TEXT_PRINT: {

			String str;
			for (int i = 0; i < p_arg_count; i++) {

				str += p_args[i]->operator String();
			}

			//str+="\n";
			print_line(str);
			r_ret = Variant();

		} break;
		case TEXT_PRINT_TABBED: {

			String str;
			for (int i = 0; i < p_arg_count; i++) {

				if (i)
					str += "\t";
				str += p_args[i]->operator String();
			}

			//str+="\n";
			print_line(str);
			r_ret = Variant();

		} break;
		case TEXT_PRINT_SPACED: {

			String str;
			for (int i = 0; i < p_arg_count; i++) {

				if (i)
					str += " ";
				str += p_args[i]->operator String();
			}

			//str+="\n";
			print_line(str);
			r_ret = Variant();

		} break;

		case TEXT_PRINTERR: {

			String str;
			for (int i = 0; i < p_arg_count; i++) {

				str += p_args[i]->operator String();
			}

			//str+="\n";
			OS::get_singleton()->printerr("%s\n", str.utf8().get_data());
			r_ret = Variant();

		} break;
		case TEXT_PRINTRAW: {
			String str;
			for (int i = 0; i < p_arg_count; i++) {

				str += p_args[i]->operator String();
			}

			//str+="\n";
			OS::get_singleton()->print("%s", str.utf8().get_data());
			r_ret = Variant();

		} break;
		case VAR_TO_STR: {
			VALIDATE_ARG_COUNT(1);
			String vars;
			VariantWriter::write_to_string(*p_args[0], vars);
			r_ret = vars;
		} break;
		case STR_TO_VAR: {
			VALIDATE_ARG_COUNT(1);
			if (p_args[0]->get_type() != Variant::STRING) {
				r_error.error = Variant::CallError::CALL_ERROR_INVALID_ARGUMENT;
				r_error.argument = 0;
				r_error.expected = Variant::STRING;
				r_ret = Variant();
				return;
			}

			VariantParser::StreamString ss;
			ss.s = *p_args[0];

			String errs;
			int line;
			Error err = VariantParser::parse(&ss, r_ret, errs, line);

			if (err != OK) {
				r_error.error = Variant::CallError::CALL_ERROR_INVALID_ARGUMENT;
				r_error.argument = 0;
				r_error.expected = Variant::STRING;
				r_ret = "Parse error at line " + itos(line) + ": " + errs;
				return;
			}

		} break;
		case VAR_TO_BYTES: {
			VALIDATE_ARG_COUNT(1);

			PoolByteArray barr;
			int len;
			Error err = encode_variant(*p_args[0], NULL, len);
			if (err) {
				r_error.error = Variant::CallError::CALL_ERROR_INVALID_ARGUMENT;
				r_error.argument = 0;
				r_error.expected = Variant::NIL;
				r_ret = "Unexpected error encoding variable to bytes, likely unserializable type found (Object or RID).";
				return;
			}

			barr.resize(len);
			{
				PoolByteArray::Write w = barr.write();
				encode_variant(*p_args[0], w.ptr(), len);
			}
			r_ret = barr;
		} break;
		case BYTES_TO_VAR: {
			VALIDATE_ARG_COUNT(1);
			if (p_args[0]->get_type() != Variant::POOL_BYTE_ARRAY) {
				r_error.error = Variant::CallError::CALL_ERROR_INVALID_ARGUMENT;
				r_error.argument = 0;
				r_error.expected = Variant::POOL_BYTE_ARRAY;
				r_ret = Variant();
				return;
			}

			PoolByteArray varr = *p_args[0];
			Variant ret;
			{
				PoolByteArray::Read r = varr.read();
				Error err = decode_variant(ret, r.ptr(), varr.size(), NULL);
				if (err != OK) {
					r_ret = RTR("Not enough bytes for decoding bytes, or invalid format.");
					r_error.error = Variant::CallError::CALL_ERROR_INVALID_ARGUMENT;
					r_error.argument = 0;
					r_error.expected = Variant::POOL_BYTE_ARRAY;
					return;
				}
			}

			r_ret = ret;

		} break;
		case GEN_RANGE: {

			switch (p_arg_count) {

				case 0: {

					r_error.error = Variant::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS;
					r_error.argument = 1;
					r_ret = Variant();

				} break;
				case 1: {

					VALIDATE_ARG_NUM(0);
					int count = *p_args[0];
					Array arr;
					if (count <= 0) {
						r_ret = arr;
						return;
					}
					Error err = arr.resize(count);
					if (err != OK) {
						r_error.error = Variant::CallError::CALL_ERROR_INVALID_METHOD;
						r_ret = Variant();
						return;
					}

					for (int i = 0; i < count; i++) {
						arr[i] = i;
					}

					r_ret = arr;
				} break;
				case 2: {

					VALIDATE_ARG_NUM(0);
					VALIDATE_ARG_NUM(1);

					int from = *p_args[0];
					int to = *p_args[1];

					Array arr;
					if (from >= to) {
						r_ret = arr;
						return;
					}
					Error err = arr.resize(to - from);
					if (err != OK) {
						r_error.error = Variant::CallError::CALL_ERROR_INVALID_METHOD;
						r_ret = Variant();
						return;
					}
					for (int i = from; i < to; i++)
						arr[i - from] = i;
					r_ret = arr;
				} break;
				case 3: {

					VALIDATE_ARG_NUM(0);
					VALIDATE_ARG_NUM(1);
					VALIDATE_ARG_NUM(2);

					int from = *p_args[0];
					int to = *p_args[1];
					int incr = *p_args[2];
					if (incr == 0) {

						r_ret = RTR("step argument is zero!");
						r_error.error = Variant::CallError::CALL_ERROR_INVALID_METHOD;
						return;
					}

					Array arr;
					if (from >= to && incr > 0) {
						r_ret = arr;
						return;
					}
					if (from <= to && incr < 0) {
						r_ret = arr;
						return;
					}

					//calculate how many
					int count = 0;
					if (incr > 0) {

						count = ((to - from - 1) / incr) + 1;
					} else {

						count = ((from - to - 1) / -incr) + 1;
					}

					Error err = arr.resize(count);

					if (err != OK) {
						r_error.error = Variant::CallError::CALL_ERROR_INVALID_METHOD;
						r_ret = Variant();
						return;
					}

					if (incr > 0) {
						int idx = 0;
						for (int i = from; i < to; i += incr) {
							arr[idx++] = i;
						}
					} else {

						int idx = 0;
						for (int i = from; i > to; i += incr) {
							arr[idx++] = i;
						}
					}

					r_ret = arr;
				} break;
				default: {

					r_error.error = Variant::CallError::CALL_ERROR_TOO_MANY_ARGUMENTS;
					r_error.argument = 3;
					r_ret = Variant();

				} break;
			}

		} break;
		case RESOURCE_LOAD: {
			VALIDATE_ARG_COUNT(1);
			if (p_args[0]->get_type() != Variant::STRING) {
				r_error.error = Variant::CallError::CALL_ERROR_INVALID_ARGUMENT;
				r_error.argument = 0;
				r_error.expected = Variant::STRING;
				r_ret = Variant();
			} else {
				r_ret = ResourceLoader::load(*p_args[0]);
			}

		} break;
		case INST2DICT: {

			VALIDATE_ARG_COUNT(1);

			if (p_args[0]->get_type() == Variant::NIL) {
				r_ret = Variant();
			} else if (p_args[0]->get_type() != Variant::OBJECT) {
				r_error.error = Variant::CallError::CALL_ERROR_INVALID_ARGUMENT;
				r_error.argument = 0;
				r_ret = Variant();
			} else {

				Object *obj = *p_args[0];
				if (!obj) {
					r_ret = Variant();

				} else if (!obj->get_script_instance() || obj->get_script_instance()->get_language() != GDScriptLanguage::get_singleton()) {

					r_error.error = Variant::CallError::CALL_ERROR_INVALID_ARGUMENT;
					r_error.argument = 0;
					r_error.expected = Variant::DICTIONARY;
					r_ret = RTR("Not a script with an instance");
					return;
				} else {

					GDInstance *ins = static_cast<GDInstance *>(obj->get_script_instance());
					Ref<GDScript> base = ins->get_script();
					if (base.is_null()) {

						r_error.error = Variant::CallError::CALL_ERROR_INVALID_ARGUMENT;
						r_error.argument = 0;
						r_error.expected = Variant::DICTIONARY;
						r_ret = RTR("Not based on a script");
						return;
					}

					GDScript *p = base.ptr();
					Vector<StringName> sname;

					while (p->_owner) {

						sname.push_back(p->name);
						p = p->_owner;
					}
					sname.invert();

					if (!p->path.is_resource_file()) {
						r_error.error = Variant::CallError::CALL_ERROR_INVALID_ARGUMENT;
						r_error.argument = 0;
						r_error.expected = Variant::DICTIONARY;
						r_ret = Variant();

						r_ret = RTR("Not based on a resource file");

						return;
					}

					NodePath cp(sname, Vector<StringName>(), false);

					Dictionary d;
					d["@subpath"] = cp;
					d["@path"] = p->path;

					p = base.ptr();

					while (p) {

						for (Set<StringName>::Element *E = p->members.front(); E; E = E->next()) {

							Variant value;
							if (ins->get(E->get(), value)) {

								String k = E->get();
								if (!d.has(k)) {
									d[k] = value;
								}
							}
						}

						p = p->_base;
					}

					r_ret = d;
				}
			}

		} break;
		case DICT2INST: {

			VALIDATE_ARG_COUNT(1);

			if (p_args[0]->get_type() != Variant::DICTIONARY) {

				r_error.error = Variant::CallError::CALL_ERROR_INVALID_ARGUMENT;
				r_error.argument = 0;
				r_error.expected = Variant::DICTIONARY;
				r_ret = Variant();

				return;
			}

			Dictionary d = *p_args[0];

			if (!d.has("@path")) {

				r_error.error = Variant::CallError::CALL_ERROR_INVALID_ARGUMENT;
				r_error.argument = 0;
				r_error.expected = Variant::OBJECT;
				r_ret = RTR("Invalid instance dictionary format (missing @path)");

				return;
			}

			Ref<Script> scr = ResourceLoader::load(d["@path"]);
			if (!scr.is_valid()) {

				r_error.error = Variant::CallError::CALL_ERROR_INVALID_ARGUMENT;
				r_error.argument = 0;
				r_error.expected = Variant::OBJECT;
				r_ret = RTR("Invalid instance dictionary format (can't load script at @path)");
				return;
			}

			Ref<GDScript> gdscr = scr;

			if (!gdscr.is_valid()) {

				r_error.error = Variant::CallError::CALL_ERROR_INVALID_ARGUMENT;
				r_error.argument = 0;
				r_error.expected = Variant::OBJECT;
				r_ret = Variant();
				r_ret = RTR("Invalid instance dictionary format (invalid script at @path)");
				return;
			}

			NodePath sub;
			if (d.has("@subpath")) {
				sub = d["@subpath"];
			}

			for (int i = 0; i < sub.get_name_count(); i++) {

				gdscr = gdscr->subclasses[sub.get_name(i)];
				if (!gdscr.is_valid()) {

					r_error.error = Variant::CallError::CALL_ERROR_INVALID_ARGUMENT;
					r_error.argument = 0;
					r_error.expected = Variant::OBJECT;
					r_ret = Variant();
					r_ret = RTR("Invalid instance dictionary (invalid subclasses)");
					return;
				}
			}

			r_ret = gdscr->_new(NULL, 0, r_error);

			GDInstance *ins = static_cast<GDInstance *>(static_cast<Object *>(r_ret)->get_script_instance());
			Ref<GDScript> gd_ref = ins->get_script();

			for (Map<StringName, GDScript::MemberInfo>::Element *E = gd_ref->member_indices.front(); E; E = E->next()) {
				if (d.has(E->key())) {
					ins->members[E->get().index] = d[E->key()];
				}
			}

		} break;
		case VALIDATE_JSON: {

			VALIDATE_ARG_COUNT(1);

			if (p_args[0]->get_type() != Variant::STRING) {
				r_error.error = Variant::CallError::CALL_ERROR_INVALID_ARGUMENT;
				r_error.argument = 0;
				r_error.expected = Variant::STRING;
				r_ret = Variant();
				return;
			}

			String errs;
			int errl;

			Error err = JSON::parse(*p_args[0], r_ret, errs, errl);

			if (err != OK) {
				r_ret = itos(errl) + ":" + errs;
			} else {
				r_ret = "";
			}

		} break;
		case PARSE_JSON: {

			VALIDATE_ARG_COUNT(1);

			if (p_args[0]->get_type() != Variant::STRING) {
				r_error.error = Variant::CallError::CALL_ERROR_INVALID_ARGUMENT;
				r_error.argument = 0;
				r_error.expected = Variant::STRING;
				r_ret = Variant();
				return;
			}

			String errs;
			int errl;

			Error err = JSON::parse(*p_args[0], r_ret, errs, errl);

			if (err != OK) {
				r_ret = Variant();
			}

		} break;
		case TO_JSON: {
			VALIDATE_ARG_COUNT(1);

			r_ret = JSON::print(*p_args[0]);
		} break;
		case HASH: {

			VALIDATE_ARG_COUNT(1);
			r_ret = p_args[0]->hash();

		} break;
		case COLOR8: {

			if (p_arg_count < 3) {
				r_error.error = Variant::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS;
				r_error.argument = 3;
				r_ret = Variant();

				return;
			}
			if (p_arg_count > 4) {
				r_error.error = Variant::CallError::CALL_ERROR_TOO_MANY_ARGUMENTS;
				r_error.argument = 4;
				r_ret = Variant();

				return;
			}

			VALIDATE_ARG_NUM(0);
			VALIDATE_ARG_NUM(1);
			VALIDATE_ARG_NUM(2);

			Color color((float)*p_args[0] / 255.0f, (float)*p_args[1] / 255.0f, (float)*p_args[2] / 255.0f);

			if (p_arg_count == 4) {
				VALIDATE_ARG_NUM(3);
				color.a = (float)*p_args[3] / 255.0f;
			}

			r_ret = color;

		} break;
		case COLORN: {

			if (p_arg_count < 1) {
				r_error.error = Variant::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS;
				r_error.argument = 1;
				r_ret = Variant();
				return;
			}

			if (p_arg_count > 2) {
				r_error.error = Variant::CallError::CALL_ERROR_TOO_MANY_ARGUMENTS;
				r_error.argument = 2;
				r_ret = Variant();
				return;
			}

			if (p_args[0]->get_type() != Variant::STRING) {
				r_error.error = Variant::CallError::CALL_ERROR_INVALID_ARGUMENT;
				r_error.argument = 0;
				r_ret = Variant();
			} else {
				Color color = Color::named(*p_args[0]);
				if (p_arg_count == 2) {
					VALIDATE_ARG_NUM(1);
					color.a = *p_args[1];
				}
				r_ret = color;
			}

		} break;

		case PRINT_STACK: {

			ScriptLanguage *script = GDScriptLanguage::get_singleton();
			for (int i = 0; i < script->debug_get_stack_level_count(); i++) {

				print_line("Frame " + itos(i) + " - " + script->debug_get_stack_level_source(i) + ":" + itos(script->debug_get_stack_level_line(i)) + " in function '" + script->debug_get_stack_level_function(i) + "'");
			};
		} break;

		case INSTANCE_FROM_ID: {

			VALIDATE_ARG_COUNT(1);
			if (p_args[0]->get_type() != Variant::INT && p_args[0]->get_type() != Variant::REAL) {
				r_error.error = Variant::CallError::CALL_ERROR_INVALID_ARGUMENT;
				r_error.argument = 0;
				r_error.expected = Variant::INT;
				r_ret = Variant();
				break;
			}

			uint32_t id = *p_args[0];
			r_ret = ObjectDB::get_instance(id);

		} break;
		case FUNC_MAX: {

			ERR_FAIL();
		} break;
	}
}

bool GDFunctions::is_deterministic(Function p_func) {

	//man i couldn't have chosen a worse function name,
	//way too controversial..

	switch (p_func) {

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
		case MATH_CEIL:
		case MATH_ROUND:
		case MATH_ABS:
		case MATH_SIGN:
		case MATH_POW:
		case MATH_LOG:
		case MATH_EXP:
		case MATH_ISNAN:
		case MATH_ISINF:
		case MATH_EASE:
		case MATH_DECIMALS:
		case MATH_STEPIFY:
		case MATH_LERP:
		case MATH_DECTIME:
		case MATH_DEG2RAD:
		case MATH_RAD2DEG:
		case MATH_LINEAR2DB:
		case MATH_DB2LINEAR:
		case LOGIC_MAX:
		case LOGIC_MIN:
		case LOGIC_CLAMP:
		case LOGIC_NEAREST_PO2:
		case TYPE_CONVERT:
		case TYPE_OF:
		case TYPE_EXISTS:
		case TEXT_CHAR:
		case TEXT_STR:
		case COLOR8:
			// enable for debug only, otherwise not desirable - case GEN_RANGE:
			return true;
		default:
			return false;
	}

	return false;
}

MethodInfo GDFunctions::get_info(Function p_func) {

#ifdef TOOLS_ENABLED
	//using a switch, so the compiler generates a jumptable

	switch (p_func) {

		case MATH_SIN: {
			MethodInfo mi("sin", PropertyInfo(Variant::REAL, "s"));
			mi.return_val.type = Variant::REAL;
			return mi;

		} break;
		case MATH_COS: {
			MethodInfo mi("cos", PropertyInfo(Variant::REAL, "s"));
			mi.return_val.type = Variant::REAL;
			return mi;
		} break;
		case MATH_TAN: {
			MethodInfo mi("tan", PropertyInfo(Variant::REAL, "s"));
			mi.return_val.type = Variant::REAL;
			return mi;
		} break;
		case MATH_SINH: {
			MethodInfo mi("sinh", PropertyInfo(Variant::REAL, "s"));
			mi.return_val.type = Variant::REAL;
			return mi;
		} break;
		case MATH_COSH: {
			MethodInfo mi("cosh", PropertyInfo(Variant::REAL, "s"));
			mi.return_val.type = Variant::REAL;
			return mi;
		} break;
		case MATH_TANH: {
			MethodInfo mi("tanh", PropertyInfo(Variant::REAL, "s"));
			mi.return_val.type = Variant::REAL;
			return mi;
		} break;
		case MATH_ASIN: {
			MethodInfo mi("asin", PropertyInfo(Variant::REAL, "s"));
			mi.return_val.type = Variant::REAL;
			return mi;
		} break;
		case MATH_ACOS: {
			MethodInfo mi("acos", PropertyInfo(Variant::REAL, "s"));
			mi.return_val.type = Variant::REAL;
			return mi;
		} break;
		case MATH_ATAN: {
			MethodInfo mi("atan", PropertyInfo(Variant::REAL, "s"));
			mi.return_val.type = Variant::REAL;
			return mi;
		} break;
		case MATH_ATAN2: {
			MethodInfo mi("atan2", PropertyInfo(Variant::REAL, "x"), PropertyInfo(Variant::REAL, "y"));
			mi.return_val.type = Variant::REAL;
			return mi;
		} break;
		case MATH_SQRT: {
			MethodInfo mi("sqrt", PropertyInfo(Variant::REAL, "s"));
			mi.return_val.type = Variant::REAL;
			return mi;
		} break;
		case MATH_FMOD: {
			MethodInfo mi("fmod", PropertyInfo(Variant::REAL, "x"), PropertyInfo(Variant::REAL, "y"));
			mi.return_val.type = Variant::REAL;
			return mi;
		} break;
		case MATH_FPOSMOD: {
			MethodInfo mi("fposmod", PropertyInfo(Variant::REAL, "x"), PropertyInfo(Variant::REAL, "y"));
			mi.return_val.type = Variant::REAL;
			return mi;
		} break;
		case MATH_FLOOR: {
			MethodInfo mi("floor", PropertyInfo(Variant::REAL, "s"));
			mi.return_val.type = Variant::REAL;
			return mi;
		} break;
		case MATH_CEIL: {
			MethodInfo mi("ceil", PropertyInfo(Variant::REAL, "s"));
			mi.return_val.type = Variant::REAL;
			return mi;
		} break;
		case MATH_ROUND: {
			MethodInfo mi("round", PropertyInfo(Variant::REAL, "s"));
			mi.return_val.type = Variant::REAL;
			return mi;
		} break;
		case MATH_ABS: {
			MethodInfo mi("abs", PropertyInfo(Variant::REAL, "s"));
			mi.return_val.type = Variant::REAL;
			return mi;
		} break;
		case MATH_SIGN: {
			MethodInfo mi("sign", PropertyInfo(Variant::REAL, "s"));
			mi.return_val.type = Variant::REAL;
			return mi;
		} break;
		case MATH_POW: {
			MethodInfo mi("pow", PropertyInfo(Variant::REAL, "x"), PropertyInfo(Variant::REAL, "y"));
			mi.return_val.type = Variant::REAL;
			return mi;
		} break;
		case MATH_LOG: {
			MethodInfo mi("log", PropertyInfo(Variant::REAL, "s"));
			mi.return_val.type = Variant::REAL;
			return mi;
		} break;
		case MATH_EXP: {
			MethodInfo mi("exp", PropertyInfo(Variant::REAL, "s"));
			mi.return_val.type = Variant::REAL;
			return mi;
		} break;
		case MATH_ISNAN: {
			MethodInfo mi("is_nan", PropertyInfo(Variant::REAL, "s"));
			mi.return_val.type = Variant::REAL;
			return mi;
		} break;
		case MATH_ISINF: {
			MethodInfo mi("is_inf", PropertyInfo(Variant::REAL, "s"));
			mi.return_val.type = Variant::REAL;
			return mi;
		} break;
		case MATH_EASE: {
			MethodInfo mi("ease", PropertyInfo(Variant::REAL, "s"), PropertyInfo(Variant::REAL, "curve"));
			mi.return_val.type = Variant::REAL;
			return mi;
		} break;
		case MATH_DECIMALS: {
			MethodInfo mi("decimals", PropertyInfo(Variant::REAL, "step"));
			mi.return_val.type = Variant::REAL;
			return mi;
		} break;
		case MATH_STEPIFY: {
			MethodInfo mi("stepify", PropertyInfo(Variant::REAL, "s"), PropertyInfo(Variant::REAL, "step"));
			mi.return_val.type = Variant::REAL;
			return mi;
		} break;
		case MATH_LERP: {
			MethodInfo mi("lerp", PropertyInfo(Variant::REAL, "from"), PropertyInfo(Variant::REAL, "to"), PropertyInfo(Variant::REAL, "weight"));
			mi.return_val.type = Variant::REAL;
			return mi;
		} break;
		case MATH_DECTIME: {
			MethodInfo mi("dectime", PropertyInfo(Variant::REAL, "value"), PropertyInfo(Variant::REAL, "amount"), PropertyInfo(Variant::REAL, "step"));
			mi.return_val.type = Variant::REAL;
			return mi;
		} break;
		case MATH_RANDOMIZE: {
			MethodInfo mi("randomize");
			mi.return_val.type = Variant::NIL;
			return mi;
		} break;
		case MATH_RAND: {
			MethodInfo mi("randi");
			mi.return_val.type = Variant::INT;
			return mi;
		} break;
		case MATH_RANDF: {
			MethodInfo mi("randf");
			mi.return_val.type = Variant::REAL;
			return mi;
		} break;
		case MATH_RANDOM: {
			MethodInfo mi("rand_range", PropertyInfo(Variant::REAL, "from"), PropertyInfo(Variant::REAL, "to"));
			mi.return_val.type = Variant::REAL;
			return mi;
		} break;
		case MATH_SEED: {
			MethodInfo mi("seed", PropertyInfo(Variant::INT, "seed"));
			mi.return_val.type = Variant::NIL;
			return mi;
		} break;
		case MATH_RANDSEED: {
			MethodInfo mi("rand_seed", PropertyInfo(Variant::INT, "seed"));
			mi.return_val.type = Variant::ARRAY;
			return mi;
		} break;
		case MATH_DEG2RAD: {
			MethodInfo mi("deg2rad", PropertyInfo(Variant::REAL, "deg"));
			mi.return_val.type = Variant::REAL;
			return mi;
		} break;
		case MATH_RAD2DEG: {
			MethodInfo mi("rad2deg", PropertyInfo(Variant::REAL, "rad"));
			mi.return_val.type = Variant::REAL;
			return mi;
		} break;
		case MATH_LINEAR2DB: {
			MethodInfo mi("linear2db", PropertyInfo(Variant::REAL, "nrg"));
			mi.return_val.type = Variant::REAL;
			return mi;
		} break;
		case MATH_DB2LINEAR: {
			MethodInfo mi("db2linear", PropertyInfo(Variant::REAL, "db"));
			mi.return_val.type = Variant::REAL;
			return mi;
		} break;
		case LOGIC_MAX: {
			MethodInfo mi("max", PropertyInfo(Variant::REAL, "a"), PropertyInfo(Variant::REAL, "b"));
			mi.return_val.type = Variant::REAL;
			return mi;

		} break;
		case LOGIC_MIN: {
			MethodInfo mi("min", PropertyInfo(Variant::REAL, "a"), PropertyInfo(Variant::REAL, "b"));
			mi.return_val.type = Variant::REAL;
			return mi;
		} break;
		case LOGIC_CLAMP: {
			MethodInfo mi("clamp", PropertyInfo(Variant::REAL, "val"), PropertyInfo(Variant::REAL, "min"), PropertyInfo(Variant::REAL, "max"));
			mi.return_val.type = Variant::REAL;
			return mi;
		} break;
		case LOGIC_NEAREST_PO2: {
			MethodInfo mi("nearest_po2", PropertyInfo(Variant::INT, "val"));
			mi.return_val.type = Variant::INT;
			return mi;
		} break;
		case OBJ_WEAKREF: {

			MethodInfo mi("weakref", PropertyInfo(Variant::OBJECT, "obj"));
			mi.return_val.type = Variant::OBJECT;
			mi.return_val.name = "WeakRef";

			return mi;

		} break;
		case FUNC_FUNCREF: {

			MethodInfo mi("funcref", PropertyInfo(Variant::OBJECT, "instance"), PropertyInfo(Variant::STRING, "funcname"));
			mi.return_val.type = Variant::OBJECT;
			mi.return_val.name = "FuncRef";
			return mi;

		} break;
		case TYPE_CONVERT: {

			MethodInfo mi("convert", PropertyInfo(Variant::NIL, "what"), PropertyInfo(Variant::INT, "type"));
			mi.return_val.type = Variant::OBJECT;
			return mi;
		} break;
		case TYPE_OF: {

			MethodInfo mi("typeof", PropertyInfo(Variant::NIL, "what"));
			mi.return_val.type = Variant::INT;
			return mi;

		} break;
		case TYPE_EXISTS: {

			MethodInfo mi("type_exists", PropertyInfo(Variant::STRING, "type"));
			mi.return_val.type = Variant::BOOL;
			return mi;

		} break;
		case TEXT_CHAR: {

			MethodInfo mi("char", PropertyInfo(Variant::INT, "ascii"));
			mi.return_val.type = Variant::STRING;
			return mi;

		} break;
		case TEXT_STR: {

			MethodInfo mi("str", PropertyInfo(Variant::NIL, "what"), PropertyInfo(Variant::NIL, "..."));
			mi.return_val.type = Variant::STRING;
			return mi;

		} break;
		case TEXT_PRINT: {

			MethodInfo mi("print", PropertyInfo(Variant::NIL, "what"), PropertyInfo(Variant::NIL, "..."));
			mi.return_val.type = Variant::NIL;
			return mi;

		} break;
		case TEXT_PRINT_TABBED: {

			MethodInfo mi("printt", PropertyInfo(Variant::NIL, "what"), PropertyInfo(Variant::NIL, "..."));
			mi.return_val.type = Variant::NIL;
			return mi;

		} break;
		case TEXT_PRINT_SPACED: {

			MethodInfo mi("prints", PropertyInfo(Variant::NIL, "what"), PropertyInfo(Variant::NIL, "..."));
			mi.return_val.type = Variant::NIL;
			return mi;

		} break;
		case TEXT_PRINTERR: {

			MethodInfo mi("printerr", PropertyInfo(Variant::NIL, "what"), PropertyInfo(Variant::NIL, "..."));
			mi.return_val.type = Variant::NIL;
			return mi;

		} break;
		case TEXT_PRINTRAW: {

			MethodInfo mi("printraw", PropertyInfo(Variant::NIL, "what"), PropertyInfo(Variant::NIL, "..."));
			mi.return_val.type = Variant::NIL;
			return mi;

		} break;
		case VAR_TO_STR: {
			MethodInfo mi("var2str", PropertyInfo(Variant::NIL, "var"));
			mi.return_val.type = Variant::STRING;
			return mi;

		} break;
		case STR_TO_VAR: {

			MethodInfo mi("str2var:Variant", PropertyInfo(Variant::STRING, "string"));
			mi.return_val.type = Variant::NIL;
			return mi;
		} break;
		case VAR_TO_BYTES: {
			MethodInfo mi("var2bytes", PropertyInfo(Variant::NIL, "var"));
			mi.return_val.type = Variant::POOL_BYTE_ARRAY;
			return mi;

		} break;
		case BYTES_TO_VAR: {

			MethodInfo mi("bytes2var:Variant", PropertyInfo(Variant::POOL_BYTE_ARRAY, "bytes"));
			mi.return_val.type = Variant::NIL;
			return mi;
		} break;
		case GEN_RANGE: {

			MethodInfo mi("range", PropertyInfo(Variant::NIL, "..."));
			mi.return_val.type = Variant::ARRAY;
			return mi;
		} break;
		case RESOURCE_LOAD: {

			MethodInfo mi("load", PropertyInfo(Variant::STRING, "path"));
			mi.return_val.type = Variant::OBJECT;
			mi.return_val.name = "Resource";
			return mi;
		} break;
		case INST2DICT: {

			MethodInfo mi("inst2dict", PropertyInfo(Variant::OBJECT, "inst"));
			mi.return_val.type = Variant::DICTIONARY;
			return mi;
		} break;
		case DICT2INST: {

			MethodInfo mi("dict2inst", PropertyInfo(Variant::DICTIONARY, "dict"));
			mi.return_val.type = Variant::OBJECT;
			return mi;
		} break;
		case VALIDATE_JSON: {

			MethodInfo mi("validate_json:Variant", PropertyInfo(Variant::STRING, "json"));
			mi.return_val.type = Variant::STRING;
			return mi;
		} break;
		case PARSE_JSON: {

			MethodInfo mi("parse_json:Variant", PropertyInfo(Variant::STRING, "json"));
			mi.return_val.type = Variant::NIL;
			return mi;
		} break;
		case TO_JSON: {

			MethodInfo mi("to_json", PropertyInfo(Variant::NIL, "var:Variant"));
			mi.return_val.type = Variant::STRING;
			return mi;
		} break;
		case HASH: {

			MethodInfo mi("hash", PropertyInfo(Variant::NIL, "var:Variant"));
			mi.return_val.type = Variant::INT;
			return mi;
		} break;
		case COLOR8: {

			MethodInfo mi("Color8", PropertyInfo(Variant::INT, "r8"), PropertyInfo(Variant::INT, "g8"), PropertyInfo(Variant::INT, "b8"), PropertyInfo(Variant::INT, "a8"));
			mi.return_val.type = Variant::COLOR;
			return mi;
		} break;
		case COLORN: {

			MethodInfo mi("ColorN", PropertyInfo(Variant::STRING, "name"), PropertyInfo(Variant::REAL, "alpha"));
			mi.return_val.type = Variant::COLOR;
			return mi;
		} break;

		case PRINT_STACK: {
			MethodInfo mi("print_stack");
			mi.return_val.type = Variant::NIL;
			return mi;
		} break;

		case INSTANCE_FROM_ID: {
			MethodInfo mi("instance_from_id", PropertyInfo(Variant::INT, "instance_id"));
			mi.return_val.type = Variant::OBJECT;
			return mi;
		} break;

		case FUNC_MAX: {

			ERR_FAIL_V(MethodInfo());
		} break;
	}
#endif

	return MethodInfo();
}
