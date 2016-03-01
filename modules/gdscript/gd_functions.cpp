/*************************************************************************/
/*  gd_functions.cpp                                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2016 Juan Linietsky, Ariel Manzur.                 */
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
#include "reference.h"
#include "func_ref.h"

const char *GDFunctions::get_func_name(Function p_func) {

	ERR_FAIL_INDEX_V(p_func,FUNC_MAX,"");

	static const char *_names[FUNC_MAX]={
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
		"hash",
		"Color8",
		"print_stack",
		"instance_from_id",
	};

	return _names[p_func];
}

void GDFunctions::call
(
	Function p_func,
	const Variant **p_args,
	int p_arg_count,
	Variant &r_ret,
	Variant::CallError &r_error
) {

	r_error.error=Variant::CallError::CALL_OK;

	// Let's say "screw case switch with a kiwi" and use const function pointers instead
	// With inlining, the jump table is as assured as with a switch case.
	void (*const function_ptrs[])(int, const Variant**, Variant&, Variant::CallError&) = {
		gd_math_sin,
		gd_math_cos,
		gd_math_tan,
		gd_math_sinh,
		gd_math_cosh,
		gd_math_tanh,
		gd_math_asin,
		gd_math_acos,
		gd_math_atan,
		gd_math_atan2,
		gd_math_sqrt,
		gd_math_fmod,
		gd_math_fposmod,
		gd_math_floor,
		gd_math_ceil,
		gd_math_round,
		gd_math_abs,
		gd_math_sign,
		gd_math_pow,
		gd_math_log,
		gd_math_exp,
		gd_math_is_nan,
		gd_math_is_inf,
		gd_math_ease,
		gd_math_decimals,
		gd_math_stepify,
		gd_math_lerp,
		gd_math_dectime,
		gd_math_randomize,
		gd_math_rand,
		gd_math_randf,
		gd_math_random,
		gd_math_seed,
		gd_math_randseed,
		gd_math_deg2rad,
		gd_math_rad2deg,
		gd_math_linear2db,
		gd_math_db2linear,
		gd_logic_max,
		gd_logic_min,
		gd_logic_clamp,
		gd_logic_nearest_po2,
		gd_obj_weakref,
		gd_func_funcref,
		gd_type_convert,
		gd_type_of,
		gd_text_str,
		gd_text_print,
		gd_text_print_tabbed,
		gd_text_print_spaced,
		gd_text_printerr,
		gd_text_printraw,
		gd_var_to_str,
		gd_str_to_var,
		gd_var_to_bytes,
		gd_bytes_to_var,
		gd_gen_range,
		gd_gen_xrange,
		gd_resource_load,
		gd_inst2dict,
		gd_dict2inst,
		gd_hash,
		gd_color8,
		gd_print_stack,
		gd_instance_from_id
	};

	if (MATH_SIN <= p_func && p_func < FUNC_MAX) {
		function_ptrs[p_func](p_arg_count, p_args, r_ret, r_error);
	} else {
		ERR_FAIL_V();
	}
}

bool GDFunctions::is_deterministic(Function p_func) {

	//man i couldn't have chosen a worse function name,
	//way too controversial..

	return (MATH_SIN <= p_func && p_func < MATH_RANDOMIZE) ||
		(MATH_DEG2RAD <= p_func && p_func < OBJ_WEAKREF)   ||
		(TYPE_CONVERT <= p_func && p_func < TEXT_PRINT)    ||
		//p_func == GEN_RANGE                                ||
		p_func == COLOR8;
}

MethodInfo GDFunctions::get_info(Function p_func) {

#ifdef TOOLS_ENABLED
	//using a switch, so the compiler generates a jumptable

	switch(p_func) {

		case MATH_SIN: {
			MethodInfo mi("sin",PropertyInfo(Variant::REAL,"s"));
			mi.return_val.type=Variant::REAL;
			return mi;

		} break;
		case MATH_COS: {
			MethodInfo mi("cos",PropertyInfo(Variant::REAL,"s"));
			mi.return_val.type=Variant::REAL;
			return mi;
		} break;
		case MATH_TAN: {
			MethodInfo mi("tan",PropertyInfo(Variant::REAL,"s"));
			mi.return_val.type=Variant::REAL;
			return mi;
		} break;
		case MATH_SINH: {
			MethodInfo mi("sinh",PropertyInfo(Variant::REAL,"s"));
			mi.return_val.type=Variant::REAL;
			return mi;
		} break;
		case MATH_COSH: {
			MethodInfo mi("cosh",PropertyInfo(Variant::REAL,"s"));
			mi.return_val.type=Variant::REAL;
			return mi;
		} break;
		case MATH_TANH: {
			MethodInfo mi("tanh",PropertyInfo(Variant::REAL,"s"));
			mi.return_val.type=Variant::REAL;
			return mi;
		} break;
		case MATH_ASIN: {
			MethodInfo mi("asin",PropertyInfo(Variant::REAL,"s"));
			mi.return_val.type=Variant::REAL;
			return mi;
		} break;
		case MATH_ACOS: {
			MethodInfo mi("acos",PropertyInfo(Variant::REAL,"s"));
			mi.return_val.type=Variant::REAL;
			return mi;
		} break;
		case MATH_ATAN: {
			MethodInfo mi("atan",PropertyInfo(Variant::REAL,"s"));
			mi.return_val.type=Variant::REAL;
			return mi;
		} break;
		case MATH_ATAN2: {
			MethodInfo mi("atan2",PropertyInfo(Variant::REAL,"x"),PropertyInfo(Variant::REAL,"y"));
			mi.return_val.type=Variant::REAL;
			return mi;
		} break;
		case MATH_SQRT: {
			MethodInfo mi("sqrt",PropertyInfo(Variant::REAL,"s"));
			mi.return_val.type=Variant::REAL;
			return mi;
		} break;
		case MATH_FMOD: {
			MethodInfo mi("fmod",PropertyInfo(Variant::REAL,"x"),PropertyInfo(Variant::REAL,"y"));
			mi.return_val.type=Variant::REAL;
			return mi;
		} break;
		case MATH_FPOSMOD: {
			MethodInfo mi("fposmod",PropertyInfo(Variant::REAL,"x"),PropertyInfo(Variant::REAL,"y"));
			mi.return_val.type=Variant::REAL;
			return mi;
		} break;
		case MATH_FLOOR: {
			MethodInfo mi("floor",PropertyInfo(Variant::REAL,"s"));
			mi.return_val.type=Variant::REAL;
			return mi;
		  } break;
		case MATH_CEIL: {
			MethodInfo mi("ceil",PropertyInfo(Variant::REAL,"s"));
			mi.return_val.type=Variant::REAL;
			return mi;
		} break;
		case MATH_ROUND: {
			MethodInfo mi("round",PropertyInfo(Variant::REAL,"s"));
			mi.return_val.type=Variant::REAL;
			return mi;
		} break;
		case MATH_ABS: {
			MethodInfo mi("abs",PropertyInfo(Variant::REAL,"s"));
			mi.return_val.type=Variant::REAL;
			return mi;
		} break;
		case MATH_SIGN: {
			MethodInfo mi("sign",PropertyInfo(Variant::REAL,"s"));
			mi.return_val.type=Variant::REAL;
			return mi;
		} break;
		case MATH_POW: {
			MethodInfo mi("pow",PropertyInfo(Variant::REAL,"x"),PropertyInfo(Variant::REAL,"y"));
			mi.return_val.type=Variant::REAL;
			return mi;
		} break;
		case MATH_LOG: {
			MethodInfo mi("log",PropertyInfo(Variant::REAL,"s"));
			mi.return_val.type=Variant::REAL;
			return mi;
		} break;
		case MATH_EXP: {
			MethodInfo mi("exp",PropertyInfo(Variant::REAL,"s"));
			mi.return_val.type=Variant::REAL;
			return mi;
		} break;
		case MATH_ISNAN: {
			MethodInfo mi("isnan",PropertyInfo(Variant::REAL,"s"));
			mi.return_val.type=Variant::REAL;
			return mi;
		} break;
		case MATH_ISINF: {
			MethodInfo mi("isinf",PropertyInfo(Variant::REAL,"s"));
			mi.return_val.type=Variant::REAL;
			return mi;
		} break;
		case MATH_EASE: {
			MethodInfo mi("ease",PropertyInfo(Variant::REAL,"s"),PropertyInfo(Variant::REAL,"curve"));
			mi.return_val.type=Variant::REAL;
			return mi;
		} break;
		case MATH_DECIMALS: {
			MethodInfo mi("decimals",PropertyInfo(Variant::REAL,"step"));
			mi.return_val.type=Variant::REAL;
			return mi;
		} break;
		case MATH_STEPIFY: {
			MethodInfo mi("stepify",PropertyInfo(Variant::REAL,"s"),PropertyInfo(Variant::REAL,"step"));
			mi.return_val.type=Variant::REAL;
			return mi;
		} break;
		case MATH_LERP: {
			MethodInfo mi("lerp",PropertyInfo(Variant::REAL,"from"),PropertyInfo(Variant::REAL,"to"), PropertyInfo(Variant::REAL,"weight"));
			mi.return_val.type=Variant::REAL;
			return mi;
		} break;
		case MATH_DECTIME: {
			MethodInfo mi("dectime",PropertyInfo(Variant::REAL,"value"),PropertyInfo(Variant::REAL,"amount"),PropertyInfo(Variant::REAL,"step"));
			mi.return_val.type=Variant::REAL;
			return mi;
		} break;
		case MATH_RANDOMIZE: {
			MethodInfo mi("randomize");
			mi.return_val.type=Variant::NIL;
			return mi;
		} break;
		case MATH_RAND: {
			MethodInfo mi("randi");
			mi.return_val.type=Variant::INT;
			return mi;
		} break;
		case MATH_RANDF: {
			MethodInfo mi("randf");
			mi.return_val.type=Variant::REAL;
			return mi;
		} break;
		case MATH_RANDOM: {
			MethodInfo mi("rand_range",PropertyInfo(Variant::REAL,"from"),PropertyInfo(Variant::REAL,"to"));
			mi.return_val.type=Variant::REAL;
			return mi;
		} break;
		case MATH_SEED: {
			MethodInfo mi("seed",PropertyInfo(Variant::REAL,"seed"));
			mi.return_val.type=Variant::NIL;
			return mi;
		} break;
		case MATH_RANDSEED: {
			MethodInfo mi("rand_seed",PropertyInfo(Variant::REAL,"seed"));
			mi.return_val.type=Variant::ARRAY;
			return mi;
		} break;
		case MATH_DEG2RAD: {
			MethodInfo mi("deg2rad",PropertyInfo(Variant::REAL,"deg"));
			mi.return_val.type=Variant::REAL;
			return mi;
		} break;
		case MATH_RAD2DEG: {
			MethodInfo mi("rad2deg",PropertyInfo(Variant::REAL,"rad"));
			mi.return_val.type=Variant::REAL;
			return mi;
		} break;
		case MATH_LINEAR2DB: {
			MethodInfo mi("linear2db",PropertyInfo(Variant::REAL,"nrg"));
			mi.return_val.type=Variant::REAL;
			return mi;
		} break;
		case MATH_DB2LINEAR: {
			MethodInfo mi("db2linear",PropertyInfo(Variant::REAL,"db"));
			mi.return_val.type=Variant::REAL;
			return mi;
		} break;
		case LOGIC_MAX: {
			MethodInfo mi("max",PropertyInfo(Variant::REAL,"a"),PropertyInfo(Variant::REAL,"b"));
			mi.return_val.type=Variant::REAL;
			return mi;

		} break;
		case LOGIC_MIN: {
			MethodInfo mi("min",PropertyInfo(Variant::REAL,"a"),PropertyInfo(Variant::REAL,"b"));
			mi.return_val.type=Variant::REAL;
			return mi;
		} break;
		case LOGIC_CLAMP: {
			MethodInfo mi("clamp",PropertyInfo(Variant::REAL,"val"),PropertyInfo(Variant::REAL,"min"),PropertyInfo(Variant::REAL,"max"));
			mi.return_val.type=Variant::REAL;
			return mi;
		} break;
		case LOGIC_NEAREST_PO2: {
			MethodInfo mi("nearest_po2",PropertyInfo(Variant::INT,"val"));
			mi.return_val.type=Variant::INT;
			return mi;
		} break;
		case OBJ_WEAKREF: {

			MethodInfo mi("weakref",PropertyInfo(Variant::OBJECT,"obj"));
			mi.return_val.type=Variant::OBJECT;
			mi.return_val.name="WeakRef";

			return mi;

		} break;
		case FUNC_FUNCREF: {

			MethodInfo mi("funcref",PropertyInfo(Variant::OBJECT,"instance"),PropertyInfo(Variant::STRING,"funcname"));
			mi.return_val.type=Variant::OBJECT;
			mi.return_val.name="FuncRef";
			return mi;

		} break;
		case TYPE_CONVERT: {

			MethodInfo mi("convert",PropertyInfo(Variant::NIL,"what"),PropertyInfo(Variant::INT,"type"));
			mi.return_val.type=Variant::OBJECT;
			return mi;
		} break;
		case TYPE_OF: {

			MethodInfo mi("typeof",PropertyInfo(Variant::NIL,"what"));
			mi.return_val.type=Variant::INT;
			return mi;

		} break;
		case TEXT_STR: {

			MethodInfo mi("str",PropertyInfo(Variant::NIL,"what"),PropertyInfo(Variant::NIL,"..."));
			mi.return_val.type=Variant::STRING;
			return mi;

		} break;
		case TEXT_PRINT: {

			MethodInfo mi("print",PropertyInfo(Variant::NIL,"what"),PropertyInfo(Variant::NIL,"..."));
			mi.return_val.type=Variant::NIL;
			return mi;

		} break;
		case TEXT_PRINT_TABBED: {

			MethodInfo mi("printt",PropertyInfo(Variant::NIL,"what"),PropertyInfo(Variant::NIL,"..."));
			mi.return_val.type=Variant::NIL;
			return mi;

		} break;
		case TEXT_PRINT_SPACED: {

			MethodInfo mi("prints",PropertyInfo(Variant::NIL,"what"),PropertyInfo(Variant::NIL,"..."));
			mi.return_val.type=Variant::NIL;
			return mi;

		} break;
		case TEXT_PRINTERR: {

			MethodInfo mi("printerr",PropertyInfo(Variant::NIL,"what"),PropertyInfo(Variant::NIL,"..."));
			mi.return_val.type=Variant::NIL;
			return mi;

		} break;
		case TEXT_PRINTRAW: {

			MethodInfo mi("printraw",PropertyInfo(Variant::NIL,"what"),PropertyInfo(Variant::NIL,"..."));
			mi.return_val.type=Variant::NIL;
			return mi;

		} break;
		case VAR_TO_STR: {
			MethodInfo mi("var2str",PropertyInfo(Variant::NIL,"var"));
			mi.return_val.type=Variant::STRING;
			return mi;

		} break;
		case STR_TO_VAR: {

			MethodInfo mi("str2var:Variant",PropertyInfo(Variant::STRING,"string"));
			mi.return_val.type=Variant::NIL;
			return mi;
		} break;
		case VAR_TO_BYTES: {
			MethodInfo mi("var2bytes",PropertyInfo(Variant::NIL,"var"));
			mi.return_val.type=Variant::RAW_ARRAY;
			return mi;

		} break;
		case BYTES_TO_VAR: {

			MethodInfo mi("bytes2var:Variant",PropertyInfo(Variant::RAW_ARRAY,"bytes"));
			mi.return_val.type=Variant::NIL;
			return mi;
		} break;
		case GEN_RANGE: {

			MethodInfo mi("range",PropertyInfo(Variant::NIL,"..."));
			mi.return_val.type=Variant::ARRAY;
			return mi;
		} break;
		case RESOURCE_LOAD: {

			MethodInfo mi("load",PropertyInfo(Variant::STRING,"path"));
			mi.return_val.type=Variant::OBJECT;
			mi.return_val.name="Resource";
			return mi;
		} break;
		case INST2DICT: {

			MethodInfo mi("inst2dict",PropertyInfo(Variant::OBJECT,"inst"));
			mi.return_val.type=Variant::DICTIONARY;
			return mi;
		} break;
		case DICT2INST: {

			MethodInfo mi("dict2inst",PropertyInfo(Variant::DICTIONARY,"dict"));
			mi.return_val.type=Variant::OBJECT;
			return mi;
		} break;
		case HASH: {

			MethodInfo mi("hash",PropertyInfo(Variant::NIL,"var:Variant"));
			mi.return_val.type=Variant::INT;
			return mi;
		} break;
		case COLOR8: {

			MethodInfo mi("Color8",PropertyInfo(Variant::INT,"r8"),PropertyInfo(Variant::INT,"g8"),PropertyInfo(Variant::INT,"b8"),PropertyInfo(Variant::INT,"a8"));
			mi.return_val.type=Variant::COLOR;
			return mi;
		} break;

		case PRINT_STACK: {
			MethodInfo mi("print_stack");
			mi.return_val.type=Variant::NIL;
			return mi;
		} break;

		case INSTANCE_FROM_ID: {
			MethodInfo mi("instance_from_id",PropertyInfo(Variant::INT,"instance_id"));
			mi.return_val.type=Variant::OBJECT;
			return mi;
		} break;

		case FUNC_MAX: {

			ERR_FAIL_V(MethodInfo());
		} break;

	}
#endif

	return MethodInfo();
}
