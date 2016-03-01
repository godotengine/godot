/*************************************************************************/
/*  gd_functions.h                                                       */
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
#ifndef GD_FUNCTIONS_H
#define GD_FUNCTIONS_H

#include "variant.h"

class GDFunctions {
public:

	enum Function {
		MATH_SIN,
		MATH_COS,
		MATH_TAN,
		MATH_SINH,
		MATH_COSH,
		MATH_TANH,
		MATH_ASIN,
		MATH_ACOS,
		MATH_ATAN,
		MATH_ATAN2,
		MATH_SQRT,
		MATH_FMOD,
		MATH_FPOSMOD,
		MATH_FLOOR,
		MATH_CEIL,
		MATH_ROUND,
		MATH_ABS,
		MATH_SIGN,
		MATH_POW,
		MATH_LOG,
		MATH_EXP,
		MATH_ISNAN,
		MATH_ISINF,
		MATH_EASE,
		MATH_DECIMALS,
		MATH_STEPIFY,
		MATH_LERP,
		MATH_DECTIME,
		MATH_RANDOMIZE,
		MATH_RAND,
		MATH_RANDF,
		MATH_RANDOM,
		MATH_SEED,
		MATH_RANDSEED,
		MATH_DEG2RAD,
		MATH_RAD2DEG,
		MATH_LINEAR2DB,
		MATH_DB2LINEAR,
		LOGIC_MAX,
		LOGIC_MIN,
		LOGIC_CLAMP,
		LOGIC_NEAREST_PO2,
		OBJ_WEAKREF,
		FUNC_FUNCREF,
		TYPE_CONVERT,
		TYPE_OF,
		TEXT_STR,
		TEXT_PRINT,
		TEXT_PRINT_TABBED,
		TEXT_PRINT_SPACED,
		TEXT_PRINTERR,
		TEXT_PRINTRAW,
		VAR_TO_STR,
		STR_TO_VAR,
		VAR_TO_BYTES,
		BYTES_TO_VAR,
		GEN_RANGE,
		GEN_XRANGE,
		RESOURCE_LOAD,
		INST2DICT,
		DICT2INST,
		HASH,
		COLOR8,
		PRINT_STACK,
		INSTANCE_FROM_ID,
		FUNC_MAX

	};

	static void gd_math_sin(int, const Variant**, Variant&, Variant::CallError&);
	static void gd_math_cos(int, const Variant**, Variant&, Variant::CallError&);
	static void gd_math_tan(int, const Variant**, Variant&, Variant::CallError&);
	static void gd_math_sinh(int, const Variant**, Variant&, Variant::CallError&);
	static void gd_math_cosh(int, const Variant**, Variant&, Variant::CallError&);
	static void gd_math_tanh(int, const Variant**, Variant&, Variant::CallError&);
	static void gd_math_asin(int, const Variant**, Variant&, Variant::CallError&);
	static void gd_math_acos(int, const Variant**, Variant&, Variant::CallError&);
	static void gd_math_atan(int, const Variant**, Variant&, Variant::CallError&);
	static void gd_math_atan2(int, const Variant**, Variant&, Variant::CallError&);
	static void gd_math_sqrt(int, const Variant**, Variant&, Variant::CallError&);
	static void gd_math_fmod(int, const Variant**, Variant&, Variant::CallError&);
	static void gd_math_fposmod(int, const Variant**, Variant&, Variant::CallError&);
	static void gd_math_floor(int, const Variant**, Variant&, Variant::CallError&);
	static void gd_math_ceil(int, const Variant**, Variant&, Variant::CallError&);
	static void gd_math_round(int, const Variant**, Variant&, Variant::CallError&);
	static void gd_math_abs(int, const Variant**, Variant&, Variant::CallError&);
	static void gd_math_sign(int, const Variant**, Variant&, Variant::CallError&);
	static void gd_math_pow(int, const Variant**, Variant&, Variant::CallError&);
	static void gd_math_log(int, const Variant**, Variant&, Variant::CallError&);
	static void gd_math_exp(int, const Variant**, Variant&, Variant::CallError&);
	static void gd_math_is_nan(int, const Variant**, Variant&, Variant::CallError&);
	static void gd_math_is_inf(int, const Variant**, Variant&, Variant::CallError&);
	static void gd_math_ease(int, const Variant**, Variant&, Variant::CallError&);
	static void gd_math_decimals(int, const Variant**, Variant&, Variant::CallError&);
	static void gd_math_stepify(int, const Variant**, Variant&, Variant::CallError&);
	static void gd_math_lerp(int, const Variant**, Variant&, Variant::CallError&);
	static void gd_math_dectime(int, const Variant**, Variant&, Variant::CallError&);
	static void gd_math_randomize(int, const Variant**, Variant&, Variant::CallError&);
	static void gd_math_rand(int, const Variant**, Variant&, Variant::CallError&);
	static void gd_math_randf(int, const Variant**, Variant&, Variant::CallError&);
	static void gd_math_random(int, const Variant**, Variant&, Variant::CallError&);
	static void gd_math_seed(int, const Variant**, Variant&, Variant::CallError&);
	static void gd_math_randseed(int, const Variant**, Variant&, Variant::CallError&);
	static void gd_math_deg2rad(int, const Variant**, Variant&, Variant::CallError&);
	static void gd_math_rad2deg(int, const Variant**, Variant&, Variant::CallError&);
	static void gd_math_linear2db(int, const Variant**, Variant&, Variant::CallError&);
	static void gd_math_db2linear(int, const Variant**, Variant&, Variant::CallError&);
	static void gd_logic_max(int, const Variant**, Variant&, Variant::CallError&);
	static void gd_logic_min(int, const Variant**, Variant&, Variant::CallError&);
	static void gd_logic_clamp(int, const Variant**, Variant&, Variant::CallError&);
	static void gd_logic_nearest_po2(int, const Variant**, Variant&, Variant::CallError&);
	static void gd_obj_weakref(int, const Variant**, Variant&, Variant::CallError&);
	static void gd_func_funcref(int, const Variant**, Variant&, Variant::CallError&);
	static void gd_type_convert(int, const Variant**, Variant&, Variant::CallError&);
	static void gd_type_of(int, const Variant**, Variant&, Variant::CallError&);
	static void gd_text_str(int, const Variant**, Variant&, Variant::CallError&);
	static void gd_text_print(int, const Variant**, Variant&, Variant::CallError&);
	static void gd_text_print_tabbed(int, const Variant**, Variant&, Variant::CallError&);
	static void gd_text_print_spaced(int, const Variant**, Variant&, Variant::CallError&);
	static void gd_text_printerr(int, const Variant**, Variant&, Variant::CallError&);
	static void gd_text_printraw(int, const Variant**, Variant&, Variant::CallError&);
	static void gd_var_to_str(int, const Variant**, Variant&, Variant::CallError&);
	static void gd_str_to_var(int, const Variant**, Variant&, Variant::CallError&);
	static void gd_var_to_bytes(int, const Variant**, Variant&, Variant::CallError&);
	static void gd_bytes_to_var(int, const Variant**, Variant&, Variant::CallError&);
	static void gd_gen_range(int, const Variant**, Variant&, Variant::CallError&);
	static void gd_gen_xrange(int, const Variant**, Variant&, Variant::CallError&);
	static void gd_resource_load(int, const Variant**, Variant&, Variant::CallError&);
	static void gd_inst2dict(int, const Variant**, Variant&, Variant::CallError&);
	static void gd_dict2inst(int, const Variant**, Variant&, Variant::CallError&);
	static void gd_hash(int, const Variant**, Variant&, Variant::CallError&);
	static void gd_color8(int, const Variant**, Variant&, Variant::CallError&);
	static void gd_print_stack(int, const Variant**, Variant&, Variant::CallError&);
	static void gd_instance_from_id(int, const Variant**, Variant&, Variant::CallError&);


	static const char *get_func_name(Function p_func);
	static void call(Function p_func,const Variant **p_args,int p_arg_count,Variant &r_ret,Variant::CallError &r_error);
	static bool is_deterministic(Function p_func);
	static MethodInfo get_info(Function p_func);

};

#endif // GD_FUNCTIONS_H
