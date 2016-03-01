/*************************************************************************/
/*  gd_functions_aux.cpp                                                 */
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
#include "math_funcs.h"
#include "gd_functions.h"
#include "gd_script.h"
#include "object_type_db.h"
#include "range_iterator.h"
#include "func_ref.h"
#include "os/os.h"
#include "variant_parser.h"
#include "io/marshalls.h"

#ifdef DEBUG_ENABLED

	#define VALIDATE_ARG_COUNT(m_count) \
		if (p_argc<m_count) {\
			r_error.error=Variant::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS;\
			r_error.argument=m_count;\
			return;\
		}\
		if (p_argc>m_count) {\
			r_error.error=Variant::CallError::CALL_ERROR_TOO_MANY_ARGUMENTS;\
			r_error.argument=m_count;\
			return;\
		}

	#define VALIDATE_ARG_NUM(m_arg) \
		if (!p_args[m_arg]->is_num()) {\
			r_error.error=Variant::CallError::CALL_ERROR_INVALID_ARGUMENT;\
			r_error.argument=m_arg;\
			r_error.expected=Variant::REAL;\
			return;\
		}

	#define VALIDATE_ARGS(m_count)          \
		VALIDATE_ARG_COUNT(m_count);        \
		for (int i = 0; i < m_count; i++) { \
			VALIDATE_ARG_NUM(i);            \
		}                                   \

#else

	#define VALIDATE_ARG_COUNT(m_count)
	#define VALIDATE_ARG_NUM(m_arg)
	#define VALIDATE_ARGS(m_count)

#endif

void GDFunctions::gd_math_sin(int p_argc, const Variant **p_args, Variant &r_ret, Variant::CallError &r_error)
{
	VALIDATE_ARGS(1);
	r_ret = Math::sin(*p_args[0]);
}

void GDFunctions::gd_math_cos(int p_argc, const Variant **p_args, Variant &r_ret, Variant::CallError &r_error)
{
	VALIDATE_ARGS(1);
	r_ret = Math::cos(*p_args[0]);
}

void GDFunctions::gd_math_tan(int p_argc, const Variant **p_args, Variant &r_ret, Variant::CallError &r_error)
{
	VALIDATE_ARGS(1);
	r_ret = Math::tan(*p_args[0]);
}

void GDFunctions::gd_math_sinh(int p_argc, const Variant **p_args, Variant &r_ret, Variant::CallError &r_error)
{
	VALIDATE_ARGS(1);
	r_ret = Math::sinh(*p_args[0]);
}

void GDFunctions::gd_math_cosh(int p_argc, const Variant **p_args, Variant &r_ret, Variant::CallError &r_error)
{
	VALIDATE_ARGS(1);
	r_ret = Math::cosh(*p_args[0]);
}

void GDFunctions::gd_math_tanh(int p_argc, const Variant **p_args, Variant &r_ret, Variant::CallError &r_error)
{
	VALIDATE_ARGS(1);
	r_ret = Math::tanh(*p_args[0]);
}

void GDFunctions::gd_math_asin(int p_argc, const Variant **p_args, Variant &r_ret, Variant::CallError &r_error)
{
	VALIDATE_ARGS(1);
	r_ret = Math::sinh(*p_args[0]);
}

void GDFunctions::gd_math_acos(int p_argc, const Variant **p_args, Variant &r_ret, Variant::CallError &r_error)
{
	VALIDATE_ARGS(1);
	r_ret = Math::acos(*p_args[0]);
}

void GDFunctions::gd_math_atan(int p_argc, const Variant **p_args, Variant &r_ret, Variant::CallError &r_error)
{
	VALIDATE_ARGS(1);
	r_ret = Math::atan(*p_args[0]);
}

void GDFunctions::gd_math_atan2(int p_argc, const Variant **p_args, Variant &r_ret, Variant::CallError &r_error)
{
	VALIDATE_ARGS(2);
	r_ret = Math::atan2(*p_args[0], *p_args[1]);
}

void GDFunctions::gd_math_sqrt(int p_argc, const Variant **p_args, Variant &r_ret, Variant::CallError &r_error)
{
	VALIDATE_ARGS(1);
	r_ret = Math::sqrt(*p_args[0]);
}

void GDFunctions::gd_math_fmod(int p_argc, const Variant **p_args, Variant &r_ret, Variant::CallError &r_error)
{
	VALIDATE_ARGS(2);
	r_ret = Math::fmod(*p_args[0], *p_args[1]);
}

void GDFunctions::gd_math_fposmod(int p_argc, const Variant **p_args, Variant &r_ret, Variant::CallError &r_error)
{
	VALIDATE_ARGS(2);
	r_ret = Math::fposmod(*p_args[0], *p_args[1]);
}

void GDFunctions::gd_math_floor(int p_argc, const Variant **p_args, Variant &r_ret, Variant::CallError &r_error)
{
	VALIDATE_ARGS(1);
	r_ret = Math::floor(*p_args[0]);
}

void GDFunctions::gd_math_ceil(int p_argc, const Variant **p_args, Variant &r_ret, Variant::CallError &r_error)
{
	VALIDATE_ARGS(1);
	r_ret = Math::ceil(*p_args[0]);
}

void GDFunctions::gd_math_round(int p_argc, const Variant **p_args, Variant &r_ret, Variant::CallError &r_error)
{
	VALIDATE_ARGS(1);
	r_ret = Math::round(*p_args[0]);
}

void GDFunctions::gd_math_abs(int p_argc, const Variant **p_args, Variant &r_ret, Variant::CallError &r_error)
{
	VALIDATE_ARG_COUNT(1);
	if (p_args[0]->get_type()==Variant::INT) {

		int64_t i = *p_args[0];
		r_ret=ABS(i);
	} else if (p_args[0]->get_type()==Variant::REAL) {

		real_t r = *p_args[0];
		r_ret=Math::abs(r);
	} else {

		r_error.error=Variant::CallError::CALL_ERROR_INVALID_ARGUMENT;
		r_error.argument=0;
		r_error.expected=Variant::REAL;
	}
}

void GDFunctions::gd_math_sign(int p_argc, const Variant **p_args, Variant &r_ret, Variant::CallError &r_error)
{
	VALIDATE_ARG_COUNT(1);
	if (p_args[0]->get_type()==Variant::INT) {

		int64_t i = *p_args[0];
		r_ret= i < 0 ? -1 : ( i > 0 ? +1 : 0);
	} else if (p_args[0]->get_type()==Variant::REAL) {

		real_t r = *p_args[0];
		r_ret= r < 0.0 ? -1.0 : ( r > 0.0 ? +1.0 : 0.0);
	} else {

		r_error.error=Variant::CallError::CALL_ERROR_INVALID_ARGUMENT;
		r_error.argument=0;
		r_error.expected=Variant::REAL;
	}
}

void GDFunctions::gd_math_pow(int p_argc, const Variant **p_args, Variant &r_ret, Variant::CallError &r_error)
{
	VALIDATE_ARGS(2);
	r_ret = Math::pow(*p_args[0], *p_args[1]);
}

void GDFunctions::gd_math_log(int p_argc, const Variant **p_args, Variant &r_ret, Variant::CallError &r_error)
{
	VALIDATE_ARGS(1);
	r_ret = Math::log(*p_args[0]);
}

void GDFunctions::gd_math_exp(int p_argc, const Variant **p_args, Variant &r_ret, Variant::CallError &r_error)
{
	VALIDATE_ARGS(1);
	r_ret = Math::exp(*p_args[0]);
}

void GDFunctions::gd_math_is_nan(int p_argc, const Variant **p_args, Variant &r_ret, Variant::CallError &r_error)
{
	VALIDATE_ARGS(1);
	r_ret = Math::is_nan(*p_args[0]);
}

void GDFunctions::gd_math_is_inf(int p_argc, const Variant **p_args, Variant &r_ret, Variant::CallError &r_error)
{
	VALIDATE_ARGS(1);
	r_ret = Math::is_inf(*p_args[0]);
}

void GDFunctions::gd_math_ease(int p_argc, const Variant **p_args, Variant &r_ret, Variant::CallError &r_error)
{
	VALIDATE_ARGS(2);
	r_ret = Math::ease(*p_args[0], *p_args[1]);
}

void GDFunctions::gd_math_decimals(int p_argc, const Variant **p_args, Variant &r_ret, Variant::CallError &r_error)
{
	VALIDATE_ARGS(1);
	r_ret = Math::decimals(*p_args[0]);
}

void GDFunctions::gd_math_stepify(int p_argc, const Variant **p_args, Variant &r_ret, Variant::CallError &r_error)
{
	VALIDATE_ARGS(2);
	r_ret = Math::stepify(*p_args[0], *p_args[1]);
}

void GDFunctions::gd_math_lerp(int p_argc, const Variant **p_args, Variant &r_ret, Variant::CallError &r_error)
{
	VALIDATE_ARGS(3);
	r_ret = Math::lerp(*p_args[0], *p_args[1], *p_args[2]);
}

void GDFunctions::gd_math_dectime(int p_argc, const Variant **p_args, Variant &r_ret, Variant::CallError &r_error)
{
	VALIDATE_ARGS(3);
	r_ret = Math::dectime(*p_args[0], *p_args[1], *p_args[2]);
}

void GDFunctions::gd_math_randomize(int p_argc, const Variant **p_args, Variant &r_ret, Variant::CallError &r_error)
{
	Math::randomize();
	r_ret = Variant();
}

void GDFunctions::gd_math_rand(int p_argc, const Variant **p_args, Variant &r_ret, Variant::CallError &r_error)
{
	r_ret = Math::rand();
}

void GDFunctions::gd_math_randf(int p_argc, const Variant **p_args, Variant &r_ret, Variant::CallError &r_error)
{
	r_ret = Math::randf();
}

void GDFunctions::gd_math_random(int p_argc, const Variant **p_args, Variant &r_ret, Variant::CallError &r_error)
{
	VALIDATE_ARGS(2);
	r_ret = Math::random(*p_args[0], *p_args[1]);
}

void GDFunctions::gd_math_seed(int p_argc, const Variant **p_args, Variant &r_ret, Variant::CallError &r_error)
{
	VALIDATE_ARGS(1);
	uint32_t seed=*p_args[0];
	Math::seed(seed);
	r_ret = Variant();
}

void GDFunctions::gd_math_randseed(int p_argc, const Variant **p_args, Variant &r_ret, Variant::CallError &r_error)
{
	VALIDATE_ARGS(1);
	uint32_t seed=*p_args[0];
	int ret = Math::rand_from_seed(&seed);
	Array reta;
	reta.push_back(ret);
	reta.push_back(seed);
	r_ret = reta;
}

void GDFunctions::gd_math_deg2rad(int p_argc, const Variant **p_args, Variant &r_ret, Variant::CallError &r_error)
{
	VALIDATE_ARGS(1);
	r_ret = Math::deg2rad(*p_args[0]);
}

void GDFunctions::gd_math_rad2deg(int p_argc, const Variant **p_args, Variant &r_ret, Variant::CallError &r_error)
{
	VALIDATE_ARGS(1);
	r_ret = Math::rad2deg(*p_args[0]);
}

void GDFunctions::gd_math_linear2db(int p_argc, const Variant **p_args, Variant &r_ret, Variant::CallError &r_error)
{
	VALIDATE_ARGS(1);
	r_ret = Math::linear2db(*p_args[0]);
}

void GDFunctions::gd_math_db2linear(int p_argc, const Variant **p_args, Variant &r_ret, Variant::CallError &r_error)
{
	VALIDATE_ARGS(1);
	r_ret = Math::db2linear(*p_args[0]);
}

void GDFunctions::gd_logic_max(int p_argc, const Variant **p_args, Variant &r_ret, Variant::CallError &r_error)
{
	VALIDATE_ARG_COUNT(2);
	if (p_args[0]->get_type()==Variant::INT && p_args[1]->get_type()==Variant::INT) {

		int64_t a = *p_args[0];
		int64_t b = *p_args[1];

		r_ret = MAX(a,b);
	} else {

		VALIDATE_ARG_NUM(0);
		VALIDATE_ARG_NUM(1);

		real_t a = *p_args[0];
		real_t b = *p_args[1];

		r_ret = MAX(a,b);
	}
}

void GDFunctions::gd_logic_min(int p_argc, const Variant **p_args, Variant &r_ret, Variant::CallError &r_error)
{
	VALIDATE_ARG_COUNT(2);
	if (p_args[0]->get_type()==Variant::INT && p_args[1]->get_type()==Variant::INT) {

		int64_t a = *p_args[0];
		int64_t b = *p_args[1];

		r_ret = MIN(a,b);
	} else {

		VALIDATE_ARG_NUM(0);
		VALIDATE_ARG_NUM(1);

		real_t a = *p_args[0];
		real_t b = *p_args[1];

		r_ret = MIN(a,b);
	}
}

void GDFunctions::gd_logic_clamp(int p_argc, const Variant **p_args, Variant &r_ret, Variant::CallError &r_error)
{
	VALIDATE_ARG_COUNT(3);
	if (p_args[0]->get_type()==Variant::INT &&
	p_args[1]->get_type()==Variant::INT &&
	p_args[2]->get_type()==Variant::INT) {

		int64_t a = *p_args[0];
		int64_t b = *p_args[1];
		int64_t c = *p_args[2];

		r_ret = CLAMP(a,b,c);
	} else {

		VALIDATE_ARG_NUM(0);
		VALIDATE_ARG_NUM(1);
		VALIDATE_ARG_NUM(2);

		real_t a = *p_args[0];
		real_t b = *p_args[1];
		real_t c = *p_args[2];

		r_ret = CLAMP(a,b,c);
	}
}

void GDFunctions::gd_logic_nearest_po2(int p_argc, const Variant **p_args, Variant &r_ret, Variant::CallError &r_error)
{
	VALIDATE_ARGS(1);
	int64_t num = *p_args[0];
	r_ret = nearest_power_of_2(num);
}

void GDFunctions::gd_obj_weakref(int p_argc, const Variant **p_args, Variant &r_ret, Variant::CallError &r_error)
{
	VALIDATE_ARG_COUNT(1);
	if (p_args[0]->get_type()!=Variant::OBJECT) {

		r_error.error=Variant::CallError::CALL_ERROR_INVALID_ARGUMENT;
		r_error.argument=0;
		r_error.expected=Variant::OBJECT;
		return;

	}

	if (p_args[0]->is_ref()) {

		REF r = *p_args[0];
		if (!r.is_valid()) {
			r_ret=Variant();
			return;
		}

		Ref<WeakRef> wref = memnew( WeakRef );
		wref->set_ref(r);
		r_ret=wref;
	} else {
		Object *obj = *p_args[0];
		if (!obj) {
			r_ret=Variant();
			return;
		}
		Ref<WeakRef> wref = memnew( WeakRef );
		wref->set_obj(obj);
		r_ret=wref;
	}
}

void GDFunctions::gd_func_funcref(int p_argc, const Variant **p_args, Variant &r_ret, Variant::CallError &r_error)
{
	VALIDATE_ARG_COUNT(2);
	if (p_args[0]->get_type()!=Variant::OBJECT) {

		r_error.error=Variant::CallError::CALL_ERROR_INVALID_ARGUMENT;
		r_error.argument=0;
		r_error.expected=Variant::OBJECT;
		r_ret=Variant();
		return;

	}
	if (p_args[1]->get_type()!=Variant::STRING && p_args[1]->get_type()!=Variant::NODE_PATH) {

		r_error.error=Variant::CallError::CALL_ERROR_INVALID_ARGUMENT;
		r_error.argument=1;
		r_error.expected=Variant::STRING;
		r_ret=Variant();
		return;

	}

	Ref<FuncRef> fr = memnew( FuncRef);

	Object *obj = *p_args[0];
	fr->set_instance(*p_args[0]);
	fr->set_function(*p_args[1]);

	r_ret=fr;
}

void GDFunctions::gd_type_convert(int p_argc, const Variant **p_args, Variant &r_ret, Variant::CallError &r_error)
{
	VALIDATE_ARG_COUNT(2);
	VALIDATE_ARG_NUM(1);
	int type=*p_args[1];
	if (type<0 || type>=Variant::VARIANT_MAX) {
		ERR_PRINT("Invalid type argument to convert()");
		r_ret = Variant::NIL;
	} else {
		r_ret = Variant::construct(Variant::Type(type),p_args,1,r_error);
	}
}

void GDFunctions::gd_type_of(int p_argc, const Variant **p_args, Variant &r_ret, Variant::CallError &r_error)
{
	VALIDATE_ARG_COUNT(1);
	r_ret = p_args[0]->get_type();
}

void GDFunctions::gd_text_str(int p_argc, const Variant **p_args, Variant &r_ret, Variant::CallError &r_error)
{
	String str;
	for (int i=0; i < p_argc; i++) {
		String os = p_args[i]->operator String();

		if (i == 0)
			str = os;
		else
			str += os;
	}

	r_ret = str;
}

void GDFunctions::gd_text_print(int p_argc, const Variant **p_args, Variant &r_ret, Variant::CallError &r_error)
{
	String str;
	for (int i=0; i < p_argc; i++) {

		str += p_args[i]->operator String();
	}

	//str+="\n";
	print_line(str);
	r_ret = Variant();
}

void GDFunctions::gd_text_print_tabbed(int p_argc, const Variant **p_args, Variant &r_ret, Variant::CallError &r_error)
{
	String str;
	for (int i=0; i < p_argc; i++) {

		if (i) {
			str+="\t";
		}
		str += p_args[i]->operator String();
	}

	//str+="\n";
	print_line(str);
	r_ret = Variant();
}

void GDFunctions::gd_text_print_spaced(int p_argc, const Variant **p_args, Variant &r_ret, Variant::CallError &r_error)
{
	String str;

	for (int i=0; i < p_argc; i++) {

		if (i) {
			str+=" ";
		}
		str += p_args[i]->operator String();
	}

	//str+="\n";
	print_line(str);
	r_ret = Variant();
}

void GDFunctions::gd_text_printerr(int p_argc, const Variant **p_args, Variant &r_ret, Variant::CallError &r_error)
{
	String str;
	for (int i=0; i < p_argc; i++) {
		str += p_args[i]->operator String();
	}

	//str+="\n";
	OS::get_singleton()->printerr("%s\n",str.utf8().get_data());
	r_ret = Variant();
}

void GDFunctions::gd_text_printraw(int p_argc, const Variant **p_args, Variant &r_ret, Variant::CallError &r_error)
{
	String str;
	for (int i=0; i < p_argc; i++) {
		str += p_args[i]->operator String();
	}

	//str+="\n";
	OS::get_singleton()->print("%s",str.utf8().get_data());
	r_ret = Variant();
}

void GDFunctions::gd_var_to_str(int p_argc, const Variant **p_args, Variant &r_ret, Variant::CallError &r_error)
{
	VALIDATE_ARG_COUNT(1);
	String vars;
	VariantWriter::write_to_string(*p_args[0],vars);
	r_ret = vars;
}

void GDFunctions::gd_str_to_var(int p_argc, const Variant **p_args, Variant &r_ret, Variant::CallError &r_error)
{
	VALIDATE_ARG_COUNT(1);
	if (p_args[0]->get_type()!=Variant::STRING) {
		r_error.error=Variant::CallError::CALL_ERROR_INVALID_ARGUMENT;
		r_error.argument=0;
		r_error.expected=Variant::STRING;
		r_ret=Variant();
		return;
	}

	VariantParser::StreamString ss;
	ss.s=*p_args[0];

	String errs;
	int line;
	Error err = VariantParser::parse(&ss,r_ret,errs,line);

	if (err!=OK) {
		r_error.error=Variant::CallError::CALL_ERROR_INVALID_ARGUMENT;
		r_error.argument=0;
		r_error.expected=Variant::STRING;
		r_ret=Variant();
	}
}

void GDFunctions::gd_var_to_bytes(int p_argc, const Variant **p_args, Variant &r_ret, Variant::CallError &r_error)
{
	VALIDATE_ARG_COUNT(1);

	ByteArray barr;
	int len;
	Error err = encode_variant(*p_args[0],NULL,len);
	if (err) {
		r_error.error=Variant::CallError::CALL_ERROR_INVALID_ARGUMENT;
		r_error.argument=0;
		r_error.expected=Variant::NIL;
		r_ret = Variant();
		return;
	}

	barr.resize(len);
	{
		ByteArray::Write w = barr.write();
		encode_variant(*p_args[0],w.ptr(),len);
	}
	r_ret = barr;
}

void GDFunctions::gd_bytes_to_var(int p_argc, const Variant **p_args, Variant &r_ret, Variant::CallError &r_error)
{
	VALIDATE_ARG_COUNT(1);
	if (p_args[0]->get_type()!=Variant::RAW_ARRAY) {
		r_error.error=Variant::CallError::CALL_ERROR_INVALID_ARGUMENT;
		r_error.argument=0;
		r_error.expected=Variant::RAW_ARRAY;
		r_ret=Variant();
		return;
	}

	ByteArray varr=*p_args[0];
	Variant ret;
	{
		ByteArray::Read r=varr.read();
		Error err = decode_variant(ret,r.ptr(),varr.size(),NULL);
		if (err!=OK) {
			ERR_PRINT("Not enough bytes for decoding..");
			r_error.error=Variant::CallError::CALL_ERROR_INVALID_ARGUMENT;
			r_error.argument=0;
			r_error.expected=Variant::RAW_ARRAY;
			r_ret=Variant();
			return;
		}

	}

	r_ret = ret;
}
void GDFunctions::gd_gen_range(int p_argc, const Variant **p_args, Variant &r_ret, Variant::CallError &r_error)
{
	switch(p_argc) {

		case 0: {

			r_error.error=Variant::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS;
			r_error.argument=1;

		} break;
		case 1: {

			VALIDATE_ARG_NUM(0);
			int count=*p_args[0];
			Array arr(true);
			if (count<=0) {
				r_ret=arr;
				return;
			}
			Error err = arr.resize(count);
			if (err!=OK) {
				r_error.error=Variant::CallError::CALL_ERROR_INVALID_METHOD;
				r_ret=Variant();
				return;
			}

			for(int i=0;i<count;i++) {
				arr[i]=i;
			}

			r_ret=arr;
		} break;
		case 2: {

			VALIDATE_ARG_NUM(0);
			VALIDATE_ARG_NUM(1);

			int from=*p_args[0];
			int to=*p_args[1];

			Array arr(true);
			if (from>=to) {
				r_ret=arr;
				return;
			}
			Error err = arr.resize(to-from);
			if (err!=OK) {
				r_error.error=Variant::CallError::CALL_ERROR_INVALID_METHOD;
				r_ret=Variant();
				return;
			}
			for(int i=from;i<to;i++)
				arr[i-from]=i;
			r_ret=arr;
		} break;
		case 3: {

			VALIDATE_ARG_NUM(0);
			VALIDATE_ARG_NUM(1);
			VALIDATE_ARG_NUM(2);

			int from=*p_args[0];
			int to=*p_args[1];
			int incr=*p_args[2];
			if (incr==0) {

				ERR_EXPLAIN("step argument is zero!");
				r_error.error=Variant::CallError::CALL_ERROR_INVALID_METHOD;
				ERR_FAIL();
			}

			Array arr(true);
			if (from>=to && incr>0) {
				r_ret=arr;
				return;
			}
			if (from<=to && incr<0) {
				r_ret=arr;
				return;
			}

			//calculate how many
			int count=0;
			if (incr>0) {

				count=((to-from-1)/incr)+1;
			} else {

				count=((from-to-1)/-incr)+1;
			}


			Error err = arr.resize(count);

			if (err!=OK) {
				r_error.error=Variant::CallError::CALL_ERROR_INVALID_METHOD;
				r_ret=Variant();
				return;
			}

			if (incr>0) {
				int idx=0;
				for(int i=from;i<to;i+=incr) {
					arr[idx++]=i;
				}
			} else {

				int idx=0;
				for(int i=from;i>to;i+=incr) {
					arr[idx++]=i;
				}
			}

			r_ret=arr;
		} break;
		default: {

			r_error.error=Variant::CallError::CALL_ERROR_TOO_MANY_ARGUMENTS;
			r_error.argument=3;
		} break;
	}

}

void GDFunctions::gd_gen_xrange(int p_argc, const Variant **p_args, Variant &r_ret, Variant::CallError &r_error)
{
	switch(p_argc) {
		case 0: {
			r_error.error=Variant::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS;
			r_error.argument=1;
		} break;
		case 1: {

			VALIDATE_ARG_NUM(0);

			int count=*p_args[0];

			Ref<RangeIterator> itr = Ref<RangeIterator>( memnew(RangeIterator) );
			if (!*itr) {
				ERR_EXPLAIN("Couldn't allocate iterator!");
				r_error.error=Variant::CallError::CALL_ERROR_INVALID_METHOD;
				ERR_FAIL();
			}
			(*itr)->set_range(count);
			r_ret=Variant(itr);
			return;
		} break;
		case 2: {

			VALIDATE_ARG_NUM(0);
			VALIDATE_ARG_NUM(1);

			int from=*p_args[0];
			int to=*p_args[1];

			Ref<RangeIterator> itr = Ref<RangeIterator>( memnew(RangeIterator) );
			if (!*itr) {
				ERR_EXPLAIN("Couldn't allocate iterator!");
				r_error.error=Variant::CallError::CALL_ERROR_INVALID_METHOD;
				ERR_FAIL();
			}
			(*itr)->set_range(from, to);
			r_ret=Variant(itr);
			return;
		} break;
		case 3: {

			VALIDATE_ARG_NUM(0);
			VALIDATE_ARG_NUM(1);
			VALIDATE_ARG_NUM(2);

			int from=*p_args[0];
			int to=*p_args[1];
			int incr=*p_args[2];

			if (incr==0) {
				ERR_EXPLAIN("step argument is zero!");
				r_error.error=Variant::CallError::CALL_ERROR_INVALID_METHOD;
				ERR_FAIL();
			}

			Ref<RangeIterator> itr = Ref<RangeIterator>( memnew(RangeIterator) );
			if (!*itr) {
				ERR_EXPLAIN("Couldn't allocate iterator!");
				r_error.error=Variant::CallError::CALL_ERROR_INVALID_METHOD;
				ERR_FAIL();
			}
			(*itr)->set_range(from, to, incr);
			r_ret=Variant(itr);
			return;
		} break;
		default: {

			r_error.error=Variant::CallError::CALL_ERROR_TOO_MANY_ARGUMENTS;
			r_error.argument=3;
		} break;
	}

}

void GDFunctions::gd_resource_load(int p_argc, const Variant **p_args, Variant &r_ret, Variant::CallError &r_error)
{
	VALIDATE_ARG_COUNT(1);
	if (p_args[0]->get_type()!=Variant::STRING) {
		r_error.error=Variant::CallError::CALL_ERROR_INVALID_ARGUMENT;
		r_error.argument=0;
		r_ret = Variant();
		// FIXME: return; ?
	}
	r_ret = ResourceLoader::load(*p_args[0]);
}

void GDFunctions::gd_inst2dict(int p_argc, const Variant **p_args, Variant &r_ret, Variant::CallError &r_error)
{
	VALIDATE_ARG_COUNT(1);

	if (p_args[0]->get_type()==Variant::NIL) {
		r_ret=Variant();
	} else if (p_args[0]->get_type()!=Variant::OBJECT) {
		r_error.error=Variant::CallError::CALL_ERROR_INVALID_ARGUMENT;
		r_error.argument=0;
		r_ret=Variant();
	} else {

		Object *obj = *p_args[0];
		if (!obj) {
			r_ret=Variant();

		} else if (!obj->get_script_instance() || obj->get_script_instance()->get_language()!=GDScriptLanguage::get_singleton()) {

			r_error.error=Variant::CallError::CALL_ERROR_INVALID_ARGUMENT;
			r_error.argument=0;
			r_error.expected=Variant::DICTIONARY;
			ERR_PRINT("Not a script with an instance");

		} else {

			GDInstance *ins = static_cast<GDInstance*>(obj->get_script_instance());
			Ref<GDScript> base = ins->get_script();
			if (base.is_null()) {

				r_error.error=Variant::CallError::CALL_ERROR_INVALID_ARGUMENT;
				r_error.argument=0;
				r_error.expected=Variant::DICTIONARY;
				ERR_PRINT("Not based on a script");
				return;

			}


			GDScript *p = base.ptr();
			Vector<StringName> sname;

			while(p->_owner) {

				sname.push_back(p->name);
				p=p->_owner;
			}
			sname.invert();


			if (!p->path.is_resource_file()) {
				r_error.error=Variant::CallError::CALL_ERROR_INVALID_ARGUMENT;
				r_error.argument=0;
				r_error.expected=Variant::DICTIONARY;
				print_line("PATH: "+p->path);
				ERR_PRINT("Not based on a resource file");

				return;
			}

			NodePath cp(sname,Vector<StringName>(),false);

			Dictionary d(true);
			d["@subpath"]=cp;
			d["@path"]=p->path;


			p = base.ptr();

			while(p) {

				for(Set<StringName>::Element *E=p->members.front();E;E=E->next()) {

					Variant value;
					if (ins->get(E->get(),value)) {

						String k = E->get();
						if (!d.has(k)) {
							d[k]=value;
						}
					}
				}

				p=p->_base;
			}

			r_ret=d;

		}
	}
}

void GDFunctions::gd_dict2inst(int p_argc, const Variant **p_args, Variant &r_ret, Variant::CallError &r_error)
{
	VALIDATE_ARG_COUNT(1);

	if (p_args[0]->get_type()!=Variant::DICTIONARY) {

		r_error.error=Variant::CallError::CALL_ERROR_INVALID_ARGUMENT;
		r_error.argument=0;
		r_error.expected=Variant::DICTIONARY;
		return;
	}

	Dictionary d = *p_args[0];

	if (!d.has("@path")) {

		r_error.error=Variant::CallError::CALL_ERROR_INVALID_ARGUMENT;
		r_error.argument=0;
		r_error.expected=Variant::OBJECT;
		return;
	}

	Ref<Script> scr = ResourceLoader::load(d["@path"]);
	if (!scr.is_valid()) {

		r_error.error=Variant::CallError::CALL_ERROR_INVALID_ARGUMENT;
		r_error.argument=0;
		r_error.expected=Variant::OBJECT;
		return;
	}

	Ref<GDScript> gdscr = scr;

	if (!gdscr.is_valid()) {

		r_error.error=Variant::CallError::CALL_ERROR_INVALID_ARGUMENT;
		r_error.argument=0;
		r_error.expected=Variant::OBJECT;
		return;
	}

	NodePath sub;
	if (d.has("@subpath")) {
		sub=d["@subpath"];
	}

	for(int i=0;i<sub.get_name_count();i++) {

		gdscr = gdscr->subclasses[ sub.get_name(i)];
		if (!gdscr.is_valid()) {

			r_error.error=Variant::CallError::CALL_ERROR_INVALID_ARGUMENT;
			r_error.argument=0;
			r_error.expected=Variant::OBJECT;
			return;
		}
	}

	r_ret = gdscr->_new(NULL,0,r_error);

	GDInstance *ins = static_cast<GDInstance*>(static_cast<Object*>(r_ret)->get_script_instance());
	Ref<GDScript> gd_ref = ins->get_script();

	for(Map<StringName,GDScript::MemberInfo>::Element *E = gd_ref->member_indices.front(); E; E = E->next()) {
		if(d.has(E->key())) {
			ins->members[E->get().index] = d[E->key()];
		}
	}
}

void GDFunctions::gd_hash(int p_argc, const Variant **p_args, Variant &r_ret, Variant::CallError &r_error)
{
	VALIDATE_ARG_COUNT(1);
	r_ret = p_args[0]->hash();
}

void GDFunctions::gd_color8(int p_argc, const Variant **p_args, Variant &r_ret, Variant::CallError &r_error)
{
	if (p_argc<3) {
		r_error.error=Variant::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS;
		r_error.argument=3;
		return;
	}
	if (p_argc>4) {
		r_error.error=Variant::CallError::CALL_ERROR_TOO_MANY_ARGUMENTS;
		r_error.argument=4;
		return;
	}

	VALIDATE_ARG_NUM(0);
	VALIDATE_ARG_NUM(1);
	VALIDATE_ARG_NUM(2);

	Color color(*p_args[0],*p_args[1],*p_args[2]);

	if (p_argc==4) {
		VALIDATE_ARG_NUM(3);
		color.a=*p_args[3];
	}

	r_ret=color;
}

void GDFunctions::gd_print_stack(int p_argc, const Variant **p_args, Variant &r_ret, Variant::CallError &r_error)
{
	ScriptLanguage* script = GDScriptLanguage::get_singleton();
	for (int i=0; i < script->debug_get_stack_level_count(); i++) {

		print_line("Frame "+itos(i)+" - "+script->debug_get_stack_level_source(i)+":"+itos(script->debug_get_stack_level_line(i))+" in function '"+script->debug_get_stack_level_function(i)+"'");
	};
}

void GDFunctions::gd_instance_from_id(int p_argc, const Variant **p_args, Variant &r_ret, Variant::CallError &r_error)
{
	VALIDATE_ARG_COUNT(1);
	if (p_args[0]->get_type()!=Variant::INT && p_args[0]->get_type()!=Variant::REAL) {
		r_error.error=Variant::CallError::CALL_ERROR_INVALID_ARGUMENT;
		r_error.argument=0;
		r_ret=Variant();
		return;
	}

	uint32_t id=*p_args[0];
	r_ret=ObjectDB::get_instance(id);
}
