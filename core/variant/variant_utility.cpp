/*************************************************************************/
/*  variant_utility.cpp                                                  */
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

#include "variant.h"

#include "core/core_string_names.h"
#include "core/io/marshalls.h"
#include "core/object/ref_counted.h"
#include "core/os/os.h"
#include "core/templates/oa_hash_map.h"
#include "core/templates/rid.h"
#include "core/templates/rid_owner.h"
#include "core/variant/binder_common.h"
#include "core/variant/variant_parser.h"

struct VariantUtilityFunctions {
	// Math

	static inline double sin(double arg) {
		return Math::sin(arg);
	}

	static inline double cos(double arg) {
		return Math::cos(arg);
	}

	static inline double tan(double arg) {
		return Math::tan(arg);
	}

	static inline double sinh(double arg) {
		return Math::sinh(arg);
	}

	static inline double cosh(double arg) {
		return Math::cosh(arg);
	}

	static inline double tanh(double arg) {
		return Math::tanh(arg);
	}

	static inline double asin(double arg) {
		return Math::asin(arg);
	}

	static inline double acos(double arg) {
		return Math::acos(arg);
	}

	static inline double atan(double arg) {
		return Math::atan(arg);
	}

	static inline double atan2(double y, double x) {
		return Math::atan2(y, x);
	}

	static inline double sqrt(double x) {
		return Math::sqrt(x);
	}

	static inline double fmod(double b, double r) {
		return Math::fmod(b, r);
	}

	static inline double fposmod(double b, double r) {
		return Math::fposmod(b, r);
	}

	static inline int64_t posmod(int64_t b, int64_t r) {
		return Math::posmod(b, r);
	}

	static inline double floor(double x) {
		return Math::floor(x);
	}

	static inline double ceil(double x) {
		return Math::ceil(x);
	}

	static inline double round(double x) {
		return Math::round(x);
	}

	static inline Variant abs(const Variant &x, Callable::CallError &r_error) {
		r_error.error = Callable::CallError::CALL_OK;
		switch (x.get_type()) {
			case Variant::INT: {
				return ABS(VariantInternalAccessor<int64_t>::get(&x));
			} break;
			case Variant::FLOAT: {
				return Math::absd(VariantInternalAccessor<double>::get(&x));
			} break;
			case Variant::VECTOR2: {
				return VariantInternalAccessor<Vector2>::get(&x).abs();
			} break;
			case Variant::VECTOR2I: {
				return VariantInternalAccessor<Vector2i>::get(&x).abs();
			} break;
			case Variant::VECTOR3: {
				return VariantInternalAccessor<Vector3>::get(&x).abs();
			} break;
			case Variant::VECTOR3I: {
				return VariantInternalAccessor<Vector3i>::get(&x).abs();
			} break;
			default: {
				r_error.error = Callable::CallError::CALL_ERROR_INVALID_METHOD;
				return Variant();
			}
		}
	}

	static inline double absf(double x) {
		return Math::absd(x);
	}

	static inline int64_t absi(int64_t x) {
		return ABS(x);
	}

	static inline Variant sign(const Variant &x, Callable::CallError &r_error) {
		r_error.error = Callable::CallError::CALL_OK;
		switch (x.get_type()) {
			case Variant::INT: {
				return SIGN(VariantInternalAccessor<int64_t>::get(&x));
			} break;
			case Variant::FLOAT: {
				return SIGN(VariantInternalAccessor<double>::get(&x));
			} break;
			case Variant::VECTOR2: {
				return VariantInternalAccessor<Vector2>::get(&x).sign();
			} break;
			case Variant::VECTOR2I: {
				return VariantInternalAccessor<Vector2i>::get(&x).sign();
			} break;
			case Variant::VECTOR3: {
				return VariantInternalAccessor<Vector3>::get(&x).sign();
			} break;
			case Variant::VECTOR3I: {
				return VariantInternalAccessor<Vector3i>::get(&x).sign();
			} break;
			default: {
				r_error.error = Callable::CallError::CALL_ERROR_INVALID_METHOD;
				return Variant();
			}
		}
	}

	static inline double signf(double x) {
		return SIGN(x);
	}

	static inline int64_t signi(int64_t x) {
		return SIGN(x);
	}

	static inline double pow(double x, double y) {
		return Math::pow(x, y);
	}

	static inline double log(double x) {
		return Math::log(x);
	}

	static inline double exp(double x) {
		return Math::exp(x);
	}

	static inline bool is_nan(double x) {
		return Math::is_nan(x);
	}

	static inline bool is_inf(double x) {
		return Math::is_inf(x);
	}

	static inline bool is_equal_approx(double x, double y) {
		return Math::is_equal_approx(x, y);
	}

	static inline bool is_zero_approx(double x) {
		return Math::is_zero_approx(x);
	}

	static inline double ease(float x, float curve) {
		return Math::ease(x, curve);
	}

	static inline int step_decimals(float step) {
		return Math::step_decimals(step);
	}

	static inline int range_step_decimals(float step) {
		return Math::range_step_decimals(step);
	}

	static inline double snapped(double value, double step) {
		return Math::snapped(value, step);
	}

	static inline double lerp(double from, double to, double weight) {
		return Math::lerp(from, to, weight);
	}

	static inline double lerp_angle(double from, double to, double weight) {
		return Math::lerp_angle(from, to, weight);
	}

	static inline double inverse_lerp(double from, double to, double weight) {
		return Math::inverse_lerp(from, to, weight);
	}

	static inline double range_lerp(double value, double istart, double istop, double ostart, double ostop) {
		return Math::range_lerp(value, istart, istop, ostart, ostop);
	}

	static inline double smoothstep(double from, double to, double val) {
		return Math::smoothstep(from, to, val);
	}

	static inline double move_toward(double from, double to, double delta) {
		return Math::move_toward(from, to, delta);
	}

	static inline double deg2rad(double angle_deg) {
		return Math::deg2rad(angle_deg);
	}

	static inline double rad2deg(double angle_rad) {
		return Math::rad2deg(angle_rad);
	}

	static inline double linear2db(double linear) {
		return Math::linear2db(linear);
	}

	static inline double db2linear(double db) {
		return Math::db2linear(db);
	}

	static inline int64_t wrapi(int64_t value, int64_t min, int64_t max) {
		return Math::wrapi(value, min, max);
	}

	static inline double wrapf(double value, double min, double max) {
		return Math::wrapf(value, min, max);
	}

	static inline double pingpong(double value, double length) {
		return Math::pingpong(value, length);
	}

	static inline Variant max(const Variant **p_args, int p_argcount, Callable::CallError &r_error) {
		if (p_argcount < 2) {
			r_error.error = Callable::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS;
			r_error.expected = 2;
			return Variant();
		}
		Variant base = *p_args[0];
		Variant ret;
		for (int i = 1; i < p_argcount; i++) {
			bool valid;
			Variant::evaluate(Variant::OP_LESS, base, *p_args[i], ret, valid);
			if (!valid) {
				r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
				r_error.expected = base.get_type();
				r_error.argument = i;
				return Variant();
			}
			if (ret.booleanize()) {
				base = *p_args[i];
			}
		}
		r_error.error = Callable::CallError::CALL_OK;
		return base;
	}

	static inline double maxf(double x, double y) {
		return MAX(x, y);
	}

	static inline int64_t maxi(int64_t x, int64_t y) {
		return MAX(x, y);
	}

	static inline Variant min(const Variant **p_args, int p_argcount, Callable::CallError &r_error) {
		if (p_argcount < 2) {
			r_error.error = Callable::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS;
			r_error.expected = 2;
			return Variant();
		}
		Variant base = *p_args[0];
		Variant ret;
		for (int i = 1; i < p_argcount; i++) {
			bool valid;
			Variant::evaluate(Variant::OP_GREATER, base, *p_args[i], ret, valid);
			if (!valid) {
				r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
				r_error.expected = base.get_type();
				r_error.argument = i;
				return Variant();
			}
			if (ret.booleanize()) {
				base = *p_args[i];
			}
		}
		r_error.error = Callable::CallError::CALL_OK;
		return base;
	}

	static inline double minf(double x, double y) {
		return MIN(x, y);
	}

	static inline int64_t mini(int64_t x, int64_t y) {
		return MIN(x, y);
	}

	static inline Variant clamp(const Variant &x, const Variant &min, const Variant &max, Callable::CallError &r_error) {
		Variant value = x;

		Variant ret;

		bool valid;
		Variant::evaluate(Variant::OP_LESS, value, min, ret, valid);
		if (!valid) {
			r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
			r_error.expected = value.get_type();
			r_error.argument = 1;
			return Variant();
		}
		if (ret.booleanize()) {
			value = min;
		}
		Variant::evaluate(Variant::OP_GREATER, value, max, ret, valid);
		if (!valid) {
			r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
			r_error.expected = value.get_type();
			r_error.argument = 2;
			return Variant();
		}
		if (ret.booleanize()) {
			value = max;
		}

		r_error.error = Callable::CallError::CALL_OK;

		return value;
	}

	static inline double clampf(double x, double min, double max) {
		return CLAMP(x, min, max);
	}

	static inline int64_t clampi(int64_t x, int64_t min, int64_t max) {
		return CLAMP(x, min, max);
	}

	static inline int64_t nearest_po2(int64_t x) {
		return nearest_power_of_2_templated(uint64_t(x));
	}

	// Random

	static inline void randomize() {
		Math::randomize();
	}

	static inline int64_t randi() {
		return Math::rand();
	}

	static inline double randf() {
		return Math::randf();
	}

	static inline int64_t randi_range(int64_t from, int64_t to) {
		return Math::random((int32_t)from, (int32_t)to);
	}

	static inline double randf_range(double from, double to) {
		return Math::random(from, to);
	}

	static inline void seed(int64_t s) {
		return Math::seed(s);
	}

	static inline PackedInt64Array rand_from_seed(int64_t seed) {
		uint64_t s = seed;
		PackedInt64Array arr;
		arr.resize(2);
		arr.write[0] = Math::rand_from_seed(&s);
		arr.write[1] = s;
		return arr;
	}

	// Utility

	static inline Variant weakref(const Variant &obj, Callable::CallError &r_error) {
		if (obj.get_type() == Variant::OBJECT) {
			r_error.error = Callable::CallError::CALL_OK;
			if (obj.is_ref()) {
				Ref<WeakRef> wref = memnew(WeakRef);
				REF r = obj;
				if (r.is_valid()) {
					wref->set_ref(r);
				}
				return wref;
			} else {
				Ref<WeakRef> wref = memnew(WeakRef);
				Object *o = obj.get_validated_object();
				if (o) {
					wref->set_obj(o);
				}
				return wref;
			}
		} else if (obj.get_type() == Variant::NIL) {
			r_error.error = Callable::CallError::CALL_OK;
			Ref<WeakRef> wref = memnew(WeakRef);
			return wref;
		} else {
			r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
			r_error.argument = 0;
			r_error.expected = Variant::OBJECT;
			return Variant();
		}
	}

	static inline int64_t _typeof(const Variant &obj) {
		return obj.get_type();
	}

	static inline String str(const Variant **p_args, int p_arg_count, Callable::CallError &r_error) {
		if (p_arg_count < 1) {
			r_error.error = Callable::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS;
			r_error.argument = 1;
			return String();
		}
		String str;
		for (int i = 0; i < p_arg_count; i++) {
			String os = p_args[i]->operator String();

			if (i == 0) {
				str = os;
			} else {
				str += os;
			}
		}

		r_error.error = Callable::CallError::CALL_OK;

		return str;
	}

	static inline String error_string(Error error) {
		if (error < 0 || error >= ERR_MAX) {
			return String("(invalid error code)");
		}

		return String(error_names[error]);
	}

	static inline void print(const Variant **p_args, int p_arg_count, Callable::CallError &r_error) {
		String str;
		for (int i = 0; i < p_arg_count; i++) {
			String os = p_args[i]->operator String();

			if (i == 0) {
				str = os;
			} else {
				str += os;
			}
		}

		print_line(str);
		r_error.error = Callable::CallError::CALL_OK;
	}

	static inline void print_verbose(const Variant **p_args, int p_arg_count, Callable::CallError &r_error) {
		if (OS::get_singleton()->is_stdout_verbose()) {
			String str;
			for (int i = 0; i < p_arg_count; i++) {
				String os = p_args[i]->operator String();

				if (i == 0) {
					str = os;
				} else {
					str += os;
				}
			}

			// No need to use `print_verbose()` as this call already only happens
			// when verbose mode is enabled. This avoids performing string argument concatenation
			// when not needed.
			print_line(str);
		}

		r_error.error = Callable::CallError::CALL_OK;
	}

	static inline void printerr(const Variant **p_args, int p_arg_count, Callable::CallError &r_error) {
		String str;
		for (int i = 0; i < p_arg_count; i++) {
			String os = p_args[i]->operator String();

			if (i == 0) {
				str = os;
			} else {
				str += os;
			}
		}

		print_error(str);
		r_error.error = Callable::CallError::CALL_OK;
	}

	static inline void printt(const Variant **p_args, int p_arg_count, Callable::CallError &r_error) {
		String str;
		for (int i = 0; i < p_arg_count; i++) {
			if (i) {
				str += "\t";
			}
			str += p_args[i]->operator String();
		}

		print_line(str);
		r_error.error = Callable::CallError::CALL_OK;
	}

	static inline void prints(const Variant **p_args, int p_arg_count, Callable::CallError &r_error) {
		String str;
		for (int i = 0; i < p_arg_count; i++) {
			if (i) {
				str += " ";
			}
			str += p_args[i]->operator String();
		}

		print_line(str);
		r_error.error = Callable::CallError::CALL_OK;
	}

	static inline void printraw(const Variant **p_args, int p_arg_count, Callable::CallError &r_error) {
		String str;
		for (int i = 0; i < p_arg_count; i++) {
			String os = p_args[i]->operator String();

			if (i == 0) {
				str = os;
			} else {
				str += os;
			}
		}

		OS::get_singleton()->print("%s", str.utf8().get_data());
		r_error.error = Callable::CallError::CALL_OK;
	}

	static inline void push_error(const Variant **p_args, int p_arg_count, Callable::CallError &r_error) {
		if (p_arg_count < 1) {
			r_error.error = Callable::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS;
			r_error.argument = 1;
		}
		String str;
		for (int i = 0; i < p_arg_count; i++) {
			String os = p_args[i]->operator String();

			if (i == 0) {
				str = os;
			} else {
				str += os;
			}
		}

		ERR_PRINT(str);
		r_error.error = Callable::CallError::CALL_OK;
	}

	static inline void push_warning(const Variant **p_args, int p_arg_count, Callable::CallError &r_error) {
		if (p_arg_count < 1) {
			r_error.error = Callable::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS;
			r_error.argument = 1;
		}
		String str;
		for (int i = 0; i < p_arg_count; i++) {
			String os = p_args[i]->operator String();

			if (i == 0) {
				str = os;
			} else {
				str += os;
			}
		}

		WARN_PRINT(str);
		r_error.error = Callable::CallError::CALL_OK;
	}

	static inline String var2str(const Variant &p_var) {
		String vars;
		VariantWriter::write_to_string(p_var, vars);
		return vars;
	}

	static inline Variant str2var(const String &p_var) {
		VariantParser::StreamString ss;
		ss.s = p_var;

		String errs;
		int line;
		Variant ret;
		(void)VariantParser::parse(&ss, ret, errs, line);

		return ret;
	}

	static inline PackedByteArray var2bytes(const Variant &p_var) {
		int len;
		Error err = encode_variant(p_var, nullptr, len, false);
		if (err != OK) {
			return PackedByteArray();
		}

		PackedByteArray barr;
		barr.resize(len);
		{
			uint8_t *w = barr.ptrw();
			err = encode_variant(p_var, w, len, false);
			if (err != OK) {
				return PackedByteArray();
			}
		}

		return barr;
	}

	static inline PackedByteArray var2bytes_with_objects(const Variant &p_var) {
		int len;
		Error err = encode_variant(p_var, nullptr, len, true);
		if (err != OK) {
			return PackedByteArray();
		}

		PackedByteArray barr;
		barr.resize(len);
		{
			uint8_t *w = barr.ptrw();
			err = encode_variant(p_var, w, len, true);
			if (err != OK) {
				return PackedByteArray();
			}
		}

		return barr;
	}

	static inline Variant bytes2var(const PackedByteArray &p_arr) {
		Variant ret;
		{
			const uint8_t *r = p_arr.ptr();
			Error err = decode_variant(ret, r, p_arr.size(), nullptr, false);
			if (err != OK) {
				return Variant();
			}
		}
		return ret;
	}

	static inline Variant bytes2var_with_objects(const PackedByteArray &p_arr) {
		Variant ret;
		{
			const uint8_t *r = p_arr.ptr();
			Error err = decode_variant(ret, r, p_arr.size(), nullptr, true);
			if (err != OK) {
				return Variant();
			}
		}
		return ret;
	}

	static inline int64_t hash(const Variant &p_arr) {
		return p_arr.hash();
	}

	static inline Object *instance_from_id(int64_t p_id) {
		ObjectID id = ObjectID((uint64_t)p_id);
		Object *ret = ObjectDB::get_instance(id);
		return ret;
	}

	static inline bool is_instance_id_valid(int64_t p_id) {
		return ObjectDB::get_instance(ObjectID((uint64_t)p_id)) != nullptr;
	}

	static inline bool is_instance_valid(const Variant &p_instance) {
		if (p_instance.get_type() != Variant::OBJECT) {
			return false;
		}
		return p_instance.get_validated_object() != nullptr;
	}

	static inline uint64_t rid_allocate_id() {
		return RID_AllocBase::_gen_id();
	}
	static inline RID rid_from_int64(uint64_t p_base) {
		return RID::from_uint64(p_base);
	}
};

#ifdef DEBUG_METHODS_ENABLED
#define VCALLR *ret = p_func(VariantCasterAndValidate<P>::cast(p_args, Is, r_error)...)
#define VCALL p_func(VariantCasterAndValidate<P>::cast(p_args, Is, r_error)...)
#else
#define VCALLR *ret = p_func(VariantCaster<P>::cast(*p_args[Is])...)
#define VCALL p_func(VariantCaster<P>::cast(*p_args[Is])...)
#endif

template <class R, class... P, size_t... Is>
static _FORCE_INLINE_ void call_helperpr(R (*p_func)(P...), Variant *ret, const Variant **p_args, Callable::CallError &r_error, IndexSequence<Is...>) {
	r_error.error = Callable::CallError::CALL_OK;
	VCALLR;
	(void)p_args; // avoid gcc warning
	(void)r_error;
}

template <class R, class... P, size_t... Is>
static _FORCE_INLINE_ void validated_call_helperpr(R (*p_func)(P...), Variant *ret, const Variant **p_args, IndexSequence<Is...>) {
	*ret = p_func(VariantCaster<P>::cast(*p_args[Is])...);
	(void)p_args;
}

template <class R, class... P, size_t... Is>
static _FORCE_INLINE_ void ptr_call_helperpr(R (*p_func)(P...), void *ret, const void **p_args, IndexSequence<Is...>) {
	PtrToArg<R>::encode(p_func(PtrToArg<P>::convert(p_args[Is])...), ret);
	(void)p_args;
}

template <class R, class... P>
static _FORCE_INLINE_ void call_helperr(R (*p_func)(P...), Variant *ret, const Variant **p_args, Callable::CallError &r_error) {
	call_helperpr(p_func, ret, p_args, r_error, BuildIndexSequence<sizeof...(P)>{});
}

template <class R, class... P>
static _FORCE_INLINE_ void validated_call_helperr(R (*p_func)(P...), Variant *ret, const Variant **p_args) {
	validated_call_helperpr(p_func, ret, p_args, BuildIndexSequence<sizeof...(P)>{});
}

template <class R, class... P>
static _FORCE_INLINE_ void ptr_call_helperr(R (*p_func)(P...), void *ret, const void **p_args) {
	ptr_call_helperpr(p_func, ret, p_args, BuildIndexSequence<sizeof...(P)>{});
}

template <class R, class... P>
static _FORCE_INLINE_ int get_arg_count_helperr(R (*p_func)(P...)) {
	return sizeof...(P);
}

template <class R, class... P>
static _FORCE_INLINE_ Variant::Type get_arg_type_helperr(R (*p_func)(P...), int p_arg) {
	return call_get_argument_type<P...>(p_arg);
}

template <class R, class... P>
static _FORCE_INLINE_ Variant::Type get_ret_type_helperr(R (*p_func)(P...)) {
	return GetTypeInfo<R>::VARIANT_TYPE;
}

// WITHOUT RET

template <class... P, size_t... Is>
static _FORCE_INLINE_ void call_helperp(void (*p_func)(P...), const Variant **p_args, Callable::CallError &r_error, IndexSequence<Is...>) {
	r_error.error = Callable::CallError::CALL_OK;
	VCALL;
	(void)p_args;
	(void)r_error;
}

template <class... P, size_t... Is>
static _FORCE_INLINE_ void validated_call_helperp(void (*p_func)(P...), const Variant **p_args, IndexSequence<Is...>) {
	p_func(VariantCaster<P>::cast(*p_args[Is])...);
	(void)p_args;
}

template <class... P, size_t... Is>
static _FORCE_INLINE_ void ptr_call_helperp(void (*p_func)(P...), const void **p_args, IndexSequence<Is...>) {
	p_func(PtrToArg<P>::convert(p_args[Is])...);
	(void)p_args;
}

template <class... P>
static _FORCE_INLINE_ void call_helper(void (*p_func)(P...), const Variant **p_args, Callable::CallError &r_error) {
	call_helperp(p_func, p_args, r_error, BuildIndexSequence<sizeof...(P)>{});
}

template <class... P>
static _FORCE_INLINE_ void validated_call_helper(void (*p_func)(P...), const Variant **p_args) {
	validated_call_helperp(p_func, p_args, BuildIndexSequence<sizeof...(P)>{});
}

template <class... P>
static _FORCE_INLINE_ void ptr_call_helper(void (*p_func)(P...), const void **p_args) {
	ptr_call_helperp(p_func, p_args, BuildIndexSequence<sizeof...(P)>{});
}

template <class... P>
static _FORCE_INLINE_ int get_arg_count_helper(void (*p_func)(P...)) {
	return sizeof...(P);
}

template <class... P>
static _FORCE_INLINE_ Variant::Type get_arg_type_helper(void (*p_func)(P...), int p_arg) {
	return call_get_argument_type<P...>(p_arg);
}

template <class... P>
static _FORCE_INLINE_ Variant::Type get_ret_type_helper(void (*p_func)(P...)) {
	return Variant::NIL;
}

#define FUNCBINDR(m_func, m_args, m_category)                                                                    \
	class Func_##m_func {                                                                                        \
	public:                                                                                                      \
		static void call(Variant *r_ret, const Variant **p_args, int p_argcount, Callable::CallError &r_error) { \
			call_helperr(VariantUtilityFunctions::m_func, r_ret, p_args, r_error);                               \
		}                                                                                                        \
		static void validated_call(Variant *r_ret, const Variant **p_args, int p_argcount) {                     \
			validated_call_helperr(VariantUtilityFunctions::m_func, r_ret, p_args);                              \
		}                                                                                                        \
		static void ptrcall(void *ret, const void **p_args, int p_argcount) {                                    \
			ptr_call_helperr(VariantUtilityFunctions::m_func, ret, p_args);                                      \
		}                                                                                                        \
		static int get_argument_count() {                                                                        \
			return get_arg_count_helperr(VariantUtilityFunctions::m_func);                                       \
		}                                                                                                        \
		static Variant::Type get_argument_type(int p_arg) {                                                      \
			return get_arg_type_helperr(VariantUtilityFunctions::m_func, p_arg);                                 \
		}                                                                                                        \
		static Variant::Type get_return_type() {                                                                 \
			return get_ret_type_helperr(VariantUtilityFunctions::m_func);                                        \
		}                                                                                                        \
		static bool has_return_type() {                                                                          \
			return true;                                                                                         \
		}                                                                                                        \
		static bool is_vararg() { return false; }                                                                \
		static Variant::UtilityFunctionType get_type() { return m_category; }                                    \
	};                                                                                                           \
	register_utility_function<Func_##m_func>(#m_func, m_args)

#define FUNCBINDVR(m_func, m_args, m_category)                                                                          \
	class Func_##m_func {                                                                                               \
	public:                                                                                                             \
		static void call(Variant *r_ret, const Variant **p_args, int p_argcount, Callable::CallError &r_error) {        \
			r_error.error = Callable::CallError::CALL_OK;                                                               \
			*r_ret = VariantUtilityFunctions::m_func(*p_args[0], r_error);                                              \
		}                                                                                                               \
		static void validated_call(Variant *r_ret, const Variant **p_args, int p_argcount) {                            \
			Callable::CallError ce;                                                                                     \
			*r_ret = VariantUtilityFunctions::m_func(*p_args[0], ce);                                                   \
		}                                                                                                               \
		static void ptrcall(void *ret, const void **p_args, int p_argcount) {                                           \
			Callable::CallError ce;                                                                                     \
			PtrToArg<Variant>::encode(VariantUtilityFunctions::m_func(PtrToArg<Variant>::convert(p_args[0]), ce), ret); \
		}                                                                                                               \
		static int get_argument_count() {                                                                               \
			return 1;                                                                                                   \
		}                                                                                                               \
		static Variant::Type get_argument_type(int p_arg) {                                                             \
			return Variant::NIL;                                                                                        \
		}                                                                                                               \
		static Variant::Type get_return_type() {                                                                        \
			return Variant::NIL;                                                                                        \
		}                                                                                                               \
		static bool has_return_type() {                                                                                 \
			return true;                                                                                                \
		}                                                                                                               \
		static bool is_vararg() { return false; }                                                                       \
		static Variant::UtilityFunctionType get_type() { return m_category; }                                           \
	};                                                                                                                  \
	register_utility_function<Func_##m_func>(#m_func, m_args)

#define FUNCBINDVR3(m_func, m_args, m_category)                                                                                                                           \
	class Func_##m_func {                                                                                                                                                 \
	public:                                                                                                                                                               \
		static void call(Variant *r_ret, const Variant **p_args, int p_argcount, Callable::CallError &r_error) {                                                          \
			r_error.error = Callable::CallError::CALL_OK;                                                                                                                 \
			*r_ret = VariantUtilityFunctions::m_func(*p_args[0], *p_args[1], *p_args[2], r_error);                                                                        \
		}                                                                                                                                                                 \
		static void validated_call(Variant *r_ret, const Variant **p_args, int p_argcount) {                                                                              \
			Callable::CallError ce;                                                                                                                                       \
			*r_ret = VariantUtilityFunctions::m_func(*p_args[0], *p_args[1], *p_args[2], ce);                                                                             \
		}                                                                                                                                                                 \
		static void ptrcall(void *ret, const void **p_args, int p_argcount) {                                                                                             \
			Callable::CallError ce;                                                                                                                                       \
			Variant r;                                                                                                                                                    \
			r = VariantUtilityFunctions::m_func(PtrToArg<Variant>::convert(p_args[0]), PtrToArg<Variant>::convert(p_args[1]), PtrToArg<Variant>::convert(p_args[2]), ce); \
			PtrToArg<Variant>::encode(r, ret);                                                                                                                            \
		}                                                                                                                                                                 \
		static int get_argument_count() {                                                                                                                                 \
			return 3;                                                                                                                                                     \
		}                                                                                                                                                                 \
		static Variant::Type get_argument_type(int p_arg) {                                                                                                               \
			return Variant::NIL;                                                                                                                                          \
		}                                                                                                                                                                 \
		static Variant::Type get_return_type() {                                                                                                                          \
			return Variant::NIL;                                                                                                                                          \
		}                                                                                                                                                                 \
		static bool has_return_type() {                                                                                                                                   \
			return true;                                                                                                                                                  \
		}                                                                                                                                                                 \
		static bool is_vararg() { return false; }                                                                                                                         \
		static Variant::UtilityFunctionType get_type() { return m_category; }                                                                                             \
	};                                                                                                                                                                    \
	register_utility_function<Func_##m_func>(#m_func, m_args)

#define FUNCBINDVARARG(m_func, m_args, m_category)                                                               \
	class Func_##m_func {                                                                                        \
	public:                                                                                                      \
		static void call(Variant *r_ret, const Variant **p_args, int p_argcount, Callable::CallError &r_error) { \
			r_error.error = Callable::CallError::CALL_OK;                                                        \
			*r_ret = VariantUtilityFunctions::m_func(p_args, p_argcount, r_error);                               \
		}                                                                                                        \
		static void validated_call(Variant *r_ret, const Variant **p_args, int p_argcount) {                     \
			Callable::CallError c;                                                                               \
			*r_ret = VariantUtilityFunctions::m_func(p_args, p_argcount, c);                                     \
		}                                                                                                        \
		static void ptrcall(void *ret, const void **p_args, int p_argcount) {                                    \
			Vector<Variant> args;                                                                                \
			for (int i = 0; i < p_argcount; i++) {                                                               \
				args.push_back(PtrToArg<Variant>::convert(p_args[i]));                                           \
			}                                                                                                    \
			Vector<const Variant *> argsp;                                                                       \
			for (int i = 0; i < p_argcount; i++) {                                                               \
				argsp.push_back(&args[i]);                                                                       \
			}                                                                                                    \
			Variant r;                                                                                           \
			validated_call(&r, (const Variant **)argsp.ptr(), p_argcount);                                       \
			PtrToArg<Variant>::encode(r, ret);                                                                   \
		}                                                                                                        \
		static int get_argument_count() {                                                                        \
			return 2;                                                                                            \
		}                                                                                                        \
		static Variant::Type get_argument_type(int p_arg) {                                                      \
			return Variant::NIL;                                                                                 \
		}                                                                                                        \
		static Variant::Type get_return_type() {                                                                 \
			return Variant::NIL;                                                                                 \
		}                                                                                                        \
		static bool has_return_type() {                                                                          \
			return true;                                                                                         \
		}                                                                                                        \
		static bool is_vararg() {                                                                                \
			return true;                                                                                         \
		}                                                                                                        \
		static Variant::UtilityFunctionType get_type() {                                                         \
			return m_category;                                                                                   \
		}                                                                                                        \
	};                                                                                                           \
	register_utility_function<Func_##m_func>(#m_func, m_args)

#define FUNCBINDVARARGS(m_func, m_args, m_category)                                                              \
	class Func_##m_func {                                                                                        \
	public:                                                                                                      \
		static void call(Variant *r_ret, const Variant **p_args, int p_argcount, Callable::CallError &r_error) { \
			r_error.error = Callable::CallError::CALL_OK;                                                        \
			*r_ret = VariantUtilityFunctions::m_func(p_args, p_argcount, r_error);                               \
		}                                                                                                        \
		static void validated_call(Variant *r_ret, const Variant **p_args, int p_argcount) {                     \
			Callable::CallError c;                                                                               \
			*r_ret = VariantUtilityFunctions::m_func(p_args, p_argcount, c);                                     \
		}                                                                                                        \
		static void ptrcall(void *ret, const void **p_args, int p_argcount) {                                    \
			Vector<Variant> args;                                                                                \
			for (int i = 0; i < p_argcount; i++) {                                                               \
				args.push_back(PtrToArg<Variant>::convert(p_args[i]));                                           \
			}                                                                                                    \
			Vector<const Variant *> argsp;                                                                       \
			for (int i = 0; i < p_argcount; i++) {                                                               \
				argsp.push_back(&args[i]);                                                                       \
			}                                                                                                    \
			Variant r;                                                                                           \
			validated_call(&r, (const Variant **)argsp.ptr(), p_argcount);                                       \
			PtrToArg<String>::encode(r.operator String(), ret);                                                  \
		}                                                                                                        \
		static int get_argument_count() {                                                                        \
			return 1;                                                                                            \
		}                                                                                                        \
		static Variant::Type get_argument_type(int p_arg) {                                                      \
			return Variant::NIL;                                                                                 \
		}                                                                                                        \
		static Variant::Type get_return_type() {                                                                 \
			return Variant::STRING;                                                                              \
		}                                                                                                        \
		static bool has_return_type() {                                                                          \
			return true;                                                                                         \
		}                                                                                                        \
		static bool is_vararg() {                                                                                \
			return true;                                                                                         \
		}                                                                                                        \
		static Variant::UtilityFunctionType get_type() {                                                         \
			return m_category;                                                                                   \
		}                                                                                                        \
	};                                                                                                           \
	register_utility_function<Func_##m_func>(#m_func, m_args)

#define FUNCBINDVARARGV(m_func, m_args, m_category)                                                              \
	class Func_##m_func {                                                                                        \
	public:                                                                                                      \
		static void call(Variant *r_ret, const Variant **p_args, int p_argcount, Callable::CallError &r_error) { \
			r_error.error = Callable::CallError::CALL_OK;                                                        \
			VariantUtilityFunctions::m_func(p_args, p_argcount, r_error);                                        \
		}                                                                                                        \
		static void validated_call(Variant *r_ret, const Variant **p_args, int p_argcount) {                     \
			Callable::CallError c;                                                                               \
			VariantUtilityFunctions::m_func(p_args, p_argcount, c);                                              \
		}                                                                                                        \
		static void ptrcall(void *ret, const void **p_args, int p_argcount) {                                    \
			Vector<Variant> args;                                                                                \
			for (int i = 0; i < p_argcount; i++) {                                                               \
				args.push_back(PtrToArg<Variant>::convert(p_args[i]));                                           \
			}                                                                                                    \
			Vector<const Variant *> argsp;                                                                       \
			for (int i = 0; i < p_argcount; i++) {                                                               \
				argsp.push_back(&args[i]);                                                                       \
			}                                                                                                    \
			Variant r;                                                                                           \
			validated_call(&r, (const Variant **)argsp.ptr(), p_argcount);                                       \
		}                                                                                                        \
		static int get_argument_count() {                                                                        \
			return 1;                                                                                            \
		}                                                                                                        \
		static Variant::Type get_argument_type(int p_arg) {                                                      \
			return Variant::NIL;                                                                                 \
		}                                                                                                        \
		static Variant::Type get_return_type() {                                                                 \
			return Variant::NIL;                                                                                 \
		}                                                                                                        \
		static bool has_return_type() {                                                                          \
			return false;                                                                                        \
		}                                                                                                        \
		static bool is_vararg() {                                                                                \
			return true;                                                                                         \
		}                                                                                                        \
		static Variant::UtilityFunctionType get_type() {                                                         \
			return m_category;                                                                                   \
		}                                                                                                        \
	};                                                                                                           \
	register_utility_function<Func_##m_func>(#m_func, m_args)

#define FUNCBIND(m_func, m_args, m_category)                                                                     \
	class Func_##m_func {                                                                                        \
	public:                                                                                                      \
		static void call(Variant *r_ret, const Variant **p_args, int p_argcount, Callable::CallError &r_error) { \
			call_helper(VariantUtilityFunctions::m_func, p_args, r_error);                                       \
		}                                                                                                        \
		static void validated_call(Variant *r_ret, const Variant **p_args, int p_argcount) {                     \
			validated_call_helper(VariantUtilityFunctions::m_func, p_args);                                      \
		}                                                                                                        \
		static void ptrcall(void *ret, const void **p_args, int p_argcount) {                                    \
			ptr_call_helper(VariantUtilityFunctions::m_func, p_args);                                            \
		}                                                                                                        \
		static int get_argument_count() {                                                                        \
			return get_arg_count_helper(VariantUtilityFunctions::m_func);                                        \
		}                                                                                                        \
		static Variant::Type get_argument_type(int p_arg) {                                                      \
			return get_arg_type_helper(VariantUtilityFunctions::m_func, p_arg);                                  \
		}                                                                                                        \
		static Variant::Type get_return_type() {                                                                 \
			return get_ret_type_helper(VariantUtilityFunctions::m_func);                                         \
		}                                                                                                        \
		static bool has_return_type() {                                                                          \
			return false;                                                                                        \
		}                                                                                                        \
		static bool is_vararg() { return false; }                                                                \
		static Variant::UtilityFunctionType get_type() { return m_category; }                                    \
	};                                                                                                           \
	register_utility_function<Func_##m_func>(#m_func, m_args)

struct VariantUtilityFunctionInfo {
	void (*call_utility)(Variant *r_ret, const Variant **p_args, int p_argcount, Callable::CallError &r_error);
	Variant::ValidatedUtilityFunction validated_call_utility;
	Variant::PTRUtilityFunction ptr_call_utility;
	Vector<String> argnames;
	bool is_vararg;
	bool returns_value;
	int argcount;
	Variant::Type (*get_arg_type)(int);
	Variant::Type return_type;
	Variant::UtilityFunctionType type;
};

static OAHashMap<StringName, VariantUtilityFunctionInfo> utility_function_table;
static List<StringName> utility_function_name_table;

template <class T>
static void register_utility_function(const String &p_name, const Vector<String> &argnames) {
	String name = p_name;
	if (name.begins_with("_")) {
		name = name.substr(1, name.length() - 1);
	}
	StringName sname = name;
	ERR_FAIL_COND(utility_function_table.has(sname));

	VariantUtilityFunctionInfo bfi;
	bfi.call_utility = T::call;
	bfi.validated_call_utility = T::validated_call;
	bfi.ptr_call_utility = T::ptrcall;
	bfi.is_vararg = T::is_vararg();
	bfi.argnames = argnames;
	bfi.argcount = T::get_argument_count();
	if (!bfi.is_vararg) {
		ERR_FAIL_COND_MSG(argnames.size() != bfi.argcount, "wrong number of arguments binding utility function: " + name);
	}
	bfi.get_arg_type = T::get_argument_type;
	bfi.return_type = T::get_return_type();
	bfi.type = T::get_type();
	bfi.returns_value = T::has_return_type();

	utility_function_table.insert(sname, bfi);
	utility_function_name_table.push_back(sname);
}

void Variant::_register_variant_utility_functions() {
	// Math

	FUNCBINDR(sin, sarray("angle_rad"), Variant::UTILITY_FUNC_TYPE_MATH);
	FUNCBINDR(cos, sarray("angle_rad"), Variant::UTILITY_FUNC_TYPE_MATH);
	FUNCBINDR(tan, sarray("angle_rad"), Variant::UTILITY_FUNC_TYPE_MATH);

	FUNCBINDR(sinh, sarray("x"), Variant::UTILITY_FUNC_TYPE_MATH);
	FUNCBINDR(cosh, sarray("x"), Variant::UTILITY_FUNC_TYPE_MATH);
	FUNCBINDR(tanh, sarray("x"), Variant::UTILITY_FUNC_TYPE_MATH);

	FUNCBINDR(asin, sarray("x"), Variant::UTILITY_FUNC_TYPE_MATH);
	FUNCBINDR(acos, sarray("x"), Variant::UTILITY_FUNC_TYPE_MATH);
	FUNCBINDR(atan, sarray("x"), Variant::UTILITY_FUNC_TYPE_MATH);

	FUNCBINDR(atan2, sarray("y", "x"), Variant::UTILITY_FUNC_TYPE_MATH);

	FUNCBINDR(sqrt, sarray("x"), Variant::UTILITY_FUNC_TYPE_MATH);
	FUNCBINDR(fmod, sarray("x", "y"), Variant::UTILITY_FUNC_TYPE_MATH);
	FUNCBINDR(fposmod, sarray("x", "y"), Variant::UTILITY_FUNC_TYPE_MATH);
	FUNCBINDR(posmod, sarray("x", "y"), Variant::UTILITY_FUNC_TYPE_MATH);
	FUNCBINDR(floor, sarray("x"), Variant::UTILITY_FUNC_TYPE_MATH);
	FUNCBINDR(ceil, sarray("x"), Variant::UTILITY_FUNC_TYPE_MATH);
	FUNCBINDR(round, sarray("x"), Variant::UTILITY_FUNC_TYPE_MATH);

	FUNCBINDVR(abs, sarray("x"), Variant::UTILITY_FUNC_TYPE_MATH);

	FUNCBINDR(absf, sarray("x"), Variant::UTILITY_FUNC_TYPE_MATH);
	FUNCBINDR(absi, sarray("x"), Variant::UTILITY_FUNC_TYPE_MATH);

	FUNCBINDVR(sign, sarray("x"), Variant::UTILITY_FUNC_TYPE_MATH);

	FUNCBINDR(signf, sarray("x"), Variant::UTILITY_FUNC_TYPE_MATH);
	FUNCBINDR(signi, sarray("x"), Variant::UTILITY_FUNC_TYPE_MATH);

	FUNCBINDR(pow, sarray("base", "exp"), Variant::UTILITY_FUNC_TYPE_MATH);
	FUNCBINDR(log, sarray("x"), Variant::UTILITY_FUNC_TYPE_MATH);
	FUNCBINDR(exp, sarray("x"), Variant::UTILITY_FUNC_TYPE_MATH);

	FUNCBINDR(is_nan, sarray("x"), Variant::UTILITY_FUNC_TYPE_MATH);
	FUNCBINDR(is_inf, sarray("x"), Variant::UTILITY_FUNC_TYPE_MATH);

	FUNCBINDR(is_equal_approx, sarray("a", "b"), Variant::UTILITY_FUNC_TYPE_MATH);
	FUNCBINDR(is_zero_approx, sarray("x"), Variant::UTILITY_FUNC_TYPE_MATH);

	FUNCBINDR(ease, sarray("x", "curve"), Variant::UTILITY_FUNC_TYPE_MATH);
	FUNCBINDR(step_decimals, sarray("x"), Variant::UTILITY_FUNC_TYPE_MATH);
	FUNCBINDR(range_step_decimals, sarray("x"), Variant::UTILITY_FUNC_TYPE_MATH);
	FUNCBINDR(snapped, sarray("x", "step"), Variant::UTILITY_FUNC_TYPE_MATH);

	FUNCBINDR(lerp, sarray("from", "to", "weight"), Variant::UTILITY_FUNC_TYPE_MATH);
	FUNCBINDR(lerp_angle, sarray("from", "to", "weight"), Variant::UTILITY_FUNC_TYPE_MATH);
	FUNCBINDR(inverse_lerp, sarray("from", "to", "weight"), Variant::UTILITY_FUNC_TYPE_MATH);
	FUNCBINDR(range_lerp, sarray("value", "istart", "istop", "ostart", "ostop"), Variant::UTILITY_FUNC_TYPE_MATH);

	FUNCBINDR(smoothstep, sarray("from", "to", "x"), Variant::UTILITY_FUNC_TYPE_MATH);
	FUNCBINDR(move_toward, sarray("from", "to", "delta"), Variant::UTILITY_FUNC_TYPE_MATH);

	FUNCBINDR(deg2rad, sarray("deg"), Variant::UTILITY_FUNC_TYPE_MATH);
	FUNCBINDR(rad2deg, sarray("rad"), Variant::UTILITY_FUNC_TYPE_MATH);
	FUNCBINDR(linear2db, sarray("lin"), Variant::UTILITY_FUNC_TYPE_MATH);
	FUNCBINDR(db2linear, sarray("db"), Variant::UTILITY_FUNC_TYPE_MATH);

	FUNCBINDR(wrapi, sarray("value", "min", "max"), Variant::UTILITY_FUNC_TYPE_MATH);
	FUNCBINDR(wrapf, sarray("value", "min", "max"), Variant::UTILITY_FUNC_TYPE_MATH);

	FUNCBINDVARARG(max, sarray(), Variant::UTILITY_FUNC_TYPE_MATH);

	FUNCBINDR(maxi, sarray("a", "b"), Variant::UTILITY_FUNC_TYPE_MATH);
	FUNCBINDR(maxf, sarray("a", "b"), Variant::UTILITY_FUNC_TYPE_MATH);

	FUNCBINDVARARG(min, sarray(), Variant::UTILITY_FUNC_TYPE_MATH);

	FUNCBINDR(mini, sarray("a", "b"), Variant::UTILITY_FUNC_TYPE_MATH);
	FUNCBINDR(minf, sarray("a", "b"), Variant::UTILITY_FUNC_TYPE_MATH);

	FUNCBINDVR3(clamp, sarray("value", "min", "max"), Variant::UTILITY_FUNC_TYPE_MATH);
	FUNCBINDR(clampi, sarray("value", "min", "max"), Variant::UTILITY_FUNC_TYPE_MATH);
	FUNCBINDR(clampf, sarray("value", "min", "max"), Variant::UTILITY_FUNC_TYPE_MATH);

	FUNCBINDR(nearest_po2, sarray("value"), Variant::UTILITY_FUNC_TYPE_MATH);
	FUNCBINDR(pingpong, sarray("value", "length"), Variant::UTILITY_FUNC_TYPE_MATH);

	// Random

	FUNCBIND(randomize, sarray(), Variant::UTILITY_FUNC_TYPE_RANDOM);
	FUNCBINDR(randi, sarray(), Variant::UTILITY_FUNC_TYPE_RANDOM);
	FUNCBINDR(randf, sarray(), Variant::UTILITY_FUNC_TYPE_RANDOM);
	FUNCBINDR(randi_range, sarray("from", "to"), Variant::UTILITY_FUNC_TYPE_RANDOM);
	FUNCBINDR(randf_range, sarray("from", "to"), Variant::UTILITY_FUNC_TYPE_RANDOM);
	FUNCBIND(seed, sarray("base"), Variant::UTILITY_FUNC_TYPE_RANDOM);
	FUNCBINDR(rand_from_seed, sarray("seed"), Variant::UTILITY_FUNC_TYPE_RANDOM);

	// Utility

	FUNCBINDVR(weakref, sarray("obj"), Variant::UTILITY_FUNC_TYPE_GENERAL);
	FUNCBINDR(_typeof, sarray("variable"), Variant::UTILITY_FUNC_TYPE_GENERAL);
	FUNCBINDVARARGS(str, sarray(), Variant::UTILITY_FUNC_TYPE_GENERAL);
	FUNCBINDR(error_string, sarray("error"), Variant::UTILITY_FUNC_TYPE_GENERAL);
	FUNCBINDVARARGV(print, sarray(), Variant::UTILITY_FUNC_TYPE_GENERAL);
	FUNCBINDVARARGV(printerr, sarray(), Variant::UTILITY_FUNC_TYPE_GENERAL);
	FUNCBINDVARARGV(printt, sarray(), Variant::UTILITY_FUNC_TYPE_GENERAL);
	FUNCBINDVARARGV(prints, sarray(), Variant::UTILITY_FUNC_TYPE_GENERAL);
	FUNCBINDVARARGV(printraw, sarray(), Variant::UTILITY_FUNC_TYPE_GENERAL);
	FUNCBINDVARARGV(print_verbose, sarray(), Variant::UTILITY_FUNC_TYPE_GENERAL);
	FUNCBINDVARARGV(push_error, sarray(), Variant::UTILITY_FUNC_TYPE_GENERAL);
	FUNCBINDVARARGV(push_warning, sarray(), Variant::UTILITY_FUNC_TYPE_GENERAL);

	FUNCBINDR(var2str, sarray("variable"), Variant::UTILITY_FUNC_TYPE_GENERAL);
	FUNCBINDR(str2var, sarray("string"), Variant::UTILITY_FUNC_TYPE_GENERAL);

	FUNCBINDR(var2bytes, sarray("variable"), Variant::UTILITY_FUNC_TYPE_GENERAL);
	FUNCBINDR(bytes2var, sarray("bytes"), Variant::UTILITY_FUNC_TYPE_GENERAL);

	FUNCBINDR(var2bytes_with_objects, sarray("variable"), Variant::UTILITY_FUNC_TYPE_GENERAL);
	FUNCBINDR(bytes2var_with_objects, sarray("bytes"), Variant::UTILITY_FUNC_TYPE_GENERAL);

	FUNCBINDR(hash, sarray("variable"), Variant::UTILITY_FUNC_TYPE_GENERAL);

	FUNCBINDR(instance_from_id, sarray("instance_id"), Variant::UTILITY_FUNC_TYPE_GENERAL);
	FUNCBINDR(is_instance_id_valid, sarray("id"), Variant::UTILITY_FUNC_TYPE_GENERAL);
	FUNCBINDR(is_instance_valid, sarray("instance"), Variant::UTILITY_FUNC_TYPE_GENERAL);

	FUNCBINDR(rid_allocate_id, Vector<String>(), Variant::UTILITY_FUNC_TYPE_GENERAL);
	FUNCBINDR(rid_from_int64, sarray("base"), Variant::UTILITY_FUNC_TYPE_GENERAL);
}

void Variant::_unregister_variant_utility_functions() {
	utility_function_table.clear();
	utility_function_name_table.clear();
}

void Variant::call_utility_function(const StringName &p_name, Variant *r_ret, const Variant **p_args, int p_argcount, Callable::CallError &r_error) {
	const VariantUtilityFunctionInfo *bfi = utility_function_table.lookup_ptr(p_name);
	if (!bfi) {
		r_error.error = Callable::CallError::CALL_ERROR_INVALID_METHOD;
		r_error.argument = 0;
		r_error.expected = 0;
		return;
	}

	if (unlikely(!bfi->is_vararg && p_argcount < bfi->argcount)) {
		r_error.error = Callable::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS;
		r_error.argument = 0;
		r_error.expected = bfi->argcount;
		return;
	}

	if (unlikely(!bfi->is_vararg && p_argcount > bfi->argcount)) {
		r_error.error = Callable::CallError::CALL_ERROR_TOO_MANY_ARGUMENTS;
		r_error.argument = 0;
		r_error.expected = bfi->argcount;
		return;
	}

	bfi->call_utility(r_ret, p_args, p_argcount, r_error);
}

bool Variant::has_utility_function(const StringName &p_name) {
	return utility_function_table.has(p_name);
}

Variant::ValidatedUtilityFunction Variant::get_validated_utility_function(const StringName &p_name) {
	const VariantUtilityFunctionInfo *bfi = utility_function_table.lookup_ptr(p_name);
	if (!bfi) {
		return nullptr;
	}

	return bfi->validated_call_utility;
}

Variant::PTRUtilityFunction Variant::get_ptr_utility_function(const StringName &p_name) {
	const VariantUtilityFunctionInfo *bfi = utility_function_table.lookup_ptr(p_name);
	if (!bfi) {
		return nullptr;
	}

	return bfi->ptr_call_utility;
}

Variant::UtilityFunctionType Variant::get_utility_function_type(const StringName &p_name) {
	const VariantUtilityFunctionInfo *bfi = utility_function_table.lookup_ptr(p_name);
	if (!bfi) {
		return Variant::UTILITY_FUNC_TYPE_MATH;
	}

	return bfi->type;
}

MethodInfo Variant::get_utility_function_info(const StringName &p_name) {
	MethodInfo info;
	const VariantUtilityFunctionInfo *bfi = utility_function_table.lookup_ptr(p_name);
	if (bfi) {
		info.name = p_name;
		if (bfi->returns_value && bfi->return_type == Variant::NIL) {
			info.return_val.usage |= PROPERTY_USAGE_NIL_IS_VARIANT;
		}
		info.return_val.type = bfi->return_type;
		if (bfi->is_vararg) {
			info.flags |= METHOD_FLAG_VARARG;
		}
		for (int i = 0; i < bfi->argnames.size(); ++i) {
			PropertyInfo arg;
			arg.type = bfi->get_arg_type(i);
			arg.name = bfi->argnames[i];
			info.arguments.push_back(arg);
		}
	}
	return info;
}

int Variant::get_utility_function_argument_count(const StringName &p_name) {
	const VariantUtilityFunctionInfo *bfi = utility_function_table.lookup_ptr(p_name);
	if (!bfi) {
		return 0;
	}

	return bfi->argcount;
}

Variant::Type Variant::get_utility_function_argument_type(const StringName &p_name, int p_arg) {
	const VariantUtilityFunctionInfo *bfi = utility_function_table.lookup_ptr(p_name);
	if (!bfi) {
		return Variant::NIL;
	}

	return bfi->get_arg_type(p_arg);
}

String Variant::get_utility_function_argument_name(const StringName &p_name, int p_arg) {
	const VariantUtilityFunctionInfo *bfi = utility_function_table.lookup_ptr(p_name);
	if (!bfi) {
		return String();
	}
	ERR_FAIL_INDEX_V(p_arg, bfi->argnames.size(), String());
	ERR_FAIL_COND_V(bfi->is_vararg, String());
	return bfi->argnames[p_arg];
}

bool Variant::has_utility_function_return_value(const StringName &p_name) {
	const VariantUtilityFunctionInfo *bfi = utility_function_table.lookup_ptr(p_name);
	if (!bfi) {
		return false;
	}
	return bfi->returns_value;
}

Variant::Type Variant::get_utility_function_return_type(const StringName &p_name) {
	const VariantUtilityFunctionInfo *bfi = utility_function_table.lookup_ptr(p_name);
	if (!bfi) {
		return Variant::NIL;
	}

	return bfi->return_type;
}

bool Variant::is_utility_function_vararg(const StringName &p_name) {
	const VariantUtilityFunctionInfo *bfi = utility_function_table.lookup_ptr(p_name);
	if (!bfi) {
		return false;
	}

	return bfi->is_vararg;
}

uint32_t Variant::get_utility_function_hash(const StringName &p_name) {
	const VariantUtilityFunctionInfo *bfi = utility_function_table.lookup_ptr(p_name);
	ERR_FAIL_COND_V(!bfi, 0);

	uint32_t hash = hash_djb2_one_32(bfi->is_vararg);
	hash = hash_djb2_one_32(bfi->returns_value, hash);
	if (bfi->returns_value) {
		hash = hash_djb2_one_32(bfi->return_type, hash);
	}
	hash = hash_djb2_one_32(bfi->argcount, hash);
	for (int i = 0; i < bfi->argcount; i++) {
		hash = hash_djb2_one_32(bfi->get_arg_type(i), hash);
	}

	return hash;
}

void Variant::get_utility_function_list(List<StringName> *r_functions) {
	for (const StringName &E : utility_function_name_table) {
		r_functions->push_back(E);
	}
}

int Variant::get_utility_function_count() {
	return utility_function_name_table.size();
}
