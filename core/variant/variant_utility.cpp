/**************************************************************************/
/*  variant_utility.cpp                                                   */
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

#include "variant_utility.h"

#include "core/io/marshalls.h"
#include "core/object/ref_counted.h"
#include "core/os/os.h"
#include "core/templates/a_hash_map.h"
#include "core/templates/rid.h"
#include "core/templates/rid_owner.h"
#include "core/variant/binder_common.h"
#include "core/variant/variant_parser.h"

// Math
double VariantUtilityFunctions::sin(double p_arg) {
	return Math::sin(p_arg);
}

double VariantUtilityFunctions::cos(double p_arg) {
	return Math::cos(p_arg);
}

double VariantUtilityFunctions::tan(double p_arg) {
	return Math::tan(p_arg);
}

double VariantUtilityFunctions::sinh(double p_arg) {
	return Math::sinh(p_arg);
}

double VariantUtilityFunctions::cosh(double p_arg) {
	return Math::cosh(p_arg);
}

double VariantUtilityFunctions::tanh(double p_arg) {
	return Math::tanh(p_arg);
}

double VariantUtilityFunctions::asin(double p_arg) {
	return Math::asin(p_arg);
}

double VariantUtilityFunctions::acos(double p_arg) {
	return Math::acos(p_arg);
}

double VariantUtilityFunctions::atan(double p_arg) {
	return Math::atan(p_arg);
}

double VariantUtilityFunctions::atan2(double p_y, double p_x) {
	return Math::atan2(p_y, p_x);
}

double VariantUtilityFunctions::asinh(double p_arg) {
	return Math::asinh(p_arg);
}

double VariantUtilityFunctions::acosh(double p_arg) {
	return Math::acosh(p_arg);
}

double VariantUtilityFunctions::atanh(double p_arg) {
	return Math::atanh(p_arg);
}

double VariantUtilityFunctions::sqrt(double p_x) {
	return Math::sqrt(p_x);
}

double VariantUtilityFunctions::fmod(double p_b, double p_r) {
	return Math::fmod(p_b, p_r);
}

double VariantUtilityFunctions::fposmod(double p_b, double p_r) {
	return Math::fposmod(p_b, p_r);
}

int64_t VariantUtilityFunctions::posmod(int64_t p_b, int64_t p_r) {
	return Math::posmod(p_b, p_r);
}

Variant VariantUtilityFunctions::floor(const Variant &p_x, Callable::CallError &r_error) {
	r_error.error = Callable::CallError::CALL_OK;
	switch (p_x.get_type()) {
		case Variant::INT: {
			return VariantInternalAccessor<int64_t>::get(&p_x);
		} break;
		case Variant::FLOAT: {
			return Math::floor(VariantInternalAccessor<double>::get(&p_x));
		} break;
		case Variant::VECTOR2: {
			return VariantInternalAccessor<Vector2>::get(&p_x).floor();
		} break;
		case Variant::VECTOR2I: {
			return VariantInternalAccessor<Vector2i>::get(&p_x);
		} break;
		case Variant::VECTOR3: {
			return VariantInternalAccessor<Vector3>::get(&p_x).floor();
		} break;
		case Variant::VECTOR3I: {
			return VariantInternalAccessor<Vector3i>::get(&p_x);
		} break;
		case Variant::VECTOR4: {
			return VariantInternalAccessor<Vector4>::get(&p_x).floor();
		} break;
		case Variant::VECTOR4I: {
			return VariantInternalAccessor<Vector4i>::get(&p_x);
		} break;
		default: {
			r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
			r_error.argument = 0;
			r_error.expected = Variant::NIL;
			return R"(Argument "x" must be "int", "float", "Vector2", "Vector2i", "Vector3", "Vector3i", "Vector4", or "Vector4i".)";
		} break;
	}
}

double VariantUtilityFunctions::floorf(double p_x) {
	return Math::floor(p_x);
}

int64_t VariantUtilityFunctions::floori(double p_x) {
	return int64_t(Math::floor(p_x));
}

Variant VariantUtilityFunctions::ceil(const Variant &p_x, Callable::CallError &r_error) {
	r_error.error = Callable::CallError::CALL_OK;
	switch (p_x.get_type()) {
		case Variant::INT: {
			return VariantInternalAccessor<int64_t>::get(&p_x);
		} break;
		case Variant::FLOAT: {
			return Math::ceil(VariantInternalAccessor<double>::get(&p_x));
		} break;
		case Variant::VECTOR2: {
			return VariantInternalAccessor<Vector2>::get(&p_x).ceil();
		} break;
		case Variant::VECTOR2I: {
			return VariantInternalAccessor<Vector2i>::get(&p_x);
		} break;
		case Variant::VECTOR3: {
			return VariantInternalAccessor<Vector3>::get(&p_x).ceil();
		} break;
		case Variant::VECTOR3I: {
			return VariantInternalAccessor<Vector3i>::get(&p_x);
		} break;
		case Variant::VECTOR4: {
			return VariantInternalAccessor<Vector4>::get(&p_x).ceil();
		} break;
		case Variant::VECTOR4I: {
			return VariantInternalAccessor<Vector4i>::get(&p_x);
		} break;
		default: {
			r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
			r_error.argument = 0;
			r_error.expected = Variant::NIL;
			return R"(Argument "x" must be "int", "float", "Vector2", "Vector2i", "Vector3", "Vector3i", "Vector4", or "Vector4i".)";
		} break;
	}
}

double VariantUtilityFunctions::ceilf(double p_x) {
	return Math::ceil(p_x);
}

int64_t VariantUtilityFunctions::ceili(double p_x) {
	return int64_t(Math::ceil(p_x));
}

Variant VariantUtilityFunctions::round(const Variant &p_x, Callable::CallError &r_error) {
	r_error.error = Callable::CallError::CALL_OK;
	switch (p_x.get_type()) {
		case Variant::INT: {
			return VariantInternalAccessor<int64_t>::get(&p_x);
		} break;
		case Variant::FLOAT: {
			return Math::round(VariantInternalAccessor<double>::get(&p_x));
		} break;
		case Variant::VECTOR2: {
			return VariantInternalAccessor<Vector2>::get(&p_x).round();
		} break;
		case Variant::VECTOR2I: {
			return VariantInternalAccessor<Vector2i>::get(&p_x);
		} break;
		case Variant::VECTOR3: {
			return VariantInternalAccessor<Vector3>::get(&p_x).round();
		} break;
		case Variant::VECTOR3I: {
			return VariantInternalAccessor<Vector3i>::get(&p_x);
		} break;
		case Variant::VECTOR4: {
			return VariantInternalAccessor<Vector4>::get(&p_x).round();
		} break;
		case Variant::VECTOR4I: {
			return VariantInternalAccessor<Vector4i>::get(&p_x);
		} break;
		default: {
			r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
			r_error.argument = 0;
			r_error.expected = Variant::NIL;
			return R"(Argument "x" must be "int", "float", "Vector2", "Vector2i", "Vector3", "Vector3i", "Vector4", or "Vector4i".)";
		} break;
	}
}

double VariantUtilityFunctions::roundf(double p_x) {
	return Math::round(p_x);
}

int64_t VariantUtilityFunctions::roundi(double p_x) {
	return int64_t(Math::round(p_x));
}

Variant VariantUtilityFunctions::abs(const Variant &p_x, Callable::CallError &r_error) {
	r_error.error = Callable::CallError::CALL_OK;
	switch (p_x.get_type()) {
		case Variant::INT: {
			return Math::abs(VariantInternalAccessor<int64_t>::get(&p_x));
		} break;
		case Variant::FLOAT: {
			return Math::abs(VariantInternalAccessor<double>::get(&p_x));
		} break;
		case Variant::VECTOR2: {
			return VariantInternalAccessor<Vector2>::get(&p_x).abs();
		} break;
		case Variant::VECTOR2I: {
			return VariantInternalAccessor<Vector2i>::get(&p_x).abs();
		} break;
		case Variant::VECTOR3: {
			return VariantInternalAccessor<Vector3>::get(&p_x).abs();
		} break;
		case Variant::VECTOR3I: {
			return VariantInternalAccessor<Vector3i>::get(&p_x).abs();
		} break;
		case Variant::VECTOR4: {
			return VariantInternalAccessor<Vector4>::get(&p_x).abs();
		} break;
		case Variant::VECTOR4I: {
			return VariantInternalAccessor<Vector4i>::get(&p_x).abs();
		} break;
		default: {
			r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
			r_error.argument = 0;
			r_error.expected = Variant::NIL;
			return R"(Argument "x" must be "int", "float", "Vector2", "Vector2i", "Vector3", "Vector3i", "Vector4", or "Vector4i".)";
		} break;
	}
}

double VariantUtilityFunctions::absf(double p_x) {
	return Math::abs(p_x);
}

int64_t VariantUtilityFunctions::absi(int64_t p_x) {
	return Math::abs(p_x);
}

Variant VariantUtilityFunctions::sign(const Variant &p_x, Callable::CallError &r_error) {
	r_error.error = Callable::CallError::CALL_OK;
	switch (p_x.get_type()) {
		case Variant::INT: {
			return SIGN(VariantInternalAccessor<int64_t>::get(&p_x));
		} break;
		case Variant::FLOAT: {
			return SIGN(VariantInternalAccessor<double>::get(&p_x));
		} break;
		case Variant::VECTOR2: {
			return VariantInternalAccessor<Vector2>::get(&p_x).sign();
		} break;
		case Variant::VECTOR2I: {
			return VariantInternalAccessor<Vector2i>::get(&p_x).sign();
		} break;
		case Variant::VECTOR3: {
			return VariantInternalAccessor<Vector3>::get(&p_x).sign();
		} break;
		case Variant::VECTOR3I: {
			return VariantInternalAccessor<Vector3i>::get(&p_x).sign();
		} break;
		case Variant::VECTOR4: {
			return VariantInternalAccessor<Vector4>::get(&p_x).sign();
		} break;
		case Variant::VECTOR4I: {
			return VariantInternalAccessor<Vector4i>::get(&p_x).sign();
		} break;
		default: {
			r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
			r_error.argument = 0;
			r_error.expected = Variant::NIL;
			return R"(Argument "x" must be "int", "float", "Vector2", "Vector2i", "Vector3", "Vector3i", "Vector4", or "Vector4i".)";
		} break;
	}
}

double VariantUtilityFunctions::signf(double p_x) {
	return SIGN(p_x);
}

int64_t VariantUtilityFunctions::signi(int64_t p_x) {
	return SIGN(p_x);
}

double VariantUtilityFunctions::pow(double p_x, double p_y) {
	return Math::pow(p_x, p_y);
}

double VariantUtilityFunctions::log(double p_x) {
	return Math::log(p_x);
}

double VariantUtilityFunctions::exp(double p_x) {
	return Math::exp(p_x);
}

bool VariantUtilityFunctions::is_nan(double p_x) {
	return Math::is_nan(p_x);
}

bool VariantUtilityFunctions::is_inf(double p_x) {
	return Math::is_inf(p_x);
}

bool VariantUtilityFunctions::is_equal_approx(double p_x, double p_y) {
	return Math::is_equal_approx(p_x, p_y);
}

bool VariantUtilityFunctions::is_zero_approx(double p_x) {
	return Math::is_zero_approx(p_x);
}

bool VariantUtilityFunctions::is_finite(double p_x) {
	return Math::is_finite(p_x);
}

double VariantUtilityFunctions::ease(double p_x, double p_curve) {
	return Math::ease(p_x, p_curve);
}

int VariantUtilityFunctions::step_decimals(double p_step) {
	return Math::step_decimals(p_step);
}

Variant VariantUtilityFunctions::snapped(const Variant &p_x, const Variant &p_step, Callable::CallError &r_error) {
	switch (p_x.get_type()) {
		case Variant::INT:
		case Variant::FLOAT:
		case Variant::VECTOR2:
		case Variant::VECTOR2I:
		case Variant::VECTOR3:
		case Variant::VECTOR3I:
		case Variant::VECTOR4:
		case Variant::VECTOR4I:
			break;
		default:
			r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
			r_error.argument = 0;
			r_error.expected = Variant::NIL;
			return R"(Argument "x" must be "int", "float", "Vector2", "Vector2i", "Vector3", "Vector3i", "Vector4", or "Vector4i".)";
	}

	if (p_x.get_type() != p_step.get_type()) {
		if (p_x.get_type() == Variant::INT || p_x.get_type() == Variant::FLOAT) {
			if (p_step.get_type() != Variant::INT && p_step.get_type() != Variant::FLOAT) {
				r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
				r_error.argument = 1;
				r_error.expected = Variant::NIL;
				return R"(Argument "step" must be "int" or "float".)";
			}
		} else {
			r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
			r_error.argument = 1;
			r_error.expected = p_x.get_type();
			return Variant();
		}
	}

	r_error.error = Callable::CallError::CALL_OK;
	switch (p_step.get_type()) {
		case Variant::INT: {
			return snappedi(p_x, VariantInternalAccessor<int64_t>::get(&p_step));
		} break;
		case Variant::FLOAT: {
			return snappedf(p_x, VariantInternalAccessor<double>::get(&p_step));
		} break;
		case Variant::VECTOR2: {
			return VariantInternalAccessor<Vector2>::get(&p_x).snapped(VariantInternalAccessor<Vector2>::get(&p_step));
		} break;
		case Variant::VECTOR2I: {
			return VariantInternalAccessor<Vector2i>::get(&p_x).snapped(VariantInternalAccessor<Vector2i>::get(&p_step));
		} break;
		case Variant::VECTOR3: {
			return VariantInternalAccessor<Vector3>::get(&p_x).snapped(VariantInternalAccessor<Vector3>::get(&p_step));
		} break;
		case Variant::VECTOR3I: {
			return VariantInternalAccessor<Vector3i>::get(&p_x).snapped(VariantInternalAccessor<Vector3i>::get(&p_step));
		} break;
		case Variant::VECTOR4: {
			return VariantInternalAccessor<Vector4>::get(&p_x).snapped(VariantInternalAccessor<Vector4>::get(&p_step));
		} break;
		case Variant::VECTOR4I: {
			return VariantInternalAccessor<Vector4i>::get(&p_x).snapped(VariantInternalAccessor<Vector4i>::get(&p_step));
		} break;
		default: {
			return Variant(); // Already handled.
		} break;
	}
}

double VariantUtilityFunctions::snappedf(double p_x, double p_step) {
	return Math::snapped(p_x, p_step);
}

int64_t VariantUtilityFunctions::snappedi(double p_x, int64_t p_step) {
	return Math::snapped(p_x, p_step);
}

Variant VariantUtilityFunctions::lerp(const Variant &p_from, const Variant &p_to, double p_weight, Callable::CallError &r_error) {
	switch (p_from.get_type()) {
		case Variant::INT:
		case Variant::FLOAT:
		case Variant::VECTOR2:
		case Variant::VECTOR3:
		case Variant::VECTOR4:
		case Variant::QUATERNION:
		case Variant::BASIS:
		case Variant::COLOR:
		case Variant::TRANSFORM2D:
		case Variant::TRANSFORM3D:
			break;
		default:
			r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
			r_error.argument = 0;
			r_error.expected = Variant::NIL;
			return R"(Argument "from" must be "int", "float", "Vector2", "Vector3", "Vector4", "Color", "Quaternion", "Basis", "Transform2D", or "Transform3D".)";
	}

	if (p_from.get_type() != p_to.get_type()) {
		r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
		r_error.argument = 1;
		r_error.expected = p_from.get_type();
		return Variant();
	}

	r_error.error = Callable::CallError::CALL_OK;
	switch (p_from.get_type()) {
		case Variant::INT: {
			return lerpf(VariantInternalAccessor<int64_t>::get(&p_from), p_to, p_weight);
		} break;
		case Variant::FLOAT: {
			return lerpf(VariantInternalAccessor<double>::get(&p_from), p_to, p_weight);
		} break;
		case Variant::VECTOR2: {
			return VariantInternalAccessor<Vector2>::get(&p_from).lerp(VariantInternalAccessor<Vector2>::get(&p_to), p_weight);
		} break;
		case Variant::VECTOR3: {
			return VariantInternalAccessor<Vector3>::get(&p_from).lerp(VariantInternalAccessor<Vector3>::get(&p_to), p_weight);
		} break;
		case Variant::VECTOR4: {
			return VariantInternalAccessor<Vector4>::get(&p_from).lerp(VariantInternalAccessor<Vector4>::get(&p_to), p_weight);
		} break;
		case Variant::QUATERNION: {
			return VariantInternalAccessor<Quaternion>::get(&p_from).slerp(VariantInternalAccessor<Quaternion>::get(&p_to), p_weight);
		} break;
		case Variant::BASIS: {
			return VariantInternalAccessor<Basis>::get(&p_from).slerp(VariantInternalAccessor<Basis>::get(&p_to), p_weight);
		} break;
		case Variant::TRANSFORM2D: {
			return VariantInternalAccessor<Transform2D>::get(&p_from).interpolate_with(VariantInternalAccessor<Transform2D>::get(&p_to), p_weight);
		} break;
		case Variant::TRANSFORM3D: {
			return VariantInternalAccessor<Transform3D>::get(&p_from).interpolate_with(VariantInternalAccessor<Transform3D>::get(&p_to), p_weight);
		} break;
		case Variant::COLOR: {
			return VariantInternalAccessor<Color>::get(&p_from).lerp(VariantInternalAccessor<Color>::get(&p_to), p_weight);
		} break;
		default: {
			return Variant(); // Already handled.
		} break;
	}
}

double VariantUtilityFunctions::lerpf(double p_from, double p_to, double p_weight) {
	return Math::lerp(p_from, p_to, p_weight);
}

double VariantUtilityFunctions::cubic_interpolate(double p_from, double p_to, double p_pre, double p_post, double p_weight) {
	return Math::cubic_interpolate(p_from, p_to, p_pre, p_post, p_weight);
}

double VariantUtilityFunctions::cubic_interpolate_angle(double p_from, double p_to, double p_pre, double p_post, double p_weight) {
	return Math::cubic_interpolate_angle(p_from, p_to, p_pre, p_post, p_weight);
}

double VariantUtilityFunctions::cubic_interpolate_in_time(double p_from, double p_to, double p_pre, double p_post, double p_weight,
		double p_to_t, double p_pre_t, double p_post_t) {
	return Math::cubic_interpolate_in_time(p_from, p_to, p_pre, p_post, p_weight, p_to_t, p_pre_t, p_post_t);
}

double VariantUtilityFunctions::cubic_interpolate_angle_in_time(double p_from, double p_to, double p_pre, double p_post, double p_weight,
		double p_to_t, double p_pre_t, double p_post_t) {
	return Math::cubic_interpolate_angle_in_time(p_from, p_to, p_pre, p_post, p_weight, p_to_t, p_pre_t, p_post_t);
}

double VariantUtilityFunctions::bezier_interpolate(double p_start, double p_control_1, double p_control_2, double p_end, double p_t) {
	return Math::bezier_interpolate(p_start, p_control_1, p_control_2, p_end, p_t);
}

double VariantUtilityFunctions::bezier_derivative(double p_start, double p_control_1, double p_control_2, double p_end, double p_t) {
	return Math::bezier_derivative(p_start, p_control_1, p_control_2, p_end, p_t);
}

double VariantUtilityFunctions::angle_difference(double p_from, double p_to) {
	return Math::angle_difference(p_from, p_to);
}

double VariantUtilityFunctions::lerp_angle(double p_from, double p_to, double p_weight) {
	return Math::lerp_angle(p_from, p_to, p_weight);
}

double VariantUtilityFunctions::inverse_lerp(double p_from, double p_to, double p_weight) {
	return Math::inverse_lerp(p_from, p_to, p_weight);
}

double VariantUtilityFunctions::remap(double p_value, double p_istart, double p_istop, double p_ostart, double p_ostop) {
	return Math::remap(p_value, p_istart, p_istop, p_ostart, p_ostop);
}

double VariantUtilityFunctions::smoothstep(double p_from, double p_to, double p_step) {
	return Math::smoothstep(p_from, p_to, p_step);
}

double VariantUtilityFunctions::move_toward(double p_from, double p_to, double p_delta) {
	return Math::move_toward(p_from, p_to, p_delta);
}

double VariantUtilityFunctions::rotate_toward(double p_from, double p_to, double p_delta) {
	return Math::rotate_toward(p_from, p_to, p_delta);
}

double VariantUtilityFunctions::deg_to_rad(double p_angle_deg) {
	return Math::deg_to_rad(p_angle_deg);
}

double VariantUtilityFunctions::rad_to_deg(double p_angle_rad) {
	return Math::rad_to_deg(p_angle_rad);
}

double VariantUtilityFunctions::linear_to_db(double p_linear) {
	return Math::linear_to_db(p_linear);
}

double VariantUtilityFunctions::db_to_linear(double p_db) {
	return Math::db_to_linear(p_db);
}

Variant VariantUtilityFunctions::wrap(const Variant &p_x, const Variant &p_min, const Variant &p_max, Callable::CallError &r_error) {
	Variant::Type x_type = p_x.get_type();
	if (x_type != Variant::INT && x_type != Variant::FLOAT) {
		r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
		r_error.argument = 0;
		r_error.expected = Variant::FLOAT;
		return Variant();
	}

	Variant::Type min_type = p_min.get_type();
	if (min_type != Variant::INT && min_type != Variant::FLOAT) {
		r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
		r_error.argument = 1;
		r_error.expected = x_type;
		return Variant();
	}

	Variant::Type max_type = p_max.get_type();
	if (max_type != Variant::INT && max_type != Variant::FLOAT) {
		r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
		r_error.argument = 2;
		r_error.expected = x_type;
		return Variant();
	}

	Variant value;

	switch (x_type) {
		case Variant::INT: {
			if (x_type != min_type || x_type != max_type) {
				value = wrapf((double)p_x, (double)p_min, (double)p_max);
			} else {
				value = wrapi((int)p_x, (int)p_min, (int)p_max);
			}
		} break;
		case Variant::FLOAT: {
			value = wrapf((double)p_x, (double)p_min, (double)p_max);
		} break;
		default:
			break;
	}

	r_error.error = Callable::CallError::CALL_OK;
	return value;
}

int64_t VariantUtilityFunctions::wrapi(int64_t p_value, int64_t p_min, int64_t p_max) {
	return Math::wrapi(p_value, p_min, p_max);
}

double VariantUtilityFunctions::wrapf(double p_value, double p_min, double p_max) {
	return Math::wrapf(p_value, p_min, p_max);
}

double VariantUtilityFunctions::pingpong(double p_value, double p_length) {
	return Math::pingpong(p_value, p_length);
}

int VariantUtilityFunctions::intdiv(int p_a, int p_b) {
	return p_a / p_b;
}

Variant VariantUtilityFunctions::max(const Variant **p_args, int p_argcount, Callable::CallError &r_error) {
	if (p_argcount < 2) {
		r_error.error = Callable::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS;
		r_error.expected = 2;
		return Variant();
	}
	Variant base = *p_args[0];
	Variant ret;

	for (int i = 0; i < p_argcount; i++) {
		Variant::Type arg_type = p_args[i]->get_type();
		if (arg_type != Variant::INT && arg_type != Variant::FLOAT) {
			r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
			r_error.argument = i;
			r_error.expected = Variant::FLOAT;
			return Variant();
		}
		if (i == 0) {
			continue;
		}
		bool valid;
		Variant::evaluate(Variant::OP_LESS, base, *p_args[i], ret, valid);
		if (!valid) {
			r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
			r_error.argument = i;
			r_error.expected = base.get_type();
			return Variant();
		}
		if (ret.booleanize()) {
			base = *p_args[i];
		}
	}
	r_error.error = Callable::CallError::CALL_OK;
	return base;
}

double VariantUtilityFunctions::maxf(double p_x, double p_y) {
	return MAX(p_x, p_y);
}

int64_t VariantUtilityFunctions::maxi(int64_t p_x, int64_t p_y) {
	return MAX(p_x, p_y);
}

Variant VariantUtilityFunctions::min(const Variant **p_args, int p_argcount, Callable::CallError &r_error) {
	if (p_argcount < 2) {
		r_error.error = Callable::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS;
		r_error.expected = 2;
		return Variant();
	}
	Variant base = *p_args[0];
	Variant ret;

	for (int i = 0; i < p_argcount; i++) {
		Variant::Type arg_type = p_args[i]->get_type();
		if (arg_type != Variant::INT && arg_type != Variant::FLOAT) {
			r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
			r_error.argument = i;
			r_error.expected = Variant::FLOAT;
			return Variant();
		}
		if (i == 0) {
			continue;
		}
		bool valid;
		Variant::evaluate(Variant::OP_GREATER, base, *p_args[i], ret, valid);
		if (!valid) {
			r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
			r_error.argument = i;
			r_error.expected = base.get_type();
			return Variant();
		}
		if (ret.booleanize()) {
			base = *p_args[i];
		}
	}
	r_error.error = Callable::CallError::CALL_OK;
	return base;
}

double VariantUtilityFunctions::minf(double p_x, double p_y) {
	return MIN(p_x, p_y);
}

int64_t VariantUtilityFunctions::mini(int64_t p_x, int64_t p_y) {
	return MIN(p_x, p_y);
}

Variant VariantUtilityFunctions::clamp(const Variant &p_x, const Variant &p_min, const Variant &p_max, Callable::CallError &r_error) {
	Variant value = p_x;

	Variant ret;

	bool valid;
	Variant::evaluate(Variant::OP_LESS, value, p_min, ret, valid);
	if (!valid) {
		r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
		r_error.argument = 1;
		r_error.expected = value.get_type();
		return Variant();
	}
	if (ret.booleanize()) {
		value = p_min;
	}
	Variant::evaluate(Variant::OP_GREATER, value, p_max, ret, valid);
	if (!valid) {
		r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
		r_error.argument = 2;
		r_error.expected = value.get_type();
		return Variant();
	}
	if (ret.booleanize()) {
		value = p_max;
	}

	r_error.error = Callable::CallError::CALL_OK;

	return value;
}

double VariantUtilityFunctions::clampf(double p_x, double p_min, double p_max) {
	return CLAMP(p_x, p_min, p_max);
}

int64_t VariantUtilityFunctions::clampi(int64_t p_x, int64_t p_min, int64_t p_max) {
	return CLAMP(p_x, p_min, p_max);
}

int64_t VariantUtilityFunctions::nearest_po2(int64_t p_x) {
	return Math::nearest_power_of_2_templated(uint64_t(p_x));
}

// Random

void VariantUtilityFunctions::randomize() {
	Math::randomize();
}

int64_t VariantUtilityFunctions::randi() {
	return Math::rand();
}

double VariantUtilityFunctions::randf() {
	return Math::randf();
}

double VariantUtilityFunctions::randfn(double p_mean, double p_deviation) {
	return Math::randfn(p_mean, p_deviation);
}

int64_t VariantUtilityFunctions::randi_range(int64_t p_from, int64_t p_to) {
	return Math::random((int32_t)p_from, (int32_t)p_to);
}

double VariantUtilityFunctions::randf_range(double p_from, double p_to) {
	return Math::random(p_from, p_to);
}

void VariantUtilityFunctions::seed(int64_t p_seed) {
	return Math::seed(p_seed);
}

PackedInt64Array VariantUtilityFunctions::rand_from_seed(int64_t p_seed) {
	uint64_t s = p_seed;
	PackedInt64Array arr;
	arr.resize(2);
	arr.write[0] = Math::rand_from_seed(&s);
	arr.write[1] = s;
	return arr;
}

// Utility

Variant VariantUtilityFunctions::weakref(const Variant &p_obj, Callable::CallError &r_error) {
	if (p_obj.get_type() == Variant::OBJECT) {
		r_error.error = Callable::CallError::CALL_OK;
		if (p_obj.is_ref_counted()) {
			Ref<WeakRef> wref = memnew(WeakRef);
			Ref<RefCounted> r = p_obj;
			if (r.is_valid()) {
				wref->set_ref(r);
			}
			return wref;
		} else {
			Ref<WeakRef> wref = memnew(WeakRef);
			Object *o = p_obj.get_validated_object();
			if (o) {
				wref->set_obj(o);
			}
			return wref;
		}
	} else if (p_obj.get_type() == Variant::NIL) {
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

int64_t VariantUtilityFunctions::_typeof(const Variant &p_obj) {
	return p_obj.get_type();
}

Variant VariantUtilityFunctions::type_convert(const Variant &p_variant, const Variant::Type p_type) {
	switch (p_type) {
		case Variant::Type::NIL:
			return Variant();
		case Variant::Type::BOOL:
			return p_variant.operator bool();
		case Variant::Type::INT:
			return p_variant.operator int64_t();
		case Variant::Type::FLOAT:
			return p_variant.operator double();
		case Variant::Type::STRING:
			return p_variant.operator String();
		case Variant::Type::VECTOR2:
			return p_variant.operator Vector2();
		case Variant::Type::VECTOR2I:
			return p_variant.operator Vector2i();
		case Variant::Type::RECT2:
			return p_variant.operator Rect2();
		case Variant::Type::RECT2I:
			return p_variant.operator Rect2i();
		case Variant::Type::VECTOR3:
			return p_variant.operator Vector3();
		case Variant::Type::VECTOR3I:
			return p_variant.operator Vector3i();
		case Variant::Type::TRANSFORM2D:
			return p_variant.operator Transform2D();
		case Variant::Type::VECTOR4:
			return p_variant.operator Vector4();
		case Variant::Type::VECTOR4I:
			return p_variant.operator Vector4i();
		case Variant::Type::PLANE:
			return p_variant.operator Plane();
		case Variant::Type::QUATERNION:
			return p_variant.operator Quaternion();
		case Variant::Type::AABB:
			return p_variant.operator ::AABB();
		case Variant::Type::BASIS:
			return p_variant.operator Basis();
		case Variant::Type::TRANSFORM3D:
			return p_variant.operator Transform3D();
		case Variant::Type::PROJECTION:
			return p_variant.operator Projection();
		case Variant::Type::COLOR:
			return p_variant.operator Color();
		case Variant::Type::STRING_NAME:
			return p_variant.operator StringName();
		case Variant::Type::NODE_PATH:
			return p_variant.operator NodePath();
		case Variant::Type::RID:
			return p_variant.operator ::RID();
		case Variant::Type::OBJECT:
			return p_variant.operator Object *();
		case Variant::Type::CALLABLE:
			return p_variant.operator Callable();
		case Variant::Type::SIGNAL:
			return p_variant.operator Signal();
		case Variant::Type::DICTIONARY:
			return p_variant.operator Dictionary();
		case Variant::Type::ARRAY:
			return p_variant.operator Array();
		case Variant::Type::PACKED_BYTE_ARRAY:
			return p_variant.operator PackedByteArray();
		case Variant::Type::PACKED_INT32_ARRAY:
			return p_variant.operator PackedInt32Array();
		case Variant::Type::PACKED_INT64_ARRAY:
			return p_variant.operator PackedInt64Array();
		case Variant::Type::PACKED_FLOAT32_ARRAY:
			return p_variant.operator PackedFloat32Array();
		case Variant::Type::PACKED_FLOAT64_ARRAY:
			return p_variant.operator PackedFloat64Array();
		case Variant::Type::PACKED_STRING_ARRAY:
			return p_variant.operator PackedStringArray();
		case Variant::Type::PACKED_VECTOR2_ARRAY:
			return p_variant.operator PackedVector2Array();
		case Variant::Type::PACKED_VECTOR3_ARRAY:
			return p_variant.operator PackedVector3Array();
		case Variant::Type::PACKED_COLOR_ARRAY:
			return p_variant.operator PackedColorArray();
		case Variant::Type::PACKED_VECTOR4_ARRAY:
			return p_variant.operator PackedVector4Array();
		case Variant::Type::VARIANT_MAX:
			ERR_PRINT("Invalid type argument p_to type_convert(), use the TYPE_* constants. Returning the unconverted Variant.");
	}
	return p_variant;
}

String VariantUtilityFunctions::str(const Variant **p_args, int p_arg_count, Callable::CallError &r_error) {
	if (p_arg_count < 1) {
		r_error.error = Callable::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS;
		r_error.expected = 1;
		return String();
	}

	r_error.error = Callable::CallError::CALL_OK;

	return join_string(p_args, p_arg_count);
}

String VariantUtilityFunctions::error_string(Error p_error) {
	if (p_error < 0 || p_error >= ERR_MAX) {
		return String("(invalid error code)");
	}

	return String(error_names[p_error]);
}

String VariantUtilityFunctions::type_string(Variant::Type p_type) {
	ERR_FAIL_INDEX_V_MSG((int)p_type, (int)Variant::VARIANT_MAX, "<invalid type>", "Invalid type argument p_to type_string(), use the TYPE_* constants.");
	return Variant::get_type_name(p_type);
}

void VariantUtilityFunctions::print(const Variant **p_args, int p_arg_count, Callable::CallError &r_error) {
	print_line(join_string(p_args, p_arg_count));
	r_error.error = Callable::CallError::CALL_OK;
}

void VariantUtilityFunctions::print_rich(const Variant **p_args, int p_arg_count, Callable::CallError &r_error) {
	print_line_rich(join_string(p_args, p_arg_count));
	r_error.error = Callable::CallError::CALL_OK;
}

void VariantUtilityFunctions::_print_verbose(const Variant **p_args, int p_arg_count, Callable::CallError &r_error) {
	if (OS::get_singleton()->is_stdout_verbose()) {
		// No need p_to use `print_verbose()` as this call already only happens
		// when verbose mode is enabled. This avoids performing string argument concatenation
		// when not needed.
		print_line(join_string(p_args, p_arg_count));
	}

	r_error.error = Callable::CallError::CALL_OK;
}

void VariantUtilityFunctions::printerr(const Variant **p_args, int p_arg_count, Callable::CallError &r_error) {
	print_error(join_string(p_args, p_arg_count));
	r_error.error = Callable::CallError::CALL_OK;
}

void VariantUtilityFunctions::printt(const Variant **p_args, int p_arg_count, Callable::CallError &r_error) {
	String s;
	for (int i = 0; i < p_arg_count; i++) {
		if (i) {
			s += "\t";
		}
		s += p_args[i]->operator String();
	}

	print_line(s);
	r_error.error = Callable::CallError::CALL_OK;
}

void VariantUtilityFunctions::prints(const Variant **p_args, int p_arg_count, Callable::CallError &r_error) {
	String s;
	for (int i = 0; i < p_arg_count; i++) {
		if (i) {
			s += " ";
		}
		s += p_args[i]->operator String();
	}

	print_line(s);
	r_error.error = Callable::CallError::CALL_OK;
}

void VariantUtilityFunctions::printraw(const Variant **p_args, int p_arg_count, Callable::CallError &r_error) {
	print_raw(join_string(p_args, p_arg_count));
	r_error.error = Callable::CallError::CALL_OK;
}

void VariantUtilityFunctions::push_error(const Variant **p_args, int p_arg_count, Callable::CallError &r_error) {
	if (p_arg_count < 1) {
		r_error.error = Callable::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS;
		r_error.expected = 1;
	}

	ERR_PRINT(join_string(p_args, p_arg_count));
	r_error.error = Callable::CallError::CALL_OK;
}

void VariantUtilityFunctions::push_warning(const Variant **p_args, int p_arg_count, Callable::CallError &r_error) {
	if (p_arg_count < 1) {
		r_error.error = Callable::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS;
		r_error.expected = 1;
	}

	WARN_PRINT(join_string(p_args, p_arg_count));
	r_error.error = Callable::CallError::CALL_OK;
}

String VariantUtilityFunctions::var_to_str(const Variant &p_var) {
	String vars;
	VariantWriter::write_to_string(p_var, vars, false);
	return vars;
}

Variant VariantUtilityFunctions::str_to_var(const String &p_var) {
	VariantParser::StreamString ss;
	ss.s = p_var;

	String errs;
	int line;
	Variant ret;
	(void)VariantParser::parse(&ss, ret, errs, line);

	return ret;
}

PackedByteArray VariantUtilityFunctions::var_to_bytes(const Variant &p_var) {
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

PackedByteArray VariantUtilityFunctions::var_to_bytes_with_objects(const Variant &p_var) {
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

Variant VariantUtilityFunctions::bytes_to_var(const PackedByteArray &p_arr) {
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

Variant VariantUtilityFunctions::bytes_to_var_with_objects(const PackedByteArray &p_arr) {
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

int64_t VariantUtilityFunctions::hash(const Variant &p_arr) {
	return p_arr.hash();
}

Object *VariantUtilityFunctions::instance_from_id(int64_t p_id) {
	ObjectID id = ObjectID((uint64_t)p_id);
	Object *ret = ObjectDB::get_instance(id);
	return ret;
}

bool VariantUtilityFunctions::is_instance_id_valid(int64_t p_id) {
	return ObjectDB::get_instance(ObjectID((uint64_t)p_id)) != nullptr;
}

bool VariantUtilityFunctions::is_instance_valid(const Variant &p_instance) {
	if (p_instance.get_type() != Variant::OBJECT) {
		return false;
	}
	return p_instance.get_validated_object() != nullptr;
}

uint64_t VariantUtilityFunctions::rid_allocate_id() {
	return RID_AllocBase::_gen_id();
}

RID VariantUtilityFunctions::rid_from_int64(uint64_t p_base) {
	return RID::from_uint64(p_base);
}

bool VariantUtilityFunctions::is_same(const Variant &p_a, const Variant &p_b) {
	return p_a.identity_compare(p_b);
}

String VariantUtilityFunctions::join_string(const Variant **p_args, int p_arg_count) {
	String s;
	for (int i = 0; i < p_arg_count; i++) {
		String os = p_args[i]->operator String();
		s += os;
	}
	return s;
}

#ifdef DEBUG_ENABLED
#define VCALLR *r_ret = p_func(VariantCasterAndValidate<P>::cast(p_args, Is, r_error)...)
#define VCALL p_func(VariantCasterAndValidate<P>::cast(p_args, Is, r_error)...)
#else
#define VCALLR *r_ret = p_func(VariantCaster<P>::cast(*p_args[Is])...)
#define VCALL p_func(VariantCaster<P>::cast(*p_args[Is])...)
#endif // DEBUG_ENABLED

template <typename R, typename... P, size_t... Is>
static _FORCE_INLINE_ void call_helperpr(R (*p_func)(P...), Variant *r_ret, const Variant **p_args, Callable::CallError &r_error, IndexSequence<Is...>) {
	r_error.error = Callable::CallError::CALL_OK;
	VCALLR;
	(void)p_args; // avoid gcc warning
	(void)r_error;
}

template <typename R, typename... P, size_t... Is>
static _FORCE_INLINE_ void validated_call_helperpr(R (*p_func)(P...), Variant *r_ret, const Variant **p_args, IndexSequence<Is...>) {
	*r_ret = p_func(VariantCaster<P>::cast(*p_args[Is])...);
	(void)p_args;
}

template <typename R, typename... P, size_t... Is>
static _FORCE_INLINE_ void ptr_call_helperpr(R (*p_func)(P...), void *r_ret, const void **p_args, IndexSequence<Is...>) {
	PtrToArg<R>::encode(p_func(PtrToArg<P>::convert(p_args[Is])...), r_ret);
	(void)p_args;
}

template <typename R, typename... P>
static _FORCE_INLINE_ void call_helperr(R (*p_func)(P...), Variant *r_ret, const Variant **p_args, Callable::CallError &r_error) {
	call_helperpr(p_func, r_ret, p_args, r_error, BuildIndexSequence<sizeof...(P)>{});
}

template <typename R, typename... P>
static _FORCE_INLINE_ void validated_call_helperr(R (*p_func)(P...), Variant *r_ret, const Variant **p_args) {
	validated_call_helperpr(p_func, r_ret, p_args, BuildIndexSequence<sizeof...(P)>{});
}

template <typename R, typename... P>
static _FORCE_INLINE_ void ptr_call_helperr(R (*p_func)(P...), void *r_ret, const void **p_args) {
	ptr_call_helperpr(p_func, r_ret, p_args, BuildIndexSequence<sizeof...(P)>{});
}

template <typename R, typename... P>
static _FORCE_INLINE_ int get_arg_count_helperr(R (*p_func)(P...)) {
	return sizeof...(P);
}

template <typename R, typename... P>
static _FORCE_INLINE_ Variant::Type get_arg_type_helperr(R (*p_func)(P...), int p_arg) {
	return call_get_argument_type<P...>(p_arg);
}

template <typename R, typename... P>
static _FORCE_INLINE_ Variant::Type get_ret_type_helperr(R (*p_func)(P...)) {
	return GetTypeInfo<R>::VARIANT_TYPE;
}

// WITHOUT RET

template <typename... P, size_t... Is>
static _FORCE_INLINE_ void call_helperp(void (*p_func)(P...), const Variant **p_args, Callable::CallError &r_error, IndexSequence<Is...>) {
	r_error.error = Callable::CallError::CALL_OK;
	VCALL;
	(void)p_args;
	(void)r_error;
}

template <typename... P, size_t... Is>
static _FORCE_INLINE_ void validated_call_helperp(void (*p_func)(P...), const Variant **p_args, IndexSequence<Is...>) {
	p_func(VariantCaster<P>::cast(*p_args[Is])...);
	(void)p_args;
}

template <typename... P, size_t... Is>
static _FORCE_INLINE_ void ptr_call_helperp(void (*p_func)(P...), const void **p_args, IndexSequence<Is...>) {
	p_func(PtrToArg<P>::convert(p_args[Is])...);
	(void)p_args;
}

template <typename... P>
static _FORCE_INLINE_ void call_helper(void (*p_func)(P...), const Variant **p_args, Callable::CallError &r_error) {
	call_helperp(p_func, p_args, r_error, BuildIndexSequence<sizeof...(P)>{});
}

template <typename... P>
static _FORCE_INLINE_ void validated_call_helper(void (*p_func)(P...), const Variant **p_args) {
	validated_call_helperp(p_func, p_args, BuildIndexSequence<sizeof...(P)>{});
}

template <typename... P>
static _FORCE_INLINE_ void ptr_call_helper(void (*p_func)(P...), const void **p_args) {
	ptr_call_helperp(p_func, p_args, BuildIndexSequence<sizeof...(P)>{});
}

template <typename... P>
static _FORCE_INLINE_ int get_arg_count_helper(void (*p_func)(P...)) {
	return sizeof...(P);
}

template <typename... P>
static _FORCE_INLINE_ Variant::Type get_arg_type_helper(void (*p_func)(P...), int p_arg) {
	return call_get_argument_type<P...>(p_arg);
}

template <typename... P>
static _FORCE_INLINE_ Variant::Type get_ret_type_helper(void (*p_func)(P...)) {
	return Variant::NIL;
}

#define FUNCBINDR(m_func, m_args, m_category) \
	class Func_##m_func { \
	public: \
		static void call(Variant *r_ret, const Variant **p_args, int p_argcount, Callable::CallError &r_error) { \
			call_helperr(VariantUtilityFunctions::m_func, r_ret, p_args, r_error); \
		} \
		static void validated_call(Variant *r_ret, const Variant **p_args, int p_argcount) { \
			validated_call_helperr(VariantUtilityFunctions::m_func, r_ret, p_args); \
		} \
		static void ptrcall(void *r_ret, const void **p_args, int p_argcount) { \
			ptr_call_helperr(VariantUtilityFunctions::m_func, r_ret, p_args); \
		} \
		static int get_argument_count() { \
			return get_arg_count_helperr(VariantUtilityFunctions::m_func); \
		} \
		static Variant::Type get_argument_type(int p_arg) { \
			return get_arg_type_helperr(VariantUtilityFunctions::m_func, p_arg); \
		} \
		static Variant::Type get_return_type() { \
			return get_ret_type_helperr(VariantUtilityFunctions::m_func); \
		} \
		static bool has_return_type() { \
			return true; \
		} \
		static bool is_vararg() { \
			return false; \
		} \
		static Variant::UtilityFunctionType get_type() { \
			return m_category; \
		} \
	}; \
	register_utility_function<Func_##m_func>(#m_func, m_args)

#define FUNCBINDVR(m_func, m_args, m_category) \
	class Func_##m_func { \
	public: \
		static void call(Variant *r_ret, const Variant **p_args, int p_argcount, Callable::CallError &r_error) { \
			r_error.error = Callable::CallError::CALL_OK; \
			*r_ret = VariantUtilityFunctions::m_func(*p_args[0], r_error); \
		} \
		static void validated_call(Variant *r_ret, const Variant **p_args, int p_argcount) { \
			Callable::CallError ce; \
			*r_ret = VariantUtilityFunctions::m_func(*p_args[0], ce); \
		} \
		static void ptrcall(void *r_ret, const void **p_args, int p_argcount) { \
			Callable::CallError ce; \
			PtrToArg<Variant>::encode(VariantUtilityFunctions::m_func(PtrToArg<Variant>::convert(p_args[0]), ce), r_ret); \
		} \
		static int get_argument_count() { \
			return 1; \
		} \
		static Variant::Type get_argument_type(int p_arg) { \
			return Variant::NIL; \
		} \
		static Variant::Type get_return_type() { \
			return Variant::NIL; \
		} \
		static bool has_return_type() { \
			return true; \
		} \
		static bool is_vararg() { \
			return false; \
		} \
		static Variant::UtilityFunctionType get_type() { \
			return m_category; \
		} \
	}; \
	register_utility_function<Func_##m_func>(#m_func, m_args)

#define FUNCBINDVR2(m_func, m_args, m_category) \
	class Func_##m_func { \
	public: \
		static void call(Variant *r_ret, const Variant **p_args, int p_argcount, Callable::CallError &r_error) { \
			r_error.error = Callable::CallError::CALL_OK; \
			*r_ret = VariantUtilityFunctions::m_func(*p_args[0], *p_args[1], r_error); \
		} \
		static void validated_call(Variant *r_ret, const Variant **p_args, int p_argcount) { \
			Callable::CallError ce; \
			*r_ret = VariantUtilityFunctions::m_func(*p_args[0], *p_args[1], ce); \
		} \
		static void ptrcall(void *r_ret, const void **p_args, int p_argcount) { \
			Callable::CallError ce; \
			Variant r; \
			r = VariantUtilityFunctions::m_func(PtrToArg<Variant>::convert(p_args[0]), PtrToArg<Variant>::convert(p_args[1]), ce); \
			PtrToArg<Variant>::encode(r, r_ret); \
		} \
		static int get_argument_count() { \
			return 2; \
		} \
		static Variant::Type get_argument_type(int p_arg) { \
			return Variant::NIL; \
		} \
		static Variant::Type get_return_type() { \
			return Variant::NIL; \
		} \
		static bool has_return_type() { \
			return true; \
		} \
		static bool is_vararg() { \
			return false; \
		} \
		static Variant::UtilityFunctionType get_type() { \
			return m_category; \
		} \
	}; \
	register_utility_function<Func_##m_func>(#m_func, m_args)

#define FUNCBINDVR3(m_func, m_args, m_category) \
	class Func_##m_func { \
	public: \
		static void call(Variant *r_ret, const Variant **p_args, int p_argcount, Callable::CallError &r_error) { \
			r_error.error = Callable::CallError::CALL_OK; \
			*r_ret = VariantUtilityFunctions::m_func(*p_args[0], *p_args[1], *p_args[2], r_error); \
		} \
		static void validated_call(Variant *r_ret, const Variant **p_args, int p_argcount) { \
			Callable::CallError ce; \
			*r_ret = VariantUtilityFunctions::m_func(*p_args[0], *p_args[1], *p_args[2], ce); \
		} \
		static void ptrcall(void *r_ret, const void **p_args, int p_argcount) { \
			Callable::CallError ce; \
			Variant r; \
			r = VariantUtilityFunctions::m_func(PtrToArg<Variant>::convert(p_args[0]), PtrToArg<Variant>::convert(p_args[1]), PtrToArg<Variant>::convert(p_args[2]), ce); \
			PtrToArg<Variant>::encode(r, r_ret); \
		} \
		static int get_argument_count() { \
			return 3; \
		} \
		static Variant::Type get_argument_type(int p_arg) { \
			return Variant::NIL; \
		} \
		static Variant::Type get_return_type() { \
			return Variant::NIL; \
		} \
		static bool has_return_type() { \
			return true; \
		} \
		static bool is_vararg() { \
			return false; \
		} \
		static Variant::UtilityFunctionType get_type() { \
			return m_category; \
		} \
	}; \
	register_utility_function<Func_##m_func>(#m_func, m_args)

#define FUNCBINDVARARG(m_func, m_args, m_category) \
	class Func_##m_func { \
	public: \
		static void call(Variant *r_ret, const Variant **p_args, int p_argcount, Callable::CallError &r_error) { \
			r_error.error = Callable::CallError::CALL_OK; \
			*r_ret = VariantUtilityFunctions::m_func(p_args, p_argcount, r_error); \
		} \
		static void validated_call(Variant *r_ret, const Variant **p_args, int p_argcount) { \
			Callable::CallError c; \
			*r_ret = VariantUtilityFunctions::m_func(p_args, p_argcount, c); \
		} \
		static void ptrcall(void *r_ret, const void **p_args, int p_argcount) { \
			Vector<Variant> args; \
			for (int i = 0; i < p_argcount; i++) { \
				args.push_back(PtrToArg<Variant>::convert(p_args[i])); \
			} \
			Vector<const Variant *> argsp; \
			for (int i = 0; i < p_argcount; i++) { \
				argsp.push_back(&args[i]); \
			} \
			Variant r; \
			validated_call(&r, (const Variant **)argsp.ptr(), p_argcount); \
			PtrToArg<Variant>::encode(r, r_ret); \
		} \
		static int get_argument_count() { \
			return 2; \
		} \
		static Variant::Type get_argument_type(int p_arg) { \
			return Variant::NIL; \
		} \
		static Variant::Type get_return_type() { \
			return Variant::NIL; \
		} \
		static bool has_return_type() { \
			return true; \
		} \
		static bool is_vararg() { \
			return true; \
		} \
		static Variant::UtilityFunctionType get_type() { \
			return m_category; \
		} \
	}; \
	register_utility_function<Func_##m_func>(#m_func, m_args)

#define FUNCBINDVARARGS(m_func, m_args, m_category) \
	class Func_##m_func { \
	public: \
		static void call(Variant *r_ret, const Variant **p_args, int p_argcount, Callable::CallError &r_error) { \
			r_error.error = Callable::CallError::CALL_OK; \
			*r_ret = VariantUtilityFunctions::m_func(p_args, p_argcount, r_error); \
		} \
		static void validated_call(Variant *r_ret, const Variant **p_args, int p_argcount) { \
			Callable::CallError c; \
			*r_ret = VariantUtilityFunctions::m_func(p_args, p_argcount, c); \
		} \
		static void ptrcall(void *r_ret, const void **p_args, int p_argcount) { \
			Vector<Variant> args; \
			for (int i = 0; i < p_argcount; i++) { \
				args.push_back(PtrToArg<Variant>::convert(p_args[i])); \
			} \
			Vector<const Variant *> argsp; \
			for (int i = 0; i < p_argcount; i++) { \
				argsp.push_back(&args[i]); \
			} \
			Variant r; \
			validated_call(&r, (const Variant **)argsp.ptr(), p_argcount); \
			PtrToArg<String>::encode(r.operator String(), r_ret); \
		} \
		static int get_argument_count() { \
			return 1; \
		} \
		static Variant::Type get_argument_type(int p_arg) { \
			return Variant::NIL; \
		} \
		static Variant::Type get_return_type() { \
			return Variant::STRING; \
		} \
		static bool has_return_type() { \
			return true; \
		} \
		static bool is_vararg() { \
			return true; \
		} \
		static Variant::UtilityFunctionType get_type() { \
			return m_category; \
		} \
	}; \
	register_utility_function<Func_##m_func>(#m_func, m_args)

#define FUNCBINDVARARGV_CNAME(m_func, m_func_cname, m_args, m_category) \
	class Func_##m_func { \
	public: \
		static void call(Variant *r_ret, const Variant **p_args, int p_argcount, Callable::CallError &r_error) { \
			r_error.error = Callable::CallError::CALL_OK; \
			VariantUtilityFunctions::m_func_cname(p_args, p_argcount, r_error); \
		} \
		static void validated_call(Variant *r_ret, const Variant **p_args, int p_argcount) { \
			Callable::CallError c; \
			VariantUtilityFunctions::m_func_cname(p_args, p_argcount, c); \
		} \
		static void ptrcall(void *r_ret, const void **p_args, int p_argcount) { \
			Vector<Variant> args; \
			for (int i = 0; i < p_argcount; i++) { \
				args.push_back(PtrToArg<Variant>::convert(p_args[i])); \
			} \
			Vector<const Variant *> argsp; \
			for (int i = 0; i < p_argcount; i++) { \
				argsp.push_back(&args[i]); \
			} \
			Variant r; \
			validated_call(&r, (const Variant **)argsp.ptr(), p_argcount); \
		} \
		static int get_argument_count() { \
			return 1; \
		} \
		static Variant::Type get_argument_type(int p_arg) { \
			return Variant::NIL; \
		} \
		static Variant::Type get_return_type() { \
			return Variant::NIL; \
		} \
		static bool has_return_type() { \
			return false; \
		} \
		static bool is_vararg() { \
			return true; \
		} \
		static Variant::UtilityFunctionType get_type() { \
			return m_category; \
		} \
	}; \
	register_utility_function<Func_##m_func>(#m_func, m_args)

#define FUNCBINDVARARGV(m_func, m_args, m_category) FUNCBINDVARARGV_CNAME(m_func, m_func, m_args, m_category)

#define FUNCBIND(m_func, m_args, m_category) \
	class Func_##m_func { \
	public: \
		static void call(Variant *r_ret, const Variant **p_args, int p_argcount, Callable::CallError &r_error) { \
			call_helper(VariantUtilityFunctions::m_func, p_args, r_error); \
		} \
		static void validated_call(Variant *r_ret, const Variant **p_args, int p_argcount) { \
			validated_call_helper(VariantUtilityFunctions::m_func, p_args); \
		} \
		static void ptrcall(void *r_ret, const void **p_args, int p_argcount) { \
			ptr_call_helper(VariantUtilityFunctions::m_func, p_args); \
		} \
		static int get_argument_count() { \
			return get_arg_count_helper(VariantUtilityFunctions::m_func); \
		} \
		static Variant::Type get_argument_type(int p_arg) { \
			return get_arg_type_helper(VariantUtilityFunctions::m_func, p_arg); \
		} \
		static Variant::Type get_return_type() { \
			return get_ret_type_helper(VariantUtilityFunctions::m_func); \
		} \
		static bool has_return_type() { \
			return false; \
		} \
		static bool is_vararg() { \
			return false; \
		} \
		static Variant::UtilityFunctionType get_type() { \
			return m_category; \
		} \
	}; \
	register_utility_function<Func_##m_func>(#m_func, m_args)

struct VariantUtilityFunctionInfo {
	void (*call_utility)(Variant *r_ret, const Variant **p_args, int p_argcount, Callable::CallError &r_error) = nullptr;
	Variant::ValidatedUtilityFunction validated_call_utility = nullptr;
	Variant::PTRUtilityFunction ptr_call_utility = nullptr;
	Vector<String> argnames;
	bool is_vararg = false;
	bool returns_value = false;
	int argcount = 0;
	Variant::Type (*get_arg_type)(int) = nullptr;
	Variant::Type return_type;
	Variant::UtilityFunctionType type;
};

static AHashMap<StringName, VariantUtilityFunctionInfo> utility_function_table;
static List<StringName> utility_function_name_table;

template <typename T>
static void register_utility_function(const String &p_name, const Vector<String> &p_argnames) {
	String name = p_name;
	if (name.begins_with("_")) {
		name = name.substr(1);
	}
	StringName sname = name;
	ERR_FAIL_COND(utility_function_table.has(sname));

	VariantUtilityFunctionInfo bfi;
	bfi.call_utility = T::call;
	bfi.validated_call_utility = T::validated_call;
	bfi.ptr_call_utility = T::ptrcall;
	bfi.is_vararg = T::is_vararg();
	bfi.argnames = p_argnames;
	bfi.argcount = T::get_argument_count();
	if (!bfi.is_vararg) {
		ERR_FAIL_COND_MSG(p_argnames.size() != bfi.argcount, vformat("Wrong number of arguments binding utility function: '%s'.", name));
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

	FUNCBINDR(asinh, sarray("x"), Variant::UTILITY_FUNC_TYPE_MATH);
	FUNCBINDR(acosh, sarray("x"), Variant::UTILITY_FUNC_TYPE_MATH);
	FUNCBINDR(atanh, sarray("x"), Variant::UTILITY_FUNC_TYPE_MATH);

	FUNCBINDR(sqrt, sarray("x"), Variant::UTILITY_FUNC_TYPE_MATH);
	FUNCBINDR(fmod, sarray("x", "y"), Variant::UTILITY_FUNC_TYPE_MATH);
	FUNCBINDR(fposmod, sarray("x", "y"), Variant::UTILITY_FUNC_TYPE_MATH);
	FUNCBINDR(posmod, sarray("x", "y"), Variant::UTILITY_FUNC_TYPE_MATH);

	FUNCBINDVR(floor, sarray("x"), Variant::UTILITY_FUNC_TYPE_MATH);
	FUNCBINDR(floorf, sarray("x"), Variant::UTILITY_FUNC_TYPE_MATH);
	FUNCBINDR(floori, sarray("x"), Variant::UTILITY_FUNC_TYPE_MATH);

	FUNCBINDVR(ceil, sarray("x"), Variant::UTILITY_FUNC_TYPE_MATH);
	FUNCBINDR(ceilf, sarray("x"), Variant::UTILITY_FUNC_TYPE_MATH);
	FUNCBINDR(ceili, sarray("x"), Variant::UTILITY_FUNC_TYPE_MATH);

	FUNCBINDVR(round, sarray("x"), Variant::UTILITY_FUNC_TYPE_MATH);
	FUNCBINDR(roundf, sarray("x"), Variant::UTILITY_FUNC_TYPE_MATH);
	FUNCBINDR(roundi, sarray("x"), Variant::UTILITY_FUNC_TYPE_MATH);

	FUNCBINDVR(abs, sarray("x"), Variant::UTILITY_FUNC_TYPE_MATH);
	FUNCBINDR(absf, sarray("x"), Variant::UTILITY_FUNC_TYPE_MATH);
	FUNCBINDR(absi, sarray("x"), Variant::UTILITY_FUNC_TYPE_MATH);

	FUNCBINDVR(sign, sarray("x"), Variant::UTILITY_FUNC_TYPE_MATH);
	FUNCBINDR(signf, sarray("x"), Variant::UTILITY_FUNC_TYPE_MATH);
	FUNCBINDR(signi, sarray("x"), Variant::UTILITY_FUNC_TYPE_MATH);

	FUNCBINDVR2(snapped, sarray("x", "step"), Variant::UTILITY_FUNC_TYPE_MATH);
	FUNCBINDR(snappedf, sarray("x", "step"), Variant::UTILITY_FUNC_TYPE_MATH);
	FUNCBINDR(snappedi, sarray("x", "step"), Variant::UTILITY_FUNC_TYPE_MATH);

	FUNCBINDR(pow, sarray("base", "exp"), Variant::UTILITY_FUNC_TYPE_MATH);
	FUNCBINDR(log, sarray("x"), Variant::UTILITY_FUNC_TYPE_MATH);
	FUNCBINDR(exp, sarray("x"), Variant::UTILITY_FUNC_TYPE_MATH);

	FUNCBINDR(is_nan, sarray("x"), Variant::UTILITY_FUNC_TYPE_MATH);
	FUNCBINDR(is_inf, sarray("x"), Variant::UTILITY_FUNC_TYPE_MATH);

	FUNCBINDR(is_equal_approx, sarray("a", "b"), Variant::UTILITY_FUNC_TYPE_MATH);
	FUNCBINDR(is_zero_approx, sarray("x"), Variant::UTILITY_FUNC_TYPE_MATH);
	FUNCBINDR(is_finite, sarray("x"), Variant::UTILITY_FUNC_TYPE_MATH);

	FUNCBINDR(ease, sarray("x", "curve"), Variant::UTILITY_FUNC_TYPE_MATH);
	FUNCBINDR(step_decimals, sarray("x"), Variant::UTILITY_FUNC_TYPE_MATH);

	FUNCBINDVR3(lerp, sarray("from", "to", "weight"), Variant::UTILITY_FUNC_TYPE_MATH);
	FUNCBINDR(lerpf, sarray("from", "to", "weight"), Variant::UTILITY_FUNC_TYPE_MATH);
	FUNCBINDR(cubic_interpolate, sarray("from", "to", "pre", "post", "weight"), Variant::UTILITY_FUNC_TYPE_MATH);
	FUNCBINDR(cubic_interpolate_angle, sarray("from", "to", "pre", "post", "weight"), Variant::UTILITY_FUNC_TYPE_MATH);
	FUNCBINDR(cubic_interpolate_in_time, sarray("from", "to", "pre", "post", "weight", "to_t", "pre_t", "post_t"), Variant::UTILITY_FUNC_TYPE_MATH);
	FUNCBINDR(cubic_interpolate_angle_in_time, sarray("from", "to", "pre", "post", "weight", "to_t", "pre_t", "post_t"), Variant::UTILITY_FUNC_TYPE_MATH);
	FUNCBINDR(bezier_interpolate, sarray("start", "control_1", "control_2", "end", "t"), Variant::UTILITY_FUNC_TYPE_MATH);
	FUNCBINDR(bezier_derivative, sarray("start", "control_1", "control_2", "end", "t"), Variant::UTILITY_FUNC_TYPE_MATH);
	FUNCBINDR(angle_difference, sarray("from", "to"), Variant::UTILITY_FUNC_TYPE_MATH);
	FUNCBINDR(lerp_angle, sarray("from", "to", "weight"), Variant::UTILITY_FUNC_TYPE_MATH);
	FUNCBINDR(inverse_lerp, sarray("from", "to", "weight"), Variant::UTILITY_FUNC_TYPE_MATH);
	FUNCBINDR(remap, sarray("value", "istart", "istop", "ostart", "ostop"), Variant::UTILITY_FUNC_TYPE_MATH);

	FUNCBINDR(smoothstep, sarray("from", "to", "x"), Variant::UTILITY_FUNC_TYPE_MATH);
	FUNCBINDR(move_toward, sarray("from", "to", "delta"), Variant::UTILITY_FUNC_TYPE_MATH);
	FUNCBINDR(rotate_toward, sarray("from", "to", "delta"), Variant::UTILITY_FUNC_TYPE_MATH);

	FUNCBINDR(deg_to_rad, sarray("deg"), Variant::UTILITY_FUNC_TYPE_MATH);
	FUNCBINDR(rad_to_deg, sarray("rad"), Variant::UTILITY_FUNC_TYPE_MATH);
	FUNCBINDR(linear_to_db, sarray("lin"), Variant::UTILITY_FUNC_TYPE_MATH);
	FUNCBINDR(db_to_linear, sarray("db"), Variant::UTILITY_FUNC_TYPE_MATH);

	FUNCBINDVR3(wrap, sarray("value", "min", "max"), Variant::UTILITY_FUNC_TYPE_MATH);
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

	FUNCBINDR(intdiv, sarray("a", "b"), Variant::UTILITY_FUNC_TYPE_MATH);

	// Random

	FUNCBIND(randomize, sarray(), Variant::UTILITY_FUNC_TYPE_RANDOM);
	FUNCBINDR(randi, sarray(), Variant::UTILITY_FUNC_TYPE_RANDOM);
	FUNCBINDR(randf, sarray(), Variant::UTILITY_FUNC_TYPE_RANDOM);
	FUNCBINDR(randi_range, sarray("from", "to"), Variant::UTILITY_FUNC_TYPE_RANDOM);
	FUNCBINDR(randf_range, sarray("from", "to"), Variant::UTILITY_FUNC_TYPE_RANDOM);
	FUNCBINDR(randfn, sarray("mean", "deviation"), Variant::UTILITY_FUNC_TYPE_RANDOM);
	FUNCBIND(seed, sarray("base"), Variant::UTILITY_FUNC_TYPE_RANDOM);
	FUNCBINDR(rand_from_seed, sarray("seed"), Variant::UTILITY_FUNC_TYPE_RANDOM);

	// Utility

	FUNCBINDVR(weakref, sarray("obj"), Variant::UTILITY_FUNC_TYPE_GENERAL);
	FUNCBINDR(_typeof, sarray("variable"), Variant::UTILITY_FUNC_TYPE_GENERAL);
	FUNCBINDR(type_convert, sarray("variant", "type"), Variant::UTILITY_FUNC_TYPE_GENERAL);
	FUNCBINDVARARGS(str, sarray(), Variant::UTILITY_FUNC_TYPE_GENERAL);
	FUNCBINDR(error_string, sarray("error"), Variant::UTILITY_FUNC_TYPE_GENERAL);
	FUNCBINDR(type_string, sarray("type"), Variant::UTILITY_FUNC_TYPE_GENERAL);
	FUNCBINDVARARGV(print, sarray(), Variant::UTILITY_FUNC_TYPE_GENERAL);
	FUNCBINDVARARGV(print_rich, sarray(), Variant::UTILITY_FUNC_TYPE_GENERAL);
	FUNCBINDVARARGV(printerr, sarray(), Variant::UTILITY_FUNC_TYPE_GENERAL);
	FUNCBINDVARARGV(printt, sarray(), Variant::UTILITY_FUNC_TYPE_GENERAL);
	FUNCBINDVARARGV(prints, sarray(), Variant::UTILITY_FUNC_TYPE_GENERAL);
	FUNCBINDVARARGV(printraw, sarray(), Variant::UTILITY_FUNC_TYPE_GENERAL);
	FUNCBINDVARARGV_CNAME(print_verbose, _print_verbose, sarray(), Variant::UTILITY_FUNC_TYPE_GENERAL);
	FUNCBINDVARARGV(push_error, sarray(), Variant::UTILITY_FUNC_TYPE_GENERAL);
	FUNCBINDVARARGV(push_warning, sarray(), Variant::UTILITY_FUNC_TYPE_GENERAL);

	FUNCBINDR(var_to_str, sarray("variable"), Variant::UTILITY_FUNC_TYPE_GENERAL);
	FUNCBINDR(str_to_var, sarray("string"), Variant::UTILITY_FUNC_TYPE_GENERAL);

	FUNCBINDR(var_to_bytes, sarray("variable"), Variant::UTILITY_FUNC_TYPE_GENERAL);
	FUNCBINDR(bytes_to_var, sarray("bytes"), Variant::UTILITY_FUNC_TYPE_GENERAL);

	FUNCBINDR(var_to_bytes_with_objects, sarray("variable"), Variant::UTILITY_FUNC_TYPE_GENERAL);
	FUNCBINDR(bytes_to_var_with_objects, sarray("bytes"), Variant::UTILITY_FUNC_TYPE_GENERAL);

	FUNCBINDR(hash, sarray("variable"), Variant::UTILITY_FUNC_TYPE_GENERAL);

	FUNCBINDR(instance_from_id, sarray("instance_id"), Variant::UTILITY_FUNC_TYPE_GENERAL);
	FUNCBINDR(is_instance_id_valid, sarray("id"), Variant::UTILITY_FUNC_TYPE_GENERAL);
	FUNCBINDR(is_instance_valid, sarray("instance"), Variant::UTILITY_FUNC_TYPE_GENERAL);

	FUNCBINDR(rid_allocate_id, Vector<String>(), Variant::UTILITY_FUNC_TYPE_GENERAL);
	FUNCBINDR(rid_from_int64, sarray("base"), Variant::UTILITY_FUNC_TYPE_GENERAL);

	FUNCBINDR(is_same, sarray("a", "b"), Variant::UTILITY_FUNC_TYPE_GENERAL);
}

void Variant::_unregister_variant_utility_functions() {
	utility_function_table.clear();
	utility_function_name_table.clear();
}

void Variant::call_utility_function(const StringName &p_name, Variant *r_ret, const Variant **p_args, int p_argcount, Callable::CallError &r_error) {
	const VariantUtilityFunctionInfo *bfi = utility_function_table.getptr(p_name);
	if (!bfi) {
		r_error.error = Callable::CallError::CALL_ERROR_INVALID_METHOD;
		r_error.argument = 0;
		r_error.expected = 0;
		return;
	}

	if (unlikely(!bfi->is_vararg && p_argcount < bfi->argcount)) {
		r_error.error = Callable::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS;
		r_error.expected = bfi->argcount;
		return;
	}

	if (unlikely(!bfi->is_vararg && p_argcount > bfi->argcount)) {
		r_error.error = Callable::CallError::CALL_ERROR_TOO_MANY_ARGUMENTS;
		r_error.expected = bfi->argcount;
		return;
	}

	bfi->call_utility(r_ret, p_args, p_argcount, r_error);
}

bool Variant::has_utility_function(const StringName &p_name) {
	return utility_function_table.has(p_name);
}

Variant::ValidatedUtilityFunction Variant::get_validated_utility_function(const StringName &p_name) {
	const VariantUtilityFunctionInfo *bfi = utility_function_table.getptr(p_name);
	if (!bfi) {
		return nullptr;
	}

	return bfi->validated_call_utility;
}

Variant::PTRUtilityFunction Variant::get_ptr_utility_function(const StringName &p_name) {
	const VariantUtilityFunctionInfo *bfi = utility_function_table.getptr(p_name);
	if (!bfi) {
		return nullptr;
	}

	return bfi->ptr_call_utility;
}

Variant::UtilityFunctionType Variant::get_utility_function_type(const StringName &p_name) {
	const VariantUtilityFunctionInfo *bfi = utility_function_table.getptr(p_name);
	if (!bfi) {
		return Variant::UTILITY_FUNC_TYPE_MATH;
	}

	return bfi->type;
}

MethodInfo Variant::get_utility_function_info(const StringName &p_name) {
	MethodInfo info;
	const VariantUtilityFunctionInfo *bfi = utility_function_table.getptr(p_name);
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
			PropertyInfo p_arg;
			p_arg.type = bfi->get_arg_type(i);
			p_arg.name = bfi->argnames[i];
			info.arguments.push_back(p_arg);
		}
	}
	return info;
}

int Variant::get_utility_function_argument_count(const StringName &p_name) {
	const VariantUtilityFunctionInfo *bfi = utility_function_table.getptr(p_name);
	if (!bfi) {
		return 0;
	}

	return bfi->argcount;
}

Variant::Type Variant::get_utility_function_argument_type(const StringName &p_name, int p_arg) {
	const VariantUtilityFunctionInfo *bfi = utility_function_table.getptr(p_name);
	if (!bfi) {
		return Variant::NIL;
	}

	return bfi->get_arg_type(p_arg);
}

String Variant::get_utility_function_argument_name(const StringName &p_name, int p_arg) {
	const VariantUtilityFunctionInfo *bfi = utility_function_table.getptr(p_name);
	if (!bfi) {
		return String();
	}
	ERR_FAIL_INDEX_V(p_arg, bfi->argnames.size(), String());
	ERR_FAIL_COND_V(bfi->is_vararg, String());
	return bfi->argnames[p_arg];
}

bool Variant::has_utility_function_return_value(const StringName &p_name) {
	const VariantUtilityFunctionInfo *bfi = utility_function_table.getptr(p_name);
	if (!bfi) {
		return false;
	}
	return bfi->returns_value;
}

Variant::Type Variant::get_utility_function_return_type(const StringName &p_name) {
	const VariantUtilityFunctionInfo *bfi = utility_function_table.getptr(p_name);
	if (!bfi) {
		return Variant::NIL;
	}

	return bfi->return_type;
}

bool Variant::is_utility_function_vararg(const StringName &p_name) {
	const VariantUtilityFunctionInfo *bfi = utility_function_table.getptr(p_name);
	if (!bfi) {
		return false;
	}

	return bfi->is_vararg;
}

uint32_t Variant::get_utility_function_hash(const StringName &p_name) {
	const VariantUtilityFunctionInfo *bfi = utility_function_table.getptr(p_name);
	ERR_FAIL_NULL_V(bfi, 0);

	uint32_t hash = hash_murmur3_one_32(bfi->is_vararg);
	hash = hash_murmur3_one_32(bfi->returns_value, hash);
	if (bfi->returns_value) {
		hash = hash_murmur3_one_32(bfi->return_type, hash);
	}
	hash = hash_murmur3_one_32(bfi->argcount, hash);
	for (int i = 0; i < bfi->argcount; i++) {
		hash = hash_murmur3_one_32(bfi->get_arg_type(i), hash);
	}

	return hash_fmix32(hash);
}

void Variant::get_utility_function_list(List<StringName> *r_functions) {
	for (const StringName &E : utility_function_name_table) {
		r_functions->push_back(E);
	}
}

int Variant::get_utility_function_count() {
	return utility_function_name_table.size();
}
