/**************************************************************************/
/*  utility_functions.cpp                                                 */
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

// THIS FILE IS GENERATED. EDITS WILL BE LOST.

#include <godot_cpp/variant/utility_functions.hpp>

#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

namespace godot {

double UtilityFunctions::sin(double p_angle_rad) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("sin")._native_ptr(), 2140049587);
	CHECK_METHOD_BIND_RET(_gde_function, (0.0));
	double p_angle_rad_encoded;
	PtrToArg<double>::encode(p_angle_rad, &p_angle_rad_encoded);
	return ::godot::internal::_call_utility_ret<double>(_gde_function, &p_angle_rad_encoded);
}

double UtilityFunctions::cos(double p_angle_rad) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("cos")._native_ptr(), 2140049587);
	CHECK_METHOD_BIND_RET(_gde_function, (0.0));
	double p_angle_rad_encoded;
	PtrToArg<double>::encode(p_angle_rad, &p_angle_rad_encoded);
	return ::godot::internal::_call_utility_ret<double>(_gde_function, &p_angle_rad_encoded);
}

double UtilityFunctions::tan(double p_angle_rad) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("tan")._native_ptr(), 2140049587);
	CHECK_METHOD_BIND_RET(_gde_function, (0.0));
	double p_angle_rad_encoded;
	PtrToArg<double>::encode(p_angle_rad, &p_angle_rad_encoded);
	return ::godot::internal::_call_utility_ret<double>(_gde_function, &p_angle_rad_encoded);
}

double UtilityFunctions::sinh(double p_x) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("sinh")._native_ptr(), 2140049587);
	CHECK_METHOD_BIND_RET(_gde_function, (0.0));
	double p_x_encoded;
	PtrToArg<double>::encode(p_x, &p_x_encoded);
	return ::godot::internal::_call_utility_ret<double>(_gde_function, &p_x_encoded);
}

double UtilityFunctions::cosh(double p_x) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("cosh")._native_ptr(), 2140049587);
	CHECK_METHOD_BIND_RET(_gde_function, (0.0));
	double p_x_encoded;
	PtrToArg<double>::encode(p_x, &p_x_encoded);
	return ::godot::internal::_call_utility_ret<double>(_gde_function, &p_x_encoded);
}

double UtilityFunctions::tanh(double p_x) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("tanh")._native_ptr(), 2140049587);
	CHECK_METHOD_BIND_RET(_gde_function, (0.0));
	double p_x_encoded;
	PtrToArg<double>::encode(p_x, &p_x_encoded);
	return ::godot::internal::_call_utility_ret<double>(_gde_function, &p_x_encoded);
}

double UtilityFunctions::asin(double p_x) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("asin")._native_ptr(), 2140049587);
	CHECK_METHOD_BIND_RET(_gde_function, (0.0));
	double p_x_encoded;
	PtrToArg<double>::encode(p_x, &p_x_encoded);
	return ::godot::internal::_call_utility_ret<double>(_gde_function, &p_x_encoded);
}

double UtilityFunctions::acos(double p_x) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("acos")._native_ptr(), 2140049587);
	CHECK_METHOD_BIND_RET(_gde_function, (0.0));
	double p_x_encoded;
	PtrToArg<double>::encode(p_x, &p_x_encoded);
	return ::godot::internal::_call_utility_ret<double>(_gde_function, &p_x_encoded);
}

double UtilityFunctions::atan(double p_x) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("atan")._native_ptr(), 2140049587);
	CHECK_METHOD_BIND_RET(_gde_function, (0.0));
	double p_x_encoded;
	PtrToArg<double>::encode(p_x, &p_x_encoded);
	return ::godot::internal::_call_utility_ret<double>(_gde_function, &p_x_encoded);
}

double UtilityFunctions::atan2(double p_y, double p_x) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("atan2")._native_ptr(), 92296394);
	CHECK_METHOD_BIND_RET(_gde_function, (0.0));
	double p_y_encoded;
	PtrToArg<double>::encode(p_y, &p_y_encoded);
	double p_x_encoded;
	PtrToArg<double>::encode(p_x, &p_x_encoded);
	return ::godot::internal::_call_utility_ret<double>(_gde_function, &p_y_encoded, &p_x_encoded);
}

double UtilityFunctions::asinh(double p_x) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("asinh")._native_ptr(), 2140049587);
	CHECK_METHOD_BIND_RET(_gde_function, (0.0));
	double p_x_encoded;
	PtrToArg<double>::encode(p_x, &p_x_encoded);
	return ::godot::internal::_call_utility_ret<double>(_gde_function, &p_x_encoded);
}

double UtilityFunctions::acosh(double p_x) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("acosh")._native_ptr(), 2140049587);
	CHECK_METHOD_BIND_RET(_gde_function, (0.0));
	double p_x_encoded;
	PtrToArg<double>::encode(p_x, &p_x_encoded);
	return ::godot::internal::_call_utility_ret<double>(_gde_function, &p_x_encoded);
}

double UtilityFunctions::atanh(double p_x) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("atanh")._native_ptr(), 2140049587);
	CHECK_METHOD_BIND_RET(_gde_function, (0.0));
	double p_x_encoded;
	PtrToArg<double>::encode(p_x, &p_x_encoded);
	return ::godot::internal::_call_utility_ret<double>(_gde_function, &p_x_encoded);
}

double UtilityFunctions::sqrt(double p_x) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("sqrt")._native_ptr(), 2140049587);
	CHECK_METHOD_BIND_RET(_gde_function, (0.0));
	double p_x_encoded;
	PtrToArg<double>::encode(p_x, &p_x_encoded);
	return ::godot::internal::_call_utility_ret<double>(_gde_function, &p_x_encoded);
}

double UtilityFunctions::fmod(double p_x, double p_y) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("fmod")._native_ptr(), 92296394);
	CHECK_METHOD_BIND_RET(_gde_function, (0.0));
	double p_x_encoded;
	PtrToArg<double>::encode(p_x, &p_x_encoded);
	double p_y_encoded;
	PtrToArg<double>::encode(p_y, &p_y_encoded);
	return ::godot::internal::_call_utility_ret<double>(_gde_function, &p_x_encoded, &p_y_encoded);
}

double UtilityFunctions::fposmod(double p_x, double p_y) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("fposmod")._native_ptr(), 92296394);
	CHECK_METHOD_BIND_RET(_gde_function, (0.0));
	double p_x_encoded;
	PtrToArg<double>::encode(p_x, &p_x_encoded);
	double p_y_encoded;
	PtrToArg<double>::encode(p_y, &p_y_encoded);
	return ::godot::internal::_call_utility_ret<double>(_gde_function, &p_x_encoded, &p_y_encoded);
}

int64_t UtilityFunctions::posmod(int64_t p_x, int64_t p_y) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("posmod")._native_ptr(), 3133453818);
	CHECK_METHOD_BIND_RET(_gde_function, (0));
	int64_t p_x_encoded;
	PtrToArg<int64_t>::encode(p_x, &p_x_encoded);
	int64_t p_y_encoded;
	PtrToArg<int64_t>::encode(p_y, &p_y_encoded);
	return ::godot::internal::_call_utility_ret<int64_t>(_gde_function, &p_x_encoded, &p_y_encoded);
}

Variant UtilityFunctions::floor(const Variant &p_x) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("floor")._native_ptr(), 4776452);
	CHECK_METHOD_BIND_RET(_gde_function, (Variant()));
	return ::godot::internal::_call_utility_ret<Variant>(_gde_function, &p_x);
}

double UtilityFunctions::floorf(double p_x) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("floorf")._native_ptr(), 2140049587);
	CHECK_METHOD_BIND_RET(_gde_function, (0.0));
	double p_x_encoded;
	PtrToArg<double>::encode(p_x, &p_x_encoded);
	return ::godot::internal::_call_utility_ret<double>(_gde_function, &p_x_encoded);
}

int64_t UtilityFunctions::floori(double p_x) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("floori")._native_ptr(), 2780425386);
	CHECK_METHOD_BIND_RET(_gde_function, (0));
	double p_x_encoded;
	PtrToArg<double>::encode(p_x, &p_x_encoded);
	return ::godot::internal::_call_utility_ret<int64_t>(_gde_function, &p_x_encoded);
}

Variant UtilityFunctions::ceil(const Variant &p_x) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("ceil")._native_ptr(), 4776452);
	CHECK_METHOD_BIND_RET(_gde_function, (Variant()));
	return ::godot::internal::_call_utility_ret<Variant>(_gde_function, &p_x);
}

double UtilityFunctions::ceilf(double p_x) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("ceilf")._native_ptr(), 2140049587);
	CHECK_METHOD_BIND_RET(_gde_function, (0.0));
	double p_x_encoded;
	PtrToArg<double>::encode(p_x, &p_x_encoded);
	return ::godot::internal::_call_utility_ret<double>(_gde_function, &p_x_encoded);
}

int64_t UtilityFunctions::ceili(double p_x) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("ceili")._native_ptr(), 2780425386);
	CHECK_METHOD_BIND_RET(_gde_function, (0));
	double p_x_encoded;
	PtrToArg<double>::encode(p_x, &p_x_encoded);
	return ::godot::internal::_call_utility_ret<int64_t>(_gde_function, &p_x_encoded);
}

Variant UtilityFunctions::round(const Variant &p_x) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("round")._native_ptr(), 4776452);
	CHECK_METHOD_BIND_RET(_gde_function, (Variant()));
	return ::godot::internal::_call_utility_ret<Variant>(_gde_function, &p_x);
}

double UtilityFunctions::roundf(double p_x) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("roundf")._native_ptr(), 2140049587);
	CHECK_METHOD_BIND_RET(_gde_function, (0.0));
	double p_x_encoded;
	PtrToArg<double>::encode(p_x, &p_x_encoded);
	return ::godot::internal::_call_utility_ret<double>(_gde_function, &p_x_encoded);
}

int64_t UtilityFunctions::roundi(double p_x) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("roundi")._native_ptr(), 2780425386);
	CHECK_METHOD_BIND_RET(_gde_function, (0));
	double p_x_encoded;
	PtrToArg<double>::encode(p_x, &p_x_encoded);
	return ::godot::internal::_call_utility_ret<int64_t>(_gde_function, &p_x_encoded);
}

Variant UtilityFunctions::abs(const Variant &p_x) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("abs")._native_ptr(), 4776452);
	CHECK_METHOD_BIND_RET(_gde_function, (Variant()));
	return ::godot::internal::_call_utility_ret<Variant>(_gde_function, &p_x);
}

double UtilityFunctions::absf(double p_x) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("absf")._native_ptr(), 2140049587);
	CHECK_METHOD_BIND_RET(_gde_function, (0.0));
	double p_x_encoded;
	PtrToArg<double>::encode(p_x, &p_x_encoded);
	return ::godot::internal::_call_utility_ret<double>(_gde_function, &p_x_encoded);
}

int64_t UtilityFunctions::absi(int64_t p_x) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("absi")._native_ptr(), 2157319888);
	CHECK_METHOD_BIND_RET(_gde_function, (0));
	int64_t p_x_encoded;
	PtrToArg<int64_t>::encode(p_x, &p_x_encoded);
	return ::godot::internal::_call_utility_ret<int64_t>(_gde_function, &p_x_encoded);
}

Variant UtilityFunctions::sign(const Variant &p_x) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("sign")._native_ptr(), 4776452);
	CHECK_METHOD_BIND_RET(_gde_function, (Variant()));
	return ::godot::internal::_call_utility_ret<Variant>(_gde_function, &p_x);
}

double UtilityFunctions::signf(double p_x) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("signf")._native_ptr(), 2140049587);
	CHECK_METHOD_BIND_RET(_gde_function, (0.0));
	double p_x_encoded;
	PtrToArg<double>::encode(p_x, &p_x_encoded);
	return ::godot::internal::_call_utility_ret<double>(_gde_function, &p_x_encoded);
}

int64_t UtilityFunctions::signi(int64_t p_x) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("signi")._native_ptr(), 2157319888);
	CHECK_METHOD_BIND_RET(_gde_function, (0));
	int64_t p_x_encoded;
	PtrToArg<int64_t>::encode(p_x, &p_x_encoded);
	return ::godot::internal::_call_utility_ret<int64_t>(_gde_function, &p_x_encoded);
}

Variant UtilityFunctions::snapped(const Variant &p_x, const Variant &p_step) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("snapped")._native_ptr(), 459914704);
	CHECK_METHOD_BIND_RET(_gde_function, (Variant()));
	return ::godot::internal::_call_utility_ret<Variant>(_gde_function, &p_x, &p_step);
}

double UtilityFunctions::snappedf(double p_x, double p_step) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("snappedf")._native_ptr(), 92296394);
	CHECK_METHOD_BIND_RET(_gde_function, (0.0));
	double p_x_encoded;
	PtrToArg<double>::encode(p_x, &p_x_encoded);
	double p_step_encoded;
	PtrToArg<double>::encode(p_step, &p_step_encoded);
	return ::godot::internal::_call_utility_ret<double>(_gde_function, &p_x_encoded, &p_step_encoded);
}

int64_t UtilityFunctions::snappedi(double p_x, int64_t p_step) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("snappedi")._native_ptr(), 3570758393);
	CHECK_METHOD_BIND_RET(_gde_function, (0));
	double p_x_encoded;
	PtrToArg<double>::encode(p_x, &p_x_encoded);
	int64_t p_step_encoded;
	PtrToArg<int64_t>::encode(p_step, &p_step_encoded);
	return ::godot::internal::_call_utility_ret<int64_t>(_gde_function, &p_x_encoded, &p_step_encoded);
}

double UtilityFunctions::pow(double p_base, double p_exp) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("pow")._native_ptr(), 92296394);
	CHECK_METHOD_BIND_RET(_gde_function, (0.0));
	double p_base_encoded;
	PtrToArg<double>::encode(p_base, &p_base_encoded);
	double p_exp_encoded;
	PtrToArg<double>::encode(p_exp, &p_exp_encoded);
	return ::godot::internal::_call_utility_ret<double>(_gde_function, &p_base_encoded, &p_exp_encoded);
}

double UtilityFunctions::log(double p_x) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("log")._native_ptr(), 2140049587);
	CHECK_METHOD_BIND_RET(_gde_function, (0.0));
	double p_x_encoded;
	PtrToArg<double>::encode(p_x, &p_x_encoded);
	return ::godot::internal::_call_utility_ret<double>(_gde_function, &p_x_encoded);
}

double UtilityFunctions::exp(double p_x) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("exp")._native_ptr(), 2140049587);
	CHECK_METHOD_BIND_RET(_gde_function, (0.0));
	double p_x_encoded;
	PtrToArg<double>::encode(p_x, &p_x_encoded);
	return ::godot::internal::_call_utility_ret<double>(_gde_function, &p_x_encoded);
}

bool UtilityFunctions::is_nan(double p_x) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("is_nan")._native_ptr(), 3569215213);
	CHECK_METHOD_BIND_RET(_gde_function, (false));
	double p_x_encoded;
	PtrToArg<double>::encode(p_x, &p_x_encoded);
	return ::godot::internal::_call_utility_ret<int8_t>(_gde_function, &p_x_encoded);
}

bool UtilityFunctions::is_inf(double p_x) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("is_inf")._native_ptr(), 3569215213);
	CHECK_METHOD_BIND_RET(_gde_function, (false));
	double p_x_encoded;
	PtrToArg<double>::encode(p_x, &p_x_encoded);
	return ::godot::internal::_call_utility_ret<int8_t>(_gde_function, &p_x_encoded);
}

bool UtilityFunctions::is_equal_approx(double p_a, double p_b) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("is_equal_approx")._native_ptr(), 1400789633);
	CHECK_METHOD_BIND_RET(_gde_function, (false));
	double p_a_encoded;
	PtrToArg<double>::encode(p_a, &p_a_encoded);
	double p_b_encoded;
	PtrToArg<double>::encode(p_b, &p_b_encoded);
	return ::godot::internal::_call_utility_ret<int8_t>(_gde_function, &p_a_encoded, &p_b_encoded);
}

bool UtilityFunctions::is_zero_approx(double p_x) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("is_zero_approx")._native_ptr(), 3569215213);
	CHECK_METHOD_BIND_RET(_gde_function, (false));
	double p_x_encoded;
	PtrToArg<double>::encode(p_x, &p_x_encoded);
	return ::godot::internal::_call_utility_ret<int8_t>(_gde_function, &p_x_encoded);
}

bool UtilityFunctions::is_finite(double p_x) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("is_finite")._native_ptr(), 3569215213);
	CHECK_METHOD_BIND_RET(_gde_function, (false));
	double p_x_encoded;
	PtrToArg<double>::encode(p_x, &p_x_encoded);
	return ::godot::internal::_call_utility_ret<int8_t>(_gde_function, &p_x_encoded);
}

double UtilityFunctions::ease(double p_x, double p_curve) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("ease")._native_ptr(), 92296394);
	CHECK_METHOD_BIND_RET(_gde_function, (0.0));
	double p_x_encoded;
	PtrToArg<double>::encode(p_x, &p_x_encoded);
	double p_curve_encoded;
	PtrToArg<double>::encode(p_curve, &p_curve_encoded);
	return ::godot::internal::_call_utility_ret<double>(_gde_function, &p_x_encoded, &p_curve_encoded);
}

int64_t UtilityFunctions::step_decimals(double p_x) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("step_decimals")._native_ptr(), 2780425386);
	CHECK_METHOD_BIND_RET(_gde_function, (0));
	double p_x_encoded;
	PtrToArg<double>::encode(p_x, &p_x_encoded);
	return ::godot::internal::_call_utility_ret<int64_t>(_gde_function, &p_x_encoded);
}

Variant UtilityFunctions::lerp(const Variant &p_from, const Variant &p_to, const Variant &p_weight) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("lerp")._native_ptr(), 3389874542);
	CHECK_METHOD_BIND_RET(_gde_function, (Variant()));
	return ::godot::internal::_call_utility_ret<Variant>(_gde_function, &p_from, &p_to, &p_weight);
}

double UtilityFunctions::lerpf(double p_from, double p_to, double p_weight) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("lerpf")._native_ptr(), 998901048);
	CHECK_METHOD_BIND_RET(_gde_function, (0.0));
	double p_from_encoded;
	PtrToArg<double>::encode(p_from, &p_from_encoded);
	double p_to_encoded;
	PtrToArg<double>::encode(p_to, &p_to_encoded);
	double p_weight_encoded;
	PtrToArg<double>::encode(p_weight, &p_weight_encoded);
	return ::godot::internal::_call_utility_ret<double>(_gde_function, &p_from_encoded, &p_to_encoded, &p_weight_encoded);
}

double UtilityFunctions::cubic_interpolate(double p_from, double p_to, double p_pre, double p_post, double p_weight) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("cubic_interpolate")._native_ptr(), 1090965791);
	CHECK_METHOD_BIND_RET(_gde_function, (0.0));
	double p_from_encoded;
	PtrToArg<double>::encode(p_from, &p_from_encoded);
	double p_to_encoded;
	PtrToArg<double>::encode(p_to, &p_to_encoded);
	double p_pre_encoded;
	PtrToArg<double>::encode(p_pre, &p_pre_encoded);
	double p_post_encoded;
	PtrToArg<double>::encode(p_post, &p_post_encoded);
	double p_weight_encoded;
	PtrToArg<double>::encode(p_weight, &p_weight_encoded);
	return ::godot::internal::_call_utility_ret<double>(_gde_function, &p_from_encoded, &p_to_encoded, &p_pre_encoded, &p_post_encoded, &p_weight_encoded);
}

double UtilityFunctions::cubic_interpolate_angle(double p_from, double p_to, double p_pre, double p_post, double p_weight) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("cubic_interpolate_angle")._native_ptr(), 1090965791);
	CHECK_METHOD_BIND_RET(_gde_function, (0.0));
	double p_from_encoded;
	PtrToArg<double>::encode(p_from, &p_from_encoded);
	double p_to_encoded;
	PtrToArg<double>::encode(p_to, &p_to_encoded);
	double p_pre_encoded;
	PtrToArg<double>::encode(p_pre, &p_pre_encoded);
	double p_post_encoded;
	PtrToArg<double>::encode(p_post, &p_post_encoded);
	double p_weight_encoded;
	PtrToArg<double>::encode(p_weight, &p_weight_encoded);
	return ::godot::internal::_call_utility_ret<double>(_gde_function, &p_from_encoded, &p_to_encoded, &p_pre_encoded, &p_post_encoded, &p_weight_encoded);
}

double UtilityFunctions::cubic_interpolate_in_time(double p_from, double p_to, double p_pre, double p_post, double p_weight, double p_to_t, double p_pre_t, double p_post_t) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("cubic_interpolate_in_time")._native_ptr(), 388121036);
	CHECK_METHOD_BIND_RET(_gde_function, (0.0));
	double p_from_encoded;
	PtrToArg<double>::encode(p_from, &p_from_encoded);
	double p_to_encoded;
	PtrToArg<double>::encode(p_to, &p_to_encoded);
	double p_pre_encoded;
	PtrToArg<double>::encode(p_pre, &p_pre_encoded);
	double p_post_encoded;
	PtrToArg<double>::encode(p_post, &p_post_encoded);
	double p_weight_encoded;
	PtrToArg<double>::encode(p_weight, &p_weight_encoded);
	double p_to_t_encoded;
	PtrToArg<double>::encode(p_to_t, &p_to_t_encoded);
	double p_pre_t_encoded;
	PtrToArg<double>::encode(p_pre_t, &p_pre_t_encoded);
	double p_post_t_encoded;
	PtrToArg<double>::encode(p_post_t, &p_post_t_encoded);
	return ::godot::internal::_call_utility_ret<double>(_gde_function, &p_from_encoded, &p_to_encoded, &p_pre_encoded, &p_post_encoded, &p_weight_encoded, &p_to_t_encoded, &p_pre_t_encoded, &p_post_t_encoded);
}

double UtilityFunctions::cubic_interpolate_angle_in_time(double p_from, double p_to, double p_pre, double p_post, double p_weight, double p_to_t, double p_pre_t, double p_post_t) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("cubic_interpolate_angle_in_time")._native_ptr(), 388121036);
	CHECK_METHOD_BIND_RET(_gde_function, (0.0));
	double p_from_encoded;
	PtrToArg<double>::encode(p_from, &p_from_encoded);
	double p_to_encoded;
	PtrToArg<double>::encode(p_to, &p_to_encoded);
	double p_pre_encoded;
	PtrToArg<double>::encode(p_pre, &p_pre_encoded);
	double p_post_encoded;
	PtrToArg<double>::encode(p_post, &p_post_encoded);
	double p_weight_encoded;
	PtrToArg<double>::encode(p_weight, &p_weight_encoded);
	double p_to_t_encoded;
	PtrToArg<double>::encode(p_to_t, &p_to_t_encoded);
	double p_pre_t_encoded;
	PtrToArg<double>::encode(p_pre_t, &p_pre_t_encoded);
	double p_post_t_encoded;
	PtrToArg<double>::encode(p_post_t, &p_post_t_encoded);
	return ::godot::internal::_call_utility_ret<double>(_gde_function, &p_from_encoded, &p_to_encoded, &p_pre_encoded, &p_post_encoded, &p_weight_encoded, &p_to_t_encoded, &p_pre_t_encoded, &p_post_t_encoded);
}

double UtilityFunctions::bezier_interpolate(double p_start, double p_control_1, double p_control_2, double p_end, double p_t) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("bezier_interpolate")._native_ptr(), 1090965791);
	CHECK_METHOD_BIND_RET(_gde_function, (0.0));
	double p_start_encoded;
	PtrToArg<double>::encode(p_start, &p_start_encoded);
	double p_control_1_encoded;
	PtrToArg<double>::encode(p_control_1, &p_control_1_encoded);
	double p_control_2_encoded;
	PtrToArg<double>::encode(p_control_2, &p_control_2_encoded);
	double p_end_encoded;
	PtrToArg<double>::encode(p_end, &p_end_encoded);
	double p_t_encoded;
	PtrToArg<double>::encode(p_t, &p_t_encoded);
	return ::godot::internal::_call_utility_ret<double>(_gde_function, &p_start_encoded, &p_control_1_encoded, &p_control_2_encoded, &p_end_encoded, &p_t_encoded);
}

double UtilityFunctions::bezier_derivative(double p_start, double p_control_1, double p_control_2, double p_end, double p_t) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("bezier_derivative")._native_ptr(), 1090965791);
	CHECK_METHOD_BIND_RET(_gde_function, (0.0));
	double p_start_encoded;
	PtrToArg<double>::encode(p_start, &p_start_encoded);
	double p_control_1_encoded;
	PtrToArg<double>::encode(p_control_1, &p_control_1_encoded);
	double p_control_2_encoded;
	PtrToArg<double>::encode(p_control_2, &p_control_2_encoded);
	double p_end_encoded;
	PtrToArg<double>::encode(p_end, &p_end_encoded);
	double p_t_encoded;
	PtrToArg<double>::encode(p_t, &p_t_encoded);
	return ::godot::internal::_call_utility_ret<double>(_gde_function, &p_start_encoded, &p_control_1_encoded, &p_control_2_encoded, &p_end_encoded, &p_t_encoded);
}

double UtilityFunctions::angle_difference(double p_from, double p_to) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("angle_difference")._native_ptr(), 92296394);
	CHECK_METHOD_BIND_RET(_gde_function, (0.0));
	double p_from_encoded;
	PtrToArg<double>::encode(p_from, &p_from_encoded);
	double p_to_encoded;
	PtrToArg<double>::encode(p_to, &p_to_encoded);
	return ::godot::internal::_call_utility_ret<double>(_gde_function, &p_from_encoded, &p_to_encoded);
}

double UtilityFunctions::lerp_angle(double p_from, double p_to, double p_weight) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("lerp_angle")._native_ptr(), 998901048);
	CHECK_METHOD_BIND_RET(_gde_function, (0.0));
	double p_from_encoded;
	PtrToArg<double>::encode(p_from, &p_from_encoded);
	double p_to_encoded;
	PtrToArg<double>::encode(p_to, &p_to_encoded);
	double p_weight_encoded;
	PtrToArg<double>::encode(p_weight, &p_weight_encoded);
	return ::godot::internal::_call_utility_ret<double>(_gde_function, &p_from_encoded, &p_to_encoded, &p_weight_encoded);
}

double UtilityFunctions::inverse_lerp(double p_from, double p_to, double p_weight) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("inverse_lerp")._native_ptr(), 998901048);
	CHECK_METHOD_BIND_RET(_gde_function, (0.0));
	double p_from_encoded;
	PtrToArg<double>::encode(p_from, &p_from_encoded);
	double p_to_encoded;
	PtrToArg<double>::encode(p_to, &p_to_encoded);
	double p_weight_encoded;
	PtrToArg<double>::encode(p_weight, &p_weight_encoded);
	return ::godot::internal::_call_utility_ret<double>(_gde_function, &p_from_encoded, &p_to_encoded, &p_weight_encoded);
}

double UtilityFunctions::remap(double p_value, double p_istart, double p_istop, double p_ostart, double p_ostop) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("remap")._native_ptr(), 1090965791);
	CHECK_METHOD_BIND_RET(_gde_function, (0.0));
	double p_value_encoded;
	PtrToArg<double>::encode(p_value, &p_value_encoded);
	double p_istart_encoded;
	PtrToArg<double>::encode(p_istart, &p_istart_encoded);
	double p_istop_encoded;
	PtrToArg<double>::encode(p_istop, &p_istop_encoded);
	double p_ostart_encoded;
	PtrToArg<double>::encode(p_ostart, &p_ostart_encoded);
	double p_ostop_encoded;
	PtrToArg<double>::encode(p_ostop, &p_ostop_encoded);
	return ::godot::internal::_call_utility_ret<double>(_gde_function, &p_value_encoded, &p_istart_encoded, &p_istop_encoded, &p_ostart_encoded, &p_ostop_encoded);
}

double UtilityFunctions::smoothstep(double p_from, double p_to, double p_x) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("smoothstep")._native_ptr(), 998901048);
	CHECK_METHOD_BIND_RET(_gde_function, (0.0));
	double p_from_encoded;
	PtrToArg<double>::encode(p_from, &p_from_encoded);
	double p_to_encoded;
	PtrToArg<double>::encode(p_to, &p_to_encoded);
	double p_x_encoded;
	PtrToArg<double>::encode(p_x, &p_x_encoded);
	return ::godot::internal::_call_utility_ret<double>(_gde_function, &p_from_encoded, &p_to_encoded, &p_x_encoded);
}

double UtilityFunctions::move_toward(double p_from, double p_to, double p_delta) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("move_toward")._native_ptr(), 998901048);
	CHECK_METHOD_BIND_RET(_gde_function, (0.0));
	double p_from_encoded;
	PtrToArg<double>::encode(p_from, &p_from_encoded);
	double p_to_encoded;
	PtrToArg<double>::encode(p_to, &p_to_encoded);
	double p_delta_encoded;
	PtrToArg<double>::encode(p_delta, &p_delta_encoded);
	return ::godot::internal::_call_utility_ret<double>(_gde_function, &p_from_encoded, &p_to_encoded, &p_delta_encoded);
}

double UtilityFunctions::rotate_toward(double p_from, double p_to, double p_delta) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("rotate_toward")._native_ptr(), 998901048);
	CHECK_METHOD_BIND_RET(_gde_function, (0.0));
	double p_from_encoded;
	PtrToArg<double>::encode(p_from, &p_from_encoded);
	double p_to_encoded;
	PtrToArg<double>::encode(p_to, &p_to_encoded);
	double p_delta_encoded;
	PtrToArg<double>::encode(p_delta, &p_delta_encoded);
	return ::godot::internal::_call_utility_ret<double>(_gde_function, &p_from_encoded, &p_to_encoded, &p_delta_encoded);
}

double UtilityFunctions::deg_to_rad(double p_deg) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("deg_to_rad")._native_ptr(), 2140049587);
	CHECK_METHOD_BIND_RET(_gde_function, (0.0));
	double p_deg_encoded;
	PtrToArg<double>::encode(p_deg, &p_deg_encoded);
	return ::godot::internal::_call_utility_ret<double>(_gde_function, &p_deg_encoded);
}

double UtilityFunctions::rad_to_deg(double p_rad) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("rad_to_deg")._native_ptr(), 2140049587);
	CHECK_METHOD_BIND_RET(_gde_function, (0.0));
	double p_rad_encoded;
	PtrToArg<double>::encode(p_rad, &p_rad_encoded);
	return ::godot::internal::_call_utility_ret<double>(_gde_function, &p_rad_encoded);
}

double UtilityFunctions::linear_to_db(double p_lin) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("linear_to_db")._native_ptr(), 2140049587);
	CHECK_METHOD_BIND_RET(_gde_function, (0.0));
	double p_lin_encoded;
	PtrToArg<double>::encode(p_lin, &p_lin_encoded);
	return ::godot::internal::_call_utility_ret<double>(_gde_function, &p_lin_encoded);
}

double UtilityFunctions::db_to_linear(double p_db) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("db_to_linear")._native_ptr(), 2140049587);
	CHECK_METHOD_BIND_RET(_gde_function, (0.0));
	double p_db_encoded;
	PtrToArg<double>::encode(p_db, &p_db_encoded);
	return ::godot::internal::_call_utility_ret<double>(_gde_function, &p_db_encoded);
}

Variant UtilityFunctions::wrap(const Variant &p_value, const Variant &p_min, const Variant &p_max) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("wrap")._native_ptr(), 3389874542);
	CHECK_METHOD_BIND_RET(_gde_function, (Variant()));
	return ::godot::internal::_call_utility_ret<Variant>(_gde_function, &p_value, &p_min, &p_max);
}

int64_t UtilityFunctions::wrapi(int64_t p_value, int64_t p_min, int64_t p_max) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("wrapi")._native_ptr(), 650295447);
	CHECK_METHOD_BIND_RET(_gde_function, (0));
	int64_t p_value_encoded;
	PtrToArg<int64_t>::encode(p_value, &p_value_encoded);
	int64_t p_min_encoded;
	PtrToArg<int64_t>::encode(p_min, &p_min_encoded);
	int64_t p_max_encoded;
	PtrToArg<int64_t>::encode(p_max, &p_max_encoded);
	return ::godot::internal::_call_utility_ret<int64_t>(_gde_function, &p_value_encoded, &p_min_encoded, &p_max_encoded);
}

double UtilityFunctions::wrapf(double p_value, double p_min, double p_max) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("wrapf")._native_ptr(), 998901048);
	CHECK_METHOD_BIND_RET(_gde_function, (0.0));
	double p_value_encoded;
	PtrToArg<double>::encode(p_value, &p_value_encoded);
	double p_min_encoded;
	PtrToArg<double>::encode(p_min, &p_min_encoded);
	double p_max_encoded;
	PtrToArg<double>::encode(p_max, &p_max_encoded);
	return ::godot::internal::_call_utility_ret<double>(_gde_function, &p_value_encoded, &p_min_encoded, &p_max_encoded);
}

Variant UtilityFunctions::max_internal(const Variant **p_args, GDExtensionInt p_arg_count) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("max")._native_ptr(), 3896050336);
	CHECK_METHOD_BIND_RET(_gde_function, (Variant()));
	Variant ret;
	_gde_function(&ret, reinterpret_cast<GDExtensionConstVariantPtr *>(p_args), p_arg_count);
	return ret;
}

int64_t UtilityFunctions::maxi(int64_t p_a, int64_t p_b) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("maxi")._native_ptr(), 3133453818);
	CHECK_METHOD_BIND_RET(_gde_function, (0));
	int64_t p_a_encoded;
	PtrToArg<int64_t>::encode(p_a, &p_a_encoded);
	int64_t p_b_encoded;
	PtrToArg<int64_t>::encode(p_b, &p_b_encoded);
	return ::godot::internal::_call_utility_ret<int64_t>(_gde_function, &p_a_encoded, &p_b_encoded);
}

double UtilityFunctions::maxf(double p_a, double p_b) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("maxf")._native_ptr(), 92296394);
	CHECK_METHOD_BIND_RET(_gde_function, (0.0));
	double p_a_encoded;
	PtrToArg<double>::encode(p_a, &p_a_encoded);
	double p_b_encoded;
	PtrToArg<double>::encode(p_b, &p_b_encoded);
	return ::godot::internal::_call_utility_ret<double>(_gde_function, &p_a_encoded, &p_b_encoded);
}

Variant UtilityFunctions::min_internal(const Variant **p_args, GDExtensionInt p_arg_count) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("min")._native_ptr(), 3896050336);
	CHECK_METHOD_BIND_RET(_gde_function, (Variant()));
	Variant ret;
	_gde_function(&ret, reinterpret_cast<GDExtensionConstVariantPtr *>(p_args), p_arg_count);
	return ret;
}

int64_t UtilityFunctions::mini(int64_t p_a, int64_t p_b) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("mini")._native_ptr(), 3133453818);
	CHECK_METHOD_BIND_RET(_gde_function, (0));
	int64_t p_a_encoded;
	PtrToArg<int64_t>::encode(p_a, &p_a_encoded);
	int64_t p_b_encoded;
	PtrToArg<int64_t>::encode(p_b, &p_b_encoded);
	return ::godot::internal::_call_utility_ret<int64_t>(_gde_function, &p_a_encoded, &p_b_encoded);
}

double UtilityFunctions::minf(double p_a, double p_b) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("minf")._native_ptr(), 92296394);
	CHECK_METHOD_BIND_RET(_gde_function, (0.0));
	double p_a_encoded;
	PtrToArg<double>::encode(p_a, &p_a_encoded);
	double p_b_encoded;
	PtrToArg<double>::encode(p_b, &p_b_encoded);
	return ::godot::internal::_call_utility_ret<double>(_gde_function, &p_a_encoded, &p_b_encoded);
}

Variant UtilityFunctions::clamp(const Variant &p_value, const Variant &p_min, const Variant &p_max) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("clamp")._native_ptr(), 3389874542);
	CHECK_METHOD_BIND_RET(_gde_function, (Variant()));
	return ::godot::internal::_call_utility_ret<Variant>(_gde_function, &p_value, &p_min, &p_max);
}

int64_t UtilityFunctions::clampi(int64_t p_value, int64_t p_min, int64_t p_max) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("clampi")._native_ptr(), 650295447);
	CHECK_METHOD_BIND_RET(_gde_function, (0));
	int64_t p_value_encoded;
	PtrToArg<int64_t>::encode(p_value, &p_value_encoded);
	int64_t p_min_encoded;
	PtrToArg<int64_t>::encode(p_min, &p_min_encoded);
	int64_t p_max_encoded;
	PtrToArg<int64_t>::encode(p_max, &p_max_encoded);
	return ::godot::internal::_call_utility_ret<int64_t>(_gde_function, &p_value_encoded, &p_min_encoded, &p_max_encoded);
}

double UtilityFunctions::clampf(double p_value, double p_min, double p_max) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("clampf")._native_ptr(), 998901048);
	CHECK_METHOD_BIND_RET(_gde_function, (0.0));
	double p_value_encoded;
	PtrToArg<double>::encode(p_value, &p_value_encoded);
	double p_min_encoded;
	PtrToArg<double>::encode(p_min, &p_min_encoded);
	double p_max_encoded;
	PtrToArg<double>::encode(p_max, &p_max_encoded);
	return ::godot::internal::_call_utility_ret<double>(_gde_function, &p_value_encoded, &p_min_encoded, &p_max_encoded);
}

int64_t UtilityFunctions::nearest_po2(int64_t p_value) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("nearest_po2")._native_ptr(), 2157319888);
	CHECK_METHOD_BIND_RET(_gde_function, (0));
	int64_t p_value_encoded;
	PtrToArg<int64_t>::encode(p_value, &p_value_encoded);
	return ::godot::internal::_call_utility_ret<int64_t>(_gde_function, &p_value_encoded);
}

double UtilityFunctions::pingpong(double p_value, double p_length) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("pingpong")._native_ptr(), 92296394);
	CHECK_METHOD_BIND_RET(_gde_function, (0.0));
	double p_value_encoded;
	PtrToArg<double>::encode(p_value, &p_value_encoded);
	double p_length_encoded;
	PtrToArg<double>::encode(p_length, &p_length_encoded);
	return ::godot::internal::_call_utility_ret<double>(_gde_function, &p_value_encoded, &p_length_encoded);
}

void UtilityFunctions::randomize() {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("randomize")._native_ptr(), 1691721052);
	CHECK_METHOD_BIND(_gde_function);
	::godot::internal::_call_utility_no_ret(_gde_function);
}

int64_t UtilityFunctions::randi() {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("randi")._native_ptr(), 701202648);
	CHECK_METHOD_BIND_RET(_gde_function, (0));
	return ::godot::internal::_call_utility_ret<int64_t>(_gde_function);
}

double UtilityFunctions::randf() {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("randf")._native_ptr(), 2086227845);
	CHECK_METHOD_BIND_RET(_gde_function, (0.0));
	return ::godot::internal::_call_utility_ret<double>(_gde_function);
}

int64_t UtilityFunctions::randi_range(int64_t p_from, int64_t p_to) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("randi_range")._native_ptr(), 3133453818);
	CHECK_METHOD_BIND_RET(_gde_function, (0));
	int64_t p_from_encoded;
	PtrToArg<int64_t>::encode(p_from, &p_from_encoded);
	int64_t p_to_encoded;
	PtrToArg<int64_t>::encode(p_to, &p_to_encoded);
	return ::godot::internal::_call_utility_ret<int64_t>(_gde_function, &p_from_encoded, &p_to_encoded);
}

double UtilityFunctions::randf_range(double p_from, double p_to) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("randf_range")._native_ptr(), 92296394);
	CHECK_METHOD_BIND_RET(_gde_function, (0.0));
	double p_from_encoded;
	PtrToArg<double>::encode(p_from, &p_from_encoded);
	double p_to_encoded;
	PtrToArg<double>::encode(p_to, &p_to_encoded);
	return ::godot::internal::_call_utility_ret<double>(_gde_function, &p_from_encoded, &p_to_encoded);
}

double UtilityFunctions::randfn(double p_mean, double p_deviation) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("randfn")._native_ptr(), 92296394);
	CHECK_METHOD_BIND_RET(_gde_function, (0.0));
	double p_mean_encoded;
	PtrToArg<double>::encode(p_mean, &p_mean_encoded);
	double p_deviation_encoded;
	PtrToArg<double>::encode(p_deviation, &p_deviation_encoded);
	return ::godot::internal::_call_utility_ret<double>(_gde_function, &p_mean_encoded, &p_deviation_encoded);
}

void UtilityFunctions::seed(int64_t p_base) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("seed")._native_ptr(), 382931173);
	CHECK_METHOD_BIND(_gde_function);
	int64_t p_base_encoded;
	PtrToArg<int64_t>::encode(p_base, &p_base_encoded);
	::godot::internal::_call_utility_no_ret(_gde_function, &p_base_encoded);
}

PackedInt64Array UtilityFunctions::rand_from_seed(int64_t p_seed) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("rand_from_seed")._native_ptr(), 1391063685);
	CHECK_METHOD_BIND_RET(_gde_function, (PackedInt64Array()));
	int64_t p_seed_encoded;
	PtrToArg<int64_t>::encode(p_seed, &p_seed_encoded);
	return ::godot::internal::_call_utility_ret<PackedInt64Array>(_gde_function, &p_seed_encoded);
}

Variant UtilityFunctions::weakref(const Variant &p_obj) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("weakref")._native_ptr(), 4776452);
	CHECK_METHOD_BIND_RET(_gde_function, (Variant()));
	return ::godot::internal::_call_utility_ret<Variant>(_gde_function, &p_obj);
}

int64_t UtilityFunctions::type_of(const Variant &p_variable) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("typeof")._native_ptr(), 326422594);
	CHECK_METHOD_BIND_RET(_gde_function, (0));
	return ::godot::internal::_call_utility_ret<int64_t>(_gde_function, &p_variable);
}

Variant UtilityFunctions::type_convert(const Variant &p_variant, int64_t p_type) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("type_convert")._native_ptr(), 2453062746);
	CHECK_METHOD_BIND_RET(_gde_function, (Variant()));
	int64_t p_type_encoded;
	PtrToArg<int64_t>::encode(p_type, &p_type_encoded);
	return ::godot::internal::_call_utility_ret<Variant>(_gde_function, &p_variant, &p_type_encoded);
}

String UtilityFunctions::str_internal(const Variant **p_args, GDExtensionInt p_arg_count) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("str")._native_ptr(), 32569176);
	CHECK_METHOD_BIND_RET(_gde_function, (String()));
	String ret;
	_gde_function(&ret, reinterpret_cast<GDExtensionConstVariantPtr *>(p_args), p_arg_count);
	return ret;
}

String UtilityFunctions::error_string(int64_t p_error) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("error_string")._native_ptr(), 942708242);
	CHECK_METHOD_BIND_RET(_gde_function, (String()));
	int64_t p_error_encoded;
	PtrToArg<int64_t>::encode(p_error, &p_error_encoded);
	return ::godot::internal::_call_utility_ret<String>(_gde_function, &p_error_encoded);
}

String UtilityFunctions::type_string(int64_t p_type) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("type_string")._native_ptr(), 942708242);
	CHECK_METHOD_BIND_RET(_gde_function, (String()));
	int64_t p_type_encoded;
	PtrToArg<int64_t>::encode(p_type, &p_type_encoded);
	return ::godot::internal::_call_utility_ret<String>(_gde_function, &p_type_encoded);
}

void UtilityFunctions::print_internal(const Variant **p_args, GDExtensionInt p_arg_count) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("print")._native_ptr(), 2648703342);
	CHECK_METHOD_BIND(_gde_function);
	Variant ret;
	_gde_function(&ret, reinterpret_cast<GDExtensionConstVariantPtr *>(p_args), p_arg_count);
}

void UtilityFunctions::print_rich_internal(const Variant **p_args, GDExtensionInt p_arg_count) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("print_rich")._native_ptr(), 2648703342);
	CHECK_METHOD_BIND(_gde_function);
	Variant ret;
	_gde_function(&ret, reinterpret_cast<GDExtensionConstVariantPtr *>(p_args), p_arg_count);
}

void UtilityFunctions::printerr_internal(const Variant **p_args, GDExtensionInt p_arg_count) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("printerr")._native_ptr(), 2648703342);
	CHECK_METHOD_BIND(_gde_function);
	Variant ret;
	_gde_function(&ret, reinterpret_cast<GDExtensionConstVariantPtr *>(p_args), p_arg_count);
}

void UtilityFunctions::printt_internal(const Variant **p_args, GDExtensionInt p_arg_count) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("printt")._native_ptr(), 2648703342);
	CHECK_METHOD_BIND(_gde_function);
	Variant ret;
	_gde_function(&ret, reinterpret_cast<GDExtensionConstVariantPtr *>(p_args), p_arg_count);
}

void UtilityFunctions::prints_internal(const Variant **p_args, GDExtensionInt p_arg_count) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("prints")._native_ptr(), 2648703342);
	CHECK_METHOD_BIND(_gde_function);
	Variant ret;
	_gde_function(&ret, reinterpret_cast<GDExtensionConstVariantPtr *>(p_args), p_arg_count);
}

void UtilityFunctions::printraw_internal(const Variant **p_args, GDExtensionInt p_arg_count) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("printraw")._native_ptr(), 2648703342);
	CHECK_METHOD_BIND(_gde_function);
	Variant ret;
	_gde_function(&ret, reinterpret_cast<GDExtensionConstVariantPtr *>(p_args), p_arg_count);
}

void UtilityFunctions::print_verbose_internal(const Variant **p_args, GDExtensionInt p_arg_count) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("print_verbose")._native_ptr(), 2648703342);
	CHECK_METHOD_BIND(_gde_function);
	Variant ret;
	_gde_function(&ret, reinterpret_cast<GDExtensionConstVariantPtr *>(p_args), p_arg_count);
}

void UtilityFunctions::push_error_internal(const Variant **p_args, GDExtensionInt p_arg_count) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("push_error")._native_ptr(), 2648703342);
	CHECK_METHOD_BIND(_gde_function);
	Variant ret;
	_gde_function(&ret, reinterpret_cast<GDExtensionConstVariantPtr *>(p_args), p_arg_count);
}

void UtilityFunctions::push_warning_internal(const Variant **p_args, GDExtensionInt p_arg_count) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("push_warning")._native_ptr(), 2648703342);
	CHECK_METHOD_BIND(_gde_function);
	Variant ret;
	_gde_function(&ret, reinterpret_cast<GDExtensionConstVariantPtr *>(p_args), p_arg_count);
}

String UtilityFunctions::var_to_str(const Variant &p_variable) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("var_to_str")._native_ptr(), 866625479);
	CHECK_METHOD_BIND_RET(_gde_function, (String()));
	return ::godot::internal::_call_utility_ret<String>(_gde_function, &p_variable);
}

Variant UtilityFunctions::str_to_var(const String &p_string) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("str_to_var")._native_ptr(), 1891498491);
	CHECK_METHOD_BIND_RET(_gde_function, (Variant()));
	return ::godot::internal::_call_utility_ret<Variant>(_gde_function, &p_string);
}

PackedByteArray UtilityFunctions::var_to_bytes(const Variant &p_variable) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("var_to_bytes")._native_ptr(), 2947269930);
	CHECK_METHOD_BIND_RET(_gde_function, (PackedByteArray()));
	return ::godot::internal::_call_utility_ret<PackedByteArray>(_gde_function, &p_variable);
}

Variant UtilityFunctions::bytes_to_var(const PackedByteArray &p_bytes) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("bytes_to_var")._native_ptr(), 4249819452);
	CHECK_METHOD_BIND_RET(_gde_function, (Variant()));
	return ::godot::internal::_call_utility_ret<Variant>(_gde_function, &p_bytes);
}

PackedByteArray UtilityFunctions::var_to_bytes_with_objects(const Variant &p_variable) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("var_to_bytes_with_objects")._native_ptr(), 2947269930);
	CHECK_METHOD_BIND_RET(_gde_function, (PackedByteArray()));
	return ::godot::internal::_call_utility_ret<PackedByteArray>(_gde_function, &p_variable);
}

Variant UtilityFunctions::bytes_to_var_with_objects(const PackedByteArray &p_bytes) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("bytes_to_var_with_objects")._native_ptr(), 4249819452);
	CHECK_METHOD_BIND_RET(_gde_function, (Variant()));
	return ::godot::internal::_call_utility_ret<Variant>(_gde_function, &p_bytes);
}

int64_t UtilityFunctions::hash(const Variant &p_variable) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("hash")._native_ptr(), 326422594);
	CHECK_METHOD_BIND_RET(_gde_function, (0));
	return ::godot::internal::_call_utility_ret<int64_t>(_gde_function, &p_variable);
}

Object *UtilityFunctions::instance_from_id(int64_t p_instance_id) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("instance_from_id")._native_ptr(), 1156694636);
	CHECK_METHOD_BIND_RET(_gde_function, (nullptr));
	int64_t p_instance_id_encoded;
	PtrToArg<int64_t>::encode(p_instance_id, &p_instance_id_encoded);
	return ::godot::internal::_call_utility_ret_obj(_gde_function, &p_instance_id_encoded);
}

bool UtilityFunctions::is_instance_id_valid(int64_t p_id) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("is_instance_id_valid")._native_ptr(), 2232439758);
	CHECK_METHOD_BIND_RET(_gde_function, (false));
	int64_t p_id_encoded;
	PtrToArg<int64_t>::encode(p_id, &p_id_encoded);
	return ::godot::internal::_call_utility_ret<int8_t>(_gde_function, &p_id_encoded);
}

int64_t UtilityFunctions::rid_allocate_id() {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("rid_allocate_id")._native_ptr(), 701202648);
	CHECK_METHOD_BIND_RET(_gde_function, (0));
	return ::godot::internal::_call_utility_ret<int64_t>(_gde_function);
}

RID UtilityFunctions::rid_from_int64(int64_t p_base) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("rid_from_int64")._native_ptr(), 3426892196);
	CHECK_METHOD_BIND_RET(_gde_function, (RID()));
	int64_t p_base_encoded;
	PtrToArg<int64_t>::encode(p_base, &p_base_encoded);
	return ::godot::internal::_call_utility_ret<RID>(_gde_function, &p_base_encoded);
}

bool UtilityFunctions::is_same(const Variant &p_a, const Variant &p_b) {
	static GDExtensionPtrUtilityFunction _gde_function = ::godot::gdextension_interface::variant_get_ptr_utility_function(StringName("is_same")._native_ptr(), 1409423524);
	CHECK_METHOD_BIND_RET(_gde_function, (false));
	return ::godot::internal::_call_utility_ret<int8_t>(_gde_function, &p_a, &p_b);
}

} // namespace godot