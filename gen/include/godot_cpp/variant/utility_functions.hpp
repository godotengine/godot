/**************************************************************************/
/*  utility_functions.hpp                                                 */
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

#pragma once

#include <godot_cpp/variant/builtin_types.hpp>
#include <godot_cpp/variant/variant.hpp>

#include <array>

namespace godot {

class UtilityFunctions {
public:
	static double sin(double p_angle_rad);
	static double cos(double p_angle_rad);
	static double tan(double p_angle_rad);
	static double sinh(double p_x);
	static double cosh(double p_x);
	static double tanh(double p_x);
	static double asin(double p_x);
	static double acos(double p_x);
	static double atan(double p_x);
	static double atan2(double p_y, double p_x);
	static double asinh(double p_x);
	static double acosh(double p_x);
	static double atanh(double p_x);
	static double sqrt(double p_x);
	static double fmod(double p_x, double p_y);
	static double fposmod(double p_x, double p_y);
	static int64_t posmod(int64_t p_x, int64_t p_y);
	static Variant floor(const Variant &p_x);
	static double floorf(double p_x);
	static int64_t floori(double p_x);
	static Variant ceil(const Variant &p_x);
	static double ceilf(double p_x);
	static int64_t ceili(double p_x);
	static Variant round(const Variant &p_x);
	static double roundf(double p_x);
	static int64_t roundi(double p_x);
	static Variant abs(const Variant &p_x);
	static double absf(double p_x);
	static int64_t absi(int64_t p_x);
	static Variant sign(const Variant &p_x);
	static double signf(double p_x);
	static int64_t signi(int64_t p_x);
	static Variant snapped(const Variant &p_x, const Variant &p_step);
	static double snappedf(double p_x, double p_step);
	static int64_t snappedi(double p_x, int64_t p_step);
	static double pow(double p_base, double p_exp);
	static double log(double p_x);
	static double exp(double p_x);
	static bool is_nan(double p_x);
	static bool is_inf(double p_x);
	static bool is_equal_approx(double p_a, double p_b);
	static bool is_zero_approx(double p_x);
	static bool is_finite(double p_x);
	static double ease(double p_x, double p_curve);
	static int64_t step_decimals(double p_x);
	static Variant lerp(const Variant &p_from, const Variant &p_to, const Variant &p_weight);
	static double lerpf(double p_from, double p_to, double p_weight);
	static double cubic_interpolate(double p_from, double p_to, double p_pre, double p_post, double p_weight);
	static double cubic_interpolate_angle(double p_from, double p_to, double p_pre, double p_post, double p_weight);
	static double cubic_interpolate_in_time(double p_from, double p_to, double p_pre, double p_post, double p_weight, double p_to_t, double p_pre_t, double p_post_t);
	static double cubic_interpolate_angle_in_time(double p_from, double p_to, double p_pre, double p_post, double p_weight, double p_to_t, double p_pre_t, double p_post_t);
	static double bezier_interpolate(double p_start, double p_control_1, double p_control_2, double p_end, double p_t);
	static double bezier_derivative(double p_start, double p_control_1, double p_control_2, double p_end, double p_t);
	static double angle_difference(double p_from, double p_to);
	static double lerp_angle(double p_from, double p_to, double p_weight);
	static double inverse_lerp(double p_from, double p_to, double p_weight);
	static double remap(double p_value, double p_istart, double p_istop, double p_ostart, double p_ostop);
	static double smoothstep(double p_from, double p_to, double p_x);
	static double move_toward(double p_from, double p_to, double p_delta);
	static double rotate_toward(double p_from, double p_to, double p_delta);
	static double deg_to_rad(double p_deg);
	static double rad_to_deg(double p_rad);
	static double linear_to_db(double p_lin);
	static double db_to_linear(double p_db);
	static Variant wrap(const Variant &p_value, const Variant &p_min, const Variant &p_max);
	static int64_t wrapi(int64_t p_value, int64_t p_min, int64_t p_max);
	static double wrapf(double p_value, double p_min, double p_max);

private:
	static Variant max_internal(const Variant **p_args, GDExtensionInt p_arg_count);

public:
	template <typename... Args>
	static Variant max(const Variant &p_arg1, const Variant &p_arg2, const Args &...p_args) {
		std::array<Variant, 2 + sizeof...(Args)> variant_args{{ p_arg1, p_arg2, Variant(p_args)... }};
		std::array<const Variant *, 2 + sizeof...(Args)> call_args;
		for (size_t i = 0; i < variant_args.size(); i++) {
			call_args[i] = &variant_args[i];
		}
		return max_internal(call_args.data(), variant_args.size());
	}
	static int64_t maxi(int64_t p_a, int64_t p_b);
	static double maxf(double p_a, double p_b);

private:
	static Variant min_internal(const Variant **p_args, GDExtensionInt p_arg_count);

public:
	template <typename... Args>
	static Variant min(const Variant &p_arg1, const Variant &p_arg2, const Args &...p_args) {
		std::array<Variant, 2 + sizeof...(Args)> variant_args{{ p_arg1, p_arg2, Variant(p_args)... }};
		std::array<const Variant *, 2 + sizeof...(Args)> call_args;
		for (size_t i = 0; i < variant_args.size(); i++) {
			call_args[i] = &variant_args[i];
		}
		return min_internal(call_args.data(), variant_args.size());
	}
	static int64_t mini(int64_t p_a, int64_t p_b);
	static double minf(double p_a, double p_b);
	static Variant clamp(const Variant &p_value, const Variant &p_min, const Variant &p_max);
	static int64_t clampi(int64_t p_value, int64_t p_min, int64_t p_max);
	static double clampf(double p_value, double p_min, double p_max);
	static int64_t nearest_po2(int64_t p_value);
	static double pingpong(double p_value, double p_length);
	static void randomize();
	static int64_t randi();
	static double randf();
	static int64_t randi_range(int64_t p_from, int64_t p_to);
	static double randf_range(double p_from, double p_to);
	static double randfn(double p_mean, double p_deviation);
	static void seed(int64_t p_base);
	static PackedInt64Array rand_from_seed(int64_t p_seed);
	static Variant weakref(const Variant &p_obj);
	static int64_t type_of(const Variant &p_variable);
	static Variant type_convert(const Variant &p_variant, int64_t p_type);

private:
	static String str_internal(const Variant **p_args, GDExtensionInt p_arg_count);

public:
	template <typename... Args>
	static String str(const Variant &p_arg1, const Args &...p_args) {
		std::array<Variant, 1 + sizeof...(Args)> variant_args{{ p_arg1, Variant(p_args)... }};
		std::array<const Variant *, 1 + sizeof...(Args)> call_args;
		for (size_t i = 0; i < variant_args.size(); i++) {
			call_args[i] = &variant_args[i];
		}
		return str_internal(call_args.data(), variant_args.size());
	}
	static String error_string(int64_t p_error);
	static String type_string(int64_t p_type);

private:
	static void print_internal(const Variant **p_args, GDExtensionInt p_arg_count);

public:
	template <typename... Args>
	static void print(const Variant &p_arg1, const Args &...p_args) {
		std::array<Variant, 1 + sizeof...(Args)> variant_args{{ p_arg1, Variant(p_args)... }};
		std::array<const Variant *, 1 + sizeof...(Args)> call_args;
		for (size_t i = 0; i < variant_args.size(); i++) {
			call_args[i] = &variant_args[i];
		}
		print_internal(call_args.data(), variant_args.size());
	}

private:
	static void print_rich_internal(const Variant **p_args, GDExtensionInt p_arg_count);

public:
	template <typename... Args>
	static void print_rich(const Variant &p_arg1, const Args &...p_args) {
		std::array<Variant, 1 + sizeof...(Args)> variant_args{{ p_arg1, Variant(p_args)... }};
		std::array<const Variant *, 1 + sizeof...(Args)> call_args;
		for (size_t i = 0; i < variant_args.size(); i++) {
			call_args[i] = &variant_args[i];
		}
		print_rich_internal(call_args.data(), variant_args.size());
	}

private:
	static void printerr_internal(const Variant **p_args, GDExtensionInt p_arg_count);

public:
	template <typename... Args>
	static void printerr(const Variant &p_arg1, const Args &...p_args) {
		std::array<Variant, 1 + sizeof...(Args)> variant_args{{ p_arg1, Variant(p_args)... }};
		std::array<const Variant *, 1 + sizeof...(Args)> call_args;
		for (size_t i = 0; i < variant_args.size(); i++) {
			call_args[i] = &variant_args[i];
		}
		printerr_internal(call_args.data(), variant_args.size());
	}

private:
	static void printt_internal(const Variant **p_args, GDExtensionInt p_arg_count);

public:
	template <typename... Args>
	static void printt(const Variant &p_arg1, const Args &...p_args) {
		std::array<Variant, 1 + sizeof...(Args)> variant_args{{ p_arg1, Variant(p_args)... }};
		std::array<const Variant *, 1 + sizeof...(Args)> call_args;
		for (size_t i = 0; i < variant_args.size(); i++) {
			call_args[i] = &variant_args[i];
		}
		printt_internal(call_args.data(), variant_args.size());
	}

private:
	static void prints_internal(const Variant **p_args, GDExtensionInt p_arg_count);

public:
	template <typename... Args>
	static void prints(const Variant &p_arg1, const Args &...p_args) {
		std::array<Variant, 1 + sizeof...(Args)> variant_args{{ p_arg1, Variant(p_args)... }};
		std::array<const Variant *, 1 + sizeof...(Args)> call_args;
		for (size_t i = 0; i < variant_args.size(); i++) {
			call_args[i] = &variant_args[i];
		}
		prints_internal(call_args.data(), variant_args.size());
	}

private:
	static void printraw_internal(const Variant **p_args, GDExtensionInt p_arg_count);

public:
	template <typename... Args>
	static void printraw(const Variant &p_arg1, const Args &...p_args) {
		std::array<Variant, 1 + sizeof...(Args)> variant_args{{ p_arg1, Variant(p_args)... }};
		std::array<const Variant *, 1 + sizeof...(Args)> call_args;
		for (size_t i = 0; i < variant_args.size(); i++) {
			call_args[i] = &variant_args[i];
		}
		printraw_internal(call_args.data(), variant_args.size());
	}

private:
	static void print_verbose_internal(const Variant **p_args, GDExtensionInt p_arg_count);

public:
	template <typename... Args>
	static void print_verbose(const Variant &p_arg1, const Args &...p_args) {
		std::array<Variant, 1 + sizeof...(Args)> variant_args{{ p_arg1, Variant(p_args)... }};
		std::array<const Variant *, 1 + sizeof...(Args)> call_args;
		for (size_t i = 0; i < variant_args.size(); i++) {
			call_args[i] = &variant_args[i];
		}
		print_verbose_internal(call_args.data(), variant_args.size());
	}

private:
	static void push_error_internal(const Variant **p_args, GDExtensionInt p_arg_count);

public:
	template <typename... Args>
	static void push_error(const Variant &p_arg1, const Args &...p_args) {
		std::array<Variant, 1 + sizeof...(Args)> variant_args{{ p_arg1, Variant(p_args)... }};
		std::array<const Variant *, 1 + sizeof...(Args)> call_args;
		for (size_t i = 0; i < variant_args.size(); i++) {
			call_args[i] = &variant_args[i];
		}
		push_error_internal(call_args.data(), variant_args.size());
	}

private:
	static void push_warning_internal(const Variant **p_args, GDExtensionInt p_arg_count);

public:
	template <typename... Args>
	static void push_warning(const Variant &p_arg1, const Args &...p_args) {
		std::array<Variant, 1 + sizeof...(Args)> variant_args{{ p_arg1, Variant(p_args)... }};
		std::array<const Variant *, 1 + sizeof...(Args)> call_args;
		for (size_t i = 0; i < variant_args.size(); i++) {
			call_args[i] = &variant_args[i];
		}
		push_warning_internal(call_args.data(), variant_args.size());
	}
	static String var_to_str(const Variant &p_variable);
	static Variant str_to_var(const String &p_string);
	static PackedByteArray var_to_bytes(const Variant &p_variable);
	static Variant bytes_to_var(const PackedByteArray &p_bytes);
	static PackedByteArray var_to_bytes_with_objects(const Variant &p_variable);
	static Variant bytes_to_var_with_objects(const PackedByteArray &p_bytes);
	static int64_t hash(const Variant &p_variable);
	static Object *instance_from_id(int64_t p_instance_id);
	static bool is_instance_id_valid(int64_t p_id);
	static int64_t rid_allocate_id();
	static RID rid_from_int64(int64_t p_base);
	static bool is_same(const Variant &p_a, const Variant &p_b);
};

} // namespace godot
