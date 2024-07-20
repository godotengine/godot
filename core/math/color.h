/**************************************************************************/
/*  color.h                                                               */
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

#ifndef COLOR_H
#define COLOR_H

#include "core/math/math_funcs.h"

class String;

struct [[nodiscard]] Color {
	union {
		struct {
			float r;
			float g;
			float b;
			float a;
		};
		float components[4] = { 0, 0, 0, 1.0 };
	};

	uint32_t to_rgba32() const;
	uint32_t to_argb32() const;
	uint32_t to_abgr32() const;
	uint64_t to_rgba64() const;
	uint64_t to_argb64() const;
	uint64_t to_abgr64() const;
	String to_html(bool p_alpha = true) const;
	float get_h() const;
	float get_s() const;
	float get_v() const;
	void set_hsv(float p_h, float p_s, float p_v, float p_alpha = 1.0f);
	float get_ok_hsl_h() const;
	float get_ok_hsl_s() const;
	float get_ok_hsl_l() const;
	void set_ok_hsl(float p_h, float p_s, float p_l, float p_alpha = 1.0f);

	_FORCE_INLINE_ float &operator[](int p_idx) {
		return components[p_idx];
	}
	_FORCE_INLINE_ const float &operator[](int p_idx) const {
		return components[p_idx];
	}

	bool operator==(const Color &p_color) const {
		return (r == p_color.r && g == p_color.g && b == p_color.b && a == p_color.a);
	}
	bool operator!=(const Color &p_color) const {
		return (r != p_color.r || g != p_color.g || b != p_color.b || a != p_color.a);
	}

	Color operator+(const Color &p_color) const;
	void operator+=(const Color &p_color);

	Color operator-() const;
	Color operator-(const Color &p_color) const;
	void operator-=(const Color &p_color);

	Color operator*(const Color &p_color) const;
	Color operator*(float p_scalar) const;
	void operator*=(const Color &p_color);
	void operator*=(float p_scalar);

	Color operator/(const Color &p_color) const;
	Color operator/(float p_scalar) const;
	void operator/=(const Color &p_color);
	void operator/=(float p_scalar);

	bool is_equal_approx(const Color &p_color) const;

	Color clamp(const Color &p_min = Color(0, 0, 0, 0), const Color &p_max = Color(1, 1, 1, 1)) const;
	void invert();
	Color inverted() const;

	_FORCE_INLINE_ float get_luminance() const {
		return 0.2126f * r + 0.7152f * g + 0.0722f * b;
	}

	_FORCE_INLINE_ Color lerp(const Color &p_to, float p_weight) const {
		Color res = *this;
		res.r = Math::lerp(res.r, p_to.r, p_weight);
		res.g = Math::lerp(res.g, p_to.g, p_weight);
		res.b = Math::lerp(res.b, p_to.b, p_weight);
		res.a = Math::lerp(res.a, p_to.a, p_weight);
		return res;
	}

	_FORCE_INLINE_ Color darkened(float p_amount) const {
		Color res = *this;
		res.r = res.r * (1.0f - p_amount);
		res.g = res.g * (1.0f - p_amount);
		res.b = res.b * (1.0f - p_amount);
		return res;
	}

	_FORCE_INLINE_ Color lightened(float p_amount) const {
		Color res = *this;
		res.r = res.r + (1.0f - res.r) * p_amount;
		res.g = res.g + (1.0f - res.g) * p_amount;
		res.b = res.b + (1.0f - res.b) * p_amount;
		return res;
	}

	_FORCE_INLINE_ uint32_t to_rgbe9995() const {
		const float pow2to9 = 512.0f;
		const float B = 15.0f;
		const float N = 9.0f;

		float sharedexp = 65408.000f; // Result of: ((pow2to9 - 1.0f) / pow2to9) * powf(2.0f, 31.0f - 15.0f)

		float cRed = MAX(0.0f, MIN(sharedexp, r));
		float cGreen = MAX(0.0f, MIN(sharedexp, g));
		float cBlue = MAX(0.0f, MIN(sharedexp, b));

		float cMax = MAX(cRed, MAX(cGreen, cBlue));

		float expp = MAX(-B - 1.0f, floor(Math::log(cMax) / (real_t)Math_LN2)) + 1.0f + B;

		float sMax = (float)floor((cMax / Math::pow(2.0f, expp - B - N)) + 0.5f);

		float exps = expp + 1.0f;

		if (0.0f <= sMax && sMax < pow2to9) {
			exps = expp;
		}

		float sRed = Math::floor((cRed / pow(2.0f, exps - B - N)) + 0.5f);
		float sGreen = Math::floor((cGreen / pow(2.0f, exps - B - N)) + 0.5f);
		float sBlue = Math::floor((cBlue / pow(2.0f, exps - B - N)) + 0.5f);

		return (uint32_t(Math::fast_ftoi(sRed)) & 0x1FF) | ((uint32_t(Math::fast_ftoi(sGreen)) & 0x1FF) << 9) | ((uint32_t(Math::fast_ftoi(sBlue)) & 0x1FF) << 18) | ((uint32_t(Math::fast_ftoi(exps)) & 0x1F) << 27);
	}

	_FORCE_INLINE_ Color blend(const Color &p_over) const {
		Color res;
		float sa = 1.0f - p_over.a;
		res.a = a * sa + p_over.a;
		if (res.a == 0) {
			return Color(0, 0, 0, 0);
		} else {
			res.r = (r * a * sa + p_over.r * p_over.a) / res.a;
			res.g = (g * a * sa + p_over.g * p_over.a) / res.a;
			res.b = (b * a * sa + p_over.b * p_over.a) / res.a;
		}
		return res;
	}

	_FORCE_INLINE_ Color srgb_to_linear() const {
		return Color(
				r < 0.04045f ? r * (1.0f / 12.92f) : Math::pow((r + 0.055f) * (float)(1.0 / (1.0 + 0.055)), 2.4f),
				g < 0.04045f ? g * (1.0f / 12.92f) : Math::pow((g + 0.055f) * (float)(1.0 / (1.0 + 0.055)), 2.4f),
				b < 0.04045f ? b * (1.0f / 12.92f) : Math::pow((b + 0.055f) * (float)(1.0 / (1.0 + 0.055)), 2.4f),
				a);
	}
	_FORCE_INLINE_ Color linear_to_srgb() const {
		return Color(
				r < 0.0031308f ? 12.92f * r : (1.0f + 0.055f) * Math::pow(r, 1.0f / 2.4f) - 0.055f,
				g < 0.0031308f ? 12.92f * g : (1.0f + 0.055f) * Math::pow(g, 1.0f / 2.4f) - 0.055f,
				b < 0.0031308f ? 12.92f * b : (1.0f + 0.055f) * Math::pow(b, 1.0f / 2.4f) - 0.055f, a);
	}

	static Color hex(uint32_t p_hex);
	static Color hex64(uint64_t p_hex);
	static Color html(const String &p_rgba);
	static bool html_is_valid(const String &p_color);
	static Color named(const String &p_name);
	static Color named(const String &p_name, const Color &p_default);
	static int find_named_color(const String &p_name);
	static int get_named_color_count();
	static String get_named_color_name(int p_idx);
	static Color get_named_color(int p_idx);
	static Color from_string(const String &p_string, const Color &p_default);
	static Color from_hsv(float p_h, float p_s, float p_v, float p_alpha = 1.0f);
	static Color from_ok_hsl(float p_h, float p_s, float p_l, float p_alpha = 1.0f);
	static Color from_rgbe9995(uint32_t p_rgbe);

	_FORCE_INLINE_ bool operator<(const Color &p_color) const; // Used in set keys.
	operator String() const;

	// For the binder.
	_FORCE_INLINE_ void set_r8(int32_t r8) { r = (CLAMP(r8, 0, 255) / 255.0f); }
	_FORCE_INLINE_ int32_t get_r8() const { return int32_t(CLAMP(Math::round(r * 255.0f), 0.0f, 255.0f)); }
	_FORCE_INLINE_ void set_g8(int32_t g8) { g = (CLAMP(g8, 0, 255) / 255.0f); }
	_FORCE_INLINE_ int32_t get_g8() const { return int32_t(CLAMP(Math::round(g * 255.0f), 0.0f, 255.0f)); }
	_FORCE_INLINE_ void set_b8(int32_t b8) { b = (CLAMP(b8, 0, 255) / 255.0f); }
	_FORCE_INLINE_ int32_t get_b8() const { return int32_t(CLAMP(Math::round(b * 255.0f), 0.0f, 255.0f)); }
	_FORCE_INLINE_ void set_a8(int32_t a8) { a = (CLAMP(a8, 0, 255) / 255.0f); }
	_FORCE_INLINE_ int32_t get_a8() const { return int32_t(CLAMP(Math::round(a * 255.0f), 0.0f, 255.0f)); }

	_FORCE_INLINE_ void set_h(float p_h) { set_hsv(p_h, get_s(), get_v(), a); }
	_FORCE_INLINE_ void set_s(float p_s) { set_hsv(get_h(), p_s, get_v(), a); }
	_FORCE_INLINE_ void set_v(float p_v) { set_hsv(get_h(), get_s(), p_v, a); }
	_FORCE_INLINE_ void set_ok_hsl_h(float p_h) { set_ok_hsl(p_h, get_ok_hsl_s(), get_ok_hsl_l(), a); }
	_FORCE_INLINE_ void set_ok_hsl_s(float p_s) { set_ok_hsl(get_ok_hsl_h(), p_s, get_ok_hsl_l(), a); }
	_FORCE_INLINE_ void set_ok_hsl_l(float p_l) { set_ok_hsl(get_ok_hsl_h(), get_ok_hsl_s(), p_l, a); }

	_FORCE_INLINE_ Color() {}

	/**
	 * RGBA construct parameters.
	 * Alpha is not optional as otherwise we can't bind the RGB version for scripting.
	 */
	_FORCE_INLINE_ Color(float p_r, float p_g, float p_b, float p_a) {
		r = p_r;
		g = p_g;
		b = p_b;
		a = p_a;
	}

	/**
	 * RGB construct parameters.
	 */
	_FORCE_INLINE_ Color(float p_r, float p_g, float p_b) {
		r = p_r;
		g = p_g;
		b = p_b;
		a = 1.0f;
	}

	/**
	 * Construct a Color from another Color, but with the specified alpha value.
	 */
	_FORCE_INLINE_ Color(const Color &p_c, float p_a) {
		r = p_c.r;
		g = p_c.g;
		b = p_c.b;
		a = p_a;
	}

	Color(const String &p_code) {
		if (html_is_valid(p_code)) {
			*this = html(p_code);
		} else {
			*this = named(p_code);
		}
	}

	Color(const String &p_code, float p_a) {
		*this = Color(p_code);
		a = p_a;
	}
};

bool Color::operator<(const Color &p_color) const {
	if (r == p_color.r) {
		if (g == p_color.g) {
			if (b == p_color.b) {
				return (a < p_color.a);
			} else {
				return (b < p_color.b);
			}
		} else {
			return g < p_color.g;
		}
	} else {
		return r < p_color.r;
	}
}

_FORCE_INLINE_ Color operator*(float p_scalar, const Color &p_color) {
	return p_color * p_scalar;
}

#endif // COLOR_H
