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
#include "core/ustring.h"

struct _NO_DISCARD_CLASS_ Color {
	union {
		struct {
			float r;
			float g;
			float b;
			float a;
		};
		float components[4];
	};

	bool operator==(const Color &p_color) const { return (r == p_color.r && g == p_color.g && b == p_color.b && a == p_color.a); }
	bool operator!=(const Color &p_color) const { return (r != p_color.r || g != p_color.g || b != p_color.b || a != p_color.a); }

	uint32_t to_rgba32() const;
	uint32_t to_argb32() const;
	uint32_t to_abgr32() const;
	uint64_t to_rgba64() const;
	uint64_t to_argb64() const;
	uint64_t to_abgr64() const;
	float gray() const;
	float get_h() const;
	float get_s() const;
	float get_v() const;
	void set_hsv(float p_h, float p_s, float p_v, float p_alpha = 1.0);

	_FORCE_INLINE_ float &operator[](int p_idx) {
		return components[p_idx];
	}
	_FORCE_INLINE_ const float &operator[](int p_idx) const {
		return components[p_idx];
	}

	Color operator+(const Color &p_color) const;
	void operator+=(const Color &p_color);

	Color operator-() const;
	Color operator-(const Color &p_color) const;
	void operator-=(const Color &p_color);

	Color operator*(const Color &p_color) const;
	Color operator*(real_t p_scalar) const;
	void operator*=(const Color &p_color);
	void operator*=(real_t p_scalar);

	Color operator/(const Color &p_color) const;
	Color operator/(real_t p_scalar) const;
	void operator/=(const Color &p_color);
	void operator/=(real_t p_scalar);

	bool is_equal_approx(const Color &p_color) const;

	void invert();
	void contrast();
	Color inverted() const;
	Color contrasted() const;

	_FORCE_INLINE_ float get_luminance() const {
		return 0.2126 * r + 0.7152 * g + 0.0722 * b;
	}

	_FORCE_INLINE_ Color linear_interpolate(const Color &p_to, float p_weight) const {
		Color res = *this;

		res.r += (p_weight * (p_to.r - r));
		res.g += (p_weight * (p_to.g - g));
		res.b += (p_weight * (p_to.b - b));
		res.a += (p_weight * (p_to.a - a));

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
		//const float Emax = 31.0f;
		const float N = 9.0f;

		float sharedexp = 65408.000f; //(( pow2to9  - 1.0f)/ pow2to9)*powf( 2.0f, 31.0f - 15.0f);

		float cRed = MAX(0.0f, MIN(sharedexp, r));
		float cGreen = MAX(0.0f, MIN(sharedexp, g));
		float cBlue = MAX(0.0f, MIN(sharedexp, b));

		float cMax = MAX(cRed, MAX(cGreen, cBlue));

		// expp = MAX(-B - 1, log2(maxc)) + 1 + B

		float expp = MAX(-B - 1.0f, floor(Math::log(cMax) / Math_LN2)) + 1.0f + B;

		float sMax = (float)floor((cMax / Math::pow(2.0f, expp - B - N)) + 0.5f);

		float exps = expp + 1.0f;

		if (0.0 <= sMax && sMax < pow2to9) {
			exps = expp;
		}

		float sRed = Math::floor((cRed / pow(2.0f, exps - B - N)) + 0.5f);
		float sGreen = Math::floor((cGreen / pow(2.0f, exps - B - N)) + 0.5f);
		float sBlue = Math::floor((cBlue / pow(2.0f, exps - B - N)) + 0.5f);

		return (uint32_t(Math::fast_ftoi(sRed)) & 0x1FF) | ((uint32_t(Math::fast_ftoi(sGreen)) & 0x1FF) << 9) | ((uint32_t(Math::fast_ftoi(sBlue)) & 0x1FF) << 18) | ((uint32_t(Math::fast_ftoi(exps)) & 0x1F) << 27);
	}

	_FORCE_INLINE_ Color blend(const Color &p_over) const {
		Color res;
		float sa = 1.0 - p_over.a;
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

	_FORCE_INLINE_ Color to_linear() const {
		return Color(
				r < 0.04045 ? r * (1.0 / 12.92) : Math::pow((r + 0.055) * (1.0 / (1 + 0.055)), 2.4),
				g < 0.04045 ? g * (1.0 / 12.92) : Math::pow((g + 0.055) * (1.0 / (1 + 0.055)), 2.4),
				b < 0.04045 ? b * (1.0 / 12.92) : Math::pow((b + 0.055) * (1.0 / (1 + 0.055)), 2.4),
				a);
	}
	_FORCE_INLINE_ Color to_srgb() const {
		return Color(
				r < 0.0031308 ? 12.92 * r : (1.0 + 0.055) * Math::pow(r, 1.0f / 2.4f) - 0.055,
				g < 0.0031308 ? 12.92 * g : (1.0 + 0.055) * Math::pow(g, 1.0f / 2.4f) - 0.055,
				b < 0.0031308 ? 12.92 * b : (1.0 + 0.055) * Math::pow(b, 1.0f / 2.4f) - 0.055, a);
	}

	static Color hex(uint32_t p_hex);
	static Color hex64(uint64_t p_hex);
	static Color html(const String &p_color);
	static bool html_is_valid(const String &p_color);
	static Color named(const String &p_name);
	String to_html(bool p_alpha = true) const;
	Color from_hsv(float p_h, float p_s, float p_v, float p_a) const;
	static Color from_rgbe9995(uint32_t p_rgbe);

	_FORCE_INLINE_ bool operator<(const Color &p_color) const; //used in set keys
	operator String() const;

	/**
	 * No construct parameters, r=0, g=0, b=0. a=255
	 */
	_FORCE_INLINE_ Color() {
		r = 0;
		g = 0;
		b = 0;
		a = 1.0;
	}

	/**
	 * RGB / RGBA construct parameters. Alpha is optional, but defaults to 1.0
	 */
	_FORCE_INLINE_ Color(float p_r, float p_g, float p_b, float p_a = 1.0) {
		r = p_r;
		g = p_g;
		b = p_b;
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

#endif // COLOR_H
