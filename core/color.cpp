/*************************************************************************/
/*  color.cpp                                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "color.h"

#include "core/color_names.inc"
#include "core/map.h"
#include "core/math/math_funcs.h"
#include "core/print_string.h"

uint32_t Color::to_argb32() const {
	uint32_t c = (uint8_t)Math::round(a * 255);
	c <<= 8;
	c |= (uint8_t)Math::round(r * 255);
	c <<= 8;
	c |= (uint8_t)Math::round(g * 255);
	c <<= 8;
	c |= (uint8_t)Math::round(b * 255);

	return c;
}

uint32_t Color::to_abgr32() const {
	uint32_t c = (uint8_t)Math::round(a * 255);
	c <<= 8;
	c |= (uint8_t)Math::round(b * 255);
	c <<= 8;
	c |= (uint8_t)Math::round(g * 255);
	c <<= 8;
	c |= (uint8_t)Math::round(r * 255);

	return c;
}

uint32_t Color::to_rgba32() const {
	uint32_t c = (uint8_t)Math::round(r * 255);
	c <<= 8;
	c |= (uint8_t)Math::round(g * 255);
	c <<= 8;
	c |= (uint8_t)Math::round(b * 255);
	c <<= 8;
	c |= (uint8_t)Math::round(a * 255);

	return c;
}

uint64_t Color::to_abgr64() const {
	uint64_t c = (uint16_t)Math::round(a * 65535);
	c <<= 16;
	c |= (uint16_t)Math::round(b * 65535);
	c <<= 16;
	c |= (uint16_t)Math::round(g * 65535);
	c <<= 16;
	c |= (uint16_t)Math::round(r * 65535);

	return c;
}

uint64_t Color::to_argb64() const {
	uint64_t c = (uint16_t)Math::round(a * 65535);
	c <<= 16;
	c |= (uint16_t)Math::round(r * 65535);
	c <<= 16;
	c |= (uint16_t)Math::round(g * 65535);
	c <<= 16;
	c |= (uint16_t)Math::round(b * 65535);

	return c;
}

uint64_t Color::to_rgba64() const {
	uint64_t c = (uint16_t)Math::round(r * 65535);
	c <<= 16;
	c |= (uint16_t)Math::round(g * 65535);
	c <<= 16;
	c |= (uint16_t)Math::round(b * 65535);
	c <<= 16;
	c |= (uint16_t)Math::round(a * 65535);

	return c;
}

float Color::get_h() const {
	float min = MIN(r, g);
	min = MIN(min, b);
	float max = MAX(r, g);
	max = MAX(max, b);

	float delta = max - min;
	if (delta == 0)
		return 0;

	float h;
	if (r == max) {
		h = (g - b) / delta; // between magenta & yellow
		if (h < 0)
			h += 6;
	} else if (g == max) {
		h = 2 + (b - r) / delta; // between yellow & cyan
	} else {
		h = 4 + (r - g) / delta; // between cyan & magenta
	}

	h /= 6.0;
	return h;
}

float Color::get_s() const {
	float max = MAX(r, g);
	max = MAX(max, b);
	if (max == 0)
		return 0;

	float min = MIN(r, g);
	min = MIN(min, b);

	return (max - min) / max;
}

float Color::get_v() const {
	float max = MAX(r, g);
	max = MAX(max, b);
	return max;
}

void Color::set_hsv(float p_h, float p_s, float p_v, float p_alpha) {
	a = p_alpha;
	if (p_s == 0 || p_v == 0) {
		r = g = b = p_v;
		return;
	}

	float min = p_v * (1 - p_s);

	p_h *= 6.0;
	p_h = Math::fmod(p_h, 6);
	int i = Math::floor(p_h);
	float f = p_h - i;
	if (i % 2 == 0) {
		f = 1 - f;
	}

	float mid = p_v * (1 - p_s * f);

	switch (i) {
		case 0: // Red dominant, Blue weakest
			r = p_v;
			g = mid;
			b = min;
			break;
		case 1: // Green dominant, Blue weakest
			r = mid;
			g = p_v;
			b = min;
			break;
		case 2: // Green dominant, Red weakest
			r = min;
			g = p_v;
			b = mid;
			break;
		case 3: // Blue dominant, Red weakest
			r = min;
			g = mid;
			b = p_v;
			break;
		case 4: // Blue dominant, Green weakest
			r = mid;
			g = min;
			b = p_v;
			break;
		default: // Red dominant, Green weakest
			r = p_v;
			g = min;
			b = mid;
			break;
	}
}

void Color::invert() {
	r = 1.0 - r;
	g = 1.0 - g;
	b = 1.0 - b;
}

void Color::contrast() {
	r = Math::fmod(r + 0.5, 1.0);
	g = Math::fmod(g + 0.5, 1.0);
	b = Math::fmod(b + 0.5, 1.0);
}

Color Color::hex(uint32_t p_hex) {
	float a = (p_hex & 0xFF) / 255.0;
	p_hex >>= 8;
	float b = (p_hex & 0xFF) / 255.0;
	p_hex >>= 8;
	float g = (p_hex & 0xFF) / 255.0;
	p_hex >>= 8;
	float r = (p_hex & 0xFF) / 255.0;

	return Color(r, g, b, a);
}

Color Color::hex64(uint64_t p_hex) {
	float a = (p_hex & 0xFFFF) / 65535.0;
	p_hex >>= 16;
	float b = (p_hex & 0xFFFF) / 65535.0;
	p_hex >>= 16;
	float g = (p_hex & 0xFFFF) / 65535.0;
	p_hex >>= 16;
	float r = (p_hex & 0xFFFF) / 65535.0;

	return Color(r, g, b, a);
}

Color Color::from_rgbe9995(uint32_t p_rgbe) {
	float r = p_rgbe & 0x1ff;
	float g = (p_rgbe >> 9) & 0x1ff;
	float b = (p_rgbe >> 18) & 0x1ff;
	float e = (p_rgbe >> 27);
	float m = Math::pow(2, e - 15.0 - 9.0);

	float rd = r * m;
	float gd = g * m;
	float bd = b * m;

	return Color(rd, gd, bd, 1.0f);
}

// Take a hex string of type ARGB and convert it to an uint32_t stored in out
// Returns a bool that indicates whether conversion was successful
static bool _parse_hex(const String &p_color, uint32_t &out) {
	String col_str = p_color;
	int len = col_str.length();
	if (len == 0)
		return false;

	// Reformat color string
	if (col_str[0] == '#') {
		len--;
		col_str = col_str.substr(1, len);
	}

	if (len == 3 || len == 4) {
		String full_hex = "";
		for (int i = 0; i < len; i++) {
			full_hex += col_str[i] + col_str[i];
		}
		len *= 2;
		col_str = full_hex;
	}

	// Check validity
	if (len != 8 && len != 6)
		return false;

	// Convert to int
	out = 255;
	out <<= 8;
	for (int i = 0; i < len; i++) {
		int c = col_str[i];

		if (c >= '0' && c <= '9') {
			out |= c - '0';
		} else if (c >= 'a' && c <= 'f') {
			out |= 10 + c - 'a';
		} else if (c >= 'A' && c <= 'F') {
			out |= 10 + c - 'A';
		} else {
			return false;
		}

		if (i != len - 1)
			out <<= 4;
	}

	return true;
}

Color Color::inverted() const {
	Color c = *this;
	c.invert();
	return c;
}

Color Color::contrasted() const {
	Color c = *this;
	c.contrast();
	return c;
}

Color Color::html(const String &p_color) {
	uint32_t col_int;
	bool hex_valid = _parse_hex(p_color, col_int);
	if (!hex_valid) {
		ERR_EXPLAIN("Invalid Color Code: " + p_color);
		ERR_FAIL_V(Color());
	}

	// Rearrange the bits of the int to convert ARGB to RGBA for hex()
	uint32_t temp = col_int >> 24;
	col_int <<= 8;
	col_int |= temp;

	return hex(col_int);
}

bool Color::html_is_valid(const String &p_color) {
	uint32_t dummy;
	if (_parse_hex(p_color, dummy))
		return true;
	else
		return false;
}

Color Color::named(const String &p_name) {
	if (_named_colors.empty()) _populate_named_colors(); // from color_names.inc
	String name = p_name;
	// Normalize name
	name = name.replace(" ", "");
	name = name.replace("-", "");
	name = name.replace("_", "");
	name = name.replace("'", "");
	name = name.replace(".", "");
	name = name.to_lower();

	const Map<String, Color>::Element *color = _named_colors.find(name);
	if (color) {
		return color->value();
	} else {
		ERR_EXPLAIN("Invalid Color Name: " + p_name);
		ERR_FAIL_V(Color());
	}
}

String _to_hex(float p_val) {
	int v = Math::round(p_val * 255);
	v = CLAMP(v, 0, 255);

	CharType c[3] = { 0, 0, 0 };
	for (int i = 0; i < 2; i++) {
		int lv = v & 0xF;
		if (lv < 10)
			c[i] = '0' + lv;
		else
			c[i] = 'a' + lv - 10;

		v >>= 4;
	}

	String cs = (const CharType *)c;
	return cs;
}

String Color::to_html(bool p_alpha) const {
	String txt;
	if (p_alpha)
		txt += _to_hex(a);
	txt += _to_hex(r);
	txt += _to_hex(g);
	txt += _to_hex(b);

	return txt;
}

Color Color::from_hsv(float p_h, float p_s, float p_v, float p_a) const {
	Color ret = Color();
	ret.set_hsv(p_h, p_s, p_v, p_a);

	return ret;
}

// FIXME: Remove once Godot 3.1 has been released
float Color::gray() const {
	ERR_EXPLAIN("Color.gray() is deprecated and will be removed in a future version. Use Color.get_v() for a better grayscale approximation.");
	WARN_DEPRECATED
	return (r + g + b) / 3.0;
}

Color::operator String() const {
	return rtos(r) + ", " + rtos(g) + ", " + rtos(b) + ", " + rtos(a);
}

Color Color::operator+(const Color &p_color) const {
	return Color(
			r + p_color.r,
			g + p_color.g,
			b + p_color.b,
			a + p_color.a);
}

void Color::operator+=(const Color &p_color) {
	r += p_color.r;
	g += p_color.g;
	b += p_color.b;
	a += p_color.a;
}

Color Color::operator-(const Color &p_color) const {
	return Color(
			r - p_color.r,
			g - p_color.g,
			b - p_color.b,
			a - p_color.a);
}

void Color::operator-=(const Color &p_color) {
	r -= p_color.r;
	g -= p_color.g;
	b -= p_color.b;
	a -= p_color.a;
}

Color Color::operator*(const Color &p_color) const {
	return Color(
			r * p_color.r,
			g * p_color.g,
			b * p_color.b,
			a * p_color.a);
}

Color Color::operator*(const real_t &rvalue) const {
	return Color(
			r * rvalue,
			g * rvalue,
			b * rvalue,
			a * rvalue);
}

void Color::operator*=(const Color &p_color) {
	r *= p_color.r;
	g *= p_color.g;
	b *= p_color.b;
	a *= p_color.a;
}

void Color::operator*=(const real_t &rvalue) {
	r *= rvalue;
	g *= rvalue;
	b *= rvalue;
	a *= rvalue;
}

Color Color::operator/(const Color &p_color) const {
	return Color(
			r / p_color.r,
			g / p_color.g,
			b / p_color.b,
			a / p_color.a);
}

Color Color::operator/(const real_t &rvalue) const {
	if (rvalue == 0)
		return Color(1.0, 1.0, 1.0, 1.0);

	return Color(
			r / rvalue,
			g / rvalue,
			b / rvalue,
			a / rvalue);
}

void Color::operator/=(const Color &p_color) {
	r /= p_color.r;
	g /= p_color.g;
	b /= p_color.b;
	a /= p_color.a;
}

void Color::operator/=(const real_t &rvalue) {
	if (rvalue == 0) {
		r = 1.0;
		g = 1.0;
		b = 1.0;
		a = 1.0;
	} else {
		r /= rvalue;
		g /= rvalue;
		b /= rvalue;
		a /= rvalue;
	}
};

Color Color::operator-() const {
	return Color(
			1.0 - r,
			1.0 - g,
			1.0 - b,
			1.0 - a);
}
