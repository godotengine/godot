/*************************************************************************/
/*  color.cpp                                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "color_names.inc"
#include "core/math/math_funcs.h"
#include "core/string/print_string.h"
#include "core/templates/map.h"

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

String _to_hex(float p_val) {
	int v = Math::round(p_val * 255);
	v = CLAMP(v, 0, 255);
	String ret;

	for (int i = 0; i < 2; i++) {
		char32_t c[2] = { 0, 0 };
		int lv = v & 0xF;
		if (lv < 10) {
			c[0] = '0' + lv;
		} else {
			c[0] = 'a' + lv - 10;
		}

		v >>= 4;
		String cs = (const char32_t *)c;
		ret = cs + ret;
	}

	return ret;
}

String Color::to_html(bool p_alpha) const {
	String txt;
	txt += _to_hex(r);
	txt += _to_hex(g);
	txt += _to_hex(b);
	if (p_alpha) {
		txt += _to_hex(a);
	}
	return txt;
}

float Color::get_h() const {
	float min = MIN(r, g);
	min = MIN(min, b);
	float max = MAX(r, g);
	max = MAX(max, b);

	float delta = max - min;

	if (delta == 0) {
		return 0;
	}

	float h;
	if (r == max) {
		h = (g - b) / delta; // between yellow & magenta
	} else if (g == max) {
		h = 2 + (b - r) / delta; // between cyan & yellow
	} else {
		h = 4 + (r - g) / delta; // between magenta & cyan
	}

	h /= 6.0;
	if (h < 0) {
		h += 1.0;
	}

	return h;
}

float Color::get_s() const {
	float min = MIN(r, g);
	min = MIN(min, b);
	float max = MAX(r, g);
	max = MAX(max, b);

	float delta = max - min;

	return (max != 0) ? (delta / max) : 0;
}

float Color::get_v() const {
	float max = MAX(r, g);
	max = MAX(max, b);
	return max;
}

void Color::set_hsv(float p_h, float p_s, float p_v, float p_alpha) {
	int i;
	float f, p, q, t;
	a = p_alpha;

	if (p_s == 0) {
		// Achromatic (grey)
		r = g = b = p_v;
		return;
	}

	p_h *= 6.0;
	p_h = Math::fmod(p_h, 6);
	i = Math::floor(p_h);

	f = p_h - i;
	p = p_v * (1 - p_s);
	q = p_v * (1 - p_s * f);
	t = p_v * (1 - p_s * (1 - f));

	switch (i) {
		case 0: // Red is the dominant color
			r = p_v;
			g = t;
			b = p;
			break;
		case 1: // Green is the dominant color
			r = q;
			g = p_v;
			b = p;
			break;
		case 2:
			r = p;
			g = p_v;
			b = t;
			break;
		case 3: // Blue is the dominant color
			r = p;
			g = q;
			b = p_v;
			break;
		case 4:
			r = t;
			g = p;
			b = p_v;
			break;
		default: // (5) Red is the dominant color
			r = p_v;
			g = p;
			b = q;
			break;
	}
}

bool Color::is_equal_approx(const Color &p_color) const {
	return Math::is_equal_approx(r, p_color.r) && Math::is_equal_approx(g, p_color.g) && Math::is_equal_approx(b, p_color.b) && Math::is_equal_approx(a, p_color.a);
}

Color Color::clamp(const Color &p_min, const Color &p_max) const {
	return Color(
			CLAMP(r, p_min.r, p_max.r),
			CLAMP(g, p_min.g, p_max.g),
			CLAMP(b, p_min.b, p_max.b),
			CLAMP(a, p_min.a, p_max.a));
}

void Color::invert() {
	r = 1.0 - r;
	g = 1.0 - g;
	b = 1.0 - b;
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

static int _parse_col4(const String &p_str, int p_ofs) {
	char character = p_str[p_ofs];

	if (character >= '0' && character <= '9') {
		return character - '0';
	} else if (character >= 'a' && character <= 'f') {
		return character + (10 - 'a');
	} else if (character >= 'A' && character <= 'F') {
		return character + (10 - 'A');
	}
	return -1;
}

static int _parse_col8(const String &p_str, int p_ofs) {
	return _parse_col4(p_str, p_ofs) * 16 + _parse_col4(p_str, p_ofs + 1);
}

Color Color::inverted() const {
	Color c = *this;
	c.invert();
	return c;
}

Color Color::html(const String &p_rgba) {
	String color = p_rgba;
	if (color.length() == 0) {
		return Color();
	}
	if (color[0] == '#') {
		color = color.substr(1);
	}

	// If enabled, use 1 hex digit per channel instead of 2.
	// Other sizes aren't in the HTML/CSS spec but we could add them if desired.
	bool is_shorthand = color.length() < 5;
	bool alpha = false;

	if (color.length() == 8) {
		alpha = true;
	} else if (color.length() == 6) {
		alpha = false;
	} else if (color.length() == 4) {
		alpha = true;
	} else if (color.length() == 3) {
		alpha = false;
	} else {
		ERR_FAIL_V_MSG(Color(), "Invalid color code: " + p_rgba + ".");
	}

	float r, g, b, a = 1.0;
	if (is_shorthand) {
		r = _parse_col4(color, 0) / 15.0;
		g = _parse_col4(color, 1) / 15.0;
		b = _parse_col4(color, 2) / 15.0;
		if (alpha) {
			a = _parse_col4(color, 3) / 15.0;
		}
	} else {
		r = _parse_col8(color, 0) / 255.0;
		g = _parse_col8(color, 2) / 255.0;
		b = _parse_col8(color, 4) / 255.0;
		if (alpha) {
			a = _parse_col8(color, 6) / 255.0;
		}
	}
	ERR_FAIL_COND_V_MSG(r < 0, Color(), "Invalid color code: " + p_rgba + ".");
	ERR_FAIL_COND_V_MSG(g < 0, Color(), "Invalid color code: " + p_rgba + ".");
	ERR_FAIL_COND_V_MSG(b < 0, Color(), "Invalid color code: " + p_rgba + ".");
	ERR_FAIL_COND_V_MSG(a < 0, Color(), "Invalid color code: " + p_rgba + ".");

	return Color(r, g, b, a);
}

bool Color::html_is_valid(const String &p_color) {
	String color = p_color;

	if (color.length() == 0) {
		return false;
	}
	if (color[0] == '#') {
		color = color.substr(1);
	}

	// Check if the amount of hex digits is valid.
	int len = color.length();
	if (!(len == 3 || len == 4 || len == 6 || len == 8)) {
		return false;
	}

	// Check if each hex digit is valid.
	for (int i = 0; i < len; i++) {
		if (_parse_col4(color, i) == -1) {
			return false;
		}
	}

	return true;
}

Color Color::named(const String &p_name) {
	int idx = find_named_color(p_name);
	if (idx == -1) {
		ERR_FAIL_V_MSG(Color(), "Invalid color name: " + p_name + ".");
		return Color();
	}
	return named_colors[idx].color;
}

Color Color::named(const String &p_name, const Color &p_default) {
	int idx = find_named_color(p_name);
	if (idx == -1) {
		return p_default;
	}
	return named_colors[idx].color;
}

int Color::find_named_color(const String &p_name) {
	String name = p_name;
	// Normalize name
	name = name.replace(" ", "");
	name = name.replace("-", "");
	name = name.replace("_", "");
	name = name.replace("'", "");
	name = name.replace(".", "");
	name = name.to_upper();

	int idx = 0;
	while (named_colors[idx].name != nullptr) {
		if (name == String(named_colors[idx].name).replace("_", "")) {
			return idx;
		}
		idx++;
	}

	return -1;
}

int Color::get_named_color_count() {
	int idx = 0;
	while (named_colors[idx].name != nullptr) {
		idx++;
	}
	return idx;
}

String Color::get_named_color_name(int p_idx) {
	ERR_FAIL_INDEX_V(p_idx, get_named_color_count(), "");
	return named_colors[p_idx].name;
}

Color Color::get_named_color(int p_idx) {
	ERR_FAIL_INDEX_V(p_idx, get_named_color_count(), Color());
	return named_colors[p_idx].color;
}

// For a version that errors on invalid values instead of returning
// a default color, use the Color(String) constructor instead.
Color Color::from_string(const String &p_string, const Color &p_default) {
	if (html_is_valid(p_string)) {
		return html(p_string);
	} else {
		return named(p_string, p_default);
	}
}

Color Color::from_hsv(float p_h, float p_s, float p_v, float p_alpha) {
	Color c;
	c.set_hsv(p_h, p_s, p_v, p_alpha);
	return c;
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

Color::operator String() const {
	return "(" + String::num(r, 4) + ", " + String::num(g, 4) + ", " + String::num(b, 4) + ", " + String::num(a, 4) + ")";
}

Color Color::operator+(const Color &p_color) const {
	return Color(
			r + p_color.r,
			g + p_color.g,
			b + p_color.b,
			a + p_color.a);
}

void Color::operator+=(const Color &p_color) {
	r = r + p_color.r;
	g = g + p_color.g;
	b = b + p_color.b;
	a = a + p_color.a;
}

Color Color::operator-(const Color &p_color) const {
	return Color(
			r - p_color.r,
			g - p_color.g,
			b - p_color.b,
			a - p_color.a);
}

void Color::operator-=(const Color &p_color) {
	r = r - p_color.r;
	g = g - p_color.g;
	b = b - p_color.b;
	a = a - p_color.a;
}

Color Color::operator*(const Color &p_color) const {
	return Color(
			r * p_color.r,
			g * p_color.g,
			b * p_color.b,
			a * p_color.a);
}

Color Color::operator*(float p_scalar) const {
	return Color(
			r * p_scalar,
			g * p_scalar,
			b * p_scalar,
			a * p_scalar);
}

void Color::operator*=(const Color &p_color) {
	r = r * p_color.r;
	g = g * p_color.g;
	b = b * p_color.b;
	a = a * p_color.a;
}

void Color::operator*=(float p_scalar) {
	r = r * p_scalar;
	g = g * p_scalar;
	b = b * p_scalar;
	a = a * p_scalar;
}

Color Color::operator/(const Color &p_color) const {
	return Color(
			r / p_color.r,
			g / p_color.g,
			b / p_color.b,
			a / p_color.a);
}

Color Color::operator/(float p_scalar) const {
	return Color(
			r / p_scalar,
			g / p_scalar,
			b / p_scalar,
			a / p_scalar);
}

void Color::operator/=(const Color &p_color) {
	r = r / p_color.r;
	g = g / p_color.g;
	b = b / p_color.b;
	a = a / p_color.a;
}

void Color::operator/=(float p_scalar) {
	r = r / p_scalar;
	g = g / p_scalar;
	b = b / p_scalar;
	a = a / p_scalar;
}

Color Color::operator-() const {
	return Color(
			1.0 - r,
			1.0 - g,
			1.0 - b,
			1.0 - a);
}
