/**************************************************************************/
/*  color.cpp                                                             */
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

#include "color.h"

#include "color_names.inc"
#include "core/math/math_funcs.h"
#include "core/string/ustring.h"
#include "core/templates/hash_map.h"

#include "thirdparty/misc/ok_color.h"

uint32_t Color::to_argb32() const {
	uint32_t c = (uint8_t)Math::round(a * 255.0f);
	c <<= 8;
	c |= (uint8_t)Math::round(r * 255.0f);
	c <<= 8;
	c |= (uint8_t)Math::round(g * 255.0f);
	c <<= 8;
	c |= (uint8_t)Math::round(b * 255.0f);

	return c;
}

uint32_t Color::to_abgr32() const {
	uint32_t c = (uint8_t)Math::round(a * 255.0f);
	c <<= 8;
	c |= (uint8_t)Math::round(b * 255.0f);
	c <<= 8;
	c |= (uint8_t)Math::round(g * 255.0f);
	c <<= 8;
	c |= (uint8_t)Math::round(r * 255.0f);

	return c;
}

uint32_t Color::to_rgba32() const {
	uint32_t c = (uint8_t)Math::round(r * 255.0f);
	c <<= 8;
	c |= (uint8_t)Math::round(g * 255.0f);
	c <<= 8;
	c |= (uint8_t)Math::round(b * 255.0f);
	c <<= 8;
	c |= (uint8_t)Math::round(a * 255.0f);

	return c;
}

uint64_t Color::to_abgr64() const {
	uint64_t c = (uint16_t)Math::round(a * 65535.0f);
	c <<= 16;
	c |= (uint16_t)Math::round(b * 65535.0f);
	c <<= 16;
	c |= (uint16_t)Math::round(g * 65535.0f);
	c <<= 16;
	c |= (uint16_t)Math::round(r * 65535.0f);

	return c;
}

uint64_t Color::to_argb64() const {
	uint64_t c = (uint16_t)Math::round(a * 65535.0f);
	c <<= 16;
	c |= (uint16_t)Math::round(r * 65535.0f);
	c <<= 16;
	c |= (uint16_t)Math::round(g * 65535.0f);
	c <<= 16;
	c |= (uint16_t)Math::round(b * 65535.0f);

	return c;
}

uint64_t Color::to_rgba64() const {
	uint64_t c = (uint16_t)Math::round(r * 65535.0f);
	c <<= 16;
	c |= (uint16_t)Math::round(g * 65535.0f);
	c <<= 16;
	c |= (uint16_t)Math::round(b * 65535.0f);
	c <<= 16;
	c |= (uint16_t)Math::round(a * 65535.0f);

	return c;
}

void _append_hex(float p_val, char32_t *string) {
	int v = Math::round(p_val * 255.0f);
	v = CLAMP(v, 0, 255);

	string[0] = hex_char_table_lower[(v >> 4) & 0xF];
	string[1] = hex_char_table_lower[v & 0xF];
}

String Color::to_html(bool p_alpha) const {
	String txt;
	txt.resize_uninitialized(p_alpha ? 9 : 7);
	char32_t *ptr = txt.ptrw();

	_append_hex(r, ptr + 0);
	_append_hex(g, ptr + 2);
	_append_hex(b, ptr + 4);
	if (p_alpha) {
		_append_hex(a, ptr + 6);
	}
	ptr[txt.size() - 1] = '\0';

	return txt;
}

float Color::get_h() const {
	float min = MIN(r, g);
	min = MIN(min, b);
	float max = MAX(r, g);
	max = MAX(max, b);

	float delta = max - min;

	if (delta == 0.0f) {
		return 0.0f;
	}

	float h;
	if (r == max) {
		h = (g - b) / delta; // between yellow & magenta
	} else if (g == max) {
		h = 2 + (b - r) / delta; // between cyan & yellow
	} else {
		h = 4 + (r - g) / delta; // between magenta & cyan
	}

	h /= 6.0f;
	if (h < 0.0f) {
		h += 1.0f;
	}

	return h;
}

float Color::get_s() const {
	float min = MIN(r, g);
	min = MIN(min, b);
	float max = MAX(r, g);
	max = MAX(max, b);

	float delta = max - min;

	return (max != 0.0f) ? (delta / max) : 0.0f;
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

	if (p_s == 0.0f) {
		// Achromatic (gray)
		r = g = b = p_v;
		return;
	}

	p_h *= 6.0f;
	p_h = Math::fmod(p_h, 6);
	i = Math::floor(p_h);

	f = p_h - i;
	p = p_v * (1.0f - p_s);
	q = p_v * (1.0f - p_s * f);
	t = p_v * (1.0f - p_s * (1.0f - f));

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

void Color::set_ok_hsl(float p_h, float p_s, float p_l, float p_alpha) {
	ok_color::HSL hsl;
	hsl.h = p_h;
	hsl.s = p_s;
	hsl.l = p_l;
	ok_color::RGB rgb = ok_color::okhsl_to_srgb(hsl);
	Color c = Color(rgb.r, rgb.g, rgb.b, p_alpha).clamp();
	r = c.r;
	g = c.g;
	b = c.b;
	a = c.a;
}

void Color::set_ok_hsv(float p_h, float p_s, float p_v, float p_alpha) {
	ok_color::HSV hsv;
	hsv.h = p_h;
	hsv.s = p_s;
	hsv.v = p_v;
	ok_color::RGB rgb = ok_color::okhsv_to_srgb(hsv);
	Color c = Color(rgb.r, rgb.g, rgb.b, p_alpha).clamp();
	r = c.r;
	g = c.g;
	b = c.b;
	a = c.a;
}

bool Color::is_equal_approx(const Color &p_color) const {
	return Math::is_equal_approx(r, p_color.r) && Math::is_equal_approx(g, p_color.g) && Math::is_equal_approx(b, p_color.b) && Math::is_equal_approx(a, p_color.a);
}

bool Color::is_same(const Color &p_color) const {
	return Math::is_same(r, p_color.r) && Math::is_same(g, p_color.g) && Math::is_same(b, p_color.b) && Math::is_same(a, p_color.a);
}

Color Color::clamp(const Color &p_min, const Color &p_max) const {
	return Color(
			CLAMP(r, p_min.r, p_max.r),
			CLAMP(g, p_min.g, p_max.g),
			CLAMP(b, p_min.b, p_max.b),
			CLAMP(a, p_min.a, p_max.a));
}

void Color::invert() {
	r = 1.0f - r;
	g = 1.0f - g;
	b = 1.0f - b;
}

Color Color::hex(uint32_t p_hex) {
	float a = (p_hex & 0xFF) / 255.0f;
	p_hex >>= 8;
	float b = (p_hex & 0xFF) / 255.0f;
	p_hex >>= 8;
	float g = (p_hex & 0xFF) / 255.0f;
	p_hex >>= 8;
	float r = (p_hex & 0xFF) / 255.0f;

	return Color(r, g, b, a);
}

Color Color::hex64(uint64_t p_hex) {
	float a = (p_hex & 0xFFFF) / 65535.0f;
	p_hex >>= 16;
	float b = (p_hex & 0xFFFF) / 65535.0f;
	p_hex >>= 16;
	float g = (p_hex & 0xFFFF) / 65535.0f;
	p_hex >>= 16;
	float r = (p_hex & 0xFFFF) / 65535.0f;

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
	if (p_rgba.is_empty()) {
		return Color();
	}

	const int current_pos = (p_rgba[0] == '#') ? 1 : 0;
	const int num_of_digits = p_rgba.length() - current_pos;

	float r, g, b, a = 1.0f;

	if (num_of_digits == 3) {
		// #rgb
		r = _parse_col4(p_rgba, current_pos) / 15.0f;
		g = _parse_col4(p_rgba, current_pos + 1) / 15.0f;
		b = _parse_col4(p_rgba, current_pos + 2) / 15.0f;
	} else if (num_of_digits == 4) {
		r = _parse_col4(p_rgba, current_pos) / 15.0f;
		g = _parse_col4(p_rgba, current_pos + 1) / 15.0f;
		b = _parse_col4(p_rgba, current_pos + 2) / 15.0f;
		a = _parse_col4(p_rgba, current_pos + 3) / 15.0f;
	} else if (num_of_digits == 6) {
		r = _parse_col8(p_rgba, current_pos) / 255.0f;
		g = _parse_col8(p_rgba, current_pos + 2) / 255.0f;
		b = _parse_col8(p_rgba, current_pos + 4) / 255.0f;
	} else if (num_of_digits == 8) {
		r = _parse_col8(p_rgba, current_pos) / 255.0f;
		g = _parse_col8(p_rgba, current_pos + 2) / 255.0f;
		b = _parse_col8(p_rgba, current_pos + 4) / 255.0f;
		a = _parse_col8(p_rgba, current_pos + 6) / 255.0f;
	} else {
		ERR_FAIL_V_MSG(Color(), "Invalid color code: " + p_rgba + ".");
	}

	ERR_FAIL_COND_V_MSG(r < 0.0f, Color(), "Invalid color code: " + p_rgba + ".");
	ERR_FAIL_COND_V_MSG(g < 0.0f, Color(), "Invalid color code: " + p_rgba + ".");
	ERR_FAIL_COND_V_MSG(b < 0.0f, Color(), "Invalid color code: " + p_rgba + ".");
	ERR_FAIL_COND_V_MSG(a < 0.0f, Color(), "Invalid color code: " + p_rgba + ".");

	return Color(r, g, b, a);
}

bool Color::html_is_valid(const String &p_color) {
	String color = p_color;

	if (color.is_empty()) {
		return false;
	}

	const int current_pos = (color[0] == '#') ? 1 : 0;
	const int len = color.length();
	const int num_of_digits = len - current_pos;
	if (!(num_of_digits == 3 || num_of_digits == 4 || num_of_digits == 6 || num_of_digits == 8)) {
		return false;
	}

	// Check if each hex digit is valid.
	for (int i = current_pos; i < len; i++) {
		if (!is_hex_digit(p_color[i])) {
			return false;
		}
	}

	return true;
}

Color Color::named(const String &p_name) {
	int idx = find_named_color(p_name);
	if (idx == -1) {
		ERR_FAIL_V_MSG(Color(), "Invalid color name: " + p_name + ".");
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
	// Normalize name.
	name = name.remove_chars(" -_'.");
	name = name.to_upper();

	static HashMap<String, int> named_colors_hashmap;
	if (unlikely(named_colors_hashmap.is_empty())) {
		const int named_color_count = get_named_color_count();
		for (int i = 0; i < named_color_count; i++) {
			named_colors_hashmap[String(named_colors[i].name).remove_char('_')] = i;
		}
	}

	const HashMap<String, int>::ConstIterator E = named_colors_hashmap.find(name);
	if (E) {
		return E->value;
	}

	return -1;
}

int Color::get_named_color_count() {
	return std_size(named_colors);
}

String Color::get_named_color_name(int p_idx) {
	ERR_FAIL_INDEX_V(p_idx, get_named_color_count(), "");
	return named_colors[p_idx].name;
}

Color Color::get_named_color(int p_idx) {
	ERR_FAIL_INDEX_V(p_idx, get_named_color_count(), Color());
	return named_colors[p_idx].color;
}

String Color::to_named() const {
	NamedColor closest_named_color = {};
	float closest_distance_squared = Math::INF;

	for (const NamedColor &named_color : named_colors) {
		Color delta = named_color.color - *this;
		float distance_squared = (delta.r * delta.r) + (delta.g * delta.g) + (delta.b * delta.b) + (delta.a * delta.a);

		if (distance_squared == 0) {
			return named_color.name;
		}
		if (distance_squared < closest_distance_squared) {
			closest_named_color = named_color;
			closest_distance_squared = distance_squared;
		}
	}

	return closest_named_color.name;
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
	float m = Math::pow(2.0f, e - 15.0f - 9.0f);

	float rd = r * m;
	float gd = g * m;
	float bd = b * m;

	return Color(rd, gd, bd, 1.0f);
}

Color Color::from_rgba8(int64_t p_r8, int64_t p_g8, int64_t p_b8, int64_t p_a8) {
	return Color(p_r8 / 255.0f, p_g8 / 255.0f, p_b8 / 255.0f, p_a8 / 255.0f);
}

Color::operator String() const {
	return "(" + String::num(r, 4) + ", " + String::num(g, 4) + ", " + String::num(b, 4) + ", " + String::num(a, 4) + ")";
}

Color Color::from_ok_hsl(float p_h, float p_s, float p_l, float p_alpha) {
	Color c;
	c.set_ok_hsl(p_h, p_s, p_l, p_alpha);
	return c;
}

Color Color::from_ok_hsv(float p_h, float p_s, float p_l, float p_alpha) {
	Color c;
	c.set_ok_hsv(p_h, p_s, p_l, p_alpha);
	return c;
}

float Color::get_ok_hsl_h() const {
	ok_color::RGB rgb;
	rgb.r = r;
	rgb.g = g;
	rgb.b = b;
	ok_color::HSL ok_hsl = ok_color::srgb_to_okhsl(rgb);
	if (Math::is_nan(ok_hsl.h)) {
		return 0.0f;
	}
	return CLAMP(ok_hsl.h, 0.0f, 1.0f);
}

float Color::get_ok_hsl_s() const {
	ok_color::RGB rgb;
	rgb.r = r;
	rgb.g = g;
	rgb.b = b;
	ok_color::HSL ok_hsl = ok_color::srgb_to_okhsl(rgb);
	if (Math::is_nan(ok_hsl.s)) {
		return 0.0f;
	}
	return CLAMP(ok_hsl.s, 0.0f, 1.0f);
}

float Color::get_ok_hsl_l() const {
	ok_color::RGB rgb;
	rgb.r = r;
	rgb.g = g;
	rgb.b = b;
	ok_color::HSL ok_hsl = ok_color::srgb_to_okhsl(rgb);
	if (Math::is_nan(ok_hsl.l)) {
		return 0.0f;
	}
	return CLAMP(ok_hsl.l, 0.0f, 1.0f);
}
