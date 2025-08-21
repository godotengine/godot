/**************************************************************************/
/*  color.hpp                                                             */
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

#pragma once
#include "syscalls_fwd.hpp"
#include <cmath>

struct Color {
	float r;
	float g;
	float b;
	float a;

	template <typename... Args>
	Variant operator()(std::string_view method, Args &&...args);

	METHOD(Color, blend);
	METHOD(Color, clamp);
	METHOD(Color, darkened);
	METHOD(Color, from_hsv);
	METHOD(Color, from_ok_hsl);
	METHOD(Color, from_rgbe9995);
	METHOD(Color, from_string);
	METHOD(float, get_luminance);
	METHOD(Color, hex);
	METHOD(Color, hex64);
	METHOD(Color, html);
	METHOD(bool, html_is_valid);
	METHOD(Color, inverted);
	METHOD(bool, is_equal_approx);
	METHOD(Color, lerp);
	METHOD(Color, lightened);
	METHOD(Color, linear_to_srgb);
	METHOD(Color, srgb_to_linear);
	METHOD(int, to_abgr32);
	METHOD(int, to_abgr64);
	METHOD(int, to_argb32);
	METHOD(int, to_argb64);
	//METHOD(String, to_html);
	METHOD(int, to_rgba32);
	METHOD(int, to_rgba64);

	Color &operator+=(const Color &other);
	Color &operator-=(const Color &other);
	Color &operator*=(const Color &other);
	Color &operator/=(const Color &other);

	Color &operator+=(float other);
	Color &operator-=(float other);
	Color &operator*=(float other);
	Color &operator/=(float other);

	bool operator==(const Color &other) const {
		return __builtin_memcmp(this, &other, sizeof(Color)) == 0;
	}
	bool operator!=(const Color &other) const {
		return !(*this == other);
	}

	constexpr Color() :
			r(0), g(0), b(0), a(0) {}
	constexpr Color(float val) :
			r(val), g(val), b(val), a(val) {}
	constexpr Color(float r, float g, float b) :
			r(r), g(g), b(b), a(1) {}
	constexpr Color(float r, float g, float b, float a) :
			r(r), g(g), b(b), a(a) {}
	Color(std::string_view code);
	Color(std::string_view code, float a);

	static const Color ALICE_BLUE;
	static const Color ANTIQUE_WHITE;
	static const Color AQUA;
	static const Color AQUAMARINE;
	static const Color AZURE;
	static const Color BEIGE;
	static const Color BISQUE;
	static const Color BLACK;
	static const Color BLANCHED_ALMOND;
	static const Color BLUE;
	static const Color BLUE_VIOLET;
	static const Color BROWN;
	static const Color BURLYWOOD;
	static const Color CADET_BLUE;
	static const Color CHARTREUSE;
	static const Color CHOCOLATE;
	static const Color CORAL;
	static const Color CORNFLOWER_BLUE;
	static const Color CORNSILK;
	static const Color CRIMSON;
	static const Color CYAN;
	static const Color DARK_BLUE;
	static const Color DARK_CYAN;
	static const Color DARK_GOLDENROD;
	static const Color DARK_GRAY;
	static const Color DARK_GREEN;
	static const Color DARK_KHAKI;
	static const Color DARK_MAGENTA;
	static const Color DARK_OLIVE_GREEN;
	static const Color DARK_ORANGE;
	static const Color DARK_ORCHID;
	static const Color DARK_RED;
	static const Color DARK_SALMON;
	static const Color DARK_SEA_GREEN;
	static const Color DARK_SLATE_BLUE;
	static const Color DARK_SLATE_GRAY;
	static const Color DARK_TURQUOISE;
	static const Color DARK_VIOLET;
	static const Color DEEP_PINK;
	static const Color DEEP_SKY_BLUE;
	static const Color DIM_GRAY;
	static const Color DODGER_BLUE;
	static const Color FIREBRICK;
	static const Color FLORAL_WHITE;
	static const Color FOREST_GREEN;
	static const Color FUCHSIA;
	static const Color GAINSBORO;
	static const Color GHOST_WHITE;
	static const Color GOLD;
	static const Color GOLDENROD;
	static const Color GRAY;
	static const Color GREEN;
	static const Color GREEN_YELLOW;
	static const Color HONEYDEW;
	static const Color HOT_PINK;
	static const Color INDIAN_RED;
	static const Color INDIGO;
	static const Color IVORY;
	static const Color KHAKI;
	static const Color LAVENDER;
	static const Color LAVENDER_BLUSH;
	static const Color LAWN_GREEN;
	static const Color LEMON_CHIFFON;
	static const Color LIGHT_BLUE;
	static const Color LIGHT_CORAL;
	static const Color LIGHT_CYAN;
	static const Color LIGHT_GOLDENROD;
	static const Color LIGHT_GRAY;
	static const Color LIGHT_GREEN;
	static const Color LIGHT_PINK;
	static const Color LIGHT_SALMON;
	static const Color LIGHT_SEA_GREEN;
	static const Color LIGHT_SKY_BLUE;
	static const Color LIGHT_SLATE_GRAY;
	static const Color LIGHT_STEEL_BLUE;
	static const Color LIGHT_YELLOW;
	static const Color LIME;
	static const Color LIME_GREEN;
	static const Color LINEN;
	static const Color MAGENTA;
	static const Color MAROON;
	static const Color MEDIUM_AQUAMARINE;
	static const Color MEDIUM_BLUE;
	static const Color MEDIUM_ORCHID;
	static const Color MEDIUM_PURPLE;
	static const Color MEDIUM_SEA_GREEN;
	static const Color MEDIUM_SLATE_BLUE;
	static const Color MEDIUM_SPRING_GREEN;
	static const Color MEDIUM_TURQUOISE;
	static const Color MEDIUM_VIOLET_RED;
	static const Color MIDNIGHT_BLUE;
	static const Color MINT_CREAM;
	static const Color MISTY_ROSE;
	static const Color MOCCASIN;
	static const Color NAVAJO_WHITE;
	static const Color NAVY_BLUE;
	static const Color OLD_LACE;
	static const Color OLIVE;
	static const Color OLIVE_DRAB;
	static const Color ORANGE;
	static const Color ORANGE_RED;
	static const Color ORCHID;
	static const Color PALE_GOLDENROD;
	static const Color PALE_GREEN;
	static const Color PALE_TURQUOISE;
	static const Color PALE_VIOLET_RED;
	static const Color PAPAYA_WHIP;
	static const Color PEACH_PUFF;
	static const Color PERU;
	static const Color PINK;
	static const Color PLUM;
	static const Color POWDER_BLUE;
	static const Color PURPLE;
	static const Color REBECCA_PURPLE;
	static const Color RED;
	static const Color ROSY_BROWN;
	static const Color ROYAL_BLUE;
	static const Color SADDLE_BROWN;
	static const Color SALMON;
	static const Color SANDY_BROWN;
	static const Color SEA_GREEN;
	static const Color SEASHELL;
	static const Color SIENNA;
	static const Color SILVER;
	static const Color SKY_BLUE;
	static const Color SLATE_BLUE;
	static const Color SLATE_GRAY;
	static const Color SNOW;
	static const Color SPRING_GREEN;
	static const Color STEEL_BLUE;
	static const Color TAN;
	static const Color TEAL;
	static const Color THISTLE;
	static const Color TOMATO;
	static const Color TRANSPARENT;
	static const Color TURQUOISE;
	static const Color VIOLET;
	static const Color WEB_GRAY;
	static const Color WEB_GREEN;
	static const Color WEB_MAROON;
	static const Color WEB_PURPLE;
	static const Color WHEAT;
	static const Color WHITE;
	static const Color WHITE_SMOKE;
	static const Color YELLOW;
	static const Color YELLOW_GREEN;
};

inline constexpr auto operator+(const Color &a, float b) noexcept {
	return Color{ a.r + b, a.g + b, a.b + b, a.a + b };
}
inline constexpr auto operator-(const Color &a, float b) noexcept {
	return Color{ a.r - b, a.g - b, a.b - b, a.a - b };
}
inline constexpr auto operator*(const Color &a, float b) noexcept {
	return Color{ a.r * b, a.g * b, a.b * b, a.a * b };
}
inline constexpr auto operator/(const Color &a, float b) noexcept {
	return Color{ a.r / b, a.g / b, a.b / b, a.a / b };
}

inline constexpr auto operator+(const Color &a, const Color &b) noexcept {
	return Color{ a.r + b.r, a.g + b.g, a.b + b.b, a.a + b.a };
}
inline constexpr auto operator-(const Color &a, const Color &b) noexcept {
	return Color{ a.r - b.r, a.g - b.g, a.b - b.b, a.a - b.a };
}
inline constexpr auto operator*(const Color &a, const Color &b) noexcept {
	return Color{ a.r * b.r, a.g * b.g, a.b * b.b, a.a * b.a };
}
inline constexpr auto operator/(const Color &a, const Color &b) noexcept {
	return Color{ a.r / b.r, a.g / b.g, a.b / b.b, a.a / b.a };
}

inline Color &Color::operator+=(const Color &other) {
	r += other.r;
	g += other.g;
	b += other.b;
	a += other.a;
	return *this;
}
inline Color &Color::operator-=(const Color &other) {
	r -= other.r;
	g -= other.g;
	b -= other.b;
	a -= other.a;
	return *this;
}
inline Color &Color::operator*=(const Color &other) {
	r *= other.r;
	g *= other.g;
	b *= other.b;
	a *= other.a;
	return *this;
}
inline Color &Color::operator/=(const Color &other) {
	r /= other.r;
	g /= other.g;
	b /= other.b;
	a /= other.a;
	return *this;
}

inline Color &Color::operator+=(float other) {
	r += other;
	g += other;
	b += other;
	a += other;
	return *this;
}
inline Color &Color::operator-=(float other) {
	r -= other;
	g -= other;
	b -= other;
	a -= other;
	return *this;
}
inline Color &Color::operator*=(float other) {
	r *= other;
	g *= other;
	b *= other;
	a *= other;
	return *this;
}
inline Color &Color::operator/=(float other) {
	r /= other;
	g /= other;
	b /= other;
	a /= other;
	return *this;
}

inline constexpr Color const ALICE_BLUE = Color(0.941176, 0.972549, 1, 1);
inline constexpr Color const ANTIQUE_WHITE = Color(0.980392, 0.921569, 0.843137, 1);
inline constexpr Color const AQUA = Color(0, 1, 1, 1);
inline constexpr Color const AQUAMARINE = Color(0.498039, 1, 0.831373, 1);
inline constexpr Color const AZURE = Color(0.941176, 1, 1, 1);
inline constexpr Color const BEIGE = Color(0.960784, 0.960784, 0.862745, 1);
inline constexpr Color const BISQUE = Color(1, 0.894118, 0.768627, 1);
inline constexpr Color const BLACK = Color(0, 0, 0, 1);
inline constexpr Color const BLANCHED_ALMOND = Color(1, 0.921569, 0.803922, 1);
inline constexpr Color const BLUE = Color(0, 0, 1, 1);
inline constexpr Color const BLUE_VIOLET = Color(0.541176, 0.168627, 0.886275, 1);
inline constexpr Color const BROWN = Color(0.647059, 0.164706, 0.164706, 1);
inline constexpr Color const BURLYWOOD = Color(0.870588, 0.721569, 0.529412, 1);
inline constexpr Color const CADET_BLUE = Color(0.372549, 0.619608, 0.627451, 1);
inline constexpr Color const CHARTREUSE = Color(0.498039, 1, 0, 1);
inline constexpr Color const CHOCOLATE = Color(0.823529, 0.411765, 0.117647, 1);
inline constexpr Color const CORAL = Color(1, 0.498039, 0.313726, 1);
inline constexpr Color const CORNFLOWER_BLUE = Color(0.392157, 0.584314, 0.929412, 1);
inline constexpr Color const CORNSILK = Color(1, 0.972549, 0.862745, 1);
inline constexpr Color const CRIMSON = Color(0.862745, 0.0784314, 0.235294, 1);
inline constexpr Color const CYAN = Color(0, 1, 1, 1);
inline constexpr Color const DARK_BLUE = Color(0, 0, 0.545098, 1);
inline constexpr Color const DARK_CYAN = Color(0, 0.545098, 0.545098, 1);
inline constexpr Color const DARK_GOLDENROD = Color(0.721569, 0.52549, 0.0431373, 1);
inline constexpr Color const DARK_GRAY = Color(0.662745, 0.662745, 0.662745, 1);
inline constexpr Color const DARK_GREEN = Color(0, 0.392157, 0, 1);
inline constexpr Color const DARK_KHAKI = Color(0.741176, 0.717647, 0.419608, 1);
inline constexpr Color const DARK_MAGENTA = Color(0.545098, 0, 0.545098, 1);
inline constexpr Color const DARK_OLIVE_GREEN = Color(0.333333, 0.419608, 0.184314, 1);
inline constexpr Color const DARK_ORANGE = Color(1, 0.54902, 0, 1);
inline constexpr Color const DARK_ORCHID = Color(0.6, 0.196078, 0.8, 1);
inline constexpr Color const DARK_RED = Color(0.545098, 0, 0, 1);
inline constexpr Color const DARK_SALMON = Color(0.913725, 0.588235, 0.478431, 1);
inline constexpr Color const DARK_SEA_GREEN = Color(0.560784, 0.737255, 0.560784, 1);
inline constexpr Color const DARK_SLATE_BLUE = Color(0.282353, 0.239216, 0.545098, 1);
inline constexpr Color const DARK_SLATE_GRAY = Color(0.184314, 0.309804, 0.309804, 1);
inline constexpr Color const DARK_TURQUOISE = Color(0, 0.807843, 0.819608, 1);
inline constexpr Color const DARK_VIOLET = Color(0.580392, 0, 0.827451, 1);
inline constexpr Color const DEEP_PINK = Color(1, 0.0784314, 0.576471, 1);
inline constexpr Color const DEEP_SKY_BLUE = Color(0, 0.74902, 1, 1);
inline constexpr Color const DIM_GRAY = Color(0.411765, 0.411765, 0.411765, 1);
inline constexpr Color const DODGER_BLUE = Color(0.117647, 0.564706, 1, 1);
inline constexpr Color const FIREBRICK = Color(0.698039, 0.133333, 0.133333, 1);
inline constexpr Color const FLORAL_WHITE = Color(1, 0.980392, 0.941176, 1);
inline constexpr Color const FOREST_GREEN = Color(0.133333, 0.545098, 0.133333, 1);
inline constexpr Color const FUCHSIA = Color(1, 0, 1, 1);
inline constexpr Color const GAINSBORO = Color(0.862745, 0.862745, 0.862745, 1);
inline constexpr Color const GHOST_WHITE = Color(0.972549, 0.972549, 1, 1);
inline constexpr Color const GOLD = Color(1, 0.843137, 0, 1);
inline constexpr Color const GOLDENROD = Color(0.854902, 0.647059, 0.12549, 1);
inline constexpr Color const GRAY = Color(0.745098, 0.745098, 0.745098, 1);
inline constexpr Color const GREEN = Color(0, 1, 0, 1);
inline constexpr Color const GREEN_YELLOW = Color(0.678431, 1, 0.184314, 1);
inline constexpr Color const HONEYDEW = Color(0.941176, 1, 0.941176, 1);
inline constexpr Color const HOT_PINK = Color(1, 0.411765, 0.705882, 1);
inline constexpr Color const INDIAN_RED = Color(0.803922, 0.360784, 0.360784, 1);
inline constexpr Color const INDIGO = Color(0.294118, 0, 0.509804, 1);
inline constexpr Color const IVORY = Color(1, 1, 0.941176, 1);
inline constexpr Color const KHAKI = Color(0.941176, 0.901961, 0.54902, 1);
inline constexpr Color const LAVENDER = Color(0.901961, 0.901961, 0.980392, 1);
inline constexpr Color const LAVENDER_BLUSH = Color(1, 0.941176, 0.960784, 1);
inline constexpr Color const LAWN_GREEN = Color(0.486275, 0.988235, 0, 1);
inline constexpr Color const LEMON_CHIFFON = Color(1, 0.980392, 0.803922, 1);
inline constexpr Color const LIGHT_BLUE = Color(0.678431, 0.847059, 0.901961, 1);
inline constexpr Color const LIGHT_CORAL = Color(0.941176, 0.501961, 0.501961, 1);
inline constexpr Color const LIGHT_CYAN = Color(0.878431, 1, 1, 1);
inline constexpr Color const LIGHT_GOLDENROD = Color(0.980392, 0.980392, 0.823529, 1);
inline constexpr Color const LIGHT_GRAY = Color(0.827451, 0.827451, 0.827451, 1);
inline constexpr Color const LIGHT_GREEN = Color(0.564706, 0.933333, 0.564706, 1);
inline constexpr Color const LIGHT_PINK = Color(1, 0.713726, 0.756863, 1);
inline constexpr Color const LIGHT_SALMON = Color(1, 0.627451, 0.478431, 1);
inline constexpr Color const LIGHT_SEA_GREEN = Color(0.12549, 0.698039, 0.666667, 1);
inline constexpr Color const LIGHT_SKY_BLUE = Color(0.529412, 0.807843, 0.980392, 1);
inline constexpr Color const LIGHT_SLATE_GRAY = Color(0.466667, 0.533333, 0.6, 1);
inline constexpr Color const LIGHT_STEEL_BLUE = Color(0.690196, 0.768627, 0.870588, 1);
inline constexpr Color const LIGHT_YELLOW = Color(1, 1, 0.878431, 1);
inline constexpr Color const LIME = Color(0, 1, 0, 1);
inline constexpr Color const LIME_GREEN = Color(0.196078, 0.803922, 0.196078, 1);
inline constexpr Color const LINEN = Color(0.980392, 0.941176, 0.901961, 1);
inline constexpr Color const MAGENTA = Color(1, 0, 1, 1);
inline constexpr Color const MAROON = Color(0.690196, 0.188235, 0.376471, 1);
inline constexpr Color const MEDIUM_AQUAMARINE = Color(0.4, 0.803922, 0.666667, 1);
inline constexpr Color const MEDIUM_BLUE = Color(0, 0, 0.803922, 1);
inline constexpr Color const MEDIUM_ORCHID = Color(0.729412, 0.333333, 0.827451, 1);
inline constexpr Color const MEDIUM_PURPLE = Color(0.576471, 0.439216, 0.858824, 1);
inline constexpr Color const MEDIUM_SEA_GREEN = Color(0.235294, 0.701961, 0.443137, 1);
inline constexpr Color const MEDIUM_SLATE_BLUE = Color(0.482353, 0.407843, 0.933333, 1);
inline constexpr Color const MEDIUM_SPRING_GREEN = Color(0, 0.980392, 0.603922, 1);
inline constexpr Color const MEDIUM_TURQUOISE = Color(0.282353, 0.819608, 0.8, 1);
inline constexpr Color const MEDIUM_VIOLET_RED = Color(0.780392, 0.0823529, 0.521569, 1);
inline constexpr Color const MIDNIGHT_BLUE = Color(0.0980392, 0.0980392, 0.439216, 1);
inline constexpr Color const MINT_CREAM = Color(0.960784, 1, 0.980392, 1);
inline constexpr Color const MISTY_ROSE = Color(1, 0.894118, 0.882353, 1);
inline constexpr Color const MOCCASIN = Color(1, 0.894118, 0.709804, 1);
inline constexpr Color const NAVAJO_WHITE = Color(1, 0.870588, 0.678431, 1);
inline constexpr Color const NAVY_BLUE = Color(0, 0, 0.501961, 1);
inline constexpr Color const OLD_LACE = Color(0.992157, 0.960784, 0.901961, 1);
inline constexpr Color const OLIVE = Color(0.501961, 0.501961, 0, 1);
inline constexpr Color const OLIVE_DRAB = Color(0.419608, 0.556863, 0.137255, 1);
inline constexpr Color const ORANGE = Color(1, 0.647059, 0, 1);
inline constexpr Color const ORANGE_RED = Color(1, 0.270588, 0, 1);
inline constexpr Color const ORCHID = Color(0.854902, 0.439216, 0.839216, 1);
inline constexpr Color const PALE_GOLDENROD = Color(0.933333, 0.909804, 0.666667, 1);
inline constexpr Color const PALE_GREEN = Color(0.596078, 0.984314, 0.596078, 1);
inline constexpr Color const PALE_TURQUOISE = Color(0.686275, 0.933333, 0.933333, 1);
inline constexpr Color const PALE_VIOLET_RED = Color(0.858824, 0.439216, 0.576471, 1);
inline constexpr Color const PAPAYA_WHIP = Color(1, 0.937255, 0.835294, 1);
inline constexpr Color const PEACH_PUFF = Color(1, 0.854902, 0.72549, 1);
inline constexpr Color const PERU = Color(0.803922, 0.521569, 0.247059, 1);
inline constexpr Color const PINK = Color(1, 0.752941, 0.796078, 1);
inline constexpr Color const PLUM = Color(0.866667, 0.627451, 0.866667, 1);
inline constexpr Color const POWDER_BLUE = Color(0.690196, 0.878431, 0.901961, 1);
inline constexpr Color const PURPLE = Color(0.627451, 0.12549, 0.941176, 1);
inline constexpr Color const REBECCA_PURPLE = Color(0.4, 0.2, 0.6, 1);
inline constexpr Color const RED = Color(1, 0, 0, 1);
inline constexpr Color const ROSY_BROWN = Color(0.737255, 0.560784, 0.560784, 1);
inline constexpr Color const ROYAL_BLUE = Color(0.254902, 0.411765, 0.882353, 1);
inline constexpr Color const SADDLE_BROWN = Color(0.545098, 0.270588, 0.0745098, 1);
inline constexpr Color const SALMON = Color(0.980392, 0.501961, 0.447059, 1);
inline constexpr Color const SANDY_BROWN = Color(0.956863, 0.643137, 0.376471, 1);
inline constexpr Color const SEA_GREEN = Color(0.180392, 0.545098, 0.341176, 1);
inline constexpr Color const SEASHELL = Color(1, 0.960784, 0.933333, 1);
inline constexpr Color const SIENNA = Color(0.627451, 0.321569, 0.176471, 1);
inline constexpr Color const SILVER = Color(0.752941, 0.752941, 0.752941, 1);
inline constexpr Color const SKY_BLUE = Color(0.529412, 0.807843, 0.921569, 1);
inline constexpr Color const SLATE_BLUE = Color(0.415686, 0.352941, 0.803922, 1);
inline constexpr Color const SLATE_GRAY = Color(0.439216, 0.501961, 0.564706, 1);
inline constexpr Color const SNOW = Color(1, 0.980392, 0.980392, 1);
inline constexpr Color const SPRING_GREEN = Color(0, 1, 0.498039, 1);
inline constexpr Color const STEEL_BLUE = Color(0.27451, 0.509804, 0.705882, 1);
inline constexpr Color const TAN = Color(0.823529, 0.705882, 0.54902, 1);
inline constexpr Color const TEAL = Color(0, 0.501961, 0.501961, 1);
inline constexpr Color const THISTLE = Color(0.847059, 0.74902, 0.847059, 1);
inline constexpr Color const TOMATO = Color(1, 0.388235, 0.278431, 1);
inline constexpr Color const TRANSPARENT = Color(1, 1, 1, 0);
inline constexpr Color const TURQUOISE = Color(0.25098, 0.878431, 0.815686, 1);
inline constexpr Color const VIOLET = Color(0.933333, 0.509804, 0.933333, 1);
inline constexpr Color const WEB_GRAY = Color(0.501961, 0.501961, 0.501961, 1);
inline constexpr Color const WEB_GREEN = Color(0, 0.501961, 0, 1);
inline constexpr Color const WEB_MAROON = Color(0.501961, 0, 0, 1);
inline constexpr Color const WEB_PURPLE = Color(0.501961, 0, 0.501961, 1);
inline constexpr Color const WHEAT = Color(0.960784, 0.870588, 0.701961, 1);
inline constexpr Color const WHITE = Color(1, 1, 1, 1);
inline constexpr Color const WHITE_SMOKE = Color(0.960784, 0.960784, 0.960784, 1);
inline constexpr Color const YELLOW = Color(1, 1, 0, 1);
inline constexpr Color const YELLOW_GREEN = Color(0.603922, 0.803922, 0.196078, 1);
