/**************************************************************************/
/*  color_mode.cpp                                                        */
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

#include "color_mode.h"

#include "core/math/color.h"
#include "scene/gui/slider.h"
#include "scene/resources/gradient_texture.h"

ColorMode::ColorMode(ColorPicker *p_color_picker) {
	color_picker = p_color_picker;
}

String ColorModeRGB::get_slider_label(int idx) const {
	ERR_FAIL_INDEX_V_MSG(idx, get_slider_count(), String(), "Couldn't get slider label.");
	return labels[idx];
}

float ColorModeRGB::get_slider_value(int idx) const {
	ERR_FAIL_INDEX_V_MSG(idx, get_slider_count(), 0, "Couldn't get slider value.");
	return color_picker->color_normalized.components[idx] * 255;
}

Color ColorModeRGB::get_color() const {
	Vector<float> values = color_picker->get_active_slider_values();
	Color color;
	for (int i = 0; i < 4; i++) {
		color.components[i] = values[i] / 255.0;
	}
	return color;
}

void ColorModeRGB::_greater_value_inputted() {
	HSlider **sliders = color_picker->sliders;
	Color color_prev = color_picker->color;
	for (int i = 0; i < 3; i++) {
		if (sliders[i]->get_value() > 255) {
			color_prev.components[i] = sliders[i]->get_value() / 255.0;
		}
	}
	Color linear_color = color_prev.srgb_to_linear();
	float multiplier = MAX(1, MAX(MAX(linear_color.r, linear_color.g), linear_color.b));
	Color srgb = Color(linear_color.r / multiplier, linear_color.g / multiplier, linear_color.b / multiplier, linear_color.a).linear_to_srgb();
	sliders[0]->set_value_no_signal(srgb.r * 255);
	sliders[1]->set_value_no_signal(srgb.g * 255);
	sliders[2]->set_value_no_signal(srgb.b * 255);

	color_picker->intensity = Math::log2(multiplier);
	color_picker->intensity_slider->set_value_no_signal(color_picker->intensity);
}

void ColorModeRGB::slider_draw(int p_which) {
	Vector<Vector2> pos;
	pos.resize(4);
	Vector<Color> col;
	col.resize(4);
	HSlider *slider = color_picker->get_slider(p_which);
	Size2 size = slider->get_size();
	Color left_color;
	Color right_color;
	Color color = color_picker->color_normalized;
	const real_t margin = 16 * color_picker->theme_cache.base_scale;

	left_color = Color(
			p_which == 0 ? 0 : color.r,
			p_which == 1 ? 0 : color.g,
			p_which == 2 ? 0 : color.b);
	right_color = Color(
			p_which == 0 ? 1 : color.r,
			p_which == 1 ? 1 : color.g,
			p_which == 2 ? 1 : color.b);

	if (rgb_texture[p_which].is_null()) {
		rgb_texture[p_which].instantiate();
		rgb_texture[p_which]->set_width(400);
		rgb_texture[p_which]->set_height(6);
	}

	Ref<GradientTexture2D> gradient_texture = rgb_texture[p_which];

	Ref<Gradient> gradient = gradient_texture->get_gradient();
	if (gradient.is_null()) {
		gradient.instantiate();
		PackedFloat32Array offsets = { 0, 1 };
		gradient->set_offsets(offsets);
		gradient->set_interpolation_color_space(Gradient::ColorSpace::GRADIENT_COLOR_SPACE_SRGB);
		gradient_texture->set_gradient(gradient);
	}

	PackedColorArray colors = { left_color, right_color };
	gradient->set_colors(colors);

	slider->draw_texture_rect(gradient_texture, Rect2(Vector2(), Vector2(size.x, margin)), false);
}

void ColorModeHSV::_value_changed() {
	Vector<float> values = color_picker->get_active_slider_values();

	if (values[1] > 0 || values[0] != cached_hue) {
		cached_hue = values[0];
	}
	if (values[2] > 0 || values[1] != cached_saturation) {
		cached_saturation = values[1];
	}

	// Cache real HSV values in ColorPicker.
	color_picker->h = color_picker->sliders[0]->get_value() / 360.0;
	color_picker->s = color_picker->sliders[1]->get_value() / 100.0;
	color_picker->v = color_picker->sliders[2]->get_value() / 100.0;

	color_picker->hsv_cached = true;
}

String ColorModeHSV::get_slider_label(int idx) const {
	ERR_FAIL_INDEX_V_MSG(idx, get_slider_count(), String(), "Couldn't get slider label.");
	return labels[idx];
}

float ColorModeHSV::get_slider_max(int idx) const {
	ERR_FAIL_INDEX_V_MSG(idx, get_slider_count(), 0, "Couldn't get slider max value.");
	return slider_max[idx];
}

float ColorModeHSV::get_slider_value(int idx) const {
	switch (idx) {
		case 0: {
			if (color_picker->color_normalized.get_s() > 0) {
				return color_picker->color_normalized.get_h() * 360.0;
			} else {
				return cached_hue;
			}
		}
		case 1: {
			if (color_picker->color_normalized.get_v() > 0) {
				return color_picker->color_normalized.get_s() * 100.0;
			} else {
				return cached_saturation;
			}
		}
		case 2:
			return color_picker->color_normalized.get_v() * 100.0;
		default:
			ERR_FAIL_V_MSG(0, "Couldn't get slider value.");
	}
}

Color ColorModeHSV::get_color() const {
	Vector<float> values = color_picker->get_active_slider_values();
	Color color = Color::from_hsv(values[0] / 360.0, values[1] / 100.0, values[2] / 100.0, values[3] / 255.0);
	return color;
}

void ColorModeHSV::slider_draw(int p_which) {
	Vector<Vector2> pos;
	pos.resize(4);
	Vector<Color> col;
	col.resize(4);
	HSlider *slider = color_picker->get_slider(p_which);
	Size2 size = slider->get_size();
	Color left_color;
	Color right_color;
	Color color = color_picker->color_normalized;
	const real_t margin = 16 * color_picker->theme_cache.base_scale;

	if (p_which == 0) {
		float v = color.get_v();
		left_color = Color(v, v, v);
		right_color = left_color;
	} else {
		Color s_col;
		Color v_col;
		s_col.set_hsv(color.get_h(), 0, color.get_v());
		left_color = (p_which == 1) ? s_col : Color(0, 0, 0);

		float s_col_hue = (Math::is_zero_approx(color.get_s())) ? cached_hue / 360.0 : color.get_h();
		s_col.set_hsv(s_col_hue, 1, color.get_v());
		v_col.set_hsv(color.get_h(), color.get_s(), 1);
		right_color = (p_which == 1) ? s_col : v_col;
	}
	col.set(0, left_color);
	col.set(1, right_color);
	col.set(2, right_color);
	col.set(3, left_color);
	pos.set(0, Vector2(0, 0));
	pos.set(1, Vector2(size.x, 0));
	pos.set(2, Vector2(size.x, margin));
	pos.set(3, Vector2(0, margin));

	slider->draw_polygon(pos, col);

	if (p_which == 0) { // H
		Ref<Texture2D> hue = color_picker->theme_cache.color_hue;
		slider->draw_texture_rect(hue, Rect2(Vector2(), Vector2(size.x, margin)), false, Color::from_hsv(0, 0, color.get_v(), color.get_s()));
	}
}

String ColorModeLinear::get_slider_label(int idx) const {
	ERR_FAIL_INDEX_V_MSG(idx, get_slider_count(), String(), "Couldn't get slider label.");
	return labels[idx];
}

float ColorModeLinear::get_slider_max(int idx) const {
	ERR_FAIL_INDEX_V_MSG(idx, get_slider_count(), 0, "Couldn't get slider max value.");
	return slider_max[idx];
}

float ColorModeLinear::get_slider_value(int idx) const {
	ERR_FAIL_INDEX_V_MSG(idx, get_slider_count(), 0, "Couldn't get slider value.");
	Color color = color_picker->color_normalized.srgb_to_linear();
	return color.components[idx];
}

float ColorModeLinear::get_alpha_slider_max() const {
	return 1;
}

float ColorModeLinear::get_alpha_slider_value() const {
	return color_picker->get_pick_color().a;
}

Color ColorModeLinear::get_color() const {
	Vector<float> values = color_picker->get_active_slider_values();
	Color color;
	for (int i = 0; i < 4; i++) {
		color.components[i] = values[i];
	}
	return color.linear_to_srgb();
}

void ColorModeLinear::_greater_value_inputted() {
	HSlider **sliders = color_picker->sliders;
	Color color_prev = color_picker->color;
	Color linear_color = color_prev.srgb_to_linear();
	for (int i = 0; i < 3; i++) {
		if (sliders[i]->get_value() > 1 + CMP_EPSILON) {
			linear_color.components[i] = sliders[i]->get_value();
		}
	}

	float multiplier = MAX(1, MAX(MAX(linear_color.r, linear_color.g), linear_color.b));

	sliders[0]->set_value_no_signal(linear_color.r / multiplier);
	sliders[1]->set_value_no_signal(linear_color.g / multiplier);
	sliders[2]->set_value_no_signal(linear_color.b / multiplier);

	color_picker->intensity = Math::log2(multiplier);
	color_picker->intensity_slider->set_value_no_signal(color_picker->intensity);
}

void ColorModeLinear::slider_draw(int p_which) {
	Vector<Vector2> pos;
	pos.resize(4);
	Vector<Color> col;
	col.resize(4);
	HSlider *slider = color_picker->get_slider(p_which);
	Size2 size = slider->get_size();
	Color left_color;
	Color right_color;
	Color color = color_picker->color_normalized;
	const real_t margin = 16 * color_picker->theme_cache.base_scale;

	left_color = Color(
			p_which == 0 ? 0 : color.r,
			p_which == 1 ? 0 : color.g,
			p_which == 2 ? 0 : color.b);
	right_color = Color(
			p_which == 0 ? 1 : color.r,
			p_which == 1 ? 1 : color.g,
			p_which == 2 ? 1 : color.b);

	if (rgb_texture[p_which].is_null()) {
		rgb_texture[p_which].instantiate();
		rgb_texture[p_which]->set_width(400);
		rgb_texture[p_which]->set_height(6);
	}

	Ref<GradientTexture2D> gradient_texture = rgb_texture[p_which];

	Ref<Gradient> gradient = gradient_texture->get_gradient();
	if (gradient.is_null()) {
		gradient.instantiate();
		PackedFloat32Array offsets = { 0, 1 };
		gradient->set_offsets(offsets);
		gradient->set_interpolation_color_space(Gradient::ColorSpace::GRADIENT_COLOR_SPACE_LINEAR_SRGB);
		gradient_texture->set_gradient(gradient);
	}

	PackedColorArray colors = { left_color, right_color };
	gradient->set_colors(colors);

	slider->draw_texture_rect(gradient_texture, Rect2(Vector2(), Vector2(size.x, margin)), false);
}

void ColorModeOKHSL::_value_changed() {
	Vector<float> values = color_picker->get_active_slider_values();

	if (values[1] > 0 || values[0] != cached_hue) {
		cached_hue = values[0];
	}
	if (values[2] > 0 || values[1] != cached_saturation) {
		cached_saturation = values[1];
	}

	// Cache real OKHSL values in ColorPicker.
	color_picker->ok_hsl_h = color_picker->sliders[0]->get_value() / 360.0;
	color_picker->ok_hsl_s = color_picker->sliders[1]->get_value() / 100.0;
	color_picker->ok_hsl_l = color_picker->sliders[2]->get_value() / 100.0;

	color_picker->okhsl_cached = true;
}

String ColorModeOKHSL::get_slider_label(int idx) const {
	ERR_FAIL_INDEX_V_MSG(idx, get_slider_count(), String(), "Couldn't get slider label.");
	return labels[idx];
}

float ColorModeOKHSL::get_slider_max(int idx) const {
	ERR_FAIL_INDEX_V_MSG(idx, get_slider_count(), 0, "Couldn't get slider max value.");
	return slider_max[idx];
}

float ColorModeOKHSL::get_slider_value(int idx) const {
	switch (idx) {
		case 0: {
			if (color_picker->color_normalized.get_ok_hsl_s() > 0) {
				return color_picker->color_normalized.get_ok_hsl_h() * 360.0;
			} else {
				return cached_hue;
			}
		}
		case 1: {
			if (color_picker->color_normalized.get_ok_hsl_l() > 0) {
				return color_picker->color_normalized.get_ok_hsl_s() * 100.0;
			} else {
				return cached_saturation;
			}
		}
		case 2:
			return color_picker->color_normalized.get_ok_hsl_l() * 100.0;
		default:
			ERR_FAIL_V_MSG(0, "Couldn't get slider value.");
	}
}

Color ColorModeOKHSL::get_color() const {
	Vector<float> values = color_picker->get_active_slider_values();
	Color color = Color::from_ok_hsl(values[0] / 360.0, values[1] / 100.0, values[2] / 100.0, values[3] / 255.0);
	return color;
}

void ColorModeOKHSL::slider_draw(int p_which) {
	HSlider *slider = color_picker->get_slider(p_which);
	Size2 size = slider->get_size();
	const real_t margin = 16 * color_picker->theme_cache.base_scale;

	Vector<Vector2> pos;
	Vector<Color> col;
	Color left_color;
	Color right_color;
	Color color = color_picker->color_normalized;
	float okhsl_l = color.get_ok_hsl_l();
	float slider_hue = (Math::is_zero_approx(color.get_ok_hsl_s())) ? cached_hue / 360.0 : color.get_ok_hsl_h();
	float slider_sat = (Math::is_zero_approx(okhsl_l) || Math::is_equal_approx(okhsl_l, 1)) ? cached_saturation / 100.0 : color.get_ok_hsl_s();

	if (p_which == 2) { // L
		pos.resize(6);
		col.resize(6);
		left_color = Color(0, 0, 0);
		Color middle_color = Color::from_ok_hsl(slider_hue, slider_sat, 0.5);
		right_color.set_ok_hsl(slider_hue, slider_sat, 1);

		col.set(0, left_color);
		col.set(1, middle_color);
		col.set(2, right_color);
		col.set(3, right_color);
		col.set(4, middle_color);
		col.set(5, left_color);
		pos.set(0, Vector2(0, 0));
		pos.set(1, Vector2(size.x * 0.5, 0));
		pos.set(2, Vector2(size.x, 0));
		pos.set(3, Vector2(size.x, margin));
		pos.set(4, Vector2(size.x * 0.5, margin));
		pos.set(5, Vector2(0, margin));
		slider->draw_polygon(pos, col);
	} else if (p_which == 1) { // S
		pos.resize(4);
		col.resize(4);

		left_color.set_ok_hsl(slider_hue, 0, okhsl_l);
		right_color.set_ok_hsl(slider_hue, 1, okhsl_l);

		col.set(0, left_color);
		col.set(1, right_color);
		col.set(2, right_color);
		col.set(3, left_color);
		pos.set(0, Vector2(0, 0));
		pos.set(1, Vector2(size.x, 0));
		pos.set(2, Vector2(size.x, margin));
		pos.set(3, Vector2(0, margin));
		slider->draw_polygon(pos, col);
	} else if (p_which == 0) { // H
		const int precision = 7;
		if (hue_texture.is_null()) {
			hue_texture.instantiate();
			hue_texture->set_width(400);
			hue_texture->set_height(6);
		}
		Ref<Gradient> hue_gradient = hue_texture->get_gradient();
		if (hue_gradient.is_null()) {
			hue_gradient.instantiate();
			PackedFloat32Array offsets;
			offsets.resize(precision);
			for (int i = 0; i < precision; i++) {
				float h = i / float(precision - 1);
				offsets.write[i] = h;
			}
			hue_gradient->set_offsets(offsets);
			hue_gradient->set_interpolation_color_space(Gradient::ColorSpace::GRADIENT_COLOR_SPACE_OKLAB);
			hue_texture->set_gradient(hue_gradient);
		}

		PackedColorArray colors;
		colors.resize(precision);

		for (int i = 0; i < precision; i++) {
			float h = i / float(precision - 1);
			colors.write[i] = Color::from_ok_hsl(h, slider_sat, okhsl_l);
		}
		hue_gradient->set_colors(colors);
		slider->draw_texture_rect(hue_texture, Rect2(Vector2(), Vector2(size.x, margin)), false);
	}
}
