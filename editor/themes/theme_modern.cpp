/**************************************************************************/
/*  theme_modern.cpp                                                      */
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

#include "theme_modern.h"

#include "core/math/math_defs.h"
#include "editor/editor_string_names.h"
#include "editor/settings/editor_settings.h"
#include "editor/themes/editor_scale.h"
#include "editor/themes/editor_theme_manager.h"
#include "scene/gui/graph_edit.h"
#include "scene/resources/dpi_texture.h"
#include "scene/resources/image_texture.h"
#include "scene/resources/style_box_flat.h"
#include "scene/resources/style_box_line.h"

// Helper.
static Color _get_base_color(EditorThemeManager::ThemeConfiguration &p_config, float p_dimness_ofs = 0.0, float p_saturation_mult = 1.0) {
	Color color = p_config.base_color;
	color.set_v(CLAMP(Math::lerp(color.get_v(), 0, p_config.contrast * p_dimness_ofs), 0, 1));
	color.set_s(color.get_s() * p_saturation_mult);
	return color;
}

void ThemeModern::populate_shared_styles(const Ref<EditorTheme> &p_theme, EditorThemeManager::ThemeConfiguration &p_config) {
	// Colors.
	{
		// Base colors.

		p_theme->set_color("base_color", EditorStringName(Editor), p_config.base_color);
		p_theme->set_color("accent_color", EditorStringName(Editor), p_config.accent_color);

		// White (dark theme) or black (light theme), will be used to generate the rest of the colors
		p_config.mono_color = p_config.dark_theme ? Color(1, 1, 1) : Color(0, 0, 0);
		p_config.mono_color_font = p_config.dark_icon_and_font ? Color(1, 1, 1) : Color(0, 0, 0);
		p_config.mono_color_inv = p_config.dark_theme ? Color(0, 0, 0) : Color(1, 1, 1);

		// Ensure base colors are in the 0..1 luminance range to avoid 8-bit integer overflow or text rendering issues.
		// Some places in the editor use 8-bit integer colors.
		p_config.dark_color_1 = p_config.base_color.lerp(Color(0, 0, 0, 1), p_config.contrast * 1.15).clamp();
		p_config.dark_color_2 = p_config.dark_theme ? Color(0, 0, 0, 0.3) : Color(1, 1, 1, 0.3);
		p_config.dark_color_3 = _get_base_color(p_config, 0.8, 0.9);

		p_config.contrast_color_1 = p_config.base_color.lerp(p_config.mono_color, MAX(p_config.contrast * 1.15, p_config.default_contrast * 1.15));
		p_config.contrast_color_2 = p_config.base_color.lerp(p_config.mono_color, MAX(p_config.contrast * 1.725, p_config.default_contrast * 1.725));

		p_config.highlight_color = Color(p_config.accent_color.r, p_config.accent_color.g, p_config.accent_color.b, 0.275);
		p_config.highlight_disabled_color = p_config.highlight_color.lerp(p_config.dark_theme ? Color(0, 0, 0) : Color(1, 1, 1), 0.5);

		p_config.success_color = Color(0.45, 0.95, 0.5);
		p_config.warning_color = Color(0.83, 0.78, 0.62);
		p_config.error_color = Color(1, 0.47, 0.42);

		// Keep dark theme colors accessible for use in the frame time gradient in the 3D editor.
		// This frame time gradient is used to colorize text for a dark background, so it should keep using bright colors
		// even when using a light theme.
		p_theme->set_color("success_color_dark_background", EditorStringName(Editor), p_config.success_color);
		p_theme->set_color("warning_color_dark_background", EditorStringName(Editor), p_config.warning_color);
		p_theme->set_color("error_color_dark_background", EditorStringName(Editor), p_config.error_color);

		if (!p_config.dark_icon_and_font) {
			// Darken some colors to be readable on a light background.
			p_config.success_color = p_config.success_color.lerp(p_config.mono_color_font, 0.35);
			p_config.warning_color = Color(0.83, 0.49, 0.01);
			p_config.error_color = Color(0.8, 0.22, 0.22);
		}

		p_theme->set_color("mono_color", EditorStringName(Editor), p_config.mono_color);
		p_theme->set_color("dark_color_1", EditorStringName(Editor), p_config.dark_color_1);
		p_theme->set_color("dark_color_2", EditorStringName(Editor), p_config.dark_color_2);
		p_theme->set_color("dark_color_3", EditorStringName(Editor), p_config.dark_color_3);
		p_theme->set_color("contrast_color_1", EditorStringName(Editor), p_config.contrast_color_1);
		p_theme->set_color("contrast_color_2", EditorStringName(Editor), p_config.contrast_color_2);
		p_theme->set_color("highlight_color", EditorStringName(Editor), p_config.highlight_color);
		p_theme->set_color("highlight_disabled_color", EditorStringName(Editor), p_config.highlight_disabled_color);
		p_theme->set_color("success_color", EditorStringName(Editor), p_config.success_color);
		p_theme->set_color("warning_color", EditorStringName(Editor), p_config.warning_color);
		p_theme->set_color("error_color", EditorStringName(Editor), p_config.error_color);
		p_theme->set_color("ruler_color", EditorStringName(Editor), p_config.base_color.lerp(p_config.mono_color_inv, 0.3) * Color(1, 1, 1, 0.8));
#ifndef DISABLE_DEPRECATED // Used before 4.3.
		p_theme->set_color("disabled_highlight_color", EditorStringName(Editor), p_config.highlight_disabled_color);
#endif

		// Only used when the Draw Extra Borders editor setting is enabled.
		p_config.extra_border_color_1 = p_config.dark_theme ? Color(1, 1, 1, 0.4) : Color(0, 0, 0, 0.4);
		p_config.extra_border_color_2 = p_config.dark_theme ? Color(1, 1, 1, 0.2) : Color(0, 0, 0, 0.2);

		p_theme->set_color("extra_border_color_1", EditorStringName(Editor), p_config.extra_border_color_1);
		p_theme->set_color("extra_border_color_2", EditorStringName(Editor), p_config.extra_border_color_2);

		// Font colors.

		p_config.font_color = p_config.mono_color_font * Color(1, 1, 1, 0.75);
		p_config.font_secondary_color = p_config.mono_color_font * Color(1, 1, 1, 0.55);
		p_config.font_focus_color = p_config.mono_color_font;
		p_config.font_hover_color = p_config.mono_color_font * Color(1, 1, 1, 0.85);
		p_config.font_pressed_color = p_config.mono_color_font * Color(1, 1, 1, 0.85);
		p_config.font_hover_pressed_color = p_config.mono_color_font;
		p_config.font_disabled_color = p_config.mono_color_font * Color(1, 1, 1, p_config.dark_icon_and_font ? 0.35 : 0.5);
		p_config.font_readonly_color = Color(p_config.mono_color_font.r, p_config.mono_color_font.g, p_config.mono_color_font.b, 0.65);
		p_config.font_placeholder_color = p_config.font_disabled_color;
		p_config.font_outline_color = Color(1, 1, 1, 0);

		// Colors designed for dark backgrounds, even when using a light theme.
		// This is used for 3D editor overlay texts.
		if (p_config.dark_theme) {
			p_config.font_dark_background_color = p_config.font_color;
			p_config.font_dark_background_focus_color = p_config.font_focus_color;
			p_config.font_dark_background_hover_color = p_config.font_hover_color;
			p_config.font_dark_background_pressed_color = p_config.font_pressed_color;
			p_config.font_dark_background_hover_pressed_color = p_config.font_hover_pressed_color;
		} else {
			p_config.font_dark_background_color = p_config.mono_color.inverted().lerp(p_config.base_color, 0.75);
			p_config.font_dark_background_focus_color = p_config.mono_color.inverted().lerp(p_config.base_color, 0.25);
			p_config.font_dark_background_hover_color = p_config.mono_color.inverted().lerp(p_config.base_color, 0.25);
			p_config.font_dark_background_pressed_color = p_config.font_dark_background_color.lerp(p_config.accent_color, 0.74);
			p_config.font_dark_background_hover_pressed_color = p_config.font_dark_background_color.lerp(p_config.accent_color, 0.5);
		}

		p_theme->set_color(SceneStringName(font_color), EditorStringName(Editor), p_config.font_color);
		p_theme->set_color("font_focus_color", EditorStringName(Editor), p_config.font_focus_color);
		p_theme->set_color("font_hover_color", EditorStringName(Editor), p_config.font_hover_color);
		p_theme->set_color("font_pressed_color", EditorStringName(Editor), p_config.font_pressed_color);
		p_theme->set_color("font_hover_pressed_color", EditorStringName(Editor), p_config.font_hover_pressed_color);
		p_theme->set_color("font_disabled_color", EditorStringName(Editor), p_config.font_disabled_color);
		p_theme->set_color("font_readonly_color", EditorStringName(Editor), p_config.font_readonly_color);
		p_theme->set_color("font_placeholder_color", EditorStringName(Editor), p_config.font_placeholder_color);
		p_theme->set_color("font_outline_color", EditorStringName(Editor), p_config.font_outline_color);

		p_theme->set_color("font_dark_background_color", EditorStringName(Editor), p_config.font_dark_background_color);
		p_theme->set_color("font_dark_background_focus_color", EditorStringName(Editor), p_config.font_dark_background_focus_color);
		p_theme->set_color("font_dark_background_hover_color", EditorStringName(Editor), p_config.font_dark_background_hover_color);
		p_theme->set_color("font_dark_background_pressed_color", EditorStringName(Editor), p_config.font_dark_background_pressed_color);
		p_theme->set_color("font_dark_background_hover_pressed_color", EditorStringName(Editor), p_config.font_dark_background_hover_pressed_color);

#ifndef DISABLE_DEPRECATED // Used before 4.3.
		p_theme->set_color("readonly_font_color", EditorStringName(Editor), p_config.font_readonly_color);
		p_theme->set_color("disabled_font_color", EditorStringName(Editor), p_config.font_disabled_color);
		p_theme->set_color("readonly_color", EditorStringName(Editor), p_config.font_readonly_color);
		p_theme->set_color("highlighted_font_color", EditorStringName(Editor), p_config.font_hover_color); // Closest equivalent.
#endif

		// Icon colors.

		p_config.icon_normal_color = Color(1, 1, 1, p_config.dark_icon_and_font ? 0.85 : 0.95);
		p_config.icon_secondary_color = Color(1, 1, 1, p_config.dark_icon_and_font ? 0.6 : 0.75);
		p_config.icon_focus_color = Color(1, 1, 1);
		p_config.icon_hover_color = Color(1, 1, 1);
		p_config.icon_pressed_color = p_config.accent_color * (p_config.dark_icon_and_font ? 1.15 : 3.5);
		p_config.icon_pressed_color.a = 1.0;
		p_config.icon_disabled_color = Color(1, 1, 1, p_config.dark_icon_and_font ? 0.35 : 0.5);

		p_theme->set_color("icon_normal_color", EditorStringName(Editor), p_config.icon_normal_color);
		p_theme->set_color("icon_focus_color", EditorStringName(Editor), p_config.icon_focus_color);
		p_theme->set_color("icon_hover_color", EditorStringName(Editor), p_config.icon_hover_color);
		p_theme->set_color("icon_pressed_color", EditorStringName(Editor), p_config.icon_pressed_color);
		p_theme->set_color("icon_disabled_color", EditorStringName(Editor), p_config.icon_disabled_color);

		// Additional GUI colors.

		p_config.surface_popup_color = _get_base_color(p_config, 1.9, 0.9);
		p_config.surface_lowest_color = _get_base_color(p_config, 1.7, 0.9);
		p_config.surface_lower_color = _get_base_color(p_config, 1.1, 0.9);
		p_config.surface_low_color = _get_base_color(p_config, 0.8);
		p_config.surface_base_color = _get_base_color(p_config);
		p_config.surface_high_color = _get_base_color(p_config, -1.3, 0.8);
		p_config.surface_higher_color = _get_base_color(p_config, -1.5, 0.8);
		p_config.surface_highest_color = _get_base_color(p_config, -2.2, 0.6);

		p_config.button_normal_color = _get_base_color(p_config, -2.0, 0.85);
		p_config.button_hover_color = _get_base_color(p_config, -2.9, 0.75);
		p_config.button_pressed_color = _get_base_color(p_config, -3.2, 0.75);
		p_config.button_disabled_color = _get_base_color(p_config, -1.5, 0.75);
		p_config.button_border_normal_color = _get_base_color(p_config, -2.5, 0.75);
		p_config.button_border_hover_color = _get_base_color(p_config, -3.4, 0.75);
		p_config.button_border_pressed_color = _get_base_color(p_config, -3.7, 0.75);

		p_config.shadow_color = Color(0, 0, 0, p_config.dark_theme ? 0.3 : 0.1);
		p_config.selection_color = p_config.accent_color * Color(1, 1, 1, 0.4);
		p_config.disabled_border_color = p_config.mono_color.inverted().lerp(p_config.base_color, 0.7);
		p_config.disabled_bg_color = p_config.mono_color.inverted().lerp(p_config.base_color, 0.9);
		p_config.separator_color = p_config.dark_theme ? Color(0, 0, 0, 0.4) : Color(0, 0, 0, 0.2);

		p_theme->set_color("selection_color", EditorStringName(Editor), p_config.selection_color);
		p_theme->set_color("disabled_border_color", EditorStringName(Editor), p_config.disabled_border_color);
		p_theme->set_color("disabled_bg_color", EditorStringName(Editor), p_config.disabled_bg_color);
		p_theme->set_color("separator_color", EditorStringName(Editor), p_config.separator_color);

		// Additional editor colors.

		p_theme->set_color("box_selection_fill_color", EditorStringName(Editor), p_config.mono_color * Color(1, 1, 1, 0.12));
		p_theme->set_color("box_selection_stroke_color", EditorStringName(Editor), p_config.mono_color * Color(1, 1, 1, 0.4));

		p_theme->set_color("axis_x_color", EditorStringName(Editor), Color(0.96, 0.20, 0.32));
		p_theme->set_color("axis_y_color", EditorStringName(Editor), Color(0.53, 0.84, 0.01));
		p_theme->set_color("axis_z_color", EditorStringName(Editor), Color(0.16, 0.55, 0.96));
		p_theme->set_color("axis_w_color", EditorStringName(Editor), Color(0.55, 0.55, 0.55));

		p_theme->set_color("property_color_x", EditorStringName(Editor), p_config.dark_icon_and_font ? Color(0.88, 0.38, 0.47) : Color(0.40, 0.04, 0.09));
		p_theme->set_color("property_color_y", EditorStringName(Editor), p_config.dark_icon_and_font ? Color(0.76, 0.93, 0.40) : Color(0.27, 0.37, 0.06));
		p_theme->set_color("property_color_z", EditorStringName(Editor), p_config.dark_icon_and_font ? Color(0.42, 0.67, 0.96) : Color(0.08, 0.22, 0.38));
		p_theme->set_color("property_color_w", EditorStringName(Editor), p_config.font_color);

		// Special colors for rendering methods.

		p_theme->set_color("forward_plus_color", EditorStringName(Editor), Color::hex(0x5d8c3fff));
		p_theme->set_color("mobile_color", EditorStringName(Editor), Color::hex(0xa5557dff));
		p_theme->set_color("gl_compatibility_color", EditorStringName(Editor), Color::hex(0x5586a4ff));
	}

	// Constants.
	{
		// Can't save single float in theme, so using Color.
		p_theme->set_color("icon_saturation", EditorStringName(Editor), Color(p_config.icon_saturation, p_config.icon_saturation, p_config.icon_saturation));

		// Controls may rely on the scale for their internal drawing logic.
		p_theme->set_default_base_scale(EDSCALE);
		p_theme->set_constant("scale", EditorStringName(Editor), EDSCALE);

		p_theme->set_constant("thumb_size", EditorStringName(Editor), p_config.thumb_size);
		p_theme->set_constant("class_icon_size", EditorStringName(Editor), p_config.class_icon_size);
		p_theme->set_constant("gizmo_handle_scale", EditorStringName(Editor), p_config.gizmo_handle_scale);

		p_theme->set_constant("base_margin", EditorStringName(Editor), p_config.base_margin);
		p_theme->set_constant("increased_margin", EditorStringName(Editor), p_config.increased_margin);
		p_theme->set_constant("window_border_margin", EditorStringName(Editor), p_config.window_border_margin);
		p_theme->set_constant("top_bar_separation", EditorStringName(Editor), p_config.top_bar_separation);

		p_theme->set_constant("dark_theme", EditorStringName(Editor), p_config.dark_theme);
	}

	// Styleboxes.
	{
		// This is the basic stylebox, used as a base for most other styleboxes (through `duplicate()`).
		p_config.base_style = EditorThemeManager::make_flat_stylebox(p_config.base_color, p_config.increased_margin * 1.5, p_config.increased_margin * 1.5, p_config.increased_margin * 1.5, p_config.increased_margin * 1.5, p_config.corner_radius);

		p_config.focus_style = p_config.base_style->duplicate();
		p_config.focus_style->set_draw_center(false);
		p_config.focus_style->set_border_color(p_config.accent_color * Color(1, 1, 1, 0.8));
		p_config.focus_style->set_border_width_all(2);

		p_config.base_empty_style = EditorThemeManager::make_empty_stylebox();

		p_config.base_empty_wide_style = EditorThemeManager::make_empty_stylebox();
		// Ensure minimum margin for wide flat buttons otherwise the topbar looks broken.
		float base_empty_wide_margin = MAX(p_config.base_margin, 3.0);
		p_config.base_empty_wide_style->set_content_margin_individual(base_empty_wide_margin * 1.5 * EDSCALE, base_empty_wide_margin * EDSCALE, base_empty_wide_margin * 1.5 * EDSCALE, base_empty_wide_margin * EDSCALE);

		// Button styles.
		{
			p_config.widget_margin = Vector2(p_config.increased_margin + 2, p_config.increased_margin + 1) * EDSCALE;

			p_config.button_style = p_config.base_style->duplicate();
			p_config.button_style->set_content_margin_individual(p_config.base_margin * 2 * EDSCALE, p_config.base_margin * 1.5 * EDSCALE, p_config.base_margin * 2 * EDSCALE, p_config.base_margin * 1.5 * EDSCALE);
			p_config.button_style->set_bg_color(p_config.button_normal_color);
			p_config.button_style->set_border_width_all(Math::round(EDSCALE));
			p_config.button_style->set_shadow_color(p_config.dark_theme ? Color(0, 0, 0, 0.005) : Color(1, 1, 1, 0.005));
			p_config.button_style->set_shadow_size(Math::ceil(8 * EDSCALE));
			p_config.button_style->set_shadow_offset(Vector2(0, 4) * EDSCALE);
			if (p_config.draw_extra_borders) {
				p_config.button_style->set_border_color(p_config.extra_border_color_1);
			} else {
				p_config.button_style->set_border_color(p_config.button_border_normal_color);
			}

			p_config.button_style_disabled = p_config.button_style->duplicate();
			p_config.button_style_disabled->set_bg_color(p_config.button_disabled_color);
			if (p_config.draw_extra_borders) {
				p_config.button_style_disabled->set_border_color(p_config.extra_border_color_2 * Color(1, 1, 1, 0.5));
			} else {
				p_config.button_style_disabled->set_border_width_all(0);
			}

			p_config.button_style_pressed = p_config.button_style->duplicate();
			p_config.button_style_pressed->set_bg_color(p_config.button_pressed_color);
			if (p_config.draw_extra_borders) {
				p_config.button_style_pressed->set_border_color(p_config.extra_border_color_1);
			} else {
				p_config.button_style_pressed->set_border_color(p_config.button_border_pressed_color);
			}

			p_config.button_style_hover = p_config.button_style->duplicate();
			p_config.button_style_hover->set_bg_color(p_config.button_hover_color);
			if (p_config.draw_extra_borders) {
				p_config.button_style_pressed->set_border_color(p_config.extra_border_color_1);
			} else {
				p_config.button_style_hover->set_border_color(p_config.button_border_hover_color);
			}

			p_config.flat_button_hover = p_config.base_style->duplicate();
			// Use the normal variation here and the hover one in the pressed state to lessen contrast.
			p_config.flat_button_hover->set_bg_color(p_config.button_normal_color);
			// This affects buttons in Tree so top and bottom margins should be kept low.
			p_config.flat_button_hover->set_content_margin_individual(p_config.base_margin * 1.5 * EDSCALE, p_config.base_margin * 0.9 * EDSCALE, p_config.base_margin * 1.5 * EDSCALE, p_config.base_margin * 0.9 * EDSCALE);
			if (p_config.draw_extra_borders) {
				p_config.button_style_hover->set_border_color(p_config.extra_border_color_1);
			}

			p_config.flat_button_pressed = p_config.flat_button_hover->duplicate();
			p_config.flat_button_pressed->set_bg_color(_get_base_color(p_config, -2.4, 0.75));
			if (p_config.draw_extra_borders) {
				p_config.flat_button_pressed->set_border_color(p_config.extra_border_color_1);
			}

			p_config.flat_button = p_config.flat_button_hover->duplicate();
			p_config.flat_button->set_draw_center(false);
		}

		// Windows and popups.
		{
			p_config.popup_panel_style = p_config.base_style->duplicate();
			p_config.popup_panel_style->set_bg_color(p_config.surface_popup_color);
			p_config.popup_panel_style->set_shadow_color(Color(0, 0, 0, 0.3));
			p_config.popup_panel_style->set_shadow_size(p_config.base_margin * 0.75 * EDSCALE);
			p_config.popup_panel_style->set_content_margin_all(p_config.popup_margin * EDSCALE);
			p_config.popup_panel_style->set_corner_radius_all(0);
			if (p_config.draw_extra_borders) {
				p_config.popup_panel_style->set_border_width_all(Math::round(EDSCALE));
				p_config.popup_panel_style->set_border_color(p_config.extra_border_color_2);
			}

			p_config.window_style = p_config.base_style->duplicate();
			p_config.window_style->set_content_margin_all(p_config.popup_margin);
			p_config.window_style->set_shadow_color(p_config.shadow_color);
			p_config.window_style->set_shadow_size(4 * EDSCALE);
			p_config.window_style->set_border_color(p_config.base_color);
			p_config.window_style->set_border_width(SIDE_TOP, 24 * EDSCALE);
			p_config.window_style->set_expand_margin(SIDE_TOP, 24 * EDSCALE);
			p_config.window_style->set_corner_radius_all(0);

			p_config.window_complex_style = p_config.window_style->duplicate();
			p_config.window_complex_style->set_bg_color(p_config.surface_lowest_color);

			p_config.dialog_style = p_config.base_style->duplicate();
			p_config.dialog_style->set_content_margin_all(p_config.popup_margin);
			p_config.dialog_style->set_corner_radius_all(0);
		}

		// Panels.
		{
			p_config.panel_container_style = p_config.button_style->duplicate();
			p_config.panel_container_style->set_draw_center(false);
			p_config.panel_container_style->set_border_width_all(0);

			// Content panel for tabs and similar containers.

			// Compensate for the border.
			const int content_panel_margin = p_config.base_margin * EDSCALE + p_config.border_width;

			p_config.content_panel_style = p_config.base_style->duplicate();
			p_config.content_panel_style->set_border_color(p_config.dark_color_3);
			p_config.content_panel_style->set_border_width_all(p_config.border_width);
			p_config.content_panel_style->set_border_width(Side::SIDE_TOP, 0);
			p_config.content_panel_style->set_corner_radius(CORNER_TOP_LEFT, 0);
			p_config.content_panel_style->set_corner_radius(CORNER_TOP_RIGHT, 0);
			p_config.content_panel_style->set_content_margin_individual(content_panel_margin, 2 * EDSCALE + content_panel_margin, content_panel_margin, content_panel_margin);

			p_config.tab_container_style = p_config.base_style->duplicate();
			p_config.tab_container_style->set_content_margin_all(p_config.increased_margin * 1.5 * EDSCALE);
			p_config.tab_container_style->set_corner_radius_individual(0, 0, p_config.corner_radius * EDSCALE, p_config.corner_radius * EDSCALE);

			p_config.foreground_panel = p_config.tab_container_style->duplicate();
			p_config.foreground_panel->set_corner_radius(CORNER_TOP_LEFT, p_config.tab_container_style->get_corner_radius(CORNER_BOTTOM_LEFT));
			p_config.foreground_panel->set_corner_radius(CORNER_TOP_RIGHT, p_config.tab_container_style->get_corner_radius(CORNER_BOTTOM_RIGHT));

			// Trees and similarly inset panels.

			p_config.tree_panel_style = p_config.base_style->duplicate();
			// Make Trees easier to distinguish from other controls by using a darker background color.
			p_config.tree_panel_style->set_bg_color(p_config.dark_color_1.lerp(p_config.dark_color_2, 0.5));
			if (p_config.draw_extra_borders) {
				p_config.tree_panel_style->set_border_width_all(Math::round(EDSCALE));
				p_config.tree_panel_style->set_border_color(p_config.extra_border_color_2);
			} else {
				p_config.tree_panel_style->set_border_color(p_config.dark_color_3);
			}
		}
	}
}

void ThemeModern::populate_standard_styles(const Ref<EditorTheme> &p_theme, EditorThemeManager::ThemeConfiguration &p_config) {
	// Panels.
	{
		// Panel.
		p_theme->set_stylebox(SceneStringName(panel), "Panel", EditorThemeManager::make_flat_stylebox(p_config.dark_color_1, 6, 4, 6, 4, p_config.corner_radius));

		// PanelContainer.
		p_theme->set_stylebox(SceneStringName(panel), "PanelContainer", p_config.base_empty_wide_style);

		// TooltipPanel & TooltipLabel.
		{
			// TooltipPanel is also used for custom tooltips, while TooltipLabel
			// is only relevant for default tooltips.

			p_theme->set_color(SceneStringName(font_color), "TooltipLabel", p_config.font_hover_color);
			p_theme->set_color("font_shadow_color", "TooltipLabel", Color(1, 1, 1, 0));

			Ref<StyleBoxFlat> tooltip_style = p_config.base_style->duplicate();
			tooltip_style->set_bg_color(p_config.surface_popup_color);
			tooltip_style->set_content_margin_all(0);
			tooltip_style->set_corner_radius_all(0);
			if (p_config.draw_extra_borders) {
				tooltip_style->set_border_width_all(Math::round(EDSCALE));
				tooltip_style->set_border_color(p_config.extra_border_color_2);
			}
			p_theme->set_stylebox(SceneStringName(panel), "TooltipPanel", tooltip_style);
		}

		// PopupPanel
		Ref<StyleBoxFlat> popup_panel_style = p_config.base_style->duplicate();
		popup_panel_style->set_bg_color(p_config.surface_popup_color);
		popup_panel_style->set_shadow_color(Color(0, 0, 0, 0.3));
		popup_panel_style->set_shadow_size(p_config.base_margin * 0.75 * EDSCALE);
		popup_panel_style->set_content_margin_all(p_config.popup_margin);
		popup_panel_style->set_corner_radius_all(0);
		if (p_config.draw_extra_borders) {
			popup_panel_style->set_border_width_all(Math::round(EDSCALE));
			popup_panel_style->set_border_color(p_config.extra_border_color_2);
		}
		p_theme->set_stylebox(SceneStringName(panel), "PopupPanel", p_config.popup_panel_style);
	}

	// Buttons.
	{
		// Button.

		p_theme->set_stylebox(CoreStringName(normal), "Button", p_config.button_style);
		p_theme->set_stylebox(SceneStringName(hover), "Button", p_config.button_style_hover);
		p_theme->set_stylebox(SceneStringName(pressed), "Button", p_config.button_style_pressed);
		p_theme->set_stylebox("hover_pressed", "Button", p_config.button_style_pressed);
		p_theme->set_stylebox("focus", "Button", p_config.focus_style);
		p_theme->set_stylebox("disabled", "Button", p_config.button_style_disabled);

		p_theme->set_color(SceneStringName(font_color), "Button", p_config.font_color);
		p_theme->set_color("font_hover_color", "Button", p_config.font_hover_color);
		p_theme->set_color("font_hover_pressed_color", "Button", p_config.font_hover_pressed_color);
		p_theme->set_color("font_focus_color", "Button", p_config.font_focus_color);
		p_theme->set_color("font_pressed_color", "Button", p_config.font_pressed_color);
		p_theme->set_color("font_disabled_color", "Button", p_config.font_disabled_color);
		p_theme->set_color("font_outline_color", "Button", p_config.font_outline_color);

		p_theme->set_color("icon_normal_color", "Button", p_config.icon_normal_color);
		p_theme->set_color("icon_hover_color", "Button", p_config.icon_hover_color);
		p_theme->set_color("icon_focus_color", "Button", p_config.icon_focus_color);
		p_theme->set_color("icon_hover_pressed_color", "Button", p_config.icon_pressed_color);
		p_theme->set_color("icon_pressed_color", "Button", p_config.icon_pressed_color);
		p_theme->set_color("icon_disabled_color", "Button", p_config.icon_disabled_color);

		p_theme->set_constant("h_separation", "Button", 4 * EDSCALE);
		p_theme->set_constant("outline_size", "Button", 0);

		p_theme->set_constant("outline_size", "Button", 0);
		p_theme->set_constant("align_to_largest_stylebox", "Button", 1); // Enabled.

		// MenuButton.

		p_theme->set_stylebox(CoreStringName(normal), "MenuButton", p_config.base_empty_wide_style);
		p_theme->set_stylebox(SceneStringName(hover), "MenuButton", p_config.flat_button_hover);
		p_theme->set_stylebox(SceneStringName(pressed), "MenuButton", p_config.flat_button_pressed);
		p_theme->set_stylebox("focus", "MenuButton", p_config.focus_style);
		p_theme->set_stylebox("disabled", "MenuButton", p_config.base_empty_wide_style);

		p_theme->set_constant("outline_size", "MenuButton", 0);

		// MenuBar.

		p_theme->set_stylebox(CoreStringName(normal), "MenuBar", p_config.button_style);
		p_theme->set_stylebox(SceneStringName(hover), "MenuBar", p_config.button_style_hover);
		p_theme->set_stylebox(SceneStringName(pressed), "MenuBar", p_config.button_style_pressed);
		p_theme->set_stylebox("disabled", "MenuBar", p_config.button_style_disabled);

		p_theme->set_color(SceneStringName(font_color), "MenuBar", p_config.font_color);
		p_theme->set_color("font_hover_color", "MenuBar", p_config.font_hover_color);
		p_theme->set_color("font_hover_pressed_color", "MenuBar", p_config.font_hover_pressed_color);
		p_theme->set_color("font_focus_color", "MenuBar", p_config.font_focus_color);
		p_theme->set_color("font_pressed_color", "MenuBar", p_config.font_pressed_color);
		p_theme->set_color("font_disabled_color", "MenuBar", p_config.font_disabled_color);
		p_theme->set_color("font_outline_color", "MenuBar", p_config.font_outline_color);

		p_theme->set_constant("h_separation", "MenuBar", 4 * EDSCALE);
		p_theme->set_constant("outline_size", "MenuBar", 0);

		// OptionButton.

		p_theme->set_stylebox("focus", "OptionButton", p_config.focus_style);
		p_theme->set_stylebox(CoreStringName(normal), "OptionButton", p_config.button_style);
		p_theme->set_stylebox(SceneStringName(hover), "OptionButton", p_config.button_style_hover);
		p_theme->set_stylebox(SceneStringName(pressed), "OptionButton", p_config.button_style_pressed);
		p_theme->set_stylebox("disabled", "OptionButton", p_config.button_style_disabled);

		p_theme->set_icon("arrow", "OptionButton", p_theme->get_icon(SNAME("GuiOptionArrow"), EditorStringName(EditorIcons)));
		p_theme->set_constant("arrow_margin", "OptionButton", p_config.base_margin * 2 * EDSCALE);
		p_theme->set_constant("modulate_arrow", "OptionButton", true);
		p_theme->set_constant("h_separation", "OptionButton", 4 * EDSCALE);
		p_theme->set_constant("outline_size", "OptionButton", 0);

		// CheckButton.

		p_theme->set_stylebox(CoreStringName(normal), "CheckButton", p_config.panel_container_style);
		p_theme->set_stylebox(SceneStringName(pressed), "CheckButton", p_config.panel_container_style);
		p_theme->set_stylebox("disabled", "CheckButton", p_config.panel_container_style);
		p_theme->set_stylebox(SceneStringName(hover), "CheckButton", p_config.panel_container_style);
		p_theme->set_stylebox("hover_pressed", "CheckButton", p_config.panel_container_style);

		p_theme->set_icon("checked", "CheckButton", p_theme->get_icon(SNAME("GuiToggleOn"), EditorStringName(EditorIcons)));
		p_theme->set_icon("checked_disabled", "CheckButton", p_theme->get_icon(SNAME("GuiToggleOnDisabled"), EditorStringName(EditorIcons)));
		p_theme->set_icon("unchecked", "CheckButton", p_theme->get_icon(SNAME("GuiToggleOff"), EditorStringName(EditorIcons)));
		p_theme->set_icon("unchecked_disabled", "CheckButton", p_theme->get_icon(SNAME("GuiToggleOffDisabled"), EditorStringName(EditorIcons)));

		p_theme->set_icon("checked_mirrored", "CheckButton", p_theme->get_icon(SNAME("GuiToggleOnMirrored"), EditorStringName(EditorIcons)));
		p_theme->set_icon("checked_disabled_mirrored", "CheckButton", p_theme->get_icon(SNAME("GuiToggleOnDisabledMirrored"), EditorStringName(EditorIcons)));
		p_theme->set_icon("unchecked_mirrored", "CheckButton", p_theme->get_icon(SNAME("GuiToggleOffMirrored"), EditorStringName(EditorIcons)));
		p_theme->set_icon("unchecked_disabled_mirrored", "CheckButton", p_theme->get_icon(SNAME("GuiToggleOffDisabledMirrored"), EditorStringName(EditorIcons)));

		p_theme->set_constant("h_separation", "CheckButton", 8 * EDSCALE);
		p_theme->set_constant("check_v_offset", "CheckButton", 0);
		p_theme->set_constant("outline_size", "CheckButton", 0);

		// CheckBox.
		{
			Ref<StyleBoxFlat> checkbox_style = p_config.panel_container_style->duplicate();
			checkbox_style->set_content_margin_individual(p_config.base_margin * 1.5 * EDSCALE, p_config.base_margin * 0.75 * EDSCALE, p_config.base_margin * 1.5 * EDSCALE, p_config.base_margin * 0.75 * EDSCALE);
			Ref<StyleBoxFlat> checkbox_style_normal = checkbox_style->duplicate();
			checkbox_style_normal->set_draw_center(false);

			p_theme->set_stylebox(CoreStringName(normal), "CheckBox", checkbox_style_normal);
			p_theme->set_stylebox(SceneStringName(pressed), "CheckBox", checkbox_style);
			p_theme->set_stylebox("disabled", "CheckBox", checkbox_style);
			p_theme->set_stylebox(SceneStringName(hover), "CheckBox", checkbox_style);
			p_theme->set_stylebox("hover_pressed", "CheckBox", checkbox_style);

			p_theme->set_icon("checked", "CheckBox", p_theme->get_icon(SNAME("GuiChecked"), EditorStringName(EditorIcons)));

			p_theme->set_icon("unchecked", "CheckBox", p_theme->get_icon(SNAME("GuiUnchecked"), EditorStringName(EditorIcons)));
			p_theme->set_icon("radio_checked", "CheckBox", p_theme->get_icon(SNAME("GuiRadioChecked"), EditorStringName(EditorIcons)));
			p_theme->set_icon("radio_unchecked", "CheckBox", p_theme->get_icon(SNAME("GuiRadioUnchecked"), EditorStringName(EditorIcons)));
			p_theme->set_icon("checked_disabled", "CheckBox", p_theme->get_icon(SNAME("GuiCheckedDisabled"), EditorStringName(EditorIcons)));
			p_theme->set_icon("unchecked_disabled", "CheckBox", p_theme->get_icon(SNAME("GuiUncheckedDisabled"), EditorStringName(EditorIcons)));
			p_theme->set_icon("radio_checked_disabled", "CheckBox", p_theme->get_icon(SNAME("GuiRadioCheckedDisabled"), EditorStringName(EditorIcons)));
			p_theme->set_icon("radio_unchecked_disabled", "CheckBox", p_theme->get_icon(SNAME("GuiRadioUncheckedDisabled"), EditorStringName(EditorIcons)));

			p_theme->set_constant("h_separation", "CheckBox", 8 * EDSCALE);
			p_theme->set_constant("check_v_offset", "CheckBox", 0);
			p_theme->set_constant("outline_size", "CheckBox", 0);
		}

		// LinkButton.

		p_theme->set_stylebox("focus", "LinkButton", p_config.base_empty_style);

		p_theme->set_color(SceneStringName(font_color), "LinkButton", p_config.font_color);
		p_theme->set_color("font_hover_color", "LinkButton", p_config.font_hover_color);
		p_theme->set_color("font_hover_pressed_color", "LinkButton", p_config.font_hover_pressed_color);
		p_theme->set_color("font_focus_color", "LinkButton", p_config.font_focus_color);
		p_theme->set_color("font_pressed_color", "LinkButton", p_config.font_pressed_color);
		p_theme->set_color("font_disabled_color", "LinkButton", p_config.font_disabled_color);
		p_theme->set_color("font_outline_color", "LinkButton", p_config.font_outline_color);

		p_theme->set_constant("outline_size", "LinkButton", 0);
	}

	// Tree & ItemList.
	{
		// Tree.
		{
			// Use empty stylebox for trees to avoid drawing unnecessary borders in docks.
			Ref<StyleBoxEmpty> style_tree_panel = p_config.base_empty_style->duplicate();
			style_tree_panel->set_content_margin_individual(p_config.base_margin * 1.5 * EDSCALE, p_config.base_margin * 2.5 * EDSCALE, p_config.base_margin * 1.5 * EDSCALE, p_config.base_margin * 2.5 * EDSCALE);

			Ref<StyleBoxFlat> style_button_pressed = p_config.flat_button_pressed->duplicate();
			style_button_pressed->set_content_margin_individual(p_config.base_margin, 0, p_config.base_margin, 0);

			p_theme->set_icon("checked", "Tree", p_theme->get_icon(SNAME("GuiChecked"), EditorStringName(EditorIcons)));
			p_theme->set_icon("checked_disabled", "Tree", p_theme->get_icon(SNAME("GuiCheckedDisabled"), EditorStringName(EditorIcons)));
			p_theme->set_icon("indeterminate", "Tree", p_theme->get_icon(SNAME("GuiIndeterminate"), EditorStringName(EditorIcons)));
			p_theme->set_icon("indeterminate_disabled", "Tree", p_theme->get_icon(SNAME("GuiIndeterminateDisabled"), EditorStringName(EditorIcons)));
			p_theme->set_icon("unchecked", "Tree", p_theme->get_icon(SNAME("GuiUnchecked"), EditorStringName(EditorIcons)));
			p_theme->set_icon("unchecked_disabled", "Tree", p_theme->get_icon(SNAME("GuiUncheckedDisabled"), EditorStringName(EditorIcons)));
			p_theme->set_icon("arrow", "Tree", p_theme->get_icon(SNAME("GuiTreeArrowDown"), EditorStringName(EditorIcons)));
			p_theme->set_icon("arrow_collapsed", "Tree", p_theme->get_icon(SNAME("GuiTreeArrowRight"), EditorStringName(EditorIcons)));
			p_theme->set_icon("arrow_collapsed_mirrored", "Tree", p_theme->get_icon(SNAME("GuiTreeArrowLeft"), EditorStringName(EditorIcons)));
			p_theme->set_icon("updown", "Tree", p_theme->get_icon(SNAME("GuiTreeUpdown"), EditorStringName(EditorIcons)));
			p_theme->set_icon("select_arrow", "Tree", p_theme->get_icon(SNAME("GuiDropdown"), EditorStringName(EditorIcons)));

			p_theme->set_stylebox(SceneStringName(panel), "Tree", style_tree_panel);
			p_theme->set_stylebox("focus", "Tree", p_config.focus_style);
			p_theme->set_stylebox("button_pressed", "Tree", style_button_pressed);
			p_theme->set_stylebox("custom_button", "Tree", p_config.flat_button);
			p_theme->set_stylebox("custom_button_pressed", "Tree", style_button_pressed);

			p_theme->set_color("custom_button_font_highlight", "Tree", p_config.font_hover_color);
			p_theme->set_color(SceneStringName(font_color), "Tree", p_config.font_color);
			p_theme->set_color("font_hovered_color", "Tree", p_config.font_hover_color);
			p_theme->set_color("font_hovered_dimmed_color", "Tree", p_config.font_hover_color);
			p_theme->set_color("font_hovered_selected_color", "Tree", p_config.mono_color_font);
			p_theme->set_color("font_selected_color", "Tree", p_config.mono_color_font);
			p_theme->set_color("font_disabled_color", "Tree", p_config.font_disabled_color);
			p_theme->set_color("font_outline_color", "Tree", p_config.font_outline_color);
			p_theme->set_color("title_button_color", "Tree", p_config.font_color);
			p_theme->set_color("drop_position_color", "Tree", p_config.accent_color);

			int tree_v_sep = p_config.enable_touch_optimizations ? p_config.separation_margin : Math::pow(p_config.base_margin * 0.2 * EDSCALE, 3);
			p_theme->set_constant("v_separation", "Tree", tree_v_sep);
			p_theme->set_constant("h_separation", "Tree", (p_config.increased_margin + 2) * EDSCALE);
			p_theme->set_constant("guide_width", "Tree", p_config.border_width);
			p_theme->set_constant("item_margin", "Tree", MAX(3 * p_config.increased_margin * EDSCALE, 12 * EDSCALE));
			p_theme->set_constant("inner_item_margin_top", "Tree", p_config.separation_margin);
			p_theme->set_constant("inner_item_margin_bottom", "Tree", p_config.separation_margin);
			p_theme->set_constant("inner_item_margin_left", "Tree", p_config.base_margin * EDSCALE);
			p_theme->set_constant("inner_item_margin_right", "Tree", p_config.base_margin * EDSCALE);
			p_theme->set_constant("button_margin", "Tree", p_config.base_margin * EDSCALE);
			p_theme->set_constant("dragging_unfold_wait_msec", "Tree", p_config.dragging_hover_wait_msec);
			p_theme->set_constant("scroll_border", "Tree", 40 * EDSCALE);
			p_theme->set_constant("scroll_speed", "Tree", 12);
			p_theme->set_constant("outline_size", "Tree", 0);
			p_theme->set_constant("scrollbar_margin_left", "Tree", 0);
			p_theme->set_constant("scrollbar_margin_top", "Tree", 0);
			p_theme->set_constant("scrollbar_margin_right", "Tree", 0);
			p_theme->set_constant("scrollbar_margin_bottom", "Tree", 0);
			p_theme->set_constant("scrollbar_h_separation", "Tree", 1 * EDSCALE);
			p_theme->set_constant("scrollbar_v_separation", "Tree", 1 * EDSCALE);

			Color relationship_line_color = p_config.mono_color * Color(1, 1, 1, p_config.relationship_line_opacity);
			Color highlight_line_color = p_config.mono_color * Color(1, 1, 1, p_config.relationship_line_opacity * 2);

			int draw_relationship_lines = 0;
			int relationship_line_width = 0;
			int highlighted_line_width = Math::ceil(EDSCALE);
			if (p_config.draw_relationship_lines == EditorThemeManager::RELATIONSHIP_ALL) {
				draw_relationship_lines = 1;
				relationship_line_width = 1;
			} else if (p_config.draw_relationship_lines == EditorThemeManager::RELATIONSHIP_SELECTED_ONLY) {
				draw_relationship_lines = 1;
			}

			p_theme->set_constant("draw_guides", "Tree", 0);
			p_theme->set_constant("draw_relationship_lines", "Tree", draw_relationship_lines && p_config.relationship_line_opacity >= 0.01);
			p_theme->set_constant("relationship_line_width", "Tree", relationship_line_width);
			p_theme->set_constant("parent_hl_line_width", "Tree", highlighted_line_width);
			p_theme->set_constant("children_hl_line_width", "Tree", 1);
			p_theme->set_constant("parent_hl_line_margin", "Tree", 3);
			p_theme->set_color("relationship_line_color", "Tree", relationship_line_color);
			p_theme->set_color("parent_hl_line_color", "Tree", highlight_line_color);
			p_theme->set_color("children_hl_line_color", "Tree", relationship_line_color);
			p_theme->set_color("drop_position_color", "Tree", p_config.icon_normal_color);
			p_theme->set_color("guide_color", "Tree", Color(1, 1, 1, 0));

			Ref<StyleBoxFlat> style_tree_hover = p_config.flat_button_hover->duplicate();
			style_tree_hover->set_bg_color(p_config.button_disabled_color);
			style_tree_hover->set_content_margin_all(0);

			p_theme->set_stylebox("button_hover", "Tree", style_tree_hover);
			p_theme->set_stylebox("hovered", "Tree", style_tree_hover);
			p_theme->set_stylebox("hovered_dimmed", "Tree", style_tree_hover);
			p_theme->set_stylebox("custom_button_hover", "Tree", style_tree_hover);
			p_theme->set_stylebox("selected", "Tree", style_tree_hover);
			p_theme->set_stylebox("selected_focus", "Tree", p_config.focus_style);

			Ref<StyleBoxFlat> style_tree_hover_selected = style_tree_hover->duplicate();
			style_tree_hover_selected->set_bg_color(p_config.button_normal_color);

			p_theme->set_stylebox("hovered_selected", "Tree", style_tree_hover_selected);
			p_theme->set_stylebox("hovered_selected_focus", "Tree", p_config.focus_style);

			// Cursor is drawn on top of the item so it needs to be transparent.
			Ref<StyleBoxFlat> style_tree_cursor = p_config.base_style->duplicate();
			style_tree_cursor->set_bg_color(p_config.mono_color * Color(1, 1, 1, 0.04));

			Ref<StyleBoxFlat> style_tree_title = p_config.base_style->duplicate();
			style_tree_title->set_bg_color(p_config.surface_lower_color);
			// Use a transparent border to separate rounded column titles.
			style_tree_title->set_border_color(Color(p_config.surface_lower_color, 0));
			style_tree_title->set_border_width(SIDE_LEFT, Math::ceil(EDSCALE));
			style_tree_title->set_border_width(SIDE_RIGHT, Math::ceil(EDSCALE));

			p_theme->set_stylebox("cursor", "Tree", style_tree_cursor);
			p_theme->set_stylebox("cursor_unfocused", "Tree", style_tree_cursor);
			p_theme->set_stylebox("title_button_normal", "Tree", style_tree_title);
			p_theme->set_stylebox("title_button_hover", "Tree", style_tree_title);
			p_theme->set_stylebox("title_button_pressed", "Tree", style_tree_title);
		}

		// ProjectList.
		{
			Ref<StyleBoxFlat> style_project_list_hover = p_config.flat_button_hover->duplicate();
			style_project_list_hover->set_bg_color(_get_base_color(p_config, -0.5, 0.75));
			style_project_list_hover->set_content_margin_all(0);

			Ref<StyleBoxFlat> style_project_list_selected = style_project_list_hover->duplicate();
			style_project_list_selected->set_bg_color(_get_base_color(p_config, -1.0, 0.75));

			Ref<StyleBoxFlat> style_project_list_hover_pressed = style_project_list_selected->duplicate();
			style_project_list_hover_pressed->set_bg_color(_get_base_color(p_config, -1.2, 0.75));

			p_theme->set_stylebox("hovered", "ProjectList", style_project_list_hover);
			p_theme->set_stylebox("selected", "ProjectList", style_project_list_selected);
			p_theme->set_stylebox("hover_pressed", "ProjectList", style_project_list_hover_pressed);
			p_theme->set_stylebox("focus", "ProjectList", p_config.focus_style);

			p_theme->set_color(SceneStringName(font_color), "ProjectList", p_config.font_color);
			p_theme->set_color("guide_color", "ProjectList", Color(1, 1, 1, 0));
		}

		// ItemList.
		{
			Ref<StyleBoxFlat> style_itemlist_bg = p_config.base_style->duplicate();
			style_itemlist_bg->set_content_margin_all(p_config.base_margin * 2 * EDSCALE);
			Ref<StyleBoxFlat> style_itemlist_cursor = p_config.base_style->duplicate();
			style_itemlist_cursor->set_bg_color(p_config.mono_color * Color(1, 1, 1, 0.04));

			p_theme->set_stylebox(SceneStringName(panel), "ItemList", style_itemlist_bg);
			p_theme->set_stylebox("focus", "ItemList", p_config.focus_style);
			p_theme->set_stylebox("cursor", "ItemList", style_itemlist_cursor);
			p_theme->set_stylebox("cursor_unfocused", "ItemList", style_itemlist_cursor);
			p_theme->set_stylebox("selected_focus", "ItemList", p_config.focus_style);
			p_theme->set_stylebox("selected", "ItemList", p_config.flat_button_hover);
			p_theme->set_stylebox("hovered", "ItemList", p_config.flat_button_hover);
			p_theme->set_stylebox("hovered_selected", "ItemList", p_config.flat_button_hover);
			p_theme->set_stylebox("hovered_selected_focus", "ItemList", p_config.focus_style);
			p_theme->set_color(SceneStringName(font_color), "ItemList", p_config.font_color);
			p_theme->set_color("font_hovered_color", "ItemList", p_config.font_hover_color);
			p_theme->set_color("font_hovered_selected_color", "ItemList", p_config.font_hover_pressed_color);
			p_theme->set_color("font_selected_color", "ItemList", p_config.font_pressed_color);
			p_theme->set_color("font_outline_color", "ItemList", p_config.font_outline_color);
			p_theme->set_color("guide_color", "ItemList", Color(1, 1, 1, 0));
			p_theme->set_constant("v_separation", "ItemList", p_config.base_margin * 1.5 * EDSCALE);
			p_theme->set_constant("h_separation", "ItemList", (p_config.increased_margin + 2) * EDSCALE);
			p_theme->set_constant("icon_margin", "ItemList", (p_config.increased_margin + 2) * EDSCALE);
			p_theme->set_constant(SceneStringName(line_separation), "ItemList", p_config.separation_margin);
			p_theme->set_constant("outline_size", "ItemList", 0);
		}
	}

	// TabBar & TabContainer.
	{
		Ref<StyleBoxFlat> style_tab_selected = p_config.base_style->duplicate();
		style_tab_selected->set_content_margin_individual(p_config.base_margin * 4 * EDSCALE, p_config.base_margin * 2.1 * EDSCALE, p_config.base_margin * 4 * EDSCALE, p_config.base_margin * 2.1 * EDSCALE);
		style_tab_selected->set_corner_radius_individual(p_config.corner_radius * EDSCALE, p_config.corner_radius * EDSCALE, 0, 0);

		Ref<StyleBoxFlat> style_tab_focus = style_tab_selected->duplicate();
		style_tab_focus->set_bg_color(p_config.base_color);
		style_tab_focus->set_border_color(p_config.accent_color);

		Ref<StyleBoxFlat> style_tab_unselected = style_tab_selected->duplicate();
		style_tab_unselected->set_bg_color(p_config.surface_lowest_color);
		style_tab_unselected->set_border_width_all(0);

		Ref<StyleBoxFlat> style_tab_hovered = style_tab_unselected->duplicate();
		style_tab_hovered->set_bg_color(p_config.surface_base_color * Color(1, 1, 1, 0.6));

		Color drop_mark_color = p_config.dark_color_2.lerp(p_config.accent_color, 0.75);

		Ref<StyleBoxFlat> style_tabbar_background = p_config.base_style->duplicate();
		style_tabbar_background->set_bg_color(p_config.surface_lowest_color);
		style_tabbar_background->set_corner_radius(CORNER_BOTTOM_LEFT, 0);
		style_tabbar_background->set_corner_radius(CORNER_BOTTOM_RIGHT, 0);
		style_tabbar_background->set_content_margin_individual(0, 0, p_config.base_margin * 0.25 * EDSCALE, 0);

		p_theme->set_stylebox("tabbar_background", "TabContainer", style_tabbar_background);
		p_theme->set_stylebox(SceneStringName(panel), "TabContainer", p_config.tab_container_style);

		p_theme->set_stylebox("tab_selected", "TabContainer", style_tab_selected);
		p_theme->set_stylebox("tab_hovered", "TabContainer", style_tab_hovered);
		p_theme->set_stylebox("tab_unselected", "TabContainer", style_tab_unselected);
		p_theme->set_stylebox("tab_disabled", "TabContainer", style_tab_unselected);
		p_theme->set_stylebox("tab_focus", "TabContainer", p_config.focus_style);
		p_theme->set_stylebox("tab_selected", "TabBar", style_tab_selected);
		p_theme->set_stylebox("tab_hovered", "TabBar", style_tab_hovered);
		p_theme->set_stylebox("tab_unselected", "TabBar", style_tab_unselected);
		p_theme->set_stylebox("tab_disabled", "TabBar", style_tab_unselected);
		p_theme->set_stylebox("tab_focus", "TabBar", p_config.focus_style);
		p_theme->set_stylebox("button_pressed", "TabBar", p_config.panel_container_style);
		p_theme->set_stylebox("button_highlight", "TabBar", p_config.panel_container_style);

		p_theme->set_color("font_selected_color", "TabContainer", p_config.font_color);
		p_theme->set_color("font_hovered_color", "TabContainer", p_config.font_hover_color);
		p_theme->set_color("font_unselected_color", "TabContainer", p_config.font_secondary_color);
		p_theme->set_color("font_disabled_color", "TabContainer", p_config.font_disabled_color * Color(1, 1, 1, 0.55));
		p_theme->set_color("font_outline_color", "TabContainer", p_config.font_outline_color);
		p_theme->set_color("font_selected_color", "TabBar", p_config.font_color);
		p_theme->set_color("font_hovered_color", "TabBar", p_config.font_hover_color);
		p_theme->set_color("font_unselected_color", "TabBar", p_config.font_secondary_color);
		p_theme->set_color("font_disabled_color", "TabBar", p_config.font_disabled_color * Color(1, 1, 1, 0.55));
		p_theme->set_color("font_outline_color", "TabBar", p_config.font_outline_color);
		p_theme->set_color("drop_mark_color", "TabContainer", drop_mark_color);
		p_theme->set_color("drop_mark_color", "TabBar", drop_mark_color);

		p_theme->set_color("icon_selected_color", "TabContainer", p_config.icon_normal_color);
		p_theme->set_color("icon_hovered_color", "TabContainer", p_config.icon_hover_color);
		p_theme->set_color("icon_unselected_color", "TabContainer", p_config.icon_secondary_color);
		p_theme->set_color("icon_disabled_color", "TabContainer", p_config.icon_disabled_color * Color(1, 1, 1, 0.55));
		p_theme->set_color("icon_selected_color", "TabBar", p_config.icon_normal_color);
		p_theme->set_color("icon_hovered_color", "TabBar", p_config.icon_hover_color);
		p_theme->set_color("icon_unselected_color", "TabBar", p_config.icon_secondary_color);
		p_theme->set_color("icon_disabled_color", "TabBar", p_config.icon_disabled_color * Color(1, 1, 1, 0.55));

		p_theme->set_icon("menu", "TabContainer", p_theme->get_icon(SNAME("GuiTabMenu"), EditorStringName(EditorIcons)));
		p_theme->set_icon("menu_highlight", "TabContainer", p_theme->get_icon(SNAME("GuiTabMenuHl"), EditorStringName(EditorIcons)));
		p_theme->set_icon("close", "TabBar", p_theme->get_icon(SNAME("GuiClose"), EditorStringName(EditorIcons)));
		p_theme->set_icon("increment", "TabContainer", p_theme->get_icon(SNAME("GuiScrollArrowRight"), EditorStringName(EditorIcons)));
		p_theme->set_icon("decrement", "TabContainer", p_theme->get_icon(SNAME("GuiScrollArrowLeft"), EditorStringName(EditorIcons)));
		p_theme->set_icon("increment", "TabBar", p_theme->get_icon(SNAME("GuiScrollArrowRight"), EditorStringName(EditorIcons)));
		p_theme->set_icon("decrement", "TabBar", p_theme->get_icon(SNAME("GuiScrollArrowLeft"), EditorStringName(EditorIcons)));
		p_theme->set_icon("increment_highlight", "TabBar", p_theme->get_icon(SNAME("GuiScrollArrowRightHl"), EditorStringName(EditorIcons)));
		p_theme->set_icon("decrement_highlight", "TabBar", p_theme->get_icon(SNAME("GuiScrollArrowLeftHl"), EditorStringName(EditorIcons)));
		p_theme->set_icon("increment_highlight", "TabContainer", p_theme->get_icon(SNAME("GuiScrollArrowRightHl"), EditorStringName(EditorIcons)));
		p_theme->set_icon("decrement_highlight", "TabContainer", p_theme->get_icon(SNAME("GuiScrollArrowLeftHl"), EditorStringName(EditorIcons)));
		p_theme->set_icon("drop_mark", "TabContainer", p_theme->get_icon(SNAME("GuiTabDropMark"), EditorStringName(EditorIcons)));
		p_theme->set_icon("drop_mark", "TabBar", p_theme->get_icon(SNAME("GuiTabDropMark"), EditorStringName(EditorIcons)));

		p_theme->set_constant("side_margin", "TabContainer", 0);
		p_theme->set_constant("outline_size", "TabContainer", 0);
		p_theme->set_constant("h_separation", "TabBar", 4 * EDSCALE);
		p_theme->set_constant("outline_size", "TabBar", 0);
		p_theme->set_constant("hover_switch_wait_msec", "TabBar", p_config.dragging_hover_wait_msec);
	}

	// Separators.
	{
		Ref<StyleBoxLine> style_h_separator = EditorThemeManager::make_line_stylebox(p_config.separator_color, Math::round(2 * EDSCALE), p_config.base_margin * -1 * EDSCALE, p_config.base_margin * -1 * EDSCALE);
		p_theme->set_stylebox("separator", "HSeparator", style_h_separator);

		Ref<StyleBoxLine> style_v_separator = style_h_separator->duplicate();
		style_v_separator->set_vertical(true);
		p_theme->set_stylebox("separator", "VSeparator", style_v_separator);

		p_theme->set_constant("separation", "Separator", p_config.base_margin * 2 * EDSCALE);
	}

	// LineEdit & TextEdit.
	{
		Ref<StyleBoxFlat> text_editor_style = p_config.base_style->duplicate();
		text_editor_style->set_bg_color(p_config.surface_lower_color);
		text_editor_style->set_content_margin_individual(p_config.base_margin * 2 * EDSCALE, ((p_config.base_margin * 0.75) + 1) * EDSCALE, p_config.base_margin * 2 * EDSCALE, ((p_config.base_margin * 0.75) + 1) * EDSCALE);
		if (p_config.draw_extra_borders) {
			text_editor_style->set_border_width_all(Math::round(EDSCALE));
			text_editor_style->set_border_color(p_config.extra_border_color_1);
		}

		Ref<StyleBoxFlat> text_editor_disabled_style = text_editor_style->duplicate();
		// Using transparent background for readonly otherwise it looks bad in the master audio bus.
		text_editor_disabled_style->set_bg_color(p_config.dark_theme ? Color(0, 0, 0, 0.2) : Color(1, 1, 1, 0.5));

		// LineEdit.

		p_theme->set_stylebox(CoreStringName(normal), "LineEdit", text_editor_style);
		p_theme->set_stylebox("focus", "LineEdit", p_config.focus_style);
		p_theme->set_stylebox("read_only", "LineEdit", text_editor_disabled_style);

		p_theme->set_icon("clear", "LineEdit", p_theme->get_icon(SNAME("GuiClose"), EditorStringName(EditorIcons)));

		p_theme->set_color(SceneStringName(font_color), "LineEdit", p_config.font_color);
		p_theme->set_color("font_selected_color", "LineEdit", p_config.font_pressed_color);
		p_theme->set_color("font_uneditable_color", "LineEdit", p_config.font_readonly_color);
		p_theme->set_color("font_placeholder_color", "LineEdit", p_config.font_placeholder_color);
		p_theme->set_color("font_outline_color", "LineEdit", p_config.font_outline_color);
		p_theme->set_color("caret_color", "LineEdit", p_config.font_color);
		p_theme->set_color("selection_color", "LineEdit", p_config.selection_color);
		p_theme->set_color("clear_button_color", "LineEdit", p_config.font_color);
		p_theme->set_color("clear_button_color_pressed", "LineEdit", p_config.accent_color);

		p_theme->set_constant("minimum_character_width", "LineEdit", 4);
		p_theme->set_constant("outline_size", "LineEdit", 0);
		p_theme->set_constant("caret_width", "LineEdit", 1);

		// TextEdit.

		p_theme->set_stylebox(CoreStringName(normal), "TextEdit", text_editor_style);
		p_theme->set_stylebox("focus", "TextEdit", p_config.focus_style);
		p_theme->set_stylebox("read_only", "TextEdit", text_editor_disabled_style);

		p_theme->set_icon("tab", "TextEdit", p_theme->get_icon(SNAME("GuiTab"), EditorStringName(EditorIcons)));
		p_theme->set_icon("space", "TextEdit", p_theme->get_icon(SNAME("GuiSpace"), EditorStringName(EditorIcons)));

		p_theme->set_color(SceneStringName(font_color), "TextEdit", p_config.font_color);
		p_theme->set_color("font_readonly_color", "TextEdit", p_config.font_readonly_color);
		p_theme->set_color("font_placeholder_color", "TextEdit", p_config.font_placeholder_color);
		p_theme->set_color("font_outline_color", "TextEdit", p_config.font_outline_color);
		p_theme->set_color("caret_color", "TextEdit", p_config.font_color);
		p_theme->set_color("selection_color", "TextEdit", p_config.selection_color);

		p_theme->set_constant("line_spacing", "TextEdit", 4 * EDSCALE);
		p_theme->set_constant("outline_size", "TextEdit", 0);
		p_theme->set_constant("caret_width", "TextEdit", 1);
	}

	// Containers.
	{
		p_theme->set_constant("separation", "BoxContainer", p_config.separation_margin);
		p_theme->set_constant("separation", "HBoxContainer", p_config.separation_margin);
		p_theme->set_constant("separation", "VBoxContainer", p_config.separation_margin);
		p_theme->set_constant("margin_left", "MarginContainer", 0);
		p_theme->set_constant("margin_top", "MarginContainer", 0);
		p_theme->set_constant("margin_right", "MarginContainer", 0);
		p_theme->set_constant("margin_bottom", "MarginContainer", 0);
		p_theme->set_constant("h_separation", "GridContainer", p_config.separation_margin);
		p_theme->set_constant("v_separation", "GridContainer", p_config.separation_margin);
		p_theme->set_constant("h_separation", "FlowContainer", p_config.separation_margin);
		p_theme->set_constant("v_separation", "FlowContainer", p_config.separation_margin);
		p_theme->set_constant("h_separation", "HFlowContainer", p_config.separation_margin);
		p_theme->set_constant("v_separation", "HFlowContainer", p_config.separation_margin);
		p_theme->set_constant("h_separation", "VFlowContainer", p_config.separation_margin);
		p_theme->set_constant("v_separation", "VFlowContainer", p_config.separation_margin);

		// SplitContainer.

		p_theme->set_icon("h_grabber", "SplitContainer", p_theme->get_icon(SNAME("GuiHsplitter"), EditorStringName(EditorIcons)));
		p_theme->set_icon("v_grabber", "SplitContainer", p_theme->get_icon(SNAME("GuiVsplitter"), EditorStringName(EditorIcons)));
		p_theme->set_icon("grabber", "VSplitContainer", p_theme->get_icon(SNAME("GuiVsplitter"), EditorStringName(EditorIcons)));
		p_theme->set_icon("grabber", "HSplitContainer", p_theme->get_icon(SNAME("GuiHsplitter"), EditorStringName(EditorIcons)));

		p_theme->set_constant("separation", "SplitContainer", p_config.base_margin * 0.75 * EDSCALE);
		p_theme->set_constant("separation", "HSplitContainer", p_config.base_margin * 0.75 * EDSCALE);
		p_theme->set_constant("separation", "VSplitContainer", p_config.base_margin * 0.75 * EDSCALE);

		p_theme->set_constant("autohide", "SplitContainer", 1);
		p_theme->set_constant("autohide", "HSplitContainer", 1);
		p_theme->set_constant("autohide", "VSplitContainer", 1);

		p_theme->set_constant("minimum_grab_thickness", "SplitContainer", p_config.base_margin * 2 * EDSCALE);
		p_theme->set_constant("minimum_grab_thickness", "HSplitContainer", p_config.base_margin * 2 * EDSCALE);
		p_theme->set_constant("minimum_grab_thickness", "VSplitContainer", p_config.base_margin * 2 * EDSCALE);

		// GridContainer.
		p_theme->set_constant("v_separation", "GridContainer", Math::round(p_config.widget_margin.y - 2 * EDSCALE));

		// FoldableContainer

		Ref<StyleBoxFlat> foldable_container_title = EditorThemeManager::make_flat_stylebox(p_config.dark_color_1.darkened(0.125), p_config.base_margin, p_config.base_margin, p_config.base_margin, p_config.base_margin);
		foldable_container_title->set_corner_radius(CORNER_BOTTOM_LEFT, 0);
		foldable_container_title->set_corner_radius(CORNER_BOTTOM_RIGHT, 0);
		p_theme->set_stylebox("title_panel", "FoldableContainer", foldable_container_title);
		Ref<StyleBoxFlat> foldable_container_hover = EditorThemeManager::make_flat_stylebox(p_config.dark_color_1.lerp(p_config.base_color, 0.4), p_config.base_margin, p_config.base_margin, p_config.base_margin, p_config.base_margin);
		foldable_container_hover->set_corner_radius(CORNER_BOTTOM_LEFT, 0);
		foldable_container_hover->set_corner_radius(CORNER_BOTTOM_RIGHT, 0);
		p_theme->set_stylebox("title_hover_panel", "FoldableContainer", foldable_container_hover);
		p_theme->set_stylebox("title_collapsed_panel", "FoldableContainer", EditorThemeManager::make_flat_stylebox(p_config.dark_color_1.darkened(0.125), p_config.base_margin, p_config.base_margin, p_config.base_margin, p_config.base_margin));
		p_theme->set_stylebox("title_collapsed_hover_panel", "FoldableContainer", EditorThemeManager::make_flat_stylebox(p_config.dark_color_1.lerp(p_config.base_color, 0.4), p_config.base_margin, p_config.base_margin, p_config.base_margin, p_config.base_margin));
		Ref<StyleBoxFlat> foldable_container_panel = EditorThemeManager::make_flat_stylebox(p_config.dark_color_1, p_config.base_margin, p_config.base_margin, p_config.base_margin, p_config.base_margin);
		foldable_container_panel->set_corner_radius(CORNER_TOP_LEFT, 0);
		foldable_container_panel->set_corner_radius(CORNER_TOP_RIGHT, 0);
		p_theme->set_stylebox(SceneStringName(panel), "FoldableContainer", foldable_container_panel);
		p_theme->set_stylebox("focus", "FoldableContainer", p_config.focus_style);

		p_theme->set_font(SceneStringName(font), "FoldableContainer", p_theme->get_font(SceneStringName(font), SNAME("HeaderSmall")));
		p_theme->set_font_size(SceneStringName(font_size), "FoldableContainer", p_theme->get_font_size(SceneStringName(font_size), SNAME("HeaderSmall")));

		p_theme->set_color(SceneStringName(font_color), "FoldableContainer", p_config.font_color);
		p_theme->set_color("hover_font_color", "FoldableContainer", p_config.font_hover_color);
		p_theme->set_color("collapsed_font_color", "FoldableContainer", p_config.font_pressed_color);
		p_theme->set_color("font_outline_color", "FoldableContainer", p_config.font_outline_color);

		p_theme->set_icon("expanded_arrow", "FoldableContainer", p_theme->get_icon(SNAME("GuiTreeArrowDown"), EditorStringName(EditorIcons)));
		p_theme->set_icon("expanded_arrow_mirrored", "FoldableContainer", p_theme->get_icon(SNAME("GuiArrowUp"), EditorStringName(EditorIcons)));
		p_theme->set_icon("folded_arrow", "FoldableContainer", p_theme->get_icon(SNAME("GuiTreeArrowRight"), EditorStringName(EditorIcons)));
		p_theme->set_icon("folded_arrow_mirrored", "FoldableContainer", p_theme->get_icon(SNAME("GuiTreeArrowLeft"), EditorStringName(EditorIcons)));

		p_theme->set_constant("outline_size", "FoldableContainer", 0);
		p_theme->set_constant("h_separation", "FoldableContainer", p_config.separation_margin);
	}

	// Window and dialogs.
	{
		// Window.

		p_theme->set_stylebox("embedded_border", "Window", p_config.window_style);
		p_theme->set_stylebox("embedded_unfocused_border", "Window", p_config.window_style);

		p_theme->set_color("title_color", "Window", p_config.font_color);
		p_theme->set_icon("close", "Window", p_theme->get_icon(SNAME("GuiClose"), EditorStringName(EditorIcons)));
		p_theme->set_icon("close_pressed", "Window", p_theme->get_icon(SNAME("GuiClose"), EditorStringName(EditorIcons)));
		p_theme->set_constant("close_h_offset", "Window", 22 * EDSCALE);
		p_theme->set_constant("close_v_offset", "Window", 20 * EDSCALE);
		p_theme->set_constant("title_height", "Window", 24 * EDSCALE);
		p_theme->set_constant("resize_margin", "Window", 4 * EDSCALE);
		p_theme->set_font("title_font", "Window", p_theme->get_font(SNAME("title"), EditorStringName(EditorFonts)));
		p_theme->set_font_size("title_font_size", "Window", p_theme->get_font_size(SNAME("title_size"), EditorStringName(EditorFonts)));

		// AcceptDialog.
		p_theme->set_stylebox(SceneStringName(panel), "AcceptDialog", p_config.dialog_style);
		p_theme->set_constant("buttons_separation", "AcceptDialog", 8 * EDSCALE);
		// Make buttons with short texts such as "OK" easier to click/tap.
		p_theme->set_constant("buttons_min_width", "AcceptDialog", p_config.dialogs_buttons_min_size.x * EDSCALE);
		p_theme->set_constant("buttons_min_height", "AcceptDialog", p_config.dialogs_buttons_min_size.y * EDSCALE);

		// FileDialog.
		p_theme->set_icon("folder", "FileDialog", p_theme->get_icon("Folder", EditorStringName(EditorIcons)));
		p_theme->set_icon("parent_folder", "FileDialog", p_theme->get_icon("ArrowUp", EditorStringName(EditorIcons)));
		p_theme->set_icon("back_folder", "FileDialog", p_theme->get_icon("Back", EditorStringName(EditorIcons)));
		p_theme->set_icon("forward_folder", "FileDialog", p_theme->get_icon("Forward", EditorStringName(EditorIcons)));
		p_theme->set_icon("reload", "FileDialog", p_theme->get_icon("Reload", EditorStringName(EditorIcons)));
		p_theme->set_icon("toggle_hidden", "FileDialog", p_theme->get_icon("GuiVisibilityVisible", EditorStringName(EditorIcons)));
		p_theme->set_icon("toggle_filename_filter", "FileDialog", p_theme->get_icon("FilenameFilter", EditorStringName(EditorIcons)));
		p_theme->set_icon("thumbnail_mode", "FileDialog", p_theme->get_icon("FileThumbnail", EditorStringName(EditorIcons)));
		p_theme->set_icon("list_mode", "FileDialog", p_theme->get_icon("FileList", EditorStringName(EditorIcons)));
		p_theme->set_icon("sort", "FileDialog", p_theme->get_icon("Sort", EditorStringName(EditorIcons)));
		p_theme->set_icon("favorite", "FileDialog", p_theme->get_icon("Favorites", EditorStringName(EditorIcons)));
		p_theme->set_icon("favorite_up", "FileDialog", p_theme->get_icon("MoveUp", EditorStringName(EditorIcons)));
		p_theme->set_icon("favorite_down", "FileDialog", p_theme->get_icon("MoveDown", EditorStringName(EditorIcons)));
		p_theme->set_icon("create_folder", "FileDialog", p_theme->get_icon("FolderCreate", EditorStringName(EditorIcons)));
		// Use a different color for folder icons to make them easier to distinguish from files.
		// On a light theme, the icon will be dark, so we need to lighten it before blending it with the accent color.
		p_theme->set_color("folder_icon_color", "FileDialog", (p_config.dark_icon_and_font ? Color(1, 1, 1) : Color(4.25, 4.25, 4.25)).lerp(p_config.accent_color, 0.7));
		p_theme->set_color("file_disabled_color", "FileDialog", p_config.font_disabled_color);

		p_theme->set_constant("thumbnail_size", "EditorFileDialog", p_config.thumb_size);

		// PopupDialog.
		p_theme->set_stylebox(SceneStringName(panel), "PopupDialog", p_config.dialog_style);

		// PopupMenu.
		{
			Ref<StyleBoxFlat> style_popup_menu = p_config.base_style->duplicate();
			style_popup_menu->set_bg_color(p_config.surface_popup_color);
			style_popup_menu->set_content_margin_all(p_config.popup_margin);
			style_popup_menu->set_corner_radius_all(0);
			if (p_config.draw_extra_borders) {
				style_popup_menu->set_border_width_all(Math::round(EDSCALE));
				style_popup_menu->set_border_color(p_config.extra_border_color_2);
			}
			p_theme->set_stylebox(SceneStringName(panel), "PopupMenu", style_popup_menu);

			Ref<StyleBoxFlat> style_popup_hover = p_config.flat_button_hover->duplicate();
			style_popup_hover->set_bg_color(_get_base_color(p_config, -0.5, 0.75));

			p_theme->set_stylebox(SceneStringName(hover), "PopupMenu", style_popup_hover);

			Ref<StyleBoxLine> style_popup_separator = EditorThemeManager::make_line_stylebox(p_config.mono_color * Color(1, 1, 1, p_config.dark_theme ? 0.075 : 0.125), Math::round(2 * EDSCALE), p_config.base_margin * -2 * EDSCALE, p_config.base_margin * -2 * EDSCALE);

			p_theme->set_stylebox("separator", "PopupMenu", style_popup_separator);
			p_theme->set_stylebox("labeled_separator_left", "PopupMenu", style_popup_separator);
			p_theme->set_stylebox("labeled_separator_right", "PopupMenu", style_popup_separator);

			p_theme->set_color(SceneStringName(font_color), "PopupMenu", p_config.font_color);
			p_theme->set_color("font_hover_color", "PopupMenu", p_config.font_hover_color);
			p_theme->set_color("font_accelerator_color", "PopupMenu", p_config.font_disabled_color);
			p_theme->set_color("font_disabled_color", "PopupMenu", p_config.font_disabled_color);
			p_theme->set_color("font_separator_color", "PopupMenu", p_config.font_disabled_color);
			p_theme->set_color("font_outline_color", "PopupMenu", p_config.font_outline_color);

			p_theme->set_icon("checked", "PopupMenu", p_theme->get_icon(SNAME("GuiChecked"), EditorStringName(EditorIcons)));
			p_theme->set_icon("unchecked", "PopupMenu", p_theme->get_icon(SNAME("GuiUnchecked"), EditorStringName(EditorIcons)));
			p_theme->set_icon("radio_checked", "PopupMenu", p_theme->get_icon(SNAME("GuiRadioChecked"), EditorStringName(EditorIcons)));
			p_theme->set_icon("radio_unchecked", "PopupMenu", p_theme->get_icon(SNAME("GuiRadioUnchecked"), EditorStringName(EditorIcons)));
			p_theme->set_icon("checked_disabled", "PopupMenu", p_theme->get_icon(SNAME("GuiCheckedDisabled"), EditorStringName(EditorIcons)));
			p_theme->set_icon("unchecked_disabled", "PopupMenu", p_theme->get_icon(SNAME("GuiUncheckedDisabled"), EditorStringName(EditorIcons)));
			p_theme->set_icon("radio_checked_disabled", "PopupMenu", p_theme->get_icon(SNAME("GuiRadioCheckedDisabled"), EditorStringName(EditorIcons)));
			p_theme->set_icon("radio_unchecked_disabled", "PopupMenu", p_theme->get_icon(SNAME("GuiRadioUncheckedDisabled"), EditorStringName(EditorIcons)));
			p_theme->set_icon("submenu", "PopupMenu", p_theme->get_icon(SNAME("ArrowRight"), EditorStringName(EditorIcons)));
			p_theme->set_icon("submenu_mirrored", "PopupMenu", p_theme->get_icon(SNAME("ArrowLeft"), EditorStringName(EditorIcons)));

			p_theme->set_constant("h_separation", "PopupMenu", p_config.base_margin * 1.75 * EDSCALE);
			int v_sep = (p_config.enable_touch_optimizations ? 12 : p_config.base_margin * 1.75) * EDSCALE;
			p_theme->set_constant("v_separation", "PopupMenu", v_sep);
			p_theme->set_constant("outline_size", "PopupMenu", 0);
			p_theme->set_constant("item_start_padding", "PopupMenu", p_config.popup_margin);
			p_theme->set_constant("item_end_padding", "PopupMenu", p_config.popup_margin);
		}
	}

	// Sliders and scrollbars.
	{
		Ref<Texture2D> empty_icon = memnew(ImageTexture);

		Ref<StyleBoxFlat> grabber_style = p_config.base_style->duplicate();
		grabber_style->set_bg_color(p_config.mono_color * Color(1, 1, 1, 0.225));

		Ref<StyleBoxFlat> grabber_hl_style = p_config.base_style->duplicate();
		grabber_hl_style->set_bg_color(p_config.mono_color * Color(1, 1, 1, 0.5));

		int scroll_margin = (p_config.enable_touch_optimizations ? 10 : 3) * EDSCALE;

		// HScrollBar.

		Ref<StyleBoxEmpty> h_scroll_style = p_config.base_empty_style->duplicate();
		h_scroll_style->set_content_margin_individual(0, scroll_margin, 0, scroll_margin);

		p_theme->set_stylebox("scroll", "HScrollBar", h_scroll_style);
		p_theme->set_stylebox("scroll_focus", "HScrollBar", p_config.focus_style);
		p_theme->set_stylebox("grabber", "HScrollBar", grabber_style);
		p_theme->set_stylebox("grabber_highlight", "HScrollBar", grabber_hl_style);
		p_theme->set_stylebox("grabber_pressed", "HScrollBar", grabber_hl_style);

		p_theme->set_icon("increment", "HScrollBar", empty_icon);
		p_theme->set_icon("increment_highlight", "HScrollBar", empty_icon);
		p_theme->set_icon("increment_pressed", "HScrollBar", empty_icon);
		p_theme->set_icon("decrement", "HScrollBar", empty_icon);
		p_theme->set_icon("decrement_highlight", "HScrollBar", empty_icon);
		p_theme->set_icon("decrement_pressed", "HScrollBar", empty_icon);

		p_theme->set_constant("padding_top", "HScrollBar", p_config.base_margin * EDSCALE);
		p_theme->set_constant("padding_bottom", "HScrollBar", p_config.base_margin * EDSCALE);

		// VScrollBar.

		Ref<StyleBoxEmpty> v_scroll_style = p_config.base_empty_style->duplicate();
		v_scroll_style->set_content_margin_individual(scroll_margin, 0, scroll_margin, 0);

		p_theme->set_stylebox("scroll", "VScrollBar", v_scroll_style);
		p_theme->set_stylebox("scroll_focus", "VScrollBar", p_config.focus_style);
		p_theme->set_stylebox("grabber", "VScrollBar", grabber_style);
		p_theme->set_stylebox("grabber_highlight", "VScrollBar", grabber_hl_style);
		p_theme->set_stylebox("grabber_pressed", "VScrollBar", grabber_hl_style);

		p_theme->set_icon("increment", "VScrollBar", empty_icon);
		p_theme->set_icon("increment_highlight", "VScrollBar", empty_icon);
		p_theme->set_icon("increment_pressed", "VScrollBar", empty_icon);
		p_theme->set_icon("decrement", "VScrollBar", empty_icon);
		p_theme->set_icon("decrement_highlight", "VScrollBar", empty_icon);
		p_theme->set_icon("decrement_pressed", "VScrollBar", empty_icon);

		p_theme->set_constant("padding_left", "VScrollBar", p_config.base_margin * EDSCALE);
		p_theme->set_constant("padding_right", "VScrollBar", p_config.base_margin * EDSCALE);

		// Slider
		const int background_margin = MAX(2, p_config.base_margin / 2);

		Ref<StyleBoxFlat> style_h_slider = p_config.base_style->duplicate();
		style_h_slider->set_bg_color(p_config.mono_color_inv * Color(1, 1, 1, 0.35));
		style_h_slider->set_content_margin_individual(0, 2 * EDSCALE, 0, 2 * EDSCALE);

		// HSlider.
		p_theme->set_icon("grabber_highlight", "HSlider", p_theme->get_icon(SNAME("GuiSliderGrabberHl"), EditorStringName(EditorIcons)));
		p_theme->set_icon("grabber", "HSlider", p_theme->get_icon(SNAME("GuiSliderGrabber"), EditorStringName(EditorIcons)));
		p_theme->set_stylebox("slider", "HSlider", style_h_slider);
		p_theme->set_stylebox("grabber_area", "HSlider", EditorThemeManager::make_flat_stylebox(p_config.contrast_color_1, 0, background_margin, 0, background_margin, p_config.corner_radius));
		p_theme->set_stylebox("grabber_area_highlight", "HSlider", EditorThemeManager::make_flat_stylebox(p_config.contrast_color_1, 0, background_margin, 0, background_margin));
		p_theme->set_constant("center_grabber", "HSlider", 0);
		p_theme->set_constant("grabber_offset", "HSlider", 0);

		Ref<StyleBoxFlat> style_v_slider = style_h_slider->duplicate();
		style_v_slider->set_content_margin_individual(2 * EDSCALE, 0, 2 * EDSCALE, 0);

		// VSlider.
		p_theme->set_icon("grabber", "VSlider", p_theme->get_icon(SNAME("GuiSliderGrabber"), EditorStringName(EditorIcons)));
		p_theme->set_icon("grabber_highlight", "VSlider", p_theme->get_icon(SNAME("GuiSliderGrabberHl"), EditorStringName(EditorIcons)));
		p_theme->set_stylebox("slider", "VSlider", style_v_slider);
		p_theme->set_stylebox("grabber_area", "VSlider", EditorThemeManager::make_flat_stylebox(p_config.contrast_color_1, background_margin, 0, background_margin, 0, p_config.corner_radius));
		p_theme->set_stylebox("grabber_area_highlight", "VSlider", EditorThemeManager::make_flat_stylebox(p_config.contrast_color_1, background_margin, 0, background_margin, 0));
		p_theme->set_constant("center_grabber", "VSlider", 0);
		p_theme->set_constant("grabber_offset", "VSlider", 0);
	}

	// Labels.
	{
		// RichTextLabel.

		Ref<StyleBoxFlat> rich_text_style = p_config.base_style->duplicate();
		rich_text_style->set_bg_color(p_config.surface_low_color);
		rich_text_style->set_content_margin_all(p_config.base_margin * 2 * EDSCALE);

		p_theme->set_stylebox(CoreStringName(normal), "RichTextLabel", rich_text_style);
		p_theme->set_stylebox("focus", "RichTextLabel", EditorThemeManager::make_empty_stylebox());

		p_theme->set_color("default_color", "RichTextLabel", p_config.font_color);
		p_theme->set_color("font_shadow_color", "RichTextLabel", Color(1, 1, 1, 0));
		p_theme->set_color("font_outline_color", "RichTextLabel", p_config.font_outline_color);
		p_theme->set_color("selection_color", "RichTextLabel", p_config.selection_color);

		p_theme->set_constant("shadow_offset_x", "RichTextLabel", 1 * EDSCALE);
		p_theme->set_constant("shadow_offset_y", "RichTextLabel", 1 * EDSCALE);
		p_theme->set_constant("shadow_outline_size", "RichTextLabel", 1 * EDSCALE);
		p_theme->set_constant("outline_size", "RichTextLabel", 0);

		// Label.

		Ref<StyleBoxEmpty> label_style = p_config.base_empty_style->duplicate();
		label_style->set_content_margin_individual(p_config.base_margin * 2 * EDSCALE, p_config.base_margin * EDSCALE, p_config.base_margin * 2 * EDSCALE, p_config.base_margin * EDSCALE);

		p_theme->set_stylebox(CoreStringName(normal), "Label", label_style);
		p_theme->set_stylebox("focus", "Label", p_config.focus_style);

		p_theme->set_color(SceneStringName(font_color), "Label", p_config.font_color);
		p_theme->set_color("font_shadow_color", "Label", Color(1, 1, 1, 0));
		p_theme->set_color("font_outline_color", "Label", p_config.font_outline_color);

		p_theme->set_constant("shadow_offset_x", "Label", 1 * EDSCALE);
		p_theme->set_constant("shadow_offset_y", "Label", 1 * EDSCALE);
		p_theme->set_constant("shadow_outline_size", "Label", 1 * EDSCALE);
		p_theme->set_constant("line_spacing", "Label", 3 * EDSCALE);
		p_theme->set_constant("outline_size", "Label", 0);
	}

	// SpinBox.
	{
		Ref<Texture2D> empty_icon = memnew(ImageTexture);
		p_theme->set_icon("updown", "SpinBox", empty_icon);
		p_theme->set_icon("up", "SpinBox", p_theme->get_icon(SNAME("GuiSpinboxUp"), EditorStringName(EditorIcons)));
		p_theme->set_icon("up_hover", "SpinBox", p_theme->get_icon(SNAME("GuiSpinboxUp"), EditorStringName(EditorIcons)));
		p_theme->set_icon("up_pressed", "SpinBox", p_theme->get_icon(SNAME("GuiSpinboxUp"), EditorStringName(EditorIcons)));
		p_theme->set_icon("up_disabled", "SpinBox", p_theme->get_icon(SNAME("GuiSpinboxUp"), EditorStringName(EditorIcons)));
		p_theme->set_icon("down", "SpinBox", p_theme->get_icon(SNAME("GuiSpinboxDown"), EditorStringName(EditorIcons)));
		p_theme->set_icon("down_hover", "SpinBox", p_theme->get_icon(SNAME("GuiSpinboxDown"), EditorStringName(EditorIcons)));
		p_theme->set_icon("down_pressed", "SpinBox", p_theme->get_icon(SNAME("GuiSpinboxDown"), EditorStringName(EditorIcons)));
		p_theme->set_icon("down_disabled", "SpinBox", p_theme->get_icon(SNAME("GuiSpinboxDown"), EditorStringName(EditorIcons)));

		p_theme->set_stylebox("up_background", "SpinBox", EditorThemeManager::make_empty_stylebox());
		p_theme->set_stylebox("up_background_hovered", "SpinBox", p_config.button_style_hover);
		p_theme->set_stylebox("up_background_pressed", "SpinBox", p_config.button_style_pressed);
		p_theme->set_stylebox("up_background_disabled", "SpinBox", EditorThemeManager::make_empty_stylebox());
		p_theme->set_stylebox("down_background", "SpinBox", EditorThemeManager::make_empty_stylebox());
		p_theme->set_stylebox("down_background_hovered", "SpinBox", p_config.button_style_hover);
		p_theme->set_stylebox("down_background_pressed", "SpinBox", p_config.button_style_pressed);
		p_theme->set_stylebox("down_background_disabled", "SpinBox", EditorThemeManager::make_empty_stylebox());

		p_theme->set_color("up_icon_modulate", "SpinBox", p_config.icon_normal_color);
		p_theme->set_color("up_hover_icon_modulate", "SpinBox", p_config.icon_hover_color);
		p_theme->set_color("up_pressed_icon_modulate", "SpinBox", p_config.icon_pressed_color);
		p_theme->set_color("up_disabled_icon_modulate", "SpinBox", p_config.icon_disabled_color);
		p_theme->set_color("down_icon_modulate", "SpinBox", p_config.icon_normal_color);
		p_theme->set_color("down_hover_icon_modulate", "SpinBox", p_config.icon_hover_color);
		p_theme->set_color("down_pressed_icon_modulate", "SpinBox", p_config.icon_pressed_color);
		p_theme->set_color("down_disabled_icon_modulate", "SpinBox", p_config.icon_disabled_color);

		p_theme->set_stylebox("field_and_buttons_separator", "SpinBox", EditorThemeManager::make_empty_stylebox());
		p_theme->set_stylebox("up_down_buttons_separator", "SpinBox", EditorThemeManager::make_empty_stylebox());

		p_theme->set_constant("buttons_vertical_separation", "SpinBox", 0);
		p_theme->set_constant("field_and_buttons_separation", "SpinBox", 2);
		p_theme->set_constant("buttons_width", "SpinBox", 16);
#ifndef DISABLE_DEPRECATED
		p_theme->set_constant("set_min_buttons_width_from_icons", "SpinBox", 1);
#endif
	}

	// ProgressBar.

	Ref<StyleBoxFlat> progress_bar_style = p_config.base_style->duplicate();
	progress_bar_style->set_bg_color(p_config.surface_lowest_color);
	progress_bar_style->set_expand_margin(SIDE_TOP, p_config.base_margin * 0.5 * EDSCALE);
	progress_bar_style->set_expand_margin(SIDE_BOTTOM, p_config.base_margin * 0.5 * EDSCALE);
	progress_bar_style->set_content_margin_all(p_config.base_margin * EDSCALE);
	if (p_config.draw_extra_borders) {
		progress_bar_style->set_border_width_all(Math::round(EDSCALE));
		progress_bar_style->set_border_color(p_config.extra_border_color_2);
	}

	Ref<StyleBoxFlat> progress_fill_style = progress_bar_style->duplicate();
	progress_fill_style->set_bg_color(p_config.button_normal_color);
	if (p_config.draw_extra_borders) {
		progress_fill_style->set_border_color(p_config.extra_border_color_1);
	}

	p_theme->set_stylebox("background", "ProgressBar", progress_bar_style);
	p_theme->set_stylebox("fill", "ProgressBar", progress_fill_style);
	p_theme->set_color(SceneStringName(font_color), "ProgressBar", p_config.font_color);
	p_theme->set_color("font_outline_color", "ProgressBar", p_config.font_outline_color);
	p_theme->set_constant("outline_size", "ProgressBar", 0);

	// PopupProgressBar

	p_theme->set_type_variation("PopupProgressBar", "ProgressBar");

	Ref<StyleBoxFlat> popup_progress_bar_style = progress_bar_style->duplicate();
	popup_progress_bar_style->set_bg_color(_get_base_color(p_config, 0.4, 0.9));

	Ref<StyleBoxFlat> popup_progress_fill_style = progress_fill_style->duplicate();
	popup_progress_fill_style->set_bg_color(_get_base_color(p_config, -1.6, 0.9));
	if (p_config.draw_extra_borders) {
		popup_progress_fill_style->set_border_color(p_config.extra_border_color_1);
	}

	p_theme->set_stylebox("background", "PopupProgressBar", popup_progress_bar_style);
	p_theme->set_stylebox("fill", "PopupProgressBar", popup_progress_fill_style);

	// GraphEdit and related nodes.
	{
		// GraphEdit.

		Ref<StyleBoxFlat> ge_panel_style = p_config.base_style->duplicate();
		ge_panel_style->set_bg_color(p_config.surface_lowest_color);

		p_theme->set_stylebox(SceneStringName(panel), "GraphEdit", ge_panel_style);
		p_theme->set_stylebox("panel_focus", "GraphEdit", p_config.focus_style);
		p_theme->set_stylebox("menu_panel", "GraphEdit", EditorThemeManager::make_flat_stylebox(p_config.surface_low_color * Color(1, 1, 1, 0.5), 4, 2, 4, 2, 3));

		float grid_base_brightness = p_config.dark_theme ? 1.0 : 0.0;
		GraphEdit::GridPattern grid_pattern = (GraphEdit::GridPattern) int(EDITOR_GET("editors/visual_editors/grid_pattern"));
		switch (grid_pattern) {
			case GraphEdit::GRID_PATTERN_LINES:
				p_theme->set_color("grid_major", "GraphEdit", Color(grid_base_brightness, grid_base_brightness, grid_base_brightness, 0.10));
				p_theme->set_color("grid_minor", "GraphEdit", Color(grid_base_brightness, grid_base_brightness, grid_base_brightness, 0.05));
				break;
			case GraphEdit::GRID_PATTERN_DOTS:
				p_theme->set_color("grid_major", "GraphEdit", Color(grid_base_brightness, grid_base_brightness, grid_base_brightness, 0.07));
				p_theme->set_color("grid_minor", "GraphEdit", Color(grid_base_brightness, grid_base_brightness, grid_base_brightness, 0.07));
				break;
			default:
				WARN_PRINT("Unknown grid pattern.");
				break;
		}

		p_theme->set_color("selection_fill", "GraphEdit", p_theme->get_color(SNAME("box_selection_fill_color"), EditorStringName(Editor)));
		p_theme->set_color("selection_stroke", "GraphEdit", p_theme->get_color(SNAME("box_selection_stroke_color"), EditorStringName(Editor)));
		p_theme->set_color("activity", "GraphEdit", p_config.mono_color);

		p_theme->set_color("connection_hover_tint_color", "GraphEdit", p_config.dark_color_2);
		p_theme->set_constant("connection_hover_thickness", "GraphEdit", 0);
		p_theme->set_color("connection_valid_target_tint_color", "GraphEdit", p_config.extra_border_color_1);
		p_theme->set_color("connection_rim_color", "GraphEdit", p_config.tree_panel_style->get_bg_color());

		p_theme->set_icon("zoom_out", "GraphEdit", p_theme->get_icon(SNAME("ZoomLess"), EditorStringName(EditorIcons)));
		p_theme->set_icon("zoom_in", "GraphEdit", p_theme->get_icon(SNAME("ZoomMore"), EditorStringName(EditorIcons)));
		p_theme->set_icon("zoom_reset", "GraphEdit", p_theme->get_icon(SNAME("ZoomReset"), EditorStringName(EditorIcons)));
		p_theme->set_icon("grid_toggle", "GraphEdit", p_theme->get_icon(SNAME("GridToggle"), EditorStringName(EditorIcons)));
		p_theme->set_icon("minimap_toggle", "GraphEdit", p_theme->get_icon(SNAME("GridMinimap"), EditorStringName(EditorIcons)));
		p_theme->set_icon("snapping_toggle", "GraphEdit", p_theme->get_icon(SNAME("SnapGrid"), EditorStringName(EditorIcons)));
		p_theme->set_icon("layout", "GraphEdit", p_theme->get_icon(SNAME("GridLayout"), EditorStringName(EditorIcons)));

		// GraphEditMinimap.
		{
			Ref<StyleBoxFlat> style_minimap_bg = EditorThemeManager::make_flat_stylebox(p_config.surface_low_color * Color(1, 1, 1, 0.3), 0, 0, 0, 0);
			style_minimap_bg->set_border_color(p_config.dark_color_3);
			style_minimap_bg->set_border_width_all(1);
			p_theme->set_stylebox(SceneStringName(panel), "GraphEditMinimap", style_minimap_bg);

			Ref<StyleBoxFlat> style_minimap_camera;
			Ref<StyleBoxFlat> style_minimap_node;
			if (p_config.dark_theme) {
				style_minimap_camera = EditorThemeManager::make_flat_stylebox(Color(0.65, 0.65, 0.65, 0.2), 0, 0, 0, 0);
				style_minimap_camera->set_border_color(Color(0.65, 0.65, 0.65, 0.45));
				style_minimap_node = EditorThemeManager::make_flat_stylebox(Color(1, 1, 1), 0, 0, 0, 0);
			} else {
				style_minimap_camera = EditorThemeManager::make_flat_stylebox(Color(0.38, 0.38, 0.38, 0.1), 0, 0, 0, 0);
				style_minimap_camera->set_border_color(Color(0.38, 0.38, 0.38, 0.45));
				style_minimap_node = EditorThemeManager::make_flat_stylebox(Color(0, 0, 0), 0, 0, 0, 0);
			}
			style_minimap_camera->set_border_width_all(1);
			style_minimap_node->set_anti_aliased(false);
			p_theme->set_stylebox("camera", "GraphEditMinimap", style_minimap_camera);
			p_theme->set_stylebox("node", "GraphEditMinimap", style_minimap_node);

			const Color minimap_resizer_color = p_config.dark_theme ? Color(1, 1, 1, 0.65) : Color(0, 0, 0, 0.65);
			p_theme->set_icon("resizer", "GraphEditMinimap", p_theme->get_icon(SNAME("GuiResizerTopLeft"), EditorStringName(EditorIcons)));
			p_theme->set_color("resizer_color", "GraphEditMinimap", minimap_resizer_color);
		}

		// GraphElement, GraphNode & GraphFrame.
		{
			const int gn_margin_top = 2;
			const int gn_margin_side = 2;
			const int gn_margin_bottom = 2;

			const int gn_corner_radius = 3;

			const Color gn_bg_color = p_config.dark_theme ? p_config.dark_color_3 : p_config.dark_color_1.lerp(p_config.mono_color, 0.09);
			const Color gn_frame_bg = _get_base_color(p_config, 2.4, 0.9);

			const bool high_contrast_borders = p_config.draw_extra_borders && p_config.dark_theme;

			Ref<StyleBoxFlat> gn_panel_style = EditorThemeManager::make_flat_stylebox(gn_frame_bg, gn_margin_side, gn_margin_top, gn_margin_side, gn_margin_bottom, p_config.corner_radius);
			gn_panel_style->set_border_width(SIDE_BOTTOM, 2 * EDSCALE);
			gn_panel_style->set_border_width(SIDE_LEFT, 2 * EDSCALE);
			gn_panel_style->set_border_width(SIDE_RIGHT, 2 * EDSCALE);
			gn_panel_style->set_border_color(high_contrast_borders ? gn_bg_color.lightened(0.2) : gn_bg_color.darkened(0.3));
			gn_panel_style->set_corner_radius_individual(0, 0, gn_corner_radius * EDSCALE, gn_corner_radius * EDSCALE);
			gn_panel_style->set_anti_aliased(true);

			Ref<StyleBoxFlat> gn_panel_selected_style = gn_panel_style->duplicate();
			gn_panel_selected_style->set_bg_color(p_config.dark_theme ? gn_bg_color.lightened(0.15) : gn_bg_color.darkened(0.15));
			gn_panel_selected_style->set_border_width(SIDE_TOP, 0);
			gn_panel_selected_style->set_border_width(SIDE_BOTTOM, 2 * EDSCALE);
			gn_panel_selected_style->set_border_width(SIDE_LEFT, 2 * EDSCALE);
			gn_panel_selected_style->set_border_width(SIDE_RIGHT, 2 * EDSCALE);
			gn_panel_selected_style->set_border_color(p_config.mono_color);

			const int gn_titlebar_margin_top = 8;
			const int gn_titlebar_margin_side = 12;
			const int gn_titlebar_margin_bottom = 8;

			Ref<StyleBoxFlat> gn_titlebar_style = EditorThemeManager::make_flat_stylebox(gn_bg_color, gn_titlebar_margin_side, gn_titlebar_margin_top, gn_titlebar_margin_side, gn_titlebar_margin_bottom, p_config.corner_radius);
			gn_titlebar_style->set_border_width(SIDE_TOP, 2 * EDSCALE);
			gn_titlebar_style->set_border_width(SIDE_LEFT, 2 * EDSCALE);
			gn_titlebar_style->set_border_width(SIDE_RIGHT, 2 * EDSCALE);
			gn_titlebar_style->set_border_color(high_contrast_borders ? gn_bg_color.lightened(0.2) : gn_bg_color.darkened(0.3));
			gn_titlebar_style->set_expand_margin(SIDE_TOP, 2 * EDSCALE);
			gn_titlebar_style->set_corner_radius_individual(gn_corner_radius * EDSCALE, gn_corner_radius * EDSCALE, 0, 0);
			gn_titlebar_style->set_anti_aliased(true);

			Ref<StyleBoxFlat> gn_titlebar_selected_style = gn_titlebar_style->duplicate();
			gn_titlebar_selected_style->set_border_color(p_config.mono_color);
			gn_titlebar_selected_style->set_border_width(SIDE_TOP, 2 * EDSCALE);
			gn_titlebar_selected_style->set_border_width(SIDE_LEFT, 2 * EDSCALE);
			gn_titlebar_selected_style->set_border_width(SIDE_RIGHT, 2 * EDSCALE);
			gn_titlebar_selected_style->set_expand_margin(SIDE_TOP, 2 * EDSCALE);

			Color gn_decoration_color = p_config.dark_color_1.inverted();

			// GraphElement.

			p_theme->set_stylebox(SceneStringName(panel), "GraphElement", gn_panel_style);
			p_theme->set_stylebox("panel_selected", "GraphElement", gn_panel_selected_style);
			p_theme->set_stylebox("titlebar", "GraphElement", gn_titlebar_style);
			p_theme->set_stylebox("titlebar_selected", "GraphElement", gn_titlebar_selected_style);

			p_theme->set_color("resizer_color", "GraphElement", gn_decoration_color);
			p_theme->set_icon("resizer", "GraphElement", p_theme->get_icon(SNAME("GuiResizer"), EditorStringName(EditorIcons)));

			// GraphNode.

			Ref<StyleBoxEmpty> gn_slot_style = EditorThemeManager::make_empty_stylebox(12, 0, 12, 0);

			p_theme->set_stylebox(SceneStringName(panel), "GraphNode", gn_panel_style);
			p_theme->set_stylebox("panel_selected", "GraphNode", gn_panel_selected_style);
			p_theme->set_stylebox("panel_focus", "GraphNode", p_config.focus_style);
			p_theme->set_stylebox("titlebar", "GraphNode", gn_titlebar_style);
			p_theme->set_stylebox("titlebar_selected", "GraphNode", gn_titlebar_selected_style);
			p_theme->set_stylebox("slot", "GraphNode", gn_slot_style);
			p_theme->set_stylebox("slot_selected", "GraphNode", p_config.focus_style);

			const Color gn_separator_color = gn_frame_bg.lerp(p_config.mono_color, 0.1);
			p_theme->set_stylebox("separator", "GraphNode", EditorThemeManager::make_line_stylebox(gn_separator_color, Math::round(2 * EDSCALE)));

			p_theme->set_color("resizer_color", "GraphNode", gn_decoration_color);

			p_theme->set_constant("port_h_offset", "GraphNode", 1);
			p_theme->set_constant("separation", "GraphNode", 1 * EDSCALE);

			Ref<DPITexture> port_icon = p_theme->get_icon(SNAME("GuiGraphNodePort"), EditorStringName(EditorIcons));
			// The true size is 24x24 This is necessary for sharp port icons at high zoom levels in GraphEdit (up to ~200%).
			port_icon->set_size_override(Size2(12, 12));
			p_theme->set_icon("port", "GraphNode", port_icon);

			// GraphNode's title Label.
			p_theme->set_type_variation("GraphNodeTitleLabel", "Label");
			p_theme->set_stylebox(CoreStringName(normal), "GraphNodeTitleLabel", EditorThemeManager::make_empty_stylebox(0, 0, 0, 0));
			p_theme->set_color("font_shadow_color", "GraphNodeTitleLabel", p_config.shadow_color);
			p_theme->set_constant("shadow_outline_size", "GraphNodeTitleLabel", 4);
			p_theme->set_constant("shadow_offset_x", "GraphNodeTitleLabel", 0);
			p_theme->set_constant("shadow_offset_y", "GraphNodeTitleLabel", 1);
			p_theme->set_constant("line_spacing", "GraphNodeTitleLabel", 3 * EDSCALE);

			// GraphFrame.

			const int gf_corner_width = 7 * EDSCALE;
			const int gf_border_width = 2 * MAX(1, EDSCALE);

			Ref<StyleBoxFlat> graphframe_sb = EditorThemeManager::make_flat_stylebox(Color(0.0, 0.0, 0.0, 0.2), gn_margin_side, gn_margin_side, gn_margin_side, gn_margin_bottom, gf_corner_width);
			graphframe_sb->set_expand_margin(SIDE_TOP, 38 * EDSCALE);
			graphframe_sb->set_border_width_all(gf_border_width);
			graphframe_sb->set_border_color(high_contrast_borders ? gn_bg_color.lightened(0.2) : gn_bg_color.darkened(0.3));
			graphframe_sb->set_shadow_size(8 * EDSCALE);
			graphframe_sb->set_shadow_color(Color(p_config.shadow_color, p_config.shadow_color.a * 0.25));
			graphframe_sb->set_anti_aliased(true);

			Ref<StyleBoxFlat> graphframe_sb_selected = graphframe_sb->duplicate();
			graphframe_sb_selected->set_border_color(p_config.mono_color);

			p_theme->set_stylebox(SceneStringName(panel), "GraphFrame", graphframe_sb);
			p_theme->set_stylebox("panel_selected", "GraphFrame", graphframe_sb_selected);
			p_theme->set_stylebox("titlebar", "GraphFrame", EditorThemeManager::make_empty_stylebox(4, 4, 4, 4));
			p_theme->set_stylebox("titlebar_selected", "GraphFrame", EditorThemeManager::make_empty_stylebox(4, 4, 4, 4));
			p_theme->set_color("resizer_color", "GraphFrame", gn_decoration_color);

			// GraphFrame's title Label.
			p_theme->set_type_variation("GraphFrameTitleLabel", "Label");
			p_theme->set_stylebox(CoreStringName(normal), "GraphFrameTitleLabel", memnew(StyleBoxEmpty));
			p_theme->set_font_size(SceneStringName(font_size), "GraphFrameTitleLabel", 22 * EDSCALE);
			p_theme->set_color(SceneStringName(font_color), "GraphFrameTitleLabel", Color(1, 1, 1));
			p_theme->set_color("font_shadow_color", "GraphFrameTitleLabel", Color(1, 1, 1, 0));
			p_theme->set_color("font_outline_color", "GraphFrameTitleLabel", Color(1, 1, 1));
			p_theme->set_constant("shadow_offset_x", "GraphFrameTitleLabel", 1 * EDSCALE);
			p_theme->set_constant("shadow_offset_y", "GraphFrameTitleLabel", 1 * EDSCALE);
			p_theme->set_constant("outline_size", "GraphFrameTitleLabel", 0);
			p_theme->set_constant("shadow_outline_size", "GraphFrameTitleLabel", 1 * EDSCALE);
			p_theme->set_constant("line_spacing", "GraphFrameTitleLabel", 3 * EDSCALE);
		}

		// VisualShader reroute node.
		{
			Ref<StyleBox> vs_reroute_panel_style = EditorThemeManager::make_empty_stylebox();
			Ref<StyleBox> vs_reroute_titlebar_style = vs_reroute_panel_style->duplicate();
			vs_reroute_titlebar_style->set_content_margin_all(16 * EDSCALE);
			p_theme->set_stylebox(SceneStringName(panel), "VSRerouteNode", vs_reroute_panel_style);
			p_theme->set_stylebox("panel_selected", "VSRerouteNode", vs_reroute_panel_style);
			p_theme->set_stylebox("titlebar", "VSRerouteNode", vs_reroute_titlebar_style);
			p_theme->set_stylebox("titlebar_selected", "VSRerouteNode", vs_reroute_titlebar_style);
			p_theme->set_stylebox("slot", "VSRerouteNode", EditorThemeManager::make_empty_stylebox());

			p_theme->set_color("drag_background", "VSRerouteNode", p_config.dark_theme ? Color(0.19, 0.21, 0.24) : Color(0.8, 0.8, 0.8));
			p_theme->set_color("selected_rim_color", "VSRerouteNode", p_config.mono_color);
		}
	}

	// ColorPicker and related nodes.
	{
		// ColorPicker.
		Ref<StyleBoxFlat> circle_style_focus = p_config.base_style->duplicate();
		circle_style_focus->set_border_color(p_config.mono_color * Color(1, 1, 1, 0.3));
		circle_style_focus->set_draw_center(false);
		circle_style_focus->set_corner_radius_all(256 * EDSCALE);
		circle_style_focus->set_corner_detail(32 * EDSCALE);

		p_theme->set_constant("margin", "ColorPicker", p_config.base_margin);
		p_theme->set_constant("sv_width", "ColorPicker", 256 * EDSCALE);
		p_theme->set_constant("sv_height", "ColorPicker", 256 * EDSCALE);
		p_theme->set_constant("h_width", "ColorPicker", 30 * EDSCALE);
		p_theme->set_constant("label_width", "ColorPicker", 10 * EDSCALE);
		p_theme->set_constant("center_slider_grabbers", "ColorPicker", 1);

		p_theme->set_stylebox("sample_focus", "ColorPicker", p_config.focus_style);
		p_theme->set_stylebox("picker_focus_rectangle", "ColorPicker", p_config.focus_style);
		p_theme->set_stylebox("picker_focus_circle", "ColorPicker", circle_style_focus);
		p_theme->set_color("focused_not_editing_cursor_color", "ColorPicker", p_config.highlight_color);

		p_theme->set_icon("screen_picker", "ColorPicker", p_theme->get_icon(SNAME("ColorPick"), EditorStringName(EditorIcons)));
		p_theme->set_icon("shape_circle", "ColorPicker", p_theme->get_icon(SNAME("PickerShapeCircle"), EditorStringName(EditorIcons)));
		p_theme->set_icon("shape_rect", "ColorPicker", p_theme->get_icon(SNAME("PickerShapeRectangle"), EditorStringName(EditorIcons)));
		p_theme->set_icon("shape_rect_wheel", "ColorPicker", p_theme->get_icon(SNAME("PickerShapeRectangleWheel"), EditorStringName(EditorIcons)));
		p_theme->set_icon("add_preset", "ColorPicker", p_theme->get_icon(SNAME("Add"), EditorStringName(EditorIcons)));
		p_theme->set_icon("sample_bg", "ColorPicker", p_theme->get_icon(SNAME("GuiMiniCheckerboard"), EditorStringName(EditorIcons)));
		p_theme->set_icon("sample_revert", "ColorPicker", p_theme->get_icon(SNAME("Reload"), EditorStringName(EditorIcons)));
		p_theme->set_icon("overbright_indicator", "ColorPicker", p_theme->get_icon(SNAME("OverbrightIndicator"), EditorStringName(EditorIcons)));
		p_theme->set_icon("bar_arrow", "ColorPicker", p_theme->get_icon(SNAME("ColorPickerBarArrow"), EditorStringName(EditorIcons)));
		p_theme->set_icon("picker_cursor", "ColorPicker", p_theme->get_icon(SNAME("PickerCursor"), EditorStringName(EditorIcons)));
		p_theme->set_icon("picker_cursor_bg", "ColorPicker", p_theme->get_icon(SNAME("PickerCursorBg"), EditorStringName(EditorIcons)));
		p_theme->set_icon("color_script", "ColorPicker", p_theme->get_icon(SNAME("Script"), EditorStringName(EditorIcons)));

		// ColorPickerButton.
		p_theme->set_icon("bg", "ColorPickerButton", p_theme->get_icon(SNAME("GuiMiniCheckerboard"), EditorStringName(EditorIcons)));

		// ColorPresetButton.
		p_theme->set_stylebox("preset_fg", "ColorPresetButton", EditorThemeManager::make_flat_stylebox(Color(1, 1, 1), 2, 2, 2, 2, 2));
		p_theme->set_icon("preset_bg", "ColorPresetButton", p_theme->get_icon(SNAME("GuiMiniCheckerboard"), EditorStringName(EditorIcons)));
		p_theme->set_icon("overbright_indicator", "ColorPresetButton", p_theme->get_icon(SNAME("OverbrightIndicator"), EditorStringName(EditorIcons)));
	}
}

void ThemeModern::populate_editor_styles(const Ref<EditorTheme> &p_theme, EditorThemeManager::ThemeConfiguration &p_config) {
	// Project manager.
	{
		Ref<StyleBoxFlat> style_project_list = p_config.base_style->duplicate();
		style_project_list->set_bg_color(p_config.surface_low_color);

		p_theme->set_stylebox("panel_container", "ProjectManager", p_config.foreground_panel);
		p_theme->set_stylebox("project_list", "ProjectManager", style_project_list);
		p_theme->set_stylebox("quick_settings_panel", "ProjectManager", style_project_list);
		p_theme->set_constant("sidebar_button_icon_separation", "ProjectManager", int(6 * EDSCALE));
		p_theme->set_icon("browse_folder", "ProjectManager", p_theme->get_icon(SNAME("FolderBrowse"), EditorStringName(EditorIcons)));
		p_theme->set_icon("browse_file", "ProjectManager", p_theme->get_icon(SNAME("FileBrowse"), EditorStringName(EditorIcons)));

		// ProjectTag.
		{
			p_theme->set_type_variation("ProjectTagButton", "Button");

			Ref<StyleBoxFlat> tag = p_config.button_style->duplicate();
			tag->set_border_width_all(0);
			tag->set_corner_radius(CORNER_TOP_LEFT, 0);
			tag->set_corner_radius(CORNER_BOTTOM_LEFT, 0);
			tag->set_corner_radius(CORNER_TOP_RIGHT, 4);
			tag->set_corner_radius(CORNER_BOTTOM_RIGHT, 4);
			p_theme->set_stylebox(CoreStringName(normal), "ProjectTagButton", tag);

			tag = p_config.button_style_hover->duplicate();
			tag->set_border_width_all(0);
			tag->set_corner_radius(CORNER_TOP_LEFT, 0);
			tag->set_corner_radius(CORNER_BOTTOM_LEFT, 0);
			tag->set_corner_radius(CORNER_TOP_RIGHT, 4);
			tag->set_corner_radius(CORNER_BOTTOM_RIGHT, 4);
			p_theme->set_stylebox(SceneStringName(hover), "ProjectTagButton", tag);

			tag = p_config.button_style_pressed->duplicate();
			tag->set_border_width_all(0);
			tag->set_corner_radius(CORNER_TOP_LEFT, 0);
			tag->set_corner_radius(CORNER_BOTTOM_LEFT, 0);
			tag->set_corner_radius(CORNER_TOP_RIGHT, 4);
			tag->set_corner_radius(CORNER_BOTTOM_RIGHT, 4);
			p_theme->set_stylebox(SceneStringName(pressed), "ProjectTagButton", tag);
		}
	}

	// Editor and main screen.
	{
		// Editor background.
		Color background_color = p_config.surface_lowest_color;
		p_theme->set_color("background", EditorStringName(Editor), background_color);
		Ref<StyleBoxFlat> style_bg = p_config.base_style->duplicate();
		style_bg->set_bg_color(background_color);
		style_bg->set_content_margin_all(0);
		style_bg->set_corner_radius_all(0);
		p_theme->set_stylebox("Background", EditorStringName(EditorStyles), style_bg);

		Ref<StyleBoxFlat> editor_panel_foreground = p_config.base_style->duplicate();
		editor_panel_foreground->set_corner_radius_all(0);
		p_theme->set_stylebox("PanelForeground", EditorStringName(EditorStyles), editor_panel_foreground);

		// Editor focus.
		p_theme->set_stylebox("Focus", EditorStringName(EditorStyles), p_config.focus_style);

		Ref<StyleBoxFlat> style_widget_focus_viewport = p_config.base_style->duplicate();
		style_widget_focus_viewport->set_corner_radius_all(0);
		style_widget_focus_viewport->set_border_width_all(2);
		style_widget_focus_viewport->set_border_color(p_config.mono_color * Color(1, 1, 1, 0.07));
		style_widget_focus_viewport->set_draw_center(false);
		p_theme->set_stylebox("FocusViewport", EditorStringName(EditorStyles), style_widget_focus_viewport);

		p_theme->set_stylebox(SceneStringName(panel), "ScrollContainer", p_config.base_empty_style);
		p_theme->set_stylebox("focus", "ScrollContainer", p_config.focus_style);

		// This stylebox is used in 3d and 2d viewports (no borders).
		Ref<StyleBoxFlat> style_content_panel_vp = p_config.content_panel_style->duplicate();
		style_content_panel_vp->set_content_margin_individual(p_config.border_width * 2, p_config.base_margin * EDSCALE, p_config.border_width * 2, p_config.border_width * 2);
		p_theme->set_stylebox("Content", EditorStringName(EditorStyles), style_content_panel_vp);

		// 3D/Spatial editor.
		Ref<StyleBoxFlat> style_info_3d_viewport = p_config.base_style->duplicate();
		Color bg_color = style_info_3d_viewport->get_bg_color() * Color(1, 1, 1, 0.5);
		if (!p_config.dark_theme) {
			// Always use a dark background for the 3D viewport, even in light themes.
			// This is displayed as an overlay of the 3D scene, whose appearance doesn't change with the editor theme.
			// On top of that, dark overlays are more readable than light overlays.
			bg_color.invert();
		}
		style_info_3d_viewport->set_bg_color(bg_color);
		style_info_3d_viewport->set_content_margin_individual(p_config.base_margin * 2 * EDSCALE, p_config.base_margin * 1.5 * EDSCALE, p_config.base_margin * 2 * EDSCALE, p_config.base_margin * 1.5 * EDSCALE);
		p_theme->set_stylebox("Information3dViewport", EditorStringName(EditorStyles), style_info_3d_viewport);

		// 2D, 3D, and Game toolbar.
		p_theme->set_type_variation("MainToolBarMargin", "MarginContainer");
		p_theme->set_constant("margin_left", "MainToolBarMargin", 4 * EDSCALE);
		p_theme->set_constant("margin_right", "MainToolBarMargin", 4 * EDSCALE);
		p_theme->set_constant("margin_top", "MainToolBarMargin", p_config.base_margin * 0.5 * EDSCALE);
		p_theme->set_constant("margin_bottom", "MainToolBarMargin", p_config.base_margin * 0.5 * EDSCALE);

		// 2D and 3D contextual toolbar.
		// Use a custom stylebox to make contextual menu items stand out from the rest.
		// This helps with editor usability as contextual menu items change when selecting nodes,
		// even though it may not be immediately obvious at first.
		Ref<StyleBoxFlat> toolbar_stylebox = p_config.base_style->duplicate();
		toolbar_stylebox->set_bg_color(p_config.surface_higher_color);
		toolbar_stylebox->set_content_margin_all(0);
		p_theme->set_stylebox("ContextualToolbar", EditorStringName(EditorStyles), toolbar_stylebox);

		// Script editor.
		p_theme->set_stylebox("ScriptEditorPanel", EditorStringName(EditorStyles), EditorThemeManager::make_empty_stylebox(p_config.base_margin, 0, p_config.base_margin, p_config.base_margin));
		p_theme->set_stylebox("ScriptEditorPanelFloating", EditorStringName(EditorStyles), EditorThemeManager::make_empty_stylebox(0, 0, 0, 0));
		p_theme->set_stylebox("ScriptEditor", EditorStringName(EditorStyles), EditorThemeManager::make_empty_stylebox(0, 0, 0, 0));

		// Game view.
		p_theme->set_type_variation("GamePanel", "PanelContainer");
		Ref<StyleBoxFlat> game_panel = p_theme->get_stylebox(SceneStringName(panel), SNAME("Panel"))->duplicate();
		game_panel->set_corner_radius_all(0);
		game_panel->set_content_margin_all(0);
		game_panel->set_draw_center(true);
		p_theme->set_stylebox(SceneStringName(panel), "GamePanel", game_panel);

		// Main menu.
		p_theme->set_stylebox(CoreStringName(normal), "MainScreenButton", p_config.base_empty_wide_style);
		p_theme->set_stylebox("normal_mirrored", "MainScreenButton", p_config.base_empty_wide_style);
		p_theme->set_stylebox(SceneStringName(pressed), "MainScreenButton", p_config.base_empty_wide_style);
		p_theme->set_stylebox("pressed_mirrored", "MainScreenButton", p_config.base_empty_wide_style);
		p_theme->set_stylebox(SceneStringName(hover), "MainScreenButton", p_config.base_empty_wide_style);
		p_theme->set_stylebox("hover_mirrored", "MainScreenButton", p_config.base_empty_wide_style);
		p_theme->set_stylebox("hover_pressed", "MainScreenButton", p_config.base_empty_wide_style);
		p_theme->set_stylebox("hover_pressed_mirrored", "MainScreenButton", p_config.base_empty_wide_style);

		p_theme->set_color("font_pressed_color", "MainScreenButton", p_config.mono_color_font);

		p_theme->set_type_variation("MainMenuBar", "FlatMenuButton");
		p_theme->set_stylebox(CoreStringName(normal), "MainMenuBar", p_config.flat_button);
		p_theme->set_stylebox(SceneStringName(pressed), "MainMenuBar", p_config.flat_button_pressed);
		p_theme->set_stylebox(SceneStringName(hover), "MainMenuBar", p_config.flat_button_hover);
		p_theme->set_stylebox("hover_pressed", "MainMenuBar", p_config.flat_button_hover);

		// Run bar.
		Ref<StyleBoxFlat> run_bar_hover = p_config.base_style->duplicate();
		run_bar_hover->set_bg_color(p_config.mono_color * Color(1, 1, 1, 0.1));
		run_bar_hover->set_content_margin_all(0);

		p_theme->set_type_variation("RunBarButton", "FlatMenuButton");
		p_theme->set_stylebox("disabled", "RunBarButton", p_config.base_empty_wide_style);
		p_theme->set_stylebox(SceneStringName(pressed), "RunBarButton", p_config.base_empty_wide_style);
		p_theme->set_stylebox(SceneStringName(hover), "RunBarButton", run_bar_hover);

		// Needs to present even if unused.
		p_theme->set_type_variation("RunBarButtonMovieMakerDisabled", "RunBarButton");

		Ref<StyleBoxFlat> movie_maker_button_enabled_hover = run_bar_hover->duplicate();
		movie_maker_button_enabled_hover->set_bg_color(p_config.accent_color.lightened(0.2));

		p_theme->set_type_variation("RunBarButtonMovieMakerEnabled", "RunBarButton");
		p_theme->set_stylebox("hover_pressed", "RunBarButtonMovieMakerEnabled", movie_maker_button_enabled_hover);
		p_theme->set_color("icon_normal_color", "RunBarButtonMovieMakerEnabled", Color(0, 0, 0, 0.7));
		p_theme->set_color("icon_pressed_color", "RunBarButtonMovieMakerEnabled", Color(0, 0, 0, 0.84));
		p_theme->set_color("icon_hover_color", "RunBarButtonMovieMakerEnabled", Color(0, 0, 0, 0.9));
		p_theme->set_color("icon_hover_pressed_color", "RunBarButtonMovieMakerEnabled", Color(0, 0, 0, 0.84));

		// Bottom panel.
		Ref<StyleBoxFlat> style_bottom_panel = p_config.content_panel_style->duplicate();
		style_bottom_panel->set_border_width(SIDE_BOTTOM, 0);
		style_bottom_panel->set_corner_radius_all(p_config.corner_radius * EDSCALE);
		style_bottom_panel->set_corner_radius(CORNER_BOTTOM_LEFT, 0);
		style_bottom_panel->set_corner_radius(CORNER_BOTTOM_RIGHT, 0);

		Ref<StyleBoxFlat> style_bottom_panel_hidden = style_bottom_panel->duplicate();
		style_bottom_panel_hidden->set_content_margin(SIDE_TOP, 0);
		Ref<StyleBoxFlat> style_bottom_panel_tabbar = p_config.content_panel_style->duplicate();
		style_bottom_panel_tabbar->set_content_margin(SIDE_TOP, 0);

		Ref<StyleBoxEmpty> style_bottom_tab = p_config.base_empty_style->duplicate();
		style_bottom_tab->set_content_margin_individual(p_config.base_margin * 2 * EDSCALE, p_config.base_margin * 1.2 * EDSCALE, p_config.base_margin * 2 * EDSCALE, p_config.base_margin * 1.2 * EDSCALE);
		Ref<StyleBoxFlat> bottom_panel_button_hover = p_config.flat_button_hover->duplicate();
		bottom_panel_button_hover->set_content_margin_individual(p_config.base_margin * 2 * EDSCALE, p_config.base_margin * 1.2 * EDSCALE, p_config.base_margin * 2 * EDSCALE, p_config.base_margin * 1.2 * EDSCALE);

		p_theme->set_stylebox("BottomPanel", EditorStringName(EditorStyles), style_bottom_panel);
		p_theme->set_type_variation("BottomPanel", "TabContainer");
		p_theme->set_stylebox(SceneStringName(panel), "BottomPanel", style_bottom_panel_hidden);
		p_theme->set_stylebox("tabbar_background", "BottomPanel", style_bottom_panel_tabbar);
		p_theme->set_stylebox("tab_selected", "BottomPanel", bottom_panel_button_hover);
		p_theme->set_stylebox("tab_hovered", "BottomPanel", bottom_panel_button_hover);
		p_theme->set_stylebox("tab_focus", "BottomPanel", p_config.focus_style);
		p_theme->set_stylebox("tab_unselected", "BottomPanel", style_bottom_tab);
		p_theme->set_color("font_unselected_color", "BottomPanel", p_config.font_color);
		p_theme->set_color("font_hovered_color", "BottomPanel", p_config.font_hover_color);
		p_theme->set_color("font_selected_color", "BottomPanel", p_config.font_hover_color);
		p_theme->set_constant("tab_separation", "BottomPanel", p_config.separation_margin);

		p_theme->set_type_variation("BottomPanelButton", "FlatMenuButton");

		// Use bigger margin for buttons in bottom panel to make them easier to press.
		Ref<StyleBoxEmpty> bottom_panel_button = p_config.base_empty_style->duplicate();
		bottom_panel_button->set_content_margin_individual(p_config.base_margin * 2 * EDSCALE, p_config.base_margin * 1.2 * EDSCALE, p_config.base_margin * 2 * EDSCALE, p_config.base_margin * 1.2 * EDSCALE);
		p_theme->set_stylebox(CoreStringName(normal), "BottomPanelButton", bottom_panel_button);

		p_theme->set_stylebox(SceneStringName(hover), "BottomPanelButton", bottom_panel_button_hover);
		p_theme->set_stylebox(SceneStringName(pressed), "BottomPanelButton", bottom_panel_button_hover);

		Ref<StyleBoxFlat> bottom_panel_button_pressed = p_config.flat_button_hover->duplicate();
		bottom_panel_button_pressed->set_content_margin_individual(p_config.base_margin * 2 * EDSCALE, p_config.base_margin * 1.2 * EDSCALE, p_config.base_margin * 2 * EDSCALE, p_config.base_margin * 1.2 * EDSCALE);
		p_theme->set_stylebox("hover_pressed", "BottomPanelButton", bottom_panel_button_pressed);

		// Don't tint the icon even when in "pressed" state.
		p_theme->set_color("icon_pressed_color", "BottomPanelButton", Color(1, 1, 1, 1));
		Color icon_hover_color = p_config.icon_normal_color * (p_config.dark_icon_and_font ? 1.15 : 1.0);
		icon_hover_color.a = 1.0;
		p_theme->set_color("icon_hover_color", "BottomPanel", icon_hover_color);
		p_theme->set_color("icon_hover_pressed_color", "BottomPanel", icon_hover_color);
		p_theme->set_color("icon_hover_color", "BottomPanelButton", icon_hover_color);
		p_theme->set_color("icon_hover_pressed_color", "BottomPanelButton", p_config.accent_color);
		p_theme->set_color("icon_pressed_color", "BottomPanelButton", p_config.accent_color);

		// Audio bus.
		Ref<StyleBoxFlat> audio_bus = p_config.base_style->duplicate();
		audio_bus->set_bg_color(p_config.surface_high_color);
		p_theme->set_stylebox(CoreStringName(normal), "EditorAudioBus", audio_bus);
		p_theme->set_stylebox("focus", "EditorAudioBus", p_config.focus_style);
		Ref<StyleBoxFlat> audio_bus_master = p_config.base_style->duplicate();
		audio_bus_master->set_bg_color(p_config.surface_higher_color);
		p_theme->set_stylebox("master", "EditorAudioBus", audio_bus_master);
	}

	// Editor GUI widgets.
	{
		// EditorSpinSlider.
		p_theme->set_color("label_color", "EditorSpinSlider", p_config.font_color);
		p_theme->set_color("read_only_label_color", "EditorSpinSlider", p_config.font_readonly_color);

		Ref<StyleBoxFlat> editor_spin_label_bg = p_config.base_style->duplicate();
		editor_spin_label_bg->set_bg_color(p_config.surface_lower_color);
		editor_spin_label_bg->set_content_margin_all(p_config.base_margin * EDSCALE);
		p_theme->set_stylebox("label_bg", "EditorSpinSlider", editor_spin_label_bg);

		// TODO Use separate arrows instead like on SpinBox. Planned for a different PR.
		p_theme->set_icon("updown", "EditorSpinSlider", p_theme->get_icon(SNAME("GuiSpinboxUpdown"), EditorStringName(EditorIcons)));
		p_theme->set_icon("updown_disabled", "EditorSpinSlider", p_theme->get_icon(SNAME("GuiSpinboxUpdownDisabled"), EditorStringName(EditorIcons)));

		// Launch Pad and Play buttons.
		Ref<StyleBoxFlat> style_launch_pad_movie = p_config.base_style->duplicate();
		style_launch_pad_movie->set_content_margin_all(p_config.base_margin * EDSCALE);
		style_launch_pad_movie->set_bg_color(p_config.accent_color * Color(1, 1, 1, 0.2));
		style_launch_pad_movie->set_border_color(p_config.accent_color * Color(1, 1, 1, 0.8));
		style_launch_pad_movie->set_border_width_all(Math::round(2 * EDSCALE));
		p_theme->set_stylebox("LaunchPadMovieMode", EditorStringName(EditorStyles), style_launch_pad_movie);

		int pad_normal_margin = style_launch_pad_movie->get_minimum_size().width / 2;
		Ref<StyleBoxEmpty> style_launch_pad = memnew(StyleBoxEmpty);
		style_launch_pad->set_content_margin_all(pad_normal_margin);
		p_theme->set_stylebox("LaunchPadNormal", EditorStringName(EditorStyles), style_launch_pad);

		Ref<StyleBoxFlat> style_launch_pad_recovery_mode = EditorThemeManager::make_flat_stylebox(p_config.dark_color_1, 2 * EDSCALE, 0, 2 * EDSCALE, 0, p_config.corner_radius);
		style_launch_pad_recovery_mode->set_bg_color(p_config.accent_color * Color(1, 1, 1, 0.1));
		style_launch_pad_recovery_mode->set_border_color(p_config.warning_color);
		style_launch_pad_recovery_mode->set_border_width_all(Math::round(2 * EDSCALE));
		style_launch_pad_recovery_mode->set_corner_radius_all(p_config.corner_radius * EDSCALE);
		p_theme->set_stylebox("LaunchPadRecoveryMode", EditorStringName(EditorStyles), style_launch_pad_recovery_mode);

		p_theme->set_stylebox("MovieWriterButtonNormal", EditorStringName(EditorStyles), EditorThemeManager::make_empty_stylebox(0, 0, 0, 0));
		Ref<StyleBoxFlat> style_write_movie_button = p_config.base_style->duplicate();
		style_write_movie_button->set_bg_color(p_config.accent_color * Color(1, 1, 1, 0.8));
		style_write_movie_button->set_content_margin_all(0);
		p_theme->set_stylebox("MovieWriterButtonPressed", EditorStringName(EditorStyles), style_write_movie_button);

		// Profiler autostart indicator panel.
		Ref<StyleBoxFlat> style_profiler_autostart = EditorThemeManager::make_flat_stylebox(p_config.dark_color_1, 2 * EDSCALE, 0, 2 * EDSCALE, 0, p_config.corner_radius);
		style_profiler_autostart->set_bg_color(Color(1, 0.867, 0.396));
		style_profiler_autostart->set_corner_radius_all(p_config.corner_radius * EDSCALE);
		p_theme->set_type_variation("ProfilerAutostartIndicator", "Button");
		p_theme->set_stylebox(CoreStringName(normal), "ProfilerAutostartIndicator", style_profiler_autostart);
		p_theme->set_stylebox(SceneStringName(pressed), "ProfilerAutostartIndicator", style_profiler_autostart);
		p_theme->set_stylebox(SceneStringName(hover), "ProfilerAutostartIndicator", style_profiler_autostart);

		// Recovery mode button style
		Ref<StyleBoxFlat> style_recovery_mode_button = p_config.button_style_pressed->duplicate();
		style_recovery_mode_button->set_bg_color(p_config.warning_color);
		style_recovery_mode_button->set_corner_radius_all(p_config.corner_radius * EDSCALE);
		style_recovery_mode_button->set_content_margin_all(0);
		// Recovery mode button is implicitly styled from the panel's background.
		// So, remove any existing borders. (e.g. from draw_extra_borders config)
		style_recovery_mode_button->set_border_width_all(0);
		style_recovery_mode_button->set_expand_margin(SIDE_RIGHT, 2 * EDSCALE);
		p_theme->set_stylebox("RecoveryModeButton", EditorStringName(EditorStyles), style_recovery_mode_button);
	}

	// Standard GUI variations.
	{
		// Custom theme type for MarginContainer with 4px margins.
		p_theme->set_type_variation("MarginContainer4px", "MarginContainer");
		p_theme->set_constant("margin_left", "MarginContainer4px", 4 * EDSCALE);
		p_theme->set_constant("margin_top", "MarginContainer4px", 4 * EDSCALE);
		p_theme->set_constant("margin_right", "MarginContainer4px", 4 * EDSCALE);
		p_theme->set_constant("margin_bottom", "MarginContainer4px", 4 * EDSCALE);

		// Header LinkButton variation.
		p_theme->set_type_variation("HeaderSmallLink", "LinkButton");
		p_theme->set_font(SceneStringName(font), "HeaderSmallLink", p_theme->get_font(SceneStringName(font), SNAME("HeaderSmall")));
		p_theme->set_font_size(SceneStringName(font_size), "HeaderSmallLink", p_theme->get_font_size(SceneStringName(font_size), SNAME("HeaderSmall")));

		// Flat button variations.
		{
			p_theme->set_stylebox(CoreStringName(normal), SceneStringName(FlatButton), p_config.base_empty_wide_style);
			p_theme->set_stylebox(SceneStringName(hover), SceneStringName(FlatButton), p_config.flat_button_hover);
			p_theme->set_stylebox(SceneStringName(pressed), SceneStringName(FlatButton), p_config.flat_button_pressed);
			p_theme->set_stylebox("hover_pressed", SceneStringName(FlatButton), p_config.flat_button_pressed);
			p_theme->set_stylebox("disabled", SceneStringName(FlatButton), p_config.base_empty_wide_style);

			p_theme->set_stylebox(CoreStringName(normal), "FlatMenuButton", p_config.base_empty_wide_style);
			p_theme->set_stylebox(SceneStringName(hover), "FlatMenuButton", p_config.flat_button_hover);
			p_theme->set_stylebox(SceneStringName(pressed), "FlatMenuButton", p_config.flat_button_pressed);
			p_theme->set_stylebox("hover_pressed", "FlatMenuButton", p_config.flat_button_pressed);
			p_theme->set_stylebox("disabled", "FlatMenuButton", p_config.base_empty_wide_style);

			// Variation for Editor Log filter buttons.

			p_theme->set_type_variation("EditorLogFilterButton", "Button");
			// When pressed, don't tint the icons with the accent color, just leave them normal.
			p_theme->set_color("icon_pressed_color", "EditorLogFilterButton", p_config.icon_normal_color);
			// When unpressed, dim the icons.
			Color icon_normal_color = Color(p_config.icon_normal_color, (p_config.dark_icon_and_font ? 0.4 : 0.8));
			p_theme->set_color("icon_normal_color", "EditorLogFilterButton", icon_normal_color);
			Color icon_hover_color = p_config.icon_normal_color * (p_config.dark_icon_and_font ? 1.15 : 1.0);
			icon_hover_color.a = 1.0;
			p_theme->set_color("icon_hover_color", "EditorLogFilterButton", icon_hover_color);
			p_theme->set_color("icon_hover_pressed_color", "EditorLogFilterButton", icon_hover_color);

			// Hover and pressed styles are swapped for toggle buttons on purpose.
			p_theme->set_stylebox(CoreStringName(normal), "EditorLogFilterButton", p_config.base_empty_style);
			p_theme->set_stylebox(SceneStringName(hover), "EditorLogFilterButton", p_config.flat_button_pressed);
			p_theme->set_stylebox(SceneStringName(pressed), "EditorLogFilterButton", p_config.flat_button_hover);
		}

		// Buttons styles that stand out against the panel background (e.g. AssetLib).
		{
			p_theme->set_type_variation("PanelBackgroundButton", "Button");

			Ref<StyleBoxFlat> panel_button_style = p_config.button_style->duplicate();
			panel_button_style->set_bg_color(p_config.base_color.lerp(p_config.mono_color, 0.08));

			Ref<StyleBoxFlat> panel_button_style_hover = p_config.button_style_hover->duplicate();
			panel_button_style_hover->set_bg_color(p_config.base_color.lerp(p_config.mono_color, 0.16));

			Ref<StyleBoxFlat> panel_button_style_pressed = p_config.button_style_pressed->duplicate();
			panel_button_style_pressed->set_bg_color(p_config.base_color.lerp(p_config.mono_color, 0.20));

			Ref<StyleBoxFlat> panel_button_style_disabled = p_config.button_style_disabled->duplicate();
			panel_button_style_disabled->set_bg_color(p_config.disabled_bg_color);

			p_theme->set_stylebox(CoreStringName(normal), "PanelBackgroundButton", panel_button_style);
			p_theme->set_stylebox(SceneStringName(hover), "PanelBackgroundButton", panel_button_style_hover);
			p_theme->set_stylebox(SceneStringName(pressed), "PanelBackgroundButton", panel_button_style_pressed);
			p_theme->set_stylebox("disabled", "PanelBackgroundButton", panel_button_style_disabled);
		}

		// Top bar selectors.
		p_theme->set_type_variation("TopBarOptionButton", "OptionButton");
		p_theme->set_font(SceneStringName(font), "TopBarOptionButton", p_theme->get_font(SNAME("bold"), EditorStringName(EditorFonts)));
		p_theme->set_font_size(SceneStringName(font_size), "TopBarOptionButton", p_theme->get_font_size(SNAME("bold_size"), EditorStringName(EditorFonts)));

		// Complex editor windows.
		p_theme->set_stylebox(SceneStringName(panel), "EditorSettingsDialog", p_config.window_complex_style);
		p_theme->set_stylebox(SceneStringName(panel), "ProjectSettingsEditor", p_config.window_complex_style);
		p_theme->set_stylebox(SceneStringName(panel), "ProjectExportDialog", p_config.window_complex_style);
		p_theme->set_stylebox(SceneStringName(panel), "SceneImportSettingsDialog", p_config.window_complex_style);
		p_theme->set_stylebox(SceneStringName(panel), "EditorAbout", p_config.window_complex_style);
		p_theme->set_stylebox(SceneStringName(panel), "ThemeItemEditorDialog", p_config.window_complex_style);

		// MarginContainers with negative margins, to negate borders. Used with scroll hints.
		{
			int panel_margin = p_theme->get_stylebox(SceneStringName(panel), SNAME("PanelContainer"))->get_content_margin(SIDE_LEFT);
			int margin = -panel_margin;

			p_theme->set_type_variation("NoBorderHorizontal", "MarginContainer");
			p_theme->set_constant("margin_left", "NoBorderHorizontal", margin);
			p_theme->set_constant("margin_right", "NoBorderHorizontal", margin);

			p_theme->set_type_variation("NoBorderHorizontalBottom", "MarginContainer");
			p_theme->set_constant("margin_left", "NoBorderHorizontalBottom", margin);
			p_theme->set_constant("margin_right", "NoBorderHorizontalBottom", margin);
			p_theme->set_constant("margin_bottom", "NoBorderHorizontalBottom", margin);

			int bottom_margin = p_theme->get_stylebox(SNAME("BottomPanel"), EditorStringName(EditorStyles))->get_content_margin(SIDE_LEFT);
			margin = -bottom_margin;

			// Used in editors residing in the bottom panel.
			p_theme->set_type_variation("NoBorderBottomPanel", "MarginContainer");
			p_theme->set_constant("margin_left", "NoBorderBottomPanel", margin);
			p_theme->set_constant("margin_right", "NoBorderBottomPanel", margin);

			margin = -panel_margin - bottom_margin;

			// Used in the animation track editor.
			p_theme->set_type_variation("NoBorderAnimation", "MarginContainer");
			p_theme->set_constant("margin_left", "NoBorderAnimation", margin);
			p_theme->set_constant("margin_right", "NoBorderAnimation", margin);

			margin = -p_theme->get_stylebox(SceneStringName(panel), SNAME("AcceptDialog"))->get_content_margin(SIDE_LEFT);

			p_theme->set_type_variation("NoBorderHorizontalWindow", "MarginContainer");
			p_theme->set_constant("margin_left", "NoBorderHorizontalWindow", margin);
			p_theme->set_constant("margin_right", "NoBorderHorizontalWindow", margin);
		}

		// Buttons in material previews.
		{
			const Color dim_light_color = p_config.icon_normal_color.darkened(0.24);
			const Color dim_light_highlighted_color = p_config.icon_normal_color.darkened(0.18);
			Ref<StyleBox> sb_empty_borderless = EditorThemeManager::make_empty_stylebox();

			p_theme->set_type_variation("PreviewLightButton", "Button");
			// When pressed, don't use the accent color tint. When unpressed, dim the icon.
			p_theme->set_color("icon_normal_color", "PreviewLightButton", dim_light_color);
			p_theme->set_color("icon_focus_color", "PreviewLightButton", dim_light_color);
			p_theme->set_color("icon_pressed_color", "PreviewLightButton", p_config.icon_normal_color);
			p_theme->set_color("icon_hover_pressed_color", "PreviewLightButton", p_config.icon_normal_color);
			// Unpressed icon is dim, so use a dim highlight.
			p_theme->set_color("icon_hover_color", "PreviewLightButton", dim_light_highlighted_color);

			p_theme->set_stylebox(CoreStringName(normal), "PreviewLightButton", sb_empty_borderless);
			p_theme->set_stylebox(SceneStringName(hover), "PreviewLightButton", sb_empty_borderless);
			p_theme->set_stylebox("focus", "PreviewLightButton", sb_empty_borderless);
			p_theme->set_stylebox(SceneStringName(pressed), "PreviewLightButton", sb_empty_borderless);
		}

		// TabContainerOdd variation.
		{
			// Used for tabs against the base color background in the classic theme.
			p_theme->set_type_variation("TabContainerOdd", "TabContainer");

			Ref<StyleBoxFlat> style_tab_selected_odd = p_theme->get_stylebox(SNAME("tab_selected"), SNAME("TabContainer"))->duplicate();
			style_tab_selected_odd->set_bg_color(p_config.surface_base_color);
			p_theme->set_stylebox("tab_selected", "TabContainerOdd", style_tab_selected_odd);

			Ref<StyleBoxFlat> style_panel_odd = p_theme->get_stylebox(SceneStringName(panel), SNAME("TabContainer"))->duplicate();
			style_panel_odd->set_bg_color(p_config.surface_base_color);
			p_theme->set_stylebox(SceneStringName(panel), "TabContainerOdd", style_panel_odd);

			p_theme->set_stylebox("tabbar_background", "TabContainerOdd", p_theme->get_stylebox(SNAME("tabbar_background"), SNAME("TabContainer")));
		}

		// TabContainerInner, TabBarInner, PanelContainerTabbarInner variations.
		{
			// Used for tabs against the base color background in the modern theme.
			p_theme->set_type_variation("TabContainerInner", "TabContainer");
			p_theme->set_type_variation("TabBarInner", "TabBar");
			p_theme->set_type_variation("PanelContainerTabbarInner", "PanelContainer");

			Ref<StyleBoxFlat> style_tab_selected_inner = p_theme->get_stylebox(SNAME("tab_selected"), SNAME("TabContainer"))->duplicate();
			style_tab_selected_inner->set_content_margin_individual(p_config.base_margin * 4 * EDSCALE, p_config.base_margin * 1.5 * EDSCALE, p_config.base_margin * 4 * EDSCALE, p_config.base_margin * 1.5 * EDSCALE);
			style_tab_selected_inner->set_corner_radius_all(p_config.corner_radius * EDSCALE);
			p_theme->set_stylebox("tab_selected", "TabContainerInner", style_tab_selected_inner);
			p_theme->set_stylebox("tab_selected", "TabBarInner", style_tab_selected_inner);

			Color background_color = p_config.surface_lower_color.lerp(p_config.mono_color_inv, 0.15);

			Ref<StyleBoxFlat> style_tab_unselected_inner = style_tab_selected_inner->duplicate();
			style_tab_unselected_inner->set_bg_color(background_color);
			p_theme->set_stylebox("tab_unselected", "TabContainerInner", style_tab_unselected_inner);
			p_theme->set_stylebox("tab_unselected", "TabBarInner", style_tab_unselected_inner);

			Ref<StyleBoxFlat> style_tab_hovered_inner = style_tab_selected_inner->duplicate();
			style_tab_hovered_inner->set_bg_color(background_color.lerp(p_config.mono_color, 0.05));
			p_theme->set_stylebox("tab_hovered", "TabContainerInner", style_tab_hovered_inner);
			p_theme->set_stylebox("tab_hovered", "TabBarInner", style_tab_hovered_inner);

			Ref<StyleBoxFlat> style_tab_disabled_inner = style_tab_selected_inner->duplicate();
			style_tab_disabled_inner->set_bg_color(background_color);
			p_theme->set_stylebox("tab_disabled", "TabContainerInner", style_tab_disabled_inner);
			p_theme->set_stylebox("tab_disabled", "TabBarInner", style_tab_disabled_inner);

			Ref<StyleBoxFlat> style_tabbar_background_inner = p_theme->get_stylebox(SNAME("tabbar_background"), SNAME("TabContainer"))->duplicate();
			style_tabbar_background_inner->set_content_margin_all(p_config.base_margin * EDSCALE);
			style_tabbar_background_inner->set_corner_radius_all(p_config.corner_radius * EDSCALE + p_config.base_margin * EDSCALE);
			style_tabbar_background_inner->set_bg_color(background_color);

			p_theme->set_stylebox("tabbar_background", "TabContainerInner", style_tabbar_background_inner);

			p_theme->set_constant("tab_separation", "TabContainerInner", p_config.separation_margin);
			p_theme->set_constant("tab_separation", "TabBarInner", p_config.separation_margin);

			p_theme->set_stylebox(SceneStringName(panel), "PanelContainerTabbarInner", style_tabbar_background_inner);
		}

		// TreeLineEdit.
		{
			Ref<StyleBoxFlat> tree_line_edit_style = p_theme->get_stylebox(CoreStringName(normal), SNAME("LineEdit"))->duplicate();
			tree_line_edit_style->set_corner_radius_all(0);

			Ref<StyleBoxFlat> tree_line_edit_style_focus = p_theme->get_stylebox("focus", SNAME("LineEdit"))->duplicate();
			tree_line_edit_style_focus->set_corner_radius_all(0);

			p_theme->set_type_variation("TreeLineEdit", "LineEdit");
			p_theme->set_stylebox(CoreStringName(normal), "TreeLineEdit", tree_line_edit_style);
			p_theme->set_stylebox("focus", "TreeLineEdit", tree_line_edit_style_focus);
		}

		// EditorValidationPanel.
		Ref<StyleBoxFlat> editor_validation_panel = p_config.base_style->duplicate();
		editor_validation_panel->set_bg_color(p_config.surface_low_color);
		editor_validation_panel->set_content_margin_individual(p_config.base_margin * EDSCALE, p_config.base_margin * 0.5 * EDSCALE, p_config.base_margin * EDSCALE, p_config.base_margin * 0.5 * EDSCALE);
		p_theme->set_stylebox(SceneStringName(panel), "EditorValidationPanel", editor_validation_panel);

		// Sidebars.
		{
			p_theme->set_type_variation("ScrollContainerSecondary", "ScrollContainer");
			p_theme->set_type_variation("TreeSecondary", "Tree");
			p_theme->set_type_variation("ItemListSecondary", "ItemList");
			p_theme->set_type_variation("EditorAudioBusEffectsTree", "Tree");

			Ref<StyleBoxFlat> style_sidebar = p_config.base_style->duplicate();
			style_sidebar->set_bg_color(p_config.surface_low_color);
			if (p_config.draw_extra_borders) {
				style_sidebar->set_border_width_all(1 * EDSCALE);
				style_sidebar->set_border_color(p_config.extra_border_color_2);
			}

			p_theme->set_stylebox(SceneStringName(panel), "ScrollContainerSecondary", style_sidebar);
			p_theme->set_stylebox(SceneStringName(panel), "TreeSecondary", style_sidebar);
			p_theme->set_stylebox(SceneStringName(panel), "ItemListSecondary", style_sidebar);
			// Use it for EditorDebuggerInspector in StackTrace to keep the default 3-column layout,
			// as the debugger inspector is too small to be considered a main area.
			p_theme->set_stylebox(SceneStringName(panel), "EditorDebuggerInspector", style_sidebar);

			// TreeSecondary title headers
			Ref<StyleBoxFlat> style_tree_title_secondary = p_config.base_style->duplicate();
			Color secondary_title_color = _get_base_color(p_config, 1.6, 0.9);
			style_tree_title_secondary->set_bg_color(secondary_title_color);
			style_tree_title_secondary->set_border_color(Color(secondary_title_color, 0));
			style_tree_title_secondary->set_border_width(SIDE_LEFT, Math::ceil(EDSCALE));
			style_tree_title_secondary->set_border_width(SIDE_RIGHT, Math::ceil(EDSCALE));

			p_theme->set_stylebox("title_button_normal", "TreeSecondary", style_tree_title_secondary);
			p_theme->set_stylebox("title_button_hover", "TreeSecondary", style_tree_title_secondary);
			p_theme->set_stylebox("title_button_pressed", "TreeSecondary", style_tree_title_secondary);

			// EditorAudioBusEffectsTree
			Ref<StyleBoxFlat> style_audio_bus_effect_tree = p_config.base_style->duplicate();
			style_audio_bus_effect_tree->set_bg_color(_get_base_color(p_config, 0.3));
			if (p_config.draw_extra_borders) {
				style_audio_bus_effect_tree->set_border_width_all(1 * EDSCALE);
				style_audio_bus_effect_tree->set_border_color(p_config.extra_border_color_2);
			}
			p_theme->set_stylebox(SceneStringName(panel), "EditorAudioBusEffectsTree", style_audio_bus_effect_tree);
		}

		// ForegroundPanel.
		{
			p_theme->set_type_variation("PanelForeground", "Panel");
			p_theme->set_type_variation("EditorInspectorForeground", "EditorInspector");

			p_theme->set_stylebox(SceneStringName(panel), "PanelForeground", p_config.foreground_panel);
			p_theme->set_stylebox(SceneStringName(panel), "EditorInspectorForeground", p_config.foreground_panel);
		}
	}

	// Editor inspector.
	{
		// Panel.
		p_theme->set_stylebox(SceneStringName(panel), "EditorInspector", p_config.base_empty_style);

		// Vertical separation between inspector areas.
		p_theme->set_type_variation("EditorInspectorContainer", "VBoxContainer");
		p_theme->set_constant("separation", "EditorInspectorContainer", Math::ceil(p_config.base_margin * EDSCALE));

		// Vertical separation between inspector sections.
		p_theme->set_type_variation("EditorSectionContainer", "VBoxContainer");
		p_theme->set_constant("separation", "EditorSectionContainer", p_config.base_margin * 0.5 * EDSCALE);

		// Vertical separation between inspector properties.
		p_theme->set_type_variation("EditorPropertyContainer", "VBoxContainer");
		p_theme->set_constant("separation", "EditorPropertyContainer", p_config.base_margin * 0.5 * EDSCALE);

		// EditorProperty.

		Ref<StyleBoxFlat> style_property_bg = p_config.base_style->duplicate();
		style_property_bg->set_bg_color(p_config.highlight_color);
		style_property_bg->set_border_width_all(0);

		Ref<StyleBoxFlat> style_property_bg_selected = p_config.base_style->duplicate();
		style_property_bg_selected->set_bg_color(p_config.mono_color * Color(1, 1, 1, 0.05));

		Ref<StyleBoxFlat> style_property_child_bg = p_config.base_style->duplicate();
		style_property_child_bg->set_bg_color(p_config.surface_lower_color);
		style_property_child_bg->set_content_margin_all(p_config.base_margin * EDSCALE);

		p_theme->set_stylebox("bg", "EditorProperty", p_config.base_empty_style);
		p_theme->set_stylebox("bg_selected", "EditorProperty", style_property_bg_selected);
		p_theme->set_stylebox("child_bg", "EditorProperty", style_property_child_bg);
		p_theme->set_constant("font_offset", "EditorProperty", 8 * EDSCALE);

		const Color property_color = p_config.font_color.lerp(Color(0.5, 0.5, 0.5), 0.5);
		const Color readonly_color = property_color.lerp(p_config.dark_icon_and_font ? Color(0, 0, 0) : Color(1, 1, 1), 0.25);
		const Color readonly_warning_color = p_config.error_color.lerp(p_config.dark_icon_and_font ? Color(0, 0, 0) : Color(1, 1, 1), 0.25);

		p_theme->set_color("property_color", "EditorProperty", p_config.font_color);
		p_theme->set_color("readonly_color", "EditorProperty", readonly_color);
		p_theme->set_color("warning_color", "EditorProperty", p_config.warning_color);
		p_theme->set_color("readonly_warning_color", "EditorProperty", readonly_warning_color);

		Ref<StyleBoxFlat> style_property_group_note = p_config.base_style->duplicate();
		Color property_group_note_color = p_config.accent_color;
		property_group_note_color.a = 0.1;
		style_property_group_note->set_bg_color(property_group_note_color);
		p_theme->set_stylebox("bg_group_note", "EditorProperty", style_property_group_note);

		// EditorInspectorSection.

		Color inspector_section_color = p_config.font_color.lerp(Color(0.5, 0.5, 0.5), 0.35);
		p_theme->set_color(SceneStringName(font_color), "EditorInspectorSection", inspector_section_color);

		Color inspector_indent_color = p_config.accent_color;
		inspector_indent_color.a = 0.2;
		Ref<StyleBoxFlat> inspector_indent_style = EditorThemeManager::make_flat_stylebox(inspector_indent_color, 2.0 * EDSCALE, 0, 2.0 * EDSCALE, 0);
		p_theme->set_stylebox("indent_box", "EditorInspectorSection", inspector_indent_style);
		p_theme->set_constant("indent_size", "EditorInspectorSection", 6.0 * EDSCALE);
		p_theme->set_constant("h_separation", "EditorInspectorSection", p_config.base_margin * EDSCALE);

		Color prop_subsection_stylebox_color = p_config.button_disabled_color.lerp(p_config.base_color, 0.48);
		p_theme->set_color("prop_subsection_stylebox_color", EditorStringName(Editor), prop_subsection_stylebox_color);

		Ref<StyleBoxFlat> prop_subsection_stylebox = p_config.base_style->duplicate();
		prop_subsection_stylebox->set_bg_color(p_theme->get_color("prop_subsection_stylebox_color", EditorStringName(Editor)));
		prop_subsection_stylebox->set_corner_radius_all(p_config.corner_radius * EDSCALE);
		p_theme->set_stylebox("prop_subsection_stylebox", EditorStringName(Editor), prop_subsection_stylebox);

		Ref<StyleBoxFlat> prop_subsection_stylebox_left = prop_subsection_stylebox->duplicate();
		prop_subsection_stylebox_left->set_corner_radius(CORNER_TOP_RIGHT, 0);
		prop_subsection_stylebox_left->set_corner_radius(CORNER_BOTTOM_RIGHT, 0);
		p_theme->set_stylebox("prop_subsection_stylebox_left", EditorStringName(Editor), prop_subsection_stylebox_left);

		Ref<StyleBoxFlat> prop_subsection_stylebox_right = prop_subsection_stylebox->duplicate();
		prop_subsection_stylebox_right->set_corner_radius(CORNER_TOP_LEFT, 0);
		prop_subsection_stylebox_right->set_corner_radius(CORNER_BOTTOM_LEFT, 0);
		p_theme->set_stylebox("prop_subsection_stylebox_right", EditorStringName(Editor), prop_subsection_stylebox_right);

		p_theme->set_color("prop_subsection", EditorStringName(Editor), Color(1, 1, 1, 0));
#ifndef DISABLE_DEPRECATED // Used before 4.3.
		p_theme->set_color("property_color", EditorStringName(Editor), p_config.dark_color_1.lerp(p_config.mono_color_font, 0.12));
#endif

		// EditorInspectorCategory.

		Ref<StyleBoxFlat> category_bg = p_config.base_style->duplicate();
		category_bg->set_bg_color(p_config.surface_high_color);
		category_bg->set_content_margin_individual(0, p_config.base_margin * EDSCALE, 0, p_config.base_margin * EDSCALE);
		if (p_config.draw_extra_borders) {
			category_bg->set_border_width_all(1);
			category_bg->set_border_color(p_config.extra_border_color_2);
		}
		p_theme->set_stylebox("bg", "EditorInspectorCategory", category_bg);

		// EditorInspectorArray.
		p_theme->set_color("bg", "EditorInspectorArray", p_config.surface_base_color);

		p_theme->set_constant("inspector_margin", EditorStringName(Editor), 12 * EDSCALE);

		// Colored EditorProperty.
		for (int i = 0; i < 16; i++) {
			Color si_base_color = p_config.accent_color;

			float hue_rotate = (i * 2 % 16) / 16.0;
			si_base_color.set_hsv(Math::fmod(float(si_base_color.get_h() + hue_rotate), float(1.0)), si_base_color.get_s(), si_base_color.get_v());
			si_base_color = p_config.accent_color.lerp(si_base_color, p_config.subresource_hue_tint);

			// Sub-inspector background.
			Ref<StyleBoxFlat> sub_inspector_bg = p_config.base_style->duplicate();
			sub_inspector_bg->set_bg_color(p_config.dark_color_1.lerp(si_base_color, 0.08));
			sub_inspector_bg->set_border_width_all(2 * EDSCALE);
			sub_inspector_bg->set_border_color(si_base_color * Color(0.7, 0.7, 0.7, 0.8));
			sub_inspector_bg->set_content_margin_all(4 * EDSCALE);
			sub_inspector_bg->set_corner_radius(CORNER_TOP_LEFT, 0);
			sub_inspector_bg->set_corner_radius(CORNER_TOP_RIGHT, 0);

			p_theme->set_stylebox("sub_inspector_bg" + itos(i + 1), EditorStringName(EditorStyles), sub_inspector_bg);

			// EditorProperty background while it has a sub-inspector open.
			Ref<StyleBoxFlat> bg_color = EditorThemeManager::make_flat_stylebox(si_base_color * Color(0.7, 0.7, 0.7, 0.8), 0, 0, 0, 0, p_config.corner_radius);
			bg_color->set_anti_aliased(false);
			bg_color->set_corner_radius(CORNER_BOTTOM_LEFT, 0);
			bg_color->set_corner_radius(CORNER_BOTTOM_RIGHT, 0);

			p_theme->set_stylebox("sub_inspector_property_bg" + itos(i + 1), EditorStringName(EditorStyles), bg_color);

			// Dictionary editor add item.
			// Expand to the left and right by 4px to compensate for the dictionary editor margins.

			Color style_dictionary_bg_color = p_config.dark_color_3.lerp(si_base_color, 0.08);
			Ref<StyleBoxFlat> style_dictionary_add_item = EditorThemeManager::make_flat_stylebox(style_dictionary_bg_color, 0, 4, 0, 4, p_config.corner_radius);
			style_dictionary_add_item->set_expand_margin(SIDE_LEFT, 2 * EDSCALE);
			style_dictionary_add_item->set_expand_margin(SIDE_RIGHT, 2 * EDSCALE);
			p_theme->set_stylebox("DictionaryAddItem" + itos(i + 1), EditorStringName(EditorStyles), style_dictionary_add_item);
		}
		Color si_base_color = p_config.accent_color;

		// Sub-inspector background.
		Ref<StyleBoxFlat> sub_inspector_bg = p_config.base_style->duplicate();
		sub_inspector_bg->set_bg_color(Color(1, 1, 1, 0));
		sub_inspector_bg->set_border_width_all(2 * EDSCALE);
		sub_inspector_bg->set_border_color(p_config.dark_color_1.lerp(si_base_color, 0.15));
		sub_inspector_bg->set_content_margin_all(4 * EDSCALE);
		sub_inspector_bg->set_corner_radius(CORNER_TOP_LEFT, 0);
		sub_inspector_bg->set_corner_radius(CORNER_TOP_RIGHT, 0);

		p_theme->set_stylebox("sub_inspector_bg0", EditorStringName(EditorStyles), sub_inspector_bg);

		// Sub-inspector background no border.

		Ref<StyleBoxFlat> sub_inspector_bg_no_border = p_config.base_style->duplicate();
		sub_inspector_bg_no_border->set_content_margin_all(2 * EDSCALE);
		sub_inspector_bg_no_border->set_bg_color(p_config.dark_color_2.lerp(p_config.dark_color_3, 0.15));
		p_theme->set_stylebox("sub_inspector_bg_no_border", EditorStringName(EditorStyles), sub_inspector_bg_no_border);

		// EditorProperty background while it has a sub-inspector open.
		Ref<StyleBoxFlat> bg_color = EditorThemeManager::make_flat_stylebox(p_config.dark_color_1.lerp(si_base_color, 0.15), 0, 0, 0, 0, p_config.corner_radius);
		bg_color->set_anti_aliased(false);
		bg_color->set_corner_radius(CORNER_BOTTOM_LEFT, 0);
		bg_color->set_corner_radius(CORNER_BOTTOM_RIGHT, 0);

		p_theme->set_stylebox("sub_inspector_property_bg0", EditorStringName(EditorStyles), bg_color);

		p_theme->set_color("sub_inspector_property_color", EditorStringName(EditorStyles), p_config.dark_icon_and_font ? Color(1, 1, 1, 1) : Color(0, 0, 0, 1));

		// Dictionary editor.
		// Expand to the left and right by 4px to compensate for the dictionary editor margins.
		Ref<StyleBoxEmpty> style_dictionary_add_item = EditorThemeManager::make_empty_stylebox(0, 4, 0, 4);
		p_theme->set_stylebox("DictionaryAddItem0", EditorStringName(EditorStyles), style_dictionary_add_item);

		// Object selector.
		p_theme->set_type_variation("ObjectSelectorMargin", "MarginContainer");
		p_theme->set_constant("margin_left", "ObjectSelectorMargin", p_config.base_margin * 2 * EDSCALE);
		p_theme->set_constant("margin_right", "ObjectSelectorMargin", p_config.base_margin * 2.5 * EDSCALE);

		// EditorInspectorButton.

		p_theme->set_type_variation("EditorInspectorButton", "Button");
		Ref<StyleBoxFlat> style_line = p_theme->get_stylebox(CoreStringName(normal), SNAME("LineEdit"));
		float vertical_margin = style_line->get_content_margin(SIDE_TOP);

		Ref<StyleBoxFlat> style_inspector_button = p_config.button_style->duplicate();
		style_inspector_button->set_content_margin(SIDE_TOP, vertical_margin);
		style_inspector_button->set_content_margin(SIDE_BOTTOM, vertical_margin);

		Ref<StyleBoxFlat> style_inspector_button_hover = p_config.button_style_hover->duplicate();
		style_inspector_button_hover->set_content_margin(SIDE_TOP, vertical_margin);
		style_inspector_button_hover->set_content_margin(SIDE_BOTTOM, vertical_margin);

		Ref<StyleBoxFlat> style_inspector_button_pressed = p_config.button_style_pressed->duplicate();
		style_inspector_button_pressed->set_content_margin(SIDE_TOP, vertical_margin);
		style_inspector_button_pressed->set_content_margin(SIDE_BOTTOM, vertical_margin);

		Ref<StyleBoxFlat> style_inspector_button_disabled = p_config.button_style_disabled->duplicate();
		style_inspector_button_disabled->set_content_margin(SIDE_TOP, vertical_margin);
		style_inspector_button_disabled->set_content_margin(SIDE_BOTTOM, vertical_margin);

		p_theme->set_stylebox(CoreStringName(normal), "EditorInspectorButton", style_inspector_button);
		p_theme->set_stylebox(SceneStringName(hover), "EditorInspectorButton", style_inspector_button_hover);
		p_theme->set_stylebox(SceneStringName(pressed), "EditorInspectorButton", style_inspector_button_pressed);
		p_theme->set_stylebox("hover_pressed", "EditorInspectorButton", style_inspector_button_pressed);
		p_theme->set_stylebox("disabled", "EditorInspectorButton", style_inspector_button_disabled);

		// Make the height for properties uniform.
		Ref<StyleBoxFlat> inspector_button_style = p_theme->get_stylebox(CoreStringName(normal), SNAME("EditorInspectorButton"));
		Ref<Font> font = p_theme->get_font(SceneStringName(font), SNAME("LineEdit"));
		int font_size = p_theme->get_font_size(SceneStringName(font_size), SNAME("LineEdit"));
		p_config.inspector_property_height = inspector_button_style->get_minimum_size().height + font->get_height(font_size);
		p_theme->set_constant("inspector_property_height", EditorStringName(Editor), p_config.inspector_property_height);

		// EditorInspectorFlatButton.

		p_theme->set_type_variation("EditorInspectorFlatButton", "FlatButton");

		Ref<StyleBoxFlat> inspector_flat_button_hover = p_config.flat_button_hover->duplicate();
		inspector_flat_button_hover->set_content_margin(SIDE_TOP, vertical_margin);
		inspector_flat_button_hover->set_content_margin(SIDE_BOTTOM, vertical_margin);

		Ref<StyleBoxFlat> inspector_flat_button_pressed = p_config.flat_button_pressed->duplicate();
		inspector_flat_button_pressed->set_content_margin(SIDE_TOP, vertical_margin);
		inspector_flat_button_pressed->set_content_margin(SIDE_BOTTOM, vertical_margin);

		p_theme->set_stylebox(CoreStringName(normal), "EditorInspectorFlatButton", p_config.base_empty_wide_style);
		p_theme->set_stylebox(SceneStringName(hover), "EditorInspectorFlatButton", p_config.flat_button_hover);
		p_theme->set_stylebox(SceneStringName(pressed), "EditorInspectorFlatButton", p_config.flat_button_pressed);
		p_theme->set_stylebox("hover_pressed", "EditorInspectorFlatButton", p_config.flat_button_pressed);
		p_theme->set_stylebox("disabled", "EditorInspectorFlatButton", p_config.base_empty_wide_style);

		// InspectorActionButton.
		p_theme->set_type_variation("InspectorActionButton", "Button");
		p_theme->set_constant("h_separation", "InspectorActionButton", p_config.base_margin * 2 * EDSCALE);
	}

	// Animation Editor.
	{
		// Timeline general.
		p_theme->set_constant("timeline_v_separation", "AnimationTrackEditor", 0);
		p_theme->set_constant("track_v_separation", "AnimationTrackEditor", 0);

		// AnimationTimelineEdit.
		// "primary" is used for integer timeline values, "secondary" for decimals.

		Ref<StyleBoxFlat> style_time_available = p_config.base_style->duplicate();
		style_time_available->set_bg_color(p_config.dark_theme ? p_config.surface_highest_color : p_config.surface_high_color);
		if (p_config.draw_extra_borders) {
			style_time_available->set_border_width_all(Math::round(EDSCALE));
			style_time_available->set_border_color(p_config.extra_border_color_2);
		}

		Ref<StyleBoxFlat> style_time_unavailable = p_config.base_style->duplicate();
		style_time_unavailable->set_bg_color(p_config.dark_theme ? p_config.surface_high_color : p_config.surface_highest_color);

		p_theme->set_stylebox("time_available", "AnimationTimelineEdit", style_time_available);
		p_theme->set_stylebox("time_unavailable", "AnimationTimelineEdit", style_time_unavailable);

		p_theme->set_color("v_line_primary_color", "AnimationTimelineEdit", p_config.mono_color * Color(1, 1, 1, 0.4));
		p_theme->set_color("v_line_secondary_color", "AnimationTimelineEdit", p_config.mono_color * Color(1, 1, 1, 0.08));
		p_theme->set_color("h_line_color", "AnimationTimelineEdit", Color(1, 1, 1, 0));
		p_theme->set_color("font_primary_color", "AnimationTimelineEdit", p_config.font_color);
		p_theme->set_color("font_secondary_color", "AnimationTimelineEdit", p_config.font_disabled_color);

		p_theme->set_constant("v_line_primary_margin", "AnimationTimelineEdit", p_config.base_margin * EDSCALE);
		p_theme->set_constant("v_line_secondary_margin", "AnimationTimelineEdit", p_config.base_margin * 1.5 * EDSCALE);
		p_theme->set_constant("v_line_primary_width", "AnimationTimelineEdit", Math::ceil(2 * EDSCALE));
		p_theme->set_constant("v_line_secondary_width", "AnimationTimelineEdit", Math::ceil(EDSCALE));
		p_theme->set_constant("text_primary_margin", "AnimationTimelineEdit", p_config.base_margin * 0.75 * EDSCALE);
		p_theme->set_constant("text_secondary_margin", "AnimationTimelineEdit", p_config.base_margin * 0.5 * EDSCALE);

		// AnimationTrackEdit.

		Ref<StyleBoxFlat> style_animation_track_odd = p_config.base_style->duplicate();
		style_animation_track_odd->set_bg_color(p_config.surface_base_color);

		Ref<StyleBoxFlat> style_animation_track_hover = p_config.base_style->duplicate();
		style_animation_track_hover->set_bg_color(p_config.surface_high_color);

		Ref<StyleBoxFlat> style_animation_track_focus = p_config.base_style->duplicate();
		style_animation_track_focus->set_content_margin_individual(p_config.base_margin * 1.5 * EDSCALE, p_config.base_margin * EDSCALE, p_config.base_margin * 1.5 * EDSCALE, p_config.base_margin * EDSCALE);
		style_animation_track_focus->set_border_width_all(2);
		style_animation_track_focus->set_border_color(p_config.surface_high_color);
		style_animation_track_focus->set_draw_center(false);

		p_theme->set_stylebox("odd", "AnimationTrackEdit", style_animation_track_odd);
		p_theme->set_stylebox(SceneStringName(hover), "AnimationTrackEdit", style_animation_track_hover);
		p_theme->set_stylebox("focus", "AnimationTrackEdit", style_animation_track_focus);

		p_theme->set_color("h_line_color", "AnimationTrackEdit", Color(1, 1, 1, 0));

		p_theme->set_constant("h_separation", "AnimationTrackEdit", p_config.base_margin * 1.5 * EDSCALE);
		p_theme->set_constant("outer_margin", "AnimationTrackEdit", p_config.increased_margin * 6 * EDSCALE);

		// AnimationTrackEditGroup.

		Ref<StyleBoxFlat> style_animation_track_header = p_config.base_style->duplicate();
		style_animation_track_header->set_bg_color(p_config.surface_low_color);
		style_animation_track_header->set_content_margin_individual(p_config.base_margin * 4 * EDSCALE, p_config.base_margin * EDSCALE, 0, p_config.base_margin * EDSCALE);

		p_theme->set_stylebox("header", "AnimationTrackEditGroup", style_animation_track_header);

		Ref<StyleBoxFlat> style_animation_track_group_hover = p_config.base_style->duplicate();
		style_animation_track_group_hover->set_bg_color(p_config.highlight_color);
		p_theme->set_stylebox(SceneStringName(hover), "AnimationTrackEditGroup", style_animation_track_group_hover);

		p_theme->set_color("bg_color", "AnimationTrackEditGroup", p_config.surface_base_color);
		p_theme->set_color("h_line_color", "AnimationTrackEditGroup", Color(1, 1, 1, 0));
		p_theme->set_color("v_line_color", "AnimationTrackEditGroup", Color(1, 1, 1, 0));

		p_theme->set_constant("h_separation", "AnimationTrackEditGroup", p_config.base_margin * 2 * EDSCALE);
		p_theme->set_constant("v_separation", "AnimationTrackEditGroup", 0);

		// AnimationBezierTrackEdit.

		p_theme->set_color("focus_color", "AnimationBezierTrackEdit", p_config.accent_color * Color(1, 1, 1, 0.8));
		p_theme->set_color("track_focus_color", "AnimationBezierTrackEdit", p_config.mono_color * Color(1, 1, 1, 0.1));
		p_theme->set_color("h_line_color", "AnimationBezierTrackEdit", p_config.mono_color * Color(1, 1, 1, 0.12));
		p_theme->set_color("v_line_color", "AnimationBezierTrackEdit", Color(1, 1, 1, 0));

		p_theme->set_constant("h_separation", "AnimationBezierTrackEdit", (p_config.increased_margin + 2) * EDSCALE);
		p_theme->set_constant("v_separation", "AnimationBezierTrackEdit", p_config.forced_even_separation * EDSCALE);
	}

	// Editor help.
	{
		Ref<StyleBoxFlat> style_editor_help = p_config.base_style->duplicate();
		style_editor_help->set_bg_color(p_config.dark_color_2);
		style_editor_help->set_border_color(p_config.dark_color_3);
		p_theme->set_stylebox("background", "EditorHelp", style_editor_help);

		const Color kbd_color = p_config.font_color.lerp(Color(0.5, 0.5, 0.5), 0.5);

		p_theme->set_color("title_color", "EditorHelp", p_config.accent_color);
		p_theme->set_color("headline_color", "EditorHelp", p_config.mono_color_font);
		p_theme->set_color("text_color", "EditorHelp", p_config.font_color);
		p_theme->set_color("comment_color", "EditorHelp", p_config.font_color * Color(1, 1, 1, 0.6));
		p_theme->set_color("symbol_color", "EditorHelp", p_config.font_color * Color(1, 1, 1, 0.6));
		p_theme->set_color("value_color", "EditorHelp", p_config.font_color * Color(1, 1, 1, 0.6));
		p_theme->set_color("qualifier_color", "EditorHelp", p_config.font_color * Color(1, 1, 1, 0.8));
		p_theme->set_color("type_color", "EditorHelp", p_config.accent_color.lerp(p_config.font_color, 0.5));
		p_theme->set_color("override_color", "EditorHelp", p_config.warning_color);
		p_theme->set_color("selection_color", "EditorHelp", p_config.selection_color);
		p_theme->set_color("link_color", "EditorHelp", p_config.accent_color.lerp(p_config.mono_color_font, 0.8));
		p_theme->set_color("code_color", "EditorHelp", p_config.accent_color.lerp(p_config.mono_color_font, 0.6));
		p_theme->set_color("kbd_color", "EditorHelp", p_config.accent_color.lerp(kbd_color, 0.6));
		p_theme->set_color("code_bg_color", "EditorHelp", _get_base_color(p_config, 1.6, 0.8));
		p_theme->set_color("kbd_bg_color", "EditorHelp", p_config.dark_color_1);
		p_theme->set_color("param_bg_color", "EditorHelp", p_config.dark_color_1);
		p_theme->set_constant(SceneStringName(line_separation), "EditorHelp", Math::round(6 * EDSCALE));
		p_theme->set_constant("table_h_separation", "EditorHelp", 16 * EDSCALE);
		p_theme->set_constant("table_v_separation", "EditorHelp", 6 * EDSCALE);
		p_theme->set_constant("text_highlight_h_padding", "EditorHelp", 1 * EDSCALE);
		p_theme->set_constant("text_highlight_v_padding", "EditorHelp", 2 * EDSCALE);
	}

	// EditorHelpBitTitle.
	{
		Ref<StyleBoxFlat> editor_help_title_style = p_config.base_style->duplicate();
		editor_help_title_style->set_bg_color(_get_base_color(p_config, 1.25, 0.75));
		editor_help_title_style->set_content_margin_individual(p_config.base_margin * 2 * EDSCALE, p_config.base_margin * EDSCALE, p_config.base_margin * 2 * EDSCALE, p_config.base_margin * EDSCALE);
		editor_help_title_style->set_corner_radius_individual(p_config.corner_radius * EDSCALE, p_config.corner_radius * EDSCALE, 0, 0);
		if (p_config.draw_extra_borders) {
			editor_help_title_style->set_border_width_all(Math::round(EDSCALE));
			editor_help_title_style->set_border_color(p_config.extra_border_color_2);
		}

		p_theme->set_type_variation("EditorHelpBitTitle", "RichTextLabel");
		p_theme->set_stylebox(CoreStringName(normal), "EditorHelpBitTitle", editor_help_title_style);
	}

	// EditorHelpBitContent.
	{
		Ref<StyleBoxFlat> editor_help_content_style = p_config.base_style->duplicate();
		editor_help_content_style->set_bg_color(p_config.surface_low_color);
		editor_help_content_style->set_content_margin_individual(p_config.base_margin * 2 * EDSCALE, p_config.base_margin * EDSCALE, p_config.base_margin * 2 * EDSCALE, p_config.base_margin * EDSCALE);
		editor_help_content_style->set_corner_radius_individual(0, 0, p_config.corner_radius * EDSCALE, p_config.corner_radius * EDSCALE);
		if (p_config.draw_extra_borders) {
			editor_help_content_style->set_border_width_all(Math::round(EDSCALE));
			editor_help_content_style->set_border_color(p_config.extra_border_color_2);
		}

		p_theme->set_type_variation("EditorHelpBitContent", "RichTextLabel");
		p_theme->set_stylebox(CoreStringName(normal), "EditorHelpBitContent", editor_help_content_style);
	}

	// Asset Library.
	p_theme->set_stylebox("bg", "AssetLib", EditorThemeManager::make_empty_stylebox(p_config.base_margin, p_config.base_margin, p_config.base_margin, p_config.base_margin));
	p_theme->set_stylebox(SceneStringName(panel), "AssetLib", p_config.foreground_panel);
	p_theme->set_stylebox("downloads", "AssetLib", p_theme->get_stylebox(SceneStringName(panel), SNAME("ScrollContainerSecondary")));
	p_theme->set_color("status_color", "AssetLib", Color(0.5, 0.5, 0.5)); // FIXME: Use a defined color instead.
	p_theme->set_icon("dismiss", "AssetLib", p_theme->get_icon(SNAME("Close"), EditorStringName(EditorIcons)));

	// Debugger.
	Ref<StyleBoxFlat> debugger_panel_style = p_config.content_panel_style->duplicate();
	debugger_panel_style->set_border_width(SIDE_BOTTOM, 0);
	p_theme->set_stylebox("DebuggerPanel", EditorStringName(EditorStyles), debugger_panel_style);

	// Resource and node editors.
	{
		// TextureRegion editor.
		Ref<StyleBoxFlat> style_texture_region_bg = p_config.tree_panel_style->duplicate();
		style_texture_region_bg->set_content_margin_all(0);
		p_theme->set_stylebox("TextureRegionPreviewBG", EditorStringName(EditorStyles), style_texture_region_bg);
		p_theme->set_stylebox("TextureRegionPreviewFG", EditorStringName(EditorStyles), EditorThemeManager::make_empty_stylebox(0, 0, 0, 0));

		// Theme editor.
		{
			p_theme->set_color("preview_picker_overlay_color", "ThemeEditor", Color(0.1, 0.1, 0.1, 0.25));

			Color theme_preview_picker_bg_color = p_config.accent_color;
			theme_preview_picker_bg_color.a = 0.2;
			Ref<StyleBoxFlat> theme_preview_picker_sb = EditorThemeManager::make_flat_stylebox(theme_preview_picker_bg_color, 0, 0, 0, 0);
			theme_preview_picker_sb->set_border_color(p_config.accent_color);
			theme_preview_picker_sb->set_border_width_all(1.0 * EDSCALE);
			p_theme->set_stylebox("preview_picker_overlay", "ThemeEditor", theme_preview_picker_sb);

			Color theme_preview_picker_label_bg_color = p_config.accent_color;
			theme_preview_picker_label_bg_color.set_v(0.5);
			Ref<StyleBoxFlat> theme_preview_picker_label_sb = EditorThemeManager::make_flat_stylebox(theme_preview_picker_label_bg_color, 4.0, 1.0, 4.0, 3.0);
			p_theme->set_stylebox("preview_picker_label", "ThemeEditor", theme_preview_picker_label_sb);

			Ref<StyleBoxFlat> style_theme_preview_tab = p_theme->get_stylebox(SNAME("tab_selected"), SNAME("TabContainer"))->duplicate();
			style_theme_preview_tab->set_expand_margin(SIDE_BOTTOM, 5 * EDSCALE);
			p_theme->set_stylebox("ThemeEditorPreviewFG", EditorStringName(EditorStyles), style_theme_preview_tab);

			Ref<StyleBoxFlat> style_theme_preview_bg_tab = p_theme->get_stylebox(SNAME("tab_unselected"), SNAME("TabContainer"))->duplicate();
			style_theme_preview_bg_tab->set_expand_margin(SIDE_BOTTOM, 2 * EDSCALE);
			p_theme->set_stylebox("ThemeEditorPreviewBG", EditorStringName(EditorStyles), style_theme_preview_bg_tab);
		}

		// VisualShader editor.
		p_theme->set_stylebox("label_style", "VShaderEditor", EditorThemeManager::make_empty_stylebox(4, 6, 4, 6));

		// StateMachine graph.
		{
			p_theme->set_stylebox(SceneStringName(panel), "GraphStateMachine", p_config.tree_panel_style);
			p_theme->set_stylebox("error_panel", "GraphStateMachine", p_config.tree_panel_style);
			p_theme->set_color("error_color", "GraphStateMachine", p_config.error_color);

			const int sm_margin_side = 10 * EDSCALE;
			const int sm_margin_bottom = 2;
			const Color sm_bg_color = p_config.dark_theme ? p_config.dark_color_3 : p_config.dark_color_1.lerp(p_config.mono_color, 0.09);

			Ref<StyleBoxFlat> sm_node_style = EditorThemeManager::make_flat_stylebox(p_config.dark_color_3 * Color(1, 1, 1, 0.7), sm_margin_side, 24 * EDSCALE, sm_margin_side, sm_margin_bottom, p_config.corner_radius);
			sm_node_style->set_border_width_all(p_config.border_width);
			sm_node_style->set_border_color(sm_bg_color);

			Ref<StyleBoxFlat> sm_node_selected_style = EditorThemeManager::make_flat_stylebox(sm_bg_color * Color(1, 1, 1, 0.9), sm_margin_side, 24 * EDSCALE, sm_margin_side, sm_margin_bottom, p_config.corner_radius);
			sm_node_selected_style->set_border_width_all(2 * EDSCALE + p_config.border_width);
			sm_node_selected_style->set_border_color(p_config.accent_color * Color(1, 1, 1, 0.9));
			sm_node_selected_style->set_shadow_size(8 * EDSCALE);
			sm_node_selected_style->set_shadow_color(p_config.shadow_color);

			Ref<StyleBoxFlat> sm_node_playing_style = sm_node_selected_style->duplicate();
			sm_node_playing_style->set_border_color(p_config.warning_color);
			sm_node_playing_style->set_shadow_color(p_config.warning_color * Color(1, 1, 1, 0.2));
			sm_node_playing_style->set_draw_center(false);

			p_theme->set_stylebox("node_frame", "GraphStateMachine", sm_node_style);
			p_theme->set_stylebox("node_frame_selected", "GraphStateMachine", sm_node_selected_style);
			p_theme->set_stylebox("node_frame_playing", "GraphStateMachine", sm_node_playing_style);

			Ref<StyleBoxFlat> sm_node_start_style = sm_node_style->duplicate();
			sm_node_start_style->set_border_width_all(1 * EDSCALE);
			sm_node_start_style->set_border_color(p_config.success_color.lightened(0.24));
			p_theme->set_stylebox("node_frame_start", "GraphStateMachine", sm_node_start_style);

			Ref<StyleBoxFlat> sm_node_end_style = sm_node_style->duplicate();
			sm_node_end_style->set_border_width_all(1 * EDSCALE);
			sm_node_end_style->set_border_color(p_config.error_color);
			p_theme->set_stylebox("node_frame_end", "GraphStateMachine", sm_node_end_style);

			p_theme->set_font("node_title_font", "GraphStateMachine", p_theme->get_font(SceneStringName(font), SNAME("Label")));
			p_theme->set_font_size("node_title_font_size", "GraphStateMachine", p_theme->get_font_size(SceneStringName(font_size), SNAME("Label")));
			p_theme->set_color("node_title_font_color", "GraphStateMachine", p_config.font_color);

			p_theme->set_color("transition_color", "GraphStateMachine", p_config.font_color);
			p_theme->set_color("transition_disabled_color", "GraphStateMachine", p_config.font_color * Color(1, 1, 1, 0.2));
			p_theme->set_color("transition_icon_color", "GraphStateMachine", Color(1, 1, 1));
			p_theme->set_color("transition_icon_disabled_color", "GraphStateMachine", Color(1, 1, 1, 0.2));
			p_theme->set_color("highlight_color", "GraphStateMachine", p_config.accent_color);
			p_theme->set_color("highlight_disabled_color", "GraphStateMachine", p_config.accent_color * Color(1, 1, 1, 0.6));
			p_theme->set_color("focus_color", "GraphStateMachine", p_config.accent_color * Color(1, 1, 1, 0.8));
			p_theme->set_color("guideline_color", "GraphStateMachine", p_config.font_color * Color(1, 1, 1, 0.3));

			p_theme->set_color("playback_color", "GraphStateMachine", p_config.font_color);
			p_theme->set_color("playback_background_color", "GraphStateMachine", p_config.font_color * Color(1, 1, 1, 0.3));
		}
	}

	// TileSet editor.
	// This editor is using Tree panel for the panel container of expanded view, while the theme
	// needs trees to be transparent, so it needs to have its own style.
	Ref<StyleBoxFlat> tile_expand_style = p_config.base_style->duplicate();
	tile_expand_style->set_corner_radius_all(0);
	p_theme->set_stylebox("expand_panel", "TileSetEditor", tile_expand_style);
}
