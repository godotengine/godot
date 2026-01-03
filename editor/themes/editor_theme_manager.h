/**************************************************************************/
/*  editor_theme_manager.h                                                */
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

#include "editor/themes/editor_theme.h"

class StyleBoxFlat;
class StyleBoxLine;
class StyleBoxTexture;

class EditorThemeManager {
	static int benchmark_run;
	static inline bool outdated_cache = false;
	static inline bool outdated_cache_dirty = true;

	static String get_benchmark_key();

	enum ColorMode {
		AUTO_COLOR,
		DARK_COLOR,
		LIGHT_COLOR,
	};

public:
	struct ThemeConfiguration {
		// Basic properties.

		String style;
		String preset;
		String spacing_preset;

		Color base_color;
		Color accent_color;
		float contrast = 1.0;
		float icon_saturation = 1.0;

		// Extra properties.

		int base_spacing = 4;
		int extra_spacing = 0;
		Size2 dialogs_buttons_min_size = Size2(105, 34);
		int border_width = 0;
		int corner_radius = 3;

		bool draw_extra_borders = false;
		int draw_relationship_lines;
		float relationship_line_opacity = 1.0;
		int thumb_size = 16;
		int class_icon_size = 16;
		bool enable_touch_optimizations = false;
		float gizmo_handle_scale = 1.0;
		int inspector_property_height = 28;
		float subresource_hue_tint = 0.0;
		float dragging_hover_wait_msec = 0;

		// Make sure to keep those in sync with the definitions in the editor settings.
		const float default_icon_saturation = 2.0;
		const int default_relationship_lines = RELATIONSHIP_SELECTED_ONLY;
		const float default_contrast = 0.3;
		const int default_corner_radius = 4;

		// Generated properties.

		bool dark_theme = false;
		bool dark_icon_and_font = false;

		int base_margin = 4;
		int increased_margin = 4;
		int separation_margin = 4;
		int popup_margin = 12;
		int window_border_margin = 8;
		int top_bar_separation = 8;
		int forced_even_separation = 0;

		Color mono_color;
		Color mono_color_font;
		Color mono_color_inv;
		Color dark_color_1;
		Color dark_color_2;
		Color dark_color_3;
		Color contrast_color_1;
		Color contrast_color_2;
		Color highlight_color;
		Color highlight_disabled_color;
		Color info_color;
		Color success_color;
		Color warning_color;
		Color error_color;
		Color extra_border_color_1;
		Color extra_border_color_2;

		Color font_color;
		Color font_secondary_color;
		Color font_focus_color;
		Color font_hover_color;
		Color font_pressed_color;
		Color font_hover_pressed_color;
		Color font_disabled_color;
		Color font_readonly_color;
		Color font_placeholder_color;
		Color font_outline_color;

		Color font_dark_background_color;
		Color font_dark_background_focus_color;
		Color font_dark_background_hover_color;
		Color font_dark_background_pressed_color;
		Color font_dark_background_hover_pressed_color;

		Color icon_normal_color;
		Color icon_secondary_color;
		Color icon_focus_color;
		Color icon_hover_color;
		Color icon_pressed_color;
		Color icon_disabled_color;

		Color surface_popup_color;
		Color surface_lowest_color;
		Color surface_lower_color;
		Color surface_low_color;
		Color surface_base_color;
		Color surface_high_color;
		Color surface_higher_color;
		Color surface_highest_color;

		Color button_normal_color;
		Color button_hover_color;
		Color button_pressed_color;
		Color button_disabled_color;
		Color button_border_normal_color;
		Color button_border_hover_color;
		Color button_border_pressed_color;

		Color shadow_color;
		Color selection_color;
		Color disabled_border_color;
		Color disabled_bg_color;
		Color separator_color;

		Ref<StyleBoxFlat> base_style;
		Ref<StyleBoxFlat> focus_style;
		Ref<StyleBoxEmpty> base_empty_style;
		Ref<StyleBoxEmpty> base_empty_wide_style;

		Ref<StyleBoxFlat> button_style;
		Ref<StyleBoxFlat> button_style_disabled;
		Ref<StyleBoxFlat> button_style_focus;
		Ref<StyleBoxFlat> button_style_pressed;
		Ref<StyleBoxFlat> button_style_hover;

		Ref<StyleBoxFlat> flat_button;
		Ref<StyleBoxFlat> flat_button_pressed;
		Ref<StyleBoxFlat> flat_button_hover;

		Ref<StyleBoxFlat> popup_style;
		Ref<StyleBoxFlat> popup_border_style;
		Ref<StyleBoxFlat> popup_panel_style;
		Ref<StyleBoxFlat> window_style;
		Ref<StyleBoxFlat> window_complex_style;
		Ref<StyleBoxFlat> dialog_style;
		Ref<StyleBoxFlat> panel_container_style;
		Ref<StyleBoxFlat> content_panel_style;
		Ref<StyleBoxFlat> tree_panel_style;
		Ref<StyleBoxFlat> tab_container_style;
		Ref<StyleBoxFlat> foreground_panel;

		Vector2 widget_margin;

		uint32_t hash();
		uint32_t hash_fonts();
		uint32_t hash_icons();
	};

	enum RelationshipLinesMode {
		RELATIONSHIP_NONE,
		RELATIONSHIP_SELECTED_ONLY,
		RELATIONSHIP_ALL,
	};

private:
	static Ref<EditorTheme> _create_base_theme(const Ref<EditorTheme> &p_old_theme = nullptr);
	static ThemeConfiguration _create_theme_config();

	static void _populate_text_editor_styles(const Ref<EditorTheme> &p_theme, ThemeConfiguration &p_config);
	static void _populate_visual_shader_styles(const Ref<EditorTheme> &p_theme, ThemeConfiguration &p_config);

	static void _reset_dirty_flag();

public:
	static Ref<StyleBoxFlat> make_flat_stylebox(Color p_color, float p_margin_left = -1, float p_margin_top = -1, float p_margin_right = -1, float p_margin_bottom = -1, int p_corner_width = 0);
	static Ref<StyleBoxEmpty> make_empty_stylebox(float p_margin_left = -1, float p_margin_top = -1, float p_margin_right = -1, float p_margin_bottom = -1);
	static Ref<StyleBoxTexture> make_stylebox(Ref<Texture2D> p_texture, float p_left, float p_top, float p_right, float p_bottom, float p_margin_left = -1, float p_margin_top = -1, float p_margin_right = -1, float p_margin_bottom = -1, bool p_draw_center = true);
	static Ref<StyleBoxLine> make_line_stylebox(Color p_color, int p_thickness = 1, float p_grow_begin = 1, float p_grow_end = 1, bool p_vertical = false);

	static Ref<EditorTheme> generate_theme(const Ref<EditorTheme> &p_old_theme = nullptr);
	static bool is_generated_theme_outdated();

	static bool is_dark_theme();
	static bool is_dark_icon_and_font();

	static void initialize();
	static void finalize();
};
