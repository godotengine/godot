/*************************************************************************/
/*  default_theme.cpp                                                    */
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

#include "default_theme.h"

#include "core/os/os.h"
#include "default_font.gen.h"
#include "default_theme_icons.gen.h"
#include "scene/resources/font.h"
#include "scene/resources/theme.h"
#include "servers/text_server.h"

#include "modules/modules_enabled.gen.h" // For svg.
#ifdef MODULE_SVG_ENABLED
#include "modules/svg/image_loader_svg.h"
#endif

static float scale = 1.0;

static const int default_margin = 4;
static const int default_corner_radius = 3;

static Ref<StyleBoxFlat> make_flat_stylebox(Color p_color, float p_margin_left = default_margin, float p_margin_top = default_margin, float p_margin_right = default_margin, float p_margin_bottom = default_margin, int p_corner_radius = default_corner_radius, bool p_draw_center = true, int p_border_width = 0) {
	Ref<StyleBoxFlat> style(memnew(StyleBoxFlat));
	style->set_bg_color(p_color);
	style->set_default_margin(SIDE_LEFT, p_margin_left * scale);
	style->set_default_margin(SIDE_RIGHT, p_margin_right * scale);
	style->set_default_margin(SIDE_BOTTOM, p_margin_bottom * scale);
	style->set_default_margin(SIDE_TOP, p_margin_top * scale);

	style->set_corner_radius_all(p_corner_radius);
	style->set_anti_aliased(true);
	// Adjust level of detail based on the corners' effective sizes.
	style->set_corner_detail(MIN(Math::ceil(1.5 * p_corner_radius), 6) * scale);

	style->set_draw_center(p_draw_center);
	style->set_border_width_all(p_border_width);

	return style;
}

static Ref<StyleBoxFlat> sb_expand(Ref<StyleBoxFlat> p_sbox, float p_left, float p_top, float p_right, float p_bottom) {
	p_sbox->set_expand_margin_size(SIDE_LEFT, p_left * scale);
	p_sbox->set_expand_margin_size(SIDE_TOP, p_top * scale);
	p_sbox->set_expand_margin_size(SIDE_RIGHT, p_right * scale);
	p_sbox->set_expand_margin_size(SIDE_BOTTOM, p_bottom * scale);
	return p_sbox;
}

// See also `editor_generate_icon()` in `editor/editor_themes.cpp`.
static Ref<ImageTexture> generate_icon(int p_index) {
	Ref<ImageTexture> icon = memnew(ImageTexture);
	Ref<Image> img = memnew(Image);

#ifdef MODULE_SVG_ENABLED
	// Upsample icon generation only if the scale isn't an integer multiplier.
	// Generating upsampled icons is slower, and the benefit is hardly visible
	// with integer scales.
	const bool upsample = !Math::is_equal_approx(Math::round(scale), scale);
	ImageLoaderSVG img_loader;
	img_loader.create_image_from_string(img, default_theme_icons_sources[p_index], scale, upsample, false);
#endif
	icon->create_from_image(img);

	return icon;
}

static Ref<StyleBox> make_empty_stylebox(float p_margin_left = -1, float p_margin_top = -1, float p_margin_right = -1, float p_margin_bottom = -1) {
	Ref<StyleBox> style(memnew(StyleBoxEmpty));

	style->set_default_margin(SIDE_LEFT, p_margin_left * scale);
	style->set_default_margin(SIDE_RIGHT, p_margin_right * scale);
	style->set_default_margin(SIDE_BOTTOM, p_margin_bottom * scale);
	style->set_default_margin(SIDE_TOP, p_margin_top * scale);

	return style;
}

void fill_default_theme(Ref<Theme> &theme, const Ref<Font> &default_font, Ref<Texture2D> &default_icon, Ref<StyleBox> &default_style, float p_scale) {
	scale = p_scale;

	// Font colors
	const Color control_font_color = Color(0.875, 0.875, 0.875);
	const Color control_font_low_color = Color(0.7, 0.7, 0.7);
	const Color control_font_lower_color = Color(0.65, 0.65, 0.65);
	const Color control_font_hover_color = Color(0.95, 0.95, 0.95);
	const Color control_font_focus_color = Color(0.95, 0.95, 0.95);
	const Color control_font_disabled_color = control_font_color * Color(1, 1, 1, 0.5);
	const Color control_font_pressed_color = Color(1, 1, 1);
	const Color control_selection_color = Color(0.5, 0.5, 0.5);

	// StyleBox colors
	const Color style_normal_color = Color(0.1, 0.1, 0.1, 0.6);
	const Color style_hover_color = Color(0.225, 0.225, 0.225, 0.6);
	const Color style_pressed_color = Color(0, 0, 0, 0.6);
	const Color style_disabled_color = Color(0.1, 0.1, 0.1, 0.3);
	const Color style_focus_color = Color(1, 1, 1, 0.75);
	const Color style_popup_color = Color(0.25, 0.25, 0.25, 1);
	const Color style_popup_border_color = Color(0.175, 0.175, 0.175, 1);
	const Color style_popup_hover_color = Color(0.4, 0.4, 0.4, 1);
	const Color style_selected_color = Color(1, 1, 1, 0.3);
	// Don't use a color too bright to keep the percentage readable.
	const Color style_progress_color = Color(1, 1, 1, 0.4);
	const Color style_separator_color = Color(0.5, 0.5, 0.5);

	// Convert the generated icon sources to a dictionary for easier access.
	// Unlike the editor icons, there is no central repository of icons in the Theme resource itself to keep it tidy.
	Dictionary icons;
	for (int i = 0; i < default_theme_icons_count; i++) {
		icons[default_theme_icons_names[i]] = generate_icon(i);
	}

	// Panel
	theme->set_stylebox("panel", "Panel", make_flat_stylebox(style_normal_color, 0, 0, 0, 0));
	theme->set_stylebox("panel_fg", "Panel", make_flat_stylebox(style_normal_color, 0, 0, 0, 0));

	// Button

	const Ref<StyleBoxFlat> button_normal = make_flat_stylebox(style_normal_color);
	const Ref<StyleBoxFlat> button_hover = make_flat_stylebox(style_hover_color);
	const Ref<StyleBoxFlat> button_pressed = make_flat_stylebox(style_pressed_color);
	const Ref<StyleBoxFlat> button_disabled = make_flat_stylebox(style_disabled_color);
	Ref<StyleBoxFlat> focus = make_flat_stylebox(style_focus_color, default_margin, default_margin, default_margin, default_margin, default_corner_radius, false, 2);
	// Make the focus outline appear to be flush with the buttons it's focusing.
	focus->set_expand_margin_size_all(2 * scale);

	theme->set_stylebox("normal", "Button", button_normal);
	theme->set_stylebox("hover", "Button", button_hover);
	theme->set_stylebox("pressed", "Button", button_pressed);
	theme->set_stylebox("disabled", "Button", button_disabled);
	theme->set_stylebox("focus", "Button", focus);

	theme->set_font("font", "Button", Ref<Font>());
	theme->set_font_size("font_size", "Button", -1);
	theme->set_constant("outline_size", "Button", 0 * scale);

	theme->set_color("font_color", "Button", control_font_color);
	theme->set_color("font_pressed_color", "Button", control_font_pressed_color);
	theme->set_color("font_hover_color", "Button", control_font_hover_color);
	theme->set_color("font_focus_color", "Button", control_font_focus_color);
	theme->set_color("font_hover_pressed_color", "Button", control_font_pressed_color);
	theme->set_color("font_disabled_color", "Button", control_font_disabled_color);
	theme->set_color("font_outline_color", "Button", Color(1, 1, 1));

	theme->set_color("icon_normal_color", "Button", Color(1, 1, 1, 1));
	theme->set_color("icon_pressed_color", "Button", Color(1, 1, 1, 1));
	theme->set_color("icon_hover_color", "Button", Color(1, 1, 1, 1));
	theme->set_color("icon_hover_pressed_color", "Button", Color(1, 1, 1, 1));
	theme->set_color("icon_focus_color", "Button", Color(1, 1, 1, 1));
	theme->set_color("icon_disabled_color", "Button", Color(1, 1, 1, 1));

	theme->set_constant("hseparation", "Button", 2 * scale);

	// LinkButton

	theme->set_stylebox("focus", "LinkButton", focus);

	theme->set_font("font", "LinkButton", Ref<Font>());
	theme->set_font_size("font_size", "LinkButton", -1);

	theme->set_color("font_color", "LinkButton", control_font_color);
	theme->set_color("font_pressed_color", "LinkButton", control_font_pressed_color);
	theme->set_color("font_hover_color", "LinkButton", control_font_hover_color);
	theme->set_color("font_focus_color", "LinkButton", control_font_focus_color);
	theme->set_color("font_outline_color", "LinkButton", Color(1, 1, 1));

	theme->set_constant("outline_size", "LinkButton", 0);
	theme->set_constant("underline_spacing", "LinkButton", 2 * scale);

	// OptionButton
	theme->set_stylebox("focus", "OptionButton", focus);

	Ref<StyleBox> sb_optbutton_normal = make_flat_stylebox(style_normal_color, 2 * default_margin, default_margin, 21, default_margin);
	Ref<StyleBox> sb_optbutton_hover = make_flat_stylebox(style_hover_color, 2 * default_margin, default_margin, 21, default_margin);
	Ref<StyleBox> sb_optbutton_pressed = make_flat_stylebox(style_pressed_color, 2 * default_margin, default_margin, 21, default_margin);
	Ref<StyleBox> sb_optbutton_disabled = make_flat_stylebox(style_disabled_color, 2 * default_margin, default_margin, 21, default_margin);

	theme->set_stylebox("normal", "OptionButton", sb_optbutton_normal);
	theme->set_stylebox("hover", "OptionButton", sb_optbutton_hover);
	theme->set_stylebox("pressed", "OptionButton", sb_optbutton_pressed);
	theme->set_stylebox("disabled", "OptionButton", sb_optbutton_disabled);

	Ref<StyleBox> sb_optbutton_normal_mirrored = make_flat_stylebox(style_normal_color, 21, default_margin, 2 * default_margin, default_margin);
	Ref<StyleBox> sb_optbutton_hover_mirrored = make_flat_stylebox(style_hover_color, 21, default_margin, 2 * default_margin, default_margin);
	Ref<StyleBox> sb_optbutton_pressed_mirrored = make_flat_stylebox(style_pressed_color, 21, default_margin, 2 * default_margin, default_margin);
	Ref<StyleBox> sb_optbutton_disabled_mirrored = make_flat_stylebox(style_disabled_color, 21, default_margin, 2 * default_margin, default_margin);

	theme->set_stylebox("normal_mirrored", "OptionButton", sb_optbutton_normal_mirrored);
	theme->set_stylebox("hover_mirrored", "OptionButton", sb_optbutton_hover_mirrored);
	theme->set_stylebox("pressed_mirrored", "OptionButton", sb_optbutton_pressed_mirrored);
	theme->set_stylebox("disabled_mirrored", "OptionButton", sb_optbutton_disabled_mirrored);

	theme->set_icon("arrow", "OptionButton", icons["option_button_arrow"]);

	theme->set_font("font", "OptionButton", Ref<Font>());
	theme->set_font_size("font_size", "OptionButton", -1);

	theme->set_color("font_color", "OptionButton", control_font_color);
	theme->set_color("font_pressed_color", "OptionButton", control_font_pressed_color);
	theme->set_color("font_hover_color", "OptionButton", control_font_hover_color);
	theme->set_color("font_focus_color", "OptionButton", control_font_focus_color);
	theme->set_color("font_disabled_color", "OptionButton", control_font_disabled_color);
	theme->set_color("font_outline_color", "OptionButton", Color(1, 1, 1));

	theme->set_constant("hseparation", "OptionButton", 2 * scale);
	theme->set_constant("arrow_margin", "OptionButton", 4 * scale);
	theme->set_constant("outline_size", "OptionButton", 0);

	// MenuButton

	theme->set_stylebox("normal", "MenuButton", button_normal);
	theme->set_stylebox("pressed", "MenuButton", button_pressed);
	theme->set_stylebox("hover", "MenuButton", button_hover);
	theme->set_stylebox("disabled", "MenuButton", button_disabled);
	theme->set_stylebox("focus", "MenuButton", focus);

	theme->set_font("font", "MenuButton", Ref<Font>());
	theme->set_font_size("font_size", "MenuButton", -1);

	theme->set_color("font_color", "MenuButton", control_font_color);
	theme->set_color("font_pressed_color", "MenuButton", control_font_pressed_color);
	theme->set_color("font_hover_color", "MenuButton", control_font_hover_color);
	theme->set_color("font_focus_color", "MenuButton", control_font_focus_color);
	theme->set_color("font_disabled_color", "MenuButton", Color(1, 1, 1, 0.3));
	theme->set_color("font_outline_color", "MenuButton", Color(1, 1, 1));

	theme->set_constant("hseparation", "MenuButton", 3 * scale);
	theme->set_constant("outline_size", "MenuButton", 0);

	// CheckBox

	Ref<StyleBox> cbx_empty = memnew(StyleBoxEmpty);
	cbx_empty->set_default_margin(SIDE_LEFT, 4 * scale);
	cbx_empty->set_default_margin(SIDE_RIGHT, 4 * scale);
	cbx_empty->set_default_margin(SIDE_TOP, 4 * scale);
	cbx_empty->set_default_margin(SIDE_BOTTOM, 4 * scale);
	Ref<StyleBox> cbx_focus = focus;
	cbx_focus->set_default_margin(SIDE_LEFT, 4 * scale);
	cbx_focus->set_default_margin(SIDE_RIGHT, 4 * scale);
	cbx_focus->set_default_margin(SIDE_TOP, 4 * scale);
	cbx_focus->set_default_margin(SIDE_BOTTOM, 4 * scale);

	theme->set_stylebox("normal", "CheckBox", cbx_empty);
	theme->set_stylebox("pressed", "CheckBox", cbx_empty);
	theme->set_stylebox("disabled", "CheckBox", cbx_empty);
	theme->set_stylebox("hover", "CheckBox", cbx_empty);
	theme->set_stylebox("hover_pressed", "CheckBox", cbx_empty);
	theme->set_stylebox("focus", "CheckBox", cbx_focus);

	theme->set_icon("checked", "CheckBox", icons["checked"]);
	theme->set_icon("checked_disabled", "CheckBox", icons["checked"]);
	theme->set_icon("unchecked", "CheckBox", icons["unchecked"]);
	theme->set_icon("unchecked_disabled", "CheckBox", icons["unchecked"]);
	theme->set_icon("radio_checked", "CheckBox", icons["radio_checked"]);
	theme->set_icon("radio_checked_disabled", "CheckBox", icons["radio_checked"]);
	theme->set_icon("radio_unchecked", "CheckBox", icons["radio_unchecked"]);
	theme->set_icon("radio_unchecked_disabled", "CheckBox", icons["radio_unchecked"]);

	theme->set_font("font", "CheckBox", Ref<Font>());
	theme->set_font_size("font_size", "CheckBox", -1);

	theme->set_color("font_color", "CheckBox", control_font_color);
	theme->set_color("font_pressed_color", "CheckBox", control_font_pressed_color);
	theme->set_color("font_hover_color", "CheckBox", control_font_hover_color);
	theme->set_color("font_hover_pressed_color", "CheckBox", control_font_pressed_color);
	theme->set_color("font_focus_color", "CheckBox", control_font_focus_color);
	theme->set_color("font_disabled_color", "CheckBox", control_font_disabled_color);
	theme->set_color("font_outline_color", "CheckBox", Color(1, 1, 1));

	theme->set_constant("hseparation", "CheckBox", 4 * scale);
	theme->set_constant("check_vadjust", "CheckBox", 0 * scale);
	theme->set_constant("outline_size", "CheckBox", 0);

	// CheckButton

	Ref<StyleBox> cb_empty = memnew(StyleBoxEmpty);
	cb_empty->set_default_margin(SIDE_LEFT, 6 * scale);
	cb_empty->set_default_margin(SIDE_RIGHT, 6 * scale);
	cb_empty->set_default_margin(SIDE_TOP, 4 * scale);
	cb_empty->set_default_margin(SIDE_BOTTOM, 4 * scale);

	theme->set_stylebox("normal", "CheckButton", cb_empty);
	theme->set_stylebox("pressed", "CheckButton", cb_empty);
	theme->set_stylebox("disabled", "CheckButton", cb_empty);
	theme->set_stylebox("hover", "CheckButton", cb_empty);
	theme->set_stylebox("hover_pressed", "CheckButton", cb_empty);
	theme->set_stylebox("focus", "CheckButton", focus);

	theme->set_icon("on", "CheckButton", icons["toggle_on"]);
	theme->set_icon("on_disabled", "CheckButton", icons["toggle_on_disabled"]);
	theme->set_icon("off", "CheckButton", icons["toggle_off"]);
	theme->set_icon("off_disabled", "CheckButton", icons["toggle_off_disabled"]);

	theme->set_icon("on_mirrored", "CheckButton", icons["toggle_on_mirrored"]);
	theme->set_icon("on_disabled_mirrored", "CheckButton", icons["toggle_on_disabled_mirrored"]);
	theme->set_icon("off_mirrored", "CheckButton", icons["toggle_off_mirrored"]);
	theme->set_icon("off_disabled_mirrored", "CheckButton", icons["toggle_off_disabled_mirrored"]);

	theme->set_font("font", "CheckButton", Ref<Font>());
	theme->set_font_size("font_size", "CheckButton", -1);

	theme->set_color("font_color", "CheckButton", control_font_color);
	theme->set_color("font_pressed_color", "CheckButton", control_font_pressed_color);
	theme->set_color("font_hover_color", "CheckButton", control_font_hover_color);
	theme->set_color("font_hover_pressed_color", "CheckButton", control_font_pressed_color);
	theme->set_color("font_focus_color", "CheckButton", control_font_focus_color);
	theme->set_color("font_disabled_color", "CheckButton", control_font_disabled_color);
	theme->set_color("font_outline_color", "CheckButton", Color(1, 1, 1));

	theme->set_constant("hseparation", "CheckButton", 4 * scale);
	theme->set_constant("check_vadjust", "CheckButton", 0 * scale);
	theme->set_constant("outline_size", "CheckButton", 0);

	// Label

	theme->set_stylebox("normal", "Label", memnew(StyleBoxEmpty));
	theme->set_font("font", "Label", Ref<Font>());
	theme->set_font_size("font_size", "Label", -1);

	theme->set_color("font_color", "Label", Color(1, 1, 1));
	theme->set_color("font_shadow_color", "Label", Color(0, 0, 0, 0));
	theme->set_color("font_outline_color", "Label", Color(1, 1, 1));

	theme->set_constant("shadow_offset_x", "Label", 1 * scale);
	theme->set_constant("shadow_offset_y", "Label", 1 * scale);
	theme->set_constant("outline_size", "Label", 0);
	theme->set_constant("shadow_outline_size", "Label", 1 * scale);
	theme->set_constant("line_spacing", "Label", 3 * scale);

	theme->set_type_variation("HeaderSmall", "Label");
	theme->set_font_size("font_size", "HeaderSmall", default_font_size + 4);

	theme->set_type_variation("HeaderMedium", "Label");
	theme->set_font_size("font_size", "HeaderMedium", default_font_size + 8);

	theme->set_type_variation("HeaderLarge", "Label");
	theme->set_font_size("font_size", "HeaderLarge", default_font_size + 12);

	// LineEdit

	Ref<StyleBoxFlat> style_line_edit = make_flat_stylebox(style_normal_color);
	// Add a line at the bottom to make LineEdits distinguishable from Buttons.
	style_line_edit->set_border_width(SIDE_BOTTOM, 2);
	style_line_edit->set_border_color(style_pressed_color);
	theme->set_stylebox("normal", "LineEdit", style_line_edit);

	theme->set_stylebox("focus", "LineEdit", focus);

	Ref<StyleBoxFlat> style_line_edit_read_only = make_flat_stylebox(style_disabled_color);
	// Add a line at the bottom to make LineEdits distinguishable from Buttons.
	style_line_edit_read_only->set_border_width(SIDE_BOTTOM, 2);
	style_line_edit_read_only->set_border_color(style_pressed_color * Color(1, 1, 1, 0.5));
	theme->set_stylebox("read_only", "LineEdit", style_line_edit_read_only);

	theme->set_font("font", "LineEdit", Ref<Font>());
	theme->set_font_size("font_size", "LineEdit", -1);

	theme->set_color("font_color", "LineEdit", control_font_color);
	theme->set_color("font_selected_color", "LineEdit", control_font_pressed_color);
	theme->set_color("font_uneditable_color", "LineEdit", control_font_disabled_color);
	theme->set_color("font_outline_color", "LineEdit", Color(1, 1, 1));
	theme->set_color("caret_color", "LineEdit", control_font_hover_color);
	theme->set_color("selection_color", "LineEdit", control_selection_color);
	theme->set_color("clear_button_color", "LineEdit", control_font_color);
	theme->set_color("clear_button_color_pressed", "LineEdit", control_font_pressed_color);

	theme->set_constant("minimum_character_width", "LineEdit", 4);
	theme->set_constant("outline_size", "LineEdit", 0);
	theme->set_constant("caret_width", "LineEdit", 1);

	theme->set_icon("clear", "LineEdit", icons["line_edit_clear"]);

	// ProgressBar

	theme->set_stylebox("bg", "ProgressBar", make_flat_stylebox(style_disabled_color, 2, 2, 2, 2, 6));
	theme->set_stylebox("fg", "ProgressBar", make_flat_stylebox(style_progress_color, 2, 2, 2, 2, 6));

	theme->set_font("font", "ProgressBar", Ref<Font>());
	theme->set_font_size("font_size", "ProgressBar", -1);

	theme->set_color("font_color", "ProgressBar", control_font_hover_color);
	theme->set_color("font_shadow_color", "ProgressBar", Color(0, 0, 0));
	theme->set_color("font_outline_color", "ProgressBar", Color(1, 1, 1));

	theme->set_constant("outline_size", "ProgressBar", 0);

	// TextEdit

	theme->set_stylebox("normal", "TextEdit", style_line_edit);
	theme->set_stylebox("focus", "TextEdit", focus);
	theme->set_stylebox("read_only", "TextEdit", style_line_edit_read_only);

	theme->set_icon("tab", "TextEdit", icons["text_edit_tab"]);
	theme->set_icon("space", "TextEdit", icons["text_edit_space"]);

	theme->set_font("font", "TextEdit", Ref<Font>());
	theme->set_font_size("font_size", "TextEdit", -1);

	theme->set_color("background_color", "TextEdit", Color(0, 0, 0, 0));
	theme->set_color("font_color", "TextEdit", control_font_color);
	theme->set_color("font_selected_color", "TextEdit", control_font_pressed_color);
	theme->set_color("font_readonly_color", "TextEdit", control_font_disabled_color);
	theme->set_color("font_outline_color", "TextEdit", Color(1, 1, 1));
	theme->set_color("selection_color", "TextEdit", control_selection_color);
	theme->set_color("current_line_color", "TextEdit", Color(0.25, 0.25, 0.26, 0.8));
	theme->set_color("caret_color", "TextEdit", control_font_color);
	theme->set_color("caret_background_color", "TextEdit", Color(0, 0, 0));
	theme->set_color("word_highlighted_color", "TextEdit", Color(0.5, 0.5, 0.5, 0.25));
	theme->set_color("search_result_color", "TextEdit", Color(0.3, 0.3, 0.3));
	theme->set_color("search_result_border_color", "TextEdit", Color(0.3, 0.3, 0.3, 0.4));

	theme->set_constant("line_spacing", "TextEdit", 4 * scale);
	theme->set_constant("outline_size", "TextEdit", 0);
	theme->set_constant("caret_width", "TextEdit", 1);

	// CodeEdit

	theme->set_stylebox("normal", "CodeEdit", style_line_edit);
	theme->set_stylebox("focus", "CodeEdit", focus);
	theme->set_stylebox("read_only", "CodeEdit", style_line_edit_read_only);
	theme->set_stylebox("completion", "CodeEdit", make_flat_stylebox(style_normal_color, 0, 0, 0, 0));

	theme->set_icon("tab", "CodeEdit", icons["text_edit_tab"]);
	theme->set_icon("space", "CodeEdit", icons["text_edit_space"]);
	theme->set_icon("breakpoint", "CodeEdit", icons["breakpoint"]);
	theme->set_icon("bookmark", "CodeEdit", icons["bookmark"]);
	theme->set_icon("executing_line", "CodeEdit", icons["arrow_right"]);
	theme->set_icon("can_fold", "CodeEdit", icons["arrow_down"]);
	theme->set_icon("folded", "CodeEdit", icons["arrow_right"]);
	theme->set_icon("folded_eol_icon", "CodeEdit", icons["text_edit_ellipsis"]);

	theme->set_font("font", "CodeEdit", Ref<Font>());
	theme->set_font_size("font_size", "CodeEdit", -1);

	theme->set_color("background_color", "CodeEdit", Color(0, 0, 0, 0));
	theme->set_color("completion_background_color", "CodeEdit", Color(0.17, 0.16, 0.2));
	theme->set_color("completion_selected_color", "CodeEdit", Color(0.26, 0.26, 0.27));
	theme->set_color("completion_existing_color", "CodeEdit", Color(0.87, 0.87, 0.87, 0.13));
	theme->set_color("completion_scroll_color", "CodeEdit", control_font_pressed_color);
	theme->set_color("completion_font_color", "CodeEdit", Color(0.67, 0.67, 0.67));
	theme->set_color("font_color", "CodeEdit", control_font_color);
	theme->set_color("font_selected_color", "CodeEdit", Color(0, 0, 0));
	theme->set_color("font_readonly_color", "CodeEdit", Color(control_font_color.r, control_font_color.g, control_font_color.b, 0.5f));
	theme->set_color("font_outline_color", "CodeEdit", Color(1, 1, 1));
	theme->set_color("selection_color", "CodeEdit", control_selection_color);
	theme->set_color("bookmark_color", "CodeEdit", Color(0.5, 0.64, 1, 0.8));
	theme->set_color("breakpoint_color", "CodeEdit", Color(0.9, 0.29, 0.3));
	theme->set_color("executing_line_color", "CodeEdit", Color(0.98, 0.89, 0.27));
	theme->set_color("current_line_color", "CodeEdit", Color(0.25, 0.25, 0.26, 0.8));
	theme->set_color("code_folding_color", "CodeEdit", Color(0.8, 0.8, 0.8, 0.8));
	theme->set_color("caret_color", "CodeEdit", control_font_color);
	theme->set_color("caret_background_color", "CodeEdit", Color(0, 0, 0));
	theme->set_color("brace_mismatch_color", "CodeEdit", Color(1, 0.2, 0.2));
	theme->set_color("line_number_color", "CodeEdit", Color(0.67, 0.67, 0.67, 0.4));
	theme->set_color("word_highlighted_color", "CodeEdit", Color(0.8, 0.9, 0.9, 0.15));
	theme->set_color("line_length_guideline_color", "CodeEdit", Color(0.3, 0.5, 0.8, 0.1));
	theme->set_color("search_result_color", "CodeEdit", Color(0.3, 0.3, 0.3));
	theme->set_color("search_result_border_color", "CodeEdit", Color(0.3, 0.3, 0.3, 0.4));

	theme->set_constant("completion_lines", "CodeEdit", 7);
	theme->set_constant("completion_max_width", "CodeEdit", 50);
	theme->set_constant("completion_scroll_width", "CodeEdit", 3);
	theme->set_constant("line_spacing", "CodeEdit", 4 * scale);
	theme->set_constant("outline_size", "CodeEdit", 0);

	Ref<Texture2D> empty_icon = memnew(ImageTexture);

	const Ref<StyleBoxFlat> style_scrollbar = make_flat_stylebox(style_normal_color, 4, 4, 4, 4, 10);
	Ref<StyleBoxFlat> style_scrollbar_grabber = make_flat_stylebox(style_progress_color, 4, 4, 4, 4, 10);
	Ref<StyleBoxFlat> style_scrollbar_grabber_highlight = make_flat_stylebox(style_focus_color, 4, 4, 4, 4, 10);
	Ref<StyleBoxFlat> style_scrollbar_grabber_pressed = make_flat_stylebox(style_focus_color * Color(0.75, 0.75, 0.75), 4, 4, 4, 4, 10);

	// HScrollBar

	theme->set_stylebox("scroll", "HScrollBar", style_scrollbar);
	theme->set_stylebox("scroll_focus", "HScrollBar", focus);
	theme->set_stylebox("grabber", "HScrollBar", style_scrollbar_grabber);
	theme->set_stylebox("grabber_highlight", "HScrollBar", style_scrollbar_grabber_highlight);
	theme->set_stylebox("grabber_pressed", "HScrollBar", style_scrollbar_grabber_pressed);

	theme->set_icon("increment", "HScrollBar", empty_icon);
	theme->set_icon("increment_highlight", "HScrollBar", empty_icon);
	theme->set_icon("increment_pressed", "HScrollBar", empty_icon);
	theme->set_icon("decrement", "HScrollBar", empty_icon);
	theme->set_icon("decrement_highlight", "HScrollBar", empty_icon);
	theme->set_icon("decrement_pressed", "HScrollBar", empty_icon);

	// VScrollBar

	theme->set_stylebox("scroll", "VScrollBar", style_scrollbar);
	theme->set_stylebox("scroll_focus", "VScrollBar", focus);
	theme->set_stylebox("grabber", "VScrollBar", style_scrollbar_grabber);
	theme->set_stylebox("grabber_highlight", "VScrollBar", style_scrollbar_grabber_highlight);
	theme->set_stylebox("grabber_pressed", "VScrollBar", style_scrollbar_grabber_pressed);

	theme->set_icon("increment", "VScrollBar", empty_icon);
	theme->set_icon("increment_highlight", "VScrollBar", empty_icon);
	theme->set_icon("increment_pressed", "VScrollBar", empty_icon);
	theme->set_icon("decrement", "VScrollBar", empty_icon);
	theme->set_icon("decrement_highlight", "VScrollBar", empty_icon);
	theme->set_icon("decrement_pressed", "VScrollBar", empty_icon);

	const Ref<StyleBoxFlat> style_slider = make_flat_stylebox(style_normal_color, 4, 4, 4, 4, 4);
	const Ref<StyleBoxFlat> style_slider_grabber = make_flat_stylebox(style_progress_color, 4, 4, 4, 4, 4);
	const Ref<StyleBoxFlat> style_slider_grabber_highlight = make_flat_stylebox(style_focus_color, 4, 4, 4, 4, 4);

	// HSlider

	theme->set_stylebox("slider", "HSlider", style_slider);
	theme->set_stylebox("grabber_area", "HSlider", style_slider_grabber);
	theme->set_stylebox("grabber_area_highlight", "HSlider", style_slider_grabber_highlight);

	theme->set_icon("grabber", "HSlider", icons["slider_grabber"]);
	theme->set_icon("grabber_highlight", "HSlider", icons["slider_grabber_hl"]);
	theme->set_icon("grabber_disabled", "HSlider", icons["slider_grabber_disabled"]);
	theme->set_icon("tick", "HSlider", icons["hslider_tick"]);

	// VSlider

	theme->set_stylebox("slider", "VSlider", style_slider);
	theme->set_stylebox("grabber_area", "VSlider", style_slider_grabber);
	theme->set_stylebox("grabber_area_highlight", "VSlider", style_slider_grabber_highlight);

	theme->set_icon("grabber", "VSlider", icons["slider_grabber"]);
	theme->set_icon("grabber_highlight", "VSlider", icons["slider_grabber_hl"]);
	theme->set_icon("grabber_disabled", "VSlider", icons["slider_grabber_disabled"]);
	theme->set_icon("tick", "VSlider", icons["vslider_tick"]);

	// SpinBox

	theme->set_icon("updown", "SpinBox", icons["updown"]);

	// ScrollContainer

	Ref<StyleBoxEmpty> empty;
	empty.instantiate();
	theme->set_stylebox("bg", "ScrollContainer", empty);

	// Window

	theme->set_stylebox("embedded_border", "Window", sb_expand(make_flat_stylebox(style_popup_color, 10, 28, 10, 8), 8, 32, 8, 6));
	theme->set_constant("scaleborder_size", "Window", 4 * scale);

	theme->set_font("title_font", "Window", Ref<Font>());
	theme->set_font_size("title_font_size", "Window", -1);
	theme->set_color("title_color", "Window", control_font_color);
	theme->set_color("title_outline_modulate", "Window", Color(1, 1, 1));
	theme->set_constant("title_outline_size", "Window", 0);
	theme->set_constant("title_height", "Window", 36 * scale);
	theme->set_constant("resize_margin", "Window", 4 * scale);

	theme->set_icon("close", "Window", icons["close"]);
	theme->set_icon("close_pressed", "Window", icons["close_hl"]);
	theme->set_constant("close_h_ofs", "Window", 18 * scale);
	theme->set_constant("close_v_ofs", "Window", 24 * scale);

	// Dialogs

	theme->set_constant("margin", "Dialogs", 8 * scale);
	theme->set_constant("button_margin", "Dialogs", 32 * scale);

	// AcceptDialog

	theme->set_stylebox("panel", "AcceptDialog", make_flat_stylebox(style_popup_color, 0, 0, 0, 0));

	// File Dialog

	theme->set_icon("parent_folder", "FileDialog", icons["folder_up"]);
	theme->set_icon("back_folder", "FileDialog", icons["arrow_left"]);
	theme->set_icon("forward_folder", "FileDialog", icons["arrow_right"]);
	theme->set_icon("reload", "FileDialog", icons["reload"]);
	theme->set_icon("toggle_hidden", "FileDialog", icons["visibility_visible"]);
	theme->set_icon("folder", "FileDialog", icons["folder"]);
	theme->set_icon("file", "FileDialog", icons["file"]);
	theme->set_color("folder_icon_modulate", "FileDialog", Color(1, 1, 1));
	theme->set_color("file_icon_modulate", "FileDialog", Color(1, 1, 1));
	theme->set_color("files_disabled", "FileDialog", Color(0, 0, 0, 0.7));

	// Popup

	theme->set_stylebox("panel", "PopupPanel", make_flat_stylebox(style_normal_color));

	// PopupDialog

	theme->set_stylebox("panel", "PopupDialog", make_flat_stylebox(style_normal_color));

	// PopupMenu

	Ref<StyleBoxLine> separator_horizontal = memnew(StyleBoxLine);
	separator_horizontal->set_thickness(Math::round(scale));
	separator_horizontal->set_color(style_separator_color);
	separator_horizontal->set_default_margin(SIDE_LEFT, default_margin);
	separator_horizontal->set_default_margin(SIDE_TOP, 0);
	separator_horizontal->set_default_margin(SIDE_RIGHT, default_margin);
	separator_horizontal->set_default_margin(SIDE_BOTTOM, 0);
	Ref<StyleBoxLine> separator_vertical = separator_horizontal->duplicate();
	separator_vertical->set_vertical(true);
	separator_vertical->set_default_margin(SIDE_LEFT, 0);
	separator_vertical->set_default_margin(SIDE_TOP, default_margin);
	separator_vertical->set_default_margin(SIDE_RIGHT, 0);
	separator_vertical->set_default_margin(SIDE_BOTTOM, default_margin);

	// Always display a border for PopupMenus so they can be distinguished from their background.
	Ref<StyleBoxFlat> style_popup_panel = make_flat_stylebox(style_popup_color);
	style_popup_panel->set_border_width_all(2);
	style_popup_panel->set_border_color(style_popup_border_color);
	Ref<StyleBoxFlat> style_popup_panel_disabled = style_popup_panel->duplicate();
	style_popup_panel_disabled->set_bg_color(style_disabled_color);

	theme->set_stylebox("panel", "PopupMenu", style_popup_panel);
	theme->set_stylebox("panel_disabled", "PopupMenu", style_popup_panel_disabled);
	theme->set_stylebox("hover", "PopupMenu", make_flat_stylebox(style_popup_hover_color));
	theme->set_stylebox("separator", "PopupMenu", separator_horizontal);
	theme->set_stylebox("labeled_separator_left", "PopupMenu", separator_horizontal);
	theme->set_stylebox("labeled_separator_right", "PopupMenu", separator_horizontal);

	theme->set_icon("checked", "PopupMenu", icons["checked"]);
	theme->set_icon("unchecked", "PopupMenu", icons["unchecked"]);
	theme->set_icon("radio_checked", "PopupMenu", icons["radio_checked"]);
	theme->set_icon("radio_unchecked", "PopupMenu", icons["radio_unchecked"]);
	theme->set_icon("submenu", "PopupMenu", icons["arrow_right"]);
	theme->set_icon("submenu_mirrored", "PopupMenu", icons["arrow_left"]);

	theme->set_font("font", "PopupMenu", Ref<Font>());
	theme->set_font_size("font_size", "PopupMenu", -1);

	theme->set_color("font_color", "PopupMenu", control_font_color);
	theme->set_color("font_accelerator_color", "PopupMenu", Color(0.7, 0.7, 0.7, 0.8));
	theme->set_color("font_disabled_color", "PopupMenu", Color(0.4, 0.4, 0.4, 0.8));
	theme->set_color("font_hover_color", "PopupMenu", control_font_color);
	theme->set_color("font_separator_color", "PopupMenu", control_font_color);
	theme->set_color("font_outline_color", "PopupMenu", Color(1, 1, 1));

	theme->set_constant("hseparation", "PopupMenu", 4 * scale);
	theme->set_constant("vseparation", "PopupMenu", 4 * scale);
	theme->set_constant("outline_size", "PopupMenu", 0);
	theme->set_constant("item_start_padding", "PopupMenu", 2 * scale);
	theme->set_constant("item_end_padding", "PopupMenu", 2 * scale);

	// GraphNode
	Ref<StyleBoxFlat> graphnode_normal = make_flat_stylebox(style_normal_color, 18, 42, 18, 12);
	graphnode_normal->set_border_width(SIDE_TOP, 30);
	graphnode_normal->set_border_color(Color(0.325, 0.325, 0.325, 0.6));
	Ref<StyleBoxFlat> graphnode_selected = graphnode_normal->duplicate();
	graphnode_selected->set_border_color(Color(0.625, 0.625, 0.625, 0.6));
	Ref<StyleBoxFlat> graphnode_comment_normal = make_flat_stylebox(style_pressed_color, 18, 42, 18, 12, 3, true, 2);
	graphnode_comment_normal->set_border_color(style_pressed_color);
	Ref<StyleBoxFlat> graphnode_comment_selected = graphnode_comment_normal->duplicate();
	graphnode_comment_selected->set_border_color(style_hover_color);
	Ref<StyleBoxFlat> graphnode_breakpoint = make_flat_stylebox(style_pressed_color, 18, 42, 18, 12, 6, true, 4);
	graphnode_breakpoint->set_border_color(Color(0.9, 0.29, 0.3));
	Ref<StyleBoxFlat> graphnode_position = make_flat_stylebox(style_pressed_color, 18, 42, 18, 12, 6, true, 4);
	graphnode_position->set_border_color(Color(0.98, 0.89, 0.27));

	theme->set_stylebox("frame", "GraphNode", graphnode_normal);
	theme->set_stylebox("selectedframe", "GraphNode", graphnode_selected);
	theme->set_stylebox("comment", "GraphNode", graphnode_comment_normal);
	theme->set_stylebox("commentfocus", "GraphNode", graphnode_comment_selected);
	theme->set_stylebox("breakpoint", "GraphNode", graphnode_breakpoint);
	theme->set_stylebox("position", "GraphNode", graphnode_position);

	theme->set_icon("port", "GraphNode", icons["graph_port"]);
	theme->set_icon("close", "GraphNode", icons["close"]);
	theme->set_icon("resizer", "GraphNode", icons["resizer_se"]);
	theme->set_font("title_font", "GraphNode", Ref<Font>());
	theme->set_color("title_color", "GraphNode", control_font_color);
	theme->set_color("close_color", "GraphNode", control_font_color);
	theme->set_color("resizer_color", "GraphNode", control_font_color);
	theme->set_constant("separation", "GraphNode", 2 * scale);
	theme->set_constant("title_offset", "GraphNode", 26 * scale);
	theme->set_constant("close_offset", "GraphNode", 22 * scale);
	theme->set_constant("port_offset", "GraphNode", 0);

	// Tree

	theme->set_stylebox("bg", "Tree", make_flat_stylebox(style_normal_color, 4, 4, 4, 5));
	theme->set_stylebox("bg_focus", "Tree", focus);
	theme->set_stylebox("selected", "Tree", make_flat_stylebox(style_selected_color));
	theme->set_stylebox("selected_focus", "Tree", make_flat_stylebox(style_selected_color));
	theme->set_stylebox("cursor", "Tree", focus);
	theme->set_stylebox("cursor_unfocused", "Tree", focus);
	theme->set_stylebox("button_pressed", "Tree", button_pressed);
	theme->set_stylebox("title_button_normal", "Tree", make_flat_stylebox(style_pressed_color, 4, 4, 4, 4));
	theme->set_stylebox("title_button_pressed", "Tree", make_flat_stylebox(style_hover_color, 4, 4, 4, 4));
	theme->set_stylebox("title_button_hover", "Tree", make_flat_stylebox(style_normal_color, 4, 4, 4, 4));
	theme->set_stylebox("custom_button", "Tree", button_normal);
	theme->set_stylebox("custom_button_pressed", "Tree", button_pressed);
	theme->set_stylebox("custom_button_hover", "Tree", button_hover);

	theme->set_icon("checked", "Tree", icons["checked"]);
	theme->set_icon("unchecked", "Tree", icons["unchecked"]);
	theme->set_icon("indeterminate", "Tree", icons["indeterminate"]);
	theme->set_icon("updown", "Tree", icons["updown"]);
	theme->set_icon("select_arrow", "Tree", icons["option_button_arrow"]);
	theme->set_icon("arrow", "Tree", icons["arrow_down"]);
	theme->set_icon("arrow_collapsed", "Tree", icons["arrow_right"]);
	theme->set_icon("arrow_collapsed_mirrored", "Tree", icons["arrow_left"]);

	theme->set_font("title_button_font", "Tree", Ref<Font>());
	theme->set_font("font", "Tree", Ref<Font>());
	theme->set_font_size("font_size", "Tree", -1);

	theme->set_color("title_button_color", "Tree", control_font_color);
	theme->set_color("font_color", "Tree", control_font_low_color);
	theme->set_color("font_selected_color", "Tree", control_font_pressed_color);
	theme->set_color("font_outline_color", "Tree", Color(1, 1, 1));
	theme->set_color("guide_color", "Tree", Color(0.7, 0.7, 0.7, 0.25));
	theme->set_color("drop_position_color", "Tree", Color(1, 0.3, 0.2));
	theme->set_color("relationship_line_color", "Tree", Color(0.27, 0.27, 0.27));
	theme->set_color("parent_hl_line_color", "Tree", Color(0.27, 0.27, 0.27));
	theme->set_color("children_hl_line_color", "Tree", Color(0.27, 0.27, 0.27));
	theme->set_color("custom_button_font_highlight", "Tree", control_font_hover_color);

	theme->set_constant("hseparation", "Tree", 4 * scale);
	theme->set_constant("vseparation", "Tree", 4 * scale);
	theme->set_constant("item_margin", "Tree", 16 * scale);
	theme->set_constant("button_margin", "Tree", 4 * scale);
	theme->set_constant("draw_relationship_lines", "Tree", 0);
	theme->set_constant("relationship_line_width", "Tree", 1);
	theme->set_constant("parent_hl_line_width", "Tree", 1);
	theme->set_constant("children_hl_line_width", "Tree", 1);
	theme->set_constant("parent_hl_line_margin", "Tree", 0);
	theme->set_constant("draw_guides", "Tree", 1);
	theme->set_constant("scroll_border", "Tree", 4);
	theme->set_constant("scroll_speed", "Tree", 12);
	theme->set_constant("outline_size", "Tree", 0);

	// ItemList

	theme->set_stylebox("bg", "ItemList", make_flat_stylebox(style_normal_color));
	theme->set_stylebox("bg_focus", "ItemList", focus);
	theme->set_constant("hseparation", "ItemList", 4);
	theme->set_constant("vseparation", "ItemList", 2);
	theme->set_constant("icon_margin", "ItemList", 4);
	theme->set_constant("line_separation", "ItemList", 2 * scale);

	theme->set_font("font", "ItemList", Ref<Font>());
	theme->set_font_size("font_size", "ItemList", -1);

	theme->set_color("font_color", "ItemList", control_font_lower_color);
	theme->set_color("font_selected_color", "ItemList", control_font_pressed_color);
	theme->set_color("font_outline_color", "ItemList", Color(1, 1, 1));
	theme->set_color("guide_color", "ItemList", Color(0, 0, 0, 0.1));
	theme->set_stylebox("selected", "ItemList", make_flat_stylebox(style_selected_color));
	theme->set_stylebox("selected_focus", "ItemList", make_flat_stylebox(style_selected_color));
	theme->set_stylebox("cursor", "ItemList", focus);
	theme->set_stylebox("cursor_unfocused", "ItemList", focus);

	theme->set_constant("outline_size", "ItemList", 0);

	// TabContainer

	Ref<StyleBoxFlat> style_tab_selected = make_flat_stylebox(style_normal_color, 10, 4, 10, 4, 0);
	style_tab_selected->set_border_width(SIDE_TOP, Math::round(2 * scale));
	style_tab_selected->set_border_color(style_focus_color);
	Ref<StyleBoxFlat> style_tab_unselected = make_flat_stylebox(style_pressed_color, 10, 4, 10, 4, 0);
	// Add some spacing between unselected tabs to make them easier to distinguish from each other.
	style_tab_unselected->set_border_width(SIDE_LEFT, Math::round(scale));
	style_tab_unselected->set_border_width(SIDE_RIGHT, Math::round(scale));
	style_tab_unselected->set_border_color(style_popup_border_color);
	Ref<StyleBoxFlat> style_tab_disabled = style_tab_unselected->duplicate();
	style_tab_disabled->set_bg_color(style_disabled_color);

	theme->set_stylebox("tab_selected", "TabContainer", style_tab_selected);
	theme->set_stylebox("tab_unselected", "TabContainer", style_tab_unselected);
	theme->set_stylebox("tab_disabled", "TabContainer", style_tab_disabled);
	theme->set_stylebox("panel", "TabContainer", make_flat_stylebox(style_normal_color, 0, 0, 0, 0));

	theme->set_icon("increment", "TabContainer", icons["scroll_button_right"]);
	theme->set_icon("increment_highlight", "TabContainer", icons["scroll_button_right_hl"]);
	theme->set_icon("decrement", "TabContainer", icons["scroll_button_left"]);
	theme->set_icon("decrement_highlight", "TabContainer", icons["scroll_button_left_hl"]);
	theme->set_icon("menu", "TabContainer", icons["tabs_menu"]);
	theme->set_icon("menu_highlight", "TabContainer", icons["tabs_menu_hl"]);

	theme->set_font("font", "TabContainer", Ref<Font>());
	theme->set_font_size("font_size", "TabContainer", -1);

	theme->set_color("font_selected_color", "TabContainer", control_font_hover_color);
	theme->set_color("font_unselected_color", "TabContainer", control_font_low_color);
	theme->set_color("font_disabled_color", "TabContainer", control_font_disabled_color);
	theme->set_color("font_outline_color", "TabContainer", Color(1, 1, 1));

	theme->set_constant("side_margin", "TabContainer", 8 * scale);
	theme->set_constant("icon_separation", "TabContainer", 4 * scale);
	theme->set_constant("outline_size", "TabContainer", 0);

	// TabBar

	theme->set_stylebox("tab_selected", "TabBar", style_tab_selected);
	theme->set_stylebox("tab_unselected", "TabBar", style_tab_unselected);
	theme->set_stylebox("tab_disabled", "TabBar", style_tab_disabled);
	theme->set_stylebox("close_bg_pressed", "TabBar", button_pressed);
	theme->set_stylebox("close_bg_highlight", "TabBar", button_normal);

	theme->set_icon("increment", "TabBar", icons["scroll_button_right"]);
	theme->set_icon("increment_highlight", "TabBar", icons["scroll_button_right_hl"]);
	theme->set_icon("decrement", "TabBar", icons["scroll_button_left"]);
	theme->set_icon("decrement_highlight", "TabBar", icons["scroll_button_left_hl"]);
	theme->set_icon("close", "TabBar", icons["close"]);

	theme->set_font("font", "TabBar", Ref<Font>());
	theme->set_font_size("font_size", "TabBar", -1);

	theme->set_color("font_selected_color", "TabBar", control_font_hover_color);
	theme->set_color("font_unselected_color", "TabBar", control_font_low_color);
	theme->set_color("font_disabled_color", "TabBar", control_font_disabled_color);
	theme->set_color("font_outline_color", "TabBar", Color(1, 1, 1));

	theme->set_constant("hseparation", "TabBar", 4 * scale);
	theme->set_constant("outline_size", "TabBar", 0);

	// Separators

	theme->set_stylebox("separator", "HSeparator", separator_horizontal);
	theme->set_stylebox("separator", "VSeparator", separator_vertical);

	theme->set_icon("close", "Icons", icons["close"]);
	theme->set_font("normal", "Fonts", Ref<Font>());
	theme->set_font("large", "Fonts", Ref<Font>());

	theme->set_constant("separation", "HSeparator", 4 * scale);
	theme->set_constant("separation", "VSeparator", 4 * scale);

	// ColorPicker

	theme->set_constant("margin", "ColorPicker", 4 * scale);
	theme->set_constant("sv_width", "ColorPicker", 256 * scale);
	theme->set_constant("sv_height", "ColorPicker", 256 * scale);
	theme->set_constant("h_width", "ColorPicker", 30 * scale);
	theme->set_constant("label_width", "ColorPicker", 10 * scale);

	theme->set_icon("screen_picker", "ColorPicker", icons["color_picker_pipette"]);
	theme->set_icon("add_preset", "ColorPicker", icons["add"]);
	theme->set_icon("color_hue", "ColorPicker", icons["color_picker_hue"]);
	theme->set_icon("color_sample", "ColorPicker", icons["color_picker_sample"]);
	theme->set_icon("sample_bg", "ColorPicker", icons["mini_checkerboard"]);
	theme->set_icon("overbright_indicator", "ColorPicker", icons["color_picker_overbright"]);
	theme->set_icon("bar_arrow", "ColorPicker", icons["color_picker_bar_arrow"]);
	theme->set_icon("picker_cursor", "ColorPicker", icons["color_picker_cursor"]);

	// ColorPickerButton

	theme->set_icon("bg", "ColorPickerButton", icons["mini_checkerboard"]);
	theme->set_stylebox("normal", "ColorPickerButton", button_normal);
	theme->set_stylebox("pressed", "ColorPickerButton", button_pressed);
	theme->set_stylebox("hover", "ColorPickerButton", button_hover);
	theme->set_stylebox("disabled", "ColorPickerButton", button_disabled);
	theme->set_stylebox("focus", "ColorPickerButton", focus);

	theme->set_font("font", "ColorPickerButton", Ref<Font>());
	theme->set_font_size("font_size", "ColorPickerButton", -1);

	theme->set_color("font_color", "ColorPickerButton", Color(1, 1, 1, 1));
	theme->set_color("font_pressed_color", "ColorPickerButton", Color(0.8, 0.8, 0.8, 1));
	theme->set_color("font_hover_color", "ColorPickerButton", Color(1, 1, 1, 1));
	theme->set_color("font_focus_color", "ColorPickerButton", Color(1, 1, 1, 1));
	theme->set_color("font_disabled_color", "ColorPickerButton", Color(0.9, 0.9, 0.9, 0.3));
	theme->set_color("font_outline_color", "ColorPickerButton", Color(1, 1, 1));

	theme->set_constant("hseparation", "ColorPickerButton", 2 * scale);
	theme->set_constant("outline_size", "ColorPickerButton", 0);

	// ColorPresetButton

	Ref<StyleBoxFlat> preset_sb = make_flat_stylebox(Color(1, 1, 1), 2, 2, 2, 2);
	preset_sb->set_corner_radius_all(2);
	preset_sb->set_corner_detail(2);
	preset_sb->set_anti_aliased(false);

	theme->set_stylebox("preset_fg", "ColorPresetButton", preset_sb);
	theme->set_icon("preset_bg", "ColorPresetButton", icons["mini_checkerboard"]);
	theme->set_icon("overbright_indicator", "ColorPresetButton", icons["color_picker_overbright"]);

	// TooltipPanel + TooltipLabel

	theme->set_stylebox("panel", "TooltipPanel",
			make_flat_stylebox(Color(0, 0, 0, 0.5), 2 * default_margin, 0.5 * default_margin, 2 * default_margin, 0.5 * default_margin));

	theme->set_font("font", "TooltipLabel", Ref<Font>());
	theme->set_font_size("font_size", "TooltipLabel", -1);

	theme->set_color("font_color", "TooltipLabel", control_font_color);
	theme->set_color("font_shadow_color", "TooltipLabel", Color(0, 0, 0, 0));
	theme->set_color("font_outline_color", "TooltipLabel", Color(0, 0, 0, 0));

	theme->set_constant("shadow_offset_x", "TooltipLabel", 1);
	theme->set_constant("shadow_offset_y", "TooltipLabel", 1);
	theme->set_constant("outline_size", "TooltipLabel", 0);

	// RichTextLabel

	theme->set_stylebox("focus", "RichTextLabel", focus);
	theme->set_stylebox("normal", "RichTextLabel", make_empty_stylebox(0, 0, 0, 0));

	theme->set_font("normal_font", "RichTextLabel", Ref<Font>());
	theme->set_font("bold_font", "RichTextLabel", Ref<Font>());
	theme->set_font("italics_font", "RichTextLabel", Ref<Font>());
	theme->set_font("bold_italics_font", "RichTextLabel", Ref<Font>());
	theme->set_font("mono_font", "RichTextLabel", Ref<Font>());

	theme->set_font_size("normal_font_size", "RichTextLabel", -1);
	theme->set_font_size("bold_font_size", "RichTextLabel", -1);
	theme->set_font_size("italics_font_size", "RichTextLabel", -1);
	theme->set_font_size("bold_italics_font_size", "RichTextLabel", -1);
	theme->set_font_size("mono_font_size", "RichTextLabel", -1);

	theme->set_color("default_color", "RichTextLabel", Color(1, 1, 1));
	theme->set_color("font_selected_color", "RichTextLabel", Color(0, 0, 0));
	theme->set_color("selection_color", "RichTextLabel", Color(0.1, 0.1, 1, 0.8));

	theme->set_color("font_shadow_color", "RichTextLabel", Color(0, 0, 0, 0));

	theme->set_color("font_outline_color", "RichTextLabel", Color(1, 1, 1));

	theme->set_constant("shadow_offset_x", "RichTextLabel", 1 * scale);
	theme->set_constant("shadow_offset_y", "RichTextLabel", 1 * scale);
	theme->set_constant("shadow_outline_size", "RichTextLabel", 1 * scale);

	theme->set_constant("line_separation", "RichTextLabel", 0 * scale);
	theme->set_constant("table_hseparation", "RichTextLabel", 3 * scale);
	theme->set_constant("table_vseparation", "RichTextLabel", 3 * scale);

	theme->set_constant("outline_size", "RichTextLabel", 0);

	theme->set_color("table_odd_row_bg", "RichTextLabel", Color(0, 0, 0, 0));
	theme->set_color("table_even_row_bg", "RichTextLabel", Color(0, 0, 0, 0));
	theme->set_color("table_border", "RichTextLabel", Color(0, 0, 0, 0));

	// Containers

	theme->set_icon("grabber", "VSplitContainer", icons["vsplitter"]);
	theme->set_icon("grabber", "HSplitContainer", icons["hsplitter"]);

	theme->set_constant("separation", "HBoxContainer", 4 * scale);
	theme->set_constant("separation", "VBoxContainer", 4 * scale);
	theme->set_constant("margin_left", "MarginContainer", 0 * scale);
	theme->set_constant("margin_top", "MarginContainer", 0 * scale);
	theme->set_constant("margin_right", "MarginContainer", 0 * scale);
	theme->set_constant("margin_bottom", "MarginContainer", 0 * scale);
	theme->set_constant("hseparation", "GridContainer", 4 * scale);
	theme->set_constant("vseparation", "GridContainer", 4 * scale);
	theme->set_constant("separation", "HSplitContainer", 12 * scale);
	theme->set_constant("separation", "VSplitContainer", 12 * scale);
	theme->set_constant("autohide", "HSplitContainer", 1 * scale);
	theme->set_constant("autohide", "VSplitContainer", 1 * scale);
	theme->set_constant("hseparation", "HFlowContainer", 4 * scale);
	theme->set_constant("vseparation", "HFlowContainer", 4 * scale);
	theme->set_constant("hseparation", "VFlowContainer", 4 * scale);
	theme->set_constant("vseparation", "VFlowContainer", 4 * scale);

	theme->set_stylebox("panel", "PanelContainer", make_flat_stylebox(style_normal_color, 0, 0, 0, 0));

	theme->set_icon("minus", "GraphEdit", icons["zoom_less"]);
	theme->set_icon("reset", "GraphEdit", icons["zoom_reset"]);
	theme->set_icon("more", "GraphEdit", icons["zoom_more"]);
	theme->set_icon("snap", "GraphEdit", icons["grid_snap"]);
	theme->set_icon("minimap", "GraphEdit", icons["grid_minimap"]);
	theme->set_icon("layout", "GraphEdit", icons["grid_layout"]);
	theme->set_stylebox("bg", "GraphEdit", make_flat_stylebox(style_normal_color, 4, 4, 4, 5));
	theme->set_color("grid_minor", "GraphEdit", Color(1, 1, 1, 0.05));
	theme->set_color("grid_major", "GraphEdit", Color(1, 1, 1, 0.2));
	theme->set_color("selection_fill", "GraphEdit", Color(1, 1, 1, 0.3));
	theme->set_color("selection_stroke", "GraphEdit", Color(1, 1, 1, 0.8));
	theme->set_color("activity", "GraphEdit", Color(1, 1, 1));
	theme->set_constant("bezier_len_pos", "GraphEdit", 80 * scale);
	theme->set_constant("bezier_len_neg", "GraphEdit", 160 * scale);

	// Visual Node Ports

	theme->set_constant("port_grab_distance_horizontal", "GraphEdit", 24 * scale);
	theme->set_constant("port_grab_distance_vertical", "GraphEdit", 26 * scale);

	theme->set_stylebox("bg", "GraphEditMinimap", make_flat_stylebox(Color(0.24, 0.24, 0.24), 0, 0, 0, 0));
	Ref<StyleBoxFlat> style_minimap_camera = make_flat_stylebox(Color(0.65, 0.65, 0.65, 0.2), 0, 0, 0, 0, 0);
	style_minimap_camera->set_border_color(Color(0.65, 0.65, 0.65, 0.45));
	style_minimap_camera->set_border_width_all(1);
	theme->set_stylebox("camera", "GraphEditMinimap", style_minimap_camera);
	theme->set_stylebox("node", "GraphEditMinimap", make_flat_stylebox(Color(1, 1, 1), 0, 0, 0, 0, 2));

	theme->set_icon("resizer", "GraphEditMinimap", icons["resizer_nw"]);
	theme->set_color("resizer_color", "GraphEditMinimap", Color(1, 1, 1, 0.85));

	// Theme

	default_icon = icons["error_icon"];
	// Same color as the error icon.
	default_style = make_flat_stylebox(Color(1, 0.365, 0.365), 4, 4, 4, 4, 0, false, 2);
}

void make_default_theme(float p_scale, Ref<Font> p_font) {
	Ref<Theme> t;
	t.instantiate();

	Ref<StyleBox> default_style;
	Ref<Texture2D> default_icon;
	Ref<Font> default_font;
	float default_scale = CLAMP(p_scale, 0.5, 8.0);

	if (p_font.is_valid()) {
		// Use the custom font defined in the Project Settings.
		default_font = p_font;
	} else {
		// Use the default DynamicFont (separate from the editor font).
		// The default DynamicFont is chosen to have a small file size since it's
		// embedded in both editor and export template binaries.
		Ref<Font> dynamic_font;
		dynamic_font.instantiate();

		Ref<FontData> dynamic_font_data;
		dynamic_font_data.instantiate();
		dynamic_font_data->set_data_ptr(_font_OpenSans_SemiBold, _font_OpenSans_SemiBold_size);
		dynamic_font->add_data(dynamic_font_data);

		default_font = dynamic_font;
	}

	fill_default_theme(t, default_font, default_icon, default_style, default_scale);

	Theme::set_default(t);
	Theme::set_fallback_base_scale(default_scale);
	Theme::set_fallback_icon(default_icon);
	Theme::set_fallback_style(default_style);
	Theme::set_fallback_font(default_font);
	Theme::set_fallback_font_size(default_font_size * default_scale);
}

void clear_default_theme() {
	Theme::set_project_default(nullptr);
	Theme::set_default(nullptr);
	Theme::set_fallback_icon(nullptr);
	Theme::set_fallback_style(nullptr);
	Theme::set_fallback_font(nullptr);
}
