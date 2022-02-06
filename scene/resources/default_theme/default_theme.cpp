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
	const Color control_font_placeholder_color = Color(control_font_color.r, control_font_color.g, control_font_color.b, 0.6f);
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
	theme->set_stylebox(SNAME("panel"), SNAME("Panel"), make_flat_stylebox(style_normal_color, 0, 0, 0, 0));
	theme->set_stylebox(SNAME("panel_fg"), SNAME("Panel"), make_flat_stylebox(style_normal_color, 0, 0, 0, 0));

	// Button

	const Ref<StyleBoxFlat> button_normal = make_flat_stylebox(style_normal_color);
	const Ref<StyleBoxFlat> button_hover = make_flat_stylebox(style_hover_color);
	const Ref<StyleBoxFlat> button_pressed = make_flat_stylebox(style_pressed_color);
	const Ref<StyleBoxFlat> button_disabled = make_flat_stylebox(style_disabled_color);
	Ref<StyleBoxFlat> focus = make_flat_stylebox(style_focus_color, default_margin, default_margin, default_margin, default_margin, default_corner_radius, false, 2);
	// Make the focus outline appear to be flush with the buttons it's focusing.
	focus->set_expand_margin_size_all(2 * scale);

	theme->set_stylebox(SNAME("normal"), SNAME("Button"), button_normal);
	theme->set_stylebox(SNAME("hover"), SNAME("Button"), button_hover);
	theme->set_stylebox(SNAME("pressed"), SNAME("Button"), button_pressed);
	theme->set_stylebox(SNAME("disabled"), SNAME("Button"), button_disabled);
	theme->set_stylebox(SNAME("focus"), SNAME("Button"), focus);

	theme->set_font(SNAME("font"), SNAME("Button"), Ref<Font>());
	theme->set_font_size(SNAME("font_size"), SNAME("Button"), -1);
	theme->set_constant(SNAME("outline_size"), SNAME("Button"), 0 * scale);

	theme->set_color(SNAME("font_color"), SNAME("Button"), control_font_color);
	theme->set_color(SNAME("font_pressed_color"), SNAME("Button"), control_font_pressed_color);
	theme->set_color(SNAME("font_hover_color"), SNAME("Button"), control_font_hover_color);
	theme->set_color(SNAME("font_focus_color"), SNAME("Button"), control_font_focus_color);
	theme->set_color(SNAME("font_hover_pressed_color"), SNAME("Button"), control_font_pressed_color);
	theme->set_color(SNAME("font_disabled_color"), SNAME("Button"), control_font_disabled_color);
	theme->set_color(SNAME("font_outline_color"), SNAME("Button"), Color(1, 1, 1));

	theme->set_color(SNAME("icon_normal_color"), SNAME("Button"), Color(1, 1, 1, 1));
	theme->set_color(SNAME("icon_pressed_color"), SNAME("Button"), Color(1, 1, 1, 1));
	theme->set_color(SNAME("icon_hover_color"), SNAME("Button"), Color(1, 1, 1, 1));
	theme->set_color(SNAME("icon_hover_pressed_color"), SNAME("Button"), Color(1, 1, 1, 1));
	theme->set_color(SNAME("icon_focus_color"), SNAME("Button"), Color(1, 1, 1, 1));
	theme->set_color(SNAME("icon_disabled_color"), SNAME("Button"), Color(1, 1, 1, 1));

	theme->set_constant(SNAME("hseparation"), SNAME("Button"), 2 * scale);

	// LinkButton

	theme->set_stylebox(SNAME("focus"), SNAME("LinkButton"), focus);

	theme->set_font(SNAME("font"), SNAME("LinkButton"), Ref<Font>());
	theme->set_font_size(SNAME("font_size"), SNAME("LinkButton"), -1);

	theme->set_color(SNAME("font_color"), SNAME("LinkButton"), control_font_color);
	theme->set_color(SNAME("font_pressed_color"), SNAME("LinkButton"), control_font_pressed_color);
	theme->set_color(SNAME("font_hover_color"), SNAME("LinkButton"), control_font_hover_color);
	theme->set_color(SNAME("font_focus_color"), SNAME("LinkButton"), control_font_focus_color);
	theme->set_color(SNAME("font_outline_color"), SNAME("LinkButton"), Color(1, 1, 1));

	theme->set_constant(SNAME("outline_size"), SNAME("LinkButton"), 0);
	theme->set_constant(SNAME("underline_spacing"), SNAME("LinkButton"), 2 * scale);

	// OptionButton
	theme->set_stylebox(SNAME("focus"), SNAME("OptionButton"), focus);

	Ref<StyleBox> sb_optbutton_normal = make_flat_stylebox(style_normal_color, 2 * default_margin, default_margin, 21, default_margin);
	Ref<StyleBox> sb_optbutton_hover = make_flat_stylebox(style_hover_color, 2 * default_margin, default_margin, 21, default_margin);
	Ref<StyleBox> sb_optbutton_pressed = make_flat_stylebox(style_pressed_color, 2 * default_margin, default_margin, 21, default_margin);
	Ref<StyleBox> sb_optbutton_disabled = make_flat_stylebox(style_disabled_color, 2 * default_margin, default_margin, 21, default_margin);

	theme->set_stylebox(SNAME("normal"), SNAME("OptionButton"), sb_optbutton_normal);
	theme->set_stylebox(SNAME("hover"), SNAME("OptionButton"), sb_optbutton_hover);
	theme->set_stylebox(SNAME("pressed"), SNAME("OptionButton"), sb_optbutton_pressed);
	theme->set_stylebox(SNAME("disabled"), SNAME("OptionButton"), sb_optbutton_disabled);

	Ref<StyleBox> sb_optbutton_normal_mirrored = make_flat_stylebox(style_normal_color, 21, default_margin, 2 * default_margin, default_margin);
	Ref<StyleBox> sb_optbutton_hover_mirrored = make_flat_stylebox(style_hover_color, 21, default_margin, 2 * default_margin, default_margin);
	Ref<StyleBox> sb_optbutton_pressed_mirrored = make_flat_stylebox(style_pressed_color, 21, default_margin, 2 * default_margin, default_margin);
	Ref<StyleBox> sb_optbutton_disabled_mirrored = make_flat_stylebox(style_disabled_color, 21, default_margin, 2 * default_margin, default_margin);

	theme->set_stylebox(SNAME("normal_mirrored"), SNAME("OptionButton"), sb_optbutton_normal_mirrored);
	theme->set_stylebox(SNAME("hover_mirrored"), SNAME("OptionButton"), sb_optbutton_hover_mirrored);
	theme->set_stylebox(SNAME("pressed_mirrored"), SNAME("OptionButton"), sb_optbutton_pressed_mirrored);
	theme->set_stylebox(SNAME("disabled_mirrored"), SNAME("OptionButton"), sb_optbutton_disabled_mirrored);

	theme->set_icon(SNAME("arrow"), SNAME("OptionButton"), icons["option_button_arrow"]);

	theme->set_font(SNAME("font"), SNAME("OptionButton"), Ref<Font>());
	theme->set_font_size(SNAME("font_size"), SNAME("OptionButton"), -1);

	theme->set_color(SNAME("font_color"), SNAME("OptionButton"), control_font_color);
	theme->set_color(SNAME("font_pressed_color"), SNAME("OptionButton"), control_font_pressed_color);
	theme->set_color(SNAME("font_hover_color"), SNAME("OptionButton"), control_font_hover_color);
	theme->set_color(SNAME("font_focus_color"), SNAME("OptionButton"), control_font_focus_color);
	theme->set_color(SNAME("font_disabled_color"), SNAME("OptionButton"), control_font_disabled_color);
	theme->set_color(SNAME("font_outline_color"), SNAME("OptionButton"), Color(1, 1, 1));

	theme->set_constant(SNAME("hseparation"), SNAME("OptionButton"), 2 * scale);
	theme->set_constant(SNAME("arrow_margin"), SNAME("OptionButton"), 4 * scale);
	theme->set_constant(SNAME("outline_size"), SNAME("OptionButton"), 0);

	// MenuButton

	theme->set_stylebox(SNAME("normal"), SNAME("MenuButton"), button_normal);
	theme->set_stylebox(SNAME("pressed"), SNAME("MenuButton"), button_pressed);
	theme->set_stylebox(SNAME("hover"), SNAME("MenuButton"), button_hover);
	theme->set_stylebox(SNAME("disabled"), SNAME("MenuButton"), button_disabled);
	theme->set_stylebox(SNAME("focus"), SNAME("MenuButton"), focus);

	theme->set_font(SNAME("font"), SNAME("MenuButton"), Ref<Font>());
	theme->set_font_size(SNAME("font_size"), SNAME("MenuButton"), -1);

	theme->set_color(SNAME("font_color"), SNAME("MenuButton"), control_font_color);
	theme->set_color(SNAME("font_pressed_color"), SNAME("MenuButton"), control_font_pressed_color);
	theme->set_color(SNAME("font_hover_color"), SNAME("MenuButton"), control_font_hover_color);
	theme->set_color(SNAME("font_focus_color"), SNAME("MenuButton"), control_font_focus_color);
	theme->set_color(SNAME("font_disabled_color"), SNAME("MenuButton"), Color(1, 1, 1, 0.3));
	theme->set_color(SNAME("font_outline_color"), SNAME("MenuButton"), Color(1, 1, 1));

	theme->set_constant(SNAME("hseparation"), SNAME("MenuButton"), 3 * scale);
	theme->set_constant(SNAME("outline_size"), SNAME("MenuButton"), 0);

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

	theme->set_stylebox(SNAME("normal"), SNAME("CheckBox"), cbx_empty);
	theme->set_stylebox(SNAME("pressed"), SNAME("CheckBox"), cbx_empty);
	theme->set_stylebox(SNAME("disabled"), SNAME("CheckBox"), cbx_empty);
	theme->set_stylebox(SNAME("hover"), SNAME("CheckBox"), cbx_empty);
	theme->set_stylebox(SNAME("hover_pressed"), SNAME("CheckBox"), cbx_empty);
	theme->set_stylebox(SNAME("focus"), SNAME("CheckBox"), cbx_focus);

	theme->set_icon(SNAME("checked"), SNAME("CheckBox"), icons["checked"]);
	theme->set_icon(SNAME("checked_disabled"), SNAME("CheckBox"), icons["checked"]);
	theme->set_icon(SNAME("unchecked"), SNAME("CheckBox"), icons["unchecked"]);
	theme->set_icon(SNAME("unchecked_disabled"), SNAME("CheckBox"), icons["unchecked"]);
	theme->set_icon(SNAME("radio_checked"), SNAME("CheckBox"), icons["radio_checked"]);
	theme->set_icon(SNAME("radio_checked_disabled"), SNAME("CheckBox"), icons["radio_checked"]);
	theme->set_icon(SNAME("radio_unchecked"), SNAME("CheckBox"), icons["radio_unchecked"]);
	theme->set_icon(SNAME("radio_unchecked_disabled"), SNAME("CheckBox"), icons["radio_unchecked"]);

	theme->set_font(SNAME("font"), SNAME("CheckBox"), Ref<Font>());
	theme->set_font_size(SNAME("font_size"), SNAME("CheckBox"), -1);

	theme->set_color(SNAME("font_color"), SNAME("CheckBox"), control_font_color);
	theme->set_color(SNAME("font_pressed_color"), SNAME("CheckBox"), control_font_pressed_color);
	theme->set_color(SNAME("font_hover_color"), SNAME("CheckBox"), control_font_hover_color);
	theme->set_color(SNAME("font_hover_pressed_color"), SNAME("CheckBox"), control_font_pressed_color);
	theme->set_color(SNAME("font_focus_color"), SNAME("CheckBox"), control_font_focus_color);
	theme->set_color(SNAME("font_disabled_color"), SNAME("CheckBox"), control_font_disabled_color);
	theme->set_color(SNAME("font_outline_color"), SNAME("CheckBox"), Color(1, 1, 1));

	theme->set_constant(SNAME("hseparation"), SNAME("CheckBox"), 4 * scale);
	theme->set_constant(SNAME("check_vadjust"), SNAME("CheckBox"), 0 * scale);
	theme->set_constant(SNAME("outline_size"), SNAME("CheckBox"), 0);

	// CheckButton

	Ref<StyleBox> cb_empty = memnew(StyleBoxEmpty);
	cb_empty->set_default_margin(SIDE_LEFT, 6 * scale);
	cb_empty->set_default_margin(SIDE_RIGHT, 6 * scale);
	cb_empty->set_default_margin(SIDE_TOP, 4 * scale);
	cb_empty->set_default_margin(SIDE_BOTTOM, 4 * scale);

	theme->set_stylebox(SNAME("normal"), SNAME("CheckButton"), cb_empty);
	theme->set_stylebox(SNAME("pressed"), SNAME("CheckButton"), cb_empty);
	theme->set_stylebox(SNAME("disabled"), SNAME("CheckButton"), cb_empty);
	theme->set_stylebox(SNAME("hover"), SNAME("CheckButton"), cb_empty);
	theme->set_stylebox(SNAME("hover_pressed"), SNAME("CheckButton"), cb_empty);
	theme->set_stylebox(SNAME("focus"), SNAME("CheckButton"), focus);

	theme->set_icon(SNAME("on"), SNAME("CheckButton"), icons["toggle_on"]);
	theme->set_icon(SNAME("on_disabled"), SNAME("CheckButton"), icons["toggle_on_disabled"]);
	theme->set_icon(SNAME("off"), SNAME("CheckButton"), icons["toggle_off"]);
	theme->set_icon(SNAME("off_disabled"), SNAME("CheckButton"), icons["toggle_off_disabled"]);

	theme->set_icon(SNAME("on_mirrored"), SNAME("CheckButton"), icons["toggle_on_mirrored"]);
	theme->set_icon(SNAME("on_disabled_mirrored"), SNAME("CheckButton"), icons["toggle_on_disabled_mirrored"]);
	theme->set_icon(SNAME("off_mirrored"), SNAME("CheckButton"), icons["toggle_off_mirrored"]);
	theme->set_icon(SNAME("off_disabled_mirrored"), SNAME("CheckButton"), icons["toggle_off_disabled_mirrored"]);

	theme->set_font(SNAME("font"), SNAME("CheckButton"), Ref<Font>());
	theme->set_font_size(SNAME("font_size"), SNAME("CheckButton"), -1);

	theme->set_color(SNAME("font_color"), SNAME("CheckButton"), control_font_color);
	theme->set_color(SNAME("font_pressed_color"), SNAME("CheckButton"), control_font_pressed_color);
	theme->set_color(SNAME("font_hover_color"), SNAME("CheckButton"), control_font_hover_color);
	theme->set_color(SNAME("font_hover_pressed_color"), SNAME("CheckButton"), control_font_pressed_color);
	theme->set_color(SNAME("font_focus_color"), SNAME("CheckButton"), control_font_focus_color);
	theme->set_color(SNAME("font_disabled_color"), SNAME("CheckButton"), control_font_disabled_color);
	theme->set_color(SNAME("font_outline_color"), SNAME("CheckButton"), Color(1, 1, 1));

	theme->set_constant(SNAME("hseparation"), SNAME("CheckButton"), 4 * scale);
	theme->set_constant(SNAME("check_vadjust"), SNAME("CheckButton"), 0 * scale);
	theme->set_constant(SNAME("outline_size"), SNAME("CheckButton"), 0);

	// Label

	theme->set_stylebox(SNAME("normal"), SNAME("Label"), memnew(StyleBoxEmpty));
	theme->set_font(SNAME("font"), SNAME("Label"), Ref<Font>());
	theme->set_font_size(SNAME("font_size"), SNAME("Label"), -1);

	theme->set_color(SNAME("font_color"), SNAME("Label"), Color(1, 1, 1));
	theme->set_color(SNAME("font_shadow_color"), SNAME("Label"), Color(0, 0, 0, 0));
	theme->set_color(SNAME("font_outline_color"), SNAME("Label"), Color(1, 1, 1));

	theme->set_constant(SNAME("shadow_offset_x"), SNAME("Label"), 1 * scale);
	theme->set_constant(SNAME("shadow_offset_y"), SNAME("Label"), 1 * scale);
	theme->set_constant(SNAME("outline_size"), SNAME("Label"), 0);
	theme->set_constant(SNAME("shadow_outline_size"), SNAME("Label"), 1 * scale);
	theme->set_constant(SNAME("line_spacing"), SNAME("Label"), 3 * scale);

	theme->set_type_variation(SNAME("HeaderSmall"), SNAME("Label"));
	theme->set_font_size(SNAME("font_size"), SNAME("HeaderSmall"), default_font_size + 4);

	theme->set_type_variation(SNAME("HeaderMedium"), SNAME("Label"));
	theme->set_font_size(SNAME("font_size"), SNAME("HeaderMedium"), default_font_size + 8);

	theme->set_type_variation(SNAME("HeaderLarge"), SNAME("Label"));
	theme->set_font_size(SNAME("font_size"), SNAME("HeaderLarge"), default_font_size + 12);

	// LineEdit

	Ref<StyleBoxFlat> style_line_edit = make_flat_stylebox(style_normal_color);
	// Add a line at the bottom to make LineEdits distinguishable from Buttons.
	style_line_edit->set_border_width(SIDE_BOTTOM, 2);
	style_line_edit->set_border_color(style_pressed_color);
	theme->set_stylebox(SNAME("normal"), SNAME("LineEdit"), style_line_edit);

	theme->set_stylebox(SNAME("focus"), SNAME("LineEdit"), focus);

	Ref<StyleBoxFlat> style_line_edit_read_only = make_flat_stylebox(style_disabled_color);
	// Add a line at the bottom to make LineEdits distinguishable from Buttons.
	style_line_edit_read_only->set_border_width(SIDE_BOTTOM, 2);
	style_line_edit_read_only->set_border_color(style_pressed_color * Color(1, 1, 1, 0.5));
	theme->set_stylebox(SNAME("read_only"), SNAME("LineEdit"), style_line_edit_read_only);

	theme->set_font(SNAME("font"), SNAME("LineEdit"), Ref<Font>());
	theme->set_font_size(SNAME("font_size"), SNAME("LineEdit"), -1);

	theme->set_color(SNAME("font_color"), SNAME("LineEdit"), control_font_color);
	theme->set_color(SNAME("font_selected_color"), SNAME("LineEdit"), control_font_pressed_color);
	theme->set_color(SNAME("font_uneditable_color"), SNAME("LineEdit"), control_font_disabled_color);
	theme->set_color(SNAME("font_placeholder_color"), SNAME("LineEdit"), control_font_placeholder_color);
	theme->set_color(SNAME("font_outline_color"), SNAME("LineEdit"), Color(1, 1, 1));
	theme->set_color(SNAME("caret_color"), SNAME("LineEdit"), control_font_hover_color);
	theme->set_color(SNAME("selection_color"), SNAME("LineEdit"), control_selection_color);
	theme->set_color(SNAME("clear_button_color"), SNAME("LineEdit"), control_font_color);
	theme->set_color(SNAME("clear_button_color_pressed"), SNAME("LineEdit"), control_font_pressed_color);

	theme->set_constant(SNAME("minimum_character_width"), SNAME("LineEdit"), 4);
	theme->set_constant(SNAME("outline_size"), SNAME("LineEdit"), 0);
	theme->set_constant(SNAME("caret_width"), SNAME("LineEdit"), 1);

	theme->set_icon(SNAME("clear"), SNAME("LineEdit"), icons["line_edit_clear"]);

	// ProgressBar

	theme->set_stylebox(SNAME("bg"), SNAME("ProgressBar"), make_flat_stylebox(style_disabled_color, 2, 2, 2, 2, 6));
	theme->set_stylebox(SNAME("fg"), SNAME("ProgressBar"), make_flat_stylebox(style_progress_color, 2, 2, 2, 2, 6));

	theme->set_font(SNAME("font"), SNAME("ProgressBar"), Ref<Font>());
	theme->set_font_size(SNAME("font_size"), SNAME("ProgressBar"), -1);

	theme->set_color(SNAME("font_color"), SNAME("ProgressBar"), control_font_hover_color);
	theme->set_color(SNAME("font_shadow_color"), SNAME("ProgressBar"), Color(0, 0, 0));
	theme->set_color(SNAME("font_outline_color"), SNAME("ProgressBar"), Color(1, 1, 1));

	theme->set_constant(SNAME("outline_size"), SNAME("ProgressBar"), 0);

	// TextEdit

	theme->set_stylebox(SNAME("normal"), SNAME("TextEdit"), style_line_edit);
	theme->set_stylebox(SNAME("focus"), SNAME("TextEdit"), focus);
	theme->set_stylebox(SNAME("read_only"), SNAME("TextEdit"), style_line_edit_read_only);

	theme->set_icon(SNAME("tab"), SNAME("TextEdit"), icons["text_edit_tab"]);
	theme->set_icon(SNAME("space"), SNAME("TextEdit"), icons["text_edit_space"]);

	theme->set_font(SNAME("font"), SNAME("TextEdit"), Ref<Font>());
	theme->set_font_size(SNAME("font_size"), SNAME("TextEdit"), -1);

	theme->set_color(SNAME("background_color"), SNAME("TextEdit"), Color(0, 0, 0, 0));
	theme->set_color(SNAME("font_color"), SNAME("TextEdit"), control_font_color);
	theme->set_color(SNAME("font_selected_color"), SNAME("TextEdit"), control_font_pressed_color);
	theme->set_color(SNAME("font_readonly_color"), SNAME("TextEdit"), control_font_disabled_color);
	theme->set_color(SNAME("font_placeholder_color"), SNAME("TextEdit"), control_font_placeholder_color);
	theme->set_color(SNAME("font_outline_color"), SNAME("TextEdit"), Color(1, 1, 1));
	theme->set_color(SNAME("selection_color"), SNAME("TextEdit"), control_selection_color);
	theme->set_color(SNAME("current_line_color"), SNAME("TextEdit"), Color(0.25, 0.25, 0.26, 0.8));
	theme->set_color(SNAME("caret_color"), SNAME("TextEdit"), control_font_color);
	theme->set_color(SNAME("caret_background_color"), SNAME("TextEdit"), Color(0, 0, 0));
	theme->set_color(SNAME("word_highlighted_color"), SNAME("TextEdit"), Color(0.5, 0.5, 0.5, 0.25));
	theme->set_color(SNAME("search_result_color"), SNAME("TextEdit"), Color(0.3, 0.3, 0.3));
	theme->set_color(SNAME("search_result_border_color"), SNAME("TextEdit"), Color(0.3, 0.3, 0.3, 0.4));

	theme->set_constant(SNAME("line_spacing"), SNAME("TextEdit"), 4 * scale);
	theme->set_constant(SNAME("outline_size"), SNAME("TextEdit"), 0);
	theme->set_constant(SNAME("caret_width"), SNAME("TextEdit"), 1);

	// CodeEdit

	theme->set_stylebox(SNAME("normal"), SNAME("CodeEdit"), style_line_edit);
	theme->set_stylebox(SNAME("focus"), SNAME("CodeEdit"), focus);
	theme->set_stylebox(SNAME("read_only"), SNAME("CodeEdit"), style_line_edit_read_only);
	theme->set_stylebox(SNAME("completion"), SNAME("CodeEdit"), make_flat_stylebox(style_normal_color, 0, 0, 0, 0));

	theme->set_icon(SNAME("tab"), SNAME("CodeEdit"), icons["text_edit_tab"]);
	theme->set_icon(SNAME("space"), SNAME("CodeEdit"), icons["text_edit_space"]);
	theme->set_icon(SNAME("breakpoint"), SNAME("CodeEdit"), icons["breakpoint"]);
	theme->set_icon(SNAME("bookmark"), SNAME("CodeEdit"), icons["bookmark"]);
	theme->set_icon(SNAME("executing_line"), SNAME("CodeEdit"), icons["arrow_right"]);
	theme->set_icon(SNAME("can_fold"), SNAME("CodeEdit"), icons["arrow_down"]);
	theme->set_icon(SNAME("folded"), SNAME("CodeEdit"), icons["arrow_right"]);
	theme->set_icon(SNAME("folded_eol_icon"), SNAME("CodeEdit"), icons["text_edit_ellipsis"]);

	theme->set_font(SNAME("font"), SNAME("CodeEdit"), Ref<Font>());
	theme->set_font_size(SNAME("font_size"), SNAME("CodeEdit"), -1);

	theme->set_color(SNAME("background_color"), SNAME("CodeEdit"), Color(0, 0, 0, 0));
	theme->set_color(SNAME("completion_background_color"), SNAME("CodeEdit"), Color(0.17, 0.16, 0.2));
	theme->set_color(SNAME("completion_selected_color"), SNAME("CodeEdit"), Color(0.26, 0.26, 0.27));
	theme->set_color(SNAME("completion_existing_color"), SNAME("CodeEdit"), Color(0.87, 0.87, 0.87, 0.13));
	theme->set_color(SNAME("completion_scroll_color"), SNAME("CodeEdit"), control_font_pressed_color);
	theme->set_color(SNAME("completion_font_color"), SNAME("CodeEdit"), Color(0.67, 0.67, 0.67));
	theme->set_color(SNAME("font_color"), SNAME("CodeEdit"), control_font_color);
	theme->set_color(SNAME("font_selected_color"), SNAME("CodeEdit"), Color(0, 0, 0));
	theme->set_color(SNAME("font_readonly_color"), SNAME("CodeEdit"), Color(control_font_color.r, control_font_color.g, control_font_color.b, 0.5f));
	theme->set_color(SNAME("font_placeholder_color"), SNAME("CodeEdit"), control_font_placeholder_color);
	theme->set_color(SNAME("font_outline_color"), SNAME("CodeEdit"), Color(1, 1, 1));
	theme->set_color(SNAME("selection_color"), SNAME("CodeEdit"), control_selection_color);
	theme->set_color(SNAME("bookmark_color"), SNAME("CodeEdit"), Color(0.5, 0.64, 1, 0.8));
	theme->set_color(SNAME("breakpoint_color"), SNAME("CodeEdit"), Color(0.9, 0.29, 0.3));
	theme->set_color(SNAME("executing_line_color"), SNAME("CodeEdit"), Color(0.98, 0.89, 0.27));
	theme->set_color(SNAME("current_line_color"), SNAME("CodeEdit"), Color(0.25, 0.25, 0.26, 0.8));
	theme->set_color(SNAME("code_folding_color"), SNAME("CodeEdit"), Color(0.8, 0.8, 0.8, 0.8));
	theme->set_color(SNAME("caret_color"), SNAME("CodeEdit"), control_font_color);
	theme->set_color(SNAME("caret_background_color"), SNAME("CodeEdit"), Color(0, 0, 0));
	theme->set_color(SNAME("brace_mismatch_color"), SNAME("CodeEdit"), Color(1, 0.2, 0.2));
	theme->set_color(SNAME("line_number_color"), SNAME("CodeEdit"), Color(0.67, 0.67, 0.67, 0.4));
	theme->set_color(SNAME("word_highlighted_color"), SNAME("CodeEdit"), Color(0.8, 0.9, 0.9, 0.15));
	theme->set_color(SNAME("line_length_guideline_color"), SNAME("CodeEdit"), Color(0.3, 0.5, 0.8, 0.1));
	theme->set_color(SNAME("search_result_color"), SNAME("CodeEdit"), Color(0.3, 0.3, 0.3));
	theme->set_color(SNAME("search_result_border_color"), SNAME("CodeEdit"), Color(0.3, 0.3, 0.3, 0.4));

	theme->set_constant(SNAME("completion_lines"), SNAME("CodeEdit"), 7);
	theme->set_constant(SNAME("completion_max_width"), SNAME("CodeEdit"), 50);
	theme->set_constant(SNAME("completion_scroll_width"), SNAME("CodeEdit"), 3);
	theme->set_constant(SNAME("line_spacing"), SNAME("CodeEdit"), 4 * scale);
	theme->set_constant(SNAME("outline_size"), SNAME("CodeEdit"), 0);

	Ref<Texture2D> empty_icon = memnew(ImageTexture);

	const Ref<StyleBoxFlat> style_scrollbar = make_flat_stylebox(style_normal_color, 4, 4, 4, 4, 10);
	Ref<StyleBoxFlat> style_scrollbar_grabber = make_flat_stylebox(style_progress_color, 4, 4, 4, 4, 10);
	Ref<StyleBoxFlat> style_scrollbar_grabber_highlight = make_flat_stylebox(style_focus_color, 4, 4, 4, 4, 10);
	Ref<StyleBoxFlat> style_scrollbar_grabber_pressed = make_flat_stylebox(style_focus_color * Color(0.75, 0.75, 0.75), 4, 4, 4, 4, 10);

	// HScrollBar

	theme->set_stylebox(SNAME("scroll"), SNAME("HScrollBar"), style_scrollbar);
	theme->set_stylebox(SNAME("scroll_focus"), SNAME("HScrollBar"), focus);
	theme->set_stylebox(SNAME("grabber"), SNAME("HScrollBar"), style_scrollbar_grabber);
	theme->set_stylebox(SNAME("grabber_highlight"), SNAME("HScrollBar"), style_scrollbar_grabber_highlight);
	theme->set_stylebox(SNAME("grabber_pressed"), SNAME("HScrollBar"), style_scrollbar_grabber_pressed);

	theme->set_icon(SNAME("increment"), SNAME("HScrollBar"), empty_icon);
	theme->set_icon(SNAME("increment_highlight"), SNAME("HScrollBar"), empty_icon);
	theme->set_icon(SNAME("increment_pressed"), SNAME("HScrollBar"), empty_icon);
	theme->set_icon(SNAME("decrement"), SNAME("HScrollBar"), empty_icon);
	theme->set_icon(SNAME("decrement_highlight"), SNAME("HScrollBar"), empty_icon);
	theme->set_icon(SNAME("decrement_pressed"), SNAME("HScrollBar"), empty_icon);

	// VScrollBar

	theme->set_stylebox(SNAME("scroll"), SNAME("VScrollBar"), style_scrollbar);
	theme->set_stylebox(SNAME("scroll_focus"), SNAME("VScrollBar"), focus);
	theme->set_stylebox(SNAME("grabber"), SNAME("VScrollBar"), style_scrollbar_grabber);
	theme->set_stylebox(SNAME("grabber_highlight"), SNAME("VScrollBar"), style_scrollbar_grabber_highlight);
	theme->set_stylebox(SNAME("grabber_pressed"), SNAME("VScrollBar"), style_scrollbar_grabber_pressed);

	theme->set_icon(SNAME("increment"), SNAME("VScrollBar"), empty_icon);
	theme->set_icon(SNAME("increment_highlight"), SNAME("VScrollBar"), empty_icon);
	theme->set_icon(SNAME("increment_pressed"), SNAME("VScrollBar"), empty_icon);
	theme->set_icon(SNAME("decrement"), SNAME("VScrollBar"), empty_icon);
	theme->set_icon(SNAME("decrement_highlight"), SNAME("VScrollBar"), empty_icon);
	theme->set_icon(SNAME("decrement_pressed"), SNAME("VScrollBar"), empty_icon);

	const Ref<StyleBoxFlat> style_slider = make_flat_stylebox(style_normal_color, 4, 4, 4, 4, 4);
	const Ref<StyleBoxFlat> style_slider_grabber = make_flat_stylebox(style_progress_color, 4, 4, 4, 4, 4);
	const Ref<StyleBoxFlat> style_slider_grabber_highlight = make_flat_stylebox(style_focus_color, 4, 4, 4, 4, 4);

	// HSlider

	theme->set_stylebox(SNAME("slider"), SNAME("HSlider"), style_slider);
	theme->set_stylebox(SNAME("grabber_area"), SNAME("HSlider"), style_slider_grabber);
	theme->set_stylebox(SNAME("grabber_area_highlight"), SNAME("HSlider"), style_slider_grabber_highlight);

	theme->set_icon(SNAME("grabber"), SNAME("HSlider"), icons["slider_grabber"]);
	theme->set_icon(SNAME("grabber_highlight"), SNAME("HSlider"), icons["slider_grabber_hl"]);
	theme->set_icon(SNAME("grabber_disabled"), SNAME("HSlider"), icons["slider_grabber_disabled"]);
	theme->set_icon(SNAME("tick"), SNAME("HSlider"), icons["hslider_tick"]);

	// VSlider

	theme->set_stylebox(SNAME("slider"), SNAME("VSlider"), style_slider);
	theme->set_stylebox(SNAME("grabber_area"), SNAME("VSlider"), style_slider_grabber);
	theme->set_stylebox(SNAME("grabber_area_highlight"), SNAME("VSlider"), style_slider_grabber_highlight);

	theme->set_icon(SNAME("grabber"), SNAME("VSlider"), icons["slider_grabber"]);
	theme->set_icon(SNAME("grabber_highlight"), SNAME("VSlider"), icons["slider_grabber_hl"]);
	theme->set_icon(SNAME("grabber_disabled"), SNAME("VSlider"), icons["slider_grabber_disabled"]);
	theme->set_icon(SNAME("tick"), SNAME("VSlider"), icons["vslider_tick"]);

	// SpinBox

	theme->set_icon(SNAME("updown"), SNAME("SpinBox"), icons["updown"]);

	// ScrollContainer

	Ref<StyleBoxEmpty> empty;
	empty.instantiate();
	theme->set_stylebox(SNAME("bg"), SNAME("ScrollContainer"), empty);

	// Window

	theme->set_stylebox(SNAME("embedded_border"), SNAME("Window"), sb_expand(make_flat_stylebox(style_popup_color, 10, 28, 10, 8), 8, 32, 8, 6));
	theme->set_constant(SNAME("scaleborder_size"), SNAME("Window"), 4 * scale);

	theme->set_font(SNAME("title_font"), SNAME("Window"), Ref<Font>());
	theme->set_font_size(SNAME("title_font_size"), SNAME("Window"), -1);
	theme->set_color(SNAME("title_color"), SNAME("Window"), control_font_color);
	theme->set_color(SNAME("title_outline_modulate"), SNAME("Window"), Color(1, 1, 1));
	theme->set_constant(SNAME("title_outline_size"), SNAME("Window"), 0);
	theme->set_constant(SNAME("title_height"), SNAME("Window"), 36 * scale);
	theme->set_constant(SNAME("resize_margin"), SNAME("Window"), 4 * scale);

	theme->set_icon(SNAME("close"), SNAME("Window"), icons["close"]);
	theme->set_icon(SNAME("close_pressed"), SNAME("Window"), icons["close_hl"]);
	theme->set_constant(SNAME("close_h_ofs"), SNAME("Window"), 18 * scale);
	theme->set_constant(SNAME("close_v_ofs"), SNAME("Window"), 24 * scale);

	// Dialogs

	theme->set_constant(SNAME("margin"), SNAME("Dialogs"), 8 * scale);
	theme->set_constant(SNAME("button_margin"), SNAME("Dialogs"), 32 * scale);

	// AcceptDialog

	theme->set_stylebox(SNAME("panel"), SNAME("AcceptDialog"), make_flat_stylebox(style_popup_color, 0, 0, 0, 0));

	// File Dialog

	theme->set_icon(SNAME("parent_folder"), SNAME("FileDialog"), icons["folder_up"]);
	theme->set_icon(SNAME("back_folder"), SNAME("FileDialog"), icons["arrow_left"]);
	theme->set_icon(SNAME("forward_folder"), SNAME("FileDialog"), icons["arrow_right"]);
	theme->set_icon(SNAME("reload"), SNAME("FileDialog"), icons["reload"]);
	theme->set_icon(SNAME("toggle_hidden"), SNAME("FileDialog"), icons["visibility_visible"]);
	theme->set_icon(SNAME("folder"), SNAME("FileDialog"), icons["folder"]);
	theme->set_icon(SNAME("file"), SNAME("FileDialog"), icons["file"]);
	theme->set_color(SNAME("folder_icon_modulate"), SNAME("FileDialog"), Color(1, 1, 1));
	theme->set_color(SNAME("file_icon_modulate"), SNAME("FileDialog"), Color(1, 1, 1));
	theme->set_color(SNAME("files_disabled"), SNAME("FileDialog"), Color(0, 0, 0, 0.7));

	// Popup

	theme->set_stylebox(SNAME("panel"), SNAME("PopupPanel"), make_flat_stylebox(style_normal_color));

	// PopupDialog

	theme->set_stylebox(SNAME("panel"), SNAME("PopupDialog"), make_flat_stylebox(style_normal_color));

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

	theme->set_stylebox(SNAME("panel"), SNAME("PopupMenu"), style_popup_panel);
	theme->set_stylebox(SNAME("panel_disabled"), SNAME("PopupMenu"), style_popup_panel_disabled);
	theme->set_stylebox(SNAME("hover"), SNAME("PopupMenu"), make_flat_stylebox(style_popup_hover_color));
	theme->set_stylebox(SNAME("separator"), SNAME("PopupMenu"), separator_horizontal);
	theme->set_stylebox(SNAME("labeled_separator_left"), SNAME("PopupMenu"), separator_horizontal);
	theme->set_stylebox(SNAME("labeled_separator_right"), SNAME("PopupMenu"), separator_horizontal);

	theme->set_icon(SNAME("checked"), SNAME("PopupMenu"), icons["checked"]);
	theme->set_icon(SNAME("unchecked"), SNAME("PopupMenu"), icons["unchecked"]);
	theme->set_icon(SNAME("radio_checked"), SNAME("PopupMenu"), icons["radio_checked"]);
	theme->set_icon(SNAME("radio_unchecked"), SNAME("PopupMenu"), icons["radio_unchecked"]);
	theme->set_icon(SNAME("submenu"), SNAME("PopupMenu"), icons["arrow_right"]);
	theme->set_icon(SNAME("submenu_mirrored"), SNAME("PopupMenu"), icons["arrow_left"]);

	theme->set_font(SNAME("font"), SNAME("PopupMenu"), Ref<Font>());
	theme->set_font_size(SNAME("font_size"), SNAME("PopupMenu"), -1);

	theme->set_color(SNAME("font_color"), SNAME("PopupMenu"), control_font_color);
	theme->set_color(SNAME("font_accelerator_color"), SNAME("PopupMenu"), Color(0.7, 0.7, 0.7, 0.8));
	theme->set_color(SNAME("font_disabled_color"), SNAME("PopupMenu"), Color(0.4, 0.4, 0.4, 0.8));
	theme->set_color(SNAME("font_hover_color"), SNAME("PopupMenu"), control_font_color);
	theme->set_color(SNAME("font_separator_color"), SNAME("PopupMenu"), control_font_color);
	theme->set_color(SNAME("font_outline_color"), SNAME("PopupMenu"), Color(1, 1, 1));

	theme->set_constant(SNAME("hseparation"), SNAME("PopupMenu"), 4 * scale);
	theme->set_constant(SNAME("vseparation"), SNAME("PopupMenu"), 4 * scale);
	theme->set_constant(SNAME("outline_size"), SNAME("PopupMenu"), 0);
	theme->set_constant(SNAME("item_start_padding"), SNAME("PopupMenu"), 2 * scale);
	theme->set_constant(SNAME("item_end_padding"), SNAME("PopupMenu"), 2 * scale);

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

	theme->set_stylebox(SNAME("frame"), SNAME("GraphNode"), graphnode_normal);
	theme->set_stylebox(SNAME("selectedframe"), SNAME("GraphNode"), graphnode_selected);
	theme->set_stylebox(SNAME("comment"), SNAME("GraphNode"), graphnode_comment_normal);
	theme->set_stylebox(SNAME("commentfocus"), SNAME("GraphNode"), graphnode_comment_selected);
	theme->set_stylebox(SNAME("breakpoint"), SNAME("GraphNode"), graphnode_breakpoint);
	theme->set_stylebox(SNAME("position"), SNAME("GraphNode"), graphnode_position);

	theme->set_icon(SNAME("port"), SNAME("GraphNode"), icons["graph_port"]);
	theme->set_icon(SNAME("close"), SNAME("GraphNode"), icons["close"]);
	theme->set_icon(SNAME("resizer"), SNAME("GraphNode"), icons["resizer_se"]);
	theme->set_font(SNAME("title_font"), SNAME("GraphNode"), Ref<Font>());
	theme->set_color(SNAME("title_color"), SNAME("GraphNode"), control_font_color);
	theme->set_color(SNAME("close_color"), SNAME("GraphNode"), control_font_color);
	theme->set_color(SNAME("resizer_color"), SNAME("GraphNode"), control_font_color);
	theme->set_constant(SNAME("separation"), SNAME("GraphNode"), 2 * scale);
	theme->set_constant(SNAME("title_offset"), SNAME("GraphNode"), 26 * scale);
	theme->set_constant(SNAME("close_offset"), SNAME("GraphNode"), 22 * scale);
	theme->set_constant(SNAME("port_offset"), SNAME("GraphNode"), 0);

	// Tree

	theme->set_stylebox(SNAME("bg"), SNAME("Tree"), make_flat_stylebox(style_normal_color, 4, 4, 4, 5));
	theme->set_stylebox(SNAME("bg_focus"), SNAME("Tree"), focus);
	theme->set_stylebox(SNAME("selected"), SNAME("Tree"), make_flat_stylebox(style_selected_color));
	theme->set_stylebox(SNAME("selected_focus"), SNAME("Tree"), make_flat_stylebox(style_selected_color));
	theme->set_stylebox(SNAME("cursor"), SNAME("Tree"), focus);
	theme->set_stylebox(SNAME("cursor_unfocused"), SNAME("Tree"), focus);
	theme->set_stylebox(SNAME("button_pressed"), SNAME("Tree"), button_pressed);
	theme->set_stylebox(SNAME("title_button_normal"), SNAME("Tree"), make_flat_stylebox(style_pressed_color, 4, 4, 4, 4));
	theme->set_stylebox(SNAME("title_button_pressed"), SNAME("Tree"), make_flat_stylebox(style_hover_color, 4, 4, 4, 4));
	theme->set_stylebox(SNAME("title_button_hover"), SNAME("Tree"), make_flat_stylebox(style_normal_color, 4, 4, 4, 4));
	theme->set_stylebox(SNAME("custom_button"), SNAME("Tree"), button_normal);
	theme->set_stylebox(SNAME("custom_button_pressed"), SNAME("Tree"), button_pressed);
	theme->set_stylebox(SNAME("custom_button_hover"), SNAME("Tree"), button_hover);

	theme->set_icon(SNAME("checked"), SNAME("Tree"), icons["checked"]);
	theme->set_icon(SNAME("unchecked"), SNAME("Tree"), icons["unchecked"]);
	theme->set_icon(SNAME("indeterminate"), SNAME("Tree"), icons["indeterminate"]);
	theme->set_icon(SNAME("updown"), SNAME("Tree"), icons["updown"]);
	theme->set_icon(SNAME("select_arrow"), SNAME("Tree"), icons["option_button_arrow"]);
	theme->set_icon(SNAME("arrow"), SNAME("Tree"), icons["arrow_down"]);
	theme->set_icon(SNAME("arrow_collapsed"), SNAME("Tree"), icons["arrow_right"]);
	theme->set_icon(SNAME("arrow_collapsed_mirrored"), SNAME("Tree"), icons["arrow_left"]);

	theme->set_font(SNAME("title_button_font"), SNAME("Tree"), Ref<Font>());
	theme->set_font(SNAME("font"), SNAME("Tree"), Ref<Font>());
	theme->set_font_size(SNAME("font_size"), SNAME("Tree"), -1);

	theme->set_color(SNAME("title_button_color"), SNAME("Tree"), control_font_color);
	theme->set_color(SNAME("font_color"), SNAME("Tree"), control_font_low_color);
	theme->set_color(SNAME("font_selected_color"), SNAME("Tree"), control_font_pressed_color);
	theme->set_color(SNAME("font_outline_color"), SNAME("Tree"), Color(1, 1, 1));
	theme->set_color(SNAME("guide_color"), SNAME("Tree"), Color(0.7, 0.7, 0.7, 0.25));
	theme->set_color(SNAME("drop_position_color"), SNAME("Tree"), Color(1, 0.3, 0.2));
	theme->set_color(SNAME("relationship_line_color"), SNAME("Tree"), Color(0.27, 0.27, 0.27));
	theme->set_color(SNAME("parent_hl_line_color"), SNAME("Tree"), Color(0.27, 0.27, 0.27));
	theme->set_color(SNAME("children_hl_line_color"), SNAME("Tree"), Color(0.27, 0.27, 0.27));
	theme->set_color(SNAME("custom_button_font_highlight"), SNAME("Tree"), control_font_hover_color);

	theme->set_constant(SNAME("hseparation"), SNAME("Tree"), 4 * scale);
	theme->set_constant(SNAME("vseparation"), SNAME("Tree"), 4 * scale);
	theme->set_constant(SNAME("item_margin"), SNAME("Tree"), 16 * scale);
	theme->set_constant(SNAME("button_margin"), SNAME("Tree"), 4 * scale);
	theme->set_constant(SNAME("draw_relationship_lines"), SNAME("Tree"), 0);
	theme->set_constant(SNAME("relationship_line_width"), SNAME("Tree"), 1);
	theme->set_constant(SNAME("parent_hl_line_width"), SNAME("Tree"), 1);
	theme->set_constant(SNAME("children_hl_line_width"), SNAME("Tree"), 1);
	theme->set_constant(SNAME("parent_hl_line_margin"), SNAME("Tree"), 0);
	theme->set_constant(SNAME("draw_guides"), SNAME("Tree"), 1);
	theme->set_constant(SNAME("scroll_border"), SNAME("Tree"), 4);
	theme->set_constant(SNAME("scroll_speed"), SNAME("Tree"), 12);
	theme->set_constant(SNAME("outline_size"), SNAME("Tree"), 0);

	// ItemList

	theme->set_stylebox(SNAME("bg"), SNAME("ItemList"), make_flat_stylebox(style_normal_color));
	theme->set_stylebox(SNAME("bg_focus"), SNAME("ItemList"), focus);
	theme->set_constant(SNAME("hseparation"), SNAME("ItemList"), 4);
	theme->set_constant(SNAME("vseparation"), SNAME("ItemList"), 2);
	theme->set_constant(SNAME("icon_margin"), SNAME("ItemList"), 4);
	theme->set_constant(SNAME("line_separation"), SNAME("ItemList"), 2 * scale);

	theme->set_font(SNAME("font"), SNAME("ItemList"), Ref<Font>());
	theme->set_font_size(SNAME("font_size"), SNAME("ItemList"), -1);

	theme->set_color(SNAME("font_color"), SNAME("ItemList"), control_font_lower_color);
	theme->set_color(SNAME("font_selected_color"), SNAME("ItemList"), control_font_pressed_color);
	theme->set_color(SNAME("font_outline_color"), SNAME("ItemList"), Color(1, 1, 1));
	theme->set_color(SNAME("guide_color"), SNAME("ItemList"), Color(0, 0, 0, 0.1));
	theme->set_stylebox(SNAME("selected"), SNAME("ItemList"), make_flat_stylebox(style_selected_color));
	theme->set_stylebox(SNAME("selected_focus"), SNAME("ItemList"), make_flat_stylebox(style_selected_color));
	theme->set_stylebox(SNAME("cursor"), SNAME("ItemList"), focus);
	theme->set_stylebox(SNAME("cursor_unfocused"), SNAME("ItemList"), focus);

	theme->set_constant(SNAME("outline_size"), SNAME("ItemList"), 0);

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

	theme->set_stylebox(SNAME("tab_selected"), SNAME("TabContainer"), style_tab_selected);
	theme->set_stylebox(SNAME("tab_unselected"), SNAME("TabContainer"), style_tab_unselected);
	theme->set_stylebox(SNAME("tab_disabled"), SNAME("TabContainer"), style_tab_disabled);
	theme->set_stylebox(SNAME("panel"), SNAME("TabContainer"), make_flat_stylebox(style_normal_color, 0, 0, 0, 0));

	theme->set_icon(SNAME("increment"), SNAME("TabContainer"), icons["scroll_button_right"]);
	theme->set_icon(SNAME("increment_highlight"), SNAME("TabContainer"), icons["scroll_button_right_hl"]);
	theme->set_icon(SNAME("decrement"), SNAME("TabContainer"), icons["scroll_button_left"]);
	theme->set_icon(SNAME("decrement_highlight"), SNAME("TabContainer"), icons["scroll_button_left_hl"]);
	theme->set_icon(SNAME("menu"), SNAME("TabContainer"), icons["tabs_menu"]);
	theme->set_icon(SNAME("menu_highlight"), SNAME("TabContainer"), icons["tabs_menu_hl"]);

	theme->set_font(SNAME("font"), SNAME("TabContainer"), Ref<Font>());
	theme->set_font_size(SNAME("font_size"), SNAME("TabContainer"), -1);

	theme->set_color(SNAME("font_selected_color"), SNAME("TabContainer"), control_font_hover_color);
	theme->set_color(SNAME("font_unselected_color"), SNAME("TabContainer"), control_font_low_color);
	theme->set_color(SNAME("font_disabled_color"), SNAME("TabContainer"), control_font_disabled_color);
	theme->set_color(SNAME("font_outline_color"), SNAME("TabContainer"), Color(1, 1, 1));

	theme->set_constant(SNAME("side_margin"), SNAME("TabContainer"), 8 * scale);
	theme->set_constant(SNAME("icon_separation"), SNAME("TabContainer"), 4 * scale);
	theme->set_constant(SNAME("outline_size"), SNAME("TabContainer"), 0);

	// TabBar

	theme->set_stylebox(SNAME("tab_selected"), SNAME("TabBar"), style_tab_selected);
	theme->set_stylebox(SNAME("tab_unselected"), SNAME("TabBar"), style_tab_unselected);
	theme->set_stylebox(SNAME("tab_disabled"), SNAME("TabBar"), style_tab_disabled);
	theme->set_stylebox(SNAME("button_pressed"), SNAME("TabBar"), button_pressed);
	theme->set_stylebox(SNAME("button_highlight"), SNAME("TabBar"), button_normal);

	theme->set_icon(SNAME("increment"), SNAME("TabBar"), icons["scroll_button_right"]);
	theme->set_icon(SNAME("increment_highlight"), SNAME("TabBar"), icons["scroll_button_right_hl"]);
	theme->set_icon(SNAME("decrement"), SNAME("TabBar"), icons["scroll_button_left"]);
	theme->set_icon(SNAME("decrement_highlight"), SNAME("TabBar"), icons["scroll_button_left_hl"]);
	theme->set_icon(SNAME("close"), SNAME("TabBar"), icons["close"]);

	theme->set_font(SNAME("font"), SNAME("TabBar"), Ref<Font>());
	theme->set_font_size(SNAME("font_size"), SNAME("TabBar"), -1);

	theme->set_color(SNAME("font_selected_color"), SNAME("TabBar"), control_font_hover_color);
	theme->set_color(SNAME("font_unselected_color"), SNAME("TabBar"), control_font_low_color);
	theme->set_color(SNAME("font_disabled_color"), SNAME("TabBar"), control_font_disabled_color);
	theme->set_color(SNAME("font_outline_color"), SNAME("TabBar"), Color(1, 1, 1));

	theme->set_constant(SNAME("hseparation"), SNAME("TabBar"), 4 * scale);
	theme->set_constant(SNAME("outline_size"), SNAME("TabBar"), 0);

	// Separators

	theme->set_stylebox(SNAME("separator"), SNAME("HSeparator"), separator_horizontal);
	theme->set_stylebox(SNAME("separator"), SNAME("VSeparator"), separator_vertical);

	theme->set_icon(SNAME("close"), SNAME("Icons"), icons["close"]);
	theme->set_font(SNAME("normal"), SNAME("Fonts"), Ref<Font>());
	theme->set_font(SNAME("large"), SNAME("Fonts"), Ref<Font>());

	theme->set_constant(SNAME("separation"), SNAME("HSeparator"), 4 * scale);
	theme->set_constant(SNAME("separation"), SNAME("VSeparator"), 4 * scale);

	// ColorPicker

	theme->set_constant(SNAME("margin"), SNAME("ColorPicker"), 4 * scale);
	theme->set_constant(SNAME("sv_width"), SNAME("ColorPicker"), 256 * scale);
	theme->set_constant(SNAME("sv_height"), SNAME("ColorPicker"), 256 * scale);
	theme->set_constant(SNAME("h_width"), SNAME("ColorPicker"), 30 * scale);
	theme->set_constant(SNAME("label_width"), SNAME("ColorPicker"), 10 * scale);

	theme->set_icon(SNAME("screen_picker"), SNAME("ColorPicker"), icons["color_picker_pipette"]);
	theme->set_icon(SNAME("add_preset"), SNAME("ColorPicker"), icons["add"]);
	theme->set_icon(SNAME("color_hue"), SNAME("ColorPicker"), icons["color_picker_hue"]);
	theme->set_icon(SNAME("color_sample"), SNAME("ColorPicker"), icons["color_picker_sample"]);
	theme->set_icon(SNAME("sample_bg"), SNAME("ColorPicker"), icons["mini_checkerboard"]);
	theme->set_icon(SNAME("overbright_indicator"), SNAME("ColorPicker"), icons["color_picker_overbright"]);
	theme->set_icon(SNAME("bar_arrow"), SNAME("ColorPicker"), icons["color_picker_bar_arrow"]);
	theme->set_icon(SNAME("picker_cursor"), SNAME("ColorPicker"), icons["color_picker_cursor"]);

	// ColorPickerButton

	theme->set_icon(SNAME("bg"), SNAME("ColorPickerButton"), icons["mini_checkerboard"]);
	theme->set_stylebox(SNAME("normal"), SNAME("ColorPickerButton"), button_normal);
	theme->set_stylebox(SNAME("pressed"), SNAME("ColorPickerButton"), button_pressed);
	theme->set_stylebox(SNAME("hover"), SNAME("ColorPickerButton"), button_hover);
	theme->set_stylebox(SNAME("disabled"), SNAME("ColorPickerButton"), button_disabled);
	theme->set_stylebox(SNAME("focus"), SNAME("ColorPickerButton"), focus);

	theme->set_font(SNAME("font"), SNAME("ColorPickerButton"), Ref<Font>());
	theme->set_font_size(SNAME("font_size"), SNAME("ColorPickerButton"), -1);

	theme->set_color(SNAME("font_color"), SNAME("ColorPickerButton"), Color(1, 1, 1, 1));
	theme->set_color(SNAME("font_pressed_color"), SNAME("ColorPickerButton"), Color(0.8, 0.8, 0.8, 1));
	theme->set_color(SNAME("font_hover_color"), SNAME("ColorPickerButton"), Color(1, 1, 1, 1));
	theme->set_color(SNAME("font_focus_color"), SNAME("ColorPickerButton"), Color(1, 1, 1, 1));
	theme->set_color(SNAME("font_disabled_color"), SNAME("ColorPickerButton"), Color(0.9, 0.9, 0.9, 0.3));
	theme->set_color(SNAME("font_outline_color"), SNAME("ColorPickerButton"), Color(1, 1, 1));

	theme->set_constant(SNAME("hseparation"), SNAME("ColorPickerButton"), 2 * scale);
	theme->set_constant(SNAME("outline_size"), SNAME("ColorPickerButton"), 0);

	// ColorPresetButton

	Ref<StyleBoxFlat> preset_sb = make_flat_stylebox(Color(1, 1, 1), 2, 2, 2, 2);
	preset_sb->set_corner_radius_all(2);
	preset_sb->set_corner_detail(2);
	preset_sb->set_anti_aliased(false);

	theme->set_stylebox(SNAME("preset_fg"), SNAME("ColorPresetButton"), preset_sb);
	theme->set_icon(SNAME("preset_bg"), SNAME("ColorPresetButton"), icons["mini_checkerboard"]);
	theme->set_icon(SNAME("overbright_indicator"), SNAME("ColorPresetButton"), icons["color_picker_overbright"]);

	// TooltipPanel + TooltipLabel

	theme->set_stylebox(SNAME("panel"), SNAME("TooltipPanel"),
			make_flat_stylebox(Color(0, 0, 0, 0.5), 2 * default_margin, 0.5 * default_margin, 2 * default_margin, 0.5 * default_margin));

	theme->set_font(SNAME("font"), SNAME("TooltipLabel"), Ref<Font>());
	theme->set_font_size(SNAME("font_size"), SNAME("TooltipLabel"), -1);

	theme->set_color(SNAME("font_color"), SNAME("TooltipLabel"), control_font_color);
	theme->set_color(SNAME("font_shadow_color"), SNAME("TooltipLabel"), Color(0, 0, 0, 0));
	theme->set_color(SNAME("font_outline_color"), SNAME("TooltipLabel"), Color(0, 0, 0, 0));

	theme->set_constant(SNAME("shadow_offset_x"), SNAME("TooltipLabel"), 1);
	theme->set_constant(SNAME("shadow_offset_y"), SNAME("TooltipLabel"), 1);
	theme->set_constant(SNAME("outline_size"), SNAME("TooltipLabel"), 0);

	// RichTextLabel

	theme->set_stylebox(SNAME("focus"), SNAME("RichTextLabel"), focus);
	theme->set_stylebox(SNAME("normal"), SNAME("RichTextLabel"), make_empty_stylebox(0, 0, 0, 0));

	theme->set_font(SNAME("normal_font"), SNAME("RichTextLabel"), Ref<Font>());
	theme->set_font(SNAME("bold_font"), SNAME("RichTextLabel"), Ref<Font>());
	theme->set_font(SNAME("italics_font"), SNAME("RichTextLabel"), Ref<Font>());
	theme->set_font(SNAME("bold_italics_font"), SNAME("RichTextLabel"), Ref<Font>());
	theme->set_font(SNAME("mono_font"), SNAME("RichTextLabel"), Ref<Font>());

	theme->set_font_size(SNAME("normal_font_size"), SNAME("RichTextLabel"), -1);
	theme->set_font_size(SNAME("bold_font_size"), SNAME("RichTextLabel"), -1);
	theme->set_font_size(SNAME("italics_font_size"), SNAME("RichTextLabel"), -1);
	theme->set_font_size(SNAME("bold_italics_font_size"), SNAME("RichTextLabel"), -1);
	theme->set_font_size(SNAME("mono_font_size"), SNAME("RichTextLabel"), -1);

	theme->set_color(SNAME("default_color"), SNAME("RichTextLabel"), Color(1, 1, 1));
	theme->set_color(SNAME("font_selected_color"), SNAME("RichTextLabel"), Color(0, 0, 0));
	theme->set_color(SNAME("selection_color"), SNAME("RichTextLabel"), Color(0.1, 0.1, 1, 0.8));

	theme->set_color(SNAME("font_shadow_color"), SNAME("RichTextLabel"), Color(0, 0, 0, 0));

	theme->set_color(SNAME("font_outline_color"), SNAME("RichTextLabel"), Color(1, 1, 1));

	theme->set_constant(SNAME("shadow_offset_x"), SNAME("RichTextLabel"), 1 * scale);
	theme->set_constant(SNAME("shadow_offset_y"), SNAME("RichTextLabel"), 1 * scale);
	theme->set_constant(SNAME("shadow_outline_size"), SNAME("RichTextLabel"), 1 * scale);

	theme->set_constant(SNAME("line_separation"), SNAME("RichTextLabel"), 0 * scale);
	theme->set_constant(SNAME("table_hseparation"), SNAME("RichTextLabel"), 3 * scale);
	theme->set_constant(SNAME("table_vseparation"), SNAME("RichTextLabel"), 3 * scale);

	theme->set_constant(SNAME("outline_size"), SNAME("RichTextLabel"), 0);

	theme->set_color(SNAME("table_odd_row_bg"), SNAME("RichTextLabel"), Color(0, 0, 0, 0));
	theme->set_color(SNAME("table_even_row_bg"), SNAME("RichTextLabel"), Color(0, 0, 0, 0));
	theme->set_color(SNAME("table_border"), SNAME("RichTextLabel"), Color(0, 0, 0, 0));

	// Containers

	theme->set_icon(SNAME("grabber"), SNAME("VSplitContainer"), icons["vsplitter"]);
	theme->set_icon(SNAME("grabber"), SNAME("HSplitContainer"), icons["hsplitter"]);

	theme->set_constant(SNAME("separation"), SNAME("HBoxContainer"), 4 * scale);
	theme->set_constant(SNAME("separation"), SNAME("VBoxContainer"), 4 * scale);
	theme->set_constant(SNAME("margin_left"), SNAME("MarginContainer"), 0 * scale);
	theme->set_constant(SNAME("margin_top"), SNAME("MarginContainer"), 0 * scale);
	theme->set_constant(SNAME("margin_right"), SNAME("MarginContainer"), 0 * scale);
	theme->set_constant(SNAME("margin_bottom"), SNAME("MarginContainer"), 0 * scale);
	theme->set_constant(SNAME("hseparation"), SNAME("GridContainer"), 4 * scale);
	theme->set_constant(SNAME("vseparation"), SNAME("GridContainer"), 4 * scale);
	theme->set_constant(SNAME("separation"), SNAME("HSplitContainer"), 12 * scale);
	theme->set_constant(SNAME("separation"), SNAME("VSplitContainer"), 12 * scale);
	theme->set_constant(SNAME("autohide"), SNAME("HSplitContainer"), 1 * scale);
	theme->set_constant(SNAME("autohide"), SNAME("VSplitContainer"), 1 * scale);
	theme->set_constant(SNAME("hseparation"), SNAME("HFlowContainer"), 4 * scale);
	theme->set_constant(SNAME("vseparation"), SNAME("HFlowContainer"), 4 * scale);
	theme->set_constant(SNAME("hseparation"), SNAME("VFlowContainer"), 4 * scale);
	theme->set_constant(SNAME("vseparation"), SNAME("VFlowContainer"), 4 * scale);

	theme->set_stylebox(SNAME("panel"), SNAME("PanelContainer"), make_flat_stylebox(style_normal_color, 0, 0, 0, 0));

	theme->set_icon(SNAME("minus"), SNAME("GraphEdit"), icons["zoom_less"]);
	theme->set_icon(SNAME("reset"), SNAME("GraphEdit"), icons["zoom_reset"]);
	theme->set_icon(SNAME("more"), SNAME("GraphEdit"), icons["zoom_more"]);
	theme->set_icon(SNAME("snap"), SNAME("GraphEdit"), icons["grid_snap"]);
	theme->set_icon(SNAME("minimap"), SNAME("GraphEdit"), icons["grid_minimap"]);
	theme->set_icon(SNAME("layout"), SNAME("GraphEdit"), icons["grid_layout"]);
	theme->set_stylebox(SNAME("bg"), SNAME("GraphEdit"), make_flat_stylebox(style_normal_color, 4, 4, 4, 5));
	theme->set_color(SNAME("grid_minor"), SNAME("GraphEdit"), Color(1, 1, 1, 0.05));
	theme->set_color(SNAME("grid_major"), SNAME("GraphEdit"), Color(1, 1, 1, 0.2));
	theme->set_color(SNAME("selection_fill"), SNAME("GraphEdit"), Color(1, 1, 1, 0.3));
	theme->set_color(SNAME("selection_stroke"), SNAME("GraphEdit"), Color(1, 1, 1, 0.8));
	theme->set_color(SNAME("activity"), SNAME("GraphEdit"), Color(1, 1, 1));
	theme->set_constant(SNAME("bezier_len_pos"), SNAME("GraphEdit"), 80 * scale);
	theme->set_constant(SNAME("bezier_len_neg"), SNAME("GraphEdit"), 160 * scale);

	// Visual Node Ports

	theme->set_constant(SNAME("port_grab_distance_horizontal"), SNAME("GraphEdit"), 24 * scale);
	theme->set_constant(SNAME("port_grab_distance_vertical"), SNAME("GraphEdit"), 26 * scale);

	theme->set_stylebox(SNAME("bg"), SNAME("GraphEditMinimap"), make_flat_stylebox(Color(0.24, 0.24, 0.24), 0, 0, 0, 0));
	Ref<StyleBoxFlat> style_minimap_camera = make_flat_stylebox(Color(0.65, 0.65, 0.65, 0.2), 0, 0, 0, 0, 0);
	style_minimap_camera->set_border_color(Color(0.65, 0.65, 0.65, 0.45));
	style_minimap_camera->set_border_width_all(1);
	theme->set_stylebox(SNAME("camera"), SNAME("GraphEditMinimap"), style_minimap_camera);
	theme->set_stylebox(SNAME("node"), SNAME("GraphEditMinimap"), make_flat_stylebox(Color(1, 1, 1), 0, 0, 0, 0, 2));

	theme->set_icon(SNAME("resizer"), SNAME("GraphEditMinimap"), icons["resizer_nw"]);
	theme->set_color(SNAME("resizer_color"), SNAME("GraphEditMinimap"), Color(1, 1, 1, 0.85));

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
