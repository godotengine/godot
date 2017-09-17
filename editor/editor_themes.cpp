/*************************************************************************/
/*  editor_themes.cpp                                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "editor_themes.h"

#include "core/io/resource_loader.h"
#include "editor_fonts.h"
#include "editor_icons.gen.h"
#include "editor_scale.h"
#include "editor_settings.h"
#include "modules/svg/image_loader_svg.h"
#include "time.h"

static Ref<StyleBoxTexture> make_stylebox(Ref<Texture> texture, float p_left, float p_top, float p_right, float p_botton, float p_margin_left = -1, float p_margin_top = -1, float p_margin_right = -1, float p_margin_botton = -1, bool p_draw_center = true) {
	Ref<StyleBoxTexture> style(memnew(StyleBoxTexture));
	style->set_texture(texture);
	style->set_margin_size(MARGIN_LEFT, p_left * EDSCALE);
	style->set_margin_size(MARGIN_RIGHT, p_right * EDSCALE);
	style->set_margin_size(MARGIN_BOTTOM, p_botton * EDSCALE);
	style->set_margin_size(MARGIN_TOP, p_top * EDSCALE);
	style->set_default_margin(MARGIN_LEFT, p_margin_left * EDSCALE);
	style->set_default_margin(MARGIN_RIGHT, p_margin_right * EDSCALE);
	style->set_default_margin(MARGIN_BOTTOM, p_margin_botton * EDSCALE);
	style->set_default_margin(MARGIN_TOP, p_margin_top * EDSCALE);
	style->set_draw_center(p_draw_center);
	return style;
}

static Ref<StyleBoxEmpty> make_empty_stylebox(float p_margin_left = -1, float p_margin_top = -1, float p_margin_right = -1, float p_margin_bottom = -1) {
	Ref<StyleBoxEmpty> style(memnew(StyleBoxEmpty));
	style->set_default_margin(MARGIN_LEFT, p_margin_left * EDSCALE);
	style->set_default_margin(MARGIN_RIGHT, p_margin_right * EDSCALE);
	style->set_default_margin(MARGIN_BOTTOM, p_margin_bottom * EDSCALE);
	style->set_default_margin(MARGIN_TOP, p_margin_top * EDSCALE);
	return style;
}

static Ref<StyleBoxFlat> make_flat_stylebox(Color color, float p_margin_left = -1, float p_margin_top = -1, float p_margin_right = -1, float p_margin_bottom = -1) {
	Ref<StyleBoxFlat> style(memnew(StyleBoxFlat));
	style->set_bg_color(color);
	style->set_default_margin(MARGIN_LEFT, p_margin_left * EDSCALE);
	style->set_default_margin(MARGIN_RIGHT, p_margin_right * EDSCALE);
	style->set_default_margin(MARGIN_BOTTOM, p_margin_bottom * EDSCALE);
	style->set_default_margin(MARGIN_TOP, p_margin_top * EDSCALE);
	return style;
}

static Ref<StyleBoxLine> make_line_stylebox(Color color, int thickness = 1, float grow = 1, bool vertical = false) {
	Ref<StyleBoxLine> style(memnew(StyleBoxLine));
	style->set_color(color);
	style->set_grow(grow);
	style->set_thickness(thickness);
	style->set_vertical(vertical);
	return style;
}

static Ref<StyleBoxFlat> change_border_color(Ref<StyleBoxFlat> p_style, Color p_color) {
	Ref<StyleBoxFlat> style = p_style->duplicate();
	style->set_border_color_all(p_color);
	return style;
}

Ref<ImageTexture> editor_generate_icon(int p_index, bool convert_color, float p_scale = EDSCALE, bool force_filter = false) {

	Ref<ImageTexture> icon = memnew(ImageTexture);
	Ref<Image> img = memnew(Image);

	// dumb gizmo check
	bool is_gizmo = String(editor_icons_names[p_index]).begins_with("Gizmo");

	ImageLoaderSVG::create_image_from_string(img, editor_icons_sources[p_index], p_scale, true, convert_color);

	if ((p_scale - (float)((int)p_scale)) > 0.0 || is_gizmo || force_filter)
		icon->create_from_image(img); // in this case filter really helps
	else
		icon->create_from_image(img, 0);

	return icon;
}

#ifndef ADD_CONVERT_COLOR
#define ADD_CONVERT_COLOR(dictionary, old_color, new_color) dictionary[Color::html(old_color)] = Color::html(new_color)
#endif

void editor_register_and_generate_icons(Ref<Theme> p_theme, bool dark_theme = true, int p_thumb_size = 32, bool only_thumbs = false) {

#ifdef SVG_ENABLED
	Dictionary dark_icon_color_dictionary;
	//convert color:                              FROM       TO
	ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#e0e0e0", "#4f4f4f"); // common icon color
	ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#ffffff", "#000000"); // white
	ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#b4b4b4", "#000000"); // script darker color

	ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#cea4f1", "#bb6dff"); // animation
	ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#fc9c9c", "#ff5f5f"); // spatial
	ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#a5b7f3", "#6d90ff"); // 2d
	ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#708cea", "#0843ff"); // 2d dark
	ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#a5efac", "#29d739"); // control

	ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#ff8484", "#ff3333"); // error
	ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#84ffb1", "#00db50"); // success
	ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#ffd684", "#ffad07"); // warning

	// rainbow
	ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#ff7070", "#ff2929"); // red
	ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#ffeb70", "#ffe337"); // yellow
	ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#9dff70", "#74ff34"); // green
	ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#70ffb9", "#2cff98"); // aqua
	ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#70deff", "#22ccff"); // blue
	ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#9f70ff", "#702aff"); // purple
	ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#ff70ac", "#ff2781"); // pink

	// audio gradient
	ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#ff8484", "#ff4040"); // red
	ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#e1dc7a", "#d6cf4b"); // yellow
	ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#84ffb1", "#00f010"); // green

	ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#ffd684", "#fea900"); // mesh (orange)
	ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#40a2ff", "#68b6ff"); // shape (blue)

	ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#84c2ff", "#5caeff"); // selection (blue)

	ADD_CONVERT_COLOR(dark_icon_color_dictionary, "#ea686c", "#e3383d"); // key xform (red)

	List<String> exceptions;
	exceptions.push_back("EditorPivot");
	exceptions.push_back("EditorHandle");
	exceptions.push_back("Editor3DHandle");
	exceptions.push_back("Godot");
	exceptions.push_back("PanoramaSky");
	exceptions.push_back("ProceduralSky");
	exceptions.push_back("EditorControlAnchor");
	exceptions.push_back("DefaultProjectIcon");

	clock_t begin_time = clock();

	ImageLoaderSVG::set_convert_colors(dark_theme ? NULL : &dark_icon_color_dictionary);

	// generate icons
	if (!only_thumbs)
		for (int i = 0; i < editor_icons_count; i++) {
			List<String>::Element *is_exception = exceptions.find(editor_icons_names[i]);
			if (is_exception) exceptions.erase(is_exception);
			Ref<ImageTexture> icon = editor_generate_icon(i, !dark_theme && !is_exception);
			p_theme->set_icon(editor_icons_names[i], "EditorIcons", icon);
		}

	// generate thumb files with the given thumb size
	bool force_filter = !(p_thumb_size == 64 && p_thumb_size == 32); // we dont need filter with original resolution
	if (p_thumb_size >= 64) {
		float scale = (float)p_thumb_size / 64.0 * EDSCALE;
		for (int i = 0; i < editor_bg_thumbs_count; i++) {
			int index = editor_bg_thumbs_indices[i];
			List<String>::Element *is_exception = exceptions.find(editor_icons_names[index]);
			if (is_exception) exceptions.erase(is_exception);
			Ref<ImageTexture> icon = editor_generate_icon(index, !dark_theme && !is_exception, scale, force_filter);
			p_theme->set_icon(editor_icons_names[index], "EditorIcons", icon);
		}
	} else {
		float scale = (float)p_thumb_size / 32.0 * EDSCALE;
		for (int i = 0; i < editor_md_thumbs_count; i++) {
			int index = editor_md_thumbs_indices[i];
			List<String>::Element *is_exception = exceptions.find(editor_icons_names[index]);
			if (is_exception) exceptions.erase(is_exception);
			Ref<ImageTexture> icon = editor_generate_icon(index, !dark_theme && !is_exception, scale, force_filter);
			p_theme->set_icon(editor_icons_names[index], "EditorIcons", icon);
		}
	}

	ImageLoaderSVG::set_convert_colors(NULL);

	clock_t end_time = clock();

	double time_d = (double)(end_time - begin_time) / CLOCKS_PER_SEC;
	print_line("SVG_GENERATION TIME: " + rtos(time_d));
#else
	print_line("Sorry no icons for you");
#endif
}

Ref<Theme> create_editor_theme(const Ref<Theme> p_theme) {

	Ref<Theme> theme = Ref<Theme>(memnew(Theme));

	const float default_contrast = 0.25;

	//Theme settings
	Color accent_color = EDITOR_DEF("interface/theme/accent_color", Color::html("#000000"));
	Color base_color = EDITOR_DEF("interface/theme/base_color", Color::html("#000000"));
	float contrast = EDITOR_DEF("interface/theme/contrast", default_contrast);

	int preset = EDITOR_DEF("interface/theme/preset", 0);
	int icon_font_color_setting = EDITOR_DEF("interface/theme/icon_and_font_color", 0);
	bool highlight_tabs = EDITOR_DEF("interface/theme/highlight_tabs", false);
	int border_size = EDITOR_DEF("interface/theme/border_size", 1);

	Color script_bg_color = EDITOR_DEF("text_editor/highlighting/background_color", Color(0, 0, 0, 0));

	switch (preset) {
		case 0: { // Default
			accent_color = Color::html("#699ce8");
			base_color = Color::html("#323b4f");
			contrast = default_contrast;
		} break;
		case 1: { // Grey
			accent_color = Color::html("#3e3e3e");
			base_color = Color::html("#3d3d3d");
			contrast = 0.2;
		} break;
		case 2: { // Godot 2
			accent_color = Color::html("#86ace2");
			base_color = Color::html("#3C3A44");
			contrast = 0.25;
		} break;
		case 3: { // Arc
			accent_color = Color::html("#5294e2");
			base_color = Color::html("#383c4a");
			contrast = 0.25;
		} break;
		case 4: { // Light
			accent_color = Color::html("#2070ff");
			base_color = Color::html("#ffffff");
			contrast = 0.08;
		} break;
	}

	//Colors
	int AUTO_COLOR = 0;
	int LIGHT_COLOR = 2;
	bool dark_theme = (icon_font_color_setting == AUTO_COLOR && ((base_color.r + base_color.g + base_color.b) / 3.0) < 0.5) || icon_font_color_setting == LIGHT_COLOR;

	const Color dark_color_1 = base_color.linear_interpolate(Color(0, 0, 0, 1), contrast);
	const Color dark_color_2 = base_color.linear_interpolate(Color(0, 0, 0, 1), contrast * 1.5);
	const Color dark_color_3 = base_color.linear_interpolate(Color(0, 0, 0, 1), contrast * 2);

	const Color background_color = dark_color_2;

	// white (dark theme) or black (light theme), will be used to generate the rest of the colors
	const Color mono_color = dark_theme ? Color(1, 1, 1) : Color(0, 0, 0);

	const Color contrast_color_1 = base_color.linear_interpolate(mono_color, MAX(contrast, default_contrast));
	const Color contrast_color_2 = base_color.linear_interpolate(mono_color, MAX(contrast * 1.5, default_contrast * 1.5));

	const Color font_color = mono_color.linear_interpolate(base_color, 0.25);
	const Color font_color_hl = mono_color.linear_interpolate(base_color, 0.15);
	const Color font_color_disabled = Color(mono_color.r, mono_color.g, mono_color.b, 0.3);
	const Color color_disabled = mono_color.inverted().linear_interpolate(base_color, 0.7);
	const Color color_disabled_bg = mono_color.inverted().linear_interpolate(base_color, 0.9);

	const Color separator_color = Color(mono_color.r, mono_color.g, mono_color.b, 0.1);

	const Color highlight_color = Color(mono_color.r, mono_color.g, mono_color.b, 0.2);

	theme->set_color("accent_color", "Editor", accent_color);
	theme->set_color("base_color", "Editor", base_color);
	theme->set_color("dark_color_1", "Editor", dark_color_1);
	theme->set_color("dark_color_2", "Editor", dark_color_2);
	theme->set_color("dark_color_3", "Editor", dark_color_3);
	theme->set_color("contrast_color_1", "Editor", contrast_color_1);
	theme->set_color("contrast_color_2", "Editor", contrast_color_2);

	theme->set_color("font_color", "Editor", font_color);

	Color success_color = accent_color.linear_interpolate(Color(.6, 1, .6), 0.8);
	Color warning_color = accent_color.linear_interpolate(Color(1, 1, .2), 0.8);
	Color error_color = accent_color.linear_interpolate(Color(1, .2, .2), 0.8);
	theme->set_color("success_color", "Editor", success_color);
	theme->set_color("warning_color", "Editor", warning_color);
	theme->set_color("error_color", "Editor", error_color);

	const int thumb_size = EDITOR_DEF("filesystem/file_dialog/thumbnail_size", 64);
	theme->set_constant("scale", "Editor", EDSCALE);
	theme->set_constant("thumb_size", "Editor", thumb_size);
	theme->set_constant("dark_theme", "Editor", dark_theme);

	//Register icons + font

	// the resolution and the icon color (dark_theme bool) has not changed, so we do not regenerate the icons
	if (p_theme != NULL && fabs(p_theme->get_constant("scale", "Editor") - EDSCALE) < 0.00001 && p_theme->get_constant("dark_theme", "Editor") == dark_theme) {
		// register already generated icons
		for (int i = 0; i < editor_icons_count; i++) {
			theme->set_icon(editor_icons_names[i], "EditorIcons", p_theme->get_icon(editor_icons_names[i], "EditorIcons"));
		}
	} else {
		editor_register_and_generate_icons(theme, dark_theme, thumb_size);
	}
	// thumbnail size has changed, so we regenerate the medium sizes
	if (p_theme != NULL && fabs(p_theme->get_constant("thumb_size", "Editor") - thumb_size) > 0.00001) {
		editor_register_and_generate_icons(p_theme, dark_theme, thumb_size, true);
	}

	editor_register_fonts(theme);

	// Highlighted tabs and border width
	Color tab_color = highlight_tabs ? base_color.linear_interpolate(font_color, contrast) : base_color;
	const int border_width = CLAMP(border_size, 0, 3) * EDSCALE;

	const int default_margin_size = 4;
	const int margin_size_extra = default_margin_size + CLAMP(border_size, 0, 3);

	// styleboxes
	// this is the most commonly used stylebox, variations should be made as duplicate of this
	Ref<StyleBoxFlat> style_default = make_flat_stylebox(base_color, default_margin_size, default_margin_size, default_margin_size, default_margin_size);
	style_default->set_border_width_all(border_width);
	style_default->set_border_color_all(base_color);
	style_default->set_draw_center(true);

	// Button and widgets
	Ref<StyleBoxFlat> style_widget = style_default->duplicate();

	style_widget->set_bg_color(dark_color_1);
	style_widget->set_default_margin(MARGIN_LEFT, 6 * EDSCALE);
	style_widget->set_default_margin(MARGIN_RIGHT, 6 * EDSCALE);
	style_widget->set_border_color_all(dark_color_2);

	Ref<StyleBoxFlat> style_widget_disabled = style_widget->duplicate();
	style_widget_disabled->set_border_color_all(color_disabled);
	style_widget_disabled->set_bg_color(color_disabled_bg);

	Ref<StyleBoxFlat> style_widget_focus = style_widget->duplicate();
	style_widget_focus->set_border_color_all(accent_color);

	Ref<StyleBoxFlat> style_widget_pressed = style_widget->duplicate();
	style_widget_pressed->set_border_color_all(accent_color);

	Ref<StyleBoxFlat> style_widget_hover = style_widget->duplicate();
	style_widget_hover->set_border_color_all(contrast_color_1);

	// style for windows, popups, etc..
	Ref<StyleBoxFlat> style_popup = style_default->duplicate();
	style_popup->set_default_margin(MARGIN_LEFT, default_margin_size * EDSCALE * 2);
	style_popup->set_default_margin(MARGIN_TOP, default_margin_size * EDSCALE * 2);
	style_popup->set_default_margin(MARGIN_RIGHT, default_margin_size * EDSCALE * 2);
	style_popup->set_default_margin(MARGIN_BOTTOM, default_margin_size * EDSCALE * 2);
	style_popup->set_border_color_all(contrast_color_1);
	style_popup->set_border_width_all(MAX(EDSCALE, border_width));
	const Color shadow_color = Color(0, 0, 0, dark_theme ? 0.3 : 0.1);
	style_popup->set_shadow_color(shadow_color);
	style_popup->set_shadow_size(4 * EDSCALE);

	Ref<StyleBoxEmpty> style_empty = make_empty_stylebox(default_margin_size, default_margin_size, default_margin_size, default_margin_size);

	// Tabs
	const int tab_default_margin_side = 10 * EDSCALE;
	Ref<StyleBoxFlat> style_tab_selected = style_default->duplicate();
	style_tab_selected->set_border_width_all(border_width);
	style_tab_selected->set_border_width(MARGIN_BOTTOM, 0);
	style_tab_selected->set_border_color_all(dark_color_3);
	style_tab_selected->set_expand_margin_size(MARGIN_BOTTOM, border_width);
	style_tab_selected->set_default_margin(MARGIN_LEFT, tab_default_margin_side);
	style_tab_selected->set_default_margin(MARGIN_RIGHT, tab_default_margin_side);
	style_tab_selected->set_bg_color(tab_color);

	Ref<StyleBoxFlat> style_tab_unselected = style_tab_selected->duplicate();
	style_tab_unselected->set_bg_color(Color(0.0, 0.0, 0.0, 0.0));
	style_tab_unselected->set_border_color_all(Color(0.0, 0.0, 0.0, 0.0));

	// Editor background
	theme->set_stylebox("Background", "EditorStyles", make_flat_stylebox(background_color, default_margin_size, default_margin_size, default_margin_size, default_margin_size));

	// Focus
	Ref<StyleBoxFlat> style_focus = style_default->duplicate();
	style_focus->set_draw_center(false);
	style_focus->set_border_color_all(contrast_color_2);
	theme->set_stylebox("Focus", "EditorStyles", style_focus);

	// Menu
	Ref<StyleBoxEmpty> style_menu = style_empty;
	theme->set_stylebox("panel", "PanelContainer", style_menu);
	theme->set_stylebox("MenuPanel", "EditorStyles", style_menu);

	// Script Editor
	theme->set_stylebox("ScriptEditorPanel", "EditorStyles", make_empty_stylebox(4, 0, 4, 4));
	theme->set_stylebox("ScriptEditor", "EditorStyles", make_empty_stylebox(0, 0, 0, 0));

	// Play button group
	theme->set_stylebox("PlayButtonPanel", "EditorStyles", style_empty); //make_stylebox(theme->get_icon("GuiPlayButtonGroup", "EditorIcons"), 16, 16, 16, 16, 8, 4, 8, 4));

	//MenuButton
	Ref<StyleBoxFlat> style_menu_hover_border = style_default->duplicate();
	style_menu_hover_border->set_draw_center(false);
	style_menu_hover_border->set_border_width_all(0);
	style_menu_hover_border->set_border_width(MARGIN_BOTTOM, border_width);
	style_menu_hover_border->set_border_color_all(accent_color);

	Ref<StyleBoxFlat> style_menu_hover_bg = style_default->duplicate();
	style_menu_hover_bg->set_border_width_all(0);
	style_menu_hover_bg->set_bg_color(dark_color_1);

	theme->set_stylebox("normal", "MenuButton", style_menu);
	theme->set_stylebox("hover", "MenuButton", style_menu);
	theme->set_stylebox("pressed", "MenuButton", style_menu);
	theme->set_stylebox("focus", "MenuButton", style_menu);
	theme->set_stylebox("disabled", "MenuButton", style_menu);

	theme->set_stylebox("normal", "PopupMenu", style_menu);
	theme->set_stylebox("hover", "PopupMenu", style_menu_hover_bg);
	theme->set_stylebox("pressed", "PopupMenu", style_menu);
	theme->set_stylebox("focus", "PopupMenu", style_menu);
	theme->set_stylebox("disabled", "PopupMenu", style_menu);

	theme->set_stylebox("normal", "ToolButton", style_menu);
	theme->set_stylebox("hover", "ToolButton", style_menu);
	theme->set_stylebox("pressed", "ToolButton", style_menu);
	theme->set_stylebox("focus", "ToolButton", style_menu);
	theme->set_stylebox("disabled", "ToolButton", style_menu);

	theme->set_color("font_color", "MenuButton", font_color);
	theme->set_color("font_color_hover", "MenuButton", font_color_hl);
	theme->set_color("font_color", "ToolButton", font_color);
	theme->set_color("font_color_hover", "ToolButton", font_color_hl);
	theme->set_color("font_color_pressed", "ToolButton", accent_color);

	theme->set_stylebox("MenuHover", "EditorStyles", style_menu_hover_border);

	// Buttons
	theme->set_stylebox("normal", "Button", style_widget);
	theme->set_stylebox("hover", "Button", style_widget_hover);
	theme->set_stylebox("pressed", "Button", style_widget_pressed);
	theme->set_stylebox("focus", "Button", style_widget_focus);
	theme->set_stylebox("disabled", "Button", style_widget_disabled);

	theme->set_color("font_color", "Button", font_color);
	theme->set_color("font_color_hover", "Button", font_color_hl);
	theme->set_color("font_color_pressed", "Button", accent_color);
	theme->set_color("font_color_disabled", "Button", font_color_disabled);
	theme->set_color("icon_color_hover", "Button", font_color_hl);
	// make icon color value bigger because icon image is not complete white
	theme->set_color("icon_color_pressed", "Button", Color(accent_color.r * 1.15, accent_color.g * 1.15, accent_color.b * 1.15, accent_color.a));

	// OptionButton
	theme->set_stylebox("normal", "OptionButton", style_widget);
	theme->set_stylebox("hover", "OptionButton", style_widget_hover);
	theme->set_stylebox("pressed", "OptionButton", style_widget_pressed);
	theme->set_stylebox("focus", "OptionButton", style_widget_focus);
	theme->set_stylebox("disabled", "OptionButton", style_widget_disabled);

	theme->set_color("font_color", "OptionButton", font_color);
	theme->set_color("font_color_hover", "OptionButton", font_color_hl);
	theme->set_color("font_color_pressed", "OptionButton", accent_color);
	theme->set_color("font_color_disabled", "OptionButton", font_color_disabled);
	theme->set_color("icon_color_hover", "OptionButton", font_color_hl);
	theme->set_icon("arrow", "OptionButton", theme->get_icon("GuiOptionArrow", "EditorIcons"));
	theme->set_constant("arrow_margin", "OptionButton", 4 * EDSCALE);
	theme->set_constant("modulate_arrow", "OptionButton", true);

	// CheckButton
	theme->set_icon("on", "CheckButton", theme->get_icon("GuiToggleOn", "EditorIcons"));
	theme->set_icon("off", "CheckButton", theme->get_icon("GuiToggleOff", "EditorIcons"));

	theme->set_color("font_color", "CheckButton", font_color);
	theme->set_color("font_color_hover", "CheckButton", font_color_hl);
	theme->set_color("font_color_pressed", "CheckButton", accent_color);
	theme->set_color("font_color_disabled", "CheckButton", font_color_disabled);
	theme->set_color("icon_color_hover", "CheckButton", font_color_hl);

	// Checkbox
	theme->set_icon("checked", "CheckBox", theme->get_icon("GuiChecked", "EditorIcons"));
	theme->set_icon("unchecked", "CheckBox", theme->get_icon("GuiUnchecked", "EditorIcons"));
	theme->set_icon("radio_checked", "CheckBox", theme->get_icon("GuiRadioChecked", "EditorIcons"));
	theme->set_icon("radio_unchecked", "CheckBox", theme->get_icon("GuiRadioUnchecked", "EditorIcons"));

	theme->set_color("font_color", "CheckBox", font_color);
	theme->set_color("font_color_hover", "CheckBox", font_color_hl);
	theme->set_color("font_color_pressed", "CheckBox", accent_color);
	theme->set_color("font_color_disabled", "CheckBox", font_color_disabled);
	theme->set_color("icon_color_hover", "CheckBox", font_color_hl);

	// PopupMenu
	Ref<StyleBoxFlat> style_popup_menu = style_popup;
	theme->set_stylebox("panel", "PopupMenu", style_popup_menu);
	theme->set_stylebox("separator", "PopupMenu", make_line_stylebox(separator_color, MAX(EDSCALE, border_width), 8 - MAX(EDSCALE, border_width)));
	theme->set_color("font_color", "PopupMenu", font_color);
	theme->set_color("font_color_hover", "PopupMenu", font_color_hl);
	theme->set_color("font_color_accel", "PopupMenu", font_color_disabled);
	theme->set_color("font_color_disabled", "PopupMenu", font_color_disabled);
	theme->set_icon("checked", "PopupMenu", theme->get_icon("GuiChecked", "EditorIcons"));
	theme->set_icon("unchecked", "PopupMenu", theme->get_icon("GuiUnchecked", "EditorIcons"));
	theme->set_icon("radio_checked", "PopupMenu", theme->get_icon("GuiChecked", "EditorIcons"));
	theme->set_icon("radio_unchecked", "PopupMenu", theme->get_icon("GuiUnchecked", "EditorIcons"));

	// Tree & ItemList background
	Ref<StyleBoxFlat> style_tree_bg = style_default->duplicate();
	style_tree_bg->set_bg_color(dark_color_1);
	style_tree_bg->set_border_color_all(dark_color_3);
	theme->set_stylebox("bg", "Tree", style_tree_bg);

	// Tree
	theme->set_icon("checked", "Tree", theme->get_icon("GuiChecked", "EditorIcons"));
	theme->set_icon("unchecked", "Tree", theme->get_icon("GuiUnchecked", "EditorIcons"));
	theme->set_icon("arrow", "Tree", theme->get_icon("GuiTreeArrowDown", "EditorIcons"));
	theme->set_icon("arrow_collapsed", "Tree", theme->get_icon("GuiTreeArrowRight", "EditorIcons"));
	theme->set_icon("select_arrow", "Tree", theme->get_icon("GuiDropdown", "EditorIcons"));
	theme->set_stylebox("bg_focus", "Tree", style_focus);
	theme->set_stylebox("custom_button", "Tree", make_empty_stylebox());
	theme->set_stylebox("custom_button_pressed", "Tree", make_empty_stylebox());
	theme->set_stylebox("custom_button_hover", "Tree", style_widget);
	theme->set_color("custom_button_font_highlight", "Tree", font_color_hl);
	theme->set_color("font_color", "Tree", font_color);
	theme->set_color("font_color_selected", "Tree", font_color);

	Ref<StyleBoxFlat> style_tree_btn = style_default->duplicate();
	style_tree_btn->set_bg_color(contrast_color_1);
	style_tree_btn->set_border_width_all(0);
	theme->set_stylebox("button_pressed", "Tree", style_tree_btn);

	Ref<StyleBoxFlat> style_tree_focus = style_default->duplicate();
	style_tree_focus->set_bg_color(highlight_color);
	style_tree_focus->set_border_width_all(0);
	theme->set_stylebox("selected_focus", "Tree", style_tree_focus);

	Ref<StyleBoxFlat> style_tree_selected = style_tree_focus->duplicate();
	theme->set_stylebox("selected", "Tree", style_tree_selected);

	Ref<StyleBoxFlat> style_tree_cursor = style_default->duplicate();
	style_tree_cursor->set_draw_center(false);
	style_tree_cursor->set_border_width_all(border_width);
	style_tree_cursor->set_border_color_all(contrast_color_1);

	Ref<StyleBoxFlat> style_tree_title = style_default->duplicate();
	style_tree_title->set_bg_color(dark_color_3);
	style_tree_title->set_border_width_all(0);
	theme->set_stylebox("cursor", "Tree", style_tree_cursor);
	theme->set_stylebox("cursor_unfocused", "Tree", style_tree_cursor);
	theme->set_stylebox("title_button_normal", "Tree", style_tree_title);
	theme->set_stylebox("title_button_hover", "Tree", style_tree_title);
	theme->set_stylebox("title_button_pressed", "Tree", style_tree_title);

	// Color prop_category_color = dark_color_1.linear_interpolate(mono_color, 0.12) : dark_color_1.linear_interpolate(Color(0, 0, 0, 1), 0.2);
	// Color prop_section_color = dark_color_1.linear_interpolate(mono_color, 0.09) : dark_color_1.linear_interpolate(Color(0, 1, 0, 1), 0.1);
	// Color prop_subsection_color = dark_color_1.linear_interpolate(mono_color, 0.06) : dark_color_1.linear_interpolate(Color(0, 0, 0, 1), 0.1);
	Color prop_category_color = dark_color_1.linear_interpolate(mono_color, 0.12);
	Color prop_section_color = dark_color_1.linear_interpolate(mono_color, 0.09);
	Color prop_subsection_color = dark_color_1.linear_interpolate(mono_color, 0.06);
	theme->set_color("prop_category", "Editor", prop_category_color);
	theme->set_color("prop_section", "Editor", prop_section_color);
	theme->set_color("prop_subsection", "Editor", prop_subsection_color);
	theme->set_color("drop_position_color", "Tree", accent_color);

	// ItemList
	Ref<StyleBoxFlat> style_itemlist_bg = style_default->duplicate();
	style_itemlist_bg->set_bg_color(dark_color_1);
	style_itemlist_bg->set_border_width_all(border_width);
	style_itemlist_bg->set_border_color_all(dark_color_3);

	Ref<StyleBoxFlat> style_itemlist_cursor = style_default->duplicate();
	style_itemlist_cursor->set_draw_center(false);
	style_itemlist_cursor->set_border_width_all(border_width);
	style_itemlist_cursor->set_border_color_all(highlight_color);
	theme->set_stylebox("cursor", "ItemList", style_itemlist_cursor);
	theme->set_stylebox("cursor_unfocused", "ItemList", style_itemlist_cursor);
	theme->set_stylebox("selected_focus", "ItemList", style_tree_focus);
	theme->set_stylebox("selected", "ItemList", style_tree_selected);
	theme->set_stylebox("bg_focus", "ItemList", style_focus);
	theme->set_stylebox("bg", "ItemList", style_itemlist_bg);
	theme->set_constant("vseparation", "ItemList", 5 * EDSCALE);
	theme->set_color("font_color", "ItemList", font_color);

	// Tabs & TabContainer
	theme->set_stylebox("tab_fg", "TabContainer", style_tab_selected);
	theme->set_stylebox("tab_bg", "TabContainer", style_tab_unselected);
	theme->set_stylebox("tab_fg", "Tabs", style_tab_selected);
	theme->set_stylebox("tab_bg", "Tabs", style_tab_unselected);
	theme->set_color("font_color_fg", "TabContainer", font_color);
	theme->set_color("font_color_bg", "TabContainer", font_color_disabled);
	theme->set_color("font_color_fg", "Tabs", font_color);
	theme->set_color("font_color_bg", "Tabs", font_color_disabled);
	theme->set_icon("menu", "TabContainer", theme->get_icon("GuiTabMenu", "EditorIcons"));
	theme->set_icon("menu_hl", "TabContainer", theme->get_icon("GuiTabMenu", "EditorIcons"));
	theme->set_stylebox("SceneTabFG", "EditorStyles", style_tab_selected);
	theme->set_stylebox("SceneTabBG", "EditorStyles", style_tab_unselected);
	theme->set_icon("close", "Tabs", theme->get_icon("GuiClose", "EditorIcons"));
	theme->set_stylebox("button_pressed", "Tabs", style_menu);
	theme->set_stylebox("button", "Tabs", style_menu);

	// Content of each tab
	Ref<StyleBoxFlat> style_content_panel = style_default->duplicate();
	style_content_panel->set_border_color_all(dark_color_3);
	style_content_panel->set_border_width_all(border_width);
	// compensate the border
	style_content_panel->set_default_margin(MARGIN_TOP, margin_size_extra);
	style_content_panel->set_default_margin(MARGIN_RIGHT, margin_size_extra);
	style_content_panel->set_default_margin(MARGIN_BOTTOM, margin_size_extra);
	style_content_panel->set_default_margin(MARGIN_LEFT, margin_size_extra);

	// this is the stylebox used in 3d and 2d viewports (no borders)
	Ref<StyleBoxFlat> style_content_panel_vp = style_content_panel->duplicate();
	style_content_panel_vp->set_default_margin(MARGIN_LEFT, border_width * 2);
	style_content_panel_vp->set_default_margin(MARGIN_TOP, default_margin_size);
	style_content_panel_vp->set_default_margin(MARGIN_RIGHT, border_width * 2);
	style_content_panel_vp->set_default_margin(MARGIN_BOTTOM, border_width * 2);
	theme->set_stylebox("panel", "TabContainer", style_content_panel);
	theme->set_stylebox("Content", "EditorStyles", style_content_panel_vp);

	// Separators (no separators)
	theme->set_stylebox("separator", "HSeparator", make_line_stylebox(separator_color, border_width));
	theme->set_stylebox("separator", "VSeparator", make_line_stylebox(separator_color, border_width, 0, true));

	// HACK Debuger panel
	Ref<StyleBoxFlat> style_panel_debugger = style_content_panel->duplicate();
	const int v_offset = theme->get_font("font", "Tabs")->get_height() + style_tab_selected->get_minimum_size().height + default_margin_size * EDSCALE;
	style_panel_debugger->set_expand_margin_size(MARGIN_TOP, -v_offset);
	theme->set_stylebox("debugger_panel", "EditorStyles", style_panel_debugger);

	// Debugger
	Ref<StyleBoxFlat> style_debugger_contents = style_content_panel->duplicate();
	style_debugger_contents->set_default_margin(MARGIN_LEFT, 0);
	style_debugger_contents->set_default_margin(MARGIN_BOTTOM, 0);
	style_debugger_contents->set_default_margin(MARGIN_RIGHT, 0);
	style_debugger_contents->set_border_width_all(0);
	style_debugger_contents->set_expand_margin_size(MARGIN_TOP, -v_offset);
	theme->set_stylebox("DebuggerPanel", "EditorStyles", style_debugger_contents);
	Ref<StyleBoxFlat> style_tab_fg_debugger = style_tab_selected->duplicate();
	style_tab_fg_debugger->set_expand_margin_size(MARGIN_LEFT, default_margin_size * EDSCALE + border_width);
	style_tab_fg_debugger->set_default_margin(MARGIN_LEFT, tab_default_margin_side - default_margin_size * EDSCALE);
	theme->set_stylebox("DebuggerTabFG", "EditorStyles", style_tab_fg_debugger);
	Ref<StyleBoxFlat> style_tab_bg_debugger = style_tab_unselected->duplicate();
	style_tab_bg_debugger->set_expand_margin_size(MARGIN_LEFT, default_margin_size * EDSCALE + border_width);
	style_tab_bg_debugger->set_default_margin(MARGIN_LEFT, tab_default_margin_side - default_margin_size * EDSCALE);
	theme->set_stylebox("DebuggerTabBG", "EditorStyles", style_tab_bg_debugger);

	// LineEdit
	theme->set_stylebox("normal", "LineEdit", style_widget);
	theme->set_stylebox("focus", "LineEdit", style_widget_focus);
	theme->set_stylebox("read_only", "LineEdit", style_widget_disabled);
	theme->set_color("read_only", "LineEdit", font_color_disabled);
	theme->set_color("font_color", "LineEdit", font_color);
	theme->set_color("cursor_color", "LineEdit", font_color);

	// TextEdit
	theme->set_stylebox("normal", "TextEdit", style_widget);
	theme->set_stylebox("focus", "TextEdit", style_widget_hover);
	theme->set_constant("side_margin", "TabContainer", 0);
	theme->set_icon("tab", "TextEdit", theme->get_icon("GuiTab", "EditorIcons"));

	// H/VSplitContainer
	theme->set_stylebox("bg", "VSplitContainer", make_stylebox(theme->get_icon("GuiVsplitBg", "EditorIcons"), 1, 1, 1, 1));
	theme->set_stylebox("bg", "HSplitContainer", make_stylebox(theme->get_icon("GuiHsplitBg", "EditorIcons"), 1, 1, 1, 1));

	theme->set_icon("grabber", "VSplitContainer", theme->get_icon("GuiVsplitter", "EditorIcons"));
	theme->set_icon("grabber", "HSplitContainer", theme->get_icon("GuiHsplitter", "EditorIcons"));

	theme->set_constant("separation", "HSplitContainer", default_margin_size * 2 * EDSCALE);
	theme->set_constant("separation", "VSplitContainer", default_margin_size * 2 * EDSCALE);

	// WindowDialog
	Ref<StyleBoxFlat> style_window = style_popup->duplicate();
	style_window->set_border_color_all(tab_color);
	style_window->set_border_width(MARGIN_TOP, 24 * EDSCALE);
	style_window->set_expand_margin_size(MARGIN_TOP, 24 * EDSCALE);
	theme->set_stylebox("panel", "WindowDialog", style_window);
	theme->set_color("title_color", "WindowDialog", font_color);
	theme->set_icon("close", "WindowDialog", theme->get_icon("GuiClose", "EditorIcons"));
	theme->set_icon("close_highlight", "WindowDialog", theme->get_icon("GuiClose", "EditorIcons"));
	theme->set_constant("close_h_ofs", "WindowDialog", 22 * EDSCALE);
	theme->set_constant("close_v_ofs", "WindowDialog", 20 * EDSCALE);
	theme->set_constant("title_height", "WindowDialog", 24 * EDSCALE);

	// complex window, for now only Editor settings and Project settings
	Ref<StyleBoxFlat> style_complex_window = style_window->duplicate();
	style_complex_window->set_bg_color(dark_color_2);
	style_complex_window->set_border_color_all(highlight_tabs ? tab_color : dark_color_2);
	theme->set_stylebox("panel", "EditorSettingsDialog", style_complex_window);
	theme->set_stylebox("panel", "ProjectSettingsEditor", style_complex_window);
	theme->set_stylebox("panel", "EditorAbout", style_complex_window);

	// HScrollBar
	Ref<Texture> empty_icon = memnew(ImageTexture);

	theme->set_stylebox("scroll", "HScrollBar", make_stylebox(theme->get_icon("GuiScrollBg", "EditorIcons"), 5, 5, 5, 5, 0, 0, 0, 0));
	theme->set_stylebox("scroll_focus", "HScrollBar", make_stylebox(theme->get_icon("GuiScrollBg", "EditorIcons"), 5, 5, 5, 5, 0, 0, 0, 0));
	theme->set_stylebox("grabber", "HScrollBar", make_stylebox(theme->get_icon("GuiScrollGrabber", "EditorIcons"), 6, 6, 6, 6, 2, 2, 2, 2));
	theme->set_stylebox("grabber_highlight", "HScrollBar", make_stylebox(theme->get_icon("GuiScrollGrabberHl", "EditorIcons"), 5, 5, 5, 5, 2, 2, 2, 2));
	theme->set_stylebox("grabber_pressed", "HScrollBar", make_stylebox(theme->get_icon("GuiScrollGrabberPressed", "EditorIcons"), 6, 6, 6, 6, 2, 2, 2, 2));

	theme->set_icon("increment", "HScrollBar", empty_icon);
	theme->set_icon("increment_highlight", "HScrollBar", empty_icon);
	theme->set_icon("decrement", "HScrollBar", empty_icon);
	theme->set_icon("decrement_highlight", "HScrollBar", empty_icon);

	// VScrollBar
	theme->set_stylebox("scroll", "VScrollBar", make_stylebox(theme->get_icon("GuiScrollBg", "EditorIcons"), 5, 5, 5, 5, 0, 0, 0, 0));
	theme->set_stylebox("scroll_focus", "VScrollBar", make_stylebox(theme->get_icon("GuiScrollBg", "EditorIcons"), 5, 5, 5, 5, 0, 0, 0, 0));
	theme->set_stylebox("grabber", "VScrollBar", make_stylebox(theme->get_icon("GuiScrollGrabber", "EditorIcons"), 6, 6, 6, 6, 2, 2, 2, 2));
	theme->set_stylebox("grabber_highlight", "VScrollBar", make_stylebox(theme->get_icon("GuiScrollGrabberHl", "EditorIcons"), 5, 5, 5, 5, 2, 2, 2, 2));
	theme->set_stylebox("grabber_pressed", "VScrollBar", make_stylebox(theme->get_icon("GuiScrollGrabberPressed", "EditorIcons"), 6, 6, 6, 6, 2, 2, 2, 2));

	theme->set_icon("increment", "VScrollBar", empty_icon);
	theme->set_icon("increment_highlight", "VScrollBar", empty_icon);
	theme->set_icon("decrement", "VScrollBar", empty_icon);
	theme->set_icon("decrement_highlight", "VScrollBar", empty_icon);

	// HSlider
	theme->set_icon("grabber_highlight", "HSlider", theme->get_icon("GuiSliderGrabberHl", "EditorIcons"));
	theme->set_icon("grabber", "HSlider", theme->get_icon("GuiSliderGrabber", "EditorIcons"));
	theme->set_stylebox("slider", "HSlider", make_flat_stylebox(dark_color_3, 0, default_margin_size / 2, 0, default_margin_size / 2));
	theme->set_stylebox("grabber_area", "HSlider", make_flat_stylebox(contrast_color_1, 0, default_margin_size / 2, 0, default_margin_size / 2));

	// VSlider
	theme->set_icon("grabber", "VSlider", theme->get_icon("GuiSliderGrabber", "EditorIcons"));
	theme->set_icon("grabber_highlight", "VSlider", theme->get_icon("GuiSliderGrabberHl", "EditorIcons"));
	theme->set_stylebox("slider", "VSlider", make_flat_stylebox(dark_color_3, default_margin_size / 2, 0, default_margin_size / 2, 0));
	theme->set_stylebox("grabber_area", "VSlider", make_flat_stylebox(contrast_color_1, default_margin_size / 2, 0, default_margin_size / 2, 0));

	//RichTextLabel
	Color rtl_combined_bg_color = dark_color_1.linear_interpolate(script_bg_color, script_bg_color.a);
	Color rtl_mono_color = (rtl_combined_bg_color.r + rtl_combined_bg_color.g + rtl_combined_bg_color.b > 1.5) ? Color(0, 0, 0) : Color(1, 1, 1);
	Color rtl_font_color = rtl_mono_color.linear_interpolate(rtl_combined_bg_color, 0.25);
	theme->set_color("default_color", "RichTextLabel", rtl_font_color);
	theme->set_stylebox("focus", "RichTextLabel", make_empty_stylebox());
	theme->set_stylebox("normal", "RichTextLabel", style_tree_bg);
	Ref<StyleBoxFlat> style_code = style_tree_bg->duplicate();
	style_code->set_bg_color(rtl_combined_bg_color);
	theme->set_stylebox("code_normal", "RichTextLabel", style_code);
	Ref<StyleBoxFlat> style_code_focus = style_tree_focus->duplicate();
	style_code_focus->set_bg_color(rtl_combined_bg_color);
	theme->set_stylebox("code_focus", "RichTextLabel", style_code_focus);

	theme->set_color("font_color", "RichTextLabel", rtl_font_color);
	theme->set_color("highlight_color", "RichTextLabel", rtl_mono_color);

	// Panel
	theme->set_stylebox("panel", "Panel", make_flat_stylebox(dark_color_1, 6, 4, 6, 4));

	// Label
	theme->set_color("font_color", "Label", font_color);

	// TooltipPanel
	Ref<StyleBoxFlat> style_tooltip = style_popup->duplicate();
	style_tooltip->set_bg_color(Color(mono_color.r, mono_color.g, mono_color.b, 0.9));
	style_tooltip->set_border_width_all(border_width);
	style_tooltip->set_border_color_all(mono_color);
	theme->set_color("font_color", "TooltipPanel", font_color);
	theme->set_stylebox("panel", "TooltipPanel", style_tooltip);

	// PopupPanel
	theme->set_stylebox("panel", "PopupPanel", style_popup);

	// SpinBox
	theme->set_icon("updown", "SpinBox", theme->get_icon("GuiSpinboxUpdown", "EditorIcons"));

	// ProgressBar
	theme->set_stylebox("bg", "ProgressBar", make_stylebox(theme->get_icon("GuiProgressBar", "EditorIcons"), 4, 4, 4, 4, 0, 0, 0, 0));
	theme->set_stylebox("fg", "ProgressBar", make_stylebox(theme->get_icon("GuiProgressFill", "EditorIcons"), 6, 6, 6, 6, 2, 1, 2, 1));
	theme->set_color("font_color", "ProgressBar", font_color);

	// GraphEdit
	theme->set_stylebox("bg", "GraphEdit", style_tree_bg);
	theme->set_color("grid_major", "GraphEdit", Color(font_color.r, font_color.g, font_color.b, 0.1));
	theme->set_color("grid_minor", "GraphEdit", Color(font_color_disabled.r, font_color_disabled.g, font_color_disabled.b, 0.05));
	theme->set_icon("minus", "GraphEdit", theme->get_icon("ZoomLess", "EditorIcons"));
	theme->set_icon("more", "GraphEdit", theme->get_icon("ZoomMore", "EditorIcons"));
	theme->set_icon("reset", "GraphEdit", theme->get_icon("ZoomReset", "EditorIcons"));

	// GraphNode

	Ref<StyleBoxFlat> graphsb = make_flat_stylebox(Color(0, 0, 0, 0.3), 16, 24, 16, 5);
	graphsb->set_border_width_all(border_width);
	graphsb->set_border_color_all(Color(1, 1, 1, 0.6));
	graphsb->set_border_width(MARGIN_TOP, 22 * EDSCALE + border_width);
	Ref<StyleBoxFlat> graphsbselected = make_flat_stylebox(Color(0, 0, 0, 0.4), 16, 24, 16, 5);
	graphsbselected->set_border_width_all(border_width);
	graphsbselected->set_border_color_all(Color(accent_color.r, accent_color.g, accent_color.b, 0.9));
	graphsbselected->set_shadow_size(8 * EDSCALE);
	graphsbselected->set_shadow_color(shadow_color);
	graphsbselected->set_border_width(MARGIN_TOP, 22 * EDSCALE + border_width);
	Ref<StyleBoxFlat> graphsbcomment = make_flat_stylebox(Color(0, 0, 0, 0.3), 16, 24, 16, 5);
	graphsbcomment->set_border_width_all(border_width);
	graphsbcomment->set_border_color_all(Color(1, 1, 1, 0.6));
	graphsbcomment->set_border_width(MARGIN_TOP, 22 * EDSCALE + border_width);
	Ref<StyleBoxFlat> graphsbcommentselected = make_flat_stylebox(Color(0, 0, 0, 0.4), 16, 24, 16, 5);
	graphsbcommentselected->set_border_width_all(border_width);
	graphsbcommentselected->set_border_color_all(Color(1, 1, 1, 0.9));
	graphsbcommentselected->set_border_width(MARGIN_TOP, 22 * EDSCALE + border_width);
	theme->set_stylebox("frame", "GraphNode", graphsb);
	theme->set_stylebox("selectedframe", "GraphNode", graphsbselected);
	theme->set_stylebox("comment", "GraphNode", graphsbcomment);
	theme->set_stylebox("commentfocus", "GraphNode", graphsbcommentselected);

	// FileDialog
	theme->set_color("files_disabled", "FileDialog", font_color_disabled);

	// adaptive script theme constants
	// for comments and elements with lower relevance
	const Color dim_color = Color(font_color.r, font_color.g, font_color.b, 0.5);

	const float mono_value = mono_color.r;
	const Color alpha1 = Color(mono_value, mono_value, mono_value, 0.1);
	const Color alpha2 = Color(mono_value, mono_value, mono_value, 0.3);
	const Color alpha3 = Color(mono_value, mono_value, mono_value, 0.5);
	const Color alpha4 = Color(mono_value, mono_value, mono_value, 0.7);

	// editor main color
	const Color main_color = Color::html(dark_theme ? "#57b3ff" : "#0480ff");

	const Color symbol_color = Color::html("#5792ff").linear_interpolate(mono_color, dark_theme ? 0.5 : 0.3);
	const Color keyword_color = main_color.linear_interpolate(mono_color, 0.4);
	const Color basetype_color = Color::html(dark_theme ? "#42ffc2" : "#00c161");
	const Color type_color = basetype_color.linear_interpolate(mono_color, dark_theme ? 0.7 : 0.5);
	const Color comment_color = dim_color;
	const Color string_color = Color::html(dark_theme ? "#ffd942" : "#ffd118").linear_interpolate(mono_color, dark_theme ? 0.5 : 0.3);

	const Color te_background_color = Color(0, 0, 0, 0);
	const Color completion_background_color = base_color;
	const Color completion_selected_color = alpha1;
	const Color completion_existing_color = alpha2;
	const Color completion_scroll_color = alpha1;
	const Color completion_font_color = font_color;
	const Color text_color = font_color;
	const Color line_number_color = dim_color;
	const Color caret_color = mono_color;
	const Color caret_background_color = mono_color.inverted();
	const Color text_selected_color = dark_color_3;
	const Color selection_color = alpha3;
	const Color brace_mismatch_color = error_color;
	const Color current_line_color = alpha1;
	const Color line_length_guideline_color = warning_color;
	const Color word_highlighted_color = alpha1;
	const Color number_color = basetype_color.linear_interpolate(mono_color, dark_theme ? 0.5 : 0.3);
	const Color function_color = main_color;
	const Color member_variable_color = mono_color;
	const Color mark_color = Color(error_color.r, error_color.g, error_color.b, 0.3);
	const Color breakpoint_color = error_color;
	const Color search_result_color = alpha1;
	const Color search_result_border_color = alpha4;

	theme->set_color("text_editor/theme/symbol_color", "Editor", symbol_color);
	theme->set_color("text_editor/theme/keyword_color", "Editor", keyword_color);
	theme->set_color("text_editor/theme/basetype_color", "Editor", basetype_color);
	theme->set_color("text_editor/theme/type_color", "Editor", type_color);
	theme->set_color("text_editor/theme/comment_color", "Editor", comment_color);
	theme->set_color("text_editor/theme/string_color", "Editor", string_color);
	theme->set_color("text_editor/theme/background_color", "Editor", te_background_color);
	theme->set_color("text_editor/theme/completion_background_color", "Editor", completion_background_color);
	theme->set_color("text_editor/theme/completion_selected_color", "Editor", completion_selected_color);
	theme->set_color("text_editor/theme/completion_existing_color", "Editor", completion_existing_color);
	theme->set_color("text_editor/theme/completion_scroll_color", "Editor", completion_scroll_color);
	theme->set_color("text_editor/theme/completion_font_color", "Editor", completion_font_color);
	theme->set_color("text_editor/theme/text_color", "Editor", text_color);
	theme->set_color("text_editor/theme/line_number_color", "Editor", line_number_color);
	theme->set_color("text_editor/theme/caret_color", "Editor", caret_color);
	theme->set_color("text_editor/theme/caret_background_color", "Editor", caret_background_color);
	theme->set_color("text_editor/theme/text_selected_color", "Editor", text_selected_color);
	theme->set_color("text_editor/theme/selection_color", "Editor", selection_color);
	theme->set_color("text_editor/theme/brace_mismatch_color", "Editor", brace_mismatch_color);
	theme->set_color("text_editor/theme/current_line_color", "Editor", current_line_color);
	theme->set_color("text_editor/theme/line_length_guideline_color", "Editor", line_length_guideline_color);
	theme->set_color("text_editor/theme/word_highlighted_color", "Editor", word_highlighted_color);
	theme->set_color("text_editor/theme/number_color", "Editor", number_color);
	theme->set_color("text_editor/theme/function_color", "Editor", function_color);
	theme->set_color("text_editor/theme/member_variable_color", "Editor", member_variable_color);
	theme->set_color("text_editor/theme/mark_color", "Editor", mark_color);
	theme->set_color("text_editor/theme/breakpoint_color", "Editor", breakpoint_color);
	theme->set_color("text_editor/theme/search_result_color", "Editor", search_result_color);
	theme->set_color("text_editor/theme/search_result_border_color", "Editor", search_result_border_color);

	return theme;
}

Ref<Theme> create_custom_theme() {
	Ref<Theme> theme;

	String custom_theme = EditorSettings::get_singleton()->get("interface/theme/custom_theme");
	if (custom_theme != "") {
		theme = ResourceLoader::load(custom_theme);
	}

	String global_font = EditorSettings::get_singleton()->get("interface/custom_font");
	if (global_font != "") {
		Ref<Font> fnt = ResourceLoader::load(global_font);
		if (fnt.is_valid()) {
			if (!theme.is_valid()) {
				theme.instance();
			}
			theme->set_default_theme_font(fnt);
		}
	}

	return theme;
}
