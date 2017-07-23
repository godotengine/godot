/*************************************************************************/
/*  editor_themes.cpp                                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
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
#include "editor_icons.h"
#include "editor_scale.h"
#include "editor_settings.h"

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
	style->set_light_color(p_color);
	style->set_dark_color(p_color);
	return style;
}

static Ref<StyleBoxFlat> add_additional_border(Ref<StyleBoxFlat> p_style, int p_left, int p_top, int p_right, int p_bottom) {
	Ref<StyleBoxFlat> style = p_style->duplicate();
	style->_set_additional_border_size(MARGIN_LEFT, p_left * EDSCALE);
	style->_set_additional_border_size(MARGIN_RIGHT, p_right * EDSCALE);
	style->_set_additional_border_size(MARGIN_TOP, p_top * EDSCALE);
	style->_set_additional_border_size(MARGIN_BOTTOM, p_bottom * EDSCALE);
	return style;
}

#define HIGHLIGHT_COLOR_LIGHT highlight_color.linear_interpolate(Color(1, 1, 1, 1), 0.3)
#define HIGHLIGHT_COLOR_DARK highlight_color.linear_interpolate(Color(0, 0, 0, 1), 0.5)

Ref<Theme> create_editor_theme() {
	Ref<Theme> theme = Ref<Theme>(memnew(Theme));

	editor_register_fonts(theme);
	editor_register_icons(theme);

	const float default_contrast = 0.25;

	// Define colors
	Color highlight_color = EDITOR_DEF("interface/theme/highlight_color", Color::html("#000000"));
	Color base_color = EDITOR_DEF("interface/theme/base_color", Color::html("#000000"));
	float contrast = EDITOR_DEF("interface/theme/contrast", default_contrast);
	int preset = EDITOR_DEF("interface/theme/preset", 0);
	bool highlight_tabs = EDITOR_DEF("interface/theme/highlight_tabs", false);
	int border_size = EDITOR_DEF("interface/theme/border_size", 1);

	switch (preset) {
		case 0: { // Default
			highlight_color = Color::html("#699ce8");
			base_color = Color::html("#323b4f");
			contrast = default_contrast;
		} break;
		case 1: { // Grey
			highlight_color = Color::html("#3e3e3e");
			base_color = Color::html("#3d3d3d");
			contrast = 0.2;
		} break;
		case 2: { // Godot 2
			highlight_color = Color::html("#86ace2");
			base_color = Color::html("#3C3A44");
			contrast = 0.25;
		} break;
		case 3: { // Arc
			highlight_color = Color::html("#5294e2");
			base_color = Color::html("#383c4a");
			contrast = 0.25;
		} break;
	}

	Color dark_color_1 = base_color.linear_interpolate(Color(0, 0, 0, 1), contrast);
	Color dark_color_2 = base_color.linear_interpolate(Color(0, 0, 0, 1), contrast * 1.5);
	Color dark_color_3 = base_color.linear_interpolate(Color(0, 0, 0, 1), contrast * 2);

	Color light_color_1 = base_color.linear_interpolate(Color(1, 1, 1, 1), contrast);
	Color light_color_2 = base_color.linear_interpolate(Color(1, 1, 1, 1), contrast * 1.5);

	const int border_width = (border_size % 3) * EDSCALE;

	Color title_color_hl = base_color;
	if (highlight_tabs)
		title_color_hl = base_color.linear_interpolate(Color(1, 1, 1, 1), contrast / default_contrast / 10);
	bool dark_bg = ((title_color_hl.r + title_color_hl.g + title_color_hl.b) / 3.0) < 0.5;
	Color title_color_hl_text_color = dark_bg ? Color(1, 1, 1, 0.9) : Color(0, 0, 0, 0.9);
	Ref<Texture> title_hl_close_icon = theme->get_icon((dark_bg ? "GuiCloseLight" : "GuiCloseDark"), "EditorIcons");

	bool dark_base = ((base_color.r + base_color.g + base_color.b) / 3.0) < 0.5;
	Color separator_color = dark_base ? Color(1, 1, 1, 0.1) : Color(0, 0, 0, 0.1);

	theme->set_color("highlight_color", "Editor", highlight_color);
	theme->set_color("base_color", "Editor", base_color);
	theme->set_color("dark_color_1", "Editor", dark_color_1);
	theme->set_color("dark_color_2", "Editor", dark_color_2);
	theme->set_color("dark_color_3", "Editor", dark_color_3);
	theme->set_color("light_color_1", "Editor", light_color_1);
	theme->set_color("light_color_2", "Editor", light_color_2);

	// Checkbox icon
	theme->set_icon("checked", "CheckBox", theme->get_icon("GuiChecked", "EditorIcons"));
	theme->set_icon("unchecked", "CheckBox", theme->get_icon("GuiUnchecked", "EditorIcons"));
	theme->set_icon("checked", "PopupMenu", theme->get_icon("GuiChecked", "EditorIcons"));
	theme->set_icon("unchecked", "PopupMenu", theme->get_icon("GuiUnchecked", "EditorIcons"));

	// Editor background
	Ref<StyleBoxFlat> style_panel = make_flat_stylebox(dark_color_2, 4, 4, 4, 4);
	theme->set_stylebox("Background", "EditorStyles", style_panel);

	// Focus
	Ref<StyleBoxFlat> focus_sbt = make_flat_stylebox(light_color_1, 4, 4, 4, 4);
	focus_sbt->set_draw_center(false);
	focus_sbt->set_border_size(border_width);
	focus_sbt = change_border_color(focus_sbt, light_color_2);
	theme->set_stylebox("Focus", "EditorStyles", focus_sbt);

	// Menu
	Ref<StyleBoxEmpty> style_menu = make_empty_stylebox(4, 4, 4, 4);
	theme->set_stylebox("panel", "PanelContainer", style_menu);
	theme->set_stylebox("MenuPanel", "EditorStyles", style_menu);

	// Play button group
	theme->set_stylebox("PlayButtonPanel", "EditorStyles", make_stylebox(theme->get_icon("GuiPlayButtonGroup", "EditorIcons"), 16, 16, 16, 16, 8, 4, 8, 4));

	Ref<StyleBoxFlat> style_menu_hover_border = make_flat_stylebox(highlight_color, 4, 4, 4, 4);
	Ref<StyleBoxFlat> style_menu_hover_bg = make_flat_stylebox(dark_color_2, 4, 4, 4, 4);

	style_menu_hover_border->set_draw_center(false);
	style_menu_hover_border->_set_additional_border_size(MARGIN_BOTTOM, border_width);
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

	theme->set_color("font_color_hover", "MenuButton", HIGHLIGHT_COLOR_LIGHT);
	theme->set_color("font_color_hover", "ToolButton", HIGHLIGHT_COLOR_LIGHT);
	theme->set_color("font_color_pressed", "ToolButton", highlight_color);

	theme->set_stylebox("MenuHover", "EditorStyles", style_menu_hover_border);

	// Content of each tab
	Ref<StyleBoxFlat> style_content_panel = make_flat_stylebox(base_color, 4, 5, 4, 4);
	style_content_panel->set_dark_color(title_color_hl);
	style_content_panel->set_light_color(title_color_hl);
	style_content_panel->set_border_size(border_width);
	style_content_panel->set_border_blend(false);
	Ref<StyleBoxFlat> style_content_panel_vp = make_flat_stylebox(base_color, border_width, 5, border_width, border_width);
	style_content_panel_vp->set_dark_color(title_color_hl);
	style_content_panel_vp->set_light_color(title_color_hl);
	style_content_panel_vp->set_border_size(border_width);
	style_content_panel_vp->set_border_blend(false);
	theme->set_stylebox("panel", "TabContainer", style_content_panel);
	theme->set_stylebox("Content", "EditorStyles", style_content_panel_vp);

	Ref<StyleBoxFlat> style_button_type = make_flat_stylebox(dark_color_1, 4, 4, 6, 4);
	style_button_type->set_draw_center(true);
	style_button_type->set_border_size(border_width);
	style_button_type->set_light_color(light_color_1);
	style_button_type->set_dark_color(light_color_1);
	style_button_type->set_border_blend(false);

	Ref<StyleBoxFlat> style_button_type_disabled = change_border_color(style_button_type, dark_color_2);

	Color button_font_color = light_color_1.linear_interpolate(Color(1, 1, 1, 1), .6);

	// Button
	theme->set_stylebox("normal", "Button", style_button_type);
	theme->set_stylebox("hover", "Button", change_border_color(style_button_type, HIGHLIGHT_COLOR_LIGHT));
	theme->set_stylebox("pressed", "Button", change_border_color(style_button_type, highlight_color));
	theme->set_stylebox("focus", "Button", change_border_color(style_button_type, highlight_color));
	theme->set_stylebox("disabled", "Button", style_button_type_disabled);
	theme->set_color("font_color", "Button", button_font_color);
	theme->set_color("font_color_hover", "Button", HIGHLIGHT_COLOR_LIGHT);
	theme->set_color("font_color_pressed", "Button", highlight_color);
	theme->set_color("icon_color_hover", "Button", HIGHLIGHT_COLOR_LIGHT);
	// make icon color value bigger because icon image is not complete white
	theme->set_color("icon_color_pressed", "Button", Color(highlight_color.r * 1.15, highlight_color.g * 1.15, highlight_color.b * 1.15, highlight_color.a));

	// OptionButton
	Ref<StyleBoxFlat> style_option_button = make_flat_stylebox(dark_color_1, 4, 4, 8, 4);
	style_option_button->set_border_size(border_width);
	style_option_button->set_light_color(light_color_1);
	style_option_button->set_dark_color(light_color_1);
	style_option_button->set_border_blend(false);
	theme->set_stylebox("hover", "OptionButton", change_border_color(style_button_type, HIGHLIGHT_COLOR_LIGHT));
	theme->set_stylebox("pressed", "OptionButton", change_border_color(style_button_type, highlight_color));
	theme->set_stylebox("focus", "OptionButton", change_border_color(style_button_type, highlight_color));
	theme->set_stylebox("disabled", "OptionButton", style_button_type_disabled);
	theme->set_stylebox("normal", "OptionButton", style_button_type);
	theme->set_color("font_color", "OptionButton", button_font_color);
	theme->set_color("font_color_hover", "OptionButton", HIGHLIGHT_COLOR_LIGHT);
	theme->set_color("font_color_pressed", "OptionButton", highlight_color);
	theme->set_color("icon_color_hover", "OptionButton", HIGHLIGHT_COLOR_LIGHT);
	theme->set_icon("arrow", "OptionButton", theme->get_icon("GuiOptionArrow", "EditorIcons"));
	theme->set_constant("arrow_margin", "OptionButton", 4);
	theme->set_constant("modulate_arrow", "OptionButton", true);

	// CheckButton
	theme->set_icon("on", "CheckButton", theme->get_icon("GuiToggleOn", "EditorIcons"));
	theme->set_icon("off", "CheckButton", theme->get_icon("GuiToggleOff", "EditorIcons"));

	// PopupMenu
	Ref<StyleBoxFlat> style_popup_menu = make_flat_stylebox(dark_color_1, 8, 8, 8, 8);
	style_popup_menu->set_border_size(border_width);
	style_popup_menu->set_light_color(light_color_1);
	style_popup_menu->set_dark_color(light_color_1);
	style_popup_menu->set_border_blend(false);
	theme->set_stylebox("panel", "PopupMenu", style_popup_menu);
	theme->set_stylebox("separator", "PopupMenu", make_line_stylebox(separator_color, border_width, 8 - border_width));

	// Tree & ItemList background
	Ref<StyleBoxFlat> style_tree_bg = make_flat_stylebox(dark_color_1, 2, 4, 2, 4);
	style_tree_bg->set_border_size(border_width);
	style_tree_bg->set_light_color(dark_color_3);
	style_tree_bg->set_dark_color(dark_color_3);
	theme->set_stylebox("bg", "Tree", style_tree_bg);
	// Script background
	Ref<StyleBoxFlat> style_script_bg = make_flat_stylebox(dark_color_1, 0, 0, 0, 0);
	style_script_bg->set_border_size(border_width);
	style_script_bg->set_light_color(dark_color_3);
	style_script_bg->set_dark_color(dark_color_3);
	theme->set_stylebox("ScriptPanel", "EditorStyles", style_script_bg);

	// Tree
	theme->set_icon("checked", "Tree", theme->get_icon("GuiChecked", "EditorIcons"));
	theme->set_icon("unchecked", "Tree", theme->get_icon("GuiUnchecked", "EditorIcons"));
	theme->set_icon("arrow", "Tree", theme->get_icon("GuiTreeArrowDown", "EditorIcons"));
	theme->set_icon("arrow_collapsed", "Tree", theme->get_icon("GuiTreeArrowRight", "EditorIcons"));
	theme->set_icon("select_arrow", "Tree", theme->get_icon("GuiDropdown", "EditorIcons"));
	theme->set_stylebox("bg_focus", "Tree", focus_sbt);
	theme->set_stylebox("custom_button", "Tree", style_button_type);
	theme->set_stylebox("custom_button_pressed", "Tree", style_button_type);
	theme->set_stylebox("custom_button_hover", "Tree", style_button_type);
	theme->set_color("custom_button_font_highlight", "Tree", HIGHLIGHT_COLOR_LIGHT);

	Ref<StyleBox> style_tree_btn = make_flat_stylebox(light_color_1, 2, 4, 2, 4);
	theme->set_stylebox("button_pressed", "Tree", style_tree_btn);

	Ref<StyleBoxFlat> style_tree_focus = make_flat_stylebox(HIGHLIGHT_COLOR_DARK, 2, 2, 2, 2);
	theme->set_stylebox("selected_focus", "Tree", style_tree_focus);

	Ref<StyleBoxFlat> style_tree_selected = make_flat_stylebox(HIGHLIGHT_COLOR_DARK, 2, 2, 2, 2);
	theme->set_stylebox("selected", "Tree", style_tree_selected);

	Ref<StyleBoxFlat> style_tree_cursor = make_flat_stylebox(HIGHLIGHT_COLOR_DARK, 4, 4, 4, 4);
	style_tree_cursor->set_draw_center(false);
	style_tree_cursor->set_border_size(border_width);
	style_tree_cursor->set_light_color(light_color_1);
	style_tree_cursor->set_dark_color(light_color_1);
	Ref<StyleBoxFlat> style_tree_title = make_flat_stylebox(dark_color_3, 4, 4, 4, 4);
	theme->set_stylebox("cursor", "Tree", style_tree_cursor);
	theme->set_stylebox("cursor_unfocused", "Tree", style_tree_cursor);
	theme->set_stylebox("title_button_normal", "Tree", style_tree_title);
	theme->set_stylebox("title_button_hover", "Tree", style_tree_title);
	theme->set_stylebox("title_button_pressed", "Tree", style_tree_title);

	theme->set_color("prop_category", "Editor", dark_color_1.linear_interpolate(Color(1, 1, 1, 1), 0.12));
	theme->set_color("prop_section", "Editor", dark_color_1.linear_interpolate(Color(1, 1, 1, 1), 0.09));
	theme->set_color("prop_subsection", "Editor", dark_color_1.linear_interpolate(Color(1, 1, 1, 1), 0.06));
	theme->set_color("fg_selected", "Editor", HIGHLIGHT_COLOR_DARK);
	theme->set_color("fg_error", "Editor", Color::html("ffbd8e8e"));
	theme->set_color("drop_position_color", "Tree", highlight_color);

	// ItemList
	Ref<StyleBoxFlat> style_itemlist_bg = make_flat_stylebox(dark_color_1, 4, 4, 4, 4);
	style_itemlist_bg->set_border_size(border_width);
	style_itemlist_bg->set_light_color(dark_color_3);
	style_itemlist_bg->set_dark_color(dark_color_3);
	Ref<StyleBoxFlat> style_itemlist_cursor = make_flat_stylebox(highlight_color, 0, 0, 0, 0);
	style_itemlist_cursor->set_draw_center(false);
	style_itemlist_cursor->set_border_size(border_width);
	style_itemlist_cursor->set_light_color(HIGHLIGHT_COLOR_DARK);
	style_itemlist_cursor->set_dark_color(HIGHLIGHT_COLOR_DARK);
	theme->set_stylebox("cursor", "ItemList", style_itemlist_cursor);
	theme->set_stylebox("cursor_unfocused", "ItemList", style_itemlist_cursor);
	theme->set_stylebox("selected_focus", "ItemList", style_tree_focus);
	theme->set_stylebox("selected", "ItemList", style_tree_selected);
	theme->set_stylebox("bg_focus", "ItemList", focus_sbt);
	theme->set_stylebox("bg", "ItemList", style_itemlist_bg);
	theme->set_constant("vseparation", "ItemList", 5 * EDSCALE);

	Ref<StyleBoxFlat> style_tab_fg = make_flat_stylebox(title_color_hl, 15, 5, 15, 5);
	Ref<StyleBoxFlat> style_tab_bg = make_flat_stylebox(base_color, 15, 5, 15, 5);
	style_tab_bg->set_draw_center(false);

	// Tabs & TabContainer
	theme->set_stylebox("tab_fg", "TabContainer", style_tab_fg);
	theme->set_stylebox("tab_bg", "TabContainer", style_tab_bg);
	theme->set_stylebox("tab_fg", "Tabs", style_tab_fg);
	theme->set_stylebox("tab_bg", "Tabs", style_tab_bg);
	theme->set_color("font_color_fg", "TabContainer", title_color_hl_text_color);
	theme->set_color("font_color_bg", "TabContainer", light_color_2);
	theme->set_icon("menu", "TabContainer", theme->get_icon("GuiTabMenu", "EditorIcons"));
	theme->set_icon("menu_hl", "TabContainer", theme->get_icon("GuiTabMenu", "EditorIcons"));
	theme->set_stylebox("SceneTabFG", "EditorStyles", make_flat_stylebox(title_color_hl, 10, 5, 10, 5));
	theme->set_stylebox("SceneTabBG", "EditorStyles", make_empty_stylebox(6, 5, 6, 5));
	theme->set_icon("close", "Tabs", title_hl_close_icon);

	// Separatos (no separatos)
	theme->set_stylebox("separator", "HSeparator", make_line_stylebox(separator_color, border_width));
	theme->set_stylebox("separator", "VSeparator", make_line_stylebox(separator_color, border_width, 0, true));

	// Debugger
	Ref<StyleBoxFlat> style_panel_debugger = make_flat_stylebox(dark_color_2, 0, 4, 0, 0);
	theme->set_stylebox("DebuggerPanel", "EditorStyles", style_panel_debugger);

	Ref<StyleBoxFlat> style_tab_fg_debugger = make_flat_stylebox(dark_color_2, 10, 5, 10, 5);
	Ref<StyleBoxFlat> style_tab_bg_debugger = make_flat_stylebox(dark_color_2, 10, 5, 10, 5);
	style_tab_bg_debugger->set_draw_center(false);

	theme->set_stylebox("DebuggerTabFG", "EditorStyles", style_tab_fg_debugger);
	theme->set_stylebox("DebuggerTabBG", "EditorStyles", style_tab_bg_debugger);

	// LineEdit
	Ref<StyleBoxFlat> style_lineedit = style_button_type;
	style_lineedit = change_border_color(style_lineedit, light_color_1);
	Ref<StyleBoxFlat> style_lineedit_disabled = change_border_color(style_lineedit, dark_color_1);
	style_lineedit_disabled->set_bg_color(Color(0, 0, 0, .1));
	Ref<StyleBoxFlat> style_lineedit_focus = change_border_color(style_lineedit, highlight_color);
	theme->set_stylebox("normal", "LineEdit", style_lineedit);
	theme->set_stylebox("focus", "LineEdit", style_lineedit_focus);
	theme->set_stylebox("read_only", "LineEdit", style_lineedit_disabled);
	theme->set_color("read_only", "LineEdit", dark_color_1);

	// TextEdit
	Ref<StyleBoxFlat> style_textedit_normal(memnew(StyleBoxFlat));
	style_textedit_normal->set_bg_color(dark_color_2);
	style_textedit_normal->set_default_margin(MARGIN_LEFT, 0);
	style_textedit_normal->set_default_margin(MARGIN_RIGHT, 0);
	style_textedit_normal->set_default_margin(MARGIN_BOTTOM, 0);
	style_textedit_normal->set_default_margin(MARGIN_TOP, 0);
	theme->set_stylebox("normal", "TextEdit", style_textedit_normal);
	theme->set_stylebox("focus", "TextEdit", focus_sbt);
	theme->set_constant("side_margin", "TabContainer", 0);

	// H/VSplitContainer
	theme->set_stylebox("bg", "VSplitContainer", make_stylebox(theme->get_icon("GuiVsplitBg", "EditorIcons"), 1, 1, 1, 1));
	theme->set_stylebox("bg", "HSplitContainer", make_stylebox(theme->get_icon("GuiHsplitBg", "EditorIcons"), 1, 1, 1, 1));

	theme->set_icon("grabber", "VSplitContainer", theme->get_icon("GuiVsplitter", "EditorIcons"));
	theme->set_icon("grabber", "HSplitContainer", theme->get_icon("GuiHsplitter", "EditorIcons"));

	theme->set_constant("separation", "HSplitContainer", 8 * EDSCALE);
	theme->set_constant("separation", "VSplitContainer", 8 * EDSCALE);

	// WindowDialog
	Ref<StyleBoxFlat> style_window = make_flat_stylebox(dark_color_2, 4, 4, 4, 4);
	style_window->set_border_size(border_width);
	style_window->set_border_blend(false);
	style_window->set_light_color(title_color_hl);
	style_window->set_dark_color(title_color_hl);
	style_window->_set_additional_border_size(MARGIN_TOP, 24 * EDSCALE);
	theme->set_stylebox("panel", "WindowDialog", style_window);
	theme->set_color("title_color", "WindowDialog", title_color_hl_text_color);
	theme->set_icon("close", "WindowDialog", title_hl_close_icon);
	theme->set_icon("close_highlight", "WindowDialog", title_hl_close_icon);
	theme->set_constant("close_h_ofs", "WindowDialog", 22 * EDSCALE);
	theme->set_constant("close_v_ofs", "WindowDialog", 20 * EDSCALE);
	theme->set_constant("title_height", "WindowDialog", 24 * EDSCALE);

	// HScrollBar
	Ref<Texture> empty_icon = memnew(ImageTexture);

	theme->set_stylebox("scroll", "HScrollBar", make_stylebox(theme->get_icon("GuiScrollBg", "EditorIcons"), 5, 5, 5, 5, 0, 0, 0, 0));
	theme->set_stylebox("scroll_focus", "HScrollBar", make_stylebox(theme->get_icon("GuiScrollBg", "EditorIcons"), 5, 5, 5, 5, 0, 0, 0, 0));
	theme->set_stylebox("grabber", "HScrollBar", make_stylebox(theme->get_icon("GuiScrollGrabber", "EditorIcons"), 6, 6, 6, 6, 2, 2, 2, 2));
	theme->set_stylebox("grabber_highlight", "HScrollBar", make_stylebox(theme->get_icon("GuiScrollGrabberHl", "EditorIcons"), 5, 5, 5, 5, 2, 2, 2, 2));

	theme->set_icon("increment", "HScrollBar", empty_icon);
	theme->set_icon("increment_highlight", "HScrollBar", empty_icon);
	theme->set_icon("decrement", "HScrollBar", empty_icon);
	theme->set_icon("decrement_highlight", "HScrollBar", empty_icon);

	// VScrollBar
	theme->set_stylebox("scroll", "VScrollBar", make_stylebox(theme->get_icon("GuiScrollBg", "EditorIcons"), 5, 5, 5, 5, 0, 0, 0, 0));
	theme->set_stylebox("scroll_focus", "VScrollBar", make_stylebox(theme->get_icon("GuiScrollBg", "EditorIcons"), 5, 5, 5, 5, 0, 0, 0, 0));
	theme->set_stylebox("grabber", "VScrollBar", make_stylebox(theme->get_icon("GuiScrollGrabber", "EditorIcons"), 6, 6, 6, 6, 2, 2, 2, 2));
	theme->set_stylebox("grabber_highlight", "VScrollBar", make_stylebox(theme->get_icon("GuiScrollGrabberHl", "EditorIcons"), 5, 5, 5, 5, 2, 2, 2, 2));

	theme->set_icon("increment", "VScrollBar", empty_icon);
	theme->set_icon("increment_highlight", "VScrollBar", empty_icon);
	theme->set_icon("decrement", "VScrollBar", empty_icon);
	theme->set_icon("decrement_highlight", "VScrollBar", empty_icon);

	// HSlider
	theme->set_stylebox("slider", "HSlider", make_stylebox(theme->get_icon("GuiHsliderBg", "EditorIcons"), 4, 4, 4, 4));
	theme->set_icon("grabber", "HSlider", theme->get_icon("GuiSliderGrabber", "EditorIcons"));
	theme->set_icon("grabber_highlight", "HSlider", theme->get_icon("GuiSliderGrabberHl", "EditorIcons"));

	// VSlider
	theme->set_stylebox("slider", "VSlider", make_stylebox(theme->get_icon("GuiVsliderBg", "EditorIcons"), 4, 4, 4, 4));
	theme->set_icon("grabber", "VSlider", theme->get_icon("GuiSliderGrabber", "EditorIcons"));
	theme->set_icon("grabber_highlight", "VSlider", theme->get_icon("GuiSliderGrabberHl", "EditorIcons"));

	// Panel
	theme->set_stylebox("panel", "Panel", style_panel);

	// TooltipPanel
	Ref<StyleBoxFlat> style_tooltip = make_flat_stylebox(Color(1, 1, 1, 0.8), 8, 8, 8, 8);
	style_tooltip->set_border_size(border_width);
	style_tooltip->set_border_blend(false);
	style_tooltip->set_light_color(Color(1, 1, 1, 0.9));
	style_tooltip->set_dark_color(Color(1, 1, 1, 0.9));
	theme->set_stylebox("panel", "TooltipPanel", style_tooltip);

	// PopupPanel
	Ref<StyleBoxFlat> style_dock_select = make_flat_stylebox(base_color);
	style_dock_select->set_light_color(light_color_1);
	style_dock_select->set_dark_color(light_color_1);
	style_dock_select = add_additional_border(style_dock_select, 2, 2, 2, 2);
	theme->set_stylebox("panel", "PopupPanel", style_dock_select);

	// SpinBox
	theme->set_icon("updown", "SpinBox", theme->get_icon("GuiSpinboxUpdown", "EditorIcons"));

	// ProgressBar
	theme->set_stylebox("bg", "ProgressBar", make_stylebox(theme->get_icon("GuiProgressBar", "EditorIcons"), 4, 4, 4, 4, 0, 0, 0, 0));
	theme->set_stylebox("fg", "ProgressBar", make_stylebox(theme->get_icon("GuiProgressFill", "EditorIcons"), 6, 6, 6, 6, 2, 1, 2, 1));

	// theme->set_font("font", "ProgressBar", default_font);

	// theme->set_color("font_color", "ProgressBar", control_font_color_hover);
	theme->set_color("font_color_shadow", "ProgressBar", Color(0, 0, 0));

	// GraphEdit
	theme->set_stylebox("bg", "GraphEdit", make_flat_stylebox(dark_color_2, 4, 4, 4, 4));

	// GraphNode
	Ref<StyleBoxFlat> graphsb = make_flat_stylebox(Color(0, 0, 0, 0.3), 16, 24, 16, 5);
	graphsb->set_border_blend(false);
	graphsb->set_border_size(border_width);
	graphsb->set_light_color(Color(1, 1, 1, 0.6));
	graphsb->set_dark_color(Color(1, 1, 1, 0.6));
	graphsb = add_additional_border(graphsb, 0, -22, 0, 0);
	Ref<StyleBoxFlat> graphsbselected = make_flat_stylebox(Color(0, 0, 0, 0.4), 16, 24, 16, 5);
	graphsbselected->set_border_blend(false);
	graphsbselected->set_border_size(border_width);
	graphsbselected->set_light_color(Color(1, 1, 1, 0.9));
	graphsbselected->set_dark_color(Color(1, 1, 1, 0.9));
	graphsbselected = add_additional_border(graphsbselected, 0, -22, 0, 0);
	Ref<StyleBoxFlat> graphsbcomment = make_flat_stylebox(Color(0, 0, 0, 0.3), 16, 24, 16, 5);
	graphsbcomment->set_border_blend(false);
	graphsbcomment->set_border_size(border_width);
	graphsbcomment->set_light_color(Color(1, 1, 1, 0.6));
	graphsbcomment->set_dark_color(Color(1, 1, 1, 0.6));
	graphsbcomment = add_additional_border(graphsbcomment, 0, -22, 0, 0);
	Ref<StyleBoxFlat> graphsbcommentselected = make_flat_stylebox(Color(0, 0, 0, 0.4), 16, 24, 16, 5);
	graphsbcommentselected->set_border_blend(false);
	graphsbcommentselected->set_border_size(border_width);
	graphsbcommentselected->set_light_color(Color(1, 1, 1, 0.9));
	graphsbcommentselected->set_dark_color(Color(1, 1, 1, 0.9));
	graphsbcommentselected = add_additional_border(graphsbcommentselected, 0, -22, 0, 0);
	theme->set_stylebox("frame", "GraphNode", graphsb);
	theme->set_stylebox("selectedframe", "GraphNode", graphsbselected);
	theme->set_stylebox("comment", "GraphNode", graphsbcomment);
	theme->set_stylebox("commentfocus", "GraphNode", graphsbcommentselected);

	// FileDialog
	Color disable_color = light_color_2;
	disable_color.a = 0.7;
	theme->set_color("files_disabled", "FileDialog", disable_color);

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
