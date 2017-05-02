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

Ref<Theme> create_editor_theme() {
	Ref<Theme> theme = Ref<Theme>(memnew(Theme));

	editor_register_fonts(theme);
	editor_register_icons(theme);

	Ref<StyleBoxTexture> focus_sbt = memnew(StyleBoxTexture);
	focus_sbt->set_texture(theme->get_icon("EditorFocus", "EditorIcons"));
	for (int i = 0; i < 4; i++) {
		focus_sbt->set_margin_size(Margin(i), 16 * EDSCALE);
		focus_sbt->set_default_margin(Margin(i), 16 * EDSCALE);
	}
	focus_sbt->set_draw_center(false);
	theme->set_stylebox("EditorFocus", "EditorStyles", focus_sbt);

	Ref<StyleBoxFlat> style_panel(memnew(StyleBoxFlat));
	style_panel->set_bg_color(Color::html("#36424e"));
	style_panel->set_default_margin(MARGIN_LEFT, 1);
	style_panel->set_default_margin(MARGIN_RIGHT, 1);
	style_panel->set_default_margin(MARGIN_BOTTOM, 1);
	style_panel->set_default_margin(MARGIN_TOP, 4 * EDSCALE);
	theme->set_stylebox("panel", "TabContainer", style_panel);
	theme->set_stylebox("EditorPanel", "EditorStyles", style_panel);

	Ref<StyleBoxFlat> style_bg(memnew(StyleBoxFlat));
	style_bg->set_bg_color(Color::html("#2b353f"));
	style_bg->set_default_margin(MARGIN_LEFT, 0);
	style_bg->set_default_margin(MARGIN_RIGHT, 0);
	style_bg->set_default_margin(MARGIN_BOTTOM, 0);
	style_bg->set_default_margin(MARGIN_TOP, 0);
	theme->set_stylebox("bg", "Tree", style_bg);
	theme->set_stylebox("bg", "ItemList", style_bg);
	theme->set_stylebox("EditorBG", "EditorStyles", style_bg);

	Ref<StyleBox> style_tree_btn = theme->get_stylebox("button_pressed", "Tree");
	style_tree_btn->set_default_margin(MARGIN_LEFT, 3 * EDSCALE);
	style_tree_btn->set_default_margin(MARGIN_RIGHT, 3 * EDSCALE);
	theme->set_stylebox("button_pressed", "Tree", style_tree_btn);

	Ref<StyleBoxFlat> style_tab(memnew(StyleBoxFlat));
	style_tab->set_default_margin(MARGIN_LEFT, 15 * EDSCALE);
	style_tab->set_default_margin(MARGIN_RIGHT, 15 * EDSCALE);
	style_tab->set_default_margin(MARGIN_BOTTOM, 5 * EDSCALE);
	style_tab->set_default_margin(MARGIN_TOP, 5 * EDSCALE);

	Ref<StyleBoxFlat> style_tab_fg = style_tab->duplicate();
	style_tab_fg->set_bg_color(Color::html("#36424e"));

	Ref<StyleBoxFlat> style_tab_bg = style_tab->duplicate();
	style_tab_bg->set_draw_center(false);

	theme->set_stylebox("tab_fg", "TabContainer", style_tab_fg);
	theme->set_stylebox("tab_bg", "TabContainer", style_tab_bg);
	theme->set_stylebox("tab_fg", "Tabs", style_tab_fg);
	theme->set_stylebox("tab_bg", "Tabs", style_tab_bg);

	Ref<StyleBoxFlat> style_panel_debugger(memnew(StyleBoxFlat));
	style_panel_debugger->set_bg_color(Color::html("#3e4c5a"));
	style_panel_debugger->set_default_margin(MARGIN_LEFT, 0);
	style_panel_debugger->set_default_margin(MARGIN_RIGHT, 0);
	style_panel_debugger->set_default_margin(MARGIN_BOTTOM, 0);
	style_panel_debugger->set_default_margin(MARGIN_TOP, 4 * EDSCALE);
	theme->set_stylebox("EditorPanelDebugger", "EditorStyles", style_panel_debugger);

	Ref<StyleBoxFlat> style_tab_fg_debugger = style_tab->duplicate();
	style_tab_fg_debugger->set_bg_color(Color::html("#3e4c5a"));
	style_tab_fg_debugger->set_default_margin(MARGIN_LEFT, 10 * EDSCALE);
	style_tab_fg_debugger->set_default_margin(MARGIN_RIGHT, 10 * EDSCALE);
	Ref<StyleBoxFlat> style_tab_bg_debugger = style_tab->duplicate();
	style_tab_bg_debugger->set_draw_center(false);
	style_tab_bg_debugger->set_default_margin(MARGIN_LEFT, 10 * EDSCALE);
	style_tab_bg_debugger->set_default_margin(MARGIN_RIGHT, 10 * EDSCALE);

	theme->set_stylebox("EditorTabFGDebugger", "EditorStyles", style_tab_fg_debugger);
	theme->set_stylebox("EditorTabBGDebugger", "EditorStyles", style_tab_bg_debugger);

	Ref<StyleBoxFlat> style_textedit_normal(memnew(StyleBoxFlat));
	style_textedit_normal->set_bg_color(Color::html("#29343d"));
	style_textedit_normal->set_default_margin(MARGIN_LEFT, 0);
	style_textedit_normal->set_default_margin(MARGIN_RIGHT, 0);
	style_textedit_normal->set_default_margin(MARGIN_BOTTOM, 0);
	style_textedit_normal->set_default_margin(MARGIN_TOP, 0);
	theme->set_stylebox("normal", "TextEdit", style_textedit_normal);

	theme->set_constant("separation", "HSplitContainer", 8 * EDSCALE);
	theme->set_constant("separation", "VSplitContainer", 8 * EDSCALE);
	theme->set_constant("side_margin", "TabContainer", 0);

	// theme->set_color("prop_category","Editor",Color::hex(0x3f3a44ff));
	// theme->set_color("prop_section","Editor",Color::hex(0x35313aff));
	// theme->set_color("prop_subsection","Editor",Color::hex(0x312e37ff));
	// theme->set_color("fg_selected","Editor",Color::html("ffbd8e8e"));
	// theme->set_color("fg_error","Editor",Color::html("ffbd8e8e"));

	return theme;
}

Ref<Theme> create_custom_theme() {
	Ref<Theme> theme;

	String custom_theme = EditorSettings::get_singleton()->get("interface/custom_theme");
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
