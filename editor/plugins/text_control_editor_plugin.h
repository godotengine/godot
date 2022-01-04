/*************************************************************************/
/*  text_control_editor_plugin.h                                         */
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

#ifndef TEXT_CONTROL_EDITOR_PLUGIN_H
#define TEXT_CONTROL_EDITOR_PLUGIN_H

#include "canvas_item_editor_plugin.h"
#include "editor/editor_file_system.h"
#include "editor/editor_inspector.h"
#include "editor/editor_node.h"
#include "editor/editor_plugin.h"
#include "scene/gui/color_rect.h"
#include "scene/gui/menu_button.h"
#include "scene/gui/option_button.h"
#include "scene/gui/popup_menu.h"

/*************************************************************************/

class TextControlEditor : public HBoxContainer {
	GDCLASS(TextControlEditor, HBoxContainer);

	enum FontInfoID {
		FONT_INFO_THEME_DEFAULT = 0,
		FONT_INFO_USER_CUSTOM = 1,
		FONT_INFO_ID = 100,
	};

	Map<String, Map<String, String>> fonts;

	OptionButton *font_list = nullptr;
	SpinBox *font_size_list = nullptr;
	OptionButton *font_style_list = nullptr;
	ColorPickerButton *font_color_picker = nullptr;
	SpinBox *outline_size_list = nullptr;
	ColorPickerButton *outline_color_picker = nullptr;
	Button *clear_formatting = nullptr;

	Control *edited_control = nullptr;
	String edited_color;
	String edited_font;
	String edited_font_size;
	Ref<Font> custom_font;

protected:
	void _notification(int p_notification);
	static void _bind_methods(){};

	void _find_resources(EditorFileSystemDirectory *p_dir);
	void _reload_fonts(const String &p_path);

	void _update_fonts_menu();
	void _update_styles_menu();
	void _update_control();

	void _font_selected(int p_id);
	void _font_style_selected(int p_id);
	void _set_font();

	void _font_size_selected(double p_size);
	void _outline_size_selected(double p_size);

	void _font_color_changed(const Color &p_color);
	void _outline_color_changed(const Color &p_color);

	void _clear_formatting();

public:
	void edit(Object *p_object);
	bool handles(Object *p_object) const;

	TextControlEditor();
};

/*************************************************************************/

class TextControlEditorPlugin : public EditorPlugin {
	GDCLASS(TextControlEditorPlugin, EditorPlugin);

	TextControlEditor *text_ctl_editor;
	EditorNode *editor;

public:
	virtual String get_name() const override { return "TextControlFontEditor"; }
	bool has_main_screen() const override { return false; }
	virtual void edit(Object *p_object) override;
	virtual bool handles(Object *p_object) const override;
	virtual void make_visible(bool p_visible) override;

	TextControlEditorPlugin(EditorNode *p_node);
};

#endif // TEXT_CONTROL_EDITOR_PLUGIN_H
