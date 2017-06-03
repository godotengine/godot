/*************************************************************************/
/*  theme_editor_plugin.h                                                */
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
#ifndef THEME_EDITOR_PLUGIN_H
#define THEME_EDITOR_PLUGIN_H

#include "scene/gui/button_group.h"
#include "scene/gui/check_box.h"
#include "scene/gui/file_dialog.h"
#include "scene/gui/option_button.h"
#include "scene/gui/scroll_container.h"
#include "scene/gui/texture_rect.h"
#include "scene/resources/theme.h"

#include "editor/editor_node.h"

class ThemeEditor : public Control {

	GDCLASS(ThemeEditor, Control);

	ScrollContainer *scroll;
	VBoxContainer *main_vb;
	Ref<Theme> theme;

	EditorFileDialog *file_dialog;

	double time_left;

	MenuButton *theme_menu;
	ConfirmationDialog *add_del_dialog;
	MenuButton *type_menu;
	LineEdit *type_edit;
	MenuButton *name_menu;
	LineEdit *name_edit;
	OptionButton *type_select;
	Label *type_select_label;
	Label *name_select_label;
	Label *dtype_select_label;

	enum PopupMode {
		POPUP_ADD,
		POPUP_CLASS_ADD,
		POPUP_REMOVE,
		POPUP_CLASS_REMOVE,
		POPUP_CREATE_EMPTY,
		POPUP_CREATE_EDITOR_EMPTY
	};

	int popup_mode;

	Tree *test_tree;

	void _save_template_cbk(String fname);
	void _dialog_cbk();
	void _type_menu_cbk(int p_option);
	void _name_menu_about_to_show();
	void _name_menu_cbk(int p_option);
	void _theme_menu_cbk(int p_option);
	void _propagate_redraw(Control *p_at);
	void _refresh_interval();

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void edit(const Ref<Theme> &p_theme);

	ThemeEditor();
};

class ThemeEditorPlugin : public EditorPlugin {

	GDCLASS(ThemeEditorPlugin, EditorPlugin);

	ThemeEditor *theme_editor;
	EditorNode *editor;
	Button *button;

public:
	virtual String get_name() const { return "Theme"; }
	bool has_main_screen() const { return false; }
	virtual void edit(Object *p_node);
	virtual bool handles(Object *p_node) const;
	virtual void make_visible(bool p_visible);

	ThemeEditorPlugin(EditorNode *p_node);
};

#endif // THEME_EDITOR_PLUGIN_H
