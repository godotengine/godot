/*************************************************************************/
/*  theme_editor_plugin.h                                                */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "scene/gui/check_box.h"
#include "scene/gui/file_dialog.h"
#include "scene/gui/margin_container.h"
#include "scene/gui/option_button.h"
#include "scene/gui/scroll_container.h"
#include "scene/gui/texture_rect.h"
#include "scene/resources/theme.h"

#include "editor/editor_node.h"

class ThemeItemEditorDialog : public AcceptDialog {
	GDCLASS(ThemeItemEditorDialog, AcceptDialog);

	Ref<Theme> edited_theme;

	ItemList *edit_type_list;
	OptionButton *edit_add_class_options;
	LineEdit *edit_add_custom_value;
	String edited_item_type;

	Button *edit_items_add_color;
	Button *edit_items_add_constant;
	Button *edit_items_add_font;
	Button *edit_items_add_font_size;
	Button *edit_items_add_icon;
	Button *edit_items_add_stylebox;
	Button *edit_items_remove_class;
	Button *edit_items_remove_custom;
	Button *edit_items_remove_all;
	Tree *edit_items_tree;

	enum ItemsTreeAction {
		ITEMS_TREE_RENAME_ITEM,
		ITEMS_TREE_REMOVE_ITEM,
		ITEMS_TREE_REMOVE_DATA_TYPE,
	};

	ConfirmationDialog *edit_theme_item_dialog;
	VBoxContainer *edit_theme_item_old_vb;
	Label *theme_item_old_name;
	LineEdit *theme_item_name;

	enum ItemPopupMode {
		CREATE_THEME_ITEM,
		RENAME_THEME_ITEM,
		ITEM_POPUP_MODE_MAX
	};

	ItemPopupMode item_popup_mode = ITEM_POPUP_MODE_MAX;
	String edit_item_old_name;
	Theme::DataType edit_item_data_type = Theme::DATA_TYPE_MAX;

	void _dialog_about_to_show();
	void _update_edit_types();
	void _edited_type_selected(int p_item_idx);

	void _update_edit_item_tree(String p_item_type);
	void _item_tree_button_pressed(Object *p_item, int p_column, int p_id);

	void _add_class_type_items();
	void _add_custom_type();
	void _add_theme_item(Theme::DataType p_data_type, String p_item_name, String p_item_type);
	void _remove_data_type_items(Theme::DataType p_data_type, String p_item_type);
	void _remove_class_items();
	void _remove_custom_items();
	void _remove_all_items();

	void _open_add_theme_item_dialog(int p_data_type);
	void _open_rename_theme_item_dialog(Theme::DataType p_data_type, String p_item_name);
	void _confirm_edit_theme_item();
	void _edit_theme_item_gui_input(const Ref<InputEvent> &p_event);

protected:
	void _notification(int p_what);

public:
	void set_edited_theme(const Ref<Theme> &p_theme);

	ThemeItemEditorDialog();
};

class ThemeEditor : public VBoxContainer {
	GDCLASS(ThemeEditor, VBoxContainer);

	Panel *main_panel;
	MarginContainer *main_container;
	Ref<Theme> theme;

	EditorFileDialog *file_dialog;

	double time_left = 0;

	Button *theme_edit_button;
	MenuButton *theme_create_menu;
	ThemeItemEditorDialog *theme_edit_dialog;

	enum CreatePopupMode {
		POPUP_CREATE_EMPTY,
		POPUP_CREATE_EDITOR_EMPTY,
		POPUP_IMPORT_EDITOR_THEME,
	};

	Tree *test_tree;

	void _save_template_cbk(String fname);
	void _theme_edit_button_cbk();
	void _theme_create_menu_cbk(int p_option);
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
	virtual String get_name() const override { return "Theme"; }
	bool has_main_screen() const override { return false; }
	virtual void edit(Object *p_node) override;
	virtual bool handles(Object *p_node) const override;
	virtual void make_visible(bool p_visible) override;

	ThemeEditorPlugin(EditorNode *p_node);
};

#endif // THEME_EDITOR_PLUGIN_H
