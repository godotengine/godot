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

class ThemeItemImportTree : public VBoxContainer {
	GDCLASS(ThemeItemImportTree, VBoxContainer);

	Ref<Theme> edited_theme;
	Ref<Theme> base_theme;

	struct ThemeItem {
		String type_name;
		Theme::DataType data_type;
		String item_name;

		bool operator<(const ThemeItem &p_item) const {
			if (type_name == p_item.type_name && data_type == p_item.data_type) {
				return item_name < p_item.item_name;
			}
			if (type_name == p_item.type_name) {
				return data_type < p_item.data_type;
			}
			return type_name < p_item.type_name;
		}
	};

	enum ItemCheckedState {
		SELECT_IMPORT_DEFINITION,
		SELECT_IMPORT_FULL,
	};

	Map<ThemeItem, ItemCheckedState> selected_items;

	LineEdit *import_items_filter;

	Tree *import_items_tree;
	List<TreeItem *> tree_color_items;
	List<TreeItem *> tree_constant_items;
	List<TreeItem *> tree_font_items;
	List<TreeItem *> tree_font_size_items;
	List<TreeItem *> tree_icon_items;
	List<TreeItem *> tree_stylebox_items;

	bool updating_tree = false;

	enum ItemActionFlag {
		IMPORT_ITEM = 1,
		IMPORT_ITEM_DATA = 2,
	};

	TextureRect *select_colors_icon;
	Label *select_colors_label;
	Button *select_all_colors_button;
	Button *select_full_colors_button;
	Button *deselect_all_colors_button;
	Label *total_selected_colors_label;

	TextureRect *select_constants_icon;
	Label *select_constants_label;
	Button *select_all_constants_button;
	Button *select_full_constants_button;
	Button *deselect_all_constants_button;
	Label *total_selected_constants_label;

	TextureRect *select_fonts_icon;
	Label *select_fonts_label;
	Button *select_all_fonts_button;
	Button *select_full_fonts_button;
	Button *deselect_all_fonts_button;
	Label *total_selected_fonts_label;

	TextureRect *select_font_sizes_icon;
	Label *select_font_sizes_label;
	Button *select_all_font_sizes_button;
	Button *select_full_font_sizes_button;
	Button *deselect_all_font_sizes_button;
	Label *total_selected_font_sizes_label;

	TextureRect *select_icons_icon;
	Label *select_icons_label;
	Button *select_all_icons_button;
	Button *select_full_icons_button;
	Button *deselect_all_icons_button;
	Label *total_selected_icons_label;

	TextureRect *select_styleboxes_icon;
	Label *select_styleboxes_label;
	Button *select_all_styleboxes_button;
	Button *select_full_styleboxes_button;
	Button *deselect_all_styleboxes_button;
	Label *total_selected_styleboxes_label;

	HBoxContainer *select_icons_warning_hb;
	TextureRect *select_icons_warning_icon;
	Label *select_icons_warning;

	Button *import_collapse_types_button;
	Button *import_expand_types_button;
	Button *import_select_all_button;
	Button *import_select_full_button;
	Button *import_deselect_all_button;

	void _update_items_tree();
	void _toggle_type_items(bool p_collapse);
	void _filter_text_changed(const String &p_value);

	void _store_selected_item(TreeItem *p_tree_item);
	void _restore_selected_item(TreeItem *p_tree_item);
	void _update_total_selected(Theme::DataType p_data_type);

	void _tree_item_edited();
	void _select_all_subitems(TreeItem *p_root_item, bool p_select_with_data);
	void _deselect_all_subitems(TreeItem *p_root_item, bool p_deselect_completely);
	void _update_parent_items(TreeItem *p_root_item);

	void _select_all_items_pressed();
	void _select_full_items_pressed();
	void _deselect_all_items_pressed();

	void _select_all_data_type_pressed(int p_data_type);
	void _select_full_data_type_pressed(int p_data_type);
	void _deselect_all_data_type_pressed(int p_data_type);

	void _import_selected();

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void set_edited_theme(const Ref<Theme> &p_theme);
	void set_base_theme(const Ref<Theme> &p_theme);
	void reset_item_tree();

	bool has_selected_items() const;

	ThemeItemImportTree();
};

class ThemeItemEditorDialog : public AcceptDialog {
	GDCLASS(ThemeItemEditorDialog, AcceptDialog);

	Ref<Theme> edited_theme;

	TabContainer *tc;

	ItemList *edit_type_list;
	LineEdit *edit_add_type_value;
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

	ThemeItemImportTree *import_default_theme_items;
	ThemeItemImportTree *import_editor_theme_items;
	ThemeItemImportTree *import_other_theme_items;

	LineEdit *import_another_theme_value;
	Button *import_another_theme_button;
	EditorFileDialog *import_another_theme_dialog;

	ConfirmationDialog *confirm_closing_dialog;

	void ok_pressed() override;
	void _close_dialog();

	void _dialog_about_to_show();
	void _update_edit_types();
	void _edited_type_selected(int p_item_idx);

	void _update_edit_item_tree(String p_item_type);
	void _item_tree_button_pressed(Object *p_item, int p_column, int p_id);

	void _add_theme_type();
	void _add_theme_item(Theme::DataType p_data_type, String p_item_name, String p_item_type);
	void _remove_data_type_items(Theme::DataType p_data_type, String p_item_type);
	void _remove_class_items();
	void _remove_custom_items();
	void _remove_all_items();

	void _open_add_theme_item_dialog(int p_data_type);
	void _open_rename_theme_item_dialog(Theme::DataType p_data_type, String p_item_name);
	void _confirm_edit_theme_item();
	void _edit_theme_item_gui_input(const Ref<InputEvent> &p_event);

	void _open_select_another_theme();
	void _select_another_theme_cbk(const String &p_path);

protected:
	void _notification(int p_what);

public:
	void set_edited_theme(const Ref<Theme> &p_theme);

	ThemeItemEditorDialog();
};

class ThemeEditor : public VBoxContainer {
	GDCLASS(ThemeEditor, VBoxContainer);

	Ref<Theme> theme;

	double time_left = 0;

	Button *theme_edit_button;
	ThemeItemEditorDialog *theme_edit_dialog;

	Panel *main_panel;
	MarginContainer *main_container;
	Tree *test_tree;

	void _theme_edit_button_cbk();
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
