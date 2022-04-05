/*************************************************************************/
/*  theme_editor_plugin.h                                                */
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

#ifndef THEME_EDITOR_PLUGIN_H
#define THEME_EDITOR_PLUGIN_H

#include "scene/gui/dialogs.h"
#include "scene/gui/margin_container.h"
#include "scene/gui/option_button.h"
#include "scene/gui/scroll_container.h"
#include "scene/gui/tabs.h"
#include "scene/gui/texture_rect.h"
#include "scene/resources/theme.h"
#include "theme_editor_preview.h"

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

	enum TypesTreeAction {
		TYPES_TREE_REMOVE_ITEM,
	};

	Tree *edit_type_list;
	LineEdit *edit_add_type_value;
	String edited_item_type;

	Button *edit_items_add_color;
	Button *edit_items_add_constant;
	Button *edit_items_add_font;
	Button *edit_items_add_icon;
	Button *edit_items_add_stylebox;
	Button *edit_items_remove_class;
	Button *edit_items_remove_custom;
	Button *edit_items_remove_all;
	Tree *edit_items_tree;
	Label *edit_items_message;

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

	void ok_pressed();
	void _close_dialog();

	void _dialog_about_to_show();
	void _update_edit_types();
	void _edited_type_selected();
	void _edited_type_button_pressed(Object *p_item, int p_column, int p_id);

	void _update_edit_item_tree(String p_item_type);
	void _item_tree_button_pressed(Object *p_item, int p_column, int p_id);

	void _add_theme_type(const String &p_new_text);
	void _add_theme_item(Theme::DataType p_data_type, String p_item_name, String p_item_type);
	void _remove_theme_type(const String &p_theme_type);
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
	static void _bind_methods();

public:
	void set_edited_theme(const Ref<Theme> &p_theme);

	ThemeItemEditorDialog();
};

class ThemeTypeDialog : public ConfirmationDialog {
	GDCLASS(ThemeTypeDialog, ConfirmationDialog);

	Ref<Theme> edited_theme;
	bool include_own_types = false;

	String pre_submitted_value;

	LineEdit *add_type_filter;
	ItemList *add_type_options;
	ConfirmationDialog *add_type_confirmation;

	void _dialog_about_to_show();
	void ok_pressed();

	void _update_add_type_options(const String &p_filter = "");

	void _add_type_filter_cbk(const String &p_value);
	void _add_type_options_cbk(int p_index);
	void _add_type_dialog_entered(const String &p_value);
	void _add_type_dialog_activated(int p_index);

	void _add_type_selected(const String &p_type_name);
	void _add_type_confirmed();

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void set_edited_theme(const Ref<Theme> &p_theme);
	void set_include_own_types(bool p_enable);

	ThemeTypeDialog();
};

class ThemeTypeEditor : public MarginContainer {
	GDCLASS(ThemeTypeEditor, MarginContainer);

	Ref<Theme> edited_theme;
	String edited_type;
	bool updating = false;

	struct LeadingStylebox {
		bool pinned = false;
		StringName item_name;
		Ref<StyleBox> stylebox;
		Ref<StyleBox> ref_stylebox;
	};

	LeadingStylebox leading_stylebox;

	OptionButton *theme_type_list;
	Button *add_type_button;

	CheckButton *show_default_items_button;

	TabContainer *data_type_tabs;
	VBoxContainer *color_items_list;
	VBoxContainer *constant_items_list;
	VBoxContainer *font_items_list;
	VBoxContainer *icon_items_list;
	VBoxContainer *stylebox_items_list;

	LineEdit *type_variation_edit;
	Button *type_variation_button;
	Label *type_variation_locked;

	enum TypeDialogMode {
		ADD_THEME_TYPE,
		ADD_VARIATION_BASE,
	};

	TypeDialogMode add_type_mode = ADD_THEME_TYPE;
	ThemeTypeDialog *add_type_dialog;

	Vector<Control *> focusables;
	Timer *update_debounce_timer;

	VBoxContainer *_create_item_list(Theme::DataType p_data_type);
	void _update_type_list();
	void _update_type_list_debounced();
	OrderedHashMap<StringName, bool> _get_type_items(String p_type_name, void (Theme::*get_list_func)(StringName, List<StringName> *) const, bool include_default);
	HBoxContainer *_create_property_control(Theme::DataType p_data_type, String p_item_name, bool p_editable);
	void _add_focusable(Control *p_control);
	void _update_type_items();

	void _list_type_selected(int p_index);
	void _add_type_button_cbk();
	void _add_default_type_items();

	void _item_add_cbk(int p_data_type, Control *p_control);
	void _item_add_lineedit_cbk(String p_value, int p_data_type, Control *p_control);
	void _item_override_cbk(int p_data_type, String p_item_name);
	void _item_remove_cbk(int p_data_type, String p_item_name);
	void _item_rename_cbk(int p_data_type, String p_item_name, Control *p_control);
	void _item_rename_confirmed(int p_data_type, String p_item_name, Control *p_control);
	void _item_rename_entered(String p_value, int p_data_type, String p_item_name, Control *p_control);
	void _item_rename_canceled(int p_data_type, String p_item_name, Control *p_control);

	void _color_item_changed(Color p_value, String p_item_name);
	void _constant_item_changed(float p_value, String p_item_name);
	void _edit_resource_item(RES p_resource, bool p_edit);
	void _font_item_changed(Ref<Font> p_value, String p_item_name);
	void _icon_item_changed(Ref<Texture> p_value, String p_item_name);
	void _stylebox_item_changed(Ref<StyleBox> p_value, String p_item_name);
	void _pin_leading_stylebox(Control *p_editor, String p_item_name);
	void _unpin_leading_stylebox();
	void _update_stylebox_from_leading();

	void _type_variation_changed(const String p_value);
	void _add_type_variation_cbk();

	void _add_type_dialog_selected(const String p_type_name);

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void set_edited_theme(const Ref<Theme> &p_theme);
	void select_type(String p_type_name);

	ThemeTypeEditor();
};

class ThemeEditor : public VBoxContainer {
	GDCLASS(ThemeEditor, VBoxContainer);

	Ref<Theme> theme;

	Tabs *preview_tabs;
	PanelContainer *preview_tabs_content;
	Button *add_preview_button;
	EditorFileDialog *preview_scene_dialog;

	ThemeTypeEditor *theme_type_editor;

	Label *theme_name;
	ThemeItemEditorDialog *theme_edit_dialog;

	void _theme_save_button_cbk(bool p_save_as);
	void _theme_edit_button_cbk();

	void _add_preview_button_cbk();
	void _preview_scene_dialog_cbk(const String &p_path);
	void _add_preview_tab(ThemeEditorPreview *p_preview_tab, const String &p_preview_name, const Ref<Texture> &p_icon);
	void _change_preview_tab(int p_tab);
	void _remove_preview_tab(int p_tab);
	void _remove_preview_tab_invalid(Node *p_tab_control);
	void _update_preview_tab(Node *p_tab_control);
	void _preview_control_picked(String p_class_name);

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void edit(const Ref<Theme> &p_theme);
	Ref<Theme> get_edited_theme();

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
