/*************************************************************************/
/*  theme_editor_plugin.cpp                                              */
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

#include "theme_editor_plugin.h"

#include "core/os/file_access.h"
#include "core/os/keyboard.h"
#include "core/version.h"
#include "editor/editor_scale.h"
#include "scene/gui/progress_bar.h"

void ThemeItemEditorDialog::_dialog_about_to_show() {
	ERR_FAIL_COND(edited_theme.is_null());

	_update_edit_types();
}

void ThemeItemEditorDialog::_update_edit_types() {
	Ref<Theme> base_theme = Theme::get_default();

	List<StringName> theme_types;
	edited_theme->get_type_list(&theme_types);
	theme_types.sort_custom<StringName::AlphCompare>();

	bool item_reselected = false;
	edit_type_list->clear();
	int e_idx = 0;
	for (List<StringName>::Element *E = theme_types.front(); E; E = E->next()) {
		Ref<Texture2D> item_icon;
		if (E->get() == "") {
			item_icon = get_theme_icon("NodeDisabled", "EditorIcons");
		} else {
			item_icon = EditorNode::get_singleton()->get_class_icon(E->get(), "NodeDisabled");
		}
		edit_type_list->add_item(E->get(), item_icon);

		if (E->get() == edited_item_type) {
			edit_type_list->select(e_idx);
			item_reselected = true;
		}
		e_idx++;
	}
	if (!item_reselected) {
		edited_item_type = "";

		if (edit_type_list->get_item_count() > 0) {
			edit_type_list->select(0);
		}
	}

	List<StringName> default_types;
	base_theme->get_type_list(&default_types);
	default_types.sort_custom<StringName::AlphCompare>();

	edit_add_class_options->clear();
	for (List<StringName>::Element *E = default_types.front(); E; E = E->next()) {
		edit_add_class_options->add_item(E->get());
	}

	String selected_type = "";
	Vector<int> selected_ids = edit_type_list->get_selected_items();
	if (selected_ids.size() > 0) {
		selected_type = edit_type_list->get_item_text(selected_ids[0]);

		edit_items_add_color->set_disabled(false);
		edit_items_add_constant->set_disabled(false);
		edit_items_add_font->set_disabled(false);
		edit_items_add_font_size->set_disabled(false);
		edit_items_add_icon->set_disabled(false);
		edit_items_add_stylebox->set_disabled(false);

		edit_items_remove_class->set_disabled(false);
		edit_items_remove_custom->set_disabled(false);
		edit_items_remove_all->set_disabled(false);
	} else {
		edit_items_add_color->set_disabled(true);
		edit_items_add_constant->set_disabled(true);
		edit_items_add_font->set_disabled(true);
		edit_items_add_font_size->set_disabled(true);
		edit_items_add_icon->set_disabled(true);
		edit_items_add_stylebox->set_disabled(true);

		edit_items_remove_class->set_disabled(true);
		edit_items_remove_custom->set_disabled(true);
		edit_items_remove_all->set_disabled(true);
	}
	_update_edit_item_tree(selected_type);
}

void ThemeItemEditorDialog::_edited_type_selected(int p_item_idx) {
	String selected_type = edit_type_list->get_item_text(p_item_idx);
	_update_edit_item_tree(selected_type);
}

void ThemeItemEditorDialog::_update_edit_item_tree(String p_item_type) {
	edited_item_type = p_item_type;

	edit_items_tree->clear();
	TreeItem *root = edit_items_tree->create_item();

	List<StringName> names;

	{
		names.clear();
		edited_theme->get_color_list(p_item_type, &names);

		if (names.size() > 0) {
			TreeItem *color_root = edit_items_tree->create_item(root);
			color_root->set_metadata(0, Theme::DATA_TYPE_COLOR);
			color_root->set_icon(0, get_theme_icon("Color", "EditorIcons"));
			color_root->set_text(0, TTR("Colors"));
			color_root->add_button(0, get_theme_icon("Clear", "EditorIcons"), ITEMS_TREE_REMOVE_DATA_TYPE, false, TTR("Remove All Color Items"));

			names.sort_custom<StringName::AlphCompare>();
			for (List<StringName>::Element *E = names.front(); E; E = E->next()) {
				TreeItem *item = edit_items_tree->create_item(color_root);
				item->set_text(0, E->get());
				item->add_button(0, get_theme_icon("Edit", "EditorIcons"), ITEMS_TREE_RENAME_ITEM, false, TTR("Rename Item"));
				item->add_button(0, get_theme_icon("Remove", "EditorIcons"), ITEMS_TREE_REMOVE_ITEM, false, TTR("Remove Item"));
			}
		}
	}

	{
		names.clear();
		edited_theme->get_constant_list(p_item_type, &names);

		if (names.size() > 0) {
			TreeItem *constant_root = edit_items_tree->create_item(root);
			constant_root->set_metadata(0, Theme::DATA_TYPE_CONSTANT);
			constant_root->set_icon(0, get_theme_icon("MemberConstant", "EditorIcons"));
			constant_root->set_text(0, TTR("Constants"));
			constant_root->add_button(0, get_theme_icon("Clear", "EditorIcons"), ITEMS_TREE_REMOVE_DATA_TYPE, false, TTR("Remove All Constant Items"));

			names.sort_custom<StringName::AlphCompare>();
			for (List<StringName>::Element *E = names.front(); E; E = E->next()) {
				TreeItem *item = edit_items_tree->create_item(constant_root);
				item->set_text(0, E->get());
				item->add_button(0, get_theme_icon("Edit", "EditorIcons"), ITEMS_TREE_RENAME_ITEM, false, TTR("Rename Item"));
				item->add_button(0, get_theme_icon("Remove", "EditorIcons"), ITEMS_TREE_REMOVE_ITEM, false, TTR("Remove Item"));
			}
		}
	}

	{
		names.clear();
		edited_theme->get_font_list(p_item_type, &names);

		if (names.size() > 0) {
			TreeItem *font_root = edit_items_tree->create_item(root);
			font_root->set_metadata(0, Theme::DATA_TYPE_FONT);
			font_root->set_icon(0, get_theme_icon("Font", "EditorIcons"));
			font_root->set_text(0, TTR("Fonts"));
			font_root->add_button(0, get_theme_icon("Clear", "EditorIcons"), ITEMS_TREE_REMOVE_DATA_TYPE, false, TTR("Remove All Font Items"));

			names.sort_custom<StringName::AlphCompare>();
			for (List<StringName>::Element *E = names.front(); E; E = E->next()) {
				TreeItem *item = edit_items_tree->create_item(font_root);
				item->set_text(0, E->get());
				item->add_button(0, get_theme_icon("Edit", "EditorIcons"), ITEMS_TREE_RENAME_ITEM, false, TTR("Rename Item"));
				item->add_button(0, get_theme_icon("Remove", "EditorIcons"), ITEMS_TREE_REMOVE_ITEM, false, TTR("Remove Item"));
			}
		}
	}

	{
		names.clear();
		edited_theme->get_font_size_list(p_item_type, &names);

		if (names.size() > 0) {
			TreeItem *font_size_root = edit_items_tree->create_item(root);
			font_size_root->set_metadata(0, Theme::DATA_TYPE_FONT_SIZE);
			font_size_root->set_icon(0, get_theme_icon("FontSize", "EditorIcons"));
			font_size_root->set_text(0, TTR("Font Sizes"));
			font_size_root->add_button(0, get_theme_icon("Clear", "EditorIcons"), ITEMS_TREE_REMOVE_DATA_TYPE, false, TTR("Remove All Font Size Items"));

			names.sort_custom<StringName::AlphCompare>();
			for (List<StringName>::Element *E = names.front(); E; E = E->next()) {
				TreeItem *item = edit_items_tree->create_item(font_size_root);
				item->set_text(0, E->get());
				item->add_button(0, get_theme_icon("Edit", "EditorIcons"), ITEMS_TREE_RENAME_ITEM, false, TTR("Rename Item"));
				item->add_button(0, get_theme_icon("Remove", "EditorIcons"), ITEMS_TREE_REMOVE_ITEM, false, TTR("Remove Item"));
			}
		}
	}

	{
		names.clear();
		edited_theme->get_icon_list(p_item_type, &names);

		if (names.size() > 0) {
			TreeItem *icon_root = edit_items_tree->create_item(root);
			icon_root->set_metadata(0, Theme::DATA_TYPE_ICON);
			icon_root->set_icon(0, get_theme_icon("ImageTexture", "EditorIcons"));
			icon_root->set_text(0, TTR("Icons"));
			icon_root->add_button(0, get_theme_icon("Clear", "EditorIcons"), ITEMS_TREE_REMOVE_DATA_TYPE, false, TTR("Remove All Icon Items"));

			names.sort_custom<StringName::AlphCompare>();
			for (List<StringName>::Element *E = names.front(); E; E = E->next()) {
				TreeItem *item = edit_items_tree->create_item(icon_root);
				item->set_text(0, E->get());
				item->add_button(0, get_theme_icon("Edit", "EditorIcons"), ITEMS_TREE_RENAME_ITEM, false, TTR("Rename Item"));
				item->add_button(0, get_theme_icon("Remove", "EditorIcons"), ITEMS_TREE_REMOVE_ITEM, false, TTR("Remove Item"));
			}
		}
	}

	{
		names.clear();
		edited_theme->get_stylebox_list(p_item_type, &names);

		if (names.size() > 0) {
			TreeItem *stylebox_root = edit_items_tree->create_item(root);
			stylebox_root->set_metadata(0, Theme::DATA_TYPE_STYLEBOX);
			stylebox_root->set_icon(0, get_theme_icon("StyleBoxFlat", "EditorIcons"));
			stylebox_root->set_text(0, TTR("Styleboxes"));
			stylebox_root->add_button(0, get_theme_icon("Clear", "EditorIcons"), ITEMS_TREE_REMOVE_DATA_TYPE, false, TTR("Remove All StyleBox Items"));

			names.sort_custom<StringName::AlphCompare>();
			for (List<StringName>::Element *E = names.front(); E; E = E->next()) {
				TreeItem *item = edit_items_tree->create_item(stylebox_root);
				item->set_text(0, E->get());
				item->add_button(0, get_theme_icon("Edit", "EditorIcons"), ITEMS_TREE_RENAME_ITEM, false, TTR("Rename Item"));
				item->add_button(0, get_theme_icon("Remove", "EditorIcons"), ITEMS_TREE_REMOVE_ITEM, false, TTR("Remove Item"));
			}
		}
	}
}

void ThemeItemEditorDialog::_item_tree_button_pressed(Object *p_item, int p_column, int p_id) {
	TreeItem *item = Object::cast_to<TreeItem>(p_item);
	if (!item) {
		return;
	}

	switch (p_id) {
		case ITEMS_TREE_RENAME_ITEM: {
			String item_name = item->get_text(0);
			int data_type = item->get_parent()->get_metadata(0);
			_open_rename_theme_item_dialog((Theme::DataType)data_type, item_name);
		} break;
		case ITEMS_TREE_REMOVE_ITEM: {
			String item_name = item->get_text(0);
			int data_type = item->get_parent()->get_metadata(0);
			edited_theme->clear_theme_item((Theme::DataType)data_type, item_name, edited_item_type);
		} break;
		case ITEMS_TREE_REMOVE_DATA_TYPE: {
			int data_type = item->get_metadata(0);
			_remove_data_type_items((Theme::DataType)data_type, edited_item_type);
		} break;
	}

	_update_edit_item_tree(edited_item_type);
}

void ThemeItemEditorDialog::_add_class_type_items() {
	int selected_idx = edit_add_class_options->get_selected();
	String type_name = edit_add_class_options->get_item_text(selected_idx);
	List<StringName> names;

	{
		names.clear();
		Theme::get_default()->get_icon_list(type_name, &names);
		for (List<StringName>::Element *E = names.front(); E; E = E->next()) {
			edited_theme->set_icon(E->get(), type_name, Ref<Texture2D>());
		}
	}
	{
		names.clear();
		Theme::get_default()->get_stylebox_list(type_name, &names);
		for (List<StringName>::Element *E = names.front(); E; E = E->next()) {
			edited_theme->set_stylebox(E->get(), type_name, Ref<StyleBox>());
		}
	}
	{
		names.clear();
		Theme::get_default()->get_font_list(type_name, &names);
		for (List<StringName>::Element *E = names.front(); E; E = E->next()) {
			edited_theme->set_font(E->get(), type_name, Ref<Font>());
		}
	}
	{
		names.clear();
		Theme::get_default()->get_font_size_list(type_name, &names);
		for (List<StringName>::Element *E = names.front(); E; E = E->next()) {
			edited_theme->set_font_size(E->get(), type_name, Theme::get_default()->get_font_size(E->get(), type_name));
		}
	}
	{
		names.clear();
		Theme::get_default()->get_color_list(type_name, &names);
		for (List<StringName>::Element *E = names.front(); E; E = E->next()) {
			edited_theme->set_color(E->get(), type_name, Theme::get_default()->get_color(E->get(), type_name));
		}
	}
	{
		names.clear();
		Theme::get_default()->get_constant_list(type_name, &names);
		for (List<StringName>::Element *E = names.front(); E; E = E->next()) {
			edited_theme->set_constant(E->get(), type_name, Theme::get_default()->get_constant(E->get(), type_name));
		}
	}

	_update_edit_types();
}

void ThemeItemEditorDialog::_add_custom_type() {
	edited_theme->add_icon_type(edit_add_custom_value->get_text());
	edited_theme->add_stylebox_type(edit_add_custom_value->get_text());
	edited_theme->add_font_type(edit_add_custom_value->get_text());
	edited_theme->add_font_size_type(edit_add_custom_value->get_text());
	edited_theme->add_color_type(edit_add_custom_value->get_text());
	edited_theme->add_constant_type(edit_add_custom_value->get_text());
	_update_edit_types();
}

void ThemeItemEditorDialog::_add_theme_item(Theme::DataType p_data_type, String p_item_name, String p_item_type) {
	switch (p_data_type) {
		case Theme::DATA_TYPE_ICON:
			edited_theme->set_icon(p_item_name, p_item_type, Ref<Texture2D>());
			break;
		case Theme::DATA_TYPE_STYLEBOX:
			edited_theme->set_stylebox(p_item_name, p_item_type, Ref<StyleBox>());
			break;
		case Theme::DATA_TYPE_FONT:
			edited_theme->set_font(p_item_name, p_item_type, Ref<Font>());
			break;
		case Theme::DATA_TYPE_FONT_SIZE:
			edited_theme->set_font_size(p_item_name, p_item_type, -1);
			break;
		case Theme::DATA_TYPE_COLOR:
			edited_theme->set_color(p_item_name, p_item_type, Color());
			break;
		case Theme::DATA_TYPE_CONSTANT:
			edited_theme->set_constant(p_item_name, p_item_type, 0);
			break;
		case Theme::DATA_TYPE_MAX:
			break; // Can't happen, but silences warning.
	}
}

void ThemeItemEditorDialog::_remove_data_type_items(Theme::DataType p_data_type, String p_item_type) {
	List<StringName> names;

	edited_theme->get_theme_item_list(p_data_type, p_item_type, &names);
	for (List<StringName>::Element *E = names.front(); E; E = E->next()) {
		edited_theme->clear_theme_item(p_data_type, E->get(), p_item_type);
	}
}

void ThemeItemEditorDialog::_remove_class_items() {
	List<StringName> names;

	for (int dt = 0; dt < Theme::DATA_TYPE_MAX; dt++) {
		Theme::DataType data_type = (Theme::DataType)dt;

		names.clear();
		Theme::get_default()->get_theme_item_list(data_type, edited_item_type, &names);
		for (List<StringName>::Element *E = names.front(); E; E = E->next()) {
			if (edited_theme->has_theme_item_nocheck(data_type, E->get(), edited_item_type)) {
				edited_theme->clear_theme_item(data_type, E->get(), edited_item_type);
			}
		}
	}

	_update_edit_item_tree(edited_item_type);
}

void ThemeItemEditorDialog::_remove_custom_items() {
	List<StringName> names;

	for (int dt = 0; dt < Theme::DATA_TYPE_MAX; dt++) {
		Theme::DataType data_type = (Theme::DataType)dt;

		names.clear();
		edited_theme->get_theme_item_list(data_type, edited_item_type, &names);
		for (List<StringName>::Element *E = names.front(); E; E = E->next()) {
			if (!Theme::get_default()->has_theme_item_nocheck(data_type, E->get(), edited_item_type)) {
				edited_theme->clear_theme_item(data_type, E->get(), edited_item_type);
			}
		}
	}

	_update_edit_item_tree(edited_item_type);
}

void ThemeItemEditorDialog::_remove_all_items() {
	List<StringName> names;

	for (int dt = 0; dt < Theme::DATA_TYPE_MAX; dt++) {
		Theme::DataType data_type = (Theme::DataType)dt;

		names.clear();
		edited_theme->get_theme_item_list(data_type, edited_item_type, &names);
		for (List<StringName>::Element *E = names.front(); E; E = E->next()) {
			edited_theme->clear_theme_item(data_type, E->get(), edited_item_type);
		}
	}

	_update_edit_item_tree(edited_item_type);
}

void ThemeItemEditorDialog::_open_add_theme_item_dialog(int p_data_type) {
	ERR_FAIL_INDEX_MSG(p_data_type, Theme::DATA_TYPE_MAX, "Theme item data type is out of bounds.");

	item_popup_mode = CREATE_THEME_ITEM;
	edit_item_data_type = (Theme::DataType)p_data_type;

	switch (edit_item_data_type) {
		case Theme::DATA_TYPE_COLOR:
			edit_theme_item_dialog->set_title(TTR("Add Color Item"));
			break;
		case Theme::DATA_TYPE_CONSTANT:
			edit_theme_item_dialog->set_title(TTR("Add Constant Item"));
			break;
		case Theme::DATA_TYPE_FONT:
			edit_theme_item_dialog->set_title(TTR("Add Font Item"));
			break;
		case Theme::DATA_TYPE_FONT_SIZE:
			edit_theme_item_dialog->set_title(TTR("Add Font Size Item"));
			break;
		case Theme::DATA_TYPE_ICON:
			edit_theme_item_dialog->set_title(TTR("Add Icon Item"));
			break;
		case Theme::DATA_TYPE_STYLEBOX:
			edit_theme_item_dialog->set_title(TTR("Add Stylebox Item"));
			break;
		case Theme::DATA_TYPE_MAX:
			break; // Can't happen, but silences warning.
	}

	edit_theme_item_old_vb->hide();
	theme_item_name->clear();
	edit_theme_item_dialog->popup_centered(Size2(380, 110) * EDSCALE);
	theme_item_name->grab_focus();
}

void ThemeItemEditorDialog::_open_rename_theme_item_dialog(Theme::DataType p_data_type, String p_item_name) {
	ERR_FAIL_INDEX_MSG(p_data_type, Theme::DATA_TYPE_MAX, "Theme item data type is out of bounds.");

	item_popup_mode = RENAME_THEME_ITEM;
	edit_item_data_type = p_data_type;
	edit_item_old_name = p_item_name;

	switch (edit_item_data_type) {
		case Theme::DATA_TYPE_COLOR:
			edit_theme_item_dialog->set_title(TTR("Rename Color Item"));
			break;
		case Theme::DATA_TYPE_CONSTANT:
			edit_theme_item_dialog->set_title(TTR("Rename Constant Item"));
			break;
		case Theme::DATA_TYPE_FONT:
			edit_theme_item_dialog->set_title(TTR("Rename Font Item"));
			break;
		case Theme::DATA_TYPE_FONT_SIZE:
			edit_theme_item_dialog->set_title(TTR("Rename Font Size Item"));
			break;
		case Theme::DATA_TYPE_ICON:
			edit_theme_item_dialog->set_title(TTR("Rename Icon Item"));
			break;
		case Theme::DATA_TYPE_STYLEBOX:
			edit_theme_item_dialog->set_title(TTR("Rename Stylebox Item"));
			break;
		case Theme::DATA_TYPE_MAX:
			break; // Can't happen, but silences warning.
	}

	edit_theme_item_old_vb->show();
	theme_item_old_name->set_text(p_item_name);
	theme_item_name->set_text(p_item_name);
	edit_theme_item_dialog->popup_centered(Size2(380, 140) * EDSCALE);
	theme_item_name->grab_focus();
}

void ThemeItemEditorDialog::_confirm_edit_theme_item() {
	if (item_popup_mode == CREATE_THEME_ITEM) {
		_add_theme_item(edit_item_data_type, theme_item_name->get_text(), edited_item_type);
	} else if (item_popup_mode == RENAME_THEME_ITEM) {
		edited_theme->rename_theme_item(edit_item_data_type, edit_item_old_name, theme_item_name->get_text(), edited_item_type);
	}

	item_popup_mode = ITEM_POPUP_MODE_MAX;
	edit_item_data_type = Theme::DATA_TYPE_MAX;
	edit_item_old_name = "";

	_update_edit_item_tree(edited_item_type);
}

void ThemeItemEditorDialog::_edit_theme_item_gui_input(const Ref<InputEvent> &p_event) {
	Ref<InputEventKey> k = p_event;

	if (k.is_valid()) {
		if (!k->is_pressed()) {
			return;
		}

		switch (k->get_keycode()) {
			case KEY_KP_ENTER:
			case KEY_ENTER: {
				_confirm_edit_theme_item();
				edit_theme_item_dialog->hide();
				edit_theme_item_dialog->set_input_as_handled();
			} break;
			case KEY_ESCAPE: {
				edit_theme_item_dialog->hide();
				edit_theme_item_dialog->set_input_as_handled();
			} break;
		}
	}
}

void ThemeItemEditorDialog::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			connect("about_to_popup", callable_mp(this, &ThemeItemEditorDialog::_dialog_about_to_show));
			[[fallthrough]];
		}
		case NOTIFICATION_THEME_CHANGED: {
			edit_items_add_color->set_icon(get_theme_icon("Color", "EditorIcons"));
			edit_items_add_constant->set_icon(get_theme_icon("MemberConstant", "EditorIcons"));
			edit_items_add_font->set_icon(get_theme_icon("Font", "EditorIcons"));
			edit_items_add_font_size->set_icon(get_theme_icon("FontSize", "EditorIcons"));
			edit_items_add_icon->set_icon(get_theme_icon("ImageTexture", "EditorIcons"));
			edit_items_add_stylebox->set_icon(get_theme_icon("StyleBoxFlat", "EditorIcons"));

			edit_items_remove_class->set_icon(get_theme_icon("Control", "EditorIcons"));
			edit_items_remove_custom->set_icon(get_theme_icon("ThemeRemoveCustomItems", "EditorIcons"));
			edit_items_remove_all->set_icon(get_theme_icon("ThemeRemoveAllItems", "EditorIcons"));
		} break;
	}
}

void ThemeItemEditorDialog::set_edited_theme(const Ref<Theme> &p_theme) {
	edited_theme = p_theme;
}

ThemeItemEditorDialog::ThemeItemEditorDialog() {
	set_title(TTR("Edit Theme Items"));

	HSplitContainer *edit_dialog_hs = memnew(HSplitContainer);
	add_child(edit_dialog_hs);

	VBoxContainer *edit_dialog_side_vb = memnew(VBoxContainer);
	edit_dialog_side_vb->set_custom_minimum_size(Size2(200.0, 0.0) * EDSCALE);
	edit_dialog_hs->add_child(edit_dialog_side_vb);

	Label *edit_type_label = memnew(Label);
	edit_type_label->set_text(TTR("Types:"));
	edit_dialog_side_vb->add_child(edit_type_label);

	edit_type_list = memnew(ItemList);
	edit_type_list->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	edit_dialog_side_vb->add_child(edit_type_list);
	edit_type_list->connect("item_selected", callable_mp(this, &ThemeItemEditorDialog::_edited_type_selected));

	Label *edit_add_class_label = memnew(Label);
	edit_add_class_label->set_text(TTR("Add Type from Class:"));
	edit_dialog_side_vb->add_child(edit_add_class_label);

	HBoxContainer *edit_add_class = memnew(HBoxContainer);
	edit_dialog_side_vb->add_child(edit_add_class);
	edit_add_class_options = memnew(OptionButton);
	edit_add_class_options->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	edit_add_class->add_child(edit_add_class_options);
	Button *edit_add_class_button = memnew(Button);
	edit_add_class_button->set_text(TTR("Add"));
	edit_add_class->add_child(edit_add_class_button);
	edit_add_class_button->connect("pressed", callable_mp(this, &ThemeItemEditorDialog::_add_class_type_items));

	Label *edit_add_custom_label = memnew(Label);
	edit_add_custom_label->set_text(TTR("Add Custom Type:"));
	edit_dialog_side_vb->add_child(edit_add_custom_label);

	HBoxContainer *edit_add_custom = memnew(HBoxContainer);
	edit_dialog_side_vb->add_child(edit_add_custom);
	edit_add_custom_value = memnew(LineEdit);
	edit_add_custom_value->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	edit_add_custom->add_child(edit_add_custom_value);
	Button *edit_add_custom_button = memnew(Button);
	edit_add_custom_button->set_text(TTR("Add"));
	edit_add_custom->add_child(edit_add_custom_button);
	edit_add_custom_button->connect("pressed", callable_mp(this, &ThemeItemEditorDialog::_add_custom_type));

	VBoxContainer *edit_items_vb = memnew(VBoxContainer);
	edit_items_vb->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	edit_dialog_hs->add_child(edit_items_vb);

	HBoxContainer *edit_items_toolbar = memnew(HBoxContainer);
	edit_items_vb->add_child(edit_items_toolbar);

	Label *edit_items_toolbar_add_label = memnew(Label);
	edit_items_toolbar_add_label->set_text(TTR("Add:"));
	edit_items_toolbar->add_child(edit_items_toolbar_add_label);

	edit_items_add_color = memnew(Button);
	edit_items_add_color->set_tooltip(TTR("Add Color Item"));
	edit_items_add_color->set_flat(true);
	edit_items_add_color->set_disabled(true);
	edit_items_toolbar->add_child(edit_items_add_color);
	edit_items_add_color->connect("pressed", callable_mp(this, &ThemeItemEditorDialog::_open_add_theme_item_dialog), varray(Theme::DATA_TYPE_COLOR));

	edit_items_add_constant = memnew(Button);
	edit_items_add_constant->set_tooltip(TTR("Add Constant Item"));
	edit_items_add_constant->set_flat(true);
	edit_items_add_constant->set_disabled(true);
	edit_items_toolbar->add_child(edit_items_add_constant);
	edit_items_add_constant->connect("pressed", callable_mp(this, &ThemeItemEditorDialog::_open_add_theme_item_dialog), varray(Theme::DATA_TYPE_CONSTANT));

	edit_items_add_font = memnew(Button);
	edit_items_add_font->set_tooltip(TTR("Add Font Item"));
	edit_items_add_font->set_flat(true);
	edit_items_add_font->set_disabled(true);
	edit_items_toolbar->add_child(edit_items_add_font);
	edit_items_add_font->connect("pressed", callable_mp(this, &ThemeItemEditorDialog::_open_add_theme_item_dialog), varray(Theme::DATA_TYPE_FONT));

	edit_items_add_font_size = memnew(Button);
	edit_items_add_font_size->set_tooltip(TTR("Add Font Size Item"));
	edit_items_add_font_size->set_flat(true);
	edit_items_add_font_size->set_disabled(true);
	edit_items_toolbar->add_child(edit_items_add_font_size);
	edit_items_add_font_size->connect("pressed", callable_mp(this, &ThemeItemEditorDialog::_open_add_theme_item_dialog), varray(Theme::DATA_TYPE_FONT_SIZE));

	edit_items_add_icon = memnew(Button);
	edit_items_add_icon->set_tooltip(TTR("Add Icon Item"));
	edit_items_add_icon->set_flat(true);
	edit_items_add_icon->set_disabled(true);
	edit_items_toolbar->add_child(edit_items_add_icon);
	edit_items_add_icon->connect("pressed", callable_mp(this, &ThemeItemEditorDialog::_open_add_theme_item_dialog), varray(Theme::DATA_TYPE_ICON));

	edit_items_add_stylebox = memnew(Button);
	edit_items_add_stylebox->set_tooltip(TTR("Add StyleBox Item"));
	edit_items_add_stylebox->set_flat(true);
	edit_items_add_stylebox->set_disabled(true);
	edit_items_toolbar->add_child(edit_items_add_stylebox);
	edit_items_add_stylebox->connect("pressed", callable_mp(this, &ThemeItemEditorDialog::_open_add_theme_item_dialog), varray(Theme::DATA_TYPE_STYLEBOX));

	edit_items_toolbar->add_child(memnew(VSeparator));

	Label *edit_items_toolbar_remove_label = memnew(Label);
	edit_items_toolbar_remove_label->set_text(TTR("Remove:"));
	edit_items_toolbar->add_child(edit_items_toolbar_remove_label);

	edit_items_remove_class = memnew(Button);
	edit_items_remove_class->set_tooltip(TTR("Remove Class Items"));
	edit_items_remove_class->set_flat(true);
	edit_items_remove_class->set_disabled(true);
	edit_items_toolbar->add_child(edit_items_remove_class);
	edit_items_remove_class->connect("pressed", callable_mp(this, &ThemeItemEditorDialog::_remove_class_items));

	edit_items_remove_custom = memnew(Button);
	edit_items_remove_custom->set_tooltip(TTR("Remove Custom Items"));
	edit_items_remove_custom->set_flat(true);
	edit_items_remove_custom->set_disabled(true);
	edit_items_toolbar->add_child(edit_items_remove_custom);
	edit_items_remove_custom->connect("pressed", callable_mp(this, &ThemeItemEditorDialog::_remove_custom_items));

	edit_items_remove_all = memnew(Button);
	edit_items_remove_all->set_tooltip(TTR("Remove All Items"));
	edit_items_remove_all->set_flat(true);
	edit_items_remove_all->set_disabled(true);
	edit_items_toolbar->add_child(edit_items_remove_all);
	edit_items_remove_all->connect("pressed", callable_mp(this, &ThemeItemEditorDialog::_remove_all_items));

	edit_items_tree = memnew(Tree);
	edit_items_tree->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	edit_items_tree->set_hide_root(true);
	edit_items_tree->set_columns(1);
	edit_items_vb->add_child(edit_items_tree);
	edit_items_tree->connect("button_pressed", callable_mp(this, &ThemeItemEditorDialog::_item_tree_button_pressed));

	edit_theme_item_dialog = memnew(ConfirmationDialog);
	edit_theme_item_dialog->set_title(TTR("Add Theme Item"));
	add_child(edit_theme_item_dialog);
	VBoxContainer *edit_theme_item_vb = memnew(VBoxContainer);
	edit_theme_item_dialog->add_child(edit_theme_item_vb);

	edit_theme_item_old_vb = memnew(VBoxContainer);
	edit_theme_item_vb->add_child(edit_theme_item_old_vb);
	Label *edit_theme_item_old = memnew(Label);
	edit_theme_item_old->set_text(TTR("Old Name:"));
	edit_theme_item_old_vb->add_child(edit_theme_item_old);
	theme_item_old_name = memnew(Label);
	edit_theme_item_old_vb->add_child(theme_item_old_name);

	Label *edit_theme_item_label = memnew(Label);
	edit_theme_item_label->set_text(TTR("Name:"));
	edit_theme_item_vb->add_child(edit_theme_item_label);
	theme_item_name = memnew(LineEdit);
	edit_theme_item_vb->add_child(theme_item_name);
	theme_item_name->connect("gui_input", callable_mp(this, &ThemeItemEditorDialog::_edit_theme_item_gui_input));
	edit_theme_item_dialog->connect("confirmed", callable_mp(this, &ThemeItemEditorDialog::_confirm_edit_theme_item));
}

void ThemeEditor::edit(const Ref<Theme> &p_theme) {
	theme = p_theme;
	theme_edit_dialog->set_edited_theme(p_theme);
	main_panel->set_theme(p_theme);
	main_container->set_theme(p_theme);
}

void ThemeEditor::_propagate_redraw(Control *p_at) {
	p_at->notification(NOTIFICATION_THEME_CHANGED);
	p_at->minimum_size_changed();
	p_at->update();
	for (int i = 0; i < p_at->get_child_count(); i++) {
		Control *a = Object::cast_to<Control>(p_at->get_child(i));
		if (a) {
			_propagate_redraw(a);
		}
	}
}

void ThemeEditor::_refresh_interval() {
	_propagate_redraw(main_panel);
	_propagate_redraw(main_container);
}

struct _TECategory {
	template <class T>
	struct RefItem {
		Ref<T> item;
		StringName name;
		bool operator<(const RefItem<T> &p) const { return item->get_instance_id() < p.item->get_instance_id(); }
	};

	template <class T>
	struct Item {
		T item;
		String name;
		bool operator<(const Item<T> &p) const { return name < p.name; }
	};

	Set<RefItem<StyleBox>> stylebox_items;
	Set<RefItem<Font>> font_items;
	Set<Item<int>> font_size_items;
	Set<RefItem<Texture2D>> icon_items;

	Set<Item<Color>> color_items;
	Set<Item<int>> constant_items;
};

void ThemeEditor::_save_template_cbk(String fname) {
	String filename = file_dialog->get_current_path();

	Map<String, _TECategory> categories;

	// Fill types.
	List<StringName> type_list;
	Theme::get_default()->get_type_list(&type_list);
	for (List<StringName>::Element *E = type_list.front(); E; E = E->next()) {
		categories.insert(E->get(), _TECategory());
	}

	// Fill default theme.
	for (Map<String, _TECategory>::Element *E = categories.front(); E; E = E->next()) {
		_TECategory &tc = E->get();

		List<StringName> stylebox_list;
		Theme::get_default()->get_stylebox_list(E->key(), &stylebox_list);
		for (List<StringName>::Element *F = stylebox_list.front(); F; F = F->next()) {
			_TECategory::RefItem<StyleBox> it;
			it.name = F->get();
			it.item = Theme::get_default()->get_stylebox(F->get(), E->key());
			tc.stylebox_items.insert(it);
		}

		List<StringName> font_list;
		Theme::get_default()->get_font_list(E->key(), &font_list);
		for (List<StringName>::Element *F = font_list.front(); F; F = F->next()) {
			_TECategory::RefItem<Font> it;
			it.name = F->get();
			it.item = Theme::get_default()->get_font(F->get(), E->key());
			tc.font_items.insert(it);
		}

		List<StringName> font_size_list;
		Theme::get_default()->get_font_size_list(E->key(), &font_list);
		for (List<StringName>::Element *F = font_size_list.front(); F; F = F->next()) {
			_TECategory::Item<int> it;
			it.name = F->get();
			it.item = Theme::get_default()->get_font_size(F->get(), E->key());
			tc.font_size_items.insert(it);
		}

		List<StringName> icon_list;
		Theme::get_default()->get_icon_list(E->key(), &icon_list);
		for (List<StringName>::Element *F = icon_list.front(); F; F = F->next()) {
			_TECategory::RefItem<Texture2D> it;
			it.name = F->get();
			it.item = Theme::get_default()->get_icon(F->get(), E->key());
			tc.icon_items.insert(it);
		}

		List<StringName> color_list;
		Theme::get_default()->get_color_list(E->key(), &color_list);
		for (List<StringName>::Element *F = color_list.front(); F; F = F->next()) {
			_TECategory::Item<Color> it;
			it.name = F->get();
			it.item = Theme::get_default()->get_color(F->get(), E->key());
			tc.color_items.insert(it);
		}

		List<StringName> constant_list;
		Theme::get_default()->get_constant_list(E->key(), &constant_list);
		for (List<StringName>::Element *F = constant_list.front(); F; F = F->next()) {
			_TECategory::Item<int> it;
			it.name = F->get();
			it.item = Theme::get_default()->get_constant(F->get(), E->key());
			tc.constant_items.insert(it);
		}
	}

	FileAccess *file = FileAccess::open(filename, FileAccess::WRITE);

	ERR_FAIL_COND_MSG(!file, "Can't save theme to file '" + filename + "'.");

	file->store_line("; ******************* ");
	file->store_line("; Template Theme File ");
	file->store_line("; ******************* ");
	file->store_line("; ");
	file->store_line("; Theme Syntax: ");
	file->store_line("; ------------- ");
	file->store_line("; ");
	file->store_line("; Must be placed in section [theme]");
	file->store_line("; ");
	file->store_line("; Type.item = [value] ");
	file->store_line("; ");
	file->store_line("; [value] examples:");
	file->store_line("; ");
	file->store_line("; Type.item = 6 ; numeric constant. ");
	file->store_line("; Type.item = #FF00FF ; HTML color (magenta).");
	file->store_line("; Type.item = #FF00FF55 ; HTML color (magenta with alpha 0x55).");
	file->store_line("; Type.item = icon(image.png) ; icon in a png file (relative to theme file).");
	file->store_line("; Type.item = font(font.xres) ; font in a resource (relative to theme file).");
	file->store_line("; Type.item = sbox(stylebox.xres) ; stylebox in a resource (relative to theme file).");
	file->store_line("; Type.item = sboxf(2,#FF00FF) ; flat stylebox with margin 2.");
	file->store_line("; Type.item = sboxf(2,#FF00FF,#FFFFFF) ; flat stylebox with margin 2 and border.");
	file->store_line("; Type.item = sboxf(2,#FF00FF,#FFFFFF,#000000) ; flat stylebox with margin 2, light & dark borders.");
	file->store_line("; Type.item = sboxt(base.png,2,2,2,2) ; textured stylebox with 3x3 stretch and stretch margins.");
	file->store_line(";   -Additionally, 4 extra integers can be added to sboxf and sboxt to specify custom padding of contents:");
	file->store_line("; Type.item = sboxt(base.png,2,2,2,2,5,4,2,4) ;");
	file->store_line(";   -Order for all is always left, top, right, bottom.");
	file->store_line("; ");
	file->store_line("; Special values:");
	file->store_line("; Type.item = default ; use the value in the default theme (must exist there).");
	file->store_line("; Type.item = @somebutton_color ; reference to a library value previously defined.");
	file->store_line("; ");
	file->store_line("; Library Syntax: ");
	file->store_line("; --------------- ");
	file->store_line("; ");
	file->store_line("; Must be placed in section [library], but usage is optional.");
	file->store_line("; ");
	file->store_line("; item = [value] ; same as Theme, but assign to library.");
	file->store_line("; ");
	file->store_line("; examples:");
	file->store_line("; ");
	file->store_line("; [library]");
	file->store_line("; ");
	file->store_line("; default_button_color = #FF00FF");
	file->store_line("; ");
	file->store_line("; [theme]");
	file->store_line("; ");
	file->store_line("; Button.color = @default_button_color ; used reference.");
	file->store_line("; ");
	file->store_line("; ******************* ");
	file->store_line("; ");
	file->store_line("; Template Generated Using: " + String(VERSION_FULL_BUILD));
	file->store_line(";    ");
	file->store_line("; ");
	file->store_line("");
	file->store_line("[library]");
	file->store_line("");
	file->store_line("; place library stuff here");
	file->store_line("");
	file->store_line("[theme]");
	file->store_line("");
	file->store_line("");

	// Write default theme.
	for (Map<String, _TECategory>::Element *E = categories.front(); E; E = E->next()) {
		_TECategory &tc = E->get();

		String underline = "; ";
		for (int i = 0; i < E->key().length(); i++) {
			underline += "*";
		}

		file->store_line("");
		file->store_line(underline);
		file->store_line("; " + E->key());
		file->store_line(underline);

		if (tc.stylebox_items.size()) {
			file->store_line("\n; StyleBox Items:\n");
		}

		for (Set<_TECategory::RefItem<StyleBox>>::Element *F = tc.stylebox_items.front(); F; F = F->next()) {
			file->store_line(E->key() + "." + F->get().name + " = default");
		}

		if (tc.font_items.size()) {
			file->store_line("\n; Font Items:\n");
		}

		for (Set<_TECategory::RefItem<Font>>::Element *F = tc.font_items.front(); F; F = F->next()) {
			file->store_line(E->key() + "." + F->get().name + " = default");
		}

		if (tc.font_size_items.size()) {
			file->store_line("\n; Font Size Items:\n");
		}

		for (Set<_TECategory::Item<int>>::Element *F = tc.font_size_items.front(); F; F = F->next()) {
			file->store_line(E->key() + "." + F->get().name + " = default");
		}

		if (tc.icon_items.size()) {
			file->store_line("\n; Icon Items:\n");
		}

		for (Set<_TECategory::RefItem<Texture2D>>::Element *F = tc.icon_items.front(); F; F = F->next()) {
			file->store_line(E->key() + "." + F->get().name + " = default");
		}

		if (tc.color_items.size()) {
			file->store_line("\n; Color Items:\n");
		}

		for (Set<_TECategory::Item<Color>>::Element *F = tc.color_items.front(); F; F = F->next()) {
			file->store_line(E->key() + "." + F->get().name + " = default");
		}

		if (tc.constant_items.size()) {
			file->store_line("\n; Constant Items:\n");
		}

		for (Set<_TECategory::Item<int>>::Element *F = tc.constant_items.front(); F; F = F->next()) {
			file->store_line(E->key() + "." + F->get().name + " = default");
		}
	}

	file->close();
	memdelete(file);
}

void ThemeEditor::_theme_create_menu_cbk(int p_option) {
	bool import = (p_option == POPUP_IMPORT_EDITOR_THEME);

	Ref<Theme> base_theme;

	if (p_option == POPUP_CREATE_EMPTY) {
		base_theme = Theme::get_default();
	} else {
		base_theme = EditorNode::get_singleton()->get_theme_base()->get_theme();
	}

	{
		List<StringName> types;
		base_theme->get_type_list(&types);

		for (List<StringName>::Element *T = types.front(); T; T = T->next()) {
			StringName type = T->get();

			List<StringName> icons;
			base_theme->get_icon_list(type, &icons);

			for (List<StringName>::Element *E = icons.front(); E; E = E->next()) {
				theme->set_icon(E->get(), type, import ? base_theme->get_icon(E->get(), type) : Ref<Texture2D>());
			}

			List<StringName> styleboxs;
			base_theme->get_stylebox_list(type, &styleboxs);

			for (List<StringName>::Element *E = styleboxs.front(); E; E = E->next()) {
				theme->set_stylebox(E->get(), type, import ? base_theme->get_stylebox(E->get(), type) : Ref<StyleBox>());
			}

			List<StringName> fonts;
			base_theme->get_font_list(type, &fonts);

			for (List<StringName>::Element *E = fonts.front(); E; E = E->next()) {
				theme->set_font(E->get(), type, Ref<Font>());
			}

			List<StringName> font_sizes;
			base_theme->get_font_size_list(type, &font_sizes);

			for (List<StringName>::Element *E = font_sizes.front(); E; E = E->next()) {
				theme->set_font_size(E->get(), type, base_theme->get_font_size(E->get(), type));
			}

			List<StringName> colors;
			base_theme->get_color_list(type, &colors);

			for (List<StringName>::Element *E = colors.front(); E; E = E->next()) {
				theme->set_color(E->get(), type, import ? base_theme->get_color(E->get(), type) : Color());
			}

			List<StringName> constants;
			base_theme->get_constant_list(type, &constants);

			for (List<StringName>::Element *E = constants.front(); E; E = E->next()) {
				theme->set_constant(E->get(), type, base_theme->get_constant(E->get(), type));
			}
		}
	}
}

void ThemeEditor::_theme_edit_button_cbk() {
	theme_edit_dialog->popup_centered(Size2(800, 640) * EDSCALE);
}

void ThemeEditor::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_PROCESS: {
			time_left -= get_process_delta_time();
			if (time_left < 0) {
				time_left = 1.5;
				_refresh_interval();
			}
		} break;
	}
}

void ThemeEditor::_bind_methods() {
}

ThemeEditor::ThemeEditor() {
	HBoxContainer *top_menu = memnew(HBoxContainer);
	add_child(top_menu);

	top_menu->add_child(memnew(Label(TTR("Preview:"))));
	top_menu->add_spacer(false);

	theme_create_menu = memnew(MenuButton);
	theme_create_menu->set_text(TTR("Create Theme..."));
	theme_create_menu->set_tooltip(TTR("Create a new Theme."));
	theme_create_menu->get_popup()->add_item(TTR("Empty Template"), POPUP_CREATE_EMPTY);
	theme_create_menu->get_popup()->add_separator();
	theme_create_menu->get_popup()->add_item(TTR("Empty Editor Template"), POPUP_CREATE_EDITOR_EMPTY);
	theme_create_menu->get_popup()->add_item(TTR("From Current Editor Theme"), POPUP_IMPORT_EDITOR_THEME);
	top_menu->add_child(theme_create_menu);
	theme_create_menu->get_popup()->connect("id_pressed", callable_mp(this, &ThemeEditor::_theme_create_menu_cbk));

	theme_edit_button = memnew(Button);
	theme_edit_button->set_text(TTR("Edit Theme Items"));
	theme_edit_button->set_tooltip(TTR("Customize Theme items."));
	theme_edit_button->set_flat(true);
	theme_edit_button->connect("pressed", callable_mp(this, &ThemeEditor::_theme_edit_button_cbk));
	top_menu->add_child(theme_edit_button);

	ScrollContainer *scroll = memnew(ScrollContainer);
	add_child(scroll);
	scroll->set_enable_v_scroll(true);
	scroll->set_enable_h_scroll(true);
	scroll->set_v_size_flags(SIZE_EXPAND_FILL);

	MarginContainer *root_container = memnew(MarginContainer);
	scroll->add_child(root_container);
	root_container->set_theme(Theme::get_default());
	root_container->set_clip_contents(true);
	root_container->set_custom_minimum_size(Size2(700, 0) * EDSCALE);
	root_container->set_v_size_flags(SIZE_EXPAND_FILL);
	root_container->set_h_size_flags(SIZE_EXPAND_FILL);

	//// Preview Controls ////

	main_panel = memnew(Panel);
	root_container->add_child(main_panel);

	main_container = memnew(MarginContainer);
	root_container->add_child(main_container);
	main_container->add_theme_constant_override("margin_right", 4 * EDSCALE);
	main_container->add_theme_constant_override("margin_top", 4 * EDSCALE);
	main_container->add_theme_constant_override("margin_left", 4 * EDSCALE);
	main_container->add_theme_constant_override("margin_bottom", 4 * EDSCALE);

	HBoxContainer *main_hb = memnew(HBoxContainer);
	main_container->add_child(main_hb);

	VBoxContainer *first_vb = memnew(VBoxContainer);
	main_hb->add_child(first_vb);
	first_vb->set_h_size_flags(SIZE_EXPAND_FILL);
	first_vb->add_theme_constant_override("separation", 10 * EDSCALE);

	first_vb->add_child(memnew(Label("Label")));

	first_vb->add_child(memnew(Button("Button")));
	Button *bt = memnew(Button);
	bt->set_text(TTR("Toggle Button"));
	bt->set_toggle_mode(true);
	bt->set_pressed(true);
	first_vb->add_child(bt);
	bt = memnew(Button);
	bt->set_text(TTR("Disabled Button"));
	bt->set_disabled(true);
	first_vb->add_child(bt);
	Button *tb = memnew(Button);
	tb->set_flat(true);
	tb->set_text("Button");
	first_vb->add_child(tb);

	CheckButton *cb = memnew(CheckButton);
	cb->set_text("CheckButton");
	first_vb->add_child(cb);
	CheckBox *cbx = memnew(CheckBox);
	cbx->set_text("CheckBox");
	first_vb->add_child(cbx);

	MenuButton *test_menu_button = memnew(MenuButton);
	test_menu_button->set_text("MenuButton");
	test_menu_button->get_popup()->add_item(TTR("Item"));
	test_menu_button->get_popup()->add_item(TTR("Disabled Item"));
	test_menu_button->get_popup()->set_item_disabled(1, true);
	test_menu_button->get_popup()->add_separator();
	test_menu_button->get_popup()->add_check_item(TTR("Check Item"));
	test_menu_button->get_popup()->add_check_item(TTR("Checked Item"));
	test_menu_button->get_popup()->set_item_checked(4, true);
	test_menu_button->get_popup()->add_separator();
	test_menu_button->get_popup()->add_radio_check_item(TTR("Radio Item"));
	test_menu_button->get_popup()->add_radio_check_item(TTR("Checked Radio Item"));
	test_menu_button->get_popup()->set_item_checked(7, true);
	test_menu_button->get_popup()->add_separator(TTR("Named Sep."));

	PopupMenu *test_submenu = memnew(PopupMenu);
	test_menu_button->get_popup()->add_child(test_submenu);
	test_submenu->set_name("submenu");
	test_menu_button->get_popup()->add_submenu_item(TTR("Submenu"), "submenu");
	test_submenu->add_item(TTR("Subitem 1"));
	test_submenu->add_item(TTR("Subitem 2"));
	first_vb->add_child(test_menu_button);

	OptionButton *test_option_button = memnew(OptionButton);
	test_option_button->add_item("OptionButton");
	test_option_button->add_separator();
	test_option_button->add_item(TTR("Has"));
	test_option_button->add_item(TTR("Many"));
	test_option_button->add_item(TTR("Options"));
	first_vb->add_child(test_option_button);
	first_vb->add_child(memnew(ColorPickerButton));

	VBoxContainer *second_vb = memnew(VBoxContainer);
	second_vb->set_h_size_flags(SIZE_EXPAND_FILL);
	main_hb->add_child(second_vb);
	second_vb->add_theme_constant_override("separation", 10 * EDSCALE);
	LineEdit *le = memnew(LineEdit);
	le->set_text("LineEdit");
	second_vb->add_child(le);
	le = memnew(LineEdit);
	le->set_text(TTR("Disabled LineEdit"));
	le->set_editable(false);
	second_vb->add_child(le);
	TextEdit *te = memnew(TextEdit);
	te->set_text("TextEdit");
	te->set_custom_minimum_size(Size2(0, 100) * EDSCALE);
	second_vb->add_child(te);
	second_vb->add_child(memnew(SpinBox));

	HBoxContainer *vhb = memnew(HBoxContainer);
	second_vb->add_child(vhb);
	vhb->set_custom_minimum_size(Size2(0, 100) * EDSCALE);
	vhb->add_child(memnew(VSlider));
	VScrollBar *vsb = memnew(VScrollBar);
	vsb->set_page(25);
	vhb->add_child(vsb);
	vhb->add_child(memnew(VSeparator));
	VBoxContainer *hvb = memnew(VBoxContainer);
	vhb->add_child(hvb);
	hvb->set_alignment(ALIGN_CENTER);
	hvb->set_h_size_flags(SIZE_EXPAND_FILL);
	hvb->add_child(memnew(HSlider));
	HScrollBar *hsb = memnew(HScrollBar);
	hsb->set_page(25);
	hvb->add_child(hsb);
	HSlider *hs = memnew(HSlider);
	hs->set_editable(false);
	hvb->add_child(hs);
	hvb->add_child(memnew(HSeparator));
	ProgressBar *pb = memnew(ProgressBar);
	pb->set_value(50);
	hvb->add_child(pb);

	VBoxContainer *third_vb = memnew(VBoxContainer);
	third_vb->set_h_size_flags(SIZE_EXPAND_FILL);
	third_vb->add_theme_constant_override("separation", 10 * EDSCALE);
	main_hb->add_child(third_vb);

	TabContainer *tc = memnew(TabContainer);
	third_vb->add_child(tc);
	tc->set_custom_minimum_size(Size2(0, 135) * EDSCALE);
	Control *tcc = memnew(Control);
	tcc->set_name(TTR("Tab 1"));
	tc->add_child(tcc);
	tcc = memnew(Control);
	tcc->set_name(TTR("Tab 2"));
	tc->add_child(tcc);
	tcc = memnew(Control);
	tcc->set_name(TTR("Tab 3"));
	tc->add_child(tcc);
	tc->set_tab_disabled(2, true);

	Tree *test_tree = memnew(Tree);
	third_vb->add_child(test_tree);
	test_tree->set_custom_minimum_size(Size2(0, 175) * EDSCALE);
	test_tree->add_theme_constant_override("draw_relationship_lines", 1);

	TreeItem *item = test_tree->create_item();
	item->set_text(0, "Tree");
	item = test_tree->create_item(test_tree->get_root());
	item->set_text(0, "Item");
	item = test_tree->create_item(test_tree->get_root());
	item->set_editable(0, true);
	item->set_text(0, TTR("Editable Item"));
	TreeItem *sub_tree = test_tree->create_item(test_tree->get_root());
	sub_tree->set_text(0, TTR("Subtree"));
	item = test_tree->create_item(sub_tree);
	item->set_cell_mode(0, TreeItem::CELL_MODE_CHECK);
	item->set_editable(0, true);
	item->set_text(0, "Check Item");
	item = test_tree->create_item(sub_tree);
	item->set_cell_mode(0, TreeItem::CELL_MODE_RANGE);
	item->set_editable(0, true);
	item->set_range_config(0, 0, 20, 0.1);
	item->set_range(0, 2);
	item = test_tree->create_item(sub_tree);
	item->set_cell_mode(0, TreeItem::CELL_MODE_RANGE);
	item->set_editable(0, true);
	item->set_text(0, TTR("Has,Many,Options"));
	item->set_range(0, 2);

	main_hb->add_theme_constant_override("separation", 20 * EDSCALE);

	theme_edit_dialog = memnew(ThemeItemEditorDialog);
	theme_edit_dialog->hide();
	add_child(theme_edit_dialog);

	file_dialog = memnew(EditorFileDialog);
	file_dialog->add_filter("*.theme ; " + TTR("Theme File"));
	add_child(file_dialog);
	file_dialog->connect("file_selected", callable_mp(this, &ThemeEditor::_save_template_cbk));
}

void ThemeEditorPlugin::edit(Object *p_node) {
	if (Object::cast_to<Theme>(p_node)) {
		theme_editor->edit(Object::cast_to<Theme>(p_node));
	} else {
		theme_editor->edit(Ref<Theme>());
	}
}

bool ThemeEditorPlugin::handles(Object *p_node) const {
	return p_node->is_class("Theme");
}

void ThemeEditorPlugin::make_visible(bool p_visible) {
	if (p_visible) {
		theme_editor->set_process(true);
		button->show();
		editor->make_bottom_panel_item_visible(theme_editor);
	} else {
		theme_editor->set_process(false);
		if (theme_editor->is_visible_in_tree()) {
			editor->hide_bottom_panel();
		}

		button->hide();
	}
}

ThemeEditorPlugin::ThemeEditorPlugin(EditorNode *p_node) {
	editor = p_node;
	theme_editor = memnew(ThemeEditor);
	theme_editor->set_custom_minimum_size(Size2(0, 200) * EDSCALE);

	button = editor->add_bottom_panel_item(TTR("Theme"), theme_editor);
	button->hide();
}
