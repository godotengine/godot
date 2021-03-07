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
#include "editor/progress_dialog.h"
#include "scene/gui/progress_bar.h"

void ThemeItemImportTree::_update_items_tree() {
	import_items_tree->clear();
	TreeItem *root = import_items_tree->create_item();

	if (base_theme.is_null()) {
		return;
	}

	String filter_text = import_items_filter->get_text();

	List<StringName> types;
	List<StringName> names;
	List<StringName> filtered_names;
	base_theme->get_type_list(&types);
	types.sort_custom<StringName::AlphCompare>();

	int color_amount = 0;
	int constant_amount = 0;
	int font_amount = 0;
	int font_size_amount = 0;
	int icon_amount = 0;
	int stylebox_amount = 0;

	tree_color_items.clear();
	tree_constant_items.clear();
	tree_font_items.clear();
	tree_font_size_items.clear();
	tree_icon_items.clear();
	tree_stylebox_items.clear();

	for (List<StringName>::Element *E = types.front(); E; E = E->next()) {
		String type_name = (String)E->get();

		TreeItem *type_node = import_items_tree->create_item(root);
		type_node->set_meta("_can_be_imported", false);
		type_node->set_collapsed(true);
		type_node->set_text(0, type_name);
		type_node->set_cell_mode(IMPORT_ITEM, TreeItem::CELL_MODE_CHECK);
		type_node->set_checked(IMPORT_ITEM, false);
		type_node->set_editable(IMPORT_ITEM, true);
		type_node->set_cell_mode(IMPORT_ITEM_DATA, TreeItem::CELL_MODE_CHECK);
		type_node->set_checked(IMPORT_ITEM_DATA, false);
		type_node->set_editable(IMPORT_ITEM_DATA, true);

		bool is_matching_filter = (filter_text.is_empty() || type_name.findn(filter_text) > -1);
		bool has_filtered_items = false;
		bool any_checked = false;
		bool any_checked_with_data = false;

		for (int i = 0; i < Theme::DATA_TYPE_MAX; i++) {
			Theme::DataType dt = (Theme::DataType)i;

			names.clear();
			filtered_names.clear();
			base_theme->get_theme_item_list(dt, E->get(), &names);

			bool data_type_has_filtered_items = false;

			for (List<StringName>::Element *F = names.front(); F; F = F->next()) {
				String item_name = (String)F->get();
				bool is_item_matching_filter = (item_name.findn(filter_text) > -1);
				if (!filter_text.is_empty() && !is_matching_filter && !is_item_matching_filter) {
					continue;
				}

				// Only mark this if actual items match the filter and not just the type group.
				if (!filter_text.is_empty() && is_item_matching_filter) {
					has_filtered_items = true;
					data_type_has_filtered_items = true;
				}
				filtered_names.push_back(F->get());
			}

			if (filtered_names.size() == 0) {
				continue;
			}

			TreeItem *data_type_node = import_items_tree->create_item(type_node);
			data_type_node->set_meta("_can_be_imported", false);
			data_type_node->set_metadata(0, i);
			data_type_node->set_collapsed(!data_type_has_filtered_items);
			data_type_node->set_cell_mode(IMPORT_ITEM, TreeItem::CELL_MODE_CHECK);
			data_type_node->set_checked(IMPORT_ITEM, false);
			data_type_node->set_editable(IMPORT_ITEM, true);
			data_type_node->set_cell_mode(IMPORT_ITEM_DATA, TreeItem::CELL_MODE_CHECK);
			data_type_node->set_checked(IMPORT_ITEM_DATA, false);
			data_type_node->set_editable(IMPORT_ITEM_DATA, true);

			List<TreeItem *> *item_list;

			switch (dt) {
				case Theme::DATA_TYPE_COLOR:
					data_type_node->set_icon(0, get_theme_icon("Color", "EditorIcons"));
					data_type_node->set_text(0, TTR("Colors"));

					item_list = &tree_color_items;
					color_amount += filtered_names.size();
					break;

				case Theme::DATA_TYPE_CONSTANT:
					data_type_node->set_icon(0, get_theme_icon("MemberConstant", "EditorIcons"));
					data_type_node->set_text(0, TTR("Constants"));

					item_list = &tree_constant_items;
					constant_amount += filtered_names.size();
					break;

				case Theme::DATA_TYPE_FONT:
					data_type_node->set_icon(0, get_theme_icon("Font", "EditorIcons"));
					data_type_node->set_text(0, TTR("Fonts"));

					item_list = &tree_font_items;
					font_amount += filtered_names.size();
					break;

				case Theme::DATA_TYPE_FONT_SIZE:
					data_type_node->set_icon(0, get_theme_icon("FontSize", "EditorIcons"));
					data_type_node->set_text(0, TTR("Font Sizes"));

					item_list = &tree_font_size_items;
					font_size_amount += filtered_names.size();
					break;

				case Theme::DATA_TYPE_ICON:
					data_type_node->set_icon(0, get_theme_icon("ImageTexture", "EditorIcons"));
					data_type_node->set_text(0, TTR("Icons"));

					item_list = &tree_icon_items;
					icon_amount += filtered_names.size();
					break;

				case Theme::DATA_TYPE_STYLEBOX:
					data_type_node->set_icon(0, get_theme_icon("StyleBoxFlat", "EditorIcons"));
					data_type_node->set_text(0, TTR("Styleboxes"));

					item_list = &tree_stylebox_items;
					stylebox_amount += filtered_names.size();
					break;

				case Theme::DATA_TYPE_MAX:
					break; // Can't happen, but silences warning.
			}

			bool data_type_any_checked = false;
			bool data_type_any_checked_with_data = false;

			filtered_names.sort_custom<StringName::AlphCompare>();
			for (List<StringName>::Element *F = filtered_names.front(); F; F = F->next()) {
				TreeItem *item_node = import_items_tree->create_item(data_type_node);
				item_node->set_meta("_can_be_imported", true);
				item_node->set_text(0, F->get());
				item_node->set_cell_mode(IMPORT_ITEM, TreeItem::CELL_MODE_CHECK);
				item_node->set_checked(IMPORT_ITEM, false);
				item_node->set_editable(IMPORT_ITEM, true);
				item_node->set_cell_mode(IMPORT_ITEM_DATA, TreeItem::CELL_MODE_CHECK);
				item_node->set_checked(IMPORT_ITEM_DATA, false);
				item_node->set_editable(IMPORT_ITEM_DATA, true);

				_restore_selected_item(item_node);
				if (item_node->is_checked(IMPORT_ITEM)) {
					data_type_any_checked = true;
					any_checked = true;
				}
				if (item_node->is_checked(IMPORT_ITEM_DATA)) {
					data_type_any_checked_with_data = true;
					any_checked_with_data = true;
				}

				item_list->push_back(item_node);
			}

			data_type_node->set_checked(IMPORT_ITEM, data_type_any_checked);
			data_type_node->set_checked(IMPORT_ITEM_DATA, data_type_any_checked && data_type_any_checked_with_data);
		}

		// Remove the item if it doesn't match the filter in any way.
		if (!is_matching_filter && !has_filtered_items) {
			root->remove_child(type_node);
			memdelete(type_node);
			continue;
		}

		// Show one level inside of a type group if there are matches in items.
		if (!filter_text.is_empty() && has_filtered_items) {
			type_node->set_collapsed(false);
		}

		type_node->set_checked(IMPORT_ITEM, any_checked);
		type_node->set_checked(IMPORT_ITEM_DATA, any_checked && any_checked_with_data);
	}

	if (color_amount > 0) {
		Array arr;
		arr.push_back(color_amount);
		select_colors_label->set_text(TTRN("One color", "{num} colors", color_amount).format(arr, "{num}"));
		select_all_colors_button->set_visible(true);
		select_full_colors_button->set_visible(true);
		deselect_all_colors_button->set_visible(true);
	} else {
		select_colors_label->set_text(TTR("No colors found."));
		select_all_colors_button->set_visible(false);
		select_full_colors_button->set_visible(false);
		deselect_all_colors_button->set_visible(false);
	}

	if (constant_amount > 0) {
		Array arr;
		arr.push_back(constant_amount);
		select_constants_label->set_text(TTRN("One constant", "{num} constants", constant_amount).format(arr, "{num}"));
		select_all_constants_button->set_visible(true);
		select_full_constants_button->set_visible(true);
		deselect_all_constants_button->set_visible(true);
	} else {
		select_constants_label->set_text(TTR("No constants found."));
		select_all_constants_button->set_visible(false);
		select_full_constants_button->set_visible(false);
		deselect_all_constants_button->set_visible(false);
	}

	if (font_amount > 0) {
		Array arr;
		arr.push_back(font_amount);
		select_fonts_label->set_text(TTRN("One font", "{num} fonts", font_amount).format(arr, "{num}"));
		select_all_fonts_button->set_visible(true);
		select_full_fonts_button->set_visible(true);
		deselect_all_fonts_button->set_visible(true);
	} else {
		select_fonts_label->set_text(TTR("No fonts found."));
		select_all_fonts_button->set_visible(false);
		select_full_fonts_button->set_visible(false);
		deselect_all_fonts_button->set_visible(false);
	}

	if (font_size_amount > 0) {
		Array arr;
		arr.push_back(font_size_amount);
		select_font_sizes_label->set_text(TTRN("One font size", "{num} font sizes", font_size_amount).format(arr, "{num}"));
		select_all_font_sizes_button->set_visible(true);
		select_full_font_sizes_button->set_visible(true);
		deselect_all_font_sizes_button->set_visible(true);
	} else {
		select_font_sizes_label->set_text(TTR("No font sizes found."));
		select_all_font_sizes_button->set_visible(false);
		select_full_font_sizes_button->set_visible(false);
		deselect_all_font_sizes_button->set_visible(false);
	}

	if (icon_amount > 0) {
		Array arr;
		arr.push_back(icon_amount);
		select_icons_label->set_text(TTRN("One icon", "{num} icons", icon_amount).format(arr, "{num}"));
		select_all_icons_button->set_visible(true);
		select_full_icons_button->set_visible(true);
		deselect_all_icons_button->set_visible(true);
		select_icons_warning_hb->set_visible(true);
	} else {
		select_icons_label->set_text(TTR("No icons found."));
		select_all_icons_button->set_visible(false);
		select_full_icons_button->set_visible(false);
		deselect_all_icons_button->set_visible(false);
		select_icons_warning_hb->set_visible(false);
	}

	if (stylebox_amount > 0) {
		Array arr;
		arr.push_back(stylebox_amount);
		select_styleboxes_label->set_text(TTRN("One stylebox", "{num} styleboxes", stylebox_amount).format(arr, "{num}"));
		select_all_styleboxes_button->set_visible(true);
		select_full_styleboxes_button->set_visible(true);
		deselect_all_styleboxes_button->set_visible(true);
	} else {
		select_styleboxes_label->set_text(TTR("No styleboxes found."));
		select_all_styleboxes_button->set_visible(false);
		select_full_styleboxes_button->set_visible(false);
		deselect_all_styleboxes_button->set_visible(false);
	}
}

void ThemeItemImportTree::_toggle_type_items(bool p_collapse) {
	TreeItem *root = import_items_tree->get_root();
	if (!root) {
		return;
	}

	TreeItem *type_node = root->get_first_child();
	while (type_node) {
		type_node->set_collapsed(p_collapse);
		type_node = type_node->get_next();
	}
}

void ThemeItemImportTree::_filter_text_changed(const String &p_value) {
	_update_items_tree();
}

void ThemeItemImportTree::_store_selected_item(TreeItem *p_tree_item) {
	if (!p_tree_item->get_meta("_can_be_imported")) {
		return;
	}

	TreeItem *data_type_node = p_tree_item->get_parent();
	if (!data_type_node || data_type_node == import_items_tree->get_root()) {
		return;
	}

	TreeItem *type_node = data_type_node->get_parent();
	if (!type_node || type_node == import_items_tree->get_root()) {
		return;
	}

	ThemeItem ti;
	ti.item_name = p_tree_item->get_text(0);
	ti.data_type = (Theme::DataType)(int)data_type_node->get_metadata(0);
	ti.type_name = type_node->get_text(0);

	bool import = p_tree_item->is_checked(IMPORT_ITEM);
	bool with_data = p_tree_item->is_checked(IMPORT_ITEM_DATA);

	if (import && with_data) {
		selected_items[ti] = SELECT_IMPORT_FULL;
	} else if (import) {
		selected_items[ti] = SELECT_IMPORT_DEFINITION;
	} else {
		selected_items.erase(ti);
	}

	_update_total_selected(ti.data_type);
}

void ThemeItemImportTree::_restore_selected_item(TreeItem *p_tree_item) {
	if (!p_tree_item->get_meta("_can_be_imported")) {
		return;
	}

	TreeItem *data_type_node = p_tree_item->get_parent();
	if (!data_type_node || data_type_node == import_items_tree->get_root()) {
		return;
	}

	TreeItem *type_node = data_type_node->get_parent();
	if (!type_node || type_node == import_items_tree->get_root()) {
		return;
	}

	ThemeItem ti;
	ti.item_name = p_tree_item->get_text(0);
	ti.data_type = (Theme::DataType)(int)data_type_node->get_metadata(0);
	ti.type_name = type_node->get_text(0);

	if (!selected_items.has(ti)) {
		p_tree_item->set_checked(IMPORT_ITEM, false);
		p_tree_item->set_checked(IMPORT_ITEM_DATA, false);
		return;
	}

	if (selected_items[ti] == SELECT_IMPORT_FULL) {
		p_tree_item->set_checked(IMPORT_ITEM, true);
		p_tree_item->set_checked(IMPORT_ITEM_DATA, true);
	} else if (selected_items[ti] == SELECT_IMPORT_DEFINITION) {
		p_tree_item->set_checked(IMPORT_ITEM, true);
		p_tree_item->set_checked(IMPORT_ITEM_DATA, false);
	}
}

void ThemeItemImportTree::_update_total_selected(Theme::DataType p_data_type) {
	ERR_FAIL_INDEX_MSG(p_data_type, Theme::DATA_TYPE_MAX, "Theme item data type is out of bounds.");

	Label *total_selected_items_label;
	switch (p_data_type) {
		case Theme::DATA_TYPE_COLOR:
			total_selected_items_label = total_selected_colors_label;
			break;

		case Theme::DATA_TYPE_CONSTANT:
			total_selected_items_label = total_selected_constants_label;
			break;

		case Theme::DATA_TYPE_FONT:
			total_selected_items_label = total_selected_fonts_label;
			break;

		case Theme::DATA_TYPE_FONT_SIZE:
			total_selected_items_label = total_selected_font_sizes_label;
			break;

		case Theme::DATA_TYPE_ICON:
			total_selected_items_label = total_selected_icons_label;
			break;

		case Theme::DATA_TYPE_STYLEBOX:
			total_selected_items_label = total_selected_styleboxes_label;
			break;

		case Theme::DATA_TYPE_MAX:
			return; // Can't happen, but silences warning.
	}

	if (!total_selected_items_label) {
		return;
	}

	int count = 0;
	for (Map<ThemeItem, ItemCheckedState>::Element *E = selected_items.front(); E; E = E->next()) {
		ThemeItem ti = E->key();
		if (ti.data_type == p_data_type) {
			count++;
		}
	}

	if (count == 0) {
		total_selected_items_label->hide();
	} else {
		Array arr;
		arr.push_back(count);
		total_selected_items_label->set_text(TTRN("{num} currently selected", "{num} currently selected", count).format(arr, "{num}"));
		total_selected_items_label->show();
	}
}

void ThemeItemImportTree::_tree_item_edited() {
	if (updating_tree) {
		return;
	}

	TreeItem *edited_item = import_items_tree->get_edited();
	if (!edited_item) {
		return;
	}

	updating_tree = true;

	int edited_column = import_items_tree->get_edited_column();
	bool is_checked = edited_item->is_checked(edited_column);
	if (is_checked) {
		if (edited_column == IMPORT_ITEM_DATA) {
			edited_item->set_checked(IMPORT_ITEM, true);
		}

		_select_all_subitems(edited_item, (edited_column == IMPORT_ITEM_DATA));
	} else {
		if (edited_column == IMPORT_ITEM) {
			edited_item->set_checked(IMPORT_ITEM_DATA, false);
		}

		_deselect_all_subitems(edited_item, (edited_column == IMPORT_ITEM));
	}

	_update_parent_items(edited_item);
	_store_selected_item(edited_item);

	updating_tree = false;
}

void ThemeItemImportTree::_select_all_subitems(TreeItem *p_root_item, bool p_select_with_data) {
	TreeItem *child_item = p_root_item->get_first_child();
	while (child_item) {
		child_item->set_checked(IMPORT_ITEM, true);
		if (p_select_with_data) {
			child_item->set_checked(IMPORT_ITEM_DATA, true);
		}
		_store_selected_item(child_item);

		_select_all_subitems(child_item, p_select_with_data);
		child_item = child_item->get_next();
	}
}

void ThemeItemImportTree::_deselect_all_subitems(TreeItem *p_root_item, bool p_deselect_completely) {
	TreeItem *child_item = p_root_item->get_first_child();
	while (child_item) {
		child_item->set_checked(IMPORT_ITEM_DATA, false);
		if (p_deselect_completely) {
			child_item->set_checked(IMPORT_ITEM, false);
		}
		_store_selected_item(child_item);

		_deselect_all_subitems(child_item, p_deselect_completely);
		child_item = child_item->get_next();
	}
}

void ThemeItemImportTree::_update_parent_items(TreeItem *p_root_item) {
	TreeItem *parent_item = p_root_item->get_parent();
	if (!parent_item) {
		return;
	}

	bool any_checked = false;
	bool any_checked_with_data = false;

	TreeItem *child_item = parent_item->get_first_child();
	while (child_item) {
		if (child_item->is_checked(IMPORT_ITEM)) {
			any_checked = true;
		}
		if (child_item->is_checked(IMPORT_ITEM_DATA)) {
			any_checked_with_data = true;
		}

		child_item = child_item->get_next();
	}

	parent_item->set_checked(IMPORT_ITEM, any_checked);
	parent_item->set_checked(IMPORT_ITEM_DATA, any_checked && any_checked_with_data);
	_update_parent_items(parent_item);
}

void ThemeItemImportTree::_select_all_items_pressed() {
	if (updating_tree) {
		return;
	}

	updating_tree = true;

	TreeItem *root = import_items_tree->get_root();
	_select_all_subitems(root, false);

	updating_tree = false;
}

void ThemeItemImportTree::_select_full_items_pressed() {
	if (updating_tree) {
		return;
	}

	updating_tree = true;

	TreeItem *root = import_items_tree->get_root();
	_select_all_subitems(root, true);

	updating_tree = false;
}

void ThemeItemImportTree::_deselect_all_items_pressed() {
	if (updating_tree) {
		return;
	}

	updating_tree = true;

	TreeItem *root = import_items_tree->get_root();
	_deselect_all_subitems(root, true);

	updating_tree = false;
}

void ThemeItemImportTree::_select_all_data_type_pressed(int p_data_type) {
	ERR_FAIL_INDEX_MSG(p_data_type, Theme::DATA_TYPE_MAX, "Theme item data type is out of bounds.");

	if (updating_tree) {
		return;
	}

	Theme::DataType data_type = (Theme::DataType)p_data_type;
	List<TreeItem *> *item_list;

	switch (data_type) {
		case Theme::DATA_TYPE_COLOR:
			item_list = &tree_color_items;
			break;

		case Theme::DATA_TYPE_CONSTANT:
			item_list = &tree_constant_items;
			break;

		case Theme::DATA_TYPE_FONT:
			item_list = &tree_font_items;
			break;

		case Theme::DATA_TYPE_FONT_SIZE:
			item_list = &tree_font_size_items;
			break;

		case Theme::DATA_TYPE_ICON:
			item_list = &tree_icon_items;
			break;

		case Theme::DATA_TYPE_STYLEBOX:
			item_list = &tree_stylebox_items;
			break;

		case Theme::DATA_TYPE_MAX:
			return; // Can't happen, but silences warning.
	}

	updating_tree = true;

	for (List<TreeItem *>::Element *E = item_list->front(); E; E = E->next()) {
		TreeItem *child_item = E->get();
		if (!child_item) {
			continue;
		}

		child_item->set_checked(IMPORT_ITEM, true);
		_update_parent_items(child_item);
		_store_selected_item(child_item);
	}

	updating_tree = false;
}

void ThemeItemImportTree::_select_full_data_type_pressed(int p_data_type) {
	ERR_FAIL_INDEX_MSG(p_data_type, Theme::DATA_TYPE_MAX, "Theme item data type is out of bounds.");

	if (updating_tree) {
		return;
	}

	Theme::DataType data_type = (Theme::DataType)p_data_type;
	List<TreeItem *> *item_list;

	switch (data_type) {
		case Theme::DATA_TYPE_COLOR:
			item_list = &tree_color_items;
			break;

		case Theme::DATA_TYPE_CONSTANT:
			item_list = &tree_constant_items;
			break;

		case Theme::DATA_TYPE_FONT:
			item_list = &tree_font_items;
			break;

		case Theme::DATA_TYPE_FONT_SIZE:
			item_list = &tree_font_size_items;
			break;

		case Theme::DATA_TYPE_ICON:
			item_list = &tree_icon_items;
			break;

		case Theme::DATA_TYPE_STYLEBOX:
			item_list = &tree_stylebox_items;
			break;

		case Theme::DATA_TYPE_MAX:
			return; // Can't happen, but silences warning.
	}

	updating_tree = true;

	for (List<TreeItem *>::Element *E = item_list->front(); E; E = E->next()) {
		TreeItem *child_item = E->get();
		if (!child_item) {
			continue;
		}

		child_item->set_checked(IMPORT_ITEM, true);
		child_item->set_checked(IMPORT_ITEM_DATA, true);
		_update_parent_items(child_item);
		_store_selected_item(child_item);
	}

	updating_tree = false;
}

void ThemeItemImportTree::_deselect_all_data_type_pressed(int p_data_type) {
	ERR_FAIL_INDEX_MSG(p_data_type, Theme::DATA_TYPE_MAX, "Theme item data type is out of bounds.");

	if (updating_tree) {
		return;
	}

	Theme::DataType data_type = (Theme::DataType)p_data_type;
	List<TreeItem *> *item_list;

	switch (data_type) {
		case Theme::DATA_TYPE_COLOR:
			item_list = &tree_color_items;
			break;

		case Theme::DATA_TYPE_CONSTANT:
			item_list = &tree_constant_items;
			break;

		case Theme::DATA_TYPE_FONT:
			item_list = &tree_font_items;
			break;

		case Theme::DATA_TYPE_FONT_SIZE:
			item_list = &tree_font_size_items;
			break;

		case Theme::DATA_TYPE_ICON:
			item_list = &tree_icon_items;
			break;

		case Theme::DATA_TYPE_STYLEBOX:
			item_list = &tree_stylebox_items;
			break;

		case Theme::DATA_TYPE_MAX:
			return; // Can't happen, but silences warning.
	}

	updating_tree = true;

	for (List<TreeItem *>::Element *E = item_list->front(); E; E = E->next()) {
		TreeItem *child_item = E->get();
		if (!child_item) {
			continue;
		}

		child_item->set_checked(IMPORT_ITEM, false);
		child_item->set_checked(IMPORT_ITEM_DATA, false);
		_update_parent_items(child_item);
		_store_selected_item(child_item);
	}

	updating_tree = false;
}

void ThemeItemImportTree::_import_selected() {
	if (selected_items.size() == 0) {
		EditorNode::get_singleton()->show_accept(TTR("Nothing was selected for the import."), TTR("OK"));
		return;
	}

	ProgressDialog::get_singleton()->add_task("import_theme_items", TTR("Importing Theme Items"), selected_items.size());

	int idx = 0;
	for (Map<ThemeItem, ItemCheckedState>::Element *E = selected_items.front(); E; E = E->next()) {
		// Arbitrary number of items to skip from reporting.
		// Reduces the number of UI updates that this causes when copying large themes.
		if (idx % 10 == 0) {
			Array arr;
			arr.push_back(idx + 1);
			arr.push_back(selected_items.size());
			ProgressDialog::get_singleton()->task_step("import_theme_items", TTR("Importing items {n}/{n}").format(arr, "{n}"), idx);
		}

		ItemCheckedState cs = E->get();
		ThemeItem ti = E->key();

		if (cs == SELECT_IMPORT_DEFINITION || cs == SELECT_IMPORT_FULL) {
			Variant item_value = Variant();

			if (cs == SELECT_IMPORT_FULL) {
				item_value = base_theme->get_theme_item(ti.data_type, ti.item_name, ti.type_name);
			} else {
				switch (ti.data_type) {
					case Theme::DATA_TYPE_COLOR:
						item_value = Color();
						break;

					case Theme::DATA_TYPE_CONSTANT:
						item_value = 0;
						break;

					case Theme::DATA_TYPE_FONT:
						item_value = Ref<Font>();
						break;

					case Theme::DATA_TYPE_FONT_SIZE:
						item_value = -1;
						break;

					case Theme::DATA_TYPE_ICON:
						item_value = Ref<Texture2D>();
						break;

					case Theme::DATA_TYPE_STYLEBOX:
						item_value = Ref<StyleBox>();
						break;

					case Theme::DATA_TYPE_MAX:
						break; // Can't happen, but silences warning.
				}
			}

			edited_theme->set_theme_item(ti.data_type, ti.item_name, ti.type_name, item_value);
		}

		idx++;
	}

	ProgressDialog::get_singleton()->end_task("import_theme_items");
	emit_signal("items_imported");
}

void ThemeItemImportTree::set_edited_theme(const Ref<Theme> &p_theme) {
	edited_theme = p_theme;
}

void ThemeItemImportTree::set_base_theme(const Ref<Theme> &p_theme) {
	base_theme = p_theme;
}

void ThemeItemImportTree::reset_item_tree() {
	import_items_filter->clear();
	selected_items.clear();

	total_selected_colors_label->hide();
	total_selected_constants_label->hide();
	total_selected_fonts_label->hide();
	total_selected_font_sizes_label->hide();
	total_selected_icons_label->hide();
	total_selected_styleboxes_label->hide();

	_update_items_tree();
}

bool ThemeItemImportTree::has_selected_items() const {
	return (selected_items.size() > 0);
}

void ThemeItemImportTree::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE:
		case NOTIFICATION_THEME_CHANGED: {
			select_icons_warning_icon->set_texture(get_theme_icon("StatusWarning", "EditorIcons"));
			select_icons_warning->add_theme_color_override("font_color", get_theme_color("disabled_font_color", "Editor"));

			// Bottom panel buttons.
			import_collapse_types_button->set_icon(get_theme_icon("CollapseTree", "EditorIcons"));
			import_expand_types_button->set_icon(get_theme_icon("ExpandTree", "EditorIcons"));

			import_select_all_button->set_icon(get_theme_icon("ThemeSelectAll", "EditorIcons"));
			import_select_full_button->set_icon(get_theme_icon("ThemeSelectFull", "EditorIcons"));
			import_deselect_all_button->set_icon(get_theme_icon("ThemeDeselectAll", "EditorIcons"));

			// Side panel buttons.
			select_colors_icon->set_texture(get_theme_icon("Color", "EditorIcons"));
			deselect_all_colors_button->set_icon(get_theme_icon("ThemeDeselectAll", "EditorIcons"));
			select_all_colors_button->set_icon(get_theme_icon("ThemeSelectAll", "EditorIcons"));
			select_full_colors_button->set_icon(get_theme_icon("ThemeSelectFull", "EditorIcons"));

			select_constants_icon->set_texture(get_theme_icon("MemberConstant", "EditorIcons"));
			deselect_all_constants_button->set_icon(get_theme_icon("ThemeDeselectAll", "EditorIcons"));
			select_all_constants_button->set_icon(get_theme_icon("ThemeSelectAll", "EditorIcons"));
			select_full_constants_button->set_icon(get_theme_icon("ThemeSelectFull", "EditorIcons"));

			select_fonts_icon->set_texture(get_theme_icon("Font", "EditorIcons"));
			deselect_all_fonts_button->set_icon(get_theme_icon("ThemeDeselectAll", "EditorIcons"));
			select_all_fonts_button->set_icon(get_theme_icon("ThemeSelectAll", "EditorIcons"));
			select_full_fonts_button->set_icon(get_theme_icon("ThemeSelectFull", "EditorIcons"));

			select_font_sizes_icon->set_texture(get_theme_icon("FontSize", "EditorIcons"));
			deselect_all_font_sizes_button->set_icon(get_theme_icon("ThemeDeselectAll", "EditorIcons"));
			select_all_font_sizes_button->set_icon(get_theme_icon("ThemeSelectAll", "EditorIcons"));
			select_full_font_sizes_button->set_icon(get_theme_icon("ThemeSelectFull", "EditorIcons"));

			select_icons_icon->set_texture(get_theme_icon("ImageTexture", "EditorIcons"));
			deselect_all_icons_button->set_icon(get_theme_icon("ThemeDeselectAll", "EditorIcons"));
			select_all_icons_button->set_icon(get_theme_icon("ThemeSelectAll", "EditorIcons"));
			select_full_icons_button->set_icon(get_theme_icon("ThemeSelectFull", "EditorIcons"));

			select_styleboxes_icon->set_texture(get_theme_icon("StyleBoxFlat", "EditorIcons"));
			deselect_all_styleboxes_button->set_icon(get_theme_icon("ThemeDeselectAll", "EditorIcons"));
			select_all_styleboxes_button->set_icon(get_theme_icon("ThemeSelectAll", "EditorIcons"));
			select_full_styleboxes_button->set_icon(get_theme_icon("ThemeSelectFull", "EditorIcons"));
		} break;
	}
}

void ThemeItemImportTree::_bind_methods() {
	ADD_SIGNAL(MethodInfo("items_imported"));
}

ThemeItemImportTree::ThemeItemImportTree() {
	HBoxContainer *import_items_filter_hb = memnew(HBoxContainer);
	add_child(import_items_filter_hb);
	Label *import_items_filter_label = memnew(Label);
	import_items_filter_label->set_text(TTR("Filter:"));
	import_items_filter_hb->add_child(import_items_filter_label);
	import_items_filter = memnew(LineEdit);
	import_items_filter->set_clear_button_enabled(true);
	import_items_filter->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	import_items_filter_hb->add_child(import_items_filter);
	import_items_filter->connect("text_changed", callable_mp(this, &ThemeItemImportTree::_filter_text_changed));

	HBoxContainer *import_main_hb = memnew(HBoxContainer);
	import_main_hb->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	add_child(import_main_hb);

	import_items_tree = memnew(Tree);
	import_items_tree->set_hide_root(true);
	import_items_tree->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	import_main_hb->add_child(import_items_tree);
	import_items_tree->connect("item_edited", callable_mp(this, &ThemeItemImportTree::_tree_item_edited));

	import_items_tree->set_columns(3);
	import_items_tree->set_column_titles_visible(true);
	import_items_tree->set_column_title(IMPORT_ITEM, TTR("Import"));
	import_items_tree->set_column_title(IMPORT_ITEM_DATA, TTR("With Data"));
	import_items_tree->set_column_expand(0, true);
	import_items_tree->set_column_expand(IMPORT_ITEM, false);
	import_items_tree->set_column_expand(IMPORT_ITEM_DATA, false);
	import_items_tree->set_column_min_width(0, 160 * EDSCALE);
	import_items_tree->set_column_min_width(IMPORT_ITEM, 80 * EDSCALE);
	import_items_tree->set_column_min_width(IMPORT_ITEM_DATA, 80 * EDSCALE);

	ScrollContainer *import_bulk_sc = memnew(ScrollContainer);
	import_bulk_sc->set_custom_minimum_size(Size2(260.0, 0.0) * EDSCALE);
	import_bulk_sc->set_enable_h_scroll(false);
	import_main_hb->add_child(import_bulk_sc);
	VBoxContainer *import_bulk_vb = memnew(VBoxContainer);
	import_bulk_vb->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	import_bulk_sc->add_child(import_bulk_vb);

	Label *import_bulk_label = memnew(Label);
	import_bulk_label->set_text(TTR("Select by data type:"));
	import_bulk_vb->add_child(import_bulk_label);

	select_colors_icon = memnew(TextureRect);
	select_colors_label = memnew(Label);
	deselect_all_colors_button = memnew(Button);
	select_all_colors_button = memnew(Button);
	select_full_colors_button = memnew(Button);
	total_selected_colors_label = memnew(Label);

	select_constants_icon = memnew(TextureRect);
	select_constants_label = memnew(Label);
	deselect_all_constants_button = memnew(Button);
	select_all_constants_button = memnew(Button);
	select_full_constants_button = memnew(Button);
	total_selected_constants_label = memnew(Label);

	select_fonts_icon = memnew(TextureRect);
	select_fonts_label = memnew(Label);
	deselect_all_fonts_button = memnew(Button);
	select_all_fonts_button = memnew(Button);
	select_full_fonts_button = memnew(Button);
	total_selected_fonts_label = memnew(Label);

	select_font_sizes_icon = memnew(TextureRect);
	select_font_sizes_label = memnew(Label);
	deselect_all_font_sizes_button = memnew(Button);
	select_all_font_sizes_button = memnew(Button);
	select_full_font_sizes_button = memnew(Button);
	total_selected_font_sizes_label = memnew(Label);

	select_icons_icon = memnew(TextureRect);
	select_icons_label = memnew(Label);
	deselect_all_icons_button = memnew(Button);
	select_all_icons_button = memnew(Button);
	select_full_icons_button = memnew(Button);
	total_selected_icons_label = memnew(Label);

	select_styleboxes_icon = memnew(TextureRect);
	select_styleboxes_label = memnew(Label);
	deselect_all_styleboxes_button = memnew(Button);
	select_all_styleboxes_button = memnew(Button);
	select_full_styleboxes_button = memnew(Button);
	total_selected_styleboxes_label = memnew(Label);

	for (int i = 0; i < Theme::DATA_TYPE_MAX; i++) {
		Theme::DataType dt = (Theme::DataType)i;

		TextureRect *select_items_icon;
		Label *select_items_label;
		Button *deselect_all_items_button;
		Button *select_all_items_button;
		Button *select_full_items_button;
		Label *total_selected_items_label;

		String items_title = "";
		String select_all_items_tooltip = "";
		String select_full_items_tooltip = "";
		String deselect_all_items_tooltip = "";

		switch (dt) {
			case Theme::DATA_TYPE_COLOR:
				select_items_icon = select_colors_icon;
				select_items_label = select_colors_label;
				deselect_all_items_button = deselect_all_colors_button;
				select_all_items_button = select_all_colors_button;
				select_full_items_button = select_full_colors_button;
				total_selected_items_label = total_selected_colors_label;

				items_title = TTR("Colors");
				select_all_items_tooltip = TTR("Select all visible color items.");
				select_full_items_tooltip = TTR("Select all visible color items and their data.");
				deselect_all_items_tooltip = TTR("Deselect all visible color items.");
				break;

			case Theme::DATA_TYPE_CONSTANT:
				select_items_icon = select_constants_icon;
				select_items_label = select_constants_label;
				deselect_all_items_button = deselect_all_constants_button;
				select_all_items_button = select_all_constants_button;
				select_full_items_button = select_full_constants_button;
				total_selected_items_label = total_selected_constants_label;

				items_title = TTR("Constants");
				select_all_items_tooltip = TTR("Select all visible constant items.");
				select_full_items_tooltip = TTR("Select all visible constant items and their data.");
				deselect_all_items_tooltip = TTR("Deselect all visible constant items.");
				break;

			case Theme::DATA_TYPE_FONT:
				select_items_icon = select_fonts_icon;
				select_items_label = select_fonts_label;
				deselect_all_items_button = deselect_all_fonts_button;
				select_all_items_button = select_all_fonts_button;
				select_full_items_button = select_full_fonts_button;
				total_selected_items_label = total_selected_fonts_label;

				items_title = TTR("Fonts");
				select_all_items_tooltip = TTR("Select all visible font items.");
				select_full_items_tooltip = TTR("Select all visible font items and their data.");
				deselect_all_items_tooltip = TTR("Deselect all visible font items.");
				break;

			case Theme::DATA_TYPE_FONT_SIZE:
				select_items_icon = select_font_sizes_icon;
				select_items_label = select_font_sizes_label;
				deselect_all_items_button = deselect_all_font_sizes_button;
				select_all_items_button = select_all_font_sizes_button;
				select_full_items_button = select_full_font_sizes_button;
				total_selected_items_label = total_selected_font_sizes_label;

				items_title = TTR("Font sizes");
				select_all_items_tooltip = TTR("Select all visible font size items.");
				select_full_items_tooltip = TTR("Select all visible font size items and their data.");
				deselect_all_items_tooltip = TTR("Deselect all visible font size items.");
				break;

			case Theme::DATA_TYPE_ICON:
				select_items_icon = select_icons_icon;
				select_items_label = select_icons_label;
				deselect_all_items_button = deselect_all_icons_button;
				select_all_items_button = select_all_icons_button;
				select_full_items_button = select_full_icons_button;
				total_selected_items_label = total_selected_icons_label;

				items_title = TTR("Icons");
				select_all_items_tooltip = TTR("Select all visible icon items.");
				select_full_items_tooltip = TTR("Select all visible icon items and their data.");
				deselect_all_items_tooltip = TTR("Deselect all visible icon items.");
				break;

			case Theme::DATA_TYPE_STYLEBOX:
				select_items_icon = select_styleboxes_icon;
				select_items_label = select_styleboxes_label;
				deselect_all_items_button = deselect_all_styleboxes_button;
				select_all_items_button = select_all_styleboxes_button;
				select_full_items_button = select_full_styleboxes_button;
				total_selected_items_label = total_selected_styleboxes_label;

				items_title = TTR("Styleboxes");
				select_all_items_tooltip = TTR("Select all visible stylebox items.");
				select_full_items_tooltip = TTR("Select all visible stylebox items and their data.");
				deselect_all_items_tooltip = TTR("Deselect all visible stylebox items.");
				break;

			case Theme::DATA_TYPE_MAX:
				continue; // Can't happen, but silences warning.
		}

		if (i > 0) {
			import_bulk_vb->add_child(memnew(HSeparator));
		}

		HBoxContainer *all_set = memnew(HBoxContainer);
		import_bulk_vb->add_child(all_set);

		HBoxContainer *label_set = memnew(HBoxContainer);
		label_set->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		all_set->add_child(label_set);
		select_items_icon->set_v_size_flags(Control::SIZE_SHRINK_CENTER);
		label_set->add_child(select_items_icon);
		select_items_label->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		select_items_label->set_clip_text(true);
		select_items_label->set_text(items_title);
		label_set->add_child(select_items_label);

		HBoxContainer *button_set = memnew(HBoxContainer);
		button_set->set_alignment(BoxContainer::ALIGN_END);
		all_set->add_child(button_set);
		select_all_items_button->set_flat(true);
		select_all_items_button->set_tooltip(select_all_items_tooltip);
		button_set->add_child(select_all_items_button);
		select_all_items_button->connect("pressed", callable_mp(this, &ThemeItemImportTree::_select_all_data_type_pressed), varray(i));
		select_full_items_button->set_flat(true);
		select_full_items_button->set_tooltip(select_full_items_tooltip);
		button_set->add_child(select_full_items_button);
		select_full_items_button->connect("pressed", callable_mp(this, &ThemeItemImportTree::_select_full_data_type_pressed), varray(i));
		deselect_all_items_button->set_flat(true);
		deselect_all_items_button->set_tooltip(deselect_all_items_tooltip);
		button_set->add_child(deselect_all_items_button);
		deselect_all_items_button->connect("pressed", callable_mp(this, &ThemeItemImportTree::_deselect_all_data_type_pressed), varray(i));

		total_selected_items_label->set_align(Label::ALIGN_RIGHT);
		total_selected_items_label->hide();
		import_bulk_vb->add_child(total_selected_items_label);

		if (dt == Theme::DATA_TYPE_ICON) {
			select_icons_warning_hb = memnew(HBoxContainer);
			import_bulk_vb->add_child(select_icons_warning_hb);

			select_icons_warning_icon = memnew(TextureRect);
			select_icons_warning_icon->set_v_size_flags(Control::SIZE_SHRINK_CENTER);
			select_icons_warning_hb->add_child(select_icons_warning_icon);

			select_icons_warning = memnew(Label);
			select_icons_warning->set_text(TTR("Caution: Adding icon data may considerably increase the size of your Theme resource."));
			select_icons_warning->set_autowrap(true);
			select_icons_warning->set_h_size_flags(Control::SIZE_EXPAND_FILL);
			select_icons_warning_hb->add_child(select_icons_warning);
		}
	}

	add_child(memnew(HSeparator));

	HBoxContainer *import_buttons = memnew(HBoxContainer);
	add_child(import_buttons);

	import_collapse_types_button = memnew(Button);
	import_collapse_types_button->set_flat(true);
	import_collapse_types_button->set_tooltip(TTR("Collapse types."));
	import_buttons->add_child(import_collapse_types_button);
	import_collapse_types_button->connect("pressed", callable_mp(this, &ThemeItemImportTree::_toggle_type_items), varray(true));
	import_expand_types_button = memnew(Button);
	import_expand_types_button->set_flat(true);
	import_expand_types_button->set_tooltip(TTR("Expand types."));
	import_buttons->add_child(import_expand_types_button);
	import_expand_types_button->connect("pressed", callable_mp(this, &ThemeItemImportTree::_toggle_type_items), varray(false));

	import_buttons->add_child(memnew(VSeparator));

	import_select_all_button = memnew(Button);
	import_select_all_button->set_flat(true);
	import_select_all_button->set_text(TTR("Select All"));
	import_select_all_button->set_tooltip(TTR("Select all Theme items."));
	import_buttons->add_child(import_select_all_button);
	import_select_all_button->connect("pressed", callable_mp(this, &ThemeItemImportTree::_select_all_items_pressed));
	import_select_full_button = memnew(Button);
	import_select_full_button->set_flat(true);
	import_select_full_button->set_text(TTR("Select With Data"));
	import_select_full_button->set_tooltip(TTR("Select all Theme items with item data."));
	import_buttons->add_child(import_select_full_button);
	import_select_full_button->connect("pressed", callable_mp(this, &ThemeItemImportTree::_select_full_items_pressed));
	import_deselect_all_button = memnew(Button);
	import_deselect_all_button->set_flat(true);
	import_deselect_all_button->set_text(TTR("Deselect All"));
	import_deselect_all_button->set_tooltip(TTR("Deselect all Theme items."));
	import_buttons->add_child(import_deselect_all_button);
	import_deselect_all_button->connect("pressed", callable_mp(this, &ThemeItemImportTree::_deselect_all_items_pressed));

	import_buttons->add_spacer();

	Button *import_add_selected_button = memnew(Button);
	import_add_selected_button->set_text(TTR("Import Selected"));
	import_buttons->add_child(import_add_selected_button);
	import_add_selected_button->connect("pressed", callable_mp(this, &ThemeItemImportTree::_import_selected));
}

void ThemeItemEditorDialog::ok_pressed() {
	if (import_default_theme_items->has_selected_items() || import_editor_theme_items->has_selected_items() || import_other_theme_items->has_selected_items()) {
		confirm_closing_dialog->set_text(TTR("Import Items tab has some items selected. Selection will be lost upon closing this window.\nClose anyway?"));
		confirm_closing_dialog->popup_centered(Size2i(380, 120) * EDSCALE);
		return;
	}

	hide();
}

void ThemeItemEditorDialog::_close_dialog() {
	hide();
}

void ThemeItemEditorDialog::_dialog_about_to_show() {
	ERR_FAIL_COND(edited_theme.is_null());

	_update_edit_types();

	import_default_theme_items->set_edited_theme(edited_theme);
	import_default_theme_items->set_base_theme(Theme::get_default());
	import_default_theme_items->reset_item_tree();

	import_editor_theme_items->set_edited_theme(edited_theme);
	import_editor_theme_items->set_base_theme(EditorNode::get_singleton()->get_theme_base()->get_theme());
	import_editor_theme_items->reset_item_tree();

	import_other_theme_items->set_edited_theme(edited_theme);
	import_other_theme_items->reset_item_tree();
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

	{ // Colors.
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

	{ // Constants.
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

	{ // Fonts.
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

	{ // Font sizes.
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

	{ // Icons.
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

	{ // Styleboxes.
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

void ThemeItemEditorDialog::_add_theme_type() {
	edited_theme->add_icon_type(edit_add_type_value->get_text());
	edited_theme->add_stylebox_type(edit_add_type_value->get_text());
	edited_theme->add_font_type(edit_add_type_value->get_text());
	edited_theme->add_font_size_type(edit_add_type_value->get_text());
	edited_theme->add_color_type(edit_add_type_value->get_text());
	edited_theme->add_constant_type(edit_add_type_value->get_text());
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

void ThemeItemEditorDialog::_open_select_another_theme() {
	import_another_theme_dialog->popup_file_dialog();
}

void ThemeItemEditorDialog::_select_another_theme_cbk(const String &p_path) {
	Ref<Theme> loaded_theme = ResourceLoader::load(p_path);
	if (loaded_theme.is_null()) {
		EditorNode::get_singleton()->show_warning(TTR("Invalid file, not a Theme resource."));
		return;
	}
	if (loaded_theme == edited_theme) {
		EditorNode::get_singleton()->show_warning(TTR("Invalid file, same as the edited Theme resource."));
		return;
	}

	import_another_theme_value->set_text(p_path);
	import_other_theme_items->set_base_theme(loaded_theme);
	import_other_theme_items->reset_item_tree();
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

			import_another_theme_button->set_icon(get_theme_icon("Folder", "EditorIcons"));

			tc->add_theme_style_override("tab_selected", get_theme_stylebox("tab_selected_odd", "TabContainer"));
			tc->add_theme_style_override("panel", get_theme_stylebox("panel_odd", "TabContainer"));
		} break;
	}
}

void ThemeItemEditorDialog::set_edited_theme(const Ref<Theme> &p_theme) {
	edited_theme = p_theme;
}

ThemeItemEditorDialog::ThemeItemEditorDialog() {
	set_title(TTR("Manage Theme Items"));
	get_ok_button()->set_text(TTR("Close"));
	set_hide_on_ok(false); // Closing may require a confirmation in some cases.

	tc = memnew(TabContainer);
	tc->set_tab_align(TabContainer::TabAlign::ALIGN_LEFT);
	add_child(tc);

	// Edit Items tab.
	HSplitContainer *edit_dialog_hs = memnew(HSplitContainer);
	tc->add_child(edit_dialog_hs);
	tc->set_tab_title(0, TTR("Edit Items"));

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

	Label *edit_add_type_label = memnew(Label);
	edit_add_type_label->set_text(TTR("Add Type:"));
	edit_dialog_side_vb->add_child(edit_add_type_label);

	HBoxContainer *edit_add_type_hb = memnew(HBoxContainer);
	edit_dialog_side_vb->add_child(edit_add_type_hb);
	edit_add_type_value = memnew(LineEdit);
	edit_add_type_value->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	edit_add_type_hb->add_child(edit_add_type_value);
	Button *edit_add_type_button = memnew(Button);
	edit_add_type_button->set_text(TTR("Add"));
	edit_add_type_hb->add_child(edit_add_type_button);
	edit_add_type_button->connect("pressed", callable_mp(this, &ThemeItemEditorDialog::_add_theme_type));

	VBoxContainer *edit_items_vb = memnew(VBoxContainer);
	edit_items_vb->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	edit_dialog_hs->add_child(edit_items_vb);

	HBoxContainer *edit_items_toolbar = memnew(HBoxContainer);
	edit_items_vb->add_child(edit_items_toolbar);

	Label *edit_items_toolbar_add_label = memnew(Label);
	edit_items_toolbar_add_label->set_text(TTR("Add Item:"));
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
	edit_items_toolbar_remove_label->set_text(TTR("Remove Items:"));
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

	// Import Items tab.
	TabContainer *import_tc = memnew(TabContainer);
	tc->add_child(import_tc);
	tc->set_tab_title(1, TTR("Import Items"));

	import_default_theme_items = memnew(ThemeItemImportTree);
	import_tc->add_child(import_default_theme_items);
	import_tc->set_tab_title(0, TTR("Default Theme"));
	import_default_theme_items->connect("items_imported", callable_mp(this, &ThemeItemEditorDialog::_update_edit_types));

	import_editor_theme_items = memnew(ThemeItemImportTree);
	import_tc->add_child(import_editor_theme_items);
	import_tc->set_tab_title(1, TTR("Editor Theme"));
	import_editor_theme_items->connect("items_imported", callable_mp(this, &ThemeItemEditorDialog::_update_edit_types));

	VBoxContainer *import_another_theme_vb = memnew(VBoxContainer);

	HBoxContainer *import_another_file_hb = memnew(HBoxContainer);
	import_another_theme_vb->add_child(import_another_file_hb);
	import_another_theme_value = memnew(LineEdit);
	import_another_theme_value->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	import_another_theme_value->set_editable(false);
	import_another_file_hb->add_child(import_another_theme_value);
	import_another_theme_button = memnew(Button);
	import_another_file_hb->add_child(import_another_theme_button);
	import_another_theme_button->connect("pressed", callable_mp(this, &ThemeItemEditorDialog::_open_select_another_theme));

	import_another_theme_dialog = memnew(EditorFileDialog);
	import_another_theme_dialog->set_file_mode(EditorFileDialog::FILE_MODE_OPEN_FILE);
	import_another_theme_dialog->set_title(TTR("Select Another Theme Resource:"));
	List<String> ext;
	ResourceLoader::get_recognized_extensions_for_type("Theme", &ext);
	for (List<String>::Element *E = ext.front(); E; E = E->next()) {
		import_another_theme_dialog->add_filter("*." + E->get() + "; Theme Resource");
	}
	import_another_file_hb->add_child(import_another_theme_dialog);
	import_another_theme_dialog->connect("file_selected", callable_mp(this, &ThemeItemEditorDialog::_select_another_theme_cbk));

	import_other_theme_items = memnew(ThemeItemImportTree);
	import_other_theme_items->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	import_another_theme_vb->add_child(import_other_theme_items);

	import_tc->add_child(import_another_theme_vb);
	import_tc->set_tab_title(2, TTR("Another Theme"));
	import_other_theme_items->connect("items_imported", callable_mp(this, &ThemeItemEditorDialog::_update_edit_types));

	confirm_closing_dialog = memnew(ConfirmationDialog);
	confirm_closing_dialog->set_autowrap(true);
	add_child(confirm_closing_dialog);
	confirm_closing_dialog->connect("confirmed", callable_mp(this, &ThemeItemEditorDialog::_close_dialog));
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

void ThemeEditor::_theme_edit_button_cbk() {
	theme_edit_dialog->popup_centered(Size2(850, 760) * EDSCALE);
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

	theme_edit_button = memnew(Button);
	theme_edit_button->set_text(TTR("Manage Items"));
	theme_edit_button->set_tooltip(TTR("Add, remove, organize and import Theme items."));
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
