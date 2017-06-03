/*************************************************************************/
/*  editor_autoload_settings.cpp                                         */
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
#include "editor_autoload_settings.h"

#include "editor_node.h"
#include "global_config.h"
#include "global_constants.h"

#define PREVIEW_LIST_MAX_SIZE 10

void EditorAutoloadSettings::_notification(int p_what) {

	if (p_what == NOTIFICATION_ENTER_TREE) {

		List<String> afn;
		ResourceLoader::get_recognized_extensions_for_type("Script", &afn);
		ResourceLoader::get_recognized_extensions_for_type("PackedScene", &afn);

		EditorFileDialog *file_dialog = autoload_add_path->get_file_dialog();

		for (List<String>::Element *E = afn.front(); E; E = E->next()) {

			file_dialog->add_filter("*." + E->get());
		}
	}
}

bool EditorAutoloadSettings::_autoload_name_is_valid(const String &p_name, String *r_error) {

	if (!p_name.is_valid_identifier()) {
		if (r_error)
			*r_error = TTR("Invalid name.") + "\n" + TTR("Valid characters:") + " a-z, A-Z, 0-9 or _";

		return false;
	}

	if (ClassDB::class_exists(p_name)) {
		if (r_error)
			*r_error = TTR("Invalid name. Must not collide with an existing engine class name.");

		return false;
	}

	for (int i = 0; i < Variant::VARIANT_MAX; i++) {
		if (Variant::get_type_name(Variant::Type(i)) == p_name) {
			if (r_error)
				*r_error = TTR("Invalid name. Must not collide with an existing buit-in type name.");

			return false;
		}
	}

	for (int i = 0; i < GlobalConstants::get_global_constant_count(); i++) {
		if (GlobalConstants::get_global_constant_name(i) == p_name) {
			if (r_error)
				*r_error = TTR("Invalid name. Must not collide with an existing global constant name.");

			return false;
		}
	}

	return true;
}

void EditorAutoloadSettings::_autoload_add() {

	String name = autoload_add_name->get_text();

	String error;
	if (!_autoload_name_is_valid(name, &error)) {
		EditorNode::get_singleton()->show_warning(error);
		return;
	}

	String path = autoload_add_path->get_line_edit()->get_text();
	if (!FileAccess::exists(path)) {
		EditorNode::get_singleton()->show_warning(TTR("Invalid Path.") + "\n" + TTR("File does not exist."));
		return;
	}

	if (!path.begins_with("res://")) {
		EditorNode::get_singleton()->show_warning(TTR("Invalid Path.") + "\n" + TTR("Not in resource path."));
		return;
	}

	name = "autoload/" + name;

	UndoRedo *undo_redo = EditorNode::get_singleton()->get_undo_redo();

	undo_redo->create_action(TTR("Add AutoLoad"));
	undo_redo->add_do_property(GlobalConfig::get_singleton(), name, "*" + path);

	if (GlobalConfig::get_singleton()->has(name)) {
		undo_redo->add_undo_property(GlobalConfig::get_singleton(), name, GlobalConfig::get_singleton()->get(name));
	} else {
		undo_redo->add_undo_property(GlobalConfig::get_singleton(), name, Variant());
	}

	undo_redo->add_do_method(this, "update_autoload");
	undo_redo->add_undo_method(this, "update_autoload");

	undo_redo->add_do_method(this, "emit_signal", autoload_changed);
	undo_redo->add_undo_method(this, "emit_signal", autoload_changed);

	undo_redo->commit_action();

	autoload_add_path->get_line_edit()->set_text("");
	autoload_add_name->set_text("");
}

void EditorAutoloadSettings::_autoload_selected() {

	TreeItem *ti = tree->get_selected();

	if (!ti)
		return;

	selected_autoload = "autoload/" + ti->get_text(0);
}

void EditorAutoloadSettings::_autoload_edited() {

	if (updating_autoload)
		return;

	TreeItem *ti = tree->get_edited();
	int column = tree->get_edited_column();

	UndoRedo *undo_redo = EditorNode::get_undo_redo();

	if (column == 0) {
		String name = ti->get_text(0);
		String old_name = selected_autoload.get_slice("/", 1);

		if (name == old_name)
			return;

		String error;
		if (!_autoload_name_is_valid(name, &error)) {
			ti->set_text(0, old_name);
			EditorNode::get_singleton()->show_warning(error);
			return;
		}

		if (GlobalConfig::get_singleton()->has("autoload/" + name)) {
			ti->set_text(0, old_name);
			EditorNode::get_singleton()->show_warning(vformat(TTR("Autoload '%s' already exists!"), name));
			return;
		}

		updating_autoload = true;

		name = "autoload/" + name;

		int order = GlobalConfig::get_singleton()->get_order(selected_autoload);
		String path = GlobalConfig::get_singleton()->get(selected_autoload);

		undo_redo->create_action(TTR("Rename Autoload"));

		undo_redo->add_do_property(GlobalConfig::get_singleton(), name, path);
		undo_redo->add_do_method(GlobalConfig::get_singleton(), "set_order", name, order);
		undo_redo->add_do_method(GlobalConfig::get_singleton(), "clear", selected_autoload);

		undo_redo->add_undo_property(GlobalConfig::get_singleton(), selected_autoload, path);
		undo_redo->add_undo_method(GlobalConfig::get_singleton(), "set_order", selected_autoload, order);
		undo_redo->add_undo_method(GlobalConfig::get_singleton(), "clear", name);

		undo_redo->add_do_method(this, "update_autoload");
		undo_redo->add_undo_method(this, "update_autoload");

		undo_redo->add_do_method(this, "emit_signal", autoload_changed);
		undo_redo->add_undo_method(this, "emit_signal", autoload_changed);

		undo_redo->commit_action();

		selected_autoload = name;
	} else if (column == 2) {
		updating_autoload = true;

		bool checked = ti->is_checked(2);
		String base = "autoload/" + ti->get_text(0);

		int order = GlobalConfig::get_singleton()->get_order(base);
		String path = GlobalConfig::get_singleton()->get(base);

		if (path.begins_with("*"))
			path = path.substr(1, path.length());

		if (checked)
			path = "*" + path;

		undo_redo->create_action(TTR("Toggle AutoLoad Globals"));

		undo_redo->add_do_property(GlobalConfig::get_singleton(), base, path);
		undo_redo->add_undo_property(GlobalConfig::get_singleton(), base, GlobalConfig::get_singleton()->get(base));

		undo_redo->add_do_method(GlobalConfig::get_singleton(), "set_order", base, order);
		undo_redo->add_undo_method(GlobalConfig::get_singleton(), "set_order", base, order);

		undo_redo->add_do_method(this, "update_autoload");
		undo_redo->add_undo_method(this, "update_autoload");

		undo_redo->add_do_method(this, "emit_signal", autoload_changed);
		undo_redo->add_undo_method(this, "emit_signal", autoload_changed);

		undo_redo->commit_action();
	}

	updating_autoload = false;
}

void EditorAutoloadSettings::_autoload_button_pressed(Object *p_item, int p_column, int p_button) {

	TreeItem *ti = p_item->cast_to<TreeItem>();

	String name = "autoload/" + ti->get_text(0);

	UndoRedo *undo_redo = EditorNode::get_undo_redo();

	switch (p_button) {

		case BUTTON_MOVE_UP:
		case BUTTON_MOVE_DOWN: {

			TreeItem *swap = NULL;

			if (p_button == BUTTON_MOVE_UP) {
				swap = ti->get_prev();
			} else {
				swap = ti->get_next();
			}

			if (!swap)
				return;

			String swap_name = "autoload/" + swap->get_text(0);

			int order = GlobalConfig::get_singleton()->get_order(name);
			int swap_order = GlobalConfig::get_singleton()->get_order(swap_name);

			undo_redo->create_action(TTR("Move Autoload"));

			undo_redo->add_do_method(GlobalConfig::get_singleton(), "set_order", name, swap_order);
			undo_redo->add_undo_method(GlobalConfig::get_singleton(), "set_order", name, order);

			undo_redo->add_do_method(GlobalConfig::get_singleton(), "set_order", swap_name, order);
			undo_redo->add_undo_method(GlobalConfig::get_singleton(), "set_order", swap_name, swap_order);

			undo_redo->add_do_method(this, "update_autoload");
			undo_redo->add_undo_method(this, "update_autoload");

			undo_redo->add_do_method(this, "emit_signal", autoload_changed);
			undo_redo->add_undo_method(this, "emit_signal", autoload_changed);

			undo_redo->commit_action();
		} break;
		case BUTTON_DELETE: {

			int order = GlobalConfig::get_singleton()->get_order(name);

			undo_redo->create_action(TTR("Remove Autoload"));

			undo_redo->add_do_property(GlobalConfig::get_singleton(), name, Variant());

			undo_redo->add_undo_property(GlobalConfig::get_singleton(), name, GlobalConfig::get_singleton()->get(name));
			undo_redo->add_undo_method(GlobalConfig::get_singleton(), "set_persisting", name, true);
			undo_redo->add_undo_method(GlobalConfig::get_singleton(), "set_order", order);

			undo_redo->add_do_method(this, "update_autoload");
			undo_redo->add_undo_method(this, "update_autoload");

			undo_redo->add_do_method(this, "emit_signal", autoload_changed);
			undo_redo->add_undo_method(this, "emit_signal", autoload_changed);

			undo_redo->commit_action();
		} break;
	}
}

void EditorAutoloadSettings::_autoload_file_callback(const String &p_path) {

	autoload_add_name->set_text(p_path.get_file().get_basename());
}

void EditorAutoloadSettings::update_autoload() {

	if (updating_autoload)
		return;

	updating_autoload = true;

	autoload_cache.clear();

	tree->clear();
	TreeItem *root = tree->create_item();

	List<PropertyInfo> props;
	GlobalConfig::get_singleton()->get_property_list(&props);

	for (List<PropertyInfo>::Element *E = props.front(); E; E = E->next()) {

		const PropertyInfo &pi = E->get();

		if (!pi.name.begins_with("autoload/"))
			continue;

		String name = pi.name.get_slice("/", 1);
		String path = GlobalConfig::get_singleton()->get(pi.name);

		if (name.empty())
			continue;

		AutoLoadInfo info;
		info.name = pi.name;
		info.order = GlobalConfig::get_singleton()->get_order(pi.name);

		autoload_cache.push_back(info);

		bool global = false;

		if (path.begins_with("*")) {
			global = true;
			path = path.substr(1, path.length());
		}

		TreeItem *item = tree->create_item(root);
		item->set_text(0, name);
		item->set_editable(0, true);

		item->set_text(1, path);
		item->set_selectable(1, false);

		item->set_cell_mode(2, TreeItem::CELL_MODE_CHECK);
		item->set_editable(2, true);
		item->set_text(2, TTR("Enable"));
		item->set_checked(2, global);

		item->add_button(3, get_icon("MoveUp", "EditorIcons"), BUTTON_MOVE_UP);
		item->add_button(3, get_icon("MoveDown", "EditorIcons"), BUTTON_MOVE_DOWN);
		item->add_button(3, get_icon("Del", "EditorIcons"), BUTTON_DELETE);
		item->set_selectable(3, false);
	}

	updating_autoload = false;
}

Variant EditorAutoloadSettings::get_drag_data_fw(const Point2 &p_point, Control *p_control) {

	if (autoload_cache.size() <= 1)
		return false;

	PoolStringArray autoloads;

	TreeItem *next = tree->get_next_selected(NULL);

	while (next) {
		autoloads.push_back(next->get_text(0));
		next = tree->get_next_selected(next);
	}

	if (autoloads.size() == 0 || autoloads.size() == autoload_cache.size())
		return Variant();

	VBoxContainer *preview = memnew(VBoxContainer);

	int max_size = MIN(PREVIEW_LIST_MAX_SIZE, autoloads.size());

	for (int i = 0; i < max_size; i++) {
		Label *label = memnew(Label(autoloads[i]));
		label->set_self_modulate(Color(1, 1, 1, Math::lerp(1, 0, float(i) / PREVIEW_LIST_MAX_SIZE)));

		preview->add_child(label);
	}

	tree->set_drop_mode_flags(Tree::DROP_MODE_INBETWEEN);
	tree->set_drag_preview(preview);

	Dictionary drop_data;
	drop_data["type"] = "autoload";
	drop_data["autoloads"] = autoloads;

	return drop_data;
}

bool EditorAutoloadSettings::can_drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_control) const {
	if (updating_autoload)
		return false;

	Dictionary drop_data = p_data;

	if (!drop_data.has("type"))
		return false;

	if (drop_data.has("type")) {
		TreeItem *ti = tree->get_item_at_pos(p_point);

		if (!ti)
			return false;

		int section = tree->get_drop_section_at_pos(p_point);

		if (section < -1)
			return false;

		return true;
	}

	return false;
}

void EditorAutoloadSettings::drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_control) {

	TreeItem *ti = tree->get_item_at_pos(p_point);

	if (!ti)
		return;

	int section = tree->get_drop_section_at_pos(p_point);

	if (section < -1)
		return;

	String name;
	bool move_to_back = false;

	if (section < 0) {
		name = ti->get_text(0);
	} else if (ti->get_next()) {
		name = ti->get_next()->get_text(0);
	} else {
		name = ti->get_text(0);
		move_to_back = true;
	}

	int order = GlobalConfig::get_singleton()->get_order("autoload/" + name);

	AutoLoadInfo aux;
	List<AutoLoadInfo>::Element *E = NULL;

	if (!move_to_back) {
		aux.order = order;
		E = autoload_cache.find(aux);
	}

	Dictionary drop_data = p_data;
	PoolStringArray autoloads = drop_data["autoloads"];

	Vector<int> orders;
	orders.resize(autoload_cache.size());

	for (int i = 0; i < autoloads.size(); i++) {
		aux.order = GlobalConfig::get_singleton()->get_order("autoload/" + autoloads[i]);

		List<AutoLoadInfo>::Element *I = autoload_cache.find(aux);

		if (move_to_back) {
			autoload_cache.move_to_back(I);
		} else if (E != I) {
			autoload_cache.move_before(I, E);
		} else if (E->next()) {
			E = E->next();
		} else {
			break;
		}
	}

	int i = 0;

	for (List<AutoLoadInfo>::Element *E = autoload_cache.front(); E; E = E->next()) {
		orders[i++] = E->get().order;
	}

	orders.sort();

	UndoRedo *undo_redo = EditorNode::get_undo_redo();

	undo_redo->create_action(TTR("Rearrange Autoloads"));

	i = 0;

	for (List<AutoLoadInfo>::Element *E = autoload_cache.front(); E; E = E->next()) {
		undo_redo->add_do_method(GlobalConfig::get_singleton(), "set_order", E->get().name, orders[i++]);
		undo_redo->add_undo_method(GlobalConfig::get_singleton(), "set_order", E->get().name, E->get().order);
	}

	orders.clear();

	undo_redo->add_do_method(this, "update_autoload");
	undo_redo->add_undo_method(this, "update_autoload");

	undo_redo->add_do_method(this, "emit_signal", autoload_changed);
	undo_redo->add_undo_method(this, "emit_signal", autoload_changed);

	undo_redo->commit_action();
}

void EditorAutoloadSettings::_bind_methods() {

	ClassDB::bind_method("_autoload_add", &EditorAutoloadSettings::_autoload_add);
	ClassDB::bind_method("_autoload_selected", &EditorAutoloadSettings::_autoload_selected);
	ClassDB::bind_method("_autoload_edited", &EditorAutoloadSettings::_autoload_edited);
	ClassDB::bind_method("_autoload_button_pressed", &EditorAutoloadSettings::_autoload_button_pressed);
	ClassDB::bind_method("_autoload_file_callback", &EditorAutoloadSettings::_autoload_file_callback);

	ClassDB::bind_method("get_drag_data_fw", &EditorAutoloadSettings::get_drag_data_fw);
	ClassDB::bind_method("can_drop_data_fw", &EditorAutoloadSettings::can_drop_data_fw);
	ClassDB::bind_method("drop_data_fw", &EditorAutoloadSettings::drop_data_fw);

	ClassDB::bind_method("update_autoload", &EditorAutoloadSettings::update_autoload);

	ADD_SIGNAL(MethodInfo("autoload_changed"));
}

EditorAutoloadSettings::EditorAutoloadSettings() {

	autoload_changed = "autoload_changed";

	updating_autoload = false;
	selected_autoload = "";

	HBoxContainer *hbc = memnew(HBoxContainer);
	add_child(hbc);

	VBoxContainer *vbc_path = memnew(VBoxContainer);
	vbc_path->set_h_size_flags(SIZE_EXPAND_FILL);

	autoload_add_path = memnew(EditorLineEditFileChooser);
	autoload_add_path->set_h_size_flags(SIZE_EXPAND_FILL);

	autoload_add_path->get_file_dialog()->set_mode(EditorFileDialog::MODE_OPEN_FILE);
	autoload_add_path->get_file_dialog()->connect("file_selected", this, "_autoload_file_callback");

	vbc_path->add_margin_child(TTR("Path:"), autoload_add_path);
	hbc->add_child(vbc_path);

	VBoxContainer *vbc_name = memnew(VBoxContainer);
	vbc_name->set_h_size_flags(SIZE_EXPAND_FILL);

	HBoxContainer *hbc_name = memnew(HBoxContainer);

	autoload_add_name = memnew(LineEdit);
	autoload_add_name->set_h_size_flags(SIZE_EXPAND_FILL);
	hbc_name->add_child(autoload_add_name);

	Button *add_autoload = memnew(Button);
	add_autoload->set_text(TTR("Add"));
	hbc_name->add_child(add_autoload);
	add_autoload->connect("pressed", this, "_autoload_add");

	vbc_name->add_margin_child(TTR("Node Name:"), hbc_name);
	hbc->add_child(vbc_name);

	tree = memnew(Tree);
	tree->set_hide_root(true);
	tree->set_select_mode(Tree::SELECT_MULTI);
	tree->set_single_select_cell_editing_only_when_already_selected(true);

	tree->set_drag_forwarding(this);

	tree->set_columns(4);
	tree->set_column_titles_visible(true);

	tree->set_column_title(0, TTR("Name"));
	tree->set_column_expand(0, true);
	tree->set_column_min_width(0, 100);

	tree->set_column_title(1, TTR("Path"));
	tree->set_column_expand(1, true);
	tree->set_column_min_width(1, 100);

	tree->set_column_title(2, TTR("Singleton"));
	tree->set_column_expand(2, false);
	tree->set_column_min_width(2, 80);

	tree->set_column_expand(3, false);
	tree->set_column_min_width(3, 80);

	tree->connect("cell_selected", this, "_autoload_selected");
	tree->connect("item_edited", this, "_autoload_edited");
	tree->connect("button_pressed", this, "_autoload_button_pressed");

	add_margin_child(TTR("List:"), tree, true);
}
