/*************************************************************************/
/*  replication_editor_plugin.cpp                                        */
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

#include "replication_editor_plugin.h"

#include "editor/editor_node.h"
#include "editor/editor_scale.h"
#include "editor/inspector_dock.h"
#include "scene/gui/dialogs.h"
#include "scene/gui/tree.h"
#include "scene/multiplayer/multiplayer_synchronizer.h"

void ReplicationEditor::_pick_node_filter_text_changed(const String &p_newtext) {
	TreeItem *root_item = pick_node->get_scene_tree()->get_scene_tree()->get_root();

	Vector<Node *> select_candidates;
	Node *to_select = nullptr;

	String filter = pick_node->get_filter_line_edit()->get_text();

	_pick_node_select_recursive(root_item, filter, select_candidates);

	if (!select_candidates.is_empty()) {
		for (int i = 0; i < select_candidates.size(); ++i) {
			Node *candidate = select_candidates[i];

			if (((String)candidate->get_name()).to_lower().begins_with(filter.to_lower())) {
				to_select = candidate;
				break;
			}
		}

		if (!to_select) {
			to_select = select_candidates[0];
		}
	}

	pick_node->get_scene_tree()->set_selected(to_select);
}

void ReplicationEditor::_pick_node_select_recursive(TreeItem *p_item, const String &p_filter, Vector<Node *> &p_select_candidates) {
	if (!p_item) {
		return;
	}

	NodePath np = p_item->get_metadata(0);
	Node *node = get_node(np);

	if (!p_filter.is_empty() && ((String)node->get_name()).findn(p_filter) != -1) {
		p_select_candidates.push_back(node);
	}

	TreeItem *c = p_item->get_first_child();

	while (c) {
		_pick_node_select_recursive(c, p_filter, p_select_candidates);
		c = c->get_next();
	}
}

void ReplicationEditor::_pick_node_filter_input(const Ref<InputEvent> &p_ie) {
	Ref<InputEventKey> k = p_ie;

	if (k.is_valid()) {
		switch (k->get_keycode()) {
			case Key::UP:
			case Key::DOWN:
			case Key::PAGEUP:
			case Key::PAGEDOWN: {
				pick_node->get_scene_tree()->get_scene_tree()->gui_input(k);
				pick_node->get_filter_line_edit()->accept_event();
			} break;
			default:
				break;
		}
	}
}

void ReplicationEditor::_pick_node_selected(NodePath p_path) {
	Node *root = current->get_node(current->get_root_path());
	ERR_FAIL_COND(!root);
	Node *node = get_node(p_path);
	ERR_FAIL_COND(!node);
	NodePath path_to = root->get_path_to(node);
	adding_node_path = path_to;
	prop_selector->select_property_from_instance(node);
}

void ReplicationEditor::_pick_new_property() {
	if (current == nullptr) {
		EditorNode::get_singleton()->show_warning(TTR("Select a replicator node in order to pick a property to add to it."));
		return;
	}
	Node *root = current->get_node(current->get_root_path());
	if (!root) {
		EditorNode::get_singleton()->show_warning(TTR("Not possible to add a new property to synchronize without a root."));
		return;
	}
	pick_node->popup_scenetree_dialog();
	pick_node->get_filter_line_edit()->clear();
	pick_node->get_filter_line_edit()->grab_focus();
}

void ReplicationEditor::_add_sync_property(String p_path) {
	config = current->get_replication_config();

	if (config.is_valid() && config->has_property(p_path)) {
		EditorNode::get_singleton()->show_warning(TTR("Property is already being synchronized."));
		return;
	}

	UndoRedo *undo_redo = EditorNode::get_singleton()->get_undo_redo();
	undo_redo->create_action(TTR("Add property to synchronizer"));

	if (config.is_null()) {
		config.instantiate();
		current->set_replication_config(config);
		undo_redo->add_do_method(current, "set_replication_config", config);
		undo_redo->add_undo_method(current, "set_replication_config", Ref<SceneReplicationConfig>());
		_update_config();
	}

	undo_redo->add_do_method(config.ptr(), "add_property", p_path);
	undo_redo->add_undo_method(config.ptr(), "remove_property", p_path);
	undo_redo->add_do_method(this, "_update_config");
	undo_redo->add_undo_method(this, "_update_config");
	undo_redo->commit_action();
}

void ReplicationEditor::_pick_node_property_selected(String p_name) {
	String adding_prop_path = String(adding_node_path) + ":" + p_name;

	_add_sync_property(adding_prop_path);
}

/// ReplicationEditor
ReplicationEditor::ReplicationEditor() {
	set_v_size_flags(SIZE_EXPAND_FILL);
	set_custom_minimum_size(Size2(0, 200) * EDSCALE);

	delete_dialog = memnew(ConfirmationDialog);
	delete_dialog->connect("cancelled", callable_mp(this, &ReplicationEditor::_dialog_closed), varray(false));
	delete_dialog->connect("confirmed", callable_mp(this, &ReplicationEditor::_dialog_closed), varray(true));
	add_child(delete_dialog);

	error_dialog = memnew(AcceptDialog);
	error_dialog->get_ok_button()->set_text(TTR("Close"));
	error_dialog->set_title(TTR("Error!"));
	add_child(error_dialog);

	VBoxContainer *vb = memnew(VBoxContainer);
	vb->set_v_size_flags(SIZE_EXPAND_FILL);
	add_child(vb);

	pick_node = memnew(SceneTreeDialog);
	add_child(pick_node);
	pick_node->register_text_enter(pick_node->get_filter_line_edit());
	pick_node->set_title(TTR("Pick a node to synchronize:"));
	pick_node->connect("selected", callable_mp(this, &ReplicationEditor::_pick_node_selected));
	pick_node->get_filter_line_edit()->connect("text_changed", callable_mp(this, &ReplicationEditor::_pick_node_filter_text_changed));
	pick_node->get_filter_line_edit()->connect("gui_input", callable_mp(this, &ReplicationEditor::_pick_node_filter_input));

	prop_selector = memnew(PropertySelector);
	add_child(prop_selector);
	prop_selector->connect("selected", callable_mp(this, &ReplicationEditor::_pick_node_property_selected));

	HBoxContainer *hb = memnew(HBoxContainer);
	vb->add_child(hb);

	add_pick_button = memnew(Button);
	add_pick_button->connect("pressed", callable_mp(this, &ReplicationEditor::_pick_new_property));
	add_pick_button->set_text(TTR("Add property to sync.."));
	hb->add_child(add_pick_button);
	VSeparator *vs = memnew(VSeparator);
	vs->set_custom_minimum_size(Size2(30 * EDSCALE, 0));
	hb->add_child(vs);
	hb->add_child(memnew(Label(TTR("Path:"))));
	np_line_edit = memnew(LineEdit);
	np_line_edit->set_placeholder(":property");
	np_line_edit->set_h_size_flags(SIZE_EXPAND_FILL);
	hb->add_child(np_line_edit);
	add_from_path_button = memnew(Button);
	add_from_path_button->connect("pressed", callable_mp(this, &ReplicationEditor::_add_pressed));
	add_from_path_button->set_text(TTR("Add from path"));
	hb->add_child(add_from_path_button);
	vs = memnew(VSeparator);
	vs->set_custom_minimum_size(Size2(30 * EDSCALE, 0));
	hb->add_child(vs);
	pin = memnew(Button);
	pin->set_flat(true);
	pin->set_toggle_mode(true);
	hb->add_child(pin);

	tree = memnew(Tree);
	tree->set_hide_root(true);
	tree->set_columns(4);
	tree->set_column_titles_visible(true);
	tree->set_column_title(0, TTR("Properties"));
	tree->set_column_expand(0, true);
	tree->set_column_title(1, TTR("Spawn"));
	tree->set_column_expand(1, false);
	tree->set_column_custom_minimum_width(1, 100);
	tree->set_column_title(2, TTR("Sync"));
	tree->set_column_custom_minimum_width(2, 100);
	tree->set_column_expand(2, false);
	tree->set_column_expand(3, false);
	tree->create_item();
	tree->connect("button_clicked", callable_mp(this, &ReplicationEditor::_tree_button_pressed));
	tree->connect("item_edited", callable_mp(this, &ReplicationEditor::_tree_item_edited));
	tree->set_v_size_flags(SIZE_EXPAND_FILL);
	vb->add_child(tree);

	drop_label = memnew(Label);
	drop_label->set_text(TTR("Add properties using the buttons above or\ndrag them them from the inspector and drop them here."));
	drop_label->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	drop_label->set_vertical_alignment(VERTICAL_ALIGNMENT_CENTER);
	tree->add_child(drop_label);
	drop_label->set_anchors_and_offsets_preset(Control::PRESET_WIDE);

	tree->set_drag_forwarding(this);
}

void ReplicationEditor::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_update_config"), &ReplicationEditor::_update_config);
	ClassDB::bind_method(D_METHOD("_update_checked", "property", "column", "checked"), &ReplicationEditor::_update_checked);
	ClassDB::bind_method("_can_drop_data_fw", &ReplicationEditor::_can_drop_data_fw);
	ClassDB::bind_method("_drop_data_fw", &ReplicationEditor::_drop_data_fw);

	ADD_SIGNAL(MethodInfo("keying_changed"));
}

bool ReplicationEditor::_can_drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) const {
	Dictionary d = p_data;
	if (!d.has("type")) {
		return false;
	}
	String t = d["type"];
	if (t != "obj_property") {
		return false;
	}
	Object *obj = d["object"];
	if (!obj) {
		return false;
	}
	Node *node = Object::cast_to<Node>(obj);
	if (!node) {
		return false;
	}

	return true;
}

void ReplicationEditor::_drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) {
	if (current == nullptr) {
		EditorNode::get_singleton()->show_warning(TTR("Select a replicator node in order to pick a property to add to it."));
		return;
	}
	Node *root = current->get_node(current->get_root_path());
	if (!root) {
		EditorNode::get_singleton()->show_warning(TTR("Not possible to add a new property to synchronize without a root."));
		return;
	}

	Dictionary d = p_data;
	if (!d.has("type")) {
		return;
	}
	String t = d["type"];
	if (t != "obj_property") {
		return;
	}
	Object *obj = d["object"];
	if (!obj) {
		return;
	}
	Node *node = Object::cast_to<Node>(obj);
	if (!node) {
		return;
	}

	String path = root->get_path_to(node);
	path += ":" + String(d["property"]);

	_add_sync_property(path);
}

void ReplicationEditor::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE:
		case EditorSettings::NOTIFICATION_EDITOR_SETTINGS_CHANGED: {
			add_theme_style_override("panel", EditorNode::get_singleton()->get_gui_base()->get_theme_stylebox(SNAME("panel"), SNAME("Panel")));
			add_pick_button->set_icon(get_theme_icon(SNAME("Add"), SNAME("EditorIcons")));
			pin->set_icon(get_theme_icon(SNAME("Pin"), SNAME("EditorIcons")));
		} break;

		case NOTIFICATION_VISIBILITY_CHANGED: {
			update_keying();
		} break;
	}
}

void ReplicationEditor::_add_pressed() {
	if (!current) {
		error_dialog->set_text(TTR("Please select a MultiplayerSynchronizer first."));
		error_dialog->popup_centered();
		return;
	}
	if (current->get_root_path().is_empty()) {
		error_dialog->set_text(TTR("The MultiplayerSynchronizer needs a root path."));
		error_dialog->popup_centered();
		return;
	}
	String np_text = np_line_edit->get_text();
	if (np_text.find(":") == -1) {
		np_text = ":" + np_text;
	}
	NodePath prop = NodePath(np_text);
	if (prop.is_empty()) {
		return;
	}
	UndoRedo *undo_redo = EditorNode::get_singleton()->get_undo_redo();
	undo_redo->create_action(TTR("Add property"));
	config = current->get_replication_config();
	if (config.is_null()) {
		config.instantiate();
		current->set_replication_config(config);
		undo_redo->add_do_method(current, "set_replication_config", config);
		undo_redo->add_undo_method(current, "set_replication_config", Ref<SceneReplicationConfig>());
		_update_config();
	}
	undo_redo->add_do_method(config.ptr(), "add_property", prop);
	undo_redo->add_undo_method(config.ptr(), "remove_property", prop);
	undo_redo->add_do_method(this, "_update_config");
	undo_redo->add_undo_method(this, "_update_config");
	undo_redo->commit_action();
}

void ReplicationEditor::_tree_item_edited() {
	TreeItem *ti = tree->get_edited();
	if (!ti || config.is_null()) {
		return;
	}
	int column = tree->get_edited_column();
	ERR_FAIL_COND(column < 1 || column > 2);
	const NodePath prop = ti->get_metadata(0);
	UndoRedo *undo_redo = EditorNode::get_singleton()->get_undo_redo();
	bool value = ti->is_checked(column);
	String method;
	if (column == 1) {
		undo_redo->create_action(TTR("Set spawn property"));
		method = "property_set_spawn";
	} else {
		undo_redo->create_action(TTR("Set sync property"));
		method = "property_set_sync";
	}
	undo_redo->add_do_method(config.ptr(), method, prop, value);
	undo_redo->add_undo_method(config.ptr(), method, prop, !value);
	undo_redo->add_do_method(this, "_update_checked", prop, column, value);
	undo_redo->add_undo_method(this, "_update_checked", prop, column, !value);
	undo_redo->commit_action();
}

void ReplicationEditor::_tree_button_pressed(Object *p_item, int p_column, int p_id, MouseButton p_button) {
	if (p_button != MouseButton::LEFT) {
		return;
	}

	TreeItem *ti = Object::cast_to<TreeItem>(p_item);
	if (!ti) {
		return;
	}
	deleting = ti->get_metadata(0);
	delete_dialog->set_text(TTR("Delete Property?") + "\n\"" + ti->get_text(0) + "\"");
	delete_dialog->popup_centered();
}

void ReplicationEditor::_dialog_closed(bool p_confirmed) {
	if (deleting.is_empty() || config.is_null()) {
		return;
	}
	if (p_confirmed) {
		const NodePath prop = deleting;
		int idx = config->property_get_index(prop);
		bool spawn = config->property_get_spawn(prop);
		bool sync = config->property_get_sync(prop);
		UndoRedo *undo_redo = EditorNode::get_singleton()->get_undo_redo();
		undo_redo->create_action(TTR("Remove Property"));
		undo_redo->add_do_method(config.ptr(), "remove_property", prop);
		undo_redo->add_undo_method(config.ptr(), "add_property", prop, idx);
		undo_redo->add_undo_method(config.ptr(), "property_set_spawn", prop, spawn);
		undo_redo->add_undo_method(config.ptr(), "property_set_sync", prop, sync);
		undo_redo->add_do_method(this, "_update_config");
		undo_redo->add_undo_method(this, "_update_config");
		undo_redo->commit_action();
	}
	deleting = NodePath();
}

void ReplicationEditor::_update_checked(const NodePath &p_prop, int p_column, bool p_checked) {
	if (!tree->get_root()) {
		return;
	}
	TreeItem *ti = tree->get_root()->get_first_child();
	while (ti) {
		if (ti->get_metadata(0).operator NodePath() == p_prop) {
			ti->set_checked(p_column, p_checked);
			return;
		}
		ti = ti->get_next();
	}
}

void ReplicationEditor::update_keying() {
	/// TODO make keying usable.
#if 0
	bool keying_enabled = false;
	EditorSelectionHistory *editor_history = EditorNode::get_singleton()->get_editor_selection_history();
	if (is_visible_in_tree() && config.is_valid() && editor_history->get_path_size() > 0) {
		Object *obj = ObjectDB::get_instance(editor_history->get_path_object(0));
		keying_enabled = Object::cast_to<Node>(obj) != nullptr;
	}

	if (keying_enabled == keying) {
		return;
	}

	keying = keying_enabled;
	emit_signal(SNAME("keying_changed"));
#endif
}

void ReplicationEditor::_update_config() {
	deleting = NodePath();
	tree->clear();
	tree->create_item();
	drop_label->set_visible(true);
	if (!config.is_valid()) {
		update_keying();
		return;
	}
	TypedArray<NodePath> props = config->get_properties();
	if (props.size()) {
		drop_label->set_visible(false);
	}
	for (int i = 0; i < props.size(); i++) {
		const NodePath path = props[i];
		_add_property(path, config->property_get_spawn(path), config->property_get_sync(path));
	}
	update_keying();
}

void ReplicationEditor::edit(MultiplayerSynchronizer *p_sync) {
	if (current == p_sync) {
		return;
	}
	current = p_sync;
	if (current) {
		config = current->get_replication_config();
	} else {
		config.unref();
	}
	_update_config();
}

Ref<Texture2D> ReplicationEditor::_get_class_icon(const Node *p_node) {
	if (!p_node || !has_theme_icon(p_node->get_class(), "EditorIcons")) {
		return get_theme_icon(SNAME("ImportFail"), SNAME("EditorIcons"));
	}
	return get_theme_icon(p_node->get_class(), "EditorIcons");
}

void ReplicationEditor::_add_property(const NodePath &p_property, bool p_spawn, bool p_sync) {
	String prop = String(p_property);
	TreeItem *item = tree->create_item();
	item->set_selectable(0, false);
	item->set_selectable(1, false);
	item->set_selectable(2, false);
	item->set_selectable(3, false);
	item->set_text(0, prop);
	item->set_metadata(0, prop);
	Node *root_node = current && !current->get_root_path().is_empty() ? current->get_node(current->get_root_path()) : nullptr;
	Ref<Texture2D> icon = _get_class_icon(root_node);
	if (root_node) {
		String path = prop.substr(0, prop.find(":"));
		String subpath = prop.substr(path.size());
		Node *node = root_node->get_node_or_null(path);
		if (!node) {
			node = root_node;
		}
		item->set_text(0, String(node->get_name()) + ":" + subpath);
		icon = _get_class_icon(node);
	}
	item->set_icon(0, icon);
	item->add_button(3, get_theme_icon(SNAME("Remove"), SNAME("EditorIcons")));
	item->set_text_alignment(1, HORIZONTAL_ALIGNMENT_CENTER);
	item->set_cell_mode(1, TreeItem::CELL_MODE_CHECK);
	item->set_checked(1, p_spawn);
	item->set_editable(1, true);
	item->set_text_alignment(2, HORIZONTAL_ALIGNMENT_CENTER);
	item->set_cell_mode(2, TreeItem::CELL_MODE_CHECK);
	item->set_checked(2, p_sync);
	item->set_editable(2, true);
}

void ReplicationEditor::property_keyed(const String &p_property) {
	ERR_FAIL_COND(!current || config.is_null());
	Node *root = current->get_node(current->get_root_path());
	ERR_FAIL_COND(!root);
	EditorSelectionHistory *history = EditorNode::get_singleton()->get_editor_selection_history();
	ERR_FAIL_COND(history->get_path_size() == 0);
	Node *node = Object::cast_to<Node>(ObjectDB::get_instance(history->get_path_object(0)));
	ERR_FAIL_COND(!node);
	if (node->is_class("MultiplayerSynchronizer")) {
		error_dialog->set_text(TTR("Properties of 'MultiplayerSynchronizer' cannot be configured for replication."));
		error_dialog->popup_centered();
		return;
	}
	if (history->get_path_size() > 1 || p_property.get_slice_count(":") > 1) {
		error_dialog->set_text(TTR("Subresources cannot yet be configured for replication."));
		error_dialog->popup_centered();
		return;
	}

	String path = root->get_path_to(node);
	for (int i = 1; i < history->get_path_size(); i++) {
		String prop = history->get_path_property(i);
		ERR_FAIL_COND(prop == "");
		path += ":" + prop;
	}
	path += ":" + p_property;

	NodePath prop = path;
	UndoRedo *undo_redo = EditorNode::get_singleton()->get_undo_redo();
	undo_redo->create_action(TTR("Add property"));
	undo_redo->add_do_method(config.ptr(), "add_property", prop);
	undo_redo->add_undo_method(config.ptr(), "remove_property", prop);
	undo_redo->add_do_method(this, "_update_config");
	undo_redo->add_undo_method(this, "_update_config");
	undo_redo->commit_action();
}

/// ReplicationEditorPlugin
ReplicationEditorPlugin::ReplicationEditorPlugin() {
	repl_editor = memnew(ReplicationEditor);
	button = EditorNode::get_singleton()->add_bottom_panel_item(TTR("Replication"), repl_editor);
	button->hide();
	repl_editor->get_pin()->connect("pressed", callable_mp(this, &ReplicationEditorPlugin::_pinned));
}

ReplicationEditorPlugin::~ReplicationEditorPlugin() {
}

void ReplicationEditorPlugin::_keying_changed() {
	// TODO make lock usable.
	//InspectorDock::get_inspector_singleton()->set_keying(repl_editor->has_keying(), this);
}

void ReplicationEditorPlugin::_property_keyed(const String &p_keyed, const Variant &p_value, bool p_advance) {
	if (!repl_editor->has_keying()) {
		return;
	}
	repl_editor->property_keyed(p_keyed);
}

void ReplicationEditorPlugin::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			//Node3DEditor::get_singleton()->connect("transform_key_request", callable_mp(this, &AnimationPlayerEditorPlugin::_transform_key_request));
			InspectorDock::get_inspector_singleton()->connect("property_keyed", callable_mp(this, &ReplicationEditorPlugin::_property_keyed));
			repl_editor->connect("keying_changed", callable_mp(this, &ReplicationEditorPlugin::_keying_changed));
			// TODO make lock usable.
			//InspectorDock::get_inspector_singleton()->connect("object_inspected", callable_mp(repl_editor, &ReplicationEditor::update_keying));
			get_tree()->connect("node_removed", callable_mp(this, &ReplicationEditorPlugin::_node_removed));
		} break;
	}
}

void ReplicationEditorPlugin::_node_removed(Node *p_node) {
	if (p_node && p_node == repl_editor->get_current()) {
		repl_editor->edit(nullptr);
		if (repl_editor->is_visible_in_tree()) {
			EditorNode::get_singleton()->hide_bottom_panel();
		}
		button->hide();
		repl_editor->get_pin()->set_pressed(false);
	}
}

void ReplicationEditorPlugin::_pinned() {
	if (!repl_editor->get_pin()->is_pressed()) {
		if (repl_editor->is_visible_in_tree()) {
			EditorNode::get_singleton()->hide_bottom_panel();
		}
		button->hide();
	}
}

void ReplicationEditorPlugin::edit(Object *p_object) {
	repl_editor->edit(Object::cast_to<MultiplayerSynchronizer>(p_object));
}

bool ReplicationEditorPlugin::handles(Object *p_object) const {
	return p_object->is_class("MultiplayerSynchronizer");
}

void ReplicationEditorPlugin::make_visible(bool p_visible) {
	if (p_visible) {
		//editor->hide_animation_player_editors();
		//editor->animation_panel_make_visible(true);
		button->show();
		EditorNode::get_singleton()->make_bottom_panel_item_visible(repl_editor);
	} else if (!repl_editor->get_pin()->is_pressed()) {
		if (repl_editor->is_visible_in_tree()) {
			EditorNode::get_singleton()->hide_bottom_panel();
		}
		button->hide();
	}
}
