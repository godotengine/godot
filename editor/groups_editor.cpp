/*************************************************************************/
/*  groups_editor.cpp                                                    */
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

#include "groups_editor.h"

#include "editor/scene_tree_editor.h"
#include "editor_node.h"
#include "editor_scale.h"
#include "scene/gui/box_container.h"
#include "scene/gui/label.h"
#include "scene/resources/packed_scene.h"

void GroupDialog::_group_selected() {
	nodes_to_add->clear();
	add_node_root = nodes_to_add->create_item();

	nodes_to_remove->clear();
	remove_node_root = nodes_to_remove->create_item();

	if (!groups->is_anything_selected()) {
		group_empty->hide();
		return;
	}

	selected_group = groups->get_selected()->get_text(0);
	_load_nodes(scene_tree->get_edited_scene_root());

	group_empty->set_visible(!remove_node_root->get_first_child());
}

void GroupDialog::_load_nodes(Node *p_current) {
	String item_name = p_current->get_name();
	if (p_current != scene_tree->get_edited_scene_root()) {
		item_name = String(p_current->get_parent()->get_name()) + "/" + item_name;
	}

	bool keep = true;
	Node *root = scene_tree->get_edited_scene_root();
	Node *owner = p_current->get_owner();
	if (owner != root && p_current != root && !owner && !root->is_editable_instance(owner)) {
		keep = false;
	}

	TreeItem *node = nullptr;
	NodePath path = scene_tree->get_edited_scene_root()->get_path_to(p_current);
	if (keep && p_current->is_in_group(selected_group)) {
		if (remove_filter->get_text().is_subsequence_ofi(String(p_current->get_name()))) {
			node = nodes_to_remove->create_item(remove_node_root);
			keep = true;
		} else {
			keep = false;
		}
	} else if (keep && add_filter->get_text().is_subsequence_ofi(String(p_current->get_name()))) {
		node = nodes_to_add->create_item(add_node_root);
		keep = true;
	} else {
		keep = false;
	}

	if (keep) {
		node->set_text(0, item_name);
		node->set_metadata(0, path);
		node->set_tooltip(0, path);

		Ref<Texture2D> icon = EditorNode::get_singleton()->get_object_icon(p_current, "Node");
		node->set_icon(0, icon);

		if (!_can_edit(p_current, selected_group)) {
			node->set_selectable(0, false);
			node->set_custom_color(0, groups->get_theme_color(SNAME("disabled_font_color"), SNAME("Editor")));
		}
	}

	for (int i = 0; i < p_current->get_child_count(); i++) {
		_load_nodes(p_current->get_child(i));
	}
}

bool GroupDialog::_can_edit(Node *p_node, String p_group) {
	Node *n = p_node;
	bool can_edit = true;
	while (n) {
		Ref<SceneState> ss = (n == EditorNode::get_singleton()->get_edited_scene()) ? n->get_scene_inherited_state() : n->get_scene_instance_state();
		if (ss.is_valid()) {
			int path = ss->find_node_by_path(n->get_path_to(p_node));
			if (path != -1) {
				if (ss->is_node_in_group(path, p_group)) {
					can_edit = false;
				}
			}
		}
		n = n->get_owner();
	}
	return can_edit;
}

void GroupDialog::_add_pressed() {
	TreeItem *selected = nodes_to_add->get_next_selected(nullptr);

	if (!selected) {
		return;
	}

	undo_redo->create_action(TTR("Add to Group"));

	while (selected) {
		Node *node = scene_tree->get_edited_scene_root()->get_node(selected->get_metadata(0));
		undo_redo->add_do_method(node, "add_to_group", selected_group, true);
		undo_redo->add_undo_method(node, "remove_from_group", selected_group);

		selected = nodes_to_add->get_next_selected(selected);
	}

	undo_redo->add_do_method(this, "_group_selected");
	undo_redo->add_undo_method(this, "_group_selected");
	undo_redo->add_do_method(this, "emit_signal", "group_edited");
	undo_redo->add_undo_method(this, "emit_signal", "group_edited");

	// To force redraw of scene tree.
	undo_redo->add_do_method(SceneTreeDock::get_singleton()->get_tree_editor(), "update_tree");
	undo_redo->add_undo_method(SceneTreeDock::get_singleton()->get_tree_editor(), "update_tree");

	undo_redo->commit_action();
}

void GroupDialog::_removed_pressed() {
	TreeItem *selected = nodes_to_remove->get_next_selected(nullptr);

	if (!selected) {
		return;
	}

	undo_redo->create_action(TTR("Remove from Group"));

	while (selected) {
		Node *node = scene_tree->get_edited_scene_root()->get_node(selected->get_metadata(0));
		undo_redo->add_do_method(node, "remove_from_group", selected_group);
		undo_redo->add_undo_method(node, "add_to_group", selected_group, true);

		selected = nodes_to_add->get_next_selected(selected);
	}

	undo_redo->add_do_method(this, "_group_selected");
	undo_redo->add_undo_method(this, "_group_selected");
	undo_redo->add_do_method(this, "emit_signal", "group_edited");
	undo_redo->add_undo_method(this, "emit_signal", "group_edited");

	// To force redraw of scene tree.
	undo_redo->add_do_method(SceneTreeDock::get_singleton()->get_tree_editor(), "update_tree");
	undo_redo->add_undo_method(SceneTreeDock::get_singleton()->get_tree_editor(), "update_tree");

	undo_redo->commit_action();
}

void GroupDialog::_remove_filter_changed(const String &p_filter) {
	_group_selected();
}

void GroupDialog::_add_filter_changed(const String &p_filter) {
	_group_selected();
}

void GroupDialog::_add_group_pressed(const String &p_name) {
	_add_group(add_group_text->get_text());
	add_group_text->clear();
}

void GroupDialog::_add_group(String p_name) {
	if (!is_visible()) {
		return; // No need to edit the dialog if it's not being used.
	}

	String name = p_name.strip_edges();
	if (name.is_empty() || groups->get_item_with_text(name)) {
		return;
	}

	TreeItem *new_group = groups->create_item(groups_root);
	new_group->set_text(0, name);
	new_group->add_button(0, groups->get_theme_icon(SNAME("Remove"), SNAME("EditorIcons")), DELETE_GROUP);
	new_group->add_button(0, groups->get_theme_icon(SNAME("ActionCopy"), SNAME("EditorIcons")), COPY_GROUP);
	new_group->set_editable(0, true);
	new_group->select(0);
	groups->ensure_cursor_is_visible();
}

void GroupDialog::_group_renamed() {
	TreeItem *renamed_group = groups->get_edited();
	if (!renamed_group) {
		return;
	}

	const String name = renamed_group->get_text(0).strip_edges();
	for (TreeItem *E = groups_root->get_first_child(); E; E = E->get_next()) {
		if (E != renamed_group && E->get_text(0) == name) {
			renamed_group->set_text(0, selected_group);
			error->set_text(TTR("Group name already exists."));
			error->popup_centered();
			return;
		}
	}

	if (name.is_empty()) {
		renamed_group->set_text(0, selected_group);
		error->set_text(TTR("Invalid group name."));
		error->popup_centered();
		return;
	}

	renamed_group->set_text(0, name); // Spaces trimmed.

	undo_redo->create_action(TTR("Rename Group"));

	List<Node *> nodes;
	scene_tree->get_nodes_in_group(selected_group, &nodes);
	bool removed_all = true;
	for (Node *node : nodes) {
		if (_can_edit(node, selected_group)) {
			undo_redo->add_do_method(node, "remove_from_group", selected_group);
			undo_redo->add_undo_method(node, "remove_from_group", name);
			undo_redo->add_do_method(node, "add_to_group", name, true);
			undo_redo->add_undo_method(node, "add_to_group", selected_group, true);
		} else {
			removed_all = false;
		}
	}

	if (!removed_all) {
		undo_redo->add_do_method(this, "_add_group", selected_group);
		undo_redo->add_undo_method(this, "_delete_group_item", selected_group);
	}

	undo_redo->add_do_method(this, "_rename_group_item", selected_group, name);
	undo_redo->add_undo_method(this, "_rename_group_item", name, selected_group);
	undo_redo->add_do_method(this, "_group_selected");
	undo_redo->add_undo_method(this, "_group_selected");
	undo_redo->add_do_method(this, "emit_signal", "group_edited");
	undo_redo->add_undo_method(this, "emit_signal", "group_edited");

	undo_redo->commit_action();
}

void GroupDialog::_rename_group_item(const String &p_old_name, const String &p_new_name) {
	if (!is_visible()) {
		return; // No need to edit the dialog if it's not being used.
	}

	selected_group = p_new_name;

	for (TreeItem *E = groups_root->get_first_child(); E; E = E->get_next()) {
		if (E->get_text(0) == p_old_name) {
			E->set_text(0, p_new_name);
			return;
		}
	}
}

void GroupDialog::_load_groups(Node *p_current) {
	List<Node::GroupInfo> gi;
	p_current->get_groups(&gi);

	for (const Node::GroupInfo &E : gi) {
		if (!E.persistent) {
			continue;
		}
		_add_group(E.name);
	}

	for (int i = 0; i < p_current->get_child_count(); i++) {
		_load_groups(p_current->get_child(i));
	}
}

void GroupDialog::_modify_group_pressed(Object *p_item, int p_column, int p_id) {
	TreeItem *ti = Object::cast_to<TreeItem>(p_item);
	if (!ti) {
		return;
	}

	switch (p_id) {
		case DELETE_GROUP: {
			String name = ti->get_text(0);

			undo_redo->create_action(TTR("Delete Group"));

			List<Node *> nodes;
			scene_tree->get_nodes_in_group(name, &nodes);
			bool removed_all = true;
			for (Node *E : nodes) {
				if (_can_edit(E, name)) {
					undo_redo->add_do_method(E, "remove_from_group", name);
					undo_redo->add_undo_method(E, "add_to_group", name, true);
				} else {
					removed_all = false;
				}
			}

			if (removed_all) {
				undo_redo->add_do_method(this, "_delete_group_item", name);
				undo_redo->add_undo_method(this, "_add_group", name);
			}

			undo_redo->add_do_method(this, "_group_selected");
			undo_redo->add_undo_method(this, "_group_selected");
			undo_redo->add_do_method(this, "emit_signal", "group_edited");
			undo_redo->add_undo_method(this, "emit_signal", "group_edited");

			// To force redraw of scene tree.
			undo_redo->add_do_method(SceneTreeDock::get_singleton()->get_tree_editor(), "update_tree");
			undo_redo->add_undo_method(SceneTreeDock::get_singleton()->get_tree_editor(), "update_tree");

			undo_redo->commit_action();
		} break;
		case COPY_GROUP: {
			DisplayServer::get_singleton()->clipboard_set(ti->get_text(p_column));
		} break;
	}
}

void GroupDialog::_delete_group_item(const String &p_name) {
	if (!is_visible()) {
		return; // No need to edit the dialog if it's not being used.
	}

	if (selected_group == p_name) {
		add_filter->clear();
		remove_filter->clear();
		nodes_to_remove->clear();
		nodes_to_add->clear();
		groups->deselect_all();
		selected_group = "";
	}

	for (TreeItem *E = groups_root->get_first_child(); E; E = E->get_next()) {
		if (E->get_text(0) == p_name) {
			groups_root->remove_child(E);
			return;
		}
	}
}

void GroupDialog::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_TRANSLATION_CHANGED:
		case Control::NOTIFICATION_LAYOUT_DIRECTION_CHANGED:
		case NOTIFICATION_ENTER_TREE:
		case NOTIFICATION_THEME_CHANGED: {
			if (is_layout_rtl()) {
				add_button->set_icon(groups->get_theme_icon(SNAME("Back"), SNAME("EditorIcons")));
				remove_button->set_icon(groups->get_theme_icon(SNAME("Forward"), SNAME("EditorIcons")));
			} else {
				add_button->set_icon(groups->get_theme_icon(SNAME("Forward"), SNAME("EditorIcons")));
				remove_button->set_icon(groups->get_theme_icon(SNAME("Back"), SNAME("EditorIcons")));
			}

			add_filter->set_right_icon(groups->get_theme_icon(SNAME("Search"), SNAME("EditorIcons")));
			add_filter->set_clear_button_enabled(true);
			remove_filter->set_right_icon(groups->get_theme_icon(SNAME("Search"), SNAME("EditorIcons")));
			remove_filter->set_clear_button_enabled(true);
		} break;
	}
}

void GroupDialog::edit() {
	popup_centered();

	groups->clear();
	groups_root = groups->create_item();

	nodes_to_add->clear();
	nodes_to_remove->clear();

	add_group_text->clear();
	add_filter->clear();
	remove_filter->clear();

	_load_groups(scene_tree->get_edited_scene_root());
}

void GroupDialog::_bind_methods() {
	ClassDB::bind_method("_delete_group_item", &GroupDialog::_delete_group_item);

	ClassDB::bind_method("_add_group", &GroupDialog::_add_group);

	ClassDB::bind_method("_rename_group_item", &GroupDialog::_rename_group_item);

	ADD_SIGNAL(MethodInfo("group_edited"));
}

GroupDialog::GroupDialog() {
	set_min_size(Size2(600, 400) * EDSCALE);

	scene_tree = SceneTree::get_singleton();

	VBoxContainer *vbc = memnew(VBoxContainer);
	add_child(vbc);
	vbc->set_anchors_and_offsets_preset(Control::PRESET_WIDE, Control::PRESET_MODE_KEEP_SIZE, 8 * EDSCALE);

	HBoxContainer *hbc = memnew(HBoxContainer);
	vbc->add_child(hbc);
	hbc->set_v_size_flags(Control::SIZE_EXPAND_FILL);

	VBoxContainer *vbc_left = memnew(VBoxContainer);
	hbc->add_child(vbc_left);
	vbc_left->set_h_size_flags(Control::SIZE_EXPAND_FILL);

	Label *group_title = memnew(Label);
	group_title->set_theme_type_variation("HeaderSmall");

	group_title->set_text(TTR("Groups"));
	vbc_left->add_child(group_title);

	groups = memnew(Tree);
	vbc_left->add_child(groups);
	groups->set_hide_root(true);
	groups->set_select_mode(Tree::SELECT_SINGLE);
	groups->set_allow_reselect(true);
	groups->set_allow_rmb_select(true);
	groups->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	groups->add_theme_constant_override("draw_guides", 1);
	groups->connect("item_selected", callable_mp(this, &GroupDialog::_group_selected));
	groups->connect("button_pressed", callable_mp(this, &GroupDialog::_modify_group_pressed));
	groups->connect("item_edited", callable_mp(this, &GroupDialog::_group_renamed));

	HBoxContainer *chbc = memnew(HBoxContainer);
	vbc_left->add_child(chbc);
	chbc->set_h_size_flags(Control::SIZE_EXPAND_FILL);

	add_group_text = memnew(LineEdit);
	chbc->add_child(add_group_text);
	add_group_text->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	add_group_text->connect("text_submitted", callable_mp(this, &GroupDialog::_add_group_pressed));

	Button *add_group_button = memnew(Button);
	add_group_button->set_text(TTR("Add"));
	chbc->add_child(add_group_button);
	add_group_button->connect("pressed", callable_mp(this, &GroupDialog::_add_group_pressed), varray(String()));

	VBoxContainer *vbc_add = memnew(VBoxContainer);
	hbc->add_child(vbc_add);
	vbc_add->set_h_size_flags(Control::SIZE_EXPAND_FILL);

	Label *out_of_group_title = memnew(Label);
	out_of_group_title->set_theme_type_variation("HeaderSmall");

	out_of_group_title->set_text(TTR("Nodes Not in Group"));
	vbc_add->add_child(out_of_group_title);

	nodes_to_add = memnew(Tree);
	vbc_add->add_child(nodes_to_add);
	nodes_to_add->set_hide_root(true);
	nodes_to_add->set_hide_folding(true);
	nodes_to_add->set_select_mode(Tree::SELECT_MULTI);
	nodes_to_add->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	nodes_to_add->add_theme_constant_override("draw_guides", 1);

	HBoxContainer *add_filter_hbc = memnew(HBoxContainer);
	add_filter_hbc->add_theme_constant_override("separate", 0);
	vbc_add->add_child(add_filter_hbc);

	add_filter = memnew(LineEdit);
	add_filter->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	add_filter->set_placeholder(TTR("Filter nodes"));
	add_filter_hbc->add_child(add_filter);
	add_filter->connect("text_changed", callable_mp(this, &GroupDialog::_add_filter_changed));

	VBoxContainer *vbc_buttons = memnew(VBoxContainer);
	hbc->add_child(vbc_buttons);
	vbc_buttons->set_h_size_flags(Control::SIZE_SHRINK_CENTER);
	vbc_buttons->set_v_size_flags(Control::SIZE_SHRINK_CENTER);

	add_button = memnew(Button);
	add_button->set_flat(true);
	add_button->set_text(TTR("Add"));
	add_button->connect("pressed", callable_mp(this, &GroupDialog::_add_pressed));

	vbc_buttons->add_child(add_button);
	vbc_buttons->add_spacer();
	vbc_buttons->add_spacer();
	vbc_buttons->add_spacer();

	remove_button = memnew(Button);
	remove_button->set_flat(true);
	remove_button->set_text(TTR("Remove"));
	remove_button->connect("pressed", callable_mp(this, &GroupDialog::_removed_pressed));

	vbc_buttons->add_child(remove_button);

	VBoxContainer *vbc_remove = memnew(VBoxContainer);
	hbc->add_child(vbc_remove);
	vbc_remove->set_h_size_flags(Control::SIZE_EXPAND_FILL);

	Label *in_group_title = memnew(Label);
	in_group_title->set_theme_type_variation("HeaderSmall");

	in_group_title->set_text(TTR("Nodes in Group"));
	vbc_remove->add_child(in_group_title);

	nodes_to_remove = memnew(Tree);
	vbc_remove->add_child(nodes_to_remove);
	nodes_to_remove->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	nodes_to_remove->set_hide_root(true);
	nodes_to_remove->set_hide_folding(true);
	nodes_to_remove->set_select_mode(Tree::SELECT_MULTI);
	nodes_to_remove->add_theme_constant_override("draw_guides", 1);

	HBoxContainer *remove_filter_hbc = memnew(HBoxContainer);
	remove_filter_hbc->add_theme_constant_override("separate", 0);
	vbc_remove->add_child(remove_filter_hbc);

	remove_filter = memnew(LineEdit);
	remove_filter->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	remove_filter->set_placeholder(TTR("Filter nodes"));
	remove_filter_hbc->add_child(remove_filter);
	remove_filter->connect("text_changed", callable_mp(this, &GroupDialog::_remove_filter_changed));

	group_empty = memnew(Label());
	group_empty->set_theme_type_variation("HeaderSmall");

	group_empty->set_text(TTR("Empty groups will be automatically removed."));
	group_empty->set_vertical_alignment(VERTICAL_ALIGNMENT_CENTER);
	group_empty->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	group_empty->set_autowrap_mode(Label::AUTOWRAP_WORD_SMART);
	group_empty->set_custom_minimum_size(Size2(100 * EDSCALE, 0));
	nodes_to_remove->add_child(group_empty);
	group_empty->set_anchors_and_offsets_preset(Control::PRESET_WIDE, Control::PRESET_MODE_KEEP_SIZE, 8 * EDSCALE);

	set_title(TTR("Group Editor"));

	error = memnew(ConfirmationDialog);
	add_child(error);
	error->get_ok_button()->set_text(TTR("Close"));
}

////////////////////////////////////////////////////////////////////////////////

void GroupsEditor::_add_group(const String &p_group) {
	if (!node) {
		return;
	}

	const String name = group_name->get_text().strip_edges();
	if (name.is_empty()) {
		return;
	}

	if (node->is_in_group(name)) {
		return;
	}

	undo_redo->create_action(TTR("Add to Group"));

	undo_redo->add_do_method(node, "add_to_group", name, true);
	undo_redo->add_undo_method(node, "remove_from_group", name);
	undo_redo->add_do_method(this, "update_tree");
	undo_redo->add_undo_method(this, "update_tree");

	// To force redraw of scene tree.
	undo_redo->add_do_method(SceneTreeDock::get_singleton()->get_tree_editor(), "update_tree");
	undo_redo->add_undo_method(SceneTreeDock::get_singleton()->get_tree_editor(), "update_tree");

	undo_redo->commit_action();

	group_name->clear();
}

void GroupsEditor::_modify_group(Object *p_item, int p_column, int p_id) {
	if (!node) {
		return;
	}

	TreeItem *ti = Object::cast_to<TreeItem>(p_item);
	if (!ti) {
		return;
	}
	switch (p_id) {
		case DELETE_GROUP: {
			String name = ti->get_text(0);
			undo_redo->create_action(TTR("Remove from Group"));

			undo_redo->add_do_method(node, "remove_from_group", name);
			undo_redo->add_undo_method(node, "add_to_group", name, true);
			undo_redo->add_do_method(this, "update_tree");
			undo_redo->add_undo_method(this, "update_tree");

			// To force redraw of scene tree.
			undo_redo->add_do_method(SceneTreeDock::get_singleton()->get_tree_editor(), "update_tree");
			undo_redo->add_undo_method(SceneTreeDock::get_singleton()->get_tree_editor(), "update_tree");

			undo_redo->commit_action();
		} break;
		case COPY_GROUP: {
			DisplayServer::get_singleton()->clipboard_set(ti->get_text(p_column));
		} break;
	}
}

struct _GroupInfoComparator {
	bool operator()(const Node::GroupInfo &p_a, const Node::GroupInfo &p_b) const {
		return p_a.name.operator String() < p_b.name.operator String();
	}
};

void GroupsEditor::update_tree() {
	tree->clear();

	if (!node) {
		return;
	}

	List<Node::GroupInfo> groups;
	node->get_groups(&groups);
	groups.sort_custom<_GroupInfoComparator>();

	TreeItem *root = tree->create_item();

	for (const GroupInfo &gi : groups) {
		if (!gi.persistent) {
			continue;
		}

		Node *n = node;
		bool can_be_deleted = true;

		while (n) {
			Ref<SceneState> ss = (n == EditorNode::get_singleton()->get_edited_scene()) ? n->get_scene_inherited_state() : n->get_scene_instance_state();

			if (ss.is_valid()) {
				int path = ss->find_node_by_path(n->get_path_to(node));
				if (path != -1) {
					if (ss->is_node_in_group(path, gi.name)) {
						can_be_deleted = false;
					}
				}
			}

			n = n->get_owner();
		}

		TreeItem *item = tree->create_item(root);
		item->set_text(0, gi.name);
		if (can_be_deleted) {
			item->add_button(0, get_theme_icon(SNAME("Remove"), SNAME("EditorIcons")), DELETE_GROUP);
			item->add_button(0, get_theme_icon(SNAME("ActionCopy"), SNAME("EditorIcons")), COPY_GROUP);
		} else {
			item->set_selectable(0, false);
		}
	}
}

void GroupsEditor::set_current(Node *p_node) {
	node = p_node;
	update_tree();
}

void GroupsEditor::_show_group_dialog() {
	group_dialog->edit();
	group_dialog->set_undo_redo(undo_redo);
}

void GroupsEditor::_bind_methods() {
	ClassDB::bind_method("update_tree", &GroupsEditor::update_tree);
}

GroupsEditor::GroupsEditor() {
	node = nullptr;

	VBoxContainer *vbc = this;

	group_dialog = memnew(GroupDialog);

	add_child(group_dialog);
	group_dialog->connect("group_edited", callable_mp(this, &GroupsEditor::update_tree));

	Button *group_dialog_button = memnew(Button);
	group_dialog_button->set_text(TTR("Manage Groups"));
	vbc->add_child(group_dialog_button);
	group_dialog_button->connect("pressed", callable_mp(this, &GroupsEditor::_show_group_dialog));

	HBoxContainer *hbc = memnew(HBoxContainer);
	vbc->add_child(hbc);

	group_name = memnew(LineEdit);
	group_name->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	hbc->add_child(group_name);
	group_name->connect("text_submitted", callable_mp(this, &GroupsEditor::_add_group));

	add = memnew(Button);
	add->set_text(TTR("Add"));
	hbc->add_child(add);
	add->connect("pressed", callable_mp(this, &GroupsEditor::_add_group), varray(String()));

	tree = memnew(Tree);
	tree->set_hide_root(true);
	tree->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	vbc->add_child(tree);
	tree->connect("button_pressed", callable_mp(this, &GroupsEditor::_modify_group));
	tree->add_theme_constant_override("draw_guides", 1);
	add_theme_constant_override("separation", 3 * EDSCALE);
}

GroupsEditor::~GroupsEditor() {
}
