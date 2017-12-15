/*************************************************************************/
/*  scene_tree_editor.cpp                                                */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
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
#include "scene_tree_editor.h"

#include "editor/plugins/canvas_item_editor_plugin.h"
#include "editor_node.h"
#include "message_queue.h"
#include "print_string.h"
#include "scene/gui/label.h"
#include "scene/main/viewport.h"
#include "scene/resources/packed_scene.h"

Node *SceneTreeEditor::get_scene_node() {

	ERR_FAIL_COND_V(!is_inside_tree(), NULL);

	return get_tree()->get_edited_scene_root();
}

void SceneTreeEditor::_cell_button_pressed(Object *p_item, int p_column, int p_id) {

	TreeItem *item = Object::cast_to<TreeItem>(p_item);
	ERR_FAIL_COND(!item);

	NodePath np = item->get_metadata(0);

	Node *n = get_node(np);
	ERR_FAIL_COND(!n);

	if (p_id == BUTTON_SUBSCENE) {
		if (n == get_scene_node()) {
			if (n && n->get_scene_inherited_state().is_valid()) {
				emit_signal("open", n->get_scene_inherited_state()->get_path());
			}
		} else {
			emit_signal("open", n->get_filename());
		}
	} else if (p_id == BUTTON_SCRIPT) {
		RefPtr script = n->get_script();
		if (!script.is_null())
			emit_signal("open_script", script);

	} else if (p_id == BUTTON_VISIBILITY) {

		if (n->is_class("Spatial")) {

			bool v = bool(n->call("is_visible"));
			undo_redo->create_action(TTR("Toggle Spatial Visible"));
			undo_redo->add_do_method(n, "set_visible", !v);
			undo_redo->add_undo_method(n, "set_visible", v);
			undo_redo->commit_action();

		} else if (n->is_class("CanvasItem")) {

			bool v = bool(n->call("is_visible"));
			undo_redo->create_action(TTR("Toggle CanvasItem Visible"));
			undo_redo->add_do_method(n, v ? "hide" : "show");
			undo_redo->add_undo_method(n, v ? "show" : "hide");
			undo_redo->commit_action();
		}

	} else if (p_id == BUTTON_LOCK) {

		if (n->is_class("CanvasItem") || n->is_class("Spatial")) {
			n->set_meta("_edit_lock_", Variant());
			_update_tree();
			emit_signal("node_changed");
		}

	} else if (p_id == BUTTON_GROUP) {
		if (n->is_class("CanvasItem")) {
			n->set_meta("_edit_group_", Variant());
			_update_tree();
			emit_signal("node_changed");
		}
	} else if (p_id == BUTTON_WARNING) {

		String config_err = n->get_configuration_warning();
		if (config_err == String())
			return;
		config_err = config_err.word_wrap(80);
		warning->set_text(config_err);
		warning->popup_centered_minsize();

	} else if (p_id == BUTTON_SIGNALS) {

		editor_selection->clear();
		editor_selection->add_node(n);

		set_selected(n);

		NodeDock::singleton->get_parent()->call("set_current_tab", NodeDock::singleton->get_index());
		NodeDock::singleton->show_connections();

	} else if (p_id == BUTTON_GROUPS) {

		editor_selection->clear();
		editor_selection->add_node(n);

		set_selected(n);

		NodeDock::singleton->get_parent()->call("set_current_tab", NodeDock::singleton->get_index());
		NodeDock::singleton->show_groups();
	}
}

bool SceneTreeEditor::_add_nodes(Node *p_node, TreeItem *p_parent) {

	if (!p_node)
		return false;

	// only owned nodes are editable, since nodes can create their own (manually owned) child nodes,
	// which the editor needs not to know about.

	bool part_of_subscene = false;

	if (!display_foreign && p_node->get_owner() != get_scene_node() && p_node != get_scene_node()) {

		if ((show_enabled_subscene || can_open_instance) && p_node->get_owner() && (get_scene_node()->is_editable_instance(p_node->get_owner()))) {

			part_of_subscene = true;
			//allow
		} else {
			return false;
		}
	} else {
		part_of_subscene = p_node != get_scene_node() && get_scene_node()->get_scene_inherited_state().is_valid() && get_scene_node()->get_scene_inherited_state()->find_node_by_path(get_scene_node()->get_path_to(p_node)) >= 0;
	}

	TreeItem *item = tree->create_item(p_parent);
	item->set_text(0, p_node->get_name());
	if (can_rename && !part_of_subscene /*(p_node->get_owner() == get_scene_node() || p_node==get_scene_node())*/)
		item->set_editable(0, true);

	item->set_selectable(0, true);
	if (can_rename) {
#ifndef DISABLE_DEPRECATED
		if (p_node->has_meta("_editor_collapsed")) {
			//remove previous way of storing folding, which did not get along with scene inheritance and instancing
			if ((bool)p_node->get_meta("_editor_collapsed"))
				p_node->set_display_folded(true);
			p_node->set_meta("_editor_collapsed", Variant());
		}
#endif
		bool collapsed = p_node->is_displayed_folded();
		if (collapsed)
			item->set_collapsed(true);
	}

	Ref<Texture> icon;
	if (p_node->has_meta("_editor_icon"))
		icon = p_node->get_meta("_editor_icon");
	else
		icon = get_icon((has_icon(p_node->get_class(), "EditorIcons") ? p_node->get_class() : String("Object")), "EditorIcons");
	item->set_icon(0, icon);
	item->set_metadata(0, p_node->get_path());

	if (part_of_subscene) {

		//item->set_selectable(0,marked_selectable);
		item->set_custom_color(0, get_color("disabled_font_color", "Editor"));

	} else if (marked.has(p_node)) {

		item->set_selectable(0, marked_selectable);
		item->set_custom_color(0, get_color("error_color", "Editor"));
	} else if (!marked_selectable && !marked_children_selectable) {

		Node *node = p_node;
		while (node) {
			if (marked.has(node)) {
				item->set_selectable(0, false);
				item->set_custom_color(0, get_color("error_color", "Editor"));
				break;
			}
			node = node->get_parent();
		}
	}

	if (can_rename) { //should be can edit..

		String warning = p_node->get_configuration_warning();
		if (warning != String()) {
			item->add_button(0, get_icon("NodeWarning", "EditorIcons"), BUTTON_WARNING, false, TTR("Node configuration warning:") + "\n" + p_node->get_configuration_warning());
		}

		bool has_connections = p_node->has_persistent_signal_connections();
		bool has_groups = p_node->has_persistent_groups();

		if (has_connections && has_groups) {
			item->add_button(0, get_icon("SignalsAndGroups", "EditorIcons"), BUTTON_SIGNALS, false, TTR("Node has connection(s) and group(s)\nClick to show signals dock."));
		} else if (has_connections) {
			item->add_button(0, get_icon("Signals", "EditorIcons"), BUTTON_SIGNALS, false, TTR("Node has connections.\nClick to show signals dock."));
		} else if (has_groups) {
			item->add_button(0, get_icon("Groups", "EditorIcons"), BUTTON_GROUPS, false, TTR("Node is in group(s).\nClick to show groups dock."));
		}
	}

	if (p_node == get_scene_node() && p_node->get_scene_inherited_state().is_valid()) {
		item->add_button(0, get_icon("InstanceOptions", "EditorIcons"), BUTTON_SUBSCENE, false, TTR("Open in Editor"));
		item->set_tooltip(0, TTR("Inherits:") + " " + p_node->get_scene_inherited_state()->get_path() + "\n" + TTR("Type:") + " " + p_node->get_class());
	} else if (p_node != get_scene_node() && p_node->get_filename() != "" && can_open_instance) {

		item->add_button(0, get_icon("InstanceOptions", "EditorIcons"), BUTTON_SUBSCENE, false, TTR("Open in Editor"));
		item->set_tooltip(0, TTR("Instance:") + " " + p_node->get_filename() + "\n" + TTR("Type:") + " " + p_node->get_class());
	} else {
		item->set_tooltip(0, String(p_node->get_name()) + "\n" + TTR("Type:") + " " + p_node->get_class());
	}

	if (can_open_instance) {

		if (!p_node->is_connected("script_changed", this, "_node_script_changed"))
			p_node->connect("script_changed", this, "_node_script_changed", varray(p_node));

		if (!p_node->get_script().is_null()) {

			item->add_button(0, get_icon("Script", "EditorIcons"), BUTTON_SCRIPT, false, TTR("Open script"));
		}

		if (p_node->is_class("CanvasItem")) {

			bool is_locked = p_node->has_meta("_edit_lock_"); //_edit_group_
			if (is_locked)
				item->add_button(0, get_icon("Lock", "EditorIcons"), BUTTON_LOCK, false, TTR("Node is locked.\nClick to unlock"));

			bool is_grouped = p_node->has_meta("_edit_group_");
			if (is_grouped)
				item->add_button(0, get_icon("Group", "EditorIcons"), BUTTON_GROUP, false, TTR("Children are not selectable.\nClick to make selectable"));

			bool v = p_node->call("is_visible");
			if (v)
				item->add_button(0, get_icon("GuiVisibilityVisible", "EditorIcons"), BUTTON_VISIBILITY, false, TTR("Toggle Visibility"));
			else
				item->add_button(0, get_icon("GuiVisibilityHidden", "EditorIcons"), BUTTON_VISIBILITY, false, TTR("Toggle Visibility"));

			if (!p_node->is_connected("visibility_changed", this, "_node_visibility_changed"))
				p_node->connect("visibility_changed", this, "_node_visibility_changed", varray(p_node));

			_update_visibility_color(p_node, item);
		} else if (p_node->is_class("Spatial")) {

			bool is_locked = p_node->has_meta("_edit_lock_");
			if (is_locked)
				item->add_button(0, get_icon("Lock", "EditorIcons"), BUTTON_LOCK, false, TTR("Node is locked.\nClick to unlock"));

			bool v = p_node->call("is_visible");
			if (v)
				item->add_button(0, get_icon("GuiVisibilityVisible", "EditorIcons"), BUTTON_VISIBILITY, false, TTR("Toggle Visibility"));
			else
				item->add_button(0, get_icon("GuiVisibilityHidden", "EditorIcons"), BUTTON_VISIBILITY, false, TTR("Toggle Visibility"));

			if (!p_node->is_connected("visibility_changed", this, "_node_visibility_changed"))
				p_node->connect("visibility_changed", this, "_node_visibility_changed", varray(p_node));

			_update_visibility_color(p_node, item);
		}
	}

	if (editor_selection) {
		if (editor_selection->is_selected(p_node)) {

			item->select(0);
		}
	}

	if (selected == p_node) {
		if (!editor_selection)
			item->select(0);
		item->set_as_cursor(0);
	}

	bool keep = (filter.is_subsequence_ofi(String(p_node->get_name())));

	for (int i = 0; i < p_node->get_child_count(); i++) {

		bool child_keep = _add_nodes(p_node->get_child(i), item);

		keep = keep || child_keep;
	}

	if (!keep) {
		memdelete(item);
		return false;
	} else {
		return true;
	}
}

void SceneTreeEditor::_node_visibility_changed(Node *p_node) {

	if (p_node != get_scene_node() && !p_node->get_owner()) {

		return;
	}
	TreeItem *item = p_node ? _find(tree->get_root(), p_node->get_path()) : NULL;
	if (!item) {

		return;
	}
	int idx = item->get_button_by_id(0, BUTTON_VISIBILITY);
	ERR_FAIL_COND(idx == -1);

	bool visible = false;

	if (p_node->is_class("CanvasItem")) {
		visible = p_node->call("is_visible");
		CanvasItemEditor::get_singleton()->get_viewport_control()->update();
	} else if (p_node->is_class("Spatial")) {
		visible = p_node->call("is_visible");
	}

	if (visible)
		item->set_button(0, idx, get_icon("GuiVisibilityVisible", "EditorIcons"));
	else
		item->set_button(0, idx, get_icon("GuiVisibilityHidden", "EditorIcons"));

	_update_visibility_color(p_node, item);
}

void SceneTreeEditor::_update_visibility_color(Node *p_node, TreeItem *p_item) {
	if (p_node->is_class("CanvasItem") || p_node->is_class("Spatial")) {
		Color color(1, 1, 1, 1);
		bool visible_on_screen = p_node->call("is_visible_in_tree");
		if (!visible_on_screen) {
			color.a = 0.6;
		}
		int idx = p_item->get_button_by_id(0, BUTTON_VISIBILITY);
		p_item->set_button_color(0, idx, color);
	}
}

void SceneTreeEditor::_node_script_changed(Node *p_node) {

	if (tree_dirty)
		return;

	MessageQueue::get_singleton()->push_call(this, "_update_tree");
	tree_dirty = true;
	/*
	changes the order :|
	TreeItem* item=p_node?_find(tree->get_root(),p_node->get_path()):NULL;
	if (p_node->get_script().is_null()) {

		int idx=item->get_button_by_id(0,2);
		if (idx>=0)
			item->erase_button(0,idx);
	} else {

		int idx=item->get_button_by_id(0,2);
		if (idx<0)
			item->add_button(0,get_icon("Script","EditorIcons"),2);

	}*/
}

void SceneTreeEditor::_node_removed(Node *p_node) {

	if (EditorNode::get_singleton()->is_exiting())
		return; //speed up exit

	if (p_node->is_connected("script_changed", this, "_node_script_changed"))
		p_node->disconnect("script_changed", this, "_node_script_changed");

	if (p_node->is_class("Spatial") || p_node->is_class("CanvasItem")) {
		if (p_node->is_connected("visibility_changed", this, "_node_visibility_changed"))
			p_node->disconnect("visibility_changed", this, "_node_visibility_changed");
	}

	if (p_node == selected) {
		selected = NULL;
		emit_signal("node_selected");
	}
}
void SceneTreeEditor::_update_tree() {

	if (!is_inside_tree()) {
		tree_dirty = false;
		return;
	}

	updating_tree = true;
	tree->clear();
	if (get_scene_node()) {
		_add_nodes(get_scene_node(), NULL);
		last_hash = hash_djb2_one_64(0);
		_compute_hash(get_scene_node(), last_hash);
	}
	updating_tree = false;

	tree_dirty = false;
}

void SceneTreeEditor::_compute_hash(Node *p_node, uint64_t &hash) {

	hash = hash_djb2_one_64(p_node->get_instance_id(), hash);
	if (p_node->get_parent())
		hash = hash_djb2_one_64(p_node->get_parent()->get_instance_id(), hash); //so a reparent still produces a different hash

	for (int i = 0; i < p_node->get_child_count(); i++) {

		_compute_hash(p_node->get_child(i), hash);
	}
}

void SceneTreeEditor::_test_update_tree() {

	pending_test_update = false;

	if (!is_inside_tree())
		return;

	if (tree_dirty)
		return; // don't even bother

	uint64_t hash = hash_djb2_one_64(0);
	if (get_scene_node())
		_compute_hash(get_scene_node(), hash);
	//test hash
	if (hash == last_hash)
		return; // did not change

	MessageQueue::get_singleton()->push_call(this, "_update_tree");
	tree_dirty = true;
}

void SceneTreeEditor::_tree_changed() {

	if (EditorNode::get_singleton()->is_exiting())
		return; //speed up exit
	if (pending_test_update)
		return;
	if (tree_dirty)
		return;

	MessageQueue::get_singleton()->push_call(this, "_test_update_tree");
	pending_test_update = true;
}

void SceneTreeEditor::_selected_changed() {

	TreeItem *s = tree->get_selected();
	ERR_FAIL_COND(!s);
	NodePath np = s->get_metadata(0);

	Node *n = get_node(np);

	if (n == selected)
		return;

	selected = get_node(np);

	blocked++;
	emit_signal("node_selected");
	blocked--;
}

void SceneTreeEditor::_deselect_items() {

	// Clear currently elected items in scene tree dock.
	if (editor_selection)
		editor_selection->clear();
}

void SceneTreeEditor::_cell_multi_selected(Object *p_object, int p_cell, bool p_selected) {

	TreeItem *item = Object::cast_to<TreeItem>(p_object);
	ERR_FAIL_COND(!item);

	NodePath np = item->get_metadata(0);

	Node *n = get_node(np);

	if (!n)
		return;

	if (!editor_selection)
		return;

	if (p_selected) {
		editor_selection->add_node(n);

	} else {
		editor_selection->remove_node(n);
	}
}

void SceneTreeEditor::_notification(int p_what) {

	if (p_what == NOTIFICATION_ENTER_TREE) {

		get_tree()->connect("tree_changed", this, "_tree_changed");
		get_tree()->connect("node_removed", this, "_node_removed");
		get_tree()->connect("node_configuration_warning_changed", this, "_warning_changed");

		tree->connect("item_collapsed", this, "_cell_collapsed");

		EditorSettings::get_singleton()->connect("settings_changed", this, "_editor_settings_changed");

		//get_scene()->connect("tree_changed",this,"_tree_changed",Vector<Variant>(),CONNECT_DEFERRED);
		//get_scene()->connect("node_removed",this,"_node_removed",Vector<Variant>(),CONNECT_DEFERRED);
		_update_tree();
	}
	if (p_what == NOTIFICATION_EXIT_TREE) {

		get_tree()->disconnect("tree_changed", this, "_tree_changed");
		get_tree()->disconnect("node_removed", this, "_node_removed");
		tree->disconnect("item_collapsed", this, "_cell_collapsed");
		get_tree()->disconnect("node_configuration_warning_changed", this, "_warning_changed");
		EditorSettings::get_singleton()->disconnect("settings_changed", this, "_editor_settings_changed");
	}
}

TreeItem *SceneTreeEditor::_find(TreeItem *p_node, const NodePath &p_path) {

	if (!p_node)
		return NULL;

	NodePath np = p_node->get_metadata(0);
	if (np == p_path)
		return p_node;

	TreeItem *children = p_node->get_children();
	while (children) {

		TreeItem *n = _find(children, p_path);
		if (n)
			return n;
		children = children->get_next();
	}

	return NULL;
}

void SceneTreeEditor::set_selected(Node *p_node, bool p_emit_selected) {

	ERR_FAIL_COND(blocked > 0);

	if (pending_test_update)
		_test_update_tree();
	if (tree_dirty)
		_update_tree();

	if (selected == p_node)
		return;

	TreeItem *item = p_node ? _find(tree->get_root(), p_node->get_path()) : NULL;

	if (item) {
		// make visible when it's collapsed
		TreeItem *node = item->get_parent();
		while (node && node != tree->get_root()) {
			node->set_collapsed(false);
			node = node->get_parent();
		}
		item->select(0);
		item->set_as_cursor(0);
		selected = p_node;
		tree->ensure_cursor_is_visible();
	} else {
		if (!p_node)
			selected = NULL;
		_update_tree();
		selected = p_node;
	}

	if (p_emit_selected) {
		emit_signal("node_selected");
	}
}

void SceneTreeEditor::_rename_node(ObjectID p_node, const String &p_name) {

	Object *o = ObjectDB::get_instance(p_node);
	ERR_FAIL_COND(!o);
	Node *n = Object::cast_to<Node>(o);
	ERR_FAIL_COND(!n);
	TreeItem *item = _find(tree->get_root(), n->get_path());
	ERR_FAIL_COND(!item);

	n->set_name(p_name);
	item->set_metadata(0, n->get_path());
	item->set_text(0, p_name);
	emit_signal("node_renamed");

	if (!tree_dirty) {
		MessageQueue::get_singleton()->push_call(this, "_update_tree");
		tree_dirty = true;
	}
}

void SceneTreeEditor::_renamed() {

	TreeItem *which = tree->get_edited();

	ERR_FAIL_COND(!which);
	NodePath np = which->get_metadata(0);
	Node *n = get_node(np);
	ERR_FAIL_COND(!n);

	String new_name = which->get_text(0);
	if (new_name.find(".") != -1 || new_name.find("/") != -1) {

		error->set_text(TTR("Invalid node name, the following characters are not allowed:") + "\n  \".\", \"/\"");
		error->popup_centered_minsize();
		new_name = n->get_name();
	}

	if (new_name == n->get_name())
		return;

	if (!undo_redo) {
		n->set_name(new_name);
		which->set_metadata(0, n->get_path());
		emit_signal("node_renamed");
	} else {
		undo_redo->create_action(TTR("Rename Node"));
		emit_signal("node_prerename", n, new_name);
		undo_redo->add_do_method(this, "_rename_node", n->get_instance_id(), new_name);
		undo_redo->add_undo_method(this, "_rename_node", n->get_instance_id(), n->get_name());
		undo_redo->commit_action();
	}
}

Node *SceneTreeEditor::get_selected() {

	return selected;
}

void SceneTreeEditor::set_marked(const Set<Node *> &p_marked, bool p_selectable, bool p_children_selectable) {

	if (tree_dirty)
		_update_tree();
	marked = p_marked;
	marked_selectable = p_selectable;
	marked_children_selectable = p_children_selectable;
	_update_tree();
}

void SceneTreeEditor::set_marked(Node *p_marked, bool p_selectable, bool p_children_selectable) {

	Set<Node *> s;
	if (p_marked)
		s.insert(p_marked);
	set_marked(s, p_selectable, p_children_selectable);
}

void SceneTreeEditor::set_filter(const String &p_filter) {

	filter = p_filter;
	_update_tree();
}

String SceneTreeEditor::get_filter() const {

	return filter;
}

void SceneTreeEditor::set_display_foreign_nodes(bool p_display) {

	display_foreign = p_display;
	_update_tree();
}
bool SceneTreeEditor::get_display_foreign_nodes() const {

	return display_foreign;
}

void SceneTreeEditor::set_editor_selection(EditorSelection *p_selection) {

	editor_selection = p_selection;
	tree->set_select_mode(Tree::SELECT_MULTI);
	tree->set_cursor_can_exit_tree(false);
	editor_selection->connect("selection_changed", this, "_selection_changed");
}

void SceneTreeEditor::_update_selection(TreeItem *item) {

	ERR_FAIL_COND(!item);

	NodePath np = item->get_metadata(0);

	if (!has_node(np))
		return;

	Node *n = get_node(np);

	if (!n)
		return;

	if (editor_selection->is_selected(n))
		item->select(0);
	else
		item->deselect(0);

	TreeItem *c = item->get_children();

	while (c) {

		_update_selection(c);
		c = c->get_next();
	}
}

void SceneTreeEditor::_selection_changed() {

	if (!editor_selection)
		return;

	TreeItem *root = tree->get_root();

	if (!root)
		return;
	_update_selection(root);
}

void SceneTreeEditor::_cell_collapsed(Object *p_obj) {

	if (updating_tree)
		return;
	if (!can_rename)
		return;

	TreeItem *ti = Object::cast_to<TreeItem>(p_obj);
	if (!ti)
		return;

	bool collapsed = ti->is_collapsed();

	NodePath np = ti->get_metadata(0);

	Node *n = get_node(np);
	ERR_FAIL_COND(!n);

	n->set_display_folded(collapsed);
}

Variant SceneTreeEditor::get_drag_data_fw(const Point2 &p_point, Control *p_from) {
	if (!can_rename)
		return Variant(); //not editable tree

	Vector<Node *> selected;
	Vector<Ref<Texture> > icons;
	TreeItem *next = tree->get_next_selected(NULL);
	while (next) {

		NodePath np = next->get_metadata(0);

		Node *n = get_node(np);
		if (n) {
			// Only allow selection if not part of an instanced scene.
			if (!n->get_owner() || n->get_owner() == get_scene_node() || n->get_owner()->get_filename() == String()) {
				selected.push_back(n);
				icons.push_back(next->get_icon(0));
			}
		}
		next = tree->get_next_selected(next);
	}

	if (selected.empty())
		return Variant();

	VBoxContainer *vb = memnew(VBoxContainer);
	Array objs;
	int list_max = 10;
	float opacity_step = 1.0f / list_max;
	float opacity_item = 1.0f;
	for (int i = 0; i < selected.size(); i++) {

		if (i < list_max) {
			HBoxContainer *hb = memnew(HBoxContainer);
			TextureRect *tf = memnew(TextureRect);
			tf->set_texture(icons[i]);
			hb->add_child(tf);
			Label *label = memnew(Label(selected[i]->get_name()));
			hb->add_child(label);
			vb->add_child(hb);
			hb->set_modulate(Color(1, 1, 1, opacity_item));
			opacity_item -= opacity_step;
		}
		NodePath p = selected[i]->get_path();
		objs.push_back(p);
	}

	set_drag_preview(vb);
	Dictionary drag_data;
	drag_data["type"] = "nodes";
	drag_data["nodes"] = objs;

	tree->set_drop_mode_flags(Tree::DROP_MODE_INBETWEEN | Tree::DROP_MODE_ON_ITEM);
	emit_signal("nodes_dragged");

	return drag_data;
}

bool SceneTreeEditor::_is_script_type(const StringName &p_type) const {
	return (script_types->find(p_type));
}

bool SceneTreeEditor::can_drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) const {

	if (!can_rename)
		return false; //not editable tree
	if (filter != String())
		return false; //can't rearrange tree with filter turned on

	Dictionary d = p_data;
	if (!d.has("type"))
		return false;

	TreeItem *item = tree->get_item_at_position(p_point);
	if (!item)
		return false;

	int section = tree->get_drop_section_at_position(p_point);
	if (section < -1 || (section == -1 && !item->get_parent()))
		return false;

	if (String(d["type"]) == "files") {

		Vector<String> files = d["files"];

		if (files.size() == 0)
			return false; //weird

		if (_is_script_type(EditorFileSystem::get_singleton()->get_file_type(files[0]))) {
			tree->set_drop_mode_flags(Tree::DROP_MODE_ON_ITEM);
			return true;
		}

		for (int i = 0; i < files.size(); i++) {
			String file = files[i];
			String ftype = EditorFileSystem::get_singleton()->get_file_type(file);
			if (ftype != "PackedScene")
				return false;
		}

		tree->set_drop_mode_flags(Tree::DROP_MODE_INBETWEEN | Tree::DROP_MODE_ON_ITEM); //so it works..

		return true;
	}

	if (String(d["type"]) == "nodes") {
		return true;
	}

	return false;
}
void SceneTreeEditor::drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) {

	if (!can_drop_data_fw(p_point, p_data, p_from))
		return;

	TreeItem *item = tree->get_item_at_position(p_point);
	if (!item)
		return;
	int section = tree->get_drop_section_at_position(p_point);
	if (section < -1)
		return;

	NodePath np = item->get_metadata(0);
	Node *n = get_node(np);
	if (!n)
		return;

	Dictionary d = p_data;

	if (String(d["type"]) == "nodes") {
		Array nodes = d["nodes"];
		emit_signal("nodes_rearranged", nodes, np, section);
	}

	if (String(d["type"]) == "files") {

		Vector<String> files = d["files"];

		String ftype = EditorFileSystem::get_singleton()->get_file_type(files[0]);
		if (_is_script_type(ftype)) {
			emit_signal("script_dropped", files[0], np);
		} else {
			emit_signal("files_dropped", files, np, section);
		}
	}
}

void SceneTreeEditor::_rmb_select(const Vector2 &p_pos) {

	emit_signal("rmb_pressed", tree->get_global_transform().xform(p_pos));
}

void SceneTreeEditor::_warning_changed(Node *p_for_node) {

	//should use a timer
	update_timer->start();
	//print_line("WARNING CHANGED "+String(p_for_node->get_name()));
}

void SceneTreeEditor::_editor_settings_changed() {
	bool enable_rl = EditorSettings::get_singleton()->get("docks/scene_tree/draw_relationship_lines");
	Color rl_color = EditorSettings::get_singleton()->get("docks/scene_tree/relationship_line_color");

	if (enable_rl) {
		tree->add_constant_override("draw_relationship_lines", 1);
		tree->add_color_override("relationship_line_color", rl_color);
	} else
		tree->add_constant_override("draw_relationship_lines", 0);
}

void SceneTreeEditor::_bind_methods() {

	ClassDB::bind_method("_tree_changed", &SceneTreeEditor::_tree_changed);
	ClassDB::bind_method("_update_tree", &SceneTreeEditor::_update_tree);
	ClassDB::bind_method("_node_removed", &SceneTreeEditor::_node_removed);
	ClassDB::bind_method("_selected_changed", &SceneTreeEditor::_selected_changed);
	ClassDB::bind_method("_deselect_items", &SceneTreeEditor::_deselect_items);
	ClassDB::bind_method("_renamed", &SceneTreeEditor::_renamed);
	ClassDB::bind_method("_rename_node", &SceneTreeEditor::_rename_node);
	ClassDB::bind_method("_test_update_tree", &SceneTreeEditor::_test_update_tree);
	ClassDB::bind_method("_cell_multi_selected", &SceneTreeEditor::_cell_multi_selected);
	ClassDB::bind_method("_selection_changed", &SceneTreeEditor::_selection_changed);
	ClassDB::bind_method("_cell_button_pressed", &SceneTreeEditor::_cell_button_pressed);
	ClassDB::bind_method("_cell_collapsed", &SceneTreeEditor::_cell_collapsed);
	ClassDB::bind_method("_rmb_select", &SceneTreeEditor::_rmb_select);
	ClassDB::bind_method("_warning_changed", &SceneTreeEditor::_warning_changed);

	ClassDB::bind_method("_node_script_changed", &SceneTreeEditor::_node_script_changed);
	ClassDB::bind_method("_node_visibility_changed", &SceneTreeEditor::_node_visibility_changed);

	ClassDB::bind_method("_editor_settings_changed", &SceneTreeEditor::_editor_settings_changed);

	ClassDB::bind_method(D_METHOD("get_drag_data_fw"), &SceneTreeEditor::get_drag_data_fw);
	ClassDB::bind_method(D_METHOD("can_drop_data_fw"), &SceneTreeEditor::can_drop_data_fw);
	ClassDB::bind_method(D_METHOD("drop_data_fw"), &SceneTreeEditor::drop_data_fw);

	ClassDB::bind_method(D_METHOD("update_tree"), &SceneTreeEditor::update_tree);

	ADD_SIGNAL(MethodInfo("node_selected"));
	ADD_SIGNAL(MethodInfo("node_renamed"));
	ADD_SIGNAL(MethodInfo("node_prerename"));
	ADD_SIGNAL(MethodInfo("node_changed"));
	ADD_SIGNAL(MethodInfo("nodes_dragged"));
	ADD_SIGNAL(MethodInfo("nodes_rearranged", PropertyInfo(Variant::ARRAY, "paths"), PropertyInfo(Variant::NODE_PATH, "to_path"), PropertyInfo(Variant::INT, "type")));
	ADD_SIGNAL(MethodInfo("files_dropped", PropertyInfo(Variant::POOL_STRING_ARRAY, "files"), PropertyInfo(Variant::NODE_PATH, "to_path"), PropertyInfo(Variant::INT, "type")));
	ADD_SIGNAL(MethodInfo("script_dropped", PropertyInfo(Variant::STRING, "file"), PropertyInfo(Variant::NODE_PATH, "to_path")));
	ADD_SIGNAL(MethodInfo("rmb_pressed", PropertyInfo(Variant::VECTOR2, "position")));

	ADD_SIGNAL(MethodInfo("open"));
	ADD_SIGNAL(MethodInfo("open_script"));
}

SceneTreeEditor::SceneTreeEditor(bool p_label, bool p_can_rename, bool p_can_open_instance) {

	undo_redo = NULL;
	tree_dirty = true;
	selected = NULL;

	marked_selectable = false;
	marked_children_selectable = false;
	can_rename = p_can_rename;
	can_open_instance = p_can_open_instance;
	display_foreign = false;
	editor_selection = NULL;

	if (p_label) {
		Label *label = memnew(Label);
		label->set_position(Point2(10, 0));
		label->set_text(TTR("Scene Tree (Nodes):"));

		add_child(label);
	}

	tree = memnew(Tree);
	tree->set_anchor(MARGIN_RIGHT, ANCHOR_END);
	tree->set_anchor(MARGIN_BOTTOM, ANCHOR_END);
	tree->set_begin(Point2(0, p_label ? 18 : 0));
	tree->set_end(Point2(0, 0));
	tree->add_constant_override("button_margin", 0);

	add_child(tree);

	tree->set_drag_forwarding(this);
	if (p_can_rename) {
		tree->set_allow_rmb_select(true);
		tree->connect("item_rmb_selected", this, "_rmb_select");
		tree->connect("empty_tree_rmb_selected", this, "_rmb_select");
	}

	tree->connect("cell_selected", this, "_selected_changed");
	tree->connect("item_edited", this, "_renamed", varray(), CONNECT_DEFERRED);
	tree->connect("multi_selected", this, "_cell_multi_selected");
	tree->connect("button_pressed", this, "_cell_button_pressed");
	tree->connect("nothing_selected", this, "_deselect_items");
	//tree->connect("item_edited", this,"_renamed",Vector<Variant>(),true);

	error = memnew(AcceptDialog);
	add_child(error);

	warning = memnew(AcceptDialog);
	add_child(warning);
	warning->set_title(TTR("Node Configuration Warning!"));

	show_enabled_subscene = false;

	last_hash = 0;
	pending_test_update = false;
	updating_tree = false;
	blocked = 0;

	update_timer = memnew(Timer);
	update_timer->connect("timeout", this, "_update_tree");
	update_timer->set_one_shot(true);
	update_timer->set_wait_time(0.5);
	add_child(update_timer);

	script_types = memnew(List<StringName>);
	ClassDB::get_inheriters_from_class("Script", script_types);
}

SceneTreeEditor::~SceneTreeEditor() {

	memdelete(script_types);
}

/******** DIALOG *********/

void SceneTreeDialog::_notification(int p_what) {

	if (p_what == NOTIFICATION_ENTER_TREE) {
		connect("confirmed", this, "_select");
	}

	if (p_what == NOTIFICATION_EXIT_TREE) {
		disconnect("confirmed", this, "_select");
	}
	if (p_what == NOTIFICATION_DRAW) {

		RID ci = get_canvas_item();
		get_stylebox("panel", "PopupMenu")->draw(ci, Rect2(Point2(), get_size()));
	}

	if (p_what == NOTIFICATION_VISIBILITY_CHANGED && is_visible_in_tree()) {

		tree->update_tree();
	}
}

void SceneTreeDialog::_cancel() {

	hide();
}
void SceneTreeDialog::_select() {

	if (tree->get_selected()) {
		emit_signal("selected", tree->get_selected()->get_path());
		hide();
	}
}

void SceneTreeDialog::_bind_methods() {

	ClassDB::bind_method("_select", &SceneTreeDialog::_select);
	ClassDB::bind_method("_cancel", &SceneTreeDialog::_cancel);
	ADD_SIGNAL(MethodInfo("selected", PropertyInfo(Variant::NODE_PATH, "path")));
}

SceneTreeDialog::SceneTreeDialog() {

	set_title(TTR("Select a Node"));

	tree = memnew(SceneTreeEditor(false, false));
	add_child(tree);
	//set_child_rect(tree);

	tree->get_scene_tree()->connect("item_activated", this, "_select");
}

SceneTreeDialog::~SceneTreeDialog() {
}
