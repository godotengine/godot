/*************************************************************************/
/*  node_dock.cpp                                                        */
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

#include "node_dock.h"

#include "editor_node.h"
#include "editor_scale.h"

void NodeDock::_set_current(bool p_enable, int p_idx) {
	ERR_FAIL_INDEX(p_idx, mode_items.size());

	if (mode_items[p_idx].control->is_visible() == p_enable) {
		return;
	}

	if (p_idx != current_idx) {
		mode_items[current_idx].control->hide();
		mode_items[current_idx].button->set_pressed(false);
		mode_items[p_idx].control->show();
		mode_items[p_idx].button->set_pressed(true);
		current_idx = p_idx;

		set_node(current_node);
	} else if (!p_enable) {
		// Toggle event, not a switch
		mode_items[current_idx].button->set_pressed(true);
	}
}

void NodeDock::_bind_methods() {
	ClassDB::bind_method(D_METHOD("add_control", "control", "title"), &NodeDock::add_control);
	ClassDB::bind_method(D_METHOD("remove_control", "control"), &NodeDock::remove_control);
	ClassDB::bind_method(D_METHOD("show_control", "control"), &NodeDock::show_control);

	ADD_SIGNAL(MethodInfo("node_changed", PropertyInfo(Variant::OBJECT, "node", PROPERTY_HINT_RESOURCE_TYPE, "Node")));
}

void NodeDock::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE:
		case NOTIFICATION_THEME_CHANGED:
			connections_button->set_icon(get_theme_icon("Signals", "EditorIcons"));
			groups_button->set_icon(get_theme_icon("Groups", "EditorIcons"));
			break;
	}
}

NodeDock *NodeDock::singleton = nullptr;

void NodeDock::update_lists() {
	emit_signal("node_changed", current_node);
}

ConnectionsDock *NodeDock::get_connections_dock() {
	return connections;
}

GroupsEditor *NodeDock::get_group_editor() {
	return groups;
}

Button *NodeDock::add_control(Control *p_control, const String &p_title) {
	Button *mb = memnew(Button);
	mb->set_flat(true);
	mb->connect("toggled", callable_mp(this, &NodeDock::_set_current), varray(mode_items.size()));
	mb->set_text(p_title);
	mb->set_toggle_mode(true);
	mb->set_pressed(false);
	mb->set_h_size_flags(SIZE_EXPAND_FILL);
	mb->set_clip_text(true);

	mode_container->add_child(mb);
	add_child(p_control);
	p_control->hide();
	p_control->set_v_size_flags(SIZE_EXPAND_FILL);

	ModeItem item;
	item.button = mb;
	item.control = p_control;
	mode_items.push_back(item);

	return mb;
}

void NodeDock::remove_control(Control *p_control) {
	for (int i = 0; i < mode_items.size(); i++) {
		if (mode_items[i].control == p_control) {
			remove_child(mode_items[i].control);
			mode_container->remove_child(mode_items[i].button);
			memdelete(mode_items[i].button);
			mode_items.remove(i);

			if (mode_items.is_empty()) {
				// This is not a good state.
				current_idx = 0;
				set_node(nullptr);
			} else {
				_set_current(true, MIN(i, mode_items.size()));
			}
			break;
		}
	}

	for (int i = 0; i < mode_items.size(); i++) {
		mode_items[i].button->disconnect("toggled", callable_mp(this, &NodeDock::_set_current));
		mode_items[i].button->connect("toggled", callable_mp(this, &NodeDock::_set_current), varray(i));
	}
}

void NodeDock::show_control(Control *p_control) {
	for (int i = 0; i < mode_items.size(); i++) {
		if (mode_items[i].control == p_control) {
			_set_current(true, i);
			return;
		}
	}
	ERR_FAIL_MSG(vformat("Control is not a %s's tab.", get_class()));
}

void NodeDock::set_node(Node *p_node) {
	current_node = p_node;
	if (p_node) {
		mode_container->show();
		select_a_node->hide();

		ERR_FAIL_COND_MSG(!mode_items.size(), "No modes found.");
		mode_items[current_idx].control->show();
		mode_items[current_idx].button->set_pressed(true);
		emit_signal("node_changed", p_node);
	} else {
		if (mode_items.size() != 0) {
			mode_items[current_idx].button->set_pressed(false);
			mode_items[current_idx].control->hide();
		}
		mode_container->hide();
		select_a_node->show();
	}
}

NodeDock::NodeDock() {
	singleton = this;

	set_name("Node");
	mode_container = memnew(GridContainer);
	mode_container->set_columns(2);
	add_child(mode_container);
	mode_container->hide();

	ConnectionsDock *connections = memnew(ConnectionsDock(EditorNode::get_singleton()));
	connections->set_undoredo(EditorNode::get_undo_redo());
	connections_button = add_control(connections, TTR("Connections"));
	connect("node_changed", callable_mp(connections, &ConnectionsDock::set_node));

	GroupsEditor *groups = memnew(GroupsEditor);
	groups->set_undo_redo(EditorNode::get_undo_redo());
	groups_button = add_control(groups, TTR("Groups"));
	connect("node_changed", callable_mp(groups, &GroupsEditor::set_current));

	select_a_node = memnew(Label);
	select_a_node->set_text(TTR("Select a single node to edit its signals and groups."));
	select_a_node->set_custom_minimum_size(Size2(100 * EDSCALE, 0));
	select_a_node->set_v_size_flags(SIZE_EXPAND_FILL);
	select_a_node->set_valign(Label::VALIGN_CENTER);
	select_a_node->set_align(Label::ALIGN_CENTER);
	select_a_node->set_autowrap(true);
	add_child(select_a_node);
}
