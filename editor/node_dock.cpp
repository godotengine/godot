/*************************************************************************/
/*  node_dock.cpp                                                        */
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
#include "node_dock.h"

#include "editor_node.h"

void NodeDock::show_groups() {

	groups_button->set_pressed(true);
	connections_button->set_pressed(false);
	groups->show();
	connections->hide();
}

void NodeDock::show_connections() {

	groups_button->set_pressed(false);
	connections_button->set_pressed(true);
	groups->hide();
	connections->show();
}

void NodeDock::_bind_methods() {

	ClassDB::bind_method(D_METHOD("show_groups"), &NodeDock::show_groups);
	ClassDB::bind_method(D_METHOD("show_connections"), &NodeDock::show_connections);
}

void NodeDock::_notification(int p_what) {

	if (p_what == NOTIFICATION_ENTER_TREE) {
		connections_button->set_icon(get_icon("Connect", "EditorIcons"));
		groups_button->set_icon(get_icon("Groups", "EditorIcons"));
	}
}

NodeDock *NodeDock::singleton = NULL;

void NodeDock::set_node(Node *p_node) {

	connections->set_node(p_node);
	groups->set_current(p_node);

	if (p_node) {
		if (connections_button->is_pressed())
			connections->show();
		else
			groups->show();

		mode_hb->show();
		select_a_node->hide();
	} else {
		connections->hide();
		groups->hide();
		mode_hb->hide();
		select_a_node->show();
	}
}

NodeDock::NodeDock() {
	singleton = this;

	set_name(TTR("Node"));
	mode_hb = memnew(HBoxContainer);
	add_child(mode_hb);
	mode_hb->hide();

	connections_button = memnew(ToolButton);
	connections_button->set_text(TTR("Signals"));
	connections_button->set_toggle_mode(true);
	connections_button->set_pressed(true);
	connections_button->set_h_size_flags(SIZE_EXPAND_FILL);
	mode_hb->add_child(connections_button);
	connections_button->connect("pressed", this, "show_connections");

	groups_button = memnew(ToolButton);
	groups_button->set_text(TTR("Groups"));
	groups_button->set_toggle_mode(true);
	groups_button->set_pressed(false);
	groups_button->set_h_size_flags(SIZE_EXPAND_FILL);
	mode_hb->add_child(groups_button);
	groups_button->connect("pressed", this, "show_groups");

	connections = memnew(ConnectionsDock(EditorNode::get_singleton()));
	connections->set_undoredo(EditorNode::get_singleton()->get_undo_redo());
	add_child(connections);
	connections->set_v_size_flags(SIZE_EXPAND_FILL);
	connections->hide();

	groups = memnew(GroupsEditor);
	groups->set_undo_redo(EditorNode::get_singleton()->get_undo_redo());
	add_child(groups);
	groups->set_v_size_flags(SIZE_EXPAND_FILL);
	groups->hide();

	select_a_node = memnew(Label);
	select_a_node->set_text(TTR("Select a Node to edit Signals and Groups."));
	select_a_node->set_v_size_flags(SIZE_EXPAND_FILL);
	select_a_node->set_valign(Label::VALIGN_CENTER);
	select_a_node->set_align(Label::ALIGN_CENTER);
	select_a_node->set_autowrap(true);
	add_child(select_a_node);
}
