/**************************************************************************/
/*  animation_state_machine_editor.cpp                                    */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "animation_state_machine_editor.h"

#include "core/config/project_settings.h"
#include "core/input/input.h"
#include "core/io/resource_loader.h"
#include "core/math/geometry_2d.h"
#include "core/os/keyboard.h"
#include "editor/editor_file_dialog.h"
#include "editor/editor_node.h"
#include "editor/editor_scale.h"
#include "editor/editor_settings.h"
#include "editor/editor_undo_redo_manager.h"
#include "scene/animation/animation_blend_tree.h"
#include "scene/animation/animation_player.h"
#include "scene/gui/menu_button.h"
#include "scene/gui/option_button.h"
#include "scene/gui/panel.h"
#include "scene/gui/panel_container.h"
#include "scene/gui/separator.h"
#include "scene/gui/tree.h"
#include "scene/main/viewport.h"
#include "scene/main/window.h"

bool AnimationNodeStateMachineEditor::can_edit(const Ref<AnimationNode> &p_node) {
	Ref<AnimationNodeStateMachine> ansm = p_node;
	return ansm.is_valid();
}

void AnimationNodeStateMachineEditor::edit(const Ref<AnimationNode> &p_node) {
	state_machine = p_node;

	read_only = false;

	if (state_machine.is_valid()) {
		read_only = EditorNode::get_singleton()->is_resource_read_only(state_machine);

		selected_transition_from = StringName();
		selected_transition_to = StringName();
		selected_transition_index = -1;
		selected_multi_transition = TransitionLine();
		selected_node = StringName();
		selected_nodes.clear();
		_update_mode();
		_update_graph();
	}

	tool_create->set_disabled(read_only);
	tool_connect->set_disabled(read_only);
}

void AnimationNodeStateMachineEditor::_state_machine_gui_input(const Ref<InputEvent> &p_event) {
	AnimationTree *tree = AnimationTreeEditor::get_singleton()->get_animation_tree();
	if (!tree) {
		return;
	}

	Ref<AnimationNodeStateMachinePlayback> playback = tree->get(AnimationTreeEditor::get_singleton()->get_base_path() + "playback");
	if (playback.is_null()) {
		return;
	}

	Ref<InputEventKey> k = p_event;
	if (tool_select->is_pressed() && k.is_valid() && k->is_pressed() && k->get_keycode() == Key::KEY_DELETE && !k->is_echo()) {
		if (selected_node != StringName() || !selected_nodes.is_empty() || selected_transition_to != StringName() || selected_transition_from != StringName()) {
			if (!read_only) {
				_erase_selected();
			}
			accept_event();
		}
	}

	// Group selected nodes on a state machine
	if (tool_select->is_pressed() && k.is_valid() && k->is_pressed() && k->is_ctrl_pressed() && !k->is_shift_pressed() && k->get_keycode() == Key::G && !k->is_echo()) {
		_group_selected_nodes();
	}

	// Ungroup state machine
	if (tool_select->is_pressed() && k.is_valid() && k->is_pressed() && k->is_ctrl_pressed() && k->is_shift_pressed() && k->get_keycode() == Key::G && !k->is_echo()) {
		_ungroup_selected_nodes();
	}

	Ref<InputEventMouseButton> mb = p_event;

	// Add new node
	if (!read_only) {
		if (mb.is_valid() && mb->is_pressed() && !box_selecting && !connecting && ((tool_select->is_pressed() && mb->get_button_index() == MouseButton::RIGHT) || (tool_create->is_pressed() && mb->get_button_index() == MouseButton::LEFT))) {
			connecting_from = StringName();
			_open_menu(mb->get_position());
		}
	}

	// Select node or push a field inside
	if (mb.is_valid() && !mb->is_shift_pressed() && !mb->is_ctrl_pressed() && mb->is_pressed() && tool_select->is_pressed() && mb->get_button_index() == MouseButton::LEFT) {
		selected_transition_from = StringName();
		selected_transition_to = StringName();
		selected_transition_index = -1;
		selected_multi_transition = TransitionLine();
		selected_node = StringName();

		for (int i = node_rects.size() - 1; i >= 0; i--) { //inverse to draw order
			if (node_rects[i].play.has_point(mb->get_position())) { //edit name
				if (play_mode->get_selected() == 1 || !playback->is_playing()) {
					//start
					playback->start(node_rects[i].node_name);
				} else {
					//travel
					playback->travel(node_rects[i].node_name);
				}
				state_machine_draw->queue_redraw();
				return;
			}

			if (!read_only) {
				if (node_rects[i].name.has_point(mb->get_position()) && state_machine->can_edit_node(node_rects[i].node_name)) { // edit name
					Ref<StyleBox> line_sb = get_theme_stylebox(SNAME("normal"), SNAME("LineEdit"));

					Rect2 edit_rect = node_rects[i].name;
					edit_rect.position -= line_sb->get_offset();
					edit_rect.size += line_sb->get_minimum_size();

					name_edit_popup->set_position(state_machine_draw->get_screen_position() + edit_rect.position);
					name_edit_popup->set_size(edit_rect.size);
					name_edit->set_text(node_rects[i].node_name);
					name_edit_popup->popup();
					name_edit->grab_focus();
					name_edit->select_all();

					prev_name = node_rects[i].node_name;
					return;
				}
			}

			if (node_rects[i].edit.has_point(mb->get_position())) { //edit name
				call_deferred(SNAME("_open_editor"), node_rects[i].node_name);
				return;
			}

			if (node_rects[i].node.has_point(mb->get_position())) { //select node since nothing else was selected
				selected_node = node_rects[i].node_name;

				if (!selected_nodes.has(selected_node)) {
					selected_nodes.clear();
				}

				selected_nodes.insert(selected_node);

				Ref<AnimationNode> anode = state_machine->get_node(selected_node);
				EditorNode::get_singleton()->push_item(anode.ptr(), "", true);
				state_machine_draw->queue_redraw();
				dragging_selected_attempt = true;
				dragging_selected = false;
				drag_from = mb->get_position();
				snap_x = StringName();
				snap_y = StringName();
				_update_mode();
				return;
			}
		}

		//test the lines now
		int closest = -1;
		float closest_d = 1e20;
		for (int i = 0; i < transition_lines.size(); i++) {
			Vector2 s[2] = {
				transition_lines[i].from,
				transition_lines[i].to
			};
			Vector2 cpoint = Geometry2D::get_closest_point_to_segment(mb->get_position(), s);
			float d = cpoint.distance_to(mb->get_position());
			if (d > transition_lines[i].width) {
				continue;
			}

			if (d < closest_d) {
				closest = i;
				closest_d = d;
			}
		}

		if (closest >= 0) {
			selected_transition_from = transition_lines[closest].from_node;
			selected_transition_to = transition_lines[closest].to_node;
			selected_transition_index = closest;

			Ref<AnimationNodeStateMachineTransition> tr = state_machine->get_transition(closest);
			EditorNode::get_singleton()->push_item(tr.ptr(), "", true);

			if (!transition_lines[closest].multi_transitions.is_empty()) {
				selected_transition_index = -1;
				selected_multi_transition = transition_lines[closest];

				Ref<EditorAnimationMultiTransitionEdit> multi;
				multi.instantiate();
				multi->add_transition(selected_transition_from, selected_transition_to, tr);

				for (int i = 0; i < transition_lines[closest].multi_transitions.size(); i++) {
					int index = transition_lines[closest].multi_transitions[i].transition_index;

					Ref<AnimationNodeStateMachineTransition> transition = state_machine->get_transition(index);
					StringName from = transition_lines[closest].multi_transitions[i].from_node;
					StringName to = transition_lines[closest].multi_transitions[i].to_node;

					multi->add_transition(from, to, transition);
				}
				EditorNode::get_singleton()->push_item(multi.ptr(), "", true);
			}
		}

		state_machine_draw->queue_redraw();
		_update_mode();
	}

	// End moving node
	if (mb.is_valid() && dragging_selected_attempt && mb->get_button_index() == MouseButton::LEFT && !mb->is_pressed()) {
		if (dragging_selected) {
			Ref<AnimationNode> an = state_machine->get_node(selected_node);
			updating = true;

			EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
			undo_redo->create_action(TTR("Move Node"));

			for (int i = 0; i < node_rects.size(); i++) {
				if (!selected_nodes.has(node_rects[i].node_name)) {
					continue;
				}

				undo_redo->add_do_method(state_machine.ptr(), "set_node_position", node_rects[i].node_name, state_machine->get_node_position(node_rects[i].node_name) + drag_ofs / EDSCALE);
				undo_redo->add_undo_method(state_machine.ptr(), "set_node_position", node_rects[i].node_name, state_machine->get_node_position(node_rects[i].node_name));
			}

			undo_redo->add_do_method(this, "_update_graph");
			undo_redo->add_undo_method(this, "_update_graph");
			undo_redo->commit_action();
			updating = false;
		}
		snap_x = StringName();
		snap_y = StringName();

		dragging_selected_attempt = false;
		dragging_selected = false;
		state_machine_draw->queue_redraw();
	}

	// Connect nodes
	if (mb.is_valid() && ((tool_select->is_pressed() && mb->is_shift_pressed()) || tool_connect->is_pressed()) && mb->get_button_index() == MouseButton::LEFT && mb->is_pressed()) {
		for (int i = node_rects.size() - 1; i >= 0; i--) { //inverse to draw order
			if (node_rects[i].node.has_point(mb->get_position())) { //select node since nothing else was selected
				connecting = true;
				connection_follows_cursor = true;
				connecting_from = node_rects[i].node_name;
				connecting_to = mb->get_position();
				connecting_to_node = StringName();
				return;
			}
		}
	}

	// End connecting nodes
	if (mb.is_valid() && connecting && mb->get_button_index() == MouseButton::LEFT && !mb->is_pressed()) {
		if (connecting_to_node != StringName()) {
			Ref<AnimationNode> node = state_machine->get_node(connecting_to_node);
			Ref<AnimationNodeStateMachine> anodesm = node;
			Ref<AnimationNodeEndState> end_node = node;

			if (state_machine->has_transition(connecting_from, connecting_to_node) && state_machine->can_edit_node(connecting_to_node) && !anodesm.is_valid()) {
				EditorNode::get_singleton()->show_warning(TTR("Transition exists!"));
				connecting = false;
			} else {
				if (anodesm.is_valid() || end_node.is_valid()) {
					_open_connect_menu(mb->get_position());
				} else {
					_add_transition();
				}
			}
		} else {
			_open_menu(mb->get_position());
		}
		connecting_to_node = StringName();
		connection_follows_cursor = false;
		state_machine_draw->queue_redraw();
	}

	// Start box selecting
	if (mb.is_valid() && mb->is_pressed() && mb->get_button_index() == MouseButton::LEFT && tool_select->is_pressed()) {
		box_selecting = true;
		box_selecting_from = box_selecting_to = state_machine_draw->get_local_mouse_position();
		box_selecting_rect = Rect2(MIN(box_selecting_from.x, box_selecting_to.x),
				MIN(box_selecting_from.y, box_selecting_to.y),
				ABS(box_selecting_from.x - box_selecting_to.x),
				ABS(box_selecting_from.y - box_selecting_to.y));

		if (mb->is_ctrl_pressed() || mb->is_shift_pressed()) {
			previous_selected = selected_nodes;
		} else {
			selected_nodes.clear();
			previous_selected.clear();
		}
	}

	// End box selecting
	if (mb.is_valid() && mb->get_button_index() == MouseButton::LEFT && !mb->is_pressed() && box_selecting) {
		box_selecting = false;
		state_machine_draw->queue_redraw();
		_update_mode();
	}

	Ref<InputEventMouseMotion> mm = p_event;

	// Pan window
	if (mm.is_valid() && mm->get_button_mask().has_flag(MouseButtonMask::MIDDLE)) {
		h_scroll->set_value(h_scroll->get_value() - mm->get_relative().x);
		v_scroll->set_value(v_scroll->get_value() - mm->get_relative().y);
	}

	// Move mouse while connecting
	if (mm.is_valid() && connecting && connection_follows_cursor && !read_only) {
		connecting_to = mm->get_position();
		connecting_to_node = StringName();
		state_machine_draw->queue_redraw();

		for (int i = node_rects.size() - 1; i >= 0; i--) { //inverse to draw order
			if (node_rects[i].node_name != connecting_from && node_rects[i].node.has_point(connecting_to)) { //select node since nothing else was selected
				connecting_to_node = node_rects[i].node_name;
				return;
			}
		}
	}

	// Move mouse while moving a node
	if (mm.is_valid() && dragging_selected_attempt && !read_only) {
		dragging_selected = true;
		drag_ofs = mm->get_position() - drag_from;
		snap_x = StringName();
		snap_y = StringName();
		{
			//snap
			Vector2 cpos = state_machine->get_node_position(selected_node) + drag_ofs / EDSCALE;
			List<StringName> nodes;
			state_machine->get_node_list(&nodes);

			float best_d_x = 1e20;
			float best_d_y = 1e20;

			for (const StringName &E : nodes) {
				if (E == selected_node) {
					continue;
				}
				Vector2 npos = state_machine->get_node_position(E);

				float d_x = ABS(npos.x - cpos.x);
				if (d_x < MIN(5, best_d_x)) {
					drag_ofs.x -= cpos.x - npos.x;
					best_d_x = d_x;
					snap_x = E;
				}

				float d_y = ABS(npos.y - cpos.y);
				if (d_y < MIN(5, best_d_y)) {
					drag_ofs.y -= cpos.y - npos.y;
					best_d_y = d_y;
					snap_y = E;
				}
			}
		}

		state_machine_draw->queue_redraw();
	}

	// Move mouse while moving box select
	if (mm.is_valid() && box_selecting) {
		box_selecting_to = state_machine_draw->get_local_mouse_position();

		box_selecting_rect = Rect2(MIN(box_selecting_from.x, box_selecting_to.x),
				MIN(box_selecting_from.y, box_selecting_to.y),
				ABS(box_selecting_from.x - box_selecting_to.x),
				ABS(box_selecting_from.y - box_selecting_to.y));

		for (int i = 0; i < node_rects.size(); i++) {
			bool in_box = node_rects[i].node.intersects(box_selecting_rect);

			if (in_box) {
				if (previous_selected.has(node_rects[i].node_name)) {
					selected_nodes.erase(node_rects[i].node_name);
				} else {
					selected_nodes.insert(node_rects[i].node_name);
				}
			} else {
				if (previous_selected.has(node_rects[i].node_name)) {
					selected_nodes.insert(node_rects[i].node_name);
				} else {
					selected_nodes.erase(node_rects[i].node_name);
				}
			}
		}

		state_machine_draw->queue_redraw();
	}

	if (mm.is_valid()) {
		state_machine_draw->grab_focus();

		String new_over_node;
		int new_over_node_what = -1;
		if (tool_select->is_pressed()) {
			for (int i = node_rects.size() - 1; i >= 0; i--) { // Inverse to draw order.

				if (!state_machine->can_edit_node(node_rects[i].node_name)) {
					continue; // start/end node can't be edited
				}

				if (node_rects[i].node.has_point(mm->get_position())) {
					new_over_node = node_rects[i].node_name;
					if (node_rects[i].play.has_point(mm->get_position())) {
						new_over_node_what = 0;
					} else if (node_rects[i].edit.has_point(mm->get_position())) {
						new_over_node_what = 1;
					}
					break;
				}
			}
		}

		if (new_over_node != over_node || new_over_node_what != over_node_what) {
			over_node = new_over_node;
			over_node_what = new_over_node_what;
			state_machine_draw->queue_redraw();
		}

		// set tooltip for transition
		if (tool_select->is_pressed()) {
			int closest = -1;
			float closest_d = 1e20;
			for (int i = 0; i < transition_lines.size(); i++) {
				Vector2 s[2] = {
					transition_lines[i].from,
					transition_lines[i].to
				};
				Vector2 cpoint = Geometry2D::get_closest_point_to_segment(mm->get_position(), s);
				float d = cpoint.distance_to(mm->get_position());
				if (d > transition_lines[i].width) {
					continue;
				}

				if (d < closest_d) {
					closest = i;
					closest_d = d;
				}
			}

			if (closest >= 0) {
				String from = String(transition_lines[closest].from_node);
				String to = String(transition_lines[closest].to_node);
				String tooltip = from + " -> " + to;

				for (int i = 0; i < transition_lines[closest].multi_transitions.size(); i++) {
					from = String(transition_lines[closest].multi_transitions[i].from_node);
					to = String(transition_lines[closest].multi_transitions[i].to_node);
					tooltip += "\n" + from + " -> " + to;
				}
				state_machine_draw->set_tooltip_text(tooltip);
			} else {
				state_machine_draw->set_tooltip_text("");
			}
		}
	}

	Ref<InputEventPanGesture> pan_gesture = p_event;
	if (pan_gesture.is_valid()) {
		h_scroll->set_value(h_scroll->get_value() + h_scroll->get_page() * pan_gesture->get_delta().x / 8);
		v_scroll->set_value(v_scroll->get_value() + v_scroll->get_page() * pan_gesture->get_delta().y / 8);
	}
}

Control::CursorShape AnimationNodeStateMachineEditor::get_cursor_shape(const Point2 &p_pos) const {
	Control::CursorShape cursor_shape = get_default_cursor_shape();
	if (!read_only) {
		// Put ibeam (text cursor) over names to make it clearer that they are editable.
		Transform2D xform = panel->get_transform() * state_machine_draw->get_transform();
		Point2 pos = xform.xform_inv(p_pos);

		for (int i = node_rects.size() - 1; i >= 0; i--) { // Inverse to draw order.
			if (node_rects[i].node.has_point(pos)) {
				if (node_rects[i].name.has_point(pos)) {
					if (state_machine->can_edit_node(node_rects[i].node_name)) {
						cursor_shape = Control::CURSOR_IBEAM;
					}
				}
				break;
			}
		}
	}
	return cursor_shape;
}

void AnimationNodeStateMachineEditor::_group_selected_nodes() {
	if (!selected_nodes.is_empty()) {
		if (selected_nodes.size() == 1 && (*selected_nodes.begin() == state_machine->start_node || *selected_nodes.begin() == state_machine->end_node))
			return;

		Ref<AnimationNodeStateMachine> group_sm = memnew(AnimationNodeStateMachine);
		Vector2 group_position;

		Vector<NodeUR> nodes_ur;
		Vector<TransitionUR> transitions_ur;

		int base = 1;
		String base_name = group_sm->get_caption();
		String group_name = base_name;

		while (state_machine->has_node(group_name) && !selected_nodes.has(group_name)) {
			base++;
			group_name = base_name + " " + itos(base);
		}

		updating = true;
		EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
		undo_redo->create_action("Group");

		// Move selected nodes to the new state machine
		for (const StringName &E : selected_nodes) {
			if (!state_machine->can_edit_node(E)) {
				continue;
			}

			Ref<AnimationNode> node = state_machine->get_node(E);
			Vector2 node_position = state_machine->get_node_position(E);
			group_position += node_position;

			NodeUR new_node;
			new_node.name = E;
			new_node.node = node;
			new_node.position = node_position;

			nodes_ur.push_back(new_node);
		}

		// Add the transitions to the new state machine
		for (int i = 0; i < state_machine->get_transition_count(); i++) {
			String from = state_machine->get_transition_from(i);
			String to = state_machine->get_transition_to(i);

			String local_from = from.get_slicec('/', 0);
			String local_to = to.get_slicec('/', 0);

			String old_from = from;
			String old_to = to;

			bool from_selected = false;
			bool to_selected = false;

			if (selected_nodes.has(local_from) && local_from != state_machine->start_node) {
				from_selected = true;
			}
			if (selected_nodes.has(local_to) && local_to != state_machine->end_node) {
				to_selected = true;
			}
			if (!from_selected && !to_selected) {
				continue;
			}

			Ref<AnimationNodeStateMachineTransition> tr = state_machine->get_transition(i);

			if (!from_selected) {
				from = "../" + old_from;
			}
			if (!to_selected) {
				to = "../" + old_to;
			}

			TransitionUR new_tr;
			new_tr.new_from = from;
			new_tr.new_to = to;
			new_tr.old_from = old_from;
			new_tr.old_to = old_to;
			new_tr.transition = tr;

			transitions_ur.push_back(new_tr);
		}

		for (int i = 0; i < nodes_ur.size(); i++) {
			undo_redo->add_do_method(state_machine.ptr(), "remove_node", nodes_ur[i].name);
			undo_redo->add_undo_method(group_sm.ptr(), "remove_node", nodes_ur[i].name);
		}

		undo_redo->add_do_method(state_machine.ptr(), "add_node", group_name, group_sm, group_position / nodes_ur.size());
		undo_redo->add_undo_method(state_machine.ptr(), "remove_node", group_name);

		for (int i = 0; i < nodes_ur.size(); i++) {
			undo_redo->add_do_method(group_sm.ptr(), "add_node", nodes_ur[i].name, nodes_ur[i].node, nodes_ur[i].position);
			undo_redo->add_undo_method(state_machine.ptr(), "add_node", nodes_ur[i].name, nodes_ur[i].node, nodes_ur[i].position);
		}

		for (int i = 0; i < transitions_ur.size(); i++) {
			undo_redo->add_do_method(group_sm.ptr(), "add_transition", transitions_ur[i].new_from, transitions_ur[i].new_to, transitions_ur[i].transition);
			undo_redo->add_undo_method(state_machine.ptr(), "add_transition", transitions_ur[i].old_from, transitions_ur[i].old_to, transitions_ur[i].transition);
		}

		undo_redo->add_do_method(this, "_update_graph");
		undo_redo->add_undo_method(this, "_update_graph");
		undo_redo->commit_action();
		updating = false;

		selected_nodes.clear();
		selected_nodes.insert(group_name);
		state_machine_draw->queue_redraw();
		accept_event();
		_update_mode();
	}
}

void AnimationNodeStateMachineEditor::_ungroup_selected_nodes() {
	bool find = false;
	HashSet<StringName> new_selected_nodes;

	for (const StringName &E : selected_nodes) {
		Ref<AnimationNodeStateMachine> group_sm = state_machine->get_node(E);

		if (group_sm.is_valid()) {
			find = true;

			Vector2 group_position = state_machine->get_node_position(E);
			StringName group_name = E;

			List<AnimationNode::ChildNode> nodes;
			group_sm->get_child_nodes(&nodes);

			Vector<NodeUR> nodes_ur;
			Vector<TransitionUR> transitions_ur;

			updating = true;
			EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
			undo_redo->create_action("Ungroup");

			// Move all child nodes to current state machine
			for (int i = 0; i < nodes.size(); i++) {
				if (!group_sm->can_edit_node(nodes[i].name)) {
					continue;
				}

				Vector2 node_position = group_sm->get_node_position(nodes[i].name);

				NodeUR new_node;
				new_node.name = nodes[i].name;
				new_node.position = node_position;
				new_node.node = nodes[i].node;

				nodes_ur.push_back(new_node);
			}

			for (int i = 0; i < group_sm->get_transition_count(); i++) {
				String from = group_sm->get_transition_from(i);
				String to = group_sm->get_transition_to(i);
				Ref<AnimationNodeStateMachineTransition> tr = group_sm->get_transition(i);

				TransitionUR new_tr;
				new_tr.new_from = from.replace_first("../", "");
				new_tr.new_to = to.replace_first("../", "");
				new_tr.old_from = from;
				new_tr.old_to = to;
				new_tr.transition = tr;

				transitions_ur.push_back(new_tr);
			}

			for (int i = 0; i < nodes_ur.size(); i++) {
				undo_redo->add_do_method(group_sm.ptr(), "remove_node", nodes_ur[i].name);
				undo_redo->add_undo_method(state_machine.ptr(), "remove_node", nodes_ur[i].name);
			}

			undo_redo->add_do_method(state_machine.ptr(), "remove_node", group_name);
			undo_redo->add_undo_method(state_machine.ptr(), "add_node", group_name, group_sm, group_position);

			for (int i = 0; i < nodes_ur.size(); i++) {
				new_selected_nodes.insert(nodes_ur[i].name);
				undo_redo->add_do_method(state_machine.ptr(), "add_node", nodes_ur[i].name, nodes_ur[i].node, nodes_ur[i].position);
				undo_redo->add_undo_method(group_sm.ptr(), "add_node", nodes_ur[i].name, nodes_ur[i].node, nodes_ur[i].position);
			}

			for (int i = 0; i < transitions_ur.size(); i++) {
				if (transitions_ur[i].old_from != state_machine->start_node && transitions_ur[i].old_to != state_machine->end_node) {
					undo_redo->add_do_method(state_machine.ptr(), "add_transition", transitions_ur[i].new_from, transitions_ur[i].new_to, transitions_ur[i].transition);
				}

				undo_redo->add_undo_method(group_sm.ptr(), "add_transition", transitions_ur[i].old_from, transitions_ur[i].old_to, transitions_ur[i].transition);
			}

			for (int i = 0; i < state_machine->get_transition_count(); i++) {
				String from = state_machine->get_transition_from(i);
				String to = state_machine->get_transition_to(i);
				Ref<AnimationNodeStateMachineTransition> tr = state_machine->get_transition(i);

				if (from == group_name || to == group_name) {
					undo_redo->add_undo_method(state_machine.ptr(), "add_transition", from, to, tr);
				}
			}

			undo_redo->add_do_method(this, "_update_graph");
			undo_redo->add_undo_method(this, "_update_graph");
			undo_redo->commit_action();
			updating = false;
		}
	}

	if (find) {
		selected_nodes = new_selected_nodes;
		selected_node = StringName();
		state_machine_draw->queue_redraw();
		accept_event();
		_update_mode();
	}
}

void AnimationNodeStateMachineEditor::_open_menu(const Vector2 &p_position) {
	AnimationTree *tree = AnimationTreeEditor::get_singleton()->get_animation_tree();
	if (!tree) {
		return;
	}

	menu->clear();
	animations_menu->clear();
	animations_to_add.clear();
	List<StringName> classes;
	classes.sort_custom<StringName::AlphCompare>();

	ClassDB::get_inheriters_from_class("AnimationRootNode", &classes);
	menu->add_submenu_item(TTR("Add Animation"), "animations");

	if (tree->has_node(tree->get_animation_player())) {
		AnimationPlayer *ap = Object::cast_to<AnimationPlayer>(tree->get_node(tree->get_animation_player()));
		if (ap) {
			List<StringName> names;
			ap->get_animation_list(&names);
			for (List<StringName>::Element *E = names.front(); E; E = E->next()) {
				animations_menu->add_icon_item(get_theme_icon("Animation", "EditorIcons"), E->get());
				animations_to_add.push_back(E->get());
			}
		}
	}

	for (List<StringName>::Element *E = classes.front(); E; E = E->next()) {
		String name = String(E->get()).replace_first("AnimationNode", "");
		if (name == "Animation" || name == "StartState" || name == "EndState") {
			continue; // nope
		}
		int idx = menu->get_item_count();
		menu->add_item(vformat(TTR("Add %s"), name), idx);
		menu->set_item_metadata(idx, E->get());
	}
	Ref<AnimationNode> clipb = EditorSettings::get_singleton()->get_resource_clipboard();

	if (clipb.is_valid()) {
		menu->add_separator();
		menu->add_item(TTR("Paste"), MENU_PASTE);
	}
	menu->add_separator();
	menu->add_item(TTR("Load..."), MENU_LOAD_FILE);

	menu->set_position(state_machine_draw->get_screen_transform().xform(p_position));
	menu->popup();
	add_node_pos = p_position / EDSCALE + state_machine->get_graph_offset();
}

void AnimationNodeStateMachineEditor::_open_connect_menu(const Vector2 &p_position) {
	ERR_FAIL_COND(connecting_to_node == StringName());

	Ref<AnimationNode> node = state_machine->get_node(connecting_to_node);
	Ref<AnimationNodeStateMachine> anodesm = node;
	Ref<AnimationNodeEndState> end_node = node;
	ERR_FAIL_COND(!anodesm.is_valid() && !end_node.is_valid());

	connect_menu->clear();
	state_machine_menu->clear();
	end_menu->clear();
	nodes_to_connect.clear();

	for (int i = connect_menu->get_child_count() - 1; i >= 0; i--) {
		Node *child = connect_menu->get_child(i);

		if (child->is_class("PopupMenu")) {
			connect_menu->remove_child(child);
		}
	}

	connect_menu->reset_size();
	state_machine_menu->reset_size();
	end_menu->reset_size();

	if (anodesm.is_valid()) {
		_create_submenu(connect_menu, anodesm, connecting_to_node, connecting_to_node);
	} else {
		_create_submenu(connect_menu, state_machine, connecting_to_node, connecting_to_node, true);
	}

	connect_menu->add_submenu_item(TTR("To") + " Animation", connecting_to_node);

	if (state_machine_menu->get_item_count() > 0 || !end_node.is_valid()) {
		connect_menu->add_submenu_item(TTR("To") + " StateMachine", "state_machines");
		connect_menu->add_child(state_machine_menu);
	}

	if (end_node.is_valid()) {
		connect_menu->add_submenu_item(TTR("To") + " End", "end_nodes");
		connect_menu->add_child(end_menu);
	} else {
		state_machine_menu->add_item(connecting_to_node, nodes_to_connect.size());
	}

	nodes_to_connect.push_back(connecting_to_node);

	if (nodes_to_connect.size() == 1) {
		_add_transition();
		return;
	}

	connect_menu->set_position(state_machine_draw->get_screen_transform().xform(p_position));
	connect_menu->popup();
}

bool AnimationNodeStateMachineEditor::_create_submenu(PopupMenu *p_menu, Ref<AnimationNodeStateMachine> p_nodesm, const StringName &p_name, const StringName &p_path, bool from_root, Vector<Ref<AnimationNodeStateMachine>> p_parents) {
	String prev_path;
	Vector<Ref<AnimationNodeStateMachine>> parents = p_parents;

	if (from_root && p_nodesm->get_prev_state_machine() == nullptr) {
		return false;
	}

	if (from_root) {
		AnimationNodeStateMachine *prev = p_nodesm->get_prev_state_machine();

		while (prev != nullptr) {
			parents.push_back(prev);
			p_nodesm = Ref<AnimationNodeStateMachine>(prev);
			prev_path += "../";
			prev = prev->get_prev_state_machine();
		}
		end_menu->add_item("Root", nodes_to_connect.size());
		nodes_to_connect.push_back(prev_path + state_machine->end_node);
		prev_path.remove_at(prev_path.size() - 1);
	}

	List<StringName> nodes;
	p_nodesm->get_node_list(&nodes);

	PopupMenu *nodes_menu = memnew(PopupMenu);
	nodes_menu->set_name(p_name);
	nodes_menu->connect("id_pressed", callable_mp(this, &AnimationNodeStateMachineEditor::_connect_to));
	p_menu->add_child(nodes_menu);

	bool node_added = false;
	for (const StringName &E : nodes) {
		if (p_nodesm->can_edit_node(E)) {
			Ref<AnimationNodeStateMachine> ansm = p_nodesm->get_node(E);

			String path;
			if (from_root) {
				path = prev_path + "/" + E;
			} else {
				path = String(p_path) + "/" + E;
			}

			if (ansm == state_machine) {
				end_menu->add_item(E, nodes_to_connect.size());
				nodes_to_connect.push_back(state_machine->end_node);
				continue;
			}

			if (ansm.is_valid()) {
				bool parent_found = false;

				for (int i = 0; i < parents.size(); i++) {
					if (parents[i] == ansm) {
						path = path.replace_first("/../" + E, "");
						parent_found = true;
						break;
					}
				}

				if (parent_found) {
					end_menu->add_item(E, nodes_to_connect.size());
					nodes_to_connect.push_back(path + "/" + state_machine->end_node);
				} else {
					state_machine_menu->add_item(E, nodes_to_connect.size());
					nodes_to_connect.push_back(path);
				}

				if (_create_submenu(nodes_menu, ansm, E, path, false, parents)) {
					nodes_menu->add_submenu_item(E, E);
					node_added = true;
				}
			} else {
				nodes_menu->add_item(E, nodes_to_connect.size());
				nodes_to_connect.push_back(path);
				node_added = true;
			}
		}
	}

	return node_added;
}

void AnimationNodeStateMachineEditor::_stop_connecting() {
	connecting = false;
	state_machine_draw->queue_redraw();
}

void AnimationNodeStateMachineEditor::_delete_selected() {
	TreeItem *item = delete_tree->get_next_selected(nullptr);
	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	while (item) {
		if (!updating) {
			updating = true;
			selected_multi_transition = TransitionLine();
			undo_redo->create_action("Transition(s) Removed");
		}

		Vector<String> path = item->get_text(0).split(" -> ");

		selected_transition_from = path[0];
		selected_transition_to = path[1];
		_erase_selected(true);

		item = delete_tree->get_next_selected(item);
	}

	if (updating) {
		undo_redo->commit_action();
		updating = false;
	}
}

void AnimationNodeStateMachineEditor::_delete_all() {
	Vector<TransitionLine> multi_transitions = selected_multi_transition.multi_transitions;
	selected_multi_transition = TransitionLine();

	updating = true;
	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action("Transition(s) Removed");
	_erase_selected(true);
	for (int i = 0; i < multi_transitions.size(); i++) {
		selected_transition_from = multi_transitions[i].from_node;
		selected_transition_to = multi_transitions[i].to_node;
		_erase_selected(true);
	}
	undo_redo->commit_action();
	updating = false;

	delete_window->hide();
}

void AnimationNodeStateMachineEditor::_delete_tree_draw() {
	TreeItem *item = delete_tree->get_next_selected(nullptr);
	while (item) {
		delete_window->get_cancel_button()->set_disabled(false);
		return;
	}
	delete_window->get_cancel_button()->set_disabled(true);
}

void AnimationNodeStateMachineEditor::_file_opened(const String &p_file) {
	file_loaded = ResourceLoader::load(p_file);
	if (file_loaded.is_valid()) {
		_add_menu_type(MENU_LOAD_FILE_CONFIRM);
	} else {
		EditorNode::get_singleton()->show_warning(TTR("This type of node can't be used. Only animation nodes are allowed."));
	}
}

void AnimationNodeStateMachineEditor::_add_menu_type(int p_index) {
	String base_name;
	Ref<AnimationRootNode> node;

	if (p_index == MENU_LOAD_FILE) {
		open_file->clear_filters();
		List<String> filters;
		ResourceLoader::get_recognized_extensions_for_type("AnimationRootNode", &filters);
		for (const String &E : filters) {
			open_file->add_filter("*." + E);
		}
		open_file->popup_file_dialog();
		return;
	} else if (p_index == MENU_LOAD_FILE_CONFIRM) {
		node = file_loaded;
		file_loaded.unref();
	} else if (p_index == MENU_PASTE) {
		node = EditorSettings::get_singleton()->get_resource_clipboard();

	} else {
		String type = menu->get_item_metadata(p_index);

		Object *obj = ClassDB::instantiate(type);
		ERR_FAIL_COND(!obj);
		AnimationNode *an = Object::cast_to<AnimationNode>(obj);
		ERR_FAIL_COND(!an);

		node = Ref<AnimationNode>(an);
		base_name = type.replace_first("AnimationNode", "");
	}

	if (!node.is_valid()) {
		EditorNode::get_singleton()->show_warning(TTR("This type of node can't be used. Only root nodes are allowed."));
		return;
	}

	if (base_name.is_empty()) {
		base_name = node->get_class().replace_first("AnimationNode", "");
	}

	int base = 1;
	String name = base_name;
	while (state_machine->has_node(name)) {
		base++;
		name = base_name + " " + itos(base);
	}

	updating = true;
	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(TTR("Add Node and Transition"));
	undo_redo->add_do_method(state_machine.ptr(), "add_node", name, node, add_node_pos);
	undo_redo->add_undo_method(state_machine.ptr(), "remove_node", name);
	connecting_to_node = name;
	_add_transition(true);
	undo_redo->commit_action();
	updating = false;

	state_machine_draw->queue_redraw();
}

void AnimationNodeStateMachineEditor::_add_animation_type(int p_index) {
	Ref<AnimationNodeAnimation> anim;
	anim.instantiate();

	anim->set_animation(animations_to_add[p_index]);

	String base_name = animations_to_add[p_index].validate_node_name();
	int base = 1;
	String name = base_name;
	while (state_machine->has_node(name)) {
		base++;
		name = base_name + " " + itos(base);
	}

	updating = true;
	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(TTR("Add Node and Transition"));
	undo_redo->add_do_method(state_machine.ptr(), "add_node", name, anim, add_node_pos);
	undo_redo->add_undo_method(state_machine.ptr(), "remove_node", name);
	connecting_to_node = name;
	_add_transition(true);
	undo_redo->commit_action();
	updating = false;

	state_machine_draw->queue_redraw();
}

void AnimationNodeStateMachineEditor::_connect_to(int p_index) {
	connecting_to_node = nodes_to_connect[p_index];
	_add_transition();
}

void AnimationNodeStateMachineEditor::_add_transition(const bool p_nested_action) {
	if (connecting_from != StringName() && connecting_to_node != StringName()) {
		if (state_machine->has_transition(connecting_from, connecting_to_node)) {
			EditorNode::get_singleton()->show_warning("Transition exists!");
			connecting = false;
			return;
		}

		Ref<AnimationNodeStateMachineTransition> tr;
		tr.instantiate();
		tr->set_advance_mode(auto_advance->is_pressed() ? AnimationNodeStateMachineTransition::AdvanceMode::ADVANCE_MODE_AUTO : AnimationNodeStateMachineTransition::AdvanceMode::ADVANCE_MODE_ENABLED);
		tr->set_switch_mode(AnimationNodeStateMachineTransition::SwitchMode(switch_mode->get_selected()));

		EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
		if (!p_nested_action) {
			updating = true;
			undo_redo->create_action(TTR("Add Transition"));
		}

		undo_redo->add_do_method(state_machine.ptr(), "add_transition", connecting_from, connecting_to_node, tr);
		undo_redo->add_undo_method(state_machine.ptr(), "remove_transition", connecting_from, connecting_to_node);
		undo_redo->add_do_method(this, "_update_graph");
		undo_redo->add_undo_method(this, "_update_graph");

		if (!p_nested_action) {
			undo_redo->commit_action();
			updating = false;
		}

		selected_transition_from = connecting_from;
		selected_transition_to = connecting_to_node;
		selected_transition_index = transition_lines.size();

		EditorNode::get_singleton()->push_item(tr.ptr(), "", true);
		_update_mode();
	}

	connecting = false;
}

void AnimationNodeStateMachineEditor::_connection_draw(const Vector2 &p_from, const Vector2 &p_to, AnimationNodeStateMachineTransition::SwitchMode p_mode, bool p_enabled, bool p_selected, bool p_travel, float p_fade_ratio, bool p_auto_advance, bool p_multi_transitions) {
	Color linecolor = get_theme_color(SNAME("font_color"), SNAME("Label"));
	Color icon_color(1, 1, 1);
	Color accent = get_theme_color(SNAME("accent_color"), SNAME("Editor"));

	if (!p_enabled) {
		linecolor.a *= 0.2;
		icon_color.a *= 0.2;
		accent.a *= 0.6;
	}

	const Ref<Texture2D> icons[] = {
		get_theme_icon(SNAME("TransitionImmediateBig"), SNAME("EditorIcons")),
		get_theme_icon(SNAME("TransitionSyncBig"), SNAME("EditorIcons")),
		get_theme_icon(SNAME("TransitionEndBig"), SNAME("EditorIcons")),
		get_theme_icon(SNAME("TransitionImmediateAutoBig"), SNAME("EditorIcons")),
		get_theme_icon(SNAME("TransitionSyncAutoBig"), SNAME("EditorIcons")),
		get_theme_icon(SNAME("TransitionEndAutoBig"), SNAME("EditorIcons"))
	};
	const int ICON_COUNT = sizeof(icons) / sizeof(*icons);

	if (p_selected) {
		state_machine_draw->draw_line(p_from, p_to, accent, 6);
	}

	if (p_travel) {
		linecolor = accent;
	}

	state_machine_draw->draw_line(p_from, p_to, linecolor, 2);

	if (p_fade_ratio > 0.0) {
		Color fade_linecolor = accent;
		fade_linecolor.set_hsv(1.0, fade_linecolor.get_s(), fade_linecolor.get_v());
		state_machine_draw->draw_line(p_from, p_from.lerp(p_to, p_fade_ratio), fade_linecolor, 2);
	}
	int icon_index = p_mode + (p_auto_advance ? ICON_COUNT / 2 : 0);
	ERR_FAIL_COND(icon_index >= ICON_COUNT);
	Ref<Texture2D> icon = icons[icon_index];

	Transform2D xf;
	xf.columns[0] = (p_to - p_from).normalized();
	xf.columns[1] = xf.columns[0].orthogonal();
	xf.columns[2] = (p_from + p_to) * 0.5 - xf.columns[1] * icon->get_height() * 0.5 - xf.columns[0] * icon->get_height() * 0.5;

	state_machine_draw->draw_set_transform_matrix(xf);
	if (p_multi_transitions) {
		state_machine_draw->draw_texture(icons[0], Vector2(-icon->get_width(), 0), icon_color);
		state_machine_draw->draw_texture(icons[0], Vector2(), icon_color);
		state_machine_draw->draw_texture(icons[0], Vector2(icon->get_width(), 0), icon_color);
	} else {
		state_machine_draw->draw_texture(icon, Vector2(), icon_color);
	}
	state_machine_draw->draw_set_transform_matrix(Transform2D());
}

void AnimationNodeStateMachineEditor::_clip_src_line_to_rect(Vector2 &r_from, const Vector2 &p_to, const Rect2 &p_rect) {
	if (p_to == r_from) {
		return;
	}

	//this could be optimized...
	Vector2 n = (p_to - r_from).normalized();
	while (p_rect.has_point(r_from)) {
		r_from += n;
	}
}

void AnimationNodeStateMachineEditor::_clip_dst_line_to_rect(const Vector2 &p_from, Vector2 &r_to, const Rect2 &p_rect) {
	if (r_to == p_from) {
		return;
	}

	//this could be optimized...
	Vector2 n = (r_to - p_from).normalized();
	while (p_rect.has_point(r_to)) {
		r_to -= n;
	}
}

void AnimationNodeStateMachineEditor::_state_machine_draw() {
	AnimationTree *tree = AnimationTreeEditor::get_singleton()->get_animation_tree();
	if (!tree) {
		return;
	}

	Ref<AnimationNodeStateMachinePlayback> playback = tree->get(AnimationTreeEditor::get_singleton()->get_base_path() + "playback");

	Ref<StyleBoxFlat> style = get_theme_stylebox(SNAME("state_machine_frame"), SNAME("GraphNode"));
	Ref<StyleBoxFlat> style_selected = get_theme_stylebox(SNAME("state_machine_selected_frame"), SNAME("GraphNode"));

	Ref<Font> font = get_theme_font(SNAME("title_font"), SNAME("GraphNode"));
	int font_size = get_theme_font_size(SNAME("title_font_size"), SNAME("GraphNode"));
	Color font_color = get_theme_color(SNAME("title_color"), SNAME("GraphNode"));
	Ref<Texture2D> play = get_theme_icon(SNAME("Play"), SNAME("EditorIcons"));
	Ref<Texture2D> edit = get_theme_icon(SNAME("Edit"), SNAME("EditorIcons"));
	Color accent = get_theme_color(SNAME("accent_color"), SNAME("Editor"));
	Color linecolor = get_theme_color(SNAME("font_color"), SNAME("Label"));
	linecolor.a *= 0.3;
	Ref<StyleBox> playing_overlay = get_theme_stylebox(SNAME("position"), SNAME("GraphNode"));

	Ref<StyleBoxFlat> start_overlay = style->duplicate();
	start_overlay->set_border_width_all(1 * EDSCALE);
	start_overlay->set_border_color(Color::html("#80f6cf"));

	Ref<StyleBoxFlat> end_overlay = style->duplicate();
	end_overlay->set_border_width_all(1 * EDSCALE);
	end_overlay->set_border_color(Color::html("#f26661"));

	bool playing = false;
	StringName current;
	StringName blend_from;
	Vector<StringName> travel_path;

	if (playback.is_valid()) {
		playing = playback->is_playing();
		current = playback->get_current_node();
		blend_from = playback->get_fading_from_node();
		travel_path = playback->get_travel_path();
	}

	if (state_machine_draw->has_focus()) {
		state_machine_draw->draw_rect(Rect2(Point2(), state_machine_draw->get_size()), accent, false);
	}
	int sep = 3 * EDSCALE;

	List<StringName> nodes;
	state_machine->get_node_list(&nodes);

	node_rects.clear();
	Rect2 scroll_range;

	//snap lines
	if (dragging_selected) {
		Vector2 from = (state_machine->get_node_position(selected_node) * EDSCALE) + drag_ofs - state_machine->get_graph_offset() * EDSCALE;
		if (snap_x != StringName()) {
			Vector2 to = (state_machine->get_node_position(snap_x) * EDSCALE) - state_machine->get_graph_offset() * EDSCALE;
			state_machine_draw->draw_line(from, to, linecolor, 2);
		}
		if (snap_y != StringName()) {
			Vector2 to = (state_machine->get_node_position(snap_y) * EDSCALE) - state_machine->get_graph_offset() * EDSCALE;
			state_machine_draw->draw_line(from, to, linecolor, 2);
		}
	}

	//pre pass nodes so we know the rectangles
	for (const StringName &E : nodes) {
		Ref<AnimationNode> anode = state_machine->get_node(E);
		String name = E;
		bool needs_editor = AnimationTreeEditor::get_singleton()->can_edit(anode);
		Ref<StyleBox> sb = selected_nodes.has(E) ? style_selected : style;

		Size2 s = sb->get_minimum_size();
		int strsize = font->get_string_size(name, HORIZONTAL_ALIGNMENT_LEFT, -1, font_size).width;
		s.width += strsize;
		s.height += MAX(font->get_height(font_size), play->get_height());
		s.width += sep + play->get_width();

		if (needs_editor) {
			s.width += sep + edit->get_width();
		}

		Vector2 offset;
		offset += state_machine->get_node_position(E) * EDSCALE;

		if (selected_nodes.has(E) && dragging_selected) {
			offset += drag_ofs;
		}

		offset -= s / 2;
		offset = offset.floor();

		//prepre rect

		NodeRect nr;
		nr.node = Rect2(offset, s);
		nr.node_name = E;

		scroll_range = scroll_range.merge(nr.node); //merge with range

		//now scroll it to draw
		nr.node.position -= state_machine->get_graph_offset() * EDSCALE;

		node_rects.push_back(nr);
	}

	transition_lines.clear();

	//draw connecting line for potential new transition
	if (connecting) {
		Vector2 from = (state_machine->get_node_position(connecting_from) * EDSCALE) - state_machine->get_graph_offset() * EDSCALE;
		Vector2 to;
		if (connecting_to_node != StringName()) {
			to = (state_machine->get_node_position(connecting_to_node) * EDSCALE) - state_machine->get_graph_offset() * EDSCALE;
		} else {
			to = connecting_to;
		}

		for (int i = 0; i < node_rects.size(); i++) {
			if (node_rects[i].node_name == connecting_from) {
				_clip_src_line_to_rect(from, to, node_rects[i].node);
			}
			if (node_rects[i].node_name == connecting_to_node) {
				_clip_dst_line_to_rect(from, to, node_rects[i].node);
			}
		}

		_connection_draw(from, to, AnimationNodeStateMachineTransition::SwitchMode(switch_mode->get_selected()), true, false, false, 0.0, false, false);
	}

	Ref<Texture2D> tr_reference_icon = get_theme_icon(SNAME("TransitionImmediateBig"), SNAME("EditorIcons"));
	float tr_bidi_offset = int(tr_reference_icon->get_height() * 0.8);

	//draw transition lines
	for (int i = 0; i < state_machine->get_transition_count(); i++) {
		TransitionLine tl;
		tl.transition_index = i;
		tl.from_node = state_machine->get_transition_from(i);
		StringName local_from = String(tl.from_node).get_slicec('/', 0);
		local_from = local_from == ".." ? state_machine->start_node : local_from;
		Vector2 ofs_from = (dragging_selected && selected_nodes.has(local_from)) ? drag_ofs : Vector2();
		tl.from = (state_machine->get_node_position(local_from) * EDSCALE) + ofs_from - state_machine->get_graph_offset() * EDSCALE;

		tl.to_node = state_machine->get_transition_to(i);
		StringName local_to = String(tl.to_node).get_slicec('/', 0);
		local_to = local_to == ".." ? state_machine->end_node : local_to;
		Vector2 ofs_to = (dragging_selected && selected_nodes.has(local_to)) ? drag_ofs : Vector2();
		tl.to = (state_machine->get_node_position(local_to) * EDSCALE) + ofs_to - state_machine->get_graph_offset() * EDSCALE;

		Ref<AnimationNodeStateMachineTransition> tr = state_machine->get_transition(i);
		tl.disabled = bool(tr->get_advance_mode() == AnimationNodeStateMachineTransition::ADVANCE_MODE_DISABLED);
		tl.auto_advance = bool(tr->get_advance_mode() == AnimationNodeStateMachineTransition::ADVANCE_MODE_AUTO);
		tl.advance_condition_name = tr->get_advance_condition_name();
		tl.advance_condition_state = false;
		tl.mode = tr->get_switch_mode();
		tl.width = tr_bidi_offset;
		tl.travel = false;
		tl.fade_ratio = 0.0;
		tl.hidden = false;

		if (state_machine->has_local_transition(local_to, local_from)) { //offset if same exists
			Vector2 offset = -(tl.from - tl.to).normalized().orthogonal() * tr_bidi_offset;
			tl.from += offset;
			tl.to += offset;
		}

		for (int j = 0; j < node_rects.size(); j++) {
			if (node_rects[j].node_name == local_from) {
				_clip_src_line_to_rect(tl.from, tl.to, node_rects[j].node);
			}
			if (node_rects[j].node_name == local_to) {
				_clip_dst_line_to_rect(tl.from, tl.to, node_rects[j].node);
			}
		}

		tl.selected = selected_transition_from == tl.from_node && selected_transition_to == tl.to_node;

		if (blend_from == local_from && current == local_to) {
			tl.travel = true;
			tl.fade_ratio = MIN(1.0, fading_pos / fading_time);
		}

		if (travel_path.size()) {
			if (current == local_from && travel_path[0] == local_to) {
				tl.travel = true;
			} else {
				for (int j = 0; j < travel_path.size() - 1; j++) {
					if (travel_path[j] == local_from && travel_path[j + 1] == local_to) {
						tl.travel = true;
						break;
					}
				}
			}
		}

		StringName fullpath = AnimationTreeEditor::get_singleton()->get_base_path() + String(tl.advance_condition_name);
		if (tl.advance_condition_name != StringName() && bool(tree->get(fullpath))) {
			tl.advance_condition_state = true;
			tl.auto_advance = true;
		}

		// check if already have this local transition
		for (int j = 0; j < transition_lines.size(); j++) {
			StringName from = String(transition_lines[j].from_node).get_slicec('/', 0);
			StringName to = String(transition_lines[j].to_node).get_slicec('/', 0);
			from = from == ".." ? state_machine->start_node : from;
			to = to == ".." ? state_machine->end_node : to;

			if (from == local_from && to == local_to) {
				tl.hidden = true;
				transition_lines.write[j].disabled = transition_lines[j].disabled && tl.disabled;
				transition_lines.write[j].multi_transitions.push_back(tl);
			}
		}
		transition_lines.push_back(tl);
	}

	for (int i = 0; i < transition_lines.size(); i++) {
		TransitionLine tl = transition_lines[i];
		if (!tl.hidden) {
			_connection_draw(tl.from, tl.to, tl.mode, !tl.disabled, tl.selected, tl.travel, tl.fade_ratio, tl.auto_advance, !tl.multi_transitions.is_empty());
		}
	}

	//draw actual nodes
	for (int i = 0; i < node_rects.size(); i++) {
		String name = node_rects[i].node_name;
		Ref<AnimationNode> anode = state_machine->get_node(name);
		bool needs_editor = AnimationTreeEditor::get_singleton()->can_edit(anode);
		Ref<StyleBox> sb = selected_nodes.has(name) ? style_selected : style;
		int strsize = font->get_string_size(name, HORIZONTAL_ALIGNMENT_LEFT, -1, font_size).width;
		NodeRect &nr = node_rects.write[i];

		Vector2 offset = nr.node.position;
		int h = nr.node.size.height;

		//prepre rect

		//now scroll it to draw
		state_machine_draw->draw_style_box(sb, nr.node);

		if (state_machine->start_node == name) {
			state_machine_draw->draw_style_box(sb == style_selected ? style_selected : start_overlay, nr.node);
		}

		if (state_machine->end_node == name) {
			state_machine_draw->draw_style_box(sb == style_selected ? style_selected : end_overlay, nr.node);
		}

		if (playing && (blend_from == name || current == name || travel_path.has(name))) {
			state_machine_draw->draw_style_box(playing_overlay, nr.node);
		}

		offset.x += sb->get_offset().x;

		nr.play.position = offset + Vector2(0, (h - play->get_height()) / 2).floor();
		nr.play.size = play->get_size();

		Ref<Texture2D> play_tex = play;

		if (over_node == name && over_node_what == 0) {
			state_machine_draw->draw_texture(play_tex, nr.play.position, accent);
		} else {
			state_machine_draw->draw_texture(play_tex, nr.play.position);
		}

		offset.x += sep + play->get_width();

		nr.name.position = offset + Vector2(0, (h - font->get_height(font_size)) / 2).floor();
		nr.name.size = Vector2(strsize, font->get_height(font_size));

		state_machine_draw->draw_string(font, nr.name.position + Vector2(0, font->get_ascent(font_size)), name, HORIZONTAL_ALIGNMENT_LEFT, -1, font_size, font_color);
		offset.x += strsize + sep;

		if (needs_editor) {
			nr.edit.position = offset + Vector2(0, (h - edit->get_height()) / 2).floor();
			nr.edit.size = edit->get_size();

			if (over_node == name && over_node_what == 1) {
				state_machine_draw->draw_texture(edit, nr.edit.position, accent);
			} else {
				state_machine_draw->draw_texture(edit, nr.edit.position);
			}
		}
	}

	//draw box select
	if (box_selecting) {
		state_machine_draw->draw_rect(box_selecting_rect, Color(0.7, 0.7, 1.0, 0.3));
	}

	scroll_range.position -= state_machine_draw->get_size();
	scroll_range.size += state_machine_draw->get_size() * 2.0;

	//adjust scrollbars
	updating = true;
	h_scroll->set_min(scroll_range.position.x);
	h_scroll->set_max(scroll_range.position.x + scroll_range.size.x);
	h_scroll->set_page(state_machine_draw->get_size().x);
	h_scroll->set_value(state_machine->get_graph_offset().x);

	v_scroll->set_min(scroll_range.position.y);
	v_scroll->set_max(scroll_range.position.y + scroll_range.size.y);
	v_scroll->set_page(state_machine_draw->get_size().y);
	v_scroll->set_value(state_machine->get_graph_offset().y);
	updating = false;

	state_machine_play_pos->queue_redraw();
}

void AnimationNodeStateMachineEditor::_state_machine_pos_draw_individual(String p_name, float p_ratio) {
	AnimationTree *tree = AnimationTreeEditor::get_singleton()->get_animation_tree();
	if (!tree) {
		return;
	}

	Ref<AnimationNodeStateMachinePlayback> playback = tree->get(AnimationTreeEditor::get_singleton()->get_base_path() + "playback");
	if (!playback.is_valid() || !playback->is_playing()) {
		return;
	}

	if (p_name == state_machine->start_node || p_name == state_machine->end_node || p_name.is_empty()) {
		return;
	}

	int idx = -1;
	for (int i = 0; i < node_rects.size(); i++) {
		if (node_rects[i].node_name == p_name) {
			idx = i;
			break;
		}
	}

	if (idx == -1) {
		return;
	}

	const NodeRect &nr = node_rects[idx];

	Vector2 from;
	from.x = nr.play.position.x;
	from.y = (nr.play.position.y + nr.play.size.y + nr.node.position.y + nr.node.size.y) * 0.5;

	Vector2 to;
	if (nr.edit.size.x) {
		to.x = nr.edit.position.x + nr.edit.size.x;
	} else {
		to.x = nr.name.position.x + nr.name.size.x;
	}
	to.y = from.y;

	float c = p_ratio;
	Color fg = get_theme_color(SNAME("font_color"), SNAME("Label"));
	Color bg = fg;
	bg.a *= 0.3;

	state_machine_play_pos->draw_line(from, to, bg, 2);

	to = from.lerp(to, c);

	state_machine_play_pos->draw_line(from, to, fg, 2);
}

void AnimationNodeStateMachineEditor::_state_machine_pos_draw_all() {
	AnimationTree *tree = AnimationTreeEditor::get_singleton()->get_animation_tree();
	if (!tree) {
		return;
	}

	Ref<AnimationNodeStateMachinePlayback> playback = tree->get(AnimationTreeEditor::get_singleton()->get_base_path() + "playback");
	if (!playback.is_valid() || !playback->is_playing()) {
		return;
	}

	{
		float len = MAX(0.0001, current_length);
		float pos = CLAMP(current_play_pos, 0, len);
		float c = pos / len;
		_state_machine_pos_draw_individual(playback->get_current_node(), c);
	}

	{
		float len = MAX(0.0001, fade_from_length);
		float pos = CLAMP(fade_from_current_play_pos, 0, len);
		float c = pos / len;
		_state_machine_pos_draw_individual(playback->get_fading_from_node(), c);
	}
}

void AnimationNodeStateMachineEditor::_update_graph() {
	if (updating) {
		return;
	}

	updating = true;

	state_machine_draw->queue_redraw();

	updating = false;
}

void AnimationNodeStateMachineEditor::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE:
		case NOTIFICATION_THEME_CHANGED:
		case NOTIFICATION_LAYOUT_DIRECTION_CHANGED:
		case NOTIFICATION_TRANSLATION_CHANGED: {
			error_panel->add_theme_style_override("panel", get_theme_stylebox(SNAME("panel"), SNAME("Tree")));
			error_label->add_theme_color_override("font_color", get_theme_color(SNAME("error_color"), SNAME("Editor")));
			panel->add_theme_style_override("panel", get_theme_stylebox(SNAME("panel"), SNAME("Tree")));

			tool_select->set_icon(get_theme_icon(SNAME("ToolSelect"), SNAME("EditorIcons")));
			tool_create->set_icon(get_theme_icon(SNAME("ToolAddNode"), SNAME("EditorIcons")));
			tool_connect->set_icon(get_theme_icon(SNAME("ToolConnect"), SNAME("EditorIcons")));

			switch_mode->clear();
			switch_mode->add_icon_item(get_theme_icon(SNAME("TransitionImmediate"), SNAME("EditorIcons")), TTR("Immediate"));
			switch_mode->add_icon_item(get_theme_icon(SNAME("TransitionSync"), SNAME("EditorIcons")), TTR("Sync"));
			switch_mode->add_icon_item(get_theme_icon(SNAME("TransitionEnd"), SNAME("EditorIcons")), TTR("At End"));

			auto_advance->set_icon(get_theme_icon(SNAME("AutoPlay"), SNAME("EditorIcons")));

			tool_erase->set_icon(get_theme_icon(SNAME("Remove"), SNAME("EditorIcons")));
			tool_group->set_icon(get_theme_icon(SNAME("Group"), SNAME("EditorIcons")));
			tool_ungroup->set_icon(get_theme_icon(SNAME("Ungroup"), SNAME("EditorIcons")));

			play_mode->clear();
			play_mode->add_icon_item(get_theme_icon(SNAME("PlayTravel"), SNAME("EditorIcons")), TTR("Travel"));
			play_mode->add_icon_item(get_theme_icon(SNAME("Play"), SNAME("EditorIcons")), TTR("Immediate"));
		} break;

		case NOTIFICATION_PROCESS: {
			AnimationTree *tree = AnimationTreeEditor::get_singleton()->get_animation_tree();
			if (!tree) {
				return;
			}

			String error;

			Ref<AnimationNodeStateMachinePlayback> playback = tree->get(AnimationTreeEditor::get_singleton()->get_base_path() + "playback");

			if (error_time > 0) {
				error = error_text;
				error_time -= get_process_delta_time();
			} else if (!tree->is_active()) {
				error = TTR("AnimationTree is inactive.\nActivate to enable playback, check node warnings if activation fails.");
			} else if (tree->is_state_invalid()) {
				error = tree->get_invalid_state_reason();
				/*} else if (state_machine->get_parent().is_valid() && state_machine->get_parent()->is_class("AnimationNodeStateMachine")) {
				if (state_machine->get_start_node() == StringName() || state_machine->get_end_node() == StringName()) {
					error = TTR("Start and end nodes are needed for a sub-transition.");
				}*/
			} else if (playback.is_null()) {
				error = vformat(TTR("No playback resource set at path: %s."), AnimationTreeEditor::get_singleton()->get_base_path() + "playback");
			}

			if (error != error_label->get_text()) {
				error_label->set_text(error);
				if (!error.is_empty()) {
					error_panel->show();
				} else {
					error_panel->hide();
				}
			}

			for (int i = 0; i < transition_lines.size(); i++) {
				int tidx = -1;
				for (int j = 0; j < state_machine->get_transition_count(); j++) {
					if (transition_lines[i].from_node == state_machine->get_transition_from(j) && transition_lines[i].to_node == state_machine->get_transition_to(j)) {
						tidx = j;
						break;
					}
				}

				if (tidx == -1) { //missing transition, should redraw
					state_machine_draw->queue_redraw();
					break;
				}

				if (transition_lines[i].disabled != bool(state_machine->get_transition(tidx)->get_advance_mode() == AnimationNodeStateMachineTransition::ADVANCE_MODE_DISABLED)) {
					state_machine_draw->queue_redraw();
					break;
				}

				if (transition_lines[i].auto_advance != bool(state_machine->get_transition(tidx)->get_advance_mode() == AnimationNodeStateMachineTransition::ADVANCE_MODE_AUTO)) {
					state_machine_draw->queue_redraw();
					break;
				}

				if (transition_lines[i].advance_condition_name != state_machine->get_transition(tidx)->get_advance_condition_name()) {
					state_machine_draw->queue_redraw();
					break;
				}

				if (transition_lines[i].mode != state_machine->get_transition(tidx)->get_switch_mode()) {
					state_machine_draw->queue_redraw();
					break;
				}

				bool acstate = transition_lines[i].advance_condition_name != StringName() && bool(tree->get(AnimationTreeEditor::get_singleton()->get_base_path() + String(transition_lines[i].advance_condition_name)));

				if (transition_lines[i].advance_condition_state != acstate) {
					state_machine_draw->queue_redraw();
					break;
				}
			}

			bool same_travel_path = true;
			Vector<StringName> tp;
			bool is_playing = false;
			StringName current_node;
			StringName fading_from_node;

			current_play_pos = 0;
			current_length = 0;

			fade_from_current_play_pos = 0;
			fade_from_length = 0;

			fading_time = 0;
			fading_pos = 0;

			if (playback.is_valid()) {
				tp = playback->get_travel_path();
				is_playing = playback->is_playing();
				current_node = playback->get_current_node();
				fading_from_node = playback->get_fading_from_node();
				current_play_pos = playback->get_current_play_pos();
				current_length = playback->get_current_length();
				fade_from_current_play_pos = playback->get_fade_from_play_pos();
				fade_from_length = playback->get_fade_from_length();
				fading_time = playback->get_fading_time();
				fading_pos = playback->get_fading_pos();
			}

			{
				if (last_travel_path.size() != tp.size()) {
					same_travel_path = false;
				} else {
					for (int i = 0; i < last_travel_path.size(); i++) {
						if (last_travel_path[i] != tp[i]) {
							same_travel_path = false;
							break;
						}
					}
				}
			}

			//redraw if travel state changed
			if (!same_travel_path ||
					last_active != is_playing ||
					last_current_node != current_node ||
					last_fading_from_node != fading_from_node ||
					last_fading_time != fading_time ||
					last_fading_pos != fading_pos) {
				state_machine_draw->queue_redraw();
				last_travel_path = tp;
				last_current_node = current_node;
				last_active = is_playing;
				last_fading_from_node = fading_from_node;
				last_fading_time = fading_time;
				last_fading_pos = fading_pos;
				state_machine_play_pos->queue_redraw();
			}

			{
				if (current_node != StringName() && state_machine->has_node(current_node)) {
					String next = current_node;
					Ref<AnimationNodeStateMachine> anodesm = state_machine->get_node(next);
					Ref<AnimationNodeStateMachinePlayback> current_node_playback;

					while (anodesm.is_valid()) {
						current_node_playback = tree->get(AnimationTreeEditor::get_singleton()->get_base_path() + next + "/playback");
						next += "/" + current_node_playback->get_current_node();
						anodesm = anodesm->get_node(current_node_playback->get_current_node());
					}

					// when current_node is a state machine, use playback of current_node to set play_pos
					if (current_node_playback.is_valid()) {
						current_play_pos = current_node_playback->get_current_play_pos();
						current_length = current_node_playback->get_current_length();
					}
				}
			}

			if (last_play_pos != current_play_pos || fade_from_last_play_pos != fade_from_current_play_pos) {
				last_play_pos = current_play_pos;
				fade_from_last_play_pos = fade_from_current_play_pos;
				state_machine_play_pos->queue_redraw();
			}
		} break;

		case NOTIFICATION_VISIBILITY_CHANGED: {
			over_node = StringName();
			set_process(is_visible_in_tree());
		} break;
	}
}

void AnimationNodeStateMachineEditor::_open_editor(const String &p_name) {
	AnimationTreeEditor::get_singleton()->enter_editor(p_name);
}

void AnimationNodeStateMachineEditor::_name_edited(const String &p_text) {
	const String &new_name = p_text;

	ERR_FAIL_COND(new_name.is_empty() || new_name.contains(".") || new_name.contains("/"));

	if (new_name == prev_name) {
		return; // Nothing to do.
	}

	const String &base_name = new_name;
	int base = 1;
	String name = base_name;
	while (state_machine->has_node(name)) {
		base++;
		name = base_name + " " + itos(base);
	}

	updating = true;
	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(TTR("Node Renamed"));
	undo_redo->add_do_method(state_machine.ptr(), "rename_node", prev_name, name);
	undo_redo->add_undo_method(state_machine.ptr(), "rename_node", name, prev_name);
	undo_redo->add_do_method(this, "_update_graph");
	undo_redo->add_undo_method(this, "_update_graph");
	undo_redo->commit_action();
	name_edit_popup->hide();
	updating = false;

	state_machine_draw->queue_redraw();
}

void AnimationNodeStateMachineEditor::_name_edited_focus_out() {
	if (updating) {
		return;
	}

	_name_edited(name_edit->get_text());
}

void AnimationNodeStateMachineEditor::_scroll_changed(double) {
	if (updating) {
		return;
	}

	state_machine->set_graph_offset(Vector2(h_scroll->get_value(), v_scroll->get_value()));
	state_machine_draw->queue_redraw();
}

void AnimationNodeStateMachineEditor::_erase_selected(const bool p_nested_action) {
	if (!selected_nodes.is_empty()) {
		if (!p_nested_action) {
			updating = true;
		}
		EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
		undo_redo->create_action(TTR("Node Removed"));

		for (int i = 0; i < node_rects.size(); i++) {
			if (node_rects[i].node_name == state_machine->start_node || node_rects[i].node_name == state_machine->end_node) {
				continue;
			}

			if (!selected_nodes.has(node_rects[i].node_name)) {
				continue;
			}

			undo_redo->add_do_method(state_machine.ptr(), "remove_node", node_rects[i].node_name);
			undo_redo->add_undo_method(state_machine.ptr(), "add_node", node_rects[i].node_name,
					state_machine->get_node(node_rects[i].node_name),
					state_machine->get_node_position(node_rects[i].node_name));

			for (int j = 0; j < state_machine->get_transition_count(); j++) {
				String from = state_machine->get_transition_from(j);
				String to = state_machine->get_transition_to(j);
				String local_from = from.get_slicec('/', 0);
				String local_to = to.get_slicec('/', 0);

				if (local_from == node_rects[i].node_name || local_to == node_rects[i].node_name) {
					undo_redo->add_undo_method(state_machine.ptr(), "add_transition", from, to, state_machine->get_transition(j));
				}
			}
		}

		undo_redo->add_do_method(this, "_update_graph");
		undo_redo->add_undo_method(this, "_update_graph");
		undo_redo->commit_action();

		if (!p_nested_action) {
			updating = false;
		}

		selected_nodes.clear();
	}

	if (!selected_multi_transition.multi_transitions.is_empty()) {
		delete_tree->clear();

		TreeItem *root = delete_tree->create_item();

		TreeItem *item = delete_tree->create_item(root);
		item->set_cell_mode(0, TreeItem::CELL_MODE_CHECK);
		item->set_text(0, String(selected_transition_from) + " -> " + selected_transition_to);
		item->set_editable(0, true);

		for (int i = 0; i < selected_multi_transition.multi_transitions.size(); i++) {
			String from = selected_multi_transition.multi_transitions[i].from_node;
			String to = selected_multi_transition.multi_transitions[i].to_node;

			item = delete_tree->create_item(root);
			item->set_cell_mode(0, TreeItem::CELL_MODE_CHECK);
			item->set_text(0, from + " -> " + to);
			item->set_editable(0, true);
		}

		delete_window->popup_centered(Vector2(400, 200));
		return;
	}

	if (selected_transition_to != StringName() && selected_transition_from != StringName() && state_machine->has_transition(selected_transition_from, selected_transition_to)) {
		Ref<AnimationNodeStateMachineTransition> tr = state_machine->get_transition(state_machine->find_transition(selected_transition_from, selected_transition_to));
		if (!p_nested_action) {
			updating = true;
		}
		EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
		undo_redo->create_action(TTR("Transition Removed"));
		undo_redo->add_do_method(state_machine.ptr(), "remove_transition", selected_transition_from, selected_transition_to);
		undo_redo->add_undo_method(state_machine.ptr(), "add_transition", selected_transition_from, selected_transition_to, tr);
		undo_redo->add_do_method(this, "_update_graph");
		undo_redo->add_undo_method(this, "_update_graph");
		undo_redo->commit_action();
		if (!p_nested_action) {
			updating = false;
		}
		selected_transition_from = StringName();
		selected_transition_to = StringName();
		selected_transition_index = -1;
		selected_multi_transition = TransitionLine();
	}

	state_machine_draw->queue_redraw();
}

void AnimationNodeStateMachineEditor::_update_mode() {
	if (tool_select->is_pressed()) {
		selection_tools_hb->show();
		bool nothing_selected = selected_nodes.is_empty() && selected_transition_from == StringName() && selected_transition_to == StringName();
		bool start_end_selected = selected_nodes.size() == 1 && (*selected_nodes.begin() == state_machine->start_node || *selected_nodes.begin() == state_machine->end_node);
		tool_erase->set_disabled(nothing_selected || start_end_selected || read_only);

		if (selected_nodes.is_empty() || start_end_selected || read_only) {
			tool_group->set_disabled(true);
			tool_group->set_visible(true);
			tool_ungroup->set_visible(false);
		} else {
			Ref<AnimationNodeStateMachine> ansm = state_machine->get_node(*selected_nodes.begin());

			if (selected_nodes.size() == 1 && ansm.is_valid()) {
				tool_group->set_disabled(true);
				tool_group->set_visible(false);
				tool_ungroup->set_visible(true);
			} else {
				tool_group->set_disabled(false);
				tool_group->set_visible(true);
				tool_ungroup->set_visible(false);
			}
		}
	} else {
		selection_tools_hb->hide();
	}

	if (tool_connect->is_pressed()) {
		transition_tools_hb->show();
	} else {
		transition_tools_hb->hide();
	}
}

void AnimationNodeStateMachineEditor::_bind_methods() {
	ClassDB::bind_method("_update_graph", &AnimationNodeStateMachineEditor::_update_graph);
	ClassDB::bind_method("_open_editor", &AnimationNodeStateMachineEditor::_open_editor);
	ClassDB::bind_method("_connect_to", &AnimationNodeStateMachineEditor::_connect_to);
	ClassDB::bind_method("_stop_connecting", &AnimationNodeStateMachineEditor::_stop_connecting);
	ClassDB::bind_method("_delete_selected", &AnimationNodeStateMachineEditor::_delete_selected);
	ClassDB::bind_method("_delete_all", &AnimationNodeStateMachineEditor::_delete_all);
	ClassDB::bind_method("_delete_tree_draw", &AnimationNodeStateMachineEditor::_delete_tree_draw);
}

AnimationNodeStateMachineEditor *AnimationNodeStateMachineEditor::singleton = nullptr;

AnimationNodeStateMachineEditor::AnimationNodeStateMachineEditor() {
	singleton = this;

	HBoxContainer *top_hb = memnew(HBoxContainer);
	add_child(top_hb);

	Ref<ButtonGroup> bg;
	bg.instantiate();

	tool_select = memnew(Button);
	tool_select->set_flat(true);
	top_hb->add_child(tool_select);
	tool_select->set_toggle_mode(true);
	tool_select->set_button_group(bg);
	tool_select->set_pressed(true);
	tool_select->set_tooltip_text(TTR("Select and move nodes.\nRMB: Add node at position clicked.\nShift+LMB+Drag: Connects the selected node with another node or creates a new node if you select an area without nodes."));
	tool_select->connect("pressed", callable_mp(this, &AnimationNodeStateMachineEditor::_update_mode), CONNECT_DEFERRED);

	tool_create = memnew(Button);
	tool_create->set_flat(true);
	top_hb->add_child(tool_create);
	tool_create->set_toggle_mode(true);
	tool_create->set_button_group(bg);
	tool_create->set_tooltip_text(TTR("Create new nodes."));
	tool_create->connect("pressed", callable_mp(this, &AnimationNodeStateMachineEditor::_update_mode), CONNECT_DEFERRED);

	tool_connect = memnew(Button);
	tool_connect->set_flat(true);
	top_hb->add_child(tool_connect);
	tool_connect->set_toggle_mode(true);
	tool_connect->set_button_group(bg);
	tool_connect->set_tooltip_text(TTR("Connect nodes."));
	tool_connect->connect("pressed", callable_mp(this, &AnimationNodeStateMachineEditor::_update_mode), CONNECT_DEFERRED);

	// Context-sensitive selection tools:
	selection_tools_hb = memnew(HBoxContainer);
	top_hb->add_child(selection_tools_hb);
	selection_tools_hb->add_child(memnew(VSeparator));

	tool_group = memnew(Button);
	tool_group->set_flat(true);
	tool_group->set_tooltip_text(TTR("Group Selected Node(s)") + " (Ctrl+G)");
	tool_group->connect("pressed", callable_mp(this, &AnimationNodeStateMachineEditor::_group_selected_nodes));
	tool_group->set_disabled(true);
	selection_tools_hb->add_child(tool_group);

	tool_ungroup = memnew(Button);
	tool_ungroup->set_flat(true);
	tool_ungroup->set_tooltip_text(TTR("Ungroup Selected Node") + " (Ctrl+Shift+G)");
	tool_ungroup->connect("pressed", callable_mp(this, &AnimationNodeStateMachineEditor::_ungroup_selected_nodes));
	tool_ungroup->set_visible(false);
	selection_tools_hb->add_child(tool_ungroup);

	tool_erase = memnew(Button);
	tool_erase->set_flat(true);
	tool_erase->set_tooltip_text(TTR("Remove selected node or transition."));
	tool_erase->connect("pressed", callable_mp(this, &AnimationNodeStateMachineEditor::_erase_selected).bind(false));
	tool_erase->set_disabled(true);
	selection_tools_hb->add_child(tool_erase);

	transition_tools_hb = memnew(HBoxContainer);
	top_hb->add_child(transition_tools_hb);
	transition_tools_hb->add_child(memnew(VSeparator));

	transition_tools_hb->add_child(memnew(Label(TTR("Transition:"))));
	switch_mode = memnew(OptionButton);
	transition_tools_hb->add_child(switch_mode);

	auto_advance = memnew(Button);
	auto_advance->set_flat(true);
	auto_advance->set_tooltip_text(TTR("New Transitions Should Auto Advance"));
	auto_advance->set_toggle_mode(true);
	auto_advance->set_pressed(true);
	transition_tools_hb->add_child(auto_advance);

	//

	top_hb->add_spacer();

	top_hb->add_child(memnew(Label(TTR("Play Mode:"))));
	play_mode = memnew(OptionButton);
	top_hb->add_child(play_mode);

	panel = memnew(PanelContainer);
	panel->set_clip_contents(true);
	panel->set_mouse_filter(Control::MOUSE_FILTER_PASS);
	add_child(panel);
	panel->set_v_size_flags(SIZE_EXPAND_FILL);

	state_machine_draw = memnew(Control);
	panel->add_child(state_machine_draw);
	state_machine_draw->connect("gui_input", callable_mp(this, &AnimationNodeStateMachineEditor::_state_machine_gui_input));
	state_machine_draw->connect("draw", callable_mp(this, &AnimationNodeStateMachineEditor::_state_machine_draw));
	state_machine_draw->set_focus_mode(FOCUS_ALL);
	state_machine_draw->set_mouse_filter(Control::MOUSE_FILTER_PASS);

	state_machine_play_pos = memnew(Control);
	state_machine_draw->add_child(state_machine_play_pos);
	state_machine_play_pos->set_mouse_filter(MOUSE_FILTER_PASS); //pass all to parent
	state_machine_play_pos->set_anchors_and_offsets_preset(PRESET_FULL_RECT);
	state_machine_play_pos->connect("draw", callable_mp(this, &AnimationNodeStateMachineEditor::_state_machine_pos_draw_all));

	v_scroll = memnew(VScrollBar);
	state_machine_draw->add_child(v_scroll);
	v_scroll->set_anchors_and_offsets_preset(PRESET_RIGHT_WIDE);
	v_scroll->connect("value_changed", callable_mp(this, &AnimationNodeStateMachineEditor::_scroll_changed));

	h_scroll = memnew(HScrollBar);
	state_machine_draw->add_child(h_scroll);
	h_scroll->set_anchors_and_offsets_preset(PRESET_BOTTOM_WIDE);
	h_scroll->set_offset(SIDE_RIGHT, -v_scroll->get_size().x * EDSCALE);
	h_scroll->connect("value_changed", callable_mp(this, &AnimationNodeStateMachineEditor::_scroll_changed));

	error_panel = memnew(PanelContainer);
	add_child(error_panel);
	error_label = memnew(Label);
	error_panel->add_child(error_label);
	error_panel->hide();

	set_custom_minimum_size(Size2(0, 300 * EDSCALE));

	menu = memnew(PopupMenu);
	add_child(menu);
	menu->connect("id_pressed", callable_mp(this, &AnimationNodeStateMachineEditor::_add_menu_type));
	menu->connect("popup_hide", callable_mp(this, &AnimationNodeStateMachineEditor::_stop_connecting));

	animations_menu = memnew(PopupMenu);
	menu->add_child(animations_menu);
	animations_menu->set_name("animations");
	animations_menu->connect("index_pressed", callable_mp(this, &AnimationNodeStateMachineEditor::_add_animation_type));

	connect_menu = memnew(PopupMenu);
	add_child(connect_menu);
	connect_menu->connect("id_pressed", callable_mp(this, &AnimationNodeStateMachineEditor::_connect_to));
	connect_menu->connect("popup_hide", callable_mp(this, &AnimationNodeStateMachineEditor::_stop_connecting));

	state_machine_menu = memnew(PopupMenu);
	state_machine_menu->set_name("state_machines");
	state_machine_menu->connect("id_pressed", callable_mp(this, &AnimationNodeStateMachineEditor::_connect_to));
	connect_menu->add_child(state_machine_menu);

	end_menu = memnew(PopupMenu);
	end_menu->set_name("end_nodes");
	end_menu->connect("id_pressed", callable_mp(this, &AnimationNodeStateMachineEditor::_connect_to));
	connect_menu->add_child(end_menu);

	name_edit_popup = memnew(Popup);
	add_child(name_edit_popup);
	name_edit = memnew(LineEdit);
	name_edit_popup->add_child(name_edit);
	name_edit->set_anchors_and_offsets_preset(PRESET_FULL_RECT);
	name_edit->connect("text_submitted", callable_mp(this, &AnimationNodeStateMachineEditor::_name_edited));
	name_edit->connect("focus_exited", callable_mp(this, &AnimationNodeStateMachineEditor::_name_edited_focus_out));

	open_file = memnew(EditorFileDialog);
	add_child(open_file);
	open_file->set_title(TTR("Open Animation Node"));
	open_file->set_file_mode(EditorFileDialog::FILE_MODE_OPEN_FILE);
	open_file->connect("file_selected", callable_mp(this, &AnimationNodeStateMachineEditor::_file_opened));

	delete_window = memnew(ConfirmationDialog);
	delete_window->set_flag(Window::FLAG_RESIZE_DISABLED, true);
	add_child(delete_window);

	delete_tree = memnew(Tree);
	delete_tree->set_hide_root(true);
	delete_tree->connect("draw", callable_mp(this, &AnimationNodeStateMachineEditor::_delete_tree_draw));
	delete_window->add_child(delete_tree);

	Button *ok = delete_window->get_cancel_button();
	ok->set_text(TTR("Delete Selected"));
	ok->connect("pressed", callable_mp(this, &AnimationNodeStateMachineEditor::_delete_selected));

	Button *delete_all = delete_window->add_button(TTR("Delete All"), true);
	delete_all->connect("pressed", callable_mp(this, &AnimationNodeStateMachineEditor::_delete_all));

	over_node_what = -1;
	dragging_selected_attempt = false;
	connecting = false;
	selected_transition_index = -1;

	last_active = false;

	error_time = 0;
}

void EditorAnimationMultiTransitionEdit::add_transition(const StringName &p_from, const StringName &p_to, Ref<AnimationNodeStateMachineTransition> p_transition) {
	Transition tr;
	tr.from = p_from;
	tr.to = p_to;
	tr.transition = p_transition;
	transitions.push_back(tr);
}

bool EditorAnimationMultiTransitionEdit::_set(const StringName &p_name, const Variant &p_property) {
	int index = String(p_name).get_slicec('/', 0).to_int();
	StringName prop = String(p_name).get_slicec('/', 1);

	bool found;
	transitions.write[index].transition->set(prop, p_property, &found);
	if (found) {
		return true;
	}

	return false;
}

bool EditorAnimationMultiTransitionEdit::_get(const StringName &p_name, Variant &r_property) const {
	int index = String(p_name).get_slicec('/', 0).to_int();
	StringName prop = String(p_name).get_slicec('/', 1);

	if (prop == "transition_path") {
		r_property = String(transitions[index].from) + " -> " + transitions[index].to;
		return true;
	}

	bool found;
	r_property = transitions[index].transition->get(prop, &found);
	if (found) {
		return true;
	}

	return false;
}

void EditorAnimationMultiTransitionEdit::_get_property_list(List<PropertyInfo> *p_list) const {
	for (int i = 0; i < transitions.size(); i++) {
		List<PropertyInfo> plist;
		transitions[i].transition->get_property_list(&plist, true);

		PropertyInfo prop_transition_path;
		prop_transition_path.type = Variant::STRING;
		prop_transition_path.name = itos(i) + "/" + "transition_path";
		p_list->push_back(prop_transition_path);

		for (List<PropertyInfo>::Element *F = plist.front(); F; F = F->next()) {
			if (F->get().name == "script" || F->get().name == "resource_name" || F->get().name == "resource_path" || F->get().name == "resource_local_to_scene") {
				continue;
			}

			if (F->get().usage != PROPERTY_USAGE_DEFAULT) {
				continue;
			}

			PropertyInfo prop = F->get();
			prop.name = itos(i) + "/" + prop.name;

			p_list->push_back(prop);
		}
	}
}
