/*************************************************************************/
/*  animation_state_machine_editor.cpp                                   */
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

#include "animation_state_machine_editor.h"

#include "core/io/resource_loader.h"
#include "core/math/delaunay.h"
#include "core/os/input.h"
#include "core/os/keyboard.h"
#include "core/project_settings.h"
#include "editor/editor_scale.h"
#include "scene/animation/animation_blend_tree.h"
#include "scene/animation/animation_player.h"
#include "scene/gui/menu_button.h"
#include "scene/gui/panel.h"
#include "scene/main/viewport.h"

bool AnimationNodeStateMachineEditor::can_edit(const Ref<AnimationNode> &p_node) {
	Ref<AnimationNodeStateMachine> ansm = p_node;
	return ansm.is_valid();
}

void AnimationNodeStateMachineEditor::edit(const Ref<AnimationNode> &p_node) {
	state_machine = p_node;

	if (state_machine.is_valid()) {
		selected_transition_from = StringName();
		selected_transition_to = StringName();
		selected_node = StringName();
		_update_mode();
		_update_graph();
	}
}

void AnimationNodeStateMachineEditor::_state_machine_gui_input(const Ref<InputEvent> &p_event) {
	Ref<AnimationNodeStateMachinePlayback> playback = AnimationTreeEditor::get_singleton()->get_tree()->get(AnimationTreeEditor::get_singleton()->get_base_path() + "playback");
	if (playback.is_null()) {
		return;
	}

	Ref<InputEventKey> k = p_event;
	if (tool_select->is_pressed() && k.is_valid() && k->is_pressed() && k->get_scancode() == KEY_DELETE && !k->is_echo()) {
		if (selected_node != StringName() || selected_transition_to != StringName() || selected_transition_from != StringName()) {
			_erase_selected();
			accept_event();
		}
	}

	Ref<InputEventMouseButton> mb = p_event;

	//Add new node
	if (mb.is_valid() && mb->is_pressed() && ((tool_select->is_pressed() && mb->get_button_index() == BUTTON_RIGHT) || (tool_create->is_pressed() && mb->get_button_index() == BUTTON_LEFT))) {
		menu->clear();
		animations_menu->clear();
		animations_to_add.clear();
		List<StringName> classes;
		classes.sort_custom<StringName::AlphCompare>();

		ClassDB::get_inheriters_from_class("AnimationRootNode", &classes);
		menu->add_submenu_item(TTR("Add Animation"), "animations");

		AnimationTree *gp = AnimationTreeEditor::get_singleton()->get_tree();
		ERR_FAIL_COND(!gp);
		if (gp && gp->has_node(gp->get_animation_player())) {
			AnimationPlayer *ap = Object::cast_to<AnimationPlayer>(gp->get_node(gp->get_animation_player()));
			if (ap) {
				List<StringName> names;
				ap->get_animation_list(&names);
				for (List<StringName>::Element *E = names.front(); E; E = E->next()) {
					animations_menu->add_icon_item(get_icon("Animation", "EditorIcons"), E->get());
					animations_to_add.push_back(E->get());
				}
			}
		}

		for (List<StringName>::Element *E = classes.front(); E; E = E->next()) {
			String name = String(E->get()).replace_first("AnimationNode", "");
			if (name == "Animation") {
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

		menu->set_global_position(state_machine_draw->get_global_transform().xform(mb->get_position()));
		menu->popup();
		add_node_pos = mb->get_position() / EDSCALE + state_machine->get_graph_offset();
	}

	// select node or push a field inside
	if (mb.is_valid() && !mb->get_shift() && mb->is_pressed() && tool_select->is_pressed() && mb->get_button_index() == BUTTON_LEFT) {
		selected_transition_from = StringName();
		selected_transition_to = StringName();
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
				state_machine_draw->update();
				return;
			}

			if (node_rects[i].name.has_point(mb->get_position())) { //edit name

				Ref<StyleBox> line_sb = get_stylebox("normal", "LineEdit");

				Rect2 edit_rect = node_rects[i].name;
				edit_rect.position -= line_sb->get_offset();
				edit_rect.size += line_sb->get_minimum_size();

				name_edit->set_global_position(state_machine_draw->get_global_transform().xform(edit_rect.position));
				name_edit->set_size(edit_rect.size);
				name_edit->set_text(node_rects[i].node_name);
				name_edit->show_modal();
				name_edit->grab_focus();
				name_edit->select_all();

				prev_name = node_rects[i].node_name;
				return;
			}

			if (node_rects[i].edit.has_point(mb->get_position())) { //edit name
				call_deferred("_open_editor", node_rects[i].node_name);
				return;
			}

			if (node_rects[i].node.has_point(mb->get_position())) { //select node since nothing else was selected
				selected_node = node_rects[i].node_name;

				Ref<AnimationNode> anode = state_machine->get_node(selected_node);
				EditorNode::get_singleton()->push_item(anode.ptr(), "", true);
				state_machine_draw->update();
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
			Vector2 cpoint = Geometry::get_closest_point_to_segment_2d(mb->get_position(), s);
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

			Ref<AnimationNodeStateMachineTransition> tr = state_machine->get_transition(closest);
			EditorNode::get_singleton()->push_item(tr.ptr(), "", true);
		}

		state_machine_draw->update();
		_update_mode();
	}

	//end moving node
	if (mb.is_valid() && dragging_selected_attempt && mb->get_button_index() == BUTTON_LEFT && !mb->is_pressed()) {
		if (dragging_selected) {
			Ref<AnimationNode> an = state_machine->get_node(selected_node);
			updating = true;
			undo_redo->create_action(TTR("Move Node"));
			undo_redo->add_do_method(state_machine.ptr(), "set_node_position", selected_node, state_machine->get_node_position(selected_node) + drag_ofs / EDSCALE);
			undo_redo->add_undo_method(state_machine.ptr(), "set_node_position", selected_node, state_machine->get_node_position(selected_node));
			undo_redo->add_do_method(this, "_update_graph");
			undo_redo->add_undo_method(this, "_update_graph");
			undo_redo->commit_action();
			updating = false;
		}
		snap_x = StringName();
		snap_y = StringName();

		dragging_selected_attempt = false;
		dragging_selected = false;
		state_machine_draw->update();
	}

	//connect nodes
	if (mb.is_valid() && ((tool_select->is_pressed() && mb->get_shift()) || tool_connect->is_pressed()) && mb->get_button_index() == BUTTON_LEFT && mb->is_pressed()) {
		for (int i = node_rects.size() - 1; i >= 0; i--) { //inverse to draw order
			if (node_rects[i].node.has_point(mb->get_position())) { //select node since nothing else was selected
				connecting = true;
				connecting_from = node_rects[i].node_name;
				connecting_to = mb->get_position();
				connecting_to_node = StringName();
				return;
			}
		}
	}

	//end connecting nodes
	if (mb.is_valid() && connecting && mb->get_button_index() == BUTTON_LEFT && !mb->is_pressed()) {
		if (connecting_to_node != StringName()) {
			if (state_machine->has_transition(connecting_from, connecting_to_node)) {
				EditorNode::get_singleton()->show_warning(TTR("Transition exists!"));

			} else {
				Ref<AnimationNodeStateMachineTransition> tr;
				tr.instance();
				tr->set_switch_mode(AnimationNodeStateMachineTransition::SwitchMode(transition_mode->get_selected()));

				updating = true;
				undo_redo->create_action(TTR("Add Transition"));
				undo_redo->add_do_method(state_machine.ptr(), "add_transition", connecting_from, connecting_to_node, tr);
				undo_redo->add_undo_method(state_machine.ptr(), "remove_transition", connecting_from, connecting_to_node);
				undo_redo->add_do_method(this, "_update_graph");
				undo_redo->add_undo_method(this, "_update_graph");
				undo_redo->commit_action();
				updating = false;

				selected_transition_from = connecting_from;
				selected_transition_to = connecting_to_node;

				EditorNode::get_singleton()->push_item(tr.ptr(), "", true);
				_update_mode();
			}
		}
		connecting_to_node = StringName();
		connecting = false;
		state_machine_draw->update();
	}

	Ref<InputEventMouseMotion> mm = p_event;

	//pan window
	if (mm.is_valid() && mm->get_button_mask() & BUTTON_MASK_MIDDLE) {
		h_scroll->set_value(h_scroll->get_value() - mm->get_relative().x);
		v_scroll->set_value(v_scroll->get_value() - mm->get_relative().y);
	}

	//move mouse while connecting
	if (mm.is_valid() && connecting) {
		connecting_to = mm->get_position();
		connecting_to_node = StringName();
		state_machine_draw->update();

		for (int i = node_rects.size() - 1; i >= 0; i--) { //inverse to draw order
			if (node_rects[i].node_name != connecting_from && node_rects[i].node.has_point(connecting_to)) { //select node since nothing else was selected
				connecting_to_node = node_rects[i].node_name;
				return;
			}
		}
	}

	//move mouse while moving a node
	if (mm.is_valid() && dragging_selected_attempt) {
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

			for (List<StringName>::Element *E = nodes.front(); E; E = E->next()) {
				if (E->get() == selected_node) {
					continue;
				}
				Vector2 npos = state_machine->get_node_position(E->get());

				float d_x = ABS(npos.x - cpos.x);
				if (d_x < MIN(5, best_d_x)) {
					drag_ofs.x -= cpos.x - npos.x;
					best_d_x = d_x;
					snap_x = E->get();
				}

				float d_y = ABS(npos.y - cpos.y);
				if (d_y < MIN(5, best_d_y)) {
					drag_ofs.y -= cpos.y - npos.y;
					best_d_y = d_y;
					snap_y = E->get();
				}
			}
		}

		state_machine_draw->update();
	}

	if (mm.is_valid()) {
		state_machine_draw->grab_focus();

		String new_over_node = StringName();
		int new_over_node_what = -1;
		if (tool_select->is_pressed()) {
			for (int i = node_rects.size() - 1; i >= 0; i--) { // Inverse to draw order.
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
			state_machine_draw->update();
		}
	}

	Ref<InputEventPanGesture> pan_gesture = p_event;
	if (pan_gesture.is_valid()) {
		h_scroll->set_value(h_scroll->get_value() + h_scroll->get_page() * pan_gesture->get_delta().x / 8);
		v_scroll->set_value(v_scroll->get_value() + v_scroll->get_page() * pan_gesture->get_delta().y / 8);
	}
}

Control::CursorShape AnimationNodeStateMachineEditor::get_cursor_shape(const Point2 &p_pos) const {
	// Put ibeam (text cursor) over names to make it clearer that they are editable.
	Transform2D xform = panel->get_transform() * state_machine_draw->get_transform();
	Point2 pos = xform.xform_inv(p_pos);
	Control::CursorShape cursor_shape = get_default_cursor_shape();

	for (int i = node_rects.size() - 1; i >= 0; i--) { // Inverse to draw order.
		if (node_rects[i].node.has_point(pos)) {
			if (node_rects[i].name.has_point(pos)) {
				cursor_shape = Control::CURSOR_IBEAM;
			}
			break;
		}
	}
	return cursor_shape;
}

void AnimationNodeStateMachineEditor::_file_opened(const String &p_file) {
	file_loaded = ResourceLoader::load(p_file);
	if (file_loaded.is_valid()) {
		_add_menu_type(MENU_LOAD_FILE_CONFIRM);
	}
}

void AnimationNodeStateMachineEditor::_add_menu_type(int p_index) {
	String base_name;
	Ref<AnimationRootNode> node;

	if (p_index == MENU_LOAD_FILE) {
		open_file->clear_filters();
		List<String> filters;
		ResourceLoader::get_recognized_extensions_for_type("AnimationRootNode", &filters);
		for (List<String>::Element *E = filters.front(); E; E = E->next()) {
			open_file->add_filter("*." + E->get());
		}
		open_file->popup_centered_ratio();
		return;
	} else if (p_index == MENU_LOAD_FILE_CONFIRM) {
		node = file_loaded;
		file_loaded.unref();
	} else if (p_index == MENU_PASTE) {
		node = EditorSettings::get_singleton()->get_resource_clipboard();

	} else {
		String type = menu->get_item_metadata(p_index);

		Object *obj = ClassDB::instance(type);
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

	if (base_name == String()) {
		base_name = node->get_class().replace_first("AnimationNode", "");
	}

	int base = 1;
	String name = base_name;
	while (state_machine->has_node(name)) {
		base++;
		name = base_name + " " + itos(base);
	}

	updating = true;
	undo_redo->create_action(TTR("Add Node"));
	undo_redo->add_do_method(state_machine.ptr(), "add_node", name, node, add_node_pos);
	undo_redo->add_undo_method(state_machine.ptr(), "remove_node", name);
	undo_redo->add_do_method(this, "_update_graph");
	undo_redo->add_undo_method(this, "_update_graph");
	undo_redo->commit_action();
	updating = false;

	state_machine_draw->update();
}

void AnimationNodeStateMachineEditor::_add_animation_type(int p_index) {
	Ref<AnimationNodeAnimation> anim;
	anim.instance();

	anim->set_animation(animations_to_add[p_index]);

	String base_name = animations_to_add[p_index];
	int base = 1;
	String name = base_name;
	while (state_machine->has_node(name)) {
		base++;
		name = base_name + " " + itos(base);
	}

	updating = true;
	undo_redo->create_action(TTR("Add Node"));
	undo_redo->add_do_method(state_machine.ptr(), "add_node", name, anim, add_node_pos);
	undo_redo->add_undo_method(state_machine.ptr(), "remove_node", name);
	undo_redo->add_do_method(this, "_update_graph");
	undo_redo->add_undo_method(this, "_update_graph");
	undo_redo->commit_action();
	updating = false;

	state_machine_draw->update();
}

void AnimationNodeStateMachineEditor::_connection_draw(const Vector2 &p_from, const Vector2 &p_to, AnimationNodeStateMachineTransition::SwitchMode p_mode, bool p_enabled, bool p_selected, bool p_travel, bool p_auto_advance) {
	Color linecolor = get_color("font_color", "Label");
	Color icon_color(1, 1, 1);
	Color accent = get_color("accent_color", "Editor");

	if (!p_enabled) {
		linecolor.a *= 0.2;
		icon_color.a *= 0.2;
		accent.a *= 0.6;
	}

	Ref<Texture> icons[6] = {
		get_icon("TransitionImmediateBig", "EditorIcons"),
		get_icon("TransitionSyncBig", "EditorIcons"),
		get_icon("TransitionEndBig", "EditorIcons"),
		get_icon("TransitionImmediateAutoBig", "EditorIcons"),
		get_icon("TransitionSyncAutoBig", "EditorIcons"),
		get_icon("TransitionEndAutoBig", "EditorIcons")
	};

	if (p_selected) {
		state_machine_draw->draw_line(p_from, p_to, accent, 6, true);
	}

	if (p_travel) {
		linecolor = accent;
		linecolor.set_hsv(1.0, linecolor.get_s(), linecolor.get_v());
	}
	state_machine_draw->draw_line(p_from, p_to, linecolor, 2, true);

	Ref<Texture> icon = icons[p_mode + (p_auto_advance ? 3 : 0)];

	Transform2D xf;
	xf.elements[0] = (p_to - p_from).normalized();
	xf.elements[1] = xf.elements[0].tangent();
	xf.elements[2] = (p_from + p_to) * 0.5 - xf.elements[1] * icon->get_height() * 0.5 - xf.elements[0] * icon->get_height() * 0.5;

	state_machine_draw->draw_set_transform_matrix(xf);
	state_machine_draw->draw_texture(icon, Vector2(), icon_color);
	state_machine_draw->draw_set_transform_matrix(Transform2D());
}

void AnimationNodeStateMachineEditor::_clip_src_line_to_rect(Vector2 &r_from, Vector2 &r_to, const Rect2 &p_rect) {
	if (r_to == r_from) {
		return;
	}

	//this could be optimized...
	Vector2 n = (r_to - r_from).normalized();
	while (p_rect.has_point(r_from)) {
		r_from += n;
	}
}

void AnimationNodeStateMachineEditor::_clip_dst_line_to_rect(Vector2 &r_from, Vector2 &r_to, const Rect2 &p_rect) {
	if (r_to == r_from) {
		return;
	}

	//this could be optimized...
	Vector2 n = (r_to - r_from).normalized();
	while (p_rect.has_point(r_to)) {
		r_to -= n;
	}
}

void AnimationNodeStateMachineEditor::_state_machine_draw() {
	Ref<AnimationNodeStateMachinePlayback> playback = AnimationTreeEditor::get_singleton()->get_tree()->get(AnimationTreeEditor::get_singleton()->get_base_path() + "playback");

	Ref<StyleBox> style = get_stylebox("state_machine_frame", "GraphNode");
	Ref<StyleBox> style_selected = get_stylebox("state_machine_selectedframe", "GraphNode");

	Ref<Font> font = get_font("title_font", "GraphNode");
	Color font_color = get_color("title_color", "GraphNode");
	Ref<Texture> play = get_icon("Play", "EditorIcons");
	Ref<Texture> auto_play = get_icon("AutoPlay", "EditorIcons");
	Ref<Texture> edit = get_icon("Edit", "EditorIcons");
	Color accent = get_color("accent_color", "Editor");
	Color linecolor = get_color("font_color", "Label");
	linecolor.a *= 0.3;
	Ref<StyleBox> playing_overlay = get_stylebox("position", "GraphNode");

	bool playing = false;
	StringName current;
	StringName blend_from;
	Vector<StringName> travel_path;

	if (playback.is_valid()) {
		playing = playback->is_playing();
		current = playback->get_current_node();
		blend_from = playback->get_blend_from_node();
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
	for (List<StringName>::Element *E = nodes.front(); E; E = E->next()) {
		Ref<AnimationNode> anode = state_machine->get_node(E->get());
		String name = E->get();
		bool needs_editor = EditorNode::get_singleton()->item_has_editor(anode.ptr());
		Ref<StyleBox> sb = E->get() == selected_node ? style_selected : style;

		Size2 s = sb->get_minimum_size();
		int strsize = font->get_string_size(name).width;
		s.width += strsize;
		s.height += MAX(font->get_height(), play->get_height());
		s.width += sep + play->get_width();
		if (needs_editor) {
			s.width += sep + edit->get_width();
		}

		Vector2 offset;
		offset += state_machine->get_node_position(E->get()) * EDSCALE;
		if (selected_node == E->get() && dragging_selected) {
			offset += drag_ofs;
		}
		offset -= s / 2;
		offset = offset.floor();

		//prepre rect

		NodeRect nr;
		nr.node = Rect2(offset, s);
		nr.node_name = E->get();

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

		_connection_draw(from, to, AnimationNodeStateMachineTransition::SwitchMode(transition_mode->get_selected()), true, false, false, false);
	}

	Ref<Texture> tr_reference_icon = get_icon("TransitionImmediateBig", "EditorIcons");
	float tr_bidi_offset = int(tr_reference_icon->get_height() * 0.8);

	//draw transition lines
	for (int i = 0; i < state_machine->get_transition_count(); i++) {
		TransitionLine tl;
		tl.from_node = state_machine->get_transition_from(i);
		Vector2 ofs_from = (dragging_selected && tl.from_node == selected_node) ? drag_ofs : Vector2();
		tl.from = (state_machine->get_node_position(tl.from_node) * EDSCALE) + ofs_from - state_machine->get_graph_offset() * EDSCALE;

		tl.to_node = state_machine->get_transition_to(i);
		Vector2 ofs_to = (dragging_selected && tl.to_node == selected_node) ? drag_ofs : Vector2();
		tl.to = (state_machine->get_node_position(tl.to_node) * EDSCALE) + ofs_to - state_machine->get_graph_offset() * EDSCALE;

		Ref<AnimationNodeStateMachineTransition> tr = state_machine->get_transition(i);
		tl.disabled = tr->is_disabled();
		tl.auto_advance = tr->has_auto_advance();
		tl.advance_condition_name = tr->get_advance_condition_name();
		tl.advance_condition_state = false;
		tl.mode = tr->get_switch_mode();
		tl.width = tr_bidi_offset;

		if (state_machine->has_transition(tl.to_node, tl.from_node)) { //offset if same exists
			Vector2 offset = -(tl.from - tl.to).normalized().tangent() * tr_bidi_offset;
			tl.from += offset;
			tl.to += offset;
		}

		for (int j = 0; j < node_rects.size(); j++) {
			if (node_rects[j].node_name == tl.from_node) {
				_clip_src_line_to_rect(tl.from, tl.to, node_rects[j].node);
			}
			if (node_rects[j].node_name == tl.to_node) {
				_clip_dst_line_to_rect(tl.from, tl.to, node_rects[j].node);
			}
		}

		bool selected = selected_transition_from == tl.from_node && selected_transition_to == tl.to_node;

		bool travel = false;

		if (blend_from == tl.from_node && current == tl.to_node) {
			travel = true;
		}

		if (travel_path.size()) {
			if (current == tl.from_node && travel_path[0] == tl.to_node) {
				travel = true;
			} else {
				for (int j = 0; j < travel_path.size() - 1; j++) {
					if (travel_path[j] == tl.from_node && travel_path[j + 1] == tl.to_node) {
						travel = true;
						break;
					}
				}
			}
		}

		bool auto_advance = tl.auto_advance;
		StringName fullpath = AnimationTreeEditor::get_singleton()->get_base_path() + String(tl.advance_condition_name);
		if (tl.advance_condition_name != StringName() && bool(AnimationTreeEditor::get_singleton()->get_tree()->get(fullpath))) {
			tl.advance_condition_state = true;
			auto_advance = true;
		}
		_connection_draw(tl.from, tl.to, tl.mode, !tl.disabled, selected, travel, auto_advance);

		transition_lines.push_back(tl);
	}

	//draw actual nodes
	for (int i = 0; i < node_rects.size(); i++) {
		String name = node_rects[i].node_name;
		Ref<AnimationNode> anode = state_machine->get_node(name);
		bool needs_editor = AnimationTreeEditor::get_singleton()->can_edit(anode);
		Ref<StyleBox> sb = name == selected_node ? style_selected : style;
		int strsize = font->get_string_size(name).width;

		NodeRect &nr = node_rects.write[i];

		Vector2 offset = nr.node.position;
		int h = nr.node.size.height;

		//prepre rect

		//now scroll it to draw
		state_machine_draw->draw_style_box(sb, nr.node);

		if (playing && (blend_from == name || current == name || travel_path.find(name) != -1)) {
			state_machine_draw->draw_style_box(playing_overlay, nr.node);
		}

		bool onstart = state_machine->get_start_node() == name;
		if (onstart) {
			state_machine_draw->draw_string(font, offset + Vector2(0, -font->get_height() - 3 * EDSCALE + font->get_ascent()), TTR("Start"), font_color);
		}

		if (state_machine->get_end_node() == name) {
			int endofs = nr.node.size.x - font->get_string_size(TTR("End")).x;
			state_machine_draw->draw_string(font, offset + Vector2(endofs, -font->get_height() - 3 * EDSCALE + font->get_ascent()), TTR("End"), font_color);
		}

		offset.x += sb->get_offset().x;

		nr.play.position = offset + Vector2(0, (h - play->get_height()) / 2).floor();
		nr.play.size = play->get_size();

		Ref<Texture> play_tex = onstart ? auto_play : play;

		if (over_node == name && over_node_what == 0) {
			state_machine_draw->draw_texture(play_tex, nr.play.position, accent);
		} else {
			state_machine_draw->draw_texture(play_tex, nr.play.position);
		}
		offset.x += sep + play->get_width();

		nr.name.position = offset + Vector2(0, (h - font->get_height()) / 2).floor();
		nr.name.size = Vector2(strsize, font->get_height());

		state_machine_draw->draw_string(font, nr.name.position + Vector2(0, font->get_ascent()), name, font_color);
		offset.x += strsize + sep;

		if (needs_editor) {
			nr.edit.position = offset + Vector2(0, (h - edit->get_height()) / 2).floor();
			nr.edit.size = edit->get_size();

			if (over_node == name && over_node_what == 1) {
				state_machine_draw->draw_texture(edit, nr.edit.position, accent);
			} else {
				state_machine_draw->draw_texture(edit, nr.edit.position);
			}
			offset.x += sep + edit->get_width();
		}
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

	state_machine_play_pos->update();
}

void AnimationNodeStateMachineEditor::_state_machine_pos_draw() {
	Ref<AnimationNodeStateMachinePlayback> playback = AnimationTreeEditor::get_singleton()->get_tree()->get(AnimationTreeEditor::get_singleton()->get_base_path() + "playback");

	if (!playback.is_valid() || !playback->is_playing()) {
		return;
	}

	int idx = -1;
	for (int i = 0; i < node_rects.size(); i++) {
		if (node_rects[i].node_name == playback->get_current_node()) {
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

	float len = MAX(0.0001, current_length);

	float pos = CLAMP(play_pos, 0, len);
	float c = pos / len;
	Color fg = get_color("font_color", "Label");
	Color bg = fg;
	bg.a *= 0.3;

	state_machine_play_pos->draw_line(from, to, bg, 2);

	to = from.linear_interpolate(to, c);

	state_machine_play_pos->draw_line(from, to, fg, 2);
}

void AnimationNodeStateMachineEditor::_update_graph() {
	if (updating) {
		return;
	}

	updating = true;

	state_machine_draw->update();

	updating = false;
}

void AnimationNodeStateMachineEditor::_notification(int p_what) {
	if (p_what == NOTIFICATION_ENTER_TREE || p_what == NOTIFICATION_THEME_CHANGED) {
		error_panel->add_style_override("panel", get_stylebox("bg", "Tree"));
		error_label->add_color_override("font_color", get_color("error_color", "Editor"));
		panel->add_style_override("panel", get_stylebox("bg", "Tree"));

		tool_select->set_icon(get_icon("ToolSelect", "EditorIcons"));
		tool_create->set_icon(get_icon("ToolAddNode", "EditorIcons"));
		tool_connect->set_icon(get_icon("ToolConnect", "EditorIcons"));

		transition_mode->clear();
		transition_mode->add_icon_item(get_icon("TransitionImmediate", "EditorIcons"), TTR("Immediate"));
		transition_mode->add_icon_item(get_icon("TransitionSync", "EditorIcons"), TTR("Sync"));
		transition_mode->add_icon_item(get_icon("TransitionEnd", "EditorIcons"), TTR("At End"));

		//force filter on those, so they deform better
		get_icon("TransitionImmediateBig", "EditorIcons")->set_flags(Texture::FLAG_FILTER);
		get_icon("TransitionEndBig", "EditorIcons")->set_flags(Texture::FLAG_FILTER);
		get_icon("TransitionSyncBig", "EditorIcons")->set_flags(Texture::FLAG_FILTER);
		get_icon("TransitionImmediateAutoBig", "EditorIcons")->set_flags(Texture::FLAG_FILTER);
		get_icon("TransitionEndAutoBig", "EditorIcons")->set_flags(Texture::FLAG_FILTER);
		get_icon("TransitionSyncAutoBig", "EditorIcons")->set_flags(Texture::FLAG_FILTER);

		tool_erase->set_icon(get_icon("Remove", "EditorIcons"));
		tool_autoplay->set_icon(get_icon("AutoPlay", "EditorIcons"));
		tool_end->set_icon(get_icon("AutoEnd", "EditorIcons"));

		play_mode->clear();
		play_mode->add_icon_item(get_icon("PlayTravel", "EditorIcons"), TTR("Travel"));
		play_mode->add_icon_item(get_icon("Play", "EditorIcons"), TTR("Immediate"));
	}

	if (p_what == NOTIFICATION_PROCESS) {
		String error;

		Ref<AnimationNodeStateMachinePlayback> playback = AnimationTreeEditor::get_singleton()->get_tree()->get(AnimationTreeEditor::get_singleton()->get_base_path() + "playback");

		if (error_time > 0) {
			error = error_text;
			error_time -= get_process_delta_time();
		} else if (!AnimationTreeEditor::get_singleton()->get_tree()->is_active()) {
			error = TTR("AnimationTree is inactive.\nActivate to enable playback, check node warnings if activation fails.");
		} else if (AnimationTreeEditor::get_singleton()->get_tree()->is_state_invalid()) {
			error = AnimationTreeEditor::get_singleton()->get_tree()->get_invalid_state_reason();
			/*} else if (state_machine->get_parent().is_valid() && state_machine->get_parent()->is_class("AnimationNodeStateMachine")) {
			if (state_machine->get_start_node() == StringName() || state_machine->get_end_node() == StringName()) {
				error = TTR("Start and end nodes are needed for a sub-transition.");
			}*/
		} else if (playback.is_null()) {
			error = vformat(TTR("No playback resource set at path: %s."), AnimationTreeEditor::get_singleton()->get_base_path() + "playback");
		}

		if (error != error_label->get_text()) {
			error_label->set_text(error);
			if (error != String()) {
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
				state_machine_draw->update();
				break;
			}

			if (transition_lines[i].disabled != state_machine->get_transition(tidx)->is_disabled()) {
				state_machine_draw->update();
				break;
			}

			if (transition_lines[i].auto_advance != state_machine->get_transition(tidx)->has_auto_advance()) {
				state_machine_draw->update();
				break;
			}

			if (transition_lines[i].advance_condition_name != state_machine->get_transition(tidx)->get_advance_condition_name()) {
				state_machine_draw->update();
				break;
			}

			if (transition_lines[i].mode != state_machine->get_transition(tidx)->get_switch_mode()) {
				state_machine_draw->update();
				break;
			}

			bool acstate = transition_lines[i].advance_condition_name != StringName() && bool(AnimationTreeEditor::get_singleton()->get_tree()->get(AnimationTreeEditor::get_singleton()->get_base_path() + String(transition_lines[i].advance_condition_name)));

			if (transition_lines[i].advance_condition_state != acstate) {
				state_machine_draw->update();
				break;
			}
		}

		bool same_travel_path = true;
		Vector<StringName> tp;
		bool is_playing = false;
		StringName current_node;
		StringName blend_from_node;
		play_pos = 0;
		current_length = 0;

		if (playback.is_valid()) {
			tp = playback->get_travel_path();
			is_playing = playback->is_playing();
			current_node = playback->get_current_node();
			blend_from_node = playback->get_blend_from_node();
			play_pos = playback->get_current_play_pos();
			current_length = playback->get_current_length();
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

		//update if travel state changed
		if (!same_travel_path || last_active != is_playing || last_current_node != current_node || last_blend_from_node != blend_from_node) {
			state_machine_draw->update();
			last_travel_path = tp;
			last_current_node = current_node;
			last_active = is_playing;
			last_blend_from_node = blend_from_node;
			state_machine_play_pos->update();
		}

		{
			if (current_node != StringName() && state_machine->has_node(current_node)) {
				String next = current_node;
				Ref<AnimationNodeStateMachine> anodesm = state_machine->get_node(next);
				Ref<AnimationNodeStateMachinePlayback> current_node_playback;

				while (anodesm.is_valid()) {
					current_node_playback = AnimationTreeEditor::get_singleton()->get_tree()->get(AnimationTreeEditor::get_singleton()->get_base_path() + next + "/playback");
					next += "/" + current_node_playback->get_current_node();
					anodesm = anodesm->get_node(current_node_playback->get_current_node());
				}

				// when current_node is a state machine, use playback of current_node to set play_pos
				if (current_node_playback.is_valid()) {
					play_pos = current_node_playback->get_current_play_pos();
					current_length = current_node_playback->get_current_length();
				}
			}
		}

		if (last_play_pos != play_pos) {
			last_play_pos = play_pos;
			state_machine_play_pos->update();
		}
	}

	if (p_what == NOTIFICATION_VISIBILITY_CHANGED) {
		over_node = StringName();
		set_process(is_visible_in_tree());
	}
}

void AnimationNodeStateMachineEditor::_open_editor(const String &p_name) {
	AnimationTreeEditor::get_singleton()->enter_editor(p_name);
}

void AnimationNodeStateMachineEditor::_removed_from_graph() {
	EditorNode::get_singleton()->edit_item(nullptr);
}

void AnimationNodeStateMachineEditor::_name_edited(const String &p_text) {
	const String &new_name = p_text;

	ERR_FAIL_COND(new_name == "" || new_name.find(".") != -1 || new_name.find("/") != -1);

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
	undo_redo->create_action(TTR("Node Renamed"));
	undo_redo->add_do_method(state_machine.ptr(), "rename_node", prev_name, name);
	undo_redo->add_undo_method(state_machine.ptr(), "rename_node", name, prev_name);
	undo_redo->add_do_method(this, "_update_graph");
	undo_redo->add_undo_method(this, "_update_graph");
	undo_redo->commit_action();
	name_edit->hide();
	updating = false;

	state_machine_draw->update();
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
	state_machine_draw->update();
}

void AnimationNodeStateMachineEditor::_erase_selected() {
	if (selected_node != StringName() && state_machine->has_node(selected_node)) {
		updating = true;
		undo_redo->create_action(TTR("Node Removed"));
		undo_redo->add_do_method(state_machine.ptr(), "remove_node", selected_node);
		undo_redo->add_undo_method(state_machine.ptr(), "add_node", selected_node, state_machine->get_node(selected_node), state_machine->get_node_position(selected_node));
		for (int i = 0; i < state_machine->get_transition_count(); i++) {
			String from = state_machine->get_transition_from(i);
			String to = state_machine->get_transition_to(i);
			if (from == selected_node || to == selected_node) {
				undo_redo->add_undo_method(state_machine.ptr(), "add_transition", from, to, state_machine->get_transition(i));
			}
		}
		if (String(state_machine->get_start_node()) == selected_node) {
			undo_redo->add_undo_method(state_machine.ptr(), "set_start_node", selected_node);
		}
		undo_redo->add_do_method(this, "_update_graph");
		undo_redo->add_undo_method(this, "_update_graph");
		undo_redo->commit_action();
		updating = false;
		selected_node = StringName();
	}

	if (selected_transition_to != StringName() && selected_transition_from != StringName() && state_machine->has_transition(selected_transition_from, selected_transition_to)) {
		Ref<AnimationNodeStateMachineTransition> tr = state_machine->get_transition(state_machine->find_transition(selected_transition_from, selected_transition_to));
		updating = true;
		undo_redo->create_action(TTR("Transition Removed"));
		undo_redo->add_do_method(state_machine.ptr(), "remove_transition", selected_transition_from, selected_transition_to);
		undo_redo->add_undo_method(state_machine.ptr(), "add_transition", selected_transition_from, selected_transition_to, tr);
		undo_redo->add_do_method(this, "_update_graph");
		undo_redo->add_undo_method(this, "_update_graph");
		undo_redo->commit_action();
		updating = false;
		selected_transition_from = StringName();
		selected_transition_to = StringName();
	}

	state_machine_draw->update();
}

void AnimationNodeStateMachineEditor::_autoplay_selected() {
	if (selected_node != StringName() && state_machine->has_node(selected_node)) {
		StringName new_start_node;
		if (state_machine->get_start_node() == selected_node) { //toggle it
			new_start_node = StringName();
		} else {
			new_start_node = selected_node;
		}

		updating = true;
		undo_redo->create_action(TTR("Set Start Node (Autoplay)"));
		undo_redo->add_do_method(state_machine.ptr(), "set_start_node", new_start_node);
		undo_redo->add_undo_method(state_machine.ptr(), "set_start_node", state_machine->get_start_node());
		undo_redo->add_do_method(this, "_update_graph");
		undo_redo->add_undo_method(this, "_update_graph");
		undo_redo->commit_action();
		updating = false;
		state_machine_draw->update();
	}
}

void AnimationNodeStateMachineEditor::_end_selected() {
	if (selected_node != StringName() && state_machine->has_node(selected_node)) {
		StringName new_end_node;
		if (state_machine->get_end_node() == selected_node) { //toggle it
			new_end_node = StringName();
		} else {
			new_end_node = selected_node;
		}

		updating = true;
		undo_redo->create_action(TTR("Set Start Node (Autoplay)"));
		undo_redo->add_do_method(state_machine.ptr(), "set_end_node", new_end_node);
		undo_redo->add_undo_method(state_machine.ptr(), "set_end_node", state_machine->get_end_node());
		undo_redo->add_do_method(this, "_update_graph");
		undo_redo->add_undo_method(this, "_update_graph");
		undo_redo->commit_action();
		updating = false;
		state_machine_draw->update();
	}
}
void AnimationNodeStateMachineEditor::_update_mode() {
	if (tool_select->is_pressed()) {
		tool_erase_hb->show();
		tool_erase->set_disabled(selected_node == StringName() && selected_transition_from == StringName() && selected_transition_to == StringName());
		tool_autoplay->set_disabled(selected_node == StringName());
		tool_end->set_disabled(selected_node == StringName());
	} else {
		tool_erase_hb->hide();
	}
}

void AnimationNodeStateMachineEditor::_bind_methods() {
	ClassDB::bind_method("_state_machine_gui_input", &AnimationNodeStateMachineEditor::_state_machine_gui_input);
	ClassDB::bind_method("_state_machine_draw", &AnimationNodeStateMachineEditor::_state_machine_draw);
	ClassDB::bind_method("_state_machine_pos_draw", &AnimationNodeStateMachineEditor::_state_machine_pos_draw);
	ClassDB::bind_method("_update_graph", &AnimationNodeStateMachineEditor::_update_graph);

	ClassDB::bind_method("_add_menu_type", &AnimationNodeStateMachineEditor::_add_menu_type);
	ClassDB::bind_method("_add_animation_type", &AnimationNodeStateMachineEditor::_add_animation_type);

	ClassDB::bind_method("_name_edited", &AnimationNodeStateMachineEditor::_name_edited);
	ClassDB::bind_method("_name_edited_focus_out", &AnimationNodeStateMachineEditor::_name_edited_focus_out);

	ClassDB::bind_method("_removed_from_graph", &AnimationNodeStateMachineEditor::_removed_from_graph);

	ClassDB::bind_method("_open_editor", &AnimationNodeStateMachineEditor::_open_editor);
	ClassDB::bind_method("_scroll_changed", &AnimationNodeStateMachineEditor::_scroll_changed);

	ClassDB::bind_method("_erase_selected", &AnimationNodeStateMachineEditor::_erase_selected);
	ClassDB::bind_method("_autoplay_selected", &AnimationNodeStateMachineEditor::_autoplay_selected);
	ClassDB::bind_method("_end_selected", &AnimationNodeStateMachineEditor::_end_selected);
	ClassDB::bind_method("_update_mode", &AnimationNodeStateMachineEditor::_update_mode);
	ClassDB::bind_method("_file_opened", &AnimationNodeStateMachineEditor::_file_opened);
}

AnimationNodeStateMachineEditor *AnimationNodeStateMachineEditor::singleton = nullptr;

AnimationNodeStateMachineEditor::AnimationNodeStateMachineEditor() {
	singleton = this;
	updating = false;

	HBoxContainer *top_hb = memnew(HBoxContainer);
	add_child(top_hb);

	Ref<ButtonGroup> bg;
	bg.instance();

	tool_select = memnew(ToolButton);
	top_hb->add_child(tool_select);
	tool_select->set_toggle_mode(true);
	tool_select->set_button_group(bg);
	tool_select->set_pressed(true);
	tool_select->set_tooltip(TTR("Select and move nodes.\nRMB to add new nodes.\nShift+LMB to create connections."));
	tool_select->connect("pressed", this, "_update_mode", varray(), CONNECT_DEFERRED);

	tool_create = memnew(ToolButton);
	top_hb->add_child(tool_create);
	tool_create->set_toggle_mode(true);
	tool_create->set_button_group(bg);
	tool_create->set_tooltip(TTR("Create new nodes."));
	tool_create->connect("pressed", this, "_update_mode", varray(), CONNECT_DEFERRED);

	tool_connect = memnew(ToolButton);
	top_hb->add_child(tool_connect);
	tool_connect->set_toggle_mode(true);
	tool_connect->set_button_group(bg);
	tool_connect->set_tooltip(TTR("Connect nodes."));
	tool_connect->connect("pressed", this, "_update_mode", varray(), CONNECT_DEFERRED);

	tool_erase_hb = memnew(HBoxContainer);
	top_hb->add_child(tool_erase_hb);
	tool_erase_hb->add_child(memnew(VSeparator));
	tool_erase = memnew(ToolButton);
	tool_erase->set_tooltip(TTR("Remove selected node or transition."));
	tool_erase_hb->add_child(tool_erase);
	tool_erase->connect("pressed", this, "_erase_selected");
	tool_erase->set_disabled(true);

	tool_erase_hb->add_child(memnew(VSeparator));

	tool_autoplay = memnew(ToolButton);
	tool_autoplay->set_tooltip(TTR("Toggle autoplay this animation on start, restart or seek to zero."));
	tool_erase_hb->add_child(tool_autoplay);
	tool_autoplay->connect("pressed", this, "_autoplay_selected");
	tool_autoplay->set_disabled(true);

	tool_end = memnew(ToolButton);
	tool_end->set_tooltip(TTR("Set the end animation. This is useful for sub-transitions."));
	tool_erase_hb->add_child(tool_end);
	tool_end->connect("pressed", this, "_end_selected");
	tool_end->set_disabled(true);

	top_hb->add_child(memnew(VSeparator));
	top_hb->add_child(memnew(Label(TTR("Transition: "))));
	transition_mode = memnew(OptionButton);
	top_hb->add_child(transition_mode);

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
	state_machine_draw->connect("gui_input", this, "_state_machine_gui_input");
	state_machine_draw->connect("draw", this, "_state_machine_draw");
	state_machine_draw->set_focus_mode(FOCUS_ALL);
	state_machine_draw->set_mouse_filter(Control::MOUSE_FILTER_PASS);

	state_machine_play_pos = memnew(Control);
	state_machine_draw->add_child(state_machine_play_pos);
	state_machine_play_pos->set_mouse_filter(MOUSE_FILTER_PASS); //pass all to parent
	state_machine_play_pos->set_anchors_and_margins_preset(PRESET_WIDE);
	state_machine_play_pos->connect("draw", this, "_state_machine_pos_draw");

	v_scroll = memnew(VScrollBar);
	state_machine_draw->add_child(v_scroll);
	v_scroll->set_anchors_and_margins_preset(PRESET_RIGHT_WIDE);
	v_scroll->connect("value_changed", this, "_scroll_changed");

	h_scroll = memnew(HScrollBar);
	state_machine_draw->add_child(h_scroll);
	h_scroll->set_anchors_and_margins_preset(PRESET_BOTTOM_WIDE);
	h_scroll->set_margin(MARGIN_RIGHT, -v_scroll->get_size().x * EDSCALE);
	h_scroll->connect("value_changed", this, "_scroll_changed");

	error_panel = memnew(PanelContainer);
	add_child(error_panel);
	error_label = memnew(Label);
	error_panel->add_child(error_label);
	error_panel->hide();

	undo_redo = EditorNode::get_undo_redo();

	set_custom_minimum_size(Size2(0, 300 * EDSCALE));

	menu = memnew(PopupMenu);
	add_child(menu);
	menu->connect("id_pressed", this, "_add_menu_type");

	animations_menu = memnew(PopupMenu);
	menu->add_child(animations_menu);
	animations_menu->set_name("animations");
	animations_menu->connect("index_pressed", this, "_add_animation_type");

	name_edit = memnew(LineEdit);
	state_machine_draw->add_child(name_edit);
	name_edit->hide();
	name_edit->connect("text_entered", this, "_name_edited");
	name_edit->connect("focus_exited", this, "_name_edited_focus_out");
	name_edit->set_as_toplevel(true);

	open_file = memnew(EditorFileDialog);
	add_child(open_file);
	open_file->set_title(TTR("Open Animation Node"));
	open_file->set_mode(EditorFileDialog::MODE_OPEN_FILE);
	open_file->connect("file_selected", this, "_file_opened");
	undo_redo = EditorNode::get_undo_redo();

	over_node_what = -1;
	dragging_selected_attempt = false;
	connecting = false;

	last_active = false;

	error_time = 0;
}
