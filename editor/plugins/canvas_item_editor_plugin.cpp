/*************************************************************************/
/*  canvas_item_editor_plugin.cpp                                        */
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
#include "canvas_item_editor_plugin.h"

#include "editor/animation_editor.h"
#include "editor/editor_node.h"
#include "editor/editor_settings.h"
#include "editor/plugins/animation_player_editor_plugin.h"
#include "editor/plugins/script_editor_plugin.h"
#include "editor/script_editor_debugger.h"
#include "os/input.h"
#include "os/keyboard.h"
#include "print_string.h"
#include "project_settings.h"
#include "scene/2d/light_2d.h"
#include "scene/2d/particles_2d.h"
#include "scene/2d/polygon_2d.h"
#include "scene/2d/screen_button.h"
#include "scene/2d/sprite.h"
#include "scene/gui/grid_container.h"
#include "scene/gui/nine_patch_rect.h"
#include "scene/main/canvas_layer.h"
#include "scene/main/viewport.h"
#include "scene/resources/packed_scene.h"

#define MIN_ZOOM 0.01
#define MAX_ZOOM 100

#define RULER_WIDTH 15 * EDSCALE

class SnapDialog : public ConfirmationDialog {

	GDCLASS(SnapDialog, ConfirmationDialog);

	friend class CanvasItemEditor;

	SpinBox *grid_offset_x;
	SpinBox *grid_offset_y;
	SpinBox *grid_step_x;
	SpinBox *grid_step_y;
	SpinBox *rotation_offset;
	SpinBox *rotation_step;

public:
	SnapDialog() :
			ConfirmationDialog() {
		const int SPIN_BOX_GRID_RANGE = 256;
		const int SPIN_BOX_ROTATION_RANGE = 360;
		Label *label;
		VBoxContainer *container;
		GridContainer *child_container;

		set_title(TTR("Configure Snap"));
		get_ok()->set_text(TTR("Close"));

		container = memnew(VBoxContainer);
		add_child(container);
		//set_child_rect(container);

		child_container = memnew(GridContainer);
		child_container->set_columns(3);
		container->add_child(child_container);

		label = memnew(Label);
		label->set_text(TTR("Grid Offset:"));
		child_container->add_child(label);
		label->set_h_size_flags(SIZE_EXPAND_FILL);

		grid_offset_x = memnew(SpinBox);
		grid_offset_x->set_min(-SPIN_BOX_GRID_RANGE);
		grid_offset_x->set_max(SPIN_BOX_GRID_RANGE);
		grid_offset_x->set_suffix("px");
		child_container->add_child(grid_offset_x);

		grid_offset_y = memnew(SpinBox);
		grid_offset_y->set_min(-SPIN_BOX_GRID_RANGE);
		grid_offset_y->set_max(SPIN_BOX_GRID_RANGE);
		grid_offset_y->set_suffix("px");
		child_container->add_child(grid_offset_y);

		label = memnew(Label);
		label->set_text(TTR("Grid Step:"));
		child_container->add_child(label);
		label->set_h_size_flags(SIZE_EXPAND_FILL);

		grid_step_x = memnew(SpinBox);
		grid_step_x->set_min(0.01);
		grid_step_x->set_max(SPIN_BOX_GRID_RANGE);
		grid_step_x->set_suffix("px");
		child_container->add_child(grid_step_x);

		grid_step_y = memnew(SpinBox);
		grid_step_y->set_min(0.01);
		grid_step_y->set_max(SPIN_BOX_GRID_RANGE);
		grid_step_y->set_suffix("px");
		child_container->add_child(grid_step_y);

		container->add_child(memnew(HSeparator));

		child_container = memnew(GridContainer);
		child_container->set_columns(2);
		container->add_child(child_container);

		label = memnew(Label);
		label->set_text(TTR("Rotation Offset:"));
		child_container->add_child(label);
		label->set_h_size_flags(SIZE_EXPAND_FILL);

		rotation_offset = memnew(SpinBox);
		rotation_offset->set_min(-SPIN_BOX_ROTATION_RANGE);
		rotation_offset->set_max(SPIN_BOX_ROTATION_RANGE);
		rotation_offset->set_suffix("deg");
		child_container->add_child(rotation_offset);

		label = memnew(Label);
		label->set_text(TTR("Rotation Step:"));
		child_container->add_child(label);
		label->set_h_size_flags(SIZE_EXPAND_FILL);

		rotation_step = memnew(SpinBox);
		rotation_step->set_min(-SPIN_BOX_ROTATION_RANGE);
		rotation_step->set_max(SPIN_BOX_ROTATION_RANGE);
		rotation_step->set_suffix("deg");
		child_container->add_child(rotation_step);
	}

	void set_fields(const Point2 p_grid_offset, const Point2 p_grid_step, const float p_rotation_offset, const float p_rotation_step) {
		grid_offset_x->set_value(p_grid_offset.x);
		grid_offset_y->set_value(p_grid_offset.y);
		grid_step_x->set_value(p_grid_step.x);
		grid_step_y->set_value(p_grid_step.y);
		rotation_offset->set_value(p_rotation_offset * (180 / Math_PI));
		rotation_step->set_value(p_rotation_step * (180 / Math_PI));
	}

	void get_fields(Point2 &p_grid_offset, Point2 &p_grid_step, float &p_rotation_offset, float &p_rotation_step) {
		p_grid_offset = Point2(grid_offset_x->get_value(), grid_offset_y->get_value());
		p_grid_step = Point2(grid_step_x->get_value(), grid_step_y->get_value());
		p_rotation_offset = rotation_offset->get_value() / (180 / Math_PI);
		p_rotation_step = rotation_step->get_value() / (180 / Math_PI);
	}
};

void CanvasItemEditor::_edit_set_pivot(const Vector2 &mouse_pos) {
	List<Node *> &selection = editor_selection->get_selected_node_list();

	undo_redo->create_action(TTR("Move Pivot"));

	for (List<Node *>::Element *E = selection.front(); E; E = E->next()) {

		Node2D *n2d = Object::cast_to<Node2D>(E->get());
		if (n2d && n2d->_edit_use_pivot()) {

			Vector2 offset = n2d->_edit_get_pivot();
			Vector2 gpos = n2d->get_global_position();

			Vector2 local_mouse_pos = n2d->get_canvas_transform().affine_inverse().xform(mouse_pos);

			Vector2 motion_ofs = gpos - local_mouse_pos;

			undo_redo->add_do_method(n2d, "set_global_position", local_mouse_pos);
			undo_redo->add_do_method(n2d, "_edit_set_pivot", offset + n2d->get_global_transform().affine_inverse().basis_xform(motion_ofs));
			undo_redo->add_undo_method(n2d, "set_global_position", gpos);
			undo_redo->add_undo_method(n2d, "_edit_set_pivot", offset);
			for (int i = 0; i < n2d->get_child_count(); i++) {
				Node2D *n2dc = Object::cast_to<Node2D>(n2d->get_child(i));
				if (!n2dc)
					continue;

				undo_redo->add_do_method(n2dc, "set_global_position", n2dc->get_global_position());
				undo_redo->add_undo_method(n2dc, "set_global_position", n2dc->get_global_position());
			}
		}

		Control *cnt = Object::cast_to<Control>(E->get());
		if (cnt) {

			Vector2 old_pivot = cnt->get_pivot_offset();
			Vector2 new_pivot = cnt->get_global_transform_with_canvas().affine_inverse().xform(mouse_pos);
			Vector2 old_pos = cnt->get_position();

			Vector2 top_pos = cnt->get_transform().get_origin(); //remember where top pos was
			cnt->set_pivot_offset(new_pivot);
			Vector2 new_top_pos = cnt->get_transform().get_origin(); //check where it is now

			Vector2 new_pos = old_pos - (new_top_pos - top_pos); //offset it back

			undo_redo->add_do_method(cnt, "set_pivot_offset", new_pivot);
			undo_redo->add_do_method(cnt, "set_position", new_pos);
			undo_redo->add_undo_method(cnt, "set_pivot_offset", old_pivot);
			undo_redo->add_undo_method(cnt, "set_position", old_pos);
		}
	}

	undo_redo->commit_action();
}

void CanvasItemEditor::_snap_if_closer_float(float p_value, float p_target_snap, float &r_current_snap, bool &r_snapped, float p_radius) {
	float radius = p_radius / zoom;
	float dist = Math::abs(p_value - p_target_snap);
	if (p_radius < 0 || dist < radius && (!r_snapped || dist < Math::abs(r_current_snap - p_value))) {
		r_current_snap = p_target_snap;
		r_snapped = true;
	}
}

void CanvasItemEditor::_snap_if_closer_point(Point2 p_value, Point2 p_target_snap, Point2 &r_current_snap, bool (&r_snapped)[2], real_t rotation, float p_radius) {
	Transform2D rot_trans = Transform2D(rotation, Point2());
	p_value = rot_trans.inverse().xform(p_value);
	p_target_snap = rot_trans.inverse().xform(p_target_snap);
	r_current_snap = rot_trans.inverse().xform(r_current_snap);

	_snap_if_closer_float(p_value.x, p_target_snap.x, r_current_snap.x, r_snapped[0], p_radius);
	_snap_if_closer_float(p_value.y, p_target_snap.y, r_current_snap.y, r_snapped[1], p_radius);

	r_current_snap = rot_trans.xform(r_current_snap);
}

void CanvasItemEditor::_snap_other_nodes(Point2 p_value, Point2 &r_current_snap, bool (&r_snapped)[2], const Node *p_current, const CanvasItem *p_to_snap) {
	const CanvasItem *canvas_item = Object::cast_to<CanvasItem>(p_current);
	if (canvas_item && (!p_to_snap || p_current != p_to_snap)) {
		Transform2D ci_transform = canvas_item->get_global_transform_with_canvas();
		Transform2D to_snap_transform = p_to_snap ? p_to_snap->get_global_transform_with_canvas() : Transform2D();
		if (fmod(ci_transform.get_rotation() - to_snap_transform.get_rotation(), (real_t)360.0) == 0.0) {
			Point2 begin = ci_transform.xform(canvas_item->_edit_get_rect().get_position());
			Point2 end = ci_transform.xform(canvas_item->_edit_get_rect().get_position() + canvas_item->_edit_get_rect().get_size());

			_snap_if_closer_point(p_value, begin, r_current_snap, r_snapped, ci_transform.get_rotation());
			_snap_if_closer_point(p_value, end, r_current_snap, r_snapped, ci_transform.get_rotation());
		}
	}
	for (int i = 0; i < p_current->get_child_count(); i++) {
		_snap_other_nodes(p_value, r_current_snap, r_snapped, p_current->get_child(i), p_to_snap);
	}
}

Point2 CanvasItemEditor::snap_point(Point2 p_target, unsigned int p_modes, const CanvasItem *p_canvas_item, unsigned int p_forced_modes) {
	Point2 dist[2];
	bool snapped[2] = { false, false };

	// Smart snap using the canvas position
	Vector2 output = p_target;
	real_t rotation = 0.0;

	if (p_canvas_item) {
		Point2 begin;
		Point2 end;
		rotation = p_canvas_item->get_global_transform_with_canvas().get_rotation();

		if ((snap_active && snap_node_parent && (p_modes & SNAP_NODE_PARENT)) || (p_forced_modes & SNAP_NODE_PARENT)) {
			// Parent sides and center
			bool can_snap = false;
			if (const Control *c = Object::cast_to<Control>(p_canvas_item)) {
				begin = p_canvas_item->get_global_transform_with_canvas().xform(_anchor_to_position(c, Point2(0, 0)));
				end = p_canvas_item->get_global_transform_with_canvas().xform(_anchor_to_position(c, Point2(1, 1)));
				can_snap = true;
			} else if (const CanvasItem *parent_ci = Object::cast_to<CanvasItem>(p_canvas_item->get_parent())) {
				begin = p_canvas_item->get_transform().affine_inverse().xform(parent_ci->_edit_get_rect().get_position());
				end = p_canvas_item->get_transform().affine_inverse().xform(parent_ci->_edit_get_rect().get_position() + parent_ci->_edit_get_rect().get_size());
				can_snap = true;
			}

			if (can_snap) {
				_snap_if_closer_point(p_target, begin, output, snapped, rotation);
				_snap_if_closer_point(p_target, (begin + end) / 2.0, output, snapped, rotation);
				_snap_if_closer_point(p_target, end, output, snapped, rotation);
			}
		}

		// Self anchors (for sides)
		if ((snap_active && snap_node_anchors && (p_modes & SNAP_NODE_ANCHORS)) || (p_forced_modes & SNAP_NODE_ANCHORS)) {
			if (const Control *c = Object::cast_to<Control>(p_canvas_item)) {
				begin = p_canvas_item->get_global_transform_with_canvas().xform(_anchor_to_position(c, Point2(c->get_anchor(MARGIN_LEFT), c->get_anchor(MARGIN_TOP))));
				end = p_canvas_item->get_global_transform_with_canvas().xform(_anchor_to_position(c, Point2(c->get_anchor(MARGIN_RIGHT), c->get_anchor(MARGIN_BOTTOM))));
				_snap_if_closer_point(p_target, begin, output, snapped, rotation);
				_snap_if_closer_point(p_target, end, output, snapped, rotation);
			}
		}

		// Self sides (for anchors)
		if ((snap_active && snap_node_sides && (p_modes & SNAP_NODE_SIDES)) || (p_forced_modes & SNAP_NODE_SIDES)) {
			begin = p_canvas_item->get_global_transform_with_canvas().xform(p_canvas_item->_edit_get_rect().get_position());
			end = p_canvas_item->get_global_transform_with_canvas().xform(p_canvas_item->_edit_get_rect().get_position() + p_canvas_item->_edit_get_rect().get_size());
			_snap_if_closer_point(p_target, begin, output, snapped, rotation);
			_snap_if_closer_point(p_target, end, output, snapped, rotation);
		}
	}

	// Other nodes sides
	if ((snap_active && snap_other_nodes && (p_modes & SNAP_OTHER_NODES)) || (p_forced_modes & SNAP_OTHER_NODES)) {
		_snap_other_nodes(p_target, output, snapped, get_tree()->get_edited_scene_root(), p_canvas_item);
	}

	if (((snap_active && snap_guides && (p_modes & SNAP_GUIDES)) || (p_forced_modes & SNAP_GUIDES)) && fmod(rotation, (real_t)360.0) == 0.0) {
		// Guides
		if (EditorNode::get_singleton()->get_edited_scene() && EditorNode::get_singleton()->get_edited_scene()->has_meta("_edit_vertical_guides_")) {
			Array vguides = EditorNode::get_singleton()->get_edited_scene()->get_meta("_edit_vertical_guides_");
			for (int i = 0; i < vguides.size(); i++) {
				_snap_if_closer_float(p_target.x, vguides[i], output.x, snapped[0]);
			}
		}

		if (EditorNode::get_singleton()->get_edited_scene() && EditorNode::get_singleton()->get_edited_scene()->has_meta("_edit_horizontal_guides_")) {
			Array hguides = EditorNode::get_singleton()->get_edited_scene()->get_meta("_edit_horizontal_guides_");
			for (int i = 0; i < hguides.size(); i++) {
				_snap_if_closer_float(p_target.y, hguides[i], output.y, snapped[1]);
			}
		}
	}

	if (((snap_active && snap_grid && (p_modes & SNAP_GRID)) || (p_forced_modes & SNAP_GRID)) && fmod(rotation, (real_t)360.0) == 0.0) {
		// Grid
		Point2 offset = grid_offset;
		if (snap_relative) {
			List<Node *> &selection = editor_selection->get_selected_node_list();
			if (selection.size() == 1 && Object::cast_to<Node2D>(selection[0])) {
				offset = Object::cast_to<Node2D>(selection[0])->get_global_position();
			} else {
				offset = _find_topleftmost_point();
			}
		}
		Point2 grid_output;
		grid_output.x = Math::stepify(p_target.x - offset.x, grid_step.x * Math::pow(2.0, grid_step_multiplier)) + offset.x;
		grid_output.y = Math::stepify(p_target.y - offset.y, grid_step.y * Math::pow(2.0, grid_step_multiplier)) + offset.y;
		_snap_if_closer_point(p_target, grid_output, output, snapped, 0.0, -1.0);
	}

	if (((snap_pixel && (p_modes & SNAP_PIXEL)) || (p_forced_modes & SNAP_PIXEL)) && rotation == 0.0) {
		// Pixel
		output = output.snapped(Size2(1, 1));
	}

	return output;
}

float CanvasItemEditor::snap_angle(float p_target, float p_start) const {
	float offset = snap_relative ? p_start : p_target;
	return (snap_rotation && snap_rotation_step != 0) ? Math::stepify(p_target - snap_rotation_offset, snap_rotation_step) + snap_rotation_offset : p_target;
}

void CanvasItemEditor::_unhandled_key_input(const Ref<InputEvent> &p_ev) {

	Ref<InputEventKey> k = p_ev;

	if (!is_visible_in_tree() || get_viewport()->gui_has_modal_stack())
		return;

	if (k->get_control())
		return;

	if (k->is_pressed() && !k->is_echo()) {
		if (drag_pivot_shortcut.is_valid() && drag_pivot_shortcut->is_shortcut(p_ev) && drag == DRAG_NONE && can_move_pivot) {
			//move drag pivot
			drag = DRAG_PIVOT;
		} else if (set_pivot_shortcut.is_valid() && set_pivot_shortcut->is_shortcut(p_ev) && drag == DRAG_NONE && can_move_pivot) {
			if (!Input::get_singleton()->is_mouse_button_pressed(0)) {
				List<Node *> &selection = editor_selection->get_selected_node_list();
				Vector2 mouse_pos = viewport->get_local_mouse_position();
				if (selection.size() && viewport->get_rect().has_point(mouse_pos)) {
					//just in case, make it work if over viewport
					mouse_pos = transform.affine_inverse().xform(mouse_pos);
					mouse_pos = snap_point(mouse_pos, SNAP_DEFAULT, _get_single_item());

					_edit_set_pivot(mouse_pos);
				}
			}
		} else if ((snap_grid || show_grid) && multiply_grid_step_shortcut.is_valid() && multiply_grid_step_shortcut->is_shortcut(p_ev)) {
			// Multiply the grid size
			grid_step_multiplier = MIN(grid_step_multiplier + 1, 12);
			viewport_base->update();
			viewport->update();
		} else if ((snap_grid || show_grid) && divide_grid_step_shortcut.is_valid() && divide_grid_step_shortcut->is_shortcut(p_ev)) {
			// Divide the grid size
			Point2 new_grid_step = grid_step * Math::pow(2.0, grid_step_multiplier - 1);
			if (new_grid_step.x >= 1.0 && new_grid_step.y >= 1.0)
				grid_step_multiplier--;
			viewport_base->update();
			viewport->update();
		}
	}
}

void CanvasItemEditor::_tool_select(int p_index) {

	ToolButton *tb[TOOL_MAX] = { select_button, list_select_button, move_button, rotate_button, pivot_button, pan_button };
	for (int i = 0; i < TOOL_MAX; i++) {
		tb[i]->set_pressed(i == p_index);
	}

	viewport->update();
	tool = (Tool)p_index;
}

Object *CanvasItemEditor::_get_editor_data(Object *p_what) {

	CanvasItem *ci = Object::cast_to<CanvasItem>(p_what);
	if (!ci)
		return NULL;

	return memnew(CanvasItemEditorSelectedItem);
}

Dictionary CanvasItemEditor::get_state() const {

	Dictionary state;
	state["zoom"] = zoom;
	state["ofs"] = Point2(h_scroll->get_value(), v_scroll->get_value());
	//state["ofs"]=-transform.get_origin();
	state["grid_offset"] = grid_offset;
	state["grid_step"] = grid_step;
	state["snap_rotation_offset"] = snap_rotation_offset;
	state["snap_rotation_step"] = snap_rotation_step;
	state["snap_active"] = snap_active;
	state["snap_node_parent"] = snap_node_parent;
	state["snap_node_anchors"] = snap_node_anchors;
	state["snap_node_sides"] = snap_node_sides;
	state["snap_other_nodes"] = snap_other_nodes;
	state["snap_grid"] = snap_grid;
	state["snap_guides"] = snap_guides;
	state["show_grid"] = show_grid;
	state["show_rulers"] = show_rulers;
	state["show_guides"] = show_guides;
	state["show_helpers"] = show_helpers;
	state["snap_rotation"] = snap_rotation;
	state["snap_relative"] = snap_relative;
	state["snap_pixel"] = snap_pixel;
	state["skeleton_show_bones"] = skeleton_show_bones;
	return state;
}
void CanvasItemEditor::set_state(const Dictionary &p_state) {

	Dictionary state = p_state;
	if (state.has("zoom")) {
		zoom = p_state["zoom"];
	}

	if (state.has("ofs")) {
		_update_scrollbars(); // i wonder how safe is calling this here..
		Point2 ofs = p_state["ofs"];
		h_scroll->set_value(ofs.x);
		v_scroll->set_value(ofs.y);
	}

	if (state.has("grid_offset")) {
		grid_offset = state["grid_offset"];
	}

	if (state.has("grid_step")) {
		grid_step = state["grid_step"];
	}

	if (state.has("snap_rotation_step")) {
		snap_rotation_step = state["snap_rotation_step"];
	}

	if (state.has("snap_rotation_offset")) {
		snap_rotation_offset = state["snap_rotation_offset"];
	}

	if (state.has("snap_active")) {
		snap_active = state["snap_active"];
		snap_button->set_pressed(snap_active);
	}

	if (state.has("snap_node_parent")) {
		snap_node_parent = state["snap_node_parent"];
		int idx = smartsnap_config_popup->get_item_index(SNAP_USE_NODE_PARENT);
		smartsnap_config_popup->set_item_checked(idx, snap_node_parent);
	}

	if (state.has("snap_node_anchors")) {
		snap_node_anchors = state["snap_node_anchors"];
		int idx = smartsnap_config_popup->get_item_index(SNAP_USE_NODE_ANCHORS);
		smartsnap_config_popup->set_item_checked(idx, snap_node_anchors);
	}

	if (state.has("snap_node_sides")) {
		snap_node_sides = state["snap_node_sides"];
		int idx = smartsnap_config_popup->get_item_index(SNAP_USE_NODE_SIDES);
		smartsnap_config_popup->set_item_checked(idx, snap_node_sides);
	}

	if (state.has("snap_other_nodes")) {
		snap_other_nodes = state["snap_other_nodes"];
		int idx = smartsnap_config_popup->get_item_index(SNAP_USE_OTHER_NODES);
		smartsnap_config_popup->set_item_checked(idx, snap_other_nodes);
	}

	if (state.has("snap_guides")) {
		snap_guides = state["snap_guides"];
		int idx = smartsnap_config_popup->get_item_index(SNAP_USE_GUIDES);
		smartsnap_config_popup->set_item_checked(idx, snap_guides);
	}

	if (state.has("snap_grid")) {
		snap_grid = state["snap_grid"];
		int idx = snap_config_menu->get_popup()->get_item_index(SNAP_USE_GRID);
		snap_config_menu->get_popup()->set_item_checked(idx, snap_grid);
	}

	if (state.has("show_grid")) {
		show_grid = state["show_grid"];
		int idx = view_menu->get_popup()->get_item_index(SHOW_GRID);
		view_menu->get_popup()->set_item_checked(idx, show_grid);
	}

	if (state.has("show_rulers")) {
		show_rulers = state["show_rulers"];
		int idx = view_menu->get_popup()->get_item_index(SHOW_RULERS);
		view_menu->get_popup()->set_item_checked(idx, show_rulers);
	}

	if (state.has("show_guides")) {
		show_guides = state["show_guides"];
		int idx = view_menu->get_popup()->get_item_index(SHOW_GUIDES);
		view_menu->get_popup()->set_item_checked(idx, show_guides);
	}

	if (state.has("show_helpers")) {
		show_helpers = state["show_helpers"];
		int idx = view_menu->get_popup()->get_item_index(SHOW_HELPERS);
		view_menu->get_popup()->set_item_checked(idx, show_helpers);
	}

	if (state.has("snap_rotation")) {
		snap_rotation = state["snap_rotation"];
		int idx = snap_config_menu->get_popup()->get_item_index(SNAP_USE_ROTATION);
		snap_config_menu->get_popup()->set_item_checked(idx, snap_rotation);
	}

	if (state.has("snap_relative")) {
		snap_relative = state["snap_relative"];
		int idx = snap_config_menu->get_popup()->get_item_index(SNAP_RELATIVE);
		snap_config_menu->get_popup()->set_item_checked(idx, snap_relative);
	}

	if (state.has("snap_pixel")) {
		snap_pixel = state["snap_pixel"];
		int idx = snap_config_menu->get_popup()->get_item_index(SNAP_USE_PIXEL);
		snap_config_menu->get_popup()->set_item_checked(idx, snap_pixel);
	}

	if (state.has("skeleton_show_bones")) {
		skeleton_show_bones = state["skeleton_show_bones"];
		int idx = skeleton_menu->get_popup()->get_item_index(SKELETON_SHOW_BONES);
		skeleton_menu->get_popup()->set_item_checked(idx, skeleton_show_bones);
	}

	viewport->update();
}

void CanvasItemEditor::_add_canvas_item(CanvasItem *p_canvas_item) {

	editor_selection->add_node(p_canvas_item);
}

void CanvasItemEditor::_remove_canvas_item(CanvasItem *p_canvas_item) {

	editor_selection->remove_node(p_canvas_item);
}
void CanvasItemEditor::_clear_canvas_items() {

	editor_selection->clear();
}

void CanvasItemEditor::_keying_changed() {

	if (AnimationPlayerEditor::singleton->get_key_editor()->is_visible_in_tree())
		animation_hb->show();
	else
		animation_hb->hide();
}

bool CanvasItemEditor::_is_part_of_subscene(CanvasItem *p_item) {

	Node *scene_node = get_tree()->get_edited_scene_root();
	Node *item_owner = p_item->get_owner();

	return item_owner && item_owner != scene_node && p_item != scene_node && item_owner->get_filename() != "";
}

void CanvasItemEditor::_find_canvas_items_at_pos(const Point2 &p_pos, Node *p_node, const Transform2D &p_parent_xform, const Transform2D &p_canvas_xform, Vector<_SelectResult> &r_items, int limit) {
	if (!p_node)
		return;
	if (Object::cast_to<Viewport>(p_node))
		return;

	CanvasItem *c = Object::cast_to<CanvasItem>(p_node);

	for (int i = p_node->get_child_count() - 1; i >= 0; i--) {

		if (c && !c->is_set_as_toplevel())
			_find_canvas_items_at_pos(p_pos, p_node->get_child(i), p_parent_xform * c->get_transform(), p_canvas_xform, r_items);
		else {
			CanvasLayer *cl = Object::cast_to<CanvasLayer>(p_node);
			_find_canvas_items_at_pos(p_pos, p_node->get_child(i), transform, cl ? cl->get_transform() : p_canvas_xform, r_items); //use base transform
		}

		if (limit != 0 && r_items.size() >= limit)
			return;
	}

	if (c && c->is_visible_in_tree() && !c->has_meta("_edit_lock_") && !Object::cast_to<CanvasLayer>(c)) {

		Rect2 rect = c->_edit_get_rect();
		Point2 local_pos = (p_parent_xform * p_canvas_xform * c->get_transform()).affine_inverse().xform(p_pos);

		if (rect.has_point(local_pos)) {
			Node2D *node = Object::cast_to<Node2D>(c);

			_SelectResult res;
			res.item = c;
			res.z = node ? node->get_z() : 0;
			res.has_z = node;
			r_items.push_back(res);
		}
	}

	return;
}

void CanvasItemEditor::_find_canvas_items_at_rect(const Rect2 &p_rect, Node *p_node, const Transform2D &p_parent_xform, const Transform2D &p_canvas_xform, List<CanvasItem *> *r_items) {

	if (!p_node)
		return;
	if (Object::cast_to<Viewport>(p_node))
		return;

	CanvasItem *c = Object::cast_to<CanvasItem>(p_node);

	bool inherited = p_node != get_tree()->get_edited_scene_root() && p_node->get_filename() != "";
	bool editable = false;
	if (inherited) {
		editable = EditorNode::get_singleton()->get_edited_scene()->is_editable_instance(p_node);
	}
	bool lock_children = p_node->has_meta("_edit_group_") && p_node->get_meta("_edit_group_");
	if (!lock_children && (!inherited || editable)) {
		for (int i = p_node->get_child_count() - 1; i >= 0; i--) {

			if (c && !c->is_set_as_toplevel())
				_find_canvas_items_at_rect(p_rect, p_node->get_child(i), p_parent_xform * c->get_transform(), p_canvas_xform, r_items);
			else {
				CanvasLayer *cl = Object::cast_to<CanvasLayer>(p_node);
				_find_canvas_items_at_rect(p_rect, p_node->get_child(i), transform, cl ? cl->get_transform() : p_canvas_xform, r_items);
			}
		}
	}

	if (c && c->is_visible_in_tree() && !c->has_meta("_edit_lock_") && !Object::cast_to<CanvasLayer>(c)) {

		Rect2 rect = c->_edit_get_rect();
		Transform2D xform = p_parent_xform * p_canvas_xform * c->get_transform();

		if (p_rect.has_point(xform.xform(rect.position)) &&
				p_rect.has_point(xform.xform(rect.position + Vector2(rect.size.x, 0))) &&
				p_rect.has_point(xform.xform(rect.position + Vector2(rect.size.x, rect.size.y))) &&
				p_rect.has_point(xform.xform(rect.position + Vector2(0, rect.size.y)))) {

			r_items->push_back(c);
		}
	}
}

void CanvasItemEditor::_select_click_on_empty_area(Point2 p_click_pos, bool p_append, bool p_box_selection) {
	if (!p_append) {
		editor_selection->clear();
		viewport->update();
		viewport_base->update();
	};

	if (p_box_selection) {
		// Start a box selection
		drag_from = transform.affine_inverse().xform(p_click_pos);
		box_selecting = true;
		box_selecting_to = drag_from;
	}
}

bool CanvasItemEditor::_select_click_on_item(CanvasItem *item, Point2 p_click_pos, bool p_append, bool p_drag) {
	bool still_selected = true;
	if (p_append) {
		if (editor_selection->is_selected(item)) {
			// Already in the selection, remove it from the selected nodes
			editor_selection->remove_node(item);
			still_selected = false;
		} else {
			// Add the item to the selection
			_append_canvas_item(item);
		}
	} else {
		if (!editor_selection->is_selected(item)) {
			// Select a new one and clear previous selection
			editor_selection->clear();
			editor_selection->add_node(item);
			// Reselect
			if (Engine::get_singleton()->is_editor_hint()) {
				editor->call("edit_node", item);
			}
		}
	}

	if (still_selected && p_drag) {
		// Drag the node(s) if requested
		_prepare_drag(p_click_pos);
	}

	viewport->update();
	viewport_base->update();
	return still_selected;
}

void CanvasItemEditor::_key_move(const Vector2 &p_dir, bool p_snap, KeyMoveMODE p_move_mode) {

	if (drag != DRAG_NONE)
		return;

	if (editor_selection->get_selected_node_list().empty())
		return;

	undo_redo->create_action(TTR("Move Action"), UndoRedo::MERGE_ENDS);

	List<Node *> &selection = editor_selection->get_selected_node_list();

	for (List<Node *>::Element *E = selection.front(); E; E = E->next()) {

		CanvasItem *canvas_item = Object::cast_to<CanvasItem>(E->get());
		if (!canvas_item || !canvas_item->is_visible_in_tree())
			continue;
		if (canvas_item->get_viewport() != EditorNode::get_singleton()->get_scene_root())
			continue;

		CanvasItemEditorSelectedItem *se = editor_selection->get_node_editor_data<CanvasItemEditorSelectedItem>(canvas_item);
		if (!se)
			continue;

		if (canvas_item->has_meta("_edit_lock_"))
			continue;

		Vector2 drag = p_dir;
		if (p_snap)
			drag *= grid_step * Math::pow(2.0, grid_step_multiplier);

		undo_redo->add_undo_method(canvas_item, "_edit_set_state", canvas_item->_edit_get_state());

		if (p_move_mode == MOVE_VIEW_BASE) {

			// drag =  transform.affine_inverse().basis_xform(p_dir); // zoom sensitive
			drag = canvas_item->get_global_transform_with_canvas().affine_inverse().basis_xform(drag);
			Rect2 local_rect = canvas_item->_edit_get_rect();
			local_rect.position += drag;
			undo_redo->add_do_method(canvas_item, "_edit_set_rect", local_rect);

		} else { // p_move_mode==MOVE_LOCAL_BASE || p_move_mode==MOVE_LOCAL_WITH_ROT

			Node2D *node_2d = Object::cast_to<Node2D>(canvas_item);
			if (node_2d) {

				if (p_move_mode == MOVE_LOCAL_WITH_ROT) {
					Transform2D m;
					m.rotate(node_2d->get_rotation());
					drag = m.xform(drag);
				}
				node_2d->set_position(node_2d->get_position() + drag);

			} else {
				Control *control = Object::cast_to<Control>(canvas_item);
				if (control)
					control->set_position(control->get_position() + drag);
			}
		}
	}

	undo_redo->commit_action();
}

Point2 CanvasItemEditor::_find_topleftmost_point() {

	Vector2 tl = Point2(1e10, 1e10);
	Rect2 r2;
	r2.position = tl;

	List<Node *> &selection = editor_selection->get_selected_node_list();

	for (List<Node *>::Element *E = selection.front(); E; E = E->next()) {

		CanvasItem *canvas_item = Object::cast_to<CanvasItem>(E->get());
		if (!canvas_item || !canvas_item->is_visible_in_tree())
			continue;
		if (canvas_item->get_viewport() != EditorNode::get_singleton()->get_scene_root())
			continue;

		Rect2 rect = canvas_item->_edit_get_rect();
		Transform2D xform = canvas_item->get_global_transform_with_canvas();

		r2.expand_to(xform.xform(rect.position));
		r2.expand_to(xform.xform(rect.position + Vector2(rect.size.x, 0)));
		r2.expand_to(xform.xform(rect.position + rect.size));
		r2.expand_to(xform.xform(rect.position + Vector2(0, rect.size.y)));
	}

	return r2.position;
}

int CanvasItemEditor::get_item_count() {

	List<Node *> &selection = editor_selection->get_selected_node_list();

	int ic = 0;
	for (List<Node *>::Element *E = selection.front(); E; E = E->next()) {

		CanvasItem *canvas_item = Object::cast_to<CanvasItem>(E->get());
		if (!canvas_item || !canvas_item->is_visible_in_tree())
			continue;

		if (canvas_item->get_viewport() != EditorNode::get_singleton()->get_scene_root())
			continue;

		ic++;
	};

	return ic;
}

CanvasItem *CanvasItemEditor::_get_single_item() {

	Map<Node *, Object *> &selection = editor_selection->get_selection();

	CanvasItem *single_item = NULL;

	for (Map<Node *, Object *>::Element *E = selection.front(); E; E = E->next()) {

		CanvasItem *canvas_item = Object::cast_to<CanvasItem>(E->key());
		if (!canvas_item || !canvas_item->is_visible_in_tree())
			continue;
		if (canvas_item->get_viewport() != EditorNode::get_singleton()->get_scene_root())
			continue;

		if (single_item)
			return NULL; //morethan one

		single_item = canvas_item;
	};

	return single_item;
}

CanvasItemEditor::DragType CanvasItemEditor::_get_resize_handle_drag_type(const Point2 &p_click, Vector2 &r_point) {
	// Returns a drag type if a resize handle is clicked
	CanvasItem *canvas_item = _get_single_item();

	ERR_FAIL_COND_V(!canvas_item, DRAG_NONE);

	Rect2 rect = canvas_item->_edit_get_rect();
	Transform2D xforml = canvas_item->get_global_transform_with_canvas();
	Transform2D xform = transform * xforml;

	Vector2 endpoints[4] = {

		xform.xform(rect.position),
		xform.xform(rect.position + Vector2(rect.size.x, 0)),
		xform.xform(rect.position + rect.size),
		xform.xform(rect.position + Vector2(0, rect.size.y))
	};

	Vector2 endpointsl[4] = {

		xforml.xform(rect.position),
		xforml.xform(rect.position + Vector2(rect.size.x, 0)),
		xforml.xform(rect.position + rect.size),
		xforml.xform(rect.position + Vector2(0, rect.size.y))
	};

	DragType dragger[] = {
		DRAG_TOP_LEFT,
		DRAG_TOP,
		DRAG_TOP_RIGHT,
		DRAG_RIGHT,
		DRAG_BOTTOM_RIGHT,
		DRAG_BOTTOM,
		DRAG_BOTTOM_LEFT,
		DRAG_LEFT
	};

	float radius = (select_handle->get_size().width / 2) * 1.5;

	for (int i = 0; i < 4; i++) {

		int prev = (i + 3) % 4;
		int next = (i + 1) % 4;

		r_point = endpointsl[i];

		Vector2 ofs = ((endpoints[i] - endpoints[prev]).normalized() + ((endpoints[i] - endpoints[next]).normalized())).normalized();
		ofs *= 1.4144 * (select_handle->get_size().width / 2);

		ofs += endpoints[i];

		if (ofs.distance_to(p_click) < radius)
			return dragger[i * 2];

		ofs = (endpoints[i] + endpoints[next]) / 2;
		ofs += (endpoints[next] - endpoints[i]).tangent().normalized() * (select_handle->get_size().width / 2);

		r_point = (endpointsl[i] + endpointsl[next]) / 2;

		if (ofs.distance_to(p_click) < radius)
			return dragger[i * 2 + 1];
	}

	return DRAG_NONE;
}

Vector2 CanvasItemEditor::_anchor_to_position(const Control *p_control, Vector2 anchor) {
	ERR_FAIL_COND_V(!p_control, Vector2());

	Transform2D parent_transform = p_control->get_transform().affine_inverse();
	Size2 parent_size = p_control->get_parent_area_size();

	return parent_transform.xform(Vector2(parent_size.x * anchor.x, parent_size.y * anchor.y));
}

Vector2 CanvasItemEditor::_position_to_anchor(const Control *p_control, Vector2 position) {
	ERR_FAIL_COND_V(!p_control, Vector2());
	Size2 parent_size = p_control->get_parent_area_size();

	return p_control->get_transform().xform(position) / parent_size;
}

CanvasItemEditor::DragType CanvasItemEditor::_get_anchor_handle_drag_type(const Point2 &p_click, Vector2 &r_point) {
	// Returns a drag type if an anchor handle is clicked
	CanvasItem *canvas_item = _get_single_item();
	ERR_FAIL_COND_V(!canvas_item, DRAG_NONE);

	Control *control = Object::cast_to<Control>(canvas_item);
	ERR_FAIL_COND_V(!control, DRAG_NONE);

	Vector2 anchor_pos[4];
	anchor_pos[0] = Vector2(control->get_anchor(MARGIN_LEFT), control->get_anchor(MARGIN_TOP));
	anchor_pos[1] = Vector2(control->get_anchor(MARGIN_RIGHT), control->get_anchor(MARGIN_TOP));
	anchor_pos[2] = Vector2(control->get_anchor(MARGIN_RIGHT), control->get_anchor(MARGIN_BOTTOM));
	anchor_pos[3] = Vector2(control->get_anchor(MARGIN_LEFT), control->get_anchor(MARGIN_BOTTOM));

	Rect2 anchor_rects[4];
	for (int i = 0; i < 4; i++) {
		anchor_pos[i] = (transform * control->get_global_transform_with_canvas()).xform(_anchor_to_position(control, anchor_pos[i]));
		anchor_rects[i] = Rect2(anchor_pos[i], anchor_handle->get_size());
		anchor_rects[i].position -= anchor_handle->get_size() * Vector2(i == 0 || i == 3, i <= 1);
	}

	DragType dragger[] = {
		DRAG_ANCHOR_TOP_LEFT,
		DRAG_ANCHOR_TOP_RIGHT,
		DRAG_ANCHOR_BOTTOM_RIGHT,
		DRAG_ANCHOR_BOTTOM_LEFT,
	};

	for (int i = 0; i < 4; i++) {
		if (anchor_rects[i].has_point(p_click)) {
			r_point = transform.affine_inverse().xform(anchor_pos[i]);
			if ((anchor_pos[0] == anchor_pos[2]) && (anchor_pos[0].distance_to(p_click) < anchor_handle->get_size().length() / 3.0)) {
				return DRAG_ANCHOR_ALL;
			} else {
				return dragger[i];
			}
		}
	}

	return DRAG_NONE;
}

void CanvasItemEditor::_prepare_drag(const Point2 &p_click_pos) {

	List<Node *> &selection = editor_selection->get_selected_node_list();

	for (List<Node *>::Element *E = selection.front(); E; E = E->next()) {

		CanvasItem *canvas_item = Object::cast_to<CanvasItem>(E->get());
		if (!canvas_item || !canvas_item->is_visible_in_tree())
			continue;
		if (canvas_item->get_viewport() != EditorNode::get_singleton()->get_scene_root())
			continue;

		CanvasItemEditorSelectedItem *se = editor_selection->get_node_editor_data<CanvasItemEditorSelectedItem>(canvas_item);
		if (!se)
			continue;

		se->undo_state = canvas_item->_edit_get_state();
		if (Object::cast_to<Node2D>(canvas_item))
			se->undo_pivot = Object::cast_to<Node2D>(canvas_item)->_edit_get_pivot();
		if (Object::cast_to<Control>(canvas_item))
			se->undo_pivot = Object::cast_to<Control>(canvas_item)->get_pivot_offset();

		se->pre_drag_xform = canvas_item->get_global_transform_with_canvas();
		se->pre_drag_rect = canvas_item->_edit_get_rect();
	}

	if (selection.size() == 1 && Object::cast_to<Node2D>(selection[0]) && bone_ik_list.size() == 0) {
		drag = DRAG_NODE_2D;
		drag_point_from = Object::cast_to<Node2D>(selection[0])->get_global_position();
	} else {
		drag = DRAG_ALL;
		drag_point_from = _find_topleftmost_point();
	}
	drag_from = transform.affine_inverse().xform(p_click_pos);
}

void CanvasItemEditor::incbeg(float &beg, float &end, float inc, float minsize, bool p_symmetric) {

	if (minsize < 0) {

		beg += inc;
		if (p_symmetric)
			end -= inc;
	} else {

		if (p_symmetric) {
			beg += inc;
			end -= inc;
			if (end - beg < minsize) {
				float center = (beg + end) / 2.0;
				beg = center - minsize / 2.0;
				end = center + minsize / 2.0;
			}

		} else {
			if (end - (beg + inc) < minsize)
				beg = end - minsize;
			else
				beg += inc;
		}
	}
}

void CanvasItemEditor::incend(float &beg, float &end, float inc, float minsize, bool p_symmetric) {

	if (minsize < 0) {

		end += inc;
		if (p_symmetric)
			beg -= inc;
	} else {

		if (p_symmetric) {

			end += inc;
			beg -= inc;
			if (end - beg < minsize) {
				float center = (beg + end) / 2.0;
				beg = center - minsize / 2.0;
				end = center + minsize / 2.0;
			}

		} else {
			if ((end + inc) - beg < minsize)
				end = beg + minsize;
			else
				end += inc;
		}
	}
}

void CanvasItemEditor::_append_canvas_item(CanvasItem *p_item) {

	editor_selection->add_node(p_item);
}

void CanvasItemEditor::_snap_changed() {
	((SnapDialog *)snap_dialog)->get_fields(grid_offset, grid_step, snap_rotation_offset, snap_rotation_step);
	grid_step_multiplier = 0;
	viewport_base->update();
	viewport->update();
}

void CanvasItemEditor::_selection_result_pressed(int p_result) {

	if (selection_results.size() <= p_result)
		return;

	CanvasItem *item = selection_results[p_result].item;

	if (item)
		_select_click_on_item(item, Point2(), additive_selection, false);
}

void CanvasItemEditor::_selection_menu_hide() {

	selection_results.clear();
	selection_menu->clear();
	selection_menu->set_size(Vector2(0, 0));
}

void CanvasItemEditor::_list_select(const Ref<InputEventMouseButton> &b) {

	Point2 click = viewport_scrollable->get_transform().affine_inverse().xform(b->get_position());

	Node *scene = editor->get_edited_scene();
	if (!scene)
		return;

	_find_canvas_items_at_pos(click, scene, transform, Transform2D(), selection_results);

	for (int i = 0; i < selection_results.size(); i++) {
		CanvasItem *item = selection_results[i].item;
		if (item != scene && item->get_owner() != scene && !scene->is_editable_instance(item->get_owner())) {
			//invalid result
			selection_results.remove(i);
			i--;
		}
	}

	if (selection_results.size() == 1) {

		CanvasItem *item = selection_results[0].item;
		selection_results.clear();

		additive_selection = b->get_shift();

		if (!_select_click_on_item(item, click, additive_selection, false))
			return;

	} else if (!selection_results.empty()) {

		selection_results.sort();

		NodePath root_path = get_tree()->get_edited_scene_root()->get_path();
		StringName root_name = root_path.get_name(root_path.get_name_count() - 1);

		for (int i = 0; i < selection_results.size(); i++) {

			CanvasItem *item = selection_results[i].item;

			Ref<Texture> icon;
			if (item->has_meta("_editor_icon"))
				icon = item->get_meta("_editor_icon");
			else
				icon = get_icon(has_icon(item->get_class(), "EditorIcons") ? item->get_class() : String("Object"), "EditorIcons");

			String node_path = "/" + root_name + "/" + root_path.rel_path_to(item->get_path());

			selection_menu->add_item(item->get_name());
			selection_menu->set_item_icon(i, icon);
			selection_menu->set_item_metadata(i, node_path);
			selection_menu->set_item_tooltip(i, String(item->get_name()) + "\nType: " + item->get_class() + "\nPath: " + node_path);
		}

		additive_selection = b->get_shift();

		selection_menu->set_global_position(b->get_global_position());
		selection_menu->popup();
		selection_menu->call_deferred("grab_click_focus");
		selection_menu->set_invalidate_click_until_motion();

		return;
	}
}

void CanvasItemEditor::_update_cursor() {

	CursorShape c = CURSOR_ARROW;
	switch (drag) {
		case DRAG_NONE:
			if (Input::get_singleton()->is_mouse_button_pressed(BUTTON_MIDDLE) || Input::get_singleton()->is_key_pressed(KEY_SPACE)) {
				c = CURSOR_DRAG;
			} else {
				switch (tool) {
					case TOOL_MOVE:
						c = CURSOR_MOVE;
						break;
					case TOOL_EDIT_PIVOT:
						c = CURSOR_CROSS;
						break;
					case TOOL_PAN:
						c = CURSOR_DRAG;
						break;
				}
			}
			break;
		case DRAG_LEFT:
		case DRAG_RIGHT:
			c = CURSOR_HSIZE;
			break;
		case DRAG_TOP:
		case DRAG_BOTTOM:
			c = CURSOR_VSIZE;
			break;
		case DRAG_TOP_LEFT:
		case DRAG_BOTTOM_RIGHT:
			c = CURSOR_FDIAGSIZE;
			break;
		case DRAG_TOP_RIGHT:
		case DRAG_BOTTOM_LEFT:
			c = CURSOR_BDIAGSIZE;
			break;
		case DRAG_ALL:
		case DRAG_NODE_2D:
			c = CURSOR_MOVE;
			break;
	}
	viewport->set_default_cursor_shape(c);
}

void CanvasItemEditor::_gui_input_viewport_base(const Ref<InputEvent> &p_event) {

	Ref<InputEventMouseButton> b = p_event;
	if (b.is_valid()) {
		if (b->get_button_index() == BUTTON_LEFT && b->is_pressed()) {
			if (show_guides && show_rulers && EditorNode::get_singleton()->get_edited_scene()) {
				Transform2D xform = viewport_scrollable->get_transform() * transform;
				// Retreive the guide lists
				Array vguides;
				if (EditorNode::get_singleton()->get_edited_scene()->has_meta("_edit_vertical_guides_")) {
					vguides = EditorNode::get_singleton()->get_edited_scene()->get_meta("_edit_vertical_guides_");
				}
				Array hguides;
				if (EditorNode::get_singleton()->get_edited_scene()->has_meta("_edit_horizontal_guides_")) {
					hguides = EditorNode::get_singleton()->get_edited_scene()->get_meta("_edit_horizontal_guides_");
				}

				// Press button
				if (b->get_position().x < RULER_WIDTH && b->get_position().y < RULER_WIDTH) {
					// Drag a new double guide
					drag = DRAG_DOUBLE_GUIDE;
					edited_guide_index = -1;
				} else if (b->get_position().x < RULER_WIDTH) {
					// Check if we drag an existing horizontal guide
					float minimum = 1e20;
					edited_guide_index = -1;
					for (int i = 0; i < hguides.size(); i++) {
						if (ABS(xform.xform(Point2(0, hguides[i])).y - b->get_position().y) < MIN(minimum, 8)) {
							edited_guide_index = i;
						}
					}

					if (edited_guide_index >= 0) {
						// Drag an existing horizontal guide
						drag = DRAG_H_GUIDE;
					} else {
						// Drag a new vertical guide
						drag = DRAG_V_GUIDE;
					}
				} else if (b->get_position().y < RULER_WIDTH) {
					// Check if we drag an existing vertical guide
					float minimum = 1e20;
					edited_guide_index = -1;
					for (int i = 0; i < vguides.size(); i++) {
						if (ABS(xform.xform(Point2(vguides[i], 0)).x - b->get_position().x) < MIN(minimum, 8)) {
							edited_guide_index = i;
						}
					}

					if (edited_guide_index >= 0) {
						// Drag an existing vertical guide
						drag = DRAG_V_GUIDE;
					} else {
						// Drag a new vertical guide
						drag = DRAG_H_GUIDE;
					}
				}
			}
		}

		if (b->get_button_index() == BUTTON_LEFT && !b->is_pressed()) {
			// Release button
			if (show_guides && EditorNode::get_singleton()->get_edited_scene()) {
				Transform2D xform = viewport_scrollable->get_transform() * transform;

				// Retreive the guide lists
				Array vguides;
				if (EditorNode::get_singleton()->get_edited_scene()->has_meta("_edit_vertical_guides_")) {
					vguides = EditorNode::get_singleton()->get_edited_scene()->get_meta("_edit_vertical_guides_");
				}
				Array hguides;
				if (EditorNode::get_singleton()->get_edited_scene()->has_meta("_edit_horizontal_guides_")) {
					hguides = EditorNode::get_singleton()->get_edited_scene()->get_meta("_edit_horizontal_guides_");
				}

				Point2 edited = snap_point(xform.affine_inverse().xform(b->get_position()), SNAP_GRID | SNAP_PIXEL | SNAP_OTHER_NODES);
				if (drag == DRAG_V_GUIDE) {
					Array prev_vguides = vguides.duplicate();
					if (b->get_position().x > RULER_WIDTH) {
						// Adds a new vertical guide
						if (edited_guide_index >= 0) {
							vguides[edited_guide_index] = edited.x;
							undo_redo->create_action(TTR("Move vertical guide"));
							undo_redo->add_do_method(EditorNode::get_singleton()->get_edited_scene(), "set_meta", "_edit_vertical_guides_", vguides);
							undo_redo->add_undo_method(EditorNode::get_singleton()->get_edited_scene(), "set_meta", "_edit_vertical_guides_", prev_vguides);
							undo_redo->add_undo_method(viewport_base, "update");
							undo_redo->commit_action();
						} else {
							vguides.push_back(edited.x);
							undo_redo->create_action(TTR("Create new vertical guide"));
							undo_redo->add_do_method(EditorNode::get_singleton()->get_edited_scene(), "set_meta", "_edit_vertical_guides_", vguides);
							undo_redo->add_undo_method(EditorNode::get_singleton()->get_edited_scene(), "set_meta", "_edit_vertical_guides_", prev_vguides);
							undo_redo->add_undo_method(viewport_base, "update");
							undo_redo->commit_action();
						}
					} else {
						if (edited_guide_index >= 0) {
							vguides.remove(edited_guide_index);
							undo_redo->create_action(TTR("Remove vertical guide"));
							undo_redo->add_do_method(EditorNode::get_singleton()->get_edited_scene(), "set_meta", "_edit_vertical_guides_", vguides);
							undo_redo->add_undo_method(EditorNode::get_singleton()->get_edited_scene(), "set_meta", "_edit_vertical_guides_", prev_vguides);
							undo_redo->add_undo_method(viewport_base, "update");
							undo_redo->commit_action();
						}
					}
				} else if (drag == DRAG_H_GUIDE) {
					Array prev_hguides = hguides.duplicate();
					if (b->get_position().y > RULER_WIDTH) {
						// Adds a new horizontal guide
						if (edited_guide_index >= 0) {
							hguides[edited_guide_index] = edited.y;
							undo_redo->create_action(TTR("Move horizontal guide"));
							undo_redo->add_do_method(EditorNode::get_singleton()->get_edited_scene(), "set_meta", "_edit_horizontal_guides_", hguides);
							undo_redo->add_undo_method(EditorNode::get_singleton()->get_edited_scene(), "set_meta", "_edit_horizontal_guides_", prev_hguides);
							undo_redo->add_undo_method(viewport_base, "update");
							undo_redo->commit_action();
						} else {
							hguides.push_back(edited.y);
							undo_redo->create_action(TTR("Create new horizontal guide"));
							undo_redo->add_do_method(EditorNode::get_singleton()->get_edited_scene(), "set_meta", "_edit_horizontal_guides_", hguides);
							undo_redo->add_undo_method(EditorNode::get_singleton()->get_edited_scene(), "set_meta", "_edit_horizontal_guides_", prev_hguides);
							undo_redo->add_undo_method(viewport_base, "update");
							undo_redo->commit_action();
						}
					} else {
						if (edited_guide_index >= 0) {
							hguides.remove(edited_guide_index);
							undo_redo->create_action(TTR("Remove horizontal guide"));
							undo_redo->add_do_method(EditorNode::get_singleton()->get_edited_scene(), "set_meta", "_edit_horizontal_guides_", hguides);
							undo_redo->add_undo_method(EditorNode::get_singleton()->get_edited_scene(), "set_meta", "_edit_horizontal_guides_", prev_hguides);
							undo_redo->add_undo_method(viewport_base, "update");
							undo_redo->commit_action();
						}
					}
				} else if (drag == DRAG_DOUBLE_GUIDE) {
					Array prev_hguides = hguides.duplicate();
					Array prev_vguides = vguides.duplicate();
					if (b->get_position().x > RULER_WIDTH && b->get_position().y > RULER_WIDTH) {
						// Adds a new horizontal guide a new vertical guide
						vguides.push_back(edited.x);
						hguides.push_back(edited.y);
						undo_redo->create_action(TTR("Create new horizontal and vertical guides"));
						undo_redo->add_do_method(EditorNode::get_singleton()->get_edited_scene(), "set_meta", "_edit_vertical_guides_", vguides);
						undo_redo->add_do_method(EditorNode::get_singleton()->get_edited_scene(), "set_meta", "_edit_horizontal_guides_", hguides);
						undo_redo->add_undo_method(EditorNode::get_singleton()->get_edited_scene(), "set_meta", "_edit_vertical_guides_", prev_vguides);
						undo_redo->add_undo_method(EditorNode::get_singleton()->get_edited_scene(), "set_meta", "_edit_horizontal_guides_", prev_hguides);
						undo_redo->add_undo_method(viewport_base, "update");
						undo_redo->commit_action();
					}
				}
			}
			if (drag == DRAG_DOUBLE_GUIDE || drag == DRAG_V_GUIDE || drag == DRAG_H_GUIDE) {
				drag = DRAG_NONE;
				viewport_base->update();
			}
		}
	}

	Ref<InputEventMouseMotion> m = p_event;
	if (m.is_valid()) {
		if (!viewport_base->has_focus() && (!get_focus_owner() || !get_focus_owner()->is_text_field())) {
			viewport_base->call_deferred("grab_focus");
		}
		if (drag == DRAG_DOUBLE_GUIDE || drag == DRAG_H_GUIDE || drag == DRAG_V_GUIDE) {
			Transform2D xform = viewport_scrollable->get_transform() * transform;
			Point2 mouse_pos = m->get_position();
			mouse_pos = xform.affine_inverse().xform(mouse_pos);
			mouse_pos = xform.xform(snap_point(mouse_pos, SNAP_GRID | SNAP_PIXEL | SNAP_OTHER_NODES));

			edited_guide_pos = mouse_pos;
			viewport_base->update();
		}
	}

	Ref<InputEventKey> k = p_event;
	if (k.is_valid()) {
		if (k->is_pressed() && drag == DRAG_NONE) {
			// Move the object with the arrow keys
			KeyMoveMODE move_mode = MOVE_VIEW_BASE;
			if (k->get_alt()) move_mode = MOVE_LOCAL_BASE;
			if (k->get_control() || k->get_metakey()) move_mode = MOVE_LOCAL_WITH_ROT;

			if (k->get_scancode() == KEY_UP)
				_key_move(Vector2(0, -1), k->get_shift(), move_mode);
			else if (k->get_scancode() == KEY_DOWN)
				_key_move(Vector2(0, 1), k->get_shift(), move_mode);
			else if (k->get_scancode() == KEY_LEFT)
				_key_move(Vector2(-1, 0), k->get_shift(), move_mode);
			else if (k->get_scancode() == KEY_RIGHT)
				_key_move(Vector2(1, 0), k->get_shift(), move_mode);
			else if (k->get_scancode() == KEY_ESCAPE) {
				editor_selection->clear();
				viewport->update();
			} else
				return;

			accept_event();
		}
	}
}

void CanvasItemEditor::_gui_input_viewport(const Ref<InputEvent> &p_event) {

	{
		EditorNode *en = editor;
		EditorPluginList *over_plugin_list = en->get_editor_plugins_over();

		if (!over_plugin_list->empty()) {
			bool discard = over_plugin_list->forward_gui_input(p_event);
			if (discard) {
				accept_event();
				return;
			}
		}
	}

	Ref<InputEventMagnifyGesture> magnify_gesture = p_event;
	if (magnify_gesture.is_valid()) {

		_zoom_on_position(zoom * magnify_gesture->get_factor(), magnify_gesture->get_position());
		return;
	}

	Ref<InputEventPanGesture> pan_gesture = p_event;
	if (pan_gesture.is_valid()) {

		const Vector2 delta = (int(EditorSettings::get_singleton()->get("editors/2d/pan_speed")) / zoom) * pan_gesture->get_delta();
		h_scroll->set_value(h_scroll->get_value() + delta.x);
		v_scroll->set_value(v_scroll->get_value() + delta.y);
		return;
	}

	Ref<InputEventMouseButton> b = p_event;
	if (b.is_valid()) {
		// Button event

		if (b->get_button_index() == BUTTON_WHEEL_DOWN) {
			// Scroll or pan down
			if (bool(EditorSettings::get_singleton()->get("editors/2d/scroll_to_pan"))) {
				v_scroll->set_value(v_scroll->get_value() + int(EditorSettings::get_singleton()->get("editors/2d/pan_speed")) / zoom * b->get_factor());
				_update_scroll(0);
				viewport->update();
			} else {
				_zoom_on_position(zoom * (1 - (0.05 * b->get_factor())), b->get_position());
			}

			return;
		}

		if (b->get_button_index() == BUTTON_WHEEL_UP) {
			// Scroll or pan up
			if (bool(EditorSettings::get_singleton()->get("editors/2d/scroll_to_pan"))) {
				v_scroll->set_value(v_scroll->get_value() - int(EditorSettings::get_singleton()->get("editors/2d/pan_speed")) / zoom * b->get_factor());
				_update_scroll(0);
				viewport->update();
			} else {
				_zoom_on_position(zoom * ((0.95 + (0.05 * b->get_factor())) / 0.95), b->get_position());
			}

			return;
		}

		if (b->get_button_index() == BUTTON_WHEEL_LEFT) {
			// Pan left
			if (bool(EditorSettings::get_singleton()->get("editors/2d/scroll_to_pan"))) {

				h_scroll->set_value(h_scroll->get_value() - int(EditorSettings::get_singleton()->get("editors/2d/pan_speed")) / zoom * b->get_factor());
			}
		}

		if (b->get_button_index() == BUTTON_WHEEL_RIGHT) {
			// Pan right
			if (bool(EditorSettings::get_singleton()->get("editors/2d/scroll_to_pan"))) {

				h_scroll->set_value(h_scroll->get_value() + int(EditorSettings::get_singleton()->get("editors/2d/pan_speed")) / zoom * b->get_factor());
			}
		}

		if (b->get_button_index() == BUTTON_RIGHT) {

			if (b->is_pressed() && (tool == TOOL_SELECT && b->get_alt())) {
				// Open the selection list
				_list_select(b);
				return;
			}

			if (get_item_count() > 0 && drag != DRAG_NONE) {
				// Cancel a drag
				if (bone_ik_list.size()) {
					for (List<BoneIK>::Element *E = bone_ik_list.back(); E; E = E->prev()) {
						E->get().node->_edit_set_state(E->get().orig_state);
					}

					bone_ik_list.clear();

				} else {
					List<Node *> &selection = editor_selection->get_selected_node_list();

					for (List<Node *>::Element *E = selection.front(); E; E = E->next()) {
						CanvasItem *canvas_item = Object::cast_to<CanvasItem>(E->get());
						if (!canvas_item || !canvas_item->is_visible_in_tree())
							continue;
						if (canvas_item->get_viewport() != EditorNode::get_singleton()->get_scene_root())
							continue;

						CanvasItemEditorSelectedItem *se = editor_selection->get_node_editor_data<CanvasItemEditorSelectedItem>(canvas_item);
						if (!se)
							continue;

						canvas_item->_edit_set_state(se->undo_state);
						if (Object::cast_to<Node2D>(canvas_item))
							Object::cast_to<Node2D>(canvas_item)->_edit_set_pivot(se->undo_pivot);
						if (Object::cast_to<Control>(canvas_item))
							Object::cast_to<Control>(canvas_item)->set_pivot_offset(se->undo_pivot);
					}
				}

				drag = DRAG_NONE;
				viewport->update();
				can_move_pivot = false;

			} else if (box_selecting) {
				// Cancel box selection
				box_selecting = false;
				viewport->update();
			}
			return;
		}

		if (b->get_button_index() == BUTTON_LEFT && tool == TOOL_LIST_SELECT) {
			if (b->is_pressed())
				// Open the selection list
				_list_select(b);
			return;
		}

		if (b->get_button_index() == BUTTON_LEFT && tool == TOOL_EDIT_PIVOT) {
			if (b->is_pressed()) {
				// Set the pivot point
				Point2 mouse_pos = b->get_position();
				mouse_pos = transform.affine_inverse().xform(mouse_pos);
				mouse_pos = snap_point(mouse_pos, SNAP_DEFAULT, _get_single_item());
				_edit_set_pivot(mouse_pos);
			}
			return;
		}

		if (tool == TOOL_PAN || b->get_button_index() != BUTTON_LEFT || Input::get_singleton()->is_key_pressed(KEY_SPACE))
			// Pan the view
			return;

		// -- From now we consider that the button is BUTTON_LEFT --

		if (!b->is_pressed()) {

			if (drag != DRAG_NONE) {
				// Stop dragging
				if (undo_redo) {

					if (bone_ik_list.size()) {
						undo_redo->create_action(TTR("Edit IK Chain"));

						for (List<BoneIK>::Element *E = bone_ik_list.back(); E; E = E->prev()) {

							undo_redo->add_do_method(E->get().node, "_edit_set_state", E->get().node->_edit_get_state());
							undo_redo->add_undo_method(E->get().node, "_edit_set_state", E->get().orig_state);
						}

						undo_redo->add_do_method(viewport, "update");
						undo_redo->add_undo_method(viewport, "update");

						bone_ik_list.clear();

						undo_redo->commit_action();
					} else {
						undo_redo->create_action(TTR("Edit CanvasItem"));

						List<Node *> &selection = editor_selection->get_selected_node_list();

						for (List<Node *>::Element *E = selection.front(); E; E = E->next()) {

							CanvasItem *canvas_item = Object::cast_to<CanvasItem>(E->get());
							if (!canvas_item || !canvas_item->is_visible_in_tree())
								continue;
							if (canvas_item->get_viewport() != EditorNode::get_singleton()->get_scene_root())
								continue;

							CanvasItemEditorSelectedItem *se = editor_selection->get_node_editor_data<CanvasItemEditorSelectedItem>(canvas_item);
							if (!se)
								continue;

							Variant state = canvas_item->_edit_get_state();
							undo_redo->add_do_method(canvas_item, "_edit_set_state", state);
							undo_redo->add_undo_method(canvas_item, "_edit_set_state", se->undo_state);
							{
								Node2D *pvt = Object::cast_to<Node2D>(canvas_item);
								if (pvt && pvt->_edit_use_pivot()) {
									undo_redo->add_do_method(canvas_item, "_edit_set_pivot", pvt->_edit_get_pivot());
									undo_redo->add_undo_method(canvas_item, "_edit_set_pivot", se->undo_pivot);
								}

								Control *cnt = Object::cast_to<Control>(canvas_item);
								if (cnt) {
									undo_redo->add_do_method(canvas_item, "set_pivot_offset", cnt->get_pivot_offset());
									undo_redo->add_undo_method(canvas_item, "set_pivot_offset", se->undo_pivot);
								}
							}
						}
						undo_redo->commit_action();
					}
				}

				drag = DRAG_NONE;
				viewport->update();
				can_move_pivot = false;
			}

			if (box_selecting) {
				// Stop box selection
				Node *scene = editor->get_edited_scene();
				if (scene) {

					List<CanvasItem *> selitems;

					Point2 bsfrom = transform.xform(drag_from);
					Point2 bsto = transform.xform(box_selecting_to);
					if (bsfrom.x > bsto.x)
						SWAP(bsfrom.x, bsto.x);
					if (bsfrom.y > bsto.y)
						SWAP(bsfrom.y, bsto.y);

					_find_canvas_items_at_rect(Rect2(bsfrom, bsto - bsfrom), scene, transform, Transform2D(), &selitems);

					for (List<CanvasItem *>::Element *E = selitems.front(); E; E = E->next()) {

						_append_canvas_item(E->get());
					}
				}

				box_selecting = false;
				viewport->update();
			}
			return;
		}

		// -- From now we consider that the button is BUTTON_LEFT and that it is pressed --

		Map<ObjectID, BoneList>::Element *Cbone = NULL; //closest

		{
			bone_ik_list.clear();
			float closest_dist = 1e20;
			int bone_width = EditorSettings::get_singleton()->get("editors/2d/bone_width");
			for (Map<ObjectID, BoneList>::Element *E = bone_list.front(); E; E = E->next()) {

				if (E->get().from == E->get().to)
					continue;
				Vector2 s[2] = {
					E->get().from,
					E->get().to
				};

				Vector2 p = Geometry::get_closest_point_to_segment_2d(b->get_position(), s);
				float d = p.distance_to(b->get_position());
				if (d < bone_width && d < closest_dist) {
					Cbone = E;
					closest_dist = d;
				}
			}

			if (Cbone) {
				Node2D *b = Object::cast_to<Node2D>(ObjectDB::get_instance(Cbone->get().bone));

				if (b) {

					bool ik_found = false;

					bool first = true;

					while (b) {

						CanvasItem *pi = b->get_parent_item();
						if (!pi)
							break;

						float len = pi->get_global_transform().get_origin().distance_to(b->get_global_position());
						b = Object::cast_to<Node2D>(pi);
						if (!b)
							break;

						if (first) {

							bone_orig_xform = b->get_global_transform();
							first = false;
						}

						BoneIK bik;
						bik.node = b;
						bik.len = len;
						bik.orig_state = b->_edit_get_state();

						bone_ik_list.push_back(bik);

						if (b->has_meta("_edit_ik_")) {

							ik_found = bone_ik_list.size() > 1;
							break;
						}

						if (!pi->has_meta("_edit_bone_"))
							break;
					}

					if (!ik_found)
						bone_ik_list.clear();
				}
			}
		}

		// Single selected item
		CanvasItem *canvas_item = _get_single_item();
		if (canvas_item) {
			CanvasItemEditorSelectedItem *se = editor_selection->get_node_editor_data<CanvasItemEditorSelectedItem>(canvas_item);
			ERR_FAIL_COND(!se);

			Point2 click = b->get_position();

			// Rotation
			if ((b->get_control() && tool == TOOL_SELECT) || tool == TOOL_ROTATE) {
				drag = DRAG_ROTATE;
				drag_from = transform.affine_inverse().xform(click);
				se->undo_state = canvas_item->_edit_get_state();
				if (Object::cast_to<Node2D>(canvas_item))
					se->undo_pivot = Object::cast_to<Node2D>(canvas_item)->_edit_get_pivot();
				if (Object::cast_to<Control>(canvas_item))
					se->undo_pivot = Object::cast_to<Control>(canvas_item)->get_pivot_offset();
				se->pre_drag_xform = canvas_item->get_global_transform_with_canvas();
				se->pre_drag_rect = canvas_item->_edit_get_rect();
				return;
			}

			if (tool == TOOL_SELECT) {
				// Open a sub-scene on double-click
				if (b->is_doubleclick()) {
					if (canvas_item->get_filename() != "" && canvas_item != editor->get_edited_scene()) {
						editor->open_request(canvas_item->get_filename());
						return;
					}
				}

				// Drag resize handles
				drag = _get_resize_handle_drag_type(click, drag_point_from);
				if (drag != DRAG_NONE) {
					drag_from = transform.affine_inverse().xform(click);
					se->undo_state = canvas_item->_edit_get_state();
					if (Object::cast_to<Node2D>(canvas_item))
						se->undo_pivot = Object::cast_to<Node2D>(canvas_item)->_edit_get_pivot();
					if (Object::cast_to<Control>(canvas_item))
						se->undo_pivot = Object::cast_to<Control>(canvas_item)->get_pivot_offset();
					se->pre_drag_xform = canvas_item->get_global_transform_with_canvas();
					se->pre_drag_rect = canvas_item->_edit_get_rect();
					return;
				}

				// Drag anchor handles
				Control *control = Object::cast_to<Control>(canvas_item);
				if (control && show_helpers && !Object::cast_to<Container>(control->get_parent())) {
					drag = _get_anchor_handle_drag_type(click, drag_point_from);
					if (drag != DRAG_NONE) {
						drag_from = transform.affine_inverse().xform(click);
						se->undo_state = canvas_item->_edit_get_state();
						se->pre_drag_xform = canvas_item->get_global_transform_with_canvas();
						se->pre_drag_rect = canvas_item->_edit_get_rect();
						return;
					}
				}
			}
		}

		// Multiple selected items
		Point2 click = b->get_position();

		if ((b->get_alt() || tool == TOOL_MOVE) && get_item_count()) {
			// Drag the nodes
			_prepare_drag(click);
			viewport->update();
			return;
		}

		CanvasItem *c = NULL;
		if (Cbone) {
			c = Object::cast_to<CanvasItem>(ObjectDB::get_instance(Cbone->get().bone));
			if (c)
				c = c->get_parent_item();
		}

		Node *scene = editor->get_edited_scene();
		if (!scene)
			return;
		// Find the item to select
		if (!c) {
			Vector<_SelectResult> selection;
			_find_canvas_items_at_pos(click, scene, transform, Transform2D(), selection, 1);
			if (!selection.empty())
				c = selection[0].item;

			CanvasItem *cn = c;
			while (cn) {
				if (cn->has_meta("_edit_group_")) {
					c = cn;
				}
				cn = cn->get_parent_item();
			}
		}

		Node *n = c;
		while ((n && n != scene && n->get_owner() != scene) || (n && !n->is_class("CanvasItem"))) {
			n = n->get_parent();
		};

		if (n) {
			c = Object::cast_to<CanvasItem>(n);
		} else {
			c = NULL;
		}

		// Select the item
		additive_selection = b->get_shift();
		if (!c) {
			_select_click_on_empty_area(click, additive_selection, true);
		} else if (!_select_click_on_item(c, click, additive_selection, true)) {
			return;
		}
	}

	Ref<InputEventMouseMotion> m = p_event;
	if (m.is_valid()) {
		// Mouse motion event
		_update_cursor();

		if (box_selecting) {
			// Update box selection
			box_selecting_to = transform.affine_inverse().xform(m->get_position());
			viewport->update();
			return;
		}

		if (drag == DRAG_NONE) {
			bool space_pressed = Input::get_singleton()->is_key_pressed(KEY_SPACE);
			bool simple_panning = EditorSettings::get_singleton()->get("editors/2d/simple_spacebar_panning");
			int button = m->get_button_mask();

			// Check if any of the panning triggers are activated
			bool panning_tool = (button & BUTTON_MASK_LEFT) && tool == TOOL_PAN;
			bool panning_middle_button = button & BUTTON_MASK_MIDDLE;
			bool panning_spacebar = (button & BUTTON_MASK_LEFT) && space_pressed;
			bool panning_spacebar_simple = space_pressed && simple_panning;

			if (panning_tool || panning_middle_button || panning_spacebar || panning_spacebar_simple) {
				// Pan the viewport
				Point2i relative;
				if (bool(EditorSettings::get_singleton()->get("editors/2d/warped_mouse_panning"))) {
					relative = Input::get_singleton()->warp_mouse_motion(m, viewport->get_global_rect());
				} else {
					relative = m->get_relative();
				}

				h_scroll->set_value(h_scroll->get_value() - relative.x / zoom);
				v_scroll->set_value(v_scroll->get_value() - relative.y / zoom);
			}

			return;
		}

		List<Node *> &selection = editor_selection->get_selected_node_list();
		for (List<Node *>::Element *E = selection.front(); E; E = E->next()) {

			CanvasItem *canvas_item = Object::cast_to<CanvasItem>(E->get());
			if (!canvas_item || !canvas_item->is_visible_in_tree())
				continue;
			if (canvas_item->get_viewport() != EditorNode::get_singleton()->get_scene_root())
				continue;

			CanvasItemEditorSelectedItem *se = editor_selection->get_node_editor_data<CanvasItemEditorSelectedItem>(canvas_item);
			if (!se)
				continue;

			bool dragging_bone = drag == DRAG_ALL && selection.size() == 1 && bone_ik_list.size();

			if (!dragging_bone) {
				canvas_item->_edit_set_state(se->undo_state); //reset state and reapply
				if (Object::cast_to<Node2D>(canvas_item))
					Object::cast_to<Node2D>(canvas_item)->_edit_set_pivot(se->undo_pivot);
				if (Object::cast_to<Control>(canvas_item))
					Object::cast_to<Control>(canvas_item)->set_pivot_offset(se->undo_pivot);
			}

			Vector2 dfrom = drag_from;
			Vector2 dto = transform.affine_inverse().xform(m->get_position());
			if (canvas_item->has_meta("_edit_lock_"))
				continue;

			if (drag == DRAG_ROTATE) {
				// Rotate the node
				Vector2 center = canvas_item->get_global_transform_with_canvas().get_origin();
				{
					Node2D *node = Object::cast_to<Node2D>(canvas_item);

					if (node) {
						real_t angle = node->get_rotation();
						node->set_rotation(snap_angle(angle + (dfrom - center).angle_to(dto - center), angle));
						display_rotate_to = dto;
						display_rotate_from = center;
						viewport->update();
					}
				}

				{
					Control *node = Object::cast_to<Control>(canvas_item);

					if (node) {
						real_t angle = node->get_rotation();
						node->set_rotation(snap_angle(angle + (dfrom - center).angle_to(dto - center), angle));
						display_rotate_to = dto;
						display_rotate_from = center;
						viewport->update();
					}
				}

				continue;
			}

			bool uniform = m->get_shift();
			bool symmetric = m->get_alt();

			Vector2 drag_vector =
					canvas_item->get_global_transform_with_canvas().affine_inverse().xform(dto) -
					canvas_item->get_global_transform_with_canvas().affine_inverse().xform(dfrom);

			switch (drag) {
				case DRAG_ALL:
				case DRAG_NODE_2D:
					dto -= drag_from - drag_point_from;
					if (uniform) {
						if (ABS(dto.x - drag_point_from.x) > ABS(dto.y - drag_point_from.y)) {
							dto.y = drag_point_from.y;
						} else {
							dto.x = drag_point_from.x;
						}
					}
					break;
			}

			Control *control = Object::cast_to<Control>(canvas_item);
			if (control) {
				// Drag and snap the anchor
				Transform2D c_trans_rev = canvas_item->get_global_transform_with_canvas().affine_inverse();

				Vector2 anchor = c_trans_rev.xform(dto - drag_from + drag_point_from);
				anchor = _position_to_anchor(control, anchor);

				Vector2 anchor_snapped = c_trans_rev.xform(snap_point(dto - drag_from + drag_point_from, SNAP_GRID | SNAP_GUIDES | SNAP_OTHER_NODES, _get_single_item(), SNAP_NODE_PARENT | SNAP_NODE_SIDES));
				anchor_snapped = _position_to_anchor(control, anchor_snapped).snapped(Vector2(0.00001, 0.00001));

				bool use_y = Math::abs(drag_vector.y) > Math::abs(drag_vector.x);

				switch (drag) {
					case DRAG_ANCHOR_TOP_LEFT:
						if (!uniform || (uniform && !use_y)) control->set_anchor(MARGIN_LEFT, anchor_snapped.x);
						if (!uniform || (uniform && use_y)) control->set_anchor(MARGIN_TOP, anchor_snapped.y);
						continue;
						break;
					case DRAG_ANCHOR_TOP_RIGHT:
						if (!uniform || (uniform && !use_y)) control->set_anchor(MARGIN_RIGHT, anchor_snapped.x);
						if (!uniform || (uniform && use_y)) control->set_anchor(MARGIN_TOP, anchor_snapped.y);
						continue;
						break;
					case DRAG_ANCHOR_BOTTOM_RIGHT:
						if (!uniform || (uniform && !use_y)) control->set_anchor(MARGIN_RIGHT, anchor_snapped.x);
						if (!uniform || (uniform && use_y)) control->set_anchor(MARGIN_BOTTOM, anchor_snapped.y);
						break;
					case DRAG_ANCHOR_BOTTOM_LEFT:
						if (!uniform || (uniform && !use_y)) control->set_anchor(MARGIN_LEFT, anchor_snapped.x);
						if (!uniform || (uniform && use_y)) control->set_anchor(MARGIN_BOTTOM, anchor_snapped.y);
						continue;
						break;
					case DRAG_ANCHOR_ALL:
						if (!uniform || (uniform && !use_y)) control->set_anchor(MARGIN_LEFT, anchor_snapped.x);
						if (!uniform || (uniform && !use_y)) control->set_anchor(MARGIN_RIGHT, anchor_snapped.x);
						if (!uniform || (uniform && use_y)) control->set_anchor(MARGIN_TOP, anchor_snapped.y);
						if (!uniform || (uniform && use_y)) control->set_anchor(MARGIN_BOTTOM, anchor_snapped.y);
						continue;
						break;
				}
			}

			dfrom = drag_point_from;
			dto = snap_point(dto, SNAP_NODE_ANCHORS | SNAP_NODE_PARENT | SNAP_OTHER_NODES | SNAP_GRID | SNAP_GUIDES | SNAP_PIXEL, _get_single_item());

			drag_vector =
					canvas_item->get_global_transform_with_canvas().affine_inverse().xform(dto) -
					canvas_item->get_global_transform_with_canvas().affine_inverse().xform(dfrom);

			Rect2 local_rect = canvas_item->_edit_get_rect();
			Vector2 begin = local_rect.position;
			Vector2 end = local_rect.position + local_rect.size;
			Vector2 minsize = canvas_item->_edit_get_minimum_size();

			if (uniform) {
				// Keep the height/width ratio of the item
				float aspect = local_rect.size.aspect();
				switch (drag) {
					case DRAG_LEFT:
						drag_vector.y = -drag_vector.x / aspect;
						break;
					case DRAG_RIGHT:
						drag_vector.y = drag_vector.x / aspect;
						break;
					case DRAG_TOP:
						drag_vector.x = -drag_vector.y * aspect;
						break;
					case DRAG_BOTTOM:
						drag_vector.x = drag_vector.y * aspect;
						break;
					case DRAG_BOTTOM_LEFT:
					case DRAG_TOP_RIGHT:
						if (aspect > 1.0) { // width > height, take x as reference
							drag_vector.y = -drag_vector.x / aspect;
						} else { // height > width, take y as reference
							drag_vector.x = -drag_vector.y * aspect;
						}
						break;
					case DRAG_BOTTOM_RIGHT:
					case DRAG_TOP_LEFT:
						if (aspect > 1.0) { // width > height, take x as reference
							drag_vector.y = drag_vector.x / aspect;
						} else { // height > width, take y as reference
							drag_vector.x = drag_vector.y * aspect;
						}
						break;
				}
			} else {
				switch (drag) {
					case DRAG_RIGHT:
					case DRAG_LEFT:
						drag_vector.y = 0;
						break;
					case DRAG_TOP:
					case DRAG_BOTTOM:
						drag_vector.x = 0;
						break;
				}
			}

			switch (drag) {
				case DRAG_ALL:
					begin += drag_vector;
					end += drag_vector;
					break;
				case DRAG_RIGHT:
				case DRAG_BOTTOM:
				case DRAG_BOTTOM_RIGHT:
					incend(begin.x, end.x, drag_vector.x, minsize.x, symmetric);
					incend(begin.y, end.y, drag_vector.y, minsize.y, symmetric);
					break;
				case DRAG_TOP_LEFT:
					incbeg(begin.x, end.x, drag_vector.x, minsize.x, symmetric);
					incbeg(begin.y, end.y, drag_vector.y, minsize.y, symmetric);
					break;
				case DRAG_TOP:
				case DRAG_TOP_RIGHT:
					incbeg(begin.y, end.y, drag_vector.y, minsize.y, symmetric);
					incend(begin.x, end.x, drag_vector.x, minsize.x, symmetric);
					break;
				case DRAG_LEFT:
				case DRAG_BOTTOM_LEFT:
					incbeg(begin.x, end.x, drag_vector.x, minsize.x, symmetric);
					incend(begin.y, end.y, drag_vector.y, minsize.y, symmetric);
					break;

				case DRAG_PIVOT:

					if (Object::cast_to<Node2D>(canvas_item)) {
						Node2D *n2d = Object::cast_to<Node2D>(canvas_item);
						n2d->_edit_set_pivot(se->undo_pivot + drag_vector);
					}
					if (Object::cast_to<Control>(canvas_item)) {
						Object::cast_to<Control>(canvas_item)->set_pivot_offset(se->undo_pivot + drag_vector);
					}
					continue;
					break;
				case DRAG_NODE_2D:

					ERR_FAIL_COND(!Object::cast_to<Node2D>(canvas_item));
					Object::cast_to<Node2D>(canvas_item)->set_global_position(dto);
					continue;
					break;
			}

			if (!dragging_bone) {

				local_rect.position = begin;
				local_rect.size = end - begin;
				canvas_item->_edit_set_rect(local_rect);

			} else {
				//ok, all that had to be done was done, now solve IK

				Node2D *n2d = Object::cast_to<Node2D>(canvas_item);
				Transform2D final_xform = bone_orig_xform;

				if (n2d) {

					float total_len = 0;
					for (List<BoneIK>::Element *E = bone_ik_list.front(); E; E = E->next()) {
						if (E->prev())
							total_len += E->get().len;
						E->get().pos = E->get().node->get_global_transform().get_origin();
					}

					{

						final_xform.elements[2] += dto - dfrom; //final_xform.affine_inverse().basis_xform_inv(drag_vector);
						//n2d->set_global_transform(final_xform);
					}

					CanvasItem *last = bone_ik_list.back()->get().node;
					if (!last)
						break;

					Vector2 root_pos = last->get_global_transform().get_origin();
					Vector2 leaf_pos = final_xform.get_origin();

					if ((leaf_pos.distance_to(root_pos)) > total_len) {
						//oops dude you went too far
						//print_line("TOO FAR!");
						Vector2 rel = leaf_pos - root_pos;
						rel = rel.normalized() * total_len;
						leaf_pos = root_pos + rel;
					}

					bone_ik_list.front()->get().pos = leaf_pos;

					//print_line("BONE IK LIST "+itos(bone_ik_list.size()));

					if (bone_ik_list.size() > 2) {
						int solver_iterations = 64;
						float solver_k = 0.3;

						for (int i = 0; i < solver_iterations; i++) {

							for (List<BoneIK>::Element *E = bone_ik_list.front(); E; E = E->next()) {

								if (E == bone_ik_list.back()) {

									break;
								}

								float len = E->next()->get().len;

								if (E->next() == bone_ik_list.back()) {

									//print_line("back");

									Vector2 rel = E->get().pos - E->next()->get().pos;
									//print_line("PREV "+E->get().pos);
									Vector2 desired = E->next()->get().pos + rel.normalized() * len;
									//print_line("DESIRED "+desired);
									E->get().pos = E->get().pos.linear_interpolate(desired, solver_k);
									//print_line("POST "+E->get().pos);

								} else if (E == bone_ik_list.front()) {
									//only adjust parent
									//print_line("front");
									Vector2 rel = E->next()->get().pos - E->get().pos;
									//print_line("PREV "+E->next()->get().pos);
									Vector2 desired = E->get().pos + rel.normalized() * len;
									//print_line("DESIRED "+desired);
									E->next()->get().pos = E->next()->get().pos.linear_interpolate(desired, solver_k);
									//print_line("POST "+E->next()->get().pos);
								} else {

									Vector2 rel = E->next()->get().pos - E->get().pos;
									Vector2 cen = (E->next()->get().pos + E->get().pos) * 0.5;
									rel = rel.linear_interpolate(rel.normalized() * len, solver_k);
									rel *= 0.5;
									E->next()->get().pos = cen + rel;
									E->get().pos = cen - rel;
									//print_line("mid");
								}
							}
						}
					}
				}

				for (List<BoneIK>::Element *E = bone_ik_list.back(); E; E = E->prev()) {

					Node2D *n = E->get().node;

					if (!E->prev()) {
						//last goes to what it was
						final_xform.set_origin(n->get_global_position());
						n->set_global_transform(final_xform);

					} else {
						Vector2 rel = (E->prev()->get().node->get_global_position() - n->get_global_position()).normalized();
						Vector2 rel2 = (E->prev()->get().pos - E->get().pos).normalized();
						float rot = rel.angle_to(rel2);
						if (n->get_global_transform().basis_determinant() < 0) {
							//mirrored, rotate the other way
							rot = -rot;
						}

						n->rotate(rot);
					}
				}

				break;
			}
		}
	}
}

void CanvasItemEditor::_draw_text_at_position(Point2 p_position, String p_string, Margin p_side) {
	Color color = get_color("font_color", "Editor");
	color.a = 0.8;
	Ref<Font> font = get_font("font", "Label");
	Size2 text_size = font->get_string_size(p_string);
	switch (p_side) {
		case MARGIN_LEFT:
			p_position += Vector2(-text_size.x - 5, text_size.y / 2);
			break;
		case MARGIN_TOP:
			p_position += Vector2(-text_size.x / 2, -5);
			break;
		case MARGIN_RIGHT:
			p_position += Vector2(5, text_size.y / 2);
			break;
		case MARGIN_BOTTOM:
			p_position += Vector2(-text_size.x / 2, text_size.y + 5);
			break;
	}
	viewport->draw_string(font, p_position, p_string, color);
}

void CanvasItemEditor::_draw_margin_at_position(int p_value, Point2 p_position, Margin p_side) {
	String str = vformat("%d px", p_value);
	if (p_value != 0) {
		_draw_text_at_position(p_position, str, p_side);
	}
}

void CanvasItemEditor::_draw_percentage_at_position(float p_value, Point2 p_position, Margin p_side) {
	String str = vformat("%.1f %%", p_value * 100.0);
	if (p_value != 0) {
		_draw_text_at_position(p_position, str, p_side);
	}
}

void CanvasItemEditor::_draw_focus() {
	// Draw the focus around the base viewport
	if (viewport_base->has_focus()) {
		get_stylebox("Focus", "EditorStyles")->draw(viewport_base->get_canvas_item(), Rect2(Point2(), viewport_base->get_size()));
	}
}

void CanvasItemEditor::_draw_guides() {

	Color guide_color = EditorSettings::get_singleton()->get("editors/2d/guides_color");
	Transform2D xform = viewport_scrollable->get_transform() * transform;

	// Guides already there
	if (EditorNode::get_singleton()->get_edited_scene() && EditorNode::get_singleton()->get_edited_scene()->has_meta("_edit_vertical_guides_")) {
		Array vguides = EditorNode::get_singleton()->get_edited_scene()->get_meta("_edit_vertical_guides_");
		for (int i = 0; i < vguides.size(); i++) {
			if (drag == DRAG_V_GUIDE && i == edited_guide_index)
				continue;
			float x = xform.xform(Point2(vguides[i], 0)).x;
			viewport_base->draw_line(Point2(x, 0), Point2(x, viewport_base->get_size().y), guide_color);
		}
	}

	if (EditorNode::get_singleton()->get_edited_scene() && EditorNode::get_singleton()->get_edited_scene()->has_meta("_edit_horizontal_guides_")) {
		Array hguides = EditorNode::get_singleton()->get_edited_scene()->get_meta("_edit_horizontal_guides_");
		for (int i = 0; i < hguides.size(); i++) {
			if (drag == DRAG_H_GUIDE && i == edited_guide_index)
				continue;
			float y = xform.xform(Point2(0, hguides[i])).y;
			viewport_base->draw_line(Point2(0, y), Point2(viewport_base->get_size().x, y), guide_color);
		}
	}

	// Dragged guide
	Color text_color = get_color("font_color", "Editor");
	text_color.a = 0.5;
	if (drag == DRAG_DOUBLE_GUIDE || drag == DRAG_V_GUIDE) {
		String str = vformat("%d px", xform.affine_inverse().xform(edited_guide_pos).x);
		Ref<Font> font = get_font("font", "Label");
		Size2 text_size = font->get_string_size(str);
		viewport_base->draw_string(font, Point2(edited_guide_pos.x + 10, RULER_WIDTH + text_size.y / 2 + 10), str, text_color);
		viewport_base->draw_line(Point2(edited_guide_pos.x, 0), Point2(edited_guide_pos.x, viewport_base->get_size().y), guide_color);
	}
	if (drag == DRAG_DOUBLE_GUIDE || drag == DRAG_H_GUIDE) {
		String str = vformat("%d px", xform.affine_inverse().xform(edited_guide_pos).y);
		Ref<Font> font = get_font("font", "Label");
		Size2 text_size = font->get_string_size(str);
		viewport_base->draw_string(font, Point2(RULER_WIDTH + 10, edited_guide_pos.y + text_size.y / 2 + 10), str, text_color);
		viewport_base->draw_line(Point2(0, edited_guide_pos.y), Point2(viewport_base->get_size().x, edited_guide_pos.y), guide_color);
	}
}

void CanvasItemEditor::_draw_rulers() {
	Color graduation_color = get_color("font_color", "Editor");
	graduation_color.a = 0.5;
	Color bg_color = get_color("dark_color_2", "Editor");
	Color font_color = get_color("font_color", "Editor");
	font_color.a = 0.8;
	Ref<Font> font = get_font("rulers", "EditorFonts");

	// The rule transform
	Transform2D ruler_transform;
	if (show_grid || snap_grid) {
		ruler_transform = Transform2D();
		if (snap_relative && get_item_count() > 0) {
			ruler_transform.translate(_find_topleftmost_point());
			ruler_transform.scale_basis(grid_step * Math::pow(2.0, grid_step_multiplier));
		} else {
			ruler_transform.translate(grid_offset);
			ruler_transform.scale_basis(grid_step * Math::pow(2.0, grid_step_multiplier));
		}
		while ((transform * ruler_transform).get_scale().x < 50 || (transform * ruler_transform).get_scale().y < 50) {

			ruler_transform.scale_basis(Point2(2, 2));
		}
	} else {
		float basic_rule = 100;
		for (int i = 0; basic_rule * zoom > 100; i++) {
			basic_rule /= (i % 2) ? 5.0 : 2.0;
		}
		for (int i = 0; basic_rule * zoom < 100; i++) {
			basic_rule *= (i % 2) ? 2.0 : 5.0;
		}
		ruler_transform = Transform2D();
		ruler_transform.scale(Size2(basic_rule, basic_rule));
	}

	// Subdivisions
	int major_subdivision = 2;
	Transform2D major_subdivide = Transform2D();
	major_subdivide.scale(Size2(1.0 / major_subdivision, 1.0 / major_subdivision));

	int minor_subdivision = 5;
	Transform2D minor_subdivide = Transform2D();
	minor_subdivide.scale(Size2(1.0 / minor_subdivision, 1.0 / minor_subdivision));

	// First and last graduations to draw (in the ruler space)
	Point2 first = (transform * ruler_transform * major_subdivide * minor_subdivide).affine_inverse().xform(Point2());
	Point2 last = (transform * ruler_transform * major_subdivide * minor_subdivide).affine_inverse().xform(viewport->get_size());

	// Draw top ruler
	viewport_base->draw_rect(Rect2(Point2(RULER_WIDTH, 0), Size2(viewport->get_size().x, RULER_WIDTH)), bg_color);
	for (int i = Math::ceil(first.x); i < last.x; i++) {
		Point2 position = (transform * ruler_transform * major_subdivide * minor_subdivide).xform(Point2(i, 0));
		if (i % (major_subdivision * minor_subdivision) == 0) {
			viewport_base->draw_line(Point2(position.x + RULER_WIDTH, 0), Point2(position.x + RULER_WIDTH, RULER_WIDTH), graduation_color);
			float val = (ruler_transform * major_subdivide * minor_subdivide).xform(Point2(i, 0)).x;
			viewport_base->draw_string(font, Point2(position.x + RULER_WIDTH + 2, font->get_height()), vformat(((int)val == val) ? "%d" : "%.1f", val), font_color);
		} else {
			if (i % minor_subdivision == 0) {
				viewport_base->draw_line(Point2(position.x + RULER_WIDTH, RULER_WIDTH * 0.33), Point2(position.x + RULER_WIDTH, RULER_WIDTH), graduation_color);
			} else {
				viewport_base->draw_line(Point2(position.x + RULER_WIDTH, RULER_WIDTH * 0.66), Point2(position.x + RULER_WIDTH, RULER_WIDTH), graduation_color);
			}
		}
	}

	// Draw left ruler
	viewport_base->draw_rect(Rect2(Point2(0, RULER_WIDTH), Size2(RULER_WIDTH, viewport->get_size().y)), bg_color);
	for (int i = Math::ceil(first.y); i < last.y; i++) {
		Point2 position = (transform * ruler_transform * major_subdivide * minor_subdivide).xform(Point2(0, i));
		if (i % (major_subdivision * minor_subdivision) == 0) {
			viewport_base->draw_line(Point2(0, position.y + RULER_WIDTH), Point2(RULER_WIDTH, position.y + RULER_WIDTH), graduation_color);
			float val = (ruler_transform * major_subdivide * minor_subdivide).xform(Point2(0, i)).y;
			viewport_base->draw_string(font, Point2(2, position.y + RULER_WIDTH + 2 + font->get_height()), vformat(((int)val == val) ? "%d" : "%.1f", val), font_color);
		} else {
			if (i % minor_subdivision == 0) {
				viewport_base->draw_line(Point2(RULER_WIDTH * 0.33, position.y + RULER_WIDTH), Point2(RULER_WIDTH, position.y + RULER_WIDTH), graduation_color);
			} else {
				viewport_base->draw_line(Point2(RULER_WIDTH * 0.66, position.y + RULER_WIDTH), Point2(RULER_WIDTH, position.y + RULER_WIDTH), graduation_color);
			}
		}
	}
	viewport_base->draw_rect(Rect2(Point2(), Size2(RULER_WIDTH, RULER_WIDTH)), graduation_color);
}

void CanvasItemEditor::_draw_grid() {
	if (show_grid) {
		//Draw the grid
		Size2 s = viewport->get_size();
		int last_cell = 0;
		Transform2D xform = transform.affine_inverse();

		Vector2 real_grid_offset;
		if (snap_relative && get_item_count() > 0) {
			Vector2 topleft = _find_topleftmost_point();
			real_grid_offset.x = fmod(topleft.x, grid_step.x * (real_t)Math::pow(2.0, grid_step_multiplier));
			real_grid_offset.y = fmod(topleft.y, grid_step.y * (real_t)Math::pow(2.0, grid_step_multiplier));
		} else {
			real_grid_offset = grid_offset;
		}

		const Color grid_minor_color = get_color("grid_minor_color", "Editor");
		if (grid_step.x != 0) {
			for (int i = 0; i < s.width; i++) {
				int cell = Math::fast_ftoi(Math::floor((xform.xform(Vector2(i, 0)).x - real_grid_offset.x) / (grid_step.x * Math::pow(2.0, grid_step_multiplier))));
				if (i == 0)
					last_cell = cell;
				if (last_cell != cell)
					viewport->draw_line(Point2(i, 0), Point2(i, s.height), grid_minor_color);
				last_cell = cell;
			}
		}

		if (grid_step.y != 0) {
			for (int i = 0; i < s.height; i++) {
				int cell = Math::fast_ftoi(Math::floor((xform.xform(Vector2(0, i)).y - real_grid_offset.y) / (grid_step.y * Math::pow(2.0, grid_step_multiplier))));
				if (i == 0)
					last_cell = cell;
				if (last_cell != cell)
					viewport->draw_line(Point2(0, i), Point2(s.width, i), grid_minor_color);
				last_cell = cell;
			}
		}
	}
}

void CanvasItemEditor::_draw_selection() {
	bool pivot_found = false;
	Ref<Texture> pivot_icon = get_icon("EditorPivot", "EditorIcons");
	bool single = _get_single_item() != NULL;
	RID ci = viewport->get_canvas_item();

	Map<Node *, Object *> &selection = editor_selection->get_selection();
	for (Map<Node *, Object *>::Element *E = selection.front(); E; E = E->next()) {

		CanvasItem *canvas_item = Object::cast_to<CanvasItem>(E->key());
		if (!canvas_item || !canvas_item->is_visible_in_tree())
			continue;
		if (canvas_item->get_viewport() != EditorNode::get_singleton()->get_scene_root())
			continue;
		CanvasItemEditorSelectedItem *se = editor_selection->get_node_editor_data<CanvasItemEditorSelectedItem>(canvas_item);
		if (!se)
			continue;

		Rect2 rect = canvas_item->_edit_get_rect();

		if (show_helpers && drag != DRAG_NONE && drag != DRAG_PIVOT) {
			const Transform2D pre_drag_xform = transform * se->pre_drag_xform;
			const Color pre_drag_color = Color(0.4, 0.6, 1, 0.7);

			Vector2 pre_drag_endpoints[4] = {

				pre_drag_xform.xform(se->pre_drag_rect.position),
				pre_drag_xform.xform(se->pre_drag_rect.position + Vector2(se->pre_drag_rect.size.x, 0)),
				pre_drag_xform.xform(se->pre_drag_rect.position + se->pre_drag_rect.size),
				pre_drag_xform.xform(se->pre_drag_rect.position + Vector2(0, se->pre_drag_rect.size.y))
			};

			for (int i = 0; i < 4; i++) {
				viewport->draw_line(pre_drag_endpoints[i], pre_drag_endpoints[(i + 1) % 4], pre_drag_color, 2);
			}
		}

		Transform2D xform = transform * canvas_item->get_global_transform_with_canvas();
		VisualServer::get_singleton()->canvas_item_add_set_transform(ci, xform);

		Vector2 endpoints[4] = {

			xform.xform(rect.position),
			xform.xform(rect.position + Vector2(rect.size.x, 0)),
			xform.xform(rect.position + rect.size),
			xform.xform(rect.position + Vector2(0, rect.size.y))
		};

		Color c = Color(1, 0.6, 0.4, 0.7);

		VisualServer::get_singleton()->canvas_item_add_set_transform(ci, Transform2D());

		for (int i = 0; i < 4; i++) {
			viewport->draw_line(endpoints[i], endpoints[(i + 1) % 4], c, 2);
		}

		if (single && (tool == TOOL_SELECT || tool == TOOL_MOVE || tool == TOOL_ROTATE || tool == TOOL_EDIT_PIVOT)) { //kind of sucks

			Node2D *node2d = Object::cast_to<Node2D>(canvas_item);
			if (node2d) {
				if (node2d->_edit_use_pivot()) {
					viewport->draw_texture(pivot_icon, xform.get_origin() + (-pivot_icon->get_size() / 2).floor());
					can_move_pivot = true;
					pivot_found = true;
				}
			}

			Control *control = Object::cast_to<Control>(canvas_item);
			if (control) {
				Vector2 pivot_ofs = control->get_pivot_offset();
				if (pivot_ofs != Vector2()) {
					viewport->draw_texture(pivot_icon, xform.xform(pivot_ofs) + (-pivot_icon->get_size() / 2).floor());
				}
				can_move_pivot = true;
				pivot_found = true;

				if (tool == TOOL_SELECT && show_helpers && !Object::cast_to<Container>(control->get_parent())) {
					// Draw the helpers
					Color color_base = Color(0.8, 0.8, 0.8, 0.5);

					float anchors_values[4];
					anchors_values[0] = control->get_anchor(MARGIN_LEFT);
					anchors_values[1] = control->get_anchor(MARGIN_TOP);
					anchors_values[2] = control->get_anchor(MARGIN_RIGHT);
					anchors_values[3] = control->get_anchor(MARGIN_BOTTOM);

					// Draw the anchors
					Vector2 anchors[4];
					Vector2 anchors_pos[4];
					for (int i = 0; i < 4; i++) {
						anchors[i] = Vector2((i % 2 == 0) ? anchors_values[i] : anchors_values[(i + 1) % 4], (i % 2 == 1) ? anchors_values[i] : anchors_values[(i + 1) % 4]);
						anchors_pos[i] = xform.xform(_anchor_to_position(control, anchors[i]));
					}

					Map<Node *, Object *> &selection = editor_selection->get_selection();
					// Get which anchor is dragged
					int dragged_anchor = -1;
					switch (drag) {
						case DRAG_ANCHOR_ALL:
						case DRAG_ANCHOR_TOP_LEFT:
							dragged_anchor = 0;
							break;
						case DRAG_ANCHOR_TOP_RIGHT:
							dragged_anchor = 1;
							break;
						case DRAG_ANCHOR_BOTTOM_RIGHT:
							dragged_anchor = 2;
							break;
						case DRAG_ANCHOR_BOTTOM_LEFT:
							dragged_anchor = 3;
							break;
					}

					if (dragged_anchor >= 0) {
						// Draw the 4 lines when dragged
						bool snapped;
						Color color_snapped = Color(0.64, 0.93, 0.67, 0.5);

						Vector2 corners_pos[4];
						for (int i = 0; i < 4; i++) {
							corners_pos[i] = xform.xform(_anchor_to_position(control, Vector2((i == 0 || i == 3) ? ANCHOR_BEGIN : ANCHOR_END, (i <= 1) ? ANCHOR_BEGIN : ANCHOR_END)));
						}

						Vector2 line_starts[4];
						Vector2 line_ends[4];
						for (int i = 0; i < 4; i++) {
							float anchor_val = (i >= 2) ? ANCHOR_END - anchors_values[i] : anchors_values[i];
							line_starts[i] = Vector2::linear_interpolate(corners_pos[i], corners_pos[(i + 1) % 4], anchor_val);
							line_ends[i] = Vector2::linear_interpolate(corners_pos[(i + 3) % 4], corners_pos[(i + 2) % 4], anchor_val);
							snapped = anchors_values[i] == 0.0 || anchors_values[i] == 0.5 || anchors_values[i] == 1.0;
							viewport->draw_line(line_starts[i], line_ends[i], snapped ? color_snapped : color_base, (i == dragged_anchor || (i + 3) % 4 == dragged_anchor) ? 2 : 1);
						}

						// Display the percentages next to the lines
						float percent_val;
						percent_val = anchors_values[(dragged_anchor + 2) % 4] - anchors_values[dragged_anchor];
						percent_val = (dragged_anchor >= 2) ? -percent_val : percent_val;
						_draw_percentage_at_position(percent_val, (anchors_pos[dragged_anchor] + anchors_pos[(dragged_anchor + 1) % 4]) / 2, (Margin)((dragged_anchor + 1) % 4));

						percent_val = anchors_values[(dragged_anchor + 3) % 4] - anchors_values[(dragged_anchor + 1) % 4];
						percent_val = ((dragged_anchor + 1) % 4 >= 2) ? -percent_val : percent_val;
						_draw_percentage_at_position(percent_val, (anchors_pos[dragged_anchor] + anchors_pos[(dragged_anchor + 3) % 4]) / 2, (Margin)(dragged_anchor));

						percent_val = anchors_values[(dragged_anchor + 1) % 4];
						percent_val = ((dragged_anchor + 1) % 4 >= 2) ? ANCHOR_END - percent_val : percent_val;
						_draw_percentage_at_position(percent_val, (line_starts[dragged_anchor] + anchors_pos[dragged_anchor]) / 2, (Margin)(dragged_anchor));

						percent_val = anchors_values[dragged_anchor];
						percent_val = (dragged_anchor >= 2) ? ANCHOR_END - percent_val : percent_val;
						_draw_percentage_at_position(percent_val, (line_ends[(dragged_anchor + 1) % 4] + anchors_pos[dragged_anchor]) / 2, (Margin)((dragged_anchor + 1) % 4));
					}

					Rect2 anchor_rects[4];
					anchor_rects[0] = Rect2(anchors_pos[0] - anchor_handle->get_size(), anchor_handle->get_size());
					anchor_rects[1] = Rect2(anchors_pos[1] - Vector2(0.0, anchor_handle->get_size().y), Point2(-anchor_handle->get_size().x, anchor_handle->get_size().y));
					anchor_rects[2] = Rect2(anchors_pos[2], -anchor_handle->get_size());
					anchor_rects[3] = Rect2(anchors_pos[3] - Vector2(anchor_handle->get_size().x, 0.0), Point2(anchor_handle->get_size().x, -anchor_handle->get_size().y));

					for (int i = 0; i < 4; i++) {
						anchor_handle->draw_rect(ci, anchor_rects[i]);
					}

					// Draw the margin values and the node width/height when dragging control side
					float ratio = 0.33;
					Transform2D parent_transform = xform * control->get_transform().affine_inverse();
					float node_pos_in_parent[4];

					node_pos_in_parent[0] = control->get_anchor(MARGIN_LEFT) * control->get_parent_area_size().width + control->get_margin(MARGIN_LEFT);
					node_pos_in_parent[1] = control->get_anchor(MARGIN_TOP) * control->get_parent_area_size().height + control->get_margin(MARGIN_TOP);
					node_pos_in_parent[2] = control->get_anchor(MARGIN_RIGHT) * control->get_parent_area_size().width + control->get_margin(MARGIN_RIGHT);
					node_pos_in_parent[3] = control->get_anchor(MARGIN_BOTTOM) * control->get_parent_area_size().height + control->get_margin(MARGIN_BOTTOM);

					switch (drag) {
						case DRAG_LEFT:
						case DRAG_TOP_LEFT:
						case DRAG_BOTTOM_LEFT:
							_draw_margin_at_position(control->get_size().width, parent_transform.xform(Vector2((node_pos_in_parent[0] + node_pos_in_parent[2]) / 2, node_pos_in_parent[3])) + Vector2(0, 5), MARGIN_BOTTOM);
						case DRAG_ALL:
							Point2 start = Vector2(node_pos_in_parent[0], Math::lerp(node_pos_in_parent[1], node_pos_in_parent[3], ratio));
							Point2 end = start - Vector2(control->get_margin(MARGIN_LEFT), 0);
							_draw_margin_at_position(control->get_margin(MARGIN_LEFT), parent_transform.xform((start + end) / 2), MARGIN_TOP);
							viewport->draw_line(parent_transform.xform(start), parent_transform.xform(end), color_base, 1);
							break;
					}
					switch (drag) {
						case DRAG_RIGHT:
						case DRAG_TOP_RIGHT:
						case DRAG_BOTTOM_RIGHT:
							_draw_margin_at_position(control->get_size().width, parent_transform.xform(Vector2((node_pos_in_parent[0] + node_pos_in_parent[2]) / 2, node_pos_in_parent[3])) + Vector2(0, 5), MARGIN_BOTTOM);
						case DRAG_ALL:
							Point2 start = Vector2(node_pos_in_parent[2], Math::lerp(node_pos_in_parent[3], node_pos_in_parent[1], ratio));
							Point2 end = start - Vector2(control->get_margin(MARGIN_RIGHT), 0);
							_draw_margin_at_position(control->get_margin(MARGIN_RIGHT), parent_transform.xform((start + end) / 2), MARGIN_BOTTOM);
							viewport->draw_line(parent_transform.xform(start), parent_transform.xform(end), color_base, 1);
							break;
					}
					switch (drag) {
						case DRAG_TOP:
						case DRAG_TOP_LEFT:
						case DRAG_TOP_RIGHT:
							_draw_margin_at_position(control->get_size().height, parent_transform.xform(Vector2(node_pos_in_parent[2], (node_pos_in_parent[1] + node_pos_in_parent[3]) / 2)) + Vector2(5, 0), MARGIN_RIGHT);
						case DRAG_ALL:
							Point2 start = Vector2(Math::lerp(node_pos_in_parent[0], node_pos_in_parent[2], ratio), node_pos_in_parent[1]);
							Point2 end = start - Vector2(0, control->get_margin(MARGIN_TOP));
							_draw_margin_at_position(control->get_margin(MARGIN_TOP), parent_transform.xform((start + end) / 2), MARGIN_LEFT);
							viewport->draw_line(parent_transform.xform(start), parent_transform.xform(end), color_base, 1);
							break;
					}
					switch (drag) {
						case DRAG_BOTTOM:
						case DRAG_BOTTOM_LEFT:
						case DRAG_BOTTOM_RIGHT:
							_draw_margin_at_position(control->get_size().height, parent_transform.xform(Vector2(node_pos_in_parent[2], (node_pos_in_parent[1] + node_pos_in_parent[3]) / 2) + Vector2(5, 0)), MARGIN_RIGHT);
						case DRAG_ALL:
							Point2 start = Vector2(Math::lerp(node_pos_in_parent[2], node_pos_in_parent[0], ratio), node_pos_in_parent[3]);
							Point2 end = start - Vector2(0, control->get_margin(MARGIN_BOTTOM));
							_draw_margin_at_position(control->get_margin(MARGIN_BOTTOM), parent_transform.xform((start + end) / 2), MARGIN_RIGHT);
							viewport->draw_line(parent_transform.xform(start), parent_transform.xform(end), color_base, 1);
							break;
					}

					switch (drag) {
						//Draw the ghost rect if the node if rotated/scaled
						case DRAG_LEFT:
						case DRAG_TOP_LEFT:
						case DRAG_TOP:
						case DRAG_TOP_RIGHT:
						case DRAG_RIGHT:
						case DRAG_BOTTOM_RIGHT:
						case DRAG_BOTTOM:
						case DRAG_BOTTOM_LEFT:
						case DRAG_ALL:
							if (control->get_rotation() != 0.0 || control->get_scale() != Vector2(1, 1)) {
								Rect2 rect = Rect2(Vector2(node_pos_in_parent[0], node_pos_in_parent[1]), control->get_size());
								viewport->draw_rect(parent_transform.xform(rect), color_base, false);
							}
							break;
					}
				}
			}

			if (tool == TOOL_SELECT) {

				for (int i = 0; i < 4; i++) {

					int prev = (i + 3) % 4;
					int next = (i + 1) % 4;

					Vector2 ofs = ((endpoints[i] - endpoints[prev]).normalized() + ((endpoints[i] - endpoints[next]).normalized())).normalized();
					ofs *= 1.4144 * (select_handle->get_size().width / 2);

					select_handle->draw(ci, (endpoints[i] + ofs - (select_handle->get_size() / 2)).floor());

					ofs = (endpoints[i] + endpoints[next]) / 2;
					ofs += (endpoints[next] - endpoints[i]).tangent().normalized() * (select_handle->get_size().width / 2);

					select_handle->draw(ci, (ofs - (select_handle->get_size() / 2)).floor());
				}
			}
		}
	}
	pivot_button->set_disabled(!pivot_found);

	if (box_selecting) {
		Point2 bsfrom = transform.xform(drag_from);
		Point2 bsto = transform.xform(box_selecting_to);

		VisualServer::get_singleton()->canvas_item_add_rect(ci, Rect2(bsfrom, bsto - bsfrom), Color(0.7, 0.7, 1.0, 0.3));
	}

	Color rotate_color(0.4, 0.7, 1.0, 0.8);
	if (drag == DRAG_ROTATE) {
		VisualServer::get_singleton()->canvas_item_add_line(ci, transform.xform(display_rotate_from), transform.xform(display_rotate_to), rotate_color);
	}
}

void CanvasItemEditor::_draw_axis() {
	RID ci = viewport->get_canvas_item();

	Color x_axis_color(1.0, 0.4, 0.4, 0.6);
	Color y_axis_color(0.4, 1.0, 0.4, 0.6);
	Color area_axis_color(0.4, 0.4, 1.0, 0.4);

	Point2 origin = transform.get_origin();
	VisualServer::get_singleton()->canvas_item_add_line(ci, Point2(0, origin.y), Point2(viewport->get_size().x, origin.y), x_axis_color);
	VisualServer::get_singleton()->canvas_item_add_line(ci, Point2(origin.x, 0), Point2(origin.x, viewport->get_size().y), y_axis_color);

	Size2 screen_size = Size2(ProjectSettings::get_singleton()->get("display/window/size/width"), ProjectSettings::get_singleton()->get("display/window/size/height"));

	Vector2 screen_endpoints[4] = {
		transform.xform(Vector2(0, 0)),
		transform.xform(Vector2(screen_size.width, 0)),
		transform.xform(Vector2(screen_size.width, screen_size.height)),
		transform.xform(Vector2(0, screen_size.height))
	};

	for (int i = 0; i < 4; i++) {
		VisualServer::get_singleton()->canvas_item_add_line(ci, screen_endpoints[i], screen_endpoints[(i + 1) % 4], area_axis_color);
	}
}

void CanvasItemEditor::_draw_bones() {
	RID ci = viewport->get_canvas_item();

	if (skeleton_show_bones) {
		int bone_width = EditorSettings::get_singleton()->get("editors/2d/bone_width");
		Color bone_color1 = EditorSettings::get_singleton()->get("editors/2d/bone_color1");
		Color bone_color2 = EditorSettings::get_singleton()->get("editors/2d/bone_color2");
		Color bone_ik_color = EditorSettings::get_singleton()->get("editors/2d/bone_ik_color");
		Color bone_selected_color = EditorSettings::get_singleton()->get("editors/2d/bone_selected_color");

		for (Map<ObjectID, BoneList>::Element *E = bone_list.front(); E; E = E->next()) {

			E->get().from = Vector2();
			E->get().to = Vector2();

			Node2D *n2d = Object::cast_to<Node2D>(ObjectDB::get_instance(E->get().bone));
			if (!n2d)
				continue;

			if (!n2d->get_parent())
				continue;

			CanvasItem *pi = n2d->get_parent_item();

			Node2D *pn2d = Object::cast_to<Node2D>(n2d->get_parent());

			if (!pn2d)
				continue;

			Vector2 from = transform.xform(pn2d->get_global_position());
			Vector2 to = transform.xform(n2d->get_global_position());

			E->get().from = from;
			E->get().to = to;

			Vector2 rel = to - from;
			Vector2 relt = rel.tangent().normalized() * bone_width;

			Vector<Vector2> bone_shape;
			bone_shape.push_back(from);
			bone_shape.push_back(from + rel * 0.2 + relt);
			bone_shape.push_back(to);
			bone_shape.push_back(from + rel * 0.2 - relt);
			Vector<Color> colors;
			if (pi->has_meta("_edit_ik_")) {

				colors.push_back(bone_ik_color);
				colors.push_back(bone_ik_color);
				colors.push_back(bone_ik_color);
				colors.push_back(bone_ik_color);
			} else {
				colors.push_back(bone_color1);
				colors.push_back(bone_color2);
				colors.push_back(bone_color1);
				colors.push_back(bone_color2);
			}

			VisualServer::get_singleton()->canvas_item_add_primitive(ci, bone_shape, colors, Vector<Vector2>(), RID());

			if (editor_selection->is_selected(pi)) {
				for (int i = 0; i < bone_shape.size(); i++) {

					VisualServer::get_singleton()->canvas_item_add_line(ci, bone_shape[i], bone_shape[(i + 1) % bone_shape.size()], bone_selected_color, 2);
				}
			}
		}
	}
}

void CanvasItemEditor::_draw_locks_and_groups(Node *p_node, const Transform2D &p_xform) {
	ERR_FAIL_COND(!p_node);

	RID viewport_ci = viewport->get_canvas_item();

	Transform2D transform_ci = p_xform;
	CanvasItem *ci = Object::cast_to<CanvasItem>(p_node);
	if (ci)
		transform_ci = transform_ci * ci->get_transform();

	for (int i = p_node->get_child_count() - 1; i >= 0; i--) {
		_draw_locks_and_groups(p_node->get_child(i), transform_ci);
	}

	if (ci) {
		Ref<Texture> lock = get_icon("LockViewport", "EditorIcons");
		if (p_node->has_meta("_edit_lock_")) {
			lock->draw(viewport_ci, transform_ci.xform(Point2(0, 0)));
		}

		Ref<Texture> group = get_icon("GroupViewport", "EditorIcons");
		if (ci->has_meta("_edit_group_")) {
			Vector2 ofs = transform_ci.xform(Point2(0, 0));
			if (ci->has_meta("_edit_lock_"))
				ofs = Point2(ofs.x + lock->get_size().x, ofs.y);
			group->draw(viewport_ci, ofs);
		}
	}
}

void CanvasItemEditor::_build_bones_list(Node *p_node) {
	ERR_FAIL_COND(!p_node);

	for (int i = 0; i < p_node->get_child_count(); i++) {
		_build_bones_list(p_node->get_child(i));
	}

	CanvasItem *c = Object::cast_to<CanvasItem>(p_node);
	if (c && c->is_visible_in_tree()) {
		if (c->has_meta("_edit_bone_")) {

			ObjectID id = c->get_instance_id();
			if (!bone_list.has(id)) {
				BoneList bone;
				bone.bone = id;
				bone_list[id] = bone;
			}

			bone_list[id].last_pass = bone_last_frame;
		}
	}
}

void CanvasItemEditor::_get_encompassing_rect(Node *p_node, Rect2 &r_rect, const Transform2D &p_xform) {
	ERR_FAIL_COND(!p_node);

	for (int i = 0; i < p_node->get_child_count(); i++) {
		_get_encompassing_rect(p_node->get_child(i), r_rect, p_xform);
	}

	CanvasItem *c = Object::cast_to<CanvasItem>(p_node);
	if (c && c->is_visible_in_tree()) {
		Rect2 rect = c->_edit_get_rect();
		Transform2D xform = p_xform * c->get_transform();
		r_rect.expand_to(xform.xform(rect.position));
		r_rect.expand_to(xform.xform(rect.position + Point2(rect.size.x, 0)));
		r_rect.expand_to(xform.xform(rect.position + Point2(0, rect.size.y)));
		r_rect.expand_to(xform.xform(rect.position + rect.size));
	}
}

void CanvasItemEditor::_draw_viewport_base() {
	if (show_rulers)
		_draw_rulers();
	if (show_guides)
		_draw_guides();
	_draw_focus();
}

void CanvasItemEditor::_draw_viewport() {

	// hide/show buttons depending on the selection
	bool all_locked = true;
	bool all_group = true;
	List<Node *> &selection = editor_selection->get_selected_node_list();
	if (selection.empty()) {
		all_locked = false;
		all_group = false;
	} else {
		for (List<Node *>::Element *E = selection.front(); E; E = E->next()) {
			if (Object::cast_to<CanvasItem>(E->get()) && !Object::cast_to<CanvasItem>(E->get())->has_meta("_edit_lock_")) {
				all_locked = false;
				break;
			}
		}
		for (List<Node *>::Element *E = selection.front(); E; E = E->next()) {
			if (Object::cast_to<CanvasItem>(E->get()) && !Object::cast_to<CanvasItem>(E->get())->has_meta("_edit_group_")) {
				all_group = false;
				break;
			}
		}
	}

	lock_button->set_visible(!all_locked);
	lock_button->set_disabled(selection.empty());
	unlock_button->set_visible(all_locked);
	group_button->set_visible(!all_group);
	group_button->set_disabled(selection.empty());
	ungroup_button->set_visible(all_group);

	_update_scrollbars();

	_draw_grid();
	_draw_selection();
	_draw_axis();
	if (editor->get_edited_scene())
		_draw_locks_and_groups(editor->get_edited_scene(), transform);

	RID ci = viewport->get_canvas_item();
	VisualServer::get_singleton()->canvas_item_add_set_transform(ci, Transform2D());

	EditorPluginList *over_plugin_list = editor->get_editor_plugins_over();
	if (!over_plugin_list->empty()) {
		over_plugin_list->forward_draw_over_viewport(viewport);
	}
	EditorPluginList *force_over_plugin_list = editor->get_editor_plugins_force_over();
	if (!force_over_plugin_list->empty()) {
		force_over_plugin_list->forward_force_draw_over_viewport(viewport);
	}

	_draw_bones();
}

void CanvasItemEditor::_notification(int p_what) {

	if (p_what == NOTIFICATION_PHYSICS_PROCESS) {

		EditorNode::get_singleton()->get_scene_root()->set_snap_controls_to_pixels(GLOBAL_GET("gui/common/snap_controls_to_pixels"));

		List<Node *> &selection = editor_selection->get_selected_node_list();

		bool all_control = true;
		bool has_control = false;

		for (List<Node *>::Element *E = selection.front(); E; E = E->next()) {

			CanvasItem *canvas_item = Object::cast_to<CanvasItem>(E->get());
			if (!canvas_item || !canvas_item->is_visible_in_tree())
				continue;

			if (canvas_item->get_viewport() != EditorNode::get_singleton()->get_scene_root())
				continue;

			if (Object::cast_to<Control>(canvas_item))
				has_control = true;
			else
				all_control = false;

			CanvasItemEditorSelectedItem *se = editor_selection->get_node_editor_data<CanvasItemEditorSelectedItem>(canvas_item);
			if (!se)
				continue;

			Rect2 r = canvas_item->_edit_get_rect();
			Transform2D xform = canvas_item->get_transform();

			if (r != se->prev_rect || xform != se->prev_xform) {
				viewport->update();
				se->prev_rect = r;
				se->prev_xform = xform;
			}

			if (Object::cast_to<Control>(canvas_item)) {
				float anchors[4];
				Vector2 pivot;

				pivot = Object::cast_to<Control>(canvas_item)->get_pivot_offset();
				anchors[MARGIN_LEFT] = Object::cast_to<Control>(canvas_item)->get_anchor(MARGIN_LEFT);
				anchors[MARGIN_RIGHT] = Object::cast_to<Control>(canvas_item)->get_anchor(MARGIN_RIGHT);
				anchors[MARGIN_TOP] = Object::cast_to<Control>(canvas_item)->get_anchor(MARGIN_TOP);
				anchors[MARGIN_BOTTOM] = Object::cast_to<Control>(canvas_item)->get_anchor(MARGIN_BOTTOM);

				if (pivot != se->prev_pivot || anchors[MARGIN_LEFT] != se->prev_anchors[MARGIN_LEFT] || anchors[MARGIN_RIGHT] != se->prev_anchors[MARGIN_RIGHT] || anchors[MARGIN_TOP] != se->prev_anchors[MARGIN_TOP] || anchors[MARGIN_BOTTOM] != se->prev_anchors[MARGIN_BOTTOM]) {
					viewport->update();
					viewport_base->update();
					se->prev_pivot = pivot;
					se->prev_anchors[MARGIN_LEFT] = anchors[MARGIN_LEFT];
					se->prev_anchors[MARGIN_RIGHT] = anchors[MARGIN_RIGHT];
					se->prev_anchors[MARGIN_TOP] = anchors[MARGIN_TOP];
					se->prev_anchors[MARGIN_BOTTOM] = anchors[MARGIN_BOTTOM];
				}
			}
		}

		if (all_control && has_control)
			presets_menu->show();
		else
			presets_menu->hide();

		for (Map<ObjectID, BoneList>::Element *E = bone_list.front(); E; E = E->next()) {

			Object *b = ObjectDB::get_instance(E->get().bone);
			if (!b) {

				viewport->update();
				break;
			}

			Node2D *b2 = Object::cast_to<Node2D>(b);
			if (!b2) {
				continue;
			}

			if (b2->get_global_transform() != E->get().xform) {

				E->get().xform = b2->get_global_transform();
				viewport->update();
			}
		}
	}

	if (p_what == NOTIFICATION_ENTER_TREE) {

		select_sb->set_texture(get_icon("EditorRect2D", "EditorIcons"));
		for (int i = 0; i < 4; i++) {
			select_sb->set_margin_size(Margin(i), 4);
			select_sb->set_default_margin(Margin(i), 4);
		}

		AnimationPlayerEditor::singleton->get_key_editor()->connect("visibility_changed", this, "_keying_changed");
		_keying_changed();

	} else if (p_what == EditorSettings::NOTIFICATION_EDITOR_SETTINGS_CHANGED) {

		select_sb->set_texture(get_icon("EditorRect2D", "EditorIcons"));
	}

	if (p_what == NOTIFICATION_ENTER_TREE || p_what == EditorSettings::NOTIFICATION_EDITOR_SETTINGS_CHANGED) {
		select_button->set_icon(get_icon("ToolSelect", "EditorIcons"));
		list_select_button->set_icon(get_icon("ListSelect", "EditorIcons"));
		move_button->set_icon(get_icon("ToolMove", "EditorIcons"));
		rotate_button->set_icon(get_icon("ToolRotate", "EditorIcons"));
		snap_button->set_icon(get_icon("Snap", "EditorIcons"));
		snap_config_menu->set_icon(get_icon("GuiMiniTabMenu", "EditorIcons"));
		skeleton_menu->set_icon(get_icon("Bone", "EditorIcons"));
		pan_button->set_icon(get_icon("ToolPan", "EditorIcons"));
		pivot_button->set_icon(get_icon("EditPivot", "EditorIcons"));
		select_handle = get_icon("EditorHandle", "EditorIcons");
		anchor_handle = get_icon("EditorControlAnchor", "EditorIcons");
		lock_button->set_icon(get_icon("Lock", "EditorIcons"));
		unlock_button->set_icon(get_icon("Unlock", "EditorIcons"));
		group_button->set_icon(get_icon("Group", "EditorIcons"));
		ungroup_button->set_icon(get_icon("Ungroup", "EditorIcons"));
		key_loc_button->set_icon(get_icon("KeyPosition", "EditorIcons"));
		key_rot_button->set_icon(get_icon("KeyRotation", "EditorIcons"));
		key_scale_button->set_icon(get_icon("KeyScale", "EditorIcons"));
		key_insert_button->set_icon(get_icon("Key", "EditorIcons"));

		zoom_minus->set_icon(get_icon("ZoomLess", "EditorIcons"));
		zoom_reset->set_icon(get_icon("ZoomReset", "EditorIcons"));
		zoom_plus->set_icon(get_icon("ZoomMore", "EditorIcons"));

		presets_menu->set_icon(get_icon("ControlLayout", "EditorIcons"));
		PopupMenu *p = presets_menu->get_popup();

		p->clear();
		p->add_icon_item(get_icon("ControlAlignTopLeft", "EditorIcons"), "Top Left", ANCHORS_AND_MARGINS_PRESET_TOP_LEFT);
		p->add_icon_item(get_icon("ControlAlignTopRight", "EditorIcons"), "Top Right", ANCHORS_AND_MARGINS_PRESET_TOP_RIGHT);
		p->add_icon_item(get_icon("ControlAlignBottomRight", "EditorIcons"), "Bottom Right", ANCHORS_AND_MARGINS_PRESET_BOTTOM_RIGHT);
		p->add_icon_item(get_icon("ControlAlignBottomLeft", "EditorIcons"), "Bottom Left", ANCHORS_AND_MARGINS_PRESET_BOTTOM_LEFT);
		p->add_separator();
		p->add_icon_item(get_icon("ControlAlignLeftCenter", "EditorIcons"), "Center Left", ANCHORS_AND_MARGINS_PRESET_CENTER_LEFT);
		p->add_icon_item(get_icon("ControlAlignTopCenter", "EditorIcons"), "Center Top", ANCHORS_AND_MARGINS_PRESET_CENTER_TOP);
		p->add_icon_item(get_icon("ControlAlignRightCenter", "EditorIcons"), "Center Right", ANCHORS_AND_MARGINS_PRESET_CENTER_RIGHT);
		p->add_icon_item(get_icon("ControlAlignBottomCenter", "EditorIcons"), "Center Bottom", ANCHORS_AND_MARGINS_PRESET_CENTER_BOTTOM);
		p->add_icon_item(get_icon("ControlAlignCenter", "EditorIcons"), "Center", ANCHORS_AND_MARGINS_PRESET_CENTER);
		p->add_separator();
		p->add_icon_item(get_icon("ControlAlignLeftWide", "EditorIcons"), "Left Wide", ANCHORS_AND_MARGINS_PRESET_LEFT_WIDE);
		p->add_icon_item(get_icon("ControlAlignTopWide", "EditorIcons"), "Top Wide", ANCHORS_AND_MARGINS_PRESET_TOP_WIDE);
		p->add_icon_item(get_icon("ControlAlignRightWide", "EditorIcons"), "Right Wide", ANCHORS_AND_MARGINS_PRESET_RIGHT_WIDE);
		p->add_icon_item(get_icon("ControlAlignBottomWide", "EditorIcons"), "Bottom Wide", ANCHORS_AND_MARGINS_PRESET_BOTTOM_WIDE);
		p->add_icon_item(get_icon("ControlVcenterWide", "EditorIcons"), "VCenter Wide ", ANCHORS_AND_MARGINS_PRESET_VCENTER_WIDE);
		p->add_icon_item(get_icon("ControlHcenterWide", "EditorIcons"), "HCenter Wide ", ANCHORS_AND_MARGINS_PRESET_HCENTER_WIDE);
		p->add_separator();
		p->add_icon_item(get_icon("ControlAlignWide", "EditorIcons"), "Full Rect", ANCHORS_AND_MARGINS_PRESET_WIDE);
		p->add_separator();
		p->add_submenu_item(TTR("Anchors only"), "Anchors");
		p->set_item_icon(20, get_icon("Anchor", "EditorIcons"));

		anchors_popup->clear();
		anchors_popup->add_icon_item(get_icon("ControlAlignTopLeft", "EditorIcons"), "Top Left", ANCHORS_PRESET_TOP_LEFT);
		anchors_popup->add_icon_item(get_icon("ControlAlignTopRight", "EditorIcons"), "Top Right", ANCHORS_PRESET_TOP_RIGHT);
		anchors_popup->add_icon_item(get_icon("ControlAlignBottomRight", "EditorIcons"), "Bottom Right", ANCHORS_PRESET_BOTTOM_RIGHT);
		anchors_popup->add_icon_item(get_icon("ControlAlignBottomLeft", "EditorIcons"), "Bottom Left", ANCHORS_PRESET_BOTTOM_LEFT);
		anchors_popup->add_separator();
		anchors_popup->add_icon_item(get_icon("ControlAlignLeftCenter", "EditorIcons"), "Center Left", ANCHORS_PRESET_CENTER_LEFT);
		anchors_popup->add_icon_item(get_icon("ControlAlignTopCenter", "EditorIcons"), "Center Top", ANCHORS_PRESET_CENTER_TOP);
		anchors_popup->add_icon_item(get_icon("ControlAlignRightCenter", "EditorIcons"), "Center Right", ANCHORS_PRESET_CENTER_RIGHT);
		anchors_popup->add_icon_item(get_icon("ControlAlignBottomCenter", "EditorIcons"), "Center Bottom", ANCHORS_PRESET_CENTER_BOTTOM);
		anchors_popup->add_icon_item(get_icon("ControlAlignCenter", "EditorIcons"), "Center", ANCHORS_PRESET_CENTER);
		anchors_popup->add_separator();
		anchors_popup->add_icon_item(get_icon("ControlAlignLeftWide", "EditorIcons"), "Left Wide", ANCHORS_PRESET_LEFT_WIDE);
		anchors_popup->add_icon_item(get_icon("ControlAlignTopWide", "EditorIcons"), "Top Wide", ANCHORS_PRESET_TOP_WIDE);
		anchors_popup->add_icon_item(get_icon("ControlAlignRightWide", "EditorIcons"), "Right Wide", ANCHORS_PRESET_RIGHT_WIDE);
		anchors_popup->add_icon_item(get_icon("ControlAlignBottomWide", "EditorIcons"), "Bottom Wide", ANCHORS_PRESET_BOTTOM_WIDE);
		anchors_popup->add_icon_item(get_icon("ControlVcenterWide", "EditorIcons"), "VCenter Wide ", ANCHORS_PRESET_VCENTER_WIDE);
		anchors_popup->add_icon_item(get_icon("ControlHcenterWide", "EditorIcons"), "HCenter Wide ", ANCHORS_PRESET_HCENTER_WIDE);
		anchors_popup->add_separator();
		anchors_popup->add_icon_item(get_icon("ControlAlignWide", "EditorIcons"), "Full Rect", ANCHORS_PRESET_WIDE);
	}
}

void CanvasItemEditor::edit(CanvasItem *p_canvas_item) {

	drag = DRAG_NONE;

	// Clear the selection
	editor_selection->clear(); //_clear_canvas_items();
	editor_selection->add_node(p_canvas_item);
	//_add_canvas_item(p_canvas_item);
	viewport->update();
	viewport_base->update();
}

void CanvasItemEditor::_update_scrollbars() {

	updating_scroll = true;

	if (show_rulers)
		viewport_scrollable->set_begin(Point2(RULER_WIDTH, RULER_WIDTH));
	else
		viewport_scrollable->set_begin(Point2());

	Size2 size = viewport->get_size();
	Size2 hmin = h_scroll->get_minimum_size();
	Size2 vmin = v_scroll->get_minimum_size();

	v_scroll->set_begin(Point2(size.width - vmin.width, 0));
	v_scroll->set_end(Point2(size.width, size.height));

	h_scroll->set_begin(Point2(0, size.height - hmin.height));
	h_scroll->set_end(Point2(size.width - vmin.width, size.height));

	Size2 screen_rect = Size2(ProjectSettings::get_singleton()->get("display/window/size/width"), ProjectSettings::get_singleton()->get("display/window/size/height"));

	Rect2 local_rect = Rect2(Point2(), viewport->get_size() - Size2(vmin.width, hmin.height));

	Rect2 canvas_item_rect = Rect2(Point2(), screen_rect);

	bone_last_frame++;

	if (editor->get_edited_scene()) {
		_build_bones_list(editor->get_edited_scene());
		_get_encompassing_rect(editor->get_edited_scene(), canvas_item_rect, Transform2D());
	}

	List<Map<ObjectID, BoneList>::Element *> bone_to_erase;

	for (Map<ObjectID, BoneList>::Element *E = bone_list.front(); E; E = E->next()) {

		if (E->get().last_pass != bone_last_frame) {
			bone_to_erase.push_back(E);
		}
	}

	while (bone_to_erase.size()) {
		bone_list.erase(bone_to_erase.front()->get());
		bone_to_erase.pop_front();
	}

	//expand area so it's easier to do animations and stuff at 0,0
	canvas_item_rect.size += screen_rect * 2;
	canvas_item_rect.position -= screen_rect;

	Point2 ofs;

	if (canvas_item_rect.size.height <= (local_rect.size.y / zoom)) {
		v_scroll->hide();
		ofs.y = canvas_item_rect.position.y;
	} else {

		v_scroll->show();
		v_scroll->set_min(canvas_item_rect.position.y);
		v_scroll->set_max(canvas_item_rect.position.y + canvas_item_rect.size.y);
		v_scroll->set_page(local_rect.size.y / zoom);
		if (first_update) {
			//so 0,0 is visible
			v_scroll->set_value(-10);
			h_scroll->set_value(-10);
			first_update = false;
		}

		ofs.y = v_scroll->get_value();
	}

	if (canvas_item_rect.size.width <= (local_rect.size.x / zoom)) {

		h_scroll->hide();
		ofs.x = canvas_item_rect.position.x;
	} else {

		h_scroll->show();
		h_scroll->set_min(canvas_item_rect.position.x);
		h_scroll->set_max(canvas_item_rect.position.x + canvas_item_rect.size.x);
		h_scroll->set_page(local_rect.size.x / zoom);
		ofs.x = h_scroll->get_value();
	}

	//transform=Matrix32();
	transform.elements[2] = -ofs * zoom;

	editor->get_scene_root()->set_global_canvas_transform(transform);

	updating_scroll = false;

	//transform.scale_basis(Vector2(zoom,zoom));
}

void CanvasItemEditor::_update_scroll(float) {

	if (updating_scroll)
		return;

	Point2 ofs;
	ofs.x = h_scroll->get_value();
	ofs.y = v_scroll->get_value();

	//current_window->set_scroll(-ofs);

	transform = Transform2D();

	transform.scale_basis(Size2(zoom, zoom));
	transform.elements[2] = -ofs;

	editor->get_scene_root()->set_global_canvas_transform(transform);

	viewport->update();
	viewport_base->update();
}

void CanvasItemEditor::_set_anchors_and_margins_preset(Control::LayoutPreset p_preset) {
	List<Node *> &selection = editor_selection->get_selected_node_list();

	undo_redo->create_action(TTR("Change Anchors and Margins"));
	for (List<Node *>::Element *E = selection.front(); E; E = E->next()) {

		Control *c = Object::cast_to<Control>(E->get());

		undo_redo->add_do_method(c, "set_anchors_preset", p_preset);
		switch (p_preset) {
			case PRESET_TOP_LEFT:
			case PRESET_TOP_RIGHT:
			case PRESET_BOTTOM_LEFT:
			case PRESET_BOTTOM_RIGHT:
			case PRESET_CENTER_LEFT:
			case PRESET_CENTER_TOP:
			case PRESET_CENTER_RIGHT:
			case PRESET_CENTER_BOTTOM:
			case PRESET_CENTER:
				undo_redo->add_do_method(c, "set_margins_preset", p_preset, Control::PRESET_MODE_KEEP_SIZE);
				break;
			case PRESET_LEFT_WIDE:
			case PRESET_TOP_WIDE:
			case PRESET_RIGHT_WIDE:
			case PRESET_BOTTOM_WIDE:
			case PRESET_VCENTER_WIDE:
			case PRESET_HCENTER_WIDE:
			case PRESET_WIDE:
				undo_redo->add_do_method(c, "set_margins_preset", p_preset, Control::PRESET_MODE_MINSIZE);
				break;
		}
		undo_redo->add_undo_method(c, "set_anchor", MARGIN_LEFT, c->get_anchor(MARGIN_LEFT));
		undo_redo->add_undo_method(c, "set_anchor", MARGIN_TOP, c->get_anchor(MARGIN_TOP));
		undo_redo->add_undo_method(c, "set_anchor", MARGIN_RIGHT, c->get_anchor(MARGIN_RIGHT));
		undo_redo->add_undo_method(c, "set_anchor", MARGIN_BOTTOM, c->get_anchor(MARGIN_BOTTOM));
		undo_redo->add_undo_method(c, "set_margin", MARGIN_LEFT, c->get_margin(MARGIN_LEFT));
		undo_redo->add_undo_method(c, "set_margin", MARGIN_TOP, c->get_margin(MARGIN_TOP));
		undo_redo->add_undo_method(c, "set_margin", MARGIN_RIGHT, c->get_margin(MARGIN_RIGHT));
		undo_redo->add_undo_method(c, "set_margin", MARGIN_BOTTOM, c->get_margin(MARGIN_BOTTOM));
	}

	undo_redo->commit_action();
}

void CanvasItemEditor::_set_anchors_preset(Control::LayoutPreset p_preset) {
	List<Node *> &selection = editor_selection->get_selected_node_list();

	undo_redo->create_action(TTR("Change Anchors"));
	for (List<Node *>::Element *E = selection.front(); E; E = E->next()) {

		Control *c = Object::cast_to<Control>(E->get());

		undo_redo->add_do_method(c, "set_anchors_preset", p_preset);
		undo_redo->add_undo_method(c, "set_anchor", MARGIN_LEFT, c->get_anchor(MARGIN_LEFT));
		undo_redo->add_undo_method(c, "set_anchor", MARGIN_TOP, c->get_anchor(MARGIN_TOP));
		undo_redo->add_undo_method(c, "set_anchor", MARGIN_RIGHT, c->get_anchor(MARGIN_RIGHT));
		undo_redo->add_undo_method(c, "set_anchor", MARGIN_BOTTOM, c->get_anchor(MARGIN_BOTTOM));
	}

	undo_redo->commit_action();
}

void CanvasItemEditor::_zoom_on_position(float p_zoom, Point2 p_position) {
	if (p_zoom < MIN_ZOOM || p_zoom > MAX_ZOOM)
		return;

	float prev_zoom = zoom;
	zoom = p_zoom;
	Point2 ofs = p_position;
	ofs = ofs / prev_zoom - ofs / zoom;
	h_scroll->set_value(Math::round(h_scroll->get_value() + ofs.x));
	v_scroll->set_value(Math::round(v_scroll->get_value() + ofs.y));

	_update_scroll(0);
	viewport->update();
	viewport_base->update();
}

void CanvasItemEditor::_zoom_minus() {
	_zoom_on_position(zoom / 2.0, viewport_scrollable->get_size() / 2.0);
}

void CanvasItemEditor::_zoom_reset() {
	_zoom_on_position(1.0, viewport_scrollable->get_size() / 2.0);
}

void CanvasItemEditor::_zoom_plus() {
	_zoom_on_position(zoom * 2.0, viewport_scrollable->get_size() / 2.0);
}

void CanvasItemEditor::_toggle_snap(bool p_status) {
	snap_active = p_status;
	viewport->update();
	viewport_base->update();
}

void CanvasItemEditor::_popup_callback(int p_op) {

	last_option = MenuOption(p_op);
	switch (p_op) {

		case SHOW_GRID: {
			show_grid = !show_grid;
			int idx = view_menu->get_popup()->get_item_index(SHOW_GRID);
			view_menu->get_popup()->set_item_checked(idx, show_grid);
			viewport->update();
			viewport_base->update();
		} break;
		case SNAP_USE_NODE_PARENT: {
			snap_node_parent = !snap_node_parent;
			int idx = smartsnap_config_popup->get_item_index(SNAP_USE_NODE_PARENT);
			smartsnap_config_popup->set_item_checked(idx, snap_node_parent);
		} break;
		case SNAP_USE_NODE_ANCHORS: {
			snap_node_anchors = !snap_node_anchors;
			int idx = smartsnap_config_popup->get_item_index(SNAP_USE_NODE_ANCHORS);
			smartsnap_config_popup->set_item_checked(idx, snap_node_anchors);
		} break;
		case SNAP_USE_NODE_SIDES: {
			snap_node_sides = !snap_node_sides;
			int idx = smartsnap_config_popup->get_item_index(SNAP_USE_NODE_SIDES);
			smartsnap_config_popup->set_item_checked(idx, snap_node_sides);
		} break;
		case SNAP_USE_OTHER_NODES: {
			snap_other_nodes = !snap_other_nodes;
			int idx = smartsnap_config_popup->get_item_index(SNAP_USE_OTHER_NODES);
			smartsnap_config_popup->set_item_checked(idx, snap_other_nodes);
		} break;
		case SNAP_USE_GUIDES: {
			snap_guides = !snap_guides;
			int idx = smartsnap_config_popup->get_item_index(SNAP_USE_GUIDES);
			smartsnap_config_popup->set_item_checked(idx, snap_guides);
		} break;
		case SNAP_USE_GRID: {
			snap_grid = !snap_grid;
			int idx = snap_config_menu->get_popup()->get_item_index(SNAP_USE_GRID);
			snap_config_menu->get_popup()->set_item_checked(idx, snap_grid);
		} break;
		case SNAP_USE_ROTATION: {
			snap_rotation = !snap_rotation;
			int idx = snap_config_menu->get_popup()->get_item_index(SNAP_USE_ROTATION);
			snap_config_menu->get_popup()->set_item_checked(idx, snap_rotation);
		} break;
		case SNAP_RELATIVE: {
			snap_relative = !snap_relative;
			int idx = snap_config_menu->get_popup()->get_item_index(SNAP_RELATIVE);
			snap_config_menu->get_popup()->set_item_checked(idx, snap_relative);
			viewport->update();
			viewport_base->update();
		} break;
		case SNAP_USE_PIXEL: {
			snap_pixel = !snap_pixel;
			int idx = snap_config_menu->get_popup()->get_item_index(SNAP_USE_PIXEL);
			snap_config_menu->get_popup()->set_item_checked(idx, snap_pixel);
		} break;
		case SNAP_CONFIGURE: {
			((SnapDialog *)snap_dialog)->set_fields(grid_offset, grid_step, snap_rotation_offset, snap_rotation_step);
			snap_dialog->popup_centered(Size2(220, 160));
		} break;
		case SKELETON_SHOW_BONES: {
			skeleton_show_bones = !skeleton_show_bones;
			int idx = skeleton_menu->get_popup()->get_item_index(SKELETON_SHOW_BONES);
			skeleton_menu->get_popup()->set_item_checked(idx, skeleton_show_bones);
			viewport->update();
		} break;
		case SHOW_HELPERS: {
			show_helpers = !show_helpers;
			int idx = view_menu->get_popup()->get_item_index(SHOW_HELPERS);
			view_menu->get_popup()->set_item_checked(idx, show_helpers);
			viewport->update();
		} break;
		case SHOW_RULERS: {
			show_rulers = !show_rulers;
			int idx = view_menu->get_popup()->get_item_index(SHOW_RULERS);
			view_menu->get_popup()->set_item_checked(idx, show_rulers);
			viewport->update();
			viewport_base->update();
		} break;
		case SHOW_GUIDES: {
			show_guides = !show_guides;
			int idx = view_menu->get_popup()->get_item_index(SHOW_GUIDES);
			view_menu->get_popup()->set_item_checked(idx, show_guides);
			viewport->update();
			viewport_base->update();
		} break;

		case LOCK_SELECTED: {

			List<Node *> &selection = editor_selection->get_selected_node_list();

			for (List<Node *>::Element *E = selection.front(); E; E = E->next()) {

				CanvasItem *canvas_item = Object::cast_to<CanvasItem>(E->get());
				if (!canvas_item || !canvas_item->is_visible_in_tree())
					continue;

				if (canvas_item->get_viewport() != EditorNode::get_singleton()->get_scene_root())
					continue;

				canvas_item->set_meta("_edit_lock_", true);
				emit_signal("item_lock_status_changed");
			}
			viewport->update();
		} break;
		case UNLOCK_SELECTED: {

			List<Node *> &selection = editor_selection->get_selected_node_list();

			for (List<Node *>::Element *E = selection.front(); E; E = E->next()) {

				CanvasItem *canvas_item = Object::cast_to<CanvasItem>(E->get());
				if (!canvas_item || !canvas_item->is_visible_in_tree())
					continue;

				if (canvas_item->get_viewport() != EditorNode::get_singleton()->get_scene_root())
					continue;

				canvas_item->set_meta("_edit_lock_", Variant());
				emit_signal("item_lock_status_changed");
			}

			viewport->update();

		} break;
		case GROUP_SELECTED: {

			List<Node *> &selection = editor_selection->get_selected_node_list();

			for (List<Node *>::Element *E = selection.front(); E; E = E->next()) {

				CanvasItem *canvas_item = Object::cast_to<CanvasItem>(E->get());
				if (!canvas_item || !canvas_item->is_visible_in_tree())
					continue;

				if (canvas_item->get_viewport() != EditorNode::get_singleton()->get_scene_root())
					continue;

				canvas_item->set_meta("_edit_group_", true);
				emit_signal("item_group_status_changed");
			}
			viewport->update();
		} break;
		case UNGROUP_SELECTED: {

			List<Node *> &selection = editor_selection->get_selected_node_list();

			for (List<Node *>::Element *E = selection.front(); E; E = E->next()) {

				CanvasItem *canvas_item = Object::cast_to<CanvasItem>(E->get());
				if (!canvas_item || !canvas_item->is_visible_in_tree())
					continue;

				if (canvas_item->get_viewport() != EditorNode::get_singleton()->get_scene_root())
					continue;

				canvas_item->set_meta("_edit_group_", Variant());
				emit_signal("item_group_status_changed");
			}

			viewport->update();

		} break;

		case ANCHORS_AND_MARGINS_PRESET_TOP_LEFT: {
			_set_anchors_and_margins_preset(PRESET_TOP_LEFT);
		} break;
		case ANCHORS_AND_MARGINS_PRESET_TOP_RIGHT: {
			_set_anchors_and_margins_preset(PRESET_TOP_RIGHT);
		} break;
		case ANCHORS_AND_MARGINS_PRESET_BOTTOM_LEFT: {
			_set_anchors_and_margins_preset(PRESET_BOTTOM_LEFT);
		} break;
		case ANCHORS_AND_MARGINS_PRESET_BOTTOM_RIGHT: {
			_set_anchors_and_margins_preset(PRESET_BOTTOM_RIGHT);
		} break;
		case ANCHORS_AND_MARGINS_PRESET_CENTER_LEFT: {
			_set_anchors_and_margins_preset(PRESET_CENTER_LEFT);
		} break;
		case ANCHORS_AND_MARGINS_PRESET_CENTER_RIGHT: {
			_set_anchors_and_margins_preset(PRESET_CENTER_RIGHT);
		} break;
		case ANCHORS_AND_MARGINS_PRESET_CENTER_TOP: {
			_set_anchors_and_margins_preset(PRESET_CENTER_TOP);
		} break;
		case ANCHORS_AND_MARGINS_PRESET_CENTER_BOTTOM: {
			_set_anchors_and_margins_preset(PRESET_CENTER_BOTTOM);
		} break;
		case ANCHORS_AND_MARGINS_PRESET_CENTER: {
			_set_anchors_and_margins_preset(PRESET_CENTER);
		} break;
		case ANCHORS_AND_MARGINS_PRESET_TOP_WIDE: {
			_set_anchors_and_margins_preset(PRESET_TOP_WIDE);
		} break;
		case ANCHORS_AND_MARGINS_PRESET_LEFT_WIDE: {
			_set_anchors_and_margins_preset(PRESET_LEFT_WIDE);
		} break;
		case ANCHORS_AND_MARGINS_PRESET_RIGHT_WIDE: {
			_set_anchors_and_margins_preset(PRESET_RIGHT_WIDE);
		} break;
		case ANCHORS_AND_MARGINS_PRESET_BOTTOM_WIDE: {
			_set_anchors_and_margins_preset(PRESET_BOTTOM_WIDE);
		} break;
		case ANCHORS_AND_MARGINS_PRESET_VCENTER_WIDE: {
			_set_anchors_and_margins_preset(PRESET_VCENTER_WIDE);
		} break;
		case ANCHORS_AND_MARGINS_PRESET_HCENTER_WIDE: {
			_set_anchors_and_margins_preset(PRESET_HCENTER_WIDE);
		} break;
		case ANCHORS_AND_MARGINS_PRESET_WIDE: {
			_set_anchors_and_margins_preset(Control::PRESET_WIDE);
		} break;

		case ANCHORS_PRESET_TOP_LEFT: {
			_set_anchors_preset(PRESET_TOP_LEFT);
		} break;
		case ANCHORS_PRESET_TOP_RIGHT: {
			_set_anchors_preset(PRESET_TOP_RIGHT);
		} break;
		case ANCHORS_PRESET_BOTTOM_LEFT: {
			_set_anchors_preset(PRESET_BOTTOM_LEFT);
		} break;
		case ANCHORS_PRESET_BOTTOM_RIGHT: {
			_set_anchors_preset(PRESET_BOTTOM_RIGHT);
		} break;
		case ANCHORS_PRESET_CENTER_LEFT: {
			_set_anchors_preset(PRESET_CENTER_LEFT);
		} break;
		case ANCHORS_PRESET_CENTER_RIGHT: {
			_set_anchors_preset(PRESET_CENTER_RIGHT);
		} break;
		case ANCHORS_PRESET_CENTER_TOP: {
			_set_anchors_preset(PRESET_CENTER_TOP);
		} break;
		case ANCHORS_PRESET_CENTER_BOTTOM: {
			_set_anchors_preset(PRESET_CENTER_BOTTOM);
		} break;
		case ANCHORS_PRESET_CENTER: {
			_set_anchors_preset(PRESET_CENTER);
		} break;
		case ANCHORS_PRESET_TOP_WIDE: {
			_set_anchors_preset(PRESET_TOP_WIDE);
		} break;
		case ANCHORS_PRESET_LEFT_WIDE: {
			_set_anchors_preset(PRESET_LEFT_WIDE);
		} break;
		case ANCHORS_PRESET_RIGHT_WIDE: {
			_set_anchors_preset(PRESET_RIGHT_WIDE);
		} break;
		case ANCHORS_PRESET_BOTTOM_WIDE: {
			_set_anchors_preset(PRESET_BOTTOM_WIDE);
		} break;
		case ANCHORS_PRESET_VCENTER_WIDE: {
			_set_anchors_preset(PRESET_VCENTER_WIDE);
		} break;
		case ANCHORS_PRESET_HCENTER_WIDE: {
			_set_anchors_preset(PRESET_HCENTER_WIDE);
		} break;
		case ANCHORS_PRESET_WIDE: {
			_set_anchors_preset(Control::PRESET_WIDE);
		} break;

		case ANIM_INSERT_KEY:
		case ANIM_INSERT_KEY_EXISTING: {

			bool existing = p_op == ANIM_INSERT_KEY_EXISTING;

			Map<Node *, Object *> &selection = editor_selection->get_selection();

			for (Map<Node *, Object *>::Element *E = selection.front(); E; E = E->next()) {

				CanvasItem *canvas_item = Object::cast_to<CanvasItem>(E->key());
				if (!canvas_item || !canvas_item->is_visible_in_tree())
					continue;

				if (canvas_item->get_viewport() != EditorNode::get_singleton()->get_scene_root())
					continue;

				if (Object::cast_to<Node2D>(canvas_item)) {
					Node2D *n2d = Object::cast_to<Node2D>(canvas_item);

					if (key_pos)
						AnimationPlayerEditor::singleton->get_key_editor()->insert_node_value_key(n2d, "position", n2d->get_position(), existing);
					if (key_rot)
						AnimationPlayerEditor::singleton->get_key_editor()->insert_node_value_key(n2d, "rotation_degrees", Math::rad2deg(n2d->get_rotation()), existing);
					if (key_scale)
						AnimationPlayerEditor::singleton->get_key_editor()->insert_node_value_key(n2d, "scale", n2d->get_scale(), existing);

					if (n2d->has_meta("_edit_bone_") && n2d->get_parent_item()) {
						//look for an IK chain
						List<Node2D *> ik_chain;

						Node2D *n = Object::cast_to<Node2D>(n2d->get_parent_item());
						bool has_chain = false;

						while (n) {

							ik_chain.push_back(n);
							if (n->has_meta("_edit_ik_")) {
								has_chain = true;
								break;
							}

							if (!n->get_parent_item())
								break;
							n = Object::cast_to<Node2D>(n->get_parent_item());
						}

						if (has_chain && ik_chain.size()) {

							for (List<Node2D *>::Element *F = ik_chain.front(); F; F = F->next()) {

								if (key_pos)
									AnimationPlayerEditor::singleton->get_key_editor()->insert_node_value_key(F->get(), "position", F->get()->get_position(), existing);
								if (key_rot)
									AnimationPlayerEditor::singleton->get_key_editor()->insert_node_value_key(F->get(), "rotation_degrees", Math::rad2deg(F->get()->get_rotation()), existing);
								if (key_scale)
									AnimationPlayerEditor::singleton->get_key_editor()->insert_node_value_key(F->get(), "scale", F->get()->get_scale(), existing);
							}
						}
					}

				} else if (Object::cast_to<Control>(canvas_item)) {

					Control *ctrl = Object::cast_to<Control>(canvas_item);

					if (key_pos)
						AnimationPlayerEditor::singleton->get_key_editor()->insert_node_value_key(ctrl, "rect_position", ctrl->get_position(), existing);
					if (key_rot)
						AnimationPlayerEditor::singleton->get_key_editor()->insert_node_value_key(ctrl, "rect_rotation", ctrl->get_rotation_degrees(), existing);
					if (key_scale)
						AnimationPlayerEditor::singleton->get_key_editor()->insert_node_value_key(ctrl, "rect_size", ctrl->get_size(), existing);
				}
			}

		} break;
		case ANIM_INSERT_POS: {

			key_pos = key_loc_button->is_pressed();
		} break;
		case ANIM_INSERT_ROT: {

			key_rot = key_rot_button->is_pressed();
		} break;
		case ANIM_INSERT_SCALE: {

			key_scale = key_scale_button->is_pressed();
		} break;
		/*
		case ANIM_INSERT_POS_ROT
		case ANIM_INSERT_POS_SCALE:
		case ANIM_INSERT_ROT_SCALE:
		case ANIM_INSERT_POS_ROT_SCALE: {
			static const bool key_toggles[7][3]={
				{true,false,false},
				{false,true,false},
				{false,false,true},
				{true,true,false},
				{true,false,true},
				{false,true,true},
				{true,true,true}
			};
			key_pos=key_toggles[p_op-ANIM_INSERT_POS][0];
			key_rot=key_toggles[p_op-ANIM_INSERT_POS][1];
			key_scale=key_toggles[p_op-ANIM_INSERT_POS][2];
			for(int i=ANIM_INSERT_POS;i<=ANIM_INSERT_POS_ROT_SCALE;i++) {
				int idx = animation_menu->get_popup()->get_item_index(i);
				animation_menu->get_popup()->set_item_checked(idx,i==p_op);
			}
		} break;*/
		case ANIM_COPY_POSE: {

			pose_clipboard.clear();

			Map<Node *, Object *> &selection = editor_selection->get_selection();

			for (Map<Node *, Object *>::Element *E = selection.front(); E; E = E->next()) {

				CanvasItem *canvas_item = Object::cast_to<CanvasItem>(E->key());
				if (!canvas_item || !canvas_item->is_visible_in_tree())
					continue;

				if (canvas_item->get_viewport() != EditorNode::get_singleton()->get_scene_root())
					continue;

				if (Object::cast_to<Node2D>(canvas_item)) {

					Node2D *n2d = Object::cast_to<Node2D>(canvas_item);
					PoseClipboard pc;
					pc.pos = n2d->get_position();
					pc.rot = n2d->get_rotation();
					pc.scale = n2d->get_scale();
					pc.id = n2d->get_instance_id();
					pose_clipboard.push_back(pc);
				}
			}

		} break;
		case ANIM_PASTE_POSE: {

			if (!pose_clipboard.size())
				break;

			undo_redo->create_action(TTR("Paste Pose"));
			for (List<PoseClipboard>::Element *E = pose_clipboard.front(); E; E = E->next()) {

				Node2D *n2d = Object::cast_to<Node2D>(ObjectDB::get_instance(E->get().id));
				if (!n2d)
					continue;
				undo_redo->add_do_method(n2d, "set_position", E->get().pos);
				undo_redo->add_do_method(n2d, "set_rotation", E->get().rot);
				undo_redo->add_do_method(n2d, "set_scale", E->get().scale);
				undo_redo->add_undo_method(n2d, "set_position", n2d->get_position());
				undo_redo->add_undo_method(n2d, "set_rotation", n2d->get_rotation());
				undo_redo->add_undo_method(n2d, "set_scale", n2d->get_scale());
			}
			undo_redo->commit_action();

		} break;
		case ANIM_CLEAR_POSE: {

			Map<Node *, Object *> &selection = editor_selection->get_selection();

			for (Map<Node *, Object *>::Element *E = selection.front(); E; E = E->next()) {

				CanvasItem *canvas_item = Object::cast_to<CanvasItem>(E->key());
				if (!canvas_item || !canvas_item->is_visible_in_tree())
					continue;

				if (canvas_item->get_viewport() != EditorNode::get_singleton()->get_scene_root())
					continue;

				if (Object::cast_to<Node2D>(canvas_item)) {
					Node2D *n2d = Object::cast_to<Node2D>(canvas_item);

					if (key_pos)
						n2d->set_position(Vector2());
					if (key_rot)
						n2d->set_rotation(0);
					if (key_scale)
						n2d->set_scale(Vector2(1, 1));
				} else if (Object::cast_to<Control>(canvas_item)) {

					Control *ctrl = Object::cast_to<Control>(canvas_item);

					if (key_pos)
						ctrl->set_position(Point2());
					/*
					if (key_scale)
						AnimationPlayerEditor::singleton->get_key_editor()->insert_node_value_key(ctrl,"rect/size",ctrl->get_size());
					*/
				}
			}

		} break;
		case VIEW_CENTER_TO_SELECTION:
		case VIEW_FRAME_TO_SELECTION: {

			_focus_selection(p_op);

		} break;
		case SKELETON_MAKE_BONES: {

			Map<Node *, Object *> &selection = editor_selection->get_selection();

			for (Map<Node *, Object *>::Element *E = selection.front(); E; E = E->next()) {

				Node2D *n2d = Object::cast_to<Node2D>(E->key());
				if (!n2d)
					continue;
				if (!n2d->is_visible_in_tree())
					continue;
				if (!n2d->get_parent_item())
					continue;

				n2d->set_meta("_edit_bone_", true);
				if (!skeleton_show_bones)
					skeleton_menu->get_popup()->activate_item(skeleton_menu->get_popup()->get_item_index(SKELETON_SHOW_BONES));
			}
			viewport->update();

		} break;
		case SKELETON_CLEAR_BONES: {

			Map<Node *, Object *> &selection = editor_selection->get_selection();

			for (Map<Node *, Object *>::Element *E = selection.front(); E; E = E->next()) {

				Node2D *n2d = Object::cast_to<Node2D>(E->key());
				if (!n2d)
					continue;
				if (!n2d->is_visible_in_tree())
					continue;

				n2d->set_meta("_edit_bone_", Variant());
				if (!skeleton_show_bones)
					skeleton_menu->get_popup()->activate_item(skeleton_menu->get_popup()->get_item_index(SKELETON_SHOW_BONES));
			}
			viewport->update();

		} break;
		case SKELETON_SET_IK_CHAIN: {

			List<Node *> &selection = editor_selection->get_selected_node_list();

			for (List<Node *>::Element *E = selection.front(); E; E = E->next()) {

				CanvasItem *canvas_item = Object::cast_to<CanvasItem>(E->get());
				if (!canvas_item || !canvas_item->is_visible_in_tree())
					continue;

				if (canvas_item->get_viewport() != EditorNode::get_singleton()->get_scene_root())
					continue;

				canvas_item->set_meta("_edit_ik_", true);
				if (!skeleton_show_bones)
					skeleton_menu->get_popup()->activate_item(skeleton_menu->get_popup()->get_item_index(SKELETON_SHOW_BONES));
			}

			viewport->update();

		} break;
		case SKELETON_CLEAR_IK_CHAIN: {

			Map<Node *, Object *> &selection = editor_selection->get_selection();

			for (Map<Node *, Object *>::Element *E = selection.front(); E; E = E->next()) {

				CanvasItem *n2d = Object::cast_to<CanvasItem>(E->key());
				if (!n2d)
					continue;
				if (!n2d->is_visible_in_tree())
					continue;

				n2d->set_meta("_edit_ik_", Variant());
				if (!skeleton_show_bones)
					skeleton_menu->get_popup()->activate_item(skeleton_menu->get_popup()->get_item_index(SKELETON_SHOW_BONES));
			}
			viewport->update();

		} break;
	}
}

void CanvasItemEditor::_focus_selection(int p_op) {
	Vector2 center(0.f, 0.f);
	Rect2 rect;
	int count = 0;

	Map<Node *, Object *> &selection = editor_selection->get_selection();
	for (Map<Node *, Object *>::Element *E = selection.front(); E; E = E->next()) {
		CanvasItem *canvas_item = Object::cast_to<CanvasItem>(E->key());
		if (!canvas_item) continue;
		if (canvas_item->get_viewport() != EditorNode::get_singleton()->get_scene_root())
			continue;

		// counting invisible items, for now
		//if (!canvas_item->is_visible_in_tree()) continue;
		++count;

		Rect2 item_rect = canvas_item->_edit_get_rect();

		Vector2 pos = canvas_item->get_global_transform().get_origin();
		Vector2 scale = canvas_item->get_global_transform().get_scale();
		real_t angle = canvas_item->get_global_transform().get_rotation();

		Transform2D t(angle, Vector2(0.f, 0.f));
		item_rect = t.xform(item_rect);
		Rect2 canvas_item_rect(pos + scale * item_rect.position, scale * item_rect.size);
		if (count == 1) {
			rect = canvas_item_rect;
		} else {
			rect = rect.merge(canvas_item_rect);
		}
	};
	if (count == 0) return;

	if (p_op == VIEW_CENTER_TO_SELECTION) {

		center = rect.position + rect.size / 2;
		Vector2 offset = viewport->get_size() / 2 - editor->get_scene_root()->get_global_canvas_transform().xform(center);
		h_scroll->set_value(h_scroll->get_value() - offset.x / zoom);
		v_scroll->set_value(v_scroll->get_value() - offset.y / zoom);

	} else { // VIEW_FRAME_TO_SELECTION

		if (rect.size.x > CMP_EPSILON && rect.size.y > CMP_EPSILON) {
			float scale_x = viewport->get_size().x / rect.size.x;
			float scale_y = viewport->get_size().y / rect.size.y;
			zoom = scale_x < scale_y ? scale_x : scale_y;
			zoom *= 0.90;
			_update_scroll(0);
			call_deferred("_popup_callback", VIEW_CENTER_TO_SELECTION);
		}
	}
}

void CanvasItemEditor::_bind_methods() {

	ClassDB::bind_method("_zoom_minus", &CanvasItemEditor::_zoom_minus);
	ClassDB::bind_method("_zoom_reset", &CanvasItemEditor::_zoom_reset);
	ClassDB::bind_method("_zoom_plus", &CanvasItemEditor::_zoom_plus);
	ClassDB::bind_method("_toggle_snap", &CanvasItemEditor::_toggle_snap);
	ClassDB::bind_method("_update_scroll", &CanvasItemEditor::_update_scroll);
	ClassDB::bind_method("_popup_callback", &CanvasItemEditor::_popup_callback);
	ClassDB::bind_method("_get_editor_data", &CanvasItemEditor::_get_editor_data);
	ClassDB::bind_method("_tool_select", &CanvasItemEditor::_tool_select);
	ClassDB::bind_method("_keying_changed", &CanvasItemEditor::_keying_changed);
	ClassDB::bind_method("_unhandled_key_input", &CanvasItemEditor::_unhandled_key_input);
	ClassDB::bind_method("_draw_viewport", &CanvasItemEditor::_draw_viewport);
	ClassDB::bind_method("_draw_viewport_base", &CanvasItemEditor::_draw_viewport_base);
	ClassDB::bind_method("_gui_input_viewport", &CanvasItemEditor::_gui_input_viewport);
	ClassDB::bind_method("_gui_input_viewport_base", &CanvasItemEditor::_gui_input_viewport_base);
	ClassDB::bind_method("_snap_changed", &CanvasItemEditor::_snap_changed);
	ClassDB::bind_method(D_METHOD("_selection_result_pressed"), &CanvasItemEditor::_selection_result_pressed);
	ClassDB::bind_method(D_METHOD("_selection_menu_hide"), &CanvasItemEditor::_selection_menu_hide);

	ADD_SIGNAL(MethodInfo("item_lock_status_changed"));
	ADD_SIGNAL(MethodInfo("item_group_status_changed"));
}

void CanvasItemEditor::add_control_to_menu_panel(Control *p_control) {

	hb->add_child(p_control);
}

HSplitContainer *CanvasItemEditor::get_palette_split() {

	return palette_split;
}

VSplitContainer *CanvasItemEditor::get_bottom_split() {

	return bottom_split;
}

void CanvasItemEditor::focus_selection() {
	_focus_selection(VIEW_CENTER_TO_SELECTION);
}

CanvasItemEditor::CanvasItemEditor(EditorNode *p_editor) {

	tool = TOOL_SELECT;
	undo_redo = p_editor->get_undo_redo();
	editor = p_editor;
	editor_selection = p_editor->get_editor_selection();
	editor_selection->add_editor_plugin(this);
	editor_selection->connect("selection_changed", this, "update");

	hb = memnew(HBoxContainer);
	add_child(hb);
	hb->set_anchors_and_margins_preset(Control::PRESET_WIDE);

	bottom_split = memnew(VSplitContainer);
	add_child(bottom_split);
	bottom_split->set_v_size_flags(SIZE_EXPAND_FILL);

	palette_split = memnew(HSplitContainer);
	bottom_split->add_child(palette_split);
	palette_split->set_v_size_flags(SIZE_EXPAND_FILL);

	viewport_base = memnew(Control);
	palette_split->add_child(viewport_base);
	viewport_base->set_clip_contents(true);
	viewport_base->connect("draw", this, "_draw_viewport_base");
	viewport_base->connect("gui_input", this, "_gui_input_viewport_base");
	viewport_base->set_focus_mode(FOCUS_ALL);
	viewport_base->set_v_size_flags(SIZE_EXPAND_FILL);
	viewport_base->set_h_size_flags(SIZE_EXPAND_FILL);

	viewport_scrollable = memnew(Control);
	viewport_base->add_child(viewport_scrollable);
	viewport_scrollable->set_mouse_filter(MOUSE_FILTER_PASS);
	viewport_scrollable->set_draw_behind_parent(true);
	viewport_scrollable->set_anchors_and_margins_preset(Control::PRESET_WIDE);
	viewport_scrollable->set_begin(Point2(RULER_WIDTH, RULER_WIDTH));

	ViewportContainer *scene_tree = memnew(ViewportContainer);
	viewport_scrollable->add_child(scene_tree);
	scene_tree->set_stretch(true);
	scene_tree->set_anchors_and_margins_preset(Control::PRESET_WIDE);
	scene_tree->add_child(p_editor->get_scene_root());

	viewport = memnew(CanvasItemEditorViewport(p_editor, this));
	viewport_scrollable->add_child(viewport);
	viewport->set_mouse_filter(MOUSE_FILTER_PASS);
	viewport->set_anchors_and_margins_preset(Control::PRESET_WIDE);
	viewport->set_clip_contents(true);
	viewport->connect("draw", this, "_draw_viewport");
	viewport->connect("gui_input", this, "_gui_input_viewport");

	h_scroll = memnew(HScrollBar);
	viewport->add_child(h_scroll);
	h_scroll->connect("value_changed", this, "_update_scroll", Vector<Variant>(), Object::CONNECT_DEFERRED);
	h_scroll->hide();

	v_scroll = memnew(VScrollBar);
	viewport->add_child(v_scroll);
	v_scroll->connect("value_changed", this, "_update_scroll", Vector<Variant>(), Object::CONNECT_DEFERRED);
	v_scroll->hide();

	HBoxContainer *zoom_hb = memnew(HBoxContainer);
	viewport->add_child(zoom_hb);
	zoom_hb->set_begin(Point2(5, 5));

	zoom_minus = memnew(ToolButton);
	zoom_hb->add_child(zoom_minus);
	zoom_minus->connect("pressed", this, "_zoom_minus");
	zoom_minus->set_focus_mode(FOCUS_NONE);

	zoom_reset = memnew(ToolButton);
	zoom_hb->add_child(zoom_reset);
	zoom_reset->connect("pressed", this, "_zoom_reset");
	zoom_reset->set_focus_mode(FOCUS_NONE);

	zoom_plus = memnew(ToolButton);
	zoom_hb->add_child(zoom_plus);
	zoom_plus->connect("pressed", this, "_zoom_plus");
	zoom_plus->set_focus_mode(FOCUS_NONE);

	updating_scroll = false;
	handle_len = 10;
	first_update = true;

	select_button = memnew(ToolButton);
	hb->add_child(select_button);
	select_button->set_toggle_mode(true);
	select_button->connect("pressed", this, "_tool_select", make_binds(TOOL_SELECT));
	select_button->set_pressed(true);
	select_button->set_shortcut(ED_SHORTCUT("canvas_item_editor/select_mode", TTR("Select Mode"), KEY_Q));
	select_button->set_tooltip(keycode_get_string(KEY_MASK_CMD) + TTR("Drag: Rotate") + "\n" + TTR("Alt+Drag: Move") + "\n" + TTR("Press 'v' to Change Pivot, 'Shift+v' to Drag Pivot (while moving).") + "\n" + TTR("Alt+RMB: Depth list selection"));

	move_button = memnew(ToolButton);
	hb->add_child(move_button);
	move_button->set_toggle_mode(true);
	move_button->connect("pressed", this, "_tool_select", make_binds(TOOL_MOVE));
	move_button->set_shortcut(ED_SHORTCUT("canvas_item_editor/move_mode", TTR("Move Mode"), KEY_W));
	move_button->set_tooltip(TTR("Move Mode"));

	rotate_button = memnew(ToolButton);
	hb->add_child(rotate_button);
	rotate_button->set_toggle_mode(true);
	rotate_button->connect("pressed", this, "_tool_select", make_binds(TOOL_ROTATE));
	rotate_button->set_shortcut(ED_SHORTCUT("canvas_item_editor/rotate_mode", TTR("Rotate Mode"), KEY_E));
	rotate_button->set_tooltip(TTR("Rotate Mode"));

	hb->add_child(memnew(VSeparator));

	list_select_button = memnew(ToolButton);
	hb->add_child(list_select_button);
	list_select_button->set_toggle_mode(true);
	list_select_button->connect("pressed", this, "_tool_select", make_binds(TOOL_LIST_SELECT));
	list_select_button->set_tooltip(TTR("Show a list of all objects at the position clicked\n(same as Alt+RMB in select mode)."));

	pivot_button = memnew(ToolButton);
	hb->add_child(pivot_button);
	pivot_button->set_toggle_mode(true);
	pivot_button->connect("pressed", this, "_tool_select", make_binds(TOOL_EDIT_PIVOT));
	pivot_button->set_tooltip(TTR("Click to change object's rotation pivot."));

	pan_button = memnew(ToolButton);
	hb->add_child(pan_button);
	pan_button->set_toggle_mode(true);
	pan_button->connect("pressed", this, "_tool_select", make_binds(TOOL_PAN));
	pan_button->set_tooltip(TTR("Pan Mode"));

	hb->add_child(memnew(VSeparator));

	snap_button = memnew(ToolButton);
	hb->add_child(snap_button);
	snap_button->set_toggle_mode(true);
	snap_button->connect("toggled", this, "_toggle_snap");
	snap_button->set_tooltip(TTR("Toggles snapping"));
	snap_button->set_shortcut(ED_SHORTCUT("canvas_item_editor/use_snap", TTR("Use Snap"), KEY_S));

	snap_config_menu = memnew(MenuButton);
	hb->add_child(snap_config_menu);
	snap_config_menu->set_h_size_flags(SIZE_SHRINK_END);
	snap_config_menu->set_tooltip(TTR("Snapping options"));

	PopupMenu *p = snap_config_menu->get_popup();
	p->connect("id_pressed", this, "_popup_callback");
	p->set_hide_on_checkable_item_selection(false);
	p->add_check_shortcut(ED_SHORTCUT("canvas_item_editor/snap_grid", TTR("Snap to grid")), SNAP_USE_GRID);
	p->add_check_shortcut(ED_SHORTCUT("canvas_item_editor/use_rotation_snap", TTR("Use Rotation Snap")), SNAP_USE_ROTATION);
	p->add_shortcut(ED_SHORTCUT("canvas_item_editor/configure_snap", TTR("Configure Snap...")), SNAP_CONFIGURE);
	p->add_check_shortcut(ED_SHORTCUT("canvas_item_editor/snap_relative", TTR("Snap Relative")), SNAP_RELATIVE);
	p->add_check_shortcut(ED_SHORTCUT("canvas_item_editor/use_pixel_snap", TTR("Use Pixel Snap")), SNAP_USE_PIXEL);
	p->add_submenu_item(TTR("Smart snapping"), "SmartSnapping");

	smartsnap_config_popup = memnew(PopupMenu);
	p->add_child(smartsnap_config_popup);
	smartsnap_config_popup->set_name("SmartSnapping");
	smartsnap_config_popup->connect("id_pressed", this, "_popup_callback");
	smartsnap_config_popup->set_hide_on_checkable_item_selection(false);
	smartsnap_config_popup->add_check_shortcut(ED_SHORTCUT("canvas_item_editor/snap_node_parent", TTR("Snap to parent")), SNAP_USE_NODE_PARENT);
	smartsnap_config_popup->add_check_shortcut(ED_SHORTCUT("canvas_item_editor/snap_node_anchors", TTR("Snap to node anchor")), SNAP_USE_NODE_ANCHORS);
	smartsnap_config_popup->add_check_shortcut(ED_SHORTCUT("canvas_item_editor/snap_node_sides", TTR("Snap to node sides")), SNAP_USE_NODE_SIDES);
	smartsnap_config_popup->add_check_shortcut(ED_SHORTCUT("canvas_item_editor/snap_other_nodes", TTR("Snap to other nodes")), SNAP_USE_OTHER_NODES);
	smartsnap_config_popup->add_check_shortcut(ED_SHORTCUT("canvas_item_editor/snap_guides", TTR("Snap to guides")), SNAP_USE_GUIDES);

	hb->add_child(memnew(VSeparator));

	lock_button = memnew(ToolButton);
	hb->add_child(lock_button);

	lock_button->connect("pressed", this, "_popup_callback", varray(LOCK_SELECTED));
	lock_button->set_tooltip(TTR("Lock the selected object in place (can't be moved)."));

	unlock_button = memnew(ToolButton);
	hb->add_child(unlock_button);
	unlock_button->connect("pressed", this, "_popup_callback", varray(UNLOCK_SELECTED));
	unlock_button->set_tooltip(TTR("Unlock the selected object (can be moved)."));

	group_button = memnew(ToolButton);
	hb->add_child(group_button);
	group_button->connect("pressed", this, "_popup_callback", varray(GROUP_SELECTED));
	group_button->set_tooltip(TTR("Makes sure the object's children are not selectable."));

	ungroup_button = memnew(ToolButton);
	hb->add_child(ungroup_button);
	ungroup_button->connect("pressed", this, "_popup_callback", varray(UNGROUP_SELECTED));
	ungroup_button->set_tooltip(TTR("Restores the object's children's ability to be selected."));

	hb->add_child(memnew(VSeparator));

	skeleton_menu = memnew(MenuButton);
	hb->add_child(skeleton_menu);

	p = skeleton_menu->get_popup();
	p->set_hide_on_checkable_item_selection(false);
	p->add_shortcut(ED_SHORTCUT("canvas_item_editor/skeleton_make_bones", TTR("Make Bones"), KEY_MASK_CMD | KEY_MASK_SHIFT | KEY_B), SKELETON_MAKE_BONES);
	p->add_shortcut(ED_SHORTCUT("canvas_item_editor/skeleton_clear_bones", TTR("Clear Bones")), SKELETON_CLEAR_BONES);
	p->add_separator();
	p->add_check_shortcut(ED_SHORTCUT("canvas_item_editor/skeleton_show_bones", TTR("Show Bones")), SKELETON_SHOW_BONES);
	p->add_separator();
	p->add_shortcut(ED_SHORTCUT("canvas_item_editor/skeleton_set_ik_chain", TTR("Make IK Chain")), SKELETON_SET_IK_CHAIN);
	p->add_shortcut(ED_SHORTCUT("canvas_item_editor/skeleton_clear_ik_chain", TTR("Clear IK Chain")), SKELETON_CLEAR_IK_CHAIN);
	p->connect("id_pressed", this, "_popup_callback");

	hb->add_child(memnew(VSeparator));

	view_menu = memnew(MenuButton);
	view_menu->set_text(TTR("View"));
	hb->add_child(view_menu);
	view_menu->get_popup()->connect("id_pressed", this, "_popup_callback");

	p = view_menu->get_popup();
	p->add_check_shortcut(ED_SHORTCUT("canvas_item_editor/show_grid", TTR("Show Grid"), KEY_G), SHOW_GRID);
	p->add_check_shortcut(ED_SHORTCUT("canvas_item_editor/show_helpers", TTR("Show helpers"), KEY_H), SHOW_HELPERS);
	p->add_check_shortcut(ED_SHORTCUT("canvas_item_editor/show_rulers", TTR("Show rulers"), KEY_R), SHOW_RULERS);
	p->add_check_shortcut(ED_SHORTCUT("canvas_item_editor/show_guides", TTR("Show guides"), KEY_Y), SHOW_GUIDES);
	p->add_separator();
	p->add_shortcut(ED_SHORTCUT("canvas_item_editor/center_selection", TTR("Center Selection"), KEY_F), VIEW_CENTER_TO_SELECTION);
	p->add_shortcut(ED_SHORTCUT("canvas_item_editor/frame_selection", TTR("Frame Selection"), KEY_MASK_SHIFT | KEY_F), VIEW_FRAME_TO_SELECTION);

	presets_menu = memnew(MenuButton);
	presets_menu->set_text(TTR("Layout"));
	hb->add_child(presets_menu);
	presets_menu->hide();

	p = presets_menu->get_popup();
	p->connect("id_pressed", this, "_popup_callback");

	anchors_popup = memnew(PopupMenu);
	p->add_child(anchors_popup);
	anchors_popup->set_name("Anchors");
	anchors_popup->connect("id_pressed", this, "_popup_callback");

	animation_hb = memnew(HBoxContainer);
	hb->add_child(animation_hb);
	animation_hb->add_child(memnew(VSeparator));
	animation_hb->hide();

	key_loc_button = memnew(Button);
	key_loc_button->set_toggle_mode(true);
	key_loc_button->set_flat(true);
	key_loc_button->set_pressed(true);
	key_loc_button->set_focus_mode(FOCUS_NONE);
	key_loc_button->connect("pressed", this, "_popup_callback", varray(ANIM_INSERT_POS));
	animation_hb->add_child(key_loc_button);
	key_rot_button = memnew(Button);
	key_rot_button->set_toggle_mode(true);
	key_rot_button->set_flat(true);
	key_rot_button->set_pressed(true);
	key_rot_button->set_focus_mode(FOCUS_NONE);
	key_rot_button->connect("pressed", this, "_popup_callback", varray(ANIM_INSERT_ROT));
	animation_hb->add_child(key_rot_button);
	key_scale_button = memnew(Button);
	key_scale_button->set_toggle_mode(true);
	key_scale_button->set_flat(true);
	key_scale_button->set_focus_mode(FOCUS_NONE);
	key_scale_button->connect("pressed", this, "_popup_callback", varray(ANIM_INSERT_SCALE));
	animation_hb->add_child(key_scale_button);
	key_insert_button = memnew(Button);
	key_insert_button->set_flat(true);
	key_insert_button->set_focus_mode(FOCUS_NONE);
	key_insert_button->connect("pressed", this, "_popup_callback", varray(ANIM_INSERT_KEY));
	key_insert_button->set_tooltip(TTR("Insert Keys"));
	key_insert_button->set_shortcut(ED_SHORTCUT("canvas_item_editor/anim_insert_key", TTR("Insert Key"), KEY_INSERT));

	animation_hb->add_child(key_insert_button);

	animation_menu = memnew(MenuButton);
	animation_menu->set_text(TTR("Animation"));
	animation_hb->add_child(animation_menu);
	animation_menu->get_popup()->connect("id_pressed", this, "_popup_callback");

	p = animation_menu->get_popup();

	p->add_shortcut(ED_GET_SHORTCUT("canvas_item_editor/anim_insert_key"), ANIM_INSERT_KEY);
	p->add_shortcut(ED_SHORTCUT("canvas_item_editor/anim_insert_key_existing_tracks", TTR("Insert Key (Existing Tracks)"), KEY_MASK_CMD + KEY_INSERT), ANIM_INSERT_KEY_EXISTING);
	p->add_separator();
	p->add_shortcut(ED_SHORTCUT("canvas_item_editor/anim_copy_pose", TTR("Copy Pose")), ANIM_COPY_POSE);
	p->add_shortcut(ED_SHORTCUT("canvas_item_editor/anim_paste_pose", TTR("Paste Pose")), ANIM_PASTE_POSE);
	p->add_shortcut(ED_SHORTCUT("canvas_item_editor/anim_clear_pose", TTR("Clear Pose"), KEY_MASK_SHIFT | KEY_K), ANIM_CLEAR_POSE);

	snap_dialog = memnew(SnapDialog);
	snap_dialog->connect("confirmed", this, "_snap_changed");
	add_child(snap_dialog);

	select_sb = Ref<StyleBoxTexture>(memnew(StyleBoxTexture));

	selection_menu = memnew(PopupMenu);
	add_child(selection_menu);
	selection_menu->set_custom_minimum_size(Vector2(100, 0));
	selection_menu->connect("id_pressed", this, "_selection_result_pressed");
	selection_menu->connect("popup_hide", this, "_selection_menu_hide");

	drag_pivot_shortcut = ED_SHORTCUT("canvas_item_editor/drag_pivot", TTR("Drag pivot from mouse position"), KEY_MASK_SHIFT | KEY_V);
	set_pivot_shortcut = ED_SHORTCUT("canvas_item_editor/set_pivot", TTR("Set pivot at mouse position"), KEY_V);

	multiply_grid_step_shortcut = ED_SHORTCUT("canvas_item_editor/multiply_grid_step", TTR("Multiply grid step by 2"), KEY_KP_MULTIPLY);
	divide_grid_step_shortcut = ED_SHORTCUT("canvas_item_editor/divide_grid_step", TTR("Divide grid step by 2"), KEY_KP_DIVIDE);

	key_pos = true;
	key_rot = true;
	key_scale = false;

	edited_guide_pos = Point2();
	edited_guide_index = -1;

	show_grid = false;
	show_helpers = false;
	show_rulers = true;
	show_guides = true;
	zoom = 1;
	grid_offset = Point2();
	grid_step = Point2(10, 10);
	grid_step_multiplier = 0;
	snap_rotation_offset = 0;
	snap_rotation_step = 15 / (180 / Math_PI);
	snap_active = false;
	snap_node_parent = true;
	snap_node_anchors = true;
	snap_node_sides = true;
	snap_other_nodes = true;
	snap_grid = true;
	snap_guides = true;
	snap_rotation = false;
	snap_pixel = false;
	skeleton_show_bones = true;
	skeleton_menu->get_popup()->set_item_checked(skeleton_menu->get_popup()->get_item_index(SKELETON_SHOW_BONES), true);
	box_selecting = false;
	//zoom=0.5;
	singleton = this;

	set_process_unhandled_key_input(true);
	can_move_pivot = false;
	drag = DRAG_NONE;
	bone_last_frame = 0;
	additive_selection = false;

	// Update the menus checkboxes
	call_deferred("set_state", get_state());
}

CanvasItemEditor *CanvasItemEditor::singleton = NULL;

void CanvasItemEditorPlugin::edit(Object *p_object) {

	canvas_item_editor->set_undo_redo(&get_undo_redo());
	canvas_item_editor->edit(Object::cast_to<CanvasItem>(p_object));
}

bool CanvasItemEditorPlugin::handles(Object *p_object) const {

	return p_object->is_class("CanvasItem");
}

void CanvasItemEditorPlugin::make_visible(bool p_visible) {

	if (p_visible) {
		canvas_item_editor->show();
		canvas_item_editor->set_physics_process(true);
		VisualServer::get_singleton()->viewport_set_hide_canvas(editor->get_scene_root()->get_viewport_rid(), false);
		canvas_item_editor->viewport_base->grab_focus();

	} else {

		canvas_item_editor->hide();
		canvas_item_editor->set_physics_process(false);
		VisualServer::get_singleton()->viewport_set_hide_canvas(editor->get_scene_root()->get_viewport_rid(), true);
	}
}

Dictionary CanvasItemEditorPlugin::get_state() const {

	return canvas_item_editor->get_state();
}
void CanvasItemEditorPlugin::set_state(const Dictionary &p_state) {

	canvas_item_editor->set_state(p_state);
}

CanvasItemEditorPlugin::CanvasItemEditorPlugin(EditorNode *p_node) {

	editor = p_node;
	canvas_item_editor = memnew(CanvasItemEditor(editor));
	canvas_item_editor->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	editor->get_viewport()->add_child(canvas_item_editor);
	canvas_item_editor->set_anchors_and_margins_preset(Control::PRESET_WIDE);
	canvas_item_editor->hide();
}

CanvasItemEditorPlugin::~CanvasItemEditorPlugin() {
}

void CanvasItemEditorViewport::_on_mouse_exit() {
	if (!selector->is_visible()) {
		_remove_preview();
	}
}

void CanvasItemEditorViewport::_on_select_type(Object *selected) {
	CheckBox *check = Object::cast_to<CheckBox>(selected);
	String type = check->get_text();
	selector->set_title(vformat(TTR("Add %s"), type));
	label->set_text(vformat(TTR("Adding %s..."), type));
}

void CanvasItemEditorViewport::_on_change_type_confirmed() {
	if (!button_group->get_pressed_button())
		return;

	CheckBox *check = Object::cast_to<CheckBox>(button_group->get_pressed_button());
	default_type = check->get_text();
	_perform_drop_data();
	selector->hide();
}

void CanvasItemEditorViewport::_on_change_type_closed() {

	_remove_preview();
}

void CanvasItemEditorViewport::_create_preview(const Vector<String> &files) const {
	label->set_position(get_global_position() + Point2(14, 14) * EDSCALE);
	label_desc->set_position(label->get_position() + Point2(0, label->get_size().height));
	for (int i = 0; i < files.size(); i++) {
		String path = files[i];
		RES res = ResourceLoader::load(path);
		Ref<Texture> texture = Ref<Texture>(Object::cast_to<Texture>(*res));
		Ref<PackedScene> scene = Ref<PackedScene>(Object::cast_to<PackedScene>(*res));
		if (texture != NULL || scene != NULL) {
			if (texture != NULL) {
				Sprite *sprite = memnew(Sprite);
				sprite->set_texture(texture);
				sprite->set_modulate(Color(1, 1, 1, 0.7f));
				preview_node->add_child(sprite);
				label->show();
				label_desc->show();
			} else {
				if (scene.is_valid()) {
					Node *instance = scene->instance();
					if (instance) {
						preview_node->add_child(instance);
					}
				}
			}
			editor->get_scene_root()->add_child(preview_node);
		}
	}
}

void CanvasItemEditorViewport::_remove_preview() {
	if (preview_node->get_parent()) {
		for (int i = preview_node->get_child_count() - 1; i >= 0; i--) {
			Node *node = preview_node->get_child(i);
			node->queue_delete();
			preview_node->remove_child(node);
		}
		editor->get_scene_root()->remove_child(preview_node);

		label->hide();
		label_desc->hide();
	}
}

bool CanvasItemEditorViewport::_cyclical_dependency_exists(const String &p_target_scene_path, Node *p_desired_node) {
	if (p_desired_node->get_filename() == p_target_scene_path) {
		return true;
	}

	int childCount = p_desired_node->get_child_count();
	for (int i = 0; i < childCount; i++) {
		Node *child = p_desired_node->get_child(i);
		if (_cyclical_dependency_exists(p_target_scene_path, child)) {
			return true;
		}
	}
	return false;
}

void CanvasItemEditorViewport::_create_nodes(Node *parent, Node *child, String &path, const Point2 &p_point) {
	child->set_name(path.get_file().get_basename());
	Ref<Texture> texture = Ref<Texture>(Object::cast_to<Texture>(ResourceCache::get(path)));
	Size2 texture_size = texture->get_size();

	if (parent) {
		editor_data->get_undo_redo().add_do_method(parent, "add_child", child);
		editor_data->get_undo_redo().add_do_method(child, "set_owner", editor->get_edited_scene());
		editor_data->get_undo_redo().add_do_reference(child);
		editor_data->get_undo_redo().add_undo_method(parent, "remove_child", child);
	} else { // if we haven't parent, lets try to make a child as a parent.
		editor_data->get_undo_redo().add_do_method(editor, "set_edited_scene", child);
		editor_data->get_undo_redo().add_do_method(child, "set_owner", editor->get_edited_scene());
		editor_data->get_undo_redo().add_do_reference(child);
		editor_data->get_undo_redo().add_undo_method(editor, "set_edited_scene", (Object *)NULL);
	}

	if (parent) {
		String new_name = parent->validate_child_name(child);
		ScriptEditorDebugger *sed = ScriptEditor::get_singleton()->get_debugger();
		editor_data->get_undo_redo().add_do_method(sed, "live_debug_create_node", editor->get_edited_scene()->get_path_to(parent), child->get_class(), new_name);
		editor_data->get_undo_redo().add_undo_method(sed, "live_debug_remove_node", NodePath(String(editor->get_edited_scene()->get_path_to(parent)) + "/" + new_name));
	}

	// handle with different property for texture
	String property = "texture";
	List<PropertyInfo> props;
	child->get_property_list(&props);
	for (const List<PropertyInfo>::Element *E = props.front(); E; E = E->next()) {
		if (E->get().name == "config/texture") { // Particles2D
			property = "config/texture";
			break;
		} else if (E->get().name == "texture/texture") { // Polygon2D
			property = "texture/texture";
			break;
		} else if (E->get().name == "normal") { // TouchScreenButton
			property = "normal";
			break;
		}
	}
	editor_data->get_undo_redo().add_do_property(child, property, texture);

	// make visible for certain node type
	if (default_type == "NinePatchRect") {
		editor_data->get_undo_redo().add_do_property(child, "rect/size", texture_size);
	} else if (default_type == "Polygon2D") {
		PoolVector<Vector2> list;
		list.push_back(Vector2(0, 0));
		list.push_back(Vector2(texture_size.width, 0));
		list.push_back(Vector2(texture_size.width, texture_size.height));
		list.push_back(Vector2(0, texture_size.height));
		editor_data->get_undo_redo().add_do_property(child, "polygon", list);
	}

	// locate at preview position
	Point2 pos = Point2(0, 0);
	if (parent && parent->has_method("get_global_position")) {
		pos = parent->call("get_global_position");
	}
	Transform2D trans = canvas->get_canvas_transform();
	Point2 target_position = (p_point - trans.get_origin()) / trans.get_scale().x - pos;
	if (default_type == "Polygon2D" || default_type == "TouchScreenButton" || default_type == "TextureRect" || default_type == "NinePatchRect") {
		target_position -= texture_size / 2;
	}
	// there's nothing to be used as source position so snapping will work as absolute if enabled
	target_position = canvas->snap_point(target_position);
	editor_data->get_undo_redo().add_do_method(child, "set_position", target_position);
}

bool CanvasItemEditorViewport::_create_instance(Node *parent, String &path, const Point2 &p_point) {
	Ref<PackedScene> sdata = ResourceLoader::load(path);
	if (!sdata.is_valid()) { // invalid scene
		return false;
	}

	Node *instanced_scene = sdata->instance(PackedScene::GEN_EDIT_STATE_INSTANCE);
	if (!instanced_scene) { // error on instancing
		return false;
	}

	if (editor->get_edited_scene()->get_filename() != "") { // cyclical instancing
		if (_cyclical_dependency_exists(editor->get_edited_scene()->get_filename(), instanced_scene)) {
			memdelete(instanced_scene);
			return false;
		}
	}

	instanced_scene->set_filename(ProjectSettings::get_singleton()->localize_path(path));

	editor_data->get_undo_redo().add_do_method(parent, "add_child", instanced_scene);
	editor_data->get_undo_redo().add_do_method(instanced_scene, "set_owner", editor->get_edited_scene());
	editor_data->get_undo_redo().add_do_reference(instanced_scene);
	editor_data->get_undo_redo().add_undo_method(parent, "remove_child", instanced_scene);

	String new_name = parent->validate_child_name(instanced_scene);
	ScriptEditorDebugger *sed = ScriptEditor::get_singleton()->get_debugger();
	editor_data->get_undo_redo().add_do_method(sed, "live_debug_instance_node", editor->get_edited_scene()->get_path_to(parent), path, new_name);
	editor_data->get_undo_redo().add_undo_method(sed, "live_debug_remove_node", NodePath(String(editor->get_edited_scene()->get_path_to(parent)) + "/" + new_name));

	CanvasItem *parent_ci = Object::cast_to<CanvasItem>(parent);
	if (parent_ci) {
		Vector2 target_pos = canvas->get_canvas_transform().affine_inverse().xform(p_point);
		target_pos = canvas->snap_point(target_pos);
		target_pos = parent_ci->get_global_transform_with_canvas().affine_inverse().xform(target_pos);
		editor_data->get_undo_redo().add_do_method(instanced_scene, "set_position", target_pos);
	}

	return true;
}

void CanvasItemEditorViewport::_perform_drop_data() {
	_remove_preview();

	Vector<String> error_files;

	editor_data->get_undo_redo().create_action(TTR("Create Node"));

	for (int i = 0; i < selected_files.size(); i++) {
		String path = selected_files[i];
		RES res = ResourceLoader::load(path);
		if (res.is_null()) {
			continue;
		}
		Ref<Texture> texture = Ref<Texture>(Object::cast_to<Texture>(*res));
		Ref<PackedScene> scene = Ref<PackedScene>(Object::cast_to<PackedScene>(*res));
		if (texture != NULL) {
			Node *child;
			if (default_type == "Light2D")
				child = memnew(Light2D);
			else if (default_type == "Particles2D")
				child = memnew(Particles2D);
			else if (default_type == "Polygon2D")
				child = memnew(Polygon2D);
			else if (default_type == "TouchScreenButton")
				child = memnew(TouchScreenButton);
			else if (default_type == "TextureRect")
				child = memnew(TextureRect);
			else if (default_type == "NinePatchRect")
				child = memnew(NinePatchRect);
			else
				child = memnew(Sprite); // default

			_create_nodes(target_node, child, path, drop_pos);
		} else if (scene != NULL) {
			bool success = _create_instance(target_node, path, drop_pos);
			if (!success) {
				error_files.push_back(path);
			}
		}
	}

	editor_data->get_undo_redo().commit_action();

	if (error_files.size() > 0) {
		String files_str;
		for (int i = 0; i < error_files.size(); i++) {
			files_str += error_files[i].get_file().get_basename() + ",";
		}
		files_str = files_str.substr(0, files_str.length() - 1);
		accept->get_ok()->set_text(TTR("Ugh"));
		accept->set_text(vformat(TTR("Error instancing scene from %s"), files_str.c_str()));
		accept->popup_centered_minsize();
	}
}

bool CanvasItemEditorViewport::can_drop_data(const Point2 &p_point, const Variant &p_data) const {
	Dictionary d = p_data;
	if (d.has("type")) {
		if (String(d["type"]) == "files") {
			Vector<String> files = d["files"];
			bool can_instance = false;
			for (int i = 0; i < files.size(); i++) { // check if dragged files contain resource or scene can be created at least one
				RES res = ResourceLoader::load(files[i]);
				if (res.is_null()) {
					continue;
				}
				String type = res->get_class();
				if (type == "PackedScene") {
					Ref<PackedScene> sdata = ResourceLoader::load(files[i]);
					Node *instanced_scene = sdata->instance(PackedScene::GEN_EDIT_STATE_INSTANCE);
					if (!instanced_scene) {
						continue;
					}
					memdelete(instanced_scene);
				} else if (type == "Texture" ||
						   type == "ImageTexture" ||
						   type == "ViewportTexture" ||
						   type == "CurveTexture" ||
						   type == "GradientTexture" ||
						   type == "StreamTexture" ||
						   type == "AtlasTexture" ||
						   type == "LargeTexture") {
					Ref<Texture> texture = ResourceLoader::load(files[i]);
					if (texture.is_valid() == false) {
						continue;
					}
				} else {
					continue;
				}
				can_instance = true;
				break;
			}
			if (can_instance) {
				if (!preview_node->get_parent()) { // create preview only once
					_create_preview(files);
				}
				Transform2D trans = canvas->get_canvas_transform();
				preview_node->set_position((p_point - trans.get_origin()) / trans.get_scale().x);
				label->set_text(vformat(TTR("Adding %s..."), default_type));
			}
			return can_instance;
		}
	}
	label->hide();
	return false;
}

void CanvasItemEditorViewport::_show_resource_type_selector() {
	List<BaseButton *> btn_list;
	button_group->get_buttons(&btn_list);

	for (int i = 0; i < btn_list.size(); i++) {
		CheckBox *check = Object::cast_to<CheckBox>(btn_list[i]);
		check->set_pressed(check->get_text() == default_type);
	}
	selector->set_title(vformat(TTR("Add %s"), default_type));
	selector->popup_centered_minsize();
}

void CanvasItemEditorViewport::drop_data(const Point2 &p_point, const Variant &p_data) {
	bool is_shift = Input::get_singleton()->is_key_pressed(KEY_SHIFT);
	bool is_alt = Input::get_singleton()->is_key_pressed(KEY_ALT);

	selected_files.clear();
	Dictionary d = p_data;
	if (d.has("type") && String(d["type"]) == "files") {
		selected_files = d["files"];
	}

	List<Node *> list = editor->get_editor_selection()->get_selected_node_list();
	if (list.size() == 0) {
		Node *root_node = editor->get_edited_scene();
		if (root_node) {
			list.push_back(root_node);
		} else {
			drop_pos = p_point;
			target_node = NULL;
			_show_resource_type_selector();
			return;
		}
	}
	if (list.size() != 1) {
		accept->get_ok()->set_text(TTR("I see.."));
		accept->set_text(TTR("This operation requires a single selected node."));
		accept->popup_centered_minsize();
		_remove_preview();
		return;
	}

	target_node = list[0];
	if (is_shift && target_node != editor->get_edited_scene()) {
		target_node = target_node->get_parent();
	}
	drop_pos = p_point;

	if (is_alt) {
		_show_resource_type_selector();
	} else {
		_perform_drop_data();
	}
}

void CanvasItemEditorViewport::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			connect("mouse_exited", this, "_on_mouse_exit");
			label->add_color_override("font_color", get_color("warning_color", "Editor"));
		} break;
		case NOTIFICATION_EXIT_TREE: {
			disconnect("mouse_exited", this, "_on_mouse_exit");
		} break;

		default: break;
	}
}

void CanvasItemEditorViewport::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_on_select_type"), &CanvasItemEditorViewport::_on_select_type);
	ClassDB::bind_method(D_METHOD("_on_change_type_confirmed"), &CanvasItemEditorViewport::_on_change_type_confirmed);
	ClassDB::bind_method(D_METHOD("_on_change_type_closed"), &CanvasItemEditorViewport::_on_change_type_closed);
	ClassDB::bind_method(D_METHOD("_on_mouse_exit"), &CanvasItemEditorViewport::_on_mouse_exit);
}

CanvasItemEditorViewport::CanvasItemEditorViewport(EditorNode *p_node, CanvasItemEditor *p_canvas) {
	default_type = "Sprite";
	// Node2D
	types.push_back("Sprite");
	types.push_back("Light2D");
	types.push_back("Particles2D");
	types.push_back("Polygon2D");
	types.push_back("TouchScreenButton");
	// Control
	types.push_back("TextureRect");
	types.push_back("NinePatchRect");

	target_node = NULL;
	editor = p_node;
	editor_data = editor->get_scene_tree_dock()->get_editor_data();
	canvas = p_canvas;
	preview_node = memnew(Node2D);

	accept = memnew(AcceptDialog);
	editor->get_gui_base()->add_child(accept);

	selector = memnew(AcceptDialog);
	editor->get_gui_base()->add_child(selector);
	selector->set_title(TTR("Change default type"));
	selector->connect("confirmed", this, "_on_change_type_confirmed");
	selector->connect("popup_hide", this, "_on_change_type_closed");

	VBoxContainer *vbc = memnew(VBoxContainer);
	selector->add_child(vbc);
	vbc->set_h_size_flags(SIZE_EXPAND_FILL);
	vbc->set_v_size_flags(SIZE_EXPAND_FILL);
	vbc->set_custom_minimum_size(Size2(200, 260) * EDSCALE);

	btn_group = memnew(VBoxContainer);
	vbc->add_child(btn_group);
	btn_group->set_h_size_flags(0);

	button_group.instance();
	for (int i = 0; i < types.size(); i++) {
		CheckBox *check = memnew(CheckBox);
		btn_group->add_child(check);
		check->set_text(types[i]);
		check->connect("button_down", this, "_on_select_type", varray(check));
		check->set_button_group(button_group);
	}

	label = memnew(Label);
	label->add_color_override("font_color_shadow", Color(0, 0, 0, 1));
	label->add_constant_override("shadow_as_outline", 1 * EDSCALE);
	label->hide();
	editor->get_gui_base()->add_child(label);

	label_desc = memnew(Label);
	label_desc->set_text(TTR("Drag & drop + Shift : Add node as sibling\nDrag & drop + Alt : Change node type"));
	label_desc->add_color_override("font_color", Color(0.6f, 0.6f, 0.6f, 1));
	label_desc->add_color_override("font_color_shadow", Color(0.2f, 0.2f, 0.2f, 1));
	label_desc->add_constant_override("shadow_as_outline", 1 * EDSCALE);
	label_desc->add_constant_override("line_spacing", 0);
	label_desc->hide();
	editor->get_gui_base()->add_child(label_desc);
}

CanvasItemEditorViewport::~CanvasItemEditorViewport() {
	memdelete(preview_node);
}
