/*************************************************************************/
/*  canvas_item_editor_plugin.cpp                                        */
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

#include "canvas_item_editor_plugin.h"

#include "core/config/project_settings.h"
#include "core/input/input.h"
#include "core/math/geometry_2d.h"
#include "core/os/keyboard.h"
#include "core/string/print_string.h"
#include "editor/debugger/editor_debugger_node.h"
#include "editor/editor_node.h"
#include "editor/editor_scale.h"
#include "editor/editor_settings.h"
#include "editor/plugins/animation_player_editor_plugin.h"
#include "editor/plugins/script_editor_plugin.h"
#include "scene/2d/cpu_particles_2d.h"
#include "scene/2d/gpu_particles_2d.h"
#include "scene/2d/light_2d.h"
#include "scene/2d/polygon_2d.h"
#include "scene/2d/skeleton_2d.h"
#include "scene/2d/sprite_2d.h"
#include "scene/2d/touch_screen_button.h"
#include "scene/gui/grid_container.h"
#include "scene/gui/nine_patch_rect.h"
#include "scene/gui/subviewport_container.h"
#include "scene/main/canvas_layer.h"
#include "scene/main/window.h"
#include "scene/resources/packed_scene.h"

// Min and Max are power of two in order to play nicely with successive increment.
// That way, we can naturally reach a 100% zoom from boundaries.
#define MIN_ZOOM 1. / 128
#define MAX_ZOOM 128

#define RULER_WIDTH (15 * EDSCALE)
#define SCALE_HANDLE_DISTANCE 25
#define MOVE_HANDLE_DISTANCE 25

class SnapDialog : public ConfirmationDialog {
	GDCLASS(SnapDialog, ConfirmationDialog);

	friend class CanvasItemEditor;

	SpinBox *grid_offset_x;
	SpinBox *grid_offset_y;
	SpinBox *grid_step_x;
	SpinBox *grid_step_y;
	SpinBox *primary_grid_steps;
	SpinBox *rotation_offset;
	SpinBox *rotation_step;
	SpinBox *scale_step;

public:
	SnapDialog() {
		const int SPIN_BOX_GRID_RANGE = 16384;
		const int SPIN_BOX_ROTATION_RANGE = 360;
		const real_t SPIN_BOX_SCALE_MIN = 0.01;
		const real_t SPIN_BOX_SCALE_MAX = 100;

		Label *label;
		VBoxContainer *container;
		GridContainer *child_container;

		set_title(TTR("Configure Snap"));

		container = memnew(VBoxContainer);
		add_child(container);

		child_container = memnew(GridContainer);
		child_container->set_columns(3);
		container->add_child(child_container);

		label = memnew(Label);
		label->set_text(TTR("Grid Offset:"));
		child_container->add_child(label);
		label->set_h_size_flags(Control::SIZE_EXPAND_FILL);

		grid_offset_x = memnew(SpinBox);
		grid_offset_x->set_min(-SPIN_BOX_GRID_RANGE);
		grid_offset_x->set_max(SPIN_BOX_GRID_RANGE);
		grid_offset_x->set_allow_lesser(true);
		grid_offset_x->set_allow_greater(true);
		grid_offset_x->set_suffix("px");
		grid_offset_x->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		child_container->add_child(grid_offset_x);

		grid_offset_y = memnew(SpinBox);
		grid_offset_y->set_min(-SPIN_BOX_GRID_RANGE);
		grid_offset_y->set_max(SPIN_BOX_GRID_RANGE);
		grid_offset_y->set_allow_lesser(true);
		grid_offset_y->set_allow_greater(true);
		grid_offset_y->set_suffix("px");
		grid_offset_y->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		child_container->add_child(grid_offset_y);

		label = memnew(Label);
		label->set_text(TTR("Grid Step:"));
		child_container->add_child(label);
		label->set_h_size_flags(Control::SIZE_EXPAND_FILL);

		grid_step_x = memnew(SpinBox);
		grid_step_x->set_min(0.01);
		grid_step_x->set_max(SPIN_BOX_GRID_RANGE);
		grid_step_x->set_allow_greater(true);
		grid_step_x->set_suffix("px");
		grid_step_x->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		child_container->add_child(grid_step_x);

		grid_step_y = memnew(SpinBox);
		grid_step_y->set_min(0.01);
		grid_step_y->set_max(SPIN_BOX_GRID_RANGE);
		grid_step_y->set_allow_greater(true);
		grid_step_y->set_suffix("px");
		grid_step_y->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		child_container->add_child(grid_step_y);

		child_container = memnew(GridContainer);
		child_container->set_columns(2);
		container->add_child(child_container);

		label = memnew(Label);
		label->set_text(TTR("Primary Line Every:"));
		label->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		child_container->add_child(label);

		primary_grid_steps = memnew(SpinBox);
		primary_grid_steps->set_min(0);
		primary_grid_steps->set_step(1);
		primary_grid_steps->set_max(100);
		primary_grid_steps->set_allow_greater(true);
		primary_grid_steps->set_suffix(TTR("steps"));
		primary_grid_steps->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		child_container->add_child(primary_grid_steps);

		container->add_child(memnew(HSeparator));

		// We need to create another GridContainer with the same column count,
		// so we can put an HSeparator above
		child_container = memnew(GridContainer);
		child_container->set_columns(2);
		container->add_child(child_container);

		label = memnew(Label);
		label->set_text(TTR("Rotation Offset:"));
		child_container->add_child(label);
		label->set_h_size_flags(Control::SIZE_EXPAND_FILL);

		rotation_offset = memnew(SpinBox);
		rotation_offset->set_min(-SPIN_BOX_ROTATION_RANGE);
		rotation_offset->set_max(SPIN_BOX_ROTATION_RANGE);
		rotation_offset->set_suffix("deg");
		rotation_offset->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		child_container->add_child(rotation_offset);

		label = memnew(Label);
		label->set_text(TTR("Rotation Step:"));
		child_container->add_child(label);
		label->set_h_size_flags(Control::SIZE_EXPAND_FILL);

		rotation_step = memnew(SpinBox);
		rotation_step->set_min(-SPIN_BOX_ROTATION_RANGE);
		rotation_step->set_max(SPIN_BOX_ROTATION_RANGE);
		rotation_step->set_suffix("deg");
		rotation_step->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		child_container->add_child(rotation_step);

		container->add_child(memnew(HSeparator));

		child_container = memnew(GridContainer);
		child_container->set_columns(2);
		container->add_child(child_container);
		label = memnew(Label);
		label->set_text(TTR("Scale Step:"));
		child_container->add_child(label);
		label->set_h_size_flags(Control::SIZE_EXPAND_FILL);

		scale_step = memnew(SpinBox);
		scale_step->set_min(SPIN_BOX_SCALE_MIN);
		scale_step->set_max(SPIN_BOX_SCALE_MAX);
		scale_step->set_allow_greater(true);
		scale_step->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		scale_step->set_step(0.01f);
		child_container->add_child(scale_step);
	}

	void set_fields(const Point2 p_grid_offset, const Point2 p_grid_step, const int p_primary_grid_steps, const real_t p_rotation_offset, const real_t p_rotation_step, const real_t p_scale_step) {
		grid_offset_x->set_value(p_grid_offset.x);
		grid_offset_y->set_value(p_grid_offset.y);
		grid_step_x->set_value(p_grid_step.x);
		grid_step_y->set_value(p_grid_step.y);
		primary_grid_steps->set_value(p_primary_grid_steps);
		rotation_offset->set_value(Math::rad2deg(p_rotation_offset));
		rotation_step->set_value(Math::rad2deg(p_rotation_step));
		scale_step->set_value(p_scale_step);
	}

	void get_fields(Point2 &p_grid_offset, Point2 &p_grid_step, int &p_primary_grid_steps, real_t &p_rotation_offset, real_t &p_rotation_step, real_t &p_scale_step) {
		p_grid_offset = Point2(grid_offset_x->get_value(), grid_offset_y->get_value());
		p_grid_step = Point2(grid_step_x->get_value(), grid_step_y->get_value());
		p_primary_grid_steps = int(primary_grid_steps->get_value());
		p_rotation_offset = Math::deg2rad(rotation_offset->get_value());
		p_rotation_step = Math::deg2rad(rotation_step->get_value());
		p_scale_step = scale_step->get_value();
	}
};

bool CanvasItemEditor::_is_node_locked(const Node *p_node) {
	return p_node->has_meta("_edit_lock_") && p_node->get_meta("_edit_lock_");
}

bool CanvasItemEditor::_is_node_movable(const Node *p_node, bool p_popup_warning) {
	if (_is_node_locked(p_node)) {
		return false;
	}
	if (Object::cast_to<Control>(p_node) && Object::cast_to<Container>(p_node->get_parent())) {
		if (p_popup_warning) {
			EditorToaster::get_singleton()->popup_str("Children of a container get their position and size determined only by their parent.", EditorToaster::SEVERITY_WARNING);
		}
		return false;
	}
	return true;
}

void CanvasItemEditor::_snap_if_closer_float(
		const real_t p_value,
		real_t &r_current_snap, SnapTarget &r_current_snap_target,
		const real_t p_target_value, const SnapTarget p_snap_target,
		const real_t p_radius) {
	const real_t radius = p_radius / zoom;
	const real_t dist = Math::abs(p_value - p_target_value);
	if ((p_radius < 0 || dist < radius) && (r_current_snap_target == SNAP_TARGET_NONE || dist < Math::abs(r_current_snap - p_value))) {
		r_current_snap = p_target_value;
		r_current_snap_target = p_snap_target;
	}
}

void CanvasItemEditor::_snap_if_closer_point(
		Point2 p_value,
		Point2 &r_current_snap, SnapTarget (&r_current_snap_target)[2],
		Point2 p_target_value, const SnapTarget p_snap_target,
		const real_t rotation,
		const real_t p_radius) {
	Transform2D rot_trans = Transform2D(rotation, Point2());
	p_value = rot_trans.inverse().xform(p_value);
	p_target_value = rot_trans.inverse().xform(p_target_value);
	r_current_snap = rot_trans.inverse().xform(r_current_snap);

	_snap_if_closer_float(
			p_value.x,
			r_current_snap.x,
			r_current_snap_target[0],
			p_target_value.x,
			p_snap_target,
			p_radius);

	_snap_if_closer_float(
			p_value.y,
			r_current_snap.y,
			r_current_snap_target[1],
			p_target_value.y,
			p_snap_target,
			p_radius);

	r_current_snap = rot_trans.xform(r_current_snap);
}

void CanvasItemEditor::_snap_other_nodes(
		const Point2 p_value,
		const Transform2D p_transform_to_snap,
		Point2 &r_current_snap, SnapTarget (&r_current_snap_target)[2],
		const SnapTarget p_snap_target, List<const CanvasItem *> p_exceptions,
		const Node *p_current) {
	const CanvasItem *canvas_item = Object::cast_to<CanvasItem>(p_current);

	// Check if the element is in the exception
	bool exception = false;
	for (const CanvasItem *&E : p_exceptions) {
		if (E == p_current) {
			exception = true;
			break;
		}
	};

	if (canvas_item && !exception) {
		Transform2D ci_transform = canvas_item->get_global_transform_with_canvas();
		if (fmod(ci_transform.get_rotation() - p_transform_to_snap.get_rotation(), (real_t)360.0) == 0.0) {
			if (canvas_item->_edit_use_rect()) {
				Point2 begin = ci_transform.xform(canvas_item->_edit_get_rect().get_position());
				Point2 end = ci_transform.xform(canvas_item->_edit_get_rect().get_position() + canvas_item->_edit_get_rect().get_size());

				_snap_if_closer_point(p_value, r_current_snap, r_current_snap_target, begin, p_snap_target, ci_transform.get_rotation());
				_snap_if_closer_point(p_value, r_current_snap, r_current_snap_target, end, p_snap_target, ci_transform.get_rotation());
			} else {
				Point2 position = ci_transform.xform(Point2());
				_snap_if_closer_point(p_value, r_current_snap, r_current_snap_target, position, p_snap_target, ci_transform.get_rotation());
			}
		}
	}
	for (int i = 0; i < p_current->get_child_count(); i++) {
		_snap_other_nodes(p_value, p_transform_to_snap, r_current_snap, r_current_snap_target, p_snap_target, p_exceptions, p_current->get_child(i));
	}
}

Point2 CanvasItemEditor::snap_point(Point2 p_target, unsigned int p_modes, unsigned int p_forced_modes, const CanvasItem *p_self_canvas_item, List<CanvasItem *> p_other_nodes_exceptions) {
	snap_target[0] = SNAP_TARGET_NONE;
	snap_target[1] = SNAP_TARGET_NONE;

	bool is_snap_active = smart_snap_active ^ Input::get_singleton()->is_key_pressed(Key::CTRL);

	// Smart snap using the canvas position
	Vector2 output = p_target;
	real_t rotation = 0.0;

	if (p_self_canvas_item) {
		rotation = p_self_canvas_item->get_global_transform_with_canvas().get_rotation();

		// Parent sides and center
		if ((is_snap_active && snap_node_parent && (p_modes & SNAP_NODE_PARENT)) || (p_forced_modes & SNAP_NODE_PARENT)) {
			if (const Control *c = Object::cast_to<Control>(p_self_canvas_item)) {
				Point2 begin = p_self_canvas_item->get_global_transform_with_canvas().xform(_anchor_to_position(c, Point2(0, 0)));
				Point2 end = p_self_canvas_item->get_global_transform_with_canvas().xform(_anchor_to_position(c, Point2(1, 1)));
				_snap_if_closer_point(p_target, output, snap_target, begin, SNAP_TARGET_PARENT, rotation);
				_snap_if_closer_point(p_target, output, snap_target, (begin + end) / 2.0, SNAP_TARGET_PARENT, rotation);
				_snap_if_closer_point(p_target, output, snap_target, end, SNAP_TARGET_PARENT, rotation);
			} else if (const CanvasItem *parent_ci = Object::cast_to<CanvasItem>(p_self_canvas_item->get_parent())) {
				if (parent_ci->_edit_use_rect()) {
					Point2 begin = p_self_canvas_item->get_transform().affine_inverse().xform(parent_ci->_edit_get_rect().get_position());
					Point2 end = p_self_canvas_item->get_transform().affine_inverse().xform(parent_ci->_edit_get_rect().get_position() + parent_ci->_edit_get_rect().get_size());
					_snap_if_closer_point(p_target, output, snap_target, begin, SNAP_TARGET_PARENT, rotation);
					_snap_if_closer_point(p_target, output, snap_target, (begin + end) / 2.0, SNAP_TARGET_PARENT, rotation);
					_snap_if_closer_point(p_target, output, snap_target, end, SNAP_TARGET_PARENT, rotation);
				} else {
					Point2 position = p_self_canvas_item->get_transform().affine_inverse().xform(Point2());
					_snap_if_closer_point(p_target, output, snap_target, position, SNAP_TARGET_PARENT, rotation);
				}
			}
		}

		// Self anchors
		if ((is_snap_active && snap_node_anchors && (p_modes & SNAP_NODE_ANCHORS)) || (p_forced_modes & SNAP_NODE_ANCHORS)) {
			if (const Control *c = Object::cast_to<Control>(p_self_canvas_item)) {
				Point2 begin = p_self_canvas_item->get_global_transform_with_canvas().xform(_anchor_to_position(c, Point2(c->get_anchor(SIDE_LEFT), c->get_anchor(SIDE_TOP))));
				Point2 end = p_self_canvas_item->get_global_transform_with_canvas().xform(_anchor_to_position(c, Point2(c->get_anchor(SIDE_RIGHT), c->get_anchor(SIDE_BOTTOM))));
				_snap_if_closer_point(p_target, output, snap_target, begin, SNAP_TARGET_SELF_ANCHORS, rotation);
				_snap_if_closer_point(p_target, output, snap_target, end, SNAP_TARGET_SELF_ANCHORS, rotation);
			}
		}

		// Self sides
		if ((is_snap_active && snap_node_sides && (p_modes & SNAP_NODE_SIDES)) || (p_forced_modes & SNAP_NODE_SIDES)) {
			if (p_self_canvas_item->_edit_use_rect()) {
				Point2 begin = p_self_canvas_item->get_global_transform_with_canvas().xform(p_self_canvas_item->_edit_get_rect().get_position());
				Point2 end = p_self_canvas_item->get_global_transform_with_canvas().xform(p_self_canvas_item->_edit_get_rect().get_position() + p_self_canvas_item->_edit_get_rect().get_size());
				_snap_if_closer_point(p_target, output, snap_target, begin, SNAP_TARGET_SELF, rotation);
				_snap_if_closer_point(p_target, output, snap_target, end, SNAP_TARGET_SELF, rotation);
			}
		}

		// Self center
		if ((is_snap_active && snap_node_center && (p_modes & SNAP_NODE_CENTER)) || (p_forced_modes & SNAP_NODE_CENTER)) {
			if (p_self_canvas_item->_edit_use_rect()) {
				Point2 center = p_self_canvas_item->get_global_transform_with_canvas().xform(p_self_canvas_item->_edit_get_rect().get_center());
				_snap_if_closer_point(p_target, output, snap_target, center, SNAP_TARGET_SELF, rotation);
			} else {
				Point2 position = p_self_canvas_item->get_global_transform_with_canvas().xform(Point2());
				_snap_if_closer_point(p_target, output, snap_target, position, SNAP_TARGET_SELF, rotation);
			}
		}
	}

	// Other nodes sides
	if ((is_snap_active && snap_other_nodes && (p_modes & SNAP_OTHER_NODES)) || (p_forced_modes & SNAP_OTHER_NODES)) {
		Transform2D to_snap_transform = Transform2D();
		List<const CanvasItem *> exceptions = List<const CanvasItem *>();
		for (const CanvasItem *E : p_other_nodes_exceptions) {
			exceptions.push_back(E);
		}
		if (p_self_canvas_item) {
			exceptions.push_back(p_self_canvas_item);
			to_snap_transform = p_self_canvas_item->get_global_transform_with_canvas();
		}

		_snap_other_nodes(
				p_target, to_snap_transform,
				output, snap_target,
				SNAP_TARGET_OTHER_NODE,
				exceptions,
				get_tree()->get_edited_scene_root());
	}

	if (((is_snap_active && snap_guides && (p_modes & SNAP_GUIDES)) || (p_forced_modes & SNAP_GUIDES)) && fmod(rotation, (real_t)360.0) == 0.0) {
		// Guides
		if (EditorNode::get_singleton()->get_edited_scene() && EditorNode::get_singleton()->get_edited_scene()->has_meta("_edit_vertical_guides_")) {
			Array vguides = EditorNode::get_singleton()->get_edited_scene()->get_meta("_edit_vertical_guides_");
			for (int i = 0; i < vguides.size(); i++) {
				_snap_if_closer_float(p_target.x, output.x, snap_target[0], vguides[i], SNAP_TARGET_GUIDE);
			}
		}

		if (EditorNode::get_singleton()->get_edited_scene() && EditorNode::get_singleton()->get_edited_scene()->has_meta("_edit_horizontal_guides_")) {
			Array hguides = EditorNode::get_singleton()->get_edited_scene()->get_meta("_edit_horizontal_guides_");
			for (int i = 0; i < hguides.size(); i++) {
				_snap_if_closer_float(p_target.y, output.y, snap_target[1], hguides[i], SNAP_TARGET_GUIDE);
			}
		}
	}

	if (((grid_snap_active && (p_modes & SNAP_GRID)) || (p_forced_modes & SNAP_GRID)) && fmod(rotation, (real_t)360.0) == 0.0) {
		// Grid
		Point2 offset = grid_offset;
		if (snap_relative) {
			List<CanvasItem *> selection = _get_edited_canvas_items();
			if (selection.size() == 1 && Object::cast_to<Node2D>(selection[0])) {
				offset = Object::cast_to<Node2D>(selection[0])->get_global_position();
			} else if (selection.size() > 0) {
				offset = _get_encompassing_rect_from_list(selection).position;
			}
		}
		Point2 grid_output;
		grid_output.x = Math::snapped(p_target.x - offset.x, grid_step.x * Math::pow(2.0, grid_step_multiplier)) + offset.x;
		grid_output.y = Math::snapped(p_target.y - offset.y, grid_step.y * Math::pow(2.0, grid_step_multiplier)) + offset.y;
		_snap_if_closer_point(p_target, output, snap_target, grid_output, SNAP_TARGET_GRID, 0.0, -1.0);
	}

	if (((snap_pixel && (p_modes & SNAP_PIXEL)) || (p_forced_modes & SNAP_PIXEL)) && rotation == 0.0) {
		// Pixel
		output = output.snapped(Size2(1, 1));
	}

	snap_transform = Transform2D(rotation, output);

	return output;
}

real_t CanvasItemEditor::snap_angle(real_t p_target, real_t p_start) const {
	if (((smart_snap_active || snap_rotation) ^ Input::get_singleton()->is_key_pressed(Key::CTRL)) && snap_rotation_step != 0) {
		if (snap_relative) {
			return Math::snapped(p_target - snap_rotation_offset, snap_rotation_step) + snap_rotation_offset + (p_start - (int)(p_start / snap_rotation_step) * snap_rotation_step);
		} else {
			return Math::snapped(p_target - snap_rotation_offset, snap_rotation_step) + snap_rotation_offset;
		}
	} else {
		return p_target;
	}
}

void CanvasItemEditor::unhandled_key_input(const Ref<InputEvent> &p_ev) {
	ERR_FAIL_COND(p_ev.is_null());

	Ref<InputEventKey> k = p_ev;

	if (!is_visible_in_tree()) {
		return;
	}

	if (k.is_valid()) {
		if (k->get_keycode() == Key::CTRL || k->get_keycode() == Key::ALT || k->get_keycode() == Key::SHIFT) {
			viewport->update();
		}

		if (k->is_pressed() && !k->is_ctrl_pressed() && !k->is_echo()) {
			if ((grid_snap_active || show_grid) && multiply_grid_step_shortcut.is_valid() && multiply_grid_step_shortcut->matches_event(p_ev)) {
				// Multiply the grid size
				grid_step_multiplier = MIN(grid_step_multiplier + 1, 12);
				viewport->update();
			} else if ((grid_snap_active || show_grid) && divide_grid_step_shortcut.is_valid() && divide_grid_step_shortcut->matches_event(p_ev)) {
				// Divide the grid size
				Point2 new_grid_step = grid_step * Math::pow(2.0, grid_step_multiplier - 1);
				if (new_grid_step.x >= 1.0 && new_grid_step.y >= 1.0) {
					grid_step_multiplier--;
				}
				viewport->update();
			}
		}
	}
}

Object *CanvasItemEditor::_get_editor_data(Object *p_what) {
	CanvasItem *ci = Object::cast_to<CanvasItem>(p_what);
	if (!ci) {
		return nullptr;
	}

	return memnew(CanvasItemEditorSelectedItem);
}

void CanvasItemEditor::_keying_changed() {
	if (AnimationPlayerEditor::get_singleton()->get_track_editor()->is_visible_in_tree()) {
		animation_hb->show();
	} else {
		animation_hb->hide();
	}
}

Rect2 CanvasItemEditor::_get_encompassing_rect_from_list(List<CanvasItem *> p_list) {
	ERR_FAIL_COND_V(p_list.is_empty(), Rect2());

	// Handles the first element
	CanvasItem *canvas_item = p_list.front()->get();
	Rect2 rect = Rect2(canvas_item->get_global_transform_with_canvas().xform(canvas_item->_edit_get_rect().get_center()), Size2());

	// Expand with the other ones
	for (CanvasItem *canvas_item2 : p_list) {
		Transform2D xform = canvas_item2->get_global_transform_with_canvas();

		Rect2 current_rect = canvas_item2->_edit_get_rect();
		rect.expand_to(xform.xform(current_rect.position));
		rect.expand_to(xform.xform(current_rect.position + Vector2(current_rect.size.x, 0)));
		rect.expand_to(xform.xform(current_rect.position + current_rect.size));
		rect.expand_to(xform.xform(current_rect.position + Vector2(0, current_rect.size.y)));
	}

	return rect;
}

void CanvasItemEditor::_expand_encompassing_rect_using_children(Rect2 &r_rect, const Node *p_node, bool &r_first, const Transform2D &p_parent_xform, const Transform2D &p_canvas_xform, bool include_locked_nodes) {
	if (!p_node) {
		return;
	}
	if (Object::cast_to<Viewport>(p_node)) {
		return;
	}

	const CanvasItem *canvas_item = Object::cast_to<CanvasItem>(p_node);

	for (int i = p_node->get_child_count() - 1; i >= 0; i--) {
		if (canvas_item && !canvas_item->is_set_as_top_level()) {
			_expand_encompassing_rect_using_children(r_rect, p_node->get_child(i), r_first, p_parent_xform * canvas_item->get_transform(), p_canvas_xform);
		} else {
			const CanvasLayer *canvas_layer = Object::cast_to<CanvasLayer>(p_node);
			_expand_encompassing_rect_using_children(r_rect, p_node->get_child(i), r_first, Transform2D(), canvas_layer ? canvas_layer->get_transform() : p_canvas_xform);
		}
	}

	if (canvas_item && canvas_item->is_visible_in_tree() && (include_locked_nodes || !_is_node_locked(canvas_item))) {
		Transform2D xform = p_parent_xform * p_canvas_xform * canvas_item->get_transform();
		Rect2 rect = canvas_item->_edit_get_rect();
		if (r_first) {
			r_rect = Rect2(xform.xform(rect.get_center()), Size2());
			r_first = false;
		}
		r_rect.expand_to(xform.xform(rect.position));
		r_rect.expand_to(xform.xform(rect.position + Point2(rect.size.x, 0)));
		r_rect.expand_to(xform.xform(rect.position + Point2(0, rect.size.y)));
		r_rect.expand_to(xform.xform(rect.position + rect.size));
	}
}

Rect2 CanvasItemEditor::_get_encompassing_rect(const Node *p_node) {
	Rect2 rect;
	bool first = true;
	_expand_encompassing_rect_using_children(rect, p_node, first);

	return rect;
}

void CanvasItemEditor::_find_canvas_items_at_pos(const Point2 &p_pos, Node *p_node, Vector<_SelectResult> &r_items, const Transform2D &p_parent_xform, const Transform2D &p_canvas_xform) {
	if (!p_node) {
		return;
	}
	if (Object::cast_to<Viewport>(p_node)) {
		return;
	}

	const real_t grab_distance = EDITOR_GET("editors/polygon_editor/point_grab_radius");
	CanvasItem *canvas_item = Object::cast_to<CanvasItem>(p_node);

	for (int i = p_node->get_child_count() - 1; i >= 0; i--) {
		if (canvas_item) {
			if (!canvas_item->is_set_as_top_level()) {
				_find_canvas_items_at_pos(p_pos, p_node->get_child(i), r_items, p_parent_xform * canvas_item->get_transform(), p_canvas_xform);
			} else {
				_find_canvas_items_at_pos(p_pos, p_node->get_child(i), r_items, canvas_item->get_transform(), p_canvas_xform);
			}
		} else {
			CanvasLayer *cl = Object::cast_to<CanvasLayer>(p_node);
			_find_canvas_items_at_pos(p_pos, p_node->get_child(i), r_items, Transform2D(), cl ? cl->get_transform() : p_canvas_xform);
		}
	}

	if (canvas_item && canvas_item->is_visible_in_tree()) {
		Transform2D xform = (p_parent_xform * p_canvas_xform * canvas_item->get_transform()).affine_inverse();
		const real_t local_grab_distance = xform.basis_xform(Vector2(grab_distance, 0)).length() / zoom;
		if (canvas_item->_edit_is_selected_on_click(xform.xform(p_pos), local_grab_distance)) {
			Node2D *node = Object::cast_to<Node2D>(canvas_item);

			_SelectResult res;
			res.item = canvas_item;
			res.z_index = node ? node->get_z_index() : 0;
			res.has_z = node;
			r_items.push_back(res);
		}
	}
}

void CanvasItemEditor::_get_canvas_items_at_pos(const Point2 &p_pos, Vector<_SelectResult> &r_items, bool p_allow_locked) {
	Node *scene = editor->get_edited_scene();

	_find_canvas_items_at_pos(p_pos, scene, r_items);

	//Remove invalid results
	for (int i = 0; i < r_items.size(); i++) {
		Node *node = r_items[i].item;

		// Make sure the selected node is in the current scene, or editable
		if (node && node != get_tree()->get_edited_scene_root()) {
			node = scene->get_deepest_editable_node(node);
		}

		CanvasItem *canvas_item = Object::cast_to<CanvasItem>(node);
		if (!p_allow_locked) {
			// Replace the node by the group if grouped
			while (node && node != scene->get_parent()) {
				CanvasItem *canvas_item_tmp = Object::cast_to<CanvasItem>(node);
				if (canvas_item_tmp && node->has_meta("_edit_group_")) {
					canvas_item = canvas_item_tmp;
				}
				node = node->get_parent();
			}
		}

		// Check if the canvas item is already in the list (for groups or scenes)
		bool duplicate = false;
		for (int j = 0; j < i; j++) {
			if (r_items[j].item == canvas_item) {
				duplicate = true;
				break;
			}
		}

		//Remove the item if invalid
		if (!canvas_item || duplicate || (canvas_item != scene && canvas_item->get_owner() != scene && !scene->is_editable_instance(canvas_item->get_owner())) || (!p_allow_locked && _is_node_locked(canvas_item))) {
			r_items.remove_at(i);
			i--;
		} else {
			r_items.write[i].item = canvas_item;
		}
	}
}

void CanvasItemEditor::_find_canvas_items_in_rect(const Rect2 &p_rect, Node *p_node, List<CanvasItem *> *r_items, const Transform2D &p_parent_xform, const Transform2D &p_canvas_xform) {
	if (!p_node) {
		return;
	}
	if (Object::cast_to<Viewport>(p_node)) {
		return;
	}

	CanvasItem *canvas_item = Object::cast_to<CanvasItem>(p_node);
	Node *scene = editor->get_edited_scene();

	bool editable = p_node == scene || p_node->get_owner() == scene || p_node == scene->get_deepest_editable_node(p_node);
	bool lock_children = p_node->has_meta("_edit_group_") && p_node->get_meta("_edit_group_");
	bool locked = _is_node_locked(p_node);

	if (!lock_children || !editable) {
		for (int i = p_node->get_child_count() - 1; i >= 0; i--) {
			if (canvas_item) {
				if (!canvas_item->is_set_as_top_level()) {
					_find_canvas_items_in_rect(p_rect, p_node->get_child(i), r_items, p_parent_xform * canvas_item->get_transform(), p_canvas_xform);
				} else {
					_find_canvas_items_in_rect(p_rect, p_node->get_child(i), r_items, canvas_item->get_transform(), p_canvas_xform);
				}
			} else {
				CanvasLayer *canvas_layer = Object::cast_to<CanvasLayer>(p_node);
				_find_canvas_items_in_rect(p_rect, p_node->get_child(i), r_items, Transform2D(), canvas_layer ? canvas_layer->get_transform() : p_canvas_xform);
			}
		}
	}

	if (canvas_item && canvas_item->is_visible_in_tree() && !locked && editable) {
		Transform2D xform = p_parent_xform * p_canvas_xform * canvas_item->get_transform();

		if (canvas_item->_edit_use_rect()) {
			Rect2 rect = canvas_item->_edit_get_rect();
			if (p_rect.has_point(xform.xform(rect.position)) &&
					p_rect.has_point(xform.xform(rect.position + Vector2(rect.size.x, 0))) &&
					p_rect.has_point(xform.xform(rect.position + Vector2(rect.size.x, rect.size.y))) &&
					p_rect.has_point(xform.xform(rect.position + Vector2(0, rect.size.y)))) {
				r_items->push_back(canvas_item);
			}
		} else {
			if (p_rect.has_point(xform.xform(Point2()))) {
				r_items->push_back(canvas_item);
			}
		}
	}
}

bool CanvasItemEditor::_select_click_on_item(CanvasItem *item, Point2 p_click_pos, bool p_append) {
	bool still_selected = true;
	if (p_append && !editor_selection->get_selected_node_list().is_empty()) {
		if (editor_selection->is_selected(item)) {
			// Already in the selection, remove it from the selected nodes
			editor_selection->remove_node(item);
			still_selected = false;

			if (editor_selection->get_selected_node_list().size() == 1) {
				editor->push_item(editor_selection->get_selected_node_list()[0]);
			}
		} else {
			// Add the item to the selection
			editor_selection->add_node(item);
		}
	} else {
		if (!editor_selection->is_selected(item)) {
			// Select a new one and clear previous selection
			editor_selection->clear();
			editor_selection->add_node(item);
			// Reselect
			if (Engine::get_singleton()->is_editor_hint()) {
				selected_from_canvas = true;
				editor->call("edit_node", item);
			}
		}
	}
	viewport->update();
	return still_selected;
}

List<CanvasItem *> CanvasItemEditor::_get_edited_canvas_items(bool retreive_locked, bool remove_canvas_item_if_parent_in_selection) {
	List<CanvasItem *> selection;
	for (const KeyValue<Node *, Object *> &E : editor_selection->get_selection()) {
		CanvasItem *canvas_item = Object::cast_to<CanvasItem>(E.key);
		if (canvas_item && canvas_item->is_visible_in_tree() && canvas_item->get_viewport() == EditorNode::get_singleton()->get_scene_root() && (retreive_locked || !_is_node_locked(canvas_item))) {
			CanvasItemEditorSelectedItem *se = editor_selection->get_node_editor_data<CanvasItemEditorSelectedItem>(canvas_item);
			if (se) {
				selection.push_back(canvas_item);
			}
		}
	}

	if (remove_canvas_item_if_parent_in_selection) {
		List<CanvasItem *> filtered_selection;
		for (CanvasItem *E : selection) {
			if (!selection.find(E->get_parent())) {
				filtered_selection.push_back(E);
			}
		}
		return filtered_selection;
	} else {
		return selection;
	}
}

Vector2 CanvasItemEditor::_anchor_to_position(const Control *p_control, Vector2 anchor) {
	ERR_FAIL_COND_V(!p_control, Vector2());

	Transform2D parent_transform = p_control->get_transform().affine_inverse();
	Rect2 parent_rect = p_control->get_parent_anchorable_rect();

	if (p_control->is_layout_rtl()) {
		return parent_transform.xform(parent_rect.position + Vector2(parent_rect.size.x - parent_rect.size.x * anchor.x, parent_rect.size.y * anchor.y));
	} else {
		return parent_transform.xform(parent_rect.position + Vector2(parent_rect.size.x * anchor.x, parent_rect.size.y * anchor.y));
	}
}

Vector2 CanvasItemEditor::_position_to_anchor(const Control *p_control, Vector2 position) {
	ERR_FAIL_COND_V(!p_control, Vector2());

	Rect2 parent_rect = p_control->get_parent_anchorable_rect();

	Vector2 output = Vector2();
	if (p_control->is_layout_rtl()) {
		output.x = (parent_rect.size.x == 0) ? 0.0 : (parent_rect.size.x - p_control->get_transform().xform(position).x - parent_rect.position.x) / parent_rect.size.x;
	} else {
		output.x = (parent_rect.size.x == 0) ? 0.0 : (p_control->get_transform().xform(position).x - parent_rect.position.x) / parent_rect.size.x;
	}
	output.y = (parent_rect.size.y == 0) ? 0.0 : (p_control->get_transform().xform(position).y - parent_rect.position.y) / parent_rect.size.y;
	return output;
}

void CanvasItemEditor::_save_canvas_item_state(List<CanvasItem *> p_canvas_items, bool save_bones) {
	for (CanvasItem *canvas_item : p_canvas_items) {
		CanvasItemEditorSelectedItem *se = editor_selection->get_node_editor_data<CanvasItemEditorSelectedItem>(canvas_item);
		if (se) {
			se->undo_state = canvas_item->_edit_get_state();
			se->pre_drag_xform = canvas_item->get_global_transform_with_canvas();
			if (canvas_item->_edit_use_rect()) {
				se->pre_drag_rect = canvas_item->_edit_get_rect();
			} else {
				se->pre_drag_rect = Rect2();
			}
		}
	}
}

void CanvasItemEditor::_restore_canvas_item_state(List<CanvasItem *> p_canvas_items, bool restore_bones) {
	for (CanvasItem *canvas_item : drag_selection) {
		CanvasItemEditorSelectedItem *se = editor_selection->get_node_editor_data<CanvasItemEditorSelectedItem>(canvas_item);
		canvas_item->_edit_set_state(se->undo_state);
	}
}

void CanvasItemEditor::_commit_canvas_item_state(List<CanvasItem *> p_canvas_items, String action_name, bool commit_bones) {
	List<CanvasItem *> modified_canvas_items;
	for (CanvasItem *canvas_item : p_canvas_items) {
		Dictionary old_state = editor_selection->get_node_editor_data<CanvasItemEditorSelectedItem>(canvas_item)->undo_state;
		Dictionary new_state = canvas_item->_edit_get_state();

		if (old_state.hash() != new_state.hash()) {
			modified_canvas_items.push_back(canvas_item);
		}
	}

	if (modified_canvas_items.is_empty()) {
		return;
	}

	undo_redo->create_action(action_name);
	for (CanvasItem *canvas_item : modified_canvas_items) {
		CanvasItemEditorSelectedItem *se = editor_selection->get_node_editor_data<CanvasItemEditorSelectedItem>(canvas_item);
		if (se) {
			undo_redo->add_do_method(canvas_item, "_edit_set_state", canvas_item->_edit_get_state());
			undo_redo->add_undo_method(canvas_item, "_edit_set_state", se->undo_state);
			if (commit_bones) {
				for (const Dictionary &F : se->pre_drag_bones_undo_state) {
					canvas_item = Object::cast_to<CanvasItem>(canvas_item->get_parent());
					undo_redo->add_do_method(canvas_item, "_edit_set_state", canvas_item->_edit_get_state());
					undo_redo->add_undo_method(canvas_item, "_edit_set_state", F);
				}
			}
		}
	}
	undo_redo->add_do_method(viewport, "update");
	undo_redo->add_undo_method(viewport, "update");
	undo_redo->commit_action();
}

void CanvasItemEditor::_snap_changed() {
	((SnapDialog *)snap_dialog)->get_fields(grid_offset, grid_step, primary_grid_steps, snap_rotation_offset, snap_rotation_step, snap_scale_step);
	grid_step_multiplier = 0;
	viewport->update();
}

void CanvasItemEditor::_selection_result_pressed(int p_result) {
	if (selection_results.size() <= p_result) {
		return;
	}

	CanvasItem *item = selection_results[p_result].item;

	if (item) {
		_select_click_on_item(item, Point2(), selection_menu_additive_selection);
	}
}

void CanvasItemEditor::_selection_menu_hide() {
	selection_results.clear();
	selection_menu->clear();
	selection_menu->reset_size();
}

void CanvasItemEditor::_add_node_pressed(int p_result) {
	if (p_result == AddNodeOption::ADD_NODE) {
		editor->get_scene_tree_dock()->open_add_child_dialog();
	} else if (p_result == AddNodeOption::ADD_INSTANCE) {
		editor->get_scene_tree_dock()->open_instance_child_dialog();
	}
}

void CanvasItemEditor::_node_created(Node *p_node) {
	if (node_create_position == Point2()) {
		return;
	}

	CanvasItem *c = Object::cast_to<CanvasItem>(p_node);
	if (c) {
		Transform2D xform = c->get_global_transform_with_canvas().affine_inverse() * c->get_transform();
		c->_edit_set_position(xform.xform(node_create_position));
	}

	call_deferred(SNAME("_reset_create_position")); // Defer the call in case more than one node is added.
}

void CanvasItemEditor::_reset_create_position() {
	node_create_position = Point2();
}

bool CanvasItemEditor::_gui_input_rulers_and_guides(const Ref<InputEvent> &p_event) {
	Ref<InputEventMouseButton> b = p_event;
	Ref<InputEventMouseMotion> m = p_event;

	if (drag_type == DRAG_NONE) {
		if (show_guides && show_rulers && EditorNode::get_singleton()->get_edited_scene()) {
			Transform2D xform = viewport_scrollable->get_transform() * transform;
			// Retrieve the guide lists
			Array vguides;
			if (EditorNode::get_singleton()->get_edited_scene()->has_meta("_edit_vertical_guides_")) {
				vguides = EditorNode::get_singleton()->get_edited_scene()->get_meta("_edit_vertical_guides_");
			}
			Array hguides;
			if (EditorNode::get_singleton()->get_edited_scene()->has_meta("_edit_horizontal_guides_")) {
				hguides = EditorNode::get_singleton()->get_edited_scene()->get_meta("_edit_horizontal_guides_");
			}

			// Hover over guides
			real_t minimum = 1e20;
			is_hovering_h_guide = false;
			is_hovering_v_guide = false;

			if (m.is_valid() && m->get_position().x < RULER_WIDTH) {
				// Check if we are hovering an existing horizontal guide
				for (int i = 0; i < hguides.size(); i++) {
					if (ABS(xform.xform(Point2(0, hguides[i])).y - m->get_position().y) < MIN(minimum, 8)) {
						is_hovering_h_guide = true;
						is_hovering_v_guide = false;
						break;
					}
				}

			} else if (m.is_valid() && m->get_position().y < RULER_WIDTH) {
				// Check if we are hovering an existing vertical guide
				for (int i = 0; i < vguides.size(); i++) {
					if (ABS(xform.xform(Point2(vguides[i], 0)).x - m->get_position().x) < MIN(minimum, 8)) {
						is_hovering_v_guide = true;
						is_hovering_h_guide = false;
						break;
					}
				}
			}

			// Start dragging a guide
			if (b.is_valid() && b->get_button_index() == MouseButton::LEFT && b->is_pressed()) {
				// Press button
				if (b->get_position().x < RULER_WIDTH && b->get_position().y < RULER_WIDTH) {
					// Drag a new double guide
					drag_type = DRAG_DOUBLE_GUIDE;
					dragged_guide_index = -1;
					return true;
				} else if (b->get_position().x < RULER_WIDTH) {
					// Check if we drag an existing horizontal guide
					dragged_guide_index = -1;
					for (int i = 0; i < hguides.size(); i++) {
						if (ABS(xform.xform(Point2(0, hguides[i])).y - b->get_position().y) < MIN(minimum, 8)) {
							dragged_guide_index = i;
						}
					}

					if (dragged_guide_index >= 0) {
						// Drag an existing horizontal guide
						drag_type = DRAG_H_GUIDE;
					} else {
						// Drag a new vertical guide
						drag_type = DRAG_V_GUIDE;
					}
					return true;
				} else if (b->get_position().y < RULER_WIDTH) {
					// Check if we drag an existing vertical guide
					dragged_guide_index = -1;
					for (int i = 0; i < vguides.size(); i++) {
						if (ABS(xform.xform(Point2(vguides[i], 0)).x - b->get_position().x) < MIN(minimum, 8)) {
							dragged_guide_index = i;
						}
					}

					if (dragged_guide_index >= 0) {
						// Drag an existing vertical guide
						drag_type = DRAG_V_GUIDE;
					} else {
						// Drag a new vertical guide
						drag_type = DRAG_H_GUIDE;
					}
					drag_from = xform.affine_inverse().xform(b->get_position());
					return true;
				}
			}
		}
	}

	if (drag_type == DRAG_DOUBLE_GUIDE || drag_type == DRAG_V_GUIDE || drag_type == DRAG_H_GUIDE) {
		// Move the guide
		if (m.is_valid()) {
			Transform2D xform = viewport_scrollable->get_transform() * transform;
			drag_to = xform.affine_inverse().xform(m->get_position());

			dragged_guide_pos = xform.xform(snap_point(drag_to, SNAP_GRID | SNAP_PIXEL | SNAP_OTHER_NODES));
			viewport->update();
			return true;
		}

		// Release confirms the guide move
		if (b.is_valid() && b->get_button_index() == MouseButton::LEFT && !b->is_pressed()) {
			if (show_guides && EditorNode::get_singleton()->get_edited_scene()) {
				Transform2D xform = viewport_scrollable->get_transform() * transform;

				// Retrieve the guide lists
				Array vguides;
				if (EditorNode::get_singleton()->get_edited_scene()->has_meta("_edit_vertical_guides_")) {
					vguides = EditorNode::get_singleton()->get_edited_scene()->get_meta("_edit_vertical_guides_");
				}
				Array hguides;
				if (EditorNode::get_singleton()->get_edited_scene()->has_meta("_edit_horizontal_guides_")) {
					hguides = EditorNode::get_singleton()->get_edited_scene()->get_meta("_edit_horizontal_guides_");
				}

				Point2 edited = snap_point(xform.affine_inverse().xform(b->get_position()), SNAP_GRID | SNAP_PIXEL | SNAP_OTHER_NODES);
				if (drag_type == DRAG_V_GUIDE) {
					Array prev_vguides = vguides.duplicate();
					if (b->get_position().x > RULER_WIDTH) {
						// Adds a new vertical guide
						if (dragged_guide_index >= 0) {
							vguides[dragged_guide_index] = edited.x;
							undo_redo->create_action(TTR("Move Vertical Guide"));
							undo_redo->add_do_method(EditorNode::get_singleton()->get_edited_scene(), "set_meta", "_edit_vertical_guides_", vguides);
							undo_redo->add_undo_method(EditorNode::get_singleton()->get_edited_scene(), "set_meta", "_edit_vertical_guides_", prev_vguides);
							undo_redo->add_undo_method(viewport, "update");
							undo_redo->commit_action();
						} else {
							vguides.push_back(edited.x);
							undo_redo->create_action(TTR("Create Vertical Guide"));
							undo_redo->add_do_method(EditorNode::get_singleton()->get_edited_scene(), "set_meta", "_edit_vertical_guides_", vguides);
							undo_redo->add_undo_method(EditorNode::get_singleton()->get_edited_scene(), "set_meta", "_edit_vertical_guides_", prev_vguides);
							undo_redo->add_undo_method(viewport, "update");
							undo_redo->commit_action();
						}
					} else {
						if (dragged_guide_index >= 0) {
							vguides.remove_at(dragged_guide_index);
							undo_redo->create_action(TTR("Remove Vertical Guide"));
							if (vguides.is_empty()) {
								undo_redo->add_do_method(EditorNode::get_singleton()->get_edited_scene(), "remove_meta", "_edit_vertical_guides_");
							} else {
								undo_redo->add_do_method(EditorNode::get_singleton()->get_edited_scene(), "set_meta", "_edit_vertical_guides_", vguides);
							}
							undo_redo->add_undo_method(EditorNode::get_singleton()->get_edited_scene(), "set_meta", "_edit_vertical_guides_", prev_vguides);
							undo_redo->add_undo_method(viewport, "update");
							undo_redo->commit_action();
						}
					}
				} else if (drag_type == DRAG_H_GUIDE) {
					Array prev_hguides = hguides.duplicate();
					if (b->get_position().y > RULER_WIDTH) {
						// Adds a new horizontal guide
						if (dragged_guide_index >= 0) {
							hguides[dragged_guide_index] = edited.y;
							undo_redo->create_action(TTR("Move Horizontal Guide"));
							undo_redo->add_do_method(EditorNode::get_singleton()->get_edited_scene(), "set_meta", "_edit_horizontal_guides_", hguides);
							undo_redo->add_undo_method(EditorNode::get_singleton()->get_edited_scene(), "set_meta", "_edit_horizontal_guides_", prev_hguides);
							undo_redo->add_undo_method(viewport, "update");
							undo_redo->commit_action();
						} else {
							hguides.push_back(edited.y);
							undo_redo->create_action(TTR("Create Horizontal Guide"));
							undo_redo->add_do_method(EditorNode::get_singleton()->get_edited_scene(), "set_meta", "_edit_horizontal_guides_", hguides);
							undo_redo->add_undo_method(EditorNode::get_singleton()->get_edited_scene(), "set_meta", "_edit_horizontal_guides_", prev_hguides);
							undo_redo->add_undo_method(viewport, "update");
							undo_redo->commit_action();
						}
					} else {
						if (dragged_guide_index >= 0) {
							hguides.remove_at(dragged_guide_index);
							undo_redo->create_action(TTR("Remove Horizontal Guide"));
							if (hguides.is_empty()) {
								undo_redo->add_do_method(EditorNode::get_singleton()->get_edited_scene(), "remove_meta", "_edit_horizontal_guides_");
							} else {
								undo_redo->add_do_method(EditorNode::get_singleton()->get_edited_scene(), "set_meta", "_edit_horizontal_guides_", hguides);
							}
							undo_redo->add_undo_method(EditorNode::get_singleton()->get_edited_scene(), "set_meta", "_edit_horizontal_guides_", prev_hguides);
							undo_redo->add_undo_method(viewport, "update");
							undo_redo->commit_action();
						}
					}
				} else if (drag_type == DRAG_DOUBLE_GUIDE) {
					Array prev_hguides = hguides.duplicate();
					Array prev_vguides = vguides.duplicate();
					if (b->get_position().x > RULER_WIDTH && b->get_position().y > RULER_WIDTH) {
						// Adds a new horizontal guide a new vertical guide
						vguides.push_back(edited.x);
						hguides.push_back(edited.y);
						undo_redo->create_action(TTR("Create Horizontal and Vertical Guides"));
						undo_redo->add_do_method(EditorNode::get_singleton()->get_edited_scene(), "set_meta", "_edit_vertical_guides_", vguides);
						undo_redo->add_do_method(EditorNode::get_singleton()->get_edited_scene(), "set_meta", "_edit_horizontal_guides_", hguides);
						undo_redo->add_undo_method(EditorNode::get_singleton()->get_edited_scene(), "set_meta", "_edit_vertical_guides_", prev_vguides);
						undo_redo->add_undo_method(EditorNode::get_singleton()->get_edited_scene(), "set_meta", "_edit_horizontal_guides_", prev_hguides);
						undo_redo->add_undo_method(viewport, "update");
						undo_redo->commit_action();
					}
				}
			}
			drag_type = DRAG_NONE;
			viewport->update();
			return true;
		}
	}
	return false;
}

bool CanvasItemEditor::_gui_input_zoom_or_pan(const Ref<InputEvent> &p_event, bool p_already_accepted) {
	Ref<InputEventMouseButton> b = p_event;
	if (b.is_valid() && !p_already_accepted) {
		const bool pan_on_scroll = bool(EditorSettings::get_singleton()->get("editors/2d/scroll_to_pan")) && !b->is_ctrl_pressed();

		if (pan_on_scroll) {
			// Perform horizontal scrolling first so we can check for Shift being held.
			if (b->is_pressed() &&
					(b->get_button_index() == MouseButton::WHEEL_LEFT || (b->is_shift_pressed() && b->get_button_index() == MouseButton::WHEEL_UP))) {
				// Pan left
				view_offset.x -= int(EditorSettings::get_singleton()->get("editors/2d/pan_speed")) / zoom * b->get_factor();
				update_viewport();
				return true;
			}

			if (b->is_pressed() &&
					(b->get_button_index() == MouseButton::WHEEL_RIGHT || (b->is_shift_pressed() && b->get_button_index() == MouseButton::WHEEL_DOWN))) {
				// Pan right
				view_offset.x += int(EditorSettings::get_singleton()->get("editors/2d/pan_speed")) / zoom * b->get_factor();
				update_viewport();
				return true;
			}
		}

		if (b->is_pressed() && b->get_button_index() == MouseButton::WHEEL_DOWN) {
			// Scroll or pan down
			if (pan_on_scroll) {
				view_offset.y += int(EditorSettings::get_singleton()->get("editors/2d/pan_speed")) / zoom * b->get_factor();
				update_viewport();
			} else {
				zoom_widget->set_zoom_by_increments(-1, Input::get_singleton()->is_key_pressed(Key::ALT));
				if (!Math::is_equal_approx(b->get_factor(), 1.0f)) {
					// Handle high-precision (analog) scrolling.
					zoom_widget->set_zoom(zoom * ((zoom_widget->get_zoom() / zoom - 1.f) * b->get_factor() + 1.f));
				}
				_zoom_on_position(zoom_widget->get_zoom(), b->get_position());
			}
			return true;
		}

		if (b->is_pressed() && b->get_button_index() == MouseButton::WHEEL_UP) {
			// Scroll or pan up
			if (pan_on_scroll) {
				view_offset.y -= int(EditorSettings::get_singleton()->get("editors/2d/pan_speed")) / zoom * b->get_factor();
				update_viewport();
			} else {
				zoom_widget->set_zoom_by_increments(1, Input::get_singleton()->is_key_pressed(Key::ALT));
				if (!Math::is_equal_approx(b->get_factor(), 1.0f)) {
					// Handle high-precision (analog) scrolling.
					zoom_widget->set_zoom(zoom * ((zoom_widget->get_zoom() / zoom - 1.f) * b->get_factor() + 1.f));
				}
				_zoom_on_position(zoom_widget->get_zoom(), b->get_position());
			}
			return true;
		}

		if (!panning) {
			if (b->is_pressed() &&
					(b->get_button_index() == MouseButton::MIDDLE ||
							(b->get_button_index() == MouseButton::LEFT && tool == TOOL_PAN) ||
							(b->get_button_index() == MouseButton::LEFT && !EditorSettings::get_singleton()->get("editors/2d/simple_panning") && pan_pressed))) {
				// Pan the viewport
				panning = true;
			}
		}

		if (panning) {
			if (!b->is_pressed() && (pan_on_scroll || (b->get_button_index() != MouseButton::WHEEL_DOWN && b->get_button_index() != MouseButton::WHEEL_UP))) {
				// Stop panning the viewport (for any mouse button press except zooming)
				panning = false;
			}
		}
	}

	Ref<InputEventKey> k = p_event;
	if (k.is_valid()) {
		if (k->is_pressed()) {
			if (ED_GET_SHORTCUT("canvas_item_editor/zoom_3.125_percent")->matches_event(p_event)) {
				_update_zoom((1.0 / 32.0) * MAX(1, EDSCALE));
			} else if (ED_GET_SHORTCUT("canvas_item_editor/zoom_6.25_percent")->matches_event(p_event)) {
				_update_zoom((1.0 / 16.0) * MAX(1, EDSCALE));
			} else if (ED_GET_SHORTCUT("canvas_item_editor/zoom_12.5_percent")->matches_event(p_event)) {
				_update_zoom((1.0 / 8.0) * MAX(1, EDSCALE));
			} else if (ED_GET_SHORTCUT("canvas_item_editor/zoom_25_percent")->matches_event(p_event)) {
				_update_zoom((1.0 / 4.0) * MAX(1, EDSCALE));
			} else if (ED_GET_SHORTCUT("canvas_item_editor/zoom_50_percent")->matches_event(p_event)) {
				_update_zoom((1.0 / 2.0) * MAX(1, EDSCALE));
			} else if (ED_GET_SHORTCUT("canvas_item_editor/zoom_100_percent")->matches_event(p_event)) {
				_update_zoom(1.0 * MAX(1, EDSCALE));
			} else if (ED_GET_SHORTCUT("canvas_item_editor/zoom_200_percent")->matches_event(p_event)) {
				_update_zoom(2.0 * MAX(1, EDSCALE));
			} else if (ED_GET_SHORTCUT("canvas_item_editor/zoom_400_percent")->matches_event(p_event)) {
				_update_zoom(4.0 * MAX(1, EDSCALE));
			} else if (ED_GET_SHORTCUT("canvas_item_editor/zoom_800_percent")->matches_event(p_event)) {
				_update_zoom(8.0 * MAX(1, EDSCALE));
			} else if (ED_GET_SHORTCUT("canvas_item_editor/zoom_1600_percent")->matches_event(p_event)) {
				_update_zoom(16.0 * MAX(1, EDSCALE));
			}
		}

		bool is_pan_key = pan_view_shortcut.is_valid() && pan_view_shortcut->matches_event(p_event);

		if (is_pan_key && (EditorSettings::get_singleton()->get("editors/2d/simple_panning") || drag_type != DRAG_NONE)) {
			if (!panning) {
				if (k->is_pressed() && !k->is_echo()) {
					//Pan the viewport
					panning = true;
				}
			} else {
				if (!k->is_pressed()) {
					// Stop panning the viewport (for any mouse button press)
					panning = false;
				}
			}
		}

		if (is_pan_key && pan_pressed != k->is_pressed()) {
			pan_pressed = k->is_pressed();
			_update_cursor();
		}
	}

	Ref<InputEventMouseMotion> m = p_event;
	if (m.is_valid()) {
		if (panning) {
			// Pan the viewport
			Point2i relative;
			if (bool(EditorSettings::get_singleton()->get("editors/2d/warped_mouse_panning"))) {
				relative = Input::get_singleton()->warp_mouse_motion(m, viewport->get_global_rect());
			} else {
				relative = m->get_relative();
			}
			view_offset.x -= relative.x / zoom;
			view_offset.y -= relative.y / zoom;
			update_viewport();
			return true;
		}
	}

	Ref<InputEventMagnifyGesture> magnify_gesture = p_event;
	if (magnify_gesture.is_valid() && !p_already_accepted) {
		// Zoom gesture
		_zoom_on_position(zoom * magnify_gesture->get_factor(), magnify_gesture->get_position());
		return true;
	}

	Ref<InputEventPanGesture> pan_gesture = p_event;
	if (pan_gesture.is_valid() && !p_already_accepted) {
		// If ctrl key pressed, then zoom instead of pan.
		if (pan_gesture->is_ctrl_pressed()) {
			const real_t factor = pan_gesture->get_delta().y;

			zoom_widget->set_zoom_by_increments(1);
			if (factor != 1.f) {
				zoom_widget->set_zoom(zoom * ((zoom_widget->get_zoom() / zoom - 1.f) * factor + 1.f));
			}
			_zoom_on_position(zoom_widget->get_zoom(), pan_gesture->get_position());

			return true;
		}

		// Pan gesture
		const Vector2 delta = (int(EditorSettings::get_singleton()->get("editors/2d/pan_speed")) / zoom) * pan_gesture->get_delta();
		view_offset.x += delta.x;
		view_offset.y += delta.y;
		update_viewport();
		return true;
	}

	return false;
}

bool CanvasItemEditor::_gui_input_pivot(const Ref<InputEvent> &p_event) {
	Ref<InputEventMouseMotion> m = p_event;
	Ref<InputEventMouseButton> b = p_event;
	Ref<InputEventKey> k = p_event;

	// Drag the pivot (in pivot mode / with V key)
	if (drag_type == DRAG_NONE) {
		if ((b.is_valid() && b->is_pressed() && b->get_button_index() == MouseButton::LEFT && tool == TOOL_EDIT_PIVOT) ||
				(k.is_valid() && k->is_pressed() && !k->is_echo() && k->get_keycode() == Key::V && tool == TOOL_SELECT && k->get_modifiers_mask() == Key::NONE)) {
			List<CanvasItem *> selection = _get_edited_canvas_items();

			// Filters the selection with nodes that allow setting the pivot
			drag_selection = List<CanvasItem *>();
			for (CanvasItem *canvas_item : selection) {
				if (canvas_item->_edit_use_pivot()) {
					drag_selection.push_back(canvas_item);
				}
			}

			// Start dragging if we still have nodes
			if (drag_selection.size() > 0) {
				_save_canvas_item_state(drag_selection);
				drag_from = transform.affine_inverse().xform((b.is_valid()) ? b->get_position() : viewport->get_local_mouse_position());
				Vector2 new_pos;
				if (drag_selection.size() == 1) {
					new_pos = snap_point(drag_from, SNAP_NODE_SIDES | SNAP_NODE_CENTER | SNAP_NODE_ANCHORS | SNAP_OTHER_NODES | SNAP_GRID | SNAP_PIXEL, 0, drag_selection[0]);
				} else {
					new_pos = snap_point(drag_from, SNAP_OTHER_NODES | SNAP_GRID | SNAP_PIXEL, 0, nullptr, drag_selection);
				}
				for (CanvasItem *canvas_item : drag_selection) {
					canvas_item->_edit_set_pivot(canvas_item->get_global_transform_with_canvas().affine_inverse().xform(new_pos));
				}

				drag_type = DRAG_PIVOT;
			}
			return true;
		}
	}

	if (drag_type == DRAG_PIVOT) {
		// Move the pivot
		if (m.is_valid()) {
			drag_to = transform.affine_inverse().xform(m->get_position());
			_restore_canvas_item_state(drag_selection);
			Vector2 new_pos;
			if (drag_selection.size() == 1) {
				new_pos = snap_point(drag_to, SNAP_NODE_SIDES | SNAP_NODE_CENTER | SNAP_NODE_ANCHORS | SNAP_OTHER_NODES | SNAP_GRID | SNAP_PIXEL, 0, drag_selection[0]);
			} else {
				new_pos = snap_point(drag_to, SNAP_OTHER_NODES | SNAP_GRID | SNAP_PIXEL);
			}
			for (CanvasItem *canvas_item : drag_selection) {
				canvas_item->_edit_set_pivot(canvas_item->get_global_transform_with_canvas().affine_inverse().xform(new_pos));
			}
			return true;
		}

		// Confirm the pivot move
		if (drag_selection.size() >= 1 &&
				((b.is_valid() && !b->is_pressed() && b->get_button_index() == MouseButton::LEFT && tool == TOOL_EDIT_PIVOT) ||
						(k.is_valid() && !k->is_pressed() && k->get_keycode() == Key::V))) {
			_commit_canvas_item_state(
					drag_selection,
					vformat(
							TTR("Set CanvasItem \"%s\" Pivot Offset to (%d, %d)"),
							drag_selection[0]->get_name(),
							drag_selection[0]->_edit_get_pivot().x,
							drag_selection[0]->_edit_get_pivot().y));
			drag_type = DRAG_NONE;
			return true;
		}

		// Cancel a drag
		if (b.is_valid() && b->get_button_index() == MouseButton::RIGHT && b->is_pressed()) {
			_restore_canvas_item_state(drag_selection);
			drag_type = DRAG_NONE;
			viewport->update();
			return true;
		}
	}
	return false;
}

bool CanvasItemEditor::_gui_input_rotate(const Ref<InputEvent> &p_event) {
	Ref<InputEventMouseButton> b = p_event;
	Ref<InputEventMouseMotion> m = p_event;

	// Start rotation
	if (drag_type == DRAG_NONE) {
		if (b.is_valid() && b->get_button_index() == MouseButton::LEFT && b->is_pressed()) {
			if ((b->is_command_pressed() && !b->is_alt_pressed() && tool == TOOL_SELECT) || tool == TOOL_ROTATE) {
				List<CanvasItem *> selection = _get_edited_canvas_items();

				// Remove not movable nodes
				for (CanvasItem *E : selection) {
					if (!_is_node_movable(E, true)) {
						selection.erase(E);
					}
				}

				drag_selection = selection;
				if (drag_selection.size() > 0) {
					drag_type = DRAG_ROTATE;
					drag_from = transform.affine_inverse().xform(b->get_position());
					CanvasItem *canvas_item = drag_selection[0];
					if (canvas_item->_edit_use_pivot()) {
						drag_rotation_center = canvas_item->get_global_transform_with_canvas().xform(canvas_item->_edit_get_pivot());
					} else {
						drag_rotation_center = canvas_item->get_global_transform_with_canvas().get_origin();
					}
					_save_canvas_item_state(drag_selection);
					return true;
				}
			}
		}
	}

	if (drag_type == DRAG_ROTATE) {
		// Rotate the node
		if (m.is_valid()) {
			_restore_canvas_item_state(drag_selection);
			for (CanvasItem *canvas_item : drag_selection) {
				drag_to = transform.affine_inverse().xform(m->get_position());
				//Rotate the opposite way if the canvas item's compounded scale has an uneven number of negative elements
				bool opposite = (canvas_item->get_global_transform().get_scale().sign().dot(canvas_item->get_transform().get_scale().sign()) == 0);
				canvas_item->_edit_set_rotation(snap_angle(canvas_item->_edit_get_rotation() + (opposite ? -1 : 1) * (drag_from - drag_rotation_center).angle_to(drag_to - drag_rotation_center), canvas_item->_edit_get_rotation()));
				viewport->update();
			}
			return true;
		}

		// Confirms the node rotation
		if (b.is_valid() && b->get_button_index() == MouseButton::LEFT && !b->is_pressed()) {
			if (drag_selection.size() != 1) {
				_commit_canvas_item_state(
						drag_selection,
						vformat(TTR("Rotate %d CanvasItems"), drag_selection.size()),
						true);
			} else {
				_commit_canvas_item_state(
						drag_selection,
						vformat(TTR("Rotate CanvasItem \"%s\" to %d degrees"),
								drag_selection[0]->get_name(),
								Math::rad2deg(drag_selection[0]->_edit_get_rotation())),
						true);
			}

			if (key_auto_insert_button->is_pressed()) {
				_insert_animation_keys(false, true, false, true);
			}

			drag_type = DRAG_NONE;
			return true;
		}

		// Cancel a drag
		if (b.is_valid() && b->get_button_index() == MouseButton::RIGHT && b->is_pressed()) {
			_restore_canvas_item_state(drag_selection);
			drag_type = DRAG_NONE;
			viewport->update();
			return true;
		}
	}
	return false;
}

bool CanvasItemEditor::_gui_input_open_scene_on_double_click(const Ref<InputEvent> &p_event) {
	Ref<InputEventMouseButton> b = p_event;

	// Open a sub-scene on double-click
	if (b.is_valid() && b->get_button_index() == MouseButton::LEFT && b->is_pressed() && b->is_double_click() && tool == TOOL_SELECT) {
		List<CanvasItem *> selection = _get_edited_canvas_items();
		if (selection.size() == 1) {
			CanvasItem *canvas_item = selection[0];
			if (canvas_item->get_scene_file_path() != "" && canvas_item != editor->get_edited_scene()) {
				editor->open_request(canvas_item->get_scene_file_path());
				return true;
			}
		}
	}
	return false;
}

bool CanvasItemEditor::_gui_input_anchors(const Ref<InputEvent> &p_event) {
	Ref<InputEventMouseButton> b = p_event;
	Ref<InputEventMouseMotion> m = p_event;

	// Starts anchor dragging if needed
	if (drag_type == DRAG_NONE) {
		if (b.is_valid() && b->get_button_index() == MouseButton::LEFT && b->is_pressed() && tool == TOOL_SELECT) {
			List<CanvasItem *> selection = _get_edited_canvas_items();
			if (selection.size() == 1) {
				Control *control = Object::cast_to<Control>(selection[0]);
				if (control && _is_node_movable(control)) {
					Vector2 anchor_pos[4];
					anchor_pos[0] = Vector2(control->get_anchor(SIDE_LEFT), control->get_anchor(SIDE_TOP));
					anchor_pos[1] = Vector2(control->get_anchor(SIDE_RIGHT), control->get_anchor(SIDE_TOP));
					anchor_pos[2] = Vector2(control->get_anchor(SIDE_RIGHT), control->get_anchor(SIDE_BOTTOM));
					anchor_pos[3] = Vector2(control->get_anchor(SIDE_LEFT), control->get_anchor(SIDE_BOTTOM));

					Rect2 anchor_rects[4];
					for (int i = 0; i < 4; i++) {
						anchor_pos[i] = (transform * control->get_global_transform_with_canvas()).xform(_anchor_to_position(control, anchor_pos[i]));
						anchor_rects[i] = Rect2(anchor_pos[i], anchor_handle->get_size());
						if (control->is_layout_rtl()) {
							anchor_rects[i].position -= anchor_handle->get_size() * Vector2(real_t(i == 1 || i == 2), real_t(i <= 1));
						} else {
							anchor_rects[i].position -= anchor_handle->get_size() * Vector2(real_t(i == 0 || i == 3), real_t(i <= 1));
						}
					}

					DragType dragger[] = {
						DRAG_ANCHOR_TOP_LEFT,
						DRAG_ANCHOR_TOP_RIGHT,
						DRAG_ANCHOR_BOTTOM_RIGHT,
						DRAG_ANCHOR_BOTTOM_LEFT,
					};

					for (int i = 0; i < 4; i++) {
						if (anchor_rects[i].has_point(b->get_position())) {
							if ((anchor_pos[0] == anchor_pos[2]) && (anchor_pos[0].distance_to(b->get_position()) < anchor_handle->get_size().length() / 3.0)) {
								drag_type = DRAG_ANCHOR_ALL;
							} else {
								drag_type = dragger[i];
							}
							drag_from = transform.affine_inverse().xform(b->get_position());
							drag_selection = List<CanvasItem *>();
							drag_selection.push_back(control);
							_save_canvas_item_state(drag_selection);
							return true;
						}
					}
				}
			}
		}
	}

	if (drag_type == DRAG_ANCHOR_TOP_LEFT || drag_type == DRAG_ANCHOR_TOP_RIGHT || drag_type == DRAG_ANCHOR_BOTTOM_RIGHT || drag_type == DRAG_ANCHOR_BOTTOM_LEFT || drag_type == DRAG_ANCHOR_ALL) {
		// Drag the anchor
		if (m.is_valid()) {
			_restore_canvas_item_state(drag_selection);
			Control *control = Object::cast_to<Control>(drag_selection[0]);

			drag_to = transform.affine_inverse().xform(m->get_position());

			Transform2D xform = control->get_global_transform_with_canvas().affine_inverse();

			Point2 previous_anchor;
			previous_anchor.x = (drag_type == DRAG_ANCHOR_TOP_LEFT || drag_type == DRAG_ANCHOR_BOTTOM_LEFT) ? control->get_anchor(SIDE_LEFT) : control->get_anchor(SIDE_RIGHT);
			previous_anchor.y = (drag_type == DRAG_ANCHOR_TOP_LEFT || drag_type == DRAG_ANCHOR_TOP_RIGHT) ? control->get_anchor(SIDE_TOP) : control->get_anchor(SIDE_BOTTOM);
			previous_anchor = xform.affine_inverse().xform(_anchor_to_position(control, previous_anchor));

			Vector2 new_anchor = xform.xform(snap_point(previous_anchor + (drag_to - drag_from), SNAP_GRID | SNAP_OTHER_NODES, SNAP_NODE_PARENT | SNAP_NODE_SIDES | SNAP_NODE_CENTER, control));
			new_anchor = _position_to_anchor(control, new_anchor).snapped(Vector2(0.001, 0.001));

			bool use_single_axis = m->is_shift_pressed();
			Vector2 drag_vector = xform.xform(drag_to) - xform.xform(drag_from);
			bool use_y = Math::abs(drag_vector.y) > Math::abs(drag_vector.x);

			switch (drag_type) {
				case DRAG_ANCHOR_TOP_LEFT:
					if (!use_single_axis || !use_y) {
						control->set_anchor(SIDE_LEFT, new_anchor.x, false, false);
					}
					if (!use_single_axis || use_y) {
						control->set_anchor(SIDE_TOP, new_anchor.y, false, false);
					}
					break;
				case DRAG_ANCHOR_TOP_RIGHT:
					if (!use_single_axis || !use_y) {
						control->set_anchor(SIDE_RIGHT, new_anchor.x, false, false);
					}
					if (!use_single_axis || use_y) {
						control->set_anchor(SIDE_TOP, new_anchor.y, false, false);
					}
					break;
				case DRAG_ANCHOR_BOTTOM_RIGHT:
					if (!use_single_axis || !use_y) {
						control->set_anchor(SIDE_RIGHT, new_anchor.x, false, false);
					}
					if (!use_single_axis || use_y) {
						control->set_anchor(SIDE_BOTTOM, new_anchor.y, false, false);
					}
					break;
				case DRAG_ANCHOR_BOTTOM_LEFT:
					if (!use_single_axis || !use_y) {
						control->set_anchor(SIDE_LEFT, new_anchor.x, false, false);
					}
					if (!use_single_axis || use_y) {
						control->set_anchor(SIDE_BOTTOM, new_anchor.y, false, false);
					}
					break;
				case DRAG_ANCHOR_ALL:
					if (!use_single_axis || !use_y) {
						control->set_anchor(SIDE_LEFT, new_anchor.x, false, true);
						control->set_anchor(SIDE_RIGHT, new_anchor.x, false, true);
					}
					if (!use_single_axis || use_y) {
						control->set_anchor(SIDE_TOP, new_anchor.y, false, true);
						control->set_anchor(SIDE_BOTTOM, new_anchor.y, false, true);
					}
					break;
				default:
					break;
			}
			return true;
		}

		// Confirms new anchor position
		if (drag_selection.size() >= 1 && b.is_valid() && b->get_button_index() == MouseButton::LEFT && !b->is_pressed()) {
			_commit_canvas_item_state(
					drag_selection,
					vformat(TTR("Move CanvasItem \"%s\" Anchor"), drag_selection[0]->get_name()));
			drag_type = DRAG_NONE;
			return true;
		}

		// Cancel a drag
		if (b.is_valid() && b->get_button_index() == MouseButton::RIGHT && b->is_pressed()) {
			_restore_canvas_item_state(drag_selection);
			drag_type = DRAG_NONE;
			viewport->update();
			return true;
		}
	}
	return false;
}

bool CanvasItemEditor::_gui_input_resize(const Ref<InputEvent> &p_event) {
	Ref<InputEventMouseButton> b = p_event;
	Ref<InputEventMouseMotion> m = p_event;

	// Drag resize handles
	if (drag_type == DRAG_NONE) {
		if (b.is_valid() && b->get_button_index() == MouseButton::LEFT && b->is_pressed() && tool == TOOL_SELECT) {
			List<CanvasItem *> selection = _get_edited_canvas_items();
			if (selection.size() == 1) {
				CanvasItem *canvas_item = selection[0];
				if (canvas_item->_edit_use_rect() && _is_node_movable(canvas_item)) {
					Rect2 rect = canvas_item->_edit_get_rect();
					Transform2D xform = transform * canvas_item->get_global_transform_with_canvas();

					Vector2 endpoints[4] = {
						xform.xform(rect.position),
						xform.xform(rect.position + Vector2(rect.size.x, 0)),
						xform.xform(rect.position + rect.size),
						xform.xform(rect.position + Vector2(0, rect.size.y))
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

					DragType resize_drag = DRAG_NONE;
					real_t radius = (select_handle->get_size().width / 2) * 1.5;

					for (int i = 0; i < 4; i++) {
						int prev = (i + 3) % 4;
						int next = (i + 1) % 4;

						Vector2 ofs = ((endpoints[i] - endpoints[prev]).normalized() + ((endpoints[i] - endpoints[next]).normalized())).normalized();
						ofs *= (select_handle->get_size().width / 2);
						ofs += endpoints[i];
						if (ofs.distance_to(b->get_position()) < radius) {
							resize_drag = dragger[i * 2];
						}

						ofs = (endpoints[i] + endpoints[next]) / 2;
						ofs += (endpoints[next] - endpoints[i]).orthogonal().normalized() * (select_handle->get_size().width / 2);
						if (ofs.distance_to(b->get_position()) < radius) {
							resize_drag = dragger[i * 2 + 1];
						}
					}

					if (resize_drag != DRAG_NONE) {
						drag_type = resize_drag;
						drag_from = transform.affine_inverse().xform(b->get_position());
						drag_selection = List<CanvasItem *>();
						drag_selection.push_back(canvas_item);
						_save_canvas_item_state(drag_selection);
						return true;
					}
				}
			}
		}
	}

	if (drag_type == DRAG_LEFT || drag_type == DRAG_RIGHT || drag_type == DRAG_TOP || drag_type == DRAG_BOTTOM ||
			drag_type == DRAG_TOP_LEFT || drag_type == DRAG_TOP_RIGHT || drag_type == DRAG_BOTTOM_LEFT || drag_type == DRAG_BOTTOM_RIGHT) {
		// Resize the node
		if (m.is_valid()) {
			CanvasItem *canvas_item = drag_selection[0];
			CanvasItemEditorSelectedItem *se = editor_selection->get_node_editor_data<CanvasItemEditorSelectedItem>(canvas_item);
			//Reset state
			canvas_item->_edit_set_state(se->undo_state);

			bool uniform = m->is_shift_pressed();
			bool symmetric = m->is_alt_pressed();

			Rect2 local_rect = canvas_item->_edit_get_rect();
			real_t aspect = local_rect.get_size().y / local_rect.get_size().x;
			Point2 current_begin = local_rect.get_position();
			Point2 current_end = local_rect.get_position() + local_rect.get_size();
			Point2 max_begin = (symmetric) ? (current_begin + current_end - canvas_item->_edit_get_minimum_size()) / 2.0 : current_end - canvas_item->_edit_get_minimum_size();
			Point2 min_end = (symmetric) ? (current_begin + current_end + canvas_item->_edit_get_minimum_size()) / 2.0 : current_begin + canvas_item->_edit_get_minimum_size();
			Point2 center = (current_begin + current_end) / 2.0;

			drag_to = transform.affine_inverse().xform(m->get_position());

			Transform2D xform = canvas_item->get_global_transform_with_canvas().affine_inverse();

			Point2 drag_to_snapped_begin;
			Point2 drag_to_snapped_end;

			// last call decides which snapping lines are drawn
			if (drag_type == DRAG_LEFT || drag_type == DRAG_TOP || drag_type == DRAG_TOP_LEFT) {
				drag_to_snapped_end = snap_point(xform.affine_inverse().xform(current_end) + (drag_to - drag_from), SNAP_NODE_ANCHORS | SNAP_NODE_PARENT | SNAP_OTHER_NODES | SNAP_GRID | SNAP_PIXEL, 0, canvas_item);
				drag_to_snapped_begin = snap_point(xform.affine_inverse().xform(current_begin) + (drag_to - drag_from), SNAP_NODE_ANCHORS | SNAP_NODE_PARENT | SNAP_OTHER_NODES | SNAP_GRID | SNAP_PIXEL, 0, canvas_item);
			} else {
				drag_to_snapped_begin = snap_point(xform.affine_inverse().xform(current_begin) + (drag_to - drag_from), SNAP_NODE_ANCHORS | SNAP_NODE_PARENT | SNAP_OTHER_NODES | SNAP_GRID | SNAP_PIXEL, 0, canvas_item);
				drag_to_snapped_end = snap_point(xform.affine_inverse().xform(current_end) + (drag_to - drag_from), SNAP_NODE_ANCHORS | SNAP_NODE_PARENT | SNAP_OTHER_NODES | SNAP_GRID | SNAP_PIXEL, 0, canvas_item);
			}

			Point2 drag_begin = xform.xform(drag_to_snapped_begin);
			Point2 drag_end = xform.xform(drag_to_snapped_end);

			// Horizontal resize
			if (drag_type == DRAG_LEFT || drag_type == DRAG_TOP_LEFT || drag_type == DRAG_BOTTOM_LEFT) {
				current_begin.x = MIN(drag_begin.x, max_begin.x);
			} else if (drag_type == DRAG_RIGHT || drag_type == DRAG_TOP_RIGHT || drag_type == DRAG_BOTTOM_RIGHT) {
				current_end.x = MAX(drag_end.x, min_end.x);
			}

			// Vertical resize
			if (drag_type == DRAG_TOP || drag_type == DRAG_TOP_LEFT || drag_type == DRAG_TOP_RIGHT) {
				current_begin.y = MIN(drag_begin.y, max_begin.y);
			} else if (drag_type == DRAG_BOTTOM || drag_type == DRAG_BOTTOM_LEFT || drag_type == DRAG_BOTTOM_RIGHT) {
				current_end.y = MAX(drag_end.y, min_end.y);
			}

			// Uniform resize
			if (uniform) {
				if (drag_type == DRAG_LEFT || drag_type == DRAG_RIGHT) {
					current_end.y = current_begin.y + aspect * (current_end.x - current_begin.x);
				} else if (drag_type == DRAG_TOP || drag_type == DRAG_BOTTOM) {
					current_end.x = current_begin.x + (current_end.y - current_begin.y) / aspect;
				} else {
					if (aspect >= 1.0) {
						if (drag_type == DRAG_TOP_LEFT || drag_type == DRAG_TOP_RIGHT) {
							current_begin.y = current_end.y - aspect * (current_end.x - current_begin.x);
						} else {
							current_end.y = current_begin.y + aspect * (current_end.x - current_begin.x);
						}
					} else {
						if (drag_type == DRAG_TOP_LEFT || drag_type == DRAG_BOTTOM_LEFT) {
							current_begin.x = current_end.x - (current_end.y - current_begin.y) / aspect;
						} else {
							current_end.x = current_begin.x + (current_end.y - current_begin.y) / aspect;
						}
					}
				}
			}

			// Symmetric resize
			if (symmetric) {
				if (drag_type == DRAG_LEFT || drag_type == DRAG_TOP_LEFT || drag_type == DRAG_BOTTOM_LEFT) {
					current_end.x = 2.0 * center.x - current_begin.x;
				} else if (drag_type == DRAG_RIGHT || drag_type == DRAG_TOP_RIGHT || drag_type == DRAG_BOTTOM_RIGHT) {
					current_begin.x = 2.0 * center.x - current_end.x;
				}
				if (drag_type == DRAG_TOP || drag_type == DRAG_TOP_LEFT || drag_type == DRAG_TOP_RIGHT) {
					current_end.y = 2.0 * center.y - current_begin.y;
				} else if (drag_type == DRAG_BOTTOM || drag_type == DRAG_BOTTOM_LEFT || drag_type == DRAG_BOTTOM_RIGHT) {
					current_begin.y = 2.0 * center.y - current_end.y;
				}
			}
			canvas_item->_edit_set_rect(Rect2(current_begin, current_end - current_begin));
			return true;
		}

		// Confirm resize
		if (drag_selection.size() >= 1 && b.is_valid() && b->get_button_index() == MouseButton::LEFT && !b->is_pressed()) {
			const Node2D *node2d = Object::cast_to<Node2D>(drag_selection[0]);
			if (node2d) {
				// Extends from Node2D.
				// Node2D doesn't have an actual stored rect size, unlike Controls.
				_commit_canvas_item_state(
						drag_selection,
						vformat(
								TTR("Scale Node2D \"%s\" to (%s, %s)"),
								drag_selection[0]->get_name(),
								Math::snapped(drag_selection[0]->_edit_get_scale().x, 0.01),
								Math::snapped(drag_selection[0]->_edit_get_scale().y, 0.01)),
						true);
			} else {
				// Extends from Control.
				_commit_canvas_item_state(
						drag_selection,
						vformat(
								TTR("Resize Control \"%s\" to (%d, %d)"),
								drag_selection[0]->get_name(),
								drag_selection[0]->_edit_get_rect().size.x,
								drag_selection[0]->_edit_get_rect().size.y),
						true);
			}

			if (key_auto_insert_button->is_pressed()) {
				_insert_animation_keys(false, false, true, true);
			}

			snap_target[0] = SNAP_TARGET_NONE;
			snap_target[1] = SNAP_TARGET_NONE;
			drag_type = DRAG_NONE;
			viewport->update();
			return true;
		}

		// Cancel a drag
		if (b.is_valid() && b->get_button_index() == MouseButton::RIGHT && b->is_pressed()) {
			_restore_canvas_item_state(drag_selection);
			snap_target[0] = SNAP_TARGET_NONE;
			snap_target[1] = SNAP_TARGET_NONE;
			drag_type = DRAG_NONE;
			viewport->update();
			return true;
		}
	}
	return false;
}

bool CanvasItemEditor::_gui_input_scale(const Ref<InputEvent> &p_event) {
	Ref<InputEventMouseButton> b = p_event;
	Ref<InputEventMouseMotion> m = p_event;

	// Drag resize handles
	if (drag_type == DRAG_NONE) {
		if (b.is_valid() && b->get_button_index() == MouseButton::LEFT && b->is_pressed() && ((b->is_alt_pressed() && b->is_ctrl_pressed()) || tool == TOOL_SCALE)) {
			List<CanvasItem *> selection = _get_edited_canvas_items();
			if (selection.size() == 1) {
				CanvasItem *canvas_item = selection[0];

				if (_is_node_movable(canvas_item)) {
					Transform2D xform = transform * canvas_item->get_global_transform_with_canvas();
					Transform2D unscaled_transform = (xform * canvas_item->get_transform().affine_inverse() * canvas_item->_edit_get_transform()).orthonormalized();
					Transform2D simple_xform = viewport->get_transform() * unscaled_transform;

					drag_type = DRAG_SCALE_BOTH;

					if (show_transformation_gizmos) {
						Size2 scale_factor = Size2(SCALE_HANDLE_DISTANCE, SCALE_HANDLE_DISTANCE);
						Rect2 x_handle_rect = Rect2(scale_factor.x * EDSCALE, -5 * EDSCALE, 10 * EDSCALE, 10 * EDSCALE);
						if (x_handle_rect.has_point(simple_xform.affine_inverse().xform(b->get_position()))) {
							drag_type = DRAG_SCALE_X;
						}
						Rect2 y_handle_rect = Rect2(-5 * EDSCALE, scale_factor.y * EDSCALE, 10 * EDSCALE, 10 * EDSCALE);
						if (y_handle_rect.has_point(simple_xform.affine_inverse().xform(b->get_position()))) {
							drag_type = DRAG_SCALE_Y;
						}
					}

					drag_from = transform.affine_inverse().xform(b->get_position());
					drag_selection = List<CanvasItem *>();
					drag_selection.push_back(canvas_item);
					_save_canvas_item_state(drag_selection);
					return true;
				}
			}
		}
	}

	if (drag_type == DRAG_SCALE_BOTH || drag_type == DRAG_SCALE_X || drag_type == DRAG_SCALE_Y) {
		// Resize the node
		if (m.is_valid()) {
			_restore_canvas_item_state(drag_selection);
			CanvasItem *canvas_item = drag_selection[0];

			drag_to = transform.affine_inverse().xform(m->get_position());

			Transform2D parent_xform = canvas_item->get_global_transform_with_canvas() * canvas_item->get_transform().affine_inverse();
			Transform2D unscaled_transform = (transform * parent_xform * canvas_item->_edit_get_transform()).orthonormalized();
			Transform2D simple_xform = (viewport->get_transform() * unscaled_transform).affine_inverse() * transform;

			bool uniform = m->is_shift_pressed();
			bool is_ctrl = Input::get_singleton()->is_key_pressed(Key::CTRL);

			Point2 drag_from_local = simple_xform.xform(drag_from);
			Point2 drag_to_local = simple_xform.xform(drag_to);
			Point2 offset = drag_to_local - drag_from_local;

			Size2 scale = canvas_item->call("get_scale");
			Size2 original_scale = scale;
			real_t ratio = scale.y / scale.x;
			if (drag_type == DRAG_SCALE_BOTH) {
				Size2 scale_factor = drag_to_local / drag_from_local;
				if (uniform) {
					scale *= (scale_factor.x + scale_factor.y) / 2.0;
				} else {
					scale *= scale_factor;
				}
			} else {
				Size2 scale_factor = Vector2(offset.x, -offset.y) / SCALE_HANDLE_DISTANCE;
				Size2 parent_scale = parent_xform.get_scale();
				scale_factor *= Vector2(1.0 / parent_scale.x, 1.0 / parent_scale.y);

				if (drag_type == DRAG_SCALE_X) {
					scale.x += scale_factor.x;
					if (uniform) {
						scale.y = scale.x * ratio;
					}
				} else if (drag_type == DRAG_SCALE_Y) {
					scale.y -= scale_factor.y;
					if (uniform) {
						scale.x = scale.y / ratio;
					}
				}
			}

			if (snap_scale && !is_ctrl) {
				if (snap_relative) {
					scale.x = original_scale.x * (roundf((scale.x / original_scale.x) / snap_scale_step) * snap_scale_step);
					scale.y = original_scale.y * (roundf((scale.y / original_scale.y) / snap_scale_step) * snap_scale_step);
				} else {
					scale.x = roundf(scale.x / snap_scale_step) * snap_scale_step;
					scale.y = roundf(scale.y / snap_scale_step) * snap_scale_step;
				}
			}

			canvas_item->call("set_scale", scale);
			return true;
		}

		// Confirm resize
		if (b.is_valid() && b->get_button_index() == MouseButton::LEFT && !b->is_pressed()) {
			if (drag_selection.size() != 1) {
				_commit_canvas_item_state(
						drag_selection,
						vformat(TTR("Scale %d CanvasItems"), drag_selection.size()),
						true);
			} else {
				_commit_canvas_item_state(
						drag_selection,
						vformat(TTR("Scale CanvasItem \"%s\" to (%s, %s)"),
								drag_selection[0]->get_name(),
								Math::snapped(drag_selection[0]->_edit_get_scale().x, 0.01),
								Math::snapped(drag_selection[0]->_edit_get_scale().y, 0.01)),
						true);
			}
			if (key_auto_insert_button->is_pressed()) {
				_insert_animation_keys(false, false, true, true);
			}

			drag_type = DRAG_NONE;
			viewport->update();
			return true;
		}

		// Cancel a drag
		if (b.is_valid() && b->get_button_index() == MouseButton::RIGHT && b->is_pressed()) {
			_restore_canvas_item_state(drag_selection);
			drag_type = DRAG_NONE;
			viewport->update();
			return true;
		}
	}
	return false;
}

bool CanvasItemEditor::_gui_input_move(const Ref<InputEvent> &p_event) {
	Ref<InputEventMouseButton> b = p_event;
	Ref<InputEventMouseMotion> m = p_event;
	Ref<InputEventKey> k = p_event;

	if (drag_type == DRAG_NONE) {
		//Start moving the nodes
		if (b.is_valid() && b->get_button_index() == MouseButton::LEFT && b->is_pressed()) {
			if ((b->is_alt_pressed() && !b->is_ctrl_pressed()) || tool == TOOL_MOVE) {
				List<CanvasItem *> selection = _get_edited_canvas_items();

				drag_selection.clear();
				for (int i = 0; i < selection.size(); i++) {
					if (_is_node_movable(selection[i], true)) {
						drag_selection.push_back(selection[i]);
					}
				}

				if (selection.size() > 0) {
					drag_type = DRAG_MOVE;

					CanvasItem *canvas_item = selection[0];
					Transform2D parent_xform = canvas_item->get_global_transform_with_canvas() * canvas_item->get_transform().affine_inverse();
					Transform2D unscaled_transform = (transform * parent_xform * canvas_item->_edit_get_transform()).orthonormalized();
					Transform2D simple_xform = viewport->get_transform() * unscaled_transform;

					if (show_transformation_gizmos) {
						Size2 move_factor = Size2(MOVE_HANDLE_DISTANCE, MOVE_HANDLE_DISTANCE);
						Rect2 x_handle_rect = Rect2(move_factor.x * EDSCALE, -5 * EDSCALE, 10 * EDSCALE, 10 * EDSCALE);
						if (x_handle_rect.has_point(simple_xform.affine_inverse().xform(b->get_position()))) {
							drag_type = DRAG_MOVE_X;
						}
						Rect2 y_handle_rect = Rect2(-5 * EDSCALE, move_factor.y * EDSCALE, 10 * EDSCALE, 10 * EDSCALE);
						if (y_handle_rect.has_point(simple_xform.affine_inverse().xform(b->get_position()))) {
							drag_type = DRAG_MOVE_Y;
						}
					}

					drag_from = transform.affine_inverse().xform(b->get_position());
					_save_canvas_item_state(drag_selection);
				}
				return true;
			}
		}
	}

	if (drag_type == DRAG_MOVE || drag_type == DRAG_MOVE_X || drag_type == DRAG_MOVE_Y) {
		// Move the nodes
		if (m.is_valid()) {
			_restore_canvas_item_state(drag_selection, true);

			drag_to = transform.affine_inverse().xform(m->get_position());
			Point2 previous_pos;
			if (!drag_selection.is_empty()) {
				if (drag_selection.size() == 1) {
					Transform2D xform = drag_selection[0]->get_global_transform_with_canvas() * drag_selection[0]->get_transform().affine_inverse();
					previous_pos = xform.xform(drag_selection[0]->_edit_get_position());
				} else {
					previous_pos = _get_encompassing_rect_from_list(drag_selection).position;
				}
			}

			Point2 new_pos = snap_point(previous_pos + (drag_to - drag_from), SNAP_GRID | SNAP_GUIDES | SNAP_PIXEL | SNAP_NODE_PARENT | SNAP_NODE_ANCHORS | SNAP_OTHER_NODES, 0, nullptr, drag_selection);

			if (drag_type == DRAG_MOVE_X) {
				new_pos.y = previous_pos.y;
			} else if (drag_type == DRAG_MOVE_Y) {
				new_pos.x = previous_pos.x;
			}

			bool single_axis = m->is_shift_pressed();
			if (single_axis) {
				if (ABS(new_pos.x - previous_pos.x) > ABS(new_pos.y - previous_pos.y)) {
					new_pos.y = previous_pos.y;
				} else {
					new_pos.x = previous_pos.x;
				}
			}

			int index = 0;
			for (CanvasItem *canvas_item : drag_selection) {
				Transform2D xform = canvas_item->get_global_transform_with_canvas().affine_inverse() * canvas_item->get_transform();

				canvas_item->_edit_set_position(canvas_item->_edit_get_position() + xform.xform(new_pos) - xform.xform(previous_pos));
				index++;
			}
			return true;
		}

		// Confirm the move (only if it was moved)
		if (b.is_valid() && !b->is_pressed() && b->get_button_index() == MouseButton::LEFT) {
			if (transform.affine_inverse().xform(b->get_position()) != drag_from) {
				if (drag_selection.size() != 1) {
					_commit_canvas_item_state(
							drag_selection,
							vformat(TTR("Move %d CanvasItems"), drag_selection.size()),
							true);
				} else {
					_commit_canvas_item_state(
							drag_selection,
							vformat(
									TTR("Move CanvasItem \"%s\" to (%d, %d)"),
									drag_selection[0]->get_name(),
									drag_selection[0]->_edit_get_position().x,
									drag_selection[0]->_edit_get_position().y),
							true);
				}
			}

			if (key_auto_insert_button->is_pressed()) {
				_insert_animation_keys(true, false, false, true);
			}

			//Make sure smart snapping lines disappear.
			snap_target[0] = SNAP_TARGET_NONE;
			snap_target[1] = SNAP_TARGET_NONE;

			drag_type = DRAG_NONE;
			viewport->update();
			return true;
		}

		// Cancel a drag
		if (b.is_valid() && b->get_button_index() == MouseButton::RIGHT && b->is_pressed()) {
			_restore_canvas_item_state(drag_selection, true);
			snap_target[0] = SNAP_TARGET_NONE;
			snap_target[1] = SNAP_TARGET_NONE;
			drag_type = DRAG_NONE;
			viewport->update();
			return true;
		}
	}

	// Move the canvas items with the arrow keys
	if (k.is_valid() && k->is_pressed() && (tool == TOOL_SELECT || tool == TOOL_MOVE) &&
			(k->get_keycode() == Key::UP || k->get_keycode() == Key::DOWN || k->get_keycode() == Key::LEFT || k->get_keycode() == Key::RIGHT)) {
		if (!k->is_echo()) {
			// Start moving the canvas items with the keyboard
			drag_selection = _get_edited_canvas_items();
			drag_type = DRAG_KEY_MOVE;
			drag_from = Vector2();
			drag_to = Vector2();
			_save_canvas_item_state(drag_selection, true);
		}

		if (drag_selection.size() > 0) {
			_restore_canvas_item_state(drag_selection, true);

			bool move_local_base = k->is_alt_pressed();
			bool move_local_base_rotated = k->is_ctrl_pressed() || k->is_meta_pressed();

			Vector2 dir;
			if (k->get_keycode() == Key::UP) {
				dir += Vector2(0, -1);
			} else if (k->get_keycode() == Key::DOWN) {
				dir += Vector2(0, 1);
			} else if (k->get_keycode() == Key::LEFT) {
				dir += Vector2(-1, 0);
			} else if (k->get_keycode() == Key::RIGHT) {
				dir += Vector2(1, 0);
			}
			if (k->is_shift_pressed()) {
				dir *= grid_step * Math::pow(2.0, grid_step_multiplier);
			}

			drag_to += dir;
			if (k->is_shift_pressed()) {
				drag_to = drag_to.snapped(grid_step * Math::pow(2.0, grid_step_multiplier));
			}

			Point2 previous_pos;
			if (drag_selection.size() == 1) {
				Transform2D xform = drag_selection[0]->get_global_transform_with_canvas() * drag_selection[0]->get_transform().affine_inverse();
				previous_pos = xform.xform(drag_selection[0]->_edit_get_position());
			} else {
				previous_pos = _get_encompassing_rect_from_list(drag_selection).position;
			}

			Point2 new_pos;
			if (drag_selection.size() == 1) {
				Node2D *node_2d = Object::cast_to<Node2D>(drag_selection[0]);
				if (node_2d && move_local_base_rotated) {
					Transform2D m2;
					m2.rotate(node_2d->get_rotation());
					new_pos += m2.xform(drag_to);
				} else if (move_local_base) {
					new_pos += drag_to;
				} else {
					new_pos = previous_pos + (drag_to - drag_from);
				}
			} else {
				new_pos = previous_pos + (drag_to - drag_from);
			}

			int index = 0;
			for (CanvasItem *canvas_item : drag_selection) {
				Transform2D xform = canvas_item->get_global_transform_with_canvas().affine_inverse() * canvas_item->get_transform();

				canvas_item->_edit_set_position(canvas_item->_edit_get_position() + xform.xform(new_pos) - xform.xform(previous_pos));
				index++;
			}
		}
		return true;
	}

	if (k.is_valid() && !k->is_pressed() && drag_type == DRAG_KEY_MOVE && (tool == TOOL_SELECT || tool == TOOL_MOVE) &&
			(k->get_keycode() == Key::UP || k->get_keycode() == Key::DOWN || k->get_keycode() == Key::LEFT || k->get_keycode() == Key::RIGHT)) {
		// Confirm canvas items move by arrow keys
		if ((!Input::get_singleton()->is_key_pressed(Key::UP)) &&
				(!Input::get_singleton()->is_key_pressed(Key::DOWN)) &&
				(!Input::get_singleton()->is_key_pressed(Key::LEFT)) &&
				(!Input::get_singleton()->is_key_pressed(Key::RIGHT))) {
			if (drag_selection.size() > 1) {
				_commit_canvas_item_state(
						drag_selection,
						vformat(TTR("Move %d CanvasItems"), drag_selection.size()),
						true);
			} else if (drag_selection.size() == 1) {
				_commit_canvas_item_state(
						drag_selection,
						vformat(TTR("Move CanvasItem \"%s\" to (%d, %d)"),
								drag_selection[0]->get_name(),
								drag_selection[0]->_edit_get_position().x,
								drag_selection[0]->_edit_get_position().y),
						true);
			}
			drag_type = DRAG_NONE;
		}
		viewport->update();
		return true;
	}

	return (k.is_valid() && (k->get_keycode() == Key::UP || k->get_keycode() == Key::DOWN || k->get_keycode() == Key::LEFT || k->get_keycode() == Key::RIGHT)); // Accept the key event in any case
}

bool CanvasItemEditor::_gui_input_select(const Ref<InputEvent> &p_event) {
	Ref<InputEventMouseButton> b = p_event;
	Ref<InputEventMouseMotion> m = p_event;
	Ref<InputEventKey> k = p_event;

	if (drag_type == DRAG_NONE) {
		if (b.is_valid() &&
				((b->get_button_index() == MouseButton::RIGHT && b->is_alt_pressed() && tool == TOOL_SELECT) ||
						(b->get_button_index() == MouseButton::LEFT && tool == TOOL_LIST_SELECT))) {
			// Popup the selection menu list
			Point2 click = transform.affine_inverse().xform(b->get_position());

			_get_canvas_items_at_pos(click, selection_results, b->is_alt_pressed() && tool == TOOL_SELECT);

			if (selection_results.size() == 1) {
				CanvasItem *item = selection_results[0].item;
				selection_results.clear();

				_select_click_on_item(item, click, b->is_shift_pressed());

				return true;
			} else if (!selection_results.is_empty()) {
				// Sorts items according the their z-index
				selection_results.sort();

				NodePath root_path = get_tree()->get_edited_scene_root()->get_path();
				StringName root_name = root_path.get_name(root_path.get_name_count() - 1);

				for (int i = 0; i < selection_results.size(); i++) {
					CanvasItem *item = selection_results[i].item;

					Ref<Texture2D> icon = EditorNode::get_singleton()->get_object_icon(item, "Node");
					String node_path = "/" + root_name + "/" + root_path.rel_path_to(item->get_path());

					int locked = 0;
					if (_is_node_locked(item)) {
						locked = 1;
					} else {
						Node *scene = editor->get_edited_scene();
						Node *node = item;

						while (node && node != scene->get_parent()) {
							CanvasItem *canvas_item_tmp = Object::cast_to<CanvasItem>(node);
							if (canvas_item_tmp && node->has_meta("_edit_group_")) {
								locked = 2;
							}
							node = node->get_parent();
						}
					}

					String suffix = String();
					if (locked == 1) {
						suffix = " (" + TTR("Locked") + ")";
					} else if (locked == 2) {
						suffix = " (" + TTR("Grouped") + ")";
					}
					selection_menu->add_item((String)item->get_name() + suffix);
					selection_menu->set_item_icon(i, icon);
					selection_menu->set_item_metadata(i, node_path);
					selection_menu->set_item_tooltip(i, String(item->get_name()) + "\nType: " + item->get_class() + "\nPath: " + node_path);
				}

				selection_menu_additive_selection = b->is_shift_pressed();
				selection_menu->set_position(get_screen_position() + b->get_position());
				selection_menu->reset_size();
				selection_menu->popup();
				return true;
			}
		}

		if (b.is_valid() && b->is_pressed() && b->get_button_index() == MouseButton::RIGHT) {
			add_node_menu->reset_size();
			add_node_menu->set_position(get_screen_transform().xform(get_local_mouse_position()));
			add_node_menu->popup();
			node_create_position = transform.affine_inverse().xform((get_local_mouse_position()));
			return true;
		}

		if (b.is_valid() && b->get_button_index() == MouseButton::LEFT && b->is_pressed() && tool == TOOL_SELECT) {
			// Single item selection
			Point2 click = transform.affine_inverse().xform(b->get_position());

			Node *scene = editor->get_edited_scene();
			if (!scene) {
				return true;
			}

			// Find the item to select
			CanvasItem *canvas_item = nullptr;

			Vector<_SelectResult> selection = Vector<_SelectResult>();
			// Retrieve the canvas items
			selection = Vector<_SelectResult>();
			_get_canvas_items_at_pos(click, selection);
			if (!selection.is_empty()) {
				canvas_item = selection[0].item;
			}

			if (!canvas_item) {
				// Start a box selection
				if (!b->is_shift_pressed()) {
					// Clear the selection if not additive
					editor_selection->clear();
					viewport->update();
					selected_from_canvas = true;
				};

				drag_from = click;
				drag_type = DRAG_BOX_SELECTION;
				box_selecting_to = drag_from;
				return true;
			} else {
				bool still_selected = _select_click_on_item(canvas_item, click, b->is_shift_pressed());
				// Start dragging
				if (still_selected) {
					// Drag the node(s) if requested
					drag_start_origin = click;
					drag_type = DRAG_QUEUED;
				}
				// Select the item
				return true;
			}
		}
	}

	if (drag_type == DRAG_QUEUED) {
		if (b.is_valid() && !b->is_pressed()) {
			drag_type = DRAG_NONE;
			return true;
		}
		if (m.is_valid()) {
			Point2 click = transform.affine_inverse().xform(m->get_position());
			bool movement_threshold_passed = drag_start_origin.distance_to(click) > (8 * MAX(1, EDSCALE)) / zoom;
			if (m.is_valid() && movement_threshold_passed) {
				List<CanvasItem *> selection2 = _get_edited_canvas_items();

				drag_selection.clear();
				for (int i = 0; i < selection2.size(); i++) {
					if (_is_node_movable(selection2[i], true)) {
						drag_selection.push_back(selection2[i]);
					}
				}

				if (selection2.size() > 0) {
					drag_type = DRAG_MOVE;
					drag_from = drag_start_origin;
					_save_canvas_item_state(drag_selection);
				}
				return true;
			}
		}
	}

	if (drag_type == DRAG_BOX_SELECTION) {
		if (b.is_valid() && !b->is_pressed() && b->get_button_index() == MouseButton::LEFT) {
			// Confirms box selection
			Node *scene = editor->get_edited_scene();
			if (scene) {
				List<CanvasItem *> selitems;

				Point2 bsfrom = drag_from;
				Point2 bsto = box_selecting_to;
				if (bsfrom.x > bsto.x) {
					SWAP(bsfrom.x, bsto.x);
				}
				if (bsfrom.y > bsto.y) {
					SWAP(bsfrom.y, bsto.y);
				}

				_find_canvas_items_in_rect(Rect2(bsfrom, bsto - bsfrom), scene, &selitems);
				if (selitems.size() == 1 && editor_selection->get_selected_node_list().is_empty()) {
					editor->push_item(selitems[0]);
				}
				for (CanvasItem *E : selitems) {
					editor_selection->add_node(E);
				}
			}

			drag_type = DRAG_NONE;
			viewport->update();
			return true;
		}

		if (b.is_valid() && b->is_pressed() && b->get_button_index() == MouseButton::RIGHT) {
			// Cancel box selection
			drag_type = DRAG_NONE;
			viewport->update();
			return true;
		}

		if (m.is_valid()) {
			// Update box selection
			box_selecting_to = transform.affine_inverse().xform(m->get_position());
			viewport->update();
			return true;
		}
	}

	if (k.is_valid() && k->is_pressed() && k->get_keycode() == Key::ESCAPE && drag_type == DRAG_NONE && tool == TOOL_SELECT) {
		// Unselect everything
		editor_selection->clear();
		viewport->update();
	}
	return false;
}

bool CanvasItemEditor::_gui_input_ruler_tool(const Ref<InputEvent> &p_event) {
	if (tool != TOOL_RULER) {
		ruler_tool_active = false;
		return false;
	}

	Ref<InputEventMouseButton> b = p_event;
	Ref<InputEventMouseMotion> m = p_event;

	Point2 previous_origin = ruler_tool_origin;
	if (!ruler_tool_active) {
		ruler_tool_origin = snap_point(viewport->get_local_mouse_position() / zoom + view_offset);
	}

	if (b.is_valid() && b->get_button_index() == MouseButton::LEFT) {
		if (b->is_pressed()) {
			ruler_tool_active = true;
		} else {
			ruler_tool_active = false;
		}

		viewport->update();
		return true;
	}

	if (m.is_valid() && (ruler_tool_active || (grid_snap_active && previous_origin != ruler_tool_origin))) {
		viewport->update();
		return true;
	}

	return false;
}

bool CanvasItemEditor::_gui_input_hover(const Ref<InputEvent> &p_event) {
	Ref<InputEventMouseMotion> m = p_event;
	if (m.is_valid()) {
		Point2 click = transform.affine_inverse().xform(m->get_position());

		// Checks if the hovered items changed, update the viewport if so
		Vector<_SelectResult> hovering_results_items;
		_get_canvas_items_at_pos(click, hovering_results_items);
		hovering_results_items.sort();

		// Compute the nodes names and icon position
		Vector<_HoverResult> hovering_results_tmp;
		for (int i = 0; i < hovering_results_items.size(); i++) {
			CanvasItem *canvas_item = hovering_results_items[i].item;

			if (canvas_item->_edit_use_rect()) {
				continue;
			}

			_HoverResult hover_result;
			hover_result.position = canvas_item->get_global_transform_with_canvas().get_origin();
			hover_result.icon = EditorNode::get_singleton()->get_object_icon(canvas_item);
			hover_result.name = canvas_item->get_name();

			hovering_results_tmp.push_back(hover_result);
		}

		// Check if changed, if so, update.
		bool changed = false;
		if (hovering_results_tmp.size() == hovering_results.size()) {
			for (int i = 0; i < hovering_results_tmp.size(); i++) {
				_HoverResult a = hovering_results_tmp[i];
				_HoverResult b = hovering_results[i];
				if (a.icon != b.icon || a.name != b.name || a.position != b.position) {
					changed = true;
					break;
				}
			}
		} else {
			changed = true;
		}

		if (changed) {
			hovering_results = hovering_results_tmp;
			viewport->update();
		}

		return true;
	}

	return false;
}

void CanvasItemEditor::_gui_input_viewport(const Ref<InputEvent> &p_event) {
	bool accepted = false;

	if (EditorSettings::get_singleton()->get("editors/2d/simple_panning") || !pan_pressed) {
		if ((accepted = _gui_input_rulers_and_guides(p_event))) {
			//printf("Rulers and guides\n");
		} else if ((accepted = editor->get_editor_plugins_over()->forward_gui_input(p_event))) {
			//printf("Plugin\n");
		} else if ((accepted = _gui_input_open_scene_on_double_click(p_event))) {
			//printf("Open scene on double click\n");
		} else if ((accepted = _gui_input_scale(p_event))) {
			//printf("Set scale\n");
		} else if ((accepted = _gui_input_pivot(p_event))) {
			//printf("Set pivot\n");
		} else if ((accepted = _gui_input_resize(p_event))) {
			//printf("Resize\n");
		} else if ((accepted = _gui_input_rotate(p_event))) {
			//printf("Rotate\n");
		} else if ((accepted = _gui_input_move(p_event))) {
			//printf("Move\n");
		} else if ((accepted = _gui_input_anchors(p_event))) {
			//printf("Anchors\n");
		} else if ((accepted = _gui_input_select(p_event))) {
			//printf("Selection\n");
		} else if ((accepted = _gui_input_ruler_tool(p_event))) {
			//printf("Measure\n");
		} else {
			//printf("Not accepted\n");
		}
	}

	accepted = (_gui_input_zoom_or_pan(p_event, accepted) || accepted);

	if (accepted) {
		accept_event();
	}

	// Handles the mouse hovering
	_gui_input_hover(p_event);

	// Change the cursor
	_update_cursor();

	// Grab focus
	if (!viewport->has_focus() && (!get_focus_owner() || !get_focus_owner()->is_text_field())) {
		viewport->call_deferred(SNAME("grab_focus"));
	}
}

void CanvasItemEditor::_update_cursor() {
	// Compute an eventual rotation of the cursor
	CursorShape rotation_array[4] = { CURSOR_HSIZE, CURSOR_BDIAGSIZE, CURSOR_VSIZE, CURSOR_FDIAGSIZE };
	int rotation_array_index = 0;

	List<CanvasItem *> selection = _get_edited_canvas_items();
	if (selection.size() == 1) {
		const double angle = Math::fposmod((double)selection[0]->get_global_transform_with_canvas().get_rotation(), Math_PI);
		if (angle > Math_PI * 7.0 / 8.0) {
			rotation_array_index = 0;
		} else if (angle > Math_PI * 5.0 / 8.0) {
			rotation_array_index = 1;
		} else if (angle > Math_PI * 3.0 / 8.0) {
			rotation_array_index = 2;
		} else if (angle > Math_PI * 1.0 / 8.0) {
			rotation_array_index = 3;
		} else {
			rotation_array_index = 0;
		}
	}

	// Choose the correct cursor
	CursorShape c = CURSOR_ARROW;
	switch (drag_type) {
		case DRAG_NONE:
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
				case TOOL_RULER:
					c = CURSOR_CROSS;
					break;
				default:
					break;
			}
			break;
		case DRAG_LEFT:
		case DRAG_RIGHT:
			c = rotation_array[rotation_array_index];
			break;
		case DRAG_V_GUIDE:
			c = CURSOR_HSIZE;
			break;
		case DRAG_TOP:
		case DRAG_BOTTOM:
			c = rotation_array[(rotation_array_index + 2) % 4];
			break;
		case DRAG_H_GUIDE:
			c = CURSOR_VSIZE;
			break;
		case DRAG_TOP_LEFT:
		case DRAG_BOTTOM_RIGHT:
			c = rotation_array[(rotation_array_index + 3) % 4];
			break;
		case DRAG_DOUBLE_GUIDE:
			c = CURSOR_FDIAGSIZE;
			break;
		case DRAG_TOP_RIGHT:
		case DRAG_BOTTOM_LEFT:
			c = rotation_array[(rotation_array_index + 1) % 4];
			break;
		case DRAG_MOVE:
			c = CURSOR_MOVE;
			break;
		default:
			break;
	}

	if (is_hovering_h_guide) {
		c = CURSOR_VSIZE;
	} else if (is_hovering_v_guide) {
		c = CURSOR_HSIZE;
	}

	if (pan_pressed) {
		c = CURSOR_DRAG;
	}

	if (c != viewport->get_default_cursor_shape()) {
		viewport->set_default_cursor_shape(c);

		// Force refresh cursor if it's over the viewport.
		if (viewport->get_global_rect().has_point(get_global_mouse_position())) {
			DisplayServer::CursorShape ds_cursor_shape = (DisplayServer::CursorShape)viewport->get_default_cursor_shape();
			DisplayServer::get_singleton()->cursor_set_shape(ds_cursor_shape);
		}
	}
}

void CanvasItemEditor::_draw_text_at_position(Point2 p_position, String p_string, Side p_side) {
	Color color = get_theme_color(SNAME("font_color"), SNAME("Editor"));
	color.a = 0.8;
	Ref<Font> font = get_theme_font(SNAME("font"), SNAME("Label"));
	int font_size = get_theme_font_size(SNAME("font_size"), SNAME("Label"));
	Size2 text_size = font->get_string_size(p_string, font_size);
	switch (p_side) {
		case SIDE_LEFT:
			p_position += Vector2(-text_size.x - 5, text_size.y / 2);
			break;
		case SIDE_TOP:
			p_position += Vector2(-text_size.x / 2, -5);
			break;
		case SIDE_RIGHT:
			p_position += Vector2(5, text_size.y / 2);
			break;
		case SIDE_BOTTOM:
			p_position += Vector2(-text_size.x / 2, text_size.y + 5);
			break;
	}
	viewport->draw_string(font, p_position, p_string, HORIZONTAL_ALIGNMENT_LEFT, -1, font_size, color);
}

void CanvasItemEditor::_draw_margin_at_position(int p_value, Point2 p_position, Side p_side) {
	String str = TS->format_number(vformat("%d " + TTR("px"), p_value));
	if (p_value != 0) {
		_draw_text_at_position(p_position, str, p_side);
	}
}

void CanvasItemEditor::_draw_percentage_at_position(real_t p_value, Point2 p_position, Side p_side) {
	String str = TS->format_number(vformat("%.1f ", p_value * 100.0)) + TS->percent_sign();
	if (p_value != 0) {
		_draw_text_at_position(p_position, str, p_side);
	}
}

void CanvasItemEditor::_draw_focus() {
	// Draw the focus around the base viewport
	if (viewport->has_focus()) {
		get_theme_stylebox(SNAME("FocusViewport"), SNAME("EditorStyles"))->draw(viewport->get_canvas_item(), Rect2(Point2(), viewport->get_size()));
	}
}

void CanvasItemEditor::_draw_guides() {
	Color guide_color = EditorSettings::get_singleton()->get("editors/2d/guides_color");
	Transform2D xform = viewport_scrollable->get_transform() * transform;

	// Guides already there
	if (EditorNode::get_singleton()->get_edited_scene() && EditorNode::get_singleton()->get_edited_scene()->has_meta("_edit_vertical_guides_")) {
		Array vguides = EditorNode::get_singleton()->get_edited_scene()->get_meta("_edit_vertical_guides_");
		for (int i = 0; i < vguides.size(); i++) {
			if (drag_type == DRAG_V_GUIDE && i == dragged_guide_index) {
				continue;
			}
			real_t x = xform.xform(Point2(vguides[i], 0)).x;
			viewport->draw_line(Point2(x, 0), Point2(x, viewport->get_size().y), guide_color, Math::round(EDSCALE));
		}
	}

	if (EditorNode::get_singleton()->get_edited_scene() && EditorNode::get_singleton()->get_edited_scene()->has_meta("_edit_horizontal_guides_")) {
		Array hguides = EditorNode::get_singleton()->get_edited_scene()->get_meta("_edit_horizontal_guides_");
		for (int i = 0; i < hguides.size(); i++) {
			if (drag_type == DRAG_H_GUIDE && i == dragged_guide_index) {
				continue;
			}
			real_t y = xform.xform(Point2(0, hguides[i])).y;
			viewport->draw_line(Point2(0, y), Point2(viewport->get_size().x, y), guide_color, Math::round(EDSCALE));
		}
	}

	// Dragged guide
	Color text_color = get_theme_color(SNAME("font_color"), SNAME("Editor"));
	Color outline_color = text_color.inverted();
	const float outline_size = 2;
	if (drag_type == DRAG_DOUBLE_GUIDE || drag_type == DRAG_V_GUIDE) {
		String str = TS->format_number(vformat("%d px", Math::round(xform.affine_inverse().xform(dragged_guide_pos).x)));
		Ref<Font> font = get_theme_font(SNAME("bold"), SNAME("EditorFonts"));
		int font_size = get_theme_font_size(SNAME("bold_size"), SNAME("EditorFonts"));
		Size2 text_size = font->get_string_size(str, font_size);
		viewport->draw_string(font, Point2(dragged_guide_pos.x + 10, RULER_WIDTH + text_size.y / 2 + 10), str, HORIZONTAL_ALIGNMENT_LEFT, -1, font_size, text_color, outline_size, outline_color);
		viewport->draw_line(Point2(dragged_guide_pos.x, 0), Point2(dragged_guide_pos.x, viewport->get_size().y), guide_color, Math::round(EDSCALE));
	}
	if (drag_type == DRAG_DOUBLE_GUIDE || drag_type == DRAG_H_GUIDE) {
		String str = TS->format_number(vformat("%d px", Math::round(xform.affine_inverse().xform(dragged_guide_pos).y)));
		Ref<Font> font = get_theme_font(SNAME("bold"), SNAME("EditorFonts"));
		int font_size = get_theme_font_size(SNAME("bold_size"), SNAME("EditorFonts"));
		Size2 text_size = font->get_string_size(str, font_size);
		viewport->draw_string(font, Point2(RULER_WIDTH + 10, dragged_guide_pos.y + text_size.y / 2 + 10), str, HORIZONTAL_ALIGNMENT_LEFT, -1, font_size, text_color, outline_size, outline_color);
		viewport->draw_line(Point2(0, dragged_guide_pos.y), Point2(viewport->get_size().x, dragged_guide_pos.y), guide_color, Math::round(EDSCALE));
	}
}

void CanvasItemEditor::_draw_smart_snapping() {
	Color line_color = EditorSettings::get_singleton()->get("editors/2d/smart_snapping_line_color");
	if (snap_target[0] != SNAP_TARGET_NONE && snap_target[0] != SNAP_TARGET_GRID) {
		viewport->draw_set_transform_matrix(viewport->get_transform() * transform * snap_transform);
		viewport->draw_line(Point2(0, -1.0e+10F), Point2(0, 1.0e+10F), line_color);
		viewport->draw_set_transform_matrix(viewport->get_transform());
	}
	if (snap_target[1] != SNAP_TARGET_NONE && snap_target[1] != SNAP_TARGET_GRID) {
		viewport->draw_set_transform_matrix(viewport->get_transform() * transform * snap_transform);
		viewport->draw_line(Point2(-1.0e+10F, 0), Point2(1.0e+10F, 0), line_color);
		viewport->draw_set_transform_matrix(viewport->get_transform());
	}
}

void CanvasItemEditor::_draw_rulers() {
	Color bg_color = get_theme_color(SNAME("dark_color_2"), SNAME("Editor"));
	Color graduation_color = get_theme_color(SNAME("font_color"), SNAME("Editor")).lerp(bg_color, 0.5);
	Color font_color = get_theme_color(SNAME("font_color"), SNAME("Editor"));
	font_color.a = 0.8;
	Ref<Font> font = get_theme_font(SNAME("rulers"), SNAME("EditorFonts"));
	int font_size = get_theme_font_size(SNAME("rulers_size"), SNAME("EditorFonts"));

	// The rule transform
	Transform2D ruler_transform = Transform2D();
	if (show_grid || grid_snap_active) {
		List<CanvasItem *> selection = _get_edited_canvas_items();
		if (snap_relative && selection.size() > 0) {
			ruler_transform.translate(_get_encompassing_rect_from_list(selection).position);
			ruler_transform.scale_basis(grid_step * Math::pow(2.0, grid_step_multiplier));
		} else {
			ruler_transform.translate(grid_offset);
			ruler_transform.scale_basis(grid_step * Math::pow(2.0, grid_step_multiplier));
		}
		while ((transform * ruler_transform).get_scale().x < 50 || (transform * ruler_transform).get_scale().y < 50) {
			ruler_transform.scale_basis(Point2(2, 2));
		}
	} else {
		real_t basic_rule = 100;
		for (int i = 0; basic_rule * zoom > 100; i++) {
			basic_rule /= (i % 2) ? 5.0 : 2.0;
		}
		for (int i = 0; basic_rule * zoom < 100; i++) {
			basic_rule *= (i % 2) ? 2.0 : 5.0;
		}
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
	Point2 first = (transform * ruler_transform * major_subdivide * minor_subdivide).affine_inverse().xform(Point2(RULER_WIDTH, RULER_WIDTH));
	Point2 last = (transform * ruler_transform * major_subdivide * minor_subdivide).affine_inverse().xform(viewport->get_size());

	// Draw top ruler
	viewport->draw_rect(Rect2(Point2(RULER_WIDTH, 0), Size2(viewport->get_size().x, RULER_WIDTH)), bg_color);
	for (int i = Math::ceil(first.x); i < last.x; i++) {
		Point2 position = (transform * ruler_transform * major_subdivide * minor_subdivide).xform(Point2(i, 0)).round();
		if (i % (major_subdivision * minor_subdivision) == 0) {
			viewport->draw_line(Point2(position.x, 0), Point2(position.x, RULER_WIDTH), graduation_color, Math::round(EDSCALE));
			real_t val = (ruler_transform * major_subdivide * minor_subdivide).xform(Point2(i, 0)).x;
			viewport->draw_string(font, Point2(position.x + 2, font->get_height(font_size)), TS->format_number(vformat(((int)val == val) ? "%d" : "%.1f", val)), HORIZONTAL_ALIGNMENT_LEFT, -1, font_size, font_color);
		} else {
			if (i % minor_subdivision == 0) {
				viewport->draw_line(Point2(position.x, RULER_WIDTH * 0.33), Point2(position.x, RULER_WIDTH), graduation_color, Math::round(EDSCALE));
			} else {
				viewport->draw_line(Point2(position.x, RULER_WIDTH * 0.75), Point2(position.x, RULER_WIDTH), graduation_color, Math::round(EDSCALE));
			}
		}
	}

	// Draw left ruler
	viewport->draw_rect(Rect2(Point2(0, RULER_WIDTH), Size2(RULER_WIDTH, viewport->get_size().y)), bg_color);
	for (int i = Math::ceil(first.y); i < last.y; i++) {
		Point2 position = (transform * ruler_transform * major_subdivide * minor_subdivide).xform(Point2(0, i)).round();
		if (i % (major_subdivision * minor_subdivision) == 0) {
			viewport->draw_line(Point2(0, position.y), Point2(RULER_WIDTH, position.y), graduation_color, Math::round(EDSCALE));
			real_t val = (ruler_transform * major_subdivide * minor_subdivide).xform(Point2(0, i)).y;

			Transform2D text_xform = Transform2D(-Math_PI / 2.0, Point2(font->get_height(font_size), position.y - 2));
			viewport->draw_set_transform_matrix(viewport->get_transform() * text_xform);
			viewport->draw_string(font, Point2(), TS->format_number(vformat(((int)val == val) ? "%d" : "%.1f", val)), HORIZONTAL_ALIGNMENT_LEFT, -1, font_size, font_color);
			viewport->draw_set_transform_matrix(viewport->get_transform());

		} else {
			if (i % minor_subdivision == 0) {
				viewport->draw_line(Point2(RULER_WIDTH * 0.33, position.y), Point2(RULER_WIDTH, position.y), graduation_color, Math::round(EDSCALE));
			} else {
				viewport->draw_line(Point2(RULER_WIDTH * 0.75, position.y), Point2(RULER_WIDTH, position.y), graduation_color, Math::round(EDSCALE));
			}
		}
	}

	// Draw the top left corner
	viewport->draw_rect(Rect2(Point2(), Size2(RULER_WIDTH, RULER_WIDTH)), graduation_color);
}

void CanvasItemEditor::_draw_grid() {
	if (show_grid || grid_snap_active) {
		// Draw the grid
		Vector2 real_grid_offset;
		const List<CanvasItem *> selection = _get_edited_canvas_items();

		if (snap_relative && selection.size() > 0) {
			const Vector2 topleft = _get_encompassing_rect_from_list(selection).position;
			real_grid_offset.x = fmod(topleft.x, grid_step.x * (real_t)Math::pow(2.0, grid_step_multiplier));
			real_grid_offset.y = fmod(topleft.y, grid_step.y * (real_t)Math::pow(2.0, grid_step_multiplier));
		} else {
			real_grid_offset = grid_offset;
		}

		// Draw a "primary" line every several lines to make measurements easier.
		// The step is configurable in the Configure Snap dialog.
		const Color secondary_grid_color = EditorSettings::get_singleton()->get("editors/2d/grid_color");
		const Color primary_grid_color =
				Color(secondary_grid_color.r, secondary_grid_color.g, secondary_grid_color.b, secondary_grid_color.a * 2.5);

		const Size2 viewport_size = viewport->get_size();
		const Transform2D xform = transform.affine_inverse();
		int last_cell = 0;

		if (grid_step.x != 0) {
			for (int i = 0; i < viewport_size.width; i++) {
				const int cell =
						Math::fast_ftoi(Math::floor((xform.xform(Vector2(i, 0)).x - real_grid_offset.x) / (grid_step.x * Math::pow(2.0, grid_step_multiplier))));

				if (i == 0) {
					last_cell = cell;
				}

				if (last_cell != cell) {
					Color grid_color;
					if (primary_grid_steps == 0) {
						grid_color = secondary_grid_color;
					} else {
						grid_color = cell % primary_grid_steps == 0 ? primary_grid_color : secondary_grid_color;
					}

					viewport->draw_line(Point2(i, 0), Point2(i, viewport_size.height), grid_color, Math::round(EDSCALE));
				}
				last_cell = cell;
			}
		}

		if (grid_step.y != 0) {
			for (int i = 0; i < viewport_size.height; i++) {
				const int cell =
						Math::fast_ftoi(Math::floor((xform.xform(Vector2(0, i)).y - real_grid_offset.y) / (grid_step.y * Math::pow(2.0, grid_step_multiplier))));

				if (i == 0) {
					last_cell = cell;
				}

				if (last_cell != cell) {
					Color grid_color;
					if (primary_grid_steps == 0) {
						grid_color = secondary_grid_color;
					} else {
						grid_color = cell % primary_grid_steps == 0 ? primary_grid_color : secondary_grid_color;
					}

					viewport->draw_line(Point2(0, i), Point2(viewport_size.width, i), grid_color, Math::round(EDSCALE));
				}
				last_cell = cell;
			}
		}
	}
}

void CanvasItemEditor::_draw_ruler_tool() {
	if (tool != TOOL_RULER) {
		return;
	}

	if (ruler_tool_active) {
		Color ruler_primary_color = get_theme_color(SNAME("accent_color"), SNAME("Editor"));
		Color ruler_secondary_color = ruler_primary_color;
		ruler_secondary_color.a = 0.5;

		Point2 begin = (ruler_tool_origin - view_offset) * zoom;
		Point2 end = snap_point(viewport->get_local_mouse_position() / zoom + view_offset) * zoom - view_offset * zoom;
		Point2 corner = Point2(begin.x, end.y);
		Vector2 length_vector = (begin - end).abs() / zoom;

		Ref<Font> font = get_theme_font(SNAME("bold"), SNAME("EditorFonts"));
		int font_size = get_theme_font_size(SNAME("bold_size"), SNAME("EditorFonts"));
		Color font_color = get_theme_color(SNAME("font_color"), SNAME("Editor"));
		Color font_secondary_color = font_color;
		font_secondary_color.set_v(font_secondary_color.get_v() > 0.5 ? 0.7 : 0.3);
		Color outline_color = font_color.inverted();
		float text_height = font->get_height(font_size);

		const float outline_size = 2;
		const float text_width = 76;
		const float angle_text_width = 54;

		Point2 text_pos = (begin + end) / 2 - Vector2(text_width / 2, text_height / 2);
		text_pos.x = CLAMP(text_pos.x, text_width / 2, viewport->get_rect().size.x - text_width * 1.5);
		text_pos.y = CLAMP(text_pos.y, text_height * 1.5, viewport->get_rect().size.y - text_height * 1.5);

		if (begin.is_equal_approx(end)) {
			viewport->draw_string(font, text_pos, (String)ruler_tool_origin, HORIZONTAL_ALIGNMENT_LEFT, -1, font_size, font_color, outline_size, outline_color);
			Ref<Texture2D> position_icon = get_theme_icon(SNAME("EditorPosition"), SNAME("EditorIcons"));
			viewport->draw_texture(get_theme_icon(SNAME("EditorPosition"), SNAME("EditorIcons")), (ruler_tool_origin - view_offset) * zoom - position_icon->get_size() / 2);
			return;
		}

		viewport->draw_string(font, text_pos, TS->format_number(vformat("%.1f px", length_vector.length())), HORIZONTAL_ALIGNMENT_LEFT, -1, font_size, font_color, outline_size, outline_color);

		bool draw_secondary_lines = !(Math::is_equal_approx(begin.y, corner.y) || Math::is_equal_approx(end.x, corner.x));

		viewport->draw_line(begin, end, ruler_primary_color, Math::round(EDSCALE * 3));
		if (draw_secondary_lines) {
			viewport->draw_line(begin, corner, ruler_secondary_color, Math::round(EDSCALE));
			viewport->draw_line(corner, end, ruler_secondary_color, Math::round(EDSCALE));
		}

		if (draw_secondary_lines) {
			const real_t horizontal_angle_rad = length_vector.angle();
			const real_t vertical_angle_rad = Math_PI / 2.0 - horizontal_angle_rad;
			const int horizontal_angle = round(180 * horizontal_angle_rad / Math_PI);
			const int vertical_angle = round(180 * vertical_angle_rad / Math_PI);

			Point2 text_pos2 = text_pos;
			text_pos2.x = begin.x < text_pos.x ? MIN(text_pos.x - text_width, begin.x - text_width / 2) : MAX(text_pos.x + text_width, begin.x - text_width / 2);
			viewport->draw_string(font, text_pos2, TS->format_number(vformat("%.1f px", length_vector.y)), HORIZONTAL_ALIGNMENT_LEFT, -1, font_size, font_secondary_color, outline_size, outline_color);

			Point2 v_angle_text_pos = Point2();
			v_angle_text_pos.x = CLAMP(begin.x - angle_text_width / 2, angle_text_width / 2, viewport->get_rect().size.x - angle_text_width);
			v_angle_text_pos.y = begin.y < end.y ? MIN(text_pos2.y - 2 * text_height, begin.y - text_height * 0.5) : MAX(text_pos2.y + text_height * 3, begin.y + text_height * 1.5);
			viewport->draw_string(font, v_angle_text_pos, TS->format_number(vformat(String::utf8("%d°"), vertical_angle)), HORIZONTAL_ALIGNMENT_LEFT, -1, font_size, font_secondary_color, outline_size, outline_color);

			text_pos2 = text_pos;
			text_pos2.y = end.y < text_pos.y ? MIN(text_pos.y - text_height * 2, end.y - text_height / 2) : MAX(text_pos.y + text_height * 2, end.y - text_height / 2);
			viewport->draw_string(font, text_pos2, TS->format_number(vformat("%.1f px", length_vector.x)), HORIZONTAL_ALIGNMENT_LEFT, -1, font_size, font_secondary_color, outline_size, outline_color);

			Point2 h_angle_text_pos = Point2();
			h_angle_text_pos.x = CLAMP(end.x - angle_text_width / 2, angle_text_width / 2, viewport->get_rect().size.x - angle_text_width);
			if (begin.y < end.y) {
				h_angle_text_pos.y = end.y + text_height * 1.5;
				if (ABS(text_pos2.x - h_angle_text_pos.x) < text_width) {
					int height_multiplier = 1.5 + (int)grid_snap_active;
					h_angle_text_pos.y = MAX(text_pos.y + height_multiplier * text_height, MAX(end.y + text_height * 1.5, text_pos2.y + height_multiplier * text_height));
				}
			} else {
				h_angle_text_pos.y = end.y - text_height * 0.5;
				if (ABS(text_pos2.x - h_angle_text_pos.x) < text_width) {
					int height_multiplier = 1 + (int)grid_snap_active;
					h_angle_text_pos.y = MIN(text_pos.y - height_multiplier * text_height, MIN(end.y - text_height * 0.5, text_pos2.y - height_multiplier * text_height));
				}
			}
			viewport->draw_string(font, h_angle_text_pos, TS->format_number(vformat(String::utf8("%d°"), horizontal_angle)), HORIZONTAL_ALIGNMENT_LEFT, -1, font_size, font_secondary_color, outline_size, outline_color);

			// Angle arcs
			int arc_point_count = 8;
			real_t arc_radius_max_length_percent = 0.1;
			real_t ruler_length = length_vector.length() * zoom;
			real_t arc_max_radius = 50.0;
			real_t arc_line_width = 2.0;

			const Vector2 end_to_begin = (end - begin);

			real_t arc_1_start_angle = end_to_begin.x < 0
					? (end_to_begin.y < 0 ? 3.0 * Math_PI / 2.0 - vertical_angle_rad : Math_PI / 2.0)
					: (end_to_begin.y < 0 ? 3.0 * Math_PI / 2.0 : Math_PI / 2.0 - vertical_angle_rad);
			real_t arc_1_end_angle = arc_1_start_angle + vertical_angle_rad;
			// Constrain arc to triangle height & max size
			real_t arc_1_radius = MIN(MIN(arc_radius_max_length_percent * ruler_length, ABS(end_to_begin.y)), arc_max_radius);

			real_t arc_2_start_angle = end_to_begin.x < 0
					? (end_to_begin.y < 0 ? 0.0 : -horizontal_angle_rad)
					: (end_to_begin.y < 0 ? Math_PI - horizontal_angle_rad : Math_PI);
			real_t arc_2_end_angle = arc_2_start_angle + horizontal_angle_rad;
			// Constrain arc to triangle width & max size
			real_t arc_2_radius = MIN(MIN(arc_radius_max_length_percent * ruler_length, ABS(end_to_begin.x)), arc_max_radius);

			viewport->draw_arc(begin, arc_1_radius, arc_1_start_angle, arc_1_end_angle, arc_point_count, ruler_primary_color, Math::round(EDSCALE * arc_line_width));
			viewport->draw_arc(end, arc_2_radius, arc_2_start_angle, arc_2_end_angle, arc_point_count, ruler_primary_color, Math::round(EDSCALE * arc_line_width));
		}

		if (grid_snap_active) {
			text_pos = (begin + end) / 2 + Vector2(-text_width / 2, text_height / 2);
			text_pos.x = CLAMP(text_pos.x, text_width / 2, viewport->get_rect().size.x - text_width * 1.5);
			text_pos.y = CLAMP(text_pos.y, text_height * 2.5, viewport->get_rect().size.y - text_height / 2);

			if (draw_secondary_lines) {
				viewport->draw_string(font, text_pos, TS->format_number(vformat("%.2f " + TTR("units"), (length_vector / grid_step).length())), HORIZONTAL_ALIGNMENT_LEFT, -1, font_size, font_color, outline_size, outline_color);

				Point2 text_pos2 = text_pos;
				text_pos2.x = begin.x < text_pos.x ? MIN(text_pos.x - text_width, begin.x - text_width / 2) : MAX(text_pos.x + text_width, begin.x - text_width / 2);
				viewport->draw_string(font, text_pos2, TS->format_number(vformat("%d " + TTR("units"), roundf(length_vector.y / grid_step.y))), HORIZONTAL_ALIGNMENT_LEFT, -1, font_size, font_secondary_color, outline_size, outline_color);

				text_pos2 = text_pos;
				text_pos2.y = end.y < text_pos.y ? MIN(text_pos.y - text_height * 2, end.y + text_height / 2) : MAX(text_pos.y + text_height * 2, end.y + text_height / 2);
				viewport->draw_string(font, text_pos2, TS->format_number(vformat("%d " + TTR("units"), roundf(length_vector.x / grid_step.x))), HORIZONTAL_ALIGNMENT_LEFT, -1, font_size, font_secondary_color, outline_size, outline_color);
			} else {
				viewport->draw_string(font, text_pos, TS->format_number(vformat("%d " + TTR("units"), roundf((length_vector / grid_step).length()))), HORIZONTAL_ALIGNMENT_LEFT, -1, font_size, font_color, outline_size, outline_color);
			}
		}
	} else {
		if (grid_snap_active) {
			Ref<Texture2D> position_icon = get_theme_icon(SNAME("EditorPosition"), SNAME("EditorIcons"));
			viewport->draw_texture(get_theme_icon(SNAME("EditorPosition"), SNAME("EditorIcons")), (ruler_tool_origin - view_offset) * zoom - position_icon->get_size() / 2);
		}
	}
}

void CanvasItemEditor::_draw_control_anchors(Control *control) {
	Transform2D xform = transform * control->get_global_transform_with_canvas();
	RID ci = viewport->get_canvas_item();
	if (tool == TOOL_SELECT && !Object::cast_to<Container>(control->get_parent())) {
		// Compute the anchors
		real_t anchors_values[4];
		anchors_values[0] = control->get_anchor(SIDE_LEFT);
		anchors_values[1] = control->get_anchor(SIDE_TOP);
		anchors_values[2] = control->get_anchor(SIDE_RIGHT);
		anchors_values[3] = control->get_anchor(SIDE_BOTTOM);

		Vector2 anchors_pos[4];
		for (int i = 0; i < 4; i++) {
			Vector2 value = Vector2((i % 2 == 0) ? anchors_values[i] : anchors_values[(i + 1) % 4], (i % 2 == 1) ? anchors_values[i] : anchors_values[(i + 1) % 4]);
			anchors_pos[i] = xform.xform(_anchor_to_position(control, value));
		}

		// Draw the anchors handles
		Rect2 anchor_rects[4];
		if (control->is_layout_rtl()) {
			anchor_rects[0] = Rect2(anchors_pos[0] - Vector2(0.0, anchor_handle->get_size().y), Point2(-anchor_handle->get_size().x, anchor_handle->get_size().y));
			anchor_rects[1] = Rect2(anchors_pos[1] - anchor_handle->get_size(), anchor_handle->get_size());
			anchor_rects[2] = Rect2(anchors_pos[2] - Vector2(anchor_handle->get_size().x, 0.0), Point2(anchor_handle->get_size().x, -anchor_handle->get_size().y));
			anchor_rects[3] = Rect2(anchors_pos[3], -anchor_handle->get_size());
		} else {
			anchor_rects[0] = Rect2(anchors_pos[0] - anchor_handle->get_size(), anchor_handle->get_size());
			anchor_rects[1] = Rect2(anchors_pos[1] - Vector2(0.0, anchor_handle->get_size().y), Point2(-anchor_handle->get_size().x, anchor_handle->get_size().y));
			anchor_rects[2] = Rect2(anchors_pos[2], -anchor_handle->get_size());
			anchor_rects[3] = Rect2(anchors_pos[3] - Vector2(anchor_handle->get_size().x, 0.0), Point2(anchor_handle->get_size().x, -anchor_handle->get_size().y));
		}

		for (int i = 0; i < 4; i++) {
			anchor_handle->draw_rect(ci, anchor_rects[i]);
		}
	}
}

void CanvasItemEditor::_draw_control_helpers(Control *control) {
	Transform2D xform = transform * control->get_global_transform_with_canvas();
	if (tool == TOOL_SELECT && show_helpers && !Object::cast_to<Container>(control->get_parent())) {
		// Draw the helpers
		Color color_base = Color(0.8, 0.8, 0.8, 0.5);

		// Compute the anchors
		real_t anchors_values[4];
		anchors_values[0] = control->get_anchor(SIDE_LEFT);
		anchors_values[1] = control->get_anchor(SIDE_TOP);
		anchors_values[2] = control->get_anchor(SIDE_RIGHT);
		anchors_values[3] = control->get_anchor(SIDE_BOTTOM);

		Vector2 anchors[4];
		Vector2 anchors_pos[4];
		for (int i = 0; i < 4; i++) {
			anchors[i] = Vector2((i % 2 == 0) ? anchors_values[i] : anchors_values[(i + 1) % 4], (i % 2 == 1) ? anchors_values[i] : anchors_values[(i + 1) % 4]);
			anchors_pos[i] = xform.xform(_anchor_to_position(control, anchors[i]));
		}

		// Get which anchor is dragged
		int dragged_anchor = -1;
		switch (drag_type) {
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
			default:
				break;
		}

		if (dragged_anchor >= 0) {
			// Draw the 4 lines when dragged
			bool anchor_snapped;
			Color color_snapped = Color(0.64, 0.93, 0.67, 0.5);

			Vector2 corners_pos[4];
			for (int i = 0; i < 4; i++) {
				corners_pos[i] = xform.xform(_anchor_to_position(control, Vector2((i == 0 || i == 3) ? ANCHOR_BEGIN : ANCHOR_END, (i <= 1) ? ANCHOR_BEGIN : ANCHOR_END)));
			}

			Vector2 line_starts[4];
			Vector2 line_ends[4];
			for (int i = 0; i < 4; i++) {
				real_t anchor_val = (i >= 2) ? ANCHOR_END - anchors_values[i] : anchors_values[i];
				line_starts[i] = corners_pos[i].lerp(corners_pos[(i + 1) % 4], anchor_val);
				line_ends[i] = corners_pos[(i + 3) % 4].lerp(corners_pos[(i + 2) % 4], anchor_val);
				anchor_snapped = anchors_values[i] == 0.0 || anchors_values[i] == 0.5 || anchors_values[i] == 1.0;
				viewport->draw_line(line_starts[i], line_ends[i], anchor_snapped ? color_snapped : color_base, (i == dragged_anchor || (i + 3) % 4 == dragged_anchor) ? 2 : 1);
			}

			// Display the percentages next to the lines
			real_t percent_val;
			percent_val = anchors_values[(dragged_anchor + 2) % 4] - anchors_values[dragged_anchor];
			percent_val = (dragged_anchor >= 2) ? -percent_val : percent_val;
			_draw_percentage_at_position(percent_val, (anchors_pos[dragged_anchor] + anchors_pos[(dragged_anchor + 1) % 4]) / 2, (Side)((dragged_anchor + 1) % 4));

			percent_val = anchors_values[(dragged_anchor + 3) % 4] - anchors_values[(dragged_anchor + 1) % 4];
			percent_val = ((dragged_anchor + 1) % 4 >= 2) ? -percent_val : percent_val;
			_draw_percentage_at_position(percent_val, (anchors_pos[dragged_anchor] + anchors_pos[(dragged_anchor + 3) % 4]) / 2, (Side)(dragged_anchor));

			percent_val = anchors_values[(dragged_anchor + 1) % 4];
			percent_val = ((dragged_anchor + 1) % 4 >= 2) ? ANCHOR_END - percent_val : percent_val;
			_draw_percentage_at_position(percent_val, (line_starts[dragged_anchor] + anchors_pos[dragged_anchor]) / 2, (Side)(dragged_anchor));

			percent_val = anchors_values[dragged_anchor];
			percent_val = (dragged_anchor >= 2) ? ANCHOR_END - percent_val : percent_val;
			_draw_percentage_at_position(percent_val, (line_ends[(dragged_anchor + 1) % 4] + anchors_pos[dragged_anchor]) / 2, (Side)((dragged_anchor + 1) % 4));
		}

		// Draw the margin values and the node width/height when dragging control side
		const real_t ratio = 0.33;
		Transform2D parent_transform = xform * control->get_transform().affine_inverse();
		real_t node_pos_in_parent[4];

		Rect2 parent_rect = control->get_parent_anchorable_rect();

		node_pos_in_parent[0] = control->get_anchor(SIDE_LEFT) * parent_rect.size.width + control->get_offset(SIDE_LEFT) + parent_rect.position.x;
		node_pos_in_parent[1] = control->get_anchor(SIDE_TOP) * parent_rect.size.height + control->get_offset(SIDE_TOP) + parent_rect.position.y;
		node_pos_in_parent[2] = control->get_anchor(SIDE_RIGHT) * parent_rect.size.width + control->get_offset(SIDE_RIGHT) + parent_rect.position.x;
		node_pos_in_parent[3] = control->get_anchor(SIDE_BOTTOM) * parent_rect.size.height + control->get_offset(SIDE_BOTTOM) + parent_rect.position.y;

		Point2 start, end;
		switch (drag_type) {
			case DRAG_LEFT:
			case DRAG_TOP_LEFT:
			case DRAG_BOTTOM_LEFT:
				_draw_margin_at_position(control->get_size().width, parent_transform.xform(Vector2((node_pos_in_parent[0] + node_pos_in_parent[2]) / 2, node_pos_in_parent[3])) + Vector2(0, 5), SIDE_BOTTOM);
				[[fallthrough]];
			case DRAG_MOVE:
				start = Vector2(node_pos_in_parent[0], Math::lerp(node_pos_in_parent[1], node_pos_in_parent[3], ratio));
				end = start - Vector2(control->get_offset(SIDE_LEFT), 0);
				_draw_margin_at_position(control->get_offset(SIDE_LEFT), parent_transform.xform((start + end) / 2), SIDE_TOP);
				viewport->draw_line(parent_transform.xform(start), parent_transform.xform(end), color_base, Math::round(EDSCALE));
				break;
			default:
				break;
		}
		switch (drag_type) {
			case DRAG_RIGHT:
			case DRAG_TOP_RIGHT:
			case DRAG_BOTTOM_RIGHT:
				_draw_margin_at_position(control->get_size().width, parent_transform.xform(Vector2((node_pos_in_parent[0] + node_pos_in_parent[2]) / 2, node_pos_in_parent[3])) + Vector2(0, 5), SIDE_BOTTOM);
				[[fallthrough]];
			case DRAG_MOVE:
				start = Vector2(node_pos_in_parent[2], Math::lerp(node_pos_in_parent[3], node_pos_in_parent[1], ratio));
				end = start - Vector2(control->get_offset(SIDE_RIGHT), 0);
				_draw_margin_at_position(control->get_offset(SIDE_RIGHT), parent_transform.xform((start + end) / 2), SIDE_BOTTOM);
				viewport->draw_line(parent_transform.xform(start), parent_transform.xform(end), color_base, Math::round(EDSCALE));
				break;
			default:
				break;
		}
		switch (drag_type) {
			case DRAG_TOP:
			case DRAG_TOP_LEFT:
			case DRAG_TOP_RIGHT:
				_draw_margin_at_position(control->get_size().height, parent_transform.xform(Vector2(node_pos_in_parent[2], (node_pos_in_parent[1] + node_pos_in_parent[3]) / 2)) + Vector2(5, 0), SIDE_RIGHT);
				[[fallthrough]];
			case DRAG_MOVE:
				start = Vector2(Math::lerp(node_pos_in_parent[0], node_pos_in_parent[2], ratio), node_pos_in_parent[1]);
				end = start - Vector2(0, control->get_offset(SIDE_TOP));
				_draw_margin_at_position(control->get_offset(SIDE_TOP), parent_transform.xform((start + end) / 2), SIDE_LEFT);
				viewport->draw_line(parent_transform.xform(start), parent_transform.xform(end), color_base, Math::round(EDSCALE));
				break;
			default:
				break;
		}
		switch (drag_type) {
			case DRAG_BOTTOM:
			case DRAG_BOTTOM_LEFT:
			case DRAG_BOTTOM_RIGHT:
				_draw_margin_at_position(control->get_size().height, parent_transform.xform(Vector2(node_pos_in_parent[2], (node_pos_in_parent[1] + node_pos_in_parent[3]) / 2) + Vector2(5, 0)), SIDE_RIGHT);
				[[fallthrough]];
			case DRAG_MOVE:
				start = Vector2(Math::lerp(node_pos_in_parent[2], node_pos_in_parent[0], ratio), node_pos_in_parent[3]);
				end = start - Vector2(0, control->get_offset(SIDE_BOTTOM));
				_draw_margin_at_position(control->get_offset(SIDE_BOTTOM), parent_transform.xform((start + end) / 2), SIDE_RIGHT);
				viewport->draw_line(parent_transform.xform(start), parent_transform.xform(end), color_base, Math::round(EDSCALE));
				break;
			default:
				break;
		}

		switch (drag_type) {
			//Draw the ghost rect if the node if rotated/scaled
			case DRAG_LEFT:
			case DRAG_TOP_LEFT:
			case DRAG_TOP:
			case DRAG_TOP_RIGHT:
			case DRAG_RIGHT:
			case DRAG_BOTTOM_RIGHT:
			case DRAG_BOTTOM:
			case DRAG_BOTTOM_LEFT:
			case DRAG_MOVE:
				if (control->get_rotation() != 0.0 || control->get_scale() != Vector2(1, 1)) {
					Rect2 rect = Rect2(Vector2(node_pos_in_parent[0], node_pos_in_parent[1]), control->get_size());
					viewport->draw_rect(parent_transform.xform(rect), color_base, false, Math::round(EDSCALE));
				}
				break;
			default:
				break;
		}
	}
}

void CanvasItemEditor::_draw_selection() {
	Ref<Texture2D> pivot_icon = get_theme_icon(SNAME("EditorPivot"), SNAME("EditorIcons"));
	Ref<Texture2D> position_icon = get_theme_icon(SNAME("EditorPosition"), SNAME("EditorIcons"));
	Ref<Texture2D> previous_position_icon = get_theme_icon(SNAME("EditorPositionPrevious"), SNAME("EditorIcons"));

	RID ci = viewport->get_canvas_item();

	List<CanvasItem *> selection = _get_edited_canvas_items(true, false);

	bool single = selection.size() == 1;
	for (CanvasItem *E : selection) {
		CanvasItem *canvas_item = Object::cast_to<CanvasItem>(E);
		CanvasItemEditorSelectedItem *se = editor_selection->get_node_editor_data<CanvasItemEditorSelectedItem>(canvas_item);

		bool item_locked = canvas_item->has_meta("_edit_lock_");

		// Draw the previous position if we are dragging the node
		if (show_helpers &&
				(drag_type == DRAG_MOVE || drag_type == DRAG_ROTATE ||
						drag_type == DRAG_LEFT || drag_type == DRAG_RIGHT || drag_type == DRAG_TOP || drag_type == DRAG_BOTTOM ||
						drag_type == DRAG_TOP_LEFT || drag_type == DRAG_TOP_RIGHT || drag_type == DRAG_BOTTOM_LEFT || drag_type == DRAG_BOTTOM_RIGHT)) {
			const Transform2D pre_drag_xform = transform * se->pre_drag_xform;
			const Color pre_drag_color = Color(0.4, 0.6, 1, 0.7);

			if (canvas_item->_edit_use_rect()) {
				Vector2 pre_drag_endpoints[4] = {
					pre_drag_xform.xform(se->pre_drag_rect.position),
					pre_drag_xform.xform(se->pre_drag_rect.position + Vector2(se->pre_drag_rect.size.x, 0)),
					pre_drag_xform.xform(se->pre_drag_rect.position + se->pre_drag_rect.size),
					pre_drag_xform.xform(se->pre_drag_rect.position + Vector2(0, se->pre_drag_rect.size.y))
				};

				for (int i = 0; i < 4; i++) {
					viewport->draw_line(pre_drag_endpoints[i], pre_drag_endpoints[(i + 1) % 4], pre_drag_color, Math::round(2 * EDSCALE));
				}
			} else {
				viewport->draw_texture(previous_position_icon, (pre_drag_xform.xform(Point2()) - (previous_position_icon->get_size() / 2)).floor());
			}
		}

		Transform2D xform = transform * canvas_item->get_global_transform_with_canvas();

		// Draw the selected items position / surrounding boxes
		if (canvas_item->_edit_use_rect()) {
			Rect2 rect = canvas_item->_edit_get_rect();
			Vector2 endpoints[4] = {
				xform.xform(rect.position),
				xform.xform(rect.position + Vector2(rect.size.x, 0)),
				xform.xform(rect.position + rect.size),
				xform.xform(rect.position + Vector2(0, rect.size.y))
			};

			Color c = Color(1, 0.6, 0.4, 0.7);

			if (item_locked) {
				c = Color(0.7, 0.7, 0.7, 0.7);
			}

			for (int i = 0; i < 4; i++) {
				viewport->draw_line(endpoints[i], endpoints[(i + 1) % 4], c, Math::round(2 * EDSCALE));
			}
		} else {
			Transform2D unscaled_transform = (xform * canvas_item->get_transform().affine_inverse() * canvas_item->_edit_get_transform()).orthonormalized();
			Transform2D simple_xform = viewport->get_transform() * unscaled_transform;
			viewport->draw_set_transform_matrix(simple_xform);
			viewport->draw_texture(position_icon, -(position_icon->get_size() / 2));
			viewport->draw_set_transform_matrix(viewport->get_transform());
		}

		if (single && !item_locked && (tool == TOOL_SELECT || tool == TOOL_MOVE || tool == TOOL_SCALE || tool == TOOL_ROTATE || tool == TOOL_EDIT_PIVOT)) { //kind of sucks
			// Draw the pivot
			if (canvas_item->_edit_use_pivot()) {
				// Draw the node's pivot
				Transform2D unscaled_transform = (xform * canvas_item->get_transform().affine_inverse() * canvas_item->_edit_get_transform()).orthonormalized();
				Transform2D simple_xform = viewport->get_transform() * unscaled_transform;

				viewport->draw_set_transform_matrix(simple_xform);
				viewport->draw_texture(pivot_icon, -(pivot_icon->get_size() / 2).floor());
				viewport->draw_set_transform_matrix(viewport->get_transform());
			}

			// Draw control-related helpers
			Control *control = Object::cast_to<Control>(canvas_item);
			if (control && _is_node_movable(control)) {
				_draw_control_anchors(control);
				_draw_control_helpers(control);
			}

			// Draw the resize handles
			if (tool == TOOL_SELECT && canvas_item->_edit_use_rect() && _is_node_movable(canvas_item)) {
				Rect2 rect = canvas_item->_edit_get_rect();
				Vector2 endpoints[4] = {
					xform.xform(rect.position),
					xform.xform(rect.position + Vector2(rect.size.x, 0)),
					xform.xform(rect.position + rect.size),
					xform.xform(rect.position + Vector2(0, rect.size.y))
				};
				for (int i = 0; i < 4; i++) {
					int prev = (i + 3) % 4;
					int next = (i + 1) % 4;

					Vector2 ofs = ((endpoints[i] - endpoints[prev]).normalized() + ((endpoints[i] - endpoints[next]).normalized())).normalized();
					ofs *= Math_SQRT2 * (select_handle->get_size().width / 2);

					select_handle->draw(ci, (endpoints[i] + ofs - (select_handle->get_size() / 2)).floor());

					ofs = (endpoints[i] + endpoints[next]) / 2;
					ofs += (endpoints[next] - endpoints[i]).orthogonal().normalized() * (select_handle->get_size().width / 2);

					select_handle->draw(ci, (ofs - (select_handle->get_size() / 2)).floor());
				}
			}

			// Draw the move handles
			bool is_ctrl = Input::get_singleton()->is_key_pressed(Key::CTRL);
			bool is_alt = Input::get_singleton()->is_key_pressed(Key::ALT);
			if (tool == TOOL_MOVE && show_transformation_gizmos) {
				if (_is_node_movable(canvas_item)) {
					Transform2D unscaled_transform = (xform * canvas_item->get_transform().affine_inverse() * canvas_item->_edit_get_transform()).orthonormalized();
					Transform2D simple_xform = viewport->get_transform() * unscaled_transform;

					Size2 move_factor = Size2(MOVE_HANDLE_DISTANCE, MOVE_HANDLE_DISTANCE);
					viewport->draw_set_transform_matrix(simple_xform);

					Vector<Point2> points;
					points.push_back(Vector2(move_factor.x * EDSCALE, 5 * EDSCALE));
					points.push_back(Vector2(move_factor.x * EDSCALE, -5 * EDSCALE));
					points.push_back(Vector2((move_factor.x + 10) * EDSCALE, 0));

					viewport->draw_colored_polygon(points, get_theme_color(SNAME("axis_x_color"), SNAME("Editor")));
					viewport->draw_line(Point2(), Point2(move_factor.x * EDSCALE, 0), get_theme_color(SNAME("axis_x_color"), SNAME("Editor")), Math::round(EDSCALE));

					points.clear();
					points.push_back(Vector2(5 * EDSCALE, move_factor.y * EDSCALE));
					points.push_back(Vector2(-5 * EDSCALE, move_factor.y * EDSCALE));
					points.push_back(Vector2(0, (move_factor.y + 10) * EDSCALE));

					viewport->draw_colored_polygon(points, get_theme_color(SNAME("axis_y_color"), SNAME("Editor")));
					viewport->draw_line(Point2(), Point2(0, move_factor.y * EDSCALE), get_theme_color(SNAME("axis_y_color"), SNAME("Editor")), Math::round(EDSCALE));

					viewport->draw_set_transform_matrix(viewport->get_transform());
				}
			}

			// Draw the rescale handles
			if (show_transformation_gizmos && ((is_alt && is_ctrl) || tool == TOOL_SCALE || drag_type == DRAG_SCALE_X || drag_type == DRAG_SCALE_Y)) {
				if (_is_node_movable(canvas_item)) {
					Transform2D unscaled_transform = (xform * canvas_item->get_transform().affine_inverse() * canvas_item->_edit_get_transform()).orthonormalized();
					Transform2D simple_xform = viewport->get_transform() * unscaled_transform;

					Size2 scale_factor = Size2(SCALE_HANDLE_DISTANCE, SCALE_HANDLE_DISTANCE);
					bool uniform = Input::get_singleton()->is_key_pressed(Key::SHIFT);
					Point2 offset = (simple_xform.affine_inverse().xform(drag_to) - simple_xform.affine_inverse().xform(drag_from)) * zoom;

					if (drag_type == DRAG_SCALE_X) {
						scale_factor.x += offset.x;
						if (uniform) {
							scale_factor.y += offset.x;
						}
					} else if (drag_type == DRAG_SCALE_Y) {
						scale_factor.y += offset.y;
						if (uniform) {
							scale_factor.x += offset.y;
						}
					}

					viewport->draw_set_transform_matrix(simple_xform);
					Rect2 x_handle_rect = Rect2(scale_factor.x * EDSCALE, -5 * EDSCALE, 10 * EDSCALE, 10 * EDSCALE);
					viewport->draw_rect(x_handle_rect, get_theme_color(SNAME("axis_x_color"), SNAME("Editor")));
					viewport->draw_line(Point2(), Point2(scale_factor.x * EDSCALE, 0), get_theme_color(SNAME("axis_x_color"), SNAME("Editor")), Math::round(EDSCALE));

					Rect2 y_handle_rect = Rect2(-5 * EDSCALE, scale_factor.y * EDSCALE, 10 * EDSCALE, 10 * EDSCALE);
					viewport->draw_rect(y_handle_rect, get_theme_color(SNAME("axis_y_color"), SNAME("Editor")));
					viewport->draw_line(Point2(), Point2(0, scale_factor.y * EDSCALE), get_theme_color(SNAME("axis_y_color"), SNAME("Editor")), Math::round(EDSCALE));

					viewport->draw_set_transform_matrix(viewport->get_transform());
				}
			}
		}
	}

	if (drag_type == DRAG_BOX_SELECTION) {
		// Draw the dragging box
		Point2 bsfrom = transform.xform(drag_from);
		Point2 bsto = transform.xform(box_selecting_to);

		viewport->draw_rect(
				Rect2(bsfrom, bsto - bsfrom),
				get_theme_color(SNAME("box_selection_fill_color"), SNAME("Editor")));

		viewport->draw_rect(
				Rect2(bsfrom, bsto - bsfrom),
				get_theme_color(SNAME("box_selection_stroke_color"), SNAME("Editor")),
				false,
				Math::round(EDSCALE));
	}

	if (drag_type == DRAG_ROTATE) {
		// Draw the line when rotating a node
		viewport->draw_line(
				transform.xform(drag_rotation_center),
				transform.xform(drag_to),
				get_theme_color(SNAME("accent_color"), SNAME("Editor")) * Color(1, 1, 1, 0.6),
				Math::round(2 * EDSCALE));
	}
}

void CanvasItemEditor::_draw_straight_line(Point2 p_from, Point2 p_to, Color p_color) {
	// Draw a line going through the whole screen from a vector
	RID ci = viewport->get_canvas_item();
	Vector<Point2> points;
	Point2 from = transform.xform(p_from);
	Point2 to = transform.xform(p_to);
	Size2 viewport_size = viewport->get_size();

	if (to.x == from.x) {
		// Vertical line
		points.push_back(Point2(to.x, 0));
		points.push_back(Point2(to.x, viewport_size.y));
	} else if (to.y == from.y) {
		// Horizontal line
		points.push_back(Point2(0, to.y));
		points.push_back(Point2(viewport_size.x, to.y));
	} else {
		real_t y_for_zero_x = (to.y * from.x - from.y * to.x) / (from.x - to.x);
		real_t x_for_zero_y = (to.x * from.y - from.x * to.y) / (from.y - to.y);
		real_t y_for_viewport_x = ((to.y - from.y) * (viewport_size.x - from.x)) / (to.x - from.x) + from.y;
		real_t x_for_viewport_y = ((to.x - from.x) * (viewport_size.y - from.y)) / (to.y - from.y) + from.x; // faux

		//bool start_set = false;
		if (y_for_zero_x >= 0 && y_for_zero_x <= viewport_size.y) {
			points.push_back(Point2(0, y_for_zero_x));
		}
		if (x_for_zero_y >= 0 && x_for_zero_y <= viewport_size.x) {
			points.push_back(Point2(x_for_zero_y, 0));
		}
		if (y_for_viewport_x >= 0 && y_for_viewport_x <= viewport_size.y) {
			points.push_back(Point2(viewport_size.x, y_for_viewport_x));
		}
		if (x_for_viewport_y >= 0 && x_for_viewport_y <= viewport_size.x) {
			points.push_back(Point2(x_for_viewport_y, viewport_size.y));
		}
	}
	if (points.size() >= 2) {
		RenderingServer::get_singleton()->canvas_item_add_line(ci, points[0], points[1], p_color);
	}
}

void CanvasItemEditor::_draw_axis() {
	if (show_origin) {
		_draw_straight_line(Point2(), Point2(1, 0), get_theme_color(SNAME("axis_x_color"), SNAME("Editor")) * Color(1, 1, 1, 0.75));
		_draw_straight_line(Point2(), Point2(0, 1), get_theme_color(SNAME("axis_y_color"), SNAME("Editor")) * Color(1, 1, 1, 0.75));
	}

	if (show_viewport) {
		RID ci = viewport->get_canvas_item();

		Color area_axis_color = EditorSettings::get_singleton()->get("editors/2d/viewport_border_color");

		Size2 screen_size = Size2(ProjectSettings::get_singleton()->get("display/window/size/width"), ProjectSettings::get_singleton()->get("display/window/size/height"));

		Vector2 screen_endpoints[4] = {
			transform.xform(Vector2(0, 0)),
			transform.xform(Vector2(screen_size.width, 0)),
			transform.xform(Vector2(screen_size.width, screen_size.height)),
			transform.xform(Vector2(0, screen_size.height))
		};

		for (int i = 0; i < 4; i++) {
			RenderingServer::get_singleton()->canvas_item_add_line(ci, screen_endpoints[i], screen_endpoints[(i + 1) % 4], area_axis_color);
		}
	}
}

void CanvasItemEditor::_draw_invisible_nodes_positions(Node *p_node, const Transform2D &p_parent_xform, const Transform2D &p_canvas_xform) {
	ERR_FAIL_COND(!p_node);

	Node *scene = editor->get_edited_scene();
	if (p_node != scene && p_node->get_owner() != scene && !scene->is_editable_instance(p_node->get_owner())) {
		return;
	}
	CanvasItem *canvas_item = Object::cast_to<CanvasItem>(p_node);
	if (canvas_item && !canvas_item->is_visible()) {
		return;
	}

	Transform2D parent_xform = p_parent_xform;
	Transform2D canvas_xform = p_canvas_xform;

	if (canvas_item && !canvas_item->is_set_as_top_level()) {
		parent_xform = parent_xform * canvas_item->get_transform();
	} else {
		CanvasLayer *cl = Object::cast_to<CanvasLayer>(p_node);
		parent_xform = Transform2D();
		canvas_xform = cl ? cl->get_transform() : p_canvas_xform;
	}

	for (int i = p_node->get_child_count() - 1; i >= 0; i--) {
		_draw_invisible_nodes_positions(p_node->get_child(i), parent_xform, canvas_xform);
	}

	if (canvas_item && !canvas_item->_edit_use_rect() && (!editor_selection->is_selected(canvas_item) || _is_node_locked(canvas_item))) {
		Transform2D xform = transform * canvas_xform * parent_xform;

		// Draw the node's position
		Ref<Texture2D> position_icon = get_theme_icon(SNAME("EditorPositionUnselected"), SNAME("EditorIcons"));
		Transform2D unscaled_transform = (xform * canvas_item->get_transform().affine_inverse() * canvas_item->_edit_get_transform()).orthonormalized();
		Transform2D simple_xform = viewport->get_transform() * unscaled_transform;
		viewport->draw_set_transform_matrix(simple_xform);
		viewport->draw_texture(position_icon, -position_icon->get_size() / 2, Color(1.0, 1.0, 1.0, 0.5));
		viewport->draw_set_transform_matrix(viewport->get_transform());
	}
}

void CanvasItemEditor::_draw_hover() {
	List<Rect2> previous_rects;

	for (int i = 0; i < hovering_results.size(); i++) {
		Ref<Texture2D> node_icon = hovering_results[i].icon;
		String node_name = hovering_results[i].name;

		Ref<Font> font = get_theme_font(SNAME("font"), SNAME("Label"));
		int font_size = get_theme_font_size(SNAME("font_size"), SNAME("Label"));
		Size2 node_name_size = font->get_string_size(node_name, font_size);
		Size2 item_size = Size2(node_icon->get_size().x + 4 + node_name_size.x, MAX(node_icon->get_size().y, node_name_size.y - 3));

		Point2 pos = transform.xform(hovering_results[i].position) - Point2(0, item_size.y) + (Point2(node_icon->get_size().x, -node_icon->get_size().y) / 4);
		// Rectify the position to avoid overlapping items
		for (const Rect2 &E : previous_rects) {
			if (E.intersects(Rect2(pos, item_size))) {
				pos.y = E.get_position().y - item_size.y;
			}
		}

		previous_rects.push_back(Rect2(pos, item_size));

		// Draw icon
		viewport->draw_texture(node_icon, pos, Color(1.0, 1.0, 1.0, 0.5));

		// Draw name
		viewport->draw_string(font, pos + Point2(node_icon->get_size().x + 4, item_size.y - 3), node_name, HORIZONTAL_ALIGNMENT_LEFT, -1, font_size, Color(1.0, 1.0, 1.0, 0.5));
	}
}

void CanvasItemEditor::_draw_locks_and_groups(Node *p_node, const Transform2D &p_parent_xform, const Transform2D &p_canvas_xform) {
	ERR_FAIL_COND(!p_node);

	Node *scene = editor->get_edited_scene();
	if (p_node != scene && p_node->get_owner() != scene && !scene->is_editable_instance(p_node->get_owner())) {
		return;
	}
	CanvasItem *canvas_item = Object::cast_to<CanvasItem>(p_node);
	if (canvas_item && !canvas_item->is_visible()) {
		return;
	}

	Transform2D parent_xform = p_parent_xform;
	Transform2D canvas_xform = p_canvas_xform;

	if (canvas_item && !canvas_item->is_set_as_top_level()) {
		parent_xform = parent_xform * canvas_item->get_transform();
	} else {
		CanvasLayer *cl = Object::cast_to<CanvasLayer>(p_node);
		parent_xform = Transform2D();
		canvas_xform = cl ? cl->get_transform() : p_canvas_xform;
	}

	for (int i = p_node->get_child_count() - 1; i >= 0; i--) {
		_draw_locks_and_groups(p_node->get_child(i), parent_xform, canvas_xform);
	}

	RID viewport_canvas_item = viewport->get_canvas_item();
	if (canvas_item) {
		real_t offset = 0;

		Ref<Texture2D> lock = get_theme_icon(SNAME("LockViewport"), SNAME("EditorIcons"));
		if (p_node->has_meta("_edit_lock_") && show_edit_locks) {
			lock->draw(viewport_canvas_item, (transform * canvas_xform * parent_xform).xform(Point2(0, 0)) + Point2(offset, 0));
			offset += lock->get_size().x;
		}

		Ref<Texture2D> group = get_theme_icon(SNAME("GroupViewport"), SNAME("EditorIcons"));
		if (canvas_item->has_meta("_edit_group_") && show_edit_locks) {
			group->draw(viewport_canvas_item, (transform * canvas_xform * parent_xform).xform(Point2(0, 0)) + Point2(offset, 0));
			//offset += group->get_size().x;
		}
	}
}

void CanvasItemEditor::_draw_viewport() {
	// Update the transform
	transform = Transform2D();
	transform.scale_basis(Size2(zoom, zoom));
	transform.elements[2] = -view_offset * zoom;
	editor->get_scene_root()->set_global_canvas_transform(transform);

	// hide/show buttons depending on the selection
	bool all_locked = true;
	bool all_group = true;
	List<Node *> selection = editor_selection->get_selected_node_list();
	if (selection.is_empty()) {
		all_locked = false;
		all_group = false;
	} else {
		for (Node *E : selection) {
			if (Object::cast_to<CanvasItem>(E) && !Object::cast_to<CanvasItem>(E)->has_meta("_edit_lock_")) {
				all_locked = false;
				break;
			}
		}
		for (Node *E : selection) {
			if (Object::cast_to<CanvasItem>(E) && !Object::cast_to<CanvasItem>(E)->has_meta("_edit_group_")) {
				all_group = false;
				break;
			}
		}
	}

	lock_button->set_visible(!all_locked);
	lock_button->set_disabled(selection.is_empty());
	unlock_button->set_visible(all_locked);
	group_button->set_visible(!all_group);
	group_button->set_disabled(selection.is_empty());
	ungroup_button->set_visible(all_group);

	_draw_grid();
	_draw_ruler_tool();
	_draw_axis();
	if (editor->get_edited_scene()) {
		_draw_locks_and_groups(editor->get_edited_scene());
		_draw_invisible_nodes_positions(editor->get_edited_scene());
	}
	_draw_selection();

	RID ci = viewport->get_canvas_item();
	RenderingServer::get_singleton()->canvas_item_add_set_transform(ci, Transform2D());

	EditorPluginList *over_plugin_list = editor->get_editor_plugins_over();
	if (!over_plugin_list->is_empty()) {
		over_plugin_list->forward_canvas_draw_over_viewport(viewport);
	}
	EditorPluginList *force_over_plugin_list = editor->get_editor_plugins_force_over();
	if (!force_over_plugin_list->is_empty()) {
		force_over_plugin_list->forward_canvas_force_draw_over_viewport(viewport);
	}

	if (show_rulers) {
		_draw_rulers();
	}
	if (show_guides) {
		_draw_guides();
	}
	_draw_smart_snapping();
	_draw_focus();
	_draw_hover();
}

void CanvasItemEditor::update_viewport() {
	_update_scrollbars();
	viewport->update();
}

void CanvasItemEditor::set_current_tool(Tool p_tool) {
	_button_tool_select(p_tool);
}

void CanvasItemEditor::_notification(int p_what) {
	if (p_what == NOTIFICATION_PHYSICS_PROCESS) {
		EditorNode::get_singleton()->get_scene_root()->set_snap_controls_to_pixels(GLOBAL_GET("gui/common/snap_controls_to_pixels"));

		bool has_container_parents = false;
		int nb_control = 0;
		int nb_having_pivot = 0;

		// Update the viewport if the canvas_item changes
		List<CanvasItem *> selection = _get_edited_canvas_items(true);
		for (CanvasItem *canvas_item : selection) {
			CanvasItemEditorSelectedItem *se = editor_selection->get_node_editor_data<CanvasItemEditorSelectedItem>(canvas_item);

			Rect2 rect;
			if (canvas_item->_edit_use_rect()) {
				rect = canvas_item->_edit_get_rect();
			} else {
				rect = Rect2();
			}
			Transform2D xform = canvas_item->get_transform();

			if (rect != se->prev_rect || xform != se->prev_xform) {
				viewport->update();
				se->prev_rect = rect;
				se->prev_xform = xform;
			}

			Control *control = Object::cast_to<Control>(canvas_item);
			if (control) {
				real_t anchors[4];
				Vector2 pivot;

				pivot = control->get_pivot_offset();
				anchors[SIDE_LEFT] = control->get_anchor(SIDE_LEFT);
				anchors[SIDE_RIGHT] = control->get_anchor(SIDE_RIGHT);
				anchors[SIDE_TOP] = control->get_anchor(SIDE_TOP);
				anchors[SIDE_BOTTOM] = control->get_anchor(SIDE_BOTTOM);

				if (pivot != se->prev_pivot || anchors[SIDE_LEFT] != se->prev_anchors[SIDE_LEFT] || anchors[SIDE_RIGHT] != se->prev_anchors[SIDE_RIGHT] || anchors[SIDE_TOP] != se->prev_anchors[SIDE_TOP] || anchors[SIDE_BOTTOM] != se->prev_anchors[SIDE_BOTTOM]) {
					se->prev_pivot = pivot;
					se->prev_anchors[SIDE_LEFT] = anchors[SIDE_LEFT];
					se->prev_anchors[SIDE_RIGHT] = anchors[SIDE_RIGHT];
					se->prev_anchors[SIDE_TOP] = anchors[SIDE_TOP];
					se->prev_anchors[SIDE_BOTTOM] = anchors[SIDE_BOTTOM];
					viewport->update();
				}
				nb_control++;

				if (Object::cast_to<Container>(control->get_parent())) {
					has_container_parents = true;
				}
			}

			if (canvas_item->_edit_use_pivot()) {
				nb_having_pivot++;
			}
		}

		// Activate / Deactivate the pivot tool
		pivot_button->set_disabled(nb_having_pivot == 0);

		// Show / Hide the layout and anchors mode buttons
		if (nb_control > 0 && nb_control == selection.size()) {
			presets_menu->set_visible(true);
			anchor_mode_button->set_visible(true);

			// Disable if the selected node is child of a container
			if (has_container_parents) {
				presets_menu->set_disabled(true);
				presets_menu->set_tooltip(TTR("Children of containers have their anchors and margins values overridden by their parent."));
				anchor_mode_button->set_disabled(true);
				anchor_mode_button->set_tooltip(TTR("Children of containers have their anchors and margins values overridden by their parent."));
			} else {
				presets_menu->set_disabled(false);
				presets_menu->set_tooltip(TTR("Presets for the anchors and margins values of a Control node."));
				anchor_mode_button->set_disabled(false);
				anchor_mode_button->set_tooltip(TTR("When active, moving Control nodes changes their anchors instead of their margins."));
			}
		} else {
			presets_menu->set_visible(false);
			anchor_mode_button->set_visible(false);
		}

		// Update the viewport if bones changes
		for (KeyValue<BoneKey, BoneList> &E : bone_list) {
			Object *b = ObjectDB::get_instance(E.key.from);
			if (!b) {
				viewport->update();
				break;
			}

			Node2D *b2 = Object::cast_to<Node2D>(b);
			if (!b2 || !b2->is_inside_tree()) {
				continue;
			}

			Transform2D global_xform = b2->get_global_transform();

			if (global_xform != E.value.xform) {
				E.value.xform = global_xform;
				viewport->update();
			}

			Bone2D *bone = Object::cast_to<Bone2D>(b);
			if (bone && bone->get_length() != E.value.length) {
				E.value.length = bone->get_length();
				viewport->update();
			}
		}
	}

	if (p_what == NOTIFICATION_ENTER_TREE) {
		select_sb->set_texture(get_theme_icon(SNAME("EditorRect2D"), SNAME("EditorIcons")));
		for (int i = 0; i < 4; i++) {
			select_sb->set_margin_size(Side(i), 4);
			select_sb->set_default_margin(Side(i), 4);
		}

		AnimationPlayerEditor::get_singleton()->get_track_editor()->connect("visibility_changed", callable_mp(this, &CanvasItemEditor::_keying_changed));
		_keying_changed();

	} else if (p_what == EditorSettings::NOTIFICATION_EDITOR_SETTINGS_CHANGED) {
		select_sb->set_texture(get_theme_icon(SNAME("EditorRect2D"), SNAME("EditorIcons")));
	}

	if (p_what == NOTIFICATION_ENTER_TREE || p_what == EditorSettings::NOTIFICATION_EDITOR_SETTINGS_CHANGED) {
		select_button->set_icon(get_theme_icon(SNAME("ToolSelect"), SNAME("EditorIcons")));
		list_select_button->set_icon(get_theme_icon(SNAME("ListSelect"), SNAME("EditorIcons")));
		move_button->set_icon(get_theme_icon(SNAME("ToolMove"), SNAME("EditorIcons")));
		scale_button->set_icon(get_theme_icon(SNAME("ToolScale"), SNAME("EditorIcons")));
		rotate_button->set_icon(get_theme_icon(SNAME("ToolRotate"), SNAME("EditorIcons")));
		smart_snap_button->set_icon(get_theme_icon(SNAME("Snap"), SNAME("EditorIcons")));
		grid_snap_button->set_icon(get_theme_icon(SNAME("SnapGrid"), SNAME("EditorIcons")));
		snap_config_menu->set_icon(get_theme_icon(SNAME("GuiTabMenuHl"), SNAME("EditorIcons")));
		skeleton_menu->set_icon(get_theme_icon(SNAME("Bone"), SNAME("EditorIcons")));
		override_camera_button->set_icon(get_theme_icon(SNAME("Camera2D"), SNAME("EditorIcons")));
		pan_button->set_icon(get_theme_icon(SNAME("ToolPan"), SNAME("EditorIcons")));
		ruler_button->set_icon(get_theme_icon(SNAME("Ruler"), SNAME("EditorIcons")));
		pivot_button->set_icon(get_theme_icon(SNAME("EditPivot"), SNAME("EditorIcons")));
		select_handle = get_theme_icon(SNAME("EditorHandle"), SNAME("EditorIcons"));
		anchor_handle = get_theme_icon(SNAME("EditorControlAnchor"), SNAME("EditorIcons"));
		lock_button->set_icon(get_theme_icon(SNAME("Lock"), SNAME("EditorIcons")));
		unlock_button->set_icon(get_theme_icon(SNAME("Unlock"), SNAME("EditorIcons")));
		group_button->set_icon(get_theme_icon(SNAME("Group"), SNAME("EditorIcons")));
		ungroup_button->set_icon(get_theme_icon(SNAME("Ungroup"), SNAME("EditorIcons")));
		key_loc_button->set_icon(get_theme_icon(SNAME("KeyPosition"), SNAME("EditorIcons")));
		key_rot_button->set_icon(get_theme_icon(SNAME("KeyRotation"), SNAME("EditorIcons")));
		key_scale_button->set_icon(get_theme_icon(SNAME("KeyScale"), SNAME("EditorIcons")));
		key_insert_button->set_icon(get_theme_icon(SNAME("Key"), SNAME("EditorIcons")));
		key_auto_insert_button->set_icon(get_theme_icon(SNAME("AutoKey"), SNAME("EditorIcons")));
		// Use a different color for the active autokey icon to make them easier
		// to distinguish from the other key icons at the top. On a light theme,
		// the icon will be dark, so we need to lighten it before blending it
		// with the red color.
		const Color key_auto_color = EditorSettings::get_singleton()->is_dark_theme() ? Color(1, 1, 1) : Color(4.25, 4.25, 4.25);
		key_auto_insert_button->add_theme_color_override("icon_pressed_color", key_auto_color.lerp(Color(1, 0, 0), 0.55));
		animation_menu->set_icon(get_theme_icon(SNAME("GuiTabMenuHl"), SNAME("EditorIcons")));

		_update_context_menu_stylebox();

		presets_menu->set_icon(get_theme_icon(SNAME("ControlLayout"), SNAME("EditorIcons")));

		PopupMenu *p = presets_menu->get_popup();

		p->clear();
		p->add_icon_item(get_theme_icon(SNAME("ControlAlignTopLeft"), SNAME("EditorIcons")), TTR("Top Left"), ANCHORS_AND_OFFSETS_PRESET_TOP_LEFT);
		p->add_icon_item(get_theme_icon(SNAME("ControlAlignTopRight"), SNAME("EditorIcons")), TTR("Top Right"), ANCHORS_AND_OFFSETS_PRESET_TOP_RIGHT);
		p->add_icon_item(get_theme_icon(SNAME("ControlAlignBottomRight"), SNAME("EditorIcons")), TTR("Bottom Right"), ANCHORS_AND_OFFSETS_PRESET_BOTTOM_RIGHT);
		p->add_icon_item(get_theme_icon(SNAME("ControlAlignBottomLeft"), SNAME("EditorIcons")), TTR("Bottom Left"), ANCHORS_AND_OFFSETS_PRESET_BOTTOM_LEFT);
		p->add_separator();
		p->add_icon_item(get_theme_icon(SNAME("ControlAlignLeftCenter"), SNAME("EditorIcons")), TTR("Center Left"), ANCHORS_AND_OFFSETS_PRESET_CENTER_LEFT);
		p->add_icon_item(get_theme_icon(SNAME("ControlAlignTopCenter"), SNAME("EditorIcons")), TTR("Center Top"), ANCHORS_AND_OFFSETS_PRESET_CENTER_TOP);
		p->add_icon_item(get_theme_icon(SNAME("ControlAlignRightCenter"), SNAME("EditorIcons")), TTR("Center Right"), ANCHORS_AND_OFFSETS_PRESET_CENTER_RIGHT);
		p->add_icon_item(get_theme_icon(SNAME("ControlAlignBottomCenter"), SNAME("EditorIcons")), TTR("Center Bottom"), ANCHORS_AND_OFFSETS_PRESET_CENTER_BOTTOM);
		p->add_icon_item(get_theme_icon(SNAME("ControlAlignCenter"), SNAME("EditorIcons")), TTR("Center"), ANCHORS_AND_OFFSETS_PRESET_CENTER);
		p->add_separator();
		p->add_icon_item(get_theme_icon(SNAME("ControlAlignLeftWide"), SNAME("EditorIcons")), TTR("Left Wide"), ANCHORS_AND_OFFSETS_PRESET_LEFT_WIDE);
		p->add_icon_item(get_theme_icon(SNAME("ControlAlignTopWide"), SNAME("EditorIcons")), TTR("Top Wide"), ANCHORS_AND_OFFSETS_PRESET_TOP_WIDE);
		p->add_icon_item(get_theme_icon(SNAME("ControlAlignRightWide"), SNAME("EditorIcons")), TTR("Right Wide"), ANCHORS_AND_OFFSETS_PRESET_RIGHT_WIDE);
		p->add_icon_item(get_theme_icon(SNAME("ControlAlignBottomWide"), SNAME("EditorIcons")), TTR("Bottom Wide"), ANCHORS_AND_OFFSETS_PRESET_BOTTOM_WIDE);
		p->add_icon_item(get_theme_icon(SNAME("ControlVcenterWide"), SNAME("EditorIcons")), TTR("VCenter Wide"), ANCHORS_AND_OFFSETS_PRESET_VCENTER_WIDE);
		p->add_icon_item(get_theme_icon(SNAME("ControlHcenterWide"), SNAME("EditorIcons")), TTR("HCenter Wide"), ANCHORS_AND_OFFSETS_PRESET_HCENTER_WIDE);
		p->add_separator();
		p->add_icon_item(get_theme_icon(SNAME("ControlAlignWide"), SNAME("EditorIcons")), TTR("Full Rect"), ANCHORS_AND_OFFSETS_PRESET_WIDE);
		p->add_icon_item(get_theme_icon(SNAME("Anchor"), SNAME("EditorIcons")), TTR("Keep Ratio"), ANCHORS_AND_OFFSETS_PRESET_KEEP_RATIO);
		p->add_separator();
		p->add_submenu_item(TTR("Anchors only"), "Anchors");
		p->set_item_icon(21, get_theme_icon(SNAME("Anchor"), SNAME("EditorIcons")));

		anchors_popup->clear();
		anchors_popup->add_icon_item(get_theme_icon(SNAME("ControlAlignTopLeft"), SNAME("EditorIcons")), TTR("Top Left"), ANCHORS_PRESET_TOP_LEFT);
		anchors_popup->add_icon_item(get_theme_icon(SNAME("ControlAlignTopRight"), SNAME("EditorIcons")), TTR("Top Right"), ANCHORS_PRESET_TOP_RIGHT);
		anchors_popup->add_icon_item(get_theme_icon(SNAME("ControlAlignBottomRight"), SNAME("EditorIcons")), TTR("Bottom Right"), ANCHORS_PRESET_BOTTOM_RIGHT);
		anchors_popup->add_icon_item(get_theme_icon(SNAME("ControlAlignBottomLeft"), SNAME("EditorIcons")), TTR("Bottom Left"), ANCHORS_PRESET_BOTTOM_LEFT);
		anchors_popup->add_separator();
		anchors_popup->add_icon_item(get_theme_icon(SNAME("ControlAlignLeftCenter"), SNAME("EditorIcons")), TTR("Center Left"), ANCHORS_PRESET_CENTER_LEFT);
		anchors_popup->add_icon_item(get_theme_icon(SNAME("ControlAlignTopCenter"), SNAME("EditorIcons")), TTR("Center Top"), ANCHORS_PRESET_CENTER_TOP);
		anchors_popup->add_icon_item(get_theme_icon(SNAME("ControlAlignRightCenter"), SNAME("EditorIcons")), TTR("Center Right"), ANCHORS_PRESET_CENTER_RIGHT);
		anchors_popup->add_icon_item(get_theme_icon(SNAME("ControlAlignBottomCenter"), SNAME("EditorIcons")), TTR("Center Bottom"), ANCHORS_PRESET_CENTER_BOTTOM);
		anchors_popup->add_icon_item(get_theme_icon(SNAME("ControlAlignCenter"), SNAME("EditorIcons")), TTR("Center"), ANCHORS_PRESET_CENTER);
		anchors_popup->add_separator();
		anchors_popup->add_icon_item(get_theme_icon(SNAME("ControlAlignLeftWide"), SNAME("EditorIcons")), TTR("Left Wide"), ANCHORS_PRESET_LEFT_WIDE);
		anchors_popup->add_icon_item(get_theme_icon(SNAME("ControlAlignTopWide"), SNAME("EditorIcons")), TTR("Top Wide"), ANCHORS_PRESET_TOP_WIDE);
		anchors_popup->add_icon_item(get_theme_icon(SNAME("ControlAlignRightWide"), SNAME("EditorIcons")), TTR("Right Wide"), ANCHORS_PRESET_RIGHT_WIDE);
		anchors_popup->add_icon_item(get_theme_icon(SNAME("ControlAlignBottomWide"), SNAME("EditorIcons")), TTR("Bottom Wide"), ANCHORS_PRESET_BOTTOM_WIDE);
		anchors_popup->add_icon_item(get_theme_icon(SNAME("ControlVcenterWide"), SNAME("EditorIcons")), TTR("VCenter Wide"), ANCHORS_PRESET_VCENTER_WIDE);
		anchors_popup->add_icon_item(get_theme_icon(SNAME("ControlHcenterWide"), SNAME("EditorIcons")), TTR("HCenter Wide"), ANCHORS_PRESET_HCENTER_WIDE);
		anchors_popup->add_separator();
		anchors_popup->add_icon_item(get_theme_icon(SNAME("ControlAlignWide"), SNAME("EditorIcons")), TTR("Full Rect"), ANCHORS_PRESET_WIDE);

		anchor_mode_button->set_icon(get_theme_icon(SNAME("Anchor"), SNAME("EditorIcons")));
	}

	if (p_what == NOTIFICATION_VISIBILITY_CHANGED) {
		if (!is_visible() && override_camera_button->is_pressed()) {
			EditorDebuggerNode *debugger = EditorDebuggerNode::get_singleton();

			debugger->set_camera_override(EditorDebuggerNode::OVERRIDE_NONE);
			override_camera_button->set_pressed(false);
		}
	}
}

void CanvasItemEditor::_selection_changed() {
	// Update the anchors_mode
	int nbValidControls = 0;
	int nbAnchorsMode = 0;
	List<Node *> selection = editor_selection->get_selected_node_list();
	for (Node *E : selection) {
		Control *control = Object::cast_to<Control>(E);
		if (!control) {
			continue;
		}
		if (Object::cast_to<Container>(control->get_parent())) {
			continue;
		}

		nbValidControls++;
		if (control->has_meta("_edit_use_anchors_") && control->get_meta("_edit_use_anchors_")) {
			nbAnchorsMode++;
		}
	}
	anchors_mode = (nbValidControls == nbAnchorsMode);
	anchor_mode_button->set_pressed(anchors_mode);

	if (!selected_from_canvas) {
		drag_type = DRAG_NONE;
	}
	selected_from_canvas = false;
}

void CanvasItemEditor::edit(CanvasItem *p_canvas_item) {
	Array selection = editor_selection->get_selected_nodes();
	if (selection.size() != 1 || (Node *)selection[0] != p_canvas_item) {
		drag_type = DRAG_NONE;

		// Clear the selection
		editor_selection->clear(); //_clear_canvas_items();
		editor_selection->add_node(p_canvas_item);
	}
}

void CanvasItemEditor::_update_context_menu_stylebox() {
	// This must be called when the theme changes to follow the new accent color.
	Ref<StyleBoxFlat> context_menu_stylebox = memnew(StyleBoxFlat);
	const Color accent_color = EditorNode::get_singleton()->get_gui_base()->get_theme_color(SNAME("accent_color"), SNAME("Editor"));
	context_menu_stylebox->set_bg_color(accent_color * Color(1, 1, 1, 0.1));
	// Add an underline to the StyleBox, but prevent its minimum vertical size from changing.
	context_menu_stylebox->set_border_color(accent_color);
	context_menu_stylebox->set_border_width(SIDE_BOTTOM, Math::round(2 * EDSCALE));
	context_menu_stylebox->set_default_margin(SIDE_BOTTOM, 0);
	context_menu_container->add_theme_style_override("panel", context_menu_stylebox);
}

void CanvasItemEditor::_update_scrollbars() {
	updating_scroll = true;

	// Move the zoom buttons.
	Point2 controls_vb_begin = Point2(5, 5);
	controls_vb_begin += (show_rulers) ? Point2(RULER_WIDTH, RULER_WIDTH) : Point2();
	controls_vb->set_begin(controls_vb_begin);

	Size2 hmin = h_scroll->get_minimum_size();
	Size2 vmin = v_scroll->get_minimum_size();

	// Get the visible frame.
	Size2 screen_rect = Size2(ProjectSettings::get_singleton()->get("display/window/size/width"), ProjectSettings::get_singleton()->get("display/window/size/height"));
	Rect2 local_rect = Rect2(Point2(), viewport->get_size() - Size2(vmin.width, hmin.height));

	// Calculate scrollable area.
	Rect2 canvas_item_rect = Rect2(Point2(), screen_rect);
	if (editor->is_inside_tree() && editor->get_edited_scene()) {
		Rect2 content_rect = _get_encompassing_rect(editor->get_edited_scene());
		canvas_item_rect.expand_to(content_rect.position);
		canvas_item_rect.expand_to(content_rect.position + content_rect.size);
	}
	canvas_item_rect.size += screen_rect * 2;
	canvas_item_rect.position -= screen_rect;

	// Constraints the view offset and updates the scrollbars.
	Size2 size = viewport->get_size();
	Point2 begin = canvas_item_rect.position;
	Point2 end = canvas_item_rect.position + canvas_item_rect.size - local_rect.size / zoom;
	bool constrain_editor_view = bool(EditorSettings::get_singleton()->get("editors/2d/constrain_editor_view"));

	if (canvas_item_rect.size.height <= (local_rect.size.y / zoom)) {
		real_t centered = -(size.y / 2) / zoom + screen_rect.y / 2;
		if (constrain_editor_view && ABS(centered - previous_update_view_offset.y) < ABS(centered - view_offset.y)) {
			view_offset.y = previous_update_view_offset.y;
		}

		v_scroll->hide();
	} else {
		if (constrain_editor_view && view_offset.y > end.y && view_offset.y > previous_update_view_offset.y) {
			view_offset.y = MAX(end.y, previous_update_view_offset.y);
		}
		if (constrain_editor_view && view_offset.y < begin.y && view_offset.y < previous_update_view_offset.y) {
			view_offset.y = MIN(begin.y, previous_update_view_offset.y);
		}

		v_scroll->show();
		v_scroll->set_min(MIN(view_offset.y, begin.y));
		v_scroll->set_max(MAX(view_offset.y, end.y) + screen_rect.y);
		v_scroll->set_page(screen_rect.y);
	}

	if (canvas_item_rect.size.width <= (local_rect.size.x / zoom)) {
		real_t centered = -(size.x / 2) / zoom + screen_rect.x / 2;
		if (constrain_editor_view && ABS(centered - previous_update_view_offset.x) < ABS(centered - view_offset.x)) {
			view_offset.x = previous_update_view_offset.x;
		}

		h_scroll->hide();
	} else {
		if (constrain_editor_view && view_offset.x > end.x && view_offset.x > previous_update_view_offset.x) {
			view_offset.x = MAX(end.x, previous_update_view_offset.x);
		}
		if (constrain_editor_view && view_offset.x < begin.x && view_offset.x < previous_update_view_offset.x) {
			view_offset.x = MIN(begin.x, previous_update_view_offset.x);
		}

		h_scroll->show();
		h_scroll->set_min(MIN(view_offset.x, begin.x));
		h_scroll->set_max(MAX(view_offset.x, end.x) + screen_rect.x);
		h_scroll->set_page(screen_rect.x);
	}

	// Move and resize the scrollbars, avoiding overlap.
	if (is_layout_rtl()) {
		v_scroll->set_begin(Point2(0, (show_rulers) ? RULER_WIDTH : 0));
		v_scroll->set_end(Point2(vmin.width, size.height - (h_scroll->is_visible() ? hmin.height : 0)));
	} else {
		v_scroll->set_begin(Point2(size.width - vmin.width, (show_rulers) ? RULER_WIDTH : 0));
		v_scroll->set_end(Point2(size.width, size.height - (h_scroll->is_visible() ? hmin.height : 0)));
	}
	h_scroll->set_begin(Point2((show_rulers) ? RULER_WIDTH : 0, size.height - hmin.height));
	h_scroll->set_end(Point2(size.width - (v_scroll->is_visible() ? vmin.width : 0), size.height));

	// Calculate scrollable area.
	v_scroll->set_value(view_offset.y);
	h_scroll->set_value(view_offset.x);

	previous_update_view_offset = view_offset;
	updating_scroll = false;
}

void CanvasItemEditor::_update_scroll(real_t) {
	if (updating_scroll) {
		return;
	}

	view_offset.x = h_scroll->get_value();
	view_offset.y = v_scroll->get_value();
	viewport->update();
}

void CanvasItemEditor::_set_anchors_and_offsets_preset(Control::LayoutPreset p_preset) {
	List<Node *> selection = editor_selection->get_selected_node_list();

	undo_redo->create_action(TTR("Change Anchors and Offsets"));

	for (Node *E : selection) {
		Control *control = Object::cast_to<Control>(E);
		if (control) {
			undo_redo->add_do_method(control, "set_anchors_preset", p_preset);
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
					undo_redo->add_do_method(control, "set_offsets_preset", p_preset, Control::PRESET_MODE_KEEP_SIZE);
					break;
				case PRESET_LEFT_WIDE:
				case PRESET_TOP_WIDE:
				case PRESET_RIGHT_WIDE:
				case PRESET_BOTTOM_WIDE:
				case PRESET_VCENTER_WIDE:
				case PRESET_HCENTER_WIDE:
				case PRESET_WIDE:
					undo_redo->add_do_method(control, "set_offsets_preset", p_preset, Control::PRESET_MODE_MINSIZE);
					break;
			}
			undo_redo->add_undo_method(control, "_edit_set_state", control->_edit_get_state());
		}
	}

	undo_redo->commit_action();

	anchors_mode = false;
	anchor_mode_button->set_pressed(anchors_mode);
}

void CanvasItemEditor::_set_anchors_and_offsets_to_keep_ratio() {
	List<Node *> selection = editor_selection->get_selected_node_list();

	undo_redo->create_action(TTR("Change Anchors and Offsets"));

	for (Node *E : selection) {
		Control *control = Object::cast_to<Control>(E);
		if (control) {
			Point2 top_left_anchor = _position_to_anchor(control, Point2());
			Point2 bottom_right_anchor = _position_to_anchor(control, control->get_size());
			undo_redo->add_do_method(control, "set_anchor", SIDE_LEFT, top_left_anchor.x, false, true);
			undo_redo->add_do_method(control, "set_anchor", SIDE_RIGHT, bottom_right_anchor.x, false, true);
			undo_redo->add_do_method(control, "set_anchor", SIDE_TOP, top_left_anchor.y, false, true);
			undo_redo->add_do_method(control, "set_anchor", SIDE_BOTTOM, bottom_right_anchor.y, false, true);
			undo_redo->add_do_method(control, "set_meta", "_edit_use_anchors_", true);

			bool use_anchors = control->has_meta("_edit_use_anchors_") && control->get_meta("_edit_use_anchors_");
			undo_redo->add_undo_method(control, "_edit_set_state", control->_edit_get_state());
			undo_redo->add_undo_method(control, "set_meta", "_edit_use_anchors_", use_anchors);

			anchors_mode = true;
			anchor_mode_button->set_pressed(anchors_mode);
		}
	}

	undo_redo->commit_action();
}

void CanvasItemEditor::_set_anchors_preset(Control::LayoutPreset p_preset) {
	List<Node *> selection = editor_selection->get_selected_node_list();

	undo_redo->create_action(TTR("Change Anchors"));
	for (Node *E : selection) {
		Control *control = Object::cast_to<Control>(E);
		if (control) {
			undo_redo->add_do_method(control, "set_anchors_preset", p_preset);
			undo_redo->add_undo_method(control, "_edit_set_state", control->_edit_get_state());
		}
	}

	undo_redo->commit_action();
}

void CanvasItemEditor::_zoom_on_position(real_t p_zoom, Point2 p_position) {
	p_zoom = CLAMP(p_zoom, MIN_ZOOM, MAX_ZOOM);

	if (p_zoom == zoom) {
		zoom_widget->set_zoom(p_zoom);
		return;
	}

	real_t prev_zoom = zoom;
	zoom = p_zoom;

	view_offset += p_position / prev_zoom - p_position / zoom;

	// We want to align in-scene pixels to screen pixels, this prevents blurry rendering
	// in small details (texts, lines).
	// This correction adds a jitter movement when zooming, so we correct only when the
	// zoom factor is an integer. (in the other cases, all pixels won't be aligned anyway)
	const real_t closest_zoom_factor = Math::round(zoom);
	if (Math::is_zero_approx(zoom - closest_zoom_factor)) {
		// make sure scene pixel at view_offset is aligned on a screen pixel
		Vector2 view_offset_int = view_offset.floor();
		Vector2 view_offset_frac = view_offset - view_offset_int;
		view_offset = view_offset_int + (view_offset_frac * closest_zoom_factor).round() / closest_zoom_factor;
	}

	zoom_widget->set_zoom(zoom);
	update_viewport();
}

void CanvasItemEditor::_update_zoom(real_t p_zoom) {
	_zoom_on_position(p_zoom, viewport_scrollable->get_size() / 2.0);
}

void CanvasItemEditor::_button_toggle_smart_snap(bool p_status) {
	smart_snap_active = p_status;
	viewport->update();
}

void CanvasItemEditor::_button_toggle_grid_snap(bool p_status) {
	grid_snap_active = p_status;
	viewport->update();
}

void CanvasItemEditor::_button_override_camera(bool p_pressed) {
	EditorDebuggerNode *debugger = EditorDebuggerNode::get_singleton();

	if (p_pressed) {
		debugger->set_camera_override(EditorDebuggerNode::OVERRIDE_2D);
	} else {
		debugger->set_camera_override(EditorDebuggerNode::OVERRIDE_NONE);
	}
}

void CanvasItemEditor::_button_tool_select(int p_index) {
	Button *tb[TOOL_MAX] = { select_button, list_select_button, move_button, scale_button, rotate_button, pivot_button, pan_button, ruler_button };
	for (int i = 0; i < TOOL_MAX; i++) {
		tb[i]->set_pressed(i == p_index);
	}

	tool = (Tool)p_index;

	viewport->update();
	_update_cursor();
}

void CanvasItemEditor::_insert_animation_keys(bool p_location, bool p_rotation, bool p_scale, bool p_on_existing) {
	Map<Node *, Object *> &selection = editor_selection->get_selection();

	for (const KeyValue<Node *, Object *> &E : selection) {
		CanvasItem *canvas_item = Object::cast_to<CanvasItem>(E.key);
		if (!canvas_item || !canvas_item->is_visible_in_tree()) {
			continue;
		}

		if (canvas_item->get_viewport() != EditorNode::get_singleton()->get_scene_root()) {
			continue;
		}

		if (Object::cast_to<Node2D>(canvas_item)) {
			Node2D *n2d = Object::cast_to<Node2D>(canvas_item);

			if (key_pos && p_location) {
				AnimationPlayerEditor::get_singleton()->get_track_editor()->insert_node_value_key(n2d, "position", n2d->get_position(), p_on_existing);
			}
			if (key_rot && p_rotation) {
				AnimationPlayerEditor::get_singleton()->get_track_editor()->insert_node_value_key(n2d, "rotation", n2d->get_rotation(), p_on_existing);
			}
			if (key_scale && p_scale) {
				AnimationPlayerEditor::get_singleton()->get_track_editor()->insert_node_value_key(n2d, "scale", n2d->get_scale(), p_on_existing);
			}

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

					if (!n->get_parent_item()) {
						break;
					}
					n = Object::cast_to<Node2D>(n->get_parent_item());
				}

				if (has_chain && ik_chain.size()) {
					for (Node2D *&F : ik_chain) {
						if (key_pos) {
							AnimationPlayerEditor::get_singleton()->get_track_editor()->insert_node_value_key(F, "position", F->get_position(), p_on_existing);
						}
						if (key_rot) {
							AnimationPlayerEditor::get_singleton()->get_track_editor()->insert_node_value_key(F, "rotation", F->get_rotation(), p_on_existing);
						}
						if (key_scale) {
							AnimationPlayerEditor::get_singleton()->get_track_editor()->insert_node_value_key(F, "scale", F->get_scale(), p_on_existing);
						}
					}
				}
			}

		} else if (Object::cast_to<Control>(canvas_item)) {
			Control *ctrl = Object::cast_to<Control>(canvas_item);

			if (key_pos) {
				AnimationPlayerEditor::get_singleton()->get_track_editor()->insert_node_value_key(ctrl, "rect_position", ctrl->get_position(), p_on_existing);
			}
			if (key_rot) {
				AnimationPlayerEditor::get_singleton()->get_track_editor()->insert_node_value_key(ctrl, "rect_rotation", ctrl->get_rotation(), p_on_existing);
			}
			if (key_scale) {
				AnimationPlayerEditor::get_singleton()->get_track_editor()->insert_node_value_key(ctrl, "rect_size", ctrl->get_size(), p_on_existing);
			}
		}
	}
}

void CanvasItemEditor::_button_toggle_anchor_mode(bool p_status) {
	List<CanvasItem *> selection = _get_edited_canvas_items(false, false);
	for (CanvasItem *E : selection) {
		Control *control = Object::cast_to<Control>(E);
		if (!control || Object::cast_to<Container>(control->get_parent())) {
			continue;
		}

		control->set_meta("_edit_use_anchors_", p_status);
	}

	anchors_mode = p_status;
	viewport->update();
}

void CanvasItemEditor::_update_override_camera_button(bool p_game_running) {
	if (p_game_running) {
		override_camera_button->set_disabled(false);
		override_camera_button->set_tooltip(TTR("Project Camera Override\nOverrides the running project's camera with the editor viewport camera."));
	} else {
		override_camera_button->set_disabled(true);
		override_camera_button->set_pressed(false);
		override_camera_button->set_tooltip(TTR("Project Camera Override\nNo project instance running. Run the project from the editor to use this feature."));
	}
}

void CanvasItemEditor::_popup_callback(int p_op) {
	last_option = MenuOption(p_op);
	switch (p_op) {
		case SHOW_GRID: {
			show_grid = !show_grid;
			int idx = view_menu->get_popup()->get_item_index(SHOW_GRID);
			view_menu->get_popup()->set_item_checked(idx, show_grid);
			viewport->update();
		} break;
		case SHOW_ORIGIN: {
			show_origin = !show_origin;
			int idx = view_menu->get_popup()->get_item_index(SHOW_ORIGIN);
			view_menu->get_popup()->set_item_checked(idx, show_origin);
			viewport->update();
		} break;
		case SHOW_VIEWPORT: {
			show_viewport = !show_viewport;
			int idx = view_menu->get_popup()->get_item_index(SHOW_VIEWPORT);
			view_menu->get_popup()->set_item_checked(idx, show_viewport);
			viewport->update();
		} break;
		case SHOW_EDIT_LOCKS: {
			show_edit_locks = !show_edit_locks;
			int idx = view_menu->get_popup()->get_item_index(SHOW_EDIT_LOCKS);
			view_menu->get_popup()->set_item_checked(idx, show_edit_locks);
			viewport->update();
		} break;
		case SHOW_TRANSFORMATION_GIZMOS: {
			show_transformation_gizmos = !show_transformation_gizmos;
			int idx = view_menu->get_popup()->get_item_index(SHOW_TRANSFORMATION_GIZMOS);
			view_menu->get_popup()->set_item_checked(idx, show_transformation_gizmos);
			viewport->update();
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
		case SNAP_USE_NODE_CENTER: {
			snap_node_center = !snap_node_center;
			int idx = smartsnap_config_popup->get_item_index(SNAP_USE_NODE_CENTER);
			smartsnap_config_popup->set_item_checked(idx, snap_node_center);
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
		case SNAP_USE_ROTATION: {
			snap_rotation = !snap_rotation;
			int idx = snap_config_menu->get_popup()->get_item_index(SNAP_USE_ROTATION);
			snap_config_menu->get_popup()->set_item_checked(idx, snap_rotation);
		} break;
		case SNAP_USE_SCALE: {
			snap_scale = !snap_scale;
			int idx = snap_config_menu->get_popup()->get_item_index(SNAP_USE_SCALE);
			snap_config_menu->get_popup()->set_item_checked(idx, snap_scale);
		} break;
		case SNAP_RELATIVE: {
			snap_relative = !snap_relative;
			int idx = snap_config_menu->get_popup()->get_item_index(SNAP_RELATIVE);
			snap_config_menu->get_popup()->set_item_checked(idx, snap_relative);
			viewport->update();
		} break;
		case SNAP_USE_PIXEL: {
			snap_pixel = !snap_pixel;
			int idx = snap_config_menu->get_popup()->get_item_index(SNAP_USE_PIXEL);
			snap_config_menu->get_popup()->set_item_checked(idx, snap_pixel);
		} break;
		case SNAP_CONFIGURE: {
			((SnapDialog *)snap_dialog)->set_fields(grid_offset, grid_step, primary_grid_steps, snap_rotation_offset, snap_rotation_step, snap_scale_step);
			snap_dialog->popup_centered(Size2(220, 160) * EDSCALE);
		} break;
		case SKELETON_SHOW_BONES: {
			List<Node *> selection = editor_selection->get_selected_node_list();
			for (Node *E : selection) {
				// Add children nodes so they are processed
				for (int child = 0; child < E->get_child_count(); child++) {
					selection.push_back(E->get_child(child));
				}

				Bone2D *bone_2d = Object::cast_to<Bone2D>(E);
				if (!bone_2d || !bone_2d->is_inside_tree()) {
					continue;
				}
				bone_2d->_editor_set_show_bone_gizmo(!bone_2d->_editor_get_show_bone_gizmo());
			}
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
			_update_scrollbars();
			viewport->update();
		} break;
		case SHOW_GUIDES: {
			show_guides = !show_guides;
			int idx = view_menu->get_popup()->get_item_index(SHOW_GUIDES);
			view_menu->get_popup()->set_item_checked(idx, show_guides);
			viewport->update();
		} break;
		case LOCK_SELECTED: {
			undo_redo->create_action(TTR("Lock Selected"));

			List<Node *> selection = editor_selection->get_selected_node_list();
			for (Node *E : selection) {
				CanvasItem *canvas_item = Object::cast_to<CanvasItem>(E);
				if (!canvas_item || !canvas_item->is_inside_tree()) {
					continue;
				}
				if (canvas_item->get_viewport() != EditorNode::get_singleton()->get_scene_root()) {
					continue;
				}

				undo_redo->add_do_method(canvas_item, "set_meta", "_edit_lock_", true);
				undo_redo->add_undo_method(canvas_item, "remove_meta", "_edit_lock_");
				undo_redo->add_do_method(this, "emit_signal", "item_lock_status_changed");
				undo_redo->add_undo_method(this, "emit_signal", "item_lock_status_changed");
			}
			undo_redo->add_do_method(viewport, "update", Variant());
			undo_redo->add_undo_method(viewport, "update", Variant());
			undo_redo->commit_action();
		} break;
		case UNLOCK_SELECTED: {
			undo_redo->create_action(TTR("Unlock Selected"));

			List<Node *> selection = editor_selection->get_selected_node_list();
			for (Node *E : selection) {
				CanvasItem *canvas_item = Object::cast_to<CanvasItem>(E);
				if (!canvas_item || !canvas_item->is_inside_tree()) {
					continue;
				}
				if (canvas_item->get_viewport() != EditorNode::get_singleton()->get_scene_root()) {
					continue;
				}

				undo_redo->add_do_method(canvas_item, "remove_meta", "_edit_lock_");
				undo_redo->add_undo_method(canvas_item, "set_meta", "_edit_lock_", true);
				undo_redo->add_do_method(this, "emit_signal", "item_lock_status_changed");
				undo_redo->add_undo_method(this, "emit_signal", "item_lock_status_changed");
			}
			undo_redo->add_do_method(viewport, "update", Variant());
			undo_redo->add_undo_method(viewport, "update", Variant());
			undo_redo->commit_action();
		} break;
		case GROUP_SELECTED: {
			undo_redo->create_action(TTR("Group Selected"));

			List<Node *> selection = editor_selection->get_selected_node_list();
			for (Node *E : selection) {
				CanvasItem *canvas_item = Object::cast_to<CanvasItem>(E);
				if (!canvas_item || !canvas_item->is_inside_tree()) {
					continue;
				}
				if (canvas_item->get_viewport() != EditorNode::get_singleton()->get_scene_root()) {
					continue;
				}

				undo_redo->add_do_method(canvas_item, "set_meta", "_edit_group_", true);
				undo_redo->add_undo_method(canvas_item, "remove_meta", "_edit_group_");
				undo_redo->add_do_method(this, "emit_signal", "item_group_status_changed");
				undo_redo->add_undo_method(this, "emit_signal", "item_group_status_changed");
			}
			undo_redo->add_do_method(viewport, "update", Variant());
			undo_redo->add_undo_method(viewport, "update", Variant());
			undo_redo->commit_action();
		} break;
		case UNGROUP_SELECTED: {
			undo_redo->create_action(TTR("Ungroup Selected"));

			List<Node *> selection = editor_selection->get_selected_node_list();
			for (Node *E : selection) {
				CanvasItem *canvas_item = Object::cast_to<CanvasItem>(E);
				if (!canvas_item || !canvas_item->is_inside_tree()) {
					continue;
				}
				if (canvas_item->get_viewport() != EditorNode::get_singleton()->get_scene_root()) {
					continue;
				}

				undo_redo->add_do_method(canvas_item, "remove_meta", "_edit_group_");
				undo_redo->add_undo_method(canvas_item, "set_meta", "_edit_group_", true);
				undo_redo->add_do_method(this, "emit_signal", "item_group_status_changed");
				undo_redo->add_undo_method(this, "emit_signal", "item_group_status_changed");
			}
			undo_redo->add_do_method(viewport, "update", Variant());
			undo_redo->add_undo_method(viewport, "update", Variant());
			undo_redo->commit_action();
		} break;
		case ANCHORS_AND_OFFSETS_PRESET_TOP_LEFT: {
			_set_anchors_and_offsets_preset(PRESET_TOP_LEFT);
		} break;
		case ANCHORS_AND_OFFSETS_PRESET_TOP_RIGHT: {
			_set_anchors_and_offsets_preset(PRESET_TOP_RIGHT);
		} break;
		case ANCHORS_AND_OFFSETS_PRESET_BOTTOM_LEFT: {
			_set_anchors_and_offsets_preset(PRESET_BOTTOM_LEFT);
		} break;
		case ANCHORS_AND_OFFSETS_PRESET_BOTTOM_RIGHT: {
			_set_anchors_and_offsets_preset(PRESET_BOTTOM_RIGHT);
		} break;
		case ANCHORS_AND_OFFSETS_PRESET_CENTER_LEFT: {
			_set_anchors_and_offsets_preset(PRESET_CENTER_LEFT);
		} break;
		case ANCHORS_AND_OFFSETS_PRESET_CENTER_RIGHT: {
			_set_anchors_and_offsets_preset(PRESET_CENTER_RIGHT);
		} break;
		case ANCHORS_AND_OFFSETS_PRESET_CENTER_TOP: {
			_set_anchors_and_offsets_preset(PRESET_CENTER_TOP);
		} break;
		case ANCHORS_AND_OFFSETS_PRESET_CENTER_BOTTOM: {
			_set_anchors_and_offsets_preset(PRESET_CENTER_BOTTOM);
		} break;
		case ANCHORS_AND_OFFSETS_PRESET_CENTER: {
			_set_anchors_and_offsets_preset(PRESET_CENTER);
		} break;
		case ANCHORS_AND_OFFSETS_PRESET_TOP_WIDE: {
			_set_anchors_and_offsets_preset(PRESET_TOP_WIDE);
		} break;
		case ANCHORS_AND_OFFSETS_PRESET_LEFT_WIDE: {
			_set_anchors_and_offsets_preset(PRESET_LEFT_WIDE);
		} break;
		case ANCHORS_AND_OFFSETS_PRESET_RIGHT_WIDE: {
			_set_anchors_and_offsets_preset(PRESET_RIGHT_WIDE);
		} break;
		case ANCHORS_AND_OFFSETS_PRESET_BOTTOM_WIDE: {
			_set_anchors_and_offsets_preset(PRESET_BOTTOM_WIDE);
		} break;
		case ANCHORS_AND_OFFSETS_PRESET_VCENTER_WIDE: {
			_set_anchors_and_offsets_preset(PRESET_VCENTER_WIDE);
		} break;
		case ANCHORS_AND_OFFSETS_PRESET_HCENTER_WIDE: {
			_set_anchors_and_offsets_preset(PRESET_HCENTER_WIDE);
		} break;
		case ANCHORS_AND_OFFSETS_PRESET_WIDE: {
			_set_anchors_and_offsets_preset(Control::PRESET_WIDE);
		} break;
		case ANCHORS_AND_OFFSETS_PRESET_KEEP_RATIO: {
			_set_anchors_and_offsets_to_keep_ratio();
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

			_insert_animation_keys(true, true, true, existing);

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
		case ANIM_COPY_POSE: {
			pose_clipboard.clear();

			Map<Node *, Object *> &selection = editor_selection->get_selection();

			for (const KeyValue<Node *, Object *> &E : selection) {
				CanvasItem *canvas_item = Object::cast_to<CanvasItem>(E.key);
				if (!canvas_item || !canvas_item->is_visible_in_tree()) {
					continue;
				}

				if (canvas_item->get_viewport() != EditorNode::get_singleton()->get_scene_root()) {
					continue;
				}

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
			if (!pose_clipboard.size()) {
				break;
			}

			undo_redo->create_action(TTR("Paste Pose"));
			for (const PoseClipboard &E : pose_clipboard) {
				Node2D *n2d = Object::cast_to<Node2D>(ObjectDB::get_instance(E.id));
				if (!n2d) {
					continue;
				}
				undo_redo->add_do_method(n2d, "set_position", E.pos);
				undo_redo->add_do_method(n2d, "set_rotation", E.rot);
				undo_redo->add_do_method(n2d, "set_scale", E.scale);
				undo_redo->add_undo_method(n2d, "set_position", n2d->get_position());
				undo_redo->add_undo_method(n2d, "set_rotation", n2d->get_rotation());
				undo_redo->add_undo_method(n2d, "set_scale", n2d->get_scale());
			}
			undo_redo->commit_action();

		} break;
		case ANIM_CLEAR_POSE: {
			Map<Node *, Object *> &selection = editor_selection->get_selection();

			for (const KeyValue<Node *, Object *> &E : selection) {
				CanvasItem *canvas_item = Object::cast_to<CanvasItem>(E.key);
				if (!canvas_item || !canvas_item->is_visible_in_tree()) {
					continue;
				}

				if (canvas_item->get_viewport() != EditorNode::get_singleton()->get_scene_root()) {
					continue;
				}

				if (Object::cast_to<Node2D>(canvas_item)) {
					Node2D *n2d = Object::cast_to<Node2D>(canvas_item);

					if (key_pos) {
						n2d->set_position(Vector2());
					}
					if (key_rot) {
						n2d->set_rotation(0);
					}
					if (key_scale) {
						n2d->set_scale(Vector2(1, 1));
					}
				} else if (Object::cast_to<Control>(canvas_item)) {
					Control *ctrl = Object::cast_to<Control>(canvas_item);

					if (key_pos) {
						ctrl->set_position(Point2());
					}
				}
			}

		} break;
		case CLEAR_GUIDES: {
			Node *const root = EditorNode::get_singleton()->get_edited_scene();

			if (root && (root->has_meta("_edit_horizontal_guides_") || root->has_meta("_edit_vertical_guides_"))) {
				undo_redo->create_action(TTR("Clear Guides"));
				if (root->has_meta("_edit_horizontal_guides_")) {
					Array hguides = root->get_meta("_edit_horizontal_guides_");

					undo_redo->add_do_method(root, "remove_meta", "_edit_horizontal_guides_");
					undo_redo->add_undo_method(root, "set_meta", "_edit_horizontal_guides_", hguides);
				}
				if (root->has_meta("_edit_vertical_guides_")) {
					Array vguides = root->get_meta("_edit_vertical_guides_");

					undo_redo->add_do_method(root, "remove_meta", "_edit_vertical_guides_");
					undo_redo->add_undo_method(root, "set_meta", "_edit_vertical_guides_", vguides);
				}
				undo_redo->add_undo_method(viewport, "update");
				undo_redo->commit_action();
			}

		} break;
		case VIEW_CENTER_TO_SELECTION:
		case VIEW_FRAME_TO_SELECTION: {
			_focus_selection(p_op);

		} break;
		case PREVIEW_CANVAS_SCALE: {
			bool preview = view_menu->get_popup()->is_item_checked(view_menu->get_popup()->get_item_index(PREVIEW_CANVAS_SCALE));
			preview = !preview;
			RS::get_singleton()->canvas_set_disable_scale(!preview);
			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(PREVIEW_CANVAS_SCALE), preview);

		} break;
		case SKELETON_MAKE_BONES: {
			Map<Node *, Object *> &selection = editor_selection->get_selection();
			Node *editor_root = EditorNode::get_singleton()->get_edited_scene()->get_tree()->get_edited_scene_root();

			undo_redo->create_action(TTR("Create Custom Bone2D(s) from Node(s)"));
			for (const KeyValue<Node *, Object *> &E : selection) {
				Node2D *n2d = Object::cast_to<Node2D>(E.key);

				Bone2D *new_bone = memnew(Bone2D);
				String new_bone_name = n2d->get_name();
				new_bone_name += "Bone2D";
				new_bone->set_name(new_bone_name);
				new_bone->set_transform(n2d->get_transform());

				Node *n2d_parent = n2d->get_parent();
				if (!n2d_parent) {
					continue;
				}

				undo_redo->add_do_method(n2d_parent, "add_child", new_bone);
				undo_redo->add_do_method(n2d_parent, "remove_child", n2d);
				undo_redo->add_do_method(new_bone, "add_child", n2d);
				undo_redo->add_do_method(n2d, "set_transform", Transform2D());
				undo_redo->add_do_method(this, "_set_owner_for_node_and_children", new_bone, editor_root);

				undo_redo->add_undo_method(new_bone, "remove_child", n2d);
				undo_redo->add_undo_method(n2d_parent, "add_child", n2d);
				undo_redo->add_undo_method(n2d, "set_transform", new_bone->get_transform());
				undo_redo->add_undo_method(new_bone, "queue_free");
				undo_redo->add_undo_method(this, "_set_owner_for_node_and_children", n2d, editor_root);
			}
			undo_redo->commit_action();

		} break;
	}
}

void CanvasItemEditor::_set_owner_for_node_and_children(Node *p_node, Node *p_owner) {
	p_node->set_owner(p_owner);
	for (int i = 0; i < p_node->get_child_count(); i++) {
		_set_owner_for_node_and_children(p_node->get_child(i), p_owner);
	}
}

void CanvasItemEditor::_focus_selection(int p_op) {
	Vector2 center(0.f, 0.f);
	Rect2 rect;
	int count = 0;

	Map<Node *, Object *> &selection = editor_selection->get_selection();
	for (const KeyValue<Node *, Object *> &E : selection) {
		CanvasItem *canvas_item = Object::cast_to<CanvasItem>(E.key);
		if (!canvas_item) {
			continue;
		}
		if (canvas_item->get_viewport() != EditorNode::get_singleton()->get_scene_root()) {
			continue;
		}

		// counting invisible items, for now
		//if (!canvas_item->is_visible_in_tree()) continue;
		++count;

		Rect2 item_rect;
		if (canvas_item->_edit_use_rect()) {
			item_rect = canvas_item->_edit_get_rect();
		} else {
			item_rect = Rect2();
		}

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

	if (p_op == VIEW_CENTER_TO_SELECTION) {
		center = rect.get_center();
		Vector2 offset = viewport->get_size() / 2 - editor->get_scene_root()->get_global_canvas_transform().xform(center);
		view_offset -= (offset / zoom).round();
		update_viewport();

	} else { // VIEW_FRAME_TO_SELECTION

		if (rect.size.x > CMP_EPSILON && rect.size.y > CMP_EPSILON) {
			real_t scale_x = viewport->get_size().x / rect.size.x;
			real_t scale_y = viewport->get_size().y / rect.size.y;
			zoom = scale_x < scale_y ? scale_x : scale_y;
			zoom *= 0.90;
			viewport->update();
			zoom_widget->set_zoom(zoom);
			call_deferred(SNAME("_popup_callback"), VIEW_CENTER_TO_SELECTION);
		}
	}
}

void CanvasItemEditor::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_update_override_camera_button", "game_running"), &CanvasItemEditor::_update_override_camera_button);
	ClassDB::bind_method("_get_editor_data", &CanvasItemEditor::_get_editor_data);

	ClassDB::bind_method(D_METHOD("set_state"), &CanvasItemEditor::set_state);
	ClassDB::bind_method(D_METHOD("update_viewport"), &CanvasItemEditor::update_viewport);
	ClassDB::bind_method(D_METHOD("_zoom_on_position"), &CanvasItemEditor::_zoom_on_position);

	ClassDB::bind_method("_set_owner_for_node_and_children", &CanvasItemEditor::_set_owner_for_node_and_children);

	ADD_SIGNAL(MethodInfo("item_lock_status_changed"));
	ADD_SIGNAL(MethodInfo("item_group_status_changed"));
}

Dictionary CanvasItemEditor::get_state() const {
	Dictionary state;
	// Take the editor scale into account.
	state["zoom"] = zoom / MAX(1, EDSCALE);
	state["ofs"] = view_offset;
	state["grid_offset"] = grid_offset;
	state["grid_step"] = grid_step;
	state["primary_grid_steps"] = primary_grid_steps;
	state["snap_rotation_offset"] = snap_rotation_offset;
	state["snap_rotation_step"] = snap_rotation_step;
	state["snap_scale_step"] = snap_scale_step;
	state["smart_snap_active"] = smart_snap_active;
	state["grid_snap_active"] = grid_snap_active;
	state["snap_node_parent"] = snap_node_parent;
	state["snap_node_anchors"] = snap_node_anchors;
	state["snap_node_sides"] = snap_node_sides;
	state["snap_node_center"] = snap_node_center;
	state["snap_other_nodes"] = snap_other_nodes;
	state["snap_guides"] = snap_guides;
	state["show_grid"] = show_grid;
	state["show_origin"] = show_origin;
	state["show_viewport"] = show_viewport;
	state["show_rulers"] = show_rulers;
	state["show_guides"] = show_guides;
	state["show_helpers"] = show_helpers;
	state["show_zoom_control"] = zoom_widget->is_visible();
	state["show_edit_locks"] = show_edit_locks;
	state["show_transformation_gizmos"] = show_transformation_gizmos;
	state["snap_rotation"] = snap_rotation;
	state["snap_scale"] = snap_scale;
	state["snap_relative"] = snap_relative;
	state["snap_pixel"] = snap_pixel;
	return state;
}

void CanvasItemEditor::set_state(const Dictionary &p_state) {
	bool update_scrollbars = false;
	Dictionary state = p_state;
	if (state.has("zoom")) {
		// Compensate the editor scale, so that the editor scale can be changed
		// and the zoom level will still be the same (relative to the editor scale).
		zoom = real_t(p_state["zoom"]) * MAX(1, EDSCALE);
		zoom_widget->set_zoom(zoom);
	}

	if (state.has("ofs")) {
		view_offset = p_state["ofs"];
		previous_update_view_offset = view_offset;
		update_scrollbars = true;
	}

	if (state.has("grid_offset")) {
		grid_offset = state["grid_offset"];
	}

	if (state.has("grid_step")) {
		grid_step = state["grid_step"];
	}

	if (state.has("primary_grid_steps")) {
		primary_grid_steps = state["primary_grid_steps"];
	}

	if (state.has("snap_rotation_step")) {
		snap_rotation_step = state["snap_rotation_step"];
	}

	if (state.has("snap_rotation_offset")) {
		snap_rotation_offset = state["snap_rotation_offset"];
	}

	if (state.has("snap_scale_step")) {
		snap_scale_step = state["snap_scale_step"];
	}

	if (state.has("smart_snap_active")) {
		smart_snap_active = state["smart_snap_active"];
		smart_snap_button->set_pressed(smart_snap_active);
	}

	if (state.has("grid_snap_active")) {
		grid_snap_active = state["grid_snap_active"];
		grid_snap_button->set_pressed(grid_snap_active);
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

	if (state.has("snap_node_center")) {
		snap_node_center = state["snap_node_center"];
		int idx = smartsnap_config_popup->get_item_index(SNAP_USE_NODE_CENTER);
		smartsnap_config_popup->set_item_checked(idx, snap_node_center);
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

	if (state.has("show_grid")) {
		show_grid = state["show_grid"];
		int idx = view_menu->get_popup()->get_item_index(SHOW_GRID);
		view_menu->get_popup()->set_item_checked(idx, show_grid);
	}

	if (state.has("show_origin")) {
		show_origin = state["show_origin"];
		int idx = view_menu->get_popup()->get_item_index(SHOW_ORIGIN);
		view_menu->get_popup()->set_item_checked(idx, show_origin);
	}

	if (state.has("show_viewport")) {
		show_viewport = state["show_viewport"];
		int idx = view_menu->get_popup()->get_item_index(SHOW_VIEWPORT);
		view_menu->get_popup()->set_item_checked(idx, show_viewport);
	}

	if (state.has("show_rulers")) {
		show_rulers = state["show_rulers"];
		int idx = view_menu->get_popup()->get_item_index(SHOW_RULERS);
		view_menu->get_popup()->set_item_checked(idx, show_rulers);
		update_scrollbars = true;
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

	if (state.has("show_edit_locks")) {
		show_edit_locks = state["show_edit_locks"];
		int idx = view_menu->get_popup()->get_item_index(SHOW_EDIT_LOCKS);
		view_menu->get_popup()->set_item_checked(idx, show_edit_locks);
	}

	if (state.has("show_transformation_gizmos")) {
		show_transformation_gizmos = state["show_transformation_gizmos"];
		int idx = view_menu->get_popup()->get_item_index(SHOW_TRANSFORMATION_GIZMOS);
		view_menu->get_popup()->set_item_checked(idx, show_transformation_gizmos);
	}

	if (state.has("show_zoom_control")) {
		// This one is not user-controllable, but instrumentable
		zoom_widget->set_visible(state["show_zoom_control"]);
	}

	if (state.has("snap_rotation")) {
		snap_rotation = state["snap_rotation"];
		int idx = snap_config_menu->get_popup()->get_item_index(SNAP_USE_ROTATION);
		snap_config_menu->get_popup()->set_item_checked(idx, snap_rotation);
	}

	if (state.has("snap_scale")) {
		snap_scale = state["snap_scale"];
		int idx = snap_config_menu->get_popup()->get_item_index(SNAP_USE_SCALE);
		snap_config_menu->get_popup()->set_item_checked(idx, snap_scale);
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

	if (update_scrollbars) {
		_update_scrollbars();
	}
	viewport->update();
}

void CanvasItemEditor::add_control_to_menu_panel(Control *p_control) {
	ERR_FAIL_COND(!p_control);

	hbc_context_menu->add_child(p_control);
}

void CanvasItemEditor::remove_control_from_menu_panel(Control *p_control) {
	hbc_context_menu->remove_child(p_control);
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
	key_pos = true;
	key_rot = true;
	key_scale = false;

	show_grid = false;
	show_origin = true;
	show_viewport = true;
	show_helpers = false;
	show_rulers = true;
	show_guides = true;
	show_transformation_gizmos = true;
	show_edit_locks = true;
	zoom = 1.0 / MAX(1, EDSCALE);
	view_offset = Point2(-150 - RULER_WIDTH, -95 - RULER_WIDTH);
	previous_update_view_offset = view_offset; // Moves the view a little bit to the left so that (0,0) is visible. The values a relative to a 16/10 screen
	grid_offset = Point2();
	grid_step = Point2(8, 8); // A power-of-two value works better as a default
	primary_grid_steps = 8; // A power-of-two value works better as a default
	grid_step_multiplier = 0;
	snap_rotation_offset = 0;
	snap_rotation_step = Math::deg2rad(15.0);
	snap_scale_step = 0.1f;
	smart_snap_active = false;
	grid_snap_active = false;
	snap_node_parent = true;
	snap_node_anchors = true;
	snap_node_sides = true;
	snap_node_center = true;
	snap_other_nodes = true;
	snap_guides = true;
	snap_rotation = false;
	snap_scale = false;
	snap_relative = false;
	// Enable pixel snapping even if pixel snap rendering is disabled in the Project Settings.
	// This results in crisper visuals by preventing 2D nodes from being placed at subpixel coordinates.
	snap_pixel = true;
	snap_target[0] = SNAP_TARGET_NONE;
	snap_target[1] = SNAP_TARGET_NONE;

	selected_from_canvas = false;
	anchors_mode = false;

	drag_type = DRAG_NONE;
	drag_from = Vector2();
	drag_to = Vector2();
	dragged_guide_pos = Point2();
	dragged_guide_index = -1;
	is_hovering_h_guide = false;
	is_hovering_v_guide = false;
	panning = false;
	pan_pressed = false;

	ruler_tool_active = false;
	ruler_tool_origin = Point2();

	bone_last_frame = 0;

	tool = TOOL_SELECT;
	undo_redo = p_editor->get_undo_redo();
	editor = p_editor;
	editor_selection = p_editor->get_editor_selection();
	editor_selection->add_editor_plugin(this);
	editor_selection->connect("selection_changed", callable_mp((CanvasItem *)this, &CanvasItem::update));
	editor_selection->connect("selection_changed", callable_mp(this, &CanvasItemEditor::_selection_changed));

	editor->get_scene_tree_dock()->connect("node_created", callable_mp(this, &CanvasItemEditor::_node_created));
	editor->get_scene_tree_dock()->connect("add_node_used", callable_mp(this, &CanvasItemEditor::_reset_create_position));

	editor->call_deferred(SNAME("connect"), "play_pressed", Callable(this, "_update_override_camera_button"), make_binds(true));
	editor->call_deferred(SNAME("connect"), "stop_pressed", Callable(this, "_update_override_camera_button"), make_binds(false));

	hb = memnew(HBoxContainer);
	add_child(hb);
	hb->set_anchors_and_offsets_preset(Control::PRESET_WIDE);

	bottom_split = memnew(VSplitContainer);
	add_child(bottom_split);
	bottom_split->set_v_size_flags(Control::SIZE_EXPAND_FILL);

	palette_split = memnew(HSplitContainer);
	bottom_split->add_child(palette_split);
	palette_split->set_v_size_flags(Control::SIZE_EXPAND_FILL);

	viewport_scrollable = memnew(Control);
	palette_split->add_child(viewport_scrollable);
	viewport_scrollable->set_mouse_filter(MOUSE_FILTER_PASS);
	viewport_scrollable->set_clip_contents(true);
	viewport_scrollable->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	viewport_scrollable->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	viewport_scrollable->connect("draw", callable_mp(this, &CanvasItemEditor::_update_scrollbars));

	SubViewportContainer *scene_tree = memnew(SubViewportContainer);
	viewport_scrollable->add_child(scene_tree);
	scene_tree->set_stretch(true);
	scene_tree->set_anchors_and_offsets_preset(Control::PRESET_WIDE);
	scene_tree->add_child(p_editor->get_scene_root());

	controls_vb = memnew(VBoxContainer);
	controls_vb->set_begin(Point2(5, 5));

	viewport = memnew(CanvasItemEditorViewport(p_editor, this));
	viewport_scrollable->add_child(viewport);
	viewport->set_mouse_filter(MOUSE_FILTER_PASS);
	viewport->set_anchors_and_offsets_preset(Control::PRESET_WIDE);
	viewport->set_clip_contents(true);
	viewport->set_focus_mode(FOCUS_ALL);
	viewport->connect("draw", callable_mp(this, &CanvasItemEditor::_draw_viewport));
	viewport->connect("gui_input", callable_mp(this, &CanvasItemEditor::_gui_input_viewport));

	h_scroll = memnew(HScrollBar);
	viewport->add_child(h_scroll);
	h_scroll->connect("value_changed", callable_mp(this, &CanvasItemEditor::_update_scroll));
	h_scroll->hide();

	v_scroll = memnew(VScrollBar);
	viewport->add_child(v_scroll);
	v_scroll->connect("value_changed", callable_mp(this, &CanvasItemEditor::_update_scroll));
	v_scroll->hide();

	viewport->add_child(controls_vb);

	zoom_widget = memnew(EditorZoomWidget);
	controls_vb->add_child(zoom_widget);
	zoom_widget->set_anchors_and_offsets_preset(Control::PRESET_TOP_LEFT, Control::PRESET_MODE_MINSIZE, 2 * EDSCALE);
	zoom_widget->connect("zoom_changed", callable_mp(this, &CanvasItemEditor::_update_zoom));

	updating_scroll = false;

	// Add some margin to the left for better aesthetics.
	// This prevents the first button's hover/pressed effect from "touching" the panel's border,
	// which looks ugly.
	Control *margin_left = memnew(Control);
	hb->add_child(margin_left);
	margin_left->set_custom_minimum_size(Size2(2, 0) * EDSCALE);

	select_button = memnew(Button);
	select_button->set_flat(true);
	hb->add_child(select_button);
	select_button->set_toggle_mode(true);
	select_button->connect("pressed", callable_mp(this, &CanvasItemEditor::_button_tool_select), make_binds(TOOL_SELECT));
	select_button->set_pressed(true);
	select_button->set_shortcut(ED_SHORTCUT("canvas_item_editor/select_mode", TTR("Select Mode"), Key::Q));
	select_button->set_shortcut_context(this);
	select_button->set_tooltip(keycode_get_string((Key)KeyModifierMask::CMD) + TTR("Drag: Rotate selected node around pivot.") + "\n" + TTR("Alt+Drag: Move selected node.") + "\n" + keycode_get_string((Key)KeyModifierMask::CMD) + TTR("Alt+Drag: Scale selected node.") + "\n" + TTR("V: Set selected node's pivot position.") + "\n" + TTR("Alt+RMB: Show list of all nodes at position clicked, including locked.") + "\n" + keycode_get_string((Key)KeyModifierMask::CMD) + TTR("RMB: Add node at position clicked."));

	hb->add_child(memnew(VSeparator));

	move_button = memnew(Button);
	move_button->set_flat(true);
	hb->add_child(move_button);
	move_button->set_toggle_mode(true);
	move_button->connect("pressed", callable_mp(this, &CanvasItemEditor::_button_tool_select), make_binds(TOOL_MOVE));
	move_button->set_shortcut(ED_SHORTCUT("canvas_item_editor/move_mode", TTR("Move Mode"), Key::W));
	move_button->set_shortcut_context(this);
	move_button->set_tooltip(TTR("Move Mode"));

	rotate_button = memnew(Button);
	rotate_button->set_flat(true);
	hb->add_child(rotate_button);
	rotate_button->set_toggle_mode(true);
	rotate_button->connect("pressed", callable_mp(this, &CanvasItemEditor::_button_tool_select), make_binds(TOOL_ROTATE));
	rotate_button->set_shortcut(ED_SHORTCUT("canvas_item_editor/rotate_mode", TTR("Rotate Mode"), Key::E));
	rotate_button->set_shortcut_context(this);
	rotate_button->set_tooltip(TTR("Rotate Mode"));

	scale_button = memnew(Button);
	scale_button->set_flat(true);
	hb->add_child(scale_button);
	scale_button->set_toggle_mode(true);
	scale_button->connect("pressed", callable_mp(this, &CanvasItemEditor::_button_tool_select), make_binds(TOOL_SCALE));
	scale_button->set_shortcut(ED_SHORTCUT("canvas_item_editor/scale_mode", TTR("Scale Mode"), Key::S));
	scale_button->set_shortcut_context(this);
	scale_button->set_tooltip(TTR("Shift: Scale proportionally."));

	hb->add_child(memnew(VSeparator));

	list_select_button = memnew(Button);
	list_select_button->set_flat(true);
	hb->add_child(list_select_button);
	list_select_button->set_toggle_mode(true);
	list_select_button->connect("pressed", callable_mp(this, &CanvasItemEditor::_button_tool_select), make_binds(TOOL_LIST_SELECT));
	list_select_button->set_tooltip(TTR("Show list of selectable nodes at position clicked."));

	pivot_button = memnew(Button);
	pivot_button->set_flat(true);
	hb->add_child(pivot_button);
	pivot_button->set_toggle_mode(true);
	pivot_button->connect("pressed", callable_mp(this, &CanvasItemEditor::_button_tool_select), make_binds(TOOL_EDIT_PIVOT));
	pivot_button->set_tooltip(TTR("Click to change object's rotation pivot."));

	pan_button = memnew(Button);
	pan_button->set_flat(true);
	hb->add_child(pan_button);
	pan_button->set_toggle_mode(true);
	pan_button->connect("pressed", callable_mp(this, &CanvasItemEditor::_button_tool_select), make_binds(TOOL_PAN));
	pan_button->set_shortcut(ED_SHORTCUT("canvas_item_editor/pan_mode", TTR("Pan Mode"), Key::G));
	pan_button->set_shortcut_context(this);
	pan_button->set_tooltip(TTR("You can also use Pan View shortcut (Space by default) to pan in any mode."));

	ruler_button = memnew(Button);
	ruler_button->set_flat(true);
	hb->add_child(ruler_button);
	ruler_button->set_toggle_mode(true);
	ruler_button->connect("pressed", callable_mp(this, &CanvasItemEditor::_button_tool_select), make_binds(TOOL_RULER));
	ruler_button->set_shortcut(ED_SHORTCUT("canvas_item_editor/ruler_mode", TTR("Ruler Mode"), Key::R));
	ruler_button->set_shortcut_context(this);
	ruler_button->set_tooltip(TTR("Ruler Mode"));

	hb->add_child(memnew(VSeparator));

	smart_snap_button = memnew(Button);
	smart_snap_button->set_flat(true);
	hb->add_child(smart_snap_button);
	smart_snap_button->set_toggle_mode(true);
	smart_snap_button->connect("toggled", callable_mp(this, &CanvasItemEditor::_button_toggle_smart_snap));
	smart_snap_button->set_tooltip(TTR("Toggle smart snapping."));
	smart_snap_button->set_shortcut(ED_SHORTCUT("canvas_item_editor/use_smart_snap", TTR("Use Smart Snap"), KeyModifierMask::SHIFT | Key::S));
	smart_snap_button->set_shortcut_context(this);

	grid_snap_button = memnew(Button);
	grid_snap_button->set_flat(true);
	hb->add_child(grid_snap_button);
	grid_snap_button->set_toggle_mode(true);
	grid_snap_button->connect("toggled", callable_mp(this, &CanvasItemEditor::_button_toggle_grid_snap));
	grid_snap_button->set_tooltip(TTR("Toggle grid snapping."));
	grid_snap_button->set_shortcut(ED_SHORTCUT("canvas_item_editor/use_grid_snap", TTR("Use Grid Snap"), KeyModifierMask::SHIFT | Key::G));
	grid_snap_button->set_shortcut_context(this);

	snap_config_menu = memnew(MenuButton);
	snap_config_menu->set_shortcut_context(this);
	hb->add_child(snap_config_menu);
	snap_config_menu->set_h_size_flags(SIZE_SHRINK_END);
	snap_config_menu->set_tooltip(TTR("Snapping Options"));
	snap_config_menu->set_switch_on_hover(true);

	PopupMenu *p = snap_config_menu->get_popup();
	p->connect("id_pressed", callable_mp(this, &CanvasItemEditor::_popup_callback));
	p->set_hide_on_checkable_item_selection(false);
	p->add_check_shortcut(ED_SHORTCUT("canvas_item_editor/use_rotation_snap", TTR("Use Rotation Snap")), SNAP_USE_ROTATION);
	p->add_check_shortcut(ED_SHORTCUT("canvas_item_editor/use_scale_snap", TTR("Use Scale Snap")), SNAP_USE_SCALE);
	p->add_check_shortcut(ED_SHORTCUT("canvas_item_editor/snap_relative", TTR("Snap Relative")), SNAP_RELATIVE);
	p->add_check_shortcut(ED_SHORTCUT("canvas_item_editor/use_pixel_snap", TTR("Use Pixel Snap")), SNAP_USE_PIXEL);
	p->add_submenu_item(TTR("Smart Snapping"), "SmartSnapping");

	p->add_separator();
	p->add_shortcut(ED_SHORTCUT("canvas_item_editor/configure_snap", TTR("Configure Snap...")), SNAP_CONFIGURE);

	smartsnap_config_popup = memnew(PopupMenu);
	p->add_child(smartsnap_config_popup);
	smartsnap_config_popup->set_name("SmartSnapping");
	smartsnap_config_popup->connect("id_pressed", callable_mp(this, &CanvasItemEditor::_popup_callback));
	smartsnap_config_popup->set_hide_on_checkable_item_selection(false);
	smartsnap_config_popup->add_check_shortcut(ED_SHORTCUT("canvas_item_editor/snap_node_parent", TTR("Snap to Parent")), SNAP_USE_NODE_PARENT);
	smartsnap_config_popup->add_check_shortcut(ED_SHORTCUT("canvas_item_editor/snap_node_anchors", TTR("Snap to Node Anchor")), SNAP_USE_NODE_ANCHORS);
	smartsnap_config_popup->add_check_shortcut(ED_SHORTCUT("canvas_item_editor/snap_node_sides", TTR("Snap to Node Sides")), SNAP_USE_NODE_SIDES);
	smartsnap_config_popup->add_check_shortcut(ED_SHORTCUT("canvas_item_editor/snap_node_center", TTR("Snap to Node Center")), SNAP_USE_NODE_CENTER);
	smartsnap_config_popup->add_check_shortcut(ED_SHORTCUT("canvas_item_editor/snap_other_nodes", TTR("Snap to Other Nodes")), SNAP_USE_OTHER_NODES);
	smartsnap_config_popup->add_check_shortcut(ED_SHORTCUT("canvas_item_editor/snap_guides", TTR("Snap to Guides")), SNAP_USE_GUIDES);

	hb->add_child(memnew(VSeparator));

	lock_button = memnew(Button);
	lock_button->set_flat(true);
	hb->add_child(lock_button);

	lock_button->connect("pressed", callable_mp(this, &CanvasItemEditor::_popup_callback), varray(LOCK_SELECTED));
	lock_button->set_tooltip(TTR("Lock selected node, preventing selection and movement."));
	// Define the shortcut globally (without a context) so that it works if the Scene tree dock is currently focused.
	lock_button->set_shortcut(ED_SHORTCUT("editor/lock_selected_nodes", TTR("Lock Selected Node(s)"), KeyModifierMask::CMD | Key::L));

	unlock_button = memnew(Button);
	unlock_button->set_flat(true);
	hb->add_child(unlock_button);
	unlock_button->connect("pressed", callable_mp(this, &CanvasItemEditor::_popup_callback), varray(UNLOCK_SELECTED));
	unlock_button->set_tooltip(TTR("Unlock selected node, allowing selection and movement."));
	// Define the shortcut globally (without a context) so that it works if the Scene tree dock is currently focused.
	unlock_button->set_shortcut(ED_SHORTCUT("editor/unlock_selected_nodes", TTR("Unlock Selected Node(s)"), KeyModifierMask::CMD | KeyModifierMask::SHIFT | Key::L));

	group_button = memnew(Button);
	group_button->set_flat(true);
	hb->add_child(group_button);
	group_button->connect("pressed", callable_mp(this, &CanvasItemEditor::_popup_callback), varray(GROUP_SELECTED));
	group_button->set_tooltip(TTR("Makes sure the object's children are not selectable."));
	// Define the shortcut globally (without a context) so that it works if the Scene tree dock is currently focused.
	group_button->set_shortcut(ED_SHORTCUT("editor/group_selected_nodes", TTR("Group Selected Node(s)"), KeyModifierMask::CMD | Key::G));

	ungroup_button = memnew(Button);
	ungroup_button->set_flat(true);
	hb->add_child(ungroup_button);
	ungroup_button->connect("pressed", callable_mp(this, &CanvasItemEditor::_popup_callback), varray(UNGROUP_SELECTED));
	ungroup_button->set_tooltip(TTR("Restores the object's children's ability to be selected."));
	// Define the shortcut globally (without a context) so that it works if the Scene tree dock is currently focused.
	ungroup_button->set_shortcut(ED_SHORTCUT("editor/ungroup_selected_nodes", TTR("Ungroup Selected Node(s)"), KeyModifierMask::CMD | KeyModifierMask::SHIFT | Key::G));

	hb->add_child(memnew(VSeparator));

	skeleton_menu = memnew(MenuButton);
	skeleton_menu->set_shortcut_context(this);
	hb->add_child(skeleton_menu);
	skeleton_menu->set_tooltip(TTR("Skeleton Options"));
	skeleton_menu->set_switch_on_hover(true);

	p = skeleton_menu->get_popup();
	p->set_hide_on_checkable_item_selection(false);
	p->add_shortcut(ED_SHORTCUT("canvas_item_editor/skeleton_show_bones", TTR("Show Bones")), SKELETON_SHOW_BONES);
	p->add_separator();
	p->add_shortcut(ED_SHORTCUT("canvas_item_editor/skeleton_make_bones", TTR("Make Bone2D Node(s) from Node(s)"), KeyModifierMask::CMD | KeyModifierMask::SHIFT | Key::B), SKELETON_MAKE_BONES);
	p->connect("id_pressed", callable_mp(this, &CanvasItemEditor::_popup_callback));

	hb->add_child(memnew(VSeparator));

	override_camera_button = memnew(Button);
	override_camera_button->set_flat(true);
	hb->add_child(override_camera_button);
	override_camera_button->connect("toggled", callable_mp(this, &CanvasItemEditor::_button_override_camera));
	override_camera_button->set_toggle_mode(true);
	override_camera_button->set_disabled(true);
	_update_override_camera_button(false);

	hb->add_child(memnew(VSeparator));

	view_menu = memnew(MenuButton);
	view_menu->set_shortcut_context(this);
	view_menu->set_text(TTR("View"));
	hb->add_child(view_menu);
	view_menu->get_popup()->connect("id_pressed", callable_mp(this, &CanvasItemEditor::_popup_callback));
	view_menu->set_switch_on_hover(true);

	p = view_menu->get_popup();
	p->set_hide_on_checkable_item_selection(false);
	p->add_check_shortcut(ED_SHORTCUT("canvas_item_editor/show_grid", TTR("Always Show Grid"), Key::NUMBERSIGN), SHOW_GRID);
	p->add_check_shortcut(ED_SHORTCUT("canvas_item_editor/show_helpers", TTR("Show Helpers"), Key::H), SHOW_HELPERS);
	p->add_check_shortcut(ED_SHORTCUT("canvas_item_editor/show_rulers", TTR("Show Rulers")), SHOW_RULERS);
	p->add_check_shortcut(ED_SHORTCUT("canvas_item_editor/show_guides", TTR("Show Guides"), Key::Y), SHOW_GUIDES);
	p->add_check_shortcut(ED_SHORTCUT("canvas_item_editor/show_origin", TTR("Show Origin")), SHOW_ORIGIN);
	p->add_check_shortcut(ED_SHORTCUT("canvas_item_editor/show_viewport", TTR("Show Viewport")), SHOW_VIEWPORT);
	p->add_check_shortcut(ED_SHORTCUT("canvas_item_editor/show_edit_locks", TTR("Show Group And Lock Icons")), SHOW_EDIT_LOCKS);
	p->add_check_shortcut(ED_SHORTCUT("canvas_item_editor/show_transformation_gizmos", TTR("Show Transformation Gizmos")), SHOW_TRANSFORMATION_GIZMOS);

	p->add_separator();
	p->add_shortcut(ED_SHORTCUT("canvas_item_editor/center_selection", TTR("Center Selection"), Key::F), VIEW_CENTER_TO_SELECTION);
	p->add_shortcut(ED_SHORTCUT("canvas_item_editor/frame_selection", TTR("Frame Selection"), KeyModifierMask::SHIFT | Key::F), VIEW_FRAME_TO_SELECTION);
	p->add_shortcut(ED_SHORTCUT("canvas_item_editor/clear_guides", TTR("Clear Guides")), CLEAR_GUIDES);
	p->add_separator();
	p->add_check_shortcut(ED_SHORTCUT("canvas_item_editor/preview_canvas_scale", TTR("Preview Canvas Scale"), KeyModifierMask::SHIFT | KeyModifierMask::CMD | Key::P), PREVIEW_CANVAS_SCALE);

	hb->add_child(memnew(VSeparator));

	context_menu_container = memnew(PanelContainer);
	hbc_context_menu = memnew(HBoxContainer);
	context_menu_container->add_child(hbc_context_menu);
	// Use a custom stylebox to make contextual menu items stand out from the rest.
	// This helps with editor usability as contextual menu items change when selecting nodes,
	// even though it may not be immediately obvious at first.
	hb->add_child(context_menu_container);
	_update_context_menu_stylebox();

	presets_menu = memnew(MenuButton);
	presets_menu->set_shortcut_context(this);
	presets_menu->set_text(TTR("Layout"));
	hbc_context_menu->add_child(presets_menu);
	presets_menu->hide();
	presets_menu->set_switch_on_hover(true);

	p = presets_menu->get_popup();
	p->connect("id_pressed", callable_mp(this, &CanvasItemEditor::_popup_callback));

	anchors_popup = memnew(PopupMenu);
	p->add_child(anchors_popup);
	anchors_popup->set_name("Anchors");
	anchors_popup->connect("id_pressed", callable_mp(this, &CanvasItemEditor::_popup_callback));

	anchor_mode_button = memnew(Button);
	anchor_mode_button->set_flat(true);
	hbc_context_menu->add_child(anchor_mode_button);
	anchor_mode_button->set_toggle_mode(true);
	anchor_mode_button->hide();
	anchor_mode_button->connect("toggled", callable_mp(this, &CanvasItemEditor::_button_toggle_anchor_mode));

	animation_hb = memnew(HBoxContainer);
	hbc_context_menu->add_child(animation_hb);
	animation_hb->add_child(memnew(VSeparator));
	animation_hb->hide();

	key_loc_button = memnew(Button);
	key_loc_button->set_flat(true);
	key_loc_button->set_toggle_mode(true);
	key_loc_button->set_pressed(true);
	key_loc_button->set_focus_mode(FOCUS_NONE);
	key_loc_button->connect("pressed", callable_mp(this, &CanvasItemEditor::_popup_callback), varray(ANIM_INSERT_POS));
	key_loc_button->set_tooltip(TTR("Translation mask for inserting keys."));
	animation_hb->add_child(key_loc_button);

	key_rot_button = memnew(Button);
	key_rot_button->set_flat(true);
	key_rot_button->set_toggle_mode(true);
	key_rot_button->set_pressed(true);
	key_rot_button->set_focus_mode(FOCUS_NONE);
	key_rot_button->connect("pressed", callable_mp(this, &CanvasItemEditor::_popup_callback), varray(ANIM_INSERT_ROT));
	key_rot_button->set_tooltip(TTR("Rotation mask for inserting keys."));
	animation_hb->add_child(key_rot_button);

	key_scale_button = memnew(Button);
	key_scale_button->set_flat(true);
	key_scale_button->set_toggle_mode(true);
	key_scale_button->set_focus_mode(FOCUS_NONE);
	key_scale_button->connect("pressed", callable_mp(this, &CanvasItemEditor::_popup_callback), varray(ANIM_INSERT_SCALE));
	key_scale_button->set_tooltip(TTR("Scale mask for inserting keys."));
	animation_hb->add_child(key_scale_button);

	key_insert_button = memnew(Button);
	key_insert_button->set_flat(true);
	key_insert_button->set_focus_mode(FOCUS_NONE);
	key_insert_button->connect("pressed", callable_mp(this, &CanvasItemEditor::_popup_callback), varray(ANIM_INSERT_KEY));
	key_insert_button->set_tooltip(TTR("Insert keys (based on mask)."));
	key_insert_button->set_shortcut(ED_SHORTCUT("canvas_item_editor/anim_insert_key", TTR("Insert Key"), Key::INSERT));
	key_insert_button->set_shortcut_context(this);
	animation_hb->add_child(key_insert_button);

	key_auto_insert_button = memnew(Button);
	key_auto_insert_button->set_flat(true);
	key_auto_insert_button->set_toggle_mode(true);
	key_auto_insert_button->set_focus_mode(FOCUS_NONE);
	key_auto_insert_button->set_tooltip(TTR("Auto insert keys when objects are translated, rotated or scaled (based on mask).\nKeys are only added to existing tracks, no new tracks will be created.\nKeys must be inserted manually for the first time."));
	key_auto_insert_button->set_shortcut(ED_SHORTCUT("canvas_item_editor/anim_auto_insert_key", TTR("Auto Insert Key")));
	key_auto_insert_button->set_shortcut_context(this);
	animation_hb->add_child(key_auto_insert_button);

	animation_menu = memnew(MenuButton);
	animation_menu->set_shortcut_context(this);
	animation_menu->set_tooltip(TTR("Animation Key and Pose Options"));
	animation_hb->add_child(animation_menu);
	animation_menu->get_popup()->connect("id_pressed", callable_mp(this, &CanvasItemEditor::_popup_callback));
	animation_menu->set_switch_on_hover(true);

	p = animation_menu->get_popup();

	p->add_shortcut(ED_GET_SHORTCUT("canvas_item_editor/anim_insert_key"), ANIM_INSERT_KEY);
	p->add_shortcut(ED_SHORTCUT("canvas_item_editor/anim_insert_key_existing_tracks", TTR("Insert Key (Existing Tracks)"), KeyModifierMask::CMD + Key::INSERT), ANIM_INSERT_KEY_EXISTING);
	p->add_separator();
	p->add_shortcut(ED_SHORTCUT("canvas_item_editor/anim_copy_pose", TTR("Copy Pose")), ANIM_COPY_POSE);
	p->add_shortcut(ED_SHORTCUT("canvas_item_editor/anim_paste_pose", TTR("Paste Pose")), ANIM_PASTE_POSE);
	p->add_shortcut(ED_SHORTCUT("canvas_item_editor/anim_clear_pose", TTR("Clear Pose"), KeyModifierMask::SHIFT | Key::K), ANIM_CLEAR_POSE);

	snap_dialog = memnew(SnapDialog);
	snap_dialog->connect("confirmed", callable_mp(this, &CanvasItemEditor::_snap_changed));
	add_child(snap_dialog);

	select_sb = Ref<StyleBoxTexture>(memnew(StyleBoxTexture));

	selection_menu = memnew(PopupMenu);
	add_child(selection_menu);
	selection_menu->set_min_size(Vector2(100, 0));
	selection_menu->connect("id_pressed", callable_mp(this, &CanvasItemEditor::_selection_result_pressed));
	selection_menu->connect("popup_hide", callable_mp(this, &CanvasItemEditor::_selection_menu_hide));

	add_node_menu = memnew(PopupMenu);
	add_child(add_node_menu);
	add_node_menu->add_icon_item(editor->get_scene_tree_dock()->get_theme_icon(SNAME("Add"), SNAME("EditorIcons")), TTR("Add Node Here"));
	add_node_menu->add_icon_item(editor->get_scene_tree_dock()->get_theme_icon(SNAME("Instance"), SNAME("EditorIcons")), TTR("Instance Scene Here"));
	add_node_menu->connect("id_pressed", callable_mp(this, &CanvasItemEditor::_add_node_pressed));

	multiply_grid_step_shortcut = ED_SHORTCUT("canvas_item_editor/multiply_grid_step", TTR("Multiply grid step by 2"), Key::KP_MULTIPLY);
	divide_grid_step_shortcut = ED_SHORTCUT("canvas_item_editor/divide_grid_step", TTR("Divide grid step by 2"), Key::KP_DIVIDE);
	pan_view_shortcut = ED_SHORTCUT("canvas_item_editor/pan_view", TTR("Pan View"), Key::SPACE);

	skeleton_menu->get_popup()->set_item_checked(skeleton_menu->get_popup()->get_item_index(SKELETON_SHOW_BONES), true);
	singleton = this;

	// To ensure that scripts can parse the list of shortcuts correctly, we have to define
	// those shortcuts one by one.
	// Resetting zoom to 100% is a duplicate shortcut of `canvas_item_editor/reset_zoom`,
	// but it ensures both 1 and Ctrl + 0 can be used to reset zoom.
	ED_SHORTCUT("canvas_item_editor/zoom_3.125_percent", TTR("Zoom to 3.125%"), KeyModifierMask::SHIFT | Key::KEY_5);
	ED_SHORTCUT("canvas_item_editor/zoom_6.25_percent", TTR("Zoom to 6.25%"), KeyModifierMask::SHIFT | Key::KEY_4);
	ED_SHORTCUT("canvas_item_editor/zoom_12.5_percent", TTR("Zoom to 12.5%"), KeyModifierMask::SHIFT | Key::KEY_3);
	ED_SHORTCUT("canvas_item_editor/zoom_25_percent", TTR("Zoom to 25%"), KeyModifierMask::SHIFT | Key::KEY_2);
	ED_SHORTCUT("canvas_item_editor/zoom_50_percent", TTR("Zoom to 50%"), KeyModifierMask::SHIFT | Key::KEY_1);
	ED_SHORTCUT("canvas_item_editor/zoom_100_percent", TTR("Zoom to 100%"), Key::KEY_1);
	ED_SHORTCUT("canvas_item_editor/zoom_200_percent", TTR("Zoom to 200%"), Key::KEY_2);
	ED_SHORTCUT("canvas_item_editor/zoom_400_percent", TTR("Zoom to 400%"), Key::KEY_3);
	ED_SHORTCUT("canvas_item_editor/zoom_800_percent", TTR("Zoom to 800%"), Key::KEY_4);
	ED_SHORTCUT("canvas_item_editor/zoom_1600_percent", TTR("Zoom to 1600%"), Key::KEY_5);

	set_process_unhandled_key_input(true);

	// Update the menus' checkboxes
	call_deferred(SNAME("set_state"), get_state());
}

CanvasItemEditor *CanvasItemEditor::singleton = nullptr;

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
		RenderingServer::get_singleton()->viewport_set_disable_2d(editor->get_scene_root()->get_viewport_rid(), false);

	} else {
		canvas_item_editor->hide();
		canvas_item_editor->set_physics_process(false);
		RenderingServer::get_singleton()->viewport_set_disable_2d(editor->get_scene_root()->get_viewport_rid(), true);
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
	editor->get_main_control()->add_child(canvas_item_editor);
	canvas_item_editor->set_anchors_and_offsets_preset(Control::PRESET_WIDE);
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
	if (!button_group->get_pressed_button()) {
		return;
	}

	CheckBox *check = Object::cast_to<CheckBox>(button_group->get_pressed_button());
	default_texture_node_type = check->get_text();
	_perform_drop_data();
	selector->hide();
}

void CanvasItemEditorViewport::_on_change_type_closed() {
	_remove_preview();
}

void CanvasItemEditorViewport::_create_preview(const Vector<String> &files) const {
	bool add_preview = false;
	for (int i = 0; i < files.size(); i++) {
		String path = files[i];
		RES res = ResourceLoader::load(path);
		ERR_FAIL_COND(res.is_null());
		Ref<Texture2D> texture = Ref<Texture2D>(Object::cast_to<Texture2D>(*res));
		Ref<PackedScene> scene = Ref<PackedScene>(Object::cast_to<PackedScene>(*res));
		if (texture != nullptr || scene != nullptr) {
			if (texture != nullptr) {
				Sprite2D *sprite = memnew(Sprite2D);
				sprite->set_texture(texture);
				sprite->set_modulate(Color(1, 1, 1, 0.7f));
				preview_node->add_child(sprite);
				label->show();
				label_desc->show();
				label_desc->set_text(TTR("Drag and drop to add as child of current scene's root node.\nHold Ctrl when dropping to add as child of selected node.\nHold Shift when dropping to add as sibling of selected node.\nHold Alt when dropping to add as a different node type."));
			} else {
				if (scene.is_valid()) {
					Node *instance = scene->instantiate();
					if (instance) {
						preview_node->add_child(instance);
						label_desc->show();
						label_desc->set_text(TTR("Drag and drop to add as child of current scene's root node.\nHold Ctrl when dropping to add as child of selected node.\nHold Shift when dropping to add as sibling of selected node."));
					}
				}
			}
			add_preview = true;
		}
	}

	if (add_preview) {
		editor->get_scene_root()->add_child(preview_node);
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
	if (p_desired_node->get_scene_file_path() == p_target_scene_path) {
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
	// Adjust casing according to project setting. The file name is expected to be in snake_case, but will work for others.
	String name = path.get_file().get_basename();
	switch (ProjectSettings::get_singleton()->get("editor/node_naming/name_casing").operator int()) {
		case NAME_CASING_PASCAL_CASE:
			name = name.capitalize().replace(" ", "");
			break;
		case NAME_CASING_CAMEL_CASE:
			name = name.capitalize().replace(" ", "");
			name[0] = name.to_lower()[0];
			break;
		case NAME_CASING_SNAKE_CASE:
			name = name.capitalize().replace(" ", "_").to_lower();
			break;
	}
	child->set_name(name);

	Ref<Texture2D> texture = Ref<Texture2D>(Object::cast_to<Texture2D>(ResourceCache::get(path)));

	if (parent) {
		editor_data->get_undo_redo().add_do_method(parent, "add_child", child, true);
		editor_data->get_undo_redo().add_do_method(child, "set_owner", editor->get_edited_scene());
		editor_data->get_undo_redo().add_do_reference(child);
		editor_data->get_undo_redo().add_undo_method(parent, "remove_child", child);
	} else { // If no parent is selected, set as root node of the scene.
		editor_data->get_undo_redo().add_do_method(editor, "set_edited_scene", child);
		editor_data->get_undo_redo().add_do_method(child, "set_owner", editor->get_edited_scene());
		editor_data->get_undo_redo().add_do_reference(child);
		editor_data->get_undo_redo().add_undo_method(editor, "set_edited_scene", (Object *)nullptr);
	}

	if (parent) {
		String new_name = parent->validate_child_name(child);
		EditorDebuggerNode *ed = EditorDebuggerNode::get_singleton();
		editor_data->get_undo_redo().add_do_method(ed, "live_debug_create_node", editor->get_edited_scene()->get_path_to(parent), child->get_class(), new_name);
		editor_data->get_undo_redo().add_undo_method(ed, "live_debug_remove_node", NodePath(String(editor->get_edited_scene()->get_path_to(parent)) + "/" + new_name));
	}

	String node_class = child->get_class();
	if (node_class == "Polygon2D") {
		editor_data->get_undo_redo().add_do_property(child, "texture/texture", texture);
	} else if (node_class == "TouchScreenButton") {
		editor_data->get_undo_redo().add_do_property(child, "normal", texture);
	} else if (node_class == "TextureButton") {
		editor_data->get_undo_redo().add_do_property(child, "texture_button", texture);
	} else {
		editor_data->get_undo_redo().add_do_property(child, "texture", texture);
	}

	// make visible for certain node type
	if (ClassDB::is_parent_class(node_class, "Control")) {
		Size2 texture_size = texture->get_size();
		editor_data->get_undo_redo().add_do_property(child, "rect_size", texture_size);
	} else if (node_class == "Polygon2D") {
		Size2 texture_size = texture->get_size();
		Vector<Vector2> list;
		list.push_back(Vector2(0, 0));
		list.push_back(Vector2(texture_size.width, 0));
		list.push_back(Vector2(texture_size.width, texture_size.height));
		list.push_back(Vector2(0, texture_size.height));
		editor_data->get_undo_redo().add_do_property(child, "polygon", list);
	}

	// Compute the global position
	Transform2D xform = canvas_item_editor->get_canvas_transform();
	Point2 target_position = xform.affine_inverse().xform(p_point);

	// there's nothing to be used as source position so snapping will work as absolute if enabled
	target_position = canvas_item_editor->snap_point(target_position);
	editor_data->get_undo_redo().add_do_method(child, "set_global_position", target_position);
}

bool CanvasItemEditorViewport::_create_instance(Node *parent, String &path, const Point2 &p_point) {
	Ref<PackedScene> sdata = ResourceLoader::load(path);
	if (!sdata.is_valid()) { // invalid scene
		return false;
	}

	Node *instantiated_scene = sdata->instantiate(PackedScene::GEN_EDIT_STATE_INSTANCE);
	if (!instantiated_scene) { // error on instancing
		return false;
	}

	if (editor->get_edited_scene()->get_scene_file_path() != "") { // cyclical instancing
		if (_cyclical_dependency_exists(editor->get_edited_scene()->get_scene_file_path(), instantiated_scene)) {
			memdelete(instantiated_scene);
			return false;
		}
	}

	instantiated_scene->set_scene_file_path(ProjectSettings::get_singleton()->localize_path(path));

	editor_data->get_undo_redo().add_do_method(parent, "add_child", instantiated_scene);
	editor_data->get_undo_redo().add_do_method(instantiated_scene, "set_owner", editor->get_edited_scene());
	editor_data->get_undo_redo().add_do_reference(instantiated_scene);
	editor_data->get_undo_redo().add_undo_method(parent, "remove_child", instantiated_scene);

	String new_name = parent->validate_child_name(instantiated_scene);
	EditorDebuggerNode *ed = EditorDebuggerNode::get_singleton();
	editor_data->get_undo_redo().add_do_method(ed, "live_debug_instance_node", editor->get_edited_scene()->get_path_to(parent), path, new_name);
	editor_data->get_undo_redo().add_undo_method(ed, "live_debug_remove_node", NodePath(String(editor->get_edited_scene()->get_path_to(parent)) + "/" + new_name));

	CanvasItem *parent_ci = Object::cast_to<CanvasItem>(parent);
	if (parent_ci) {
		Vector2 target_pos = canvas_item_editor->get_canvas_transform().affine_inverse().xform(p_point);
		target_pos = canvas_item_editor->snap_point(target_pos);
		target_pos = parent_ci->get_global_transform_with_canvas().affine_inverse().xform(target_pos);
		// Preserve instance position of the original scene.
		CanvasItem *instance_ci = Object::cast_to<CanvasItem>(instantiated_scene);
		if (instance_ci) {
			target_pos += instance_ci->_edit_get_position();
		}
		editor_data->get_undo_redo().add_do_method(instantiated_scene, "set_position", target_pos);
	}

	return true;
}

void CanvasItemEditorViewport::_perform_drop_data() {
	_remove_preview();

	// Without root dropping multiple files is not allowed
	if (!target_node && selected_files.size() > 1) {
		accept->set_text(TTR("Cannot instantiate multiple nodes without root."));
		accept->popup_centered();
		return;
	}

	Vector<String> error_files;

	editor_data->get_undo_redo().create_action(TTR("Create Node"));

	for (int i = 0; i < selected_files.size(); i++) {
		String path = selected_files[i];
		RES res = ResourceLoader::load(path);
		if (res.is_null()) {
			continue;
		}
		Ref<PackedScene> scene = Ref<PackedScene>(Object::cast_to<PackedScene>(*res));
		if (scene != nullptr && scene.is_valid()) {
			if (!target_node) {
				// Without root node act the same as "Load Inherited Scene"
				Error err = EditorNode::get_singleton()->load_scene(path, false, true);
				if (err != OK) {
					error_files.push_back(path);
				}
			} else {
				bool success = _create_instance(target_node, path, drop_pos);
				if (!success) {
					error_files.push_back(path);
				}
			}
		} else {
			Ref<Texture2D> texture = Ref<Texture2D>(Object::cast_to<Texture2D>(*res));
			if (texture != nullptr && texture.is_valid()) {
				Node *child = _make_texture_node_type(default_texture_node_type);
				_create_nodes(target_node, child, path, drop_pos);
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
		accept->set_text(vformat(TTR("Error instancing scene from %s"), files_str.get_data()));
		accept->popup_centered();
	}
}

bool CanvasItemEditorViewport::can_drop_data(const Point2 &p_point, const Variant &p_data) const {
	Dictionary d = p_data;
	if (d.has("type")) {
		if (String(d["type"]) == "files") {
			Vector<String> files = d["files"];
			bool can_instantiate = false;
			for (int i = 0; i < files.size(); i++) { // check if dragged files contain resource or scene can be created at least once
				RES res = ResourceLoader::load(files[i]);
				if (res.is_null()) {
					continue;
				}
				String type = res->get_class();
				if (type == "PackedScene") {
					Ref<PackedScene> sdata = Ref<PackedScene>(Object::cast_to<PackedScene>(*res));
					Node *instantiated_scene = sdata->instantiate(PackedScene::GEN_EDIT_STATE_INSTANCE);
					if (!instantiated_scene) {
						continue;
					}
					memdelete(instantiated_scene);
				} else if (ClassDB::is_parent_class(type, "Texture2D")) {
					Ref<Texture2D> texture = Ref<Texture2D>(Object::cast_to<Texture2D>(*res));
					if (!texture.is_valid()) {
						continue;
					}
				} else {
					continue;
				}
				can_instantiate = true;
				break;
			}
			if (can_instantiate) {
				if (!preview_node->get_parent()) { // create preview only once
					_create_preview(files);
				}
				Transform2D trans = canvas_item_editor->get_canvas_transform();
				preview_node->set_position((p_point - trans.get_origin()) / trans.get_scale().x);
				label->set_text(vformat(TTR("Adding %s..."), default_texture_node_type));
			}
			return can_instantiate;
		}
	}
	label->hide();
	return false;
}

void CanvasItemEditorViewport::_show_resource_type_selector() {
	_remove_preview();
	List<BaseButton *> btn_list;
	button_group->get_buttons(&btn_list);

	for (int i = 0; i < btn_list.size(); i++) {
		CheckBox *check = Object::cast_to<CheckBox>(btn_list[i]);
		check->set_pressed(check->get_text() == default_texture_node_type);
	}
	selector->set_title(vformat(TTR("Add %s"), default_texture_node_type));
	selector->popup_centered();
}

bool CanvasItemEditorViewport::_only_packed_scenes_selected() const {
	for (int i = 0; i < selected_files.size(); ++i) {
		if (ResourceLoader::load(selected_files[i])->get_class() != "PackedScene") {
			return false;
		}
	}

	return true;
}

void CanvasItemEditorViewport::drop_data(const Point2 &p_point, const Variant &p_data) {
	bool is_shift = Input::get_singleton()->is_key_pressed(Key::SHIFT);
	bool is_ctrl = Input::get_singleton()->is_key_pressed(Key::CTRL);
	bool is_alt = Input::get_singleton()->is_key_pressed(Key::ALT);

	selected_files.clear();
	Dictionary d = p_data;
	if (d.has("type") && String(d["type"]) == "files") {
		selected_files = d["files"];
	}
	if (selected_files.size() == 0) {
		return;
	}

	List<Node *> selected_nodes = editor->get_editor_selection()->get_selected_node_list();
	Node *root_node = editor->get_edited_scene();
	if (selected_nodes.size() > 0) {
		Node *selected_node = selected_nodes[0];
		target_node = root_node;
		if (is_ctrl) {
			target_node = selected_node;
		} else if (is_shift && selected_node != root_node) {
			target_node = selected_node->get_parent();
		}
	} else {
		if (root_node) {
			target_node = root_node;
		} else {
			drop_pos = p_point;
			target_node = nullptr;
		}
	}

	drop_pos = p_point;

	if (is_alt && !_only_packed_scenes_selected()) {
		_show_resource_type_selector();
	} else {
		_perform_drop_data();
	}
}

Node *CanvasItemEditorViewport::_make_texture_node_type(String texture_node_type) {
	Node *node = nullptr;
	if (texture_node_type == "Sprite2D") {
		node = memnew(Sprite2D);
	} else if (texture_node_type == "PointLight2D") {
		node = memnew(PointLight2D);
	} else if (texture_node_type == "CPUParticles2D") {
		node = memnew(CPUParticles2D);
	} else if (texture_node_type == "GPUParticles2D") {
		node = memnew(GPUParticles2D);
	} else if (texture_node_type == "Polygon2D") {
		node = memnew(Polygon2D);
	} else if (texture_node_type == "TouchScreenButton") {
		node = memnew(TouchScreenButton);
	} else if (texture_node_type == "TextureRect") {
		node = memnew(TextureRect);
	} else if (texture_node_type == "TextureButton") {
		node = memnew(TextureButton);
	} else if (texture_node_type == "NinePatchRect") {
		node = memnew(NinePatchRect);
	}
	return node;
}

void CanvasItemEditorViewport::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			connect("mouse_exited", callable_mp(this, &CanvasItemEditorViewport::_on_mouse_exit));
			label->add_theme_color_override("font_color", get_theme_color(SNAME("warning_color"), SNAME("Editor")));
		} break;
		case NOTIFICATION_EXIT_TREE: {
			disconnect("mouse_exited", callable_mp(this, &CanvasItemEditorViewport::_on_mouse_exit));
		} break;

		default:
			break;
	}
}

void CanvasItemEditorViewport::_bind_methods() {
}

CanvasItemEditorViewport::CanvasItemEditorViewport(EditorNode *p_node, CanvasItemEditor *p_canvas_item_editor) {
	default_texture_node_type = "Sprite2D";
	// Node2D
	texture_node_types.push_back("Sprite2D");
	texture_node_types.push_back("PointLight2D");
	texture_node_types.push_back("CPUParticles2D");
	texture_node_types.push_back("GPUParticles2D");
	texture_node_types.push_back("Polygon2D");
	texture_node_types.push_back("TouchScreenButton");
	// Control
	texture_node_types.push_back("TextureRect");
	texture_node_types.push_back("TextureButton");
	texture_node_types.push_back("NinePatchRect");

	target_node = nullptr;
	editor = p_node;
	editor_data = editor->get_scene_tree_dock()->get_editor_data();
	canvas_item_editor = p_canvas_item_editor;
	preview_node = memnew(Control);

	accept = memnew(AcceptDialog);
	editor->get_gui_base()->add_child(accept);

	selector = memnew(AcceptDialog);
	editor->get_gui_base()->add_child(selector);
	selector->set_title(TTR("Change Default Type"));
	selector->connect("confirmed", callable_mp(this, &CanvasItemEditorViewport::_on_change_type_confirmed));
	selector->connect("cancelled", callable_mp(this, &CanvasItemEditorViewport::_on_change_type_closed));

	VBoxContainer *vbc = memnew(VBoxContainer);
	selector->add_child(vbc);
	vbc->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	vbc->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	vbc->set_custom_minimum_size(Size2(240, 260) * EDSCALE);

	btn_group = memnew(VBoxContainer);
	vbc->add_child(btn_group);
	btn_group->set_h_size_flags(0);

	button_group.instantiate();
	for (int i = 0; i < texture_node_types.size(); i++) {
		CheckBox *check = memnew(CheckBox);
		btn_group->add_child(check);
		check->set_text(texture_node_types[i]);
		check->connect("button_down", callable_mp(this, &CanvasItemEditorViewport::_on_select_type), varray(check));
		check->set_button_group(button_group);
	}

	label = memnew(Label);
	label->add_theme_color_override("font_shadow_color", Color(0, 0, 0, 1));
	label->add_theme_constant_override("shadow_outline_size", 1 * EDSCALE);
	label->hide();
	canvas_item_editor->get_controls_container()->add_child(label);

	label_desc = memnew(Label);
	label_desc->add_theme_color_override("font_color", Color(0.6f, 0.6f, 0.6f, 1));
	label_desc->add_theme_color_override("font_shadow_color", Color(0.2f, 0.2f, 0.2f, 1));
	label_desc->add_theme_constant_override("shadow_outline_size", 1 * EDSCALE);
	label_desc->add_theme_constant_override("line_spacing", 0);
	label_desc->hide();
	canvas_item_editor->get_controls_container()->add_child(label_desc);

	RS::get_singleton()->canvas_set_disable_scale(true);
}

CanvasItemEditorViewport::~CanvasItemEditorViewport() {
	memdelete(preview_node);
}
