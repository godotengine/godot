/**************************************************************************/
/*  canvas_item_manipulator.h                                             */
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

#pragma once

#ifdef DEBUG_ENABLED
#include "core/input/input.h"
#include "core/math/vector2.h"
#include "core/object/ref_counted.h"
#include "scene/debugger/debugger_helpers.h"

class CanvasItem;
class Control;
class InputEvent;
class Node;
class Shortcut;
class Viewport;

class CanvasItemManipulator : public RefCounted {
	GDCLASS(CanvasItemManipulator, RefCounted);

public:
	constexpr static real_t GIZMO_HANDLE_DISTANCE = 25;
	constexpr static Rect2 GIZMO_HANDLE_X_RECT = Rect2(GIZMO_HANDLE_DISTANCE, -5, 10, 10);
	constexpr static Rect2 GIZMO_HANDLE_Y_RECT = Rect2(-5, GIZMO_HANDLE_DISTANCE, 10, 10);

	constexpr static real_t RESIZE_HANDLE_DIAMETER = 10;

	constexpr static Size2 ANCHOR_HANDLE_SIZE = Size2(16, 16);

	enum Tool {
		TOOL_SELECT,
		TOOL_SCENE_PAINT,
		TOOL_LIST_SELECT,
		TOOL_MOVE,
		TOOL_SCALE,
		TOOL_ROTATE,
		TOOL_EDIT_PIVOT,
		TOOL_PAN,
		TOOL_RULER,
		TOOL_MAX,
	};

	enum DragType {
		DRAG_NONE,
		DRAG_BOX_SELECTION,
		DRAG_LEFT,
		DRAG_TOP_LEFT,
		DRAG_TOP,
		DRAG_TOP_RIGHT,
		DRAG_RIGHT,
		DRAG_BOTTOM_RIGHT,
		DRAG_BOTTOM,
		DRAG_BOTTOM_LEFT,
		DRAG_ANCHOR_TOP_LEFT,
		DRAG_ANCHOR_TOP_RIGHT,
		DRAG_ANCHOR_BOTTOM_RIGHT,
		DRAG_ANCHOR_BOTTOM_LEFT,
		DRAG_ANCHOR_ALL,
		DRAG_QUEUED,
		DRAG_MOVE,
		DRAG_MOVE_X,
		DRAG_MOVE_Y,
		DRAG_SCALE_X,
		DRAG_SCALE_Y,
		DRAG_SCALE_BOTH,
		DRAG_ROTATE,
		DRAG_PIVOT,
		DRAG_TEMP_PIVOT,
		DRAG_V_GUIDE,
		DRAG_H_GUIDE,
		DRAG_DOUBLE_GUIDE,
		DRAG_KEY_MOVE,
	};

	enum ShortcutName {
		SHORTCUT_CANCEL_TRANSFORM,
		SHORTCUT_MAX,
	};

	enum SnapTarget {
		SNAP_TARGET_NONE,
		SNAP_TARGET_PARENT,
		SNAP_TARGET_SELF_ANCHORS,
		SNAP_TARGET_SELF,
		SNAP_TARGET_OTHER_NODE,
		SNAP_TARGET_GUIDE,
		SNAP_TARGET_GRID,
		SNAP_TARGET_PIXEL,
	};

	enum SnapMode {
		SNAP_GRID = 1 << 0,
		SNAP_GUIDES = 1 << 1,
		SNAP_PIXEL = 1 << 2,
		SNAP_NODE_PARENT = 1 << 3,
		SNAP_NODE_ANCHORS = 1 << 4,
		SNAP_NODE_SIDES = 1 << 5,
		SNAP_NODE_CENTER = 1 << 6,
		SNAP_OTHER_NODES = 1 << 7,

		SNAP_DEFAULT = SNAP_GRID | SNAP_GUIDES | SNAP_PIXEL,
	};

private:
	Tool tool = TOOL_SELECT;
	DragType drag_type = DRAG_NONE;
	Input::CursorShape cursor_shape = Input::CursorShape::CURSOR_ARROW;

	bool local_space = false;
	bool show_transformation_gizmos = false;
	real_t grab_distance = 1;
	real_t drag_threshold = 0;
	real_t scale = 1;
	real_t zoom = 1;
	Point2 view_offset;
	bool anchors_mode = false;
	Viewport *viewport = nullptr;

	Point2 box_selecting_to;
	Point2 drag_start_origin;

	Point2 drag_from;
	Point2 drag_to;
	Point2 drag_rotation_center;
	Point2 temp_pivot = Point2(Math::INF, Math::INF);
	Array drag_selection;

	int snap_mode = SNAP_DEFAULT;
	bool smart_snap = false;
	bool snap_relative = false;
	bool snap_rotation = false;
	real_t snap_rotation_step = 0;
	real_t snap_rotation_offset = 0;
	bool snap_scale = false;
	real_t snap_scale_step = 0;
	Transform2D snap_transform;
	SnapTarget snap_target[2];

	bool grid_snap = false;
	Point2 grid_step;
	int grid_step_multiplier = 1;
	Point2 grid_offset;

	bool ruler_tool_active = false;
	Point2 ruler_tool_origin;

	bool show_guides = false;
	bool show_rulers = false;
	bool is_hovering_h_guide = false;
	bool is_hovering_v_guide = false;
	Point2 dragged_guide_pos;
	int dragged_guide_index = -1;
	real_t ruler_width = 0;

	HashMap<int, Ref<Shortcut>> inputs;

	Callable find_items_start_callback;
	Callable point_selected_callback;
	Callable get_selection_callback;
	Callable local_transform_callback;
	Callable global_transform_callback;
	Callable local_mouse_pos_callback;
	Callable plugin_input_callback;

	bool _gui_input_rulers_and_guides(const Ref<InputEvent> &p_event);
	bool _gui_input_open_scene_on_double_click(const Ref<InputEvent> &p_event);
	bool _gui_input_scale(const Ref<InputEvent> &p_event);
	bool _gui_input_pivot(const Ref<InputEvent> &p_event);
	bool _gui_input_resize(const Ref<InputEvent> &p_event);
	bool _gui_input_rotate(const Ref<InputEvent> &p_event);
	bool _gui_input_move(const Ref<InputEvent> &p_event);
	bool _gui_input_anchors(const Ref<InputEvent> &p_event);
	bool _gui_input_ruler_tool(const Ref<InputEvent> &p_event);
	bool _gui_input_select(const Ref<InputEvent> &p_event);
	bool _gui_input_cancel_drag(const Ref<InputEvent> &p_event, bool p_reset_snap_targets = false, bool p_restore_bones = false);

	bool _validate_drag_selection();

	Rect2 _encompass_selection_rect(const Array &p_selected);
	Point2 _position_to_anchor(const Control *p_control, Vector2 position);

	void _snap_if_closer_float(
			const real_t p_value,
			real_t &r_current_snap, SnapTarget &r_current_snap_target,
			const real_t p_target_value, const SnapTarget p_snap_target,
			const real_t p_radius = 10.0);
	void _snap_if_closer_point(
			Point2 p_value,
			Point2 &r_current_snap, SnapTarget (&r_current_snap_target)[2],
			Point2 p_target_value, const SnapTarget p_snap_target,
			const real_t rotation = 0.0,
			const real_t p_radius = 10.0);
	void _snap_other_nodes(
			const Point2 p_value,
			const Transform2D p_transform_to_snap,
			Point2 &r_current_snap, SnapTarget (&r_current_snap_target)[2],
			const SnapTarget p_snap_target, const Array &p_exceptions,
			const Node *p_current);

	bool _is_node_movable(const Node *p_node, bool p_popup_warning = false);

	void _update_cursor_shape();

protected:
	static void _bind_methods();

public:
	bool gui_input(const Ref<InputEvent> &p_event);

	Point2 snap_point(Point2 p_target, unsigned int p_modes = SNAP_DEFAULT, unsigned int p_forced_modes = 0, const CanvasItem *p_self_canvas_item = nullptr, const Array &p_other_nodes_exceptions = Array());
	Point2 anchor_to_position(const Control *p_control, const Point2 p_anchor);

	void get_canvas_items_at_pos(const Point2 &p_pos, Vector<DebuggerHelpers::SelectResult> &r_items, bool p_allow_locked = false);
	void find_canvas_items_at_pos(const Point2 &p_pos, Node *p_node, Vector<DebuggerHelpers::SelectResult> &r_items, const Transform2D &p_parent_xform = Transform2D(), const Transform2D &p_canvas_xform = Transform2D());
	void find_canvas_items_in_rect(const Rect2 &p_rect, Node *p_node, Vector<DebuggerHelpers::SelectResult> &r_items, const Transform2D &p_parent_xform = Transform2D(), const Transform2D &p_canvas_xform = Transform2D());

	Input::CursorShape get_cursor_shape() const { return cursor_shape; }

	void commit_drag();
	void reset_drag();

	bool reset_temp_pivot();

	void set_tool(Tool p_tool);
	void set_drag_type(DragType p_type) { drag_type = p_type; }
	void set_local_space_enabled(bool p_enabled) { local_space = p_enabled; }
	void set_grab_distance(const real_t p_distance) { grab_distance = p_distance; }
	void set_drag_threshold(const real_t p_threshold) { drag_threshold = p_threshold; }
	void set_scale(const real_t p_scale) { scale = p_scale; }
	void set_zoom(const real_t p_zoom) { zoom = p_zoom; }
	void set_view_offset(const Point2 p_offset) { view_offset = p_offset; }
	void set_anchors_mode_enabled(bool p_enabled) { anchors_mode = p_enabled; }
	void set_viewport(Viewport *p_viewport) { viewport = p_viewport; }

	void set_grid_snap_enabled(bool p_enabled) { grid_snap = p_enabled; }
	void set_grid_step(const Point2 &p_step) { grid_step = p_step; }
	void set_grid_step_multiplier(const int &p_multiplier) { grid_step_multiplier = p_multiplier; }
	void set_grid_offset(const Point2 p_offset) { grid_offset = p_offset; }

	void set_snap_mode(const int p_flag) { snap_mode = p_flag; }
	void set_smart_snap_enabled(bool p_enabled) { smart_snap = p_enabled; }
	void set_snap_relative_enabled(bool p_enabled) { snap_relative = p_enabled; }
	void set_snap_rotation_enabled(bool p_enabled) { snap_rotation = p_enabled; }
	void set_snap_rotation_step(const real_t p_step) { snap_rotation_step = p_step; }
	void set_snap_rotation_offset(const real_t p_offset) { snap_rotation_offset = p_offset; }
	void set_snap_scale_enabled(bool p_enabled) { snap_scale = p_enabled; }
	void set_snap_scale_step(const real_t p_step) { snap_scale_step = p_step; }

	void set_show_transformation_gizmos(bool p_enabled) { show_transformation_gizmos = p_enabled; }
	void set_show_guides(bool p_show) { show_guides = p_show; }
	void set_show_rulers(bool p_show) { show_rulers = p_show; }
	void set_ruler_width(const real_t p_width) { ruler_width = p_width; }

	void set_shortcut(const ShortcutName p_name, const Ref<Shortcut> &p_shortcut);
	void set_callbacks(const Callable p_find_items_start, const Callable p_point_selected, const Callable p_get_selection, const Callable p_local_xform, const Callable p_global_xform, const Callable p_local_mouse_pos, const Callable p_plugin_input);

	bool is_node_locked(const Node *p_node);
	bool is_node_movable(const Node *p_node) { return _is_node_movable(p_node); }

	Tool get_tool() const { return tool; }
	DragType get_drag_type() const { return drag_type; }
	bool is_local_space_enabled() { return local_space; }

	Point2 get_drag_to() const { return drag_to; }
	Point2 get_drag_from() const { return drag_from; }
	Point2 get_drag_rotation_center() const { return drag_rotation_center; }
	Array get_drag_selection() const { return drag_selection.duplicate(); }

	Point2 get_temp_pivot() const { return temp_pivot; }

	bool is_grid_snap_enabled() { return grid_snap; }
	Point2 get_grid_step() const { return grid_step; }
	Point2 get_grid_offset() const { return grid_offset; }
	int get_grid_step_multiplier() const { return grid_step_multiplier; }

	int get_snap_mode() const { return snap_mode; }
	bool is_smart_snap_enabled() { return smart_snap; }
	bool is_snap_relative_enabled() { return snap_relative; }
	bool is_snap_rotation_enabled() { return snap_rotation; }
	real_t get_snap_rotation_step() { return snap_rotation_step; }
	real_t get_snap_rotation_offset() { return snap_rotation_offset; }
	bool is_snap_scale_enabled() { return snap_scale; }
	real_t get_snap_scale_step() { return snap_scale_step; }
	Transform2D get_snap_transform() const { return snap_transform; }
	Pair<SnapTarget, SnapTarget> get_snap_target() const { return Pair(snap_target[0], snap_target[1]); }

	Point2 get_dragged_guide_position() const { return dragged_guide_pos; }
	int get_dragged_guide_index() const { return dragged_guide_index; }

	bool is_showing_transformation_gizmos() { return show_transformation_gizmos; }
	bool is_showing_guides() { return show_guides; }
	bool is_showing_rulers() { return show_rulers; }
	real_t get_ruler_width() const { return ruler_width; }

	bool is_ruler_tool_active() { return ruler_tool_active; }
	Point2 get_ruler_tool_origin() const { return ruler_tool_origin; }
};
#endif // DEBUG_ENABLED
