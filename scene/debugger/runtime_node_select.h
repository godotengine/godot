/**************************************************************************/
/*  runtime_node_select.h                                                 */
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

#include "core/object/ref_counted.h"

#ifndef _3D_DISABLED
#include "scene/debugger/view_3d_controller.h"
#include "scene/resources/mesh.h"
#endif // _3D_DISABLED

class InputEvent;
class InputEventMouseMotion;
class InputEventWithModifiers;
class Node;
class PopupMenu;
class ViewPanner;

#ifndef _3D_DISABLED
class ArrayMesh;
#endif

class RuntimeNodeSelect : public Object {
	GDCLASS(RuntimeNodeSelect, Object);

public:
	enum NodeType {
		NODE_TYPE_NONE,
		NODE_TYPE_2D,
		NODE_TYPE_3D,
		NODE_TYPE_MAX,
	};

	enum SelectMode {
		SELECT_MODE_SINGLE,
		SELECT_MODE_LIST,
		SELECT_MODE_MAX,
	};

private:
	friend class SceneDebugger;

	NodeType node_select_type = NODE_TYPE_2D;
	SelectMode node_select_mode = SELECT_MODE_SINGLE;

	struct SelectResult {
		Node *item = nullptr;
		real_t order = 0;
		_FORCE_INLINE_ bool operator<(const SelectResult &p_rr) const { return p_rr.order < order; }
	};

	const int SELECTION_MIN_AREA = 8 * 8;
	enum SelectionDragState {
		SELECTION_DRAG_NONE,
		SELECTION_DRAG_MOVE,
		SELECTION_DRAG_END,
	};
	SelectionDragState selection_drag_state = SELECTION_DRAG_NONE;

	bool has_selection = false;
	int max_selection = 1;
	Point2 selection_position = Point2(Math::INF, Math::INF);
	Rect2 selection_drag_area;
	PopupMenu *selection_list = nullptr;
	Color selection_area_fill;
	Color selection_area_outline;
	bool selection_visible = true;
	bool selection_update_queued = false;

	bool avoid_locked_nodes = false;
	bool prefer_group_selection = false;

	bool multi_shortcut_pressed = false;
	bool list_shortcut_pressed = false;
	RID draw_canvas;
	RID sel_drag_ci;

	bool camera_override = false;
	bool camera_first_override = true;

	// Values taken from EditorZoomWidget.
	const float VIEW_2D_MIN_ZOOM = 1.0 / 128;
	const float VIEW_2D_MAX_ZOOM = 128;

	Ref<ViewPanner> panner;
	Vector2 view_2d_offset;
	real_t view_2d_zoom = 1.0;
	bool warped_panning = false;

	LocalVector<ObjectID> selected_ci_nodes;
	real_t sel_2d_grab_dist = 0;
	int sel_2d_scale = 1;

	Color srect_color;
	RID srect_ci;

#ifndef _3D_DISABLED
	Ref<View3DController> view_3d_controller;

	real_t camera_fov = 0;
	real_t camera_znear = 0;
	real_t camera_zfar = 0;

	struct SelectionBox : public RefCounted {
		RID instance;
		RID instance_ofs;
		RID instance_xray;
		RID instance_xray_ofs;

		Transform3D transform;
		AABB bounds;

		~SelectionBox();
	};
	HashMap<ObjectID, Ref<SelectionBox>> selected_3d_nodes;

	Color sbox_color;
	Ref<ArrayMesh> sbox_mesh;
	Ref<ArrayMesh> sbox_mesh_xray;
#endif // _3D_DISABLED

	void _setup(const Dictionary &p_settings);

	void _node_set_type(NodeType p_type);
	void _select_set_mode(SelectMode p_mode);

	void _set_camera_override_enabled(bool p_enabled);

	void _root_window_input(const Ref<InputEvent> &p_event);
	void _items_popup_index_pressed(int p_index, PopupMenu *p_popup);
	void _update_input_state();

	void _process_frame();
	void _physics_frame();

	void _send_ids(const Vector<Node *> &p_picked_nodes, bool p_invert_new_selections = true);
	void _set_selected_nodes(const Vector<Node *> &p_nodes);
	void _queue_selection_update();
	void _update_selection();
	void _clear_selection();
	void _update_selection_drag(const Point2 &p_end_pos = Point2());
	void _set_selection_visible(bool p_visible);
	void _set_avoid_locked(bool p_enabled);
	void _set_prefer_group(bool p_enabled);

	void _open_selection_list(const Vector<SelectResult> &p_items, const Point2 &p_pos);
	void _close_selection_list();

	void _find_canvas_items_at_pos(const Point2 &p_pos, Node *p_node, Vector<SelectResult> &r_items, const Transform2D &p_parent_xform = Transform2D(), const Transform2D &p_canvas_xform = Transform2D());
	void _find_canvas_items_at_rect(const Rect2 &p_rect, Node *p_node, Vector<SelectResult> &r_items, const Transform2D &p_parent_xform = Transform2D(), const Transform2D &p_canvas_xform = Transform2D());
	void _pan_callback(Vector2 p_scroll_vec, Ref<InputEvent> p_event);
	void _zoom_callback(float p_zoom_factor, Vector2 p_origin, Ref<InputEvent> p_event);
	void _reset_camera_2d();
	void _update_view_2d();

#ifndef _3D_DISABLED
	void _find_3d_items_at_pos(const Point2 &p_pos, Vector<SelectResult> &r_items);
	void _find_3d_items_at_rect(const Rect2 &p_rect, Vector<SelectResult> &r_items);
	Vector3 _get_screen_to_space(const Vector3 &p_vector3);

	void _fov_scaled();
	void _cursor_interpolated();

	bool _handle_3d_input(const Ref<InputEvent> &p_event);
	void _reset_camera_3d();
#endif // _3D_DISABLED

	RuntimeNodeSelect() { singleton = this; }

	inline static RuntimeNodeSelect *singleton = nullptr;

public:
	static RuntimeNodeSelect *get_singleton();

	~RuntimeNodeSelect();
};
#endif // DEBUG_ENABLED
