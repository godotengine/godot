/**************************************************************************/
/*  node_3d_editor_viewport.h                                             */
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

#include "editor/plugins/editor_plugin.h"
#include "editor/scene/3d/node_3d_editor_gizmos.h"
#include "editor/themes/editor_scale.h"
#include "scene/debugger/view_3d_controller.h"
#include "scene/gui/margin_container.h"

class AcceptDialog;
class CheckBox;
class EditorSelection;
class Gradient;
class ImmediateMesh;
class MenuButton;
class MeshInstance3D;
class Node3DEditor;
class Node3DEditorViewport;
class PanelContainer;
class RichTextLabel;
class SplitContainer;
class SubViewport;
class SubViewportContainer;
class VBoxContainer;

class Node3DEditorSelectedItem : public Object {
	GDCLASS(Node3DEditorSelectedItem, Object);

public:
	AABB aabb;
	Transform3D original; // original location when moving
	Transform3D original_local;
	Transform3D last_xform; // last transform
	bool last_xform_dirty;
	Node3D *sp = nullptr;
	RID sbox_instance;
	RID sbox_instance_offset;
	RID sbox_instance_xray;
	RID sbox_instance_xray_offset;
	Ref<EditorNode3DGizmo> gizmo;
	HashMap<int, Transform3D> subgizmos; // Key: Subgizmo ID, Value: Initial subgizmo transform.

	Node3DEditorSelectedItem() {
		sp = nullptr;
		last_xform_dirty = true;
	}
	~Node3DEditorSelectedItem();
};

class ViewportNavigationControl : public Control {
	GDCLASS(ViewportNavigationControl, Control);

	Node3DEditorViewport *viewport = nullptr;
	Vector2i focused_mouse_start;
	Vector2 focused_pos;
	bool hovered = false;
	int focused_index = -1;
	View3DController::NavigationMode nav_mode = View3DController::NavigationMode::NAV_MODE_NONE;

	const float AXIS_CIRCLE_RADIUS = 30.0f * EDSCALE;

protected:
	void _notification(int p_what);
	virtual void gui_input(const Ref<InputEvent> &p_event) override;
	void _draw();
	void _process_click(int p_index, Vector2 p_position, bool p_pressed);
	void _process_drag(int p_index, Vector2 p_position, Vector2 p_relative_position);
	void _update_navigation();

public:
	void set_navigation_mode(View3DController::NavigationMode p_nav_mode);
	void set_viewport(Node3DEditorViewport *p_viewport);
};

class ViewportRotationControl : public Control {
	GDCLASS(ViewportRotationControl, Control);

	struct Axis2D {
		Vector2 screen_point;
		float z_axis = -99.0;
		int axis = -1;
		bool is_positive = true;
	};

	struct Axis2DCompare {
		_FORCE_INLINE_ bool operator()(const Axis2D &l, const Axis2D &r) const {
			return l.z_axis < r.z_axis;
		}
	};

	Node3DEditorViewport *viewport = nullptr;
	Vector<Color> axis_colors;
	Vector<int> axis_menu_options;
	Vector2i orbiting_mouse_start;
	Point2 original_mouse_pos;
	View3DController::Cursor saved_cursor;
	int orbiting_index = -1;
	int focused_axis = -2;
	bool gizmo_activated = false;

	const float AXIS_CIRCLE_RADIUS = 8.0f * EDSCALE;

protected:
	void _notification(int p_what);
	virtual void gui_input(const Ref<InputEvent> &p_event) override;
	void _draw();
	void _draw_axis(const Axis2D &p_axis);
	void _get_sorted_axis(Vector<Axis2D> &r_axis);
	void _update_focus();
	void _process_click(int p_index, Vector2 p_position, bool p_pressed);
	void _process_drag(Ref<InputEventWithModifiers> p_event, int p_index, Vector2 p_position, Vector2 p_relative_position);

public:
	void set_viewport(Node3DEditorViewport *p_viewport);
};

class Node3DEditorViewport : public Control {
	GDCLASS(Node3DEditorViewport, Control);
	friend class Node3DEditor;
	friend class ViewportNavigationControl;
	friend class ViewportRotationControl;
	enum {
		VIEW_TOP,
		VIEW_BOTTOM,
		VIEW_LEFT,
		VIEW_RIGHT,
		VIEW_FRONT,
		VIEW_REAR,
		VIEW_CENTER_TO_ORIGIN,
		VIEW_CENTER_TO_SELECTION,
		VIEW_ALIGN_TRANSFORM_WITH_VIEW,
		VIEW_ALIGN_ROTATION_WITH_VIEW,
		VIEW_PERSPECTIVE,
		VIEW_ENVIRONMENT,
		VIEW_ORTHOGONAL,
		VIEW_SWITCH_PERSPECTIVE_ORTHOGONAL,
		VIEW_HALF_RESOLUTION,
		VIEW_AUDIO_LISTENER,
		VIEW_AUDIO_DOPPLER,
		VIEW_GIZMOS,
		VIEW_TRANSFORM_GIZMO,
		VIEW_GRID,
		VIEW_INFORMATION,
		VIEW_FRAME_TIME,

		// < Keep in sync with menu.
		VIEW_DISPLAY_NORMAL,
		VIEW_DISPLAY_WIREFRAME,
		VIEW_DISPLAY_OVERDRAW,
		VIEW_DISPLAY_LIGHTING,
		VIEW_DISPLAY_UNSHADED,
		VIEW_DISPLAY_ADVANCED,
		// Advanced menu:
		VIEW_DISPLAY_DEBUG_PSSM_SPLITS,
		VIEW_DISPLAY_NORMAL_BUFFER,
		VIEW_DISPLAY_DEBUG_SHADOW_ATLAS,
		VIEW_DISPLAY_DEBUG_DIRECTIONAL_SHADOW_ATLAS,
		VIEW_DISPLAY_DEBUG_DECAL_ATLAS,
		VIEW_DISPLAY_DEBUG_AREA_LIGHT_ATLAS,
		VIEW_DISPLAY_DEBUG_VOXEL_GI_ALBEDO,
		VIEW_DISPLAY_DEBUG_VOXEL_GI_LIGHTING,
		VIEW_DISPLAY_DEBUG_VOXEL_GI_EMISSION,
		VIEW_DISPLAY_DEBUG_SDFGI,
		VIEW_DISPLAY_DEBUG_SDFGI_PROBES,
		VIEW_DISPLAY_DEBUG_SCENE_LUMINANCE,
		VIEW_DISPLAY_DEBUG_SSAO,
		VIEW_DISPLAY_DEBUG_SSIL,
		VIEW_DISPLAY_DEBUG_GI_BUFFER,
		VIEW_DISPLAY_DEBUG_DISABLE_LOD,
		VIEW_DISPLAY_DEBUG_CLUSTER_OMNI_LIGHTS,
		VIEW_DISPLAY_DEBUG_CLUSTER_SPOT_LIGHTS,
		VIEW_DISPLAY_DEBUG_CLUSTER_AREA_LIGHTS,
		VIEW_DISPLAY_DEBUG_CLUSTER_DECALS,
		VIEW_DISPLAY_DEBUG_CLUSTER_REFLECTION_PROBES,
		VIEW_DISPLAY_DEBUG_OCCLUDERS,
		VIEW_DISPLAY_MOTION_VECTORS,
		VIEW_DISPLAY_INTERNAL_BUFFER,
		VIEW_DISPLAY_MAX,
		// > Keep in sync with menu.

		VIEW_LOCK_ROTATION,
		VIEW_CINEMATIC_PREVIEW,
		VIEW_AUTO_ORTHOGONAL,
		VIEW_MAX
	};

public:
	static constexpr int32_t GIZMO_BASE_LAYER = 27;
	static constexpr int32_t GIZMO_EDIT_LAYER = 26;
	static constexpr int32_t GIZMO_GRID_LAYER = 25;
	static constexpr int32_t MISC_TOOL_LAYER = 24;

	static constexpr int32_t FRAME_TIME_HISTORY = 20;

private:
	double cpu_time_history[FRAME_TIME_HISTORY];
	int cpu_time_history_index;
	double gpu_time_history[FRAME_TIME_HISTORY];
	int gpu_time_history_index;

	Node *ruler = nullptr;
	Node3D *ruler_start_point = nullptr;
	Node3D *ruler_end_point = nullptr;
	Ref<ImmediateMesh> geometry;
	Ref<ImmediateMesh> geometry_xray;
	MeshInstance3D *ruler_line = nullptr;
	MeshInstance3D *ruler_line_xray = nullptr;
	Label *ruler_label = nullptr;
	Ref<StandardMaterial3D> ruler_material;
	Ref<StandardMaterial3D> ruler_material_xray;
	Ref<StandardMaterial3D> ruler_triangle_material;
	Ref<StandardMaterial3D> ruler_triangle_material_xray;
	MeshInstance3D *ruler_triangle_lines = nullptr;
	MeshInstance3D *ruler_triangle_lines_xray = nullptr;
	Label *ruler_label_x = nullptr;
	Label *ruler_label_y = nullptr;
	Label *ruler_label_z = nullptr;

	int index;
	void _menu_option(int p_option);
	Node3D *preview_node = nullptr;
	bool update_preview_node = false;
	Point2 preview_node_viewport_pos;
	Vector3 preview_node_pos;
	AABB *preview_bounds = nullptr;
	Vector<String> selected_files;
	AcceptDialog *accept = nullptr;

	Node *target_node = nullptr;
	Point2 drop_pos;

	ObjectID focused_node_id;

	EditorSelection *editor_selection = nullptr;

	Button *translation_preview_button = nullptr;
	Button *follow_mode = nullptr;
	CheckBox *preview_camera = nullptr;
	CheckBox *pilot_camera = nullptr;
	SubViewportContainer *subviewport_container = nullptr;

	MenuButton *view_display_menu = nullptr;
	PopupMenu *display_submenu = nullptr;

	Control *surface = nullptr;
	SubViewport *viewport = nullptr;
	Camera3D *camera = nullptr;
	bool transforming = false;
	bool transform_gizmo_visible = true;
	bool collision_reposition = false;
	real_t gizmo_scale;

	bool vertex_snap_mode = false;
	Key vertex_snap_keycode = Key::NONE;
	bool vertex_snap_dragging = false;
	Vector3 vertex_snap_source;
	Plane vertex_snap_drag_plane;
	Vector3 vertex_snap_target;
	bool vertex_snap_has_target = false;
	bool vertex_snap_has_source = false;
	HashMap<ObjectID, Vector3> vertex_snap_original_positions;

	PanelContainer *info_panel = nullptr;
	Label *info_label = nullptr;
	Label *cinema_label = nullptr;
	Label *locked_label = nullptr;
	Label *zoom_limit_label = nullptr;

	RichTextLabel *tooltip_panel = nullptr;

	VBoxContainer *top_right_vbox = nullptr;
	VBoxContainer *bottom_center_vbox = nullptr;
	ViewportNavigationControl *position_control = nullptr;
	ViewportNavigationControl *look_control = nullptr;
	ViewportRotationControl *rotation_control = nullptr;
	Ref<Gradient> frame_time_gradient;
	PanelContainer *frame_time_panel = nullptr;
	VBoxContainer *frame_time_vbox = nullptr;
	Label *cpu_time_label = nullptr;
	Label *gpu_time_label = nullptr;
	Label *fps_label = nullptr;

	struct _RayResult {
		Node3D *item = nullptr;
		real_t depth = 0;
		_FORCE_INLINE_ bool operator<(const _RayResult &p_rr) const { return depth < p_rr.depth; }
	};

	void _view_state_changed();

	void _update_name();
	void _compute_edit(const Point2 &p_point);
	void _clear_selected();
	bool _is_rotation_arc_visible() const;
	void _select_clicked(bool p_allow_locked);
	ObjectID _select_ray(const Point2 &p_pos) const;
	void _find_items_at_pos(const Point2 &p_pos, Vector<_RayResult> &r_results, bool p_include_locked);

	float _min_screen_dist_to_aabb(const AABB &p_aabb, const Transform3D &p_transform, const Point2 &p_cursor) const;
	bool _find_closest_vertex_on_node(const Point2 &p_screen_pos, Node3D *p_node, float &r_closest_screen_dist, Vector3 &r_vertex_world) const;
	bool _find_closest_vertex_in_scene(const Point2 &p_screen_pos, float p_threshold, Vector3 &r_vertex_world, const HashMap<ObjectID, Vector3> *p_exclude = nullptr);
	void _vertex_snap_update_source(const Point2 &p_screen_pos);
	void _vertex_snap_commit();
	void _vertex_snap_cancel();
	bool _is_vertex_occluded(const Vector3 &p_world_pos, const Vector2 &p_screen_pos) const;

	Transform3D _get_camera_transform() const;
	int get_selected_count() const;
	bool _has_unlocked_selection() const;
	void cancel_transform();
	void _update_shrink();

	Vector3 _get_camera_position() const;
	Vector3 _get_camera_normal() const;
	Vector3 _get_screen_to_space(const Vector3 &p_vector3);
	Vector<Plane> _build_screen_frustum(const Point2 &p_min, const Point2 &p_max);

	void _select_region();
	bool _transform_gizmo_select(const Vector2 &p_screenpos, bool p_highlight_only = false);
	void _transform_gizmo_apply(Node3D *p_node, const Transform3D &p_transform, bool p_local);

	bool _is_shortcut_empty(const String &p_name);
	bool _is_nav_modifier_pressed(const String &p_name);

	float get_znear() const;
	float get_zfar() const;
	float get_fov() const;

	void _show_tooltip(const String &p_title, const String &p_description) const;

	ObjectID clicked;
	ObjectID material_target;
	Vector<Node3D *> selection_results;
	Vector<Node3D *> selection_results_menu;
	bool clicked_wants_append = false;
	bool selection_in_progress = false;
	bool movement_threshold_passed = false;

	PopupMenu *selection_menu = nullptr;

	enum TransformMode {
		TRANSFORM_NONE,
		TRANSFORM_ROTATE,
		TRANSFORM_TRANSLATE,
		TRANSFORM_SCALE
	};
	enum TransformPlane {
		TRANSFORM_VIEW,
		TRANSFORM_X_AXIS,
		TRANSFORM_Y_AXIS,
		TRANSFORM_Z_AXIS,
		TRANSFORM_YZ,
		TRANSFORM_XZ,
		TRANSFORM_XY,
	};
	enum TransformType {
		POSITION,
		ROTATION,
		SCALE,
	};

	struct EditData {
		TransformMode mode;
		TransformPlane plane;
		Transform3D original;
		Vector3 click_ray;
		Vector3 click_ray_pos;
		Vector3 center;
		Point2 mouse_pos;
		Point2 original_mouse_pos;
		bool show_rotation_line = false;
		bool is_trackball = false;
		Ref<EditorNode3DGizmo> gizmo;
		int gizmo_handle = 0;
		bool gizmo_handle_secondary = false;
		Variant gizmo_initial_value;
		bool original_local;
		bool instant;

		// Numeric blender-style transforms (e.g. 'g5x').
		// numeric_input tracks the current input value, e.g. 1.23.
		// numeric_negate indicates whether '-' has been pressed to negate the value
		// while numeric_next_decimal is 0, numbers are input before the decimal point
		// after pressing '.', numeric next decimal changes to -1, and decrements after each press.
		double numeric_input = 0.0;
		bool numeric_negate = false;
		int numeric_next_decimal = 0;

		Vector3 rotation_axis;
		Vector3 view_axis_local;
		double accumulated_rotation_angle = 0.0;
		double rotation_angle = 0.0;
		Vector3 initial_click_vector;
		Vector3 previous_rotation_vector;
		bool gizmo_initiated = false;

		HashMap<Node3D *, Transform3D> children_original_globals;
	} _edit;

	Ref<View3DController> view_3d_controller;
	void _update_view_3d_controller(bool p_update_all = true);

	void _cursor_interpolated();
	void _cursor_distance_scaled();

	void _freelook_changed();
	void _freelook_speed_scaled();

	real_t zoom_indicator_delay;
	int zoom_failed_attempts_count = 0;

	RID move_gizmo_instance[3], move_plane_gizmo_instance[3], rotate_gizmo_instance[4], scale_gizmo_instance[3], scale_plane_gizmo_instance[3], axis_gizmo_instance[3];
	RID trackball_sphere_instance;

	String last_message;
	String message;
	double message_time;

	void set_message(const String &p_message, float p_time = 5);

	void _view_settings_confirmed(real_t p_interp_delta);
	void _update_navigation_controls_visibility();
	void _draw();

	// These allow tool scripts to set the 3D cursor location by updating the camera transform.
	Transform3D last_camera_transform;
	bool _camera_moved_externally();
	void _apply_camera_transform_to_cursor();

	void _surface_mouse_enter();
	void _surface_mouse_exit();
	void _surface_focus_enter();
	void _surface_focus_exit();

	void input(const Ref<InputEvent> &p_event) override;
	void _sinput(const Ref<InputEvent> &p_event);
	Node3DEditor *spatial_editor = nullptr;

	Camera3D *previewing = nullptr;
	Camera3D *preview = nullptr;

	bool previewing_camera = false;
	bool previewing_cinema = false;
	int times_focused_consecutively = 0;
	bool pilot_preview_enabled = false;

	bool pilot_undo_session_active = false;
	real_t pilot_undo_idle_time = 0.0;
	Transform3D pilot_undo_initial_transform;
	void _pilot_ensure_undo_session();
	void _pilot_commit_undo_session();
	void _pilot_tick_undo_session(real_t p_delta);

	bool _is_node_locked(const Node *p_node) const;
	void _preview_exited_scene();
	void _preview_camera_property_changed();
	void _sync_cursor_from_transform(const Transform3D &p_transform);
	void _update_centered_labels();
	void _disable_follow_mode();
	void _reset_follow_mode_count();
	void _toggle_camera_preview(bool);
	void _toggle_pilot_preview(bool);
	void _toggle_cinema_preview(bool);
	void _init_gizmo_instance(int p_idx);
	void _finish_gizmo_instances();
	void _selection_result_pressed(int);
	void _selection_menu_hide();
	void _list_select(Ref<InputEventMouseButton> b);

	Vector3 _get_instance_position(const Point2 &p_pos, Node3D *p_node) const;
	static AABB _calculate_spatial_bounds(const Node3D *p_parent, bool p_omit_top_level = false, const Transform3D *p_bounds_orientation = nullptr);

	Node *_sanitize_preview_node(Node *p_node) const;

	void _create_preview_node(const Vector<String> &files) const;
	void _remove_preview_node();
	bool _apply_preview_material(ObjectID p_target, const Point2 &p_point) const;
	void _reset_preview_material() const;
	void _remove_preview_material();
	bool _cyclical_dependency_exists(const String &p_target_scene_path, Node *p_desired_node) const;
	bool _create_instance(Node *p_parent, const String &p_path, const Point2 &p_point);
	bool _create_audio_node(Node *p_parent, const String &p_path, const Point2 &p_point);
	void _perform_drop_data();

	bool can_drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from);
	void drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from);

	void _project_settings_changed();

	Transform3D _compute_transform(TransformMode p_mode, const Transform3D &p_original, const Transform3D &p_original_local, Vector3 p_motion, double p_extra, bool p_local, bool p_orthogonal, bool p_view_axis);

	void _reset_transform(TransformType p_type);

	void begin_transform(TransformMode p_mode, bool instant);
	void commit_transform();
	void apply_transform(Vector3 p_motion, double p_snap);
	void update_transform(bool p_shift);
	void update_transform_numeric();
	void finish_transform();

	void _load_viewport_inputs();
	void register_shortcut_action(const String &p_path, const String &p_name, Key p_keycode, bool p_physical = false);
	void shortcut_changed_callback(const Ref<Shortcut> p_shortcut, const String &p_shortcut_path);

	// Supported rendering methods for advanced debug draw mode items.
	enum SupportedRenderingMethods {
		ALL,
		FORWARD_PLUS,
		FORWARD_PLUS_MOBILE,
	};

	void _set_lock_view_rotation(bool p_lock_rotation);
	void _add_advanced_debug_draw_mode_item(PopupMenu *p_popup, const String &p_name, int p_value, SupportedRenderingMethods p_rendering_methods = SupportedRenderingMethods::ALL, const String &p_tooltip = "");

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void update_surface() { surface->queue_redraw(); }
	void update_transform_gizmo_view();
	void update_transform_gizmo_highlight();

	void set_can_preview(Camera3D *p_preview);
	void switch_preview_camera(Camera3D *p_new_camera);
	void set_state(const Dictionary &p_state);
	Dictionary get_state() const;
	void reset();

	Vector3 get_ray_pos(const Vector2 &p_pos) const;
	Vector3 get_ray(const Vector2 &p_pos) const;
	Point2 point_to_screen(const Vector3 &p_point);

	void focus_selection();

	void assign_pending_data_pointers(
			Node3D *p_preview_node,
			AABB *p_preview_bounds,
			AcceptDialog *p_accept);

	SubViewport *get_viewport_node() { return viewport; }
	Camera3D *get_camera_3d() { return camera; } // return the default camera object.
	Control *get_surface() { return surface; }

	Node3DEditorViewport(Node3DEditor *p_spatial_editor, int p_index);
	~Node3DEditorViewport();
};

class Node3DEditorViewportContainer : public MarginContainer {
	GDCLASS(Node3DEditorViewportContainer, MarginContainer);

public:
	enum View {
		VIEW_USE_1_VIEWPORT,
		VIEW_USE_2_VIEWPORTS,
		VIEW_USE_2_VIEWPORTS_ALT,
		VIEW_USE_3_VIEWPORTS,
		VIEW_USE_3_VIEWPORTS_ALT,
		VIEW_USE_4_VIEWPORTS,
	};

private:
	View view = VIEW_USE_1_VIEWPORT;
	SplitContainer *main_split = nullptr;
	SplitContainer *first_split = nullptr;
	SplitContainer *second_split = nullptr;

	void _update_split_drag_margin();

protected:
	void _notification(int p_what);

public:
	void set_view(View p_view);
	View get_view();

	void add_viewport(Node3DEditorViewport *p_viewport, int p_index);

	Dictionary get_split_state() const;
	void set_split_state(const Dictionary &p_state);

	Node3DEditorViewportContainer();
};
