/*************************************************************************/
/*  node_3d_editor_viewport.h                                            */
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

#ifndef NODE_3D_EDITOR_VIEWPORT_H
#define NODE_3D_EDITOR_VIEWPORT_H

#include "editor/editor_data.h"
#include "node_3d_editor_gizmos.h"
#include "scene/gui/check_box.h"
#include "scene/gui/control.h"
#include "scene/gui/dialogs.h"
#include "scene/gui/menu_button.h"
#include "scene/gui/subviewport_container.h"

class Node3DEditor;
class ViewportNavigationControl;
class ViewportRotationControl;

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
		VIEW_HALF_RESOLUTION,
		VIEW_AUDIO_LISTENER,
		VIEW_AUDIO_DOPPLER,
		VIEW_GIZMOS,
		VIEW_INFORMATION,
		VIEW_FRAME_TIME,
		VIEW_DISPLAY_NORMAL,
		VIEW_DISPLAY_WIREFRAME,
		VIEW_DISPLAY_OVERDRAW,
		VIEW_DISPLAY_SHADELESS,
		VIEW_DISPLAY_LIGHTING,
		VIEW_DISPLAY_ADVANCED,
		VIEW_DISPLAY_NORMAL_BUFFER,
		VIEW_DISPLAY_DEBUG_SHADOW_ATLAS,
		VIEW_DISPLAY_DEBUG_DIRECTIONAL_SHADOW_ATLAS,
		VIEW_DISPLAY_DEBUG_VOXEL_GI_ALBEDO,
		VIEW_DISPLAY_DEBUG_VOXEL_GI_LIGHTING,
		VIEW_DISPLAY_DEBUG_VOXEL_GI_EMISSION,
		VIEW_DISPLAY_DEBUG_SCENE_LUMINANCE,
		VIEW_DISPLAY_DEBUG_SSAO,
		VIEW_DISPLAY_DEBUG_SSIL,
		VIEW_DISPLAY_DEBUG_PSSM_SPLITS,
		VIEW_DISPLAY_DEBUG_DECAL_ATLAS,
		VIEW_DISPLAY_DEBUG_SDFGI,
		VIEW_DISPLAY_DEBUG_SDFGI_PROBES,
		VIEW_DISPLAY_DEBUG_GI_BUFFER,
		VIEW_DISPLAY_DEBUG_DISABLE_LOD,
		VIEW_DISPLAY_DEBUG_CLUSTER_OMNI_LIGHTS,
		VIEW_DISPLAY_DEBUG_CLUSTER_SPOT_LIGHTS,
		VIEW_DISPLAY_DEBUG_CLUSTER_DECALS,
		VIEW_DISPLAY_DEBUG_CLUSTER_REFLECTION_PROBES,
		VIEW_DISPLAY_DEBUG_OCCLUDERS,
		VIEW_DISPLAY_MOTION_VECTORS,

		VIEW_LOCK_ROTATION,
		VIEW_CINEMATIC_PREVIEW,
		VIEW_AUTO_ORTHOGONAL,
		VIEW_MAX
	};

	enum ViewType {
		VIEW_TYPE_USER,
		VIEW_TYPE_TOP,
		VIEW_TYPE_BOTTOM,
		VIEW_TYPE_LEFT,
		VIEW_TYPE_RIGHT,
		VIEW_TYPE_FRONT,
		VIEW_TYPE_REAR,
	};

public:
	enum {
		GIZMO_BASE_LAYER = 27,
		GIZMO_EDIT_LAYER = 26,
		GIZMO_GRID_LAYER = 25,
		MISC_TOOL_LAYER = 24,

		FRAME_TIME_HISTORY = 20,
	};

	enum NavigationScheme {
		NAVIGATION_GODOT,
		NAVIGATION_MAYA,
		NAVIGATION_MODO,
	};

	enum FreelookNavigationScheme {
		FREELOOK_DEFAULT,
		FREELOOK_PARTIALLY_AXIS_LOCKED,
		FREELOOK_FULLY_AXIS_LOCKED,
	};

private:
	double cpu_time_history[FRAME_TIME_HISTORY];
	int cpu_time_history_index;
	double gpu_time_history[FRAME_TIME_HISTORY];
	int gpu_time_history_index;

	int index;
	ViewType view_type;
	void _menu_option(int p_option);
	void _set_auto_orthogonal();
	Node3D *preview_node = nullptr;
	bool update_preview_node = false;
	Point2 preview_node_viewport_pos;
	Vector3 preview_node_pos;
	AABB *preview_bounds = nullptr;
	Vector<String> selected_files;
	AcceptDialog *accept = nullptr;

	Node *target_node = nullptr;
	Point2 drop_pos;

	EditorSelection *editor_selection = nullptr;

	CheckBox *preview_camera = nullptr;
	SubViewportContainer *subviewport_container = nullptr;

	MenuButton *view_menu = nullptr;
	PopupMenu *display_submenu = nullptr;

	Control *surface = nullptr;
	SubViewport *viewport = nullptr;
	Camera3D *camera = nullptr;
	bool transforming = false;
	bool orthogonal;
	bool auto_orthogonal;
	bool lock_rotation;
	real_t gizmo_scale;

	bool freelook_active;
	real_t freelook_speed;
	Vector2 previous_mouse_position;

	Label *info_label = nullptr;
	Label *cinema_label = nullptr;
	Label *locked_label = nullptr;
	Label *zoom_limit_label = nullptr;

	Label *preview_material_label = nullptr;
	Label *preview_material_label_desc = nullptr;

	VBoxContainer *top_right_vbox = nullptr;
	VBoxContainer *bottom_center_vbox = nullptr;
	ViewportNavigationControl *position_control = nullptr;
	ViewportNavigationControl *look_control = nullptr;
	ViewportRotationControl *rotation_control = nullptr;
	Gradient *frame_time_gradient = nullptr;
	Label *cpu_time_label = nullptr;
	Label *gpu_time_label = nullptr;
	Label *fps_label = nullptr;

	struct _RayResult {
		Node3D *item = nullptr;
		real_t depth = 0;
		_FORCE_INLINE_ bool operator<(const _RayResult &p_rr) const { return depth < p_rr.depth; }
	};

	void _update_name();
	void _compute_edit(const Point2 &p_point);
	void _clear_selected();
	void _select_clicked(bool p_allow_locked);
	ObjectID _select_ray(const Point2 &p_pos) const;
	void _find_items_at_pos(const Point2 &p_pos, Vector<_RayResult> &r_results, bool p_include_locked);
	Vector3 _get_ray_pos(const Vector2 &p_pos) const;
	Vector3 _get_ray(const Vector2 &p_pos) const;
	Point2 _point_to_screen(const Vector3 &p_point);
	Transform3D _get_camera_transform() const;
	int get_selected_count() const;
	void cancel_transform();
	void _update_shrink();

	Vector3 _get_camera_position() const;
	Vector3 _get_camera_normal() const;
	Vector3 _get_screen_to_space(const Vector3 &p_vector3);

	void _select_region();
	bool _transform_gizmo_select(const Vector2 &p_screenpos, bool p_highlight_only = false);
	void _transform_gizmo_apply(Node3D *p_node, const Transform3D &p_transform, bool p_local);

	void _nav_pan(Ref<InputEventWithModifiers> p_event, const Vector2 &p_relative);
	void _nav_zoom(Ref<InputEventWithModifiers> p_event, const Vector2 &p_relative);
	void _nav_orbit(Ref<InputEventWithModifiers> p_event, const Vector2 &p_relative);
	void _nav_look(Ref<InputEventWithModifiers> p_event, const Vector2 &p_relative);

	float get_znear() const;
	float get_zfar() const;
	float get_fov() const;

	ObjectID clicked;
	ObjectID material_target;
	Vector<_RayResult> selection_results;
	bool clicked_wants_append = false;
	bool selection_in_progress = false;

	PopupMenu *selection_menu = nullptr;

	enum NavigationZoomStyle {
		NAVIGATION_ZOOM_VERTICAL,
		NAVIGATION_ZOOM_HORIZONTAL
	};

	enum NavigationMode {
		NAVIGATION_NONE,
		NAVIGATION_PAN,
		NAVIGATION_ZOOM,
		NAVIGATION_ORBIT,
		NAVIGATION_LOOK,
		NAVIGATION_MOVE
	};
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

	struct EditData {
		TransformMode mode;
		TransformPlane plane;
		Transform3D original;
		Vector3 click_ray;
		Vector3 click_ray_pos;
		Vector3 center;
		Point2 mouse_pos;
		Point2 original_mouse_pos;
		bool snap = false;
		bool show_rotation_line = false;
		Ref<EditorNode3DGizmo> gizmo;
		int gizmo_handle = 0;
		bool gizmo_handle_secondary = false;
		Variant gizmo_initial_value;
		bool original_local;
		bool instant;
	} _edit;

	struct Cursor {
		Vector3 pos;
		real_t x_rot, y_rot, distance, fov_scale;
		Vector3 eye_pos; // Used in freelook mode
		bool region_select;
		Point2 region_begin, region_end;

		Cursor() {
			// These rotations place the camera in +X +Y +Z, aka south east, facing north west.
			x_rot = 0.5;
			y_rot = -0.5;
			distance = 4;
			fov_scale = 1.0;
			region_select = false;
		}
	};
	// Viewport camera supports movement smoothing,
	// so one cursor is the real cursor, while the other can be an interpolated version.
	Cursor cursor; // Immediate cursor
	Cursor camera_cursor; // That one may be interpolated (don't modify this one except for smoothing purposes)

	void scale_fov(real_t p_fov_offset);
	void reset_fov();
	void scale_cursor_distance(real_t scale);

	void set_freelook_active(bool active_now);
	void scale_freelook_speed(real_t scale);

	real_t zoom_indicator_delay;
	int zoom_failed_attempts_count = 0;

	RID move_gizmo_instance[3], move_plane_gizmo_instance[3], rotate_gizmo_instance[4], scale_gizmo_instance[3], scale_plane_gizmo_instance[3], axis_gizmo_instance[3];

	String last_message;
	String message;
	double message_time;

	void set_message(String p_message, float p_time = 5);

	void _view_settings_confirmed(real_t p_interp_delta);
	void _update_camera(real_t p_interp_delta);
	void _update_navigation_controls_visibility();
	Transform3D to_camera_transform(const Cursor &p_cursor) const;
	void _draw();

	void _surface_mouse_enter();
	void _surface_mouse_exit();
	void _surface_focus_enter();
	void _surface_focus_exit();

	void _sinput(const Ref<InputEvent> &p_event);
	void _update_freelook(real_t delta);
	Node3DEditor *spatial_editor = nullptr;

	Camera3D *previewing = nullptr;
	Camera3D *preview = nullptr;

	bool previewing_camera;
	bool previewing_cinema;
	bool _is_node_locked(const Node *p_node);
	void _preview_exited_scene();
	void _toggle_camera_preview(bool);
	void _toggle_cinema_preview(bool);
	void _init_gizmo_instance(int p_idx);
	void _finish_gizmo_instances();
	void _selection_result_pressed(int);
	void _selection_menu_hide();
	void _list_select(Ref<InputEventMouseButton> b);
	Point2i _get_warped_mouse_motion(const Ref<InputEventMouseMotion> &p_ev_mouse_motion) const;

	Vector3 _get_instance_position(const Point2 &p_pos) const;
	static AABB _calculate_spatial_bounds(const Node3D *p_parent, bool p_exclude_top_level_transform = true);

	Node *_sanitize_preview_node(Node *p_node) const;

	void _create_preview_node(const Vector<String> &files) const;
	void _remove_preview_node();
	bool _apply_preview_material(ObjectID p_target, const Point2 &p_point) const;
	void _reset_preview_material() const;
	void _remove_preview_material();
	bool _cyclical_dependency_exists(const String &p_target_scene_path, Node *p_desired_node);
	bool _create_instance(Node *parent, String &path, const Point2 &p_point);
	void _perform_drop_data();

	bool can_drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from);
	void drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from);

	void _project_settings_changed();

	Transform3D _compute_transform(TransformMode p_mode, const Transform3D &p_original, const Transform3D &p_original_local, Vector3 p_motion, double p_extra, bool p_local, bool p_orthogonal);

	void begin_transform(TransformMode p_mode, bool instant);
	void commit_transform();
	void update_transform(Point2 p_mousepos, bool p_shift);
	void finish_transform();

	void register_shortcut_action(const String &p_path, const String &p_name, Key p_keycode);
	void shortcut_changed_callback(const Ref<Shortcut> p_shortcut, const String &p_shortcut_path);

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void update_surface() { surface->queue_redraw(); }
	void update_transform_gizmo_view();

	void set_can_preview(Camera3D *p_preview);
	void set_state(const Dictionary &p_state);
	Dictionary get_state() const;
	void reset();
	bool is_freelook_active() const { return freelook_active; }

	void focus_selection();

	void assign_pending_data_pointers(
			Node3D *p_preview_node,
			AABB *p_preview_bounds,
			AcceptDialog *p_accept);

	SubViewport *get_viewport_node() { return viewport; }
	Camera3D *get_camera_3d() { return camera; } // return the default camera object.

	Node3DEditorViewport(Node3DEditor *p_spatial_editor, int p_index);
	~Node3DEditorViewport();
};

#endif // NODE_3D_EDITOR_VIEWPORT_H
