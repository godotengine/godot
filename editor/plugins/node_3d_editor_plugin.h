/*************************************************************************/
/*  node_3d_editor_plugin.h                                              */
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

#ifndef NODE_3D_EDITOR_PLUGIN_H
#define NODE_3D_EDITOR_PLUGIN_H

#include "editor/editor_node.h"
#include "editor/editor_plugin.h"
#include "editor/editor_scale.h"
#include "editor/plugins/node_3d_editor_gizmos.h"
#include "scene/3d/camera_3d.h"
#include "scene/3d/light_3d.h"
#include "scene/3d/visual_instance_3d.h"
#include "scene/3d/world_environment.h"
#include "scene/gui/panel_container.h"
#include "scene/resources/environment.h"
#include "scene/resources/fog_material.h"
#include "scene/resources/sky_material.h"

class Node3DEditor;
class Node3DEditorViewport;
class SubViewportContainer;
class DirectionalLight3D;
class WorldEnvironment;

class ViewportRotationControl : public Control {
	GDCLASS(ViewportRotationControl, Control);

	struct Axis2D {
		Vector2i screen_point;
		float z_axis = -99.0;
		int axis = -1;
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
	bool orbiting = false;
	int focused_axis = -2;

	const float AXIS_CIRCLE_RADIUS = 8.0f * EDSCALE;

protected:
	void _notification(int p_what);
	virtual void gui_input(const Ref<InputEvent> &p_event) override;
	void _draw();
	void _draw_axis(const Axis2D &p_axis);
	void _get_sorted_axis(Vector<Axis2D> &r_axis);
	void _update_focus();
	void _on_mouse_exited();

public:
	void set_viewport(Node3DEditorViewport *p_viewport);
};

class Node3DEditorViewport : public Control {
	GDCLASS(Node3DEditorViewport, Control);
	friend class Node3DEditor;
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
	Node3D *preview_node;
	AABB *preview_bounds;
	Vector<String> selected_files;
	AcceptDialog *accept;

	Node *target_node;
	Point2 drop_pos;

	EditorNode *editor;
	EditorData *editor_data;
	EditorSelection *editor_selection;
	UndoRedo *undo_redo;

	CheckBox *preview_camera;
	SubViewportContainer *subviewport_container;

	MenuButton *view_menu;
	PopupMenu *display_submenu;

	Control *surface;
	SubViewport *viewport;
	Camera3D *camera;
	bool transforming;
	bool orthogonal;
	bool auto_orthogonal;
	bool lock_rotation;
	real_t gizmo_scale;

	bool freelook_active;
	real_t freelook_speed;
	Vector2 previous_mouse_position;

	Label *info_label;
	Label *cinema_label;
	Label *locked_label;
	Label *zoom_limit_label;

	VBoxContainer *top_right_vbox;
	ViewportRotationControl *rotation_control;
	Gradient *frame_time_gradient;
	Label *cpu_time_label;
	Label *gpu_time_label;
	Label *fps_label;

	struct _RayResult {
		Node3D *item = nullptr;
		real_t depth = 0;
		_FORCE_INLINE_ bool operator<(const _RayResult &p_rr) const { return depth < p_rr.depth; }
	};

	void _update_name();
	void _compute_edit(const Point2 &p_point);
	void _clear_selected();
	void _select_clicked(bool p_allow_locked);
	ObjectID _select_ray(const Point2 &p_pos);
	void _find_items_at_pos(const Point2 &p_pos, Vector<_RayResult> &r_results, bool p_include_locked);
	Vector3 _get_ray_pos(const Vector2 &p_pos) const;
	Vector3 _get_ray(const Vector2 &p_pos) const;
	Point2 _point_to_screen(const Vector3 &p_point);
	Transform3D _get_camera_transform() const;
	int get_selected_count() const;
	void cancel_transform();

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
	Vector<_RayResult> selection_results;
	bool clicked_wants_append;

	PopupMenu *selection_menu;

	enum NavigationZoomStyle {
		NAVIGATION_ZOOM_VERTICAL,
		NAVIGATION_ZOOM_HORIZONTAL
	};

	enum NavigationMode {
		NAVIGATION_NONE,
		NAVIGATION_PAN,
		NAVIGATION_ZOOM,
		NAVIGATION_ORBIT,
		NAVIGATION_LOOK
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
	Transform3D to_camera_transform(const Cursor &p_cursor) const;
	void _draw();

	void _surface_mouse_enter();
	void _surface_mouse_exit();
	void _surface_focus_enter();
	void _surface_focus_exit();

	void _sinput(const Ref<InputEvent> &p_event);
	void _update_freelook(real_t delta);
	Node3DEditor *spatial_editor;

	Camera3D *previewing;
	Camera3D *preview;

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

	void _create_preview(const Vector<String> &files) const;
	void _remove_preview();
	bool _cyclical_dependency_exists(const String &p_target_scene_path, Node *p_desired_node);
	bool _create_instance(Node *parent, String &path, const Point2 &p_point);
	void _perform_drop_data();

	bool can_drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) const;
	void drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from);

	void _project_settings_changed();

	Transform3D _compute_transform(TransformMode p_mode, const Transform3D &p_original, const Transform3D &p_original_local, Vector3 p_motion, double p_extra, bool p_local, bool p_orthogonal);

	void begin_transform(TransformMode p_mode, bool instant);
	void commit_transform();
	void update_transform(Point2 p_mousepos, bool p_shift);

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void update_surface() { surface->update(); }
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

	Node3DEditorViewport(Node3DEditor *p_spatial_editor, EditorNode *p_editor, int p_index);
	~Node3DEditorViewport();
};

class Node3DEditorSelectedItem : public Object {
	GDCLASS(Node3DEditorSelectedItem, Object);

public:
	AABB aabb;
	Transform3D original; // original location when moving
	Transform3D original_local;
	Transform3D last_xform; // last transform
	bool last_xform_dirty;
	Node3D *sp;
	RID sbox_instance;
	RID sbox_instance_offset;
	RID sbox_instance_xray;
	RID sbox_instance_xray_offset;
	Ref<EditorNode3DGizmo> gizmo;
	Map<int, Transform3D> subgizmos; // map ID -> initial transform

	Node3DEditorSelectedItem() {
		sp = nullptr;
		last_xform_dirty = true;
	}
	~Node3DEditorSelectedItem();
};

class Node3DEditorViewportContainer : public Container {
	GDCLASS(Node3DEditorViewportContainer, Container);

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
	View view;
	bool mouseover;
	real_t ratio_h;
	real_t ratio_v;

	bool hovering_v;
	bool hovering_h;

	bool dragging_v;
	bool dragging_h;
	Vector2 drag_begin_pos;
	Vector2 drag_begin_ratio;

	virtual void gui_input(const Ref<InputEvent> &p_event) override;

protected:
	void _notification(int p_what);

public:
	void set_view(View p_view);
	View get_view();

	Node3DEditorViewportContainer();
};

class Node3DEditor : public VBoxContainer {
	GDCLASS(Node3DEditor, VBoxContainer);

public:
	static const unsigned int VIEWPORTS_COUNT = 4;

	enum ToolMode {
		TOOL_MODE_SELECT,
		TOOL_MODE_MOVE,
		TOOL_MODE_ROTATE,
		TOOL_MODE_SCALE,
		TOOL_MODE_LIST_SELECT,
		TOOL_LOCK_SELECTED,
		TOOL_UNLOCK_SELECTED,
		TOOL_GROUP_SELECTED,
		TOOL_UNGROUP_SELECTED,
		TOOL_MAX
	};

	enum ToolOptions {
		TOOL_OPT_LOCAL_COORDS,
		TOOL_OPT_USE_SNAP,
		TOOL_OPT_OVERRIDE_CAMERA,
		TOOL_OPT_MAX

	};

private:
	EditorNode *editor;
	EditorSelection *editor_selection;

	Node3DEditorViewportContainer *viewport_base;
	Node3DEditorViewport *viewports[VIEWPORTS_COUNT];
	VSplitContainer *shader_split;
	HSplitContainer *palette_split;

	/////

	ToolMode tool_mode;

	RID origin;
	RID origin_instance;
	bool origin_enabled;
	RID grid[3];
	RID grid_instance[3];
	bool grid_visible[3]; //currently visible
	bool grid_enable[3]; //should be always visible if true
	bool grid_enabled;
	bool grid_init_draw = false;
	Camera3D::Projection grid_camera_last_update_perspective = Camera3D::PROJECTION_PERSPECTIVE;
	Vector3 grid_camera_last_update_position = Vector3();

	Ref<ArrayMesh> move_gizmo[3], move_plane_gizmo[3], rotate_gizmo[4], scale_gizmo[3], scale_plane_gizmo[3], axis_gizmo[3];
	Ref<StandardMaterial3D> gizmo_color[3];
	Ref<StandardMaterial3D> plane_gizmo_color[3];
	Ref<ShaderMaterial> rotate_gizmo_color[3];
	Ref<StandardMaterial3D> gizmo_color_hl[3];
	Ref<StandardMaterial3D> plane_gizmo_color_hl[3];
	Ref<ShaderMaterial> rotate_gizmo_color_hl[3];

	Ref<Node3DGizmo> current_hover_gizmo;
	int current_hover_gizmo_handle;
	bool current_hover_gizmo_handle_secondary;

	real_t snap_translate_value;
	real_t snap_rotate_value;
	real_t snap_scale_value;

	Ref<ArrayMesh> selection_box_xray;
	Ref<ArrayMesh> selection_box;
	RID indicators;
	RID indicators_instance;
	RID cursor_mesh;
	RID cursor_instance;
	Ref<StandardMaterial3D> indicator_mat;
	Ref<ShaderMaterial> grid_mat[3];
	Ref<StandardMaterial3D> cursor_material;

	// Scene drag and drop support
	Node3D *preview_node;
	AABB preview_bounds;

	struct Gizmo {
		bool visible = false;
		real_t scale = 0;
		Transform3D transform;
	} gizmo;

	enum MenuOption {
		MENU_TOOL_SELECT,
		MENU_TOOL_MOVE,
		MENU_TOOL_ROTATE,
		MENU_TOOL_SCALE,
		MENU_TOOL_LIST_SELECT,
		MENU_TOOL_LOCAL_COORDS,
		MENU_TOOL_USE_SNAP,
		MENU_TOOL_OVERRIDE_CAMERA,
		MENU_TRANSFORM_CONFIGURE_SNAP,
		MENU_TRANSFORM_DIALOG,
		MENU_VIEW_USE_1_VIEWPORT,
		MENU_VIEW_USE_2_VIEWPORTS,
		MENU_VIEW_USE_2_VIEWPORTS_ALT,
		MENU_VIEW_USE_3_VIEWPORTS,
		MENU_VIEW_USE_3_VIEWPORTS_ALT,
		MENU_VIEW_USE_4_VIEWPORTS,
		MENU_VIEW_ORIGIN,
		MENU_VIEW_GRID,
		MENU_VIEW_GIZMOS_3D_ICONS,
		MENU_VIEW_CAMERA_SETTINGS,
		MENU_LOCK_SELECTED,
		MENU_UNLOCK_SELECTED,
		MENU_GROUP_SELECTED,
		MENU_UNGROUP_SELECTED,
		MENU_SNAP_TO_FLOOR
	};

	Button *tool_button[TOOL_MAX];
	Button *tool_option_button[TOOL_OPT_MAX];

	MenuButton *transform_menu;
	PopupMenu *gizmos_menu;
	MenuButton *view_menu;

	AcceptDialog *accept;

	ConfirmationDialog *snap_dialog;
	ConfirmationDialog *xform_dialog;
	ConfirmationDialog *settings_dialog;

	bool snap_enabled;
	bool snap_key_enabled;
	LineEdit *snap_translate;
	LineEdit *snap_rotate;
	LineEdit *snap_scale;

	LineEdit *xform_translate[3];
	LineEdit *xform_rotate[3];
	LineEdit *xform_scale[3];
	OptionButton *xform_type;

	VBoxContainer *settings_vbc;
	SpinBox *settings_fov;
	SpinBox *settings_znear;
	SpinBox *settings_zfar;

	void _snap_changed();
	void _snap_update();
	void _xform_dialog_action();
	void _menu_item_pressed(int p_option);
	void _menu_item_toggled(bool pressed, int p_option);
	void _menu_gizmo_toggled(int p_option);
	void _update_camera_override_button(bool p_game_running);
	void _update_camera_override_viewport(Object *p_viewport);
	HBoxContainer *hbc_menu;
	// Used for secondary menu items which are displayed depending on the currently selected node
	// (such as MeshInstance's "Mesh" menu).
	PanelContainer *context_menu_container;
	HBoxContainer *hbc_context_menu;

	void _generate_selection_boxes();
	UndoRedo *undo_redo;

	int camera_override_viewport_id;

	void _init_indicators();
	void _update_context_menu_stylebox();
	void _update_gizmos_menu();
	void _update_gizmos_menu_theme();
	void _init_grid();
	void _finish_indicators();
	void _finish_grid();

	void _toggle_maximize_view(Object *p_viewport);

	Node *custom_camera;

	Object *_get_editor_data(Object *p_what);

	Ref<Environment> viewport_environment;

	Node3D *selected;

	void _request_gizmo(Object *p_obj);
	void _set_subgizmo_selection(Object *p_obj, Ref<Node3DGizmo> p_gizmo, int p_id, Transform3D p_transform = Transform3D());
	void _clear_subgizmo_selection(Object *p_obj = nullptr);

	static Node3DEditor *singleton;

	void _node_added(Node *p_node);
	void _node_removed(Node *p_node);
	Vector<Ref<EditorNode3DGizmoPlugin>> gizmo_plugins_by_priority;
	Vector<Ref<EditorNode3DGizmoPlugin>> gizmo_plugins_by_name;

	void _register_all_gizmos();

	void _selection_changed();
	void _refresh_menu_icons();

	// Preview Sun and Environment

	uint32_t world_env_count = 0;
	uint32_t directional_light_count = 0;

	Button *sun_button;
	Label *sun_state;
	Label *sun_title;
	VBoxContainer *sun_vb;
	Popup *sun_environ_popup;
	Control *sun_direction;
	EditorSpinSlider *sun_angle_altitude;
	EditorSpinSlider *sun_angle_azimuth;
	ColorPickerButton *sun_color;
	EditorSpinSlider *sun_energy;
	EditorSpinSlider *sun_max_distance;
	Button *sun_add_to_scene;

	void _sun_direction_draw();
	void _sun_direction_input(const Ref<InputEvent> &p_event);
	void _sun_direction_angle_set();

	Vector2 sun_rotation;

	Ref<Shader> sun_direction_shader;
	Ref<ShaderMaterial> sun_direction_material;

	Button *environ_button;
	Label *environ_state;
	Label *environ_title;
	VBoxContainer *environ_vb;
	ColorPickerButton *environ_sky_color;
	ColorPickerButton *environ_ground_color;
	EditorSpinSlider *environ_energy;
	Button *environ_ao_button;
	Button *environ_glow_button;
	Button *environ_tonemap_button;
	Button *environ_gi_button;
	Button *environ_add_to_scene;

	Button *sun_environ_settings;

	DirectionalLight3D *preview_sun;
	WorldEnvironment *preview_environment;
	Ref<Environment> environment;
	Ref<ProceduralSkyMaterial> sky_material;

	bool sun_environ_updating = false;

	void _load_default_preview_settings();
	void _update_preview_environment();

	void _preview_settings_changed();
	void _sun_environ_settings_pressed();

	void _add_sun_to_scene(bool p_already_added_environment = false);
	void _add_environment_to_scene(bool p_already_added_sun = false);

	void _update_theme();

protected:
	void _notification(int p_what);
	//void _gui_input(InputEvent p_event);
	virtual void unhandled_key_input(const Ref<InputEvent> &p_event) override;

	static void _bind_methods();

public:
	static Node3DEditor *get_singleton() { return singleton; }

	Vector3 snap_point(Vector3 p_target, Vector3 p_start = Vector3(0, 0, 0)) const;

	float get_znear() const { return settings_znear->get_value(); }
	float get_zfar() const { return settings_zfar->get_value(); }
	float get_fov() const { return settings_fov->get_value(); }

	Transform3D get_gizmo_transform() const { return gizmo.transform; }
	bool is_gizmo_visible() const;

	ToolMode get_tool_mode() const { return tool_mode; }
	bool are_local_coords_enabled() const { return tool_option_button[Node3DEditor::TOOL_OPT_LOCAL_COORDS]->is_pressed(); }
	void set_local_coords_enabled(bool on) const { tool_option_button[Node3DEditor::TOOL_OPT_LOCAL_COORDS]->set_pressed(on); }
	bool is_snap_enabled() const { return snap_enabled ^ snap_key_enabled; }
	double get_translate_snap() const;
	double get_rotate_snap() const;
	double get_scale_snap() const;

	Ref<ArrayMesh> get_move_gizmo(int idx) const { return move_gizmo[idx]; }
	Ref<ArrayMesh> get_axis_gizmo(int idx) const { return axis_gizmo[idx]; }
	Ref<ArrayMesh> get_move_plane_gizmo(int idx) const { return move_plane_gizmo[idx]; }
	Ref<ArrayMesh> get_rotate_gizmo(int idx) const { return rotate_gizmo[idx]; }
	Ref<ArrayMesh> get_scale_gizmo(int idx) const { return scale_gizmo[idx]; }
	Ref<ArrayMesh> get_scale_plane_gizmo(int idx) const { return scale_plane_gizmo[idx]; }

	void update_grid();
	void update_transform_gizmo();
	void update_all_gizmos(Node *p_node = nullptr);
	void snap_selected_nodes_to_floor();
	void select_gizmo_highlight_axis(int p_axis);
	void set_custom_camera(Node *p_camera) { custom_camera = p_camera; }

	void set_undo_redo(UndoRedo *p_undo_redo) { undo_redo = p_undo_redo; }
	Dictionary get_state() const;
	void set_state(const Dictionary &p_state);

	Ref<Environment> get_viewport_environment() { return viewport_environment; }

	UndoRedo *get_undo_redo() { return undo_redo; }

	void add_control_to_menu_panel(Control *p_control);
	void remove_control_from_menu_panel(Control *p_control);

	VSplitContainer *get_shader_split();
	HSplitContainer *get_palette_split();

	Node3D *get_single_selected_node() { return selected; }
	bool is_current_selected_gizmo(const EditorNode3DGizmo *p_gizmo);
	bool is_subgizmo_selected(int p_id);
	Vector<int> get_subgizmo_selection();

	Ref<EditorNode3DGizmo> get_current_hover_gizmo() const { return current_hover_gizmo; }
	void set_current_hover_gizmo(Ref<EditorNode3DGizmo> p_gizmo) { current_hover_gizmo = p_gizmo; }

	void set_current_hover_gizmo_handle(int p_id, bool p_secondary) {
		current_hover_gizmo_handle = p_id;
		current_hover_gizmo_handle_secondary = p_secondary;
	}

	int get_current_hover_gizmo_handle(bool &r_secondary) const {
		r_secondary = current_hover_gizmo_handle_secondary;
		return current_hover_gizmo_handle;
	}

	void set_can_preview(Camera3D *p_preview);

	Node3DEditorViewport *get_editor_viewport(int p_idx) {
		ERR_FAIL_INDEX_V(p_idx, static_cast<int>(VIEWPORTS_COUNT), nullptr);
		return viewports[p_idx];
	}

	void add_gizmo_plugin(Ref<EditorNode3DGizmoPlugin> p_plugin);
	void remove_gizmo_plugin(Ref<EditorNode3DGizmoPlugin> p_plugin);

	void edit(Node3D *p_spatial);
	void clear();

	Node3DEditor(EditorNode *p_editor);
	~Node3DEditor();
};

class Node3DEditorPlugin : public EditorPlugin {
	GDCLASS(Node3DEditorPlugin, EditorPlugin);

	Node3DEditor *spatial_editor;
	EditorNode *editor;

public:
	Node3DEditor *get_spatial_editor() { return spatial_editor; }
	virtual String get_name() const override { return "3D"; }
	bool has_main_screen() const override { return true; }
	virtual void make_visible(bool p_visible) override;
	virtual void edit(Object *p_object) override;
	virtual bool handles(Object *p_object) const override;

	virtual Dictionary get_state() const override;
	virtual void set_state(const Dictionary &p_state) override;
	virtual void clear() override { spatial_editor->clear(); }

	virtual void edited_scene_changed() override;

	Node3DEditorPlugin(EditorNode *p_node);
	~Node3DEditorPlugin();
};

#endif // NODE_3D_EDITOR_PLUGIN_H
