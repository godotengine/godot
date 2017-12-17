/*************************************************************************/
/*  spatial_editor_plugin.h                                              */
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
#ifndef SPATIAL_EDITOR_PLUGIN_H
#define SPATIAL_EDITOR_PLUGIN_H

#include "editor/editor_node.h"
#include "editor/editor_plugin.h"
#include "scene/3d/immediate_geometry.h"
#include "scene/3d/light.h"
#include "scene/3d/visual_instance.h"
#include "scene/gui/panel_container.h"
/**
	@author Juan Linietsky <reduzio@gmail.com>
*/

class Camera;
class SpatialEditor;
class SpatialEditorGizmos;

class SpatialEditorGizmo : public SpatialGizmo {

	GDCLASS(SpatialEditorGizmo, SpatialGizmo);

	bool selected;
	bool instanced;

public:
	void set_selected(bool p_selected) { selected = p_selected; }
	bool is_selected() const { return selected; }

	virtual String get_handle_name(int p_idx) const;
	virtual Variant get_handle_value(int p_idx) const;
	virtual void set_handle(int p_idx, Camera *p_camera, const Point2 &p_point);
	virtual void commit_handle(int p_idx, const Variant &p_restore, bool p_cancel = false);

	virtual bool intersect_frustum(const Camera *p_camera, const Vector<Plane> &p_frustum);
	virtual bool intersect_ray(const Camera *p_camera, const Point2 &p_point, Vector3 &r_pos, Vector3 &r_normal, int *r_gizmo_handle = NULL, bool p_sec_first = false);
	SpatialEditorGizmo();
};

class SpatialEditorViewport : public Control {

	GDCLASS(SpatialEditorViewport, Control);
	friend class SpatialEditor;
	enum {

		VIEW_TOP,
		VIEW_BOTTOM,
		VIEW_LEFT,
		VIEW_RIGHT,
		VIEW_FRONT,
		VIEW_REAR,
		VIEW_CENTER_TO_ORIGIN,
		VIEW_CENTER_TO_SELECTION,
		VIEW_ALIGN_SELECTION_WITH_VIEW,
		VIEW_PERSPECTIVE,
		VIEW_ENVIRONMENT,
		VIEW_ORTHOGONAL,
		VIEW_HALF_RESOLUTION,
		VIEW_AUDIO_LISTENER,
		VIEW_AUDIO_DOPPLER,
		VIEW_GIZMOS,
		VIEW_INFORMATION,
		VIEW_FPS,
		VIEW_DISPLAY_NORMAL,
		VIEW_DISPLAY_WIREFRAME,
		VIEW_DISPLAY_OVERDRAW,
		VIEW_DISPLAY_SHADELESS
	};

public:
	enum {
		GIZMO_BASE_LAYER = 27,
		GIZMO_EDIT_LAYER = 26,
		GIZMO_GRID_LAYER = 25
	};

private:
	int index;
	String name;
	void _menu_option(int p_option);

	Spatial *preview_node;
	AABB *preview_bounds;
	Vector<String> selected_files;
	AcceptDialog *accept;

	Node *target_node;
	Point2 drop_pos;

	EditorNode *editor;
	EditorData *editor_data;
	EditorSelection *editor_selection;
	UndoRedo *undo_redo;

	Button *preview_camera;
	ViewportContainer *viewport_container;

	MenuButton *view_menu;

	Control *surface;
	Viewport *viewport;
	Camera *camera;
	bool transforming;
	bool orthogonal;
	float gizmo_scale;

	bool freelook_active;
	real_t freelook_speed;

	Label *info_label;
	Label *fps_label;

	struct _RayResult {

		Spatial *item;
		float depth;
		int handle;
		_FORCE_INLINE_ bool operator<(const _RayResult &p_rr) const { return depth < p_rr.depth; }
	};

	void _update_name();
	void _compute_edit(const Point2 &p_point);
	void _clear_selected();
	void _select_clicked(bool p_append, bool p_single);
	void _select(Spatial *p_node, bool p_append, bool p_single);
	ObjectID _select_ray(const Point2 &p_pos, bool p_append, bool &r_includes_current, int *r_gizmo_handle = NULL, bool p_alt_select = false);
	void _find_items_at_pos(const Point2 &p_pos, bool &r_includes_current, Vector<_RayResult> &results, bool p_alt_select = false);
	Vector3 _get_ray_pos(const Vector2 &p_pos) const;
	Vector3 _get_ray(const Vector2 &p_pos) const;
	Point2 _point_to_screen(const Vector3 &p_point);
	Transform _get_camera_transform() const;
	int get_selected_count() const;

	Vector3 _get_camera_position() const;
	Vector3 _get_camera_normal() const;
	Vector3 _get_screen_to_space(const Vector3 &p_vector3);

	void _select_region();
	bool _gizmo_select(const Vector2 &p_screenpos, bool p_highlight_only = false);

	void _nav_pan(Ref<InputEventWithModifiers> p_event, const Vector2 &p_relative);
	void _nav_zoom(Ref<InputEventWithModifiers> p_event, const Vector2 &p_relative);
	void _nav_orbit(Ref<InputEventWithModifiers> p_event, const Vector2 &p_relative);
	void _nav_look(Ref<InputEventWithModifiers> p_event, const Vector2 &p_relative);

	float get_znear() const;
	float get_zfar() const;
	float get_fov() const;

	ObjectID clicked;
	Vector<_RayResult> selection_results;
	bool clicked_includes_current;
	bool clicked_wants_append;

	PopupMenu *selection_menu;

	enum NavigationScheme {
		NAVIGATION_GODOT,
		NAVIGATION_MAYA,
		NAVIGATION_MODO,
	};

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
		Transform original;
		Vector3 click_ray;
		Vector3 click_ray_pos;
		Vector3 center;
		Vector3 orig_gizmo_pos;
		int edited_gizmo;
		Point2 mouse_pos;
		bool snap;
		Ref<SpatialEditorGizmo> gizmo;
		int gizmo_handle;
		Variant gizmo_initial_value;
		Vector3 gizmo_initial_pos;
	} _edit;

	struct Cursor {

		Vector3 pos;
		float x_rot, y_rot, distance;
		Vector3 eye_pos; // Used in freelook mode
		bool region_select;
		Point2 region_begin, region_end;

		Cursor() {
			x_rot = y_rot = 0.5;
			distance = 4;
			region_select = false;
		}
	};
	// Viewport camera supports movement smoothing,
	// so one cursor is the real cursor, while the other can be an interpolated version.
	Cursor cursor; // Immediate cursor
	Cursor camera_cursor; // That one may be interpolated (don't modify this one except for smoothing purposes)

	void scale_cursor_distance(real_t scale);

	void set_freelook_active(bool active_now);
	void scale_freelook_speed(real_t scale);

	real_t zoom_indicator_delay;

	RID move_gizmo_instance[3], move_plane_gizmo_instance[3], rotate_gizmo_instance[3], scale_gizmo_instance[3], scale_plane_gizmo_instance[3];

	String last_message;
	String message;
	float message_time;

	void set_message(String p_message, float p_time = 5);

	//
	void _update_camera(float p_interp_delta);
	Transform to_camera_transform(const Cursor &p_cursor) const;
	void _draw();

	void _smouseenter();
	void _smouseexit();
	void _sinput(const Ref<InputEvent> &p_event);
	void _update_freelook(real_t delta);
	SpatialEditor *spatial_editor;

	Camera *previewing;
	Camera *preview;

	void _preview_exited_scene();
	void _toggle_camera_preview(bool);
	void _init_gizmo_instance(int p_idx);
	void _finish_gizmo_instances();
	void _selection_result_pressed(int);
	void _selection_menu_hide();
	void _list_select(Ref<InputEventMouseButton> b);
	Point2i _get_warped_mouse_motion(const Ref<InputEventMouseMotion> &p_ev_mouse_motion) const;

	Vector3 _get_instance_position(const Point2 &p_pos) const;
	static AABB _calculate_spatial_bounds(const Spatial *p_parent, const AABB p_bounds);
	void _create_preview(const Vector<String> &files) const;
	void _remove_preview();
	bool _cyclical_dependency_exists(const String &p_target_scene_path, Node *p_desired_node);
	bool _create_instance(Node *parent, String &path, const Point2 &p_point);
	void _perform_drop_data();

	bool can_drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) const;
	void drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from);

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void update_surface() { surface->update(); }
	void update_transform_gizmo_view();

	void set_can_preview(Camera *p_preview);
	void set_state(const Dictionary &p_state);
	Dictionary get_state() const;
	void reset();
	bool is_freelook_active() const { return freelook_active; }

	void focus_selection();

	void assign_pending_data_pointers(
			Spatial *p_preview_node,
			AABB *p_preview_bounds,
			AcceptDialog *p_accept);

	Viewport *get_viewport_node() { return viewport; }

	SpatialEditorViewport(SpatialEditor *p_spatial_editor, EditorNode *p_editor, int p_index);
};

class SpatialEditorSelectedItem : public Object {

	GDCLASS(SpatialEditorSelectedItem, Object);

public:
	AABB aabb;
	Transform original; // original location when moving
	Transform original_local;
	Transform last_xform; // last transform
	Spatial *sp;
	RID sbox_instance;

	SpatialEditorSelectedItem() { sp = NULL; }
	~SpatialEditorSelectedItem();
};

class SpatialEditorViewportContainer : public Container {

	GDCLASS(SpatialEditorViewportContainer, Container)
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
	float ratio_h;
	float ratio_v;

	bool dragging_v;
	bool dragging_h;
	Vector2 drag_begin_pos;
	Vector2 drag_begin_ratio;

	void _gui_input(const Ref<InputEvent> &p_event);

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void set_view(View p_view);
	View get_view();

	SpatialEditorViewportContainer();
};

class SpatialEditor : public VBoxContainer {

	GDCLASS(SpatialEditor, VBoxContainer);

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
		TOOL_MAX

	};

	enum ToolOptions {

		TOOL_OPT_LOCAL_COORDS,
		TOOL_OPT_USE_SNAP,
		TOOL_OPT_MAX

	};

private:
	EditorNode *editor;
	EditorSelection *editor_selection;

	SpatialEditorViewportContainer *viewport_base;
	SpatialEditorViewport *viewports[VIEWPORTS_COUNT];
	VSplitContainer *shader_split;
	HSplitContainer *palette_split;

	/////

	ToolMode tool_mode;
	bool orthogonal;

	VisualServer::ScenarioDebugMode scenario_debug;

	RID origin;
	RID origin_instance;
	RID grid[3];
	RID grid_instance[3];
	bool grid_visible[3]; //currently visible
	float last_grid_snap;
	bool grid_enable[3]; //should be always visible if true
	bool grid_enabled;

	Ref<ArrayMesh> move_gizmo[3], move_plane_gizmo[3], rotate_gizmo[3], scale_gizmo[3], scale_plane_gizmo[3];
	Ref<SpatialMaterial> gizmo_color[3];
	Ref<SpatialMaterial> plane_gizmo_color[3];
	Ref<SpatialMaterial> gizmo_hl;

	int over_gizmo_handle;

	Ref<ArrayMesh> selection_box;
	RID indicators;
	RID indicators_instance;
	RID cursor_mesh;
	RID cursor_instance;
	Ref<SpatialMaterial> indicator_mat;
	Ref<SpatialMaterial> cursor_material;

	// Scene drag and drop support
	Spatial *preview_node;
	AABB preview_bounds;

	struct Gizmo {

		bool visible;
		float scale;
		Transform transform;
	} gizmo;

	enum MenuOption {

		MENU_TOOL_SELECT,
		MENU_TOOL_MOVE,
		MENU_TOOL_ROTATE,
		MENU_TOOL_SCALE,
		MENU_TOOL_LIST_SELECT,
		MENU_TOOL_LOCAL_COORDS,
		MENU_TOOL_USE_SNAP,
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
		MENU_VIEW_CAMERA_SETTINGS,
		MENU_LOCK_SELECTED,
		MENU_UNLOCK_SELECTED,
		MENU_VISIBILITY_SKELETON
	};

	Button *tool_button[TOOL_MAX];
	Button *tool_option_button[TOOL_OPT_MAX];

	MenuButton *transform_menu;
	MenuButton *view_menu;

	ToolButton *lock_button;
	ToolButton *unlock_button;

	AcceptDialog *accept;

	ConfirmationDialog *snap_dialog;
	ConfirmationDialog *xform_dialog;
	ConfirmationDialog *settings_dialog;

	bool snap_enabled;
	LineEdit *snap_translate;
	LineEdit *snap_rotate;
	LineEdit *snap_scale;
	PanelContainer *menu_panel;

	LineEdit *xform_translate[3];
	LineEdit *xform_rotate[3];
	LineEdit *xform_scale[3];
	OptionButton *xform_type;

	VBoxContainer *settings_vbc;
	SpinBox *settings_fov;
	SpinBox *settings_znear;
	SpinBox *settings_zfar;

	void _xform_dialog_action();
	void _menu_item_pressed(int p_option);
	void _menu_item_toggled(bool pressed, int p_option);

	HBoxContainer *hbc_menu;

	void _generate_selection_box();
	UndoRedo *undo_redo;

	void _instance_scene();
	void _init_indicators();
	void _finish_indicators();

	void _toggle_maximize_view(Object *p_viewport);

	Node *custom_camera;

	Object *_get_editor_data(Object *p_what);

	Ref<Environment> viewport_environment;

	Spatial *selected;

	void _request_gizmo(Object *p_obj);

	static SpatialEditor *singleton;

	void _node_removed(Node *p_node);
	SpatialEditorGizmos *gizmos;
	SpatialEditor();

	bool is_any_freelook_active() const;

	void _refresh_menu_icons();

protected:
	void _notification(int p_what);
	//void _gui_input(InputEvent p_event);
	void _unhandled_key_input(Ref<InputEvent> p_event);

	static void _bind_methods();

public:
	static SpatialEditor *get_singleton() { return singleton; }
	void snap_cursor_to_plane(const Plane &p_plane);

	Vector3 snap_point(Vector3 p_target, Vector3 p_start = Vector3(0, 0, 0)) const;

	float get_znear() const { return settings_znear->get_value(); }
	float get_zfar() const { return settings_zfar->get_value(); }
	float get_fov() const { return settings_fov->get_value(); }

	Transform get_gizmo_transform() const { return gizmo.transform; }
	bool is_gizmo_visible() const { return gizmo.visible; }

	ToolMode get_tool_mode() const { return tool_mode; }
	bool are_local_coords_enabled() const { return tool_option_button[SpatialEditor::TOOL_OPT_LOCAL_COORDS]->is_pressed(); }
	bool is_snap_enabled() const { return snap_enabled; }
	float get_translate_snap() const { return snap_translate->get_text().to_double(); }
	float get_rotate_snap() const { return snap_rotate->get_text().to_double(); }
	float get_scale_snap() const { return snap_scale->get_text().to_double(); }

	Ref<ArrayMesh> get_move_gizmo(int idx) const { return move_gizmo[idx]; }
	Ref<ArrayMesh> get_move_plane_gizmo(int idx) const { return move_plane_gizmo[idx]; }
	Ref<ArrayMesh> get_rotate_gizmo(int idx) const { return rotate_gizmo[idx]; }
	Ref<ArrayMesh> get_scale_gizmo(int idx) const { return scale_gizmo[idx]; }
	Ref<ArrayMesh> get_scale_plane_gizmo(int idx) const { return scale_plane_gizmo[idx]; }

	int get_skeleton_visibility_state() const;

	void update_transform_gizmo();
	void update_all_gizmos();

	void select_gizmo_highlight_axis(int p_axis);
	void set_custom_camera(Node *p_camera) { custom_camera = p_camera; }

	void set_undo_redo(UndoRedo *p_undo_redo) { undo_redo = p_undo_redo; }
	Dictionary get_state() const;
	void set_state(const Dictionary &p_state);

	Ref<Environment> get_viewport_environment() { return viewport_environment; }

	UndoRedo *get_undo_redo() { return undo_redo; }

	void add_control_to_menu_panel(Control *p_control);

	VSplitContainer *get_shader_split();
	HSplitContainer *get_palette_split();

	Spatial *get_selected() { return selected; }

	int get_over_gizmo_handle() const { return over_gizmo_handle; }
	void set_over_gizmo_handle(int idx) { over_gizmo_handle = idx; }

	void set_can_preview(Camera *p_preview);

	SpatialEditorViewport *get_editor_viewport(int p_idx) {
		ERR_FAIL_INDEX_V(p_idx, 4, NULL);
		return viewports[p_idx];
	}

	Camera *get_camera() { return NULL; }
	void edit(Spatial *p_spatial);
	void clear();

	SpatialEditor(EditorNode *p_editor);
	~SpatialEditor();
};

class SpatialEditorPlugin : public EditorPlugin {

	GDCLASS(SpatialEditorPlugin, EditorPlugin);

	SpatialEditor *spatial_editor;
	EditorNode *editor;

protected:
	static void _bind_methods();

public:
	void snap_cursor_to_plane(const Plane &p_plane);

	SpatialEditor *get_spatial_editor() { return spatial_editor; }
	virtual String get_name() const { return "3D"; }
	bool has_main_screen() const { return true; }
	virtual void make_visible(bool p_visible);
	virtual void edit(Object *p_object);
	virtual bool handles(Object *p_object) const;

	virtual Dictionary get_state() const;
	virtual void set_state(const Dictionary &p_state);
	virtual void clear() { spatial_editor->clear(); }

	SpatialEditorPlugin(EditorNode *p_node);
	~SpatialEditorPlugin();
};

#endif
