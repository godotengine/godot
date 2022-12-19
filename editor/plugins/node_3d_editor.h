/*************************************************************************/
/*  node_3d_editor.h                                                     */
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

#ifndef NODE_3D_EDITOR_H
#define NODE_3D_EDITOR_H

#include "editor/editor_spin_slider.h"
#include "node_3d_editor_viewport.h"
#include "node_3d_editor_viewport_container.h"
#include "scene/3d/light_3d.h"
#include "scene/3d/world_environment.h"
#include "scene/gui/color_picker.h"
#include "scene/gui/option_button.h"
#include "scene/gui/panel_container.h"
#include "scene/gui/spin_box.h"
#include "scene/gui/split_container.h"
#include "scene/resources/sky_material.h"

class Node3DEditor : public VBoxContainer {
	GDCLASS(Node3DEditor, VBoxContainer);

public:
	static const unsigned int VIEWPORTS_COUNT = 4;

	static inline const real_t DISTANCE_DEFAULT = 4;

	static inline const real_t GIZMO_ARROW_SIZE = 0.35;
	static inline const real_t GIZMO_RING_HALF_WIDTH = 0.1;
	static inline const real_t GIZMO_PLANE_SIZE = 0.2;
	static inline const real_t GIZMO_PLANE_DST = 0.3;
	static inline const real_t GIZMO_CIRCLE_SIZE = 1.1;
	static inline const real_t GIZMO_SCALE_OFFSET = GIZMO_CIRCLE_SIZE + 0.3;
	static inline const real_t GIZMO_ARROW_OFFSET = GIZMO_CIRCLE_SIZE + 0.3;

	static inline const real_t ZOOM_FREELOOK_MIN = 0.01;
	static inline const real_t ZOOM_FREELOOK_MULTIPLIER = 1.08;
	static inline const real_t ZOOM_FREELOOK_INDICATOR_DELAY_S = 1.5;

#ifdef REAL_T_IS_DOUBLE
	static inline const double ZOOM_FREELOOK_MAX = 1'000'000'000'000;
#else
	static inline const float ZOOM_FREELOOK_MAX = 10'000;
#endif

	static inline const real_t MIN_Z = 0.01;
	static inline const real_t MAX_Z = 1000000.0;

	static inline const real_t MIN_FOV = 0.01;
	static inline const real_t MAX_FOV = 179;

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
	EditorSelection *editor_selection = nullptr;

	Node3DEditorViewportContainer *viewport_base = nullptr;
	Node3DEditorViewport *viewports[VIEWPORTS_COUNT];
	VSplitContainer *shader_split = nullptr;
	HSplitContainer *left_panel_split = nullptr;
	HSplitContainer *right_panel_split = nullptr;

	/////

	ToolMode tool_mode;

	RID origin;
	RID origin_instance;
	bool origin_enabled = false;
	RID grid[3];
	RID grid_instance[3];
	bool grid_visible[3]; //currently visible
	bool grid_enable[3]; //should be always visible if true
	bool grid_enabled = false;
	bool grid_init_draw = false;
	Camera3D::ProjectionType grid_camera_last_update_perspective = Camera3D::PROJECTION_PERSPECTIVE;
	Vector3 grid_camera_last_update_position;

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
	Node3D *preview_node = nullptr;
	AABB preview_bounds;

	Ref<Material> preview_material;
	Ref<Material> preview_reset_material;
	ObjectID preview_material_target;
	int preview_material_surface = -1;

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

	MenuButton *transform_menu = nullptr;
	PopupMenu *gizmos_menu = nullptr;
	MenuButton *view_menu = nullptr;

	AcceptDialog *accept = nullptr;

	ConfirmationDialog *snap_dialog = nullptr;
	ConfirmationDialog *xform_dialog = nullptr;
	ConfirmationDialog *settings_dialog = nullptr;

	bool snap_enabled;
	bool snap_key_enabled;
	LineEdit *snap_translate = nullptr;
	LineEdit *snap_rotate = nullptr;
	LineEdit *snap_scale = nullptr;

	LineEdit *xform_translate[3];
	LineEdit *xform_rotate[3];
	LineEdit *xform_scale[3];
	OptionButton *xform_type = nullptr;

	VBoxContainer *settings_vbc = nullptr;
	SpinBox *settings_fov = nullptr;
	SpinBox *settings_znear = nullptr;
	SpinBox *settings_zfar = nullptr;

	void _snap_changed();
	void _snap_update();
	void _xform_dialog_action();
	void _menu_item_pressed(int p_option);
	void _menu_item_toggled(bool pressed, int p_option);
	void _menu_gizmo_toggled(int p_option);
	void _update_camera_override_button(bool p_game_running);
	void _update_camera_override_viewport(Object *p_viewport);
	// Used for secondary menu items which are displayed depending on the currently selected node
	// (such as MeshInstance's "Mesh" menu).
	PanelContainer *context_menu_panel = nullptr;
	HBoxContainer *context_menu_hbox = nullptr;

	void _generate_selection_boxes();

	int camera_override_viewport_id;

	void _init_indicators();
	void _update_gizmos_menu();
	void _update_gizmos_menu_theme();
	void _init_grid();
	void _finish_indicators();
	void _finish_grid();

	void _toggle_maximize_view(Object *p_viewport);

	Node *custom_camera = nullptr;

	Object *_get_editor_data(Object *p_what);

	Ref<Environment> viewport_environment;

	Node3D *selected = nullptr;

	void _request_gizmo(Object *p_obj);
	void _request_gizmo_for_id(ObjectID p_id);
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

	bool do_snap_selected_nodes_to_floor = false;
	void _snap_selected_nodes_to_floor();

	// Preview Sun and Environment

	uint32_t world_env_count = 0;
	uint32_t directional_light_count = 0;

	Button *sun_button = nullptr;
	Label *sun_state = nullptr;
	Label *sun_title = nullptr;
	VBoxContainer *sun_vb = nullptr;
	Popup *sun_environ_popup = nullptr;
	Control *sun_direction = nullptr;
	EditorSpinSlider *sun_angle_altitude = nullptr;
	EditorSpinSlider *sun_angle_azimuth = nullptr;
	ColorPickerButton *sun_color = nullptr;
	EditorSpinSlider *sun_energy = nullptr;
	EditorSpinSlider *sun_max_distance = nullptr;
	Button *sun_add_to_scene = nullptr;

	void _sun_direction_draw();
	void _sun_direction_input(const Ref<InputEvent> &p_event);
	void _sun_direction_angle_set();

	Vector2 sun_rotation;

	Ref<Shader> sun_direction_shader;
	Ref<ShaderMaterial> sun_direction_material;

	Button *environ_button = nullptr;
	Label *environ_state = nullptr;
	Label *environ_title = nullptr;
	VBoxContainer *environ_vb = nullptr;
	ColorPickerButton *environ_sky_color = nullptr;
	ColorPickerButton *environ_ground_color = nullptr;
	EditorSpinSlider *environ_energy = nullptr;
	Button *environ_ao_button = nullptr;
	Button *environ_glow_button = nullptr;
	Button *environ_tonemap_button = nullptr;
	Button *environ_gi_button = nullptr;
	Button *environ_add_to_scene = nullptr;

	Button *sun_environ_settings = nullptr;

	DirectionalLight3D *preview_sun = nullptr;
	bool preview_sun_dangling = false;
	WorldEnvironment *preview_environment = nullptr;
	bool preview_env_dangling = false;
	Ref<Environment> environment;
	Ref<CameraAttributesPractical> camera_attributes;
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
	virtual void shortcut_input(const Ref<InputEvent> &p_event) override;

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

	Dictionary get_state() const;
	void set_state(const Dictionary &p_state);

	Ref<Environment> get_viewport_environment() { return viewport_environment; }

	void add_control_to_menu_panel(Control *p_control);
	void remove_control_from_menu_panel(Control *p_control);

	void add_control_to_left_panel(Control *p_control);
	void remove_control_from_left_panel(Control *p_control);

	void add_control_to_right_panel(Control *p_control);
	void remove_control_from_right_panel(Control *p_control);

	void move_control_to_left_panel(Control *p_control);
	void move_control_to_right_panel(Control *p_control);

	VSplitContainer *get_shader_split();

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

	void set_preview_material(Ref<Material> p_material) { preview_material = p_material; }
	Ref<Material> get_preview_material() { return preview_material; }
	void set_preview_reset_material(Ref<Material> p_material) { preview_reset_material = p_material; }
	Ref<Material> get_preview_reset_material() const { return preview_reset_material; }
	void set_preview_material_target(ObjectID p_object_id) { preview_material_target = p_object_id; }
	ObjectID get_preview_material_target() const { return preview_material_target; }
	void set_preview_material_surface(int p_surface) { preview_material_surface = p_surface; }
	int get_preview_material_surface() const { return preview_material_surface; }

	Node3DEditorViewport *get_editor_viewport(int p_idx) {
		ERR_FAIL_INDEX_V(p_idx, static_cast<int>(VIEWPORTS_COUNT), nullptr);
		return viewports[p_idx];
	}

	void add_gizmo_plugin(Ref<EditorNode3DGizmoPlugin> p_plugin);
	void remove_gizmo_plugin(Ref<EditorNode3DGizmoPlugin> p_plugin);

	void edit(Node3D *p_spatial);
	void clear();

	Node3DEditor();
	~Node3DEditor();
};

#endif // NODE_3D_EDITOR_H
