/*************************************************************************/
/*  canvas_item_editor_plugin.h                                          */
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

#ifndef CONTROL_EDITOR_PLUGIN_H
#define CONTROL_EDITOR_PLUGIN_H

#include "editor/editor_node.h"
#include "editor/editor_plugin.h"
#include "editor/editor_zoom_widget.h"
#include "scene/gui/box_container.h"
#include "scene/gui/check_box.h"
#include "scene/gui/label.h"
#include "scene/gui/panel_container.h"
#include "scene/gui/spin_box.h"
#include "scene/main/canvas_item.h"

class CanvasItemEditorViewport;

class CanvasItemEditorSelectedItem : public Object {
	GDCLASS(CanvasItemEditorSelectedItem, Object);

public:
	Transform2D prev_xform;
	real_t prev_rot = 0;
	Rect2 prev_rect;
	Vector2 prev_pivot;
	real_t prev_anchors[4] = { (real_t)0.0 };

	Transform2D pre_drag_xform;
	Rect2 pre_drag_rect;

	List<real_t> pre_drag_bones_length;
	List<Dictionary> pre_drag_bones_undo_state;

	Dictionary undo_state;

	CanvasItemEditorSelectedItem() {}
};

class CanvasItemEditor : public VBoxContainer {
	GDCLASS(CanvasItemEditor, VBoxContainer);

public:
	enum Tool {
		TOOL_SELECT,
		TOOL_LIST_SELECT,
		TOOL_MOVE,
		TOOL_SCALE,
		TOOL_ROTATE,
		TOOL_EDIT_PIVOT,
		TOOL_PAN,
		TOOL_RULER,
		TOOL_MAX
	};

	enum AddNodeOption {
		ADD_NODE,
		ADD_INSTANCE,
	};

private:
	EditorNode *editor;

	enum SnapTarget {
		SNAP_TARGET_NONE = 0,
		SNAP_TARGET_PARENT,
		SNAP_TARGET_SELF_ANCHORS,
		SNAP_TARGET_SELF,
		SNAP_TARGET_OTHER_NODE,
		SNAP_TARGET_GUIDE,
		SNAP_TARGET_GRID,
		SNAP_TARGET_PIXEL
	};

	enum MenuOption {
		SNAP_USE,
		SNAP_USE_NODE_PARENT,
		SNAP_USE_NODE_ANCHORS,
		SNAP_USE_NODE_SIDES,
		SNAP_USE_NODE_CENTER,
		SNAP_USE_OTHER_NODES,
		SNAP_USE_GRID,
		SNAP_USE_GUIDES,
		SNAP_USE_ROTATION,
		SNAP_USE_SCALE,
		SNAP_RELATIVE,
		SNAP_CONFIGURE,
		SNAP_USE_PIXEL,
		SHOW_GRID,
		SHOW_HELPERS,
		SHOW_RULERS,
		SHOW_GUIDES,
		SHOW_ORIGIN,
		SHOW_VIEWPORT,
		SHOW_EDIT_LOCKS,
		SHOW_TRANSFORMATION_GIZMOS,
		LOCK_SELECTED,
		UNLOCK_SELECTED,
		GROUP_SELECTED,
		UNGROUP_SELECTED,
		ANCHORS_AND_OFFSETS_PRESET_TOP_LEFT,
		ANCHORS_AND_OFFSETS_PRESET_TOP_RIGHT,
		ANCHORS_AND_OFFSETS_PRESET_BOTTOM_LEFT,
		ANCHORS_AND_OFFSETS_PRESET_BOTTOM_RIGHT,
		ANCHORS_AND_OFFSETS_PRESET_CENTER_LEFT,
		ANCHORS_AND_OFFSETS_PRESET_CENTER_RIGHT,
		ANCHORS_AND_OFFSETS_PRESET_CENTER_TOP,
		ANCHORS_AND_OFFSETS_PRESET_CENTER_BOTTOM,
		ANCHORS_AND_OFFSETS_PRESET_CENTER,
		ANCHORS_AND_OFFSETS_PRESET_TOP_WIDE,
		ANCHORS_AND_OFFSETS_PRESET_LEFT_WIDE,
		ANCHORS_AND_OFFSETS_PRESET_RIGHT_WIDE,
		ANCHORS_AND_OFFSETS_PRESET_BOTTOM_WIDE,
		ANCHORS_AND_OFFSETS_PRESET_VCENTER_WIDE,
		ANCHORS_AND_OFFSETS_PRESET_HCENTER_WIDE,
		ANCHORS_AND_OFFSETS_PRESET_WIDE,
		ANCHORS_AND_OFFSETS_PRESET_KEEP_RATIO,
		ANCHORS_PRESET_TOP_LEFT,
		ANCHORS_PRESET_TOP_RIGHT,
		ANCHORS_PRESET_BOTTOM_LEFT,
		ANCHORS_PRESET_BOTTOM_RIGHT,
		ANCHORS_PRESET_CENTER_LEFT,
		ANCHORS_PRESET_CENTER_RIGHT,
		ANCHORS_PRESET_CENTER_TOP,
		ANCHORS_PRESET_CENTER_BOTTOM,
		ANCHORS_PRESET_CENTER,
		ANCHORS_PRESET_TOP_WIDE,
		ANCHORS_PRESET_LEFT_WIDE,
		ANCHORS_PRESET_RIGHT_WIDE,
		ANCHORS_PRESET_BOTTOM_WIDE,
		ANCHORS_PRESET_VCENTER_WIDE,
		ANCHORS_PRESET_HCENTER_WIDE,
		ANCHORS_PRESET_WIDE,
		OFFSETS_PRESET_TOP_LEFT,
		OFFSETS_PRESET_TOP_RIGHT,
		OFFSETS_PRESET_BOTTOM_LEFT,
		OFFSETS_PRESET_BOTTOM_RIGHT,
		OFFSETS_PRESET_CENTER_LEFT,
		OFFSETS_PRESET_CENTER_RIGHT,
		OFFSETS_PRESET_CENTER_TOP,
		OFFSETS_PRESET_CENTER_BOTTOM,
		OFFSETS_PRESET_CENTER,
		OFFSETS_PRESET_TOP_WIDE,
		OFFSETS_PRESET_LEFT_WIDE,
		OFFSETS_PRESET_RIGHT_WIDE,
		OFFSETS_PRESET_BOTTOM_WIDE,
		OFFSETS_PRESET_VCENTER_WIDE,
		OFFSETS_PRESET_HCENTER_WIDE,
		OFFSETS_PRESET_WIDE,
		ANIM_INSERT_KEY,
		ANIM_INSERT_KEY_EXISTING,
		ANIM_INSERT_POS,
		ANIM_INSERT_ROT,
		ANIM_INSERT_SCALE,
		ANIM_COPY_POSE,
		ANIM_PASTE_POSE,
		ANIM_CLEAR_POSE,
		CLEAR_GUIDES,
		VIEW_CENTER_TO_SELECTION,
		VIEW_FRAME_TO_SELECTION,
		PREVIEW_CANVAS_SCALE,
		SKELETON_MAKE_BONES,
		SKELETON_SHOW_BONES
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
		DRAG_V_GUIDE,
		DRAG_H_GUIDE,
		DRAG_DOUBLE_GUIDE,
		DRAG_KEY_MOVE
	};

	bool selection_menu_additive_selection;

	Tool tool;
	Control *viewport;
	Control *viewport_scrollable;

	HScrollBar *h_scroll;
	VScrollBar *v_scroll;
	HBoxContainer *hb;
	// Used for secondary menu items which are displayed depending on the currently selected node
	// (such as MeshInstance's "Mesh" menu).
	PanelContainer *context_menu_container;
	HBoxContainer *hbc_context_menu;

	Transform2D transform;
	bool show_grid;
	bool show_rulers;
	bool show_guides;
	bool show_origin;
	bool show_viewport;
	bool show_helpers;
	bool show_edit_locks;
	bool show_transformation_gizmos;

	real_t zoom;
	Point2 view_offset;
	Point2 previous_update_view_offset;

	bool selected_from_canvas;
	bool anchors_mode;

	Point2 grid_offset;
	Point2 grid_step;
	int primary_grid_steps;
	int grid_step_multiplier;

	real_t snap_rotation_step;
	real_t snap_rotation_offset;
	real_t snap_scale_step;
	bool smart_snap_active;
	bool grid_snap_active;

	bool snap_node_parent;
	bool snap_node_anchors;
	bool snap_node_sides;
	bool snap_node_center;
	bool snap_other_nodes;
	bool snap_guides;
	bool snap_rotation;
	bool snap_scale;
	bool snap_relative;
	bool snap_pixel;
	bool key_pos;
	bool key_rot;
	bool key_scale;
	bool panning;
	bool pan_pressed;

	bool ruler_tool_active;
	Point2 ruler_tool_origin;
	Point2 node_create_position;

	MenuOption last_option;

	struct _SelectResult {
		CanvasItem *item = nullptr;
		real_t z_index = 0;
		bool has_z = true;
		_FORCE_INLINE_ bool operator<(const _SelectResult &p_rr) const {
			return has_z && p_rr.has_z ? p_rr.z_index < z_index : p_rr.has_z;
		}
	};
	Vector<_SelectResult> selection_results;

	struct _HoverResult {
		Point2 position;
		Ref<Texture2D> icon;
		String name;
	};
	Vector<_HoverResult> hovering_results;

	struct BoneList {
		Transform2D xform;
		real_t length = 0;
		uint64_t last_pass = 0;
	};

	uint64_t bone_last_frame;

	struct BoneKey {
		ObjectID from;
		ObjectID to;
		_FORCE_INLINE_ bool operator<(const BoneKey &p_key) const {
			if (from == p_key.from) {
				return to < p_key.to;
			} else {
				return from < p_key.from;
			}
		}
	};

	Map<BoneKey, BoneList> bone_list;

	struct PoseClipboard {
		Vector2 pos;
		Vector2 scale;
		real_t rot = 0;
		ObjectID id;
	};
	List<PoseClipboard> pose_clipboard;

	Button *select_button;

	Button *move_button;
	Button *scale_button;
	Button *rotate_button;

	Button *list_select_button;
	Button *pivot_button;
	Button *pan_button;

	Button *ruler_button;

	Button *smart_snap_button;
	Button *grid_snap_button;
	MenuButton *snap_config_menu;
	PopupMenu *smartsnap_config_popup;

	Button *lock_button;
	Button *unlock_button;

	Button *group_button;
	Button *ungroup_button;

	MenuButton *skeleton_menu;
	Button *override_camera_button;
	MenuButton *view_menu;
	HBoxContainer *animation_hb;
	MenuButton *animation_menu;

	MenuButton *presets_menu;
	PopupMenu *anchors_and_margins_popup;
	PopupMenu *anchors_popup;

	Button *anchor_mode_button;

	Button *key_loc_button;
	Button *key_rot_button;
	Button *key_scale_button;
	Button *key_insert_button;
	Button *key_auto_insert_button;

	PopupMenu *selection_menu;
	PopupMenu *add_node_menu;

	Control *top_ruler;
	Control *left_ruler;

	Point2 drag_start_origin;
	DragType drag_type;
	Point2 drag_from;
	Point2 drag_to;
	Point2 drag_rotation_center;
	List<CanvasItem *> drag_selection;
	int dragged_guide_index;
	Point2 dragged_guide_pos;
	bool is_hovering_h_guide;
	bool is_hovering_v_guide;

	bool updating_value_dialog;

	Point2 box_selecting_to;

	Ref<StyleBoxTexture> select_sb;
	Ref<Texture2D> select_handle;
	Ref<Texture2D> anchor_handle;

	Ref<Shortcut> drag_pivot_shortcut;
	Ref<Shortcut> set_pivot_shortcut;
	Ref<Shortcut> multiply_grid_step_shortcut;
	Ref<Shortcut> divide_grid_step_shortcut;
	Ref<Shortcut> pan_view_shortcut;

	bool _is_node_locked(const Node *p_node);
	bool _is_node_movable(const Node *p_node, bool p_popup_warning = false);
	void _find_canvas_items_at_pos(const Point2 &p_pos, Node *p_node, Vector<_SelectResult> &r_items, const Transform2D &p_parent_xform = Transform2D(), const Transform2D &p_canvas_xform = Transform2D());
	void _get_canvas_items_at_pos(const Point2 &p_pos, Vector<_SelectResult> &r_items, bool p_allow_locked = false);

	void _find_canvas_items_in_rect(const Rect2 &p_rect, Node *p_node, List<CanvasItem *> *r_items, const Transform2D &p_parent_xform = Transform2D(), const Transform2D &p_canvas_xform = Transform2D());
	bool _select_click_on_item(CanvasItem *item, Point2 p_click_pos, bool p_append);

	ConfirmationDialog *snap_dialog;

	CanvasItem *ref_item;

	void _save_canvas_item_state(List<CanvasItem *> p_canvas_items, bool save_bones = false);
	void _restore_canvas_item_state(List<CanvasItem *> p_canvas_items, bool restore_bones = false);
	void _commit_canvas_item_state(List<CanvasItem *> p_canvas_items, String action_name, bool commit_bones = false);

	Vector2 _anchor_to_position(const Control *p_control, Vector2 anchor);
	Vector2 _position_to_anchor(const Control *p_control, Vector2 position);

	void _popup_callback(int p_op);
	bool updating_scroll;
	void _update_scroll(real_t);
	void _update_scrollbars();
	void _snap_changed();
	void _selection_result_pressed(int);
	void _selection_menu_hide();
	void _add_node_pressed(int p_result);
	void _node_created(Node *p_node);
	void _reset_create_position();

	UndoRedo *undo_redo;

	List<CanvasItem *> _get_edited_canvas_items(bool retreive_locked = false, bool remove_canvas_item_if_parent_in_selection = true);
	Rect2 _get_encompassing_rect_from_list(List<CanvasItem *> p_list);
	void _expand_encompassing_rect_using_children(Rect2 &r_rect, const Node *p_node, bool &r_first, const Transform2D &p_parent_xform = Transform2D(), const Transform2D &p_canvas_xform = Transform2D(), bool include_locked_nodes = true);
	Rect2 _get_encompassing_rect(const Node *p_node);

	Object *_get_editor_data(Object *p_what);

	void _insert_animation_keys(bool p_location, bool p_rotation, bool p_scale, bool p_on_existing);

	void _keying_changed();

	virtual void unhandled_key_input(const Ref<InputEvent> &p_ev) override;

	void _draw_text_at_position(Point2 p_position, String p_string, Side p_side);
	void _draw_margin_at_position(int p_value, Point2 p_position, Side p_side);
	void _draw_percentage_at_position(real_t p_value, Point2 p_position, Side p_side);
	void _draw_straight_line(Point2 p_from, Point2 p_to, Color p_color);

	void _draw_smart_snapping();
	void _draw_rulers();
	void _draw_guides();
	void _draw_focus();
	void _draw_grid();
	void _draw_ruler_tool();
	void _draw_control_anchors(Control *control);
	void _draw_control_helpers(Control *control);
	void _draw_selection();
	void _draw_axis();
	void _draw_invisible_nodes_positions(Node *p_node, const Transform2D &p_parent_xform = Transform2D(), const Transform2D &p_canvas_xform = Transform2D());
	void _draw_locks_and_groups(Node *p_node, const Transform2D &p_parent_xform = Transform2D(), const Transform2D &p_canvas_xform = Transform2D());
	void _draw_hover();

	void _draw_viewport();

	bool _gui_input_anchors(const Ref<InputEvent> &p_event);
	bool _gui_input_move(const Ref<InputEvent> &p_event);
	bool _gui_input_open_scene_on_double_click(const Ref<InputEvent> &p_event);
	bool _gui_input_scale(const Ref<InputEvent> &p_event);
	bool _gui_input_pivot(const Ref<InputEvent> &p_event);
	bool _gui_input_resize(const Ref<InputEvent> &p_event);
	bool _gui_input_rotate(const Ref<InputEvent> &p_event);
	bool _gui_input_select(const Ref<InputEvent> &p_event);
	bool _gui_input_ruler_tool(const Ref<InputEvent> &p_event);
	bool _gui_input_zoom_or_pan(const Ref<InputEvent> &p_event, bool p_already_accepted);
	bool _gui_input_rulers_and_guides(const Ref<InputEvent> &p_event);
	bool _gui_input_hover(const Ref<InputEvent> &p_event);

	void _gui_input_viewport(const Ref<InputEvent> &p_event);
	void _update_cursor();

	void _selection_changed();

	void _focus_selection(int p_op);

	SnapTarget snap_target[2];
	Transform2D snap_transform;
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
			const SnapTarget p_snap_target, List<const CanvasItem *> p_exceptions,
			const Node *p_current);

	void _set_anchors_preset(Control::LayoutPreset p_preset);
	void _set_anchors_and_offsets_preset(Control::LayoutPreset p_preset);
	void _set_anchors_and_offsets_to_keep_ratio();

	void _button_toggle_anchor_mode(bool p_status);

	VBoxContainer *controls_vb;
	EditorZoomWidget *zoom_widget;
	void _update_zoom(real_t p_zoom);
	void _zoom_on_position(real_t p_zoom, Point2 p_position = Point2());
	void _button_toggle_smart_snap(bool p_status);
	void _button_toggle_grid_snap(bool p_status);
	void _button_override_camera(bool p_pressed);
	void _button_tool_select(int p_index);

	void _update_override_camera_button(bool p_game_running);

	HSplitContainer *palette_split;
	VSplitContainer *bottom_split;

	void _update_context_menu_stylebox();

	void _set_owner_for_node_and_children(Node *p_node, Node *p_owner);

	friend class CanvasItemEditorPlugin;

protected:
	void _notification(int p_what);

	static void _bind_methods();

	HBoxContainer *get_panel_hb() { return hb; }

	static CanvasItemEditor *singleton;

public:
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

	Point2 snap_point(Point2 p_target, unsigned int p_modes = SNAP_DEFAULT, unsigned int p_forced_modes = 0, const CanvasItem *p_self_canvas_item = nullptr, List<CanvasItem *> p_other_nodes_exceptions = List<CanvasItem *>());
	real_t snap_angle(real_t p_target, real_t p_start = 0) const;

	Transform2D get_canvas_transform() const { return transform; }

	static CanvasItemEditor *get_singleton() { return singleton; }
	Dictionary get_state() const;
	void set_state(const Dictionary &p_state);

	void add_control_to_menu_panel(Control *p_control);
	void remove_control_from_menu_panel(Control *p_control);

	HSplitContainer *get_palette_split();
	VSplitContainer *get_bottom_split();

	Control *get_viewport_control() { return viewport; }

	Control *get_controls_container() { return controls_vb; }

	void update_viewport();

	Tool get_current_tool() { return tool; }
	void set_current_tool(Tool p_tool);

	void set_undo_redo(UndoRedo *p_undo_redo) { undo_redo = p_undo_redo; }
	void edit(CanvasItem *p_canvas_item);

	void focus_selection();

	bool is_anchors_mode_enabled() { return anchors_mode; };

	EditorSelection *editor_selection;

	CanvasItemEditor(EditorNode *p_editor);
};

class CanvasItemEditorPlugin : public EditorPlugin {
	GDCLASS(CanvasItemEditorPlugin, EditorPlugin);

	CanvasItemEditor *canvas_item_editor;
	EditorNode *editor;

public:
	virtual String get_name() const override { return "2D"; }
	bool has_main_screen() const override { return true; }
	virtual void edit(Object *p_object) override;
	virtual bool handles(Object *p_object) const override;
	virtual void make_visible(bool p_visible) override;
	virtual Dictionary get_state() const override;
	virtual void set_state(const Dictionary &p_state) override;

	CanvasItemEditor *get_canvas_item_editor() { return canvas_item_editor; }

	CanvasItemEditorPlugin(EditorNode *p_node);
	~CanvasItemEditorPlugin();
};

class CanvasItemEditorViewport : public Control {
	GDCLASS(CanvasItemEditorViewport, Control);

	// The type of node that will be created when dropping texture into the viewport.
	String default_texture_node_type;
	// Node types that are available to select from when dropping texture into viewport.
	Vector<String> texture_node_types;

	Vector<String> selected_files;
	Node *target_node;
	Point2 drop_pos;

	EditorNode *editor;
	EditorData *editor_data;
	CanvasItemEditor *canvas_item_editor;
	Control *preview_node;
	AcceptDialog *accept;
	AcceptDialog *selector;
	Label *selector_label;
	Label *label;
	Label *label_desc;
	VBoxContainer *btn_group;
	Ref<ButtonGroup> button_group;

	void _on_mouse_exit();
	void _on_select_type(Object *selected);
	void _on_change_type_confirmed();
	void _on_change_type_closed();
	Node *_make_texture_node_type(String texture_node_type);

	void _create_preview(const Vector<String> &files) const;
	void _remove_preview();

	bool _cyclical_dependency_exists(const String &p_target_scene_path, Node *p_desired_node);
	bool _only_packed_scenes_selected() const;
	void _create_nodes(Node *parent, Node *child, String &path, const Point2 &p_point);
	bool _create_instance(Node *parent, String &path, const Point2 &p_point);
	void _perform_drop_data();
	void _show_resource_type_selector();

	static void _bind_methods();

protected:
	void _notification(int p_what);

public:
	virtual bool can_drop_data(const Point2 &p_point, const Variant &p_data) const override;
	virtual void drop_data(const Point2 &p_point, const Variant &p_data) override;

	CanvasItemEditorViewport(EditorNode *p_node, CanvasItemEditor *p_canvas_item_editor);
	~CanvasItemEditorViewport();
};

#endif
