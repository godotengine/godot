/*************************************************************************/
/*  canvas_item_editor_plugin.h                                          */
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
#ifndef CONTROL_EDITOR_PLUGIN_H
#define CONTROL_EDITOR_PLUGIN_H

#include "editor/editor_node.h"
#include "editor/editor_plugin.h"
#include "scene/2d/canvas_item.h"
#include "scene/gui/box_container.h"
#include "scene/gui/check_box.h"
#include "scene/gui/label.h"
#include "scene/gui/panel_container.h"
#include "scene/gui/spin_box.h"
/**
	@author Juan Linietsky <reduzio@gmail.com>
*/

class CanvasItemEditorViewport;

class CanvasItemEditorSelectedItem : public Object {

	GDCLASS(CanvasItemEditorSelectedItem, Object);

public:
	Variant undo_state;
	Vector2 undo_pivot;

	Transform2D prev_xform;
	float prev_rot;
	Rect2 prev_rect;
	Vector2 prev_pivot;
	float prev_anchors[4];

	Transform2D pre_drag_xform;
	Rect2 pre_drag_rect;

	CanvasItemEditorSelectedItem() { prev_rot = 0; }
};

class CanvasItemEditor : public VBoxContainer {

	GDCLASS(CanvasItemEditor, VBoxContainer);

	EditorNode *editor;

	enum Tool {
		TOOL_SELECT,
		TOOL_LIST_SELECT,
		TOOL_MOVE,
		TOOL_ROTATE,
		TOOL_EDIT_PIVOT,
		TOOL_PAN,
		TOOL_MAX
	};

	enum MenuOption {
		SNAP_USE,
		SNAP_USE_NODE_PARENT,
		SNAP_USE_NODE_ANCHORS,
		SNAP_USE_NODE_SIDES,
		SNAP_USE_OTHER_NODES,
		SNAP_USE_GRID,
		SNAP_USE_GUIDES,
		SNAP_USE_ROTATION,
		SNAP_RELATIVE,
		SNAP_CONFIGURE,
		SNAP_USE_PIXEL,
		SHOW_GRID,
		SHOW_HELPERS,
		SHOW_RULERS,
		SHOW_GUIDES,
		LOCK_SELECTED,
		UNLOCK_SELECTED,
		GROUP_SELECTED,
		UNGROUP_SELECTED,
		ANCHORS_AND_MARGINS_PRESET_TOP_LEFT,
		ANCHORS_AND_MARGINS_PRESET_TOP_RIGHT,
		ANCHORS_AND_MARGINS_PRESET_BOTTOM_LEFT,
		ANCHORS_AND_MARGINS_PRESET_BOTTOM_RIGHT,
		ANCHORS_AND_MARGINS_PRESET_CENTER_LEFT,
		ANCHORS_AND_MARGINS_PRESET_CENTER_RIGHT,
		ANCHORS_AND_MARGINS_PRESET_CENTER_TOP,
		ANCHORS_AND_MARGINS_PRESET_CENTER_BOTTOM,
		ANCHORS_AND_MARGINS_PRESET_CENTER,
		ANCHORS_AND_MARGINS_PRESET_TOP_WIDE,
		ANCHORS_AND_MARGINS_PRESET_LEFT_WIDE,
		ANCHORS_AND_MARGINS_PRESET_RIGHT_WIDE,
		ANCHORS_AND_MARGINS_PRESET_BOTTOM_WIDE,
		ANCHORS_AND_MARGINS_PRESET_VCENTER_WIDE,
		ANCHORS_AND_MARGINS_PRESET_HCENTER_WIDE,
		ANCHORS_AND_MARGINS_PRESET_WIDE,
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
		MARGINS_PRESET_TOP_LEFT,
		MARGINS_PRESET_TOP_RIGHT,
		MARGINS_PRESET_BOTTOM_LEFT,
		MARGINS_PRESET_BOTTOM_RIGHT,
		MARGINS_PRESET_CENTER_LEFT,
		MARGINS_PRESET_CENTER_RIGHT,
		MARGINS_PRESET_CENTER_TOP,
		MARGINS_PRESET_CENTER_BOTTOM,
		MARGINS_PRESET_CENTER,
		MARGINS_PRESET_TOP_WIDE,
		MARGINS_PRESET_LEFT_WIDE,
		MARGINS_PRESET_RIGHT_WIDE,
		MARGINS_PRESET_BOTTOM_WIDE,
		MARGINS_PRESET_VCENTER_WIDE,
		MARGINS_PRESET_HCENTER_WIDE,
		MARGINS_PRESET_WIDE,
		ANIM_INSERT_KEY,
		ANIM_INSERT_KEY_EXISTING,
		ANIM_INSERT_POS,
		ANIM_INSERT_ROT,
		ANIM_INSERT_SCALE,
		ANIM_COPY_POSE,
		ANIM_PASTE_POSE,
		ANIM_CLEAR_POSE,
		VIEW_CENTER_TO_SELECTION,
		VIEW_FRAME_TO_SELECTION,
		SKELETON_MAKE_BONES,
		SKELETON_CLEAR_BONES,
		SKELETON_SHOW_BONES,
		SKELETON_SET_IK_CHAIN,
		SKELETON_CLEAR_IK_CHAIN

	};

	enum DragType {
		DRAG_NONE,
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
		DRAG_ALL,
		DRAG_ROTATE,
		DRAG_PIVOT,
		DRAG_NODE_2D,
		DRAG_V_GUIDE,
		DRAG_H_GUIDE,
		DRAG_DOUBLE_GUIDE,
	};

	enum KeyMoveMODE {
		MOVE_VIEW_BASE,
		MOVE_LOCAL_BASE,
		MOVE_LOCAL_WITH_ROT
	};

	EditorSelection *editor_selection;
	bool additive_selection;

	Tool tool;
	bool first_update;
	Control *viewport;
	Control *viewport_base;
	Control *viewport_scrollable;

	bool can_move_pivot;

	HScrollBar *h_scroll;
	VScrollBar *v_scroll;
	HBoxContainer *hb;

	ToolButton *zoom_minus;
	ToolButton *zoom_reset;
	ToolButton *zoom_plus;

	Transform2D transform;
	bool show_grid;
	bool show_rulers;
	bool show_guides;
	bool show_helpers;
	float zoom;

	Point2 grid_offset;
	Point2 grid_step;
	int grid_step_multiplier;

	float snap_rotation_step;
	float snap_rotation_offset;
	bool snap_active;
	bool snap_node_parent;
	bool snap_node_anchors;
	bool snap_node_sides;
	bool snap_other_nodes;
	bool snap_grid;
	bool snap_guides;
	bool snap_rotation;
	bool snap_relative;
	bool snap_pixel;
	bool skeleton_show_bones;
	bool box_selecting;
	Point2 box_selecting_to;
	bool key_pos;
	bool key_rot;
	bool key_scale;

	void _tool_select(int p_index);

	MenuOption last_option;

	struct _SelectResult {

		CanvasItem *item;
		float z;
		bool has_z;
		_FORCE_INLINE_ bool operator<(const _SelectResult &p_rr) const {
			return has_z && p_rr.has_z ? p_rr.z < z : p_rr.has_z;
		}
	};

	Vector<_SelectResult> selection_results;

	struct BoneList {

		Transform2D xform;
		Vector2 from;
		Vector2 to;
		ObjectID bone;
		uint64_t last_pass;
	};

	uint64_t bone_last_frame;
	Map<ObjectID, BoneList> bone_list;

	Transform2D bone_orig_xform;

	struct BoneIK {

		Variant orig_state;
		Vector2 pos;
		float len;
		Node2D *node;
	};

	List<BoneIK> bone_ik_list;

	struct PoseClipboard {

		Vector2 pos;
		Vector2 scale;
		float rot;
		ObjectID id;
	};

	List<PoseClipboard> pose_clipboard;

	ToolButton *select_button;
	ToolButton *list_select_button;
	ToolButton *move_button;
	ToolButton *rotate_button;

	ToolButton *snap_button;
	MenuButton *snap_config_menu;
	PopupMenu *smartsnap_config_popup;

	ToolButton *pivot_button;
	ToolButton *pan_button;

	ToolButton *lock_button;
	ToolButton *unlock_button;

	ToolButton *group_button;
	ToolButton *ungroup_button;

	MenuButton *skeleton_menu;
	MenuButton *view_menu;
	HBoxContainer *animation_hb;
	MenuButton *animation_menu;

	MenuButton *presets_menu;
	PopupMenu *anchors_and_margins_popup;
	PopupMenu *anchors_popup;

	Button *key_loc_button;
	Button *key_rot_button;
	Button *key_scale_button;
	Button *key_insert_button;

	PopupMenu *selection_menu;

	Control *top_ruler;
	Control *left_ruler;

	//PopupMenu *popup;
	DragType drag;
	Point2 drag_from;
	Point2 drag_point_from;
	bool updating_value_dialog;
	Point2 display_rotate_from;
	Point2 display_rotate_to;

	int edited_guide_index;
	Point2 edited_guide_pos;

	Ref<StyleBoxTexture> select_sb;
	Ref<Texture> select_handle;
	Ref<Texture> anchor_handle;

	Ref<ShortCut> drag_pivot_shortcut;
	Ref<ShortCut> set_pivot_shortcut;
	Ref<ShortCut> multiply_grid_step_shortcut;
	Ref<ShortCut> divide_grid_step_shortcut;

	int handle_len;
	bool _is_part_of_subscene(CanvasItem *p_item);
	void _find_canvas_items_at_pos(const Point2 &p_pos, Node *p_node, const Transform2D &p_parent_xform, const Transform2D &p_canvas_xform, Vector<_SelectResult> &r_items, int limit = 0);
	void _find_canvas_items_at_rect(const Rect2 &p_rect, Node *p_node, const Transform2D &p_parent_xform, const Transform2D &p_canvas_xform, List<CanvasItem *> *r_items);

	void _select_click_on_empty_area(Point2 p_click_pos, bool p_append, bool p_box_selection);
	bool _select_click_on_item(CanvasItem *item, Point2 p_click_pos, bool p_append, bool p_drag);

	ConfirmationDialog *snap_dialog;

	CanvasItem *ref_item;

	void _edit_set_pivot(const Vector2 &mouse_pos);
	void _add_canvas_item(CanvasItem *p_canvas_item);
	void _remove_canvas_item(CanvasItem *p_canvas_item);
	void _clear_canvas_items();
	void _key_move(const Vector2 &p_dir, bool p_snap, KeyMoveMODE p_move_mode);
	void _list_select(const Ref<InputEventMouseButton> &b);

	DragType _get_resize_handle_drag_type(const Point2 &p_click, Vector2 &r_point);
	void _prepare_drag(const Point2 &p_click_pos);
	DragType _get_anchor_handle_drag_type(const Point2 &p_click, Vector2 &r_point);

	Vector2 _anchor_to_position(const Control *p_control, Vector2 anchor);
	Vector2 _position_to_anchor(const Control *p_control, Vector2 position);

	void _popup_callback(int p_op);
	bool updating_scroll;
	void _update_scroll(float);
	void _update_scrollbars();
	void _update_cursor();
	void incbeg(float &beg, float &end, float inc, float minsize, bool p_symmetric);
	void incend(float &beg, float &end, float inc, float minsize, bool p_symmetric);

	void _append_canvas_item(CanvasItem *p_item);
	void _snap_changed();
	void _selection_result_pressed(int);
	void _selection_menu_hide();

	UndoRedo *undo_redo;

	Point2 _find_topleftmost_point();

	void _build_bones_list(Node *p_node);

	void _get_encompassing_rect(Node *p_node, Rect2 &r_rect, const Transform2D &p_xform);

	Object *_get_editor_data(Object *p_what);

	CanvasItem *_get_single_item();
	int get_item_count();
	void _keying_changed();

	void _unhandled_key_input(const Ref<InputEvent> &p_ev);

	void _draw_text_at_position(Point2 p_position, String p_string, Margin p_side);
	void _draw_margin_at_position(int p_value, Point2 p_position, Margin p_side);
	void _draw_percentage_at_position(float p_value, Point2 p_position, Margin p_side);

	void _draw_rulers();
	void _draw_guides();
	void _draw_focus();
	void _draw_grid();
	void _draw_selection();
	void _draw_axis();
	void _draw_bones();
	void _draw_locks_and_groups(Node *p_node, const Transform2D &p_xform);

	void _draw_viewport();
	void _draw_viewport_base();

	void _gui_input_viewport(const Ref<InputEvent> &p_event);
	void _gui_input_viewport_base(const Ref<InputEvent> &p_event);

	void _focus_selection(int p_op);

	void _snap_if_closer_float(float p_value, float p_target_snap, float &r_current_snap, bool &r_snapped, float p_radius = 10.0);
	void _snap_if_closer_point(Point2 p_value, Point2 p_target_snap, Point2 &r_current_snap, bool (&r_snapped)[2], real_t rotation = 0.0, float p_radius = 10.0);
	void _snap_other_nodes(Point2 p_value, Point2 &r_current_snap, bool (&r_snapped)[2], const Node *p_current, const CanvasItem *p_to_snap = NULL);

	void _set_anchors_preset(Control::LayoutPreset p_preset);
	void _set_margins_preset(Control::LayoutPreset p_preset);
	void _set_anchors_and_margins_preset(Control::LayoutPreset p_preset);

	void _zoom_on_position(float p_zoom, Point2 p_position = Point2());
	void _zoom_minus();
	void _zoom_reset();
	void _zoom_plus();

	void _toggle_snap(bool p_status);

	HSplitContainer *palette_split;
	VSplitContainer *bottom_split;

	friend class CanvasItemEditorPlugin;

protected:
	void _notification(int p_what);

	static void _bind_methods();
	void end_drag();
	void box_selection_start(Point2 &click);
	bool box_selection_end();

	HBoxContainer *get_panel_hb() { return hb; }

	struct compare_items_x {
		bool operator()(const CanvasItem *a, const CanvasItem *b) const {
			return a->get_global_transform().elements[2].x < b->get_global_transform().elements[2].x;
		}
	};

	struct compare_items_y {
		bool operator()(const CanvasItem *a, const CanvasItem *b) const {
			return a->get_global_transform().elements[2].y < b->get_global_transform().elements[2].y;
		}
	};

	struct proj_vector2_x {
		float get(const Vector2 &v) { return v.x; }
		void set(Vector2 &v, float f) { v.x = f; }
	};

	struct proj_vector2_y {
		float get(const Vector2 &v) { return v.y; }
		void set(Vector2 &v, float f) { v.y = f; }
	};

	template <class P, class C>
	void space_selected_items();

	static CanvasItemEditor *singleton;

public:
	enum SnapMode {
		SNAP_GRID = 1 << 0,
		SNAP_GUIDES = 1 << 1,
		SNAP_PIXEL = 1 << 2,
		SNAP_NODE_PARENT = 1 << 3,
		SNAP_NODE_ANCHORS = 1 << 4,
		SNAP_NODE_SIDES = 1 << 5,
		SNAP_OTHER_NODES = 1 << 6,

		SNAP_DEFAULT = 0x07,
	};

	Point2 snap_point(Point2 p_target, unsigned int p_modes = SNAP_DEFAULT, const CanvasItem *p_canvas_item = NULL, unsigned int p_forced_modes = 0);
	float snap_angle(float p_target, float p_start = 0) const;

	Transform2D get_canvas_transform() const { return transform; }

	static CanvasItemEditor *get_singleton() { return singleton; }
	Dictionary get_state() const;
	void set_state(const Dictionary &p_state);

	void add_control_to_menu_panel(Control *p_control);

	HSplitContainer *get_palette_split();
	VSplitContainer *get_bottom_split();

	Control *get_viewport_control() { return viewport; }

	void set_undo_redo(UndoRedo *p_undo_redo) { undo_redo = p_undo_redo; }
	void edit(CanvasItem *p_canvas_item);

	void focus_selection();

	CanvasItemEditor(EditorNode *p_editor);
};

class CanvasItemEditorPlugin : public EditorPlugin {

	GDCLASS(CanvasItemEditorPlugin, EditorPlugin);

	CanvasItemEditor *canvas_item_editor;
	EditorNode *editor;

public:
	virtual String get_name() const { return "2D"; }
	bool has_main_screen() const { return true; }
	virtual void edit(Object *p_object);
	virtual bool handles(Object *p_object) const;
	virtual void make_visible(bool p_visible);
	virtual Dictionary get_state() const;
	virtual void set_state(const Dictionary &p_state);

	CanvasItemEditor *get_canvas_item_editor() { return canvas_item_editor; }

	CanvasItemEditorPlugin(EditorNode *p_node);
	~CanvasItemEditorPlugin();
};

class CanvasItemEditorViewport : public Control {
	GDCLASS(CanvasItemEditorViewport, Control);

	String default_type;
	Vector<String> types;

	Vector<String> selected_files;
	Node *target_node;
	Point2 drop_pos;

	EditorNode *editor;
	EditorData *editor_data;
	CanvasItemEditor *canvas;
	Node2D *preview_node;
	AcceptDialog *accept;
	WindowDialog *selector;
	Label *selector_label;
	Label *label;
	Label *label_desc;
	VBoxContainer *btn_group;
	Ref<ButtonGroup> button_group;

	void _on_mouse_exit();
	void _on_select_type(Object *selected);
	void _on_change_type_confirmed();
	void _on_change_type_closed();

	void _create_preview(const Vector<String> &files) const;
	void _remove_preview();

	bool _cyclical_dependency_exists(const String &p_target_scene_path, Node *p_desired_node);
	void _create_nodes(Node *parent, Node *child, String &path, const Point2 &p_point);
	bool _create_instance(Node *parent, String &path, const Point2 &p_point);
	void _perform_drop_data();
	void _show_resource_type_selector();

	static void _bind_methods();

protected:
	void _notification(int p_what);

public:
	virtual bool can_drop_data(const Point2 &p_point, const Variant &p_data) const;
	virtual void drop_data(const Point2 &p_point, const Variant &p_data);

	CanvasItemEditorViewport(EditorNode *p_node, CanvasItemEditor *p_canvas);
	~CanvasItemEditorViewport();
};

#endif
