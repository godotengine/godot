/*************************************************************************/
/*  canvas_item_editor_plugin.h                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2018 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2018 Godot Engine contributors (cf. AUTHORS.md)    */
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
	Transform2D prev_xform;
	float prev_rot;
	Rect2 prev_rect;
	Vector2 prev_pivot;
	float prev_anchors[4];

	Transform2D pre_drag_xform;
	Rect2 pre_drag_rect;

	List<float> pre_drag_bones_length;
	List<Dictionary> pre_drag_bones_undo_state;

	Dictionary undo_state;

	CanvasItemEditorSelectedItem() { prev_rot = 0; }
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
		TOOL_MAX
	};

private:
	EditorNode *editor;

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
		DRAG_MOVE,
		DRAG_SCALE_X,
		DRAG_SCALE_Y,
		DRAG_SCALE_BOTH,
		DRAG_ROTATE,
		DRAG_PIVOT,
		DRAG_V_GUIDE,
		DRAG_H_GUIDE,
		DRAG_DOUBLE_GUIDE,
		DRAG_KEY_MOVE,
		DRAG_PAN
	};

	EditorSelection *editor_selection;
	bool selection_menu_additive_selection;

	Tool tool;
	Control *viewport;
	Control *viewport_scrollable;

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
	bool show_origin;
	bool show_viewport;
	bool show_helpers;
	bool show_edit_locks;
	float zoom;
	Point2 view_offset;
	Point2 previous_update_view_offset;

	Point2 grid_offset;
	Point2 grid_step;
	int grid_step_multiplier;

	float snap_rotation_step;
	float snap_rotation_offset;
	bool snap_active;
	bool snap_node_parent;
	bool snap_node_anchors;
	bool snap_node_sides;
	bool snap_node_center;
	bool snap_other_nodes;
	bool snap_grid;
	bool snap_guides;
	bool snap_rotation;
	bool snap_relative;
	bool snap_pixel;
	bool skeleton_show_bones;
	bool key_pos;
	bool key_rot;
	bool key_scale;

	MenuOption last_option;

	struct _SelectResult {

		CanvasItem *item;
		float z_index;
		bool has_z;
		_FORCE_INLINE_ bool operator<(const _SelectResult &p_rr) const {
			return has_z && p_rr.has_z ? p_rr.z_index < z_index : p_rr.has_z;
		}
	};
	Vector<_SelectResult> selection_results;

	struct _HoverResult {

		Point2 position;
		Ref<Texture> icon;
		String name;
	};
	Vector<_HoverResult> hovering_results;

	struct BoneList {

		Transform2D xform;
		float length;
		uint64_t last_pass;

		BoneList() :
				length(0.f),
				last_pass(0) {}
	};

	uint64_t bone_last_frame;

	struct BoneKey {
		ObjectID from;
		ObjectID to;
		_FORCE_INLINE_ bool operator<(const BoneKey &p_key) const {
			if (from == p_key.from)
				return to < p_key.to;
			else
				return from < p_key.from;
		}
	};

	Map<BoneKey, BoneList> bone_list;

	struct PoseClipboard {
		Vector2 pos;
		Vector2 scale;
		float rot;
		ObjectID id;
	};
	List<PoseClipboard> pose_clipboard;

	ToolButton *select_button;

	ToolButton *move_button;
	ToolButton *scale_button;
	ToolButton *rotate_button;

	ToolButton *list_select_button;
	ToolButton *pivot_button;
	ToolButton *pan_button;

	ToolButton *snap_button;
	MenuButton *snap_config_menu;
	PopupMenu *smartsnap_config_popup;

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

	DragType drag_type;
	Point2 drag_from;
	Point2 drag_to;
	Point2 drag_rotation_center;
	List<CanvasItem *> drag_selection;
	int dragged_guide_index;
	Point2 dragged_guide_pos;

	bool updating_value_dialog;

	Point2 box_selecting_to;

	Ref<StyleBoxTexture> select_sb;
	Ref<Texture> select_handle;
	Ref<Texture> anchor_handle;

	Ref<ShortCut> drag_pivot_shortcut;
	Ref<ShortCut> set_pivot_shortcut;
	Ref<ShortCut> multiply_grid_step_shortcut;
	Ref<ShortCut> divide_grid_step_shortcut;

	bool _is_node_editable(const Node *p_node);
	void _find_canvas_items_at_pos(const Point2 &p_pos, Node *p_node, Vector<_SelectResult> &r_items, int p_limit = 0, const Transform2D &p_parent_xform = Transform2D(), const Transform2D &p_canvas_xform = Transform2D());
	void _get_canvas_items_at_pos(const Point2 &p_pos, Vector<_SelectResult> &r_items, int p_limit = 0);
	void _get_bones_at_pos(const Point2 &p_pos, Vector<_SelectResult> &r_items);

	void _find_canvas_items_in_rect(const Rect2 &p_rect, Node *p_node, List<CanvasItem *> *r_items, const Transform2D &p_parent_xform = Transform2D(), const Transform2D &p_canvas_xform = Transform2D());
	bool _select_click_on_item(CanvasItem *item, Point2 p_click_pos, bool p_append);

	ConfirmationDialog *snap_dialog;

	CanvasItem *ref_item;

	void _add_canvas_item(CanvasItem *p_canvas_item);

	void _save_canvas_item_ik_chain(const CanvasItem *p_canvas_item, List<float> *p_bones_length, List<Dictionary> *p_bones_state);
	void _save_canvas_item_state(List<CanvasItem *> p_canvas_items, bool save_bones = false);
	void _restore_canvas_item_ik_chain(CanvasItem *p_canvas_item, const List<Dictionary> *p_bones_state);
	void _restore_canvas_item_state(List<CanvasItem *> p_canvas_items, bool restore_bones = false);
	void _commit_canvas_item_state(List<CanvasItem *> p_canvas_items, String action_name, bool commit_bones = false);

	Vector2 _anchor_to_position(const Control *p_control, Vector2 anchor);
	Vector2 _position_to_anchor(const Control *p_control, Vector2 position);

	void _popup_callback(int p_op);
	bool updating_scroll;
	void _update_scroll(float);
	void _update_scrollbars();
	void _append_canvas_item(CanvasItem *p_item);
	void _snap_changed();
	void _selection_result_pressed(int);
	void _selection_menu_hide();

	UndoRedo *undo_redo;
	bool _build_bones_list(Node *p_node);
	bool _get_bone_shape(Vector<Vector2> *shape, Vector<Vector2> *outline_shape, Map<BoneKey, BoneList>::Element *bone);

	List<CanvasItem *> _get_edited_canvas_items(bool retreive_locked = false, bool remove_canvas_item_if_parent_in_selection = true);
	Rect2 _get_encompassing_rect_from_list(List<CanvasItem *> p_list);
	void _expand_encompassing_rect_using_children(Rect2 &p_rect, const Node *p_node, bool &r_first, const Transform2D &p_parent_xform = Transform2D(), const Transform2D &p_canvas_xform = Transform2D(), bool include_locked_nodes = true);
	Rect2 _get_encompassing_rect(const Node *p_node);

	Object *_get_editor_data(Object *p_what);

	void _keying_changed();

	void _unhandled_key_input(const Ref<InputEvent> &p_ev);

	void _draw_text_at_position(Point2 p_position, String p_string, Margin p_side);
	void _draw_margin_at_position(int p_value, Point2 p_position, Margin p_side);
	void _draw_percentage_at_position(float p_value, Point2 p_position, Margin p_side);
	void _draw_straight_line(Point2 p_from, Point2 p_to, Color p_color);

	void _draw_rulers();
	void _draw_guides();
	void _draw_focus();
	void _draw_grid();
	void _draw_control_helpers(Control *control);
	void _draw_selection();
	void _draw_axis();
	void _draw_bones();
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
	bool _gui_input_zoom_or_pan(const Ref<InputEvent> &p_event);
	bool _gui_input_rulers_and_guides(const Ref<InputEvent> &p_event);
	bool _gui_input_hover(const Ref<InputEvent> &p_event);

	void _gui_input_viewport(const Ref<InputEvent> &p_event);

	void _focus_selection(int p_op);

	void _solve_IK(Node2D *leaf_node, Point2 target_position);

	void _snap_if_closer_float(float p_value, float p_target_snap, float &r_current_snap, bool &r_snapped, float p_radius = 10.0);
	void _snap_if_closer_point(Point2 p_value, Point2 p_target_snap, Point2 &r_current_snap, bool (&r_snapped)[2], real_t rotation = 0.0, float p_radius = 10.0);
	void _snap_other_nodes(Point2 p_value, Point2 &r_current_snap, bool (&r_snapped)[2], const Node *p_current, const CanvasItem *p_to_snap = NULL);

	void _set_anchors_preset(Control::LayoutPreset p_preset);
	void _set_margins_preset(Control::LayoutPreset p_preset);
	void _set_anchors_and_margins_preset(Control::LayoutPreset p_preset);

	HBoxContainer *zoom_hb;
	void _zoom_on_position(float p_zoom, Point2 p_position = Point2());
	void _button_zoom_minus();
	void _button_zoom_reset();
	void _button_zoom_plus();
	void _button_toggle_snap(bool p_status);
	void _button_tool_select(int p_index);

	HSplitContainer *palette_split;
	VSplitContainer *bottom_split;

	bool bone_list_dirty;
	void _queue_update_bone_list();
	void _update_bone_list();
	void _tree_changed(Node *);

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
		SNAP_NODE_CENTER = 1 << 6,
		SNAP_OTHER_NODES = 1 << 7,

		SNAP_DEFAULT = SNAP_GRID | SNAP_GUIDES | SNAP_PIXEL,
	};

	Point2 snap_point(Point2 p_target, unsigned int p_modes = SNAP_DEFAULT, const CanvasItem *p_canvas_item = NULL, unsigned int p_forced_modes = 0);
	float snap_angle(float p_target, float p_start = 0) const;

	Transform2D get_canvas_transform() const { return transform; }

	static CanvasItemEditor *get_singleton() { return singleton; }
	Dictionary get_state() const;
	void set_state(const Dictionary &p_state);

	void add_control_to_menu_panel(Control *p_control);
	void remove_control_from_menu_panel(Control *p_control);

	HSplitContainer *get_palette_split();
	VSplitContainer *get_bottom_split();

	Control *get_viewport_control() { return viewport; }

	void update_viewport();

	Tool get_current_tool() { return tool; }

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
	bool _only_packed_scenes_selected() const;
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
