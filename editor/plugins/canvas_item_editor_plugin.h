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
		SNAP_SHOW_GRID,
		SNAP_USE_ROTATION,
		SNAP_RELATIVE,
		SNAP_CONFIGURE,
		SNAP_USE_PIXEL,
		ZOOM_IN,
		ZOOM_OUT,
		ZOOM_RESET,
		ZOOM_SET,
		LOCK_SELECTED,
		UNLOCK_SELECTED,
		GROUP_SELECTED,
		UNGROUP_SELECTED,
		ANCHOR_ALIGN_TOP_LEFT,
		ANCHOR_ALIGN_TOP_RIGHT,
		ANCHOR_ALIGN_BOTTOM_LEFT,
		ANCHOR_ALIGN_BOTTOM_RIGHT,
		ANCHOR_ALIGN_CENTER_LEFT,
		ANCHOR_ALIGN_CENTER_RIGHT,
		ANCHOR_ALIGN_CENTER_TOP,
		ANCHOR_ALIGN_CENTER_BOTTOM,
		ANCHOR_ALIGN_CENTER,
		ANCHOR_ALIGN_TOP_WIDE,
		ANCHOR_ALIGN_LEFT_WIDE,
		ANCHOR_ALIGN_RIGHT_WIDE,
		ANCHOR_ALIGN_BOTTOM_WIDE,
		ANCHOR_ALIGN_VCENTER_WIDE,
		ANCHOR_ALIGN_HCENTER_WIDE,
		ANCHOR_ALIGN_WIDE,
		ANCHOR_ALIGN_WIDE_FIT,
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

	bool can_move_pivot;

	HScrollBar *h_scroll;
	VScrollBar *v_scroll;
	HBoxContainer *hb;

	Transform2D transform;
	float zoom;
	Vector2 snap_offset;
	Vector2 snap_step;
	float snap_rotation_step;
	float snap_rotation_offset;
	bool snap_grid;
	bool snap_show_grid;
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

	struct LockList {
		Point2 pos;
		bool lock;
		bool group;
		LockList() {
			lock = false;
			group = false;
		}
	};

	List<LockList> lock_list;

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

	ToolButton *pivot_button;
	ToolButton *pan_button;

	ToolButton *lock_button;
	ToolButton *unlock_button;

	ToolButton *group_button;
	ToolButton *ungroup_button;

	MenuButton *edit_menu;
	PopupMenu *skeleton_menu;
	MenuButton *view_menu;
	HBoxContainer *animation_hb;
	MenuButton *animation_menu;
	MenuButton *anchor_menu;

	Button *key_loc_button;
	Button *key_rot_button;
	Button *key_scale_button;
	Button *key_insert_button;

	PopupMenu *selection_menu;

	//PopupMenu *popup;
	DragType drag;
	Point2 drag_from;
	Point2 drag_point_from;
	bool updating_value_dialog;
	Point2 display_rotate_from;
	Point2 display_rotate_to;

	Ref<StyleBoxTexture> select_sb;
	Ref<Texture> select_handle;
	Ref<Texture> anchor_handle;

	int handle_len;
	bool _is_part_of_subscene(CanvasItem *p_item);
	void _find_canvas_items_at_pos(const Point2 &p_pos, Node *p_node, const Transform2D &p_parent_xform, const Transform2D &p_canvas_xform, Vector<_SelectResult> &r_items, int limit = 0);
	void _find_canvas_items_at_rect(const Rect2 &p_rect, Node *p_node, const Transform2D &p_parent_xform, const Transform2D &p_canvas_xform, List<CanvasItem *> *r_items);

	void _select_click_on_empty_area(Point2 p_click_pos, bool p_append, bool p_box_selection);
	bool _select_click_on_item(CanvasItem *item, Point2 p_click_pos, bool p_append, bool p_drag);

	ConfirmationDialog *snap_dialog;

	AcceptDialog *value_dialog;
	Label *dialog_label;
	SpinBox *dialog_val;

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

	float _anchor_snap(float anchor, bool *snapped = NULL, float p_opposite_anchor = -1);
	Vector2 _anchor_to_position(Control *p_control, Vector2 anchor);
	Vector2 _position_to_anchor(Control *p_control, Vector2 position);

	void _popup_callback(int p_op);
	bool updating_scroll;
	void _update_scroll(float);
	void _update_scrollbars();
	void _update_cursor();
	void incbeg(float &beg, float &end, float inc, float minsize, bool p_symmetric);
	void incend(float &beg, float &end, float inc, float minsize, bool p_symmetric);

	void _append_canvas_item(CanvasItem *p_item);
	void _dialog_value_changed(double);
	void _snap_changed();
	void _selection_result_pressed(int);
	void _selection_menu_hide();

	UndoRedo *undo_redo;

	Point2 _find_topleftmost_point();

	void _find_canvas_items_span(Node *p_node, Rect2 &r_rect, const Transform2D &p_xform);

	Object *_get_editor_data(Object *p_what);

	CanvasItem *get_single_item();
	int get_item_count();
	void _keying_changed();

	void _unhandled_key_input(const Ref<InputEvent> &p_ev);

	void _draw_percentage_at_position(float p_value, Point2 p_position, Margin p_side);

	void _viewport_gui_input(const Ref<InputEvent> &p_event);
	void _viewport_draw();

	void _focus_selection(int p_op);

	void _set_anchors_preset(Control::LayoutPreset p_preset);
	void _set_full_rect();

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
	Vector2 snap_point(Vector2 p_target, Vector2 p_start = Vector2(0, 0)) const;
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
	void _on_change_type();

	void _create_preview(const Vector<String> &files) const;
	void _remove_preview();

	bool _cyclical_dependency_exists(const String &p_target_scene_path, Node *p_desired_node);
	void _create_nodes(Node *parent, Node *child, String &path, const Point2 &p_point);
	bool _create_instance(Node *parent, String &path, const Point2 &p_point);
	void _perform_drop_data();

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
