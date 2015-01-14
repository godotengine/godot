/*************************************************************************/
/*  canvas_item_editor_plugin.h                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                 */
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

#include "tools/editor/editor_plugin.h"
#include "tools/editor/editor_node.h"
#include "scene/gui/spin_box.h"
#include "scene/gui/panel_container.h"
#include "scene/gui/box_container.h"
#include "scene/2d/canvas_item.h"

/**
	@author Juan Linietsky <reduzio@gmail.com>
*/




class CanvasItemEditorSelectedItem : public Object {

	OBJ_TYPE(CanvasItemEditorSelectedItem,Object);
public:
	Variant undo_state;
	Vector2 undo_pivot;

	Matrix32 prev_xform;
	float prev_rot;
	Rect2 prev_rect;

	CanvasItemEditorSelectedItem() { prev_rot=0; }
};

class CanvasItemEditor : public VBoxContainer {

	OBJ_TYPE(CanvasItemEditor, VBoxContainer );

	EditorNode *editor;


	enum Tool {

		TOOL_SELECT,
		TOOL_MOVE,
		TOOL_ROTATE,
		TOOL_PAN,
		TOOL_MAX
	};

	enum MenuOption {
		SNAP_USE,
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
		ALIGN_HORIZONTAL,
		ALIGN_VERTICAL,
		SPACE_HORIZONTAL,
		SPACE_VERTICAL,
		EXPAND_TO_PARENT,
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
		DRAG_ALL,
		DRAG_ROTATE,
		DRAG_PIVOT,

	};

	enum KeyMoveMODE {
		MOVE_VIEW_BASE,
		MOVE_LOCAL_BASE,
		MOVE_LOCAL_WITH_ROT
	};

	EditorSelection *editor_selection;

	Tool tool;
	bool first_update;
	Control *viewport;

	bool can_move_pivot;

	HScrollBar *h_scroll;
	VScrollBar *v_scroll;
	HBoxContainer *hb;

	Matrix32 transform;
	float zoom;
	int snap;
	bool pixel_snap;
	bool box_selecting;
	Point2 box_selecting_to;
	bool key_pos;
	bool key_rot;
	bool key_scale;

	void _tool_select(int p_index);


	MenuOption last_option;

	struct LockList {
		Point2 pos;
		bool lock;
		bool group;
		LockList() { lock=false; group=false; }
	};

	List<LockList> lock_list;

	struct BoneList {

		Matrix32 xform;
		Vector2 from;
		Vector2 to;
		ObjectID bone;
	};

	List<BoneList> bone_list;
	Matrix32 bone_orig_xform;

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
	ToolButton *move_button;
	ToolButton *rotate_button;

	ToolButton *pan_button;

	ToolButton *lock_button;
	ToolButton *unlock_button;

	ToolButton *group_button;
	ToolButton *ungroup_button;

	MenuButton *edit_menu;
	MenuButton *view_menu;
	HBoxContainer *animation_hb;
	MenuButton *animation_menu;

	Button *key_loc_button;
	Button *key_rot_button;
	Button *key_scale_button;
	Button *key_insert_button;

	//PopupMenu *popup;
	DragType drag;
	Point2 drag_from;
	Point2 drag_point_from;
	bool updating_value_dialog;
	Point2 display_rotate_from;
	Point2 display_rotate_to;
#if 0
	struct EditInfo {

		Variant undo_state;

		Matrix32 prev_xform;
		float prev_rot;
		Rect2 prev_rect;
		EditInfo() { prev_rot=0; }
	};

	typedef Map<CanvasItem*,EditInfo> CanvasItemMap;
	CanvasItemMap canvas_items;
#endif
	Ref<StyleBoxTexture> select_sb;
	Ref<Texture> select_handle;


	int handle_len;
	CanvasItem* _select_canvas_item_at_pos(const Point2 &p_pos,Node* p_node,const Matrix32& p_parent_xform,const Matrix32& p_canvas_xform);
	void _find_canvas_items_at_rect(const Rect2& p_rect,Node* p_node,const Matrix32& p_parent_xform,const Matrix32& p_canvas_xform,List<CanvasItem*> *r_items);

	AcceptDialog *value_dialog;
	Label *dialog_label;
	SpinBox *dialog_val;
	
	CanvasItem *ref_item;

	void _add_canvas_item(CanvasItem *p_canvas_item);
	void _remove_canvas_item(CanvasItem *p_canvas_item);
	void _clear_canvas_items();
	void _visibility_changed(ObjectID p_canvas_item);
	void _key_move(const Vector2& p_dir, bool p_snap, KeyMoveMODE p_move_mode);

	DragType _find_drag_type(const Matrix32& p_xform, const Rect2& p_local_rect, const Point2& p_click, Vector2& r_point);

	Point2 snapify(const Point2& p_pos) const;
	void _popup_callback(int p_op);
	bool updating_scroll;
	void _update_scroll(float);
	void _update_scrollbars();
	void incbeg(float& beg,float& end, float inc, float minsize,bool p_symmetric);
	void incend(float& beg,float& end, float inc, float minsize,bool p_symmetric);

	void _append_canvas_item(CanvasItem *p_item);
	void _dialog_value_changed(double);
	UndoRedo *undo_redo;

	Point2 _find_topleftmost_point();


	void _find_canvas_items_span(Node *p_node, Rect2& r_rect, const Matrix32& p_xform);


	Object *_get_editor_data(Object *p_what);

	CanvasItem *get_single_item();
	int get_item_count();
	void _keying_changed(bool p_changed);

	void _unhandled_key_input(const InputEvent& p_ev);

	void _viewport_input_event(const InputEvent& p_event);
	void _viewport_draw();

	HSplitContainer *palette_split;
	VSplitContainer *bottom_split;

friend class CanvasItemEditorPlugin;
protected:


	void _notification(int p_what);

	void _node_removed(Node *p_node);
	static void _bind_methods();
	void end_drag();
	void box_selection_start( Point2 &click );
	bool box_selection_end();

	HBoxContainer *get_panel_hb() { return hb; }
	
	struct compare_items_x {
		bool operator()( const CanvasItem *a, const CanvasItem *b ) const {
			return a->get_global_transform().elements[2].x < b->get_global_transform().elements[2].x;
		}
	};
	
	struct compare_items_y {
		bool operator()( const CanvasItem *a, const CanvasItem *b ) const {
			return a->get_global_transform().elements[2].y < b->get_global_transform().elements[2].y;
		}
	};
	
	struct proj_vector2_x {
		float get( const Vector2 &v ) { return v.x; }
		void set( Vector2 &v, float f ) { v.x = f; }
	};
	
	struct proj_vector2_y {
		float get( const Vector2 &v ) { return v.y; }
		void set( Vector2 &v, float f ) { v.y = f; }
	};
	
	template< class P, class C > void space_selected_items();

	static CanvasItemEditor *singleton;
public:

	bool is_snap_active() const;
	int get_snap() const { return snap; }

	Matrix32 get_canvas_transform() const { return transform; }

	static CanvasItemEditor *get_singleton() { return singleton; }
	Dictionary get_state() const;
	void set_state(const Dictionary& p_state);

	void add_control_to_menu_panel(Control *p_control);

	HSplitContainer *get_palette_split();
	VSplitContainer *get_bottom_split();

	Control *get_viewport_control() { return viewport; }


	bool get_remove_list(List<Node*> *p_list);
	void set_undo_redo(UndoRedo *p_undo_redo) {undo_redo=p_undo_redo; }
	void edit(CanvasItem *p_canvas_item);
	CanvasItemEditor(EditorNode *p_editor);
};

class CanvasItemEditorPlugin : public EditorPlugin {

	OBJ_TYPE( CanvasItemEditorPlugin, EditorPlugin );

	CanvasItemEditor *canvas_item_editor;
	EditorNode *editor;

public:

	virtual String get_name() const { return "2D"; }
	bool has_main_screen() const { return true; }
	virtual void edit(Object *p_object);
	virtual bool handles(Object *p_object) const;
	virtual void make_visible(bool p_visible);
	virtual bool get_remove_list(List<Node*> *p_list) { return canvas_item_editor->get_remove_list(p_list); }
	virtual Dictionary get_state() const;
	virtual void set_state(const Dictionary& p_state);

	CanvasItemEditor *get_canvas_item_editor() { return canvas_item_editor; }

	CanvasItemEditorPlugin(EditorNode *p_node);
	~CanvasItemEditorPlugin();

};

#endif
