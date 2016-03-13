/*************************************************************************/
/*  control_editor_plugin.h                                              */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2016 Juan Linietsky, Ariel Manzur.                 */
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
/**
	@author Juan Linietsky <reduzio@gmail.com>
*/

#if 0
class ControlEditor : public Control {

	OBJ_TYPE(ControlEditor, Control );

	EditorNode *editor;

	enum {
		SNAP_USE,
		SNAP_CONFIGURE
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
		DRAG_ALL
	};

	HScrollBar *h_scroll;
	VScrollBar *v_scroll;

	Matrix32 transform;
	float zoom;

	Control *current_window;
	PopupMenu *popup;
	DragType drag;
	Point2 drag_from;

	struct EditInfo {

		Point2 drag_pos;
		Point2 drag_size;
		Point2 drag_limit;
		Rect2 last_rect;
	};

	typedef Map<Control*,EditInfo> ControlMap;
	ControlMap controls;
	int handle_len;
	Control* _select_control_at_pos(const Point2& p_pos,Node* p_node);

	ConfirmationDialog *snap_dialog;
	LineEdit *snap_val;

	void _add_control(Control *p_control,const EditInfo& p_info);
	void _remove_control(Control *p_control);
	void _clear_controls();
	void _visibility_changed(ObjectID p_control);
	void _key_move(const Vector2& p_dir, bool p_snap);


	Point2i snapify(const Point2i& p_pos) const;
	void _popup_callback(int p_op);
	bool updating_scroll;
	void _update_scroll(float);
	void _update_scrollbars();
	UndoRedo *undo_redo;

	void _find_controls_span(Node *p_node, Rect2& r_rect);

protected:
	void _notification(int p_what);
	void _input_event(InputEvent p_event);
	void _node_removed(Node *p_node);
	static void _bind_methods();
public:

	bool get_remove_list(List<Node*> *p_list);
	void set_undo_redo(UndoRedo *p_undo_redo) {undo_redo=p_undo_redo; }
	void edit(Control *p_control);
	ControlEditor(EditorNode *p_editor);
};

class ControlEditorPlugin : public EditorPlugin {

	OBJ_TYPE( ControlEditorPlugin, EditorPlugin );

	ControlEditor *control_editor;
	EditorNode *editor;

public:

	virtual String get_name() const { return "GUI"; }
	bool has_main_screen() const { return true; }
	virtual void edit(Object *p_object);
	virtual bool handles(Object *p_object) const;
	virtual void make_visible(bool p_visible);
	virtual bool get_remove_list(List<Node*> *p_list) { return control_editor->get_remove_list(p_list); }


	ControlEditorPlugin(EditorNode *p_node);
	~ControlEditorPlugin();

};
#endif
#endif
