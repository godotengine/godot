/*************************************************************************/
/*  collision_shape_2d_editor_plugin.h                                   */
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
#ifndef COLLISION_SHAPE_2D_EDITOR_PLUGIN_H
#define COLLISION_SHAPE_2D_EDITOR_PLUGIN_H

#include "editor/editor_node.h"
#include "editor/editor_plugin.h"

#include "scene/2d/collision_shape_2d.h"

class CanvasItemEditor;

class CollisionShape2DEditor : public Control {
	GDCLASS(CollisionShape2DEditor, Control);

	enum ShapeType {
		CAPSULE_SHAPE,
		CIRCLE_SHAPE,
		CONCAVE_POLYGON_SHAPE,
		CONVEX_POLYGON_SHAPE,
		LINE_SHAPE,
		RAY_SHAPE,
		RECTANGLE_SHAPE,
		SEGMENT_SHAPE
	};

	EditorNode *editor;
	UndoRedo *undo_redo;
	CanvasItemEditor *canvas_item_editor;
	CollisionShape2D *node;

	Vector<Point2> handles;

	int shape_type;
	int edit_handle;
	bool pressed;
	Variant original;

	Variant get_handle_value(int idx) const;
	void set_handle(int idx, Point2 &p_point);
	void commit_handle(int idx, Variant &p_org);

	void _get_current_shape_type();

protected:
	static void _bind_methods();

public:
	bool forward_canvas_gui_input(const Ref<InputEvent> &p_event);
	void forward_draw_over_viewport(Control *p_overlay);
	void edit(Node *p_node);

	CollisionShape2DEditor(EditorNode *p_editor);
};

class CollisionShape2DEditorPlugin : public EditorPlugin {
	GDCLASS(CollisionShape2DEditorPlugin, EditorPlugin);

	CollisionShape2DEditor *collision_shape_2d_editor;
	EditorNode *editor;

public:
	virtual bool forward_canvas_gui_input(const Ref<InputEvent> &p_event) { return collision_shape_2d_editor->forward_canvas_gui_input(p_event); }
	virtual void forward_draw_over_viewport(Control *p_overlay) { return collision_shape_2d_editor->forward_draw_over_viewport(p_overlay); }

	virtual String get_name() const { return "CollisionShape2D"; }
	bool has_main_screen() const { return false; }
	virtual void edit(Object *p_obj);
	virtual bool handles(Object *p_obj) const;
	virtual void make_visible(bool visible);

	CollisionShape2DEditorPlugin(EditorNode *p_editor);
	~CollisionShape2DEditorPlugin();
};

#endif //COLLISION_SHAPE_2D_EDITOR_PLUGIN_H
