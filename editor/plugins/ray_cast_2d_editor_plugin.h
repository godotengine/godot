/*************************************************************************/
/*  ray_cast_2d_editor_plugin.h                                          */
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

#ifndef RAY_CAST_2D_EDITOR_PLUGIN_H
#define RAY_CAST_2D_EDITOR_PLUGIN_H

#include "editor/editor_plugin.h"
#include "scene/2d/ray_cast_2d.h"

class CanvasItemEditor;

class RayCast2DEditor : public Control {
	GDCLASS(RayCast2DEditor, Control);

	UndoRedo *undo_redo = nullptr;
	CanvasItemEditor *canvas_item_editor = nullptr;
	RayCast2D *node;

	bool pressed = false;
	Point2 original_cast_to;

protected:
	void _notification(int p_what);
	void _node_removed(Node *p_node);
	static void _bind_methods();

public:
	bool forward_canvas_gui_input(const Ref<InputEvent> &p_event);
	void forward_canvas_draw_over_viewport(Control *p_overlay);
	void edit(Node *p_node);

	RayCast2DEditor();
};

class RayCast2DEditorPlugin : public EditorPlugin {
	GDCLASS(RayCast2DEditorPlugin, EditorPlugin);

	RayCast2DEditor *ray_cast_2d_editor = nullptr;

public:
	virtual bool forward_canvas_gui_input(const Ref<InputEvent> &p_event) { return ray_cast_2d_editor->forward_canvas_gui_input(p_event); }
	virtual void forward_canvas_draw_over_viewport(Control *p_overlay) { ray_cast_2d_editor->forward_canvas_draw_over_viewport(p_overlay); }

	virtual String get_name() const { return "RayCast2D"; }
	bool has_main_screen() const { return false; }
	virtual void edit(Object *p_object);
	virtual bool handles(Object *p_object) const;
	virtual void make_visible(bool visible);

	RayCast2DEditorPlugin(EditorNode *p_editor);
};

#endif // RAY_CAST_2D_EDITOR_PLUGIN_H
