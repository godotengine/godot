/*************************************************************************/
/*  light_occluder_2d_editor_plugin.h                                    */
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
#ifndef LIGHT_OCCLUDER_2D_EDITOR_PLUGIN_H
#define LIGHT_OCCLUDER_2D_EDITOR_PLUGIN_H



#include "tools/editor/editor_plugin.h"
#include "tools/editor/editor_node.h"
#include "scene/2d/light_occluder_2d.h"
#include "scene/gui/tool_button.h"
#include "scene/gui/button_group.h"

/**
	@author Juan Linietsky <reduzio@gmail.com>
*/
class CanvasItemEditor;

class LightOccluder2DEditor : public HBoxContainer {

	OBJ_TYPE(LightOccluder2DEditor, HBoxContainer );

	UndoRedo *undo_redo;
	enum Mode {

		MODE_CREATE,
		MODE_EDIT,

	};

	Mode mode;

	ToolButton *button_create;
	ToolButton *button_edit;

	CanvasItemEditor *canvas_item_editor;
	EditorNode *editor;
	Panel *panel;
	LightOccluder2D *node;
	MenuButton *options;

	int edited_point;
	Vector2 edited_point_pos;
	Vector<Vector2> pre_move_edit;
	Vector<Vector2> wip;
	bool wip_active;

	ConfirmationDialog *create_poly;

	void _wip_close(bool p_closed);
	void _canvas_draw();
	void _menu_option(int p_option);
	void _create_poly();

protected:
	void _notification(int p_what);
	void _node_removed(Node *p_node);
	static void _bind_methods();
public:

	Vector2 snap_point(const Vector2& p_point) const;
	bool forward_input_event(const InputEvent& p_event);
	void edit(Node *p_collision_polygon);
	LightOccluder2DEditor(EditorNode *p_editor);
};

class LightOccluder2DEditorPlugin : public EditorPlugin {

	OBJ_TYPE( LightOccluder2DEditorPlugin, EditorPlugin );

	LightOccluder2DEditor *collision_polygon_editor;
	EditorNode *editor;

public:

	virtual bool forward_canvas_input_event(const Matrix32& p_canvas_xform,const InputEvent& p_event) { return collision_polygon_editor->forward_input_event(p_event); }

	virtual String get_name() const { return "LightOccluder2D"; }
	bool has_main_screen() const { return false; }
	virtual void edit(Object *p_node);
	virtual bool handles(Object *p_node) const;
	virtual void make_visible(bool p_visible);

	LightOccluder2DEditorPlugin(EditorNode *p_node);
	~LightOccluder2DEditorPlugin();

};

#endif // LIGHT_OCCLUDER_2D_EDITOR_PLUGIN_H
