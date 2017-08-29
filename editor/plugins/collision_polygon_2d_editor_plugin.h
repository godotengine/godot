/*************************************************************************/
/*  collision_polygon_2d_editor_plugin.h                                 */
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
#ifndef COLLISION_POLYGON_2D_EDITOR_PLUGIN_H
#define COLLISION_POLYGON_2D_EDITOR_PLUGIN_H

#include "editor/plugins/abstract_polygon_2d_editor.h"
#include "scene/2d/collision_polygon_2d.h"

/**
	@author Juan Linietsky <reduzio@gmail.com>
*/
class CollisionPolygon2DEditor : public AbstractPolygon2DEditor {

	GDCLASS(CollisionPolygon2DEditor, AbstractPolygon2DEditor);

	enum Mode {

		MODE_CREATE,
		MODE_EDIT,

	};

	Mode mode;

	ToolButton *button_create;
	ToolButton *button_edit;

	CollisionPolygon2D *node;

	void _menu_option(int p_option);

protected:
	virtual void _enter_edit_mode();
	virtual bool _is_in_create_mode() const;
	virtual bool _is_in_edit_mode() const;

	virtual Node2D *_get_node() const;
	virtual void _set_node(Node *p_node);

	virtual int _get_polygon_count() const;
	virtual Vector<Vector2> _get_polygon(int i) const;
	virtual void _set_polygon(int p_polygon, const Vector<Vector2> &p_points) const;
	virtual Vector2 _get_offset() const;

	void _notification(int p_what);
	static void _bind_methods();

public:
	CollisionPolygon2DEditor(EditorNode *p_editor);
};

class CollisionPolygon2DEditorPlugin : public EditorPlugin {

	GDCLASS(CollisionPolygon2DEditorPlugin, EditorPlugin);

	CollisionPolygon2DEditor *collision_polygon_editor;
	EditorNode *editor;

public:
	virtual bool forward_canvas_gui_input(const Transform2D &p_canvas_xform, const Ref<InputEvent> &p_event) { return collision_polygon_editor->forward_gui_input(p_event); }

	virtual String get_name() const { return "CollisionPolygon2D"; }
	bool has_main_screen() const { return false; }
	virtual void edit(Object *p_object);
	virtual bool handles(Object *p_object) const;
	virtual void make_visible(bool p_visible);

	CollisionPolygon2DEditorPlugin(EditorNode *p_node);
	~CollisionPolygon2DEditorPlugin();
};

#endif // COLLISION_POLYGON_2D_EDITOR_PLUGIN_H
