/*************************************************************************/
/*  navigation_polygon_editor_plugin.h                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
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
#ifndef NAVIGATIONPOLYGONEDITORPLUGIN_H
#define NAVIGATIONPOLYGONEDITORPLUGIN_H

#include "editor/editor_node.h"
#include "editor/editor_plugin.h"
#include "scene/2d/navigation_polygon.h"
#include "scene/gui/button_group.h"
#include "scene/gui/tool_button.h"

/**
	@author Juan Linietsky <reduzio@gmail.com>
*/
class CanvasItemEditor;

class NavigationPolygonEditor : public HBoxContainer {

	GDCLASS(NavigationPolygonEditor, HBoxContainer);

	UndoRedo *undo_redo;
	enum Mode {

		MODE_CREATE,
		MODE_EDIT,

	};

	Mode mode;

	ToolButton *button_create;
	ToolButton *button_edit;

	ConfirmationDialog *create_nav;

	CanvasItemEditor *canvas_item_editor;
	EditorNode *editor;
	Panel *panel;
	NavigationPolygonInstance *node;
	MenuButton *options;

	int edited_outline;
	int edited_point;
	Vector2 edited_point_pos;
	PoolVector<Vector2> pre_move_edit;
	Vector<Vector2> wip;
	bool wip_active;

	void _wip_close();
	void _canvas_draw();
	void _create_nav();

	void _menu_option(int p_option);

protected:
	void _notification(int p_what);
	void _node_removed(Node *p_node);
	static void _bind_methods();

public:
	bool forward_gui_input(const InputEvent &p_event);
	void edit(Node *p_collision_polygon);
	NavigationPolygonEditor(EditorNode *p_editor);
};

class NavigationPolygonEditorPlugin : public EditorPlugin {

	GDCLASS(NavigationPolygonEditorPlugin, EditorPlugin);

	NavigationPolygonEditor *collision_polygon_editor;
	EditorNode *editor;

public:
	virtual bool forward_canvas_gui_input(const Transform2D &p_canvas_xform, const InputEvent &p_event) { return collision_polygon_editor->forward_gui_input(p_event); }

	virtual String get_name() const { return "NavigationPolygonInstance"; }
	bool has_main_screen() const { return false; }
	virtual void edit(Object *p_node);
	virtual bool handles(Object *p_node) const;
	virtual void make_visible(bool p_visible);

	NavigationPolygonEditorPlugin(EditorNode *p_node);
	~NavigationPolygonEditorPlugin();
};

#endif // NAVIGATIONPOLYGONEDITORPLUGIN_H
