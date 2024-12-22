/**************************************************************************/
/*  visual_shape_2d_editor_plugin.h                                       */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#ifndef VISUAL_SHAPE_2D_EDITOR_PLUGIN_H
#define VISUAL_SHAPE_2D_EDITOR_PLUGIN_H

#include "editor/plugins/editor_plugin.h"

class MenuButton;
class VisualShape2D;

class VisualShape2DEditor : public Control {
	GDCLASS(VisualShape2DEditor, Control);

	friend class VisualShape2DEditorPlugin;

	enum Menu {
		MENU_OPTION_CONVERT_TO_MESH_2D,
		MENU_OPTION_CONVERT_TO_POLYGON_2D,
		MENU_OPTION_CREATE_COLLISION_SHAPE_2D,
		MENU_OPTION_CREATE_LIGHT_OCCLUDER_2D,
	};

	VisualShape2D *visual_shape_2d = nullptr;

	MenuButton *options = nullptr;

	void _menu_option(int p_option);

	void _convert_to_mesh_2d_node();
	void _convert_to_polygon_2d_node();
	void _create_collision_shape_2d_node();
	void _create_light_occluder_2d_node();

	void _add_as_sibling_or_child(Node *p_own_node, Node *p_new_node);

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void edit(VisualShape2D *p_visual_shape_2d);
	VisualShape2DEditor();
};

class VisualShape2DEditorPlugin : public EditorPlugin {
	GDCLASS(VisualShape2DEditorPlugin, EditorPlugin);

	VisualShape2DEditor *visual_shape_editor = nullptr;

public:
	virtual String get_name() const override { return "VisualShape2D"; }
	bool has_main_screen() const override { return false; }
	virtual void edit(Object *p_object) override;
	virtual bool handles(Object *p_object) const override;
	virtual void make_visible(bool p_visible) override;

	VisualShape2DEditorPlugin();
};

#endif // VISUAL_SHAPE_2D_EDITOR_PLUGIN_H
