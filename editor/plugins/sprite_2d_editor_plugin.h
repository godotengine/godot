/*************************************************************************/
/*  sprite_2d_editor_plugin.h                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef SPRITE_EDITOR_PLUGIN_H
#define SPRITE_EDITOR_PLUGIN_H

#include "editor/editor_node.h"
#include "editor/editor_plugin.h"
#include "scene/2d/sprite_2d.h"
#include "scene/gui/spin_box.h"

class Sprite2DEditor : public Control {
	GDCLASS(Sprite2DEditor, Control);

	enum Menu {
		MENU_OPTION_CONVERT_TO_MESH_2D,
		MENU_OPTION_CONVERT_TO_POLYGON_2D,
		MENU_OPTION_CREATE_COLLISION_POLY_2D,
		MENU_OPTION_CREATE_LIGHT_OCCLUDER_2D
	};

	Menu selected_menu_item;

	Sprite2D *node;

	MenuButton *options;

	ConfirmationDialog *outline_dialog;

	AcceptDialog *err_dialog;

	ConfirmationDialog *debug_uv_dialog;
	Control *debug_uv;
	Vector<Vector2> uv_lines;
	Vector<Vector<Vector2>> outline_lines;
	Vector<Vector<Vector2>> computed_outline_lines;
	Vector<Vector2> computed_vertices;
	Vector<Vector2> computed_uv;
	Vector<int> computed_indices;

	SpinBox *simplification;
	SpinBox *grow_pixels;
	SpinBox *shrink_pixels;
	Button *update_preview;

	void _menu_option(int p_option);

	//void _create_uv_lines();
	friend class Sprite2DEditorPlugin;

	void _debug_uv_draw();
	void _update_mesh_data();

	void _create_node();
	void _convert_to_mesh_2d_node();
	void _convert_to_polygon_2d_node();
	void _create_collision_polygon_2d_node();
	void _create_light_occluder_2d_node();

	void _add_as_sibling_or_child(Node *p_own_node, Node *p_new_node);

protected:
	void _node_removed(Node *p_node);
	static void _bind_methods();

public:
	void edit(Sprite2D *p_sprite);
	Sprite2DEditor();
};

class Sprite2DEditorPlugin : public EditorPlugin {
	GDCLASS(Sprite2DEditorPlugin, EditorPlugin);

	Sprite2DEditor *sprite_editor;
	EditorNode *editor;

public:
	virtual String get_name() const override { return "Sprite2D"; }
	bool has_main_screen() const override { return false; }
	virtual void edit(Object *p_object) override;
	virtual bool handles(Object *p_object) const override;
	virtual void make_visible(bool p_visible) override;

	Sprite2DEditorPlugin(EditorNode *p_node);
	~Sprite2DEditorPlugin();
};

#endif // SPRITE_EDITOR_PLUGIN_H
