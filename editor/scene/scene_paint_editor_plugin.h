/**************************************************************************/
/*  scene_paint_editor_plugin.h                                           */
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

#pragma once

#include "editor/plugins/editor_plugin.h"
#include "scene/gui/box_container.h"
#include "scene/gui/item_list.h"

class ScenePalette;
class BaseButton;
class SpinBox;
class Control;
class Node2D;

class ScenePaintEditor : public VBoxContainer {
	GDCLASS(ScenePaintEditor, VBoxContainer);

	friend class ScenePaintEditorPlugin;

	bool snap_grid = false;
	bool is_tool_selected = false;

	Ref<ScenePalette> palette;

	Control *viewport = nullptr;
	Node2D *node = nullptr;

	HBoxContainer *toolbar = nullptr;

	Button *new_palette_button = nullptr;
	Button *load_palette_button = nullptr;
	Button *save_palette_button = nullptr;

	Button *add_scene_button = nullptr;
	Button *open_scene_button = nullptr;
	Button *remove_scene_button = nullptr;

	Button *snap_grid_button = nullptr;
	SpinBox *grid_step_x = nullptr;
	SpinBox *grid_step_y = nullptr;

	ItemList *item_list = nullptr;

	void _draw();
	void _draw_grid();
	void _draw_grid_highlight();
	void _update_draw();

	Vector2 _get_mouse_grid_cell();

	void _snap_grid_toggled(bool p_toggled);

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void set_grid_step(const Size2i size);
	Size2i get_grid_step() const;

	ScenePaintEditor();
	~ScenePaintEditor();
};

class ScenePaintEditorPlugin : public EditorPlugin {
	GDCLASS(ScenePaintEditorPlugin, EditorPlugin);

	ScenePaintEditor *scene_paint_editor = nullptr;
	Button *panel_button = nullptr;

	mutable bool is_node_2d = false;

	void _canvas_item_tool_changed(int p_tool);

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	virtual String get_plugin_name() const override { return "ScenePaint"; }
	bool has_main_screen() const override { return false; }
	virtual void edit(Object *p_object) override;
	virtual bool handles(Object *p_object) const override;
	virtual void make_visible(bool p_visible) override;
};

class ScenePalette : public Resource {
	GDCLASS(ScenePalette, Resource);

	Size2 _grid_step = Size2(8, 8);

protected:
	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;
	void _validate_property(PropertyInfo &p_property) const;

	static void _bind_methods();

public:
	ScenePalette();
	~ScenePalette();
};
