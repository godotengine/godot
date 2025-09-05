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

#include "core/templates/hash_map.h"
#include "editor/gui/editor_file_dialog.h"
#include "editor/plugins/editor_plugin.h"
#include "scene/2d/node_2d.h"
#include "scene/gui/box_container.h"
#include "scene/gui/control.h"
#include "scene/gui/item_list.h"
#include "scene/gui/spin_box.h"
#include "scene/gui/tree.h"

class ScenePalette;
class ScenePaletteFolderData;
class ScenePaletteSceneData;
class BaseButton;

class ScenePaintEditor : public VBoxContainer {
	GDCLASS(ScenePaintEditor, VBoxContainer);

	friend class ScenePaintEditorPlugin;

	enum {
		PICK_FILE_SYSTEM,
		PICK_SCENE_TREE,
		PICK_CANVAS_ITEM,
	};

	bool scene_picker = false;
	bool edit_properties = false;
	bool grid = false;
	bool snap_grid = false;
	bool use_local_grid = false;
	bool is_tool_selected = false;
	bool is_painting = false;
	bool is_erasing = false;

	Vector2 last_paint_pos;
	Rect2 paint_rect;

	Ref<ScenePalette> palette;

	EditorFileDialog *file = nullptr;
	Control *viewport = nullptr;
	Node2D *node = nullptr;
	Node2D *preview = nullptr;
	Node2D *instance = nullptr;

	HBoxContainer *toolbar = nullptr;

	Button *new_palette_button = nullptr;
	Button *load_palette_button = nullptr;
	Button *save_palette_button = nullptr;

	Button *add_folder_button = nullptr;
	Button *remove_folder_button = nullptr;

	Button *add_scene_button = nullptr;
	Button *open_scene_button = nullptr;
	Button *remove_scene_button = nullptr;
	Button *scene_picker_button = nullptr;
	Button *edit_properties_button = nullptr;

	Button *grid_toggle_button = nullptr;
	Button *snap_grid_button = nullptr;
	SpinBox *grid_step_x = nullptr;
	SpinBox *grid_step_y = nullptr;

	StringName edited_folder;
	int selected_scene_index = -1;
	Ref<PackedScene> selected_scene;
	String scene_path;

	Tree *folders = nullptr;
	ItemList *item_list = nullptr;

	void _edit(Object *p_object);

	void _draw();
	void _draw_grid();
	void _draw_grid_highlight();
	void _update_draw();
	void _update_preview();
	void _update_paint_rect(const Vector2 &mouse_pos, const Transform2D &canvas_xform);
	void _update_preview_position(const Vector2 &mouse_pos, const Transform2D &canvas_xform);

	void _gui_input_viewport(const Ref<InputEvent> &p_event);
	void _add_node_at_position(const Vector2 &mouse_pos, const Transform2D &canvas_xform);
	void _remove_node_at_position();
	String _get_scene_path_at_position(const Vector2 &mouse_pos, const Transform2D &canvas_xform);

	Vector2 _get_mouse_grid_cell();

	void _new_scene_palette();
	void _load_scene_palette();
	void _save_scene_palette();

	void _load_scene_palette_request(const String &p_path);
	void _save_scene_palette_request(const String &p_path);

	void _add_folder();
	void _edit_folder();
	void _remove_folder();

	void _add_scene();
	void _open_scene();
	void _remove_scene();
	void _edit_properties();

	void _scene_picker_toggled(bool p_pressed);
	void _file_system_input(const Ref<InputEvent> &p_event);
	void _scene_tree_input(const Ref<InputEvent> &p_event);
	void _update_scene_picker(int p_mode);
	void _edit_properties_toggled(bool p_pressed);
	void _grid_toggle_toggled(bool p_toggled);
	void _snap_grid_toggled(bool p_toggled);
	void _grid_step_changed();
	void _update_grid_step();

	void _folder_selected();
	void _folder_deselected();

	void _update_folder_list();

	void _scene_deselected();
	void _scene_selected(const int p_index);

	bool _add_scene_request(const String &p_path);
	void _update_scene_list();
	void _scene_thumbnail_done(const String &p_path, const Ref<Texture2D> &p_preview, const Ref<Texture2D> &p_small_preview, const Variant &p_ud);

	void _update_toolbar_buttons();

	bool _can_drop_data_fw(const Point2 &p_point, const Variant &p_data) const;
	void _drop_data_fw(const Point2 &p_point, const Variant &p_data);

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void set_grid_step(const Size2i p_size);
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

	Dictionary _data;

protected:
	static void _bind_methods();

public:
	void set_grid_step(const Size2i &p_size, const String p_folder = "", const String p_scene = "");
	Size2i get_grid_step(const String p_folder = "", String p_scene = "") const;

	void add_folder(const StringName &p_name);
	void edit_folder(const StringName &p_name, const String &p_new_name);
	void remove_folder(const StringName &p_name);
	bool has_folder(const StringName &p_name) const { return _data.has(p_name); }
	void set_folders(const Array &p_folders);
	Array get_folders() const;
	Ref<ScenePaletteFolderData> get_folder(const StringName &p_name);

	//ScenePalette();
	//~ScenePalette();
};

class ScenePaletteFolderData : public Resource {
	GDCLASS(ScenePaletteFolderData, Resource);

	String folder_name;
	Size2i grid_step;
	TypedArray<Ref<ScenePaletteSceneData>> scenes;

protected:
	static void _bind_methods();

public:
	String get_folder_name() const { return folder_name; }
	void set_folder_name(const String &p_name) { folder_name = p_name; }
	Size2i get_grid_step() const { return grid_step; }
	void set_grid_step(const Size2i &p_size) { grid_step = p_size; }
	void set_scenes(const TypedArray<Ref<ScenePaletteSceneData>> &p_scenes) { scenes = p_scenes; }
	TypedArray<Ref<ScenePaletteSceneData>> get_scenes() const { return scenes; }
	void add_scene(const String &p_scene_path, const String &p_display_name);
	bool has_scene(const String p_scene_path);
	Ref<ScenePaletteSceneData> get_scene(const int p_index) const;
	void remove_scene(const int p_index);
};

class ScenePaletteSceneData : public Resource {
	GDCLASS(ScenePaletteSceneData, Resource);

	String scene_path;
	String display_name;
	Size2i grid_step;

protected:
	static void _bind_methods();

public:
	String get_scene_path() const { return scene_path; }
	void set_scene_path(const String &p_path) { scene_path = p_path; }
	String get_display_name() const { return display_name; }
	void set_display_name(const String &p_name) { display_name = p_name; }
	Size2i get_grid_step() const { return grid_step; }
	void set_grid_step(const Size2i &p_size) { grid_step = p_size; }
};
