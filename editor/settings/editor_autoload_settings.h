/**************************************************************************/
/*  editor_autoload_settings.h                                            */
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

#include "scene/gui/box_container.h"

class Button;
class EditorFileDialog;
class SceneCreateDialog;
class ScriptCreateDialog;
class Tree;

class EditorAutoloadSettings : public VBoxContainer {
	GDCLASS(EditorAutoloadSettings, VBoxContainer);

	enum {
		BUTTON_MOVE_UP,
		BUTTON_MOVE_DOWN,
		BUTTON_DELETE,
	};

	String path = "res://";
	String autoload_changed = "autoload_changed";

	struct AutoloadInfo {
		String name;
		String path;
		bool is_singleton = false;
		bool in_editor = false;
		int order = 0;
		Node *node = nullptr;

		bool operator==(const AutoloadInfo &p_info) const {
			return order == p_info.order;
		}
	};

	List<AutoloadInfo> autoload_cache;

	bool updating_autoload = false;
	String selected_autoload;

	Tree *tree = nullptr;
	Button *browse_button = nullptr;
	Button *create_script_autoload = nullptr;
	Button *create_scene_autoload = nullptr;
	EditorFileDialog *autoload_file_dialog = nullptr;
	EditorFileDialog *scene_file_dialog = nullptr;
	ScriptCreateDialog *script_create_dialog = nullptr;
	SceneCreateDialog *scene_create_dialog = nullptr;

	bool _autoload_name_is_valid(const String &p_name, String *r_error = nullptr);

	void _autoload_selected();
	void _autoload_edited();
	void _autoload_button_pressed(Object *p_item, int p_column, int p_button, MouseButton p_mouse_button);
	void _autoload_activated();
	void _autoload_open(const String &fpath);
	Node *_create_autoload(const String &p_path);

	void _create_script_autoload();
	void _create_scene_autoload();
	void _autoload_file_selected(const String &p_path);
	void _scene_file_selected(const String &p_path);

	void _script_created(Ref<Script> p_script);
	void _scene_created();
	void _add_autoload(const String &p_name, const String &p_path);

	Variant get_drag_data_fw(const Point2 &p_point, Control *p_control);
	bool can_drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_control) const;
	void drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_control);

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void init_autoloads();
	void update_autoload();
	bool autoload_add(const String &p_name, const String &p_path);
	void autoload_remove(const String &p_name);

	EditorAutoloadSettings();
	~EditorAutoloadSettings();
};
