/*************************************************************************/
/*  editor_autoload_settings.h                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef EDITOR_AUTOLOAD_SETTINGS_H
#define EDITOR_AUTOLOAD_SETTINGS_H

#include "scene/gui/tree.h"

#include "editor_file_dialog.h"

class EditorAutoloadSettings : public VBoxContainer {
	GDCLASS(EditorAutoloadSettings, VBoxContainer);

	enum {
		BUTTON_OPEN,
		BUTTON_MOVE_UP,
		BUTTON_MOVE_DOWN,
		BUTTON_DELETE
	};

	String autoload_changed;

	struct AutoLoadInfo {
		String name;
		String path;
		bool is_singleton;
		bool in_editor;
		int order;
		Node *node;

		bool operator==(const AutoLoadInfo &p_info) const {
			return order == p_info.order;
		}

		AutoLoadInfo() {
			is_singleton = false;
			in_editor = false;
			node = nullptr;
		}
	};

	List<AutoLoadInfo> autoload_cache;

	bool updating_autoload;
	int number_of_autoloads;
	String selected_autoload;

	Tree *tree;
	LineEdit *autoload_add_name;
	Button *add_autoload;
	LineEdit *autoload_add_path;
	Button *browse_button;
	EditorFileDialog *file_dialog;

	bool _autoload_name_is_valid(const String &p_name, String *r_error = nullptr);

	void _autoload_add();
	void _autoload_selected();
	void _autoload_edited();
	void _autoload_button_pressed(Object *p_item, int p_column, int p_button);
	void _autoload_activated();
	void _autoload_path_text_changed(const String p_path);
	void _autoload_text_entered(const String p_name);
	void _autoload_text_changed(const String p_name);
	void _autoload_open(const String &fpath);
	void _autoload_file_callback(const String &p_path);
	Node *_create_autoload(const String &p_path);

	Variant get_drag_data_fw(const Point2 &p_point, Control *p_control);
	bool can_drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_control) const;
	void drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_control);

	void _set_autoload_add_path(const String &p_text);
	void _browse_autoload_add_path();

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void update_autoload();
	bool autoload_add(const String &p_name, const String &p_path);
	void autoload_remove(const String &p_name);

	EditorAutoloadSettings();
	~EditorAutoloadSettings();
};

#endif
