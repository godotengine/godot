/**************************************************************************/
/*  editor_global_variables_settings.h                                    */
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

#ifndef EDITOR_GLOBAL_VARIABLES_SETTINGS_H
#define EDITOR_GLOBAL_VARIABLES_SETTINGS_H

#include "scene/gui/box_container.h"
#include "scene/gui/button.h"
#include "scene/gui/tree.h"

class EditorFileDialog;

class EditorGlobalVariablesSettings : public VBoxContainer {
	GDCLASS(EditorGlobalVariablesSettings, VBoxContainer);

	enum {
		BUTTON_DELETE
	};

	String path = "res://";
	String global_variables_changed = "global_variables_changed";

	struct GlobalVariableInfo {
		String name;

		bool operator<(const GlobalVariableInfo &p_arg) const {
			return name < p_arg.name;
		}
	};

	List<GlobalVariableInfo> global_variables_cache; // ????

	bool updating_global_variables = false;
	String selected_global_variable;

	Tree *tree = nullptr;
	LineEdit *global_variable_add_name = nullptr;
	Button *add_global_variable = nullptr;
	Label *error_message = nullptr;

	void _global_variable_add();
	void _global_variable_selected();
	void _global_variable_edited();
	void _global_variable_button_pressed(Object *p_item, int p_column, int p_button, MouseButton p_mouse_button);
	void _global_variable_text_submitted(const String p_name);
	void _global_variable_text_changed(const String p_name);
	void _global_variable_file_callback(const String &p_path);

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	static bool global_variable_name_is_valid(const String &p_name, String *r_error = nullptr);

	void update_global_variables();
	bool global_variable_add(const String &p_name);
	void global_variable_remove(const String &p_name);

	EditorGlobalVariablesSettings();
	~EditorGlobalVariablesSettings();
};

#endif // EDITOR_GLOBAL_VARIABLES_SETTINGS_H
