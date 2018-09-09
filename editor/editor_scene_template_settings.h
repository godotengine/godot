/*************************************************************************/
/*  editor_scene_template_settings.h                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2018 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2018 Godot Engine contributors (cf. AUTHORS.md)    */
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

#ifndef EDITOR_SCENE_TEMPLATE_SETTINGS_H
#define EDITOR_SCENE_TEMPLATE_SETTINGS_H

#include "scene/gui/tree.h"

#include "editor_file_dialog.h"

class EditorSceneTemplateSettings : public VBoxContainer {

	GDCLASS(EditorSceneTemplateSettings, VBoxContainer);

	enum {
		BUTTON_OPEN,
		BUTTON_MOVE_UP,
		BUTTON_MOVE_DOWN,
		BUTTON_DELETE
	};

	String scene_template_changed;

	struct SceneTemplateInfo {
		StringName name;
		String path;
		int order;

		bool operator==(const SceneTemplateInfo &p_info) {
			return order == p_info.order;
		}

		SceneTemplateInfo(const StringName &p_name = "", const String &p_path = "", int p_order = 0) {
			name = p_name;
			path = p_path;
			order = p_order;
		}
	};

	HashMap<StringName, SceneTemplateInfo *> scene_template_map;
	List<SceneTemplateInfo> scene_template_cache;
	bool updating_scene_template;
	int number_of_scene_templates;
	String selected_scene_template;

	Tree *tree;
	EditorLineEditFileChooser *scene_template_add_path;
	LineEdit *scene_template_add_name;

	bool _scene_template_name_is_valid(const String &p_name, String *r_error = NULL);

	void _scene_template_add();
	void _scene_template_selected();
	void _scene_template_edited();
	void _scene_template_button_pressed(Object *p_item, int p_column, int p_button);
	void _scene_template_activated();
	void _scene_template_open(const String &fpath);
	void _scene_template_file_callback(const String &p_path);
	Node *_create_scene_template(const String &p_path);

	Variant get_drag_data_fw(const Point2 &p_point, Control *p_control);
	bool can_drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_control) const;
	void drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_control);

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void update_scene_templates();
	void scene_template_add(const String &p_name, const String &p_path);
	void scene_template_remove(const String &p_name);

	void set_template_name(const String &p_name);
	String get_template_name() const;
	void set_template_path(const String &p_path);
	String get_template_path() const;
	LineEdit *get_name_line_edit() { return scene_template_add_name; }
	LineEdit *get_path_line_edit() { return scene_template_add_path->get_line_edit(); }

	void get_scene_template_list(List<StringName> *p_list) const { scene_template_map.get_key_list(p_list); }
	bool scene_template_exists(const StringName &p_name) const { return scene_template_map.has(p_name); }
	String scene_template_get_path(const StringName &p_name) const { return scene_template_map.has(p_name) && scene_template_map[p_name] ? scene_template_map[p_name]->path : String(); }
	StringName scene_template_get_base(const StringName &p_name) const;
	StringName scene_template_get_base_file(const String &p_path) const;
	StringName scene_template_get_name(const String &p_path) const;
	Node *scene_template_instance(const StringName &p_name);
	bool scene_template_is_parent(const StringName &p_name, const StringName &p_inherits);

	EditorSceneTemplateSettings();
	~EditorSceneTemplateSettings();
};

#endif // EDITOR_SCENE_TEMPLATE_SETTINGS_H
