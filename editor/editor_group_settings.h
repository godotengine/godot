/*************************************************************************/
/*  editor_group_settings.h                                           */
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

#ifndef EDITOR_GROUP_SETTINGS_H
#define EDITOR_GROUP_SETTINGS_H

#include "core/ordered_hash_map.h"
#include "core/os/file_access.h"
#include "core/ustring.h"
#include "scene/gui/tree.h"

#include "editor_file_dialog.h"
#include "editor_file_system.h"

class EditorGroupSettings : public VBoxContainer {

	GDCLASS(EditorGroupSettings, VBoxContainer);

	enum {
		BUTTON_MOVE_UP,
		BUTTON_MOVE_DOWN,
		BUTTON_DELETE
	};

	String group_changed;

	struct GroupInfo {
		String name;
		String description;
		int order;

		bool operator==(const GroupInfo &p_info) {
			return order == p_info.order;
		}

		GroupInfo() {
		}
	};

	struct StringComparator {

		bool operator()(const String &g_a, const String &g_b) const {

			return is_str_less(g_a.ptr(), g_b.ptr());
		}
	};

	Vector<String> names;
	HashMap<String, GroupInfo> group_cache;

	bool updating_group;
	String selected_group;

	Tree *tree;
	LineEdit *create_group_name;
	LineEdit *create_group_description;

	ConfirmationDialog *remove_confirmation;

	bool _group_exists(const String &p_name);
	bool _group_name_is_valid(const String &p_name, String *r_error = NULL);

	void _item_edited();
	void _item_selected();
	void _item_activated();
	void _item_button_pressed(Object *p_item, int p_column, int p_button);

	void _create_button_pressed();
	void _confirm_delete();

	void _remove_node_references(Node *p_node, const String &p_name);
	void _rename_node_references(Node *p_node, const String &p_old_name, const String &p_new_name);

	void _get_node(const String &p_name);
	void _get_all_scenes(EditorFileSystemDirectory *p_dir, Set<String> &r_list);

	void _create_group(const String &p_name, const String &p_description);
	void _delete_group(const String &p_name);
	void _rename_group(const String &p_old_name, const String &p_new_name);
	void _set_description(const String &p_name, const String &p_description);

	void _remove_references(const String &p_name);
	void _rename_references(const String &p_old_name, const String &p_new_name);

	void _init_groups();
	void _save_groups();

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void update_tree();

	void create_group(const String &p_name, const String &p_description);
	void delete_group(const String &p_name);
	void rename_group(const String &p_old_name, const String &p_new_name);

	void get_groups(List<String> *current_groups);

	EditorGroupSettings();
	~EditorGroupSettings();
};

#endif
