/*************************************************************************/
/*  editor_data.h                                                        */
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

#ifndef EDITOR_DATA_H
#define EDITOR_DATA_H

#include "core/object/undo_redo.h"
#include "core/templates/list.h"
#include "core/templates/pair.h"
#include "editor/editor_plugin.h"
#include "editor/plugins/script_editor_plugin.h"
#include "scene/resources/texture.h"

class EditorHistory {
	enum {
		HISTORY_MAX = 64
	};

	struct Obj {
		REF ref;
		ObjectID object;
		String property;
		bool inspector_only = false;
	};

	struct History {
		Vector<Obj> path;
		int level = 0;
	};
	friend class EditorData;

	Vector<History> history;
	int current;

	//Vector<EditorPlugin*> editor_plugins;

	struct PropertyData {
		String name;
		Variant value;
	};

	void _add_object(ObjectID p_object, const String &p_property, int p_level_change, bool p_inspector_only = false);

public:
	void cleanup_history();

	bool is_at_beginning() const;
	bool is_at_end() const;

	void add_object_inspector_only(ObjectID p_object);
	void add_object(ObjectID p_object);
	void add_object(ObjectID p_object, const String &p_subprop);
	void add_object(ObjectID p_object, int p_relevel);

	int get_history_len();
	int get_history_pos();
	ObjectID get_history_obj(int p_obj) const;
	bool is_history_obj_inspector_only(int p_obj) const;

	bool next();
	bool previous();
	ObjectID get_current();
	bool is_current_inspector_only() const;

	int get_path_size() const;
	ObjectID get_path_object(int p_index) const;
	String get_path_property(int p_index) const;

	void clear();

	EditorHistory();
};

class EditorSelection;

class EditorData {
public:
	struct CustomType {
		String name;
		Ref<Script> script;
		Ref<Texture2D> icon;
	};

	struct EditedScene {
		Node *root = nullptr;
		String path;
		uint64_t file_modified_time = 0;
		Dictionary editor_states;
		List<Node *> selection;
		Vector<EditorHistory::History> history_stored;
		int history_current = 0;
		Dictionary custom_state;
		uint64_t version = 0;
		NodePath live_edit_root;
	};

private:
	Vector<EditorPlugin *> editor_plugins;

	struct PropertyData {
		String name;
		Variant value;
	};
	Map<String, Vector<CustomType>> custom_types;

	List<PropertyData> clipboard;
	UndoRedo undo_redo;
	Vector<Callable> undo_redo_callbacks;

	void _cleanup_history();

	Vector<EditedScene> edited_scene;
	int current_edited_scene;

	bool _find_updated_instances(Node *p_root, Node *p_node, Set<String> &checked_paths);

	HashMap<StringName, String> _script_class_icon_paths;
	HashMap<String, StringName> _script_class_file_to_path;

public:
	EditorPlugin *get_editor(Object *p_object);
	Vector<EditorPlugin *> get_subeditors(Object *p_object);
	EditorPlugin *get_editor(String p_name);

	void copy_object_params(Object *p_object);
	void paste_object_params(Object *p_object);

	Dictionary get_editor_states() const;
	Dictionary get_scene_editor_states(int p_idx) const;
	void set_editor_states(const Dictionary &p_states);
	void get_editor_breakpoints(List<String> *p_breakpoints);
	void clear_editor_states();
	void save_editor_external_data();
	void apply_changes_in_editors();

	void add_editor_plugin(EditorPlugin *p_plugin);
	void remove_editor_plugin(EditorPlugin *p_plugin);

	int get_editor_plugin_count() const;
	EditorPlugin *get_editor_plugin(int p_idx);

	UndoRedo &get_undo_redo();
	void add_undo_redo_inspector_hook_callback(Callable p_callable); // Callbacks should have 4 args: (Object* undo_redo, Object *modified_object, String property, Variant new_value)
	void remove_undo_redo_inspector_hook_callback(Callable p_callable);
	const Vector<Callable> get_undo_redo_inspector_hook_callback();

	void save_editor_global_states();
	void restore_editor_global_states();

	void add_custom_type(const String &p_type, const String &p_inherits, const Ref<Script> &p_script, const Ref<Texture2D> &p_icon);
	Variant instance_custom_type(const String &p_type, const String &p_inherits);
	void remove_custom_type(const String &p_type);
	const Map<String, Vector<CustomType>> &get_custom_types() const { return custom_types; }

	int add_edited_scene(int p_at_pos);
	void move_edited_scene_index(int p_idx, int p_to_idx);
	void remove_scene(int p_idx);
	void set_edited_scene(int p_idx);
	void set_edited_scene_root(Node *p_root);
	int get_edited_scene() const;
	Node *get_edited_scene_root(int p_idx = -1);
	int get_edited_scene_count() const;
	Vector<EditedScene> get_edited_scenes() const;
	String get_scene_title(int p_idx, bool p_always_strip_extension = false) const;
	String get_scene_path(int p_idx) const;
	String get_scene_type(int p_idx) const;
	void set_scene_path(int p_idx, const String &p_path);
	Ref<Script> get_scene_root_script(int p_idx) const;
	void set_edited_scene_version(uint64_t version, int p_scene_idx = -1);
	uint64_t get_scene_version(int p_idx) const;
	void set_scene_modified_time(int p_idx, uint64_t p_time);
	uint64_t get_scene_modified_time(int p_idx) const;
	void clear_edited_scenes();
	void set_edited_scene_live_edit_root(const NodePath &p_root);
	NodePath get_edited_scene_live_edit_root();
	bool check_and_update_scene(int p_idx);
	void move_edited_scene_to_index(int p_idx);
	bool call_build();

	void set_plugin_window_layout(Ref<ConfigFile> p_layout);
	void get_plugin_window_layout(Ref<ConfigFile> p_layout);

	void save_edited_scene_state(EditorSelection *p_selection, EditorHistory *p_history, const Dictionary &p_custom);
	Dictionary restore_edited_scene_state(EditorSelection *p_selection, EditorHistory *p_history);
	void notify_edited_scene_changed();
	void notify_resource_saved(const Ref<Resource> &p_resource);

	bool script_class_is_parent(const String &p_class, const String &p_inherits);
	StringName script_class_get_base(const String &p_class) const;
	Variant script_class_instance(const String &p_class);

	Ref<Script> script_class_load_script(const String &p_class) const;

	StringName script_class_get_name(const String &p_path) const;
	void script_class_set_name(const String &p_path, const StringName &p_class);

	String script_class_get_icon_path(const String &p_class) const;
	void script_class_set_icon_path(const String &p_class, const String &p_icon_path);
	void script_class_clear_icon_paths() { _script_class_icon_paths.clear(); }
	void script_class_save_icon_paths();
	void script_class_load_icon_paths();

	EditorData();
};

class EditorSelection : public Object {
	GDCLASS(EditorSelection, Object);

private:
	Map<Node *, Object *> selection;

	bool emitted;
	bool changed;
	bool nl_changed;

	void _node_removed(Node *p_node);

	List<Object *> editor_plugins;
	List<Node *> selected_node_list;

	void _update_nl();
	Array _get_transformable_selected_nodes();
	void _emit_change();

protected:
	static void _bind_methods();

public:
	TypedArray<Node> get_selected_nodes();
	void add_node(Node *p_node);
	void remove_node(Node *p_node);
	bool is_selected(Node *) const;

	template <class T>
	T *get_node_editor_data(Node *p_node) {
		if (!selection.has(p_node)) {
			return nullptr;
		}
		return Object::cast_to<T>(selection[p_node]);
	}

	void add_editor_plugin(Object *p_object);

	void update();
	void clear();

	List<Node *> &get_selected_node_list();
	List<Node *> get_full_selected_node_list();
	Map<Node *, Object *> &get_selection() { return selection; }

	EditorSelection();
	~EditorSelection();
};

#endif // EDITOR_DATA_H
