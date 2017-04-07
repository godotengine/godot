/*************************************************************************/
/*  editor_data.h                                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "editor/editor_plugin.h"
#include "list.h"
#include "pair.h"
#include "scene/resources/texture.h"
#include "undo_redo.h"

class EditorHistory {

	enum {

		HISTORY_MAX = 64
	};

	struct Obj {

		REF ref;
		ObjectID object;
		String property;
	};

	struct History {

		Vector<Obj> path;
		int level;
	};
	friend class EditorData;

	Vector<History> history;
	int current;

	//Vector<EditorPlugin*> editor_plugins;

	struct PropertyData {

		String name;
		Variant value;
	};

	void _cleanup_history();

	void _add_object(ObjectID p_object, const String &p_property, int p_level_change);

public:
	bool is_at_begining() const;
	bool is_at_end() const;

	void add_object(ObjectID p_object);
	void add_object(ObjectID p_object, const String &p_subprop);
	void add_object(ObjectID p_object, int p_relevel);

	int get_history_len();
	int get_history_pos();
	ObjectID get_history_obj(int p_obj) const;

	bool next();
	bool previous();
	ObjectID get_current();

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
		Ref<Texture> icon;
	};

private:
	Vector<EditorPlugin *> editor_plugins;

	struct PropertyData {

		String name;
		Variant value;
	};
	Map<String, Vector<CustomType> > custom_types;

	List<PropertyData> clipboard;
	UndoRedo undo_redo;

	void _cleanup_history();

	struct EditedScene {
		Node *root;
		Dictionary editor_states;
		List<Node *> selection;
		Vector<EditorHistory::History> history_stored;
		int history_current;
		Dictionary custom_state;
		uint64_t version;
		NodePath live_edit_root;
	};

	Vector<EditedScene> edited_scene;
	int current_edited_scene;

	bool _find_updated_instances(Node *p_root, Node *p_node, Set<String> &checked_paths);

public:
	EditorPlugin *get_editor(Object *p_object);
	EditorPlugin *get_subeditor(Object *p_object);
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

	void save_editor_global_states();
	void restore_editor_global_states();

	void add_custom_type(const String &p_type, const String &p_inherits, const Ref<Script> &p_script, const Ref<Texture> &p_icon);
	void remove_custom_type(const String &p_type);
	const Map<String, Vector<CustomType> > &get_custom_types() const { return custom_types; }

	int add_edited_scene(int p_at_pos);
	void move_edited_scene_index(int p_idx, int p_to_idx);
	void remove_scene(int p_idx);
	void set_edited_scene(int p_idx);
	void set_edited_scene_root(Node *p_root);
	int get_edited_scene() const;
	Node *get_edited_scene_root(int p_idx = -1);
	int get_edited_scene_count() const;
	String get_scene_title(int p_idx) const;
	String get_scene_path(int p_idx) const;
	String get_scene_type(int p_idx) const;
	Ref<Script> get_scene_root_script(int p_idx) const;
	void set_edited_scene_version(uint64_t version, int p_scene_idx = -1);
	uint64_t get_edited_scene_version() const;
	uint64_t get_scene_version(int p_idx) const;
	void clear_edited_scenes();
	void set_edited_scene_live_edit_root(const NodePath &p_root);
	NodePath get_edited_scene_live_edit_root();
	bool check_and_update_scene(int p_idx);
	void move_edited_scene_to_index(int p_idx);

	void set_plugin_window_layout(Ref<ConfigFile> p_layout);
	void get_plugin_window_layout(Ref<ConfigFile> p_layout);

	void save_edited_scene_state(EditorSelection *p_selection, EditorHistory *p_history, const Dictionary &p_custom);
	Dictionary restore_edited_scene_state(EditorSelection *p_selection, EditorHistory *p_history);
	void notify_edited_scene_changed();

	EditorData();
};

class EditorSelection : public Object {

	GDCLASS(EditorSelection, Object);

public:
	Map<Node *, Object *> selection;

	bool changed;
	bool nl_changed;

	void _node_removed(Node *p_node);

	List<Object *> editor_plugins;
	List<Node *> selected_node_list;

	void _update_nl();
	Array _get_selected_nodes();
	Array _get_transformable_selected_nodes();

protected:
	static void _bind_methods();

public:
	void add_node(Node *p_node);
	void remove_node(Node *p_node);
	bool is_selected(Node *) const;

	template <class T>
	T *get_node_editor_data(Node *p_node) {
		if (!selection.has(p_node))
			return NULL;
		Object *obj = selection[p_node];
		if (!obj)
			return NULL;
		return obj->cast_to<T>();
	}

	void add_editor_plugin(Object *p_object);

	void update();
	void clear();

	List<Node *> &get_selected_node_list();
	Map<Node *, Object *> &get_selection() { return selection; }

	EditorSelection();
	~EditorSelection();
};

#endif // EDITOR_DATA_H
