/*************************************************************************/
/*  scene_debugger.h                                                     */
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

#ifndef SCENE_DEBUGGER_H
#define SCENE_DEBUGGER_H

#include "core/array.h"
#include "core/class_db.h"
#include "core/pair.h"
#include "core/ustring.h"

class Script;

class SceneDebugger {
public:
	static void initialize();
	static void deinitialize();

#ifdef DEBUG_ENABLED
private:
	static void _save_node(ObjectID id, const String &p_path);
	static void _set_object_property(ObjectID p_id, const String &p_property, const Variant &p_value);
	static void _send_object_id(ObjectID p_id, int p_max_size = 1 << 20);

public:
	static Error parse_message(void *p_user, const String &p_msg, const Array &p_args, bool &r_captured);
	static void add_to_cache(const String &p_filename, Node *p_node);
	static void remove_from_cache(const String &p_filename, Node *p_node);
#endif
};

#ifdef DEBUG_ENABLED
class SceneDebuggerObject {
private:
	void _parse_script_properties(Script *p_script, ScriptInstance *p_instance);

public:
	typedef Pair<PropertyInfo, Variant> SceneDebuggerProperty;
	ObjectID id;
	String class_name;
	List<SceneDebuggerProperty> properties;

	SceneDebuggerObject(ObjectID p_id);
	SceneDebuggerObject() {}

	void serialize(Array &r_arr, int p_max_size = 1 << 20);
	void deserialize(const Array &p_arr);
};

class SceneDebuggerTree {
public:
	struct RemoteNode {
		int child_count;
		String name;
		String type_name;
		ObjectID id;

		RemoteNode(int p_child, const String &p_name, const String &p_type, ObjectID p_id) {
			child_count = p_child;
			name = p_name;
			type_name = p_type;
			id = p_id;
		}

		RemoteNode() {}
	};

	List<RemoteNode> nodes;

	void serialize(Array &r_arr);
	void deserialize(const Array &p_arr);
	SceneDebuggerTree(Node *p_root);
	SceneDebuggerTree() {}
};

class LiveEditor {
private:
	friend class SceneDebugger;
	Map<int, NodePath> live_edit_node_path_cache;
	Map<int, String> live_edit_resource_cache;

	NodePath live_edit_root;
	String live_edit_scene;

	Map<String, Set<Node *>> live_scene_edit_cache;
	Map<Node *, Map<ObjectID, Node *>> live_edit_remove_list;

	void _send_tree();

	void _node_path_func(const NodePath &p_path, int p_id);
	void _res_path_func(const String &p_path, int p_id);

	void _node_set_func(int p_id, const StringName &p_prop, const Variant &p_value);
	void _node_set_res_func(int p_id, const StringName &p_prop, const String &p_value);
	void _node_call_func(int p_id, const StringName &p_method, VARIANT_ARG_DECLARE);
	void _res_set_func(int p_id, const StringName &p_prop, const Variant &p_value);
	void _res_set_res_func(int p_id, const StringName &p_prop, const String &p_value);
	void _res_call_func(int p_id, const StringName &p_method, VARIANT_ARG_DECLARE);
	void _root_func(const NodePath &p_scene_path, const String &p_scene_from);

	void _create_node_func(const NodePath &p_parent, const String &p_type, const String &p_name);
	void _instance_node_func(const NodePath &p_parent, const String &p_path, const String &p_name);
	void _remove_node_func(const NodePath &p_at);
	void _remove_and_keep_node_func(const NodePath &p_at, ObjectID p_keep_id);
	void _restore_node_func(ObjectID p_id, const NodePath &p_at, int p_at_pos);
	void _duplicate_node_func(const NodePath &p_at, const String &p_new_name);
	void _reparent_node_func(const NodePath &p_at, const NodePath &p_new_place, const String &p_new_name, int p_at_pos);

	LiveEditor() {
		singleton = this;
		live_edit_root = NodePath("/root");
	};

	static LiveEditor *singleton;

public:
	static LiveEditor *get_singleton();
};
#endif

#endif
