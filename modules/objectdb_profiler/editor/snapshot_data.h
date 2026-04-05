/**************************************************************************/
/*  snapshot_data.h                                                       */
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

#include "editor/debugger/editor_debugger_inspector.h"

class GameStateSnapshot;

class SnapshotDataObject : public Object {
	GDCLASS(SnapshotDataObject, Object);

	HashSet<ObjectID> _unique_references(const HashMap<String, ObjectID> &p_refs);
	String _get_script_name(Ref<Script> p_script);

public:
	GameStateSnapshot *snapshot = nullptr;
	Dictionary extra_debug_data;
	HashMap<String, ObjectID> outbound_references;
	HashMap<String, ObjectID> inbound_references;

	HashSet<ObjectID> get_unique_outbound_refernces();
	HashSet<ObjectID> get_unique_inbound_references();

	uint64_t remote_object_id = 0;
	String type_name;
	LocalVector<PropertyInfo> prop_list;
	HashMap<StringName, Variant> prop_values;
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;

	struct ResourceCache {
		HashMap<String, Ref<Resource>> cache;
		int misses = 0;
		int hits = 0;
	};

	SnapshotDataObject(SceneDebuggerObject &p_obj, GameStateSnapshot *p_snapshot, ResourceCache &resource_cache);

	String get_name();
	String get_node_path();
	bool is_refcounted();
	bool is_node();
	bool is_class(const String &p_base_class);

protected:
	// Snapshots are inherently read-only. Can't edit the past.
	bool _is_read_only() { return true; }
	static void _bind_methods();
};

class GameStateSnapshot : public RefCounted {
	GDCLASS(GameStateSnapshot, RefCounted);

	void _get_outbound_references(Variant &p_var, HashMap<String, ObjectID> &r_ret_val, const String &p_current_path = "");
	void _get_rc_cycles(SnapshotDataObject *p_obj, SnapshotDataObject *p_source_obj, HashSet<SnapshotDataObject *> p_traversed_objs, LocalVector<String> &r_ret_val, const String &p_current_path = "");

public:
	String name;
	HashMap<ObjectID, SnapshotDataObject *> objects;
	Dictionary snapshot_context;

	static Ref<GameStateSnapshot> create_ref(const String &p_snapshot_name, const Vector<uint8_t> &p_snapshot_buffer);
	~GameStateSnapshot();

	void recompute_references();
};
