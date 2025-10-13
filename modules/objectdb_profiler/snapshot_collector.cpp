/**************************************************************************/
/*  snapshot_collector.cpp                                                */
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

#include "snapshot_collector.h"

#include "core/core_bind.h"
#include "core/debugger/engine_debugger.h"
#include "core/io/compression.h"
#include "core/os/time.h"
#include "core/version.h"
#include "scene/main/node.h"
#include "scene/main/window.h"

void SnapshotCollector::initialize() {
	pending_snapshots.clear();
	EngineDebugger::register_message_capture("snapshot", EngineDebugger::Capture(nullptr, SnapshotCollector::parse_message));
}

void SnapshotCollector::deinitialize() {
	EngineDebugger::unregister_message_capture("snapshot");
	pending_snapshots.clear();
}

void SnapshotCollector::snapshot_objects(Array *p_arr, Dictionary &p_snapshot_context) {
	print_verbose("Starting to snapshot");
	p_arr->clear();

	// Gather all ObjectIDs first. The ObjectDB will be locked in debug_objects, so we can't serialize until it exits.

	// In rare cases, the object may be deleted as the snapshot is taken. So, we store the object's class name to give users a clue about what went wrong.
	LocalVector<Pair<ObjectID, StringName>> debugger_object_ids;
	debugger_object_ids.reserve(ObjectDB::get_object_count());

	ObjectDB::debug_objects(
			[](Object *p_obj, void *p_user_data) {
				LocalVector<Pair<ObjectID, StringName>> *debugger_object_ids_ptr = (LocalVector<Pair<ObjectID, StringName>> *)p_user_data;
				debugger_object_ids_ptr->push_back(Pair<ObjectID, StringName>(p_obj->get_instance_id(), p_obj->get_class_name()));
			},
			(void *)&debugger_object_ids);

	// Get SnapshotDataTransportObject from ObjectID list now that DB is unlocked.
	LocalVector<SnapshotDataTransportObject> debugger_objects;
	debugger_objects.reserve(debugger_object_ids.size());
	for (Pair<ObjectID, StringName> ids : debugger_object_ids) {
		ObjectID oid = ids.first;
		Object *obj = ObjectDB::get_instance(oid);
		if (unlikely(obj == nullptr)) {
			print_verbose(vformat("Object of class '%s' with ID %ud was found to be deleted after ObjectDB was snapshotted.", ids.second, (uint64_t)oid));
			continue;
		}

		if (ids.second == SNAME("EditorInterface")) {
			// The EditorInterface + EditorNode is _kind of_ constructed in a debug game, but many properties are null
			// We can prevent it from being constructed, but that would break other projects so better to just skip it.
			continue;
		}

		// This is the same way objects in the remote scene tree are serialized,
		// but here we add a few extra properties via the extra_debug_data dictionary.
		SnapshotDataTransportObject debug_data(obj);

		// If we're RefCounted, send over our RefCount too. Could add code here to add a few other interesting properties.
		RefCounted *ref = Object::cast_to<RefCounted>(obj);
		if (ref) {
			debug_data.extra_debug_data["ref_count"] = ref->get_reference_count();
		}

		Node *node = Object::cast_to<Node>(obj);
		if (node) {
			debug_data.extra_debug_data["node_name"] = node->get_name();
			if (node->get_parent() != nullptr) {
				debug_data.extra_debug_data["node_parent"] = node->get_parent()->get_instance_id();
			}

			debug_data.extra_debug_data["node_is_scene_root"] = SceneTree::get_singleton()->get_root() == node;

			Array children;
			for (int i = 0; i < node->get_child_count(); i++) {
				children.push_back(node->get_child(i)->get_instance_id());
			}
			debug_data.extra_debug_data["node_children"] = children;
		}

		debugger_objects.push_back(debug_data);
	}

	// Add a header to the snapshot with general data about the state of the game, not tied to any particular object.
	p_snapshot_context["mem_usage"] = Memory::get_mem_usage();
	p_snapshot_context["mem_max_usage"] = Memory::get_mem_max_usage();
	p_snapshot_context["timestamp"] = Time::get_singleton()->get_unix_time_from_system();
	p_snapshot_context["game_version"] = get_godot_version_string();
	p_arr->push_back(p_snapshot_context);
	for (SnapshotDataTransportObject &debug_data : debugger_objects) {
		debug_data.serialize(*p_arr);
		p_arr->push_back(debug_data.extra_debug_data);
	}

	print_verbose("Snapshot size: " + String::num_uint64(p_arr->size()));
}

Error SnapshotCollector::parse_message(void *p_user, const String &p_msg, const Array &p_args, bool &r_captured) {
	r_captured = true;
	if (p_msg == "request_prepare_snapshot") {
		int request_id = p_args[0];
		Dictionary snapshot_context;
		snapshot_context["editor_version"] = (String)p_args[1];
		Array objects;
		snapshot_objects(&objects, snapshot_context);
		// Debugger networking has a limit on both how many objects can be queued to send and how
		// many bytes can be queued to send. Serializing to a string means we never hit the object
		// limit, and only have to deal with the byte limit.
		// Compress the snapshot in the game client to make sending the snapshot from game to editor a little faster.
		CoreBind::Marshalls *m = CoreBind::Marshalls::get_singleton();
		Vector<uint8_t> objs_buffer = m->base64_to_raw(m->variant_to_base64(objects));
		Vector<uint8_t> objs_buffer_compressed;
		objs_buffer_compressed.resize(objs_buffer.size());
		int new_size = Compression::compress(objs_buffer_compressed.ptrw(), objs_buffer.ptrw(), objs_buffer.size(), Compression::MODE_DEFLATE);
		objs_buffer_compressed.resize(new_size);
		pending_snapshots[request_id] = objs_buffer_compressed;

		// Tell the editor how long the snapshot is.
		Array resp = { request_id, pending_snapshots[request_id].size() };
		EngineDebugger::get_singleton()->send_message("snapshot:snapshot_prepared", resp);

	} else if (p_msg == "request_snapshot_chunk") {
		int request_id = p_args[0];
		int begin = p_args[1];
		int end = p_args[2];

		Array resp = { request_id, pending_snapshots[request_id].slice(begin, end) };
		EngineDebugger::get_singleton()->send_message("snapshot:snapshot_chunk", resp);

		// If we sent the last part of the string, delete it locally.
		if (end >= pending_snapshots[request_id].size()) {
			pending_snapshots.erase(request_id);
		}
	} else {
		r_captured = false;
	}
	return OK;
}

String SnapshotCollector::get_godot_version_string() {
	String hash = String(GODOT_VERSION_HASH);
	if (hash.length() != 0) {
		hash = " " + vformat("[%s]", hash.left(9));
	}
	return "v" GODOT_VERSION_FULL_BUILD + hash;
}
