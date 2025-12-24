/**************************************************************************/
/*  snapshot_data.cpp                                                     */
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

#include "snapshot_data.h"

#include "core/core_bind.h"
#include "core/io/compression.h"
#include "core/object/script_language.h"
#include "scene/debugger/scene_debugger.h"

#if defined(MODULE_GDSCRIPT_ENABLED) && defined(DEBUG_ENABLED)
#include "modules/gdscript/gdscript.h"
#endif

SnapshotDataObject::SnapshotDataObject(SceneDebuggerObject &p_obj, GameStateSnapshot *p_snapshot, ResourceCache &resource_cache) :
		snapshot(p_snapshot) {
	remote_object_id = p_obj.id;
	type_name = p_obj.class_name;

	for (const SceneDebuggerObject::SceneDebuggerProperty &prop : p_obj.properties) {
		PropertyInfo pinfo = prop.first;
		Variant pvalue = prop.second;

		if (pinfo.type == Variant::OBJECT && pvalue.is_string()) {
			String path = pvalue;
			// If a resource is followed by a ::, it is a nested resource (like a sub_resource in a .tscn file).
			// To get a reference to it, first we load the parent resource (the .tscn, for example), then,
			// we load the child resource. The parent resource (dependency) should not be destroyed before the child
			// resource (pvalue) is loaded.
			if (path.is_resource_file()) {
				// Built-in resource.
				String base_path = path.get_slice("::", 0);
				if (!resource_cache.cache.has(base_path)) {
					resource_cache.cache[base_path] = ResourceLoader::load(base_path);
					resource_cache.misses++;
				} else {
					resource_cache.hits++;
				}
			}
			if (!resource_cache.cache.has(path)) {
				resource_cache.cache[path] = ResourceLoader::load(path);
				resource_cache.misses++;
			} else {
				resource_cache.hits++;
			}
			pvalue = resource_cache.cache[path];

			if (pinfo.hint_string == "Script") {
				if (get_script() != pvalue) {
					set_script(Ref<RefCounted>());
					Ref<Script> scr(pvalue);
					if (scr.is_valid()) {
						ScriptInstance *scr_instance = scr->placeholder_instance_create(this);
						if (scr_instance) {
							set_script_instance(scr_instance);
						}
					}
				}
			}
		}
		prop_list.push_back(pinfo);
		prop_values[pinfo.name] = pvalue;
	}
}

bool SnapshotDataObject::_get(const StringName &p_name, Variant &r_ret) const {
	String name = p_name;

	if (name.begins_with("Metadata/")) {
		name = name.replace_first("Metadata/", "metadata/");
	}
	if (!prop_values.has(name)) {
		return false;
	}

	r_ret = prop_values[p_name];
	return true;
}

void SnapshotDataObject::_get_property_list(List<PropertyInfo> *p_list) const {
	p_list->clear(); // Sorry, don't want any categories.
	for (const PropertyInfo &prop : prop_list) {
		if (prop.name == "script") {
			// Skip the script property, it's always added by the non-virtual method.
			continue;
		}

		p_list->push_back(prop);
	}
}

void SnapshotDataObject::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_is_read_only"), &SnapshotDataObject::_is_read_only);
}

String SnapshotDataObject::get_node_path() {
	if (!is_node()) {
		return "";
	}
	SnapshotDataObject *current = this;
	String path;

	while (true) {
		String current_node_name = current->extra_debug_data["node_name"];
		if (current_node_name != "") {
			if (path != "") {
				path = current_node_name + "/" + path;
			} else {
				path = current_node_name;
			}
		}
		if (!current->extra_debug_data.has("node_parent")) {
			break;
		}
		current = snapshot->objects[current->extra_debug_data["node_parent"]];
	}
	return path;
}

String SnapshotDataObject::_get_script_name(Ref<Script> p_script) {
#if defined(MODULE_GDSCRIPT_ENABLED) && defined(DEBUG_ENABLED)
	// GDScripts have more specific names than base scripts, so use those names if possible.
	return GDScript::debug_get_script_name(p_script);
#else
	// Otherwise fallback to the base script's name.
	return p_script->get_global_name();
#endif
}

String SnapshotDataObject::get_name() {
	String found_type_name = type_name;

	// Ideally, we will name it after the script attached to it.
	Ref<Script> maybe_script = get_script();
	if (maybe_script.is_valid()) {
		String full_name;
		while (maybe_script.is_valid()) {
			String global_name = _get_script_name(maybe_script);
			if (global_name != "") {
				if (full_name != "") {
					full_name = global_name + "/" + full_name;
				} else {
					full_name = global_name;
				}
			}
			maybe_script = maybe_script->get_base_script().ptr();
		}

		found_type_name = type_name + "/" + full_name;
	}

	return found_type_name + "_" + uitos(remote_object_id);
}

bool SnapshotDataObject::is_refcounted() {
	return is_class(RefCounted::get_class_static());
}

bool SnapshotDataObject::is_node() {
	return is_class(Node::get_class_static());
}

bool SnapshotDataObject::is_class(const String &p_base_class) {
	return ClassDB::is_parent_class(type_name, p_base_class);
}

HashSet<ObjectID> SnapshotDataObject::_unique_references(const HashMap<String, ObjectID> &p_refs) {
	HashSet<ObjectID> obj_set;

	for (const KeyValue<String, ObjectID> &pair : p_refs) {
		obj_set.insert(pair.value);
	}

	return obj_set;
}

HashSet<ObjectID> SnapshotDataObject::get_unique_outbound_refernces() {
	return _unique_references(outbound_references);
}

HashSet<ObjectID> SnapshotDataObject::get_unique_inbound_references() {
	return _unique_references(inbound_references);
}

void GameStateSnapshot::_get_outbound_references(Variant &p_var, HashMap<String, ObjectID> &r_ret_val, const String &p_current_path) {
	String path_divider = p_current_path.size() > 0 ? "/" : ""; // Make sure we don't start with a /.
	switch (p_var.get_type()) {
		case Variant::Type::INT:
		case Variant::Type::OBJECT: { // Means ObjectID.
			ObjectID as_id = ObjectID((uint64_t)p_var);
			if (!objects.has(as_id)) {
				return;
			}
			r_ret_val[p_current_path] = as_id;
			break;
		}
		case Variant::Type::DICTIONARY: {
			Dictionary dict = (Dictionary)p_var;
			LocalVector<Variant> keys = dict.get_key_list();
			for (Variant &k : keys) {
				// The dictionary key _could be_ an object. If it is, we name the key property with the same name as the value, but with _key appended to it.
				_get_outbound_references(k, r_ret_val, p_current_path + path_divider + (String)k + "_key");
				Variant v = dict.get(k, Variant());
				_get_outbound_references(v, r_ret_val, p_current_path + path_divider + (String)k);
			}
			break;
		}
		case Variant::Type::ARRAY: {
			Array arr = (Array)p_var;
			int i = 0;
			for (Variant &v : arr) {
				_get_outbound_references(v, r_ret_val, p_current_path + path_divider + itos(i));
				i++;
			}
			break;
		}
		default: {
			break;
		}
	}
}

void GameStateSnapshot::_get_rc_cycles(
		SnapshotDataObject *p_obj,
		SnapshotDataObject *p_source_obj,
		HashSet<SnapshotDataObject *> p_traversed_objs,
		LocalVector<String> &r_ret_val,
		const String &p_current_path) {
	// We're at the end of this branch and it was a cycle.
	if (p_obj == p_source_obj && p_current_path != "") {
		r_ret_val.push_back(p_current_path);
		return;
	}

	// Go through each of our children and try traversing them.
	for (const KeyValue<String, ObjectID> &next_child : p_obj->outbound_references) {
		SnapshotDataObject *next_obj = p_obj->snapshot->objects[next_child.value];
		String next_name = next_obj == p_source_obj ? "self" : next_obj->get_name();
		String current_name = p_obj == p_source_obj ? "self" : p_obj->get_name();
		String child_path = current_name + "[\"" + next_child.key + U"\"] → " + next_name;
		if (p_current_path != "") {
			child_path = p_current_path + "\n" + child_path;
		}

		SnapshotDataObject *next = objects[next_child.value];
		if (next != nullptr && next->is_class(RefCounted::get_class_static()) && !next->is_class(WeakRef::get_class_static()) && !p_traversed_objs.has(next)) {
			HashSet<SnapshotDataObject *> traversed_copy = p_traversed_objs;
			if (p_obj != p_source_obj) {
				traversed_copy.insert(p_obj);
			}
			_get_rc_cycles(next, p_source_obj, traversed_copy, r_ret_val, child_path);
		}
	}
}

void GameStateSnapshot::recompute_references() {
	for (const KeyValue<ObjectID, SnapshotDataObject *> &obj : objects) {
		Dictionary values;
		for (const KeyValue<StringName, Variant> &kv : obj.value->prop_values) {
			// Should only ever be one entry in this context.
			values[kv.key] = kv.value;
		}

		Variant values_variant(values);
		HashMap<String, ObjectID> refs;
		_get_outbound_references(values_variant, refs);

		obj.value->outbound_references = refs;

		for (const KeyValue<String, ObjectID> &kv : refs) {
			// Get the guy we are pointing to, and indicate the name of _our_ property that is pointing to them.
			if (objects.has(kv.value)) {
				objects[kv.value]->inbound_references[kv.key] = obj.key;
			}
		}
	}

	for (const KeyValue<ObjectID, SnapshotDataObject *> &obj : objects) {
		if (!obj.value->is_class(RefCounted::get_class_static()) || obj.value->is_class(WeakRef::get_class_static())) {
			continue;
		}
		HashSet<SnapshotDataObject *> traversed_objs;
		LocalVector<String> cycles;

		_get_rc_cycles(obj.value, obj.value, traversed_objs, cycles, "");
		Array cycles_array;
		for (const String &cycle : cycles) {
			cycles_array.push_back(cycle);
		}
		obj.value->extra_debug_data["ref_cycles"] = cycles_array;
	}
}

Ref<GameStateSnapshot> GameStateSnapshot::create_ref(const String &p_snapshot_name, const Vector<uint8_t> &p_snapshot_buffer) {
	Ref<GameStateSnapshot> snapshot;
	snapshot.instantiate();
	snapshot->name = p_snapshot_name;

	// Snapshots may have been created by an older version of the editor. Handle parsing old snapshot versions here based on the version number.

	Vector<uint8_t> snapshot_buffer_decompressed;
	int success = Compression::decompress_dynamic(&snapshot_buffer_decompressed, -1, p_snapshot_buffer.ptr(), p_snapshot_buffer.size(), Compression::MODE_DEFLATE);
	ERR_FAIL_COND_V_MSG(success != Z_OK, nullptr, "ObjectDB Snapshot could not be parsed. Failed to decompress snapshot.");
	CoreBind::Marshalls *m = CoreBind::Marshalls::get_singleton();
	Array snapshot_data = m->base64_to_variant(m->raw_to_base64(snapshot_buffer_decompressed));
	ERR_FAIL_COND_V_MSG(snapshot_data.is_empty(), nullptr, "ObjectDB Snapshot could not be parsed. Variant array is empty.");
	const Variant &first_item = snapshot_data[0];
	ERR_FAIL_COND_V_MSG(first_item.get_type() != Variant::DICTIONARY, nullptr, "ObjectDB Snapshot could not be parsed. First item is not a Dictionary.");
	snapshot->snapshot_context = first_item;

	SnapshotDataObject::ResourceCache resource_cache;
	for (int i = 1; i < snapshot_data.size(); i += 4) {
		SceneDebuggerObject obj;
		obj.deserialize(uint64_t(snapshot_data[i + 0]), snapshot_data[i + 1], snapshot_data[i + 2]);
		ERR_FAIL_COND_V_MSG(snapshot_data[i + 3].get_type() != Variant::DICTIONARY, nullptr, "ObjectDB Snapshot could not be parsed. Extra debug data is not a Dictionary.");

		if (obj.id.is_null()) {
			continue;
		}

		snapshot->objects[obj.id] = memnew(SnapshotDataObject(obj, snapshot.ptr(), resource_cache));
		snapshot->objects[obj.id]->extra_debug_data = (Dictionary)snapshot_data[i + 3];
	}

	snapshot->recompute_references();
	print_verbose("Resource cache hits: " + String::num(resource_cache.hits) + ". Resource cache misses: " + String::num(resource_cache.misses));
	return snapshot;
}

GameStateSnapshot::~GameStateSnapshot() {
	for (const KeyValue<ObjectID, SnapshotDataObject *> &item : objects) {
		memdelete(item.value);
	}
}
