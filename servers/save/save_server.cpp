/**************************************************************************/
/*  save_server.cpp                                                       */
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

#include "save_server.h"

#include "core/config/engine.h"
#include "core/config/project_settings.h"
#include "core/crypto/crypto_core.h"
#include "core/io/dir_access.h"
#include "core/io/file_access.h"
#include "core/io/marshalls.h"
#include "core/io/resource_loader.h"
#include "core/io/resource_saver.h"
#include "core/os/time.h"
#include "core/variant/variant_utility.h"
#include "scene/main/node.h"
#include "scene/resources/packed_scene.h"
#include "scene/resources/snapshot.h"

SaveServer *SaveServer::singleton = nullptr;

SaveServer *SaveServer::get_singleton() {
	return singleton;
}

void SaveServer::register_settings() {
	// Register persistence settings with defaults
	GLOBAL_DEF_BASIC("application/persistence/save_format", 0);
	GLOBAL_DEF_BASIC("application/persistence/encryption_key", "");
	GLOBAL_DEF_BASIC("application/persistence/compression_enabled", true);
	GLOBAL_DEF_BASIC("application/persistence/backup_enabled", true);
	GLOBAL_DEF_BASIC("application/persistence/max_backups", 2);
	GLOBAL_DEF_BASIC("application/persistence/integrity_check_level", 1);
	GLOBAL_DEF_BASIC("application/persistence/save_path", "user://saves/");
}

void SaveServer::_bind_methods() {
	// Public API
	ClassDB::bind_method(D_METHOD("save_slot", "slot_name", "data", "async", "metadata", "thumbnail"), &SaveServer::save_slot, DEFVAL(true), DEFVAL(Dictionary()), DEFVAL(Ref<Resource>()));
	ClassDB::bind_method(D_METHOD("load_slot", "slot_name"), &SaveServer::load_slot);
	ClassDB::bind_method(D_METHOD("has_slot", "slot_name"), &SaveServer::has_slot);
	ClassDB::bind_method(D_METHOD("delete_slot", "slot_name"), &SaveServer::delete_slot);
	ClassDB::bind_method(D_METHOD("delete_snapshot", "snapshot_name"), &SaveServer::delete_snapshot);

	ClassDB::bind_method(D_METHOD("save_snapshot", "root", "slot_name", "async", "tags", "metadata", "thumbnail"), &SaveServer::save_snapshot, DEFVAL(true), DEFVAL(TypedArray<StringName>()), DEFVAL(Dictionary()), DEFVAL(Ref<Resource>()));
	ClassDB::bind_method(D_METHOD("load_snapshot", "root", "slot_name", "callback", "dynamic_respawn"), &SaveServer::load_snapshot, DEFVAL(Callable()), DEFVAL(false));
	ClassDB::bind_method(D_METHOD("has_snapshot", "slot_name"), &SaveServer::has_snapshot);

	ClassDB::bind_method(D_METHOD("_finish_load_async", "slot_name", "node_id", "data", "callback", "dynamic_respawn"), &SaveServer::_finish_load_async);

	ClassDB::bind_method(D_METHOD("amend_save", "root", "slot_name"), &SaveServer::amend_save);

	ClassDB::bind_method(D_METHOD("register_id", "id", "obj_id"), &SaveServer::register_id);
	ClassDB::bind_method(D_METHOD("unregister_id", "id"), &SaveServer::unregister_id);
	ClassDB::bind_method(D_METHOD("get_object_by_id", "id"), &SaveServer::get_object_by_id);
	ClassDB::bind_method(D_METHOD("stage_change", "obj_id", "tag"), &SaveServer::stage_change, DEFVAL(SNAME("general")));
	ClassDB::bind_method(D_METHOD("stage_deletion", "root_context", "node"), &SaveServer::stage_deletion);
	ClassDB::bind_method(D_METHOD("clear_staged"), &SaveServer::clear_staged);

	ClassDB::bind_method(D_METHOD("register_migration", "from", "to", "callback"), &SaveServer::register_migration);

	ClassDB::bind_method(D_METHOD("set_save_format", "format"), &SaveServer::set_save_format);
	ClassDB::bind_method(D_METHOD("get_save_format"), &SaveServer::get_save_format);

	ClassDB::bind_method(D_METHOD("set_encryption_key", "key"), &SaveServer::set_encryption_key);
	ClassDB::bind_method(D_METHOD("get_encryption_key"), &SaveServer::get_encryption_key);

	ClassDB::bind_method(D_METHOD("set_compression_enabled", "enabled"), &SaveServer::set_compression_enabled);
	ClassDB::bind_method(D_METHOD("is_compression_enabled"), &SaveServer::is_compression_enabled);

	ClassDB::bind_method(D_METHOD("set_backup_enabled", "enabled"), &SaveServer::set_backup_enabled);
	ClassDB::bind_method(D_METHOD("is_backup_enabled"), &SaveServer::is_backup_enabled);

	ClassDB::bind_method(D_METHOD("set_max_backups", "max"), &SaveServer::set_max_backups);
	ClassDB::bind_method(D_METHOD("get_max_backups"), &SaveServer::get_max_backups);

	ClassDB::bind_method(D_METHOD("set_integrity_check_level", "level"), &SaveServer::set_integrity_check_level);
	ClassDB::bind_method(D_METHOD("get_integrity_check_level"), &SaveServer::get_integrity_check_level);

	ClassDB::bind_method(D_METHOD("set_save_path", "path"), &SaveServer::set_save_path);
	ClassDB::bind_method(D_METHOD("get_save_path"), &SaveServer::get_save_path);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "save_format", PROPERTY_HINT_ENUM, "Text,Binary"), "set_save_format", "get_save_format");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "encryption_key"), "set_encryption_key", "get_encryption_key");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "compression_enabled"), "set_compression_enabled", "is_compression_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "backup_enabled"), "set_backup_enabled", "is_backup_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "max_backups"), "set_max_backups", "get_max_backups");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "integrity_check_level", PROPERTY_HINT_ENUM, "None,Signature,Strict"), "set_integrity_check_level", "get_integrity_check_level");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "save_path"), "set_save_path", "get_save_path");

	BIND_ENUM_CONSTANT(FORMAT_TEXT);
	BIND_ENUM_CONSTANT(FORMAT_BINARY);

	BIND_ENUM_CONSTANT(INTEGRITY_NONE);
	BIND_ENUM_CONSTANT(INTEGRITY_SIGNATURE);
	BIND_ENUM_CONSTANT(INTEGRITY_STRICT);

	BIND_ENUM_CONSTANT(SAVE_OK);
	BIND_ENUM_CONSTANT(SAVE_ERR_FILE_LOCKED);
	BIND_ENUM_CONSTANT(SAVE_ERR_ENCRYPTION);
	BIND_ENUM_CONSTANT(SAVE_ERR_DISK_FULL);
	BIND_ENUM_CONSTANT(SAVE_ERR_INVALID_DATA);
	BIND_ENUM_CONSTANT(SAVE_ERR_CHECKSUM_MISMATCH);
	BIND_ENUM_CONSTANT(SAVE_ERR_VERSION_MISMATCH);

	ADD_SIGNAL(MethodInfo("save_corrupted", PropertyInfo(Variant::STRING, "slot_name")));
	ADD_SIGNAL(MethodInfo("backup_restored", PropertyInfo(Variant::STRING, "slot_name"), PropertyInfo(Variant::STRING, "file_path")));
	ADD_SIGNAL(MethodInfo("save_successful", PropertyInfo(Variant::STRING, "slot_name")));
}

void SaveServer::_queue_save_task(const String &p_slot_name, Ref<Snapshot> p_snapshot, bool p_async) {
	SaveTask task;
	task.type = TASK_SAVE;
	task.slot_name = p_slot_name;
	task.snapshot = p_snapshot;
	task.format = current_format;
	task.encryption_key = encryption_key;
	task.compression_enabled = compression_enabled;

	if (p_async) {
		MutexLock lock(mutex);
		queue.push_back(task);
		semaphore.post();
	} else {
		_save_to_disk(task);
	}
}

void SaveServer::save_slot(const String &p_slot_name, const Dictionary &p_data, bool p_async, const Dictionary &p_metadata, Ref<Resource> p_thumbnail) {
	String slot_name = _sanitize_slot_name(p_slot_name);
	ERR_FAIL_COND_MSG(slot_name.is_empty(), "Slot name cannot be empty.");

	// Create Snapshot resource with reference counting (no deep copy needed)
	Ref<Snapshot> snapshot_res;
	snapshot_res.instantiate();
	snapshot_res->set_snapshot(p_data);
	snapshot_res->set_metadata(p_metadata);
	snapshot_res->set_thumbnail(p_thumbnail);

	// Inject version
	String project_version = GLOBAL_GET("application/config/version");
	snapshot_res->set_version(project_version);

	// Calculate and set checksum
	snapshot_res->set_checksum(_calculate_checksum(p_data));

	// Update base snapshot for amend saves if full save succeeds
	if (base_snapshot.is_null()) {
		base_snapshot.instantiate();
	}
	base_snapshot->set_snapshot(p_data); // Always update the base on full save
	current_slot_name = slot_name;

	_queue_save_task(slot_name, snapshot_res, p_async);
}

Dictionary SaveServer::load_slot(const String &p_slot_name) {
	String slot_name = _sanitize_slot_name(p_slot_name);
	return _load_from_disk(slot_name);
}

bool SaveServer::has_slot(const String &p_slot_name) const {
	String slot_name = _sanitize_slot_name(p_slot_name);
	String res_path = save_path.path_join(slot_name + ".tres");
	String data_path = save_path.path_join(slot_name + ".data");
	return FileAccess::exists(res_path) || FileAccess::exists(data_path);
}

void SaveServer::delete_slot(const String &p_slot_name) {
	String slot_name = _sanitize_slot_name(p_slot_name);
	String res_path = save_path.path_join(slot_name + ".tres");
	String data_path = save_path.path_join(slot_name + ".data");
	if (FileAccess::exists(res_path)) {
		DirAccess::remove_absolute(res_path);
		DirAccess::remove_absolute(res_path + ".bak");
	}
	if (FileAccess::exists(data_path)) {
		DirAccess::remove_absolute(data_path);
		DirAccess::remove_absolute(data_path + ".bak");
	}
}

void SaveServer::delete_snapshot(const String &p_snapshot_name) {
	String main_slot = _sanitize_slot_name(p_snapshot_name);
	String path = save_path.path_join(main_slot);
	String ext = get_save_format() == FORMAT_TEXT ? ".tres" : ".data";

	if (FileAccess::exists(path + ext)) {
		Ref<Resource> res = ResourceLoader::load(path + ext);
		Ref<Snapshot> snapshot_res = res;

		if (snapshot_res.is_valid()) {
			Dictionary tag_slots = snapshot_res->get_tag_slots();
			Array tags = tag_slots.keys();
			for (int i = 0; i < tags.size(); i++) {
				String tagged_slot = tag_slots[tags[i]];
				delete_slot(tagged_slot);
			}
		}
	}

	delete_slot(main_slot);
}

bool SaveServer::save_snapshot(Node *p_root, const String &p_slot_name, bool p_async, const TypedArray<StringName> &p_tags, const Dictionary &p_metadata, Ref<Resource> p_thumbnail) {
	ERR_FAIL_NULL_V(p_root, false);
	p_root->propagate_notification(Node::NOTIFICATION_SAVE_PREPARE);

	String main_slot = _sanitize_slot_name(p_slot_name);
	TypedArray<StringName> tags_to_process = p_tags;

	// 1. If tags are empty, we need to discover ALL tags in the tree
	Dictionary full_snapshot = _save_node_recursive(p_root, TypedArray<StringName>()); // Get everything

	if (tags_to_process.is_empty()) {
		// Discover tags from the root level of the snapshot (keys that don't start with '.')
		Array keys = full_snapshot.keys();
		for (int i = 0; i < keys.size(); i++) {
			String key = keys[i];
			if (!key.begins_with(".")) {
				tags_to_process.push_back(key);
			}
		}
	}

	// 2. Identify/Load Manifest
	Ref<Snapshot> manifest;
	String path = get_save_path().path_join(main_slot);
	String ext = get_save_format() == FORMAT_TEXT ? ".tres" : ".data";
	if (FileAccess::exists(path + ext)) {
		manifest = ResourceLoader::load(path + ext);
	}
	if (manifest.is_null()) {
		manifest.instantiate();
	}

	Dictionary tag_slots = manifest->get_tag_slots();
	bool manifest_changed = false;

	// 3. Process each tag into its own satellite
	for (int i = 0; i < tags_to_process.size(); i++) {
		StringName tag = tags_to_process[i];
		String satellite_slot = (tag == SNAME("general")) ? main_slot : main_slot + "_" + String(tag);

		Dictionary tag_data = _filter_snapshot_by_tag(full_snapshot, tag);

		if (tag == SNAME("general")) {
			// Main slot also carries the checksum and version
			manifest->set_snapshot(tag_data);
			manifest->set_version(GLOBAL_GET("application/config/version"));
			manifest->set_checksum(_calculate_checksum(tag_data));
		} else {
			save_slot(satellite_slot, tag_data, p_async);
			if (!tag_slots.has(tag)) {
				tag_slots[tag] = satellite_slot;
				manifest_changed = true;
			}
		}
	}

	// 4. Update Manifest metadata
	if (!p_metadata.is_empty()) {
		manifest->set_metadata(p_metadata);
		manifest_changed = true;
	}
	if (p_thumbnail.is_valid()) {
		manifest->set_thumbnail(p_thumbnail);
		manifest_changed = true;
	}

	if (manifest_changed || p_tags.is_empty()) {
		manifest->set_tag_slots(tag_slots);
		_queue_save_task(main_slot, manifest, p_async);
	}

	p_root->propagate_notification(Node::NOTIFICATION_SAVE_COMPLETED);
	return true;
}

void SaveServer::load_snapshot(Node *p_root, const String &p_slot_name, const Callable &p_callback, bool p_dynamic_respawn) {
	ERR_FAIL_NULL(p_root);

	String slot_name = _sanitize_slot_name(p_slot_name);

	SaveTask task;
	task.type = TASK_LOAD;
	task.slot_name = slot_name;
	task.target_node_id = p_root->get_instance_id();
	task.user_callback = p_callback;
	task.dynamic_respawn = p_dynamic_respawn;

	{
		MutexLock lock(mutex);
		queue.push_back(task);
	}
	semaphore.post();
}

void SaveServer::_finish_load_async(const String &p_slot_name, ObjectID p_node_id, const Dictionary &p_data, const Callable &p_callback, bool p_dynamic_respawn) {
	Object *obj = ObjectDB::get_instance(p_node_id);
	Node *root = Object::cast_to<Node>(obj);

	if (root) {
		if (!p_data.is_empty()) {
			// Update base snapshot context for amend saves
			if (base_snapshot.is_null()) {
				base_snapshot.instantiate();
			}
			base_snapshot->set_snapshot(p_data);
			current_slot_name = p_slot_name;

			root->propagate_notification(Node::NOTIFICATION_LOAD_STARTED);
			_load_node_recursive(root, p_data, p_dynamic_respawn);
			root->propagate_notification(Node::NOTIFICATION_LOAD_COMPLETED);
		} else {
			WARN_PRINT("SaveServer: Async load returned empty data (or file missing).");
		}
	}

	if (p_callback.is_valid()) {
		p_callback.call();
	}
}

Error SaveServer::_save_to_disk(const SaveTask &p_task) {
#ifdef DEBUG_ENABLED
	uint64_t start_time = OS::get_singleton()->get_ticks_msec();
#endif

	// Determine extension and format based on settings
	SaveFormat format = p_task.format;
	String ext = (format == FORMAT_TEXT) ? ".tres" : ".data";
	String full_path = save_path.path_join(p_task.slot_name + ext);
	String temp_path = full_path + ".tmp";

	// Create directory if it doesn't exist
	Ref<DirAccess> da = DirAccess::create(DirAccess::ACCESS_USERDATA);
	if (!da->dir_exists(save_path)) {
		da->make_dir_recursive(save_path);
	}

	// Snapshot is already prepared in save_slot()
	Ref<Snapshot> snapshot_res = p_task.snapshot;
	ERR_FAIL_COND_V(snapshot_res.is_null(), ERR_INVALID_DATA);

	Error err = OK;

	if (format == FORMAT_TEXT) {
		// Write to temp file first
		err = ResourceSaver::save(snapshot_res, temp_path);
	} else {
		// Individual task settings for thread safety
		String key = p_task.encryption_key;
		bool compress = p_task.compression_enabled;

		// Adaptive compression: only compress if data is large enough
		// Small files (<4KB) often get LARGER when compressed
		int estimated_size = snapshot_res->get_snapshot().size() * 100; // Rough estimate
		bool should_compress = compress && (estimated_size > 4096);

		Ref<FileAccess> f;
		if (!key.is_empty()) {
			f = FileAccess::open_encrypted_pass(temp_path, FileAccess::WRITE, key);
		} else if (should_compress) {
			f = FileAccess::open_compressed(temp_path, FileAccess::WRITE, FileAccess::COMPRESSION_ZSTD);
		} else {
			f = FileAccess::open(temp_path, FileAccess::WRITE);
		}

		ERR_FAIL_COND_V_MSG(f.is_null(), ERR_FILE_CANT_OPEN, "Cannot open save file for writing: " + temp_path);
		f->store_var(snapshot_res, true); // Serialize full Snapshot object
		f->close();
	}

	if (err != OK) {
		return err;
	}

	// Create backup of existing file before overwriting
	_create_backup(p_task.slot_name);

	// Atomic rename: rename temp to final
	// This will overwrite the existing file atomically on most filesystems
	da->rename(temp_path, full_path);

	// Notify success
	call_deferred("emit_signal", "save_successful", p_task.slot_name);

#ifdef DEBUG_ENABLED
	uint64_t end_time = OS::get_singleton()->get_ticks_msec();
	uint64_t file_size = FileAccess::get_file_as_bytes(full_path).size();
	print_line(vformat("SaveServer: Saved '%s' in %d ms, size: %d KB",
			p_task.slot_name, end_time - start_time, file_size / 1024));
#endif

	return OK;
}

void SaveServer::_merge_dictionaries_recursive(Dictionary &p_target, const Dictionary &p_source) {
	Array keys = p_source.keys();
	for (int i = 0; i < keys.size(); i++) {
		Variant key = keys[i];
		if (p_target.has(key) && p_target[key].get_type() == Variant::DICTIONARY && p_source[key].get_type() == Variant::DICTIONARY) {
			Dictionary target_sub = p_target[key];
			_merge_dictionaries_recursive(target_sub, p_source[key]);
			p_target[key] = target_sub;
		} else {
			p_target[key] = p_source[key];
		}
	}
}

Dictionary SaveServer::_load_from_disk(const String &p_slot_name) {
	Ref<Snapshot> snapshot_res = _read_snapshot_from_disk(p_slot_name);

	// Restore from backup if needed
	if (snapshot_res.is_null() && backup_enabled) {
		// 1. Try new timestamped backups
		String backup_path = _get_latest_backup(p_slot_name);
		if (!backup_path.is_empty()) {
			if (backup_path.ends_with(".tres")) {
				snapshot_res = ResourceLoader::load(backup_path);
			} else if (backup_path.ends_with(".data")) {
				String key = GLOBAL_GET("application/persistence/encryption_key");
				Ref<FileAccess> f;
				if (!key.is_empty()) {
					f = FileAccess::open_encrypted_pass(backup_path, FileAccess::READ, key);
				} else {
					f = FileAccess::open_compressed(backup_path, FileAccess::READ, FileAccess::COMPRESSION_ZSTD);
					if (f.is_null()) {
						f = FileAccess::open(backup_path, FileAccess::READ);
					}
				}

				if (f.is_valid()) {
					Variant v = f->get_var(true);
					f->close();
					snapshot_res = v;
				}
			}

			if (snapshot_res.is_valid()) {
				print_line("SaveServer: Main save corrupted or missing, restored from backup: " + backup_path);
				call_deferred("emit_signal", "save_corrupted", p_slot_name);
				call_deferred("emit_signal", "backup_restored", p_slot_name, backup_path);
			}
		}

		// 2. Fallback to legacy backup (.bak)
		if (snapshot_res.is_null()) {
			snapshot_res = _read_snapshot_from_disk(p_slot_name + ".bak");
			if (snapshot_res.is_valid()) {
				print_line("SaveServer: Main save corrupted or missing, restored from legacy backup: " + p_slot_name + ".bak");
				call_deferred("emit_signal", "save_corrupted", p_slot_name);
				call_deferred("emit_signal", "backup_restored", p_slot_name, p_slot_name + ".bak");
			}
		}
	}

	if (snapshot_res.is_null()) {
		return Dictionary();
	}

	// Integrity validation
	if (integrity_level >= INTEGRITY_SIGNATURE) {
		String calculated = _calculate_checksum(snapshot_res->get_snapshot());
		if (calculated != snapshot_res->get_checksum()) {
			WARN_PRINT("SaveServer: Integrity check failed (Signature mismatch) for slot: " + p_slot_name);
			if (integrity_level == INTEGRITY_STRICT) {
				ERR_PRINT("SaveServer: STRICT level active, rejecting save.");
				return Dictionary();
			}
		}
	}

	// 3. Versioning and Migrations
	_apply_migrations(snapshot_res);

	Dictionary full_data = snapshot_res->get_snapshot();
	Dictionary tag_slots = snapshot_res->get_tag_slots();

	// 4. Merge satellite slots if present
	if (!tag_slots.is_empty()) {
		Array tags = tag_slots.keys();
		for (int i = 0; i < tags.size(); i++) {
			String satellite_slot = tag_slots[tags[i]];
			// Recursive call to support nested manifests or satellite integrity checks
			Dictionary satellite_data = _load_from_disk(satellite_slot);
			if (!satellite_data.is_empty()) {
				_merge_dictionaries_recursive(full_data, satellite_data);
			}
		}
	}

	return full_data;
}

Ref<Snapshot> SaveServer::_read_snapshot_from_disk(const String &p_slot_name) {
	// Try text format first (.tres)
	String res_path = save_path.path_join(p_slot_name);
	if (!res_path.ends_with(".tres") && !res_path.ends_with(".data") && !res_path.ends_with(".bak")) {
		res_path += ".tres";
	}

	if (FileAccess::exists(res_path)) {
		Ref<Snapshot> res = ResourceLoader::load(res_path);
		if (res.is_valid()) {
			return res;
		}
	}

	// Try binary format if not already tried (.data)
	String data_path = save_path.path_join(p_slot_name);
	if (!data_path.ends_with(".data") && !data_path.ends_with(".bak")) {
		data_path += ".data";
	}

	if (FileAccess::exists(data_path)) {
		String key = GLOBAL_GET("application/persistence/encryption_key");
		Ref<FileAccess> f;
		if (!key.is_empty()) {
			f = FileAccess::open_encrypted_pass(data_path, FileAccess::READ, key);
		} else {
			f = FileAccess::open_compressed(data_path, FileAccess::READ, FileAccess::COMPRESSION_ZSTD);
			if (f.is_null()) {
				f = FileAccess::open(data_path, FileAccess::READ);
			}
		}

		if (f.is_valid()) {
			Variant v = f->get_var(true);
			f->close();
			Ref<Snapshot> res = v;
			if (res.is_valid()) {
				return res;
			}
		}
	}

	return Ref<Snapshot>();
}

String SaveServer::_calculate_checksum(const Dictionary &p_data) {
	// Use MD5 streaming hash to avoid full serialization
	// For integrity checking (not cryptographic security), MD5 is sufficient and much faster
	CryptoCore::MD5Context ctx;
	ctx.start();

	// Hash dictionary keys and values incrementally
	Array keys = p_data.keys();
	for (int i = 0; i < keys.size(); i++) {
		String key_str = String(keys[i]);
		ctx.update((const unsigned char *)key_str.utf8().get_data(), key_str.utf8().length());

		Variant value = p_data[keys[i]];
		String value_str = value.stringify();
		ctx.update((const unsigned char *)value_str.utf8().get_data(), value_str.utf8().length());
	}

	unsigned char hash[16];
	ctx.finish(hash);

	return String::hex_encode_buffer(hash, 16);
}

void SaveServer::register_migration(const String &p_from, const String &p_to, const Callable &p_callback) {
	Migration m;
	m.from = p_from;
	m.to = p_to;
	m.callback = p_callback;
	migrations.push_back(m);
}

void SaveServer::_apply_migrations(Ref<Snapshot> p_snapshot) {
	String current_ver = p_snapshot->get_version();
	String target_ver = GLOBAL_GET("application/config/version");

	if (current_ver == target_ver) {
		return;
	}

	print_line(vformat("SaveServer: Migrating save from %s to %s", current_ver, target_ver));

	// Track visited versions to detect cycles
	HashSet<String> visited_versions;
	visited_versions.insert(current_ver);

	bool changed = true;
	int max_iterations = 100; // Safety limit
	int iteration = 0;

	while (changed && current_ver != target_ver && iteration < max_iterations) {
		changed = false;
		iteration++;

		for (const Migration &m : migrations) {
			if (m.from == current_ver) {
				// Check for cycle
				if (visited_versions.has(m.to)) {
					ERR_PRINT(vformat("SaveServer: Migration cycle detected! %s -> %s creates a loop.", m.from, m.to));
					return;
				}

				Dictionary data = p_snapshot->get_snapshot();
				const Variant v_data = data;
				const Variant *args[1] = { &v_data };
				Variant ret;
				Callable::CallError ce;
				m.callback.callp(args, 1, ret, ce);

				if (ce.error == Callable::CallError::CALL_OK) {
					current_ver = m.to;
					visited_versions.insert(current_ver);
					p_snapshot->set_version(current_ver);
					changed = true;
					break;
				}
			}
		}
	}

	if (iteration >= max_iterations) {
		ERR_PRINT("SaveServer: Migration exceeded maximum iterations. Possible infinite loop.");
	}
}

void SaveServer::_save_thread_func(void *p_userdata) {
	SaveServer *ss = (SaveServer *)p_userdata;
	while (!ss->exit_thread.is_set()) {
		ss->semaphore.wait();
		if (ss->exit_thread.is_set()) {
			break;
		}
		ss->_process_queue();
	}
}

void SaveServer::_process_queue() {
	while (true) {
		SaveTask task;
		{
			MutexLock lock(mutex);
			if (queue.is_empty()) {
				break;
			}
			task = queue.front()->get();
			queue.pop_front();
		}

		if (task.type == TASK_SAVE) {
			_save_to_disk(task);
		} else if (task.type == TASK_LOAD) {
			Dictionary data = _load_from_disk(task.slot_name);
			// Dispatch back to main thread
			call_deferred("_finish_load_async", task.slot_name, task.target_node_id, data, task.user_callback, task.dynamic_respawn);
		}
	}
}

void SaveServer::set_save_format(SaveFormat p_format) {
	current_format = p_format;
}

SaveServer::SaveFormat SaveServer::get_save_format() const {
	return current_format;
}

void SaveServer::set_encryption_key(const String &p_key) {
	encryption_key = p_key;
}

String SaveServer::get_encryption_key() const {
	return encryption_key;
}

void SaveServer::set_compression_enabled(bool p_enabled) {
	compression_enabled = p_enabled;
}

bool SaveServer::is_compression_enabled() const {
	return compression_enabled;
}

void SaveServer::set_save_path(const String &p_path) {
	save_path = p_path;
}

String SaveServer::get_save_path() const {
	return save_path;
}

void SaveServer::set_backup_enabled(bool p_enabled) {
	backup_enabled = p_enabled;
}

bool SaveServer::is_backup_enabled() const {
	return backup_enabled;
}

void SaveServer::set_integrity_check_level(IntegrityCheckLevel p_level) {
	integrity_level = p_level;
}

SaveServer::IntegrityCheckLevel SaveServer::get_integrity_check_level() const {
	return integrity_level;
}

void SaveServer::register_id(const StringName &p_id, ObjectID p_obj) {
	MutexLock lock(staged_mutex);
	if (id_registry.has(p_id)) {
		WARN_PRINT(vformat("SaveServer: Overwriting persistence ID '%s'. Ensure IDs are unique.", p_id));
	}
	id_registry[p_id] = p_obj;
}

void SaveServer::unregister_id(const StringName &p_id) {
	MutexLock lock(staged_mutex);
	id_registry.erase(p_id);
}

Object *SaveServer::get_object_by_id(const StringName &p_id) const {
	MutexLock lock(staged_mutex);
	if (id_registry.has(p_id)) {
		return ObjectDB::get_instance(id_registry[p_id]);
	}
	return nullptr;
}

void SaveServer::stage_change(ObjectID p_obj, const StringName &p_tag) {
	MutexLock lock(staged_mutex);
	staged_objects[p_obj].insert(p_tag);
}

void SaveServer::stage_deletion(Node *p_root_context, Node *p_node) {
	ERR_FAIL_NULL(p_root_context);
	ERR_FAIL_NULL(p_node);
	ERR_FAIL_COND(!p_root_context->is_ancestor_of(p_node) && p_root_context != p_node);

	NodePath rel_path = p_root_context->get_path_to(p_node);
	MutexLock lock(staged_mutex);
	staged_deletions.insert(rel_path);
}

void SaveServer::clear_staged() {
	MutexLock lock(staged_mutex);
	staged_objects.clear();
	staged_deletions.clear();
}

bool SaveServer::_patch_snapshot_data(Dictionary &p_target, const NodePath &p_relative_path, const Dictionary &p_new_data) {
	Dictionary current = p_target;

	// If path is empty (root node itself changed), handle directly
	if (p_relative_path.get_name_count() == 0) {
		// Store children before overwrite
		Dictionary children;
		if (current.has(".children")) {
			children = current[".children"];
		}

		// Apply new data (overwrite properties)
		current.merge(p_new_data, true);

		// Restore children map (crucial step!)
		if (!children.is_empty()) {
			current[".children"] = children;
		}
		return true;
	}

	// Navigate hierarchy
	for (int i = 0; i < p_relative_path.get_name_count(); i++) {
		StringName segment = p_relative_path.get_name(i);

		if (!current.has(".children")) {
			// Path broken (parent missing in snapshot). Cannot patch.
			return false;
		}

		Dictionary children = current[".children"];
		if (!children.has(segment)) {
			// Child missing in snapshot. Cannot patch.
			return false;
		}

		// Move down
		current = children[segment]; // Reference to inner dictionary
	}

	// 'current' is now the dictionary of the target node
	// We need to preserve its children
	Dictionary children;
	if (current.has(".children")) {
		children = current[".children"];
	}

	// Apply change
	current.merge(p_new_data, true);

	// Restore children
	if (!children.is_empty()) {
		current[".children"] = children;
	}

	return true;
}

bool SaveServer::_remove_node_from_snapshot(Dictionary &p_target, const NodePath &p_relative_path) {
	if (p_relative_path.is_empty()) {
		p_target.clear();
		return true;
	}

	Dictionary current = p_target;
	for (int i = 0; i < p_relative_path.get_name_count(); i++) {
		StringName name = p_relative_path.get_name(i);

		if (!current.has(".children")) {
			return false;
		}

		Dictionary children = current[".children"];
		if (!children.has(name)) {
			return false;
		}

		if (i == p_relative_path.get_name_count() - 1) {
			// Found the parent, remove the child
			children.erase(name);

			// Cleanup parent if no children left
			if (children.is_empty()) {
				current.erase(".children");
			}
			return true;
		}

		current = children[name];
	}

	return false;
}

bool SaveServer::amend_save(Node *p_root, const String &p_slot_name) {
	ERR_FAIL_NULL_V(p_root, false);

	String main_slot = _sanitize_slot_name(p_slot_name);

	// Context check - If we switched slots, we need a full save/load cycle to establish a new base
	if (current_slot_name != main_slot) {
		return save_snapshot(p_root, main_slot);
	}

	HashMap<StringName, HashSet<ObjectID>> dirty_tags;
	HashSet<NodePath> deletions;

	{
		MutexLock lock(staged_mutex);
		if (staged_objects.is_empty() && staged_deletions.is_empty()) {
			return true;
		}

		// Group objects by tag
		for (const KeyValue<ObjectID, HashSet<StringName>> &E : staged_objects) {
			for (const StringName &tag : E.value) {
				dirty_tags[tag].insert(E.key);
			}
		}
		deletions = staged_deletions;
		// We don't clear here yet, we wait to see if we successfully patch
	}

	bool data_modified = false;

	// 1. Handle Deletions (Affects the Main Slot/Manifest by default)
	if (!deletions.is_empty()) {
		Dictionary patched_main = _load_from_disk(main_slot);
		bool modified = false;
		for (const NodePath &path : deletions) {
			if (_remove_node_from_snapshot(patched_main, path)) {
				modified = true;
			}
		}
		if (modified) {
			save_slot(main_slot, patched_main, true);
			data_modified = true;
		}
	}

	// 2. Intelligent Tag Updates (Satellite patching)
	for (const KeyValue<StringName, HashSet<ObjectID>> &E : dirty_tags) {
		StringName tag = E.key;
		String target_slot = (tag == SNAME("general")) ? main_slot : main_slot + "_" + String(tag);

		Dictionary satellite_data = _load_from_disk(target_slot);
		bool satellite_modified = false;

		for (const ObjectID &id : E.value) {
			Object *obj = ObjectDB::get_instance(id);
			Node *node = Object::cast_to<Node>(obj);
			if (!node) {
				continue;
			}

			// Only process nodes that belong to the current root hierarchy
			if (!p_root->is_ancestor_of(node) && p_root != node) {
				continue;
			}

			NodePath rel_path = p_root->get_path_to(node);

			// Get ONLY the data for this specific tag
			TypedArray<StringName> filter_tags;
			filter_tags.push_back(tag);
			Dictionary node_tag_data = node->get_persistent_properties(filter_tags);

			// get_persistent_properties returns grouped by tag, we need the inner dict
			Dictionary inner_data;
			if (node_tag_data.has(tag)) {
				inner_data = node_tag_data[tag];

				// Add identity markers
				StringName pid = node->get_persistence_id();
				if (!pid.is_empty()) {
					inner_data[".id"] = pid;
				}
			}

			if (!inner_data.is_empty()) {
				if (_patch_snapshot_data(satellite_data, rel_path, inner_data)) {
					satellite_modified = true;
				}
			}
		}

		if (satellite_modified) {
			save_slot(target_slot, satellite_data, true);
			data_modified = true;
		}
	}

	if (data_modified) {
		clear_staged();
	}

	return true;
}

SaveServer::SaveServer() {
	singleton = this;
	exit_thread.clear();

	// Load configuration
	backup_enabled = GLOBAL_GET("application/persistence/backup_enabled");
	max_backups = GLOBAL_GET("application/persistence/max_backups");
	int integrity_val = GLOBAL_GET("application/persistence/integrity_check_level");
	integrity_level = (IntegrityCheckLevel)integrity_val;
	save_path = GLOBAL_GET("application/persistence/save_path");

	// Auto-generate project-specific encryption key if empty (Editor-only)
	if (Engine::get_singleton()->is_editor_hint()) {
		String key = GLOBAL_GET("application/persistence/encryption_key");
		if (key.is_empty()) {
			// Generate a 32-char hex string based on time and random
			String allowed = "abcdef0123456789";
			String new_key = "";
			for (int i = 0; i < 32; i++) {
				int r = Math::rand() % allowed.length();
				new_key += String::chr(allowed[r]);
			}
			ProjectSettings::get_singleton()->set_setting("application/persistence/encryption_key", new_key);
			ProjectSettings::get_singleton()->save();
			print_line("Persistence System: Auto-generated unique encryption key for the project.");
		}
	}

	save_thread.start(_save_thread_func, this);
}

SaveServer::~SaveServer() {
	exit_thread.set();
	semaphore.post();
	if (save_thread.is_started()) {
		save_thread.wait_to_finish();
	}

	// Flush remaining save tasks to ensure no data loss on shutdown
	if (!queue.is_empty()) {
#ifdef DEBUG_ENABLED
		print_line("SaveServer: Flushing persistent queue on shutdown...");
#endif
		while (true) {
			SaveTask task;
			{
				MutexLock lock(mutex);
				if (queue.is_empty()) {
					break;
				}
				task = queue.front()->get();
				queue.pop_front();
			}

			// Only process saves, ignore loads during shutdown
			if (task.type == TASK_SAVE) {
				_save_to_disk(task);
			}
		}
	}

	id_registry.clear();
	staged_objects.clear();
	base_snapshot.unref();

	singleton = nullptr;
}

Dictionary SaveServer::_filter_snapshot_by_tag(const Dictionary &p_full_snapshot, const StringName &p_tag) {
	Dictionary filtered;

	// 1. Extract tag properties
	if (p_full_snapshot.has(p_tag)) {
		filtered[p_tag] = p_full_snapshot[p_tag].duplicate();
	}

	// 2. Copy Identity metadata
	if (p_full_snapshot.has(".id")) {
		filtered[".id"] = p_full_snapshot[".id"];
	}
	if (p_full_snapshot.has(".scene")) {
		filtered[".scene"] = p_full_snapshot[".scene"];
	}

	// 3. Recursively filter children
	if (p_full_snapshot.has(".children")) {
		Dictionary children = p_full_snapshot[".children"];
		Dictionary filtered_children;
		Array keys = children.keys();
		for (int i = 0; i < keys.size(); i++) {
			Dictionary child_filtered = _filter_snapshot_by_tag(children[keys[i]], p_tag);
			// Only add child if it has data for the tag or has children with tag data
			if (child_filtered.has(p_tag) || child_filtered.has(".children")) {
				filtered_children[keys[i]] = child_filtered;
			}
		}
		if (!filtered_children.is_empty()) {
			filtered[".children"] = filtered_children;
		}
	}

	return filtered;
}

Dictionary SaveServer::_save_node_recursive(Node *p_node, const TypedArray<StringName> &p_tags) {
	if (!p_node) {
		return Dictionary();
	}

	// Check policy
	if (p_node->get_save_policy() == Node::SAVE_POLICY_NEVER) {
		return Dictionary();
	}

	// Save this node
	Dictionary snapshot = p_node->get_persistent_properties(p_tags);

	// Add ID if present, or auto-generate for dynamic instances
	StringName pid = p_node->get_persistence_id();
	if (pid.is_empty() && !p_node->get_scene_file_path().is_empty()) {
		// Auto-generate deterministic-ish ID for dynamic spawns to ensure they have an identity
		// Use memory address + ticks + random to be unique in runtime
		pid = p_node->get_scene_file_path().get_file().get_basename() + "_" + String::num_uint64(p_node->get_instance_id()) + "_" + String::num_int64(Math::rand());
		p_node->set_persistence_id(pid);
	}

	if (!pid.is_empty()) {
		snapshot[".id"] = pid;
	}

	// Save scene path for dynamic spawning
	if (!p_node->get_scene_file_path().is_empty()) {
		snapshot[".scene"] = p_node->get_scene_file_path();
	}

	// Recursively save children
	Dictionary children_snapshots;
	int child_count = p_node->get_child_count(false); // Exclude internal nodes
	for (int i = 0; i < child_count; i++) {
		Node *child = p_node->get_child(i, false);
		Dictionary child_data = _save_node_recursive(child, p_tags);
		if (!child_data.is_empty()) {
			children_snapshots[child->get_name()] = child_data;
		}
	}

	if (!children_snapshots.is_empty()) {
		snapshot[".children"] = children_snapshots;
	}

	return snapshot;
}

void SaveServer::_load_node_recursive(Node *p_node, const Dictionary &p_data, bool p_dynamic_respawn) {
	if (!p_node) {
		return;
	}

	// Restore this node
	p_node->set_persistent_properties(p_data);

	// Restore children
	if (p_data.has(".children")) {
		Dictionary children_data = p_data[".children"];

		// Orphan Cleanup: Remove nodes that exist in the scene but are missing from the snapshot
		if (p_dynamic_respawn) {
			List<Node *> to_remove;
			for (int i = 0; i < p_node->get_child_count(); i++) {
				Node *child = p_node->get_child(i);
				StringName child_name = child->get_name();

				// If the node is persistent but not in the save, it was destroyed/removed
				if (!children_data.has(child_name)) {
					// Check if node is persistent-enabled (has ID, explicit save policy ALWAYS, or has persistent properties)
					bool is_persistent = !child->get_persistence_id().is_empty() ||
							child->get_save_policy() == Node::SAVE_POLICY_ALWAYS ||
							!child->get_persistent_properties().is_empty();

					if (is_persistent) {
						to_remove.push_back(child);
					}
				}
			}

			for (Node *child : to_remove) {
#ifdef DEBUG_ENABLED
				print_line(vformat("SaveServer: Orphan Cleanup - Removing destroyed node '%s'", child->get_name()));
#endif
				child->queue_free();
			}
		}

		Array child_names = children_data.keys();

		for (int i = 0; i < child_names.size(); i++) {
			StringName child_name = child_names[i];
			Dictionary child_snapshot = children_data[child_name];

			// Attempt to find child by name
			Node *child = p_node->get_node_or_null(NodePath(child_name));

			// Dynamic Instantiation Logic
			if (!child && child_snapshot.has(".scene") && p_dynamic_respawn) {
				String scene_path = child_snapshot[".scene"];
				Ref<PackedScene> scene = ResourceLoader::load(scene_path);

				if (scene.is_valid()) {
					Node *instance = scene->instantiate();
					if (instance) {
						instance->set_name(child_name);
						p_node->add_child(instance, true); // Force readable name
						child = instance;
#ifdef DEBUG_ENABLED
						print_line(vformat("SaveServer: Dynamically spawned node '%s' from '%s'", child_name, scene_path));
#endif
					}
				} else {
					ERR_PRINT(vformat("SaveServer: Failed to load scene '%s' for dynamic spawn '%s'", scene_path, child_name));
				}
			}

			if (child) {
				_load_node_recursive(child, child_snapshot, p_dynamic_respawn);
			} else {
				// Node missing and no scene path to respawn. Data orphaned.
				// WARN_PRINT(vformat("SaveServer: Node '%s' not found and no scene path saved. Skipping load for this branch.", child_name));
			}
		}
	}
}

void SaveServer::_create_backup(const String &p_slot_name) {
	if (!backup_enabled) {
		return;
	}

	// Identify the file to back up (could be text or binary, check existence)
	String base_path = save_path.path_join(p_slot_name);
	String src_path = "";
	String ext = "";

	if (FileAccess::exists(base_path + ".data")) {
		src_path = base_path + ".data";
		ext = ".data";
	} else if (FileAccess::exists(base_path + ".tres")) {
		src_path = base_path + ".tres";
		ext = ".tres";
	} else {
		return; // Nothing to backup
	}

	// Create backups directory
	Ref<DirAccess> da = DirAccess::create(DirAccess::ACCESS_USERDATA);
	String backup_dir = save_path.path_join("backups");
	if (!da->dir_exists(backup_dir)) {
		da->make_dir(backup_dir);
	}

	// Generate timestamped filename
	// Format: slot_YYYY-MM-DD_HH-mm-ss.ext
	String timestamp = Time::get_singleton()->get_datetime_string_from_system().replace(":", "-");
	String dest_path = backup_dir.path_join(p_slot_name + "_" + timestamp + ext);

	// Copy the file
	da->copy(src_path, dest_path);

	// Cleanup old backups
	_prune_backups(p_slot_name);
}

void SaveServer::_prune_backups(const String &p_slot_name) {
	String backup_dir = save_path.path_join("backups");
	Ref<DirAccess> da = DirAccess::open(backup_dir);
	if (da.is_null()) {
		return;
	}

	List<String> backups;
	da->list_dir_begin();
	String f = da->get_next();
	while (!f.is_empty()) {
		if (!da->current_is_dir() && f.begins_with(p_slot_name + "_")) {
			backups.push_back(f);
		}
		f = da->get_next();
	}
	da->list_dir_end();

	backups.sort(); // Datetime string sort works correctly

	// Keep only max_backups most recent
	while (backups.size() > max_backups) {
		String to_delete = backup_dir.path_join(backups.front()->get());
		da->remove(to_delete);
		backups.pop_front();
	}
}

String SaveServer::_sanitize_slot_name(const String &p_slot_name) const {
	String sanitized = p_slot_name.replace("..", ""); // Prevent path traversal
	sanitized = sanitized.replace("\\", "/");

	// Allow slashes for subfolders, but ensure no illegal chars
	// Windows restricted chars: < > : " / \ | ? *
	// But / is allowed for subfolders in Godot user://
	// So we should just be careful with : * ? " < > |

	// A simple approach is to rely on Godot's FileAccess::open failure for invalid chars,
	// but removing .. is crucial.
	// Also strip leading/trailing slashes/spaces.
	sanitized = sanitized.strip_edges();
	if (sanitized.begins_with("/")) {
		sanitized = sanitized.substr(1);
	}

	// Remove invalid chars for file names (conservatively)
	// On Windows : is driver separator, but we are inside user://
	// But let's replace : with _ just in case
	sanitized = sanitized.replace(":", "_");
	sanitized = sanitized.replace("*", "_");
	sanitized = sanitized.replace("?", "_");
	sanitized = sanitized.replace("\"", "_");
	sanitized = sanitized.replace("<", "_");
	sanitized = sanitized.replace(">", "_");
	sanitized = sanitized.replace("|", "_");

	return sanitized;
}

void SaveServer::set_max_backups(int p_max) {
	max_backups = MAX(0, p_max);
}

int SaveServer::get_max_backups() const {
	return max_backups;
}

String SaveServer::_get_latest_backup(const String &p_slot_name) {
	String backup_dir = save_path.path_join("backups");
	Ref<DirAccess> da = DirAccess::open(backup_dir);
	if (da.is_null()) {
		return "";
	}

	List<String> backups;
	da->list_dir_begin();
	String f = da->get_next();
	while (!f.is_empty()) {
		if (!da->current_is_dir() && f.begins_with(p_slot_name + "_")) {
			backups.push_back(f);
		}
		f = da->get_next();
	}
	da->list_dir_end();

	if (backups.is_empty()) {
		return "";
	}

	backups.sort();
	return backup_dir.path_join(backups.back()->get());
}
