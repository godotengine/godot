/**************************************************************************/
/*  save_server.h                                                         */
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

#include "core/io/resource.h"
#include "core/object/class_db.h"
#include "core/os/mutex.h"
#include "core/os/semaphore.h"
#include "core/os/thread.h"
#include "core/templates/list.h"
#include "core/variant/dictionary.h"
#include "core/variant/typed_array.h"

class Node;
class Snapshot;

class SaveServer : public Object {
	GDCLASS(SaveServer, Object);

public:
	enum SaveFormat {
		FORMAT_TEXT,
		FORMAT_BINARY
	};

	enum SaveResult {
		SAVE_OK,
		SAVE_ERR_FILE_LOCKED,
		SAVE_ERR_ENCRYPTION,
		SAVE_ERR_DISK_FULL,
		SAVE_ERR_INVALID_DATA,
		SAVE_ERR_CHECKSUM_MISMATCH,
		SAVE_ERR_VERSION_MISMATCH
	};

	enum IntegrityCheckLevel {
		INTEGRITY_NONE,
		INTEGRITY_SIGNATURE,
		INTEGRITY_STRICT
	};

private:
	enum TaskType {
		TASK_SAVE,
		TASK_LOAD
	};

	struct SaveTask {
		TaskType type = TASK_SAVE;
		String slot_name;
		Ref<Snapshot> snapshot; // Use Ref<> for reference counting instead of deep copy
		Dictionary metadata;
		Ref<Resource> thumbnail;
		SaveFormat format;
		String encryption_key;
		bool compression_enabled;

		// For Async Load
		ObjectID target_node_id;
		Callable user_callback;
		bool dynamic_respawn = false;
	};

	static SaveServer *singleton;

	// Threading
	Thread save_thread;
	Mutex mutex;
	Semaphore semaphore;
	SafeFlag exit_thread;
	List<SaveTask> queue;
	mutable Mutex staged_mutex;

	// Configuration
	SaveFormat current_format = FORMAT_TEXT;
	String encryption_key;
	bool compression_enabled = true;
	bool backup_enabled = true;
	IntegrityCheckLevel integrity_level = INTEGRITY_SIGNATURE;

	// Modular/Amend Persistence
	Ref<Snapshot> base_snapshot;
	String current_slot_name; // Tracks the context of the base_snapshot
	HashMap<StringName, ObjectID> id_registry;
	HashMap<ObjectID, HashSet<StringName>> staged_objects;
	HashSet<NodePath> staged_deletions;

	String save_path = "user://saves/";

	struct Migration {
		String from;
		String to;
		Callable callback;
	};
	List<Migration> migrations;

	// ... threading ...
	static void _save_thread_func(void *p_userdata);
	void _process_queue();
	void _finish_load_async(const String &p_slot_name, ObjectID p_node_id, const Dictionary &p_data, const Callable &p_callback, bool p_dynamic_respawn);
	void _queue_save_task(const String &p_slot_name, Ref<Snapshot> p_snapshot, bool p_async);
	void _merge_dictionaries_recursive(Dictionary &p_target, const Dictionary &p_source);

	Error _save_to_disk(const SaveTask &p_task);
	Ref<Snapshot> _read_snapshot_from_disk(const String &p_slot_name);
	Dictionary _load_from_disk(const String &p_slot_name);

	String _calculate_checksum(const Dictionary &p_data);
	void _apply_migrations(Ref<Snapshot> p_snapshot);

	// Internal Recursive Logic
	Dictionary _save_node_recursive(Node *p_node, const TypedArray<StringName> &p_tags);
	Dictionary _filter_snapshot_by_tag(const Dictionary &p_full_snapshot, const StringName &p_tag);
	void _load_node_recursive(Node *p_node, const Dictionary &p_data, bool p_dynamic_respawn);
	bool _patch_snapshot_data(Dictionary &p_root_data, const NodePath &p_relative_path, const Dictionary &p_new_node_data);
	bool _remove_node_from_snapshot(Dictionary &p_root_data, const NodePath &p_relative_path);

protected:
	static void _bind_methods();

public:
	static SaveServer *get_singleton();
	static void register_settings();

	// Slot Management
	void save_slot(const String &p_slot_name, const Dictionary &p_data, bool p_async = true, const Dictionary &p_metadata = Dictionary(), Ref<Resource> p_thumbnail = Ref<Resource>());
	Dictionary load_slot(const String &p_slot_name);
	bool has_slot(const String &p_slot_name) const;
	void delete_slot(const String &p_slot_name);
	void delete_snapshot(const String &p_snapshot_name);

	// Snapshot API (Node-based)
	bool save_snapshot(Node *p_root, const String &p_slot_name, bool p_async = true, const TypedArray<StringName> &p_tags = TypedArray<StringName>(), const Dictionary &p_metadata = Dictionary(), Ref<Resource> p_thumbnail = Ref<Resource>());
	void load_snapshot(Node *p_root, const String &p_slot_name, const Callable &p_callback = Callable(), bool p_dynamic_respawn = false);
	bool has_snapshot(const String &p_slot_name) const { return has_slot(p_slot_name); }

	// Migrations
	void register_migration(const String &p_from, const String &p_to, const Callable &p_callback);

	// ID and Amend Management
	void register_id(const StringName &p_id, ObjectID p_obj);
	void unregister_id(const StringName &p_id);
	Object *get_object_by_id(const StringName &p_id) const;

	void stage_change(ObjectID p_obj, const StringName &p_tag = SNAME("general"));
	void stage_deletion(Node *p_root_context, Node *p_node);
	void clear_staged();

	bool amend_save(Node *p_root, const String &p_slot_name);

private:
	// V3.0: Backup & Security System
	void _create_backup(const String &p_slot_name);
	void _prune_backups(const String &p_slot_name);
	String _get_latest_backup(const String &p_slot_name);
	String _sanitize_slot_name(const String &p_slot_name) const;

	int max_backups = 2;

	// Configuration
	void set_save_format(SaveFormat p_format);
	SaveFormat get_save_format() const;

	void set_encryption_key(const String &p_key);
	String get_encryption_key() const;

	void set_compression_enabled(bool p_enabled);
	bool is_compression_enabled() const;

	void set_save_path(const String &p_path);
	String get_save_path() const;

	void set_backup_enabled(bool p_enabled);
	bool is_backup_enabled() const;

	void set_max_backups(int p_max);
	int get_max_backups() const;

	void set_integrity_check_level(IntegrityCheckLevel p_level);
	IntegrityCheckLevel get_integrity_check_level() const;

public:
	SaveServer();
	~SaveServer();
};

VARIANT_ENUM_CAST(SaveServer::SaveFormat);
VARIANT_ENUM_CAST(SaveServer::SaveResult);
VARIANT_ENUM_CAST(SaveServer::IntegrityCheckLevel);
