/**************************************************************************/
/*  async_pck_installer.cpp                                               */
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

#include "async_pck_installer.h"

#include "core/config/engine.h"
#include "core/io/resource_loader.h"
#include "core/os/os.h"

void AsyncPCKInstaller::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
		} break;

		case NOTIFICATION_READY: {
			if (autostart) {
				start();
			}
		} break;

		case NOTIFICATION_PROCESS: {
			update();
		} break;

		case NOTIFICATION_EXIT_TREE: {
			set_process(false);
		} break;
	}
}

void AsyncPCKInstaller::update() {
	PackedStringArray processed_file_paths = _get_processed_file_paths();

	InstallerStatus new_status = get_status();
	switch (new_status) {
		case INSTALLER_STATUS_IDLE:
		case INSTALLER_STATUS_LOADING: {
			// Do nothing.
		} break;

		case INSTALLER_STATUS_INSTALLED:
		case INSTALLER_STATUS_ERROR: {
			set_process(false);
			return;
		} break;

		case INSTALLER_STATUS_MAX: {
			set_process(false);
			ERR_FAIL();
		} break;
	}

	const static String KEY_FILES = "files";
	const static String KEY_STATUS = "status";
	const static String KEY_SIZE = "size";
	const static String KEY_PROGRESS = "progress";
	const static String KEY_PROGRESS_RATIO = "progress_ratio";
	const static String KEY_ERRORS = "errors";

	const static String STATUS_IDLE = "STATUS_IDLE";
	const static String STATUS_LOADING = "STATUS_LOADING";
	const static String STATUS_ERROR = "STATUS_ERROR";
	const static String STATUS_INSTALLED = "STATUS_INSTALLED";

	HashMap<String, Dictionary> files_status;

	// Utility lambdas.
	auto _l_get_status_enum_value = [](const String &l_status_value) -> InstallerStatus {
		if (l_status_value == STATUS_IDLE) {
			return INSTALLER_STATUS_IDLE;
		} else if (l_status_value == STATUS_LOADING) {
			return INSTALLER_STATUS_LOADING;
		} else if (l_status_value == STATUS_ERROR) {
			return INSTALLER_STATUS_ERROR;
		} else if (l_status_value == STATUS_INSTALLED) {
			return INSTALLER_STATUS_INSTALLED;
		}
		ERR_FAIL_V(INSTALLER_STATUS_ERROR);
	};

	auto _l_get_file_progress_dictionary = [&](const Dictionary &l_file_progress) -> Dictionary {
		Dictionary file_progress;
		InstallerStatus file_status = _l_get_status_enum_value(l_file_progress[KEY_STATUS]);

		file_progress[KEY_STATUS] = file_status;
		file_progress[KEY_SIZE] = l_file_progress[KEY_SIZE];
		file_progress[KEY_PROGRESS] = l_file_progress[KEY_PROGRESS];
		file_progress[KEY_PROGRESS_RATIO] = l_file_progress[KEY_PROGRESS_RATIO];

		if (file_status == INSTALLER_STATUS_ERROR) {
			file_progress[KEY_ERRORS] = l_file_progress[KEY_ERRORS];
		}

		return file_progress;
	};

	// Update status of each file.
	for (const KeyValue<String, InstallerStatus> &key_value : file_paths_status) {
		String file_path = key_value.key;

		Dictionary status = OS::get_singleton()->async_pck_install_file_get_status(file_path);
		Dictionary files = status[KEY_FILES];
		for (const KeyValue<Variant, Variant> &file_key_value : files) {
			if (files_status.has(file_key_value.key)) {
				continue;
			}
			files_status.insert(file_key_value.key, file_key_value.value);
		}

		InstallerStatus file_status = _l_get_status_enum_value(status[KEY_STATUS]);
		set_file_path_status(file_path, file_status);

		switch (file_status) {
			case INSTALLER_STATUS_IDLE: {
				// Do nothing.
			} break;

			case INSTALLER_STATUS_LOADING: {
				emit_signal(SIGNAL_FILE_PROGRESS, file_path, _l_get_file_progress_dictionary(status));
			} break;

			case INSTALLER_STATUS_INSTALLED: {
				emit_signal(SIGNAL_FILE_PROGRESS, file_path, _l_get_file_progress_dictionary(status));
				emit_signal(SIGNAL_FILE_INSTALLED, file_path);
			} break;

			case INSTALLER_STATUS_ERROR: {
				emit_signal(SIGNAL_FILE_ERROR, status[KEY_ERRORS]);
			} break;

			case INSTALLER_STATUS_MAX: {
				ERR_FAIL();
			} break;
		}
	}

	// Trigger signals based on the new status.
	new_status = get_status();

	switch (new_status) {
		case INSTALLER_STATUS_IDLE:
		case INSTALLER_STATUS_ERROR: {
			// Do nothing.
		} break;

		case INSTALLER_STATUS_LOADING:
		case INSTALLER_STATUS_INSTALLED: {
			uint64_t progress_total = 0;
			uint64_t size_total = 0;
			double progress_ratio = 0;

			for (const KeyValue<String, Dictionary> &key_value : files_status) {
				size_total += (uint64_t)key_value.value[KEY_SIZE];
				progress_total += (uint64_t)key_value.value[KEY_PROGRESS];
			}

			Dictionary files_progress;
			files_progress[KEY_SIZE] = size_total;
			files_progress[KEY_PROGRESS] = progress_total;
			if (size_total > 0) {
				progress_ratio = (double)progress_total / (double)size_total;
			}
			files_progress[KEY_PROGRESS_RATIO] = progress_ratio;

			emit_signal(SIGNAL_PROGRESS, files_progress);
		} break;

		case INSTALLER_STATUS_MAX: {
			ERR_FAIL();
		} break;
	}
}

void AsyncPCKInstaller::start() {
	if (Engine::get_singleton()->is_editor_hint()) {
		return;
	}
	if (started) {
		return;
	}
	started = true;

	PackedStringArray processed_file_paths = _get_processed_file_paths();

	// Check if the resources are already installed.
	for (const String &file_path : processed_file_paths) {
		if (OS::get_singleton()->async_pck_is_supported()) {
			if (!OS::get_singleton()->async_pck_is_file_installable(file_path)) {
				set_file_path_status(file_path, INSTALLER_STATUS_INSTALLED);
			}
		} else {
			if (ResourceLoader::exists(file_path)) {
				set_file_path_status(file_path, INSTALLER_STATUS_INSTALLED);
			} else {
				set_file_path_status(file_path, INSTALLER_STATUS_ERROR);
				emit_signal(SIGNAL_FILE_ERROR, file_path, Array({ vformat(R"*(File "%s" doesn't exist.)*", file_path) }));
			}
		}
	}

	for (const String &file_path : processed_file_paths) {
		if (file_paths_status.has(file_path) && file_paths_status[file_path] != INSTALLER_STATUS_IDLE) {
			continue;
		}

		Error err = OS::get_singleton()->async_pck_install_file(file_path);
		if (err == OK) {
			set_file_path_status(file_path, INSTALLER_STATUS_LOADING);
		} else {
			set_file_path_status(file_path, INSTALLER_STATUS_ERROR);
		}
	}

	set_process(true);
}

bool AsyncPCKInstaller::set_file_path_status(const String &p_path, InstallerStatus p_status) {
	ERR_FAIL_COND_V_MSG(!_get_processed_file_paths().has(ResourceUID::ensure_path(p_path)), false, vformat(R"*("%s" is not in `file_paths`.)*", p_path));

	InstallerStatus old_status = get_status();

	if (file_paths_status.has(p_path) && file_paths_status.get(p_path) == p_status) {
		// Nothing changed.
		return false;
	} else {
		file_paths_status.insert(p_path, p_status);
	}

	// Set the flags as dirty.
	status_dirty = true;
	install_needed_dirty = true;

	// Check for changed installer status.
	InstallerStatus new_status = get_status();
	if (old_status == new_status) {
		// The installer status didn't change.
		// We still return `true` here because the path status did change.
		return true;
	}

	// Installer state change side-effects.
	String status_name;
	switch (new_status) {
		case INSTALLER_STATUS_IDLE: {
			status_name = "INSTALLER_STATUS_IDLE";
		} break;
		case INSTALLER_STATUS_LOADING: {
			status_name = "INSTALLER_STATUS_LOADING";
		} break;
		case INSTALLER_STATUS_INSTALLED: {
			status_name = "INSTALLER_STATUS_INSTALLED";
		} break;
		case INSTALLER_STATUS_ERROR: {
			status_name = "INSTALLER_STATUS_ERROR";
		} break;
		case INSTALLER_STATUS_MAX: {
			status_name = "INSTALLER_STATUS_MAX";
		} break;
	}
	emit_signal(SIGNAL_STATUS_CHANGED);

	switch (new_status) {
		case INSTALLER_STATUS_IDLE: {
			set_process(started);
		} break;

		case INSTALLER_STATUS_LOADING: {
			set_process(true);
		} break;

		case INSTALLER_STATUS_INSTALLED:
		case INSTALLER_STATUS_ERROR: {
			set_process(false);
		} break;

		case INSTALLER_STATUS_MAX: {
			ERR_FAIL_V(true);
		} break;
	}

	// Return that the path status did change.
	return true;
}

void AsyncPCKInstaller::set_autostart(bool p_autostart) {
	autostart = p_autostart;
}

bool AsyncPCKInstaller::get_autostart() const {
	return autostart;
}

void AsyncPCKInstaller::set_file_paths(const PackedStringArray &p_file_paths) {
	ERR_MAIN_THREAD_GUARD;

	if (file_paths == p_file_paths) {
		return;
	}

	PackedStringArray before_processed_file_paths;
	HashSet<String> removed_paths;

	before_processed_file_paths = _get_processed_file_paths();

	file_paths = p_file_paths;

	// Gather removed paths.
	for (const KeyValue<String, InstallerStatus> &key_value : file_paths_status) {
		if (file_paths.has(key_value.key)) {
			continue;
		}
		removed_paths.insert(key_value.key);
	}

	// Actually remove paths.
	for (const String &path_to_remove : removed_paths) {
		file_paths.erase(path_to_remove);
	}

	// Check if `file_paths` actually changed.
	PackedStringArray current_processed_file_paths = _get_processed_file_paths();
	if (current_processed_file_paths == before_processed_file_paths) {
		return;
	}

	// Emit `file_removed` signal.
	for (const String &file_path : before_processed_file_paths) {
		if (current_processed_file_paths.has(file_path)) {
			continue;
		}
		emit_signal(SIGNAL_FILE_REMOVED, file_path);
	}

	// Emit `file_added` signal.
	for (const String &file_path : current_processed_file_paths) {
		if (before_processed_file_paths.has(file_path)) {
			continue;
		}
		emit_signal(SIGNAL_FILE_ADDED, file_path);
	}

	if (!started) {
		return;
	}

	// Start new installing files.
	for (const String &file_path : current_processed_file_paths) {
		if (before_processed_file_paths.has(file_path)) {
			continue;
		}
		if (file_paths_status.has(file_path) && file_paths_status[file_path] != INSTALLER_STATUS_IDLE) {
			continue;
		}

		if (OS::get_singleton()->async_pck_is_supported()) {
			if (OS::get_singleton()->async_pck_is_file_installable(file_path)) {
				Error err = OS::get_singleton()->async_pck_install_file(file_path);
				if (err == OK) {
					set_file_path_status(file_path, INSTALLER_STATUS_LOADING);
				} else {
					set_file_path_status(file_path, INSTALLER_STATUS_ERROR);
				}
			} else {
				set_file_path_status(file_path, INSTALLER_STATUS_INSTALLED);
			}
		} else {
			if (ResourceLoader::exists(file_path)) {
				set_file_path_status(file_path, INSTALLER_STATUS_INSTALLED);
			} else {
				set_file_path_status(file_path, INSTALLER_STATUS_ERROR);
			}
		}
	}
}

PackedStringArray AsyncPCKInstaller::get_file_paths() const {
	return file_paths;
}

String AsyncPCKInstaller::_process_file_path(const String &p_path) const {
	String path = p_path.strip_edges();
	if (path.is_empty()) {
		return path;
	}
	return ResourceUID::ensure_path(path);
}

PackedStringArray AsyncPCKInstaller::_get_processed_file_paths() const {
	HashSet<String> processed_file_paths_set;
	for (const String &file_path : file_paths) {
		String processed_file_path = _process_file_path(file_path);
		if (processed_file_path.is_empty()) {
			continue;
		}
		processed_file_paths_set.insert(processed_file_path);
	}

	PackedStringArray processed_file_paths;
	for (const String &processed_file_path : processed_file_paths_set) {
		processed_file_paths.push_back(processed_file_path);
	}
	return processed_file_paths;
}

AsyncPCKInstaller::InstallerStatus AsyncPCKInstaller::get_status() const {
	if (!status_dirty) {
		return status_cached;
	}

	InstallerStatus status = INSTALLER_STATUS_IDLE;

	if (file_paths_status.is_empty()) {
		return INSTALLER_STATUS_IDLE;
	}

	for (const KeyValue<String, InstallerStatus> &key_value : file_paths_status) {
#define CASE_INSTALLER_STATUS_MAX           \
	case INSTALLER_STATUS_MAX: {            \
		ERR_FAIL_V(INSTALLER_STATUS_ERROR); \
	} break

		switch (status) {
			case INSTALLER_STATUS_IDLE: {
				switch (key_value.value) {
					case INSTALLER_STATUS_IDLE: {
						// Do nothing, the state is the same.
					} break;

					case INSTALLER_STATUS_LOADING:
					case INSTALLER_STATUS_INSTALLED: {
						status = key_value.value;
					} break;

					case INSTALLER_STATUS_ERROR: {
						return INSTALLER_STATUS_ERROR;
					} break;

						CASE_INSTALLER_STATUS_MAX;
				}
			} break;

			case INSTALLER_STATUS_LOADING: {
				switch (key_value.value) {
					case INSTALLER_STATUS_LOADING: {
						// Do nothing, the state is the same.
					} break;

					case INSTALLER_STATUS_IDLE: {
						// Do nothing, as `INSTALLER_STATUS_LOADING` > `INSTALLER_STATUS_IDLE`.
					} break;

					case INSTALLER_STATUS_INSTALLED: {
						// Do nothing, as the state is still loading even if there's
						// some files that are done.
					} break;

					case INSTALLER_STATUS_ERROR: {
						return INSTALLER_STATUS_ERROR;
					} break;

						CASE_INSTALLER_STATUS_MAX;
				}
			} break;

			case INSTALLER_STATUS_INSTALLED: {
				switch (key_value.value) {
					case INSTALLER_STATUS_INSTALLED: {
						// Do nothing, the state is the same.
					} break;

					case INSTALLER_STATUS_IDLE:
					case INSTALLER_STATUS_LOADING: {
						// As there's some status that are installed,
						// we can assume that the idle files will be
						// loaded in a few moments.
						status = INSTALLER_STATUS_LOADING;
					} break;

					case INSTALLER_STATUS_ERROR: {
						return INSTALLER_STATUS_ERROR;
					} break;

						CASE_INSTALLER_STATUS_MAX;
				}
			} break;

			case INSTALLER_STATUS_ERROR: {
				return INSTALLER_STATUS_ERROR;
			} break;

				CASE_INSTALLER_STATUS_MAX;
		}

#undef CASE_INSTALLER_STATUS_MAX
	}

	status_cached = status;
	status_dirty = false;
	return status;
}

bool AsyncPCKInstaller::are_files_installable() const {
	if (!install_needed_dirty) {
		return install_needed_cached;
	}

	bool install_needed = false;

	PackedStringArray processed_file_paths = _get_processed_file_paths();
	for (const String &file_path : processed_file_paths) {
		if (OS::get_singleton()->async_pck_is_file_installable(file_path)) {
			install_needed = true;
			break;
		}
	}

	install_needed_cached = install_needed;
	install_needed_dirty = false;
	return install_needed;
}

void AsyncPCKInstaller::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_autostart", "autostart"), &AsyncPCKInstaller::set_autostart);
	ClassDB::bind_method(D_METHOD("get_autostart"), &AsyncPCKInstaller::get_autostart);
	ClassDB::bind_method(D_METHOD("set_file_paths", "file_paths"), &AsyncPCKInstaller::set_file_paths);
	ClassDB::bind_method(D_METHOD("get_file_paths"), &AsyncPCKInstaller::get_file_paths);

	ClassDB::bind_method(D_METHOD("get_status"), &AsyncPCKInstaller::get_status);
	ClassDB::bind_method(D_METHOD("are_files_installable"), &AsyncPCKInstaller::are_files_installable);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "autostart"), "set_autostart", "get_autostart");
	ADD_PROPERTY(PropertyInfo(Variant::PACKED_STRING_ARRAY, "file_paths", PROPERTY_HINT_ARRAY_TYPE, MAKE_FILE_ARRAY_TYPE_HINT("*")), "set_file_paths", "get_file_paths");

	ADD_SIGNAL(MethodInfo(SIGNAL_FILE_ADDED, PropertyInfo(Variant::STRING, "file", PROPERTY_HINT_FILE)));
	ADD_SIGNAL(MethodInfo(SIGNAL_FILE_REMOVED, PropertyInfo(Variant::STRING, "file", PROPERTY_HINT_FILE)));
	ADD_SIGNAL(MethodInfo(SIGNAL_FILE_PROGRESS, PropertyInfo(Variant::STRING, "file", PROPERTY_HINT_FILE), PropertyInfo(Variant::DICTIONARY, "progress_data")));
	ADD_SIGNAL(MethodInfo(SIGNAL_FILE_INSTALLED, PropertyInfo(Variant::STRING, "file", PROPERTY_HINT_FILE)));
	ADD_SIGNAL(MethodInfo(SIGNAL_FILE_ERROR, PropertyInfo(Variant::ARRAY, "errors")));
	ADD_SIGNAL(MethodInfo(SIGNAL_PROGRESS, PropertyInfo(Variant::DICTIONARY, "progress_data")));
	ADD_SIGNAL(MethodInfo(SIGNAL_STATUS_CHANGED));

	BIND_ENUM_CONSTANT(INSTALLER_STATUS_IDLE);
	BIND_ENUM_CONSTANT(INSTALLER_STATUS_LOADING);
	BIND_ENUM_CONSTANT(INSTALLER_STATUS_INSTALLED);
	BIND_ENUM_CONSTANT(INSTALLER_STATUS_ERROR);
	BIND_ENUM_CONSTANT(INSTALLER_STATUS_MAX);
}
