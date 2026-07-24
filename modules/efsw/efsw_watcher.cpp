/**************************************************************************/
/*  efsw_watcher.cpp                                                      */
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

#include "efsw_watcher.h"

#include "core/config/project_settings.h"
#include "core/object/class_db.h"
#include "core/string/ustring.h"

class EFSWListenerProxy : public efsw::FileWatchListener {
private:
	Callable file_action_handler;
	Callable missed_file_actions_handler;

public:
	void set_file_action_handler(const Callable &p_handler) {
		ERR_FAIL_COND(!p_handler.is_valid());
		file_action_handler = p_handler;
	}

	void set_missed_file_actions_handler(const Callable &p_handler) {
		ERR_FAIL_COND(!p_handler.is_valid());
		missed_file_actions_handler = p_handler;
	}

	void handleFileAction(efsw::WatchID p_watchid, const std::string &p_dir, const std::string &p_filename, bool p_is_dir, efsw::Actions::Action p_action, std::string p_old_filename) override {
		if (file_action_handler.is_valid()) {
			file_action_handler.call(p_watchid, String::utf8(p_dir.c_str()), String::utf8(p_filename.c_str()), p_is_dir, static_cast<EFSWListener::FileAction>(p_action), String::utf8(p_old_filename.c_str()));
		}
	}

	void handleMissedFileActions(efsw::WatchID p_watchid, const std::string &p_dir) override {
		if (missed_file_actions_handler.is_valid()) {
			missed_file_actions_handler.call(p_watchid, String::utf8(p_dir.c_str()));
		}
	}

	EFSWListenerProxy() {}
	~EFSWListenerProxy() {
		file_action_handler = Callable();
		missed_file_actions_handler = Callable();
	}
};

void EFSWListener::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_file_action_handler", "handler"), &EFSWListener::set_file_action_handler);
	ClassDB::bind_method(D_METHOD("set_missed_file_actions_handler", "handler"), &EFSWListener::set_missed_file_actions_handler);

	BIND_ENUM_CONSTANT(ACTION_ADD);
	BIND_ENUM_CONSTANT(ACTION_DELETE);
	BIND_ENUM_CONSTANT(ACTION_MODIFIED);
	BIND_ENUM_CONSTANT(ACTION_MOVED);
}

void EFSWListener::set_file_action_handler(const Callable &p_handler) {
	ERR_FAIL_NULL(proxy);
	proxy->set_file_action_handler(p_handler);
}

void EFSWListener::set_missed_file_actions_handler(const Callable &p_handler) {
	ERR_FAIL_NULL(proxy);
	proxy->set_missed_file_actions_handler(p_handler);
}

EFSWListener::EFSWListener() {
	proxy = memnew(EFSWListenerProxy);
}

EFSWListener::~EFSWListener() {
	ERR_FAIL_NULL(proxy);
	memdelete(proxy);
	proxy = nullptr;
}

class EFSWWatcherProxy : public efsw::FileWatcher {
public:
	EFSWWatcherProxy(bool force) :
			efsw::FileWatcher(force) {}
};

void EFSWWatcher::_bind_methods() {
	ClassDB::bind_method(D_METHOD("add_watch", "directory", "listener", "recursive"), &EFSWWatcher::add_watch, DEFVAL(true));
	ClassDB::bind_method(D_METHOD("get_watch_directories"), &EFSWWatcher::get_watch_directories);
	ClassDB::bind_method(D_METHOD("remove_watch_by_id", "watch_id"), &EFSWWatcher::remove_watch_by_id);
	ClassDB::bind_method(D_METHOD("remove_watch_by_path", "directory"), &EFSWWatcher::remove_watch_by_path);
	ClassDB::bind_method(D_METHOD("watch"), &EFSWWatcher::watch);

	ClassDB::bind_method(D_METHOD("set_allow_out_of_scope_links", "enable"), &EFSWWatcher::set_allow_out_of_scope_links);
	ClassDB::bind_method(D_METHOD("get_allow_out_of_scope_links"), &EFSWWatcher::get_allow_out_of_scope_links);
	ClassDB::bind_method(D_METHOD("set_follow_symlinks", "enable"), &EFSWWatcher::set_follow_symlinks);
	ClassDB::bind_method(D_METHOD("get_follow_symlinks"), &EFSWWatcher::get_follow_symlinks);
	ClassDB::bind_method(D_METHOD("set_force_generic", "enable"), &EFSWWatcher::set_force_generic);
	ClassDB::bind_method(D_METHOD("get_force_generic"), &EFSWWatcher::get_force_generic);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "allow_out_of_scope_links"), "set_allow_out_of_scope_links", "get_allow_out_of_scope_links");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "follow_symlinks"), "set_follow_symlinks", "get_follow_symlinks");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "force_generic"), "set_force_generic", "get_force_generic");

	BIND_ENUM_CONSTANT(NO_ERROR);
	BIND_ENUM_CONSTANT(FILE_NOT_FOUND);
	BIND_ENUM_CONSTANT(FILE_REPEATED);
	BIND_ENUM_CONSTANT(FILE_OUT_OF_SCOPE);
	BIND_ENUM_CONSTANT(FILE_NOT_READABLE);
	BIND_ENUM_CONSTANT(FILE_REMOTE);
	BIND_ENUM_CONSTANT(WATCHER_FAILED);
	BIND_ENUM_CONSTANT(UNSPECIFIED);
}

int EFSWWatcher::add_watch(const String &p_dir_path, const EFSWListener *p_listener, bool p_recursive) {
	ERR_FAIL_NULL_V(proxy, -1);
	ERR_FAIL_NULL_V(p_listener, -1);

	const String &global_path = ProjectSettings::get_singleton()->globalize_path(p_dir_path);
	int watch_id = -1;
	watch_id = proxy->addWatch(global_path.utf8().get_data(), p_listener->get_proxy(), p_recursive);
	return watch_id;
}

TypedArray<String> EFSWWatcher::get_watch_directories() const {
	std::vector<std::string> dirs = proxy->directories();

	TypedArray<String> watch_dirs;
	watch_dirs.resize(dirs.size());

	for (const std::string &dir : dirs) {
		watch_dirs.push_back(String::utf8(dir.c_str()));
	}

	return watch_dirs;
}

void EFSWWatcher::remove_watch_by_id(int p_watch_id) {
	ERR_FAIL_NULL(proxy);
	proxy->removeWatch(p_watch_id);
}

void EFSWWatcher::remove_watch_by_path(const String &p_dir_path) {
	ERR_FAIL_NULL(proxy);
	const String &global_path = ProjectSettings::get_singleton()->globalize_path(p_dir_path);
	proxy->removeWatch(global_path.utf8().get_data());
}

void EFSWWatcher::watch() {
	ERR_FAIL_NULL(proxy);
	proxy->watch();
}

void EFSWWatcher::set_follow_symlinks(bool p_enable) {
	proxy->followSymlinks(p_enable);
}

bool EFSWWatcher::get_follow_symlinks() const {
	return proxy->followSymlinks();
}

void EFSWWatcher::set_allow_out_of_scope_links(bool p_enable) {
	proxy->allowOutOfScopeLinks(p_enable);
}

bool EFSWWatcher::get_allow_out_of_scope_links() const {
	return proxy->allowOutOfScopeLinks();
}

void EFSWWatcher::set_force_generic(bool p_force) {
	if (force_generic == p_force) {
		return;
	}
	ERR_FAIL_NULL(proxy);
	delete proxy;

	force_generic = p_force;
	proxy = new EFSWWatcherProxy(force_generic);
}

EFSWWatcher::EFSWWatcher() {
	proxy = new EFSWWatcherProxy(force_generic);
}

EFSWWatcher::~EFSWWatcher() {
	ERR_FAIL_NULL(proxy);
	delete proxy;
	proxy = nullptr;
}
