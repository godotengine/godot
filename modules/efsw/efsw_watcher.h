/**************************************************************************/
/*  efsw_watcher.h                                                        */
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

#include "core/object/object.h"
#include "core/variant/typed_array.h"

#include <efsw/efsw.hpp>

class EFSWListenerProxy;

class EFSWListener : public Object {
	GDCLASS(EFSWListener, Object);

	EFSWListenerProxy *proxy = nullptr;

protected:
	static void _bind_methods();

public:
	enum FileAction {
		ACTION_ADD = efsw::Actions::Add, // A file or directory is created or moved in from outside the scope.
		ACTION_DELETE = efsw::Actions::Delete, // A file or directory is deleted or moved out of scope.
		ACTION_MODIFIED = efsw::Actions::Modified, // A file or directory is modified.
		ACTION_MOVED = efsw::Actions::Moved, // A file or directory is moved.
	};

	// The parameter list required for the callable is (int, const String&, const String&, int, const String&).
	void set_file_action_handler(const Callable &p_handler);
	// The parameter list required for the callable is (int, const String&).
	void set_missed_file_actions_handler(const Callable &p_handler);

	EFSWListenerProxy *get_proxy() const { return proxy; }

	EFSWListener();
	~EFSWListener();
};

class EFSWWatcherProxy;

class EFSWWatcher : public Object {
	GDCLASS(EFSWWatcher, Object);

	bool force_generic = false; // Force the use of generic backend.

	EFSWWatcherProxy *proxy = nullptr;

protected:
	static void _bind_methods();

public:
	enum WatcherError {
		NO_ERROR = efsw::Errors::NoError,
		FILE_NOT_FOUND = efsw::Errors::FileNotFound,
		FILE_REPEATED = efsw::Errors::FileRepeated,
		FILE_OUT_OF_SCOPE = efsw::Errors::FileOutOfScope,
		FILE_NOT_READABLE = efsw::Errors::FileNotReadable,
		FILE_REMOTE = efsw::Errors::FileRemote, // Directory in remote file system. Please create a generic backend instance to watch this directory.
		WATCHER_FAILED = efsw::Errors::WatcherFailed, // File system watcher failed to watch for changes.
		UNSPECIFIED = efsw::Errors::Unspecified,
	};

	int add_watch(const String &p_dir_path, const EFSWListener *p_listener, bool p_recursive = true);
	TypedArray<String> get_watch_directories() const;
	void remove_watch_by_id(int p_watch_id);
	void remove_watch_by_path(const String &p_dir_path);
	void watch();

	void set_follow_symlinks(bool p_follow);
	bool get_follow_symlinks() const;

	void set_allow_out_of_scope_links(bool p_allow);
	bool get_allow_out_of_scope_links() const;

	void set_force_generic(bool p_force);
	bool get_force_generic() const { return force_generic; }

	EFSWWatcher();
	~EFSWWatcher();
};

VARIANT_ENUM_CAST(EFSWListener::FileAction);
VARIANT_ENUM_CAST(EFSWWatcher::WatcherError);
