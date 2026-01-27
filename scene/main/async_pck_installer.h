/**************************************************************************/
/*  async_pck_installer.h                                                 */
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

#include "scene/main/node.h"

class AsyncPCKInstaller : public Node {
	GDCLASS(AsyncPCKInstaller, Node);

	const static inline String SIGNAL_FILE_ADDED = "file_added";
	const static inline String SIGNAL_FILE_REMOVED = "file_removed";
	const static inline String SIGNAL_FILE_PROGRESS = "file_progress";
	const static inline String SIGNAL_FILE_INSTALLED = "file_installed";
	const static inline String SIGNAL_FILE_ERROR = "file_error";
	const static inline String SIGNAL_PROGRESS = "progress";
	const static inline String SIGNAL_STATUS_CHANGED = "status_changed";

public:
	enum InstallerStatus {
		INSTALLER_STATUS_IDLE,
		INSTALLER_STATUS_LOADING,
		INSTALLER_STATUS_INSTALLED,
		INSTALLER_STATUS_ERROR,
		INSTALLER_STATUS_MAX,
	};

private:
	bool autostart = false;
	bool started = false;

	mutable bool status_dirty = true;
	mutable InstallerStatus status_cached = INSTALLER_STATUS_IDLE;

	mutable bool install_needed_dirty = true;
	mutable bool install_needed_cached = false;

	PackedStringArray file_paths;
	HashMap<String, InstallerStatus> file_paths_status;

	String _process_file_path(const String &p_path) const;
	PackedStringArray _get_processed_file_paths() const;

protected:
	void _notification(int p_what);
	static void _bind_methods();

	void update();

	bool set_file_path_status(const String &p_path, InstallerStatus p_status);

public:
	void start();

	void set_autostart(bool p_autostart);
	bool get_autostart() const;

	void set_file_paths(const PackedStringArray &p_resources_paths);
	PackedStringArray get_file_paths() const;

	InstallerStatus get_status() const;
	bool are_files_installable() const;
};

VARIANT_ENUM_CAST(AsyncPCKInstaller::InstallerStatus);
