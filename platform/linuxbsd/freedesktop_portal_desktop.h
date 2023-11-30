/**************************************************************************/
/*  freedesktop_portal_desktop.h                                          */
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

#ifndef FREEDESKTOP_PORTAL_DESKTOP_H
#define FREEDESKTOP_PORTAL_DESKTOP_H

#ifdef DBUS_ENABLED

#include "core/os/thread.h"
#include "servers/display_server.h"

struct DBusMessage;
struct DBusConnection;
struct DBusMessageIter;

class FreeDesktopPortalDesktop : public Object {
private:
	bool unsupported = false;

	static bool try_parse_variant(DBusMessage *p_reply_message, int p_type, void *r_value);
	// Read a setting from org.freekdesktop.portal.Settings
	bool read_setting(const char *p_namespace, const char *p_key, int p_type, void *r_value);

	static void append_dbus_string(DBusMessageIter *p_iter, const String &p_string);
	static void append_dbus_dict_filters(DBusMessageIter *p_iter, const Vector<String> &p_filter_names, const Vector<String> &p_filter_exts);
	static void append_dbus_dict_string(DBusMessageIter *p_iter, const String &p_key, const String &p_value, bool p_as_byte_array = false);
	static void append_dbus_dict_bool(DBusMessageIter *p_iter, const String &p_key, bool p_value);
	static bool file_chooser_parse_response(DBusMessageIter *p_iter, const Vector<String> &p_names, bool &r_cancel, Vector<String> &r_urls, int &r_index);

	void _file_dialog_callback(const Callable &p_callable, const Variant &p_status, const Variant &p_list, const Variant &p_index);

	struct FileDialogData {
		Vector<String> filter_names;
		DBusConnection *connection = nullptr;
		DisplayServer::WindowID prev_focus = DisplayServer::INVALID_WINDOW_ID;
		Callable callback;
		String path;
	};

	Mutex file_dialog_mutex;
	Vector<FileDialogData> file_dialogs;
	Thread file_dialog_thread;
	SafeFlag file_dialog_thread_abort;

	static void _thread_file_dialog_monitor(void *p_ud);

public:
	FreeDesktopPortalDesktop();
	~FreeDesktopPortalDesktop();

	bool is_supported() { return !unsupported; }

	Error file_dialog_show(DisplayServer::WindowID p_window_id, const String &p_xid, const String &p_title, const String &p_current_directory, const String &p_filename, DisplayServer::FileDialogMode p_mode, const Vector<String> &p_filters, const Callable &p_callback);

	// Retrieve the system's preferred color scheme.
	// 0: No preference or unknown.
	// 1: Prefer dark appearance.
	// 2: Prefer light appearance.
	uint32_t get_appearance_color_scheme();
};

#endif // DBUS_ENABLED

#endif // FREEDESKTOP_PORTAL_DESKTOP_H
