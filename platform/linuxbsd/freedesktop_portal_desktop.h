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
#include "core/os/thread_safe.h"
#include "scene/resources/texture.h"
#include "servers/display_server.h"

#ifdef SOWRAP_ENABLED
#include "dbus-so_wrap.h"
#else
#include <dbus/dbus.h>
#endif

class FreeDesktopPortalDesktop : public Object {
private:
	bool unsupported = false;

	static bool try_parse_variant(DBusMessage *p_reply_message, int p_type, void *r_value);
	// Read a setting from org.freekdesktop.portal.Settings
	bool read_setting(const char *p_namespace, const char *p_key, int p_type, void *r_value);

	static void append_dbus_string(DBusMessageIter *p_iter, const String &p_string);
	static void append_dbus_dict_options(DBusMessageIter *p_iter, const TypedArray<Dictionary> &p_options);
	static void append_dbus_dict_filters(DBusMessageIter *p_iter, const Vector<String> &p_filter_names, const Vector<String> &p_filter_exts);
	static void append_dbus_dict_string(DBusMessageIter *p_iter, const String &p_key, const String &p_value, bool p_as_byte_array = false);
	static void append_dbus_dict_bool(DBusMessageIter *p_iter, const String &p_key, bool p_value);
	static bool file_chooser_parse_response(DBusMessageIter *p_iter, const Vector<String> &p_names, bool &r_cancel, Vector<String> &r_urls, int &r_index, Dictionary &r_options);

	struct FileDialogData {
		Vector<String> filter_names;
		DisplayServer::WindowID prev_focus = DisplayServer::INVALID_WINDOW_ID;
		Callable callback;
		String filter;
		String path;
		bool opt_in_cb = false;
	};

	struct FileDialogCallback {
		Callable callback;
		Variant status;
		Variant files;
		Variant index;
		Variant options;
		bool opt_in_cb = false;
	};
	List<FileDialogCallback> pending_cbs;

	Mutex dbus_mutex;
	Vector<FileDialogData> file_dialogs;
	Thread monitor_thread;
	SafeFlag monitor_thread_abort;
	DBusConnection *monitor_connection = nullptr;

	Vector<DBusWatch *> dbus_watches;

	String theme_path;
	Callable system_theme_changed;

	struct StatusNotifierItem {
		DisplayServer::IndicatorID id = DisplayServer::INVALID_INDICATOR_ID;
		String name;
		String tooltip;
		Callable activate_callback;
		Size2i icon_size;
		Vector<uint8_t> icon_data;
	};

	HashMap<DisplayServer::IndicatorID, StatusNotifierItem> indicators;
	HashMap<String, DisplayServer::IndicatorID> indicator_id_map;

	Size2i icon_image_size;
	Vector<uint8_t> icon_image_data;

	struct DBusObjectPathVTable status_indicator_item_vtable = {
		_status_notifier_item_unregister, // unregister_function
		_status_notifier_item_handle_message, // message_function
		_dbus_arg_noop, // dbus_internal_pad1
		_dbus_arg_noop, // dbus_internal_pad2
		_dbus_arg_noop, // dbus_internal_pad3
		_dbus_arg_noop, // dbus_internal_pad4
	};

	static void _dbus_connection_reply_error(DBusConnection *p_connection, DBusMessage *p_message, String p_error_message);
	void _system_theme_changed_callback();

	static void _dbus_arg_noop(void *arg) {}

	// Useful for responding to property requests.
	static dbus_bool_t _dbus_messsage_iter_append_basic_variant(DBusMessageIter *p_iter, int p_type, const void *p_value);
	static dbus_bool_t _dbus_message_iter_append_bool_variant(DBusMessageIter *p_iter, bool p_bool);
	static dbus_bool_t _dbus_message_iter_append_uint32_variant(DBusMessageIter *p_iter, uint32_t p_uint32);

	// Implements the StatusNotifierItem icon spec.
	static dbus_bool_t _dbus_message_iter_append_pixmap(DBusMessageIter *p_iter, Size2i p_size, Vector<uint8_t> p_data);

	static DBusHandlerResult _handle_message(DBusConnection *connection, DBusMessage *message, void *user_data);
	static dbus_bool_t _handle_add_watch(DBusWatch *watch, void *data);
	static void _handle_remove_watch(DBusWatch *watch, void *data);
	static void _handle_watch_toggled(DBusWatch *watch, void *data);

	static void _status_notifier_item_unregister(DBusConnection *connection, void *user_data);
	static DBusHandlerResult _status_notifier_item_handle_message(DBusConnection *connection, DBusMessage *message, void *user_data);

	static void _thread_monitor(void *p_ud);

public:
	FreeDesktopPortalDesktop();
	~FreeDesktopPortalDesktop();

	bool is_supported() { return !unsupported; }

	Error file_dialog_show(DisplayServer::WindowID p_window_id, const String &p_xid, const String &p_title, const String &p_current_directory, const String &p_root, const String &p_filename, DisplayServer::FileDialogMode p_mode, const Vector<String> &p_filters, const TypedArray<Dictionary> &p_options, const Callable &p_callback, bool p_options_in_cb);
	void process_file_dialog_callbacks();

	// Retrieve the system's preferred color scheme.
	// 0: No preference or unknown.
	// 1: Prefer dark appearance.
	// 2: Prefer light appearance.
	uint32_t get_appearance_color_scheme();
	void set_system_theme_change_callback(const Callable &p_system_theme_changed) {
		system_theme_changed = p_system_theme_changed;
	}

	bool indicator_register(DisplayServer::IndicatorID p_id);
	bool indicator_create(DisplayServer::IndicatorID p_id, const Ref<Texture2D> &p_icon);
	Error indicator_set_icon(DisplayServer::IndicatorID p_id, const Ref<Texture2D> &p_icon);
	void indicator_set_tooltip(DisplayServer::IndicatorID p_id, const String &p_tooltip);
	void indicator_set_callback(DisplayServer::IndicatorID p_id, const Callable &p_callback);
	void indicator_destroy(DisplayServer::IndicatorID p_id);
};

#endif // DBUS_ENABLED

#endif // FREEDESKTOP_PORTAL_DESKTOP_H
