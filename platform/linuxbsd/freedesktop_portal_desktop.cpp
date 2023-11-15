/**************************************************************************/
/*  freedesktop_portal_desktop.cpp                                        */
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

#include "freedesktop_portal_desktop.h"

#ifdef DBUS_ENABLED

#include "core/crypto/crypto_core.h"
#include "core/error/error_macros.h"
#include "core/os/os.h"
#include "core/string/ustring.h"
#include "core/variant/variant.h"

#ifdef SOWRAP_ENABLED
#include "dbus-so_wrap.h"
#else
#include <dbus/dbus.h>
#endif

#include <unistd.h>

#define BUS_OBJECT_NAME "org.freedesktop.portal.Desktop"
#define BUS_OBJECT_PATH "/org/freedesktop/portal/desktop"

#define BUS_INTERFACE_SETTINGS "org.freedesktop.portal.Settings"
#define BUS_INTERFACE_FILE_CHOOSER "org.freedesktop.portal.FileChooser"

bool FreeDesktopPortalDesktop::try_parse_variant(DBusMessage *p_reply_message, int p_type, void *r_value) {
	DBusMessageIter iter[3];

	dbus_message_iter_init(p_reply_message, &iter[0]);
	if (dbus_message_iter_get_arg_type(&iter[0]) != DBUS_TYPE_VARIANT) {
		return false;
	}

	dbus_message_iter_recurse(&iter[0], &iter[1]);
	if (dbus_message_iter_get_arg_type(&iter[1]) != DBUS_TYPE_VARIANT) {
		return false;
	}

	dbus_message_iter_recurse(&iter[1], &iter[2]);
	if (dbus_message_iter_get_arg_type(&iter[2]) != p_type) {
		return false;
	}

	dbus_message_iter_get_basic(&iter[2], r_value);
	return true;
}

bool FreeDesktopPortalDesktop::read_setting(const char *p_namespace, const char *p_key, int p_type, void *r_value) {
	if (unsupported) {
		return false;
	}

	DBusError error;
	dbus_error_init(&error);

	DBusConnection *bus = dbus_bus_get(DBUS_BUS_SESSION, &error);
	if (dbus_error_is_set(&error)) {
		if (OS::get_singleton()->is_stdout_verbose()) {
			ERR_PRINT(vformat("Error opening D-Bus connection: %s", error.message));
		}
		dbus_error_free(&error);
		unsupported = true;
		return false;
	}

	DBusMessage *message = dbus_message_new_method_call(
			BUS_OBJECT_NAME, BUS_OBJECT_PATH, BUS_INTERFACE_SETTINGS,
			"Read");
	dbus_message_append_args(
			message,
			DBUS_TYPE_STRING, &p_namespace,
			DBUS_TYPE_STRING, &p_key,
			DBUS_TYPE_INVALID);

	DBusMessage *reply = dbus_connection_send_with_reply_and_block(bus, message, 50, &error);
	dbus_message_unref(message);
	if (dbus_error_is_set(&error)) {
		if (OS::get_singleton()->is_stdout_verbose()) {
			ERR_PRINT(vformat("Error on D-Bus communication: %s", error.message));
		}
		dbus_error_free(&error);
		dbus_connection_unref(bus);
		return false;
	}

	bool success = try_parse_variant(reply, p_type, r_value);

	dbus_message_unref(reply);
	dbus_connection_unref(bus);

	return success;
}

uint32_t FreeDesktopPortalDesktop::get_appearance_color_scheme() {
	if (unsupported) {
		return 0;
	}

	uint32_t value = 0;
	read_setting("org.freedesktop.appearance", "color-scheme", DBUS_TYPE_UINT32, &value);
	return value;
}

static const char *cs_empty = "";

void FreeDesktopPortalDesktop::append_dbus_string(DBusMessageIter *p_iter, const String &p_string) {
	CharString cs = p_string.utf8();
	const char *cs_ptr = cs.ptr();
	if (cs_ptr) {
		dbus_message_iter_append_basic(p_iter, DBUS_TYPE_STRING, &cs_ptr);
	} else {
		dbus_message_iter_append_basic(p_iter, DBUS_TYPE_STRING, &cs_empty);
	}
}

void FreeDesktopPortalDesktop::append_dbus_dict_filters(DBusMessageIter *p_iter, const Vector<String> &p_filter_names, const Vector<String> &p_filter_exts) {
	DBusMessageIter dict_iter;
	DBusMessageIter var_iter;
	DBusMessageIter arr_iter;
	const char *filters_key = "filters";

	ERR_FAIL_COND(p_filter_names.size() != p_filter_exts.size());

	dbus_message_iter_open_container(p_iter, DBUS_TYPE_DICT_ENTRY, nullptr, &dict_iter);
	dbus_message_iter_append_basic(&dict_iter, DBUS_TYPE_STRING, &filters_key);
	dbus_message_iter_open_container(&dict_iter, DBUS_TYPE_VARIANT, "a(sa(us))", &var_iter);
	dbus_message_iter_open_container(&var_iter, DBUS_TYPE_ARRAY, "(sa(us))", &arr_iter);
	for (int i = 0; i < p_filter_names.size(); i++) {
		DBusMessageIter struct_iter;
		DBusMessageIter array_iter;
		DBusMessageIter array_struct_iter;
		dbus_message_iter_open_container(&arr_iter, DBUS_TYPE_STRUCT, nullptr, &struct_iter);
		append_dbus_string(&struct_iter, p_filter_names[i]);

		dbus_message_iter_open_container(&struct_iter, DBUS_TYPE_ARRAY, "(us)", &array_iter);
		String flt = p_filter_exts[i];
		int filter_slice_count = flt.get_slice_count(",");
		for (int j = 0; j < filter_slice_count; j++) {
			dbus_message_iter_open_container(&array_iter, DBUS_TYPE_STRUCT, nullptr, &array_struct_iter);
			String str = (flt.get_slice(",", j).strip_edges());
			{
				const unsigned nil = 0;
				dbus_message_iter_append_basic(&array_struct_iter, DBUS_TYPE_UINT32, &nil);
			}
			append_dbus_string(&array_struct_iter, str);
			dbus_message_iter_close_container(&array_iter, &array_struct_iter);
		}
		dbus_message_iter_close_container(&struct_iter, &array_iter);
		dbus_message_iter_close_container(&arr_iter, &struct_iter);
	}
	dbus_message_iter_close_container(&var_iter, &arr_iter);
	dbus_message_iter_close_container(&dict_iter, &var_iter);
	dbus_message_iter_close_container(p_iter, &dict_iter);
}

void FreeDesktopPortalDesktop::append_dbus_dict_string(DBusMessageIter *p_iter, const String &p_key, const String &p_value, bool p_as_byte_array) {
	DBusMessageIter dict_iter;
	DBusMessageIter var_iter;
	dbus_message_iter_open_container(p_iter, DBUS_TYPE_DICT_ENTRY, nullptr, &dict_iter);
	append_dbus_string(&dict_iter, p_key);

	if (p_as_byte_array) {
		DBusMessageIter arr_iter;
		dbus_message_iter_open_container(&dict_iter, DBUS_TYPE_VARIANT, "ay", &var_iter);
		dbus_message_iter_open_container(&var_iter, DBUS_TYPE_ARRAY, "y", &arr_iter);
		CharString cs = p_value.utf8();
		const char *cs_ptr = cs.get_data();
		do {
			dbus_message_iter_append_basic(&arr_iter, DBUS_TYPE_BYTE, cs_ptr);
		} while (*cs_ptr++);
		dbus_message_iter_close_container(&var_iter, &arr_iter);
	} else {
		dbus_message_iter_open_container(&dict_iter, DBUS_TYPE_VARIANT, "s", &var_iter);
		append_dbus_string(&var_iter, p_value);
	}

	dbus_message_iter_close_container(&dict_iter, &var_iter);
	dbus_message_iter_close_container(p_iter, &dict_iter);
}

void FreeDesktopPortalDesktop::append_dbus_dict_bool(DBusMessageIter *p_iter, const String &p_key, bool p_value) {
	DBusMessageIter dict_iter;
	DBusMessageIter var_iter;
	dbus_message_iter_open_container(p_iter, DBUS_TYPE_DICT_ENTRY, nullptr, &dict_iter);
	append_dbus_string(&dict_iter, p_key);

	dbus_message_iter_open_container(&dict_iter, DBUS_TYPE_VARIANT, "b", &var_iter);
	{
		int val = p_value;
		dbus_message_iter_append_basic(&var_iter, DBUS_TYPE_BOOLEAN, &val);
	}

	dbus_message_iter_close_container(&dict_iter, &var_iter);
	dbus_message_iter_close_container(p_iter, &dict_iter);
}

bool FreeDesktopPortalDesktop::file_chooser_parse_response(DBusMessageIter *p_iter, const Vector<String> &p_names, bool &r_cancel, Vector<String> &r_urls, int &r_index) {
	ERR_FAIL_COND_V(dbus_message_iter_get_arg_type(p_iter) != DBUS_TYPE_UINT32, false);

	dbus_uint32_t resp_code;
	dbus_message_iter_get_basic(p_iter, &resp_code);
	if (resp_code != 0) {
		r_cancel = true;
	} else {
		r_cancel = false;
		ERR_FAIL_COND_V(!dbus_message_iter_next(p_iter), false);
		ERR_FAIL_COND_V(dbus_message_iter_get_arg_type(p_iter) != DBUS_TYPE_ARRAY, false);

		DBusMessageIter dict_iter;
		dbus_message_iter_recurse(p_iter, &dict_iter);
		while (dbus_message_iter_get_arg_type(&dict_iter) == DBUS_TYPE_DICT_ENTRY) {
			DBusMessageIter iter;
			dbus_message_iter_recurse(&dict_iter, &iter);
			if (dbus_message_iter_get_arg_type(&iter) == DBUS_TYPE_STRING) {
				const char *key;
				dbus_message_iter_get_basic(&iter, &key);
				dbus_message_iter_next(&iter);

				DBusMessageIter var_iter;
				dbus_message_iter_recurse(&iter, &var_iter);
				if (strcmp(key, "current_filter") == 0) { // (sa(us))
					if (dbus_message_iter_get_arg_type(&var_iter) == DBUS_TYPE_STRUCT) {
						DBusMessageIter struct_iter;
						dbus_message_iter_recurse(&var_iter, &struct_iter);
						while (dbus_message_iter_get_arg_type(&struct_iter) == DBUS_TYPE_STRING) {
							const char *value;
							dbus_message_iter_get_basic(&struct_iter, &value);
							String name = String::utf8(value);

							r_index = p_names.find(name);
							if (!dbus_message_iter_next(&struct_iter)) {
								break;
							}
						}
					}
				} else if (strcmp(key, "uris") == 0) { // as
					if (dbus_message_iter_get_arg_type(&var_iter) == DBUS_TYPE_ARRAY) {
						DBusMessageIter uri_iter;
						dbus_message_iter_recurse(&var_iter, &uri_iter);
						while (dbus_message_iter_get_arg_type(&uri_iter) == DBUS_TYPE_STRING) {
							const char *value;
							dbus_message_iter_get_basic(&uri_iter, &value);
							r_urls.push_back(String::utf8(value).trim_prefix("file://").uri_decode());
							if (!dbus_message_iter_next(&uri_iter)) {
								break;
							}
						}
					}
				}
			}
			if (!dbus_message_iter_next(&dict_iter)) {
				break;
			}
		}
	}
	return true;
}

Error FreeDesktopPortalDesktop::file_dialog_show(DisplayServer::WindowID p_window_id, const String &p_xid, const String &p_title, const String &p_current_directory, const String &p_filename, DisplayServer::FileDialogMode p_mode, const Vector<String> &p_filters, const Callable &p_callback) {
	if (unsupported) {
		return FAILED;
	}

	ERR_FAIL_INDEX_V(int(p_mode), DisplayServer::FILE_DIALOG_MODE_SAVE_MAX, FAILED);

	Vector<String> filter_names;
	Vector<String> filter_exts;
	for (int i = 0; i < p_filters.size(); i++) {
		Vector<String> tokens = p_filters[i].split(";");
		if (tokens.size() >= 1) {
			String flt = tokens[0].strip_edges();
			if (!flt.is_empty()) {
				if (tokens.size() == 2) {
					filter_exts.push_back(flt);
					filter_names.push_back(tokens[1]);
				} else {
					filter_exts.push_back(flt);
					filter_names.push_back(flt);
				}
			}
		}
	}
	if (filter_names.is_empty()) {
		filter_exts.push_back("*.*");
		filter_names.push_back(RTR("All Files"));
	}

	DBusError err;
	dbus_error_init(&err);

	// Open connection and add signal handler.
	FileDialogData fd;
	fd.callback = p_callback;
	fd.prev_focus = p_window_id;
	fd.filter_names = filter_names;

	CryptoCore::RandomGenerator rng;
	ERR_FAIL_COND_V_MSG(rng.init(), FAILED, "Failed to initialize random number generator.");
	uint8_t uuid[64];
	Error rng_err = rng.get_random_bytes(uuid, 64);
	ERR_FAIL_COND_V_MSG(rng_err, rng_err, "Failed to generate unique token.");

	fd.connection = dbus_bus_get(DBUS_BUS_SESSION, &err);
	if (dbus_error_is_set(&err)) {
		ERR_PRINT(vformat("Failed to open DBus connection: %s", err.message));
		dbus_error_free(&err);
		unsupported = true;
		return FAILED;
	}

	String dbus_unique_name = String::utf8(dbus_bus_get_unique_name(fd.connection));
	String token = String::hex_encode_buffer(uuid, 64);
	String path = vformat("/org/freedesktop/portal/desktop/request/%s/%s", dbus_unique_name.replace(".", "_").replace(":", ""), token);

	fd.path = vformat("type='signal',sender='org.freedesktop.portal.Desktop',path='%s',interface='org.freedesktop.portal.Request',member='Response',destination='%s'", path, dbus_unique_name);
	dbus_bus_add_match(fd.connection, fd.path.utf8().get_data(), &err);
	if (dbus_error_is_set(&err)) {
		ERR_PRINT(vformat("Failed to add DBus match: %s", err.message));
		dbus_error_free(&err);
		dbus_connection_unref(fd.connection);
		return FAILED;
	}

	// Generate FileChooser message.
	const char *method = nullptr;
	if (p_mode == DisplayServer::FILE_DIALOG_MODE_SAVE_FILE) {
		method = "SaveFile";
	} else {
		method = "OpenFile";
	}

	DBusMessage *message = dbus_message_new_method_call(BUS_OBJECT_NAME, BUS_OBJECT_PATH, BUS_INTERFACE_FILE_CHOOSER, method);
	{
		DBusMessageIter iter;
		dbus_message_iter_init_append(message, &iter);

		append_dbus_string(&iter, p_xid);
		append_dbus_string(&iter, p_title);

		DBusMessageIter arr_iter;
		dbus_message_iter_open_container(&iter, DBUS_TYPE_ARRAY, "{sv}", &arr_iter);

		append_dbus_dict_string(&arr_iter, "handle_token", token);
		append_dbus_dict_bool(&arr_iter, "multiple", p_mode == DisplayServer::FILE_DIALOG_MODE_OPEN_FILES);
		append_dbus_dict_bool(&arr_iter, "directory", p_mode == DisplayServer::FILE_DIALOG_MODE_OPEN_DIR);
		append_dbus_dict_filters(&arr_iter, filter_names, filter_exts);
		append_dbus_dict_string(&arr_iter, "current_folder", p_current_directory, true);
		if (p_mode == DisplayServer::FILE_DIALOG_MODE_SAVE_FILE) {
			append_dbus_dict_string(&arr_iter, "current_name", p_filename);
		}

		dbus_message_iter_close_container(&iter, &arr_iter);
	}

	DBusMessage *reply = dbus_connection_send_with_reply_and_block(fd.connection, message, DBUS_TIMEOUT_INFINITE, &err);
	dbus_message_unref(message);

	if (!reply || dbus_error_is_set(&err)) {
		ERR_PRINT(vformat("Failed to send DBus message: %s", err.message));
		dbus_error_free(&err);
		dbus_bus_remove_match(fd.connection, fd.path.utf8().get_data(), &err);
		dbus_connection_unref(fd.connection);
		return FAILED;
	}

	// Update signal path.
	{
		DBusMessageIter iter;
		if (dbus_message_iter_init(reply, &iter)) {
			if (dbus_message_iter_get_arg_type(&iter) == DBUS_TYPE_OBJECT_PATH) {
				const char *new_path = nullptr;
				dbus_message_iter_get_basic(&iter, &new_path);
				if (String::utf8(new_path) != path) {
					dbus_bus_remove_match(fd.connection, fd.path.utf8().get_data(), &err);
					if (dbus_error_is_set(&err)) {
						ERR_PRINT(vformat("Failed to remove DBus match: %s", err.message));
						dbus_error_free(&err);
						dbus_connection_unref(fd.connection);
						return FAILED;
					}
					fd.path = String::utf8(new_path);
					dbus_bus_add_match(fd.connection, fd.path.utf8().get_data(), &err);
					if (dbus_error_is_set(&err)) {
						ERR_PRINT(vformat("Failed to add DBus match: %s", err.message));
						dbus_error_free(&err);
						dbus_connection_unref(fd.connection);
						return FAILED;
					}
				}
			}
		}
	}
	dbus_message_unref(reply);

	MutexLock lock(file_dialog_mutex);
	file_dialogs.push_back(fd);

	return OK;
}

void FreeDesktopPortalDesktop::_file_dialog_callback(const Callable &p_callable, const Variant &p_status, const Variant &p_list, const Variant &p_index) {
	Variant ret;
	Callable::CallError ce;
	const Variant *args[3] = { &p_status, &p_list, &p_index };

	p_callable.callp(args, 3, ret, ce);
	if (ce.error != Callable::CallError::CALL_OK) {
		ERR_PRINT(vformat("Failed to execute file dialogs callback: %s.", Variant::get_callable_error_text(p_callable, args, 3, ce)));
	}
}

void FreeDesktopPortalDesktop::_thread_file_dialog_monitor(void *p_ud) {
	FreeDesktopPortalDesktop *portal = (FreeDesktopPortalDesktop *)p_ud;

	while (!portal->file_dialog_thread_abort.is_set()) {
		{
			MutexLock lock(portal->file_dialog_mutex);
			for (int i = portal->file_dialogs.size() - 1; i >= 0; i--) {
				bool remove = false;
				{
					FreeDesktopPortalDesktop::FileDialogData &fd = portal->file_dialogs.write[i];
					if (fd.connection) {
						while (true) {
							DBusMessage *msg = dbus_connection_pop_message(fd.connection);
							if (!msg) {
								break;
							} else if (dbus_message_is_signal(msg, "org.freedesktop.portal.Request", "Response")) {
								DBusMessageIter iter;
								if (dbus_message_iter_init(msg, &iter)) {
									bool cancel = false;
									Vector<String> uris;
									int index = 0;
									file_chooser_parse_response(&iter, fd.filter_names, cancel, uris, index);

									if (fd.callback.is_valid()) {
										callable_mp(portal, &FreeDesktopPortalDesktop::_file_dialog_callback).call_deferred(fd.callback, !cancel, uris, index);
									}
									if (fd.prev_focus != DisplayServer::INVALID_WINDOW_ID) {
										callable_mp(DisplayServer::get_singleton(), &DisplayServer::window_move_to_foreground).call_deferred(fd.prev_focus);
									}
								}
								dbus_message_unref(msg);

								DBusError err;
								dbus_error_init(&err);
								dbus_bus_remove_match(fd.connection, fd.path.utf8().get_data(), &err);
								dbus_error_free(&err);
								dbus_connection_unref(fd.connection);
								remove = true;
								break;
							}
							dbus_message_unref(msg);
						}
						dbus_connection_read_write(fd.connection, 0);
					}
				}
				if (remove) {
					portal->file_dialogs.remove_at(i);
				}
			}
		}
		usleep(50000);
	}
}

FreeDesktopPortalDesktop::FreeDesktopPortalDesktop() {
#ifdef SOWRAP_ENABLED
#ifdef DEBUG_ENABLED
	int dylibloader_verbose = 1;
#else
	int dylibloader_verbose = 0;
#endif
	unsupported = (initialize_dbus(dylibloader_verbose) != 0);
#else
	unsupported = false;
#endif

	if (unsupported) {
		return;
	}

	bool ver_ok = false;
	int version_major = 0;
	int version_minor = 0;
	int version_rev = 0;
	dbus_get_version(&version_major, &version_minor, &version_rev);
	ver_ok = (version_major == 1 && version_minor >= 10) || (version_major > 1); // 1.10.0
	print_verbose(vformat("PortalDesktop: DBus %d.%d.%d detected.", version_major, version_minor, version_rev));
	if (!ver_ok) {
		print_verbose("PortalDesktop: Unsupported DBus library version!");
		unsupported = true;
	}

	if (!unsupported) {
		file_dialog_thread_abort.clear();
		file_dialog_thread.start(FreeDesktopPortalDesktop::_thread_file_dialog_monitor, this);
	}
}

FreeDesktopPortalDesktop::~FreeDesktopPortalDesktop() {
	file_dialog_thread_abort.set();
	if (file_dialog_thread.is_started()) {
		file_dialog_thread.wait_to_finish();
	}
	for (FreeDesktopPortalDesktop::FileDialogData &fd : file_dialogs) {
		if (fd.connection) {
			DBusError err;
			dbus_error_init(&err);
			dbus_bus_remove_match(fd.connection, fd.path.utf8().get_data(), &err);
			dbus_error_free(&err);
			dbus_connection_unref(fd.connection);
		}
	}
}

#endif // DBUS_ENABLED
