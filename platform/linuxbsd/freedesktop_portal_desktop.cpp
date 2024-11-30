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

#include <poll.h>
#include <strings.h>
#include <unistd.h>

#define BUS_OBJECT_NAME "org.freedesktop.portal.Desktop"
#define BUS_OBJECT_PATH "/org/freedesktop/portal/desktop"

#define BUS_INTERFACE_PROPERTIES "org.freedesktop.DBus.Properties"

#define BUS_INTERFACE_SETTINGS "org.freedesktop.portal.Settings"
#define BUS_INTERFACE_FILE_CHOOSER "org.freedesktop.portal.FileChooser"

#define BUS_INTERFACE_STATUS_NOTIFIER_ITEM "org.freedesktop.StatusNotifierItem"
#define BUS_STATUS_NOTIFIER_ITEM_NAME "org.freedesktop.StatusNotifierItem"
#define BUS_STATUS_NOTIFIER_ITEM_PATH "/StatusNotifierItem"

#define BUS_INTERFACE_STATUS_NOTIFIER_WATCHER "org.freedesktop.StatusNotifierWatcher"
#define BUS_STATUS_NOTIFIER_WATCHER_NAME "org.freedesktop.StatusNotifierWatcher"
#define BUS_STATUS_NOTIFIER_WATCHER_PATH "/StatusNotifierWatcher"

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

void FreeDesktopPortalDesktop::append_dbus_dict_options(DBusMessageIter *p_iter, const TypedArray<Dictionary> &p_options) {
	DBusMessageIter dict_iter;
	DBusMessageIter var_iter;
	DBusMessageIter arr_iter;
	const char *choices_key = "choices";

	dbus_message_iter_open_container(p_iter, DBUS_TYPE_DICT_ENTRY, nullptr, &dict_iter);
	dbus_message_iter_append_basic(&dict_iter, DBUS_TYPE_STRING, &choices_key);
	dbus_message_iter_open_container(&dict_iter, DBUS_TYPE_VARIANT, "a(ssa(ss)s)", &var_iter);
	dbus_message_iter_open_container(&var_iter, DBUS_TYPE_ARRAY, "(ss(ss)s)", &arr_iter);

	for (int i = 0; i < p_options.size(); i++) {
		const Dictionary &item = p_options[i];
		if (!item.has("name") || !item.has("values") || !item.has("default")) {
			continue;
		}
		const String &name = item["name"];
		const Vector<String> &options = item["values"];
		int default_idx = item["default"];

		DBusMessageIter struct_iter;
		DBusMessageIter array_iter;
		DBusMessageIter array_struct_iter;
		dbus_message_iter_open_container(&arr_iter, DBUS_TYPE_STRUCT, nullptr, &struct_iter);
		append_dbus_string(&struct_iter, name); // ID.
		append_dbus_string(&struct_iter, name); // User visible name.

		dbus_message_iter_open_container(&struct_iter, DBUS_TYPE_ARRAY, "(ss)", &array_iter);
		for (int j = 0; j < options.size(); j++) {
			dbus_message_iter_open_container(&array_iter, DBUS_TYPE_STRUCT, nullptr, &array_struct_iter);
			append_dbus_string(&array_struct_iter, itos(j));
			append_dbus_string(&array_struct_iter, options[j]);
			dbus_message_iter_close_container(&array_iter, &array_struct_iter);
		}
		dbus_message_iter_close_container(&struct_iter, &array_iter);
		if (options.is_empty()) {
			append_dbus_string(&struct_iter, (default_idx) ? "true" : "false"); // Default selection.
		} else {
			append_dbus_string(&struct_iter, itos(default_idx)); // Default selection.
		}

		dbus_message_iter_close_container(&arr_iter, &struct_iter);
	}
	dbus_message_iter_close_container(&var_iter, &arr_iter);
	dbus_message_iter_close_container(&dict_iter, &var_iter);
	dbus_message_iter_close_container(p_iter, &dict_iter);
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
		const String &flt_orig = p_filter_exts[i];
		String flt;
		for (int j = 0; j < flt_orig.length(); j++) {
			if (is_unicode_letter(flt_orig[j])) {
				flt += vformat("[%c%c]", String::char_lowercase(flt_orig[j]), String::char_uppercase(flt_orig[j]));
			} else {
				flt += flt_orig[j];
			}
		}
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

bool FreeDesktopPortalDesktop::file_chooser_parse_response(DBusMessageIter *p_iter, const Vector<String> &p_names, bool &r_cancel, Vector<String> &r_urls, int &r_index, Dictionary &r_options) {
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
				} else if (strcmp(key, "choices") == 0) { // a(ss) {
					if (dbus_message_iter_get_arg_type(&var_iter) == DBUS_TYPE_ARRAY) {
						DBusMessageIter struct_iter;
						dbus_message_iter_recurse(&var_iter, &struct_iter);
						while (dbus_message_iter_get_arg_type(&struct_iter) == DBUS_TYPE_STRUCT) {
							DBusMessageIter opt_iter;
							dbus_message_iter_recurse(&struct_iter, &opt_iter);
							const char *opt_key = nullptr;
							dbus_message_iter_get_basic(&opt_iter, &opt_key);
							String opt_skey = String::utf8(opt_key);

							dbus_message_iter_next(&opt_iter);
							const char *opt_val = nullptr;
							dbus_message_iter_get_basic(&opt_iter, &opt_val);
							String opt_sval = String::utf8(opt_val);
							if (opt_sval == "true") {
								r_options[opt_skey] = true;
							} else if (opt_sval == "false") {
								r_options[opt_skey] = false;
							} else {
								r_options[opt_skey] = opt_sval.to_int();
							}

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

Error FreeDesktopPortalDesktop::file_dialog_show(DisplayServer::WindowID p_window_id, const String &p_xid, const String &p_title, const String &p_current_directory, const String &p_root, const String &p_filename, DisplayServer::FileDialogMode p_mode, const Vector<String> &p_filters, const TypedArray<Dictionary> &p_options, const Callable &p_callback, bool p_options_in_cb) {
	if (unsupported) {
		return FAILED;
	}

	ERR_FAIL_INDEX_V(int(p_mode), DisplayServer::FILE_DIALOG_MODE_SAVE_MAX, FAILED);
	ERR_FAIL_NULL_V(monitor_connection, FAILED);

	Vector<String> filter_names;
	Vector<String> filter_exts;
	for (int i = 0; i < p_filters.size(); i++) {
		Vector<String> tokens = p_filters[i].split(";");
		if (tokens.size() >= 1) {
			String flt = tokens[0].strip_edges();
			if (!flt.is_empty()) {
				if (tokens.size() == 2) {
					if (flt == "*.*") {
						filter_exts.push_back("*");
					} else {
						filter_exts.push_back(flt);
					}
					filter_names.push_back(tokens[1]);
				} else {
					if (flt == "*.*") {
						filter_exts.push_back("*");
						filter_names.push_back(RTR("All Files") + " (*)");
					} else {
						filter_exts.push_back(flt);
						filter_names.push_back(flt);
					}
				}
			}
		}
	}
	if (filter_names.is_empty()) {
		filter_exts.push_back("*");
		filter_names.push_back(RTR("All Files") + " (*)");
	}

	DBusError err;
	dbus_error_init(&err);

	// Open connection and add signal handler.
	FileDialogData fd;
	fd.callback = p_callback;
	fd.prev_focus = p_window_id;
	fd.filter_names = filter_names;
	fd.opt_in_cb = p_options_in_cb;

	CryptoCore::RandomGenerator rng;
	ERR_FAIL_COND_V_MSG(rng.init(), FAILED, "Failed to initialize random number generator.");
	uint8_t uuid[64];
	Error rng_err = rng.get_random_bytes(uuid, 64);
	ERR_FAIL_COND_V_MSG(rng_err, rng_err, "Failed to generate unique token.");

	String dbus_unique_name = String::utf8(dbus_bus_get_unique_name(monitor_connection));
	String token = String::hex_encode_buffer(uuid, 64);
	String path = vformat("/org/freedesktop/portal/desktop/request/%s/%s", dbus_unique_name.replace(".", "_").replace(":", ""), token);

	fd.path = path;
	fd.filter = vformat("type='signal',sender='org.freedesktop.portal.Desktop',path='%s',interface='org.freedesktop.portal.Request',member='Response',destination='%s'", path, dbus_unique_name);
	dbus_bus_add_match(monitor_connection, fd.filter.utf8().get_data(), &err);
	if (dbus_error_is_set(&err)) {
		ERR_PRINT(vformat("Failed to add DBus match: %s", err.message));
		dbus_error_free(&err);
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

		append_dbus_dict_options(&arr_iter, p_options);
		append_dbus_dict_string(&arr_iter, "current_folder", p_current_directory, true);
		if (p_mode == DisplayServer::FILE_DIALOG_MODE_SAVE_FILE) {
			append_dbus_dict_string(&arr_iter, "current_name", p_filename);
		}

		dbus_message_iter_close_container(&iter, &arr_iter);
	}

	DBusMessage *reply = dbus_connection_send_with_reply_and_block(monitor_connection, message, DBUS_TIMEOUT_INFINITE, &err);
	dbus_message_unref(message);

	if (!reply || dbus_error_is_set(&err)) {
		ERR_PRINT(vformat("Failed to send DBus message: %s", err.message));
		dbus_error_free(&err);
		dbus_bus_remove_match(monitor_connection, fd.filter.utf8().get_data(), &err);
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
					dbus_bus_remove_match(monitor_connection, fd.filter.utf8().get_data(), &err);
					if (dbus_error_is_set(&err)) {
						ERR_PRINT(vformat("Failed to remove DBus match: %s", err.message));
						dbus_error_free(&err);
						return FAILED;
					}
					fd.filter = String::utf8(new_path);
					dbus_bus_add_match(monitor_connection, fd.filter.utf8().get_data(), &err);
					if (dbus_error_is_set(&err)) {
						ERR_PRINT(vformat("Failed to add DBus match: %s", err.message));
						dbus_error_free(&err);
						return FAILED;
					}
				}
			}
		}
	}
	dbus_message_unref(reply);

	//MutexLock lock(dbus_mutex);
	file_dialogs.push_back(fd);

	return OK;
}

dbus_bool_t FreeDesktopPortalDesktop::_dbus_message_iter_append_pixmap(DBusMessageIter *p_iter, Size2i p_size, Vector<uint8_t> p_data) {
	DBusMessageIter iter_data;
	DBusMessageIter image_array;
	DBusMessageIter image_struct;
	ERR_FAIL_COND_V(!dbus_message_iter_open_container(p_iter, DBUS_TYPE_VARIANT, "a(iiay)", &iter_data), FALSE);
	ERR_FAIL_COND_V(!dbus_message_iter_open_container(&iter_data, DBUS_TYPE_ARRAY, "(iiay)", &image_array), FALSE);
	ERR_FAIL_COND_V(!dbus_message_iter_open_container(&image_array, DBUS_TYPE_STRUCT, nullptr, &image_struct), FALSE);

	// Image size
	ERR_FAIL_COND_V(!dbus_message_iter_append_basic(&image_struct, DBUS_TYPE_INT32, &p_size.width), FALSE);
	ERR_FAIL_COND_V(!dbus_message_iter_append_basic(&image_struct, DBUS_TYPE_INT32, &p_size.height), FALSE);

	DBusMessageIter image_data_array;
	ERR_FAIL_COND_V(!dbus_message_iter_open_container(&image_struct, DBUS_TYPE_ARRAY, "y", &image_data_array), FALSE);

	// Image data
	const char *data_ptr = (const char *)p_data.ptr();
	ERR_FAIL_COND_V(!dbus_message_iter_append_fixed_array(&image_data_array, DBUS_TYPE_BYTE, &data_ptr, p_data.size()), FALSE);

	ERR_FAIL_COND_V(!dbus_message_iter_close_container(&image_struct, &image_data_array), FALSE);
	ERR_FAIL_COND_V(!dbus_message_iter_close_container(&image_array, &image_struct), FALSE);
	ERR_FAIL_COND_V(!dbus_message_iter_close_container(&iter_data, &image_array), FALSE);
	ERR_FAIL_COND_V(!dbus_message_iter_close_container(p_iter, &iter_data), FALSE);

	return TRUE;
}

dbus_bool_t FreeDesktopPortalDesktop::_dbus_messsage_iter_append_basic_variant(DBusMessageIter *p_iter, int p_type, const void *p_value) {
	ERR_FAIL_COND_V(!dbus_type_is_basic(p_type), FALSE);

	char content_signature[2] = { (char)p_type, 0 };

	const void *data = p_value;

	if (!dbus_type_is_fixed(p_type)) {
		// Non-fixed types are string-like and as such are already a pointer by
		// themselves. To avoid going insane, we'll detect that and make the passed
		// value a pointer-to-a-pointer ourselves.
		data = &p_value;
	}

	DBusMessageIter variant_data;

	ERR_FAIL_COND_V(!dbus_message_iter_open_container(p_iter, DBUS_TYPE_VARIANT, content_signature, &variant_data), FALSE);
	ERR_FAIL_COND_V(!dbus_message_iter_append_basic(&variant_data, p_type, data), FALSE);
	ERR_FAIL_COND_V(!dbus_message_iter_close_container(p_iter, &variant_data), FALSE);

	return TRUE;
}

dbus_bool_t FreeDesktopPortalDesktop::_dbus_message_iter_append_bool_variant(DBusMessageIter *p_iter, bool p_bool) {
	dbus_bool_t value = p_bool ? TRUE : FALSE;
	return _dbus_messsage_iter_append_basic_variant(p_iter, DBUS_TYPE_BOOLEAN, &value);
}

dbus_bool_t FreeDesktopPortalDesktop::_dbus_message_iter_append_uint32_variant(DBusMessageIter *p_iter, uint32_t p_uint32) {
	dbus_uint32_t value = p_uint32;
	return _dbus_messsage_iter_append_basic_variant(p_iter, DBUS_TYPE_BOOLEAN, &value);
}

DBusHandlerResult FreeDesktopPortalDesktop::_handle_message(DBusConnection *connection, DBusMessage *message, void *user_data) {
	FreeDesktopPortalDesktop *portal = (FreeDesktopPortalDesktop *)user_data;

	if (dbus_message_is_signal(message, BUS_INTERFACE_STATUS_NOTIFIER_WATCHER, "StatusNotifierHostRegistered")) {
		// Looks like a new status bar is in town. We'll re-register all indicators to let it know them.
		for (KeyValue<DisplayServer::IndicatorID, StatusNotifierItem> &kv : portal->indicators) {
			portal->indicator_register(kv.key);
		}
		return DBUS_HANDLER_RESULT_HANDLED;
	}

	if (dbus_message_is_signal(message, "org.freedesktop.portal.Settings", "SettingChanged")) {
		DBusMessageIter iter;
		if (dbus_message_iter_init(message, &iter)) {
			const char *value;
			dbus_message_iter_get_basic(&iter, &value);
			String name_space = String::utf8(value);
			dbus_message_iter_next(&iter);
			dbus_message_iter_get_basic(&iter, &value);
			String key = String::utf8(value);

			if (name_space == "org.freedesktop.appearance" && key == "color-scheme") {
				callable_mp(portal, &FreeDesktopPortalDesktop::_system_theme_changed_callback).call_deferred();
				return DBUS_HANDLER_RESULT_HANDLED;
			}
		}
	}

	if (dbus_message_is_signal(message, "org.freedesktop.portal.Request", "Response")) {
		String path = String::utf8(dbus_message_get_path(message));
		MutexLock lock(portal->dbus_mutex);
		for (int i = 0; i < portal->file_dialogs.size(); i++) {
			FreeDesktopPortalDesktop::FileDialogData &fd = portal->file_dialogs.write[i];
			if (fd.path == path) {
				DBusMessageIter iter;
				if (dbus_message_iter_init(message, &iter)) {
					bool cancel = false;
					Vector<String> uris;
					Dictionary options;
					int index = 0;
					file_chooser_parse_response(&iter, fd.filter_names, cancel, uris, index, options);

					if (fd.callback.is_valid()) {
						FileDialogCallback cb;
						cb.callback = fd.callback;
						cb.status = !cancel;
						cb.files = uris;
						cb.index = index;
						cb.options = options;
						cb.opt_in_cb = fd.opt_in_cb;
						portal->pending_cbs.push_back(cb);
					}

					if (fd.prev_focus != DisplayServer::INVALID_WINDOW_ID) {
						callable_mp(DisplayServer::get_singleton(), &DisplayServer::window_move_to_foreground).call_deferred(fd.prev_focus);
					}
				}

				DBusError err;
				dbus_error_init(&err);
				dbus_bus_remove_match(portal->monitor_connection, fd.filter.utf8(), &err);
				dbus_error_free(&err);

				portal->file_dialogs.remove_at(i);
				break;
			}
		}
		return DBUS_HANDLER_RESULT_HANDLED;
	}

	return DBUS_HANDLER_RESULT_NOT_YET_HANDLED;
}

dbus_bool_t FreeDesktopPortalDesktop::_handle_add_watch(DBusWatch *watch, void *data) {
	FreeDesktopPortalDesktop *portal = (FreeDesktopPortalDesktop *)data;
	portal->dbus_watches.append(watch);

	return TRUE;
}

void FreeDesktopPortalDesktop::_handle_remove_watch(DBusWatch *watch, void *data) {
	FreeDesktopPortalDesktop *portal = (FreeDesktopPortalDesktop *)data;
	portal->dbus_watches.erase(watch);
}

void FreeDesktopPortalDesktop::_handle_watch_toggled(DBusWatch *watch, void *data) {
}

void FreeDesktopPortalDesktop::process_file_dialog_callbacks() {
	MutexLock lock(dbus_mutex);
	while (!pending_cbs.is_empty()) {
		FileDialogCallback cb = pending_cbs.front()->get();
		pending_cbs.pop_front();

		if (cb.opt_in_cb) {
			Variant ret;
			Callable::CallError ce;
			const Variant *args[4] = { &cb.status, &cb.files, &cb.index, &cb.options };

			cb.callback.callp(args, 4, ret, ce);
			if (ce.error != Callable::CallError::CALL_OK) {
				ERR_PRINT(vformat("Failed to execute file dialog callback: %s.", Variant::get_callable_error_text(cb.callback, args, 4, ce)));
			}
		} else {
			Variant ret;
			Callable::CallError ce;
			const Variant *args[3] = { &cb.status, &cb.files, &cb.index };

			cb.callback.callp(args, 3, ret, ce);
			if (ce.error != Callable::CallError::CALL_OK) {
				ERR_PRINT(vformat("Failed to execute file dialog callback: %s.", Variant::get_callable_error_text(cb.callback, args, 3, ce)));
			}
		}
	}
}

void FreeDesktopPortalDesktop::_thread_monitor(void *p_ud) {
	FreeDesktopPortalDesktop *portal = (FreeDesktopPortalDesktop *)p_ud;

	while (true) {
		LocalVector<struct pollfd> fds;

		{
			MutexLock mutex_lock(portal->dbus_mutex);

			for (DBusWatch *watch : portal->dbus_watches) {
				struct pollfd fd = {
					dbus_watch_get_unix_fd(watch), // fd
					POLLERR | POLLHUP, // events
					0, // revents
				};

				DBusWatchFlags watch_flags = (DBusWatchFlags)dbus_watch_get_flags(watch);

				if (watch_flags & DBUS_WATCH_READABLE) {
					fd.events |= POLLIN;
				}

				if (watch_flags & DBUS_WATCH_WRITABLE) {
					fd.events |= POLLOUT;
				}

				fds.push_back(fd);
			}
		}

		if (fds.ptr() == nullptr || fds.size() == 0) {
			continue;
		}

		poll(fds.ptr(), fds.size(), -1);

		for (unsigned int i = 0; i < fds.size(); ++i) {
			struct pollfd &fd = fds[i];
			if (fd.revents) {
				DBusWatchFlags flags = (DBusWatchFlags)0;

				if (fd.events & POLLIN) {
					flags = (DBusWatchFlags)(flags | (int)DBUS_WATCH_READABLE);
				}

				if (fd.events & POLLOUT) {
					flags = (DBusWatchFlags)(flags | (int)DBUS_WATCH_WRITABLE);
				}

				// Sweet lord, I can't think of a better way of associating an fd with a
				// watch.
				dbus_watch_handle(portal->dbus_watches[i], flags);
			}
		}

		MutexLock mutex_lock(portal->dbus_mutex);

		while (dbus_connection_get_dispatch_status(portal->monitor_connection) == DBUS_DISPATCH_DATA_REMAINS) {
			dbus_connection_dispatch(portal->monitor_connection);
		}

		if (!dbus_connection_get_is_connected(portal->monitor_connection) || portal->monitor_thread_abort.is_set()) {
			break;
		}
	}
}

void FreeDesktopPortalDesktop::_dbus_connection_reply_error(DBusConnection *p_connection, DBusMessage *p_message, String p_error_message) {
	ERR_PRINT(p_error_message);

	DBusMessage *reply = dbus_message_new_error(p_message, DBUS_ERROR_UNKNOWN_PROPERTY, p_error_message.utf8());

	dbus_uint32_t serial = dbus_message_get_serial(p_message);
	dbus_connection_send(p_connection, reply, &serial);

	dbus_message_unref(reply);
}

void FreeDesktopPortalDesktop::_system_theme_changed_callback() {
	if (system_theme_changed.is_valid()) {
		Variant ret;
		Callable::CallError ce;
		system_theme_changed.callp(nullptr, 0, ret, ce);
		if (ce.error != Callable::CallError::CALL_OK) {
			ERR_PRINT(vformat("Failed to execute system theme changed callback: %s.", Variant::get_callable_error_text(system_theme_changed, nullptr, 0, ce)));
		}
	}
}

void FreeDesktopPortalDesktop::_status_notifier_item_unregister(DBusConnection *connection, void *user_data) {
}

DBusHandlerResult FreeDesktopPortalDesktop::_status_notifier_item_handle_message(DBusConnection *connection, DBusMessage *message, void *user_data) {
	FreeDesktopPortalDesktop *portal = (FreeDesktopPortalDesktop *)user_data;
	ERR_FAIL_NULL_V(portal, DBUS_HANDLER_RESULT_HANDLED);

	String interface = dbus_message_get_interface(message);
	String member = dbus_message_get_member(message);

	const char *destination_ptr = dbus_message_get_destination(message);
	ERR_FAIL_NULL_V(destination_ptr, DBUS_HANDLER_RESULT_HANDLED);

	String destination;
	destination.parse_utf8(destination_ptr);

	ERR_FAIL_COND_V(!portal->indicator_id_map.has(destination), DBUS_HANDLER_RESULT_HANDLED);
	DisplayServer::IndicatorID id = portal->indicator_id_map[destination];

	ERR_FAIL_COND_V(!portal->indicators.has(id), DBUS_HANDLER_RESULT_HANDLED);
	StatusNotifierItem &item = portal->indicators[id];

	if (dbus_message_is_method_call(message, BUS_INTERFACE_STATUS_NOTIFIER_ITEM, "Activate")) {
		if (item.activate_callback.is_valid()) {
			dbus_int32_t x = 0;
			dbus_int32_t y = 0;

			DBusError error;
			dbus_error_init(&error);
			dbus_message_get_args(message, &error, DBUS_TYPE_INT32, &x, DBUS_TYPE_INT32, &y, DBUS_TYPE_INVALID);
			if (dbus_error_is_set(&error)) {
				_dbus_connection_reply_error(connection, message, vformat("Invalid arguments for %s.%s at %s", interface, member, destination));
				dbus_error_free(&error);
				return DBUS_HANDLER_RESULT_HANDLED;
			}

			item.activate_callback.call_deferred(MouseButton::LEFT, Point2i(x, y));
		}

		return DBUS_HANDLER_RESULT_HANDLED;
	}

	if (dbus_message_is_method_call(message, BUS_INTERFACE_STATUS_NOTIFIER_ITEM, "SecondaryActivate")) {
		if (item.activate_callback.is_valid()) {
			dbus_int32_t x = 0;
			dbus_int32_t y = 0;

			DBusError error;
			dbus_error_init(&error);
			dbus_message_get_args(message, &error, DBUS_TYPE_INT32, &x, DBUS_TYPE_INT32, &y, DBUS_TYPE_INVALID);
			if (dbus_error_is_set(&error)) {
				_dbus_connection_reply_error(connection, message, vformat("Invalid arguments for %s.%s at %s", interface, member, destination));
				dbus_error_free(&error);
				return DBUS_HANDLER_RESULT_HANDLED;
			}

			item.activate_callback.call_deferred(MouseButton::MIDDLE, Point2i(x, y));
		}

		return DBUS_HANDLER_RESULT_HANDLED;
	}

	if (dbus_message_get_type(message) == DBUS_MESSAGE_TYPE_METHOD_CALL && interface == BUS_INTERFACE_PROPERTIES && member == "Get") {
		const char *interface_name;
		const char *property;

		DBusError error;
		dbus_error_init(&error);
		dbus_message_get_args(message, &error, DBUS_TYPE_STRING, &interface_name, DBUS_TYPE_STRING, &property, DBUS_TYPE_INVALID);
		if (dbus_error_is_set(&error)) {
			_dbus_connection_reply_error(connection, message, vformat("Invalid arguments for %s.%s at %s", interface, member, destination));
			dbus_error_free(&error);
			return DBUS_HANDLER_RESULT_HANDLED;
		}

		if (strcmp(interface_name, BUS_STATUS_NOTIFIER_ITEM_NAME) == 0) {
			DBusMessage *reply = dbus_message_new_method_return(message);

			if (!reply) {
				_dbus_connection_reply_error(connection, message, "Failed to build response");
				return DBUS_HANDLER_RESULT_HANDLED;
			}

			DBusMessageIter reply_iter;
			dbus_message_iter_init_append(reply, &reply_iter);

			dbus_bool_t success = FALSE;
			if (strcmp(property, "Category") == 0) {
				success = _dbus_messsage_iter_append_basic_variant(&reply_iter, DBUS_TYPE_STRING, "ApplicationStatus");
			} else if (strcmp(property, "Id") == 0) {
				// TODO: No idea what would be a good value.
				// The documentation says that this is pretty much an app id.
				success = _dbus_messsage_iter_append_basic_variant(&reply_iter, DBUS_TYPE_STRING, "org.godotengine.Godot");
			} else if (strcmp(property, "Title") == 0) {
				// TODO: No idea what would be a good value.
				// Same thing as above, although it's meant to be a little bit more verbose.
				success = _dbus_messsage_iter_append_basic_variant(&reply_iter, DBUS_TYPE_STRING, "Godot game engine");
			} else if (strcmp(property, "Status") == 0) {
				success = _dbus_messsage_iter_append_basic_variant(&reply_iter, DBUS_TYPE_STRING, "Active");
			} else if (strcmp(property, "WindowId") == 0) {
				success = _dbus_message_iter_append_uint32_variant(&reply_iter, 0);
			} else if (strcmp(property, "IconName") == 0) {
				success = _dbus_messsage_iter_append_basic_variant(&reply_iter, DBUS_TYPE_STRING, "");
			} else if (strcmp(property, "IconPixmap") == 0) {
				success = _dbus_message_iter_append_pixmap(&reply_iter, item.icon_size, item.icon_data);
			} else if (strcmp(property, "OverlayIconName") == 0) {
				success = _dbus_messsage_iter_append_basic_variant(&reply_iter, DBUS_TYPE_STRING, "");
			} else if (strcmp(property, "OverlayIconPixmap") == 0) {
				success = _dbus_message_iter_append_pixmap(&reply_iter, Size2i(), Vector<uint8_t>());
			} else if (strcmp(property, "AttentionIconName") == 0) {
				success = _dbus_messsage_iter_append_basic_variant(&reply_iter, DBUS_TYPE_STRING, "");
			} else if (strcmp(property, "AttentionIconPixmap") == 0) {
				success = _dbus_message_iter_append_pixmap(&reply_iter, Size2i(), Vector<uint8_t>());
			} else if (strcmp(property, "AttentionMovieName") == 0) {
				success = _dbus_messsage_iter_append_basic_variant(&reply_iter, DBUS_TYPE_STRING, "");
			} else if (strcmp(property, "ToolTip") == 0) {
				success = _dbus_messsage_iter_append_basic_variant(&reply_iter, DBUS_TYPE_STRING, item.tooltip.utf8());
			} else if (strcmp(property, "ItemIsMenu") == 0) {
				success = _dbus_message_iter_append_bool_variant(&reply_iter, false);
			} else if (strcmp(property, "Menu") == 0) {
				success = _dbus_messsage_iter_append_basic_variant(&reply_iter, DBUS_TYPE_OBJECT_PATH, "/");
			} else {
				_dbus_connection_reply_error(connection, message, vformat("Property %s not found.", property));
				return DBUS_HANDLER_RESULT_HANDLED;
			}

			if (!success) {
				_dbus_connection_reply_error(connection, message, "Failed to build response");
				return DBUS_HANDLER_RESULT_HANDLED;
			}

			dbus_uint32_t serial = dbus_message_get_serial(message);
			dbus_connection_send(connection, reply, &serial);

			dbus_message_unref(reply);

			return DBUS_HANDLER_RESULT_HANDLED;
		}
	}

	return DBUS_HANDLER_RESULT_NOT_YET_HANDLED;
}

bool FreeDesktopPortalDesktop::indicator_register(DisplayServer::IndicatorID p_id) {
	MutexLock mutex_lock(dbus_mutex);

	ERR_FAIL_COND_V(!indicators.has(p_id), false);

	StatusNotifierItem &item = indicators[p_id];

	const char *item_name = item.name.utf8();

	DBusMessage *message = dbus_message_new_method_call(BUS_STATUS_NOTIFIER_WATCHER_NAME, BUS_STATUS_NOTIFIER_WATCHER_PATH, BUS_INTERFACE_STATUS_NOTIFIER_WATCHER, "RegisterStatusNotifierItem");
	dbus_message_append_args(message, DBUS_TYPE_STRING, &item_name, DBUS_TYPE_INVALID);

	dbus_connection_send(monitor_connection, message, NULL);
	dbus_message_unref(message);

	return true;
}

bool FreeDesktopPortalDesktop::indicator_create(DisplayServer::IndicatorID p_id, const Ref<Texture2D> &p_icon) {
	MutexLock mutex_lock(dbus_mutex);

	ERR_FAIL_COND_V(indicators.has(p_id), false);

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

	int pid = OS::get_singleton()->get_process_id();

	String name = vformat("org.freedesktop.StatusNotifierItem-%d-%d", pid, p_id);
	dbus_bus_request_name(bus, name.utf8(), DBUS_NAME_FLAG_DO_NOT_QUEUE, &error);
	if (dbus_error_is_set(&error)) {
		dbus_error_free(&error);
		return false;
	}

	indicator_id_map[name] = p_id;

	StatusNotifierItem &item = indicators[p_id];
	item.id = p_id;
	item.name = name;

	// TODO: Error handling.
	indicator_set_icon(p_id, p_icon);
	indicator_register(p_id);

	return true;
}

Error FreeDesktopPortalDesktop::indicator_set_icon(DisplayServer::IndicatorID p_id, const Ref<Texture2D> &p_icon) {
	MutexLock mutex_lock(dbus_mutex);

	ERR_FAIL_COND_V(!indicators.has(p_id), ERR_UNCONFIGURED);
	ERR_FAIL_COND_V(p_icon->get_size().x == 0, ERR_INVALID_PARAMETER);
	ERR_FAIL_COND_V(p_icon->get_size().y == 0, ERR_INVALID_PARAMETER);
	ERR_FAIL_COND_V(p_icon.is_null(), FAILED);

	// We'll have to manipulate the icon a bit.
	Ref<Image> image = p_icon->get_image();

	if (image->is_compressed()) {
		Error err = image->decompress();
		ERR_FAIL_COND_V_MSG(err != OK, err, "Couldn't decompress VRAM-compressed status-icon. Switch to a lossless compression mode in the Import dock.");
	}

	// TODO: Figure out the size limits (at least swaybar errors on big pixmaps)
	image->resize(64, 64);

	// The API requires ARGB32 (or ARGB8888), which `Image` does not support.
	// Luckily, we can convert to RGBA8 (which is actually RGBA8888), so we'll just
	// need to reorder the bytes ourselves.
	image->convert(Image::FORMAT_RGBA8);

	Vector<uint8_t> image_data = image->get_data();

	StatusNotifierItem &item = indicators[p_id];
	item.icon_size = image->get_size();

	item.icon_data.clear();
	for (int i = 0; i < image_data.size(); i += 4) {
		item.icon_data.append(image_data[i + 3]); // A
		item.icon_data.append(image_data[i + 0]); // R
		item.icon_data.append(image_data[i + 1]); // G
		item.icon_data.append(image_data[i + 2]); // B
	}

	DBusMessage *signal = dbus_message_new_signal(BUS_STATUS_NOTIFIER_ITEM_PATH, BUS_INTERFACE_STATUS_NOTIFIER_ITEM, "NewIcon");
	dbus_connection_send(monitor_connection, signal, NULL);
	dbus_message_unref(signal);

	return OK;
}

void FreeDesktopPortalDesktop::indicator_set_tooltip(DisplayServer::IndicatorID p_id, const String &p_tooltip) {
	MutexLock mutex_lock(dbus_mutex);

	ERR_FAIL_COND(!indicators.has(p_id));

	StatusNotifierItem &item = indicators[p_id];

	item.tooltip = p_tooltip;

	DBusMessage *signal = dbus_message_new_signal(BUS_STATUS_NOTIFIER_ITEM_PATH, BUS_INTERFACE_STATUS_NOTIFIER_ITEM, "NewToolTip");
	dbus_connection_send(monitor_connection, signal, NULL);
	dbus_message_unref(signal);
}

void FreeDesktopPortalDesktop::indicator_set_callback(DisplayServer::IndicatorID p_id, const Callable &p_callback) {
	MutexLock mutex_lock(dbus_mutex);

	ERR_FAIL_COND(!indicators.has(p_id));

	StatusNotifierItem &item = indicators[p_id];

	item.activate_callback = p_callback;
}

void FreeDesktopPortalDesktop::indicator_destroy(DisplayServer::IndicatorID p_id) {
	ERR_FAIL_COND(!indicators.has(p_id));

	StatusNotifierItem &item = indicators[p_id];

	DBusError error;
	dbus_error_init(&error);

	dbus_bus_release_name(monitor_connection, item.name.utf8(), &error);

	dbus_error_free(&error);

	indicator_id_map.erase(item.name);
	indicators.erase(p_id);
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

	DBusError err;
	dbus_error_init(&err);
	monitor_connection = dbus_bus_get(DBUS_BUS_SESSION, &err);
	if (dbus_error_is_set(&err)) {
		dbus_error_free(&err);
	} else {
		theme_path = "type='signal',sender='org.freedesktop.portal.Desktop',interface='org.freedesktop.portal.Settings',member='SettingChanged'";
		dbus_bus_add_match(monitor_connection, theme_path.utf8().get_data(), &err);
		if (dbus_error_is_set(&err)) {
			dbus_error_free(&err);
			dbus_connection_unref(monitor_connection);
			monitor_connection = nullptr;
		}
	}

	if (!unsupported) {
		dbus_threads_init_default();
		dbus_connection_add_filter(monitor_connection, _handle_message, this, _dbus_arg_noop);

		dbus_connection_set_watch_functions(monitor_connection, _handle_add_watch, _handle_remove_watch, _handle_watch_toggled, this, _dbus_arg_noop);

		monitor_thread_abort.clear();
		monitor_thread.start(FreeDesktopPortalDesktop::_thread_monitor, this);

		DBusError error;
		dbus_error_init(&error);

		dbus_connection_try_register_object_path(monitor_connection, BUS_STATUS_NOTIFIER_ITEM_PATH, &status_indicator_item_vtable, this, &error);
		if (dbus_error_is_set(&error)) {
			if (OS::get_singleton()->is_stdout_verbose()) {
				ERR_PRINT(vformat("Error on D-Bus communication: %s", error.message));
			}
		}
		dbus_error_free(&error);

		// We need this for catching status bar reloads (e.g. swaybar restarts)
		const char *signal_match = "type='signal',interface='org.freedesktop.StatusNotifierWatcher',member='StatusNotifierHostRegistered'";
		dbus_bus_add_match(monitor_connection, signal_match, &error);
		if (dbus_error_is_set(&error)) {
			if (OS::get_singleton()->is_stdout_verbose()) {
				ERR_PRINT(vformat("Error on D-Bus communication: %s", error.message));
			}
			dbus_error_free(&error);
		}
	}
}

FreeDesktopPortalDesktop::~FreeDesktopPortalDesktop() {
	monitor_thread_abort.set();
	if (monitor_thread.is_started()) {
		monitor_thread.wait_to_finish();
	}

	if (monitor_connection) {
		DBusError err;
		for (FreeDesktopPortalDesktop::FileDialogData &fd : file_dialogs) {
			dbus_error_init(&err);
			dbus_bus_remove_match(monitor_connection, fd.filter.utf8().get_data(), &err);
			dbus_error_free(&err);
		}
		dbus_error_init(&err);
		dbus_bus_remove_match(monitor_connection, theme_path.utf8().get_data(), &err);
		dbus_error_free(&err);
		dbus_connection_unref(monitor_connection);
	}
}

#endif // DBUS_ENABLED
