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

#include "core/error/error_macros.h"
#include "core/os/os.h"
#include "core/string/ustring.h"

#ifdef SOWRAP_ENABLED
#include "dbus-so_wrap.h"
#else
#include <dbus/dbus.h>
#endif

#include "core/variant/variant.h"

#define BUS_OBJECT_NAME "org.freedesktop.portal.Desktop"
#define BUS_OBJECT_PATH "/org/freedesktop/portal/desktop"

#define BUS_INTERFACE_SETTINGS "org.freedesktop.portal.Settings"

static bool try_parse_variant(DBusMessage *p_reply_message, int p_type, void *r_value) {
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
		dbus_error_free(&error);
		unsupported = true;
		if (OS::get_singleton()->is_stdout_verbose()) {
			ERR_PRINT(String() + "Error opening D-Bus connection: " + error.message);
		}
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
		dbus_error_free(&error);
		dbus_connection_unref(bus);
		if (OS::get_singleton()->is_stdout_verbose()) {
			ERR_PRINT(String() + "Error on D-Bus communication: " + error.message);
		}
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
}

#endif // DBUS_ENABLED
