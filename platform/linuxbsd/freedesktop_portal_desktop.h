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

#include <stdint.h>

class FreeDesktopPortalDesktop {
private:
	bool unsupported = false;

	// Read a setting from org.freekdesktop.portal.Settings
	bool read_setting(const char *p_namespace, const char *p_key, int p_type, void *r_value);

public:
	FreeDesktopPortalDesktop();

	bool is_supported() { return !unsupported; }

	// Retrieve the system's preferred color scheme.
	// 0: No preference or unknown.
	// 1: Prefer dark appearance.
	// 2: Prefer light appearance.
	uint32_t get_appearance_color_scheme();
};

#endif // DBUS_ENABLED

#endif // FREEDESKTOP_PORTAL_DESKTOP_H
