/**************************************************************************/
/*  openxr_user_presence_extension.cpp                                    */
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

#include "openxr_user_presence_extension.h"

#include "../openxr_interface.h"
#include "core/config/project_settings.h"

OpenXRUserPresenceExtension *OpenXRUserPresenceExtension::singleton = nullptr;

OpenXRUserPresenceExtension *OpenXRUserPresenceExtension::get_singleton() {
	return singleton;
}

OpenXRUserPresenceExtension::OpenXRUserPresenceExtension() {
	singleton = this;
}

OpenXRUserPresenceExtension::~OpenXRUserPresenceExtension() {
	singleton = nullptr;
}

HashMap<String, bool *> OpenXRUserPresenceExtension::get_requested_extensions(XrVersion p_version) {
	HashMap<String, bool *> request_extensions;
	if (GLOBAL_GET("xr/openxr/extensions/user_presence")) {
		request_extensions[XR_EXT_USER_PRESENCE_EXTENSION_NAME] = &available;
	}
	return request_extensions;
}

void *OpenXRUserPresenceExtension::set_system_properties_and_get_next_pointer(void *p_next_pointer) {
	if (!available) {
		return p_next_pointer;
	}

	properties.type = XR_TYPE_SYSTEM_USER_PRESENCE_PROPERTIES_EXT;
	properties.next = p_next_pointer;
	properties.supportsUserPresence = false;

	return &properties;
}

bool OpenXRUserPresenceExtension::is_active() const {
	return available && properties.supportsUserPresence;
}

void OpenXRUserPresenceExtension::on_state_ready() {
	user_present = true;
}

void OpenXRUserPresenceExtension::on_state_stopping() {
	user_present = false;
}

bool OpenXRUserPresenceExtension::on_event_polled(const XrEventDataBuffer &event) {
	if (!is_active() || event.type != XR_TYPE_EVENT_DATA_USER_PRESENCE_CHANGED_EXT) {
		return false;
	}

	const XrEventDataUserPresenceChangedEXT *user_presence_changed_event = (XrEventDataUserPresenceChangedEXT *)&event;
	if (user_present != (bool)user_presence_changed_event->isUserPresent) {
		user_present = user_presence_changed_event->isUserPresent;
		OpenXRInterface *xr_interface = OpenXRAPI::get_singleton()->get_xr_interface();
		if (xr_interface) {
			xr_interface->emit_signal(SNAME("user_presence_changed"), user_present);
		}
	}
	return true;
}

bool OpenXRUserPresenceExtension::is_user_present() const {
	return user_present;
}
