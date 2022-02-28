/*************************************************************************/
/*  accessibility_server_extension.cpp                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#include "accessibility_server_extension.h"

void AccessibilityServerExtension::_bind_methods() {
	GDVIRTUAL_BIND(_has_feature, "feature");
	GDVIRTUAL_BIND(_get_name);
	GDVIRTUAL_BIND(_get_features);

	GDVIRTUAL_BIND(_create_window_context, "window", "root_node");
	GDVIRTUAL_BIND(_destroy_window_context, "window", "root_node");

	GDVIRTUAL_BIND(_post_tree_update, "update_data", "root_node", "kbd_focus", "mouse_focus", "window");

	GDVIRTUAL_BIND(_native_window_callback, "object", "wparam", "lparam", "window");
}

bool AccessibilityServerExtension::has_feature(Feature p_feature) const {
	bool ret;
	if (GDVIRTUAL_CALL(_has_feature, p_feature, ret)) {
		return ret;
	}
	return false;
}

String AccessibilityServerExtension::get_name() const {
	String ret;
	if (GDVIRTUAL_CALL(_get_name, ret)) {
		return ret;
	}
	return "Unknown";
}

int64_t AccessibilityServerExtension::get_features() const {
	int64_t ret;
	if (GDVIRTUAL_CALL(_get_features, ret)) {
		return ret;
	}
	return 0;
}

void AccessibilityServerExtension::create_window_context(DisplayServer::WindowID p_window, ObjectID p_root_node) {
	GDVIRTUAL_CALL(_create_window_context, p_window, p_root_node);
}

void AccessibilityServerExtension::destroy_window_context(DisplayServer::WindowID p_window, ObjectID p_root_node) {
	GDVIRTUAL_CALL(_destroy_window_context, p_window, p_root_node);
}

void AccessibilityServerExtension::post_tree_update(const PackedInt64Array &p_update_data, ObjectID p_root_node, ObjectID p_kbd_focus, ObjectID p_mouse_focus, DisplayServer::WindowID p_window) {
	GDVIRTUAL_CALL(_post_tree_update, p_update_data, p_root_node, p_kbd_focus, p_mouse_focus, p_window);
}

int64_t AccessibilityServerExtension::native_window_callback(int64_t p_object, int64_t p_wparam, int64_t p_lparam, DisplayServer::WindowID p_window) {
	int64_t ret;
	if (GDVIRTUAL_CALL(_native_window_callback, p_object, p_wparam, p_lparam, p_window, ret)) {
		return ret;
	}
	return 0;
}
