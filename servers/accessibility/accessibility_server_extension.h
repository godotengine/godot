/*************************************************************************/
/*  accessibility_server_extension.h                                     */
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

#ifndef ACCESSIBILITY_SERVER_EXTENSION_H
#define ACCESSIBILITY_SERVER_EXTENSION_H

#include "core/object/gdvirtual.gen.inc"
#include "core/object/script_language.h"
#include "core/os/thread_safe.h"
#include "core/variant/native_ptr.h"
#include "servers/accessibility_server.h"

class AccessibilityServerExtension : public AccessibilityServer {
	GDCLASS(AccessibilityServerExtension, AccessibilityServer);

protected:
	_THREAD_SAFE_CLASS_

	static void _bind_methods();

public:
	virtual bool has_feature(Feature p_feature) const override;
	virtual String get_name() const override;
	virtual int64_t get_features() const override;
	GDVIRTUAL1RC(bool, _has_feature, Feature);
	GDVIRTUAL0RC(String, _get_name);
	GDVIRTUAL0RC(int64_t, _get_features);

	virtual void create_window_context(DisplayServer::WindowID p_window, ObjectID p_root_node) override;
	virtual void destroy_window_context(DisplayServer::WindowID p_window, ObjectID p_root_node) override;
	GDVIRTUAL2(_create_window_context, DisplayServer::WindowID, ObjectID);
	GDVIRTUAL2(_destroy_window_context, DisplayServer::WindowID, ObjectID);

	virtual void post_tree_update(const PackedInt64Array &p_update_data, ObjectID p_root_node, ObjectID p_kbd_focus, ObjectID p_mouse_focus, DisplayServer::WindowID p_window) override;
	GDVIRTUAL5(_post_tree_update, const PackedInt64Array &, ObjectID, ObjectID, ObjectID, DisplayServer::WindowID);

	virtual int64_t native_window_callback(int64_t p_object, int64_t p_wparam, int64_t p_lparam, DisplayServer::WindowID p_window) override;
	GDVIRTUAL4R(int64_t, _native_window_callback, int64_t, int64_t, int64_t, DisplayServer::WindowID);
};

#endif // ACCESSIBILITY_SERVER_EXTENSION_H
