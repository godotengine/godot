/*************************************************************************/
/*  accessibility_server.h                                               */
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

#ifndef ACCESSIBILITY_SERVER_H
#define ACCESSIBILITY_SERVER_H

#include "core/object/object_id.h"
#include "core/object/ref_counted.h"
#include "core/os/os.h"
#include "core/variant/variant.h"
#include "servers/display_server.h"

class AccessibilityServer : public RefCounted {
	GDCLASS(AccessibilityServer, RefCounted);

public:
	enum Role {
		ROLE_UNKNOWN,
		ROLE_BUTTON,
		ROLE_CHECK_BOX,
		ROLE_CHECK_BUTTON, // AccessKit ToggleButton
		ROLE_CONTAINER, // AccessKit GenericContainer
		ROLE_ITEM_LIST, // AccessKit ListBox ? (or List ?)
		ROLE_ITEM_LIST_ITEM, // AccessKit ListBoxOption? (or ListItem ?)
		ROLE_LABEL,
		ROLE_LINE_EDIT, // AccessKit TextBox
		ROLE_LINK,
		ROLE_MENU_BUTTON, // AccessKit PopupButton ?
		ROLE_OPTION_BUTTON, // AccessKit ComboBoxMenuButton ?
		ROLE_PROGRESS_BAR, // AccessKit ProgressIndicator
		ROLE_RICH_TEXT, // AccessKit Documnet ? (or StaticText)
		ROLE_SCROLL_BAR,
		ROLE_SCROLL_CONTAINER, // AccessKit ScrollView
		ROLE_SLIDER,
		ROLE_SPIN_BOX, // AccessKit SpinButton
		ROLE_TAB_BAR, // AccessKit TabList
		ROLE_TAB,
		ROLE_TEXT_EDIT, // AccessKit TextBox + multiline flag ?
		ROLE_TREE, // AccessKit Tree or TreeGrid
		ROLE_TREE_ITEM,
		ROLE_WINDOW
		//TODO add other roles
	};

	enum NodeRelation {
		NODE_RELATION_INDIRECT_CHILD,
		NODE_RELATION_ERROR_MESSAGE,
		NODE_RELATION_LINK_TARGET,
		NODE_RELATION_GROUP_MEMBER,
		NODE_RELATION_POPUP,
		NODE_RELATION_CONTROLS,
		NODE_RELATION_DETAILS,
		NODE_RELATION_DESCRIBED,
		NODE_RELATION_FLOW_TO,
		NODE_RELATION_LABELLED,
		NODE_RELATION_RADIO_GROUP_MEMBER,
	};

	enum Feature {
		FEATURE_SCREEN_READER_SUPPORT = 1 << 0,
		FEATURE_USE_NATIVE_CB = 1 << 1,
		FEATURE_USE_TREE_UPDATES = 1 << 2,
	};

protected:
	static void _bind_methods();

public:
	virtual bool has_feature(Feature p_feature) const = 0;
	virtual String get_name() const = 0;
	virtual int64_t get_features() const = 0;

	virtual void create_window_context(DisplayServer::WindowID p_window, ObjectID p_root_node) = 0;
	virtual void destroy_window_context(DisplayServer::WindowID p_window, ObjectID p_root_node) = 0;

	virtual void post_tree_update(const PackedInt64Array &p_update_data, ObjectID p_root_node, ObjectID p_kbd_focus, ObjectID p_mouse_focus, DisplayServer::WindowID p_window) = 0;

	virtual int64_t native_window_callback(int64_t p_object, int64_t p_wparam, int64_t p_lparam, DisplayServer::WindowID p_window) = 0;
};

/*************************************************************************/

class AccessibilityServerManager : public Object {
	GDCLASS(AccessibilityServerManager, Object);

protected:
	static void _bind_methods();

private:
	static AccessibilityServerManager *singleton;

	Ref<AccessibilityServer> primary_interface;
	Vector<Ref<AccessibilityServer>> interfaces;

public:
	_FORCE_INLINE_ static AccessibilityServerManager *get_singleton() {
		return singleton;
	}

	void add_interface(const Ref<AccessibilityServer> &p_interface);
	void remove_interface(const Ref<AccessibilityServer> &p_interface);
	int get_interface_count() const;
	Ref<AccessibilityServer> get_interface(int p_index) const;
	Ref<AccessibilityServer> find_interface(const String &p_name) const;
	Array get_interfaces() const;

	_FORCE_INLINE_ Ref<AccessibilityServer> get_primary_interface() const {
		return primary_interface;
	}
	void set_primary_interface(const Ref<AccessibilityServer> &p_primary_interface);

	AccessibilityServerManager();
	~AccessibilityServerManager();
};

/*************************************************************************/

#define ACS AccessibilityServerManager::get_singleton()->get_primary_interface()

VARIANT_ENUM_CAST(AccessibilityServer::Role);
VARIANT_ENUM_CAST(AccessibilityServer::NodeRelation);
VARIANT_ENUM_CAST(AccessibilityServer::Feature);

#endif // ACCESSIBILITY_SERVER_H
