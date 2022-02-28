/*************************************************************************/
/*  accessibility_server.cpp                                             */
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

#include "servers/accessibility_server.h"

AccessibilityServerManager *AccessibilityServerManager::singleton = nullptr;

void AccessibilityServerManager::_bind_methods() {
	ClassDB::bind_method(D_METHOD("add_interface", "interface"), &AccessibilityServerManager::add_interface);
	ClassDB::bind_method(D_METHOD("get_interface_count"), &AccessibilityServerManager::get_interface_count);
	ClassDB::bind_method(D_METHOD("remove_interface", "interface"), &AccessibilityServerManager::remove_interface);
	ClassDB::bind_method(D_METHOD("get_interface", "idx"), &AccessibilityServerManager::get_interface);
	ClassDB::bind_method(D_METHOD("get_interfaces"), &AccessibilityServerManager::get_interfaces);
	ClassDB::bind_method(D_METHOD("find_interface", "name"), &AccessibilityServerManager::find_interface);

	ClassDB::bind_method(D_METHOD("set_primary_interface", "index"), &AccessibilityServerManager::set_primary_interface);
	ClassDB::bind_method(D_METHOD("get_primary_interface"), &AccessibilityServerManager::get_primary_interface);

	ADD_SIGNAL(MethodInfo("interface_added", PropertyInfo(Variant::STRING_NAME, "interface_name")));
	ADD_SIGNAL(MethodInfo("interface_removed", PropertyInfo(Variant::STRING_NAME, "interface_name")));
}

void AccessibilityServerManager::add_interface(const Ref<AccessibilityServer> &p_interface) {
	ERR_FAIL_COND(p_interface.is_null());

	for (int i = 0; i < interfaces.size(); i++) {
		if (interfaces[i] == p_interface) {
			ERR_PRINT("AccessibilityServer: Interface was already added.");
			return;
		};
	};

	interfaces.push_back(p_interface);
	print_verbose("AccessibilityServer: Added interface \"" + p_interface->get_name() + "\"");
	emit_signal(SNAME("interface_added"), p_interface->get_name());
}

void AccessibilityServerManager::remove_interface(const Ref<AccessibilityServer> &p_interface) {
	ERR_FAIL_COND(p_interface.is_null());
	ERR_FAIL_COND_MSG(p_interface == primary_interface, "AccessibilityServer: Can't remove primary interface.");

	int idx = -1;
	for (int i = 0; i < interfaces.size(); i++) {
		if (interfaces[i] == p_interface) {
			idx = i;
			break;
		};
	};

	ERR_FAIL_COND(idx == -1);
	print_verbose("AccessibilityServer: Removed interface \"" + p_interface->get_name() + "\"");
	emit_signal(SNAME("interface_removed"), p_interface->get_name());
	interfaces.remove_at(idx);
}

int AccessibilityServerManager::get_interface_count() const {
	return interfaces.size();
}

Ref<AccessibilityServer> AccessibilityServerManager::get_interface(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, interfaces.size(), nullptr);
	return interfaces[p_index];
}

Ref<AccessibilityServer> AccessibilityServerManager::find_interface(const String &p_name) const {
	int idx = -1;
	for (int i = 0; i < interfaces.size(); i++) {
		if (interfaces[i]->get_name() == p_name) {
			idx = i;
			break;
		};
	};

	ERR_FAIL_COND_V(idx == -1, nullptr);
	return interfaces[idx];
}

Array AccessibilityServerManager::get_interfaces() const {
	Array ret;

	for (int i = 0; i < interfaces.size(); i++) {
		Dictionary iface_info;

		iface_info["id"] = i;
		iface_info["name"] = interfaces[i]->get_name();

		ret.push_back(iface_info);
	};

	return ret;
}

void AccessibilityServerManager::set_primary_interface(const Ref<AccessibilityServer> &p_primary_interface) {
	if (p_primary_interface.is_null()) {
		print_verbose("AccessibilityServer: Clearing primary interface");
		primary_interface.unref();
	} else {
		primary_interface = p_primary_interface;
		print_verbose("AccessibilityServer: Primary interface set to: \"" + primary_interface->get_name() + "\".");

		if (OS::get_singleton()->get_main_loop()) {
			OS::get_singleton()->get_main_loop()->notification(MainLoop::NOTIFICATION_TEXT_SERVER_CHANGED);
		}
	}
}

AccessibilityServerManager::AccessibilityServerManager() {
	singleton = this;
}

AccessibilityServerManager::~AccessibilityServerManager() {
	if (primary_interface.is_valid()) {
		primary_interface.unref();
	}
	while (interfaces.size() > 0) {
		interfaces.remove_at(0);
	}
	singleton = nullptr;
}

/*************************************************************************/

void AccessibilityServer::_bind_methods() {
	ClassDB::bind_method(D_METHOD("has_feature", "feature"), &AccessibilityServer::has_feature);
	ClassDB::bind_method(D_METHOD("get_name"), &AccessibilityServer::get_name);
	ClassDB::bind_method(D_METHOD("get_features"), &AccessibilityServer::get_features);

	ClassDB::bind_method(D_METHOD("create_window_context", "window"), &AccessibilityServer::create_window_context);
	ClassDB::bind_method(D_METHOD("destroy_window_context", "window"), &AccessibilityServer::destroy_window_context);

	ClassDB::bind_method(D_METHOD("post_tree_update", "update_data", "kdb_focused", "mouse_focus", "window"), &AccessibilityServer::post_tree_update);

	/* Role */
	BIND_ENUM_CONSTANT(ROLE_UNKNOWN);
	BIND_ENUM_CONSTANT(ROLE_BUTTON);
	BIND_ENUM_CONSTANT(ROLE_CHECK_BOX);
	BIND_ENUM_CONSTANT(ROLE_CHECK_BUTTON); // AccessKit ToggleButton
	BIND_ENUM_CONSTANT(ROLE_CONTAINER); // AccessKit GenericContainer
	BIND_ENUM_CONSTANT(ROLE_ITEM_LIST); // AccessKit ListBox ? (or List ?)
	BIND_ENUM_CONSTANT(ROLE_ITEM_LIST_ITEM); // AccessKit ListBoxOption? (or ListItem ?)
	BIND_ENUM_CONSTANT(ROLE_LABEL);
	BIND_ENUM_CONSTANT(ROLE_LINE_EDIT); // AccessKit TextBox
	BIND_ENUM_CONSTANT(ROLE_LINK);
	BIND_ENUM_CONSTANT(ROLE_MENU_BUTTON); // AccessKit PopupButton ?
	BIND_ENUM_CONSTANT(ROLE_OPTION_BUTTON); // AccessKit ComboBoxMenuButton ?
	BIND_ENUM_CONSTANT(ROLE_PROGRESS_BAR); // AccessKit ProgressIndicator
	BIND_ENUM_CONSTANT(ROLE_RICH_TEXT); // AccessKit Documnet ? (or StaticText)
	BIND_ENUM_CONSTANT(ROLE_SCROLL_BAR);
	BIND_ENUM_CONSTANT(ROLE_SCROLL_CONTAINER); // AccessKit ScrollView
	BIND_ENUM_CONSTANT(ROLE_SLIDER);
	BIND_ENUM_CONSTANT(ROLE_SPIN_BOX); // AccessKit SpinButton
	BIND_ENUM_CONSTANT(ROLE_TAB_BAR); // AccessKit TabList
	BIND_ENUM_CONSTANT(ROLE_TAB);
	BIND_ENUM_CONSTANT(ROLE_TEXT_EDIT); // AccessKit TextBox + multiline flag ?
	BIND_ENUM_CONSTANT(ROLE_TREE); // AccessKit Tree or TreeGrid
	BIND_ENUM_CONSTANT(ROLE_TREE_ITEM);
	BIND_ENUM_CONSTANT(ROLE_WINDOW);
	//TODO add other roles

	/* NodeRelation */
	BIND_ENUM_CONSTANT(NODE_RELATION_INDIRECT_CHILD);
	BIND_ENUM_CONSTANT(NODE_RELATION_ERROR_MESSAGE);
	BIND_ENUM_CONSTANT(NODE_RELATION_LINK_TARGET);
	BIND_ENUM_CONSTANT(NODE_RELATION_GROUP_MEMBER);
	BIND_ENUM_CONSTANT(NODE_RELATION_POPUP);
	BIND_ENUM_CONSTANT(NODE_RELATION_CONTROLS);
	BIND_ENUM_CONSTANT(NODE_RELATION_DETAILS);
	BIND_ENUM_CONSTANT(NODE_RELATION_DESCRIBED);
	BIND_ENUM_CONSTANT(NODE_RELATION_FLOW_TO);
	BIND_ENUM_CONSTANT(NODE_RELATION_LABELLED);
	BIND_ENUM_CONSTANT(NODE_RELATION_RADIO_GROUP_MEMBER);

	/* Feature */
	BIND_ENUM_CONSTANT(FEATURE_SCREEN_READER_SUPPORT);
	BIND_ENUM_CONSTANT(FEATURE_USE_NATIVE_CB);
	BIND_ENUM_CONSTANT(FEATURE_USE_TREE_UPDATES);
}
