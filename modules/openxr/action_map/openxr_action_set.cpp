/*************************************************************************/
/*  openxr_action_set.cpp                                                */
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

#include "openxr_action_set.h"

void OpenXRActionSet::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_localized_name", "localized_name"), &OpenXRActionSet::set_localized_name);
	ClassDB::bind_method(D_METHOD("get_localized_name"), &OpenXRActionSet::get_localized_name);
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "localized_name"), "set_localized_name", "get_localized_name");

	ClassDB::bind_method(D_METHOD("set_priority", "priority"), &OpenXRActionSet::set_priority);
	ClassDB::bind_method(D_METHOD("get_priority"), &OpenXRActionSet::get_priority);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "priority"), "set_priority", "get_priority");

	ClassDB::bind_method(D_METHOD("set_actions", "actions"), &OpenXRActionSet::set_actions);
	ClassDB::bind_method(D_METHOD("get_actions"), &OpenXRActionSet::get_actions);
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "actions", PROPERTY_HINT_RESOURCE_TYPE, "OpenXRAction", PROPERTY_USAGE_NO_EDITOR), "set_actions", "get_actions");

	ClassDB::bind_method(D_METHOD("add_action", "action"), &OpenXRActionSet::add_action);
	ClassDB::bind_method(D_METHOD("remove_action", "action"), &OpenXRActionSet::remove_action);
}

Ref<OpenXRActionSet> OpenXRActionSet::new_action_set(const char *p_name, const char *p_localized_name, const int p_priority) {
	// This is a helper function to help build our default action sets

	Ref<OpenXRActionSet> action_set;
	action_set.instantiate();
	action_set->set_name(String(p_name));
	action_set->set_localized_name(p_localized_name);
	action_set->set_priority(p_priority);

	return action_set;
}

void OpenXRActionSet::set_localized_name(const String p_localized_name) {
	localized_name = p_localized_name;
}

String OpenXRActionSet::get_localized_name() const {
	return localized_name;
}

void OpenXRActionSet::set_priority(const int p_priority) {
	priority = p_priority;
}

int OpenXRActionSet::get_priority() const {
	return priority;
}

void OpenXRActionSet::set_actions(Array p_actions) {
	actions = p_actions;
}

Array OpenXRActionSet::get_actions() const {
	return actions;
}

void OpenXRActionSet::add_action(Ref<OpenXRAction> p_action) {
	ERR_FAIL_COND(p_action.is_null());

	if (actions.find(p_action) == -1) {
		actions.push_back(p_action);
	}
}

void OpenXRActionSet::remove_action(Ref<OpenXRAction> p_action) {
	int idx = actions.find(p_action);
	if (idx != -1) {
		actions.remove_at(idx);
	}
}

Ref<OpenXRAction> OpenXRActionSet::add_new_action(const char *p_name, const char *p_localized_name, const OpenXRAction::ActionType p_action_type, const char *p_toplevel_paths) {
	// This is a helper function to help build our default action sets

	Ref<OpenXRAction> new_action = OpenXRAction::new_action(p_name, p_localized_name, p_action_type, p_toplevel_paths);
	add_action(new_action);
	return new_action;
}

OpenXRActionSet::~OpenXRActionSet() {
	actions.clear();
}
