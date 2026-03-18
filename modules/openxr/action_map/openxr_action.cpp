/**************************************************************************/
/*  openxr_action.cpp                                                     */
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

#include "openxr_action.h"

#include "openxr_action_set.h"

void OpenXRAction::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_localized_name", "localized_name"), &OpenXRAction::set_localized_name);
	ClassDB::bind_method(D_METHOD("get_localized_name"), &OpenXRAction::get_localized_name);
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "localized_name"), "set_localized_name", "get_localized_name");

	ClassDB::bind_method(D_METHOD("set_action_type", "action_type"), &OpenXRAction::set_action_type);
	ClassDB::bind_method(D_METHOD("get_action_type"), &OpenXRAction::get_action_type);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "action_type", PROPERTY_HINT_ENUM, "bool,float,vector2,pose"), "set_action_type", "get_action_type");

	ClassDB::bind_method(D_METHOD("set_toplevel_paths", "toplevel_paths"), &OpenXRAction::set_toplevel_paths);
	ClassDB::bind_method(D_METHOD("get_toplevel_paths"), &OpenXRAction::get_toplevel_paths);
	ADD_PROPERTY(PropertyInfo(Variant::PACKED_STRING_ARRAY, "toplevel_paths"), "set_toplevel_paths", "get_toplevel_paths");

	BIND_ENUM_CONSTANT(OPENXR_ACTION_BOOL);
	BIND_ENUM_CONSTANT(OPENXR_ACTION_FLOAT);
	BIND_ENUM_CONSTANT(OPENXR_ACTION_VECTOR2);
	BIND_ENUM_CONSTANT(OPENXR_ACTION_POSE);
}

Ref<OpenXRAction> OpenXRAction::new_action(const char *p_name, const char *p_localized_name, const ActionType p_action_type, const char *p_toplevel_paths) {
	// This is a helper function to help build our default action sets

	Ref<OpenXRAction> action;
	action.instantiate();
	action->set_name(String(p_name));
	action->set_localized_name(String(p_localized_name));
	action->set_action_type(p_action_type);
	action->parse_toplevel_paths(String(p_toplevel_paths));

	return action;
}

String OpenXRAction::get_name_with_set() const {
	String action_name = get_name();

	if (action_set != nullptr) {
		action_name = action_set->get_name() + "/" + action_name;
	}

	return action_name;
}

void OpenXRAction::set_localized_name(const String p_localized_name) {
	localized_name = p_localized_name;
	emit_changed();
}

String OpenXRAction::get_localized_name() const {
	return localized_name;
}

void OpenXRAction::set_action_type(const OpenXRAction::ActionType p_action_type) {
	action_type = p_action_type;
	emit_changed();
}

OpenXRAction::ActionType OpenXRAction::get_action_type() const {
	return action_type;
}

void OpenXRAction::set_toplevel_paths(const PackedStringArray p_toplevel_paths) {
	toplevel_paths = p_toplevel_paths;
	emit_changed();
}

PackedStringArray OpenXRAction::get_toplevel_paths() const {
	return toplevel_paths;
}

void OpenXRAction::add_toplevel_path(const String p_toplevel_path) {
	if (!toplevel_paths.has(p_toplevel_path)) {
		toplevel_paths.push_back(p_toplevel_path);
		emit_changed();
	}
}

void OpenXRAction::rem_toplevel_path(const String p_toplevel_path) {
	if (toplevel_paths.has(p_toplevel_path)) {
		toplevel_paths.erase(p_toplevel_path);
		emit_changed();
	}
}

void OpenXRAction::parse_toplevel_paths(const String p_toplevel_paths) {
	toplevel_paths = p_toplevel_paths.split(",", false);
	emit_changed();
}
