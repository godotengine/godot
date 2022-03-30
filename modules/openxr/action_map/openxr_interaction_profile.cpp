/*************************************************************************/
/*  openxr_interaction_profile.cpp                                       */
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

#include "openxr_interaction_profile.h"

void OpenXRIPBinding::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_action", "action"), &OpenXRIPBinding::set_action);
	ClassDB::bind_method(D_METHOD("get_action"), &OpenXRIPBinding::get_action);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "action", PROPERTY_HINT_RESOURCE_TYPE, "OpenXRAction"), "set_action", "get_action");

	ClassDB::bind_method(D_METHOD("set_paths", "paths"), &OpenXRIPBinding::set_paths);
	ClassDB::bind_method(D_METHOD("get_paths"), &OpenXRIPBinding::get_paths);
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "paths", PROPERTY_HINT_ARRAY_TYPE, "STRING"), "set_paths", "get_paths");
}

Ref<OpenXRIPBinding> OpenXRIPBinding::new_binding(const Ref<OpenXRAction> p_action, const char *p_paths) {
	// This is a helper function to help build our default action sets

	Ref<OpenXRIPBinding> binding;
	binding.instantiate();
	binding->set_action(p_action);
	binding->parse_paths(String(p_paths));

	return binding;
}

void OpenXRIPBinding::set_action(const Ref<OpenXRAction> p_action) {
	action = p_action;
}

Ref<OpenXRAction> OpenXRIPBinding::get_action() const {
	return action;
}

void OpenXRIPBinding::set_paths(const PackedStringArray p_paths) {
	paths = p_paths;
}

PackedStringArray OpenXRIPBinding::get_paths() const {
	return paths;
}

void OpenXRIPBinding::parse_paths(const String p_paths) {
	paths = p_paths.split(",", false);
}

OpenXRIPBinding::~OpenXRIPBinding() {
	action.unref();
}

void OpenXRInteractionProfile::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_interaction_profile_path", "interaction_profile_path"), &OpenXRInteractionProfile::set_interaction_profile_path);
	ClassDB::bind_method(D_METHOD("get_interaction_profile_path"), &OpenXRInteractionProfile::get_interaction_profile_path);
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "interaction_profile_path"), "set_interaction_profile_path", "get_interaction_profile_path");

	ClassDB::bind_method(D_METHOD("set_bindings", "bindings"), &OpenXRInteractionProfile::set_bindings);
	ClassDB::bind_method(D_METHOD("get_bindings"), &OpenXRInteractionProfile::get_bindings);
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "bindings", PROPERTY_HINT_RESOURCE_TYPE, "OpenXRIPBinding", PROPERTY_USAGE_NO_EDITOR), "set_bindings", "get_bindings");
}

Ref<OpenXRInteractionProfile> OpenXRInteractionProfile::new_profile(const char *p_input_profile_path) {
	Ref<OpenXRInteractionProfile> profile;
	profile.instantiate();
	profile->set_interaction_profile_path(String(p_input_profile_path));

	return profile;
}

void OpenXRInteractionProfile::set_interaction_profile_path(const String p_input_profile_path) {
	interaction_profile_path = p_input_profile_path;
}

String OpenXRInteractionProfile::get_interaction_profile_path() const {
	return interaction_profile_path;
}

void OpenXRInteractionProfile::set_bindings(Array p_bindings) {
	bindings = p_bindings;
}

Array OpenXRInteractionProfile::get_bindings() const {
	return bindings;
}

void OpenXRInteractionProfile::add_binding(Ref<OpenXRIPBinding> p_binding) {
	ERR_FAIL_COND(p_binding.is_null());

	if (bindings.find(p_binding) == -1) {
		bindings.push_back(p_binding);
	}
}

void OpenXRInteractionProfile::remove_binding(Ref<OpenXRIPBinding> p_binding) {
	int idx = bindings.find(p_binding);
	if (idx != -1) {
		bindings.remove_at(idx);
	}
}

void OpenXRInteractionProfile::add_new_binding(const Ref<OpenXRAction> p_action, const char *p_paths) {
	// This is a helper function to help build our default action sets

	Ref<OpenXRIPBinding> binding = OpenXRIPBinding::new_binding(p_action, p_paths);
	add_binding(binding);
}

OpenXRInteractionProfile::~OpenXRInteractionProfile() {
	bindings.clear();
}
