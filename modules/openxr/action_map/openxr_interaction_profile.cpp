/**************************************************************************/
/*  openxr_interaction_profile.cpp                                        */
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

#include "openxr_interaction_profile.h"

void OpenXRIPBinding::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_action", "action"), &OpenXRIPBinding::set_action);
	ClassDB::bind_method(D_METHOD("get_action"), &OpenXRIPBinding::get_action);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "action", PROPERTY_HINT_RESOURCE_TYPE, "OpenXRAction"), "set_action", "get_action");

	ClassDB::bind_method(D_METHOD("get_path_count"), &OpenXRIPBinding::get_path_count);
	ClassDB::bind_method(D_METHOD("set_paths", "paths"), &OpenXRIPBinding::set_paths);
	ClassDB::bind_method(D_METHOD("get_paths"), &OpenXRIPBinding::get_paths);
	ADD_PROPERTY(PropertyInfo(Variant::PACKED_STRING_ARRAY, "paths"), "set_paths", "get_paths");

	ClassDB::bind_method(D_METHOD("has_path", "path"), &OpenXRIPBinding::has_path);
	ClassDB::bind_method(D_METHOD("add_path", "path"), &OpenXRIPBinding::add_path);
	ClassDB::bind_method(D_METHOD("remove_path", "path"), &OpenXRIPBinding::remove_path);
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
	emit_changed();
}

Ref<OpenXRAction> OpenXRIPBinding::get_action() const {
	return action;
}

int OpenXRIPBinding::get_path_count() const {
	return paths.size();
}

void OpenXRIPBinding::set_paths(const PackedStringArray p_paths) {
	paths = p_paths;
	emit_changed();
}

PackedStringArray OpenXRIPBinding::get_paths() const {
	return paths;
}

void OpenXRIPBinding::parse_paths(const String p_paths) {
	paths = p_paths.split(",", false);
	emit_changed();
}

bool OpenXRIPBinding::has_path(const String p_path) const {
	return paths.has(p_path);
}

void OpenXRIPBinding::add_path(const String p_path) {
	if (!paths.has(p_path)) {
		paths.push_back(p_path);
		emit_changed();
	}
}

void OpenXRIPBinding::remove_path(const String p_path) {
	if (paths.has(p_path)) {
		paths.erase(p_path);
		emit_changed();
	}
}

OpenXRIPBinding::~OpenXRIPBinding() {
	action.unref();
}

void OpenXRInteractionProfile::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_interaction_profile_path", "interaction_profile_path"), &OpenXRInteractionProfile::set_interaction_profile_path);
	ClassDB::bind_method(D_METHOD("get_interaction_profile_path"), &OpenXRInteractionProfile::get_interaction_profile_path);
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "interaction_profile_path"), "set_interaction_profile_path", "get_interaction_profile_path");

	ClassDB::bind_method(D_METHOD("get_binding_count"), &OpenXRInteractionProfile::get_binding_count);
	ClassDB::bind_method(D_METHOD("get_binding", "index"), &OpenXRInteractionProfile::get_binding);
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
	OpenXRInteractionProfileMetadata *pmd = OpenXRInteractionProfileMetadata::get_singleton();
	ERR_FAIL_NULL(pmd);

	interaction_profile_path = pmd->check_profile_name(p_input_profile_path);
	emit_changed();
}

String OpenXRInteractionProfile::get_interaction_profile_path() const {
	return interaction_profile_path;
}

int OpenXRInteractionProfile::get_binding_count() const {
	return bindings.size();
}

Ref<OpenXRIPBinding> OpenXRInteractionProfile::get_binding(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, bindings.size(), Ref<OpenXRIPBinding>());

	return bindings[p_index];
}

void OpenXRInteractionProfile::set_bindings(Array p_bindings) {
	// TODO add check here that our bindings don't contain duplicate actions

	bindings = p_bindings;
	emit_changed();
}

Array OpenXRInteractionProfile::get_bindings() const {
	return bindings;
}

Ref<OpenXRIPBinding> OpenXRInteractionProfile::get_binding_for_action(const Ref<OpenXRAction> p_action) const {
	for (int i = 0; i < bindings.size(); i++) {
		Ref<OpenXRIPBinding> binding = bindings[i];
		if (binding->get_action() == p_action) {
			return binding;
		}
	}

	return Ref<OpenXRIPBinding>();
}

void OpenXRInteractionProfile::add_binding(Ref<OpenXRIPBinding> p_binding) {
	ERR_FAIL_COND(p_binding.is_null());

	if (!bindings.has(p_binding)) {
		ERR_FAIL_COND_MSG(get_binding_for_action(p_binding->get_action()).is_valid(), "There is already a binding for this action in this interaction profile");

		bindings.push_back(p_binding);
		emit_changed();
	}
}

void OpenXRInteractionProfile::remove_binding(Ref<OpenXRIPBinding> p_binding) {
	int idx = bindings.find(p_binding);
	if (idx != -1) {
		bindings.remove_at(idx);
		emit_changed();
	}
}

void OpenXRInteractionProfile::add_new_binding(const Ref<OpenXRAction> p_action, const char *p_paths) {
	// This is a helper function to help build our default action sets

	Ref<OpenXRIPBinding> binding = OpenXRIPBinding::new_binding(p_action, p_paths);
	add_binding(binding);
}

void OpenXRInteractionProfile::remove_binding_for_action(const Ref<OpenXRAction> p_action) {
	for (int i = bindings.size() - 1; i >= 0; i--) {
		Ref<OpenXRIPBinding> binding = bindings[i];
		if (binding->get_action() == p_action) {
			remove_binding(binding);
		}
	}
}

bool OpenXRInteractionProfile::has_binding_for_action(const Ref<OpenXRAction> p_action) {
	for (int i = bindings.size() - 1; i >= 0; i--) {
		Ref<OpenXRIPBinding> binding = bindings[i];
		if (binding->get_action() == p_action) {
			return true;
		}
	}

	return false;
}

OpenXRInteractionProfile::~OpenXRInteractionProfile() {
	bindings.clear();
}
