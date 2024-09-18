/**************************************************************************/
/*  openxr_dpad_binding_extension.cpp                                     */
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

#include "openxr_dpad_binding_extension.h"
#include "../openxr_api.h"
#include "core/math/math_funcs.h"

// Implementation for:
// https://registry.khronos.org/OpenXR/specs/1.1/html/xrspec.html#XR_EXT_dpad_binding

///////////////////////////////////////////////////////////////////////////////////////////////////
// OpenXRDPadBindingExtension

OpenXRDPadBindingExtension *OpenXRDPadBindingExtension::singleton = nullptr;

OpenXRDPadBindingExtension *OpenXRDPadBindingExtension::get_singleton() {
	return singleton;
}

OpenXRDPadBindingExtension::OpenXRDPadBindingExtension() {
	singleton = this;
}

OpenXRDPadBindingExtension::~OpenXRDPadBindingExtension() {
	singleton = nullptr;
}

HashMap<String, bool *> OpenXRDPadBindingExtension::get_requested_extensions() {
	HashMap<String, bool *> request_extensions;

	// Note, we're dependent on the binding modifier extension, this may be requested by multiple extension wrappers.
	request_extensions[XR_KHR_BINDING_MODIFICATION_EXTENSION_NAME] = &binding_modifier_ext;
	request_extensions[XR_EXT_DPAD_BINDING_EXTENSION_NAME] = &dpad_binding_ext;

	return request_extensions;
}

bool OpenXRDPadBindingExtension::is_available() {
	return binding_modifier_ext && dpad_binding_ext;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// OpenXRDpadBindingModifier

void OpenXRDpadBindingModifier::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_action_set", "action_set"), &OpenXRDpadBindingModifier::set_action_set);
	ClassDB::bind_method(D_METHOD("get_action_set"), &OpenXRDpadBindingModifier::get_action_set);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "action_set", PROPERTY_HINT_RESOURCE_TYPE, "OpenXRActionSet"), "set_action_set", "get_action_set");

	ClassDB::bind_method(D_METHOD("set_threshold", "threshold"), &OpenXRDpadBindingModifier::set_threshold);
	ClassDB::bind_method(D_METHOD("get_threshold"), &OpenXRDpadBindingModifier::get_threshold);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "threshold", PROPERTY_HINT_RANGE, "0.0,1.0,0.01"), "set_threshold", "get_threshold");

	ClassDB::bind_method(D_METHOD("set_threshold_released", "threshold_released"), &OpenXRDpadBindingModifier::set_threshold_released);
	ClassDB::bind_method(D_METHOD("get_threshold_released"), &OpenXRDpadBindingModifier::get_threshold_released);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "threshold_released", PROPERTY_HINT_RANGE, "0.0,1.0,0.01"), "set_threshold_released", "get_threshold_released");

	ClassDB::bind_method(D_METHOD("set_center_region", "center_region"), &OpenXRDpadBindingModifier::set_center_region);
	ClassDB::bind_method(D_METHOD("get_center_region"), &OpenXRDpadBindingModifier::get_center_region);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "center_region", PROPERTY_HINT_RANGE, "0.0,1.0,0.01"), "set_center_region", "get_center_region");

	ClassDB::bind_method(D_METHOD("set_wedge_angle", "wedge_angle"), &OpenXRDpadBindingModifier::set_wedge_angle);
	ClassDB::bind_method(D_METHOD("get_wedge_angle"), &OpenXRDpadBindingModifier::get_wedge_angle);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "wedge_angle"), "set_wedge_angle", "get_wedge_angle");

	ClassDB::bind_method(D_METHOD("set_is_sticky", "is_sticky"), &OpenXRDpadBindingModifier::set_is_sticky);
	ClassDB::bind_method(D_METHOD("get_is_sticky"), &OpenXRDpadBindingModifier::get_is_sticky);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "is_sticky"), "set_is_sticky", "get_is_sticky");
}

OpenXRDpadBindingModifier::OpenXRDpadBindingModifier() {
	dpad_bindings.type = XR_TYPE_INTERACTION_PROFILE_DPAD_BINDING_EXT;
	dpad_bindings.next = nullptr;

	dpad_bindings.forceThreshold = 0.6;
	dpad_bindings.forceThresholdReleased = 0.4;
	dpad_bindings.centerRegion = 0.1;
	dpad_bindings.wedgeAngle = Math::deg_to_rad(90.0);
	dpad_bindings.isSticky = false;
}

void OpenXRDpadBindingModifier::set_action_set(const Ref<OpenXRActionSet> p_action_set) {
	action_set = p_action_set;
}

Ref<OpenXRActionSet> OpenXRDpadBindingModifier::get_action_set() const {
	return action_set;
}

void OpenXRDpadBindingModifier::set_threshold(float p_threshold) {
	ERR_FAIL_COND(p_threshold < 0.0 || p_threshold > 1.0);

	dpad_bindings.forceThreshold = p_threshold;
	emit_changed();
}

float OpenXRDpadBindingModifier::get_threshold() const {
	return dpad_bindings.forceThresholdReleased;
}

void OpenXRDpadBindingModifier::set_threshold_released(float p_threshold) {
	ERR_FAIL_COND(p_threshold < 0.0 || p_threshold > 1.0);

	dpad_bindings.forceThresholdReleased = p_threshold;
	emit_changed();
}

float OpenXRDpadBindingModifier::get_threshold_released() const {
	return dpad_bindings.forceThresholdReleased;
}

void OpenXRDpadBindingModifier::set_center_region(float p_center_region) {
	ERR_FAIL_COND(p_center_region < 0.0 || p_center_region > 1.0);

	dpad_bindings.centerRegion = p_center_region;
	emit_changed();
}

float OpenXRDpadBindingModifier::get_center_region() const {
	return dpad_bindings.centerRegion;
}

void OpenXRDpadBindingModifier::set_wedge_angle(float p_wedge_angle) {
	dpad_bindings.wedgeAngle = p_wedge_angle;
	emit_changed();
}

float OpenXRDpadBindingModifier::get_wedge_angle() const {
	return dpad_bindings.wedgeAngle;
}

void OpenXRDpadBindingModifier::set_wedge_angle_deg(float p_wedge_angle) {
	dpad_bindings.wedgeAngle = Math::deg_to_rad(p_wedge_angle);
	emit_changed();
}

float OpenXRDpadBindingModifier::get_wedge_angle_deg() const {
	return Math::rad_to_deg(dpad_bindings.wedgeAngle);
}

void OpenXRDpadBindingModifier::set_is_sticky(bool p_sticky) {
	dpad_bindings.isSticky = p_sticky;
	emit_changed();
}

bool OpenXRDpadBindingModifier::get_is_sticky() const {
	return dpad_bindings.isSticky;
}

OpenXRBindingModifier::BindingModifierType OpenXRDpadBindingModifier::get_binding_modifier_type() const {
	return BINDING_MODIFIER_IO_PATH;
}

int OpenXRDpadBindingModifier::get_binding_modification_struct_size() const {
	return sizeof(XrInteractionProfileDpadBindingEXT);
}

const XrBindingModificationBaseHeaderKHR *OpenXRDpadBindingModifier::get_binding_modification() {
	OpenXRDPadBindingExtension *dpad_binding_ext = OpenXRDPadBindingExtension::get_singleton();
	if (!dpad_binding_ext || !dpad_binding_ext->is_available()) {
		// Extension not enabled!
		WARN_PRINT("DPad binding extension is not enabled or available.");
		return nullptr;
	}

	ERR_FAIL_COND_V(!action.is_valid(), nullptr);

	OpenXRAPI *openxr_api = OpenXRAPI::get_singleton();
	ERR_FAIL_NULL_V(openxr_api, nullptr);

	dpad_bindings.binding = openxr_api->get_xr_path(input_path);
	ERR_FAIL_COND_V(dpad_bindings.binding == XR_NULL_PATH, nullptr);

	// Get our action set
	ERR_FAIL_COND_V(!action_set.is_valid(), nullptr);
	RID action_set_rid = openxr_api->find_action_set(action_set->get_name());
	ERR_FAIL_COND_V(!action_set_rid.is_valid(), nullptr);
	dpad_bindings.actionSet = openxr_api->action_set_get_handle(action_set_rid);

	// These are set already:
	// - forceThreshold
	// - forceThresholdReleased
	// - centerRegion
	// - wedgeAngle
	// - isSticky

	// Not yet supported
	dpad_bindings.onHaptic = nullptr;
	dpad_bindings.offHaptic = nullptr;

	return (XrBindingModificationBaseHeaderKHR *)&dpad_bindings;
}
