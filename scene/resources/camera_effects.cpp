/*************************************************************************/
/*  camera_effects.cpp                                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "camera_effects.h"

#include "servers/rendering_server.h"

RID CameraEffects::get_rid() const {
	return camera_effects;
}

// DOF blur

void CameraEffects::set_dof_blur_far_enabled(bool p_enabled) {
	dof_blur_far_enabled = p_enabled;
	_update_dof_blur();
	notify_property_list_changed();
}

bool CameraEffects::is_dof_blur_far_enabled() const {
	return dof_blur_far_enabled;
}

void CameraEffects::set_dof_blur_far_distance(float p_distance) {
	dof_blur_far_distance = p_distance;
	_update_dof_blur();
}

float CameraEffects::get_dof_blur_far_distance() const {
	return dof_blur_far_distance;
}

void CameraEffects::set_dof_blur_far_transition(float p_distance) {
	dof_blur_far_transition = p_distance;
	_update_dof_blur();
}

float CameraEffects::get_dof_blur_far_transition() const {
	return dof_blur_far_transition;
}

void CameraEffects::set_dof_blur_near_enabled(bool p_enabled) {
	dof_blur_near_enabled = p_enabled;
	_update_dof_blur();
	notify_property_list_changed();
}

bool CameraEffects::is_dof_blur_near_enabled() const {
	return dof_blur_near_enabled;
}

void CameraEffects::set_dof_blur_near_distance(float p_distance) {
	dof_blur_near_distance = p_distance;
	_update_dof_blur();
}

float CameraEffects::get_dof_blur_near_distance() const {
	return dof_blur_near_distance;
}

void CameraEffects::set_dof_blur_near_transition(float p_distance) {
	dof_blur_near_transition = p_distance;
	_update_dof_blur();
}

float CameraEffects::get_dof_blur_near_transition() const {
	return dof_blur_near_transition;
}

void CameraEffects::set_dof_blur_amount(float p_amount) {
	dof_blur_amount = p_amount;
	_update_dof_blur();
}

float CameraEffects::get_dof_blur_amount() const {
	return dof_blur_amount;
}

void CameraEffects::_update_dof_blur() {
	RS::get_singleton()->camera_effects_set_dof_blur(
			camera_effects,
			dof_blur_far_enabled,
			dof_blur_far_distance,
			dof_blur_far_transition,
			dof_blur_near_enabled,
			dof_blur_near_distance,
			dof_blur_near_transition,
			dof_blur_amount);
}

// Custom exposure

void CameraEffects::set_override_exposure_enabled(bool p_enabled) {
	override_exposure_enabled = p_enabled;
	_update_override_exposure();
	notify_property_list_changed();
}

bool CameraEffects::is_override_exposure_enabled() const {
	return override_exposure_enabled;
}

void CameraEffects::set_override_exposure(float p_exposure) {
	override_exposure = p_exposure;
	_update_override_exposure();
}

float CameraEffects::get_override_exposure() const {
	return override_exposure;
}

void CameraEffects::_update_override_exposure() {
	RS::get_singleton()->camera_effects_set_custom_exposure(
			camera_effects,
			override_exposure_enabled,
			override_exposure);
}

// Private methods, constructor and destructor

void CameraEffects::_validate_property(PropertyInfo &property) const {
	if ((!dof_blur_far_enabled && (property.name == "dof_blur_far_distance" || property.name == "dof_blur_far_transition")) ||
			(!dof_blur_near_enabled && (property.name == "dof_blur_near_distance" || property.name == "dof_blur_near_transition")) ||
			(!override_exposure_enabled && property.name == "override_exposure")) {
		property.usage = PROPERTY_USAGE_NO_EDITOR;
	}
}

void CameraEffects::_bind_methods() {
	// DOF blur

	ClassDB::bind_method(D_METHOD("set_dof_blur_far_enabled", "enabled"), &CameraEffects::set_dof_blur_far_enabled);
	ClassDB::bind_method(D_METHOD("is_dof_blur_far_enabled"), &CameraEffects::is_dof_blur_far_enabled);
	ClassDB::bind_method(D_METHOD("set_dof_blur_far_distance", "distance"), &CameraEffects::set_dof_blur_far_distance);
	ClassDB::bind_method(D_METHOD("get_dof_blur_far_distance"), &CameraEffects::get_dof_blur_far_distance);
	ClassDB::bind_method(D_METHOD("set_dof_blur_far_transition", "distance"), &CameraEffects::set_dof_blur_far_transition);
	ClassDB::bind_method(D_METHOD("get_dof_blur_far_transition"), &CameraEffects::get_dof_blur_far_transition);

	ClassDB::bind_method(D_METHOD("set_dof_blur_near_enabled", "enabled"), &CameraEffects::set_dof_blur_near_enabled);
	ClassDB::bind_method(D_METHOD("is_dof_blur_near_enabled"), &CameraEffects::is_dof_blur_near_enabled);
	ClassDB::bind_method(D_METHOD("set_dof_blur_near_distance", "distance"), &CameraEffects::set_dof_blur_near_distance);
	ClassDB::bind_method(D_METHOD("get_dof_blur_near_distance"), &CameraEffects::get_dof_blur_near_distance);
	ClassDB::bind_method(D_METHOD("set_dof_blur_near_transition", "distance"), &CameraEffects::set_dof_blur_near_transition);
	ClassDB::bind_method(D_METHOD("get_dof_blur_near_transition"), &CameraEffects::get_dof_blur_near_transition);

	ClassDB::bind_method(D_METHOD("set_dof_blur_amount", "amount"), &CameraEffects::set_dof_blur_amount);
	ClassDB::bind_method(D_METHOD("get_dof_blur_amount"), &CameraEffects::get_dof_blur_amount);

	ADD_GROUP("DOF Blur", "dof_blur_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "dof_blur_far_enabled"), "set_dof_blur_far_enabled", "is_dof_blur_far_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "dof_blur_far_distance", PROPERTY_HINT_RANGE, "0.01,8192,0.01,exp"), "set_dof_blur_far_distance", "get_dof_blur_far_distance");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "dof_blur_far_transition", PROPERTY_HINT_RANGE, "0.01,8192,0.01,exp"), "set_dof_blur_far_transition", "get_dof_blur_far_transition");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "dof_blur_near_enabled"), "set_dof_blur_near_enabled", "is_dof_blur_near_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "dof_blur_near_distance", PROPERTY_HINT_RANGE, "0.01,8192,0.01,exp"), "set_dof_blur_near_distance", "get_dof_blur_near_distance");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "dof_blur_near_transition", PROPERTY_HINT_RANGE, "0.01,8192,0.01,exp"), "set_dof_blur_near_transition", "get_dof_blur_near_transition");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "dof_blur_amount", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_dof_blur_amount", "get_dof_blur_amount");

	// Override exposure

	ClassDB::bind_method(D_METHOD("set_override_exposure_enabled", "enabled"), &CameraEffects::set_override_exposure_enabled);
	ClassDB::bind_method(D_METHOD("is_override_exposure_enabled"), &CameraEffects::is_override_exposure_enabled);
	ClassDB::bind_method(D_METHOD("set_override_exposure", "exposure"), &CameraEffects::set_override_exposure);
	ClassDB::bind_method(D_METHOD("get_override_exposure"), &CameraEffects::get_override_exposure);

	ADD_GROUP("Override Exposure", "override_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "override_exposure_enabled"), "set_override_exposure_enabled", "is_override_exposure_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "override_exposure", PROPERTY_HINT_RANGE, "0,16,0.01"), "set_override_exposure", "get_override_exposure");
}

CameraEffects::CameraEffects() {
	camera_effects = RS::get_singleton()->camera_effects_create();

	_update_dof_blur();
	_update_override_exposure();
}

CameraEffects::~CameraEffects() {
	RS::get_singleton()->free(camera_effects);
}
