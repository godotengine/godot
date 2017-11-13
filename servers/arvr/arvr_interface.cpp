/*************************************************************************/
/*  arvr_interface.cpp                                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "arvr_interface.h"

void ARVRInterface::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_name"), &ARVRInterface::get_name);
	ClassDB::bind_method(D_METHOD("get_capabilities"), &ARVRInterface::get_capabilities);

	ClassDB::bind_method(D_METHOD("is_primary"), &ARVRInterface::is_primary);
	ClassDB::bind_method(D_METHOD("set_is_primary", "enable"), &ARVRInterface::set_is_primary);

	ClassDB::bind_method(D_METHOD("is_initialized"), &ARVRInterface::is_initialized);
	ClassDB::bind_method(D_METHOD("set_is_initialized", "initialized"), &ARVRInterface::set_is_initialized);
	ClassDB::bind_method(D_METHOD("initialize"), &ARVRInterface::initialize);
	ClassDB::bind_method(D_METHOD("uninitialize"), &ARVRInterface::uninitialize);

	ClassDB::bind_method(D_METHOD("get_tracking_status"), &ARVRInterface::get_tracking_status);

	ClassDB::bind_method(D_METHOD("get_render_targetsize"), &ARVRInterface::get_render_targetsize);
	ClassDB::bind_method(D_METHOD("is_stereo"), &ARVRInterface::is_stereo);

	ADD_GROUP("Interface", "interface_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "interface_is_primary"), "set_is_primary", "is_primary");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "interface_is_initialized"), "set_is_initialized", "is_initialized");

	// we don't have any properties specific to VR yet....

	// but we do have properties specific to AR....
	ClassDB::bind_method(D_METHOD("get_anchor_detection_is_enabled"), &ARVRInterface::get_anchor_detection_is_enabled);
	ClassDB::bind_method(D_METHOD("set_anchor_detection_is_enabled", "enable"), &ARVRInterface::set_anchor_detection_is_enabled);

	ADD_GROUP("AR", "ar_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "ar_is_anchor_detection_enabled"), "set_anchor_detection_is_enabled", "get_anchor_detection_is_enabled");

	BIND_ENUM_CONSTANT(ARVR_NONE);
	BIND_ENUM_CONSTANT(ARVR_MONO);
	BIND_ENUM_CONSTANT(ARVR_STEREO);
	BIND_ENUM_CONSTANT(ARVR_AR);
	BIND_ENUM_CONSTANT(ARVR_EXTERNAL);

	BIND_ENUM_CONSTANT(EYE_MONO);
	BIND_ENUM_CONSTANT(EYE_LEFT);
	BIND_ENUM_CONSTANT(EYE_RIGHT);

	BIND_ENUM_CONSTANT(ARVR_NORMAL_TRACKING);
	BIND_ENUM_CONSTANT(ARVR_EXCESSIVE_MOTION);
	BIND_ENUM_CONSTANT(ARVR_INSUFFICIENT_FEATURES);
	BIND_ENUM_CONSTANT(ARVR_UNKNOWN_TRACKING);
	BIND_ENUM_CONSTANT(ARVR_NOT_TRACKING);
};

StringName ARVRInterface::get_name() const {
	return "Unknown";
};

bool ARVRInterface::is_primary() {
	ARVRServer *arvr_server = ARVRServer::get_singleton();
	ERR_FAIL_NULL_V(arvr_server, false);

	return arvr_server->get_primary_interface() == this;
};

void ARVRInterface::set_is_primary(bool p_is_primary) {
	ARVRServer *arvr_server = ARVRServer::get_singleton();
	ERR_FAIL_NULL(arvr_server);

	if (p_is_primary) {
		ERR_FAIL_COND(!is_initialized());

		arvr_server->set_primary_interface(this);
	} else {
		arvr_server->clear_primary_interface_if(this);
	};
};

void ARVRInterface::set_is_initialized(bool p_initialized) {
	if (p_initialized) {
		if (!is_initialized()) {
			initialize();
		};
	} else {
		if (is_initialized()) {
			uninitialize();
		};
	};
};

ARVRInterface::Tracking_status ARVRInterface::get_tracking_status() const {
	return tracking_state;
};

ARVRInterface::ARVRInterface() {
	tracking_state = ARVR_UNKNOWN_TRACKING;
};

ARVRInterface::~ARVRInterface(){};

/** these will only be implemented on AR interfaces, so we want dummies for VR **/
bool ARVRInterface::get_anchor_detection_is_enabled() const {
	return false;
};

void ARVRInterface::set_anchor_detection_is_enabled(bool p_enable){
	// don't do anything here, this needs to be implemented on AR interface to enable/disable things like plane detection etc.
};
