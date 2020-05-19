/*************************************************************************/
/*  xr_interface.cpp                                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "xr_interface.h"

void XRInterface::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_name"), &XRInterface::get_name);
	ClassDB::bind_method(D_METHOD("get_capabilities"), &XRInterface::get_capabilities);

	ClassDB::bind_method(D_METHOD("is_primary"), &XRInterface::is_primary);
	ClassDB::bind_method(D_METHOD("set_is_primary", "enable"), &XRInterface::set_is_primary);

	ClassDB::bind_method(D_METHOD("is_initialized"), &XRInterface::is_initialized);
	ClassDB::bind_method(D_METHOD("set_is_initialized", "initialized"), &XRInterface::set_is_initialized);
	ClassDB::bind_method(D_METHOD("initialize"), &XRInterface::initialize);
	ClassDB::bind_method(D_METHOD("uninitialize"), &XRInterface::uninitialize);

	ClassDB::bind_method(D_METHOD("get_tracking_status"), &XRInterface::get_tracking_status);

	ClassDB::bind_method(D_METHOD("get_render_targetsize"), &XRInterface::get_render_targetsize);
	ClassDB::bind_method(D_METHOD("is_stereo"), &XRInterface::is_stereo);

	ADD_GROUP("Interface", "interface_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "interface_is_primary"), "set_is_primary", "is_primary");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "interface_is_initialized"), "set_is_initialized", "is_initialized");

	// we don't have any properties specific to VR yet....

	// but we do have properties specific to AR....
	ClassDB::bind_method(D_METHOD("get_anchor_detection_is_enabled"), &XRInterface::get_anchor_detection_is_enabled);
	ClassDB::bind_method(D_METHOD("set_anchor_detection_is_enabled", "enable"), &XRInterface::set_anchor_detection_is_enabled);
	ClassDB::bind_method(D_METHOD("get_camera_feed_id"), &XRInterface::get_camera_feed_id);

	ADD_GROUP("AR", "ar_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "ar_is_anchor_detection_enabled"), "set_anchor_detection_is_enabled", "get_anchor_detection_is_enabled");

	BIND_ENUM_CONSTANT(XR_NONE);
	BIND_ENUM_CONSTANT(XR_MONO);
	BIND_ENUM_CONSTANT(XR_STEREO);
	BIND_ENUM_CONSTANT(XR_AR);
	BIND_ENUM_CONSTANT(XR_EXTERNAL);

	BIND_ENUM_CONSTANT(EYE_MONO);
	BIND_ENUM_CONSTANT(EYE_LEFT);
	BIND_ENUM_CONSTANT(EYE_RIGHT);

	BIND_ENUM_CONSTANT(XR_NORMAL_TRACKING);
	BIND_ENUM_CONSTANT(XR_EXCESSIVE_MOTION);
	BIND_ENUM_CONSTANT(XR_INSUFFICIENT_FEATURES);
	BIND_ENUM_CONSTANT(XR_UNKNOWN_TRACKING);
	BIND_ENUM_CONSTANT(XR_NOT_TRACKING);
};

StringName XRInterface::get_name() const {
	return "Unknown";
};

bool XRInterface::is_primary() {
	XRServer *xr_server = XRServer::get_singleton();
	ERR_FAIL_NULL_V(xr_server, false);

	return xr_server->get_primary_interface() == this;
};

void XRInterface::set_is_primary(bool p_is_primary) {
	XRServer *xr_server = XRServer::get_singleton();
	ERR_FAIL_NULL(xr_server);

	if (p_is_primary) {
		ERR_FAIL_COND(!is_initialized());

		xr_server->set_primary_interface(this);
	} else {
		xr_server->clear_primary_interface_if(this);
	};
};

void XRInterface::set_is_initialized(bool p_initialized) {
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

XRInterface::Tracking_status XRInterface::get_tracking_status() const {
	return tracking_state;
};

XRInterface::XRInterface() {
	tracking_state = XR_UNKNOWN_TRACKING;
};

XRInterface::~XRInterface() {}

// optional render to external texture which enhances performance on those platforms that require us to submit our end result into special textures.
unsigned int XRInterface::get_external_texture_for_eye(XRInterface::Eyes p_eye) {
	return 0;
};

/** these will only be implemented on AR interfaces, so we want dummies for VR **/
bool XRInterface::get_anchor_detection_is_enabled() const {
	return false;
};

void XRInterface::set_anchor_detection_is_enabled(bool p_enable) {
	// don't do anything here, this needs to be implemented on AR interface to enable/disable things like plane detection etc.
}

int XRInterface::get_camera_feed_id() {
	// don't do anything here, this needs to be implemented on AR interface to enable/disable things like plane detection etc.

	return 0;
};
