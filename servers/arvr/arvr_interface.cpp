/*************************************************************************/
/*  arvr_interface.cpp                                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
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

	ClassDB::bind_method(D_METHOD("is_primary"), &ARVRInterface::is_primary);
	ClassDB::bind_method(D_METHOD("set_is_primary", "enable"), &ARVRInterface::set_is_primary);

	ClassDB::bind_method(D_METHOD("is_installed"), &ARVRInterface::is_installed);
	ClassDB::bind_method(D_METHOD("hmd_is_present"), &ARVRInterface::hmd_is_present);
	ClassDB::bind_method(D_METHOD("supports_hmd"), &ARVRInterface::supports_hmd);
	ClassDB::bind_method(D_METHOD("is_initialized"), &ARVRInterface::is_initialized);
	ClassDB::bind_method(D_METHOD("initialize"), &ARVRInterface::initialize);
	ClassDB::bind_method(D_METHOD("uninitialize"), &ARVRInterface::uninitialize);

	ClassDB::bind_method(D_METHOD("get_recommended_render_targetsize"), &ARVRInterface::get_recommended_render_targetsize);

	//	These are now purely used internally, we may expose them again if we expose CameraMatrix through Variant but reduz is not a fan for good reasons :)
	//	ClassDB::bind_method(D_METHOD("get_transform_for_eye", "eye", "cam_transform"), &ARVRInterface::get_transform_for_eye);
	//	ClassDB::bind_method(D_METHOD("get_projection_for_eye", "eye"), &ARVRInterface::get_projection_for_eye);
	//	ClassDB::bind_method(D_METHOD("commit_for_eye", "node:viewport"), &ARVRInterface::commit_for_eye);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "primary"), "set_is_primary", "is_primary");

	BIND_CONSTANT(EYE_MONO);
	BIND_CONSTANT(EYE_LEFT);
	BIND_CONSTANT(EYE_RIGHT);
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
		ERR_FAIL_COND(!supports_hmd());

		arvr_server->set_primary_interface(this);
	} else {
		arvr_server->clear_primary_interface_if(this);
	};
};
