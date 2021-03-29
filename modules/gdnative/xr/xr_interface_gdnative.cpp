/*************************************************************************/
/*  xr_interface_gdnative.cpp                                            */
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

#include "xr_interface_gdnative.h"
#include "core/input/input.h"
#include "servers/rendering/rendering_server_globals.h"
#include "servers/xr/xr_positional_tracker.h"

void XRInterfaceGDNative::_bind_methods() {
	ADD_PROPERTY_DEFAULT("interface_is_initialized", false);
	ADD_PROPERTY_DEFAULT("ar_is_anchor_detection_enabled", false);
}

XRInterfaceGDNative::XRInterfaceGDNative() {
	print_verbose("Construct gdnative interface\n");

	// we won't have our data pointer until our library gets set
	data = nullptr;

	interface = nullptr;
}

XRInterfaceGDNative::~XRInterfaceGDNative() {
	print_verbose("Destruct gdnative interface\n");

	if (interface != nullptr && is_initialized()) {
		uninitialize();
	};

	// cleanup after ourselves
	cleanup();
}

void XRInterfaceGDNative::cleanup() {
	if (interface != nullptr) {
		interface->destructor(data);
		data = nullptr;
		interface = nullptr;
	}
}

void XRInterfaceGDNative::set_interface(const godot_xr_interface_gdnative *p_interface) {
	// this should only be called once, just being paranoid..
	if (interface) {
		cleanup();
	}

	// bind to our interface
	interface = p_interface;

	// Now we do our constructing...
	data = interface->constructor((godot_object *)this);
}

StringName XRInterfaceGDNative::get_name() const {
	ERR_FAIL_COND_V(interface == nullptr, StringName());

	godot_string result = interface->get_name(data);

	StringName name = *(String *)&result;

	godot_string_destroy(&result);

	return name;
}

int XRInterfaceGDNative::get_capabilities() const {
	int capabilities;

	ERR_FAIL_COND_V(interface == nullptr, 0); // 0 = None

	capabilities = interface->get_capabilities(data);

	return capabilities;
}

bool XRInterfaceGDNative::get_anchor_detection_is_enabled() const {
	ERR_FAIL_COND_V(interface == nullptr, false);

	return interface->get_anchor_detection_is_enabled(data);
}

void XRInterfaceGDNative::set_anchor_detection_is_enabled(bool p_enable) {
	ERR_FAIL_COND(interface == nullptr);

	interface->set_anchor_detection_is_enabled(data, p_enable);
}

int XRInterfaceGDNative::get_camera_feed_id() {
	ERR_FAIL_COND_V(interface == nullptr, 0);

	return (unsigned int)interface->get_camera_feed_id(data);
}

bool XRInterfaceGDNative::is_stereo() {
	bool stereo;

	ERR_FAIL_COND_V(interface == nullptr, false);

	stereo = interface->is_stereo(data);

	return stereo;
}

bool XRInterfaceGDNative::is_initialized() const {
	ERR_FAIL_COND_V(interface == nullptr, false);

	return interface->is_initialized(data);
}

bool XRInterfaceGDNative::initialize() {
	ERR_FAIL_COND_V(interface == nullptr, false);

	bool initialized = interface->initialize(data);

	if (initialized) {
		// if we successfully initialize our interface and we don't have a primary interface yet, this becomes our primary interface

		XRServer *xr_server = XRServer::get_singleton();
		if ((xr_server != nullptr) && (xr_server->get_primary_interface() == nullptr)) {
			xr_server->set_primary_interface(this);
		};
	};

	return initialized;
}

void XRInterfaceGDNative::uninitialize() {
	ERR_FAIL_COND(interface == nullptr);

	XRServer *xr_server = XRServer::get_singleton();
	if (xr_server != nullptr) {
		// Whatever happens, make sure this is no longer our primary interface
		xr_server->clear_primary_interface_if(this);
	}

	interface->uninitialize(data);
}

Size2 XRInterfaceGDNative::get_render_targetsize() {
	ERR_FAIL_COND_V(interface == nullptr, Size2());

	godot_vector2 result = interface->get_render_targetsize(data);
	Vector2 *vec = (Vector2 *)&result;

	return *vec;
}

Transform XRInterfaceGDNative::get_transform_for_eye(XRInterface::Eyes p_eye, const Transform &p_cam_transform) {
	Transform *ret;

	ERR_FAIL_COND_V(interface == nullptr, Transform());

	godot_transform t = interface->get_transform_for_eye(data, (int)p_eye, (godot_transform *)&p_cam_transform);

	ret = (Transform *)&t;

	return *ret;
}

CameraMatrix XRInterfaceGDNative::get_projection_for_eye(XRInterface::Eyes p_eye, real_t p_aspect, real_t p_z_near, real_t p_z_far) {
	CameraMatrix cm;

	ERR_FAIL_COND_V(interface == nullptr, CameraMatrix());

	interface->fill_projection_for_eye(data, (godot_float *)cm.matrix, (godot_int)p_eye, p_aspect, p_z_near, p_z_far);

	return cm;
}

unsigned int XRInterfaceGDNative::get_external_texture_for_eye(XRInterface::Eyes p_eye) {
	ERR_FAIL_COND_V(interface == nullptr, 0);

	return (unsigned int)interface->get_external_texture_for_eye(data, (godot_int)p_eye);
}

void XRInterfaceGDNative::commit_for_eye(XRInterface::Eyes p_eye, RID p_render_target, const Rect2 &p_screen_rect) {
	ERR_FAIL_COND(interface == nullptr);

	interface->commit_for_eye(data, (godot_int)p_eye, (godot_rid *)&p_render_target, (godot_rect2 *)&p_screen_rect);
}

void XRInterfaceGDNative::process() {
	ERR_FAIL_COND(interface == nullptr);

	interface->process(data);
}

void XRInterfaceGDNative::notification(int p_what) {
	ERR_FAIL_COND(interface == nullptr);

	interface->notification(data, p_what);
}

/////////////////////////////////////////////////////////////////////////////////////
// some helper callbacks

extern "C" {

void GDAPI godot_xr_register_interface(const godot_xr_interface_gdnative *p_interface) {
	// Must be on a version 4 plugin
	ERR_FAIL_COND_MSG(p_interface->version.major < 4, "GDNative XR interfaces build for Godot 3.x are not supported.");

	Ref<XRInterfaceGDNative> new_interface;
	new_interface.instance();
	new_interface->set_interface((const godot_xr_interface_gdnative *)p_interface);
	XRServer::get_singleton()->add_interface(new_interface);
}

godot_float GDAPI godot_xr_get_worldscale() {
	XRServer *xr_server = XRServer::get_singleton();
	ERR_FAIL_NULL_V(xr_server, 1.0);

	return xr_server->get_world_scale();
}

godot_transform GDAPI godot_xr_get_reference_frame() {
	godot_transform reference_frame;
	Transform *reference_frame_ptr = (Transform *)&reference_frame;

	XRServer *xr_server = XRServer::get_singleton();
	if (xr_server != nullptr) {
		*reference_frame_ptr = xr_server->get_reference_frame();
	} else {
		memnew_placement(&reference_frame, Transform);
	}

	return reference_frame;
}

void GDAPI godot_xr_blit(godot_int p_eye, godot_rid *p_render_target, godot_rect2 *p_rect) {
	// blits out our texture as is, handy for preview display of one of the eyes that is already rendered with lens distortion on an external HMD
	XRInterface::Eyes eye = (XRInterface::Eyes)p_eye;
#if 0
	RID *render_target = (RID *)p_render_target;
#endif
	Rect2 screen_rect = *(Rect2 *)p_rect;

	if (eye == XRInterface::EYE_LEFT) {
		screen_rect.size.x /= 2.0;
	} else if (p_eye == XRInterface::EYE_RIGHT) {
		screen_rect.size.x /= 2.0;
		screen_rect.position.x += screen_rect.size.x;
	}
#ifndef _MSC_VER
#warning this needs to be redone
#endif
#if 0
	RSG::rasterizer->blit_render_target_to_screen(*render_target, screen_rect, 0);
#endif
}

godot_int GDAPI godot_xr_get_texid(godot_rid *p_render_target) {
	// In order to send off our textures to display on our hardware we need the opengl texture ID instead of the render target RID
	// This is a handy function to expose that.
#if 0
	RID *render_target = (RID *)p_render_target;

	RID eye_texture = RSG::storage->render_target_get_texture(*render_target);
#endif

#ifndef _MSC_VER
#warning need to obtain this ID again
#endif
	uint32_t texid = 0; //RS::get_singleton()->texture_get_texid(eye_texture);

	return texid;
}

godot_int GDAPI godot_xr_add_controller(char *p_device_name, godot_int p_hand, godot_bool p_tracks_orientation, godot_bool p_tracks_position) {
	XRServer *xr_server = XRServer::get_singleton();
	ERR_FAIL_NULL_V(xr_server, 0);

	Input *input = Input::get_singleton();
	ERR_FAIL_NULL_V(input, 0);

	Ref<XRPositionalTracker> new_tracker;
	new_tracker.instance();
	new_tracker->set_tracker_name(p_device_name);
	new_tracker->set_tracker_type(XRServer::TRACKER_CONTROLLER);
	if (p_hand == 1) {
		new_tracker->set_tracker_hand(XRPositionalTracker::TRACKER_HAND_LEFT);
	} else if (p_hand == 2) {
		new_tracker->set_tracker_hand(XRPositionalTracker::TRACKER_HAND_RIGHT);
	}

	// also register as joystick...
	int joyid = input->get_unused_joy_id();
	if (joyid != -1) {
		new_tracker->set_joy_id(joyid);
		input->joy_connection_changed(joyid, true, p_device_name, "");
	}

	if (p_tracks_orientation) {
		Basis orientation;
		new_tracker->set_orientation(orientation);
	}
	if (p_tracks_position) {
		Vector3 position;
		new_tracker->set_position(position);
	}

	// add our tracker to our server and remember its pointer
	xr_server->add_tracker(new_tracker);

	// note, this ID is only unique within controllers!
	return new_tracker->get_tracker_id();
}

void GDAPI godot_xr_remove_controller(godot_int p_controller_id) {
	XRServer *xr_server = XRServer::get_singleton();
	ERR_FAIL_NULL(xr_server);

	Input *input = Input::get_singleton();
	ERR_FAIL_NULL(input);

	Ref<XRPositionalTracker> remove_tracker = xr_server->find_by_type_and_id(XRServer::TRACKER_CONTROLLER, p_controller_id);
	if (remove_tracker.is_valid()) {
		// unset our joystick if applicable
		int joyid = remove_tracker->get_joy_id();
		if (joyid != -1) {
			input->joy_connection_changed(joyid, false, "", "");
			remove_tracker->set_joy_id(-1);
		}

		// remove our tracker from our server
		xr_server->remove_tracker(remove_tracker);
		remove_tracker.unref();
	}
}

void GDAPI godot_xr_set_controller_transform(godot_int p_controller_id, godot_transform *p_transform, godot_bool p_tracks_orientation, godot_bool p_tracks_position) {
	XRServer *xr_server = XRServer::get_singleton();
	ERR_FAIL_NULL(xr_server);

	Ref<XRPositionalTracker> tracker = xr_server->find_by_type_and_id(XRServer::TRACKER_CONTROLLER, p_controller_id);
	if (tracker.is_valid()) {
		Transform *transform = (Transform *)p_transform;
		if (p_tracks_orientation) {
			tracker->set_orientation(transform->basis);
		}
		if (p_tracks_position) {
			tracker->set_rw_position(transform->origin);
		}
	}
}

void GDAPI godot_xr_set_controller_button(godot_int p_controller_id, godot_int p_button, godot_bool p_is_pressed) {
	XRServer *xr_server = XRServer::get_singleton();
	ERR_FAIL_NULL(xr_server);

	Input *input = Input::get_singleton();
	ERR_FAIL_NULL(input);

	Ref<XRPositionalTracker> tracker = xr_server->find_by_type_and_id(XRServer::TRACKER_CONTROLLER, p_controller_id);
	if (tracker.is_valid()) {
		int joyid = tracker->get_joy_id();
		if (joyid != -1) {
			input->joy_button(joyid, p_button, p_is_pressed);
		}
	}
}

void GDAPI godot_xr_set_controller_axis(godot_int p_controller_id, godot_int p_axis, godot_float p_value, godot_bool p_can_be_negative) {
	XRServer *xr_server = XRServer::get_singleton();
	ERR_FAIL_NULL(xr_server);

	Input *input = Input::get_singleton();
	ERR_FAIL_NULL(input);

	Ref<XRPositionalTracker> tracker = xr_server->find_by_type_and_id(XRServer::TRACKER_CONTROLLER, p_controller_id);
	if (tracker.is_valid()) {
		int joyid = tracker->get_joy_id();
		if (joyid != -1) {
			Input::JoyAxisValue jx;
			jx.min = p_can_be_negative ? -1 : 0;
			jx.value = p_value;
			input->joy_axis(joyid, p_axis, jx);
		}
	}
}

godot_float GDAPI godot_xr_get_controller_rumble(godot_int p_controller_id) {
	XRServer *xr_server = XRServer::get_singleton();
	ERR_FAIL_NULL_V(xr_server, 0.0);

	Ref<XRPositionalTracker> tracker = xr_server->find_by_type_and_id(XRServer::TRACKER_CONTROLLER, p_controller_id);
	if (tracker.is_valid()) {
		return tracker->get_rumble();
	}

	return 0.0;
}
}
