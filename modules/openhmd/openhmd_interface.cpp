/*************************************************************************/
/*  openhmd_interface.cpp                                                */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
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

#include "openhmd_interface.h"
#include "core/core_string_names.h"
#include "core/os/os.h"
#include "os/os.h"
#include "project_settings.h"
#include "scene/scene_string_names.h"
#include "servers/visual/visual_server_global.h"

// make this settable?
#define OVERSAMPLE_SCALE 2.0

StringName OpenHMDInterface::get_name() const {
	return "OpenHMD";
};

int OpenHMDInterface::get_capabilities() const {
	return ARVRInterface::ARVR_STEREO; // + ARVRInterface::ARVR_EXTERNAL once we open our own rendering window
};

void OpenHMDInterface::_bind_methods() {
	ClassDB::bind_method(D_METHOD("scan_for_devices"), &OpenHMDInterface::scan_for_devices);

	ClassDB::bind_method(D_METHOD("get_device_count"), &OpenHMDInterface::get_device_count);
	ClassDB::bind_method(D_METHOD("get_device_names"), &OpenHMDInterface::get_device_names);

	ClassDB::bind_method(D_METHOD("set_auto_init_device_zero", "auto_init"), &OpenHMDInterface::set_auto_init_device_zero);
	ClassDB::bind_method(D_METHOD("auto_init_device_zero"), &OpenHMDInterface::auto_init_device_zero);

	ClassDB::bind_method(D_METHOD("init_hmd_device", "device_no"), &OpenHMDInterface::init_hmd_device);
	ClassDB::bind_method(D_METHOD("close_hmd_device"), &OpenHMDInterface::close_hmd_device);

	ClassDB::bind_method(D_METHOD("init_tracking_device", "device_no"), &OpenHMDInterface::init_tracking_device);
	ClassDB::bind_method(D_METHOD("close_tracking_device"), &OpenHMDInterface::close_tracking_device);

	ClassDB::bind_method(D_METHOD("init_controller_device", "device_no"), &OpenHMDInterface::init_controller_device);
};

void OpenHMDInterface::scan_for_devices() {
	ERR_FAIL_NULL(ohmd_ctx);

	// Calling ohmd_ctx_probe will initialize our list of active devices.
	// Until it is called again our indices should not change.
	num_devices = ohmd_ctx_probe(ohmd_ctx);
	if (num_devices < 0) {
		String err_text = ohmd_ctx_get_error(ohmd_ctx);
		print_line("OpenHMD: failed to get device count - " + err_text);
		return;
	};
};

bool OpenHMDInterface::auto_init_device_zero() const {
	return do_auto_init_device_zero;
};

void OpenHMDInterface::set_auto_init_device_zero(bool p_auto_init) {
	do_auto_init_device_zero = p_auto_init;
};

bool OpenHMDInterface::init_hmd_device(int p_device) {
	ERR_FAIL_NULL_V(ohmd_ctx, false);

	if (hmd_device != NULL) {
		close_hmd_device();
	};

	if (num_devices <= p_device) {
		print_line("OpenHMD: Device ID out of bounds");
		return false;
	};

	print_line("Initialising device no " + itos(p_device) + " as the HMD device");

	// create our device instance
	hmd_device = ohmd_list_open_device_s(ohmd_ctx, p_device, ohmd_settings);
	if (hmd_device == NULL) {
		String err_text = ohmd_ctx_get_error(ohmd_ctx);
		print_line("OpenHMD: failed to open device - " + err_text);
		return false;
	} else {
		// get resolution
		ohmd_device_geti(hmd_device, OHMD_SCREEN_HORIZONTAL_RESOLUTION, &width);
		width /= 2; /* need half this */
		ohmd_device_geti(hmd_device, OHMD_SCREEN_VERTICAL_RESOLUTION, &height);

		// now copy some of these into our shader..
		if (ohmd_shader != NULL) {
			ohmd_shader->set_device_parameters(hmd_device);
		};

		// need to check if we can actually use this device!

		String device_name = ohmd_list_gets(ohmd_ctx, p_device, OHMD_VENDOR);
		device_name += " - ";
		device_name += ohmd_list_gets(ohmd_ctx, p_device, OHMD_PRODUCT);

		print_line("OpenHMD: initialized hmd " + device_name);
	};

	return true;
};

void OpenHMDInterface::close_hmd_device() {
	if (hmd_device != NULL) {
		print_line("Closing HMD OpenHMD device");

		ohmd_close_device(hmd_device);

		hmd_device = NULL;
	};
};

bool OpenHMDInterface::init_tracking_device(int p_device) {
	ERR_FAIL_NULL_V(ohmd_ctx, false);

	if (tracking_device != NULL) {
		close_tracking_device();
	};

	if (num_devices <= p_device) {
		print_line("OpenHMD: Device ID out of bounds");
		return false;
	};

	print_line("Initialising device no " + itos(p_device) + " as the tracking device");

	// create our device instance
	tracking_device = ohmd_list_open_device_s(ohmd_ctx, p_device, ohmd_settings);
	if (tracking_device == NULL) {
		String err_text = ohmd_ctx_get_error(ohmd_ctx);
		print_line("OpenHMD: failed to open device - " + err_text);
		return false;
	} else {
		// need to check if we can actually use this device!

		String device_name = ohmd_list_gets(ohmd_ctx, p_device, OHMD_VENDOR);
		device_name += " - ";
		device_name += ohmd_list_gets(ohmd_ctx, p_device, OHMD_PRODUCT);

		print_line("OpenHMD: initialized tracking device " + device_name);
	};

	return true;
};

void OpenHMDInterface::close_tracking_device() {
	if (tracking_device != NULL) {
		print_line("Closing OpenHMD tracking device");

		ohmd_close_device(tracking_device);

		tracking_device = NULL;
	};
};

bool OpenHMDInterface::init_controller_device(int p_device) {
	// I believe this should be an internal function with process adding/removing controller devices as they are turned on/off by the player.
	// This way the programmer writing a Godot game doesn't need to worry about handling this.
	// But for now having to call this manually will do for a first attempt at getting this to work...

	ERR_FAIL_NULL_V(ohmd_ctx, false);

	if (num_devices <= p_device) {
		print_line("OpenHMD: Device ID out of bounds");
		return false;
	};

	print_line("Initialising device no " + itos(p_device) + " as a controller device");

	// create our device instance
	ohmd_device *device = ohmd_list_open_device_s(ohmd_ctx, p_device, ohmd_settings);
	if (device == NULL) {
		String err_text = ohmd_ctx_get_error(ohmd_ctx);
		print_line("OpenHMD: failed to open controller - " + err_text);
		return false;
	} else {
		// need to check if we can actually use this device!

		int idx = add_controller_device(device);
		if (idx == -1) {
			// failed to add?
			ohmd_close_device(device);
		}
	};

	return true;
};

int OpenHMDInterface::get_device_count() const {
	if (ohmd_ctx == NULL) {
		return 0;
	} else {
		return num_devices;
	};
};

Array OpenHMDInterface::get_device_names() const {
	Array arr;

	if (ohmd_ctx != NULL && num_devices > 0) {
		for (int i = 0; i < num_devices; i++) {
			String device_name = ohmd_list_gets(ohmd_ctx, i, OHMD_VENDOR);
			device_name += " - ";
			device_name += ohmd_list_gets(ohmd_ctx, i, OHMD_PRODUCT);
			arr.push_back(device_name);
		};
	};

	return arr;
};

bool OpenHMDInterface::is_stereo() {
	// needs stereo...
	return true;
};

bool OpenHMDInterface::is_initialized() {
	return ohmd_ctx != NULL;
};

bool OpenHMDInterface::initialize() {
	ARVRServer *arvr_server = ARVRServer::get_singleton();
	ERR_FAIL_NULL_V(arvr_server, false);

	if (ohmd_ctx == NULL) {
		print_line("Initialising OpenHMD");

		// initialize openhmd
		ohmd_ctx = ohmd_ctx_create();

		// should we build this once and just remember it? or keep building it like this?
		ohmd_settings = ohmd_device_settings_create(ohmd_ctx);

		// we turn our automatic updates off, we're calling this from our process call which is called from our main render thread
		// which guarantees this gets called atleast once every frame and as close to our rendering as possible.
		int auto_update = 0;
		ohmd_device_settings_seti(ohmd_settings, OHMD_IDS_AUTOMATIC_UPDATE, &auto_update);

		// create our lens distortion shader
		ohmd_shader = new OpenHMDShader();

		// populate our initial list of available devices
		scan_for_devices();

		// initialize our first device?
		if (do_auto_init_device_zero) {
			init_hmd_device(0);
		};

		// make this our primary interface
		arvr_server->set_primary_interface(this);
	};

	return true;
};

void OpenHMDInterface::uninitialize() {
	if (ohmd_ctx != NULL) {
		ARVRServer *arvr_server = ARVRServer::get_singleton();
		if (arvr_server != NULL) {
			// no longer our primary interface
			arvr_server->clear_primary_interface_if(this);
		}

		while (controller_tracker_mapping.size() > 0) {
			// hmm, a bit overdone but....
			ohmd_device *device = controller_tracker_mapping[0].controller_device;
			remove_controller_device(device);
		}

		close_tracking_device();
		close_hmd_device();

		ohmd_device_settings_destroy(ohmd_settings);
		ohmd_ctx_destroy(ohmd_ctx);
		ohmd_ctx = NULL;
	};

	if (ohmd_shader != NULL) {
		delete ohmd_shader;
		ohmd_shader = NULL;
	};
};

Size2 OpenHMDInterface::get_render_targetsize() {
	_THREAD_SAFE_METHOD_

	Size2 target_size;

	if (hmd_device != NULL) {
		target_size.x = width * OVERSAMPLE_SCALE;
		target_size.y = height * OVERSAMPLE_SCALE;
	} else {
		/* just return something so we can show something instead of crashing */
		target_size.x = 600;
		target_size.y = 900;
	};

	return target_size;
};

// Converts a openhmd matrix to a transform, note, don't use this for a projection matrix!
Transform OpenHMDInterface::get_ohmd_matrix_as_transform(ohmd_device *p_device, ohmd_float_value p_type, float p_position_scale) {
	Transform newtransform;

	if (p_device != NULL) {
		float m[4][4];
		ohmd_device_getf(p_device, p_type, (float *)m);

		///@TODO row vs column?
		newtransform.basis.set(
				m[0][0], m[0][1], m[0][2],
				m[1][0], m[1][1], m[1][2],
				m[2][0], m[2][1], m[2][2]);

		newtransform.origin.x = m[0][3] * p_position_scale;
		newtransform.origin.y = m[1][3] * p_position_scale;
		newtransform.origin.z = m[2][3] * p_position_scale;
	};

	return newtransform;
};

Transform OpenHMDInterface::get_ohmd_rot_pos_as_transform(ohmd_device *p_device, float p_position_scale) {
	Transform newtransform;

	if (p_device != NULL) {
		// construct it from source, this is purely called when we're creating our reference frame
		float ohmd_q[4];

		// convert orientation quad to position, should add helper function for this :)
		Quat q;
		ohmd_device_getf(p_device, OHMD_ROTATION_QUAT, ohmd_q);
		q.x = ohmd_q[0];
		q.y = ohmd_q[1];
		q.z = ohmd_q[2];
		q.w = ohmd_q[3];
		newtransform.basis = Basis(q);

		float ohmd_v[4];
		ohmd_device_getf(p_device, OHMD_POSITION_VECTOR, ohmd_v);
		newtransform.origin = Vector3(ohmd_v[0], ohmd_v[1], ohmd_v[2]);
	};

	return newtransform;
};

Transform OpenHMDInterface::get_transform_for_eye(ARVRInterface::Eyes p_eye, const Transform &p_cam_transform) {
	_THREAD_SAFE_METHOD_

	Transform transform_for_eye;

	ARVRServer *arvr_server = ARVRServer::get_singleton();
	ERR_FAIL_NULL_V(arvr_server, transform_for_eye);
	float world_scale = arvr_server->get_world_scale();

	if (tracking_device != NULL) {
		// Our tracker will only have location and position data, OHMD_LEFT_EYE_GL_MODELVIEW_MATRIX and OHMD_RIGHT_EYE_GL_MODELVIEW_MATRIX would return the same thing
		transform_for_eye = get_ohmd_rot_pos_as_transform(tracking_device, world_scale);

		// now we manually add our IPD from our HMD
		if (hmd_device != NULL) {
			float ipd;
			ohmd_device_getf(hmd_device, OHMD_EYE_IPD, &ipd);

			if (p_eye == ARVRInterface::EYE_LEFT) {
				Transform t_ipd;
				t_ipd.origin.x = world_scale * ipd / 2.0;
				transform_for_eye *= t_ipd;
			} else if (p_eye == ARVRInterface::EYE_RIGHT) {
				Transform t_ipd;
				t_ipd.origin.x = world_scale * -ipd / 2.0;
				transform_for_eye *= t_ipd;
			};
		};

		// and build our final transform
		transform_for_eye = p_cam_transform * (arvr_server->get_reference_frame()) * transform_for_eye;

	} else if (hmd_device != NULL) {
		// Get our view matrices from OpenHMD
		if (p_eye == ARVRInterface::EYE_LEFT) {
			transform_for_eye = get_ohmd_matrix_as_transform(hmd_device, OHMD_LEFT_EYE_GL_MODELVIEW_MATRIX, world_scale);
		} else if (p_eye == ARVRInterface::EYE_RIGHT) {
			transform_for_eye = get_ohmd_matrix_as_transform(hmd_device, OHMD_RIGHT_EYE_GL_MODELVIEW_MATRIX, world_scale);
		} else {
			// 'mono' will be requested purely for scene positioning feedback, no longer used by renderer
			transform_for_eye = get_ohmd_rot_pos_as_transform(hmd_device, world_scale);
		};

		// and build our final transform
		transform_for_eye = p_cam_transform * (arvr_server->get_reference_frame()) * transform_for_eye;
	} else {
		// huh? we'll just return what we got....
		transform_for_eye = p_cam_transform;
	};

	return transform_for_eye;
};

CameraMatrix OpenHMDInterface::get_ohmd_matrix_as_camera_matrix(ohmd_float_value p_type) {
	float m[4][4];
	CameraMatrix newcamera;

	if (hmd_device != NULL) {
		ohmd_device_getf(hmd_device, p_type, (float *)m);

		///@TODO row vs column?
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				newcamera.matrix[i][j] = m[i][j];
			};
		};
	};

	return newcamera;
};

CameraMatrix OpenHMDInterface::get_projection_for_eye(ARVRInterface::Eyes p_eye, real_t p_aspect, real_t p_z_near, real_t p_z_far) {
	_THREAD_SAFE_METHOD_

	if (hmd_device != NULL) {
		float z_near = p_z_near;
		float z_far = p_z_far;
		ohmd_device_setf(hmd_device, OHMD_PROJECTION_ZNEAR, &z_near);
		ohmd_device_setf(hmd_device, OHMD_PROJECTION_ZFAR, &z_far);

		return get_ohmd_matrix_as_camera_matrix(p_eye == ARVRInterface::EYE_LEFT ? OHMD_LEFT_EYE_GL_PROJECTION_MATRIX : OHMD_RIGHT_EYE_GL_PROJECTION_MATRIX);
	} else {
		// just return a normal camera
		CameraMatrix eye;
		eye.set_perspective(60.0, p_aspect, p_z_near, p_z_far, false);
		return eye;
	};
};

void OpenHMDInterface::commit_for_eye(ARVRInterface::Eyes p_eye, RID p_render_target, const Rect2 &p_screen_rect) {
	_THREAD_SAFE_METHOD_

	// We must have a valid render target
	ERR_FAIL_COND(!p_render_target.is_valid());

	// Because we are rendering to our device we must use our main viewport!
	ERR_FAIL_COND(p_screen_rect == Rect2());

	// output to main
	VSG::rasterizer->set_current_render_target(RID());

	// is our shader setup?
	if (hmd_device != NULL && ohmd_shader != NULL) {
		// get our color buffer texture id
		RID eye_texture = VSG::storage->render_target_get_texture(p_render_target);
		uint32_t texid = VS::get_singleton()->texture_get_texid(eye_texture);

		ohmd_shader->render_eye(texid, p_eye == ARVRInterface::EYE_LEFT ? 0 : 1);
	} else {
		// fall back on blitting to screen as is

		// adjust our screen rect
		Rect2 screen_rect = p_screen_rect;

		if (p_eye == ARVRInterface::EYE_LEFT) {
			screen_rect.size.x /= 2.0;
		} else if (p_eye == ARVRInterface::EYE_RIGHT) {
			screen_rect.size.x /= 2.0;
			screen_rect.position.x += screen_rect.size.x;
		}

		VSG::rasterizer->blit_render_target_to_screen(p_render_target, screen_rect, 0);
	};
};

int OpenHMDInterface::add_controller_device(ohmd_device *p_device) {
	ARVRServer *arvr_server = ARVRServer::get_singleton();
	InputDefault *input = (InputDefault *)Input::get_singleton();
	ERR_FAIL_COND_V(p_device == NULL, -1);

	///@TODO check if our device is indeed a controller...

	// see if we have already added our controller
	for (int i = 0; i < controller_tracker_mapping.size(); i++) {

		if (controller_tracker_mapping[i].controller_device == p_device) {
			ERR_PRINT("Controller was already added");
			return -1;
		};
	};

	// create tracker
	char device_name[256];
	ARVRPositionalTracker *new_tracker = memnew(ARVRPositionalTracker);
	new_tracker->set_type(ARVRServer::TRACKER_CONTROLLER);

	///@TODO should see if we can get a name from our device...
	sprintf(device_name, "Controller %li", new_tracker->get_tracker_id());
	new_tracker->set_name(device_name);

	// init these to set our flags
	Basis orientation;
	new_tracker->set_orientation(orientation);
	Vector3 position;
	new_tracker->set_position(position);

	// also register as joystick...
	int joyid = input->get_unused_joy_id();
	if (joyid != -1) {
		new_tracker->set_joy_id(joyid);
		input->joy_connection_changed(joyid, true, device_name, "");
	};

	// add new controller mapping
	ohmd_controller_tracker new_map;
	new_map.controller_device = p_device;
	new_map.controller_tracker = new_tracker;
	controller_tracker_mapping.push_back(new_map);

	// register our tracker
	arvr_server->add_tracker(new_tracker);

	// and return our new index
	return controller_tracker_mapping.size() - 1;
};

void OpenHMDInterface::remove_controller_device(ohmd_device *p_device) {
	ARVRServer *arvr_server = ARVRServer::get_singleton();
	InputDefault *input = (InputDefault *)Input::get_singleton();
	ERR_FAIL_COND(p_device == NULL);

	// Find and remove all copies
	for (int i = controller_tracker_mapping.size() - 1; i >= 0; i--) {
		if (controller_tracker_mapping[i].controller_device == p_device) {
			// copy our map
			ohmd_controller_tracker remove_map = controller_tracker_mapping[i];

			// remove our mapping
			controller_tracker_mapping.remove(i);

			// remove our joystick
			int joyid = remove_map.controller_tracker->get_joy_id();
			if (joyid != -1) {
				input->joy_connection_changed(joyid, false, "", "");
				remove_map.controller_tracker->set_joy_id(-1);
			};

			// remove and destroy our tracker
			arvr_server->remove_tracker(remove_map.controller_tracker);
			memdelete(remove_map.controller_tracker);

			// free our device
			ohmd_close_device(remove_map.controller_device);
		};
	};
};

void OpenHMDInterface::process() {
	_THREAD_SAFE_METHOD_

	if (ohmd_ctx != NULL) {
		ohmd_ctx_update(ohmd_ctx);

		///@TODO add code here to check if we have new controllers that were turned on by the user or if
		// controllers were turned off.

		// check all our existing controllers and copy data
		for (int i = 0; i < controller_tracker_mapping.size(); i++) {
			ohmd_device *device = controller_tracker_mapping[i].controller_device;
			ARVRPositionalTracker *tracker = controller_tracker_mapping[i].controller_tracker;
			Transform controller_transform = get_ohmd_rot_pos_as_transform(device, 1.0);

			tracker->set_orientation(controller_transform.basis);

			float ohmd_v[4];
			ohmd_device_getf(device, OHMD_POSITION_VECTOR, ohmd_v);
			tracker->set_rw_position(controller_transform.origin);
		};
	};
};

OpenHMDInterface::OpenHMDInterface() {
	ohmd_ctx = NULL;
	hmd_device = NULL;
	tracking_device = NULL;
	do_auto_init_device_zero = true;
	ohmd_shader = NULL;
	num_devices = -1;
};

OpenHMDInterface::~OpenHMDInterface() {
	// and make sure we cleanup if we haven't already
	if (is_initialized()) {
		uninitialize();
	};
};
