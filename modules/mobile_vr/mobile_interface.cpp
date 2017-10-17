/*************************************************************************/
/*  mobile_interface.cpp                                                 */
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

#include "mobile_interface.h"
#include "core/os/input.h"
#include "core/os/os.h"
#include "servers/visual/visual_server_global.h"

StringName MobileVRInterface::get_name() const {
	return "Native mobile";
};

int MobileVRInterface::get_capabilities() const {
	return ARVRInterface::ARVR_STEREO;
};

Vector3 MobileVRInterface::scale_magneto(const Vector3 &p_magnetometer) {
	// Our magnetometer doesn't give us nice clean data.
	// Well it may on Mac OS X because we're getting a calibrated value in the current implementation but Android we're getting raw data.
	// This is a fairly simple adjustment we can do to correct for the magnetometer data being elliptical

	Vector3 mag_raw = p_magnetometer;
	Vector3 mag_scaled = p_magnetometer;

	// update our variables every x frames
	if (mag_count > 20) {
		mag_current_min = mag_next_min;
		mag_current_max = mag_next_max;
		mag_count = 0;
	} else {
		mag_count++;
	};

	// adjust our min and max
	if (mag_raw.x > mag_next_max.x) mag_next_max.x = mag_raw.x;
	if (mag_raw.y > mag_next_max.y) mag_next_max.y = mag_raw.y;
	if (mag_raw.z > mag_next_max.z) mag_next_max.z = mag_raw.z;

	if (mag_raw.x < mag_next_min.x) mag_next_min.x = mag_raw.x;
	if (mag_raw.y < mag_next_min.y) mag_next_min.y = mag_raw.y;
	if (mag_raw.z < mag_next_min.z) mag_next_min.z = mag_raw.z;

	// scale our x, y and z
	if (!(mag_current_max.x - mag_current_min.x)) {
		mag_raw.x -= (mag_current_min.x + mag_current_max.x) / 2.0;
		mag_scaled.x = (mag_raw.x - mag_current_min.x) / ((mag_current_max.x - mag_current_min.x) * 2.0 - 1.0);
	};

	if (!(mag_current_max.y - mag_current_min.y)) {
		mag_raw.y -= (mag_current_min.y + mag_current_max.y) / 2.0;
		mag_scaled.y = (mag_raw.y - mag_current_min.y) / ((mag_current_max.y - mag_current_min.y) * 2.0 - 1.0);
	};

	if (!(mag_current_max.z - mag_current_min.z)) {
		mag_raw.z -= (mag_current_min.z + mag_current_max.z) / 2.0;
		mag_scaled.z = (mag_raw.z - mag_current_min.z) / ((mag_current_max.z - mag_current_min.z) * 2.0 - 1.0);
	};

	return mag_scaled;
};

Basis MobileVRInterface::combine_acc_mag(const Vector3 &p_grav, const Vector3 &p_magneto) {
	// yup, stock standard cross product solution...
	Vector3 up = -p_grav.normalized();

	Vector3 magneto_east = up.cross(p_magneto.normalized()); // or is this west?, but should be horizon aligned now
	magneto_east.normalize();

	Vector3 magneto = up.cross(magneto_east); // and now we have a horizon aligned north
	magneto.normalize();

	// We use our gravity and magnetometer vectors to construct our matrix
	Basis acc_mag_m3;
	acc_mag_m3.elements[0] = -magneto_east;
	acc_mag_m3.elements[1] = up;
	acc_mag_m3.elements[2] = magneto;

	return acc_mag_m3;
};

void MobileVRInterface::set_position_from_sensors() {
	_THREAD_SAFE_METHOD_

	// this is a helper function that attempts to adjust our transform using our 9dof sensors
	// 9dof is a misleading marketing term coming from 3 accelerometer axis + 3 gyro axis + 3 magnetometer axis = 9 axis
	// but in reality this only offers 3 dof (yaw, pitch, roll) orientation

	uint64_t ticks = OS::get_singleton()->get_ticks_usec();
	uint64_t ticks_elapsed = ticks - last_ticks;
	float delta_time = (double)ticks_elapsed / 1000000.0;

	// few things we need
	Input *input = Input::get_singleton();
	Vector3 down(0.0, -1.0, 0.0); // Down is Y negative
	Vector3 north(0.0, 0.0, 1.0); // North is Z positive

	// make copies of our inputs
	Vector3 acc = input->get_accelerometer();
	Vector3 gyro = input->get_gyroscope();
	Vector3 grav = input->get_gravity();
	Vector3 magneto = scale_magneto(input->get_magnetometer()); // this may be overkill on iOS because we're already getting a calibrated magnetometer reading

	if (sensor_first) {
		sensor_first = false;
	} else {
		acc = scrub(acc, last_accerometer_data, 2, 0.2);
		magneto = scrub(magneto, last_magnetometer_data, 3, 0.3);
	};

	last_accerometer_data = acc;
	last_magnetometer_data = magneto;

	if (grav.length() < 0.1) {
		// not ideal but use our accelerometer, this will contain shakey shakey user behaviour
		// maybe look into some math but I'm guessing that if this isn't available, its because we lack the gyro sensor to actually work out
		// what a stable gravity vector is
		grav = acc;
		if (grav.length() > 0.1) {
			has_gyro = true;
		};
	} else {
		has_gyro = true;
	};

	bool has_magneto = magneto.length() > 0.1;
	bool has_grav = grav.length() > 0.1;

#ifdef ANDROID_ENABLED
	///@TODO needs testing, i don't have a gyro, potentially can be removed depending on what comes out of issue #8101
	// On Android x and z axis seem inverted
	gyro.x = -gyro.x;
	gyro.z = -gyro.z;
	grav.x = -grav.x;
	grav.z = -grav.z;
	magneto.x = -magneto.x;
	magneto.z = -magneto.z;
#endif

	if (has_gyro) {
		// start with applying our gyro (do NOT smooth our gyro!)
		Basis rotate;
		rotate.rotate(orientation.get_axis(0), gyro.x * delta_time);
		rotate.rotate(orientation.get_axis(1), gyro.y * delta_time);
		rotate.rotate(orientation.get_axis(2), gyro.z * delta_time);
		orientation = rotate * orientation;

		tracking_state = ARVRInterface::ARVR_NORMAL_TRACKING;
	};

	///@TODO improve this, the magnetometer is very fidgity sometimes flipping the axis for no apparent reason (probably a bug on my part)
	// if you have a gyro + accelerometer that combo tends to be better then combining all three but without a gyro you need the magnetometer..
	if (has_magneto && has_grav && !has_gyro) {
		// convert to quaternions, easier to smooth those out
		Quat transform_quat(orientation);
		Quat acc_mag_quat(combine_acc_mag(grav, magneto));
		transform_quat = transform_quat.slerp(acc_mag_quat, 0.1);
		orientation = Basis(transform_quat);

		tracking_state = ARVRInterface::ARVR_NORMAL_TRACKING;
	} else if (has_grav) {
		// use gravity vector to make sure down is down...
		// transform gravity into our world space
		grav.normalize();
		Vector3 grav_adj = orientation.xform(grav);
		float dot = grav_adj.dot(down);
		if ((dot > -1.0) && (dot < 1.0)) {
			// axis around which we have this rotation
			Vector3 axis = grav_adj.cross(down);
			axis.normalize();

			Basis drift_compensation(axis, acos(dot) * delta_time * 10);
			orientation = drift_compensation * orientation;
		};
	};

	// JIC
	orientation.orthonormalize();

	last_ticks = ticks;
};

void MobileVRInterface::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_iod", "iod"), &MobileVRInterface::set_iod);
	ClassDB::bind_method(D_METHOD("get_iod"), &MobileVRInterface::get_iod);

	ClassDB::bind_method(D_METHOD("set_display_width", "display_width"), &MobileVRInterface::set_display_width);
	ClassDB::bind_method(D_METHOD("get_display_width"), &MobileVRInterface::get_display_width);

	ClassDB::bind_method(D_METHOD("set_display_to_lens", "display_to_lens"), &MobileVRInterface::set_display_to_lens);
	ClassDB::bind_method(D_METHOD("get_display_to_lens"), &MobileVRInterface::get_display_to_lens);

	ClassDB::bind_method(D_METHOD("set_oversample", "oversample"), &MobileVRInterface::set_oversample);
	ClassDB::bind_method(D_METHOD("get_oversample"), &MobileVRInterface::get_oversample);

	ClassDB::bind_method(D_METHOD("set_k1", "k"), &MobileVRInterface::set_k1);
	ClassDB::bind_method(D_METHOD("get_k1"), &MobileVRInterface::get_k1);

	ClassDB::bind_method(D_METHOD("set_k2", "k"), &MobileVRInterface::set_k2);
	ClassDB::bind_method(D_METHOD("get_k2"), &MobileVRInterface::get_k2);

	ADD_PROPERTY(PropertyInfo(Variant::REAL, "iod", PROPERTY_HINT_RANGE, "4.0,10.0,0.1"), "set_iod", "get_iod");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "display_width", PROPERTY_HINT_RANGE, "5.0,25.0,0.1"), "set_display_width", "get_display_width");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "display_to_lens", PROPERTY_HINT_RANGE, "5.0,25.0,0.1"), "set_display_to_lens", "get_display_to_lens");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "oversample", PROPERTY_HINT_RANGE, "1.0,2.0,0.1"), "set_oversample", "get_oversample");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "k1", PROPERTY_HINT_RANGE, "0.1,10.0,0.0001"), "set_k1", "get_k1");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "k2", PROPERTY_HINT_RANGE, "0.1,10.0,0.0001"), "set_k2", "get_k2");
}

void MobileVRInterface::set_iod(const real_t p_iod) {
	intraocular_dist = p_iod;
};

real_t MobileVRInterface::get_iod() const {
	return intraocular_dist;
};

void MobileVRInterface::set_display_width(const real_t p_display_width) {
	display_width = p_display_width;
};

real_t MobileVRInterface::get_display_width() const {
	return display_width;
};

void MobileVRInterface::set_display_to_lens(const real_t p_display_to_lens) {
	display_to_lens = p_display_to_lens;
};

real_t MobileVRInterface::get_display_to_lens() const {
	return display_to_lens;
};

void MobileVRInterface::set_oversample(const real_t p_oversample) {
	oversample = p_oversample;
};

real_t MobileVRInterface::get_oversample() const {
	return oversample;
};

void MobileVRInterface::set_k1(const real_t p_k1) {
	k1 = p_k1;
};

real_t MobileVRInterface::get_k1() const {
	return k1;
};

void MobileVRInterface::set_k2(const real_t p_k2) {
	k2 = p_k2;
};

real_t MobileVRInterface::get_k2() const {
	return k2;
};

bool MobileVRInterface::is_stereo() {
	// needs stereo...
	return true;
};

bool MobileVRInterface::is_initialized() {
	return (initialized);
};

bool MobileVRInterface::initialize() {
	ARVRServer *arvr_server = ARVRServer::get_singleton();
	ERR_FAIL_NULL_V(arvr_server, false);

	if (!initialized) {
		// reset our sensor data and orientation
		mag_count = 0;
		has_gyro = false;
		sensor_first = true;
		mag_next_min = Vector3(10000, 10000, 10000);
		mag_next_max = Vector3(-10000, -10000, -10000);
		mag_current_min = Vector3(0, 0, 0);
		mag_current_max = Vector3(0, 0, 0);

		// reset our orientation
		orientation = Basis();

		// make this our primary interface
		arvr_server->set_primary_interface(this);

		last_ticks = OS::get_singleton()->get_ticks_usec();
		;
		initialized = true;
	};

	return true;
};

void MobileVRInterface::uninitialize() {
	if (initialized) {
		ARVRServer *arvr_server = ARVRServer::get_singleton();
		if (arvr_server != NULL) {
			// no longer our primary interface
			arvr_server->clear_primary_interface_if(this);
		}

		initialized = false;
	};
};

Size2 MobileVRInterface::get_recommended_render_targetsize() {
	_THREAD_SAFE_METHOD_

	// we use half our window size
	Size2 target_size = OS::get_singleton()->get_window_size();
	target_size.x *= 0.5 * oversample;
	target_size.y *= oversample;

	return target_size;
};

Transform MobileVRInterface::get_transform_for_eye(ARVRInterface::Eyes p_eye, const Transform &p_cam_transform) {
	_THREAD_SAFE_METHOD_

	Transform transform_for_eye;

	ARVRServer *arvr_server = ARVRServer::get_singleton();
	ERR_FAIL_NULL_V(arvr_server, transform_for_eye);

	if (initialized) {
		float world_scale = arvr_server->get_world_scale();

		// we don't need to check for the existance of our HMD, doesn't effect our values...
		// note * 0.01 to convert cm to m and * 0.5 as we're moving half in each direction...
		if (p_eye == ARVRInterface::EYE_LEFT) {
			transform_for_eye.origin.x = -(intraocular_dist * 0.01 * 0.5 * world_scale);
		} else if (p_eye == ARVRInterface::EYE_RIGHT) {
			transform_for_eye.origin.x = intraocular_dist * 0.01 * 0.5 * world_scale;
		} else {
			// for mono we don't reposition, we want our center position.
		};

		// just scale our origin point of our transform
		Transform hmd_transform;
		hmd_transform.basis = orientation;
		hmd_transform.origin = Vector3(0.0, eye_height * world_scale, 0.0);

		transform_for_eye = p_cam_transform * (arvr_server->get_reference_frame()) * hmd_transform * transform_for_eye;
	} else {
		// huh? well just return what we got....
		transform_for_eye = p_cam_transform;
	};

	return transform_for_eye;
};

CameraMatrix MobileVRInterface::get_projection_for_eye(ARVRInterface::Eyes p_eye, real_t p_aspect, real_t p_z_near, real_t p_z_far) {
	_THREAD_SAFE_METHOD_

	CameraMatrix eye;

	if (p_eye == ARVRInterface::EYE_MONO) {
		///@TODO for now hardcode some of this, what is really needed here is that this needs to be in sync with the real cameras properties
		// which probably means implementing a specific class for iOS and Android. For now this is purely here as an example.
		// Note also that if you use a normal viewport with AR/VR turned off you can still use the tracker output of this interface
		// to position a stock standard Godot camera and have control over this.
		// This will make more sense when we implement ARkit on iOS (probably a separate interface).
		eye.set_perspective(60.0, p_aspect, p_z_near, p_z_far, false);
	} else {
		eye.set_for_hmd(p_eye == ARVRInterface::EYE_LEFT ? 1 : 2, p_aspect, intraocular_dist, display_width, display_to_lens, oversample, p_z_near, p_z_far);
	};

	return eye;
};

void MobileVRInterface::commit_for_eye(ARVRInterface::Eyes p_eye, RID p_render_target, const Rect2 &p_screen_rect) {
	_THREAD_SAFE_METHOD_

	// We must have a valid render target
	ERR_FAIL_COND(!p_render_target.is_valid());

	// Because we are rendering to our device we must use our main viewport!
	ERR_FAIL_COND(p_screen_rect == Rect2());

	float offset_x = 0.0;
	float aspect_ratio = 0.5 * p_screen_rect.size.x / p_screen_rect.size.y;
	Vector2 eye_center;

	if (p_eye == ARVRInterface::EYE_LEFT) {
		offset_x = -1.0;
		eye_center.x = ((-intraocular_dist / 2.0) + (display_width / 4.0)) / (display_width / 2.0);
	} else if (p_eye == ARVRInterface::EYE_RIGHT) {
		eye_center.x = ((intraocular_dist / 2.0) - (display_width / 4.0)) / (display_width / 2.0);
	}

	// unset our render target so we are outputting to our main screen by making RasterizerStorageGLES3::system_fbo our current FBO
	VSG::rasterizer->set_current_render_target(RID());

	// now output to screen
	//	VSG::rasterizer->blit_render_target_to_screen(p_render_target, screen_rect, 0);

	// get our render target
	RID eye_texture = VSG::storage->render_target_get_texture(p_render_target);
	uint32_t texid = VS::get_singleton()->texture_get_texid(eye_texture);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, texid);

	lens_shader.bind();
	lens_shader.set_uniform(LensDistortedShaderGLES3::OFFSET_X, offset_x);
	lens_shader.set_uniform(LensDistortedShaderGLES3::K1, k1);
	lens_shader.set_uniform(LensDistortedShaderGLES3::K2, k2);
	lens_shader.set_uniform(LensDistortedShaderGLES3::EYE_CENTER, eye_center);
	lens_shader.set_uniform(LensDistortedShaderGLES3::UPSCALE, oversample);
	lens_shader.set_uniform(LensDistortedShaderGLES3::ASPECT_RATIO, aspect_ratio);

	glBindVertexArray(half_screen_array);
	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	glBindVertexArray(0);
};

void MobileVRInterface::process() {
	_THREAD_SAFE_METHOD_

	if (initialized) {
		set_position_from_sensors();
	};
};

MobileVRInterface::MobileVRInterface() {
	initialized = false;

	// Just set some defaults for these. At some point we need to look at adding a lookup table for common device + headset combos and/or support reading cardboard QR codes
	eye_height = 1.85;
	intraocular_dist = 6.0;
	display_width = 14.5;
	display_to_lens = 4.0;
	oversample = 1.5;
	k1 = 0.215;
	k2 = 0.215;
	last_ticks = 0;

	// create our shader stuff
	lens_shader.init();

	{
		glGenBuffers(1, &half_screen_quad);
		glBindBuffer(GL_ARRAY_BUFFER, half_screen_quad);
		{
			const float qv[16] = {
				0, -1,
				-1, -1,
				0, 1,
				-1, 1,
				1, 1,
				1, 1,
				1, -1,
				1, -1,
			};

			glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 16, qv, GL_STATIC_DRAW);
		}

		glBindBuffer(GL_ARRAY_BUFFER, 0); //unbind

		glGenVertexArrays(1, &half_screen_array);
		glBindVertexArray(half_screen_array);
		glBindBuffer(GL_ARRAY_BUFFER, half_screen_quad);
		glVertexAttribPointer(VS::ARRAY_VERTEX, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 4, 0);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(VS::ARRAY_TEX_UV, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 4, ((uint8_t *)NULL) + 8);
		glEnableVertexAttribArray(4);
		glBindVertexArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, 0); //unbind
	}
};

MobileVRInterface::~MobileVRInterface() {
	// and make sure we cleanup if we haven't already
	if (is_initialized()) {
		uninitialize();
	};
};
