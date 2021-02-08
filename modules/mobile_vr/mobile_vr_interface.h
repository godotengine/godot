/*************************************************************************/
/*  mobile_vr_interface.h                                                */
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

#ifndef MOBILE_VR_INTERFACE_H
#define MOBILE_VR_INTERFACE_H

#include "servers/xr/xr_interface.h"
#include "servers/xr/xr_positional_tracker.h"

/**
	@author Bastiaan Olij <mux213@gmail.com>

	The mobile interface is a native VR interface that can be used on Android and iOS phones.
	It contains a basic implementation supporting 3DOF tracking if a gyroscope and accelerometer are
	present and sets up the proper projection matrices based on the values provided.

	We're planning to eventually do separate interfaces towards mobile SDKs that have far more capabilities and
	do not rely on the user providing most of these settings (though enhancing this with auto detection features
	based on the device we're running on would be cool). I'm mostly adding this as an example or base plate for
	more advanced interfaces.
*/

class MobileVRInterface : public XRInterface {
	GDCLASS(MobileVRInterface, XRInterface);

private:
	bool initialized = false;
	Basis orientation;

	// Just set some defaults for these. At some point we need to look at adding a lookup table for common device + headset combos and/or support reading cardboard QR codes
	float eye_height = 1.85;
	uint64_t last_ticks = 0;

	real_t intraocular_dist = 6.0;
	real_t display_width = 14.5;
	real_t display_to_lens = 4.0;
	real_t oversample = 1.5;

	//@TODO not yet used, these are needed in our distortion shader...
	real_t k1 = 0.215;
	real_t k2 = 0.215;

	/*
		logic for processing our sensor data, this was originally in our positional tracker logic but I think
		that doesn't make sense in hindsight. It only makes marginally more sense to park it here for now,
		this probably deserves an object of its own
	*/
	Vector3 scale_magneto(const Vector3 &p_magnetometer);
	Basis combine_acc_mag(const Vector3 &p_grav, const Vector3 &p_magneto);

	int mag_count = 0;
	bool has_gyro = false;
	bool sensor_first = false;
	Vector3 last_accerometer_data;
	Vector3 last_magnetometer_data;
	Vector3 mag_current_min;
	Vector3 mag_current_max;
	Vector3 mag_next_min;
	Vector3 mag_next_max;

	///@TODO a few support functions for trackers, most are math related and should likely be moved elsewhere
	float floor_decimals(float p_value, float p_decimals) {
		float power_of_10 = pow(10.0f, p_decimals);
		return floor(p_value * power_of_10) / power_of_10;
	};

	Vector3 floor_decimals(const Vector3 &p_vector, float p_decimals) {
		return Vector3(floor_decimals(p_vector.x, p_decimals), floor_decimals(p_vector.y, p_decimals), floor_decimals(p_vector.z, p_decimals));
	};

	Vector3 low_pass(const Vector3 &p_vector, const Vector3 &p_last_vector, float p_factor) {
		return p_vector + (p_factor * (p_last_vector - p_vector));
	};

	Vector3 scrub(const Vector3 &p_vector, const Vector3 &p_last_vector, float p_decimals, float p_factor) {
		return low_pass(floor_decimals(p_vector, p_decimals), p_last_vector, p_factor);
	};

	void set_position_from_sensors();

protected:
	static void _bind_methods();

public:
	void set_eye_height(const real_t p_eye_height);
	real_t get_eye_height() const;

	void set_iod(const real_t p_iod);
	real_t get_iod() const;

	void set_display_width(const real_t p_display_width);
	real_t get_display_width() const;

	void set_display_to_lens(const real_t p_display_to_lens);
	real_t get_display_to_lens() const;

	void set_oversample(const real_t p_oversample);
	real_t get_oversample() const;

	void set_k1(const real_t p_k1);
	real_t get_k1() const;

	void set_k2(const real_t p_k2);
	real_t get_k2() const;

	virtual StringName get_name() const override;
	virtual int get_capabilities() const override;

	virtual bool is_initialized() const override;
	virtual bool initialize() override;
	virtual void uninitialize() override;

	virtual Size2 get_render_targetsize() override;
	virtual bool is_stereo() override;
	virtual Transform get_transform_for_eye(XRInterface::Eyes p_eye, const Transform &p_cam_transform) override;
	virtual CameraMatrix get_projection_for_eye(XRInterface::Eyes p_eye, real_t p_aspect, real_t p_z_near, real_t p_z_far) override;
	virtual void commit_for_eye(XRInterface::Eyes p_eye, RID p_render_target, const Rect2 &p_screen_rect) override;

	virtual void process() override;
	virtual void notification(int p_what) override {}

	MobileVRInterface();
	~MobileVRInterface();
};

#endif // !MOBILE_VR_INTERFACE_H
