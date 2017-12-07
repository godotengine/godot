/*************************************************************************/
/*  openhmd_interface.h                                                  */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2016 Juan Linietsky, Ariel Manzur.                 */
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

#ifndef OPENHMD_INTERFACE_H
#define OPENHMD_INTERFACE_H

#include "main/input_default.h"
#include "modules/openhmd/openhmd_shader.h"
#include "object.h"
#include "os/main_loop.h"
#include "os/thread_safe.h"
#include "servers/arvr/arvr_interface.h"
#include "servers/arvr/arvr_positional_tracker.h"

#include <openhmd.h>

class OpenHMDInterface : public ARVRInterface {

	GDCLASS(OpenHMDInterface, ARVRInterface);

private:
	struct ohmd_controller_tracker {
		ohmd_device *controller_device;
		ARVRPositionalTracker *controller_tracker;
	};

	bool do_auto_init_device_zero;
	int num_devices;
	int width, height;

	ohmd_context *ohmd_ctx; /* OpenHMD context we're using */
	ohmd_device_settings *ohmd_settings; /* Settings we're using */
	ohmd_device *hmd_device; /* HMD device we're rendering to */
	ohmd_device *tracking_device; /* if not NULL, alternative device we're using to track the position and orientation of our HMD */

	Vector<ohmd_controller_tracker> controller_tracker_mapping; /* for every controller device we need a positional tracker */

	OpenHMDShader *ohmd_shader;

	int add_controller_device(ohmd_device *p_device);
	void remove_controller_device(ohmd_device *p_device);
	Transform get_ohmd_matrix_as_transform(ohmd_device *p_device, ohmd_float_value p_type, float p_position_scale = 1.0);
	CameraMatrix get_ohmd_matrix_as_camera_matrix(ohmd_float_value p_type);
	Transform get_ohmd_rot_pos_as_transform(ohmd_device *p_device, float p_position_scale = 1.0);

protected:
	static void _bind_methods();

	int get_device_count() const;
	Array get_device_names() const;

public:
	virtual StringName get_name() const;
	virtual int get_capabilities() const;

	void scan_for_devices();

	bool auto_init_device_zero() const;
	void set_auto_init_device_zero(bool p_auto_init);

	bool init_hmd_device(int p_device);
	void close_hmd_device();

	bool init_tracking_device(int p_device);
	void close_tracking_device();

	bool init_controller_device(int p_device);

	virtual bool is_initialized();
	virtual bool initialize();
	virtual void uninitialize();

	virtual Size2 get_render_targetsize();
	virtual bool is_stereo();
	virtual Transform get_transform_for_eye(ARVRInterface::Eyes p_eye, const Transform &p_cam_transform);
	virtual CameraMatrix get_projection_for_eye(ARVRInterface::Eyes p_eye, real_t p_aspect, real_t p_z_near, real_t p_z_far);
	virtual void commit_for_eye(ARVRInterface::Eyes p_eye, RID p_render_target, const Rect2 &p_screen_rect);

	virtual void process();

	OpenHMDInterface();
	~OpenHMDInterface();
};

#endif // OPENHMD_INTERFACE_H