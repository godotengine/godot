/*************************************************************************/
/*  openvr.h                                                             */
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

#ifndef OPENVR_H
#define OPENVR_H

#include "object.h"
#include "os/main_loop.h"
#include "os/thread_safe.h"
#include "main/input_default.h"
#include "scene/main/viewport.h"
#include "core/math/frustum.h"

#include <openvr.h>

class OpenVR : public Object {

	OBJ_TYPE(OpenVR, Object);

	static OpenVR * singleton;

private:
	vr::IVRSystem * hmd;
	vr::IVRRenderModels * render_models;
	vr::TrackedDevicePose_t tracked_device_pose[vr::k_unMaxTrackedDeviceCount];
	vr::VRControllerState_t tracked_device_state[vr::k_unMaxTrackedDeviceCount];

	int trackers[vr::k_unMaxTrackedDeviceCount];
	int joyids[vr::k_unMaxTrackedDeviceCount];

	void attach_device(uint32_t p_device_index);
	void detach_device(uint32_t p_device_index);
	const char * get_tracked_device_name(vr::TrackedDeviceIndex_t p_tracked_device_index, int pMaxLen) const;

protected:
	InputDefault * input;

	static void _bind_methods();

public:
	static OpenVR *get_singleton();

	enum Eyes {
		EYE_LEFT,
		EYE_RIGHT
	};

	bool is_installed();
	bool hmd_is_present();

	bool is_initialized();
	bool initialize();
	void uninitialize();

	Point2 get_recommended_render_targetsize();
	Transform get_transform_for_eye(OpenVR::Eyes p_eye, const Transform& p_head_position);
	Rect2 get_frustum_for_eye(OpenVR::Eyes p_eye);

	void commit_eye_texture(OpenVR::Eyes p_eye, Node* p_viewport);

	void process();

	OpenVR();
	~OpenVR();
};

VARIANT_ENUM_CAST(OpenVR::Eyes);


#endif // OPENVR_H
