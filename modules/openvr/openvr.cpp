/*************************************************************************/
/*  openvr.cpp                                                           */
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

#include "openvr.h"
#include "core/core_string_names.h"
#include "core/os/os.h"
#include "os/os.h"
#include "project_settings.h"
#include "scene/scene_string_names.h"
#include "servers/visual/visual_server_global.h"

StringName OpenVR::get_name() const {
	return "OpenVR";
};

int OpenVR::get_capabilities() const {
	return ARVR_STEREO + ARVR_EXTERNAL;
};

void OpenVR::attach_device(uint32_t p_device_index) {
	ARVRServer *arvr_server = ARVRServer::get_singleton();
	ERR_FAIL_NULL(arvr_server);

	if (trackers[p_device_index] == NULL) {
		ARVRPositionalTracker *new_tracker = NULL;

		char device_name[256];
		strcpy(device_name, get_tracked_device_name(p_device_index, 255));
		print_line("Device " + itos(p_device_index) + " attached (" + device_name + ")");

		// add tracker for our device
		if (p_device_index == vr::k_unTrackedDeviceIndex_Hmd) {
			// we no longer track our HMD, this is all handled in ARVROrigin :)
		} else if (strstr(device_name, "basestation") != NULL) {
			new_tracker = memnew(ARVRPositionalTracker);

			sprintf(&device_name[strlen(device_name)], "_%i", p_device_index);
			new_tracker->set_name(device_name);
			new_tracker->set_type(ARVRServer::TRACKER_BASESTATION);
		} else if (strstr(device_name, "camera") != NULL) {
			new_tracker = memnew(ARVRPositionalTracker);

			sprintf(&device_name[strlen(device_name)], "_%i", p_device_index);
			new_tracker->set_name(device_name);
			new_tracker->set_type(ARVRServer::TRACKER_BASESTATION);
		} else {
			new_tracker = memnew(ARVRPositionalTracker);

			sprintf(&device_name[strlen(device_name)], "_%i", p_device_index);
			new_tracker->set_name(device_name);
			new_tracker->set_type(ARVRServer::TRACKER_CONTROLLER);

			// get our controller role
			vr::ETrackedPropertyError error;
			int32_t controllerRole = hmd->GetInt32TrackedDeviceProperty(p_device_index, vr::Prop_ControllerRoleHint_Int32, &error);
			if (controllerRole == vr::TrackedControllerRole_RightHand) {
				new_tracker->set_hand(ARVRPositionalTracker::TRACKER_RIGHT_HAND);
			} else if (controllerRole == vr::TrackedControllerRole_LeftHand) {
				new_tracker->set_hand(ARVRPositionalTracker::TRACKER_LEFT_HAND);
			}

			// also register as joystick...
			int joyid = input->get_unused_joy_id();
			if (joyid != -1) {
				new_tracker->set_joy_id(joyid);
				input->joy_connection_changed(joyid, true, device_name, "");
			};
		};

		if (new_tracker != NULL) {
			// init these to set our flags
			Basis orientation;
			new_tracker->set_orientation(orientation);
			Vector3 position;
			new_tracker->set_position(position);

			// add our tracker to our server and remember its pointer
			arvr_server->add_tracker(new_tracker);
		}
		trackers[p_device_index] = new_tracker;
	};
};

void OpenVR::detach_device(uint32_t p_device_index) {
	ARVRServer *arvr_server = ARVRServer::get_singleton();
	if (trackers[p_device_index] != NULL) {
		// unset our joystick if applicable
		int joyid = trackers[p_device_index]->get_joy_id();
		if (joyid != -1) {
			input->joy_connection_changed(joyid, false, "", "");
			trackers[p_device_index]->set_joy_id(-1);
		};

		// remove our tracker from our server
		print_line("Device " + itos(p_device_index) + " removed (" + trackers[p_device_index]->get_name() + ")");
		if (arvr_server != NULL) {
			arvr_server->remove_tracker(trackers[p_device_index]);
		};
		memdelete(trackers[p_device_index]);
		trackers[p_device_index] = NULL;
	};
};

bool OpenVR::is_initialized() {
	return hmd != NULL;
};

bool OpenVR::initialize() {
	_THREAD_SAFE_METHOD_

	ARVRServer *arvr_server = ARVRServer::get_singleton();
	ERR_FAIL_NULL_V(arvr_server, false);

	if (hmd == NULL) {
		bool success = true;
		vr::EVRInitError error = vr::VRInitError_None;

		// reset some stuff
		for (uint32_t i = 0; i < vr::k_unMaxTrackedDeviceCount; i++) {
			trackers[i] = NULL;
		};

		if (!vr::VR_IsRuntimeInstalled()) {
			print_line("SteamVR has not been installed.");
			success = false;
		};

		if (success) {
			// Loading the SteamVR Runtime
			hmd = vr::VR_Init(&error, vr::VRApplication_Scene);

			if (error != vr::VRInitError_None) {
				success = false;
				print_line("Unable to init VR runtime: " + String(vr::VR_GetVRInitErrorAsEnglishDescription(error)));
			} else {
				print_line("Main OpenVR interface has been initialized");
			};
		};

		if (success) {
			// render models give us access to mesh representations of the various controllers
			render_models = (vr::IVRRenderModels *)vr::VR_GetGenericInterface(vr::IVRRenderModels_Version, &error);
			if (!render_models) {
				success = false;

				print_line("Unable to get render model interface: " + String(vr::VR_GetVRInitErrorAsEnglishDescription(error)));
			} else {
				print_line("Main render models interface has been initialized");
			};
		};

		if (!vr::VRCompositor()) {
			success = false;

			print_line("Compositor initialization failed. See log file for details");
		};

		if (success) {
			// find any already attached devices
			for (uint32_t i = vr::k_unTrackedDeviceIndex_Hmd; i < vr::k_unMaxTrackedDeviceCount; i++) {
				if (hmd->IsTrackedDeviceConnected(i)) {
					attach_device(i);
				};
			};
		};

		if (success) {
			// make this our primary interface as we'll be rendering to this
			arvr_server->set_primary_interface(this);
		} else {
			uninitialize();
		};
	};

	return hmd != NULL;
};

void OpenVR::uninitialize() {
	_THREAD_SAFE_METHOD_

	if (hmd != NULL) {
		// no longer our primary interface
		ARVRServer *arvr_server = ARVRServer::get_singleton();
		if (arvr_server != NULL) {
			arvr_server->clear_primary_interface_if(this);
		}

		// detach all our divices
		for (uint32_t i = 0; i < vr::k_unMaxTrackedDeviceCount; i++) {
			detach_device(i);
		};

		vr::VR_Shutdown();
		hmd = NULL;
		render_models = NULL;
	};
};

Point2 OpenVR::get_recommended_render_targetsize() {
	_THREAD_SAFE_METHOD_

	if (hmd != NULL) {
		uint32_t width, height;

		hmd->GetRecommendedRenderTargetSize(&width, &height);

		return Point2(width, height);
	} else {
		return Point2(512, 512);
	};
};

bool OpenVR::is_stereo() {
	return true;
}

Transform OpenVR::get_transform_for_eye(ARVRInterface::Eyes p_eye, const Transform &p_head_position) {
	_THREAD_SAFE_METHOD_

	Transform newtransform;

	ARVRServer *arvr_server = ARVRServer::get_singleton();
	ERR_FAIL_NULL_V(arvr_server, newtransform);

	real_t world_scale = arvr_server->get_world_scale();

	if (hmd != NULL) {
		vr::HmdMatrix34_t matrix = hmd->GetEyeToHeadTransform(p_eye == ARVRInterface::EYE_LEFT ? vr::Eye_Left : vr::Eye_Right);

		newtransform.basis.set(
				matrix.m[0][0], matrix.m[0][1], matrix.m[0][2],
				matrix.m[1][0], matrix.m[1][1], matrix.m[1][2],
				matrix.m[2][0], matrix.m[2][1], matrix.m[2][2]);

		newtransform.origin.x = matrix.m[0][3] * world_scale;
		newtransform.origin.y = matrix.m[1][3] * world_scale;
		newtransform.origin.z = matrix.m[2][3] * world_scale;
	} else {
		if (p_eye == ARVRInterface::EYE_LEFT) {
			newtransform.origin.x = -0.035 * world_scale;
		} else {
			newtransform.origin.x = 0.035 * world_scale;
		};
	};

	newtransform = p_head_position * (arvr_server->get_reference_frame()) * hmd_transform * newtransform;

	return newtransform;
};

CameraMatrix OpenVR::get_projection_for_eye(ARVRInterface::Eyes p_eye, real_t p_aspect, real_t p_z_near, real_t p_z_far) {
	_THREAD_SAFE_METHOD_

	CameraMatrix camera;
	if (hmd != NULL) {
		vr::HmdMatrix44_t matrix = hmd->GetProjectionMatrix(p_eye == ARVRInterface::EYE_LEFT ? vr::Eye_Left : vr::Eye_Right, p_z_near, p_z_far);

		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				camera.matrix[i][j] = matrix.m[j][i];
			};
		};
	} else {
		// just return a pretty basic stereoscopic frustum
		camera.set_perspective(60.0, p_aspect, p_z_near, p_z_far, false, p_eye == ARVRInterface::EYE_LEFT ? 1 : 2, 0.065, 1.0);
	};

	return camera;
};

void OpenVR::commit_for_eye(ARVRInterface::Eyes p_eye, RID p_render_target, const Rect2 &p_screen_rect) {
	_THREAD_SAFE_METHOD_

	// make sure our render target is unset
	VSG::rasterizer->set_current_render_target(RID());

	// if we're commiting our left eye to our main viewport, also blit it out, no lens distortion needed here, this is for spectating..
	// if you do not want this, use a separate viewport as the AR viewport and do whatever you wish on the main viewport.
	if (p_eye == ARVRInterface::EYE_LEFT && p_screen_rect != Rect2()) {
		VSG::rasterizer->blit_render_target_to_screen(p_render_target, p_screen_rect, 0);
	}

	if (hmd != NULL) {
		vr::VRTextureBounds_t bounds;
		bounds.uMin = 0.0;
		bounds.uMax = 1.0;
		bounds.vMin = 0.0;
		bounds.vMax = 1.0;

		RID eye_texture = VSG::storage->render_target_get_texture(p_render_target);
		uint32_t texid = VS::get_singleton()->texture_get_texid(eye_texture);

		vr::Texture_t eyeTexture = { (void *)(uintptr_t)texid, vr::TextureType_OpenGL, vr::ColorSpace_Gamma };
		vr::EVRCompositorError vrerr = vr::VRCompositor()->Submit(p_eye == ARVRInterface::EYE_LEFT ? vr::Eye_Left : vr::Eye_Right, &eyeTexture, &bounds);
		if (vrerr != vr::VRCompositorError_None) {
			print_line("OpenVR reports: " + itos(vrerr));
		}
	};
};

void OpenVR::process() {
	_THREAD_SAFE_METHOD_

	ARVRServer *arvr_server = ARVRServer::get_singleton();
	ERR_FAIL_NULL(arvr_server);

	if (hmd != NULL) {
		// Process SteamVR events
		vr::VREvent_t event;
		while (hmd->PollNextEvent(&event, sizeof(event))) {
			switch (event.eventType) {
				case vr::VREvent_TrackedDeviceActivated: {
					attach_device(event.trackedDeviceIndex);
				}; break;
				case vr::VREvent_TrackedDeviceDeactivated: {
					detach_device(event.trackedDeviceIndex);
				}; break;
				default: {
					// ignored for now...
				}; break;
			};
		};

		///@TODO we should time how long it takes between calling WaitGetPoses and committing the output to the HMD and using that as the 4th parameter...

		// update our poses structure, this tracks our controllers
		vr::VRCompositor()->WaitGetPoses(tracked_device_pose, vr::k_unMaxTrackedDeviceCount, NULL, 0);

		// we scale all our positions by our world scale
		real_t world_scale = arvr_server->get_world_scale();

		// update trackers and joysticks
		for (uint32_t i = 0; i < vr::k_unMaxTrackedDeviceCount; i++) {
			// update tracker
			if (tracked_device_pose[i].bPoseIsValid) {
				// bit wasteful copying it but I don't want to type so much!
				vr::HmdMatrix34_t matPose = tracked_device_pose[i].mDeviceToAbsoluteTracking;

				Basis orientation;
				orientation.set(
						matPose.m[0][0], matPose.m[0][1], matPose.m[0][2],
						matPose.m[1][0], matPose.m[1][1], matPose.m[1][2],
						matPose.m[2][0], matPose.m[2][1], matPose.m[2][2]);

				Vector3 position;
				position.x = matPose.m[0][3];
				position.y = matPose.m[1][3];
				position.z = matPose.m[2][3];

				if (i == 0) {
					// store our HMD transform
					hmd_transform.basis = orientation;
					hmd_transform.origin = position * world_scale; // should move applying worldscale into get_transform_for_eye
				} else if (trackers[i] != NULL) {
					trackers[i]->set_orientation(orientation);
					trackers[i]->set_rw_position(position);

					int joyid = trackers[i]->get_joy_id();
					if (joyid != -1) {
						// update our button state structure
						vr::VRControllerState_t new_state;
						hmd->GetControllerState(i, &new_state, sizeof(vr::VRControllerState_t));
						if (tracked_device_state[i].unPacketNum != new_state.unPacketNum) {
							// we currently have 8 defined buttons on VIVE controllers.
							for (int button = 0; button < 8; button++) {
								input->joy_button(joyid, button, new_state.ulButtonPressed & vr::ButtonMaskFromId((vr::EVRButtonId)button));
							};

							// support 3 axis for now, this may need to be enhanced
							InputDefault::JoyAxis jx;
							jx.min = -1;
							jx.value = new_state.rAxis[vr::k_EButton_SteamVR_Touchpad].x;
							input->joy_axis(joyid, JOY_AXIS_0, jx);
							jx.value = new_state.rAxis[vr::k_EButton_SteamVR_Touchpad].y;
							input->joy_axis(joyid, JOY_AXIS_1, jx);
							jx.min = 0;
							jx.value = new_state.rAxis[vr::k_EButton_SteamVR_Touchpad].x;
							input->joy_axis(joyid, JOY_AXIS_0, jx);

							tracked_device_state[i] = new_state;
						};
					};
				};
			};
		};
	};
};

const char *OpenVR::get_tracked_device_name(vr::TrackedDeviceIndex_t p_tracked_device_index, int pMaxLen) const {
	static char returnstring[1025] = "Not initialised";

	// don't go bigger then this...
	if (pMaxLen > 1024) {
		pMaxLen = 1024;
	};

	if ((hmd != NULL) && (p_tracked_device_index != vr::k_unTrackedDeviceIndexInvalid)) {
		uint32_t namelength = hmd->GetStringTrackedDeviceProperty(p_tracked_device_index, vr::Prop_RenderModelName_String, NULL, 0, NULL);
		if (namelength > 0) {
			if (namelength > pMaxLen) {
				namelength = pMaxLen;
			};

			hmd->GetStringTrackedDeviceProperty(p_tracked_device_index, vr::Prop_RenderModelName_String, returnstring, namelength, NULL);
		};
	};

	return returnstring;
};

OpenVR::OpenVR() {
	hmd = NULL;
	render_models = NULL;
	input = (InputDefault *)Input::get_singleton();
};

OpenVR::~OpenVR() {
	if (hmd != NULL) {
		uninitialize();
	};
};
