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
#include "globals.h"
#include "core/core_string_names.h"
#include "scene/scene_string_names.h"
#include "os/os.h"

OpenVR *OpenVR::singleton = NULL;

OpenVR *OpenVR::get_singleton() {
	return singleton;
};

void OpenVR::_bind_methods() {
	ObjectTypeDB::bind_method(_MD("is_installed"), &OpenVR::is_installed);
	ObjectTypeDB::bind_method(_MD("hmd_is_present"), &OpenVR::hmd_is_present);
	ObjectTypeDB::bind_method(_MD("is_initialized"), &OpenVR::is_initialized);
	ObjectTypeDB::bind_method(_MD("initialize"), &OpenVR::initialize);
	ObjectTypeDB::bind_method(_MD("uninitialize"), &OpenVR::uninitialize);

	ObjectTypeDB::bind_method(_MD("get_recommended_render_targetsize"), &OpenVR::get_recommended_render_targetsize);
	ObjectTypeDB::bind_method(_MD("get_transform_for_eye", "eye", "head_position"), &OpenVR::get_transform_for_eye);
	ObjectTypeDB::bind_method(_MD("get_frustum_for_eye", "eye"), &OpenVR::get_frustum_for_eye);

	ObjectTypeDB::bind_method(_MD("commit_eye_texture", "eye", "node:viewport"), &OpenVR::commit_eye_texture);

	///@TODO once this is automatic, we can remove this
	ObjectTypeDB::bind_method(_MD("process"), &OpenVR::process);

	BIND_CONSTANT(EYE_LEFT);
	BIND_CONSTANT(EYE_RIGHT);
};

void OpenVR::attach_device(uint32_t p_device_index) {
	if (trackers[p_device_index] == -1) {
		char device_name[256];
		strcpy(device_name, get_tracked_device_name(p_device_index, 255));
		printf("Device %u attached (%s)", p_device_index, device_name);

		// add tracker for our device
		if (p_device_index == vr::k_unTrackedDeviceIndex_Hmd) {
			trackers[p_device_index] = input->add_tracker(Input::TRACKER_HMD, device_name, true, true);
		} else if (strstr(device_name, "basestation") != NULL) {
			sprintf(&device_name[strlen(device_name)],"_%i", p_device_index);
			trackers[p_device_index] = input->add_tracker(Input::TRACKER_BASESTATION, device_name, true, true);
		} else {
			sprintf(&device_name[strlen(device_name)],"_%i", p_device_index);
			trackers[p_device_index] = input->add_tracker(Input::TRACKER_CONTROLLER, device_name, true, true);

			if (joyids[p_device_index] == -1) {
				// also register as joystick...
				joyids[p_device_index] = input->get_unused_joy_id();					
				if (joyids[p_device_index] != -1) {
					///@TODO what should we give as a GUID here? Also do we need to tell it about the buttons and axis?
					input->joy_connection_changed(joyids[p_device_index], true, device_name, "");
				};
			};
		};
	};
};

void OpenVR::detach_device(uint32_t p_device_index) {
	if (trackers[p_device_index] != -1) {
		input->remove_tracker(trackers[p_device_index]);
		trackers[p_device_index] = -1;

		// Also remove joystick if applicable...
		if (joyids[p_device_index] != -1) {
			input->joy_connection_changed(joyids[p_device_index], false, "", "");
			joyids[p_device_index] = -1;
		};
	};
};

bool OpenVR::is_installed() {
	return vr::VR_IsHmdPresent();
};

bool OpenVR::hmd_is_present() {
	return vr::VR_IsRuntimeInstalled();
};

bool OpenVR::is_initialized() {
	return hmd != NULL;
};

bool OpenVR::initialize() {
	if (hmd == NULL) {
		bool success = true;

		// reset some stuff
		for (uint32_t i = 0; i < vr::k_unMaxTrackedDeviceCount; i++) {
			trackers[i] = -1;
			joyids[i] = -1;
		};

		// Loading the SteamVR Runtime
		vr::EVRInitError error = vr::VRInitError_None;
		hmd = vr::VR_Init( &error, vr::VRApplication_Scene );

		if ( error != vr::VRInitError_None ) {
			success = false;
			printf("Unable to init VR runtime: %s\n", vr::VR_GetVRInitErrorAsEnglishDescription(error));
		} else {
			printf("Main OpenVR interface has been initialized\n");
		};

		if (success) {
			// render models give us access to mesh representations of the various controllers
			render_models = (vr::IVRRenderModels *)vr::VR_GetGenericInterface(vr::IVRRenderModels_Version, &error);
			if( !render_models ) {
				success = false;

				printf("Unable to get render model interface: %s\n", vr::VR_GetVRInitErrorAsEnglishDescription(error));
			} else {
				printf("Main render models interface has been initialized\n");
			};
		};

		if ( !vr::VRCompositor() ) {
			success = false;

			printf( "Compositor initialization failed. See log file for details\n" );
		};

		if (success) {
			for (uint32_t i = vr::k_unTrackedDeviceIndex_Hmd; i < vr::k_unMaxTrackedDeviceCount; i++ ) {
				if( hmd->IsTrackedDeviceConnected(i) ) {
					attach_device(i);
				};
			};
		};

		if (!success) {
			vr::VR_Shutdown();
			hmd = NULL;
			render_models = NULL;		
		};
	};

	return hmd != NULL;
};

void OpenVR::uninitialize() {
	if (hmd != NULL) {
		for (uint32_t i = 0; i < vr::k_unMaxTrackedDeviceCount; i++) {
			if (trackers[i] != -1) {
				input->remove_tracker(trackers[i]);
				trackers[i] = -1;
			};
			if (joyids[i] != -1) {
				input->joy_connection_changed(joyids[i], false, "", "");
				joyids[i] = -1;
			};
		};

		vr::VR_Shutdown();
		hmd = NULL;
		render_models = NULL;
	};
};

Point2 OpenVR::get_recommended_render_targetsize() {
	if (hmd != NULL) {
		uint32_t width, height;

		hmd->GetRecommendedRenderTargetSize(&width, &height);

		return Point2(width, height);
	} else {
		return Point2(512,512);
	};
};

Transform OpenVR::get_transform_for_eye(OpenVR::Eyes p_eye, const Transform& p_head_position) {
	Transform newtransform;

	if (hmd != NULL) {
		vr::HmdMatrix34_t matrix = hmd->GetEyeToHeadTransform(p_eye == OpenVR::EYE_LEFT ? vr::Eye_Left : vr::Eye_Right);

		newtransform.basis.set(
			matrix.m[0][0], matrix.m[0][1], matrix.m[0][2],
			matrix.m[1][0], matrix.m[1][1], matrix.m[1][2],
			matrix.m[2][0], matrix.m[2][1], matrix.m[2][2]
		);

		newtransform.origin.x = matrix.m[0][3];
		newtransform.origin.y = matrix.m[1][3];
		newtransform.origin.z = matrix.m[2][3];
	} else {
		if (p_eye == OpenVR::EYE_LEFT) {
			newtransform.origin.x = -0.035;
		} else {
			newtransform.origin.x = 0.035;
		};
	};

	newtransform = p_head_position * newtransform;

	return newtransform;
};

Rect2 OpenVR::get_frustum_for_eye(OpenVR::Eyes p_eye) {
	Frustum frustum;
	if (hmd != NULL) {
		Point2 size = get_recommended_render_targetsize();

		// note that openvr is upside down in relation to godot
		hmd->GetProjectionRaw(p_eye == OpenVR::EYE_LEFT ? vr::Eye_Left : vr::Eye_Right, &frustum.left, &frustum.right, &frustum.bottom, &frustum.top);
	
		// Godot will (re)apply our aspect ratio so we need to unapply it
		frustum.left = frustum.left * size.y / size.x;
		frustum.right = frustum.right * size.y / size.x;
	} else {
		// just return a pretty basic stereoscopic frustum
		frustum.set_frustum(60.0, p_eye == OpenVR::EYE_LEFT ? Frustum::EYE_LEFT : Frustum::EYE_RIGHT, 0.065, 1.0);
	};

///@TODO Return Frustum instead of Rect2, need to add Frustum support to Variants
//	return frustum;
	return Rect2(frustum.left, frustum.top, frustum.right, frustum.bottom);
};

void OpenVR::commit_eye_texture(OpenVR::Eyes p_eye, Node* p_viewport) {
	///@TODO this needs to be greatly improved, I'm sure it can be sped up...

	if (hmd != NULL) {
		uint32_t texid;
		Ref<Texture> eye_texture;

		Viewport *vp=p_viewport->cast_to<Viewport>();
		ERR_FAIL_COND(!vp);

		Ref<RenderTargetTexture> rtt = vp->get_render_target_texture();
		eye_texture=rtt;

		// added texture_get_texid to visual server (and rasterers), possibly temporary solution as gles3 may change access to the opengl texture id
		if (eye_texture.is_valid()) {
			// upside down!
			vr::VRTextureBounds_t bounds;
			bounds.uMin = 0.0;
			bounds.uMax = 1.0;
			bounds.vMin = 1.0;
			bounds.vMax = 0.0;

			texid = VS::get_singleton()->texture_get_texid(eye_texture->get_rid());

			vr::Texture_t eyeTexture = {(void*)texid, vr::TextureType_OpenGL, vr::ColorSpace_Gamma};
			vr::VRCompositor()->Submit(p_eye == OpenVR::EYE_LEFT ? vr::Eye_Left : vr::Eye_Right, &eyeTexture, &bounds);
		};
	};
};

///@TODO need to find a way to call this automatically while OpenVR is initialized...
void OpenVR::process() {
	if (hmd != NULL) {
		// Process SteamVR events
		vr::VREvent_t event;
		while( hmd->PollNextEvent(&event, sizeof(event))) {
			switch( event.eventType ) {
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
		vr::VRCompositor()->WaitGetPoses(tracked_device_pose, vr::k_unMaxTrackedDeviceCount, NULL, 0 );

		// update trackers and joysticks
		for (uint32_t i = 0; i < vr::k_unMaxTrackedDeviceCount; i++) {
			// update tracker
			Transform newtransform;
			if ((trackers[i] != -1) && (tracked_device_pose[i].bPoseIsValid)) {
				// bit wasteful copying it but I don't want to type so much!
				vr::HmdMatrix34_t matPose = tracked_device_pose[i].mDeviceToAbsoluteTracking;

				newtransform.basis.set(
					matPose.m[0][0], matPose.m[0][1], matPose.m[0][2],
					matPose.m[1][0], matPose.m[1][1], matPose.m[1][2],
					matPose.m[2][0], matPose.m[2][1], matPose.m[2][2]
					);

				newtransform.origin.x = matPose.m[0][3];
				newtransform.origin.y = matPose.m[1][3];
				newtransform.origin.z = matPose.m[2][3];

				input->set_tracker_transform(trackers[i], newtransform);
			};

			if (joyids[i] != -1) {
				uint32_t last_id = 0; ///@TODO initialise this property
				// update our button state structure
				vr::VRControllerState_t new_state;
				hmd->GetControllerState(i, &new_state, sizeof(vr::VRControllerState_t));
				if (tracked_device_state[i].unPacketNum != new_state.unPacketNum) {
					// we currently have 8 defined buttons on VIVE controllers.
					for (int button = 0; button < 8; button++) {
						last_id = input->joy_button(last_id, joyids[i], button, new_state.ulButtonPressed & vr::ButtonMaskFromId((vr::EVRButtonId) button));
					};
					
					// support 3 axis for now, this may need to be enhanced
					InputDefault::JoyAxis jx;
					jx.min = -1;
					jx.value = new_state.rAxis[vr::k_EButton_SteamVR_Touchpad].x;
					last_id = input->joy_axis(last_id, joyids[i], JOY_AXIS_0, jx);
					jx.value = new_state.rAxis[vr::k_EButton_SteamVR_Touchpad].y;
					last_id = input->joy_axis(last_id, joyids[i], JOY_AXIS_1, jx);
					jx.min = 0;
					jx.value = new_state.rAxis[vr::k_EButton_SteamVR_Touchpad].x;
					last_id = input->joy_axis(last_id, joyids[i], JOY_AXIS_0, jx);

					tracked_device_state[i] = new_state;
				};				
			};
		};
	};
};

const char * OpenVR::get_tracked_device_name(vr::TrackedDeviceIndex_t p_tracked_device_index, int pMaxLen) const {
	static char returnstring[1025] = "Not initialised";

	// don't go bigger then this...
	if (pMaxLen > 1024) {
		pMaxLen = 1024;
	};

	if ((hmd != NULL) && (p_tracked_device_index != vr::k_unTrackedDeviceIndexInvalid)) {
		uint32_t namelength = hmd->GetStringTrackedDeviceProperty(p_tracked_device_index, vr::Prop_RenderModelName_String, NULL, 0, NULL);
		if (namelength > 0 ) {
			if (namelength > pMaxLen) {
				namelength = pMaxLen;
			};
			
			hmd->GetStringTrackedDeviceProperty(p_tracked_device_index, vr::Prop_RenderModelName_String, returnstring, namelength, NULL);
		};
 	};

	return returnstring;
};

OpenVR::OpenVR() {
	singleton = this;
	input = (InputDefault *) Input::get_singleton();
	hmd = NULL;
	render_models = NULL;
};

OpenVR::~OpenVR() {
	uninitialize();
	singleton = NULL;
};
