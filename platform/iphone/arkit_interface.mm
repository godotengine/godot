/*************************************************************************/
/*  arkit_interface.cpp                                                  */
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

#ifdef ARKIT_ENABLED

#include "arkit_interface.h"
#include "camera_ios.h"
#include "core/os/input.h"
#include "core/os/os.h"
#include "servers/visual/visual_server_global.h"

#import <ARKit/ARKit.h>
#import <UIKit/UIKit.h>

// just a dirty workaround for now, declare these as globals. I'll probably encapsulate ARSession and associated logic into an mm object and change ARKitInterface to a normal cpp object that consumes it.
ARSession *ar_session;
NSTimeInterval last_timestamp;

/* this is called when we initialize or when we come back from having our app pushed to the background, just (re)start our session */
void ARKitInterface::start_session() {
	// We delay our session starting if we've not yet initialized.
	session_start = true;

	if (initialized) {
		print_line("Starting ar session");
		ARWorldTrackingConfiguration *configuration = [ARWorldTrackingConfiguration new];
		configuration.lightEstimationEnabled = light_estimation_is_enabled;
		configuration.planeDetection = plane_detection_is_enabled;

		[ar_session runWithConfiguration:configuration];
	};
};

void ARKitInterface::stop_session() {
	session_start = false;

	if (initialized) {
		[ar_session pause];
	};
};

bool ARKitInterface::get_anchor_detection_is_enabled() const {
	return plane_detection_is_enabled;
};

void ARKitInterface::set_anchor_detection_is_enabled(bool p_enable) {
	if (plane_detection_is_enabled != p_enable) {
		plane_detection_is_enabled = p_enable;

		// do we need to restart our session? If not plane detection won't change until later.
		if (initialized && session_start) {
			start_session();
		};
	};
};

bool ARKitInterface::get_light_estimation_is_enabled() const {
	return light_estimation_is_enabled;
};

void ARKitInterface::set_light_estimation_is_enabled(bool p_enable) {
	if (light_estimation_is_enabled != p_enable) {
		light_estimation_is_enabled = p_enable;

		// do we need to restart our session? If not plane detection won't change until later.
		if (initialized && session_start) {
			start_session();
		};
	};
};

real_t ARKitInterface::get_ambient_intensity() const {
	return ambient_intensity;
};

real_t ARKitInterface::get_ambient_color_temperature() const {
	return ambient_color_temperature;
};

StringName ARKitInterface::get_name() const {
	return "ARKit";
};

int ARKitInterface::get_capabilities() const {
	return ARKitInterface::ARVR_MONO + ARKitInterface::ARVR_AR;
};

Array ARKitInterface::raycast(Vector2 p_screen_coord) {
	Array arr;
	Size2 screen_size = OS::get_singleton()->get_window_size();
	CGPoint point;
	point.x = p_screen_coord.x / screen_size.x;
	point.y = p_screen_coord.y / screen_size.y;

	///@TODO maybe give more options here, for now we're taking just ARAchors into account that were found during plane detection keeping their size into account
	NSArray<ARHitTestResult *> *results = [ar_session.currentFrame hittest:point types:ARHitTestResultTypeExistingPlaneUsingExtent];

	for (ARHitTestResult *result in results) {
		Transform transform;

		matrix_float4x4 m44 = result.worldTransform;
		transform.basis.elements[0].x = m44.columns[0][0];
		transform.basis.elements[1].x = m44.columns[0][1];
		transform.basis.elements[2].x = m44.columns[0][2];
		transform.basis.elements[0].y = m44.columns[1][0];
		transform.basis.elements[1].y = m44.columns[1][1];
		transform.basis.elements[2].y = m44.columns[1][2];
		transform.basis.elements[0].z = m44.columns[2][0];
		transform.basis.elements[1].z = m44.columns[2][1];
		transform.basis.elements[2].z = m44.columns[2][2];
		transform.origin.x = m44.columns[3][0];
		transform.origin.y = m44.columns[3][1];
		transform.origin.z = m44.columns[3][2];

		/* important, NOT scaled to world_scale !! */
		arr.push_back(transform);
	};

	return arr;
};

void ARKitInterface::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_light_estimation_is_enabled", "enable"), &ARKitInterface::set_light_estimation_is_enabled);
	ClassDB::bind_method(D_METHOD("get_light_estimation_is_enabled"), &ARKitInterface::get_light_estimation_is_enabled);
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "light_estimation"), "set_light_estimation_is_enabled", "get_light_estimation_is_enabled");

	ClassDB::bind_method(D_METHOD("get_ambient_intensity"), &ARKitInterface::get_ambient_intensity);
	ClassDB::bind_method(D_METHOD("get_ambient_color_temperature"), &ARKitInterface::get_ambient_color_temperature);

	ClassDB::bind_method(D_METHOD("raycast", "screen_coord"), &ARKitInterface::raycast);
};

bool ARKitInterface::is_stereo() {
	// this is a mono device...
	return false;
};

bool ARKitInterface::is_initialized() {
	return (initialized);
};

bool ARKitInterface::initialize() {
	ARVRServer *arvr_server = ARVRServer::get_singleton();
	ERR_FAIL_NULL_V(arvr_server, false);

	if (!initialized) {
		print_line("initializing ARKit");

		// create our ar session
		ar_session = [ARSession new];

		// reset our transform
		transform = Transform();

		// make this our primary interface
		arvr_server->set_primary_interface(this);

		// yeah!
		initialized = true;

		// our start_session was called before we were initialized? then we start it now...
		if (session_start) {
			start_session();
		};
	};

	return true;
};

void ARKitInterface::uninitialize() {
	if (initialized) {
		ARVRServer *arvr_server = ARVRServer::get_singleton();
		if (arvr_server != NULL) {
			// no longer our primary interface
			arvr_server->clear_primary_interface_if(this);
		}

		remove_all_anchors();

		[ar_session release];
		ar_session = NULL;
		initialized = false;
		session_start = false;
	};
};

Size2 ARKitInterface::get_render_targetsize() {
	_THREAD_SAFE_METHOD_

	Size2 target_size = OS::get_singleton()->get_window_size();

	return target_size;
};

Transform ARKitInterface::get_transform_for_eye(ARVRInterface::Eyes p_eye, const Transform &p_cam_transform) {
	_THREAD_SAFE_METHOD_

	Transform transform_for_eye;

	ARVRServer *arvr_server = ARVRServer::get_singleton();
	ERR_FAIL_NULL_V(arvr_server, transform_for_eye);

	if (initialized) {
		float world_scale = arvr_server->get_world_scale();

		// just scale our origin point of our transform, note that we really shouldn't be using world_scale in ARKit but....
		transform_for_eye = transform;
		transform_for_eye.origin *= world_scale;

		transform_for_eye = p_cam_transform * (arvr_server->get_reference_frame()) * transform_for_eye;
	} else {
		// huh? well just return what we got....
		transform_for_eye = p_cam_transform;
	};

	return transform_for_eye;
};

CameraMatrix ARKitInterface::get_projection_for_eye(ARVRInterface::Eyes p_eye, real_t p_aspect, real_t p_z_near, real_t p_z_far) {
	// Remember our near and far, we'll use it next frame
	z_near = p_z_near;
	z_far = p_z_far;

	return projection;
};

void ARKitInterface::commit_for_eye(ARVRInterface::Eyes p_eye, RID p_render_target, const Rect2 &p_screen_rect) {
	_THREAD_SAFE_METHOD_

	// We must have a valid render target
	ERR_FAIL_COND(!p_render_target.is_valid());

	// Because we are rendering to our device we must use our main viewport!
	ERR_FAIL_COND(p_screen_rect == Rect2());

	// get the size of our screen
	Rect2 screen_rect = p_screen_rect;

	//		screen_rect.position.x += screen_rect.size.x;
	//		screen_rect.size.x = -screen_rect.size.x;
	//		screen_rect.position.y += screen_rect.size.y;
	//		screen_rect.size.y = -screen_rect.size.y;

	VSG::rasterizer->set_current_render_target(RID());
	VSG::rasterizer->blit_render_target_to_screen(p_render_target, screen_rect, 0);
};

ARVRPositionalTracker *ARKitInterface::get_anchor_for_uuid(const unsigned char *p_uuid) {
	if (anchors == NULL) {
		num_anchors = 0;
		max_anchors = 10;
		anchors = (anchor_map *)malloc(sizeof(anchor_map) * max_anchors);
	};

	ERR_FAIL_NULL_V(anchors, NULL);

	for (unsigned int i = 0; i < num_anchors; i++) {
		if (memcmp(anchors[i].uuid, p_uuid, 16) == 0) {
			return anchors[i].tracker;
		};
	};

	if (num_anchors + 1 == max_anchors) {
		max_anchors += 10;
		anchors = (anchor_map *)realloc(anchors, sizeof(anchor_map) * max_anchors);
		ERR_FAIL_NULL_V(anchors, NULL);
	};

	ARVRPositionalTracker *new_tracker = memnew(ARVRPositionalTracker);
	new_tracker->set_type(ARVRServer::TRACKER_ANCHOR);

	char tracker_name[256];
	sprintf(tracker_name, "Anchor %02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x", p_uuid[0], p_uuid[1], p_uuid[2], p_uuid[3], p_uuid[4], p_uuid[5], p_uuid[6], p_uuid[7], p_uuid[8], p_uuid[9], p_uuid[10], p_uuid[11], p_uuid[12], p_uuid[13], p_uuid[14], p_uuid[15]);

	String name = tracker_name;
	print_line("Adding tracker " + name);

	// add our tracker
	ARVRServer::get_singleton()->add_tracker(new_tracker);
	anchors[num_anchors].tracker = new_tracker;
	memcpy(anchors[num_anchors].uuid, p_uuid, 16);
	num_anchors++;

	return new_tracker;
};

void ARKitInterface::remove_anchor(const unsigned int p_idx) {
	if (anchors == NULL) {
		// ignore
	} else if (p_idx < num_anchors) {
		ARVRServer::get_singleton()->remove_tracker(anchors[p_idx].tracker);
		memdelete(anchors[p_idx].tracker);
		for (unsigned int i = p_idx + 1; i < num_anchors; i++) {
			anchors[i - 1] = anchors[i];
		};
		num_anchors--;
	};
};

void ARKitInterface::remove_all_anchors() {
	if (anchors != NULL) {
		while (num_anchors > 0) {
			remove_anchor(0);
		};

		free(anchors);
		anchors = NULL;
	};
};

void ARKitInterface::process() {
	_THREAD_SAFE_METHOD_

	if (initialized) {
		// get our next ARFrame
		ARFrame *current_frame = ar_session.currentFrame;
		if (last_timestamp != current_frame.timestamp) {
			// only process if we have a new frame
			last_timestamp = current_frame.timestamp;

			CameraFeed *feed = CameraIOS::get_arkit_feed();

			// Grab our camera image for our backbuffer
			CVPixelBufferRef pixelBuffer = current_frame.capturedImage;
			if ((CVPixelBufferGetPlaneCount(pixelBuffer) == 2) && (feed != NULL)) {
				// Plane 0 is our Y and Plane 1 is our CbCr buffer

				// not sure if we can make the assumption both planes are sized the same, I assume so
				image_width = CVPixelBufferGetWidth(pixelBuffer);
				image_height = CVPixelBufferGetHeight(pixelBuffer);

				//				printf("Pixel buffer %i - %i\n", image_width, image_height);

				// It says that we need to lock this on the documentation pages but it's not in the samples
				// need to lock our base address so we can access our pixel buffers, better safe then sorry?
				CVPixelBufferLockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);

				// get our buffers
				unsigned char *dataY = (unsigned char *)CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 0);
				unsigned char *dataCbCr = (unsigned char *)CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 1);

				if (dataY == NULL) {
					print_line("Couldn't access Y pixel buffer data");
				} else if (dataCbCr == NULL) {
					print_line("Couldn't access CbCr pixel buffer data");
				} else {
					// set our texture...
					feed->set_texture_data_YCbCr(
							dataY, CVPixelBufferGetWidthOfPlane(pixelBuffer, 0), CVPixelBufferGetHeightOfPlane(pixelBuffer, 0), dataCbCr, CVPixelBufferGetWidthOfPlane(pixelBuffer, 1), CVPixelBufferGetHeightOfPlane(pixelBuffer, 1));
				}

				// and unlock
				CVPixelBufferUnlockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);
			};

			// Record light estimation to apply to our scene
			if (light_estimation_is_enabled) {
				ambient_intensity = current_frame.lightEstimate.ambientIntensity;

				///@TODO it's there, but not there.. what to do with this...
				// https://developer.apple.com/documentation/arkit/arlightestimate?language=objc
				//				ambient_color_temperature = current_frame.lightEstimate.ambientColorTemperature;
			};

			// Process our camera
			ARCamera *camera = current_frame.camera;

			// strangely enough we have to states, rolling them up into one
			if (camera.trackingState == ARTrackingStateNotAvailable) {
				// no tracking, would be good if we black out the screen or something...
				tracking_state = ARVRInterface::ARVR_NOT_TRACKING;
			} else {
				if (camera.trackingState == ARTrackingStateNormal) {
					tracking_state = ARVRInterface::ARVR_NORMAL_TRACKING;
				} else if (camera.trackingStateReason == ARTrackingStateReasonExcessiveMotion) {
					tracking_state = ARVRInterface::ARVR_EXCESSIVE_MOTION;
				} else if (camera.trackingStateReason == ARTrackingStateReasonInsufficientFeatures) {
					tracking_state = ARVRInterface::ARVR_INSUFFICIENT_FEATURES;
				} else {
					tracking_state = ARVRInterface::ARVR_UNKNOWN_TRACKING;
				};

				// copy our current frame transform
				matrix_float4x4 m44 = camera.transform;
				transform.basis.elements[0].x = m44.columns[0][0];
				transform.basis.elements[1].x = m44.columns[0][1];
				transform.basis.elements[2].x = m44.columns[0][2];
				transform.basis.elements[0].y = m44.columns[1][0];
				transform.basis.elements[1].y = m44.columns[1][1];
				transform.basis.elements[2].y = m44.columns[1][2];
				transform.basis.elements[0].z = m44.columns[2][0];
				transform.basis.elements[1].z = m44.columns[2][1];
				transform.basis.elements[2].z = m44.columns[2][2];
				transform.origin.x = m44.columns[3][0];
				transform.origin.y = m44.columns[3][1];
				transform.origin.z = m44.columns[3][2];

				// copy our current frame projection, investigate using projectionMatrixWithViewportSize:orientation:zNear:zFar: so we can set our own near and far
				// near and far that ARKit uses by default is 0.001 and 1000.0 which are ok enough for Godot.
				m44 = camera.projectionMatrix;
				projection.matrix[0][0] = m44.columns[0][0];
				projection.matrix[1][0] = m44.columns[1][0];
				projection.matrix[2][0] = m44.columns[2][0];
				projection.matrix[3][0] = m44.columns[3][0];
				projection.matrix[0][1] = m44.columns[0][1];
				projection.matrix[1][1] = m44.columns[1][1];
				projection.matrix[2][1] = m44.columns[2][1];
				projection.matrix[3][1] = m44.columns[3][1];
				projection.matrix[0][2] = m44.columns[0][2];
				projection.matrix[1][2] = m44.columns[1][2];
				projection.matrix[2][2] = m44.columns[2][2];
				projection.matrix[3][2] = m44.columns[3][2];
				projection.matrix[0][3] = m44.columns[0][3];
				projection.matrix[1][3] = m44.columns[1][3];
				projection.matrix[2][3] = m44.columns[2][3];
				projection.matrix[3][3] = m44.columns[3][3];
			};

			// Remove old anchors, check from last to first
			if (anchors != NULL) {
				for (int i = num_anchors - 1; i >= 0; i--) {
					// there should be a faster way of doing this but....
					bool anchor_still_exists = false;

					for (ARAnchor *anchor in current_frame.anchors) {
						if (!anchor_still_exists) {
							unsigned char uuid[16];
							[anchor.identifier getUUIDBytes:uuid];

							if (memcmp(uuid, anchors[i].uuid, 16) == 0) {
								anchor_still_exists = true;
							};
						};
					};

					if (!anchor_still_exists) {
						remove_anchor(i);
					};
				};
			};

			// Find new anchors and update existing ones
			for (ARAnchor *anchor in current_frame.anchors) {
				unsigned char uuid[16];
				[anchor.identifier getUUIDBytes:uuid];

				ARVRPositionalTracker *tracker = get_anchor_for_uuid(uuid);
				if (tracker != NULL) {
					// Note, this also contains a scale factor which gives us an idea of the size of the anchor
					// We may extract that in our ARVRAnchor class
					Basis b;
					matrix_float4x4 m44 = anchor.transform;
					b.elements[0].x = m44.columns[0][0];
					b.elements[1].x = m44.columns[0][1];
					b.elements[2].x = m44.columns[0][2];
					b.elements[0].y = m44.columns[1][0];
					b.elements[1].y = m44.columns[1][1];
					b.elements[2].y = m44.columns[1][2];
					b.elements[0].z = m44.columns[2][0];
					b.elements[1].z = m44.columns[2][1];
					b.elements[2].z = m44.columns[2][2];
					///@TODO possibly extract our scale value here, or do so in ARVRAnchor?
					tracker->set_orientation(b);

					///@TODO once we support this in our ARVRServer, change this to set_real_world_position()
					tracker->set_rw_position(Vector3(m44.columns[3][0], m44.columns[3][1], m44.columns[3][2]));
				};
			};
		};
	};
};

ARKitInterface::ARKitInterface() {
	initialized = false;
	session_start = false;
	plane_detection_is_enabled = false;
	light_estimation_is_enabled = false;
	ar_session = NULL;
	z_near = 0.01;
	z_far = 1000.0;
	projection.set_perspective(60.0, 1.0, z_near, z_far, false);
	anchors = NULL;
	num_anchors = 0;
	ambient_intensity = 1.0;
	ambient_color_temperature = 1.0;
};

ARKitInterface::~ARKitInterface() {
	remove_all_anchors();

	// and make sure we cleanup if we haven't already
	if (is_initialized()) {
		uninitialize();
	};
};

#endif
