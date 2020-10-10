/*************************************************************************/
/*  arkit_interface.mm                                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "core/input/input.h"
#include "core/os/os.h"
#include "scene/resources/surface_tool.h"
#include "servers/rendering/rendering_server_globals.h"

#import <ARKit/ARKit.h>
#import <UIKit/UIKit.h>

#include <dlfcn.h>

#include "arkit_interface.h"
#include "arkit_session_delegate.h"

// just a dirty workaround for now, declare these as globals. I'll probably encapsulate ARSession and associated logic into an mm object and change ARKitInterface to a normal cpp object that consumes it.
API_AVAILABLE(ios(11.0))
ARSession *ar_session;

ARKitSessionDelegate *ar_delegate;
NSTimeInterval last_timestamp;

/* this is called when we initialize or when we come back from having our app pushed to the background, just (re)start our session */
void ARKitInterface::start_session() {
	// We're active...
	session_was_started = true;

	// Ignore this if we're not initialized...
	if (initialized) {
		print_line("Starting ARKit session");

		if (@available(iOS 11, *)) {
			Class ARWorldTrackingConfigurationClass = NSClassFromString(@"ARWorldTrackingConfiguration");
			ARWorldTrackingConfiguration *configuration = [ARWorldTrackingConfigurationClass new];

			configuration.lightEstimationEnabled = light_estimation_is_enabled;
			if (plane_detection_is_enabled) {
				if (@available(iOS 11.3, *)) {
					configuration.planeDetection = ARPlaneDetectionVertical | ARPlaneDetectionHorizontal;
				} else {
					configuration.planeDetection = ARPlaneDetectionHorizontal;
				}
			} else {
				configuration.planeDetection = 0;
			}

			// make sure our camera is on
			if (feed.is_valid()) {
				feed->set_active(true);
			}

			[ar_session runWithConfiguration:configuration];
		}
	}
}

void ARKitInterface::stop_session() {
	session_was_started = false;

	// Ignore this if we're not initialized...
	if (initialized) {
		// make sure our camera is off
		if (feed.is_valid()) {
			feed->set_active(false);
		}

		if (@available(iOS 11.0, *)) {
			[ar_session pause];
		}
	}
}

void ARKitInterface::notification(int p_what) {
	// TODO, this is not being called, need to find out why, possibly because this is not a node.
	// in that case we need to find a way to get these notifications!
	switch (p_what) {
		case DisplayServer::WINDOW_EVENT_FOCUS_IN: {
			print_line("Focus in");

			start_session();
		}; break;
		case DisplayServer::WINDOW_EVENT_FOCUS_OUT: {
			print_line("Focus out");

			stop_session();
		}; break;
		default:
			break;
	}
}

bool ARKitInterface::get_anchor_detection_is_enabled() const {
	return plane_detection_is_enabled;
}

void ARKitInterface::set_anchor_detection_is_enabled(bool p_enable) {
	if (plane_detection_is_enabled != p_enable) {
		plane_detection_is_enabled = p_enable;

		// Restart our session (this will be ignore if we're not initialised)
		if (session_was_started) {
			start_session();
		}
	}
}

int ARKitInterface::get_camera_feed_id() {
	if (feed.is_null()) {
		return 0;
	} else {
		return feed->get_id();
	}
}

bool ARKitInterface::get_light_estimation_is_enabled() const {
	return light_estimation_is_enabled;
}

void ARKitInterface::set_light_estimation_is_enabled(bool p_enable) {
	if (light_estimation_is_enabled != p_enable) {
		light_estimation_is_enabled = p_enable;

		// Restart our session (this will be ignore if we're not initialised)
		if (session_was_started) {
			start_session();
		}
	}
}

real_t ARKitInterface::get_ambient_intensity() const {
	return ambient_intensity;
}

real_t ARKitInterface::get_ambient_color_temperature() const {
	return ambient_color_temperature;
}

StringName ARKitInterface::get_name() const {
	return "ARKit";
}

int ARKitInterface::get_capabilities() const {
	return ARKitInterface::XR_MONO + ARKitInterface::XR_AR;
}

Array ARKitInterface::raycast(Vector2 p_screen_coord) {
	if (@available(iOS 11, *)) {
		Array arr;
		Size2 screen_size = DisplayServer::get_singleton()->screen_get_size();
		CGPoint point;
		point.x = p_screen_coord.x / screen_size.x;
		point.y = p_screen_coord.y / screen_size.y;

		///@TODO maybe give more options here, for now we're taking just ARAchors into account that were found during plane detection keeping their size into account

		NSArray<ARHitTestResult *> *results = [ar_session.currentFrame hitTest:point types:ARHitTestResultTypeExistingPlaneUsingExtent];

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
		}

		return arr;
	} else {
		return Array();
	}
}

void ARKitInterface::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_notification", "what"), &ARKitInterface::_notification);

	ClassDB::bind_method(D_METHOD("set_light_estimation_is_enabled", "enable"), &ARKitInterface::set_light_estimation_is_enabled);
	ClassDB::bind_method(D_METHOD("get_light_estimation_is_enabled"), &ARKitInterface::get_light_estimation_is_enabled);
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "light_estimation"), "set_light_estimation_is_enabled", "get_light_estimation_is_enabled");

	ClassDB::bind_method(D_METHOD("get_ambient_intensity"), &ARKitInterface::get_ambient_intensity);
	ClassDB::bind_method(D_METHOD("get_ambient_color_temperature"), &ARKitInterface::get_ambient_color_temperature);

	ClassDB::bind_method(D_METHOD("raycast", "screen_coord"), &ARKitInterface::raycast);
}

bool ARKitInterface::is_stereo() {
	// this is a mono device...
	return false;
}

bool ARKitInterface::is_initialized() const {
	return initialized;
}

bool ARKitInterface::initialize() {
	XRServer *xr_server = XRServer::get_singleton();
	ERR_FAIL_NULL_V(xr_server, false);

	if (@available(iOS 11, *)) {
		if (!initialized) {
			print_line("initializing ARKit");

			// create our ar session and delegate
			Class ARSessionClass = NSClassFromString(@"ARSession");
			if (ARSessionClass == Nil) {
				void *arkit_handle = dlopen("/System/Library/Frameworks/ARKit.framework/ARKit", RTLD_NOW);
				if (arkit_handle) {
					ARSessionClass = NSClassFromString(@"ARSession");
				} else {
					print_line("ARKit init failed");
					return false;
				}
			}
			ar_session = [ARSessionClass new];
			ar_delegate = [ARKitSessionDelegate new];
			ar_delegate.arkit_interface = this;
			ar_session.delegate = ar_delegate;

			// reset our transform
			transform = Transform();

			// make this our primary interface
			xr_server->set_primary_interface(this);

			// make sure we have our feed setup
			if (feed.is_null()) {
				feed.instance();
				feed->set_name("ARKit");

				CameraServer *cs = CameraServer::get_singleton();
				if (cs != NULL) {
					cs->add_feed(feed);
				}
			}
			feed->set_active(true);

			// yeah!
			initialized = true;

			// Start our session...
			start_session();
		}

		return true;
	} else {
		return false;
	}
}

void ARKitInterface::uninitialize() {
	if (initialized) {
		XRServer *xr_server = XRServer::get_singleton();
		if (xr_server != NULL) {
			// no longer our primary interface
			xr_server->clear_primary_interface_if(this);
		}

		if (feed.is_valid()) {
			CameraServer *cs = CameraServer::get_singleton();
			if ((cs != NULL)) {
				cs->remove_feed(feed);
			}
			feed.unref();
		}

		remove_all_anchors();

		if (@available(iOS 11.0, *)) {
			ar_session = nil;
		}

		ar_delegate = nil;
		initialized = false;
		session_was_started = false;
	}
}

Size2 ARKitInterface::get_render_targetsize() {
	// _THREAD_SAFE_METHOD_

	Size2 target_size = DisplayServer::get_singleton()->screen_get_size();

	return target_size;
}

Transform ARKitInterface::get_transform_for_eye(XRInterface::Eyes p_eye, const Transform &p_cam_transform) {
	// _THREAD_SAFE_METHOD_

	Transform transform_for_eye;

	XRServer *xr_server = XRServer::get_singleton();
	ERR_FAIL_NULL_V(xr_server, transform_for_eye);

	if (initialized) {
		float world_scale = xr_server->get_world_scale();

		// just scale our origin point of our transform, note that we really shouldn't be using world_scale in ARKit but....
		transform_for_eye = transform;
		transform_for_eye.origin *= world_scale;

		transform_for_eye = p_cam_transform * xr_server->get_reference_frame() * transform_for_eye;
	} else {
		// huh? well just return what we got....
		transform_for_eye = p_cam_transform;
	}

	return transform_for_eye;
}

CameraMatrix ARKitInterface::get_projection_for_eye(XRInterface::Eyes p_eye, real_t p_aspect, real_t p_z_near, real_t p_z_far) {
	// Remember our near and far, it will be used in process when we obtain our projection from our ARKit session.
	z_near = p_z_near;
	z_far = p_z_far;

	return projection;
}

void ARKitInterface::commit_for_eye(XRInterface::Eyes p_eye, RID p_render_target, const Rect2 &p_screen_rect) {
	// _THREAD_SAFE_METHOD_

	// We must have a valid render target
	ERR_FAIL_COND(!p_render_target.is_valid());

	// Because we are rendering to our device we must use our main viewport!
	ERR_FAIL_COND(p_screen_rect == Rect2());

	// get the size of our screen
	//	Rect2 screen_rect = p_screen_rect;

	//		screen_rect.position.x += screen_rect.size.x;
	//		screen_rect.size.x = -screen_rect.size.x;
	//		screen_rect.position.y += screen_rect.size.y;
	//		screen_rect.size.y = -screen_rect.size.y;

	//	VSG::rasterizer->set_current_render_target(RID());
	//	VSG::rasterizer->blit_render_target_to_screen(p_render_target, screen_rect, 0);
}

XRPositionalTracker *ARKitInterface::get_anchor_for_uuid(const unsigned char *p_uuid) {
	if (anchors == NULL) {
		num_anchors = 0;
		max_anchors = 10;
		anchors = (anchor_map *)malloc(sizeof(anchor_map) * max_anchors);
	}

	ERR_FAIL_NULL_V(anchors, NULL);

	for (unsigned int i = 0; i < num_anchors; i++) {
		if (memcmp(anchors[i].uuid, p_uuid, 16) == 0) {
			return anchors[i].tracker;
		}
	}

	if (num_anchors + 1 == max_anchors) {
		max_anchors += 10;
		anchors = (anchor_map *)realloc(anchors, sizeof(anchor_map) * max_anchors);
		ERR_FAIL_NULL_V(anchors, NULL);
	}

	XRPositionalTracker *new_tracker = memnew(XRPositionalTracker);
	new_tracker->set_type(XRServer::TRACKER_ANCHOR);

	char tracker_name[256];
	sprintf(tracker_name, "Anchor %02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x", p_uuid[0], p_uuid[1], p_uuid[2], p_uuid[3], p_uuid[4], p_uuid[5], p_uuid[6], p_uuid[7], p_uuid[8], p_uuid[9], p_uuid[10], p_uuid[11], p_uuid[12], p_uuid[13], p_uuid[14], p_uuid[15]);

	String name = tracker_name;
	print_line("Adding tracker " + name);
	new_tracker->set_name(name);

	// add our tracker
	XRServer::get_singleton()->add_tracker(new_tracker);
	anchors[num_anchors].tracker = new_tracker;
	memcpy(anchors[num_anchors].uuid, p_uuid, 16);
	num_anchors++;

	return new_tracker;
}

void ARKitInterface::remove_anchor_for_uuid(const unsigned char *p_uuid) {
	if (anchors != NULL) {
		for (unsigned int i = 0; i < num_anchors; i++) {
			if (memcmp(anchors[i].uuid, p_uuid, 16) == 0) {
				// remove our tracker
				XRServer::get_singleton()->remove_tracker(anchors[i].tracker);
				memdelete(anchors[i].tracker);

				// bring remaining forward
				for (unsigned int j = i + 1; j < num_anchors; j++) {
					anchors[j - 1] = anchors[j];
				};

				// decrease count
				num_anchors--;
				return;
			}
		}
	}
}

void ARKitInterface::remove_all_anchors() {
	if (anchors != NULL) {
		for (unsigned int i = 0; i < num_anchors; i++) {
			// remove our tracker
			XRServer::get_singleton()->remove_tracker(anchors[i].tracker);
			memdelete(anchors[i].tracker);
		};

		free(anchors);
		anchors = NULL;
		num_anchors = 0;
	}
}

void ARKitInterface::process() {
	// _THREAD_SAFE_METHOD_

	if (@available(iOS 11.0, *)) {
		if (initialized) {
			// get our next ARFrame
			ARFrame *current_frame = ar_session.currentFrame;
			if (last_timestamp != current_frame.timestamp) {
				// only process if we have a new frame
				last_timestamp = current_frame.timestamp;

				// get some info about our screen and orientation
				Size2 screen_size = DisplayServer::get_singleton()->screen_get_size();
				UIInterfaceOrientation orientation = UIInterfaceOrientationUnknown;

				if (@available(iOS 13, *)) {
					orientation = [UIApplication sharedApplication].delegate.window.windowScene.interfaceOrientation;
#if !defined(TARGET_OS_SIMULATOR) || !TARGET_OS_SIMULATOR
				} else {
					orientation = [[UIApplication sharedApplication] statusBarOrientation];
#endif
				}

				// Grab our camera image for our backbuffer
				CVPixelBufferRef pixelBuffer = current_frame.capturedImage;
				if ((CVPixelBufferGetPlaneCount(pixelBuffer) == 2) && (feed != NULL)) {
					// Plane 0 is our Y and Plane 1 is our CbCr buffer

					// ignored, we check each plane separately
					// image_width = CVPixelBufferGetWidth(pixelBuffer);
					// image_height = CVPixelBufferGetHeight(pixelBuffer);

					// printf("Pixel buffer %i - %i\n", image_width, image_height);

					CVPixelBufferLockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);

					// get our buffers
					unsigned char *dataY = (unsigned char *)CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 0);
					unsigned char *dataCbCr = (unsigned char *)CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 1);

					if (dataY == NULL) {
						print_line("Couldn't access Y pixel buffer data");
					} else if (dataCbCr == NULL) {
						print_line("Couldn't access CbCr pixel buffer data");
					} else {
						Ref<Image> img[2];
						size_t extraLeft, extraRight, extraTop, extraBottom;

						CVPixelBufferGetExtendedPixels(pixelBuffer, &extraLeft, &extraRight, &extraTop, &extraBottom);

						{
							// do Y
							size_t new_width = CVPixelBufferGetWidthOfPlane(pixelBuffer, 0);
							size_t new_height = CVPixelBufferGetHeightOfPlane(pixelBuffer, 0);
							size_t bytes_per_row = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 0);

							if ((image_width[0] != new_width) || (image_height[0] != new_height)) {
								printf("- Camera padding l:%lu r:%lu t:%lu b:%lu\n", extraLeft, extraRight, extraTop, extraBottom);
								printf("- Camera Y plane size: %lu, %lu - %lu\n", new_width, new_height, bytes_per_row);

								image_width[0] = new_width;
								image_height[0] = new_height;
								img_data[0].resize(new_width * new_height);
							}

							uint8_t *w = img_data[0].ptrw();
							if (new_width == bytes_per_row) {
								memcpy(w, dataY, new_width * new_height);
							} else {
								size_t offset_a = 0;
								size_t offset_b = extraLeft + (extraTop * bytes_per_row);
								for (size_t r = 0; r < new_height; r++) {
									memcpy(w + offset_a, dataY + offset_b, new_width);
									offset_a += new_width;
									offset_b += bytes_per_row;
								}
							}

							img[0].instance();
							img[0]->create(new_width, new_height, 0, Image::FORMAT_R8, img_data[0]);
						}

						{
							// do CbCr
							size_t new_width = CVPixelBufferGetWidthOfPlane(pixelBuffer, 1);
							size_t new_height = CVPixelBufferGetHeightOfPlane(pixelBuffer, 1);
							size_t bytes_per_row = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 0);

							if ((image_width[1] != new_width) || (image_height[1] != new_height)) {
								printf("- Camera CbCr plane size: %lu, %lu - %lu\n", new_width, new_height, bytes_per_row);

								image_width[1] = new_width;
								image_height[1] = new_height;
								img_data[1].resize(2 * new_width * new_height);
							}

							uint8_t *w = img_data[1].ptrw();
							if ((2 * new_width) == bytes_per_row) {
								memcpy(w, dataCbCr, 2 * new_width * new_height);
							} else {
								size_t offset_a = 0;
								size_t offset_b = extraLeft + (extraTop * bytes_per_row);
								for (size_t r = 0; r < new_height; r++) {
									memcpy(w + offset_a, dataCbCr + offset_b, 2 * new_width);
									offset_a += 2 * new_width;
									offset_b += bytes_per_row;
								}
							}

							img[1].instance();
							img[1]->create(new_width, new_height, 0, Image::FORMAT_RG8, img_data[1]);
						}

						// set our texture...
						feed->set_YCbCr_imgs(img[0], img[1]);

						// now build our transform to display this as a background image that matches our camera
						CGAffineTransform affine_transform = [current_frame displayTransformForOrientation:orientation viewportSize:CGSizeMake(screen_size.width, screen_size.height)];

						// we need to invert this, probably row v.s. column notation
						affine_transform = CGAffineTransformInvert(affine_transform);

						if (orientation != UIInterfaceOrientationPortrait) {
							affine_transform.b = -affine_transform.b;
							affine_transform.d = -affine_transform.d;
							affine_transform.ty = 1.0 - affine_transform.ty;
						} else {
							affine_transform.c = -affine_transform.c;
							affine_transform.a = -affine_transform.a;
							affine_transform.tx = 1.0 - affine_transform.tx;
						}

						Transform2D display_transform = Transform2D(
								affine_transform.a, affine_transform.b,
								affine_transform.c, affine_transform.d,
								affine_transform.tx, affine_transform.ty);

						feed->set_transform(display_transform);
					}

					// and unlock
					CVPixelBufferUnlockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);
				}

				// Record light estimation to apply to our scene
				if (light_estimation_is_enabled) {
					ambient_intensity = current_frame.lightEstimate.ambientIntensity;

					///@TODO it's there, but not there.. what to do with this...
					// https://developer.apple.com/documentation/arkit/arlightestimate?language=objc
					//				ambient_color_temperature = current_frame.lightEstimate.ambientColorTemperature;
				}

				// Process our camera
				ARCamera *camera = current_frame.camera;

				// strangely enough we have to states, rolling them up into one
				if (camera.trackingState == ARTrackingStateNotAvailable) {
					// no tracking, would be good if we black out the screen or something...
					tracking_state = XRInterface::XR_NOT_TRACKING;
				} else {
					if (camera.trackingState == ARTrackingStateNormal) {
						tracking_state = XRInterface::XR_NORMAL_TRACKING;
					} else if (camera.trackingStateReason == ARTrackingStateReasonExcessiveMotion) {
						tracking_state = XRInterface::XR_EXCESSIVE_MOTION;
					} else if (camera.trackingStateReason == ARTrackingStateReasonInsufficientFeatures) {
						tracking_state = XRInterface::XR_INSUFFICIENT_FEATURES;
					} else {
						tracking_state = XRInterface::XR_UNKNOWN_TRACKING;
					}

					// copy our current frame transform
					matrix_float4x4 m44 = camera.transform;
					if (orientation == UIInterfaceOrientationLandscapeLeft) {
						transform.basis.elements[0].x = m44.columns[0][0];
						transform.basis.elements[1].x = m44.columns[0][1];
						transform.basis.elements[2].x = m44.columns[0][2];
						transform.basis.elements[0].y = m44.columns[1][0];
						transform.basis.elements[1].y = m44.columns[1][1];
						transform.basis.elements[2].y = m44.columns[1][2];
					} else if (orientation == UIInterfaceOrientationPortrait) {
						transform.basis.elements[0].x = m44.columns[1][0];
						transform.basis.elements[1].x = m44.columns[1][1];
						transform.basis.elements[2].x = m44.columns[1][2];
						transform.basis.elements[0].y = -m44.columns[0][0];
						transform.basis.elements[1].y = -m44.columns[0][1];
						transform.basis.elements[2].y = -m44.columns[0][2];
					} else if (orientation == UIInterfaceOrientationLandscapeRight) {
						transform.basis.elements[0].x = -m44.columns[0][0];
						transform.basis.elements[1].x = -m44.columns[0][1];
						transform.basis.elements[2].x = -m44.columns[0][2];
						transform.basis.elements[0].y = -m44.columns[1][0];
						transform.basis.elements[1].y = -m44.columns[1][1];
						transform.basis.elements[2].y = -m44.columns[1][2];
					} else if (orientation == UIInterfaceOrientationPortraitUpsideDown) {
						// this may not be correct
						transform.basis.elements[0].x = m44.columns[1][0];
						transform.basis.elements[1].x = m44.columns[1][1];
						transform.basis.elements[2].x = m44.columns[1][2];
						transform.basis.elements[0].y = m44.columns[0][0];
						transform.basis.elements[1].y = m44.columns[0][1];
						transform.basis.elements[2].y = m44.columns[0][2];
					}
					transform.basis.elements[0].z = m44.columns[2][0];
					transform.basis.elements[1].z = m44.columns[2][1];
					transform.basis.elements[2].z = m44.columns[2][2];
					transform.origin.x = m44.columns[3][0];
					transform.origin.y = m44.columns[3][1];
					transform.origin.z = m44.columns[3][2];

					// copy our current frame projection, investigate using projectionMatrixWithViewportSize:orientation:zNear:zFar: so we can set our own near and far
					m44 = [camera projectionMatrixForOrientation:orientation viewportSize:CGSizeMake(screen_size.width, screen_size.height) zNear:z_near zFar:z_far];
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
				}
			}
		}
	}
}

void ARKitInterface::_add_or_update_anchor(GodotARAnchor *p_anchor) {
	// _THREAD_SAFE_METHOD_

	if (@available(iOS 11.0, *)) {
		ARAnchor *anchor = (ARAnchor *)p_anchor;

		unsigned char uuid[16];
		[anchor.identifier getUUIDBytes:uuid];

		XRPositionalTracker *tracker = get_anchor_for_uuid(uuid);
		if (tracker != NULL) {
			// lets update our mesh! (using Arjens code as is for now)
			// we should also probably limit how often we do this...

			// can we safely cast this?
			ARPlaneAnchor *planeAnchor = (ARPlaneAnchor *)anchor;

			if (@available(iOS 11.3, *)) {
				if (planeAnchor.geometry.triangleCount > 0) {
					Ref<SurfaceTool> surftool;
					surftool.instance();
					surftool->begin(Mesh::PRIMITIVE_TRIANGLES);

					for (int j = planeAnchor.geometry.triangleCount * 3 - 1; j >= 0; j--) {
						int16_t index = planeAnchor.geometry.triangleIndices[j];
						simd_float3 vrtx = planeAnchor.geometry.vertices[index];
						simd_float2 textcoord = planeAnchor.geometry.textureCoordinates[index];
						surftool->add_uv(Vector2(textcoord[0], textcoord[1]));
						surftool->add_color(Color(0.8, 0.8, 0.8));
						surftool->add_vertex(Vector3(vrtx[0], vrtx[1], vrtx[2]));
					}

					surftool->generate_normals();
					tracker->set_mesh(surftool->commit());
				} else {
					Ref<Mesh> nomesh;
					tracker->set_mesh(nomesh);
				}
			} else {
				Ref<Mesh> nomesh;
				tracker->set_mesh(nomesh);
			}

			// Note, this also contains a scale factor which gives us an idea of the size of the anchor
			// We may extract that in our XRAnchor class
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
			tracker->set_orientation(b);
			tracker->set_rw_position(Vector3(m44.columns[3][0], m44.columns[3][1], m44.columns[3][2]));
		}
	}
}

void ARKitInterface::_remove_anchor(GodotARAnchor *p_anchor) {
	// _THREAD_SAFE_METHOD_

	if (@available(iOS 11.0, *)) {
		ARAnchor *anchor = (ARAnchor *)p_anchor;

		unsigned char uuid[16];
		[anchor.identifier getUUIDBytes:uuid];

		remove_anchor_for_uuid(uuid);
	}
}

ARKitInterface::ARKitInterface() {
	initialized = false;
	session_was_started = false;
	plane_detection_is_enabled = false;
	light_estimation_is_enabled = false;
	if (@available(iOS 11.0, *)) {
		ar_session = nil;
	}
	z_near = 0.01;
	z_far = 1000.0;
	projection.set_perspective(60.0, 1.0, z_near, z_far, false);
	anchors = NULL;
	num_anchors = 0;
	ambient_intensity = 1.0;
	ambient_color_temperature = 1.0;
	image_width[0] = 0;
	image_width[1] = 0;
	image_height[0] = 0;
	image_height[1] = 0;
}

ARKitInterface::~ARKitInterface() {
	remove_all_anchors();

	// and make sure we cleanup if we haven't already
	if (is_initialized()) {
		uninitialize();
	}
}
