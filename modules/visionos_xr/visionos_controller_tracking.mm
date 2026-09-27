/**************************************************************************/
/*  visionos_controller_tracking.mm                                       */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#ifdef VISIONOS_ENABLED

#include "visionos_controller_tracking.h"

#include "visionos_simd_helpers.h"
#include "visionos_xr_interface.h"

void VisionOSControllerTracking::initialize(XRServer *p_xr_server, VisionOSXRInterface *p_xr_interface) {
	accessories = ar_accessories_create();

	xr_interface = p_xr_interface;

	// Scan existing controllers
	for (GCController *controller in GCController.controllers) {
		init_for_controller(controller);
	}

	setup_controller_notifications();

	left_controller_tracker.instantiate();
	left_controller_tracker->set_tracker_hand(XRPositionalTracker::TRACKER_HAND_LEFT);
	left_controller_tracker->set_tracker_name("left_hand");
	left_controller_tracker->set_tracker_desc("visionOS Left Controller");
	p_xr_server->add_tracker(left_controller_tracker);

	right_controller_tracker.instantiate();
	right_controller_tracker->set_tracker_hand(XRPositionalTracker::TRACKER_HAND_RIGHT);
	right_controller_tracker->set_tracker_name("right_hand");
	right_controller_tracker->set_tracker_desc("visionOS Right Controller");
	p_xr_server->add_tracker(right_controller_tracker);

	ar_accessory_tracking_configuration_t accessory_tracking_configuration = ar_accessory_tracking_configuration_create();
	ar_accessory_tracking_configuration_set_accessories(accessory_tracking_configuration, accessories);
	accessory_tracking_provider = ar_accessory_tracking_provider_create(accessory_tracking_configuration);
}

void VisionOSControllerTracking::uninitialize(XRServer *p_xr_server) {
	if (accessory_tracking_provider != nullptr) {
		accessory_tracking_provider = nullptr;
	}

	if (p_xr_server != nullptr) {
		if (left_controller_tracker.is_valid()) {
			p_xr_server->remove_tracker(left_controller_tracker);
			left_controller_tracker.unref();
		}
		if (right_controller_tracker.is_valid()) {
			p_xr_server->remove_tracker(right_controller_tracker);
			right_controller_tracker.unref();
		}
	}

	left_controller_anchor = nullptr;
	right_controller_anchor = nullptr;
	left_gc_controller = nullptr;
	right_gc_controller = nullptr;
	left_haptic_engine = nullptr;
	right_haptic_engine = nullptr;

	cleanup_controller_notifications();
}

void VisionOSControllerTracking::init_for_controller(GCController *p_controller) {
	ar_accessory_load_from_device(
			p_controller,
			^(id<GCDevice> _Nonnull device, bool success, ar_error_t _Nullable error, ar_accessory_t _Nullable accessory) {
				ERR_FAIL_COND_MSG(!success, "Error loading from GCDevice...");

				dispatch_async(dispatch_get_main_queue(), ^{
					ar_accessory_chirality_t chirality = ar_accessory_get_inherent_chirality(accessory);
					if (chirality == ar_accessory_chirality_left) {
						left_gc_controller = p_controller;
						if (left_gc_controller != nullptr && left_gc_controller.haptics != nullptr) {
							left_haptic_engine = [left_gc_controller.haptics createEngineWithLocality:GCHapticsLocalityDefault];
						}
						ar_accessories_add_accessory(accessories, accessory);
					} else if (chirality == ar_accessory_chirality_right) {
						right_gc_controller = p_controller;
						if (right_gc_controller != nullptr && right_gc_controller.haptics != nullptr) {
							right_haptic_engine = [right_gc_controller.haptics createEngineWithLocality:GCHapticsLocalityDefault];
						}
						ar_accessories_add_accessory(accessories, accessory);
					} else {
						ERR_PRINT("Accessory with undefined chirality...");
					}
					update_accessories_list();
				});
			});
}

void VisionOSControllerTracking::setup_controller_notifications() {
	controller_observer = [NSNotificationCenter.defaultCenter
			addObserverForName:GCControllerDidConnectNotification
						object:nil
						 queue:NSOperationQueue.mainQueue
					usingBlock:^(NSNotification *notification) {
						GCController *controller = (GCController *)notification.object;
						init_for_controller(controller);
					}];

	controller_disconnect_observer = [NSNotificationCenter.defaultCenter
			addObserverForName:GCControllerDidDisconnectNotification
						object:nil
						 queue:NSOperationQueue.mainQueue
					usingBlock:^(NSNotification *notification) {
						GCController *controller = (GCController *)notification.object;
						handle_controller_disconnect(controller);
					}];
}

void VisionOSControllerTracking::handle_controller_disconnect(GCController *p_controller) {
	if (left_gc_controller != nullptr && left_gc_controller == p_controller) {
		if (left_controller_anchor != nullptr) {
			ar_accessory_t accessory_left = ar_accessory_anchor_get_accessory(left_controller_anchor);
			if (accessory_left != nullptr) {
				ar_accessories_remove_accessory(accessories, accessory_left);
				left_controller_anchor = nullptr;
				left_gc_controller = nullptr;
				left_haptic_engine = nullptr;
				update_accessories_list();
			}
		}
	} else if (right_gc_controller != nullptr && right_gc_controller == p_controller) {
		if (right_controller_anchor != nullptr) {
			ar_accessory_t accessory_right = ar_accessory_anchor_get_accessory(right_controller_anchor);
			if (accessory_right != nullptr) {
				ar_accessories_remove_accessory(accessories, accessory_right);
				right_controller_anchor = nullptr;
				right_gc_controller = nullptr;
				right_haptic_engine = nullptr;
				update_accessories_list();
			}
		}
	}
}

void VisionOSControllerTracking::cleanup_controller_notifications() {
	if (controller_observer) {
		[NSNotificationCenter.defaultCenter removeObserver:controller_observer];
		controller_observer = nullptr;
	}

	if (controller_disconnect_observer) {
		[NSNotificationCenter.defaultCenter removeObserver:controller_disconnect_observer];
		controller_disconnect_observer = nullptr;
	}
}

void VisionOSControllerTracking::update_accessories_list() {
	// Restarting the ar_session when the configuration changed
	ar_accessory_tracking_configuration_t accessory_tracking_configuration = ar_accessory_tracking_configuration_create();
	ar_accessory_tracking_configuration_set_accessories(accessory_tracking_configuration, accessories);

	accessory_tracking_provider = ar_accessory_tracking_provider_create(accessory_tracking_configuration);

	if (authorization == VisionOSAuthorizationStatus::ALLOWED) {
		xr_interface->run_ar_session();
	}
}

// Button mapping logic
namespace {

struct ButtonMapping {
	GCInputButtonName gc_input;
	// Used for triggers
	const char *action_name_float;
	// Used for button presses
	const char *action_name_bool;
};

static const ButtonMapping button_mappings[] = {
	{ GCInputGripButton, "grip", "grip_click" },
	{ GCInputTrigger, "trigger", "trigger_click" },
	{ GCInputButtonA, nullptr, "ax_button" },
	{ GCInputButtonB, nullptr, "by_button" },
	{ GCInputButtonMenu, nullptr, "menu_button" },
	{ GCInputThumbstickButton, nullptr, "primary_click" },
};

struct PoseMapping {
	const char *action_name;
	const char *ar_accessory_location_name;
};

static const PoseMapping pose_mappings[] = {
	{ "default", ar_accessory_location_name_aim },
	{ "aim", ar_accessory_location_name_aim },
	{ "grip", ar_accessory_location_name_grip },
	{ "palm", ar_accessory_location_name_grip_surface }
};

void process_button(GCControllerLiveInput *input, const ButtonMapping &button_mapping, Ref<XRControllerTracker> controller_tracker) {
	id<GCButtonElement> button = input.buttons[button_mapping.gc_input];
	if (button != nullptr) {
		if (button_mapping.action_name_float) {
			controller_tracker->set_input(button_mapping.action_name_float, button.pressedInput.value);
		}
		if (button_mapping.action_name_bool) {
			controller_tracker->set_input(button_mapping.action_name_bool, button.pressedInput.isPressed);
		}
	}
}

void process_thumbstick(GCControllerLiveInput *input, Ref<XRControllerTracker> controller_tracker) {
	id<GCDirectionPadElement> thumbstick = input.dpads[GCInputThumbstick];
	if (thumbstick != nullptr) {
		float x_value = thumbstick.xAxis.value;
		float y_value = thumbstick.yAxis.value;
		controller_tracker->set_input("primary", Vector2(x_value, y_value));
	}
}

} // namespace

void VisionOSControllerTracking::update_controller_from_anchor(Ref<XRControllerTracker> p_controller_tracker,
		ar_accessory_anchor_t p_controller_anchor, GCController *p_gc_controller) {
	simd_float4x4 origin_from_controller_anchor_simd = ar_anchor_get_origin_from_anchor_transform(p_controller_anchor);
	Transform3D origin_from_controller_anchor = MTL::simd_to_transform3D(origin_from_controller_anchor_simd);

	for (const PoseMapping &mapping : pose_mappings) {
		simd_float4x4 anchor_from_location = ar_accessory_anchor_get_anchor_from_location_transform_with_correction(p_controller_anchor, mapping.ar_accessory_location_name, ar_transform_correction_rendered);
		Transform3D pose = origin_from_controller_anchor * MTL::simd_to_transform3D(anchor_from_location);
		p_controller_tracker->set_pose(mapping.action_name, pose, Vector3(), Vector3());
	}

	// Button inputs
	if (p_gc_controller && p_gc_controller.input) {
		GCControllerLiveInput *input = p_gc_controller.input;

		for (const ButtonMapping &mapping : button_mappings) {
			process_button(input, mapping, p_controller_tracker);
		}
		process_thumbstick(input, p_controller_tracker);
	}
}

void VisionOSControllerTracking::update_controller_trackers_from_arkit(CFTimeInterval p_trackable_anchor_time) {
	if (accessory_tracking_provider != nullptr) {
		ar_accessory_anchors_t accessory_anchors = ar_accessory_tracking_provider_get_latest_anchors(accessory_tracking_provider);

		__block bool left_found = false;
		__block bool right_found = false;

		if (accessory_anchors != nullptr) {
			ar_accessory_anchors_enumerate_anchors(accessory_anchors, ^bool(ar_accessory_anchor_t accessory_anchor) {
				ar_data_provider_state_t provider_state = ar_data_provider_get_state((ar_data_provider_t)accessory_tracking_provider);
				if (provider_state != ar_data_provider_state_running) {
					return true;
				}

				if (!ar_trackable_anchor_is_tracked(accessory_anchor)) {
					return true;
				}

				// Predict anchor at target time if we have timing info
				if (p_trackable_anchor_time != 0) {
					bool success = ar_accessory_tracking_provider_predict_anchor_at_timestamp(
							accessory_tracking_provider, accessory_anchor, p_trackable_anchor_time, accessory_anchor);
					if (!success) {
						return true;
					}
				}

				ar_accessory_t accessory = ar_accessory_anchor_get_accessory(accessory_anchor);
				ar_accessory_chirality_t chirality = ar_accessory_get_inherent_chirality(accessory);

				if (chirality == ar_accessory_chirality_left && !left_found) {
					left_controller_anchor = accessory_anchor;
					update_controller_from_anchor(left_controller_tracker, left_controller_anchor, left_gc_controller);
					left_found = true;
				} else if (chirality == ar_accessory_chirality_right && !right_found) {
					right_controller_anchor = accessory_anchor;
					update_controller_from_anchor(right_controller_tracker, right_controller_anchor, right_gc_controller);
					right_found = true;
				}
				return true;
			});
		}
	}
}

void VisionOSControllerTracking::trigger_haptic_pulse(const String &p_action_name, const StringName &p_tracker_name, double p_frequency, double p_amplitude, double p_duration_sec, double p_delay_sec) {
	GCController *target_controller = nullptr;
	CHHapticEngine *target_engine = nullptr;

	if (p_tracker_name == left_controller_tracker->get_tracker_name()) {
		target_controller = left_gc_controller;
		target_engine = left_haptic_engine;
	} else if (p_tracker_name == right_controller_tracker->get_tracker_name()) {
		target_controller = right_gc_controller;
		target_engine = right_haptic_engine;
	}

	ERR_FAIL_NULL_MSG(target_controller, "Controller is nil. No haptics supported.");

	ERR_FAIL_NULL_MSG(target_engine, "Haptic engine is nil.");

	float intensity = CLAMP(p_amplitude, 0.0f, 1.0f);
	float sharpness = CLAMP(p_frequency / 1000.0f, 0.0f, 1.0f);

	NSError *error = nil;

	[target_engine startAndReturnError:&error];
	if (error) {
		ERR_FAIL_MSG(vformat("Failed to start engine: %s.", [error.localizedDescription UTF8String]));
	}

	CHHapticEventParameter *intensity_param = [[CHHapticEventParameter alloc] initWithParameterID:CHHapticEventParameterIDHapticIntensity value:intensity];
	CHHapticEventParameter *sharpness_param = [[CHHapticEventParameter alloc] initWithParameterID:CHHapticEventParameterIDHapticSharpness value:sharpness];

	CHHapticEvent *event;
	if (p_duration_sec > 0) {
		event = [[CHHapticEvent alloc]
				initWithEventType:CHHapticEventTypeHapticContinuous
					   parameters:@[ intensity_param, sharpness_param ]
					 relativeTime:0.0
						 duration:p_duration_sec];
	} else {
		event = [[CHHapticEvent alloc]
				initWithEventType:CHHapticEventTypeHapticTransient
					   parameters:@[ intensity_param, sharpness_param ]
					 relativeTime:0.0];
	}

	CHHapticPattern *pattern = [[CHHapticPattern alloc]
			 initWithEvents:@[ event ]
			parameterCurves:@[]
					  error:&error];

	if (error) {
		ERR_FAIL_MSG(vformat("Failed to create a haptics pattern: %s.", [error.localizedDescription UTF8String]));
	}

	id<CHHapticPatternPlayer> player = [target_engine createPlayerWithPattern:pattern error:&error];
	if (error) {
		ERR_FAIL_MSG(vformat("Failed to create a haptics player: %s.", [error.localizedDescription UTF8String]));
	}

	[player startAtTime:CHHapticTimeImmediate error:&error];
	if (error) {
		WARN_PRINT_ONCE(vformat("Failed to start a haptics playback: %s.", [error.localizedDescription UTF8String]));
	}

	// Stopping the engine
	{
		int64_t delay;
		if (p_duration_sec <= 0) {
			// Transient - can stop engine soon after
			delay = (int64_t)(0.1 * NSEC_PER_SEC);
		} else {
			// Continuous - stop after duration + small buffer
			delay = (int64_t)((p_duration_sec + 0.1) * NSEC_PER_SEC);
		}
		dispatch_after(dispatch_time(DISPATCH_TIME_NOW, delay), dispatch_get_main_queue(), ^{
			[target_engine stopWithCompletionHandler:^(NSError *_Nullable error) {
				if (error) {
					ERR_PRINT(vformat("Error stopping haptics engine: %s.", [error.localizedDescription UTF8String]));
				}
			}];
		});
	}
}

#endif // VISIONOS_ENABLED
