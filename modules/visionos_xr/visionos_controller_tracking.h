/**************************************************************************/
/*  visionos_controller_tracking.h                                        */
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

#pragma once

#ifdef VISIONOS_ENABLED

#include "visionos_definitions.h"

#ifdef __OBJC__
#define Key GodotKey
#import <CoreHaptics/CoreHaptics.h>
#import <GameController/GameController.h>
#undef Key
#else // __OBJC__
typedef struct GCController *GCController_t;
typedef struct CHHapticEngine *CHHapticEngine_t;
#endif // __OBJC__

class VisionOSXRInterface;

// Controller tracking using ARKit and GCController
struct VisionOSControllerTracking {
	bool enabled = false;
	VisionOSAuthorizationStatus authorization = VisionOSAuthorizationStatus::NOT_DETERMINED;

	bool active() const { return enabled && authorization == VisionOSAuthorizationStatus::ALLOWED; }

	// ARKit state
	ar_accessory_tracking_provider_t accessory_tracking_provider = nullptr;
	ar_accessories_t accessories = nullptr;
	ar_accessory_anchor_t left_controller_anchor = nullptr;
	ar_accessory_anchor_t right_controller_anchor = nullptr;

	// Controller state
	Ref<XRControllerTracker> left_controller_tracker;
	Ref<XRControllerTracker> right_controller_tracker;
	GCController *left_gc_controller = nullptr;
	GCController *right_gc_controller = nullptr;
	CHHapticEngine *left_haptic_engine = nullptr;
	CHHapticEngine *right_haptic_engine = nullptr;

	// Notification observers
	id controller_observer = nullptr;
	id controller_disconnect_observer = nullptr;

	void initialize(XRServer *p_xr_server, VisionOSXRInterface *p_xr_interface);
	void uninitialize(XRServer *p_xr_server);

	void init_for_controller(GCController *p_controller);
	void setup_controller_notifications();
	void cleanup_controller_notifications();
	void handle_controller_disconnect(GCController *p_controller);
	void update_accessories_list();
	void update_controller_trackers_from_arkit(CFTimeInterval p_trackable_anchor_time);
	void update_controller_from_anchor(Ref<XRControllerTracker> p_controller_tracker, ar_accessory_anchor_t p_controller_anchor, GCController *p_gc_controller);

	// Called directly from the same function on the XRInterface
	void trigger_haptic_pulse(const String &p_action_name, const StringName &p_tracker_name, double p_frequency, double p_amplitude, double p_duration_sec, double p_delay_sec);

	VisionOSXRInterface *xr_interface = nullptr; // assigned on initialization
};

#endif // VISIONOS_ENABLED
