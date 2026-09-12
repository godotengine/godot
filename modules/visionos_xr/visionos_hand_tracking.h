/**************************************************************************/
/*  visionos_hand_tracking.h                                              */
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

// Hand tracking using ARKit
struct VisionOSHandTracking {
	bool enabled = false;
	VisionOSAuthorizationStatus authorization = VisionOSAuthorizationStatus::NOT_DETERMINED;

	bool active() const { return enabled && authorization == VisionOSAuthorizationStatus::ALLOWED; }

	ar_hand_tracking_provider_t hand_tracking_provider = nullptr;

	// Hand trackers
	Ref<XRHandTracker> left_hand_tracker;
	Ref<XRHandTracker> right_hand_tracker;
	ar_hand_anchor_t left_hand_anchor = nullptr;
	ar_hand_anchor_t right_hand_anchor = nullptr;

	void initialize(XRServer *xr_server);
	void update_hand_trackers_from_arkit(CFTimeInterval trackable_anchor_time);
	void reset_hand_tracker_data(Ref<XRHandTracker> hand_tracker);
	void set_hand_tracker_data_from_arkit(Ref<XRHandTracker> hand_tracker, ar_hand_anchor_t hand_anchor);
};

#endif // VISIONOS_ENABLED
