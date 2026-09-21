/**************************************************************************/
/*  visionos_spatial_events.mm                                            */
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

#include "visionos_spatial_events.h"

#include "servers/xr/xr_controller_tracker.h"
#include "servers/xr/xr_positional_tracker.h"
#include "servers/xr/xr_server.h"

#include <stdio.h>

void VisionOSSpatialEventTracking::initialize(XRServer *p_xr_server) {
	for (size_t hand_index = 0; hand_index < 2; hand_index++) {
		Hand &hand = hands[hand_index];
		hand.tracker.instantiate();
		hand.ray.instantiate();

		if (hand_index == 0) {
			hand.tracker->set_tracker_hand(XRPositionalTracker::TRACKER_HAND_LEFT);
			hand.tracker->set_tracker_name("visionos/left_hand_pinch");
			hand.tracker->set_tracker_desc("visionOS Left Hand Spatial Event");
			hand.ray->set_tracker_name("visionos/left_hand_ray");
			hand.ray->set_tracker_desc("visionOS Left Selection Ray");
		} else {
			hand.tracker->set_tracker_hand(XRPositionalTracker::TRACKER_HAND_RIGHT);
			hand.tracker->set_tracker_name("visionos/right_hand_pinch");
			hand.tracker->set_tracker_desc("visionOS Left Hand Spatial Event");
			hand.ray->set_tracker_name("visionos/right_hand_ray");
			hand.ray->set_tracker_desc("visionOS Right Selection Ray");
		}
		p_xr_server->add_tracker(hand.tracker);
		p_xr_server->add_tracker(hand.ray);
	}
}

namespace {

void uninitialize_tracker(Ref<XRControllerTracker> &p_tracker, XRServer *p_xr_server) {
	if (p_tracker.is_valid()) {
		p_xr_server->remove_tracker(p_tracker);
		p_tracker.unref();
	}
}

} // namespace

void VisionOSSpatialEventTracking::uninitialize(XRServer *p_xr_server) {
	if (p_xr_server) {
		for (Hand &hand : hands) {
			uninitialize_tracker(hand.tracker, p_xr_server);
			uninitialize_tracker(hand.ray, p_xr_server);
		}
	}
}

void VisionOSSpatialEventTracking::on_spatial_event(const VisionOSSpatialEvent &p_event) {
	const bool active = (p_event.phase == VisionOSSpatialEvent::Phase::active);

	// Hand
	Hand *hand = nullptr;
	switch (p_event.chirality) {
		case VisionOSSpatialEvent::Chirality::left:
			hand = &hands[0];
			break;
		case VisionOSSpatialEvent::Chirality::right:
			hand = &hands[1];
			break;
		default:
			break;
	}

	// Updating the pose first and sending the input second.
	if (hand) {
		// Updating the ray.
		if (active && p_event.has_ray) {
			hand->ray->set_pose("default", p_event.ray, Vector3(), Vector3());
		}

		// Updating the hand pose.
		hand->tracker->set_pose("default", p_event.hand_pose, Vector3(), Vector3());

		// Submitting input events.
		hand->tracker->set_input("trigger_click", active);
		hand->ray->set_input("trigger_click", active);
	}
}

#endif // VISIONOS_ENABLED
