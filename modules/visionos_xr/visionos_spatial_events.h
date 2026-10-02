/**************************************************************************/
/*  visionos_spatial_events.h                                             */
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

#include "core/math/transform_3d.h"
#include "servers/xr/xr_controller_tracker.h"

// Equivalent to https://developer.apple.com/documentation/swiftui/spatialeventcollection/event
struct VisionOSSpatialEvent {
	// Ray
	bool has_ray;
	Transform3D ray;

	// Hand
	enum class Chirality : int {
		none = 0,
		left = 1,
		right = 2
	};
	Chirality chirality;
	Transform3D hand_pose;

	// Phase
	enum class Phase : int {
		unknown = 0,
		active = 1,
		cancelled = 2,
		ended = 3
	};
	Phase phase;
};

// Godot representation of visionOS spatial events.
struct VisionOSSpatialEventTracking {
	// Ray from center of the head to
	// the direction of the eyes, when a
	// pinch gesture begins.
	Ref<XRControllerTracker> eyes_ray;

	struct Hand {
		// Hand pose when pinching and dragging.
		VisionOSSharedController *controller;

		// Update the ray only once per gesture.
		bool ray_submitted = false;

		// Correcting the transforms from each hand to
		// map to the Godot and OpenXR convention:
		// https://registry.khronos.org/OpenXR/specs/1.1/html/xrspec.html#XR_EXT_hand_interaction
		Transform3D transform_correction;
	};

	// Left and right hands.
	Hand left_hand, right_hand;

	void initialize(XRServer *p_xr_server, VisionOSSharedController &p_left_hand,
			VisionOSSharedController &p_right_hand);
	void uninitialize(XRServer *p_xr_server);

	void on_spatial_event(const VisionOSSpatialEvent &);
};

#endif // VISIONOS_ENABLED
