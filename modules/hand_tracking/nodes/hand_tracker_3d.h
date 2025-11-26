/**************************************************************************/
/*  hand_tracker_3d.h                                                     */
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

#include "scene/3d/node_3d.h"
#include "servers/xr/xr_hand_tracker.h"

/**
 * HandTracker3D is a Node3D that automatically tracks a specific hand joint.
 *
 * This node updates its transform each frame to match the position and orientation
 * of a selected hand joint from the HandTrackingServer. It's useful for attaching
 * objects to specific points on the hand, such as held items, UI elements, or
 * visual markers.
 *
 * Example usage:
 * ```gdscript
 * @onready var index_tip = $HandTracker3D
 * index_tip.hand = HandTracker3D.HAND_LEFT
 * index_tip.joint = XRHandTracker.HAND_JOINT_INDEX_FINGER_TIP
 * index_tip.track_rotation = true
 * ```
 */
class HandTracker3D : public Node3D {
	GDCLASS(HandTracker3D, Node3D);

public:
	enum Hand {
		HAND_LEFT,
		HAND_RIGHT,
		HAND_MAX
	};

private:
	Hand hand = HAND_LEFT;
	XRHandTracker::HandJoint joint = XRHandTracker::HAND_JOINT_WRIST;
	bool track_position = true;
	bool track_rotation = true;
	bool hide_when_invalid = true;
	float smoothing = 0.0f; // 0 = no smoothing, 0-1 = smoothing factor

	Ref<XRHandTracker> tracker;
	Transform3D target_transform;
	bool was_visible = true;

	void _update_tracking();
	void _update_tracker_reference();

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	// Hand selection
	void set_hand(Hand p_hand);
	Hand get_hand() const;

	// Joint selection
	void set_joint(XRHandTracker::HandJoint p_joint);
	XRHandTracker::HandJoint get_joint() const;

	// Tracking options
	void set_track_position(bool p_enable);
	bool get_track_position() const;

	void set_track_rotation(bool p_enable);
	bool get_track_rotation() const;

	// Visibility control
	void set_hide_when_invalid(bool p_enable);
	bool get_hide_when_invalid() const;

	// Smoothing
	void set_smoothing(float p_smoothing);
	float get_smoothing() const;

	// Status queries
	bool is_tracking_valid() const;
	Ref<XRHandTracker> get_tracker() const;

	HandTracker3D();
	~HandTracker3D();
};

VARIANT_ENUM_CAST(HandTracker3D::Hand);
