/**************************************************************************/
/*  hand_tracking_server.h                                                */
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

#include "core/object/class_db.h"
#include "servers/xr/xr_hand_tracker.h"

class HandTrackingServer : public Object {
	GDCLASS(HandTrackingServer, Object);

private:
	static HandTrackingServer *singleton;

	Ref<XRHandTracker> left_hand_tracker;
	Ref<XRHandTracker> right_hand_tracker;
	bool initialized = false;

	void _ensure_trackers();
	void _update_hand_tracker(Ref<XRHandTracker> tracker, const struct godot_hand_joint *joints, int joint_count);

protected:
	static void _bind_methods();

public:
	static HandTrackingServer *get_singleton();

	/**
	 * Check if hand tracking is available/initialized.
	 */
	bool is_hand_tracking_available() const;

	/**
	 * Get the left hand tracker.
	 * Returns null if not initialized.
	 */
	Ref<XRHandTracker> get_left_hand_tracker() const;

	/**
	 * Get the right hand tracker.
	 * Returns null if not initialized.
	 */
	Ref<XRHandTracker> get_right_hand_tracker() const;

	/**
	 * Update hand tracking data from the native platform layer.
	 * Called automatically each frame when data is available.
	 */
	void update_hand_tracking();

	HandTrackingServer();
	~HandTrackingServer();
};
