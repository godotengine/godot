/**************************************************************************/
/*  hand_gesture_detector.h                                               */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md).*/
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

#ifndef HAND_GESTURE_DETECTOR_H
#define HAND_GESTURE_DETECTOR_H

#include "hand_gesture.h"
#include "scene/3d/node_3d.h"
#include "servers/xr/xr_hand_tracker.h"

/// HandGestureDetector monitors hand tracking data and emits signals
/// when specific gestures are detected
class HandGestureDetector : public Node3D {
	GDCLASS(HandGestureDetector, Node3D);

public:
	enum Hand {
		HAND_LEFT,
		HAND_RIGHT,
		HAND_MAX
	};

private:
	// Configuration
	Hand hand = HAND_LEFT;
	TypedArray<HandGesture> gestures; // Gestures to detect
	bool enabled = true;
	float detection_rate = 0.016f; // ~60 FPS detection

	// Internal state
	Ref<XRHandTracker> tracker;
	float time_since_last_check = 0.0f;

	// Gesture state tracking
	struct GestureState {
		bool is_active = false;
		float current_confidence = 0.0f;
		float time_held = 0.0f;
	};
	Vector<GestureState> gesture_states;

	// Hysteresis to prevent flickering
	float activation_threshold = 0.0f;   // Use gesture's confidence_threshold
	float deactivation_threshold = 0.0f; // Slightly lower than activation
	float hysteresis_margin = 0.1f;      // How much lower deactivation is

protected:
	static void _bind_methods();
	void _notification(int p_what);

public:
	HandGestureDetector();
	~HandGestureDetector();

	// Configuration
	void set_hand(Hand p_hand);
	Hand get_hand() const { return hand; }

	void set_gestures(const TypedArray<HandGesture> &p_gestures);
	TypedArray<HandGesture> get_gestures() const { return gestures; }

	void add_gesture(const Ref<HandGesture> &p_gesture);
	void remove_gesture(const Ref<HandGesture> &p_gesture);
	void clear_gestures();

	void set_enabled(bool p_enabled) { enabled = p_enabled; }
	bool is_enabled() const { return enabled; }

	void set_detection_rate(float p_rate) { detection_rate = p_rate; }
	float get_detection_rate() const { return detection_rate; }

	void set_hysteresis_margin(float p_margin) { hysteresis_margin = p_margin; }
	float get_hysteresis_margin() const { return hysteresis_margin; }

	// Query methods
	bool is_gesture_active(int p_index) const;
	bool is_gesture_active_by_name(const String &p_name) const;
	float get_gesture_confidence(int p_index) const;
	float get_gesture_time_held(int p_index) const;

	// Helpers
	int find_gesture_by_name(const String &p_name) const;
	int get_gesture_count() const { return gestures.size(); }

private:
	void _update_tracker_reference();
	void _detect_gestures(float delta);
	void _check_gesture(int index, float delta);
};

VARIANT_ENUM_CAST(HandGestureDetector::Hand);

#endif // HAND_GESTURE_DETECTOR_H
