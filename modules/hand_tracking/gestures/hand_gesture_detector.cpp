/**************************************************************************/
/*  hand_gesture_detector.cpp                                             */
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

#include "hand_gesture_detector.h"
#include "../hand_tracking_server.h"
#include "servers/xr/xr_server.h"

void HandGestureDetector::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_hand", "hand"), &HandGestureDetector::set_hand);
	ClassDB::bind_method(D_METHOD("get_hand"), &HandGestureDetector::get_hand);

	ClassDB::bind_method(D_METHOD("set_gestures", "gestures"), &HandGestureDetector::set_gestures);
	ClassDB::bind_method(D_METHOD("get_gestures"), &HandGestureDetector::get_gestures);

	ClassDB::bind_method(D_METHOD("add_gesture", "gesture"), &HandGestureDetector::add_gesture);
	ClassDB::bind_method(D_METHOD("remove_gesture", "gesture"), &HandGestureDetector::remove_gesture);
	ClassDB::bind_method(D_METHOD("clear_gestures"), &HandGestureDetector::clear_gestures);

	ClassDB::bind_method(D_METHOD("set_enabled", "enabled"), &HandGestureDetector::set_enabled);
	ClassDB::bind_method(D_METHOD("is_enabled"), &HandGestureDetector::is_enabled);

	ClassDB::bind_method(D_METHOD("set_detection_rate", "rate"), &HandGestureDetector::set_detection_rate);
	ClassDB::bind_method(D_METHOD("get_detection_rate"), &HandGestureDetector::get_detection_rate);

	ClassDB::bind_method(D_METHOD("set_hysteresis_margin", "margin"), &HandGestureDetector::set_hysteresis_margin);
	ClassDB::bind_method(D_METHOD("get_hysteresis_margin"), &HandGestureDetector::get_hysteresis_margin);

	ClassDB::bind_method(D_METHOD("is_gesture_active", "index"), &HandGestureDetector::is_gesture_active);
	ClassDB::bind_method(D_METHOD("is_gesture_active_by_name", "name"), &HandGestureDetector::is_gesture_active_by_name);
	ClassDB::bind_method(D_METHOD("get_gesture_confidence", "index"), &HandGestureDetector::get_gesture_confidence);
	ClassDB::bind_method(D_METHOD("get_gesture_time_held", "index"), &HandGestureDetector::get_gesture_time_held);
	ClassDB::bind_method(D_METHOD("find_gesture_by_name", "name"), &HandGestureDetector::find_gesture_by_name);
	ClassDB::bind_method(D_METHOD("get_gesture_count"), &HandGestureDetector::get_gesture_count);

	// Signals
	ADD_SIGNAL(MethodInfo("gesture_started",
			PropertyInfo(Variant::STRING, "gesture_name"),
			PropertyInfo(Variant::INT, "gesture_index"),
			PropertyInfo(Variant::FLOAT, "confidence")));

	ADD_SIGNAL(MethodInfo("gesture_ended",
			PropertyInfo(Variant::STRING, "gesture_name"),
			PropertyInfo(Variant::INT, "gesture_index"),
			PropertyInfo(Variant::FLOAT, "time_held")));

	ADD_SIGNAL(MethodInfo("gesture_held",
			PropertyInfo(Variant::STRING, "gesture_name"),
			PropertyInfo(Variant::INT, "gesture_index"),
			PropertyInfo(Variant::FLOAT, "confidence"),
			PropertyInfo(Variant::FLOAT, "time_held")));

	// Properties
	ADD_PROPERTY(PropertyInfo(Variant::INT, "hand", PROPERTY_HINT_ENUM, "Left,Right"),
			"set_hand", "get_hand");
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "gestures", PROPERTY_HINT_ARRAY_TYPE,
						 vformat("%s/%s:%s", Variant::OBJECT, PROPERTY_HINT_RESOURCE_TYPE, "HandGesture")),
			"set_gestures", "get_gestures");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "enabled"), "set_enabled", "is_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "detection_rate", PROPERTY_HINT_RANGE, "0.001,1.0,0.001"),
			"set_detection_rate", "get_detection_rate");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "hysteresis_margin", PROPERTY_HINT_RANGE, "0.0,0.5,0.01"),
			"set_hysteresis_margin", "get_hysteresis_margin");

	// Enum
	BIND_ENUM_CONSTANT(HAND_LEFT);
	BIND_ENUM_CONSTANT(HAND_RIGHT);
	BIND_ENUM_CONSTANT(HAND_MAX);
}

void HandGestureDetector::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			_update_tracker_reference();
		} break;

		case NOTIFICATION_PROCESS: {
			if (!enabled) {
				return;
			}

			float delta = get_process_delta_time();
			time_since_last_check += delta;

			if (time_since_last_check >= detection_rate) {
				_detect_gestures(time_since_last_check);
				time_since_last_check = 0.0f;
			}
		} break;

		case NOTIFICATION_EXIT_TREE: {
			// Clear tracker reference
			tracker.unref();
		} break;
	}
}

HandGestureDetector::HandGestureDetector() {
	set_process(true);
}

HandGestureDetector::~HandGestureDetector() {
}

// ========================================
// Configuration
// ========================================

void HandGestureDetector::set_hand(Hand p_hand) {
	if (hand != p_hand) {
		hand = p_hand;
		_update_tracker_reference();
	}
}

void HandGestureDetector::set_gestures(const TypedArray<HandGesture> &p_gestures) {
	gestures = p_gestures;
	gesture_states.resize(gestures.size());
	for (int i = 0; i < gesture_states.size(); i++) {
		gesture_states.write[i] = GestureState();
	}
}

void HandGestureDetector::add_gesture(const Ref<HandGesture> &p_gesture) {
	if (p_gesture.is_null()) {
		return;
	}
	gestures.append(p_gesture);
	gesture_states.resize(gestures.size());
}

void HandGestureDetector::remove_gesture(const Ref<HandGesture> &p_gesture) {
	if (p_gesture.is_null()) {
		return;
	}
	int index = gestures.find(p_gesture);
	if (index >= 0) {
		gestures.remove_at(index);
		gesture_states.remove_at(index);
	}
}

void HandGestureDetector::clear_gestures() {
	gestures.clear();
	gesture_states.clear();
}

// ========================================
// Query Methods
// ========================================

bool HandGestureDetector::is_gesture_active(int p_index) const {
	if (p_index < 0 || p_index >= gesture_states.size()) {
		return false;
	}
	return gesture_states[p_index].is_active;
}

bool HandGestureDetector::is_gesture_active_by_name(const String &p_name) const {
	int index = find_gesture_by_name(p_name);
	return is_gesture_active(index);
}

float HandGestureDetector::get_gesture_confidence(int p_index) const {
	if (p_index < 0 || p_index >= gesture_states.size()) {
		return 0.0f;
	}
	return gesture_states[p_index].current_confidence;
}

float HandGestureDetector::get_gesture_time_held(int p_index) const {
	if (p_index < 0 || p_index >= gesture_states.size()) {
		return 0.0f;
	}
	return gesture_states[p_index].time_held;
}

int HandGestureDetector::find_gesture_by_name(const String &p_name) const {
	for (int i = 0; i < gestures.size(); i++) {
		Ref<HandGesture> gesture = gestures[i];
		if (gesture.is_valid() && gesture->get_gesture_name() == p_name) {
			return i;
		}
	}
	return -1;
}

// ========================================
// Internal Logic
// ========================================

void HandGestureDetector::_update_tracker_reference() {
	tracker.unref();

	HandTrackingServer *server = HandTrackingServer::get_singleton();
	if (!server) {
		return;
	}

	XRServer *xr_server = XRServer::get_singleton();
	if (!xr_server) {
		return;
	}

	// Get tracker name based on hand selection
	StringName tracker_name = hand == HAND_LEFT ? "/user/hand_tracker/left" : "/user/hand_tracker/right";

	// Get the tracker from XRServer
	Ref<XRPositionalTracker> positional_tracker = xr_server->get_tracker(tracker_name);
	if (positional_tracker.is_valid()) {
		tracker = Ref<XRHandTracker>(Object::cast_to<XRHandTracker>(positional_tracker.ptr()));
	}
}

void HandGestureDetector::_detect_gestures(float delta) {
	// Ensure we have a valid tracker
	if (tracker.is_null()) {
		_update_tracker_reference();
		if (tracker.is_null()) {
			return;
		}
	}

	// Check if tracker has data
	if (!tracker->get_has_tracking_data()) {
		// No tracking data - deactivate all gestures
		for (int i = 0; i < gesture_states.size(); i++) {
			if (gesture_states[i].is_active) {
				gesture_states.write[i].is_active = false;
				Ref<HandGesture> gesture = gestures[i];
				if (gesture.is_valid()) {
					emit_signal("gesture_ended", gesture->get_gesture_name(), i, gesture_states[i].time_held);
				}
				gesture_states.write[i].time_held = 0.0f;
			}
			gesture_states.write[i].current_confidence = 0.0f;
		}
		return;
	}

	// Check each gesture
	for (int i = 0; i < gestures.size(); i++) {
		_check_gesture(i, delta);
	}
}

void HandGestureDetector::_check_gesture(int index, float delta) {
	if (index < 0 || index >= gestures.size()) {
		return;
	}

	Ref<HandGesture> gesture = gestures[index];
	if (gesture.is_null()) {
		return;
	}

	// Calculate current confidence
	float confidence = gesture->calculate_confidence(tracker);
	gesture_states.write[index].current_confidence = confidence;

	// Determine thresholds with hysteresis
	float activation = gesture->get_confidence_threshold();
	float deactivation = activation - hysteresis_margin;

	bool was_active = gesture_states[index].is_active;
	bool is_active = false;

	if (was_active) {
		// Currently active - check if we should deactivate
		is_active = confidence >= deactivation;
	} else {
		// Currently inactive - check if we should activate
		is_active = confidence >= activation;
	}

	// Handle state transitions
	if (is_active != was_active) {
		gesture_states.write[index].is_active = is_active;

		if (is_active) {
			// Gesture just started
			gesture_states.write[index].time_held = 0.0f;
			emit_signal("gesture_started", gesture->get_gesture_name(), index, confidence);
		} else {
			// Gesture just ended
			emit_signal("gesture_ended", gesture->get_gesture_name(), index, gesture_states[index].time_held);
			gesture_states.write[index].time_held = 0.0f;
		}
	} else if (is_active) {
		// Gesture continues - update time held
		gesture_states.write[index].time_held += delta;
		emit_signal("gesture_held", gesture->get_gesture_name(), index, confidence, gesture_states[index].time_held);
	}
}
