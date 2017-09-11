/*************************************************************************/
/*  arvr_postional_tracker.cpp                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "arvr_positional_tracker.h"
#include "core/os/input.h"

void ARVRPositionalTracker::_bind_methods() {
	BIND_ENUM_CONSTANT(TRACKER_HAND_UNKNOWN);
	BIND_ENUM_CONSTANT(TRACKER_LEFT_HAND);
	BIND_ENUM_CONSTANT(TRACKER_RIGHT_HAND);

	// this class is read only from GDScript, so we only have access to getters..
	ClassDB::bind_method(D_METHOD("get_type"), &ARVRPositionalTracker::get_type);
	ClassDB::bind_method(D_METHOD("get_name"), &ARVRPositionalTracker::get_name);
	ClassDB::bind_method(D_METHOD("get_joy_id"), &ARVRPositionalTracker::get_joy_id);
	ClassDB::bind_method(D_METHOD("get_tracks_orientation"), &ARVRPositionalTracker::get_tracks_orientation);
	ClassDB::bind_method(D_METHOD("get_orientation"), &ARVRPositionalTracker::get_orientation);
	ClassDB::bind_method(D_METHOD("get_tracks_position"), &ARVRPositionalTracker::get_tracks_position);
	ClassDB::bind_method(D_METHOD("get_position"), &ARVRPositionalTracker::get_position);
	ClassDB::bind_method(D_METHOD("get_hand"), &ARVRPositionalTracker::get_hand);
	ClassDB::bind_method(D_METHOD("get_transform", "adjust_by_reference_frame"), &ARVRPositionalTracker::get_transform);

	// these functions we don't want to expose to normal users but do need to be callable from GDNative
	ClassDB::bind_method(D_METHOD("_set_type", "type"), &ARVRPositionalTracker::set_type);
	ClassDB::bind_method(D_METHOD("_set_name", "name"), &ARVRPositionalTracker::set_name);
	ClassDB::bind_method(D_METHOD("_set_joy_id", "joy_id"), &ARVRPositionalTracker::set_joy_id);
	ClassDB::bind_method(D_METHOD("_set_orientation", "orientation"), &ARVRPositionalTracker::set_orientation);
	ClassDB::bind_method(D_METHOD("_set_rw_position", "rw_position"), &ARVRPositionalTracker::set_rw_position);
};

void ARVRPositionalTracker::set_type(ARVRServer::TrackerType p_type) {
	if (type != p_type) {
		type = p_type;

		ARVRServer *arvr_server = ARVRServer::get_singleton();
		ERR_FAIL_NULL(arvr_server);

		// get a tracker id for our type
		tracker_id = arvr_server->get_free_tracker_id_for_type(p_type);
	}
};

ARVRServer::TrackerType ARVRPositionalTracker::get_type() const {
	return type;
};

void ARVRPositionalTracker::set_name(const String p_name) {
	name = p_name;
};

StringName ARVRPositionalTracker::get_name() const {
	return name;
};

int ARVRPositionalTracker::get_tracker_id() const {
	return tracker_id;
};

void ARVRPositionalTracker::set_joy_id(int p_joy_id) {
	joy_id = p_joy_id;
};

int ARVRPositionalTracker::get_joy_id() const {
	return joy_id;
};

bool ARVRPositionalTracker::get_tracks_orientation() const {
	return tracks_orientation;
};

void ARVRPositionalTracker::set_orientation(const Basis &p_orientation) {
	_THREAD_SAFE_METHOD_

	tracks_orientation = true; // obviously we have this
	orientation = p_orientation;
};

Basis ARVRPositionalTracker::get_orientation() const {
	_THREAD_SAFE_METHOD_

	return orientation;
};

bool ARVRPositionalTracker::get_tracks_position() const {
	return tracks_position;
};

void ARVRPositionalTracker::set_position(const Vector3 &p_position) {
	_THREAD_SAFE_METHOD_

	ARVRServer *arvr_server = ARVRServer::get_singleton();
	ERR_FAIL_NULL(arvr_server);
	real_t world_scale = arvr_server->get_world_scale();
	ERR_FAIL_COND(world_scale == 0);

	tracks_position = true; // obviously we have this
	rw_position = p_position / world_scale;
};

Vector3 ARVRPositionalTracker::get_position() const {
	_THREAD_SAFE_METHOD_

	ARVRServer *arvr_server = ARVRServer::get_singleton();
	ERR_FAIL_NULL_V(arvr_server, rw_position);
	real_t world_scale = arvr_server->get_world_scale();

	return rw_position * world_scale;
};

void ARVRPositionalTracker::set_rw_position(const Vector3 &p_rw_position) {
	_THREAD_SAFE_METHOD_

	tracks_position = true; // obviously we have this
	rw_position = p_rw_position;
};

Vector3 ARVRPositionalTracker::get_rw_position() const {
	_THREAD_SAFE_METHOD_

	return rw_position;
};

ARVRPositionalTracker::TrackerHand ARVRPositionalTracker::get_hand() const {
	return hand;
};

void ARVRPositionalTracker::set_hand(const ARVRPositionalTracker::TrackerHand p_hand) {
	hand = p_hand;
};

Transform ARVRPositionalTracker::get_transform(bool p_adjust_by_reference_frame) const {
	Transform new_transform;

	new_transform.basis = get_orientation();
	new_transform.origin = get_position();

	if (p_adjust_by_reference_frame) {
		ARVRServer *arvr_server = ARVRServer::get_singleton();
		ERR_FAIL_NULL_V(arvr_server, new_transform);

		new_transform = arvr_server->get_reference_frame() * new_transform;
	};

	return new_transform;
};

ARVRPositionalTracker::ARVRPositionalTracker() {
	type = ARVRServer::TRACKER_UNKNOWN;
	name = "Unknown";
	joy_id = -1;
	tracker_id = 0;
	tracks_orientation = false;
	tracks_position = false;
	hand = TRACKER_HAND_UNKNOWN;
};

ARVRPositionalTracker::~ARVRPositionalTracker(){

};
