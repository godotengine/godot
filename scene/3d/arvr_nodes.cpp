/*************************************************************************/
/*  arvr_nodes.cpp                                                       */
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

#include "arvr_nodes.h"
#include "core/os/input.h"
#include "servers/arvr/arvr_interface.h"
#include "servers/arvr_server.h"

////////////////////////////////////////////////////////////////////////////////////////////////////

void ARVRCamera::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			// need to find our ARVROrigin parent and let it know we're it's camera!
			ARVROrigin *origin = Object::cast_to<ARVROrigin>(get_parent());
			if (origin != NULL) {
				origin->set_tracked_camera(this);
			}
		}; break;
		case NOTIFICATION_EXIT_TREE: {
			// need to find our ARVROrigin parent and let it know we're no longer it's camera!
			ARVROrigin *origin = Object::cast_to<ARVROrigin>(get_parent());
			if (origin != NULL) {
				origin->clear_tracked_camera_if(this);
			}
		}; break;
	};
};

String ARVRCamera::get_configuration_warning() const {
	if (!is_visible() || !is_inside_tree())
		return String();

	// must be child node of ARVROrigin!
	ARVROrigin *origin = Object::cast_to<ARVROrigin>(get_parent());
	if (origin == NULL) {
		return TTR("ARVRCamera must have an ARVROrigin node as its parent");
	};

	return String();
};

ARVRCamera::ARVRCamera(){
	// nothing to do here yet for now..
};

ARVRCamera::~ARVRCamera(){
	// nothing to do here yet for now..
};

////////////////////////////////////////////////////////////////////////////////////////////////////

void ARVRController::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			set_process_internal(true);
		}; break;
		case NOTIFICATION_EXIT_TREE: {
			set_process_internal(false);
		}; break;
		case NOTIFICATION_INTERNAL_PROCESS: {
			// get our ARVRServer
			ARVRServer *arvr_server = ARVRServer::get_singleton();
			ERR_FAIL_NULL(arvr_server);

			// find the tracker for our controller
			ARVRPositionalTracker *tracker = arvr_server->find_by_type_and_id(ARVRServer::TRACKER_CONTROLLER, controller_id);
			if (tracker == NULL) {
				// this controller is currently turned off
				is_active = false;
				button_states = 0;
			} else {
				is_active = true;
				set_transform(tracker->get_transform(true));

				int joy_id = tracker->get_joy_id();
				if (joy_id >= 0) {
					int mask = 1;
					// check button states
					for (int i = 0; i < 16; i++) {
						bool was_pressed = (button_states && mask) == mask;
						bool is_pressed = Input::get_singleton()->is_joy_button_pressed(joy_id, i);

						if (!was_pressed && is_pressed) {
							emit_signal("button_pressed", i);
							button_states += mask;
						} else if (was_pressed && !is_pressed) {
							emit_signal("button_release", i);
							button_states -= mask;
						};

						mask = mask << 1;
					};

				} else {
					button_states = 0;
				};
			};
		}; break;
		default:
			break;
	};
};

void ARVRController::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_controller_id", "controller_id"), &ARVRController::set_controller_id);
	ClassDB::bind_method(D_METHOD("get_controller_id"), &ARVRController::get_controller_id);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "controller_id"), "set_controller_id", "get_controller_id");
	ClassDB::bind_method(D_METHOD("get_controller_name"), &ARVRController::get_controller_name);

	// passthroughs to information about our related joystick
	ClassDB::bind_method(D_METHOD("get_joystick_id"), &ARVRController::get_joystick_id);
	ClassDB::bind_method(D_METHOD("is_button_pressed", "button"), &ARVRController::is_button_pressed);
	ClassDB::bind_method(D_METHOD("get_joystick_axis", "axis"), &ARVRController::get_joystick_axis);

	ClassDB::bind_method(D_METHOD("get_is_active"), &ARVRController::get_is_active);
	ClassDB::bind_method(D_METHOD("get_hand"), &ARVRController::get_hand);

	ADD_SIGNAL(MethodInfo("button_pressed", PropertyInfo(Variant::INT, "button")));
	ADD_SIGNAL(MethodInfo("button_release", PropertyInfo(Variant::INT, "button")));
};

void ARVRController::set_controller_id(int p_controller_id) {
	// we don't check any bounds here, this controller may not yet be active and just be a place holder until it is.
	controller_id = p_controller_id;
};

int ARVRController::get_controller_id(void) const {
	return controller_id;
};

String ARVRController::get_controller_name(void) const {
	// get our ARVRServer
	ARVRServer *arvr_server = ARVRServer::get_singleton();
	ERR_FAIL_NULL_V(arvr_server, String());

	ARVRPositionalTracker *tracker = arvr_server->find_by_type_and_id(ARVRServer::TRACKER_CONTROLLER, controller_id);
	if (tracker == NULL) {
		return String("Not connected");
	};

	return tracker->get_name();
};

int ARVRController::get_joystick_id() const {
	// get our ARVRServer
	ARVRServer *arvr_server = ARVRServer::get_singleton();
	ERR_FAIL_NULL_V(arvr_server, 0);

	ARVRPositionalTracker *tracker = arvr_server->find_by_type_and_id(ARVRServer::TRACKER_CONTROLLER, controller_id);
	if (tracker == NULL) {
		return 0;
	};

	return tracker->get_joy_id();
};

int ARVRController::is_button_pressed(int p_button) const {
	int joy_id = get_joystick_id();
	if (joy_id == -1) {
		return false;
	};

	return Input::get_singleton()->is_joy_button_pressed(joy_id, p_button);
};

float ARVRController::get_joystick_axis(int p_axis) const {
	int joy_id = get_joystick_id();
	if (joy_id == -1) {
		return 0.0;
	};

	return Input::get_singleton()->get_joy_axis(joy_id, p_axis);
};

bool ARVRController::get_is_active() const {
	return is_active;
};

ARVRPositionalTracker::TrackerHand ARVRController::get_hand() const {
	// get our ARVRServer
	ARVRServer *arvr_server = ARVRServer::get_singleton();
	ERR_FAIL_NULL_V(arvr_server, ARVRPositionalTracker::TRACKER_HAND_UNKNOWN);

	ARVRPositionalTracker *tracker = arvr_server->find_by_type_and_id(ARVRServer::TRACKER_CONTROLLER, controller_id);
	if (tracker == NULL) {
		return ARVRPositionalTracker::TRACKER_HAND_UNKNOWN;
	};

	return tracker->get_hand();
};

String ARVRController::get_configuration_warning() const {
	if (!is_visible() || !is_inside_tree())
		return String();

	// must be child node of ARVROrigin!
	ARVROrigin *origin = Object::cast_to<ARVROrigin>(get_parent());
	if (origin == NULL) {
		return TTR("ARVRController must have an ARVROrigin node as its parent");
	};

	if (controller_id == 0) {
		return TTR("The controller id must not be 0 or this controller will not be bound to an actual controller");
	};

	return String();
};

ARVRController::ARVRController() {
	controller_id = 0;
	is_active = true;
};

ARVRController::~ARVRController(){
	// nothing to do here yet for now..
};

////////////////////////////////////////////////////////////////////////////////////////////////////

void ARVRAnchor::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			set_process_internal(true);
		}; break;
		case NOTIFICATION_EXIT_TREE: {
			set_process_internal(false);
		}; break;
		case NOTIFICATION_INTERNAL_PROCESS: {
			// get our ARVRServer
			ARVRServer *arvr_server = ARVRServer::get_singleton();
			ERR_FAIL_NULL(arvr_server);

			// find the tracker for our anchor
			ARVRPositionalTracker *tracker = arvr_server->find_by_type_and_id(ARVRServer::TRACKER_ANCHOR, anchor_id);
			if (tracker == NULL) {
				// this anchor is currently not available
				is_active = false;
			} else {
				is_active = true;
				Transform transform;

				// we'll need our world_scale
				real_t world_scale = arvr_server->get_world_scale();

				// get our info from our tracker
				transform.basis = tracker->get_orientation();
				transform.origin = tracker->get_position(); // <-- already adjusted to world scale

				// our basis is scaled to the size of the plane the anchor is tracking
				// extract the size from our basis and reset the scale
				size = transform.basis.get_scale() * world_scale;
				transform.basis.orthonormalize();

				// apply our reference frame and set our transform
				set_transform(arvr_server->get_reference_frame() * transform);
			};
		}; break;
		default:
			break;
	};
};

void ARVRAnchor::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_anchor_id", "anchor_id"), &ARVRAnchor::set_anchor_id);
	ClassDB::bind_method(D_METHOD("get_anchor_id"), &ARVRAnchor::get_anchor_id);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "anchor_id"), "set_anchor_id", "get_anchor_id");
	ClassDB::bind_method(D_METHOD("get_anchor_name"), &ARVRAnchor::get_anchor_name);

	ClassDB::bind_method(D_METHOD("get_is_active"), &ARVRAnchor::get_is_active);
	ClassDB::bind_method(D_METHOD("get_size"), &ARVRAnchor::get_size);
};

void ARVRAnchor::set_anchor_id(int p_anchor_id) {
	// we don't check any bounds here, this anchor may not yet be active and just be a place holder until it is.
	anchor_id = p_anchor_id;
};

int ARVRAnchor::get_anchor_id(void) const {
	return anchor_id;
};

Vector3 ARVRAnchor::get_size() const {
	return size;
};

String ARVRAnchor::get_anchor_name(void) const {
	// get our ARVRServer
	ARVRServer *arvr_server = ARVRServer::get_singleton();
	ERR_FAIL_NULL_V(arvr_server, String());

	ARVRPositionalTracker *tracker = arvr_server->find_by_type_and_id(ARVRServer::TRACKER_ANCHOR, anchor_id);
	if (tracker == NULL) {
		return String("Not connected");
	};

	return tracker->get_name();
};

bool ARVRAnchor::get_is_active() const {
	return is_active;
};

String ARVRAnchor::get_configuration_warning() const {
	if (!is_visible() || !is_inside_tree())
		return String();

	// must be child node of ARVROrigin!
	ARVROrigin *origin = Object::cast_to<ARVROrigin>(get_parent());
	if (origin == NULL) {
		return TTR("ARVRAnchor must have an ARVROrigin node as its parent");
	};

	if (anchor_id == 0) {
		return TTR("The anchor id must not be 0 or this anchor will not be bound to an actual anchor");
	};

	return String();
};

ARVRAnchor::ARVRAnchor() {
	anchor_id = 0;
	is_active = true;
};

ARVRAnchor::~ARVRAnchor(){
	// nothing to do here yet for now..
};

////////////////////////////////////////////////////////////////////////////////////////////////////

String ARVROrigin::get_configuration_warning() const {
	if (!is_visible() || !is_inside_tree())
		return String();

	if (tracked_camera == NULL)
		return TTR("ARVROrigin requires an ARVRCamera child node");

	return String();
};

void ARVROrigin::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_world_scale", "world_scale"), &ARVROrigin::set_world_scale);
	ClassDB::bind_method(D_METHOD("get_world_scale"), &ARVROrigin::get_world_scale);
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "world_scale"), "set_world_scale", "get_world_scale");
};

void ARVROrigin::set_tracked_camera(ARVRCamera *p_tracked_camera) {
	tracked_camera = p_tracked_camera;
};

void ARVROrigin::clear_tracked_camera_if(ARVRCamera *p_tracked_camera) {
	if (tracked_camera == p_tracked_camera) {
		tracked_camera = NULL;
	};
};

float ARVROrigin::get_world_scale() const {
	// get our ARVRServer
	ARVRServer *arvr_server = ARVRServer::get_singleton();
	ERR_FAIL_NULL_V(arvr_server, 1.0);

	return arvr_server->get_world_scale();
};

void ARVROrigin::set_world_scale(float p_world_scale) {
	// get our ARVRServer
	ARVRServer *arvr_server = ARVRServer::get_singleton();
	ERR_FAIL_NULL(arvr_server);

	arvr_server->set_world_scale(p_world_scale);
};

void ARVROrigin::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			set_process_internal(true);
		}; break;
		case NOTIFICATION_EXIT_TREE: {
			set_process_internal(false);
		}; break;
		case NOTIFICATION_INTERNAL_PROCESS: {
			// get our ARVRServer
			ARVRServer *arvr_server = ARVRServer::get_singleton();
			ERR_FAIL_NULL(arvr_server);

			// set our world origin to our node transform
			arvr_server->set_world_origin(get_global_transform());

			// check if we have a primary interface
			Ref<ARVRInterface> arvr_interface = arvr_server->get_primary_interface();
			if (arvr_interface.is_valid() && tracked_camera != NULL) {
				// get our positioning transform for our headset
				Transform t = arvr_interface->get_transform_for_eye(ARVRInterface::EYE_MONO, Transform());

				// now apply this to our camera
				tracked_camera->set_transform(t);
			};
		}; break;
		default:
			break;
	};
};

ARVROrigin::ARVROrigin() {
	tracked_camera = NULL;
};

ARVROrigin::~ARVROrigin(){
	// nothing to do here yet for now..
};
