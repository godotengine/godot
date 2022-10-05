/**************************************************************************/
/*  vr_editor_avatar.cpp                                                  */
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

#include "vr_editor_avatar.h"

/* VRPoke */

void VRPoke::_set_ray_visible_length(float p_length) {
	if (ray_mesh.is_valid()) {
		ray_mesh->set_size(Vector3(0.001, 0.001, p_length));
	}
	if (cast) {
		cast->set_position(Vector3(0.0, 0.0, -0.5 * p_length));
	}
}

void VRPoke::_notification(int p_notification) {
	switch (p_notification) {
		case NOTIFICATION_INTERNAL_PROCESS: {
			VRCollision *collision = nullptr;
			Vector3 col_position;
			float col_distance = 0.0;
			bool col_is_pressed[2] = { false, false };
			float col_pressure[2] = { 0.0, 0.0 };
			Vector2 col_scroll;

			if (touch_enabled || ray_enabled) {
				Transform3D poke_transform = get_global_transform();
				Vector3 poke_position = poke_transform.origin;
				Vector3 poke_direction = -poke_transform.basis.get_column(2);
				bool poke_is_pressed[2];
				poke_is_pressed[0] = is_select();
				poke_is_pressed[1] = is_alt_select();

				Vector<VRCollision *> collisions = VRCollision::get_hit_tests(true, false);
				for (int i = 0; i < collisions.size(); i++) {
					VRCollision *test_collision = collisions[i];
					Vector3 test_position;
					float test_distance = 9999.99;
					bool test_is_pressed[2] = { false, false };
					float test_pressure[2] = { 0.0, 0.0 };
					Vector2 test_scroll;

					if (touch_enabled && test_collision->within_sphere(poke_position, radius, test_position)) {
						// only simulate one button when we're touching
						test_distance = (test_position - poke_position).length_squared();
						test_is_pressed[0] = true;
						test_pressure[0] = CLAMP(1.0 - (Math::sqrt(test_distance) / radius), 0.0, 1.0);
					} else if (ray_enabled && test_collision->raycast(poke_position, poke_direction, test_position)) {
						test_distance = (test_position - poke_position).length_squared();
						for (int j = 0; j < 2; j++) {
							test_is_pressed[j] = poke_is_pressed[j];
							test_pressure[j] = poke_is_pressed[j] ? 1.0 : 0.0;
						}
						test_scroll = get_scroll();
					} else {
						test_collision = nullptr;
					}

					// if we have a collision, check if we're closer
					if (test_collision && (!collision || (test_distance < col_distance))) {
						collision = test_collision;
						col_position = test_position;
						col_distance = test_distance;
						col_scroll = test_scroll;
						for (int j = 0; j < 2; j++) {
							col_is_pressed[j] = test_is_pressed[j];
							col_pressure[j] = test_pressure[j];
						}
					}
				}
			}

			// Last collision no longer relevant?
			if (last_collision && last_collision != collision) {
				// Might want to leave first and forego on the releases?
				if (last_was_pressed[1]) {
					last_collision->_on_interact_released(last_position, MouseButton::RIGHT);
					last_was_pressed[1] = false;
				}
				if (last_was_pressed[0]) {
					last_collision->_on_interact_released(last_position, MouseButton::LEFT);
					last_was_pressed[0] = false;
				}
				last_collision->_on_interact_leave(last_position);

				// reset
				last_collision = nullptr;
			}

			// Hit something?
			if (collision) {
				_set_ray_visible_length(Math::sqrt(col_distance));
				material->set_albedo(touch_color);

				if (last_collision != collision) {
					last_collision = collision;
					last_position = col_position; // We don't want to trigger a move event here.
					collision->_on_interact_enter(col_position);
				}

				for (int i = 0; i < 2; i++) {
					if (col_is_pressed[i] && !last_was_pressed[i]) {
						collision->_on_interact_pressed(col_position, MouseButton(i + 1));
					} else if (!col_is_pressed[i] && last_was_pressed[i]) {
						collision->_on_interact_released(col_position, MouseButton(i + 1));
					}
				}

				if (last_position != col_position) {
					collision->_on_interact_moved(col_position, col_pressure[0]);
				}

				last_position = col_position;
				for (int i = 0; i < 2; i++) {
					last_was_pressed[i] = col_is_pressed[i];
				}

				// Forward scroll values
				collision->_on_interact_scrolled(col_position, col_scroll);
			} else {
				material->set_albedo(normal_color);
				_set_ray_visible_length(5.0);
			}
		} break;
		default: {
			// ignore
		} break;
	}
}

void VRPoke::set_touch_enabled(bool p_enabled) {
	touch_enabled = p_enabled;

	if (sphere) {
		sphere->set_visible(touch_enabled);
	}
}

void VRPoke::set_ray_enabled(bool p_enabled) {
	ray_enabled = p_enabled;

	if (cast) {
		cast->set_visible(ray_enabled);
	}
}

Vector2 VRPoke::get_scroll() {
	XRController3D *controller = Object::cast_to<XRController3D>(get_parent());
	ERR_FAIL_NULL_V(controller, Vector2());

	return controller->get_vector2("scroll");
}

bool VRPoke::is_select() {
	// Returns true if our select action is triggered on the controller we're a child off.

	XRController3D *controller = Object::cast_to<XRController3D>(get_parent());
	ERR_FAIL_NULL_V(controller, false);

	return controller->is_button_pressed("select");
}

bool VRPoke::is_alt_select() {
	// Returns true if our alternative select action is triggered on the controller we're a child off.

	XRController3D *controller = Object::cast_to<XRController3D>(get_parent());
	ERR_FAIL_NULL_V(controller, false);

	return controller->is_button_pressed("alt_select");
}

VRPoke::VRPoke() {
	material.instantiate();
	material->set_shading_mode(BaseMaterial3D::SHADING_MODE_UNSHADED);
	material->set_transparency(BaseMaterial3D::TRANSPARENCY_ALPHA);
	material->set_albedo(normal_color);

	Ref<SphereMesh> sphere_mesh;
	sphere_mesh.instantiate();
	sphere_mesh->set_radius(radius);
	sphere_mesh->set_height(radius * 2.0);
	sphere_mesh->set_radial_segments(16);
	sphere_mesh->set_rings(8);
	sphere_mesh->set_material(material);

	sphere = memnew(MeshInstance3D);
	sphere->set_mesh(sphere_mesh);
	sphere->set_visible(touch_enabled);
	add_child(sphere);

	ray_mesh.instantiate();
	ray_mesh->set_material(material);

	cast = memnew(MeshInstance3D);
	cast->set_mesh(ray_mesh);
	cast->set_visible(ray_enabled);
	_set_ray_visible_length(5.0);
	add_child(cast);

	set_process_internal(true);
}

VRPoke::~VRPoke() {
}

/* VRGrabDetect */

void VRGrabDetect::set_radius(float p_radius) {
	radius = p_radius;
}

void VRGrabDetect::_on_grab_pressed() {
	// Find any object within range and grab the closest one
}

void VRGrabDetect::_on_grab_released() {
	// If we are holding something, release it
}

VRGrabDetect::VRGrabDetect() {
}

VRGrabDetect::~VRGrabDetect() {
}

/* VRHand */

void VRHand::_notification(int p_notification) {
	switch (p_notification) {
		case NOTIFICATION_INTERNAL_PROCESS: {
			// Controllers that have a pressure sensor can have unreliable thresholds so we hardcode this.
			// Controllers that don't have a pressure sensor simply return 0.0 or 1.0
			float grab_value = get_float("grab");
			if (grab_value < 0.35 && grab_pressed) {
				grab_pressed = false;

				// maybe replace this with signals? or overkill?
				if (grab_detect) {
					grab_detect->_on_grab_released();
				}
			} else if (grab_value > 0.65 && !grab_pressed) {
				grab_pressed = true;

				// maybe replace this with signals? or overkill?
				if (grab_detect) {
					grab_detect->_on_grab_pressed();
				}
			}
		} break;
		default: {
			// ignore
		} break;
	}
}

VRHand::VRHand(Hands p_hand) {
	// We need to make something nicer for our hands
	// maybe use hand tracking if available and see if we can support all fingers.
	// But for now just fingers on our default position is fine.

	set_name(p_hand == HAND_LEFT ? "left_hand" : "right_hand");
	set_tracker(p_hand == HAND_LEFT ? "left_hand" : "right_hand");
	set_pose_name(SNAME("tool_pose"));

	poke = memnew(VRPoke);
	poke->set_position(Vector3(0.0, 0.0, -0.01));
	add_child(poke);

	grab_detect = memnew(VRGrabDetect);
	add_child(grab_detect);

	set_process_internal(true);
}

VRHand::~VRHand() {
}

/* VREditorAvatar */

void VREditorAvatar::set_ray_active_on_hand(VRHand::Hands p_hand) {
	ray_active_on_hand = p_hand;
	left_hand->set_ray_enabled(p_hand == VRHand::HAND_LEFT);
	right_hand->set_ray_enabled(p_hand == VRHand::HAND_RIGHT);
}

void VREditorAvatar::_on_button_pressed_on_hand(const String p_action, int p_hand) {
	if (p_action == "select") {
		set_ray_active_on_hand(VRHand::Hands(p_hand));
	}
}

void VREditorAvatar::set_hud_offset(real_t p_offset) {
	hud_offset = p_offset;

	Vector3 position = hud_pivot->get_position();
	position.y = hud_offset;
	hud_pivot->set_position(position);
}

void VREditorAvatar::set_hud_distance(real_t p_distance) {
	hud_distance = p_distance;

	Vector3 position = hud_root->get_position();
	position.z = -hud_distance;
	hud_root->set_position(position);
}

void VREditorAvatar::set_camera_cull_layers(uint32_t p_layers) {
	ERR_FAIL_NULL(camera);

	camera->set_cull_mask(p_layers);
}

VREditorAvatar::VREditorAvatar() {
	// TODO once https://github.com/godotengine/godot/pull/63607 is merged we need to add an enhancement
	// to make this node the "current" XROrigin3D node.
	// For now this will be the current node but if a VR project is loaded things could go haywire.

	camera = memnew(XRCamera3D);
	camera->set_name("camera");
	add_child(camera);

	// Our hud pivot will follow our camera around at a constant height.
	// TODO add a button press or other mechanism to rotate our hud pivot
	// so our hud is recentered infront of our player.
	hud_pivot = memnew(Node3D);
	hud_pivot->set_name("hud_pivot");
	hud_pivot->set_position(Vector3(0.0, 1.6 + hud_offset, 0.0)); // we don't know our eye height yet
	add_child(hud_pivot);

	// Our hud root extends our hud outwards to a certain distance away
	// from our player.
	hud_root = memnew(Node3D);
	hud_root->set_name("hud_root");
	hud_root->set_position(Vector3(0.0, 0.0, -hud_distance));
	hud_pivot->add_child(hud_root);

	// Add our hands
	left_hand = memnew(VRHand(VRHand::HAND_LEFT));
	left_hand->connect("button_pressed", callable_mp(this, &VREditorAvatar::_on_button_pressed_on_hand).bind(int(VRHand::HAND_LEFT)));
	add_child(left_hand);

	right_hand = memnew(VRHand(VRHand::HAND_RIGHT));
	right_hand->connect("button_pressed", callable_mp(this, &VREditorAvatar::_on_button_pressed_on_hand).bind(int(VRHand::HAND_RIGHT)));
	add_child(right_hand);

	// TODO add callback for select so we can activate ray on last used hand

	set_ray_active_on_hand(ray_active_on_hand);

	// Add virtual keyboard
	keyboard = memnew(VRKeyboard);
	keyboard->set_name("VRKeyboard");
	keyboard->set_rotation(Vector3(20.0 * Math_PI / 180.0, 0.0, 0.0)); // put at a slight angle for comfort
	keyboard->set_position(Vector3(0.0, -0.4, 0.2)); // should make this a setting or something we can change
	hud_root->add_child(keyboard);

	set_process(true);

	// Our default transform logic in XROrigin3D is disabled in editor mode,
	// this should be our only active XROrigin3D node in our VR editor
	set_notify_local_transform(true);
	set_notify_transform(true);
}

VREditorAvatar::~VREditorAvatar() {
}

void VREditorAvatar::_notification(int p_notification) {
	switch (p_notification) {
		case NOTIFICATION_PROCESS: {
			double delta = get_process_delta_time();

			XRPose::TrackingConfidence confidence = camera->get_tracking_confidence();
			if (confidence == XRPose::XR_TRACKING_CONFIDENCE_NONE) {
				if (camera_is_tracking) {
					// We are not tracking so keep things where they are, user is likely not wearing the headset
					camera_is_tracking = false;

					print_line("HMD is no longer tracking");
				}
			} else {
				// Center our hud on our camera, start by calculating our desired location
				Vector3 desired_location = camera->get_position();
				desired_location.y = MAX(0.5, desired_location.y + hud_offset);

				if (!camera_is_tracking) {
					// If we weren't tracking, reposition our HUD right away, user likely just put on their headset
					camera_is_tracking = true;
					print_line("HMD is now tracking");

					hud_pivot->set_position(desired_location);
				} else {
					// If we were tracking, have HUD follow head movement, this prevents motion sickness

					// Now update our transform.
					bool update_location = false;
					Vector3 hud_location = hud_pivot->get_position();

					// If our desired location is more then 10cm away,
					// we start moving until our hud is within 1 cm of
					// the desired location.
					if ((desired_location - hud_location).length() > (hud_moving ? 0.01 : 0.2)) {
						hud_location = hud_location.lerp(desired_location, delta);
						hud_moving = true;
						update_location = true;
					} else {
						hud_moving = false;
					}

					if (update_location) {
						hud_pivot->set_position(hud_location);
					}
				}
			}
		} break;
		case NOTIFICATION_LOCAL_TRANSFORM_CHANGED:
		case NOTIFICATION_TRANSFORM_CHANGED: {
			XRServer::get_singleton()->set_world_origin(get_global_transform());
		} break;
	}
}
