/*************************************************************************/
/*  character_body_3d.cpp                                                */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "character_body_3d.h"

void CharacterBody3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_strafe_speed", "p_speed"), &CharacterBody3D::set_strafe_speed);
	ClassDB::bind_method(D_METHOD("get_strafe_speed"), &CharacterBody3D::get_strafe_speed);
	ClassDB::bind_method(D_METHOD("set_strafe_lerp_rate", "p_lerp_rate"), &CharacterBody3D::set_strafe_lerp_rate);
	ClassDB::bind_method(D_METHOD("get_strafe_lerp_rate"), &CharacterBody3D::get_strafe_lerp_rate);
	ClassDB::bind_method(D_METHOD("set_jump_speed", "p_speed"), &CharacterBody3D::set_jump_speed);
	ClassDB::bind_method(D_METHOD("get_jump_speed"), &CharacterBody3D::get_jump_speed);
	ClassDB::bind_method(D_METHOD("move_to", "p_position", "p_all_or_nothing"), &CharacterBody3D::move_to);
	ClassDB::bind_method(D_METHOD("strafe", "p_direction"), &CharacterBody3D::strafe);
	ClassDB::bind_method(D_METHOD("jump"), &CharacterBody3D::jump);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "strafe_speed", PROPERTY_HINT_EXP_RANGE, "0,100,0.1"), "set_strafe_speed", "get_strafe_speed");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "strafe_lerp_rate", PROPERTY_HINT_EXP_RANGE, "0,1,0.01"), "set_strafe_lerp_rate", "get_strafe_lerp_rate");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "jump_speed", PROPERTY_HINT_EXP_RANGE, "0,100,0.1"), "set_jump_speed", "get_jump_speed");
}

void CharacterBody3D::set_strafe_speed(real_t p_speed) {
	strafe_speed = p_speed;
}

real_t CharacterBody3D::get_strafe_speed() const {
	return strafe_speed;
}

void CharacterBody3D::set_strafe_lerp_rate(real_t p_lerp_rate) {
	strafe_lerp_rate = p_lerp_rate;
}

real_t CharacterBody3D::get_strafe_lerp_rate() const {
	return strafe_lerp_rate;
}

void CharacterBody3D::set_jump_speed(real_t p_speed) {
	jump_speed = p_speed;
}

real_t CharacterBody3D::get_jump_speed() const {
	return jump_speed;
}

void CharacterBody3D::move_to(Vector3 p_position, bool p_all_or_nothing) {
	Vector3 motion = p_position - get_global_transform().get_origin();
	Collision col;
	if (p_all_or_nothing && move_and_collide(motion, true, col, false, true)) {
		return;
	}
	move_and_collide(motion, true, col, false, false);
}

void CharacterBody3D::strafe(Vector3 p_direction) {
	Vector3 direction = p_direction.normalized();
	Vector3 current_velocity = get_linear_velocity();
	Vector3 target_velocity = direction * strafe_speed;
	target_velocity.x = Math::lerp(current_velocity.x, target_velocity.x, strafe_lerp_rate);
	target_velocity.y = Math::lerp(current_velocity.y, target_velocity.y, strafe_lerp_rate);
	target_velocity.z = Math::lerp(current_velocity.z, target_velocity.z, strafe_lerp_rate);
	set_linear_velocity(target_velocity);
}

void CharacterBody3D::jump() {
	Vector3 velocity = get_linear_velocity();
	velocity.y = jump_speed;
	set_linear_velocity(velocity);
}

CharacterBody3D::CharacterBody3D() {
	set_mode(MODE_CHARACTER);
}
