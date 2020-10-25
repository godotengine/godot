/*************************************************************************/
/*  character_body_3d.h                                                  */
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

#ifndef CHARACTER_BODY_3D_H
#define CHARACTER_BODY_3D_H

#include "rigid_body_3d.h"

class CharacterBody3D : public RigidBody3D {
	GDCLASS(CharacterBody3D, RigidBody3D);

	real_t strafe_speed = 10;
	real_t strafe_lerp_rate = 1;
	real_t jump_speed = 10;

protected:
	static void _bind_methods();

public:
	void set_strafe_speed(real_t p_speed);
	real_t get_strafe_speed() const;

	void set_strafe_lerp_rate(real_t p_lerp_rate);
	real_t get_strafe_lerp_rate() const;

	void set_jump_speed(real_t p_speed);
	real_t get_jump_speed() const;

	void move_to(Vector3 p_position, bool p_all_or_nothing);
	void strafe(Vector3 p_direction);
	void jump();

	CharacterBody3D();
};

#endif // CHARACTER_BODY_3D_H
