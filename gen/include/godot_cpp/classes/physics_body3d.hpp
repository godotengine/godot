/**************************************************************************/
/*  physics_body3d.hpp                                                    */
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

// THIS FILE IS GENERATED. EDITS WILL BE LOST.

#pragma once

#include <godot_cpp/classes/collision_object3d.hpp>
#include <godot_cpp/classes/kinematic_collision3d.hpp>
#include <godot_cpp/classes/physics_server3d.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/variant/typed_array.hpp>
#include <godot_cpp/variant/vector3.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Node;
struct Transform3D;

class PhysicsBody3D : public CollisionObject3D {
	GDEXTENSION_CLASS(PhysicsBody3D, CollisionObject3D)

public:
	Ref<KinematicCollision3D> move_and_collide(const Vector3 &p_motion, bool p_test_only = false, float p_safe_margin = 0.001, bool p_recovery_as_collision = false, int32_t p_max_collisions = 1);
	bool test_move(const Transform3D &p_from, const Vector3 &p_motion, const Ref<KinematicCollision3D> &p_collision = nullptr, float p_safe_margin = 0.001, bool p_recovery_as_collision = false, int32_t p_max_collisions = 1);
	Vector3 get_gravity() const;
	void set_axis_lock(PhysicsServer3D::BodyAxis p_axis, bool p_lock);
	bool get_axis_lock(PhysicsServer3D::BodyAxis p_axis) const;
	TypedArray<PhysicsBody3D> get_collision_exceptions();
	void add_collision_exception_with(Node *p_body);
	void remove_collision_exception_with(Node *p_body);

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		CollisionObject3D::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

