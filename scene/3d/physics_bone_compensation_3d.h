/*************************************************************************/
/*  physics_bone_compensation_3d.h                                       */
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

#ifndef PHYSICS_BONE_COMPENSATION_3D_H
#define PHYSICS_BONE_COMPENSATION_3D_H

#include "scene/3d/node_3d.h"

class PhysicsBody3D;
class PhysicsBone3D;

class PhysicsBoneCompensation3D : public Node3D {
	GDCLASS(PhysicsBoneCompensation3D, Node3D);

	PhysicsBody3D *tracking_node = nullptr;
	NodePath tracking_node_path;

	Vector3 translation_delta;

	Basis rotation_delta;
	real_t rotation_delta_angle;

	Transform last_transform;
	Transform transform_delta;

	Vector3 last_angular_velocity;
	Vector3 last_linear_velocity;

	Vector3 angular_velocity_delta;
	Vector3 linear_velocity_delta;

private:
	void update_tracking_node();

protected:
	void _notification(int p_what);
	static void _bind_methods();

	void _get_property_list(List<PropertyInfo> *p_list) const;

public:
	void set_tracking_node_path(const NodePath &p_path);
	const NodePath &get_tracking_node_path() const { return tracking_node_path; }

	const Vector3 &get_last_angular_velocity() const { return last_angular_velocity; }
	const Vector3 &get_last_linear_velocity() const { return last_linear_velocity; }

	const Vector3 &get_angular_velocity_delta() const { return angular_velocity_delta; }
	const Vector3 &get_linear_velocity_delta() const { return linear_velocity_delta; }

	const Transform &get_last_transform() const { return last_transform; }
	const Transform &get_transform_delta() const { return transform_delta; }

	real_t get_rotation_angle_delta() const { return rotation_delta_angle; }

	PhysicsBoneCompensation3D();
	~PhysicsBoneCompensation3D();
};

#endif // PHYSICS_BONE_COMPENSATION_3D_H
