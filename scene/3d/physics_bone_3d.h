/*************************************************************************/
/*  physics_bone_3d.h                                                    */
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

#ifndef PHYSICS_BONE_3D_H
#define PHYSICS_BONE_3D_H

#include "scene/3d/physics_body_3d.h"
#include "scene/3d/skeleton_3d.h"

class PhysicsBoneCompensation3D;

/**
 *	@author Marios Staikopoulos <marios@staik.net>
 */

class PhysicsBone3D : public RigidBody3D {
	GDCLASS(PhysicsBone3D, RigidBody3D);

private:
	bool free_simulation = false;
	String bone_name;

	Skeleton3D *skeleton = nullptr;
	BoneId bone_id = -1;
	Vector3 bone_offset;

	PhysicsBoneCompensation3D *compensation_node = nullptr;
	real_t velocity_comp_amount = 1.0;
	real_t rotation_comp_amount = 1.0;
	real_t rotation_teleport_threshold_angle = 1.74533; // 100 degrees

	bool has_teleported = false;

private:
	PhysicsBoneCompensation3D *find_compensation_parent(Node *p_parent);
	PhysicsBoneCompensation3D *find_compensation_parent();

	Skeleton3D *find_skeleton_parent(Node *p_parent);
	Skeleton3D *find_skeleton_parent();

	void update_compensation_node(PhysicsBoneCompensation3D *p_comp_node);
	void update_skeleton_node(Skeleton3D *p_skeleton);
	void update_bone_id();
	void update_body_type();

	void apply_compensation(PhysicsDirectBodyState3D *p_state);

protected:
	void _get_property_list(List<PropertyInfo> *p_list) const;
	void _notification(int p_what);

	static void _bind_methods();

	virtual void _direct_state_changed(Object *p_state) override;

public:
	BoneId get_bone_id() const { return bone_id; }

	void set_bone_name(const String &p_name);
	String get_bone_name() const { return bone_name; }

	void set_velocity_compensation_amount(real_t p_amount) { velocity_comp_amount = p_amount; }
	real_t get_velocity_compensation_amount() const { return velocity_comp_amount; }

	void set_rotation_compensation_amount(real_t p_amount) { rotation_comp_amount = p_amount; }
	real_t get_rotation_compensation_amount() const { return rotation_comp_amount; }

	void set_rotation_teleport_threshold_angle(real_t p_angle) { rotation_teleport_threshold_angle = p_angle; }
	real_t get_rotation_teleport_threshold_angle() const { return rotation_teleport_threshold_angle; }

	void set_free_simulation(bool p_free_sim);
	bool get_free_simulation() const { return free_simulation; }

	void set_bone_offset(const Vector3 &p_offset);
	const Vector3 &get_bone_offset() const { return bone_offset; }

	// Gets the *PhysicsBodys* global rest position - not to be confused with bone global position
	Transform get_p_body_global_rest() const;

	const Skeleton3D *get_skeleton() const { return skeleton; }

	PhysicsBone3D();
	~PhysicsBone3D();
};

#endif // PHYSICS_BONE_3D_H
