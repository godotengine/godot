/**************************************************************************/
/*  soft_body_bullet.h                                                    */
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

#ifndef SOFT_BODY_BULLET_H
#define SOFT_BODY_BULLET_H

#include "collision_object_bullet.h"
#include "scene/resources/material.h" // TODO remove this please

#ifdef None
/// This is required to remove the macro None defined by x11 compiler because this word "None" is used internally by Bullet
#undef None
#define x11_None 0L
#endif

#include "BulletSoftBody/btSoftBodyHelpers.h"
#include "collision_object_bullet.h"
#include "scene/resources/mesh.h"
#include "servers/physics_server.h"

#ifdef x11_None
/// This is required to re add the macro None defined by x11 compiler
#undef x11_None
#define None 0L
#endif

/**
	@author AndreaCatania
*/

class SoftBodyBullet : public CollisionObjectBullet {
private:
	btSoftBody *bt_soft_body;
	Vector<Vector<int>> indices_table;
	btSoftBody::Material *mat0; // This is just a copy of pointer managed by btSoftBody
	bool isScratched;

	Ref<Mesh> soft_mesh;

	int simulation_precision;
	real_t total_mass;
	real_t linear_stiffness; // [0,1]
	real_t areaAngular_stiffness; // [0,1]
	real_t volume_stiffness; // [0,1]
	real_t pressure_coefficient; // [-inf,+inf]
	real_t pose_matching_coefficient; // [0,1]
	real_t damping_coefficient; // [0,1]
	real_t drag_coefficient; // [0,1]
	Vector<int> pinned_nodes;

	// Other property to add
	//btScalar				kVC;			// Volume conversation coefficient [0,+inf]
	//btScalar				kDF;			// Dynamic friction coefficient [0,1]
	//btScalar				kMT;			// Pose matching coefficient [0,1]
	//btScalar				kCHR;			// Rigid contacts hardness [0,1]
	//btScalar				kKHR;			// Kinetic contacts hardness [0,1]
	//btScalar				kSHR;			// Soft contacts hardness [0,1]

public:
	SoftBodyBullet();
	~SoftBodyBullet();

	virtual void reload_body();
	virtual void set_space(SpaceBullet *p_space);

	virtual void dispatch_callbacks() {}
	virtual void on_collision_filters_change() {}
	virtual void on_collision_checker_start() {}
	virtual void on_collision_checker_end() {}
	virtual void on_enter_area(AreaBullet *p_area);
	virtual void on_exit_area(AreaBullet *p_area);

	_FORCE_INLINE_ btSoftBody *get_bt_soft_body() const { return bt_soft_body; }

	void update_visual_server(class SoftBodyVisualServerHandler *p_visual_server_handler);

	void set_soft_mesh(const Ref<Mesh> &p_mesh);
	void destroy_soft_body();

	// Special function. This function has bad performance
	void set_soft_transform(const Transform &p_transform);

	void move_all_nodes(const Transform &p_transform);
	void set_node_position(int node_index, const Vector3 &p_global_position);
	void set_node_position(int node_index, const btVector3 &p_global_position);
	void get_node_position(int node_index, Vector3 &r_position) const;
	// Heavy function, Please cache this info
	void get_node_offset(int node_index, Vector3 &r_offset) const;
	// Heavy function, Please cache this info
	void get_node_offset(int node_index, btVector3 &r_offset) const;

	void set_node_mass(int node_index, btScalar p_mass);
	btScalar get_node_mass(int node_index) const;
	void reset_all_node_mass();
	void reset_all_node_positions();

	void set_activation_state(bool p_active);

	void set_total_mass(real_t p_val);
	_FORCE_INLINE_ real_t get_total_mass() const { return total_mass; }

	void set_linear_stiffness(real_t p_val);
	_FORCE_INLINE_ real_t get_linear_stiffness() const { return linear_stiffness; }

	void set_areaAngular_stiffness(real_t p_val);
	_FORCE_INLINE_ real_t get_areaAngular_stiffness() const { return areaAngular_stiffness; }

	void set_volume_stiffness(real_t p_val);
	_FORCE_INLINE_ real_t get_volume_stiffness() const { return volume_stiffness; }

	void set_simulation_precision(int p_val);
	_FORCE_INLINE_ int get_simulation_precision() const { return simulation_precision; }

	void set_pressure_coefficient(real_t p_val);
	_FORCE_INLINE_ real_t get_pressure_coefficient() const { return pressure_coefficient; }

	void set_pose_matching_coefficient(real_t p_val);
	_FORCE_INLINE_ real_t get_pose_matching_coefficient() const { return pose_matching_coefficient; }

	void set_damping_coefficient(real_t p_val);
	_FORCE_INLINE_ real_t get_damping_coefficient() const { return damping_coefficient; }

	void set_drag_coefficient(real_t p_val);
	_FORCE_INLINE_ real_t get_drag_coefficient() const { return drag_coefficient; }

private:
	void set_trimesh_body_shape(PoolVector<int> p_indices, PoolVector<Vector3> p_vertices);
	void setup_soft_body();

	void pin_node(int p_node_index);
	void unpin_node(int p_node_index);
	int search_node_pinned(int p_node_index) const;
};

#endif // SOFT_BODY_BULLET_H
