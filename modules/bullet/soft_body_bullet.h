/*************************************************************************/
/*  soft_body_bullet.h                                                   */
/*  Author: AndreaCatania                                                */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
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

#ifndef SOFT_BODY_BULLET_H
#define SOFT_BODY_BULLET_H

#ifdef None
/// This is required to remove the macro None defined by x11 compiler because this word "None" is used internally by Bullet
#undef None
#define x11_None 0L
#endif

#include "BulletSoftBody/btSoftBodyHelpers.h"
#include "collision_object_bullet.h"

#ifdef x11_None
/// This is required to re add the macro None defined by x11 compiler
#undef x11_None
#define None 0L
#endif

#include "scene/resources/material.h" // TODO remove thsi please

struct SoftShapeData {};
struct TrimeshSoftShapeData : public SoftShapeData {
	PoolVector<int> m_triangles_indices;
	PoolVector<Vector3> m_vertices;
	int m_triangles_num;
};

class SoftBodyBullet : public CollisionObjectBullet {
public:
	enum SoftShapeType {
		SOFT_SHAPETYPE_NONE = 0,
		SOFT_SHAPE_TYPE_TRIMESH
	};

private:
	btSoftBody *bt_soft_body;
	btSoftBody::Material *mat0; // This is just a copy of pointer managed by btSoftBody
	SoftShapeType soft_shape_type;
	bool isScratched;

	SoftShapeData *soft_body_shape_data;

	Transform transform;
	int simulation_precision;
	real_t mass;
	real_t stiffness; // [0,1]
	real_t pressure_coefficient; // [-inf,+inf]
	real_t damping_coefficient; // [0,1]
	real_t drag_coefficient; // [0,1]

	class ImmediateGeometry *test_geometry; // TODO remove this please
	Ref<SpatialMaterial> red_mat; // TODO remove this please
	bool test_is_in_scene; // TODO remove this please

public:
	SoftBodyBullet();
	~SoftBodyBullet();

	virtual void reload_body();
	virtual void set_space(SpaceBullet *p_space);

	virtual void dispatch_callbacks();
	virtual void on_collision_filters_change();
	virtual void on_collision_checker_start();
	virtual void on_enter_area(AreaBullet *p_area);
	virtual void on_exit_area(AreaBullet *p_area);

	_FORCE_INLINE_ btSoftBody *get_bt_soft_body() const { return bt_soft_body; }

	void set_trimesh_body_shape(PoolVector<int> p_indices, PoolVector<Vector3> p_vertices, int p_triangles_num);
	void set_body_shape_data(SoftShapeData *p_soft_shape_data, SoftShapeType p_type);

	void set_transform(const Transform &p_transform);
	/// This function doesn't return the exact COM transform.
	/// It returns the origin only of first node (vertice) of current soft body
	/// ---
	/// The soft body doesn't have a fixed center of mass, but is a group of nodes (vertices)
	/// that each has its own position in the world.
	/// For this reason return the correct COM is not so simple and must be calculate
	/// Check this to improve this function http://bulletphysics.org/Bullet/phpBB3/viewtopic.php?t=8803
	const Transform &get_transform() const;
	void get_first_node_origin(btVector3 &p_out_origin) const;

	void set_activation_state(bool p_active);

	void set_mass(real_t p_val);
	_FORCE_INLINE_ real_t get_mass() const { return mass; }
	void set_stiffness(real_t p_val);
	_FORCE_INLINE_ real_t get_stiffness() const { return stiffness; }
	void set_simulation_precision(int p_val);
	_FORCE_INLINE_ int get_simulation_precision() const { return simulation_precision; }
	void set_pressure_coefficient(real_t p_val);
	_FORCE_INLINE_ real_t get_pressure_coefficient() const { return pressure_coefficient; }
	void set_damping_coefficient(real_t p_val);
	_FORCE_INLINE_ real_t get_damping_coefficient() const { return damping_coefficient; }
	void set_drag_coefficient(real_t p_val);
	_FORCE_INLINE_ real_t get_drag_coefficient() const { return drag_coefficient; }

private:
	void reload_soft_body();
	void create_soft_body();
	void destroy_soft_body();
};

#endif // SOFT_BODY_BULLET_H
