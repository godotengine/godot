/*************************************************************************/
/*  particles.h                                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
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
#ifndef VISUALINSTANCEPARTICLES_H
#define VISUALINSTANCEPARTICLES_H

#include "rid.h"
#include "scene/3d/visual_instance.h"
#include "scene/main/timer.h"
#include "scene/resources/material.h"

/**
	@author Juan Linietsky <reduzio@gmail.com>
*/
#if 0
class Particles : public GeometryInstance {
public:

	enum Variable {
		VAR_LIFETIME=VS::PARTICLE_LIFETIME,
		VAR_SPREAD=VS::PARTICLE_SPREAD,
		VAR_GRAVITY=VS::PARTICLE_GRAVITY,
		VAR_LINEAR_VELOCITY=VS::PARTICLE_LINEAR_VELOCITY,
		VAR_ANGULAR_VELOCITY=VS::PARTICLE_ANGULAR_VELOCITY,
		VAR_LINEAR_ACCELERATION=VS::PARTICLE_LINEAR_ACCELERATION,
		VAR_DRAG=VS::PARTICLE_RADIAL_ACCELERATION,
		VAR_TANGENTIAL_ACCELERATION=VS::PARTICLE_TANGENTIAL_ACCELERATION,
		VAR_DAMPING=VS::PARTICLE_DAMPING,
		VAR_INITIAL_SIZE=VS::PARTICLE_INITIAL_SIZE,
		VAR_FINAL_SIZE=VS::PARTICLE_FINAL_SIZE,
		VAR_INITIAL_ANGLE=VS::PARTICLE_INITIAL_ANGLE,
		VAR_HEIGHT=VS::PARTICLE_HEIGHT,
		VAR_HEIGHT_SPEED_SCALE=VS::PARTICLE_HEIGHT_SPEED_SCALE,
		VAR_MAX=VS::PARTICLE_VAR_MAX
	};

private:
	GDCLASS( Particles, GeometryInstance );

	RID particles;

	int amount;
	bool emitting;
	float emit_timeout;
	AABB visibility_aabb;
	Vector3 gravity_normal;
	Vector3 emission_half_extents;
	bool using_points;
	float var[VAR_MAX];
	float var_random[VAR_MAX];
	bool height_from_velocity;
	Vector3 emission_base_velocity;
	bool local_coordinates;

	struct ColorPhase {

		Color color;
		float pos;
	};

	virtual bool _can_gizmo_scale() const;
	virtual RES _get_gizmo_geometry() const;

	int color_phase_count;

	ColorPhase color_phase[4];

	Ref<Material> material;

	Timer* timer;
	void setup_timer();

protected:

	static void _bind_methods();

public:


	AABB get_aabb() const;
	PoolVector<Face3> get_faces(uint32_t p_usage_flags) const;

	void set_amount(int p_amount);
	int get_amount() const;

	void set_emitting(bool p_emitting);
	bool is_emitting() const;

	void set_visibility_aabb(const AABB& p_aabb);
	AABB get_visibility_aabb() const;

	void set_emission_half_extents(const Vector3& p_half_extents);
	Vector3 get_emission_half_extents() const;

	void set_emission_base_velocity(const Vector3& p_base_velocity);
	Vector3 get_emission_base_velocity() const;

	void set_emission_points(const PoolVector<Vector3>& p_points);
	PoolVector<Vector3> get_emission_points() const;

	void set_gravity_normal(const Vector3& p_normal);
	Vector3 get_gravity_normal() const;

	void set_variable(Variable p_variable,float p_value);
	float get_variable(Variable p_variable) const;

	void set_randomness(Variable p_variable,float p_randomness);
	float get_randomness(Variable p_variable) const;

	void set_color_phases(int p_phases);
	int get_color_phases() const;

	void set_color_phase_pos(int p_phase, float p_pos);
	float get_color_phase_pos(int p_phase) const;

	void set_color_phase_color(int p_phase, const Color& p_color);
	Color get_color_phase_color(int p_phase) const;

	void set_height_from_velocity(bool p_enable);
	bool has_height_from_velocity() const;

	void set_material(const Ref<Material>& p_material);
	Ref<Material> get_material() const;

	void set_emit_timeout(float p_timeout);
	float get_emit_timeout() const;

	void set_use_local_coordinates(bool p_use);
	bool is_using_local_coordinates() const;

	void start_emitting(float p_time);


	Particles();
	~Particles();

};

VARIANT_ENUM_CAST( Particles::Variable );
#endif
#endif
