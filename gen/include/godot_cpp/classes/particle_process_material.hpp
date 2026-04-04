/**************************************************************************/
/*  particle_process_material.hpp                                         */
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

#include <godot_cpp/classes/material.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/variant/color.hpp>
#include <godot_cpp/variant/vector2.hpp>
#include <godot_cpp/variant/vector3.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Texture2D;

class ParticleProcessMaterial : public Material {
	GDEXTENSION_CLASS(ParticleProcessMaterial, Material)

public:
	enum Parameter {
		PARAM_INITIAL_LINEAR_VELOCITY = 0,
		PARAM_ANGULAR_VELOCITY = 1,
		PARAM_ORBIT_VELOCITY = 2,
		PARAM_LINEAR_ACCEL = 3,
		PARAM_RADIAL_ACCEL = 4,
		PARAM_TANGENTIAL_ACCEL = 5,
		PARAM_DAMPING = 6,
		PARAM_ANGLE = 7,
		PARAM_SCALE = 8,
		PARAM_HUE_VARIATION = 9,
		PARAM_ANIM_SPEED = 10,
		PARAM_ANIM_OFFSET = 11,
		PARAM_RADIAL_VELOCITY = 15,
		PARAM_DIRECTIONAL_VELOCITY = 16,
		PARAM_SCALE_OVER_VELOCITY = 17,
		PARAM_MAX = 18,
		PARAM_TURB_VEL_INFLUENCE = 13,
		PARAM_TURB_INIT_DISPLACEMENT = 14,
		PARAM_TURB_INFLUENCE_OVER_LIFE = 12,
	};

	enum ParticleFlags {
		PARTICLE_FLAG_ALIGN_Y_TO_VELOCITY = 0,
		PARTICLE_FLAG_ROTATE_Y = 1,
		PARTICLE_FLAG_DISABLE_Z = 2,
		PARTICLE_FLAG_DAMPING_AS_FRICTION = 3,
		PARTICLE_FLAG_MAX = 4,
	};

	enum EmissionShape {
		EMISSION_SHAPE_POINT = 0,
		EMISSION_SHAPE_SPHERE = 1,
		EMISSION_SHAPE_SPHERE_SURFACE = 2,
		EMISSION_SHAPE_BOX = 3,
		EMISSION_SHAPE_POINTS = 4,
		EMISSION_SHAPE_DIRECTED_POINTS = 5,
		EMISSION_SHAPE_RING = 6,
		EMISSION_SHAPE_MAX = 7,
	};

	enum SubEmitterMode {
		SUB_EMITTER_DISABLED = 0,
		SUB_EMITTER_CONSTANT = 1,
		SUB_EMITTER_AT_END = 2,
		SUB_EMITTER_AT_COLLISION = 3,
		SUB_EMITTER_AT_START = 4,
		SUB_EMITTER_MAX = 5,
	};

	enum CollisionMode {
		COLLISION_DISABLED = 0,
		COLLISION_RIGID = 1,
		COLLISION_HIDE_ON_CONTACT = 2,
		COLLISION_MAX = 3,
	};

	void set_direction(const Vector3 &p_degrees);
	Vector3 get_direction() const;
	void set_inherit_velocity_ratio(double p_ratio);
	double get_inherit_velocity_ratio();
	void set_spread(float p_degrees);
	float get_spread() const;
	void set_flatness(float p_amount);
	float get_flatness() const;
	void set_param(ParticleProcessMaterial::Parameter p_param, const Vector2 &p_value);
	Vector2 get_param(ParticleProcessMaterial::Parameter p_param) const;
	void set_param_min(ParticleProcessMaterial::Parameter p_param, float p_value);
	float get_param_min(ParticleProcessMaterial::Parameter p_param) const;
	void set_param_max(ParticleProcessMaterial::Parameter p_param, float p_value);
	float get_param_max(ParticleProcessMaterial::Parameter p_param) const;
	void set_param_texture(ParticleProcessMaterial::Parameter p_param, const Ref<Texture2D> &p_texture);
	Ref<Texture2D> get_param_texture(ParticleProcessMaterial::Parameter p_param) const;
	void set_color(const Color &p_color);
	Color get_color() const;
	void set_color_ramp(const Ref<Texture2D> &p_ramp);
	Ref<Texture2D> get_color_ramp() const;
	void set_alpha_curve(const Ref<Texture2D> &p_curve);
	Ref<Texture2D> get_alpha_curve() const;
	void set_emission_curve(const Ref<Texture2D> &p_curve);
	Ref<Texture2D> get_emission_curve() const;
	void set_color_initial_ramp(const Ref<Texture2D> &p_ramp);
	Ref<Texture2D> get_color_initial_ramp() const;
	void set_velocity_limit_curve(const Ref<Texture2D> &p_curve);
	Ref<Texture2D> get_velocity_limit_curve() const;
	void set_particle_flag(ParticleProcessMaterial::ParticleFlags p_particle_flag, bool p_enable);
	bool get_particle_flag(ParticleProcessMaterial::ParticleFlags p_particle_flag) const;
	void set_velocity_pivot(const Vector3 &p_pivot);
	Vector3 get_velocity_pivot();
	void set_emission_shape(ParticleProcessMaterial::EmissionShape p_shape);
	ParticleProcessMaterial::EmissionShape get_emission_shape() const;
	void set_emission_sphere_radius(float p_radius);
	float get_emission_sphere_radius() const;
	void set_emission_box_extents(const Vector3 &p_extents);
	Vector3 get_emission_box_extents() const;
	void set_emission_point_texture(const Ref<Texture2D> &p_texture);
	Ref<Texture2D> get_emission_point_texture() const;
	void set_emission_normal_texture(const Ref<Texture2D> &p_texture);
	Ref<Texture2D> get_emission_normal_texture() const;
	void set_emission_color_texture(const Ref<Texture2D> &p_texture);
	Ref<Texture2D> get_emission_color_texture() const;
	void set_emission_point_count(int32_t p_point_count);
	int32_t get_emission_point_count() const;
	void set_emission_ring_axis(const Vector3 &p_axis);
	Vector3 get_emission_ring_axis() const;
	void set_emission_ring_height(float p_height);
	float get_emission_ring_height() const;
	void set_emission_ring_radius(float p_radius);
	float get_emission_ring_radius() const;
	void set_emission_ring_inner_radius(float p_inner_radius);
	float get_emission_ring_inner_radius() const;
	void set_emission_ring_cone_angle(float p_cone_angle);
	float get_emission_ring_cone_angle() const;
	void set_emission_shape_offset(const Vector3 &p_emission_shape_offset);
	Vector3 get_emission_shape_offset() const;
	void set_emission_shape_scale(const Vector3 &p_emission_shape_scale);
	Vector3 get_emission_shape_scale() const;
	bool get_turbulence_enabled() const;
	void set_turbulence_enabled(bool p_turbulence_enabled);
	float get_turbulence_noise_strength() const;
	void set_turbulence_noise_strength(float p_turbulence_noise_strength);
	float get_turbulence_noise_scale() const;
	void set_turbulence_noise_scale(float p_turbulence_noise_scale);
	float get_turbulence_noise_speed_random() const;
	void set_turbulence_noise_speed_random(float p_turbulence_noise_speed_random);
	Vector3 get_turbulence_noise_speed() const;
	void set_turbulence_noise_speed(const Vector3 &p_turbulence_noise_speed);
	Vector3 get_gravity() const;
	void set_gravity(const Vector3 &p_accel_vec);
	void set_lifetime_randomness(double p_randomness);
	double get_lifetime_randomness() const;
	ParticleProcessMaterial::SubEmitterMode get_sub_emitter_mode() const;
	void set_sub_emitter_mode(ParticleProcessMaterial::SubEmitterMode p_mode);
	double get_sub_emitter_frequency() const;
	void set_sub_emitter_frequency(double p_hz);
	int32_t get_sub_emitter_amount_at_end() const;
	void set_sub_emitter_amount_at_end(int32_t p_amount);
	int32_t get_sub_emitter_amount_at_collision() const;
	void set_sub_emitter_amount_at_collision(int32_t p_amount);
	int32_t get_sub_emitter_amount_at_start() const;
	void set_sub_emitter_amount_at_start(int32_t p_amount);
	bool get_sub_emitter_keep_velocity() const;
	void set_sub_emitter_keep_velocity(bool p_enable);
	void set_attractor_interaction_enabled(bool p_enabled);
	bool is_attractor_interaction_enabled() const;
	void set_collision_mode(ParticleProcessMaterial::CollisionMode p_mode);
	ParticleProcessMaterial::CollisionMode get_collision_mode() const;
	void set_collision_use_scale(bool p_radius);
	bool is_collision_using_scale() const;
	void set_collision_friction(float p_friction);
	float get_collision_friction() const;
	void set_collision_bounce(float p_bounce);
	float get_collision_bounce() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Material::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(ParticleProcessMaterial::Parameter);
VARIANT_ENUM_CAST(ParticleProcessMaterial::ParticleFlags);
VARIANT_ENUM_CAST(ParticleProcessMaterial::EmissionShape);
VARIANT_ENUM_CAST(ParticleProcessMaterial::SubEmitterMode);
VARIANT_ENUM_CAST(ParticleProcessMaterial::CollisionMode);

