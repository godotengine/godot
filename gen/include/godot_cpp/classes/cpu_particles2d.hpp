/**************************************************************************/
/*  cpu_particles2d.hpp                                                   */
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

#include <godot_cpp/classes/node2d.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/variant/color.hpp>
#include <godot_cpp/variant/packed_color_array.hpp>
#include <godot_cpp/variant/packed_vector2_array.hpp>
#include <godot_cpp/variant/vector2.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Curve;
class Gradient;
class Node;
class Texture2D;

class CPUParticles2D : public Node2D {
	GDEXTENSION_CLASS(CPUParticles2D, Node2D)

public:
	enum DrawOrder {
		DRAW_ORDER_INDEX = 0,
		DRAW_ORDER_LIFETIME = 1,
	};

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
		PARAM_MAX = 12,
	};

	enum ParticleFlags {
		PARTICLE_FLAG_ALIGN_Y_TO_VELOCITY = 0,
		PARTICLE_FLAG_ROTATE_Y = 1,
		PARTICLE_FLAG_DISABLE_Z = 2,
		PARTICLE_FLAG_MAX = 3,
	};

	enum EmissionShape {
		EMISSION_SHAPE_POINT = 0,
		EMISSION_SHAPE_SPHERE = 1,
		EMISSION_SHAPE_SPHERE_SURFACE = 2,
		EMISSION_SHAPE_RECTANGLE = 3,
		EMISSION_SHAPE_POINTS = 4,
		EMISSION_SHAPE_DIRECTED_POINTS = 5,
		EMISSION_SHAPE_RING = 6,
		EMISSION_SHAPE_MAX = 7,
	};

	void set_emitting(bool p_emitting);
	void set_amount(int32_t p_amount);
	void set_lifetime(double p_secs);
	void set_one_shot(bool p_enable);
	void set_pre_process_time(double p_secs);
	void set_explosiveness_ratio(float p_ratio);
	void set_randomness_ratio(float p_ratio);
	void set_lifetime_randomness(double p_random);
	void set_use_local_coordinates(bool p_enable);
	void set_fixed_fps(int32_t p_fps);
	void set_fractional_delta(bool p_enable);
	void set_speed_scale(double p_scale);
	void request_particles_process(float p_process_time);
	bool is_emitting() const;
	int32_t get_amount() const;
	double get_lifetime() const;
	bool get_one_shot() const;
	double get_pre_process_time() const;
	float get_explosiveness_ratio() const;
	float get_randomness_ratio() const;
	double get_lifetime_randomness() const;
	bool get_use_local_coordinates() const;
	int32_t get_fixed_fps() const;
	bool get_fractional_delta() const;
	double get_speed_scale() const;
	void set_use_fixed_seed(bool p_use_fixed_seed);
	bool get_use_fixed_seed() const;
	void set_seed(uint32_t p_seed);
	uint32_t get_seed() const;
	void set_draw_order(CPUParticles2D::DrawOrder p_order);
	CPUParticles2D::DrawOrder get_draw_order() const;
	void set_texture(const Ref<Texture2D> &p_texture);
	Ref<Texture2D> get_texture() const;
	void restart(bool p_keep_seed = false);
	void set_direction(const Vector2 &p_direction);
	Vector2 get_direction() const;
	void set_spread(float p_spread);
	float get_spread() const;
	void set_param_min(CPUParticles2D::Parameter p_param, float p_value);
	float get_param_min(CPUParticles2D::Parameter p_param) const;
	void set_param_max(CPUParticles2D::Parameter p_param, float p_value);
	float get_param_max(CPUParticles2D::Parameter p_param) const;
	void set_param_curve(CPUParticles2D::Parameter p_param, const Ref<Curve> &p_curve);
	Ref<Curve> get_param_curve(CPUParticles2D::Parameter p_param) const;
	void set_color(const Color &p_color);
	Color get_color() const;
	void set_color_ramp(const Ref<Gradient> &p_ramp);
	Ref<Gradient> get_color_ramp() const;
	void set_color_initial_ramp(const Ref<Gradient> &p_ramp);
	Ref<Gradient> get_color_initial_ramp() const;
	void set_particle_flag(CPUParticles2D::ParticleFlags p_particle_flag, bool p_enable);
	bool get_particle_flag(CPUParticles2D::ParticleFlags p_particle_flag) const;
	void set_emission_shape(CPUParticles2D::EmissionShape p_shape);
	CPUParticles2D::EmissionShape get_emission_shape() const;
	void set_emission_sphere_radius(float p_radius);
	float get_emission_sphere_radius() const;
	void set_emission_rect_extents(const Vector2 &p_extents);
	Vector2 get_emission_rect_extents() const;
	void set_emission_points(const PackedVector2Array &p_array);
	PackedVector2Array get_emission_points() const;
	void set_emission_normals(const PackedVector2Array &p_array);
	PackedVector2Array get_emission_normals() const;
	void set_emission_colors(const PackedColorArray &p_array);
	PackedColorArray get_emission_colors() const;
	void set_emission_ring_inner_radius(float p_inner_radius);
	float get_emission_ring_inner_radius() const;
	void set_emission_ring_radius(float p_radius);
	float get_emission_ring_radius() const;
	Vector2 get_gravity() const;
	void set_gravity(const Vector2 &p_accel_vec);
	bool get_split_scale();
	void set_split_scale(bool p_split_scale);
	Ref<Curve> get_scale_curve_x() const;
	void set_scale_curve_x(const Ref<Curve> &p_scale_curve);
	Ref<Curve> get_scale_curve_y() const;
	void set_scale_curve_y(const Ref<Curve> &p_scale_curve);
	void convert_from_particles(Node *p_particles);

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Node2D::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(CPUParticles2D::DrawOrder);
VARIANT_ENUM_CAST(CPUParticles2D::Parameter);
VARIANT_ENUM_CAST(CPUParticles2D::ParticleFlags);
VARIANT_ENUM_CAST(CPUParticles2D::EmissionShape);

