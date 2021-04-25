/*************************************************************************/
/*  cpu_particles_3d.h                                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef CPU_PARTICLES_H
#define CPU_PARTICLES_H

#include "core/templates/rid.h"
#include "core/templates/safe_refcount.h"
#include "scene/3d/visual_instance_3d.h"

class CPUParticles3D : public GeometryInstance3D {
private:
	GDCLASS(CPUParticles3D, GeometryInstance3D);

public:
	enum DrawOrder {
		DRAW_ORDER_INDEX,
		DRAW_ORDER_LIFETIME,
		DRAW_ORDER_VIEW_DEPTH,
		DRAW_ORDER_MAX
	};

	enum Parameter {
		PARAM_INITIAL_LINEAR_VELOCITY,
		PARAM_ANGULAR_VELOCITY,
		PARAM_ORBIT_VELOCITY,
		PARAM_LINEAR_ACCEL,
		PARAM_RADIAL_ACCEL,
		PARAM_TANGENTIAL_ACCEL,
		PARAM_DAMPING,
		PARAM_ANGLE,
		PARAM_SCALE,
		PARAM_HUE_VARIATION,
		PARAM_ANIM_SPEED,
		PARAM_ANIM_OFFSET,
		PARAM_MAX
	};

	enum ParticleFlags {
		PARTICLE_FLAG_ALIGN_Y_TO_VELOCITY,
		PARTICLE_FLAG_ROTATE_Y,
		PARTICLE_FLAG_DISABLE_Z,
		PARTICLE_FLAG_MAX
	};

	enum EmissionShape {
		EMISSION_SHAPE_POINT,
		EMISSION_SHAPE_SPHERE,
		EMISSION_SHAPE_BOX,
		EMISSION_SHAPE_POINTS,
		EMISSION_SHAPE_DIRECTED_POINTS,
		EMISSION_SHAPE_MAX
	};

private:
	bool emitting = false;

	struct Particle {
		Transform transform;
		Color color;
		float custom[4] = {};
		Vector3 velocity;
		bool active = false;
		float angle_rand = 0.0;
		float scale_rand = 0.0;
		float hue_rot_rand = 0.0;
		float anim_offset_rand = 0.0;
		float time = 0.0;
		float lifetime = 0.0;
		Color base_color;

		uint32_t seed = 0;
	};

	float time = 0.0;
	float inactive_time = 0.0;
	float frame_remainder = 0.0;
	int cycle = 0;
	bool redraw = false;

	RID multimesh;

	Vector<Particle> particles;
	Vector<float> particle_data;
	Vector<int> particle_order;

	struct SortLifetime {
		const Particle *particles = nullptr;

		bool operator()(int p_a, int p_b) const {
			return particles[p_a].time > particles[p_b].time;
		}
	};

	struct SortAxis {
		const Particle *particles = nullptr;
		Vector3 axis;
		bool operator()(int p_a, int p_b) const {
			return axis.dot(particles[p_a].transform.origin) < axis.dot(particles[p_b].transform.origin);
		}
	};

	//

	bool one_shot = false;

	float lifetime = 1.0;
	float pre_process_time = 0.0;
	float explosiveness_ratio = 0.0;
	float randomness_ratio = 0.0;
	float lifetime_randomness = 0.0;
	float speed_scale = 1.0;
	bool local_coords = true;
	int fixed_fps = 0;
	bool fractional_delta = true;

	Transform inv_emission_transform;

	SafeFlag can_update;

	DrawOrder draw_order = DRAW_ORDER_INDEX;

	Ref<Mesh> mesh;

	////////

	Vector3 direction = Vector3(1, 0, 0);
	float spread = 45.0;
	float flatness = 0.0;

	float parameters[PARAM_MAX];
	float randomness[PARAM_MAX] = {};

	Ref<Curve> curve_parameters[PARAM_MAX];
	Color color = Color(1, 1, 1, 1);
	Ref<Gradient> color_ramp;

	bool particle_flags[PARTICLE_FLAG_MAX] = {};

	EmissionShape emission_shape = EMISSION_SHAPE_POINT;
	float emission_sphere_radius = 1.0;
	Vector3 emission_box_extents = Vector3(1, 1, 1);
	Vector<Vector3> emission_points;
	Vector<Vector3> emission_normals;
	Vector<Color> emission_colors;
	int emission_point_count = 0;

	Vector3 gravity = Vector3(0, -9.8, 0);

	void _update_internal();
	void _particles_process(float p_delta);
	void _update_particle_data_buffer();

	Mutex update_mutex;

	void _update_render_thread();

	void _set_redraw(bool p_redraw);

protected:
	static void _bind_methods();
	void _notification(int p_what);
	virtual void _validate_property(PropertyInfo &property) const override;

public:
	AABB get_aabb() const override;
	Vector<Face3> get_faces(uint32_t p_usage_flags) const override;

	void set_emitting(bool p_emitting);
	void set_amount(int p_amount);
	void set_lifetime(float p_lifetime);
	void set_one_shot(bool p_one_shot);
	void set_pre_process_time(float p_time);
	void set_explosiveness_ratio(float p_ratio);
	void set_randomness_ratio(float p_ratio);
	void set_lifetime_randomness(float p_random);
	void set_visibility_aabb(const AABB &p_aabb);
	void set_use_local_coordinates(bool p_enable);
	void set_speed_scale(float p_scale);

	bool is_emitting() const;
	int get_amount() const;
	float get_lifetime() const;
	bool get_one_shot() const;
	float get_pre_process_time() const;
	float get_explosiveness_ratio() const;
	float get_randomness_ratio() const;
	float get_lifetime_randomness() const;
	AABB get_visibility_aabb() const;
	bool get_use_local_coordinates() const;
	float get_speed_scale() const;

	void set_fixed_fps(int p_count);
	int get_fixed_fps() const;

	void set_fractional_delta(bool p_enable);
	bool get_fractional_delta() const;

	void set_draw_order(DrawOrder p_order);
	DrawOrder get_draw_order() const;

	void set_draw_passes(int p_count);
	int get_draw_passes() const;

	void set_mesh(const Ref<Mesh> &p_mesh);
	Ref<Mesh> get_mesh() const;

	///////////////////

	void set_direction(Vector3 p_direction);
	Vector3 get_direction() const;

	void set_spread(float p_spread);
	float get_spread() const;

	void set_flatness(float p_flatness);
	float get_flatness() const;

	void set_param(Parameter p_param, float p_value);
	float get_param(Parameter p_param) const;

	void set_param_randomness(Parameter p_param, float p_value);
	float get_param_randomness(Parameter p_param) const;

	void set_param_curve(Parameter p_param, const Ref<Curve> &p_curve);
	Ref<Curve> get_param_curve(Parameter p_param) const;

	void set_color(const Color &p_color);
	Color get_color() const;

	void set_color_ramp(const Ref<Gradient> &p_ramp);
	Ref<Gradient> get_color_ramp() const;

	void set_particle_flag(ParticleFlags p_particle_flag, bool p_enable);
	bool get_particle_flag(ParticleFlags p_particle_flag) const;

	void set_emission_shape(EmissionShape p_shape);
	void set_emission_sphere_radius(float p_radius);
	void set_emission_box_extents(Vector3 p_extents);
	void set_emission_points(const Vector<Vector3> &p_points);
	void set_emission_normals(const Vector<Vector3> &p_normals);
	void set_emission_colors(const Vector<Color> &p_colors);
	void set_emission_point_count(int p_count);

	EmissionShape get_emission_shape() const;
	float get_emission_sphere_radius() const;
	Vector3 get_emission_box_extents() const;
	Vector<Vector3> get_emission_points() const;
	Vector<Vector3> get_emission_normals() const;
	Vector<Color> get_emission_colors() const;
	int get_emission_point_count() const;

	void set_gravity(const Vector3 &p_gravity);
	Vector3 get_gravity() const;

	TypedArray<String> get_configuration_warnings() const override;

	void restart();

	void convert_from_particles(Node *p_particles);

	CPUParticles3D();
	~CPUParticles3D();
};

VARIANT_ENUM_CAST(CPUParticles3D::DrawOrder)
VARIANT_ENUM_CAST(CPUParticles3D::Parameter)
VARIANT_ENUM_CAST(CPUParticles3D::ParticleFlags)
VARIANT_ENUM_CAST(CPUParticles3D::EmissionShape)

#endif // CPU_PARTICLES_H
