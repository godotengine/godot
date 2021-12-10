/*************************************************************************/
/*  cpu_particles_2d.h                                                   */
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

#ifndef CPU_PARTICLES_2D_H
#define CPU_PARTICLES_2D_H

#include "core/rid.h"
#include "scene/2d/node_2d.h"
#include "scene/resources/texture.h"

class CPUParticles2D : public Node2D {
private:
	GDCLASS(CPUParticles2D, Node2D);

public:
	enum DrawOrder {
		DRAW_ORDER_INDEX,
		DRAW_ORDER_LIFETIME,
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

	enum Flags {
		FLAG_ALIGN_Y_TO_VELOCITY,
		FLAG_ROTATE_Y, // Unused, but exposed for consistency with 3D.
		FLAG_DISABLE_Z, // Unused, but exposed for consistency with 3D.
		FLAG_MAX
	};

	enum EmissionShape {
		EMISSION_SHAPE_POINT,
		EMISSION_SHAPE_SPHERE,
		EMISSION_SHAPE_RECTANGLE,
		EMISSION_SHAPE_POINTS,
		EMISSION_SHAPE_DIRECTED_POINTS,
		EMISSION_SHAPE_MAX
	};

private:
	bool emitting;

	// warning - beware of adding non-trivial types
	// to this structure as it is zeroed to initialize in set_amount()
	struct Particle {
		Transform2D transform;
		Color color;
		float custom[4];
		float rotation;
		Vector2 velocity;
		bool active;
		float angle_rand;
		float scale_rand;
		float hue_rot_rand;
		float anim_offset_rand;
		Color start_color_rand;
		float time;
		float lifetime;
		Color base_color;

		uint32_t seed;
	};

	float time;
	float inactive_time;
	float frame_remainder;
	int cycle;
	bool redraw;

	RID mesh;
	RID multimesh;

	PoolVector<Particle> particles;
	PoolVector<float> particle_data;
	PoolVector<int> particle_order;

	struct SortLifetime {
		const Particle *particles;

		bool operator()(int p_a, int p_b) const {
			return particles[p_a].time > particles[p_b].time;
		}
	};

	struct SortAxis {
		const Particle *particles;
		Vector2 axis;
		bool operator()(int p_a, int p_b) const {
			return axis.dot(particles[p_a].transform[2]) < axis.dot(particles[p_b].transform[2]);
		}
	};

	//

	bool one_shot;

	float lifetime;
	float pre_process_time;
	float explosiveness_ratio;
	float randomness_ratio;
	float lifetime_randomness;
	float speed_scale;
	bool local_coords;
	int fixed_fps;
	bool fractional_delta;

	Transform2D inv_emission_transform;

	DrawOrder draw_order;

	Ref<Texture> texture;
	Ref<Texture> normalmap;

	////////

	Vector2 direction;
	float spread;

	float parameters[PARAM_MAX];
	float randomness[PARAM_MAX];

	Ref<Curve> curve_parameters[PARAM_MAX];
	Color color;
	Ref<Gradient> color_ramp;
	Ref<Gradient> color_initial_ramp;

	bool flags[FLAG_MAX];

	EmissionShape emission_shape;
	float emission_sphere_radius;
	Vector2 emission_rect_extents;
	PoolVector<Vector2> emission_points;
	PoolVector<Vector2> emission_normals;
	PoolVector<Color> emission_colors;
	int emission_point_count;

	Vector2 gravity;

	void _update_internal();
	void _particles_process(float p_delta);
	void _update_particle_data_buffer();

	Mutex update_mutex;

	void _update_render_thread();

	void _update_mesh_texture();

	void _set_redraw(bool p_redraw);

	void _texture_changed();

protected:
	static void _bind_methods();
	void _notification(int p_what);
	virtual void _validate_property(PropertyInfo &property) const;

public:
	void set_emitting(bool p_emitting);
	void set_amount(int p_amount);
	void set_lifetime(float p_lifetime);
	void set_one_shot(bool p_one_shot);
	void set_pre_process_time(float p_time);
	void set_explosiveness_ratio(float p_ratio);
	void set_randomness_ratio(float p_ratio);
	void set_lifetime_randomness(float p_random);
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
	bool get_use_local_coordinates() const;
	float get_speed_scale() const;

	void set_fixed_fps(int p_count);
	int get_fixed_fps() const;

	void set_fractional_delta(bool p_enable);
	bool get_fractional_delta() const;

	void set_draw_order(DrawOrder p_order);
	DrawOrder get_draw_order() const;

	void set_texture(const Ref<Texture> &p_texture);
	Ref<Texture> get_texture() const;

	void set_normalmap(const Ref<Texture> &p_normalmap);
	Ref<Texture> get_normalmap() const;

	///////////////////

	void set_direction(Vector2 p_direction);
	Vector2 get_direction() const;

	void set_spread(float p_spread);
	float get_spread() const;

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

	void set_color_initial_ramp(const Ref<Gradient> &p_ramp);
	Ref<Gradient> get_color_initial_ramp() const;

	void set_particle_flag(Flags p_flag, bool p_enable);
	bool get_particle_flag(Flags p_flag) const;

	void set_emission_shape(EmissionShape p_shape);
	void set_emission_sphere_radius(float p_radius);
	void set_emission_rect_extents(Vector2 p_extents);
	void set_emission_points(const PoolVector<Vector2> &p_points);
	void set_emission_normals(const PoolVector<Vector2> &p_normals);
	void set_emission_colors(const PoolVector<Color> &p_colors);

	EmissionShape get_emission_shape() const;
	float get_emission_sphere_radius() const;
	Vector2 get_emission_rect_extents() const;
	PoolVector<Vector2> get_emission_points() const;
	PoolVector<Vector2> get_emission_normals() const;
	PoolVector<Color> get_emission_colors() const;

	void set_gravity(const Vector2 &p_gravity);
	Vector2 get_gravity() const;

	virtual String get_configuration_warning() const;

	void restart();

	void convert_from_particles(Node *p_particles);

	CPUParticles2D();
	~CPUParticles2D();
};

VARIANT_ENUM_CAST(CPUParticles2D::DrawOrder)
VARIANT_ENUM_CAST(CPUParticles2D::Parameter)
VARIANT_ENUM_CAST(CPUParticles2D::Flags)
VARIANT_ENUM_CAST(CPUParticles2D::EmissionShape)

#endif // CPU_PARTICLES_2D_H
