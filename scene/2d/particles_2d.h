/*************************************************************************/
/*  particles_2d.h                                                       */
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
#ifndef PARTICLES_FRAME_H
#define PARTICLES_FRAME_H

#include "scene/2d/node_2d.h"
#include "scene/resources/color_ramp.h"
#include "scene/resources/texture.h"

class Particles2D;
class ParticleAttractor2D : public Node2D {

	GDCLASS(ParticleAttractor2D, Node2D);

	friend class Particles2D;
	bool enabled;
	float radius;
	float disable_radius;
	float gravity;
	float absorption;
	NodePath path;

	Particles2D *owner;

	void _update_owner();
	void _owner_exited();
	void _set_owner(Particles2D *p_owner);

	void _notification(int p_what);
	static void _bind_methods();

public:
	void set_enabled(bool p_enabled);
	bool is_enabled() const;

	void set_radius(float p_radius);
	float get_radius() const;

	void set_disable_radius(float p_disable_radius);
	float get_disable_radius() const;

	void set_gravity(float p_gravity);
	float get_gravity() const;

	void set_absorption(float p_absorption);
	float get_absorption() const;

	void set_particles_path(NodePath p_path);
	NodePath get_particles_path() const;

	virtual String get_configuration_warning() const;

	ParticleAttractor2D();
};

class Particles2D : public Node2D {

	GDCLASS(Particles2D, Node2D);

public:
	enum Parameter {
		PARAM_DIRECTION,
		PARAM_SPREAD,
		PARAM_LINEAR_VELOCITY,
		PARAM_SPIN_VELOCITY,
		PARAM_ORBIT_VELOCITY,
		PARAM_GRAVITY_DIRECTION,
		PARAM_GRAVITY_STRENGTH,
		PARAM_RADIAL_ACCEL,
		PARAM_TANGENTIAL_ACCEL,
		PARAM_DAMPING,
		PARAM_INITIAL_ANGLE,
		PARAM_INITIAL_SIZE,
		PARAM_FINAL_SIZE,
		PARAM_HUE_VARIATION,
		PARAM_ANIM_SPEED_SCALE,
		PARAM_ANIM_INITIAL_POS,
		PARAM_MAX
	};

	enum {
		MAX_COLOR_PHASES = 4
	};

	enum ProcessMode {
		PROCESS_FIXED,
		PROCESS_IDLE,
	};

private:
	float param[PARAM_MAX];
	float randomness[PARAM_MAX];

	struct Particle {
		bool active;
		Point2 pos;
		Vector2 velocity;
		float rot;
		float frame;
		uint64_t seed;
		Particle() {
			active = false;
			seed = 123465789;
			rot = 0;
			frame = 0;
		}
	};

	Vector<Particle> particles;

	struct AttractorCache {

		Vector2 pos;
		ParticleAttractor2D *attractor;
	};

	Vector<AttractorCache> attractor_cache;

	float explosiveness;
	float preprocess;
	float lifetime;
	bool emitting;
	bool local_space;
	float emit_timeout;
	float time_to_live;
	float time_scale;
	bool flip_h;
	bool flip_v;
	int h_frames;
	int v_frames;
	Point2 emissor_offset;
	Vector2 initial_velocity;
	Vector2 extents;
	PoolVector<Vector2> emission_points;

	ProcessMode process_mode;

	float time;
	int active_count;

	Ref<Texture> texture;

	//If no color ramp is set then default color is used. Created as simple alternative to color_ramp.
	Color default_color;
	Ref<ColorRamp> color_ramp;

	void _process_particles(float p_delta);
	friend class ParticleAttractor2D;

	Set<ParticleAttractor2D *> attractors;

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void set_emitting(bool p_emitting);
	bool is_emitting() const;

	void set_process_mode(ProcessMode p_mode);
	ProcessMode get_process_mode() const;

	void set_amount(int p_amount);
	int get_amount() const;

	void set_lifetime(float p_lifetime);
	float get_lifetime() const;

	void set_time_scale(float p_time_scale);
	float get_time_scale() const;

	void set_pre_process_time(float p_pre_process_time);
	float get_pre_process_time() const;

	void set_emit_timeout(float p_timeout);
	float get_emit_timeout() const;

	void set_emission_half_extents(const Vector2 &p_extents);
	Vector2 get_emission_half_extents() const;

	void set_param(Parameter p_param, float p_value);
	float get_param(Parameter p_param) const;

	void set_randomness(Parameter p_randomness, float p_value);
	float get_randomness(Parameter p_randomness) const;

	void set_explosiveness(float p_value);
	float get_explosiveness() const;

	void set_flip_h(bool p_flip);
	bool is_flipped_h() const;

	void set_flip_v(bool p_flip);
	bool is_flipped_v() const;

	void set_h_frames(int p_frames);
	int get_h_frames() const;

	void set_v_frames(int p_frames);
	int get_v_frames() const;

	void set_color_phases(int p_phases);
	int get_color_phases() const;

	void set_color_phase_color(int p_phase, const Color &p_color);
	Color get_color_phase_color(int p_phase) const;

	void set_color_phase_pos(int p_phase, float p_pos);
	float get_color_phase_pos(int p_phase) const;

	void set_texture(const Ref<Texture> &p_texture);
	Ref<Texture> get_texture() const;

	void set_color(const Color &p_color);
	Color get_color() const;

	void set_color_ramp(const Ref<ColorRamp> &p_texture);
	Ref<ColorRamp> get_color_ramp() const;

	void set_emissor_offset(const Point2 &p_offset);
	Point2 get_emissor_offset() const;

	void set_use_local_space(bool p_use);
	bool is_using_local_space() const;

	void set_initial_velocity(const Vector2 &p_velocity);
	Vector2 get_initial_velocity() const;

	void set_emission_points(const PoolVector<Vector2> &p_points);
	PoolVector<Vector2> get_emission_points() const;

	void pre_process(float p_delta);
	void reset();

	Particles2D();
};

VARIANT_ENUM_CAST(Particles2D::ProcessMode);
VARIANT_ENUM_CAST(Particles2D::Parameter);

#endif // PARTICLES_FRAME_H
