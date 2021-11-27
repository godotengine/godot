/*************************************************************************/
/*  particles_material.h                                                 */
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

#include "core/templates/rid.h"
#include "scene/resources/material.h"

#ifndef PARTICLES_MATERIAL_H
#define PARTICLES_MATERIAL_H

/*
 TODO:
-Path following
-Emitter positions deformable by bones
-Proper trails
*/

class ParticlesMaterial : public Material {
	GDCLASS(ParticlesMaterial, Material);

public:
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

	// When extending, make sure not to overflow the size of the MaterialKey below.
	enum ParticleFlags {
		PARTICLE_FLAG_ALIGN_Y_TO_VELOCITY,
		PARTICLE_FLAG_ROTATE_Y,
		PARTICLE_FLAG_DISABLE_Z,
		PARTICLE_FLAG_MAX
	};

	// When extending, make sure not to overflow the size of the MaterialKey below.
	enum EmissionShape {
		EMISSION_SHAPE_POINT,
		EMISSION_SHAPE_SPHERE,
		EMISSION_SHAPE_SPHERE_SURFACE,
		EMISSION_SHAPE_BOX,
		EMISSION_SHAPE_POINTS,
		EMISSION_SHAPE_DIRECTED_POINTS,
		EMISSION_SHAPE_RING,
		EMISSION_SHAPE_MAX
	};

	// When extending, make sure not to overflow the size of the MaterialKey below.
	enum SubEmitterMode {
		SUB_EMITTER_DISABLED,
		SUB_EMITTER_CONSTANT,
		SUB_EMITTER_AT_END,
		SUB_EMITTER_AT_COLLISION,
		SUB_EMITTER_MAX
	};

private:
	union MaterialKey {
		// The bit size of the struct must be kept below or equal to 32 bits.
		// Consider this when extending ParticleFlags, EmissionShape, or SubEmitterMode.
		struct {
			uint32_t texture_mask : 16;
			uint32_t texture_color : 1;
			uint32_t particle_flags : 4;
			uint32_t emission_shape : 3;
			uint32_t invalid_key : 1;
			uint32_t has_emission_color : 1;
			uint32_t sub_emitter : 2;
			uint32_t attractor_enabled : 1;
			uint32_t collision_enabled : 1;
			uint32_t collision_scale : 1;
		};

		uint32_t key = 0;

		bool operator<(const MaterialKey &p_key) const {
			return key < p_key.key;
		}
	};

	struct ShaderData {
		RID shader;
		int users = 0;
	};

	static Map<MaterialKey, ShaderData> shader_map;

	MaterialKey current_key;

	_FORCE_INLINE_ MaterialKey _compute_key() const {
		MaterialKey mk;
		mk.key = 0;
		for (int i = 0; i < PARAM_MAX; i++) {
			if (tex_parameters[i].is_valid()) {
				mk.texture_mask |= (1 << i);
			}
		}
		for (int i = 0; i < PARTICLE_FLAG_MAX; i++) {
			if (particle_flags[i]) {
				mk.particle_flags |= (1 << i);
			}
		}

		mk.texture_color = color_ramp.is_valid() ? 1 : 0;
		mk.emission_shape = emission_shape;
		mk.has_emission_color = emission_shape >= EMISSION_SHAPE_POINTS && emission_color_texture.is_valid();
		mk.sub_emitter = sub_emitter_mode;
		mk.collision_enabled = collision_enabled;
		mk.attractor_enabled = attractor_interaction_enabled;
		mk.collision_scale = collision_scale;

		return mk;
	}

	static Mutex material_mutex;
	static SelfList<ParticlesMaterial>::List *dirty_materials;

	struct ShaderNames {
		StringName direction;
		StringName spread;
		StringName flatness;
		StringName initial_linear_velocity_min;
		StringName initial_angle_min;
		StringName angular_velocity_min;
		StringName orbit_velocity_min;
		StringName linear_accel_min;
		StringName radial_accel_min;
		StringName tangent_accel_min;
		StringName damping_min;
		StringName scale_min;
		StringName hue_variation_min;
		StringName anim_speed_min;
		StringName anim_offset_min;

		StringName initial_linear_velocity_max;
		StringName initial_angle_max;
		StringName angular_velocity_max;
		StringName orbit_velocity_max;
		StringName linear_accel_max;
		StringName radial_accel_max;
		StringName tangent_accel_max;
		StringName damping_max;
		StringName scale_max;
		StringName hue_variation_max;
		StringName anim_speed_max;
		StringName anim_offset_max;

		StringName angle_texture;
		StringName angular_velocity_texture;
		StringName orbit_velocity_texture;
		StringName linear_accel_texture;
		StringName radial_accel_texture;
		StringName tangent_accel_texture;
		StringName damping_texture;
		StringName scale_texture;
		StringName hue_variation_texture;
		StringName anim_speed_texture;
		StringName anim_offset_texture;

		StringName color;
		StringName color_ramp;

		StringName emission_sphere_radius;
		StringName emission_box_extents;
		StringName emission_texture_point_count;
		StringName emission_texture_points;
		StringName emission_texture_normal;
		StringName emission_texture_color;
		StringName emission_ring_axis;
		StringName emission_ring_height;
		StringName emission_ring_radius;
		StringName emission_ring_inner_radius;

		StringName gravity;

		StringName lifetime_randomness;

		StringName sub_emitter_frequency;
		StringName sub_emitter_amount_at_end;
		StringName sub_emitter_keep_velocity;

		StringName collision_friction;
		StringName collision_bounce;
	};

	static ShaderNames *shader_names;

	SelfList<ParticlesMaterial> element;

	void _update_shader();
	_FORCE_INLINE_ void _queue_shader_change();
	_FORCE_INLINE_ bool _is_shader_dirty() const;

	bool is_initialized = false;
	Vector3 direction;
	float spread;
	float flatness;

	float params_min[PARAM_MAX];
	float params_max[PARAM_MAX];

	Ref<Texture2D> tex_parameters[PARAM_MAX];
	Color color;
	Ref<Texture2D> color_ramp;

	bool particle_flags[PARTICLE_FLAG_MAX];

	EmissionShape emission_shape;
	float emission_sphere_radius;
	Vector3 emission_box_extents;
	Ref<Texture2D> emission_point_texture;
	Ref<Texture2D> emission_normal_texture;
	Ref<Texture2D> emission_color_texture;
	Vector3 emission_ring_axis;
	real_t emission_ring_height;
	real_t emission_ring_radius;
	real_t emission_ring_inner_radius;
	int emission_point_count = 1;

	bool anim_loop;

	Vector3 gravity;

	double lifetime_randomness;

	SubEmitterMode sub_emitter_mode;
	double sub_emitter_frequency;
	int sub_emitter_amount_at_end;
	bool sub_emitter_keep_velocity;
	//do not save emission points here

	bool attractor_interaction_enabled;
	bool collision_enabled;
	bool collision_scale;
	float collision_friction;
	float collision_bounce;

protected:
	static void _bind_methods();
	virtual void _validate_property(PropertyInfo &property) const override;

public:
	void set_direction(Vector3 p_direction);
	Vector3 get_direction() const;

	void set_spread(float p_spread);
	float get_spread() const;

	void set_flatness(float p_flatness);
	float get_flatness() const;

	void set_param_min(Parameter p_param, float p_value);
	float get_param_min(Parameter p_param) const;

	void set_param_max(Parameter p_param, float p_value);
	float get_param_max(Parameter p_param) const;

	void set_param_texture(Parameter p_param, const Ref<Texture2D> &p_texture);
	Ref<Texture2D> get_param_texture(Parameter p_param) const;

	void set_color(const Color &p_color);
	Color get_color() const;

	void set_color_ramp(const Ref<Texture2D> &p_texture);
	Ref<Texture2D> get_color_ramp() const;

	void set_particle_flag(ParticleFlags p_particle_flag, bool p_enable);
	bool get_particle_flag(ParticleFlags p_particle_flag) const;

	void set_emission_shape(EmissionShape p_shape);
	void set_emission_sphere_radius(real_t p_radius);
	void set_emission_box_extents(Vector3 p_extents);
	void set_emission_point_texture(const Ref<Texture2D> &p_points);
	void set_emission_normal_texture(const Ref<Texture2D> &p_normals);
	void set_emission_color_texture(const Ref<Texture2D> &p_colors);
	void set_emission_ring_axis(Vector3 p_axis);
	void set_emission_ring_height(real_t p_height);
	void set_emission_ring_radius(real_t p_radius);
	void set_emission_ring_inner_radius(real_t p_radius);
	void set_emission_point_count(int p_count);

	EmissionShape get_emission_shape() const;
	real_t get_emission_sphere_radius() const;
	Vector3 get_emission_box_extents() const;
	Ref<Texture2D> get_emission_point_texture() const;
	Ref<Texture2D> get_emission_normal_texture() const;
	Ref<Texture2D> get_emission_color_texture() const;
	Vector3 get_emission_ring_axis() const;
	real_t get_emission_ring_height() const;
	real_t get_emission_ring_radius() const;
	real_t get_emission_ring_inner_radius() const;
	int get_emission_point_count() const;

	void set_gravity(const Vector3 &p_gravity);
	Vector3 get_gravity() const;

	void set_lifetime_randomness(double p_lifetime);
	double get_lifetime_randomness() const;

	void set_attractor_interaction_enabled(bool p_enable);
	bool is_attractor_interaction_enabled() const;

	void set_collision_enabled(bool p_enabled);
	bool is_collision_enabled() const;

	void set_collision_use_scale(bool p_scale);
	bool is_collision_using_scale() const;

	void set_collision_friction(float p_friction);
	float get_collision_friction() const;

	void set_collision_bounce(float p_bounce);
	float get_collision_bounce() const;

	static void init_shaders();
	static void finish_shaders();
	static void flush_changes();

	void set_sub_emitter_mode(SubEmitterMode p_sub_emitter_mode);
	SubEmitterMode get_sub_emitter_mode() const;

	void set_sub_emitter_frequency(double p_frequency);
	double get_sub_emitter_frequency() const;

	void set_sub_emitter_amount_at_end(int p_amount);
	int get_sub_emitter_amount_at_end() const;

	void set_sub_emitter_keep_velocity(bool p_enable);
	bool get_sub_emitter_keep_velocity() const;

	virtual RID get_shader_rid() const override;

	virtual Shader::Mode get_shader_mode() const override;

	ParticlesMaterial();
	~ParticlesMaterial();
};

VARIANT_ENUM_CAST(ParticlesMaterial::Parameter)
VARIANT_ENUM_CAST(ParticlesMaterial::ParticleFlags)
VARIANT_ENUM_CAST(ParticlesMaterial::EmissionShape)
VARIANT_ENUM_CAST(ParticlesMaterial::SubEmitterMode)

#endif // PARTICLES_MATERIAL_H
