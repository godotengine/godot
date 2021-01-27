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

#include "core/rid.h"
#include "scene/resources/material.h"

#ifndef PARTICLES_MATERIAL_H
#define PARTICLES_MATERIAL_H

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

	enum Flags {
		FLAG_ALIGN_Y_TO_VELOCITY,
		FLAG_ROTATE_Y,
		FLAG_DISABLE_Z,
		FLAG_MAX
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
	union MaterialKey {

		struct {
			uint32_t texture_mask : 16;
			uint32_t texture_color : 1;
			uint32_t flags : 4;
			uint32_t emission_shape : 2;
			uint32_t trail_size_texture : 1;
			uint32_t trail_color_texture : 1;
			uint32_t invalid_key : 1;
			uint32_t has_emission_color : 1;
		};

		uint32_t key;

		bool operator<(const MaterialKey &p_key) const {
			return key < p_key.key;
		}
	};

	struct ShaderData {
		RID shader;
		int users;
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
		for (int i = 0; i < FLAG_MAX; i++) {
			if (flags[i]) {
				mk.flags |= (1 << i);
			}
		}

		mk.texture_color = color_ramp.is_valid() ? 1 : 0;
		mk.emission_shape = emission_shape;
		mk.trail_color_texture = trail_color_modifier.is_valid() ? 1 : 0;
		mk.trail_size_texture = trail_size_modifier.is_valid() ? 1 : 0;
		mk.has_emission_color = emission_shape >= EMISSION_SHAPE_POINTS && emission_color_texture.is_valid();

		return mk;
	}

	static Mutex material_mutex;
	static SelfList<ParticlesMaterial>::List *dirty_materials;

	struct ShaderNames {
		StringName direction;
		StringName spread;
		StringName flatness;
		StringName initial_linear_velocity;
		StringName initial_angle;
		StringName angular_velocity;
		StringName orbit_velocity;
		StringName linear_accel;
		StringName radial_accel;
		StringName tangent_accel;
		StringName damping;
		StringName scale;
		StringName hue_variation;
		StringName anim_speed;
		StringName anim_offset;

		StringName initial_linear_velocity_random;
		StringName initial_angle_random;
		StringName angular_velocity_random;
		StringName orbit_velocity_random;
		StringName linear_accel_random;
		StringName radial_accel_random;
		StringName tangent_accel_random;
		StringName damping_random;
		StringName scale_random;
		StringName hue_variation_random;
		StringName anim_speed_random;
		StringName anim_offset_random;

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

		StringName trail_divisor;
		StringName trail_size_modifier;
		StringName trail_color_modifier;

		StringName gravity;

		StringName lifetime_randomness;
	};

	static ShaderNames *shader_names;

	SelfList<ParticlesMaterial> element;

	void _update_shader();
	_FORCE_INLINE_ void _queue_shader_change();
	_FORCE_INLINE_ bool _is_shader_dirty() const;

	Vector3 direction;
	float spread;
	float flatness;

	float parameters[PARAM_MAX];
	float randomness[PARAM_MAX];

	Ref<Texture> tex_parameters[PARAM_MAX];
	Color color;
	Ref<Texture> color_ramp;

	bool flags[FLAG_MAX];

	EmissionShape emission_shape;
	float emission_sphere_radius;
	Vector3 emission_box_extents;
	Ref<Texture> emission_point_texture;
	Ref<Texture> emission_normal_texture;
	Ref<Texture> emission_color_texture;
	int emission_point_count;

	bool anim_loop;

	int trail_divisor;

	Ref<CurveTexture> trail_size_modifier;
	Ref<GradientTexture> trail_color_modifier;

	Vector3 gravity;

	float lifetime_randomness;

	//do not save emission points here

protected:
	static void _bind_methods();
	virtual void _validate_property(PropertyInfo &property) const;

public:
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

	void set_param_texture(Parameter p_param, const Ref<Texture> &p_texture);
	Ref<Texture> get_param_texture(Parameter p_param) const;

	void set_color(const Color &p_color);
	Color get_color() const;

	void set_color_ramp(const Ref<Texture> &p_texture);
	Ref<Texture> get_color_ramp() const;

	void set_flag(Flags p_flag, bool p_enable);
	bool get_flag(Flags p_flag) const;

	void set_emission_shape(EmissionShape p_shape);
	void set_emission_sphere_radius(float p_radius);
	void set_emission_box_extents(Vector3 p_extents);
	void set_emission_point_texture(const Ref<Texture> &p_points);
	void set_emission_normal_texture(const Ref<Texture> &p_normals);
	void set_emission_color_texture(const Ref<Texture> &p_colors);
	void set_emission_point_count(int p_count);

	EmissionShape get_emission_shape() const;
	float get_emission_sphere_radius() const;
	Vector3 get_emission_box_extents() const;
	Ref<Texture> get_emission_point_texture() const;
	Ref<Texture> get_emission_normal_texture() const;
	Ref<Texture> get_emission_color_texture() const;
	int get_emission_point_count() const;

	void set_trail_divisor(int p_divisor);
	int get_trail_divisor() const;

	void set_trail_size_modifier(const Ref<CurveTexture> &p_trail_size_modifier);
	Ref<CurveTexture> get_trail_size_modifier() const;

	void set_trail_color_modifier(const Ref<GradientTexture> &p_trail_color_modifier);
	Ref<GradientTexture> get_trail_color_modifier() const;

	void set_gravity(const Vector3 &p_gravity);
	Vector3 get_gravity() const;

	void set_lifetime_randomness(float p_lifetime);
	float get_lifetime_randomness() const;

	static void init_shaders();
	static void finish_shaders();
	static void flush_changes();

	RID get_shader_rid() const;

	virtual Shader::Mode get_shader_mode() const;

	ParticlesMaterial();
	~ParticlesMaterial();
};

VARIANT_ENUM_CAST(ParticlesMaterial::Parameter)
VARIANT_ENUM_CAST(ParticlesMaterial::Flags)
VARIANT_ENUM_CAST(ParticlesMaterial::EmissionShape)

#endif // PARTICLES_MATERIAL_H
