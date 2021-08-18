/*************************************************************************/
/*  gpu_particles_2d.h                                                   */
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

#ifndef PARTICLES_2D_H
#define PARTICLES_2D_H

#include "core/templates/rid.h"
#include "scene/2d/node_2d.h"
#include "scene/resources/texture.h"

class GPUParticles2D : public Node2D {
private:
	GDCLASS(GPUParticles2D, Node2D);

public:
	enum DrawOrder {
		DRAW_ORDER_INDEX,
		DRAW_ORDER_LIFETIME,
		DRAW_ORDER_REVERSE_LIFETIME,
	};

private:
	RID particles;

	bool one_shot;
	int amount;
	float lifetime;
	float pre_process_time;
	float explosiveness_ratio;
	float randomness_ratio;
	float speed_scale;
	Rect2 visibility_rect;
	bool local_coords;
	int fixed_fps;
	bool fractional_delta;

	Ref<Material> process_material;

	DrawOrder draw_order;

	Ref<Texture2D> texture;

	void _update_particle_emission_transform();

	NodePath sub_emitter;
	float collision_base_size = 1.0;

	bool trail_enabled = false;
	float trail_length = 0.3;
	int trail_sections = 8;
	int trail_section_subdivisions = 4;

	RID mesh;

protected:
	static void _bind_methods();
	virtual void _validate_property(PropertyInfo &property) const override;
	void _notification(int p_what);

	void _update_collision_size();

public:
	void set_emitting(bool p_emitting);
	void set_amount(int p_amount);
	void set_lifetime(float p_lifetime);
	void set_one_shot(bool p_enable);
	void set_pre_process_time(float p_time);
	void set_explosiveness_ratio(float p_ratio);
	void set_randomness_ratio(float p_ratio);
	void set_visibility_rect(const Rect2 &p_visibility_rect);
	void set_use_local_coordinates(bool p_enable);
	void set_process_material(const Ref<Material> &p_material);
	void set_speed_scale(float p_scale);
	void set_collision_base_size(float p_ratio);
	void set_trail_enabled(bool p_enabled);
	void set_trail_length(float p_seconds);
	void set_trail_sections(int p_sections);
	void set_trail_section_subdivisions(int p_subdivisions);

	bool is_emitting() const;
	int get_amount() const;
	float get_lifetime() const;
	bool get_one_shot() const;
	float get_pre_process_time() const;
	float get_explosiveness_ratio() const;
	float get_randomness_ratio() const;
	Rect2 get_visibility_rect() const;
	bool get_use_local_coordinates() const;
	Ref<Material> get_process_material() const;
	float get_speed_scale() const;

	float get_collision_base_size() const;
	bool is_trail_enabled() const;
	float get_trail_length() const;
	int get_trail_sections() const;
	int get_trail_section_subdivisions() const;

	void set_fixed_fps(int p_count);
	int get_fixed_fps() const;

	void set_fractional_delta(bool p_enable);
	bool get_fractional_delta() const;

	void set_draw_order(DrawOrder p_order);
	DrawOrder get_draw_order() const;

	void set_texture(const Ref<Texture2D> &p_texture);
	Ref<Texture2D> get_texture() const;

	TypedArray<String> get_configuration_warnings() const override;

	void restart();
	Rect2 capture_rect() const;
	GPUParticles2D();
	~GPUParticles2D();
};

VARIANT_ENUM_CAST(GPUParticles2D::DrawOrder)

#endif // PARTICLES_2D_H
