/**************************************************************************/
/*  gpu_particles3d.hpp                                                   */
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

#include <godot_cpp/classes/geometry_instance3d.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/variant/aabb.hpp>
#include <godot_cpp/variant/node_path.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

struct Color;
class Material;
class Mesh;
class Node;
class Skin;
struct Transform3D;
struct Vector3;

class GPUParticles3D : public GeometryInstance3D {
	GDEXTENSION_CLASS(GPUParticles3D, GeometryInstance3D)

public:
	enum DrawOrder {
		DRAW_ORDER_INDEX = 0,
		DRAW_ORDER_LIFETIME = 1,
		DRAW_ORDER_REVERSE_LIFETIME = 2,
		DRAW_ORDER_VIEW_DEPTH = 3,
	};

	enum EmitFlags {
		EMIT_FLAG_POSITION = 1,
		EMIT_FLAG_ROTATION_SCALE = 2,
		EMIT_FLAG_VELOCITY = 4,
		EMIT_FLAG_COLOR = 8,
		EMIT_FLAG_CUSTOM = 16,
	};

	enum TransformAlign {
		TRANSFORM_ALIGN_DISABLED = 0,
		TRANSFORM_ALIGN_Z_BILLBOARD = 1,
		TRANSFORM_ALIGN_Y_TO_VELOCITY = 2,
		TRANSFORM_ALIGN_Z_BILLBOARD_Y_TO_VELOCITY = 3,
	};

	static const int MAX_DRAW_PASSES = 4;

	void set_emitting(bool p_emitting);
	void set_amount(int32_t p_amount);
	void set_lifetime(double p_secs);
	void set_one_shot(bool p_enable);
	void set_pre_process_time(double p_secs);
	void set_explosiveness_ratio(float p_ratio);
	void set_randomness_ratio(float p_ratio);
	void set_visibility_aabb(const AABB &p_aabb);
	void set_use_local_coordinates(bool p_enable);
	void set_fixed_fps(int32_t p_fps);
	void set_fractional_delta(bool p_enable);
	void set_interpolate(bool p_enable);
	void set_process_material(const Ref<Material> &p_material);
	void set_speed_scale(double p_scale);
	void set_collision_base_size(float p_size);
	void set_interp_to_end(float p_interp);
	bool is_emitting() const;
	int32_t get_amount() const;
	double get_lifetime() const;
	bool get_one_shot() const;
	double get_pre_process_time() const;
	float get_explosiveness_ratio() const;
	float get_randomness_ratio() const;
	AABB get_visibility_aabb() const;
	bool get_use_local_coordinates() const;
	int32_t get_fixed_fps() const;
	bool get_fractional_delta() const;
	bool get_interpolate() const;
	Ref<Material> get_process_material() const;
	double get_speed_scale() const;
	float get_collision_base_size() const;
	float get_interp_to_end() const;
	void set_use_fixed_seed(bool p_use_fixed_seed);
	bool get_use_fixed_seed() const;
	void set_seed(uint32_t p_seed);
	uint32_t get_seed() const;
	void set_draw_order(GPUParticles3D::DrawOrder p_order);
	GPUParticles3D::DrawOrder get_draw_order() const;
	void set_draw_passes(int32_t p_passes);
	void set_draw_pass_mesh(int32_t p_pass, const Ref<Mesh> &p_mesh);
	int32_t get_draw_passes() const;
	Ref<Mesh> get_draw_pass_mesh(int32_t p_pass) const;
	void set_skin(const Ref<Skin> &p_skin);
	Ref<Skin> get_skin() const;
	void restart(bool p_keep_seed = false);
	AABB capture_aabb() const;
	void set_sub_emitter(const NodePath &p_path);
	NodePath get_sub_emitter() const;
	void emit_particle(const Transform3D &p_xform, const Vector3 &p_velocity, const Color &p_color, const Color &p_custom, uint32_t p_flags);
	void set_trail_enabled(bool p_enabled);
	void set_trail_lifetime(double p_secs);
	bool is_trail_enabled() const;
	double get_trail_lifetime() const;
	void set_transform_align(GPUParticles3D::TransformAlign p_align);
	GPUParticles3D::TransformAlign get_transform_align() const;
	void convert_from_particles(Node *p_particles);
	void set_amount_ratio(float p_ratio);
	float get_amount_ratio() const;
	void request_particles_process(float p_process_time);

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		GeometryInstance3D::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(GPUParticles3D::DrawOrder);
VARIANT_ENUM_CAST(GPUParticles3D::EmitFlags);
VARIANT_ENUM_CAST(GPUParticles3D::TransformAlign);

