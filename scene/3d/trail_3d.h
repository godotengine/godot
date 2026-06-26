/**************************************************************************/
/*  trail_3d.h                                                            */
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

#pragma once

#include "scene/3d/line_3d.h"
#include "scene/resources/curve.h"
#include "scene/resources/gradient.h"
#include "scene/resources/mesh.h"

class Trail3D : public Line3D {
	GDCLASS(Trail3D, Line3D);

	// Lots of code for this node is inspired/ported from
	// https://codeberg.org/MajorMcDoom/cozy-cube-godot-addons/src/branch/main
	// thank you so much for putting it out there!

protected:
	void _notification(int p_what);
	void _validate_property(PropertyInfo &p_property) const;
	static void _bind_methods();

public:
	static void init_shaders();
	static void finish_shaders();

	void set_width(float p_width);
	float get_width() const;

	void set_emitting(bool p_emitting);
	bool is_emitting() const;

	void set_width_curve(Ref<Curve> p_curve);
	Ref<Curve> get_width_curve() const;

	void set_color(const Color &p_color);
	Color get_color() const;

	void set_color_gradient(Ref<Gradient> p_color_gradient);
	Ref<Gradient> get_color_gradient() const;

	void set_material_mode(MaterialMode p_material_mode);
	MaterialMode get_material_mode() const;

	void set_material(Ref<ShaderMaterial> p_material);
	Ref<ShaderMaterial> get_material() const;

	void set_mesh_alignment(MeshAlignment p_alignment);
	MeshAlignment get_mesh_alignment() const;

	void set_tiling_mode(TilingMode p_tiling_mode);
	TilingMode get_tiling_mode() const;

	void set_tiling_multiplier(float p_tiling_multiplier);
	float get_tiling_multiplier() const;

	void rebuild();
	void clear();

	// Trail3D

	enum LimitMode {
		LIMIT_MODE_LIFETIME,
		LIMIT_MODE_MAX_LENGTH,
		LIMIT_MODE_MAX,
	};

	void set_limit_mode(LimitMode p_limit_mode);
	LimitMode get_limit_mode() const;

	void set_min_section_length(real_t p_min_section_length);
	real_t get_min_section_length() const;

	void set_lifetime(real_t p_lifetime);
	real_t get_lifetime() const;

	void set_max_length(real_t p_max_length);
	real_t get_max_length() const;

	void set_pin_uv(bool p_pin_uv);
	bool get_pin_uv() const;

	real_t get_current_length() const;

private:
	// Ribbon
	bool emitting = true;
	float width = 0.3;
	Ref<Curve> width_curve;
	Color color = Color(1.0, 1.0, 1.0, 1.0);
	Ref<Gradient> color_gradient;
	MaterialMode material_mode = MATERIAL_MODE_MIX;
	MeshAlignment alignment = MESH_ALIGNMENT_BILLBOARD;
	TilingMode tiling_mode = TILING_MODE_LENGTH;
	float tiling_multiplier = 1.0;
	float tiling_offset = 0.0;
	PackedVector3Array points;
	PackedVector3Array normals;
	PackedRealArray _velocities;
	PackedVector3Array tangents;
	Ref<ShaderMaterial> material;

	// Mesh optimization code
	uint32_t mesh_surface_offsets[RSE::ARRAY_MAX];
	PackedByteArray vertex_buffer;
	PackedByteArray attribute_buffer;
	Vector<uint8_t> index_buffer;

	uint32_t vertex_stride = 0;
	uint32_t normal_tangent_stride = 0;
	uint32_t attrib_stride = 0;
	uint32_t skin_stride = 0;
	uint32_t mesh_surface_format = 0;

	bool _needs_rebuilding = false;
	real_t _time = 0.0;
	PackedRealArray _times;
	// Reasonable number that's also a multiple of 3, otherwise the renderer screams at us
	int _last_vertex_count = 600;
	int _last_index_count = 0;
	Ref<ArrayMesh> _mesh;

	static inline Ref<Shader> billboard_additive_shader;
	static inline Ref<Shader> billboard_shader;
	static inline Ref<Shader> local_additive_shader;
	static inline Ref<Shader> local_shader;

	static inline Ref<ShaderMaterial> billboard_additive_material;
	static inline Ref<ShaderMaterial> billboard_material;
	static inline Ref<ShaderMaterial> local_additive_material;
	static inline Ref<ShaderMaterial> local_material;

	real_t min_section_length = 0.2;

	//Trail3D
	real_t lifetime = 0.2;
	real_t max_length = 5.0;
	LimitMode limit_mode = LIMIT_MODE_LIFETIME;
	bool pin_uv = false;
	real_t _last_pinned_u = 0.0;
	Vector3 _previous_position;
	bool _transform_changed = false;

	//Ribbon
	void _do_rebuild();
	real_t _calc_current_length() const;
	void _ensure_material();
	void _init_clear_mesh();
	void _process_trail();
	void _encode_vertex(const Vector3 &p_vertex, int p_index, uint8_t *buffer);
	uint32_t _encode_normal(const Vector3 &p_normal);
	void _write_normal(const uint32_t p_normal, int p_index, uint8_t *buffer);
	void _encode_uv(const Vector2 &p_uv, int p_index, uint8_t *buffer);
	void _encode_color(const Color &p_color, uint8_t *r_color);
	void _write_color(uint8_t *p_color, int p_index, uint8_t *r_buffer);

	Trail3D();
};

VARIANT_ENUM_CAST(Trail3D::LimitMode)
