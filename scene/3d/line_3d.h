/**************************************************************************/
/*  line_3d.h                                                             */
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

#include "scene/3d/mesh_instance_3d.h"
#include "scene/resources/3d/shape_3d.h"
#include "scene/resources/gradient.h"

class Line3D : public MeshInstance3D {
	GDCLASS(Line3D, MeshInstance3D);

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

	enum LineMode {
		LINE_MODE_TRAIL,
		LINE_MODE_BEAM,
		LINE_MODE_MANUAL,
		LINE_MODE_MAX
	};

	// Line3D
	enum MeshAlignment {
		MESH_ALIGNMENT_LOCAL,
		MESH_ALIGNMENT_BILLBOARD,
		MESH_ALIGNMENT_MAX
	};

	enum Tiling {
		TLING_UNIT,
		TILING_LENGTH,
		TILING_MAX
	};

	enum MaterialMode {
		MATERIAL_MODE_MIX,
		MATERIAL_MODE_ADD,
		MATERIAL_MODE_CUSTOM,
		MATERIAL_MODE_MAX
	};

	void set_line_mode(LineMode p_line_mode);
	LineMode get_line_mode() const;

	void set_global_space(bool p_line_mode);
	bool get_global_space() const;

	void set_width(float p_width);
	float get_width() const;

	void set_width_curve(Ref<Curve> p_curve);
	Ref<Curve> get_width_curve();

	void set_color(Color p_color);
	Color get_color() const;

	void set_color_gradient(Ref<Gradient> p_color_gradient);
	Ref<Gradient> get_color_gradient();

	void set_material_mode(MaterialMode p_material_mode);
	MaterialMode get_material_mode() const;

	void set_material(Ref<ShaderMaterial> p_material);
	Ref<ShaderMaterial> get_material();

	void set_mesh_alignment(MeshAlignment p_alignment);
	MeshAlignment get_alignment() const;

	void set_tiling_mode(Tiling p_tiling_mode);
	Tiling get_tiling_mode() const;

	void set_tiling_multiplier(float p_tiling_multiplier);
	float get_tiling_multiplier() const;

	void set_tiling_offset(float p_tiling_offset);
	float get_tiling_offset() const;

	void set_points(PackedVector3Array p_points);
	PackedVector3Array get_points() const;

	void set_normals(PackedVector3Array p_normals);
	PackedVector3Array get_normals() const;

	void rebuild();
	void clear();

	// Beam

	void set_target(Vector3 target);
	Vector3 get_target() const;

	// Trail3D

	enum LimitMode {
		LIMIT_MODE_LIFETIME,
		LIMIT_MODE_MAX_LENGTH,
		LIMIT_MODE_MAX
	};

	void set_limit_mode(LimitMode p_limit_mode);
	LimitMode get_limit_mode() const;

	void set_emitting(bool p_emitting);
	bool get_emitting() const;

	void set_max_section_length(real_t p_max_section_length);
	real_t get_max_section_length() const;

	void set_lifetime(real_t p_lifetime);
	real_t get_lifetime() const;

	void set_max_length(real_t p_max_length);
	real_t get_max_length() const;

	real_t get_current_length();

private:
	// Line3D
	LineMode line_mode;
	bool global_space = true;
	float width;
	Ref<Curve> width_curve;
	Color color;
	Ref<Gradient> color_gradient;
	MaterialMode material_mode;
	MeshAlignment alignment;
	Tiling tiling_mode;
	float tiling_multiplier = 1.0;
	float tiling_offset = 0.0;
	PackedVector3Array points;
	PackedVector3Array normals;
	Ref<ShaderMaterial> material;

	bool _needs_rebuilding = false;
	float _time;
	PackedFloat64Array _times;
	real_t _last_pinned_u = 0.0;

	static inline Ref<Shader> billboard_additive_shader;
	static inline Ref<Shader> billboard_shader;
	static inline Ref<Shader> local_additive_shader;
	static inline Ref<Shader> local_shader;

	static inline Ref<ShaderMaterial> billboard_additive_material;
	static inline Ref<ShaderMaterial> billboard_material;
	static inline Ref<ShaderMaterial> local_additive_material;
	static inline Ref<ShaderMaterial> local_material;

	//Trail and beam
	real_t max_section_length;

	//Trail3D
	bool emitting = true;
	real_t lifetime;
	real_t max_length;
	LimitMode limit_mode = LIMIT_MODE_LIFETIME;
	bool pin_texture = true;

	//Beam
	Vector3 target;

	//Line3D
	void _do_rebuild();
	real_t _calc_current_length();
	void _ensure_material();
	void _process_beam();
	void _process_trail(real_t p_delta);

	Line3D();
	~Line3D();
};

VARIANT_ENUM_CAST(Line3D::LineMode)
VARIANT_ENUM_CAST(Line3D::Tiling)
VARIANT_ENUM_CAST(Line3D::MeshAlignment)
VARIANT_ENUM_CAST(Line3D::LimitMode)
VARIANT_ENUM_CAST(Line3D::MaterialMode)
