/**************************************************************************/
/*  ribbon.h                                                              */
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
#include "scene/resources/curve.h"
#include "scene/resources/gradient.h"

class Ribbon : public MeshInstance3D {
	GDCLASS(Ribbon, MeshInstance3D);

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

	enum RibbonMode {
		RIBBON_MODE_TRAIL,
		RIBBON_MODE_BEAM,
		RIBBON_MODE_MAX
	};

	// Ribbon
	enum MeshAlignment {
		MESH_ALIGNMENT_LOCAL,
		MESH_ALIGNMENT_BILLBOARD,
		MESH_ALIGNMENT_MAX
	};

	enum TilingMode {
		TILING_MODE_UNIT,
		TILING_MODE_LENGTH,
		TILING_MAX
	};

	enum MaterialMode {
		MATERIAL_MODE_MIX,
		MATERIAL_MODE_ADD,
		MATERIAL_MODE_CUSTOM,
		MATERIAL_MODE_MAX
	};

	void set_ribbon_mode(RibbonMode p_ribbon_mode);
	RibbonMode get_ribbon_mode() const;

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
	MeshAlignment get_alignment() const;

	void set_tiling_mode(TilingMode p_tiling_mode);
	TilingMode get_tiling_mode() const;

	void set_tiling_multiplier(float p_tiling_multiplier);
	float get_tiling_multiplier() const;

	void set_tiling_offset(float p_tiling_offset);
	float get_tiling_offset() const;

	void rebuild();
	void clear();

	// Beam

	void set_beam_length(real_t p_beam_length);
	real_t get_beam_length() const;

	// Trail3D

	enum LimitMode {
		LIMIT_MODE_LIFETIME,
		LIMIT_MODE_MAX_LENGTH,
		LIMIT_MODE_MAX
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
	RibbonMode ribbon_mode = RIBBON_MODE_TRAIL;
	bool emitting = true;
	float width = 1.0;
	Ref<Curve> width_curve = nullptr;
	Color color = Color(1.0, 1.0, 1.0, 1.0);
	Ref<Gradient> color_gradient = nullptr;
	MaterialMode material_mode = MATERIAL_MODE_MIX;
	MeshAlignment alignment = MESH_ALIGNMENT_BILLBOARD;
	TilingMode tiling_mode = TILING_MODE_LENGTH;
	float tiling_multiplier = 1.0;
	float tiling_offset = 0.0;
	PackedVector3Array points;
	PackedVector3Array normals;
	PackedRealArray velocities;
	Ref<ShaderMaterial> material;

	bool _needs_rebuilding = false;
	real_t _time = 0.;
	PackedRealArray _times;

	static inline Ref<Shader> billboard_additive_shader;
	static inline Ref<Shader> billboard_shader;
	static inline Ref<Shader> local_additive_shader;
	static inline Ref<Shader> local_shader;

	static inline Ref<ShaderMaterial> billboard_additive_material;
	static inline Ref<ShaderMaterial> billboard_material;
	static inline Ref<ShaderMaterial> local_additive_material;
	static inline Ref<ShaderMaterial> local_material;

	//Trail and beam
	real_t min_section_length = 0.2;

	//Trail3D
	real_t lifetime = 0.2;
	real_t max_length = 0.1;
	LimitMode limit_mode = LIMIT_MODE_LIFETIME;
	bool pin_uv = false;
	real_t _last_section_speed = 0.;
	real_t _last_pinned_u = 0.0;

	//Beam
	real_t beam_length = 3.0;

	//Ribbon
	void _do_rebuild();
	real_t _calc_current_length() const;
	void _ensure_material();
	void _process_beam();
	void _process_trail(real_t p_delta);

	Ribbon();
	~Ribbon();
};

VARIANT_ENUM_CAST(Ribbon::RibbonMode)
VARIANT_ENUM_CAST(Ribbon::TilingMode)
VARIANT_ENUM_CAST(Ribbon::MeshAlignment)
VARIANT_ENUM_CAST(Ribbon::LimitMode)
VARIANT_ENUM_CAST(Ribbon::MaterialMode)
