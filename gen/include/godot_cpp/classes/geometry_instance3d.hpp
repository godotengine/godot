/**************************************************************************/
/*  geometry_instance3d.hpp                                               */
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

#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/visual_instance3d.hpp>
#include <godot_cpp/variant/aabb.hpp>
#include <godot_cpp/variant/variant.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Material;
class StringName;

class GeometryInstance3D : public VisualInstance3D {
	GDEXTENSION_CLASS(GeometryInstance3D, VisualInstance3D)

public:
	enum ShadowCastingSetting {
		SHADOW_CASTING_SETTING_OFF = 0,
		SHADOW_CASTING_SETTING_ON = 1,
		SHADOW_CASTING_SETTING_DOUBLE_SIDED = 2,
		SHADOW_CASTING_SETTING_SHADOWS_ONLY = 3,
	};

	enum GIMode {
		GI_MODE_DISABLED = 0,
		GI_MODE_STATIC = 1,
		GI_MODE_DYNAMIC = 2,
	};

	enum LightmapScale {
		LIGHTMAP_SCALE_1X = 0,
		LIGHTMAP_SCALE_2X = 1,
		LIGHTMAP_SCALE_4X = 2,
		LIGHTMAP_SCALE_8X = 3,
		LIGHTMAP_SCALE_MAX = 4,
	};

	enum VisibilityRangeFadeMode {
		VISIBILITY_RANGE_FADE_DISABLED = 0,
		VISIBILITY_RANGE_FADE_SELF = 1,
		VISIBILITY_RANGE_FADE_DEPENDENCIES = 2,
	};

	void set_material_override(const Ref<Material> &p_material);
	Ref<Material> get_material_override() const;
	void set_material_overlay(const Ref<Material> &p_material);
	Ref<Material> get_material_overlay() const;
	void set_cast_shadows_setting(GeometryInstance3D::ShadowCastingSetting p_shadow_casting_setting);
	GeometryInstance3D::ShadowCastingSetting get_cast_shadows_setting() const;
	void set_lod_bias(float p_bias);
	float get_lod_bias() const;
	void set_transparency(float p_transparency);
	float get_transparency() const;
	void set_visibility_range_end_margin(float p_distance);
	float get_visibility_range_end_margin() const;
	void set_visibility_range_end(float p_distance);
	float get_visibility_range_end() const;
	void set_visibility_range_begin_margin(float p_distance);
	float get_visibility_range_begin_margin() const;
	void set_visibility_range_begin(float p_distance);
	float get_visibility_range_begin() const;
	void set_visibility_range_fade_mode(GeometryInstance3D::VisibilityRangeFadeMode p_mode);
	GeometryInstance3D::VisibilityRangeFadeMode get_visibility_range_fade_mode() const;
	void set_instance_shader_parameter(const StringName &p_name, const Variant &p_value);
	Variant get_instance_shader_parameter(const StringName &p_name) const;
	void set_extra_cull_margin(float p_margin);
	float get_extra_cull_margin() const;
	void set_lightmap_texel_scale(float p_scale);
	float get_lightmap_texel_scale() const;
	void set_lightmap_scale(GeometryInstance3D::LightmapScale p_scale);
	GeometryInstance3D::LightmapScale get_lightmap_scale() const;
	void set_gi_mode(GeometryInstance3D::GIMode p_mode);
	GeometryInstance3D::GIMode get_gi_mode() const;
	void set_ignore_occlusion_culling(bool p_ignore_culling);
	bool is_ignoring_occlusion_culling();
	void set_custom_aabb(const AABB &p_aabb);
	AABB get_custom_aabb() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		VisualInstance3D::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(GeometryInstance3D::ShadowCastingSetting);
VARIANT_ENUM_CAST(GeometryInstance3D::GIMode);
VARIANT_ENUM_CAST(GeometryInstance3D::LightmapScale);
VARIANT_ENUM_CAST(GeometryInstance3D::VisibilityRangeFadeMode);

