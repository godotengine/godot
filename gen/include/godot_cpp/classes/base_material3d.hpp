/**************************************************************************/
/*  base_material3d.hpp                                                   */
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

#include <godot_cpp/classes/material.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/variant/color.hpp>
#include <godot_cpp/variant/vector3.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Texture2D;

class BaseMaterial3D : public Material {
	GDEXTENSION_CLASS(BaseMaterial3D, Material)

public:
	enum TextureParam {
		TEXTURE_ALBEDO = 0,
		TEXTURE_METALLIC = 1,
		TEXTURE_ROUGHNESS = 2,
		TEXTURE_EMISSION = 3,
		TEXTURE_NORMAL = 4,
		TEXTURE_BENT_NORMAL = 18,
		TEXTURE_RIM = 5,
		TEXTURE_CLEARCOAT = 6,
		TEXTURE_FLOWMAP = 7,
		TEXTURE_AMBIENT_OCCLUSION = 8,
		TEXTURE_HEIGHTMAP = 9,
		TEXTURE_SUBSURFACE_SCATTERING = 10,
		TEXTURE_SUBSURFACE_TRANSMITTANCE = 11,
		TEXTURE_BACKLIGHT = 12,
		TEXTURE_REFRACTION = 13,
		TEXTURE_DETAIL_MASK = 14,
		TEXTURE_DETAIL_ALBEDO = 15,
		TEXTURE_DETAIL_NORMAL = 16,
		TEXTURE_ORM = 17,
		TEXTURE_MAX = 19,
	};

	enum TextureFilter {
		TEXTURE_FILTER_NEAREST = 0,
		TEXTURE_FILTER_LINEAR = 1,
		TEXTURE_FILTER_NEAREST_WITH_MIPMAPS = 2,
		TEXTURE_FILTER_LINEAR_WITH_MIPMAPS = 3,
		TEXTURE_FILTER_NEAREST_WITH_MIPMAPS_ANISOTROPIC = 4,
		TEXTURE_FILTER_LINEAR_WITH_MIPMAPS_ANISOTROPIC = 5,
		TEXTURE_FILTER_MAX = 6,
	};

	enum DetailUV {
		DETAIL_UV_1 = 0,
		DETAIL_UV_2 = 1,
	};

	enum Transparency {
		TRANSPARENCY_DISABLED = 0,
		TRANSPARENCY_ALPHA = 1,
		TRANSPARENCY_ALPHA_SCISSOR = 2,
		TRANSPARENCY_ALPHA_HASH = 3,
		TRANSPARENCY_ALPHA_DEPTH_PRE_PASS = 4,
		TRANSPARENCY_MAX = 5,
	};

	enum ShadingMode {
		SHADING_MODE_UNSHADED = 0,
		SHADING_MODE_PER_PIXEL = 1,
		SHADING_MODE_PER_VERTEX = 2,
		SHADING_MODE_MAX = 3,
	};

	enum Feature {
		FEATURE_EMISSION = 0,
		FEATURE_NORMAL_MAPPING = 1,
		FEATURE_RIM = 2,
		FEATURE_CLEARCOAT = 3,
		FEATURE_ANISOTROPY = 4,
		FEATURE_AMBIENT_OCCLUSION = 5,
		FEATURE_HEIGHT_MAPPING = 6,
		FEATURE_SUBSURFACE_SCATTERING = 7,
		FEATURE_SUBSURFACE_TRANSMITTANCE = 8,
		FEATURE_BACKLIGHT = 9,
		FEATURE_REFRACTION = 10,
		FEATURE_DETAIL = 11,
		FEATURE_BENT_NORMAL_MAPPING = 12,
		FEATURE_MAX = 13,
	};

	enum BlendMode {
		BLEND_MODE_MIX = 0,
		BLEND_MODE_ADD = 1,
		BLEND_MODE_SUB = 2,
		BLEND_MODE_MUL = 3,
		BLEND_MODE_PREMULT_ALPHA = 4,
	};

	enum AlphaAntiAliasing {
		ALPHA_ANTIALIASING_OFF = 0,
		ALPHA_ANTIALIASING_ALPHA_TO_COVERAGE = 1,
		ALPHA_ANTIALIASING_ALPHA_TO_COVERAGE_AND_TO_ONE = 2,
	};

	enum DepthDrawMode {
		DEPTH_DRAW_OPAQUE_ONLY = 0,
		DEPTH_DRAW_ALWAYS = 1,
		DEPTH_DRAW_DISABLED = 2,
	};

	enum DepthTest {
		DEPTH_TEST_DEFAULT = 0,
		DEPTH_TEST_INVERTED = 1,
	};

	enum CullMode {
		CULL_BACK = 0,
		CULL_FRONT = 1,
		CULL_DISABLED = 2,
	};

	enum Flags {
		FLAG_DISABLE_DEPTH_TEST = 0,
		FLAG_ALBEDO_FROM_VERTEX_COLOR = 1,
		FLAG_SRGB_VERTEX_COLOR = 2,
		FLAG_USE_POINT_SIZE = 3,
		FLAG_FIXED_SIZE = 4,
		FLAG_BILLBOARD_KEEP_SCALE = 5,
		FLAG_UV1_USE_TRIPLANAR = 6,
		FLAG_UV2_USE_TRIPLANAR = 7,
		FLAG_UV1_USE_WORLD_TRIPLANAR = 8,
		FLAG_UV2_USE_WORLD_TRIPLANAR = 9,
		FLAG_AO_ON_UV2 = 10,
		FLAG_EMISSION_ON_UV2 = 11,
		FLAG_ALBEDO_TEXTURE_FORCE_SRGB = 12,
		FLAG_DONT_RECEIVE_SHADOWS = 13,
		FLAG_DISABLE_AMBIENT_LIGHT = 14,
		FLAG_USE_SHADOW_TO_OPACITY = 15,
		FLAG_USE_TEXTURE_REPEAT = 16,
		FLAG_INVERT_HEIGHTMAP = 17,
		FLAG_SUBSURFACE_MODE_SKIN = 18,
		FLAG_PARTICLE_TRAILS_MODE = 19,
		FLAG_ALBEDO_TEXTURE_MSDF = 20,
		FLAG_DISABLE_FOG = 21,
		FLAG_DISABLE_SPECULAR_OCCLUSION = 22,
		FLAG_USE_Z_CLIP_SCALE = 23,
		FLAG_USE_FOV_OVERRIDE = 24,
		FLAG_MAX = 25,
	};

	enum DiffuseMode {
		DIFFUSE_BURLEY = 0,
		DIFFUSE_LAMBERT = 1,
		DIFFUSE_LAMBERT_WRAP = 2,
		DIFFUSE_TOON = 3,
	};

	enum SpecularMode {
		SPECULAR_SCHLICK_GGX = 0,
		SPECULAR_TOON = 1,
		SPECULAR_DISABLED = 2,
	};

	enum BillboardMode {
		BILLBOARD_DISABLED = 0,
		BILLBOARD_ENABLED = 1,
		BILLBOARD_FIXED_Y = 2,
		BILLBOARD_PARTICLES = 3,
	};

	enum TextureChannel {
		TEXTURE_CHANNEL_RED = 0,
		TEXTURE_CHANNEL_GREEN = 1,
		TEXTURE_CHANNEL_BLUE = 2,
		TEXTURE_CHANNEL_ALPHA = 3,
		TEXTURE_CHANNEL_GRAYSCALE = 4,
	};

	enum EmissionOperator {
		EMISSION_OP_ADD = 0,
		EMISSION_OP_MULTIPLY = 1,
	};

	enum DistanceFadeMode {
		DISTANCE_FADE_DISABLED = 0,
		DISTANCE_FADE_PIXEL_ALPHA = 1,
		DISTANCE_FADE_PIXEL_DITHER = 2,
		DISTANCE_FADE_OBJECT_DITHER = 3,
	};

	enum StencilMode {
		STENCIL_MODE_DISABLED = 0,
		STENCIL_MODE_OUTLINE = 1,
		STENCIL_MODE_XRAY = 2,
		STENCIL_MODE_CUSTOM = 3,
	};

	enum StencilFlags {
		STENCIL_FLAG_READ = 1,
		STENCIL_FLAG_WRITE = 2,
		STENCIL_FLAG_WRITE_DEPTH_FAIL = 4,
	};

	enum StencilCompare {
		STENCIL_COMPARE_ALWAYS = 0,
		STENCIL_COMPARE_LESS = 1,
		STENCIL_COMPARE_EQUAL = 2,
		STENCIL_COMPARE_LESS_OR_EQUAL = 3,
		STENCIL_COMPARE_GREATER = 4,
		STENCIL_COMPARE_NOT_EQUAL = 5,
		STENCIL_COMPARE_GREATER_OR_EQUAL = 6,
	};

	void set_albedo(const Color &p_albedo);
	Color get_albedo() const;
	void set_transparency(BaseMaterial3D::Transparency p_transparency);
	BaseMaterial3D::Transparency get_transparency() const;
	void set_alpha_antialiasing(BaseMaterial3D::AlphaAntiAliasing p_alpha_aa);
	BaseMaterial3D::AlphaAntiAliasing get_alpha_antialiasing() const;
	void set_alpha_antialiasing_edge(float p_edge);
	float get_alpha_antialiasing_edge() const;
	void set_shading_mode(BaseMaterial3D::ShadingMode p_shading_mode);
	BaseMaterial3D::ShadingMode get_shading_mode() const;
	void set_specular(float p_specular);
	float get_specular() const;
	void set_metallic(float p_metallic);
	float get_metallic() const;
	void set_roughness(float p_roughness);
	float get_roughness() const;
	void set_emission(const Color &p_emission);
	Color get_emission() const;
	void set_emission_energy_multiplier(float p_emission_energy_multiplier);
	float get_emission_energy_multiplier() const;
	void set_emission_intensity(float p_emission_energy_multiplier);
	float get_emission_intensity() const;
	void set_normal_scale(float p_normal_scale);
	float get_normal_scale() const;
	void set_rim(float p_rim);
	float get_rim() const;
	void set_rim_tint(float p_rim_tint);
	float get_rim_tint() const;
	void set_clearcoat(float p_clearcoat);
	float get_clearcoat() const;
	void set_clearcoat_roughness(float p_clearcoat_roughness);
	float get_clearcoat_roughness() const;
	void set_anisotropy(float p_anisotropy);
	float get_anisotropy() const;
	void set_heightmap_scale(float p_heightmap_scale);
	float get_heightmap_scale() const;
	void set_subsurface_scattering_strength(float p_strength);
	float get_subsurface_scattering_strength() const;
	void set_transmittance_color(const Color &p_color);
	Color get_transmittance_color() const;
	void set_transmittance_depth(float p_depth);
	float get_transmittance_depth() const;
	void set_transmittance_boost(float p_boost);
	float get_transmittance_boost() const;
	void set_backlight(const Color &p_backlight);
	Color get_backlight() const;
	void set_refraction(float p_refraction);
	float get_refraction() const;
	void set_point_size(float p_point_size);
	float get_point_size() const;
	void set_detail_uv(BaseMaterial3D::DetailUV p_detail_uv);
	BaseMaterial3D::DetailUV get_detail_uv() const;
	void set_blend_mode(BaseMaterial3D::BlendMode p_blend_mode);
	BaseMaterial3D::BlendMode get_blend_mode() const;
	void set_depth_draw_mode(BaseMaterial3D::DepthDrawMode p_depth_draw_mode);
	BaseMaterial3D::DepthDrawMode get_depth_draw_mode() const;
	void set_depth_test(BaseMaterial3D::DepthTest p_depth_test);
	BaseMaterial3D::DepthTest get_depth_test() const;
	void set_cull_mode(BaseMaterial3D::CullMode p_cull_mode);
	BaseMaterial3D::CullMode get_cull_mode() const;
	void set_diffuse_mode(BaseMaterial3D::DiffuseMode p_diffuse_mode);
	BaseMaterial3D::DiffuseMode get_diffuse_mode() const;
	void set_specular_mode(BaseMaterial3D::SpecularMode p_specular_mode);
	BaseMaterial3D::SpecularMode get_specular_mode() const;
	void set_flag(BaseMaterial3D::Flags p_flag, bool p_enable);
	bool get_flag(BaseMaterial3D::Flags p_flag) const;
	void set_texture_filter(BaseMaterial3D::TextureFilter p_mode);
	BaseMaterial3D::TextureFilter get_texture_filter() const;
	void set_feature(BaseMaterial3D::Feature p_feature, bool p_enable);
	bool get_feature(BaseMaterial3D::Feature p_feature) const;
	void set_texture(BaseMaterial3D::TextureParam p_param, const Ref<Texture2D> &p_texture);
	Ref<Texture2D> get_texture(BaseMaterial3D::TextureParam p_param) const;
	void set_detail_blend_mode(BaseMaterial3D::BlendMode p_detail_blend_mode);
	BaseMaterial3D::BlendMode get_detail_blend_mode() const;
	void set_uv1_scale(const Vector3 &p_scale);
	Vector3 get_uv1_scale() const;
	void set_uv1_offset(const Vector3 &p_offset);
	Vector3 get_uv1_offset() const;
	void set_uv1_triplanar_blend_sharpness(float p_sharpness);
	float get_uv1_triplanar_blend_sharpness() const;
	void set_uv2_scale(const Vector3 &p_scale);
	Vector3 get_uv2_scale() const;
	void set_uv2_offset(const Vector3 &p_offset);
	Vector3 get_uv2_offset() const;
	void set_uv2_triplanar_blend_sharpness(float p_sharpness);
	float get_uv2_triplanar_blend_sharpness() const;
	void set_billboard_mode(BaseMaterial3D::BillboardMode p_mode);
	BaseMaterial3D::BillboardMode get_billboard_mode() const;
	void set_particles_anim_h_frames(int32_t p_frames);
	int32_t get_particles_anim_h_frames() const;
	void set_particles_anim_v_frames(int32_t p_frames);
	int32_t get_particles_anim_v_frames() const;
	void set_particles_anim_loop(bool p_loop);
	bool get_particles_anim_loop() const;
	void set_heightmap_deep_parallax(bool p_enable);
	bool is_heightmap_deep_parallax_enabled() const;
	void set_heightmap_deep_parallax_min_layers(int32_t p_layer);
	int32_t get_heightmap_deep_parallax_min_layers() const;
	void set_heightmap_deep_parallax_max_layers(int32_t p_layer);
	int32_t get_heightmap_deep_parallax_max_layers() const;
	void set_heightmap_deep_parallax_flip_tangent(bool p_flip);
	bool get_heightmap_deep_parallax_flip_tangent() const;
	void set_heightmap_deep_parallax_flip_binormal(bool p_flip);
	bool get_heightmap_deep_parallax_flip_binormal() const;
	void set_grow(float p_amount);
	float get_grow() const;
	void set_emission_operator(BaseMaterial3D::EmissionOperator p_operator);
	BaseMaterial3D::EmissionOperator get_emission_operator() const;
	void set_ao_light_affect(float p_amount);
	float get_ao_light_affect() const;
	void set_alpha_scissor_threshold(float p_threshold);
	float get_alpha_scissor_threshold() const;
	void set_alpha_hash_scale(float p_threshold);
	float get_alpha_hash_scale() const;
	void set_grow_enabled(bool p_enable);
	bool is_grow_enabled() const;
	void set_metallic_texture_channel(BaseMaterial3D::TextureChannel p_channel);
	BaseMaterial3D::TextureChannel get_metallic_texture_channel() const;
	void set_roughness_texture_channel(BaseMaterial3D::TextureChannel p_channel);
	BaseMaterial3D::TextureChannel get_roughness_texture_channel() const;
	void set_ao_texture_channel(BaseMaterial3D::TextureChannel p_channel);
	BaseMaterial3D::TextureChannel get_ao_texture_channel() const;
	void set_refraction_texture_channel(BaseMaterial3D::TextureChannel p_channel);
	BaseMaterial3D::TextureChannel get_refraction_texture_channel() const;
	void set_proximity_fade_enabled(bool p_enabled);
	bool is_proximity_fade_enabled() const;
	void set_proximity_fade_distance(float p_distance);
	float get_proximity_fade_distance() const;
	void set_msdf_pixel_range(float p_range);
	float get_msdf_pixel_range() const;
	void set_msdf_outline_size(float p_size);
	float get_msdf_outline_size() const;
	void set_distance_fade(BaseMaterial3D::DistanceFadeMode p_mode);
	BaseMaterial3D::DistanceFadeMode get_distance_fade() const;
	void set_distance_fade_max_distance(float p_distance);
	float get_distance_fade_max_distance() const;
	void set_distance_fade_min_distance(float p_distance);
	float get_distance_fade_min_distance() const;
	void set_z_clip_scale(float p_scale);
	float get_z_clip_scale() const;
	void set_fov_override(float p_scale);
	float get_fov_override() const;
	void set_stencil_mode(BaseMaterial3D::StencilMode p_stencil_mode);
	BaseMaterial3D::StencilMode get_stencil_mode() const;
	void set_stencil_flags(int32_t p_stencil_flags);
	int32_t get_stencil_flags() const;
	void set_stencil_compare(BaseMaterial3D::StencilCompare p_stencil_compare);
	BaseMaterial3D::StencilCompare get_stencil_compare() const;
	void set_stencil_reference(int32_t p_stencil_reference);
	int32_t get_stencil_reference() const;
	void set_stencil_effect_color(const Color &p_stencil_color);
	Color get_stencil_effect_color() const;
	void set_stencil_effect_outline_thickness(float p_stencil_outline_thickness);
	float get_stencil_effect_outline_thickness() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Material::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(BaseMaterial3D::TextureParam);
VARIANT_ENUM_CAST(BaseMaterial3D::TextureFilter);
VARIANT_ENUM_CAST(BaseMaterial3D::DetailUV);
VARIANT_ENUM_CAST(BaseMaterial3D::Transparency);
VARIANT_ENUM_CAST(BaseMaterial3D::ShadingMode);
VARIANT_ENUM_CAST(BaseMaterial3D::Feature);
VARIANT_ENUM_CAST(BaseMaterial3D::BlendMode);
VARIANT_ENUM_CAST(BaseMaterial3D::AlphaAntiAliasing);
VARIANT_ENUM_CAST(BaseMaterial3D::DepthDrawMode);
VARIANT_ENUM_CAST(BaseMaterial3D::DepthTest);
VARIANT_ENUM_CAST(BaseMaterial3D::CullMode);
VARIANT_ENUM_CAST(BaseMaterial3D::Flags);
VARIANT_ENUM_CAST(BaseMaterial3D::DiffuseMode);
VARIANT_ENUM_CAST(BaseMaterial3D::SpecularMode);
VARIANT_ENUM_CAST(BaseMaterial3D::BillboardMode);
VARIANT_ENUM_CAST(BaseMaterial3D::TextureChannel);
VARIANT_ENUM_CAST(BaseMaterial3D::EmissionOperator);
VARIANT_ENUM_CAST(BaseMaterial3D::DistanceFadeMode);
VARIANT_ENUM_CAST(BaseMaterial3D::StencilMode);
VARIANT_ENUM_CAST(BaseMaterial3D::StencilFlags);
VARIANT_ENUM_CAST(BaseMaterial3D::StencilCompare);

