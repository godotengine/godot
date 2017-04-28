/*************************************************************************/
/*  material.h                                                           */
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
#ifndef MATERIAL_H
#define MATERIAL_H

#include "resource.h"
#include "scene/resources/shader.h"
#include "scene/resources/texture.h"
#include "self_list.h"
#include "servers/visual/shader_language.h"
#include "servers/visual_server.h"
/**
	@author Juan Linietsky <reduzio@gmail.com>
*/

class Material : public Resource {

	GDCLASS(Material, Resource);
	RES_BASE_EXTENSION("mtl");
	OBJ_SAVE_TYPE(Material);

	RID material;

protected:
	_FORCE_INLINE_ RID _get_material() const { return material; }

public:
	virtual RID get_rid() const;
	Material();
	virtual ~Material();
};

class ShaderMaterial : public Material {

	GDCLASS(ShaderMaterial, Material);
	Ref<Shader> shader;

protected:
	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;

	static void _bind_methods();

	void get_argument_options(const StringName &p_function, int p_idx, List<String> *r_options) const;

public:
	void set_shader(const Ref<Shader> &p_shader);
	Ref<Shader> get_shader() const;

	void set_shader_param(const StringName &p_param, const Variant &p_value);
	Variant get_shader_param(const StringName &p_param) const;

	ShaderMaterial();
	~ShaderMaterial();
};

class SpatialMaterial : public Material {

	GDCLASS(SpatialMaterial, Material)

public:
	enum TextureParam {
		TEXTURE_ALBEDO,
		TEXTURE_SPECULAR,
		TEXTURE_EMISSION,
		TEXTURE_NORMAL,
		TEXTURE_RIM,
		TEXTURE_CLEARCOAT,
		TEXTURE_FLOWMAP,
		TEXTURE_AMBIENT_OCCLUSION,
		TEXTURE_HEIGHT,
		TEXTURE_SUBSURFACE_SCATTERING,
		TEXTURE_REFRACTION,
		TEXTURE_REFRACTION_ROUGHNESS,
		TEXTURE_DETAIL_MASK,
		TEXTURE_DETAIL_ALBEDO,
		TEXTURE_DETAIL_NORMAL,
		TEXTURE_MAX

	};

	enum DetailUV {
		DETAIL_UV_1,
		DETAIL_UV_2
	};

	enum Feature {
		FEATURE_TRANSPARENT,
		FEATURE_EMISSION,
		FEATURE_NORMAL_MAPPING,
		FEATURE_RIM,
		FEATURE_CLEARCOAT,
		FEATURE_ANISOTROPY,
		FEATURE_AMBIENT_OCCLUSION,
		FEATURE_HEIGHT_MAPPING,
		FEATURE_SUBSURACE_SCATTERING,
		FEATURE_REFRACTION,
		FEATURE_DETAIL,
		FEATURE_MAX
	};

	enum BlendMode {
		BLEND_MODE_MIX,
		BLEND_MODE_ADD,
		BLEND_MODE_SUB,
		BLEND_MODE_MUL,
	};

	enum DepthDrawMode {
		DEPTH_DRAW_OPAQUE_ONLY,
		DEPTH_DRAW_ALWAYS,
		DEPTH_DRAW_DISABLED,
		DEPTH_DRAW_ALPHA_OPAQUE_PREPASS

	};

	enum CullMode {
		CULL_BACK,
		CULL_FRONT,
		CULL_DISABLED
	};

	enum Flags {
		FLAG_UNSHADED,
		FLAG_ONTOP,
		FLAG_ALBEDO_FROM_VERTEX_COLOR,
		FLAG_SRGB_VERTEX_COLOR,
		FLAG_USE_POINT_SIZE,
		FLAG_FIXED_SIZE,
		FLAG_MAX
	};

	enum DiffuseMode {
		DIFFUSE_LAMBERT,
		DIFFUSE_LAMBERT_WRAP,
		DIFFUSE_OREN_NAYAR,
		DIFFUSE_BURLEY,
	};

	enum SpecularMode {
		SPECULAR_MODE_METALLIC,
		SPECULAR_MODE_SPECULAR,
	};

	enum BillboardMode {
		BILLBOARD_DISABLED,
		BILLBOARD_ENABLED,
		BILLBOARD_FIXED_Y,
		BILLBOARD_PARTICLES,
	};

private:
	union MaterialKey {

		struct {
			uint32_t feature_mask : 11;
			uint32_t detail_uv : 1;
			uint32_t blend_mode : 2;
			uint32_t depth_draw_mode : 2;
			uint32_t cull_mode : 2;
			uint32_t flags : 6;
			uint32_t detail_blend_mode : 2;
			uint32_t diffuse_mode : 2;
			uint32_t invalid_key : 1;
			uint32_t specular_mode : 1;
			uint32_t billboard_mode : 2;
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
		for (int i = 0; i < FEATURE_MAX; i++) {
			if (features[i]) {
				mk.feature_mask |= (1 << i);
			}
		}
		mk.detail_uv = detail_uv;
		mk.blend_mode = blend_mode;
		mk.depth_draw_mode = depth_draw_mode;
		mk.cull_mode = cull_mode;
		for (int i = 0; i < FLAG_MAX; i++) {
			if (flags[i]) {
				mk.flags |= (1 << i);
			}
		}
		mk.detail_blend_mode = detail_blend_mode;
		mk.diffuse_mode = diffuse_mode;
		mk.specular_mode = specular_mode;
		mk.billboard_mode = billboard_mode;

		return mk;
	}

	struct ShaderNames {
		StringName albedo;
		StringName specular;
		StringName metalness;
		StringName roughness;
		StringName emission;
		StringName emission_energy;
		StringName normal_scale;
		StringName rim;
		StringName rim_tint;
		StringName clearcoat;
		StringName clearcoat_gloss;
		StringName anisotropy;
		StringName height_scale;
		StringName subsurface_scattering_strength;
		StringName refraction;
		StringName refraction_roughness;
		StringName point_size;
		StringName uv1_scale;
		StringName uv1_offset;
		StringName uv2_scale;
		StringName uv2_offset;
		StringName particle_h_frames;
		StringName particle_v_frames;
		StringName particles_anim_loop;
		StringName texture_names[TEXTURE_MAX];
	};

	static Mutex *material_mutex;
	static SelfList<SpatialMaterial>::List dirty_materials;
	static ShaderNames *shader_names;

	SelfList<SpatialMaterial> element;

	void _update_shader();
	_FORCE_INLINE_ void _queue_shader_change();
	_FORCE_INLINE_ bool _is_shader_dirty() const;

	Color albedo;
	Color specular;
	float metalness;
	float roughness;
	Color emission;
	float emission_energy;
	float normal_scale;
	float rim;
	float rim_tint;
	float clearcoat;
	float clearcoat_gloss;
	float anisotropy;
	float height_scale;
	float subsurface_scattering_strength;
	float refraction;
	float refraction_roughness;
	float line_width;
	float point_size;
	int particles_anim_h_frames;
	int particles_anim_v_frames;
	bool particles_anim_loop;

	Vector2 uv1_scale;
	Vector2 uv1_offset;

	Vector2 uv2_scale;
	Vector2 uv2_offset;

	DetailUV detail_uv;

	BlendMode blend_mode;
	BlendMode detail_blend_mode;
	DepthDrawMode depth_draw_mode;
	CullMode cull_mode;
	bool flags[FLAG_MAX];
	DiffuseMode diffuse_mode;
	SpecularMode specular_mode;
	BillboardMode billboard_mode;

	bool features[FEATURE_MAX];

	Ref<Texture> textures[TEXTURE_MAX];

	_FORCE_INLINE_ void _validate_feature(const String &text, Feature feature, PropertyInfo &property) const;

protected:
	static void _bind_methods();
	void _validate_property(PropertyInfo &property) const;

public:
	void set_albedo(const Color &p_albedo);
	Color get_albedo() const;

	void set_specular_mode(SpecularMode p_mode);
	SpecularMode get_specular_mode() const;

	void set_specular(const Color &p_specular);
	Color get_specular() const;

	void set_metalness(float p_metalness);
	float get_metalness() const;

	void set_roughness(float p_roughness);
	float get_roughness() const;

	void set_emission(const Color &p_emission);
	Color get_emission() const;

	void set_emission_energy(float p_emission_energy);
	float get_emission_energy() const;

	void set_normal_scale(float p_normal_scale);
	float get_normal_scale() const;

	void set_rim(float p_rim);
	float get_rim() const;

	void set_rim_tint(float p_rim_tint);
	float get_rim_tint() const;

	void set_clearcoat(float p_clearcoat);
	float get_clearcoat() const;

	void set_clearcoat_gloss(float p_clearcoat_gloss);
	float get_clearcoat_gloss() const;

	void set_anisotropy(float p_anisotropy);
	float get_anisotropy() const;

	void set_height_scale(float p_height_scale);
	float get_height_scale() const;

	void set_subsurface_scattering_strength(float p_strength);
	float get_subsurface_scattering_strength() const;

	void set_refraction(float p_refraction);
	float get_refraction() const;

	void set_refraction_roughness(float p_refraction_roughness);
	float get_refraction_roughness() const;

	void set_line_width(float p_line_width);
	float get_line_width() const;

	void set_point_size(float p_point_size);
	float get_point_size() const;

	void set_detail_uv(DetailUV p_detail_uv);
	DetailUV get_detail_uv() const;

	void set_blend_mode(BlendMode p_mode);
	BlendMode get_blend_mode() const;

	void set_detail_blend_mode(BlendMode p_mode);
	BlendMode get_detail_blend_mode() const;

	void set_depth_draw_mode(DepthDrawMode p_mode);
	DepthDrawMode get_depth_draw_mode() const;

	void set_cull_mode(CullMode p_mode);
	CullMode get_cull_mode() const;

	void set_diffuse_mode(DiffuseMode p_mode);
	DiffuseMode get_diffuse_mode() const;

	void set_flag(Flags p_flag, bool p_enabled);
	bool get_flag(Flags p_flag) const;

	void set_texture(TextureParam p_param, const Ref<Texture> &p_texture);
	Ref<Texture> get_texture(TextureParam p_param) const;

	void set_feature(Feature p_feature, bool p_enabled);
	bool get_feature(Feature p_feature) const;

	void set_uv1_scale(const Vector2 &p_scale);
	Vector2 get_uv1_scale() const;

	void set_uv1_offset(const Vector2 &p_offset);
	Vector2 get_uv1_offset() const;

	void set_uv2_scale(const Vector2 &p_scale);
	Vector2 get_uv2_scale() const;

	void set_uv2_offset(const Vector2 &p_offset);
	Vector2 get_uv2_offset() const;

	void set_billboard_mode(BillboardMode p_mode);
	BillboardMode get_billboard_mode() const;

	void set_particles_anim_h_frames(int p_frames);
	int get_particles_anim_h_frames() const;
	void set_particles_anim_v_frames(int p_frames);
	int get_particles_anim_v_frames() const;

	void set_particles_anim_loop(int p_frames);
	int get_particles_anim_loop() const;

	static void init_shaders();
	static void finish_shaders();
	static void flush_changes();

	SpatialMaterial();
	virtual ~SpatialMaterial();
};

VARIANT_ENUM_CAST(SpatialMaterial::TextureParam)
VARIANT_ENUM_CAST(SpatialMaterial::DetailUV)
VARIANT_ENUM_CAST(SpatialMaterial::Feature)
VARIANT_ENUM_CAST(SpatialMaterial::BlendMode)
VARIANT_ENUM_CAST(SpatialMaterial::DepthDrawMode)
VARIANT_ENUM_CAST(SpatialMaterial::CullMode)
VARIANT_ENUM_CAST(SpatialMaterial::Flags)
VARIANT_ENUM_CAST(SpatialMaterial::DiffuseMode)
VARIANT_ENUM_CAST(SpatialMaterial::SpecularMode)
VARIANT_ENUM_CAST(SpatialMaterial::BillboardMode)

//////////////////////

#endif
