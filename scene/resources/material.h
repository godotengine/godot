/*************************************************************************/
/*  material.h                                                           */
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

#ifndef MATERIAL_H
#define MATERIAL_H

#include "core/resource.h"
#include "core/self_list.h"
#include "scene/resources/shader.h"
#include "scene/resources/texture.h"
#include "servers/visual/shader_language.h"
#include "servers/visual_server.h"

class Material : public Resource {

	GDCLASS(Material, Resource);
	RES_BASE_EXTENSION("material")
	OBJ_SAVE_TYPE(Material);

	RID material;
	Ref<Material> next_pass;
	int render_priority;

protected:
	_FORCE_INLINE_ RID _get_material() const { return material; }
	static void _bind_methods();
	virtual bool _can_do_next_pass() const { return false; }

	void _validate_property(PropertyInfo &property) const;

public:
	enum {
		RENDER_PRIORITY_MAX = VS::MATERIAL_RENDER_PRIORITY_MAX,
		RENDER_PRIORITY_MIN = VS::MATERIAL_RENDER_PRIORITY_MIN,
	};
	void set_next_pass(const Ref<Material> &p_pass);
	Ref<Material> get_next_pass() const;

	void set_render_priority(int p_priority);
	int get_render_priority() const;

	virtual RID get_rid() const;

	virtual Shader::Mode get_shader_mode() const = 0;
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
	bool property_can_revert(const String &p_name);
	Variant property_get_revert(const String &p_name);

	static void _bind_methods();

	void get_argument_options(const StringName &p_function, int p_idx, List<String> *r_options) const;

	virtual bool _can_do_next_pass() const;

	void _shader_changed();

public:
	void set_shader(const Ref<Shader> &p_shader);
	Ref<Shader> get_shader() const;

	void set_shader_param(const StringName &p_param, const Variant &p_value);
	Variant get_shader_param(const StringName &p_param) const;

	virtual Shader::Mode get_shader_mode() const;

	ShaderMaterial();
	~ShaderMaterial();
};

class SpatialMaterial : public Material {

	GDCLASS(SpatialMaterial, Material);

public:
	enum TextureParam {
		TEXTURE_ALBEDO,
		TEXTURE_METALLIC,
		TEXTURE_ROUGHNESS,
		TEXTURE_EMISSION,
		TEXTURE_NORMAL,
		TEXTURE_RIM,
		TEXTURE_CLEARCOAT,
		TEXTURE_FLOWMAP,
		TEXTURE_AMBIENT_OCCLUSION,
		TEXTURE_DEPTH,
		TEXTURE_SUBSURFACE_SCATTERING,
		TEXTURE_TRANSMISSION,
		TEXTURE_REFRACTION,
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
		FEATURE_DEPTH_MAPPING,
		FEATURE_SUBSURACE_SCATTERING,
		FEATURE_TRANSMISSION,
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
		FLAG_USE_VERTEX_LIGHTING,
		FLAG_DISABLE_DEPTH_TEST,
		FLAG_ALBEDO_FROM_VERTEX_COLOR,
		FLAG_SRGB_VERTEX_COLOR,
		FLAG_USE_POINT_SIZE,
		FLAG_FIXED_SIZE,
		FLAG_BILLBOARD_KEEP_SCALE,
		FLAG_UV1_USE_TRIPLANAR,
		FLAG_UV2_USE_TRIPLANAR,
		FLAG_TRIPLANAR_USE_WORLD,
		FLAG_AO_ON_UV2,
		FLAG_EMISSION_ON_UV2,
		FLAG_USE_ALPHA_SCISSOR,
		FLAG_ALBEDO_TEXTURE_FORCE_SRGB,
		FLAG_DONT_RECEIVE_SHADOWS,
		FLAG_ENSURE_CORRECT_NORMALS,
		FLAG_DISABLE_AMBIENT_LIGHT,
		FLAG_USE_SHADOW_TO_OPACITY,
		FLAG_MAX
	};

	enum DiffuseMode {
		DIFFUSE_BURLEY,
		DIFFUSE_LAMBERT,
		DIFFUSE_LAMBERT_WRAP,
		DIFFUSE_OREN_NAYAR,
		DIFFUSE_TOON,
	};

	enum SpecularMode {
		SPECULAR_SCHLICK_GGX,
		SPECULAR_BLINN,
		SPECULAR_PHONG,
		SPECULAR_TOON,
		SPECULAR_DISABLED,
	};

	enum BillboardMode {
		BILLBOARD_DISABLED,
		BILLBOARD_ENABLED,
		BILLBOARD_FIXED_Y,
		BILLBOARD_PARTICLES,
	};

	enum TextureChannel {
		TEXTURE_CHANNEL_RED,
		TEXTURE_CHANNEL_GREEN,
		TEXTURE_CHANNEL_BLUE,
		TEXTURE_CHANNEL_ALPHA,
		TEXTURE_CHANNEL_GRAYSCALE
	};

	enum EmissionOperator {
		EMISSION_OP_ADD,
		EMISSION_OP_MULTIPLY
	};

	enum DistanceFadeMode {
		DISTANCE_FADE_DISABLED,
		DISTANCE_FADE_PIXEL_ALPHA,
		DISTANCE_FADE_PIXEL_DITHER,
		DISTANCE_FADE_OBJECT_DITHER,
	};

private:
	union MaterialKey {

		struct {
			uint64_t feature_mask : 12;
			uint64_t detail_uv : 1;
			uint64_t blend_mode : 2;
			uint64_t depth_draw_mode : 2;
			uint64_t cull_mode : 2;
			uint64_t flags : 19;
			uint64_t detail_blend_mode : 2;
			uint64_t diffuse_mode : 3;
			uint64_t specular_mode : 3;
			uint64_t invalid_key : 1;
			uint64_t deep_parallax : 1;
			uint64_t billboard_mode : 2;
			uint64_t grow : 1;
			uint64_t proximity_fade : 1;
			uint64_t distance_fade : 2;
			uint64_t emission_op : 1;
			uint64_t texture_metallic : 1;
			uint64_t texture_roughness : 1;
		};

		uint64_t key;

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
				mk.feature_mask |= ((uint64_t)1 << i);
			}
		}
		mk.detail_uv = detail_uv;
		mk.blend_mode = blend_mode;
		mk.depth_draw_mode = depth_draw_mode;
		mk.cull_mode = cull_mode;
		for (int i = 0; i < FLAG_MAX; i++) {
			if (flags[i]) {
				mk.flags |= ((uint64_t)1 << i);
			}
		}
		mk.detail_blend_mode = detail_blend_mode;
		mk.diffuse_mode = diffuse_mode;
		mk.specular_mode = specular_mode;
		mk.billboard_mode = billboard_mode;
		mk.deep_parallax = deep_parallax ? 1 : 0;
		mk.grow = grow_enabled;
		mk.proximity_fade = proximity_fade_enabled;
		mk.distance_fade = distance_fade;
		mk.emission_op = emission_op;
		mk.texture_metallic = textures[TEXTURE_METALLIC].is_valid() ? 1 : 0;
		mk.texture_roughness = textures[TEXTURE_ROUGHNESS].is_valid() ? 1 : 0;

		return mk;
	}

	struct ShaderNames {
		StringName albedo;
		StringName specular;
		StringName metallic;
		StringName roughness;
		StringName emission;
		StringName emission_energy;
		StringName normal_scale;
		StringName rim;
		StringName rim_tint;
		StringName clearcoat;
		StringName clearcoat_gloss;
		StringName anisotropy;
		StringName depth_scale;
		StringName subsurface_scattering_strength;
		StringName transmission;
		StringName refraction;
		StringName point_size;
		StringName uv1_scale;
		StringName uv1_offset;
		StringName uv2_scale;
		StringName uv2_offset;
		StringName particles_anim_h_frames;
		StringName particles_anim_v_frames;
		StringName particles_anim_loop;
		StringName depth_min_layers;
		StringName depth_max_layers;
		StringName depth_flip;
		StringName uv1_blend_sharpness;
		StringName uv2_blend_sharpness;
		StringName grow;
		StringName proximity_fade_distance;
		StringName distance_fade_min;
		StringName distance_fade_max;
		StringName ao_light_affect;

		StringName metallic_texture_channel;
		StringName roughness_texture_channel;
		StringName ao_texture_channel;
		StringName clearcoat_texture_channel;
		StringName rim_texture_channel;
		StringName depth_texture_channel;
		StringName refraction_texture_channel;
		StringName alpha_scissor_threshold;

		StringName texture_names[TEXTURE_MAX];
	};

	static Mutex material_mutex;
	static SelfList<SpatialMaterial>::List *dirty_materials;
	static ShaderNames *shader_names;

	SelfList<SpatialMaterial> element;

	void _update_shader();
	_FORCE_INLINE_ void _queue_shader_change();
	_FORCE_INLINE_ bool _is_shader_dirty() const;

	Color albedo;
	float specular;
	float metallic;
	float roughness;
	Color emission;
	float emission_energy;
	float normal_scale;
	float rim;
	float rim_tint;
	float clearcoat;
	float clearcoat_gloss;
	float anisotropy;
	float depth_scale;
	float subsurface_scattering_strength;
	Color transmission;
	float refraction;
	float line_width;
	float point_size;
	float alpha_scissor_threshold;
	bool grow_enabled;
	float ao_light_affect;
	float grow;
	int particles_anim_h_frames;
	int particles_anim_v_frames;
	bool particles_anim_loop;

	Vector3 uv1_scale;
	Vector3 uv1_offset;
	float uv1_triplanar_sharpness;

	Vector3 uv2_scale;
	Vector3 uv2_offset;
	float uv2_triplanar_sharpness;

	DetailUV detail_uv;

	bool deep_parallax;
	int deep_parallax_min_layers;
	int deep_parallax_max_layers;
	bool depth_parallax_flip_tangent;
	bool depth_parallax_flip_binormal;

	bool proximity_fade_enabled;
	float proximity_fade_distance;

	DistanceFadeMode distance_fade;
	float distance_fade_max_distance;
	float distance_fade_min_distance;

	BlendMode blend_mode;
	BlendMode detail_blend_mode;
	DepthDrawMode depth_draw_mode;
	CullMode cull_mode;
	bool flags[FLAG_MAX];
	SpecularMode specular_mode;
	DiffuseMode diffuse_mode;
	BillboardMode billboard_mode;
	EmissionOperator emission_op;

	TextureChannel metallic_texture_channel;
	TextureChannel roughness_texture_channel;
	TextureChannel ao_texture_channel;
	TextureChannel refraction_texture_channel;

	bool features[FEATURE_MAX];

	Ref<Texture> textures[TEXTURE_MAX];

	_FORCE_INLINE_ void _validate_feature(const String &text, Feature feature, PropertyInfo &property) const;

	static const int MAX_MATERIALS_FOR_2D = 128;

	static Ref<SpatialMaterial> materials_for_2d[MAX_MATERIALS_FOR_2D]; //used by Sprite3D and other stuff

	void _validate_high_end(const String &text, PropertyInfo &property) const;

protected:
	static void _bind_methods();
	void _validate_property(PropertyInfo &property) const;
	virtual bool _can_do_next_pass() const { return true; }

public:
	void set_albedo(const Color &p_albedo);
	Color get_albedo() const;

	void set_specular(float p_specular);
	float get_specular() const;

	void set_metallic(float p_metallic);
	float get_metallic() const;

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

	void set_ao_light_affect(float p_ao_light_affect);
	float get_ao_light_affect() const;

	void set_clearcoat(float p_clearcoat);
	float get_clearcoat() const;

	void set_clearcoat_gloss(float p_clearcoat_gloss);
	float get_clearcoat_gloss() const;

	void set_anisotropy(float p_anisotropy);
	float get_anisotropy() const;

	void set_depth_scale(float p_depth_scale);
	float get_depth_scale() const;

	void set_depth_deep_parallax(bool p_enable);
	bool is_depth_deep_parallax_enabled() const;

	void set_depth_deep_parallax_min_layers(int p_layer);
	int get_depth_deep_parallax_min_layers() const;

	void set_depth_deep_parallax_max_layers(int p_layer);
	int get_depth_deep_parallax_max_layers() const;

	void set_depth_deep_parallax_flip_tangent(bool p_flip);
	bool get_depth_deep_parallax_flip_tangent() const;

	void set_depth_deep_parallax_flip_binormal(bool p_flip);
	bool get_depth_deep_parallax_flip_binormal() const;

	void set_subsurface_scattering_strength(float p_subsurface_scattering_strength);
	float get_subsurface_scattering_strength() const;

	void set_transmission(const Color &p_transmission);
	Color get_transmission() const;

	void set_refraction(float p_refraction);
	float get_refraction() const;

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

	void set_specular_mode(SpecularMode p_mode);
	SpecularMode get_specular_mode() const;

	void set_flag(Flags p_flag, bool p_enabled);
	bool get_flag(Flags p_flag) const;

	void set_texture(TextureParam p_param, const Ref<Texture> &p_texture);
	Ref<Texture> get_texture(TextureParam p_param) const;
	// Used only for shader material conversion
	Ref<Texture> get_texture_by_name(StringName p_name) const;

	void set_feature(Feature p_feature, bool p_enabled);
	bool get_feature(Feature p_feature) const;

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

	void set_billboard_mode(BillboardMode p_mode);
	BillboardMode get_billboard_mode() const;

	void set_particles_anim_h_frames(int p_frames);
	int get_particles_anim_h_frames() const;
	void set_particles_anim_v_frames(int p_frames);
	int get_particles_anim_v_frames() const;

	void set_particles_anim_loop(bool p_loop);
	bool get_particles_anim_loop() const;

	void set_grow_enabled(bool p_enable);
	bool is_grow_enabled() const;

	void set_grow(float p_grow);
	float get_grow() const;

	void set_alpha_scissor_threshold(float p_threshold);
	float get_alpha_scissor_threshold() const;

	void set_on_top_of_alpha();

	void set_proximity_fade(bool p_enable);
	bool is_proximity_fade_enabled() const;

	void set_proximity_fade_distance(float p_distance);
	float get_proximity_fade_distance() const;

	void set_distance_fade(DistanceFadeMode p_mode);
	DistanceFadeMode get_distance_fade() const;

	void set_distance_fade_max_distance(float p_distance);
	float get_distance_fade_max_distance() const;

	void set_distance_fade_min_distance(float p_distance);
	float get_distance_fade_min_distance() const;

	void set_emission_operator(EmissionOperator p_op);
	EmissionOperator get_emission_operator() const;

	void set_metallic_texture_channel(TextureChannel p_channel);
	TextureChannel get_metallic_texture_channel() const;
	void set_roughness_texture_channel(TextureChannel p_channel);
	TextureChannel get_roughness_texture_channel() const;
	void set_ao_texture_channel(TextureChannel p_channel);
	TextureChannel get_ao_texture_channel() const;
	void set_refraction_texture_channel(TextureChannel p_channel);
	TextureChannel get_refraction_texture_channel() const;

	static void init_shaders();
	static void finish_shaders();
	static void flush_changes();

	static RID get_material_rid_for_2d(bool p_shaded, bool p_transparent, bool p_double_sided, bool p_cut_alpha, bool p_opaque_prepass, bool p_billboard = false, bool p_billboard_y = false);

	RID get_shader_rid() const;

	virtual Shader::Mode get_shader_mode() const;

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
VARIANT_ENUM_CAST(SpatialMaterial::TextureChannel)
VARIANT_ENUM_CAST(SpatialMaterial::EmissionOperator)
VARIANT_ENUM_CAST(SpatialMaterial::DistanceFadeMode)

//////////////////////

#endif
