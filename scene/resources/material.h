/*************************************************************************/
/*  material.h                                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2016 Juan Linietsky, Ariel Manzur.                 */
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

#include "servers/visual_server.h"
#include "scene/resources/texture.h"
#include "scene/resources/shader.h"
#include "resource.h"
#include "servers/visual/shader_language.h"
#include "self_list.h"
/**
	@author Juan Linietsky <reduzio@gmail.com>
*/

class Material : public Resource {

	OBJ_TYPE(Material,Resource);
	RES_BASE_EXTENSION("mtl");
	OBJ_SAVE_TYPE( Material );

	RID material;
protected:

	_FORCE_INLINE_  RID _get_material() const { return material; }
public:

	virtual RID get_rid() const;
	Material();
	virtual ~Material();
};


class FixedSpatialMaterial : public Material {

	OBJ_TYPE(FixedSpatialMaterial,Material)


public:

	enum TextureParam {
		TEXTURE_ALBEDO,
		TEXTURE_SPECULAR,
		TEXTURE_EMISSION,
		TEXTURE_NORMAL,
		TEXTURE_SHEEN,
		TEXTURE_CLEARCOAT,
		TEXTURE_ANISOTROPY,
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
		FEATURE_SHEEN,
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
		FLAG_MAX
	};

	enum DiffuseMode {
		DIFFUSE_LAMBERT,
		DIFFUSE_LAMBERT_WRAP,
		DIFFUSE_OREN_NAYAR,
		DIFFUSE_BURLEY,
	};

private:
	union MaterialKey {

		struct {
			uint32_t feature_mask : 15;
			uint32_t detail_uv : 1;
			uint32_t blend_mode : 2;
			uint32_t depth_draw_mode : 2;
			uint32_t cull_mode : 2;
			uint32_t flags : 5;
			uint32_t detail_blend_mode : 2;
			uint32_t diffuse_mode : 2;
			uint32_t invalid_key : 1;
		};

		uint32_t key;

		bool operator<(const MaterialKey& p_key) const {
			return key < p_key.key;
		}

	};

	struct ShaderData {
		RID shader;
		int users;
	};

	static Map<MaterialKey,ShaderData> shader_map;

	MaterialKey current_key;

	_FORCE_INLINE_ MaterialKey _compute_key() const {

		MaterialKey mk;
		mk.key=0;
		for(int i=0;i<FEATURE_MAX;i++) {
			if (features[i]) {
				mk.feature_mask|=(1<<i);
			}
		}
		mk.detail_uv=detail_uv;
		mk.blend_mode=blend_mode;
		mk.depth_draw_mode=depth_draw_mode;
		mk.cull_mode=cull_mode;
		for(int i=0;i<FLAG_MAX;i++) {
			if (flags[i]) {
				mk.flags|=(1<<i);
			}
		}
		mk.detail_blend_mode=detail_blend_mode;
		mk.diffuse_mode=diffuse_mode;

		return mk;
	}

	struct ShaderNames {
		StringName albedo;
		StringName specular;
		StringName roughness;
		StringName emission;
		StringName normal_scale;
		StringName sheen;
		StringName sheen_color;
		StringName clearcoat;
		StringName clearcoat_gloss;
		StringName anisotropy;
		StringName height_scale;
		StringName subsurface_scattering;
		StringName refraction;
		StringName refraction_roughness;
		StringName point_size;
		StringName texture_names[TEXTURE_MAX];

	};

	static Mutex *material_mutex;
	static SelfList<FixedSpatialMaterial>::List dirty_materials;
	static ShaderNames* shader_names;

	SelfList<FixedSpatialMaterial> element;

	void _update_shader();
	_FORCE_INLINE_ void _queue_shader_change();
	_FORCE_INLINE_ bool _is_shader_dirty() const;

	Color albedo;
	Color specular;
	float roughness;
	Color emission;
	float normal_scale;
	float sheen;
	Color sheen_color;
	float clearcoat;
	float clearcoat_gloss;
	float anisotropy;
	float height_scale;
	float subsurface_scattering;
	float refraction;
	float refraction_roughness;
	float line_width;
	float point_size;

	DetailUV detail_uv;

	BlendMode blend_mode;
	BlendMode detail_blend_mode;
	DepthDrawMode depth_draw_mode;
	CullMode cull_mode;
	bool flags[FLAG_MAX];
	DiffuseMode diffuse_mode;

	bool features[FEATURE_MAX];

	Ref<Texture> textures[TEXTURE_MAX];

	_FORCE_INLINE_ void _validate_feature(const String& text, Feature feature,PropertyInfo& property) const;

protected:

	static void _bind_methods();
	void _validate_property(PropertyInfo& property) const;

public:


	void set_albedo(const Color& p_albedo);
	Color get_albedo() const;

	void set_specular(const Color& p_specular);
	Color get_specular() const;

	void set_roughness(float p_roughness);
	float get_roughness() const;

	void set_emission(const Color& p_emission);
	Color get_emission() const;

	void set_normal_scale(float p_normal_scale);
	float get_normal_scale() const;

	void set_sheen(float p_sheen);
	float get_sheen() const;

	void set_sheen_color(const Color& p_sheen_color);
	Color get_sheen_color() const;

	void set_clearcoat(float p_clearcoat);
	float get_clearcoat() const;

	void set_clearcoat_gloss(float p_clearcoat_gloss);
	float get_clearcoat_gloss() const;

	void set_anisotropy(float p_anisotropy);
	float get_anisotropy() const;

	void set_height_scale(float p_height_scale);
	float get_height_scale() const;

	void set_subsurface_scattering(float p_subsurface_scattering);
	float get_subsurface_scattering() const;

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

	void set_flag(Flags p_flag,bool p_enabled);
	bool get_flag(Flags p_flag) const;

	void set_texture(TextureParam p_param,const Ref<Texture>& p_texture);
	Ref<Texture> get_texture(TextureParam p_param) const;

	void set_feature(Feature p_feature,bool p_enabled);
	bool get_feature(Feature p_feature) const;

	static void init_shaders();
	static void finish_shaders();
	static void flush_changes();

	FixedSpatialMaterial();
	virtual ~FixedSpatialMaterial();
};

VARIANT_ENUM_CAST( FixedSpatialMaterial::TextureParam )
VARIANT_ENUM_CAST( FixedSpatialMaterial::DetailUV )
VARIANT_ENUM_CAST( FixedSpatialMaterial::Feature )
VARIANT_ENUM_CAST( FixedSpatialMaterial::BlendMode )
VARIANT_ENUM_CAST( FixedSpatialMaterial::DepthDrawMode )
VARIANT_ENUM_CAST( FixedSpatialMaterial::CullMode )
VARIANT_ENUM_CAST( FixedSpatialMaterial::Flags )
VARIANT_ENUM_CAST( FixedSpatialMaterial::DiffuseMode )

//////////////////////






#endif
