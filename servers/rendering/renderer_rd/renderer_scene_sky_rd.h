/*************************************************************************/
/*  renderer_scene_sky_rd.h                                              */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef RENDERING_SERVER_SCENE_SKY_RD_H
#define RENDERING_SERVER_SCENE_SKY_RD_H

#include "core/templates/rid_owner.h"
#include "servers/rendering/renderer_compositor.h"
#include "servers/rendering/renderer_rd/renderer_scene_environment_rd.h"
#include "servers/rendering/renderer_rd/renderer_storage_rd.h"
#include "servers/rendering/renderer_rd/shaders/sky.glsl.gen.h"
#include "servers/rendering/renderer_scene_render.h"
#include "servers/rendering/rendering_device.h"

// Forward declare RendererSceneRenderRD so we can pass it into some of our methods, these classes are pretty tightly bound
class RendererSceneRenderRD;

class RendererSceneSkyRD {
public:
	enum SkySet {
		SKY_SET_UNIFORMS,
		SKY_SET_MATERIAL,
		SKY_SET_TEXTURES,
		SKY_SET_FOG,
		SKY_SET_MAX
	};

	// Skys need less info from Directional Lights than the normal shaders
	struct SkyDirectionalLightData {
		float direction[3];
		float energy;
		float color[3];
		float size;
		uint32_t enabled;
		uint32_t pad[3];
	};

private:
	RendererStorageRD *storage;
	RD::DataFormat texture_format = RD::DATA_FORMAT_R16G16B16A16_SFLOAT;

	RID index_buffer;
	RID index_array;

	enum SkyTextureSetVersion {
		SKY_TEXTURE_SET_BACKGROUND,
		SKY_TEXTURE_SET_HALF_RES,
		SKY_TEXTURE_SET_QUARTER_RES,
		SKY_TEXTURE_SET_CUBEMAP,
		SKY_TEXTURE_SET_CUBEMAP_HALF_RES,
		SKY_TEXTURE_SET_CUBEMAP_QUARTER_RES,
		SKY_TEXTURE_SET_MAX
	};

	enum SkyVersion {
		SKY_VERSION_BACKGROUND,
		SKY_VERSION_HALF_RES,
		SKY_VERSION_QUARTER_RES,
		SKY_VERSION_CUBEMAP,
		SKY_VERSION_CUBEMAP_HALF_RES,
		SKY_VERSION_CUBEMAP_QUARTER_RES,

		SKY_VERSION_BACKGROUND_MULTIVIEW,
		SKY_VERSION_HALF_RES_MULTIVIEW,
		SKY_VERSION_QUARTER_RES_MULTIVIEW,

		SKY_VERSION_MAX
	};

	struct SkyPushConstant {
		float orientation[12]; // 48 - 48
		float projections[RendererSceneRender::MAX_RENDER_VIEWS][4]; // 2 x 16 - 80
		float position[3]; // 12 - 92
		float multiplier; // 4 - 96
		float time; // 4 - 100
		float luminance_multiplier; // 4 - 104
		float pad[2]; // 8 - 112 // Using pad to align on 16 bytes
		// 128 is the max size of a push constant. We can replace "pad" but we can't add any more.
	};

	struct SkyShaderData : public RendererStorageRD::ShaderData {
		bool valid;
		RID version;

		PipelineCacheRD pipelines[SKY_VERSION_MAX];
		Map<StringName, ShaderLanguage::ShaderNode::Uniform> uniforms;
		Vector<ShaderCompiler::GeneratedCode::Texture> texture_uniforms;

		Vector<uint32_t> ubo_offsets;
		uint32_t ubo_size;

		String path;
		String code;
		Map<StringName, Map<int, RID>> default_texture_params;

		bool uses_time;
		bool uses_position;
		bool uses_half_res;
		bool uses_quarter_res;
		bool uses_light;

		virtual void set_code(const String &p_Code);
		virtual void set_default_texture_param(const StringName &p_name, RID p_texture, int p_index);
		virtual void get_param_list(List<PropertyInfo> *p_param_list) const;
		virtual void get_instance_param_list(List<RendererStorage::InstanceShaderParam> *p_param_list) const;
		virtual bool is_param_texture(const StringName &p_param) const;
		virtual bool is_animated() const;
		virtual bool casts_shadows() const;
		virtual Variant get_default_parameter(const StringName &p_parameter) const;
		virtual RS::ShaderNativeSourceCode get_native_source_code() const;
		SkyShaderData();
		virtual ~SkyShaderData();
	};

	void _render_sky(RD::DrawListID p_list, float p_time, RID p_fb, PipelineCacheRD *p_pipeline, RID p_uniform_set, RID p_texture_set, uint32_t p_view_count, const CameraMatrix *p_projections, const Basis &p_orientation, float p_multiplier, const Vector3 &p_position, float p_luminance_multiplier);

public:
	struct SkySceneState {
		struct UBO {
			uint32_t volumetric_fog_enabled;
			float volumetric_fog_inv_length;
			float volumetric_fog_detail_spread;

			float fog_aerial_perspective;

			float fog_light_color[3];
			float fog_sun_scatter;

			uint32_t fog_enabled;
			float fog_density;

			float z_far;
			uint32_t directional_light_count;
		};

		UBO ubo;

		SkyDirectionalLightData *directional_lights;
		SkyDirectionalLightData *last_frame_directional_lights;
		uint32_t max_directional_lights;
		uint32_t last_frame_directional_light_count;
		RID directional_light_buffer;
		RID uniform_set;
		RID uniform_buffer;
		RID fog_uniform_set;
		RID default_fog_uniform_set;

		RID fog_shader;
		RID fog_material;
		RID fog_only_texture_uniform_set;
	} sky_scene_state;

	struct ReflectionData {
		struct Layer {
			struct Mipmap {
				RID framebuffers[6];
				RID views[6];
				Size2i size;
			};
			Vector<Mipmap> mipmaps; //per-face view
			Vector<RID> views; // per-cubemap view
		};

		struct DownsampleLayer {
			struct Mipmap {
				RID view;
				Size2i size;

				// for mobile only
				RID views[6];
				RID framebuffers[6];
			};
			Vector<Mipmap> mipmaps;
		};

		RID radiance_base_cubemap; //cubemap for first layer, first cubemap
		RID downsampled_radiance_cubemap;
		DownsampleLayer downsampled_layer;
		RID coefficient_buffer;

		bool dirty = true;

		Vector<Layer> layers;

		void clear_reflection_data();
		void update_reflection_data(RendererStorageRD *p_storage, int p_size, int p_mipmaps, bool p_use_array, RID p_base_cube, int p_base_layer, bool p_low_quality, int p_roughness_layers, RD::DataFormat p_texture_format);
		void create_reflection_fast_filter(RendererStorageRD *p_storage, bool p_use_arrays);
		void create_reflection_importance_sample(RendererStorageRD *p_storage, bool p_use_arrays, int p_cube_side, int p_base_layer, uint32_t p_sky_ggx_samples_quality);
		void update_reflection_mipmaps(RendererStorageRD *p_storage, int p_start, int p_end);
	};

	/* Sky shader */

	struct SkyShader {
		SkyShaderRD shader;
		ShaderCompiler compiler;

		RID default_shader;
		RID default_material;
		RID default_shader_rd;
	} sky_shader;

	struct SkyMaterialData : public RendererStorageRD::MaterialData {
		SkyShaderData *shader_data;
		RID uniform_set;
		bool uniform_set_updated;

		virtual void set_render_priority(int p_priority) {}
		virtual void set_next_pass(RID p_pass) {}
		virtual bool update_parameters(const Map<StringName, Variant> &p_parameters, bool p_uniform_dirty, bool p_textures_dirty);
		virtual ~SkyMaterialData();
	};

	struct Sky {
		RID radiance;
		RID half_res_pass;
		RID half_res_framebuffer;
		RID quarter_res_pass;
		RID quarter_res_framebuffer;
		Size2i screen_size;

		RID texture_uniform_sets[SKY_TEXTURE_SET_MAX];
		RID uniform_set;

		RID material;
		RID uniform_buffer;

		int radiance_size = 256;

		RS::SkyMode mode = RS::SKY_MODE_AUTOMATIC;

		ReflectionData reflection;
		bool dirty = false;
		int processing_layer = 0;
		Sky *dirty_list = nullptr;

		//State to track when radiance cubemap needs updating
		SkyMaterialData *prev_material;
		Vector3 prev_position;
		float prev_time;

		void free(RendererStorageRD *p_storage);

		RID get_textures(RendererStorageRD *p_storage, SkyTextureSetVersion p_version, RID p_default_shader_rd);
		bool set_radiance_size(int p_radiance_size);
		bool set_mode(RS::SkyMode p_mode);
		bool set_material(RID p_material);
		Ref<Image> bake_panorama(RendererStorageRD *p_storage, float p_energy, int p_roughness_layers, const Size2i &p_size);
	};

	uint32_t sky_ggx_samples_quality;
	bool sky_use_cubemap_array;
	Sky *dirty_sky_list = nullptr;
	mutable RID_Owner<Sky, true> sky_owner;
	int roughness_layers;

	RendererStorageRD::ShaderData *_create_sky_shader_func();
	static RendererStorageRD::ShaderData *_create_sky_shader_funcs();

	RendererStorageRD::MaterialData *_create_sky_material_func(SkyShaderData *p_shader);
	static RendererStorageRD::MaterialData *_create_sky_material_funcs(RendererStorageRD::ShaderData *p_shader);

	RendererSceneSkyRD();
	void init(RendererStorageRD *p_storage);
	void set_texture_format(RD::DataFormat p_texture_format);
	~RendererSceneSkyRD();

	void setup(RendererSceneEnvironmentRD *p_env, RID p_render_buffers, const CameraMatrix &p_projection, const Transform3D &p_transform, const Size2i p_screen_size, RendererSceneRenderRD *p_scene_render);
	void update(RendererSceneEnvironmentRD *p_env, const CameraMatrix &p_projection, const Transform3D &p_transform, double p_time, float p_luminance_multiplier = 1.0);
	void draw(RendererSceneEnvironmentRD *p_env, bool p_can_continue_color, bool p_can_continue_depth, RID p_fb, uint32_t p_view_count, const CameraMatrix *p_projections, const Transform3D &p_transform, double p_time); // only called by clustered renderer
	void update_res_buffers(RendererSceneEnvironmentRD *p_env, uint32_t p_view_count, const CameraMatrix *p_projections, const Transform3D &p_transform, double p_time, float p_luminance_multiplier = 1.0);
	void draw(RD::DrawListID p_draw_list, RendererSceneEnvironmentRD *p_env, RID p_fb, uint32_t p_view_count, const CameraMatrix *p_projections, const Transform3D &p_transform, double p_time, float p_luminance_multiplier = 1.0);

	void invalidate_sky(Sky *p_sky);
	void update_dirty_skys();

	RID sky_get_material(RID p_sky) const;

	RID allocate_sky_rid();
	void initialize_sky_rid(RID p_rid);
	Sky *get_sky(RID p_sky) const;
	void free_sky(RID p_sky);
	void sky_set_radiance_size(RID p_sky, int p_radiance_size);
	void sky_set_mode(RID p_sky, RS::SkyMode p_mode);
	void sky_set_material(RID p_sky, RID p_material);
	Ref<Image> sky_bake_panorama(RID p_sky, float p_energy, bool p_bake_irradiance, const Size2i &p_size);

	RID sky_get_radiance_texture_rd(RID p_sky) const;
};

#endif /* RENDERING_SERVER_SCENE_SKY_RD_H */
