/**************************************************************************/
/*  deferred_render_lighting.cpp                                          */
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

#include "deferred_render_lighting.h"
#include "render_forward_clustered.h"
#include "servers/rendering/renderer_rd/storage_rd/light_storage.h"
#include "servers/rendering/renderer_rd/storage_rd/material_storage.h"
#include "servers/rendering/renderer_rd/uniform_set_cache_rd.h"

using namespace RendererRD;

void DeferredRendererLighting::init_shader(uint32_t p_max_directional_lights) {
	String defines = "\n#define MAX_DIRECTIONAL_LIGHT_DATA_STRUCTS " + itos(p_max_directional_lights) + "\n";
	defines += "#define SAMPLERS_BINDING_FIRST_INDEX " + itos(SAMPLERS_BINDING_FIRST_INDEX) + "\n";

	// Initialize deferred lighting
	Vector<String> deferred_lighting_modes;
	deferred_lighting_modes.push_back("\n");
	deferred_lighting_modes.push_back("\n#define USE_MULTIVIEW\n");

	deferred_lighting.compute_shader.initialize(deferred_lighting_modes, defines);
	deferred_lighting.shader_version = deferred_lighting.compute_shader.version_create();

	for (int i = 0; i < DEFERRED_LIGHTING_MODE_MAX; i++) {
		deferred_lighting.compute_pipelines[i] = RD::get_singleton()->compute_pipeline_create(deferred_lighting.compute_shader.version_get_shader(deferred_lighting.shader_version, i));
	}
}

DeferredRendererLighting::DeferredRendererLighting() {
	{
		RD::SamplerState sampler;
		sampler.mag_filter = RD::SAMPLER_FILTER_LINEAR;
		sampler.min_filter = RD::SAMPLER_FILTER_LINEAR;
		sampler.enable_compare = true;
		sampler.compare_op = RD::COMPARE_OP_GREATER;
		shadow_sampler = RD::get_singleton()->sampler_create(sampler);
	}
}

DeferredRendererLighting::~DeferredRendererLighting() {
	if (RD::get_singleton()->uniform_set_is_valid(deferred_lighting.uniform_set)) {
		RD::get_singleton()->free(deferred_lighting.uniform_set);
	}

	deferred_lighting.compute_shader.version_free(deferred_lighting.shader_version);
	RD::get_singleton()->free(shadow_sampler);
}

void DeferredRendererLighting::apply_lighting(RenderDataRD *p_render_data, bool p_use_directional_shadow_atlas, const RendererRD::MaterialStorage::Samplers &p_samplers, RS::LightProjectorFilter p_light_projection_filter) {
	MaterialStorage *material_storage = MaterialStorage::get_singleton();
	ERR_FAIL_NULL(material_storage);
	TextureStorage *texture_storage = TextureStorage::get_singleton();
	ERR_FAIL_NULL(texture_storage);
	LightStorage *light_storage = LightStorage::get_singleton();
	ERR_FAIL_NULL(light_storage);
	UniformSetCacheRD *uniform_set_cache = UniformSetCacheRD::get_singleton();
	ERR_FAIL_NULL(uniform_set_cache);

	ERR_FAIL_NULL(p_render_data);
	Ref<RenderSceneBuffersRD> render_buffers = p_render_data->render_buffers;
	ERR_FAIL_COND(render_buffers.is_null());

	// TODO check if there are any lights to process or if we should just exit.

	// Do our lighting passes, ambient is already written into our color buffer.

	RD::get_singleton()->draw_command_begin_label("Deferred Lighting Pass");

	RENDER_TIMESTAMP("Deferred Lighting Pass");

	uint32_t view_count = render_buffers->get_view_count();
	Size2i size = render_buffers->get_internal_size();

	DRPushConstant push_constant;
	push_constant.width = size.x;
	push_constant.height = size.y;

	uint32_t x_groups = (size.x - 1) / 8 + 1;
	uint32_t y_groups = (size.y - 1) / 8 + 1;

	DeferredLightingModes mode = view_count > 1 ? DEFERRED_LIGHTING_MODE_MULTIVIEW : DEFERRED_LIGHTING_MODE_DEFAULT;

	RID shader = deferred_lighting.compute_shader.version_get_shader(deferred_lighting.shader_version, mode);
	ERR_FAIL_COND(shader.is_null());

	// build our uniform set
	thread_local LocalVector<RD::Uniform> uniforms;

	{
		RD::Uniform u;
		u.binding = 0;
		u.uniform_type = RD::UNIFORM_TYPE_SAMPLER;
		u.append_id(shadow_sampler);
		uniforms.push_back(u);
	}
	{
		RD::Uniform u;
		u.binding = 1;
		u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
		u.append_id(RendererRD::LightStorage::get_singleton()->get_omni_light_buffer());
		uniforms.push_back(u);
	}
	{
		RD::Uniform u;
		u.binding = 2;
		u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
		u.append_id(RendererRD::LightStorage::get_singleton()->get_spot_light_buffer());
		uniforms.push_back(u);
	}
	{
		RD::Uniform u;
		u.binding = 3;
		u.uniform_type = RD::UNIFORM_TYPE_UNIFORM_BUFFER;
		u.append_id(RendererRD::LightStorage::get_singleton()->get_directional_light_buffer());
		uniforms.push_back(u);
	}
	{
		RID scene_data_buffer;
		RenderSceneData *scene_data = p_render_data->get_render_scene_data();
		if (scene_data) {
			scene_data_buffer = scene_data->get_uniform_buffer();
		}

		RD::Uniform u;
		u.binding = 4;
		u.uniform_type = RD::UNIFORM_TYPE_UNIFORM_BUFFER;
		u.append_id(scene_data_buffer);
		uniforms.push_back(u);
	}
	{
		RD::Uniform u;
		u.binding = 5;
		u.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
		RID texture;
		if (p_render_data && p_render_data->shadow_atlas.is_valid()) {
			texture = RendererRD::LightStorage::get_singleton()->shadow_atlas_get_texture(p_render_data->shadow_atlas);
		}
		if (!texture.is_valid()) {
			texture = texture_storage->texture_rd_get_default(RendererRD::TextureStorage::DEFAULT_RD_TEXTURE_DEPTH);
		}
		u.append_id(texture);
		uniforms.push_back(u);
	}
	{
		RD::Uniform u;
		u.binding = 6;
		u.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
		if (p_use_directional_shadow_atlas && RendererRD::LightStorage::get_singleton()->directional_shadow_get_texture().is_valid()) {
			u.append_id(RendererRD::LightStorage::get_singleton()->directional_shadow_get_texture());
		} else {
			u.append_id(texture_storage->texture_rd_get_default(RendererRD::TextureStorage::DEFAULT_RD_TEXTURE_DEPTH));
		}
		uniforms.push_back(u);
	}
	{
		// Note, decal atlas is also used for projectors
		RD::Uniform u;
		u.binding = 7;
		u.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
		RID decal_atlas = RendererRD::TextureStorage::get_singleton()->decal_atlas_get_texture();
		u.append_id(decal_atlas);
		uniforms.push_back(u);
	}
	{
		RD::Uniform u;
		u.binding = 8;
		u.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
		RID decal_atlas = RendererRD::TextureStorage::get_singleton()->decal_atlas_get_texture_srgb();
		u.append_id(decal_atlas);
		uniforms.push_back(u);
	}
	{
		RD::Uniform u;
		u.binding = 9;
		u.uniform_type = RD::UNIFORM_TYPE_SAMPLER;
		RID sampler;
		switch (p_light_projection_filter) {
			case RS::LIGHT_PROJECTOR_FILTER_NEAREST: {
				sampler = p_samplers.get_sampler(RS::CANVAS_ITEM_TEXTURE_FILTER_NEAREST, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);
			} break;
			case RS::LIGHT_PROJECTOR_FILTER_LINEAR: {
				sampler = p_samplers.get_sampler(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);
			} break;
			case RS::LIGHT_PROJECTOR_FILTER_NEAREST_MIPMAPS: {
				sampler = p_samplers.get_sampler(RS::CANVAS_ITEM_TEXTURE_FILTER_NEAREST_WITH_MIPMAPS, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);
			} break;
			case RS::LIGHT_PROJECTOR_FILTER_LINEAR_MIPMAPS: {
				sampler = p_samplers.get_sampler(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);
			} break;
			case RS::LIGHT_PROJECTOR_FILTER_NEAREST_MIPMAPS_ANISOTROPIC: {
				sampler = p_samplers.get_sampler(RS::CANVAS_ITEM_TEXTURE_FILTER_NEAREST_WITH_MIPMAPS_ANISOTROPIC, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);
			} break;
			case RS::LIGHT_PROJECTOR_FILTER_LINEAR_MIPMAPS_ANISOTROPIC: {
				sampler = p_samplers.get_sampler(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS_ANISOTROPIC, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);
			} break;
		}

		u.append_id(sampler);
		uniforms.push_back(u);
	}
	{
		RID texture = render_buffers->get_texture(RB_SCOPE_FORWARD_CLUSTERED, RB_TEX_DR_ALBEDO);
		if (!texture.is_valid()) {
			texture = texture_storage->texture_rd_get_default(RendererRD::TextureStorage::DEFAULT_RD_TEXTURE_BLACK);
		}

		RD::Uniform u;
		u.binding = 10;
		u.uniform_type = RD::UNIFORM_TYPE_IMAGE;
		u.append_id(texture);
		uniforms.push_back(u);
	}
	{
		RID texture = render_buffers->get_texture(RB_SCOPE_FORWARD_CLUSTERED, RB_TEX_DR_NORMAL);
		if (!texture.is_valid()) {
			texture = texture_storage->texture_rd_get_default(RendererRD::TextureStorage::DEFAULT_RD_TEXTURE_NORMAL);
		}

		RD::Uniform u;
		u.binding = 11;
		u.uniform_type = RD::UNIFORM_TYPE_IMAGE;
		u.append_id(texture);
		uniforms.push_back(u);
	}
	{
		RID texture = render_buffers->get_texture(RB_SCOPE_FORWARD_CLUSTERED, RB_TEX_DR_ORM);
		if (!texture.is_valid()) {
			texture = texture_storage->texture_rd_get_default(RendererRD::TextureStorage::DEFAULT_RD_TEXTURE_BLACK);
		}

		RD::Uniform u;
		u.binding = 12;
		u.uniform_type = RD::UNIFORM_TYPE_IMAGE;
		u.append_id(texture);
		uniforms.push_back(u);
	}
	{
		RID texture = render_buffers->get_texture(RB_SCOPE_FORWARD_CLUSTERED, RB_TEX_DR_POSITION);
		if (!texture.is_valid()) {
			texture = texture_storage->texture_rd_get_default(RendererRD::TextureStorage::DEFAULT_RD_TEXTURE_BLACK);
		}

		RD::Uniform u;
		u.binding = 13;
		u.uniform_type = RD::UNIFORM_TYPE_IMAGE;
		u.append_id(texture);
		uniforms.push_back(u);
	}

	material_storage->samplers_rd_get_default().append_uniforms(uniforms, SAMPLERS_BINDING_FIRST_INDEX);

	if (RD::get_singleton()->uniform_set_is_valid(deferred_lighting.uniform_set)) {
		RD::get_singleton()->free(deferred_lighting.uniform_set);
	}
	deferred_lighting.uniform_set = RD::get_singleton()->uniform_set_create(uniforms, shader, 0);

	// Process for each eye individually
	for (uint32_t v = 0; v < view_count; v++) {
		bool use_msaa = render_buffers->get_msaa_3d() != RS::VIEWPORT_MSAA_DISABLED;
		RID color = use_msaa ? render_buffers->get_texture_slice(RB_SCOPE_BUFFERS, RB_TEX_COLOR_MSAA, v, 0) : render_buffers->get_internal_texture(v);

		RD::Uniform u_color(RD::UNIFORM_TYPE_IMAGE, 0, Vector<RID>({ color }));

		push_constant.view = v;

		RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();
		RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, deferred_lighting.compute_pipelines[mode]);
		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, deferred_lighting.uniform_set, 0);
		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 1, u_color), 1);

		RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(DRPushConstant));
		RD::get_singleton()->compute_list_dispatch(compute_list, x_groups, y_groups, 1);

		RD::get_singleton()->compute_list_end();
	}

	RD::get_singleton()->draw_command_end_label();
}
