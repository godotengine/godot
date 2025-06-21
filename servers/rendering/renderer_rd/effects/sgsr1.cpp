/**************************************************************************/
/*  sgsr1.cpp                                                             */
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

#include "sgsr1.h"

#include "servers/rendering/renderer_rd/framebuffer_cache_rd.h"
#include "servers/rendering/renderer_rd/storage_rd/material_storage.h"
#include "servers/rendering/renderer_rd/uniform_set_cache_rd.h"

using namespace RendererRD;

SGSR1::SGSR1() {
	Vector<String> sgsr1_upscale_modes;
	sgsr1_upscale_modes.push_back("\n#define MODE_SGSR1_UPSCALE_NORMAL\n");
	sgsr1_upscale_modes.push_back("\n#define MODE_SGSR1_UPSCALE_FALLBACK\n");
	sgsr1_shader.initialize(sgsr1_upscale_modes);

	shader_version = sgsr1_shader.version_create();

	for (int i = 0; i < SGSR1_SHADER_VARIANT_MAX; i++) {
		pipelines[i].setup(
				sgsr1_shader.version_get_shader(shader_version, i),
				RD::RENDER_PRIMITIVE_TRIANGLES,
				RD::PipelineRasterizationState(),
				RD::PipelineMultisampleState(),
				RD::PipelineDepthStencilState(),
				RD::PipelineColorBlendState::create_disabled(),
				0);
	}

	if (RD::get_singleton()->has_feature(RD::SUPPORTS_HALF_FLOAT)) {
		current_variant = SGSR1_SHADER_VARIANT_NORMAL;
	} else {
		current_variant = SGSR1_SHADER_VARIANT_FALLBACK;
	}

	viewport_info_ubo = RD::get_singleton()->uniform_buffer_create(sizeof(ViewportInfoUBO));
}

SGSR1::~SGSR1() {
	sgsr1_shader.version_free(shader_version);
	RD::get_singleton()->free(viewport_info_ubo);
}

void SGSR1::process(Ref<RenderSceneBuffersRD> p_render_buffers,
		RID p_source_rd_texture,
		RID p_destination_texture) {
	UniformSetCacheRD *uniform_set_cache = UniformSetCacheRD::get_singleton();
	ERR_FAIL_NULL(uniform_set_cache);
	MaterialStorage *material_storage = MaterialStorage::get_singleton();
	ERR_FAIL_NULL(material_storage);

	Size2i internal_size = p_render_buffers->get_internal_size();

	ViewportInfoUBO viewport_ubo;
	viewport_ubo.data[0] = 1.0f / internal_size.x;
	viewport_ubo.data[1] = 1.0f / internal_size.y;
	viewport_ubo.data[2] = internal_size.x;
	viewport_ubo.data[3] = internal_size.y;

	RD::get_singleton()->buffer_update(viewport_info_ubo, 0, sizeof(ViewportInfoUBO), &viewport_ubo);

	RID shader = sgsr1_shader.version_get_shader(shader_version, current_variant);
	ERR_FAIL_COND(shader.is_null());

	RID framebuffer = FramebufferCacheRD::get_singleton()->get_cache(p_destination_texture);
	RD::FramebufferFormatID framebuffer_format = RD::get_singleton()->framebuffer_get_format(framebuffer);

	RID default_sampler = material_storage->sampler_rd_get_default(
			RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR,
			RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);

	RD::Uniform u_viewport_ubo;
	u_viewport_ubo.uniform_type = RD::UNIFORM_TYPE_UNIFORM_BUFFER;
	u_viewport_ubo.binding = 0;
	u_viewport_ubo.append_id(viewport_info_ubo);

	RD::Uniform u_source_texture;
	u_source_texture.uniform_type = RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE;
	u_source_texture.binding = 1;
	u_source_texture.append_id(default_sampler);
	u_source_texture.append_id(p_source_rd_texture);

	RID uniform_set = uniform_set_cache->get_cache(shader, 0, u_viewport_ubo, u_source_texture);

	RD::DrawListID draw_list = RD::get_singleton()->draw_list_begin(framebuffer);
	RD::get_singleton()->draw_list_bind_render_pipeline(draw_list, pipelines[current_variant].get_render_pipeline(RD::INVALID_ID, framebuffer_format, false, RD::get_singleton()->draw_list_get_current_pass()));
	RD::get_singleton()->draw_list_bind_uniform_set(draw_list, uniform_set, 0);
	RD::get_singleton()->draw_list_draw(draw_list, false, 1u, 3u);
	RD::get_singleton()->draw_list_end();
}
