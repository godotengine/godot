/*************************************************************************/
/*  renderer_compositor_rd.cpp                                           */
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

#include "renderer_compositor_rd.h"

#include "core/config/project_settings.h"
#include "core/io/dir_access.h"

void RendererCompositorRD::prepare_for_blitting_render_targets() {
	RD::get_singleton()->prepare_screen_for_drawing();
}

void RendererCompositorRD::blit_render_targets_to_screen(DisplayServer::WindowID p_screen, const BlitToScreen *p_render_targets, int p_amount) {
	RD::DrawListID draw_list = RD::get_singleton()->draw_list_begin_for_screen(p_screen);

	for (int i = 0; i < p_amount; i++) {
		RID texture = storage->render_target_get_texture(p_render_targets[i].render_target);
		ERR_CONTINUE(texture.is_null());
		RID rd_texture = storage->texture_get_rd_texture(texture);
		ERR_CONTINUE(rd_texture.is_null());

		// TODO if keep_3d_linear was set when rendering to this render target we need to add a linear->sRGB conversion in.

		if (!render_target_descriptors.has(rd_texture) || !RD::get_singleton()->uniform_set_is_valid(render_target_descriptors[rd_texture])) {
			Vector<RD::Uniform> uniforms;
			RD::Uniform u;
			u.uniform_type = RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE;
			u.binding = 0;
			u.ids.push_back(blit.sampler);
			u.ids.push_back(rd_texture);
			uniforms.push_back(u);
			RID uniform_set = RD::get_singleton()->uniform_set_create(uniforms, blit.shader.version_get_shader(blit.shader_version, BLIT_MODE_NORMAL), 0);

			render_target_descriptors[rd_texture] = uniform_set;
		}

		Size2 screen_size(RD::get_singleton()->screen_get_width(p_screen), RD::get_singleton()->screen_get_height(p_screen));
		BlitMode mode = p_render_targets[i].lens_distortion.apply ? BLIT_MODE_LENS : (p_render_targets[i].multi_view.use_layer ? BLIT_MODE_USE_LAYER : BLIT_MODE_NORMAL);
		RD::get_singleton()->draw_list_bind_render_pipeline(draw_list, blit.pipelines[mode]);
		RD::get_singleton()->draw_list_bind_index_array(draw_list, blit.array);
		RD::get_singleton()->draw_list_bind_uniform_set(draw_list, render_target_descriptors[rd_texture], 0);

		blit.push_constant.src_rect[0] = p_render_targets[i].src_rect.position.x;
		blit.push_constant.src_rect[1] = p_render_targets[i].src_rect.position.y;
		blit.push_constant.src_rect[2] = p_render_targets[i].src_rect.size.width;
		blit.push_constant.src_rect[3] = p_render_targets[i].src_rect.size.height;
		blit.push_constant.dst_rect[0] = p_render_targets[i].dst_rect.position.x / screen_size.width;
		blit.push_constant.dst_rect[1] = p_render_targets[i].dst_rect.position.y / screen_size.height;
		blit.push_constant.dst_rect[2] = p_render_targets[i].dst_rect.size.width / screen_size.width;
		blit.push_constant.dst_rect[3] = p_render_targets[i].dst_rect.size.height / screen_size.height;
		blit.push_constant.layer = p_render_targets[i].multi_view.layer;
		blit.push_constant.eye_center[0] = p_render_targets[i].lens_distortion.eye_center.x;
		blit.push_constant.eye_center[1] = p_render_targets[i].lens_distortion.eye_center.y;
		blit.push_constant.k1 = p_render_targets[i].lens_distortion.k1;
		blit.push_constant.k2 = p_render_targets[i].lens_distortion.k2;
		blit.push_constant.upscale = p_render_targets[i].lens_distortion.upscale;
		blit.push_constant.aspect_ratio = p_render_targets[i].lens_distortion.aspect_ratio;

		RD::get_singleton()->draw_list_set_push_constant(draw_list, &blit.push_constant, sizeof(BlitPushConstant));
		RD::get_singleton()->draw_list_draw(draw_list, true);
	}

	RD::get_singleton()->draw_list_end();
}

void RendererCompositorRD::begin_frame(double frame_step) {
	frame++;
	delta = frame_step;
	time += frame_step;

	double time_roll_over = GLOBAL_GET("rendering/limits/time/time_rollover_secs");
	time = Math::fmod(time, time_roll_over);

	canvas->set_time(time);
	scene->set_time(time, frame_step);
}

void RendererCompositorRD::end_frame(bool p_swap_buffers) {
#ifndef _MSC_VER
#warning TODO: likely pass a bool to swap buffers to avoid display?
#endif
	RD::get_singleton()->swap_buffers(); //probably should pass some bool to avoid display?
}

void RendererCompositorRD::initialize() {
	{
		// Initialize blit
		Vector<String> blit_modes;
		blit_modes.push_back("\n");
		blit_modes.push_back("\n#define USE_LAYER\n");
		blit_modes.push_back("\n#define USE_LAYER\n#define APPLY_LENS_DISTORTION\n");
		blit_modes.push_back("\n");

		blit.shader.initialize(blit_modes);

		blit.shader_version = blit.shader.version_create();

		for (int i = 0; i < BLIT_MODE_MAX; i++) {
			blit.pipelines[i] = RD::get_singleton()->render_pipeline_create(blit.shader.version_get_shader(blit.shader_version, i), RD::get_singleton()->screen_get_framebuffer_format(), RD::INVALID_ID, RD::RENDER_PRIMITIVE_TRIANGLES, RD::PipelineRasterizationState(), RD::PipelineMultisampleState(), RD::PipelineDepthStencilState(), i == BLIT_MODE_NORMAL_ALPHA ? RenderingDevice::PipelineColorBlendState::create_blend() : RenderingDevice::PipelineColorBlendState::create_disabled(), 0);
		}

		//create index array for copy shader
		Vector<uint8_t> pv;
		pv.resize(6 * 4);
		{
			uint8_t *w = pv.ptrw();
			int *p32 = (int *)w;
			p32[0] = 0;
			p32[1] = 1;
			p32[2] = 2;
			p32[3] = 0;
			p32[4] = 2;
			p32[5] = 3;
		}
		blit.index_buffer = RD::get_singleton()->index_buffer_create(6, RenderingDevice::INDEX_BUFFER_FORMAT_UINT32, pv);
		blit.array = RD::get_singleton()->index_array_create(blit.index_buffer, 0, 6);

		blit.sampler = RD::get_singleton()->sampler_create(RD::SamplerState());
	}
}

uint64_t RendererCompositorRD::frame = 1;

void RendererCompositorRD::finalize() {
	memdelete(scene);
	memdelete(canvas);
	memdelete(storage);

	//only need to erase these, the rest are erased by cascade
	blit.shader.version_free(blit.shader_version);
	RD::get_singleton()->free(blit.index_buffer);
	RD::get_singleton()->free(blit.sampler);
}

void RendererCompositorRD::set_boot_image(const Ref<Image> &p_image, const Color &p_color, RenderingServer::SplashStretchMode p_stretch_mode, bool p_use_filter) {
	RD::get_singleton()->prepare_screen_for_drawing();

	RID texture = storage->texture_allocate();
	storage->texture_2d_initialize(texture, p_image);
	RID rd_texture = storage->texture_get_rd_texture(texture);

	RID uset;
	{
		Vector<RD::Uniform> uniforms;
		RD::Uniform u;
		u.uniform_type = RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE;
		u.binding = 0;
		u.ids.push_back(blit.sampler);
		u.ids.push_back(rd_texture);
		uniforms.push_back(u);
		uset = RD::get_singleton()->uniform_set_create(uniforms, blit.shader.version_get_shader(blit.shader_version, BLIT_MODE_NORMAL), 0);
	}

	Size2 window_size = DisplayServer::get_singleton()->window_get_size();

	Rect2 imgrect(0, 0, p_image->get_width(), p_image->get_height());
	Rect2 screenrect;
	switch (p_stretch_mode) {
		case RenderingServer::SPLASH_STRETCH_MODE_DISABLED: {
			screenrect = imgrect;
			screenrect.position += ((window_size - screenrect.size) / 2.0).floor();
		} break;
		case RenderingServer::SPLASH_STRETCH_MODE_KEEP: {
			if (window_size.width > window_size.height) {
				// Scale horizontally.
				screenrect.size.y = window_size.height;
				screenrect.size.x = imgrect.size.x * window_size.height / imgrect.size.y;
				screenrect.position.x = (window_size.width - screenrect.size.x) / 2;
			} else {
				// Scale vertically.
				screenrect.size.x = window_size.width;
				screenrect.size.y = imgrect.size.y * window_size.width / imgrect.size.x;
				screenrect.position.y = (window_size.height - screenrect.size.y) / 2;
			}
		} break;
		case RenderingServer::SPLASH_STRETCH_MODE_KEEP_WIDTH: {
			// Scale vertically.
			screenrect.size.x = window_size.width;
			screenrect.size.y = imgrect.size.y * window_size.width / imgrect.size.x;
			screenrect.position.y = (window_size.height - screenrect.size.y) / 2;
		} break;
		case RenderingServer::SPLASH_STRETCH_MODE_KEEP_HEIGHT: {
			// Scale horizontally.
			screenrect.size.y = window_size.height;
			screenrect.size.x = imgrect.size.x * window_size.height / imgrect.size.y;
			screenrect.position.x = (window_size.width - screenrect.size.x) / 2;
		} break;
		case RenderingServer::SPLASH_STRETCH_MODE_COVER: {
			double window_aspect = (double)window_size.width / window_size.height;
			double img_aspect = imgrect.size.x / imgrect.size.y;

			if (window_aspect > img_aspect) {
				// Scale vertically.
				screenrect.size.x = window_size.width;
				screenrect.size.y = imgrect.size.y * window_size.width / imgrect.size.x;
				screenrect.position.y = (window_size.height - screenrect.size.y) / 2;
			} else {
				// Scale horizontally.
				screenrect.size.y = window_size.height;
				screenrect.size.x = imgrect.size.x * window_size.height / imgrect.size.y;
				screenrect.position.x = (window_size.width - screenrect.size.x) / 2;
			}
		} break;
		case RenderingServer::SPLASH_STRETCH_MODE_EXPAND: {
			screenrect.size.x = window_size.width;
			screenrect.size.y = window_size.height;
		} break;
	}

	screenrect.position /= window_size;
	screenrect.size /= window_size;

	RD::DrawListID draw_list = RD::get_singleton()->draw_list_begin_for_screen(DisplayServer::MAIN_WINDOW_ID, p_color);

	RD::get_singleton()->draw_list_bind_render_pipeline(draw_list, blit.pipelines[BLIT_MODE_NORMAL_ALPHA]);
	RD::get_singleton()->draw_list_bind_index_array(draw_list, blit.array);
	RD::get_singleton()->draw_list_bind_uniform_set(draw_list, uset, 0);

	blit.push_constant.src_rect[0] = 0.0;
	blit.push_constant.src_rect[1] = 0.0;
	blit.push_constant.src_rect[2] = 1.0;
	blit.push_constant.src_rect[3] = 1.0;
	blit.push_constant.dst_rect[0] = screenrect.position.x;
	blit.push_constant.dst_rect[1] = screenrect.position.y;
	blit.push_constant.dst_rect[2] = screenrect.size.width;
	blit.push_constant.dst_rect[3] = screenrect.size.height;
	blit.push_constant.layer = 0;
	blit.push_constant.eye_center[0] = 0;
	blit.push_constant.eye_center[1] = 0;
	blit.push_constant.k1 = 0;
	blit.push_constant.k2 = 0;
	blit.push_constant.upscale = 1.0;
	blit.push_constant.aspect_ratio = 1.0;

	RD::get_singleton()->draw_list_set_push_constant(draw_list, &blit.push_constant, sizeof(BlitPushConstant));
	RD::get_singleton()->draw_list_draw(draw_list, true);

	RD::get_singleton()->draw_list_end();

	RD::get_singleton()->swap_buffers();

	storage->free(texture);
}

RendererCompositorRD *RendererCompositorRD::singleton = nullptr;

RendererCompositorRD::RendererCompositorRD() {
	{
		String shader_cache_dir = Engine::get_singleton()->get_shader_cache_path();
		if (shader_cache_dir.is_empty()) {
			shader_cache_dir = "user://";
		}
		DirAccessRef da = DirAccess::open(shader_cache_dir);
		if (!da) {
			ERR_PRINT("Can't create shader cache folder, no shader caching will happen: " + shader_cache_dir);
		} else {
			Error err = da->change_dir("shader_cache");
			if (err != OK) {
				err = da->make_dir("shader_cache");
			}
			if (err != OK) {
				ERR_PRINT("Can't create shader cache folder, no shader caching will happen: " + shader_cache_dir);
			} else {
				shader_cache_dir = shader_cache_dir.plus_file("shader_cache");

				bool shader_cache_enabled = GLOBAL_GET("rendering/shader_compiler/shader_cache/enabled");
				if (!Engine::get_singleton()->is_editor_hint() && !shader_cache_enabled) {
					shader_cache_dir = String(); //disable only if not editor
				}

				if (!shader_cache_dir.is_empty()) {
					bool compress = GLOBAL_GET("rendering/shader_compiler/shader_cache/compress");
					bool use_zstd = GLOBAL_GET("rendering/shader_compiler/shader_cache/use_zstd_compression");
					bool strip_debug = GLOBAL_GET("rendering/shader_compiler/shader_cache/strip_debug");

					ShaderRD::set_shader_cache_dir(shader_cache_dir);
					ShaderRD::set_shader_cache_save_compressed(compress);
					ShaderRD::set_shader_cache_save_compressed_zstd(use_zstd);
					ShaderRD::set_shader_cache_save_debug(!strip_debug);
				}
			}
		}
	}

	singleton = this;
	time = 0;

	storage = memnew(RendererStorageRD);
	canvas = memnew(RendererCanvasRenderRD(storage));

	back_end = (bool)(int)GLOBAL_GET("rendering/vulkan/rendering/back_end");
	uint32_t textures_per_stage = RD::get_singleton()->limit_get(RD::LIMIT_MAX_TEXTURES_PER_SHADER_STAGE);

	if (back_end || textures_per_stage < 48) {
		scene = memnew(RendererSceneRenderImplementation::RenderForwardMobile(storage));
	} else { // back_end == false
		// default to our high end renderer
		scene = memnew(RendererSceneRenderImplementation::RenderForwardClustered(storage));
	}

	scene->init();

	// now we're ready to create our effects,
	storage->init_effects(!scene->_render_buffers_can_be_storage());
}

RendererCompositorRD::~RendererCompositorRD() {
	ShaderRD::set_shader_cache_dir(String());
}
