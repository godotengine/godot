/*************************************************************************/
/*  rasterizer_rd.cpp                                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "rasterizer_rd.h"

#include "core/project_settings.h"

void RasterizerRD::prepare_for_blitting_render_targets() {
	RD::get_singleton()->prepare_screen_for_drawing();
}

void RasterizerRD::blit_render_targets_to_screen(DisplayServer::WindowID p_screen, const BlitToScreen *p_render_targets, int p_amount) {
	RD::DrawListID draw_list = RD::get_singleton()->draw_list_begin_for_screen(p_screen);

	for (int i = 0; i < p_amount; i++) {
		RID texture = storage->render_target_get_texture(p_render_targets[i].render_target);
		ERR_CONTINUE(texture.is_null());
		RID rd_texture = storage->texture_get_rd_texture(texture);
		ERR_CONTINUE(rd_texture.is_null());
		if (!render_target_descriptors.has(rd_texture) || !RD::get_singleton()->uniform_set_is_valid(render_target_descriptors[rd_texture])) {
			Vector<RD::Uniform> uniforms;
			RD::Uniform u;
			u.type = RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE;
			u.binding = 0;
			u.ids.push_back(copy_viewports_sampler);
			u.ids.push_back(rd_texture);
			uniforms.push_back(u);
			RID uniform_set = RD::get_singleton()->uniform_set_create(uniforms, copy_viewports_rd_shader, 0);

			render_target_descriptors[rd_texture] = uniform_set;
		}

		Size2 screen_size(RD::get_singleton()->screen_get_width(p_screen), RD::get_singleton()->screen_get_height(p_screen));

		RD::get_singleton()->draw_list_bind_render_pipeline(draw_list, copy_viewports_rd_pipeline);
		RD::get_singleton()->draw_list_bind_index_array(draw_list, copy_viewports_rd_array);
		RD::get_singleton()->draw_list_bind_uniform_set(draw_list, render_target_descriptors[rd_texture], 0);

		float push_constant[4] = {
			p_render_targets[i].rect.position.x / screen_size.width,
			p_render_targets[i].rect.position.y / screen_size.height,
			p_render_targets[i].rect.size.width / screen_size.width,
			p_render_targets[i].rect.size.height / screen_size.height,
		};
		RD::get_singleton()->draw_list_set_push_constant(draw_list, push_constant, 4 * sizeof(float));
		RD::get_singleton()->draw_list_draw(draw_list, true);
	}

	RD::get_singleton()->draw_list_end();
}

void RasterizerRD::begin_frame(double frame_step) {
	frame++;
	delta = frame_step;
	time += frame_step;

	double time_roll_over = GLOBAL_GET("rendering/limits/time/time_rollover_secs");
	time = Math::fmod(time, time_roll_over);

	canvas->set_time(time);
	scene->set_time(time, frame_step);
}

void RasterizerRD::end_frame(bool p_swap_buffers) {
#ifndef _MSC_VER
#warning TODO: likely pass a bool to swap buffers to avoid display?
#endif
	RD::get_singleton()->swap_buffers(); //probably should pass some bool to avoid display?
}

void RasterizerRD::initialize() {
	{ //create framebuffer copy shader
		RenderingDevice::ShaderStageData vert;
		vert.shader_stage = RenderingDevice::SHADER_STAGE_VERTEX;
		vert.spir_v = RenderingDevice::get_singleton()->shader_compile_from_source(RenderingDevice::SHADER_STAGE_VERTEX,
				"#version 450\n"
				"layout(push_constant, binding = 0, std140) uniform Pos { vec4 dst_rect; } pos;\n"
				"layout(location =0) out vec2 uv;\n"
				"void main() { \n"
				" vec2 base_arr[4] = vec2[](vec2(0.0,0.0),vec2(0.0,1.0),vec2(1.0,1.0),vec2(1.0,0.0));\n"
				" uv = base_arr[gl_VertexIndex];\n"
				" vec2 vtx = pos.dst_rect.xy+uv*pos.dst_rect.zw;\n"
				" gl_Position = vec4(vtx * 2.0 - 1.0,0.0,1.0);\n"
				"}\n");

		RenderingDevice::ShaderStageData frag;
		frag.shader_stage = RenderingDevice::SHADER_STAGE_FRAGMENT;
		frag.spir_v = RenderingDevice::get_singleton()->shader_compile_from_source(RenderingDevice::SHADER_STAGE_FRAGMENT,
				"#version 450\n"
				"layout (location = 0) in vec2 uv;\n"
				"layout (location = 0) out vec4 color;\n"
				"layout (binding = 0) uniform sampler2D src_rt;\n"
				"void main() { color=texture(src_rt,uv); }\n");

		Vector<RenderingDevice::ShaderStageData> source;
		source.push_back(vert);
		source.push_back(frag);
		String error;
		copy_viewports_rd_shader = RD::get_singleton()->shader_create(source);
		if (!copy_viewports_rd_shader.is_valid()) {
			print_line("Failed compilation: " + error);
		}
	}

	{ //create index array for copy shader
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
		copy_viewports_rd_index_buffer = RD::get_singleton()->index_buffer_create(6, RenderingDevice::INDEX_BUFFER_FORMAT_UINT32, pv);
		copy_viewports_rd_array = RD::get_singleton()->index_array_create(copy_viewports_rd_index_buffer, 0, 6);
	}

	{ //pipeline
		copy_viewports_rd_pipeline = RD::get_singleton()->render_pipeline_create(copy_viewports_rd_shader, RD::get_singleton()->screen_get_framebuffer_format(), RD::INVALID_ID, RD::RENDER_PRIMITIVE_TRIANGLES, RD::PipelineRasterizationState(), RD::PipelineMultisampleState(), RD::PipelineDepthStencilState(), RenderingDevice::PipelineColorBlendState::create_disabled(), 0);
	}
	{ // sampler
		copy_viewports_sampler = RD::get_singleton()->sampler_create(RD::SamplerState());
	}
}

ThreadWorkPool RasterizerRD::thread_work_pool;
uint64_t RasterizerRD::frame = 1;

void RasterizerRD::finalize() {
	thread_work_pool.finish();

	memdelete(scene);
	memdelete(canvas);
	memdelete(storage);

	//only need to erase these, the rest are erased by cascade
	RD::get_singleton()->free(copy_viewports_rd_index_buffer);
	RD::get_singleton()->free(copy_viewports_rd_shader);
	RD::get_singleton()->free(copy_viewports_sampler);
}

RasterizerRD *RasterizerRD::singleton = nullptr;

RasterizerRD::RasterizerRD() {
	singleton = this;
	thread_work_pool.init();
	time = 0;

	storage = memnew(RasterizerStorageRD);
	canvas = memnew(RasterizerCanvasRD(storage));
	scene = memnew(RasterizerSceneHighEndRD(storage));
}
