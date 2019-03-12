/*************************************************************************/
/*  rasterizer_gles3.cpp                                                 */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include <glad/glad.h>

#include "rasterizer_gles3.h"

#include "core/os/os.h"
#include "core/project_settings.h"
#include "drivers/gl_context/debug_gl.h"

RasterizerStorage *RasterizerGLES3::get_storage() {

	return storage;
}

RasterizerCanvas *RasterizerGLES3::get_canvas() {

	return canvas;
}

RasterizerScene *RasterizerGLES3::get_scene() {

	return scene;
}

Error RasterizerGLES3::is_viable() {

#ifdef GLES_OVER_GL
	// On Desktop OpenGL, we require OpenGL 3.3
	if (!GLAD_GL_VERSION_3_3) {
		return ERR_UNAVAILABLE;
	}
#else
	// Otherwise we require OpenGL ES 3.0
	if (!GLAD_GL_ES_VERSION_3_0) {
		return ERR_UNAVAILABLE;
	}
#endif

	return OK;
}

void RasterizerGLES3::initialize() {

	print_verbose("Using GLES3 video driver");

	DebugGL::initialize();

	const GLubyte *renderer = glGetString(GL_RENDERER);
	print_line("OpenGL ES 3.0 Renderer: " + String((const char *)renderer));
	storage->initialize();
	canvas->initialize();
	scene->initialize();
}

void RasterizerGLES3::begin_frame(double frame_step) {

	time_total += frame_step;

	if (frame_step == 0) {
		//to avoid hiccups
		frame_step = 0.001;
	}

	double time_roll_over = GLOBAL_GET("rendering/limits/time/time_rollover_secs");
	if (time_total > time_roll_over)
		time_total = 0; //roll over every day (should be customz

	storage->frame.time[0] = time_total;
	storage->frame.time[1] = Math::fmod(time_total, 3600);
	storage->frame.time[2] = Math::fmod(time_total, 900);
	storage->frame.time[3] = Math::fmod(time_total, 60);
	storage->frame.count++;
	storage->frame.delta = frame_step;

	storage->update_dirty_resources();

	storage->info.render_final = storage->info.render;
	storage->info.render.reset();

	scene->iteration();
}

void RasterizerGLES3::set_current_render_target(RID p_render_target) {

	if (!p_render_target.is_valid() && storage->frame.current_rt && storage->frame.clear_request) {
		//handle pending clear request, if the framebuffer was not cleared
		glBindFramebuffer(GL_FRAMEBUFFER, storage->frame.current_rt->fbo);

		glClearColor(
				storage->frame.clear_request_color.r,
				storage->frame.clear_request_color.g,
				storage->frame.clear_request_color.b,
				storage->frame.clear_request_color.a);

		glClear(GL_COLOR_BUFFER_BIT);
	}

	if (p_render_target.is_valid()) {
		RasterizerStorageGLES3::RenderTarget *rt = storage->render_target_owner.getornull(p_render_target);
		storage->frame.current_rt = rt;
		ERR_FAIL_COND(!rt);
		storage->frame.clear_request = false;

		glViewport(0, 0, rt->width, rt->height);

	} else {
		storage->frame.current_rt = NULL;
		storage->frame.clear_request = false;
		glViewport(0, 0, OS::get_singleton()->get_window_size().width, OS::get_singleton()->get_window_size().height);
		glBindFramebuffer(GL_FRAMEBUFFER, RasterizerStorageGLES3::system_fbo);
	}
}

void RasterizerGLES3::restore_render_target() {

	ERR_FAIL_COND(storage->frame.current_rt == NULL);
	RasterizerStorageGLES3::RenderTarget *rt = storage->frame.current_rt;
	glBindFramebuffer(GL_FRAMEBUFFER, rt->fbo);
	glViewport(0, 0, rt->width, rt->height);
}

void RasterizerGLES3::clear_render_target(const Color &p_color) {

	ERR_FAIL_COND(!storage->frame.current_rt);

	storage->frame.clear_request = true;
	storage->frame.clear_request_color = p_color;
}

void RasterizerGLES3::set_boot_image(const Ref<Image> &p_image, const Color &p_color, bool p_scale) {

	if (p_image.is_null() || p_image->empty())
		return;

	begin_frame(0.0);

	int window_w = OS::get_singleton()->get_video_mode(0).width;
	int window_h = OS::get_singleton()->get_video_mode(0).height;

	glBindFramebuffer(GL_FRAMEBUFFER, RasterizerStorageGLES3::system_fbo);
	glViewport(0, 0, window_w, window_h);
	glDisable(GL_BLEND);
	glDepthMask(GL_FALSE);
	if (OS::get_singleton()->get_window_per_pixel_transparency_enabled()) {
		glClearColor(0.0, 0.0, 0.0, 0.0);
	} else {
		glClearColor(p_color.r, p_color.g, p_color.b, 1.0);
	}
	glClear(GL_COLOR_BUFFER_BIT);
	canvas->canvas_begin();

	RID texture = storage->texture_create();
	storage->texture_allocate(texture, p_image->get_width(), p_image->get_height(), 0, p_image->get_format(), VS::TEXTURE_TYPE_2D, VS::TEXTURE_FLAG_FILTER);
	storage->texture_set_data(texture, p_image);

	Rect2 imgrect(0, 0, p_image->get_width(), p_image->get_height());
	Rect2 screenrect;
	if (p_scale) {

		if (window_w > window_h) {
			//scale horizontally
			screenrect.size.y = window_h;
			screenrect.size.x = imgrect.size.x * window_h / imgrect.size.y;
			screenrect.position.x = (window_w - screenrect.size.x) / 2;

		} else {
			//scale vertically
			screenrect.size.x = window_w;
			screenrect.size.y = imgrect.size.y * window_w / imgrect.size.x;
			screenrect.position.y = (window_h - screenrect.size.y) / 2;
		}
	} else {

		screenrect = imgrect;
		screenrect.position += ((Size2(window_w, window_h) - screenrect.size) / 2.0).floor();
	}

	RasterizerStorageGLES3::Texture *t = storage->texture_owner.get(texture);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, t->tex_id);
	canvas->draw_generic_textured_rect(screenrect, Rect2(0, 0, 1, 1));
	glBindTexture(GL_TEXTURE_2D, 0);
	canvas->canvas_end();

	storage->free(texture); // free since it's only one frame that stays there

	end_frame(true);
}

void RasterizerGLES3::blit_render_target_to_screen(RID p_render_target, const Rect2 &p_screen_rect, int p_screen) {

	DEBUG_GL_REGION("blit_render_target_to_screen");

	ERR_FAIL_COND(storage->frame.current_rt);

	RasterizerStorageGLES3::RenderTarget *rt = storage->render_target_owner.getornull(p_render_target);
	ERR_FAIL_COND(!rt);

#if 1

	Size2 win_size = OS::get_singleton()->get_window_size();
	glBindFramebuffer(GL_READ_FRAMEBUFFER, rt->fbo);
	glReadBuffer(GL_COLOR_ATTACHMENT0);
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, RasterizerStorageGLES3::system_fbo);
	glBlitFramebuffer(0, 0, rt->width, rt->height, p_screen_rect.position.x, win_size.height - p_screen_rect.position.y - p_screen_rect.size.height, p_screen_rect.position.x + p_screen_rect.size.width, win_size.height - p_screen_rect.position.y, GL_COLOR_BUFFER_BIT, GL_NEAREST);

#else
	canvas->canvas_begin();
	glDisable(GL_BLEND);
	glBindFramebuffer(GL_FRAMEBUFFER, RasterizerStorageGLES3::system_fbo);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, rt->color);
	//glBindTexture(GL_TEXTURE_2D, rt->effects.mip_maps[0].color);
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, storage->resources.normal_tex);

	canvas->draw_generic_textured_rect(p_screen_rect, Rect2(0, 0, 1, -1));
	glBindTexture(GL_TEXTURE_2D, 0);
	canvas->canvas_end();
#endif
}

void RasterizerGLES3::output_lens_distorted_to_screen(RID p_render_target, const Rect2 &p_screen_rect, float p_k1, float p_k2, const Vector2 &p_eye_center, float p_oversample) {
	ERR_FAIL_COND(storage->frame.current_rt);

	RasterizerStorageGLES3::RenderTarget *rt = storage->render_target_owner.getornull(p_render_target);
	ERR_FAIL_COND(!rt);

	glDisable(GL_BLEND);

	// render to our framebuffer
	glBindFramebuffer(GL_FRAMEBUFFER, RasterizerStorageGLES3::system_fbo);

	// output our texture
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, rt->color);

	canvas->draw_lens_distortion_rect(p_screen_rect, p_k1, p_k2, p_eye_center, p_oversample);

	glBindTexture(GL_TEXTURE_2D, 0);
}

void RasterizerGLES3::end_frame(bool p_swap_buffers) {

	if (OS::get_singleton()->is_layered_allowed()) {
		if (OS::get_singleton()->get_window_per_pixel_transparency_enabled()) {
#if (defined WINDOWS_ENABLED) && !(defined UWP_ENABLED)
			Size2 wndsize = OS::get_singleton()->get_layered_buffer_size();
			uint8_t *data = OS::get_singleton()->get_layered_buffer_data();
			if (data) {
				glReadPixels(0, 0, wndsize.x, wndsize.y, GL_BGRA, GL_UNSIGNED_BYTE, data);
				OS::get_singleton()->swap_layered_buffer();

				return;
			}
#endif
		} else {
			//clear alpha
			glColorMask(false, false, false, true);
			glClearColor(0, 0, 0, 1);
			glClear(GL_COLOR_BUFFER_BIT);
			glColorMask(true, true, true, true);
		}
	}

	if (p_swap_buffers)
		OS::get_singleton()->swap_buffers();
	else
		glFinish();
}

void RasterizerGLES3::finalize() {

	storage->finalize();
	canvas->finalize();
}

Rasterizer *RasterizerGLES3::_create_current() {

	return memnew(RasterizerGLES3);
}

void RasterizerGLES3::make_current() {
	_create_func = _create_current;
}

void RasterizerGLES3::register_config() {

	GLOBAL_DEF("rendering/quality/filters/anisotropic_filter_level", 4);
	ProjectSettings::get_singleton()->set_custom_property_info("rendering/quality/filters/anisotropic_filter_level", PropertyInfo(Variant::INT, "rendering/quality/filters/anisotropic_filter_level", PROPERTY_HINT_RANGE, "1,16,1"));
	GLOBAL_DEF("rendering/limits/time/time_rollover_secs", 3600);
	ProjectSettings::get_singleton()->set_custom_property_info("rendering/limits/time/time_rollover_secs", PropertyInfo(Variant::REAL, "rendering/limits/time/time_rollover_secs", PROPERTY_HINT_RANGE, "0,10000,1,or_greater"));
}

RasterizerGLES3::RasterizerGLES3() {

	storage = memnew(RasterizerStorageGLES3);
	canvas = memnew(RasterizerCanvasGLES3);
	scene = memnew(RasterizerSceneGLES3);
	canvas->storage = storage;
	canvas->scene_render = scene;
	storage->canvas = canvas;
	scene->storage = storage;
	storage->scene = scene;

	time_total = 0;
}

RasterizerGLES3::~RasterizerGLES3() {

	memdelete(storage);
	memdelete(canvas);
	memdelete(scene);
}
