/*************************************************************************/
/*  rasterizer_gles3.cpp                                                 */
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

#include "rasterizer_gles3.h"

#ifdef GLES3_ENABLED

#include "core/config/project_settings.h"
#include "core/os/os.h"

#define _EXT_DEBUG_OUTPUT_SYNCHRONOUS_ARB 0x8242
#define _EXT_DEBUG_NEXT_LOGGED_MESSAGE_LENGTH_ARB 0x8243
#define _EXT_DEBUG_CALLBACK_FUNCTION_ARB 0x8244
#define _EXT_DEBUG_CALLBACK_USER_PARAM_ARB 0x8245
#define _EXT_DEBUG_SOURCE_API_ARB 0x8246
#define _EXT_DEBUG_SOURCE_WINDOW_SYSTEM_ARB 0x8247
#define _EXT_DEBUG_SOURCE_SHADER_COMPILER_ARB 0x8248
#define _EXT_DEBUG_SOURCE_THIRD_PARTY_ARB 0x8249
#define _EXT_DEBUG_SOURCE_APPLICATION_ARB 0x824A
#define _EXT_DEBUG_SOURCE_OTHER_ARB 0x824B
#define _EXT_DEBUG_TYPE_ERROR_ARB 0x824C
#define _EXT_DEBUG_TYPE_DEPRECATED_BEHAVIOR_ARB 0x824D
#define _EXT_DEBUG_TYPE_UNDEFINED_BEHAVIOR_ARB 0x824E
#define _EXT_DEBUG_TYPE_PORTABILITY_ARB 0x824F
#define _EXT_DEBUG_TYPE_PERFORMANCE_ARB 0x8250
#define _EXT_DEBUG_TYPE_OTHER_ARB 0x8251
#define _EXT_MAX_DEBUG_MESSAGE_LENGTH_ARB 0x9143
#define _EXT_MAX_DEBUG_LOGGED_MESSAGES_ARB 0x9144
#define _EXT_DEBUG_LOGGED_MESSAGES_ARB 0x9145
#define _EXT_DEBUG_SEVERITY_HIGH_ARB 0x9146
#define _EXT_DEBUG_SEVERITY_MEDIUM_ARB 0x9147
#define _EXT_DEBUG_SEVERITY_LOW_ARB 0x9148
#define _EXT_DEBUG_OUTPUT 0x92E0

#ifndef GLAPIENTRY
#if defined(WINDOWS_ENABLED) && !defined(UWP_ENABLED)
#define GLAPIENTRY APIENTRY
#else
#define GLAPIENTRY
#endif
#endif

#if !defined(IPHONE_ENABLED) && !defined(JAVASCRIPT_ENABLED)
// We include EGL below to get debug callback on GLES2 platforms,
// but EGL is not available on iOS.
#define CAN_DEBUG
#endif

#if !defined(GLES_OVER_GL) && defined(CAN_DEBUG)
#include <GLES3/gl3.h>
#include <GLES3/gl3ext.h>
#include <GLES3/gl3platform.h>

#include <EGL/egl.h>
#include <EGL/eglext.h>
#endif

#if defined(MINGW_ENABLED) || defined(_MSC_VER)
#define strcpy strcpy_s
#endif

void RasterizerGLES3::begin_frame(double frame_step) {
	frame++;
	delta = frame_step;

	time_total += frame_step;

	double time_roll_over = GLOBAL_GET("rendering/limits/time/time_rollover_secs");
	time_total = Math::fmod(time_total, time_roll_over);

	storage.frame.time = time_total;
	storage.frame.count++;
	storage.frame.delta = frame_step;

	storage.update_dirty_resources();

	storage.info.render_final = storage.info.render;
	storage.info.render.reset();

	//scene->iteration();
}

void RasterizerGLES3::end_frame(bool p_swap_buffers) {
	//	if (OS::get_singleton()->is_layered_allowed()) {
	//		if (!OS::get_singleton()->get_window_per_pixel_transparency_enabled()) {
	//clear alpha
	//			glColorMask(false, false, false, true);
	//			glClearColor(0.5, 0, 0, 1);
	//			glClear(GL_COLOR_BUFFER_BIT);
	//			glColorMask(true, true, true, true);
	//		}
	//	}

	//	glClearColor(1, 0, 0, 1);
	//	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_ACCUM_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

	if (p_swap_buffers) {
		DisplayServer::get_singleton()->swap_buffers();
	} else {
		glFinish();
	}
}

#ifdef CAN_DEBUG
static void GLAPIENTRY _gl_debug_print(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar *message, const GLvoid *userParam) {
	if (type == _EXT_DEBUG_TYPE_OTHER_ARB)
		return;

	if (type == _EXT_DEBUG_TYPE_PERFORMANCE_ARB)
		return; //these are ultimately annoying, so removing for now

	char debSource[256], debType[256], debSev[256];

	if (source == _EXT_DEBUG_SOURCE_API_ARB)
		strcpy(debSource, "OpenGL");
	else if (source == _EXT_DEBUG_SOURCE_WINDOW_SYSTEM_ARB)
		strcpy(debSource, "Windows");
	else if (source == _EXT_DEBUG_SOURCE_SHADER_COMPILER_ARB)
		strcpy(debSource, "Shader Compiler");
	else if (source == _EXT_DEBUG_SOURCE_THIRD_PARTY_ARB)
		strcpy(debSource, "Third Party");
	else if (source == _EXT_DEBUG_SOURCE_APPLICATION_ARB)
		strcpy(debSource, "Application");
	else if (source == _EXT_DEBUG_SOURCE_OTHER_ARB)
		strcpy(debSource, "Other");

	if (type == _EXT_DEBUG_TYPE_ERROR_ARB)
		strcpy(debType, "Error");
	else if (type == _EXT_DEBUG_TYPE_DEPRECATED_BEHAVIOR_ARB)
		strcpy(debType, "Deprecated behavior");
	else if (type == _EXT_DEBUG_TYPE_UNDEFINED_BEHAVIOR_ARB)
		strcpy(debType, "Undefined behavior");
	else if (type == _EXT_DEBUG_TYPE_PORTABILITY_ARB)
		strcpy(debType, "Portability");
	else if (type == _EXT_DEBUG_TYPE_PERFORMANCE_ARB)
		strcpy(debType, "Performance");
	else if (type == _EXT_DEBUG_TYPE_OTHER_ARB)
		strcpy(debType, "Other");

	if (severity == _EXT_DEBUG_SEVERITY_HIGH_ARB)
		strcpy(debSev, "High");
	else if (severity == _EXT_DEBUG_SEVERITY_MEDIUM_ARB)
		strcpy(debSev, "Medium");
	else if (severity == _EXT_DEBUG_SEVERITY_LOW_ARB)
		strcpy(debSev, "Low");

	String output = String() + "GL ERROR: Source: " + debSource + "\tType: " + debType + "\tID: " + itos(id) + "\tSeverity: " + debSev + "\tMessage: " + message;

	ERR_PRINT(output);
}
#endif

typedef void (*DEBUGPROCARB)(GLenum source,
		GLenum type,
		GLuint id,
		GLenum severity,
		GLsizei length,
		const char *message,
		const void *userParam);

typedef void (*DebugMessageCallbackARB)(DEBUGPROCARB callback, const void *userParam);

void RasterizerGLES3::initialize() {
	print_verbose("Using OpenGL video driver");

	storage._main_thread_id = Thread::get_caller_id();

#ifdef GLAD_ENABLED
	if (!gladLoadGL()) {
		ERR_PRINT("Error initializing GLAD");
		return;
	}
#endif

#ifdef GLAD_ENABLED
	if (OS::get_singleton()->is_stdout_verbose()) {
		if (GLAD_GL_ARB_debug_output) {
			glEnable(_EXT_DEBUG_OUTPUT_SYNCHRONOUS_ARB);
			glDebugMessageCallbackARB(_gl_debug_print, NULL);
			glEnable(_EXT_DEBUG_OUTPUT);
		} else {
			print_line("OpenGL debugging not supported!");
		}
	}
#endif // GLAD_ENABLED

	// For debugging
#ifdef CAN_DEBUG
#ifdef GLES_OVER_GL
	if (OS::get_singleton()->is_stdout_verbose() && GLAD_GL_ARB_debug_output) {
		glDebugMessageControlARB(_EXT_DEBUG_SOURCE_API_ARB, _EXT_DEBUG_TYPE_ERROR_ARB, _EXT_DEBUG_SEVERITY_HIGH_ARB, 0, NULL, GL_TRUE);
		glDebugMessageControlARB(_EXT_DEBUG_SOURCE_API_ARB, _EXT_DEBUG_TYPE_DEPRECATED_BEHAVIOR_ARB, _EXT_DEBUG_SEVERITY_HIGH_ARB, 0, NULL, GL_TRUE);
		glDebugMessageControlARB(_EXT_DEBUG_SOURCE_API_ARB, _EXT_DEBUG_TYPE_UNDEFINED_BEHAVIOR_ARB, _EXT_DEBUG_SEVERITY_HIGH_ARB, 0, NULL, GL_TRUE);
		glDebugMessageControlARB(_EXT_DEBUG_SOURCE_API_ARB, _EXT_DEBUG_TYPE_PORTABILITY_ARB, _EXT_DEBUG_SEVERITY_HIGH_ARB, 0, NULL, GL_TRUE);
		glDebugMessageControlARB(_EXT_DEBUG_SOURCE_API_ARB, _EXT_DEBUG_TYPE_PERFORMANCE_ARB, _EXT_DEBUG_SEVERITY_HIGH_ARB, 0, NULL, GL_TRUE);
		glDebugMessageControlARB(_EXT_DEBUG_SOURCE_API_ARB, _EXT_DEBUG_TYPE_OTHER_ARB, _EXT_DEBUG_SEVERITY_HIGH_ARB, 0, NULL, GL_TRUE);
		//		 glDebugMessageInsertARB(
		//			GL_DEBUG_SOURCE_API_ARB,
		//			GL_DEBUG_TYPE_OTHER_ARB, 1,
		//			GL_DEBUG_SEVERITY_HIGH_ARB, 5, "hello");
	}
#else
	if (OS::get_singleton()->is_stdout_verbose()) {
		DebugMessageCallbackARB callback = (DebugMessageCallbackARB)eglGetProcAddress("glDebugMessageCallback");
		if (!callback) {
			callback = (DebugMessageCallbackARB)eglGetProcAddress("glDebugMessageCallbackKHR");
		}

		if (callback) {
			print_line("godot: ENABLING GL DEBUG");
			glEnable(_EXT_DEBUG_OUTPUT_SYNCHRONOUS_ARB);
			callback(_gl_debug_print, NULL);
			glEnable(_EXT_DEBUG_OUTPUT);
		}
	}
#endif // GLES_OVER_GL
#endif // CAN_DEBUG

	print_line("OpenGL Renderer: " + RS::get_singleton()->get_video_adapter_name());
	storage.initialize();
	canvas.initialize();
	//	scene.initialize();

	// make sure the OS knows to only access the renderer from the main thread
	OS::get_singleton()->set_render_main_thread_mode(OS::RENDER_MAIN_THREAD_ONLY);
}

RasterizerGLES3::RasterizerGLES3() {
	canvas.storage = &storage;
	canvas.scene_render = &scene;
	storage.canvas = &canvas;
	//scene.storage = &storage;
	storage.scene = &scene;
}

void RasterizerGLES3::prepare_for_blitting_render_targets() {
}

void RasterizerGLES3::_blit_render_target_to_screen(RID p_render_target, DisplayServer::WindowID p_screen, const Rect2 &p_screen_rect) {
	ERR_FAIL_COND(storage.frame.current_rt);

	RasterizerStorageGLES3::RenderTarget *rt = storage.render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND(!rt);

	// TODO: do we need a keep 3d linear option?

	if (rt->external.fbo != 0) {
		glBindFramebuffer(GL_READ_FRAMEBUFFER, rt->external.fbo);
	} else {
		glBindFramebuffer(GL_READ_FRAMEBUFFER, rt->fbo);
	}
	glReadBuffer(GL_COLOR_ATTACHMENT0);
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, RasterizerStorageGLES3::system_fbo);
	glBlitFramebuffer(0, 0, rt->width, rt->height, 0, p_screen_rect.size.y, p_screen_rect.size.x, 0, GL_COLOR_BUFFER_BIT, GL_NEAREST);
}

// is this p_screen useless in a multi window environment?
void RasterizerGLES3::blit_render_targets_to_screen(DisplayServer::WindowID p_screen, const BlitToScreen *p_render_targets, int p_amount) {
	// do this once off for all blits
	storage.bind_framebuffer_system();

	storage.frame.current_rt = nullptr;

	for (int i = 0; i < p_amount; i++) {
		const BlitToScreen &blit = p_render_targets[i];

		RID rid_rt = blit.render_target;

		Rect2 dst_rect = blit.dst_rect;
		_blit_render_target_to_screen(rid_rt, p_screen, dst_rect);
	}
}

void RasterizerGLES3::set_boot_image(const Ref<Image> &p_image, const Color &p_color, bool p_scale, bool p_use_filter) {
	if (p_image.is_null() || p_image->is_empty())
		return;

	Size2i win_size = DisplayServer::get_singleton()->screen_get_size();

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glViewport(0, 0, win_size.width, win_size.height);
	glDisable(GL_BLEND);
	glDepthMask(GL_FALSE);
	if (false) {
		//	if (OS::get_singleton()->get_window_per_pixel_transparency_enabled()) {
		glClearColor(0.0, 0.0, 0.0, 0.0);
	} else {
		glClearColor(p_color.r, p_color.g, p_color.b, 1.0);
	}
	glClear(GL_COLOR_BUFFER_BIT);

	canvas.canvas_begin();

	RID texture = storage.texture_create();
	//storage.texture_allocate(texture, p_image->get_width(), p_image->get_height(), 0, p_image->get_format(), VS::TEXTURE_TYPE_2D, p_use_filter ? VS::TEXTURE_FLAG_FILTER : 0);
	storage._texture_allocate_internal(texture, p_image->get_width(), p_image->get_height(), 0, p_image->get_format(), RenderingDevice::TEXTURE_TYPE_2D);
	storage.texture_set_data(texture, p_image);

	Rect2 imgrect(0, 0, p_image->get_width(), p_image->get_height());
	Rect2 screenrect;
	if (p_scale) {
		if (win_size.width > win_size.height) {
			//scale horizontally
			screenrect.size.y = win_size.height;
			screenrect.size.x = imgrect.size.x * win_size.height / imgrect.size.y;
			screenrect.position.x = (win_size.width - screenrect.size.x) / 2;

		} else {
			//scale vertically
			screenrect.size.x = win_size.width;
			screenrect.size.y = imgrect.size.y * win_size.width / imgrect.size.x;
			screenrect.position.y = (win_size.height - screenrect.size.y) / 2;
		}
	} else {
		screenrect = imgrect;
		screenrect.position += ((Size2(win_size.width, win_size.height) - screenrect.size) / 2.0).floor();
	}

	RasterizerStorageGLES3::Texture *t = storage.texture_owner.get_or_null(texture);
	glActiveTexture(GL_TEXTURE0 + storage.config.max_texture_image_units - 1);
	glBindTexture(GL_TEXTURE_2D, t->tex_id);
	glBindTexture(GL_TEXTURE_2D, 0);
	canvas.canvas_end();

	storage.free(texture);

	end_frame(true);
}

#endif // GLES3_ENABLED
