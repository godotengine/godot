/*************************************************************************/
/*  rasterizer_gles3.cpp                                                 */
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
#include "rasterizer_gles3.h"

#include "gl_context/context_gl.h"
#include "global_config.h"
#include "os/os.h"
#include <string.h>
RasterizerStorage *RasterizerGLES3::get_storage() {

	return storage;
}

RasterizerCanvas *RasterizerGLES3::get_canvas() {

	return canvas;
}

RasterizerScene *RasterizerGLES3::get_scene() {

	return scene;
}

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

#ifdef WINDOWS_ENABLED
#define GLAPIENTRY APIENTRY
#else
#define GLAPIENTRY
#endif

static void GLAPIENTRY _gl_debug_print(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar *message, const GLvoid *userParam) {

	if (type == _EXT_DEBUG_TYPE_OTHER_ARB)
		return;

	print_line("mesege");
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

	ERR_PRINTS(output);
}

typedef void (*DEBUGPROCARB)(GLenum source,
		GLenum type,
		GLuint id,
		GLenum severity,
		GLsizei length,
		const char *message,
		const void *userParam);

typedef void (*DebugMessageCallbackARB)(DEBUGPROCARB callback, const void *userParam);

void RasterizerGLES3::initialize() {

	if (OS::get_singleton()->is_stdout_verbose()) {
		print_line("Using GLES3 video driver");
	}

#ifdef GLEW_ENABLED
	GLuint res = glewInit();
	ERR_FAIL_COND(res != GLEW_OK);
	if (OS::get_singleton()->is_stdout_verbose()) {
		print_line(String("GLES2: Using GLEW ") + (const char *)glewGetString(GLEW_VERSION));
	}

	// Check for GL 2.1 compatibility, if not bail out
	if (!glewIsSupported("GL_VERSION_3_0")) {
		ERR_PRINT("Your system's graphic drivers seem not to support OpenGL 3.0+ / GLES 3.0, sorry :(\n"
				  "Try a drivers update, buy a new GPU or try software rendering on Linux; Godot will now crash with a segmentation fault.");
		OS::get_singleton()->alert("Your system's graphic drivers seem not to support OpenGL 3.0+ / GLES 3.0, sorry :(\n"
								   "Godot Engine will self-destruct as soon as you acknowledge this error message.",
				"Fatal error: Insufficient OpenGL / GLES drivers");
		// TODO: If it's even possible, we should stop the execution without segfault and memory leaks :)
	}
#endif

#ifdef GLAD_ENABLED

	if (!gladLoadGL()) {
		ERR_PRINT("Error initializing GLAD");
	}

#ifdef __APPLE__
// FIXME glDebugMessageCallbackARB does not seem to work on Mac OS X and opengl 3, this may be an issue with our opengl canvas..
#else
	glEnable(_EXT_DEBUG_OUTPUT_SYNCHRONOUS_ARB);
	glDebugMessageCallbackARB(_gl_debug_print, NULL);
	glEnable(_EXT_DEBUG_OUTPUT);
#endif

#endif

	/*	glDebugMessageControlARB(GL_DEBUG_SOURCE_API_ARB,GL_DEBUG_TYPE_ERROR_ARB,GL_DEBUG_SEVERITY_HIGH_ARB,0,NULL,GL_TRUE);
	glDebugMessageControlARB(GL_DEBUG_SOURCE_API_ARB,GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR_ARB,GL_DEBUG_SEVERITY_HIGH_ARB,0,NULL,GL_TRUE);
	glDebugMessageControlARB(GL_DEBUG_SOURCE_API_ARB,GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR_ARB,GL_DEBUG_SEVERITY_HIGH_ARB,0,NULL,GL_TRUE);
	glDebugMessageControlARB(GL_DEBUG_SOURCE_API_ARB,GL_DEBUG_TYPE_PORTABILITY_ARB,GL_DEBUG_SEVERITY_HIGH_ARB,0,NULL,GL_TRUE);
	glDebugMessageControlARB(GL_DEBUG_SOURCE_API_ARB,GL_DEBUG_TYPE_PERFORMANCE_ARB,GL_DEBUG_SEVERITY_HIGH_ARB,0,NULL,GL_TRUE);
	glDebugMessageControlARB(GL_DEBUG_SOURCE_API_ARB,GL_DEBUG_TYPE_OTHER_ARB,GL_DEBUG_SEVERITY_HIGH_ARB,0,NULL,GL_TRUE);
	glDebugMessageInsertARB(

			GL_DEBUG_SOURCE_API_ARB,
			GL_DEBUG_TYPE_OTHER_ARB, 1,
			GL_DEBUG_SEVERITY_HIGH_ARB,5, "hello");

*/
	storage->initialize();
	canvas->initialize();
	scene->initialize();
}

void RasterizerGLES3::begin_frame() {

	uint64_t tick = OS::get_singleton()->get_ticks_usec();

	double time_total = double(tick) / 1000000.0;

	storage->frame.time[0] = time_total;
	storage->frame.time[1] = Math::fmod(time_total, 3600);
	storage->frame.time[2] = Math::fmod(time_total, 900);
	storage->frame.time[3] = Math::fmod(time_total, 60);
	storage->frame.count++;
	storage->frame.delta = double(tick - storage->frame.prev_tick) / 1000000.0;
	if (storage->frame.prev_tick == 0) {
		//to avoid hiccups
		storage->frame.delta = 0.001;
	}

	storage->frame.prev_tick = tick;

	storage->update_dirty_resources();

	storage->info.render_object_count = 0;
	storage->info.render_material_switch_count = 0;
	storage->info.render_surface_switch_count = 0;
	storage->info.render_shader_rebind_count = 0;
	storage->info.render_vertices_count = 0;

	scene->iteration();
}

void RasterizerGLES3::set_current_render_target(RID p_render_target) {

	if (!p_render_target.is_valid() && storage->frame.current_rt && storage->frame.clear_request) {
		//handle pending clear request, if the framebuffer was not cleared
		glBindFramebuffer(GL_FRAMEBUFFER, storage->frame.current_rt->fbo);
		print_line("unbind clear of: " + storage->frame.clear_request_color);
		glClearColor(
				storage->frame.clear_request_color.r,
				storage->frame.clear_request_color.g,
				storage->frame.clear_request_color.b,
				storage->frame.clear_request_color.a);

		glClear(GL_COLOR_BUFFER_BIT);
	}

	if (p_render_target.is_valid()) {
		RasterizerStorageGLES3::RenderTarget *rt = storage->render_target_owner.getornull(p_render_target);
		if (!rt) {
			storage->frame.current_rt = NULL;
		}
		ERR_FAIL_COND(!rt);
		storage->frame.current_rt = rt;
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

void RasterizerGLES3::blit_render_target_to_screen(RID p_render_target, const Rect2 &p_screen_rect, int p_screen) {

	ERR_FAIL_COND(storage->frame.current_rt);

	RasterizerStorageGLES3::RenderTarget *rt = storage->render_target_owner.getornull(p_render_target);
	ERR_FAIL_COND(!rt);

	canvas->canvas_begin();
	glDisable(GL_BLEND);
	glBindFramebuffer(GL_FRAMEBUFFER, RasterizerStorageGLES3::system_fbo);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, rt->color);
	canvas->draw_generic_textured_rect(p_screen_rect, Rect2(0, 0, 1, -1));
	glBindTexture(GL_TEXTURE_2D, 0);
	canvas->canvas_end();
}

void RasterizerGLES3::end_frame() {

#if 0
	canvas->canvas_begin();
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D,storage->resources.white_tex);
	glDisable(GL_BLEND);
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);


	float vtx[8]={0,0,
	0,1,
	1,1,
	1,0
	};

	glBindBuffer(GL_ARRAY_BUFFER,0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,0);

	glEnableVertexAttribArray(VS::ARRAY_VERTEX);
	glVertexAttribPointer( VS::ARRAY_VERTEX, 2 ,GL_FLOAT, false, 0, vtx );


	//glBindBuffer(GL_ARRAY_BUFFER,canvas->data.canvas_quad_vertices);
	//glEnableVertexAttribArray(VS::ARRAY_VERTEX);
	//glVertexAttribPointer( VS::ARRAY_VERTEX, 2 ,GL_FLOAT, false, 0, 0 );

	glBindVertexArray(canvas->data.canvas_quad_array);

	canvas->draw_generic_textured_rect(Rect2(0,0,15,15),Rect2(0,0,1,1));
#endif
	OS::get_singleton()->swap_buffers();

	/*	print_line("objects: "+itos(storage->info.render_object_count));
	print_line("material chages: "+itos(storage->info.render_material_switch_count));
	print_line("surface changes: "+itos(storage->info.render_surface_switch_count));
	print_line("shader changes: "+itos(storage->info.render_shader_rebind_count));
	print_line("vertices: "+itos(storage->info.render_vertices_count));
*/
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

	GLOBAL_DEF("rendering/gles3/render_architecture", 0);
	GlobalConfig::get_singleton()->set_custom_property_info("rendering/gles3/render_architecture", PropertyInfo(Variant::INT, "", PROPERTY_HINT_ENUM, "Desktop,Mobile"));
	GLOBAL_DEF("rendering/quality/use_nearest_mipmap_filter", false);
	GLOBAL_DEF("rendering/quality/anisotropic_filter_level", 4.0);
}

RasterizerGLES3::RasterizerGLES3() {

	storage = memnew(RasterizerStorageGLES3);
	canvas = memnew(RasterizerCanvasGLES3);
	scene = memnew(RasterizerSceneGLES3);
	canvas->storage = storage;
	storage->canvas = canvas;
	scene->storage = storage;
	storage->scene = scene;
}

RasterizerGLES3::~RasterizerGLES3() {

	memdelete(storage);
	memdelete(canvas);
}
