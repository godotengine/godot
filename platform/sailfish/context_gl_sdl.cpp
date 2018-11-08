/*************************************************************************/
/*  context_gl_sdl.cpp                                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2018 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2018 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "context_gl_sdl.h"

#ifdef SDL_ENABLED
#if defined(GLES2_ENABLED)
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <GLES2/gl2.h>
//#include <GLES3/gl3.h>

struct ContextGL_SDL_Private {
	SDL_GLContext gl_context;
};

void ContextGL_SDL::release_current() {
	SDL_GL_MakeCurrent(sdl_window, NULL);
}

void ContextGL_SDL::make_current() {
	SDL_GL_MakeCurrent(sdl_window, p->gl_context);
}

void ContextGL_SDL::swap_buffers() {
	SDL_GL_SwapWindow(sdl_window);
}

Error ContextGL_SDL::initialize() {
	SDL_GL_SetAttribute(SDL_GL_RED_SIZE, 1);
	SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, 1);
	SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, 1);
	SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
	SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);

	// if (opengl_3_context == true) {
	// 	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
	// 	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
	// } else {
		// Try OpenGL ES 2.0
		SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 2);
		SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);
		SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_ES);
		SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, SDL_GL_CONTEXT_FORWARD_COMPATIBLE_FLAG);
	// }
	SDL_DisplayMode dm;
	OS::get_singleton()->print("Get display mode\n");
	SDL_GetCurrentDisplayMode(0, &dm);
	OS::get_singleton()->print("Resolution is: %ix%i\n",dm.w,dm.h);
	OS::get_singleton()->print("Try create SDL_Window\n");

	sdl_window = SDL_CreateWindow("Godot", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, dm.w, dm.h, SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN);
	ERR_FAIL_COND_V(!sdl_window, ERR_UNCONFIGURED);

	p->gl_context = SDL_GL_CreateContext(sdl_window);

	if(p->gl_context == NULL) {
		ERR_EXPLAIN("Could not obtain an OpenGL ES 2.0 context!");
		ERR_FAIL_COND_V(p->gl_context == NULL, ERR_UNCONFIGURED);
	}

	return OK;
}

int ContextGL_SDL::get_window_width() {
	int w;
	SDL_GetWindowSize(sdl_window, &w, NULL);

	return w;
}

int ContextGL_SDL::get_window_height() {
	int h;
	SDL_GetWindowSize(sdl_window, NULL, &h);

	return h;
}

void ContextGL_SDL::set_use_vsync(bool p_use) {
	if (p_use) {
		if (SDL_GL_SetSwapInterval(1) < 0) printf("Warning: Unable to enable vsync! SDL Error: %s\n", SDL_GetError());
	} else {
		if (SDL_GL_SetSwapInterval(0) < 0) printf("Warning: Unable to disable vsync! SDL Error: %s\n", SDL_GetError());
	}

	use_vsync = p_use;
}

bool ContextGL_SDL::is_using_vsync() const {
	return use_vsync;
}

SDL_Window* ContextGL_SDL::get_window_pointer() {
	return sdl_window;
}

ContextGL_SDL::ContextGL_SDL(::SDL_DisplayMode *p_sdl_display_mode, const OS::VideoMode &p_default_video_mode, bool p_opengl_3_context) {
	default_video_mode = p_default_video_mode;
	sdl_display_mode = p_sdl_display_mode;

	opengl_3_context = false; //p_opengl_3_context;

	p = memnew(ContextGL_SDL_Private);
	p->gl_context = 0;
	use_vsync = false;
}

ContextGL_SDL::~ContextGL_SDL() {
	SDL_GL_DeleteContext(p->gl_context);
	SDL_DestroyWindow(sdl_window);
	memdelete(p);
}

#endif
#endif
