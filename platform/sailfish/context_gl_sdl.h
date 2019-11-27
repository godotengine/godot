/*************************************************************************/
/*  context_gl_sdl.h                                                     */
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

#ifndef CONTEXT_GL_SDL_H
#define CONTEXT_GL_SDL_H

/**
	@author Juan Linietsky <reduzio@gmail.com>
	@author Joshua Bemenderfer <tribex10@gmail.com>
*/
#ifdef SDL_ENABLED

#if defined(GLES2_ENABLED)

#include "os/os.h"
#include <SDL.h>

struct ContextGL_SDL_Private;

class ContextGL_SDL {

	ContextGL_SDL_Private *p;
	OS::VideoMode default_video_mode;
	::SDL_DisplayMode *sdl_display_mode;
	::SDL_Window *sdl_window;
	bool opengl_3_context;
	bool use_vsync;

	friend struct ContextGL_SDL_Private;
	int width, height;

public:
	virtual void release_current();
	virtual void make_current();
	virtual void swap_buffers();
	virtual int get_window_width();
	virtual int get_window_height();

	virtual Error initialize();

	virtual void set_use_vsync(bool p_use);
	virtual bool is_using_vsync() const;
	virtual SDL_Window *get_window_pointer();

	void set_screen_orientation(OS::ScreenOrientation p_orientation);
	// is SDL_DisplayOrientation value
	void set_ext_surface_orientation(int orientation);

	ContextGL_SDL(::SDL_DisplayMode *p_sdl_display_mode, const OS::VideoMode &p_default_video_mode, bool p_opengl_3_context);
	~ContextGL_SDL();
};

#endif

#endif
#endif
