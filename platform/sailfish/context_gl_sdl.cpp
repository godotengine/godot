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
// #include <GLES2/gl2.h>
#include <EGL/egl.h>
#include <SDL_opengles2.h>
#include <SDL_video.h>
#include <vector>
#include <string>

struct ContextGL_SDL_Private {
	SDL_GLContext gl_context;
	int display_index;
	OS::ScreenOrientation allowed_orientation_enum;
	std::string           allowed_orientation_str;
};

#define print_verbose(x) \
	if(OS::get_singleton()->is_stdout_verbose()) {\
		OS::get_singleton()->print(x);\
	}

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
	print_verbose("Begin SDL2 initialization\n");

	//  if (opengl_3_context == true) {
	//  	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
	//  	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
	//  } else {
		// Try OpenGL ES 2.0
	// SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 2);
	// SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);
	// SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_COMPATIBILITY);
	// SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, SDL_GL_CONTEXT_FORWARD_COMPATIBLE_FLAG);
	//  }
	SDL_GL_SetAttribute(SDL_GL_RED_SIZE, 8);
	SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, 8);
	SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, 8);
	SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 8);
	SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);

	SDL_DisplayMode dm;
	OS::get_singleton()->print("Get display mode\n");
	SDL_GetCurrentDisplayMode(0, &dm);
	OS::get_singleton()->print("Resolution is: %ix%i\n",dm.w,dm.h);
	OS::get_singleton()->print("Try create SDL_Window\n");
	width = dm.w;
	height = dm.h;

	// dm.orientation;

	// SDL_GetDisplayMode()
	String app_name = OS::get_singleton()->get_name();
	if( app_name.empty() )
		app_name = "Godot";
	sdl_window = SDL_CreateWindow( app_name.utf8().ptr(), SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, dm.w, dm.h, SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN | SDL_WINDOW_FULLSCREEN); //| SDL_WINDOW_FULLSCREEN

	if( !sdl_window ) { 
		OS::get_singleton()->print("SDL_Error \"%s\"",SDL_GetError());
		return FAILED;
	}
	ERR_FAIL_COND_V(!sdl_window, ERR_UNCONFIGURED);

	p->display_index = SDL_GetWindowDisplayIndex(sdl_window);
	OS::get_singleton()->print("DisplayIndex is %i \n", p->display_index);

	OS::get_singleton()->print("\nSDL_RENDER_DRIVER available:\n");
    for( int i = 0; i < SDL_GetNumRenderDrivers(); ++i )
    {
        SDL_RendererInfo info;
        SDL_GetRenderDriverInfo( i, &info );
        OS::get_singleton()->print("[%i] %s\n",i,info.name);
    }

	print_verbose("Create GL context.\n");
	p->gl_context = SDL_GL_CreateContext(sdl_window); //SDL_GL_GetCurrentContext();
	if(p->gl_context == NULL) {
		// ERR_EXPLAIN("Could not obtain an OpenGL ES 2.0 context!");
		OS::get_singleton()->print("ERROR: Could not obtain an OpenGL ES 2.0 context!");
		ERR_FAIL_COND_V(p->gl_context == NULL, ERR_UNCONFIGURED);
		return FAILED;
	}
	return OK;
}

int ContextGL_SDL::get_window_width() {
	// int w;
	// SDL_GetWindowSize(sdl_window, &w, NULL);

	return width;
}

int ContextGL_SDL::get_window_height() {
	// int h;
	// SDL_GetWindowSize(sdl_window, NULL, &h);

	return height;
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

void ContextGL_SDL::set_ext_surface_orientation(int sdl_orientation)
{
	OS::ScreenOrientation screen_orientation = OS::get_singleton()->get_screen_orientation();
	if(OS::get_singleton()->is_stdout_verbose())
	{
		// OS::get_singleton()->print("set_ext_surface_orientation %i\n", sdl_orientation);
		// OS::get_singleton()->print("ContextGL_SDL.p->allowed_orientation_enum is %i\n", p->allowed_orientation_enum);
		OS::get_singleton()->print("OS current screen orientation is \"%i\"\n", screen_orientation);
	}
	switch (p->allowed_orientation_enum) {
	// case SCREEN_LANDSCAPE:
	// 	qt_extended_surface_set_content_orientation(p->qt_ext_surface, QT_EXTENDED_SURFACE_ORIENTATION_LANDSCAPEORIENTATION);
	// 	break;
	// case SCREEN_PORTRAIT:
	// 	qt_extended_surface_set_content_orientation(p->qt_ext_surface, QT_EXTENDED_SURFACE_ORIENTATION_PORTRAITORIENTATION);
	// 	break;
	// case SCREEN_REVERSE_LANDSCAPE:
	// 	qt_extended_surface_set_content_orientation(p->qt_ext_surface, QT_EXTENDED_SURFACE_ORIENTATION_INVERTEDLANDSCAPEORIENTATION);
	// 	break;
	// case SCREEN_REVERSE_PORTRAIT:
	// 	qt_extended_surface_set_content_orientation(p->qt_ext_surface, QT_EXTENDED_SURFACE_ORIENTATION_INVERTEDPORTRAITORIENTATION);
	// 	break;
	case OS::SCREEN_SENSOR_LANDSCAPE:
		{
			switch(sdl_orientation) {
			case SDL_ORIENTATION_LANDSCAPE:
				// qt_extended_surface_set_content_orientation(p->qt_ext_surface, QT_EXTENDED_SURFACE_ORIENTATION_LANDSCAPEORIENTATION);
				p->allowed_orientation_str = "landscape";
				if(OS::get_singleton()->is_stdout_verbose())
					OS::get_singleton()->print("set_Screen_orientation OS::SCREEN_LANDSCAPE\n");
				screen_orientation = (OS::SCREEN_LANDSCAPE);
				break;
			case SDL_ORIENTATION_LANDSCAPE_FLIPPED:
				// qt_extended_surface_set_content_orientation(p->qt_ext_surface, QT_EXTENDED_SURFACE_ORIENTATION_INVERTEDLANDSCAPEORIENTATION);
				p->allowed_orientation_str = "inverted-landscape";
				if(OS::get_singleton()->is_stdout_verbose())
					OS::get_singleton()->print("set_Screen_orientation OS::SCREEN_REVERSE_LANDSCAPE\n");
				screen_orientation = (OS::SCREEN_REVERSE_LANDSCAPE);
				break;
			}
		}
		break;
	case OS::SCREEN_SENSOR_PORTRAIT:
		{
			switch(sdl_orientation) {
			case SDL_ORIENTATION_PORTRAIT:
				// qt_extended_surface_set_content_orientation(p->qt_ext_surface, QT_EXTENDED_SURFACE_ORIENTATION_PORTRAITORIENTATION);
				p->allowed_orientation_str = "portrait";
				if(OS::get_singleton()->is_stdout_verbose())
					OS::get_singleton()->print("set_screen_orientation OS::SCREEN_PORTRAIT\n");
				screen_orientation = (OS::SCREEN_PORTRAIT);
				break;
			case SDL_ORIENTATION_PORTRAIT_FLIPPED:
				// qt_extended_surface_set_content_orientation(p->qt_ext_surface, QT_EXTENDED_SURFACE_ORIENTATION_INVERTEDPORTRAITORIENTATION);
				p->allowed_orientation_str = "inverted-portrait";
				if(OS::get_singleton()->is_stdout_verbose())
					OS::get_singleton()->print("set_screen_orientation OS::SCREEN_REVERSE_PORTRAIT\n");
				screen_orientation = (OS::SCREEN_REVERSE_PORTRAIT);
				break;
			}
		}
		break;
	case OS::SCREEN_SENSOR:
		switch(sdl_orientation) {
			case SDL_ORIENTATION_LANDSCAPE:
				// qt_extended_surface_set_content_orientation(p->qt_ext_surface, QT_EXTENDED_SURFACE_ORIENTATION_LANDSCAPEORIENTATION);
				p->allowed_orientation_str = "landscape";
				if(OS::get_singleton()->is_stdout_verbose())
					OS::get_singleton()->print("set_screen_orientation OS::SCREEN_LANDSCAPE\n");
				screen_orientation = (OS::SCREEN_LANDSCAPE);
				break;
			case SDL_ORIENTATION_LANDSCAPE_FLIPPED:
				// qt_extended_surface_set_content_orientation(p->qt_ext_surface, QT_EXTENDED_SURFACE_ORIENTATION_INVERTEDLANDSCAPEORIENTATION);
				p->allowed_orientation_str = "inverted-landscape";
				if(OS::get_singleton()->is_stdout_verbose())
					OS::get_singleton()->print("set_screen_orientation OS::SCREEN_REVERSE_LANDSCAPE\n");
				screen_orientation = (OS::SCREEN_REVERSE_LANDSCAPE);
				break;
			case SDL_ORIENTATION_PORTRAIT:
				// qt_extended_surface_set_content_orientation0(p->qt_ext_surface, QT_EXTENDED_SURFACE_ORIENTATION_PORTRAITORIENTATION);
				p->allowed_orientation_str = "portrait";
				if(OS::get_singleton()->is_stdout_verbose())
					OS::get_singleton()->print("set_screen_orientation OS::SCREEN_PORTRAIT\n");
				screen_orientation = (OS::SCREEN_PORTRAIT);
				break;
			case SDL_ORIENTATION_PORTRAIT_FLIPPED:
				// qt_extended_surface_set_content_orientation(p->qt_ext_surface, QT_EXTENDED_SURFACE_ORIENTATION_INVERTEDPORTRAITORIENTATION);
				p->allowed_orientation_str = "inverted-portrait";
				if(OS::get_singleton()->is_stdout_verbose())
					OS::get_singleton()->print("set_screen_orientation OS::SCREEN_REVERSE_PORTRAIT\n");
				screen_orientation = (OS::SCREEN_REVERSE_PORTRAIT);
				break;
		}
		break;
	}

	// if( screen_orientation != OS::get_singleton()->get_screen_orientation() ) {
		if( OS::get_singleton()->is_stdout_verbose() )
			OS::get_singleton()->print("SDL_SetHint(%s, %s)", SDL_HINT_QTWAYLAND_CONTENT_ORIENTATION,p->allowed_orientation_str.c_str() );
		if(  SDL_SetHintWithPriority(SDL_HINT_QTWAYLAND_CONTENT_ORIENTATION, p->allowed_orientation_str.c_str(), SDL_HINT_OVERRIDE) == SDL_FALSE 
			&& OS::get_singleton()->is_stdout_verbose() )
			OS::get_singleton()->print("WARGNING: Cant set hint for orinetation events");
		OS::get_singleton()->set_screen_orientation(screen_orientation);
	// }
}

void ContextGL_SDL::set_screen_orientation(OS::ScreenOrientation p_orientation) {
#ifdef SAILFISH_FORCE_LANDSCAPE	
	// if(p->qt_ext_surface)
	// "primary" (default), "portrait", "landscape", "inverted-portrait", "inverted-landscape"
	switch(p_orientation) {
	case OS::SCREEN_LANDSCAPE:
		// qt_extended_surface_set_content_orientation(p->qt_ext_surface, QT_EXTENDED_SURFACE_ORIENTATION_LANDSCAPEORIENTATION );
		p->allowed_orientation_str = "landscape";
		break;
	case OS::SCREEN_PORTRAIT:
		// qt_extended_surface_set_content_orientation(p->qt_ext_surface, QT_EXTENDED_SURFACE_ORIENTATION_PORTRAITORIENTATION );
		p->allowed_orientation_str = "portrait";//"Portrait";
		break;
	case OS::SCREEN_REVERSE_LANDSCAPE:
		// qt_extended_surface_set_content_orientation(p->qt_ext_surface, QT_EXTENDED_SURFACE_ORIENTATION_INVERTEDLANDSCAPEORIENTATION );
		p->allowed_orientation_str = "inverted-landscape";
		break;
	case OS::SCREEN_REVERSE_PORTRAIT:
		// qt_extended_surface_set_content_orientation(p->qt_ext_surface, QT_EXTENDED_SURFACE_ORIENTATION_INVERTEDPORTRAITORIENTATION );
		p->allowed_orientation_str = "inverted-portrait";
		break;
	case OS::SCREEN_SENSOR_LANDSCAPE:
		p->allowed_orientation_str = "landscape";
		break;
	case OS::SCREEN_SENSOR_PORTRAIT: 
		p->allowed_orientation_str = "portrait";
		break;
	case OS::SCREEN_SENSOR: 
		p->allowed_orientation_str = "primary";//"LandscapeLeft LandscapeRight Portrait PortraitUpsideDown";
		break;
	}

	if (OS::get_singleton()->is_stdout_verbose())
		OS::get_singleton()->print("Set allowed orientations to \"%s\"\n", p->allowed_orientation_str.c_str());

	/*
	LandscapeLeft         top of device left
	LandscapeRight        top of device right
	Portrait              top of device up
	PortraitUpsideDown    top of device down
	*/

    if( SDL_SetHintWithPriority(SDL_HINT_ORIENTATIONS, p->allowed_orientation_str.c_str(), SDL_HINT_OVERRIDE) == SDL_FALSE 
	// if( SDL_SetHintWithPriority(SDL_HINT_QTWAYLAND_CONTENT_ORIENTATION, "landscape", SDL_HINT_OVERRIDE) == SDL_FALSE 
		&& OS::get_singleton()->is_stdout_verbose() )
		OS::get_singleton()->print("WARGNING: Cant set hint for orinetation events");
	// else if (OS::get_singleton()->is_stdout_verbose())
		// OS::get_singleton()->print("No qt_extended surface handler yet\n", p->allowed_orientation_enum);
#endif
	p->allowed_orientation_enum = p_orientation;
	// if(OS::get_singleton()->is_stdout_verbose())
		// OS::get_singleton()->print("ContextGL_SDL orientation store as %i\n", p->allowed_orientation_enum);
}

ContextGL_SDL::ContextGL_SDL(::SDL_DisplayMode *p_sdl_display_mode, const OS::VideoMode &p_default_video_mode, bool p_opengl_3_context) {
	default_video_mode = p_default_video_mode;
	sdl_display_mode = p_sdl_display_mode;

	opengl_3_context = false; //p_opengl_3_context;

	p = memnew(ContextGL_SDL_Private);
	p->gl_context = NULL;
	// p->qt_ext_surface = NULL;
	use_vsync = false;
}

ContextGL_SDL::~ContextGL_SDL() {
	SDL_GL_DeleteContext(p->gl_context);
	SDL_DestroyWindow(sdl_window);
	memdelete(p);
}

#endif
#endif
