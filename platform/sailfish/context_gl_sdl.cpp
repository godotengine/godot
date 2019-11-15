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
#include <video/SDL_sysvideo.h>
#include <SDL_opengles2.h>
#include <SDL_video.h>
#include <video/wayland/SDL_waylandwindow.h>
#include <vector>

struct ContextGL_SDL_Private {
	SDL_GLContext gl_context;
	int display_index;
	OS::ScreenOrientation orientation;
	struct qt_extended_surface *qt_ext_surface;
	// struct wl_output *output; // no need? with SDL 2.0.9
};

/// Qt Extended Surface Listener
/**
		 * onscreen_visibility - (none)
		 * @visible: (none)
		 */
// void qt_extended_surface_handle_onscreen_visibility(void *data,
// 		struct qt_extended_surface *qt_extended_surface,
// 		int32_t visible) {
// 	//hello world
// }
// /**
// 		 * set_generic_property - (none)
// 		 * @name: (none)
// 		 * @value: (none)
// 		 */
// void qt_extended_surface_handle_set_generic_property(void *data,
// 		struct qt_extended_surface *qt_extended_surface,
// 		const char *name,
// 		struct wl_array *value) {
// 	//hello world
// }
// /**
// 		 * close - (none)
// 		 */
// void qt_extended_surface_handle_close(void *data,
// 		struct qt_extended_surface *qt_extended_surface) {
// 	//hello world
// }

// struct qt_extended_surface_listener extended_surface_listener = {
// 	qt_extended_surface_handle_onscreen_visibility,
// 	qt_extended_surface_handle_set_generic_property,
// 	qt_extended_surface_handle_close,
// };

/// 

#define print_verbose(x) \
	if(OS::get_singleton()->is_stdout_verbose()) {\
		OS::get_singleton()->print(x);\
	}

/////////////////////////////////////////////////////////////////////////////////////////////////////
/// Wayland Output Listener (for listen when device change orientation) 
// static void
// output_handle_geometry(void *data, struct wl_output *wl_output, int32_t x, int32_t y,
//                        int32_t physical_width, int32_t physical_height, int32_t subpixel,
//                        const char *make, const char *model, int32_t transform)
// {
// 	if(OS::get_singleton()->is_stdout_verbose()) {
// 		OS::get_singleton()->print("output_handle_geometry: x=%i, y=%i, tr=%i, make=%s;\n", x,y,transform,make);
// 	}
// 	ContextGL_SDL_Private *p = reinterpret_cast<ContextGL_SDL_Private*>(data);
// 	if(p)
// 	{
// 		switch (p->orientation) {
// 			// case SCREEN_LANDSCAPE:
// 			// 	qt_extended_surface_set_content_orientation(p->qt_ext_surface, QT_EXTENDED_SURFACE_ORIENTATION_LANDSCAPEORIENTATION);
// 			// 	break;
// 			// case SCREEN_PORTRAIT:
// 			// 	qt_extended_surface_set_content_orientation(p->qt_ext_surface, QT_EXTENDED_SURFACE_ORIENTATION_PORTRAITORIENTATION);
// 			// 	break;
// 			// case SCREEN_REVERSE_LANDSCAPE:
// 			// 	qt_extended_surface_set_content_orientation(p->qt_ext_surface, QT_EXTENDED_SURFACE_ORIENTATION_INVERTEDLANDSCAPEORIENTATION);
// 			// 	break;
// 			// case SCREEN_REVERSE_PORTRAIT:
// 			// 	qt_extended_surface_set_content_orientation(p->qt_ext_surface, QT_EXTENDED_SURFACE_ORIENTATION_INVERTEDPORTRAITORIENTATION);
// 			// 	break;
// 			case OS::SCREEN_SENSOR_LANDSCAPE:
// 				{
// 					switch(transform) {
// 					case WL_OUTPUT_TRANSFORM_FLIPPED_270:
// 					case WL_OUTPUT_TRANSFORM_90:
// 						qt_extended_surface_set_content_orientation(p->qt_ext_surface, QT_EXTENDED_SURFACE_ORIENTATION_LANDSCAPEORIENTATION);
// 						break;
// 					case WL_OUTPUT_TRANSFORM_FLIPPED_90:
// 					case WL_OUTPUT_TRANSFORM_270:
// 						qt_extended_surface_set_content_orientation(p->qt_ext_surface, QT_EXTENDED_SURFACE_ORIENTATION_INVERTEDLANDSCAPEORIENTATION);
// 						break;
// 					}
// 				}
// 				break;
// 			case OS::SCREEN_SENSOR_PORTRAIT:
// 				{
// 					switch(transform) {
// 					case WL_OUTPUT_TRANSFORM_NORMAL:
// 						qt_extended_surface_set_content_orientation(p->qt_ext_surface, QT_EXTENDED_SURFACE_ORIENTATION_PORTRAITORIENTATION);
// 						break;
// 					case  WL_OUTPUT_TRANSFORM_180:
// 					case WL_OUTPUT_TRANSFORM_FLIPPED:
// 						qt_extended_surface_set_content_orientation(p->qt_ext_surface, QT_EXTENDED_SURFACE_ORIENTATION_INVERTEDPORTRAITORIENTATION);
// 						break;
// 					}
// 				}
// 				break;
// 			case OS::SCREEN_SENSOR:
// 				switch(transform) {
// 					case WL_OUTPUT_TRANSFORM_FLIPPED_270:
// 					case WL_OUTPUT_TRANSFORM_90:
// 						qt_extended_surface_set_content_orientation(p->qt_ext_surface, QT_EXTENDED_SURFACE_ORIENTATION_LANDSCAPEORIENTATION);
// 						break;
// 					case WL_OUTPUT_TRANSFORM_FLIPPED_90:
// 					case WL_OUTPUT_TRANSFORM_270:
// 						qt_extended_surface_set_content_orientation(p->qt_ext_surface, QT_EXTENDED_SURFACE_ORIENTATION_INVERTEDLANDSCAPEORIENTATION);
// 						break;
// 					case WL_OUTPUT_TRANSFORM_NORMAL:
// 						qt_extended_surface_set_content_orientation(p->qt_ext_surface, QT_EXTENDED_SURFACE_ORIENTATION_PORTRAITORIENTATION);
// 						break;
// 					case WL_OUTPUT_TRANSFORM_180:
// 					case WL_OUTPUT_TRANSFORM_FLIPPED:
// 						qt_extended_surface_set_content_orientation(p->qt_ext_surface, QT_EXTENDED_SURFACE_ORIENTATION_INVERTEDPORTRAITORIENTATION);
// 						break;
// 				}

// 				break;
// 		}
// 	}
// }

// static void
// output_handle_mode(void *data, struct wl_output *wl_output,
//                    uint32_t flags, int32_t width, int32_t height, int32_t refresh)
// {}

// static void
// output_handle_scale(void *data, struct wl_output *wl_output, int32_t factor)
// {}

// static void
// output_handle_done(void *data, struct wl_output* wl_output)
// {}

// static struct wl_output_listener output_listener = {
// 	output_handle_geometry,
// 	output_handle_mode,
// 	output_handle_done,
// 	output_handle_scale,
// };
/////////////////////////////////////////////////////////////////////////////////////////////////////


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

	sdl_window = SDL_CreateWindow("Godot", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, dm.w, dm.h, SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN | SDL_WINDOW_FULLSCREEN); //| SDL_WINDOW_FULLSCREEN

	if( !sdl_window ) {
		OS::get_singleton()->print("SDL_Error \"%s\"",SDL_GetError());
		return FAILED;
	}
	ERR_FAIL_COND_V(!sdl_window, ERR_UNCONFIGURED);

	p->display_index = SDL_GetWindowDisplayIndex(sdl_window);
	OS::get_singleton()->print("DisplayIndex is %i \n", p->display_index);
	// SDL_DisplayMode mode;
	// int orientation = SDL_GetDisplayOrientation(p->display_index);

	// switch(orientation) {
	// 	case SDL_ORIENTATION_LANDSCAPE:
	// 		OS::get_singleton()->print("SDL_DisplayOrientation is SDL_ORIENTATION_LANDSCAPE ");
	// 		break;
	// 	case SDL_ORIENTATION_LANDSCAPE_FLIPPED:
	// 		OS::get_singleton()->print("SDL_DisplayOrientation is SDL_ORIENTATION_LANDSCAPE_FLIPPED");
	// 		break;
	// 	case SDL_ORIENTATION_PORTRAIT:
	// 		OS::get_singleton()->print("SDL_DisplayOrientation is SDL_ORIENTATION_PORTRAIT");
	// 		break;
	// 	case SDL_ORIENTATION_PORTRAIT_FLIPPED:
	// 		OS::get_singleton()->print("SDL_DisplayOrientation is SDL_ORIENTATION_PORTRAIT_FLIPPED");
	// 		break;
	// 	case SDL_ORIENTATION_UNKNOWN:
	// 		OS::get_singleton()->print("SDL_DisplayOrientation is SDL_ORIENTATION_UNKNOWN");
	// 		break;
	// }

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
		ERR_EXPLAIN("Could not obtain an OpenGL ES 2.0 context!");
		ERR_FAIL_COND_V(p->gl_context == NULL, ERR_UNCONFIGURED);
		return FAILED;
	}

    // p->sdl_renderer = SDL_CreateRenderer( sdl_window, -1, SDL_RENDERER_ACCELERATED );
    // if( NULL == p->sdl_renderer )
    // {
    //     OS::get_singleton()->print("SDL_CreateRenderer(): %s \n" ,SDL_GetError());
    //     return FAILED;
    // }
    // SDL_RendererInfo info;
    // SDL_GetRendererInfo( p->sdl_renderer, &info );
    // OS::get_singleton()->print("SDL_RENDER_DRIVER selected : %s \n", info.name );

	//sdl_window.
	print_verbose("Get SDL_WindowData.\n");
	SDL_WindowData* wdata = (SDL_WindowData*)sdl_window->driverdata;
	if( wdata == NULL )
	{
		OS::get_singleton()->print("SDL_WindowData is empty!");
	}
	print_verbose("Get SDL_VideData (waylandData).\n");
	SDL_VideoData* sdl_videodata = wdata->waylandData;
	print_verbose("Get qt_extended_surface (for setting screen orientation)\n");
	p->qt_ext_surface = wdata->extended_surface;
	if(OS::get_singleton()->is_stdout_verbose()) 
	{
		if( sdl_videodata != NULL )
		{
			OS::get_singleton()->print("SDL videodata is handled;\n");
		}
	}
	print_verbose("Set orientation for qt_extended_surface\n");
	set_screen_orientation(p->orientation);
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

void ContextGL_SDL::set_ext_surface_orientation(int orientation)
{
	if(OS::get_singleton()->is_stdout_verbose())
	{
		OS::get_singleton()->print("set_ext_surface_orientation %i\n", orientation);
		OS::get_singleton()->print("ContextGL_SDL.p->orientation is %i\n", p->orientation);
	}
	switch (p->orientation) {
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
			switch(orientation) {
			case SDL_ORIENTATION_LANDSCAPE:
				qt_extended_surface_set_content_orientation(p->qt_ext_surface, QT_EXTENDED_SURFACE_ORIENTATION_LANDSCAPEORIENTATION);
				if(OS::get_singleton()->is_stdout_verbose())
					OS::get_singleton()->print("set_Screen_orientation %i\n", OS::SCREEN_LANDSCAPE);
				OS::get_singleton()->set_screen_orientation(OS::SCREEN_LANDSCAPE);
				break;
			case SDL_ORIENTATION_LANDSCAPE_FLIPPED:
				qt_extended_surface_set_content_orientation(p->qt_ext_surface, QT_EXTENDED_SURFACE_ORIENTATION_INVERTEDLANDSCAPEORIENTATION);
				if(OS::get_singleton()->is_stdout_verbose())
					OS::get_singleton()->print("set_Screen_orientation %i\n", OS::SCREEN_REVERSE_LANDSCAPE);
				OS::get_singleton()->set_screen_orientation(OS::SCREEN_REVERSE_LANDSCAPE);
				break;
			}
		}
		break;
	case OS::SCREEN_SENSOR_PORTRAIT:
		{
			switch(orientation) {
			case SDL_ORIENTATION_PORTRAIT:
				qt_extended_surface_set_content_orientation(p->qt_ext_surface, QT_EXTENDED_SURFACE_ORIENTATION_PORTRAITORIENTATION);
				if(OS::get_singleton()->is_stdout_verbose())
					OS::get_singleton()->print("set_Screen_orientation %i\n", OS::SCREEN_PORTRAIT);
				OS::get_singleton()->set_screen_orientation(OS::SCREEN_PORTRAIT);
				break;
			case SDL_ORIENTATION_PORTRAIT_FLIPPED:
				qt_extended_surface_set_content_orientation(p->qt_ext_surface, QT_EXTENDED_SURFACE_ORIENTATION_INVERTEDPORTRAITORIENTATION);
				if(OS::get_singleton()->is_stdout_verbose())
					OS::get_singleton()->print("set_Screen_orientation %i\n", OS::SCREEN_REVERSE_PORTRAIT);
				OS::get_singleton()->set_screen_orientation(OS::SCREEN_REVERSE_PORTRAIT);
				break;
			}
		}
		break;
	case OS::SCREEN_SENSOR:
		switch(orientation) {
			case SDL_ORIENTATION_LANDSCAPE:
				qt_extended_surface_set_content_orientation(p->qt_ext_surface, QT_EXTENDED_SURFACE_ORIENTATION_LANDSCAPEORIENTATION);
				if(OS::get_singleton()->is_stdout_verbose())
					OS::get_singleton()->print("set_Screen_orientation %i\n", OS::SCREEN_LANDSCAPE);
				OS::get_singleton()->set_screen_orientation(OS::SCREEN_LANDSCAPE);
				break;
			case SDL_ORIENTATION_LANDSCAPE_FLIPPED:
				qt_extended_surface_set_content_orientation(p->qt_ext_surface, QT_EXTENDED_SURFACE_ORIENTATION_INVERTEDLANDSCAPEORIENTATION);
				if(OS::get_singleton()->is_stdout_verbose())
					OS::get_singleton()->print("set_Screen_orientation %i\n", OS::SCREEN_REVERSE_LANDSCAPE);
				OS::get_singleton()->set_screen_orientation(OS::SCREEN_REVERSE_LANDSCAPE);
				break;
			case SDL_ORIENTATION_PORTRAIT:
				qt_extended_surface_set_content_orientation(p->qt_ext_surface, QT_EXTENDED_SURFACE_ORIENTATION_PORTRAITORIENTATION);
				if(OS::get_singleton()->is_stdout_verbose())
					OS::get_singleton()->print("set_Screen_orientation %i\n", OS::SCREEN_PORTRAIT);
				OS::get_singleton()->set_screen_orientation(OS::SCREEN_PORTRAIT);
				break;
			case SDL_ORIENTATION_PORTRAIT_FLIPPED:
				qt_extended_surface_set_content_orientation(p->qt_ext_surface, QT_EXTENDED_SURFACE_ORIENTATION_INVERTEDPORTRAITORIENTATION);
				if(OS::get_singleton()->is_stdout_verbose())
					OS::get_singleton()->print("set_Screen_orientation %i\n", OS::SCREEN_REVERSE_PORTRAIT);
				OS::get_singleton()->set_screen_orientation(OS::SCREEN_REVERSE_PORTRAIT);
				break;
		}
		break;
	}
}
void ContextGL_SDL::set_screen_orientation(OS::ScreenOrientation p_orientation) {
#ifdef SAILFISH_FORCE_LANDSCAPE	
	if(p->qt_ext_surface)
	switch(p_orientation) {
	case OS::SCREEN_LANDSCAPE:
		qt_extended_surface_set_content_orientation(p->qt_ext_surface, QT_EXTENDED_SURFACE_ORIENTATION_LANDSCAPEORIENTATION );
		break;
	case OS::SCREEN_PORTRAIT:
		qt_extended_surface_set_content_orientation(p->qt_ext_surface, QT_EXTENDED_SURFACE_ORIENTATION_PORTRAITORIENTATION );
		break;
	case OS::SCREEN_REVERSE_LANDSCAPE:
		qt_extended_surface_set_content_orientation(p->qt_ext_surface, QT_EXTENDED_SURFACE_ORIENTATION_INVERTEDLANDSCAPEORIENTATION );
		break;
	case OS::SCREEN_REVERSE_PORTRAIT:
		qt_extended_surface_set_content_orientation(p->qt_ext_surface, QT_EXTENDED_SURFACE_ORIENTATION_INVERTEDPORTRAITORIENTATION );
		break;
	case OS::SCREEN_SENSOR_LANDSCAPE:
		break;
	case OS::SCREEN_SENSOR_PORTRAIT:
		break;
	case OS::SCREEN_SENSOR:
		break;
	}
	else if (OS::get_singleton()->is_stdout_verbose())
		OS::get_singleton()->print("No qt_extended surface handler yet\n", p->orientation);
#endif
	p->orientation = p_orientation;
	if(OS::get_singleton()->is_stdout_verbose())
		OS::get_singleton()->print("ContextGL_SDL orientation store as %i\n", p->orientation);
}

ContextGL_SDL::ContextGL_SDL(::SDL_DisplayMode *p_sdl_display_mode, const OS::VideoMode &p_default_video_mode, bool p_opengl_3_context) {
	default_video_mode = p_default_video_mode;
	sdl_display_mode = p_sdl_display_mode;

	opengl_3_context = false; //p_opengl_3_context;

	p = memnew(ContextGL_SDL_Private);
	p->gl_context = NULL;
	p->qt_ext_surface = NULL;
	use_vsync = false;
}

ContextGL_SDL::~ContextGL_SDL() {
	SDL_GL_DeleteContext(p->gl_context);
	SDL_DestroyWindow(sdl_window);
	memdelete(p);
}

#endif
#endif
