/*************************************************************************/
/*  gl_manager_osx_legacy.h                                              */
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

#ifndef GL_MANAGER_OSX_LEGACY_H
#define GL_MANAGER_OSX_LEGACY_H

#if defined(OSX_ENABLED) && defined(GLES3_ENABLED)

#include "core/error/error_list.h"
#include "core/os/os.h"
#include "core/templates/local_vector.h"
#include "servers/display_server.h"

#import <AppKit/AppKit.h>
#import <ApplicationServices/ApplicationServices.h>
#import <CoreVideo/CoreVideo.h>

class GLManager_OSX {
public:
	enum ContextType {
		GLES_3_0_COMPATIBLE,
	};

private:
	struct GLWindow {
		int width = 0;
		int height = 0;

		id window_view = nullptr;
		NSOpenGLContext *context = nullptr;
	};

	Map<DisplayServer::WindowID, GLWindow> windows;

	NSOpenGLContext *shared_context = nullptr;
	DisplayServer::WindowID current_window = DisplayServer::INVALID_WINDOW_ID;

	Error create_context(GLWindow &win);

	bool use_vsync = false;
	ContextType context_type;

public:
	Error window_create(DisplayServer::WindowID p_window_id, id p_view, int p_width, int p_height);
	void window_destroy(DisplayServer::WindowID p_window_id);
	void window_resize(DisplayServer::WindowID p_window_id, int p_width, int p_height);

	int window_get_width(DisplayServer::WindowID p_window_id = 0);
	int window_get_height(DisplayServer::WindowID p_window_id = 0);

	void release_current();
	void make_current();
	void swap_buffers();

	void window_make_current(DisplayServer::WindowID p_window_id);

	void window_update(DisplayServer::WindowID p_window_id);
	void window_set_per_pixel_transparency_enabled(DisplayServer::WindowID p_window_id, bool p_enabled);

	Error initialize();

	void set_use_vsync(bool p_use);
	bool is_using_vsync() const;

	GLManager_OSX(ContextType p_context_type);
	~GLManager_OSX();
};

#endif // OSX_ENABLED && GLES3_ENABLED
#endif // GL_MANAGER_OSX_LEGACY_H
