/**************************************************************************/
/*  gl_manager.h                                                          */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#pragma once

#if defined(GLES3_ENABLED)
// These must come first to avoid windows.h mess.
#include "platform_gl.h"
#endif

#include "core/error/error_macros.h"
#include "servers/display_server.h"
#include "servers/rendering/rendering_native_surface.h"

class GLManager {
public:
	virtual Error open_display(void *p_display) = 0;
	virtual Error window_create(DisplayServer::WindowID p_window_id, Ref<RenderingNativeSurface> p_native_surface, int p_width, int p_height) = 0;
	virtual void window_resize(DisplayServer::WindowID p_window_id, int p_width, int p_height) = 0;
	virtual void window_destroy(DisplayServer::WindowID p_window_id) = 0;
	virtual Size2i window_get_size(DisplayServer::WindowID p_id) = 0;
	virtual int window_get_render_target(DisplayServer::WindowID p_window_id) const = 0;
	virtual int window_get_color_texture(DisplayServer::WindowID p_id) const = 0;
	virtual void release_current() = 0;
	virtual void swap_buffers() = 0;

	virtual void window_make_current(DisplayServer::WindowID p_window_id) = 0;

	virtual void set_use_vsync(bool p_use) = 0;
	virtual bool is_using_vsync() const = 0;

	virtual Error initialize(void *p_native_display = nullptr) = 0;

	GLManager() {}
	virtual ~GLManager() {}
};
