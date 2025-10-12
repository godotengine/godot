/**************************************************************************/
/*  embedded_gl_manager.h                                                 */
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

#if defined(MACOS_ENABLED) && defined(GLES3_ENABLED)

#include "core/os/os.h"
#include "core/templates/local_vector.h"
#include "servers/display/display_server.h"

#import <AppKit/AppKit.h>
#import <ApplicationServices/ApplicationServices.h>
#import <CoreVideo/CoreVideo.h>

GODOT_CLANG_WARNING_PUSH_AND_IGNORE("-Wdeprecated-declarations") // OpenGL is deprecated in macOS 10.14.

typedef CGLContextObj (*CGLGetCurrentContextPtr)(void);
typedef CGLError (*CGLTexImageIOSurface2DPtr)(CGLContextObj ctx, GLenum target, GLenum internal_format,
		GLsizei width, GLsizei height, GLenum format, GLenum type, IOSurfaceRef ioSurface, GLuint plane);
typedef const char *(*CGLErrorStringPtr)(CGLError);

class GLManagerEmbedded {
	/// @brief The number of framebuffers to create for each window.
	///
	/// Triple-buffering is used to avoid stuttering.
	static constexpr uint32_t BUFFER_COUNT = 3;

	// The display ID for which vsync is used. If this value is -1, vsync is disabled.
	constexpr static uint32_t INVALID_DISPLAY_ID = static_cast<uint32_t>(-1);

	struct FrameBuffer {
		IOSurfaceRef surface = nullptr;
		unsigned int tex = 0;
		unsigned int fbo = 0;
	};

	struct GLWindow {
		uint32_t width = 0;
		uint32_t height = 0;
		CALayer *layer = nullptr;
		NSOpenGLContext *context = nullptr;
		FrameBuffer framebuffers[BUFFER_COUNT];
		uint32_t current_fb = 0;
		bool is_valid = false;

		void destroy_framebuffers();

		~GLWindow() { destroy_framebuffers(); }
	};

	RBMap<DisplayServer::WindowID, GLWindow> windows;
	typedef RBMap<DisplayServer::WindowID, GLWindow>::Element GLWindowElement;

	NSOpenGLContext *shared_context = nullptr;
	DisplayServer::WindowID current_window = DisplayServer::INVALID_WINDOW_ID;

	Error create_context(GLWindow &p_win);

	bool framework_loaded = false;
	CGLGetCurrentContextPtr CGLGetCurrentContext = nullptr;
	CGLTexImageIOSurface2DPtr CGLTexImageIOSurface2D = nullptr;
	CGLErrorStringPtr CGLErrorString = nullptr;

	uint32_t display_id = INVALID_DISPLAY_ID;
	CVDisplayLinkRef display_link = nullptr;
	bool vsync_enabled = false;
	bool display_link_running = false;
	dispatch_semaphore_t display_semaphore = nullptr;

	void create_display_link();
	void release_display_link();

public:
	Error window_create(DisplayServer::WindowID p_window_id, CALayer *p_layer, int p_width, int p_height);
	void window_destroy(DisplayServer::WindowID p_window_id);
	void window_resize(DisplayServer::WindowID p_window_id, int p_width, int p_height);
	Size2i window_get_size(DisplayServer::WindowID p_window_id) const;

	void set_display_id(uint32_t p_display_id);
	void set_vsync_enabled(bool p_enabled);
	bool is_vsync_enabled() const { return vsync_enabled; }

	void release_current();
	void swap_buffers();

	void window_make_current(DisplayServer::WindowID p_window_id);

	Error initialize();

	GLManagerEmbedded();
	~GLManagerEmbedded();
};

GODOT_CLANG_WARNING_PUSH

#endif // MACOS_ENABLED && GLES3_ENABLED
