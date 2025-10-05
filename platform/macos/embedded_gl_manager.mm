/**************************************************************************/
/*  embedded_gl_manager.mm                                                */
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

#import "embedded_gl_manager.h"

#if defined(MACOS_ENABLED) && defined(GLES3_ENABLED)

#import "drivers/gles3/storage/texture_storage.h"
#import "platform_gl.h"

#import <QuartzCore/QuartzCore.h>
#include <dlfcn.h>

GODOT_CLANG_WARNING_PUSH_AND_IGNORE("-Wdeprecated-declarations") // OpenGL is deprecated in macOS 10.14.

Error GLManagerEmbedded::create_context(GLWindow &p_win) {
	NSOpenGLPixelFormatAttribute attributes[] = {
		NSOpenGLPFADoubleBuffer,
		NSOpenGLPFAClosestPolicy,
		NSOpenGLPFAOpenGLProfile, NSOpenGLProfileVersion3_2Core,
		NSOpenGLPFAColorSize, 32,
		NSOpenGLPFADepthSize, 24,
		NSOpenGLPFAStencilSize, 8,
		0
	};

	NSOpenGLPixelFormat *pixel_format = [[NSOpenGLPixelFormat alloc] initWithAttributes:attributes];
	ERR_FAIL_NULL_V(pixel_format, ERR_CANT_CREATE);

	p_win.context = [[NSOpenGLContext alloc] initWithFormat:pixel_format shareContext:shared_context];
	ERR_FAIL_NULL_V(p_win.context, ERR_CANT_CREATE);
	if (shared_context == nullptr) {
		shared_context = p_win.context;
	}

	[p_win.context makeCurrentContext];

	return OK;
}

Error GLManagerEmbedded::window_create(DisplayServer::WindowID p_window_id, CALayer *p_layer, int p_width, int p_height) {
	GLWindow win;
	win.layer = p_layer;
	win.width = 0;
	win.height = 0;

	if (create_context(win) != OK) {
		return FAILED;
	}

	windows[p_window_id] = win;
	window_make_current(p_window_id);

	return OK;
}

void GLManagerEmbedded::window_resize(DisplayServer::WindowID p_window_id, int p_width, int p_height) {
	GLWindowElement *el = windows.find(p_window_id);
	ERR_FAIL_NULL_MSG(el, "Window resize failed: window does not exist.");

	GLWindow &win = el->get();

	if (win.width == (uint32_t)p_width && win.height == (uint32_t)p_height) {
		return;
	}

	win.width = (uint32_t)p_width;
	win.height = (uint32_t)p_height;

	win.destroy_framebuffers();

	for (FrameBuffer &fb : win.framebuffers) {
		NSDictionary *surfaceProps = @{
			(NSString *)kIOSurfaceWidth : @(p_width),
			(NSString *)kIOSurfaceHeight : @(p_height),
			(NSString *)kIOSurfaceBytesPerElement : @(4),
			(NSString *)kIOSurfacePixelFormat : @(kCVPixelFormatType_32BGRA),
		};
		fb.surface = IOSurfaceCreate((__bridge CFDictionaryRef)surfaceProps);
		if (fb.surface == nullptr) {
			ERR_PRINT(vformat("Failed to create IOSurface: width=%d, height=%d", p_width, p_height));
			win.destroy_framebuffers();
			return;
		}

		glGenTextures(1, &fb.tex);
		glBindTexture(GL_TEXTURE_RECTANGLE, fb.tex);

		glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

		CGLError err = CGLTexImageIOSurface2D(CGLGetCurrentContext(),
				GL_TEXTURE_RECTANGLE,
				GL_RGBA,
				p_width,
				p_height,
				GL_BGRA,
				GL_UNSIGNED_INT_8_8_8_8_REV,
				fb.surface,
				0);
		if (err != kCGLNoError) {
			String err_string = String(CGLErrorString(err));
			ERR_PRINT(vformat("CGLTexImageIOSurface2D failed (%d): %s", err, err_string));
			win.destroy_framebuffers();
			return;
		}

		glGenFramebuffers(1, &fb.fbo);
		glBindFramebuffer(GL_FRAMEBUFFER, fb.fbo);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_RECTANGLE, fb.tex, 0);

		if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
			ERR_PRINT("Unable to create framebuffer from IOSurface texture.");
			win.destroy_framebuffers();
			return;
		}

		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		glBindTexture(GL_TEXTURE_RECTANGLE, 0);
	}
	win.current_fb = 0;
	GLES3::TextureStorage::system_fbo = win.framebuffers[win.current_fb].fbo;
	win.is_valid = true;
}

void GLManagerEmbedded::GLWindow::destroy_framebuffers() {
	is_valid = false;
	GLES3::TextureStorage::system_fbo = 0;

	for (FrameBuffer &fb : framebuffers) {
		if (fb.fbo) {
			glDeleteFramebuffers(1, &fb.fbo);
			fb.fbo = 0;
		}

		if (fb.tex) {
			glDeleteTextures(1, &fb.tex);
			fb.tex = 0;
		}

		if (fb.surface) {
			IOSurfaceRef old_surface = fb.surface;
			fb.surface = nullptr;
			CFRelease(old_surface);
		}
	}
}

Size2i GLManagerEmbedded::window_get_size(DisplayServer::WindowID p_window_id) const {
	const GLWindowElement *el = windows.find(p_window_id);
	if (el == nullptr) {
		return Size2i();
	}

	const GLWindow &win = el->value();
	return Size2i(win.width, win.height);
}

void GLManagerEmbedded::window_destroy(DisplayServer::WindowID p_window_id) {
	GLWindowElement *el = windows.find(p_window_id);
	if (el == nullptr) {
		return;
	}

	if (current_window == p_window_id) {
		current_window = DisplayServer::INVALID_WINDOW_ID;
	}

	windows.erase(el);
}

void GLManagerEmbedded::release_current() {
	if (current_window == DisplayServer::INVALID_WINDOW_ID) {
		return;
	}

	[NSOpenGLContext clearCurrentContext];
	current_window = DisplayServer::INVALID_WINDOW_ID;
}

void GLManagerEmbedded::window_make_current(DisplayServer::WindowID p_window_id) {
	if (current_window == p_window_id) {
		return;
	}

	const GLWindowElement *el = windows.find(p_window_id);
	if (el == nullptr) {
		return;
	}

	const GLWindow &win = el->value();
	[win.context makeCurrentContext];

	current_window = p_window_id;
}

void GLManagerEmbedded::swap_buffers() {
	GLWindow &win = windows[current_window];
	[win.context flushBuffer];

	static bool last_valid = false;
	if (!win.is_valid) {
		if (last_valid) {
			ERR_PRINT("GLWindow framebuffers are invalid.");
			last_valid = false;
		}
		return;
	}
	last_valid = true;

	if (display_link_running) {
		dispatch_semaphore_wait(display_semaphore, DISPATCH_TIME_FOREVER);
	}

	[CATransaction begin];
	[CATransaction setDisableActions:YES];
	win.layer.contents = (__bridge id)win.framebuffers[win.current_fb].surface;
	[CATransaction commit];
	win.current_fb = (win.current_fb + 1) % BUFFER_COUNT;
	GLES3::TextureStorage::system_fbo = win.framebuffers[win.current_fb].fbo;
}

Error GLManagerEmbedded::initialize() {
	return framework_loaded ? OK : ERR_CANT_CREATE;
}

void GLManagerEmbedded::create_display_link() {
	DEV_ASSERT(display_link == nullptr);

	CVReturn err = CVDisplayLinkCreateWithCGDisplay(CGMainDisplayID(), &display_link);
	ERR_FAIL_COND_MSG(err != kCVReturnSuccess, "Failed to create display link.");

	__block dispatch_semaphore_t local_semaphore = display_semaphore;

	CVDisplayLinkSetOutputHandler(display_link, ^CVReturn(CVDisplayLinkRef p_display_link, const CVTimeStamp *p_now, const CVTimeStamp *p_output_time, CVOptionFlags p_flags, CVOptionFlags *p_flags_out) {
		dispatch_semaphore_signal(local_semaphore);
		return kCVReturnSuccess;
	});
}

void GLManagerEmbedded::release_display_link() {
	DEV_ASSERT(display_link != nullptr);
	if (CVDisplayLinkIsRunning(display_link)) {
		CVDisplayLinkStop(display_link);
	}
	CVDisplayLinkRelease(display_link);
	display_link = nullptr;
}

void GLManagerEmbedded::set_display_id(uint32_t p_display_id) {
	if (display_id == p_display_id) {
		return;
	}

	CVReturn err = CVDisplayLinkSetCurrentCGDisplay(display_link, static_cast<CGDirectDisplayID>(p_display_id));
	ERR_FAIL_COND_MSG(err != kCVReturnSuccess, "Failed to set display ID for display link.");
}

void GLManagerEmbedded::set_vsync_enabled(bool p_enabled) {
	if (p_enabled == vsync_enabled) {
		return;
	}

	vsync_enabled = p_enabled;

	if (vsync_enabled) {
		if (!CVDisplayLinkIsRunning(display_link)) {
			CVReturn err = CVDisplayLinkStart(display_link);
			ERR_FAIL_COND_MSG(err != kCVReturnSuccess, "Failed to start display link.");
			display_link_running = true;
		}
	} else {
		if (CVDisplayLinkIsRunning(display_link)) {
			CVReturn err = CVDisplayLinkStop(display_link);
			ERR_FAIL_COND_MSG(err != kCVReturnSuccess, "Failed to stop display link.");
			display_link_running = false;
		}
	}
}

GLManagerEmbedded::GLManagerEmbedded() {
	display_semaphore = dispatch_semaphore_create(BUFFER_COUNT);

	create_display_link();

	NSBundle *framework = [NSBundle bundleWithIdentifier:@"com.apple.opengl"];
	if ([framework load]) {
		void *library_handle = dlopen([framework.executablePath UTF8String], RTLD_NOW);
		if (library_handle) {
			CGLGetCurrentContext = (CGLGetCurrentContextPtr)dlsym(library_handle, "CGLGetCurrentContext");
			CGLTexImageIOSurface2D = (CGLTexImageIOSurface2DPtr)dlsym(library_handle, "CGLTexImageIOSurface2D");
			CGLErrorString = (CGLErrorStringPtr)dlsym(library_handle, "CGLErrorString");
			framework_loaded = CGLGetCurrentContext && CGLTexImageIOSurface2D && CGLErrorString;
		}
	}
}

GLManagerEmbedded::~GLManagerEmbedded() {
	release_display_link();
	release_current();
}

GODOT_CLANG_WARNING_POP

#endif // MACOS_ENABLED && GLES3_ENABLED
