/*************************************************************************/
/*  context_gl_osx.mm                                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "context_gl_osx.h"

#if defined(GLES3_ENABLED) || defined(GLES_ENABLED)

void ContextGL_OSX::release_current() {
	[NSOpenGLContext clearCurrentContext];
}

void ContextGL_OSX::make_current() {
	[context makeCurrentContext];
}

void ContextGL_OSX::update() {
	[context update];
}

void ContextGL_OSX::set_opacity(GLint p_opacity) {
	[context setValues:&p_opacity forParameter:NSOpenGLCPSurfaceOpacity];
}

int ContextGL_OSX::get_window_width() {
	return OS::get_singleton()->get_video_mode().width;
}

int ContextGL_OSX::get_window_height() {
	return OS::get_singleton()->get_video_mode().height;
}

void ContextGL_OSX::swap_buffers() {
	[context flushBuffer];
}

void ContextGL_OSX::set_use_vsync(bool p_use) {
	CGLContextObj ctx = CGLGetCurrentContext();
	if (ctx) {
		GLint swapInterval = p_use ? 1 : 0;
		CGLSetParameter(ctx, kCGLCPSwapInterval, &swapInterval);
		use_vsync = p_use;
	}
}

bool ContextGL_OSX::is_using_vsync() const {
	return use_vsync;
}

Error ContextGL_OSX::initialize() {
	framework = CFBundleGetBundleWithIdentifier(CFSTR("com.apple.opengl"));
	ERR_FAIL_COND_V(!framework, ERR_CANT_CREATE);

	unsigned int attributeCount = 0;

	// OS X needs non-zero color size, so set reasonable values
	int colorBits = 32;

	// Fail if a robustness strategy was requested

#define ADD_ATTR(x) \
	{ attributes[attributeCount++] = x; }
#define ADD_ATTR2(x, y) \
	{                   \
		ADD_ATTR(x);    \
		ADD_ATTR(y);    \
	}

	// Arbitrary array size here
	NSOpenGLPixelFormatAttribute attributes[40];

	ADD_ATTR(NSOpenGLPFADoubleBuffer);
	ADD_ATTR(NSOpenGLPFAClosestPolicy);

	if (!gles3_context) {
		ADD_ATTR2(NSOpenGLPFAOpenGLProfile, NSOpenGLProfileVersionLegacy);
	} else {
		//we now need OpenGL 3 or better, maybe even change this to 3_3Core ?
		ADD_ATTR2(NSOpenGLPFAOpenGLProfile, NSOpenGLProfileVersion3_2Core);
	}

	ADD_ATTR2(NSOpenGLPFAColorSize, colorBits);

	/*
	if (fbconfig->alphaBits > 0)
		ADD_ATTR2(NSOpenGLPFAAlphaSize, fbconfig->alphaBits);
*/

	ADD_ATTR2(NSOpenGLPFADepthSize, 24);

	ADD_ATTR2(NSOpenGLPFAStencilSize, 8);

	/*
	if (fbconfig->stereo)
		ADD_ATTR(NSOpenGLPFAStereo);
*/

	/*
	if (fbconfig->samples > 0) {
		ADD_ATTR2(NSOpenGLPFASampleBuffers, 1);
		ADD_ATTR2(NSOpenGLPFASamples, fbconfig->samples);
	}
*/

	// NOTE: All NSOpenGLPixelFormats on the relevant cards support sRGB
	//       framebuffer, so there's no need (and no way) to request it

	ADD_ATTR(0);

#undef ADD_ATTR
#undef ADD_ATTR2

	pixelFormat = [[NSOpenGLPixelFormat alloc] initWithAttributes:attributes];
	ERR_FAIL_COND_V(pixelFormat == nil, ERR_CANT_CREATE);

	context = [[NSOpenGLContext alloc] initWithFormat:pixelFormat shareContext:nil];

	ERR_FAIL_COND_V(context == nil, ERR_CANT_CREATE);

	[context setView:window_view];

	[context makeCurrentContext];

	return OK;
}

ContextGL_OSX::ContextGL_OSX(id p_view, bool p_gles3_context) {
	gles3_context = p_gles3_context;
	window_view = p_view;
	use_vsync = false;
}

ContextGL_OSX::~ContextGL_OSX() {}

#endif
