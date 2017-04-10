/*************************************************************************/
/*  context_gl_osx.cpp                                                   */
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
#include "context_gl_osx.h"

#ifdef OSX_ENABLED
#if defined(OPENGL_ENABLED) || defined(LEGACYGL_ENABLED)
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define GLX_CONTEXT_MAJOR_VERSION_ARB 0x2091
#define GLX_CONTEXT_MINOR_VERSION_ARB 0x2092

void ContextGL_OSX::release_current() {

	aglSetCurrentContext(context);
}

void ContextGL_OSX::make_current() {

	aglSetCurrentContext(NULL);
}
void ContextGL_OSX::swap_buffers() {

	aglSwapBuffers(context);
}

Error ContextGL_OSX::initialize() {

	if ((Ptr)kUnresolvedCFragSymbolAddress == (Ptr)aglChoosePixelFormat)
		return FAILED;

	GLint attributes[] = { AGL_RGBA,
		AGL_DOUBLEBUFFER,
		AGL_DEPTH_SIZE, 32,
		AGL_NO_RECOVERY,
		AGL_NONE,
		AGL_NONE };

	AGLPixelFormat format = NULL;

	format = aglChoosePixelFormat(NULL, 0, attributes);

	if (!format)
		return FAILED;

	context = aglCreateContext(format, 0);

	if (!context)
		return FAILED;

	aglDestroyPixelFormat(format);

	aglSetWindowRef(context, window);

	GLint swapInterval = 1;
	aglSetInteger(context, AGL_SWAP_INTERVAL, &swapInterval);

	aglSetCurrentContext(context);

	return OK;
}

ContextGL_OSX::ContextGL_OSX(WindowRef p_window) {

	window = p_window;
}

ContextGL_OSX::~ContextGL_OSX() {

	if (context)
		aglDestroyContext(context);
}

#endif
#endif
