/**************************************************************************/
/*  gl_manager_x11_egl.cpp                                                */
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

#include "gl_manager_x11_egl.h"

#if defined(X11_ENABLED) && defined(GLES3_ENABLED)

#include <stdio.h>
#include <stdlib.h>

const char *GLManagerEGL_X11::_get_platform_extension_name() const {
	return "EGL_KHR_platform_x11";
}

EGLenum GLManagerEGL_X11::_get_platform_extension_enum() const {
	return EGL_PLATFORM_X11_KHR;
}

Vector<EGLAttrib> GLManagerEGL_X11::_get_platform_display_attributes() const {
	return Vector<EGLAttrib>();
}

EGLenum GLManagerEGL_X11::_get_platform_api_enum() const {
	return EGL_OPENGL_ES_API;
}

Vector<EGLint> GLManagerEGL_X11::_get_platform_context_attribs() const {
	Vector<EGLint> ret;
	ret.push_back(EGL_CONTEXT_CLIENT_VERSION);
	ret.push_back(3);
	ret.push_back(EGL_NONE);

	return ret;
}

#endif // WINDOWS_ENABLED && GLES3_ENABLED
