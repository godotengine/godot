/*************************************************************************/
/*  egl_manager_wayland.cpp                                              */
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

#include "gl_manager_wayland.h"

const char *EGLManagerWayland::_get_platform_extension_name() const {
	return "EGL_KHR_platform_wayland";
}

EGLenum EGLManagerWayland::_get_platform_extension_enum() const {
	return EGL_PLATFORM_WAYLAND_KHR;
}

Error EGLManagerWayland::initialize() {
	String extensions_string = eglQueryString(EGL_NO_DISPLAY, EGL_EXTENSIONS);
	// The above method should always work. If it doesn't, something's very wrong.
	ERR_FAIL_COND_V(eglGetError() != EGL_SUCCESS, ERR_BUG);

	const char *platform = _get_platform_extension_name();
	if (!extensions_string.split(" ").find(platform)) {
		ERR_FAIL_V_MSG(ERR_UNAVAILABLE, vformat("EGL platform extension \"%s\" not found.", platform));
	}

	return OK;
}
