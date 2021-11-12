/**************************************************************************/
/*  gl_manager_windows_angle.cpp                                          */
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

#include "gl_manager_windows_angle.h"

#ifdef WINDOWS_ENABLED
#ifdef USE_OPENGL_ANGLE

#include <stdio.h>
#include <stdlib.h>

#include "thirdparty/angle/EGL/eglext_angle.h"

const char *GLManager_Windows::_get_platform_extension_name() const {
	return "EGL_ANGLE_platform_angle";
}

EGLenum GLManager_Windows::_get_platform_extension_enum() const {
	return EGL_PLATFORM_ANGLE_ANGLE;
}

Vector<EGLAttrib> GLManager_Windows::_get_platform_display_attributes() const {
	EGLint angle_platform_type = EGL_PLATFORM_ANGLE_TYPE_D3D11_ANGLE;

	List<String> args = OS::get_singleton()->get_cmdline_args();
	for (const List<String>::Element *E = args.front(); E; E = E->next()) {
		if (E->get() == "--angle-platform-type" && E->next()) {
			String cmd = E->next()->get().to_lower();
			if (cmd == "vulkan") {
				angle_platform_type = EGL_PLATFORM_ANGLE_TYPE_VULKAN_ANGLE;
			} else if (cmd == "d3d11") {
				angle_platform_type = EGL_PLATFORM_ANGLE_TYPE_D3D11_ANGLE;
			} else if (cmd == "opengl") {
				angle_platform_type = EGL_PLATFORM_ANGLE_TYPE_OPENGL_ANGLE;
			} else {
				WARN_PRINT("Invalid ANGLE platform type, it should be \"vulkan\", \"d3d9\", \"d3d11\" or \"opengl\".");
			}
		}
	}

	Vector<EGLAttrib> ret;
	ret.push_back(EGL_PLATFORM_ANGLE_TYPE_ANGLE);
	ret.push_back(angle_platform_type);
	ret.push_back(EGL_NONE);

	return ret;
}

EGLenum GLManager_Windows::_get_platform_api_enum() const {
	return EGL_OPENGL_ES_API;
}

Vector<EGLint> GLManager_Windows::_get_platform_context_attribs() const {
	Vector<EGLint> ret;
	ret.push_back(EGL_CONTEXT_CLIENT_VERSION);
	ret.push_back(3);
	ret.push_back(EGL_NONE);

	return ret;
}

#endif // USE_OPENGL_ANGLE
#endif // WINDOWS_ENABLED
