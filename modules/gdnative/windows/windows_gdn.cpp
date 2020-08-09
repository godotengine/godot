/*************************************************************************/
/*  windows_gdn.cpp                                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "modules/gdnative/gdnative.h"

#ifdef WIN32
#include "platform/windows/os_windows.h"
/*
#else
#define HDC void *
#define HGLRC void *
*/
#endif

#ifdef __cplusplus
extern "C" {
#endif

void *GDAPI godot_windows_get_hdc() {
#if defined(WIN32) && (defined(OPENGL_ENABLED) || defined(GLES_ENABLED))
	OS_Windows *os_windows = (OS_Windows *)OS::get_singleton();
	return os_windows->get_gl_context()->get_hdc();
#else
	return NULL;
#endif
}

void *GDAPI godot_windows_get_hglrc() {
#if defined(WIN32) && (defined(OPENGL_ENABLED) || defined(GLES_ENABLED))
	OS_Windows *os_windows = (OS_Windows *)OS::get_singleton();
	return os_windows->get_gl_context()->get_hglrc();
#else
	return NULL;
#endif
}

#ifdef __cplusplus
}
#endif
