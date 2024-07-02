/**************************************************************************/
/*  api.cpp                                                               */
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

#include "api.h"

#ifdef LINUXBSD_ENABLED
#include "core/object/class_db.h"

#ifdef WAYLAND_ENABLED
#include "platform/linuxbsd/wayland/rendering_native_surface_wayland.h"
#endif

#ifdef X11_ENABLED
#include "platform/linuxbsd/x11/rendering_native_surface_x11.h"
#endif

#endif

void register_core_linuxbsd_api() {
#ifdef LINUXBSD_ENABLED
#ifdef WAYLAND_ENABLED
	GDREGISTER_ABSTRACT_CLASS(RenderingNativeSurfaceWayland);
#endif
#ifdef X11_ENABLED
	GDREGISTER_ABSTRACT_CLASS(RenderingNativeSurfaceX11);
#endif
#endif
}

void unregister_core_linuxbsd_api() {
}

void register_linuxbsd_api() {
}

void unregister_linuxbsd_api() {
}
