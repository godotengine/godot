/**************************************************************************/
/*  renderer_compositor.cpp                                               */
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

#include "renderer_compositor.h"

#include "core/config/project_settings.h"

#ifndef XR_DISABLED
#include "servers/xr/xr_server.h"
#endif // XR_DISABLED

RendererCompositor *RendererCompositor::singleton = nullptr;

RendererCompositor *(*RendererCompositor::_create_func)() = nullptr;
bool RendererCompositor::low_end = false;

RendererCompositor *RendererCompositor::create() {
	return _create_func();
}

void RendererCompositor::set_boot_image(const Ref<Image> &p_image, const Color &p_color, bool p_scale, bool p_use_filter) {
	RenderingServer::SplashStretchMode stretch_mode = RenderingServer::map_scaling_option_to_stretch_mode(p_scale);
	set_boot_image_with_stretch(p_image, p_color, stretch_mode, p_use_filter);
}

bool RendererCompositor::is_xr_enabled() const {
	return xr_enabled;
}

RendererCompositor::RendererCompositor() {
	ERR_FAIL_COND_MSG(singleton != nullptr, "A RendererCompositor singleton already exists.");
	singleton = this;

#ifndef XR_DISABLED
	if (XRServer::get_xr_mode() == XRServer::XRMODE_DEFAULT) {
		xr_enabled = GLOBAL_GET("xr/shaders/enabled");
	} else {
		xr_enabled = XRServer::get_xr_mode() == XRServer::XRMODE_ON;
	}
#endif // XR_DISABLED
}

RendererCompositor::~RendererCompositor() {
	singleton = nullptr;
}
