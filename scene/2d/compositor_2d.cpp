/**************************************************************************/
/*  compositor_2d.cpp                                                     */
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

#include "compositor_2d.h"

void Compositor2D::_render_callback(RenderCanvasDataRD *p_data) {
	GDVIRTUAL_CALL(_render_callback, p_data);
}

void Compositor2D::_notification(int p_what) {
	if (p_what == NOTIFICATION_DRAW) {
		RenderingServer::get_singleton()->canvas_item_add_rendering_callback(get_canvas_item(), callable_mp(this, &Compositor2D::_render_callback));
	}
}

void Compositor2D::enable_backbuffer() {
	RenderingServer::get_singleton()->canvas_item_set_copy_to_backbuffer(get_canvas_item(), true, Rect2());
}

void Compositor2D::disable_backbuffer() {
	RenderingServer::get_singleton()->canvas_item_set_copy_to_backbuffer(get_canvas_item(), false, Rect2());
}

PackedStringArray Compositor2D::get_configuration_warnings() const {
	PackedStringArray warnings = Node2D::get_configuration_warnings();

	if (OS::get_singleton()->get_current_rendering_method() == "gl_compatibility" || OS::get_singleton()->get_current_rendering_method() == "dummy") {
		warnings.push_back(RTR("Compositor2D only works when using the Forward+ or Mobile renderer."));
	}

	return warnings;
}

void Compositor2D::_bind_methods() {
	GDVIRTUAL_BIND(_render_callback, "render_data");

	ClassDB::bind_method(D_METHOD("enable_backbuffer"), &Compositor2D::enable_backbuffer);
	ClassDB::bind_method(D_METHOD("disable_backbuffer"), &Compositor2D::disable_backbuffer);
}
