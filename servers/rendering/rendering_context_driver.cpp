/**************************************************************************/
/*  rendering_context_driver.cpp                                          */
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

#include "rendering_context_driver.h"

RenderingContextDriver::~RenderingContextDriver() {
}

RenderingContextDriver::SurfaceID RenderingContextDriver::surface_get_from_window(DisplayServer::WindowID p_window) const {
	HashMap<DisplayServer::WindowID, SurfaceID>::ConstIterator it = window_surface_map.find(p_window);
	if (it != window_surface_map.end()) {
		return it->value;
	} else {
		return SurfaceID();
	}
}

Error RenderingContextDriver::window_create(DisplayServer::WindowID p_window, const void *p_platform_data) {
	SurfaceID surface = surface_create(p_platform_data);
	if (surface != 0) {
		window_surface_map[p_window] = surface;
		return OK;
	} else {
		return ERR_CANT_CREATE;
	}
}

void RenderingContextDriver::window_set_size(DisplayServer::WindowID p_window, uint32_t p_width, uint32_t p_height) {
	SurfaceID surface = surface_get_from_window(p_window);
	if (surface) {
		surface_set_size(surface, p_width, p_height);
	}
}

void RenderingContextDriver::window_set_vsync_mode(DisplayServer::WindowID p_window, DisplayServer::VSyncMode p_vsync_mode) {
	SurfaceID surface = surface_get_from_window(p_window);
	if (surface) {
		surface_set_vsync_mode(surface, p_vsync_mode);
	}
}

DisplayServer::VSyncMode RenderingContextDriver::window_get_vsync_mode(DisplayServer::WindowID p_window) const {
	SurfaceID surface = surface_get_from_window(p_window);
	if (surface) {
		return surface_get_vsync_mode(surface);
	} else {
		return DisplayServer::VSYNC_DISABLED;
	}
}

void RenderingContextDriver::window_destroy(DisplayServer::WindowID p_window) {
	SurfaceID surface = surface_get_from_window(p_window);
	if (surface) {
		surface_destroy(surface);
	}

	window_surface_map.erase(p_window);
}

const char *RenderingContextDriver::get_tracked_object_name(uint32_t p_type_index) const {
	return "Tracking Unsupported by API";
}

uint64_t RenderingContextDriver::get_tracked_object_type_count() const {
	return 0;
}

uint64_t RenderingContextDriver::get_driver_total_memory() const {
	return 0;
}

uint64_t RenderingContextDriver::get_driver_allocation_count() const {
	return 0;
}

uint64_t RenderingContextDriver::get_driver_memory_by_object_type(uint32_t) const {
	return 0;
}

uint64_t RenderingContextDriver::get_driver_allocs_by_object_type(uint32_t) const {
	return 0;
}

uint64_t RenderingContextDriver::get_device_total_memory() const {
	return 0;
}

uint64_t RenderingContextDriver::get_device_allocation_count() const {
	return 0;
}

uint64_t RenderingContextDriver::get_device_memory_by_object_type(uint32_t) const {
	return 0;
}

uint64_t RenderingContextDriver::get_device_allocs_by_object_type(uint32_t) const {
	return 0;
}
