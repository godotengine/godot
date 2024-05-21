/**************************************************************************/
/*  rendering_context_driver.h                                            */
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

#ifndef RENDERING_CONTEXT_DRIVER_H
#define RENDERING_CONTEXT_DRIVER_H

#include "core/object/object.h"
#include "servers/display_server.h"

class RenderingDeviceDriver;

class RenderingContextDriver {
public:
	typedef uint64_t SurfaceID;

private:
	HashMap<DisplayServer::WindowID, SurfaceID> window_surface_map;

public:
	SurfaceID surface_get_from_window(DisplayServer::WindowID p_window) const;
	Error window_create(DisplayServer::WindowID p_window, const void *p_platform_data);
	void window_set_size(DisplayServer::WindowID p_window, uint32_t p_width, uint32_t p_height);
	void window_set_vsync_mode(DisplayServer::WindowID p_window, DisplayServer::VSyncMode p_vsync_mode);
	DisplayServer::VSyncMode window_get_vsync_mode(DisplayServer::WindowID p_window) const;
	void window_destroy(DisplayServer::WindowID p_window);

public:
	enum Vendor {
		VENDOR_UNKNOWN = 0x0,
		VENDOR_AMD = 0x1002,
		VENDOR_IMGTEC = 0x1010,
		VENDOR_APPLE = 0x106B,
		VENDOR_NVIDIA = 0x10DE,
		VENDOR_ARM = 0x13B5,
		VENDOR_MICROSOFT = 0x1414,
		VENDOR_QUALCOMM = 0x5143,
		VENDOR_INTEL = 0x8086
	};

	enum DeviceType {
		DEVICE_TYPE_OTHER = 0x0,
		DEVICE_TYPE_INTEGRATED_GPU = 0x1,
		DEVICE_TYPE_DISCRETE_GPU = 0x2,
		DEVICE_TYPE_VIRTUAL_GPU = 0x3,
		DEVICE_TYPE_CPU = 0x4,
		DEVICE_TYPE_MAX = 0x5
	};

	struct Workarounds {
		bool avoid_compute_after_draw = false;
	};

	struct Device {
		String name = "Unknown";
		Vendor vendor = VENDOR_UNKNOWN;
		DeviceType type = DEVICE_TYPE_OTHER;
		Workarounds workarounds;
	};

	virtual ~RenderingContextDriver();
	virtual Error initialize() = 0;
	virtual const Device &device_get(uint32_t p_device_index) const = 0;
	virtual uint32_t device_get_count() const = 0;
	virtual bool device_supports_present(uint32_t p_device_index, SurfaceID p_surface) const = 0;
	virtual RenderingDeviceDriver *driver_create() = 0;
	virtual void driver_free(RenderingDeviceDriver *p_driver) = 0;
	virtual SurfaceID surface_create(const void *p_platform_data) = 0;
	virtual void surface_set_size(SurfaceID p_surface, uint32_t p_width, uint32_t p_height) = 0;
	virtual void surface_set_vsync_mode(SurfaceID p_surface, DisplayServer::VSyncMode p_vsync_mode) = 0;
	virtual DisplayServer::VSyncMode surface_get_vsync_mode(SurfaceID p_surface) const = 0;
	virtual uint32_t surface_get_width(SurfaceID p_surface) const = 0;
	virtual uint32_t surface_get_height(SurfaceID p_surface) const = 0;
	virtual void surface_set_needs_resize(SurfaceID p_surface, bool p_needs_resize) = 0;
	virtual bool surface_get_needs_resize(SurfaceID p_surface) const = 0;
	virtual void surface_destroy(SurfaceID p_surface) = 0;
	virtual bool is_debug_utils_enabled() const = 0;
};

#endif // RENDERING_CONTEXT_DRIVER_H
