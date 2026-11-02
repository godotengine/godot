/**************************************************************************/
/*  rendering_context_driver_webgpu.h                                     */
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

#pragma once

#ifdef WEBGPU_ENABLED

#include "servers/rendering/rendering_context_driver.h"

#include <webgpu/webgpu.h>

class RenderingContextDriverWebGPU : public RenderingContextDriver {
	WGPUInstance instance = nullptr;
	WGPUAdapter adapter = nullptr;
	WGPUDevice device = nullptr;
	WGPUQueue queue = nullptr;

	Device device_info; // Single device description.
	String dawn_cache_dir; // Native Dawn's compiled shader and pipeline cache, empty when disabled.
	Error initialize_native();
	SurfaceID surface_create_native(const void *p_platform_data);

	struct Surface {
		String canvas_selector;
		WGPUSurface handle = nullptr;
		uint32_t width = 0;
		uint32_t height = 0;
		DisplayServerEnums::VSyncMode vsync_mode = DisplayServerEnums::VSYNC_ENABLED;
		bool needs_resize = false;
		WGPUTextureFormat format = WGPUTextureFormat_BGRA8Unorm;
		WGPUCompositeAlphaMode alpha_mode = WGPUCompositeAlphaMode_Opaque;
		Vector<WGPUPresentMode> present_modes;
	};

	HashMap<SurfaceID, Surface> surfaces;
	SurfaceID next_surface_id = 1;

public:
	// Plain platform data so the desktop display servers can put it in their
	// existing window-data union without depending on native SDK headers.
	struct WindowPlatformData {
		enum Kind { KIND_WIN32,
			KIND_XLIB,
			KIND_WAYLAND,
			KIND_METAL_LAYER } kind;
		void *display;
		void *window;
		uint64_t window_id;
	};
	RenderingContextDriverWebGPU();
	virtual ~RenderingContextDriverWebGPU() override;

	// --- RenderingContextDriver interface ---

	virtual Error initialize() override final;
	virtual const Device &device_get(uint32_t p_device_index) const override final;
	virtual uint32_t device_get_count() const override final;
	virtual bool device_supports_present(uint32_t p_device_index, SurfaceID p_surface) const override final;
	virtual RenderingDeviceDriver *driver_create() override final;
	virtual void driver_free(RenderingDeviceDriver *p_driver) override final;
	virtual SurfaceID surface_create(const void *p_platform_data) override final;
	virtual void surface_set_size(SurfaceID p_surface, uint32_t p_width, uint32_t p_height) override final;
	virtual void surface_set_vsync_mode(SurfaceID p_surface, DisplayServerEnums::VSyncMode p_vsync_mode) override final;
	virtual DisplayServerEnums::VSyncMode surface_get_vsync_mode(SurfaceID p_surface) const override final;
	virtual uint32_t surface_get_width(SurfaceID p_surface) const override final;
	virtual uint32_t surface_get_height(SurfaceID p_surface) const override final;
	virtual void surface_set_needs_resize(SurfaceID p_surface, bool p_needs_resize) override final;
	virtual bool surface_get_needs_resize(SurfaceID p_surface) const override final;
	virtual void surface_destroy(SurfaceID p_surface) override final;
	virtual bool is_debug_utils_enabled() const override final;

	// WebGPU canvas output remains SDR.
	virtual void surface_set_hdr_output_enabled(SurfaceID p_surface, bool p_enabled) override final {}
	virtual bool surface_get_hdr_output_enabled(SurfaceID p_surface) const override final { return false; }
	virtual void surface_set_hdr_output_reference_luminance(SurfaceID p_surface, float p_reference_luminance) override final {}
	virtual float surface_get_hdr_output_reference_luminance(SurfaceID p_surface) const override final { return 80.0f; }
	virtual void surface_set_hdr_output_max_luminance(SurfaceID p_surface, float p_max_luminance) override final {}
	virtual float surface_get_hdr_output_max_luminance(SurfaceID p_surface) const override final { return 80.0f; }
	virtual void surface_set_hdr_output_linear_luminance_scale(SurfaceID p_surface, float p_linear_luminance_scale) override final {}
	virtual float surface_get_hdr_output_linear_luminance_scale(SurfaceID p_surface) const override final { return 1.0f; }
	virtual float surface_get_hdr_output_max_value(SurfaceID p_surface) const override final { return 1.0f; }

	// --- Accessors for RenderingDeviceDriverWebGPU ---

	WGPUDevice get_device() const { return device; }
	WGPUQueue get_queue() const { return queue; }
	WGPUInstance get_instance() const { return instance; }
	WGPUSurface surface_get_handle(SurfaceID p_surface) const;
	WGPUTextureFormat surface_get_format(SurfaceID p_surface) const;
	WGPUCompositeAlphaMode surface_get_alpha_mode(SurfaceID p_surface) const;
	WGPUPresentMode surface_get_present_mode(SurfaceID p_surface) const;
};

#endif // WEBGPU_ENABLED
