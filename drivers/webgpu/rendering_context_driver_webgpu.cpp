/**************************************************************************/
/*  rendering_context_driver_webgpu.cpp                                   */
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

#ifdef WEBGPU_ENABLED

#include "rendering_context_driver_webgpu.h"

#include "drivers/webgpu/rendering_device_driver_webgpu.h"

// html5_webgpu.h was removed in Emscripten 5.x when USE_WEBGPU was dropped.
// Device is now created from C++ using the emdawnwebgpu port's Dawn API.
#ifdef WEB_ENABLED
#include <emscripten/emscripten.h>
#include <emscripten/html5.h>
#include <cstdlib>
EM_JS_DEPS(webgpu_context, "$stringToNewUTF8");
#endif

RenderingContextDriverWebGPU::RenderingContextDriverWebGPU() {
}

RenderingContextDriverWebGPU::~RenderingContextDriverWebGPU() {
	for (const auto &surface : surfaces) {
		wgpuSurfaceUnconfigure(surface.value.handle);
		wgpuSurfaceRelease(surface.value.handle);
	}
	if (queue) {
		wgpuQueueRelease(queue);
		queue = nullptr;
	}
	if (device) {
#ifndef WEB_ENABLED
		wgpuDeviceDestroy(device);
		wgpuInstanceProcessEvents(instance);
#endif
		wgpuDeviceRelease(device);
		device = nullptr;
	}
	if (adapter) {
		wgpuAdapterRelease(adapter);
		adapter = nullptr;
	}
	if (instance) {
		wgpuInstanceRelease(instance);
		instance = nullptr;
	}
}

Error RenderingContextDriverWebGPU::initialize() {
#ifndef WEB_ENABLED
	return initialize_native();
#else
	// The HTML shell pre-initializes a GPUDevice and stores it in Module.preinitializedWebGPUDevice.
	// We use the emdawnwebgpu port's WebGPU.importJsDevice() to wrap it in a C WGPUDevice handle.
	// Note: emdawnwebgpu is a thin JS wrapper around the browser's WebGPU API.
	// SPIR-V is NOT supported — we use Tint (C++, linked in) for SPIR-V → WGSL
	// conversion in shader_create_from_container() instead.
	// Create a WGPUInstance — needed for wgpuInstanceProcessEvents() which
	// processes async callbacks (buffer map readback, query results, etc.).
	WGPUInstanceDescriptor inst_desc = {};
	instance = wgpuCreateInstance(&inst_desc);
	if (!instance) {
		WARN_PRINT("WebGPU: wgpuCreateInstance returned null — async readback may not work.");
	}

	device = (WGPUDevice)(uintptr_t)EM_ASM_PTR({
		var d = Module["preinitializedWebGPUDevice"];
		if (!d) {
			return 0;
		}
		return WebGPU["importJsDevice"](d);
	});
	ERR_FAIL_COND_V_MSG(device == nullptr, ERR_CANT_CREATE, "WebGPU: Failed to get pre-initialized device. Ensure JS shell calls navigator.gpu.requestDevice() before WASM.");

	queue = wgpuDeviceGetQueue(device);
	ERR_FAIL_COND_V_MSG(queue == nullptr, ERR_CANT_CREATE, "WebGPU: Failed to get device queue.");

	// Populate device info.
	char *adapter_name = (char *)EM_ASM_PTR({
		var info = Module['preinitializedWebGPUDevice']['godotAdapterInfo'] || {};
		return stringToNewUTF8([info.vendor, info.description || info.architecture || info.device].filter(Boolean).join(' ') || 'WebGPU Device');
	});
	device_info.name = String::utf8(adapter_name);
	free(adapter_name);
	device_info.vendor = EM_ASM_INT({
		var vendor = (Module['preinitializedWebGPUDevice']['godotAdapterInfo'] || {}).vendor || '';
		vendor = vendor.toLowerCase();
		if (vendor.includes('nvidia')) return 0x10de;
		if (vendor.includes('amd') || vendor.startsWith('ati')) return 0x1002;
		if (vendor.includes('intel')) return 0x8086;
		if (vendor.includes('apple')) return 0x106b;
		if (vendor.includes('qualcomm')) return 0x5143;
		if (vendor.includes('arm')) return 0x13b5;
		return 0;
	});
	device_info.type = DEVICE_TYPE_INTEGRATED_GPU;

	print_verbose("WebGPU: Device imported from JS successfully.");
	return OK;
#endif
}

const RenderingContextDriver::Device &RenderingContextDriverWebGPU::device_get(uint32_t p_device_index) const {
	DEV_ASSERT(p_device_index == 0);
	return device_info;
}

uint32_t RenderingContextDriverWebGPU::device_get_count() const {
	return 1; // Browser exposes a single GPU context.
}

bool RenderingContextDriverWebGPU::device_supports_present(uint32_t p_device_index, SurfaceID p_surface) const {
	return true; // Single device always supports the canvas surface.
}

RenderingDeviceDriver *RenderingContextDriverWebGPU::driver_create() {
	return memnew(RenderingDeviceDriverWebGPU(this));
}

void RenderingContextDriverWebGPU::driver_free(RenderingDeviceDriver *p_driver) {
	memdelete(p_driver);
}

RenderingContextDriver::SurfaceID RenderingContextDriverWebGPU::surface_create(const void *p_platform_data) {
#ifndef WEB_ENABLED
	return surface_create_native(p_platform_data);
#else
	// Use the configured canvas selector; embedded hosts can have distinct IDs.
	const char *canvas_selector = p_platform_data ? static_cast<const char *>(p_platform_data) : "#canvas";

	// Emscripten 5.x / emdawnwebgpu renamed this struct.
	WGPUEmscriptenSurfaceSourceCanvasHTMLSelector canvas_desc = {};
	canvas_desc.chain.sType = WGPUSType_EmscriptenSurfaceSourceCanvasHTMLSelector;
	canvas_desc.selector = WGPUStringView{ canvas_selector, WGPU_STRLEN };

	WGPUSurfaceDescriptor surface_desc = {};
	surface_desc.nextInChain = (WGPUChainedStruct *)&canvas_desc;

	// Note: We need an instance to create a surface. If we don't have one,
	// create a minimal one. In Emscripten, the instance is a lightweight wrapper.
	if (instance == nullptr) {
		WGPUInstanceDescriptor inst_desc = {};
		instance = wgpuCreateInstance(&inst_desc);
	}

	WGPUSurface wgpu_surface = wgpuInstanceCreateSurface(instance, &surface_desc);
	ERR_FAIL_COND_V_MSG(wgpu_surface == nullptr, 0, "WebGPU: Failed to create surface from canvas.");

	SurfaceID id = next_surface_id++;
	Surface &surface = surfaces[id];
	surface.handle = wgpu_surface;
	surface.canvas_selector = canvas_selector;
	surface.format = EM_ASM_INT({ return navigator.gpu.getPreferredCanvasFormat() === 'rgba8unorm'; }) ? WGPUTextureFormat_RGBA8Unorm : WGPUTextureFormat_BGRA8Unorm;
	surface.width = 0;
	surface.height = 0;
	surface.needs_resize = true;

	return id;
#endif
}

void RenderingContextDriverWebGPU::surface_set_size(SurfaceID p_surface, uint32_t p_width, uint32_t p_height) {
	ERR_FAIL_COND(!surfaces.has(p_surface));
	Surface &surface = surfaces[p_surface];
#ifdef WEB_ENABLED
	int canvas_width = 0;
	int canvas_height = 0;
	if (emscripten_get_canvas_element_size(surface.canvas_selector.utf8().get_data(), &canvas_width, &canvas_height) == EMSCRIPTEN_RESULT_SUCCESS) {
		p_width = MAX(canvas_width, 0);
		p_height = MAX(canvas_height, 0);
	}
#endif
	if (surface.width != p_width || surface.height != p_height) {
		surface.width = p_width;
		surface.height = p_height;
		surface.needs_resize = true;
	}
}

void RenderingContextDriverWebGPU::surface_set_vsync_mode(SurfaceID p_surface, DisplayServerEnums::VSyncMode p_vsync_mode) {
	ERR_FAIL_COND(!surfaces.has(p_surface));
	if (surfaces[p_surface].vsync_mode != p_vsync_mode) {
		surfaces[p_surface].vsync_mode = p_vsync_mode;
		surfaces[p_surface].needs_resize = true;
	}
}

DisplayServerEnums::VSyncMode RenderingContextDriverWebGPU::surface_get_vsync_mode(SurfaceID p_surface) const {
	ERR_FAIL_COND_V(!surfaces.has(p_surface), DisplayServerEnums::VSYNC_ENABLED);
	return surfaces[p_surface].vsync_mode;
}

uint32_t RenderingContextDriverWebGPU::surface_get_width(SurfaceID p_surface) const {
	ERR_FAIL_COND_V(!surfaces.has(p_surface), 0);
	return surfaces[p_surface].width;
}

uint32_t RenderingContextDriverWebGPU::surface_get_height(SurfaceID p_surface) const {
	ERR_FAIL_COND_V(!surfaces.has(p_surface), 0);
	return surfaces[p_surface].height;
}

void RenderingContextDriverWebGPU::surface_set_needs_resize(SurfaceID p_surface, bool p_needs_resize) {
	ERR_FAIL_COND(!surfaces.has(p_surface));
	surfaces[p_surface].needs_resize = p_needs_resize;
}

bool RenderingContextDriverWebGPU::surface_get_needs_resize(SurfaceID p_surface) const {
	ERR_FAIL_COND_V(!surfaces.has(p_surface), false);
	return surfaces[p_surface].needs_resize;
}

void RenderingContextDriverWebGPU::surface_destroy(SurfaceID p_surface) {
	ERR_FAIL_COND(!surfaces.has(p_surface));
	Surface &surface = surfaces[p_surface];
	if (surface.handle) {
		wgpuSurfaceRelease(surface.handle);
	}
	surfaces.erase(p_surface);
}

bool RenderingContextDriverWebGPU::is_debug_utils_enabled() const {
	return false; // No debug utils in browser WebGPU.
}

WGPUSurface RenderingContextDriverWebGPU::surface_get_handle(SurfaceID p_surface) const {
	const Surface *s = surfaces.getptr(p_surface);
	ERR_FAIL_COND_V(s == nullptr, nullptr);
	return s->handle;
}

WGPUTextureFormat RenderingContextDriverWebGPU::surface_get_format(SurfaceID p_surface) const {
	const Surface *surface = surfaces.getptr(p_surface);
	ERR_FAIL_NULL_V(surface, WGPUTextureFormat_Undefined);
	return surface->format;
}

WGPUCompositeAlphaMode RenderingContextDriverWebGPU::surface_get_alpha_mode(SurfaceID p_surface) const {
	const Surface *surface = surfaces.getptr(p_surface);
	ERR_FAIL_NULL_V(surface, WGPUCompositeAlphaMode_Opaque);
	return surface->alpha_mode;
}

WGPUPresentMode RenderingContextDriverWebGPU::surface_get_present_mode(SurfaceID p_surface) const {
	const Surface *surface = surfaces.getptr(p_surface);
	ERR_FAIL_NULL_V(surface, WGPUPresentMode_Fifo);
	if (surface->vsync_mode == DisplayServerEnums::VSYNC_DISABLED && surface->present_modes.has(WGPUPresentMode_Immediate)) {
		return WGPUPresentMode_Immediate;
	}
	if (surface->vsync_mode == DisplayServerEnums::VSYNC_MAILBOX && surface->present_modes.has(WGPUPresentMode_Mailbox)) {
		return WGPUPresentMode_Mailbox;
	}
	return WGPUPresentMode_Fifo;
}

#endif // WEBGPU_ENABLED
