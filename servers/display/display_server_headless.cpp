/**************************************************************************/
/*  display_server_headless.cpp                                           */
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

#include "display_server_headless.h"

#include "core/input/input.h"
#include "core/input/input_event.h"
#include "servers/display/native_menu.h"
#include "servers/rendering/dummy/rasterizer_dummy.h"

#if defined(RD_ENABLED)
#include "servers/rendering/renderer_rd/renderer_compositor_rd.h"
#include "servers/rendering/rendering_device.h"
#endif
#if defined(VULKAN_ENABLED)
#include "drivers/vulkan/rendering_context_driver_vulkan.h"
#endif
#if defined(D3D12_ENABLED)
#include "drivers/d3d12/rendering_context_driver_d3d12.h"
#endif
#if defined(METAL_ENABLED)
#include "drivers/metal/rendering_context_driver_metal.h"
#endif
#if defined(WEBGPU_ENABLED)
#include "drivers/webgpu/rendering_context_driver_webgpu.h"
#endif

DisplayServer *DisplayServerHeadless::create_func(const String &p_rendering_driver, DisplayServerEnums::WindowMode p_mode, DisplayServerEnums::VSyncMode p_vsync_mode, uint32_t p_flags, const Vector2i *p_position, const Vector2i &p_resolution, int p_screen, DisplayServerEnums::Context p_context, int64_t p_parent_window, Error &r_error) {
	r_error = OK;
	DisplayServerHeadless *display_server = memnew(DisplayServerHeadless);
#if defined(RD_ENABLED)
	// 2dog: an explicit rendering driver or method asks for a window-less rendering device, which the shader baker
	// needs. Plain headless runs keep the dummy rasterizer.
	const OS *os = OS::get_singleton();
	const bool rendering_requested = os->get_current_rendering_driver_name_source() == OS::RENDERING_SOURCE_COMMANDLINE || os->get_current_rendering_method_source() == OS::RENDERING_SOURCE_COMMANDLINE;
	if (rendering_requested && p_rendering_driver != "dummy" && !p_rendering_driver.begins_with("opengl3")) {
		if (display_server->_initialize_rendering_device(p_rendering_driver)) {
			return display_server;
		}
	}
#endif
	RasterizerDummy::make_current();
	return display_server;
}

#if defined(RD_ENABLED)
RenderingContextDriver *DisplayServerHeadless::_create_rendering_context(const String &p_driver) {
#if defined(VULKAN_ENABLED)
	if (p_driver == "vulkan") {
		return memnew(RenderingContextDriverVulkan);
	}
#endif
#if defined(D3D12_ENABLED)
	if (p_driver == "d3d12") {
		return memnew(RenderingContextDriverD3D12);
	}
#endif
#if defined(METAL_ENABLED)
	if (p_driver == "metal") {
		GODOT_CLANG_WARNING_PUSH_AND_IGNORE("-Wunguarded-availability")
		// Eliminate "RenderingContextDriverMetal is only available on iOS 14.0 or newer".
		RenderingContextDriver *context = memnew(RenderingContextDriverMetal);
		GODOT_CLANG_WARNING_POP
		return context;
	}
#endif
#if defined(WEBGPU_ENABLED)
	if (p_driver == "webgpu") {
		return memnew(RenderingContextDriverWebGPU);
	}
#endif
	return nullptr;
}

bool DisplayServerHeadless::_initialize_rendering_device(const String &p_driver) {
	// The requested driver first, then the other compiled-in rendering device drivers.
	Vector<String> candidates;
	candidates.push_back(p_driver);
	for (const char *name : { "vulkan", "d3d12", "metal", "webgpu" }) {
		if (p_driver != name) {
			candidates.push_back(name);
		}
	}
	for (const String &driver : candidates) {
		RenderingContextDriver *context = _create_rendering_context(driver);
		if (context == nullptr) {
			continue;
		}
		if (context->initialize() != OK) {
			memdelete(context);
			continue;
		}
		RenderingDevice *device = memnew(RenderingDevice);
		if (device->initialize(context) != OK) {
			memdelete(device);
			memdelete(context);
			continue;
		}
		if (driver != p_driver) {
			WARN_PRINT(vformat("Headless rendering: the %s driver is unavailable, using %s.", p_driver, driver));
			OS::get_singleton()->set_current_rendering_driver_name(driver, OS::RENDERING_SOURCE_FALLBACK);
		}
		rendering_context = context;
		rendering_device = device;
		RendererCompositorRD::make_current();
		return true;
	}
	WARN_PRINT(vformat("Headless rendering: no rendering device is available for %s. Shaders cannot be baked.", p_driver));
	return false;
}
#endif

void DisplayServerHeadless::_dispatch_input_events(const Ref<InputEvent> &p_event) {
	static_cast<DisplayServerHeadless *>(get_singleton())->_dispatch_input_event(p_event);
}

void DisplayServerHeadless::_dispatch_input_event(const Ref<InputEvent> &p_event) {
	if (input_event_callback.is_valid()) {
		input_event_callback.call(p_event);
	}
}

void DisplayServerHeadless::process_events() {
	Input::get_singleton()->flush_buffered_events();
}

DisplayServerHeadless::DisplayServerHeadless() {
	native_menu = memnew(NativeMenu);
	Input::get_singleton()->set_event_dispatch_function(_dispatch_input_events);
}

DisplayServerHeadless::~DisplayServerHeadless() {
	if (native_menu) {
		memdelete(native_menu);
		native_menu = nullptr;
	}
#if defined(RD_ENABLED)
	if (rendering_device) {
		memdelete(rendering_device);
		rendering_device = nullptr;
	}
	if (rendering_context) {
		memdelete(rendering_context);
		rendering_context = nullptr;
	}
#endif
}
