/**************************************************************************/
/*  rendering_context_driver_d3d12.h                                      */
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

#include "core/os/mutex.h"
#include "core/string/ustring.h"
#include "core/templates/rid_owner.h"
#include "rendering_device_driver_d3d12.h"
#include "servers/display/display_server.h"
#include "servers/rendering/rendering_context_driver.h"

#if !defined(_MSC_VER) && !defined(__REQUIRED_RPCNDR_H_VERSION__)
// Match current version used by MinGW, MSVC and Direct3D 12 headers use 500.
#define __REQUIRED_RPCNDR_H_VERSION__ 475
#endif // !defined(_MSC_VER) && !defined(__REQUIRED_RPCNDR_H_VERSION__)

GODOT_GCC_WARNING_PUSH
GODOT_GCC_WARNING_IGNORE("-Wimplicit-fallthrough")
GODOT_GCC_WARNING_IGNORE("-Wmissing-field-initializers")
GODOT_GCC_WARNING_IGNORE("-Wnon-virtual-dtor")
GODOT_GCC_WARNING_IGNORE("-Wshadow")
GODOT_GCC_WARNING_IGNORE("-Wswitch")
GODOT_CLANG_WARNING_PUSH
GODOT_CLANG_WARNING_IGNORE("-Wimplicit-fallthrough")
GODOT_CLANG_WARNING_IGNORE("-Wmissing-field-initializers")
GODOT_CLANG_WARNING_IGNORE("-Wnon-virtual-dtor")
GODOT_CLANG_WARNING_IGNORE("-Wstring-plus-int")
GODOT_CLANG_WARNING_IGNORE("-Wswitch")

#include <thirdparty/directx_headers/include/directx/d3dx12.h>

GODOT_GCC_WARNING_POP
GODOT_CLANG_WARNING_POP

#if defined(AS)
#undef AS
#endif

#ifdef DCOMP_ENABLED
#include <dcomp.h>
#endif

#include <wrl/client.h>

#define ARRAY_SIZE(a) std_size(a)

class RenderingContextDriverD3D12 : public RenderingContextDriver {
	Microsoft::WRL::ComPtr<ID3D12DeviceFactory> device_factory;
	Microsoft::WRL::ComPtr<IDXGIFactory2> dxgi_factory;
	TightLocalVector<Device> driver_devices;
	bool tearing_supported = false;

	Error _init_device_factory();
	Error _initialize_debug_layers();
	Error _initialize_devices();

public:
	virtual Error initialize() override;
	virtual const Device &device_get(uint32_t p_device_index) const override;
	virtual uint32_t device_get_count() const override;
	virtual bool device_supports_present(uint32_t p_device_index, SurfaceID p_surface) const override;
	virtual RenderingDeviceDriver *driver_create() override;
	virtual void driver_free(RenderingDeviceDriver *p_driver) override;
	virtual SurfaceID surface_create(const void *p_platform_data) override;
	virtual void surface_set_size(SurfaceID p_surface, uint32_t p_width, uint32_t p_height) override;
	virtual void surface_set_vsync_mode(SurfaceID p_surface, DisplayServer::VSyncMode p_vsync_mode) override;
	virtual DisplayServer::VSyncMode surface_get_vsync_mode(SurfaceID p_surface) const override;
	virtual uint32_t surface_get_width(SurfaceID p_surface) const override;
	virtual uint32_t surface_get_height(SurfaceID p_surface) const override;
	virtual void surface_set_needs_resize(SurfaceID p_surface, bool p_needs_resize) override;
	virtual bool surface_get_needs_resize(SurfaceID p_surface) const override;
	virtual void surface_destroy(SurfaceID p_surface) override;
	virtual bool is_debug_utils_enabled() const override;

	// Platform-specific data for the Windows embedded in this driver.
	struct WindowPlatformData {
		HWND window;
	};

	// D3D12-only methods.
	struct Surface {
		HWND hwnd = nullptr;
		uint32_t width = 0;
		uint32_t height = 0;
		DisplayServer::VSyncMode vsync_mode = DisplayServer::VSYNC_ENABLED;
		bool needs_resize = false;
#ifdef DCOMP_ENABLED
		Microsoft::WRL::ComPtr<IDCompositionDevice> composition_device;
		Microsoft::WRL::ComPtr<IDCompositionTarget> composition_target;
		Microsoft::WRL::ComPtr<IDCompositionVisual> composition_visual;
#endif
	};

	HMODULE lib_d3d12 = nullptr;
	HMODULE lib_dxgi = nullptr;
#ifdef DCOMP_ENABLED
	HMODULE lib_dcomp = nullptr;
#endif

	IDXGIAdapter1 *create_adapter(uint32_t p_adapter_index) const;
	ID3D12DeviceFactory *device_factory_get() const;
	IDXGIFactory2 *dxgi_factory_get() const;
	bool get_tearing_supported() const;
	bool use_validation_layers() const;

	RenderingContextDriverD3D12();
	virtual ~RenderingContextDriverD3D12() override;
};
