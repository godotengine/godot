/**************************************************************************/
/*  d3d12_context.h                                                       */
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

#ifndef D3D12_CONTEXT_H
#define D3D12_CONTEXT_H

#include "core/error/error_list.h"
#include "core/os/mutex.h"
#include "core/string/ustring.h"
#include "core/templates/rid_owner.h"
#include "rendering_device_driver_d3d12.h"
#include "servers/display_server.h"
#include "servers/rendering/renderer_rd/api_context_rd.h"

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wswitch"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wimplicit-fallthrough"
#endif

#if defined(AS)
#undef AS
#endif

#include "d3dx12.h"
#include <dxgi1_6.h>

#include <wrl/client.h>

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

using Microsoft::WRL::ComPtr;

#define ARRAY_SIZE(a) (sizeof(a) / sizeof(a[0]))

class D3D12Context : public ApiContextRD {
public:
	struct DeviceLimits {
		uint64_t max_srvs_per_shader_stage;
		uint64_t max_cbvs_per_shader_stage;
		uint64_t max_samplers_across_all_stages;
		uint64_t max_uavs_across_all_stages;
		uint64_t timestamp_frequency;
	};

	struct SubgroupCapabilities {
		uint32_t size;
		bool wave_ops_supported;
		uint32_t supported_stages_flags_rd() const;
		uint32_t supported_operations_flags_rd() const;
	};

	struct VRSCapabilities {
		bool draw_call_supported; // We can specify our fragment rate on a draw call level.
		bool primitive_supported; // We can specify our fragment rate on each drawcall.
		bool primitive_in_multiviewport;
		bool ss_image_supported; // We can provide a density map attachment on our framebuffer.
		uint32_t ss_image_tile_size;
		bool additional_rates_supported;
	};

	struct ShaderCapabilities {
		D3D_SHADER_MODEL shader_model;
		bool native_16bit_ops;
	};

	struct StorageBufferCapabilities {
		bool storage_buffer_16_bit_access_is_supported;
	};

	struct FormatCapabilities {
		bool relaxed_casting_supported;
	};

private:
	enum {
		FRAME_LAG = 2,
		IMAGE_COUNT = FRAME_LAG + 1,
	};

	ComPtr<ID3D12DeviceFactory> device_factory;
	ComPtr<IDXGIFactory2> dxgi_factory;
	ComPtr<IDXGIAdapter> gpu;
	DeviceLimits gpu_limits = {};
	struct DeviceBasics {
		ComPtr<ID3D12Device> device;
		ComPtr<ID3D12CommandQueue> queue;
		ComPtr<ID3D12Fence> fence;
		HANDLE fence_event = nullptr;
		UINT64 fence_value = 0;
		RenderingDeviceDriverD3D12 *driver = nullptr;
	} md; // 'Main device', as opposed to local device.

	uint32_t feature_level = 0; // Major * 10 + minor.
	bool tearing_supported = false;
	SubgroupCapabilities subgroup_capabilities;
	RDD::MultiviewCapabilities multiview_capabilities;
	VRSCapabilities vrs_capabilities;
	ShaderCapabilities shader_capabilities;
	StorageBufferCapabilities storage_buffer_capabilities;
	FormatCapabilities format_capabilities;

	String adapter_vendor;
	String adapter_name;
	RenderingDevice::DeviceType adapter_type = {};
	String pipeline_cache_id;

	bool buffers_prepared = false;

	DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
	uint32_t frame = 0;
	ComPtr<ID3D12Fence> frame_fence;
	HANDLE frame_fence_event = nullptr;

	struct Window {
		HWND hwnd = nullptr;
		ComPtr<IDXGISwapChain3> swapchain;
		UINT swapchain_flags = 0;
		UINT sync_interval = 1;
		UINT present_flags = 0;
		ComPtr<ID3D12Resource> render_targets[IMAGE_COUNT];
		uint32_t current_buffer = 0;
		int width = 0;
		int height = 0;
		DisplayServer::VSyncMode vsync_mode = DisplayServer::VSYNC_ENABLED;
		RenderingDeviceDriverD3D12::RenderPassInfo render_pass;
		RenderingDeviceDriverD3D12::FramebufferInfo framebuffers[IMAGE_COUNT];
	};

	struct LocalDevice : public DeviceBasics {
		bool waiting = false;
		HANDLE fence_event = nullptr;
		UINT64 fence_value = 0;
	};

	RID_Owner<LocalDevice, true> local_device_owner;

	HashMap<DisplayServer::WindowID, Window> windows;

	// Commands.

	LocalVector<ID3D12CommandList *> command_list_queue;
	uint32_t command_list_count = 1;

	static void STDMETHODCALLTYPE _debug_message_func(
			D3D12_MESSAGE_CATEGORY p_category,
			D3D12_MESSAGE_SEVERITY p_severity,
			D3D12_MESSAGE_ID p_id,
			LPCSTR p_description,
			void *p_context);

	Error _initialize_debug_layers();
	void _init_device_factory();

	Error _select_adapter(int &r_index);
	void _dump_adapter_info(int p_index);
	Error _create_device(DeviceBasics &r_basics);
	Error _get_device_limits();
	Error _check_capabilities();

	Error _update_swap_chain(Window *window);

	void _wait_for_idle_queue(ID3D12CommandQueue *p_queue);

protected:
	virtual bool _use_validation_layers();

public:
	virtual const char *get_api_name() const override final { return "D3D12"; };
	virtual RenderingDevice::Capabilities get_device_capabilities() const override final;
	const SubgroupCapabilities &get_subgroup_capabilities() const { return subgroup_capabilities; };
	virtual const RDD::MultiviewCapabilities &get_multiview_capabilities() const override final { return multiview_capabilities; };
	const VRSCapabilities &get_vrs_capabilities() const { return vrs_capabilities; };
	const ShaderCapabilities &get_shader_capabilities() const { return shader_capabilities; };
	const StorageBufferCapabilities &get_storage_buffer_capabilities() const { return storage_buffer_capabilities; };
	const FormatCapabilities &get_format_capabilities() const { return format_capabilities; };

	ID3D12Device *get_device();
	IDXGIAdapter *get_adapter();
	virtual int get_swapchain_image_count() const override final;

	struct WindowPlatformData {
		HWND window;
	};
	virtual Error window_create(DisplayServer::WindowID p_window_id, DisplayServer::VSyncMode p_vsync_mode, int p_width, int p_height, const void *p_platform_data) override final;
	virtual void window_resize(DisplayServer::WindowID p_window_id, int p_width, int p_height) override final;
	virtual int window_get_width(DisplayServer::WindowID p_window = 0) override final;
	virtual int window_get_height(DisplayServer::WindowID p_window = 0) override final;
	virtual bool window_is_valid_swapchain(DisplayServer::WindowID p_window = 0) override final;
	virtual void window_destroy(DisplayServer::WindowID p_window_id) override final;
	virtual RDD::RenderPassID window_get_render_pass(DisplayServer::WindowID p_window = 0) override final;
	virtual RDD::FramebufferID window_get_framebuffer(DisplayServer::WindowID p_window = 0) override final;

	virtual RID local_device_create() override final;
	virtual void local_device_push_command_buffers(RID p_local_device, const RDD::CommandBufferID *p_buffers, int p_count) override final;
	virtual void local_device_sync(RID p_local_device) override final;
	virtual void local_device_free(RID p_local_device) override final;

	DXGI_FORMAT get_screen_format() const;
	const DeviceLimits &get_device_limits() const;

	virtual void set_setup_buffer(RDD::CommandBufferID p_command_buffer) override final;
	virtual void append_command_buffer(RDD::CommandBufferID p_command_buffer) override final;
	void resize_notify();
	virtual void flush(bool p_flush_setup = false, bool p_flush_pending = false, bool p_sync = true) override final;
	virtual Error prepare_buffers(RDD::CommandBufferID p_command_buffer) override final;
	virtual void postpare_buffers(RDD::CommandBufferID p_command_buffer) override final;
	virtual Error swap_buffers() override final;
	virtual Error initialize() override final;

	void set_object_name(ID3D12Object *p_object, String p_object_name);

	virtual String get_device_vendor_name() const override final;
	virtual String get_device_name() const override final;
	virtual RDD::DeviceType get_device_type() const override final;
	virtual String get_device_api_version() const override final;
	virtual String get_device_pipeline_cache_uuid() const override final;

	virtual void set_vsync_mode(DisplayServer::WindowID p_window, DisplayServer::VSyncMode p_mode) override final;
	virtual DisplayServer::VSyncMode get_vsync_mode(DisplayServer::WindowID p_window = 0) const override final;

	virtual RenderingDeviceDriver *get_driver(RID p_local_device = RID()) override final;
	virtual bool is_debug_utils_enabled() const override final;

	D3D12Context();
	virtual ~D3D12Context();
};

#endif // D3D12_CONTEXT_H
