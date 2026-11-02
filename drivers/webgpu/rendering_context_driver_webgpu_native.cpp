/**************************************************************************/
/*  rendering_context_driver_webgpu_native.cpp                            */
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

#if defined(WEBGPU_ENABLED) && !defined(WEB_ENABLED)

#include "core/config/engine.h"
#include "core/config/project_settings.h"
#include "core/crypto/crypto_core.h"
#include "core/io/dir_access.h"
#include "core/io/file_access.h"
#include "core/os/os.h"
#include "core/os/thread.h"
#include "drivers/webgpu/native/dawn_boundary.h"
#include "drivers/webgpu/native/dawn_loader.gen.h"
#include "drivers/webgpu/rendering_context_driver_webgpu.h"

namespace {
String webgpu_message(WGPUStringView p_message) {
	return p_message.data ? String::utf8(p_message.data, p_message.length == WGPU_STRLEN ? -1 : int(p_message.length)) : String();
}

String dawn_cache_path(const String &p_dir, const uint8_t *p_key, size_t p_key_size) {
	unsigned char hash[32];
	CryptoCore::sha256(p_key, p_key_size, hash);
	return p_dir.path_join(String::hex_encode_buffer(hash, 32));
}

// Dawn asks for an entry's size with a null value first, then loads it with exactly that size.
size_t dawn_cache_load(size_t p_key_size, const uint8_t *p_key, size_t p_value_size, uint8_t *p_value, void *p_dir, void *) {
	Ref<FileAccess> file = FileAccess::open(dawn_cache_path(*static_cast<const String *>(p_dir), p_key, p_key_size), FileAccess::READ);
	if (file.is_null()) {
		return 0;
	}
	const size_t size = file->get_length();
	if (p_value == nullptr) {
		return size;
	}
	return size == p_value_size ? file->get_buffer(p_value, size) : 0;
}

void dawn_cache_store(size_t p_key_size, const uint8_t *p_key, size_t p_value_size, const uint8_t *p_value, void *p_dir, void *) {
	const String path = dawn_cache_path(*static_cast<const String *>(p_dir), p_key, p_key_size);
	// Store beside the entry, then rename, so a concurrent load never sees a partial entry.
	const String temp_path = path + "." + itos(Thread::get_caller_id()) + ".tmp";
	{
		Ref<FileAccess> file = FileAccess::open(temp_path, FileAccess::WRITE);
		if (file.is_null() || !file->store_buffer(p_value, p_value_size)) {
			return;
		}
	}
	DirAccess::create(DirAccess::ACCESS_FILESYSTEM)->rename(temp_path, path);
}
} // namespace

Error RenderingContextDriverWebGPU::initialize_native() {
	ERR_FAIL_COND_V_MSG(!load_native_dawn(), ERR_UNAVAILABLE, "WebGPU could not load the native Dawn runtime beside libgodot. Check its runtime dependencies (macOS 13+, or the Visual C++ runtime on Windows).");
	WGPUInstanceFeatureName timed_wait = WGPUInstanceFeatureName_TimedWaitAny;
	WGPUInstanceDescriptor instance_desc = WGPU_INSTANCE_DESCRIPTOR_INIT;
	instance_desc.requiredFeatureCount = 1;
	instance_desc.requiredFeatures = &timed_wait;
	WGPULoggingCallbackInfo logging = WGPU_LOGGING_CALLBACK_INFO_INIT;
	logging.callback = [](WGPULoggingType p_type, WGPUStringView p_message, void *, void *) {
		if (p_type == WGPULoggingType_Error) {
			ERR_PRINT("Dawn: " + webgpu_message(p_message));
		} else if (p_type == WGPULoggingType_Warning) {
			WARN_PRINT("Dawn: " + webgpu_message(p_message));
		} else {
			print_verbose("Dawn: " + webgpu_message(p_message));
		}
	};
	instance = godot_webgpu_create_instance(&instance_desc, &logging);
	ERR_FAIL_NULL_V_MSG(instance, ERR_CANT_CREATE, "WebGPU: native Dawn instance creation failed.");

	WGPURequestAdapterOptions options = WGPU_REQUEST_ADAPTER_OPTIONS_INIT;
	options.powerPreference = OS::get_singleton()->get_user_prefers_integrated_gpu() ? WGPUPowerPreference_LowPower : WGPUPowerPreference_HighPerformance;
	String backend = GLOBAL_GET("rendering/rendering_device/webgpu/native_backend");
	if (backend == "auto") {
#ifdef WINDOWS_ENABLED
		backend = "d3d12";
#elif defined(MACOS_ENABLED)
		backend = "metal";
#else
		backend = "vulkan";
#endif
	}
	if (backend == "vulkan") {
		options.backendType = WGPUBackendType_Vulkan;
	} else if (backend == "d3d12") {
		options.backendType = WGPUBackendType_D3D12;
	} else if (backend == "d3d11") {
		options.backendType = WGPUBackendType_D3D11;
	} else if (backend == "metal") {
		options.backendType = WGPUBackendType_Metal;
	}
	struct AdapterResult {
		WGPUAdapter handle = nullptr;
		String error;
	} adapter_result;
	WGPURequestAdapterCallbackInfo adapter_callback = WGPU_REQUEST_ADAPTER_CALLBACK_INFO_INIT;
	adapter_callback.mode = WGPUCallbackMode_WaitAnyOnly;
	adapter_callback.userdata1 = &adapter_result;
	adapter_callback.callback = [](WGPURequestAdapterStatus p_status, WGPUAdapter p_adapter, WGPUStringView p_message, void *p_data, void *) {
		auto &result = *static_cast<AdapterResult *>(p_data);
		if (p_status == WGPURequestAdapterStatus_Success) {
			result.handle = p_adapter;
		} else {
			result.error = webgpu_message(p_message);
		}
	};
	// Dawn picks one adapter by power preference or LUID, so this context reports a single device.
	if (Engine::get_singleton()->get_gpu_index() >= 0) {
		WARN_PRINT("WebGPU: --gpu-index is ignored; native Dawn selects one adapter automatically or by --gpu-luid on Windows.");
	}
	uint64_t requested_luid = Engine::get_singleton()->get_gpu_index() < 0 ? Engine::get_singleton()->get_gpu_luid() : 0;
	WGPUFutureWaitInfo adapter_wait = { godot_webgpu_request_adapter(instance, &options, requested_luid, adapter_callback), false };
	ERR_FAIL_COND_V(wgpuInstanceWaitAny(instance, 1, &adapter_wait, UINT64_MAX) != WGPUWaitStatus_Success, ERR_CANT_CREATE);
	if (!adapter_result.handle && requested_luid != 0) {
		// RenderingDevice reports the same fallback warning as the other drivers.
		adapter_wait = { godot_webgpu_request_adapter(instance, &options, 0, adapter_callback), false };
		ERR_FAIL_COND_V(wgpuInstanceWaitAny(instance, 1, &adapter_wait, UINT64_MAX) != WGPUWaitStatus_Success, ERR_CANT_CREATE);
	}
	adapter = adapter_result.handle;
	ERR_FAIL_NULL_V_MSG(adapter, ERR_CANT_CREATE, "WebGPU: native adapter request failed: " + adapter_result.error);

	WGPULimits limits = WGPU_LIMITS_INIT;
	ERR_FAIL_COND_V(wgpuAdapterGetLimits(adapter, &limits) != WGPUStatus_Success, ERR_CANT_CREATE);
	// Test hook mirroring the browser smoke's ?limited-bindings: cap the sampled textures per stage to rehearse
	// another browser's adapter, such as Safari's 44.
	const String sampled_texture_limit = OS::get_singleton()->get_environment("GODOT_WEBGPU_SAMPLED_TEXTURE_LIMIT");
	if (sampled_texture_limit.is_valid_int()) {
		limits.maxSampledTexturesPerShaderStage = MIN(limits.maxSampledTexturesPerShaderStage, (uint32_t)sampled_texture_limit.to_int());
	}
	Vector<WGPUFeatureName> features;
	// Keep in sync with requestWebGPUDevice in platform/web/js/engine/engine.js (the last feature is Dawn-only).
	const WGPUFeatureName optional_features[] = {
		WGPUFeatureName_Depth32FloatStencil8,
		WGPUFeatureName_Float32Filterable,
		WGPUFeatureName_Float32Blendable,
		WGPUFeatureName_TextureFormatsTier1,
		WGPUFeatureName_TextureFormatsTier2,
		WGPUFeatureName_RG11B10UfloatRenderable,
		WGPUFeatureName_ClipDistances,
		WGPUFeatureName_DualSourceBlending,
		WGPUFeatureName_IndirectFirstInstance,
		WGPUFeatureName_TextureCompressionBC,
		WGPUFeatureName_TextureCompressionETC2,
		WGPUFeatureName_TextureCompressionASTC,
		WGPUFeatureName_ImplicitDeviceSynchronization,
	};
	for (WGPUFeatureName feature : optional_features) {
		if (wgpuAdapterHasFeature(adapter, feature)) {
			features.push_back(feature);
		}
	}
	WGPUDeviceDescriptor device_desc = WGPU_DEVICE_DESCRIPTOR_INIT;
	device_desc.requiredLimits = &limits;
	device_desc.requiredFeatureCount = features.size();
	device_desc.requiredFeatures = features.ptr();
	device_desc.uncapturedErrorCallbackInfo.callback = [](WGPUDevice const *, WGPUErrorType, WGPUStringView p_message, void *, void *) {
		ERR_PRINT("WebGPU: " + webgpu_message(p_message));
	};
	device_desc.deviceLostCallbackInfo.mode = WGPUCallbackMode_AllowProcessEvents;
	device_desc.deviceLostCallbackInfo.callback = [](WGPUDevice const *, WGPUDeviceLostReason p_reason, WGPUStringView p_message, void *, void *) {
		if (p_reason != WGPUDeviceLostReason_Destroyed && p_reason != WGPUDeviceLostReason_CallbackCancelled) {
			ERR_PRINT("WebGPU device lost: " + webgpu_message(p_message));
		}
	};
	// Like Vulkan's pipeline cache: Dawn otherwise recompiles every backend shader and pipeline on each run.
	WGPUDawnCacheDeviceDescriptor cache_desc = WGPU_DAWN_CACHE_DEVICE_DESCRIPTOR_INIT;
	if (GLOBAL_GET("rendering/rendering_device/pipeline_cache/enable")) {
		dawn_cache_dir = ProjectSettings::get_singleton()->globalize_path("user://webgpu/dawn_cache_" + backend);
		if (DirAccess::make_dir_recursive_absolute(dawn_cache_dir) == OK) {
			cache_desc.isolationKey = { "godot", WGPU_STRLEN };
			cache_desc.dawnLoadCacheDataCallbackInfo.callback = dawn_cache_load;
			cache_desc.dawnLoadCacheDataCallbackInfo.userdata1 = &dawn_cache_dir;
			cache_desc.dawnStoreCacheDataCallbackInfo.callback = dawn_cache_store;
			cache_desc.dawnStoreCacheDataCallbackInfo.userdata1 = &dawn_cache_dir;
			device_desc.nextInChain = &cache_desc.chain;
		}
	}
	struct DeviceResult {
		WGPUDevice handle = nullptr;
		String error;
	} device_result;
	WGPURequestDeviceCallbackInfo device_callback = WGPU_REQUEST_DEVICE_CALLBACK_INFO_INIT;
	device_callback.mode = WGPUCallbackMode_WaitAnyOnly;
	device_callback.userdata1 = &device_result;
	device_callback.callback = [](WGPURequestDeviceStatus p_status, WGPUDevice p_device, WGPUStringView p_message, void *p_data, void *) {
		auto &result = *static_cast<DeviceResult *>(p_data);
		if (p_status == WGPURequestDeviceStatus_Success) {
			result.handle = p_device;
		} else {
			result.error = webgpu_message(p_message);
		}
	};
	WGPUFutureWaitInfo device_wait = { wgpuAdapterRequestDevice(adapter, &device_desc, device_callback), false };
	ERR_FAIL_COND_V(wgpuInstanceWaitAny(instance, 1, &device_wait, UINT64_MAX) != WGPUWaitStatus_Success, ERR_CANT_CREATE);
	device = device_result.handle;
	ERR_FAIL_NULL_V_MSG(device, ERR_CANT_CREATE, "WebGPU: native device request failed: " + device_result.error);
	queue = wgpuDeviceGetQueue(device);
	ERR_FAIL_NULL_V(queue, ERR_CANT_CREATE);

	WGPUAdapterInfo info = WGPU_ADAPTER_INFO_INIT;
	WGPUAdapterPropertiesD3D d3d_info = WGPU_ADAPTER_PROPERTIES_D3D_INIT;
	if (wgpuAdapterHasFeature(adapter, WGPUFeatureName_AdapterPropertiesD3D)) {
		info.nextInChain = &d3d_info.chain;
	}
	if (wgpuAdapterGetInfo(adapter, &info) == WGPUStatus_Success) {
		device_info.name = webgpu_message(info.device);
		device_info.vendor = info.vendorID;
		device_info.type = info.adapterType == WGPUAdapterType_DiscreteGPU ? DEVICE_TYPE_DISCRETE_GPU : (info.adapterType == WGPUAdapterType_IntegratedGPU ? DEVICE_TYPE_INTEGRATED_GPU : (info.adapterType == WGPUAdapterType_CPU ? DEVICE_TYPE_CPU : DEVICE_TYPE_OTHER));
		if (info.nextInChain) {
			device_info.luid = (uint64_t(d3d_info.adapterLUIDHighPart) << 32) | d3d_info.adapterLUIDLowPart;
			device_info.luid_valid = true;
		}
		wgpuAdapterInfoFreeMembers(info);
	}
	return OK;
}

RenderingContextDriver::SurfaceID RenderingContextDriverWebGPU::surface_create_native(const void *p_platform_data) {
	ERR_FAIL_NULL_V(p_platform_data, SurfaceID());
	const auto &platform = *static_cast<const WindowPlatformData *>(p_platform_data);
	WGPUSurfaceSourceWindowsHWND win32 = WGPU_SURFACE_SOURCE_WINDOWS_HWND_INIT;
	WGPUSurfaceSourceXlibWindow xlib = WGPU_SURFACE_SOURCE_XLIB_WINDOW_INIT;
	WGPUSurfaceSourceWaylandSurface wayland = WGPU_SURFACE_SOURCE_WAYLAND_SURFACE_INIT;
	WGPUSurfaceSourceMetalLayer metal = WGPU_SURFACE_SOURCE_METAL_LAYER_INIT;
	WGPUSurfaceDescriptor descriptor = WGPU_SURFACE_DESCRIPTOR_INIT;
	switch (platform.kind) {
		case WindowPlatformData::KIND_WIN32:
			win32.hinstance = platform.display;
			win32.hwnd = platform.window;
			descriptor.nextInChain = &win32.chain;
			break;
		case WindowPlatformData::KIND_XLIB:
			xlib.display = platform.display;
			xlib.window = platform.window_id;
			descriptor.nextInChain = &xlib.chain;
			break;
		case WindowPlatformData::KIND_WAYLAND:
			wayland.display = platform.display;
			wayland.surface = platform.window;
			descriptor.nextInChain = &wayland.chain;
			break;
		case WindowPlatformData::KIND_METAL_LAYER:
			metal.layer = platform.window;
			descriptor.nextInChain = &metal.chain;
			break;
	}
	WGPUSurface handle = wgpuInstanceCreateSurface(instance, &descriptor);
	ERR_FAIL_NULL_V_MSG(handle, SurfaceID(), "WebGPU: native surface creation failed.");
	WGPUSurfaceCapabilities capabilities = WGPU_SURFACE_CAPABILITIES_INIT;
	if (wgpuSurfaceGetCapabilities(handle, adapter, &capabilities) != WGPUStatus_Success || capabilities.formatCount == 0) {
		wgpuSurfaceCapabilitiesFreeMembers(capabilities);
		wgpuSurfaceRelease(handle);
		ERR_FAIL_V_MSG(SurfaceID(), "WebGPU: selected adapter cannot present to this surface.");
	}
	Surface surface;
	surface.handle = handle;
	surface.format = capabilities.formats[0];
	for (size_t i = 0; i < capabilities.formatCount; i++) {
		if (capabilities.formats[i] == WGPUTextureFormat_BGRA8Unorm || capabilities.formats[i] == WGPUTextureFormat_RGBA8Unorm) {
			surface.format = capabilities.formats[i];
			break;
		}
	}
	if (capabilities.alphaModeCount > 0) {
		surface.alpha_mode = capabilities.alphaModes[0];
	}
	for (size_t i = 0; i < capabilities.presentModeCount; i++) {
		surface.present_modes.push_back(capabilities.presentModes[i]);
	}
	wgpuSurfaceCapabilitiesFreeMembers(capabilities);
	SurfaceID id = next_surface_id++;
	surfaces[id] = surface;
	return id;
}

#endif
