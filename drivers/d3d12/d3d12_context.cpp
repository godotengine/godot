/**************************************************************************/
/*  d3d12_context.cpp                                                     */
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

#include "d3d12_context.h"

#include "core/config/engine.h"
#include "core/config/project_settings.h"
#include "core/string/ustring.h"
#include "core/templates/local_vector.h"
#include "core/version.h"
#include "servers/rendering/rendering_device.h"

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wswitch"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#endif

#include "dxcapi.h"

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

#if !defined(_MSC_VER)
#include <guiddef.h>

#include <dxguids.h>
#endif

// Note: symbol is not available in MinGW and old MSVC import libraries.
const CLSID CLSID_D3D12DeviceFactoryGodot = __uuidof(ID3D12DeviceFactory);
const CLSID CLSID_D3D12DebugGodot = __uuidof(ID3D12Debug);
const CLSID CLSID_D3D12SDKConfigurationGodot = __uuidof(ID3D12SDKConfiguration);

extern "C" {
char godot_nir_arch_name[32];
}

#ifdef PIX_ENABLED
#if defined(__GNUC__)
#define _MSC_VER 1800
#endif
#define USE_PIX
#include "WinPixEventRuntime/pix3.h"
#if defined(__GNUC__)
#undef _MSC_VER
#endif
#endif

#define D3D12_DEBUG_LAYER_BREAK_ON_ERROR 0

void D3D12Context::_debug_message_func(
		D3D12_MESSAGE_CATEGORY p_category,
		D3D12_MESSAGE_SEVERITY p_severity,
		D3D12_MESSAGE_ID p_id,
		LPCSTR p_description,
		void *p_context) {
	String type_string;
	switch (p_category) {
		case D3D12_MESSAGE_CATEGORY_APPLICATION_DEFINED:
			type_string = "APPLICATION_DEFINED";
			break;
		case D3D12_MESSAGE_CATEGORY_MISCELLANEOUS:
			type_string = "MISCELLANEOUS";
			break;
		case D3D12_MESSAGE_CATEGORY_INITIALIZATION:
			type_string = "INITIALIZATION";
			break;
		case D3D12_MESSAGE_CATEGORY_CLEANUP:
			type_string = "CLEANUP";
			break;
		case D3D12_MESSAGE_CATEGORY_COMPILATION:
			type_string = "COMPILATION";
			break;
		case D3D12_MESSAGE_CATEGORY_STATE_CREATION:
			type_string = "STATE_CREATION";
			break;
		case D3D12_MESSAGE_CATEGORY_STATE_SETTING:
			type_string = "STATE_SETTING";
			break;
		case D3D12_MESSAGE_CATEGORY_STATE_GETTING:
			type_string = "STATE_GETTING";
			break;
		case D3D12_MESSAGE_CATEGORY_RESOURCE_MANIPULATION:
			type_string = "RESOURCE_MANIPULATION";
			break;
		case D3D12_MESSAGE_CATEGORY_EXECUTION:
			type_string = "EXECUTION";
			break;
		case D3D12_MESSAGE_CATEGORY_SHADER:
			type_string = "SHADER";
			break;
	}

	String error_message(type_string +
			" - Message Id Number: " + String::num_int64(p_id) +
			"\n\t" + p_description);

	// Convert D3D12 severity to our own log macros.
	switch (p_severity) {
		case D3D12_MESSAGE_SEVERITY_MESSAGE:
			print_verbose(error_message);
			break;
		case D3D12_MESSAGE_SEVERITY_INFO:
			print_line(error_message);
			break;
		case D3D12_MESSAGE_SEVERITY_WARNING:
			WARN_PRINT(error_message);
			break;
		case D3D12_MESSAGE_SEVERITY_ERROR:
		case D3D12_MESSAGE_SEVERITY_CORRUPTION:
			ERR_PRINT(error_message);
			CRASH_COND_MSG(Engine::get_singleton()->is_abort_on_gpu_errors_enabled(),
					"Crashing, because abort on GPU errors is enabled.");
			break;
	}
}

uint32_t D3D12Context::SubgroupCapabilities::supported_stages_flags_rd() const {
	// If there's a way to check exactly which are supported, I have yet to find it.
	return (
			RenderingDevice::ShaderStage::SHADER_STAGE_FRAGMENT_BIT |
			RenderingDevice::ShaderStage::SHADER_STAGE_COMPUTE_BIT);
}

uint32_t D3D12Context::SubgroupCapabilities::supported_operations_flags_rd() const {
	if (!wave_ops_supported) {
		return 0;
	} else {
		return (
				RenderingDevice::SubgroupOperations::SUBGROUP_BASIC_BIT |
				RenderingDevice::SubgroupOperations::SUBGROUP_BALLOT_BIT |
				RenderingDevice::SubgroupOperations::SUBGROUP_VOTE_BIT |
				RenderingDevice::SubgroupOperations::SUBGROUP_SHUFFLE_BIT |
				RenderingDevice::SubgroupOperations::SUBGROUP_SHUFFLE_RELATIVE_BIT |
				RenderingDevice::SubgroupOperations::SUBGROUP_QUAD_BIT |
				RenderingDevice::SubgroupOperations::SUBGROUP_ARITHMETIC_BIT |
				RenderingDevice::SubgroupOperations::SUBGROUP_CLUSTERED_BIT);
	}
}

Error D3D12Context::_check_capabilities() {
	// Assume not supported until proven otherwise.
	vrs_capabilities.draw_call_supported = false;
	vrs_capabilities.primitive_supported = false;
	vrs_capabilities.primitive_in_multiviewport = false;
	vrs_capabilities.ss_image_supported = false;
	vrs_capabilities.ss_image_tile_size = 1;
	vrs_capabilities.additional_rates_supported = false;
	multiview_capabilities.is_supported = false;
	multiview_capabilities.geometry_shader_is_supported = false;
	multiview_capabilities.tessellation_shader_is_supported = false;
	multiview_capabilities.max_view_count = 0;
	multiview_capabilities.max_instance_count = 0;
	multiview_capabilities.is_supported = false;
	subgroup_capabilities.size = 0;
	subgroup_capabilities.wave_ops_supported = false;
	shader_capabilities.shader_model = D3D_SHADER_MODEL_6_0;
	shader_capabilities.native_16bit_ops = false;
	storage_buffer_capabilities.storage_buffer_16_bit_access_is_supported = false;
	format_capabilities.relaxed_casting_supported = false;

	{
		D3D12_FEATURE_DATA_SHADER_MODEL shader_model = {};
		shader_model.HighestShaderModel = MIN(D3D_HIGHEST_SHADER_MODEL, D3D_SHADER_MODEL_6_6);
		HRESULT res = md.device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &shader_model, sizeof(shader_model));
		ERR_FAIL_COND_V_MSG(!SUCCEEDED(res), ERR_CANT_CREATE, "CheckFeatureSupport failed with error " + vformat("0x%08ux", (uint64_t)res) + ".");
		shader_capabilities.shader_model = shader_model.HighestShaderModel;
	}
	print_verbose("- Shader:");
	print_verbose("  model: " + itos(shader_capabilities.shader_model >> 4) + "." + itos(shader_capabilities.shader_model & 0xf));

	D3D12_FEATURE_DATA_D3D12_OPTIONS options = {};
	HRESULT res = md.device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &options, sizeof(options));
	if (SUCCEEDED(res)) {
		storage_buffer_capabilities.storage_buffer_16_bit_access_is_supported = options.TypedUAVLoadAdditionalFormats;
	}

	D3D12_FEATURE_DATA_D3D12_OPTIONS1 options1 = {};
	res = md.device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, &options1, sizeof(options1));
	if (SUCCEEDED(res)) {
		subgroup_capabilities.size = options1.WaveLaneCountMin;
		subgroup_capabilities.wave_ops_supported = options1.WaveOps;
	}

	D3D12_FEATURE_DATA_D3D12_OPTIONS3 options3 = {};
	res = md.device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS3, &options3, sizeof(options3));
	if (SUCCEEDED(res)) {
		// https://docs.microsoft.com/en-us/windows/win32/api/d3d12/ne-d3d12-d3d12_view_instancing_tier
		// https://microsoft.github.io/DirectX-Specs/d3d/ViewInstancing.html#sv_viewid
		if (options3.ViewInstancingTier >= D3D12_VIEW_INSTANCING_TIER_1) {
			multiview_capabilities.is_supported = true;
			multiview_capabilities.geometry_shader_is_supported = options3.ViewInstancingTier >= D3D12_VIEW_INSTANCING_TIER_3;
			multiview_capabilities.tessellation_shader_is_supported = options3.ViewInstancingTier >= D3D12_VIEW_INSTANCING_TIER_3;
			multiview_capabilities.max_view_count = D3D12_MAX_VIEW_INSTANCE_COUNT;
			multiview_capabilities.max_instance_count = UINT32_MAX;
		}
	}

	D3D12_FEATURE_DATA_D3D12_OPTIONS4 options4 = {};
	res = md.device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS4, &options4, sizeof(options4));
	if (SUCCEEDED(res)) {
		shader_capabilities.native_16bit_ops = options4.Native16BitShaderOpsSupported;
	}
	print_verbose(String("  16-bit ops supported: ") + (shader_capabilities.native_16bit_ops ? "yes" : "no"));

	D3D12_FEATURE_DATA_D3D12_OPTIONS6 options6 = {};
	res = md.device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS6, &options6, sizeof(options6));
	if (SUCCEEDED(res)) {
		if (options6.VariableShadingRateTier >= D3D12_VARIABLE_SHADING_RATE_TIER_1) {
			vrs_capabilities.draw_call_supported = true;
			if (options6.VariableShadingRateTier >= D3D12_VARIABLE_SHADING_RATE_TIER_2) {
				vrs_capabilities.primitive_supported = true;
				vrs_capabilities.primitive_in_multiviewport = options6.PerPrimitiveShadingRateSupportedWithViewportIndexing;
				vrs_capabilities.ss_image_supported = true;
				vrs_capabilities.ss_image_tile_size = options6.ShadingRateImageTileSize;
				vrs_capabilities.additional_rates_supported = options6.AdditionalShadingRatesSupported;
			}
		}
	}

	D3D12_FEATURE_DATA_D3D12_OPTIONS12 options12 = {};
	res = md.device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS12, &options12, sizeof(options12));
	if (SUCCEEDED(res)) {
		format_capabilities.relaxed_casting_supported = options12.RelaxedFormatCastingSupported;
	}

	if (vrs_capabilities.draw_call_supported || vrs_capabilities.primitive_supported || vrs_capabilities.ss_image_supported) {
		print_verbose("- D3D12 Variable Rate Shading supported:");
		if (vrs_capabilities.draw_call_supported) {
			print_verbose("  Draw call");
		}
		if (vrs_capabilities.primitive_supported) {
			print_verbose(String("  Per-primitive (multi-viewport: ") + (vrs_capabilities.primitive_in_multiviewport ? "yes" : "no") + ")");
		}
		if (vrs_capabilities.ss_image_supported) {
			print_verbose(String("  Screen-space image (tile size: ") + itos(vrs_capabilities.ss_image_tile_size) + ")");
		}
		if (vrs_capabilities.additional_rates_supported) {
			print_verbose(String("  Additional rates: ") + (vrs_capabilities.additional_rates_supported ? "yes" : "no"));
		}
	} else {
		print_verbose("- D3D12 Variable Rate Shading not supported");
	}

	if (multiview_capabilities.is_supported) {
		print_verbose("- D3D12 multiview supported:");
		print_verbose("  max view count: " + itos(multiview_capabilities.max_view_count));
		//print_verbose("  max instances: " + itos(multiview_capabilities.max_instance_count)); // Hardcoded; not very useful at the moment.
	} else {
		print_verbose("- D3D12 multiview not supported");
	}

	if (format_capabilities.relaxed_casting_supported) {
		print_verbose("- Relaxed casting supported");
	} else {
		print_verbose("- Relaxed casting not supported");
	}

	return OK;
}

Error D3D12Context::_initialize_debug_layers() {
	ComPtr<ID3D12Debug> debug_controller;
	HRESULT res;
	if (device_factory) {
		res = device_factory->GetConfigurationInterface(CLSID_D3D12DebugGodot, IID_PPV_ARGS(&debug_controller));
	} else {
		res = D3D12GetDebugInterface(IID_PPV_ARGS(&debug_controller));
	}
	ERR_FAIL_COND_V(!SUCCEEDED(res), ERR_QUERY_FAILED);
	debug_controller->EnableDebugLayer();
	return OK;
}

Error D3D12Context::_select_adapter(int &r_index) {
	{
		UINT flags = _use_validation_layers() ? DXGI_CREATE_FACTORY_DEBUG : 0;
		HRESULT res = CreateDXGIFactory2(flags, IID_PPV_ARGS(&dxgi_factory));
		ERR_FAIL_COND_V(!SUCCEEDED(res), ERR_CANT_CREATE);
	}

	ComPtr<IDXGIFactory6> factory6;
	dxgi_factory.As(&factory6);

	// TODO: Use IDXCoreAdapterList, which gives more comprehensive information.
	LocalVector<IDXGIAdapter1 *> adapters;
	while (true) {
		IDXGIAdapter1 *curr_adapter = nullptr;
		if (factory6) {
			if (factory6->EnumAdapterByGpuPreference(adapters.size(), DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&curr_adapter)) == DXGI_ERROR_NOT_FOUND) {
				break;
			}
		} else {
			if (dxgi_factory->EnumAdapters1(adapters.size(), &curr_adapter) == DXGI_ERROR_NOT_FOUND) {
				break;
			}
		}
		adapters.push_back(curr_adapter);
	}

	ERR_FAIL_COND_V_MSG(adapters.size() == 0, ERR_CANT_CREATE, "Adapters enumeration reported zero accessible devices.");

	// The device should really be a preference, but for now choosing a discrete GPU over the
	// integrated one is better than the default.

	int32_t adapter_index = -1;
	int type_selected = -1;
	LocalVector<RenderingDevice::DeviceType> adapter_types;
	print_verbose("D3D12 devices:");
	for (uint32_t i = 0; i < adapters.size(); ++i) {
		DXGI_ADAPTER_DESC1 desc = {};
		adapters[i]->GetDesc1(&desc);

		String name = desc.Description;
		String dev_type;
		RenderingDevice::DeviceType type = {};
		if (((desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE))) {
			type = RenderingDevice::DEVICE_TYPE_CPU;
		} else {
			type = desc.DedicatedVideoMemory ? RenderingDevice::DEVICE_TYPE_DISCRETE_GPU : RenderingDevice::DEVICE_TYPE_INTEGRATED_GPU;
		}
		adapter_types.push_back(type);

		switch (type) {
			case RenderingDevice::DEVICE_TYPE_DISCRETE_GPU: {
				dev_type = "Discrete";
			} break;
			case RenderingDevice::DEVICE_TYPE_INTEGRATED_GPU: {
				dev_type = "Integrated";
			} break;
			case RenderingDevice::DEVICE_TYPE_VIRTUAL_GPU: {
				dev_type = "Virtual";
			} break;
			case RenderingDevice::DEVICE_TYPE_CPU: {
				dev_type = "CPU";
			} break;
			default: {
				dev_type = "Other";
			} break;
		}
		print_verbose("  #" + itos(i) + ": " + name + ", " + dev_type);

		switch (type) {
			case RenderingDevice::DEVICE_TYPE_DISCRETE_GPU: {
				if (type_selected < 4) {
					type_selected = 4;
					adapter_index = i;
				}
			} break;
			case RenderingDevice::DEVICE_TYPE_INTEGRATED_GPU: {
				if (type_selected < 3) {
					type_selected = 3;
					adapter_index = i;
				}
			} break;
			case RenderingDevice::DEVICE_TYPE_VIRTUAL_GPU: {
				if (type_selected < 2) {
					type_selected = 2;
					adapter_index = i;
				}
			} break;
			case RenderingDevice::DEVICE_TYPE_CPU: {
				if (type_selected < 1) {
					type_selected = 1;
					adapter_index = i;
				}
			} break;
			default: {
				if (type_selected < 0) {
					type_selected = 0;
					adapter_index = i;
				}
			} break;
		}
	}

	int32_t user_adapter_index = Engine::get_singleton()->get_gpu_index(); // Force user selected GPU.
	if (user_adapter_index >= 0 && user_adapter_index < (int32_t)adapters.size()) {
		adapter_index = user_adapter_index;
	}

	ERR_FAIL_COND_V_MSG(adapter_index == -1, ERR_CANT_CREATE, "None of D3D12 devices supports hardware rendering.");

	gpu = adapters[adapter_index];
	for (uint32_t i = 0; i < adapters.size(); ++i) {
		adapters[i]->Release();
	}

	adapter_type = adapter_types[adapter_index];

	ComPtr<IDXGIFactory5> factory5;
	dxgi_factory.As(&factory5);
	if (factory5) {
		BOOL result = FALSE; // sizeof(bool) != sizeof(BOOL), in general.
		HRESULT res = factory5->CheckFeatureSupport(DXGI_FEATURE_PRESENT_ALLOW_TEARING, &result, sizeof(result));
		if (SUCCEEDED(res)) {
			tearing_supported = result;
		} else {
			ERR_PRINT("CheckFeatureSupport failed with error " + vformat("0x%08ux", (uint64_t)res) + ".");
		}
	}

	r_index = adapter_index;

	return OK;
}

void D3D12Context::_dump_adapter_info(int p_index) {
	{
		const D3D_FEATURE_LEVEL FEATURE_LEVELS[] = {
			D3D_FEATURE_LEVEL_11_0,
			D3D_FEATURE_LEVEL_11_1,
			D3D_FEATURE_LEVEL_12_0,
			D3D_FEATURE_LEVEL_12_1,
			D3D_FEATURE_LEVEL_12_2,
		};

		D3D12_FEATURE_DATA_FEATURE_LEVELS feat_levels = {};
		feat_levels.NumFeatureLevels = ARRAY_SIZE(FEATURE_LEVELS);
		feat_levels.pFeatureLevelsRequested = FEATURE_LEVELS;

		HRESULT res = md.device->CheckFeatureSupport(D3D12_FEATURE_FEATURE_LEVELS, &feat_levels, sizeof(feat_levels));
		ERR_FAIL_COND_MSG(!SUCCEEDED(res), "CheckFeatureSupport failed with error " + vformat("0x%08ux", (uint64_t)res) + ".");

		// Example: D3D_FEATURE_LEVEL_12_1 = 0xc100.
		uint32_t feat_level_major = feat_levels.MaxSupportedFeatureLevel >> 12;
		uint32_t feat_level_minor = (feat_levels.MaxSupportedFeatureLevel >> 16) & 0xff;
		feature_level = feat_level_major * 10 + feat_level_minor;
	}

	String rendering_method;
	if (OS::get_singleton()->get_current_rendering_method() == "mobile") {
		rendering_method = "Forward Mobile";
	} else {
		rendering_method = "Forward+";
	}

	static const struct {
		uint32_t id;
		const char *name;
	} vendor_names[] = {
		{ 0x1002, "AMD" },
		{ 0x1010, "ImgTec" },
		{ 0x106B, "Apple" },
		{ 0x10DE, "NVIDIA" },
		{ 0x13B5, "ARM" },
		{ 0x1414, "Microsoft" },
		{ 0x5143, "Qualcomm" },
		{ 0x8086, "Intel" },
		{ 0, nullptr },
	};

	DXGI_ADAPTER_DESC gpu_desc = {};
	gpu->GetDesc(&gpu_desc);

	adapter_name = gpu_desc.Description;
	pipeline_cache_id = String::hex_encode_buffer((uint8_t *)&gpu_desc.AdapterLuid, sizeof(LUID));
	pipeline_cache_id += "-driver-" + itos(gpu_desc.Revision);
	{
		adapter_vendor = "Unknown";
		uint32_t vendor_idx = 0;
		while (vendor_names[vendor_idx].name != nullptr) {
			if (gpu_desc.VendorId == vendor_names[vendor_idx].id) {
				adapter_vendor = vendor_names[vendor_idx].name;
				break;
			}
			vendor_idx++;
		}
	}

	print_line(vformat("D3D12 feature level %s - %s - Using D3D12 Adapter #%d: %s", get_device_api_version(), rendering_method, p_index, adapter_name));
}

Error D3D12Context::_create_device(DeviceBasics &r_basics) {
	HRESULT res;
	if (device_factory) {
		res = device_factory->CreateDevice(gpu.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(r_basics.device.GetAddressOf()));
	} else {
		res = D3D12CreateDevice(gpu.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(r_basics.device.GetAddressOf()));
	}
	ERR_FAIL_COND_V_MSG(!SUCCEEDED(res), ERR_CANT_CREATE, "D3D12CreateDevice failed with error " + vformat("0x%08ux", (uint64_t)res) + ".");

	// Create direct command queue.
	D3D12_COMMAND_QUEUE_DESC queue_desc = {};
	queue_desc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
	queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
	res = r_basics.device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(r_basics.queue.GetAddressOf()));
	ERR_FAIL_COND_V(!SUCCEEDED(res), ERR_CANT_CREATE);

	// Create sync objects.
	res = r_basics.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(r_basics.fence.GetAddressOf()));
	ERR_FAIL_COND_V(!SUCCEEDED(res), ERR_CANT_CREATE);
	r_basics.fence_event = CreateEvent(nullptr, FALSE, FALSE, nullptr);
	ERR_FAIL_NULL_V(r_basics.fence_event, ERR_CANT_CREATE);

	if (_use_validation_layers()) {
		ComPtr<ID3D12InfoQueue> info_queue;
		res = r_basics.device.As(&info_queue);
		ERR_FAIL_COND_V(!SUCCEEDED(res), ERR_CANT_CREATE);

		ComPtr<ID3D12InfoQueue1> info_queue_1;
		r_basics.device.As(&info_queue_1);
		if (info_queue_1) {
			// Custom printing supported (added in Windows 10 Release Preview build 20236).

			info_queue_1->SetMuteDebugOutput(TRUE);

			res = info_queue_1->RegisterMessageCallback(&_debug_message_func, D3D12_MESSAGE_CALLBACK_IGNORE_FILTERS, nullptr, 0);
			ERR_FAIL_COND_V(!SUCCEEDED(res), ERR_CANT_CREATE);
		} else {
			// Rely on D3D12's own debug printing.

			if (Engine::get_singleton()->is_abort_on_gpu_errors_enabled()) {
				res = info_queue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_WARNING, TRUE);
				ERR_FAIL_COND_V(!SUCCEEDED(res), ERR_CANT_CREATE);
				res = info_queue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_ERROR, TRUE);
				ERR_FAIL_COND_V(!SUCCEEDED(res), ERR_CANT_CREATE);
				res = info_queue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_CORRUPTION, TRUE);
				ERR_FAIL_COND_V(!SUCCEEDED(res), ERR_CANT_CREATE);
			}
		}
		D3D12_MESSAGE_SEVERITY severities_to_mute[] = {
			D3D12_MESSAGE_SEVERITY_INFO,
		};

		D3D12_MESSAGE_ID messages_to_mute[] = {
			D3D12_MESSAGE_ID_CLEARRENDERTARGETVIEW_MISMATCHINGCLEARVALUE,
			D3D12_MESSAGE_ID_CLEARDEPTHSTENCILVIEW_MISMATCHINGCLEARVALUE,
			// These happen due to how D3D12MA manages buffers; seem bening.
			D3D12_MESSAGE_ID_HEAP_ADDRESS_RANGE_HAS_NO_RESOURCE,
			D3D12_MESSAGE_ID_HEAP_ADDRESS_RANGE_INTERSECTS_MULTIPLE_BUFFERS,
		};

		D3D12_INFO_QUEUE_FILTER filter = {};
		filter.DenyList.NumSeverities = ARRAY_SIZE(severities_to_mute);
		filter.DenyList.pSeverityList = severities_to_mute;
		filter.DenyList.NumIDs = ARRAY_SIZE(messages_to_mute);
		filter.DenyList.pIDList = messages_to_mute;

		res = info_queue->PushStorageFilter(&filter);
		ERR_FAIL_COND_V(!SUCCEEDED(res), ERR_CANT_CREATE);

#if D3D12_DEBUG_LAYER_BREAK_ON_ERROR
		res = info_queue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_ERROR, true);
		ERR_FAIL_COND_V(!SUCCEEDED(res), ERR_CANT_CREATE);
#endif
	}

	return OK;
}

Error D3D12Context::_get_device_limits() {
	D3D12_FEATURE_DATA_D3D12_OPTIONS options = {};
	HRESULT res = md.device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &options, sizeof(options));
	ERR_FAIL_COND_V_MSG(!SUCCEEDED(res), ERR_UNAVAILABLE, "CheckFeatureSupport failed with error " + vformat("0x%08ux", (uint64_t)res) + ".");

	// https://docs.microsoft.com/en-us/windows/win32/direct3d12/hardware-support
	gpu_limits.max_srvs_per_shader_stage = options.ResourceBindingTier == D3D12_RESOURCE_BINDING_TIER_1 ? 128 : UINT64_MAX;
	gpu_limits.max_cbvs_per_shader_stage = options.ResourceBindingTier <= D3D12_RESOURCE_BINDING_TIER_2 ? 14 : UINT64_MAX;
	gpu_limits.max_samplers_across_all_stages = options.ResourceBindingTier == D3D12_RESOURCE_BINDING_TIER_1 ? 16 : 2048;
	if (options.ResourceBindingTier == D3D12_RESOURCE_BINDING_TIER_1) {
		gpu_limits.max_uavs_across_all_stages = feature_level <= 110 ? 8 : 64;
	} else if (options.ResourceBindingTier == D3D12_RESOURCE_BINDING_TIER_2) {
		gpu_limits.max_uavs_across_all_stages = 64;
	} else {
		gpu_limits.max_uavs_across_all_stages = UINT64_MAX;
	}

	md.queue->GetTimestampFrequency(&gpu_limits.timestamp_frequency);

	return OK;
}

bool D3D12Context::_use_validation_layers() {
	return Engine::get_singleton()->is_validation_layers_enabled();
}

Error D3D12Context::window_create(DisplayServer::WindowID p_window_id, DisplayServer::VSyncMode p_vsync_mode, int p_width, int p_height, const void *p_platform_data) {
	ERR_FAIL_COND_V(windows.has(p_window_id), ERR_INVALID_PARAMETER);

	Window window;
	window.hwnd = ((const WindowPlatformData *)p_platform_data)->window;
	window.width = p_width;
	window.height = p_height;
	window.vsync_mode = p_vsync_mode;

	{
		RDD::Attachment attachment;
		attachment.samples = RD::TEXTURE_SAMPLES_1;
		attachment.load_op = RDD::ATTACHMENT_LOAD_OP_CLEAR;
		attachment.store_op = RDD::ATTACHMENT_STORE_OP_STORE;
		window.render_pass.attachments.push_back(attachment);

		RDD::Subpass subpass;
		{
			RDD::AttachmentReference color_ref;
			color_ref.attachment = 0;
			color_ref.aspect.set_flag(RDD::TEXTURE_ASPECT_COLOR_BIT);
			subpass.color_references.push_back(color_ref);
		}
		window.render_pass.subpasses.push_back(subpass);
	}

	for (uint32_t i = 0; i < IMAGE_COUNT; i++) {
		Error err = window.framebuffers[i].rtv_heap.allocate(md.device.Get(), D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, false);
		ERR_FAIL_COND_V(err != OK, ERR_CANT_CREATE);
		window.framebuffers[i].is_screen = true;
		window.framebuffers[i].attachments_handle_inds.push_back(0);
	}

	Error err = _update_swap_chain(&window);
	ERR_FAIL_COND_V(err != OK, ERR_CANT_CREATE);

	windows[p_window_id] = window;

	return OK;
}

void D3D12Context::window_resize(DisplayServer::WindowID p_window, int p_width, int p_height) {
	ERR_FAIL_COND(!windows.has(p_window));
	windows[p_window].width = p_width;
	windows[p_window].height = p_height;
	_update_swap_chain(&windows[p_window]);
}

int D3D12Context::window_get_width(DisplayServer::WindowID p_window) {
	ERR_FAIL_COND_V(!windows.has(p_window), -1);
	return windows[p_window].width;
}

int D3D12Context::window_get_height(DisplayServer::WindowID p_window) {
	ERR_FAIL_COND_V(!windows.has(p_window), -1);
	return windows[p_window].height;
}

bool D3D12Context::window_is_valid_swapchain(DisplayServer::WindowID p_window) {
	ERR_FAIL_COND_V(!windows.has(p_window), false);
	Window *w = &windows[p_window];
	return (bool)w->swapchain;
}

RDD::RenderPassID D3D12Context::window_get_render_pass(DisplayServer::WindowID p_window) {
	ERR_FAIL_COND_V(!windows.has(p_window), RDD::RenderPassID());
	Window *w = &windows[p_window];
	return RDD::RenderPassID(&w->render_pass);
}

RDD::FramebufferID D3D12Context::window_get_framebuffer(DisplayServer::WindowID p_window) {
	ERR_FAIL_COND_V(!windows.has(p_window), RDD::FramebufferID());
	ERR_FAIL_COND_V(!buffers_prepared, RDD::FramebufferID());
	Window *w = &windows[p_window];
	if (w->swapchain) {
		return RDD::FramebufferID(&w->framebuffers[w->current_buffer]);
	} else {
		return RDD::FramebufferID();
	}
}

void D3D12Context::window_destroy(DisplayServer::WindowID p_window_id) {
	ERR_FAIL_COND(!windows.has(p_window_id));
	_wait_for_idle_queue(md.queue.Get());
	windows.erase(p_window_id);
}

Error D3D12Context::_update_swap_chain(Window *window) {
	if (window->width == 0 || window->height == 0) {
		// Likely window minimized, no swapchain created.
		return ERR_SKIP;
	}

	DisplayServer::VSyncMode curr_vsync_mode = window->vsync_mode;
	bool vsync_mode_available = false;
	UINT swapchain_flags = 0;
	do {
		switch (window->vsync_mode) {
			case DisplayServer::VSYNC_MAILBOX: {
				window->sync_interval = 1;
				window->present_flags = DXGI_PRESENT_RESTART;
				swapchain_flags = 0;
				vsync_mode_available = true;
			} break;
			case DisplayServer::VSYNC_ADAPTIVE: {
				vsync_mode_available = false; // I don't know how to set this up.
			} break;
			case DisplayServer::VSYNC_ENABLED: {
				window->sync_interval = 1;
				window->present_flags = 0;
				swapchain_flags = 0;
				vsync_mode_available = true;
			} break;
			case DisplayServer::VSYNC_DISABLED: {
				window->sync_interval = 0;
				window->present_flags = tearing_supported ? DXGI_PRESENT_ALLOW_TEARING : 0;
				swapchain_flags = tearing_supported ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0;
				vsync_mode_available = true;
			} break;
		}

		// Set the windows swap effect if it is available, otherwise FLIP_DISCARD is used.
		if (vsync_mode_available) {
			if (window->vsync_mode != curr_vsync_mode || !window->swapchain) {
				window->vsync_mode = curr_vsync_mode;
				print_verbose("Using swapchain flags: " + itos(swapchain_flags) + ", sync interval: " + itos(window->sync_interval) + ", present flags: " + itos(window->present_flags));
			}
		} else {
			String present_mode_string;
			switch (window->vsync_mode) {
				case DisplayServer::VSYNC_MAILBOX:
					present_mode_string = "Mailbox";
					break;
				case DisplayServer::VSYNC_ADAPTIVE:
					present_mode_string = "Adaptive";
					break;
				case DisplayServer::VSYNC_ENABLED:
					present_mode_string = "Enabled";
					break;
				case DisplayServer::VSYNC_DISABLED:
					present_mode_string = "Disabled";
					break;
			}
			WARN_PRINT(vformat("The requested V-Sync mode %s is not available. Falling back to V-Sync mode Enabled.", present_mode_string));
			window->vsync_mode = DisplayServer::VSYNC_ENABLED; // Set to default.
		}
	} while (!vsync_mode_available);

	if (window->swapchain) {
		_wait_for_idle_queue(md.queue.Get());
		for (uint32_t i = 0; i < IMAGE_COUNT; i++) {
			window->render_targets[i].Reset();
		}

		// D3D12 docs: "IDXGISwapChain::ResizeBuffers can't be used to add or remove this flag."
		bool allow_tearing_flag_changed = (swapchain_flags & DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING) != (window->swapchain_flags & DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING);
		if (allow_tearing_flag_changed) {
			window->swapchain.Reset();
		}
	}

	if (!window->swapchain) {
		DXGI_SWAP_CHAIN_DESC1 swapchain_desc = {};
		swapchain_desc.BufferCount = IMAGE_COUNT;
		swapchain_desc.Width = 0;
		swapchain_desc.Height = 0;
		swapchain_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
		swapchain_desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
		swapchain_desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
		swapchain_desc.SampleDesc.Count = 1;
		swapchain_desc.Flags = swapchain_flags;
		swapchain_desc.Scaling = DXGI_SCALING_NONE;

		ComPtr<IDXGISwapChain1> swapchain;
		HRESULT res = dxgi_factory->CreateSwapChainForHwnd(md.queue.Get(), window->hwnd, &swapchain_desc, nullptr, nullptr, swapchain.GetAddressOf());
		ERR_FAIL_COND_V(!SUCCEEDED(res), ERR_CANT_CREATE);
		swapchain.As(&window->swapchain);
		ERR_FAIL_NULL_V(window->swapchain, ERR_CANT_CREATE);

		format = swapchain_desc.Format;

		res = dxgi_factory->MakeWindowAssociation(window->hwnd, DXGI_MWA_NO_ALT_ENTER | DXGI_MWA_NO_WINDOW_CHANGES);
		ERR_FAIL_COND_V(!SUCCEEDED(res), ERR_CANT_CREATE);

		res = window->swapchain->GetDesc1(&swapchain_desc);
		ERR_FAIL_COND_V(!SUCCEEDED(res), ERR_CANT_CREATE);
		ERR_FAIL_COND_V(swapchain_desc.BufferCount != IMAGE_COUNT, ERR_BUG);
		window->width = swapchain_desc.Width;
		window->height = swapchain_desc.Height;

	} else {
		HRESULT res = window->swapchain->ResizeBuffers(IMAGE_COUNT, window->width, window->height, DXGI_FORMAT_UNKNOWN, swapchain_flags);
		ERR_FAIL_COND_V(!SUCCEEDED(res), ERR_UNAVAILABLE);
	}

	window->swapchain_flags = swapchain_flags;
	window->current_buffer = window->swapchain->GetCurrentBackBufferIndex();

	for (uint32_t i = 0; i < IMAGE_COUNT; i++) {
		RenderingDeviceDriverD3D12::FramebufferInfo *fb_info = &window->framebuffers[i];
		RenderingDeviceDriverD3D12::DescriptorsHeap::Walker walker = fb_info->rtv_heap.make_walker();

		HRESULT res = window->swapchain->GetBuffer(i, IID_PPV_ARGS(&window->render_targets[i]));
		ERR_FAIL_COND_V(!SUCCEEDED(res), ERR_CANT_CREATE);

		md.device->CreateRenderTargetView(window->render_targets[i].Get(), nullptr, walker.get_curr_cpu_handle());
	}

	return OK;
}

void D3D12Context::_init_device_factory() {
	uint32_t agility_sdk_version = GLOBAL_GET("rendering/rendering_device/d3d12/agility_sdk_version");
	String agility_sdk_path = String(".\\") + Engine::get_singleton()->get_architecture_name();

	// Note: symbol is not available in MinGW import library.
	PFN_D3D12_GET_INTERFACE d3d_D3D12GetInterface = (PFN_D3D12_GET_INTERFACE)GetProcAddress(LoadLibraryW(L"D3D12.dll"), "D3D12GetInterface");
	ERR_FAIL_COND(!d3d_D3D12GetInterface);

	ID3D12SDKConfiguration *sdk_config = nullptr;
	if (SUCCEEDED(d3d_D3D12GetInterface(CLSID_D3D12SDKConfigurationGodot, IID_PPV_ARGS(&sdk_config)))) {
		ID3D12SDKConfiguration1 *sdk_config1 = nullptr;
		if (SUCCEEDED(sdk_config->QueryInterface(&sdk_config1))) {
			if (SUCCEEDED(sdk_config1->CreateDeviceFactory(agility_sdk_version, agility_sdk_path.ascii().get_data(), IID_PPV_ARGS(device_factory.GetAddressOf())))) {
				d3d_D3D12GetInterface(CLSID_D3D12DeviceFactoryGodot, IID_PPV_ARGS(device_factory.GetAddressOf()));
			} else if (SUCCEEDED(sdk_config1->CreateDeviceFactory(agility_sdk_version, ".\\", IID_PPV_ARGS(device_factory.GetAddressOf())))) {
				d3d_D3D12GetInterface(CLSID_D3D12DeviceFactoryGodot, IID_PPV_ARGS(device_factory.GetAddressOf()));
			}
			sdk_config1->Release();
		}
		sdk_config->Release();
	}
}

Error D3D12Context::initialize() {
	_init_device_factory();

	if (_use_validation_layers()) {
		Error err = _initialize_debug_layers();
		ERR_FAIL_COND_V(err, ERR_CANT_CREATE);
	}

	int adapter_index = 0;

	Error err = _select_adapter(adapter_index);
	ERR_FAIL_COND_V(err, ERR_CANT_CREATE);

	err = _create_device(md);
	ERR_FAIL_COND_V(err, ERR_CANT_CREATE);

	_dump_adapter_info(adapter_index);

	err = _check_capabilities();
	ERR_FAIL_COND_V(err, ERR_CANT_CREATE);

	err = _get_device_limits();
	ERR_FAIL_COND_V(err, ERR_CANT_CREATE);

	{
		HRESULT res = md.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(frame_fence.GetAddressOf()));
		ERR_FAIL_COND_V(!SUCCEEDED(res), ERR_CANT_CREATE);
		frame_fence_event = CreateEvent(nullptr, FALSE, FALSE, nullptr);
		ERR_FAIL_NULL_V(frame_fence_event, ERR_CANT_CREATE);
	}

	md.driver = memnew(RenderingDeviceDriverD3D12(this, md.device.Get(), IMAGE_COUNT + 1));

	return OK;
}

void D3D12Context::set_setup_buffer(RDD::CommandBufferID p_command_buffer) {
	const RenderingDeviceDriverD3D12::CommandBufferInfo *cmd_buf_info = (const RenderingDeviceDriverD3D12::CommandBufferInfo *)p_command_buffer.id;
	command_list_queue[0] = cmd_buf_info->cmd_list.Get();
}

void D3D12Context::append_command_buffer(RDD::CommandBufferID p_command_buffer) {
	if (command_list_queue.size() <= command_list_count) {
		command_list_queue.resize(command_list_count + 1);
	}

	const RenderingDeviceDriverD3D12::CommandBufferInfo *cmd_buf_info = (const RenderingDeviceDriverD3D12::CommandBufferInfo *)p_command_buffer.id;
	command_list_queue[command_list_count] = cmd_buf_info->cmd_list.Get();
	command_list_count++;
}

void D3D12Context::_wait_for_idle_queue(ID3D12CommandQueue *p_queue) {
	md.fence_value++;
	p_queue->Signal(md.fence.Get(), md.fence_value);
	md.fence->SetEventOnCompletion(md.fence_value, md.fence_event);
	WaitForSingleObjectEx(md.fence_event, INFINITE, FALSE);
#ifdef PIX_ENABLED
	PIXNotifyWakeFromFenceSignal(md.fence_event);
#endif
}

void D3D12Context::flush(bool p_flush_setup, bool p_flush_pending, bool p_sync) {
	ERR_FAIL_COND_MSG(!p_sync, "Flush without sync is not supported."); // This is a special case for Vulkan on mobile XR hardware, not applicable to D3D12

	if (p_flush_setup && command_list_queue[0]) {
		md.queue->ExecuteCommandLists(1, command_list_queue.ptr());
		command_list_queue[0] = nullptr;
	}

	if (p_flush_pending && command_list_count > 1) {
		md.queue->ExecuteCommandLists(command_list_count - 1, command_list_queue.ptr() + 1);
		command_list_count = 1;
	}

	if (p_flush_setup || p_flush_pending) {
		_wait_for_idle_queue(md.queue.Get());
	}
}

Error D3D12Context::prepare_buffers(RDD::CommandBufferID p_command_buffer) {
	// Ensure no more than FRAME_LAG renderings are outstanding.
	if (frame >= IMAGE_COUNT) {
		UINT64 min_value = frame - IMAGE_COUNT;
		if (frame_fence->GetCompletedValue() < min_value) {
			frame_fence->SetEventOnCompletion(min_value, frame_fence_event);
			WaitForSingleObjectEx(frame_fence_event, INFINITE, FALSE);
#ifdef PIX_ENABLED
			PIXNotifyWakeFromFenceSignal(frame_fence_event);
#endif
		}
	}

	D3D12_RESOURCE_BARRIER *barriers = (D3D12_RESOURCE_BARRIER *)alloca(windows.size() * sizeof(D3D12_RESOURCE_BARRIER));
	uint32_t n = 0;
	for (KeyValue<int, Window> &E : windows) {
		Window *w = &E.value;
		w->current_buffer = w->swapchain->GetCurrentBackBufferIndex();
		barriers[n++] = CD3DX12_RESOURCE_BARRIER::Transition(w->render_targets[w->current_buffer].Get(), D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET);
	}
	const RenderingDeviceDriverD3D12::CommandBufferInfo *cmd_buf_info = (const RenderingDeviceDriverD3D12::CommandBufferInfo *)p_command_buffer.id;
	cmd_buf_info->cmd_list->ResourceBarrier(n, barriers);

	buffers_prepared = true;

	return OK;
}

void D3D12Context::postpare_buffers(RDD::CommandBufferID p_command_buffer) {
	D3D12_RESOURCE_BARRIER *barriers = (D3D12_RESOURCE_BARRIER *)alloca(windows.size() * sizeof(D3D12_RESOURCE_BARRIER));

	uint32_t n = 0;
	for (KeyValue<int, Window> &E : windows) {
		Window *w = &E.value;
		barriers[n++] = CD3DX12_RESOURCE_BARRIER::Transition(w->render_targets[w->current_buffer].Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT);
	}

	const RenderingDeviceDriverD3D12::CommandBufferInfo *cmd_buf_info = (const RenderingDeviceDriverD3D12::CommandBufferInfo *)p_command_buffer.id;
	cmd_buf_info->cmd_list->ResourceBarrier(n, barriers);
}

Error D3D12Context::swap_buffers() {
	ID3D12CommandList *const *commands_ptr = nullptr;
	UINT commands_to_submit = 0;

	if (command_list_queue[0] == nullptr) {
		// No setup command, but commands to submit, submit from the first and skip command.
		if (command_list_count > 1) {
			commands_ptr = command_list_queue.ptr() + 1;
			commands_to_submit = command_list_count - 1;
		}
	} else {
		commands_ptr = command_list_queue.ptr();
		commands_to_submit = command_list_count;
	}

	md.queue->ExecuteCommandLists(commands_to_submit, commands_ptr);

	command_list_queue[0] = nullptr;
	command_list_count = 1;

	for (KeyValue<int, Window> &E : windows) {
		Window *w = &E.value;

		if (!w->swapchain) {
			continue;
		}
		HRESULT res = w->swapchain->Present(w->sync_interval, w->present_flags);
		if (!SUCCEEDED(res)) {
			print_verbose("D3D12: Presenting swapchain of window " + itos(E.key) + " failed with error " + vformat("0x%08ux", (uint64_t)res) + ".");
		}
	}

	md.queue->Signal(frame_fence.Get(), frame);
	frame++;

	buffers_prepared = false;
	return OK;
}

void D3D12Context::resize_notify() {
}

RenderingDevice::Capabilities D3D12Context::get_device_capabilities() const {
	RenderingDevice::Capabilities c;
	c.device_family = RenderingDevice::DEVICE_DIRECTX;
	c.version_major = feature_level / 10;
	c.version_minor = feature_level % 10;
	return c;
}

ID3D12Device *D3D12Context::get_device() {
	return md.device.Get();
}

IDXGIAdapter *D3D12Context::get_adapter() {
	return gpu.Get();
}

int D3D12Context::get_swapchain_image_count() const {
	return IMAGE_COUNT;
}

DXGI_FORMAT D3D12Context::get_screen_format() const {
	return format;
}

const D3D12Context::DeviceLimits &D3D12Context::get_device_limits() const {
	return gpu_limits;
}

RID D3D12Context::local_device_create() {
	LocalDevice ld;
	_create_device(ld);
	ld.driver = memnew(RenderingDeviceDriverD3D12(this, ld.device.Get(), 1));
	return local_device_owner.make_rid(ld);
}

void D3D12Context::local_device_push_command_buffers(RID p_local_device, const RDD::CommandBufferID *p_buffers, int p_count) {
	LocalDevice *ld = local_device_owner.get_or_null(p_local_device);
	ERR_FAIL_COND(ld->waiting);

	ld->queue->ExecuteCommandLists(p_count, (ID3D12CommandList *const *)p_buffers);

	ld->waiting = true;
}

void D3D12Context::local_device_sync(RID p_local_device) {
	LocalDevice *ld = local_device_owner.get_or_null(p_local_device);
	ERR_FAIL_COND(!ld->waiting);

	ld->fence_value++;
	ld->queue->Signal(ld->fence.Get(), ld->fence_value);
	ld->fence->SetEventOnCompletion(ld->fence_value, ld->fence_event);
	WaitForSingleObjectEx(ld->fence_event, INFINITE, FALSE);
#ifdef PIX_ENABLED
	PIXNotifyWakeFromFenceSignal(ld->fence_event);
#endif

	ld->waiting = false;
}

void D3D12Context::local_device_free(RID p_local_device) {
	LocalDevice *ld = local_device_owner.get_or_null(p_local_device);
	memdelete(ld->driver);
	CloseHandle(ld->fence_event);
	local_device_owner.free(p_local_device);
}

void D3D12Context::set_object_name(ID3D12Object *p_object, String p_object_name) {
	ERR_FAIL_NULL(p_object);
	int name_len = p_object_name.size();
	WCHAR *name_w = (WCHAR *)alloca(sizeof(WCHAR) * (name_len + 1));
	MultiByteToWideChar(CP_UTF8, 0, p_object_name.utf8().get_data(), -1, name_w, name_len);
	p_object->SetName(name_w);
}

String D3D12Context::get_device_vendor_name() const {
	return adapter_vendor;
}
String D3D12Context::get_device_name() const {
	return adapter_name;
}

RenderingDevice::DeviceType D3D12Context::get_device_type() const {
	return adapter_type;
}

String D3D12Context::get_device_api_version() const {
	return vformat("%d_%d", feature_level / 10, feature_level % 10);
}

String D3D12Context::get_device_pipeline_cache_uuid() const {
	return pipeline_cache_id;
}

DisplayServer::VSyncMode D3D12Context::get_vsync_mode(DisplayServer::WindowID p_window) const {
	ERR_FAIL_COND_V_MSG(!windows.has(p_window), DisplayServer::VSYNC_ENABLED, "Could not get V-Sync mode for window with WindowID " + itos(p_window) + " because it does not exist.");
	return windows[p_window].vsync_mode;
}

void D3D12Context::set_vsync_mode(DisplayServer::WindowID p_window, DisplayServer::VSyncMode p_mode) {
	ERR_FAIL_COND_MSG(!windows.has(p_window), "Could not set V-Sync mode for window with WindowID " + itos(p_window) + " because it does not exist.");
	windows[p_window].vsync_mode = p_mode;
	_update_swap_chain(&windows[p_window]);
}

RenderingDeviceDriver *D3D12Context::get_driver(RID p_local_device) {
	if (p_local_device.is_valid()) {
		LocalDevice *ld = local_device_owner.get_or_null(p_local_device);
		ERR_FAIL_NULL_V(ld, nullptr);
		return ld->driver;
	} else {
		return md.driver;
	}
}

bool D3D12Context::is_debug_utils_enabled() const {
#ifdef PIX_ENABLED
	return true;
#else
	return false;
#endif
}

D3D12Context::D3D12Context() {
	command_list_queue.resize(1); // First one is always the setup command.
	command_list_queue[0] = nullptr;

	CharString cs = Engine::get_singleton()->get_architecture_name().ascii();
	memcpy(godot_nir_arch_name, (const char *)cs.get_data(), cs.size());
}

D3D12Context::~D3D12Context() {
	if (md.fence_event) {
		CloseHandle(md.fence_event);
	}
	if (frame_fence_event) {
		CloseHandle(frame_fence_event);
	}
}
