/**************************************************************************/
/*  dawn_boundary.cpp                                                     */
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

#include "dawn_boundary.h"

#include <dawn/native/DawnNative.h>

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#ifdef _WIN32
#include <windows.h>

#include <dawn/native/D3DBackend.h>
#else
#include <dlfcn.h>
#endif

// Reference Dawn's complete C entry-point table when linking its static SDK.
// Export the WebGPU entry points and native helpers, keeping C++ symbols private.
extern "C" WGPUProc godot_webgpu_dawn_resolve(WGPUStringView p_name) {
	return wgpuGetProcAddress(p_name);
}

extern "C" WGPUInstance godot_webgpu_create_instance(const WGPUInstanceDescriptor *p_descriptor, const WGPULoggingCallbackInfo *p_logging) {
	std::vector<std::string> paths;
	auto add_path = [&](const std::filesystem::path &p_path) {
		auto utf8 = p_path.u8string();
		std::string path(reinterpret_cast<const char *>(utf8.data()), utf8.size());
		if (!path.empty() && path.back() != '/' && path.back() != '\\') {
			path += std::filesystem::path::preferred_separator;
		}
		paths.push_back(std::move(path));
	};
#ifdef _WIN32
	HMODULE module = nullptr;
	if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
				reinterpret_cast<LPCWSTR>(&godot_webgpu_create_instance), &module)) {
		wchar_t filename[32768];
		DWORD length = GetModuleFileNameW(module, filename, 32768);
		if (length > 0 && length < 32768) {
			add_path(std::filesystem::path(std::wstring(filename, length)).parent_path());
		}
	}
	wchar_t system_directory[32768];
	UINT length = GetSystemDirectoryW(system_directory, 32768);
	if (length > 0 && length < 32768) {
		add_path(std::filesystem::path(std::wstring(system_directory, length)));
	}
#else
	Dl_info module = {};
	if (dladdr(reinterpret_cast<const void *>(&godot_webgpu_create_instance), &module) && module.dli_fname) {
		add_path(std::filesystem::absolute(module.dli_fname).parent_path());
	}
#endif
	std::vector<std::string_view> views;
	for (const auto &path : paths) {
		views.emplace_back(path);
	}
	dawn::native::DawnInstanceDescriptor native;
	native.additionalRuntimeSearchPaths = views;
	if (p_logging) {
		native.loggingCallbackInfo = *p_logging;
	}
	WGPUInstanceDescriptor descriptor = *p_descriptor;
	auto *chain = reinterpret_cast<WGPUChainedStruct *>(&native);
	chain->next = descriptor.nextInChain;
	descriptor.nextInChain = chain;
	return wgpuCreateInstance(&descriptor);
}

extern "C" WGPUFuture godot_webgpu_request_adapter(WGPUInstance p_instance, const WGPURequestAdapterOptions *p_options, uint64_t p_luid, WGPURequestAdapterCallbackInfo p_callback) {
#ifdef _WIN32
	if (p_luid != 0 && (p_options->backendType == WGPUBackendType_Undefined || p_options->backendType == WGPUBackendType_D3D12 || p_options->backendType == WGPUBackendType_D3D11)) {
		dawn::native::d3d::RequestAdapterOptionsLUID luid;
		luid.adapterLUID.LowPart = uint32_t(p_luid);
		luid.adapterLUID.HighPart = int32_t(p_luid >> 32);
		WGPURequestAdapterOptions options = *p_options;
		auto *chain = reinterpret_cast<WGPUChainedStruct *>(&luid);
		chain->next = options.nextInChain;
		options.nextInChain = chain;
		return wgpuInstanceRequestAdapter(p_instance, &options, p_callback);
	}
#endif
	return wgpuInstanceRequestAdapter(p_instance, p_options, p_callback);
}
