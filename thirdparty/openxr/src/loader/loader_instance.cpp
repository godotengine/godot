// Copyright (c) 2017-2025 The Khronos Group Inc.
// Copyright (c) 2017-2019 Valve Corporation
// Copyright (c) 2017-2019 LunarG, Inc.
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
// Initial Author: Mark Young <marky@lunarg.com>
//

#if defined(_MSC_VER) && !defined(_CRT_SECURE_NO_WARNINGS)
#define _CRT_SECURE_NO_WARNINGS
#endif  // defined(_MSC_VER) && !defined(_CRT_SECURE_NO_WARNINGS)

#include "loader_instance.hpp"

#include "api_layer_interface.hpp"
#include "hex_and_handles.h"
#include "loader_logger.hpp"
#include "runtime_interface.hpp"
#include "xr_generated_dispatch_table_core.h"
#include "xr_generated_loader.hpp"

#include <openxr/openxr.h>
#include <openxr/openxr_loader_negotiation.h>

#include <cstring>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {
std::unique_ptr<LoaderInstance>& GetSetCurrentLoaderInstance() {
    static std::unique_ptr<LoaderInstance> current_loader_instance;
    return current_loader_instance;
}
}  // namespace

namespace ActiveLoaderInstance {
XrResult Set(std::unique_ptr<LoaderInstance> loader_instance, const char* log_function_name) {
    if (GetSetCurrentLoaderInstance() != nullptr) {
        LoaderLogger::LogErrorMessage(log_function_name, "Active XrInstance handle already exists");
        return XR_ERROR_LIMIT_REACHED;
    }

    GetSetCurrentLoaderInstance() = std::move(loader_instance);
    return XR_SUCCESS;
}

XrResult Get(LoaderInstance** loader_instance, const char* log_function_name) {
    *loader_instance = GetSetCurrentLoaderInstance().get();
    if (*loader_instance == nullptr) {
        LoaderLogger::LogErrorMessage(log_function_name, "No active XrInstance handle.");
        return XR_ERROR_HANDLE_INVALID;
    }

    return XR_SUCCESS;
}

bool IsAvailable() { return GetSetCurrentLoaderInstance() != nullptr; }

void Remove() { GetSetCurrentLoaderInstance().reset(nullptr); }
}  // namespace ActiveLoaderInstance

// Extensions that are supported by the loader, but may not be supported
// the the runtime.
const std::array<XrExtensionProperties, 1>& LoaderInstance::LoaderSpecificExtensions() {
    static const std::array<XrExtensionProperties, 1> extensions{{XrExtensionProperties{
        XR_TYPE_EXTENSION_PROPERTIES, nullptr, {XR_EXT_DEBUG_UTILS_EXTENSION_NAME}, XR_EXT_debug_utils_SPEC_VERSION}}};
    return extensions;
}

namespace {
class InstanceCreateInfoManager {
   public:
    explicit InstanceCreateInfoManager(const XrInstanceCreateInfo* info) : original_create_info(info), modified_create_info(*info) {
        Reset();
    }

    // Reset the "modified" state to match the original state.
    void Reset() {
        enabled_extensions_cstr.clear();
        enabled_extensions_cstr.reserve(original_create_info->enabledExtensionCount);

        for (uint32_t i = 0; i < original_create_info->enabledExtensionCount; ++i) {
            enabled_extensions_cstr.push_back(original_create_info->enabledExtensionNames[i]);
        }
        Update();
    }

    // Remove extensions named in the parameter and return a pointer to the current state.
    const XrInstanceCreateInfo* FilterOutExtensions(const std::vector<const char*>& extensions_to_skip) {
        if (enabled_extensions_cstr.empty()) {
            return Get();
        }
        if (extensions_to_skip.empty()) {
            return Get();
        }
        for (auto& ext : extensions_to_skip) {
            FilterOutExtension(ext);
        }
        return Update();
    }
    // Remove the extension named in the parameter and return a pointer to the current state.
    const XrInstanceCreateInfo* FilterOutExtension(const char* extension_to_skip) {
        if (enabled_extensions_cstr.empty()) {
            return &modified_create_info;
        }
        auto b = enabled_extensions_cstr.begin();
        auto e = enabled_extensions_cstr.end();
        auto it = std::find_if(b, e, [&](const char* extension) { return strcmp(extension_to_skip, extension) == 0; });
        if (it != e) {
            // Just that one element goes away
            enabled_extensions_cstr.erase(it);
        }
        return Update();
    }

    // Get the current modified XrInstanceCreateInfo
    const XrInstanceCreateInfo* Get() const { return &modified_create_info; }

   private:
    const XrInstanceCreateInfo* Update() {
        modified_create_info.enabledExtensionCount = static_cast<uint32_t>(enabled_extensions_cstr.size());
        modified_create_info.enabledExtensionNames = enabled_extensions_cstr.empty() ? nullptr : enabled_extensions_cstr.data();
        return &modified_create_info;
    }
    const XrInstanceCreateInfo* original_create_info;

    XrInstanceCreateInfo modified_create_info;
    std::vector<const char*> enabled_extensions_cstr;
};
}  // namespace

// Factory method
XrResult LoaderInstance::CreateInstance(PFN_xrGetInstanceProcAddr get_instance_proc_addr_term,
                                        PFN_xrCreateInstance create_instance_term,
                                        PFN_xrCreateApiLayerInstance create_api_layer_instance_term,
                                        std::vector<std::unique_ptr<ApiLayerInterface>> api_layer_interfaces,
                                        const XrInstanceCreateInfo* info, std::unique_ptr<LoaderInstance>* loader_instance) {
    LoaderLogger::LogVerboseMessage("xrCreateInstance", "Entering LoaderInstance::CreateInstance");

    // Check the list of enabled extensions to make sure something supports them, and, if we do,
    // add it to the list of enabled extensions
    XrResult last_error = XR_SUCCESS;
    for (uint32_t ext = 0; ext < info->enabledExtensionCount; ++ext) {
        bool found = false;
        // First check the runtime
        if (RuntimeInterface::GetRuntime().SupportsExtension(info->enabledExtensionNames[ext])) {
            found = true;
        }
        // Next check the loader
        if (!found) {
            for (auto& loader_extension : LoaderInstance::LoaderSpecificExtensions()) {
                if (strcmp(loader_extension.extensionName, info->enabledExtensionNames[ext]) == 0) {
                    found = true;
                    break;
                }
            }
        }
        // Finally, check the enabled layers
        if (!found) {
            for (auto& layer_interface : api_layer_interfaces) {
                if (layer_interface->SupportsExtension(info->enabledExtensionNames[ext])) {
                    found = true;
                    break;
                }
            }
        }
        if (!found) {
            std::string msg = "LoaderInstance::CreateInstance, no support found for requested extension: ";
            msg += info->enabledExtensionNames[ext];
            LoaderLogger::LogErrorMessage("xrCreateInstance", msg);
            last_error = XR_ERROR_EXTENSION_NOT_PRESENT;
            break;
        }
    }

    // Topmost means "closest to the application"
    PFN_xrGetInstanceProcAddr topmost_gipa = get_instance_proc_addr_term;
    XrInstance instance{XR_NULL_HANDLE};

    if (XR_SUCCEEDED(last_error)) {
        // Remove the loader-supported-extensions (debug utils), if it's in the list of enabled extensions but not supported by
        // the runtime.
        InstanceCreateInfoManager create_info_manager{info};
        const XrInstanceCreateInfo* modified_create_info = info;
        if (info->enabledExtensionCount > 0) {
            std::vector<const char*> extensions_to_skip;
            for (const auto& ext : LoaderInstance::LoaderSpecificExtensions()) {
                if (!RuntimeInterface::GetRuntime().SupportsExtension(ext.extensionName)) {
                    extensions_to_skip.emplace_back(ext.extensionName);
                }
            }
            modified_create_info = create_info_manager.FilterOutExtensions(extensions_to_skip);
        }

        // Only start the xrCreateApiLayerInstance stack if we have layers.
        if (!api_layer_interfaces.empty()) {
            // Initialize an array of ApiLayerNextInfo structs
            std::unique_ptr<XrApiLayerNextInfo[]> next_info_list(new XrApiLayerNextInfo[api_layer_interfaces.size()]);
            size_t ni_index = api_layer_interfaces.size() - 1;
            for (size_t i = 0; i <= ni_index; i++) {
                next_info_list[i].structType = XR_LOADER_INTERFACE_STRUCT_API_LAYER_NEXT_INFO;
                next_info_list[i].structVersion = XR_API_LAYER_NEXT_INFO_STRUCT_VERSION;
                next_info_list[i].structSize = sizeof(XrApiLayerNextInfo);
            }

            // Go through all layers, and override the instance pointers with the layer version.  However,
            // go backwards through the layer list so we replace in reverse order so the layers can call their next function
            // appropriately.
            PFN_xrCreateApiLayerInstance topmost_cali_fp = create_api_layer_instance_term;
            XrApiLayerNextInfo* topmost_nextinfo = nullptr;
            for (auto layer_interface = api_layer_interfaces.rbegin(); layer_interface != api_layer_interfaces.rend();
                 ++layer_interface) {
                // Collect current layer's function pointers
                PFN_xrGetInstanceProcAddr cur_gipa_fp = (*layer_interface)->GetInstanceProcAddrFuncPointer();
                PFN_xrCreateApiLayerInstance cur_cali_fp = (*layer_interface)->GetCreateApiLayerInstanceFuncPointer();

                // Fill in layer info and link previous (lower) layer fxn pointers
                strncpy(next_info_list[ni_index].layerName, (*layer_interface)->LayerName().c_str(),
                        XR_MAX_API_LAYER_NAME_SIZE - 1);
                next_info_list[ni_index].layerName[XR_MAX_API_LAYER_NAME_SIZE - 1] = '\0';
                next_info_list[ni_index].next = topmost_nextinfo;
                next_info_list[ni_index].nextGetInstanceProcAddr = topmost_gipa;
                next_info_list[ni_index].nextCreateApiLayerInstance = topmost_cali_fp;

                // Update saved pointers for next iteration
                topmost_nextinfo = &next_info_list[ni_index];
                topmost_gipa = cur_gipa_fp;
                topmost_cali_fp = cur_cali_fp;
                ni_index--;
            }

            // Populate the ApiLayerCreateInfo struct and pass to topmost CreateApiLayerInstance()
            XrApiLayerCreateInfo api_layer_ci = {};
            api_layer_ci.structType = XR_LOADER_INTERFACE_STRUCT_API_LAYER_CREATE_INFO;
            api_layer_ci.structVersion = XR_API_LAYER_CREATE_INFO_STRUCT_VERSION;
            api_layer_ci.structSize = sizeof(XrApiLayerCreateInfo);
            api_layer_ci.loaderInstance = nullptr;  // Not used.
            api_layer_ci.settings_file_location[0] = '\0';
            api_layer_ci.nextInfo = next_info_list.get();
            //! @todo do we filter our create info extension list here?
            //! Think that actually each layer might need to filter...
            last_error = topmost_cali_fp(modified_create_info, &api_layer_ci, &instance);

        } else {
            // The loader's terminator is the topmost CreateInstance if there are no layers.
            last_error = create_instance_term(modified_create_info, &instance);
        }

        if (XR_FAILED(last_error)) {
            LoaderLogger::LogErrorMessage("xrCreateInstance", "LoaderInstance::CreateInstance chained CreateInstance call failed");
        }
    }

    if (XR_SUCCEEDED(last_error)) {
        loader_instance->reset(new LoaderInstance(instance, info, topmost_gipa, std::move(api_layer_interfaces)));

        std::ostringstream oss;
        oss << "LoaderInstance::CreateInstance succeeded with ";
        oss << (*loader_instance)->LayerInterfaces().size();
        oss << " layers enabled and runtime interface - created instance = ";
        oss << HandleToHexString((*loader_instance)->GetInstanceHandle());
        LoaderLogger::LogInfoMessage("xrCreateInstance", oss.str());
    }

    return last_error;
}

XrResult LoaderInstance::GetInstanceProcAddr(const char* name, PFN_xrVoidFunction* function) {
    return _topmost_gipa(_runtime_instance, name, function);
}

LoaderInstance::LoaderInstance(XrInstance instance, const XrInstanceCreateInfo* create_info, PFN_xrGetInstanceProcAddr topmost_gipa,
                               std::vector<std::unique_ptr<ApiLayerInterface>> api_layer_interfaces)
    : _runtime_instance(instance),
      _topmost_gipa(topmost_gipa),
      _api_layer_interfaces(std::move(api_layer_interfaces)),
      _dispatch_table(new XrGeneratedDispatchTableCore{}) {
    for (uint32_t ext = 0; ext < create_info->enabledExtensionCount; ++ext) {
        _enabled_extensions.push_back(create_info->enabledExtensionNames[ext]);
    }

    GeneratedXrPopulateDispatchTableCore(_dispatch_table.get(), instance, topmost_gipa);
}

LoaderInstance::~LoaderInstance() {
    std::ostringstream oss;
    oss << "Destroying LoaderInstance = ";
    oss << PointerToHexString(this);
    LoaderLogger::LogInfoMessage("xrDestroyInstance", oss.str());
}

bool LoaderInstance::ExtensionIsEnabled(const std::string& extension) {
    for (std::string& cur_enabled : _enabled_extensions) {
        if (cur_enabled == extension) {
            return true;
        }
    }
    return false;
}
