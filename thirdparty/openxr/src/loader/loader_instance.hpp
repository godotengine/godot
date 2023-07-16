// Copyright (c) 2017-2023, The Khronos Group Inc.
// Copyright (c) 2017-2019 Valve Corporation
// Copyright (c) 2017-2019 LunarG, Inc.
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
// Initial Author: Mark Young <marky@lunarg.com>
//

#pragma once

#include "extra_algorithms.h"
#include "loader_interfaces.h"

#include <openxr/openxr.h>

#include <array>
#include <cmath>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

class ApiLayerInterface;
struct XrGeneratedDispatchTable;
class LoaderInstance;

// Manage the single loader instance that is available.
namespace ActiveLoaderInstance {
// Set the active loader instance. This will fail if there is already an active loader instance.
XrResult Set(std::unique_ptr<LoaderInstance> loader_instance, const char* log_function_name);

// Returns true if there is an active loader instance.
bool IsAvailable();

// Get the active LoaderInstance.
XrResult Get(LoaderInstance** loader_instance, const char* log_function_name);

// Destroy the currently active LoaderInstance if there is one. This will make the loader able to create a new XrInstance if needed.
void Remove();
};  // namespace ActiveLoaderInstance

// Manages information needed by the loader for an XrInstance, such as what extensions are available and the dispatch table.
class LoaderInstance {
   public:
    // Factory method
    static XrResult CreateInstance(PFN_xrGetInstanceProcAddr get_instance_proc_addr_term, PFN_xrCreateInstance create_instance_term,
                                   PFN_xrCreateApiLayerInstance create_api_layer_instance_term,
                                   std::vector<std::unique_ptr<ApiLayerInterface>> layer_interfaces,
                                   const XrInstanceCreateInfo* createInfo, std::unique_ptr<LoaderInstance>* loader_instance);
    static const std::array<XrExtensionProperties, 1>& LoaderSpecificExtensions();

    virtual ~LoaderInstance();

    XrInstance GetInstanceHandle() { return _runtime_instance; }
    const std::unique_ptr<XrGeneratedDispatchTable>& DispatchTable() { return _dispatch_table; }
    std::vector<std::unique_ptr<ApiLayerInterface>>& LayerInterfaces() { return _api_layer_interfaces; }
    bool ExtensionIsEnabled(const std::string& extension);
    XrDebugUtilsMessengerEXT DefaultDebugUtilsMessenger() { return _messenger; }
    void SetDefaultDebugUtilsMessenger(XrDebugUtilsMessengerEXT messenger) { _messenger = messenger; }
    XrResult GetInstanceProcAddr(const char* name, PFN_xrVoidFunction* function);

   private:
    LoaderInstance(XrInstance instance, const XrInstanceCreateInfo* createInfo, PFN_xrGetInstanceProcAddr topmost_gipa,
                   std::vector<std::unique_ptr<ApiLayerInterface>> api_layer_interfaces);

   private:
    XrInstance _runtime_instance{XR_NULL_HANDLE};
    PFN_xrGetInstanceProcAddr _topmost_gipa{nullptr};
    std::vector<std::string> _enabled_extensions;
    std::vector<std::unique_ptr<ApiLayerInterface>> _api_layer_interfaces;

    std::unique_ptr<XrGeneratedDispatchTable> _dispatch_table;
    // Internal debug messenger created during xrCreateInstance
    XrDebugUtilsMessengerEXT _messenger{XR_NULL_HANDLE};
};
