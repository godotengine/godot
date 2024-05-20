// Copyright (c) 2017-2024, The Khronos Group Inc.
// Copyright (c) 2017-2019 Valve Corporation
// Copyright (c) 2017-2019 LunarG, Inc.
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
// Initial Author: Mark Young <marky@lunarg.com>
//

#pragma once

#include <string>
#include <vector>
#include <memory>

#include <openxr/openxr.h>
#include <openxr/openxr_loader_negotiation.h>

#include "loader_platform.hpp"

struct XrGeneratedDispatchTable;

class ApiLayerInterface {
   public:
    // Factory method
    static XrResult LoadApiLayers(const std::string& openxr_command, uint32_t enabled_api_layer_count,
                                  const char* const* enabled_api_layer_names,
                                  std::vector<std::unique_ptr<ApiLayerInterface>>& api_layer_interfaces);
    // Static queries
    static XrResult GetApiLayerProperties(const std::string& openxr_command, uint32_t incoming_count, uint32_t* outgoing_count,
                                          XrApiLayerProperties* api_layer_properties);
    static XrResult GetInstanceExtensionProperties(const std::string& openxr_command, const char* layer_name,
                                                   std::vector<XrExtensionProperties>& extension_properties);

    ApiLayerInterface(const std::string& layer_name, LoaderPlatformLibraryHandle layer_library,
                      std::vector<std::string>& supported_extensions, PFN_xrGetInstanceProcAddr get_instance_proc_addr,
                      PFN_xrCreateApiLayerInstance create_api_layer_instance);
    virtual ~ApiLayerInterface();

    PFN_xrGetInstanceProcAddr GetInstanceProcAddrFuncPointer() { return _get_instance_proc_addr; }
    PFN_xrCreateApiLayerInstance GetCreateApiLayerInstanceFuncPointer() { return _create_api_layer_instance; }

    std::string LayerName() { return _layer_name; }

    // Generated methods
    bool SupportsExtension(const std::string& extension_name) const;

   private:
    std::string _layer_name;
    LoaderPlatformLibraryHandle _layer_library;
    PFN_xrGetInstanceProcAddr _get_instance_proc_addr;
    PFN_xrCreateApiLayerInstance _create_api_layer_instance;
    std::vector<std::string> _supported_extensions;
};
