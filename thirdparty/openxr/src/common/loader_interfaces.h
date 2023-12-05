// Copyright (c) 2017-2023, The Khronos Group Inc.
// Copyright (c) 2017 Valve Corporation
// Copyright (c) 2017 LunarG, Inc.
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
// Initial Author: Mark Young <marky@lunarg.com>
//

#pragma once

#include <openxr/openxr.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declare.
typedef struct XrApiLayerCreateInfo XrApiLayerCreateInfo;

// Function pointer prototype for the xrCreateApiLayerInstance function used in place of xrCreateInstance.
// This function allows us to pass special API layer information to each layer during the process of creating an Instance.
typedef XrResult(XRAPI_PTR *PFN_xrCreateApiLayerInstance)(const XrInstanceCreateInfo *info,
                                                          const XrApiLayerCreateInfo *apiLayerInfo, XrInstance *instance);

// Loader/API Layer Interface versions
//  1 - First version, introduces negotiation structure and functions
#define XR_CURRENT_LOADER_API_LAYER_VERSION 1

// Loader/Runtime Interface versions
//  1 - First version, introduces negotiation structure and functions
#define XR_CURRENT_LOADER_RUNTIME_VERSION 1

// Version negotiation values
typedef enum XrLoaderInterfaceStructs {
    XR_LOADER_INTERFACE_STRUCT_UNINTIALIZED = 0,
    XR_LOADER_INTERFACE_STRUCT_LOADER_INFO,
    XR_LOADER_INTERFACE_STRUCT_API_LAYER_REQUEST,
    XR_LOADER_INTERFACE_STRUCT_RUNTIME_REQUEST,
    XR_LOADER_INTERFACE_STRUCT_API_LAYER_CREATE_INFO,
    XR_LOADER_INTERFACE_STRUCT_API_LAYER_NEXT_INFO,
} XrLoaderInterfaceStructs;

#define XR_LOADER_INFO_STRUCT_VERSION 1
typedef struct XrNegotiateLoaderInfo {
    XrLoaderInterfaceStructs structType;  // XR_LOADER_INTERFACE_STRUCT_LOADER_INFO
    uint32_t structVersion;               // XR_LOADER_INFO_STRUCT_VERSION
    size_t structSize;                    // sizeof(XrNegotiateLoaderInfo)
    uint32_t minInterfaceVersion;
    uint32_t maxInterfaceVersion;
    XrVersion minApiVersion;
    XrVersion maxApiVersion;
} XrNegotiateLoaderInfo;

#define XR_API_LAYER_INFO_STRUCT_VERSION 1
typedef struct XrNegotiateApiLayerRequest {
    XrLoaderInterfaceStructs structType;  // XR_LOADER_INTERFACE_STRUCT_API_LAYER_REQUEST
    uint32_t structVersion;               // XR_API_LAYER_INFO_STRUCT_VERSION
    size_t structSize;                    // sizeof(XrNegotiateApiLayerRequest)
    uint32_t layerInterfaceVersion;       // CURRENT_LOADER_API_LAYER_VERSION
    XrVersion layerApiVersion;
    PFN_xrGetInstanceProcAddr getInstanceProcAddr;
    PFN_xrCreateApiLayerInstance createApiLayerInstance;
} XrNegotiateApiLayerRequest;

#define XR_RUNTIME_INFO_STRUCT_VERSION 1
typedef struct XrNegotiateRuntimeRequest {
    XrLoaderInterfaceStructs structType;  // XR_LOADER_INTERFACE_STRUCT_RUNTIME_REQUEST
    uint32_t structVersion;               // XR_RUNTIME_INFO_STRUCT_VERSION
    size_t structSize;                    // sizeof(XrNegotiateRuntimeRequest)
    uint32_t runtimeInterfaceVersion;     // CURRENT_LOADER_RUNTIME_VERSION
    XrVersion runtimeApiVersion;
    PFN_xrGetInstanceProcAddr getInstanceProcAddr;
} XrNegotiateRuntimeRequest;

// Function used to negotiate an interface betewen the loader and an API layer.  Each library exposing one or
// more API layers needs to expose at least this function.
typedef XrResult(XRAPI_PTR *PFN_xrNegotiateLoaderApiLayerInterface)(const XrNegotiateLoaderInfo *loaderInfo,
                                                                    const char *apiLayerName,
                                                                    XrNegotiateApiLayerRequest *apiLayerRequest);

// Function used to negotiate an interface betewen the loader and a runtime.  Each runtime should expose
// at least this function.
typedef XrResult(XRAPI_PTR *PFN_xrNegotiateLoaderRuntimeInterface)(const XrNegotiateLoaderInfo *loaderInfo,
                                                                   XrNegotiateRuntimeRequest *runtimeRequest);

// Forward declare.
typedef struct XrApiLayerNextInfo XrApiLayerNextInfo;

#define XR_API_LAYER_NEXT_INFO_STRUCT_VERSION 1
struct XrApiLayerNextInfo {
    XrLoaderInterfaceStructs structType;                      // XR_LOADER_INTERFACE_STRUCT_API_LAYER_NEXT_INFO
    uint32_t structVersion;                                   // XR_API_LAYER_NEXT_INFO_STRUCT_VERSION
    size_t structSize;                                        // sizeof(XrApiLayerNextInfo)
    char layerName[XR_MAX_API_LAYER_NAME_SIZE];               // Name of API layer which should receive this info
    PFN_xrGetInstanceProcAddr nextGetInstanceProcAddr;        // Pointer to next API layer's xrGetInstanceProcAddr
    PFN_xrCreateApiLayerInstance nextCreateApiLayerInstance;  // Pointer to next API layer's xrCreateApiLayerInstance
    XrApiLayerNextInfo *next;                                 // Pointer to the next API layer info in the sequence
};

#define XR_API_LAYER_MAX_SETTINGS_PATH_SIZE 512
#define XR_API_LAYER_CREATE_INFO_STRUCT_VERSION 1
typedef struct XrApiLayerCreateInfo {
    XrLoaderInterfaceStructs structType;                               // XR_LOADER_INTERFACE_STRUCT_API_LAYER_CREATE_INFO
    uint32_t structVersion;                                            // XR_API_LAYER_CREATE_INFO_STRUCT_VERSION
    size_t structSize;                                                 // sizeof(XrApiLayerCreateInfo)
    void *loaderInstance;                                              // Pointer to the LoaderInstance class
    char settings_file_location[XR_API_LAYER_MAX_SETTINGS_PATH_SIZE];  // Location to the found settings file (or empty '\0')
    XrApiLayerNextInfo *nextInfo;                                      // Pointer to the next API layer's Info
} XrApiLayerCreateInfo;

#ifdef __cplusplus
}  // extern "C"
#endif
