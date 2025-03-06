// Copyright (c) 2017-2024, The Khronos Group Inc.
// Copyright (c) 2017-2019 Valve Corporation
// Copyright (c) 2017-2019 LunarG, Inc.
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
// Initial Author: Mark Young <marky@lunarg.com>
//

#include "runtime_interface.hpp"

#include <openxr/openxr.h>
#include <openxr/openxr_loader_negotiation.h>

#include "manifest_file.hpp"
#include "loader_init_data.hpp"
#include "loader_logger.hpp"
#include "loader_platform.hpp"
#include "xr_generated_dispatch_table_core.h"

#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef XR_USE_PLATFORM_ANDROID
#include <json/value.h>

// Needed for the loader init struct
#include <xr_dependencies.h>
#include <openxr/openxr_platform.h>
#endif  // XR_USE_PLATFORM_ANDROID

#if defined(XR_KHR_LOADER_INIT_SUPPORT) && defined(XR_USE_PLATFORM_ANDROID)
XrResult GetPlatformRuntimeVirtualManifest(Json::Value& out_manifest) {
    using wrap::android::content::Context;
    auto& initData = LoaderInitData::instance();
    if (!initData.initialized()) {
        return XR_ERROR_INITIALIZATION_FAILED;
    }
    auto context = Context(reinterpret_cast<jobject>(initData.getData().applicationContext));
    if (context.isNull()) {
        return XR_ERROR_INITIALIZATION_FAILED;
    }
    Json::Value virtualManifest;
    if (0 != openxr_android::getActiveRuntimeVirtualManifest(context, virtualManifest)) {
        return XR_ERROR_INITIALIZATION_FAILED;
    }
    out_manifest = virtualManifest;
    return XR_SUCCESS;
}
#endif  // defined(XR_USE_PLATFORM_ANDROID) && defined(XR_KHR_LOADER_INIT_SUPPORT)

XrResult RuntimeInterface::TryLoadingSingleRuntime(const std::string& openxr_command,
                                                   std::unique_ptr<RuntimeManifestFile>& manifest_file) {
    LoaderPlatformLibraryHandle runtime_library = LoaderPlatformLibraryOpen(manifest_file->LibraryPath());
    if (nullptr == runtime_library) {
        std::string library_message = LoaderPlatformLibraryOpenError(manifest_file->LibraryPath());
        std::string warning_message = "RuntimeInterface::LoadRuntime skipping manifest file ";
        warning_message += manifest_file->Filename();
        warning_message += ", failed to load with message \"";
        warning_message += library_message;
        warning_message += "\"";
        LoaderLogger::LogErrorMessage(openxr_command, warning_message);
        return XR_ERROR_FILE_ACCESS_ERROR;
    }
#ifdef XR_KHR_LOADER_INIT_SUPPORT
    if (!LoaderInitData::instance().initialized()) {
        LoaderLogger::LogErrorMessage(openxr_command, "RuntimeInterface::LoadRuntime skipping manifest file " +
                                                          manifest_file->Filename() +
                                                          " because xrInitializeLoaderKHR was not yet called.");

        LoaderPlatformLibraryClose(runtime_library);
        return XR_ERROR_VALIDATION_FAILURE;
    }
    bool forwardedInitLoader = false;
    {
        // If we have xrInitializeLoaderKHR exposed as an export, forward call to it.
        const auto function_name = manifest_file->GetFunctionName("xrInitializeLoaderKHR");
        auto initLoader =
            reinterpret_cast<PFN_xrInitializeLoaderKHR>(LoaderPlatformLibraryGetProcAddr(runtime_library, function_name));
        if (initLoader != nullptr) {
            // we found the entry point one way or another.
            LoaderLogger::LogInfoMessage(openxr_command,
                                         "RuntimeInterface::LoadRuntime forwarding xrInitializeLoaderKHR call to runtime before "
                                         "calling xrNegotiateLoaderRuntimeInterface.");
            XrResult res = initLoader(LoaderInitData::instance().getParam());
            if (!XR_SUCCEEDED(res)) {
                LoaderLogger::LogErrorMessage(openxr_command,
                                              "RuntimeInterface::LoadRuntime forwarded call to xrInitializeLoaderKHR failed.");

                LoaderPlatformLibraryClose(runtime_library);
                return res;
            }
            forwardedInitLoader = true;
        }
    }
#endif

    // Get and settle on an runtime interface version (using any provided name if required).
    std::string function_name = manifest_file->GetFunctionName("xrNegotiateLoaderRuntimeInterface");
    auto negotiate =
        reinterpret_cast<PFN_xrNegotiateLoaderRuntimeInterface>(LoaderPlatformLibraryGetProcAddr(runtime_library, function_name));

    // Loader info for negotiation
    XrNegotiateLoaderInfo loader_info = {};
    loader_info.structType = XR_LOADER_INTERFACE_STRUCT_LOADER_INFO;
    loader_info.structVersion = XR_LOADER_INFO_STRUCT_VERSION;
    loader_info.structSize = sizeof(XrNegotiateLoaderInfo);
    loader_info.minInterfaceVersion = 1;
    loader_info.maxInterfaceVersion = XR_CURRENT_LOADER_RUNTIME_VERSION;
    loader_info.minApiVersion = XR_MAKE_VERSION(1, 0, 0);
    loader_info.maxApiVersion = XR_MAKE_VERSION(1, 0x3ff, 0xfff);  // Maximum allowed version for this major version.

    // Set up the runtime return structure
    XrNegotiateRuntimeRequest runtime_info = {};
    runtime_info.structType = XR_LOADER_INTERFACE_STRUCT_RUNTIME_REQUEST;
    runtime_info.structVersion = XR_RUNTIME_INFO_STRUCT_VERSION;
    runtime_info.structSize = sizeof(XrNegotiateRuntimeRequest);

    // Skip calling the negotiate function and fail if the function pointer
    // could not get loaded
    XrResult res = XR_ERROR_RUNTIME_FAILURE;
    if (nullptr != negotiate) {
        res = negotiate(&loader_info, &runtime_info);
    } else {
        std::string error_message = "RuntimeInterface::LoadRuntime failed to find negotiate function ";
        error_message += function_name;
        LoaderLogger::LogErrorMessage(openxr_command, error_message);
    }
    // If we supposedly succeeded, but got a nullptr for GetInstanceProcAddr
    // then something still went wrong, so return with an error.
    if (XR_SUCCEEDED(res)) {
        uint32_t runtime_major = XR_VERSION_MAJOR(runtime_info.runtimeApiVersion);
        uint32_t runtime_minor = XR_VERSION_MINOR(runtime_info.runtimeApiVersion);
        uint32_t loader_major = XR_VERSION_MAJOR(XR_CURRENT_API_VERSION);
        if (nullptr == runtime_info.getInstanceProcAddr) {
            std::string error_message = "RuntimeInterface::LoadRuntime skipping manifest file ";
            error_message += manifest_file->Filename();
            error_message += ", negotiation succeeded but returned NULL getInstanceProcAddr";
            LoaderLogger::LogErrorMessage(openxr_command, error_message);
            res = XR_ERROR_FILE_CONTENTS_INVALID;
        } else if (0 >= runtime_info.runtimeInterfaceVersion ||
                   XR_CURRENT_LOADER_RUNTIME_VERSION < runtime_info.runtimeInterfaceVersion) {
            std::string error_message = "RuntimeInterface::LoadRuntime skipping manifest file ";
            error_message += manifest_file->Filename();
            error_message += ", negotiation succeeded but returned invalid interface version";
            LoaderLogger::LogErrorMessage(openxr_command, error_message);
            res = XR_ERROR_FILE_CONTENTS_INVALID;
        } else if (runtime_major != loader_major || (runtime_major == 0 && runtime_minor == 0)) {
            std::string error_message = "RuntimeInterface::LoadRuntime skipping manifest file ";
            error_message += manifest_file->Filename();
            error_message += ", OpenXR version returned not compatible with this loader";
            LoaderLogger::LogErrorMessage(openxr_command, error_message);
            res = XR_ERROR_FILE_CONTENTS_INVALID;
        }
    }
#ifdef XR_KHR_LOADER_INIT_SUPPORT
    if (XR_SUCCEEDED(res) && !forwardedInitLoader) {
        // Forward initialize loader call, where possible and if we did not do so before.
        PFN_xrVoidFunction initializeVoid = nullptr;
        PFN_xrInitializeLoaderKHR initialize = nullptr;

        // Now we may try asking xrGetInstanceProcAddr
        if (XR_SUCCEEDED(runtime_info.getInstanceProcAddr(XR_NULL_HANDLE, "xrInitializeLoaderKHR", &initializeVoid))) {
            if (initializeVoid == nullptr) {
                LoaderLogger::LogErrorMessage(openxr_command,
                                              "RuntimeInterface::LoadRuntime got success from xrGetInstanceProcAddr "
                                              "for xrInitializeLoaderKHR, but output a null pointer.");
                res = XR_ERROR_RUNTIME_FAILURE;
            } else {
                initialize = reinterpret_cast<PFN_xrInitializeLoaderKHR>(initializeVoid);
            }
        }
        if (initialize != nullptr) {
            // we found the entry point one way or another.
            LoaderLogger::LogInfoMessage(openxr_command,
                                         "RuntimeInterface::LoadRuntime forwarding xrInitializeLoaderKHR call to runtime after "
                                         "calling xrNegotiateLoaderRuntimeInterface.");
            res = initialize(LoaderInitData::instance().getParam());
            if (!XR_SUCCEEDED(res)) {
                LoaderLogger::LogErrorMessage(openxr_command,
                                              "RuntimeInterface::LoadRuntime forwarded call to xrInitializeLoaderKHR failed.");
            }
        }
    }
#endif
    if (XR_FAILED(res)) {
        std::string warning_message = "RuntimeInterface::LoadRuntime skipping manifest file ";
        warning_message += manifest_file->Filename();
        warning_message += ", negotiation failed with error ";
        warning_message += std::to_string(res);
        LoaderLogger::LogErrorMessage(openxr_command, warning_message);
        LoaderPlatformLibraryClose(runtime_library);
        return res;
    }

    std::string info_message = "RuntimeInterface::LoadRuntime succeeded loading runtime defined in manifest file ";
    info_message += manifest_file->Filename();
    info_message += " using interface version ";
    info_message += std::to_string(runtime_info.runtimeInterfaceVersion);
    info_message += " and OpenXR API version ";
    info_message += std::to_string(XR_VERSION_MAJOR(runtime_info.runtimeApiVersion));
    info_message += ".";
    info_message += std::to_string(XR_VERSION_MINOR(runtime_info.runtimeApiVersion));
    LoaderLogger::LogInfoMessage(openxr_command, info_message);

    // Use this runtime
    GetInstance().reset(new RuntimeInterface(runtime_library, runtime_info.getInstanceProcAddr));

    // Grab the list of extensions this runtime supports for easy filtering after the
    // xrCreateInstance call
    std::vector<std::string> supported_extensions;
    std::vector<XrExtensionProperties> extension_properties;
    GetInstance()->GetInstanceExtensionProperties(extension_properties);
    supported_extensions.reserve(extension_properties.size());
    for (XrExtensionProperties ext_prop : extension_properties) {
        supported_extensions.emplace_back(ext_prop.extensionName);
    }
    GetInstance()->SetSupportedExtensions(supported_extensions);

    return XR_SUCCESS;
}

XrResult RuntimeInterface::LoadRuntime(const std::string& openxr_command) {
    // If something's already loaded, we're done here.
    if (GetInstance() != nullptr) {
        return XR_SUCCESS;
    }
#ifdef XR_KHR_LOADER_INIT_SUPPORT
    if (!LoaderInitData::instance().initialized()) {
        LoaderLogger::LogErrorMessage(
            openxr_command, "RuntimeInterface::LoadRuntime cannot run because xrInitializeLoaderKHR was not successfully called.");
        return XR_ERROR_INITIALIZATION_FAILED;
    }
#endif  // XR_KHR_LOADER_INIT_SUPPORT

    std::vector<std::unique_ptr<RuntimeManifestFile>> runtime_manifest_files = {};

    // Find the available runtimes which we may need to report information for.
    XrResult last_error = RuntimeManifestFile::FindManifestFiles(openxr_command, runtime_manifest_files);
    if (XR_FAILED(last_error)) {
        LoaderLogger::LogErrorMessage(openxr_command, "RuntimeInterface::LoadRuntimes - unknown error");
    } else {
        last_error = XR_ERROR_RUNTIME_UNAVAILABLE;
        for (std::unique_ptr<RuntimeManifestFile>& manifest_file : runtime_manifest_files) {
            last_error = RuntimeInterface::TryLoadingSingleRuntime(openxr_command, manifest_file);
            if (XR_SUCCEEDED(last_error)) {
                break;
            }
        }
    }

    // Unsuccessful in loading any runtime, throw the runtime unavailable message.
    if (XR_FAILED(last_error)) {
        LoaderLogger::LogErrorMessage(openxr_command, "RuntimeInterface::LoadRuntimes - failed to load a runtime");
        last_error = XR_ERROR_RUNTIME_UNAVAILABLE;
    }

    return last_error;
}

void RuntimeInterface::UnloadRuntime(const std::string& openxr_command) {
    if (GetInstance()) {
        LoaderLogger::LogInfoMessage(openxr_command, "RuntimeInterface::UnloadRuntime - Unloading RuntimeInterface");
        GetInstance().reset();
    }
}

XrResult RuntimeInterface::GetInstanceProcAddr(XrInstance instance, const char* name, PFN_xrVoidFunction* function) {
    return GetInstance()->_get_instance_proc_addr(instance, name, function);
}

const XrGeneratedDispatchTableCore* RuntimeInterface::GetDispatchTable(XrInstance instance) {
    XrGeneratedDispatchTableCore* table = nullptr;
    std::lock_guard<std::mutex> mlock(GetInstance()->_dispatch_table_mutex);
    auto it = GetInstance()->_dispatch_table_map.find(instance);
    if (it != GetInstance()->_dispatch_table_map.end()) {
        table = it->second.get();
    }
    return table;
}

const XrGeneratedDispatchTableCore* RuntimeInterface::GetDebugUtilsMessengerDispatchTable(XrDebugUtilsMessengerEXT messenger) {
    XrInstance runtime_instance = XR_NULL_HANDLE;
    {
        std::lock_guard<std::mutex> mlock(GetInstance()->_messenger_to_instance_mutex);
        auto it = GetInstance()->_messenger_to_instance_map.find(messenger);
        if (it != GetInstance()->_messenger_to_instance_map.end()) {
            runtime_instance = it->second;
        }
    }
    return GetDispatchTable(runtime_instance);
}

RuntimeInterface::RuntimeInterface(LoaderPlatformLibraryHandle runtime_library, PFN_xrGetInstanceProcAddr get_instance_proc_addr)
    : _runtime_library(runtime_library), _get_instance_proc_addr(get_instance_proc_addr) {}

RuntimeInterface::~RuntimeInterface() {
    std::string info_message = "RuntimeInterface being destroyed.";
    LoaderLogger::LogInfoMessage("", info_message);
    {
        std::lock_guard<std::mutex> mlock(_dispatch_table_mutex);
        _dispatch_table_map.clear();
    }
    LoaderPlatformLibraryClose(_runtime_library);
}

void RuntimeInterface::GetInstanceExtensionProperties(std::vector<XrExtensionProperties>& extension_properties) {
    std::vector<XrExtensionProperties> runtime_extension_properties;
    PFN_xrEnumerateInstanceExtensionProperties rt_xrEnumerateInstanceExtensionProperties;
    _get_instance_proc_addr(XR_NULL_HANDLE, "xrEnumerateInstanceExtensionProperties",
                            reinterpret_cast<PFN_xrVoidFunction*>(&rt_xrEnumerateInstanceExtensionProperties));
    uint32_t count = 0;
    uint32_t count_output = 0;
    // Get the count from the runtime
    rt_xrEnumerateInstanceExtensionProperties(nullptr, count, &count_output, nullptr);
    if (count_output > 0) {
        XrExtensionProperties example_properties{};
        example_properties.type = XR_TYPE_EXTENSION_PROPERTIES;
        runtime_extension_properties.resize(count_output, example_properties);
        count = count_output;
        rt_xrEnumerateInstanceExtensionProperties(nullptr, count, &count_output, runtime_extension_properties.data());
    }
    size_t ext_count = runtime_extension_properties.size();
    size_t props_count = extension_properties.size();
    for (size_t ext = 0; ext < ext_count; ++ext) {
        bool found = false;
        for (size_t prop = 0; prop < props_count; ++prop) {
            // If we find it, then make sure the spec version matches that of the runtime instead of the
            // layer.
            if (strcmp(extension_properties[prop].extensionName, runtime_extension_properties[ext].extensionName) == 0) {
                // Make sure the spec version used is the runtime's
                extension_properties[prop].extensionVersion = runtime_extension_properties[ext].extensionVersion;
                found = true;
                break;
            }
        }
        if (!found) {
            extension_properties.push_back(runtime_extension_properties[ext]);
        }
    }
}

XrResult RuntimeInterface::CreateInstance(const XrInstanceCreateInfo* info, XrInstance* instance) {
    XrResult res = XR_SUCCESS;
    bool create_succeeded = false;
    PFN_xrCreateInstance rt_xrCreateInstance;
    _get_instance_proc_addr(XR_NULL_HANDLE, "xrCreateInstance", reinterpret_cast<PFN_xrVoidFunction*>(&rt_xrCreateInstance));
    res = rt_xrCreateInstance(info, instance);
    if (XR_SUCCEEDED(res)) {
        create_succeeded = true;
        std::unique_ptr<XrGeneratedDispatchTableCore> dispatch_table(new XrGeneratedDispatchTableCore());
        GeneratedXrPopulateDispatchTableCore(dispatch_table.get(), *instance, _get_instance_proc_addr);
        std::lock_guard<std::mutex> mlock(_dispatch_table_mutex);
        _dispatch_table_map[*instance] = std::move(dispatch_table);
    }

    // If the failure occurred during the populate, clean up the instance we had picked up from the runtime
    if (XR_FAILED(res) && create_succeeded) {
        PFN_xrDestroyInstance rt_xrDestroyInstance;
        _get_instance_proc_addr(*instance, "xrDestroyInstance", reinterpret_cast<PFN_xrVoidFunction*>(&rt_xrDestroyInstance));
        rt_xrDestroyInstance(*instance);
        *instance = XR_NULL_HANDLE;
    }

    return res;
}

XrResult RuntimeInterface::DestroyInstance(XrInstance instance) {
    if (XR_NULL_HANDLE != instance) {
        // Destroy the dispatch table for this instance first
        {
            std::lock_guard<std::mutex> mlock(_dispatch_table_mutex);
            auto map_iter = _dispatch_table_map.find(instance);
            if (map_iter != _dispatch_table_map.end()) {
                _dispatch_table_map.erase(map_iter);
            }
        }
        // Now delete the instance
        PFN_xrDestroyInstance rt_xrDestroyInstance;
        _get_instance_proc_addr(instance, "xrDestroyInstance", reinterpret_cast<PFN_xrVoidFunction*>(&rt_xrDestroyInstance));
        rt_xrDestroyInstance(instance);
    }
    return XR_SUCCESS;
}

bool RuntimeInterface::TrackDebugMessenger(XrInstance instance, XrDebugUtilsMessengerEXT messenger) {
    std::lock_guard<std::mutex> mlock(_messenger_to_instance_mutex);
    _messenger_to_instance_map[messenger] = instance;
    return true;
}

void RuntimeInterface::ForgetDebugMessenger(XrDebugUtilsMessengerEXT messenger) {
    if (XR_NULL_HANDLE != messenger) {
        std::lock_guard<std::mutex> mlock(_messenger_to_instance_mutex);
        _messenger_to_instance_map.erase(messenger);
    }
}

void RuntimeInterface::SetSupportedExtensions(std::vector<std::string>& supported_extensions) {
    _supported_extensions = supported_extensions;
}

bool RuntimeInterface::SupportsExtension(const std::string& extension_name) {
    bool found_prop = false;
    for (const std::string& supported_extension : _supported_extensions) {
        if (supported_extension == extension_name) {
            found_prop = true;
            break;
        }
    }
    return found_prop;
}
