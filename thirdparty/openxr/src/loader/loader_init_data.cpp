// Copyright (c) 2017-2025 The Khronos Group Inc.
// Copyright (c) 2017-2019 Valve Corporation
// Copyright (c) 2017-2019 LunarG, Inc.
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
// Initial Author: Mark Young <marky@lunarg.com>
//

#include "loader_logger.hpp"
#include "runtime_interface.hpp"
#include "loader_instance.hpp"
#include "loader_init_data.hpp"
#include "loader_properties.hpp"

XrResult LoaderInitData::initialize(const XrLoaderInitInfoBaseHeaderKHR* info) {
    // We iterate the chain per struct type, so we only pick the first of each type in the chain.

    XrResult result = initializeProperties(info);
    if (result != XR_SUCCESS) {
        return result;
    }

#if defined(XR_HAS_REQUIRED_PLATFORM_LOADER_INIT_STRUCT)
    result = initializePlatform(info);
    if (result != XR_SUCCESS) {
        return result;
    }
#endif  // defined(XR_HAS_REQUIRED_PLATFORM_LOADER_INIT_STRUCT)

    _initialized = true;
    return XR_SUCCESS;
}

XrResult LoaderInitData::initializeProperties(const XrLoaderInitInfoBaseHeaderKHR* info) {
    while (info != nullptr) {
        if (info->type == XR_TYPE_LOADER_INIT_INFO_PROPERTIES_EXT) {
            const auto* propertyInfo = reinterpret_cast<XrLoaderInitInfoPropertiesEXT const*>(info);

            // Validate the inputs first.
            for (uint32_t i = 0; i < propertyInfo->propertyValueCount; i++) {
                if (propertyInfo->propertyValues[i].name == nullptr) {
                    return XR_ERROR_VALIDATION_FAILURE;
                }
                if (propertyInfo->propertyValues[i].value == nullptr) {
                    return XR_ERROR_VALIDATION_FAILURE;
                }
                std::string view{propertyInfo->propertyValues[i].name};
                if (view.size() == 0) {
                    return XR_ERROR_VALIDATION_FAILURE;
                }
            }

            // Inject provided properties into the loader property store.
            LoaderProperty::ClearOverrides();
            for (uint32_t i = 0; i < propertyInfo->propertyValueCount; i++) {
                LoaderProperty::SetOverride(propertyInfo->propertyValues[i].name, propertyInfo->propertyValues[i].value);
            }
            // Take only the first such struct.
            return XR_SUCCESS;
        }
        info = reinterpret_cast<const XrLoaderInitInfoBaseHeaderKHR*>(info->next);
    }

    // fine if we don't find this.
    return XR_SUCCESS;
}

#if defined(XR_USE_PLATFORM_ANDROID) && defined(XR_HAS_REQUIRED_PLATFORM_LOADER_INIT_STRUCT)
XrResult LoaderInitData::initializePlatform(const XrLoaderInitInfoBaseHeaderKHR* info) {
    // Check and copy the Android-specific init data.
    while (info != nullptr) {
        if (info->type == XR_TYPE_LOADER_INIT_INFO_ANDROID_KHR) {
            auto cast_info = reinterpret_cast<XrLoaderInitInfoAndroidKHR const*>(info);

            if (cast_info->applicationVM == nullptr) {
                return XR_ERROR_VALIDATION_FAILURE;
            }
            if (cast_info->applicationContext == nullptr) {
                return XR_ERROR_VALIDATION_FAILURE;
            }

            // Copy and store the JVM pointer and Android Context, ensuring the JVM is initialised.
            _platform_info = *cast_info;
            _platform_info.next = nullptr;  // Not safe to store next pointer since the memory may not exist later.

            if (_platform_info.applicationVM == nullptr) {
                return XR_ERROR_VALIDATION_FAILURE;
            }
            if (_platform_info.applicationContext == nullptr) {
                return XR_ERROR_VALIDATION_FAILURE;
            }
            jni::init(static_cast<jni::JavaVM*>(_platform_info.applicationVM));
            const jni::Object context = jni::Object{static_cast<jni::jobject>(_platform_info.applicationContext)};

            // Retrieve a reference to the Android AssetManager.
            const auto assetManager = context.call<jni::Object>("getAssets()Landroid/content/res/AssetManager;");
            _android_asset_manager = AAssetManager_fromJava(jni::env(), assetManager.getHandle());

            // Retrieve the path to the native libraries.
            const auto applicationContext = context.call<jni::Object>("getApplicationContext()Landroid/content/Context;");
            const auto applicationInfo = context.call<jni::Object>("getApplicationInfo()Landroid/content/pm/ApplicationInfo;");
            _android_native_library_path = applicationInfo.get<std::string>("nativeLibraryDir");

            // Take only the first such struct.
            return XR_SUCCESS;
        }
        info = reinterpret_cast<const XrLoaderInitInfoBaseHeaderKHR*>(info->next);
    }

    // We didn't find one.
    return XR_ERROR_VALIDATION_FAILURE;
}
#endif  // defined(XR_USE_PLATFORM_ANDROID) && defined(XR_HAS_REQUIRED_PLATFORM_LOADER_INIT_STRUCT)

XrResult InitializeLoaderInitData(const XrLoaderInitInfoBaseHeaderKHR* loaderInitInfo) {
    if (!ActiveLoaderInstance::IsAvailable()) {
        LoaderLogger::LogVerboseMessage("InitializeLoaderInitData", "Unloading any previously loaded runtime");
        // This will not shutdown the runtime, only unload the library.
        RuntimeInterface::UnloadRuntime("InitializeLoaderInitData");
    } else {
        LoaderLogger::LogErrorMessage("InitializeLoaderInitData",
                                      "An active instance currently exists while trying to reinitialize the loader");
        return XR_ERROR_INITIALIZATION_FAILED;
    }
    return LoaderInitData::instance().initialize(loaderInitInfo);
}

#if defined(XR_USE_PLATFORM_ANDROID) && defined(XR_HAS_REQUIRED_PLATFORM_LOADER_INIT_STRUCT)
std::string GetAndroidNativeLibraryDir() { return LoaderInitData::instance()._android_native_library_path; }

void* GetAndroidAssetManager() { return LoaderInitData::instance()._android_asset_manager; }
#endif  // defined(XR_USE_PLATFORM_ANDROID) && defined(XR_HAS_REQUIRED_PLATFORM_LOADER_INIT_STRUCT)
