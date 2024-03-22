// Copyright (c) 2017-2024, The Khronos Group Inc.
// Copyright (c) 2017-2019 Valve Corporation
// Copyright (c) 2017-2019 LunarG, Inc.
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
// Initial Author: Mark Young <marky@lunarg.com>
//

#include "loader_init_data.hpp"

#ifdef XR_KHR_LOADER_INIT_SUPPORT

// Check and copy the Android-specific init data.
XrResult LoaderInitData::initialize(const XrLoaderInitInfoBaseHeaderKHR* info) {
#if defined(XR_USE_PLATFORM_ANDROID)
    if (info->type != XR_TYPE_LOADER_INIT_INFO_ANDROID_KHR) {
        return XR_ERROR_VALIDATION_FAILURE;
    }
    auto cast_info = reinterpret_cast<XrLoaderInitInfoAndroidKHR const*>(info);

    if (cast_info->applicationVM == nullptr) {
        return XR_ERROR_VALIDATION_FAILURE;
    }
    if (cast_info->applicationContext == nullptr) {
        return XR_ERROR_VALIDATION_FAILURE;
    }

    // Copy and store the JVM pointer and Android Context, ensuring the JVM is initialised.
    _data = *cast_info;
    _data.next = nullptr;
    jni::init(static_cast<jni::JavaVM*>(_data.applicationVM));
    const jni::Object context = jni::Object{static_cast<jni::jobject>(_data.applicationContext)};

    // Retrieve a reference to the Android AssetManager.
    const auto assetManager = context.call<jni::Object>("getAssets()Landroid/content/res/AssetManager;");
    _android_asset_manager = AAssetManager_fromJava(jni::env(), assetManager.getHandle());

    // Retrieve the path to the native libraries.
    const auto applicationContext = context.call<jni::Object>("getApplicationContext()Landroid/content/Context;");
    const auto applicationInfo = context.call<jni::Object>("getApplicationInfo()Landroid/content/pm/ApplicationInfo;");
    _native_library_path = applicationInfo.get<std::string>("nativeLibraryDir");
#else
#error "Platform specific XR_KHR_loader_init structure is not defined for this platform."
#endif  // XR_USE_PLATFORM_ANDROID

    _initialized = true;
    return XR_SUCCESS;
}

XrResult InitializeLoaderInitData(const XrLoaderInitInfoBaseHeaderKHR* loaderInitInfo) {
    return LoaderInitData::instance().initialize(loaderInitInfo);
}

#ifdef XR_USE_PLATFORM_ANDROID
std::string GetAndroidNativeLibraryDir() { return LoaderInitData::instance()._native_library_path; }

void* Android_Get_Asset_Manager() { return LoaderInitData::instance()._android_asset_manager; }
#endif  // XR_USE_PLATFORM_ANDROID

#endif  // XR_KHR_LOADER_INIT_SUPPORT
