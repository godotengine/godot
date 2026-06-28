// Copyright (c) 2017-2025 The Khronos Group Inc.
// Copyright (c) 2017-2019 Valve Corporation
// Copyright (c) 2017-2019 LunarG, Inc.
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
// Initial Author: Mark Young <marky@lunarg.com>
//

#pragma once

#include <xr_dependencies.h>
#include <openxr/openxr_platform.h>

#if defined(XR_USE_PLATFORM_ANDROID)
#include <json/value.h>
#include <android/asset_manager_jni.h>
#include "android_utilities.h"

// Warning: For system software use only. Enabling this define in regular applications
// will make it incompatible with many conformant runtimes.
#if !defined(XR_DISABLE_LOADER_INIT_ANDROID_LOADER_KHR)
/*!
 * Later code will assume that if this is defined then a platform-specific struct
 * must be provided for successful initialization.
 */
#define XR_HAS_REQUIRED_PLATFORM_LOADER_INIT_STRUCT
#endif  // !defined(XR_DISABLE_LOADER_INIT_ANDROID_LOADER_KHR)
#endif  // defined(XR_USE_PLATFORM_ANDROID)

/*!
 * Stores a copy of the data passed to the xrInitializeLoaderKHR function in a singleton.
 */
class LoaderInitData {
   public:
    /*!
     * Singleton accessor.
     */
    static LoaderInitData& instance() {
        static LoaderInitData obj;
        return obj;
    }

#if defined(XR_USE_PLATFORM_ANDROID) && defined(XR_HAS_REQUIRED_PLATFORM_LOADER_INIT_STRUCT)
    /*!
     * Type alias for the platform-specific structure type.
     */
    using PlatformStructType = XrLoaderInitInfoAndroidKHR;

    /*!
     * Native library path.
     */
    std::string _android_native_library_path;
    /*!
     * Android asset manager.
     */
    AAssetManager* _android_asset_manager;
#endif

    /*!
     * Get our copy of the data, casted to pass to the runtime's matching method.
     * Only some platforms have platform-specific loader init info.
     */
    // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
    const XrLoaderInitInfoBaseHeaderKHR* getPlatformParam() {
#if defined(XR_HAS_REQUIRED_PLATFORM_LOADER_INIT_STRUCT)
        return reinterpret_cast<const XrLoaderInitInfoBaseHeaderKHR*>(&_platform_info);
#else
        return nullptr;
#endif
    }

#if defined(XR_HAS_REQUIRED_PLATFORM_LOADER_INIT_STRUCT)

    /*!
     * Get the data via its real structure type.
     */
    const PlatformStructType& getPlatformData() const { return _platform_info; }
#endif

    /*!
     * Has this been correctly initialized?
     */
    bool initialized() const noexcept { return _initialized; }

    /*!
     * Initialize loader data - called by InitializeLoaderInitData() and thus ultimately by the loader's xrInitializeLoaderKHR
     * implementation.
     */
    XrResult initialize(const XrLoaderInitInfoBaseHeaderKHR* info);

   private:
    //! Private constructor, forces use of singleton accessor.
    LoaderInitData() = default;

    /*!
     * Initialize non-platform-specific loader data.
     */
    // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
    XrResult initializeProperties(const XrLoaderInitInfoBaseHeaderKHR* info);

    /*!
     * Initialize platform-specific loader data - called ultimately by the loader's xrInitializeLoaderKHR
     * implementation. Each platform that needs this extension will provide an implementation of this.
     */
    XrResult initializePlatform(const XrLoaderInitInfoBaseHeaderKHR* info);

#if defined(XR_HAS_REQUIRED_PLATFORM_LOADER_INIT_STRUCT)
    /*!
     * Only some platforms have platform-specific loader init info.
     */
    PlatformStructType _platform_info;
#endif  // defined(XR_HAS_REQUIRED_PLATFORM_LOADER_INIT_STRUCT)

    //! Flag for indicating whether _platform_info is valid.
    bool _initialized = false;
};

//! Initialize loader init data, where required.
XrResult InitializeLoaderInitData(const XrLoaderInitInfoBaseHeaderKHR* loaderInitInfo);

#if defined(XR_USE_PLATFORM_ANDROID) && defined(XR_HAS_REQUIRED_PLATFORM_LOADER_INIT_STRUCT)
XrResult GetPlatformRuntimeVirtualManifest(Json::Value& out_manifest);
std::string GetAndroidNativeLibraryDir();
void* GetAndroidAssetManager();
#endif  // defined(XR_USE_PLATFORM_ANDROID) && defined(XR_HAS_REQUIRED_PLATFORM_LOADER_INIT_STRUCT)
