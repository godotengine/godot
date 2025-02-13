// Copyright (c) 2017-2024, The Khronos Group Inc.
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
#pragma once

#ifdef XR_USE_PLATFORM_ANDROID

#include <memory>
#include <android/asset_manager.h>

namespace deleters {
struct AAssetDeleter {
    void operator()(AAsset* asset) const noexcept {
        if (asset != nullptr) {
            AAsset_close(asset);
        }
    }
};

struct AAssetDirDeleter {
    void operator()(AAssetDir* dir) const noexcept {
        if (dir != nullptr) {
            AAssetDir_close(dir);
        }
    }
};

}  // namespace deleters

using UniqueAsset = std::unique_ptr<AAsset, deleters::AAssetDeleter>;
using UniqueAssetDir = std::unique_ptr<AAssetDir, deleters::AAssetDirDeleter>;

#endif
