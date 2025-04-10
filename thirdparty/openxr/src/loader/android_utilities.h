// Copyright (c) 2020-2024, The Khronos Group Inc.
// Copyright (c) 2020-2021, Collabora, Ltd.
//
// SPDX-License-Identifier:  Apache-2.0 OR MIT
//
// Initial Author: Rylie Pavlik <rylie.pavlik@collabora.com>

#pragma once
#ifdef __ANDROID__

#include "wrap/android.content.h"

#include <string>
namespace Json {
class Value;
}  // namespace Json

namespace openxr_android {
using wrap::android::content::Context;

/*!
 * Find the single active OpenXR runtime on the system, and return a constructed JSON object representing it.
 *
 * @param context An Android context, preferably an Activity Context.
 * @param[out] virtualManifest The Json::Value to fill with the virtual manifest.
 *
 * @return 0 on success, something else on failure.
 */
int getActiveRuntimeVirtualManifest(wrap::android::content::Context const &context, Json::Value &virtualManifest);
}  // namespace openxr_android

#endif  // __ANDROID__
