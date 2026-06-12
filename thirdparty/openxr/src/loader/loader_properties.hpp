// Copyright (c) 2017-2025 The Khronos Group Inc.
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
//

#pragma once

#include <string>

// Exposes a centralized way to read properties which may be passed to the loader through xrInitializeLoaderKHR or available through
// environment variables.
namespace LoaderProperty {
std::string Get(const std::string& name);
std::string GetSecure(const std::string& name);
bool IsSet(const std::string& name);
void SetOverride(std::string name, std::string value);
void ClearOverrides();
}  // namespace LoaderProperty
