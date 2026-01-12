// Copyright (c) 2017-2025 The Khronos Group Inc.
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
//

#include "loader_properties.hpp"
#include <platform_utils.hpp>

#include <string>
#include <unordered_map>
#include <mutex>

namespace {

std::mutex& GetOverridePropertiesMutex() {
    static std::mutex override_properties_mutex;
    return override_properties_mutex;
}

std::unordered_map<std::string, std::string>& GetOverrideProperties() {
    static std::unordered_map<std::string, std::string> override_properties;
    return override_properties;
}

const std::string* TryGetPropertyOverride(const std::string& name) {
    const auto& overrideProperties = GetOverrideProperties();
    const auto& overrideProperty = overrideProperties.find(name);
    if (overrideProperty != overrideProperties.end()) {
        return &overrideProperty->second;
    }
    return nullptr;
}

}  // namespace

// Loader property overrides take precedence over system environment variables because environment variables are not always
// safe to use (and thus would be ignored). For example, override properties may be the only way to redirect XR_RUNTIME_JSON
// from an elevated process on Windows.

namespace LoaderProperty {

std::string Get(const std::string& name) {
    std::lock_guard<std::mutex> lock(GetOverridePropertiesMutex());
    const std::string* propertyOverride = TryGetPropertyOverride(name);
    if (propertyOverride != nullptr) {
        return *propertyOverride;
    } else {
        return PlatformUtilsGetEnv(name.c_str());
    }
}

std::string GetSecure(const std::string& name) {
    std::lock_guard<std::mutex> lock(GetOverridePropertiesMutex());
    const std::string* propertyOverride = TryGetPropertyOverride(name);
    if (propertyOverride != nullptr) {
        return *propertyOverride;
    } else {
        return PlatformUtilsGetSecureEnv(name.c_str());
    }
}

bool IsSet(const std::string& name) {
    std::lock_guard<std::mutex> lock(GetOverridePropertiesMutex());
    const std::string* propertyOverride = TryGetPropertyOverride(name);
    return propertyOverride != nullptr || PlatformUtilsGetEnvSet(name.c_str());
}

void SetOverride(std::string name, std::string value) {
    std::lock_guard<std::mutex> lock(GetOverridePropertiesMutex());
    auto& overrideProperties = GetOverrideProperties();
    overrideProperties.insert(std::make_pair(std::move(name), std::move(value)));
}

void ClearOverrides() {
    std::lock_guard<std::mutex> lock(GetOverridePropertiesMutex());
    auto& overrideProperties = GetOverrideProperties();
    overrideProperties.clear();
}

}  // namespace LoaderProperty
