// Copyright (c) 2017-2024, The Khronos Group Inc.
// Copyright (c) 2017-2019 Valve Corporation
// Copyright (c) 2017-2019 LunarG, Inc.
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
// Initial Authors: Mark Young <marky@lunarg.com>, Dave Houlton <daveh@lunarg.com>
//

#pragma once

#include "xr_dependencies.h"
#include <string>
#include <stdint.h>
#include <stdlib.h>

// OpenXR paths and registry key locations
#define OPENXR_RELATIVE_PATH "openxr/"
#define OPENXR_IMPLICIT_API_LAYER_RELATIVE_PATH "/api_layers/implicit.d"
#define OPENXR_EXPLICIT_API_LAYER_RELATIVE_PATH "/api_layers/explicit.d"
#ifdef XR_OS_WINDOWS
#define OPENXR_REGISTRY_LOCATION "SOFTWARE\\Khronos\\OpenXR\\"
#define OPENXR_IMPLICIT_API_LAYER_REGISTRY_LOCATION "\\ApiLayers\\Implicit"
#define OPENXR_EXPLICIT_API_LAYER_REGISTRY_LOCATION "\\ApiLayers\\Explicit"
#endif

// OpenXR Loader environment variables of interest
#define OPENXR_RUNTIME_JSON_ENV_VAR "XR_RUNTIME_JSON"
#define OPENXR_API_LAYER_PATH_ENV_VAR "XR_API_LAYER_PATH"

// This is a CMake generated file with #defines for any functions/includes
// that it found present and build-time configuration.
// If you don't have this file, on non-Windows you'll need to define
// one of HAVE_SECURE_GETENV or HAVE___SECURE_GETENV depending on which
// of secure_getenv or __secure_getenv are present
#ifdef OPENXR_HAVE_COMMON_CONFIG
#include "common_config.h"
#endif  // OPENXR_HAVE_COMMON_CONFIG

#if defined(__x86_64__) && defined(__ILP32__)
#define XR_ARCH_ABI "x32"
#elif defined(_M_X64) || defined(__x86_64__)
#define XR_ARCH_ABI "x86_64"
#elif defined(_M_IX86) || defined(__i386__) || defined(_X86_)
#define XR_ARCH_ABI "i686"
#elif (defined(__aarch64__) && defined(__LP64__)) || defined(_M_ARM64)
#define XR_ARCH_ABI "aarch64"
#elif (defined(__ARM_ARCH) && __ARM_ARCH >= 7 && (defined(__ARM_PCS_VFP) || defined(__ANDROID__))) || defined(_M_ARM)
#define XR_ARCH_ABI "armv7a-vfp"
#elif defined(__ARM_ARCH_5TE__) || (defined(__ARM_ARCH) && __ARM_ARCH > 5)
#define XR_ARCH_ABI "armv5te"
#elif defined(__mips64)
#define XR_ARCH_ABI "mips64"
#elif defined(__mips)
#define XR_ARCH_ABI "mips"
#elif defined(__powerpc64__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#define XR_ARCH_ABI "ppc64"
#elif defined(__powerpc__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
#define XR_ARCH_ABI "ppc64el"
#elif defined(__s390x__) || defined(__zarch__)
#define XR_ARCH_ABI "s390x"
#elif defined(__hppa__)
#define XR_ARCH_ABI "hppa"
#elif defined(__alpha__)
#define XR_ARCH_ABI "alpha"
#elif defined(__ia64__) || defined(_M_IA64)
#define XR_ARCH_ABI "ia64"
#elif defined(__m68k__)
#define XR_ARCH_ABI "m68k"
#elif defined(__riscv_xlen) && (__riscv_xlen == 64)
#define XR_ARCH_ABI "riscv64"
#elif defined(__sparc__) && defined(__arch64__)
#define XR_ARCH_ABI "sparc64"
#else
#error "No architecture string known!"
#endif

// Consumers of this file must ensure this function is implemented. For example, the loader will implement this function so that it
// can route messages through the loader's logging system.
void LogPlatformUtilsError(const std::string& message);

// Environment variables
#if defined(XR_OS_LINUX) || defined(XR_OS_APPLE)

#include <unistd.h>
#include <fcntl.h>
#include <iostream>
#include <sys/stat.h>

namespace detail {

static inline char* ImplGetEnv(const char* name) { return getenv(name); }

static inline char* ImplGetSecureEnv(const char* name) {
#ifdef HAVE_SECURE_GETENV
    return secure_getenv(name);
#elif defined(HAVE___SECURE_GETENV)
    return __secure_getenv(name);
#else
// clang-format off
#pragma message(                                                    \
    "Warning:  Falling back to non-secure getenv for environmental" \
    "lookups!  Consider updating to a different libc.")
    // clang-format on

    return ImplGetEnv(name);
#endif
}
}  // namespace detail

#endif  // defined(XR_OS_LINUX) || defined(XR_OS_APPLE)

#if defined(XR_OS_ANDROID) || defined(XR_OS_APPLE)

#include <sys/stat.h>

namespace detail {

static inline bool ImplTryRuntimeFilename(const char* rt_dir_prefix, uint16_t major_version, std::string& file_name) {
    auto decorated_path = rt_dir_prefix + std::to_string(major_version) + "/active_runtime." XR_ARCH_ABI ".json";
    auto undecorated_path = rt_dir_prefix + std::to_string(major_version) + "/active_runtime.json";

    struct stat buf {};
    if (0 == stat(decorated_path.c_str(), &buf)) {
        file_name = decorated_path;
        return true;
    }
    if (0 == stat(undecorated_path.c_str(), &buf)) {
        file_name = undecorated_path;
        return true;
    }
    return false;
}

}  // namespace detail
#endif  // defined(XR_OS_ANDROID) || defined(XR_OS_APPLE)
#if defined(XR_OS_LINUX)

static inline std::string PlatformUtilsGetEnv(const char* name) {
    auto str = detail::ImplGetEnv(name);
    if (str == nullptr) {
        return {};
    }
    return str;
}

static inline std::string PlatformUtilsGetSecureEnv(const char* name) {
    auto str = detail::ImplGetSecureEnv(name);
    if (str == nullptr) {
        str = detail::ImplGetEnv(name);
        if (str != nullptr && !std::string(str).empty()) {
            LogPlatformUtilsError(std::string("!!! WARNING !!! Environment variable ") + name +
                                  " is being ignored due to running with secure execution. The value '" + str +
                                  "' will NOT be used.");
        }
        return {};
    }
    return str;
}

static inline bool PlatformUtilsGetEnvSet(const char* name) { return detail::ImplGetEnv(name) != nullptr; }

#elif defined(XR_OS_APPLE)

static inline std::string PlatformUtilsGetEnv(const char* name) {
    auto str = detail::ImplGetEnv(name);
    if (str == nullptr) {
        return {};
    }
    return str;
}

static inline std::string PlatformUtilsGetSecureEnv(const char* name) {
    auto str = detail::ImplGetSecureEnv(name);
    if (str == nullptr) {
        return {};
    }
    return str;
}

static inline bool PlatformUtilsGetEnvSet(const char* name) { return detail::ImplGetEnv(name) != nullptr; }

static inline bool PlatformGetGlobalRuntimeFileName(uint16_t major_version, std::string& file_name) {
    return detail::ImplTryRuntimeFilename("/usr/local/share/openxr/", major_version, file_name);
}

#elif defined(XR_OS_WINDOWS)

inline std::wstring utf8_to_wide(const std::string& utf8Text) {
    if (utf8Text.empty()) {
        return {};
    }

    std::wstring wideText;
    const int wideLength = ::MultiByteToWideChar(CP_UTF8, 0, utf8Text.data(), (int)utf8Text.size(), nullptr, 0);
    if (wideLength == 0) {
        LogPlatformUtilsError("utf8_to_wide get size error: " + std::to_string(::GetLastError()));
        return {};
    }

    // MultiByteToWideChar returns number of chars of the input buffer, regardless of null terminator
    wideText.resize(wideLength, 0);
    wchar_t* wideString = const_cast<wchar_t*>(wideText.data());  // mutable data() only exists in c++17
    const int length = ::MultiByteToWideChar(CP_UTF8, 0, utf8Text.data(), (int)utf8Text.size(), wideString, wideLength);
    if (length != wideLength) {
        LogPlatformUtilsError("utf8_to_wide convert string error: " + std::to_string(::GetLastError()));
        return {};
    }

    return wideText;
}

inline std::string wide_to_utf8(const std::wstring& wideText) {
    if (wideText.empty()) {
        return {};
    }

    std::string narrowText;
    int narrowLength = ::WideCharToMultiByte(CP_UTF8, 0, wideText.data(), (int)wideText.size(), nullptr, 0, nullptr, nullptr);
    if (narrowLength == 0) {
        LogPlatformUtilsError("wide_to_utf8 get size error: " + std::to_string(::GetLastError()));
        return {};
    }

    // WideCharToMultiByte returns number of chars of the input buffer, regardless of null terminator
    narrowText.resize(narrowLength, 0);
    char* narrowString = const_cast<char*>(narrowText.data());  // mutable data() only exists in c++17
    const int length =
        ::WideCharToMultiByte(CP_UTF8, 0, wideText.data(), (int)wideText.size(), narrowString, narrowLength, nullptr, nullptr);
    if (length != narrowLength) {
        LogPlatformUtilsError("wide_to_utf8 convert string error: " + std::to_string(::GetLastError()));
        return {};
    }

    return narrowText;
}

// Returns true if the current process has an integrity level > SECURITY_MANDATORY_MEDIUM_RID.
static inline bool IsHighIntegrityLevel() {
    // Execute this check once and save the value as a static bool.
    static bool isHighIntegrityLevel = ([] {
        HANDLE processToken;
        if (OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY | TOKEN_QUERY_SOURCE, &processToken)) {
            // Maximum possible size of SID_AND_ATTRIBUTES is maximum size of a SID + size of attributes DWORD.
            uint8_t mandatoryLabelBuffer[SECURITY_MAX_SID_SIZE + sizeof(DWORD)]{};
            DWORD bufferSize;
            if (GetTokenInformation(processToken, TokenIntegrityLevel, mandatoryLabelBuffer, sizeof(mandatoryLabelBuffer),
                                    &bufferSize) != 0) {
                const auto mandatoryLabel = reinterpret_cast<const TOKEN_MANDATORY_LABEL*>(mandatoryLabelBuffer);
                if (mandatoryLabel->Label.Sid != 0) {
                    const DWORD subAuthorityCount = *GetSidSubAuthorityCount(mandatoryLabel->Label.Sid);
                    const DWORD integrityLevel = *GetSidSubAuthority(mandatoryLabel->Label.Sid, subAuthorityCount - 1);
                    CloseHandle(processToken);
                    return integrityLevel > SECURITY_MANDATORY_MEDIUM_RID;
                }
            }

            CloseHandle(processToken);
        }

        return false;
    })();

    return isHighIntegrityLevel;
}

// Returns true if the given environment variable exists.
// The name is a case-sensitive UTF8 string.
static inline bool PlatformUtilsGetEnvSet(const char* name) {
    const std::wstring wname = utf8_to_wide(name);
    const DWORD valSize = ::GetEnvironmentVariableW(wname.c_str(), nullptr, 0);
    // GetEnvironmentVariable returns 0 when environment variable does not exist or there is an error.
    return 0 != valSize;
}

// Returns the environment variable value for the given name.
// Returns an empty string if the environment variable doesn't exist or if it exists but is empty.
// Use PlatformUtilsGetEnvSet to tell if it exists.
// The name is a case-sensitive UTF8 string.
static inline std::string PlatformUtilsGetEnv(const char* name) {
    const std::wstring wname = utf8_to_wide(name);
    const DWORD valSize = ::GetEnvironmentVariableW(wname.c_str(), nullptr, 0);
    // GetEnvironmentVariable returns 0 when environment variable does not exist or there is an error.
    // The size includes the null-terminator, so a size of 1 is means the variable was explicitly set to empty.
    if (valSize == 0 || valSize == 1) {
        return {};
    }

    // GetEnvironmentVariable returns size including null terminator for "query size" call.
    std::wstring wValue(valSize, 0);
    wchar_t* wValueData = &wValue[0];

    // GetEnvironmentVariable returns string length, excluding null terminator for "get value"
    // call if there was enough capacity. Else it returns the required capacity (including null terminator).
    const DWORD length = ::GetEnvironmentVariableW(wname.c_str(), wValueData, (DWORD)wValue.size());
    if ((length == 0) || (length >= wValue.size())) {  // If error or the variable increased length between calls...
        LogPlatformUtilsError("GetEnvironmentVariable get value error: " + std::to_string(::GetLastError()));
        return {};
    }

    wValue.resize(length);  // Strip the null terminator.

    return wide_to_utf8(wValue);
}

// Acts the same as PlatformUtilsGetEnv except returns an empty string if IsHighIntegrityLevel.
static inline std::string PlatformUtilsGetSecureEnv(const char* name) {
    // No secure version for Windows so the below integrity check is needed.
    const std::string envValue = PlatformUtilsGetEnv(name);

    // Do not allow high integrity processes to act on data that can be controlled by medium integrity processes.
    // Specifically, medium integrity processes can set environment variables which could then
    // be read by high integrity processes.
    if (IsHighIntegrityLevel()) {
        if (!envValue.empty()) {
            LogPlatformUtilsError(std::string("!!! WARNING !!! Environment variable ") + name +
                                  " is being ignored due to running from an elevated context. The value '" + envValue +
                                  "' will NOT be used.");
        }
        return {};
    }

    return envValue;
}

#elif defined(XR_OS_ANDROID)

#include <sys/system_properties.h>

static inline bool PlatformUtilsGetEnvSet(const char* /* name */) {
    // Stub func
    return false;
}

static inline std::string PlatformUtilsGetEnv(const char* /* name */) {
    // Stub func
    return {};
}

static inline std::string PlatformUtilsGetSecureEnv(const char* /* name */) {
    // Stub func
    return {};
}

// Intended to be only used as a fallback on Android, with a more open, "native" technique used in most cases
static inline bool PlatformGetGlobalRuntimeFileName(uint16_t major_version, std::string& file_name) {
    // Prefix for the runtime JSON file name
    static const char* rt_dir_prefixes[] = {"/product", "/odm", "/oem", "/vendor", "/system"};

    static const std::string subdir = "/etc/openxr/";
    for (const auto prefix : rt_dir_prefixes) {
        const std::string rt_dir_prefix = prefix + subdir;
        if (detail::ImplTryRuntimeFilename(rt_dir_prefix.c_str(), major_version, file_name)) {
            return true;
        }
    }

    return false;
}

// Android system properties are sufficiently different from environment variables that we are not re-using
// PlatformUtilsGetEnv for this purpose
static inline std::string PlatformUtilsGetAndroidSystemProperty(const char* name) {
    std::string result;
    const prop_info* pi = __system_property_find(name);
    if (pi == nullptr) {
        return {};
    }

#if __ANDROID_API__ >= 26
    // use callback to retrieve > 92 character sys prop values, if available
    __system_property_read_callback(
        pi,
        [](void* cookie, const char*, const char* value, uint32_t) {
            auto property_value = reinterpret_cast<std::string*>(cookie);
            *property_value = value;
        },
        reinterpret_cast<void*>(&result));
#endif  // __ANDROID_API__ >= 26
    // fallback to __system_property_get if no value retrieved via callback
    if (result.empty()) {
        char value[PROP_VALUE_MAX] = {};
        if (__system_property_get(name, value) != 0) {
            result = value;
        }
    }

    return result;
}

#else  // Not Linux, Apple, nor Windows

static inline bool PlatformUtilsGetEnvSet(const char* /* name */) {
    // Stub func
    return false;
}

static inline std::string PlatformUtilsGetEnv(const char* /* name */) {
    // Stub func
    return {};
}

static inline std::string PlatformUtilsGetSecureEnv(const char* /* name */) {
    // Stub func
    return {};
}

static inline bool PlatformGetGlobalRuntimeFileName(uint16_t /* major_version */, std::string const& /* file_name */) {
    // Stub func
    return false;
}

#endif
