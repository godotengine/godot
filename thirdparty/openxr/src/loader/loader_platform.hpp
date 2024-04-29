// Copyright (c) 2017-2023, The Khronos Group Inc.
// Copyright (c) 2017-2019 Valve Corporation
// Copyright (c) 2017-2019 LunarG, Inc.
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
// Initial Authors: Mark Young <marky@lunarg.com>, Dave Houlton <daveh@lunarg.com>
//

#pragma once

#include <cassert>
#include <sstream>
#include <string>

#include "xr_dependencies.h"
#include "platform_utils.hpp"

#if defined(__GNUC__) && __GNUC__ >= 4
#define LOADER_EXPORT __attribute__((visibility("default")))
#elif defined(__SUNPRO_C) && (__SUNPRO_C >= 0x590)
#define LOADER_EXPORT __attribute__((visibility("default")))
#else
#define LOADER_EXPORT
#endif

// Environment variables
#if defined(XR_OS_LINUX) || defined(XR_OS_APPLE) || defined(XR_OS_ANDROID)

#include <sys/types.h>
#include <sys/stat.h>
#include <dlfcn.h>
#include <unistd.h>
#include <stdlib.h>
#include <dirent.h>

#ifndef PATH_MAX
#define PATH_MAX 1024
#endif

#define PATH_SEPARATOR ':'
#define DIRECTORY_SYMBOL '/'

// Dynamic Loading of libraries:
typedef void *LoaderPlatformLibraryHandle;
static inline LoaderPlatformLibraryHandle LoaderPlatformLibraryOpen(const std::string &path) {
    // When loading the library, we use RTLD_LAZY so that not all symbols have to be
    // resolved at this time (which improves performance). Note that if not all symbols
    // can be resolved, this could cause crashes later.
    // For experimenting/debugging: Define the LD_BIND_NOW environment variable to force all
    // symbols to be resolved here.
    return dlopen(path.c_str(), RTLD_LAZY | RTLD_LOCAL);
}

static inline const char *LoaderPlatformLibraryOpenError(const std::string &path) {
    (void)path;
    return dlerror();
}

static inline void LoaderPlatformLibraryClose(LoaderPlatformLibraryHandle library) { dlclose(library); }

static inline void *LoaderPlatformLibraryGetProcAddr(LoaderPlatformLibraryHandle library, const std::string &name) {
    assert(library);
    assert(!name.empty());
    return dlsym(library, name.c_str());
}

static inline const char *LoaderPlatformLibraryGetProcAddrError(const std::string &name) {
    (void)name;
    return dlerror();
}

#elif defined(XR_OS_WINDOWS)

#define PATH_SEPARATOR ';'
#define DIRECTORY_SYMBOL '\\'

// Workaround for MS VS 2010/2013 missing snprintf and vsnprintf
#if defined(_MSC_VER) && _MSC_VER < 1900
#include <stdint.h>

static inline int32_t xr_vsnprintf(char *result_buffer, size_t buffer_size, const char *print_format, va_list varying_list) {
    int32_t copy_count = -1;
    if (buffer_size != 0) {
        copy_count = _vsnprintf_s(result_buffer, buffer_size, _TRUNCATE, print_format, varying_list);
    }
    if (copy_count == -1) {
        copy_count = _vscprintf(print_format, varying_list);
    }
    return copy_count;
}

static inline int32_t xr_snprintf(char *result_buffer, size_t buffer_size, const char *print_format, ...) {
    va_list varying_list;
    va_start(varying_list, print_format);
    int32_t copy_count = xr_vsnprintf(result_buffer, buffer_size, print_format, varying_list);
    va_end(varying_list);
    return copy_count;
}

#define snprintf xr_snprintf
#define vsnprintf xr_vsnprintf

#endif

static inline std::string DescribeError(uint32_t code, bool prefixErrorCode = true) {
    std::string str;

    if (prefixErrorCode) {
        char prefixBuffer[64];
        snprintf(prefixBuffer, sizeof(prefixBuffer), "0x%llx (%lld): ", (uint64_t)code, (int64_t)code);
        str = prefixBuffer;
    }

    // Could use FORMAT_MESSAGE_FROM_HMODULE to specify an error source.
    WCHAR errorBufferW[1024]{};
    const DWORD errorBufferWCapacity = sizeof(errorBufferW) / sizeof(errorBufferW[0]);
    const DWORD length = FormatMessageW(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, nullptr, (DWORD)code,
                                        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), errorBufferW, errorBufferWCapacity, nullptr);

    if (length) {  // If errorBufferW contains what we are looking for...
        str += wide_to_utf8(errorBufferW);
    } else {
        str = "(unknown)";
    }

    return str;
}

// Dynamic Loading:
typedef HMODULE LoaderPlatformLibraryHandle;
static inline LoaderPlatformLibraryHandle LoaderPlatformLibraryOpen(const std::string &path) {
    const std::wstring pathW = utf8_to_wide(path);

    // Try loading the library the original way first.
    LoaderPlatformLibraryHandle handle = LoadLibraryW(pathW.c_str());

    if (handle == NULL && GetLastError() == ERROR_MOD_NOT_FOUND) {
        const DWORD dwAttrib = GetFileAttributesW(pathW.c_str());
        const bool fileExists = (dwAttrib != INVALID_FILE_ATTRIBUTES && !(dwAttrib & FILE_ATTRIBUTE_DIRECTORY));
        if (fileExists) {
            // If that failed, then try loading it with broader search folders.
            handle = LoadLibraryExW(pathW.c_str(), NULL, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR);
        }
    }

    return handle;
}

static inline std::string LoaderPlatformLibraryOpenError(const std::string &path) {
    std::stringstream ss;
    const DWORD dwLastError = GetLastError();
    const std::string strError = DescribeError(dwLastError);
    ss << "Failed to open dynamic library " << path << " with error " << dwLastError << ": " << strError;
    return ss.str();
}

static inline void LoaderPlatformLibraryClose(LoaderPlatformLibraryHandle library) { FreeLibrary(library); }

static inline void *LoaderPlatformLibraryGetProcAddr(LoaderPlatformLibraryHandle library, const std::string &name) {
    assert(library);
    assert(name.size() > 0);
    return reinterpret_cast<void *>(GetProcAddress(library, name.c_str()));
}

static inline std::string LoaderPlatformLibraryGetProcAddrAddrError(const std::string &name) {
    std::stringstream ss;
    ss << "Failed to find function " << name << " in dynamic library";
    return ss.str();
}

#else  // Not Linux or Windows

#define PATH_SEPARATOR ':'
#define DIRECTORY_SYMBOL '/'

static inline LoaderPlatformLibraryHandle LoaderPlatformLibraryOpen(const std::string &path) {
// Stub func
#error("Unknown platform, undefined dynamic library routines resulting");
    (void)path;
}

static inline const char *LoaderPlatformLibraryOpenError(const std::string &path) {
    // Stub func
    (void)path;
}

static inline void LoaderPlatformLibraryClose(LoaderPlatformLibraryHandle library) {
    // Stub func
    (void)library;
}

static inline void *LoaderPlatformLibraryGetProcAddr(LoaderPlatformLibraryHandle library, const std::string &name) {
    // Stub func
    void(library);
    void(name);
}

static inline const char *LoaderPlatformLibraryGetProcAddrError(const std::string &name) {
    // Stub func
    (void)name;
}

#endif
