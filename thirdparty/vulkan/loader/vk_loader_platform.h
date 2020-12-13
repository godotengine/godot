/*
 *
 * Copyright (c) 2015-2018 The Khronos Group Inc.
 * Copyright (c) 2015-2018 Valve Corporation
 * Copyright (c) 2015-2018 LunarG, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Author: Ian Elliot <ian@lunarg.com>
 * Author: Jon Ashburn <jon@lunarg.com>
 * Author: Lenny Komow <lenny@lunarg.com>
 *
 */
#pragma once

#if defined(_WIN32)
// WinSock2.h must be included *BEFORE* windows.h
#include <winsock2.h>
#endif  // _WIN32

#include "vulkan/vk_platform.h"
#include "vulkan/vk_sdk_platform.h"

#if defined(__linux__) || defined(__APPLE__)
/* Linux-specific common code: */

// Headers:
//#ifndef _GNU_SOURCE
//#define _GNU_SOURCE 1
//#endif
#include <unistd.h>
// Note: The following file is for dynamic loading:
#include <dlfcn.h>
#include <pthread.h>
#include <assert.h>
#include <string.h>
#include <stdbool.h>
#include <stdlib.h>
#include <libgen.h>

// VK Library Filenames, Paths, etc.:
#define PATH_SEPARATOR ':'
#define DIRECTORY_SYMBOL '/'

#define VULKAN_DIR "vulkan/"
#define VULKAN_ICDCONF_DIR "icd.d"
#define VULKAN_ICD_DIR "icd"
#define VULKAN_SETTINGSCONF_DIR "settings.d"
#define VULKAN_ELAYERCONF_DIR "explicit_layer.d"
#define VULKAN_ILAYERCONF_DIR "implicit_layer.d"
#define VULKAN_LAYER_DIR "layer"

#define VK_DRIVERS_INFO_RELATIVE_DIR VULKAN_DIR VULKAN_ICDCONF_DIR
#define VK_SETTINGS_INFO_RELATIVE_DIR VULKAN_DIR VULKAN_SETTINGSCONF_DIR
#define VK_ELAYERS_INFO_RELATIVE_DIR VULKAN_DIR VULKAN_ELAYERCONF_DIR
#define VK_ILAYERS_INFO_RELATIVE_DIR VULKAN_DIR VULKAN_ILAYERCONF_DIR

#define VK_DRIVERS_INFO_REGISTRY_LOC ""
#define VK_SETTINGS_INFO_REGISTRY_LOC ""
#define VK_ELAYERS_INFO_REGISTRY_LOC ""
#define VK_ILAYERS_INFO_REGISTRY_LOC ""

#if !defined(DEFAULT_VK_LAYERS_PATH)
#define DEFAULT_VK_LAYERS_PATH ""
#endif
#if !defined(LAYERS_SOURCE_PATH)
#define LAYERS_SOURCE_PATH NULL
#endif
#define LAYERS_PATH_ENV "VK_LAYER_PATH"
#define ENABLED_LAYERS_ENV "VK_INSTANCE_LAYERS"

// C99:
#define PRINTF_SIZE_T_SPECIFIER "%zu"

// File IO
static inline bool loader_platform_file_exists(const char *path) {
    if (access(path, F_OK))
        return false;
    else
        return true;
}

static inline bool loader_platform_is_path_absolute(const char *path) {
    if (path[0] == '/')
        return true;
    else
        return false;
}

static inline char *loader_platform_dirname(char *path) { return dirname(path); }

#if defined(__linux__)

// find application path + name. Path cannot be longer than 1024, returns NULL if it is greater than that.
static inline char *loader_platform_executable_path(char *buffer, size_t size) {
    ssize_t count = readlink("/proc/self/exe", buffer, size);
    if (count == -1) return NULL;
    if (count == 0) return NULL;
    buffer[count] = '\0';
    return buffer;
}
#elif defined(__APPLE__)  // defined(__linux__)
#include <libproc.h>
static inline char *loader_platform_executable_path(char *buffer, size_t size) {
    pid_t pid = getpid();
    int ret = proc_pidpath(pid, buffer, size);
    if (ret <= 0) return NULL;
    buffer[ret] = '\0';
    return buffer;
}
#endif  // defined (__APPLE__)

// Compatability with compilers that don't support __has_feature
#ifndef __has_feature
#define __has_feature(x) 0
#endif

#if __has_feature(address_sanitizer) || defined(__SANITIZE_ADDRESS__)
#define LOADER_ADDRESS_SANITIZER
#endif

// Dynamic Loading of libraries:
typedef void *loader_platform_dl_handle;
static inline loader_platform_dl_handle loader_platform_open_library(const char *libPath) {
// When loading the library, we use RTLD_LAZY so that not all symbols have to be
// resolved at this time (which improves performance). Note that if not all symbols
// can be resolved, this could cause crashes later. Use the LD_BIND_NOW environment
// variable to force all symbols to be resolved here.
    return dlopen(libPath, RTLD_LAZY | RTLD_LOCAL);
}
static inline const char *loader_platform_open_library_error(const char *libPath) { return dlerror(); }
static inline void loader_platform_close_library(loader_platform_dl_handle library) { dlclose(library); }
static inline void *loader_platform_get_proc_address(loader_platform_dl_handle library, const char *name) {
    assert(library);
    assert(name);
    return dlsym(library, name);
}
static inline const char *loader_platform_get_proc_address_error(const char *name) { return dlerror(); }

// Threads:
typedef pthread_t loader_platform_thread;
#define THREAD_LOCAL_DECL __thread

// The once init functionality is not used on Linux
#define LOADER_PLATFORM_THREAD_ONCE_DECLARATION(var)
#define LOADER_PLATFORM_THREAD_ONCE_DEFINITION(var)
#define LOADER_PLATFORM_THREAD_ONCE(ctl, func)

// Thread IDs:
typedef pthread_t loader_platform_thread_id;
static inline loader_platform_thread_id loader_platform_get_thread_id() { return pthread_self(); }

// Thread mutex:
typedef pthread_mutex_t loader_platform_thread_mutex;
static inline void loader_platform_thread_create_mutex(loader_platform_thread_mutex *pMutex) { pthread_mutex_init(pMutex, NULL); }
static inline void loader_platform_thread_lock_mutex(loader_platform_thread_mutex *pMutex) { pthread_mutex_lock(pMutex); }
static inline void loader_platform_thread_unlock_mutex(loader_platform_thread_mutex *pMutex) { pthread_mutex_unlock(pMutex); }
static inline void loader_platform_thread_delete_mutex(loader_platform_thread_mutex *pMutex) { pthread_mutex_destroy(pMutex); }
typedef pthread_cond_t loader_platform_thread_cond;
static inline void loader_platform_thread_init_cond(loader_platform_thread_cond *pCond) { pthread_cond_init(pCond, NULL); }
static inline void loader_platform_thread_cond_wait(loader_platform_thread_cond *pCond, loader_platform_thread_mutex *pMutex) {
    pthread_cond_wait(pCond, pMutex);
}
static inline void loader_platform_thread_cond_broadcast(loader_platform_thread_cond *pCond) { pthread_cond_broadcast(pCond); }

#define loader_stack_alloc(size) alloca(size)

#elif defined(_WIN32)  // defined(__linux__)
/* Windows-specific common code: */
// WinBase.h defines CreateSemaphore and synchapi.h defines CreateEvent
//  undefine them to avoid conflicts with VkLayerDispatchTable struct members.
#ifdef CreateSemaphore
#undef CreateSemaphore
#endif
#ifdef CreateEvent
#undef CreateEvent
#endif
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <io.h>
#include <stdbool.h>
#include <shlwapi.h>
#include <direct.h>
#ifdef __cplusplus
#include <iostream>
#include <string>
#endif  // __cplusplus

// VK Library Filenames, Paths, etc.:
#define PATH_SEPARATOR ';'
#define DIRECTORY_SYMBOL '\\'
#define DEFAULT_VK_REGISTRY_HIVE HKEY_LOCAL_MACHINE
#define DEFAULT_VK_REGISTRY_HIVE_STR "HKEY_LOCAL_MACHINE"
#define SECONDARY_VK_REGISTRY_HIVE HKEY_CURRENT_USER
#define SECONDARY_VK_REGISTRY_HIVE_STR "HKEY_CURRENT_USER"

#define VK_DRIVERS_INFO_RELATIVE_DIR ""
#define VK_SETTINGS_INFO_RELATIVE_DIR ""
#define VK_ELAYERS_INFO_RELATIVE_DIR ""
#define VK_ILAYERS_INFO_RELATIVE_DIR ""

#ifdef _WIN64
#define HKR_VK_DRIVER_NAME API_NAME "DriverName"
#else
#define HKR_VK_DRIVER_NAME API_NAME "DriverNameWow"
#endif
#define VK_DRIVERS_INFO_REGISTRY_LOC "SOFTWARE\\Khronos\\" API_NAME "\\Drivers"
#define VK_SETTINGS_INFO_REGISTRY_LOC "SOFTWARE\\Khronos\\" API_NAME "\\Settings"
#define VK_ELAYERS_INFO_REGISTRY_LOC "SOFTWARE\\Khronos\\" API_NAME "\\ExplicitLayers"
#define VK_ILAYERS_INFO_REGISTRY_LOC "SOFTWARE\\Khronos\\" API_NAME "\\ImplicitLayers"

#if !defined(DEFAULT_VK_LAYERS_PATH)
#define DEFAULT_VK_LAYERS_PATH ""
#endif
#if !defined(LAYERS_SOURCE_PATH)
#define LAYERS_SOURCE_PATH NULL
#endif
#define LAYERS_PATH_ENV "VK_LAYER_PATH"
#define ENABLED_LAYERS_ENV "VK_INSTANCE_LAYERS"

#define PRINTF_SIZE_T_SPECIFIER "%Iu"

#if defined(_WIN32)
// Get the key for the plug n play driver registry
// The string returned by this function should NOT be freed
static inline const char *LoaderPnpDriverRegistry() {
    BOOL is_wow;
    IsWow64Process(GetCurrentProcess(), &is_wow);
    return is_wow ? "VulkanDriverNameWow" : "VulkanDriverName";
}
static inline const wchar_t *LoaderPnpDriverRegistryWide() {
    BOOL is_wow;
    IsWow64Process(GetCurrentProcess(), &is_wow);
    return is_wow ? L"VulkanDriverNameWow" : L"VulkanDriverName";
}

// Get the key for the plug 'n play explicit layer registry
// The string returned by this function should NOT be freed
static inline const char *LoaderPnpELayerRegistry() {
    BOOL is_wow;
    IsWow64Process(GetCurrentProcess(), &is_wow);
    return is_wow ? "VulkanExplicitLayersWow" : "VulkanExplicitLayers";
}
static inline const wchar_t *LoaderPnpELayerRegistryWide() {
    BOOL is_wow;
    IsWow64Process(GetCurrentProcess(), &is_wow);
    return is_wow ? L"VulkanExplicitLayersWow" : L"VulkanExplicitLayers";
}

// Get the key for the plug 'n play implicit layer registry
// The string returned by this function should NOT be freed
static inline const char *LoaderPnpILayerRegistry() {
    BOOL is_wow;
    IsWow64Process(GetCurrentProcess(), &is_wow);
    return is_wow ? "VulkanImplicitLayersWow" : "VulkanImplicitLayers";
}
static inline const wchar_t *LoaderPnpILayerRegistryWide() {
    BOOL is_wow;
    IsWow64Process(GetCurrentProcess(), &is_wow);
    return is_wow ? L"VulkanImplicitLayersWow" : L"VulkanImplicitLayers";
}
#endif

// File IO
static bool loader_platform_file_exists(const char *path) {
    if ((_access(path, 0)) == -1)
        return false;
    else
        return true;
}

static bool loader_platform_is_path_absolute(const char *path) {
    if (!path || !*path) {
        return false;
    }
    if (*path == DIRECTORY_SYMBOL || path[1] == ':') {
        return true;
    }
    return false;
}

// WIN32 runtime doesn't have dirname().
static inline char *loader_platform_dirname(char *path) {
    char *current, *next;

    // TODO/TBD: Do we need to deal with the Windows's ":" character?

    for (current = path; *current != '\0'; current = next) {
        next = strchr(current, DIRECTORY_SYMBOL);
        if (next == NULL) {
            if (current != path) *(current - 1) = '\0';
            return path;
        } else {
            // Point one character past the DIRECTORY_SYMBOL:
            next++;
        }
    }
    return path;
}

static inline char *loader_platform_executable_path(char *buffer, size_t size) {
    DWORD ret = GetModuleFileName(NULL, buffer, (DWORD)size);
    if (ret == 0) return NULL;
    if (ret > size) return NULL;
    buffer[ret] = '\0';
    return buffer;
}

// Dynamic Loading:
typedef HMODULE loader_platform_dl_handle;
static loader_platform_dl_handle loader_platform_open_library(const char *lib_path) {
    // Try loading the library the original way first.
    loader_platform_dl_handle lib_handle = LoadLibrary(lib_path);
    if (lib_handle == NULL && GetLastError() == ERROR_MOD_NOT_FOUND) {
        // If that failed, then try loading it with broader search folders.
        lib_handle = LoadLibraryEx(lib_path, NULL, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR);
    }
    return lib_handle;
}
static char *loader_platform_open_library_error(const char *libPath) {
    static char errorMsg[164];
    (void)snprintf(errorMsg, 163, "Failed to open dynamic library \"%s\" with error %lu", libPath, GetLastError());
    return errorMsg;
}
static void loader_platform_close_library(loader_platform_dl_handle library) { FreeLibrary(library); }
static void *loader_platform_get_proc_address(loader_platform_dl_handle library, const char *name) {
    assert(library);
    assert(name);
    return (void *)GetProcAddress(library, name);
}
static char *loader_platform_get_proc_address_error(const char *name) {
    static char errorMsg[120];
    (void)snprintf(errorMsg, 119, "Failed to find function \"%s\" in dynamic library", name);
    return errorMsg;
}

// Threads:
typedef HANDLE loader_platform_thread;

// __declspec(thread) is not supported by MinGW compiler (ignored with warning or
//                    cause error depending on compiler switches)
//
// __thread should be used instead
//
// __MINGW32__ defined for both 32 and 64 bit MinGW compilers, so it is enough to
// detect any (32 or 64) flavor of MinGW compiler.
//
// @note __GNUC__ could be used as a more generic way to detect _any_
//       GCC[-compatible] compiler on Windows, but this fix was tested
//       only with MinGW, so keep it explicit at the moment.
#if defined(__MINGW32__)
#define THREAD_LOCAL_DECL __thread
#else
#define THREAD_LOCAL_DECL __declspec(thread)
#endif

// The once init functionality is not used when building a DLL on Windows. This is because there is no way to clean up the
// resources allocated by anything allocated by once init. This isn't a problem for static libraries, but it is for dynamic
// ones. When building a DLL, we use DllMain() instead to allow properly cleaning up resources.
#if defined(LOADER_DYNAMIC_LIB)
#define LOADER_PLATFORM_THREAD_ONCE_DECLARATION(var)
#define LOADER_PLATFORM_THREAD_ONCE_DEFINITION(var)
#define LOADER_PLATFORM_THREAD_ONCE(ctl, func)
#else
#define LOADER_PLATFORM_THREAD_ONCE_DECLARATION(var) INIT_ONCE var = INIT_ONCE_STATIC_INIT;
#define LOADER_PLATFORM_THREAD_ONCE_DEFINITION(var) INIT_ONCE var;
#define LOADER_PLATFORM_THREAD_ONCE(ctl, func) loader_platform_thread_once_fn(ctl, func)
static BOOL CALLBACK InitFuncWrapper(PINIT_ONCE InitOnce, PVOID Parameter, PVOID *Context) {
    void (*func)(void) = (void (*)(void))Parameter;
    func();
    return TRUE;
}
static void loader_platform_thread_once_fn(void *ctl, void (*func)(void)) {
    assert(func != NULL);
    assert(ctl != NULL);
    InitOnceExecuteOnce((PINIT_ONCE)ctl, InitFuncWrapper, (void *)func, NULL);
}
#endif

// Thread IDs:
typedef DWORD loader_platform_thread_id;
static loader_platform_thread_id loader_platform_get_thread_id() { return GetCurrentThreadId(); }

// Thread mutex:
typedef CRITICAL_SECTION loader_platform_thread_mutex;
static void loader_platform_thread_create_mutex(loader_platform_thread_mutex *pMutex) { InitializeCriticalSection(pMutex); }
static void loader_platform_thread_lock_mutex(loader_platform_thread_mutex *pMutex) { EnterCriticalSection(pMutex); }
static void loader_platform_thread_unlock_mutex(loader_platform_thread_mutex *pMutex) { LeaveCriticalSection(pMutex); }
static void loader_platform_thread_delete_mutex(loader_platform_thread_mutex *pMutex) { DeleteCriticalSection(pMutex); }
typedef CONDITION_VARIABLE loader_platform_thread_cond;
static void loader_platform_thread_init_cond(loader_platform_thread_cond *pCond) { InitializeConditionVariable(pCond); }
static void loader_platform_thread_cond_wait(loader_platform_thread_cond *pCond, loader_platform_thread_mutex *pMutex) {
    SleepConditionVariableCS(pCond, pMutex, INFINITE);
}
static void loader_platform_thread_cond_broadcast(loader_platform_thread_cond *pCond) { WakeAllConditionVariable(pCond); }

#define loader_stack_alloc(size) _alloca(size)
#else  // defined(_WIN32)

#error The "loader_platform.h" file must be modified for this OS.

// NOTE: In order to support another OS, an #elif needs to be added (above the
// "#else // defined(_WIN32)") for that OS, and OS-specific versions of the
// contents of this file must be created.

// NOTE: Other OS-specific changes are also needed for this OS.  Search for
// files with "WIN32" in it, as a quick way to find files that must be changed.

#endif  // defined(_WIN32)

// returns true if the given string appears to be a relative or absolute
// path, as opposed to a bare filename.
static inline bool loader_platform_is_path(const char *path) { return strchr(path, DIRECTORY_SYMBOL) != NULL; }
