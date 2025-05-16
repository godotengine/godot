/*
  Simple DirectMedia Layer
  Copyright (C) 1997-2025 Sam Lantinga <slouken@libsdl.org>

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.
*/
#include "SDL_internal.h"

#if defined(SDL_VIDEO_VULKAN) && defined(SDL_VIDEO_DRIVER_OFFSCREEN)

#include "../SDL_vulkan_internal.h"
#include "../SDL_sysvideo.h"


static const char *s_defaultPaths[] = {
#if defined(SDL_PLATFORM_WINDOWS)
    "vulkan-1.dll"
#elif defined(SDL_PLATFORM_APPLE)
    "vulkan.framework/vulkan",
    "libvulkan.1.dylib",
    "libvulkan.dylib",
    "MoltenVK.framework/MoltenVK",
    "libMoltenVK.dylib"
#elif defined(SDL_PLATFORM_OPENBSD)
    "libvulkan.so"
#else
    "libvulkan.so.1"
#endif
};

#if defined( SDL_PLATFORM_APPLE )
#include <dlfcn.h>

// Since libSDL is most likely a .dylib, need RTLD_DEFAULT not RTLD_SELF.
#define DEFAULT_HANDLE RTLD_DEFAULT
#endif

/*Should the whole driver fail if it can't create a surface? Rendering to an offscreen buffer is still possible without a surface.
    At the time of writing. I need the driver to minimally work even if the surface extension isn't present.
    And account for the inability to create a surface on the consumer side.
    So for now I'm targeting my specific use case -Dave Kircher*/
#define HEADLESS_SURFACE_EXTENSION_REQUIRED_TO_LOAD 0


bool OFFSCREEN_Vulkan_LoadLibrary(SDL_VideoDevice *_this, const char *path)
{
    VkExtensionProperties *extensions = NULL;
    Uint32 extensionCount = 0;
    bool hasSurfaceExtension = false;
    bool hasHeadlessSurfaceExtension = false;
    PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr = NULL;
    Uint32 i;
    const char **paths;
    const char *foundPath = NULL;
    Uint32 numPaths;

    if (_this->vulkan_config.loader_handle) {
        return SDL_SetError("Vulkan already loaded");
    }

    // Load the Vulkan loader library
    if (!path) {
        path = SDL_GetHint(SDL_HINT_VULKAN_LIBRARY);
    }

#if defined(SDL_PLATFORM_APPLE)
    if (!path) {
        // Handle the case where Vulkan Portability is linked statically.
        vkGetInstanceProcAddr =
            (PFN_vkGetInstanceProcAddr)dlsym(DEFAULT_HANDLE,
                                             "vkGetInstanceProcAddr");
    }

    if (vkGetInstanceProcAddr) {
        _this->vulkan_config.loader_handle = DEFAULT_HANDLE;
    } else
#endif
    {
        if (path) {
            paths = &path;
            numPaths = 1;
        } else {
            paths = s_defaultPaths;
            numPaths = SDL_arraysize(s_defaultPaths);
        }

        for (i = 0; i < numPaths && _this->vulkan_config.loader_handle == NULL; i++) {
            foundPath = paths[i];
            _this->vulkan_config.loader_handle = SDL_LoadObject(foundPath);
        }

        if (_this->vulkan_config.loader_handle == NULL) {
            return SDL_SetError("Failed to load Vulkan Portability library");
        }

        SDL_strlcpy(_this->vulkan_config.loader_path, foundPath,
                    SDL_arraysize(_this->vulkan_config.loader_path));
        vkGetInstanceProcAddr = (PFN_vkGetInstanceProcAddr)SDL_LoadFunction(
            _this->vulkan_config.loader_handle, "vkGetInstanceProcAddr");

        if (!vkGetInstanceProcAddr) {
            SDL_SetError("Failed to load vkGetInstanceProcAddr from Vulkan Portability library");
            goto fail;
        }
    }

    _this->vulkan_config.vkGetInstanceProcAddr = (SDL_FunctionPointer)vkGetInstanceProcAddr;
    _this->vulkan_config.vkEnumerateInstanceExtensionProperties =
        (SDL_FunctionPointer)((PFN_vkGetInstanceProcAddr)_this->vulkan_config.vkGetInstanceProcAddr)(
            VK_NULL_HANDLE, "vkEnumerateInstanceExtensionProperties");
    if (!_this->vulkan_config.vkEnumerateInstanceExtensionProperties) {
        goto fail;
    }
    extensions = SDL_Vulkan_CreateInstanceExtensionsList(
        (PFN_vkEnumerateInstanceExtensionProperties)
            _this->vulkan_config.vkEnumerateInstanceExtensionProperties,
        &extensionCount);
    if (!extensions) {
        goto fail;
    }
    for (i = 0; i < extensionCount; i++) {
        if (SDL_strcmp(VK_KHR_SURFACE_EXTENSION_NAME, extensions[i].extensionName) == 0) {
            hasSurfaceExtension = true;
        } else if (SDL_strcmp(VK_EXT_HEADLESS_SURFACE_EXTENSION_NAME, extensions[i].extensionName) == 0) {
            hasHeadlessSurfaceExtension = true;
        }
    }
    SDL_free(extensions);
    if (!hasSurfaceExtension) {
        SDL_SetError("Installed Vulkan doesn't implement the " VK_KHR_SURFACE_EXTENSION_NAME " extension");
        goto fail;
    }
    if (!hasHeadlessSurfaceExtension) {
#if (HEADLESS_SURFACE_EXTENSION_REQUIRED_TO_LOAD != 0)
        SDL_SetError("Installed Vulkan doesn't implement the " VK_EXT_HEADLESS_SURFACE_EXTENSION_NAME " extension");
        goto fail;
#else
        // Let's at least leave a breadcrumb for people to find if they have issues
        SDL_Log("Installed Vulkan doesn't implement the " VK_EXT_HEADLESS_SURFACE_EXTENSION_NAME " extension");
#endif
    }
    return true;

fail:
    SDL_UnloadObject(_this->vulkan_config.loader_handle);
    _this->vulkan_config.loader_handle = NULL;
    return false;
}

void OFFSCREEN_Vulkan_UnloadLibrary(SDL_VideoDevice *_this)
{
    if (_this->vulkan_config.loader_handle) {
        SDL_UnloadObject(_this->vulkan_config.loader_handle);
        _this->vulkan_config.loader_handle = NULL;
    }
}

char const *const *OFFSCREEN_Vulkan_GetInstanceExtensions(SDL_VideoDevice *_this,
                                                          Uint32 *count)
{
#if (HEADLESS_SURFACE_EXTENSION_REQUIRED_TO_LOAD == 0)
    VkExtensionProperties *enumerateExtensions = NULL;
    Uint32 enumerateExtensionCount = 0;
    bool hasHeadlessSurfaceExtension = false;
    Uint32 i;
#endif

    static const char *const returnExtensions[] = { VK_KHR_SURFACE_EXTENSION_NAME, VK_EXT_HEADLESS_SURFACE_EXTENSION_NAME };
    if (count) {
#       if (HEADLESS_SURFACE_EXTENSION_REQUIRED_TO_LOAD == 0)
        {
            /* In optional mode, only return VK_EXT_HEADLESS_SURFACE_EXTENSION_NAME if it's already supported by the instance
                There's probably a better way to cache the presence of the extension during OFFSCREEN_Vulkan_LoadLibrary().
                But both SDL_VideoData and SDL_VideoDevice::vulkan_config seem like I'd need to touch a bunch of code to do properly.
                And I want a smaller footprint for the first pass*/
            if ( _this->vulkan_config.vkEnumerateInstanceExtensionProperties ) {
                enumerateExtensions = SDL_Vulkan_CreateInstanceExtensionsList(
                    (PFN_vkEnumerateInstanceExtensionProperties)
                        _this->vulkan_config.vkEnumerateInstanceExtensionProperties,
                    &enumerateExtensionCount);
                for (i = 0; i < enumerateExtensionCount; i++) {
                    if (SDL_strcmp(VK_EXT_HEADLESS_SURFACE_EXTENSION_NAME, enumerateExtensions[i].extensionName) == 0) {
                        hasHeadlessSurfaceExtension = true;
                    }
                }
                SDL_free(enumerateExtensions);
            }
            if ( hasHeadlessSurfaceExtension == true ) {
                *count = SDL_arraysize(returnExtensions);
            } else {
                *count = SDL_arraysize(returnExtensions) - 1; // assumes VK_EXT_HEADLESS_SURFACE_EXTENSION_NAME is last
            }
        }
#       else
        {
            *count = SDL_arraysize(returnExtensions);
        }
#       endif
    }
    return returnExtensions;
}

bool OFFSCREEN_Vulkan_CreateSurface(SDL_VideoDevice *_this,
                                   SDL_Window *window,
                                   VkInstance instance,
                                   const struct VkAllocationCallbacks *allocator,
                                   VkSurfaceKHR *surface)
{
    *surface = VK_NULL_HANDLE;

    if (!_this->vulkan_config.loader_handle) {
        return SDL_SetError("Vulkan is not loaded");
    }

    PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr = (PFN_vkGetInstanceProcAddr)_this->vulkan_config.vkGetInstanceProcAddr;
    PFN_vkCreateHeadlessSurfaceEXT vkCreateHeadlessSurfaceEXT =
        (PFN_vkCreateHeadlessSurfaceEXT)vkGetInstanceProcAddr(instance, "vkCreateHeadlessSurfaceEXT");
    VkHeadlessSurfaceCreateInfoEXT createInfo;
    VkResult result;
    if (!vkCreateHeadlessSurfaceEXT) {
        /* This may be surprising to the consumer when HEADLESS_SURFACE_EXTENSION_REQUIRED_TO_LOAD == 0
           But this is the tradeoff for allowing offscreen rendering to a buffer to continue working without requiring the extension during driver load */
        return SDL_SetError(VK_EXT_HEADLESS_SURFACE_EXTENSION_NAME
                            " extension is not enabled in the Vulkan instance.");
    }
    SDL_zero(createInfo);
    createInfo.sType = VK_STRUCTURE_TYPE_HEADLESS_SURFACE_CREATE_INFO_EXT;
    createInfo.pNext = NULL;
    createInfo.flags = 0;
    result = vkCreateHeadlessSurfaceEXT(instance, &createInfo, allocator, surface);
    if (result != VK_SUCCESS) {
        return SDL_SetError("vkCreateHeadlessSurfaceEXT failed: %s", SDL_Vulkan_GetResultString(result));
    }
    return true;
}

void OFFSCREEN_Vulkan_DestroySurface(SDL_VideoDevice *_this,
                                     VkInstance instance,
                                     VkSurfaceKHR surface,
                                     const struct VkAllocationCallbacks *allocator)
{
    if (_this->vulkan_config.loader_handle) {
        SDL_Vulkan_DestroySurface_Internal(_this->vulkan_config.vkGetInstanceProcAddr, instance, surface, allocator);
    }
}

#endif
