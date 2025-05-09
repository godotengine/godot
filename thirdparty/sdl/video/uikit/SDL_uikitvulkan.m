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

/*
 * @author Mark Callow, www.edgewise-consulting.com. Based on Jacob Lifshay's
 * SDL_x11vulkan.c.
 */

#include "SDL_internal.h"

#if defined(SDL_VIDEO_VULKAN) && defined(SDL_VIDEO_DRIVER_UIKIT)

#include "SDL_uikitvideo.h"
#include "SDL_uikitwindow.h"

#include "SDL_uikitvulkan.h"
#include "SDL_uikitmetalview.h"

#include <dlfcn.h>

const char *defaultPaths[] = {
    "libvulkan.dylib",
};

/* Since libSDL is static, could use RTLD_SELF. Using RTLD_DEFAULT is future
 * proofing. */
#define DEFAULT_HANDLE RTLD_DEFAULT

bool UIKit_Vulkan_LoadLibrary(SDL_VideoDevice *_this, const char *path)
{
    VkExtensionProperties *extensions = NULL;
    Uint32 extensionCount = 0;
    bool hasSurfaceExtension = false;
    bool hasMetalSurfaceExtension = false;
    bool hasIOSSurfaceExtension = false;
    PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr = NULL;

    if (_this->vulkan_config.loader_handle) {
        return SDL_SetError("Vulkan Portability library is already loaded.");
    }

    // Load the Vulkan loader library
    if (!path) {
        path = SDL_GetHint(SDL_HINT_VULKAN_LIBRARY);
    }

    if (!path) {
        // Handle the case where Vulkan Portability is linked statically.
        vkGetInstanceProcAddr =
            (PFN_vkGetInstanceProcAddr)dlsym(DEFAULT_HANDLE,
                                             "vkGetInstanceProcAddr");
    }

    if (vkGetInstanceProcAddr) {
        _this->vulkan_config.loader_handle = DEFAULT_HANDLE;
    } else {
        const char **paths;
        const char *foundPath = NULL;
        int numPaths;
        int i;

        if (path) {
            paths = &path;
            numPaths = 1;
        } else {
            // Look for the .dylib packaged with the application instead.
            paths = defaultPaths;
            numPaths = SDL_arraysize(defaultPaths);
        }

        for (i = 0; i < numPaths && _this->vulkan_config.loader_handle == NULL; i++) {
            foundPath = paths[i];
            _this->vulkan_config.loader_handle = SDL_LoadObject(foundPath);
        }

        if (_this->vulkan_config.loader_handle == NULL) {
            return SDL_SetError("Failed to load Vulkan Portability library");
        }

        SDL_strlcpy(_this->vulkan_config.loader_path, path,
                    SDL_arraysize(_this->vulkan_config.loader_path));
        vkGetInstanceProcAddr =
            (PFN_vkGetInstanceProcAddr)SDL_LoadFunction(
                _this->vulkan_config.loader_handle,
                "vkGetInstanceProcAddr");
    }

    if (!vkGetInstanceProcAddr) {
        SDL_SetError("Failed to find %s in either executable or %s: %s",
                     "vkGetInstanceProcAddr",
                     "linked Vulkan Portability library",
                     (const char *)dlerror());
        goto fail;
    }

    _this->vulkan_config.vkGetInstanceProcAddr = (void *)vkGetInstanceProcAddr;
    _this->vulkan_config.vkEnumerateInstanceExtensionProperties =
        (void *)((PFN_vkGetInstanceProcAddr)_this->vulkan_config.vkGetInstanceProcAddr)(
            VK_NULL_HANDLE, "vkEnumerateInstanceExtensionProperties");

    if (!_this->vulkan_config.vkEnumerateInstanceExtensionProperties) {
        SDL_SetError("No vkEnumerateInstanceExtensionProperties found.");
        goto fail;
    }

    extensions = SDL_Vulkan_CreateInstanceExtensionsList(
        (PFN_vkEnumerateInstanceExtensionProperties)
            _this->vulkan_config.vkEnumerateInstanceExtensionProperties,
        &extensionCount);

    if (!extensions) {
        goto fail;
    }

    for (Uint32 i = 0; i < extensionCount; i++) {
        if (SDL_strcmp(VK_KHR_SURFACE_EXTENSION_NAME, extensions[i].extensionName) == 0) {
            hasSurfaceExtension = true;
        } else if (SDL_strcmp(VK_EXT_METAL_SURFACE_EXTENSION_NAME, extensions[i].extensionName) == 0) {
            hasMetalSurfaceExtension = true;
        } else if (SDL_strcmp(VK_MVK_IOS_SURFACE_EXTENSION_NAME, extensions[i].extensionName) == 0) {
            hasIOSSurfaceExtension = true;
        }
    }

    SDL_free(extensions);

    if (!hasSurfaceExtension) {
        SDL_SetError("Installed Vulkan Portability doesn't implement the " VK_KHR_SURFACE_EXTENSION_NAME " extension");
        goto fail;
    } else if (!hasMetalSurfaceExtension && !hasIOSSurfaceExtension) {
        SDL_SetError("Installed Vulkan Portability doesn't implement the " VK_EXT_METAL_SURFACE_EXTENSION_NAME " or " VK_MVK_IOS_SURFACE_EXTENSION_NAME " extensions");
        goto fail;
    }

    return true;

fail:
    _this->vulkan_config.loader_handle = NULL;
    return false;
}

void UIKit_Vulkan_UnloadLibrary(SDL_VideoDevice *_this)
{
    if (_this->vulkan_config.loader_handle) {
        if (_this->vulkan_config.loader_handle != DEFAULT_HANDLE) {
            SDL_UnloadObject(_this->vulkan_config.loader_handle);
        }
        _this->vulkan_config.loader_handle = NULL;
    }
}

char const* const* UIKit_Vulkan_GetInstanceExtensions(SDL_VideoDevice *_this,
                                            Uint32 *count)
{
    static const char *const extensionsForUIKit[] = {
        VK_KHR_SURFACE_EXTENSION_NAME, VK_EXT_METAL_SURFACE_EXTENSION_NAME
    };
    if(count) {
        *count = SDL_arraysize(extensionsForUIKit);
    }
    return extensionsForUIKit;
}

bool UIKit_Vulkan_CreateSurface(SDL_VideoDevice *_this,
                               SDL_Window *window,
                               VkInstance instance,
                               const struct VkAllocationCallbacks *allocator,
                               VkSurfaceKHR *surface)
{
    PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr =
        (PFN_vkGetInstanceProcAddr)_this->vulkan_config.vkGetInstanceProcAddr;
    PFN_vkCreateMetalSurfaceEXT vkCreateMetalSurfaceEXT =
        (PFN_vkCreateMetalSurfaceEXT)vkGetInstanceProcAddr(
            (VkInstance)instance,
            "vkCreateMetalSurfaceEXT");
    PFN_vkCreateIOSSurfaceMVK vkCreateIOSSurfaceMVK =
        (PFN_vkCreateIOSSurfaceMVK)vkGetInstanceProcAddr(
            (VkInstance)instance,
            "vkCreateIOSSurfaceMVK");
    VkResult result;
    SDL_MetalView metalview;

    if (!_this->vulkan_config.loader_handle) {
        return SDL_SetError("Vulkan is not loaded");
    }

    if (!vkCreateMetalSurfaceEXT && !vkCreateIOSSurfaceMVK) {
        return SDL_SetError(VK_EXT_METAL_SURFACE_EXTENSION_NAME " or " VK_MVK_IOS_SURFACE_EXTENSION_NAME
                            " extensions are not enabled in the Vulkan instance.");
    }

    metalview = UIKit_Metal_CreateView(_this, window);
    if (metalview == NULL) {
        return false;
    }

    if (vkCreateMetalSurfaceEXT) {
        VkMetalSurfaceCreateInfoEXT createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_METAL_SURFACE_CREATE_INFO_EXT;
        createInfo.pNext = NULL;
        createInfo.flags = 0;
        createInfo.pLayer = (__bridge const CAMetalLayer *)
            UIKit_Metal_GetLayer(_this, metalview);
        result = vkCreateMetalSurfaceEXT(instance, &createInfo, allocator, surface);
        if (result != VK_SUCCESS) {
            UIKit_Metal_DestroyView(_this, metalview);
            return SDL_SetError("vkCreateMetalSurfaceEXT failed: %s", SDL_Vulkan_GetResultString(result));
        }
    } else {
        VkIOSSurfaceCreateInfoMVK createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_IOS_SURFACE_CREATE_INFO_MVK;
        createInfo.pNext = NULL;
        createInfo.flags = 0;
        createInfo.pView = (const void *)metalview;
        result = vkCreateIOSSurfaceMVK(instance, &createInfo,
                                       allocator, surface);
        if (result != VK_SUCCESS) {
            UIKit_Metal_DestroyView(_this, metalview);
            return SDL_SetError("vkCreateIOSSurfaceMVK failed: %s", SDL_Vulkan_GetResultString(result));
        }
    }

    /* Unfortunately there's no SDL_Vulkan_DestroySurface function we can call
     * Metal_DestroyView from. Right now the metal view's ref count is +2 (one
     * from returning a new view object in CreateView, and one because it's
     * a subview of the window.) If we release the view here to make it +1, it
     * will be destroyed when the window is destroyed.
     *
     * TODO: Now that we have SDL_Vulkan_DestroySurface someone with enough
     * knowledge of Metal can proceed. */
    CFBridgingRelease(metalview);

    return true;
}

void UIKit_Vulkan_DestroySurface(SDL_VideoDevice *_this,
                                 VkInstance instance,
                                 VkSurfaceKHR surface,
                                 const struct VkAllocationCallbacks *allocator)
{
    if (_this->vulkan_config.loader_handle) {
        SDL_Vulkan_DestroySurface_Internal(_this->vulkan_config.vkGetInstanceProcAddr, instance, surface, allocator);
        // TODO: Add CFBridgingRelease(metalview) here perhaps?
    }
}

#endif
