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

#if defined(SDL_VIDEO_VULKAN) && defined(SDL_VIDEO_DRIVER_COCOA)

#include "SDL_cocoavideo.h"
#include "SDL_cocoawindow.h"

#include "SDL_cocoametalview.h"
#include "SDL_cocoavulkan.h"

#include <dlfcn.h>

const char *defaultPaths[] = {
    "vulkan.framework/vulkan",
    "libvulkan.1.dylib",
    "libvulkan.dylib",
    "MoltenVK.framework/MoltenVK",
    "libMoltenVK.dylib"
};

// Since libSDL is most likely a .dylib, need RTLD_DEFAULT not RTLD_SELF.
#define DEFAULT_HANDLE RTLD_DEFAULT

bool Cocoa_Vulkan_LoadLibrary(SDL_VideoDevice *_this, const char *path)
{
    VkExtensionProperties *extensions = NULL;
    Uint32 extensionCount = 0;
    bool hasSurfaceExtension = false;
    bool hasMetalSurfaceExtension = false;
    bool hasMacOSSurfaceExtension = false;
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
            /* Look for framework or .dylib packaged with the application
             * instead. */
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

        SDL_strlcpy(_this->vulkan_config.loader_path, foundPath,
                    SDL_arraysize(_this->vulkan_config.loader_path));
        vkGetInstanceProcAddr = (PFN_vkGetInstanceProcAddr)SDL_LoadFunction(
            _this->vulkan_config.loader_handle, "vkGetInstanceProcAddr");
    }

    if (!vkGetInstanceProcAddr) {
        SDL_SetError("Failed to find %s in either executable or %s: %s",
                     "vkGetInstanceProcAddr",
                     _this->vulkan_config.loader_path,
                     (const char *)dlerror());
        goto fail;
    }

    _this->vulkan_config.vkGetInstanceProcAddr = (void *)vkGetInstanceProcAddr;
    _this->vulkan_config.vkEnumerateInstanceExtensionProperties =
        (void *)((PFN_vkGetInstanceProcAddr)_this->vulkan_config.vkGetInstanceProcAddr)(
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
    for (Uint32 i = 0; i < extensionCount; i++) {
        if (SDL_strcmp(VK_KHR_SURFACE_EXTENSION_NAME, extensions[i].extensionName) == 0) {
            hasSurfaceExtension = true;
        } else if (SDL_strcmp(VK_EXT_METAL_SURFACE_EXTENSION_NAME, extensions[i].extensionName) == 0) {
            hasMetalSurfaceExtension = true;
        } else if (SDL_strcmp(VK_MVK_MACOS_SURFACE_EXTENSION_NAME, extensions[i].extensionName) == 0) {
            hasMacOSSurfaceExtension = true;
        }
    }
    SDL_free(extensions);
    if (!hasSurfaceExtension) {
        SDL_SetError("Installed Vulkan Portability library doesn't implement the " VK_KHR_SURFACE_EXTENSION_NAME " extension");
        goto fail;
    } else if (!hasMetalSurfaceExtension && !hasMacOSSurfaceExtension) {
        SDL_SetError("Installed Vulkan Portability library doesn't implement the " VK_EXT_METAL_SURFACE_EXTENSION_NAME " or " VK_MVK_MACOS_SURFACE_EXTENSION_NAME " extensions");
        goto fail;
    }
    return true;

fail:
    SDL_UnloadObject(_this->vulkan_config.loader_handle);
    _this->vulkan_config.loader_handle = NULL;
    return false;
}

void Cocoa_Vulkan_UnloadLibrary(SDL_VideoDevice *_this)
{
    if (_this->vulkan_config.loader_handle) {
        if (_this->vulkan_config.loader_handle != DEFAULT_HANDLE) {
            SDL_UnloadObject(_this->vulkan_config.loader_handle);
        }
        _this->vulkan_config.loader_handle = NULL;
    }
}

char const* const* Cocoa_Vulkan_GetInstanceExtensions(SDL_VideoDevice *_this,
                                            Uint32 *count)
{
    static const char *const extensionsForCocoa[] = {
        VK_KHR_SURFACE_EXTENSION_NAME, VK_EXT_METAL_SURFACE_EXTENSION_NAME, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME
    };
    if(count) {
        *count = SDL_arraysize(extensionsForCocoa);
    }
    return extensionsForCocoa;
}

static bool Cocoa_Vulkan_CreateSurfaceViaMetalView(SDL_VideoDevice *_this,
                                                   SDL_Window *window,
                                                   VkInstance instance,
                                                   const struct VkAllocationCallbacks *allocator,
                                                   VkSurfaceKHR *surface,
                                                   PFN_vkCreateMetalSurfaceEXT vkCreateMetalSurfaceEXT,
                                                   PFN_vkCreateMacOSSurfaceMVK vkCreateMacOSSurfaceMVK)
{
    VkResult rc;
    SDL_MetalView metalview = Cocoa_Metal_CreateView(_this, window);
    if (metalview == NULL) {
        return false;
    }

    if (vkCreateMetalSurfaceEXT) {
        VkMetalSurfaceCreateInfoEXT createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_METAL_SURFACE_CREATE_INFO_EXT;
        createInfo.pNext = NULL;
        createInfo.flags = 0;
        createInfo.pLayer = (__bridge const CAMetalLayer *)
            Cocoa_Metal_GetLayer(_this, metalview);
        rc = vkCreateMetalSurfaceEXT(instance, &createInfo, allocator, surface);
        if (rc != VK_SUCCESS) {
            Cocoa_Metal_DestroyView(_this, metalview);
            return SDL_SetError("vkCreateMetalSurfaceEXT failed: %s", SDL_Vulkan_GetResultString(rc));
        }
    } else {
        VkMacOSSurfaceCreateInfoMVK createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_MACOS_SURFACE_CREATE_INFO_MVK;
        createInfo.pNext = NULL;
        createInfo.flags = 0;
        createInfo.pView = (const void *)metalview;
        rc = vkCreateMacOSSurfaceMVK(instance, &createInfo,
                                         NULL, surface);
        if (rc != VK_SUCCESS) {
            Cocoa_Metal_DestroyView(_this, metalview);
            return SDL_SetError("vkCreateMacOSSurfaceMVK failed: %s", SDL_Vulkan_GetResultString(rc));
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

    return true;  // success!
}

bool Cocoa_Vulkan_CreateSurface(SDL_VideoDevice *_this,
                               SDL_Window *window,
                               VkInstance instance,
                               const struct VkAllocationCallbacks *allocator,
                               VkSurfaceKHR *surface)
{
    PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr =
        (PFN_vkGetInstanceProcAddr)_this->vulkan_config.vkGetInstanceProcAddr;
    PFN_vkCreateMetalSurfaceEXT vkCreateMetalSurfaceEXT =
        (PFN_vkCreateMetalSurfaceEXT)vkGetInstanceProcAddr(
            instance,
            "vkCreateMetalSurfaceEXT");
    PFN_vkCreateMacOSSurfaceMVK vkCreateMacOSSurfaceMVK =
        (PFN_vkCreateMacOSSurfaceMVK)vkGetInstanceProcAddr(
            instance,
            "vkCreateMacOSSurfaceMVK");
    VkResult rc;

    if (!_this->vulkan_config.loader_handle) {
        return SDL_SetError("Vulkan is not loaded");
    }

    if (!vkCreateMetalSurfaceEXT && !vkCreateMacOSSurfaceMVK) {
        return SDL_SetError(VK_EXT_METAL_SURFACE_EXTENSION_NAME " or " VK_MVK_MACOS_SURFACE_EXTENSION_NAME
                            " extensions are not enabled in the Vulkan instance.");
    }

    if (window->flags & SDL_WINDOW_EXTERNAL) {
        @autoreleasepool {
            SDL_CocoaWindowData *data = (__bridge SDL_CocoaWindowData *)window->internal;
            if (![data.sdlContentView.layer isKindOfClass:[CAMetalLayer class]]) {
                [data.sdlContentView setLayer:[CAMetalLayer layer]];
            }

            if (vkCreateMetalSurfaceEXT) {
                VkMetalSurfaceCreateInfoEXT createInfo = {};
                createInfo.sType = VK_STRUCTURE_TYPE_METAL_SURFACE_CREATE_INFO_EXT;
                createInfo.pNext = NULL;
                createInfo.flags = 0;
                createInfo.pLayer = (CAMetalLayer *)data.sdlContentView.layer;
                rc = vkCreateMetalSurfaceEXT(instance, &createInfo, allocator, surface);
                if (rc != VK_SUCCESS) {
                    return SDL_SetError("vkCreateMetalSurfaceEXT failed: %s", SDL_Vulkan_GetResultString(rc));
                }
            } else {
                VkMacOSSurfaceCreateInfoMVK createInfo = {};
                createInfo.sType = VK_STRUCTURE_TYPE_MACOS_SURFACE_CREATE_INFO_MVK;
                createInfo.pNext = NULL;
                createInfo.flags = 0;
                createInfo.pView = (__bridge const void *)data.sdlContentView;
                rc = vkCreateMacOSSurfaceMVK(instance, &createInfo,
                                                 allocator, surface);
                if (rc != VK_SUCCESS) {
                    return SDL_SetError("vkCreateMacOSSurfaceMVK failed: %s", SDL_Vulkan_GetResultString(rc));
                }
            }
        }
    } else {
        return Cocoa_Vulkan_CreateSurfaceViaMetalView(_this, window, instance, allocator, surface, vkCreateMetalSurfaceEXT, vkCreateMacOSSurfaceMVK);
    }

    return true;
}

void Cocoa_Vulkan_DestroySurface(SDL_VideoDevice *_this,
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
