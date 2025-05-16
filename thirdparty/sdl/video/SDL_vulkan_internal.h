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
#ifndef SDL_vulkan_internal_h_
#define SDL_vulkan_internal_h_

#include "SDL_internal.h"

#ifdef SDL_VIDEO_VULKAN
#ifdef SDL_VIDEO_DRIVER_ANDROID
#define VK_USE_PLATFORM_ANDROID_KHR
#endif
#ifdef SDL_VIDEO_DRIVER_COCOA
#define VK_USE_PLATFORM_METAL_EXT
#define VK_USE_PLATFORM_MACOS_MVK
#endif
#ifdef SDL_VIDEO_DRIVER_UIKIT
#define VK_USE_PLATFORM_METAL_EXT
#define VK_USE_PLATFORM_IOS_MVK
#endif
#ifdef SDL_VIDEO_DRIVER_WAYLAND
#define VK_USE_PLATFORM_WAYLAND_KHR
#include "wayland/SDL_waylanddyn.h"
#endif
#ifdef SDL_VIDEO_DRIVER_WINDOWS
#define VK_USE_PLATFORM_WIN32_KHR
#include "../core/windows/SDL_windows.h"
#endif
#ifdef SDL_VIDEO_DRIVER_X11
#define VK_USE_PLATFORM_XLIB_KHR
#define VK_USE_PLATFORM_XCB_KHR
#endif

#define VK_NO_PROTOTYPES
#include "./khronos/vulkan/vulkan.h"

#include <SDL3/SDL_vulkan.h>

extern const char *SDL_Vulkan_GetResultString(VkResult result);

extern VkExtensionProperties *SDL_Vulkan_CreateInstanceExtensionsList(
    PFN_vkEnumerateInstanceExtensionProperties vkEnumerateInstanceExtensionProperties,
    Uint32 *extensionCount); // free returned list with SDL_free

/* Create a surface directly from a display connected to a physical device
 * using the DisplayKHR extension.
 * This needs to be passed an instance that was created with the VK_KHR_DISPLAY_EXTENSION_NAME
 * extension. */
extern bool SDL_Vulkan_Display_CreateSurface(void *vkGetInstanceProcAddr,
                                             VkInstance instance,
                                             const struct VkAllocationCallbacks *allocator,
                                             VkSurfaceKHR *surface);

/* Platform independent base function for destroying the Vulkan surface. Unlike surface
 * creation, surface destruction doesn't require platform specific extensions like
 * VK_KHR_wayland_surface, VK_KHR_android_surface or VK_EXT_metal_surface. The only
 * necessary extension is cross platform VK_KHR_surface, which is a dependency to all
 * WSI platform extensions, so we can handle surface destruction in an platform-independent
 * manner. */
extern void SDL_Vulkan_DestroySurface_Internal(void *vkGetInstanceProcAddr,
                                               VkInstance instance,
                                               VkSurfaceKHR surface,
                                               const struct VkAllocationCallbacks *allocator);
#else

// No SDL Vulkan support, just include the header for typedefs
#include <SDL3/SDL_vulkan.h>

typedef void (*PFN_vkGetInstanceProcAddr)(void);
typedef int (*PFN_vkEnumerateInstanceExtensionProperties)(void);

#endif // SDL_VIDEO_VULKAN

#endif // SDL_vulkan_internal_h_
