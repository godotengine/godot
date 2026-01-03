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

#ifdef SDL_USE_LIBUDEV
static const SDL_UDEV_Symbols *udev_ctx = NULL;

#define udev_device_get_devnode                       udev_ctx->udev_device_get_devnode
#define udev_device_get_parent_with_subsystem_devtype udev_ctx->udev_device_get_parent_with_subsystem_devtype
#define udev_device_get_sysattr_value                 udev_ctx->udev_device_get_sysattr_value
#define udev_device_get_syspath                       udev_ctx->udev_device_get_syspath
#define udev_device_new_from_devnum                   udev_ctx->udev_device_new_from_devnum
#define udev_device_new_from_syspath                  udev_ctx->udev_device_new_from_syspath
#define udev_device_unref                             udev_ctx->udev_device_unref
#define udev_enumerate_add_match_subsystem            udev_ctx->udev_enumerate_add_match_subsystem
#define udev_enumerate_get_list_entry                 udev_ctx->udev_enumerate_get_list_entry
#define udev_enumerate_new                            udev_ctx->udev_enumerate_new
#define udev_enumerate_scan_devices                   udev_ctx->udev_enumerate_scan_devices
#define udev_enumerate_unref                          udev_ctx->udev_enumerate_unref
#define udev_list_entry_get_name                      udev_ctx->udev_list_entry_get_name
#define udev_list_entry_get_next                      udev_ctx->udev_list_entry_get_next
#define udev_new                                      udev_ctx->udev_new
#define udev_unref                                    udev_ctx->udev_unref

#undef HIDAPI_H__
#define HIDAPI_ALLOW_BUILD_WORKAROUND_KERNEL_2_6_39
#include "linux/hid.c"
#define HAVE_PLATFORM_BACKEND 1

#endif /* SDL_USE_LIBUDEV */
