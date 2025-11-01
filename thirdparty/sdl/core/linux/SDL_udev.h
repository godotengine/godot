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

#ifndef SDL_udev_h_
#define SDL_udev_h_

#if defined(HAVE_LIBUDEV_H) && defined(HAVE_LINUX_INPUT_H)

#ifndef SDL_USE_LIBUDEV
#define SDL_USE_LIBUDEV 1
#endif

//#include <libudev.h>
#include "thirdparty/linuxbsd_headers/udev/libudev.h"
#include <sys/time.h>
#include <sys/types.h>

/**
 *  Device type
 */

typedef enum
{
    SDL_UDEV_DEVICEADDED = 1,
    SDL_UDEV_DEVICEREMOVED
} SDL_UDEV_deviceevent;

typedef void (*SDL_UDEV_Callback)(SDL_UDEV_deviceevent udev_type, int udev_class, const char *devpath);

typedef struct SDL_UDEV_CallbackList
{
    SDL_UDEV_Callback callback;
    struct SDL_UDEV_CallbackList *next;
} SDL_UDEV_CallbackList;

typedef struct SDL_UDEV_Symbols
{
    const char *(*udev_device_get_action)(struct udev_device *);
    const char *(*udev_device_get_devnode)(struct udev_device *);
    const char *(*udev_device_get_syspath)(struct udev_device *);
    const char *(*udev_device_get_subsystem)(struct udev_device *);
    struct udev_device *(*udev_device_get_parent_with_subsystem_devtype)(struct udev_device *udev_device, const char *subsystem, const char *devtype);
    const char *(*udev_device_get_property_value)(struct udev_device *, const char *);
    const char *(*udev_device_get_sysattr_value)(struct udev_device *udev_device, const char *sysattr);
    struct udev_device *(*udev_device_new_from_syspath)(struct udev *, const char *);
    void (*udev_device_unref)(struct udev_device *);
    int (*udev_enumerate_add_match_property)(struct udev_enumerate *, const char *, const char *);
    int (*udev_enumerate_add_match_subsystem)(struct udev_enumerate *, const char *);
    struct udev_list_entry *(*udev_enumerate_get_list_entry)(struct udev_enumerate *);
    struct udev_enumerate *(*udev_enumerate_new)(struct udev *);
    int (*udev_enumerate_scan_devices)(struct udev_enumerate *);
    void (*udev_enumerate_unref)(struct udev_enumerate *);
    const char *(*udev_list_entry_get_name)(struct udev_list_entry *);
    struct udev_list_entry *(*udev_list_entry_get_next)(struct udev_list_entry *);
    int (*udev_monitor_enable_receiving)(struct udev_monitor *);
    int (*udev_monitor_filter_add_match_subsystem_devtype)(struct udev_monitor *, const char *, const char *);
    int (*udev_monitor_get_fd)(struct udev_monitor *);
    struct udev_monitor *(*udev_monitor_new_from_netlink)(struct udev *, const char *);
    struct udev_device *(*udev_monitor_receive_device)(struct udev_monitor *);
    void (*udev_monitor_unref)(struct udev_monitor *);
    struct udev *(*udev_new)(void);
    void (*udev_unref)(struct udev *);
    struct udev_device *(*udev_device_new_from_devnum)(struct udev *udev, char type, dev_t devnum);
    dev_t (*udev_device_get_devnum)(struct udev_device *udev_device);
} SDL_UDEV_Symbols;

typedef struct SDL_UDEV_PrivateData
{
    const char *udev_library;
    SDL_SharedObject *udev_handle;
    struct udev *udev;
    struct udev_monitor *udev_mon;
    int ref_count;
    SDL_UDEV_CallbackList *first, *last;

    // Function pointers
    SDL_UDEV_Symbols syms;
} SDL_UDEV_PrivateData;

extern bool SDL_UDEV_Init(void);
extern void SDL_UDEV_Quit(void);
extern void SDL_UDEV_UnloadLibrary(void);
extern bool SDL_UDEV_LoadLibrary(void);
extern void SDL_UDEV_Poll(void);
extern bool SDL_UDEV_Scan(void);
extern bool SDL_UDEV_GetProductInfo(const char *device_path, Uint16 *vendor, Uint16 *product, Uint16 *version, int *class);
extern bool SDL_UDEV_AddCallback(SDL_UDEV_Callback cb);
extern void SDL_UDEV_DelCallback(SDL_UDEV_Callback cb);
extern const SDL_UDEV_Symbols *SDL_UDEV_GetUdevSyms(void);
extern void SDL_UDEV_ReleaseUdevSyms(void);

#endif // HAVE_LIBUDEV_H && HAVE_LINUX_INPUT_H

#endif // SDL_udev_h_
