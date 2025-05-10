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

#ifdef SDL_JOYSTICK_USBHID

/*
 * Joystick driver for the uhid(4) / ujoy(4) interface found in OpenBSD,
 * NetBSD and FreeBSD.
 *
 * Maintainer: <vedge at csoft.org>
 */

#include <sys/param.h>
#include <sys/stat.h>

#include <unistd.h>
#include <fcntl.h>
#include <errno.h>

#ifndef __FreeBSD_kernel_version
#define __FreeBSD_kernel_version __FreeBSD_version
#endif

#ifdef HAVE_USB_H
#include <usb.h>
#endif
#ifdef __DragonFly__
#include <bus/u4b/usb.h>
#include <bus/u4b/usbhid.h>
#else
#include <dev/usb/usb.h>
#include <dev/usb/usbhid.h>
#endif

#ifdef HAVE_USBHID_H
#include <usbhid.h>
#elif defined(HAVE_LIBUSB_H)
#include <libusb.h>
#elif defined(HAVE_LIBUSBHID_H)
#include <libusbhid.h>
#endif

#if defined(SDL_PLATFORM_FREEBSD)
#include <osreldate.h>
#if __FreeBSD_kernel_version > 800063
#include <dev/usb/usb_ioctl.h>
#elif defined(__DragonFly__)
#include <bus/u4b/usb_ioctl.h>
#endif
#include <sys/joystick.h>
#endif

#ifdef SDL_HAVE_MACHINE_JOYSTICK_H
#include <machine/joystick.h>
#endif

#include "../SDL_sysjoystick.h"
#include "../SDL_joystick_c.h"
#include "../hidapi/SDL_hidapijoystick_c.h"

#if defined(SDL_PLATFORM_FREEBSD) || defined(SDL_HAVE_MACHINE_JOYSTICK_H) || defined(__FreeBSD_kernel__) || defined(__DragonFly_)
#define SUPPORT_JOY_GAMEPORT
#endif

#define MAX_UHID_JOYS 64
#define MAX_JOY_JOYS  2
#define MAX_JOYS      (MAX_UHID_JOYS + MAX_JOY_JOYS)

#ifdef SDL_PLATFORM_OPENBSD

#define HUG_DPAD_UP    0x90
#define HUG_DPAD_DOWN  0x91
#define HUG_DPAD_RIGHT 0x92
#define HUG_DPAD_LEFT  0x93

#define HAT_UP        0x01
#define HAT_RIGHT     0x02
#define HAT_DOWN      0x04
#define HAT_LEFT      0x08

#endif

struct report
{
#if defined(SDL_PLATFORM_FREEBSD) && (__FreeBSD_kernel_version > 900000) || \
    defined(__DragonFly__)
    void *buf; // Buffer
#elif defined(SDL_PLATFORM_FREEBSD) && (__FreeBSD_kernel_version > 800063)
    struct usb_gen_descriptor *buf; // Buffer
#else
    struct usb_ctl_report *buf; // Buffer
#endif
    size_t size; // Buffer size
    int rid;     // Report ID
    enum
    {
        SREPORT_UNINIT,
        SREPORT_CLEAN,
        SREPORT_DIRTY
    } status;
};

static struct
{
    int uhid_report;
    hid_kind_t kind;
    const char *name;
} const repinfo[] = {
    { UHID_INPUT_REPORT, hid_input, "input" },
    { UHID_OUTPUT_REPORT, hid_output, "output" },
    { UHID_FEATURE_REPORT, hid_feature, "feature" }
};

enum
{
    REPORT_INPUT = 0,
    REPORT_OUTPUT = 1,
    REPORT_FEATURE = 2
};

enum
{
    JOYAXE_X,
    JOYAXE_Y,
    JOYAXE_Z,
    JOYAXE_SLIDER,
    JOYAXE_WHEEL,
    JOYAXE_RX,
    JOYAXE_RY,
    JOYAXE_RZ,
    JOYAXE_count
};

struct joystick_hwdata
{
    int fd;
    enum
    {
        BSDJOY_UHID, // uhid(4)
        BSDJOY_JOY   // joy(4)
    } type;

    int naxes;
    int nbuttons;
    int nhats;
    struct report_desc *repdesc;
    struct report inreport;
    int axis_map[JOYAXE_count]; /* map present JOYAXE_* to 0,1,.. */
};

// A linked list of available joysticks
typedef struct SDL_joylist_item
{
    SDL_JoystickID device_instance;
    char *path; // "/dev/uhid0" or whatever
    char *name; // "SideWinder 3D Pro" or whatever
    SDL_GUID guid;
    dev_t devnum;
    struct SDL_joylist_item *next;
} SDL_joylist_item;

static SDL_joylist_item *SDL_joylist = NULL;
static SDL_joylist_item *SDL_joylist_tail = NULL;
static int numjoysticks = 0;

static bool report_alloc(struct report *, struct report_desc *, int);
static void report_free(struct report *);

#if defined(USBHID_UCR_DATA) || (defined(__FreeBSD_kernel__) && __FreeBSD_kernel_version <= 800063)
#define REP_BUF_DATA(rep) ((rep)->buf->ucr_data)
#elif (defined(SDL_PLATFORM_FREEBSD) && (__FreeBSD_kernel_version > 900000)) || \
    defined(__DragonFly__)
#define REP_BUF_DATA(rep) ((rep)->buf)
#elif (defined(SDL_PLATFORM_FREEBSD) && (__FreeBSD_kernel_version > 800063))
#define REP_BUF_DATA(rep) ((rep)->buf->ugd_data)
#else
#define REP_BUF_DATA(rep) ((rep)->buf->data)
#endif

static int usage_to_joyaxe(int usage)
{
    int joyaxe;
    switch (usage) {
    case HUG_X:
        joyaxe = JOYAXE_X;
        break;
    case HUG_Y:
        joyaxe = JOYAXE_Y;
        break;
    case HUG_Z:
        joyaxe = JOYAXE_Z;
        break;
    case HUG_SLIDER:
        joyaxe = JOYAXE_SLIDER;
        break;
    case HUG_WHEEL:
        joyaxe = JOYAXE_WHEEL;
        break;
    case HUG_RX:
        joyaxe = JOYAXE_RX;
        break;
    case HUG_RY:
        joyaxe = JOYAXE_RY;
        break;
    case HUG_RZ:
        joyaxe = JOYAXE_RZ;
        break;
    default:
        joyaxe = -1;
    }
    return joyaxe;
}

static void FreeJoylistItem(SDL_joylist_item *item)
{
    SDL_free(item->path);
    SDL_free(item->name);
    SDL_free(item);
}

static void FreeHwData(struct joystick_hwdata *hw)
{
    if (hw->type == BSDJOY_UHID) {
        report_free(&hw->inreport);

        if (hw->repdesc) {
            hid_dispose_report_desc(hw->repdesc);
        }
    }
    close(hw->fd);
    SDL_free(hw);
}

static struct joystick_hwdata *CreateHwData(const char *path)
{
    struct joystick_hwdata *hw;
    struct hid_item hitem;
    struct hid_data *hdata;
    struct report *rep = NULL;
    int fd;
    int i;

    fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd == -1) {
        SDL_SetError("%s: %s", path, strerror(errno));
        return NULL;
    }

    hw = (struct joystick_hwdata *)
        SDL_calloc(1, sizeof(struct joystick_hwdata));
    if (!hw) {
        close(fd);
        return NULL;
    }
    hw->fd = fd;

#ifdef SUPPORT_JOY_GAMEPORT
    if (SDL_strncmp(path, "/dev/joy", 8) == 0) {
        hw->type = BSDJOY_JOY;
        hw->naxes = 2;
        hw->nbuttons = 2;
    } else
#endif
    {
        hw->type = BSDJOY_UHID;
        {
            int ax;
            for (ax = 0; ax < JOYAXE_count; ax++) {
                hw->axis_map[ax] = -1;
            }
        }
        hw->repdesc = hid_get_report_desc(fd);
        if (!hw->repdesc) {
            SDL_SetError("%s: USB_GET_REPORT_DESC: %s", path,
                         strerror(errno));
            goto usberr;
        }
        rep = &hw->inreport;
#if defined(SDL_PLATFORM_FREEBSD) && (__FreeBSD_kernel_version > 800063) || defined(__FreeBSD_kernel__) || defined(__DragonFly__)
        rep->rid = hid_get_report_id(fd);
        if (rep->rid < 0) {
#else
        if (ioctl(fd, USB_GET_REPORT_ID, &rep->rid) < 0) {
#endif
            rep->rid = -1; // XXX
        }
        if (!report_alloc(rep, hw->repdesc, REPORT_INPUT)) {
            goto usberr;
        }
        if (rep->size <= 0) {
            SDL_SetError("%s: Input report descriptor has invalid length",
                         path);
            goto usberr;
        }
#if defined(USBHID_NEW) || (defined(SDL_PLATFORM_FREEBSD) && __FreeBSD_kernel_version >= 500111) || defined(__FreeBSD_kernel__) || defined(__DragonFly__)
        hdata = hid_start_parse(hw->repdesc, 1 << hid_input, rep->rid);
#else
        hdata = hid_start_parse(hw->repdesc, 1 << hid_input);
#endif
        if (!hdata) {
            SDL_SetError("%s: Cannot start HID parser", path);
            goto usberr;
        }
        for (i = 0; i < JOYAXE_count; i++) {
            hw->axis_map[i] = -1;
        }

        while (hid_get_item(hdata, &hitem) > 0) {
            switch (hitem.kind) {
            case hid_input:
                switch (HID_PAGE(hitem.usage)) {
                case HUP_GENERIC_DESKTOP:
                {
                    int usage = HID_USAGE(hitem.usage);
                    int joyaxe = usage_to_joyaxe(usage);
                    if (joyaxe >= 0) {
                        hw->axis_map[joyaxe] = 1;
                    } else if (usage == HUG_HAT_SWITCH
#ifdef SDL_PLATFORM_OPENBSD
                               || usage == HUG_DPAD_UP
#endif
                    ) {
                        hw->nhats++;
                    }
                    break;
                }
                case HUP_BUTTON:
                {
                    int usage = HID_USAGE(hitem.usage);
                    if (usage > hw->nbuttons) {
                        hw->nbuttons = usage;
                    }
                } break;
                default:
                    break;
                }
                break;
            default:
                break;
            }
        }
        hid_end_parse(hdata);
        for (i = 0; i < JOYAXE_count; i++) {
            if (hw->axis_map[i] > 0) {
                hw->axis_map[i] = hw->naxes++;
            }
        }

        if (hw->naxes == 0 && hw->nbuttons == 0 && hw->nhats == 0) {
            SDL_SetError("%s: Not a joystick, ignoring", path);
            goto usberr;
        }
    }

    // The poll blocks the event thread.
    fcntl(fd, F_SETFL, O_NONBLOCK);
#ifdef SDL_PLATFORM_NETBSD
    // Flush pending events
    if (rep) {
        while (read(fd, REP_BUF_DATA(rep), rep->size) == rep->size)
            ;
    }
#endif

    return hw;

usberr:
    FreeHwData(hw);
    return NULL;
}

static bool MaybeAddDevice(const char *path)
{
    struct stat sb;
    char *name = NULL;
    SDL_GUID guid;
    SDL_joylist_item *item;
    struct joystick_hwdata *hw;

    if (!path) {
        return false;
    }

    if (stat(path, &sb) == -1) {
        return false;
    }

    // Check to make sure it's not already in list.
    for (item = SDL_joylist; item; item = item->next) {
        if (sb.st_rdev == item->devnum) {
            return false; // already have this one
        }
    }

    hw = CreateHwData(path);
    if (!hw) {
        return false;
    }

    if (hw->type == BSDJOY_JOY) {
        name = SDL_strdup("Gameport joystick");
        guid = SDL_CreateJoystickGUIDForName(name);
    } else {
#ifdef USB_GET_DEVICEINFO
        struct usb_device_info di;
        if (ioctl(hw->fd, USB_GET_DEVICEINFO, &di) != -1) {
            name = SDL_CreateJoystickName(di.udi_vendorNo, di.udi_productNo, di.udi_vendor, di.udi_product);
            guid = SDL_CreateJoystickGUID(SDL_HARDWARE_BUS_USB, di.udi_vendorNo, di.udi_productNo, di.udi_releaseNo, di.udi_vendor, di.udi_product, 0, 0);

            if (SDL_ShouldIgnoreJoystick(di.udi_vendorNo, di.udi_productNo, di.udi_releaseNo, name) ||
                SDL_JoystickHandledByAnotherDriver(&SDL_BSD_JoystickDriver, di.udi_vendorNo, di.udi_productNo, di.udi_releaseNo, name)) {
                SDL_free(name);
                FreeHwData(hw);
                return false;
            }
        }
#endif // USB_GET_DEVICEINFO
    }
    if (!name) {
        name = SDL_strdup(path);
        guid = SDL_CreateJoystickGUIDForName(name);
    }
    FreeHwData(hw);

    item = (SDL_joylist_item *)SDL_calloc(1, sizeof(SDL_joylist_item));
    if (!item) {
        SDL_free(name);
        return false;
    }

    item->devnum = sb.st_rdev;
    item->path = SDL_strdup(path);
    item->name = name;
    item->guid = guid;

    if ((!item->path) || (!item->name)) {
        FreeJoylistItem(item);
        return false;
    }

    item->device_instance = SDL_GetNextObjectID();
    if (!SDL_joylist_tail) {
        SDL_joylist = SDL_joylist_tail = item;
    } else {
        SDL_joylist_tail->next = item;
        SDL_joylist_tail = item;
    }

    // Need to increment the joystick count before we post the event
    ++numjoysticks;

    SDL_PrivateJoystickAdded(item->device_instance);

    return true;
}

static bool BSD_JoystickInit(void)
{
    char s[16];
    int i;

    for (i = 0; i < MAX_UHID_JOYS; i++) {
#if defined(SDL_PLATFORM_OPENBSD) && (OpenBSD >= 202105)
        SDL_snprintf(s, SDL_arraysize(s), "/dev/ujoy/%d", i);
#else
        SDL_snprintf(s, SDL_arraysize(s), "/dev/uhid%d", i);
#endif
        MaybeAddDevice(s);
    }
#ifdef SUPPORT_JOY_GAMEPORT
    for (i = 0; i < MAX_JOY_JOYS; i++) {
        SDL_snprintf(s, SDL_arraysize(s), "/dev/joy%d", i);
        MaybeAddDevice(s);
    }
#endif // SUPPORT_JOY_GAMEPORT

    // Read the default USB HID usage table.
    hid_init(NULL);

    return true;
}

static int BSD_JoystickGetCount(void)
{
    return numjoysticks;
}

static void BSD_JoystickDetect(void)
{
}

static bool BSD_JoystickIsDevicePresent(Uint16 vendor_id, Uint16 product_id, Uint16 version, const char *name)
{
    // We don't override any other drivers
    return false;
}

static SDL_joylist_item *GetJoystickByDevIndex(int device_index)
{
    SDL_joylist_item *item = SDL_joylist;

    if ((device_index < 0) || (device_index >= numjoysticks)) {
        return NULL;
    }

    while (device_index > 0) {
        SDL_assert(item != NULL);
        device_index--;
        item = item->next;
    }

    return item;
}

static const char *BSD_JoystickGetDeviceName(int device_index)
{
    return GetJoystickByDevIndex(device_index)->name;
}

static const char *BSD_JoystickGetDevicePath(int device_index)
{
    return GetJoystickByDevIndex(device_index)->path;
}

static int BSD_JoystickGetDeviceSteamVirtualGamepadSlot(int device_index)
{
    return -1;
}

static int BSD_JoystickGetDevicePlayerIndex(int device_index)
{
    return -1;
}

static void BSD_JoystickSetDevicePlayerIndex(int device_index, int player_index)
{
}

static SDL_GUID BSD_JoystickGetDeviceGUID(int device_index)
{
    return GetJoystickByDevIndex(device_index)->guid;
}

// Function to perform the mapping from device index to the instance id for this index
static SDL_JoystickID BSD_JoystickGetDeviceInstanceID(int device_index)
{
    return GetJoystickByDevIndex(device_index)->device_instance;
}

static unsigned hatval_to_sdl(Sint32 hatval)
{
    static const unsigned hat_dir_map[8] = {
        SDL_HAT_UP, SDL_HAT_RIGHTUP, SDL_HAT_RIGHT, SDL_HAT_RIGHTDOWN,
        SDL_HAT_DOWN, SDL_HAT_LEFTDOWN, SDL_HAT_LEFT, SDL_HAT_LEFTUP
    };
    unsigned result;
    if ((hatval & 7) == hatval)
        result = hat_dir_map[hatval];
    else
        result = SDL_HAT_CENTERED;
    return result;
}

static bool BSD_JoystickOpen(SDL_Joystick *joy, int device_index)
{
    SDL_joylist_item *item = GetJoystickByDevIndex(device_index);
    struct joystick_hwdata *hw;

    if (!item) {
        return SDL_SetError("No such device");
    }

    hw = CreateHwData(item->path);
    if (!hw) {
        return false;
    }

    joy->hwdata = hw;
    joy->naxes = hw->naxes;
    joy->nbuttons = hw->nbuttons;
    joy->nhats = hw->nhats;

    return true;
}

static void BSD_JoystickUpdate(SDL_Joystick *joy)
{
    struct hid_item hitem;
    struct hid_data *hdata;
    struct report *rep;
    int nbutton, naxe = -1;
    Sint32 v;
#ifdef SDL_PLATFORM_OPENBSD
    Sint32 dpad[4] = { 0, 0, 0, 0 };
#endif
    Uint64 timestamp = SDL_GetTicksNS();

#ifdef SUPPORT_JOY_GAMEPORT
    struct joystick gameport;
    static int x, y, xmin = 0xffff, ymin = 0xffff, xmax = 0, ymax = 0;

    if (joy->hwdata->type == BSDJOY_JOY) {
        while (read(joy->hwdata->fd, &gameport, sizeof(gameport)) == sizeof(gameport)) {
            if (SDL_abs(x - gameport.x) > 8) {
                x = gameport.x;
                if (x < xmin) {
                    xmin = x;
                }
                if (x > xmax) {
                    xmax = x;
                }
                if (xmin == xmax) {
                    xmin--;
                    xmax++;
                }
                v = (((SDL_JOYSTICK_AXIS_MAX - SDL_JOYSTICK_AXIS_MIN) * ((Sint32)x - xmin)) / (xmax - xmin)) + SDL_JOYSTICK_AXIS_MIN;
                SDL_SendJoystickAxis(timestamp, joy, 0, v);
            }
            if (SDL_abs(y - gameport.y) > 8) {
                y = gameport.y;
                if (y < ymin) {
                    ymin = y;
                }
                if (y > ymax) {
                    ymax = y;
                }
                if (ymin == ymax) {
                    ymin--;
                    ymax++;
                }
                v = (((SDL_JOYSTICK_AXIS_MAX - SDL_JOYSTICK_AXIS_MIN) * ((Sint32)y - ymin)) / (ymax - ymin)) + SDL_JOYSTICK_AXIS_MIN;
                SDL_SendJoystickAxis(timestamp, joy, 1, v);
            }
            SDL_SendJoystickButton(timestamp, joy, 0, (gameport.b1 != 0));
            SDL_SendJoystickButton(timestamp, joy, 1, (gameport.b2 != 0));
        }
        return;
    }
#endif // SUPPORT_JOY_GAMEPORT

    rep = &joy->hwdata->inreport;

    while (read(joy->hwdata->fd, REP_BUF_DATA(rep), rep->size) == rep->size) {
#if defined(USBHID_NEW) || (defined(SDL_PLATFORM_FREEBSD) && __FreeBSD_kernel_version >= 500111) || defined(__FreeBSD_kernel__) || defined(__DragonFly__)
        hdata = hid_start_parse(joy->hwdata->repdesc, 1 << hid_input, rep->rid);
#else
        hdata = hid_start_parse(joy->hwdata->repdesc, 1 << hid_input);
#endif
        if (!hdata) {
            // fprintf(stderr, "%s: Cannot start HID parser\n", joy->hwdata->path);
            continue;
        }

        while (hid_get_item(hdata, &hitem) > 0) {
            switch (hitem.kind) {
            case hid_input:
                switch (HID_PAGE(hitem.usage)) {
                case HUP_GENERIC_DESKTOP:
                {
                    int usage = HID_USAGE(hitem.usage);
                    int joyaxe = usage_to_joyaxe(usage);
                    if (joyaxe >= 0) {
                        naxe = joy->hwdata->axis_map[joyaxe];
                        // scaleaxe
                        v = (Sint32)hid_get_data(REP_BUF_DATA(rep), &hitem);
                        v = (((SDL_JOYSTICK_AXIS_MAX - SDL_JOYSTICK_AXIS_MIN) * (v - hitem.logical_minimum)) / (hitem.logical_maximum - hitem.logical_minimum)) + SDL_JOYSTICK_AXIS_MIN;
                        SDL_SendJoystickAxis(timestamp, joy, naxe, v);
                    } else if (usage == HUG_HAT_SWITCH) {
                        v = (Sint32)hid_get_data(REP_BUF_DATA(rep), &hitem);
                        SDL_SendJoystickHat(timestamp, joy, 0,
                                               hatval_to_sdl(v) -
                                                   hitem.logical_minimum);
                    }
#ifdef SDL_PLATFORM_OPENBSD
                    /* here D-pad directions are reported like separate buttons.
                     * calculate the SDL hat value from the 4 separate values.
                     */
                    switch (usage) {
                    case HUG_DPAD_UP:
                        dpad[0] = (Sint32)hid_get_data(REP_BUF_DATA(rep), &hitem);
                        break;
                    case HUG_DPAD_DOWN:
                        dpad[1] = (Sint32)hid_get_data(REP_BUF_DATA(rep), &hitem);
                        break;
                    case HUG_DPAD_RIGHT:
                        dpad[2] = (Sint32)hid_get_data(REP_BUF_DATA(rep), &hitem);
                        break;
                    case HUG_DPAD_LEFT:
                        dpad[3] = (Sint32)hid_get_data(REP_BUF_DATA(rep), &hitem);
                        break;
                    //default:
                        // no-op
                    }
                    SDL_PrivateJoystickHat(joy, 0, (dpad[0] * HAT_UP) |
                                                   (dpad[1] * HAT_DOWN) |
                                                   (dpad[2] * HAT_RIGHT) |
                                                   (dpad[3] * HAT_LEFT) );
#endif
                    break;
                }
                case HUP_BUTTON:
                    v = (Sint32)hid_get_data(REP_BUF_DATA(rep), &hitem);
                    nbutton = HID_USAGE(hitem.usage) - 1; // SDL buttons are zero-based
                    SDL_SendJoystickButton(timestamp, joy, nbutton, (v != 0));
                    break;
                default:
                    continue;
                }
                break;
            default:
                break;
            }
        }
        hid_end_parse(hdata);
    }
}

// Function to close a joystick after use
static void BSD_JoystickClose(SDL_Joystick *joy)
{
    if (joy->hwdata) {
        FreeHwData(joy->hwdata);
        joy->hwdata = NULL;
    }
}

static void BSD_JoystickQuit(void)
{
    SDL_joylist_item *item = NULL;
    SDL_joylist_item *next = NULL;

    for (item = SDL_joylist; item; item = next) {
        next = item->next;
        FreeJoylistItem(item);
    }

    SDL_joylist = SDL_joylist_tail = NULL;

    numjoysticks = 0;
}

static bool report_alloc(struct report *r, struct report_desc *rd, int repind)
{
    int len;

#ifdef __DragonFly__
    len = hid_report_size(rd, repinfo[repind].kind, r->rid);
#elif defined(SDL_PLATFORM_FREEBSD)
#if (__FreeBSD_kernel_version >= 460000) || defined(__FreeBSD_kernel__)
#if (__FreeBSD_kernel_version <= 500111)
    len = hid_report_size(rd, r->rid, repinfo[repind].kind);
#else
    len = hid_report_size(rd, repinfo[repind].kind, r->rid);
#endif
#else
    len = hid_report_size(rd, repinfo[repind].kind, &r->rid);
#endif
#else
#ifdef USBHID_NEW
    len = hid_report_size(rd, repinfo[repind].kind, r->rid);
#else
    len = hid_report_size(rd, repinfo[repind].kind, &r->rid);
#endif
#endif

    if (len < 0) {
        return SDL_SetError("Negative HID report size");
    }
    r->size = len;

    if (r->size > 0) {
#if defined(SDL_PLATFORM_FREEBSD) && (__FreeBSD_kernel_version > 900000) || defined(__DragonFly__)
        r->buf = SDL_malloc(r->size);
#else
        r->buf = SDL_malloc(sizeof(*r->buf) - sizeof(REP_BUF_DATA(r)) +
                            r->size);
#endif
        if (!r->buf) {
            return false;
        }
    } else {
        r->buf = NULL;
    }

    r->status = SREPORT_CLEAN;
    return true;
}

static void report_free(struct report *r)
{
    SDL_free(r->buf);
    r->status = SREPORT_UNINIT;
}

static bool BSD_JoystickRumble(SDL_Joystick *joystick, Uint16 low_frequency_rumble, Uint16 high_frequency_rumble)
{
    return SDL_Unsupported();
}

static bool BSD_JoystickRumbleTriggers(SDL_Joystick *joystick, Uint16 left_rumble, Uint16 right_rumble)
{
    return SDL_Unsupported();
}

static bool BSD_JoystickGetGamepadMapping(int device_index, SDL_GamepadMapping *out)
{
    return false;
}

static bool BSD_JoystickSetLED(SDL_Joystick *joystick, Uint8 red, Uint8 green, Uint8 blue)
{
    return SDL_Unsupported();
}

static bool BSD_JoystickSendEffect(SDL_Joystick *joystick, const void *data, int size)
{
    return SDL_Unsupported();
}

static bool BSD_JoystickSetSensorsEnabled(SDL_Joystick *joystick, bool enabled)
{
    return SDL_Unsupported();
}

SDL_JoystickDriver SDL_BSD_JoystickDriver = {
    BSD_JoystickInit,
    BSD_JoystickGetCount,
    BSD_JoystickDetect,
    BSD_JoystickIsDevicePresent,
    BSD_JoystickGetDeviceName,
    BSD_JoystickGetDevicePath,
    BSD_JoystickGetDeviceSteamVirtualGamepadSlot,
    BSD_JoystickGetDevicePlayerIndex,
    BSD_JoystickSetDevicePlayerIndex,
    BSD_JoystickGetDeviceGUID,
    BSD_JoystickGetDeviceInstanceID,
    BSD_JoystickOpen,
    BSD_JoystickRumble,
    BSD_JoystickRumbleTriggers,
    BSD_JoystickSetLED,
    BSD_JoystickSendEffect,
    BSD_JoystickSetSensorsEnabled,
    BSD_JoystickUpdate,
    BSD_JoystickClose,
    BSD_JoystickQuit,
    BSD_JoystickGetGamepadMapping
};

#endif // SDL_JOYSTICK_USBHID
