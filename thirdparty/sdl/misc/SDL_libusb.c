/*
  Simple DirectMedia Layer
  Copyright (C) 1997-2026 Sam Lantinga <slouken@libsdl.org>

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

#include "SDL_libusb.h"

#ifdef HAVE_LIBUSB

#ifdef SDL_LIBUSB_DYNAMIC
SDL_ELF_NOTE_DLOPEN(
    "libusb",
    "Support for joysticks through libusb",
    SDL_ELF_NOTE_DLOPEN_PRIORITY_SUGGESTED,
    SDL_LIBUSB_DYNAMIC
)
#endif

static SDL_AtomicInt SDL_libusb_refcount;
static bool SDL_libusb_loaded;
static SDL_SharedObject *SDL_libusb_handle;
static SDL_LibUSBContext SDL_libusb_context;

bool SDL_InitLibUSB(SDL_LibUSBContext **ctx)
{
    if (SDL_AtomicIncRef(&SDL_libusb_refcount) == 0) {
#ifdef SDL_LIBUSB_DYNAMIC
        SDL_libusb_handle = SDL_LoadObject(SDL_LIBUSB_DYNAMIC);
        if (SDL_libusb_handle)
#endif
        {
            SDL_libusb_loaded = true;
#ifdef SDL_LIBUSB_DYNAMIC
#define LOAD_LIBUSB_SYMBOL(type, func)                                                                      \
    if ((SDL_libusb_context.func = (type)SDL_LoadFunction(SDL_libusb_handle, "libusb_" #func)) == NULL) {   \
        SDL_libusb_loaded = false;                                                                          \
    }
#else
#define LOAD_LIBUSB_SYMBOL(type, func) \
    SDL_libusb_context.func = libusb_##func;
#endif
            LOAD_LIBUSB_SYMBOL(int (LIBUSB_CALL *)(libusb_context **), init)
            LOAD_LIBUSB_SYMBOL(void (LIBUSB_CALL *)(libusb_context *), exit)
            LOAD_LIBUSB_SYMBOL(ssize_t (LIBUSB_CALL *)(libusb_context *, libusb_device ***), get_device_list)
            LOAD_LIBUSB_SYMBOL(void (LIBUSB_CALL *)(libusb_device **, int), free_device_list)
            LOAD_LIBUSB_SYMBOL(int (LIBUSB_CALL *)(libusb_device *, struct libusb_device_descriptor *), get_device_descriptor)
            LOAD_LIBUSB_SYMBOL(int (LIBUSB_CALL *)(libusb_device *, struct libusb_config_descriptor **), get_active_config_descriptor)
            LOAD_LIBUSB_SYMBOL(int (LIBUSB_CALL *)(libusb_device *, uint8_t, struct libusb_config_descriptor **), get_config_descriptor)
            LOAD_LIBUSB_SYMBOL(void (LIBUSB_CALL *)(struct libusb_config_descriptor *), free_config_descriptor)
            LOAD_LIBUSB_SYMBOL(uint8_t (LIBUSB_CALL *)(libusb_device *), get_bus_number)
#ifdef SDL_PLATFORM_FREEBSD
            LOAD_LIBUSB_SYMBOL(int (LIBUSB_CALL *)(libusb_device *dev, uint8_t *port_numbers, uint8_t port_numbers_len), get_port_numbers)
#else
            LOAD_LIBUSB_SYMBOL(int (LIBUSB_CALL *)(libusb_device *dev, uint8_t *port_numbers, int port_numbers_len), get_port_numbers)
#endif
            LOAD_LIBUSB_SYMBOL(uint8_t (LIBUSB_CALL *)(libusb_device *), get_device_address)
            LOAD_LIBUSB_SYMBOL(int (LIBUSB_CALL *)(libusb_device *, libusb_device_handle **), open)
            LOAD_LIBUSB_SYMBOL(void (LIBUSB_CALL *)(libusb_device_handle *), close)
            LOAD_LIBUSB_SYMBOL(libusb_device * (LIBUSB_CALL *)(libusb_device_handle *dev_handle), get_device)
            LOAD_LIBUSB_SYMBOL(int (LIBUSB_CALL *)(libusb_device_handle *, int), claim_interface)
            LOAD_LIBUSB_SYMBOL(int (LIBUSB_CALL *)(libusb_device_handle *, int), release_interface)
            LOAD_LIBUSB_SYMBOL(int (LIBUSB_CALL *)(libusb_device_handle *, int), kernel_driver_active)
            LOAD_LIBUSB_SYMBOL(int (LIBUSB_CALL *)(libusb_device_handle *, int), detach_kernel_driver)
            LOAD_LIBUSB_SYMBOL(int (LIBUSB_CALL *)(libusb_device_handle *, int), attach_kernel_driver)
            LOAD_LIBUSB_SYMBOL(int (LIBUSB_CALL *)(libusb_device_handle *, int), set_auto_detach_kernel_driver)
            LOAD_LIBUSB_SYMBOL(int (LIBUSB_CALL *)(libusb_device_handle *, int, int), set_interface_alt_setting)
            LOAD_LIBUSB_SYMBOL(struct libusb_transfer * (LIBUSB_CALL *)(int), alloc_transfer)
            LOAD_LIBUSB_SYMBOL(int (LIBUSB_CALL *)(struct libusb_transfer *), submit_transfer)
            LOAD_LIBUSB_SYMBOL(int (LIBUSB_CALL *)(struct libusb_transfer *), cancel_transfer)
            LOAD_LIBUSB_SYMBOL(void (LIBUSB_CALL *)(struct libusb_transfer *), free_transfer)
            LOAD_LIBUSB_SYMBOL(int (LIBUSB_CALL *)(libusb_device_handle *, uint8_t, uint8_t, uint16_t, uint16_t, unsigned char *, uint16_t, unsigned int), control_transfer)
            LOAD_LIBUSB_SYMBOL(int (LIBUSB_CALL *)(libusb_device_handle *, unsigned char, unsigned char *, int, int *, unsigned int), interrupt_transfer)
            LOAD_LIBUSB_SYMBOL(int (LIBUSB_CALL *)(libusb_device_handle *, unsigned char, unsigned char *, int, int *, unsigned int), bulk_transfer)
            LOAD_LIBUSB_SYMBOL(int (LIBUSB_CALL *)(libusb_context *), handle_events)
            LOAD_LIBUSB_SYMBOL(int (LIBUSB_CALL *)(libusb_context *, int *), handle_events_completed)
            LOAD_LIBUSB_SYMBOL(const char * (LIBUSB_CALL *)(int), error_name)
#undef LOAD_LIBUSB_SYMBOL
        }
    }

    if (SDL_libusb_loaded) {
        *ctx = &SDL_libusb_context;
        return true;
    } else {
        SDL_QuitLibUSB();
        *ctx = NULL;
        return false;
    }
}

void SDL_QuitLibUSB(void)
{
    if (SDL_AtomicDecRef(&SDL_libusb_refcount)) {
        if (SDL_libusb_handle) {
            SDL_UnloadObject(SDL_libusb_handle);
            SDL_libusb_handle = NULL;
        }
        SDL_libusb_loaded = false;
    }
}

#endif // HAVE_LIBUSB
