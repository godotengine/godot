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

#ifdef HAVE_LIBUSB
// libusb HIDAPI Implementation

// Include this now, for our dynamically-loaded libusb context
#include <libusb.h>

typedef struct SDL_LibUSBContext
{
/* *INDENT-OFF* */ // clang-format off
    int (LIBUSB_CALL *init)(libusb_context **ctx);
    void (LIBUSB_CALL *exit)(libusb_context *ctx);
    ssize_t (LIBUSB_CALL *get_device_list)(libusb_context *ctx, libusb_device ***list);
    void (LIBUSB_CALL *free_device_list)(libusb_device **list, int unref_devices);
    int (LIBUSB_CALL *get_device_descriptor)(libusb_device *dev, struct libusb_device_descriptor *desc);
    int (LIBUSB_CALL *get_active_config_descriptor)(libusb_device *dev,    struct libusb_config_descriptor **config);
    int (LIBUSB_CALL *get_config_descriptor)(
        libusb_device *dev,
        uint8_t config_index,
        struct libusb_config_descriptor **config
    );
    void (LIBUSB_CALL *free_config_descriptor)(struct libusb_config_descriptor *config);
    uint8_t (LIBUSB_CALL *get_bus_number)(libusb_device *dev);
#ifdef SDL_PLATFORM_FREEBSD
    int (LIBUSB_CALL *get_port_numbers)(libusb_device *dev, uint8_t *port_numbers, uint8_t port_numbers_len);
#else
    int (LIBUSB_CALL *get_port_numbers)(libusb_device *dev, uint8_t *port_numbers, int port_numbers_len);
#endif
    uint8_t (LIBUSB_CALL *get_device_address)(libusb_device *dev);
    int (LIBUSB_CALL *open)(libusb_device *dev, libusb_device_handle **dev_handle);
    void (LIBUSB_CALL *close)(libusb_device_handle *dev_handle);
    libusb_device *(LIBUSB_CALL *get_device)(libusb_device_handle *dev_handle);
    int (LIBUSB_CALL *claim_interface)(libusb_device_handle *dev_handle, int interface_number);
    int (LIBUSB_CALL *release_interface)(libusb_device_handle *dev_handle, int interface_number);
    int (LIBUSB_CALL *kernel_driver_active)(libusb_device_handle *dev_handle, int interface_number);
    int (LIBUSB_CALL *detach_kernel_driver)(libusb_device_handle *dev_handle, int interface_number);
    int (LIBUSB_CALL *attach_kernel_driver)(libusb_device_handle *dev_handle, int interface_number);
    int (LIBUSB_CALL *set_auto_detach_kernel_driver)(libusb_device_handle *dev_handle, int enable);
    int (LIBUSB_CALL *set_interface_alt_setting)(libusb_device_handle *dev, int interface_number, int alternate_setting);
    struct libusb_transfer * (LIBUSB_CALL *alloc_transfer)(int iso_packets);
    int (LIBUSB_CALL *submit_transfer)(struct libusb_transfer *transfer);
    int (LIBUSB_CALL *cancel_transfer)(struct libusb_transfer *transfer);
    void (LIBUSB_CALL *free_transfer)(struct libusb_transfer *transfer);
    int (LIBUSB_CALL *control_transfer)(
        libusb_device_handle *dev_handle,
        uint8_t request_type,
        uint8_t bRequest,
        uint16_t wValue,
        uint16_t wIndex,
        unsigned char *data,
        uint16_t wLength,
        unsigned int timeout
    );
    int (LIBUSB_CALL *interrupt_transfer)(
        libusb_device_handle *dev_handle,
        unsigned char endpoint,
        unsigned char *data,
        int length,
        int *actual_length,
        unsigned int timeout
    );
    int (LIBUSB_CALL *bulk_transfer)(
        libusb_device_handle *dev_handle,
        unsigned char endpoint,
        unsigned char *data,
        int length,
        int *transferred,
        unsigned int timeout
    );
    int (LIBUSB_CALL *handle_events)(libusb_context *ctx);
    int (LIBUSB_CALL *handle_events_completed)(libusb_context *ctx, int *completed);
    const char * (LIBUSB_CALL *error_name)(int errcode);
/* *INDENT-ON* */ // clang-format on

} SDL_LibUSBContext;

extern bool SDL_InitLibUSB(SDL_LibUSBContext **ctx);
extern void SDL_QuitLibUSB(void);

#endif // HAVE_LIBUSB
