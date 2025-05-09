/*******************************************************
 HIDAPI - Multi-Platform library for
 communication with HID devices.

 libusb/hidapi Team

 Copyright 2021, All Rights Reserved.

 At the discretion of the user of this library,
 this software may be licensed under the terms of the
 GNU General Public License v3, a BSD-Style license, or the
 original HIDAPI license as outlined in the LICENSE.txt,
 LICENSE-gpl3.txt, LICENSE-bsd.txt, and LICENSE-orig.txt
 files located at the root of the source distribution.
 These files may also be found in the public source
 code repository located at:
        https://github.com/libusb/hidapi .
********************************************************/

/** @file
 * @defgroup API hidapi API

 * Since version 0.11.0, @ref HID_API_VERSION >= HID_API_MAKE_VERSION(0, 11, 0).
 */

#ifndef HIDAPI_LIBUSB_H__
#define HIDAPI_LIBUSB_H__

#include <stdint.h>

#include "../hidapi/hidapi.h"

#ifdef __cplusplus
extern "C" {
#endif

		/** @brief Open a HID device using libusb_wrap_sys_device.
			See https://libusb.sourceforge.io/api-1.0/group__libusb__dev.html#ga98f783e115ceff4eaf88a60e6439563c,
			for details on libusb_wrap_sys_device.

			@ingroup API
			@param sys_dev Platform-specific file descriptor that can be recognised by libusb.
			@param interface_num USB interface number of the device to be used as HID interface.
			Pass -1 to select first HID interface of the device.

			@returns
				This function returns a pointer to a #hid_device object on
				success or NULL on failure.
		*/
		HID_API_EXPORT hid_device * HID_API_CALL hid_libusb_wrap_sys_device(intptr_t sys_dev, int interface_num);

#ifdef __cplusplus
}
#endif

#endif
