/*******************************************************
 HIDAPI - Multi-Platform library for
 communication with HID devices.

 libusb/hidapi Team

 Copyright 2022, All Rights Reserved.

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

 * Since version 0.12.0, @ref HID_API_VERSION >= HID_API_MAKE_VERSION(0, 12, 0)
 */

#ifndef HIDAPI_DARWIN_H__
#define HIDAPI_DARWIN_H__

#include <stdint.h>

#include "../hidapi/hidapi.h"

#ifdef __cplusplus
extern "C" {
#endif

		/** @brief Get the location ID for a HID device.

			Since version 0.12.0, @ref HID_API_VERSION >= HID_API_MAKE_VERSION(0, 12, 0)

			@ingroup API
			@param dev A device handle returned from hid_open().
			@param location_id The device's location ID on return.

			@returns
				This function returns 0 on success and -1 on error.
		*/
		int HID_API_EXPORT_CALL hid_darwin_get_location_id(hid_device *dev, uint32_t *location_id);


		/** @brief Changes the behavior of all further calls to @ref hid_open or @ref hid_open_path.

			By default on Darwin platform all devices opened by HIDAPI with @ref hid_open or @ref hid_open_path
			are opened in exclusive mode (see kIOHIDOptionsTypeSeizeDevice).

			Since version 0.12.0, @ref HID_API_VERSION >= HID_API_MAKE_VERSION(0, 12, 0)

			@ingroup API
			@param open_exclusive When set to 0 - all further devices will be opened
				in non-exclusive mode. Otherwise - all further devices will be opened
				in exclusive mode.

			@note During the initialisation by @ref hid_init - this property is set to 1 (TRUE).
			This is done to preserve full backward compatibility with previous behavior.

			@note Calling this function before @ref hid_init or after @ref hid_exit has no effect.
		*/
		void HID_API_EXPORT_CALL hid_darwin_set_open_exclusive(int open_exclusive);

		/** @brief Getter for option set by @ref hid_darwin_set_open_exclusive.

			Since version 0.12.0, @ref HID_API_VERSION >= HID_API_MAKE_VERSION(0, 12, 0)

			@ingroup API
			@return 1 if all further devices will be opened in exclusive mode.

			@note Value returned by this function before calling to @ref hid_init or after @ref hid_exit
			is not reliable.
		*/
		int HID_API_EXPORT_CALL hid_darwin_get_open_exclusive(void);

		/** @brief Check how the device was opened.

			Since version 0.12.0, @ref HID_API_VERSION >= HID_API_MAKE_VERSION(0, 12, 0)

			@ingroup API
			@param dev A device to get property from.

			@return 1 if the device is opened in exclusive mode, 0 - opened in non-exclusive,
			-1 - if dev is invalid.
		*/
		int HID_API_EXPORT_CALL hid_darwin_is_device_open_exclusive(hid_device *dev);

#ifdef __cplusplus
}
#endif

#endif
