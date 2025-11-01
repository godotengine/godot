/*******************************************************
 HIDAPI - Multi-Platform library for
 communication with HID devices.

 Alan Ott
 Signal 11 Software

 libusb/hidapi Team

 Copyright 2023, All Rights Reserved.

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
 */

#ifndef HIDAPI_H__
#define HIDAPI_H__

#include <wchar.h>

/* #480: this is to be refactored properly for v1.0 */
#ifdef _WIN32
   #ifndef HID_API_NO_EXPORT_DEFINE
      #define HID_API_EXPORT __declspec(dllexport)
   #endif
#endif
#ifndef HID_API_EXPORT
   #define HID_API_EXPORT /**< API export macro */
#endif
/* To be removed in v1.0 */
#define HID_API_CALL /**< API call macro */

#define HID_API_EXPORT_CALL HID_API_EXPORT HID_API_CALL /**< API export and call macro*/

/** @brief Static/compile-time major version of the library.

	@ingroup API
*/
#define HID_API_VERSION_MAJOR 0
/** @brief Static/compile-time minor version of the library.

	@ingroup API
*/
#define HID_API_VERSION_MINOR 14
/** @brief Static/compile-time patch version of the library.

	@ingroup API
*/
#define HID_API_VERSION_PATCH 0

/* Helper macros */
#define HID_API_AS_STR_IMPL(x) #x
#define HID_API_AS_STR(x) HID_API_AS_STR_IMPL(x)
#define HID_API_TO_VERSION_STR(v1, v2, v3) HID_API_AS_STR(v1.v2.v3)

/** @brief Coverts a version as Major/Minor/Patch into a number:
	<8 bit major><16 bit minor><8 bit patch>.

	This macro was added in version 0.12.0.

	Convenient function to be used for compile-time checks, like:
	@code{.c}
	#if HID_API_VERSION >= HID_API_MAKE_VERSION(0, 12, 0)
	@endcode

	@ingroup API
*/
#define HID_API_MAKE_VERSION(mj, mn, p) (((mj) << 24) | ((mn) << 8) | (p))

/** @brief Static/compile-time version of the library.

	This macro was added in version 0.12.0.

	@see @ref HID_API_MAKE_VERSION.

	@ingroup API
*/
#define HID_API_VERSION HID_API_MAKE_VERSION(HID_API_VERSION_MAJOR, HID_API_VERSION_MINOR, HID_API_VERSION_PATCH)

/** @brief Static/compile-time string version of the library.

	@ingroup API
*/
#define HID_API_VERSION_STR HID_API_TO_VERSION_STR(HID_API_VERSION_MAJOR, HID_API_VERSION_MINOR, HID_API_VERSION_PATCH)

/** @brief Maximum expected HID Report descriptor size in bytes.

	Since version 0.13.0, @ref HID_API_VERSION >= HID_API_MAKE_VERSION(0, 13, 0)

	@ingroup API
*/
#define HID_API_MAX_REPORT_DESCRIPTOR_SIZE 4096

#ifdef __cplusplus
extern "C" {
#endif
#ifndef DEFINED_HID_TYPES
#define DEFINED_HID_TYPES
		/** A structure to hold the version numbers. */
		struct hid_api_version {
			int major; /**< major version number */
			int minor; /**< minor version number */
			int patch; /**< patch version number */
		};

		struct hid_device_;
		typedef struct hid_device_ hid_device; /**< opaque hidapi structure */

		/** @brief HID underlying bus types.

			@ingroup API
		*/
		typedef enum {
			/** Unknown bus type */
			HID_API_BUS_UNKNOWN = 0x00,

			/** USB bus
			   Specifications:
			   https://usb.org/hid */
			HID_API_BUS_USB = 0x01,

			/** Bluetooth or Bluetooth LE bus
			   Specifications:
			   https://www.bluetooth.com/specifications/specs/human-interface-device-profile-1-1-1/
			   https://www.bluetooth.com/specifications/specs/hid-service-1-0/
			   https://www.bluetooth.com/specifications/specs/hid-over-gatt-profile-1-0/ */
			HID_API_BUS_BLUETOOTH = 0x02,

			/** I2C bus
			   Specifications:
			   https://docs.microsoft.com/previous-versions/windows/hardware/design/dn642101(v=vs.85) */
			HID_API_BUS_I2C = 0x03,

			/** SPI bus
			   Specifications:
			   https://www.microsoft.com/download/details.aspx?id=103325 */
			HID_API_BUS_SPI = 0x04,
		} hid_bus_type;

		/** hidapi info structure */
		struct hid_device_info {
			/** Platform-specific device path */
			char *path;
			/** Device Vendor ID */
			unsigned short vendor_id;
			/** Device Product ID */
			unsigned short product_id;
			/** Serial Number */
			wchar_t *serial_number;
			/** Device Release Number in binary-coded decimal,
			    also known as Device Version Number */
			unsigned short release_number;
			/** Manufacturer String */
			wchar_t *manufacturer_string;
			/** Product string */
			wchar_t *product_string;
			/** Usage Page for this Device/Interface
			    (Windows/Mac/hidraw only) */
			unsigned short usage_page;
			/** Usage for this Device/Interface
			    (Windows/Mac/hidraw only) */
			unsigned short usage;
			/** The USB interface which this logical device
			    represents.

			    Valid only if the device is a USB HID device.
			    Set to -1 in all other cases.
			*/
			int interface_number;

			/** Pointer to the next device */
			struct hid_device_info *next;

			/** Underlying bus type
			    Since version 0.13.0, @ref HID_API_VERSION >= HID_API_MAKE_VERSION(0, 13, 0)
			*/
			hid_bus_type bus_type;

			/** Additional information about the USB interface.
			    (libusb only) */
			int interface_class;
			int interface_subclass;
			int interface_protocol;
		};

#endif /* DEFINED_HID_TYPES */


		/** @brief Initialize the HIDAPI library.

			This function initializes the HIDAPI library. Calling it is not
			strictly necessary, as it will be called automatically by
			hid_enumerate() and any of the hid_open_*() functions if it is
			needed.  This function should be called at the beginning of
			execution however, if there is a chance of HIDAPI handles
			being opened by different threads simultaneously.

			@ingroup API

			@returns
				This function returns 0 on success and -1 on error.
				Call hid_error(NULL) to get the failure reason.
		*/
		int HID_API_EXPORT HID_API_CALL hid_init(void);

		/** @brief Finalize the HIDAPI library.

			This function frees all of the static data associated with
			HIDAPI. It should be called at the end of execution to avoid
			memory leaks.

			@ingroup API

			@returns
				This function returns 0 on success and -1 on error.
		*/
		int HID_API_EXPORT HID_API_CALL hid_exit(void);

		/** @brief Enumerate the HID Devices.

			This function returns a linked list of all the HID devices
			attached to the system which match vendor_id and product_id.
			If @p vendor_id is set to 0 then any vendor matches.
			If @p product_id is set to 0 then any product matches.
			If @p vendor_id and @p product_id are both set to 0, then
			all HID devices will be returned.

			@ingroup API
			@param vendor_id The Vendor ID (VID) of the types of device
				to open.
			@param product_id The Product ID (PID) of the types of
				device to open.

			@returns
				This function returns a pointer to a linked list of type
				struct #hid_device_info, containing information about the HID devices
				attached to the system,
				or NULL in the case of failure or if no HID devices present in the system.
				Call hid_error(NULL) to get the failure reason.

			@note The returned value by this function must to be freed by calling hid_free_enumeration(),
			      when not needed anymore.
		*/
		struct hid_device_info HID_API_EXPORT * HID_API_CALL hid_enumerate(unsigned short vendor_id, unsigned short product_id);

		/** @brief Free an enumeration Linked List

			This function frees a linked list created by hid_enumerate().

			@ingroup API
			@param devs Pointer to a list of struct_device returned from
			            hid_enumerate().
		*/
		void  HID_API_EXPORT HID_API_CALL hid_free_enumeration(struct hid_device_info *devs);

		/** @brief Open a HID device using a Vendor ID (VID), Product ID
			(PID) and optionally a serial number.

			If @p serial_number is NULL, the first device with the
			specified VID and PID is opened.

			@ingroup API
			@param vendor_id The Vendor ID (VID) of the device to open.
			@param product_id The Product ID (PID) of the device to open.
			@param serial_number The Serial Number of the device to open
			                     (Optionally NULL).

			@returns
				This function returns a pointer to a #hid_device object on
				success or NULL on failure.
				Call hid_error(NULL) to get the failure reason.

			@note The returned object must be freed by calling hid_close(),
			      when not needed anymore.
		*/
		HID_API_EXPORT hid_device * HID_API_CALL hid_open(unsigned short vendor_id, unsigned short product_id, const wchar_t *serial_number);

		/** @brief Open a HID device by its path name.

			The path name be determined by calling hid_enumerate(), or a
			platform-specific path name can be used (eg: /dev/hidraw0 on
			Linux).

			@ingroup API
			@param path The path name of the device to open

			@returns
				This function returns a pointer to a #hid_device object on
				success or NULL on failure.
				Call hid_error(NULL) to get the failure reason.

			@note The returned object must be freed by calling hid_close(),
			      when not needed anymore.
		*/
		HID_API_EXPORT hid_device * HID_API_CALL hid_open_path(const char *path);

		/** @brief Write an Output report to a HID device.

			The first byte of @p data[] must contain the Report ID. For
			devices which only support a single report, this must be set
			to 0x0. The remaining bytes contain the report data. Since
			the Report ID is mandatory, calls to hid_write() will always
			contain one more byte than the report contains. For example,
			if a hid report is 16 bytes long, 17 bytes must be passed to
			hid_write(), the Report ID (or 0x0, for devices with a
			single report), followed by the report data (16 bytes). In
			this example, the length passed in would be 17.

			hid_write() will send the data on the first OUT endpoint, if
			one exists. If it does not, it will send the data through
			the Control Endpoint (Endpoint 0).

			@ingroup API
			@param dev A device handle returned from hid_open().
			@param data The data to send, including the report number as
				the first byte.
			@param length The length in bytes of the data to send.

			@returns
				This function returns the actual number of bytes written and
				-1 on error.
				Call hid_error(dev) to get the failure reason.
		*/
		int  HID_API_EXPORT HID_API_CALL hid_write(hid_device *dev, const unsigned char *data, size_t length);

		/** @brief Read an Input report from a HID device with timeout.

			Input reports are returned
			to the host through the INTERRUPT IN endpoint. The first byte will
			contain the Report number if the device uses numbered reports.

			@ingroup API
			@param dev A device handle returned from hid_open().
			@param data A buffer to put the read data into.
			@param length The number of bytes to read. For devices with
				multiple reports, make sure to read an extra byte for
				the report number.
			@param milliseconds timeout in milliseconds or -1 for blocking wait.

			@returns
				This function returns the actual number of bytes read and
				-1 on error.
				Call hid_error(dev) to get the failure reason.
				If no packet was available to be read within
				the timeout period, this function returns 0.
		*/
		int HID_API_EXPORT HID_API_CALL hid_read_timeout(hid_device *dev, unsigned char *data, size_t length, int milliseconds);

		/** @brief Read an Input report from a HID device.

			Input reports are returned
			to the host through the INTERRUPT IN endpoint. The first byte will
			contain the Report number if the device uses numbered reports.

			@ingroup API
			@param dev A device handle returned from hid_open().
			@param data A buffer to put the read data into.
			@param length The number of bytes to read. For devices with
				multiple reports, make sure to read an extra byte for
				the report number.

			@returns
				This function returns the actual number of bytes read and
				-1 on error.
				Call hid_error(dev) to get the failure reason.
				If no packet was available to be read and
				the handle is in non-blocking mode, this function returns 0.
		*/
		int  HID_API_EXPORT HID_API_CALL hid_read(hid_device *dev, unsigned char *data, size_t length);

		/** @brief Set the device handle to be non-blocking.

			In non-blocking mode calls to hid_read() will return
			immediately with a value of 0 if there is no data to be
			read. In blocking mode, hid_read() will wait (block) until
			there is data to read before returning.

			Nonblocking can be turned on and off at any time.

			@ingroup API
			@param dev A device handle returned from hid_open().
			@param nonblock enable or not the nonblocking reads
			 - 1 to enable nonblocking
			 - 0 to disable nonblocking.

			@returns
				This function returns 0 on success and -1 on error.
				Call hid_error(dev) to get the failure reason.
		*/
		int  HID_API_EXPORT HID_API_CALL hid_set_nonblocking(hid_device *dev, int nonblock);

		/** @brief Send a Feature report to the device.

			Feature reports are sent over the Control endpoint as a
			Set_Report transfer.  The first byte of @p data[] must
			contain the Report ID. For devices which only support a
			single report, this must be set to 0x0. The remaining bytes
			contain the report data. Since the Report ID is mandatory,
			calls to hid_send_feature_report() will always contain one
			more byte than the report contains. For example, if a hid
			report is 16 bytes long, 17 bytes must be passed to
			hid_send_feature_report(): the Report ID (or 0x0, for
			devices which do not use numbered reports), followed by the
			report data (16 bytes). In this example, the length passed
			in would be 17.

			@ingroup API
			@param dev A device handle returned from hid_open().
			@param data The data to send, including the report number as
				the first byte.
			@param length The length in bytes of the data to send, including
				the report number.

			@returns
				This function returns the actual number of bytes written and
				-1 on error.
				Call hid_error(dev) to get the failure reason.
		*/
		int HID_API_EXPORT HID_API_CALL hid_send_feature_report(hid_device *dev, const unsigned char *data, size_t length);

		/** @brief Get a feature report from a HID device.

			Set the first byte of @p data[] to the Report ID of the
			report to be read.  Make sure to allow space for this
			extra byte in @p data[]. Upon return, the first byte will
			still contain the Report ID, and the report data will
			start in data[1].

			@ingroup API
			@param dev A device handle returned from hid_open().
			@param data A buffer to put the read data into, including
				the Report ID. Set the first byte of @p data[] to the
				Report ID of the report to be read, or set it to zero
				if your device does not use numbered reports.
			@param length The number of bytes to read, including an
				extra byte for the report ID. The buffer can be longer
				than the actual report.

			@returns
				This function returns the number of bytes read plus
				one for the report ID (which is still in the first
				byte), or -1 on error.
				Call hid_error(dev) to get the failure reason.
		*/
		int HID_API_EXPORT HID_API_CALL hid_get_feature_report(hid_device *dev, unsigned char *data, size_t length);

		/** @brief Get a input report from a HID device.

			Since version 0.10.0, @ref HID_API_VERSION >= HID_API_MAKE_VERSION(0, 10, 0)

			Set the first byte of @p data[] to the Report ID of the
			report to be read. Make sure to allow space for this
			extra byte in @p data[]. Upon return, the first byte will
			still contain the Report ID, and the report data will
			start in data[1].

			@ingroup API
			@param dev A device handle returned from hid_open().
			@param data A buffer to put the read data into, including
				the Report ID. Set the first byte of @p data[] to the
				Report ID of the report to be read, or set it to zero
				if your device does not use numbered reports.
			@param length The number of bytes to read, including an
				extra byte for the report ID. The buffer can be longer
				than the actual report.

			@returns
				This function returns the number of bytes read plus
				one for the report ID (which is still in the first
				byte), or -1 on error.
				Call hid_error(dev) to get the failure reason.
		*/
		int HID_API_EXPORT HID_API_CALL hid_get_input_report(hid_device *dev, unsigned char *data, size_t length);

		/** @brief Close a HID device.

			@ingroup API
			@param dev A device handle returned from hid_open().
		*/
		void HID_API_EXPORT HID_API_CALL hid_close(hid_device *dev);

		/** @brief Get The Manufacturer String from a HID device.

			@ingroup API
			@param dev A device handle returned from hid_open().
			@param string A wide string buffer to put the data into.
			@param maxlen The length of the buffer in multiples of wchar_t.

			@returns
				This function returns 0 on success and -1 on error.
				Call hid_error(dev) to get the failure reason.
		*/
		int HID_API_EXPORT_CALL hid_get_manufacturer_string(hid_device *dev, wchar_t *string, size_t maxlen);

		/** @brief Get The Product String from a HID device.

			@ingroup API
			@param dev A device handle returned from hid_open().
			@param string A wide string buffer to put the data into.
			@param maxlen The length of the buffer in multiples of wchar_t.

			@returns
				This function returns 0 on success and -1 on error.
				Call hid_error(dev) to get the failure reason.
		*/
		int HID_API_EXPORT_CALL hid_get_product_string(hid_device *dev, wchar_t *string, size_t maxlen);

		/** @brief Get The Serial Number String from a HID device.

			@ingroup API
			@param dev A device handle returned from hid_open().
			@param string A wide string buffer to put the data into.
			@param maxlen The length of the buffer in multiples of wchar_t.

			@returns
				This function returns 0 on success and -1 on error.
				Call hid_error(dev) to get the failure reason.
		*/
		int HID_API_EXPORT_CALL hid_get_serial_number_string(hid_device *dev, wchar_t *string, size_t maxlen);

		/** @brief Get The struct #hid_device_info from a HID device.

			Since version 0.13.0, @ref HID_API_VERSION >= HID_API_MAKE_VERSION(0, 13, 0)

			@ingroup API
			@param dev A device handle returned from hid_open().

			@returns
				This function returns a pointer to the struct #hid_device_info
				for this hid_device, or NULL in the case of failure.
				Call hid_error(dev) to get the failure reason.
				This struct is valid until the device is closed with hid_close().

			@note The returned object is owned by the @p dev, and SHOULD NOT be freed by the user.
		*/
		struct hid_device_info HID_API_EXPORT * HID_API_CALL hid_get_device_info(hid_device *dev);

		/** @brief Get a string from a HID device, based on its string index.

			@ingroup API
			@param dev A device handle returned from hid_open().
			@param string_index The index of the string to get.
			@param string A wide string buffer to put the data into.
			@param maxlen The length of the buffer in multiples of wchar_t.

			@returns
				This function returns 0 on success and -1 on error.
				Call hid_error(dev) to get the failure reason.
		*/
		int HID_API_EXPORT_CALL hid_get_indexed_string(hid_device *dev, int string_index, wchar_t *string, size_t maxlen);

		/** @brief Get a report descriptor from a HID device.

			Since version 0.14.0, @ref HID_API_VERSION >= HID_API_MAKE_VERSION(0, 14, 0)

			User has to provide a preallocated buffer where descriptor will be copied to.
			The recommended size for preallocated buffer is @ref HID_API_MAX_REPORT_DESCRIPTOR_SIZE bytes.

			@ingroup API
			@param dev A device handle returned from hid_open().
			@param buf The buffer to copy descriptor into.
			@param buf_size The size of the buffer in bytes.

			@returns
				This function returns non-negative number of bytes actually copied, or -1 on error.
		*/
		int HID_API_EXPORT_CALL hid_get_report_descriptor(hid_device *dev, unsigned char *buf, size_t buf_size);

		/** @brief Get a string describing the last error which occurred.

			This function is intended for logging/debugging purposes.

			This function guarantees to never return NULL.
			If there was no error in the last function call -
			the returned string clearly indicates that.

			Any HIDAPI function that can explicitly indicate an execution failure
			(e.g. by an error code, or by returning NULL) - may set the error string,
			to be returned by this function.

			Strings returned from hid_error() must not be freed by the user,
			i.e. owned by HIDAPI library.
			Device-specific error string may remain allocated at most until hid_close() is called.
			Global error string may remain allocated at most until hid_exit() is called.

			@ingroup API
			@param dev A device handle returned from hid_open(),
			  or NULL to get the last non-device-specific error
			  (e.g. for errors in hid_open() or hid_enumerate()).

			@returns
				A string describing the last error (if any).
		*/
		HID_API_EXPORT const wchar_t* HID_API_CALL hid_error(hid_device *dev);

		/** @brief Get a runtime version of the library.

			This function is thread-safe.

			@ingroup API

			@returns
				Pointer to statically allocated struct, that contains version.
		*/
		HID_API_EXPORT const  struct hid_api_version* HID_API_CALL hid_version(void);


		/** @brief Get a runtime version string of the library.

			This function is thread-safe.

			@ingroup API

			@returns
				Pointer to statically allocated string, that contains version string.
		*/
		HID_API_EXPORT const char* HID_API_CALL hid_version_str(void);

#ifdef __cplusplus
}
#endif

#endif
