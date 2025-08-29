/*******************************************************
 HIDAPI - Multi-Platform library for
 communication with HID devices.

 Alan Ott
 Signal 11 Software

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

#if defined(_MSC_VER) && !defined(_CRT_SECURE_NO_WARNINGS)
/* Do not warn about wcsncpy usage.
   https://docs.microsoft.com/cpp/c-runtime-library/security-features-in-the-crt */
#define _CRT_SECURE_NO_WARNINGS
#endif

#ifdef __cplusplus
extern "C" {
#endif

#include "hidapi_winapi.h"

#include <windows.h>

#ifndef _NTDEF_
typedef LONG NTSTATUS;
#endif

#ifndef WC_ERR_INVALID_CHARS
#define WC_ERR_INVALID_CHARS 0x00000080
#endif
#ifndef _WIN32_WINNT_WIN8
#define _WIN32_WINNT_WIN8 0x0602
#endif

#ifdef __CYGWIN__
#include <ntdef.h>
#include <wctype.h>
#define _wcsdup wcsdup
#endif

/*#define HIDAPI_USE_DDK*/

#include "hidapi_cfgmgr32.h"
#include "hidapi_hidclass.h"
#include "hidapi_hidsdi.h"

#ifndef HIDAPI_USING_SDL_RUNTIME
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#endif

#ifdef MIN
#undef MIN
#endif
#define MIN(x,y) ((x) < (y)? (x): (y))

/* MAXIMUM_USB_STRING_LENGTH from usbspec.h is 255 */
/* BLUETOOTH_DEVICE_NAME_SIZE from bluetoothapis.h is 256 */
#define MAX_STRING_WCHARS 256

/* For certain USB devices, using a buffer larger or equal to 127 wchars results
   in successful completion of HID API functions, but a broken string is stored
   in the output buffer. This behaviour persists even if HID API is bypassed and
   HID IOCTLs are passed to the HID driver directly. Therefore, for USB devices,
   the buffer MUST NOT exceed 126 WCHARs.
*/

#define MAX_STRING_WCHARS_USB 126

static struct hid_api_version api_version = {
	.major = HID_API_VERSION_MAJOR,
	.minor = HID_API_VERSION_MINOR,
	.patch = HID_API_VERSION_PATCH
};

#ifndef HIDAPI_USE_DDK
/* Since we're not building with the DDK, and the HID header
   files aren't part of the Windows SDK, we define what we need ourselves.
   In lookup_functions(), the function pointers
   defined below are set. */

static HidD_GetHidGuid_ HidD_GetHidGuid;
static HidD_GetAttributes_ HidD_GetAttributes;
static HidD_GetSerialNumberString_ HidD_GetSerialNumberString;
static HidD_GetManufacturerString_ HidD_GetManufacturerString;
static HidD_GetProductString_ HidD_GetProductString;
static HidD_SetFeature_ HidD_SetFeature;
static HidD_GetFeature_ HidD_GetFeature;
static HidD_GetInputReport_ HidD_GetInputReport;
static HidD_GetIndexedString_ HidD_GetIndexedString;
static HidD_GetPreparsedData_ HidD_GetPreparsedData;
static HidD_FreePreparsedData_ HidD_FreePreparsedData;
static HidP_GetCaps_ HidP_GetCaps;
static HidD_SetNumInputBuffers_ HidD_SetNumInputBuffers;
static HidD_SetOutputReport_ HidD_SetOutputReport;

static CM_Locate_DevNodeW_ CM_Locate_DevNodeW = NULL;
static CM_Get_Parent_ CM_Get_Parent = NULL;
static CM_Get_DevNode_PropertyW_ CM_Get_DevNode_PropertyW = NULL;
static CM_Get_Device_Interface_PropertyW_ CM_Get_Device_Interface_PropertyW = NULL;
static CM_Get_Device_Interface_List_SizeW_ CM_Get_Device_Interface_List_SizeW = NULL;
static CM_Get_Device_Interface_ListW_ CM_Get_Device_Interface_ListW = NULL;

static HMODULE hid_lib_handle = NULL;
static HMODULE cfgmgr32_lib_handle = NULL;
static BOOLEAN hidapi_initialized = FALSE;

static void free_library_handles(void)
{
	if (hid_lib_handle)
		FreeLibrary(hid_lib_handle);
	hid_lib_handle = NULL;
	if (cfgmgr32_lib_handle)
		FreeLibrary(cfgmgr32_lib_handle);
	cfgmgr32_lib_handle = NULL;
}

#if defined(__GNUC__) && __GNUC__ >= 8
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wcast-function-type"
#endif

static int lookup_functions(void)
{
	hid_lib_handle = LoadLibraryW(L"hid.dll");
	if (hid_lib_handle == NULL) {
		goto err;
	}

	cfgmgr32_lib_handle = LoadLibraryW(L"cfgmgr32.dll");
	if (cfgmgr32_lib_handle == NULL) {
		goto err;
	}

#define RESOLVE(lib_handle, x) x = (x##_)GetProcAddress(lib_handle, #x); if (!x) goto err;

	RESOLVE(hid_lib_handle, HidD_GetHidGuid);
	RESOLVE(hid_lib_handle, HidD_GetAttributes);
	RESOLVE(hid_lib_handle, HidD_GetSerialNumberString);
	RESOLVE(hid_lib_handle, HidD_GetManufacturerString);
	RESOLVE(hid_lib_handle, HidD_GetProductString);
	RESOLVE(hid_lib_handle, HidD_SetFeature);
	RESOLVE(hid_lib_handle, HidD_GetFeature);
	RESOLVE(hid_lib_handle, HidD_GetInputReport);
	RESOLVE(hid_lib_handle, HidD_GetIndexedString);
	RESOLVE(hid_lib_handle, HidD_GetPreparsedData);
	RESOLVE(hid_lib_handle, HidD_FreePreparsedData);
	RESOLVE(hid_lib_handle, HidP_GetCaps);
	RESOLVE(hid_lib_handle, HidD_SetNumInputBuffers);
	RESOLVE(hid_lib_handle, HidD_SetOutputReport);

	RESOLVE(cfgmgr32_lib_handle, CM_Locate_DevNodeW);
	RESOLVE(cfgmgr32_lib_handle, CM_Get_Parent);
	RESOLVE(cfgmgr32_lib_handle, CM_Get_DevNode_PropertyW);
	RESOLVE(cfgmgr32_lib_handle, CM_Get_Device_Interface_PropertyW);
	RESOLVE(cfgmgr32_lib_handle, CM_Get_Device_Interface_List_SizeW);
	RESOLVE(cfgmgr32_lib_handle, CM_Get_Device_Interface_ListW);

#undef RESOLVE

	return 0;

err:
	free_library_handles();
	return -1;
}

#if defined(__GNUC__) && __GNUC__ >= 8
# pragma GCC diagnostic pop
#endif

#endif /* HIDAPI_USE_DDK */

struct hid_device_ {
		HANDLE device_handle;
		BOOL blocking;
		USHORT output_report_length;
		unsigned char *write_buf;
		size_t input_report_length;
		USHORT feature_report_length;
		unsigned char *feature_buf;
		wchar_t *last_error_str;
		BOOL read_pending;
		char *read_buf;
		OVERLAPPED ol;
		OVERLAPPED write_ol;
		struct hid_device_info* device_info;
		BOOL use_hid_write_output_report;
};

static BOOL IsWindowsVersionOrGreater(WORD wMajorVersion, WORD wMinorVersion, WORD wServicePackMajor)
{
	OSVERSIONINFOEXW osvi;
	DWORDLONG const dwlConditionMask = VerSetConditionMask(
		VerSetConditionMask(
			VerSetConditionMask(
				0, VER_MAJORVERSION, VER_GREATER_EQUAL ),
			VER_MINORVERSION, VER_GREATER_EQUAL ),
		VER_SERVICEPACKMAJOR, VER_GREATER_EQUAL );

	memset(&osvi, 0, sizeof(osvi));
	osvi.dwOSVersionInfoSize = sizeof( osvi );
	osvi.dwMajorVersion = wMajorVersion;
	osvi.dwMinorVersion = wMinorVersion;
	osvi.wServicePackMajor = wServicePackMajor;

	return VerifyVersionInfoW(&osvi, VER_MAJORVERSION | VER_MINORVERSION | VER_SERVICEPACKMAJOR, dwlConditionMask) != FALSE;
}

static hid_device *new_hid_device(void)
{
	hid_device *dev = (hid_device*) calloc(1, sizeof(hid_device));

	if (dev == NULL) {
		return NULL;
	}

	dev->device_handle = INVALID_HANDLE_VALUE;
	dev->blocking = TRUE;
	dev->output_report_length = 0;
	dev->write_buf = NULL;
	dev->input_report_length = 0;
	dev->feature_report_length = 0;
	dev->feature_buf = NULL;
	dev->last_error_str = NULL;
	dev->read_pending = FALSE;
	dev->read_buf = NULL;
	memset(&dev->ol, 0, sizeof(dev->ol));
	dev->ol.hEvent = CreateEvent(NULL, FALSE, FALSE /*initial state f=nonsignaled*/, NULL);
	memset(&dev->write_ol, 0, sizeof(dev->write_ol));
	dev->write_ol.hEvent = CreateEvent(NULL, FALSE, FALSE /*initial state f=nonsignaled*/, NULL);
	dev->device_info = NULL;

	return dev;
}

static void free_hid_device(hid_device *dev)
{
	CloseHandle(dev->ol.hEvent);
	CloseHandle(dev->write_ol.hEvent);
	CloseHandle(dev->device_handle);
	free(dev->last_error_str);
	dev->last_error_str = NULL;
	free(dev->write_buf);
	free(dev->feature_buf);
	free(dev->read_buf);
	hid_free_enumeration(dev->device_info);
	free(dev);
}

static void register_winapi_error_to_buffer(wchar_t **error_buffer, const WCHAR *op)
{
	WCHAR system_err_buf[1024];
	DWORD error_code = GetLastError();

	free(*error_buffer);
	*error_buffer = NULL;

#ifdef HIDAPI_USING_SDL_RUNTIME
	/* Thread-safe error handling */
	SDL_ClearError();
#endif

	/* Only clear out error messages if NULL is passed into op */
	if (!op) {
		return;
	}

	DWORD system_err_len = FormatMessageW(
		FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
		NULL,
		error_code,
		MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
		system_err_buf, ARRAYSIZE(system_err_buf),
		NULL);

	DWORD op_len = (DWORD)wcslen(op);

	DWORD op_prefix_len =
		op_len
		+ 15 /*: (0x00000000) */
		;
	DWORD msg_len =
		+ op_prefix_len
		+ system_err_len
		;

	WCHAR *msg = (WCHAR *)calloc(msg_len + 1, sizeof (WCHAR));

	if (!msg)
		return;

	int printf_written = swprintf(msg, msg_len + 1, L"%.*ls: (0x%08X) %.*ls", (int)op_len, op, error_code, (int)system_err_len, system_err_buf);

	if (printf_written < 0)
	{
		/* Highly unlikely */
		msg[0] = L'\0';
		return;
	}

	/* Get rid of the CR and LF that FormatMessage() sticks at the
	   end of the message. Thanks Microsoft! */
	while(msg[msg_len-1] == L'\r' || msg[msg_len-1] == L'\n' || msg[msg_len-1] == L' ')
	{
		msg[msg_len-1] = L'\0';
		msg_len--;
	}

#ifdef HIDAPI_USING_SDL_RUNTIME
	/* Thread-safe error handling */
	char *error_utf8 = SDL_iconv_wchar_utf8(msg);
	if (error_utf8) {
		SDL_SetError("%s", error_utf8);
		SDL_free(error_utf8);
	}
	free(msg);
#else
	*error_buffer = msg;
#endif
}

#if defined(__GNUC__) && (__GNUC__ + (__GNUC_MINOR__ >= 6) > 4)
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Warray-bounds"
#endif
/* A bug in GCC/mingw gives:
 * error: array subscript 0 is outside array bounds of 'wchar_t *[0]' {aka 'short unsigned int *[]'} [-Werror=array-bounds]
 * |         free(*error_buffer);
 * Which doesn't make sense in this context. */

static void register_string_error_to_buffer(wchar_t **error_buffer, const WCHAR *string_error)
{
	free(*error_buffer);
	*error_buffer = NULL;

#ifdef HIDAPI_USING_SDL_RUNTIME
	/* Thread-safe error handling */
	char *error_utf8 = string_error ? SDL_iconv_wchar_utf8(string_error) : NULL;
	if (error_utf8) {
		SDL_SetError("%s", error_utf8);
		SDL_free(error_utf8);
	} else {
		SDL_ClearError();
	}
#else
	if (string_error) {
		*error_buffer = _wcsdup(string_error);
	}
#endif /* HIDAPI_USING_SDL_RUNTIME */
}

#if defined(__GNUC__) && (__GNUC__ + (__GNUC_MINOR__ >= 6) > 4)
# pragma GCC diagnostic pop
#endif

static void register_winapi_error(hid_device *dev, const WCHAR *op)
{
	register_winapi_error_to_buffer(&dev->last_error_str, op);
}

static void register_string_error(hid_device *dev, const WCHAR *string_error)
{
	register_string_error_to_buffer(&dev->last_error_str, string_error);
}

static wchar_t *last_global_error_str = NULL;

static void register_global_winapi_error(const WCHAR *op)
{
	register_winapi_error_to_buffer(&last_global_error_str, op);
}

static void register_global_error(const WCHAR *string_error)
{
	register_string_error_to_buffer(&last_global_error_str, string_error);
}

static HANDLE open_device(const wchar_t *path, BOOL open_rw)
{
	HANDLE handle;
	DWORD desired_access = (open_rw)? (GENERIC_WRITE | GENERIC_READ): 0;
	DWORD share_mode = FILE_SHARE_READ|FILE_SHARE_WRITE;

	handle = CreateFileW(path,
		desired_access,
		share_mode,
		NULL,
		OPEN_EXISTING,
		FILE_FLAG_OVERLAPPED,/*FILE_ATTRIBUTE_NORMAL,*/
		0);

	return handle;
}

HID_API_EXPORT const struct hid_api_version* HID_API_CALL hid_version(void)
{
	return &api_version;
}

HID_API_EXPORT const char* HID_API_CALL hid_version_str(void)
{
	return HID_API_VERSION_STR;
}

int HID_API_EXPORT hid_init(void)
{
	register_global_error(NULL);
#ifndef HIDAPI_USE_DDK
	if (!hidapi_initialized) {
		if (lookup_functions() < 0) {
			register_global_winapi_error(L"resolve DLL functions");
			return -1;
		}
		hidapi_initialized = TRUE;
	}
#endif
	return 0;
}

int HID_API_EXPORT hid_exit(void)
{
#ifndef HIDAPI_USE_DDK
	free_library_handles();
	hidapi_initialized = FALSE;
#endif
	register_global_error(NULL);
	return 0;
}

static void* hid_internal_get_devnode_property(DEVINST dev_node, const DEVPROPKEY* property_key, DEVPROPTYPE expected_property_type)
{
	ULONG len = 0;
	CONFIGRET cr;
	DEVPROPTYPE property_type;
	PBYTE property_value = NULL;

	cr = CM_Get_DevNode_PropertyW(dev_node, property_key, &property_type, NULL, &len, 0);
	if (cr != CR_BUFFER_SMALL || property_type != expected_property_type)
		return NULL;

	property_value = (PBYTE)calloc(len, sizeof(BYTE));
	cr = CM_Get_DevNode_PropertyW(dev_node, property_key, &property_type, property_value, &len, 0);
	if (cr != CR_SUCCESS) {
		free(property_value);
		return NULL;
	}

	return property_value;
}

static void* hid_internal_get_device_interface_property(const wchar_t* interface_path, const DEVPROPKEY* property_key, DEVPROPTYPE expected_property_type)
{
	ULONG len = 0;
	CONFIGRET cr;
	DEVPROPTYPE property_type;
	PBYTE property_value = NULL;

	cr = CM_Get_Device_Interface_PropertyW(interface_path, property_key, &property_type, NULL, &len, 0);
	if (cr != CR_BUFFER_SMALL || property_type != expected_property_type)
		return NULL;

	property_value = (PBYTE)calloc(len, sizeof(BYTE));
	cr = CM_Get_Device_Interface_PropertyW(interface_path, property_key, &property_type, property_value, &len, 0);
	if (cr != CR_SUCCESS) {
		free(property_value);
		return NULL;
	}

	return property_value;
}

static void hid_internal_towupper(wchar_t* string)
{
	for (wchar_t* p = string; *p; ++p) *p = towupper(*p);
}

static int hid_internal_extract_int_token_value(wchar_t* string, const wchar_t* token)
{
	int token_value;
	wchar_t* startptr, * endptr;

	startptr = wcsstr(string, token);
	if (!startptr)
		return -1;

	startptr += wcslen(token);
	token_value = wcstol(startptr, &endptr, 16);
	if (endptr == startptr)
		return -1;

	return token_value;
}

static void hid_internal_get_usb_info(struct hid_device_info* dev, DEVINST dev_node)
{
	wchar_t *device_id = NULL, *hardware_ids = NULL;

	device_id = hid_internal_get_devnode_property(dev_node, &DEVPKEY_Device_InstanceId, DEVPROP_TYPE_STRING);
	if (!device_id)
		goto end;

	/* Normalize to upper case */
	hid_internal_towupper(device_id);

	/* Check for Xbox Common Controller class (XUSB) device.
	   https://docs.microsoft.com/windows/win32/xinput/directinput-and-xusb-devices
	   https://docs.microsoft.com/windows/win32/xinput/xinput-and-directinput
	*/
	if (hid_internal_extract_int_token_value(device_id, L"IG_") != -1) {
		/* Get devnode parent to reach out USB device. */
		if (CM_Get_Parent(&dev_node, dev_node, 0) != CR_SUCCESS)
			goto end;
	}

	/* Get the hardware ids from devnode */
	hardware_ids = hid_internal_get_devnode_property(dev_node, &DEVPKEY_Device_HardwareIds, DEVPROP_TYPE_STRING_LIST);
	if (!hardware_ids)
		goto end;

	/* Get additional information from USB device's Hardware ID
	   https://docs.microsoft.com/windows-hardware/drivers/install/standard-usb-identifiers
	   https://docs.microsoft.com/windows-hardware/drivers/usbcon/enumeration-of-interfaces-not-grouped-in-collections
	*/
	for (wchar_t* hardware_id = hardware_ids; *hardware_id; hardware_id += wcslen(hardware_id) + 1) {
		/* Normalize to upper case */
		hid_internal_towupper(hardware_id);

		if (dev->release_number == 0) {
			/* USB_DEVICE_DESCRIPTOR.bcdDevice value. */
			int release_number = hid_internal_extract_int_token_value(hardware_id, L"REV_");
			if (release_number != -1) {
				dev->release_number = (unsigned short)release_number;
			}
		}

		if (dev->interface_number == -1) {
			/* USB_INTERFACE_DESCRIPTOR.bInterfaceNumber value. */
			int interface_number = hid_internal_extract_int_token_value(hardware_id, L"MI_");
			if (interface_number != -1) {
				dev->interface_number = interface_number;
			}
		}
	}

	/* Try to get USB device manufacturer string if not provided by HidD_GetManufacturerString. */
	if (wcslen(dev->manufacturer_string) == 0) {
		wchar_t* manufacturer_string = hid_internal_get_devnode_property(dev_node, &DEVPKEY_Device_Manufacturer, DEVPROP_TYPE_STRING);
		if (manufacturer_string) {
			free(dev->manufacturer_string);
			dev->manufacturer_string = manufacturer_string;
		}
	}

	/* Try to get USB device serial number if not provided by HidD_GetSerialNumberString. */
	if (wcslen(dev->serial_number) == 0) {
		DEVINST usb_dev_node = dev_node;
		if (dev->interface_number != -1) {
			/* Get devnode parent to reach out composite parent USB device.
			   https://docs.microsoft.com/windows-hardware/drivers/usbcon/enumeration-of-the-composite-parent-device
			*/
			if (CM_Get_Parent(&usb_dev_node, dev_node, 0) != CR_SUCCESS)
				goto end;
		}

		/* Get the device id of the USB device. */
		free(device_id);
		device_id = hid_internal_get_devnode_property(usb_dev_node, &DEVPKEY_Device_InstanceId, DEVPROP_TYPE_STRING);
		if (!device_id)
			goto end;

		/* Extract substring after last '\\' of Instance ID.
		   For USB devices it may contain device's serial number.
		   https://docs.microsoft.com/windows-hardware/drivers/install/instance-ids
		*/
		for (wchar_t *ptr = device_id + wcslen(device_id); ptr > device_id; --ptr) {
			/* Instance ID is unique only within the scope of the bus.
			   For USB devices it means that serial number is not available. Skip. */
			if (*ptr == L'&')
				break;

			if (*ptr == L'\\') {
				free(dev->serial_number);
				dev->serial_number = _wcsdup(ptr + 1);
				break;
			}
		}
	}

	/* If we can't get the interface number, it means that there is only one interface. */
	if (dev->interface_number == -1)
		dev->interface_number = 0;

end:
	free(device_id);
	free(hardware_ids);
}

/* HidD_GetProductString/HidD_GetManufacturerString/HidD_GetSerialNumberString is not working for BLE HID devices
   Request this info via dev node properties instead.
   https://docs.microsoft.com/answers/questions/401236/hidd-getproductstring-with-ble-hid-device.html
*/
static void hid_internal_get_ble_info(struct hid_device_info* dev, DEVINST dev_node)
{
	if (wcslen(dev->manufacturer_string) == 0) {
		/* Manufacturer Name String (UUID: 0x2A29) */
		wchar_t* manufacturer_string = hid_internal_get_devnode_property(dev_node, (const DEVPROPKEY*)&PKEY_DeviceInterface_Bluetooth_Manufacturer, DEVPROP_TYPE_STRING);
		if (manufacturer_string) {
			free(dev->manufacturer_string);
			dev->manufacturer_string = manufacturer_string;
		}
	}

	if (wcslen(dev->serial_number) == 0) {
		/* Serial Number String (UUID: 0x2A25) */
		wchar_t* serial_number = hid_internal_get_devnode_property(dev_node, (const DEVPROPKEY*)&PKEY_DeviceInterface_Bluetooth_DeviceAddress, DEVPROP_TYPE_STRING);
		if (serial_number) {
			free(dev->serial_number);
			dev->serial_number = serial_number;
		}
	}

	if (wcslen(dev->product_string) == 0) {
		/* Model Number String (UUID: 0x2A24) */
		wchar_t* product_string = hid_internal_get_devnode_property(dev_node, (const DEVPROPKEY*)&PKEY_DeviceInterface_Bluetooth_ModelNumber, DEVPROP_TYPE_STRING);
		if (!product_string) {
			DEVINST parent_dev_node = 0;
			/* Fallback: Get devnode grandparent to reach out Bluetooth LE device node */
			if (CM_Get_Parent(&parent_dev_node, dev_node, 0) == CR_SUCCESS) {
				/* Device Name (UUID: 0x2A00) */
				product_string = hid_internal_get_devnode_property(parent_dev_node, &DEVPKEY_NAME, DEVPROP_TYPE_STRING);
			}
		}

		if (product_string) {
			free(dev->product_string);
			dev->product_string = product_string;
		}
	}
}

#ifdef HIDAPI_IGNORE_DEVICE
static hid_bus_type get_bus_type(const wchar_t* interface_path)
{
	wchar_t *device_id = NULL, *compatible_ids = NULL;
	CONFIGRET cr;
	DEVINST dev_node;
	hid_bus_type bus_type = HID_API_BUS_UNKNOWN;

	/* Get the device id from interface path */
	device_id = hid_internal_get_device_interface_property(interface_path, &DEVPKEY_Device_InstanceId, DEVPROP_TYPE_STRING);
	if (!device_id)
		goto end;

	/* Open devnode from device id */
	cr = CM_Locate_DevNodeW(&dev_node, (DEVINSTID_W)device_id, CM_LOCATE_DEVNODE_NORMAL);
	if (cr != CR_SUCCESS)
		goto end;

	/* Get devnode parent */
	cr = CM_Get_Parent(&dev_node, dev_node, 0);
	if (cr != CR_SUCCESS)
		goto end;

	/* Get the compatible ids from parent devnode */
	compatible_ids = hid_internal_get_devnode_property(dev_node, &DEVPKEY_Device_CompatibleIds, DEVPROP_TYPE_STRING_LIST);
	if (!compatible_ids)
		goto end;

	/* Now we can parse parent's compatible IDs to find out the device bus type */
	for (wchar_t* compatible_id = compatible_ids; *compatible_id; compatible_id += wcslen(compatible_id) + 1) {
		/* Normalize to upper case */
		hid_internal_towupper(compatible_id);

		/* USB devices
		   https://docs.microsoft.com/windows-hardware/drivers/hid/plug-and-play-support
		   https://docs.microsoft.com/windows-hardware/drivers/install/standard-usb-identifiers */
		if (wcsstr(compatible_id, L"USB") != NULL) {
			bus_type = HID_API_BUS_USB;
			break;
		}

		/* Bluetooth devices
		   https://docs.microsoft.com/windows-hardware/drivers/bluetooth/installing-a-bluetooth-device */
		if (wcsstr(compatible_id, L"BTHENUM") != NULL) {
			bus_type = HID_API_BUS_BLUETOOTH;
			break;
		}

		/* Bluetooth LE devices */
		if (wcsstr(compatible_id, L"BTHLEDEVICE") != NULL) {
			bus_type = HID_API_BUS_BLUETOOTH;
			break;
		}

		/* I2C devices
		   https://docs.microsoft.com/windows-hardware/drivers/hid/plug-and-play-support-and-power-management */
		if (wcsstr(compatible_id, L"PNP0C50") != NULL) {
			bus_type = HID_API_BUS_I2C;
			break;
		}

		/* SPI devices
		   https://docs.microsoft.com/windows-hardware/drivers/hid/plug-and-play-for-spi */
		if (wcsstr(compatible_id, L"PNP0C51") != NULL) {
			bus_type = HID_API_BUS_SPI;
			break;
		}
	}
end:
	free(device_id);
	free(compatible_ids);
	return bus_type;
}
#endif /* HIDAPI_IGNORE_DEVICE */

/* Unfortunately, HID_API_BUS_xxx constants alone aren't enough to distinguish between BLUETOOTH and BLE */

#define HID_API_BUS_FLAG_BLE 0x01

typedef struct hid_internal_detect_bus_type_result_ {
	DEVINST dev_node;
	hid_bus_type bus_type;
	unsigned int bus_flags;
} hid_internal_detect_bus_type_result;

static hid_internal_detect_bus_type_result hid_internal_detect_bus_type(const wchar_t* interface_path)
{
	wchar_t *device_id = NULL, *compatible_ids = NULL;
	CONFIGRET cr;
	DEVINST dev_node;
	hid_internal_detect_bus_type_result result = { 0 };

	/* Get the device id from interface path */
	device_id = hid_internal_get_device_interface_property(interface_path, &DEVPKEY_Device_InstanceId, DEVPROP_TYPE_STRING);
	if (!device_id)
		goto end;

	/* Open devnode from device id */
	cr = CM_Locate_DevNodeW(&dev_node, (DEVINSTID_W)device_id, CM_LOCATE_DEVNODE_NORMAL);
	if (cr != CR_SUCCESS)
		goto end;

	/* Get devnode parent */
	cr = CM_Get_Parent(&dev_node, dev_node, 0);
	if (cr != CR_SUCCESS)
		goto end;

	/* Get the compatible ids from parent devnode */
	compatible_ids = hid_internal_get_devnode_property(dev_node, &DEVPKEY_Device_CompatibleIds, DEVPROP_TYPE_STRING_LIST);
	if (!compatible_ids)
		goto end;

	/* Now we can parse parent's compatible IDs to find out the device bus type */
	for (wchar_t* compatible_id = compatible_ids; *compatible_id; compatible_id += wcslen(compatible_id) + 1) {
		/* Normalize to upper case */
		hid_internal_towupper(compatible_id);

		/* USB devices
		   https://docs.microsoft.com/windows-hardware/drivers/hid/plug-and-play-support
		   https://docs.microsoft.com/windows-hardware/drivers/install/standard-usb-identifiers */
		if (wcsstr(compatible_id, L"USB") != NULL) {
			result.bus_type = HID_API_BUS_USB;
			break;
		}

		/* Bluetooth devices
		   https://docs.microsoft.com/windows-hardware/drivers/bluetooth/installing-a-bluetooth-device */
		if (wcsstr(compatible_id, L"BTHENUM") != NULL) {
			result.bus_type = HID_API_BUS_BLUETOOTH;
			break;
		}

		/* Bluetooth LE devices */
		if (wcsstr(compatible_id, L"BTHLEDEVICE") != NULL) {
			result.bus_type = HID_API_BUS_BLUETOOTH;
			result.bus_flags |= HID_API_BUS_FLAG_BLE;
			break;
		}

		/* I2C devices
		   https://docs.microsoft.com/windows-hardware/drivers/hid/plug-and-play-support-and-power-management */
		if (wcsstr(compatible_id, L"PNP0C50") != NULL) {
			result.bus_type = HID_API_BUS_I2C;
			break;
		}

		/* SPI devices
		   https://docs.microsoft.com/windows-hardware/drivers/hid/plug-and-play-for-spi */
		if (wcsstr(compatible_id, L"PNP0C51") != NULL) {
			result.bus_type = HID_API_BUS_SPI;
			break;
		}
	}

	result.dev_node = dev_node;

end:
	free(device_id);
	free(compatible_ids);
	return result;
}

static char *hid_internal_UTF16toUTF8(const wchar_t *src)
{
	char *dst = NULL;
#ifdef HIDAPI_USING_SDL_RUNTIME
	int len = WIN_WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, src, -1, NULL, 0, NULL, NULL);
#else
	int len = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, src, -1, NULL, 0, NULL, NULL);
#endif
	if (len) {
		dst = (char*)calloc(len, sizeof(char));
		if (dst == NULL) {
			return NULL;
		}
#ifdef HIDAPI_USING_SDL_RUNTIME
		WIN_WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, src, -1, dst, len, NULL, NULL);
#else
		WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, src, -1, dst, len, NULL, NULL);
#endif
	}

	return dst;
}

static wchar_t *hid_internal_UTF8toUTF16(const char *src)
{
	wchar_t *dst = NULL;
	int len = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, src, -1, NULL, 0);
	if (len) {
		dst = (wchar_t*)calloc(len, sizeof(wchar_t));
		if (dst == NULL) {
			return NULL;
		}
		MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, src, -1, dst, len);
	}

	return dst;
}

static struct hid_device_info *hid_internal_get_device_info(const wchar_t *path, HANDLE handle)
{
	struct hid_device_info *dev = NULL; /* return object */
	HIDD_ATTRIBUTES attrib;
	PHIDP_PREPARSED_DATA pp_data = NULL;
	HIDP_CAPS caps;
	wchar_t string[MAX_STRING_WCHARS + 1];
	ULONG len;
	ULONG size;
	hid_internal_detect_bus_type_result detect_bus_type_result;

	/* Create the record. */
	dev = (struct hid_device_info*)calloc(1, sizeof(struct hid_device_info));

	if (dev == NULL) {
		return NULL;
	}

	/* Fill out the record */
	dev->next = NULL;
	dev->path = hid_internal_UTF16toUTF8(path);
	dev->interface_number = -1;

	attrib.Size = sizeof(HIDD_ATTRIBUTES);
	if (HidD_GetAttributes(handle, &attrib)) {
		/* VID/PID */
		dev->vendor_id = attrib.VendorID;
		dev->product_id = attrib.ProductID;

		/* Release Number */
		dev->release_number = attrib.VersionNumber;
	}

	/* Get the Usage Page and Usage for this device. */
	if (HidD_GetPreparsedData(handle, &pp_data)) {
		if (HidP_GetCaps(pp_data, &caps) == HIDP_STATUS_SUCCESS) {
			dev->usage_page = caps.UsagePage;
			dev->usage = caps.Usage;
		}

		HidD_FreePreparsedData(pp_data);
	}

	/* detect bus type before reading string descriptors */
	detect_bus_type_result = hid_internal_detect_bus_type(path);
	dev->bus_type = detect_bus_type_result.bus_type;

	len = dev->bus_type == HID_API_BUS_USB ? MAX_STRING_WCHARS_USB : MAX_STRING_WCHARS;
	string[len] = L'\0';
	size = len * sizeof(wchar_t);

	/* Serial Number */
	string[0] = L'\0';
	HidD_GetSerialNumberString(handle, string, size);
	dev->serial_number = _wcsdup(string);

	/* Manufacturer String */
	string[0] = L'\0';
	HidD_GetManufacturerString(handle, string, size);
	dev->manufacturer_string = _wcsdup(string);

	/* Product String */
	string[0] = L'\0';
	HidD_GetProductString(handle, string, size);
	dev->product_string = _wcsdup(string);

	/* now, the portion that depends on string descriptors */
	switch (dev->bus_type) {
	case HID_API_BUS_USB:
		hid_internal_get_usb_info(dev, detect_bus_type_result.dev_node);
		break;

	case HID_API_BUS_BLUETOOTH:
		if (detect_bus_type_result.bus_flags & HID_API_BUS_FLAG_BLE)
			hid_internal_get_ble_info(dev, detect_bus_type_result.dev_node);
		break;

	case HID_API_BUS_UNKNOWN:
	case HID_API_BUS_SPI:
	case HID_API_BUS_I2C:
		/* shut down -Wswitch */
		break;
	}

	return dev;
}

static int hid_blacklist(unsigned short vendor_id, unsigned short product_id)
{
	size_t i;
	static const struct { unsigned short vid; unsigned short pid; } known_bad[] = {
		{ 0x045E, 0x0822 },  /* Microsoft Precision Mouse - causes deadlock asking for device details */
		{ 0x0738, 0x2217 },  /* SPEEDLINK COMPETITION PRO - turns into an Android controller when enumerated */
		{ 0x0D8C, 0x0014 },  /* Sharkoon Skiller SGH2 headset - causes deadlock asking for device details */
		{ 0x1532, 0x0109 },  /* Razer Lycosa Gaming keyboard - causes deadlock asking for device details */
		{ 0x1532, 0x010B },  /* Razer Arctosa Gaming keyboard - causes deadlock asking for device details */
		{ 0x1B1C, 0x1B3D },  /* Corsair Gaming keyboard - causes deadlock asking for device details */
		{ 0x1CCF, 0x0000 }  /* All Konami Amusement Devices - causes deadlock asking for device details */
	};

	for (i = 0; i < (sizeof(known_bad)/sizeof(known_bad[0])); i++) {
		if ((vendor_id == known_bad[i].vid) && (product_id == known_bad[i].pid || known_bad[i].pid == 0x0000)) {
			return 1;
		}
	}

	return 0;
}

struct hid_device_info HID_API_EXPORT * HID_API_CALL hid_enumerate(unsigned short vendor_id, unsigned short product_id)
{
	struct hid_device_info *root = NULL; /* return object */
	struct hid_device_info *cur_dev = NULL;
	GUID interface_class_guid;
	CONFIGRET cr;
	wchar_t* device_interface_list = NULL;
	DWORD len;

	if (hid_init() < 0) {
		/* register_global_error: global error is reset by hid_init */
		return NULL;
	}

	/* Retrieve HID Interface Class GUID
	   https://docs.microsoft.com/windows-hardware/drivers/install/guid-devinterface-hid */
	HidD_GetHidGuid(&interface_class_guid);

	/* Get the list of all device interfaces belonging to the HID class. */
	/* Retry in case of list was changed between calls to
	  CM_Get_Device_Interface_List_SizeW and CM_Get_Device_Interface_ListW */
	do {
		cr = CM_Get_Device_Interface_List_SizeW(&len, &interface_class_guid, NULL, CM_GET_DEVICE_INTERFACE_LIST_PRESENT);
		if (cr != CR_SUCCESS) {
			register_global_error(L"Failed to get size of HID device interface list");
			break;
		}

		if (device_interface_list != NULL) {
			free(device_interface_list);
		}

		device_interface_list = (wchar_t*)calloc(len, sizeof(wchar_t));
		if (device_interface_list == NULL) {
			register_global_error(L"Failed to allocate memory for HID device interface list");
			return NULL;
		}
		cr = CM_Get_Device_Interface_ListW(&interface_class_guid, NULL, device_interface_list, len, CM_GET_DEVICE_INTERFACE_LIST_PRESENT);
		if (cr != CR_SUCCESS && cr != CR_BUFFER_SMALL) {
			register_global_error(L"Failed to get HID device interface list");
		}
	} while (cr == CR_BUFFER_SMALL);

	if (cr != CR_SUCCESS) {
		goto end_of_function;
	}

	/* Iterate over each device interface in the HID class, looking for the right one. */
	for (wchar_t* device_interface = device_interface_list; *device_interface; device_interface += wcslen(device_interface) + 1) {
		HANDLE device_handle = INVALID_HANDLE_VALUE;
		HIDD_ATTRIBUTES attrib;

        /* XInput devices don't get real HID reports and are better handled by the raw input driver */
        if (wcsstr(device_interface, L"&IG_") != NULL) {
            continue;
        }

		/* Open read-only handle to the device */
		device_handle = open_device(device_interface, FALSE);

		/* Check validity of device_handle. */
		if (device_handle == INVALID_HANDLE_VALUE) {
			/* Unable to open the device. */
			continue;
		}

		/* Get the Vendor ID and Product ID for this device. */
		attrib.Size = sizeof(HIDD_ATTRIBUTES);
		if (!HidD_GetAttributes(device_handle, &attrib)) {
			goto cont_close;
		}

#ifdef HIDAPI_IGNORE_DEVICE
		/* See if there are any devices we should skip in enumeration */
		hid_bus_type bus_type = get_bus_type(device_interface);
		PHIDP_PREPARSED_DATA pp_data = NULL;
		HIDP_CAPS caps = { 0 };
		if (HidD_GetPreparsedData(device_handle, &pp_data)) {
			HidP_GetCaps(pp_data, &caps);
			HidD_FreePreparsedData(pp_data);
		}
		if (HIDAPI_IGNORE_DEVICE(bus_type, attrib.VendorID, attrib.ProductID, caps.UsagePage, caps.Usage)) {
			goto cont_close;
		}
#endif

		/* Check the VID/PID to see if we should add this
		   device to the enumeration list. */
		if ((vendor_id == 0x0 || attrib.VendorID == vendor_id) &&
		    (product_id == 0x0 || attrib.ProductID == product_id) &&
		    !hid_blacklist(attrib.VendorID, attrib.ProductID)) {

			/* VID/PID match. Create the record. */
			struct hid_device_info *tmp = hid_internal_get_device_info(device_interface, device_handle);

			if (tmp == NULL) {
				goto cont_close;
			}

			if (cur_dev) {
				cur_dev->next = tmp;
			}
			else {
				root = tmp;
			}
			cur_dev = tmp;
		}

cont_close:
		CloseHandle(device_handle);
	}

	if (root == NULL) {
		if (vendor_id == 0 && product_id == 0) {
			register_global_error(L"No HID devices found in the system.");
		} else {
			register_global_error(L"No HID devices with requested VID/PID found in the system.");
		}
	}

end_of_function:
	free(device_interface_list);

	return root;
}

void  HID_API_EXPORT HID_API_CALL hid_free_enumeration(struct hid_device_info *devs)
{
	/* TODO: Merge this with the Linux version. This function is platform-independent. */
	struct hid_device_info *d = devs;
	while (d) {
		struct hid_device_info *next = d->next;
		free(d->path);
		free(d->serial_number);
		free(d->manufacturer_string);
		free(d->product_string);
		free(d);
		d = next;
	}
}

HID_API_EXPORT hid_device * HID_API_CALL hid_open(unsigned short vendor_id, unsigned short product_id, const wchar_t *serial_number)
{
	/* TODO: Merge this functions with the Linux version. This function should be platform independent. */
	struct hid_device_info *devs, *cur_dev;
	const char *path_to_open = NULL;
	hid_device *handle = NULL;

	/* register_global_error: global error is reset by hid_enumerate/hid_init */
	devs = hid_enumerate(vendor_id, product_id);
	if (!devs) {
		/* register_global_error: global error is already set by hid_enumerate */
		return NULL;
	}

	cur_dev = devs;
	while (cur_dev) {
		if (cur_dev->vendor_id == vendor_id &&
		    cur_dev->product_id == product_id) {
			if (serial_number) {
				if (cur_dev->serial_number && wcscmp(serial_number, cur_dev->serial_number) == 0) {
					path_to_open = cur_dev->path;
					break;
				}
			}
			else {
				path_to_open = cur_dev->path;
				break;
			}
		}
		cur_dev = cur_dev->next;
	}

	if (path_to_open) {
		/* Open the device */
		handle = hid_open_path(path_to_open);
	} else {
		register_global_error(L"Device with requested VID/PID/(SerialNumber) not found");
	}

	hid_free_enumeration(devs);

	return handle;
}

HID_API_EXPORT hid_device * HID_API_CALL hid_open_path(const char *path)
{
	hid_device *dev = NULL;
	wchar_t* interface_path = NULL;
	HANDLE device_handle = INVALID_HANDLE_VALUE;
	PHIDP_PREPARSED_DATA pp_data = NULL;
	HIDP_CAPS caps;

	if (hid_init() < 0) {
		/* register_global_error: global error is reset by hid_init */
		goto end_of_function;
	}

	interface_path = hid_internal_UTF8toUTF16(path);
	if (!interface_path) {
		register_global_error(L"Path conversion failure");
		goto end_of_function;
	}

	/* Open a handle to the device */
	device_handle = open_device(interface_path, TRUE);

	/* Check validity of write_handle. */
	if (device_handle == INVALID_HANDLE_VALUE) {
		/* System devices, such as keyboards and mice, cannot be opened in
		   read-write mode, because the system takes exclusive control over
		   them.  This is to prevent keyloggers.  However, feature reports
		   can still be sent and received.  Retry opening the device, but
		   without read/write access. */
		device_handle = open_device(interface_path, FALSE);

		/* Check the validity of the limited device_handle. */
		if (device_handle == INVALID_HANDLE_VALUE) {
			register_global_winapi_error(L"open_device");
			goto end_of_function;
		}
	}

	/* Set the Input Report buffer size to 64 reports. */
	if (!HidD_SetNumInputBuffers(device_handle, 64)) {
		register_global_winapi_error(L"set input buffers");
		goto end_of_function;
	}

	/* Get the Input Report length for the device. */
	if (!HidD_GetPreparsedData(device_handle, &pp_data)) {
		register_global_winapi_error(L"get preparsed data");
		goto end_of_function;
	}

	if (HidP_GetCaps(pp_data, &caps) != HIDP_STATUS_SUCCESS) {
		register_global_error(L"HidP_GetCaps");
		goto end_of_function;
	}

	dev = new_hid_device();

	if (dev == NULL) {
		register_global_error(L"hid_device allocation error");
		goto end_of_function;
	}

	dev->device_handle = device_handle;
	device_handle = INVALID_HANDLE_VALUE;

	dev->output_report_length = caps.OutputReportByteLength;
	dev->input_report_length = caps.InputReportByteLength;
	dev->feature_report_length = caps.FeatureReportByteLength;
	dev->read_buf = (char*) malloc(dev->input_report_length);
	dev->device_info = hid_internal_get_device_info(interface_path, dev->device_handle);

	/* On Windows 7, we need to use hid_write_output_report() over Bluetooth */
	if (dev->output_report_length > 512) {
		dev->use_hid_write_output_report = !IsWindowsVersionOrGreater( HIBYTE( _WIN32_WINNT_WIN8 ), LOBYTE( _WIN32_WINNT_WIN8 ), 0 );
	}

end_of_function:
	free(interface_path);
	CloseHandle(device_handle);

	if (pp_data) {
		HidD_FreePreparsedData(pp_data);
	}

	return dev;
}

static int hid_write_output_report(hid_device *dev, const unsigned char *data, size_t length)
{
	BOOL res;
	res = HidD_SetOutputReport(dev->device_handle, (void *)data, (ULONG)length);
	if (res)
		return (int)length;
	else
		return -1;
}

int HID_API_EXPORT HID_API_CALL hid_write(hid_device *dev, const unsigned char *data, size_t length)
{
	DWORD bytes_written = 0;
	int function_result = -1;
	BOOL res;
	BOOL overlapped = FALSE;

	unsigned char *buf;

	if (!data || !length) {
		register_string_error(dev, L"Zero buffer/length");
		return function_result;
	}

	register_string_error(dev, NULL);

	if (dev->use_hid_write_output_report) {
		return hid_write_output_report(dev, data, length);
	}

	/* Make sure the right number of bytes are passed to WriteFile. Windows
	   expects the number of bytes which are in the _longest_ report (plus
	   one for the report number) bytes even if the data is a report
	   which is shorter than that. Windows gives us this value in
	   caps.OutputReportByteLength. If a user passes in fewer bytes than this,
	   use cached temporary buffer which is the proper size. */
	if (length >= dev->output_report_length) {
		/* The user passed the right number of bytes. Use the buffer as-is. */
		buf = (unsigned char *) data;
	} else {
		if (dev->write_buf == NULL)
			dev->write_buf = (unsigned char *) malloc(dev->output_report_length);
		buf = dev->write_buf;
		memcpy(buf, data, length);
		memset(buf + length, 0, dev->output_report_length - length);
		length = dev->output_report_length;
	}

	res = WriteFile(dev->device_handle, buf, (DWORD) length, &bytes_written, &dev->write_ol);

	if (!res) {
		if (GetLastError() != ERROR_IO_PENDING) {
			/* WriteFile() failed. Return error. */
			register_winapi_error(dev, L"WriteFile");
			goto end_of_function;
		}
		overlapped = TRUE;
	} else {
		/* WriteFile() succeeded synchronously. */
		function_result = bytes_written;
	}

	if (overlapped) {
		/* Wait for the transaction to complete. This makes
		   hid_write() synchronous. */
		res = WaitForSingleObject(dev->write_ol.hEvent, 1000);
		if (res != WAIT_OBJECT_0) {
			/* There was a Timeout. */
			register_winapi_error(dev, L"hid_write/WaitForSingleObject");
			goto end_of_function;
		}

		/* Get the result. */
		res = GetOverlappedResult(dev->device_handle, &dev->write_ol, &bytes_written, FALSE/*wait*/);
		if (res) {
			function_result = bytes_written;
		}
		else {
			/* The Write operation failed. */
			register_winapi_error(dev, L"hid_write/GetOverlappedResult");
			goto end_of_function;
		}
	}

end_of_function:
	return function_result;
}


int HID_API_EXPORT HID_API_CALL hid_read_timeout(hid_device *dev, unsigned char *data, size_t length, int milliseconds)
{
	DWORD bytes_read = 0;
	size_t copy_len = 0;
	BOOL res = FALSE;
	BOOL overlapped = FALSE;

	if (!data || !length) {
		register_string_error(dev, L"Zero buffer/length");
		return -1;
	}

	register_string_error(dev, NULL);

	/* Copy the handle for convenience. */
	HANDLE ev = dev->ol.hEvent;

	if (!dev->read_pending) {
		/* Start an Overlapped I/O read. */
		dev->read_pending = TRUE;
		memset(dev->read_buf, 0, dev->input_report_length);
		ResetEvent(ev);
		res = ReadFile(dev->device_handle, dev->read_buf, (DWORD) dev->input_report_length, &bytes_read, &dev->ol);

		if (!res) {
			if (GetLastError() != ERROR_IO_PENDING) {
				/* ReadFile() has failed.
				   Clean up and return error. */
				register_winapi_error(dev, L"ReadFile");
				CancelIo(dev->device_handle);
				dev->read_pending = FALSE;
				goto end_of_function;
			}
			overlapped = TRUE;
		}
	}
	else {
		overlapped = TRUE;
	}

	if (overlapped) {
		/* See if there is any data yet. */
		res = WaitForSingleObject(ev, milliseconds >= 0 ? (DWORD)milliseconds : INFINITE);
		if (res != WAIT_OBJECT_0) {
			/* There was no data this time. Return zero bytes available,
			   but leave the Overlapped I/O running. */
			return 0;
		}

		/* Get the number of bytes read. The actual data has been copied to the data[]
		   array which was passed to ReadFile(). We must not wait here because we've
		   already waited on our event above, and since it's auto-reset, it will have
		   been reset back to unsignalled by now. */
		res = GetOverlappedResult(dev->device_handle, &dev->ol, &bytes_read, FALSE/*don't wait now - already did on the prev step*/);
	}
	/* Set pending back to false, even if GetOverlappedResult() returned error. */
	dev->read_pending = FALSE;

	if (res && bytes_read > 0) {
		if (dev->read_buf[0] == 0x0) {
			/* If report numbers aren't being used, but Windows sticks a report
			   number (0x0) on the beginning of the report anyway. To make this
			   work like the other platforms, and to make it work more like the
			   HID spec, we'll skip over this byte. */
			bytes_read--;
			copy_len = length > bytes_read ? bytes_read : length;
			memcpy(data, dev->read_buf+1, copy_len);
		}
		else {
			/* Copy the whole buffer, report number and all. */
			copy_len = length > bytes_read ? bytes_read : length;
			memcpy(data, dev->read_buf, copy_len);
		}
	}
	if (!res) {
		register_winapi_error(dev, L"hid_read_timeout/GetOverlappedResult");
	}

end_of_function:
	if (!res) {
		return -1;
	}

	return (int) copy_len;
}

int HID_API_EXPORT HID_API_CALL hid_read(hid_device *dev, unsigned char *data, size_t length)
{
	return hid_read_timeout(dev, data, length, (dev->blocking)? -1: 0);
}

int HID_API_EXPORT HID_API_CALL hid_set_nonblocking(hid_device *dev, int nonblock)
{
	dev->blocking = !nonblock;
	return 0; /* Success */
}

int HID_API_EXPORT HID_API_CALL hid_send_feature_report(hid_device *dev, const unsigned char *data, size_t length)
{
	BOOL res = FALSE;
	unsigned char *buf;
	size_t length_to_send;

	if (!data || !length) {
		register_string_error(dev, L"Zero buffer/length");
		return -1;
	}

	register_string_error(dev, NULL);

	/* Windows expects at least caps.FeatureReportByteLength bytes passed
	   to HidD_SetFeature(), even if the report is shorter. Any less sent and
	   the function fails with error ERROR_INVALID_PARAMETER set. Any more
	   and HidD_SetFeature() silently truncates the data sent in the report
	   to caps.FeatureReportByteLength. */
	if (length >= dev->feature_report_length) {
		buf = (unsigned char *) data;
		length_to_send = length;
	} else {
		if (dev->feature_buf == NULL)
			dev->feature_buf = (unsigned char *) malloc(dev->feature_report_length);
		buf = dev->feature_buf;
		memcpy(buf, data, length);
		memset(buf + length, 0, dev->feature_report_length - length);
		length_to_send = dev->feature_report_length;
	}

	res = HidD_SetFeature(dev->device_handle, (PVOID)buf, (DWORD) length_to_send);

	if (!res) {
		register_winapi_error(dev, L"HidD_SetFeature");
		return -1;
	}

	return (int) length;
}

static int hid_get_report(hid_device *dev, DWORD report_type, unsigned char *data, size_t length)
{
	BOOL res;
	DWORD bytes_returned = 0;

	OVERLAPPED ol;
	memset(&ol, 0, sizeof(ol));

	if (!data || !length) {
		register_string_error(dev, L"Zero buffer/length");
		return -1;
	}

	register_string_error(dev, NULL);

	res = DeviceIoControl(dev->device_handle,
		report_type,
		data, (DWORD) length,
		data, (DWORD) length,
		&bytes_returned, &ol);

	if (!res) {
		if (GetLastError() != ERROR_IO_PENDING) {
			/* DeviceIoControl() failed. Return error. */
			register_winapi_error(dev, L"Get Input/Feature Report DeviceIoControl");
			return -1;
		}
	}

	/* Wait here until the write is done. This makes
	   hid_get_feature_report() synchronous. */
	res = GetOverlappedResult(dev->device_handle, &ol, &bytes_returned, TRUE/*wait*/);
	if (!res) {
		/* The operation failed. */
		register_winapi_error(dev, L"Get Input/Feature Report GetOverLappedResult");
		return -1;
	}

	/* When numbered reports aren't used,
	   bytes_returned seem to include only what is actually received from the device
	   (not including the first byte with 0, as an indication "no numbered reports"). */
	if (data[0] == 0x0) {
		bytes_returned++;
	}

	return bytes_returned;
}

int HID_API_EXPORT HID_API_CALL hid_get_feature_report(hid_device *dev, unsigned char *data, size_t length)
{
	/* We could use HidD_GetFeature() instead, but it doesn't give us an actual length, unfortunately */
	return hid_get_report(dev, IOCTL_HID_GET_FEATURE, data, length);
}

int HID_API_EXPORT HID_API_CALL hid_get_input_report(hid_device *dev, unsigned char *data, size_t length)
{
	/* We could use HidD_GetInputReport() instead, but it doesn't give us an actual length, unfortunately */
	return hid_get_report(dev, IOCTL_HID_GET_INPUT_REPORT, data, length);
}

#if defined(__GNUC__) && __GNUC__ >= 8
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wcast-function-type"
#endif
void HID_API_EXPORT HID_API_CALL hid_close(hid_device *dev)
{
	typedef BOOL (WINAPI *CancelIoEx_t)(HANDLE hFile, LPOVERLAPPED lpOverlapped);
	CancelIoEx_t CancelIoExFunc = (CancelIoEx_t)GetProcAddress(GetModuleHandle(TEXT("kernel32.dll")), "CancelIoEx");

	if (!dev)
		return;

	if (CancelIoExFunc) {
		CancelIoExFunc(dev->device_handle, NULL);
	} else {
		/* Windows XP, this will only cancel I/O on the current thread */
		CancelIo(dev->device_handle);
	}
	if (dev->read_pending) {
		DWORD bytes_read = 0;
		GetOverlappedResult(dev->device_handle, &dev->ol, &bytes_read, TRUE/*wait*/);
	}
	free_hid_device(dev);
}
#if defined(__GNUC__) && __GNUC__ >= 8
# pragma GCC diagnostic pop
#endif

int HID_API_EXPORT_CALL HID_API_CALL hid_get_manufacturer_string(hid_device *dev, wchar_t *string, size_t maxlen)
{
	if (!string || !maxlen) {
		register_string_error(dev, L"Zero buffer/length");
		return -1;
	}

	if (!dev->device_info) {
		register_string_error(dev, L"NULL device info");
		return -1;
	}

	wcsncpy(string, dev->device_info->manufacturer_string, maxlen);
	string[maxlen - 1] = L'\0';

	register_string_error(dev, NULL);

	return 0;
}

int HID_API_EXPORT_CALL HID_API_CALL hid_get_product_string(hid_device *dev, wchar_t *string, size_t maxlen)
{
	if (!string || !maxlen) {
		register_string_error(dev, L"Zero buffer/length");
		return -1;
	}

	if (!dev->device_info) {
		register_string_error(dev, L"NULL device info");
		return -1;
	}

	wcsncpy(string, dev->device_info->product_string, maxlen);
	string[maxlen - 1] = L'\0';

	register_string_error(dev, NULL);

	return 0;
}

int HID_API_EXPORT_CALL HID_API_CALL hid_get_serial_number_string(hid_device *dev, wchar_t *string, size_t maxlen)
{
	if (!string || !maxlen) {
		register_string_error(dev, L"Zero buffer/length");
		return -1;
	}

	if (!dev->device_info) {
		register_string_error(dev, L"NULL device info");
		return -1;
	}

	wcsncpy(string, dev->device_info->serial_number, maxlen);
	string[maxlen - 1] = L'\0';

	register_string_error(dev, NULL);

	return 0;
}

HID_API_EXPORT struct hid_device_info * HID_API_CALL hid_get_device_info(hid_device *dev) {
	if (!dev->device_info)
	{
		register_string_error(dev, L"NULL device info");
		return NULL;
	}

	return dev->device_info;
}

int HID_API_EXPORT_CALL HID_API_CALL hid_get_indexed_string(hid_device *dev, int string_index, wchar_t *string, size_t maxlen)
{
	BOOL res;

	if (dev->device_info && dev->device_info->bus_type == HID_API_BUS_USB && maxlen > MAX_STRING_WCHARS_USB) {
		string[MAX_STRING_WCHARS_USB] = L'\0';
		maxlen = MAX_STRING_WCHARS_USB;
	}

	res = HidD_GetIndexedString(dev->device_handle, string_index, string, (ULONG)maxlen * sizeof(wchar_t));
	if (!res) {
		register_winapi_error(dev, L"HidD_GetIndexedString");
		return -1;
	}

	register_string_error(dev, NULL);

	return 0;
}

int HID_API_EXPORT_CALL hid_winapi_get_container_id(hid_device *dev, GUID *container_id)
{
	wchar_t *interface_path = NULL, *device_id = NULL;
	CONFIGRET cr = CR_FAILURE;
	DEVINST dev_node;
	DEVPROPTYPE property_type;
	ULONG len;

	if (!container_id) {
		register_string_error(dev, L"Invalid Container ID");
		return -1;
	}

	register_string_error(dev, NULL);

	interface_path = hid_internal_UTF8toUTF16(dev->device_info->path);
	if (!interface_path) {
		register_string_error(dev, L"Path conversion failure");
		goto end;
	}

	/* Get the device id from interface path */
	device_id = hid_internal_get_device_interface_property(interface_path, &DEVPKEY_Device_InstanceId, DEVPROP_TYPE_STRING);
	if (!device_id) {
		register_string_error(dev, L"Failed to get device interface property InstanceId");
		goto end;
	}

	/* Open devnode from device id */
	cr = CM_Locate_DevNodeW(&dev_node, (DEVINSTID_W)device_id, CM_LOCATE_DEVNODE_NORMAL);
	if (cr != CR_SUCCESS) {
		register_string_error(dev, L"Failed to locate device node");
		goto end;
	}

	/* Get the container id from devnode */
	len = sizeof(*container_id);
	cr = CM_Get_DevNode_PropertyW(dev_node, &DEVPKEY_Device_ContainerId, &property_type, (PBYTE)container_id, &len, 0);
	if (cr == CR_SUCCESS && property_type != DEVPROP_TYPE_GUID)
		cr = CR_FAILURE;

	if (cr != CR_SUCCESS)
		register_string_error(dev, L"Failed to read ContainerId property from device node");

end:
	free(interface_path);
	free(device_id);

	return cr == CR_SUCCESS ? 0 : -1;
}


int HID_API_EXPORT_CALL hid_get_report_descriptor(hid_device *dev, unsigned char *buf, size_t buf_size)
{
	PHIDP_PREPARSED_DATA pp_data = NULL;

	if (!HidD_GetPreparsedData(dev->device_handle, &pp_data) || pp_data == NULL) {
		register_string_error(dev, L"HidD_GetPreparsedData");
		return -1;
	}

	int res = hid_winapi_descriptor_reconstruct_pp_data(pp_data, buf, buf_size);

	HidD_FreePreparsedData(pp_data);

	return res;
}

HID_API_EXPORT const wchar_t * HID_API_CALL  hid_error(hid_device *dev)
{
	if (dev) {
		if (dev->last_error_str == NULL)
			return L"Success";
		return (wchar_t*)dev->last_error_str;
	}

	if (last_global_error_str == NULL)
		return L"Success";
	return last_global_error_str;
}

#ifndef hidapi_winapi_EXPORTS
#include "hidapi_descriptor_reconstruct.c"
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif
