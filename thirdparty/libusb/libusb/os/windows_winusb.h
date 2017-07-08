/*
 * Windows backend for libusb 1.0
 * Copyright © 2009-2012 Pete Batard <pete@akeo.ie>
 * With contributions from Michael Plante, Orin Eman et al.
 * Parts of this code adapted from libusb-win32-v1 by Stephan Meyer
 * Major code testing contribution by Xiaofan Chen
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#pragma once

#include "windows_common.h"
#include "windows_nt_common.h"

#if defined(_MSC_VER)
// disable /W4 MSVC warnings that are benign
#pragma warning(disable:4100)  // unreferenced formal parameter
#pragma warning(disable:4127)  // conditional expression is constant
#pragma warning(disable:4201)  // nameless struct/union
#pragma warning(disable:4214)  // bit field types other than int
#pragma warning(disable:4996)  // deprecated API calls
#pragma warning(disable:28159) // more deprecated API calls
#endif

// Missing from MSVC6 setupapi.h
#ifndef SPDRP_ADDRESS
#define SPDRP_ADDRESS		28
#endif
#ifndef SPDRP_INSTALL_STATE
#define SPDRP_INSTALL_STATE	34
#endif

#define MAX_CTRL_BUFFER_LENGTH	4096
#define MAX_USB_DEVICES		256
#define MAX_USB_STRING_LENGTH	128
#define MAX_HID_REPORT_SIZE	1024
#define MAX_HID_DESCRIPTOR_SIZE	256
#define MAX_GUID_STRING_LENGTH	40
#define MAX_PATH_LENGTH		128
#define MAX_KEY_LENGTH		256
#define LIST_SEPARATOR		';'

// Handle code for HID interface that have been claimed ("dibs")
#define INTERFACE_CLAIMED	((HANDLE)(intptr_t)0xD1B5)
// Additional return code for HID operations that completed synchronously
#define LIBUSB_COMPLETED	(LIBUSB_SUCCESS + 1)

// http://msdn.microsoft.com/en-us/library/ff545978.aspx
// http://msdn.microsoft.com/en-us/library/ff545972.aspx
// http://msdn.microsoft.com/en-us/library/ff545982.aspx
#ifndef GUID_DEVINTERFACE_USB_HOST_CONTROLLER
const GUID GUID_DEVINTERFACE_USB_HOST_CONTROLLER = { 0x3ABF6F2D, 0x71C4, 0x462A, {0x8A, 0x92, 0x1E, 0x68, 0x61, 0xE6, 0xAF, 0x27} };
#endif
#ifndef GUID_DEVINTERFACE_USB_DEVICE
const GUID GUID_DEVINTERFACE_USB_DEVICE = { 0xA5DCBF10, 0x6530, 0x11D2, {0x90, 0x1F, 0x00, 0xC0, 0x4F, 0xB9, 0x51, 0xED} };
#endif
#ifndef GUID_DEVINTERFACE_USB_HUB
const GUID GUID_DEVINTERFACE_USB_HUB = { 0xF18A0E88, 0xC30C, 0x11D0, {0x88, 0x15, 0x00, 0xA0, 0xC9, 0x06, 0xBE, 0xD8} };
#endif
#ifndef GUID_DEVINTERFACE_LIBUSB0_FILTER
const GUID GUID_DEVINTERFACE_LIBUSB0_FILTER = { 0xF9F3FF14, 0xAE21, 0x48A0, {0x8A, 0x25, 0x80, 0x11, 0xA7, 0xA9, 0x31, 0xD9} };
#endif


/*
 * Multiple USB API backend support
 */
#define USB_API_UNSUPPORTED	0
#define USB_API_HUB		1
#define USB_API_COMPOSITE	2
#define USB_API_WINUSBX		3
#define USB_API_HID		4
#define USB_API_MAX		5
// The following is used to indicate if the HID or composite extra props have already been set.
#define USB_API_SET		(1 << USB_API_MAX)

// Sub-APIs for WinUSB-like driver APIs (WinUSB, libusbK, libusb-win32 through the libusbK DLL)
// Must have the same values as the KUSB_DRVID enum from libusbk.h
#define SUB_API_NOTSET		-1
#define SUB_API_LIBUSBK		0
#define SUB_API_LIBUSB0		1
#define SUB_API_WINUSB		2
#define SUB_API_MAX		3

#define WINUSBX_DRV_NAMES	{"libusbK", "libusb0", "WinUSB"}

struct windows_usb_api_backend {
	const uint8_t id;
	const char *designation;
	const char **driver_name_list; // Driver name, without .sys, e.g. "usbccgp"
	const uint8_t nb_driver_names;
	int (*init)(int sub_api, struct libusb_context *ctx);
	int (*exit)(int sub_api);
	int (*open)(int sub_api, struct libusb_device_handle *dev_handle);
	void (*close)(int sub_api, struct libusb_device_handle *dev_handle);
	int (*configure_endpoints)(int sub_api, struct libusb_device_handle *dev_handle, int iface);
	int (*claim_interface)(int sub_api, struct libusb_device_handle *dev_handle, int iface);
	int (*set_interface_altsetting)(int sub_api, struct libusb_device_handle *dev_handle, int iface, int altsetting);
	int (*release_interface)(int sub_api, struct libusb_device_handle *dev_handle, int iface);
	int (*clear_halt)(int sub_api, struct libusb_device_handle *dev_handle, unsigned char endpoint);
	int (*reset_device)(int sub_api, struct libusb_device_handle *dev_handle);
	int (*submit_bulk_transfer)(int sub_api, struct usbi_transfer *itransfer);
	int (*submit_iso_transfer)(int sub_api, struct usbi_transfer *itransfer);
	int (*submit_control_transfer)(int sub_api, struct usbi_transfer *itransfer);
	int (*abort_control)(int sub_api, struct usbi_transfer *itransfer);
	int (*abort_transfers)(int sub_api, struct usbi_transfer *itransfer);
	int (*copy_transfer_data)(int sub_api, struct usbi_transfer *itransfer, uint32_t io_size);
};

extern const struct windows_usb_api_backend usb_api_backend[USB_API_MAX];

#define PRINT_UNSUPPORTED_API(fname)				\
	usbi_dbg("unsupported API call for '"			\
		#fname "' (unrecognized device driver)");	\
	return LIBUSB_ERROR_NOT_SUPPORTED;

/*
 * private structures definition
 * with inline pseudo constructors/destructors
 */

// TODO (v2+): move hid desc to libusb.h?
struct libusb_hid_descriptor {
	uint8_t bLength;
	uint8_t bDescriptorType;
	uint16_t bcdHID;
	uint8_t bCountryCode;
	uint8_t bNumDescriptors;
	uint8_t bClassDescriptorType;
	uint16_t wClassDescriptorLength;
};

#define LIBUSB_DT_HID_SIZE		9
#define HID_MAX_CONFIG_DESC_SIZE (LIBUSB_DT_CONFIG_SIZE + LIBUSB_DT_INTERFACE_SIZE \
	+ LIBUSB_DT_HID_SIZE + 2 * LIBUSB_DT_ENDPOINT_SIZE)
#define HID_MAX_REPORT_SIZE		1024
#define HID_IN_EP			0x81
#define HID_OUT_EP			0x02
#define LIBUSB_REQ_RECIPIENT(request_type)	((request_type) & 0x1F)
#define LIBUSB_REQ_TYPE(request_type)		((request_type) & (0x03 << 5))
#define LIBUSB_REQ_IN(request_type)		((request_type) & LIBUSB_ENDPOINT_IN)
#define LIBUSB_REQ_OUT(request_type)		(!LIBUSB_REQ_IN(request_type))

// The following are used for HID reports IOCTLs
#define HID_CTL_CODE(id) \
	CTL_CODE (FILE_DEVICE_KEYBOARD, (id), METHOD_NEITHER, FILE_ANY_ACCESS)
#define HID_BUFFER_CTL_CODE(id) \
	CTL_CODE (FILE_DEVICE_KEYBOARD, (id), METHOD_BUFFERED, FILE_ANY_ACCESS)
#define HID_IN_CTL_CODE(id) \
	CTL_CODE (FILE_DEVICE_KEYBOARD, (id), METHOD_IN_DIRECT, FILE_ANY_ACCESS)
#define HID_OUT_CTL_CODE(id) \
	CTL_CODE (FILE_DEVICE_KEYBOARD, (id), METHOD_OUT_DIRECT, FILE_ANY_ACCESS)

#define IOCTL_HID_GET_FEATURE		HID_OUT_CTL_CODE(100)
#define IOCTL_HID_GET_INPUT_REPORT	HID_OUT_CTL_CODE(104)
#define IOCTL_HID_SET_FEATURE		HID_IN_CTL_CODE(100)
#define IOCTL_HID_SET_OUTPUT_REPORT	HID_IN_CTL_CODE(101)

enum libusb_hid_request_type {
	HID_REQ_GET_REPORT = 0x01,
	HID_REQ_GET_IDLE = 0x02,
	HID_REQ_GET_PROTOCOL = 0x03,
	HID_REQ_SET_REPORT = 0x09,
	HID_REQ_SET_IDLE = 0x0A,
	HID_REQ_SET_PROTOCOL = 0x0B
};

enum libusb_hid_report_type {
	HID_REPORT_TYPE_INPUT = 0x01,
	HID_REPORT_TYPE_OUTPUT = 0x02,
	HID_REPORT_TYPE_FEATURE = 0x03
};

struct hid_device_priv {
	uint16_t vid;
	uint16_t pid;
	uint8_t config;
	uint8_t nb_interfaces;
	bool uses_report_ids[3]; // input, ouptput, feature
	uint16_t input_report_size;
	uint16_t output_report_size;
	uint16_t feature_report_size;
	WCHAR string[3][MAX_USB_STRING_LENGTH];
	uint8_t string_index[3]; // man, prod, ser
};

struct windows_device_priv {
	uint8_t depth; // distance to HCD
	uint8_t port;  // port number on the hub
	uint8_t active_config;
	struct windows_usb_api_backend const *apib;
	char *path;  // device interface path
	int sub_api; // for WinUSB-like APIs
	struct {
		char *path; // each interface needs a device interface path,
		struct windows_usb_api_backend const *apib; // an API backend (multiple drivers support),
		int sub_api;
		int8_t nb_endpoints; // and a set of endpoint addresses (USB_MAXENDPOINTS)
		uint8_t *endpoint;
		bool restricted_functionality;  // indicates if the interface functionality is restricted
                                                // by Windows (eg. HID keyboards or mice cannot do R/W)
	} usb_interface[USB_MAXINTERFACES];
	struct hid_device_priv *hid;
	USB_DEVICE_DESCRIPTOR dev_descriptor;
	unsigned char **config_descriptor; // list of pointers to the cached config descriptors
};

static inline struct windows_device_priv *_device_priv(struct libusb_device *dev)
{
	return (struct windows_device_priv *)dev->os_priv;
}

static inline struct windows_device_priv *windows_device_priv_init(struct libusb_device *dev)
{
	struct windows_device_priv *p = _device_priv(dev);
	int i;

	p->apib = &usb_api_backend[USB_API_UNSUPPORTED];
	p->sub_api = SUB_API_NOTSET;
	for (i = 0; i < USB_MAXINTERFACES; i++) {
		p->usb_interface[i].apib = &usb_api_backend[USB_API_UNSUPPORTED];
		p->usb_interface[i].sub_api = SUB_API_NOTSET;
	}

	return p;
}

static inline void windows_device_priv_release(struct libusb_device *dev)
{
	struct windows_device_priv *p = _device_priv(dev);
	int i;

	free(p->path);
	if ((dev->num_configurations > 0) && (p->config_descriptor != NULL)) {
		for (i = 0; i < dev->num_configurations; i++)
			free(p->config_descriptor[i]);
	}
	free(p->config_descriptor);
	free(p->hid);
	for (i = 0; i < USB_MAXINTERFACES; i++) {
		free(p->usb_interface[i].path);
		free(p->usb_interface[i].endpoint);
	}
}

struct interface_handle_t {
	HANDLE dev_handle; // WinUSB needs an extra handle for the file
	HANDLE api_handle; // used by the API to communicate with the device
};

struct windows_device_handle_priv {
	int active_interface;
	struct interface_handle_t interface_handle[USB_MAXINTERFACES];
	int autoclaim_count[USB_MAXINTERFACES]; // For auto-release
};

static inline struct windows_device_handle_priv *_device_handle_priv(
	struct libusb_device_handle *handle)
{
	return (struct windows_device_handle_priv *)handle->os_priv;
}

// used for async polling functions
struct windows_transfer_priv {
	struct winfd pollable_fd;
	uint8_t interface_number;
	uint8_t *hid_buffer; // 1 byte extended data buffer, required for HID
	uint8_t *hid_dest;   // transfer buffer destination, required for HID
	size_t hid_expected_size;
};

// used to match a device driver (including filter drivers) against a supported API
struct driver_lookup {
	char list[MAX_KEY_LENGTH + 1]; // REG_MULTI_SZ list of services (driver) names
	const DWORD reg_prop;          // SPDRP registry key to use to retrieve list
	const char* designation;       // internal designation (for debug output)
};

/* OLE32 dependency */
DLL_DECLARE_HANDLE(OLE32);
DLL_DECLARE_FUNC_PREFIXED(WINAPI, HRESULT, p, CLSIDFromString, (LPCOLESTR, LPCLSID));

/* Kernel32 dependencies */
DLL_DECLARE_HANDLE(Kernel32);
/* This call is only available from XP SP2 */
DLL_DECLARE_FUNC_PREFIXED(WINAPI, BOOL, p, IsWow64Process, (HANDLE, PBOOL));

/* SetupAPI dependencies */
DLL_DECLARE_HANDLE(SetupAPI);
DLL_DECLARE_FUNC_PREFIXED(WINAPI, HDEVINFO, p, SetupDiGetClassDevsA, (const GUID*, PCSTR, HWND, DWORD));
DLL_DECLARE_FUNC_PREFIXED(WINAPI, BOOL, p, SetupDiEnumDeviceInfo, (HDEVINFO, DWORD, PSP_DEVINFO_DATA));
DLL_DECLARE_FUNC_PREFIXED(WINAPI, BOOL, p, SetupDiEnumDeviceInterfaces, (HDEVINFO, PSP_DEVINFO_DATA,
			const GUID*, DWORD, PSP_DEVICE_INTERFACE_DATA));
DLL_DECLARE_FUNC_PREFIXED(WINAPI, BOOL, p, SetupDiGetDeviceInterfaceDetailA, (HDEVINFO, PSP_DEVICE_INTERFACE_DATA,
			PSP_DEVICE_INTERFACE_DETAIL_DATA_A, DWORD, PDWORD, PSP_DEVINFO_DATA));
DLL_DECLARE_FUNC_PREFIXED(WINAPI, BOOL, p, SetupDiDestroyDeviceInfoList, (HDEVINFO));
DLL_DECLARE_FUNC_PREFIXED(WINAPI, HKEY, p, SetupDiOpenDevRegKey, (HDEVINFO, PSP_DEVINFO_DATA, DWORD, DWORD, DWORD, REGSAM));
DLL_DECLARE_FUNC_PREFIXED(WINAPI, BOOL, p, SetupDiGetDeviceRegistryPropertyA, (HDEVINFO,
			PSP_DEVINFO_DATA, DWORD, PDWORD, PBYTE, DWORD, PDWORD));
DLL_DECLARE_FUNC_PREFIXED(WINAPI, HKEY, p, SetupDiOpenDeviceInterfaceRegKey, (HDEVINFO, PSP_DEVICE_INTERFACE_DATA, DWORD, DWORD));

/* AdvAPI32 dependencies */
DLL_DECLARE_HANDLE(AdvAPI32);
DLL_DECLARE_FUNC_PREFIXED(WINAPI, LONG, p, RegQueryValueExW, (HKEY, LPCWSTR, LPDWORD, LPDWORD, LPBYTE, LPDWORD));
DLL_DECLARE_FUNC_PREFIXED(WINAPI, LONG, p, RegCloseKey, (HKEY));

/*
 * Windows DDK API definitions. Most of it copied from MinGW's includes
 */
typedef DWORD DEVNODE, DEVINST;
typedef DEVNODE *PDEVNODE, *PDEVINST;
typedef DWORD RETURN_TYPE;
typedef RETURN_TYPE CONFIGRET;

#define CR_SUCCESS				0x00000000

#ifndef USB_GET_DESCRIPTOR_FROM_NODE_CONNECTION
#define USB_GET_DESCRIPTOR_FROM_NODE_CONNECTION	260
#endif
#ifndef USB_GET_NODE_CONNECTION_INFORMATION_EX
#define USB_GET_NODE_CONNECTION_INFORMATION_EX	274
#endif
#ifndef USB_GET_NODE_CONNECTION_INFORMATION_EX_V2
#define USB_GET_NODE_CONNECTION_INFORMATION_EX_V2	279
#endif

#ifndef METHOD_BUFFERED
#define METHOD_BUFFERED				0
#endif
#ifndef FILE_ANY_ACCESS
#define FILE_ANY_ACCESS				0x00000000
#endif
#ifndef FILE_DEVICE_UNKNOWN
#define FILE_DEVICE_UNKNOWN			0x00000022
#endif
#ifndef FILE_DEVICE_USB
#define FILE_DEVICE_USB				FILE_DEVICE_UNKNOWN
#endif

#ifndef CTL_CODE
#define CTL_CODE(DeviceType, Function, Method, Access) \
	(((DeviceType) << 16) | ((Access) << 14) | ((Function) << 2) | (Method))
#endif

typedef enum USB_CONNECTION_STATUS {
	NoDeviceConnected,
	DeviceConnected,
	DeviceFailedEnumeration,
	DeviceGeneralFailure,
	DeviceCausedOvercurrent,
	DeviceNotEnoughPower,
	DeviceNotEnoughBandwidth,
	DeviceHubNestedTooDeeply,
	DeviceInLegacyHub
} USB_CONNECTION_STATUS, *PUSB_CONNECTION_STATUS;

typedef enum USB_HUB_NODE {
	UsbHub,
	UsbMIParent
} USB_HUB_NODE;

/* Cfgmgr32.dll interface */
DLL_DECLARE_HANDLE(Cfgmgr32);
DLL_DECLARE_FUNC(WINAPI, CONFIGRET, CM_Get_Parent, (PDEVINST, DEVINST, ULONG));
DLL_DECLARE_FUNC(WINAPI, CONFIGRET, CM_Get_Child, (PDEVINST, DEVINST, ULONG));
DLL_DECLARE_FUNC(WINAPI, CONFIGRET, CM_Get_Sibling, (PDEVINST, DEVINST, ULONG));
DLL_DECLARE_FUNC(WINAPI, CONFIGRET, CM_Get_Device_IDA, (DEVINST, PCHAR, ULONG, ULONG));

#define IOCTL_USB_GET_DESCRIPTOR_FROM_NODE_CONNECTION \
	CTL_CODE(FILE_DEVICE_USB, USB_GET_DESCRIPTOR_FROM_NODE_CONNECTION, METHOD_BUFFERED, FILE_ANY_ACCESS)

#define IOCTL_USB_GET_NODE_CONNECTION_INFORMATION_EX \
	CTL_CODE(FILE_DEVICE_USB, USB_GET_NODE_CONNECTION_INFORMATION_EX, METHOD_BUFFERED, FILE_ANY_ACCESS)

#define IOCTL_USB_GET_NODE_CONNECTION_INFORMATION_EX_V2 \
	CTL_CODE(FILE_DEVICE_USB, USB_GET_NODE_CONNECTION_INFORMATION_EX_V2, METHOD_BUFFERED, FILE_ANY_ACCESS)

// Most of the structures below need to be packed
#include <pshpack1.h>

typedef struct _USB_DESCRIPTOR_REQUEST {
	ULONG ConnectionIndex;
	struct {
		UCHAR bmRequest;
		UCHAR bRequest;
		USHORT wValue;
		USHORT wIndex;
		USHORT wLength;
	} SetupPacket;
//	UCHAR Data[0];
} USB_DESCRIPTOR_REQUEST, *PUSB_DESCRIPTOR_REQUEST;

typedef struct _USB_CONFIGURATION_DESCRIPTOR_SHORT {
	USB_DESCRIPTOR_REQUEST req;
	USB_CONFIGURATION_DESCRIPTOR desc;
} USB_CONFIGURATION_DESCRIPTOR_SHORT;

typedef struct USB_INTERFACE_DESCRIPTOR {
	UCHAR bLength;
	UCHAR bDescriptorType;
	UCHAR bInterfaceNumber;
	UCHAR bAlternateSetting;
	UCHAR bNumEndpoints;
	UCHAR bInterfaceClass;
	UCHAR bInterfaceSubClass;
	UCHAR bInterfaceProtocol;
	UCHAR iInterface;
} USB_INTERFACE_DESCRIPTOR, *PUSB_INTERFACE_DESCRIPTOR;

typedef struct _USB_NODE_CONNECTION_INFORMATION_EX {
	ULONG ConnectionIndex;
	USB_DEVICE_DESCRIPTOR DeviceDescriptor;
	UCHAR CurrentConfigurationValue;
	UCHAR Speed;
	BOOLEAN DeviceIsHub;
	USHORT DeviceAddress;
	ULONG NumberOfOpenPipes;
	USB_CONNECTION_STATUS ConnectionStatus;
//	USB_PIPE_INFO PipeList[0];
} USB_NODE_CONNECTION_INFORMATION_EX, *PUSB_NODE_CONNECTION_INFORMATION_EX;

typedef union _USB_PROTOCOLS {
	ULONG ul;
	struct {
		ULONG Usb110:1;
		ULONG Usb200:1;
		ULONG Usb300:1;
		ULONG ReservedMBZ:29;
	};
} USB_PROTOCOLS, *PUSB_PROTOCOLS;

typedef union _USB_NODE_CONNECTION_INFORMATION_EX_V2_FLAGS {
	ULONG ul;
	struct {
		ULONG DeviceIsOperatingAtSuperSpeedOrHigher:1;
		ULONG DeviceIsSuperSpeedCapableOrHigher:1;
		ULONG ReservedMBZ:30;
	};
} USB_NODE_CONNECTION_INFORMATION_EX_V2_FLAGS, *PUSB_NODE_CONNECTION_INFORMATION_EX_V2_FLAGS;

typedef struct _USB_NODE_CONNECTION_INFORMATION_EX_V2 {
	ULONG ConnectionIndex;
	ULONG Length;
	USB_PROTOCOLS SupportedUsbProtocols;
	USB_NODE_CONNECTION_INFORMATION_EX_V2_FLAGS Flags;
} USB_NODE_CONNECTION_INFORMATION_EX_V2, *PUSB_NODE_CONNECTION_INFORMATION_EX_V2;

#include <poppack.h>

/* winusb.dll interface */

#define SHORT_PACKET_TERMINATE	0x01
#define AUTO_CLEAR_STALL	0x02
#define PIPE_TRANSFER_TIMEOUT	0x03
#define IGNORE_SHORT_PACKETS	0x04
#define ALLOW_PARTIAL_READS	0x05
#define AUTO_FLUSH		0x06
#define RAW_IO			0x07
#define MAXIMUM_TRANSFER_SIZE	0x08
#define AUTO_SUSPEND		0x81
#define SUSPEND_DELAY		0x83
#define DEVICE_SPEED		0x01
#define LowSpeed		0x01
#define FullSpeed		0x02
#define HighSpeed		0x03

typedef enum _USBD_PIPE_TYPE {
	UsbdPipeTypeControl,
	UsbdPipeTypeIsochronous,
	UsbdPipeTypeBulk,
	UsbdPipeTypeInterrupt
} USBD_PIPE_TYPE;

typedef struct _WINUSB_PIPE_INFORMATION {
	USBD_PIPE_TYPE PipeType;
	UCHAR PipeId;
	USHORT MaximumPacketSize;
	UCHAR Interval;
} WINUSB_PIPE_INFORMATION, *PWINUSB_PIPE_INFORMATION;

#include <pshpack1.h>

typedef struct _WINUSB_SETUP_PACKET {
	UCHAR RequestType;
	UCHAR Request;
	USHORT Value;
	USHORT Index;
	USHORT Length;
} WINUSB_SETUP_PACKET, *PWINUSB_SETUP_PACKET;

#include <poppack.h>

typedef void *WINUSB_INTERFACE_HANDLE, *PWINUSB_INTERFACE_HANDLE;

typedef BOOL (WINAPI *WinUsb_AbortPipe_t)(
	WINUSB_INTERFACE_HANDLE InterfaceHandle,
	UCHAR PipeID
);
typedef BOOL (WINAPI *WinUsb_ControlTransfer_t)(
	WINUSB_INTERFACE_HANDLE InterfaceHandle,
	WINUSB_SETUP_PACKET SetupPacket,
	PUCHAR Buffer,
	ULONG BufferLength,
	PULONG LengthTransferred,
	LPOVERLAPPED Overlapped
);
typedef BOOL (WINAPI *WinUsb_FlushPipe_t)(
	WINUSB_INTERFACE_HANDLE InterfaceHandle,
	UCHAR PipeID
);
typedef BOOL (WINAPI *WinUsb_Free_t)(
	WINUSB_INTERFACE_HANDLE InterfaceHandle
);
typedef BOOL (WINAPI *WinUsb_GetAssociatedInterface_t)(
	WINUSB_INTERFACE_HANDLE InterfaceHandle,
	UCHAR AssociatedInterfaceIndex,
	PWINUSB_INTERFACE_HANDLE AssociatedInterfaceHandle
);
typedef BOOL (WINAPI *WinUsb_GetCurrentAlternateSetting_t)(
	WINUSB_INTERFACE_HANDLE InterfaceHandle,
	PUCHAR AlternateSetting
);
typedef BOOL (WINAPI *WinUsb_GetDescriptor_t)(
	WINUSB_INTERFACE_HANDLE InterfaceHandle,
	UCHAR DescriptorType,
	UCHAR Index,
	USHORT LanguageID,
	PUCHAR Buffer,
	ULONG BufferLength,
	PULONG LengthTransferred
);
typedef BOOL (WINAPI *WinUsb_GetOverlappedResult_t)(
	WINUSB_INTERFACE_HANDLE InterfaceHandle,
	LPOVERLAPPED lpOverlapped,
	LPDWORD lpNumberOfBytesTransferred,
	BOOL bWait
);
typedef BOOL (WINAPI *WinUsb_GetPipePolicy_t)(
	WINUSB_INTERFACE_HANDLE InterfaceHandle,
	UCHAR PipeID,
	ULONG PolicyType,
	PULONG ValueLength,
	PVOID Value
);
typedef BOOL (WINAPI *WinUsb_GetPowerPolicy_t)(
	WINUSB_INTERFACE_HANDLE InterfaceHandle,
	ULONG PolicyType,
	PULONG ValueLength,
	PVOID Value
);
typedef BOOL (WINAPI *WinUsb_Initialize_t)(
	HANDLE DeviceHandle,
	PWINUSB_INTERFACE_HANDLE InterfaceHandle
);
typedef BOOL (WINAPI *WinUsb_QueryDeviceInformation_t)(
	WINUSB_INTERFACE_HANDLE InterfaceHandle,
	ULONG InformationType,
	PULONG BufferLength,
	PVOID Buffer
);
typedef BOOL (WINAPI *WinUsb_QueryInterfaceSettings_t)(
	WINUSB_INTERFACE_HANDLE InterfaceHandle,
	UCHAR AlternateSettingNumber,
	PUSB_INTERFACE_DESCRIPTOR UsbAltInterfaceDescriptor
);
typedef BOOL (WINAPI *WinUsb_QueryPipe_t)(
	WINUSB_INTERFACE_HANDLE InterfaceHandle,
	UCHAR AlternateInterfaceNumber,
	UCHAR PipeIndex,
	PWINUSB_PIPE_INFORMATION PipeInformation
);
typedef BOOL (WINAPI *WinUsb_ReadPipe_t)(
	WINUSB_INTERFACE_HANDLE InterfaceHandle,
	UCHAR PipeID,
	PUCHAR Buffer,
	ULONG BufferLength,
	PULONG LengthTransferred,
	LPOVERLAPPED Overlapped
);
typedef BOOL (WINAPI *WinUsb_ResetPipe_t)(
	WINUSB_INTERFACE_HANDLE InterfaceHandle,
	UCHAR PipeID
);
typedef BOOL (WINAPI *WinUsb_SetCurrentAlternateSetting_t)(
	WINUSB_INTERFACE_HANDLE InterfaceHandle,
	UCHAR AlternateSetting
);
typedef BOOL (WINAPI *WinUsb_SetPipePolicy_t)(
	WINUSB_INTERFACE_HANDLE InterfaceHandle,
	UCHAR PipeID,
	ULONG PolicyType,
	ULONG ValueLength,
	PVOID Value
);
typedef BOOL (WINAPI *WinUsb_SetPowerPolicy_t)(
	WINUSB_INTERFACE_HANDLE InterfaceHandle,
	ULONG PolicyType,
	ULONG ValueLength,
	PVOID Value
);
typedef BOOL (WINAPI *WinUsb_WritePipe_t)(
	WINUSB_INTERFACE_HANDLE InterfaceHandle,
	UCHAR PipeID,
	PUCHAR Buffer,
	ULONG BufferLength,
	PULONG LengthTransferred,
	LPOVERLAPPED Overlapped
);
typedef BOOL (WINAPI *WinUsb_ResetDevice_t)(
	WINUSB_INTERFACE_HANDLE InterfaceHandle
);

/* /!\ These must match the ones from the official libusbk.h */
typedef enum _KUSB_FNID {
	KUSB_FNID_Init,
	KUSB_FNID_Free,
	KUSB_FNID_ClaimInterface,
	KUSB_FNID_ReleaseInterface,
	KUSB_FNID_SetAltInterface,
	KUSB_FNID_GetAltInterface,
	KUSB_FNID_GetDescriptor,
	KUSB_FNID_ControlTransfer,
	KUSB_FNID_SetPowerPolicy,
	KUSB_FNID_GetPowerPolicy,
	KUSB_FNID_SetConfiguration,
	KUSB_FNID_GetConfiguration,
	KUSB_FNID_ResetDevice,
	KUSB_FNID_Initialize,
	KUSB_FNID_SelectInterface,
	KUSB_FNID_GetAssociatedInterface,
	KUSB_FNID_Clone,
	KUSB_FNID_QueryInterfaceSettings,
	KUSB_FNID_QueryDeviceInformation,
	KUSB_FNID_SetCurrentAlternateSetting,
	KUSB_FNID_GetCurrentAlternateSetting,
	KUSB_FNID_QueryPipe,
	KUSB_FNID_SetPipePolicy,
	KUSB_FNID_GetPipePolicy,
	KUSB_FNID_ReadPipe,
	KUSB_FNID_WritePipe,
	KUSB_FNID_ResetPipe,
	KUSB_FNID_AbortPipe,
	KUSB_FNID_FlushPipe,
	KUSB_FNID_IsoReadPipe,
	KUSB_FNID_IsoWritePipe,
	KUSB_FNID_GetCurrentFrameNumber,
	KUSB_FNID_GetOverlappedResult,
	KUSB_FNID_GetProperty,
	KUSB_FNID_COUNT,
} KUSB_FNID;

typedef struct _KLIB_VERSION {
	INT Major;
	INT Minor;
	INT Micro;
	INT Nano;
} KLIB_VERSION;
typedef KLIB_VERSION* PKLIB_VERSION;

typedef BOOL (WINAPI *LibK_GetProcAddress_t)(
	PVOID *ProcAddress,
	ULONG DriverID,
	ULONG FunctionID
);

typedef VOID (WINAPI *LibK_GetVersion_t)(
	PKLIB_VERSION Version
);

struct winusb_interface {
	bool initialized;
	WinUsb_AbortPipe_t AbortPipe;
	WinUsb_ControlTransfer_t ControlTransfer;
	WinUsb_FlushPipe_t FlushPipe;
	WinUsb_Free_t Free;
	WinUsb_GetAssociatedInterface_t GetAssociatedInterface;
	WinUsb_GetCurrentAlternateSetting_t GetCurrentAlternateSetting;
	WinUsb_GetDescriptor_t GetDescriptor;
	WinUsb_GetOverlappedResult_t GetOverlappedResult;
	WinUsb_GetPipePolicy_t GetPipePolicy;
	WinUsb_GetPowerPolicy_t GetPowerPolicy;
	WinUsb_Initialize_t Initialize;
	WinUsb_QueryDeviceInformation_t QueryDeviceInformation;
	WinUsb_QueryInterfaceSettings_t QueryInterfaceSettings;
	WinUsb_QueryPipe_t QueryPipe;
	WinUsb_ReadPipe_t ReadPipe;
	WinUsb_ResetPipe_t ResetPipe;
	WinUsb_SetCurrentAlternateSetting_t SetCurrentAlternateSetting;
	WinUsb_SetPipePolicy_t SetPipePolicy;
	WinUsb_SetPowerPolicy_t SetPowerPolicy;
	WinUsb_WritePipe_t WritePipe;
	WinUsb_ResetDevice_t ResetDevice;
};

/* hid.dll interface */

#define HIDP_STATUS_SUCCESS	0x110000
typedef void * PHIDP_PREPARSED_DATA;

#include <pshpack1.h>

typedef struct _HIDD_ATTIRBUTES {
	ULONG Size;
	USHORT VendorID;
	USHORT ProductID;
	USHORT VersionNumber;
} HIDD_ATTRIBUTES, *PHIDD_ATTRIBUTES;

#include <poppack.h>

typedef USHORT USAGE;
typedef struct _HIDP_CAPS {
	USAGE Usage;
	USAGE UsagePage;
	USHORT InputReportByteLength;
	USHORT OutputReportByteLength;
	USHORT FeatureReportByteLength;
	USHORT Reserved[17];
	USHORT NumberLinkCollectionNodes;
	USHORT NumberInputButtonCaps;
	USHORT NumberInputValueCaps;
	USHORT NumberInputDataIndices;
	USHORT NumberOutputButtonCaps;
	USHORT NumberOutputValueCaps;
	USHORT NumberOutputDataIndices;
	USHORT NumberFeatureButtonCaps;
	USHORT NumberFeatureValueCaps;
	USHORT NumberFeatureDataIndices;
} HIDP_CAPS, *PHIDP_CAPS;

typedef enum _HIDP_REPORT_TYPE {
	HidP_Input,
	HidP_Output,
	HidP_Feature
} HIDP_REPORT_TYPE;

typedef struct _HIDP_VALUE_CAPS {
	USAGE UsagePage;
	UCHAR ReportID;
	BOOLEAN IsAlias;
	USHORT BitField;
	USHORT LinkCollection;
	USAGE LinkUsage;
	USAGE LinkUsagePage;
	BOOLEAN IsRange;
	BOOLEAN IsStringRange;
	BOOLEAN IsDesignatorRange;
	BOOLEAN IsAbsolute;
	BOOLEAN HasNull;
	UCHAR Reserved;
	USHORT BitSize;
	USHORT ReportCount;
	USHORT Reserved2[5];
	ULONG UnitsExp;
	ULONG Units;
	LONG LogicalMin, LogicalMax;
	LONG PhysicalMin, PhysicalMax;
	union {
		struct {
			USAGE UsageMin, UsageMax;
			USHORT StringMin, StringMax;
			USHORT DesignatorMin, DesignatorMax;
			USHORT DataIndexMin, DataIndexMax;
		} Range;
		struct {
			USAGE Usage, Reserved1;
			USHORT StringIndex, Reserved2;
			USHORT DesignatorIndex, Reserved3;
			USHORT DataIndex, Reserved4;
		} NotRange;
	} u;
} HIDP_VALUE_CAPS, *PHIDP_VALUE_CAPS;

DLL_DECLARE_HANDLE(hid);
DLL_DECLARE_FUNC(WINAPI, BOOL, HidD_GetAttributes, (HANDLE, PHIDD_ATTRIBUTES));
DLL_DECLARE_FUNC(WINAPI, VOID, HidD_GetHidGuid, (LPGUID));
DLL_DECLARE_FUNC(WINAPI, BOOL, HidD_GetPreparsedData, (HANDLE, PHIDP_PREPARSED_DATA *));
DLL_DECLARE_FUNC(WINAPI, BOOL, HidD_FreePreparsedData, (PHIDP_PREPARSED_DATA));
DLL_DECLARE_FUNC(WINAPI, BOOL, HidD_GetManufacturerString, (HANDLE, PVOID, ULONG));
DLL_DECLARE_FUNC(WINAPI, BOOL, HidD_GetProductString, (HANDLE, PVOID, ULONG));
DLL_DECLARE_FUNC(WINAPI, BOOL, HidD_GetSerialNumberString, (HANDLE, PVOID, ULONG));
DLL_DECLARE_FUNC(WINAPI, LONG, HidP_GetCaps, (PHIDP_PREPARSED_DATA, PHIDP_CAPS));
DLL_DECLARE_FUNC(WINAPI, BOOL, HidD_SetNumInputBuffers, (HANDLE, ULONG));
DLL_DECLARE_FUNC(WINAPI, BOOL, HidD_SetFeature, (HANDLE, PVOID, ULONG));
DLL_DECLARE_FUNC(WINAPI, BOOL, HidD_GetFeature, (HANDLE, PVOID, ULONG));
DLL_DECLARE_FUNC(WINAPI, BOOL, HidD_GetPhysicalDescriptor, (HANDLE, PVOID, ULONG));
DLL_DECLARE_FUNC(WINAPI, BOOL, HidD_GetInputReport, (HANDLE, PVOID, ULONG));
DLL_DECLARE_FUNC(WINAPI, BOOL, HidD_SetOutputReport, (HANDLE, PVOID, ULONG));
DLL_DECLARE_FUNC(WINAPI, BOOL, HidD_FlushQueue, (HANDLE));
DLL_DECLARE_FUNC(WINAPI, BOOL, HidP_GetValueCaps, (HIDP_REPORT_TYPE, PHIDP_VALUE_CAPS, PULONG, PHIDP_PREPARSED_DATA));