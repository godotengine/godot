/*
* windows UsbDk backend for libusb 1.0
* Copyright © 2014 Red Hat, Inc.

* Authors:
* Dmitry Fleytman <dmitry@daynix.com>
* Pavel Gurvich <pavel@daynix.com>
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

typedef struct tag_USB_DK_DEVICE_ID {
	WCHAR DeviceID[MAX_DEVICE_ID_LEN];
	WCHAR InstanceID[MAX_DEVICE_ID_LEN];
} USB_DK_DEVICE_ID, *PUSB_DK_DEVICE_ID;

static inline void UsbDkFillIDStruct(USB_DK_DEVICE_ID *ID, PCWCHAR DeviceID, PCWCHAR InstanceID)
{
	wcsncpy_s(ID->DeviceID, DeviceID, MAX_DEVICE_ID_LEN);
	wcsncpy_s(ID->InstanceID, InstanceID, MAX_DEVICE_ID_LEN);
}

typedef struct tag_USB_DK_DEVICE_INFO {
	USB_DK_DEVICE_ID ID;
	ULONG64 FilterID;
	ULONG64 Port;
	ULONG64 Speed;
	USB_DEVICE_DESCRIPTOR DeviceDescriptor;
} USB_DK_DEVICE_INFO, *PUSB_DK_DEVICE_INFO;

typedef struct tag_USB_DK_CONFIG_DESCRIPTOR_REQUEST {
	USB_DK_DEVICE_ID ID;
	ULONG64 Index;
} USB_DK_CONFIG_DESCRIPTOR_REQUEST, *PUSB_DK_CONFIG_DESCRIPTOR_REQUEST;

typedef struct tag_USB_DK_ISO_TRANSFER_RESULT {
	ULONG64 ActualLength;
	ULONG64 TransferResult;
} USB_DK_ISO_TRANSFER_RESULT, *PUSB_DK_ISO_TRANSFER_RESULT;

typedef struct tag_USB_DK_GEN_TRANSFER_RESULT {
	ULONG64 BytesTransferred;
	ULONG64 UsbdStatus; // USBD_STATUS code
} USB_DK_GEN_TRANSFER_RESULT, *PUSB_DK_GEN_TRANSFER_RESULT;

typedef struct tag_USB_DK_TRANSFER_RESULT {
	USB_DK_GEN_TRANSFER_RESULT GenResult;
	PVOID64 IsochronousResultsArray; // array of USB_DK_ISO_TRANSFER_RESULT
} USB_DK_TRANSFER_RESULT, *PUSB_DK_TRANSFER_RESULT;

typedef struct tag_USB_DK_TRANSFER_REQUEST {
	ULONG64 EndpointAddress;
	PVOID64 Buffer;
	ULONG64 BufferLength;
	ULONG64 TransferType;
	ULONG64 IsochronousPacketsArraySize;
	PVOID64 IsochronousPacketsArray;

	USB_DK_TRANSFER_RESULT Result;
} USB_DK_TRANSFER_REQUEST, *PUSB_DK_TRANSFER_REQUEST;

typedef enum {
	TransferFailure = 0,
	TransferSuccess,
	TransferSuccessAsync
} TransferResult;

typedef enum {
	NoSpeed = 0,
	LowSpeed,
	FullSpeed,
	HighSpeed,
	SuperSpeed
} USB_DK_DEVICE_SPEED;

typedef enum {
	ControlTransferType,
	BulkTransferType,
	IntertuptTransferType,
	IsochronousTransferType
} USB_DK_TRANSFER_TYPE;

typedef BOOL (__cdecl *USBDK_GET_DEVICES_LIST)(
	PUSB_DK_DEVICE_INFO *DeviceInfo,
	PULONG DeviceNumber
);
typedef void (__cdecl *USBDK_RELEASE_DEVICES_LIST)(
	PUSB_DK_DEVICE_INFO DeviceInfo
);
typedef HANDLE (__cdecl *USBDK_START_REDIRECT)(
	PUSB_DK_DEVICE_ID DeviceId
);
typedef BOOL (__cdecl *USBDK_STOP_REDIRECT)(
	HANDLE DeviceHandle
);
typedef BOOL (__cdecl *USBDK_GET_CONFIGURATION_DESCRIPTOR)(
	PUSB_DK_CONFIG_DESCRIPTOR_REQUEST Request,
	PUSB_CONFIGURATION_DESCRIPTOR *Descriptor,
	PULONG Length
);
typedef void (__cdecl *USBDK_RELEASE_CONFIGURATION_DESCRIPTOR)(
	PUSB_CONFIGURATION_DESCRIPTOR Descriptor
);
typedef TransferResult (__cdecl *USBDK_WRITE_PIPE)(
	HANDLE DeviceHandle,
	PUSB_DK_TRANSFER_REQUEST Request,
	LPOVERLAPPED lpOverlapped
);
typedef TransferResult (__cdecl *USBDK_READ_PIPE)(
	HANDLE DeviceHandle,
	PUSB_DK_TRANSFER_REQUEST Request,
	LPOVERLAPPED lpOverlapped
);
typedef BOOL (__cdecl *USBDK_ABORT_PIPE)(
	HANDLE DeviceHandle,
	ULONG64 PipeAddress
);
typedef BOOL (__cdecl *USBDK_RESET_PIPE)(
	HANDLE DeviceHandle,
	ULONG64 PipeAddress
);
typedef BOOL (__cdecl *USBDK_SET_ALTSETTING)(
	HANDLE DeviceHandle,
	ULONG64 InterfaceIdx,
	ULONG64 AltSettingIdx
);
typedef BOOL (__cdecl *USBDK_RESET_DEVICE)(
	HANDLE DeviceHandle
);
typedef HANDLE (__cdecl *USBDK_GET_REDIRECTOR_SYSTEM_HANDLE)(
	HANDLE DeviceHandle
);
