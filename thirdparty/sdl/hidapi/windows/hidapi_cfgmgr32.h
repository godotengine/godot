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

#ifndef HIDAPI_CFGMGR32_H
#define HIDAPI_CFGMGR32_H

#ifdef HIDAPI_USE_DDK

#include <cfgmgr32.h>
#include <initguid.h>
#include <devpkey.h>
#include <propkey.h>

#else

/* This part of the header mimics cfgmgr32.h,
    but only what is used by HIDAPI */

//#include <initguid.h>
#include <devpropdef.h>
//#include <propkeydef.h>

#ifndef PROPERTYKEY_DEFINED
#define PROPERTYKEY_DEFINED

typedef struct
{
    GUID fmtid;
    DWORD pid;
} PROPERTYKEY;

#endif /* PROPERTYKEY_DEFINED */

typedef DWORD RETURN_TYPE;
typedef RETURN_TYPE CONFIGRET;
typedef DWORD DEVNODE, DEVINST;
typedef DEVNODE* PDEVNODE, * PDEVINST;
typedef WCHAR* DEVNODEID_W, * DEVINSTID_W;

#define CR_SUCCESS (0x00000000)
#define CR_BUFFER_SMALL (0x0000001A)
#define CR_FAILURE (0x00000013)

#define CM_LOCATE_DEVNODE_NORMAL 0x00000000

#define CM_GET_DEVICE_INTERFACE_LIST_PRESENT (0x00000000)

typedef CONFIGRET(__stdcall* CM_Locate_DevNodeW_)(PDEVINST pdnDevInst, DEVINSTID_W pDeviceID, ULONG ulFlags);
typedef CONFIGRET(__stdcall* CM_Get_Parent_)(PDEVINST pdnDevInst, DEVINST dnDevInst, ULONG ulFlags);
typedef CONFIGRET(__stdcall* CM_Get_DevNode_PropertyW_)(DEVINST dnDevInst, CONST DEVPROPKEY* PropertyKey, DEVPROPTYPE* PropertyType, PBYTE PropertyBuffer, PULONG PropertyBufferSize, ULONG ulFlags);
typedef CONFIGRET(__stdcall* CM_Get_Device_Interface_PropertyW_)(LPCWSTR pszDeviceInterface, CONST DEVPROPKEY* PropertyKey, DEVPROPTYPE* PropertyType, PBYTE PropertyBuffer, PULONG PropertyBufferSize, ULONG ulFlags);
typedef CONFIGRET(__stdcall* CM_Get_Device_Interface_List_SizeW_)(PULONG pulLen, LPGUID InterfaceClassGuid, DEVINSTID_W pDeviceID, ULONG ulFlags);
typedef CONFIGRET(__stdcall* CM_Get_Device_Interface_ListW_)(LPGUID InterfaceClassGuid, DEVINSTID_W pDeviceID, WCHAR* /*PZZWSTR*/ Buffer, ULONG BufferLen, ULONG ulFlags);

// from devpkey.h
static DEVPROPKEY DEVPKEY_NAME = { { 0xb725f130, 0x47ef, 0x101a, {0xa5, 0xf1, 0x02, 0x60, 0x8c, 0x9e, 0xeb, 0xac} }, 10 }; // DEVPROP_TYPE_STRING
static DEVPROPKEY DEVPKEY_Device_Manufacturer = { { 0xa45c254e, 0xdf1c, 0x4efd, {0x80, 0x20, 0x67, 0xd1, 0x46, 0xa8, 0x50, 0xe0} }, 13 }; // DEVPROP_TYPE_STRING
static DEVPROPKEY DEVPKEY_Device_InstanceId = { { 0x78c34fc8, 0x104a, 0x4aca, {0x9e, 0xa4, 0x52, 0x4d, 0x52, 0x99, 0x6e, 0x57} }, 256 }; // DEVPROP_TYPE_STRING
static DEVPROPKEY DEVPKEY_Device_HardwareIds = { { 0xa45c254e, 0xdf1c, 0x4efd, {0x80, 0x20, 0x67, 0xd1, 0x46, 0xa8, 0x50, 0xe0} }, 3 }; // DEVPROP_TYPE_STRING_LIST
static DEVPROPKEY DEVPKEY_Device_CompatibleIds = { { 0xa45c254e, 0xdf1c, 0x4efd, {0x80, 0x20, 0x67, 0xd1, 0x46, 0xa8, 0x50, 0xe0} }, 4 }; // DEVPROP_TYPE_STRING_LIST
static DEVPROPKEY DEVPKEY_Device_ContainerId = { { 0x8c7ed206, 0x3f8a, 0x4827, {0xb3, 0xab, 0xae, 0x9e, 0x1f, 0xae, 0xfc, 0x6c} }, 2 }; // DEVPROP_TYPE_GUID

// from propkey.h
static PROPERTYKEY PKEY_DeviceInterface_Bluetooth_DeviceAddress = { { 0x2bd67d8b, 0x8beb, 0x48d5, {0x87, 0xe0, 0x6c, 0xda, 0x34, 0x28, 0x04, 0x0a} }, 1 }; // DEVPROP_TYPE_STRING
static PROPERTYKEY PKEY_DeviceInterface_Bluetooth_Manufacturer = { { 0x2bd67d8b, 0x8beb, 0x48d5, {0x87, 0xe0, 0x6c, 0xda, 0x34, 0x28, 0x04, 0x0a} }, 4 }; // DEVPROP_TYPE_STRING
static PROPERTYKEY PKEY_DeviceInterface_Bluetooth_ModelNumber = { { 0x2BD67D8B, 0x8BEB, 0x48D5, {0x87, 0xE0, 0x6C, 0xDA, 0x34, 0x28, 0x04, 0x0A} }, 5 }; // DEVPROP_TYPE_STRING

#endif

#endif /* HIDAPI_CFGMGR32_H */
