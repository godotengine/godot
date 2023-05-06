/**************************************************************************/
/*  hid_proxy.cpp                                                         */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include <windows.h>
#include <cstdint>
#include <cstring>

// This X macro contains a list of all the exports in the original `hid.dll` shared library.
#define HID_ORIGINAL_EXPORTS                \
	X(HidD_FlushQueue)                      \
	X(HidD_FreePreparsedData)               \
	X(HidD_GetAttributes)                   \
	X(HidD_GetConfiguration)                \
	X(HidD_GetFeature)                      \
	X(HidD_GetHidGuid)                      \
	X(HidD_GetIndexedString)                \
	X(HidD_GetInputReport)                  \
	X(HidD_GetManufacturerString)           \
	X(HidD_GetMsGenreDescriptor)            \
	X(HidD_GetNumInputBuffers)              \
	X(HidD_GetPhysicalDescriptor)           \
	X(HidD_GetPreparsedData)                \
	X(HidD_GetProductString)                \
	X(HidD_GetSerialNumberString)           \
	X(HidD_Hello)                           \
	X(HidD_SetConfiguration)                \
	X(HidD_SetFeature)                      \
	X(HidD_SetNumInputBuffers)              \
	X(HidD_SetOutputReport)                 \
	X(HidP_GetButtonArray)                  \
	X(HidP_GetButtonCaps)                   \
	X(HidP_GetCaps)                         \
	X(HidP_GetData)                         \
	X(HidP_GetExtendedAttributes)           \
	X(HidP_GetLinkCollectionNodes)          \
	X(HidP_GetScaledUsageValue)             \
	X(HidP_GetSpecificButtonCaps)           \
	X(HidP_GetSpecificValueCaps)            \
	X(HidP_GetUsageValue)                   \
	X(HidP_GetUsageValueArray)              \
	X(HidP_GetUsages)                       \
	X(HidP_GetUsagesEx)                     \
	X(HidP_GetValueCaps)                    \
	X(HidP_GetVersionInternal)              \
	X(HidP_InitializeReportForID)           \
	X(HidP_MaxDataListLength)               \
	X(HidP_MaxUsageListLength)              \
	X(HidP_SetButtonArray)                  \
	X(HidP_SetData)                         \
	X(HidP_SetScaledUsageValue)             \
	X(HidP_SetUsageValue)                   \
	X(HidP_SetUsageValueArray)              \
	X(HidP_SetUsages)                       \
	X(HidP_TranslateUsagesToI8042ScanCodes) \
	X(HidP_UnsetUsages)                     \
	X(HidP_UsageListDifference)

// For each export in `hid.dll`, generate a pointer that contains its address in the original module.
#define X(name) void *hid_##name;
HID_ORIGINAL_EXPORTS;
#undef X

// Nasty `hidpi.h` stuff which can't be selectively included.
#define USAGE USHORT

struct HIDP_CAPS {
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
};

#define HID_USAGE_PAGE_GENERIC 0x01
#define HID_USAGE_GENERIC_JOYSTICK 0x04
#define HID_USAGE_GENERIC_GAMEPAD 0x05
#define HID_USAGE_GENERIC_MULTI_AXIS_CONTROLLER 0x08

#ifndef FACILITY_HID_ERROR_CODE
#define FACILITY_HID_ERROR_CODE 0x11
#endif

#define HIDP_ERROR_CODES(SEV, CODE) \
	((NTSTATUS)(((SEV) << 28) | (FACILITY_HID_ERROR_CODE << 16) | (CODE)))

#define HIDP_STATUS_SUCCESS (HIDP_ERROR_CODES(0x0, 0))

void hid_proxy_start() {
	// Locate the original `hid.dll` file.
	char hid_path[MAX_PATH];
	GetSystemDirectoryA(hid_path, MAX_PATH);
	strcat_s(hid_path, "\\hid.dll");

	// Define the original `hid.dll` exports.
	HMODULE hid = LoadLibraryA(hid_path);
#define X(name) hid_##name = reinterpret_cast<void *>(GetProcAddress(hid, #name));
	HID_ORIGINAL_EXPORTS;
#undef X
}

bool hid_is_controller(HANDLE hid_handle) {
	void *hid_preparsed = nullptr;
	BOOLEAN preparsed_res = reinterpret_cast<BOOLEAN(__stdcall *)(HANDLE, void *)>(hid_HidD_GetPreparsedData)(hid_handle, &hid_preparsed);
	if (!preparsed_res) {
		return false;
	}

	HIDP_CAPS hid_caps = {};
	NTSTATUS caps_res = reinterpret_cast<NTSTATUS(__stdcall *)(void *, void *)>(hid_HidP_GetCaps)(hid_preparsed, &hid_caps);
	reinterpret_cast<BOOLEAN(__stdcall *)(void *)>(hid_HidD_FreePreparsedData)(hid_preparsed);
	if (caps_res != HIDP_STATUS_SUCCESS) {
		return false;
	}

	if (hid_caps.UsagePage != HID_USAGE_PAGE_GENERIC) {
		return false;
	}

	if (hid_caps.Usage == HID_USAGE_GENERIC_JOYSTICK || hid_caps.Usage == HID_USAGE_GENERIC_GAMEPAD || hid_caps.Usage == HID_USAGE_GENERIC_MULTI_AXIS_CONTROLLER) {
		return true;
	}

	return false;
}

extern "C" {
__declspec(dllexport) BOOLEAN __stdcall HidD_FlushQueue(HANDLE HidDeviceObject) {
	return reinterpret_cast<BOOLEAN(__stdcall *)(HANDLE)>(hid_HidD_FlushQueue)(HidDeviceObject);
}

__declspec(dllexport) BOOLEAN __stdcall HidD_FreePreparsedData(void *PreparsedData) {
	return reinterpret_cast<BOOLEAN(__stdcall *)(void *)>(hid_HidD_FreePreparsedData)(PreparsedData);
}

__declspec(dllexport) BOOLEAN __stdcall HidD_GetAttributes(HANDLE HidDeviceObject, void *Attributes) {
	return reinterpret_cast<BOOLEAN(__stdcall *)(HANDLE, void *)>(hid_HidD_GetAttributes)(HidDeviceObject, Attributes);
}

__declspec(dllexport) BOOLEAN __stdcall HidD_GetConfiguration(HANDLE HidDeviceObject, void *Configuration, ULONG ConfigurationLength) {
	return reinterpret_cast<BOOLEAN(__stdcall *)(HANDLE, void *, ULONG)>(hid_HidD_GetConfiguration)(HidDeviceObject, Configuration, ConfigurationLength);
}

__declspec(dllexport) BOOLEAN __stdcall HidD_GetFeature(HANDLE HidDeviceObject, void *ReportBuffer, ULONG ReportBufferLength) {
	return reinterpret_cast<BOOLEAN(__stdcall *)(HANDLE, void *, ULONG)>(hid_HidD_GetFeature)(HidDeviceObject, ReportBuffer, ReportBufferLength);
}

__declspec(dllexport) void __stdcall HidD_GetHidGuid(LPGUID HidGuid) {
	reinterpret_cast<void(__stdcall *)(LPGUID)>(hid_HidD_GetHidGuid)(HidGuid);
}

__declspec(dllexport) BOOLEAN __stdcall HidD_GetIndexedString(HANDLE HidDeviceObject, ULONG StringIndex, void *Buffer, ULONG BufferLength) {
	return reinterpret_cast<BOOLEAN(__stdcall *)(HANDLE, ULONG, void *, ULONG)>(hid_HidD_GetIndexedString)(HidDeviceObject, StringIndex, Buffer, BufferLength);
}

__declspec(dllexport) BOOLEAN __stdcall HidD_GetInputReport(HANDLE HidDeviceObject, void *ReportBuffer, ULONG ReportBufferLength) {
	return reinterpret_cast<BOOLEAN(__stdcall *)(HANDLE, void *, ULONG)>(hid_HidD_GetInputReport)(HidDeviceObject, ReportBuffer, ReportBufferLength);
}

__declspec(dllexport) BOOLEAN __stdcall HidD_GetManufacturerString(HANDLE HidDeviceObject, void *Buffer, ULONG BufferLength) {
	return reinterpret_cast<BOOLEAN(__stdcall *)(HANDLE, void *, ULONG)>(hid_HidD_GetManufacturerString)(HidDeviceObject, Buffer, BufferLength);
}

__declspec(dllexport) BOOLEAN __stdcall HidD_GetMsGenreDescriptor(HANDLE HidDeviceObject, void *Buffer, ULONG BufferLength) {
	return reinterpret_cast<BOOLEAN(__stdcall *)(HANDLE, void *, ULONG)>(hid_HidD_GetMsGenreDescriptor)(HidDeviceObject, Buffer, BufferLength);
}

__declspec(dllexport) BOOLEAN __stdcall HidD_GetNumInputBuffers(HANDLE HidDeviceObject, void *NumberBuffers) {
	return reinterpret_cast<BOOLEAN(__stdcall *)(HANDLE, void *)>(hid_HidD_GetNumInputBuffers)(HidDeviceObject, NumberBuffers);
}

__declspec(dllexport) BOOLEAN __stdcall HidD_GetPhysicalDescriptor(HANDLE HidDeviceObject, void *Buffer, ULONG BufferLength) {
	return reinterpret_cast<BOOLEAN(__stdcall *)(HANDLE, void *, ULONG)>(hid_HidD_GetPhysicalDescriptor)(HidDeviceObject, Buffer, BufferLength);
}

__declspec(dllexport) BOOLEAN __stdcall HidD_GetPreparsedData(HANDLE HidDeviceObject, void *PreparsedData) {
	return reinterpret_cast<BOOLEAN(__stdcall *)(HANDLE, void *)>(hid_HidD_GetPreparsedData)(HidDeviceObject, PreparsedData);
}

const wchar_t unknown_product_string[] = L"Unknown HID Device";
size_t unknown_product_length = sizeof(unknown_product_string);
__declspec(dllexport) BOOLEAN __stdcall HidD_GetProductString(HANDLE HidDeviceObject, void *Buffer, ULONG BufferLength) {
	if (hid_is_controller(HidDeviceObject)) {
		return reinterpret_cast<BOOLEAN(__stdcall *)(HANDLE, void *, ULONG)>(hid_HidD_GetProductString)(HidDeviceObject, Buffer, BufferLength);
	}

	// The HID is (probably) not a controller, so we don't care about returning its actual product string.
	// This avoids stalls on `EnumDevices` because DirectInput attempts to enumerate all HIDs, including some DACs
	// and other devices which take too long to respond to those requests, added to the lack of a shorter timeout.
	if (BufferLength >= unknown_product_length) {
		memcpy(Buffer, unknown_product_string, unknown_product_length);
		return TRUE;
	}
	return FALSE;
}

__declspec(dllexport) BOOLEAN __stdcall HidD_GetSerialNumberString(HANDLE HidDeviceObject, void *Buffer, ULONG BufferLength) {
	return reinterpret_cast<BOOLEAN(__stdcall *)(HANDLE, void *, ULONG)>(hid_HidD_GetSerialNumberString)(HidDeviceObject, Buffer, BufferLength);
}

__declspec(dllexport) int __stdcall HidD_Hello(void *Buffer, size_t BufferLength) {
	return reinterpret_cast<int(__stdcall *)(void *, size_t)>(hid_HidD_Hello)(Buffer, BufferLength);
}

__declspec(dllexport) BOOLEAN __stdcall HidD_SetConfiguration(HANDLE HidDeviceObject, void *Configuration, ULONG ConfigurationLength) {
	return reinterpret_cast<BOOLEAN(__stdcall *)(HANDLE, void *, ULONG)>(hid_HidD_SetConfiguration)(HidDeviceObject, Configuration, ConfigurationLength);
}

__declspec(dllexport) BOOLEAN __stdcall HidD_SetFeature(HANDLE HidDeviceObject, void *ReportBuffer, ULONG ReportBufferLength) {
	return reinterpret_cast<BOOLEAN(__stdcall *)(HANDLE, void *, ULONG)>(hid_HidD_SetFeature)(HidDeviceObject, ReportBuffer, ReportBufferLength);
}

__declspec(dllexport) BOOLEAN __stdcall HidD_SetNumInputBuffers(HANDLE HidDeviceObject, ULONG NumberBuffers) {
	return reinterpret_cast<BOOLEAN(__stdcall *)(HANDLE, ULONG)>(hid_HidD_SetNumInputBuffers)(HidDeviceObject, NumberBuffers);
}

__declspec(dllexport) BOOLEAN __stdcall HidD_SetOutputReport(HANDLE HidDeviceObject, void *ReportBuffer, ULONG ReportBufferLength) {
	return reinterpret_cast<BOOLEAN(__stdcall *)(HANDLE, void *, ULONG)>(hid_HidD_SetOutputReport)(HidDeviceObject, ReportBuffer, ReportBufferLength);
}

__declspec(dllexport) LONG __stdcall HidP_GetButtonArray(int ReportType, USAGE UsagePage, USHORT LinkCollection, USAGE Usage, void *ButtonData, void *ButtonDataLength, void *PreparsedData, void *Report, ULONG ReportLength) {
	return reinterpret_cast<LONG(__stdcall *)(int, USAGE, USHORT, USAGE, void *, void *, void *, void *, ULONG)>(hid_HidP_GetButtonArray)(ReportType, UsagePage, LinkCollection, Usage, ButtonData, ButtonDataLength, PreparsedData, Report, ReportLength);
}

__declspec(dllexport) NTSTATUS __stdcall HidP_GetButtonCaps(int ReportType, void *ButtonCaps, void *ButtonCapsLength, void *PreparsedData) {
	return reinterpret_cast<NTSTATUS(__stdcall *)(int, void *, void *, void *)>(hid_HidP_GetButtonCaps)(ReportType, ButtonCaps, ButtonCapsLength, PreparsedData);
}

__declspec(dllexport) NTSTATUS __stdcall HidP_GetCaps(void *PreparsedData, void *Capabilities) {
	return reinterpret_cast<NTSTATUS(__stdcall *)(void *, void *)>(hid_HidP_GetCaps)(PreparsedData, Capabilities);
}

__declspec(dllexport) NTSTATUS __stdcall HidP_GetData(int ReportType, void *DataList, void *DataLength, void *PreparsedData, void *Report, ULONG ReportLength) {
	return reinterpret_cast<NTSTATUS(__stdcall *)(int, void *, void *, void *, void *, ULONG)>(hid_HidP_GetData)(ReportType, DataList, DataLength, PreparsedData, Report, ReportLength);
}

__declspec(dllexport) NTSTATUS __stdcall HidP_GetExtendedAttributes(int ReportType, USHORT DataIndex, void *PreparsedData, void *Attributes, PULONG LengthAttributes) {
	return reinterpret_cast<NTSTATUS(__stdcall *)(int, USHORT, void *, void *, PULONG)>(hid_HidP_GetExtendedAttributes)(ReportType, DataIndex, PreparsedData, Attributes, LengthAttributes);
}

__declspec(dllexport) NTSTATUS __stdcall HidP_GetLinkCollectionNodes(void *LinkCollectionNodes, void *LinkCollectionNodesLength, void *PreparsedData) {
	return reinterpret_cast<NTSTATUS(__stdcall *)(void *, void *, void *)>(hid_HidP_GetLinkCollectionNodes)(LinkCollectionNodes, LinkCollectionNodesLength, PreparsedData);
}

__declspec(dllexport) NTSTATUS __stdcall HidP_GetScaledUsageValue(int ReportType, USAGE UsagePage, USHORT LinkCollection, USAGE Usage, void *UsageValue, void *PreparsedData, void *Report, ULONG ReportLength) {
	return reinterpret_cast<NTSTATUS(__stdcall *)(int, USAGE, USHORT, USAGE, void *, void *, void *, ULONG)>(hid_HidP_GetScaledUsageValue)(ReportType, UsagePage, LinkCollection, Usage, UsageValue, PreparsedData, Report, ReportLength);
}

__declspec(dllexport) NTSTATUS __stdcall HidP_GetSpecificButtonCaps(int ReportType, USAGE UsagePage, USHORT LinkCollection, USAGE Usage, void *ButtonCaps, void *ButtonCapsLength, void *PreparsedData) {
	return reinterpret_cast<NTSTATUS(__stdcall *)(int, USAGE, USHORT, USAGE, void *, void *, void *)>(hid_HidP_GetSpecificButtonCaps)(ReportType, UsagePage, LinkCollection, Usage, ButtonCaps, ButtonCapsLength, PreparsedData);
}

__declspec(dllexport) NTSTATUS __stdcall HidP_GetSpecificValueCaps(int ReportType, USAGE UsagePage, USHORT LinkCollection, USAGE Usage, void *ValueCaps, void *ValueCapsLength, void *PreparsedData) {
	return reinterpret_cast<NTSTATUS(__stdcall *)(int, USAGE, USHORT, USAGE, void *, void *, void *)>(hid_HidP_GetSpecificValueCaps)(ReportType, UsagePage, LinkCollection, Usage, ValueCaps, ValueCapsLength, PreparsedData);
}

__declspec(dllexport) NTSTATUS __stdcall HidP_GetUsageValue(int ReportType, USAGE UsagePage, USHORT LinkCollection, USAGE Usage, void *UsageValue, void *PreparsedData, void *Report, ULONG ReportLength) {
	return reinterpret_cast<NTSTATUS(__stdcall *)(int, USAGE, USHORT, USAGE, void *, void *, void *, ULONG)>(hid_HidP_GetUsageValue)(ReportType, UsagePage, LinkCollection, Usage, UsageValue, PreparsedData, Report, ReportLength);
}

__declspec(dllexport) NTSTATUS __stdcall HidP_GetUsageValueArray(int ReportType, USAGE UsagePage, USHORT LinkCollection, USAGE Usage, void *UsageValue, USHORT UsageValueByteLength, void *PreparsedData, void *Report, ULONG ReportLength) {
	return reinterpret_cast<NTSTATUS(__stdcall *)(int, USAGE, USHORT, USAGE, void *, USHORT, void *, void *, ULONG)>(hid_HidP_GetUsageValueArray)(ReportType, UsagePage, LinkCollection, Usage, UsageValue, UsageValueByteLength, PreparsedData, Report, ReportLength);
}

__declspec(dllexport) NTSTATUS __stdcall HidP_GetUsages(int ReportType, USAGE Usage, USHORT LinkCollection, void *UsageList, void *UsageLength, void *PreparsedData, void *Report, ULONG ReportLength) {
	return reinterpret_cast<NTSTATUS(__stdcall *)(int, USAGE, USHORT, void *, void *, void *, void *, ULONG)>(hid_HidP_GetUsages)(ReportType, Usage, LinkCollection, UsageList, UsageLength, PreparsedData, Report, ReportLength);
}

__declspec(dllexport) NTSTATUS __stdcall HidP_GetUsagesEx(int ReportType, USHORT LinkCollection, void *ButtonList, void *UsageLength, void *PreparsedData, void *Report, ULONG ReportLength) {
	return reinterpret_cast<NTSTATUS(__stdcall *)(int, USHORT, void *, void *, void *, void *, ULONG)>(hid_HidP_GetUsagesEx)(ReportType, LinkCollection, ButtonList, UsageLength, PreparsedData, Report, ReportLength);
}

__declspec(dllexport) NTSTATUS __stdcall HidP_GetValueCaps(int ReportType, void *ValueCaps, void *ValueCapsLength, void *PreparsedData) {
	return reinterpret_cast<NTSTATUS(__stdcall *)(int, void *, void *, void *)>(hid_HidP_GetValueCaps)(ReportType, ValueCaps, ValueCapsLength, PreparsedData);
}

__declspec(dllexport) int __stdcall HidP_GetVersionInternal(void *Version) {
	return reinterpret_cast<int(__stdcall *)(void *)>(hid_HidP_GetVersionInternal)(Version);
}

__declspec(dllexport) NTSTATUS __stdcall HidP_InitializeReportForID(int ReportType, UCHAR ReportID, void *PreparsedData, void *Report, ULONG ReportLength) {
	return reinterpret_cast<NTSTATUS(__stdcall *)(int, UCHAR, void *, void *, ULONG)>(hid_HidP_InitializeReportForID)(ReportType, ReportID, PreparsedData, Report, ReportLength);
}

__declspec(dllexport) ULONG __stdcall HidP_MaxDataListLength(int ReportType, void *PreparsedData) {
	return reinterpret_cast<ULONG(__stdcall *)(int, void *)>(hid_HidP_MaxDataListLength)(ReportType, PreparsedData);
}

__declspec(dllexport) ULONG __stdcall HidP_MaxUsageListLength(int ReportType, USAGE UsagePage, void *PreparsedData) {
	return reinterpret_cast<ULONG(__stdcall *)(int, USAGE, void *)>(hid_HidP_MaxUsageListLength)(ReportType, UsagePage, PreparsedData);
}

__declspec(dllexport) NTSTATUS __stdcall HidP_SetButtonArray(int ReportType, USAGE UsagePage, USHORT LinkCollection, USAGE Usage, void *ButtonData, USHORT ButtonDataLength, void *PreparsedData, void *Report, ULONG ReportLength) {
	return reinterpret_cast<NTSTATUS(__stdcall *)(int, USAGE, USHORT, USAGE, void *, USHORT, void *, void *, ULONG)>(hid_HidP_SetButtonArray)(ReportType, UsagePage, LinkCollection, Usage, ButtonData, ButtonDataLength, PreparsedData, Report, ReportLength);
}

__declspec(dllexport) NTSTATUS __stdcall HidP_SetData(int ReportType, void *DataList, void *DataLength, void *PreparsedData, void *Report, ULONG ReportLength) {
	return reinterpret_cast<NTSTATUS(__stdcall *)(int, void *, void *, void *, void *, ULONG)>(hid_HidP_SetData)(ReportType, DataList, DataLength, PreparsedData, Report, ReportLength);
}

__declspec(dllexport) NTSTATUS __stdcall HidP_SetScaledUsageValue(int ReportType, USAGE UsagePage, USHORT LinkCollection, USAGE Usage, LONG UsageValue, void *PreparsedData, void *Report, ULONG ReportLength) {
	return reinterpret_cast<NTSTATUS(__stdcall *)(int, USAGE, USHORT, USAGE, LONG, void *, void *, ULONG)>(hid_HidP_SetScaledUsageValue)(ReportType, UsagePage, LinkCollection, Usage, UsageValue, PreparsedData, Report, ReportLength);
}

__declspec(dllexport) NTSTATUS __stdcall HidP_SetUsageValue(int ReportType, USAGE UsagePage, USHORT LinkCollection, USAGE Usage, ULONG UsageValue, void *PreparsedData, void *Report, ULONG ReportLength) {
	return reinterpret_cast<NTSTATUS(__stdcall *)(int, USAGE, USHORT, USAGE, ULONG, void *, void *, ULONG)>(hid_HidP_SetUsageValue)(ReportType, UsagePage, LinkCollection, Usage, UsageValue, PreparsedData, Report, ReportLength);
}

__declspec(dllexport) NTSTATUS __stdcall HidP_SetUsageValueArray(int ReportType, USAGE UsagePage, USHORT LinkCollection, USAGE Usage, void *UsageValue, USHORT UsageValueByteLength, void *PreparsedData, void *Report, ULONG ReportLength) {
	return reinterpret_cast<NTSTATUS(__stdcall *)(int, USAGE, USHORT, USAGE, void *, USHORT, void *, void *, ULONG)>(hid_HidP_SetUsageValueArray)(ReportType, UsagePage, LinkCollection, Usage, UsageValue, UsageValueByteLength, PreparsedData, Report, ReportLength);
}

__declspec(dllexport) NTSTATUS __stdcall HidP_SetUsages(int ReportType, USAGE UsagePage, USHORT LinkCollection, void *UsageList, void *UsageLength, void *PreparsedData, void *Report, ULONG ReportLength) {
	return reinterpret_cast<NTSTATUS(__stdcall *)(int, USAGE, USHORT, void *, void *, void *, void *, ULONG)>(hid_HidP_SetUsages)(ReportType, UsagePage, LinkCollection, UsageList, UsageLength, PreparsedData, Report, ReportLength);
}

__declspec(dllexport) NTSTATUS __stdcall HidP_TranslateUsagesToI8042ScanCodes(void *ChangedUsageList, ULONG UsageListLength, void *KeyAction, void *ModifierState, void *InsertCodesProcedure, void *InsertCodesContext) {
	return reinterpret_cast<NTSTATUS(__stdcall *)(void *, ULONG, void *, void *, void *, void *)>(hid_HidP_TranslateUsagesToI8042ScanCodes)(ChangedUsageList, UsageListLength, KeyAction, ModifierState, InsertCodesProcedure, InsertCodesContext);
}

__declspec(dllexport) NTSTATUS __stdcall HidP_UnsetUsages(int ReportType, USAGE UsagePage, USHORT LinkCollection, void *UsageList, void *UsageLength, void *PreparsedData, void *Report, ULONG ReportLength) {
	return reinterpret_cast<NTSTATUS(__stdcall *)(int, USAGE, USHORT, void *, void *, void *, void *, ULONG)>(hid_HidP_UnsetUsages)(ReportType, UsagePage, LinkCollection, UsageList, UsageLength, PreparsedData, Report, ReportLength);
}

__declspec(dllexport) NTSTATUS __stdcall HidP_UsageListDifference(void *PreviousUsageList, void *CurrentUsageList, void *BreakUsageList, void *MakeUsageList, ULONG UsageListLength) {
	return reinterpret_cast<NTSTATUS(__stdcall *)(void *, void *, void *, void *, ULONG)>(hid_HidP_UsageListDifference)(PreviousUsageList, CurrentUsageList, BreakUsageList, MakeUsageList, UsageListLength);
}
}

BOOL WINAPI DllMain(HINSTANCE hinstDLL, DWORD fdwReason, LPVOID lpvReserved) {
	switch (fdwReason) {
		case DLL_PROCESS_ATTACH:
			hid_proxy_start();
			break;
		case DLL_PROCESS_DETACH:
			break;
		default:
			break;
	}

	return TRUE;
}
