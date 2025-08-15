/*
  Simple DirectMedia Layer
  Copyright (C) 1997-2025 Sam Lantinga <slouken@libsdl.org>

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

#ifndef SDL_hid_h_
#define SDL_hid_h_

#include "SDL_windows.h"

typedef LONG NTSTATUS;
typedef USHORT USAGE;
typedef struct _HIDP_PREPARSED_DATA *PHIDP_PREPARSED_DATA;

typedef struct _HIDD_ATTRIBUTES
{
    ULONG Size;
    USHORT VendorID;
    USHORT ProductID;
    USHORT VersionNumber;
} HIDD_ATTRIBUTES, *PHIDD_ATTRIBUTES;

typedef enum
{
    HidP_Input = 0,
    HidP_Output = 1,
    HidP_Feature = 2
} HIDP_REPORT_TYPE;

typedef struct
{
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
    ULONG Reserved[10];
    union
    {
        struct
        {
            USAGE UsageMin;
            USAGE UsageMax;
            USHORT StringMin;
            USHORT StringMax;
            USHORT DesignatorMin;
            USHORT DesignatorMax;
            USHORT DataIndexMin;
            USHORT DataIndexMax;
        } Range;
        struct
        {
            USAGE Usage;
            USAGE Reserved1;
            USHORT StringIndex;
            USHORT Reserved2;
            USHORT DesignatorIndex;
            USHORT Reserved3;
            USHORT DataIndex;
            USHORT Reserved4;
        } NotRange;
    };
} HIDP_BUTTON_CAPS, *PHIDP_BUTTON_CAPS;

typedef struct
{
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
    LONG LogicalMin;
    LONG LogicalMax;
    LONG PhysicalMin;
    LONG PhysicalMax;
    union
    {
        struct
        {
            USAGE UsageMin;
            USAGE UsageMax;
            USHORT StringMin;
            USHORT StringMax;
            USHORT DesignatorMin;
            USHORT DesignatorMax;
            USHORT DataIndexMin;
            USHORT DataIndexMax;
        } Range;
        struct
        {
            USAGE Usage;
            USAGE Reserved1;
            USHORT StringIndex;
            USHORT Reserved2;
            USHORT DesignatorIndex;
            USHORT Reserved3;
            USHORT DataIndex;
            USHORT Reserved4;
        } NotRange;
    };
} HIDP_VALUE_CAPS, *PHIDP_VALUE_CAPS;

typedef struct
{
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

typedef struct
{
    USHORT DataIndex;
    USHORT Reserved;
    union
    {
        ULONG RawValue;
        BOOLEAN On;
    };
} HIDP_DATA, *PHIDP_DATA;

#define HIDP_ERROR_CODES(p1, p2)            ((NTSTATUS)(((p1) << 28) | (0x11 << 16) | (p2)))
#define HIDP_STATUS_SUCCESS                 HIDP_ERROR_CODES(0x0, 0x0000)
#define HIDP_STATUS_NULL                    HIDP_ERROR_CODES(0x8, 0x0001)
#define HIDP_STATUS_INVALID_PREPARSED_DATA  HIDP_ERROR_CODES(0xC, 0x0001)
#define HIDP_STATUS_INVALID_REPORT_TYPE     HIDP_ERROR_CODES(0xC, 0x0002)
#define HIDP_STATUS_INVALID_REPORT_LENGTH   HIDP_ERROR_CODES(0xC, 0x0003)
#define HIDP_STATUS_USAGE_NOT_FOUND         HIDP_ERROR_CODES(0xC, 0x0004)
#define HIDP_STATUS_VALUE_OUT_OF_RANGE      HIDP_ERROR_CODES(0xC, 0x0005)
#define HIDP_STATUS_BAD_LOG_PHY_VALUES      HIDP_ERROR_CODES(0xC, 0x0006)
#define HIDP_STATUS_BUFFER_TOO_SMALL        HIDP_ERROR_CODES(0xC, 0x0007)
#define HIDP_STATUS_INTERNAL_ERROR          HIDP_ERROR_CODES(0xC, 0x0008)
#define HIDP_STATUS_I8042_TRANS_UNKNOWN     HIDP_ERROR_CODES(0xC, 0x0009)
#define HIDP_STATUS_INCOMPATIBLE_REPORT_ID  HIDP_ERROR_CODES(0xC, 0x000A)
#define HIDP_STATUS_NOT_VALUE_ARRAY         HIDP_ERROR_CODES(0xC, 0x000B)
#define HIDP_STATUS_IS_VALUE_ARRAY          HIDP_ERROR_CODES(0xC, 0x000C)
#define HIDP_STATUS_DATA_INDEX_NOT_FOUND    HIDP_ERROR_CODES(0xC, 0x000D)
#define HIDP_STATUS_DATA_INDEX_OUT_OF_RANGE HIDP_ERROR_CODES(0xC, 0x000E)
#define HIDP_STATUS_BUTTON_NOT_PRESSED      HIDP_ERROR_CODES(0xC, 0x000F)
#define HIDP_STATUS_REPORT_DOES_NOT_EXIST   HIDP_ERROR_CODES(0xC, 0x0010)
#define HIDP_STATUS_NOT_IMPLEMENTED         HIDP_ERROR_CODES(0xC, 0x0020)

extern bool WIN_LoadHIDDLL(void);
extern void WIN_UnloadHIDDLL(void);

typedef BOOLEAN (WINAPI *HidD_GetAttributes_t)(HANDLE HidDeviceObject, PHIDD_ATTRIBUTES Attributes);
typedef BOOLEAN (WINAPI *HidD_GetString_t)(HANDLE HidDeviceObject, PVOID Buffer, ULONG BufferLength);
typedef NTSTATUS (WINAPI *HidP_GetCaps_t)(PHIDP_PREPARSED_DATA PreparsedData, PHIDP_CAPS Capabilities);
typedef NTSTATUS (WINAPI *HidP_GetButtonCaps_t)(HIDP_REPORT_TYPE ReportType, PHIDP_BUTTON_CAPS ButtonCaps, PUSHORT ButtonCapsLength, PHIDP_PREPARSED_DATA PreparsedData);
typedef NTSTATUS (WINAPI *HidP_GetValueCaps_t)(HIDP_REPORT_TYPE ReportType, PHIDP_VALUE_CAPS ValueCaps, PUSHORT ValueCapsLength, PHIDP_PREPARSED_DATA PreparsedData);
typedef ULONG (WINAPI *HidP_MaxDataListLength_t)(HIDP_REPORT_TYPE ReportType, PHIDP_PREPARSED_DATA PreparsedData);
typedef NTSTATUS (WINAPI *HidP_GetData_t)(HIDP_REPORT_TYPE ReportType, PHIDP_DATA DataList, PULONG DataLength, PHIDP_PREPARSED_DATA PreparsedData, PCHAR Report, ULONG ReportLength);

extern HidD_GetAttributes_t SDL_HidD_GetAttributes;
extern HidD_GetString_t SDL_HidD_GetManufacturerString;
extern HidD_GetString_t SDL_HidD_GetProductString;
extern HidP_GetCaps_t SDL_HidP_GetCaps;
extern HidP_GetButtonCaps_t SDL_HidP_GetButtonCaps;
extern HidP_GetValueCaps_t SDL_HidP_GetValueCaps;
extern HidP_MaxDataListLength_t SDL_HidP_MaxDataListLength;
extern HidP_GetData_t SDL_HidP_GetData;

void WIN_InitDeviceNotification(void);
Uint64 WIN_GetLastDeviceNotification(void);
void WIN_QuitDeviceNotification(void);

#endif // SDL_hid_h_
