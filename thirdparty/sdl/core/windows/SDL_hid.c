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

#include "SDL_hid.h"

HidD_GetAttributes_t SDL_HidD_GetAttributes;
HidD_GetString_t SDL_HidD_GetManufacturerString;
HidD_GetString_t SDL_HidD_GetProductString;
HidP_GetCaps_t SDL_HidP_GetCaps;
HidP_GetButtonCaps_t SDL_HidP_GetButtonCaps;
HidP_GetValueCaps_t SDL_HidP_GetValueCaps;
HidP_MaxDataListLength_t SDL_HidP_MaxDataListLength;
HidP_GetData_t SDL_HidP_GetData;

static HMODULE s_pHIDDLL = 0;
static int s_HIDDLLRefCount = 0;


bool WIN_LoadHIDDLL(void)
{
    if (s_pHIDDLL) {
        SDL_assert(s_HIDDLLRefCount > 0);
        s_HIDDLLRefCount++;
        return true; // already loaded
    }

    s_pHIDDLL = LoadLibrary(TEXT("hid.dll"));
    if (!s_pHIDDLL) {
        return false;
    }

    SDL_assert(s_HIDDLLRefCount == 0);
    s_HIDDLLRefCount = 1;

    SDL_HidD_GetAttributes = (HidD_GetAttributes_t)GetProcAddress(s_pHIDDLL, "HidD_GetAttributes");
    SDL_HidD_GetManufacturerString = (HidD_GetString_t)GetProcAddress(s_pHIDDLL, "HidD_GetManufacturerString");
    SDL_HidD_GetProductString = (HidD_GetString_t)GetProcAddress(s_pHIDDLL, "HidD_GetProductString");
    SDL_HidP_GetCaps = (HidP_GetCaps_t)GetProcAddress(s_pHIDDLL, "HidP_GetCaps");
    SDL_HidP_GetButtonCaps = (HidP_GetButtonCaps_t)GetProcAddress(s_pHIDDLL, "HidP_GetButtonCaps");
    SDL_HidP_GetValueCaps = (HidP_GetValueCaps_t)GetProcAddress(s_pHIDDLL, "HidP_GetValueCaps");
    SDL_HidP_MaxDataListLength = (HidP_MaxDataListLength_t)GetProcAddress(s_pHIDDLL, "HidP_MaxDataListLength");
    SDL_HidP_GetData = (HidP_GetData_t)GetProcAddress(s_pHIDDLL, "HidP_GetData");
    if (!SDL_HidD_GetManufacturerString || !SDL_HidD_GetProductString ||
        !SDL_HidP_GetCaps || !SDL_HidP_GetButtonCaps ||
        !SDL_HidP_GetValueCaps || !SDL_HidP_MaxDataListLength || !SDL_HidP_GetData) {
        WIN_UnloadHIDDLL();
        return false;
    }

    return true;
}

void WIN_UnloadHIDDLL(void)
{
    if (s_pHIDDLL) {
        SDL_assert(s_HIDDLLRefCount > 0);
        if (--s_HIDDLLRefCount == 0) {
            FreeLibrary(s_pHIDDLL);
            s_pHIDDLL = NULL;
        }
    } else {
        SDL_assert(s_HIDDLLRefCount == 0);
    }
}

#if !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)

// CM_Register_Notification definitions

#define CR_SUCCESS 0

DECLARE_HANDLE(HCMNOTIFICATION);
typedef HCMNOTIFICATION *PHCMNOTIFICATION;

typedef enum _CM_NOTIFY_FILTER_TYPE
{
    CM_NOTIFY_FILTER_TYPE_DEVICEINTERFACE = 0,
    CM_NOTIFY_FILTER_TYPE_DEVICEHANDLE,
    CM_NOTIFY_FILTER_TYPE_DEVICEINSTANCE,
    CM_NOTIFY_FILTER_TYPE_MAX
} CM_NOTIFY_FILTER_TYPE, *PCM_NOTIFY_FILTER_TYPE;

typedef struct _CM_NOTIFY_FILTER
{
    DWORD cbSize;
    DWORD Flags;
    CM_NOTIFY_FILTER_TYPE FilterType;
    DWORD Reserved;
    union
    {
        struct
        {
            GUID ClassGuid;
        } DeviceInterface;
        struct
        {
            HANDLE hTarget;
        } DeviceHandle;
        struct
        {
            WCHAR InstanceId[200];
        } DeviceInstance;
    } u;
} CM_NOTIFY_FILTER, *PCM_NOTIFY_FILTER;

typedef enum _CM_NOTIFY_ACTION
{
    CM_NOTIFY_ACTION_DEVICEINTERFACEARRIVAL = 0,
    CM_NOTIFY_ACTION_DEVICEINTERFACEREMOVAL,
    CM_NOTIFY_ACTION_DEVICEQUERYREMOVE,
    CM_NOTIFY_ACTION_DEVICEQUERYREMOVEFAILED,
    CM_NOTIFY_ACTION_DEVICEREMOVEPENDING,
    CM_NOTIFY_ACTION_DEVICEREMOVECOMPLETE,
    CM_NOTIFY_ACTION_DEVICECUSTOMEVENT,
    CM_NOTIFY_ACTION_DEVICEINSTANCEENUMERATED,
    CM_NOTIFY_ACTION_DEVICEINSTANCESTARTED,
    CM_NOTIFY_ACTION_DEVICEINSTANCEREMOVED,
    CM_NOTIFY_ACTION_MAX
} CM_NOTIFY_ACTION, *PCM_NOTIFY_ACTION;

typedef struct _CM_NOTIFY_EVENT_DATA
{
    CM_NOTIFY_FILTER_TYPE FilterType;
    DWORD Reserved;
    union
    {
        struct
        {
            GUID ClassGuid;
            WCHAR SymbolicLink[ANYSIZE_ARRAY];
        } DeviceInterface;
        struct
        {
            GUID EventGuid;
            LONG NameOffset;
            DWORD DataSize;
            BYTE Data[ANYSIZE_ARRAY];
        } DeviceHandle;
        struct
        {
            WCHAR InstanceId[ANYSIZE_ARRAY];
        } DeviceInstance;
    } u;
} CM_NOTIFY_EVENT_DATA, *PCM_NOTIFY_EVENT_DATA;

typedef DWORD (CALLBACK *PCM_NOTIFY_CALLBACK)(HCMNOTIFICATION hNotify, PVOID Context, CM_NOTIFY_ACTION Action, PCM_NOTIFY_EVENT_DATA EventData, DWORD EventDataSize);

typedef DWORD (WINAPI *CM_Register_NotificationFunc)(PCM_NOTIFY_FILTER pFilter, PVOID pContext, PCM_NOTIFY_CALLBACK pCallback, PHCMNOTIFICATION pNotifyContext);
typedef DWORD (WINAPI *CM_Unregister_NotificationFunc)(HCMNOTIFICATION NotifyContext);

static GUID GUID_DEVINTERFACE_HID = { 0x4D1E55B2L, 0xF16F, 0x11CF, { 0x88, 0xCB, 0x00, 0x11, 0x11, 0x00, 0x00, 0x30 } };

static int s_DeviceNotificationsRequested;
static HMODULE cfgmgr32_lib_handle;
static CM_Register_NotificationFunc CM_Register_Notification;
static CM_Unregister_NotificationFunc CM_Unregister_Notification;
static HCMNOTIFICATION s_DeviceNotificationFuncHandle;
static Uint64 s_LastDeviceNotification = 1;

static DWORD CALLBACK SDL_DeviceNotificationFunc(HCMNOTIFICATION hNotify, PVOID context, CM_NOTIFY_ACTION action, PCM_NOTIFY_EVENT_DATA eventData, DWORD event_data_size)
{
    if (action == CM_NOTIFY_ACTION_DEVICEINTERFACEARRIVAL ||
        action == CM_NOTIFY_ACTION_DEVICEINTERFACEREMOVAL) {
        s_LastDeviceNotification = SDL_GetTicksNS();
    }
    return ERROR_SUCCESS;
}

void WIN_InitDeviceNotification(void)
{
    ++s_DeviceNotificationsRequested;
    if (s_DeviceNotificationsRequested > 1) {
        return;
    }

    cfgmgr32_lib_handle = LoadLibraryA("cfgmgr32.dll");
    if (cfgmgr32_lib_handle) {
        CM_Register_Notification = (CM_Register_NotificationFunc)GetProcAddress(cfgmgr32_lib_handle, "CM_Register_Notification");
        CM_Unregister_Notification = (CM_Unregister_NotificationFunc)GetProcAddress(cfgmgr32_lib_handle, "CM_Unregister_Notification");
        if (CM_Register_Notification && CM_Unregister_Notification) {
            CM_NOTIFY_FILTER notify_filter;

            SDL_zero(notify_filter);
            notify_filter.cbSize = sizeof(notify_filter);
            notify_filter.FilterType = CM_NOTIFY_FILTER_TYPE_DEVICEINTERFACE;
            notify_filter.u.DeviceInterface.ClassGuid = GUID_DEVINTERFACE_HID;
            if (CM_Register_Notification(&notify_filter, NULL, SDL_DeviceNotificationFunc, &s_DeviceNotificationFuncHandle) == CR_SUCCESS) {
                return;
            }
        }
    }

    // FIXME: Should we log errors?
}

Uint64 WIN_GetLastDeviceNotification(void)
{
    return s_LastDeviceNotification;
}

void WIN_QuitDeviceNotification(void)
{
    if (--s_DeviceNotificationsRequested > 0) {
        return;
    }
    // Make sure we have balanced calls to init/quit
    SDL_assert(s_DeviceNotificationsRequested == 0);

    if (cfgmgr32_lib_handle) {
        if (s_DeviceNotificationFuncHandle && CM_Unregister_Notification) {
            CM_Unregister_Notification(s_DeviceNotificationFuncHandle);
            s_DeviceNotificationFuncHandle = NULL;
        }

        FreeLibrary(cfgmgr32_lib_handle);
        cfgmgr32_lib_handle = NULL;
    }
}

#else

void WIN_InitDeviceNotification(void)
{
}

Uint64 WIN_GetLastDeviceNotification( void )
{
    return 0;
}

void WIN_QuitDeviceNotification(void)
{
}

#endif // !SDL_PLATFORM_XBOXONE && !SDL_PLATFORM_XBOXSERIES
