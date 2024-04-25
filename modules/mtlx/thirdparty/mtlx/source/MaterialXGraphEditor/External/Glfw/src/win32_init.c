//========================================================================
// GLFW 3.4 Win32 - www.glfw.org
//------------------------------------------------------------------------
// Copyright (c) 2002-2006 Marcus Geelnard
// Copyright (c) 2006-2019 Camilla Löwy <elmindreda@glfw.org>
//
// This software is provided 'as-is', without any express or implied
// warranty. In no event will the authors be held liable for any damages
// arising from the use of this software.
//
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it
// freely, subject to the following restrictions:
//
// 1. The origin of this software must not be misrepresented; you must not
//    claim that you wrote the original software. If you use this software
//    in a product, an acknowledgment in the product documentation would
//    be appreciated but is not required.
//
// 2. Altered source versions must be plainly marked as such, and must not
//    be misrepresented as being the original software.
//
// 3. This notice may not be removed or altered from any source
//    distribution.
//
//========================================================================
// Please use C89 style variable declarations in this file because VS 2010
//========================================================================

#include "internal.h"

#include <stdlib.h>
#include <malloc.h>

static const GUID _glfw_GUID_DEVINTERFACE_HID =
    {0x4d1e55b2,0xf16f,0x11cf,{0x88,0xcb,0x00,0x11,0x11,0x00,0x00,0x30}};

#define GUID_DEVINTERFACE_HID _glfw_GUID_DEVINTERFACE_HID

#if defined(_GLFW_USE_HYBRID_HPG) || defined(_GLFW_USE_OPTIMUS_HPG)

// Executables (but not DLLs) exporting this symbol with this value will be
// automatically directed to the high-performance GPU on Nvidia Optimus systems
// with up-to-date drivers
//
__declspec(dllexport) DWORD NvOptimusEnablement = 1;

// Executables (but not DLLs) exporting this symbol with this value will be
// automatically directed to the high-performance GPU on AMD PowerXpress systems
// with up-to-date drivers
//
__declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1;

#endif // _GLFW_USE_HYBRID_HPG

#if defined(_GLFW_BUILD_DLL)

// GLFW DLL entry point
//
BOOL WINAPI DllMain(HINSTANCE instance, DWORD reason, LPVOID reserved)
{
    return TRUE;
}

#endif // _GLFW_BUILD_DLL

// Load necessary libraries (DLLs)
//
static GLFWbool loadLibraries(void)
{
    _glfw.win32.winmm.instance = LoadLibraryA("winmm.dll");
    if (!_glfw.win32.winmm.instance)
    {
        _glfwInputErrorWin32(GLFW_PLATFORM_ERROR,
                             "Win32: Failed to load winmm.dll");
        return GLFW_FALSE;
    }

    _glfw.win32.winmm.GetTime = (PFN_timeGetTime)
        GetProcAddress(_glfw.win32.winmm.instance, "timeGetTime");

    _glfw.win32.user32.instance = LoadLibraryA("user32.dll");
    if (!_glfw.win32.user32.instance)
    {
        _glfwInputErrorWin32(GLFW_PLATFORM_ERROR,
                             "Win32: Failed to load user32.dll");
        return GLFW_FALSE;
    }

    _glfw.win32.user32.SetProcessDPIAware_ = (PFN_SetProcessDPIAware)
        GetProcAddress(_glfw.win32.user32.instance, "SetProcessDPIAware");
    _glfw.win32.user32.ChangeWindowMessageFilterEx_ = (PFN_ChangeWindowMessageFilterEx)
        GetProcAddress(_glfw.win32.user32.instance, "ChangeWindowMessageFilterEx");
    _glfw.win32.user32.EnableNonClientDpiScaling_ = (PFN_EnableNonClientDpiScaling)
        GetProcAddress(_glfw.win32.user32.instance, "EnableNonClientDpiScaling");
    _glfw.win32.user32.SetProcessDpiAwarenessContext_ = (PFN_SetProcessDpiAwarenessContext)
        GetProcAddress(_glfw.win32.user32.instance, "SetProcessDpiAwarenessContext");
    _glfw.win32.user32.GetDpiForWindow_ = (PFN_GetDpiForWindow)
        GetProcAddress(_glfw.win32.user32.instance, "GetDpiForWindow");
    _glfw.win32.user32.AdjustWindowRectExForDpi_ = (PFN_AdjustWindowRectExForDpi)
        GetProcAddress(_glfw.win32.user32.instance, "AdjustWindowRectExForDpi");

    _glfw.win32.dinput8.instance = LoadLibraryA("dinput8.dll");
    if (_glfw.win32.dinput8.instance)
    {
        _glfw.win32.dinput8.Create = (PFN_DirectInput8Create)
            GetProcAddress(_glfw.win32.dinput8.instance, "DirectInput8Create");
    }

    {
        int i;
        const char* names[] =
        {
            "xinput1_4.dll",
            "xinput1_3.dll",
            "xinput9_1_0.dll",
            "xinput1_2.dll",
            "xinput1_1.dll",
            NULL
        };

        for (i = 0;  names[i];  i++)
        {
            _glfw.win32.xinput.instance = LoadLibraryA(names[i]);
            if (_glfw.win32.xinput.instance)
            {
                _glfw.win32.xinput.GetCapabilities = (PFN_XInputGetCapabilities)
                    GetProcAddress(_glfw.win32.xinput.instance, "XInputGetCapabilities");
                _glfw.win32.xinput.GetState = (PFN_XInputGetState)
                    GetProcAddress(_glfw.win32.xinput.instance, "XInputGetState");

                break;
            }
        }
    }

    _glfw.win32.dwmapi.instance = LoadLibraryA("dwmapi.dll");
    if (_glfw.win32.dwmapi.instance)
    {
        _glfw.win32.dwmapi.IsCompositionEnabled = (PFN_DwmIsCompositionEnabled)
            GetProcAddress(_glfw.win32.dwmapi.instance, "DwmIsCompositionEnabled");
        _glfw.win32.dwmapi.Flush = (PFN_DwmFlush)
            GetProcAddress(_glfw.win32.dwmapi.instance, "DwmFlush");
        _glfw.win32.dwmapi.EnableBlurBehindWindow = (PFN_DwmEnableBlurBehindWindow)
            GetProcAddress(_glfw.win32.dwmapi.instance, "DwmEnableBlurBehindWindow");
    }

    _glfw.win32.shcore.instance = LoadLibraryA("shcore.dll");
    if (_glfw.win32.shcore.instance)
    {
        _glfw.win32.shcore.SetProcessDpiAwareness_ = (PFN_SetProcessDpiAwareness)
            GetProcAddress(_glfw.win32.shcore.instance, "SetProcessDpiAwareness");
        _glfw.win32.shcore.GetDpiForMonitor_ = (PFN_GetDpiForMonitor)
            GetProcAddress(_glfw.win32.shcore.instance, "GetDpiForMonitor");
    }

    _glfw.win32.ntdll.instance = LoadLibraryA("ntdll.dll");
    if (_glfw.win32.ntdll.instance)
    {
        _glfw.win32.ntdll.RtlVerifyVersionInfo_ = (PFN_RtlVerifyVersionInfo)
            GetProcAddress(_glfw.win32.ntdll.instance, "RtlVerifyVersionInfo");
    }

    return GLFW_TRUE;
}

// Unload used libraries (DLLs)
//
static void freeLibraries(void)
{
    if (_glfw.win32.xinput.instance)
        FreeLibrary(_glfw.win32.xinput.instance);

    if (_glfw.win32.dinput8.instance)
        FreeLibrary(_glfw.win32.dinput8.instance);

    if (_glfw.win32.winmm.instance)
        FreeLibrary(_glfw.win32.winmm.instance);

    if (_glfw.win32.user32.instance)
        FreeLibrary(_glfw.win32.user32.instance);

    if (_glfw.win32.dwmapi.instance)
        FreeLibrary(_glfw.win32.dwmapi.instance);

    if (_glfw.win32.shcore.instance)
        FreeLibrary(_glfw.win32.shcore.instance);

    if (_glfw.win32.ntdll.instance)
        FreeLibrary(_glfw.win32.ntdll.instance);
}

// Create key code translation tables
//
static void createKeyTables(void)
{
    int scancode;

    memset(_glfw.win32.keycodes, -1, sizeof(_glfw.win32.keycodes));
    memset(_glfw.win32.scancodes, -1, sizeof(_glfw.win32.scancodes));

    _glfw.win32.keycodes[0x00B] = GLFW_KEY_0;
    _glfw.win32.keycodes[0x002] = GLFW_KEY_1;
    _glfw.win32.keycodes[0x003] = GLFW_KEY_2;
    _glfw.win32.keycodes[0x004] = GLFW_KEY_3;
    _glfw.win32.keycodes[0x005] = GLFW_KEY_4;
    _glfw.win32.keycodes[0x006] = GLFW_KEY_5;
    _glfw.win32.keycodes[0x007] = GLFW_KEY_6;
    _glfw.win32.keycodes[0x008] = GLFW_KEY_7;
    _glfw.win32.keycodes[0x009] = GLFW_KEY_8;
    _glfw.win32.keycodes[0x00A] = GLFW_KEY_9;
    _glfw.win32.keycodes[0x01E] = GLFW_KEY_A;
    _glfw.win32.keycodes[0x030] = GLFW_KEY_B;
    _glfw.win32.keycodes[0x02E] = GLFW_KEY_C;
    _glfw.win32.keycodes[0x020] = GLFW_KEY_D;
    _glfw.win32.keycodes[0x012] = GLFW_KEY_E;
    _glfw.win32.keycodes[0x021] = GLFW_KEY_F;
    _glfw.win32.keycodes[0x022] = GLFW_KEY_G;
    _glfw.win32.keycodes[0x023] = GLFW_KEY_H;
    _glfw.win32.keycodes[0x017] = GLFW_KEY_I;
    _glfw.win32.keycodes[0x024] = GLFW_KEY_J;
    _glfw.win32.keycodes[0x025] = GLFW_KEY_K;
    _glfw.win32.keycodes[0x026] = GLFW_KEY_L;
    _glfw.win32.keycodes[0x032] = GLFW_KEY_M;
    _glfw.win32.keycodes[0x031] = GLFW_KEY_N;
    _glfw.win32.keycodes[0x018] = GLFW_KEY_O;
    _glfw.win32.keycodes[0x019] = GLFW_KEY_P;
    _glfw.win32.keycodes[0x010] = GLFW_KEY_Q;
    _glfw.win32.keycodes[0x013] = GLFW_KEY_R;
    _glfw.win32.keycodes[0x01F] = GLFW_KEY_S;
    _glfw.win32.keycodes[0x014] = GLFW_KEY_T;
    _glfw.win32.keycodes[0x016] = GLFW_KEY_U;
    _glfw.win32.keycodes[0x02F] = GLFW_KEY_V;
    _glfw.win32.keycodes[0x011] = GLFW_KEY_W;
    _glfw.win32.keycodes[0x02D] = GLFW_KEY_X;
    _glfw.win32.keycodes[0x015] = GLFW_KEY_Y;
    _glfw.win32.keycodes[0x02C] = GLFW_KEY_Z;

    _glfw.win32.keycodes[0x028] = GLFW_KEY_APOSTROPHE;
    _glfw.win32.keycodes[0x02B] = GLFW_KEY_BACKSLASH;
    _glfw.win32.keycodes[0x033] = GLFW_KEY_COMMA;
    _glfw.win32.keycodes[0x00D] = GLFW_KEY_EQUAL;
    _glfw.win32.keycodes[0x029] = GLFW_KEY_GRAVE_ACCENT;
    _glfw.win32.keycodes[0x01A] = GLFW_KEY_LEFT_BRACKET;
    _glfw.win32.keycodes[0x00C] = GLFW_KEY_MINUS;
    _glfw.win32.keycodes[0x034] = GLFW_KEY_PERIOD;
    _glfw.win32.keycodes[0x01B] = GLFW_KEY_RIGHT_BRACKET;
    _glfw.win32.keycodes[0x027] = GLFW_KEY_SEMICOLON;
    _glfw.win32.keycodes[0x035] = GLFW_KEY_SLASH;
    _glfw.win32.keycodes[0x056] = GLFW_KEY_WORLD_2;

    _glfw.win32.keycodes[0x00E] = GLFW_KEY_BACKSPACE;
    _glfw.win32.keycodes[0x153] = GLFW_KEY_DELETE;
    _glfw.win32.keycodes[0x14F] = GLFW_KEY_END;
    _glfw.win32.keycodes[0x01C] = GLFW_KEY_ENTER;
    _glfw.win32.keycodes[0x001] = GLFW_KEY_ESCAPE;
    _glfw.win32.keycodes[0x147] = GLFW_KEY_HOME;
    _glfw.win32.keycodes[0x152] = GLFW_KEY_INSERT;
    _glfw.win32.keycodes[0x15D] = GLFW_KEY_MENU;
    _glfw.win32.keycodes[0x151] = GLFW_KEY_PAGE_DOWN;
    _glfw.win32.keycodes[0x149] = GLFW_KEY_PAGE_UP;
    _glfw.win32.keycodes[0x045] = GLFW_KEY_PAUSE;
    _glfw.win32.keycodes[0x146] = GLFW_KEY_PAUSE;
    _glfw.win32.keycodes[0x039] = GLFW_KEY_SPACE;
    _glfw.win32.keycodes[0x00F] = GLFW_KEY_TAB;
    _glfw.win32.keycodes[0x03A] = GLFW_KEY_CAPS_LOCK;
    _glfw.win32.keycodes[0x145] = GLFW_KEY_NUM_LOCK;
    _glfw.win32.keycodes[0x046] = GLFW_KEY_SCROLL_LOCK;
    _glfw.win32.keycodes[0x03B] = GLFW_KEY_F1;
    _glfw.win32.keycodes[0x03C] = GLFW_KEY_F2;
    _glfw.win32.keycodes[0x03D] = GLFW_KEY_F3;
    _glfw.win32.keycodes[0x03E] = GLFW_KEY_F4;
    _glfw.win32.keycodes[0x03F] = GLFW_KEY_F5;
    _glfw.win32.keycodes[0x040] = GLFW_KEY_F6;
    _glfw.win32.keycodes[0x041] = GLFW_KEY_F7;
    _glfw.win32.keycodes[0x042] = GLFW_KEY_F8;
    _glfw.win32.keycodes[0x043] = GLFW_KEY_F9;
    _glfw.win32.keycodes[0x044] = GLFW_KEY_F10;
    _glfw.win32.keycodes[0x057] = GLFW_KEY_F11;
    _glfw.win32.keycodes[0x058] = GLFW_KEY_F12;
    _glfw.win32.keycodes[0x064] = GLFW_KEY_F13;
    _glfw.win32.keycodes[0x065] = GLFW_KEY_F14;
    _glfw.win32.keycodes[0x066] = GLFW_KEY_F15;
    _glfw.win32.keycodes[0x067] = GLFW_KEY_F16;
    _glfw.win32.keycodes[0x068] = GLFW_KEY_F17;
    _glfw.win32.keycodes[0x069] = GLFW_KEY_F18;
    _glfw.win32.keycodes[0x06A] = GLFW_KEY_F19;
    _glfw.win32.keycodes[0x06B] = GLFW_KEY_F20;
    _glfw.win32.keycodes[0x06C] = GLFW_KEY_F21;
    _glfw.win32.keycodes[0x06D] = GLFW_KEY_F22;
    _glfw.win32.keycodes[0x06E] = GLFW_KEY_F23;
    _glfw.win32.keycodes[0x076] = GLFW_KEY_F24;
    _glfw.win32.keycodes[0x038] = GLFW_KEY_LEFT_ALT;
    _glfw.win32.keycodes[0x01D] = GLFW_KEY_LEFT_CONTROL;
    _glfw.win32.keycodes[0x02A] = GLFW_KEY_LEFT_SHIFT;
    _glfw.win32.keycodes[0x15B] = GLFW_KEY_LEFT_SUPER;
    _glfw.win32.keycodes[0x137] = GLFW_KEY_PRINT_SCREEN;
    _glfw.win32.keycodes[0x138] = GLFW_KEY_RIGHT_ALT;
    _glfw.win32.keycodes[0x11D] = GLFW_KEY_RIGHT_CONTROL;
    _glfw.win32.keycodes[0x036] = GLFW_KEY_RIGHT_SHIFT;
    _glfw.win32.keycodes[0x15C] = GLFW_KEY_RIGHT_SUPER;
    _glfw.win32.keycodes[0x150] = GLFW_KEY_DOWN;
    _glfw.win32.keycodes[0x14B] = GLFW_KEY_LEFT;
    _glfw.win32.keycodes[0x14D] = GLFW_KEY_RIGHT;
    _glfw.win32.keycodes[0x148] = GLFW_KEY_UP;

    _glfw.win32.keycodes[0x052] = GLFW_KEY_KP_0;
    _glfw.win32.keycodes[0x04F] = GLFW_KEY_KP_1;
    _glfw.win32.keycodes[0x050] = GLFW_KEY_KP_2;
    _glfw.win32.keycodes[0x051] = GLFW_KEY_KP_3;
    _glfw.win32.keycodes[0x04B] = GLFW_KEY_KP_4;
    _glfw.win32.keycodes[0x04C] = GLFW_KEY_KP_5;
    _glfw.win32.keycodes[0x04D] = GLFW_KEY_KP_6;
    _glfw.win32.keycodes[0x047] = GLFW_KEY_KP_7;
    _glfw.win32.keycodes[0x048] = GLFW_KEY_KP_8;
    _glfw.win32.keycodes[0x049] = GLFW_KEY_KP_9;
    _glfw.win32.keycodes[0x04E] = GLFW_KEY_KP_ADD;
    _glfw.win32.keycodes[0x053] = GLFW_KEY_KP_DECIMAL;
    _glfw.win32.keycodes[0x135] = GLFW_KEY_KP_DIVIDE;
    _glfw.win32.keycodes[0x11C] = GLFW_KEY_KP_ENTER;
    _glfw.win32.keycodes[0x059] = GLFW_KEY_KP_EQUAL;
    _glfw.win32.keycodes[0x037] = GLFW_KEY_KP_MULTIPLY;
    _glfw.win32.keycodes[0x04A] = GLFW_KEY_KP_SUBTRACT;

    for (scancode = 0;  scancode < 512;  scancode++)
    {
        if (_glfw.win32.keycodes[scancode] > 0)
            _glfw.win32.scancodes[_glfw.win32.keycodes[scancode]] = scancode;
    }
}

// Creates a dummy window for behind-the-scenes work
//
static GLFWbool createHelperWindow(void)
{
    MSG msg;

    _glfw.win32.helperWindowHandle =
        CreateWindowExW(WS_EX_OVERLAPPEDWINDOW,
                        _GLFW_WNDCLASSNAME,
                        L"GLFW message window",
                        WS_CLIPSIBLINGS | WS_CLIPCHILDREN,
                        0, 0, 1, 1,
                        NULL, NULL,
                        GetModuleHandleW(NULL),
                        NULL);

    if (!_glfw.win32.helperWindowHandle)
    {
        _glfwInputErrorWin32(GLFW_PLATFORM_ERROR,
                             "Win32: Failed to create helper window");
        return GLFW_FALSE;
    }

    // HACK: The command to the first ShowWindow call is ignored if the parent
    //       process passed along a STARTUPINFO, so clear that with a no-op call
    ShowWindow(_glfw.win32.helperWindowHandle, SW_HIDE);

    // Register for HID device notifications
    {
        DEV_BROADCAST_DEVICEINTERFACE_W dbi;
        ZeroMemory(&dbi, sizeof(dbi));
        dbi.dbcc_size = sizeof(dbi);
        dbi.dbcc_devicetype = DBT_DEVTYP_DEVICEINTERFACE;
        dbi.dbcc_classguid = GUID_DEVINTERFACE_HID;

        _glfw.win32.deviceNotificationHandle =
            RegisterDeviceNotificationW(_glfw.win32.helperWindowHandle,
                                        (DEV_BROADCAST_HDR*) &dbi,
                                        DEVICE_NOTIFY_WINDOW_HANDLE);
    }

    while (PeekMessageW(&msg, _glfw.win32.helperWindowHandle, 0, 0, PM_REMOVE))
    {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }

   return GLFW_TRUE;
}


//////////////////////////////////////////////////////////////////////////
//////                       GLFW internal API                      //////
//////////////////////////////////////////////////////////////////////////

// Returns a wide string version of the specified UTF-8 string
//
WCHAR* _glfwCreateWideStringFromUTF8Win32(const char* source)
{
    WCHAR* target;
    int count;

    count = MultiByteToWideChar(CP_UTF8, 0, source, -1, NULL, 0);
    if (!count)
    {
        _glfwInputErrorWin32(GLFW_PLATFORM_ERROR,
                             "Win32: Failed to convert string from UTF-8");
        return NULL;
    }

    target = calloc(count, sizeof(WCHAR));

    if (!MultiByteToWideChar(CP_UTF8, 0, source, -1, target, count))
    {
        _glfwInputErrorWin32(GLFW_PLATFORM_ERROR,
                             "Win32: Failed to convert string from UTF-8");
        free(target);
        return NULL;
    }

    return target;
}

// Returns a UTF-8 string version of the specified wide string
//
char* _glfwCreateUTF8FromWideStringWin32(const WCHAR* source)
{
    char* target;
    int size;

    size = WideCharToMultiByte(CP_UTF8, 0, source, -1, NULL, 0, NULL, NULL);
    if (!size)
    {
        _glfwInputErrorWin32(GLFW_PLATFORM_ERROR,
                             "Win32: Failed to convert string to UTF-8");
        return NULL;
    }

    target = calloc(size, 1);

    if (!WideCharToMultiByte(CP_UTF8, 0, source, -1, target, size, NULL, NULL))
    {
        _glfwInputErrorWin32(GLFW_PLATFORM_ERROR,
                             "Win32: Failed to convert string to UTF-8");
        free(target);
        return NULL;
    }

    return target;
}

// Reports the specified error, appending information about the last Win32 error
//
void _glfwInputErrorWin32(int error, const char* description)
{
    WCHAR buffer[_GLFW_MESSAGE_SIZE] = L"";
    char message[_GLFW_MESSAGE_SIZE] = "";

    FormatMessageW(FORMAT_MESSAGE_FROM_SYSTEM |
                       FORMAT_MESSAGE_IGNORE_INSERTS |
                       FORMAT_MESSAGE_MAX_WIDTH_MASK,
                   NULL,
                   GetLastError() & 0xffff,
                   MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                   buffer,
                   sizeof(buffer) / sizeof(WCHAR),
                   NULL);
    WideCharToMultiByte(CP_UTF8, 0, buffer, -1, message, sizeof(message), NULL, NULL);

    _glfwInputError(error, "%s: %s", description, message);
}

// Updates key names according to the current keyboard layout
//
void _glfwUpdateKeyNamesWin32(void)
{
    int key;
    BYTE state[256] = {0};

    memset(_glfw.win32.keynames, 0, sizeof(_glfw.win32.keynames));

    for (key = GLFW_KEY_SPACE;  key <= GLFW_KEY_LAST;  key++)
    {
        UINT vk;
        int scancode, length;
        WCHAR chars[16];

        scancode = _glfw.win32.scancodes[key];
        if (scancode == -1)
            continue;

        if (key >= GLFW_KEY_KP_0 && key <= GLFW_KEY_KP_ADD)
        {
            const UINT vks[] = {
                VK_NUMPAD0,  VK_NUMPAD1,  VK_NUMPAD2, VK_NUMPAD3,
                VK_NUMPAD4,  VK_NUMPAD5,  VK_NUMPAD6, VK_NUMPAD7,
                VK_NUMPAD8,  VK_NUMPAD9,  VK_DECIMAL, VK_DIVIDE,
                VK_MULTIPLY, VK_SUBTRACT, VK_ADD
            };

            vk = vks[key - GLFW_KEY_KP_0];
        }
        else
            vk = MapVirtualKey(scancode, MAPVK_VSC_TO_VK);

        length = ToUnicode(vk, scancode, state,
                           chars, sizeof(chars) / sizeof(WCHAR),
                           0);

        if (length == -1)
        {
            length = ToUnicode(vk, scancode, state,
                               chars, sizeof(chars) / sizeof(WCHAR),
                               0);
        }

        if (length < 1)
            continue;

        WideCharToMultiByte(CP_UTF8, 0, chars, 1,
                            _glfw.win32.keynames[key],
                            sizeof(_glfw.win32.keynames[key]),
                            NULL, NULL);
    }
}

// Replacement for IsWindowsVersionOrGreater as MinGW lacks versionhelpers.h
//
BOOL _glfwIsWindowsVersionOrGreaterWin32(WORD major, WORD minor, WORD sp)
{
    OSVERSIONINFOEXW osvi = { sizeof(osvi), major, minor, 0, 0, {0}, sp };
    DWORD mask = VER_MAJORVERSION | VER_MINORVERSION | VER_SERVICEPACKMAJOR;
    ULONGLONG cond = VerSetConditionMask(0, VER_MAJORVERSION, VER_GREATER_EQUAL);
    cond = VerSetConditionMask(cond, VER_MINORVERSION, VER_GREATER_EQUAL);
    cond = VerSetConditionMask(cond, VER_SERVICEPACKMAJOR, VER_GREATER_EQUAL);
    // HACK: Use RtlVerifyVersionInfo instead of VerifyVersionInfoW as the
    //       latter lies unless the user knew to embed a non-default manifest
    //       announcing support for Windows 10 via supportedOS GUID
    return RtlVerifyVersionInfo(&osvi, mask, cond) == 0;
}

// Checks whether we are on at least the specified build of Windows 10
//
BOOL _glfwIsWindows10BuildOrGreaterWin32(WORD build)
{
    OSVERSIONINFOEXW osvi = { sizeof(osvi), 10, 0, build };
    DWORD mask = VER_MAJORVERSION | VER_MINORVERSION | VER_BUILDNUMBER;
    ULONGLONG cond = VerSetConditionMask(0, VER_MAJORVERSION, VER_GREATER_EQUAL);
    cond = VerSetConditionMask(cond, VER_MINORVERSION, VER_GREATER_EQUAL);
    cond = VerSetConditionMask(cond, VER_BUILDNUMBER, VER_GREATER_EQUAL);
    // HACK: Use RtlVerifyVersionInfo instead of VerifyVersionInfoW as the
    //       latter lies unless the user knew to embed a non-default manifest
    //       announcing support for Windows 10 via supportedOS GUID
    return RtlVerifyVersionInfo(&osvi, mask, cond) == 0;
}


//////////////////////////////////////////////////////////////////////////
//////                       GLFW platform API                      //////
//////////////////////////////////////////////////////////////////////////

int _glfwPlatformInit(void)
{
    // To make SetForegroundWindow work as we want, we need to fiddle
    // with the FOREGROUNDLOCKTIMEOUT system setting (we do this as early
    // as possible in the hope of still being the foreground process)
    SystemParametersInfoW(SPI_GETFOREGROUNDLOCKTIMEOUT, 0,
                          &_glfw.win32.foregroundLockTimeout, 0);
    SystemParametersInfoW(SPI_SETFOREGROUNDLOCKTIMEOUT, 0, UIntToPtr(0),
                          SPIF_SENDCHANGE);

    if (!loadLibraries())
        return GLFW_FALSE;

    createKeyTables();
    _glfwUpdateKeyNamesWin32();

    if (_glfwIsWindows10CreatorsUpdateOrGreaterWin32())
        SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
    else if (IsWindows8Point1OrGreater())
        SetProcessDpiAwareness(PROCESS_PER_MONITOR_DPI_AWARE);
    else if (IsWindowsVistaOrGreater())
        SetProcessDPIAware();

    if (!_glfwRegisterWindowClassWin32())
        return GLFW_FALSE;

    if (!createHelperWindow())
        return GLFW_FALSE;

    _glfwInitTimerWin32();
    _glfwInitJoysticksWin32();

    _glfwPollMonitorsWin32();
    return GLFW_TRUE;
}

void _glfwPlatformTerminate(void)
{
    if (_glfw.win32.deviceNotificationHandle)
        UnregisterDeviceNotification(_glfw.win32.deviceNotificationHandle);

    if (_glfw.win32.helperWindowHandle)
        DestroyWindow(_glfw.win32.helperWindowHandle);

    _glfwUnregisterWindowClassWin32();

    // Restore previous foreground lock timeout system setting
    SystemParametersInfoW(SPI_SETFOREGROUNDLOCKTIMEOUT, 0,
                          UIntToPtr(_glfw.win32.foregroundLockTimeout),
                          SPIF_SENDCHANGE);

    free(_glfw.win32.clipboardString);
    free(_glfw.win32.rawInput);

    _glfwTerminateWGL();
    _glfwTerminateEGL();

    _glfwTerminateJoysticksWin32();

    freeLibraries();
}

const char* _glfwPlatformGetVersionString(void)
{
    return _GLFW_VERSION_NUMBER " Win32 WGL EGL OSMesa"
#if defined(__MINGW32__)
        " MinGW"
#elif defined(_MSC_VER)
        " VisualC"
#endif
#if defined(_GLFW_USE_HYBRID_HPG) || defined(_GLFW_USE_OPTIMUS_HPG)
        " hybrid-GPU"
#endif
#if defined(_GLFW_BUILD_DLL)
        " DLL"
#endif
        ;
}

