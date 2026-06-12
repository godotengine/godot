/*
  Simple DirectMedia Layer
  Copyright (C) 1997-2026 Sam Lantinga <slouken@libsdl.org>

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

#if defined(SDL_PLATFORM_WINDOWS)

#include "SDL_windows.h"

// #include "../../video/SDL_surface_c.h"

#include <shellapi.h> // CommandLineToArgvW()

#include <objbase.h> // for CoInitialize/CoUninitialize (Win32 only)
#ifdef HAVE_ROAPI_H
#include <roapi.h> // For RoInitialize/RoUninitialize (Win32 only)
#else
typedef enum RO_INIT_TYPE
{
    RO_INIT_SINGLETHREADED = 0,
    RO_INIT_MULTITHREADED = 1
} RO_INIT_TYPE;
#endif

#ifndef LOAD_LIBRARY_SEARCH_SYSTEM32
#define LOAD_LIBRARY_SEARCH_SYSTEM32 0x00000800
#endif

#ifndef WC_ERR_INVALID_CHARS
#define WC_ERR_INVALID_CHARS 0x00000080
#endif

// Dark mode support
typedef enum {
    UXTHEME_APPMODE_DEFAULT,
    UXTHEME_APPMODE_ALLOW_DARK,
    UXTHEME_APPMODE_FORCE_DARK,
    UXTHEME_APPMODE_FORCE_LIGHT,
    UXTHEME_APPMODE_MAX
} UxthemePreferredAppMode;

typedef enum {
    WCA_UNDEFINED = 0,
    WCA_USEDARKMODECOLORS = 26,
    WCA_LAST = 27
} WINDOWCOMPOSITIONATTRIB;

typedef struct {
    WINDOWCOMPOSITIONATTRIB Attrib;
    PVOID pvData;
    SIZE_T cbData;
} WINDOWCOMPOSITIONATTRIBDATA;

typedef struct {
    ULONG dwOSVersionInfoSize;
    ULONG dwMajorVersion;
    ULONG dwMinorVersion;
    ULONG dwBuildNumber;
    ULONG dwPlatformId;
    WCHAR szCSDVersion[128];
} NT_OSVERSIONINFOW;

typedef bool (WINAPI *ShouldAppsUseDarkMode_t)(void);
typedef void (WINAPI *AllowDarkModeForWindow_t)(HWND, bool);
typedef void (WINAPI *AllowDarkModeForApp_t)(bool);
typedef void (WINAPI *RefreshImmersiveColorPolicyState_t)(void);
typedef UxthemePreferredAppMode (WINAPI *SetPreferredAppMode_t)(UxthemePreferredAppMode);
typedef BOOL (WINAPI *SetWindowCompositionAttribute_t)(HWND, const WINDOWCOMPOSITIONATTRIBDATA *);
typedef void (NTAPI *RtlGetVersion_t)(NT_OSVERSIONINFOW *);

// Fake window to help with DirectInput events.
HWND SDL_HelperWindow = NULL;
static const TCHAR *SDL_HelperWindowClassName = TEXT("SDLHelperWindowInputCatcher");
static const TCHAR *SDL_HelperWindowName = TEXT("SDLHelperWindowInputMsgWindow");
static ATOM SDL_HelperWindowClass = 0;

/*
 * Creates a HelperWindow used for DirectInput.
 */
bool SDL_HelperWindowCreate(void)
{
    HINSTANCE hInstance = GetModuleHandle(NULL);
    WNDCLASS wce;

    // Make sure window isn't created twice.
    if (SDL_HelperWindow != NULL) {
        return true;
    }

    // Create the class.
    SDL_zero(wce);
    wce.lpfnWndProc = DefWindowProc;
    wce.lpszClassName = SDL_HelperWindowClassName;
    wce.hInstance = hInstance;

    // Register the class.
    SDL_HelperWindowClass = RegisterClass(&wce);
    if (SDL_HelperWindowClass == 0 && GetLastError() != ERROR_CLASS_ALREADY_EXISTS) {
        return WIN_SetError("Unable to create Helper Window Class");
    }

    // Create the window.
    SDL_HelperWindow = CreateWindowEx(0, SDL_HelperWindowClassName,
                                      SDL_HelperWindowName,
                                      WS_OVERLAPPED, CW_USEDEFAULT,
                                      CW_USEDEFAULT, CW_USEDEFAULT,
                                      CW_USEDEFAULT, HWND_MESSAGE, NULL,
                                      hInstance, NULL);
    if (!SDL_HelperWindow) {
        UnregisterClass(SDL_HelperWindowClassName, hInstance);
        return WIN_SetError("Unable to create Helper Window");
    }

    return true;
}

/*
 * Destroys the HelperWindow previously created with SDL_HelperWindowCreate.
 */
void SDL_HelperWindowDestroy(void)
{
    HINSTANCE hInstance = GetModuleHandle(NULL);

    // Destroy the window.
    if (SDL_HelperWindow != NULL) {
        if (DestroyWindow(SDL_HelperWindow) == 0) {
            WIN_SetError("Unable to destroy Helper Window");
            return;
        }
        SDL_HelperWindow = NULL;
    }

    // Unregister the class.
    if (SDL_HelperWindowClass != 0) {
        if ((UnregisterClass(SDL_HelperWindowClassName, hInstance)) == 0) {
            WIN_SetError("Unable to destroy Helper Window Class");
            return;
        }
        SDL_HelperWindowClass = 0;
    }
}

// Sets an error message based on an HRESULT
bool WIN_SetErrorFromHRESULT(const char *prefix, HRESULT hr)
{
    TCHAR buffer[1024];
    char *message;
    TCHAR *p = buffer;
    DWORD c = FormatMessage(FORMAT_MESSAGE_FROM_SYSTEM, NULL, hr, 0,
                            buffer, SDL_arraysize(buffer), NULL);
    buffer[c] = 0;
    // kill CR/LF that FormatMessage() sticks at the end
    while (*p) {
        if (*p == '\r') {
            *p = 0;
            break;
        }
        ++p;
    }
    message = WIN_StringToUTF8(buffer);
    SDL_SetError("%s%s%s", prefix ? prefix : "", prefix ? ": " : "", message);
    SDL_free(message);
    return false;
}

// Sets an error message based on GetLastError()
bool WIN_SetError(const char *prefix)
{
    return WIN_SetErrorFromHRESULT(prefix, GetLastError());
}

HRESULT
WIN_CoInitialize(void)
{
    /* SDL handles any threading model, so initialize with the default, which
       is compatible with OLE and if that doesn't work, try multi-threaded mode.

       If you need multi-threaded mode, call CoInitializeEx() before SDL_Init()
    */
#if defined(SDL_PLATFORM_XBOXONE) || defined(SDL_PLATFORM_XBOXSERIES)
    // On Xbox, there's no need to call CoInitializeEx (and it's not implemented)
    return S_OK;
#else
    HRESULT hr = CoInitializeEx(NULL, COINIT_APARTMENTTHREADED);
    if (hr == RPC_E_CHANGED_MODE) {
        hr = CoInitializeEx(NULL, COINIT_MULTITHREADED);
    }

    // S_FALSE means success, but someone else already initialized.
    // You still need to call CoUninitialize in this case!
    if (hr == S_FALSE) {
        return S_OK;
    }

    return hr;
#endif
}

void WIN_CoUninitialize(void)
{
    CoUninitialize();
}

FARPROC WIN_LoadComBaseFunction(const char *name)
{
    static bool s_bLoaded;
    static HMODULE s_hComBase;

    if (!s_bLoaded) {
        s_hComBase = LoadLibraryEx(TEXT("combase.dll"), NULL, LOAD_LIBRARY_SEARCH_SYSTEM32);
        s_bLoaded = true;
    }
    if (s_hComBase) {
        return GetProcAddress(s_hComBase, name);
    } else {
        return NULL;
    }
}

HRESULT
WIN_RoInitialize(void)
{
    typedef HRESULT(WINAPI * RoInitialize_t)(RO_INIT_TYPE initType);
    RoInitialize_t RoInitializeFunc = (RoInitialize_t)WIN_LoadComBaseFunction("RoInitialize");
    if (RoInitializeFunc) {
        // RO_INIT_SINGLETHREADED is equivalent to COINIT_APARTMENTTHREADED
        HRESULT hr = RoInitializeFunc(RO_INIT_SINGLETHREADED);
        if (hr == RPC_E_CHANGED_MODE) {
            hr = RoInitializeFunc(RO_INIT_MULTITHREADED);
        }

        // S_FALSE means success, but someone else already initialized.
        // You still need to call RoUninitialize in this case!
        if (hr == S_FALSE) {
            return S_OK;
        }

        return hr;
    } else {
        return E_NOINTERFACE;
    }
}

void WIN_RoUninitialize(void)
{
    typedef void(WINAPI * RoUninitialize_t)(void);
    RoUninitialize_t RoUninitializeFunc = (RoUninitialize_t)WIN_LoadComBaseFunction("RoUninitialize");
    if (RoUninitializeFunc) {
        RoUninitializeFunc();
    }
}

#if !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)
static BOOL IsWindowsVersionOrGreater(WORD wMajorVersion, WORD wMinorVersion, WORD wServicePackMajor)
{
    OSVERSIONINFOEXW osvi;
    DWORDLONG const dwlConditionMask = VerSetConditionMask(
        VerSetConditionMask(
            VerSetConditionMask(
                0, VER_MAJORVERSION, VER_GREATER_EQUAL),
            VER_MINORVERSION, VER_GREATER_EQUAL),
        VER_SERVICEPACKMAJOR, VER_GREATER_EQUAL);

    SDL_zero(osvi);
    osvi.dwOSVersionInfoSize = sizeof(osvi);
    osvi.dwMajorVersion = wMajorVersion;
    osvi.dwMinorVersion = wMinorVersion;
    osvi.wServicePackMajor = wServicePackMajor;

    return VerifyVersionInfoW(&osvi, VER_MAJORVERSION | VER_MINORVERSION | VER_SERVICEPACKMAJOR, dwlConditionMask) != FALSE;
}

static DWORD WIN_BuildNumber = 0;
static BOOL IsWindowsBuildVersionAtLeast(DWORD dwBuildNumber)
{
    if (WIN_BuildNumber != 0) {
        return (WIN_BuildNumber >= dwBuildNumber);
    }

    HMODULE ntdll = LoadLibrary(TEXT("ntdll.dll"));
    if (!ntdll) {
        return false;
    }
    // There is no function to get Windows build number, so let's get it here via RtlGetVersion
    RtlGetVersion_t RtlGetVersionFunc = (RtlGetVersion_t)GetProcAddress(ntdll, "RtlGetVersion");
    NT_OSVERSIONINFOW os_info;
    os_info.dwOSVersionInfoSize = sizeof(NT_OSVERSIONINFOW);
    os_info.dwBuildNumber = 0;
    if (RtlGetVersionFunc) {
        RtlGetVersionFunc(&os_info);
    }
    FreeLibrary(ntdll);

    WIN_BuildNumber = (os_info.dwBuildNumber & ~0xF0000000);
    return (WIN_BuildNumber >= dwBuildNumber);
}
#else
static BOOL IsWindowsBuildVersionAtLeast(DWORD dwBuildNumber)
{
    return TRUE;
}
#endif

// apply some static variables so we only call into the Win32 API once per process for each check.
#if defined(SDL_PLATFORM_XBOXONE) || defined(SDL_PLATFORM_XBOXSERIES)
    #define CHECKWINVER(notdesktop_platform_result, test) return (notdesktop_platform_result);
#else
    #define CHECKWINVER(notdesktop_platform_result, test) \
        static bool checked = false; \
        static BOOL result = FALSE; \
        if (!checked) { \
            result = (test); \
            checked = true; \
        } \
        return result;
#endif

BOOL WIN_IsWine(void)
{
    static bool checked;
    static bool is_wine;

    if (!checked) {
        HMODULE ntdll = LoadLibrary(TEXT("ntdll.dll"));
        if (ntdll) {
            if (GetProcAddress(ntdll, "wine_get_version") != NULL) {
                is_wine = true;
            }
            FreeLibrary(ntdll);
        }
        checked = true;
    }
    return is_wine;
}

// this is the oldest thing we run on (and we may lose support for this in SDL3 at any time!),
//  so there's no "OrGreater" as that would always be TRUE. The other functions are here to
//  ask "can we support a specific feature?" but this function is here to ask "do we need to do
//  something different for an OS version we probably should abandon?"  :)
BOOL WIN_IsWindowsXP(void)
{
    CHECKWINVER(FALSE, !WIN_IsWindowsVistaOrGreater() && IsWindowsVersionOrGreater(HIBYTE(_WIN32_WINNT_WINXP), LOBYTE(_WIN32_WINNT_WINXP), 0));
}

BOOL WIN_IsWindowsVistaOrGreater(void)
{
    CHECKWINVER(TRUE, IsWindowsVersionOrGreater(HIBYTE(_WIN32_WINNT_VISTA), LOBYTE(_WIN32_WINNT_VISTA), 0));
}

BOOL WIN_IsWindows7OrGreater(void)
{
    CHECKWINVER(TRUE, IsWindowsVersionOrGreater(HIBYTE(_WIN32_WINNT_WIN7), LOBYTE(_WIN32_WINNT_WIN7), 0));
}

BOOL WIN_IsWindows8OrGreater(void)
{
    CHECKWINVER(TRUE, IsWindowsVersionOrGreater(HIBYTE(_WIN32_WINNT_WIN8), LOBYTE(_WIN32_WINNT_WIN8), 0));
}

BOOL WIN_IsWindows11OrGreater(void)
{
    return IsWindowsBuildVersionAtLeast(22000);
}

#undef CHECKWINVER


/*
WAVExxxCAPS gives you 31 bytes for the device name, and just truncates if it's
longer. However, since WinXP, you can use the WAVExxxCAPS2 structure, which
will give you a name GUID. The full name is in the Windows Registry under
that GUID, located here: HKLM\System\CurrentControlSet\Control\MediaCategories

Note that drivers can report GUID_NULL for the name GUID, in which case,
Windows makes a best effort to fill in those 31 bytes in the usual place.
This info summarized from MSDN:

http://web.archive.org/web/20131027093034/http://msdn.microsoft.com/en-us/library/windows/hardware/ff536382(v=vs.85).aspx

Always look this up in the registry if possible, because the strings are
different! At least on Win10, I see "Yeti Stereo Microphone" in the
Registry, and a unhelpful "Microphone(Yeti Stereo Microph" in winmm. Sigh.

(Also, DirectSound shouldn't be limited to 32 chars, but its device enum
has the same problem.)

WASAPI doesn't need this. This is just for DirectSound/WinMM.
*/
char *WIN_LookupAudioDeviceName(const WCHAR *name, const GUID *guid)
{
#if defined(SDL_PLATFORM_XBOXONE) || defined(SDL_PLATFORM_XBOXSERIES)
    return WIN_StringToUTF8W(name); // No registry access on Xbox, go with what we've got.
#else
    static const GUID nullguid = { 0 };
    const unsigned char *ptr;
    char keystr[128];
    WCHAR *strw = NULL;
    bool rc;
    HKEY hkey;
    DWORD len = 0;
    char *result = NULL;

    if (WIN_IsEqualGUID(guid, &nullguid)) {
        return WIN_StringToUTF8W(name); // No GUID, go with what we've got.
    }

    ptr = (const unsigned char *)guid;
    (void)SDL_snprintf(keystr, sizeof(keystr),
                       "System\\CurrentControlSet\\Control\\MediaCategories\\{%02X%02X%02X%02X-%02X%02X-%02X%02X-%02X%02X-%02X%02X%02X%02X%02X%02X}",
                       ptr[3], ptr[2], ptr[1], ptr[0], ptr[5], ptr[4], ptr[7], ptr[6],
                       ptr[8], ptr[9], ptr[10], ptr[11], ptr[12], ptr[13], ptr[14], ptr[15]);

    strw = WIN_UTF8ToStringW(keystr);
    rc = (RegOpenKeyExW(HKEY_LOCAL_MACHINE, strw, 0, KEY_QUERY_VALUE, &hkey) == ERROR_SUCCESS);
    SDL_free(strw);
    if (!rc) {
        return WIN_StringToUTF8W(name); // oh well.
    }

    rc = (RegQueryValueExW(hkey, L"Name", NULL, NULL, NULL, &len) == ERROR_SUCCESS);
    if (!rc) {
        RegCloseKey(hkey);
        return WIN_StringToUTF8W(name); // oh well.
    }

    strw = (WCHAR *)SDL_malloc(len + sizeof(WCHAR));
    if (!strw) {
        RegCloseKey(hkey);
        return WIN_StringToUTF8W(name); // oh well.
    }

    rc = (RegQueryValueExW(hkey, L"Name", NULL, NULL, (LPBYTE)strw, &len) == ERROR_SUCCESS);
    RegCloseKey(hkey);
    if (!rc) {
        SDL_free(strw);
        return WIN_StringToUTF8W(name); // oh well.
    }

    strw[len / 2] = 0; // make sure it's null-terminated.

    result = WIN_StringToUTF8W(strw);
    SDL_free(strw);
    return result ? result : WIN_StringToUTF8W(name);
#endif
}

BOOL WIN_IsEqualGUID(const GUID *a, const GUID *b)
{
    return (SDL_memcmp(a, b, sizeof(*a)) == 0);
}

BOOL WIN_IsEqualIID(REFIID a, REFIID b)
{
    return (SDL_memcmp(a, b, sizeof(*a)) == 0);
}

void WIN_RECTToRect(const RECT *winrect, SDL_Rect *sdlrect)
{
    sdlrect->x = winrect->left;
    sdlrect->w = (winrect->right - winrect->left) + 1;
    sdlrect->y = winrect->top;
    sdlrect->h = (winrect->bottom - winrect->top) + 1;
}

void WIN_RectToRECT(const SDL_Rect *sdlrect, RECT *winrect)
{
    winrect->left = sdlrect->x;
    winrect->right = sdlrect->x + sdlrect->w - 1;
    winrect->top = sdlrect->y;
    winrect->bottom = sdlrect->y + sdlrect->h - 1;
}

bool WIN_WindowRectValid(const RECT *rect)
{
    // A window can be resized to zero height, but not zero width
    return (rect->right > 0);
}


// Some GUIDs we need to know without linking to libraries that aren't available before Vista.
/* *INDENT-OFF* */ // clang-format off
static const GUID SDL_KSDATAFORMAT_SUBTYPE_PCM = { 0x00000001, 0x0000, 0x0010,{ 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71 } };
static const GUID SDL_KSDATAFORMAT_SUBTYPE_IEEE_FLOAT = { 0x00000003, 0x0000, 0x0010,{ 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71 } };
/* *INDENT-ON* */ // clang-format on

SDL_AudioFormat SDL_WaveFormatExToSDLFormat(WAVEFORMATEX *waveformat)
{
    if ((waveformat->wFormatTag == WAVE_FORMAT_IEEE_FLOAT) && (waveformat->wBitsPerSample == 32)) {
        return SDL_AUDIO_F32;
    } else if ((waveformat->wFormatTag == WAVE_FORMAT_PCM) && (waveformat->wBitsPerSample == 16)) {
        return SDL_AUDIO_S16;
    } else if ((waveformat->wFormatTag == WAVE_FORMAT_PCM) && (waveformat->wBitsPerSample == 32)) {
        return SDL_AUDIO_S32;
    } else if (waveformat->wFormatTag == WAVE_FORMAT_EXTENSIBLE) {
        const WAVEFORMATEXTENSIBLE *ext = (const WAVEFORMATEXTENSIBLE *)waveformat;
        if ((SDL_memcmp(&ext->SubFormat, &SDL_KSDATAFORMAT_SUBTYPE_IEEE_FLOAT, sizeof(GUID)) == 0) && (waveformat->wBitsPerSample == 32)) {
            return SDL_AUDIO_F32;
        } else if ((SDL_memcmp(&ext->SubFormat, &SDL_KSDATAFORMAT_SUBTYPE_PCM, sizeof(GUID)) == 0) && (waveformat->wBitsPerSample == 16)) {
            return SDL_AUDIO_S16;
        } else if ((SDL_memcmp(&ext->SubFormat, &SDL_KSDATAFORMAT_SUBTYPE_PCM, sizeof(GUID)) == 0) && (waveformat->wBitsPerSample == 32)) {
            return SDL_AUDIO_S32;
        }
    }
    return SDL_AUDIO_UNKNOWN;
}


int WIN_WideCharToMultiByte(UINT CodePage, DWORD dwFlags, LPCWCH lpWideCharStr, int cchWideChar, LPSTR lpMultiByteStr, int cbMultiByte, LPCCH lpDefaultChar, LPBOOL lpUsedDefaultChar)
{
    if (WIN_IsWindowsXP()) {
        dwFlags &= ~WC_ERR_INVALID_CHARS;  // not supported before Vista. Without this flag, it will just replace bogus chars with U+FFFD. You're on your own, WinXP.
    }
    return WideCharToMultiByte(CodePage, dwFlags, lpWideCharStr, cchWideChar, lpMultiByteStr, cbMultiByte, lpDefaultChar, lpUsedDefaultChar);
}

const char *WIN_CheckDefaultArgcArgv(int *pargc, char ***pargv, void **pallocated)
{
    // If the provided argv is valid, we pass it to the main function as-is, since it's probably what the user wants.
    // Otherwise, we take a NULL argv as an instruction for SDL to parse the command line into an argv.
    // On Windows, when SDL provides the main entry point, argv is always NULL.

    const char *out_of_mem_str = "Out of memory - aborting";
    const char *proc_err_str = "Error processing command line arguments - aborting";

    *pallocated = NULL;

    if (*pargv) {
        return NULL;  // just go with what was provided, no error message.
    }

    // We need to be careful about how we allocate/free memory here. We can't use SDL_alloc()/SDL_free()
    // because the application might have used SDL_SetMemoryFunctions() to change the allocator.
    LPWSTR *argvw = NULL;
    char **argv = NULL;

    const LPWSTR command_line = GetCommandLineW();

    // Because of how the Windows command line is structured, we know for sure that the buffer size required to
    // store all argument strings converted to UTF-8 (with null terminators) is guaranteed to be less than or equal
    // to the size of the original command line string converted to UTF-8.
    const int argdata_size = WideCharToMultiByte(CP_UTF8, 0, command_line, -1, NULL, 0, NULL, NULL); // Includes the null terminator
    if (!argdata_size) {
        return proc_err_str;
    }

    int argc = -1;
    argvw = CommandLineToArgvW(command_line, &argc);
    if (!argvw || argc < 0) {
        return out_of_mem_str;
    }

    // Allocate argv followed by the argument string buffer as one contiguous allocation.
    argv = (char **)HeapAlloc(GetProcessHeap(), HEAP_ZERO_MEMORY, (argc + 1) * sizeof(*argv) + argdata_size);
    if (!argv) {
        LocalFree(argvw);
        return out_of_mem_str;
    }

    char *argdata = ((char *)argv) + (argc + 1) * sizeof(*argv);
    int argdata_index = 0;

    for (int i = 0; i < argc; ++i) {
        const int bytes_written = WideCharToMultiByte(CP_UTF8, 0, argvw[i], -1, argdata + argdata_index, argdata_size - argdata_index, NULL, NULL);
        if (!bytes_written) {
            HeapFree(GetProcessHeap(), 0, argv);
            LocalFree(argvw);
            return proc_err_str;
        }
        argv[i] = argdata + argdata_index;
        argdata_index += bytes_written;
    }

    argv[argc] = NULL;

    LocalFree(argvw);

    *pargc = argc;
    *pallocated = argv;
    *pargv = argv;

    return NULL;  // no error string.
}

#endif // defined(SDL_PLATFORM_WINDOWS)
