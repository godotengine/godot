// Windows/NtCheck.h

#ifndef ZIP7_INC_WINDOWS_NT_CHECK_H
#define ZIP7_INC_WINDOWS_NT_CHECK_H

#ifdef _WIN32

#include "../Common/MyWindows.h"

#if !defined(_WIN64) && !defined(UNDER_CE)

#if defined(_MSC_VER) && _MSC_VER >= 1900
#pragma warning(push)
// GetVersionExW was declared deprecated
#pragma warning(disable : 4996)
#endif
static inline bool IsItWindowsNT()
{
  OSVERSIONINFO vi;
  vi.dwOSVersionInfoSize = sizeof(vi);
  return (::GetVersionEx(&vi) && vi.dwPlatformId == VER_PLATFORM_WIN32_NT);
}
#if defined(_MSC_VER) && _MSC_VER >= 1900
#pragma warning(pop)
#endif

#endif

#ifndef _UNICODE
    extern
    bool g_IsNT;
  #if defined(_WIN64) || defined(UNDER_CE)
    bool g_IsNT = true;
    #define SET_IS_NT
  #else
    bool g_IsNT = false;
    #define SET_IS_NT g_IsNT = IsItWindowsNT();
  #endif
  #define NT_CHECK_ACTION
  // #define NT_CHECK_ACTION { NT_CHECK_FAIL_ACTION }
#else
  #if !defined(_WIN64) && !defined(UNDER_CE)
    #define NT_CHECK_ACTION if (!IsItWindowsNT()) { NT_CHECK_FAIL_ACTION }
  #else
    #define NT_CHECK_ACTION
  #endif
  #define SET_IS_NT
#endif

#define NT_CHECK  NT_CHECK_ACTION SET_IS_NT

#else

#define NT_CHECK

#endif

#endif
