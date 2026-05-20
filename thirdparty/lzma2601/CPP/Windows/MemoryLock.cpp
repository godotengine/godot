// Windows/MemoryLock.cpp

#include "StdAfx.h"

#include "../../C/CpuArch.h"

#include "MemoryLock.h"

namespace NWindows {
namespace NSecurity {

#ifndef UNDER_CE

#ifdef _UNICODE
#define MY_FUNC_SELECT(f) :: f
#else
#define MY_FUNC_SELECT(f) my_ ## f
extern "C" {
typedef BOOL (WINAPI * Func_OpenProcessToken)(HANDLE ProcessHandle, DWORD DesiredAccess, PHANDLE TokenHandle);
typedef BOOL (WINAPI * Func_LookupPrivilegeValue)(LPCTSTR lpSystemName, LPCTSTR lpName, PLUID lpLuid);
typedef BOOL (WINAPI * Func_AdjustTokenPrivileges)(HANDLE TokenHandle, BOOL DisableAllPrivileges,
    PTOKEN_PRIVILEGES NewState, DWORD BufferLength, PTOKEN_PRIVILEGES PreviousState, PDWORD ReturnLength);
}

#define GET_PROC_ADDR(fff, name)  \
  const Func_ ## fff  my_ ## fff = Z7_GET_PROC_ADDRESS( \
        Func_ ## fff, hModule, name);
#endif

bool EnablePrivilege(LPCTSTR privilegeName, bool enable)
{
  bool res = false;

  #ifndef _UNICODE

  const HMODULE hModule = ::LoadLibrary(TEXT("advapi32.dll"));
  if (!hModule)
    return false;
  
Z7_DIAGNOSTIC_IGNORE_CAST_FUNCTION

  GET_PROC_ADDR(
     OpenProcessToken,
    "OpenProcessToken")
  GET_PROC_ADDR(
     LookupPrivilegeValue,
    "LookupPrivilegeValueA")
  GET_PROC_ADDR(
     AdjustTokenPrivileges,
    "AdjustTokenPrivileges")
  
  if (my_OpenProcessToken &&
      my_AdjustTokenPrivileges &&
      my_LookupPrivilegeValue)
  
  #endif

  {
    HANDLE token;
    if (MY_FUNC_SELECT(OpenProcessToken)(::GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES, &token))
    {
      TOKEN_PRIVILEGES tp;
      if (MY_FUNC_SELECT(LookupPrivilegeValue)(NULL, privilegeName, &(tp.Privileges[0].Luid)))
      {
        tp.PrivilegeCount = 1;
        tp.Privileges[0].Attributes = (enable ? SE_PRIVILEGE_ENABLED : 0);
        if (MY_FUNC_SELECT(AdjustTokenPrivileges)(token, FALSE, &tp, 0, NULL, NULL))
          res = (GetLastError() == ERROR_SUCCESS);
      }
      ::CloseHandle(token);
    }
  }
    
  #ifndef _UNICODE

  ::FreeLibrary(hModule);
  
  #endif

  return res;
}


Z7_DIAGNOSTIC_IGNORE_CAST_FUNCTION

typedef void (WINAPI * Func_RtlGetVersion) (OSVERSIONINFOEXW *);

/*
  We suppose that Window 10 works incorrectly with "Large Pages" at:
    - Windows 10 1703 (15063) : incorrect allocating after VirtualFree()
    - Windows 10 1709 (16299) : incorrect allocating after VirtualFree()
    - Windows 10 1809 (17763) : the failures for blocks of 1 GiB and larger,
                                if CPU doesn't support 1 GB pages.
  Windows 10 1903 (18362) probably works correctly.
*/

unsigned Get_LargePages_RiskLevel()
{
  OSVERSIONINFOEXW vi;
  const HMODULE ntdll = ::GetModuleHandleW(L"ntdll.dll");
  if (!ntdll)
    return 0;
  const
  Func_RtlGetVersion func = Z7_GET_PROC_ADDRESS(
  Func_RtlGetVersion, ntdll,
      "RtlGetVersion");
  if (!func)
    return 0;
  func(&vi);
  if (vi.dwPlatformId != VER_PLATFORM_WIN32_NT)
    return 0;
  if (vi.dwMajorVersion + vi.dwMinorVersion != 10)
    return 0;
  if (vi.dwBuildNumber <= 16299)
    return 1;

  #ifdef MY_CPU_X86_OR_AMD64
  if (vi.dwBuildNumber < 18362 && !CPU_IsSupported_PageGB())
    return 1;
  #endif

  return 0;
}

#endif

}}
