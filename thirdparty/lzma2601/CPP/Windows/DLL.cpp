// Windows/DLL.cpp

#include "StdAfx.h"

#include "DLL.h"

#ifdef _WIN32

#ifndef _UNICODE
extern bool g_IsNT;
#endif

extern HINSTANCE g_hInstance;

namespace NWindows {
namespace NDLL {

bool CLibrary::Free() throw()
{
  if (_module == NULL)
    return true;
  if (!::FreeLibrary(_module))
    return false;
  _module = NULL;
  return true;
}

bool CLibrary::LoadEx(CFSTR path, DWORD flags) throw()
{
  if (!Free())
    return false;
  #ifndef _UNICODE
  if (!g_IsNT)
  {
    _module = ::LoadLibraryEx(fs2fas(path), NULL, flags);
  }
  else
  #endif
  {
    _module = ::LoadLibraryExW(fs2us(path), NULL, flags);
  }
  return (_module != NULL);
}

bool CLibrary::Load(CFSTR path) throw()
{
  if (!Free())
    return false;
  #ifndef _UNICODE
  if (!g_IsNT)
  {
    _module = ::LoadLibrary(fs2fas(path));
  }
  else
  #endif
  {
    _module = ::LoadLibraryW(fs2us(path));
  }
  return (_module != NULL);
}

bool MyGetModuleFileName(FString &path)
{
  const HMODULE hModule = g_hInstance;
  path.Empty();
  #ifndef _UNICODE
  if (!g_IsNT)
  {
    TCHAR s[MAX_PATH + 2];
    s[0] = 0;
    const DWORD size = ::GetModuleFileName(hModule, s, MAX_PATH + 1);
    if (size <= MAX_PATH && size != 0)
    {
      path = fas2fs(s);
      return true;
    }
  }
  else
  #endif
  {
    WCHAR s[MAX_PATH + 2];
    s[0] = 0;
    const DWORD size = ::GetModuleFileNameW(hModule, s, MAX_PATH + 1);
    if (size <= MAX_PATH && size != 0)
    {
      path = us2fs(s);
      return true;
    }
  }
  return false;
}

#ifndef Z7_SFX

FString GetModuleDirPrefix()
{
  FString s;
  if (MyGetModuleFileName(s))
  {
    const int pos = s.ReverseFind_PathSepar();
    if (pos >= 0)
      s.DeleteFrom((unsigned)(pos + 1));
  }
  if (s.IsEmpty())
    s = "." STRING_PATH_SEPARATOR;
  return s;
}

#endif

}}

#else // _WIN32

#include <dlfcn.h>
#include <stdlib.h>

// FARPROC
void *GetProcAddress(HMODULE module, LPCSTR procName)
{
  void *ptr = NULL;
  if (module)
    ptr = dlsym(module, procName);
  return ptr;
}

namespace NWindows {
namespace NDLL {

bool CLibrary::Free() throw()
{
  if (!_module)
    return true;
  const int ret = dlclose(_module);
  if (ret != 0)
    return false;
  _module = NULL;
  return true;
}

bool CLibrary::Load(CFSTR path) throw()
{
  if (!Free())
    return false;

  int options = 0;

  #ifdef RTLD_LOCAL
    options |= RTLD_LOCAL;
  #endif

  #ifdef RTLD_NOW
    options |= RTLD_NOW;
  #endif

  #ifdef RTLD_GROUP
    #if ! (defined(hpux) || defined(__hpux))
      options |= RTLD_GROUP; // mainly for solaris but not for HPUX
    #endif
  #endif
  
  _module = dlopen(path, options);
  return (_module != NULL);
}

/*
// FARPROC
void * CLibrary::GetProc(LPCSTR procName) const
{
  // return My_GetProcAddress(_module, procName);
  return local_GetProcAddress(_module, procName);
  // return NULL;
}
*/

}}

#endif
