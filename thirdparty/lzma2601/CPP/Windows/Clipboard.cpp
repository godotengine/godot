// Windows/Clipboard.cpp

#include "StdAfx.h"

#ifdef UNDER_CE
#include <winuserm.h>
#endif

#include "../Common/StringConvert.h"

#include "Clipboard.h"
#include "Defs.h"
#include "MemoryGlobal.h"
#include "Shell.h"

namespace NWindows {

bool CClipboard::Open(HWND wndNewOwner) throw()
{
  m_Open = BOOLToBool(::OpenClipboard(wndNewOwner));
  return m_Open;
}

bool CClipboard::Close() throw()
{
  if (!m_Open)
    return true;
  m_Open = !BOOLToBool(CloseClipboard());
  return !m_Open;
}

bool ClipboardIsFormatAvailableHDROP()
{
  return BOOLToBool(IsClipboardFormatAvailable(CF_HDROP));
}

/*
bool ClipboardGetTextString(AString &s)
{
  s.Empty();
  if (!IsClipboardFormatAvailable(CF_TEXT))
    return false;
  CClipboard clipboard;

  if (!clipboard.Open(NULL))
    return false;

  HGLOBAL h = ::GetClipboardData(CF_TEXT);
  if (h != NULL)
  {
    NMemory::CGlobalLock globalLock(h);
    const char *p = (const char *)globalLock.GetPointer();
    if (p != NULL)
    {
      s = p;
      return true;
    }
  }
  return false;
}
*/

/*
bool ClipboardGetFileNames(UStringVector &names)
{
  names.Clear();
  if (!IsClipboardFormatAvailable(CF_HDROP))
    return false;
  CClipboard clipboard;

  if (!clipboard.Open(NULL))
    return false;

  HGLOBAL h = ::GetClipboardData(CF_HDROP);
  if (h != NULL)
  {
    NMemory::CGlobalLock globalLock(h);
    void *p = (void *)globalLock.GetPointer();
    if (p != NULL)
    {
      NShell::CDrop drop(false);
      drop.Attach((HDROP)p);
      drop.QueryFileNames(names);
      return true;
    }
  }
  return false;
}
*/

static bool ClipboardSetData(UINT uFormat, const void *data, size_t size) throw()
{
  NMemory::CGlobal global;
  if (!global.Alloc(GMEM_DDESHARE | GMEM_MOVEABLE, size))
    return false;
  {
    NMemory::CGlobalLock globalLock(global);
    LPVOID p = globalLock.GetPointer();
    if (!p)
      return false;
    memcpy(p, data, size);
  }
  if (::SetClipboardData(uFormat, global) == NULL)
    return false;
  global.Detach();
  return true;
}

bool ClipboardSetText(HWND owner, const UString &s)
{
  CClipboard clipboard;
  if (!clipboard.Open(owner))
    return false;
  if (!::EmptyClipboard())
    return false;

  bool res;
  res = ClipboardSetData(CF_UNICODETEXT, (const wchar_t *)s, (s.Len() + 1) * sizeof(wchar_t));
  #ifndef _UNICODE
  AString a (UnicodeStringToMultiByte(s, CP_ACP));
  if (ClipboardSetData(CF_TEXT, (const char *)a, (a.Len() + 1) * sizeof(char)))
    res = true;
  a = UnicodeStringToMultiByte(s, CP_OEMCP);
  if (ClipboardSetData(CF_OEMTEXT, (const char *)a, (a.Len() + 1) * sizeof(char)))
    res = true;
  #endif
  return res;
}
 
}
