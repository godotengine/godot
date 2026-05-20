// Archive/Common/ItemNameUtils.cpp

#include "StdAfx.h"

#include "ItemNameUtils.h"

namespace NArchive {
namespace NItemName {

static const wchar_t kOsPathSepar = WCHAR_PATH_SEPARATOR;

#if WCHAR_PATH_SEPARATOR != L'/'
static const wchar_t kUnixPathSepar = L'/';
#endif

void ReplaceSlashes_OsToUnix
#if WCHAR_PATH_SEPARATOR != L'/'
  (UString &name)
  {
    name.Replace(kOsPathSepar, kUnixPathSepar);
  }
#else
  (UString &) {}
#endif


UString GetOsPath(const UString &name)
{
  #if WCHAR_PATH_SEPARATOR != L'/'
    UString newName = name;
    newName.Replace(kUnixPathSepar, kOsPathSepar);
    return newName;
  #else
    return name;
  #endif
}


UString GetOsPath_Remove_TailSlash(const UString &name)
{
  if (name.IsEmpty())
    return UString();
  UString newName = GetOsPath(name);
  if (newName.Back() == kOsPathSepar)
    newName.DeleteBack();
  return newName;
}


#if WCHAR_PATH_SEPARATOR != L'/'
void ReplaceToWinSlashes(UString &name, bool useBackslashReplacement)
{
  // name.Replace(kUnixPathSepar, kOsPathSepar);
  const unsigned len = name.Len();
  for (unsigned i = 0; i < len; i++)
  {
    wchar_t c = name[i];
    if (c == L'/')
      c = WCHAR_PATH_SEPARATOR;
    else if (useBackslashReplacement && c == L'\\')
      c = WCHAR_IN_FILE_NAME_BACKSLASH_REPLACEMENT; // WSL scheme
    else
      continue;
    name.ReplaceOneCharAtPos(i, c);
  }
}
#endif

void ReplaceToOsSlashes_Remove_TailSlash(UString &name, bool
    #if WCHAR_PATH_SEPARATOR != L'/'
      useBackslashReplacement
    #endif
    )
{
  if (name.IsEmpty())
    return;

  #if WCHAR_PATH_SEPARATOR != L'/'
  ReplaceToWinSlashes(name, useBackslashReplacement);
  #endif
    
  if (name.Back() == kOsPathSepar)
    name.DeleteBack();
}


void NormalizeSlashes_in_FileName_for_OsPath(wchar_t *name, unsigned len)
{
  for (unsigned i = 0; i < len; i++)
  {
    wchar_t c = name[i];
    if (c == L'/')
      c = L'_';
   #if WCHAR_PATH_SEPARATOR != L'/'
    else if (c == L'\\')
      c = WCHAR_IN_FILE_NAME_BACKSLASH_REPLACEMENT; // WSL scheme
   #endif
    else
      continue;
    name[i] = c;
  }
}

void NormalizeSlashes_in_FileName_for_OsPath(UString &name)
{
  NormalizeSlashes_in_FileName_for_OsPath(name.GetBuf(), name.Len());
}


bool HasTailSlash(const AString &name, UINT
  #if defined(_WIN32) && !defined(UNDER_CE)
    codePage
  #endif
  )
{
  if (name.IsEmpty())
    return false;
  char c;
    #if defined(_WIN32) && !defined(UNDER_CE)
    if (codePage != CP_UTF8)
      c = *CharPrevExA((WORD)codePage, name, name.Ptr(name.Len()), 0);
    else
    #endif
    {
      c = name.Back();
    }
  return (c == '/');
}


#ifndef _WIN32
UString WinPathToOsPath(const UString &name)
{
  UString newName = name;
  newName.Replace(L'\\', WCHAR_PATH_SEPARATOR);
  return newName;
}
#endif

}}
