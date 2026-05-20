// SysIconUtils.cpp

#include "StdAfx.h"

#ifndef _UNICODE
#include "../../../Common/StringConvert.h"
#endif

#include "../../../Windows/FileDir.h"

#include "SysIconUtils.h"

#if defined(__MINGW32__) || defined(__MINGW64__)
#include <shlobj.h>
#else
#include <ShlObj.h>
#endif

#ifndef _UNICODE
extern bool g_IsNT;
#endif

CExtToIconMap g_Ext_to_Icon_Map;

int Shell_GetFileInfo_SysIconIndex_for_CSIDL(int csidl)
{
  LPITEMIDLIST pidl = NULL;
  SHGetSpecialFolderLocation(NULL, csidl, &pidl);
  if (pidl)
  {
    SHFILEINFO shFileInfo;
    shFileInfo.iIcon = -1;
    const DWORD_PTR res = SHGetFileInfo((LPCTSTR)(const void *)(pidl),
        FILE_ATTRIBUTE_DIRECTORY,
        &shFileInfo, sizeof(shFileInfo),
        SHGFI_PIDL | SHGFI_SYSICONINDEX);
    /*
    IMalloc *pMalloc;
    SHGetMalloc(&pMalloc);
    if (pMalloc)
    {
      pMalloc->Free(pidl);
      pMalloc->Release();
    }
    */
    // we use OLE2.dll function here
    CoTaskMemFree(pidl);
    if (res)
      return shFileInfo.iIcon;
  }
  return -1;
}

#ifndef _UNICODE
Z7_DIAGNOSTIC_IGNORE_CAST_FUNCTION
typedef DWORD_PTR (WINAPI * Func_SHGetFileInfoW)(LPCWSTR pszPath, DWORD attrib, SHFILEINFOW *psfi, UINT cbFileInfo, UINT uFlags);

static struct C_SHGetFileInfo_Init
{
  Func_SHGetFileInfoW f_SHGetFileInfoW;
  C_SHGetFileInfo_Init()
  {
       f_SHGetFileInfoW = Z7_GET_PROC_ADDRESS(
    Func_SHGetFileInfoW, ::GetModuleHandleW(L"shell32.dll"),
        "SHGetFileInfoW");
    // f_SHGetFileInfoW = NULL; // for debug
  }
} g_SHGetFileInfo_Init;
#endif

#ifdef _UNICODE
#define My_SHGetFileInfoW SHGetFileInfoW
#else
static DWORD_PTR My_SHGetFileInfoW(LPCWSTR pszPath, DWORD attrib, SHFILEINFOW *psfi, UINT cbFileInfo, UINT uFlags)
{
  if (!g_SHGetFileInfo_Init.f_SHGetFileInfoW)
    return 0;
  return g_SHGetFileInfo_Init.f_SHGetFileInfoW(pszPath, attrib, psfi, cbFileInfo, uFlags);
}
#endif

DWORD_PTR Shell_GetFileInfo_SysIconIndex_for_Path_attrib_iconIndexRef(
    CFSTR path, DWORD attrib, int &iconIndex)
{
#ifndef _UNICODE
  if (!g_IsNT || !g_SHGetFileInfo_Init.f_SHGetFileInfoW)
  {
    SHFILEINFO shFileInfo;
    // ZeroMemory(&shFileInfo, sizeof(shFileInfo));
    shFileInfo.iIcon = -1;   // optional
    const DWORD_PTR res = ::SHGetFileInfo(fs2fas(path),
        attrib ? attrib : FILE_ATTRIBUTE_ARCHIVE,
        &shFileInfo, sizeof(shFileInfo),
        SHGFI_USEFILEATTRIBUTES | SHGFI_SYSICONINDEX);
    iconIndex = shFileInfo.iIcon;
    return res;
  }
  else
#endif
  {
    SHFILEINFOW shFileInfo;
    // ZeroMemory(&shFileInfo, sizeof(shFileInfo));
    shFileInfo.iIcon = -1;   // optional
    const DWORD_PTR res = ::My_SHGetFileInfoW(fs2us(path),
        attrib ? attrib : FILE_ATTRIBUTE_ARCHIVE,
        &shFileInfo, sizeof(shFileInfo),
        SHGFI_USEFILEATTRIBUTES | SHGFI_SYSICONINDEX);
    // (shFileInfo.iIcon == 0) returned for unknown extensions and files without extension
    iconIndex = shFileInfo.iIcon;
    // we use SHGFI_USEFILEATTRIBUTES, and
    //   (res != 0) is expected for main cases, even if there are no such file.
    //   (res == 0) for path with kSuperPrefix "\\?\"
    // Also SHGFI_USEFILEATTRIBUTES still returns icon inside exe.
    // So we can use SHGFI_USEFILEATTRIBUTES for any case.
    // UString temp = fs2us(path); // for debug
    // UString tempName = temp.Ptr(temp.ReverseFind_PathSepar() + 1); // for debug
    // iconIndex = -1; // for debug
    return res;
  }
}

int Shell_GetFileInfo_SysIconIndex_for_Path(CFSTR path, DWORD attrib)
{
  int iconIndex = -1;
  if (!Shell_GetFileInfo_SysIconIndex_for_Path_attrib_iconIndexRef(
      path, attrib, iconIndex))
    iconIndex = -1;
  return iconIndex;
}


HRESULT Shell_GetFileInfo_SysIconIndex_for_Path_return_HRESULT(
    CFSTR path, DWORD attrib, Int32 *iconIndex)
{
  *iconIndex = -1;
  int iconIndexTemp;
  if (Shell_GetFileInfo_SysIconIndex_for_Path_attrib_iconIndexRef(
      path, attrib, iconIndexTemp))
  {
    *iconIndex = iconIndexTemp;
    return S_OK;
  }
  return GetLastError_noZero_HRESULT();
}

/*
DWORD_PTR Shell_GetFileInfo_SysIconIndex_for_Path(const UString &fileName, DWORD attrib, int &iconIndex, UString *typeName)
{
  #ifndef _UNICODE
  if (!g_IsNT)
  {
    SHFILEINFO shFileInfo;
    shFileInfo.szTypeName[0] = 0;
    DWORD_PTR res = ::SHGetFileInfoA(GetSystemString(fileName), FILE_ATTRIBUTE_ARCHIVE | attrib, &shFileInfo,
        sizeof(shFileInfo), SHGFI_USEFILEATTRIBUTES | SHGFI_SYSICONINDEX | SHGFI_TYPENAME);
    if (typeName)
      *typeName = GetUnicodeString(shFileInfo.szTypeName);
    iconIndex = shFileInfo.iIcon;
    return res;
  }
  else
  #endif
  {
    SHFILEINFOW shFileInfo;
    shFileInfo.szTypeName[0] = 0;
    DWORD_PTR res = ::My_SHGetFileInfoW(fileName, FILE_ATTRIBUTE_ARCHIVE | attrib, &shFileInfo,
        sizeof(shFileInfo), SHGFI_USEFILEATTRIBUTES | SHGFI_SYSICONINDEX | SHGFI_TYPENAME);
    if (typeName)
      *typeName = shFileInfo.szTypeName;
    iconIndex = shFileInfo.iIcon;
    return res;
  }
}
*/

static int FindInSorted_Attrib(const CRecordVector<CAttribIconPair> &vect, DWORD attrib, unsigned &insertPos)
{
  unsigned left = 0, right = vect.Size();
  while (left != right)
  {
    const unsigned mid = (left + right) / 2;
    const DWORD midAttrib = vect[mid].Attrib;
    if (attrib == midAttrib)
      return (int)mid;
    if (attrib < midAttrib)
      right = mid;
    else
      left = mid + 1;
  }
  insertPos = left;
  return -1;
}

static int FindInSorted_Ext(const CObjectVector<CExtIconPair> &vect, const wchar_t *ext, unsigned &insertPos)
{
  unsigned left = 0, right = vect.Size();
  while (left != right)
  {
    const unsigned mid = (left + right) / 2;
    const int compare = MyStringCompareNoCase(ext, vect[mid].Ext);
    if (compare == 0)
      return (int)mid;
    if (compare < 0)
      right = mid;
    else
      left = mid + 1;
  }
  insertPos = left;
  return -1;
}


// bool DoItemAlwaysStart(const UString &name);

int CExtToIconMap::GetIconIndex(DWORD attrib, const wchar_t *fileName /*, UString *typeName */)
{
  int dotPos = -1;
  unsigned i;
  for (i = 0;; i++)
  {
    const wchar_t c = fileName[i];
    if (c == 0)
      break;
    if (c == '.')
      dotPos = (int)i;
    // we don't need IS_PATH_SEPAR check, because (fileName) doesn't include path prefix.
    // if (IS_PATH_SEPAR(c) || c == ':') dotPos = -1;
  }

  /*
  if (MyStringCompareNoCase(fileName, L"$Recycle.Bin") == 0)
  {
    char s[256];
    sprintf(s, "SPEC i = %3d, attr = %7x", _attribMap.Size(), attrib);
    OutputDebugStringA(s);
    OutputDebugStringW(fileName);
  }
  */

  if ((attrib & FILE_ATTRIBUTE_DIRECTORY) || dotPos < 0)
  for (unsigned k = 0;; k++)
  {
    if (k >= 2)
      return -1;
    unsigned insertPos = 0;
    const int index = FindInSorted_Attrib(_attribMap, attrib, insertPos);
    if (index >= 0)
    {
      // if (typeName) *typeName = _attribMap[index].TypeName;
      return _attribMap[(unsigned)index].IconIndex;
    }
    CAttribIconPair pair;
    pair.IconIndex = Shell_GetFileInfo_SysIconIndex_for_Path(
        #ifdef UNDER_CE
        FTEXT("\\")
        #endif
        FTEXT("__DIR__")
        , attrib
        // , pair.TypeName
        );
    if (_attribMap.Size() < (1u << 16) // we limit cache size
       || attrib < (1u << 15)) // we want to put all items with basic attribs to cache
    {
      /*
      char s[256];
      sprintf(s, "i = %3d, attr = %7x", _attribMap.Size(), attrib);
      OutputDebugStringA(s);
      */
      pair.Attrib = attrib;
      _attribMap.Insert(insertPos, pair);
      // if (typeName) *typeName = pair.TypeName;
      return pair.IconIndex;
    }
    if (pair.IconIndex >= 0)
      return pair.IconIndex;
    attrib = (attrib & FILE_ATTRIBUTE_DIRECTORY) ?
        FILE_ATTRIBUTE_DIRECTORY :
        FILE_ATTRIBUTE_ARCHIVE;
  }

  CObjectVector<CExtIconPair> &map =
      (attrib & FILE_ATTRIBUTE_COMPRESSED) ?
          _extMap_Compressed : _extMap_Normal;
  const wchar_t *ext = fileName + dotPos + 1;
  unsigned insertPos = 0;
  const int index = FindInSorted_Ext(map, ext, insertPos);
  if (index >= 0)
  {
    const CExtIconPair &pa = map[index];
    // if (typeName) *typeName = pa.TypeName;
    return pa.IconIndex;
  }

  for (i = 0;; i++)
  {
    const wchar_t c = ext[i];
    if (c == 0)
      break;
    if (c < L'0' || c > L'9')
      break;
  }
  if (i != 0 && ext[i] == 0)
  {
    // Shell_GetFileInfo_SysIconIndex_for_Path is too slow for big number of split extensions: .001, .002, .003
    if (!SplitIconIndex_Defined)
    {
      Shell_GetFileInfo_SysIconIndex_for_Path_attrib_iconIndexRef(
          #ifdef UNDER_CE
          FTEXT("\\")
          #endif
          FTEXT("__FILE__.001"), FILE_ATTRIBUTE_ARCHIVE, SplitIconIndex);
      SplitIconIndex_Defined = true;
    }
    return SplitIconIndex;
  }

  CExtIconPair pair;
  pair.Ext = ext;
  pair.IconIndex = Shell_GetFileInfo_SysIconIndex_for_Path(
      us2fs(fileName + dotPos),
      attrib & FILE_ATTRIBUTE_COMPRESSED ?
          FILE_ATTRIBUTE_ARCHIVE | FILE_ATTRIBUTE_COMPRESSED:
          FILE_ATTRIBUTE_ARCHIVE);
  if (map.Size() < (1u << 16)  // we limit cache size
      // || DoItemAlwaysStart(fileName + dotPos) // we want some popular extensions in cache
      )
    map.Insert(insertPos, pair);
  // if (typeName) *typeName = pair.TypeName;
  return pair.IconIndex;
}


HIMAGELIST Shell_Get_SysImageList_smallIcons(bool smallIcons)
{
  SHFILEINFO shFileInfo;
  // shFileInfo.hIcon = NULL; // optional
  const DWORD_PTR res = SHGetFileInfo(TEXT(""),
      /* FILE_ATTRIBUTE_ARCHIVE | */
      FILE_ATTRIBUTE_DIRECTORY,
      &shFileInfo, sizeof(shFileInfo),
      SHGFI_USEFILEATTRIBUTES |
      SHGFI_SYSICONINDEX |
      (smallIcons ? SHGFI_SMALLICON : SHGFI_LARGEICON));
#if 0
  // (shFileInfo.hIcon == NULL), because we don't use SHGFI_ICON.
  // so DestroyIcon() is not required
  if (res && shFileInfo.hIcon) // unexpected
    DestroyIcon(shFileInfo.hIcon);
#endif
  return (HIMAGELIST)res;
}
