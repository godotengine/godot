// Windows/CommonDialog.cpp

#include "StdAfx.h"

#include "../Common/MyBuffer.h"

#ifdef UNDER_CE
#include <commdlg.h>
#endif

#ifndef _UNICODE
#include "../Common/StringConvert.h"
#endif

#include "CommonDialog.h"
#include "Defs.h"
// #include "FileDir.h"

#ifndef _UNICODE
extern bool g_IsNT;
#endif

namespace NWindows {

/*
  GetSaveFileName()
  GetOpenFileName()
  OPENFILENAME

(lpstrInitialDir) : the initial directory.
DOCs: the algorithm for selecting the initial directory varies on different platforms:
{
  Win2000/XP/Vista:
    1. If lpstrFile contains a path, that path is the initial directory.
    2. Otherwise, lpstrInitialDir specifies the initial directory.

  Win7:
    If lpstrInitialDir has the same value as was passed the first time
    the application used an Open or Save As dialog box, the path
    most recently selected by the user is used as the initial directory.
}

Win10:
 in:
  function supports (lpstrInitialDir) path with super prefix "\\\\?\\"
  function supports (lpstrInitialDir) path with long path
  function doesn't support absolute (lpstrFile) path with super prefix "\\\\?\\"
  function doesn't support absolute (lpstrFile) path with long path
 out: the path with super prefix "\\\\?\\" will be returned, if selected path is long

WinXP-64 and Win10: if no filters, the system shows all files.
    but DOCs say: If all three members are zero or NULL,
        the system does not use any filters and does not
        show any files in the file list control of the dialog box.

in Win7+: GetOpenFileName() and GetSaveFileName()
    do not support pstrCustomFilter feature anymore
*/

#ifdef UNDER_CE
#define MY_OFN_PROJECT  0x00400000
#define MY_OFN_SHOW_ALL 0x01000000
#endif


/*
structures
  OPENFILENAMEW
  OPENFILENAMEA
contain additional members:
#if (_WIN32_WINNT >= 0x0500)
  void *pvReserved;
  DWORD dwReserved;
  DWORD FlagsEx;
#endif

If we compile the source code with (_WIN32_WINNT >= 0x0500), some functions
will not work at NT 4.0, if we use sizeof(OPENFILENAME).
We try to use reduced structure OPENFILENAME_NT4.
*/

// #if defined(_WIN64) || (defined(_WIN32_WINNT) && _WIN32_WINNT >= 0x0500)
#if defined(__GNUC__) && (__GNUC__ <= 9) || defined(Z7_OLD_WIN_SDK)
  #ifndef _UNICODE
  #define my_compatib_OPENFILENAMEA       OPENFILENAMEA
  #endif
  #define my_compatib_OPENFILENAMEW       OPENFILENAMEW

  // MinGW doesn't support some required macros. So we define them here:
  #ifndef CDSIZEOF_STRUCT
  #define CDSIZEOF_STRUCT(structname, member)  (((int)((LPBYTE)(&((structname*)0)->member) - ((LPBYTE)((structname*)0)))) + sizeof(((structname*)0)->member))
  #endif
  #ifndef _UNICODE
  #ifndef OPENFILENAME_SIZE_VERSION_400A
  #define OPENFILENAME_SIZE_VERSION_400A  CDSIZEOF_STRUCT(OPENFILENAMEA,lpTemplateName)
  #endif
  #endif
  #ifndef OPENFILENAME_SIZE_VERSION_400W
  #define OPENFILENAME_SIZE_VERSION_400W  CDSIZEOF_STRUCT(OPENFILENAMEW,lpTemplateName)
  #endif
  
  #ifndef _UNICODE
  #define my_compatib_OPENFILENAMEA_size OPENFILENAME_SIZE_VERSION_400A
  #endif
  #define my_compatib_OPENFILENAMEW_size OPENFILENAME_SIZE_VERSION_400W
#else
  #ifndef _UNICODE
  #define my_compatib_OPENFILENAMEA       OPENFILENAME_NT4A
  #define my_compatib_OPENFILENAMEA_size  sizeof(my_compatib_OPENFILENAMEA)
  #endif
  #define my_compatib_OPENFILENAMEW       OPENFILENAME_NT4W
  #define my_compatib_OPENFILENAMEW_size  sizeof(my_compatib_OPENFILENAMEW)
#endif
/*
#elif defined(UNDER_CE) || defined(_WIN64) || (_WIN32_WINNT < 0x0500)
// || !defined(WINVER)
  #ifndef _UNICODE
  #define my_compatib_OPENFILENAMEA       OPENFILENAMEA
  #define my_compatib_OPENFILENAMEA_size sizeof(OPENFILENAMEA)
  #endif
  #define my_compatib_OPENFILENAMEW       OPENFILENAMEW
  #define my_compatib_OPENFILENAMEW_size sizeof(OPENFILENAMEW)
#else

#endif
*/

#ifndef _UNICODE
#define CONV_U_To_A(dest, src, temp) AString temp; if (src) { temp = GetSystemString(src); dest = temp; }
#endif

bool CCommonDialogInfo::CommonDlg_BrowseForFile(LPCWSTR lpstrInitialDir, const UStringVector &filters)
{
  /* GetSaveFileName() and GetOpenFileName() could change current dir,
     if OFN_NOCHANGEDIR is not used.
     We can restore current dir manually, if it's required.
     22.02: we use OFN_NOCHANGEDIR. So we don't need to restore current dir manually. */
  // NFile::NDir::CCurrentDirRestorer curDirRestorer;

#ifndef _UNICODE
  if (!g_IsNT)
  {
    AString tempPath;
    AStringVector f;
    unsigned i;
    for (i = 0; i < filters.Size(); i++)
      f.Add(GetSystemString(filters[i]));
    unsigned size = f.Size() + 1;
    for (i = 0; i < f.Size(); i++)
      size += f[i].Len();
    CObjArray<char> filterBuf(size);
    // memset(filterBuf, 0, size * sizeof(char));
    {
      char *dest = filterBuf;
      for (i = 0; i < f.Size(); i++)
      {
        const AString &s = f[i];
        MyStringCopy(dest, s);
        dest += s.Len() + 1;
      }
      *dest = 0;
    }
    my_compatib_OPENFILENAMEA p;
    memset(&p, 0, sizeof(p));
    p.lStructSize = my_compatib_OPENFILENAMEA_size;
    p.hwndOwner = hwndOwner;
    if (size > 1)
    {
      p.lpstrFilter = filterBuf;
      p.nFilterIndex = (DWORD)(FilterIndex + 1);
    }

    CONV_U_To_A(p.lpstrInitialDir, lpstrInitialDir, initialDir_a)
    CONV_U_To_A(p.lpstrTitle, lpstrTitle, title_a)

    const AString filePath_a = GetSystemString(FilePath);
    const unsigned bufSize = MAX_PATH * 8
        + filePath_a.Len()
        + initialDir_a.Len();
    p.nMaxFile = bufSize;
    p.lpstrFile = tempPath.GetBuf(bufSize);
    MyStringCopy(p.lpstrFile, filePath_a);
    p.Flags =
          OFN_EXPLORER
        | OFN_HIDEREADONLY
        | OFN_NOCHANGEDIR;
    const BOOL b = SaveMode ?
        ::GetSaveFileNameA((LPOPENFILENAMEA)(void *)&p) :
        ::GetOpenFileNameA((LPOPENFILENAMEA)(void *)&p);
    if (!b)
      return false;
    {
      tempPath.ReleaseBuf_CalcLen(bufSize);
      FilePath = GetUnicodeString(tempPath);
      FilterIndex = (int)p.nFilterIndex - 1;
      return true;
    }
  }
  else
#endif
  {
    UString tempPath;
    unsigned size = filters.Size() + 1;
    unsigned i;
    for (i = 0; i < filters.Size(); i++)
      size += filters[i].Len();
    CObjArray<wchar_t> filterBuf(size);
    // memset(filterBuf, 0, size * sizeof(wchar_t));
    {
      wchar_t *dest = filterBuf;
      for (i = 0; i < filters.Size(); i++)
      {
        const UString &s = filters[i];
        MyStringCopy(dest, s);
        dest += s.Len() + 1;
      }
      *dest = 0;
      // if ((unsigned)(dest + 1 - filterBuf) != size) return false;
    }
    my_compatib_OPENFILENAMEW p;
    memset(&p, 0, sizeof(p));
    p.lStructSize = my_compatib_OPENFILENAMEW_size;
    p.hwndOwner = hwndOwner;
    if (size > 1)
    {
      p.lpstrFilter = filterBuf;
      p.nFilterIndex = (DWORD)(FilterIndex + 1);
    }
    unsigned bufSize = MAX_PATH * 8 + FilePath.Len();
    if (lpstrInitialDir)
    {
      p.lpstrInitialDir = lpstrInitialDir;
      bufSize += MyStringLen(lpstrInitialDir);
    }
    p.nMaxFile = bufSize;
    p.lpstrFile = tempPath.GetBuf(bufSize);
    MyStringCopy(p.lpstrFile, FilePath);
    p.lpstrTitle = lpstrTitle;
    p.Flags =
          OFN_EXPLORER
        | OFN_HIDEREADONLY
        | OFN_NOCHANGEDIR
        // | OFN_FORCESHOWHIDDEN // Win10 shows hidden items even without this flag
        // | OFN_PATHMUSTEXIST
      #ifdef UNDER_CE
        | (OpenFolderMode ? (MY_OFN_PROJECT | MY_OFN_SHOW_ALL) : 0)
      #endif
        ;
    const BOOL b = SaveMode ?
        ::GetSaveFileNameW((LPOPENFILENAMEW)(void *)&p) :
        ::GetOpenFileNameW((LPOPENFILENAMEW)(void *)&p);
    /* DOCs: lpstrFile :
        if the buffer is too small, then:
        - the function returns FALSE
        - the CommDlgExtendedError() returns FNERR_BUFFERTOOSMALL
        - the first two bytes of the lpstrFile buffer contain the
          required size, in bytes or characters. */
    if (!b)
      return false;
    {
      tempPath.ReleaseBuf_CalcLen(bufSize);
      FilePath = tempPath;
      FilterIndex = (int)p.nFilterIndex - 1;
      return true;
    }
  }
}

}
