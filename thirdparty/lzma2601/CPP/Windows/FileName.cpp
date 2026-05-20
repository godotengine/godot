// Windows/FileName.cpp

#include "StdAfx.h"

#ifndef _WIN32
#include <limits.h>
#include <unistd.h>
#include "../Common/StringConvert.h"
#endif

#include "FileDir.h"
#include "FileName.h"

#ifndef _UNICODE
extern bool g_IsNT;
#endif

namespace NWindows {
namespace NFile {
namespace NName {

#define IS_SEPAR(c) IS_PATH_SEPAR(c)

int FindSepar(const wchar_t *s) throw()
{
  for (const wchar_t *p = s;; p++)
  {
    const wchar_t c = *p;
    if (c == 0)
      return -1;
    if (IS_SEPAR(c))
      return (int)(p - s);
  }
}

#ifndef USE_UNICODE_FSTRING
int FindSepar(const FChar *s) throw()
{
  for (const FChar *p = s;; p++)
  {
    const FChar c = *p;
    if (c == 0)
      return -1;
    if (IS_SEPAR(c))
      return (int)(p - s);
  }
}
#endif

#ifndef USE_UNICODE_FSTRING
void NormalizeDirPathPrefix(FString &dirPath)
{
  if (dirPath.IsEmpty())
    return;
  if (!IsPathSepar(dirPath.Back()))
    dirPath.Add_PathSepar();
}
#endif

void NormalizeDirPathPrefix(UString &dirPath)
{
  if (dirPath.IsEmpty())
    return;
  if (!IsPathSepar(dirPath.Back()))
    dirPath.Add_PathSepar();
}


#define IS_LETTER_CHAR(c) ((((unsigned)(int)(c) | 0x20) - (unsigned)'a' <= (unsigned)('z' - 'a')))
bool IsDrivePath (const wchar_t *s) throw() { return IS_LETTER_CHAR(s[0]) && s[1] == ':' && IS_SEPAR(s[2]); }
// bool IsDriveName2(const wchar_t *s) throw() { return IS_LETTER_CHAR(s[0]) && s[1] == ':' && s[2] == 0; }

#ifdef _WIN32

bool IsDrivePath2(const wchar_t *s) throw() { return IS_LETTER_CHAR(s[0]) && s[1] == ':'; }

#ifndef USE_UNICODE_FSTRING
#ifdef Z7_LONG_PATH
static void NormalizeDirSeparators(UString &s)
{
  const unsigned len = s.Len();
  for (unsigned i = 0; i < len; i++)
    if (s[i] == '/')
      s.ReplaceOneCharAtPos(i, WCHAR_PATH_SEPARATOR);
}
#endif
#endif

void NormalizeDirSeparators(FString &s)
{
  const unsigned len = s.Len();
  for (unsigned i = 0; i < len; i++)
    if (s[i] == '/')
      s.ReplaceOneCharAtPos(i, FCHAR_PATH_SEPARATOR);
}

bool IsAltPathPrefix(CFSTR s) throw()
{
  unsigned len = MyStringLen(s);
  if (len == 0)
    return false;
  if (s[len - 1] != ':')
    return false;

  #if defined(_WIN32) && !defined(UNDER_CE)
  if (IsDevicePath(s))
    return false;
  if (IsSuperPath(s))
  {
    s += kSuperPathPrefixSize;
    len -= kSuperPathPrefixSize;
  }
  if (len == 2 && IsDrivePath2(s))
    return false;
  #endif

  return true;
}

#endif // _WIN32


const char * const kSuperPathPrefix =
    STRING_PATH_SEPARATOR
    STRING_PATH_SEPARATOR "?"
    STRING_PATH_SEPARATOR;
#ifdef Z7_LONG_PATH
static const char * const kSuperUncPrefix =
    STRING_PATH_SEPARATOR
    STRING_PATH_SEPARATOR "?"
    STRING_PATH_SEPARATOR "UNC"
    STRING_PATH_SEPARATOR;
#endif

#define IS_DEVICE_PATH(s)          (IS_SEPAR((s)[0]) && IS_SEPAR((s)[1]) && (s)[2] == '.' && IS_SEPAR((s)[3]))
#define IS_SUPER_PREFIX(s)         (IS_SEPAR((s)[0]) && IS_SEPAR((s)[1]) && (s)[2] == '?' && IS_SEPAR((s)[3]))

#define IS_UNC_WITH_SLASH(s) ( \
     ((s)[0] == 'U' || (s)[0] == 'u') \
  && ((s)[1] == 'N' || (s)[1] == 'n') \
  && ((s)[2] == 'C' || (s)[2] == 'c') \
  && IS_SEPAR((s)[3]))

static const unsigned kDrivePrefixSize = 3; /* c:\ */

bool IsSuperPath(const wchar_t *s) throw();
bool IsSuperPath(const wchar_t *s) throw() { return IS_SUPER_PREFIX(s); }
// bool IsSuperUncPath(const wchar_t *s) throw() { return (IS_SUPER_PREFIX(s) && IS_UNC_WITH_SLASH(s + kSuperPathPrefixSize)); }

#if defined(_WIN32) && !defined(UNDER_CE)

#define IS_SUPER_OR_DEVICE_PATH(s) (IS_SEPAR((s)[0]) && IS_SEPAR((s)[1]) && ((s)[2] == '?' || (s)[2] == '.') && IS_SEPAR((s)[3]))
bool IsSuperOrDevicePath(const wchar_t *s) throw() { return IS_SUPER_OR_DEVICE_PATH(s); }
bool IsDevicePath(CFSTR s) throw()
{
  #ifdef UNDER_CE

  s = s;
  return false;
  /*
  // actually we don't know the way to open device file in WinCE.
  unsigned len = MyStringLen(s);
  if (len < 5 || len > 5 || !IsString1PrefixedByString2(s, "DSK"))
    return false;
  if (s[4] != ':')
    return false;
  // for reading use SG_REQ sg; if (DeviceIoControl(dsk, IOCTL_DISK_READ));
  */
  
  #else
  
  if (!IS_DEVICE_PATH(s))
    return false;
  const unsigned len = MyStringLen(s);
  if (len == 6 && s[5] == ':')
    return true;
  if (len < 18 || len > 22 || !IsString1PrefixedByString2(s + kDevicePathPrefixSize, "PhysicalDrive"))
    return false;
  for (unsigned i = 17; i < len; i++)
    if (s[i] < '0' || s[i] > '9')
      return false;
  return true;
  
  #endif
}

bool IsSuperUncPath(CFSTR s) throw() { return (IS_SUPER_PREFIX(s) && IS_UNC_WITH_SLASH(s + kSuperPathPrefixSize)); }
bool IsNetworkPath(CFSTR s) throw()
{
  if (!IS_SEPAR(s[0]) || !IS_SEPAR(s[1]))
    return false;
  if (IsSuperUncPath(s))
    return true;
  const FChar c = s[2];
  return (c != '.' && c != '?');
}

unsigned GetNetworkServerPrefixSize(CFSTR s) throw()
{
  if (!IS_SEPAR(s[0]) || !IS_SEPAR(s[1]))
    return 0;
  unsigned prefixSize = 2;
  if (IsSuperUncPath(s))
    prefixSize = kSuperUncPathPrefixSize;
  else
  {
    const FChar c = s[2];
    if (c == '.' || c == '?')
      return 0;
  }
  const int pos = FindSepar(s + prefixSize);
  if (pos < 0)
    return 0;
  return prefixSize + (unsigned)(pos + 1);
}

bool IsNetworkShareRootPath(CFSTR s) throw()
{
  const unsigned prefixSize = GetNetworkServerPrefixSize(s);
  if (prefixSize == 0)
    return false;
  s += prefixSize;
  const int pos = FindSepar(s);
  if (pos < 0)
    return true;
  return s[(unsigned)pos + 1] == 0;
}

bool IsAltStreamPrefixWithColon(const UString &s) throw()
{
  if (s.IsEmpty())
    return false;
  if (s.Back() != ':')
    return false;
  unsigned pos = 0;
  if (IsSuperPath(s))
    pos = kSuperPathPrefixSize;
  if (s.Len() - pos == 2 && IsDrivePath2(s.Ptr(pos)))
    return false;
  return true;
}

bool If_IsSuperPath_RemoveSuperPrefix(UString &s)
{
  if (!IsSuperPath(s))
    return false;
  unsigned start = 0;
  unsigned count = kSuperPathPrefixSize;
  const wchar_t *s2 = s.Ptr(kSuperPathPrefixSize);
  if (IS_UNC_WITH_SLASH(s2))
  {
    start = 2;
    count = kSuperUncPathPrefixSize - 2;
  }
  s.Delete(start, count);
  return true;
}


#ifndef USE_UNICODE_FSTRING
bool IsDrivePath2(CFSTR s) throw() { return IS_LETTER_CHAR(s[0]) && s[1] == ':'; }
// bool IsDriveName2(CFSTR s) throw() { return IS_LETTER_CHAR(s[0]) && s[1] == ':' && s[2] == 0; }
bool IsDrivePath(CFSTR s) throw() { return IS_LETTER_CHAR(s[0]) && s[1] == ':' && IS_SEPAR(s[2]); }
bool IsSuperPath(CFSTR s) throw() { return IS_SUPER_PREFIX(s); }
bool IsSuperOrDevicePath(CFSTR s) throw() { return IS_SUPER_OR_DEVICE_PATH(s); }
#endif // USE_UNICODE_FSTRING

bool IsDrivePath_SuperAllowed(CFSTR s) throw()
{
  if (IsSuperPath(s))
    s += kSuperPathPrefixSize;
  return IsDrivePath(s);
}

bool IsDriveRootPath_SuperAllowed(CFSTR s) throw()
{
  if (IsSuperPath(s))
    s += kSuperPathPrefixSize;
  return IsDrivePath(s) && s[kDrivePrefixSize] == 0;
}

bool IsAbsolutePath(const wchar_t *s) throw()
{
  return IS_SEPAR(s[0]) || IsDrivePath2(s);
}

int FindAltStreamColon(CFSTR path) throw()
{
  unsigned i = 0;
  if (IsSuperPath(path))
    i = kSuperPathPrefixSize;
  if (IsDrivePath2(path + i))
    i += 2;
  int colonPos = -1;
  for (;; i++)
  {
    const FChar c = path[i];
    if (c == 0)
      return colonPos;
    if (c == ':')
    {
      if (colonPos < 0)
        colonPos = (int)i;
      continue;
    }
    if (IS_SEPAR(c))
      colonPos = -1;
  }
}

#ifndef USE_UNICODE_FSTRING

static unsigned GetRootPrefixSize_Of_NetworkPath(CFSTR s)
{
  // Network path: we look "server\path\" as root prefix
  const int pos = FindSepar(s);
  if (pos < 0)
    return 0;
  const int pos2 = FindSepar(s + (unsigned)pos + 1);
  if (pos2 < 0)
    return 0;
  return (unsigned)pos + (unsigned)pos2 + 2;
}

static unsigned GetRootPrefixSize_Of_SimplePath(CFSTR s)
{
  if (IsDrivePath(s))
    return kDrivePrefixSize;
  if (!IS_SEPAR(s[0]))
    return 0;
  if (s[1] == 0 || !IS_SEPAR(s[1]))
    return 1;
  const unsigned size = GetRootPrefixSize_Of_NetworkPath(s + 2);
  return (size == 0) ? 0 : 2 + size;
}

static unsigned GetRootPrefixSize_Of_SuperPath(CFSTR s)
{
  if (IS_UNC_WITH_SLASH(s + kSuperPathPrefixSize))
  {
    const unsigned size = GetRootPrefixSize_Of_NetworkPath(s + kSuperUncPathPrefixSize);
    return (size == 0) ? 0 : kSuperUncPathPrefixSize + size;
  }
  // we support \\?\c:\ paths and volume GUID paths \\?\Volume{GUID}\"
  const int pos = FindSepar(s + kSuperPathPrefixSize);
  if (pos < 0)
    return 0;
  return kSuperPathPrefixSize + (unsigned)pos + 1;
}

unsigned GetRootPrefixSize(CFSTR s) throw()
{
  if (IS_DEVICE_PATH(s))
    return kDevicePathPrefixSize;
  if (IsSuperPath(s))
    return GetRootPrefixSize_Of_SuperPath(s);
  return GetRootPrefixSize_Of_SimplePath(s);
}

#endif // USE_UNICODE_FSTRING
#endif // _WIN32


static unsigned GetRootPrefixSize_Of_NetworkPath(const wchar_t *s) throw()
{
  // Network path: we look "server\path\" as root prefix
  const int pos = FindSepar(s);
  if (pos < 0)
    return 0;
  const int pos2 = FindSepar(s + (unsigned)pos + 1);
  if (pos2 < 0)
    return 0;
  return (unsigned)(pos + pos2 + 2);
}

static unsigned GetRootPrefixSize_Of_SimplePath(const wchar_t *s) throw()
{
  if (IsDrivePath(s))
    return kDrivePrefixSize;
  if (!IS_SEPAR(s[0]))
    return 0;
  if (s[1] == 0 || !IS_SEPAR(s[1]))
    return 1;
  const unsigned size = GetRootPrefixSize_Of_NetworkPath(s + 2);
  return (size == 0) ? 0 : 2 + size;
}

static unsigned GetRootPrefixSize_Of_SuperPath(const wchar_t *s) throw()
{
  if (IS_UNC_WITH_SLASH(s + kSuperPathPrefixSize))
  {
    const unsigned size = GetRootPrefixSize_Of_NetworkPath(s + kSuperUncPathPrefixSize);
    return (size == 0) ? 0 : kSuperUncPathPrefixSize + size;
  }
  // we support \\?\c:\ paths and volume GUID paths \\?\Volume{GUID}\"
  const int pos = FindSepar(s + kSuperPathPrefixSize);
  if (pos < 0)
    return 0;
  return kSuperPathPrefixSize + (unsigned)(pos + 1);
}

#ifdef _WIN32
unsigned GetRootPrefixSize(const wchar_t *s) throw()
#else
unsigned GetRootPrefixSize_WINDOWS(const wchar_t *s) throw()
#endif
{
  if (IS_DEVICE_PATH(s))
    return kDevicePathPrefixSize;
  if (IsSuperPath(s))
    return GetRootPrefixSize_Of_SuperPath(s);
  return GetRootPrefixSize_Of_SimplePath(s);
}

#ifndef _WIN32

bool IsAbsolutePath(const wchar_t *s) throw() { return IS_SEPAR(s[0]); }

#ifndef USE_UNICODE_FSTRING
unsigned GetRootPrefixSize(CFSTR s) throw();
unsigned GetRootPrefixSize(CFSTR s) throw() { return IS_SEPAR(s[0]) ? 1 : 0; }
#endif
unsigned GetRootPrefixSize(const wchar_t *s) throw() { return IS_SEPAR(s[0]) ? 1 : 0; }

#endif // _WIN32


#ifndef UNDER_CE


#ifdef USE_UNICODE_FSTRING

#define GetCurDir NDir::GetCurrentDir

#else

static bool GetCurDir(UString &path)
{
  path.Empty();
  FString s;
  if (!NDir::GetCurrentDir(s))
    return false;
  path = fs2us(s);
  return true;
}

#endif


static bool ResolveDotsFolders(UString &s)
{
  #ifdef _WIN32
  // s.Replace(L'/', WCHAR_PATH_SEPARATOR);
  #endif
  
  for (unsigned i = 0;;)
  {
    const wchar_t c = s[i];
    if (c == 0)
      return true;
    if (c == '.' && (i == 0 || IS_SEPAR(s[i - 1])))
    {
      const wchar_t c1 = s[i + 1];
      if (c1 == '.')
      {
        const wchar_t c2 = s[i + 2];
        if (IS_SEPAR(c2) || c2 == 0)
        {
          if (i == 0)
            return false;
          int k = (int)i - 2;
          i += 2;
          
          for (;; k--)
          {
            if (k < 0)
              return false;
            if (!IS_SEPAR(s[(unsigned)k]))
              break;
          }

          do
            k--;
          while (k >= 0 && !IS_SEPAR(s[(unsigned)k]));
          
          unsigned num;
          
          if (k >= 0)
          {
            num = i - (unsigned)k;
            i = (unsigned)k;
          }
          else
          {
            num = (c2 == 0 ? i : (i + 1));
            i = 0;
          }
          
          s.Delete(i, num);
          continue;
        }
      }
      else if (IS_SEPAR(c1) || c1 == 0)
      {
        unsigned num = 2;
        if (i != 0)
          i--;
        else if (c1 == 0)
          num = 1;
        s.Delete(i, num);
        continue;
      }
    }

    i++;
  }
}

#endif // UNDER_CE

#define LONG_PATH_DOTS_FOLDERS_PARSING


/*
Windows (at least 64-bit XP) can't resolve "." or ".." in paths that start with SuperPrefix \\?\
To solve that problem we check such path:
   - super path contains        "." or ".." - we use kSuperPathType_UseOnlySuper
   - super path doesn't contain "." or ".." - we use kSuperPathType_UseOnlyMain
*/
#ifdef LONG_PATH_DOTS_FOLDERS_PARSING
#ifndef UNDER_CE
static bool AreThereDotsFolders(CFSTR s)
{
  for (unsigned i = 0;; i++)
  {
    FChar c = s[i];
    if (c == 0)
      return false;
    if (c == '.' && (i == 0 || IS_SEPAR(s[i - 1])))
    {
      FChar c1 = s[i + 1];
      if (c1 == 0 || IS_SEPAR(c1) ||
          (c1 == '.' && (s[i + 2] == 0 || IS_SEPAR(s[i + 2]))))
        return true;
    }
  }
}
#endif
#endif // LONG_PATH_DOTS_FOLDERS_PARSING

#ifdef Z7_LONG_PATH

/*
Most of Windows versions have problems, if some file or dir name
contains '.' or ' ' at the end of name (Bad Path).
To solve that problem, we always use Super Path ("\\?\" prefix and full path)
in such cases. Note that "." and ".." are not bad names.

There are 3 cases:
  1) If the path is already Super Path, we use that path
  2) If the path is not Super Path :
     2.1) Bad Path;  we use only Super Path.
     2.2) Good Path; we use Main Path. If it fails, we use Super Path.

 NeedToUseOriginalPath returns:
    kSuperPathType_UseOnlyMain    : Super already
    kSuperPathType_UseOnlySuper    : not Super, Bad Path
    kSuperPathType_UseMainAndSuper : not Super, Good Path
*/

int GetUseSuperPathType(CFSTR s) throw()
{
  if (IsSuperOrDevicePath(s))
  {
    #ifdef LONG_PATH_DOTS_FOLDERS_PARSING
    if ((s)[2] != '.')
      if (AreThereDotsFolders(s + kSuperPathPrefixSize))
        return kSuperPathType_UseOnlySuper;
    #endif
    return kSuperPathType_UseOnlyMain;
  }

  for (unsigned i = 0;; i++)
  {
    FChar c = s[i];
    if (c == 0)
      return kSuperPathType_UseMainAndSuper;
    if (c == '.' || c == ' ')
    {
      FChar c2 = s[i + 1];
      if (c2 == 0 || IS_SEPAR(c2))
      {
        // if it's "." or "..", it's not bad name.
        if (c == '.')
        {
          if (i == 0 || IS_SEPAR(s[i - 1]))
            continue;
          if (s[i - 1] == '.')
          {
            if (i - 1 == 0 || IS_SEPAR(s[i - 2]))
              continue;
          }
        }
        return kSuperPathType_UseOnlySuper;
      }
    }
  }
}



/*
   returns false in two cases:
     - if GetCurDir was used, and GetCurDir returned error.
     - if we can't resolve ".." name.
   if path is ".", "..", res is empty.
   if it's Super Path already, res is empty.
   for \**** , and if GetCurDir is not drive (c:\), res is empty
   for absolute paths, returns true, res is Super path.
*/

static bool GetSuperPathBase(CFSTR s, UString &res)
{
  res.Empty();
  
  FChar c = s[0];
  if (c == 0)
    return true;
  if (c == '.' && (s[1] == 0 || (s[1] == '.' && s[2] == 0)))
    return true;
  
  if (IsSuperOrDevicePath(s))
  {
    #ifdef LONG_PATH_DOTS_FOLDERS_PARSING
    
    if ((s)[2] == '.')
      return true;

    // we will return true here, so we will try to use these problem paths.

    if (!AreThereDotsFolders(s + kSuperPathPrefixSize))
      return true;
    
    UString temp = fs2us(s);
    const unsigned fixedSize = GetRootPrefixSize_Of_SuperPath(temp);
    if (fixedSize == 0)
      return true;

    UString rem = temp.Ptr(fixedSize);
    if (!ResolveDotsFolders(rem))
      return true;

    temp.DeleteFrom(fixedSize);
    res += temp;
    res += rem;
    
    #endif

    return true;
  }

  if (IS_SEPAR(c))
  {
    if (IS_SEPAR(s[1]))
    {
      UString temp = fs2us(s + 2);
      const unsigned fixedSize = GetRootPrefixSize_Of_NetworkPath(temp);
      // we ignore that error to allow short network paths server\share?
      /*
      if (fixedSize == 0)
        return false;
      */
      UString rem = temp.Ptr(fixedSize);
      if (!ResolveDotsFolders(rem))
        return false;
      res += kSuperUncPrefix;
      temp.DeleteFrom(fixedSize);
      res += temp;
      res += rem;
      return true;
    }
  }
  else
  {
    if (IsDrivePath2(s))
    {
      UString temp = fs2us(s);
      unsigned prefixSize = 2;
      if (IsDrivePath(s))
        prefixSize = kDrivePrefixSize;
      UString rem = temp.Ptr(prefixSize);
      if (!ResolveDotsFolders(rem))
        return true;
      res += kSuperPathPrefix;
      temp.DeleteFrom(prefixSize);
      res += temp;
      res += rem;
      return true;
    }
  }

  UString curDir;
  if (!GetCurDir(curDir))
    return false;
  NormalizeDirPathPrefix(curDir);

  unsigned fixedSizeStart = 0;
  unsigned fixedSize = 0;
  const char *superMarker = NULL;
  if (IsSuperPath(curDir))
  {
    fixedSize = GetRootPrefixSize_Of_SuperPath(curDir);
    if (fixedSize == 0)
      return false;
  }
  else
  {
    if (IsDrivePath(curDir))
    {
      superMarker = kSuperPathPrefix;
      fixedSize = kDrivePrefixSize;
    }
    else
    {
      if (!IsPathSepar(curDir[0]) || !IsPathSepar(curDir[1]))
        return false;
      fixedSizeStart = 2;
      fixedSize = GetRootPrefixSize_Of_NetworkPath(curDir.Ptr(2));
      if (fixedSize == 0)
        return false;
      superMarker = kSuperUncPrefix;
    }
  }
  
  UString temp;
  if (IS_SEPAR(c))
  {
    temp = fs2us(s + 1);
  }
  else
  {
    temp += &curDir[fixedSizeStart + fixedSize];
    temp += fs2us(s);
  }
  if (!ResolveDotsFolders(temp))
    return false;
  if (superMarker)
    res += superMarker;
  res += curDir.Mid(fixedSizeStart, fixedSize);
  res += temp;
  return true;
}


/*
  In that case if GetSuperPathBase doesn't return new path, we don't need
  to use same path that was used as main path
                        
  GetSuperPathBase  superPath.IsEmpty() onlyIfNew
     false            *                *          GetCurDir Error
     true            false             *          use Super path
     true            true             true        don't use any path, we already used mainPath
     true            true             false       use main path as Super Path, we don't try mainMath
                                                  That case is possible now if GetCurDir returns unknown
                                                  type of path (not drive and not network)

  We can change that code if we want to try mainPath, if GetSuperPathBase returns error,
  and we didn't try mainPath still.
  If we want to work that way, we don't need to use GetSuperPathBase return code.
*/

bool GetSuperPath(CFSTR path, UString &superPath, bool onlyIfNew)
{
  if (GetSuperPathBase(path, superPath))
  {
    if (superPath.IsEmpty())
    {
      // actually the only possible when onlyIfNew == true and superPath is empty
      // is case when

      if (onlyIfNew)
        return false;
      superPath = fs2us(path);
    }
    
    NormalizeDirSeparators(superPath);
    return true;
  }
  return false;
}

bool GetSuperPaths(CFSTR s1, CFSTR s2, UString &d1, UString &d2, bool onlyIfNew)
{
  if (!GetSuperPathBase(s1, d1) ||
      !GetSuperPathBase(s2, d2))
    return false;
 
  NormalizeDirSeparators(d1);
  NormalizeDirSeparators(d2);

  if (d1.IsEmpty() && d2.IsEmpty() && onlyIfNew)
    return false;
  if (d1.IsEmpty()) d1 = fs2us(s1);
  if (d2.IsEmpty()) d2 = fs2us(s2);
  return true;
}


/*
// returns true, if we need additional use with New Super path.
bool GetSuperPath(CFSTR path, UString &superPath)
{
  if (GetSuperPathBase(path, superPath))
    return !superPath.IsEmpty();
  return false;
}
*/
#endif // Z7_LONG_PATH

bool GetFullPath(CFSTR dirPrefix, CFSTR s, FString &res)
{
  res = s;

  #ifdef UNDER_CE

  if (!IS_SEPAR(s[0]))
  {
    if (!dirPrefix)
      return false;
    res = dirPrefix;
    res += s;
  }

  #else

  const unsigned prefixSize = GetRootPrefixSize(s);
  if (prefixSize != 0)
#ifdef _WIN32
  if (prefixSize != 1)
#endif
  {
    if (!AreThereDotsFolders(s + prefixSize))
      return true;
    
    UString rem = fs2us(s + prefixSize);
    if (!ResolveDotsFolders(rem))
      return true; // maybe false;
    res.DeleteFrom(prefixSize);
    res += us2fs(rem);
    return true;
  }

  UString curDir;
  if (dirPrefix && prefixSize == 0)
    curDir = fs2us(dirPrefix);  // we use (dirPrefix), only if (s) path is relative
  else
  {
    if (!GetCurDir(curDir))
      return false;
  }
  NormalizeDirPathPrefix(curDir);

  unsigned fixedSize = GetRootPrefixSize(curDir);

  UString temp;
#ifdef _WIN32
  if (prefixSize != 0)
  {
    /* (s) is absolute path, but only (prefixSize == 1) is possible here.
       So for full resolving we need root of current folder and
       relative part of (s). */
    s += prefixSize;
    // (s) is relative part now
    if (fixedSize == 0)
    {
      // (curDir) is not absolute.
      // That case is unexpected, but we support it too.
      curDir.Empty();
      curDir.Add_PathSepar();
      fixedSize = 1;
      // (curDir) now is just Separ character.
      // So final (res) path later also will have Separ prefix.
    }
  }
  else
#endif // _WIN32
  {
    // (s) is relative path
    temp = curDir.Ptr(fixedSize);
    // (temp) is relative_part_of(curDir)
  }
  temp += fs2us(s);
  if (!ResolveDotsFolders(temp))
    return false;
  curDir.DeleteFrom(fixedSize);
  // (curDir) now contains only absolute prefix part
  res = us2fs(curDir);
  res += us2fs(temp);
  
  #endif // UNDER_CE

  return true;
}


bool GetFullPath(CFSTR path, FString &fullPath)
{
  return GetFullPath(NULL, path, fullPath);
}

}}}
