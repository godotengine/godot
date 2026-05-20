// Windows/FileName.h

#ifndef ZIP7_INC_WINDOWS_FILE_NAME_H
#define ZIP7_INC_WINDOWS_FILE_NAME_H

#include "../Common/MyString.h"

namespace NWindows {
namespace NFile {
namespace NName {

int FindSepar(const wchar_t *s) throw();
#ifndef USE_UNICODE_FSTRING
int FindSepar(const FChar *s) throw();
#endif

void NormalizeDirPathPrefix(FString &dirPath); // ensures that it ended with '\\', if dirPath is not epmty
void NormalizeDirPathPrefix(UString &dirPath);

#ifdef _WIN32
void NormalizeDirSeparators(FString &s);
#endif

bool IsDrivePath(const wchar_t *s) throw();  // first 3 chars are drive chars like "a:\\"

bool IsAltPathPrefix(CFSTR s) throw(); /* name: */

extern const char * const kSuperPathPrefix; /* \\?\ */
const unsigned kDevicePathPrefixSize = 4;
const unsigned kSuperPathPrefixSize = 4;
const unsigned kSuperUncPathPrefixSize = kSuperPathPrefixSize + 4;

#if defined(_WIN32) && !defined(UNDER_CE)

bool IsDevicePath(CFSTR s) throw();   /* \\.\ */
bool IsSuperUncPath(CFSTR s) throw(); /* \\?\UNC\ */
bool IsNetworkPath(CFSTR s) throw();  /* \\?\UNC\ or \\SERVER */

/* GetNetworkServerPrefixSize() returns size of server prefix:
     \\?\UNC\SERVER\
     \\SERVER\
  in another cases it returns 0
*/

unsigned GetNetworkServerPrefixSize(CFSTR s) throw();

bool IsNetworkShareRootPath(CFSTR s) throw();  /* \\?\UNC\SERVER\share or \\SERVER\share or with slash */

bool IsDrivePath_SuperAllowed(CFSTR s) throw();  // first chars are drive chars like "a:\" or "\\?\a:\"
bool IsDriveRootPath_SuperAllowed(CFSTR s) throw();  // exact drive root path "a:\" or "\\?\a:\"

bool IsDrivePath2(const wchar_t *s) throw(); // first 2 chars are drive chars like "a:"
// bool IsDriveName2(const wchar_t *s) throw(); // is drive name like "a:"
bool IsSuperPath(const wchar_t *s) throw();
bool IsSuperOrDevicePath(const wchar_t *s) throw();

bool IsAltStreamPrefixWithColon(const UString &s) throw();
// returns true, if super prefix was removed
bool If_IsSuperPath_RemoveSuperPrefix(UString &s);

#ifndef USE_UNICODE_FSTRING
bool IsDrivePath2(CFSTR s) throw(); // first 2 chars are drive chars like "a:"
// bool IsDriveName2(CFSTR s) throw(); // is drive name like "a:"
bool IsDrivePath(CFSTR s) throw();
bool IsSuperPath(CFSTR s) throw();
bool IsSuperOrDevicePath(CFSTR s) throw();

/* GetRootPrefixSize() returns size of ROOT PREFIX for cases:
     \
     \\.\
     C:\
     \\?\C:\
     \\?\UNC\SERVER\Shared\
     \\SERVER\Shared\
  in another cases it returns 0
*/

unsigned GetRootPrefixSize(CFSTR s) throw();

#endif

int FindAltStreamColon(CFSTR path) throw();

#endif // _WIN32

bool IsAbsolutePath(const wchar_t *s) throw();
unsigned GetRootPrefixSize(const wchar_t *s) throw();

#ifndef _WIN32
/* GetRootPrefixSize_WINDOWS() is called in linux, but it parses path by windows rules.
   It supports only paths system (linux) slash separators (STRING_PATH_SEPARATOR),
   It doesn't parses paths with backslash (windows) separators.
   "c:/dir/file" is supported.
*/
unsigned GetRootPrefixSize_WINDOWS(const wchar_t *s) throw();
#endif

#ifdef Z7_LONG_PATH

const int kSuperPathType_UseOnlyMain = 0;
const int kSuperPathType_UseOnlySuper = 1;
const int kSuperPathType_UseMainAndSuper = 2;

int GetUseSuperPathType(CFSTR s) throw();
bool GetSuperPath(CFSTR path, UString &superPath, bool onlyIfNew);
bool GetSuperPaths(CFSTR s1, CFSTR s2, UString &d1, UString &d2, bool onlyIfNew);

#define USE_MAIN_PATH (_useSuperPathType != kSuperPathType_UseOnlySuper)
#define USE_MAIN_PATH_2 (_useSuperPathType1 != kSuperPathType_UseOnlySuper && _useSuperPathType2 != kSuperPathType_UseOnlySuper)

#define USE_SUPER_PATH (_useSuperPathType != kSuperPathType_UseOnlyMain)
#define USE_SUPER_PATH_2 (_useSuperPathType1 != kSuperPathType_UseOnlyMain || _useSuperPathType2 != kSuperPathType_UseOnlyMain)

#define IF_USE_MAIN_PATH int _useSuperPathType = GetUseSuperPathType(path); if (USE_MAIN_PATH)
#define IF_USE_MAIN_PATH_2(x1, x2) \
    int _useSuperPathType1 = GetUseSuperPathType(x1); \
    int _useSuperPathType2 = GetUseSuperPathType(x2); \
    if (USE_MAIN_PATH_2)

#else

#define IF_USE_MAIN_PATH
#define IF_USE_MAIN_PATH_2(x1, x2)

#endif // Z7_LONG_PATH

/*
  if (dirPrefix != NULL && (path) is relative)
  {
    (dirPrefix) will be used
    result (fullPath) will contain prefix part of (dirPrefix).
  }
  Current_Dir path can be used in 2 cases:
    1) if (path) is relative && dirPrefix == NULL
    2) for _WIN32: if (path) is absolute starting wuth "\"
*/
bool GetFullPath(CFSTR dirPrefix, CFSTR path, FString &fullPath);
bool GetFullPath(CFSTR path, FString &fullPath);

}}}

#endif
