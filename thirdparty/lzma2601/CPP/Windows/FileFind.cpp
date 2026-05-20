// Windows/FileFind.cpp

#include "StdAfx.h"

// #include <stdio.h>

#ifndef _WIN32
#include <fcntl.h>           /* Definition of AT_* constants */
#include "TimeUtils.h"
// for major
// #include <sys/sysmacros.h>
#endif

#include "FileFind.h"
#include "FileIO.h"
#include "FileName.h"

#ifndef _UNICODE
extern bool g_IsNT;
#endif

using namespace NWindows;
using namespace NFile;
using namespace NName;

#if defined(_WIN32) && !defined(UNDER_CE)

#if !defined(Z7_WIN32_WINNT_MIN) || Z7_WIN32_WINNT_MIN < 0x0502  // Win2003
#define Z7_USE_DYN_FindFirstStream
#endif

#ifdef Z7_USE_DYN_FindFirstStream

Z7_DIAGNOSTIC_IGNORE_CAST_FUNCTION

EXTERN_C_BEGIN

typedef enum
{
  My_FindStreamInfoStandard,
  My_FindStreamInfoMaxInfoLevel
} MY_STREAM_INFO_LEVELS;

typedef struct
{
  LARGE_INTEGER StreamSize;
  WCHAR cStreamName[MAX_PATH + 36];
} MY_WIN32_FIND_STREAM_DATA, *MY_PWIN32_FIND_STREAM_DATA;

typedef HANDLE (WINAPI *Func_FindFirstStreamW)(LPCWSTR fileName, MY_STREAM_INFO_LEVELS infoLevel,
    LPVOID findStreamData, DWORD flags);

typedef BOOL (APIENTRY *Func_FindNextStreamW)(HANDLE findStream, LPVOID findStreamData);

EXTERN_C_END

#else

#define MY_WIN32_FIND_STREAM_DATA  WIN32_FIND_STREAM_DATA
#define My_FindStreamInfoStandard  FindStreamInfoStandard

#endif
#endif // defined(_WIN32) && !defined(UNDER_CE)


namespace NWindows {
namespace NFile {


#ifdef _WIN32
#ifdef Z7_DEVICE_FILE
namespace NSystem
{
bool MyGetDiskFreeSpace(CFSTR rootPath, UInt64 &clusterSize, UInt64 &totalSize, UInt64 &freeSize);
}
#endif
#endif

namespace NFind {

/*
#ifdef _WIN32
#define MY_CLEAR_FILETIME(ft) ft.dwLowDateTime = ft.dwHighDateTime = 0;
#else
#define MY_CLEAR_FILETIME(ft) ft.tv_sec = 0;  ft.tv_nsec = 0;
#endif
*/

void CFileInfoBase::ClearBase() throw()
{
  Size = 0;
  FiTime_Clear(CTime);
  FiTime_Clear(ATime);
  FiTime_Clear(MTime);
 
 #ifdef _WIN32
  Attrib = 0;
  // ReparseTag = 0;
  IsAltStream = false;
  IsDevice = false;
 #else
  dev = 0;
  ino = 0;
  mode = 0;
  nlink = 0;
  uid = 0;
  gid = 0;
  rdev = 0;
 #endif
}


bool CFileInfoBase::SetAs_StdInFile()
{
  ClearBase();
  Size = (UInt64)(Int64)-1;
  NTime::GetCurUtc_FiTime(MTime);
  CTime = ATime = MTime;

#ifdef _WIN32

  /* in GUI mode : GetStdHandle(STD_INPUT_HANDLE) returns NULL,
     and it doesn't set LastError.  */
#if 1
  SetLastError(0);
  const HANDLE h = GetStdHandle(STD_INPUT_HANDLE);
  if (!h || h == INVALID_HANDLE_VALUE)
  {
    if (GetLastError() == 0)
      SetLastError(ERROR_INVALID_HANDLE);
    return false;
  }
  BY_HANDLE_FILE_INFORMATION info;
  if (GetFileInformationByHandle(h, &info)
      && info.dwVolumeSerialNumber)
  {
    Size = (((UInt64)info.nFileSizeHigh) << 32) + info.nFileSizeLow;
    // FileID_Low = (((UInt64)info.nFileIndexHigh) << 32) + info.nFileIndexLow;
    // NumLinks = SupportHardLinks ? info.nNumberOfLinks : 1;
    Attrib = info.dwFileAttributes;
    CTime = info.ftCreationTime;
    ATime = info.ftLastAccessTime;
    MTime = info.ftLastWriteTime;
  }
#if 0
  printf(
    "\ndwFileAttributes = %8x"
    "\nftCreationTime   = %8x"
    "\nftLastAccessTime = %8x"
    "\nftLastWriteTime  = %8x"
    "\ndwVolumeSerialNumber  = %8x"
    "\nnFileSizeHigh  = %8x"
    "\nnFileSizeLow   = %8x"
    "\nnNumberOfLinks  = %8x"
    "\nnFileIndexHigh  = %8x"
    "\nnFileIndexLow   = %8x \n",
      (unsigned)info.dwFileAttributes,
      (unsigned)info.ftCreationTime.dwHighDateTime,
      (unsigned)info.ftLastAccessTime.dwHighDateTime,
      (unsigned)info.ftLastWriteTime.dwHighDateTime,
      (unsigned)info.dwVolumeSerialNumber,
      (unsigned)info.nFileSizeHigh,
      (unsigned)info.nFileSizeLow,
      (unsigned)info.nNumberOfLinks,
      (unsigned)info.nFileIndexHigh,
      (unsigned)info.nFileIndexLow);
#endif
#endif

#else // non-Wiondow

  mode = S_IFIFO | 0777; // 0755 : 0775 : 0664 : 0644 :
#if 1
  struct stat st;
  if (fstat(0, &st) == 0)
  {
    SetFrom_stat(st);
    if (!S_ISREG(st.st_mode)
        // S_ISFIFO(st->st_mode)
        || st.st_size == 0)
    {
      Size = (UInt64)(Int64)-1;
      // mode = S_IFIFO | 0777;
    }
  }
#endif
#endif

  return true;
}

bool CFileInfo::IsDots() const throw()
{
  if (!IsDir() || Name.IsEmpty())
    return false;
  if (Name[0] != '.')
    return false;
  return Name.Len() == 1 || (Name.Len() == 2 && Name[1] == '.');
}


#ifdef _WIN32


#define WIN_FD_TO_MY_FI(fi, fd) \
  fi.Attrib = fd.dwFileAttributes; \
  fi.CTime = fd.ftCreationTime; \
  fi.ATime = fd.ftLastAccessTime; \
  fi.MTime = fd.ftLastWriteTime; \
  fi.Size = (((UInt64)fd.nFileSizeHigh) << 32) + fd.nFileSizeLow; \
  /* fi.ReparseTag = fd.dwReserved0; */ \
  fi.IsAltStream = false; \
  fi.IsDevice = false;

  /*
  #ifdef UNDER_CE
  fi.ObjectID = fd.dwOID;
  #else
  fi.ReparseTag = fd.dwReserved0;
  #endif
  */

static void Convert_WIN32_FIND_DATA_to_FileInfo(const WIN32_FIND_DATAW &fd, CFileInfo &fi)
{
  WIN_FD_TO_MY_FI(fi, fd)
  fi.Name = us2fs(fd.cFileName);
  #if defined(_WIN32) && !defined(UNDER_CE)
  // fi.ShortName = us2fs(fd.cAlternateFileName);
  #endif
}

#ifndef _UNICODE
static void Convert_WIN32_FIND_DATA_to_FileInfo(const WIN32_FIND_DATA &fd, CFileInfo &fi)
{
  WIN_FD_TO_MY_FI(fi, fd)
  fi.Name = fas2fs(fd.cFileName);
  #if defined(_WIN32) && !defined(UNDER_CE)
  // fi.ShortName = fas2fs(fd.cAlternateFileName);
  #endif
}
#endif
  
////////////////////////////////
// CFindFile

bool CFindFileBase::Close() throw()
{
  if (_handle == INVALID_HANDLE_VALUE)
    return true;
  if (!::FindClose(_handle))
    return false;
  _handle = INVALID_HANDLE_VALUE;
  return true;
}

/*
WinXP-64 FindFirstFile():
  ""      -  ERROR_PATH_NOT_FOUND
  folder\ -  ERROR_FILE_NOT_FOUND
  \       -  ERROR_FILE_NOT_FOUND
  c:\     -  ERROR_FILE_NOT_FOUND
  c:      -  ERROR_FILE_NOT_FOUND, if current dir is ROOT     ( c:\ )
  c:      -  OK,                   if current dir is NOT ROOT ( c:\folder )
  folder  -  OK

  \\               - ERROR_INVALID_NAME
  \\Server         - ERROR_INVALID_NAME
  \\Server\        - ERROR_INVALID_NAME
      
  \\Server\Share            - ERROR_BAD_NETPATH
  \\Server\Share            - ERROR_BAD_NET_NAME (Win7).
             !!! There is problem : Win7 makes some requests for "\\Server\Shar" (look in Procmon),
                 when we call it for "\\Server\Share"
                      
  \\Server\Share\           - ERROR_FILE_NOT_FOUND
  
  \\?\UNC\Server\Share      - ERROR_INVALID_NAME
  \\?\UNC\Server\Share      - ERROR_BAD_PATHNAME (Win7)
  \\?\UNC\Server\Share\     - ERROR_FILE_NOT_FOUND
  
  \\Server\Share_RootDrive  - ERROR_INVALID_NAME
  \\Server\Share_RootDrive\ - ERROR_INVALID_NAME
  
  e:\* - ERROR_FILE_NOT_FOUND, if there are no items in that root folder
  w:\* - ERROR_PATH_NOT_FOUND, if there is no such drive w:
*/

bool CFindFile::FindFirst(CFSTR path, CFileInfo &fi)
{
  if (!Close())
    return false;
  #ifndef _UNICODE
  if (!g_IsNT)
  {
    WIN32_FIND_DATAA fd;
    _handle = ::FindFirstFileA(fs2fas(path), &fd);
    if (_handle == INVALID_HANDLE_VALUE)
      return false;
    Convert_WIN32_FIND_DATA_to_FileInfo(fd, fi);
  }
  else
  #endif
  {
    WIN32_FIND_DATAW fd;

    IF_USE_MAIN_PATH
      _handle = ::FindFirstFileW(fs2us(path), &fd);
    #ifdef Z7_LONG_PATH
    if (_handle == INVALID_HANDLE_VALUE && USE_SUPER_PATH)
    {
      UString superPath;
      if (GetSuperPath(path, superPath, USE_MAIN_PATH))
        _handle = ::FindFirstFileW(superPath, &fd);
    }
    #endif
    if (_handle == INVALID_HANDLE_VALUE)
      return false;
    Convert_WIN32_FIND_DATA_to_FileInfo(fd, fi);
  }
  return true;
}

bool CFindFile::FindNext(CFileInfo &fi)
{
  #ifndef _UNICODE
  if (!g_IsNT)
  {
    WIN32_FIND_DATAA fd;
    if (!::FindNextFileA(_handle, &fd))
      return false;
    Convert_WIN32_FIND_DATA_to_FileInfo(fd, fi);
  }
  else
  #endif
  {
    WIN32_FIND_DATAW fd;
    if (!::FindNextFileW(_handle, &fd))
      return false;
    Convert_WIN32_FIND_DATA_to_FileInfo(fd, fi);
  }
  return true;
}

#if defined(_WIN32) && !defined(UNDER_CE)

////////////////////////////////
// AltStreams

#ifdef Z7_USE_DYN_FindFirstStream
static Func_FindFirstStreamW g_FindFirstStreamW;
static Func_FindNextStreamW  g_FindNextStreamW;
#define MY_FindFirstStreamW  g_FindFirstStreamW
#define MY_FindNextStreamW   g_FindNextStreamW
static struct CFindStreamLoader
{
  CFindStreamLoader()
  {
    const HMODULE hm = ::GetModuleHandleA("kernel32.dll");
       g_FindFirstStreamW = Z7_GET_PROC_ADDRESS(
    Func_FindFirstStreamW, hm,
        "FindFirstStreamW");
       g_FindNextStreamW = Z7_GET_PROC_ADDRESS(
    Func_FindNextStreamW, hm,
        "FindNextStreamW");
  }
} g_FindStreamLoader;
#else
#define MY_FindFirstStreamW  FindFirstStreamW
#define MY_FindNextStreamW   FindNextStreamW
#endif


bool CStreamInfo::IsMainStream() const throw()
{
  return StringsAreEqualNoCase_Ascii(Name, "::$DATA");
}

UString CStreamInfo::GetReducedName() const
{
  // remove ":$DATA" postfix, but keep postfix, if Name is "::$DATA"
  UString s (Name);
  if (s.Len() > 6 + 1 && StringsAreEqualNoCase_Ascii(s.RightPtr(6), ":$DATA"))
    s.DeleteFrom(s.Len() - 6);
  return s;
}

/*
UString CStreamInfo::GetReducedName2() const
{
  UString s = GetReducedName();
  if (!s.IsEmpty() && s[0] == ':')
    s.Delete(0);
  return s;
}
*/

static void Convert_WIN32_FIND_STREAM_DATA_to_StreamInfo(const MY_WIN32_FIND_STREAM_DATA &sd, CStreamInfo &si)
{
  si.Size = (UInt64)sd.StreamSize.QuadPart;
  si.Name = sd.cStreamName;
}

/*
  WinXP-64 FindFirstStream():
  ""      -  ERROR_PATH_NOT_FOUND
  folder\ -  OK
  folder  -  OK
  \       -  OK
  c:\     -  OK
  c:      -  OK, if current dir is ROOT     ( c:\ )
  c:      -  OK, if current dir is NOT ROOT ( c:\folder )
  \\Server\Share   - OK
  \\Server\Share\  - OK

  \\               - ERROR_INVALID_NAME
  \\Server         - ERROR_INVALID_NAME
  \\Server\        - ERROR_INVALID_NAME
*/

bool CFindStream::FindFirst(CFSTR path, CStreamInfo &si)
{
  if (!Close())
    return false;
#ifdef Z7_USE_DYN_FindFirstStream
  if (!g_FindFirstStreamW)
  {
    ::SetLastError(ERROR_CALL_NOT_IMPLEMENTED);
    return false;
  }
#endif
  {
    MY_WIN32_FIND_STREAM_DATA sd;
    SetLastError(0);
    IF_USE_MAIN_PATH
      _handle = MY_FindFirstStreamW(fs2us(path), My_FindStreamInfoStandard, &sd, 0);
    if (_handle == INVALID_HANDLE_VALUE)
    {
      if (::GetLastError() == ERROR_HANDLE_EOF)
        return false;
      // long name can be tricky for path like ".\dirName".
      #ifdef Z7_LONG_PATH
      if (USE_SUPER_PATH)
      {
        UString superPath;
        if (GetSuperPath(path, superPath, USE_MAIN_PATH))
          _handle = MY_FindFirstStreamW(superPath, My_FindStreamInfoStandard, &sd, 0);
      }
      #endif
    }
    if (_handle == INVALID_HANDLE_VALUE)
      return false;
    Convert_WIN32_FIND_STREAM_DATA_to_StreamInfo(sd, si);
  }
  return true;
}

bool CFindStream::FindNext(CStreamInfo &si)
{
#ifdef Z7_USE_DYN_FindFirstStream
  if (!g_FindNextStreamW)
  {
    ::SetLastError(ERROR_CALL_NOT_IMPLEMENTED);
    return false;
  }
#endif
  {
    MY_WIN32_FIND_STREAM_DATA sd;
    if (!MY_FindNextStreamW(_handle, &sd))
      return false;
    Convert_WIN32_FIND_STREAM_DATA_to_StreamInfo(sd, si);
  }
  return true;
}

bool CStreamEnumerator::Next(CStreamInfo &si, bool &found)
{
  bool res;
  if (_find.IsHandleAllocated())
    res = _find.FindNext(si);
  else
    res = _find.FindFirst(_filePath, si);
  if (res)
  {
    found = true;
    return true;
  }
  found = false;
  return (::GetLastError() == ERROR_HANDLE_EOF);
}

#endif


/*
WinXP-64 GetFileAttributes():
  If the function fails, it returns INVALID_FILE_ATTRIBUTES and use GetLastError() to get error code

  \    - OK
  C:\  - OK, if there is such drive,
  D:\  - ERROR_PATH_NOT_FOUND, if there is no such drive,

  C:\folder     - OK
  C:\folder\    - OK
  C:\folderBad  - ERROR_FILE_NOT_FOUND

  \\Server\BadShare  - ERROR_BAD_NETPATH
  \\Server\Share     - WORKS OK, but MSDN says:
                          GetFileAttributes for a network share, the function fails, and GetLastError
                          returns ERROR_BAD_NETPATH. You must specify a path to a subfolder on that share.
*/

DWORD GetFileAttrib(CFSTR path)
{
  #ifndef _UNICODE
  if (!g_IsNT)
    return ::GetFileAttributes(fs2fas(path));
  else
  #endif
  {
    IF_USE_MAIN_PATH
    {
      DWORD dw = ::GetFileAttributesW(fs2us(path));
      if (dw != INVALID_FILE_ATTRIBUTES)
        return dw;
    }
    #ifdef Z7_LONG_PATH
    if (USE_SUPER_PATH)
    {
      UString superPath;
      if (GetSuperPath(path, superPath, USE_MAIN_PATH))
        return ::GetFileAttributesW(superPath);
    }
    #endif
    return INVALID_FILE_ATTRIBUTES;
  }
}

/* if path is "c:" or "c::" then CFileInfo::Find() returns name of current folder for that disk
   so instead of absolute path we have relative path in Name. That is not good in some calls */

/* In CFileInfo::Find() we want to support same names for alt streams as in CreateFile(). */

/* CFileInfo::Find()
We alow the following paths (as FindFirstFile):
  C:\folder
  c:                      - if current dir is NOT ROOT ( c:\folder )

also we support paths that are not supported by FindFirstFile:
  \
  \\.\c:
  c:\                     - Name will be without tail slash ( c: )
  \\?\c:\                 - Name will be without tail slash ( c: )
  \\Server\Share
  \\?\UNC\Server\Share

  c:\folder:stream  - Name = folder:stream
  c:\:stream        - Name = :stream
  c::stream         - Name = c::stream
*/

bool CFileInfo::Find(CFSTR path, bool followLink)
{
  #ifdef Z7_DEVICE_FILE
  
  if (IS_PATH_SEPAR(path[0]) &&
      IS_PATH_SEPAR(path[1]) &&
      path[2] == '.' &&
      path[3] == 0)
  {
    // 22.00 : it's virtual directory for devices
    // IsDevice = true;
    ClearBase();
    Name = path + 2;
    Attrib = FILE_ATTRIBUTE_DIRECTORY;
    return true;
  }
  
  if (IsDevicePath(path))
  {
    ClearBase();
    Name = path + 4;
    IsDevice = true;
    
    if (NName::IsDrivePath2(path + 4) && path[6] == 0)
    {
      FChar drive[4] = { path[4], ':', '\\', 0 };
      UInt64 clusterSize, totalSize, freeSize;
      if (NSystem::MyGetDiskFreeSpace(drive, clusterSize, totalSize, freeSize))
      {
        Size = totalSize;
        return true;
      }
    }

    NIO::CInFile inFile;
    // ::OutputDebugStringW(path);
    if (!inFile.Open(path))
      return false;
    // ::OutputDebugStringW(L"---");
    if (inFile.SizeDefined)
      Size = inFile.Size;
    return true;
  }
  #endif

  #if defined(_WIN32) && !defined(UNDER_CE)

  const int colonPos = FindAltStreamColon(path);
  if (colonPos >= 0 && path[(unsigned)colonPos + 1] != 0)
  {
    UString streamName = fs2us(path + (unsigned)colonPos);
    FString filePath (path);
    filePath.DeleteFrom((unsigned)colonPos);
    /* we allow both cases:
      name:stream
      name:stream:$DATA
    */
    const unsigned kPostfixSize = 6;
    if (streamName.Len() <= kPostfixSize
        || !StringsAreEqualNoCase_Ascii(streamName.RightPtr(kPostfixSize), ":$DATA"))
      streamName += ":$DATA";

    bool isOk = true;
    
    if (IsDrivePath2(filePath) &&
        (colonPos == 2 || (colonPos == 3 && filePath[2] == '\\')))
    {
      // FindFirstFile doesn't work for "c:\" and for "c:" (if current dir is ROOT)
      ClearBase();
      Name.Empty();
      if (colonPos == 2)
        Name = filePath;
    }
    else
      isOk = Find(filePath, followLink); // check it (followLink)

    if (isOk)
    {
      Attrib &= ~(DWORD)(FILE_ATTRIBUTE_DIRECTORY | FILE_ATTRIBUTE_REPARSE_POINT);
      Size = 0;
      CStreamEnumerator enumerator(filePath);
      for (;;)
      {
        CStreamInfo si;
        bool found;
        if (!enumerator.Next(si, found))
          return false;
        if (!found)
        {
          ::SetLastError(ERROR_FILE_NOT_FOUND);
          return false;
        }
        if (si.Name.IsEqualTo_NoCase(streamName))
        {
          // we delete postfix, if alt stream name is not "::$DATA"
          if (si.Name.Len() > kPostfixSize + 1)
            si.Name.DeleteFrom(si.Name.Len() - kPostfixSize);
          Name += us2fs(si.Name);
          Size = si.Size;
          IsAltStream = true;
          return true;
        }
      }
    }
  }
  
  #endif

  CFindFile finder;

  #if defined(_WIN32) && !defined(UNDER_CE)
  {
    /*
    DWORD lastError = GetLastError();
    if (lastError == ERROR_FILE_NOT_FOUND
        || lastError == ERROR_BAD_NETPATH  // XP64: "\\Server\Share"
        || lastError == ERROR_BAD_NET_NAME // Win7: "\\Server\Share"
        || lastError == ERROR_INVALID_NAME // XP64: "\\?\UNC\Server\Share"
        || lastError == ERROR_BAD_PATHNAME // Win7: "\\?\UNC\Server\Share"
        )
    */
    
    unsigned rootSize = 0;
    if (IsSuperPath(path))
      rootSize = kSuperPathPrefixSize;
    
    if (NName::IsDrivePath(path + rootSize) && path[rootSize + 3] == 0)
    {
      DWORD attrib = GetFileAttrib(path);
      if (attrib != INVALID_FILE_ATTRIBUTES && (attrib & FILE_ATTRIBUTE_DIRECTORY) != 0)
      {
        ClearBase();
        Attrib = attrib;
        Name = path + rootSize;
        Name.DeleteFrom(2);
        if (!Fill_From_ByHandleFileInfo(path))
        {
        }
        return true;
      }
    }
    else if (IS_PATH_SEPAR(path[0]))
    {
      if (path[1] == 0)
      {
        DWORD attrib = GetFileAttrib(path);
        if (attrib != INVALID_FILE_ATTRIBUTES && (attrib & FILE_ATTRIBUTE_DIRECTORY) != 0)
        {
          ClearBase();
          Name.Empty();
          Attrib = attrib;
          return true;
        }
      }
      else
      {
        const unsigned prefixSize = GetNetworkServerPrefixSize(path);
        if (prefixSize > 0 && path[prefixSize] != 0)
        {
          if (NName::FindSepar(path + prefixSize) < 0)
          {
            if (Fill_From_ByHandleFileInfo(path))
            {
              Name = path + prefixSize;
              return true;
            }

            FString s (path);
            s.Add_PathSepar();
            s.Add_Char('*'); // CHAR_ANY_MASK
            bool isOK = false;
            if (finder.FindFirst(s, *this))
            {
              if (Name.IsEqualTo("."))
              {
                Name = path + prefixSize;
                return true;
              }
              isOK = true;
              /* if "\\server\share" maps to root folder "d:\", there is no "." item.
                 But it's possible that there are another items */
            }
            {
              const DWORD attrib = GetFileAttrib(path);
              if (isOK || (attrib != INVALID_FILE_ATTRIBUTES && (attrib & FILE_ATTRIBUTE_DIRECTORY) != 0))
              {
                ClearBase();
                if (attrib != INVALID_FILE_ATTRIBUTES)
                  Attrib = attrib;
                else
                  SetAsDir();
                Name = path + prefixSize;
                return true;
              }
            }
            // ::SetLastError(lastError);
          }
        }
      }
    }
  }
  #endif

  bool res = finder.FindFirst(path, *this);
  if (!followLink
      || !res
      || !HasReparsePoint())
    return res;

  // return FollowReparse(path, IsDir());
  return Fill_From_ByHandleFileInfo(path);
/*
  // Fill_From_ByHandleFileInfo returns false (with Access Denied error),
  // if there is reparse link file (not directory reparse item).
  if (Fill_From_ByHandleFileInfo(path))
    return true;
  return HasReparsePoint();
*/
}

bool CFileInfoBase::Fill_From_ByHandleFileInfo(CFSTR path)
{
  BY_HANDLE_FILE_INFORMATION info;
  if (!NIO::CFileBase::GetFileInformation(path, &info))
    return false;
  {
    Size = (((UInt64)info.nFileSizeHigh) << 32) + info.nFileSizeLow;
    CTime = info.ftCreationTime;
    ATime = info.ftLastAccessTime;
    MTime = info.ftLastWriteTime;
    Attrib = info.dwFileAttributes;
    return true;
  }
}

/*
bool CFileInfo::FollowReparse(CFSTR path, bool isDir)
{
  if (isDir)
  {
    FString prefix = path;
    prefix.Add_PathSepar();

    // "folder/." refers to folder itself. So we can't use that path
    // we must use enumerator and search "." item
    CEnumerator enumerator;
    enumerator.SetDirPrefix(prefix);
    for (;;)
    {
      CFileInfo fi;
      if (!enumerator.NextAny(fi))
        break;
      if (fi.Name.IsEqualTo_Ascii_NoCase("."))
      {
        // we can copy preperies;
        CTime = fi.CTime;
        ATime = fi.ATime;
        MTime = fi.MTime;
        Attrib = fi.Attrib;
        Size = fi.Size;
        return true;
      }
      break;
    }
    // LastError(lastError);
    return false;
  }

  {
    NIO::CInFile inFile;
    if (inFile.Open(path))
    {
      BY_HANDLE_FILE_INFORMATION info;
      if (inFile.GetFileInformation(&info))
      {
        ClearBase();
        Size = (((UInt64)info.nFileSizeHigh) << 32) + info.nFileSizeLow;
        CTime = info.ftCreationTime;
        ATime = info.ftLastAccessTime;
        MTime = info.ftLastWriteTime;
        Attrib = info.dwFileAttributes;
        return true;
      }
    }
    return false;
  }
}
*/

bool DoesFileExist_Raw(CFSTR name)
{
  CFileInfo fi;
  return fi.Find(name) && !fi.IsDir();
}

bool DoesFileExist_FollowLink(CFSTR name)
{
  CFileInfo fi;
  return fi.Find_FollowLink(name) && !fi.IsDir();
}

bool DoesDirExist(CFSTR name, bool followLink)
{
  CFileInfo fi;
  return fi.Find(name, followLink) && fi.IsDir();
}

bool DoesFileOrDirExist(CFSTR name)
{
  CFileInfo fi;
  return fi.Find(name);
}


void CEnumerator::SetDirPrefix(const FString &dirPrefix)
{
  _wildcard = dirPrefix;
  _wildcard.Add_Char('*');
}

bool CEnumerator::NextAny(CFileInfo &fi)
{
  if (_findFile.IsHandleAllocated())
    return _findFile.FindNext(fi);
  else
    return _findFile.FindFirst(_wildcard, fi);
}

bool CEnumerator::Next(CFileInfo &fi)
{
  for (;;)
  {
    if (!NextAny(fi))
      return false;
    if (!fi.IsDots())
      return true;
  }
}

bool CEnumerator::Next(CFileInfo &fi, bool &found)
{
  /*
  for (;;)
  {
    if (!NextAny(fi))
      break;
    if (!fi.IsDots())
    {
      found = true;
      return true;
    }
  }
  */

  if (Next(fi))
  {
    found = true;
    return true;
  }

  found = false;
  DWORD lastError = ::GetLastError();
  if (_findFile.IsHandleAllocated())
    return (lastError == ERROR_NO_MORE_FILES);
  // we support the case for empty root folder: FindFirstFile("c:\\*") returns ERROR_FILE_NOT_FOUND
  if (lastError == ERROR_FILE_NOT_FOUND)
    return true;
  if (lastError == ERROR_ACCESS_DENIED)
  {
    // here we show inaccessible root system folder as empty folder to eliminate redundant user warnings
    const char *s = "System Volume Information" STRING_PATH_SEPARATOR "*";
    const int len = (int)strlen(s);
    const int delta = (int)_wildcard.Len() - len;
    if (delta == 0 || (delta > 0 && IS_PATH_SEPAR(_wildcard[(unsigned)delta - 1])))
      if (StringsAreEqual_Ascii(_wildcard.Ptr((unsigned)delta), s))
        return true;
  }
  return false;
}


////////////////////////////////
// CFindChangeNotification
// FindFirstChangeNotification can return 0. MSDN doesn't tell about it.

bool CFindChangeNotification::Close() throw()
{
  if (!IsHandleAllocated())
    return true;
  if (!::FindCloseChangeNotification(_handle))
    return false;
  _handle = INVALID_HANDLE_VALUE;
  return true;
}
           
HANDLE CFindChangeNotification::FindFirst(CFSTR path, bool watchSubtree, DWORD notifyFilter)
{
  #ifndef _UNICODE
  if (!g_IsNT)
    _handle = ::FindFirstChangeNotification(fs2fas(path), BoolToBOOL(watchSubtree), notifyFilter);
  else
  #endif
  {
    IF_USE_MAIN_PATH
    _handle = ::FindFirstChangeNotificationW(fs2us(path), BoolToBOOL(watchSubtree), notifyFilter);
    #ifdef Z7_LONG_PATH
    if (!IsHandleAllocated())
    {
      UString superPath;
      if (GetSuperPath(path, superPath, USE_MAIN_PATH))
        _handle = ::FindFirstChangeNotificationW(superPath, BoolToBOOL(watchSubtree), notifyFilter);
    }
    #endif
  }
  return _handle;
}

#ifndef UNDER_CE

bool MyGetLogicalDriveStrings(CObjectVector<FString> &driveStrings)
{
  driveStrings.Clear();
  #ifndef _UNICODE
  if (!g_IsNT)
  {
    driveStrings.Clear();
    UINT32 size = GetLogicalDriveStrings(0, NULL);
    if (size == 0)
      return false;
    CObjArray<char> buf(size);
    UINT32 newSize = GetLogicalDriveStrings(size, buf);
    if (newSize == 0 || newSize > size)
      return false;
    AString s;
    UINT32 prev = 0;
    for (UINT32 i = 0; i < newSize; i++)
    {
      if (buf[i] == 0)
      {
        s = buf + prev;
        prev = i + 1;
        driveStrings.Add(fas2fs(s));
      }
    }
    return prev == newSize;
  }
  else
  #endif
  {
    UINT32 size = GetLogicalDriveStringsW(0, NULL);
    if (size == 0)
      return false;
    CObjArray<wchar_t> buf(size);
    UINT32 newSize = GetLogicalDriveStringsW(size, buf);
    if (newSize == 0 || newSize > size)
      return false;
    UString s;
    UINT32 prev = 0;
    for (UINT32 i = 0; i < newSize; i++)
    {
      if (buf[i] == 0)
      {
        s = buf + prev;
        prev = i + 1;
        driveStrings.Add(us2fs(s));
      }
    }
    return prev == newSize;
  }
}

#endif // UNDER_CE



#else // _WIN32

// ---------- POSIX ----------

static int MY_lstat(CFSTR path, struct stat *st, bool followLink)
{
  memset(st, 0, sizeof(*st));
  int res;
  // #ifdef ENV_HAVE_LSTAT
  if (/* global_use_lstat && */ !followLink)
  {
    // printf("\nlstat\n");
    res = lstat(path, st);
  }
  else
  // #endif
  {
    // printf("\nstat\n");
    res = stat(path, st);
  }
#if 0
#if defined(__clang__) && __clang_major__ >= 14
  #pragma GCC diagnostic ignored "-Wc++98-compat-pedantic"
#endif

  printf("\n st_dev = %lld", (long long)(st->st_dev));
  printf("\n st_ino = %lld", (long long)(st->st_ino));
  printf("\n st_mode = %llx", (long long)(st->st_mode));
  printf("\n st_nlink = %lld", (long long)(st->st_nlink));
  printf("\n st_uid = %lld", (long long)(st->st_uid));
  printf("\n st_gid = %lld", (long long)(st->st_gid));
  printf("\n st_size = %lld", (long long)(st->st_size));
  printf("\n st_blksize = %lld", (long long)(st->st_blksize));
  printf("\n st_blocks = %lld", (long long)(st->st_blocks));
  printf("\n st_ctim = %lld", (long long)(ST_CTIME((*st)).tv_sec));
  printf("\n st_mtim = %lld", (long long)(ST_MTIME((*st)).tv_sec));
  printf("\n st_atim = %lld", (long long)(ST_ATIME((*st)).tv_sec));
     printf(S_ISFIFO(st->st_mode) ? "\n FIFO" : "\n NO FIFO");
  printf("\n");
#endif

  return res;
}


static const char *Get_Name_from_Path(CFSTR path) throw()
{
  size_t len = strlen(path);
  if (len == 0)
    return path;
  const char *p = path + len - 1;
  {
    if (p == path)
      return path;
    p--;
  }
  for (;;)
  {
    char c = *p;
    if (IS_PATH_SEPAR(c))
      return p + 1;
    if (p == path)
      return path;
    p--;
  }
}


UInt32 Get_WinAttribPosix_From_PosixMode(UInt32 mode)
{
  UInt32 attrib = S_ISDIR(mode) ?
      FILE_ATTRIBUTE_DIRECTORY :
      FILE_ATTRIBUTE_ARCHIVE;
  if ((mode & 0222) == 0) // S_IWUSR in p7zip
    attrib |= FILE_ATTRIBUTE_READONLY;
  return attrib | FILE_ATTRIBUTE_UNIX_EXTENSION | ((mode & 0xFFFF) << 16);
}

/*
UInt32 Get_WinAttrib_From_stat(const struct stat &st)
{
  UInt32 attrib = S_ISDIR(st.st_mode) ?
    FILE_ATTRIBUTE_DIRECTORY :
    FILE_ATTRIBUTE_ARCHIVE;

  if ((st.st_mode & 0222) == 0) // check it !!!
    attrib |= FILE_ATTRIBUTE_READONLY;

  attrib |= FILE_ATTRIBUTE_UNIX_EXTENSION + ((st.st_mode & 0xFFFF) << 16);
  return attrib;
}
*/

void CFileInfoBase::SetFrom_stat(const struct stat &st)
{
  // IsDevice = false;

  if (S_ISDIR(st.st_mode))
  {
    Size = 0;
  }
  else
  {
    Size = (UInt64)st.st_size; // for a symbolic link, size = size of filename
  }

  // Attrib = Get_WinAttribPosix_From_PosixMode(st.st_mode);

  // NTime::UnixTimeToFileTime(st.st_ctime, CTime);
  // NTime::UnixTimeToFileTime(st.st_mtime, MTime);
  // NTime::UnixTimeToFileTime(st.st_atime, ATime);
  #ifdef __APPLE__
  // #ifdef _DARWIN_FEATURE_64_BIT_INODE
  /*
    here we can use birthtime instead of st_ctimespec.
    but we use st_ctimespec for compatibility with previous versions and p7zip.
    st_birthtimespec in OSX
    st_birthtim : at FreeBSD, NetBSD
  */
  // timespec_To_FILETIME(st.st_birthtimespec, CTime);
  // #else
  // timespec_To_FILETIME(st.st_ctimespec, CTime);
  // #endif
  // timespec_To_FILETIME(st.st_mtimespec, MTime);
  // timespec_To_FILETIME(st.st_atimespec, ATime);
  CTime = st.st_ctimespec;
  MTime = st.st_mtimespec;
  ATime = st.st_atimespec;

  #elif defined(__QNXNTO__) && defined(__ARM__) && !defined(__aarch64__)
  
  // CTime = ST_CTIME(st);
  // MTime = ST_MTIME(st);
  // ATime = ST_ATIME(st);
  CTime.tv_sec = st.st_ctime;  CTime.tv_nsec = 0;
  MTime.tv_sec = st.st_mtime;  MTime.tv_nsec = 0;
  ATime.tv_sec = st.st_atime;  ATime.tv_nsec = 0;

  #else
  // timespec_To_FILETIME(st.st_ctim, CTime, &CTime_ns100);
  // timespec_To_FILETIME(st.st_mtim, MTime, &MTime_ns100);
  // timespec_To_FILETIME(st.st_atim, ATime, &ATime_ns100);
  CTime = st.st_ctim;
  MTime = st.st_mtim;
  ATime = st.st_atim;

  #endif

  dev = st.st_dev;
  ino = st.st_ino;
  mode = st.st_mode;
  nlink = st.st_nlink;
  uid = st.st_uid;
  gid = st.st_gid;
  rdev = st.st_rdev;

  /*
  printf("\n sizeof timespec = %d", (int)sizeof(timespec));
  printf("\n sizeof st_rdev = %d", (int)sizeof(rdev));
  printf("\n sizeof st_ino = %d", (int)sizeof(ino));
  printf("\n sizeof mode_t = %d", (int)sizeof(mode_t));
  printf("\n sizeof nlink_t = %d", (int)sizeof(nlink_t));
  printf("\n sizeof uid_t = %d", (int)sizeof(uid_t));
  printf("\n");
  */
  /*
  printf("\n st_rdev = %llx", (long long)rdev);
  printf("\n st_dev  = %llx", (long long)dev);
  printf("\n dev  : major = %5x minor = %5x", (unsigned)major(dev), (unsigned)minor(dev));
  printf("\n st_ino = %lld", (long long)(ino));
  printf("\n rdev : major = %5x minor = %5x", (unsigned)major(rdev), (unsigned)minor(rdev));
  printf("\n size = %lld \n", (long long)(Size));
  printf("\n");
  */
}

/*
int Uid_To_Uname(uid_t uid, AString &name)
{
  name.Empty();
  struct passwd *passwd;

  if (uid != 0 && uid == cached_no_such_uid)
    {
      *uname = xstrdup ("");
      return;
    }

  if (!cached_uname || uid != cached_uid)
    {
      passwd = getpwuid (uid);
      if (passwd)
  {
    cached_uid = uid;
    assign_string (&cached_uname, passwd->pw_name);
  }
      else
  {
    cached_no_such_uid = uid;
    *uname = xstrdup ("");
    return;
  }
    }
  *uname = xstrdup (cached_uname);
}
*/

bool CFileInfo::Find_DontFill_Name(CFSTR path, bool followLink)
{
  struct stat st;
  if (MY_lstat(path, &st, followLink) != 0)
    return false;
  // printf("\nFind_DontFill_Name : name=%s\n", path);
  SetFrom_stat(st);
  return true;
}


bool CFileInfo::Find(CFSTR path, bool followLink)
{
  // printf("\nCEnumerator::Find() name = %s\n", path);
  if (!Find_DontFill_Name(path, followLink))
    return false;

  // printf("\nOK\n");

  Name = Get_Name_from_Path(path);
  if (!Name.IsEmpty())
  {
    char c = Name.Back();
    if (IS_PATH_SEPAR(c))
      Name.DeleteBack();
  }
  return true;
}


bool DoesFileExist_Raw(CFSTR name)
{
  // FIXME for symbolic links.
  struct stat st;
  if (MY_lstat(name, &st, false) != 0)
    return false;
  return !S_ISDIR(st.st_mode);
}

bool DoesFileExist_FollowLink(CFSTR name)
{
  // FIXME for symbolic links.
  struct stat st;
  if (MY_lstat(name, &st, true) != 0)
    return false;
  return !S_ISDIR(st.st_mode);
}

bool DoesDirExist(CFSTR name, bool followLink)
{
  struct stat st;
  if (MY_lstat(name, &st, followLink) != 0)
    return false;
  return S_ISDIR(st.st_mode);
}

bool DoesFileOrDirExist(CFSTR name)
{
  struct stat st;
  if (MY_lstat(name, &st, false) != 0)
    return false;
  return true;
}


CEnumerator::~CEnumerator()
{
  if (_dir)
    closedir(_dir);
}

void CEnumerator::SetDirPrefix(const FString &dirPrefix)
{
  _wildcard = dirPrefix;
}

bool CDirEntry::IsDots() const throw()
{
  /* some systems (like CentOS 7.x on XFS) have (Type == DT_UNKNOWN)
     we can call fstatat() for that case, but we use only (Name) check here */

#if !defined(_AIX) && !defined(__sun) && !defined(__QNXNTO__)
  if (Type != DT_DIR && Type != DT_UNKNOWN)
    return false;
#endif

  return Name.Len() != 0
      && Name.Len() <= 2
      && Name[0] == '.'
      && (Name.Len() == 1 || Name[1] == '.');
}


bool CEnumerator::NextAny(CDirEntry &fi, bool &found)
{
  found = false;

  if (!_dir)
  {
    const char *w = "./";
    if (!_wildcard.IsEmpty())
      w = _wildcard.Ptr();
    _dir = ::opendir((const char *)w);
    if (_dir == NULL)
      return false;
  }

  // To distinguish end of stream from an error, we must set errno to zero before readdir()
  errno = 0;

  struct dirent *de = readdir(_dir);
  if (!de)
  {
    if (errno == 0)
      return true; // it's end of stream, and we report it with (found = false)
    // it's real error
    return false;
  }

  fi.iNode = de->d_ino;
  
#if !defined(_AIX) && !defined(__sun) && !defined(__QNXNTO__)
  fi.Type = de->d_type;
  /* some systems (like CentOS 7.x on XFS) have (Type == DT_UNKNOWN)
     we can set (Type) from fstatat() in that case.
     But (Type) is not too important. So we don't set it here with slow fstatat() */
  /*
  // fi.Type = DT_UNKNOWN; // for debug
  if (fi.Type == DT_UNKNOWN)
  {
    struct stat st;
    if (fstatat(dirfd(_dir), de->d_name, &st, AT_SYMLINK_NOFOLLOW) == 0)
      if (S_ISDIR(st.st_mode))
        fi.Type = DT_DIR;
  }
  */
#endif
  
  /*
  if (de->d_type == DT_DIR)
    fi.Attrib = FILE_ATTRIBUTE_DIRECTORY | FILE_ATTRIBUTE_UNIX_EXTENSION | ((UInt32)(S_IFDIR) << 16);
  else if (de->d_type < 16)
    fi.Attrib = FILE_ATTRIBUTE_UNIX_EXTENSION | ((UInt32)(de->d_type) << (16 + 12));
  */
  fi.Name = de->d_name;

  /*
  printf("\nCEnumerator::NextAny; len = %d %s \n", (int)fi.Name.Len(), fi.Name.Ptr());
  for (unsigned i = 0; i < fi.Name.Len(); i++)
    printf (" %02x", (unsigned)(Byte)de->d_name[i]);
  printf("\n");
  */

  found = true;
  return true;
}


bool CEnumerator::Next(CDirEntry &fi, bool &found)
{
  // printf("\nCEnumerator::Next()\n");
  // PrintName("Next", "");
  for (;;)
  {
    if (!NextAny(fi, found))
      return false;
    if (!found)
      return true;
    if (!fi.IsDots())
    {
      /*
      if (!NeedFullStat)
        return true;
      // we silently skip error file here - it can be wrong link item
      if (fi.Find_DontFill_Name(path))
        return true;
      */
      return true;
    }
  }
}

/*
bool CEnumerator::Next(CDirEntry &fileInfo, bool &found)
{
  bool found;
  if (!Next(fi, found))
    return false;
  return found;
}
*/

bool CEnumerator::Fill_FileInfo(const CDirEntry &de, CFileInfo &fileInfo, bool followLink) const
{
  // printf("\nCEnumerator::Fill_FileInfo()\n");
  struct stat st;
  // probably it's OK to use fstatat() even if it changes file position dirfd(_dir)
  int res = fstatat(dirfd(_dir), de.Name, &st, followLink ? 0 : AT_SYMLINK_NOFOLLOW);
  // if fstatat() is not supported, we can use stat() / lstat()
  
  /*
  const FString path = _wildcard + s;
  int res = MY_lstat(path, &st, followLink);
  */
  
  if (res != 0)
    return false;
  // printf("\nname=%s\n", de.Name.Ptr());
  fileInfo.SetFrom_stat(st);
  fileInfo.Name = de.Name;
  return true;
}

#endif // _WIN32

}}}
