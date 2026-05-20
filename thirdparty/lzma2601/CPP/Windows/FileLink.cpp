// Windows/FileLink.cpp

#include "StdAfx.h"

#include "../../C/CpuArch.h"

#ifndef _WIN32
#include <unistd.h>
#endif

#ifdef Z7_DEVICE_FILE
#include "../../C/Alloc.h"
#endif

#include "../Common/UTFConvert.h"
#include "../Common/StringConvert.h"

#include "FileDir.h"
#include "FileFind.h"
#include "FileIO.h"
#include "FileName.h"

#ifdef Z7_OLD_WIN_SDK
#ifndef ERROR_INVALID_REPARSE_DATA
#define ERROR_INVALID_REPARSE_DATA       4392L
#endif
#ifndef ERROR_REPARSE_TAG_INVALID
#define ERROR_REPARSE_TAG_INVALID        4393L
#endif
#endif

#ifndef _UNICODE
extern bool g_IsNT;
#endif

namespace NWindows {
namespace NFile {

using namespace NName;

/*
Win10 Junctions/SymLinks:
  - (/) slash doesn't work as path separator
  - Win10 preinstalled junctions don't use tail backslash, but tail backslashes also work.
  - double backslash works only after drive prefix "c:\\dir1\dir2\",
    and doesn't work in another places.
  - absolute path without \??\ prefix doesn't work
  - absolute path "c:" doesn't work
*/

/*
  Reparse Points (Junctions and Symbolic Links):
  struct
  {
    UInt32 Tag;
    UInt16 Size;     // not including starting 8 bytes
    UInt16 Reserved; // = 0, DOCs: // Length, in bytes, of the unparsed portion of
       // the file name pointed to by the FileName member of the associated file object.
       // This member is only valid for create operations when the I/O fails with STATUS_REPARSE.
    
    UInt16 SubstituteOffset; // offset in bytes from  start of namesChars
    UInt16 SubstituteLen;    // size in bytes, it doesn't include tailed NUL
    UInt16 PrintOffset;      // offset in bytes from  start of namesChars
    UInt16 PrintLen;         // size in bytes, it doesn't include tailed NUL
    
    [UInt32] Flags;  // for Symbolic Links only.
    
    UInt16 namesChars[]
  }

  MOUNT_POINT (Junction point):
    1) there is NUL wchar after path
    2) Default Order in table:
         Substitute Path
         Print Path
    3) pathnames can not contain dot directory names

  SYMLINK:
    1) there is no NUL wchar after path
    2) Default Order in table:
         Print Path
         Substitute Path

DOCS:
  The print name SHOULD be an informative pathname, suitable for display
  to a user, that also identifies the target of the mount point.
  Neither of these pathnames can contain dot directory names.

reparse tags, with the exception of IO_REPARSE_TAG_SYMLINK,
are processed on the server and are not processed by a client
after transmission over the wire.
Clients SHOULD treat associated reparse data as opaque data.
*/

/*
Win10 WSL2:
admin rights + sudo: it creates normal windows symbolic link.
in another cases   : it creates IO_REPARSE_TAG_LX_SYMLINK repare point.
*/

/*
static const UInt32 kReparseFlags_Alias       = (1 << 29);
static const UInt32 kReparseFlags_HighLatency = (1 << 30);
static const UInt32 kReparseFlags_Microsoft   = ((UInt32)1 << 31);

#define Z7_WIN_IO_REPARSE_TAG_HSM          (0xC0000004L)
#define Z7_WIN_IO_REPARSE_TAG_HSM2         (0x80000006L)
#define Z7_WIN_IO_REPARSE_TAG_SIS          (0x80000007L)
#define Z7_WIN_IO_REPARSE_TAG_WIM          (0x80000008L)
#define Z7_WIN_IO_REPARSE_TAG_CSV          (0x80000009L)
#define Z7_WIN_IO_REPARSE_TAG_DFS          (0x8000000AL)
#define Z7_WIN_IO_REPARSE_TAG_DFSR         (0x80000012L)
*/

#define Get16(p) GetUi16(p)
#define Get32(p) GetUi32(p)

static const char * const k_LinkPrefix = "\\??\\";
static const char * const k_LinkPrefix_UNC = "\\??\\UNC\\";
static const unsigned k_LinkPrefix_Size = 4;

static bool IsLinkPrefix(const wchar_t *s)
{
  return IsString1PrefixedByString2(s, k_LinkPrefix);
}

/*
static const char * const k_VolumePrefix = "Volume{";
static const bool IsVolumeName(const wchar_t *s)
{
  return IsString1PrefixedByString2(s, k_VolumePrefix);
}
*/

#if defined(_WIN32) && !defined(UNDER_CE)

#define Set16(p, v) SetUi16(p, v)
#define Set32(p, v) SetUi32(p, v)

static void WriteString(Byte *dest, const wchar_t *path)
{
  for (;;)
  {
    const wchar_t c = *path++;
    if (c == 0)
      return;
    Set16(dest, (UInt16)c)
    dest += 2;
  }
}

#ifdef _WIN32
void Convert_WinPath_to_WslLinuxPath(FString &s, bool convertDrivePath)
{
  if (convertDrivePath && IsDrivePath(s))
  {
    FChar c = s[0];
    c = MyCharLower_Ascii(c);
    s.DeleteFrontal(2);
    s.InsertAtFront(c);
    s.Insert(0, FTEXT("/mnt/"));
  }
  s.Replace(FCHAR_PATH_SEPARATOR, FTEXT('/'));
}
#endif


static const unsigned k_Link_Size_Limit = 1u << 16; // 16-bit field is used for size.

void FillLinkData_WslLink(CByteBuffer &dest, const wchar_t *path)
{
  // dest.Free(); // it's empty already
  // WSL probably uses Replacement Character UTF-16 0xFFFD for unsupported characters?
  AString utf;
  ConvertUnicodeToUTF8(path, utf);
  const unsigned size = 4 + utf.Len();
  if (size >= k_Link_Size_Limit)
    return;
  dest.Alloc(8 + size);
  Byte *p = dest;
  Set32(p, Z7_WIN_IO_REPARSE_TAG_LX_SYMLINK)
  // Set32(p + 4, (UInt32)size)
  Set16(p + 4, (UInt16)size)
  Set16(p + 6, 0)
  Set32(p + 8, Z7_WIN_LX_SYMLINK_VERSION_2)
  memcpy(p + 12, utf.Ptr(), utf.Len());
}


void FillLinkData_WinLink(CByteBuffer &dest, const wchar_t *path, bool isSymLink)
{
  // dest.Free(); // it's empty already
  bool isAbs = false;
  if (IS_PATH_SEPAR(path[0]))
  {
    // root paths "\dir1\path" are marked as relative
    if (IS_PATH_SEPAR(path[1]))
      isAbs = true;
  }
  else
    isAbs = IsAbsolutePath(path);
  if (!isAbs && !isSymLink)
  {
    // Win10 allows us to create relative MOUNT_POINT.
    // But relative MOUNT_POINT will not work when accessing it.
    // So we prevent useless creation of a relative MOUNT_POINT.
    return;
  }

  bool needPrintName = true;
  UString subs (path);
  if (isAbs)
  {
    const bool isSuperPath = IsSuperPath(path);
    if (!isSuperPath && NName::IsNetworkPath(us2fs(path)))
    {
      subs = k_LinkPrefix_UNC;
      subs += (path + 2);
    }
    else
    {
      if (isSuperPath)
      {
        // we remove super prefix:
        path += kSuperPathPrefixSize;
        // we want to get correct abolute path in PrintName still.
        if (!IsDrivePath(path))
          needPrintName = false; // we need "\\server\path" for print name.
      }
      subs = k_LinkPrefix;
      subs += path;
    }
  }
  const size_t len1 = subs.Len() * 2;
  size_t len2 = (size_t)MyStringLen(path) * 2;
  if (!needPrintName)
    len2 = 0;
  size_t totalNamesSize = len1 + len2;
  /* some WIM imagex software uses old scheme for symbolic links.
     so we can use old scheme for byte to byte compatibility */
  const bool newOrderScheme = isSymLink;
  // newOrderScheme = false;
  if (!newOrderScheme)
    totalNamesSize += 2 * 2; // we use NULL terminators in old scheme.

  const size_t size = 8 + 8 + (isSymLink ? 4 : 0) + totalNamesSize;
  if (size >= k_Link_Size_Limit)
    return;
  dest.Alloc(size);
  memset(dest, 0, size);
  const UInt32 tag = isSymLink ?
      Z7_WIN_IO_REPARSE_TAG_SYMLINK :
      Z7_WIN_IO_REPARSE_TAG_MOUNT_POINT;
  Byte *p = dest;
  Set32(p, tag)
  // Set32(p + 4, (UInt32)(size - 8))
  Set16(p + 4, (UInt16)(size - 8))
  Set16(p + 6, 0)
  p += 8;

  unsigned subOffs = 0;
  unsigned printOffs = 0;
  if (newOrderScheme)
    subOffs = (unsigned)len2;
  else
    printOffs = (unsigned)len1 + 2;

  Set16(p + 0, (UInt16)subOffs)
  Set16(p + 2, (UInt16)len1)
  Set16(p + 4, (UInt16)printOffs)
  Set16(p + 6, (UInt16)len2)
  p += 8;
  if (isSymLink)
  {
    const UInt32 flags = isAbs ? 0 : Z7_WIN_SYMLINK_FLAG_RELATIVE;
    Set32(p, flags)
    p += 4;
  }
  WriteString(p + subOffs, subs);
  if (needPrintName)
    WriteString(p + printOffs, path);
}

#endif // defined(_WIN32) && !defined(UNDER_CE)


static void GetString(const Byte *p, unsigned len, UString &res)
{
  wchar_t *s = res.GetBuf(len);
  unsigned i;
  for (i = 0; i < len; i++)
  {
    const wchar_t c = Get16(p + (size_t)i * 2);
    if (c == 0)
      break;
    s[i] = c;
  }
  s[i] = 0;
  res.ReleaseBuf_SetLen(i);
}


bool CReparseAttr::Parse(const Byte *p, size_t size)
{
  ErrorCode = (DWORD)ERROR_INVALID_REPARSE_DATA;
  HeaderError = true;
  TagIsUnknown = true;
  MinorError = false;

  if (size < 8)
    return false;
  Tag = Get32(p);
  if (Get16(p + 6) != 0) // padding
  {
    // DOCs: Reserved : the field SHOULD be set to 0
    // and MUST be ignored (by parser).
    // Win10 ignores it.
    MinorError = true; // optional
  }
  unsigned len = Get16(p + 4);
  p += 8;
  size -= 8;
  if (len != size)
  // if (len > size)
    return false;
  /*
  if ((type & kReparseFlags_Alias) == 0 ||
      (type & kReparseFlags_Microsoft) == 0 ||
      (type & 0xFFFF) != 3)
  */
  HeaderError = false;

  if (   Tag != Z7_WIN_IO_REPARSE_TAG_MOUNT_POINT
      && Tag != Z7_WIN_IO_REPARSE_TAG_SYMLINK
      && Tag != Z7_WIN_IO_REPARSE_TAG_LX_SYMLINK)
  {
    // for unsupported reparse points
    ErrorCode = (DWORD)ERROR_REPARSE_TAG_INVALID; // ERROR_REPARSE_TAG_MISMATCH
    // errorCode = ERROR_REPARSE_TAG_MISMATCH; // ERROR_REPARSE_TAG_INVALID
    return false;
  }

  TagIsUnknown = false;
 
  if (Tag == Z7_WIN_IO_REPARSE_TAG_LX_SYMLINK)
  {
    if (len < 4)
      return false;
    if (Get32(p) != Z7_WIN_LX_SYMLINK_VERSION_2)
      return false;
    len -= 4;
    p += 4;
    char *s = WslName.GetBuf(len);
    unsigned i;
    for (i = 0; i < len; i++)
    {
      const char c = (char)p[i];
      s[i] = c;
      if (c == 0)
        break;
    }
    s[i] = 0;
    WslName.ReleaseBuf_SetLen(i);
    MinorError = (i != len);
    ErrorCode = 0;
    return true;
  }
  
  if (len < 8)
    return false;
  const unsigned subOffs = Get16(p);
  const unsigned subLen = Get16(p + 2);
  const unsigned printOffs = Get16(p + 4);
  const unsigned printLen = Get16(p + 6);
  len -= 8;
  p += 8;

  Flags = 0;
  if (Tag == Z7_WIN_IO_REPARSE_TAG_SYMLINK)
  {
    if (len < 4)
      return false;
    Flags = Get32(p);
    len -= 4;
    p += 4;
  }

  if ((subOffs & 1) != 0 || subOffs > len || len - subOffs < subLen)
    return false;
  if ((printOffs & 1) != 0 || printOffs > len || len - printOffs < printLen)
    return false;
  GetString(p + subOffs, subLen >> 1, SubsName);
  GetString(p + printOffs, printLen >> 1, PrintName);

  ErrorCode = 0;
  return true;
}


bool CReparseShortInfo::Parse(const Byte *p, size_t size)
{
  const Byte * const start = p;
  Offset = 0;
  Size = 0;
  if (size < 8)
    return false;
  const UInt32 Tag = Get32(p);
  UInt32 len = Get16(p + 4);
  /*
  if (len + 8 > size)
    return false;
  */
  /*
  if ((type & kReparseFlags_Alias) == 0 ||
      (type & kReparseFlags_Microsoft) == 0 ||
      (type & 0xFFFF) != 3)
  */
  if (Tag != Z7_WIN_IO_REPARSE_TAG_MOUNT_POINT &&
      Tag != Z7_WIN_IO_REPARSE_TAG_SYMLINK)
    // return true;
    return false;
  /*
  if (Get16(p + 6) != 0) // padding
    return false;
  */
  p += 8;
  size -= 8;
  if (len != size) // do we need that check?
    return false;
  if (len < 8)
    return false;
  unsigned subOffs = Get16(p);
  unsigned subLen = Get16(p + 2);
  unsigned printOffs = Get16(p + 4);
  unsigned printLen = Get16(p + 6);
  len -= 8;
  p += 8;

  // UInt32 Flags = 0;
  if (Tag == Z7_WIN_IO_REPARSE_TAG_SYMLINK)
  {
    if (len < 4)
      return false;
    // Flags = Get32(p);
    len -= 4;
    p += 4;
  }

  if ((subOffs & 1) != 0 || subOffs > len || len - subOffs < subLen)
    return false;
  if ((printOffs & 1) != 0 || printOffs > len || len - printOffs < printLen)
    return false;

  Offset = (unsigned)(p - start) + subOffs;
  Size = subLen;
  return true;
}

bool CReparseAttr::IsOkNamePair() const
{
  if (IsLinkPrefix(SubsName))
  {
    if (PrintName == GetPath())
      return true;
/*
    if (!IsDrivePath(SubsName.Ptr(k_LinkPrefix_Size)))
      return PrintName.IsEmpty();
    if (wcscmp(SubsName.Ptr(k_LinkPrefix_Size), PrintName) == 0)
      return true;
*/
  }
  return wcscmp(SubsName, PrintName) == 0;
}

/*
bool CReparseAttr::IsVolume() const
{
  if (!IsLinkPrefix(SubsName))
    return false;
  return IsVolumeName(SubsName.Ptr(k_LinkPrefix_Size));
}
*/

UString CReparseAttr::GetPath() const
{
  UString s (SubsName);
  if (IsSymLink_WSL())
  {
    // if (CheckUTF8(attr.WslName)
    if (!ConvertUTF8ToUnicode(WslName, s))
      MultiByteToUnicodeString2(s, WslName);
  }
  else if (IsLinkPrefix(s))
  {
    if (IsString1PrefixedByString2_NoCase_Ascii(s.Ptr(), k_LinkPrefix_UNC))
    {
      s.DeleteFrontal(6);
      s.ReplaceOneCharAtPos(0, '\\');
    }
    else
    {
      s.ReplaceOneCharAtPos(1, '\\'); // we normalize prefix from "\??\" to "\\?\"
      if (IsDrivePath(s.Ptr(k_LinkPrefix_Size)))
        s.DeleteFrontal(k_LinkPrefix_Size);
    }
  }
  return s;
}

#ifdef Z7_DEVICE_FILE

namespace NSystem
{
bool MyGetDiskFreeSpace(CFSTR rootPath, UInt64 &clusterSize, UInt64 &totalSize, UInt64 &freeSize);
}
#endif // Z7_DEVICE_FILE

#if defined(_WIN32) && !defined(UNDER_CE)

namespace NIO {

bool GetReparseData(CFSTR path, CByteBuffer &reparseData, BY_HANDLE_FILE_INFORMATION *fileInfo)
{
  reparseData.Free();
  CInFile file;
  if (!file.OpenReparse(path))
    return false;

  if (fileInfo)
    file.GetFileInformation(fileInfo);

  const unsigned kBufSize = MAXIMUM_REPARSE_DATA_BUFFER_SIZE;
  CByteArr buf(kBufSize);
  DWORD returnedSize;
  if (!file.DeviceIoControlOut(my_FSCTL_GET_REPARSE_POINT, buf, kBufSize, &returnedSize))
    return false;
  reparseData.CopyFrom(buf, returnedSize);
  return true;
}

static bool CreatePrefixDirOfFile(CFSTR path)
{
  FString path2 (path);
  const int pos = path2.ReverseFind_PathSepar();
  if (pos < 0)
    return true;
  #ifdef _WIN32
  if (pos == 2 && path2[1] == L':')
    return true; // we don't create Disk folder;
  #endif
  path2.DeleteFrom((unsigned)pos);
  return NDir::CreateComplexDir(path2);
}


static bool OutIoReparseData(DWORD controlCode, CFSTR path, void *data, DWORD size)
{
  COutFile file;
  if (!file.Open(path,
      FILE_SHARE_WRITE,
      OPEN_EXISTING,
      FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_BACKUP_SEMANTICS))
    return false;

  DWORD returnedSize;
  return file.DeviceIoControl(controlCode, data, size, NULL, 0, &returnedSize);
}


// MOUNT_POINT (Junction Point) and LX_SYMLINK (WSL) can be written without administrator rights.
// SYMLINK requires administrator rights.
// If there is Reparse data already, it still writes new Reparse data
bool SetReparseData(CFSTR path, bool isDir, const void *data, DWORD size)
{
  NFile::NFind::CFileInfo fi;
  if (fi.Find(path))
  {
    if (fi.IsDir() != isDir)
    {
      ::SetLastError(ERROR_DIRECTORY);
      return false;
    }
  }
  else
  {
    if (isDir)
    {
      if (!NDir::CreateComplexDir(path))
        return false;
    }
    else
    {
      CreatePrefixDirOfFile(path);
      COutFile file;
      if (!file.Create_NEW(path))
        return false;
    }
  }

  return OutIoReparseData(my_FSCTL_SET_REPARSE_POINT, path, (void *)(const Byte *)(data), size);
}


bool DeleteReparseData(CFSTR path)
{
  CByteBuffer reparseData;
  if (!GetReparseData(path, reparseData, NULL))
    return false;
  /* MSDN: The tag specified in the ReparseTag member of this structure
     must match the tag of the reparse point to be deleted,
     and the ReparseDataLength member must be zero */
  #define my_REPARSE_DATA_BUFFER_HEADER_SIZE 8
  if (reparseData.Size() < my_REPARSE_DATA_BUFFER_HEADER_SIZE)
  {
    SetLastError(ERROR_INVALID_REPARSE_DATA);
    return false;
  }
  // BYTE buf[my_REPARSE_DATA_BUFFER_HEADER_SIZE];
  // memset(buf, 0, sizeof(buf));
  // memcpy(buf, reparseData, 4); // tag
  memset(reparseData + 4, 0, my_REPARSE_DATA_BUFFER_HEADER_SIZE - 4);
  return OutIoReparseData(my_FSCTL_DELETE_REPARSE_POINT, path, reparseData, my_REPARSE_DATA_BUFFER_HEADER_SIZE);
}

}

#endif //  defined(_WIN32) && !defined(UNDER_CE)


#ifndef _WIN32

namespace NIO {

bool GetReparseData(CFSTR path, CByteBuffer &reparseData)
{
  reparseData.Free();

  #define MAX_PATHNAME_LEN 1024
  char buf[MAX_PATHNAME_LEN + 2];
  const size_t request = sizeof(buf) - 1;

  // printf("\nreadlink() path = %s \n", path);
  const ssize_t size = readlink(path, buf, request);
  // there is no tail zero

  if (size < 0)
    return false;
  if ((size_t)size >= request)
  {
    SetLastError(EINVAL); // check it: ENAMETOOLONG
    return false;
  }

  // printf("\nreadlink() res = %s size = %d \n", buf, (int)size);
  reparseData.CopyFrom((const Byte *)buf, (size_t)size);
  return true;
}


/*
// If there is Reparse data already, it still writes new Reparse data
bool SetReparseData(CFSTR path, bool isDir, const void *data, DWORD size)
{
  // AString s;
  // s.SetFrom_CalcLen(data, size);
  // return (symlink(s, path) == 0);
  UNUSED_VAR(path)
  UNUSED_VAR(isDir)
  UNUSED_VAR(data)
  UNUSED_VAR(size)
  SetLastError(ENOSYS);
  return false;
}
*/

bool SetSymLink(CFSTR from, CFSTR to)
{
  // printf("\nsymlink() %s -> %s\n", from, to);
  int ir;
  // ir = unlink(path);
  // if (ir == 0)
  ir = symlink(to, from);
  return (ir == 0);
}

bool SetSymLink_UString(CFSTR from, const UString &to)
{
  AString utf;
  ConvertUnicodeToUTF8(to, utf);
  return SetSymLink(from, utf);
}

}

#endif // !_WIN32

}}
