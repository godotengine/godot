// Windows/FileIO.cpp

#include "StdAfx.h"

#ifdef Z7_DEVICE_FILE
#include "../../C/Alloc.h"
#endif

// #include <stdio.h>

/*
#ifndef _WIN32
// for ioctl BLKGETSIZE64
#include <sys/ioctl.h>
#include <linux/fs.h>
#endif
*/

#include "FileIO.h"
#include "FileName.h"

HRESULT GetLastError_noZero_HRESULT()
{
  const DWORD res = ::GetLastError();
  if (res == 0)
    return E_FAIL;
  return HRESULT_FROM_WIN32(res);
}

#ifdef _WIN32

#ifndef _UNICODE
extern bool g_IsNT;
#endif

using namespace NWindows;
using namespace NFile;
using namespace NName;

namespace NWindows {
namespace NFile {

#ifdef Z7_DEVICE_FILE

namespace NSystem
{
bool MyGetDiskFreeSpace(CFSTR rootPath, UInt64 &clusterSize, UInt64 &totalSize, UInt64 &freeSize);
}
#endif

namespace NIO {

/*
WinXP-64 CreateFile():
  ""             -  ERROR_PATH_NOT_FOUND
  :stream        -  OK
  .:stream       -  ERROR_PATH_NOT_FOUND
  .\:stream      -  OK
  
  folder\:stream -  ERROR_INVALID_NAME
  folder:stream  -  OK

  c:\:stream     -  OK

  c::stream      -  ERROR_INVALID_NAME, if current dir is NOT ROOT ( c:\dir1 )
  c::stream      -  OK,                 if current dir is ROOT     ( c:\ )
*/

bool CFileBase::Create(CFSTR path, DWORD desiredAccess,
    DWORD shareMode, DWORD creationDisposition, DWORD flagsAndAttributes)
{
  if (!Close())
    return false;

  #ifdef Z7_DEVICE_FILE
  IsDeviceFile = false;
  #endif

  #ifndef _UNICODE
  if (!g_IsNT)
  {
    _handle = ::CreateFile(fs2fas(path), desiredAccess, shareMode,
        (LPSECURITY_ATTRIBUTES)NULL, creationDisposition, flagsAndAttributes, (HANDLE)NULL);
  }
  else
  #endif
  {
    IF_USE_MAIN_PATH
      _handle = ::CreateFileW(fs2us(path), desiredAccess, shareMode,
        (LPSECURITY_ATTRIBUTES)NULL, creationDisposition, flagsAndAttributes, (HANDLE)NULL);
    #ifdef Z7_LONG_PATH
    if (_handle == INVALID_HANDLE_VALUE && USE_SUPER_PATH)
    {
      UString superPath;
      if (GetSuperPath(path, superPath, USE_MAIN_PATH))
        _handle = ::CreateFileW(superPath, desiredAccess, shareMode,
            (LPSECURITY_ATTRIBUTES)NULL, creationDisposition, flagsAndAttributes, (HANDLE)NULL);
    }
    #endif
  }
  
  /*
  #ifndef UNDER_CE
  #ifndef Z7_SFX
  if (_handle == INVALID_HANDLE_VALUE)
  {
    // it's debug hack to open symbolic links in Windows XP and WSL links in Windows 10
    DWORD lastError = GetLastError();
    if (lastError == ERROR_CANT_ACCESS_FILE)
    {
      CByteBuffer buf;
      if (NIO::GetReparseData(path, buf, NULL))
      {
        CReparseAttr attr;
        if (attr.Parse(buf, buf.Size()))
        {
          FString dirPrefix, fileName;
          if (NDir::GetFullPathAndSplit(path, dirPrefix, fileName))
          {
            FString fullPath;
            if (GetFullPath(dirPrefix, us2fs(attr.GetPath()), fullPath))
            {
              // FIX IT: recursion levels must be restricted
              return Create(fullPath, desiredAccess,
                shareMode, creationDisposition, flagsAndAttributes);
            }
          }
        }
      }
      SetLastError(lastError);
    }
  }
  #endif
  #endif
  */

  return (_handle != INVALID_HANDLE_VALUE);
}

bool CFileBase::Close() throw()
{
  if (_handle == INVALID_HANDLE_VALUE)
    return true;
#if 0
  if (!IsStdStream)
#endif
  {
    if (!::CloseHandle(_handle))
      return false;
  }
#if 0
  IsStdStream = false;
  IsStdPipeStream = false;
#endif
  _handle = INVALID_HANDLE_VALUE;
  return true;
}

bool CFileBase::GetLength(UInt64 &length) const throw()
{
  #ifdef Z7_DEVICE_FILE
  if (IsDeviceFile && SizeDefined)
  {
    length = Size;
    return true;
  }
  #endif

  DWORD high = 0;
  const DWORD low = ::GetFileSize(_handle, &high);
  if (low == INVALID_FILE_SIZE)
    if (::GetLastError() != NO_ERROR)
      return false;
  length = (((UInt64)high) << 32) + low;
  return true;

  /*
  LARGE_INTEGER fileSize;
  // GetFileSizeEx() is unsupported in 98/ME/NT, and supported in Win2000+
  if (!GetFileSizeEx(_handle, &fileSize))
    return false;
  length = (UInt64)fileSize.QuadPart;
  return true;
  */
}


/* Specification for SetFilePointer():
   
   If a new file pointer is a negative value,
   {
     the function fails,
     the file pointer is not moved,
     the code returned by GetLastError() is ERROR_NEGATIVE_SEEK.
   }

   If the hFile handle is opened with the FILE_FLAG_NO_BUFFERING flag set
   {
     an application can move the file pointer only to sector-aligned positions.
     A sector-aligned position is a position that is a whole number multiple of
     the volume sector size.
     An application can obtain a volume sector size by calling the GetDiskFreeSpace.
   }

   It is not an error to set a file pointer to a position beyond the end of the file.
   The size of the file does not increase until you call the SetEndOfFile, WriteFile, or WriteFileEx function.

   If the return value is INVALID_SET_FILE_POINTER and if lpDistanceToMoveHigh is non-NULL,
   an application must call GetLastError to determine whether or not the function has succeeded or failed.
*/

bool CFileBase::GetPosition(UInt64 &position) const throw()
{
  LONG high = 0;
  const DWORD low = ::SetFilePointer(_handle, 0, &high, FILE_CURRENT);
  if (low == INVALID_SET_FILE_POINTER && GetLastError() != NO_ERROR)
  {
    // for error case we can set (position)  to (-1) or (0) or leave (position) unchanged.
    // position = (UInt64)(Int64)-1; // for debug
    position = 0;
    return false;
  }
  position = (((UInt64)(UInt32)high) << 32) + low;
  return true;
  // we don't want recursed GetPosition()
  // return Seek(0, FILE_CURRENT, position);
}

bool CFileBase::Seek(Int64 distanceToMove, DWORD moveMethod, UInt64 &newPosition) const throw()
{
  #ifdef Z7_DEVICE_FILE
  if (IsDeviceFile && SizeDefined && moveMethod == FILE_END)
  {
    distanceToMove += Size;
    moveMethod = FILE_BEGIN;
  }
  #endif

  LONG high = (LONG)(distanceToMove >> 32);
  const DWORD low = ::SetFilePointer(_handle, (LONG)(distanceToMove & 0xFFFFFFFF), &high, moveMethod);
  if (low == INVALID_SET_FILE_POINTER)
  {
    const DWORD lastError = ::GetLastError();
    if (lastError != NO_ERROR)
    {
      // 21.07: we set (newPosition) to real position even after error.
      GetPosition(newPosition);
      SetLastError(lastError); // restore LastError
      return false;
    }
  }
  newPosition = (((UInt64)(UInt32)high) << 32) + low;
  return true;
}

bool CFileBase::Seek(UInt64 position, UInt64 &newPosition) const throw()
{
  return Seek((Int64)position, FILE_BEGIN, newPosition);
}

bool CFileBase::SeekToBegin() const throw()
{
  UInt64 newPosition = 0;
  return Seek(0, newPosition) && (newPosition == 0);
}

bool CFileBase::SeekToEnd(UInt64 &newPosition) const throw()
{
  return Seek(0, FILE_END, newPosition);
}

// ---------- CInFile ---------

#ifdef Z7_DEVICE_FILE

void CInFile::CorrectDeviceSize()
{
  // maybe we must decrease kClusterSize to 1 << 12, if we want correct size at tail
  const UInt32 kClusterSize = 1 << 14;
  UInt64 pos = Size & ~(UInt64)(kClusterSize - 1);
  UInt64 realNewPosition;
  if (!Seek(pos, realNewPosition))
    return;
  Byte *buf = (Byte *)MidAlloc(kClusterSize);

  bool needbackward = true;

  for (;;)
  {
    UInt32 processed = 0;
    // up test is slow for "PhysicalDrive".
    // processed size for latest block for "PhysicalDrive0" is 0.
    if (!Read1(buf, kClusterSize, processed))
      break;
    if (processed == 0)
      break;
    needbackward = false;
    Size = pos + processed;
    if (processed != kClusterSize)
      break;
    pos += kClusterSize;
  }

  if (needbackward && pos != 0)
  {
    pos -= kClusterSize;
    for (;;)
    {
      // break;
      if (!Seek(pos, realNewPosition))
        break;
      if (!buf)
      {
        buf = (Byte *)MidAlloc(kClusterSize);
        if (!buf)
          break;
      }
      UInt32 processed = 0;
      // that code doesn't work for "PhysicalDrive0"
      if (!Read1(buf, kClusterSize, processed))
        break;
      if (processed != 0)
      {
        Size = pos + processed;
        break;
      }
      if (pos == 0)
        break;
      pos -= kClusterSize;
    }
  }
  MidFree(buf);
}


void CInFile::CalcDeviceSize(CFSTR s)
{
  SizeDefined = false;
  Size = 0;
  if (_handle == INVALID_HANDLE_VALUE || !IsDeviceFile)
    return;
  #ifdef UNDER_CE

  SizeDefined = true;
  Size = 128 << 20;
  
  #else
  
  PARTITION_INFORMATION partInfo;
  bool needCorrectSize = true;

  /*
    WinXP 64-bit:

    HDD \\.\PhysicalDrive0 (MBR):
      GetPartitionInfo == GeometryEx :  corrrect size? (includes tail)
      Geometry   :  smaller than GeometryEx (no tail, maybe correct too?)
      MyGetDiskFreeSpace : FAIL
      Size correction is slow and block size (kClusterSize) must be small?

    HDD partition \\.\N: (NTFS):
      MyGetDiskFreeSpace   :  Size of NTFS clusters. Same size can be calculated after correction
      GetPartitionInfo     :  size of partition data: NTFS clusters + TAIL; TAIL contains extra empty sectors and copy of first sector of NTFS
      Geometry / CdRomGeometry / GeometryEx :  size of HDD (not that partition)

    CD-ROM drive (ISO):
      MyGetDiskFreeSpace   :  correct size. Same size can be calculated after correction
      Geometry == CdRomGeometry  :  smaller than corrrect size
      GetPartitionInfo == GeometryEx :  larger than corrrect size

    Floppy \\.\a: (FAT):
      Geometry :  correct size.
      CdRomGeometry / GeometryEx / GetPartitionInfo / MyGetDiskFreeSpace - FAIL
      correction works OK for FAT.
      correction works OK for non-FAT, if kClusterSize = 512.
  */

  if (GetPartitionInfo(&partInfo))
  {
    Size = (UInt64)partInfo.PartitionLength.QuadPart;
    SizeDefined = true;
    needCorrectSize = false;
    if ((s)[0] == '\\' && (s)[1] == '\\' && (s)[2] == '.' && (s)[3] == '\\' && (s)[5] == ':' && (s)[6] == 0)
    {
      FChar path[4] = { s[4], ':', '\\', 0 };
      UInt64 clusterSize, totalSize, freeSize;
      if (NSystem::MyGetDiskFreeSpace(path, clusterSize, totalSize, freeSize))
        Size = totalSize;
      else
        needCorrectSize = true;
    }
  }
  
  if (!SizeDefined)
  {
    my_DISK_GEOMETRY_EX geomEx;
    SizeDefined = GetGeometryEx(&geomEx);
    if (SizeDefined)
      Size = (UInt64)geomEx.DiskSize.QuadPart;
    else
    {
      DISK_GEOMETRY geom;
      SizeDefined = GetGeometry(&geom);
      if (!SizeDefined)
        SizeDefined = GetCdRomGeometry(&geom);
      if (SizeDefined)
        Size = (UInt64)geom.Cylinders.QuadPart * geom.TracksPerCylinder * geom.SectorsPerTrack * geom.BytesPerSector;
    }
  }
  
  if (needCorrectSize && SizeDefined && Size != 0)
  {
    CorrectDeviceSize();
    SeekToBegin();
  }

  // SeekToBegin();
  #endif
}

// ((desiredAccess & (FILE_WRITE_DATA | FILE_APPEND_DATA | GENERIC_WRITE)) == 0 &&

#define MY_DEVICE_EXTRA_CODE \
  IsDeviceFile = IsDevicePath(fileName); \
  CalcDeviceSize(fileName);
#else
#define MY_DEVICE_EXTRA_CODE
#endif

bool CInFile::Open(CFSTR fileName, DWORD shareMode, DWORD creationDisposition, DWORD flagsAndAttributes)
{
  DWORD desiredAccess = GENERIC_READ;
  
  #ifdef _WIN32
  if (PreserveATime)
    desiredAccess |= FILE_WRITE_ATTRIBUTES;
  #endif
  
  bool res = Create(fileName, desiredAccess, shareMode, creationDisposition, flagsAndAttributes);

  #ifdef _WIN32
  if (res && PreserveATime)
  {
    FILETIME ft;
    ft.dwHighDateTime = ft.dwLowDateTime = 0xFFFFFFFF;
    ::SetFileTime(_handle, NULL, &ft, NULL);
  }
  #endif

  MY_DEVICE_EXTRA_CODE
  return res;
}

bool CInFile::OpenShared(CFSTR fileName, bool shareForWrite)
{ return Open(fileName, FILE_SHARE_READ | (shareForWrite ? FILE_SHARE_WRITE : 0), OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL); }

bool CInFile::Open(CFSTR fileName)
  { return OpenShared(fileName, false); }

// ReadFile and WriteFile functions in Windows have BUG:
// If you Read or Write 64MB or more (probably min_failure_size = 64MB - 32KB + 1)
// from/to Network file, it returns ERROR_NO_SYSTEM_RESOURCES
// (Insufficient system resources exist to complete the requested service).

// Probably in some version of Windows there are problems with other sizes:
// for 32 MB (maybe also for 16 MB).
// And message can be "Network connection was lost"

static const UInt32 kChunkSizeMax = 1 << 22;

bool CInFile::Read1(void *data, UInt32 size, UInt32 &processedSize) throw()
{
  DWORD processedLoc = 0;
  const bool res = BOOLToBool(::ReadFile(_handle, data, size, &processedLoc, NULL));
  processedSize = (UInt32)processedLoc;
  return res;
}

bool CInFile::ReadPart(void *data, UInt32 size, UInt32 &processedSize) throw()
{
#if 0
  const UInt32 chunkSizeMax = (0 || IsStdStream) ? (1 << 20) : kChunkSizeMax;
  if (size > chunkSizeMax)
      size = chunkSizeMax;
#else
  if (size > kChunkSizeMax)
      size = kChunkSizeMax;
#endif
  return Read1(data, size, processedSize);
}

bool CInFile::Read(void *data, UInt32 size, UInt32 &processedSize) throw()
{
  processedSize = 0;
  do
  {
    UInt32 processedLoc = 0;
    const bool res = ReadPart(data, size, processedLoc);
    processedSize += processedLoc;
    if (!res)
      return false;
    if (processedLoc == 0)
      return true;
    data = (void *)((Byte *)data + processedLoc);
    size -= processedLoc;
  }
  while (size);
  return true;
}

bool CInFile::ReadFull(void *data, size_t size, size_t &processedSize) throw()
{
  processedSize = 0;
  do
  {
    UInt32 processedLoc = 0;
    const UInt32 sizeLoc = (size > kChunkSizeMax ? (UInt32)kChunkSizeMax : (UInt32)size);
    const bool res = Read1(data, sizeLoc, processedLoc);
    processedSize += processedLoc;
    if (!res)
      return false;
    if (processedLoc == 0)
      return true;
    data = (void *)((Byte *)data + processedLoc);
    size -= processedLoc;
  }
  while (size);
  return true;
}

// ---------- COutFile ---------

bool COutFile::Open(CFSTR fileName, DWORD shareMode, DWORD creationDisposition, DWORD flagsAndAttributes)
  { return CFileBase::Create(fileName, GENERIC_WRITE, shareMode, creationDisposition, flagsAndAttributes); }

bool COutFile::Open_Disposition(CFSTR fileName, DWORD creationDisposition)
  { return Open(fileName, FILE_SHARE_READ, creationDisposition, FILE_ATTRIBUTE_NORMAL); }

bool COutFile::Create_ALWAYS_with_Attribs(CFSTR fileName, DWORD flagsAndAttributes)
  { return Open(fileName, FILE_SHARE_READ, CREATE_ALWAYS, flagsAndAttributes); }

bool COutFile::SetTime(const FILETIME *cTime, const FILETIME *aTime, const FILETIME *mTime) throw()
  { return BOOLToBool(::SetFileTime(_handle, cTime, aTime, mTime)); }

bool COutFile::SetMTime(const FILETIME *mTime) throw() {  return SetTime(NULL, NULL, mTime); }

bool COutFile::WritePart(const void *data, UInt32 size, UInt32 &processedSize) throw()
{
  if (size > kChunkSizeMax)
    size = kChunkSizeMax;
  DWORD processedLoc = 0;
  bool res = BOOLToBool(::WriteFile(_handle, data, size, &processedLoc, NULL));
  processedSize = (UInt32)processedLoc;
  return res;
}

bool COutFile::Write(const void *data, UInt32 size, UInt32 &processedSize) throw()
{
  processedSize = 0;
  do
  {
    UInt32 processedLoc = 0;
    const bool res = WritePart(data, size, processedLoc);
    processedSize += processedLoc;
    if (!res)
      return false;
    if (processedLoc == 0)
      return true;
    data = (const void *)((const Byte *)data + processedLoc);
    size -= processedLoc;
  }
  while (size);
  return true;
}

bool COutFile::WriteFull(const void *data, size_t size) throw()
{
  do
  {
    UInt32 processedLoc = 0;
    const UInt32 sizeCur = (size > kChunkSizeMax ? kChunkSizeMax : (UInt32)size);
    if (!WritePart(data, sizeCur, processedLoc))
      return false;
    if (processedLoc == 0)
      return (size == 0);
    data = (const void *)((const Byte *)data + processedLoc);
    size -= processedLoc;
  }
  while (size);
  return true;
}

bool COutFile::SetEndOfFile() throw() { return BOOLToBool(::SetEndOfFile(_handle)); }

bool COutFile::SetLength(UInt64 length) throw()
{
  UInt64 newPosition;
  if (!Seek(length, newPosition))
    return false;
  if (newPosition != length)
    return false;
  return SetEndOfFile();
}

bool COutFile::SetLength_KeepPosition(UInt64 length) throw()
{
  UInt64 currentPos = 0;
  if (!GetPosition(currentPos))
    return false;
  DWORD lastError = 0;
  const bool result = SetLength(length);
  if (!result)
    lastError = GetLastError();
  UInt64 currentPos2;
  const bool result2 = Seek(currentPos, currentPos2);
  if (lastError != 0)
    SetLastError(lastError);
  return (result && result2);
}

}}}

#else // _WIN32


// POSIX

#include <fcntl.h>
#include <unistd.h>

namespace NWindows {
namespace NFile {

namespace NDir {
bool SetDirTime(CFSTR path, const CFiTime *cTime, const CFiTime *aTime, const CFiTime *mTime);
}

namespace NIO {

bool CFileBase::OpenBinary(const char *name, int flags, mode_t mode)
{
  #ifdef O_BINARY
  flags |= O_BINARY;
  #endif

  Close();
  _handle = ::open(name, flags, mode);
  return _handle != -1;

  /*
  if (_handle == -1)
    return false;
  if (IsString1PrefixedByString2(name, "/dev/"))
  {
    // /dev/sda
    // IsDeviceFile = true; // for debug
    // SizeDefined = false;
    // SizeDefined = (GetDeviceSize_InBytes(Size) == 0);
  }
  return true;
  */
}

bool CFileBase::Close()
{
  if (_handle == -1)
    return true;
  if (close(_handle) != 0)
    return false;
  _handle = -1;
  /*
  IsDeviceFile = false;
  SizeDefined = false;
  */
  return true;
}

bool CFileBase::GetLength(UInt64 &length) const
{
  length = 0;
  // length = (UInt64)(Int64)-1; // for debug
  const off_t curPos = seekToCur();
  if (curPos == -1)
    return false;
  const off_t lengthTemp = seek(0, SEEK_END);
  seek(curPos, SEEK_SET);
  length = (UInt64)lengthTemp;

  /*
  // 22.00:
  if (lengthTemp == 1)
  if (IsDeviceFile && SizeDefined)
  {
    length = Size;
    return true;
  }
  */

  return (lengthTemp != -1);
}

off_t CFileBase::seek(off_t distanceToMove, int moveMethod) const
{
  /*
  if (IsDeviceFile && SizeDefined && moveMethod == SEEK_END)
  {
    printf("\n seek : IsDeviceFile moveMethod = %d distanceToMove = %ld\n", moveMethod, distanceToMove);
    distanceToMove += Size;
    moveMethod = SEEK_SET;
  }
  */

  // printf("\nCFileBase::seek() moveMethod = %d, distanceToMove = %lld", moveMethod, (long long)distanceToMove);
  // off_t res = ::lseek(_handle, distanceToMove, moveMethod);
  // printf("\n lseek : moveMethod = %d distanceToMove = %ld\n", moveMethod, distanceToMove);
  return ::lseek(_handle, distanceToMove, moveMethod);
  // return res;
}

off_t CFileBase::seekToBegin() const throw()
{
  return seek(0, SEEK_SET);
}

off_t CFileBase::seekToCur() const throw()
{
  return seek(0, SEEK_CUR);
}

/*
bool CFileBase::SeekToBegin() const throw()
{
  return (::seek(0, SEEK_SET) != -1);
}
*/


/////////////////////////
// CInFile

bool CInFile::Open(const char *name)
{
  return CFileBase::OpenBinary(name, O_RDONLY);
}

bool CInFile::OpenShared(const char *name, bool)
{
  return Open(name);
}


/*
int CFileBase::my_ioctl_BLKGETSIZE64(unsigned long long *numBlocks)
{
  // we can read "/sys/block/sda/size" "/sys/block/sda/sda1/size" - partition
  // #include <linux/fs.h>
  return ioctl(_handle, BLKGETSIZE64, numBlocks);
  // in block size
}

int CFileBase::GetDeviceSize_InBytes(UInt64 &size)
{
  size = 0;
  unsigned long long numBlocks;
  int res = my_ioctl_BLKGETSIZE64(&numBlocks);
  if (res == 0)
    size = numBlocks; // another blockSize s possible?
  printf("\nGetDeviceSize_InBytes res = %d, size = %lld\n", res, (long long)size);
  return res;
}
*/

/*
On Linux (32-bit and 64-bit):
read(), write() (and similar system calls) will transfer at most
0x7ffff000 = (2GiB - 4 KiB) bytes, returning the number of bytes actually transferred.
*/

static const size_t kChunkSizeMax = ((size_t)1 << 22);

ssize_t CInFile::read_part(void *data, size_t size) throw()
{
  if (size > kChunkSizeMax)
    size = kChunkSizeMax;
  return ::read(_handle, data, size);
}

bool CInFile::ReadFull(void *data, size_t size, size_t &processed) throw()
{
  processed = 0;
  do
  {
    const ssize_t res = read_part(data, size);
    if (res < 0)
      return false;
    if (res == 0)
      break;
    data = (void *)((Byte *)data + (size_t)res);
    processed += (size_t)res;
    size -= (size_t)res;
  }
  while (size);
  return true;
}


/////////////////////////
// COutFile

bool COutFile::OpenBinary_forWrite_oflag(const char *name, int oflag)
{
  Path = name; // change it : set it only if open is success.
  return OpenBinary(name, oflag, mode_for_Create);
}


/*
  windows           exist  non-exist  posix
  CREATE_NEW        Fail   Create     O_CREAT | O_EXCL
  CREATE_ALWAYS     Trunc  Create     O_CREAT | O_TRUNC
  OPEN_ALWAYS       Open   Create     O_CREAT
  OPEN_EXISTING     Open   Fail       0
  TRUNCATE_EXISTING Trunc  Fail       O_TRUNC ???

  // O_CREAT = If the file exists, this flag has no effect except as noted under O_EXCL below.
  // If O_CREAT and O_EXCL are set, open() shall fail if the file exists.
  // O_TRUNC : If the file exists and the file is successfully opened, its length shall be truncated to 0.
*/
bool COutFile::Open_EXISTING(const char *name)
  { return OpenBinary_forWrite_oflag(name, O_WRONLY); }
bool COutFile::Create_ALWAYS(const char *name)
  { return OpenBinary_forWrite_oflag(name, O_WRONLY | O_CREAT | O_TRUNC); }
bool COutFile::Create_NEW(const char *name)
  { return OpenBinary_forWrite_oflag(name, O_WRONLY | O_CREAT | O_EXCL);  }
bool COutFile::Create_ALWAYS_or_Open_ALWAYS(const char *name, bool createAlways)
{
  return OpenBinary_forWrite_oflag(name,
      createAlways ?
        O_WRONLY | O_CREAT | O_TRUNC :
        O_WRONLY | O_CREAT);
}
/*
bool COutFile::Create_ALWAYS_or_NEW(const char *name, bool createAlways)
{
  return OpenBinary_forWrite_oflag(name,
      createAlways ?
        O_WRONLY | O_CREAT | O_TRUNC :
        O_WRONLY | O_CREAT | O_EXCL);
}
bool COutFile::Open_Disposition(const char *name, DWORD creationDisposition)
{
  int flag;
  switch (creationDisposition)
  {
    case CREATE_NEW:        flag = O_WRONLY | O_CREAT | O_EXCL;  break;
    case CREATE_ALWAYS:     flag = O_WRONLY | O_CREAT | O_TRUNC;  break;
    case OPEN_ALWAYS:       flag = O_WRONLY | O_CREAT;  break;
    case OPEN_EXISTING:     flag = O_WRONLY;  break;
    case TRUNCATE_EXISTING: flag = O_WRONLY | O_TRUNC; break;
    default:
      SetLastError(EINVAL);
      return false;
  }
  return OpenBinary_forWrite_oflag(name, flag);
}
*/

ssize_t COutFile::write_part(const void *data, size_t size) throw()
{
  if (size > kChunkSizeMax)
    size = kChunkSizeMax;
  return ::write(_handle, data, size);
}

ssize_t COutFile::write_full(const void *data, size_t size, size_t &processed) throw()
{
  processed = 0;
  do
  {
    const ssize_t res = write_part(data, size);
    if (res < 0)
      return res;
    if (res == 0)
      break;
    data = (const void *)((const Byte *)data + (size_t)res);
    processed += (size_t)res;
    size -= (size_t)res;
  }
  while (size);
  return (ssize_t)processed;
}

bool COutFile::SetLength(UInt64 length) throw()
{
  const off_t len2 = (off_t)length;
  if ((Int64)length != len2)
  {
    SetLastError(EFBIG);
    return false;
  }
  // The value of the seek pointer shall not be modified by a call to ftruncate().
  const int iret = ftruncate(_handle, len2);
  return (iret == 0);
}

bool COutFile::Close()
{
  const bool res = CFileBase::Close();
  if (!res)
    return res;
  if (CTime_defined || ATime_defined || MTime_defined)
  {
    /* bool res2 = */ NWindows::NFile::NDir::SetDirTime(Path,
        CTime_defined ? &CTime : NULL,
        ATime_defined ? &ATime : NULL,
        MTime_defined ? &MTime : NULL);
  }
  return res;
}

bool COutFile::SetTime(const CFiTime *cTime, const CFiTime *aTime, const CFiTime *mTime) throw()
{
  // On some OS (cygwin, MacOSX ...), you must close the file before updating times
  // return true;

  if (cTime) { CTime = *cTime; CTime_defined = true; } else CTime_defined = false;
  if (aTime) { ATime = *aTime; ATime_defined = true; } else ATime_defined = false;
  if (mTime) { MTime = *mTime; MTime_defined = true; } else MTime_defined = false;
  return true;

  /*
  struct timespec times[2];
  UNUSED_VAR(cTime)
  if (!aTime && !mTime)
    return true;
  bool needChange;
  needChange  = FiTime_To_timespec(aTime, times[0]);
  needChange |= FiTime_To_timespec(mTime, times[1]);
  if (!needChange)
    return true;
  return futimens(_handle, times) == 0;
  */
}

bool COutFile::SetMTime(const CFiTime *mTime) throw()
{
  if (mTime) { MTime = *mTime; MTime_defined = true; } else MTime_defined = false;
  return true;
}

}}}


#endif
