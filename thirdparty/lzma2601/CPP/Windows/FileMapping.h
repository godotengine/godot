// Windows/FileMapping.h

#ifndef ZIP7_INC_WINDOWS_FILE_MAPPING_H
#define ZIP7_INC_WINDOWS_FILE_MAPPING_H

#include "../Common/MyTypes.h"

#include "Handle.h"

namespace NWindows {

class CFileMapping: public CHandle
{
public:
  WRes Create(DWORD protect, UInt64 maxSize, LPCTSTR name)
  {
    _handle = ::CreateFileMapping(INVALID_HANDLE_VALUE, NULL, protect, (DWORD)(maxSize >> 32), (DWORD)maxSize, name);
    return ::GetLastError();
  }

  WRes Open(DWORD
      #ifndef UNDER_CE
      desiredAccess
      #endif
      , LPCTSTR name)
  {
    #ifdef UNDER_CE
    WRes res = Create(PAGE_READONLY, 0, name);
    if (res == ERROR_ALREADY_EXISTS)
      return 0;
    Close();
    if (res == 0)
      res = ERROR_FILE_NOT_FOUND;
    return res;
    #else
    _handle = ::OpenFileMapping(desiredAccess, FALSE, name);
    if (_handle != NULL)
      return 0;
    return ::GetLastError();
    #endif
  }

  LPVOID Map(DWORD desiredAccess, UInt64 fileOffset, SIZE_T numberOfBytesToMap)
  {
    return ::MapViewOfFile(_handle, desiredAccess, (DWORD)(fileOffset >> 32), (DWORD)fileOffset, numberOfBytesToMap);
  }

  #ifndef UNDER_CE
  LPVOID Map(DWORD desiredAccess, UInt64 fileOffset, SIZE_T numberOfBytesToMap, LPVOID baseAddress)
  {
    return ::MapViewOfFileEx(_handle, desiredAccess, (DWORD)(fileOffset >> 32), (DWORD)fileOffset, numberOfBytesToMap, baseAddress);
  }
  #endif
};

class CFileUnmapper
{
  const void *_data;
public:
  CFileUnmapper(const void *data) : _data(data) {}
  ~CFileUnmapper() { ::UnmapViewOfFile(_data); }
};

}

#endif
