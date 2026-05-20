// InOutTempBuffer.cpp

#include "StdAfx.h"

#include "../../../C/Alloc.h"

#include "InOutTempBuffer.h"

#include "StreamUtils.h"

#ifdef USE_InOutTempBuffer_FILE

#include "../../../C/7zCrc.h"

#define kTempFilePrefixString FTEXT("7zt")
/*
  Total buffer size limit, if we use temp file scheme:
    32-bit:  16 MiB = 1 MiB *   16 buffers
    64-bit:   4 GiB = 1 MiB * 4096 buffers
*/
static const size_t kNumBufsMax = (size_t)1 << (sizeof(size_t) * 2 - 4);

#endif

static const size_t kBufSize = (size_t)1 << 20;


CInOutTempBuffer::CInOutTempBuffer():
    _size(0),
    _bufs(NULL),
    _numBufs(0),
    _numFilled(0)
{
 #ifdef USE_InOutTempBuffer_FILE
  _tempFile_Created = false;
  _useMemOnly = false;
  _crc = CRC_INIT_VAL;
 #endif
}

CInOutTempBuffer::~CInOutTempBuffer()
{
  for (size_t i = 0; i < _numBufs; i++)
    MyFree(_bufs[i]);
  MyFree(_bufs);
}


void *CInOutTempBuffer::GetBuf(size_t index)
{
  if (index >= _numBufs)
  {
    const size_t num = (_numBufs == 0 ? 16 : _numBufs * 2);
    void **p = (void **)MyRealloc(_bufs, num * sizeof(void *));
    if (!p)
      return NULL;
    _bufs = p;
    memset(p + _numBufs, 0, (num - _numBufs) * sizeof(void *));
    _numBufs = num;
  }
  
  void *buf = _bufs[index];
  if (!buf)
  {
    buf = MyAlloc(kBufSize);
    if (buf)
      _bufs[index] = buf;
  }
  return buf;
}


HRESULT CInOutTempBuffer::Write_HRESULT(const void *data, UInt32 size)
{
  if (size == 0)
    return S_OK;
  
 #ifdef USE_InOutTempBuffer_FILE
  if (!_tempFile_Created)
 #endif
  for (;;)  // loop for additional attemp to allocate memory after file creation error
  {
   #ifdef USE_InOutTempBuffer_FILE
    bool allocError = false;
   #endif
  
    for (;;)  // loop for writing to buffers
    {
      const size_t index = (size_t)(_size / kBufSize);
      
     #ifdef USE_InOutTempBuffer_FILE
      if (index >= kNumBufsMax && !_useMemOnly)
        break;
     #endif
    
      void *buf = GetBuf(index);
      if (!buf)
      {
       #ifdef USE_InOutTempBuffer_FILE
        if (!_useMemOnly)
        {
          allocError = true;
          break;
        }
       #endif
        return E_OUTOFMEMORY;
      }
      
      const size_t offset = (size_t)(_size) & (kBufSize - 1);
      size_t cur = kBufSize - offset;
      if (cur > size)
        cur = size;
      memcpy((Byte *)buf + offset, data, cur);
      _size += cur;
      if (index >= _numFilled)
        _numFilled = index + 1;
      data = (const void *)((const Byte *)data + cur);
      size -= (UInt32)cur;
      if (size == 0)
        return S_OK;
    }

   #ifdef USE_InOutTempBuffer_FILE
   #ifndef _WIN32
    _outFile.mode_for_Create = 0600;  // only owner will have the rights to access this file
   #endif
    if (_tempFile.CreateRandomInTempFolder(kTempFilePrefixString, &_outFile))
    {
      _tempFile_Created = true;
      break;
    }
    _useMemOnly = true;
    if (allocError)
      return GetLastError_noZero_HRESULT();
   #endif
  }

 #ifdef USE_InOutTempBuffer_FILE
  if (!_outFile.WriteFull(data, size))
    return GetLastError_noZero_HRESULT();
  _crc = CrcUpdate(_crc, data, size);
  _size += size;
  return S_OK;
 #endif
}


HRESULT CInOutTempBuffer::WriteToStream(ISequentialOutStream *stream)
{
  UInt64 rem = _size;
  // if (rem == 0) return S_OK;

  const size_t numFilled = _numFilled;
  _numFilled = 0;

  for (size_t i = 0; i < numFilled; i++)
  {
    if (rem == 0)
      return E_FAIL;
    size_t cur = kBufSize;
    if (cur > rem)
      cur = (size_t)rem;
    RINOK(WriteStream(stream, _bufs[i], cur))
    rem -= cur;
   #ifdef USE_InOutTempBuffer_FILE
    // we will use _bufs[0] later for writing from temp file
    if (i != 0 || !_tempFile_Created)
   #endif
    {
      MyFree(_bufs[i]);
      _bufs[i] = NULL;
    }
  }


 #ifdef USE_InOutTempBuffer_FILE

  if (rem == 0)
    return _tempFile_Created ? E_FAIL : S_OK;

  if (!_tempFile_Created)
    return E_FAIL;

  if (!_outFile.Close())
    return GetLastError_noZero_HRESULT();

  HRESULT hres;
  void *buf = GetBuf(0); // index
  if (!buf)
    hres = E_OUTOFMEMORY;
  else
  {
    NWindows::NFile::NIO::CInFile inFile;
    if (!inFile.Open(_tempFile.GetPath()))
      hres = GetLastError_noZero_HRESULT();
    else
    {
      UInt32 crc = CRC_INIT_VAL;
      for (;;)
      {
        size_t processed;
        if (!inFile.ReadFull(buf, kBufSize, processed))
        {
          hres = GetLastError_noZero_HRESULT();
          break;
        }
        if (processed == 0)
        {
          // we compare crc without CRC_GET_DIGEST
          hres = (_crc == crc ? S_OK : E_FAIL);
          break;
        }
        size_t n = processed;
        if (n > rem)
          n = (size_t)rem;
        hres = WriteStream(stream, buf, n);
        if (hres != S_OK)
          break;
        crc = CrcUpdate(crc, buf, n);
        rem -= n;
        if (n != processed)
        {
          hres = E_FAIL;
          break;
        }
      }
    }
  }

  // _tempFile.DisableDeleting(); // for debug
  _tempFile.Remove();
  RINOK(hres)

 #endif

  return rem == 0 ? S_OK : E_FAIL;
}
