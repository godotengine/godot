// StreamBinder.cpp

#include "StdAfx.h"

#include "../../Common/MyCom.h"

#include "StreamBinder.h"

Z7_CLASS_IMP_COM_1(
  CBinderInStream
  , ISequentialInStream
)
  CStreamBinder *_binder;
public:
  ~CBinderInStream() { _binder->CloseRead_CallOnce(); }
  CBinderInStream(CStreamBinder *binder): _binder(binder) {}
};

Z7_COM7F_IMF(CBinderInStream::Read(void *data, UInt32 size, UInt32 *processedSize))
  { return _binder->Read(data, size, processedSize); }


Z7_CLASS_IMP_COM_1(
  CBinderOutStream
  , ISequentialOutStream
)
  CStreamBinder *_binder;
public:
  ~CBinderOutStream() { _binder->CloseWrite(); }
  CBinderOutStream(CStreamBinder *binder): _binder(binder) {}
};

Z7_COM7F_IMF(CBinderOutStream::Write(const void *data, UInt32 size, UInt32 *processedSize))
  { return _binder->Write(data, size, processedSize); }


static HRESULT Event_Create_or_Reset(NWindows::NSynchronization::CAutoResetEvent &event)
{
  const WRes wres = event.CreateIfNotCreated_Reset();
  return HRESULT_FROM_WIN32(wres);
}

HRESULT CStreamBinder::Create_ReInit()
{
  RINOK(Event_Create_or_Reset(_canRead_Event))
  // RINOK(Event_Create_or_Reset(_canWrite_Event))

  // _canWrite_Semaphore.Close();
  // we need at least 3 items of maxCount: 1 for normal unlock in Read(), 2 items for unlock in CloseRead_CallOnce()
  _canWrite_Semaphore.OptCreateInit(0, 3);

  // _readingWasClosed = false;
  _readingWasClosed2 = false;

  _waitWrite = true;
  _bufSize = 0;
  _buf = NULL;
  ProcessedSize = 0;
  // WritingWasCut = false;
  return S_OK;
}


void CStreamBinder::CreateStreams2(CMyComPtr<ISequentialInStream> &inStream, CMyComPtr<ISequentialOutStream> &outStream)
{
  inStream = new CBinderInStream(this);
  outStream = new CBinderOutStream(this);
}

// (_canRead_Event && _bufSize == 0) means that stream is finished.

HRESULT CStreamBinder::Read(void *data, UInt32 size, UInt32 *processedSize)
{
  if (processedSize)
    *processedSize = 0;
  if (size != 0)
  {
    if (_waitWrite)
    {
      WRes wres = _canRead_Event.Lock();
      if (wres != 0)
        return HRESULT_FROM_WIN32(wres);
      _waitWrite = false;
    }
    if (size > _bufSize)
      size = _bufSize;
    if (size != 0)
    {
      memcpy(data, _buf, size);
      _buf = ((const Byte *)_buf) + size;
      ProcessedSize += size;
      if (processedSize)
        *processedSize = size;
      _bufSize -= size;

      /*
      if (_bufSize == 0), then we have read whole buffer
      we have two ways here:
        - if we       check (_bufSize == 0) here, we unlock Write only after full data Reading - it reduces the number of syncs
        - if we don't check (_bufSize == 0) here, we unlock Write after partial data Reading
      */
      if (_bufSize == 0)
      {
        _waitWrite = true;
        // _canWrite_Event.Set();
        _canWrite_Semaphore.Release();
      }
    }
  }
  return S_OK;
}


HRESULT CStreamBinder::Write(const void *data, UInt32 size, UInt32 *processedSize)
{
  if (processedSize)
    *processedSize = 0;
  if (size == 0)
    return S_OK;

  if (!_readingWasClosed2)
  {
    _buf = data;
    _bufSize = size;
    _canRead_Event.Set();
    
    /*
    _canWrite_Event.Lock();
    if (_readingWasClosed)
      _readingWasClosed2 = true;
    */

    _canWrite_Semaphore.Lock();

    // _bufSize : is remain size that was not read
    size -= _bufSize;

    // size : is size of data that was read
    if (size != 0)
    {
      // if some data was read, then we report that size and return
      if (processedSize)
        *processedSize = size;
      return S_OK;
    }
    _readingWasClosed2 = true;
  }

  // WritingWasCut = true;
  return k_My_HRESULT_WritingWasCut;
}
