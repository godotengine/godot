// StreamBinder.h

#ifndef ZIP7_INC_STREAM_BINDER_H
#define ZIP7_INC_STREAM_BINDER_H

#include "../../Windows/Synchronization.h"

#include "../IStream.h"

/*
We can use one from two code versions here: with Event or with Semaphore to unlock Writer thread
The difference for cases where Reading must be closed before Writing closing

1) Event Version: _canWrite_Event
  We call _canWrite_Event.Set() without waiting _canRead_Event in CloseRead() function.
  The writer thread can get (_readingWasClosed) status in one from two iterations.
  It's ambiguity of processing flow. But probably it's SAFE to use, if Event functions provide memory barriers.
  reader thread:
     _canWrite_Event.Set();
     _readingWasClosed = true;
     _canWrite_Event.Set();
  writer thread:
     _canWrite_Event.Wait()
      if (_readingWasClosed)

2) Semaphore Version: _canWrite_Semaphore
  writer thread always will detect closing of reading in latest iteration after all data processing iterations
*/

class CStreamBinder
{
  NWindows::NSynchronization::CAutoResetEvent _canRead_Event;
  // NWindows::NSynchronization::CAutoResetEvent _canWrite_Event;
  NWindows::NSynchronization::CSemaphore _canWrite_Semaphore;

  // bool _readingWasClosed;  // set it in reader thread and check it in write thread
  bool _readingWasClosed2; // use it in writer thread
  // bool WritingWasCut;
  bool _waitWrite;         // use it in reader thread
  UInt32 _bufSize;
  const void *_buf;
public:
  UInt64 ProcessedSize;   // the size that was read by reader thread

  void CreateStreams2(CMyComPtr<ISequentialInStream> &inStream, CMyComPtr<ISequentialOutStream> &outStream);
  
  HRESULT Create_ReInit();
  
  HRESULT Read(void *data, UInt32 size, UInt32 *processedSize);
  HRESULT Write(const void *data, UInt32 size, UInt32 *processedSize);

  void CloseRead_CallOnce()
  {
    // call it only once: for example, in destructor
    
    /*
    _readingWasClosed = true;
    _canWrite_Event.Set();
    */

    /*
    We must relase Semaphore only once !!!
    we must release at least 2 items of Semaphore:
      one item to unlock partial Write(), if Read() have read some items
      then additional item to stop writing (_bufSize will be 0)
    */
    _canWrite_Semaphore.Release(2);
  }
  
  void CloseWrite()
  {
    _buf = NULL;
    _bufSize = 0;
    _canRead_Event.Set();
  }
};

#endif
