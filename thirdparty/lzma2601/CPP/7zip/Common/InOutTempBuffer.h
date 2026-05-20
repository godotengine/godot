// InOutTempBuffer.h

#ifndef ZIP7_INC_IN_OUT_TEMP_BUFFER_H
#define ZIP7_INC_IN_OUT_TEMP_BUFFER_H

// #ifdef _WIN32
#define USE_InOutTempBuffer_FILE
// #endif

#ifdef USE_InOutTempBuffer_FILE
#include "../../Windows/FileDir.h"
#endif

#include "../IStream.h"

class CInOutTempBuffer
{
  UInt64 _size;
  void **_bufs;
  size_t _numBufs;
  size_t _numFilled;

 #ifdef USE_InOutTempBuffer_FILE
  
  bool _tempFile_Created;
  bool _useMemOnly;
  UInt32 _crc;
  // COutFile object must be declared after CTempFile object for correct destructor order
  NWindows::NFile::NDir::CTempFile _tempFile;
  NWindows::NFile::NIO::COutFile _outFile;

 #endif

  void *GetBuf(size_t index);

  Z7_CLASS_NO_COPY(CInOutTempBuffer)
public:
  CInOutTempBuffer();
  ~CInOutTempBuffer();
  HRESULT Write_HRESULT(const void *data, UInt32 size);
  HRESULT WriteToStream(ISequentialOutStream *stream);
  UInt64 GetDataSize() const { return _size; }
};

#endif
