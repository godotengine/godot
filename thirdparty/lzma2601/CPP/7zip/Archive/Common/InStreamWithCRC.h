// InStreamWithCRC.h

#ifndef ZIP7_INC_IN_STREAM_WITH_CRC_H
#define ZIP7_INC_IN_STREAM_WITH_CRC_H

#include "../../../../C/7zCrc.h"

#include "../../../Common/MyCom.h"

#include "../../IStream.h"

Z7_CLASS_IMP_NOQIB_2(
  CSequentialInStreamWithCRC
  , ISequentialInStream
  , IStreamGetSize
)
  CMyComPtr<ISequentialInStream> _stream;
  UInt64 _size;
  UInt32 _crc;
  bool _wasFinished;
  UInt64 _fullSize;
public:
  
  CSequentialInStreamWithCRC():
    _fullSize((UInt64)(Int64)-1)
    {}

  void SetStream(ISequentialInStream *stream) { _stream = stream; }
  void SetFullSize(UInt64 fullSize) { _fullSize = fullSize; }
  void Init()
  {
    _size = 0;
    _crc = CRC_INIT_VAL;
    _wasFinished = false;
  }
  void ReleaseStream() { _stream.Release(); }
  UInt32 GetCRC() const { return CRC_GET_DIGEST(_crc); }
  UInt64 GetSize() const { return _size; }
  bool WasFinished() const { return _wasFinished; }
};


Z7_CLASS_IMP_IInStream(
  CInStreamWithCRC
)
  CMyComPtr<IInStream> _stream;
  UInt64 _size;
  UInt32 _crc;
  // bool _wasFinished;
public:
  void SetStream(IInStream *stream) { _stream = stream; }
  void Init()
  {
    _size = 0;
    // _wasFinished = false;
    _crc = CRC_INIT_VAL;
  }
  void ReleaseStream() { _stream.Release(); }
  UInt32 GetCRC() const { return CRC_GET_DIGEST(_crc); }
  UInt64 GetSize() const { return _size; }
  // bool WasFinished() const { return _wasFinished; }
};

#endif
