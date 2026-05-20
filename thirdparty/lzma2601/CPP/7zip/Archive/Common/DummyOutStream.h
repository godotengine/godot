// DummyOutStream.h

#ifndef ZIP7_INC_DUMMY_OUT_STREAM_H
#define ZIP7_INC_DUMMY_OUT_STREAM_H

#include "../../../Common/MyCom.h"

#include "../../IStream.h"

Z7_CLASS_IMP_NOQIB_1(
  CDummyOutStream
  , ISequentialOutStream
)
  CMyComPtr<ISequentialOutStream> _stream;
  UInt64 _size;
public:
  void SetStream(ISequentialOutStream *outStream) { _stream = outStream; }
  void ReleaseStream() { _stream.Release(); }
  void Init() { _size = 0; }
  UInt64 GetSize() const { return _size; }
};

#endif
