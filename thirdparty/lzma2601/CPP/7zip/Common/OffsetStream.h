// OffsetStream.h

#ifndef ZIP7_INC_OFFSET_STREAM_H
#define ZIP7_INC_OFFSET_STREAM_H

#include "../../Common/MyCom.h"

#include "../IStream.h"

Z7_CLASS_IMP_NOQIB_1(
  COffsetOutStream
  , IOutStream
)
  Z7_IFACE_COM7_IMP(ISequentialOutStream)

  CMyComPtr<IOutStream> _stream;
  UInt64 _offset;
public:
  HRESULT Init(IOutStream *stream, UInt64 offset);
};

#endif
