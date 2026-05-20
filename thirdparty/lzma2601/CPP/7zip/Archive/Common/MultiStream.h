// MultiStream.h

#ifndef ZIP7_INC_MULTI_STREAM_H
#define ZIP7_INC_MULTI_STREAM_H

#include "../../../Common/MyCom.h"
#include "../../../Common/MyVector.h"

#include "../../IStream.h"
#include "../../Archive/IArchive.h"

Z7_CLASS_IMP_IInStream(
  CMultiStream
)

  unsigned _streamIndex;
  UInt64 _pos;
  UInt64 _totalLength;

public:

  struct CSubStreamInfo
  {
    CMyComPtr<IInStream> Stream;
    UInt64 Size;
    UInt64 GlobalOffset;
    UInt64 LocalPos;
    CSubStreamInfo(): Size(0), GlobalOffset(0), LocalPos(0) {}
  };

  CMyComPtr<IArchiveUpdateCallbackFile> updateCallbackFile;
  CObjectVector<CSubStreamInfo> Streams;
 
  HRESULT Init()
  {
    UInt64 total = 0;
    FOR_VECTOR (i, Streams)
    {
      CSubStreamInfo &s = Streams[i];
      s.GlobalOffset = total;
      total += s.Size;
      s.LocalPos = 0;
      {
        // it was already set to start
        // RINOK(InStream_GetPos(s.Stream, s.LocalPos));
      }
    }
    _totalLength = total;
    _pos = 0;
    _streamIndex = 0;
    return S_OK;
  }
};

/*
Z7_CLASS_IMP_COM_1(
  COutMultiStream,
  IOutStream
)
  Z7_IFACE_COM7_IMP(ISequentialOutStream)

  unsigned _streamIndex; // required stream
  UInt64 _offsetPos; // offset from start of _streamIndex index
  UInt64 _absPos;
  UInt64 _length;

  struct CSubStreamInfo
  {
    CMyComPtr<ISequentialOutStream> Stream;
    UInt64 Size;
    UInt64 Pos;
 };
  CObjectVector<CSubStreamInfo> Streams;
public:
  CMyComPtr<IArchiveUpdateCallback2> VolumeCallback;
  void Init()
  {
    _streamIndex = 0;
    _offsetPos = 0;
    _absPos = 0;
    _length = 0;
  }
};
*/

#endif
