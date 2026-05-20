// 7zSpecStream.h

#ifndef ZIP7_INC_7Z_SPEC_STREAM_H
#define ZIP7_INC_7Z_SPEC_STREAM_H

#include "../../../Common/MyCom.h"

#include "../../ICoder.h"

/*
#define Z7_COM_QI_ENTRY_AG_2(i, sub0, sub) else if (iid == IID_ ## i) \
  { if (!sub) RINOK(sub0->QueryInterface(IID_ ## i, (void **)&sub)) \
    { i *ti = this;  *outObject = ti; }  }

class CSequentialInStreamSizeCount2 Z7_final:
  public ISequentialInStream,
  public ICompressGetSubStreamSize,
  public ICompressInSubStreams,
  public CMyUnknownImp
{
  Z7_COM_QI_BEGIN2(ISequentialInStream)
    Z7_COM_QI_ENTRY(ICompressGetSubStreamSize)
    Z7_COM_QI_ENTRY_AG_2(ISequentialInStream, _stream, _compressGetSubStreamSize)
  Z7_COM_QI_END
  Z7_COM_ADDREF_RELEASE

  Z7_IFACE_COM7_IMP(ISequentialInStream)
  Z7_IFACE_COM7_IMP(ICompressGetSubStreamSize)
  Z7_IFACE_COM7_IMP(ICompressInSubStreams)

  CMyComPtr<ISequentialInStream> _stream;
  CMyComPtr<ICompressGetSubStreamSize> _getSubStreamSize;
  CMyComPtr<ICompressInSubStreams> _compressGetSubStreamSize;
  UInt64 _size;
public:
  void Init(ISequentialInStream *stream)
  {
    _size = 0;
    _getSubStreamSize.Release();
    _compressGetSubStreamSize.Release();
    _stream = stream;
    _stream.QueryInterface(IID_ICompressGetSubStreamSize, &_getSubStreamSize);
    _stream.QueryInterface(IID_ICompressInSubStreams, &_compressGetSubStreamSize);
  }
  UInt64 GetSize() const { return _size; }
};
*/

#endif
