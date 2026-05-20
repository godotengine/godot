// Lzma2Decoder.h

#ifndef ZIP7_INC_LZMA2_DECODER_H
#define ZIP7_INC_LZMA2_DECODER_H

#include "../../../C/Lzma2DecMt.h"

#include "../Common/CWrappers.h"

namespace NCompress {
namespace NLzma2 {

class CDecoder Z7_final:
  public ICompressCoder,
  public ICompressSetDecoderProperties2,
  public ICompressSetFinishMode,
  public ICompressGetInStreamProcessedSize,
  public ICompressSetBufSize,
 #ifndef Z7_NO_READ_FROM_CODER
  public ICompressSetInStream,
  public ICompressSetOutStreamSize,
  public ISequentialInStream,
 #endif
 #ifndef Z7_ST
  public ICompressSetCoderMt,
  public ICompressSetMemLimit,
 #endif
  public CMyUnknownImp
{
  Z7_COM_QI_BEGIN2(ICompressCoder)
  Z7_COM_QI_ENTRY(ICompressSetDecoderProperties2)
  Z7_COM_QI_ENTRY(ICompressSetFinishMode)
  Z7_COM_QI_ENTRY(ICompressGetInStreamProcessedSize)
  Z7_COM_QI_ENTRY(ICompressSetBufSize)
 #ifndef Z7_NO_READ_FROM_CODER
  Z7_COM_QI_ENTRY(ICompressSetInStream)
  Z7_COM_QI_ENTRY(ICompressSetOutStreamSize)
  Z7_COM_QI_ENTRY(ISequentialInStream)
 #endif
 #ifndef Z7_ST
  Z7_COM_QI_ENTRY(ICompressSetCoderMt)
  Z7_COM_QI_ENTRY(ICompressSetMemLimit)
 #endif
  Z7_COM_QI_END
  Z7_COM_ADDREF_RELEASE

  Z7_IFACE_COM7_IMP(ICompressCoder)
  Z7_IFACE_COM7_IMP(ICompressSetDecoderProperties2)
  Z7_IFACE_COM7_IMP(ICompressSetFinishMode)
  Z7_IFACE_COM7_IMP(ICompressGetInStreamProcessedSize)
  Z7_IFACE_COM7_IMP(ICompressSetBufSize)
 #ifndef Z7_NO_READ_FROM_CODER
  Z7_IFACE_COM7_IMP(ICompressSetOutStreamSize)
  Z7_IFACE_COM7_IMP(ICompressSetInStream)
  Z7_IFACE_COM7_IMP(ISequentialInStream)
 #endif
 #ifndef Z7_ST
  Z7_IFACE_COM7_IMP(ICompressSetCoderMt)
  Z7_IFACE_COM7_IMP(ICompressSetMemLimit)
 #endif

  CLzma2DecMtHandle _dec;
  UInt64 _inProcessed;
  Byte _prop;
  int _finishMode;
  UInt32 _inBufSize;
  UInt32 _outStep;

 #ifndef Z7_ST
  int _tryMt;
  UInt32 _numThreads;
  UInt64 _memUsage;
 #endif

 #ifndef Z7_NO_READ_FROM_CODER
  CMyComPtr<ISequentialInStream> _inStream;
  CSeqInStreamWrap _inWrap;
 #endif

public:
  CDecoder();
  ~CDecoder();
};

}}

#endif
