// Bcj2Coder.h

#ifndef ZIP7_INC_COMPRESS_BCJ2_CODER_H
#define ZIP7_INC_COMPRESS_BCJ2_CODER_H

#include "../../../C/Bcj2.h"

#include "../../Common/MyCom.h"

#include "../ICoder.h"

namespace NCompress {
namespace NBcj2 {

class CBaseCoder
{
protected:
  Byte *_bufs[BCJ2_NUM_STREAMS + 1];
  UInt32 _bufsSizes[BCJ2_NUM_STREAMS + 1];
  UInt32 _bufsSizes_New[BCJ2_NUM_STREAMS + 1];

  HRESULT Alloc(bool allocForOrig = true);
public:
  CBaseCoder();
  ~CBaseCoder();
};


#ifndef Z7_EXTRACT_ONLY

class CEncoder Z7_final:
  public ICompressCoder2,
  public ICompressSetCoderProperties,
  public ICompressSetBufSize,
  public CMyUnknownImp,
  public CBaseCoder
{
  Z7_IFACES_IMP_UNK_3(
      ICompressCoder2,
      ICompressSetCoderProperties,
      ICompressSetBufSize)

  UInt32 _relatLim;
  // UInt32 _excludeRangeBits;

  HRESULT CodeReal(
      ISequentialInStream * const *inStreams, const UInt64 * const *inSizes, UInt32 numInStreams,
      ISequentialOutStream * const *outStreams, const UInt64 * const *outSizes, UInt32 numOutStreams,
      ICompressProgressInfo *progress);
public:
  CEncoder();
  ~CEncoder();
};

#endif



class CBaseDecoder: public CBaseCoder
{
protected:
  HRESULT _readRes[BCJ2_NUM_STREAMS];
  unsigned _extraSizes[BCJ2_NUM_STREAMS];
  UInt64 _readSizes[BCJ2_NUM_STREAMS];

  CBcj2Dec dec;

  UInt64 GetProcessedSize_ForInStream(unsigned i) const
  {
    return _readSizes[i] - ((size_t)(dec.lims[i] - dec.bufs[i]) + _extraSizes[i]);
  }
  void InitCommon();
  void ReadInStream(ISequentialInStream *inStream);
};


class CDecoder Z7_final:
  public ICompressCoder2,
  public ICompressSetFinishMode,
  public ICompressGetInStreamProcessedSize2,
  public ICompressSetBufSize,
#ifndef Z7_NO_READ_FROM_CODER
  public ICompressSetInStream2,
  public ICompressSetOutStreamSize,
  public ISequentialInStream,
#endif
  public CMyUnknownImp,
  public CBaseDecoder
{
  Z7_COM_QI_BEGIN2(ICompressCoder2)
    Z7_COM_QI_ENTRY(ICompressSetFinishMode)
    Z7_COM_QI_ENTRY(ICompressGetInStreamProcessedSize2)
    Z7_COM_QI_ENTRY(ICompressSetBufSize)
  #ifndef Z7_NO_READ_FROM_CODER
    Z7_COM_QI_ENTRY(ICompressSetInStream2)
    Z7_COM_QI_ENTRY(ICompressSetOutStreamSize)
    Z7_COM_QI_ENTRY(ISequentialInStream)
  #endif
  Z7_COM_QI_END
  Z7_COM_ADDREF_RELEASE
  
  Z7_IFACE_COM7_IMP(ICompressCoder2)
  Z7_IFACE_COM7_IMP(ICompressSetFinishMode)
  Z7_IFACE_COM7_IMP(ICompressGetInStreamProcessedSize2)
  Z7_IFACE_COM7_IMP(ICompressSetBufSize)
#ifndef Z7_NO_READ_FROM_CODER
  Z7_IFACE_COM7_IMP(ICompressSetInStream2)
  Z7_IFACE_COM7_IMP(ICompressSetOutStreamSize)
  Z7_IFACE_COM7_IMP(ISequentialInStream)
#endif

  bool _finishMode;

#ifndef Z7_NO_READ_FROM_CODER
  bool _outSizeDefined;
  UInt64 _outSize;
  UInt64 _outSize_Processed;
  CMyComPtr<ISequentialInStream> _inStreams[BCJ2_NUM_STREAMS];
#endif
 
public:
  CDecoder();
};

}}

#endif
