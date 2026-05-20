// LzmaEncoder.h

#ifndef ZIP7_INC_LZMA_ENCODER_H
#define ZIP7_INC_LZMA_ENCODER_H

#include "../../../C/LzmaEnc.h"

#include "../../Common/MyCom.h"

#include "../ICoder.h"

namespace NCompress {
namespace NLzma {

class CEncoder Z7_final:
  public ICompressCoder,
  public ICompressSetCoderProperties,
  public ICompressWriteCoderProperties,
  public ICompressSetCoderPropertiesOpt,
  public CMyUnknownImp
{
  Z7_COM_UNKNOWN_IMP_4(
      ICompressCoder,
      ICompressSetCoderProperties,
      ICompressWriteCoderProperties,
      ICompressSetCoderPropertiesOpt)
  Z7_IFACE_COM7_IMP(ICompressCoder)
public:
  Z7_IFACE_COM7_IMP(ICompressSetCoderProperties)
  Z7_IFACE_COM7_IMP(ICompressWriteCoderProperties)
  Z7_IFACE_COM7_IMP(ICompressSetCoderPropertiesOpt)

  CLzmaEncHandle _encoder;
  UInt64 _inputProcessed;

  CEncoder();
  ~CEncoder();

  UInt64 GetInputProcessedSize() const { return _inputProcessed; }
  bool IsWriteEndMark() const { return LzmaEnc_IsWriteEndMark(_encoder) != 0; }
};

}}

#endif
