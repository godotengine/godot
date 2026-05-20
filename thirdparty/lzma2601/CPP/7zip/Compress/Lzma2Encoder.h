// Lzma2Encoder.h

#ifndef ZIP7_INC_LZMA2_ENCODER_H
#define ZIP7_INC_LZMA2_ENCODER_H

#include "../../../C/Lzma2Enc.h"

#include "../../Common/MyCom.h"

#include "../ICoder.h"

namespace NCompress {
namespace NLzma2 {

Z7_CLASS_IMP_COM_4(
  CEncoder
  , ICompressCoder
  , ICompressSetCoderProperties
  , ICompressWriteCoderProperties
  , ICompressSetCoderPropertiesOpt
)
  CLzma2EncHandle _encoder;
public:
  CEncoder();
  ~CEncoder();
};

}}

#endif
