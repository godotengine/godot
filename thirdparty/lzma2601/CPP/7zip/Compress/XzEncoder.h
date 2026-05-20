// XzEncoder.h

#ifndef ZIP7_INC_XZ_ENCODER_H
#define ZIP7_INC_XZ_ENCODER_H

#include "../../../C/XzEnc.h"

#include "../../Common/MyCom.h"

#include "../ICoder.h"

namespace NCompress {
namespace NXz {

Z7_CLASS_IMP_COM_3(
  CEncoder
  , ICompressCoder
  , ICompressSetCoderProperties
  , ICompressSetCoderPropertiesOpt
)
  CXzEncHandle _encoder;
public:
  CXzProps xzProps;

  void InitCoderProps();
  HRESULT SetCheckSize(UInt32 checkSizeInBytes);
  HRESULT SetCoderProp(PROPID propID, const PROPVARIANT &prop);

  CEncoder();
  ~CEncoder();
};

}}

#endif
