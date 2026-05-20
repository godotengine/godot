// PpmdEncoder.h

#ifndef ZIP7_INC_COMPRESS_PPMD_ENCODER_H
#define ZIP7_INC_COMPRESS_PPMD_ENCODER_H

#include "../../../C/Ppmd7.h"

#include "../../Common/MyCom.h"

#include "../ICoder.h"

#include "../Common/CWrappers.h"

namespace NCompress {
namespace NPpmd {

struct CEncProps
{
  UInt32 MemSize;
  UInt32 ReduceSize;
  int Order;
  
  CEncProps()
  {
    MemSize = (UInt32)(Int32)-1;
    ReduceSize = (UInt32)(Int32)-1;
    Order = -1;
  }
  void Normalize(int level);
};

Z7_CLASS_IMP_COM_3(
  CEncoder
  , ICompressCoder
  , ICompressSetCoderProperties
  , ICompressWriteCoderProperties
)
  Byte *_inBuf;
  CByteOutBufWrap _outStream;
  CPpmd7 _ppmd;
  CEncProps _props;
public:
  CEncoder();
  ~CEncoder();
};

}}

#endif
