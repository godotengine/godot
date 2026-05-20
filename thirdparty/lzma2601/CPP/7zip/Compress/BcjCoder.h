// BcjCoder.h

#ifndef ZIP7_INC_COMPRESS_BCJ_CODER_H
#define ZIP7_INC_COMPRESS_BCJ_CODER_H

#include "../../../C/Bra.h"

#include "../../Common/MyCom.h"

#include "../ICoder.h"

namespace NCompress {
namespace NBcj {

/* CCoder in old versions used another constructor parameter CCoder(int encode).
   And some code called it as CCoder(0).
   We have changed constructor parameter type.
   So we have changed the name of class also to CCoder2. */

Z7_CLASS_IMP_COM_1(
  CCoder2
  , ICompressFilter
)
  UInt32 _pc;
  UInt32 _state;
  z7_Func_BranchConvSt _convFunc;
public:
  CCoder2(z7_Func_BranchConvSt convFunc):
      _pc(0),
      _state(Z7_BRANCH_CONV_ST_X86_STATE_INIT_VAL),
      _convFunc(convFunc)
    {}
};

}}

#endif
