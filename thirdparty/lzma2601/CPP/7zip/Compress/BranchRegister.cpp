// BranchRegister.cpp

#include "StdAfx.h"

#include "../Common/RegisterCodec.h"

#include "BranchMisc.h"

namespace NCompress {
namespace NBranch {

#ifdef Z7_EXTRACT_ONLY
#define GET_CREATE_FUNC(x) NULL
#define CREATE_BRA_E(n)
#else
#define GET_CREATE_FUNC(x) x
#define CREATE_BRA_E(n) \
    REGISTER_FILTER_CREATE(CreateBra_Encoder_ ## n, CCoder(Z7_BRANCH_CONV_ENC_2(n)))
#endif

#define CREATE_BRA(n) \
    REGISTER_FILTER_CREATE(CreateBra_Decoder_ ## n, CCoder(Z7_BRANCH_CONV_DEC_2(n))) \
    CREATE_BRA_E(n)

CREATE_BRA(BranchConv_PPC)
CREATE_BRA(BranchConv_IA64)
CREATE_BRA(BranchConv_ARM)
CREATE_BRA(BranchConv_ARMT)
CREATE_BRA(BranchConv_SPARC)

#define METHOD_ITEM(n, id, name) \
    REGISTER_FILTER_ITEM( \
      CreateBra_Decoder_ ## n, GET_CREATE_FUNC( \
      CreateBra_Encoder_ ## n), \
      0x3030000 + id, name)

REGISTER_CODECS_VAR
{
  METHOD_ITEM(BranchConv_PPC,   0x205, "PPC"),
  METHOD_ITEM(BranchConv_IA64,  0x401, "IA64"),
  METHOD_ITEM(BranchConv_ARM,   0x501, "ARM"),
  METHOD_ITEM(BranchConv_ARMT,  0x701, "ARMT"),
  METHOD_ITEM(BranchConv_SPARC, 0x805, "SPARC")
};

REGISTER_CODECS(Branch)


#define REGISTER_FILTER_E_BRANCH(id, n, name, alignment) \
    REGISTER_FILTER_E(n, \
      CDecoder(Z7_BRANCH_CONV_DEC(n), alignment), \
      CEncoder(Z7_BRANCH_CONV_ENC(n), alignment), \
      id, name)

REGISTER_FILTER_E_BRANCH(0xa, ARM64, "ARM64", 3)
REGISTER_FILTER_E_BRANCH(0xb, RISCV, "RISCV", 1)

}}
