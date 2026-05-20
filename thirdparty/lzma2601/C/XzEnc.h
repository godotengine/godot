/* XzEnc.h -- Xz Encode
: Igor Pavlov : Public domain */

#ifndef ZIP7_INC_XZ_ENC_H
#define ZIP7_INC_XZ_ENC_H

#include "Lzma2Enc.h"

#include "Xz.h"

EXTERN_C_BEGIN


#define XZ_PROPS_BLOCK_SIZE_AUTO   LZMA2_ENC_PROPS_BLOCK_SIZE_AUTO
#define XZ_PROPS_BLOCK_SIZE_SOLID  LZMA2_ENC_PROPS_BLOCK_SIZE_SOLID


typedef struct
{
  UInt32 id;
  UInt32 delta;
  UInt32 ip;
  int ipDefined;
} CXzFilterProps;

void XzFilterProps_Init(CXzFilterProps *p);


typedef struct
{
  CLzma2EncProps lzma2Props;
  CXzFilterProps filterProps;
  unsigned checkId;
  unsigned numThreadGroups; // 0 : no groups
  UInt64 blockSize;
  int numBlockThreads_Reduced;
  int numBlockThreads_Max;
  int numTotalThreads;
  int forceWriteSizesInHeader;
  UInt64 reduceSize;
} CXzProps;

void XzProps_Init(CXzProps *p);

typedef struct CXzEnc CXzEnc;
typedef CXzEnc * CXzEncHandle;
// Z7_DECLARE_HANDLE(CXzEncHandle)

CXzEncHandle XzEnc_Create(ISzAllocPtr alloc, ISzAllocPtr allocBig);
void XzEnc_Destroy(CXzEncHandle p);
SRes XzEnc_SetProps(CXzEncHandle p, const CXzProps *props);
void XzEnc_SetDataSize(CXzEncHandle p, UInt64 expectedDataSiize);
SRes XzEnc_Encode(CXzEncHandle p, ISeqOutStreamPtr outStream, ISeqInStreamPtr inStream, ICompressProgressPtr progress);

SRes Xz_Encode(ISeqOutStreamPtr outStream, ISeqInStreamPtr inStream,
    const CXzProps *props, ICompressProgressPtr progress);

SRes Xz_EncodeEmpty(ISeqOutStreamPtr outStream);

EXTERN_C_END

#endif
