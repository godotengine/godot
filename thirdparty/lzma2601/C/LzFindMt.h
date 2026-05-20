/* LzFindMt.h -- multithreaded Match finder for LZ algorithms
: Igor Pavlov : Public domain */

#ifndef ZIP7_INC_LZ_FIND_MT_H
#define ZIP7_INC_LZ_FIND_MT_H

#include "LzFind.h"
#include "Threads.h"

EXTERN_C_BEGIN

typedef struct
{
  UInt32 numProcessedBlocks;
  Int32 affinityGroup;
  UInt64 affinityInGroup;
  UInt64 affinity;
  CThread thread;

  BoolInt wasCreated;
  BoolInt needStart;
  BoolInt csWasInitialized;
  BoolInt csWasEntered;

  BoolInt exit;
  BoolInt stopWriting;

  CAutoResetEvent canStart;
  CAutoResetEvent wasStopped;
  CSemaphore freeSemaphore;
  CSemaphore filledSemaphore;
  CCriticalSection cs;
  // UInt32 numBlocks_Sent;
} CMtSync;


struct CMatchFinderMt_;

typedef UInt32 * (*Mf_Mix_Matches)(struct CMatchFinderMt_ *p, UInt32 matchMinPos, UInt32 *distances);

/* kMtCacheLineDummy must be >= size_of_CPU_cache_line */
#define kMtCacheLineDummy 128

typedef void (*Mf_GetHeads)(const Byte *buffer, UInt32 pos,
  UInt32 *hash, UInt32 hashMask, UInt32 *heads, UInt32 numHeads, const UInt32 *crc);

typedef struct CMatchFinderMt_
{
  /* LZ */
  const Byte *pointerToCurPos;
  UInt32 *btBuf;
  const UInt32 *btBufPos;
  const UInt32 *btBufPosLimit;
  UInt32 lzPos;
  UInt32 btNumAvailBytes;

  UInt32 *hash;
  UInt32 fixedHashSize;
  // UInt32 hash4Mask;
  UInt32 historySize;
  const UInt32 *crc;

  Mf_Mix_Matches MixMatchesFunc;
  UInt32 failure_LZ_BT; // failure in BT transfered to LZ
  // UInt32 failure_LZ_LZ; // failure in LZ tables
  UInt32 failureBuf[1];
  // UInt32 crc[256];

  /* LZ + BT */
  CMtSync btSync;
  Byte btDummy[kMtCacheLineDummy];

  /* BT */
  UInt32 *hashBuf;
  UInt32 hashBufPos;
  UInt32 hashBufPosLimit;
  UInt32 hashNumAvail;
  UInt32 failure_BT;


  CLzRef *son;
  UInt32 matchMaxLen;
  UInt32 numHashBytes;
  UInt32 pos;
  const Byte *buffer;
  UInt32 cyclicBufferPos;
  UInt32 cyclicBufferSize; /* it must be = (historySize + 1) */
  UInt32 cutValue;

  /* BT + Hash */
  CMtSync hashSync;
  /* Byte hashDummy[kMtCacheLineDummy]; */
  
  /* Hash */
  Mf_GetHeads GetHeadsFunc;
  CMatchFinder *MatchFinder;
  // CMatchFinder MatchFinder;
} CMatchFinderMt;

// only for Mt part
void MatchFinderMt_Construct(CMatchFinderMt *p);
void MatchFinderMt_Destruct(CMatchFinderMt *p, ISzAllocPtr alloc);

SRes MatchFinderMt_Create(CMatchFinderMt *p, UInt32 historySize, UInt32 keepAddBufferBefore,
    UInt32 matchMaxLen, UInt32 keepAddBufferAfter, ISzAllocPtr alloc);
void MatchFinderMt_CreateVTable(CMatchFinderMt *p, IMatchFinder2 *vTable);

/* call MatchFinderMt_InitMt() before IMatchFinder::Init() */
SRes MatchFinderMt_InitMt(CMatchFinderMt *p);
void MatchFinderMt_ReleaseStream(CMatchFinderMt *p);

EXTERN_C_END

#endif
