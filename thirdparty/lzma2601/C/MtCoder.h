/* MtCoder.h -- Multi-thread Coder
: Igor Pavlov : Public domain */

#ifndef ZIP7_INC_MT_CODER_H
#define ZIP7_INC_MT_CODER_H

#include "MtDec.h"

EXTERN_C_BEGIN

/*
  if (    defined MTCODER_USE_WRITE_THREAD) : main thread writes all data blocks to output stream
  if (not defined MTCODER_USE_WRITE_THREAD) : any coder thread can write data blocks to output stream
*/
/* #define MTCODER_USE_WRITE_THREAD */

#ifndef Z7_ST
  #define MTCODER_GET_NUM_BLOCKS_FROM_THREADS(numThreads) ((numThreads) + (numThreads) / 8 + 1)
  #define MTCODER_THREADS_MAX 256
  #define MTCODER_BLOCKS_MAX (MTCODER_GET_NUM_BLOCKS_FROM_THREADS(MTCODER_THREADS_MAX) + 3)
#else
  #define MTCODER_THREADS_MAX 1
  #define MTCODER_BLOCKS_MAX 1
#endif


#ifndef Z7_ST


typedef struct
{
  ICompressProgress vt;
  CMtProgress *mtProgress;
  UInt64 inSize;
  UInt64 outSize;
} CMtProgressThunk;

void MtProgressThunk_CreateVTable(CMtProgressThunk *p);
    
#define MtProgressThunk_INIT(p) { (p)->inSize = 0; (p)->outSize = 0; }


struct CMtCoder_;


typedef struct
{
  struct CMtCoder_ *mtCoder;
  unsigned index;
  int stop;
  Byte *inBuf;

  CAutoResetEvent startEvent;
  CThread thread;
} CMtCoderThread;


typedef struct
{
  SRes (*Code)(void *p, unsigned coderIndex, unsigned outBufIndex,
      const Byte *src, size_t srcSize, int finished);
  SRes (*Write)(void *p, unsigned outBufIndex);
} IMtCoderCallback2;


typedef struct
{
  SRes res;
  unsigned bufIndex;
  BoolInt finished;
} CMtCoderBlock;


typedef struct CMtCoder_
{
  /* input variables */
  
  size_t blockSize;        /* size of input block */
  unsigned numThreadsMax;
  unsigned numThreadGroups;
  UInt64 expectedDataSize;

  ISeqInStreamPtr inStream;
  const Byte *inData;
  size_t inDataSize;

  ICompressProgressPtr progress;
  ISzAllocPtr allocBig;

  IMtCoderCallback2 *mtCallback;
  void *mtCallbackObject;

  
  /* internal variables */
  
  size_t allocatedBufsSize;

  CAutoResetEvent readEvent;
  CSemaphore blocksSemaphore;

  BoolInt stopReading;
  SRes readRes;

  #ifdef MTCODER_USE_WRITE_THREAD
    CAutoResetEvent writeEvents[MTCODER_BLOCKS_MAX];
  #else
    CAutoResetEvent finishedEvent;
    SRes writeRes;
    unsigned writeIndex;
    Byte ReadyBlocks[MTCODER_BLOCKS_MAX];
    LONG numFinishedThreads;
  #endif

  unsigned numStartedThreadsLimit;
  unsigned numStartedThreads;

  unsigned numBlocksMax;
  unsigned blockIndex;
  UInt64 readProcessed;

  CCriticalSection cs;

  unsigned freeBlockHead;
  unsigned freeBlockList[MTCODER_BLOCKS_MAX];

  CMtProgress mtProgress;
  CMtCoderBlock blocks[MTCODER_BLOCKS_MAX];
  CMtCoderThread threads[MTCODER_THREADS_MAX];

  CThreadNextGroup nextGroup;
} CMtCoder;


void MtCoder_Construct(CMtCoder *p);
void MtCoder_Destruct(CMtCoder *p);
SRes MtCoder_Code(CMtCoder *p);


#endif


EXTERN_C_END

#endif
