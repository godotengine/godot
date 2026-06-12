/* MtDec.h -- Multi-thread Decoder
2023-04-02 : Igor Pavlov : Public domain */

#ifndef ZIP7_INC_MT_DEC_H
#define ZIP7_INC_MT_DEC_H

#include "7zTypes.h"

#ifndef Z7_ST
#include "Threads.h"
#endif

EXTERN_C_BEGIN

#ifndef Z7_ST

#ifndef Z7_ST
  #define MTDEC_THREADS_MAX 32
#else
  #define MTDEC_THREADS_MAX 1
#endif


typedef struct
{
  ICompressProgressPtr progress;
  SRes res;
  UInt64 totalInSize;
  UInt64 totalOutSize;
  CCriticalSection cs;
} CMtProgress;

void MtProgress_Init(CMtProgress *p, ICompressProgressPtr progress);
SRes MtProgress_Progress_ST(CMtProgress *p);
SRes MtProgress_ProgressAdd(CMtProgress *p, UInt64 inSize, UInt64 outSize);
SRes MtProgress_GetError(CMtProgress *p);
void MtProgress_SetError(CMtProgress *p, SRes res);

struct CMtDec;

typedef struct
{
  struct CMtDec_ *mtDec;
  unsigned index;
  void *inBuf;

  size_t inDataSize_Start; // size of input data in start block
  UInt64 inDataSize;       // total size of input data in all blocks

  CThread thread;
  CAutoResetEvent canRead;
  CAutoResetEvent canWrite;
  void  *allocaPtr;
} CMtDecThread;

void MtDecThread_FreeInBufs(CMtDecThread *t);


typedef enum
{
  MTDEC_PARSE_CONTINUE, // continue this block with more input data
  MTDEC_PARSE_OVERFLOW, // MT buffers overflow, need switch to single-thread
  MTDEC_PARSE_NEW,      // new block
  MTDEC_PARSE_END       // end of block threading. But we still can return to threading after Write(&needContinue)
} EMtDecParseState;

typedef struct
{
  // in
  int startCall;
  const Byte *src;
  size_t srcSize;
      // in  : (srcSize == 0) is allowed
      // out : it's allowed to return less that actually was used ?
  int srcFinished;

  // out
  EMtDecParseState state;
  BoolInt canCreateNewThread;
  UInt64 outPos; // check it (size_t)
} CMtDecCallbackInfo;


typedef struct
{
  void (*Parse)(void *p, unsigned coderIndex, CMtDecCallbackInfo *ci);
  
  // PreCode() and Code():
  // (SRes_return_result != SZ_OK) means stop decoding, no need another blocks
  SRes (*PreCode)(void *p, unsigned coderIndex);
  SRes (*Code)(void *p, unsigned coderIndex,
      const Byte *src, size_t srcSize, int srcFinished,
      UInt64 *inCodePos, UInt64 *outCodePos, int *stop);
  // stop - means stop another Code calls


  /* Write() must be called, if Parse() was called
      set (needWrite) if
      {
         && (was not interrupted by progress)
         && (was not interrupted in previous block)
      }

    out:
      if (*needContinue), decoder still need to continue decoding with new iteration,
         even after MTDEC_PARSE_END
      if (*canRecode), we didn't flush current block data, so we still can decode current block later.
  */
  SRes (*Write)(void *p, unsigned coderIndex,
      BoolInt needWriteToStream,
      const Byte *src, size_t srcSize, BoolInt isCross,
      // int srcFinished,
      BoolInt *needContinue,
      BoolInt *canRecode);

} IMtDecCallback2;



typedef struct CMtDec_
{
  /* input variables */
  
  size_t inBufSize;        /* size of input block */
  unsigned numThreadsMax;
  // size_t inBlockMax;
  unsigned numThreadsMax_2;

  ISeqInStreamPtr inStream;
  // const Byte *inData;
  // size_t inDataSize;

  ICompressProgressPtr progress;
  ISzAllocPtr alloc;

  IMtDecCallback2 *mtCallback;
  void *mtCallbackObject;

  
  /* internal variables */
  
  size_t allocatedBufsSize;

  BoolInt exitThread;
  WRes exitThreadWRes;

  UInt64 blockIndex;
  BoolInt isAllocError;
  BoolInt overflow;
  SRes threadingErrorSRes;

  BoolInt needContinue;

  // CAutoResetEvent finishedEvent;

  SRes readRes;
  SRes codeRes;

  BoolInt wasInterrupted;

  unsigned numStartedThreads_Limit;
  unsigned numStartedThreads;

  Byte *crossBlock;
  size_t crossStart;
  size_t crossEnd;
  UInt64 readProcessed;
  BoolInt readWasFinished;
  UInt64 inProcessed;

  unsigned filledThreadStart;
  unsigned numFilledThreads;

  #ifndef Z7_ST
  BoolInt needInterrupt;
  UInt64 interruptIndex;
  CMtProgress mtProgress;
  CMtDecThread threads[MTDEC_THREADS_MAX];
  #endif
} CMtDec;


void MtDec_Construct(CMtDec *p);
void MtDec_Destruct(CMtDec *p);

/*
MtDec_Code() returns:
  SZ_OK - in most cases
  MY_SRes_HRESULT_FROM_WRes(WRes_error) - in case of unexpected error in threading function
*/
  
SRes MtDec_Code(CMtDec *p);
Byte *MtDec_GetCrossBuff(CMtDec *p);

int MtDec_PrepareRead(CMtDec *p);
const Byte *MtDec_Read(CMtDec *p, size_t *inLim);

#endif

EXTERN_C_END

#endif
