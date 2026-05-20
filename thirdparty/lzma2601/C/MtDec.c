/* MtDec.c -- Multi-thread Decoder
2024-02-20 : Igor Pavlov : Public domain */

#include "Precomp.h"

// #define SHOW_DEBUG_INFO

// #include <stdio.h>
#include <string.h>

#ifdef SHOW_DEBUG_INFO
#include <stdio.h>
#endif

#include "MtDec.h"

#ifndef Z7_ST

#ifdef SHOW_DEBUG_INFO
#define PRF(x) x
#else
#define PRF(x)
#endif

#define PRF_STR_INT(s, d) PRF(printf("\n" s " %d\n", (unsigned)d))

void MtProgress_Init(CMtProgress *p, ICompressProgressPtr progress)
{
  p->progress = progress;
  p->res = SZ_OK;
  p->totalInSize = 0;
  p->totalOutSize = 0;
}


SRes MtProgress_Progress_ST(CMtProgress *p)
{
  if (p->res == SZ_OK && p->progress)
    if (ICompressProgress_Progress(p->progress, p->totalInSize, p->totalOutSize) != SZ_OK)
      p->res = SZ_ERROR_PROGRESS;
  return p->res;
}


SRes MtProgress_ProgressAdd(CMtProgress *p, UInt64 inSize, UInt64 outSize)
{
  SRes res;
  CriticalSection_Enter(&p->cs);
  
  p->totalInSize += inSize;
  p->totalOutSize += outSize;
  if (p->res == SZ_OK && p->progress)
    if (ICompressProgress_Progress(p->progress, p->totalInSize, p->totalOutSize) != SZ_OK)
      p->res = SZ_ERROR_PROGRESS;
  res = p->res;
  
  CriticalSection_Leave(&p->cs);
  return res;
}


SRes MtProgress_GetError(CMtProgress *p)
{
  SRes res;
  CriticalSection_Enter(&p->cs);
  res = p->res;
  CriticalSection_Leave(&p->cs);
  return res;
}


void MtProgress_SetError(CMtProgress *p, SRes res)
{
  CriticalSection_Enter(&p->cs);
  if (p->res == SZ_OK)
    p->res = res;
  CriticalSection_Leave(&p->cs);
}


#define RINOK_THREAD(x) RINOK_WRes(x)


struct CMtDecBufLink_
{
  struct CMtDecBufLink_ *next;
  void *pad[3];
};

typedef struct CMtDecBufLink_ CMtDecBufLink;

#define MTDEC__LINK_DATA_OFFSET sizeof(CMtDecBufLink)
#define MTDEC__DATA_PTR_FROM_LINK(link) ((Byte *)(link) + MTDEC__LINK_DATA_OFFSET)



static THREAD_FUNC_DECL MtDec_ThreadFunc(void *pp);


static WRes MtDecThread_CreateEvents(CMtDecThread *t)
{
  WRes wres = AutoResetEvent_OptCreate_And_Reset(&t->canWrite);
  if (wres == 0)
  {
    wres = AutoResetEvent_OptCreate_And_Reset(&t->canRead);
    if (wres == 0)
      return SZ_OK;
  }
  return wres;
}


static SRes MtDecThread_CreateAndStart(CMtDecThread *t)
{
  WRes wres = MtDecThread_CreateEvents(t);
  // wres = 17; // for test
  if (wres == 0)
  {
    if (Thread_WasCreated(&t->thread))
      return SZ_OK;
    wres = Thread_Create(&t->thread, MtDec_ThreadFunc, t);
    if (wres == 0)
      return SZ_OK;
  }
  return MY_SRes_HRESULT_FROM_WRes(wres);
}


void MtDecThread_FreeInBufs(CMtDecThread *t)
{
  if (t->inBuf)
  {
    void *link = t->inBuf;
    t->inBuf = NULL;
    do
    {
      void *next = ((CMtDecBufLink *)link)->next;
      ISzAlloc_Free(t->mtDec->alloc, link);
      link = next;
    }
    while (link);
  }
}


static void MtDecThread_CloseThread(CMtDecThread *t)
{
  if (Thread_WasCreated(&t->thread))
  {
    Event_Set(&t->canWrite); /* we can disable it. There are no threads waiting canWrite in normal cases */
    Event_Set(&t->canRead);
    Thread_Wait_Close(&t->thread);
  }

  Event_Close(&t->canRead);
  Event_Close(&t->canWrite);
}

static void MtDec_CloseThreads(CMtDec *p)
{
  unsigned i;
  for (i = 0; i < MTDEC_THREADS_MAX; i++)
    MtDecThread_CloseThread(&p->threads[i]);
}

static void MtDecThread_Destruct(CMtDecThread *t)
{
  MtDecThread_CloseThread(t);
  MtDecThread_FreeInBufs(t);
}



static SRes MtDec_GetError_Spec(CMtDec *p, UInt64 interruptIndex, BoolInt *wasInterrupted)
{
  SRes res;
  CriticalSection_Enter(&p->mtProgress.cs);
  *wasInterrupted = (p->needInterrupt && interruptIndex > p->interruptIndex);
  res = p->mtProgress.res;
  CriticalSection_Leave(&p->mtProgress.cs);
  return res;
}

static SRes MtDec_Progress_GetError_Spec(CMtDec *p, UInt64 inSize, UInt64 outSize, UInt64 interruptIndex, BoolInt *wasInterrupted)
{
  SRes res;
  CriticalSection_Enter(&p->mtProgress.cs);

  p->mtProgress.totalInSize += inSize;
  p->mtProgress.totalOutSize += outSize;
  if (p->mtProgress.res == SZ_OK && p->mtProgress.progress)
    if (ICompressProgress_Progress(p->mtProgress.progress, p->mtProgress.totalInSize, p->mtProgress.totalOutSize) != SZ_OK)
      p->mtProgress.res = SZ_ERROR_PROGRESS;

  *wasInterrupted = (p->needInterrupt && interruptIndex > p->interruptIndex);
  res = p->mtProgress.res;
  
  CriticalSection_Leave(&p->mtProgress.cs);

  return res;
}

static void MtDec_Interrupt(CMtDec *p, UInt64 interruptIndex)
{
  CriticalSection_Enter(&p->mtProgress.cs);
  if (!p->needInterrupt || interruptIndex < p->interruptIndex)
  {
    p->interruptIndex = interruptIndex;
    p->needInterrupt = True;
  }
  CriticalSection_Leave(&p->mtProgress.cs);
}

Byte *MtDec_GetCrossBuff(CMtDec *p)
{
  Byte *cr = p->crossBlock;
  if (!cr)
  {
    cr = (Byte *)ISzAlloc_Alloc(p->alloc, MTDEC__LINK_DATA_OFFSET + p->inBufSize);
    if (!cr)
      return NULL;
    p->crossBlock = cr;
  }
  return MTDEC__DATA_PTR_FROM_LINK(cr);
}


/*
  MtDec_ThreadFunc2() returns:
  0      - in all normal cases (even for stream error or memory allocation error)
  (!= 0) - WRes error return by system threading function
*/

// #define MTDEC_ProgessStep (1 << 22)
#define MTDEC_ProgessStep (1 << 0)

static WRes MtDec_ThreadFunc2(CMtDecThread *t)
{
  CMtDec *p = t->mtDec;

  PRF_STR_INT("MtDec_ThreadFunc2", t->index)

  // SetThreadAffinityMask(GetCurrentThread(), 1 << t->index);

  for (;;)
  {
    SRes res, codeRes;
    BoolInt wasInterrupted, isAllocError, overflow, finish;
    SRes threadingErrorSRes;
    BoolInt needCode, needWrite, needContinue;
    
    size_t inDataSize_Start;
    UInt64 inDataSize;
    // UInt64 inDataSize_Full;
    
    UInt64 blockIndex;

    UInt64 inPrev = 0;
    UInt64 outPrev = 0;
    UInt64 inCodePos;
    UInt64 outCodePos;
    
    Byte *afterEndData = NULL;
    size_t afterEndData_Size = 0;
    BoolInt afterEndData_IsCross = False;

    BoolInt canCreateNewThread = False;
    // CMtDecCallbackInfo parse;
    CMtDecThread *nextThread;

    PRF_STR_INT("=============== Event_Wait(&t->canRead)", t->index)

    RINOK_THREAD(Event_Wait(&t->canRead))
    if (p->exitThread)
      return 0;

    PRF_STR_INT("after Event_Wait(&t->canRead)", t->index)

    // if (t->index == 3) return 19; // for test

    blockIndex = p->blockIndex++;

    // PRF(printf("\ncanRead\n"))

    res = MtDec_Progress_GetError_Spec(p, 0, 0, blockIndex, &wasInterrupted);

    finish = p->readWasFinished;
    needCode = False;
    needWrite = False;
    isAllocError = False;
    overflow = False;

    inDataSize_Start = 0;
    inDataSize = 0;
    // inDataSize_Full = 0;

    if (res == SZ_OK && !wasInterrupted)
    {
      // if (p->inStream)
      {
        CMtDecBufLink *prev = NULL;
        CMtDecBufLink *link = (CMtDecBufLink *)t->inBuf;
        size_t crossSize = p->crossEnd - p->crossStart;

        PRF(printf("\ncrossSize = %d\n", crossSize));

        for (;;)
        {
          if (!link)
          {
            link = (CMtDecBufLink *)ISzAlloc_Alloc(p->alloc, MTDEC__LINK_DATA_OFFSET + p->inBufSize);
            if (!link)
            {
              finish = True;
              // p->allocError_for_Read_BlockIndex = blockIndex;
              isAllocError = True;
              break;
            }
            link->next = NULL;
            if (prev)
            {
              // static unsigned g_num = 0;
              // printf("\n%6d : %x", ++g_num, (unsigned)(size_t)((Byte *)link - (Byte *)prev));
              prev->next = link;
            }
            else
              t->inBuf = (void *)link;
          }

          {
            Byte *data = MTDEC__DATA_PTR_FROM_LINK(link);
            Byte *parseData = data;
            size_t size;

            if (crossSize != 0)
            {
              inDataSize = crossSize;
              // inDataSize_Full = inDataSize;
              inDataSize_Start = crossSize;
              size = crossSize;
              parseData = MTDEC__DATA_PTR_FROM_LINK(p->crossBlock) + p->crossStart;
              PRF(printf("\ncross : crossStart = %7d  crossEnd = %7d finish = %1d",
                  (int)p->crossStart, (int)p->crossEnd, (int)finish));
            }
            else
            {
              size = p->inBufSize;
              
              res = SeqInStream_ReadMax(p->inStream, data, &size);
              
              // size = 10; // test

              inDataSize += size;
              // inDataSize_Full = inDataSize;
              if (!prev)
                inDataSize_Start = size;

              p->readProcessed += size;
              finish = (size != p->inBufSize);
              if (finish)
                p->readWasFinished = True;
              
              // res = E_INVALIDARG; // test

              if (res != SZ_OK)
              {
                // PRF(printf("\nRead error = %d\n", res))
                // we want to decode all data before error
                p->readRes = res;
                // p->readError_BlockIndex = blockIndex;
                p->readWasFinished = True;
                finish = True;
                res = SZ_OK;
                // break;
              }

              if (inDataSize - inPrev >= MTDEC_ProgessStep)
              {
                res = MtDec_Progress_GetError_Spec(p, 0, 0, blockIndex, &wasInterrupted);
                if (res != SZ_OK || wasInterrupted)
                  break;
                inPrev = inDataSize;
              }
            }

            {
              CMtDecCallbackInfo parse;

              parse.startCall = (prev == NULL);
              parse.src = parseData;
              parse.srcSize = size;
              parse.srcFinished = finish;
              parse.canCreateNewThread = True;

              PRF(printf("\nParse size = %d\n", (unsigned)size));

              p->mtCallback->Parse(p->mtCallbackObject, t->index, &parse);

              PRF(printf("   Parse processed = %d, state = %d \n", (unsigned)parse.srcSize, (unsigned)parse.state));

              needWrite = True;
              canCreateNewThread = parse.canCreateNewThread;

              // printf("\n\n%12I64u %12I64u", (UInt64)p->mtProgress.totalInSize, (UInt64)p->mtProgress.totalOutSize);
              
              if (
                  // parseRes != SZ_OK ||
                  // inDataSize - (size - parse.srcSize) > p->inBlockMax
                  // ||
                  parse.state == MTDEC_PARSE_OVERFLOW
                  // || wasInterrupted
                  )
              {
                // Overflow or Parse error - switch from MT decoding to ST decoding
                finish = True;
                overflow = True;

                {
                  PRF(printf("\n Overflow"));
                  // PRF(printf("\nisBlockFinished = %d", (unsigned)parse.blockWasFinished));
                  PRF(printf("\n inDataSize = %d", (unsigned)inDataSize));
                }
                
                if (crossSize != 0)
                  memcpy(data, parseData, size);
                p->crossStart = 0;
                p->crossEnd = 0;
                break;
              }

              if (crossSize != 0)
              {
                memcpy(data, parseData, parse.srcSize);
                p->crossStart += parse.srcSize;
              }

              if (parse.state != MTDEC_PARSE_CONTINUE || finish)
              {
                // we don't need to parse in current thread anymore

                if (parse.state == MTDEC_PARSE_END)
                  finish = True;

                needCode = True;
                // p->crossFinished = finish;

                if (parse.srcSize == size)
                {
                  // full parsed - no cross transfer
                  p->crossStart = 0;
                  p->crossEnd = 0;
                  break;
                }

                if (parse.state == MTDEC_PARSE_END)
                {
                  afterEndData = parseData + parse.srcSize;
                  afterEndData_Size = size - parse.srcSize;
                  if (crossSize != 0)
                    afterEndData_IsCross = True;
                  // we reduce data size to required bytes (parsed only)
                  inDataSize -= afterEndData_Size;
                  if (!prev)
                    inDataSize_Start = parse.srcSize;
                  break;
                }

                {
                  // partial parsed - need cross transfer
                  if (crossSize != 0)
                    inDataSize = parse.srcSize; // it's only parsed now
                  else
                  {
                    // partial parsed - is not in initial cross block - we need to copy new data to cross block
                    Byte *cr = MtDec_GetCrossBuff(p);
                    if (!cr)
                    {
                      {
                        PRF(printf("\ncross alloc error error\n"));
                        // res = SZ_ERROR_MEM;
                        finish = True;
                        // p->allocError_for_Read_BlockIndex = blockIndex;
                        isAllocError = True;
                        break;
                      }
                    }

                    {
                      size_t crSize = size - parse.srcSize;
                      inDataSize -= crSize;
                      p->crossEnd = crSize;
                      p->crossStart = 0;
                      memcpy(cr, parseData + parse.srcSize, crSize);
                    }
                  }

                  // inDataSize_Full = inDataSize;
                  if (!prev)
                    inDataSize_Start = parse.srcSize; // it's partial size (parsed only)

                  finish = False;
                  break;
                }
              }

              if (parse.srcSize != size)
              {
                res = SZ_ERROR_FAIL;
                PRF(printf("\nfinished error SZ_ERROR_FAIL = %d\n", res));
                break;
              }
            }
          }
          
          prev = link;
          link = link->next;

          if (crossSize != 0)
          {
            crossSize = 0;
            p->crossStart = 0;
            p->crossEnd = 0;
          }
        }
      }

      if (res == SZ_OK)
        res = MtDec_GetError_Spec(p, blockIndex, &wasInterrupted);
    }

    codeRes = SZ_OK;

    if (res == SZ_OK && needCode && !wasInterrupted)
    {
      codeRes = p->mtCallback->PreCode(p->mtCallbackObject, t->index);
      if (codeRes != SZ_OK)
      {
        needCode = False;
        finish = True;
        // SZ_ERROR_MEM is expected error here.
        //   if (codeRes == SZ_ERROR_MEM) - we will try single-thread decoding later.
        //   if (codeRes != SZ_ERROR_MEM) - we can stop decoding or try single-thread decoding.
      }
    }
    
    if (res != SZ_OK || wasInterrupted)
      finish = True;
    
    nextThread = NULL;
    threadingErrorSRes = SZ_OK;

    if (!finish)
    {
      if (p->numStartedThreads < p->numStartedThreads_Limit && canCreateNewThread)
      {
        SRes res2 = MtDecThread_CreateAndStart(&p->threads[p->numStartedThreads]);
        if (res2 == SZ_OK)
        {
          // if (p->numStartedThreads % 1000 == 0) PRF(printf("\n numStartedThreads=%d\n", p->numStartedThreads));
          p->numStartedThreads++;
        }
        else
        {
          PRF(printf("\nERROR: numStartedThreads=%d\n", p->numStartedThreads));
          if (p->numStartedThreads == 1)
          {
            // if only one thread is possible, we leave muti-threading code
            finish = True;
            needCode = False;
            threadingErrorSRes = res2;
          }
          else
            p->numStartedThreads_Limit = p->numStartedThreads;
        }
      }
      
      if (!finish)
      {
        unsigned nextIndex = t->index + 1;
        nextThread = &p->threads[nextIndex >= p->numStartedThreads ? 0 : nextIndex];
        RINOK_THREAD(Event_Set(&nextThread->canRead))
        // We have started executing for new iteration (with next thread)
        // And that next thread now is responsible for possible exit from decoding (threading_code)
      }
    }

    // each call of Event_Set(&nextThread->canRead) must be followed by call of Event_Set(&nextThread->canWrite)
    // if ( !finish ) we must call Event_Set(&nextThread->canWrite) in any case
    // if (  finish ) we switch to single-thread mode and there are 2 ways at the end of current iteration (current block):
    //   - if (needContinue) after Write(&needContinue), we restore decoding with new iteration
    //   - otherwise we stop decoding and exit from MtDec_ThreadFunc2()

    // Don't change (finish) variable in the further code


    // ---------- CODE ----------

    inPrev = 0;
    outPrev = 0;
    inCodePos = 0;
    outCodePos = 0;

    if (res == SZ_OK && needCode && codeRes == SZ_OK)
    {
      BoolInt isStartBlock = True;
      CMtDecBufLink *link = (CMtDecBufLink *)t->inBuf;

      for (;;)
      {
        size_t inSize;
        int stop;

        if (isStartBlock)
          inSize = inDataSize_Start;
        else
        {
          UInt64 rem = inDataSize - inCodePos;
          inSize = p->inBufSize;
          if (inSize > rem)
            inSize = (size_t)rem;
        }

        inCodePos += inSize;
        stop = True;

        codeRes = p->mtCallback->Code(p->mtCallbackObject, t->index,
            (const Byte *)MTDEC__DATA_PTR_FROM_LINK(link), inSize,
            (inCodePos == inDataSize), // srcFinished
            &inCodePos, &outCodePos, &stop);
        
        if (codeRes != SZ_OK)
        {
          PRF(printf("\nCode Interrupt error = %x\n", codeRes));
          // we interrupt only later blocks
          MtDec_Interrupt(p, blockIndex);
          break;
        }

        if (stop || inCodePos == inDataSize)
          break;
  
        {
          const UInt64 inDelta = inCodePos - inPrev;
          const UInt64 outDelta = outCodePos - outPrev;
          if (inDelta >= MTDEC_ProgessStep || outDelta >= MTDEC_ProgessStep)
          {
            // Sleep(1);
            res = MtDec_Progress_GetError_Spec(p, inDelta, outDelta, blockIndex, &wasInterrupted);
            if (res != SZ_OK || wasInterrupted)
              break;
            inPrev = inCodePos;
            outPrev = outCodePos;
          }
        }

        link = link->next;
        isStartBlock = False;
      }
    }


    // ---------- WRITE ----------
   
    RINOK_THREAD(Event_Wait(&t->canWrite))

  {
    BoolInt isErrorMode = False;
    BoolInt canRecode = True;
    BoolInt needWriteToStream = needWrite;

    if (p->exitThread) return 0; // it's never executed in normal cases

    if (p->wasInterrupted)
      wasInterrupted = True;
    else
    {
      if (codeRes != SZ_OK) // || !needCode // check it !!!
      {
        p->wasInterrupted = True;
        p->codeRes = codeRes;
        if (codeRes == SZ_ERROR_MEM)
          isAllocError = True;
      }
      
      if (threadingErrorSRes)
      {
        p->wasInterrupted = True;
        p->threadingErrorSRes = threadingErrorSRes;
        needWriteToStream = False;
      }
      if (isAllocError)
      {
        p->wasInterrupted = True;
        p->isAllocError = True;
        needWriteToStream = False;
      }
      if (overflow)
      {
        p->wasInterrupted = True;
        p->overflow = True;
        needWriteToStream = False;
      }
    }

    if (needCode)
    {
      if (wasInterrupted)
      {
        inCodePos = 0;
        outCodePos = 0;
      }
      {
        const UInt64 inDelta = inCodePos - inPrev;
        const UInt64 outDelta = outCodePos - outPrev;
        // if (inDelta != 0 || outDelta != 0)
        res = MtProgress_ProgressAdd(&p->mtProgress, inDelta, outDelta);
      }
    }

    needContinue = (!finish);

    // if (res == SZ_OK && needWrite && !wasInterrupted)
    if (needWrite)
    {
      // p->inProcessed += inCodePos;

      PRF(printf("\n--Write afterSize = %d\n", (unsigned)afterEndData_Size));

      res = p->mtCallback->Write(p->mtCallbackObject, t->index,
          res == SZ_OK && needWriteToStream && !wasInterrupted, // needWrite
          afterEndData, afterEndData_Size, afterEndData_IsCross,
          &needContinue,
          &canRecode);

      // res = SZ_ERROR_FAIL; // for test

      PRF(printf("\nAfter Write needContinue = %d\n", (unsigned)needContinue));
      PRF(printf("\nprocessed = %d\n", (unsigned)p->inProcessed));

      if (res != SZ_OK)
      {
        PRF(printf("\nWrite error = %d\n", res));
        isErrorMode = True;
        p->wasInterrupted = True;
      }
      if (res != SZ_OK
          || (!needContinue && !finish))
      {
        PRF(printf("\nWrite Interrupt error = %x\n", res));
        MtDec_Interrupt(p, blockIndex);
      }
    }

    if (canRecode)
    if (!needCode
        || res != SZ_OK
        || p->wasInterrupted
        || codeRes != SZ_OK
        || wasInterrupted
        || p->numFilledThreads != 0
        || isErrorMode)
    {
      if (p->numFilledThreads == 0)
        p->filledThreadStart = t->index;
      if (inDataSize != 0 || !finish)
      {
        t->inDataSize_Start = inDataSize_Start;
        t->inDataSize = inDataSize;
        p->numFilledThreads++;
      }
      PRF(printf("\np->numFilledThreads = %d\n", p->numFilledThreads));
      PRF(printf("p->filledThreadStart = %d\n", p->filledThreadStart));
    }

    if (!finish)
    {
      RINOK_THREAD(Event_Set(&nextThread->canWrite))
    }
    else
    {
      if (needContinue)
      {
        // we restore decoding with new iteration
        RINOK_THREAD(Event_Set(&p->threads[0].canWrite))
      }
      else
      {
        // we exit from decoding
        if (t->index == 0)
          return SZ_OK;
        p->exitThread = True;
      }
      RINOK_THREAD(Event_Set(&p->threads[0].canRead))
    }
  }
  }
}

#ifdef _WIN32
#define USE_ALLOCA
#endif

#ifdef USE_ALLOCA
#ifdef _WIN32
#include <malloc.h>
#else
#include <stdlib.h>
#endif
#endif


typedef
  #ifdef _WIN32
    UINT_PTR
  #elif 1
    uintptr_t
  #else
    ptrdiff_t
  #endif
    MY_uintptr_t;

static THREAD_FUNC_DECL MtDec_ThreadFunc1(void *pp)
{
  WRes res;

  CMtDecThread *t = (CMtDecThread *)pp;
  CMtDec *p;

  // fprintf(stdout, "\n%d = %p\n", t->index, &t);

  res = MtDec_ThreadFunc2(t);
  p = t->mtDec;
  if (res == 0)
    return (THREAD_FUNC_RET_TYPE)(MY_uintptr_t)p->exitThreadWRes;
  {
    // it's unexpected situation for some threading function error
    if (p->exitThreadWRes == 0)
      p->exitThreadWRes = res;
    PRF(printf("\nthread exit error = %d\n", res));
    p->exitThread = True;
    Event_Set(&p->threads[0].canRead);
    Event_Set(&p->threads[0].canWrite);
    MtProgress_SetError(&p->mtProgress, MY_SRes_HRESULT_FROM_WRes(res));
  }
  return (THREAD_FUNC_RET_TYPE)(MY_uintptr_t)res;
}

static Z7_NO_INLINE THREAD_FUNC_DECL MtDec_ThreadFunc(void *pp)
{
  #ifdef USE_ALLOCA
  CMtDecThread *t = (CMtDecThread *)pp;
  // fprintf(stderr, "\n%d = %p - before", t->index, &t);
  t->allocaPtr = alloca(t->index * 128);
  #endif
  return MtDec_ThreadFunc1(pp);
}


int MtDec_PrepareRead(CMtDec *p)
{
  if (p->crossBlock && p->crossStart == p->crossEnd)
  {
    ISzAlloc_Free(p->alloc, p->crossBlock);
    p->crossBlock = NULL;
  }
    
  {
    unsigned i;
    for (i = 0; i < MTDEC_THREADS_MAX; i++)
      if (i > p->numStartedThreads
          || p->numFilledThreads <=
            (i >= p->filledThreadStart ?
              i - p->filledThreadStart :
              i + p->numStartedThreads - p->filledThreadStart))
        MtDecThread_FreeInBufs(&p->threads[i]);
  }

  return (p->numFilledThreads != 0) || (p->crossStart != p->crossEnd);
}

    
const Byte *MtDec_Read(CMtDec *p, size_t *inLim)
{
  while (p->numFilledThreads != 0)
  {
    CMtDecThread *t = &p->threads[p->filledThreadStart];
    
    if (*inLim != 0)
    {
      {
        void *link = t->inBuf;
        void *next = ((CMtDecBufLink *)link)->next;
        ISzAlloc_Free(p->alloc, link);
        t->inBuf = next;
      }
      
      if (t->inDataSize == 0)
      {
        MtDecThread_FreeInBufs(t);
        if (--p->numFilledThreads == 0)
          break;
        if (++p->filledThreadStart == p->numStartedThreads)
          p->filledThreadStart = 0;
        t = &p->threads[p->filledThreadStart];
      }
    }
    
    {
      size_t lim = t->inDataSize_Start;
      if (lim != 0)
        t->inDataSize_Start = 0;
      else
      {
        UInt64 rem = t->inDataSize;
        lim = p->inBufSize;
        if (lim > rem)
          lim = (size_t)rem;
      }
      t->inDataSize -= lim;
      *inLim = lim;
      return (const Byte *)MTDEC__DATA_PTR_FROM_LINK(t->inBuf);
    }
  }

  {
    size_t crossSize = p->crossEnd - p->crossStart;
    if (crossSize != 0)
    {
      const Byte *data = MTDEC__DATA_PTR_FROM_LINK(p->crossBlock) + p->crossStart;
      *inLim = crossSize;
      p->crossStart = 0;
      p->crossEnd = 0;
      return data;
    }
    *inLim = 0;
    if (p->crossBlock)
    {
      ISzAlloc_Free(p->alloc, p->crossBlock);
      p->crossBlock = NULL;
    }
    return NULL;
  }
}


void MtDec_Construct(CMtDec *p)
{
  unsigned i;
  
  p->inBufSize = (size_t)1 << 18;

  p->numThreadsMax = 0;

  p->inStream = NULL;
  
  // p->inData = NULL;
  // p->inDataSize = 0;

  p->crossBlock = NULL;
  p->crossStart = 0;
  p->crossEnd = 0;

  p->numFilledThreads = 0;

  p->progress = NULL;
  p->alloc = NULL;

  p->mtCallback = NULL;
  p->mtCallbackObject = NULL;

  p->allocatedBufsSize = 0;

  for (i = 0; i < MTDEC_THREADS_MAX; i++)
  {
    CMtDecThread *t = &p->threads[i];
    t->mtDec = p;
    t->index = i;
    t->inBuf = NULL;
    Event_Construct(&t->canRead);
    Event_Construct(&t->canWrite);
    Thread_CONSTRUCT(&t->thread)
  }

  // Event_Construct(&p->finishedEvent);

  CriticalSection_Init(&p->mtProgress.cs);
}


static void MtDec_Free(CMtDec *p)
{
  unsigned i;

  p->exitThread = True;

  for (i = 0; i < MTDEC_THREADS_MAX; i++)
    MtDecThread_Destruct(&p->threads[i]);

  // Event_Close(&p->finishedEvent);

  if (p->crossBlock)
  {
    ISzAlloc_Free(p->alloc, p->crossBlock);
    p->crossBlock = NULL;
  }
}


void MtDec_Destruct(CMtDec *p)
{
  MtDec_Free(p);

  CriticalSection_Delete(&p->mtProgress.cs);
}


SRes MtDec_Code(CMtDec *p)
{
  unsigned i;

  p->inProcessed = 0;

  p->blockIndex = 1; // it must be larger than not_defined index (0)
  p->isAllocError = False;
  p->overflow = False;
  p->threadingErrorSRes = SZ_OK;

  p->needContinue = True;

  p->readWasFinished = False;
  p->needInterrupt = False;
  p->interruptIndex = (UInt64)(Int64)-1;

  p->readProcessed = 0;
  p->readRes = SZ_OK;
  p->codeRes = SZ_OK;
  p->wasInterrupted = False;

  p->crossStart = 0;
  p->crossEnd = 0;

  p->filledThreadStart = 0;
  p->numFilledThreads = 0;

  {
    unsigned numThreads = p->numThreadsMax;
    if (numThreads > MTDEC_THREADS_MAX)
      numThreads = MTDEC_THREADS_MAX;
    p->numStartedThreads_Limit = numThreads;
    p->numStartedThreads = 0;
  }

  if (p->inBufSize != p->allocatedBufsSize)
  {
    for (i = 0; i < MTDEC_THREADS_MAX; i++)
    {
      CMtDecThread *t = &p->threads[i];
      if (t->inBuf)
        MtDecThread_FreeInBufs(t);
    }
    if (p->crossBlock)
    {
      ISzAlloc_Free(p->alloc, p->crossBlock);
      p->crossBlock = NULL;
    }

    p->allocatedBufsSize = p->inBufSize;
  }

  MtProgress_Init(&p->mtProgress, p->progress);

  // RINOK_THREAD(AutoResetEvent_OptCreate_And_Reset(&p->finishedEvent))
  p->exitThread = False;
  p->exitThreadWRes = 0;

  {
    WRes wres;
    SRes sres;
    CMtDecThread *nextThread = &p->threads[p->numStartedThreads++];
    // wres = MtDecThread_CreateAndStart(nextThread);
    wres = MtDecThread_CreateEvents(nextThread);
    if (wres == 0) { wres = Event_Set(&nextThread->canWrite);
    if (wres == 0) { wres = Event_Set(&nextThread->canRead);
    if (wres == 0) { THREAD_FUNC_RET_TYPE res = MtDec_ThreadFunc(nextThread);
    wres = (WRes)(MY_uintptr_t)res;
    if (wres != 0)
    {
      p->needContinue = False;
      MtDec_CloseThreads(p);
    }}}}

    // wres = 17; // for test
    // wres = Event_Wait(&p->finishedEvent);

    sres = MY_SRes_HRESULT_FROM_WRes(wres);

    if (sres != 0)
      p->threadingErrorSRes = sres;

    if (
        // wres == 0
        // wres != 0
        // || p->mtc.codeRes == SZ_ERROR_MEM
        p->isAllocError
        || p->threadingErrorSRes != SZ_OK
        || p->overflow)
    {
      // p->needContinue = True;
    }
    else
      p->needContinue = False;
    
    if (p->needContinue)
      return SZ_OK;

    // if (sres != SZ_OK)
    return sres;
    // return SZ_ERROR_FAIL;
  }
}

#endif

#undef PRF
