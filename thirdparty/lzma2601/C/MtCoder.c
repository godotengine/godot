/* MtCoder.c -- Multi-thread Coder
: Igor Pavlov : Public domain */

#include "Precomp.h"

#include "MtCoder.h"

#ifndef Z7_ST

static SRes MtProgressThunk_Progress(ICompressProgressPtr pp, UInt64 inSize, UInt64 outSize)
{
  Z7_CONTAINER_FROM_VTBL_TO_DECL_VAR_pp_vt_p(CMtProgressThunk)
  UInt64 inSize2 = 0;
  UInt64 outSize2 = 0;
  if (inSize != (UInt64)(Int64)-1)
  {
    inSize2 = inSize - p->inSize;
    p->inSize = inSize;
  }
  if (outSize != (UInt64)(Int64)-1)
  {
    outSize2 = outSize - p->outSize;
    p->outSize = outSize;
  }
  return MtProgress_ProgressAdd(p->mtProgress, inSize2, outSize2);
}


void MtProgressThunk_CreateVTable(CMtProgressThunk *p)
{
  p->vt.Progress = MtProgressThunk_Progress;
}



#define RINOK_THREAD(x) { if ((x) != 0) return SZ_ERROR_THREAD; }


static THREAD_FUNC_DECL ThreadFunc(void *pp);


static SRes MtCoderThread_CreateAndStart(CMtCoderThread *t
#ifdef _WIN32
    , CMtCoder * const mtc
#endif
    )
{
  WRes wres = AutoResetEvent_OptCreate_And_Reset(&t->startEvent);
  // printf("\n====== MtCoderThread_CreateAndStart : \n");
  if (wres == 0)
  {
    t->stop = False;
    if (!Thread_WasCreated(&t->thread))
    {
#ifdef _WIN32
      if (mtc->numThreadGroups)
        wres = Thread_Create_With_Group(&t->thread, ThreadFunc, t,
            ThreadNextGroup_GetNext(&mtc->nextGroup), // group
            0); // affinityMask
      else
#endif
        wres = Thread_Create(&t->thread, ThreadFunc, t);
    }
    if (wres == 0)
      wres = Event_Set(&t->startEvent);
  }
  if (wres == 0)
    return SZ_OK;
  return MY_SRes_HRESULT_FROM_WRes(wres);
}


Z7_FORCE_INLINE
static void MtCoderThread_Destruct(CMtCoderThread *t)
{
  if (Thread_WasCreated(&t->thread))
  {
    t->stop = 1;
    Event_Set(&t->startEvent);
    Thread_Wait_Close(&t->thread);
  }

  Event_Close(&t->startEvent);

  if (t->inBuf)
  {
    ISzAlloc_Free(t->mtCoder->allocBig, t->inBuf);
    t->inBuf = NULL;
  }
}




/*
  ThreadFunc2() returns:
  SZ_OK           - in all normal cases (even for stream error or memory allocation error)
  SZ_ERROR_THREAD - in case of failure in system synch function
*/

static SRes ThreadFunc2(CMtCoderThread *t)
{
  CMtCoder * const mtc = t->mtCoder;

  for (;;)
  {
    unsigned bi;
    SRes res;
    SRes res2;
    BoolInt finished;
    unsigned bufIndex;
    size_t size;
    const Byte *inData;
    UInt64 readProcessed = 0;
    
    RINOK_THREAD(Event_Wait(&mtc->readEvent))

    /* after Event_Wait(&mtc->readEvent) we must call Event_Set(&mtc->readEvent) in any case to unlock another threads */

    if (mtc->stopReading)
    {
      return Event_Set(&mtc->readEvent) == 0 ? SZ_OK : SZ_ERROR_THREAD;
    }

    res = MtProgress_GetError(&mtc->mtProgress);
    
    size = 0;
    inData = NULL;
    finished = True;

    if (res == SZ_OK)
    {
      size = mtc->blockSize;
      if (mtc->inStream)
      {
        if (!t->inBuf)
        {
          t->inBuf = (Byte *)ISzAlloc_Alloc(mtc->allocBig, mtc->blockSize);
          if (!t->inBuf)
            res = SZ_ERROR_MEM;
        }
        if (res == SZ_OK)
        {
          res = SeqInStream_ReadMax(mtc->inStream, t->inBuf, &size);
          readProcessed = mtc->readProcessed + size;
          mtc->readProcessed = readProcessed;
        }
        if (res != SZ_OK)
        {
          mtc->readRes = res;
          /* after reading error - we can stop encoding of previous blocks */
          MtProgress_SetError(&mtc->mtProgress, res);
        }
        else
          finished = (size != mtc->blockSize);
      }
      else
      {
        size_t rem;
        readProcessed = mtc->readProcessed;
        rem = mtc->inDataSize - (size_t)readProcessed;
        if (size > rem)
          size = rem;
        inData = mtc->inData + (size_t)readProcessed;
        readProcessed += size;
        mtc->readProcessed = readProcessed;
        finished = (mtc->inDataSize == (size_t)readProcessed);
      }
    }

    /* we must get some block from blocksSemaphore before Event_Set(&mtc->readEvent) */

    res2 = SZ_OK;

    if (Semaphore_Wait(&mtc->blocksSemaphore) != 0)
    {
      res2 = SZ_ERROR_THREAD;
      if (res == SZ_OK)
      {
        res = res2;
        // MtProgress_SetError(&mtc->mtProgress, res);
      }
    }

    bi = mtc->blockIndex;

    if (++mtc->blockIndex >= mtc->numBlocksMax)
      mtc->blockIndex = 0;

    bufIndex = (unsigned)(int)-1;

    if (res == SZ_OK)
      res = MtProgress_GetError(&mtc->mtProgress);

    if (res != SZ_OK)
      finished = True;

    if (!finished)
    {
      if (mtc->numStartedThreads < mtc->numStartedThreadsLimit
          && mtc->expectedDataSize != readProcessed)
      {
        res = MtCoderThread_CreateAndStart(&mtc->threads[mtc->numStartedThreads]
#ifdef _WIN32
            , mtc
#endif
            );
        if (res == SZ_OK)
          mtc->numStartedThreads++;
        else
        {
          MtProgress_SetError(&mtc->mtProgress, res);
          finished = True;
        }
      }
    }

    if (finished)
      mtc->stopReading = True;

    RINOK_THREAD(Event_Set(&mtc->readEvent))

    if (res2 != SZ_OK)
      return res2;

    if (res == SZ_OK)
    {
      CriticalSection_Enter(&mtc->cs);
      bufIndex = mtc->freeBlockHead;
      mtc->freeBlockHead = mtc->freeBlockList[bufIndex];
      CriticalSection_Leave(&mtc->cs);
      
      res = mtc->mtCallback->Code(mtc->mtCallbackObject, t->index, bufIndex,
          mtc->inStream ? t->inBuf : inData, size, finished);
      
      // MtProgress_Reinit(&mtc->mtProgress, t->index);

      if (res != SZ_OK)
        MtProgress_SetError(&mtc->mtProgress, res);
    }

    {
      CMtCoderBlock * const block = &mtc->blocks[bi];
      block->res = res;
      block->bufIndex = bufIndex;
      block->finished = finished;
    }
    
    #ifdef MTCODER_USE_WRITE_THREAD
      RINOK_THREAD(Event_Set(&mtc->writeEvents[bi]))
    #else
    {
      unsigned wi;
      {
        CriticalSection_Enter(&mtc->cs);
        wi = mtc->writeIndex;
        if (wi == bi)
          mtc->writeIndex = (unsigned)(int)-1;
        else
          mtc->ReadyBlocks[bi] = True;
        CriticalSection_Leave(&mtc->cs);
      }

      if (wi != bi)
      {
        if (res != SZ_OK || finished)
          return 0;
        continue;
      }

      if (mtc->writeRes != SZ_OK)
        res = mtc->writeRes;

      for (;;)
      {
        if (res == SZ_OK && bufIndex != (unsigned)(int)-1)
        {
          res = mtc->mtCallback->Write(mtc->mtCallbackObject, bufIndex);
          if (res != SZ_OK)
          {
            mtc->writeRes = res;
            MtProgress_SetError(&mtc->mtProgress, res);
          }
        }

        if (++wi >= mtc->numBlocksMax)
          wi = 0;
        {
          BoolInt isReady;

          CriticalSection_Enter(&mtc->cs);
          
          if (bufIndex != (unsigned)(int)-1)
          {
            mtc->freeBlockList[bufIndex] = mtc->freeBlockHead;
            mtc->freeBlockHead = bufIndex;
          }
          
          isReady = mtc->ReadyBlocks[wi];
          
          if (isReady)
            mtc->ReadyBlocks[wi] = False;
          else
            mtc->writeIndex = wi;
          
          CriticalSection_Leave(&mtc->cs);

          RINOK_THREAD(Semaphore_Release1(&mtc->blocksSemaphore))

          if (!isReady)
            break;
        }

        {
          CMtCoderBlock *block = &mtc->blocks[wi];
          if (res == SZ_OK && block->res != SZ_OK)
            res = block->res;
          bufIndex = block->bufIndex;
          finished = block->finished;
        }
      }
    }
    #endif
      
    if (finished || res != SZ_OK)
      return 0;
  }
}


static THREAD_FUNC_DECL ThreadFunc(void *pp)
{
  CMtCoderThread * const t = (CMtCoderThread *)pp;
  for (;;)
  {
    if (Event_Wait(&t->startEvent) != 0)
      return (THREAD_FUNC_RET_TYPE)SZ_ERROR_THREAD;
    if (t->stop)
      return 0;
    {
      const SRes res = ThreadFunc2(t);
      CMtCoder *mtc = t->mtCoder;
      if (res != SZ_OK)
      {
        MtProgress_SetError(&mtc->mtProgress, res);
      }
      
      #ifndef MTCODER_USE_WRITE_THREAD
      {
        const unsigned numFinished = (unsigned)InterlockedIncrement(&mtc->numFinishedThreads);
        if (numFinished == mtc->numStartedThreads)
          if (Event_Set(&mtc->finishedEvent) != 0)
            return (THREAD_FUNC_RET_TYPE)SZ_ERROR_THREAD;
      }
      #endif
    }
  }
}



void MtCoder_Construct(CMtCoder *p)
{
  unsigned i;
  
  p->blockSize = 0;
  p->numThreadsMax = 0;
  p->numThreadGroups = 0;
  p->expectedDataSize = (UInt64)(Int64)-1;

  p->inStream = NULL;
  p->inData = NULL;
  p->inDataSize = 0;

  p->progress = NULL;
  p->allocBig = NULL;

  p->mtCallback = NULL;
  p->mtCallbackObject = NULL;

  p->allocatedBufsSize = 0;

  Event_Construct(&p->readEvent);
  Semaphore_Construct(&p->blocksSemaphore);

  for (i = 0; i < MTCODER_THREADS_MAX; i++)
  {
    CMtCoderThread *t = &p->threads[i];
    t->mtCoder = p;
    t->index = i;
    t->inBuf = NULL;
    t->stop = False;
    Event_Construct(&t->startEvent);
    Thread_CONSTRUCT(&t->thread)
  }

  #ifdef MTCODER_USE_WRITE_THREAD
    for (i = 0; i < MTCODER_BLOCKS_MAX; i++)
      Event_Construct(&p->writeEvents[i]);
  #else
    Event_Construct(&p->finishedEvent);
  #endif

  CriticalSection_Init(&p->cs);
  CriticalSection_Init(&p->mtProgress.cs);
}




static void MtCoder_Free(CMtCoder *p)
{
  unsigned i;

  /*
  p->stopReading = True;
  if (Event_IsCreated(&p->readEvent))
    Event_Set(&p->readEvent);
  */

  for (i = 0; i < MTCODER_THREADS_MAX; i++)
    MtCoderThread_Destruct(&p->threads[i]);

  Event_Close(&p->readEvent);
  Semaphore_Close(&p->blocksSemaphore);

  #ifdef MTCODER_USE_WRITE_THREAD
    for (i = 0; i < MTCODER_BLOCKS_MAX; i++)
      Event_Close(&p->writeEvents[i]);
  #else
    Event_Close(&p->finishedEvent);
  #endif
}


void MtCoder_Destruct(CMtCoder *p)
{
  MtCoder_Free(p);

  CriticalSection_Delete(&p->cs);
  CriticalSection_Delete(&p->mtProgress.cs);
}


SRes MtCoder_Code(CMtCoder *p)
{
  unsigned numThreads = p->numThreadsMax;
  unsigned numBlocksMax;
  unsigned i;
  SRes res = SZ_OK;

  // printf("\n====== MtCoder_Code : \n");

  if (numThreads > MTCODER_THREADS_MAX)
      numThreads = MTCODER_THREADS_MAX;
  numBlocksMax = MTCODER_GET_NUM_BLOCKS_FROM_THREADS(numThreads);
  
  if (p->blockSize < ((UInt32)1 << 26)) numBlocksMax++;
  if (p->blockSize < ((UInt32)1 << 24)) numBlocksMax++;
  if (p->blockSize < ((UInt32)1 << 22)) numBlocksMax++;

  if (numBlocksMax > MTCODER_BLOCKS_MAX)
      numBlocksMax = MTCODER_BLOCKS_MAX;

  if (p->blockSize != p->allocatedBufsSize)
  {
    for (i = 0; i < MTCODER_THREADS_MAX; i++)
    {
      CMtCoderThread *t = &p->threads[i];
      if (t->inBuf)
      {
        ISzAlloc_Free(p->allocBig, t->inBuf);
        t->inBuf = NULL;
      }
    }
    p->allocatedBufsSize = p->blockSize;
  }

  p->readRes = SZ_OK;

  MtProgress_Init(&p->mtProgress, p->progress);

  #ifdef MTCODER_USE_WRITE_THREAD
    for (i = 0; i < numBlocksMax; i++)
    {
      RINOK_THREAD(AutoResetEvent_OptCreate_And_Reset(&p->writeEvents[i]))
    }
  #else
    RINOK_THREAD(AutoResetEvent_OptCreate_And_Reset(&p->finishedEvent))
  #endif

  {
    RINOK_THREAD(AutoResetEvent_OptCreate_And_Reset(&p->readEvent))
    RINOK_THREAD(Semaphore_OptCreateInit(&p->blocksSemaphore, (UInt32)numBlocksMax, (UInt32)numBlocksMax))
  }

  for (i = 0; i < MTCODER_BLOCKS_MAX - 1; i++)
    p->freeBlockList[i] = i + 1;
  p->freeBlockList[MTCODER_BLOCKS_MAX - 1] = (unsigned)(int)-1;
  p->freeBlockHead = 0;

  p->readProcessed = 0;
  p->blockIndex = 0;
  p->numBlocksMax = numBlocksMax;
  p->stopReading = False;

  #ifndef MTCODER_USE_WRITE_THREAD
    p->writeIndex = 0;
    p->writeRes = SZ_OK;
    for (i = 0; i < MTCODER_BLOCKS_MAX; i++)
      p->ReadyBlocks[i] = False;
    p->numFinishedThreads = 0;
  #endif

  p->numStartedThreadsLimit = numThreads;
  p->numStartedThreads = 0;
  ThreadNextGroup_Init(&p->nextGroup, p->numThreadGroups, 0); // startGroup

  // for (i = 0; i < numThreads; i++)
  {
    // here we create new thread for first block.
    // And each new thread will create another new thread after block reading
    // until numStartedThreadsLimit is reached.
    CMtCoderThread *nextThread = &p->threads[p->numStartedThreads++];
    {
      const SRes res2 = MtCoderThread_CreateAndStart(nextThread
#ifdef _WIN32
            , p
#endif
            );
      RINOK(res2)
    }
  }

  RINOK_THREAD(Event_Set(&p->readEvent))

  #ifdef MTCODER_USE_WRITE_THREAD
  {
    unsigned bi = 0;

    for (;; bi++)
    {
      if (bi >= numBlocksMax)
        bi = 0;

      RINOK_THREAD(Event_Wait(&p->writeEvents[bi]))

      {
        const CMtCoderBlock * const block = &p->blocks[bi];
        const unsigned bufIndex = block->bufIndex;
        const BoolInt finished = block->finished;
        if (res == SZ_OK && block->res != SZ_OK)
          res = block->res;

        if (bufIndex != (unsigned)(int)-1)
        {
          if (res == SZ_OK)
          {
            res = p->mtCallback->Write(p->mtCallbackObject, bufIndex);
            if (res != SZ_OK)
              MtProgress_SetError(&p->mtProgress, res);
          }
          
          CriticalSection_Enter(&p->cs);
          {
            p->freeBlockList[bufIndex] = p->freeBlockHead;
            p->freeBlockHead = bufIndex;
          }
          CriticalSection_Leave(&p->cs);
        }
        
        RINOK_THREAD(Semaphore_Release1(&p->blocksSemaphore))

        if (finished)
          break;
      }
    }
  }
  #else
  {
    const WRes wres = Event_Wait(&p->finishedEvent);
    res = MY_SRes_HRESULT_FROM_WRes(wres);
  }
  #endif

  if (res == SZ_OK)
    res = p->readRes;

  if (res == SZ_OK)
    res = p->mtProgress.res;

  #ifndef MTCODER_USE_WRITE_THREAD
    if (res == SZ_OK)
      res = p->writeRes;
  #endif

  if (res != SZ_OK)
    MtCoder_Free(p);
  return res;
}

#endif

#undef RINOK_THREAD
