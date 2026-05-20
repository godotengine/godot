/* LzFindMt.c -- multithreaded Match finder for LZ algorithms
: Igor Pavlov : Public domain */

#include "Precomp.h"

// #include <stdio.h>

#include "CpuArch.h"

#include "LzHash.h"
#include "LzFindMt.h"

// #define LOG_ITERS

// #define LOG_THREAD

#ifdef LOG_THREAD
#include <stdio.h>
#define PRF(x) x
#else
#define PRF(x)
#endif

#ifdef LOG_ITERS
#include <stdio.h>
extern UInt64 g_NumIters_Tree;
extern UInt64 g_NumIters_Loop;
extern UInt64 g_NumIters_Bytes;
#define LOG_ITER(x) x
#else
#define LOG_ITER(x)
#endif

#define kMtHashBlockSize ((UInt32)1 << 17)
#define kMtHashNumBlocks (1 << 1)

#define GET_HASH_BLOCK_OFFSET(i)  (((i) & (kMtHashNumBlocks - 1)) * kMtHashBlockSize)

#define kMtBtBlockSize ((UInt32)1 << 16)
#define kMtBtNumBlocks (1 << 4)

#define GET_BT_BLOCK_OFFSET(i)  (((i) & (kMtBtNumBlocks - 1)) * (size_t)kMtBtBlockSize)

/*
  HASH functions:
  We use raw 8/16 bits from a[1] and a[2],
  xored with crc(a[0]) and crc(a[3]).
  We check a[0], a[3] only. We don't need to compare a[1] and a[2] in matches.
  our crc() function provides one-to-one correspondence for low 8-bit values:
    (crc[0...0xFF] & 0xFF) <-> [0...0xFF]
*/

#define MF(mt) ((mt)->MatchFinder)
#define MF_CRC (p->crc)

// #define MF(mt) (&(mt)->MatchFinder)
// #define MF_CRC (p->MatchFinder.crc)

#define MT_HASH2_CALC \
  h2 = (MF_CRC[cur[0]] ^ cur[1]) & (kHash2Size - 1);

#define MT_HASH3_CALC { \
  UInt32 temp = MF_CRC[cur[0]] ^ cur[1]; \
  h2 = temp & (kHash2Size - 1); \
  h3 = (temp ^ ((UInt32)cur[2] << 8)) & (kHash3Size - 1); }

/*
#define MT_HASH3_CALC__NO_2 { \
  UInt32 temp = p->crc[cur[0]] ^ cur[1]; \
  h3 = (temp ^ ((UInt32)cur[2] << 8)) & (kHash3Size - 1); }

#define MT_HASH4_CALC { \
  UInt32 temp = p->crc[cur[0]] ^ cur[1]; \
  h2 = temp & (kHash2Size - 1); \
  temp ^= ((UInt32)cur[2] << 8); \
  h3 = temp & (kHash3Size - 1); \
  h4 = (temp ^ (p->crc[cur[3]] << kLzHash_CrcShift_1)) & p->hash4Mask; }
  // (kHash4Size - 1);
*/


Z7_NO_INLINE
static void MtSync_Construct(CMtSync *p)
{
  p->affinityGroup = -1;
  p->affinityInGroup = 0;
  p->affinity = 0;
  p->wasCreated = False;
  p->csWasInitialized = False;
  p->csWasEntered = False;
  Thread_CONSTRUCT(&p->thread)
  Event_Construct(&p->canStart);
  Event_Construct(&p->wasStopped);
  Semaphore_Construct(&p->freeSemaphore);
  Semaphore_Construct(&p->filledSemaphore);
}


// #define DEBUG_BUFFER_LOCK   // define it to debug lock state

#ifdef DEBUG_BUFFER_LOCK
#include <stdlib.h>
#define BUFFER_MUST_BE_LOCKED(p)    if (!(p)->csWasEntered) exit(1);
#define BUFFER_MUST_BE_UNLOCKED(p)  if ( (p)->csWasEntered) exit(1);
#else
#define BUFFER_MUST_BE_LOCKED(p)
#define BUFFER_MUST_BE_UNLOCKED(p)
#endif

#define LOCK_BUFFER(p) { \
    BUFFER_MUST_BE_UNLOCKED(p); \
    CriticalSection_Enter(&(p)->cs); \
    (p)->csWasEntered = True; }

#define UNLOCK_BUFFER(p) { \
    BUFFER_MUST_BE_LOCKED(p); \
    CriticalSection_Leave(&(p)->cs); \
    (p)->csWasEntered = False; }


Z7_NO_INLINE
static UInt32 MtSync_GetNextBlock(CMtSync *p)
{
  UInt32 numBlocks = 0;
  if (p->needStart)
  {
    BUFFER_MUST_BE_UNLOCKED(p)
    p->numProcessedBlocks = 1;
    p->needStart = False;
    p->stopWriting = False;
    p->exit = False;
    Event_Reset(&p->wasStopped);
    Event_Set(&p->canStart);
  }
  else
  {
    UNLOCK_BUFFER(p)
    // we free current block
    numBlocks = p->numProcessedBlocks++;
    Semaphore_Release1(&p->freeSemaphore);
  }

  // buffer is UNLOCKED here
  Semaphore_Wait(&p->filledSemaphore);
  LOCK_BUFFER(p)
  return numBlocks;
}


/* if Writing (Processing) thread was started, we must call MtSync_StopWriting() */

Z7_NO_INLINE
static void MtSync_StopWriting(CMtSync *p)
{
  if (!Thread_WasCreated(&p->thread) || p->needStart)
    return;

    PRF(printf("\nMtSync_StopWriting %p\n", p));

  if (p->csWasEntered)
  {
    /* we don't use buffer in this thread after StopWriting().
       So we UNLOCK buffer.
       And we restore default UNLOCKED state for stopped thread */
    UNLOCK_BUFFER(p)
  }

  /* We send (p->stopWriting) message and release freeSemaphore
     to free current block.
     So the thread will see (p->stopWriting) at some
     iteration after Wait(freeSemaphore).
     The thread doesn't need to fill all avail free blocks,
     so we can get fast thread stop.
  */

  p->stopWriting = True;
  Semaphore_Release1(&p->freeSemaphore); // check semaphore count !!!

    PRF(printf("\nMtSync_StopWriting %p : Event_Wait(&p->wasStopped)\n", p));
  Event_Wait(&p->wasStopped);
    PRF(printf("\nMtSync_StopWriting %p : Event_Wait() finsihed\n", p));

  /* 21.03 : we don't restore samaphore counters here.
     We will recreate and reinit samaphores in next start */

  p->needStart = True;
}


Z7_NO_INLINE
static void MtSync_Destruct(CMtSync *p)
{
    PRF(printf("\nMtSync_Destruct %p\n", p));
  
  if (Thread_WasCreated(&p->thread))
  {
    /* we want thread to be in Stopped state before sending EXIT command.
       note: stop(btSync) will stop (htSync) also */
    MtSync_StopWriting(p);
    /* thread in Stopped state here : (p->needStart == true) */
    p->exit = True;
    // if (p->needStart)  // it's (true)
    Event_Set(&p->canStart);  // we send EXIT command to thread
    Thread_Wait_Close(&p->thread);  // we wait thread finishing
  }

  if (p->csWasInitialized)
  {
    CriticalSection_Delete(&p->cs);
    p->csWasInitialized = False;
  }
  p->csWasEntered = False;

  Event_Close(&p->canStart);
  Event_Close(&p->wasStopped);
  Semaphore_Close(&p->freeSemaphore);
  Semaphore_Close(&p->filledSemaphore);

  p->wasCreated = False;
}


// #define RINOK_THREAD(x) { if ((x) != 0) return SZ_ERROR_THREAD; }
// we want to get real system error codes here instead of SZ_ERROR_THREAD
#define RINOK_THREAD(x)  RINOK_WRes(x)


// call it before each new file (when new starting is required):
Z7_NO_INLINE
static SRes MtSync_Init(CMtSync *p, UInt32 numBlocks)
{
  WRes wres;
  // BUFFER_MUST_BE_UNLOCKED(p)
  if (!p->needStart || p->csWasEntered)
    return SZ_ERROR_FAIL;
  wres = Semaphore_OptCreateInit(&p->freeSemaphore, numBlocks, numBlocks);
  if (wres == 0)
    wres = Semaphore_OptCreateInit(&p->filledSemaphore, 0, numBlocks);
  return MY_SRes_HRESULT_FROM_WRes(wres);
}


static WRes MtSync_Create_WRes(CMtSync *p, THREAD_FUNC_TYPE startAddress, void *obj)
{
  WRes wres;

  if (p->wasCreated)
    return SZ_OK;

  RINOK_THREAD(CriticalSection_Init(&p->cs))
  p->csWasInitialized = True;
  p->csWasEntered = False;

  RINOK_THREAD(AutoResetEvent_CreateNotSignaled(&p->canStart))
  RINOK_THREAD(AutoResetEvent_CreateNotSignaled(&p->wasStopped))

  p->needStart = True;
  p->exit = True;  /* p->exit is unused before (canStart) Event.
     But in case of some unexpected code failure we will get fast exit from thread */

  // return ERROR_TOO_MANY_POSTS; // for debug
  // return EINVAL; // for debug

#ifdef _WIN32
  if (p->affinityGroup >= 0)
    wres = Thread_Create_With_Group(&p->thread, startAddress, obj,
        (unsigned)(UInt32)p->affinityGroup, (CAffinityMask)p->affinityInGroup);
  else
#endif
  if (p->affinity != 0)
    wres = Thread_Create_With_Affinity(&p->thread, startAddress, obj, (CAffinityMask)p->affinity);
  else
    wres = Thread_Create(&p->thread, startAddress, obj);

  RINOK_THREAD(wres)
  p->wasCreated = True;
  return SZ_OK;
}


Z7_NO_INLINE
static SRes MtSync_Create(CMtSync *p, THREAD_FUNC_TYPE startAddress, void *obj)
{
  const WRes wres = MtSync_Create_WRes(p, startAddress, obj);
  if (wres == 0)
    return 0;
  MtSync_Destruct(p);
  return MY_SRes_HRESULT_FROM_WRes(wres);
}


// ---------- HASH THREAD ----------

#define kMtMaxValForNormalize 0xFFFFFFFF
// #define kMtMaxValForNormalize ((1 << 21)) // for debug
// #define kNormalizeAlign (1 << 7) // alignment for speculated accesses

#ifdef MY_CPU_LE_UNALIGN
  #define GetUi24hi_from32(p) ((UInt32)GetUi32(p) >> 8)
#else
  #define GetUi24hi_from32(p) ((p)[1] ^ ((UInt32)(p)[2] << 8) ^ ((UInt32)(p)[3] << 16))
#endif

#define GetHeads_DECL(name) \
    static void GetHeads ## name(const Byte *p, UInt32 pos, \
      UInt32 *hash, UInt32 hashMask, UInt32 *heads, UInt32 numHeads, const UInt32 *crc)

#define GetHeads_LOOP(v) \
    for (; numHeads != 0; numHeads--) { \
      const UInt32 value = (v); \
      p++; \
      *heads++ = pos - hash[value]; \
      hash[value] = pos++; }

#define DEF_GetHeads2(name, v, action) \
    GetHeads_DECL(name) { action \
    GetHeads_LOOP(v) }
 
#define DEF_GetHeads(name, v) DEF_GetHeads2(name, v, ;)

DEF_GetHeads2(2, GetUi16(p), UNUSED_VAR(hashMask); UNUSED_VAR(crc); )
DEF_GetHeads(3,  (crc[p[0]] ^ GetUi16(p + 1)) & hashMask)
DEF_GetHeads2(3b, GetUi16(p) ^ ((UInt32)(p)[2] << 16), UNUSED_VAR(hashMask); UNUSED_VAR(crc); )
// BT3 is not good for crc collisions for big hashMask values.

/*
GetHeads_DECL(3b)
{
  UNUSED_VAR(hashMask);
  UNUSED_VAR(crc);
  {
  const Byte *pLim = p + numHeads;
  if (numHeads == 0)
    return;
  pLim--;
  while (p < pLim)
  {
    UInt32 v1 = GetUi32(p);
    UInt32 v0 = v1 & 0xFFFFFF;
    UInt32 h0, h1;
    p += 2;
    v1 >>= 8;
    h0 = hash[v0]; hash[v0] = pos; heads[0] = pos - h0; pos++;
    h1 = hash[v1]; hash[v1] = pos; heads[1] = pos - h1; pos++;
    heads += 2;
  }
  if (p == pLim)
  {
    UInt32 v0 = GetUi16(p) ^ ((UInt32)(p)[2] << 16);
    *heads = pos - hash[v0];
    hash[v0] = pos;
  }
  }
}
*/

/*
GetHeads_DECL(4)
{
  unsigned sh = 0;
  UNUSED_VAR(crc)
  while ((hashMask & 0x80000000) == 0)
  {
    hashMask <<= 1;
    sh++;
  }
  GetHeads_LOOP((GetUi32(p) * 0xa54a1) >> sh)
}
#define GetHeads4b GetHeads4
*/

#define USE_GetHeads_LOCAL_CRC

#ifdef USE_GetHeads_LOCAL_CRC

GetHeads_DECL(4)
{
  UInt32 crc0[256];
  UInt32 crc1[256];
  {
    unsigned i;
    for (i = 0; i < 256; i++)
    {
      UInt32 v = crc[i];
      crc0[i] = v & hashMask;
      crc1[i] = (v << kLzHash_CrcShift_1) & hashMask;
      // crc1[i] = rotlFixed(v, 8) & hashMask;
    }
  }
  GetHeads_LOOP(crc0[p[0]] ^ crc1[p[3]] ^ (UInt32)GetUi16(p+1))
}

GetHeads_DECL(4b)
{
  UInt32 crc0[256];
  {
    unsigned i;
    for (i = 0; i < 256; i++)
      crc0[i] = crc[i] & hashMask;
  }
  GetHeads_LOOP(crc0[p[0]] ^ GetUi24hi_from32(p))
}

GetHeads_DECL(5)
{
  UInt32 crc0[256];
  UInt32 crc1[256];
  UInt32 crc2[256];
  {
    unsigned i;
    for (i = 0; i < 256; i++)
    {
      UInt32 v = crc[i];
      crc0[i] = v & hashMask;
      crc1[i] = (v << kLzHash_CrcShift_1) & hashMask;
      crc2[i] = (v << kLzHash_CrcShift_2) & hashMask;
    }
  }
  GetHeads_LOOP(crc0[p[0]] ^ crc1[p[3]] ^ crc2[p[4]] ^ (UInt32)GetUi16(p+1))
}

GetHeads_DECL(5b)
{
  UInt32 crc0[256];
  UInt32 crc1[256];
  {
    unsigned i;
    for (i = 0; i < 256; i++)
    {
      UInt32 v = crc[i];
      crc0[i] = v & hashMask;
      crc1[i] = (v << kLzHash_CrcShift_1) & hashMask;
    }
  }
  GetHeads_LOOP(crc0[p[0]] ^ crc1[p[4]] ^ GetUi24hi_from32(p))
}

#else

DEF_GetHeads(4,  (crc[p[0]] ^ (crc[p[3]] << kLzHash_CrcShift_1) ^ (UInt32)GetUi16(p+1)) & hashMask)
DEF_GetHeads(4b, (crc[p[0]] ^ GetUi24hi_from32(p)) & hashMask)
DEF_GetHeads(5,  (crc[p[0]] ^ (crc[p[3]] << kLzHash_CrcShift_1) ^ (crc[p[4]] << kLzHash_CrcShift_2) ^ (UInt32)GetUi16(p + 1)) & hashMask)
DEF_GetHeads(5b, (crc[p[0]] ^ (crc[p[4]] << kLzHash_CrcShift_1) ^ GetUi24hi_from32(p)) & hashMask)

#endif
 

static void HashThreadFunc(CMatchFinderMt *mt)
{
  CMtSync *p = &mt->hashSync;
    PRF(printf("\nHashThreadFunc\n"));
  
  for (;;)
  {
    UInt32 blockIndex = 0;
      PRF(printf("\nHashThreadFunc : Event_Wait(&p->canStart)\n"));
    Event_Wait(&p->canStart);
      PRF(printf("\nHashThreadFunc : Event_Wait(&p->canStart) : after \n"));
    if (p->exit)
    {
      PRF(printf("\nHashThreadFunc : exit \n"));
      return;
    }

    MatchFinder_Init_HighHash(MF(mt));

    for (;;)
    {
      PRF(printf("Hash thread block = %d pos = %d\n", (unsigned)blockIndex, mt->MatchFinder->pos));

      {
        CMatchFinder *mf = MF(mt);
        if (MatchFinder_NeedMove(mf))
        {
          CriticalSection_Enter(&mt->btSync.cs);
          CriticalSection_Enter(&mt->hashSync.cs);
          {
            const Byte *beforePtr = Inline_MatchFinder_GetPointerToCurrentPos(mf);
            ptrdiff_t offset;
            MatchFinder_MoveBlock(mf);
            offset = beforePtr - Inline_MatchFinder_GetPointerToCurrentPos(mf);
            mt->pointerToCurPos -= offset;
            mt->buffer -= offset;
          }
          CriticalSection_Leave(&mt->hashSync.cs);
          CriticalSection_Leave(&mt->btSync.cs);
          continue;
        }

        Semaphore_Wait(&p->freeSemaphore);

        if (p->exit) // exit is unexpected here. But we check it here for some failure case
          return;

        // for faster stop : we check (p->stopWriting) after Wait(freeSemaphore)
        if (p->stopWriting)
          break;

        MatchFinder_ReadIfRequired(mf);
        {
          UInt32 *heads = mt->hashBuf + GET_HASH_BLOCK_OFFSET(blockIndex++);
          UInt32 num = Inline_MatchFinder_GetNumAvailableBytes(mf);
          heads[0] = 2;
          heads[1] = num;

          /* heads[1] contains the number of avail bytes:
             if (avail < mf->numHashBytes) :
             {
               it means that stream was finished
               HASH_THREAD and BT_TREAD must move position for heads[1] (avail) bytes.
               HASH_THREAD doesn't stop,
               HASH_THREAD fills only the header (2 numbers) for all next blocks:
               {2, NumHashBytes - 1}, {2,0}, {2,0}, ... , {2,0}
             }
             else
             {
               HASH_THREAD and BT_TREAD must move position for (heads[0] - 2) bytes;
             }
          */

          if (num >= mf->numHashBytes)
          {
            num = num - mf->numHashBytes + 1;
            if (num > kMtHashBlockSize - 2)
              num = kMtHashBlockSize - 2;

            if (mf->pos > (UInt32)kMtMaxValForNormalize - num)
            {
              const UInt32 subValue = (mf->pos - mf->historySize - 1); // & ~(UInt32)(kNormalizeAlign - 1);
              MatchFinder_REDUCE_OFFSETS(mf, subValue)
              MatchFinder_Normalize3(subValue, mf->hash + mf->fixedHashSize, (size_t)mf->hashMask + 1);
            }

            heads[0] = 2 + num;
            mt->GetHeadsFunc(mf->buffer, mf->pos, mf->hash + mf->fixedHashSize, mf->hashMask, heads + 2, num, mf->crc);
          }

          mf->pos += num;  // wrap over zero is allowed at the end of stream
          mf->buffer += num;
        }
      }

      Semaphore_Release1(&p->filledSemaphore);
    } // for() processing end

    // p->numBlocks_Sent = blockIndex;
    Event_Set(&p->wasStopped);
  } // for() thread end
}




// ---------- BT THREAD ----------

/* we use one variable instead of two (cyclicBufferPos == pos) before CyclicBuf wrap.
   here we define fixed offset of (p->pos) from (p->cyclicBufferPos) */
#define CYC_TO_POS_OFFSET 0
// #define CYC_TO_POS_OFFSET 1 // for debug

#define MFMT_GM_INLINE

#ifdef MFMT_GM_INLINE

/*
  we use size_t for (pos) instead of UInt32
  to eliminate "movsx" BUG in old MSVC x64 compiler.
*/


UInt32 * Z7_FASTCALL GetMatchesSpecN_2(const Byte *lenLimit, size_t pos, const Byte *cur, CLzRef *son,
    UInt32 _cutValue, UInt32 *d, size_t _maxLen, const UInt32 *hash, const UInt32 *limit, const UInt32 *size,
    size_t _cyclicBufferPos, UInt32 _cyclicBufferSize,
    UInt32 *posRes);

#endif


static void BtGetMatches(CMatchFinderMt *p, UInt32 *d)
{
  UInt32 numProcessed = 0;
  UInt32 curPos = 2;
  
  /* GetMatchesSpec() functions don't create (len = 1)
     in [len, dist] match pairs, if (p->numHashBytes >= 2)
     Also we suppose here that (matchMaxLen >= 2).
     So the following code for (reserve) is not required
     UInt32 reserve = (p->matchMaxLen * 2);
     const UInt32 kNumHashBytes_Max = 5; // BT_HASH_BYTES_MAX
     if (reserve < kNumHashBytes_Max - 1)
        reserve = kNumHashBytes_Max - 1;
     const UInt32 limit = kMtBtBlockSize - (reserve);
  */

  const UInt32 limit = kMtBtBlockSize - (p->matchMaxLen * 2);

  d[1] = p->hashNumAvail;

  if (p->failure_BT)
  {
    // printf("\n == 1 BtGetMatches() p->failure_BT\n");
    d[0] = 0;
    // d[1] = 0;
    return;
  }
  
  while (curPos < limit)
  {
    if (p->hashBufPos == p->hashBufPosLimit)
    {
      // MatchFinderMt_GetNextBlock_Hash(p);
      UInt32 avail;
      {
        const UInt32 bi = MtSync_GetNextBlock(&p->hashSync);
        const UInt32 k = GET_HASH_BLOCK_OFFSET(bi);
        const UInt32 *h = p->hashBuf + k;
        avail = h[1];
        p->hashBufPosLimit = k + h[0];
        p->hashNumAvail = avail;
        p->hashBufPos = k + 2;
      }

      {
        /* we must prevent UInt32 overflow for avail total value,
           if avail was increased with new hash block */
        UInt32 availSum = numProcessed + avail;
        if (availSum < numProcessed)
          availSum = (UInt32)(Int32)-1;
        d[1] = availSum;
      }

      if (avail >= p->numHashBytes)
        continue;

      // if (p->hashBufPos != p->hashBufPosLimit) exit(1);

      /* (avail < p->numHashBytes)
         It means that stream was finished.
         And (avail) - is a number of remaining bytes,
         we fill (d) for (avail) bytes for LZ_THREAD (receiver).
         but we don't update (p->pos) and (p->cyclicBufferPos) here in BT_THREAD */

      /* here we suppose that we have space enough:
         (kMtBtBlockSize - curPos >= p->hashNumAvail) */
      p->hashNumAvail = 0;
      d[0] = curPos + avail;
      d += curPos;
      for (; avail != 0; avail--)
        *d++ = 0;
      return;
    }
    {
      UInt32 size = p->hashBufPosLimit - p->hashBufPos;
      UInt32 pos = p->pos;
      UInt32 cyclicBufferPos = p->cyclicBufferPos;
      UInt32 lenLimit = p->matchMaxLen;
      if (lenLimit >= p->hashNumAvail)
        lenLimit = p->hashNumAvail;
      {
        UInt32 size2 = p->hashNumAvail - lenLimit + 1;
        if (size2 < size)
          size = size2;
        size2 = p->cyclicBufferSize - cyclicBufferPos;
        if (size2 < size)
          size = size2;
      }
      
      if (pos > (UInt32)kMtMaxValForNormalize - size)
      {
        const UInt32 subValue = (pos - p->cyclicBufferSize); // & ~(UInt32)(kNormalizeAlign - 1);
        pos -= subValue;
        p->pos = pos;
        MatchFinder_Normalize3(subValue, p->son, (size_t)p->cyclicBufferSize * 2);
      }

      #ifndef MFMT_GM_INLINE
      while (curPos < limit && size-- != 0)
      {
        UInt32 *startDistances = d + curPos;
        UInt32 num = (UInt32)(GetMatchesSpec1(lenLimit, pos - p->hashBuf[p->hashBufPos++],
            pos, p->buffer, p->son, cyclicBufferPos, p->cyclicBufferSize, p->cutValue,
            startDistances + 1, p->numHashBytes - 1) - startDistances);
        *startDistances = num - 1;
        curPos += num;
        cyclicBufferPos++;
        pos++;
        p->buffer++;
      }
      #else
      {
        UInt32 posRes = pos;
        const UInt32 *d_end;
        {
          d_end = GetMatchesSpecN_2(
              p->buffer + lenLimit - 1,
              pos, p->buffer, p->son, p->cutValue, d + curPos,
              p->numHashBytes - 1, p->hashBuf + p->hashBufPos,
              d + limit, p->hashBuf + p->hashBufPos + size,
              cyclicBufferPos, p->cyclicBufferSize,
              &posRes);
        }
        {
          if (!d_end)
          {
            // printf("\n == 2 BtGetMatches() p->failure_BT\n");
            // internal data failure
            p->failure_BT = True;
            d[0] = 0;
            // d[1] = 0;
            return;
          }
        }
        curPos = (UInt32)(d_end - d);
        {
          const UInt32 processed = posRes - pos;
          pos = posRes;
          p->hashBufPos += processed;
          cyclicBufferPos += processed;
          p->buffer += processed;
        }
      }
      #endif

      {
        const UInt32 processed = pos - p->pos;
        numProcessed += processed;
        p->hashNumAvail -= processed;
        p->pos = pos;
      }
      if (cyclicBufferPos == p->cyclicBufferSize)
        cyclicBufferPos = 0;
      p->cyclicBufferPos = cyclicBufferPos;
    }
  }
  
  d[0] = curPos;
}


static void BtFillBlock(CMatchFinderMt *p, UInt32 globalBlockIndex)
{
  CMtSync *sync = &p->hashSync;
  
  BUFFER_MUST_BE_UNLOCKED(sync)
  
  if (!sync->needStart)
  {
    LOCK_BUFFER(sync)
  }
  
  BtGetMatches(p, p->btBuf + GET_BT_BLOCK_OFFSET(globalBlockIndex));
  
  /* We suppose that we have called GetNextBlock() from start.
     So buffer is LOCKED */

  UNLOCK_BUFFER(sync)
}


Z7_NO_INLINE
static void BtThreadFunc(CMatchFinderMt *mt)
{
  CMtSync *p = &mt->btSync;
  for (;;)
  {
    UInt32 blockIndex = 0;
    Event_Wait(&p->canStart);

    for (;;)
    {
        PRF(printf("  BT thread block = %d  pos = %d\n", (unsigned)blockIndex, mt->pos));
      /* (p->exit == true) is possible after (p->canStart) at first loop iteration
         and is unexpected after more Wait(freeSemaphore) iterations */
      if (p->exit)
        return;

      Semaphore_Wait(&p->freeSemaphore);
      
      // for faster stop : we check (p->stopWriting) after Wait(freeSemaphore)
      if (p->stopWriting)
        break;

      BtFillBlock(mt, blockIndex++);
      
      Semaphore_Release1(&p->filledSemaphore);
    }

    // we stop HASH_THREAD here
    MtSync_StopWriting(&mt->hashSync);

    // p->numBlocks_Sent = blockIndex;
    Event_Set(&p->wasStopped);
  }
}


void MatchFinderMt_Construct(CMatchFinderMt *p)
{
  p->hashBuf = NULL;
  MtSync_Construct(&p->hashSync);
  MtSync_Construct(&p->btSync);
}

static void MatchFinderMt_FreeMem(CMatchFinderMt *p, ISzAllocPtr alloc)
{
  ISzAlloc_Free(alloc, p->hashBuf);
  p->hashBuf = NULL;
}

void MatchFinderMt_Destruct(CMatchFinderMt *p, ISzAllocPtr alloc)
{
  /*
     HASH_THREAD can use CriticalSection(s) btSync.cs and hashSync.cs.
     So we must be sure that HASH_THREAD will not use CriticalSection(s)
     after deleting CriticalSection here.

     we call ReleaseStream(p)
       that calls StopWriting(btSync)
         that calls StopWriting(hashSync), if it's required to stop HASH_THREAD.
     after StopWriting() it's safe to destruct MtSync(s) in any order */

  MatchFinderMt_ReleaseStream(p);

  MtSync_Destruct(&p->btSync);
  MtSync_Destruct(&p->hashSync);

  LOG_ITER(
  printf("\nTree %9d * %7d iter = %9d = sum  :  bytes = %9d\n",
      (UInt32)(g_NumIters_Tree / 1000),
      (UInt32)(((UInt64)g_NumIters_Loop * 1000) / (g_NumIters_Tree + 1)),
      (UInt32)(g_NumIters_Loop / 1000),
      (UInt32)(g_NumIters_Bytes / 1000)
      ));

  MatchFinderMt_FreeMem(p, alloc);
}


#define kHashBufferSize (kMtHashBlockSize * kMtHashNumBlocks)
#define kBtBufferSize (kMtBtBlockSize * kMtBtNumBlocks)


static THREAD_FUNC_DECL HashThreadFunc2(void *p) { HashThreadFunc((CMatchFinderMt *)p);  return 0; }
static THREAD_FUNC_DECL BtThreadFunc2(void *p)
{
  Byte allocaDummy[0x180];
  unsigned i = 0;
  for (i = 0; i < 16; i++)
    allocaDummy[i] = (Byte)0;
  if (allocaDummy[0] == 0)
    BtThreadFunc((CMatchFinderMt *)p);
  return 0;
}


SRes MatchFinderMt_Create(CMatchFinderMt *p, UInt32 historySize, UInt32 keepAddBufferBefore,
    UInt32 matchMaxLen, UInt32 keepAddBufferAfter, ISzAllocPtr alloc)
{
  CMatchFinder *mf = MF(p);
  p->historySize = historySize;
  if (kMtBtBlockSize <= matchMaxLen * 4)
    return SZ_ERROR_PARAM;
  if (!p->hashBuf)
  {
    p->hashBuf = (UInt32 *)ISzAlloc_Alloc(alloc, ((size_t)kHashBufferSize + (size_t)kBtBufferSize) * sizeof(UInt32));
    if (!p->hashBuf)
      return SZ_ERROR_MEM;
    p->btBuf = p->hashBuf + kHashBufferSize;
  }
  keepAddBufferBefore += (kHashBufferSize + kBtBufferSize);
  keepAddBufferAfter += kMtHashBlockSize;
  if (!MatchFinder_Create(mf, historySize, keepAddBufferBefore, matchMaxLen, keepAddBufferAfter, alloc))
    return SZ_ERROR_MEM;

  RINOK(MtSync_Create(&p->hashSync, HashThreadFunc2, p))
  RINOK(MtSync_Create(&p->btSync, BtThreadFunc2, p))
  return SZ_OK;
}


SRes MatchFinderMt_InitMt(CMatchFinderMt *p)
{
  RINOK(MtSync_Init(&p->hashSync, kMtHashNumBlocks))
  return MtSync_Init(&p->btSync, kMtBtNumBlocks);
}


static void MatchFinderMt_Init(void *_p)
{
  CMatchFinderMt *p = (CMatchFinderMt *)_p;
  CMatchFinder *mf = MF(p);
  
  p->btBufPos =
  p->btBufPosLimit = NULL;
  p->hashBufPos =
  p->hashBufPosLimit = 0;
  p->hashNumAvail = 0; // 21.03
  
  p->failure_BT = False;

  /* Init without data reading. We don't want to read data in this thread */
  MatchFinder_Init_4(mf);

  MatchFinder_Init_LowHash(mf);
  
  p->pointerToCurPos = Inline_MatchFinder_GetPointerToCurrentPos(mf);
  p->btNumAvailBytes = 0;
  p->failure_LZ_BT = False;
  // p->failure_LZ_LZ = False;
  
  p->lzPos =
      1; // optimal smallest value
      // 0; // for debug: ignores match to start
      // kNormalizeAlign; // for debug

  p->hash = mf->hash;
  p->fixedHashSize = mf->fixedHashSize;
  // p->hash4Mask = mf->hash4Mask;
  p->crc = mf->crc;
  // memcpy(p->crc, mf->crc, sizeof(mf->crc));

  p->son = mf->son;
  p->matchMaxLen = mf->matchMaxLen;
  p->numHashBytes = mf->numHashBytes;
  
  /* (mf->pos) and (mf->streamPos) were already initialized to 1 in MatchFinder_Init_4() */
  // mf->streamPos = mf->pos = 1; // optimal smallest value
      // 0; // for debug: ignores match to start
      // kNormalizeAlign; // for debug

  /* we must init (p->pos = mf->pos) for BT, because
     BT code needs (p->pos == delta_value_for_empty_hash_record == mf->pos) */
  p->pos = mf->pos; // do not change it
  
  p->cyclicBufferPos = (p->pos - CYC_TO_POS_OFFSET);
  p->cyclicBufferSize = mf->cyclicBufferSize;
  p->buffer = mf->buffer;
  p->cutValue = mf->cutValue;
  // p->son[0] = p->son[1] = 0; // unused: to init skipped record for speculated accesses.
}


/* ReleaseStream is required to finish multithreading */
void MatchFinderMt_ReleaseStream(CMatchFinderMt *p)
{
  // Sleep(1); // for debug
  MtSync_StopWriting(&p->btSync);
  // Sleep(200); // for debug
  /* p->MatchFinder->ReleaseStream(); */
}


Z7_NO_INLINE
static UInt32 MatchFinderMt_GetNextBlock_Bt(CMatchFinderMt *p)
{
  if (p->failure_LZ_BT)
    p->btBufPos = p->failureBuf;
  else
  {
    const UInt32 bi = MtSync_GetNextBlock(&p->btSync);
    const UInt32 *bt = p->btBuf + GET_BT_BLOCK_OFFSET(bi);
    {
      const UInt32 numItems = bt[0];
      p->btBufPosLimit = bt + numItems;
      p->btNumAvailBytes = bt[1];
      p->btBufPos = bt + 2;
      if (numItems < 2 || numItems > kMtBtBlockSize)
      {
        p->failureBuf[0] = 0;
        p->btBufPos = p->failureBuf;
        p->btBufPosLimit = p->failureBuf + 1;
        p->failure_LZ_BT = True;
        // p->btNumAvailBytes = 0;
        /* we don't want to decrease AvailBytes, that was load before.
            that can be unxepected for the code that have loaded anopther value before */
      }
    }
  
    if (p->lzPos >= (UInt32)kMtMaxValForNormalize - (UInt32)kMtBtBlockSize)
    {
      /* we don't check (lzPos) over exact avail bytes in (btBuf).
         (fixedHashSize) is small, so normalization is fast */
      const UInt32 subValue = (p->lzPos - p->historySize - 1); // & ~(UInt32)(kNormalizeAlign - 1);
      p->lzPos -= subValue;
      MatchFinder_Normalize3(subValue, p->hash, p->fixedHashSize);
    }
  }
  return p->btNumAvailBytes;
}



static const Byte * MatchFinderMt_GetPointerToCurrentPos(void *_p)
{
  CMatchFinderMt *p = (CMatchFinderMt *)_p;
  return p->pointerToCurPos;
}


#define GET_NEXT_BLOCK_IF_REQUIRED if (p->btBufPos == p->btBufPosLimit) MatchFinderMt_GetNextBlock_Bt(p);


static UInt32 MatchFinderMt_GetNumAvailableBytes(void *_p)
{
  CMatchFinderMt *p = (CMatchFinderMt *)_p;
  if (p->btBufPos != p->btBufPosLimit)
    return p->btNumAvailBytes;
  return MatchFinderMt_GetNextBlock_Bt(p);
}


// #define CHECK_FAILURE_LZ(_match_, _pos_) if (_match_ >= _pos_) { p->failure_LZ_LZ = True;  return d; }
#define CHECK_FAILURE_LZ(_match_, _pos_)

static UInt32 * MixMatches2(CMatchFinderMt *p, UInt32 matchMinPos, UInt32 *d)
{
  UInt32 h2, c2;
  UInt32 *hash = p->hash;
  const Byte *cur = p->pointerToCurPos;
  const UInt32 m = p->lzPos;
  MT_HASH2_CALC
      
  c2 = hash[h2];
  hash[h2] = m;

  if (c2 >= matchMinPos)
  {
    CHECK_FAILURE_LZ(c2, m)
    if (cur[(ptrdiff_t)c2 - (ptrdiff_t)m] == cur[0])
    {
      *d++ = 2;
      *d++ = m - c2 - 1;
    }
  }
  
  return d;
}

static UInt32 * MixMatches3(CMatchFinderMt *p, UInt32 matchMinPos, UInt32 *d)
{
  UInt32 h2, h3, c2, c3;
  UInt32 *hash = p->hash;
  const Byte *cur = p->pointerToCurPos;
  const UInt32 m = p->lzPos;
  MT_HASH3_CALC

  c2 = hash[h2];
  c3 = (hash + kFix3HashSize)[h3];
  
  hash[h2] = m;
  (hash + kFix3HashSize)[h3] = m;

  if (c2 >= matchMinPos)
  {
    CHECK_FAILURE_LZ(c2, m)
    if (cur[(ptrdiff_t)c2 - (ptrdiff_t)m] == cur[0])
    {
      d[1] = m - c2 - 1;
      if (cur[(ptrdiff_t)c2 - (ptrdiff_t)m + 2] == cur[2])
      {
        d[0] = 3;
        return d + 2;
      }
      d[0] = 2;
      d += 2;
    }
  }
  
  if (c3 >= matchMinPos)
  {
    CHECK_FAILURE_LZ(c3, m)
    if (cur[(ptrdiff_t)c3 - (ptrdiff_t)m] == cur[0])
    {
      *d++ = 3;
      *d++ = m - c3 - 1;
    }
  }
  
  return d;
}


#define INCREASE_LZ_POS p->lzPos++; p->pointerToCurPos++;

/*
static
UInt32* MatchFinderMt_GetMatches_Bt4(CMatchFinderMt *p, UInt32 *d)
{
  const UInt32 *bt = p->btBufPos;
  const UInt32 len = *bt++;
  const UInt32 *btLim = bt + len;
  UInt32 matchMinPos;
  UInt32 avail = p->btNumAvailBytes - 1;
  p->btBufPos = btLim;

  {
    p->btNumAvailBytes = avail;

    #define BT_HASH_BYTES_MAX 5
      
    matchMinPos = p->lzPos;

    if (len != 0)
      matchMinPos -= bt[1];
    else if (avail < (BT_HASH_BYTES_MAX - 1) - 1)
    {
      INCREASE_LZ_POS
      return d;
    }
    else
    {
      const UInt32 hs = p->historySize;
      if (matchMinPos > hs)
        matchMinPos -= hs;
      else
        matchMinPos = 1;
    }
  }

  for (;;)
  {
  
  UInt32 h2, h3, c2, c3;
  UInt32 *hash = p->hash;
  const Byte *cur = p->pointerToCurPos;
  UInt32 m = p->lzPos;
  MT_HASH3_CALC

  c2 = hash[h2];
  c3 = (hash + kFix3HashSize)[h3];
 
  hash[h2] = m;
  (hash + kFix3HashSize)[h3] = m;

  if (c2 >= matchMinPos && cur[(ptrdiff_t)c2 - (ptrdiff_t)m] == cur[0])
  {
    d[1] = m - c2 - 1;
    if (cur[(ptrdiff_t)c2 - (ptrdiff_t)m + 2] == cur[2])
    {
      d[0] = 3;
      d += 2;
      break;
    }
    // else
    {
      d[0] = 2;
      d += 2;
    }
  }
  if (c3 >= matchMinPos && cur[(ptrdiff_t)c3 - (ptrdiff_t)m] == cur[0])
  {
    *d++ = 3;
    *d++ = m - c3 - 1;
  }
  break;
  }

  if (len != 0)
  {
    do
    {
      const UInt32 v0 = bt[0];
      const UInt32 v1 = bt[1];
      bt += 2;
      d[0] = v0;
      d[1] = v1;
      d += 2;
    }
    while (bt != btLim);
  }
  INCREASE_LZ_POS
  return d;
}
*/


static UInt32 * MixMatches4(CMatchFinderMt *p, UInt32 matchMinPos, UInt32 *d)
{
  UInt32 h2, h3, /* h4, */ c2, c3 /* , c4 */;
  UInt32 *hash = p->hash;
  const Byte *cur = p->pointerToCurPos;
  const UInt32 m = p->lzPos;
  MT_HASH3_CALC
  // MT_HASH4_CALC
  c2 = hash[h2];
  c3 = (hash + kFix3HashSize)[h3];
  // c4 = (hash + kFix4HashSize)[h4];
  
  hash[h2] = m;
  (hash + kFix3HashSize)[h3] = m;
  // (hash + kFix4HashSize)[h4] = m;

  // #define BT5_USE_H2
  // #ifdef BT5_USE_H2
  if (c2 >= matchMinPos && cur[(ptrdiff_t)c2 - (ptrdiff_t)m] == cur[0])
  {
    d[1] = m - c2 - 1;
    if (cur[(ptrdiff_t)c2 - (ptrdiff_t)m + 2] == cur[2])
    {
      // d[0] = (cur[(ptrdiff_t)c2 - (ptrdiff_t)m + 3] == cur[3]) ? 4 : 3;
      // return d + 2;

      if (cur[(ptrdiff_t)c2 - (ptrdiff_t)m + 3] == cur[3])
      {
        d[0] = 4;
        return d + 2;
      }
      d[0] = 3;
      d += 2;

      #ifdef BT5_USE_H4
      if (c4 >= matchMinPos)
        if (
          cur[(ptrdiff_t)c4 - (ptrdiff_t)m]     == cur[0] &&
          cur[(ptrdiff_t)c4 - (ptrdiff_t)m + 3] == cur[3]
          )
      {
        *d++ = 4;
        *d++ = m - c4 - 1;
      }
      #endif
      return d;
    }
    d[0] = 2;
    d += 2;
  }
  // #endif
  
  if (c3 >= matchMinPos && cur[(ptrdiff_t)c3 - (ptrdiff_t)m] == cur[0])
  {
    d[1] = m - c3 - 1;
    if (cur[(ptrdiff_t)c3 - (ptrdiff_t)m + 3] == cur[3])
    {
      d[0] = 4;
      return d + 2;
    }
    d[0] = 3;
    d += 2;
  }

  #ifdef BT5_USE_H4
  if (c4 >= matchMinPos)
    if (
      cur[(ptrdiff_t)c4 - (ptrdiff_t)m]     == cur[0] &&
      cur[(ptrdiff_t)c4 - (ptrdiff_t)m + 3] == cur[3]
      )
    {
      *d++ = 4;
      *d++ = m - c4 - 1;
    }
  #endif
  
  return d;
}


static UInt32 * MatchFinderMt2_GetMatches(void *_p, UInt32 *d)
{
  CMatchFinderMt *p = (CMatchFinderMt *)_p;
  const UInt32 *bt = p->btBufPos;
  const UInt32 len = *bt++;
  const UInt32 *btLim = bt + len;
  p->btBufPos = btLim;
  p->btNumAvailBytes--;
  INCREASE_LZ_POS
  {
    while (bt != btLim)
    {
      const UInt32 v0 = bt[0];
      const UInt32 v1 = bt[1];
      bt += 2;
      d[0] = v0;
      d[1] = v1;
      d += 2;
    }
  }
  return d;
}



static UInt32 * MatchFinderMt_GetMatches(void *_p, UInt32 *d)
{
  CMatchFinderMt *p = (CMatchFinderMt *)_p;
  const UInt32 *bt = p->btBufPos;
  UInt32 len = *bt++;
  const UInt32 avail = p->btNumAvailBytes - 1;
  p->btNumAvailBytes = avail;
  p->btBufPos = bt + len;
  if (len == 0)
  {
    #define BT_HASH_BYTES_MAX 5
    if (avail >= (BT_HASH_BYTES_MAX - 1) - 1)
    {
      UInt32 m = p->lzPos;
      if (m > p->historySize)
        m -= p->historySize;
      else
        m = 1;
      d = p->MixMatchesFunc(p, m, d);
    }
  }
  else
  {
    /*
      first match pair from BinTree: (match_len, match_dist),
      (match_len >= numHashBytes).
      MixMatchesFunc() inserts only hash matches that are nearer than (match_dist)
    */
    d = p->MixMatchesFunc(p, p->lzPos - bt[1], d);
    // if (d) // check for failure
    do
    {
      const UInt32 v0 = bt[0];
      const UInt32 v1 = bt[1];
      bt += 2;
      d[0] = v0;
      d[1] = v1;
      d += 2;
    }
    while (len -= 2);
  }
  INCREASE_LZ_POS
  return d;
}

#define SKIP_HEADER2_MT  do { GET_NEXT_BLOCK_IF_REQUIRED
#define SKIP_HEADER_MT(n) SKIP_HEADER2_MT if (p->btNumAvailBytes-- >= (n)) { const Byte *cur = p->pointerToCurPos; UInt32 *hash = p->hash;
#define SKIP_FOOTER_MT } INCREASE_LZ_POS p->btBufPos += (size_t)*p->btBufPos + 1; } while (--num != 0);

static void MatchFinderMt0_Skip(void *_p, UInt32 num)
{
  CMatchFinderMt *p = (CMatchFinderMt *)_p;
  SKIP_HEADER2_MT { p->btNumAvailBytes--;
  SKIP_FOOTER_MT
}

static void MatchFinderMt2_Skip(void *_p, UInt32 num)
{
  CMatchFinderMt *p = (CMatchFinderMt *)_p;
  SKIP_HEADER_MT(2)
      UInt32 h2;
      MT_HASH2_CALC
      hash[h2] = p->lzPos;
  SKIP_FOOTER_MT
}

static void MatchFinderMt3_Skip(void *_p, UInt32 num)
{
  CMatchFinderMt *p = (CMatchFinderMt *)_p;
  SKIP_HEADER_MT(3)
      UInt32 h2, h3;
      MT_HASH3_CALC
      (hash + kFix3HashSize)[h3] =
      hash[                h2] =
        p->lzPos;
  SKIP_FOOTER_MT
}

/*
// MatchFinderMt4_Skip() is similar to MatchFinderMt3_Skip().
// The difference is that MatchFinderMt3_Skip() updates hash for last 3 bytes of stream.

static void MatchFinderMt4_Skip(CMatchFinderMt *p, UInt32 num)
{
  SKIP_HEADER_MT(4)
      UInt32 h2, h3; // h4
      MT_HASH3_CALC
      // MT_HASH4_CALC
      // (hash + kFix4HashSize)[h4] =
      (hash + kFix3HashSize)[h3] =
      hash[                h2] =
        p->lzPos;
  SKIP_FOOTER_MT
}
*/

void MatchFinderMt_CreateVTable(CMatchFinderMt *p, IMatchFinder2 *vTable)
{
  vTable->Init = MatchFinderMt_Init;
  vTable->GetNumAvailableBytes = MatchFinderMt_GetNumAvailableBytes;
  vTable->GetPointerToCurrentPos = MatchFinderMt_GetPointerToCurrentPos;
  vTable->GetMatches = MatchFinderMt_GetMatches;
  
  switch (MF(p)->numHashBytes)
  {
    case 2:
      p->GetHeadsFunc = GetHeads2;
      p->MixMatchesFunc = NULL;
      vTable->Skip = MatchFinderMt0_Skip;
      vTable->GetMatches = MatchFinderMt2_GetMatches;
      break;
    case 3:
      p->GetHeadsFunc = MF(p)->bigHash ? GetHeads3b : GetHeads3;
      p->MixMatchesFunc = MixMatches2;
      vTable->Skip = MatchFinderMt2_Skip;
      break;
    case 4:
      p->GetHeadsFunc = MF(p)->bigHash ? GetHeads4b : GetHeads4;

      // it's fast inline version of GetMatches()
      // vTable->GetMatches = MatchFinderMt_GetMatches_Bt4;

      p->MixMatchesFunc = MixMatches3;
      vTable->Skip = MatchFinderMt3_Skip;
      break;
    default:
      p->GetHeadsFunc = MF(p)->bigHash ? GetHeads5b : GetHeads5;
      p->MixMatchesFunc = MixMatches4;
      vTable->Skip =
          MatchFinderMt3_Skip;
          // MatchFinderMt4_Skip;
      break;
  }
}

#undef RINOK_THREAD
#undef PRF
#undef MF
#undef GetUi24hi_from32
#undef LOCK_BUFFER
#undef UNLOCK_BUFFER
