/* LzFindOpt.c -- multithreaded Match finder for LZ algorithms
2023-04-02 : Igor Pavlov : Public domain */

#include "Precomp.h"

#include "CpuArch.h"
#include "LzFind.h"

// #include "LzFindMt.h"

// #define LOG_ITERS

// #define LOG_THREAD

#ifdef LOG_THREAD
#include <stdio.h>
#define PRF(x) x
#else
// #define PRF(x)
#endif

#ifdef LOG_ITERS
#include <stdio.h>
UInt64 g_NumIters_Tree;
UInt64 g_NumIters_Loop;
UInt64 g_NumIters_Bytes;
#define LOG_ITER(x) x
#else
#define LOG_ITER(x)
#endif

// ---------- BT THREAD ----------

#define USE_SON_PREFETCH
#define USE_LONG_MATCH_OPT

#define kEmptyHashValue 0

// #define CYC_TO_POS_OFFSET 0

// #define CYC_TO_POS_OFFSET 1 // for debug

/*
Z7_NO_INLINE
UInt32 * Z7_FASTCALL GetMatchesSpecN_1(const Byte *lenLimit, size_t pos, const Byte *cur, CLzRef *son,
    UInt32 _cutValue, UInt32 *d, size_t _maxLen, const UInt32 *hash, const UInt32 *limit, const UInt32 *size, UInt32 *posRes)
{
  do
  {
    UInt32 delta;
    if (hash == size)
      break;
    delta = *hash++;

    if (delta == 0 || delta > (UInt32)pos)
      return NULL;

    lenLimit++;

    if (delta == (UInt32)pos)
    {
      CLzRef *ptr1 = son + ((size_t)pos << 1) - CYC_TO_POS_OFFSET * 2;
      *d++ = 0;
      ptr1[0] = kEmptyHashValue;
      ptr1[1] = kEmptyHashValue;
    }
else
{
  UInt32 *_distances = ++d;

  CLzRef *ptr0 = son + ((size_t)(pos) << 1) - CYC_TO_POS_OFFSET * 2 + 1;
  CLzRef *ptr1 = son + ((size_t)(pos) << 1) - CYC_TO_POS_OFFSET * 2;

  const Byte *len0 = cur, *len1 = cur;
  UInt32 cutValue = _cutValue;
  const Byte *maxLen = cur + _maxLen;

  for (LOG_ITER(g_NumIters_Tree++);;)
  {
    LOG_ITER(g_NumIters_Loop++);
    {
      const ptrdiff_t diff = (ptrdiff_t)0 - (ptrdiff_t)delta;
      CLzRef *pair = son + ((size_t)(((ptrdiff_t)pos - CYC_TO_POS_OFFSET) + diff) << 1);
      const Byte *len = (len0 < len1 ? len0 : len1);

    #ifdef USE_SON_PREFETCH
      const UInt32 pair0 = *pair;
    #endif

      if (len[diff] == len[0])
      {
        if (++len != lenLimit && len[diff] == len[0])
          while (++len != lenLimit)
          {
            LOG_ITER(g_NumIters_Bytes++);
            if (len[diff] != len[0])
              break;
          }
        if (maxLen < len)
        {
          maxLen = len;
          *d++ = (UInt32)(len - cur);
          *d++ = delta - 1;
          
          if (len == lenLimit)
          {
            const UInt32 pair1 = pair[1];
            *ptr1 =
              #ifdef USE_SON_PREFETCH
                pair0;
              #else
                pair[0];
              #endif
            *ptr0 = pair1;

            _distances[-1] = (UInt32)(d - _distances);

            #ifdef USE_LONG_MATCH_OPT

                if (hash == size || *hash != delta || lenLimit[diff] != lenLimit[0] || d >= limit)
                  break;

            {
              for (;;)
              {
                hash++;
                pos++;
                cur++;
                lenLimit++;
                {
                  CLzRef *ptr = son + ((size_t)(pos) << 1) - CYC_TO_POS_OFFSET * 2;
                  #if 0
                  *(UInt64 *)(void *)ptr = ((const UInt64 *)(const void *)ptr)[diff];
                  #else
                  const UInt32 p0 = ptr[0 + (diff * 2)];
                  const UInt32 p1 = ptr[1 + (diff * 2)];
                  ptr[0] = p0;
                  ptr[1] = p1;
                  // ptr[0] = ptr[0 + (diff * 2)];
                  // ptr[1] = ptr[1 + (diff * 2)];
                  #endif
                }
                // PrintSon(son + 2, pos - 1);
                // printf("\npos = %x delta = %x\n", pos, delta);
                len++;
                *d++ = 2;
                *d++ = (UInt32)(len - cur);
                *d++ = delta - 1;
                if (hash == size || *hash != delta || lenLimit[diff] != lenLimit[0] || d >= limit)
                  break;
              }
            }
            #endif

            break;
          }
        }
      }

      {
        const UInt32 curMatch = (UInt32)pos - delta; // (UInt32)(pos + diff);
        if (len[diff] < len[0])
        {
          delta = pair[1];
          if (delta >= curMatch)
            return NULL;
          *ptr1 = curMatch;
          ptr1 = pair + 1;
          len1 = len;
        }
        else
        {
          delta = *pair;
          if (delta >= curMatch)
            return NULL;
          *ptr0 = curMatch;
          ptr0 = pair;
          len0 = len;
        }

        delta = (UInt32)pos - delta;
 
        if (--cutValue == 0 || delta >= pos)
        {
          *ptr0 = *ptr1 = kEmptyHashValue;
          _distances[-1] = (UInt32)(d - _distances);
          break;
        }
      }
    }
  } // for (tree iterations)
}
    pos++;
    cur++;
  }
  while (d < limit);
  *posRes = (UInt32)pos;
  return d;
}
*/

/* define cbs if you use 2 functions.
       GetMatchesSpecN_1() :  (pos <  _cyclicBufferSize)
       GetMatchesSpecN_2() :  (pos >= _cyclicBufferSize)

  do not define cbs if you use 1 function:
       GetMatchesSpecN_2()
*/

// #define cbs _cyclicBufferSize

/*
  we use size_t for (pos) and (_cyclicBufferPos_ instead of UInt32
  to eliminate "movsx" BUG in old MSVC x64 compiler.
*/

UInt32 * Z7_FASTCALL GetMatchesSpecN_2(const Byte *lenLimit, size_t pos, const Byte *cur, CLzRef *son,
    UInt32 _cutValue, UInt32 *d, size_t _maxLen, const UInt32 *hash, const UInt32 *limit, const UInt32 *size,
    size_t _cyclicBufferPos, UInt32 _cyclicBufferSize,
    UInt32 *posRes);

Z7_NO_INLINE
UInt32 * Z7_FASTCALL GetMatchesSpecN_2(const Byte *lenLimit, size_t pos, const Byte *cur, CLzRef *son,
    UInt32 _cutValue, UInt32 *d, size_t _maxLen, const UInt32 *hash, const UInt32 *limit, const UInt32 *size,
    size_t _cyclicBufferPos, UInt32 _cyclicBufferSize,
    UInt32 *posRes)
{
  do // while (hash != size)
  {
    UInt32 delta;
    
  #ifndef cbs
    UInt32 cbs;
  #endif

    if (hash == size)
      break;

    delta = *hash++;

    if (delta == 0)
      return NULL;

    lenLimit++;

  #ifndef cbs
    cbs = _cyclicBufferSize;
    if ((UInt32)pos < cbs)
    {
      if (delta > (UInt32)pos)
        return NULL;
      cbs = (UInt32)pos;
    }
  #endif

    if (delta >= cbs)
    {
      CLzRef *ptr1 = son + ((size_t)_cyclicBufferPos << 1);
      *d++ = 0;
      ptr1[0] = kEmptyHashValue;
      ptr1[1] = kEmptyHashValue;
    }
else
{
  UInt32 *_distances = ++d;

  CLzRef *ptr0 = son + ((size_t)_cyclicBufferPos << 1) + 1;
  CLzRef *ptr1 = son + ((size_t)_cyclicBufferPos << 1);

  UInt32 cutValue = _cutValue;
  const Byte *len0 = cur, *len1 = cur;
  const Byte *maxLen = cur + _maxLen;

  // if (cutValue == 0) { *ptr0 = *ptr1 = kEmptyHashValue; } else
  for (LOG_ITER(g_NumIters_Tree++);;)
  {
    LOG_ITER(g_NumIters_Loop++);
    {
      // SPEC code
      CLzRef *pair = son + ((size_t)((ptrdiff_t)_cyclicBufferPos - (ptrdiff_t)delta
          + (ptrdiff_t)(UInt32)(_cyclicBufferPos < delta ? cbs : 0)
          ) << 1);

      const ptrdiff_t diff = (ptrdiff_t)0 - (ptrdiff_t)delta;
      const Byte *len = (len0 < len1 ? len0 : len1);

    #ifdef USE_SON_PREFETCH
      const UInt32 pair0 = *pair;
    #endif

      if (len[diff] == len[0])
      {
        if (++len != lenLimit && len[diff] == len[0])
          while (++len != lenLimit)
          {
            LOG_ITER(g_NumIters_Bytes++);
            if (len[diff] != len[0])
              break;
          }
        if (maxLen < len)
        {
          maxLen = len;
          *d++ = (UInt32)(len - cur);
          *d++ = delta - 1;
          
          if (len == lenLimit)
          {
            const UInt32 pair1 = pair[1];
            *ptr1 =
              #ifdef USE_SON_PREFETCH
                pair0;
              #else
                pair[0];
              #endif
            *ptr0 = pair1;

            _distances[-1] = (UInt32)(d - _distances);

            #ifdef USE_LONG_MATCH_OPT

                if (hash == size || *hash != delta || lenLimit[diff] != lenLimit[0] || d >= limit)
                  break;

            {
              for (;;)
              {
                *d++ = 2;
                *d++ = (UInt32)(lenLimit - cur);
                *d++ = delta - 1;
                cur++;
                lenLimit++;
                // SPEC
                _cyclicBufferPos++;
                {
                  // SPEC code
                  CLzRef *dest = son + ((size_t)(_cyclicBufferPos) << 1);
                  const CLzRef *src = dest + ((diff
                      + (ptrdiff_t)(UInt32)((_cyclicBufferPos < delta) ? cbs : 0)) << 1);
                  // CLzRef *ptr = son + ((size_t)(pos) << 1) - CYC_TO_POS_OFFSET * 2;
                  #if 0
                  *(UInt64 *)(void *)dest = *((const UInt64 *)(const void *)src);
                  #else
                  const UInt32 p0 = src[0];
                  const UInt32 p1 = src[1];
                  dest[0] = p0;
                  dest[1] = p1;
                  #endif
                }
                pos++;
                hash++;
                if (hash == size || *hash != delta || lenLimit[diff] != lenLimit[0] || d >= limit)
                  break;
              } // for() end for long matches
            }
            #endif

            break; // break from TREE iterations
          }
        }
      }
      {
        const UInt32 curMatch = (UInt32)pos - delta; // (UInt32)(pos + diff);
        if (len[diff] < len[0])
        {
          delta = pair[1];
          *ptr1 = curMatch;
          ptr1 = pair + 1;
          len1 = len;
          if (delta >= curMatch)
            return NULL;
        }
        else
        {
          delta = *pair;
          *ptr0 = curMatch;
          ptr0 = pair;
          len0 = len;
          if (delta >= curMatch)
            return NULL;
        }
        delta = (UInt32)pos - delta;
 
        if (--cutValue == 0 || delta >= cbs)
        {
          *ptr0 = *ptr1 = kEmptyHashValue;
          _distances[-1] = (UInt32)(d - _distances);
          break;
        }
      }
    }
  } // for (tree iterations)
}
    pos++;
    _cyclicBufferPos++;
    cur++;
  }
  while (d < limit);
  *posRes = (UInt32)pos;
  return d;
}



/*
typedef UInt32 uint32plus; // size_t

UInt32 * Z7_FASTCALL GetMatchesSpecN_3(uint32plus lenLimit, size_t pos, const Byte *cur, CLzRef *son,
    UInt32 _cutValue, UInt32 *d, uint32plus _maxLen, const UInt32 *hash, const UInt32 *limit, const UInt32 *size,
    size_t _cyclicBufferPos, UInt32 _cyclicBufferSize,
    UInt32 *posRes)
{
  do // while (hash != size)
  {
    UInt32 delta;

  #ifndef cbs
    UInt32 cbs;
  #endif

    if (hash == size)
      break;

    delta = *hash++;

    if (delta == 0)
      return NULL;

  #ifndef cbs
    cbs = _cyclicBufferSize;
    if ((UInt32)pos < cbs)
    {
      if (delta > (UInt32)pos)
        return NULL;
      cbs = (UInt32)pos;
    }
  #endif
    
    if (delta >= cbs)
    {
      CLzRef *ptr1 = son + ((size_t)_cyclicBufferPos << 1);
      *d++ = 0;
      ptr1[0] = kEmptyHashValue;
      ptr1[1] = kEmptyHashValue;
    }
else
{
  CLzRef *ptr0 = son + ((size_t)_cyclicBufferPos << 1) + 1;
  CLzRef *ptr1 = son + ((size_t)_cyclicBufferPos << 1);
  UInt32 *_distances = ++d;
  uint32plus len0 = 0, len1 = 0;
  UInt32 cutValue = _cutValue;
  uint32plus maxLen = _maxLen;
  // lenLimit++; // const Byte *lenLimit = cur + _lenLimit;

  for (LOG_ITER(g_NumIters_Tree++);;)
  {
    LOG_ITER(g_NumIters_Loop++);
    {
      // const ptrdiff_t diff = (ptrdiff_t)0 - (ptrdiff_t)delta;
      CLzRef *pair = son + ((size_t)((ptrdiff_t)_cyclicBufferPos - delta
          + (ptrdiff_t)(UInt32)(_cyclicBufferPos < delta ? cbs : 0)
          ) << 1);
      const Byte *pb = cur - delta;
      uint32plus len = (len0 < len1 ? len0 : len1);

    #ifdef USE_SON_PREFETCH
      const UInt32 pair0 = *pair;
    #endif

      if (pb[len] == cur[len])
      {
        if (++len != lenLimit && pb[len] == cur[len])
          while (++len != lenLimit)
            if (pb[len] != cur[len])
              break;
        if (maxLen < len)
        {
          maxLen = len;
          *d++ = (UInt32)len;
          *d++ = delta - 1;
          if (len == lenLimit)
          {
            {
              const UInt32 pair1 = pair[1];
              *ptr0 = pair1;
              *ptr1 =
              #ifdef USE_SON_PREFETCH
                pair0;
              #else
                pair[0];
              #endif
            }

            _distances[-1] = (UInt32)(d - _distances);

            #ifdef USE_LONG_MATCH_OPT

                if (hash == size || *hash != delta || pb[lenLimit] != cur[lenLimit] || d >= limit)
                  break;

            {
              const ptrdiff_t diff = (ptrdiff_t)0 - (ptrdiff_t)delta;
              for (;;)
              {
                *d++ = 2;
                *d++ = (UInt32)lenLimit;
                *d++ = delta - 1;
                _cyclicBufferPos++;
                {
                  CLzRef *dest = son + ((size_t)_cyclicBufferPos << 1);
                  const CLzRef *src = dest + ((diff +
                      (ptrdiff_t)(UInt32)(_cyclicBufferPos < delta ? cbs : 0)) << 1);
                #if 0
                  *(UInt64 *)(void *)dest = *((const UInt64 *)(const void *)src);
                #else
                  const UInt32 p0 = src[0];
                  const UInt32 p1 = src[1];
                  dest[0] = p0;
                  dest[1] = p1;
                #endif
                }
                hash++;
                pos++;
                cur++;
                pb++;
                if (hash == size || *hash != delta || pb[lenLimit] != cur[lenLimit] || d >= limit)
                  break;
              }
            }
            #endif

            break;
          }
        }
      }
      {
        const UInt32 curMatch = (UInt32)pos - delta;
        if (pb[len] < cur[len])
        {
          delta = pair[1];
          *ptr1 = curMatch;
          ptr1 = pair + 1;
          len1 = len;
        }
        else
        {
          delta = *pair;
          *ptr0 = curMatch;
          ptr0 = pair;
          len0 = len;
        }

        {
          if (delta >= curMatch)
            return NULL;
          delta = (UInt32)pos - delta;
          if (delta >= cbs
              // delta >= _cyclicBufferSize || delta >= pos
              || --cutValue == 0)
          {
            *ptr0 = *ptr1 = kEmptyHashValue;
            _distances[-1] = (UInt32)(d - _distances);
            break;
          }
        }
      }
    }
  } // for (tree iterations)
}
    pos++;
    _cyclicBufferPos++;
    cur++;
  }
  while (d < limit);
  *posRes = (UInt32)pos;
  return d;
}
*/
