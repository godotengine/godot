/*
   LZ4 HC - High Compression Mode of LZ4
   Header File
   Copyright (C) 2011-2016, Yann Collet.
   BSD 2-Clause License (http://www.opensource.org/licenses/bsd-license.php)

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:

       * Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
       * Redistributions in binary form must reproduce the above
   copyright notice, this list of conditions and the following disclaimer
   in the documentation and/or other materials provided with the
   distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

   You can contact the author at :
   - LZ4 source repository : https://github.com/lz4/lz4
   - LZ4 public forum : https://groups.google.com/forum/#!forum/lz4c
*/
#ifndef LZ4_HC_H_19834876238432
#define LZ4_HC_H_19834876238432

#if defined (__cplusplus)
extern "C" {
#endif

/* --- Dependency --- */
/* note : lz4hc is not an independent module, it requires lz4.h/lz4.c for proper compilation */
#include "lz4.h"   /* stddef, LZ4LIB_API, LZ4_DEPRECATED */


/* --- Useful constants --- */
#define LZ4HC_CLEVEL_MIN         3
#define LZ4HC_CLEVEL_DEFAULT     9
#define LZ4HC_CLEVEL_OPT_MIN    11
#define LZ4HC_CLEVEL_MAX        12


/*-************************************
 *  Block Compression
 **************************************/
/*! LZ4_compress_HC() :
 * Compress data from `src` into `dst`, using the more powerful but slower "HC" algorithm.
 * `dst` must be already allocated.
 * Compression is guaranteed to succeed if `dstCapacity >= LZ4_compressBound(srcSize)` (see "lz4.h")
 * Max supported `srcSize` value is LZ4_MAX_INPUT_SIZE (see "lz4.h")
 * `compressionLevel` : Recommended values are between 4 and 9, although any value between 1 and LZ4HC_MAX_CLEVEL will work.
 *                      Values >LZ4HC_MAX_CLEVEL behave the same as LZ4HC_MAX_CLEVEL.
 * @return : the number of bytes written into 'dst'
 *           or 0 if compression fails.
 */
LZ4LIB_API int LZ4_compress_HC (const char* src, char* dst, int srcSize, int dstCapacity, int compressionLevel);


/* Note :
 *   Decompression functions are provided within "lz4.h" (BSD license)
 */


/*! LZ4_compress_HC_extStateHC() :
 * Same as LZ4_compress_HC(), but using an externally allocated memory segment for `state`.
 * `state` size is provided by LZ4_sizeofStateHC().
 * Memory segment must be aligned on 8-bytes boundaries (which a normal malloc() will do properly).
 */
LZ4LIB_API int LZ4_compress_HC_extStateHC(void* state, const char* src, char* dst, int srcSize, int maxDstSize, int compressionLevel);
LZ4LIB_API int LZ4_sizeofStateHC(void);


/*-************************************
 *  Streaming Compression
 *  Bufferless synchronous API
 **************************************/
 typedef union LZ4_streamHC_u LZ4_streamHC_t;   /* incomplete type (defined later) */

/*! LZ4_createStreamHC() and LZ4_freeStreamHC() :
 * These functions create and release memory for LZ4 HC streaming state.
 * Newly created states are automatically initialized.
 * Existing states can be re-used several times, using LZ4_resetStreamHC().
 * These methods are API and ABI stable, they can be used in combination with a DLL.
 */
LZ4LIB_API LZ4_streamHC_t* LZ4_createStreamHC(void);
LZ4LIB_API int             LZ4_freeStreamHC (LZ4_streamHC_t* streamHCPtr);

LZ4LIB_API void LZ4_resetStreamHC (LZ4_streamHC_t* streamHCPtr, int compressionLevel);
LZ4LIB_API int  LZ4_loadDictHC (LZ4_streamHC_t* streamHCPtr, const char* dictionary, int dictSize);

LZ4LIB_API int LZ4_compress_HC_continue (LZ4_streamHC_t* streamHCPtr, const char* src, char* dst, int srcSize, int maxDstSize);

LZ4LIB_API int LZ4_saveDictHC (LZ4_streamHC_t* streamHCPtr, char* safeBuffer, int maxDictSize);

/*
  These functions compress data in successive blocks of any size, using previous blocks as dictionary.
  One key assumption is that previous blocks (up to 64 KB) remain read-accessible while compressing next blocks.
  There is an exception for ring buffers, which can be smaller than 64 KB.
  Ring buffers scenario is automatically detected and handled by LZ4_compress_HC_continue().

  Before starting compression, state must be properly initialized, using LZ4_resetStreamHC().
  A first "fictional block" can then be designated as initial dictionary, using LZ4_loadDictHC() (Optional).

  Then, use LZ4_compress_HC_continue() to compress each successive block.
  Previous memory blocks (including initial dictionary when present) must remain accessible and unmodified during compression.
  'dst' buffer should be sized to handle worst case scenarios, using LZ4_compressBound(), to ensure operation success.

  If, for any reason, previous data blocks can't be preserved unmodified in memory during next compression block,
  you must save it to a safer memory space, using LZ4_saveDictHC().
  Return value of LZ4_saveDictHC() is the size of dictionary effectively saved into 'safeBuffer'.
*/


/*-******************************************
 * !!!!!   STATIC LINKING ONLY   !!!!!
 *******************************************/

 /*-*************************************
 * PRIVATE DEFINITIONS :
 * Do not use these definitions.
 * They are exposed to allow static allocation of `LZ4_streamHC_t`.
 * Using these definitions makes the code vulnerable to potential API break when upgrading LZ4
 **************************************/
#define LZ4HC_DICTIONARY_LOGSIZE 17
#define LZ4HC_MAXD (1<<LZ4HC_DICTIONARY_LOGSIZE)
#define LZ4HC_MAXD_MASK (LZ4HC_MAXD - 1)

#define LZ4HC_HASH_LOG 15
#define LZ4HC_HASHTABLESIZE (1 << LZ4HC_HASH_LOG)
#define LZ4HC_HASH_MASK (LZ4HC_HASHTABLESIZE - 1)


#if defined(__cplusplus) || (defined (__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) /* C99 */)
#include <stdint.h>

typedef struct
{
    uint32_t   hashTable[LZ4HC_HASHTABLESIZE];
    uint16_t   chainTable[LZ4HC_MAXD];
    const uint8_t* end;        /* next block here to continue on current prefix */
    const uint8_t* base;       /* All index relative to this position */
    const uint8_t* dictBase;   /* alternate base for extDict */
    uint8_t* inputBuffer;      /* deprecated */
    uint32_t   dictLimit;        /* below that point, need extDict */
    uint32_t   lowLimit;         /* below that point, no more dict */
    uint32_t   nextToUpdate;     /* index from which to continue dictionary update */
    uint32_t   searchNum;        /* only for optimal parser */
    uint32_t   compressionLevel;
} LZ4HC_CCtx_internal;

#else

typedef struct
{
    unsigned int   hashTable[LZ4HC_HASHTABLESIZE];
    unsigned short   chainTable[LZ4HC_MAXD];
    const unsigned char* end;        /* next block here to continue on current prefix */
    const unsigned char* base;       /* All index relative to this position */
    const unsigned char* dictBase;   /* alternate base for extDict */
    unsigned char* inputBuffer;      /* deprecated */
    unsigned int   dictLimit;        /* below that point, need extDict */
    unsigned int   lowLimit;         /* below that point, no more dict */
    unsigned int   nextToUpdate;     /* index from which to continue dictionary update */
    unsigned int   searchNum;        /* only for optimal parser */
    unsigned int   compressionLevel;
} LZ4HC_CCtx_internal;

#endif

#define LZ4_STREAMHCSIZE       (4*LZ4HC_HASHTABLESIZE + 2*LZ4HC_MAXD + 56) /* 393268 */
#define LZ4_STREAMHCSIZE_SIZET (LZ4_STREAMHCSIZE / sizeof(size_t))
union LZ4_streamHC_u {
    size_t table[LZ4_STREAMHCSIZE_SIZET];
    LZ4HC_CCtx_internal internal_donotuse;
};   /* previously typedef'd to LZ4_streamHC_t */
/*
  LZ4_streamHC_t :
  This structure allows static allocation of LZ4 HC streaming state.
  State must be initialized using LZ4_resetStreamHC() before first use.

  Static allocation shall only be used in combination with static linking.
  When invoking LZ4 from a DLL, use create/free functions instead, which are API and ABI stable.
*/


/*-************************************
*  Deprecated Functions
**************************************/
/* see lz4.h LZ4_DISABLE_DEPRECATE_WARNINGS to turn off deprecation warnings */

/* deprecated compression functions */
/* these functions will trigger warning messages in future releases */
LZ4_DEPRECATED("use LZ4_compress_HC() instead") int LZ4_compressHC               (const char* source, char* dest, int inputSize);
LZ4_DEPRECATED("use LZ4_compress_HC() instead") int LZ4_compressHC_limitedOutput (const char* source, char* dest, int inputSize, int maxOutputSize);
LZ4_DEPRECATED("use LZ4_compress_HC() instead") int LZ4_compressHC2 (const char* source, char* dest, int inputSize, int compressionLevel);
LZ4_DEPRECATED("use LZ4_compress_HC() instead") int LZ4_compressHC2_limitedOutput (const char* source, char* dest, int inputSize, int maxOutputSize, int compressionLevel);
LZ4_DEPRECATED("use LZ4_compress_HC_extStateHC() instead") int LZ4_compressHC_withStateHC               (void* state, const char* source, char* dest, int inputSize);
LZ4_DEPRECATED("use LZ4_compress_HC_extStateHC() instead") int LZ4_compressHC_limitedOutput_withStateHC (void* state, const char* source, char* dest, int inputSize, int maxOutputSize);
LZ4_DEPRECATED("use LZ4_compress_HC_extStateHC() instead") int LZ4_compressHC2_withStateHC (void* state, const char* source, char* dest, int inputSize, int compressionLevel);
LZ4_DEPRECATED("use LZ4_compress_HC_extStateHC() instead") int LZ4_compressHC2_limitedOutput_withStateHC(void* state, const char* source, char* dest, int inputSize, int maxOutputSize, int compressionLevel);
LZ4_DEPRECATED("use LZ4_compress_HC_continue() instead") int LZ4_compressHC_continue               (LZ4_streamHC_t* LZ4_streamHCPtr, const char* source, char* dest, int inputSize);
LZ4_DEPRECATED("use LZ4_compress_HC_continue() instead") int LZ4_compressHC_limitedOutput_continue (LZ4_streamHC_t* LZ4_streamHCPtr, const char* source, char* dest, int inputSize, int maxOutputSize);

/* Deprecated Streaming functions using older model; should no longer be used */
LZ4_DEPRECATED("use LZ4_createStreamHC() instead") void* LZ4_createHC (char* inputBuffer);
LZ4_DEPRECATED("use LZ4_saveDictHC() instead")     char* LZ4_slideInputBufferHC (void* LZ4HC_Data);
LZ4_DEPRECATED("use LZ4_freeStreamHC() instead")   int   LZ4_freeHC (void* LZ4HC_Data);
LZ4_DEPRECATED("use LZ4_compress_HC_continue() instead") int LZ4_compressHC2_continue (void* LZ4HC_Data, const char* source, char* dest, int inputSize, int compressionLevel);
LZ4_DEPRECATED("use LZ4_compress_HC_continue() instead") int LZ4_compressHC2_limitedOutput_continue (void* LZ4HC_Data, const char* source, char* dest, int inputSize, int maxOutputSize, int compressionLevel);
LZ4_DEPRECATED("use LZ4_createStreamHC() instead") int   LZ4_sizeofStreamStateHC(void);
LZ4_DEPRECATED("use LZ4_resetStreamHC() instead")  int   LZ4_resetStreamStateHC(void* state, char* inputBuffer);


#if defined (__cplusplus)
}
#endif

#endif /* LZ4_HC_H_19834876238432 */
