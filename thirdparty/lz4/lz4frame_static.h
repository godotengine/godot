/*
   LZ4 auto-framing library
   Header File for static linking only
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

#ifndef LZ4FRAME_STATIC_H_0398209384
#define LZ4FRAME_STATIC_H_0398209384

#if defined (__cplusplus)
extern "C" {
#endif

/* lz4frame_static.h should be used solely in the context of static linking.
 * It contains definitions which are not stable and may change in the future.
 * Never use it in the context of DLL linking.
 * */


/* ---   Dependency   --- */
#include "lz4frame.h"


/* ---   Error List   --- */
#define LZ4F_LIST_ERRORS(ITEM) \
        ITEM(OK_NoError) ITEM(ERROR_GENERIC) \
        ITEM(ERROR_maxBlockSize_invalid) ITEM(ERROR_blockMode_invalid) ITEM(ERROR_contentChecksumFlag_invalid) \
        ITEM(ERROR_compressionLevel_invalid) \
        ITEM(ERROR_headerVersion_wrong) ITEM(ERROR_blockChecksum_unsupported) ITEM(ERROR_reservedFlag_set) \
        ITEM(ERROR_allocation_failed) \
        ITEM(ERROR_srcSize_tooLarge) ITEM(ERROR_dstMaxSize_tooSmall) \
        ITEM(ERROR_frameHeader_incomplete) ITEM(ERROR_frameType_unknown) ITEM(ERROR_frameSize_wrong) \
        ITEM(ERROR_srcPtr_wrong) \
        ITEM(ERROR_decompressionFailed) \
        ITEM(ERROR_headerChecksum_invalid) ITEM(ERROR_contentChecksum_invalid) \
        ITEM(ERROR_maxCode)

#define LZ4F_DISABLE_OLD_ENUMS   /* comment to enable deprecated enums */
#ifndef LZ4F_DISABLE_OLD_ENUMS
#  define LZ4F_GENERATE_ENUM(ENUM) LZ4F_##ENUM, ENUM = LZ4F_##ENUM,
#else
#  define LZ4F_GENERATE_ENUM(ENUM) LZ4F_##ENUM,
#endif
typedef enum { LZ4F_LIST_ERRORS(LZ4F_GENERATE_ENUM) } LZ4F_errorCodes;  /* enum is exposed, to handle specific errors; compare function result to -enum value */

LZ4F_errorCodes LZ4F_getErrorCode(size_t functionResult);


#if defined (__cplusplus)
}
#endif

#endif /* LZ4FRAME_STATIC_H_0398209384 */
