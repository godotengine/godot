/* Lzma86.h -- LZMA + x86 (BCJ) Filter
2023-03-03 : Igor Pavlov : Public domain */

#ifndef ZIP7_INC_LZMA86_H
#define ZIP7_INC_LZMA86_H

#include "7zTypes.h"

EXTERN_C_BEGIN

#define LZMA86_SIZE_OFFSET (1 + 5)
#define LZMA86_HEADER_SIZE (LZMA86_SIZE_OFFSET + 8)

/*
It's an example for LZMA + x86 Filter use.
You can use .lzma86 extension, if you write that stream to file.
.lzma86 header adds one additional byte to standard .lzma header.
.lzma86 header (14 bytes):
  Offset Size  Description
    0     1    = 0 - no filter, pure LZMA
               = 1 - x86 filter + LZMA
    1     1    lc, lp and pb in encoded form
    2     4    dictSize (little endian)
    6     8    uncompressed size (little endian)


Lzma86_Encode
-------------
level - compression level: 0 <= level <= 9, the default value for "level" is 5.

dictSize - The dictionary size in bytes. The maximum value is
        128 MB = (1 << 27) bytes for 32-bit version
          1 GB = (1 << 30) bytes for 64-bit version
     The default value is 16 MB = (1 << 24) bytes, for level = 5.
     It's recommended to use the dictionary that is larger than 4 KB and
     that can be calculated as (1 << N) or (3 << N) sizes.
     For better compression ratio dictSize must be >= inSize.

filterMode:
    SZ_FILTER_NO   - no Filter
    SZ_FILTER_YES  - x86 Filter
    SZ_FILTER_AUTO - it tries both alternatives to select best.
              Encoder will use 2 or 3 passes:
              2 passes when FILTER_NO provides better compression.
              3 passes when FILTER_YES provides better compression.

Lzma86Encode allocates Data with MyAlloc functions.
RAM Requirements for compressing:
  RamSize = dictionarySize * 11.5 + 6MB + FilterBlockSize
      filterMode     FilterBlockSize
     SZ_FILTER_NO         0
     SZ_FILTER_YES      inSize
     SZ_FILTER_AUTO     inSize


Return code:
  SZ_OK               - OK
  SZ_ERROR_MEM        - Memory allocation error
  SZ_ERROR_PARAM      - Incorrect paramater
  SZ_ERROR_OUTPUT_EOF - output buffer overflow
  SZ_ERROR_THREAD     - errors in multithreading functions (only for Mt version)
*/

enum ESzFilterMode
{
  SZ_FILTER_NO,
  SZ_FILTER_YES,
  SZ_FILTER_AUTO
};

SRes Lzma86_Encode(Byte *dest, size_t *destLen, const Byte *src, size_t srcLen,
    int level, UInt32 dictSize, int filterMode);


/*
Lzma86_GetUnpackSize:
  In:
    src      - input data
    srcLen   - input data size
  Out:
    unpackSize - size of uncompressed stream
  Return code:
    SZ_OK               - OK
    SZ_ERROR_INPUT_EOF  - Error in headers
*/

SRes Lzma86_GetUnpackSize(const Byte *src, SizeT srcLen, UInt64 *unpackSize);

/*
Lzma86_Decode:
  In:
    dest     - output data
    destLen  - output data size
    src      - input data
    srcLen   - input data size
  Out:
    destLen  - processed output size
    srcLen   - processed input size
  Return code:
    SZ_OK           - OK
    SZ_ERROR_DATA  - Data error
    SZ_ERROR_MEM   - Memory allocation error
    SZ_ERROR_UNSUPPORTED - unsupported file
    SZ_ERROR_INPUT_EOF - it needs more bytes in input buffer
*/

SRes Lzma86_Decode(Byte *dest, SizeT *destLen, const Byte *src, SizeT *srcLen);

EXTERN_C_END

#endif
