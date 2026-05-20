LZMA compression
----------------
Version: 24.07

This file describes LZMA encoding and decoding functions written in C language.

LZMA is an improved version of famous LZ77 compression algorithm. 
It was improved in way of maximum increasing of compression ratio,
keeping high decompression speed and low memory requirements for 
decompressing.

Note: you can read also LZMA Specification (lzma-specification.txt from LZMA SDK)

Also you can look source code for LZMA encoding and decoding:
  C/Util/Lzma/LzmaUtil.c


LZMA compressed file format
---------------------------
Offset Size Description
  0     1   Special LZMA properties (lc,lp, pb in encoded form)
  1     4   Dictionary size (little endian)
  5     8   Uncompressed size (little endian). -1 means unknown size
 13         Compressed data



ANSI-C LZMA Decoder
~~~~~~~~~~~~~~~~~~~

Please note that interfaces for ANSI-C code were changed in LZMA SDK 4.58.
If you want to use old interfaces you can download previous version of LZMA SDK
from sourceforge.net site.

To use ANSI-C LZMA Decoder you need the following files:
1) LzmaDec.h + LzmaDec.c + 7zTypes.h + Precomp.h + Compiler.h

Look example code:
  C/Util/Lzma/LzmaUtil.c


Memory requirements for LZMA decoding
-------------------------------------

Stack usage of LZMA decoding function for local variables is not 
larger than 200-400 bytes.

LZMA Decoder uses dictionary buffer and internal state structure.
Internal state structure consumes
  state_size = (4 + (1.5 << (lc + lp))) KB
by default (lc=3, lp=0), state_size = 16 KB.


How To decompress data
----------------------

LZMA Decoder (ANSI-C version) now supports 2 interfaces:
1) Single-call Decompressing
2) Multi-call State Decompressing (zlib-like interface)

You must use external allocator:
Example:
void *SzAlloc(void *p, size_t size) { p = p; return malloc(size); }
void SzFree(void *p, void *address) { p = p; free(address); }
ISzAlloc alloc = { SzAlloc, SzFree };

You can use p = p; operator to disable compiler warnings.


Single-call Decompressing
-------------------------
When to use: RAM->RAM decompressing
Compile files: LzmaDec.h + LzmaDec.c + 7zTypes.h
Compile defines: no defines
Memory Requirements:
  - Input buffer: compressed size
  - Output buffer: uncompressed size
  - LZMA Internal Structures: state_size (16 KB for default settings) 

Interface:
  int LzmaDecode(Byte *dest, SizeT *destLen, const Byte *src, SizeT *srcLen,
      const Byte *propData, unsigned propSize, ELzmaFinishMode finishMode, 
      ELzmaStatus *status, ISzAlloc *alloc);
  In: 
    dest     - output data
    destLen  - output data size
    src      - input data
    srcLen   - input data size
    propData - LZMA properties  (5 bytes)
    propSize - size of propData buffer (5 bytes)
    finishMode - It has meaning only if the decoding reaches output limit (*destLen).
         LZMA_FINISH_ANY - Decode just destLen bytes.
         LZMA_FINISH_END - Stream must be finished after (*destLen).
                           You can use LZMA_FINISH_END, when you know that 
                           current output buffer covers last bytes of stream. 
    alloc    - Memory allocator.

  Out: 
    destLen  - processed output size 
    srcLen   - processed input size 

  Output:
    SZ_OK
      status:
        LZMA_STATUS_FINISHED_WITH_MARK
        LZMA_STATUS_NOT_FINISHED 
        LZMA_STATUS_MAYBE_FINISHED_WITHOUT_MARK
    SZ_ERROR_DATA - Data error
    SZ_ERROR_MEM  - Memory allocation error
    SZ_ERROR_UNSUPPORTED - Unsupported properties
    SZ_ERROR_INPUT_EOF - It needs more bytes in input buffer (src).

  If LZMA decoder sees end_marker before reaching output limit, it returns OK result,
  and output value of destLen will be less than output buffer size limit.

  You can use multiple checks to test data integrity after full decompression:
    1) Check Result and "status" variable.
    2) Check that output(destLen) = uncompressedSize, if you know real uncompressedSize.
    3) Check that output(srcLen) = compressedSize, if you know real compressedSize. 
       You must use correct finish mode in that case. */ 


Multi-call State Decompressing (zlib-like interface)
----------------------------------------------------

When to use: file->file decompressing 
Compile files: LzmaDec.h + LzmaDec.c + 7zTypes.h

Memory Requirements:
 - Buffer for input stream: any size (for example, 16 KB)
 - Buffer for output stream: any size (for example, 16 KB)
 - LZMA Internal Structures: state_size (16 KB for default settings) 
 - LZMA dictionary (dictionary size is encoded in LZMA properties header)

1) read LZMA properties (5 bytes) and uncompressed size (8 bytes, little-endian) to header:
   unsigned char header[LZMA_PROPS_SIZE + 8];
   ReadFile(inFile, header, sizeof(header)

2) Allocate CLzmaDec structures (state + dictionary) using LZMA properties

  CLzmaDec state;
  LzmaDec_Constr(&state);
  res = LzmaDec_Allocate(&state, header, LZMA_PROPS_SIZE, &g_Alloc);
  if (res != SZ_OK)
    return res;

3) Init LzmaDec structure before any new LZMA stream. And call LzmaDec_DecodeToBuf in loop

  LzmaDec_Init(&state);
  for (;;)
  {
    ... 
    int res = LzmaDec_DecodeToBuf(CLzmaDec *p, Byte *dest, SizeT *destLen, 
        const Byte *src, SizeT *srcLen, ELzmaFinishMode finishMode);
    ...
  }


4) Free all allocated structures
  LzmaDec_Free(&state, &g_Alloc);

Look example code:
  C/Util/Lzma/LzmaUtil.c


How To compress data
--------------------

Compile files: 
  7zTypes.h
  Threads.h	
  Threads.c	
  LzmaEnc.h
  LzmaEnc.c
  LzFind.h
  LzFind.c
  LzFindMt.h
  LzFindMt.c
  LzFindOpt.c
  LzHash.h

Memory Requirements:
  - (dictSize * 11.5 + 6 MB) + state_size

Lzma Encoder can use two memory allocators:
1) alloc - for small arrays.
2) allocBig - for big arrays.

For example, you can use Large RAM Pages (2 MB) in allocBig allocator for 
better compression speed. Note that Windows has bad implementation for 
Large RAM Pages. 
It's OK to use same allocator for alloc and allocBig.


Single-call Compression with callbacks
--------------------------------------

Look example code:
  C/Util/Lzma/LzmaUtil.c

When to use: file->file compressing 

1) you must implement callback structures for interfaces:
ISeqInStream
ISeqOutStream
ICompressProgress
ISzAlloc

static void *SzAlloc(void *p, size_t size) { p = p; return MyAlloc(size); }
static void SzFree(void *p, void *address) {  p = p; MyFree(address); }
static ISzAlloc g_Alloc = { SzAlloc, SzFree };

  CFileSeqInStream inStream;
  CFileSeqOutStream outStream;

  inStream.funcTable.Read = MyRead;
  inStream.file = inFile;
  outStream.funcTable.Write = MyWrite;
  outStream.file = outFile;


2) Create CLzmaEncHandle object;

  CLzmaEncHandle enc;

  enc = LzmaEnc_Create(&g_Alloc);
  if (enc == 0)
    return SZ_ERROR_MEM;


3) initialize CLzmaEncProps properties;

  LzmaEncProps_Init(&props);

  Then you can change some properties in that structure.

4) Send LZMA properties to LZMA Encoder

  res = LzmaEnc_SetProps(enc, &props);

5) Write encoded properties to header

    Byte header[LZMA_PROPS_SIZE + 8];
    size_t headerSize = LZMA_PROPS_SIZE;
    UInt64 fileSize;
    int i;

    res = LzmaEnc_WriteProperties(enc, header, &headerSize);
    fileSize = MyGetFileLength(inFile);
    for (i = 0; i < 8; i++)
      header[headerSize++] = (Byte)(fileSize >> (8 * i));
    MyWriteFileAndCheck(outFile, header, headerSize)

6) Call encoding function:
      res = LzmaEnc_Encode(enc, &outStream.funcTable, &inStream.funcTable, 
        NULL, &g_Alloc, &g_Alloc);

7) Destroy LZMA Encoder Object
  LzmaEnc_Destroy(enc, &g_Alloc, &g_Alloc);


If callback function return some error code, LzmaEnc_Encode also returns that code
or it can return the code like SZ_ERROR_READ, SZ_ERROR_WRITE or SZ_ERROR_PROGRESS.


Single-call RAM->RAM Compression
--------------------------------

Single-call RAM->RAM Compression is similar to Compression with callbacks,
but you provide pointers to buffers instead of pointers to stream callbacks:

SRes LzmaEncode(Byte *dest, SizeT *destLen, const Byte *src, SizeT srcLen,
    const CLzmaEncProps *props, Byte *propsEncoded, SizeT *propsSize, int writeEndMark, 
    ICompressProgress *progress, ISzAlloc *alloc, ISzAlloc *allocBig);

Return code:
  SZ_OK               - OK
  SZ_ERROR_MEM        - Memory allocation error 
  SZ_ERROR_PARAM      - Incorrect paramater
  SZ_ERROR_OUTPUT_EOF - output buffer overflow
  SZ_ERROR_THREAD     - errors in multithreading functions (only for Mt version)



Defines
-------

Z7_LZMA_SIZE_OPT - Enable some code size optimizations in LZMA Decoder to get smaller executable code.

Z7_LZMA_PROB32   - It can increase the speed on some 32-bit CPUs, but memory usage for 
                   some structures will be doubled in that case.

Z7_DECL_Int32_AS_long  - Define it if int is 16-bit on your compiler and long is 32-bit.

Z7_DECL_SizeT_AS_unsigned_int  - Define it if you don't want to use size_t type.


Defines for 7z decoder written in C
-----------------------------------
These defines are for 7zDec.c only (the decoder in C). 
C++ 7z decoder doesn't uses these macros.

Z7_PPMD_SUPPORT        - define it if you need PPMD method support.
Z7_NO_METHODS_FILTERS  - do not use filters (except of BCJ2 filter).
Z7_USE_NATIVE_BRANCH_FILTER - use filter for native ISA:
                                 use x86 filter, if compiled to x86 executable,
                		 use arm64 filter, if compiled to arm64 executable. 


C++ LZMA Encoder/Decoder 
~~~~~~~~~~~~~~~~~~~~~~~~
C++ LZMA code use COM-like interfaces. So if you want to use it, 
you can study basics of COM/OLE.
C++ LZMA code is just wrapper over ANSI-C code.


C++ Notes
~~~~~~~~~~~~~~~~~~~~~~~~
If you use some C++ code folders in 7-Zip (for example, C++ code for 7z archive handling),
you must check that you correctly work with "new" operator.
7-Zip can be compiled with MSVC 6.0 that doesn't throw "exception" from "new" operator.
So 7-Zip uses "CPP\Common\NewHandler.cpp" that redefines "new" operator,
if compiled by old MSVC compilers (MSVC before version VS 2010):

operator new(size_t size)
{
  void *p = ::malloc(size);
  if (!p)
    throw CNewException();
  return p;
}

If the compiler is VS 2010 or newer, NewHandler.cpp doesn't redefine "new" operator.
Sp if you use new compiler (VS 2010 or newer), you still can include "NewHandler.cpp" 
to compilation, and it will not redefine operator new.
Also you can compile without "NewHandler.cpp" with new compilers. 
If 7-zip doesn't redefine operator "new", standard exception will be used instead of CNewException. 
Some code of 7-Zip catches any exception in internal code and converts it to HRESULT code.
So you don't need to catch CNewException, if you call COM interfaces of 7-Zip.

---

http://www.7-zip.org
http://www.7-zip.org/sdk.html
http://www.7-zip.org/support.html
