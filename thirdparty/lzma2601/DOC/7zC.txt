7z ANSI-C Decoder 9.35
----------------------

7z ANSI-C provides 7z/LZMA decoding.
7z ANSI-C version is simplified version ported from C++ code.

LZMA is default and general compression method of 7z format
in 7-Zip compression program (www.7-zip.org). LZMA provides high 
compression ratio and very fast decompression.


LICENSE
-------

7z ANSI-C Decoder is part of the LZMA SDK.
LZMA SDK is written and placed in the public domain by Igor Pavlov.

Files
---------------------

7zDecode.*   - Low level 7z decoding
7zExtract.*  - High level 7z decoding
7zHeader.*   - .7z format constants
7zIn.*       - .7z archive opening
7zItem.*     - .7z structures
7zMain.c     - Test application


How To Use
----------

You can create .7z archive with 7z.exe, 7za.exe or 7zr.exe:

  7z.exe a archive.7z *.htm -r -mx -m0fb=255

If you have big number of files in archive, and you need fast extracting, 
you can use partly-solid archives:
  
  7za.exe a archive.7z *.htm -ms=512K -r -mx -m0fb=255 -m0d=512K

In that example 7-Zip will use 512KB solid blocks. So it needs to decompress only 
512KB for extracting one file from such archive.


Limitations of current version of 7z ANSI-C Decoder
---------------------------------------------------

 - It reads only "FileName", "Size", "LastWriteTime" and "CRC" information for each file in archive.
 - It supports only LZMA and Copy (no compression) methods with BCJ or BCJ2 filters.
 - It converts original UTF-16 Unicode file names to UTF-8 Unicode file names.
 
These limitations will be fixed in future versions.


Using 7z ANSI-C Decoder Test application:
-----------------------------------------

Usage: 7zDec <command> <archive_name>

<Command>:
  e: Extract files from archive
  l: List contents of archive
  t: Test integrity of archive

Example: 

  7zDec l archive.7z

lists contents of archive.7z

  7zDec e archive.7z

extracts files from archive.7z to current folder.


How to use .7z Decoder
----------------------

Memory allocation
~~~~~~~~~~~~~~~~~

7z Decoder uses two memory pools:
1) Temporary pool
2) Main pool
Such scheme can allow you to avoid fragmentation of allocated blocks.


Steps for using 7z decoder
--------------------------

Use code at 7zMain.c as example.

1) Declare variables:
  inStream                 /* implements ILookInStream interface */
  CSzArEx db;              /* 7z archive database structure */
  ISzAlloc allocImp;       /* memory functions for main pool */
  ISzAlloc allocTempImp;   /* memory functions for temporary pool */

2) call CrcGenerateTable(); function to initialize CRC structures.

3) call SzArEx_Init(&db); function to initialize db structures.

4) call SzArEx_Open(&db, inStream, &allocMain, &allocTemp) to open archive

This function opens archive "inStream" and reads headers to "db".
All items in "db" will be allocated with "allocMain" functions.
SzArEx_Open function allocates and frees temporary structures by "allocTemp" functions.

5) List items or Extract items

  Listing code:
  ~~~~~~~~~~~~~

    Use SzArEx_GetFileNameUtf16 function. Look example code in C\Util\7z\7zMain.c file. 
    

  Extracting code:
  ~~~~~~~~~~~~~~~~

  SZ_RESULT SzAr_Extract(
    CArchiveDatabaseEx *db,
    ILookInStream *inStream, 
    UInt32 fileIndex,         /* index of file */
    UInt32 *blockIndex,       /* index of solid block */
    Byte **outBuffer,         /* pointer to pointer to output buffer (allocated with allocMain) */
    size_t *outBufferSize,    /* buffer size for output buffer */
    size_t *offset,           /* offset of stream for required file in *outBuffer */
    size_t *outSizeProcessed, /* size of file in *outBuffer */
    ISzAlloc *allocMain,
    ISzAlloc *allocTemp);

  If you need to decompress more than one file, you can send these values from previous call:
    blockIndex, 
    outBuffer, 
    outBufferSize,
  You can consider "outBuffer" as cache of solid block. If your archive is solid, 
  it will increase decompression speed.

  After decompressing you must free "outBuffer":
  allocImp.Free(outBuffer);

6) call SzArEx_Free(&db, allocImp.Free) to free allocated items in "db".




Memory requirements for .7z decoding 
------------------------------------

Memory usage for Archive opening:
  - Temporary pool:
     - Memory for uncompressed .7z headers
     - some other temporary blocks
  - Main pool:
     - Memory for database: 
       Estimated size of one file structures in solid archive:
         - Size (4 or 8 Bytes)
         - CRC32 (4 bytes)
         - LastWriteTime (8 bytes)
         - Some file information (4 bytes)
         - File Name (variable length) + pointer + allocation structures

Memory usage for archive Decompressing:
  - Temporary pool:
     - Memory for LZMA decompressing structures
  - Main pool:
     - Memory for decompressed solid block
     - Memory for temprorary buffers, if BCJ2 fileter is used. Usually these 
       temprorary buffers can be about 15% of solid block size. 
  

7z Decoder doesn't allocate memory for compressed blocks. 
Instead of this, you must allocate buffer with desired 
size before calling 7z Decoder. Use 7zMain.c as example.


Defines
-------

_SZ_ALLOC_DEBUG   - define it if you want to debug alloc/free operations to stderr.


---

http://www.7-zip.org
http://www.7-zip.org/sdk.html
http://www.7-zip.org/support.html
