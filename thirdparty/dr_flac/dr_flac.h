/*
FLAC audio decoder. Choice of public domain or MIT-0. See license statements at the end of this file.
dr_flac - v0.12.41 - 2023-06-17

David Reid - mackron@gmail.com

GitHub: https://github.com/mackron/dr_libs
*/

/*
RELEASE NOTES - v0.12.0
=======================
Version 0.12.0 has breaking API changes including changes to the existing API and the removal of deprecated APIs.


Improved Client-Defined Memory Allocation
-----------------------------------------
The main change with this release is the addition of a more flexible way of implementing custom memory allocation routines. The
existing system of DRFLAC_MALLOC, DRFLAC_REALLOC and DRFLAC_FREE are still in place and will be used by default when no custom
allocation callbacks are specified.

To use the new system, you pass in a pointer to a drflac_allocation_callbacks object to drflac_open() and family, like this:

    void* my_malloc(size_t sz, void* pUserData)
    {
        return malloc(sz);
    }
    void* my_realloc(void* p, size_t sz, void* pUserData)
    {
        return realloc(p, sz);
    }
    void my_free(void* p, void* pUserData)
    {
        free(p);
    }

    ...

    drflac_allocation_callbacks allocationCallbacks;
    allocationCallbacks.pUserData = &myData;
    allocationCallbacks.onMalloc  = my_malloc;
    allocationCallbacks.onRealloc = my_realloc;
    allocationCallbacks.onFree    = my_free;
    drflac* pFlac = drflac_open_file("my_file.flac", &allocationCallbacks);

The advantage of this new system is that it allows you to specify user data which will be passed in to the allocation routines.

Passing in null for the allocation callbacks object will cause dr_flac to use defaults which is the same as DRFLAC_MALLOC,
DRFLAC_REALLOC and DRFLAC_FREE and the equivalent of how it worked in previous versions.

Every API that opens a drflac object now takes this extra parameter. These include the following:

    drflac_open()
    drflac_open_relaxed()
    drflac_open_with_metadata()
    drflac_open_with_metadata_relaxed()
    drflac_open_file()
    drflac_open_file_with_metadata()
    drflac_open_memory()
    drflac_open_memory_with_metadata()
    drflac_open_and_read_pcm_frames_s32()
    drflac_open_and_read_pcm_frames_s16()
    drflac_open_and_read_pcm_frames_f32()
    drflac_open_file_and_read_pcm_frames_s32()
    drflac_open_file_and_read_pcm_frames_s16()
    drflac_open_file_and_read_pcm_frames_f32()
    drflac_open_memory_and_read_pcm_frames_s32()
    drflac_open_memory_and_read_pcm_frames_s16()
    drflac_open_memory_and_read_pcm_frames_f32()



Optimizations
-------------
Seeking performance has been greatly improved. A new binary search based seeking algorithm has been introduced which significantly
improves performance over the brute force method which was used when no seek table was present. Seek table based seeking also takes
advantage of the new binary search seeking system to further improve performance there as well. Note that this depends on CRC which
means it will be disabled when DR_FLAC_NO_CRC is used.

The SSE4.1 pipeline has been cleaned up and optimized. You should see some improvements with decoding speed of 24-bit files in
particular. 16-bit streams should also see some improvement.

drflac_read_pcm_frames_s16() has been optimized. Previously this sat on top of drflac_read_pcm_frames_s32() and performed it's s32
to s16 conversion in a second pass. This is now all done in a single pass. This includes SSE2 and ARM NEON optimized paths.

A minor optimization has been implemented for drflac_read_pcm_frames_s32(). This will now use an SSE2 optimized pipeline for stereo
channel reconstruction which is the last part of the decoding process.

The ARM build has seen a few improvements. The CLZ (count leading zeroes) and REV (byte swap) instructions are now used when
compiling with GCC and Clang which is achieved using inline assembly. The CLZ instruction requires ARM architecture version 5 at
compile time and the REV instruction requires ARM architecture version 6.

An ARM NEON optimized pipeline has been implemented. To enable this you'll need to add -mfpu=neon to the command line when compiling.


Removed APIs
------------
The following APIs were deprecated in version 0.11.0 and have been completely removed in version 0.12.0:

    drflac_read_s32()                   -> drflac_read_pcm_frames_s32()
    drflac_read_s16()                   -> drflac_read_pcm_frames_s16()
    drflac_read_f32()                   -> drflac_read_pcm_frames_f32()
    drflac_seek_to_sample()             -> drflac_seek_to_pcm_frame()
    drflac_open_and_decode_s32()        -> drflac_open_and_read_pcm_frames_s32()
    drflac_open_and_decode_s16()        -> drflac_open_and_read_pcm_frames_s16()
    drflac_open_and_decode_f32()        -> drflac_open_and_read_pcm_frames_f32()
    drflac_open_and_decode_file_s32()   -> drflac_open_file_and_read_pcm_frames_s32()
    drflac_open_and_decode_file_s16()   -> drflac_open_file_and_read_pcm_frames_s16()
    drflac_open_and_decode_file_f32()   -> drflac_open_file_and_read_pcm_frames_f32()
    drflac_open_and_decode_memory_s32() -> drflac_open_memory_and_read_pcm_frames_s32()
    drflac_open_and_decode_memory_s16() -> drflac_open_memory_and_read_pcm_frames_s16()
    drflac_open_and_decode_memory_f32() -> drflac_open_memroy_and_read_pcm_frames_f32()

Prior versions of dr_flac operated on a per-sample basis whereas now it operates on PCM frames. The removed APIs all relate
to the old per-sample APIs. You now need to use the "pcm_frame" versions.
*/


/*
Introduction
============
dr_flac is a single file library. To use it, do something like the following in one .c file.

    ```c
    #define DR_FLAC_IMPLEMENTATION
    #include "dr_flac.h"
    ```

You can then #include this file in other parts of the program as you would with any other header file. To decode audio data, do something like the following:

    ```c
    drflac* pFlac = drflac_open_file("MySong.flac", NULL);
    if (pFlac == NULL) {
        // Failed to open FLAC file
    }

    drflac_int32* pSamples = malloc(pFlac->totalPCMFrameCount * pFlac->channels * sizeof(drflac_int32));
    drflac_uint64 numberOfInterleavedSamplesActuallyRead = drflac_read_pcm_frames_s32(pFlac, pFlac->totalPCMFrameCount, pSamples);
    ```

The drflac object represents the decoder. It is a transparent type so all the information you need, such as the number of channels and the bits per sample,
should be directly accessible - just make sure you don't change their values. Samples are always output as interleaved signed 32-bit PCM. In the example above
a native FLAC stream was opened, however dr_flac has seamless support for Ogg encapsulated FLAC streams as well.

You do not need to decode the entire stream in one go - you just specify how many samples you'd like at any given time and the decoder will give you as many
samples as it can, up to the amount requested. Later on when you need the next batch of samples, just call it again. Example:

    ```c
    while (drflac_read_pcm_frames_s32(pFlac, chunkSizeInPCMFrames, pChunkSamples) > 0) {
        do_something();
    }
    ```

You can seek to a specific PCM frame with `drflac_seek_to_pcm_frame()`.

If you just want to quickly decode an entire FLAC file in one go you can do something like this:

    ```c
    unsigned int channels;
    unsigned int sampleRate;
    drflac_uint64 totalPCMFrameCount;
    drflac_int32* pSampleData = drflac_open_file_and_read_pcm_frames_s32("MySong.flac", &channels, &sampleRate, &totalPCMFrameCount, NULL);
    if (pSampleData == NULL) {
        // Failed to open and decode FLAC file.
    }

    ...

    drflac_free(pSampleData, NULL);
    ```

You can read samples as signed 16-bit integer and 32-bit floating-point PCM with the *_s16() and *_f32() family of APIs respectively, but note that these
should be considered lossy.


If you need access to metadata (album art, etc.), use `drflac_open_with_metadata()`, `drflac_open_file_with_metdata()` or `drflac_open_memory_with_metadata()`.
The rationale for keeping these APIs separate is that they're slightly slower than the normal versions and also just a little bit harder to use. dr_flac
reports metadata to the application through the use of a callback, and every metadata block is reported before `drflac_open_with_metdata()` returns.

The main opening APIs (`drflac_open()`, etc.) will fail if the header is not present. The presents a problem in certain scenarios such as broadcast style
streams or internet radio where the header may not be present because the user has started playback mid-stream. To handle this, use the relaxed APIs:
    
    `drflac_open_relaxed()`
    `drflac_open_with_metadata_relaxed()`

It is not recommended to use these APIs for file based streams because a missing header would usually indicate a corrupt or perverse file. In addition, these
APIs can take a long time to initialize because they may need to spend a lot of time finding the first frame.



Build Options
=============
#define these options before including this file.

#define DR_FLAC_NO_STDIO
  Disable `drflac_open_file()` and family.

#define DR_FLAC_NO_OGG
  Disables support for Ogg/FLAC streams.

#define DR_FLAC_BUFFER_SIZE <number>
  Defines the size of the internal buffer to store data from onRead(). This buffer is used to reduce the number of calls back to the client for more data.
  Larger values means more memory, but better performance. My tests show diminishing returns after about 4KB (which is the default). Consider reducing this if
  you have a very efficient implementation of onRead(), or increase it if it's very inefficient. Must be a multiple of 8.

#define DR_FLAC_NO_CRC
  Disables CRC checks. This will offer a performance boost when CRC is unnecessary. This will disable binary search seeking. When seeking, the seek table will
  be used if available. Otherwise the seek will be performed using brute force.

#define DR_FLAC_NO_SIMD
  Disables SIMD optimizations (SSE on x86/x64 architectures, NEON on ARM architectures). Use this if you are having compatibility issues with your compiler.

#define DR_FLAC_NO_WCHAR
  Disables all functions ending with `_w`. Use this if your compiler does not provide wchar.h. Not required if DR_FLAC_NO_STDIO is also defined.



Notes
=====
- dr_flac does not support changing the sample rate nor channel count mid stream.
- dr_flac is not thread-safe, but its APIs can be called from any thread so long as you do your own synchronization.
- When using Ogg encapsulation, a corrupted metadata block will result in `drflac_open_with_metadata()` and `drflac_open()` returning inconsistent samples due
  to differences in corrupted stream recorvery logic between the two APIs.
*/

#ifndef dr_flac_h
#define dr_flac_h

#ifdef __cplusplus
extern "C" {
#endif

#define DRFLAC_STRINGIFY(x)      #x
#define DRFLAC_XSTRINGIFY(x)     DRFLAC_STRINGIFY(x)

#define DRFLAC_VERSION_MAJOR     0
#define DRFLAC_VERSION_MINOR     12
#define DRFLAC_VERSION_REVISION  41
#define DRFLAC_VERSION_STRING    DRFLAC_XSTRINGIFY(DRFLAC_VERSION_MAJOR) "." DRFLAC_XSTRINGIFY(DRFLAC_VERSION_MINOR) "." DRFLAC_XSTRINGIFY(DRFLAC_VERSION_REVISION)

#include <stddef.h> /* For size_t. */

/* Sized Types */
typedef   signed char           drflac_int8;
typedef unsigned char           drflac_uint8;
typedef   signed short          drflac_int16;
typedef unsigned short          drflac_uint16;
typedef   signed int            drflac_int32;
typedef unsigned int            drflac_uint32;
#if defined(_MSC_VER) && !defined(__clang__)
    typedef   signed __int64    drflac_int64;
    typedef unsigned __int64    drflac_uint64;
#else
    #if defined(__clang__) || (defined(__GNUC__) && (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 6)))
        #pragma GCC diagnostic push
        #pragma GCC diagnostic ignored "-Wlong-long"
        #if defined(__clang__)
            #pragma GCC diagnostic ignored "-Wc++11-long-long"
        #endif
    #endif
    typedef   signed long long  drflac_int64;
    typedef unsigned long long  drflac_uint64;
    #if defined(__clang__) || (defined(__GNUC__) && (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 6)))
        #pragma GCC diagnostic pop
    #endif
#endif
#if defined(__LP64__) || defined(_WIN64) || (defined(__x86_64__) && !defined(__ILP32__)) || defined(_M_X64) || defined(__ia64) || defined(_M_IA64) || defined(__aarch64__) || defined(_M_ARM64) || defined(__powerpc64__)
    typedef drflac_uint64       drflac_uintptr;
#else
    typedef drflac_uint32       drflac_uintptr;
#endif
typedef drflac_uint8            drflac_bool8;
typedef drflac_uint32           drflac_bool32;
#define DRFLAC_TRUE             1
#define DRFLAC_FALSE            0
/* End Sized Types */

/* Decorations */
#if !defined(DRFLAC_API)
    #if defined(DRFLAC_DLL)
        #if defined(_WIN32)
            #define DRFLAC_DLL_IMPORT  __declspec(dllimport)
            #define DRFLAC_DLL_EXPORT  __declspec(dllexport)
            #define DRFLAC_DLL_PRIVATE static
        #else
            #if defined(__GNUC__) && __GNUC__ >= 4
                #define DRFLAC_DLL_IMPORT  __attribute__((visibility("default")))
                #define DRFLAC_DLL_EXPORT  __attribute__((visibility("default")))
                #define DRFLAC_DLL_PRIVATE __attribute__((visibility("hidden")))
            #else
                #define DRFLAC_DLL_IMPORT
                #define DRFLAC_DLL_EXPORT
                #define DRFLAC_DLL_PRIVATE static
            #endif
        #endif

        #if defined(DR_FLAC_IMPLEMENTATION) || defined(DRFLAC_IMPLEMENTATION)
            #define DRFLAC_API  DRFLAC_DLL_EXPORT
        #else
            #define DRFLAC_API  DRFLAC_DLL_IMPORT
        #endif
        #define DRFLAC_PRIVATE DRFLAC_DLL_PRIVATE
    #else
        #define DRFLAC_API extern
        #define DRFLAC_PRIVATE static
    #endif
#endif
/* End Decorations */

#if defined(_MSC_VER) && _MSC_VER >= 1700   /* Visual Studio 2012 */
    #define DRFLAC_DEPRECATED       __declspec(deprecated)
#elif (defined(__GNUC__) && __GNUC__ >= 4)  /* GCC 4 */
    #define DRFLAC_DEPRECATED       __attribute__((deprecated))
#elif defined(__has_feature)                /* Clang */
    #if __has_feature(attribute_deprecated)
        #define DRFLAC_DEPRECATED   __attribute__((deprecated))
    #else
        #define DRFLAC_DEPRECATED
    #endif
#else
    #define DRFLAC_DEPRECATED
#endif

DRFLAC_API void drflac_version(drflac_uint32* pMajor, drflac_uint32* pMinor, drflac_uint32* pRevision);
DRFLAC_API const char* drflac_version_string(void);

/* Allocation Callbacks */
typedef struct
{
    void* pUserData;
    void* (* onMalloc)(size_t sz, void* pUserData);
    void* (* onRealloc)(void* p, size_t sz, void* pUserData);
    void  (* onFree)(void* p, void* pUserData);
} drflac_allocation_callbacks;
/* End Allocation Callbacks */

/*
As data is read from the client it is placed into an internal buffer for fast access. This controls the size of that buffer. Larger values means more speed,
but also more memory. In my testing there is diminishing returns after about 4KB, but you can fiddle with this to suit your own needs. Must be a multiple of 8.
*/
#ifndef DR_FLAC_BUFFER_SIZE
#define DR_FLAC_BUFFER_SIZE   4096
#endif


/* Architecture Detection */
#if defined(_WIN64) || defined(_LP64) || defined(__LP64__)
#define DRFLAC_64BIT
#endif

#if defined(__x86_64__) || defined(_M_X64)
    #define DRFLAC_X64
#elif defined(__i386) || defined(_M_IX86)
    #define DRFLAC_X86
#elif defined(__arm__) || defined(_M_ARM) || defined(__arm64) || defined(__arm64__) || defined(__aarch64__) || defined(_M_ARM64)
    #define DRFLAC_ARM
#endif
/* End Architecture Detection */


#ifdef DRFLAC_64BIT
typedef drflac_uint64 drflac_cache_t;
#else
typedef drflac_uint32 drflac_cache_t;
#endif

/* The various metadata block types. */
#define DRFLAC_METADATA_BLOCK_TYPE_STREAMINFO       0
#define DRFLAC_METADATA_BLOCK_TYPE_PADDING          1
#define DRFLAC_METADATA_BLOCK_TYPE_APPLICATION      2
#define DRFLAC_METADATA_BLOCK_TYPE_SEEKTABLE        3
#define DRFLAC_METADATA_BLOCK_TYPE_VORBIS_COMMENT   4
#define DRFLAC_METADATA_BLOCK_TYPE_CUESHEET         5
#define DRFLAC_METADATA_BLOCK_TYPE_PICTURE          6
#define DRFLAC_METADATA_BLOCK_TYPE_INVALID          127

/* The various picture types specified in the PICTURE block. */
#define DRFLAC_PICTURE_TYPE_OTHER                   0
#define DRFLAC_PICTURE_TYPE_FILE_ICON               1
#define DRFLAC_PICTURE_TYPE_OTHER_FILE_ICON         2
#define DRFLAC_PICTURE_TYPE_COVER_FRONT             3
#define DRFLAC_PICTURE_TYPE_COVER_BACK              4
#define DRFLAC_PICTURE_TYPE_LEAFLET_PAGE            5
#define DRFLAC_PICTURE_TYPE_MEDIA                   6
#define DRFLAC_PICTURE_TYPE_LEAD_ARTIST             7
#define DRFLAC_PICTURE_TYPE_ARTIST                  8
#define DRFLAC_PICTURE_TYPE_CONDUCTOR               9
#define DRFLAC_PICTURE_TYPE_BAND                    10
#define DRFLAC_PICTURE_TYPE_COMPOSER                11
#define DRFLAC_PICTURE_TYPE_LYRICIST                12
#define DRFLAC_PICTURE_TYPE_RECORDING_LOCATION      13
#define DRFLAC_PICTURE_TYPE_DURING_RECORDING        14
#define DRFLAC_PICTURE_TYPE_DURING_PERFORMANCE      15
#define DRFLAC_PICTURE_TYPE_SCREEN_CAPTURE          16
#define DRFLAC_PICTURE_TYPE_BRIGHT_COLORED_FISH     17
#define DRFLAC_PICTURE_TYPE_ILLUSTRATION            18
#define DRFLAC_PICTURE_TYPE_BAND_LOGOTYPE           19
#define DRFLAC_PICTURE_TYPE_PUBLISHER_LOGOTYPE      20

typedef enum
{
    drflac_container_native,
    drflac_container_ogg,
    drflac_container_unknown
} drflac_container;

typedef enum
{
    drflac_seek_origin_start,
    drflac_seek_origin_current
} drflac_seek_origin;

/* The order of members in this structure is important because we map this directly to the raw data within the SEEKTABLE metadata block. */
typedef struct
{
    drflac_uint64 firstPCMFrame;
    drflac_uint64 flacFrameOffset;   /* The offset from the first byte of the header of the first frame. */
    drflac_uint16 pcmFrameCount;
} drflac_seekpoint;

typedef struct
{
    drflac_uint16 minBlockSizeInPCMFrames;
    drflac_uint16 maxBlockSizeInPCMFrames;
    drflac_uint32 minFrameSizeInPCMFrames;
    drflac_uint32 maxFrameSizeInPCMFrames;
    drflac_uint32 sampleRate;
    drflac_uint8  channels;
    drflac_uint8  bitsPerSample;
    drflac_uint64 totalPCMFrameCount;
    drflac_uint8  md5[16];
} drflac_streaminfo;

typedef struct
{
    /*
    The metadata type. Use this to know how to interpret the data below. Will be set to one of the
    DRFLAC_METADATA_BLOCK_TYPE_* tokens.
    */
    drflac_uint32 type;

    /*
    A pointer to the raw data. This points to a temporary buffer so don't hold on to it. It's best to
    not modify the contents of this buffer. Use the structures below for more meaningful and structured
    information about the metadata. It's possible for this to be null.
    */
    const void* pRawData;

    /* The size in bytes of the block and the buffer pointed to by pRawData if it's non-NULL. */
    drflac_uint32 rawDataSize;

    union
    {
        drflac_streaminfo streaminfo;

        struct
        {
            int unused;
        } padding;

        struct
        {
            drflac_uint32 id;
            const void* pData;
            drflac_uint32 dataSize;
        } application;

        struct
        {
            drflac_uint32 seekpointCount;
            const drflac_seekpoint* pSeekpoints;
        } seektable;

        struct
        {
            drflac_uint32 vendorLength;
            const char* vendor;
            drflac_uint32 commentCount;
            const void* pComments;
        } vorbis_comment;

        struct
        {
            char catalog[128];
            drflac_uint64 leadInSampleCount;
            drflac_bool32 isCD;
            drflac_uint8 trackCount;
            const void* pTrackData;
        } cuesheet;

        struct
        {
            drflac_uint32 type;
            drflac_uint32 mimeLength;
            const char* mime;
            drflac_uint32 descriptionLength;
            const char* description;
            drflac_uint32 width;
            drflac_uint32 height;
            drflac_uint32 colorDepth;
            drflac_uint32 indexColorCount;
            drflac_uint32 pictureDataSize;
            const drflac_uint8* pPictureData;
        } picture;
    } data;
} drflac_metadata;


/*
Callback for when data needs to be read from the client.


Parameters
----------
pUserData (in)
    The user data that was passed to drflac_open() and family.

pBufferOut (out)
    The output buffer.

bytesToRead (in)
    The number of bytes to read.


Return Value
------------
The number of bytes actually read.


Remarks
-------
A return value of less than bytesToRead indicates the end of the stream. Do _not_ return from this callback until either the entire bytesToRead is filled or
you have reached the end of the stream.
*/
typedef size_t (* drflac_read_proc)(void* pUserData, void* pBufferOut, size_t bytesToRead);

/*
Callback for when data needs to be seeked.


Parameters
----------
pUserData (in)
    The user data that was passed to drflac_open() and family.

offset (in)
    The number of bytes to move, relative to the origin. Will never be negative.

origin (in)
    The origin of the seek - the current position or the start of the stream.


Return Value
------------
Whether or not the seek was successful.


Remarks
-------
The offset will never be negative. Whether or not it is relative to the beginning or current position is determined by the "origin" parameter which will be
either drflac_seek_origin_start or drflac_seek_origin_current.

When seeking to a PCM frame using drflac_seek_to_pcm_frame(), dr_flac may call this with an offset beyond the end of the FLAC stream. This needs to be detected
and handled by returning DRFLAC_FALSE.
*/
typedef drflac_bool32 (* drflac_seek_proc)(void* pUserData, int offset, drflac_seek_origin origin);

/*
Callback for when a metadata block is read.


Parameters
----------
pUserData (in)
    The user data that was passed to drflac_open() and family.

pMetadata (in)
    A pointer to a structure containing the data of the metadata block.


Remarks
-------
Use pMetadata->type to determine which metadata block is being handled and how to read the data. This
will be set to one of the DRFLAC_METADATA_BLOCK_TYPE_* tokens.
*/
typedef void (* drflac_meta_proc)(void* pUserData, drflac_metadata* pMetadata);


/* Structure for internal use. Only used for decoders opened with drflac_open_memory. */
typedef struct
{
    const drflac_uint8* data;
    size_t dataSize;
    size_t currentReadPos;
} drflac__memory_stream;

/* Structure for internal use. Used for bit streaming. */
typedef struct
{
    /* The function to call when more data needs to be read. */
    drflac_read_proc onRead;

    /* The function to call when the current read position needs to be moved. */
    drflac_seek_proc onSeek;

    /* The user data to pass around to onRead and onSeek. */
    void* pUserData;


    /*
    The number of unaligned bytes in the L2 cache. This will always be 0 until the end of the stream is hit. At the end of the
    stream there will be a number of bytes that don't cleanly fit in an L1 cache line, so we use this variable to know whether
    or not the bistreamer needs to run on a slower path to read those last bytes. This will never be more than sizeof(drflac_cache_t).
    */
    size_t unalignedByteCount;

    /* The content of the unaligned bytes. */
    drflac_cache_t unalignedCache;

    /* The index of the next valid cache line in the "L2" cache. */
    drflac_uint32 nextL2Line;

    /* The number of bits that have been consumed by the cache. This is used to determine how many valid bits are remaining. */
    drflac_uint32 consumedBits;

    /*
    The cached data which was most recently read from the client. There are two levels of cache. Data flows as such:
    Client -> L2 -> L1. The L2 -> L1 movement is aligned and runs on a fast path in just a few instructions.
    */
    drflac_cache_t cacheL2[DR_FLAC_BUFFER_SIZE/sizeof(drflac_cache_t)];
    drflac_cache_t cache;

    /*
    CRC-16. This is updated whenever bits are read from the bit stream. Manually set this to 0 to reset the CRC. For FLAC, this
    is reset to 0 at the beginning of each frame.
    */
    drflac_uint16 crc16;
    drflac_cache_t crc16Cache;              /* A cache for optimizing CRC calculations. This is filled when when the L1 cache is reloaded. */
    drflac_uint32 crc16CacheIgnoredBytes;   /* The number of bytes to ignore when updating the CRC-16 from the CRC-16 cache. */
} drflac_bs;

typedef struct
{
    /* The type of the subframe: SUBFRAME_CONSTANT, SUBFRAME_VERBATIM, SUBFRAME_FIXED or SUBFRAME_LPC. */
    drflac_uint8 subframeType;

    /* The number of wasted bits per sample as specified by the sub-frame header. */
    drflac_uint8 wastedBitsPerSample;

    /* The order to use for the prediction stage for SUBFRAME_FIXED and SUBFRAME_LPC. */
    drflac_uint8 lpcOrder;

    /* A pointer to the buffer containing the decoded samples in the subframe. This pointer is an offset from drflac::pExtraData. */
    drflac_int32* pSamplesS32;
} drflac_subframe;

typedef struct
{
    /*
    If the stream uses variable block sizes, this will be set to the index of the first PCM frame. If fixed block sizes are used, this will
    always be set to 0. This is 64-bit because the decoded PCM frame number will be 36 bits.
    */
    drflac_uint64 pcmFrameNumber;

    /*
    If the stream uses fixed block sizes, this will be set to the frame number. If variable block sizes are used, this will always be 0. This
    is 32-bit because in fixed block sizes, the maximum frame number will be 31 bits.
    */
    drflac_uint32 flacFrameNumber;

    /* The sample rate of this frame. */
    drflac_uint32 sampleRate;

    /* The number of PCM frames in each sub-frame within this frame. */
    drflac_uint16 blockSizeInPCMFrames;

    /*
    The channel assignment of this frame. This is not always set to the channel count. If interchannel decorrelation is being used this
    will be set to DRFLAC_CHANNEL_ASSIGNMENT_LEFT_SIDE, DRFLAC_CHANNEL_ASSIGNMENT_RIGHT_SIDE or DRFLAC_CHANNEL_ASSIGNMENT_MID_SIDE.
    */
    drflac_uint8 channelAssignment;

    /* The number of bits per sample within this frame. */
    drflac_uint8 bitsPerSample;

    /* The frame's CRC. */
    drflac_uint8 crc8;
} drflac_frame_header;

typedef struct
{
    /* The header. */
    drflac_frame_header header;

    /*
    The number of PCM frames left to be read in this FLAC frame. This is initially set to the block size. As PCM frames are read,
    this will be decremented. When it reaches 0, the decoder will see this frame as fully consumed and load the next frame.
    */
    drflac_uint32 pcmFramesRemaining;

    /* The list of sub-frames within the frame. There is one sub-frame for each channel, and there's a maximum of 8 channels. */
    drflac_subframe subframes[8];
} drflac_frame;

typedef struct
{
    /* The function to call when a metadata block is read. */
    drflac_meta_proc onMeta;

    /* The user data posted to the metadata callback function. */
    void* pUserDataMD;

    /* Memory allocation callbacks. */
    drflac_allocation_callbacks allocationCallbacks;


    /* The sample rate. Will be set to something like 44100. */
    drflac_uint32 sampleRate;

    /*
    The number of channels. This will be set to 1 for monaural streams, 2 for stereo, etc. Maximum 8. This is set based on the
    value specified in the STREAMINFO block.
    */
    drflac_uint8 channels;

    /* The bits per sample. Will be set to something like 16, 24, etc. */
    drflac_uint8 bitsPerSample;

    /* The maximum block size, in samples. This number represents the number of samples in each channel (not combined). */
    drflac_uint16 maxBlockSizeInPCMFrames;

    /*
    The total number of PCM Frames making up the stream. Can be 0 in which case it's still a valid stream, but just means
    the total PCM frame count is unknown. Likely the case with streams like internet radio.
    */
    drflac_uint64 totalPCMFrameCount;


    /* The container type. This is set based on whether or not the decoder was opened from a native or Ogg stream. */
    drflac_container container;

    /* The number of seekpoints in the seektable. */
    drflac_uint32 seekpointCount;


    /* Information about the frame the decoder is currently sitting on. */
    drflac_frame currentFLACFrame;


    /* The index of the PCM frame the decoder is currently sitting on. This is only used for seeking. */
    drflac_uint64 currentPCMFrame;

    /* The position of the first FLAC frame in the stream. This is only ever used for seeking. */
    drflac_uint64 firstFLACFramePosInBytes;


    /* A hack to avoid a malloc() when opening a decoder with drflac_open_memory(). */
    drflac__memory_stream memoryStream;


    /* A pointer to the decoded sample data. This is an offset of pExtraData. */
    drflac_int32* pDecodedSamples;

    /* A pointer to the seek table. This is an offset of pExtraData, or NULL if there is no seek table. */
    drflac_seekpoint* pSeekpoints;

    /* Internal use only. Only used with Ogg containers. Points to a drflac_oggbs object. This is an offset of pExtraData. */
    void* _oggbs;

    /* Internal use only. Used for profiling and testing different seeking modes. */
    drflac_bool32 _noSeekTableSeek    : 1;
    drflac_bool32 _noBinarySearchSeek : 1;
    drflac_bool32 _noBruteForceSeek   : 1;

    /* The bit streamer. The raw FLAC data is fed through this object. */
    drflac_bs bs;

    /* Variable length extra data. We attach this to the end of the object so we can avoid unnecessary mallocs. */
    drflac_uint8 pExtraData[1];
} drflac;


/*
Opens a FLAC decoder.


Parameters
----------
onRead (in)
    The function to call when data needs to be read from the client.

onSeek (in)
    The function to call when the read position of the client data needs to move.

pUserData (in, optional)
    A pointer to application defined data that will be passed to onRead and onSeek.

pAllocationCallbacks (in, optional)
    A pointer to application defined callbacks for managing memory allocations.


Return Value
------------
Returns a pointer to an object representing the decoder.


Remarks
-------
Close the decoder with `drflac_close()`.

`pAllocationCallbacks` can be NULL in which case it will use `DRFLAC_MALLOC`, `DRFLAC_REALLOC` and `DRFLAC_FREE`.

This function will automatically detect whether or not you are attempting to open a native or Ogg encapsulated FLAC, both of which should work seamlessly
without any manual intervention. Ogg encapsulation also works with multiplexed streams which basically means it can play FLAC encoded audio tracks in videos.

This is the lowest level function for opening a FLAC stream. You can also use `drflac_open_file()` and `drflac_open_memory()` to open the stream from a file or
from a block of memory respectively.

The STREAMINFO block must be present for this to succeed. Use `drflac_open_relaxed()` to open a FLAC stream where the header may not be present.

Use `drflac_open_with_metadata()` if you need access to metadata.


Seek Also
---------
drflac_open_file()
drflac_open_memory()
drflac_open_with_metadata()
drflac_close()
*/
DRFLAC_API drflac* drflac_open(drflac_read_proc onRead, drflac_seek_proc onSeek, void* pUserData, const drflac_allocation_callbacks* pAllocationCallbacks);

/*
Opens a FLAC stream with relaxed validation of the header block.


Parameters
----------
onRead (in)
    The function to call when data needs to be read from the client.

onSeek (in)
    The function to call when the read position of the client data needs to move.

container (in)
    Whether or not the FLAC stream is encapsulated using standard FLAC encapsulation or Ogg encapsulation.

pUserData (in, optional)
    A pointer to application defined data that will be passed to onRead and onSeek.

pAllocationCallbacks (in, optional)
    A pointer to application defined callbacks for managing memory allocations.


Return Value
------------
A pointer to an object representing the decoder.


Remarks
-------
The same as drflac_open(), except attempts to open the stream even when a header block is not present.

Because the header is not necessarily available, the caller must explicitly define the container (Native or Ogg). Do not set this to `drflac_container_unknown`
as that is for internal use only.

Opening in relaxed mode will continue reading data from onRead until it finds a valid frame. If a frame is never found it will continue forever. To abort,
force your `onRead` callback to return 0, which dr_flac will use as an indicator that the end of the stream was found.

Use `drflac_open_with_metadata_relaxed()` if you need access to metadata.
*/
DRFLAC_API drflac* drflac_open_relaxed(drflac_read_proc onRead, drflac_seek_proc onSeek, drflac_container container, void* pUserData, const drflac_allocation_callbacks* pAllocationCallbacks);

/*
Opens a FLAC decoder and notifies the caller of the metadata chunks (album art, etc.).


Parameters
----------
onRead (in)
    The function to call when data needs to be read from the client.

onSeek (in)
    The function to call when the read position of the client data needs to move.

onMeta (in)
    The function to call for every metadata block.

pUserData (in, optional)
    A pointer to application defined data that will be passed to onRead, onSeek and onMeta.

pAllocationCallbacks (in, optional)
    A pointer to application defined callbacks for managing memory allocations.


Return Value
------------
A pointer to an object representing the decoder.


Remarks
-------
Close the decoder with `drflac_close()`.

`pAllocationCallbacks` can be NULL in which case it will use `DRFLAC_MALLOC`, `DRFLAC_REALLOC` and `DRFLAC_FREE`.

This is slower than `drflac_open()`, so avoid this one if you don't need metadata. Internally, this will allocate and free memory on the heap for every
metadata block except for STREAMINFO and PADDING blocks.

The caller is notified of the metadata via the `onMeta` callback. All metadata blocks will be handled before the function returns. This callback takes a
pointer to a `drflac_metadata` object which is a union containing the data of all relevant metadata blocks. Use the `type` member to discriminate against
the different metadata types.

The STREAMINFO block must be present for this to succeed. Use `drflac_open_with_metadata_relaxed()` to open a FLAC stream where the header may not be present.

Note that this will behave inconsistently with `drflac_open()` if the stream is an Ogg encapsulated stream and a metadata block is corrupted. This is due to
the way the Ogg stream recovers from corrupted pages. When `drflac_open_with_metadata()` is being used, the open routine will try to read the contents of the
metadata block, whereas `drflac_open()` will simply seek past it (for the sake of efficiency). This inconsistency can result in different samples being
returned depending on whether or not the stream is being opened with metadata.


Seek Also
---------
drflac_open_file_with_metadata()
drflac_open_memory_with_metadata()
drflac_open()
drflac_close()
*/
DRFLAC_API drflac* drflac_open_with_metadata(drflac_read_proc onRead, drflac_seek_proc onSeek, drflac_meta_proc onMeta, void* pUserData, const drflac_allocation_callbacks* pAllocationCallbacks);

/*
The same as drflac_open_with_metadata(), except attempts to open the stream even when a header block is not present.

See Also
--------
drflac_open_with_metadata()
drflac_open_relaxed()
*/
DRFLAC_API drflac* drflac_open_with_metadata_relaxed(drflac_read_proc onRead, drflac_seek_proc onSeek, drflac_meta_proc onMeta, drflac_container container, void* pUserData, const drflac_allocation_callbacks* pAllocationCallbacks);

/*
Closes the given FLAC decoder.


Parameters
----------
pFlac (in)
    The decoder to close.


Remarks
-------
This will destroy the decoder object.


See Also
--------
drflac_open()
drflac_open_with_metadata()
drflac_open_file()
drflac_open_file_w()
drflac_open_file_with_metadata()
drflac_open_file_with_metadata_w()
drflac_open_memory()
drflac_open_memory_with_metadata()
*/
DRFLAC_API void drflac_close(drflac* pFlac);


/*
Reads sample data from the given FLAC decoder, output as interleaved signed 32-bit PCM.


Parameters
----------
pFlac (in)
    The decoder.

framesToRead (in)
    The number of PCM frames to read.

pBufferOut (out, optional)
    A pointer to the buffer that will receive the decoded samples.


Return Value
------------
Returns the number of PCM frames actually read. If the return value is less than `framesToRead` it has reached the end.


Remarks
-------
pBufferOut can be null, in which case the call will act as a seek, and the return value will be the number of frames seeked.
*/
DRFLAC_API drflac_uint64 drflac_read_pcm_frames_s32(drflac* pFlac, drflac_uint64 framesToRead, drflac_int32* pBufferOut);


/*
Reads sample data from the given FLAC decoder, output as interleaved signed 16-bit PCM.


Parameters
----------
pFlac (in)
    The decoder.

framesToRead (in)
    The number of PCM frames to read.

pBufferOut (out, optional)
    A pointer to the buffer that will receive the decoded samples.


Return Value
------------
Returns the number of PCM frames actually read. If the return value is less than `framesToRead` it has reached the end.


Remarks
-------
pBufferOut can be null, in which case the call will act as a seek, and the return value will be the number of frames seeked.

Note that this is lossy for streams where the bits per sample is larger than 16.
*/
DRFLAC_API drflac_uint64 drflac_read_pcm_frames_s16(drflac* pFlac, drflac_uint64 framesToRead, drflac_int16* pBufferOut);

/*
Reads sample data from the given FLAC decoder, output as interleaved 32-bit floating point PCM.


Parameters
----------
pFlac (in)
    The decoder.

framesToRead (in)
    The number of PCM frames to read.

pBufferOut (out, optional)
    A pointer to the buffer that will receive the decoded samples.


Return Value
------------
Returns the number of PCM frames actually read. If the return value is less than `framesToRead` it has reached the end.


Remarks
-------
pBufferOut can be null, in which case the call will act as a seek, and the return value will be the number of frames seeked.

Note that this should be considered lossy due to the nature of floating point numbers not being able to exactly represent every possible number.
*/
DRFLAC_API drflac_uint64 drflac_read_pcm_frames_f32(drflac* pFlac, drflac_uint64 framesToRead, float* pBufferOut);

/*
Seeks to the PCM frame at the given index.


Parameters
----------
pFlac (in)
    The decoder.

pcmFrameIndex (in)
    The index of the PCM frame to seek to. See notes below.


Return Value
-------------
`DRFLAC_TRUE` if successful; `DRFLAC_FALSE` otherwise.
*/
DRFLAC_API drflac_bool32 drflac_seek_to_pcm_frame(drflac* pFlac, drflac_uint64 pcmFrameIndex);



#ifndef DR_FLAC_NO_STDIO
/*
Opens a FLAC decoder from the file at the given path.


Parameters
----------
pFileName (in)
    The path of the file to open, either absolute or relative to the current directory.

pAllocationCallbacks (in, optional)
    A pointer to application defined callbacks for managing memory allocations.


Return Value
------------
A pointer to an object representing the decoder.


Remarks
-------
Close the decoder with drflac_close().


Remarks
-------
This will hold a handle to the file until the decoder is closed with drflac_close(). Some platforms will restrict the number of files a process can have open
at any given time, so keep this mind if you have many decoders open at the same time.


See Also
--------
drflac_open_file_with_metadata()
drflac_open()
drflac_close()
*/
DRFLAC_API drflac* drflac_open_file(const char* pFileName, const drflac_allocation_callbacks* pAllocationCallbacks);
DRFLAC_API drflac* drflac_open_file_w(const wchar_t* pFileName, const drflac_allocation_callbacks* pAllocationCallbacks);

/*
Opens a FLAC decoder from the file at the given path and notifies the caller of the metadata chunks (album art, etc.)


Parameters
----------
pFileName (in)
    The path of the file to open, either absolute or relative to the current directory.

pAllocationCallbacks (in, optional)
    A pointer to application defined callbacks for managing memory allocations.

onMeta (in)
    The callback to fire for each metadata block.

pUserData (in)
    A pointer to the user data to pass to the metadata callback.

pAllocationCallbacks (in)
    A pointer to application defined callbacks for managing memory allocations.


Remarks
-------
Look at the documentation for drflac_open_with_metadata() for more information on how metadata is handled.


See Also
--------
drflac_open_with_metadata()
drflac_open()
drflac_close()
*/
DRFLAC_API drflac* drflac_open_file_with_metadata(const char* pFileName, drflac_meta_proc onMeta, void* pUserData, const drflac_allocation_callbacks* pAllocationCallbacks);
DRFLAC_API drflac* drflac_open_file_with_metadata_w(const wchar_t* pFileName, drflac_meta_proc onMeta, void* pUserData, const drflac_allocation_callbacks* pAllocationCallbacks);
#endif

/*
Opens a FLAC decoder from a pre-allocated block of memory


Parameters
----------
pData (in)
    A pointer to the raw encoded FLAC data.

dataSize (in)
    The size in bytes of `data`.

pAllocationCallbacks (in)
    A pointer to application defined callbacks for managing memory allocations.


Return Value
------------
A pointer to an object representing the decoder.


Remarks
-------
This does not create a copy of the data. It is up to the application to ensure the buffer remains valid for the lifetime of the decoder.


See Also
--------
drflac_open()
drflac_close()
*/
DRFLAC_API drflac* drflac_open_memory(const void* pData, size_t dataSize, const drflac_allocation_callbacks* pAllocationCallbacks);

/*
Opens a FLAC decoder from a pre-allocated block of memory and notifies the caller of the metadata chunks (album art, etc.)


Parameters
----------
pData (in)
    A pointer to the raw encoded FLAC data.

dataSize (in)
    The size in bytes of `data`.

onMeta (in)
    The callback to fire for each metadata block.

pUserData (in)
    A pointer to the user data to pass to the metadata callback.

pAllocationCallbacks (in)
    A pointer to application defined callbacks for managing memory allocations.


Remarks
-------
Look at the documentation for drflac_open_with_metadata() for more information on how metadata is handled.


See Also
-------
drflac_open_with_metadata()
drflac_open()
drflac_close()
*/
DRFLAC_API drflac* drflac_open_memory_with_metadata(const void* pData, size_t dataSize, drflac_meta_proc onMeta, void* pUserData, const drflac_allocation_callbacks* pAllocationCallbacks);



/* High Level APIs */

/*
Opens a FLAC stream from the given callbacks and fully decodes it in a single operation. The return value is a
pointer to the sample data as interleaved signed 32-bit PCM. The returned data must be freed with drflac_free().

You can pass in custom memory allocation callbacks via the pAllocationCallbacks parameter. This can be NULL in which
case it will use DRFLAC_MALLOC, DRFLAC_REALLOC and DRFLAC_FREE.

Sometimes a FLAC file won't keep track of the total sample count. In this situation the function will continuously
read samples into a dynamically sized buffer on the heap until no samples are left.

Do not call this function on a broadcast type of stream (like internet radio streams and whatnot).
*/
DRFLAC_API drflac_int32* drflac_open_and_read_pcm_frames_s32(drflac_read_proc onRead, drflac_seek_proc onSeek, void* pUserData, unsigned int* channels, unsigned int* sampleRate, drflac_uint64* totalPCMFrameCount, const drflac_allocation_callbacks* pAllocationCallbacks);

/* Same as drflac_open_and_read_pcm_frames_s32(), except returns signed 16-bit integer samples. */
DRFLAC_API drflac_int16* drflac_open_and_read_pcm_frames_s16(drflac_read_proc onRead, drflac_seek_proc onSeek, void* pUserData, unsigned int* channels, unsigned int* sampleRate, drflac_uint64* totalPCMFrameCount, const drflac_allocation_callbacks* pAllocationCallbacks);

/* Same as drflac_open_and_read_pcm_frames_s32(), except returns 32-bit floating-point samples. */
DRFLAC_API float* drflac_open_and_read_pcm_frames_f32(drflac_read_proc onRead, drflac_seek_proc onSeek, void* pUserData, unsigned int* channels, unsigned int* sampleRate, drflac_uint64* totalPCMFrameCount, const drflac_allocation_callbacks* pAllocationCallbacks);

#ifndef DR_FLAC_NO_STDIO
/* Same as drflac_open_and_read_pcm_frames_s32() except opens the decoder from a file. */
DRFLAC_API drflac_int32* drflac_open_file_and_read_pcm_frames_s32(const char* filename, unsigned int* channels, unsigned int* sampleRate, drflac_uint64* totalPCMFrameCount, const drflac_allocation_callbacks* pAllocationCallbacks);

/* Same as drflac_open_file_and_read_pcm_frames_s32(), except returns signed 16-bit integer samples. */
DRFLAC_API drflac_int16* drflac_open_file_and_read_pcm_frames_s16(const char* filename, unsigned int* channels, unsigned int* sampleRate, drflac_uint64* totalPCMFrameCount, const drflac_allocation_callbacks* pAllocationCallbacks);

/* Same as drflac_open_file_and_read_pcm_frames_s32(), except returns 32-bit floating-point samples. */
DRFLAC_API float* drflac_open_file_and_read_pcm_frames_f32(const char* filename, unsigned int* channels, unsigned int* sampleRate, drflac_uint64* totalPCMFrameCount, const drflac_allocation_callbacks* pAllocationCallbacks);
#endif

/* Same as drflac_open_and_read_pcm_frames_s32() except opens the decoder from a block of memory. */
DRFLAC_API drflac_int32* drflac_open_memory_and_read_pcm_frames_s32(const void* data, size_t dataSize, unsigned int* channels, unsigned int* sampleRate, drflac_uint64* totalPCMFrameCount, const drflac_allocation_callbacks* pAllocationCallbacks);

/* Same as drflac_open_memory_and_read_pcm_frames_s32(), except returns signed 16-bit integer samples. */
DRFLAC_API drflac_int16* drflac_open_memory_and_read_pcm_frames_s16(const void* data, size_t dataSize, unsigned int* channels, unsigned int* sampleRate, drflac_uint64* totalPCMFrameCount, const drflac_allocation_callbacks* pAllocationCallbacks);

/* Same as drflac_open_memory_and_read_pcm_frames_s32(), except returns 32-bit floating-point samples. */
DRFLAC_API float* drflac_open_memory_and_read_pcm_frames_f32(const void* data, size_t dataSize, unsigned int* channels, unsigned int* sampleRate, drflac_uint64* totalPCMFrameCount, const drflac_allocation_callbacks* pAllocationCallbacks);

/*
Frees memory that was allocated internally by dr_flac.

Set pAllocationCallbacks to the same object that was passed to drflac_open_*_and_read_pcm_frames_*(). If you originally passed in NULL, pass in NULL for this.
*/
DRFLAC_API void drflac_free(void* p, const drflac_allocation_callbacks* pAllocationCallbacks);


/* Structure representing an iterator for vorbis comments in a VORBIS_COMMENT metadata block. */
typedef struct
{
    drflac_uint32 countRemaining;
    const char* pRunningData;
} drflac_vorbis_comment_iterator;

/*
Initializes a vorbis comment iterator. This can be used for iterating over the vorbis comments in a VORBIS_COMMENT
metadata block.
*/
DRFLAC_API void drflac_init_vorbis_comment_iterator(drflac_vorbis_comment_iterator* pIter, drflac_uint32 commentCount, const void* pComments);

/*
Goes to the next vorbis comment in the given iterator. If null is returned it means there are no more comments. The
returned string is NOT null terminated.
*/
DRFLAC_API const char* drflac_next_vorbis_comment(drflac_vorbis_comment_iterator* pIter, drflac_uint32* pCommentLengthOut);


/* Structure representing an iterator for cuesheet tracks in a CUESHEET metadata block. */
typedef struct
{
    drflac_uint32 countRemaining;
    const char* pRunningData;
} drflac_cuesheet_track_iterator;

/* The order of members here is important because we map this directly to the raw data within the CUESHEET metadata block. */
typedef struct
{
    drflac_uint64 offset;
    drflac_uint8 index;
    drflac_uint8 reserved[3];
} drflac_cuesheet_track_index;

typedef struct
{
    drflac_uint64 offset;
    drflac_uint8 trackNumber;
    char ISRC[12];
    drflac_bool8 isAudio;
    drflac_bool8 preEmphasis;
    drflac_uint8 indexCount;
    const drflac_cuesheet_track_index* pIndexPoints;
} drflac_cuesheet_track;

/*
Initializes a cuesheet track iterator. This can be used for iterating over the cuesheet tracks in a CUESHEET metadata
block.
*/
DRFLAC_API void drflac_init_cuesheet_track_iterator(drflac_cuesheet_track_iterator* pIter, drflac_uint32 trackCount, const void* pTrackData);

/* Goes to the next cuesheet track in the given iterator. If DRFLAC_FALSE is returned it means there are no more comments. */
DRFLAC_API drflac_bool32 drflac_next_cuesheet_track(drflac_cuesheet_track_iterator* pIter, drflac_cuesheet_track* pCuesheetTrack);


#ifdef __cplusplus
}
#endif
#endif  /* dr_flac_h */


/************************************************************************************************************************************************************
 ************************************************************************************************************************************************************

 IMPLEMENTATION

 ************************************************************************************************************************************************************
 ************************************************************************************************************************************************************/
#if defined(DR_FLAC_IMPLEMENTATION) || defined(DRFLAC_IMPLEMENTATION)
#ifndef dr_flac_c
#define dr_flac_c

/* Disable some annoying warnings. */
#if defined(__clang__) || (defined(__GNUC__) && (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 6)))
    #pragma GCC diagnostic push
    #if __GNUC__ >= 7
    #pragma GCC diagnostic ignored "-Wimplicit-fallthrough"
    #endif
#endif

#ifdef __linux__
    #ifndef _BSD_SOURCE
        #define _BSD_SOURCE
    #endif
    #ifndef _DEFAULT_SOURCE
        #define _DEFAULT_SOURCE
    #endif
    #ifndef __USE_BSD
        #define __USE_BSD
    #endif
    #include <endian.h>
#endif

#include <stdlib.h>
#include <string.h>

/* Inline */
#ifdef _MSC_VER
    #define DRFLAC_INLINE __forceinline
#elif defined(__GNUC__)
    /*
    I've had a bug report where GCC is emitting warnings about functions possibly not being inlineable. This warning happens when
    the __attribute__((always_inline)) attribute is defined without an "inline" statement. I think therefore there must be some
    case where "__inline__" is not always defined, thus the compiler emitting these warnings. When using -std=c89 or -ansi on the
    command line, we cannot use the "inline" keyword and instead need to use "__inline__". In an attempt to work around this issue
    I am using "__inline__" only when we're compiling in strict ANSI mode.
    */
    #if defined(__STRICT_ANSI__)
        #define DRFLAC_GNUC_INLINE_HINT __inline__
    #else
        #define DRFLAC_GNUC_INLINE_HINT inline
    #endif

    #if (__GNUC__ > 3 || (__GNUC__ == 3 && __GNUC_MINOR__ >= 2)) || defined(__clang__)
        #define DRFLAC_INLINE DRFLAC_GNUC_INLINE_HINT __attribute__((always_inline))
    #else
        #define DRFLAC_INLINE DRFLAC_GNUC_INLINE_HINT
    #endif
#elif defined(__WATCOMC__)
    #define DRFLAC_INLINE __inline
#else
    #define DRFLAC_INLINE
#endif
/* End Inline */

/*
Intrinsics Support

There's a bug in GCC 4.2.x which results in an incorrect compilation error when using _mm_slli_epi32() where it complains with

    "error: shift must be an immediate"

Unfortuantely dr_flac depends on this for a few things so we're just going to disable SSE on GCC 4.2 and below.
*/
#if !defined(DR_FLAC_NO_SIMD)
    #if defined(DRFLAC_X64) || defined(DRFLAC_X86)
        #if defined(_MSC_VER) && !defined(__clang__)
            /* MSVC. */
            #if _MSC_VER >= 1400 && !defined(DRFLAC_NO_SSE2)    /* 2005 */
                #define DRFLAC_SUPPORT_SSE2
            #endif
            #if _MSC_VER >= 1600 && !defined(DRFLAC_NO_SSE41)   /* 2010 */
                #define DRFLAC_SUPPORT_SSE41
            #endif
        #elif defined(__clang__) || (defined(__GNUC__) && (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 3)))
            /* Assume GNUC-style. */
            #if defined(__SSE2__) && !defined(DRFLAC_NO_SSE2)
                #define DRFLAC_SUPPORT_SSE2
            #endif
            #if defined(__SSE4_1__) && !defined(DRFLAC_NO_SSE41)
                #define DRFLAC_SUPPORT_SSE41
            #endif
        #endif

        /* If at this point we still haven't determined compiler support for the intrinsics just fall back to __has_include. */
        #if !defined(__GNUC__) && !defined(__clang__) && defined(__has_include)
            #if !defined(DRFLAC_SUPPORT_SSE2) && !defined(DRFLAC_NO_SSE2) && __has_include(<emmintrin.h>)
                #define DRFLAC_SUPPORT_SSE2
            #endif
            #if !defined(DRFLAC_SUPPORT_SSE41) && !defined(DRFLAC_NO_SSE41) && __has_include(<smmintrin.h>)
                #define DRFLAC_SUPPORT_SSE41
            #endif
        #endif

        #if defined(DRFLAC_SUPPORT_SSE41)
            #include <smmintrin.h>
        #elif defined(DRFLAC_SUPPORT_SSE2)
            #include <emmintrin.h>
        #endif
    #endif

    #if defined(DRFLAC_ARM)
        #if !defined(DRFLAC_NO_NEON) && (defined(__ARM_NEON) || defined(__aarch64__) || defined(_M_ARM64))
            #define DRFLAC_SUPPORT_NEON
            #include <arm_neon.h>
        #endif
    #endif
#endif

/* Compile-time CPU feature support. */
#if !defined(DR_FLAC_NO_SIMD) && (defined(DRFLAC_X86) || defined(DRFLAC_X64))
    #if defined(_MSC_VER) && !defined(__clang__)
        #if _MSC_VER >= 1400
            #include <intrin.h>
            static void drflac__cpuid(int info[4], int fid)
            {
                __cpuid(info, fid);
            }
        #else
            #define DRFLAC_NO_CPUID
        #endif
    #else
        #if defined(__GNUC__) || defined(__clang__)
            static void drflac__cpuid(int info[4], int fid)
            {
                /*
                It looks like the -fPIC option uses the ebx register which GCC complains about. We can work around this by just using a different register, the
                specific register of which I'm letting the compiler decide on. The "k" prefix is used to specify a 32-bit register. The {...} syntax is for
                supporting different assembly dialects.

                What's basically happening is that we're saving and restoring the ebx register manually.
                */
                #if defined(DRFLAC_X86) && defined(__PIC__)
                    __asm__ __volatile__ (
                        "xchg{l} {%%}ebx, %k1;"
                        "cpuid;"
                        "xchg{l} {%%}ebx, %k1;"
                        : "=a"(info[0]), "=&r"(info[1]), "=c"(info[2]), "=d"(info[3]) : "a"(fid), "c"(0)
                    );
                #else
                    __asm__ __volatile__ (
                        "cpuid" : "=a"(info[0]), "=b"(info[1]), "=c"(info[2]), "=d"(info[3]) : "a"(fid), "c"(0)
                    );
                #endif
            }
        #else
            #define DRFLAC_NO_CPUID
        #endif
    #endif
#else
    #define DRFLAC_NO_CPUID
#endif

static DRFLAC_INLINE drflac_bool32 drflac_has_sse2(void)
{
#if defined(DRFLAC_SUPPORT_SSE2)
    #if (defined(DRFLAC_X64) || defined(DRFLAC_X86)) && !defined(DRFLAC_NO_SSE2)
        #if defined(DRFLAC_X64)
            return DRFLAC_TRUE;    /* 64-bit targets always support SSE2. */
        #elif (defined(_M_IX86_FP) && _M_IX86_FP == 2) || defined(__SSE2__)
            return DRFLAC_TRUE;    /* If the compiler is allowed to freely generate SSE2 code we can assume support. */
        #else
            #if defined(DRFLAC_NO_CPUID)
                return DRFLAC_FALSE;
            #else
                int info[4];
                drflac__cpuid(info, 1);
                return (info[3] & (1 << 26)) != 0;
            #endif
        #endif
    #else
        return DRFLAC_FALSE;       /* SSE2 is only supported on x86 and x64 architectures. */
    #endif
#else
    return DRFLAC_FALSE;           /* No compiler support. */
#endif
}

static DRFLAC_INLINE drflac_bool32 drflac_has_sse41(void)
{
#if defined(DRFLAC_SUPPORT_SSE41)
    #if (defined(DRFLAC_X64) || defined(DRFLAC_X86)) && !defined(DRFLAC_NO_SSE41)
        #if defined(__SSE4_1__) || defined(__AVX__)
            return DRFLAC_TRUE;    /* If the compiler is allowed to freely generate SSE41 code we can assume support. */
        #else
            #if defined(DRFLAC_NO_CPUID)
                return DRFLAC_FALSE;
            #else
                int info[4];
                drflac__cpuid(info, 1);
                return (info[2] & (1 << 19)) != 0;
            #endif
        #endif
    #else
        return DRFLAC_FALSE;       /* SSE41 is only supported on x86 and x64 architectures. */
    #endif
#else
    return DRFLAC_FALSE;           /* No compiler support. */
#endif
}


#if defined(_MSC_VER) && _MSC_VER >= 1500 && (defined(DRFLAC_X86) || defined(DRFLAC_X64)) && !defined(__clang__)
    #define DRFLAC_HAS_LZCNT_INTRINSIC
#elif (defined(__GNUC__) && ((__GNUC__ > 4) || (__GNUC__ == 4 && __GNUC_MINOR__ >= 7)))
    #define DRFLAC_HAS_LZCNT_INTRINSIC
#elif defined(__clang__)
    #if defined(__has_builtin)
        #if __has_builtin(__builtin_clzll) || __has_builtin(__builtin_clzl)
            #define DRFLAC_HAS_LZCNT_INTRINSIC
        #endif
    #endif
#endif

#if defined(_MSC_VER) && _MSC_VER >= 1400 && !defined(__clang__)
    #define DRFLAC_HAS_BYTESWAP16_INTRINSIC
    #define DRFLAC_HAS_BYTESWAP32_INTRINSIC
    #define DRFLAC_HAS_BYTESWAP64_INTRINSIC
#elif defined(__clang__)
    #if defined(__has_builtin)
        #if __has_builtin(__builtin_bswap16)
            #define DRFLAC_HAS_BYTESWAP16_INTRINSIC
        #endif
        #if __has_builtin(__builtin_bswap32)
            #define DRFLAC_HAS_BYTESWAP32_INTRINSIC
        #endif
        #if __has_builtin(__builtin_bswap64)
            #define DRFLAC_HAS_BYTESWAP64_INTRINSIC
        #endif
    #endif
#elif defined(__GNUC__)
    #if ((__GNUC__ > 4) || (__GNUC__ == 4 && __GNUC_MINOR__ >= 3))
        #define DRFLAC_HAS_BYTESWAP32_INTRINSIC
        #define DRFLAC_HAS_BYTESWAP64_INTRINSIC
    #endif
    #if ((__GNUC__ > 4) || (__GNUC__ == 4 && __GNUC_MINOR__ >= 8))
        #define DRFLAC_HAS_BYTESWAP16_INTRINSIC
    #endif
#elif defined(__WATCOMC__) && defined(__386__)
    #define DRFLAC_HAS_BYTESWAP16_INTRINSIC
    #define DRFLAC_HAS_BYTESWAP32_INTRINSIC
    #define DRFLAC_HAS_BYTESWAP64_INTRINSIC
    extern __inline drflac_uint16 _watcom_bswap16(drflac_uint16);
    extern __inline drflac_uint32 _watcom_bswap32(drflac_uint32);
    extern __inline drflac_uint64 _watcom_bswap64(drflac_uint64);
#pragma aux _watcom_bswap16 = \
    "xchg al, ah" \
    parm  [ax]    \
    value [ax]    \
    modify nomemory;
#pragma aux _watcom_bswap32 = \
    "bswap eax" \
    parm  [eax] \
    value [eax] \
    modify nomemory;
#pragma aux _watcom_bswap64 = \
    "bswap eax"     \
    "bswap edx"     \
    "xchg eax,edx"  \
    parm [eax edx]  \
    value [eax edx] \
    modify nomemory;
#endif


/* Standard library stuff. */
#ifndef DRFLAC_ASSERT
#include <assert.h>
#define DRFLAC_ASSERT(expression)           assert(expression)
#endif
#ifndef DRFLAC_MALLOC
#define DRFLAC_MALLOC(sz)                   malloc((sz))
#endif
#ifndef DRFLAC_REALLOC
#define DRFLAC_REALLOC(p, sz)               realloc((p), (sz))
#endif
#ifndef DRFLAC_FREE
#define DRFLAC_FREE(p)                      free((p))
#endif
#ifndef DRFLAC_COPY_MEMORY
#define DRFLAC_COPY_MEMORY(dst, src, sz)    memcpy((dst), (src), (sz))
#endif
#ifndef DRFLAC_ZERO_MEMORY
#define DRFLAC_ZERO_MEMORY(p, sz)           memset((p), 0, (sz))
#endif
#ifndef DRFLAC_ZERO_OBJECT
#define DRFLAC_ZERO_OBJECT(p)               DRFLAC_ZERO_MEMORY((p), sizeof(*(p)))
#endif

#define DRFLAC_MAX_SIMD_VECTOR_SIZE                     64  /* 64 for AVX-512 in the future. */

/* Result Codes */
typedef drflac_int32 drflac_result;
#define DRFLAC_SUCCESS                                   0
#define DRFLAC_ERROR                                    -1   /* A generic error. */
#define DRFLAC_INVALID_ARGS                             -2
#define DRFLAC_INVALID_OPERATION                        -3
#define DRFLAC_OUT_OF_MEMORY                            -4
#define DRFLAC_OUT_OF_RANGE                             -5
#define DRFLAC_ACCESS_DENIED                            -6
#define DRFLAC_DOES_NOT_EXIST                           -7
#define DRFLAC_ALREADY_EXISTS                           -8
#define DRFLAC_TOO_MANY_OPEN_FILES                      -9
#define DRFLAC_INVALID_FILE                             -10
#define DRFLAC_TOO_BIG                                  -11
#define DRFLAC_PATH_TOO_LONG                            -12
#define DRFLAC_NAME_TOO_LONG                            -13
#define DRFLAC_NOT_DIRECTORY                            -14
#define DRFLAC_IS_DIRECTORY                             -15
#define DRFLAC_DIRECTORY_NOT_EMPTY                      -16
#define DRFLAC_END_OF_FILE                              -17
#define DRFLAC_NO_SPACE                                 -18
#define DRFLAC_BUSY                                     -19
#define DRFLAC_IO_ERROR                                 -20
#define DRFLAC_INTERRUPT                                -21
#define DRFLAC_UNAVAILABLE                              -22
#define DRFLAC_ALREADY_IN_USE                           -23
#define DRFLAC_BAD_ADDRESS                              -24
#define DRFLAC_BAD_SEEK                                 -25
#define DRFLAC_BAD_PIPE                                 -26
#define DRFLAC_DEADLOCK                                 -27
#define DRFLAC_TOO_MANY_LINKS                           -28
#define DRFLAC_NOT_IMPLEMENTED                          -29
#define DRFLAC_NO_MESSAGE                               -30
#define DRFLAC_BAD_MESSAGE                              -31
#define DRFLAC_NO_DATA_AVAILABLE                        -32
#define DRFLAC_INVALID_DATA                             -33
#define DRFLAC_TIMEOUT                                  -34
#define DRFLAC_NO_NETWORK                               -35
#define DRFLAC_NOT_UNIQUE                               -36
#define DRFLAC_NOT_SOCKET                               -37
#define DRFLAC_NO_ADDRESS                               -38
#define DRFLAC_BAD_PROTOCOL                             -39
#define DRFLAC_PROTOCOL_UNAVAILABLE                     -40
#define DRFLAC_PROTOCOL_NOT_SUPPORTED                   -41
#define DRFLAC_PROTOCOL_FAMILY_NOT_SUPPORTED            -42
#define DRFLAC_ADDRESS_FAMILY_NOT_SUPPORTED             -43
#define DRFLAC_SOCKET_NOT_SUPPORTED                     -44
#define DRFLAC_CONNECTION_RESET                         -45
#define DRFLAC_ALREADY_CONNECTED                        -46
#define DRFLAC_NOT_CONNECTED                            -47
#define DRFLAC_CONNECTION_REFUSED                       -48
#define DRFLAC_NO_HOST                                  -49
#define DRFLAC_IN_PROGRESS                              -50
#define DRFLAC_CANCELLED                                -51
#define DRFLAC_MEMORY_ALREADY_MAPPED                    -52
#define DRFLAC_AT_END                                   -53

#define DRFLAC_CRC_MISMATCH                             -100
/* End Result Codes */


#define DRFLAC_SUBFRAME_CONSTANT                        0
#define DRFLAC_SUBFRAME_VERBATIM                        1
#define DRFLAC_SUBFRAME_FIXED                           8
#define DRFLAC_SUBFRAME_LPC                             32
#define DRFLAC_SUBFRAME_RESERVED                        255

#define DRFLAC_RESIDUAL_CODING_METHOD_PARTITIONED_RICE  0
#define DRFLAC_RESIDUAL_CODING_METHOD_PARTITIONED_RICE2 1

#define DRFLAC_CHANNEL_ASSIGNMENT_INDEPENDENT           0
#define DRFLAC_CHANNEL_ASSIGNMENT_LEFT_SIDE             8
#define DRFLAC_CHANNEL_ASSIGNMENT_RIGHT_SIDE            9
#define DRFLAC_CHANNEL_ASSIGNMENT_MID_SIDE              10

#define DRFLAC_SEEKPOINT_SIZE_IN_BYTES                  18
#define DRFLAC_CUESHEET_TRACK_SIZE_IN_BYTES             36
#define DRFLAC_CUESHEET_TRACK_INDEX_SIZE_IN_BYTES       12

#define drflac_align(x, a)                              ((((x) + (a) - 1) / (a)) * (a))


DRFLAC_API void drflac_version(drflac_uint32* pMajor, drflac_uint32* pMinor, drflac_uint32* pRevision)
{
    if (pMajor) {
        *pMajor = DRFLAC_VERSION_MAJOR;
    }

    if (pMinor) {
        *pMinor = DRFLAC_VERSION_MINOR;
    }

    if (pRevision) {
        *pRevision = DRFLAC_VERSION_REVISION;
    }
}

DRFLAC_API const char* drflac_version_string(void)
{
    return DRFLAC_VERSION_STRING;
}


/* CPU caps. */
#if defined(__has_feature)
    #if __has_feature(thread_sanitizer)
        #define DRFLAC_NO_THREAD_SANITIZE __attribute__((no_sanitize("thread")))
    #else
        #define DRFLAC_NO_THREAD_SANITIZE
    #endif
#else
    #define DRFLAC_NO_THREAD_SANITIZE
#endif

#if defined(DRFLAC_HAS_LZCNT_INTRINSIC)
static drflac_bool32 drflac__gIsLZCNTSupported = DRFLAC_FALSE;
#endif

#ifndef DRFLAC_NO_CPUID
static drflac_bool32 drflac__gIsSSE2Supported  = DRFLAC_FALSE;
static drflac_bool32 drflac__gIsSSE41Supported = DRFLAC_FALSE;

/*
I've had a bug report that Clang's ThreadSanitizer presents a warning in this function. Having reviewed this, this does
actually make sense. However, since CPU caps should never differ for a running process, I don't think the trade off of
complicating internal API's by passing around CPU caps versus just disabling the warnings is worthwhile. I'm therefore
just going to disable these warnings. This is disabled via the DRFLAC_NO_THREAD_SANITIZE attribute.
*/
DRFLAC_NO_THREAD_SANITIZE static void drflac__init_cpu_caps(void)
{
    static drflac_bool32 isCPUCapsInitialized = DRFLAC_FALSE;

    if (!isCPUCapsInitialized) {
        /* LZCNT */
#if defined(DRFLAC_HAS_LZCNT_INTRINSIC)
        int info[4] = {0};
        drflac__cpuid(info, 0x80000001);
        drflac__gIsLZCNTSupported = (info[2] & (1 << 5)) != 0;
#endif

        /* SSE2 */
        drflac__gIsSSE2Supported = drflac_has_sse2();

        /* SSE4.1 */
        drflac__gIsSSE41Supported = drflac_has_sse41();

        /* Initialized. */
        isCPUCapsInitialized = DRFLAC_TRUE;
    }
}
#else
static drflac_bool32 drflac__gIsNEONSupported  = DRFLAC_FALSE;

static DRFLAC_INLINE drflac_bool32 drflac__has_neon(void)
{
#if defined(DRFLAC_SUPPORT_NEON)
    #if defined(DRFLAC_ARM) && !defined(DRFLAC_NO_NEON)
        #if (defined(__ARM_NEON) || defined(__aarch64__) || defined(_M_ARM64))
            return DRFLAC_TRUE;    /* If the compiler is allowed to freely generate NEON code we can assume support. */
        #else
            /* TODO: Runtime check. */
            return DRFLAC_FALSE;
        #endif
    #else
        return DRFLAC_FALSE;       /* NEON is only supported on ARM architectures. */
    #endif
#else
    return DRFLAC_FALSE;           /* No compiler support. */
#endif
}

DRFLAC_NO_THREAD_SANITIZE static void drflac__init_cpu_caps(void)
{
    drflac__gIsNEONSupported = drflac__has_neon();

#if defined(DRFLAC_HAS_LZCNT_INTRINSIC) && defined(DRFLAC_ARM) && (defined(__ARM_ARCH) && __ARM_ARCH >= 5)
    drflac__gIsLZCNTSupported = DRFLAC_TRUE;
#endif
}
#endif


/* Endian Management */
static DRFLAC_INLINE drflac_bool32 drflac__is_little_endian(void)
{
#if defined(DRFLAC_X86) || defined(DRFLAC_X64)
    return DRFLAC_TRUE;
#elif defined(__BYTE_ORDER) && defined(__LITTLE_ENDIAN) && __BYTE_ORDER == __LITTLE_ENDIAN
    return DRFLAC_TRUE;
#else
    int n = 1;
    return (*(char*)&n) == 1;
#endif
}

static DRFLAC_INLINE drflac_uint16 drflac__swap_endian_uint16(drflac_uint16 n)
{
#ifdef DRFLAC_HAS_BYTESWAP16_INTRINSIC
    #if defined(_MSC_VER) && !defined(__clang__)
        return _byteswap_ushort(n);
    #elif defined(__GNUC__) || defined(__clang__)
        return __builtin_bswap16(n);
    #elif defined(__WATCOMC__) && defined(__386__)
        return _watcom_bswap16(n);
    #else
        #error "This compiler does not support the byte swap intrinsic."
    #endif
#else
    return ((n & 0xFF00) >> 8) |
           ((n & 0x00FF) << 8);
#endif
}

static DRFLAC_INLINE drflac_uint32 drflac__swap_endian_uint32(drflac_uint32 n)
{
#ifdef DRFLAC_HAS_BYTESWAP32_INTRINSIC
    #if defined(_MSC_VER) && !defined(__clang__)
        return _byteswap_ulong(n);
    #elif defined(__GNUC__) || defined(__clang__)
        #if defined(DRFLAC_ARM) && (defined(__ARM_ARCH) && __ARM_ARCH >= 6) && !defined(DRFLAC_64BIT)   /* <-- 64-bit inline assembly has not been tested, so disabling for now. */
            /* Inline assembly optimized implementation for ARM. In my testing, GCC does not generate optimized code with __builtin_bswap32(). */
            drflac_uint32 r;
            __asm__ __volatile__ (
            #if defined(DRFLAC_64BIT)
                "rev %w[out], %w[in]" : [out]"=r"(r) : [in]"r"(n)   /* <-- This is untested. If someone in the community could test this, that would be appreciated! */
            #else
                "rev %[out], %[in]" : [out]"=r"(r) : [in]"r"(n)
            #endif
            );
            return r;
        #else
            return __builtin_bswap32(n);
        #endif
    #elif defined(__WATCOMC__) && defined(__386__)
        return _watcom_bswap32(n);
    #else
        #error "This compiler does not support the byte swap intrinsic."
    #endif
#else
    return ((n & 0xFF000000) >> 24) |
           ((n & 0x00FF0000) >>  8) |
           ((n & 0x0000FF00) <<  8) |
           ((n & 0x000000FF) << 24);
#endif
}

static DRFLAC_INLINE drflac_uint64 drflac__swap_endian_uint64(drflac_uint64 n)
{
#ifdef DRFLAC_HAS_BYTESWAP64_INTRINSIC
    #if defined(_MSC_VER) && !defined(__clang__)
        return _byteswap_uint64(n);
    #elif defined(__GNUC__) || defined(__clang__)
        return __builtin_bswap64(n);
    #elif defined(__WATCOMC__) && defined(__386__)
        return _watcom_bswap64(n);
    #else
        #error "This compiler does not support the byte swap intrinsic."
    #endif
#else
    /* Weird "<< 32" bitshift is required for C89 because it doesn't support 64-bit constants. Should be optimized out by a good compiler. */
    return ((n & ((drflac_uint64)0xFF000000 << 32)) >> 56) |
           ((n & ((drflac_uint64)0x00FF0000 << 32)) >> 40) |
           ((n & ((drflac_uint64)0x0000FF00 << 32)) >> 24) |
           ((n & ((drflac_uint64)0x000000FF << 32)) >>  8) |
           ((n & ((drflac_uint64)0xFF000000      )) <<  8) |
           ((n & ((drflac_uint64)0x00FF0000      )) << 24) |
           ((n & ((drflac_uint64)0x0000FF00      )) << 40) |
           ((n & ((drflac_uint64)0x000000FF      )) << 56);
#endif
}


static DRFLAC_INLINE drflac_uint16 drflac__be2host_16(drflac_uint16 n)
{
    if (drflac__is_little_endian()) {
        return drflac__swap_endian_uint16(n);
    }

    return n;
}

static DRFLAC_INLINE drflac_uint32 drflac__be2host_32(drflac_uint32 n)
{
    if (drflac__is_little_endian()) {
        return drflac__swap_endian_uint32(n);
    }

    return n;
}

static DRFLAC_INLINE drflac_uint32 drflac__be2host_32_ptr_unaligned(const void* pData)
{
    const drflac_uint8* pNum = (drflac_uint8*)pData;
    return *(pNum) << 24 | *(pNum+1) << 16 | *(pNum+2) << 8 | *(pNum+3);
}

static DRFLAC_INLINE drflac_uint64 drflac__be2host_64(drflac_uint64 n)
{
    if (drflac__is_little_endian()) {
        return drflac__swap_endian_uint64(n);
    }

    return n;
}


static DRFLAC_INLINE drflac_uint32 drflac__le2host_32(drflac_uint32 n)
{
    if (!drflac__is_little_endian()) {
        return drflac__swap_endian_uint32(n);
    }

    return n;
}

static DRFLAC_INLINE drflac_uint32 drflac__le2host_32_ptr_unaligned(const void* pData)
{
    const drflac_uint8* pNum = (drflac_uint8*)pData;
    return *pNum | *(pNum+1) << 8 |  *(pNum+2) << 16 | *(pNum+3) << 24;
}


static DRFLAC_INLINE drflac_uint32 drflac__unsynchsafe_32(drflac_uint32 n)
{
    drflac_uint32 result = 0;
    result |= (n & 0x7F000000) >> 3;
    result |= (n & 0x007F0000) >> 2;
    result |= (n & 0x00007F00) >> 1;
    result |= (n & 0x0000007F) >> 0;

    return result;
}



/* The CRC code below is based on this document: http://zlib.net/crc_v3.txt */
static drflac_uint8 drflac__crc8_table[] = {
    0x00, 0x07, 0x0E, 0x09, 0x1C, 0x1B, 0x12, 0x15, 0x38, 0x3F, 0x36, 0x31, 0x24, 0x23, 0x2A, 0x2D,
    0x70, 0x77, 0x7E, 0x79, 0x6C, 0x6B, 0x62, 0x65, 0x48, 0x4F, 0x46, 0x41, 0x54, 0x53, 0x5A, 0x5D,
    0xE0, 0xE7, 0xEE, 0xE9, 0xFC, 0xFB, 0xF2, 0xF5, 0xD8, 0xDF, 0xD6, 0xD1, 0xC4, 0xC3, 0xCA, 0xCD,
    0x90, 0x97, 0x9E, 0x99, 0x8C, 0x8B, 0x82, 0x85, 0xA8, 0xAF, 0xA6, 0xA1, 0xB4, 0xB3, 0xBA, 0xBD,
    0xC7, 0xC0, 0xC9, 0xCE, 0xDB, 0xDC, 0xD5, 0xD2, 0xFF, 0xF8, 0xF1, 0xF6, 0xE3, 0xE4, 0xED, 0xEA,
    0xB7, 0xB0, 0xB9, 0xBE, 0xAB, 0xAC, 0xA5, 0xA2, 0x8F, 0x88, 0x81, 0x86, 0x93, 0x94, 0x9D, 0x9A,
    0x27, 0x20, 0x29, 0x2E, 0x3B, 0x3C, 0x35, 0x32, 0x1F, 0x18, 0x11, 0x16, 0x03, 0x04, 0x0D, 0x0A,
    0x57, 0x50, 0x59, 0x5E, 0x4B, 0x4C, 0x45, 0x42, 0x6F, 0x68, 0x61, 0x66, 0x73, 0x74, 0x7D, 0x7A,
    0x89, 0x8E, 0x87, 0x80, 0x95, 0x92, 0x9B, 0x9C, 0xB1, 0xB6, 0xBF, 0xB8, 0xAD, 0xAA, 0xA3, 0xA4,
    0xF9, 0xFE, 0xF7, 0xF0, 0xE5, 0xE2, 0xEB, 0xEC, 0xC1, 0xC6, 0xCF, 0xC8, 0xDD, 0xDA, 0xD3, 0xD4,
    0x69, 0x6E, 0x67, 0x60, 0x75, 0x72, 0x7B, 0x7C, 0x51, 0x56, 0x5F, 0x58, 0x4D, 0x4A, 0x43, 0x44,
    0x19, 0x1E, 0x17, 0x10, 0x05, 0x02, 0x0B, 0x0C, 0x21, 0x26, 0x2F, 0x28, 0x3D, 0x3A, 0x33, 0x34,
    0x4E, 0x49, 0x40, 0x47, 0x52, 0x55, 0x5C, 0x5B, 0x76, 0x71, 0x78, 0x7F, 0x6A, 0x6D, 0x64, 0x63,
    0x3E, 0x39, 0x30, 0x37, 0x22, 0x25, 0x2C, 0x2B, 0x06, 0x01, 0x08, 0x0F, 0x1A, 0x1D, 0x14, 0x13,
    0xAE, 0xA9, 0xA0, 0xA7, 0xB2, 0xB5, 0xBC, 0xBB, 0x96, 0x91, 0x98, 0x9F, 0x8A, 0x8D, 0x84, 0x83,
    0xDE, 0xD9, 0xD0, 0xD7, 0xC2, 0xC5, 0xCC, 0xCB, 0xE6, 0xE1, 0xE8, 0xEF, 0xFA, 0xFD, 0xF4, 0xF3
};

static drflac_uint16 drflac__crc16_table[] = {
    0x0000, 0x8005, 0x800F, 0x000A, 0x801B, 0x001E, 0x0014, 0x8011,
    0x8033, 0x0036, 0x003C, 0x8039, 0x0028, 0x802D, 0x8027, 0x0022,
    0x8063, 0x0066, 0x006C, 0x8069, 0x0078, 0x807D, 0x8077, 0x0072,
    0x0050, 0x8055, 0x805F, 0x005A, 0x804B, 0x004E, 0x0044, 0x8041,
    0x80C3, 0x00C6, 0x00CC, 0x80C9, 0x00D8, 0x80DD, 0x80D7, 0x00D2,
    0x00F0, 0x80F5, 0x80FF, 0x00FA, 0x80EB, 0x00EE, 0x00E4, 0x80E1,
    0x00A0, 0x80A5, 0x80AF, 0x00AA, 0x80BB, 0x00BE, 0x00B4, 0x80B1,
    0x8093, 0x0096, 0x009C, 0x8099, 0x0088, 0x808D, 0x8087, 0x0082,
    0x8183, 0x0186, 0x018C, 0x8189, 0x0198, 0x819D, 0x8197, 0x0192,
    0x01B0, 0x81B5, 0x81BF, 0x01BA, 0x81AB, 0x01AE, 0x01A4, 0x81A1,
    0x01E0, 0x81E5, 0x81EF, 0x01EA, 0x81FB, 0x01FE, 0x01F4, 0x81F1,
    0x81D3, 0x01D6, 0x01DC, 0x81D9, 0x01C8, 0x81CD, 0x81C7, 0x01C2,
    0x0140, 0x8145, 0x814F, 0x014A, 0x815B, 0x015E, 0x0154, 0x8151,
    0x8173, 0x0176, 0x017C, 0x8179, 0x0168, 0x816D, 0x8167, 0x0162,
    0x8123, 0x0126, 0x012C, 0x8129, 0x0138, 0x813D, 0x8137, 0x0132,
    0x0110, 0x8115, 0x811F, 0x011A, 0x810B, 0x010E, 0x0104, 0x8101,
    0x8303, 0x0306, 0x030C, 0x8309, 0x0318, 0x831D, 0x8317, 0x0312,
    0x0330, 0x8335, 0x833F, 0x033A, 0x832B, 0x032E, 0x0324, 0x8321,
    0x0360, 0x8365, 0x836F, 0x036A, 0x837B, 0x037E, 0x0374, 0x8371,
    0x8353, 0x0356, 0x035C, 0x8359, 0x0348, 0x834D, 0x8347, 0x0342,
    0x03C0, 0x83C5, 0x83CF, 0x03CA, 0x83DB, 0x03DE, 0x03D4, 0x83D1,
    0x83F3, 0x03F6, 0x03FC, 0x83F9, 0x03E8, 0x83ED, 0x83E7, 0x03E2,
    0x83A3, 0x03A6, 0x03AC, 0x83A9, 0x03B8, 0x83BD, 0x83B7, 0x03B2,
    0x0390, 0x8395, 0x839F, 0x039A, 0x838B, 0x038E, 0x0384, 0x8381,
    0x0280, 0x8285, 0x828F, 0x028A, 0x829B, 0x029E, 0x0294, 0x8291,
    0x82B3, 0x02B6, 0x02BC, 0x82B9, 0x02A8, 0x82AD, 0x82A7, 0x02A2,
    0x82E3, 0x02E6, 0x02EC, 0x82E9, 0x02F8, 0x82FD, 0x82F7, 0x02F2,
    0x02D0, 0x82D5, 0x82DF, 0x02DA, 0x82CB, 0x02CE, 0x02C4, 0x82C1,
    0x8243, 0x0246, 0x024C, 0x8249, 0x0258, 0x825D, 0x8257, 0x0252,
    0x0270, 0x8275, 0x827F, 0x027A, 0x826B, 0x026E, 0x0264, 0x8261,
    0x0220, 0x8225, 0x822F, 0x022A, 0x823B, 0x023E, 0x0234, 0x8231,
    0x8213, 0x0216, 0x021C, 0x8219, 0x0208, 0x820D, 0x8207, 0x0202
};

static DRFLAC_INLINE drflac_uint8 drflac_crc8_byte(drflac_uint8 crc, drflac_uint8 data)
{
    return drflac__crc8_table[crc ^ data];
}

static DRFLAC_INLINE drflac_uint8 drflac_crc8(drflac_uint8 crc, drflac_uint32 data, drflac_uint32 count)
{
#ifdef DR_FLAC_NO_CRC
    (void)crc;
    (void)data;
    (void)count;
    return 0;
#else
#if 0
    /* REFERENCE (use of this implementation requires an explicit flush by doing "drflac_crc8(crc, 0, 8);") */
    drflac_uint8 p = 0x07;
    for (int i = count-1; i >= 0; --i) {
        drflac_uint8 bit = (data & (1 << i)) >> i;
        if (crc & 0x80) {
            crc = ((crc << 1) | bit) ^ p;
        } else {
            crc = ((crc << 1) | bit);
        }
    }
    return crc;
#else
    drflac_uint32 wholeBytes;
    drflac_uint32 leftoverBits;
    drflac_uint64 leftoverDataMask;

    static drflac_uint64 leftoverDataMaskTable[8] = {
        0x00, 0x01, 0x03, 0x07, 0x0F, 0x1F, 0x3F, 0x7F
    };

    DRFLAC_ASSERT(count <= 32);

    wholeBytes = count >> 3;
    leftoverBits = count - (wholeBytes*8);
    leftoverDataMask = leftoverDataMaskTable[leftoverBits];

    switch (wholeBytes) {
        case 4: crc = drflac_crc8_byte(crc, (drflac_uint8)((data & (0xFF000000UL << leftoverBits)) >> (24 + leftoverBits)));
        case 3: crc = drflac_crc8_byte(crc, (drflac_uint8)((data & (0x00FF0000UL << leftoverBits)) >> (16 + leftoverBits)));
        case 2: crc = drflac_crc8_byte(crc, (drflac_uint8)((data & (0x0000FF00UL << leftoverBits)) >> ( 8 + leftoverBits)));
        case 1: crc = drflac_crc8_byte(crc, (drflac_uint8)((data & (0x000000FFUL << leftoverBits)) >> ( 0 + leftoverBits)));
        case 0: if (leftoverBits > 0) crc = (drflac_uint8)((crc << leftoverBits) ^ drflac__crc8_table[(crc >> (8 - leftoverBits)) ^ (data & leftoverDataMask)]);
    }
    return crc;
#endif
#endif
}

static DRFLAC_INLINE drflac_uint16 drflac_crc16_byte(drflac_uint16 crc, drflac_uint8 data)
{
    return (crc << 8) ^ drflac__crc16_table[(drflac_uint8)(crc >> 8) ^ data];
}

static DRFLAC_INLINE drflac_uint16 drflac_crc16_cache(drflac_uint16 crc, drflac_cache_t data)
{
#ifdef DRFLAC_64BIT
    crc = drflac_crc16_byte(crc, (drflac_uint8)((data >> 56) & 0xFF));
    crc = drflac_crc16_byte(crc, (drflac_uint8)((data >> 48) & 0xFF));
    crc = drflac_crc16_byte(crc, (drflac_uint8)((data >> 40) & 0xFF));
    crc = drflac_crc16_byte(crc, (drflac_uint8)((data >> 32) & 0xFF));
#endif
    crc = drflac_crc16_byte(crc, (drflac_uint8)((data >> 24) & 0xFF));
    crc = drflac_crc16_byte(crc, (drflac_uint8)((data >> 16) & 0xFF));
    crc = drflac_crc16_byte(crc, (drflac_uint8)((data >>  8) & 0xFF));
    crc = drflac_crc16_byte(crc, (drflac_uint8)((data >>  0) & 0xFF));

    return crc;
}

static DRFLAC_INLINE drflac_uint16 drflac_crc16_bytes(drflac_uint16 crc, drflac_cache_t data, drflac_uint32 byteCount)
{
    switch (byteCount)
    {
#ifdef DRFLAC_64BIT
    case 8: crc = drflac_crc16_byte(crc, (drflac_uint8)((data >> 56) & 0xFF));
    case 7: crc = drflac_crc16_byte(crc, (drflac_uint8)((data >> 48) & 0xFF));
    case 6: crc = drflac_crc16_byte(crc, (drflac_uint8)((data >> 40) & 0xFF));
    case 5: crc = drflac_crc16_byte(crc, (drflac_uint8)((data >> 32) & 0xFF));
#endif
    case 4: crc = drflac_crc16_byte(crc, (drflac_uint8)((data >> 24) & 0xFF));
    case 3: crc = drflac_crc16_byte(crc, (drflac_uint8)((data >> 16) & 0xFF));
    case 2: crc = drflac_crc16_byte(crc, (drflac_uint8)((data >>  8) & 0xFF));
    case 1: crc = drflac_crc16_byte(crc, (drflac_uint8)((data >>  0) & 0xFF));
    }

    return crc;
}

#if 0
static DRFLAC_INLINE drflac_uint16 drflac_crc16__32bit(drflac_uint16 crc, drflac_uint32 data, drflac_uint32 count)
{
#ifdef DR_FLAC_NO_CRC
    (void)crc;
    (void)data;
    (void)count;
    return 0;
#else
#if 0
    /* REFERENCE (use of this implementation requires an explicit flush by doing "drflac_crc16(crc, 0, 16);") */
    drflac_uint16 p = 0x8005;
    for (int i = count-1; i >= 0; --i) {
        drflac_uint16 bit = (data & (1ULL << i)) >> i;
        if (r & 0x8000) {
            r = ((r << 1) | bit) ^ p;
        } else {
            r = ((r << 1) | bit);
        }
    }

    return crc;
#else
    drflac_uint32 wholeBytes;
    drflac_uint32 leftoverBits;
    drflac_uint64 leftoverDataMask;

    static drflac_uint64 leftoverDataMaskTable[8] = {
        0x00, 0x01, 0x03, 0x07, 0x0F, 0x1F, 0x3F, 0x7F
    };

    DRFLAC_ASSERT(count <= 64);

    wholeBytes = count >> 3;
    leftoverBits = count & 7;
    leftoverDataMask = leftoverDataMaskTable[leftoverBits];

    switch (wholeBytes) {
        default:
        case 4: crc = drflac_crc16_byte(crc, (drflac_uint8)((data & (0xFF000000UL << leftoverBits)) >> (24 + leftoverBits)));
        case 3: crc = drflac_crc16_byte(crc, (drflac_uint8)((data & (0x00FF0000UL << leftoverBits)) >> (16 + leftoverBits)));
        case 2: crc = drflac_crc16_byte(crc, (drflac_uint8)((data & (0x0000FF00UL << leftoverBits)) >> ( 8 + leftoverBits)));
        case 1: crc = drflac_crc16_byte(crc, (drflac_uint8)((data & (0x000000FFUL << leftoverBits)) >> ( 0 + leftoverBits)));
        case 0: if (leftoverBits > 0) crc = (crc << leftoverBits) ^ drflac__crc16_table[(crc >> (16 - leftoverBits)) ^ (data & leftoverDataMask)];
    }
    return crc;
#endif
#endif
}

static DRFLAC_INLINE drflac_uint16 drflac_crc16__64bit(drflac_uint16 crc, drflac_uint64 data, drflac_uint32 count)
{
#ifdef DR_FLAC_NO_CRC
    (void)crc;
    (void)data;
    (void)count;
    return 0;
#else
    drflac_uint32 wholeBytes;
    drflac_uint32 leftoverBits;
    drflac_uint64 leftoverDataMask;

    static drflac_uint64 leftoverDataMaskTable[8] = {
        0x00, 0x01, 0x03, 0x07, 0x0F, 0x1F, 0x3F, 0x7F
    };

    DRFLAC_ASSERT(count <= 64);

    wholeBytes = count >> 3;
    leftoverBits = count & 7;
    leftoverDataMask = leftoverDataMaskTable[leftoverBits];

    switch (wholeBytes) {
        default:
        case 8: crc = drflac_crc16_byte(crc, (drflac_uint8)((data & (((drflac_uint64)0xFF000000 << 32) << leftoverBits)) >> (56 + leftoverBits)));    /* Weird "<< 32" bitshift is required for C89 because it doesn't support 64-bit constants. Should be optimized out by a good compiler. */
        case 7: crc = drflac_crc16_byte(crc, (drflac_uint8)((data & (((drflac_uint64)0x00FF0000 << 32) << leftoverBits)) >> (48 + leftoverBits)));
        case 6: crc = drflac_crc16_byte(crc, (drflac_uint8)((data & (((drflac_uint64)0x0000FF00 << 32) << leftoverBits)) >> (40 + leftoverBits)));
        case 5: crc = drflac_crc16_byte(crc, (drflac_uint8)((data & (((drflac_uint64)0x000000FF << 32) << leftoverBits)) >> (32 + leftoverBits)));
        case 4: crc = drflac_crc16_byte(crc, (drflac_uint8)((data & (((drflac_uint64)0xFF000000      ) << leftoverBits)) >> (24 + leftoverBits)));
        case 3: crc = drflac_crc16_byte(crc, (drflac_uint8)((data & (((drflac_uint64)0x00FF0000      ) << leftoverBits)) >> (16 + leftoverBits)));
        case 2: crc = drflac_crc16_byte(crc, (drflac_uint8)((data & (((drflac_uint64)0x0000FF00      ) << leftoverBits)) >> ( 8 + leftoverBits)));
        case 1: crc = drflac_crc16_byte(crc, (drflac_uint8)((data & (((drflac_uint64)0x000000FF      ) << leftoverBits)) >> ( 0 + leftoverBits)));
        case 0: if (leftoverBits > 0) crc = (crc << leftoverBits) ^ drflac__crc16_table[(crc >> (16 - leftoverBits)) ^ (data & leftoverDataMask)];
    }
    return crc;
#endif
}


static DRFLAC_INLINE drflac_uint16 drflac_crc16(drflac_uint16 crc, drflac_cache_t data, drflac_uint32 count)
{
#ifdef DRFLAC_64BIT
    return drflac_crc16__64bit(crc, data, count);
#else
    return drflac_crc16__32bit(crc, data, count);
#endif
}
#endif


#ifdef DRFLAC_64BIT
#define drflac__be2host__cache_line drflac__be2host_64
#else
#define drflac__be2host__cache_line drflac__be2host_32
#endif

/*
BIT READING ATTEMPT #2

This uses a 32- or 64-bit bit-shifted cache - as bits are read, the cache is shifted such that the first valid bit is sitting
on the most significant bit. It uses the notion of an L1 and L2 cache (borrowed from CPU architecture), where the L1 cache
is a 32- or 64-bit unsigned integer (depending on whether or not a 32- or 64-bit build is being compiled) and the L2 is an
array of "cache lines", with each cache line being the same size as the L1. The L2 is a buffer of about 4KB and is where data
from onRead() is read into.
*/
#define DRFLAC_CACHE_L1_SIZE_BYTES(bs)                      (sizeof((bs)->cache))
#define DRFLAC_CACHE_L1_SIZE_BITS(bs)                       (sizeof((bs)->cache)*8)
#define DRFLAC_CACHE_L1_BITS_REMAINING(bs)                  (DRFLAC_CACHE_L1_SIZE_BITS(bs) - (bs)->consumedBits)
#define DRFLAC_CACHE_L1_SELECTION_MASK(_bitCount)           (~((~(drflac_cache_t)0) >> (_bitCount)))
#define DRFLAC_CACHE_L1_SELECTION_SHIFT(bs, _bitCount)      (DRFLAC_CACHE_L1_SIZE_BITS(bs) - (_bitCount))
#define DRFLAC_CACHE_L1_SELECT(bs, _bitCount)               (((bs)->cache) & DRFLAC_CACHE_L1_SELECTION_MASK(_bitCount))
#define DRFLAC_CACHE_L1_SELECT_AND_SHIFT(bs, _bitCount)     (DRFLAC_CACHE_L1_SELECT((bs), (_bitCount)) >>  DRFLAC_CACHE_L1_SELECTION_SHIFT((bs), (_bitCount)))
#define DRFLAC_CACHE_L1_SELECT_AND_SHIFT_SAFE(bs, _bitCount)(DRFLAC_CACHE_L1_SELECT((bs), (_bitCount)) >> (DRFLAC_CACHE_L1_SELECTION_SHIFT((bs), (_bitCount)) & (DRFLAC_CACHE_L1_SIZE_BITS(bs)-1)))
#define DRFLAC_CACHE_L2_SIZE_BYTES(bs)                      (sizeof((bs)->cacheL2))
#define DRFLAC_CACHE_L2_LINE_COUNT(bs)                      (DRFLAC_CACHE_L2_SIZE_BYTES(bs) / sizeof((bs)->cacheL2[0]))
#define DRFLAC_CACHE_L2_LINES_REMAINING(bs)                 (DRFLAC_CACHE_L2_LINE_COUNT(bs) - (bs)->nextL2Line)


#ifndef DR_FLAC_NO_CRC
static DRFLAC_INLINE void drflac__reset_crc16(drflac_bs* bs)
{
    bs->crc16 = 0;
    bs->crc16CacheIgnoredBytes = bs->consumedBits >> 3;
}

static DRFLAC_INLINE void drflac__update_crc16(drflac_bs* bs)
{
    if (bs->crc16CacheIgnoredBytes == 0) {
        bs->crc16 = drflac_crc16_cache(bs->crc16, bs->crc16Cache);
    } else {
        bs->crc16 = drflac_crc16_bytes(bs->crc16, bs->crc16Cache, DRFLAC_CACHE_L1_SIZE_BYTES(bs) - bs->crc16CacheIgnoredBytes);
        bs->crc16CacheIgnoredBytes = 0;
    }
}

static DRFLAC_INLINE drflac_uint16 drflac__flush_crc16(drflac_bs* bs)
{
    /* We should never be flushing in a situation where we are not aligned on a byte boundary. */
    DRFLAC_ASSERT((DRFLAC_CACHE_L1_BITS_REMAINING(bs) & 7) == 0);

    /*
    The bits that were read from the L1 cache need to be accumulated. The number of bytes needing to be accumulated is determined
    by the number of bits that have been consumed.
    */
    if (DRFLAC_CACHE_L1_BITS_REMAINING(bs) == 0) {
        drflac__update_crc16(bs);
    } else {
        /* We only accumulate the consumed bits. */
        bs->crc16 = drflac_crc16_bytes(bs->crc16, bs->crc16Cache >> DRFLAC_CACHE_L1_BITS_REMAINING(bs), (bs->consumedBits >> 3) - bs->crc16CacheIgnoredBytes);

        /*
        The bits that we just accumulated should never be accumulated again. We need to keep track of how many bytes were accumulated
        so we can handle that later.
        */
        bs->crc16CacheIgnoredBytes = bs->consumedBits >> 3;
    }

    return bs->crc16;
}
#endif

static DRFLAC_INLINE drflac_bool32 drflac__reload_l1_cache_from_l2(drflac_bs* bs)
{
    size_t bytesRead;
    size_t alignedL1LineCount;

    /* Fast path. Try loading straight from L2. */
    if (bs->nextL2Line < DRFLAC_CACHE_L2_LINE_COUNT(bs)) {
        bs->cache = bs->cacheL2[bs->nextL2Line++];
        return DRFLAC_TRUE;
    }

    /*
    If we get here it means we've run out of data in the L2 cache. We'll need to fetch more from the client, if there's
    any left.
    */
    if (bs->unalignedByteCount > 0) {
        return DRFLAC_FALSE;   /* If we have any unaligned bytes it means there's no more aligned bytes left in the client. */
    }

    bytesRead = bs->onRead(bs->pUserData, bs->cacheL2, DRFLAC_CACHE_L2_SIZE_BYTES(bs));

    bs->nextL2Line = 0;
    if (bytesRead == DRFLAC_CACHE_L2_SIZE_BYTES(bs)) {
        bs->cache = bs->cacheL2[bs->nextL2Line++];
        return DRFLAC_TRUE;
    }


    /*
    If we get here it means we were unable to retrieve enough data to fill the entire L2 cache. It probably
    means we've just reached the end of the file. We need to move the valid data down to the end of the buffer
    and adjust the index of the next line accordingly. Also keep in mind that the L2 cache must be aligned to
    the size of the L1 so we'll need to seek backwards by any misaligned bytes.
    */
    alignedL1LineCount = bytesRead / DRFLAC_CACHE_L1_SIZE_BYTES(bs);

    /* We need to keep track of any unaligned bytes for later use. */
    bs->unalignedByteCount = bytesRead - (alignedL1LineCount * DRFLAC_CACHE_L1_SIZE_BYTES(bs));
    if (bs->unalignedByteCount > 0) {
        bs->unalignedCache = bs->cacheL2[alignedL1LineCount];
    }

    if (alignedL1LineCount > 0) {
        size_t offset = DRFLAC_CACHE_L2_LINE_COUNT(bs) - alignedL1LineCount;
        size_t i;
        for (i = alignedL1LineCount; i > 0; --i) {
            bs->cacheL2[i-1 + offset] = bs->cacheL2[i-1];
        }

        bs->nextL2Line = (drflac_uint32)offset;
        bs->cache = bs->cacheL2[bs->nextL2Line++];
        return DRFLAC_TRUE;
    } else {
        /* If we get into this branch it means we weren't able to load any L1-aligned data. */
        bs->nextL2Line = DRFLAC_CACHE_L2_LINE_COUNT(bs);
        return DRFLAC_FALSE;
    }
}

static drflac_bool32 drflac__reload_cache(drflac_bs* bs)
{
    size_t bytesRead;

#ifndef DR_FLAC_NO_CRC
    drflac__update_crc16(bs);
#endif

    /* Fast path. Try just moving the next value in the L2 cache to the L1 cache. */
    if (drflac__reload_l1_cache_from_l2(bs)) {
        bs->cache = drflac__be2host__cache_line(bs->cache);
        bs->consumedBits = 0;
#ifndef DR_FLAC_NO_CRC
        bs->crc16Cache = bs->cache;
#endif
        return DRFLAC_TRUE;
    }

    /* Slow path. */

    /*
    If we get here it means we have failed to load the L1 cache from the L2. Likely we've just reached the end of the stream and the last
    few bytes did not meet the alignment requirements for the L2 cache. In this case we need to fall back to a slower path and read the
    data from the unaligned cache.
    */
    bytesRead = bs->unalignedByteCount;
    if (bytesRead == 0) {
        bs->consumedBits = DRFLAC_CACHE_L1_SIZE_BITS(bs);   /* <-- The stream has been exhausted, so marked the bits as consumed. */
        return DRFLAC_FALSE;
    }

    DRFLAC_ASSERT(bytesRead < DRFLAC_CACHE_L1_SIZE_BYTES(bs));
    bs->consumedBits = (drflac_uint32)(DRFLAC_CACHE_L1_SIZE_BYTES(bs) - bytesRead) * 8;

    bs->cache = drflac__be2host__cache_line(bs->unalignedCache);
    bs->cache &= DRFLAC_CACHE_L1_SELECTION_MASK(DRFLAC_CACHE_L1_BITS_REMAINING(bs));    /* <-- Make sure the consumed bits are always set to zero. Other parts of the library depend on this property. */
    bs->unalignedByteCount = 0;     /* <-- At this point the unaligned bytes have been moved into the cache and we thus have no more unaligned bytes. */

#ifndef DR_FLAC_NO_CRC
    bs->crc16Cache = bs->cache >> bs->consumedBits;
    bs->crc16CacheIgnoredBytes = bs->consumedBits >> 3;
#endif
    return DRFLAC_TRUE;
}

static void drflac__reset_cache(drflac_bs* bs)
{
    bs->nextL2Line   = DRFLAC_CACHE_L2_LINE_COUNT(bs);  /* <-- This clears the L2 cache. */
    bs->consumedBits = DRFLAC_CACHE_L1_SIZE_BITS(bs);   /* <-- This clears the L1 cache. */
    bs->cache = 0;
    bs->unalignedByteCount = 0;                         /* <-- This clears the trailing unaligned bytes. */
    bs->unalignedCache = 0;

#ifndef DR_FLAC_NO_CRC
    bs->crc16Cache = 0;
    bs->crc16CacheIgnoredBytes = 0;
#endif
}


static DRFLAC_INLINE drflac_bool32 drflac__read_uint32(drflac_bs* bs, unsigned int bitCount, drflac_uint32* pResultOut)
{
    DRFLAC_ASSERT(bs != NULL);
    DRFLAC_ASSERT(pResultOut != NULL);
    DRFLAC_ASSERT(bitCount > 0);
    DRFLAC_ASSERT(bitCount <= 32);

    if (bs->consumedBits == DRFLAC_CACHE_L1_SIZE_BITS(bs)) {
        if (!drflac__reload_cache(bs)) {
            return DRFLAC_FALSE;
        }
    }

    if (bitCount <= DRFLAC_CACHE_L1_BITS_REMAINING(bs)) {
        /*
        If we want to load all 32-bits from a 32-bit cache we need to do it slightly differently because we can't do
        a 32-bit shift on a 32-bit integer. This will never be the case on 64-bit caches, so we can have a slightly
        more optimal solution for this.
        */
#ifdef DRFLAC_64BIT
        *pResultOut = (drflac_uint32)DRFLAC_CACHE_L1_SELECT_AND_SHIFT(bs, bitCount);
        bs->consumedBits += bitCount;
        bs->cache <<= bitCount;
#else
        if (bitCount < DRFLAC_CACHE_L1_SIZE_BITS(bs)) {
            *pResultOut = (drflac_uint32)DRFLAC_CACHE_L1_SELECT_AND_SHIFT(bs, bitCount);
            bs->consumedBits += bitCount;
            bs->cache <<= bitCount;
        } else {
            /* Cannot shift by 32-bits, so need to do it differently. */
            *pResultOut = (drflac_uint32)bs->cache;
            bs->consumedBits = DRFLAC_CACHE_L1_SIZE_BITS(bs);
            bs->cache = 0;
        }
#endif

        return DRFLAC_TRUE;
    } else {
        /* It straddles the cached data. It will never cover more than the next chunk. We just read the number in two parts and combine them. */
        drflac_uint32 bitCountHi = DRFLAC_CACHE_L1_BITS_REMAINING(bs);
        drflac_uint32 bitCountLo = bitCount - bitCountHi;
        drflac_uint32 resultHi;

        DRFLAC_ASSERT(bitCountHi > 0);
        DRFLAC_ASSERT(bitCountHi < 32);
        resultHi = (drflac_uint32)DRFLAC_CACHE_L1_SELECT_AND_SHIFT(bs, bitCountHi);

        if (!drflac__reload_cache(bs)) {
            return DRFLAC_FALSE;
        }
        if (bitCountLo > DRFLAC_CACHE_L1_BITS_REMAINING(bs)) {
            /* This happens when we get to end of stream */
            return DRFLAC_FALSE;
        }

        *pResultOut = (resultHi << bitCountLo) | (drflac_uint32)DRFLAC_CACHE_L1_SELECT_AND_SHIFT(bs, bitCountLo);
        bs->consumedBits += bitCountLo;
        bs->cache <<= bitCountLo;
        return DRFLAC_TRUE;
    }
}

static drflac_bool32 drflac__read_int32(drflac_bs* bs, unsigned int bitCount, drflac_int32* pResult)
{
    drflac_uint32 result;

    DRFLAC_ASSERT(bs != NULL);
    DRFLAC_ASSERT(pResult != NULL);
    DRFLAC_ASSERT(bitCount > 0);
    DRFLAC_ASSERT(bitCount <= 32);

    if (!drflac__read_uint32(bs, bitCount, &result)) {
        return DRFLAC_FALSE;
    }

    /* Do not attempt to shift by 32 as it's undefined. */
    if (bitCount < 32) {
        drflac_uint32 signbit;
        signbit = ((result >> (bitCount-1)) & 0x01);
        result |= (~signbit + 1) << bitCount;
    }

    *pResult = (drflac_int32)result;
    return DRFLAC_TRUE;
}

#ifdef DRFLAC_64BIT
static drflac_bool32 drflac__read_uint64(drflac_bs* bs, unsigned int bitCount, drflac_uint64* pResultOut)
{
    drflac_uint32 resultHi;
    drflac_uint32 resultLo;

    DRFLAC_ASSERT(bitCount <= 64);
    DRFLAC_ASSERT(bitCount >  32);

    if (!drflac__read_uint32(bs, bitCount - 32, &resultHi)) {
        return DRFLAC_FALSE;
    }

    if (!drflac__read_uint32(bs, 32, &resultLo)) {
        return DRFLAC_FALSE;
    }

    *pResultOut = (((drflac_uint64)resultHi) << 32) | ((drflac_uint64)resultLo);
    return DRFLAC_TRUE;
}
#endif

/* Function below is unused, but leaving it here in case I need to quickly add it again. */
#if 0
static drflac_bool32 drflac__read_int64(drflac_bs* bs, unsigned int bitCount, drflac_int64* pResultOut)
{
    drflac_uint64 result;
    drflac_uint64 signbit;

    DRFLAC_ASSERT(bitCount <= 64);

    if (!drflac__read_uint64(bs, bitCount, &result)) {
        return DRFLAC_FALSE;
    }

    signbit = ((result >> (bitCount-1)) & 0x01);
    result |= (~signbit + 1) << bitCount;

    *pResultOut = (drflac_int64)result;
    return DRFLAC_TRUE;
}
#endif

static drflac_bool32 drflac__read_uint16(drflac_bs* bs, unsigned int bitCount, drflac_uint16* pResult)
{
    drflac_uint32 result;

    DRFLAC_ASSERT(bs != NULL);
    DRFLAC_ASSERT(pResult != NULL);
    DRFLAC_ASSERT(bitCount > 0);
    DRFLAC_ASSERT(bitCount <= 16);

    if (!drflac__read_uint32(bs, bitCount, &result)) {
        return DRFLAC_FALSE;
    }

    *pResult = (drflac_uint16)result;
    return DRFLAC_TRUE;
}

#if 0
static drflac_bool32 drflac__read_int16(drflac_bs* bs, unsigned int bitCount, drflac_int16* pResult)
{
    drflac_int32 result;

    DRFLAC_ASSERT(bs != NULL);
    DRFLAC_ASSERT(pResult != NULL);
    DRFLAC_ASSERT(bitCount > 0);
    DRFLAC_ASSERT(bitCount <= 16);

    if (!drflac__read_int32(bs, bitCount, &result)) {
        return DRFLAC_FALSE;
    }

    *pResult = (drflac_int16)result;
    return DRFLAC_TRUE;
}
#endif

static drflac_bool32 drflac__read_uint8(drflac_bs* bs, unsigned int bitCount, drflac_uint8* pResult)
{
    drflac_uint32 result;

    DRFLAC_ASSERT(bs != NULL);
    DRFLAC_ASSERT(pResult != NULL);
    DRFLAC_ASSERT(bitCount > 0);
    DRFLAC_ASSERT(bitCount <= 8);

    if (!drflac__read_uint32(bs, bitCount, &result)) {
        return DRFLAC_FALSE;
    }

    *pResult = (drflac_uint8)result;
    return DRFLAC_TRUE;
}

static drflac_bool32 drflac__read_int8(drflac_bs* bs, unsigned int bitCount, drflac_int8* pResult)
{
    drflac_int32 result;

    DRFLAC_ASSERT(bs != NULL);
    DRFLAC_ASSERT(pResult != NULL);
    DRFLAC_ASSERT(bitCount > 0);
    DRFLAC_ASSERT(bitCount <= 8);

    if (!drflac__read_int32(bs, bitCount, &result)) {
        return DRFLAC_FALSE;
    }

    *pResult = (drflac_int8)result;
    return DRFLAC_TRUE;
}


static drflac_bool32 drflac__seek_bits(drflac_bs* bs, size_t bitsToSeek)
{
    if (bitsToSeek <= DRFLAC_CACHE_L1_BITS_REMAINING(bs)) {
        bs->consumedBits += (drflac_uint32)bitsToSeek;
        bs->cache <<= bitsToSeek;
        return DRFLAC_TRUE;
    } else {
        /* It straddles the cached data. This function isn't called too frequently so I'm favouring simplicity here. */
        bitsToSeek       -= DRFLAC_CACHE_L1_BITS_REMAINING(bs);
        bs->consumedBits += DRFLAC_CACHE_L1_BITS_REMAINING(bs);
        bs->cache         = 0;

        /* Simple case. Seek in groups of the same number as bits that fit within a cache line. */
#ifdef DRFLAC_64BIT
        while (bitsToSeek >= DRFLAC_CACHE_L1_SIZE_BITS(bs)) {
            drflac_uint64 bin;
            if (!drflac__read_uint64(bs, DRFLAC_CACHE_L1_SIZE_BITS(bs), &bin)) {
                return DRFLAC_FALSE;
            }
            bitsToSeek -= DRFLAC_CACHE_L1_SIZE_BITS(bs);
        }
#else
        while (bitsToSeek >= DRFLAC_CACHE_L1_SIZE_BITS(bs)) {
            drflac_uint32 bin;
            if (!drflac__read_uint32(bs, DRFLAC_CACHE_L1_SIZE_BITS(bs), &bin)) {
                return DRFLAC_FALSE;
            }
            bitsToSeek -= DRFLAC_CACHE_L1_SIZE_BITS(bs);
        }
#endif

        /* Whole leftover bytes. */
        while (bitsToSeek >= 8) {
            drflac_uint8 bin;
            if (!drflac__read_uint8(bs, 8, &bin)) {
                return DRFLAC_FALSE;
            }
            bitsToSeek -= 8;
        }

        /* Leftover bits. */
        if (bitsToSeek > 0) {
            drflac_uint8 bin;
            if (!drflac__read_uint8(bs, (drflac_uint32)bitsToSeek, &bin)) {
                return DRFLAC_FALSE;
            }
            bitsToSeek = 0; /* <-- Necessary for the assert below. */
        }

        DRFLAC_ASSERT(bitsToSeek == 0);
        return DRFLAC_TRUE;
    }
}


/* This function moves the bit streamer to the first bit after the sync code (bit 15 of the of the frame header). It will also update the CRC-16. */
static drflac_bool32 drflac__find_and_seek_to_next_sync_code(drflac_bs* bs)
{
    DRFLAC_ASSERT(bs != NULL);

    /*
    The sync code is always aligned to 8 bits. This is convenient for us because it means we can do byte-aligned movements. The first
    thing to do is align to the next byte.
    */
    if (!drflac__seek_bits(bs, DRFLAC_CACHE_L1_BITS_REMAINING(bs) & 7)) {
        return DRFLAC_FALSE;
    }

    for (;;) {
        drflac_uint8 hi;

#ifndef DR_FLAC_NO_CRC
        drflac__reset_crc16(bs);
#endif

        if (!drflac__read_uint8(bs, 8, &hi)) {
            return DRFLAC_FALSE;
        }

        if (hi == 0xFF) {
            drflac_uint8 lo;
            if (!drflac__read_uint8(bs, 6, &lo)) {
                return DRFLAC_FALSE;
            }

            if (lo == 0x3E) {
                return DRFLAC_TRUE;
            } else {
                if (!drflac__seek_bits(bs, DRFLAC_CACHE_L1_BITS_REMAINING(bs) & 7)) {
                    return DRFLAC_FALSE;
                }
            }
        }
    }

    /* Should never get here. */
    /*return DRFLAC_FALSE;*/
}


#if defined(DRFLAC_HAS_LZCNT_INTRINSIC)
#define DRFLAC_IMPLEMENT_CLZ_LZCNT
#endif
#if  defined(_MSC_VER) && _MSC_VER >= 1400 && (defined(DRFLAC_X64) || defined(DRFLAC_X86)) && !defined(__clang__)
#define DRFLAC_IMPLEMENT_CLZ_MSVC
#endif
#if  defined(__WATCOMC__) && defined(__386__)
#define DRFLAC_IMPLEMENT_CLZ_WATCOM
#endif
#ifdef __MRC__
#include <intrinsics.h>
#define DRFLAC_IMPLEMENT_CLZ_MRC
#endif

static DRFLAC_INLINE drflac_uint32 drflac__clz_software(drflac_cache_t x)
{
    drflac_uint32 n;
    static drflac_uint32 clz_table_4[] = {
        0,
        4,
        3, 3,
        2, 2, 2, 2,
        1, 1, 1, 1, 1, 1, 1, 1
    };

    if (x == 0) {
        return sizeof(x)*8;
    }

    n = clz_table_4[x >> (sizeof(x)*8 - 4)];
    if (n == 0) {
#ifdef DRFLAC_64BIT
        if ((x & ((drflac_uint64)0xFFFFFFFF << 32)) == 0) { n  = 32; x <<= 32; }
        if ((x & ((drflac_uint64)0xFFFF0000 << 32)) == 0) { n += 16; x <<= 16; }
        if ((x & ((drflac_uint64)0xFF000000 << 32)) == 0) { n += 8;  x <<= 8;  }
        if ((x & ((drflac_uint64)0xF0000000 << 32)) == 0) { n += 4;  x <<= 4;  }
#else
        if ((x & 0xFFFF0000) == 0) { n  = 16; x <<= 16; }
        if ((x & 0xFF000000) == 0) { n += 8;  x <<= 8;  }
        if ((x & 0xF0000000) == 0) { n += 4;  x <<= 4;  }
#endif
        n += clz_table_4[x >> (sizeof(x)*8 - 4)];
    }

    return n - 1;
}

#ifdef DRFLAC_IMPLEMENT_CLZ_LZCNT
static DRFLAC_INLINE drflac_bool32 drflac__is_lzcnt_supported(void)
{
    /* Fast compile time check for ARM. */
#if defined(DRFLAC_HAS_LZCNT_INTRINSIC) && defined(DRFLAC_ARM) && (defined(__ARM_ARCH) && __ARM_ARCH >= 5)
    return DRFLAC_TRUE;
#elif defined(__MRC__)
    return DRFLAC_TRUE;
#else
    /* If the compiler itself does not support the intrinsic then we'll need to return false. */
    #ifdef DRFLAC_HAS_LZCNT_INTRINSIC
        return drflac__gIsLZCNTSupported;
    #else
        return DRFLAC_FALSE;
    #endif
#endif
}

static DRFLAC_INLINE drflac_uint32 drflac__clz_lzcnt(drflac_cache_t x)
{
    /*
    It's critical for competitive decoding performance that this function be highly optimal. With MSVC we can use the __lzcnt64() and __lzcnt() intrinsics
    to achieve good performance, however on GCC and Clang it's a little bit more annoying. The __builtin_clzl() and __builtin_clzll() intrinsics leave
    it undefined as to the return value when `x` is 0. We need this to be well defined as returning 32 or 64, depending on whether or not it's a 32- or
    64-bit build. To work around this we would need to add a conditional to check for the x = 0 case, but this creates unnecessary inefficiency. To work
    around this problem I have written some inline assembly to emit the LZCNT (x86) or CLZ (ARM) instruction directly which removes the need to include
    the conditional. This has worked well in the past, but for some reason Clang's MSVC compatible driver, clang-cl, does not seem to be handling this
    in the same way as the normal Clang driver. It seems that `clang-cl` is just outputting the wrong results sometimes, maybe due to some register
    getting clobbered?

    I'm not sure if this is a bug with dr_flac's inlined assembly (most likely), a bug in `clang-cl` or just a misunderstanding on my part with inline
    assembly rules for `clang-cl`. If somebody can identify an error in dr_flac's inlined assembly I'm happy to get that fixed.

    Fortunately there is an easy workaround for this. Clang implements MSVC-specific intrinsics for compatibility. It also defines _MSC_VER for extra
    compatibility. We can therefore just check for _MSC_VER and use the MSVC intrinsic which, fortunately for us, Clang supports. It would still be nice
    to know how to fix the inlined assembly for correctness sake, however.
    */

#if defined(_MSC_VER) /*&& !defined(__clang__)*/    /* <-- Intentionally wanting Clang to use the MSVC __lzcnt64/__lzcnt intrinsics due to above ^. */
    #ifdef DRFLAC_64BIT
        return (drflac_uint32)__lzcnt64(x);
    #else
        return (drflac_uint32)__lzcnt(x);
    #endif
#else
    #if defined(__GNUC__) || defined(__clang__)
        #if defined(DRFLAC_X64)
            {
                drflac_uint64 r;
                __asm__ __volatile__ (
                    "lzcnt{ %1, %0| %0, %1}" : "=r"(r) : "r"(x) : "cc"
                );

                return (drflac_uint32)r;
            }
        #elif defined(DRFLAC_X86)
            {
                drflac_uint32 r;
                __asm__ __volatile__ (
                    "lzcnt{l %1, %0| %0, %1}" : "=r"(r) : "r"(x) : "cc"
                );

                return r;
            }
        #elif defined(DRFLAC_ARM) && (defined(__ARM_ARCH) && __ARM_ARCH >= 5) && !defined(DRFLAC_64BIT)   /* <-- I haven't tested 64-bit inline assembly, so only enabling this for the 32-bit build for now. */
            {
                unsigned int r;
                __asm__ __volatile__ (
                #if defined(DRFLAC_64BIT)
                    "clz %w[out], %w[in]" : [out]"=r"(r) : [in]"r"(x)   /* <-- This is untested. If someone in the community could test this, that would be appreciated! */
                #else
                    "clz %[out], %[in]" : [out]"=r"(r) : [in]"r"(x)
                #endif
                );

                return r;
            }
        #else
            if (x == 0) {
                return sizeof(x)*8;
            }
            #ifdef DRFLAC_64BIT
                return (drflac_uint32)__builtin_clzll((drflac_uint64)x);
            #else
                return (drflac_uint32)__builtin_clzl((drflac_uint32)x);
            #endif
        #endif
    #else
        /* Unsupported compiler. */
        #error "This compiler does not support the lzcnt intrinsic."
    #endif
#endif
}
#endif

#ifdef DRFLAC_IMPLEMENT_CLZ_MSVC
#include <intrin.h> /* For BitScanReverse(). */

static DRFLAC_INLINE drflac_uint32 drflac__clz_msvc(drflac_cache_t x)
{
    drflac_uint32 n;

    if (x == 0) {
        return sizeof(x)*8;
    }

#ifdef DRFLAC_64BIT
    _BitScanReverse64((unsigned long*)&n, x);
#else
    _BitScanReverse((unsigned long*)&n, x);
#endif
    return sizeof(x)*8 - n - 1;
}
#endif

#ifdef DRFLAC_IMPLEMENT_CLZ_WATCOM
static __inline drflac_uint32 drflac__clz_watcom (drflac_uint32);
#ifdef DRFLAC_IMPLEMENT_CLZ_WATCOM_LZCNT
/* Use the LZCNT instruction (only available on some processors since the 2010s). */
#pragma aux drflac__clz_watcom_lzcnt = \
    "db 0F3h, 0Fh, 0BDh, 0C0h" /* lzcnt eax, eax */ \
    parm [eax] \
    value [eax] \
    modify nomemory;
#else
/* Use the 386+-compatible implementation. */
#pragma aux drflac__clz_watcom = \
    "bsr eax, eax" \
    "xor eax, 31" \
    parm [eax] nomemory \
    value [eax] \
    modify exact [eax] nomemory;
#endif
#endif

static DRFLAC_INLINE drflac_uint32 drflac__clz(drflac_cache_t x)
{
#ifdef DRFLAC_IMPLEMENT_CLZ_LZCNT
    if (drflac__is_lzcnt_supported()) {
        return drflac__clz_lzcnt(x);
    } else
#endif
    {
#ifdef DRFLAC_IMPLEMENT_CLZ_MSVC
        return drflac__clz_msvc(x);
#elif defined(DRFLAC_IMPLEMENT_CLZ_WATCOM_LZCNT)
        return drflac__clz_watcom_lzcnt(x);
#elif defined(DRFLAC_IMPLEMENT_CLZ_WATCOM)
        return (x == 0) ? sizeof(x)*8 : drflac__clz_watcom(x);
#elif defined(__MRC__)
        return __cntlzw(x);
#else
        return drflac__clz_software(x);
#endif
    }
}


static DRFLAC_INLINE drflac_bool32 drflac__seek_past_next_set_bit(drflac_bs* bs, unsigned int* pOffsetOut)
{
    drflac_uint32 zeroCounter = 0;
    drflac_uint32 setBitOffsetPlus1;

    while (bs->cache == 0) {
        zeroCounter += (drflac_uint32)DRFLAC_CACHE_L1_BITS_REMAINING(bs);
        if (!drflac__reload_cache(bs)) {
            return DRFLAC_FALSE;
        }
    }

    if (bs->cache == 1) {
        /* Not catching this would lead to undefined behaviour: a shift of a 32-bit number by 32 or more is undefined */
        *pOffsetOut = zeroCounter + (drflac_uint32)DRFLAC_CACHE_L1_BITS_REMAINING(bs) - 1;
        if (!drflac__reload_cache(bs)) {
            return DRFLAC_FALSE;
        }

        return DRFLAC_TRUE;
    }

    setBitOffsetPlus1 = drflac__clz(bs->cache);
    setBitOffsetPlus1 += 1;

    if (setBitOffsetPlus1 > DRFLAC_CACHE_L1_BITS_REMAINING(bs)) {
        /* This happens when we get to end of stream */
        return DRFLAC_FALSE;
    }

    bs->consumedBits += setBitOffsetPlus1;
    bs->cache <<= setBitOffsetPlus1;

    *pOffsetOut = zeroCounter + setBitOffsetPlus1 - 1;
    return DRFLAC_TRUE;
}



static drflac_bool32 drflac__seek_to_byte(drflac_bs* bs, drflac_uint64 offsetFromStart)
{
    DRFLAC_ASSERT(bs != NULL);
    DRFLAC_ASSERT(offsetFromStart > 0);

    /*
    Seeking from the start is not quite as trivial as it sounds because the onSeek callback takes a signed 32-bit integer (which
    is intentional because it simplifies the implementation of the onSeek callbacks), however offsetFromStart is unsigned 64-bit.
    To resolve we just need to do an initial seek from the start, and then a series of offset seeks to make up the remainder.
    */
    if (offsetFromStart > 0x7FFFFFFF) {
        drflac_uint64 bytesRemaining = offsetFromStart;
        if (!bs->onSeek(bs->pUserData, 0x7FFFFFFF, drflac_seek_origin_start)) {
            return DRFLAC_FALSE;
        }
        bytesRemaining -= 0x7FFFFFFF;

        while (bytesRemaining > 0x7FFFFFFF) {
            if (!bs->onSeek(bs->pUserData, 0x7FFFFFFF, drflac_seek_origin_current)) {
                return DRFLAC_FALSE;
            }
            bytesRemaining -= 0x7FFFFFFF;
        }

        if (bytesRemaining > 0) {
            if (!bs->onSeek(bs->pUserData, (int)bytesRemaining, drflac_seek_origin_current)) {
                return DRFLAC_FALSE;
            }
        }
    } else {
        if (!bs->onSeek(bs->pUserData, (int)offsetFromStart, drflac_seek_origin_start)) {
            return DRFLAC_FALSE;
        }
    }

    /* The cache should be reset to force a reload of fresh data from the client. */
    drflac__reset_cache(bs);
    return DRFLAC_TRUE;
}


static drflac_result drflac__read_utf8_coded_number(drflac_bs* bs, drflac_uint64* pNumberOut, drflac_uint8* pCRCOut)
{
    drflac_uint8 crc;
    drflac_uint64 result;
    drflac_uint8 utf8[7] = {0};
    int byteCount;
    int i;

    DRFLAC_ASSERT(bs != NULL);
    DRFLAC_ASSERT(pNumberOut != NULL);
    DRFLAC_ASSERT(pCRCOut != NULL);

    crc = *pCRCOut;

    if (!drflac__read_uint8(bs, 8, utf8)) {
        *pNumberOut = 0;
        return DRFLAC_AT_END;
    }
    crc = drflac_crc8(crc, utf8[0], 8);

    if ((utf8[0] & 0x80) == 0) {
        *pNumberOut = utf8[0];
        *pCRCOut = crc;
        return DRFLAC_SUCCESS;
    }

    /*byteCount = 1;*/
    if ((utf8[0] & 0xE0) == 0xC0) {
        byteCount = 2;
    } else if ((utf8[0] & 0xF0) == 0xE0) {
        byteCount = 3;
    } else if ((utf8[0] & 0xF8) == 0xF0) {
        byteCount = 4;
    } else if ((utf8[0] & 0xFC) == 0xF8) {
        byteCount = 5;
    } else if ((utf8[0] & 0xFE) == 0xFC) {
        byteCount = 6;
    } else if ((utf8[0] & 0xFF) == 0xFE) {
        byteCount = 7;
    } else {
        *pNumberOut = 0;
        return DRFLAC_CRC_MISMATCH;     /* Bad UTF-8 encoding. */
    }

    /* Read extra bytes. */
    DRFLAC_ASSERT(byteCount > 1);

    result = (drflac_uint64)(utf8[0] & (0xFF >> (byteCount + 1)));
    for (i = 1; i < byteCount; ++i) {
        if (!drflac__read_uint8(bs, 8, utf8 + i)) {
            *pNumberOut = 0;
            return DRFLAC_AT_END;
        }
        crc = drflac_crc8(crc, utf8[i], 8);

        result = (result << 6) | (utf8[i] & 0x3F);
    }

    *pNumberOut = result;
    *pCRCOut = crc;
    return DRFLAC_SUCCESS;
}


static DRFLAC_INLINE drflac_uint32 drflac__ilog2_u32(drflac_uint32 x)
{
#if 1   /* Needs optimizing. */
    drflac_uint32 result = 0;
    while (x > 0) {
        result += 1;
        x >>= 1;
    }

    return result;
#endif
}

static DRFLAC_INLINE drflac_bool32 drflac__use_64_bit_prediction(drflac_uint32 bitsPerSample, drflac_uint32 order, drflac_uint32 precision)
{
    /* https://web.archive.org/web/20220205005724/https://github.com/ietf-wg-cellar/flac-specification/blob/37a49aa48ba4ba12e8757badfc59c0df35435fec/rfc_backmatter.md */
    return bitsPerSample + precision + drflac__ilog2_u32(order) > 32;
}


/*
The next two functions are responsible for calculating the prediction.

When the bits per sample is >16 we need to use 64-bit integer arithmetic because otherwise we'll run out of precision. It's
safe to assume this will be slower on 32-bit platforms so we use a more optimal solution when the bits per sample is <=16.
*/
#if defined(__clang__)
__attribute__((no_sanitize("signed-integer-overflow")))
#endif
static DRFLAC_INLINE drflac_int32 drflac__calculate_prediction_32(drflac_uint32 order, drflac_int32 shift, const drflac_int32* coefficients, drflac_int32* pDecodedSamples)
{
    drflac_int32 prediction = 0;

    DRFLAC_ASSERT(order <= 32);

    /* 32-bit version. */

    /* VC++ optimizes this to a single jmp. I've not yet verified this for other compilers. */
    switch (order)
    {
    case 32: prediction += coefficients[31] * pDecodedSamples[-32];
    case 31: prediction += coefficients[30] * pDecodedSamples[-31];
    case 30: prediction += coefficients[29] * pDecodedSamples[-30];
    case 29: prediction += coefficients[28] * pDecodedSamples[-29];
    case 28: prediction += coefficients[27] * pDecodedSamples[-28];
    case 27: prediction += coefficients[26] * pDecodedSamples[-27];
    case 26: prediction += coefficients[25] * pDecodedSamples[-26];
    case 25: prediction += coefficients[24] * pDecodedSamples[-25];
    case 24: prediction += coefficients[23] * pDecodedSamples[-24];
    case 23: prediction += coefficients[22] * pDecodedSamples[-23];
    case 22: prediction += coefficients[21] * pDecodedSamples[-22];
    case 21: prediction += coefficients[20] * pDecodedSamples[-21];
    case 20: prediction += coefficients[19] * pDecodedSamples[-20];
    case 19: prediction += coefficients[18] * pDecodedSamples[-19];
    case 18: prediction += coefficients[17] * pDecodedSamples[-18];
    case 17: prediction += coefficients[16] * pDecodedSamples[-17];
    case 16: prediction += coefficients[15] * pDecodedSamples[-16];
    case 15: prediction += coefficients[14] * pDecodedSamples[-15];
    case 14: prediction += coefficients[13] * pDecodedSamples[-14];
    case 13: prediction += coefficients[12] * pDecodedSamples[-13];
    case 12: prediction += coefficients[11] * pDecodedSamples[-12];
    case 11: prediction += coefficients[10] * pDecodedSamples[-11];
    case 10: prediction += coefficients[ 9] * pDecodedSamples[-10];
    case  9: prediction += coefficients[ 8] * pDecodedSamples[- 9];
    case  8: prediction += coefficients[ 7] * pDecodedSamples[- 8];
    case  7: prediction += coefficients[ 6] * pDecodedSamples[- 7];
    case  6: prediction += coefficients[ 5] * pDecodedSamples[- 6];
    case  5: prediction += coefficients[ 4] * pDecodedSamples[- 5];
    case  4: prediction += coefficients[ 3] * pDecodedSamples[- 4];
    case  3: prediction += coefficients[ 2] * pDecodedSamples[- 3];
    case  2: prediction += coefficients[ 1] * pDecodedSamples[- 2];
    case  1: prediction += coefficients[ 0] * pDecodedSamples[- 1];
    }

    return (drflac_int32)(prediction >> shift);
}

static DRFLAC_INLINE drflac_int32 drflac__calculate_prediction_64(drflac_uint32 order, drflac_int32 shift, const drflac_int32* coefficients, drflac_int32* pDecodedSamples)
{
    drflac_int64 prediction;

    DRFLAC_ASSERT(order <= 32);

    /* 64-bit version. */

    /* This method is faster on the 32-bit build when compiling with VC++. See note below. */
#ifndef DRFLAC_64BIT
    if (order == 8)
    {
        prediction  = coefficients[0] * (drflac_int64)pDecodedSamples[-1];
        prediction += coefficients[1] * (drflac_int64)pDecodedSamples[-2];
        prediction += coefficients[2] * (drflac_int64)pDecodedSamples[-3];
        prediction += coefficients[3] * (drflac_int64)pDecodedSamples[-4];
        prediction += coefficients[4] * (drflac_int64)pDecodedSamples[-5];
        prediction += coefficients[5] * (drflac_int64)pDecodedSamples[-6];
        prediction += coefficients[6] * (drflac_int64)pDecodedSamples[-7];
        prediction += coefficients[7] * (drflac_int64)pDecodedSamples[-8];
    }
    else if (order == 7)
    {
        prediction  = coefficients[0] * (drflac_int64)pDecodedSamples[-1];
        prediction += coefficients[1] * (drflac_int64)pDecodedSamples[-2];
        prediction += coefficients[2] * (drflac_int64)pDecodedSamples[-3];
        prediction += coefficients[3] * (drflac_int64)pDecodedSamples[-4];
        prediction += coefficients[4] * (drflac_int64)pDecodedSamples[-5];
        prediction += coefficients[5] * (drflac_int64)pDecodedSamples[-6];
        prediction += coefficients[6] * (drflac_int64)pDecodedSamples[-7];
    }
    else if (order == 3)
    {
        prediction  = coefficients[0] * (drflac_int64)pDecodedSamples[-1];
        prediction += coefficients[1] * (drflac_int64)pDecodedSamples[-2];
        prediction += coefficients[2] * (drflac_int64)pDecodedSamples[-3];
    }
    else if (order == 6)
    {
        prediction  = coefficients[0] * (drflac_int64)pDecodedSamples[-1];
        prediction += coefficients[1] * (drflac_int64)pDecodedSamples[-2];
        prediction += coefficients[2] * (drflac_int64)pDecodedSamples[-3];
        prediction += coefficients[3] * (drflac_int64)pDecodedSamples[-4];
        prediction += coefficients[4] * (drflac_int64)pDecodedSamples[-5];
        prediction += coefficients[5] * (drflac_int64)pDecodedSamples[-6];
    }
    else if (order == 5)
    {
        prediction  = coefficients[0] * (drflac_int64)pDecodedSamples[-1];
        prediction += coefficients[1] * (drflac_int64)pDecodedSamples[-2];
        prediction += coefficients[2] * (drflac_int64)pDecodedSamples[-3];
        prediction += coefficients[3] * (drflac_int64)pDecodedSamples[-4];
        prediction += coefficients[4] * (drflac_int64)pDecodedSamples[-5];
    }
    else if (order == 4)
    {
        prediction  = coefficients[0] * (drflac_int64)pDecodedSamples[-1];
        prediction += coefficients[1] * (drflac_int64)pDecodedSamples[-2];
        prediction += coefficients[2] * (drflac_int64)pDecodedSamples[-3];
        prediction += coefficients[3] * (drflac_int64)pDecodedSamples[-4];
    }
    else if (order == 12)
    {
        prediction  = coefficients[0]  * (drflac_int64)pDecodedSamples[-1];
        prediction += coefficients[1]  * (drflac_int64)pDecodedSamples[-2];
        prediction += coefficients[2]  * (drflac_int64)pDecodedSamples[-3];
        prediction += coefficients[3]  * (drflac_int64)pDecodedSamples[-4];
        prediction += coefficients[4]  * (drflac_int64)pDecodedSamples[-5];
        prediction += coefficients[5]  * (drflac_int64)pDecodedSamples[-6];
        prediction += coefficients[6]  * (drflac_int64)pDecodedSamples[-7];
        prediction += coefficients[7]  * (drflac_int64)pDecodedSamples[-8];
        prediction += coefficients[8]  * (drflac_int64)pDecodedSamples[-9];
        prediction += coefficients[9]  * (drflac_int64)pDecodedSamples[-10];
        prediction += coefficients[10] * (drflac_int64)pDecodedSamples[-11];
        prediction += coefficients[11] * (drflac_int64)pDecodedSamples[-12];
    }
    else if (order == 2)
    {
        prediction  = coefficients[0] * (drflac_int64)pDecodedSamples[-1];
        prediction += coefficients[1] * (drflac_int64)pDecodedSamples[-2];
    }
    else if (order == 1)
    {
        prediction = coefficients[0] * (drflac_int64)pDecodedSamples[-1];
    }
    else if (order == 10)
    {
        prediction  = coefficients[0]  * (drflac_int64)pDecodedSamples[-1];
        prediction += coefficients[1]  * (drflac_int64)pDecodedSamples[-2];
        prediction += coefficients[2]  * (drflac_int64)pDecodedSamples[-3];
        prediction += coefficients[3]  * (drflac_int64)pDecodedSamples[-4];
        prediction += coefficients[4]  * (drflac_int64)pDecodedSamples[-5];
        prediction += coefficients[5]  * (drflac_int64)pDecodedSamples[-6];
        prediction += coefficients[6]  * (drflac_int64)pDecodedSamples[-7];
        prediction += coefficients[7]  * (drflac_int64)pDecodedSamples[-8];
        prediction += coefficients[8]  * (drflac_int64)pDecodedSamples[-9];
        prediction += coefficients[9]  * (drflac_int64)pDecodedSamples[-10];
    }
    else if (order == 9)
    {
        prediction  = coefficients[0]  * (drflac_int64)pDecodedSamples[-1];
        prediction += coefficients[1]  * (drflac_int64)pDecodedSamples[-2];
        prediction += coefficients[2]  * (drflac_int64)pDecodedSamples[-3];
        prediction += coefficients[3]  * (drflac_int64)pDecodedSamples[-4];
        prediction += coefficients[4]  * (drflac_int64)pDecodedSamples[-5];
        prediction += coefficients[5]  * (drflac_int64)pDecodedSamples[-6];
        prediction += coefficients[6]  * (drflac_int64)pDecodedSamples[-7];
        prediction += coefficients[7]  * (drflac_int64)pDecodedSamples[-8];
        prediction += coefficients[8]  * (drflac_int64)pDecodedSamples[-9];
    }
    else if (order == 11)
    {
        prediction  = coefficients[0]  * (drflac_int64)pDecodedSamples[-1];
        prediction += coefficients[1]  * (drflac_int64)pDecodedSamples[-2];
        prediction += coefficients[2]  * (drflac_int64)pDecodedSamples[-3];
        prediction += coefficients[3]  * (drflac_int64)pDecodedSamples[-4];
        prediction += coefficients[4]  * (drflac_int64)pDecodedSamples[-5];
        prediction += coefficients[5]  * (drflac_int64)pDecodedSamples[-6];
        prediction += coefficients[6]  * (drflac_int64)pDecodedSamples[-7];
        prediction += coefficients[7]  * (drflac_int64)pDecodedSamples[-8];
        prediction += coefficients[8]  * (drflac_int64)pDecodedSamples[-9];
        prediction += coefficients[9]  * (drflac_int64)pDecodedSamples[-10];
        prediction += coefficients[10] * (drflac_int64)pDecodedSamples[-11];
    }
    else
    {
        int j;

        prediction = 0;
        for (j = 0; j < (int)order; ++j) {
            prediction += coefficients[j] * (drflac_int64)pDecodedSamples[-j-1];
        }
    }
#endif

    /*
    VC++ optimizes this to a single jmp instruction, but only the 64-bit build. The 32-bit build generates less efficient code for some
    reason. The ugly version above is faster so we'll just switch between the two depending on the target platform.
    */
#ifdef DRFLAC_64BIT
    prediction = 0;
    switch (order)
    {
    case 32: prediction += coefficients[31] * (drflac_int64)pDecodedSamples[-32];
    case 31: prediction += coefficients[30] * (drflac_int64)pDecodedSamples[-31];
    case 30: prediction += coefficients[29] * (drflac_int64)pDecodedSamples[-30];
    case 29: prediction += coefficients[28] * (drflac_int64)pDecodedSamples[-29];
    case 28: prediction += coefficients[27] * (drflac_int64)pDecodedSamples[-28];
    case 27: prediction += coefficients[26] * (drflac_int64)pDecodedSamples[-27];
    case 26: prediction += coefficients[25] * (drflac_int64)pDecodedSamples[-26];
    case 25: prediction += coefficients[24] * (drflac_int64)pDecodedSamples[-25];
    case 24: prediction += coefficients[23] * (drflac_int64)pDecodedSamples[-24];
    case 23: prediction += coefficients[22] * (drflac_int64)pDecodedSamples[-23];
    case 22: prediction += coefficients[21] * (drflac_int64)pDecodedSamples[-22];
    case 21: prediction += coefficients[20] * (drflac_int64)pDecodedSamples[-21];
    case 20: prediction += coefficients[19] * (drflac_int64)pDecodedSamples[-20];
    case 19: prediction += coefficients[18] * (drflac_int64)pDecodedSamples[-19];
    case 18: prediction += coefficients[17] * (drflac_int64)pDecodedSamples[-18];
    case 17: prediction += coefficients[16] * (drflac_int64)pDecodedSamples[-17];
    case 16: prediction += coefficients[15] * (drflac_int64)pDecodedSamples[-16];
    case 15: prediction += coefficients[14] * (drflac_int64)pDecodedSamples[-15];
    case 14: prediction += coefficients[13] * (drflac_int64)pDecodedSamples[-14];
    case 13: prediction += coefficients[12] * (drflac_int64)pDecodedSamples[-13];
    case 12: prediction += coefficients[11] * (drflac_int64)pDecodedSamples[-12];
    case 11: prediction += coefficients[10] * (drflac_int64)pDecodedSamples[-11];
    case 10: prediction += coefficients[ 9] * (drflac_int64)pDecodedSamples[-10];
    case  9: prediction += coefficients[ 8] * (drflac_int64)pDecodedSamples[- 9];
    case  8: prediction += coefficients[ 7] * (drflac_int64)pDecodedSamples[- 8];
    case  7: prediction += coefficients[ 6] * (drflac_int64)pDecodedSamples[- 7];
    case  6: prediction += coefficients[ 5] * (drflac_int64)pDecodedSamples[- 6];
    case  5: prediction += coefficients[ 4] * (drflac_int64)pDecodedSamples[- 5];
    case  4: prediction += coefficients[ 3] * (drflac_int64)pDecodedSamples[- 4];
    case  3: prediction += coefficients[ 2] * (drflac_int64)pDecodedSamples[- 3];
    case  2: prediction += coefficients[ 1] * (drflac_int64)pDecodedSamples[- 2];
    case  1: prediction += coefficients[ 0] * (drflac_int64)pDecodedSamples[- 1];
    }
#endif

    return (drflac_int32)(prediction >> shift);
}


#if 0
/*
Reference implementation for reading and decoding samples with residual. This is intentionally left unoptimized for the
sake of readability and should only be used as a reference.
*/
static drflac_bool32 drflac__decode_samples_with_residual__rice__reference(drflac_bs* bs, drflac_uint32 bitsPerSample, drflac_uint32 count, drflac_uint8 riceParam, drflac_uint32 lpcOrder, drflac_int32 lpcShift, drflac_uint32 lpcPrecision, const drflac_int32* coefficients, drflac_int32* pSamplesOut)
{
    drflac_uint32 i;

    DRFLAC_ASSERT(bs != NULL);
    DRFLAC_ASSERT(pSamplesOut != NULL);

    for (i = 0; i < count; ++i) {
        drflac_uint32 zeroCounter = 0;
        for (;;) {
            drflac_uint8 bit;
            if (!drflac__read_uint8(bs, 1, &bit)) {
                return DRFLAC_FALSE;
            }

            if (bit == 0) {
                zeroCounter += 1;
            } else {
                break;
            }
        }

        drflac_uint32 decodedRice;
        if (riceParam > 0) {
            if (!drflac__read_uint32(bs, riceParam, &decodedRice)) {
                return DRFLAC_FALSE;
            }
        } else {
            decodedRice = 0;
        }

        decodedRice |= (zeroCounter << riceParam);
        if ((decodedRice & 0x01)) {
            decodedRice = ~(decodedRice >> 1);
        } else {
            decodedRice =  (decodedRice >> 1);
        }


        if (drflac__use_64_bit_prediction(bitsPerSample, lpcOrder, lpcPrecision)) {
            pSamplesOut[i] = decodedRice + drflac__calculate_prediction_64(lpcOrder, lpcShift, coefficients, pSamplesOut + i);
        } else {
            pSamplesOut[i] = decodedRice + drflac__calculate_prediction_32(lpcOrder, lpcShift, coefficients, pSamplesOut + i);
        }
    }

    return DRFLAC_TRUE;
}
#endif

#if 0
static drflac_bool32 drflac__read_rice_parts__reference(drflac_bs* bs, drflac_uint8 riceParam, drflac_uint32* pZeroCounterOut, drflac_uint32* pRiceParamPartOut)
{
    drflac_uint32 zeroCounter = 0;
    drflac_uint32 decodedRice;

    for (;;) {
        drflac_uint8 bit;
        if (!drflac__read_uint8(bs, 1, &bit)) {
            return DRFLAC_FALSE;
        }

        if (bit == 0) {
            zeroCounter += 1;
        } else {
            break;
        }
    }

    if (riceParam > 0) {
        if (!drflac__read_uint32(bs, riceParam, &decodedRice)) {
            return DRFLAC_FALSE;
        }
    } else {
        decodedRice = 0;
    }

    *pZeroCounterOut = zeroCounter;
    *pRiceParamPartOut = decodedRice;
    return DRFLAC_TRUE;
}
#endif

#if 0
static DRFLAC_INLINE drflac_bool32 drflac__read_rice_parts(drflac_bs* bs, drflac_uint8 riceParam, drflac_uint32* pZeroCounterOut, drflac_uint32* pRiceParamPartOut)
{
    drflac_cache_t riceParamMask;
    drflac_uint32 zeroCounter;
    drflac_uint32 setBitOffsetPlus1;
    drflac_uint32 riceParamPart;
    drflac_uint32 riceLength;

    DRFLAC_ASSERT(riceParam > 0);   /* <-- riceParam should never be 0. drflac__read_rice_parts__param_equals_zero() should be used instead for this case. */

    riceParamMask = DRFLAC_CACHE_L1_SELECTION_MASK(riceParam);

    zeroCounter = 0;
    while (bs->cache == 0) {
        zeroCounter += (drflac_uint32)DRFLAC_CACHE_L1_BITS_REMAINING(bs);
        if (!drflac__reload_cache(bs)) {
            return DRFLAC_FALSE;
        }
    }

    setBitOffsetPlus1 = drflac__clz(bs->cache);
    zeroCounter += setBitOffsetPlus1;
    setBitOffsetPlus1 += 1;

    riceLength = setBitOffsetPlus1 + riceParam;
    if (riceLength < DRFLAC_CACHE_L1_BITS_REMAINING(bs)) {
        riceParamPart = (drflac_uint32)((bs->cache & (riceParamMask >> setBitOffsetPlus1)) >> DRFLAC_CACHE_L1_SELECTION_SHIFT(bs, riceLength));

        bs->consumedBits += riceLength;
        bs->cache <<= riceLength;
    } else {
        drflac_uint32 bitCountLo;
        drflac_cache_t resultHi;

        bs->consumedBits += riceLength;
        bs->cache <<= setBitOffsetPlus1 & (DRFLAC_CACHE_L1_SIZE_BITS(bs)-1);    /* <-- Equivalent to "if (setBitOffsetPlus1 < DRFLAC_CACHE_L1_SIZE_BITS(bs)) { bs->cache <<= setBitOffsetPlus1; }" */

        /* It straddles the cached data. It will never cover more than the next chunk. We just read the number in two parts and combine them. */
        bitCountLo = bs->consumedBits - DRFLAC_CACHE_L1_SIZE_BITS(bs);
        resultHi = DRFLAC_CACHE_L1_SELECT_AND_SHIFT(bs, riceParam);  /* <-- Use DRFLAC_CACHE_L1_SELECT_AND_SHIFT_SAFE() if ever this function allows riceParam=0. */

        if (bs->nextL2Line < DRFLAC_CACHE_L2_LINE_COUNT(bs)) {
#ifndef DR_FLAC_NO_CRC
            drflac__update_crc16(bs);
#endif
            bs->cache = drflac__be2host__cache_line(bs->cacheL2[bs->nextL2Line++]);
            bs->consumedBits = 0;
#ifndef DR_FLAC_NO_CRC
            bs->crc16Cache = bs->cache;
#endif
        } else {
            /* Slow path. We need to fetch more data from the client. */
            if (!drflac__reload_cache(bs)) {
                return DRFLAC_FALSE;
            }
            if (bitCountLo > DRFLAC_CACHE_L1_BITS_REMAINING(bs)) {
                /* This happens when we get to end of stream */
                return DRFLAC_FALSE;
            }
        }

        riceParamPart = (drflac_uint32)(resultHi | DRFLAC_CACHE_L1_SELECT_AND_SHIFT_SAFE(bs, bitCountLo));

        bs->consumedBits += bitCountLo;
        bs->cache <<= bitCountLo;
    }

    pZeroCounterOut[0] = zeroCounter;
    pRiceParamPartOut[0] = riceParamPart;

    return DRFLAC_TRUE;
}
#endif

static DRFLAC_INLINE drflac_bool32 drflac__read_rice_parts_x1(drflac_bs* bs, drflac_uint8 riceParam, drflac_uint32* pZeroCounterOut, drflac_uint32* pRiceParamPartOut)
{
    drflac_uint32  riceParamPlus1 = riceParam + 1;
    /*drflac_cache_t riceParamPlus1Mask  = DRFLAC_CACHE_L1_SELECTION_MASK(riceParamPlus1);*/
    drflac_uint32  riceParamPlus1Shift = DRFLAC_CACHE_L1_SELECTION_SHIFT(bs, riceParamPlus1);
    drflac_uint32  riceParamPlus1MaxConsumedBits = DRFLAC_CACHE_L1_SIZE_BITS(bs) - riceParamPlus1;

    /*
    The idea here is to use local variables for the cache in an attempt to encourage the compiler to store them in registers. I have
    no idea how this will work in practice...
    */
    drflac_cache_t bs_cache = bs->cache;
    drflac_uint32  bs_consumedBits = bs->consumedBits;

    /* The first thing to do is find the first unset bit. Most likely a bit will be set in the current cache line. */
    drflac_uint32  lzcount = drflac__clz(bs_cache);
    if (lzcount < sizeof(bs_cache)*8) {
        pZeroCounterOut[0] = lzcount;

        /*
        It is most likely that the riceParam part (which comes after the zero counter) is also on this cache line. When extracting
        this, we include the set bit from the unary coded part because it simplifies cache management. This bit will be handled
        outside of this function at a higher level.
        */
    extract_rice_param_part:
        bs_cache       <<= lzcount;
        bs_consumedBits += lzcount;

        if (bs_consumedBits <= riceParamPlus1MaxConsumedBits) {
            /* Getting here means the rice parameter part is wholly contained within the current cache line. */
            pRiceParamPartOut[0] = (drflac_uint32)(bs_cache >> riceParamPlus1Shift);
            bs_cache       <<= riceParamPlus1;
            bs_consumedBits += riceParamPlus1;
        } else {
            drflac_uint32 riceParamPartHi;
            drflac_uint32 riceParamPartLo;
            drflac_uint32 riceParamPartLoBitCount;

            /*
            Getting here means the rice parameter part straddles the cache line. We need to read from the tail of the current cache
            line, reload the cache, and then combine it with the head of the next cache line.
            */

            /* Grab the high part of the rice parameter part. */
            riceParamPartHi = (drflac_uint32)(bs_cache >> riceParamPlus1Shift);

            /* Before reloading the cache we need to grab the size in bits of the low part. */
            riceParamPartLoBitCount = bs_consumedBits - riceParamPlus1MaxConsumedBits;
            DRFLAC_ASSERT(riceParamPartLoBitCount > 0 && riceParamPartLoBitCount < 32);

            /* Now reload the cache. */
            if (bs->nextL2Line < DRFLAC_CACHE_L2_LINE_COUNT(bs)) {
            #ifndef DR_FLAC_NO_CRC
                drflac__update_crc16(bs);
            #endif
                bs_cache = drflac__be2host__cache_line(bs->cacheL2[bs->nextL2Line++]);
                bs_consumedBits = riceParamPartLoBitCount;
            #ifndef DR_FLAC_NO_CRC
                bs->crc16Cache = bs_cache;
            #endif
            } else {
                /* Slow path. We need to fetch more data from the client. */
                if (!drflac__reload_cache(bs)) {
                    return DRFLAC_FALSE;
                }
                if (riceParamPartLoBitCount > DRFLAC_CACHE_L1_BITS_REMAINING(bs)) {
                    /* This happens when we get to end of stream */
                    return DRFLAC_FALSE;
                }

                bs_cache = bs->cache;
                bs_consumedBits = bs->consumedBits + riceParamPartLoBitCount;
            }

            /* We should now have enough information to construct the rice parameter part. */
            riceParamPartLo = (drflac_uint32)(bs_cache >> (DRFLAC_CACHE_L1_SELECTION_SHIFT(bs, riceParamPartLoBitCount)));
            pRiceParamPartOut[0] = riceParamPartHi | riceParamPartLo;

            bs_cache <<= riceParamPartLoBitCount;
        }
    } else {
        /*
        Getting here means there are no bits set on the cache line. This is a less optimal case because we just wasted a call
        to drflac__clz() and we need to reload the cache.
        */
        drflac_uint32 zeroCounter = (drflac_uint32)(DRFLAC_CACHE_L1_SIZE_BITS(bs) - bs_consumedBits);
        for (;;) {
            if (bs->nextL2Line < DRFLAC_CACHE_L2_LINE_COUNT(bs)) {
            #ifndef DR_FLAC_NO_CRC
                drflac__update_crc16(bs);
            #endif
                bs_cache = drflac__be2host__cache_line(bs->cacheL2[bs->nextL2Line++]);
                bs_consumedBits = 0;
            #ifndef DR_FLAC_NO_CRC
                bs->crc16Cache = bs_cache;
            #endif
            } else {
                /* Slow path. We need to fetch more data from the client. */
                if (!drflac__reload_cache(bs)) {
                    return DRFLAC_FALSE;
                }

                bs_cache = bs->cache;
                bs_consumedBits = bs->consumedBits;
            }

            lzcount = drflac__clz(bs_cache);
            zeroCounter += lzcount;

            if (lzcount < sizeof(bs_cache)*8) {
                break;
            }
        }

        pZeroCounterOut[0] = zeroCounter;
        goto extract_rice_param_part;
    }

    /* Make sure the cache is restored at the end of it all. */
    bs->cache = bs_cache;
    bs->consumedBits = bs_consumedBits;

    return DRFLAC_TRUE;
}

static DRFLAC_INLINE drflac_bool32 drflac__seek_rice_parts(drflac_bs* bs, drflac_uint8 riceParam)
{
    drflac_uint32  riceParamPlus1 = riceParam + 1;
    drflac_uint32  riceParamPlus1MaxConsumedBits = DRFLAC_CACHE_L1_SIZE_BITS(bs) - riceParamPlus1;

    /*
    The idea here is to use local variables for the cache in an attempt to encourage the compiler to store them in registers. I have
    no idea how this will work in practice...
    */
    drflac_cache_t bs_cache = bs->cache;
    drflac_uint32  bs_consumedBits = bs->consumedBits;

    /* The first thing to do is find the first unset bit. Most likely a bit will be set in the current cache line. */
    drflac_uint32  lzcount = drflac__clz(bs_cache);
    if (lzcount < sizeof(bs_cache)*8) {
        /*
        It is most likely that the riceParam part (which comes after the zero counter) is also on this cache line. When extracting
        this, we include the set bit from the unary coded part because it simplifies cache management. This bit will be handled
        outside of this function at a higher level.
        */
    extract_rice_param_part:
        bs_cache       <<= lzcount;
        bs_consumedBits += lzcount;

        if (bs_consumedBits <= riceParamPlus1MaxConsumedBits) {
            /* Getting here means the rice parameter part is wholly contained within the current cache line. */
            bs_cache       <<= riceParamPlus1;
            bs_consumedBits += riceParamPlus1;
        } else {
            /*
            Getting here means the rice parameter part straddles the cache line. We need to read from the tail of the current cache
            line, reload the cache, and then combine it with the head of the next cache line.
            */

            /* Before reloading the cache we need to grab the size in bits of the low part. */
            drflac_uint32 riceParamPartLoBitCount = bs_consumedBits - riceParamPlus1MaxConsumedBits;
            DRFLAC_ASSERT(riceParamPartLoBitCount > 0 && riceParamPartLoBitCount < 32);

            /* Now reload the cache. */
            if (bs->nextL2Line < DRFLAC_CACHE_L2_LINE_COUNT(bs)) {
            #ifndef DR_FLAC_NO_CRC
                drflac__update_crc16(bs);
            #endif
                bs_cache = drflac__be2host__cache_line(bs->cacheL2[bs->nextL2Line++]);
                bs_consumedBits = riceParamPartLoBitCount;
            #ifndef DR_FLAC_NO_CRC
                bs->crc16Cache = bs_cache;
            #endif
            } else {
                /* Slow path. We need to fetch more data from the client. */
                if (!drflac__reload_cache(bs)) {
                    return DRFLAC_FALSE;
                }

                if (riceParamPartLoBitCount > DRFLAC_CACHE_L1_BITS_REMAINING(bs)) {
                    /* This happens when we get to end of stream */
                    return DRFLAC_FALSE;
                }

                bs_cache = bs->cache;
                bs_consumedBits = bs->consumedBits + riceParamPartLoBitCount;
            }

            bs_cache <<= riceParamPartLoBitCount;
        }
    } else {
        /*
        Getting here means there are no bits set on the cache line. This is a less optimal case because we just wasted a call
        to drflac__clz() and we need to reload the cache.
        */
        for (;;) {
            if (bs->nextL2Line < DRFLAC_CACHE_L2_LINE_COUNT(bs)) {
            #ifndef DR_FLAC_NO_CRC
                drflac__update_crc16(bs);
            #endif
                bs_cache = drflac__be2host__cache_line(bs->cacheL2[bs->nextL2Line++]);
                bs_consumedBits = 0;
            #ifndef DR_FLAC_NO_CRC
                bs->crc16Cache = bs_cache;
            #endif
            } else {
                /* Slow path. We need to fetch more data from the client. */
                if (!drflac__reload_cache(bs)) {
                    return DRFLAC_FALSE;
                }

                bs_cache = bs->cache;
                bs_consumedBits = bs->consumedBits;
            }

            lzcount = drflac__clz(bs_cache);
            if (lzcount < sizeof(bs_cache)*8) {
                break;
            }
        }

        goto extract_rice_param_part;
    }

    /* Make sure the cache is restored at the end of it all. */
    bs->cache = bs_cache;
    bs->consumedBits = bs_consumedBits;

    return DRFLAC_TRUE;
}


static drflac_bool32 drflac__decode_samples_with_residual__rice__scalar_zeroorder(drflac_bs* bs, drflac_uint32 bitsPerSample, drflac_uint32 count, drflac_uint8 riceParam, drflac_uint32 order, drflac_int32 shift, const drflac_int32* coefficients, drflac_int32* pSamplesOut)
{
    drflac_uint32 t[2] = {0x00000000, 0xFFFFFFFF};
    drflac_uint32 zeroCountPart0;
    drflac_uint32 riceParamPart0;
    drflac_uint32 riceParamMask;
    drflac_uint32 i;

    DRFLAC_ASSERT(bs != NULL);
    DRFLAC_ASSERT(pSamplesOut != NULL);

    (void)bitsPerSample;
    (void)order;
    (void)shift;
    (void)coefficients;

    riceParamMask  = (drflac_uint32)~((~0UL) << riceParam);

    i = 0;
    while (i < count) {
        /* Rice extraction. */
        if (!drflac__read_rice_parts_x1(bs, riceParam, &zeroCountPart0, &riceParamPart0)) {
            return DRFLAC_FALSE;
        }

        /* Rice reconstruction. */
        riceParamPart0 &= riceParamMask;
        riceParamPart0 |= (zeroCountPart0 << riceParam);
        riceParamPart0  = (riceParamPart0 >> 1) ^ t[riceParamPart0 & 0x01];

        pSamplesOut[i] = riceParamPart0;

        i += 1;
    }

    return DRFLAC_TRUE;
}

static drflac_bool32 drflac__decode_samples_with_residual__rice__scalar(drflac_bs* bs, drflac_uint32 bitsPerSample, drflac_uint32 count, drflac_uint8 riceParam, drflac_uint32 lpcOrder, drflac_int32 lpcShift, drflac_uint32 lpcPrecision, const drflac_int32* coefficients, drflac_int32* pSamplesOut)
{
    drflac_uint32 t[2] = {0x00000000, 0xFFFFFFFF};
    drflac_uint32 zeroCountPart0 = 0;
    drflac_uint32 zeroCountPart1 = 0;
    drflac_uint32 zeroCountPart2 = 0;
    drflac_uint32 zeroCountPart3 = 0;
    drflac_uint32 riceParamPart0 = 0;
    drflac_uint32 riceParamPart1 = 0;
    drflac_uint32 riceParamPart2 = 0;
    drflac_uint32 riceParamPart3 = 0;
    drflac_uint32 riceParamMask;
    const drflac_int32* pSamplesOutEnd;
    drflac_uint32 i;

    DRFLAC_ASSERT(bs != NULL);
    DRFLAC_ASSERT(pSamplesOut != NULL);

    if (lpcOrder == 0) {
        return drflac__decode_samples_with_residual__rice__scalar_zeroorder(bs, bitsPerSample, count, riceParam, lpcOrder, lpcShift, coefficients, pSamplesOut);
    }

    riceParamMask  = (drflac_uint32)~((~0UL) << riceParam);
    pSamplesOutEnd = pSamplesOut + (count & ~3);

    if (drflac__use_64_bit_prediction(bitsPerSample, lpcOrder, lpcPrecision)) {
        while (pSamplesOut < pSamplesOutEnd) {
            /*
            Rice extraction. It's faster to do this one at a time against local variables than it is to use the x4 version
            against an array. Not sure why, but perhaps it's making more efficient use of registers?
            */
            if (!drflac__read_rice_parts_x1(bs, riceParam, &zeroCountPart0, &riceParamPart0) ||
                !drflac__read_rice_parts_x1(bs, riceParam, &zeroCountPart1, &riceParamPart1) ||
                !drflac__read_rice_parts_x1(bs, riceParam, &zeroCountPart2, &riceParamPart2) ||
                !drflac__read_rice_parts_x1(bs, riceParam, &zeroCountPart3, &riceParamPart3)) {
                return DRFLAC_FALSE;
            }

            riceParamPart0 &= riceParamMask;
            riceParamPart1 &= riceParamMask;
            riceParamPart2 &= riceParamMask;
            riceParamPart3 &= riceParamMask;

            riceParamPart0 |= (zeroCountPart0 << riceParam);
            riceParamPart1 |= (zeroCountPart1 << riceParam);
            riceParamPart2 |= (zeroCountPart2 << riceParam);
            riceParamPart3 |= (zeroCountPart3 << riceParam);

            riceParamPart0  = (riceParamPart0 >> 1) ^ t[riceParamPart0 & 0x01];
            riceParamPart1  = (riceParamPart1 >> 1) ^ t[riceParamPart1 & 0x01];
            riceParamPart2  = (riceParamPart2 >> 1) ^ t[riceParamPart2 & 0x01];
            riceParamPart3  = (riceParamPart3 >> 1) ^ t[riceParamPart3 & 0x01];

            pSamplesOut[0] = riceParamPart0 + drflac__calculate_prediction_64(lpcOrder, lpcShift, coefficients, pSamplesOut + 0);
            pSamplesOut[1] = riceParamPart1 + drflac__calculate_prediction_64(lpcOrder, lpcShift, coefficients, pSamplesOut + 1);
            pSamplesOut[2] = riceParamPart2 + drflac__calculate_prediction_64(lpcOrder, lpcShift, coefficients, pSamplesOut + 2);
            pSamplesOut[3] = riceParamPart3 + drflac__calculate_prediction_64(lpcOrder, lpcShift, coefficients, pSamplesOut + 3);

            pSamplesOut += 4;
        }
    } else {
        while (pSamplesOut < pSamplesOutEnd) {
            if (!drflac__read_rice_parts_x1(bs, riceParam, &zeroCountPart0, &riceParamPart0) ||
                !drflac__read_rice_parts_x1(bs, riceParam, &zeroCountPart1, &riceParamPart1) ||
                !drflac__read_rice_parts_x1(bs, riceParam, &zeroCountPart2, &riceParamPart2) ||
                !drflac__read_rice_parts_x1(bs, riceParam, &zeroCountPart3, &riceParamPart3)) {
                return DRFLAC_FALSE;
            }

            riceParamPart0 &= riceParamMask;
            riceParamPart1 &= riceParamMask;
            riceParamPart2 &= riceParamMask;
            riceParamPart3 &= riceParamMask;

            riceParamPart0 |= (zeroCountPart0 << riceParam);
            riceParamPart1 |= (zeroCountPart1 << riceParam);
            riceParamPart2 |= (zeroCountPart2 << riceParam);
            riceParamPart3 |= (zeroCountPart3 << riceParam);

            riceParamPart0  = (riceParamPart0 >> 1) ^ t[riceParamPart0 & 0x01];
            riceParamPart1  = (riceParamPart1 >> 1) ^ t[riceParamPart1 & 0x01];
            riceParamPart2  = (riceParamPart2 >> 1) ^ t[riceParamPart2 & 0x01];
            riceParamPart3  = (riceParamPart3 >> 1) ^ t[riceParamPart3 & 0x01];

            pSamplesOut[0] = riceParamPart0 + drflac__calculate_prediction_32(lpcOrder, lpcShift, coefficients, pSamplesOut + 0);
            pSamplesOut[1] = riceParamPart1 + drflac__calculate_prediction_32(lpcOrder, lpcShift, coefficients, pSamplesOut + 1);
            pSamplesOut[2] = riceParamPart2 + drflac__calculate_prediction_32(lpcOrder, lpcShift, coefficients, pSamplesOut + 2);
            pSamplesOut[3] = riceParamPart3 + drflac__calculate_prediction_32(lpcOrder, lpcShift, coefficients, pSamplesOut + 3);

            pSamplesOut += 4;
        }
    }

    i = (count & ~3);
    while (i < count) {
        /* Rice extraction. */
        if (!drflac__read_rice_parts_x1(bs, riceParam, &zeroCountPart0, &riceParamPart0)) {
            return DRFLAC_FALSE;
        }

        /* Rice reconstruction. */
        riceParamPart0 &= riceParamMask;
        riceParamPart0 |= (zeroCountPart0 << riceParam);
        riceParamPart0  = (riceParamPart0 >> 1) ^ t[riceParamPart0 & 0x01];
        /*riceParamPart0  = (riceParamPart0 >> 1) ^ (~(riceParamPart0 & 0x01) + 1);*/

        /* Sample reconstruction. */
        if (drflac__use_64_bit_prediction(bitsPerSample, lpcOrder, lpcPrecision)) {
            pSamplesOut[0] = riceParamPart0 + drflac__calculate_prediction_64(lpcOrder, lpcShift, coefficients, pSamplesOut + 0);
        } else {
            pSamplesOut[0] = riceParamPart0 + drflac__calculate_prediction_32(lpcOrder, lpcShift, coefficients, pSamplesOut + 0);
        }

        i += 1;
        pSamplesOut += 1;
    }

    return DRFLAC_TRUE;
}

#if defined(DRFLAC_SUPPORT_SSE2)
static DRFLAC_INLINE __m128i drflac__mm_packs_interleaved_epi32(__m128i a, __m128i b)
{
    __m128i r;

    /* Pack. */
    r = _mm_packs_epi32(a, b);

    /* a3a2 a1a0 b3b2 b1b0 -> a3a2 b3b2 a1a0 b1b0 */
    r = _mm_shuffle_epi32(r, _MM_SHUFFLE(3, 1, 2, 0));

    /* a3a2 b3b2 a1a0 b1b0 -> a3b3 a2b2 a1b1 a0b0 */
    r = _mm_shufflehi_epi16(r, _MM_SHUFFLE(3, 1, 2, 0));
    r = _mm_shufflelo_epi16(r, _MM_SHUFFLE(3, 1, 2, 0));

    return r;
}
#endif

#if defined(DRFLAC_SUPPORT_SSE41)
static DRFLAC_INLINE __m128i drflac__mm_not_si128(__m128i a)
{
    return _mm_xor_si128(a, _mm_cmpeq_epi32(_mm_setzero_si128(), _mm_setzero_si128()));
}

static DRFLAC_INLINE __m128i drflac__mm_hadd_epi32(__m128i x)
{
    __m128i x64 = _mm_add_epi32(x, _mm_shuffle_epi32(x, _MM_SHUFFLE(1, 0, 3, 2)));
    __m128i x32 = _mm_shufflelo_epi16(x64, _MM_SHUFFLE(1, 0, 3, 2));
    return _mm_add_epi32(x64, x32);
}

static DRFLAC_INLINE __m128i drflac__mm_hadd_epi64(__m128i x)
{
    return _mm_add_epi64(x, _mm_shuffle_epi32(x, _MM_SHUFFLE(1, 0, 3, 2)));
}

static DRFLAC_INLINE __m128i drflac__mm_srai_epi64(__m128i x, int count)
{
    /*
    To simplify this we are assuming count < 32. This restriction allows us to work on a low side and a high side. The low side
    is shifted with zero bits, whereas the right side is shifted with sign bits.
    */
    __m128i lo = _mm_srli_epi64(x, count);
    __m128i hi = _mm_srai_epi32(x, count);

    hi = _mm_and_si128(hi, _mm_set_epi32(0xFFFFFFFF, 0, 0xFFFFFFFF, 0));    /* The high part needs to have the low part cleared. */

    return _mm_or_si128(lo, hi);
}

static drflac_bool32 drflac__decode_samples_with_residual__rice__sse41_32(drflac_bs* bs, drflac_uint32 count, drflac_uint8 riceParam, drflac_uint32 order, drflac_int32 shift, const drflac_int32* coefficients, drflac_int32* pSamplesOut)
{
    int i;
    drflac_uint32 riceParamMask;
    drflac_int32* pDecodedSamples    = pSamplesOut;
    drflac_int32* pDecodedSamplesEnd = pSamplesOut + (count & ~3);
    drflac_uint32 zeroCountParts0 = 0;
    drflac_uint32 zeroCountParts1 = 0;
    drflac_uint32 zeroCountParts2 = 0;
    drflac_uint32 zeroCountParts3 = 0;
    drflac_uint32 riceParamParts0 = 0;
    drflac_uint32 riceParamParts1 = 0;
    drflac_uint32 riceParamParts2 = 0;
    drflac_uint32 riceParamParts3 = 0;
    __m128i coefficients128_0;
    __m128i coefficients128_4;
    __m128i coefficients128_8;
    __m128i samples128_0;
    __m128i samples128_4;
    __m128i samples128_8;
    __m128i riceParamMask128;

    const drflac_uint32 t[2] = {0x00000000, 0xFFFFFFFF};

    riceParamMask    = (drflac_uint32)~((~0UL) << riceParam);
    riceParamMask128 = _mm_set1_epi32(riceParamMask);

    /* Pre-load. */
    coefficients128_0 = _mm_setzero_si128();
    coefficients128_4 = _mm_setzero_si128();
    coefficients128_8 = _mm_setzero_si128();

    samples128_0 = _mm_setzero_si128();
    samples128_4 = _mm_setzero_si128();
    samples128_8 = _mm_setzero_si128();

    /*
    Pre-loading the coefficients and prior samples is annoying because we need to ensure we don't try reading more than
    what's available in the input buffers. It would be convenient to use a fall-through switch to do this, but this results
    in strict aliasing warnings with GCC. To work around this I'm just doing something hacky. This feels a bit convoluted
    so I think there's opportunity for this to be simplified.
    */
#if 1
    {
        int runningOrder = order;

        /* 0 - 3. */
        if (runningOrder >= 4) {
            coefficients128_0 = _mm_loadu_si128((const __m128i*)(coefficients + 0));
            samples128_0      = _mm_loadu_si128((const __m128i*)(pSamplesOut  - 4));
            runningOrder -= 4;
        } else {
            switch (runningOrder) {
                case 3: coefficients128_0 = _mm_set_epi32(0, coefficients[2], coefficients[1], coefficients[0]); samples128_0 = _mm_set_epi32(pSamplesOut[-1], pSamplesOut[-2], pSamplesOut[-3], 0); break;
                case 2: coefficients128_0 = _mm_set_epi32(0, 0,               coefficients[1], coefficients[0]); samples128_0 = _mm_set_epi32(pSamplesOut[-1], pSamplesOut[-2], 0,               0); break;
                case 1: coefficients128_0 = _mm_set_epi32(0, 0,               0,               coefficients[0]); samples128_0 = _mm_set_epi32(pSamplesOut[-1], 0,               0,               0); break;
            }
            runningOrder = 0;
        }

        /* 4 - 7 */
        if (runningOrder >= 4) {
            coefficients128_4 = _mm_loadu_si128((const __m128i*)(coefficients + 4));
            samples128_4      = _mm_loadu_si128((const __m128i*)(pSamplesOut  - 8));
            runningOrder -= 4;
        } else {
            switch (runningOrder) {
                case 3: coefficients128_4 = _mm_set_epi32(0, coefficients[6], coefficients[5], coefficients[4]); samples128_4 = _mm_set_epi32(pSamplesOut[-5], pSamplesOut[-6], pSamplesOut[-7], 0); break;
                case 2: coefficients128_4 = _mm_set_epi32(0, 0,               coefficients[5], coefficients[4]); samples128_4 = _mm_set_epi32(pSamplesOut[-5], pSamplesOut[-6], 0,               0); break;
                case 1: coefficients128_4 = _mm_set_epi32(0, 0,               0,               coefficients[4]); samples128_4 = _mm_set_epi32(pSamplesOut[-5], 0,               0,               0); break;
            }
            runningOrder = 0;
        }

        /* 8 - 11 */
        if (runningOrder == 4) {
            coefficients128_8 = _mm_loadu_si128((const __m128i*)(coefficients + 8));
            samples128_8      = _mm_loadu_si128((const __m128i*)(pSamplesOut  - 12));
            runningOrder -= 4;
        } else {
            switch (runningOrder) {
                case 3: coefficients128_8 = _mm_set_epi32(0, coefficients[10], coefficients[9], coefficients[8]); samples128_8 = _mm_set_epi32(pSamplesOut[-9], pSamplesOut[-10], pSamplesOut[-11], 0); break;
                case 2: coefficients128_8 = _mm_set_epi32(0, 0,                coefficients[9], coefficients[8]); samples128_8 = _mm_set_epi32(pSamplesOut[-9], pSamplesOut[-10], 0,                0); break;
                case 1: coefficients128_8 = _mm_set_epi32(0, 0,                0,               coefficients[8]); samples128_8 = _mm_set_epi32(pSamplesOut[-9], 0,                0,                0); break;
            }
            runningOrder = 0;
        }

        /* Coefficients need to be shuffled for our streaming algorithm below to work. Samples are already in the correct order from the loading routine above. */
        coefficients128_0 = _mm_shuffle_epi32(coefficients128_0, _MM_SHUFFLE(0, 1, 2, 3));
        coefficients128_4 = _mm_shuffle_epi32(coefficients128_4, _MM_SHUFFLE(0, 1, 2, 3));
        coefficients128_8 = _mm_shuffle_epi32(coefficients128_8, _MM_SHUFFLE(0, 1, 2, 3));
    }
#else
    /* This causes strict-aliasing warnings with GCC. */
    switch (order)
    {
    case 12: ((drflac_int32*)&coefficients128_8)[0] = coefficients[11]; ((drflac_int32*)&samples128_8)[0] = pDecodedSamples[-12];
    case 11: ((drflac_int32*)&coefficients128_8)[1] = coefficients[10]; ((drflac_int32*)&samples128_8)[1] = pDecodedSamples[-11];
    case 10: ((drflac_int32*)&coefficients128_8)[2] = coefficients[ 9]; ((drflac_int32*)&samples128_8)[2] = pDecodedSamples[-10];
    case 9:  ((drflac_int32*)&coefficients128_8)[3] = coefficients[ 8]; ((drflac_int32*)&samples128_8)[3] = pDecodedSamples[- 9];
    case 8:  ((drflac_int32*)&coefficients128_4)[0] = coefficients[ 7]; ((drflac_int32*)&samples128_4)[0] = pDecodedSamples[- 8];
    case 7:  ((drflac_int32*)&coefficients128_4)[1] = coefficients[ 6]; ((drflac_int32*)&samples128_4)[1] = pDecodedSamples[- 7];
    case 6:  ((drflac_int32*)&coefficients128_4)[2] = coefficients[ 5]; ((drflac_int32*)&samples128_4)[2] = pDecodedSamples[- 6];
    case 5:  ((drflac_int32*)&coefficients128_4)[3] = coefficients[ 4]; ((drflac_int32*)&samples128_4)[3] = pDecodedSamples[- 5];
    case 4:  ((drflac_int32*)&coefficients128_0)[0] = coefficients[ 3]; ((drflac_int32*)&samples128_0)[0] = pDecodedSamples[- 4];
    case 3:  ((drflac_int32*)&coefficients128_0)[1] = coefficients[ 2]; ((drflac_int32*)&samples128_0)[1] = pDecodedSamples[- 3];
    case 2:  ((drflac_int32*)&coefficients128_0)[2] = coefficients[ 1]; ((drflac_int32*)&samples128_0)[2] = pDecodedSamples[- 2];
    case 1:  ((drflac_int32*)&coefficients128_0)[3] = coefficients[ 0]; ((drflac_int32*)&samples128_0)[3] = pDecodedSamples[- 1];
    }
#endif

    /* For this version we are doing one sample at a time. */
    while (pDecodedSamples < pDecodedSamplesEnd) {
        __m128i prediction128;
        __m128i zeroCountPart128;
        __m128i riceParamPart128;

        if (!drflac__read_rice_parts_x1(bs, riceParam, &zeroCountParts0, &riceParamParts0) ||
            !drflac__read_rice_parts_x1(bs, riceParam, &zeroCountParts1, &riceParamParts1) ||
            !drflac__read_rice_parts_x1(bs, riceParam, &zeroCountParts2, &riceParamParts2) ||
            !drflac__read_rice_parts_x1(bs, riceParam, &zeroCountParts3, &riceParamParts3)) {
            return DRFLAC_FALSE;
        }

        zeroCountPart128 = _mm_set_epi32(zeroCountParts3, zeroCountParts2, zeroCountParts1, zeroCountParts0);
        riceParamPart128 = _mm_set_epi32(riceParamParts3, riceParamParts2, riceParamParts1, riceParamParts0);

        riceParamPart128 = _mm_and_si128(riceParamPart128, riceParamMask128);
        riceParamPart128 = _mm_or_si128(riceParamPart128, _mm_slli_epi32(zeroCountPart128, riceParam));
        riceParamPart128 = _mm_xor_si128(_mm_srli_epi32(riceParamPart128, 1), _mm_add_epi32(drflac__mm_not_si128(_mm_and_si128(riceParamPart128, _mm_set1_epi32(0x01))), _mm_set1_epi32(0x01)));  /* <-- SSE2 compatible */
        /*riceParamPart128 = _mm_xor_si128(_mm_srli_epi32(riceParamPart128, 1), _mm_mullo_epi32(_mm_and_si128(riceParamPart128, _mm_set1_epi32(0x01)), _mm_set1_epi32(0xFFFFFFFF)));*/   /* <-- Only supported from SSE4.1 and is slower in my testing... */

        if (order <= 4) {
            for (i = 0; i < 4; i += 1) {
                prediction128 = _mm_mullo_epi32(coefficients128_0, samples128_0);

                /* Horizontal add and shift. */
                prediction128 = drflac__mm_hadd_epi32(prediction128);
                prediction128 = _mm_srai_epi32(prediction128, shift);
                prediction128 = _mm_add_epi32(riceParamPart128, prediction128);

                samples128_0 = _mm_alignr_epi8(prediction128, samples128_0, 4);
                riceParamPart128 = _mm_alignr_epi8(_mm_setzero_si128(), riceParamPart128, 4);
            }
        } else if (order <= 8) {
            for (i = 0; i < 4; i += 1) {
                prediction128 =                              _mm_mullo_epi32(coefficients128_4, samples128_4);
                prediction128 = _mm_add_epi32(prediction128, _mm_mullo_epi32(coefficients128_0, samples128_0));

                /* Horizontal add and shift. */
                prediction128 = drflac__mm_hadd_epi32(prediction128);
                prediction128 = _mm_srai_epi32(prediction128, shift);
                prediction128 = _mm_add_epi32(riceParamPart128, prediction128);

                samples128_4 = _mm_alignr_epi8(samples128_0,  samples128_4, 4);
                samples128_0 = _mm_alignr_epi8(prediction128, samples128_0, 4);
                riceParamPart128 = _mm_alignr_epi8(_mm_setzero_si128(), riceParamPart128, 4);
            }
        } else {
            for (i = 0; i < 4; i += 1) {
                prediction128 =                              _mm_mullo_epi32(coefficients128_8, samples128_8);
                prediction128 = _mm_add_epi32(prediction128, _mm_mullo_epi32(coefficients128_4, samples128_4));
                prediction128 = _mm_add_epi32(prediction128, _mm_mullo_epi32(coefficients128_0, samples128_0));

                /* Horizontal add and shift. */
                prediction128 = drflac__mm_hadd_epi32(prediction128);
                prediction128 = _mm_srai_epi32(prediction128, shift);
                prediction128 = _mm_add_epi32(riceParamPart128, prediction128);

                samples128_8 = _mm_alignr_epi8(samples128_4,  samples128_8, 4);
                samples128_4 = _mm_alignr_epi8(samples128_0,  samples128_4, 4);
                samples128_0 = _mm_alignr_epi8(prediction128, samples128_0, 4);
                riceParamPart128 = _mm_alignr_epi8(_mm_setzero_si128(), riceParamPart128, 4);
            }
        }

        /* We store samples in groups of 4. */
        _mm_storeu_si128((__m128i*)pDecodedSamples, samples128_0);
        pDecodedSamples += 4;
    }

    /* Make sure we process the last few samples. */
    i = (count & ~3);
    while (i < (int)count) {
        /* Rice extraction. */
        if (!drflac__read_rice_parts_x1(bs, riceParam, &zeroCountParts0, &riceParamParts0)) {
            return DRFLAC_FALSE;
        }

        /* Rice reconstruction. */
        riceParamParts0 &= riceParamMask;
        riceParamParts0 |= (zeroCountParts0 << riceParam);
        riceParamParts0  = (riceParamParts0 >> 1) ^ t[riceParamParts0 & 0x01];

        /* Sample reconstruction. */
        pDecodedSamples[0] = riceParamParts0 + drflac__calculate_prediction_32(order, shift, coefficients, pDecodedSamples);

        i += 1;
        pDecodedSamples += 1;
    }

    return DRFLAC_TRUE;
}

static drflac_bool32 drflac__decode_samples_with_residual__rice__sse41_64(drflac_bs* bs, drflac_uint32 count, drflac_uint8 riceParam, drflac_uint32 order, drflac_int32 shift, const drflac_int32* coefficients, drflac_int32* pSamplesOut)
{
    int i;
    drflac_uint32 riceParamMask;
    drflac_int32* pDecodedSamples    = pSamplesOut;
    drflac_int32* pDecodedSamplesEnd = pSamplesOut + (count & ~3);
    drflac_uint32 zeroCountParts0 = 0;
    drflac_uint32 zeroCountParts1 = 0;
    drflac_uint32 zeroCountParts2 = 0;
    drflac_uint32 zeroCountParts3 = 0;
    drflac_uint32 riceParamParts0 = 0;
    drflac_uint32 riceParamParts1 = 0;
    drflac_uint32 riceParamParts2 = 0;
    drflac_uint32 riceParamParts3 = 0;
    __m128i coefficients128_0;
    __m128i coefficients128_4;
    __m128i coefficients128_8;
    __m128i samples128_0;
    __m128i samples128_4;
    __m128i samples128_8;
    __m128i prediction128;
    __m128i riceParamMask128;

    const drflac_uint32 t[2] = {0x00000000, 0xFFFFFFFF};

    DRFLAC_ASSERT(order <= 12);

    riceParamMask    = (drflac_uint32)~((~0UL) << riceParam);
    riceParamMask128 = _mm_set1_epi32(riceParamMask);

    prediction128 = _mm_setzero_si128();

    /* Pre-load. */
    coefficients128_0  = _mm_setzero_si128();
    coefficients128_4  = _mm_setzero_si128();
    coefficients128_8  = _mm_setzero_si128();

    samples128_0  = _mm_setzero_si128();
    samples128_4  = _mm_setzero_si128();
    samples128_8  = _mm_setzero_si128();

#if 1
    {
        int runningOrder = order;

        /* 0 - 3. */
        if (runningOrder >= 4) {
            coefficients128_0 = _mm_loadu_si128((const __m128i*)(coefficients + 0));
            samples128_0      = _mm_loadu_si128((const __m128i*)(pSamplesOut  - 4));
            runningOrder -= 4;
        } else {
            switch (runningOrder) {
                case 3: coefficients128_0 = _mm_set_epi32(0, coefficients[2], coefficients[1], coefficients[0]); samples128_0 = _mm_set_epi32(pSamplesOut[-1], pSamplesOut[-2], pSamplesOut[-3], 0); break;
                case 2: coefficients128_0 = _mm_set_epi32(0, 0,               coefficients[1], coefficients[0]); samples128_0 = _mm_set_epi32(pSamplesOut[-1], pSamplesOut[-2], 0,               0); break;
                case 1: coefficients128_0 = _mm_set_epi32(0, 0,               0,               coefficients[0]); samples128_0 = _mm_set_epi32(pSamplesOut[-1], 0,               0,               0); break;
            }
            runningOrder = 0;
        }

        /* 4 - 7 */
        if (runningOrder >= 4) {
            coefficients128_4 = _mm_loadu_si128((const __m128i*)(coefficients + 4));
            samples128_4      = _mm_loadu_si128((const __m128i*)(pSamplesOut  - 8));
            runningOrder -= 4;
        } else {
            switch (runningOrder) {
                case 3: coefficients128_4 = _mm_set_epi32(0, coefficients[6], coefficients[5], coefficients[4]); samples128_4 = _mm_set_epi32(pSamplesOut[-5], pSamplesOut[-6], pSamplesOut[-7], 0); break;
                case 2: coefficients128_4 = _mm_set_epi32(0, 0,               coefficients[5], coefficients[4]); samples128_4 = _mm_set_epi32(pSamplesOut[-5], pSamplesOut[-6], 0,               0); break;
                case 1: coefficients128_4 = _mm_set_epi32(0, 0,               0,               coefficients[4]); samples128_4 = _mm_set_epi32(pSamplesOut[-5], 0,               0,               0); break;
            }
            runningOrder = 0;
        }

        /* 8 - 11 */
        if (runningOrder == 4) {
            coefficients128_8 = _mm_loadu_si128((const __m128i*)(coefficients + 8));
            samples128_8      = _mm_loadu_si128((const __m128i*)(pSamplesOut  - 12));
            runningOrder -= 4;
        } else {
            switch (runningOrder) {
                case 3: coefficients128_8 = _mm_set_epi32(0, coefficients[10], coefficients[9], coefficients[8]); samples128_8 = _mm_set_epi32(pSamplesOut[-9], pSamplesOut[-10], pSamplesOut[-11], 0); break;
                case 2: coefficients128_8 = _mm_set_epi32(0, 0,                coefficients[9], coefficients[8]); samples128_8 = _mm_set_epi32(pSamplesOut[-9], pSamplesOut[-10], 0,                0); break;
                case 1: coefficients128_8 = _mm_set_epi32(0, 0,                0,               coefficients[8]); samples128_8 = _mm_set_epi32(pSamplesOut[-9], 0,                0,                0); break;
            }
            runningOrder = 0;
        }

        /* Coefficients need to be shuffled for our streaming algorithm below to work. Samples are already in the correct order from the loading routine above. */
        coefficients128_0 = _mm_shuffle_epi32(coefficients128_0, _MM_SHUFFLE(0, 1, 2, 3));
        coefficients128_4 = _mm_shuffle_epi32(coefficients128_4, _MM_SHUFFLE(0, 1, 2, 3));
        coefficients128_8 = _mm_shuffle_epi32(coefficients128_8, _MM_SHUFFLE(0, 1, 2, 3));
    }
#else
    switch (order)
    {
    case 12: ((drflac_int32*)&coefficients128_8)[0] = coefficients[11]; ((drflac_int32*)&samples128_8)[0] = pDecodedSamples[-12];
    case 11: ((drflac_int32*)&coefficients128_8)[1] = coefficients[10]; ((drflac_int32*)&samples128_8)[1] = pDecodedSamples[-11];
    case 10: ((drflac_int32*)&coefficients128_8)[2] = coefficients[ 9]; ((drflac_int32*)&samples128_8)[2] = pDecodedSamples[-10];
    case 9:  ((drflac_int32*)&coefficients128_8)[3] = coefficients[ 8]; ((drflac_int32*)&samples128_8)[3] = pDecodedSamples[- 9];
    case 8:  ((drflac_int32*)&coefficients128_4)[0] = coefficients[ 7]; ((drflac_int32*)&samples128_4)[0] = pDecodedSamples[- 8];
    case 7:  ((drflac_int32*)&coefficients128_4)[1] = coefficients[ 6]; ((drflac_int32*)&samples128_4)[1] = pDecodedSamples[- 7];
    case 6:  ((drflac_int32*)&coefficients128_4)[2] = coefficients[ 5]; ((drflac_int32*)&samples128_4)[2] = pDecodedSamples[- 6];
    case 5:  ((drflac_int32*)&coefficients128_4)[3] = coefficients[ 4]; ((drflac_int32*)&samples128_4)[3] = pDecodedSamples[- 5];
    case 4:  ((drflac_int32*)&coefficients128_0)[0] = coefficients[ 3]; ((drflac_int32*)&samples128_0)[0] = pDecodedSamples[- 4];
    case 3:  ((drflac_int32*)&coefficients128_0)[1] = coefficients[ 2]; ((drflac_int32*)&samples128_0)[1] = pDecodedSamples[- 3];
    case 2:  ((drflac_int32*)&coefficients128_0)[2] = coefficients[ 1]; ((drflac_int32*)&samples128_0)[2] = pDecodedSamples[- 2];
    case 1:  ((drflac_int32*)&coefficients128_0)[3] = coefficients[ 0]; ((drflac_int32*)&samples128_0)[3] = pDecodedSamples[- 1];
    }
#endif

    /* For this version we are doing one sample at a time. */
    while (pDecodedSamples < pDecodedSamplesEnd) {
        __m128i zeroCountPart128;
        __m128i riceParamPart128;

        if (!drflac__read_rice_parts_x1(bs, riceParam, &zeroCountParts0, &riceParamParts0) ||
            !drflac__read_rice_parts_x1(bs, riceParam, &zeroCountParts1, &riceParamParts1) ||
            !drflac__read_rice_parts_x1(bs, riceParam, &zeroCountParts2, &riceParamParts2) ||
            !drflac__read_rice_parts_x1(bs, riceParam, &zeroCountParts3, &riceParamParts3)) {
            return DRFLAC_FALSE;
        }

        zeroCountPart128 = _mm_set_epi32(zeroCountParts3, zeroCountParts2, zeroCountParts1, zeroCountParts0);
        riceParamPart128 = _mm_set_epi32(riceParamParts3, riceParamParts2, riceParamParts1, riceParamParts0);

        riceParamPart128 = _mm_and_si128(riceParamPart128, riceParamMask128);
        riceParamPart128 = _mm_or_si128(riceParamPart128, _mm_slli_epi32(zeroCountPart128, riceParam));
        riceParamPart128 = _mm_xor_si128(_mm_srli_epi32(riceParamPart128, 1), _mm_add_epi32(drflac__mm_not_si128(_mm_and_si128(riceParamPart128, _mm_set1_epi32(1))), _mm_set1_epi32(1)));

        for (i = 0; i < 4; i += 1) {
            prediction128 = _mm_xor_si128(prediction128, prediction128);    /* Reset to 0. */

            switch (order)
            {
            case 12:
            case 11: prediction128 = _mm_add_epi64(prediction128, _mm_mul_epi32(_mm_shuffle_epi32(coefficients128_8, _MM_SHUFFLE(1, 1, 0, 0)), _mm_shuffle_epi32(samples128_8, _MM_SHUFFLE(1, 1, 0, 0))));
            case 10:
            case  9: prediction128 = _mm_add_epi64(prediction128, _mm_mul_epi32(_mm_shuffle_epi32(coefficients128_8, _MM_SHUFFLE(3, 3, 2, 2)), _mm_shuffle_epi32(samples128_8, _MM_SHUFFLE(3, 3, 2, 2))));
            case  8:
            case  7: prediction128 = _mm_add_epi64(prediction128, _mm_mul_epi32(_mm_shuffle_epi32(coefficients128_4, _MM_SHUFFLE(1, 1, 0, 0)), _mm_shuffle_epi32(samples128_4, _MM_SHUFFLE(1, 1, 0, 0))));
            case  6:
            case  5: prediction128 = _mm_add_epi64(prediction128, _mm_mul_epi32(_mm_shuffle_epi32(coefficients128_4, _MM_SHUFFLE(3, 3, 2, 2)), _mm_shuffle_epi32(samples128_4, _MM_SHUFFLE(3, 3, 2, 2))));
            case  4:
            case  3: prediction128 = _mm_add_epi64(prediction128, _mm_mul_epi32(_mm_shuffle_epi32(coefficients128_0, _MM_SHUFFLE(1, 1, 0, 0)), _mm_shuffle_epi32(samples128_0, _MM_SHUFFLE(1, 1, 0, 0))));
            case  2:
            case  1: prediction128 = _mm_add_epi64(prediction128, _mm_mul_epi32(_mm_shuffle_epi32(coefficients128_0, _MM_SHUFFLE(3, 3, 2, 2)), _mm_shuffle_epi32(samples128_0, _MM_SHUFFLE(3, 3, 2, 2))));
            }

            /* Horizontal add and shift. */
            prediction128 = drflac__mm_hadd_epi64(prediction128);
            prediction128 = drflac__mm_srai_epi64(prediction128, shift);
            prediction128 = _mm_add_epi32(riceParamPart128, prediction128);

            /* Our value should be sitting in prediction128[0]. We need to combine this with our SSE samples. */
            samples128_8 = _mm_alignr_epi8(samples128_4,  samples128_8, 4);
            samples128_4 = _mm_alignr_epi8(samples128_0,  samples128_4, 4);
            samples128_0 = _mm_alignr_epi8(prediction128, samples128_0, 4);

            /* Slide our rice parameter down so that the value in position 0 contains the next one to process. */
            riceParamPart128 = _mm_alignr_epi8(_mm_setzero_si128(), riceParamPart128, 4);
        }

        /* We store samples in groups of 4. */
        _mm_storeu_si128((__m128i*)pDecodedSamples, samples128_0);
        pDecodedSamples += 4;
    }

    /* Make sure we process the last few samples. */
    i = (count & ~3);
    while (i < (int)count) {
        /* Rice extraction. */
        if (!drflac__read_rice_parts_x1(bs, riceParam, &zeroCountParts0, &riceParamParts0)) {
            return DRFLAC_FALSE;
        }

        /* Rice reconstruction. */
        riceParamParts0 &= riceParamMask;
        riceParamParts0 |= (zeroCountParts0 << riceParam);
        riceParamParts0  = (riceParamParts0 >> 1) ^ t[riceParamParts0 & 0x01];

        /* Sample reconstruction. */
        pDecodedSamples[0] = riceParamParts0 + drflac__calculate_prediction_64(order, shift, coefficients, pDecodedSamples);

        i += 1;
        pDecodedSamples += 1;
    }

    return DRFLAC_TRUE;
}

static drflac_bool32 drflac__decode_samples_with_residual__rice__sse41(drflac_bs* bs, drflac_uint32 bitsPerSample, drflac_uint32 count, drflac_uint8 riceParam, drflac_uint32 lpcOrder, drflac_int32 lpcShift, drflac_uint32 lpcPrecision, const drflac_int32* coefficients, drflac_int32* pSamplesOut)
{
    DRFLAC_ASSERT(bs != NULL);
    DRFLAC_ASSERT(pSamplesOut != NULL);

    /* In my testing the order is rarely > 12, so in this case I'm going to simplify the SSE implementation by only handling order <= 12. */
    if (lpcOrder > 0 && lpcOrder <= 12) {
        if (drflac__use_64_bit_prediction(bitsPerSample, lpcOrder, lpcPrecision)) {
            return drflac__decode_samples_with_residual__rice__sse41_64(bs, count, riceParam, lpcOrder, lpcShift, coefficients, pSamplesOut);
        } else {
            return drflac__decode_samples_with_residual__rice__sse41_32(bs, count, riceParam, lpcOrder, lpcShift, coefficients, pSamplesOut);
        }
    } else {
        return drflac__decode_samples_with_residual__rice__scalar(bs, bitsPerSample, count, riceParam, lpcOrder, lpcShift, lpcPrecision, coefficients, pSamplesOut);
    }
}
#endif

#if defined(DRFLAC_SUPPORT_NEON)
static DRFLAC_INLINE void drflac__vst2q_s32(drflac_int32* p, int32x4x2_t x)
{
    vst1q_s32(p+0, x.val[0]);
    vst1q_s32(p+4, x.val[1]);
}

static DRFLAC_INLINE void drflac__vst2q_u32(drflac_uint32* p, uint32x4x2_t x)
{
    vst1q_u32(p+0, x.val[0]);
    vst1q_u32(p+4, x.val[1]);
}

static DRFLAC_INLINE void drflac__vst2q_f32(float* p, float32x4x2_t x)
{
    vst1q_f32(p+0, x.val[0]);
    vst1q_f32(p+4, x.val[1]);
}

static DRFLAC_INLINE void drflac__vst2q_s16(drflac_int16* p, int16x4x2_t x)
{
    vst1q_s16(p, vcombine_s16(x.val[0], x.val[1]));
}

static DRFLAC_INLINE void drflac__vst2q_u16(drflac_uint16* p, uint16x4x2_t x)
{
    vst1q_u16(p, vcombine_u16(x.val[0], x.val[1]));
}

static DRFLAC_INLINE int32x4_t drflac__vdupq_n_s32x4(drflac_int32 x3, drflac_int32 x2, drflac_int32 x1, drflac_int32 x0)
{
    drflac_int32 x[4];
    x[3] = x3;
    x[2] = x2;
    x[1] = x1;
    x[0] = x0;
    return vld1q_s32(x);
}

static DRFLAC_INLINE int32x4_t drflac__valignrq_s32_1(int32x4_t a, int32x4_t b)
{
    /* Equivalent to SSE's _mm_alignr_epi8(a, b, 4) */

    /* Reference */
    /*return drflac__vdupq_n_s32x4(
        vgetq_lane_s32(a, 0),
        vgetq_lane_s32(b, 3),
        vgetq_lane_s32(b, 2),
        vgetq_lane_s32(b, 1)
    );*/

    return vextq_s32(b, a, 1);
}

static DRFLAC_INLINE uint32x4_t drflac__valignrq_u32_1(uint32x4_t a, uint32x4_t b)
{
    /* Equivalent to SSE's _mm_alignr_epi8(a, b, 4) */

    /* Reference */
    /*return drflac__vdupq_n_s32x4(
        vgetq_lane_s32(a, 0),
        vgetq_lane_s32(b, 3),
        vgetq_lane_s32(b, 2),
        vgetq_lane_s32(b, 1)
    );*/

    return vextq_u32(b, a, 1);
}

static DRFLAC_INLINE int32x2_t drflac__vhaddq_s32(int32x4_t x)
{
    /* The sum must end up in position 0. */

    /* Reference */
    /*return vdupq_n_s32(
        vgetq_lane_s32(x, 3) +
        vgetq_lane_s32(x, 2) +
        vgetq_lane_s32(x, 1) +
        vgetq_lane_s32(x, 0)
    );*/

    int32x2_t r = vadd_s32(vget_high_s32(x), vget_low_s32(x));
    return vpadd_s32(r, r);
}

static DRFLAC_INLINE int64x1_t drflac__vhaddq_s64(int64x2_t x)
{
    return vadd_s64(vget_high_s64(x), vget_low_s64(x));
}

static DRFLAC_INLINE int32x4_t drflac__vrevq_s32(int32x4_t x)
{
    /* Reference */
    /*return drflac__vdupq_n_s32x4(
        vgetq_lane_s32(x, 0),
        vgetq_lane_s32(x, 1),
        vgetq_lane_s32(x, 2),
        vgetq_lane_s32(x, 3)
    );*/

    return vrev64q_s32(vcombine_s32(vget_high_s32(x), vget_low_s32(x)));
}

static DRFLAC_INLINE int32x4_t drflac__vnotq_s32(int32x4_t x)
{
    return veorq_s32(x, vdupq_n_s32(0xFFFFFFFF));
}

static DRFLAC_INLINE uint32x4_t drflac__vnotq_u32(uint32x4_t x)
{
    return veorq_u32(x, vdupq_n_u32(0xFFFFFFFF));
}

static drflac_bool32 drflac__decode_samples_with_residual__rice__neon_32(drflac_bs* bs, drflac_uint32 count, drflac_uint8 riceParam, drflac_uint32 order, drflac_int32 shift, const drflac_int32* coefficients, drflac_int32* pSamplesOut)
{
    int i;
    drflac_uint32 riceParamMask;
    drflac_int32* pDecodedSamples    = pSamplesOut;
    drflac_int32* pDecodedSamplesEnd = pSamplesOut + (count & ~3);
    drflac_uint32 zeroCountParts[4];
    drflac_uint32 riceParamParts[4];
    int32x4_t coefficients128_0;
    int32x4_t coefficients128_4;
    int32x4_t coefficients128_8;
    int32x4_t samples128_0;
    int32x4_t samples128_4;
    int32x4_t samples128_8;
    uint32x4_t riceParamMask128;
    int32x4_t riceParam128;
    int32x2_t shift64;
    uint32x4_t one128;

    const drflac_uint32 t[2] = {0x00000000, 0xFFFFFFFF};

    riceParamMask    = (drflac_uint32)~((~0UL) << riceParam);
    riceParamMask128 = vdupq_n_u32(riceParamMask);

    riceParam128 = vdupq_n_s32(riceParam);
    shift64 = vdup_n_s32(-shift); /* Negate the shift because we'll be doing a variable shift using vshlq_s32(). */
    one128 = vdupq_n_u32(1);

    /*
    Pre-loading the coefficients and prior samples is annoying because we need to ensure we don't try reading more than
    what's available in the input buffers. It would be conenient to use a fall-through switch to do this, but this results
    in strict aliasing warnings with GCC. To work around this I'm just doing something hacky. This feels a bit convoluted
    so I think there's opportunity for this to be simplified.
    */
    {
        int runningOrder = order;
        drflac_int32 tempC[4] = {0, 0, 0, 0};
        drflac_int32 tempS[4] = {0, 0, 0, 0};

        /* 0 - 3. */
        if (runningOrder >= 4) {
            coefficients128_0 = vld1q_s32(coefficients + 0);
            samples128_0      = vld1q_s32(pSamplesOut  - 4);
            runningOrder -= 4;
        } else {
            switch (runningOrder) {
                case 3: tempC[2] = coefficients[2]; tempS[1] = pSamplesOut[-3]; /* fallthrough */
                case 2: tempC[1] = coefficients[1]; tempS[2] = pSamplesOut[-2]; /* fallthrough */
                case 1: tempC[0] = coefficients[0]; tempS[3] = pSamplesOut[-1]; /* fallthrough */
            }

            coefficients128_0 = vld1q_s32(tempC);
            samples128_0      = vld1q_s32(tempS);
            runningOrder = 0;
        }

        /* 4 - 7 */
        if (runningOrder >= 4) {
            coefficients128_4 = vld1q_s32(coefficients + 4);
            samples128_4      = vld1q_s32(pSamplesOut  - 8);
            runningOrder -= 4;
        } else {
            switch (runningOrder) {
                case 3: tempC[2] = coefficients[6]; tempS[1] = pSamplesOut[-7]; /* fallthrough */
                case 2: tempC[1] = coefficients[5]; tempS[2] = pSamplesOut[-6]; /* fallthrough */
                case 1: tempC[0] = coefficients[4]; tempS[3] = pSamplesOut[-5]; /* fallthrough */
            }

            coefficients128_4 = vld1q_s32(tempC);
            samples128_4      = vld1q_s32(tempS);
            runningOrder = 0;
        }

        /* 8 - 11 */
        if (runningOrder == 4) {
            coefficients128_8 = vld1q_s32(coefficients + 8);
            samples128_8      = vld1q_s32(pSamplesOut  - 12);
            runningOrder -= 4;
        } else {
            switch (runningOrder) {
                case 3: tempC[2] = coefficients[10]; tempS[1] = pSamplesOut[-11]; /* fallthrough */
                case 2: tempC[1] = coefficients[ 9]; tempS[2] = pSamplesOut[-10]; /* fallthrough */
                case 1: tempC[0] = coefficients[ 8]; tempS[3] = pSamplesOut[- 9]; /* fallthrough */
            }

            coefficients128_8 = vld1q_s32(tempC);
            samples128_8      = vld1q_s32(tempS);
            runningOrder = 0;
        }

        /* Coefficients need to be shuffled for our streaming algorithm below to work. Samples are already in the correct order from the loading routine above. */
        coefficients128_0 = drflac__vrevq_s32(coefficients128_0);
        coefficients128_4 = drflac__vrevq_s32(coefficients128_4);
        coefficients128_8 = drflac__vrevq_s32(coefficients128_8);
    }

    /* For this version we are doing one sample at a time. */
    while (pDecodedSamples < pDecodedSamplesEnd) {
        int32x4_t prediction128;
        int32x2_t prediction64;
        uint32x4_t zeroCountPart128;
        uint32x4_t riceParamPart128;

        if (!drflac__read_rice_parts_x1(bs, riceParam, &zeroCountParts[0], &riceParamParts[0]) ||
            !drflac__read_rice_parts_x1(bs, riceParam, &zeroCountParts[1], &riceParamParts[1]) ||
            !drflac__read_rice_parts_x1(bs, riceParam, &zeroCountParts[2], &riceParamParts[2]) ||
            !drflac__read_rice_parts_x1(bs, riceParam, &zeroCountParts[3], &riceParamParts[3])) {
            return DRFLAC_FALSE;
        }

        zeroCountPart128 = vld1q_u32(zeroCountParts);
        riceParamPart128 = vld1q_u32(riceParamParts);

        riceParamPart128 = vandq_u32(riceParamPart128, riceParamMask128);
        riceParamPart128 = vorrq_u32(riceParamPart128, vshlq_u32(zeroCountPart128, riceParam128));
        riceParamPart128 = veorq_u32(vshrq_n_u32(riceParamPart128, 1), vaddq_u32(drflac__vnotq_u32(vandq_u32(riceParamPart128, one128)), one128));

        if (order <= 4) {
            for (i = 0; i < 4; i += 1) {
                prediction128 = vmulq_s32(coefficients128_0, samples128_0);

                /* Horizontal add and shift. */
                prediction64 = drflac__vhaddq_s32(prediction128);
                prediction64 = vshl_s32(prediction64, shift64);
                prediction64 = vadd_s32(prediction64, vget_low_s32(vreinterpretq_s32_u32(riceParamPart128)));

                samples128_0 = drflac__valignrq_s32_1(vcombine_s32(prediction64, vdup_n_s32(0)), samples128_0);
                riceParamPart128 = drflac__valignrq_u32_1(vdupq_n_u32(0), riceParamPart128);
            }
        } else if (order <= 8) {
            for (i = 0; i < 4; i += 1) {
                prediction128 =                vmulq_s32(coefficients128_4, samples128_4);
                prediction128 = vmlaq_s32(prediction128, coefficients128_0, samples128_0);

                /* Horizontal add and shift. */
                prediction64 = drflac__vhaddq_s32(prediction128);
                prediction64 = vshl_s32(prediction64, shift64);
                prediction64 = vadd_s32(prediction64, vget_low_s32(vreinterpretq_s32_u32(riceParamPart128)));

                samples128_4 = drflac__valignrq_s32_1(samples128_0, samples128_4);
                samples128_0 = drflac__valignrq_s32_1(vcombine_s32(prediction64, vdup_n_s32(0)), samples128_0);
                riceParamPart128 = drflac__valignrq_u32_1(vdupq_n_u32(0), riceParamPart128);
            }
        } else {
            for (i = 0; i < 4; i += 1) {
                prediction128 =                vmulq_s32(coefficients128_8, samples128_8);
                prediction128 = vmlaq_s32(prediction128, coefficients128_4, samples128_4);
                prediction128 = vmlaq_s32(prediction128, coefficients128_0, samples128_0);

                /* Horizontal add and shift. */
                prediction64 = drflac__vhaddq_s32(prediction128);
                prediction64 = vshl_s32(prediction64, shift64);
                prediction64 = vadd_s32(prediction64, vget_low_s32(vreinterpretq_s32_u32(riceParamPart128)));

                samples128_8 = drflac__valignrq_s32_1(samples128_4, samples128_8);
                samples128_4 = drflac__valignrq_s32_1(samples128_0, samples128_4);
                samples128_0 = drflac__valignrq_s32_1(vcombine_s32(prediction64, vdup_n_s32(0)), samples128_0);
                riceParamPart128 = drflac__valignrq_u32_1(vdupq_n_u32(0), riceParamPart128);
            }
        }

        /* We store samples in groups of 4. */
        vst1q_s32(pDecodedSamples, samples128_0);
        pDecodedSamples += 4;
    }

    /* Make sure we process the last few samples. */
    i = (count & ~3);
    while (i < (int)count) {
        /* Rice extraction. */
        if (!drflac__read_rice_parts_x1(bs, riceParam, &zeroCountParts[0], &riceParamParts[0])) {
            return DRFLAC_FALSE;
        }

        /* Rice reconstruction. */
        riceParamParts[0] &= riceParamMask;
        riceParamParts[0] |= (zeroCountParts[0] << riceParam);
        riceParamParts[0]  = (riceParamParts[0] >> 1) ^ t[riceParamParts[0] & 0x01];

        /* Sample reconstruction. */
        pDecodedSamples[0] = riceParamParts[0] + drflac__calculate_prediction_32(order, shift, coefficients, pDecodedSamples);

        i += 1;
        pDecodedSamples += 1;
    }

    return DRFLAC_TRUE;
}

static drflac_bool32 drflac__decode_samples_with_residual__rice__neon_64(drflac_bs* bs, drflac_uint32 count, drflac_uint8 riceParam, drflac_uint32 order, drflac_int32 shift, const drflac_int32* coefficients, drflac_int32* pSamplesOut)
{
    int i;
    drflac_uint32 riceParamMask;
    drflac_int32* pDecodedSamples    = pSamplesOut;
    drflac_int32* pDecodedSamplesEnd = pSamplesOut + (count & ~3);
    drflac_uint32 zeroCountParts[4];
    drflac_uint32 riceParamParts[4];
    int32x4_t coefficients128_0;
    int32x4_t coefficients128_4;
    int32x4_t coefficients128_8;
    int32x4_t samples128_0;
    int32x4_t samples128_4;
    int32x4_t samples128_8;
    uint32x4_t riceParamMask128;
    int32x4_t riceParam128;
    int64x1_t shift64;
    uint32x4_t one128;
    int64x2_t prediction128 = { 0 };
    uint32x4_t zeroCountPart128;
    uint32x4_t riceParamPart128;

    const drflac_uint32 t[2] = {0x00000000, 0xFFFFFFFF};

    riceParamMask    = (drflac_uint32)~((~0UL) << riceParam);
    riceParamMask128 = vdupq_n_u32(riceParamMask);

    riceParam128 = vdupq_n_s32(riceParam);
    shift64 = vdup_n_s64(-shift); /* Negate the shift because we'll be doing a variable shift using vshlq_s32(). */
    one128 = vdupq_n_u32(1);

    /*
    Pre-loading the coefficients and prior samples is annoying because we need to ensure we don't try reading more than
    what's available in the input buffers. It would be convenient to use a fall-through switch to do this, but this results
    in strict aliasing warnings with GCC. To work around this I'm just doing something hacky. This feels a bit convoluted
    so I think there's opportunity for this to be simplified.
    */
    {
        int runningOrder = order;
        drflac_int32 tempC[4] = {0, 0, 0, 0};
        drflac_int32 tempS[4] = {0, 0, 0, 0};

        /* 0 - 3. */
        if (runningOrder >= 4) {
            coefficients128_0 = vld1q_s32(coefficients + 0);
            samples128_0      = vld1q_s32(pSamplesOut  - 4);
            runningOrder -= 4;
        } else {
            switch (runningOrder) {
                case 3: tempC[2] = coefficients[2]; tempS[1] = pSamplesOut[-3]; /* fallthrough */
                case 2: tempC[1] = coefficients[1]; tempS[2] = pSamplesOut[-2]; /* fallthrough */
                case 1: tempC[0] = coefficients[0]; tempS[3] = pSamplesOut[-1]; /* fallthrough */
            }

            coefficients128_0 = vld1q_s32(tempC);
            samples128_0      = vld1q_s32(tempS);
            runningOrder = 0;
        }

        /* 4 - 7 */
        if (runningOrder >= 4) {
            coefficients128_4 = vld1q_s32(coefficients + 4);
            samples128_4      = vld1q_s32(pSamplesOut  - 8);
            runningOrder -= 4;
        } else {
            switch (runningOrder) {
                case 3: tempC[2] = coefficients[6]; tempS[1] = pSamplesOut[-7]; /* fallthrough */
                case 2: tempC[1] = coefficients[5]; tempS[2] = pSamplesOut[-6]; /* fallthrough */
                case 1: tempC[0] = coefficients[4]; tempS[3] = pSamplesOut[-5]; /* fallthrough */
            }

            coefficients128_4 = vld1q_s32(tempC);
            samples128_4      = vld1q_s32(tempS);
            runningOrder = 0;
        }

        /* 8 - 11 */
        if (runningOrder == 4) {
            coefficients128_8 = vld1q_s32(coefficients + 8);
            samples128_8      = vld1q_s32(pSamplesOut  - 12);
            runningOrder -= 4;
        } else {
            switch (runningOrder) {
                case 3: tempC[2] = coefficients[10]; tempS[1] = pSamplesOut[-11]; /* fallthrough */
                case 2: tempC[1] = coefficients[ 9]; tempS[2] = pSamplesOut[-10]; /* fallthrough */
                case 1: tempC[0] = coefficients[ 8]; tempS[3] = pSamplesOut[- 9]; /* fallthrough */
            }

            coefficients128_8 = vld1q_s32(tempC);
            samples128_8      = vld1q_s32(tempS);
            runningOrder = 0;
        }

        /* Coefficients need to be shuffled for our streaming algorithm below to work. Samples are already in the correct order from the loading routine above. */
        coefficients128_0 = drflac__vrevq_s32(coefficients128_0);
        coefficients128_4 = drflac__vrevq_s32(coefficients128_4);
        coefficients128_8 = drflac__vrevq_s32(coefficients128_8);
    }

    /* For this version we are doing one sample at a time. */
    while (pDecodedSamples < pDecodedSamplesEnd) {
        if (!drflac__read_rice_parts_x1(bs, riceParam, &zeroCountParts[0], &riceParamParts[0]) ||
            !drflac__read_rice_parts_x1(bs, riceParam, &zeroCountParts[1], &riceParamParts[1]) ||
            !drflac__read_rice_parts_x1(bs, riceParam, &zeroCountParts[2], &riceParamParts[2]) ||
            !drflac__read_rice_parts_x1(bs, riceParam, &zeroCountParts[3], &riceParamParts[3])) {
            return DRFLAC_FALSE;
        }

        zeroCountPart128 = vld1q_u32(zeroCountParts);
        riceParamPart128 = vld1q_u32(riceParamParts);

        riceParamPart128 = vandq_u32(riceParamPart128, riceParamMask128);
        riceParamPart128 = vorrq_u32(riceParamPart128, vshlq_u32(zeroCountPart128, riceParam128));
        riceParamPart128 = veorq_u32(vshrq_n_u32(riceParamPart128, 1), vaddq_u32(drflac__vnotq_u32(vandq_u32(riceParamPart128, one128)), one128));

        for (i = 0; i < 4; i += 1) {
            int64x1_t prediction64;

            prediction128 = veorq_s64(prediction128, prediction128);    /* Reset to 0. */
            switch (order)
            {
            case 12:
            case 11: prediction128 = vaddq_s64(prediction128, vmull_s32(vget_low_s32(coefficients128_8), vget_low_s32(samples128_8)));
            case 10:
            case  9: prediction128 = vaddq_s64(prediction128, vmull_s32(vget_high_s32(coefficients128_8), vget_high_s32(samples128_8)));
            case  8:
            case  7: prediction128 = vaddq_s64(prediction128, vmull_s32(vget_low_s32(coefficients128_4), vget_low_s32(samples128_4)));
            case  6:
            case  5: prediction128 = vaddq_s64(prediction128, vmull_s32(vget_high_s32(coefficients128_4), vget_high_s32(samples128_4)));
            case  4:
            case  3: prediction128 = vaddq_s64(prediction128, vmull_s32(vget_low_s32(coefficients128_0), vget_low_s32(samples128_0)));
            case  2:
            case  1: prediction128 = vaddq_s64(prediction128, vmull_s32(vget_high_s32(coefficients128_0), vget_high_s32(samples128_0)));
            }

            /* Horizontal add and shift. */
            prediction64 = drflac__vhaddq_s64(prediction128);
            prediction64 = vshl_s64(prediction64, shift64);
            prediction64 = vadd_s64(prediction64, vdup_n_s64(vgetq_lane_u32(riceParamPart128, 0)));

            /* Our value should be sitting in prediction64[0]. We need to combine this with our SSE samples. */
            samples128_8 = drflac__valignrq_s32_1(samples128_4, samples128_8);
            samples128_4 = drflac__valignrq_s32_1(samples128_0, samples128_4);
            samples128_0 = drflac__valignrq_s32_1(vcombine_s32(vreinterpret_s32_s64(prediction64), vdup_n_s32(0)), samples128_0);

            /* Slide our rice parameter down so that the value in position 0 contains the next one to process. */
            riceParamPart128 = drflac__valignrq_u32_1(vdupq_n_u32(0), riceParamPart128);
        }

        /* We store samples in groups of 4. */
        vst1q_s32(pDecodedSamples, samples128_0);
        pDecodedSamples += 4;
    }

    /* Make sure we process the last few samples. */
    i = (count & ~3);
    while (i < (int)count) {
        /* Rice extraction. */
        if (!drflac__read_rice_parts_x1(bs, riceParam, &zeroCountParts[0], &riceParamParts[0])) {
            return DRFLAC_FALSE;
        }

        /* Rice reconstruction. */
        riceParamParts[0] &= riceParamMask;
        riceParamParts[0] |= (zeroCountParts[0] << riceParam);
        riceParamParts[0]  = (riceParamParts[0] >> 1) ^ t[riceParamParts[0] & 0x01];

        /* Sample reconstruction. */
        pDecodedSamples[0] = riceParamParts[0] + drflac__calculate_prediction_64(order, shift, coefficients, pDecodedSamples);

        i += 1;
        pDecodedSamples += 1;
    }

    return DRFLAC_TRUE;
}

static drflac_bool32 drflac__decode_samples_with_residual__rice__neon(drflac_bs* bs, drflac_uint32 bitsPerSample, drflac_uint32 count, drflac_uint8 riceParam, drflac_uint32 lpcOrder, drflac_int32 lpcShift, drflac_uint32 lpcPrecision, const drflac_int32* coefficients, drflac_int32* pSamplesOut)
{
    DRFLAC_ASSERT(bs != NULL);
    DRFLAC_ASSERT(pSamplesOut != NULL);

    /* In my testing the order is rarely > 12, so in this case I'm going to simplify the NEON implementation by only handling order <= 12. */
    if (lpcOrder > 0 && lpcOrder <= 12) {
        if (drflac__use_64_bit_prediction(bitsPerSample, lpcOrder, lpcPrecision)) {
            return drflac__decode_samples_with_residual__rice__neon_64(bs, count, riceParam, lpcOrder, lpcShift, coefficients, pSamplesOut);
        } else {
            return drflac__decode_samples_with_residual__rice__neon_32(bs, count, riceParam, lpcOrder, lpcShift, coefficients, pSamplesOut);
        }
    } else {
        return drflac__decode_samples_with_residual__rice__scalar(bs, bitsPerSample, count, riceParam, lpcOrder, lpcShift, lpcPrecision, coefficients, pSamplesOut);
    }
}
#endif

static drflac_bool32 drflac__decode_samples_with_residual__rice(drflac_bs* bs, drflac_uint32 bitsPerSample, drflac_uint32 count, drflac_uint8 riceParam, drflac_uint32 lpcOrder, drflac_int32 lpcShift, drflac_uint32 lpcPrecision, const drflac_int32* coefficients, drflac_int32* pSamplesOut)
{
#if defined(DRFLAC_SUPPORT_SSE41)
    if (drflac__gIsSSE41Supported) {
        return drflac__decode_samples_with_residual__rice__sse41(bs, bitsPerSample, count, riceParam, lpcOrder, lpcShift, lpcPrecision, coefficients, pSamplesOut);
    } else
#elif defined(DRFLAC_SUPPORT_NEON)
    if (drflac__gIsNEONSupported) {
        return drflac__decode_samples_with_residual__rice__neon(bs, bitsPerSample, count, riceParam, lpcOrder, lpcShift, lpcPrecision, coefficients, pSamplesOut);
    } else
#endif
    {
        /* Scalar fallback. */
    #if 0
        return drflac__decode_samples_with_residual__rice__reference(bs, bitsPerSample, count, riceParam, lpcOrder, lpcShift, lpcPrecision, coefficients, pSamplesOut);
    #else
        return drflac__decode_samples_with_residual__rice__scalar(bs, bitsPerSample, count, riceParam, lpcOrder, lpcShift, lpcPrecision, coefficients, pSamplesOut);
    #endif
    }
}

/* Reads and seeks past a string of residual values as Rice codes. The decoder should be sitting on the first bit of the Rice codes. */
static drflac_bool32 drflac__read_and_seek_residual__rice(drflac_bs* bs, drflac_uint32 count, drflac_uint8 riceParam)
{
    drflac_uint32 i;

    DRFLAC_ASSERT(bs != NULL);

    for (i = 0; i < count; ++i) {
        if (!drflac__seek_rice_parts(bs, riceParam)) {
            return DRFLAC_FALSE;
        }
    }

    return DRFLAC_TRUE;
}

#if defined(__clang__)
__attribute__((no_sanitize("signed-integer-overflow")))
#endif
static drflac_bool32 drflac__decode_samples_with_residual__unencoded(drflac_bs* bs, drflac_uint32 bitsPerSample, drflac_uint32 count, drflac_uint8 unencodedBitsPerSample, drflac_uint32 lpcOrder, drflac_int32 lpcShift, drflac_uint32 lpcPrecision, const drflac_int32* coefficients, drflac_int32* pSamplesOut)
{
    drflac_uint32 i;

    DRFLAC_ASSERT(bs != NULL);
    DRFLAC_ASSERT(unencodedBitsPerSample <= 31);    /* <-- unencodedBitsPerSample is a 5 bit number, so cannot exceed 31. */
    DRFLAC_ASSERT(pSamplesOut != NULL);

    for (i = 0; i < count; ++i) {
        if (unencodedBitsPerSample > 0) {
            if (!drflac__read_int32(bs, unencodedBitsPerSample, pSamplesOut + i)) {
                return DRFLAC_FALSE;
            }
        } else {
            pSamplesOut[i] = 0;
        }

        if (drflac__use_64_bit_prediction(bitsPerSample, lpcOrder, lpcPrecision)) {
            pSamplesOut[i] += drflac__calculate_prediction_64(lpcOrder, lpcShift, coefficients, pSamplesOut + i);
        } else {
            pSamplesOut[i] += drflac__calculate_prediction_32(lpcOrder, lpcShift, coefficients, pSamplesOut + i);
        }
    }

    return DRFLAC_TRUE;
}


/*
Reads and decodes the residual for the sub-frame the decoder is currently sitting on. This function should be called
when the decoder is sitting at the very start of the RESIDUAL block. The first <order> residuals will be ignored. The
<blockSize> and <order> parameters are used to determine how many residual values need to be decoded.
*/
static drflac_bool32 drflac__decode_samples_with_residual(drflac_bs* bs, drflac_uint32 bitsPerSample, drflac_uint32 blockSize, drflac_uint32 lpcOrder, drflac_int32 lpcShift, drflac_uint32 lpcPrecision, const drflac_int32* coefficients, drflac_int32* pDecodedSamples)
{
    drflac_uint8 residualMethod;
    drflac_uint8 partitionOrder;
    drflac_uint32 samplesInPartition;
    drflac_uint32 partitionsRemaining;

    DRFLAC_ASSERT(bs != NULL);
    DRFLAC_ASSERT(blockSize != 0);
    DRFLAC_ASSERT(pDecodedSamples != NULL);       /* <-- Should we allow NULL, in which case we just seek past the residual rather than do a full decode? */

    if (!drflac__read_uint8(bs, 2, &residualMethod)) {
        return DRFLAC_FALSE;
    }

    if (residualMethod != DRFLAC_RESIDUAL_CODING_METHOD_PARTITIONED_RICE && residualMethod != DRFLAC_RESIDUAL_CODING_METHOD_PARTITIONED_RICE2) {
        return DRFLAC_FALSE;    /* Unknown or unsupported residual coding method. */
    }

    /* Ignore the first <order> values. */
    pDecodedSamples += lpcOrder;

    if (!drflac__read_uint8(bs, 4, &partitionOrder)) {
        return DRFLAC_FALSE;
    }

    /*
    From the FLAC spec:
      The Rice partition order in a Rice-coded residual section must be less than or equal to 8.
    */
    if (partitionOrder > 8) {
        return DRFLAC_FALSE;
    }

    /* Validation check. */
    if ((blockSize / (1 << partitionOrder)) < lpcOrder) {
        return DRFLAC_FALSE;
    }

    samplesInPartition = (blockSize / (1 << partitionOrder)) - lpcOrder;
    partitionsRemaining = (1 << partitionOrder);
    for (;;) {
        drflac_uint8 riceParam = 0;
        if (residualMethod == DRFLAC_RESIDUAL_CODING_METHOD_PARTITIONED_RICE) {
            if (!drflac__read_uint8(bs, 4, &riceParam)) {
                return DRFLAC_FALSE;
            }
            if (riceParam == 15) {
                riceParam = 0xFF;
            }
        } else if (residualMethod == DRFLAC_RESIDUAL_CODING_METHOD_PARTITIONED_RICE2) {
            if (!drflac__read_uint8(bs, 5, &riceParam)) {
                return DRFLAC_FALSE;
            }
            if (riceParam == 31) {
                riceParam = 0xFF;
            }
        }

        if (riceParam != 0xFF) {
            if (!drflac__decode_samples_with_residual__rice(bs, bitsPerSample, samplesInPartition, riceParam, lpcOrder, lpcShift, lpcPrecision, coefficients, pDecodedSamples)) {
                return DRFLAC_FALSE;
            }
        } else {
            drflac_uint8 unencodedBitsPerSample = 0;
            if (!drflac__read_uint8(bs, 5, &unencodedBitsPerSample)) {
                return DRFLAC_FALSE;
            }

            if (!drflac__decode_samples_with_residual__unencoded(bs, bitsPerSample, samplesInPartition, unencodedBitsPerSample, lpcOrder, lpcShift, lpcPrecision, coefficients, pDecodedSamples)) {
                return DRFLAC_FALSE;
            }
        }

        pDecodedSamples += samplesInPartition;

        if (partitionsRemaining == 1) {
            break;
        }

        partitionsRemaining -= 1;

        if (partitionOrder != 0) {
            samplesInPartition = blockSize / (1 << partitionOrder);
        }
    }

    return DRFLAC_TRUE;
}

/*
Reads and seeks past the residual for the sub-frame the decoder is currently sitting on. This function should be called
when the decoder is sitting at the very start of the RESIDUAL block. The first <order> residuals will be set to 0. The
<blockSize> and <order> parameters are used to determine how many residual values need to be decoded.
*/
static drflac_bool32 drflac__read_and_seek_residual(drflac_bs* bs, drflac_uint32 blockSize, drflac_uint32 order)
{
    drflac_uint8 residualMethod;
    drflac_uint8 partitionOrder;
    drflac_uint32 samplesInPartition;
    drflac_uint32 partitionsRemaining;

    DRFLAC_ASSERT(bs != NULL);
    DRFLAC_ASSERT(blockSize != 0);

    if (!drflac__read_uint8(bs, 2, &residualMethod)) {
        return DRFLAC_FALSE;
    }

    if (residualMethod != DRFLAC_RESIDUAL_CODING_METHOD_PARTITIONED_RICE && residualMethod != DRFLAC_RESIDUAL_CODING_METHOD_PARTITIONED_RICE2) {
        return DRFLAC_FALSE;    /* Unknown or unsupported residual coding method. */
    }

    if (!drflac__read_uint8(bs, 4, &partitionOrder)) {
        return DRFLAC_FALSE;
    }

    /*
    From the FLAC spec:
      The Rice partition order in a Rice-coded residual section must be less than or equal to 8.
    */
    if (partitionOrder > 8) {
        return DRFLAC_FALSE;
    }

    /* Validation check. */
    if ((blockSize / (1 << partitionOrder)) <= order) {
        return DRFLAC_FALSE;
    }

    samplesInPartition = (blockSize / (1 << partitionOrder)) - order;
    partitionsRemaining = (1 << partitionOrder);
    for (;;)
    {
        drflac_uint8 riceParam = 0;
        if (residualMethod == DRFLAC_RESIDUAL_CODING_METHOD_PARTITIONED_RICE) {
            if (!drflac__read_uint8(bs, 4, &riceParam)) {
                return DRFLAC_FALSE;
            }
            if (riceParam == 15) {
                riceParam = 0xFF;
            }
        } else if (residualMethod == DRFLAC_RESIDUAL_CODING_METHOD_PARTITIONED_RICE2) {
            if (!drflac__read_uint8(bs, 5, &riceParam)) {
                return DRFLAC_FALSE;
            }
            if (riceParam == 31) {
                riceParam = 0xFF;
            }
        }

        if (riceParam != 0xFF) {
            if (!drflac__read_and_seek_residual__rice(bs, samplesInPartition, riceParam)) {
                return DRFLAC_FALSE;
            }
        } else {
            drflac_uint8 unencodedBitsPerSample = 0;
            if (!drflac__read_uint8(bs, 5, &unencodedBitsPerSample)) {
                return DRFLAC_FALSE;
            }

            if (!drflac__seek_bits(bs, unencodedBitsPerSample * samplesInPartition)) {
                return DRFLAC_FALSE;
            }
        }


        if (partitionsRemaining == 1) {
            break;
        }

        partitionsRemaining -= 1;
        samplesInPartition = blockSize / (1 << partitionOrder);
    }

    return DRFLAC_TRUE;
}


static drflac_bool32 drflac__decode_samples__constant(drflac_bs* bs, drflac_uint32 blockSize, drflac_uint32 subframeBitsPerSample, drflac_int32* pDecodedSamples)
{
    drflac_uint32 i;

    /* Only a single sample needs to be decoded here. */
    drflac_int32 sample;
    if (!drflac__read_int32(bs, subframeBitsPerSample, &sample)) {
        return DRFLAC_FALSE;
    }

    /*
    We don't really need to expand this, but it does simplify the process of reading samples. If this becomes a performance issue (unlikely)
    we'll want to look at a more efficient way.
    */
    for (i = 0; i < blockSize; ++i) {
        pDecodedSamples[i] = sample;
    }

    return DRFLAC_TRUE;
}

static drflac_bool32 drflac__decode_samples__verbatim(drflac_bs* bs, drflac_uint32 blockSize, drflac_uint32 subframeBitsPerSample, drflac_int32* pDecodedSamples)
{
    drflac_uint32 i;

    for (i = 0; i < blockSize; ++i) {
        drflac_int32 sample;
        if (!drflac__read_int32(bs, subframeBitsPerSample, &sample)) {
            return DRFLAC_FALSE;
        }

        pDecodedSamples[i] = sample;
    }

    return DRFLAC_TRUE;
}

static drflac_bool32 drflac__decode_samples__fixed(drflac_bs* bs, drflac_uint32 blockSize, drflac_uint32 subframeBitsPerSample, drflac_uint8 lpcOrder, drflac_int32* pDecodedSamples)
{
    drflac_uint32 i;

    static drflac_int32 lpcCoefficientsTable[5][4] = {
        {0,  0, 0,  0},
        {1,  0, 0,  0},
        {2, -1, 0,  0},
        {3, -3, 1,  0},
        {4, -6, 4, -1}
    };

    /* Warm up samples and coefficients. */
    for (i = 0; i < lpcOrder; ++i) {
        drflac_int32 sample;
        if (!drflac__read_int32(bs, subframeBitsPerSample, &sample)) {
            return DRFLAC_FALSE;
        }

        pDecodedSamples[i] = sample;
    }

    if (!drflac__decode_samples_with_residual(bs, subframeBitsPerSample, blockSize, lpcOrder, 0, 4, lpcCoefficientsTable[lpcOrder], pDecodedSamples)) {
        return DRFLAC_FALSE;
    }

    return DRFLAC_TRUE;
}

static drflac_bool32 drflac__decode_samples__lpc(drflac_bs* bs, drflac_uint32 blockSize, drflac_uint32 bitsPerSample, drflac_uint8 lpcOrder, drflac_int32* pDecodedSamples)
{
    drflac_uint8 i;
    drflac_uint8 lpcPrecision;
    drflac_int8 lpcShift;
    drflac_int32 coefficients[32];

    /* Warm up samples. */
    for (i = 0; i < lpcOrder; ++i) {
        drflac_int32 sample;
        if (!drflac__read_int32(bs, bitsPerSample, &sample)) {
            return DRFLAC_FALSE;
        }

        pDecodedSamples[i] = sample;
    }

    if (!drflac__read_uint8(bs, 4, &lpcPrecision)) {
        return DRFLAC_FALSE;
    }
    if (lpcPrecision == 15) {
        return DRFLAC_FALSE;    /* Invalid. */
    }
    lpcPrecision += 1;

    if (!drflac__read_int8(bs, 5, &lpcShift)) {
        return DRFLAC_FALSE;
    }

    /*
    From the FLAC specification:

        Quantized linear predictor coefficient shift needed in bits (NOTE: this number is signed two's-complement)

    Emphasis on the "signed two's-complement". In practice there does not seem to be any encoders nor decoders supporting negative shifts. For now dr_flac is
    not going to support negative shifts as I don't have any reference files. However, when a reference file comes through I will consider adding support.
    */
    if (lpcShift < 0) {
        return DRFLAC_FALSE;
    }

    DRFLAC_ZERO_MEMORY(coefficients, sizeof(coefficients));
    for (i = 0; i < lpcOrder; ++i) {
        if (!drflac__read_int32(bs, lpcPrecision, coefficients + i)) {
            return DRFLAC_FALSE;
        }
    }

    if (!drflac__decode_samples_with_residual(bs, bitsPerSample, blockSize, lpcOrder, lpcShift, lpcPrecision, coefficients, pDecodedSamples)) {
        return DRFLAC_FALSE;
    }

    return DRFLAC_TRUE;
}


static drflac_bool32 drflac__read_next_flac_frame_header(drflac_bs* bs, drflac_uint8 streaminfoBitsPerSample, drflac_frame_header* header)
{
    const drflac_uint32 sampleRateTable[12]  = {0, 88200, 176400, 192000, 8000, 16000, 22050, 24000, 32000, 44100, 48000, 96000};
    const drflac_uint8 bitsPerSampleTable[8] = {0, 8, 12, (drflac_uint8)-1, 16, 20, 24, (drflac_uint8)-1};   /* -1 = reserved. */

    DRFLAC_ASSERT(bs != NULL);
    DRFLAC_ASSERT(header != NULL);

    /* Keep looping until we find a valid sync code. */
    for (;;) {
        drflac_uint8 crc8 = 0xCE; /* 0xCE = drflac_crc8(0, 0x3FFE, 14); */
        drflac_uint8 reserved = 0;
        drflac_uint8 blockingStrategy = 0;
        drflac_uint8 blockSize = 0;
        drflac_uint8 sampleRate = 0;
        drflac_uint8 channelAssignment = 0;
        drflac_uint8 bitsPerSample = 0;
        drflac_bool32 isVariableBlockSize;

        if (!drflac__find_and_seek_to_next_sync_code(bs)) {
            return DRFLAC_FALSE;
        }

        if (!drflac__read_uint8(bs, 1, &reserved)) {
            return DRFLAC_FALSE;
        }
        if (reserved == 1) {
            continue;
        }
        crc8 = drflac_crc8(crc8, reserved, 1);

        if (!drflac__read_uint8(bs, 1, &blockingStrategy)) {
            return DRFLAC_FALSE;
        }
        crc8 = drflac_crc8(crc8, blockingStrategy, 1);

        if (!drflac__read_uint8(bs, 4, &blockSize)) {
            return DRFLAC_FALSE;
        }
        if (blockSize == 0) {
            continue;
        }
        crc8 = drflac_crc8(crc8, blockSize, 4);

        if (!drflac__read_uint8(bs, 4, &sampleRate)) {
            return DRFLAC_FALSE;
        }
        crc8 = drflac_crc8(crc8, sampleRate, 4);

        if (!drflac__read_uint8(bs, 4, &channelAssignment)) {
            return DRFLAC_FALSE;
        }
        if (channelAssignment > 10) {
            continue;
        }
        crc8 = drflac_crc8(crc8, channelAssignment, 4);

        if (!drflac__read_uint8(bs, 3, &bitsPerSample)) {
            return DRFLAC_FALSE;
        }
        if (bitsPerSample == 3 || bitsPerSample == 7) {
            continue;
        }
        crc8 = drflac_crc8(crc8, bitsPerSample, 3);


        if (!drflac__read_uint8(bs, 1, &reserved)) {
            return DRFLAC_FALSE;
        }
        if (reserved == 1) {
            continue;
        }
        crc8 = drflac_crc8(crc8, reserved, 1);


        isVariableBlockSize = blockingStrategy == 1;
        if (isVariableBlockSize) {
            drflac_uint64 pcmFrameNumber;
            drflac_result result = drflac__read_utf8_coded_number(bs, &pcmFrameNumber, &crc8);
            if (result != DRFLAC_SUCCESS) {
                if (result == DRFLAC_AT_END) {
                    return DRFLAC_FALSE;
                } else {
                    continue;
                }
            }
            header->flacFrameNumber  = 0;
            header->pcmFrameNumber = pcmFrameNumber;
        } else {
            drflac_uint64 flacFrameNumber = 0;
            drflac_result result = drflac__read_utf8_coded_number(bs, &flacFrameNumber, &crc8);
            if (result != DRFLAC_SUCCESS) {
                if (result == DRFLAC_AT_END) {
                    return DRFLAC_FALSE;
                } else {
                    continue;
                }
            }
            header->flacFrameNumber  = (drflac_uint32)flacFrameNumber;   /* <-- Safe cast. */
            header->pcmFrameNumber = 0;
        }


        DRFLAC_ASSERT(blockSize > 0);
        if (blockSize == 1) {
            header->blockSizeInPCMFrames = 192;
        } else if (blockSize <= 5) {
            DRFLAC_ASSERT(blockSize >= 2);
            header->blockSizeInPCMFrames = 576 * (1 << (blockSize - 2));
        } else if (blockSize == 6) {
            if (!drflac__read_uint16(bs, 8, &header->blockSizeInPCMFrames)) {
                return DRFLAC_FALSE;
            }
            crc8 = drflac_crc8(crc8, header->blockSizeInPCMFrames, 8);
            header->blockSizeInPCMFrames += 1;
        } else if (blockSize == 7) {
            if (!drflac__read_uint16(bs, 16, &header->blockSizeInPCMFrames)) {
                return DRFLAC_FALSE;
            }
            crc8 = drflac_crc8(crc8, header->blockSizeInPCMFrames, 16);
            if (header->blockSizeInPCMFrames == 0xFFFF) {
                return DRFLAC_FALSE;    /* Frame is too big. This is the size of the frame minus 1. The STREAMINFO block defines the max block size which is 16-bits. Adding one will make it 17 bits and therefore too big. */
            }
            header->blockSizeInPCMFrames += 1;
        } else {
            DRFLAC_ASSERT(blockSize >= 8);
            header->blockSizeInPCMFrames = 256 * (1 << (blockSize - 8));
        }


        if (sampleRate <= 11) {
            header->sampleRate = sampleRateTable[sampleRate];
        } else if (sampleRate == 12) {
            if (!drflac__read_uint32(bs, 8, &header->sampleRate)) {
                return DRFLAC_FALSE;
            }
            crc8 = drflac_crc8(crc8, header->sampleRate, 8);
            header->sampleRate *= 1000;
        } else if (sampleRate == 13) {
            if (!drflac__read_uint32(bs, 16, &header->sampleRate)) {
                return DRFLAC_FALSE;
            }
            crc8 = drflac_crc8(crc8, header->sampleRate, 16);
        } else if (sampleRate == 14) {
            if (!drflac__read_uint32(bs, 16, &header->sampleRate)) {
                return DRFLAC_FALSE;
            }
            crc8 = drflac_crc8(crc8, header->sampleRate, 16);
            header->sampleRate *= 10;
        } else {
            continue;  /* Invalid. Assume an invalid block. */
        }


        header->channelAssignment = channelAssignment;

        header->bitsPerSample = bitsPerSampleTable[bitsPerSample];
        if (header->bitsPerSample == 0) {
            header->bitsPerSample = streaminfoBitsPerSample;
        }

        if (header->bitsPerSample != streaminfoBitsPerSample) {
            /* If this subframe has a different bitsPerSample then streaminfo or the first frame, reject it */
            return DRFLAC_FALSE;
        }

        if (!drflac__read_uint8(bs, 8, &header->crc8)) {
            return DRFLAC_FALSE;
        }

#ifndef DR_FLAC_NO_CRC
        if (header->crc8 != crc8) {
            continue;    /* CRC mismatch. Loop back to the top and find the next sync code. */
        }
#endif
        return DRFLAC_TRUE;
    }
}

static drflac_bool32 drflac__read_subframe_header(drflac_bs* bs, drflac_subframe* pSubframe)
{
    drflac_uint8 header;
    int type;

    if (!drflac__read_uint8(bs, 8, &header)) {
        return DRFLAC_FALSE;
    }

    /* First bit should always be 0. */
    if ((header & 0x80) != 0) {
        return DRFLAC_FALSE;
    }

    type = (header & 0x7E) >> 1;
    if (type == 0) {
        pSubframe->subframeType = DRFLAC_SUBFRAME_CONSTANT;
    } else if (type == 1) {
        pSubframe->subframeType = DRFLAC_SUBFRAME_VERBATIM;
    } else {
        if ((type & 0x20) != 0) {
            pSubframe->subframeType = DRFLAC_SUBFRAME_LPC;
            pSubframe->lpcOrder = (drflac_uint8)(type & 0x1F) + 1;
        } else if ((type & 0x08) != 0) {
            pSubframe->subframeType = DRFLAC_SUBFRAME_FIXED;
            pSubframe->lpcOrder = (drflac_uint8)(type & 0x07);
            if (pSubframe->lpcOrder > 4) {
                pSubframe->subframeType = DRFLAC_SUBFRAME_RESERVED;
                pSubframe->lpcOrder = 0;
            }
        } else {
            pSubframe->subframeType = DRFLAC_SUBFRAME_RESERVED;
        }
    }

    if (pSubframe->subframeType == DRFLAC_SUBFRAME_RESERVED) {
        return DRFLAC_FALSE;
    }

    /* Wasted bits per sample. */
    pSubframe->wastedBitsPerSample = 0;
    if ((header & 0x01) == 1) {
        unsigned int wastedBitsPerSample;
        if (!drflac__seek_past_next_set_bit(bs, &wastedBitsPerSample)) {
            return DRFLAC_FALSE;
        }
        pSubframe->wastedBitsPerSample = (drflac_uint8)wastedBitsPerSample + 1;
    }

    return DRFLAC_TRUE;
}

static drflac_bool32 drflac__decode_subframe(drflac_bs* bs, drflac_frame* frame, int subframeIndex, drflac_int32* pDecodedSamplesOut)
{
    drflac_subframe* pSubframe;
    drflac_uint32 subframeBitsPerSample;

    DRFLAC_ASSERT(bs != NULL);
    DRFLAC_ASSERT(frame != NULL);

    pSubframe = frame->subframes + subframeIndex;
    if (!drflac__read_subframe_header(bs, pSubframe)) {
        return DRFLAC_FALSE;
    }

    /* Side channels require an extra bit per sample. Took a while to figure that one out... */
    subframeBitsPerSample = frame->header.bitsPerSample;
    if ((frame->header.channelAssignment == DRFLAC_CHANNEL_ASSIGNMENT_LEFT_SIDE || frame->header.channelAssignment == DRFLAC_CHANNEL_ASSIGNMENT_MID_SIDE) && subframeIndex == 1) {
        subframeBitsPerSample += 1;
    } else if (frame->header.channelAssignment == DRFLAC_CHANNEL_ASSIGNMENT_RIGHT_SIDE && subframeIndex == 0) {
        subframeBitsPerSample += 1;
    }

    if (subframeBitsPerSample > 32) {
        /* libFLAC and ffmpeg reject 33-bit subframes as well */
        return DRFLAC_FALSE;
    }

    /* Need to handle wasted bits per sample. */
    if (pSubframe->wastedBitsPerSample >= subframeBitsPerSample) {
        return DRFLAC_FALSE;
    }
    subframeBitsPerSample -= pSubframe->wastedBitsPerSample;

    pSubframe->pSamplesS32 = pDecodedSamplesOut;

    switch (pSubframe->subframeType)
    {
        case DRFLAC_SUBFRAME_CONSTANT:
        {
            drflac__decode_samples__constant(bs, frame->header.blockSizeInPCMFrames, subframeBitsPerSample, pSubframe->pSamplesS32);
        } break;

        case DRFLAC_SUBFRAME_VERBATIM:
        {
            drflac__decode_samples__verbatim(bs, frame->header.blockSizeInPCMFrames, subframeBitsPerSample, pSubframe->pSamplesS32);
        } break;

        case DRFLAC_SUBFRAME_FIXED:
        {
            drflac__decode_samples__fixed(bs, frame->header.blockSizeInPCMFrames, subframeBitsPerSample, pSubframe->lpcOrder, pSubframe->pSamplesS32);
        } break;

        case DRFLAC_SUBFRAME_LPC:
        {
            drflac__decode_samples__lpc(bs, frame->header.blockSizeInPCMFrames, subframeBitsPerSample, pSubframe->lpcOrder, pSubframe->pSamplesS32);
        } break;

        default: return DRFLAC_FALSE;
    }

    return DRFLAC_TRUE;
}

static drflac_bool32 drflac__seek_subframe(drflac_bs* bs, drflac_frame* frame, int subframeIndex)
{
    drflac_subframe* pSubframe;
    drflac_uint32 subframeBitsPerSample;

    DRFLAC_ASSERT(bs != NULL);
    DRFLAC_ASSERT(frame != NULL);

    pSubframe = frame->subframes + subframeIndex;
    if (!drflac__read_subframe_header(bs, pSubframe)) {
        return DRFLAC_FALSE;
    }

    /* Side channels require an extra bit per sample. Took a while to figure that one out... */
    subframeBitsPerSample = frame->header.bitsPerSample;
    if ((frame->header.channelAssignment == DRFLAC_CHANNEL_ASSIGNMENT_LEFT_SIDE || frame->header.channelAssignment == DRFLAC_CHANNEL_ASSIGNMENT_MID_SIDE) && subframeIndex == 1) {
        subframeBitsPerSample += 1;
    } else if (frame->header.channelAssignment == DRFLAC_CHANNEL_ASSIGNMENT_RIGHT_SIDE && subframeIndex == 0) {
        subframeBitsPerSample += 1;
    }

    /* Need to handle wasted bits per sample. */
    if (pSubframe->wastedBitsPerSample >= subframeBitsPerSample) {
        return DRFLAC_FALSE;
    }
    subframeBitsPerSample -= pSubframe->wastedBitsPerSample;

    pSubframe->pSamplesS32 = NULL;

    switch (pSubframe->subframeType)
    {
        case DRFLAC_SUBFRAME_CONSTANT:
        {
            if (!drflac__seek_bits(bs, subframeBitsPerSample)) {
                return DRFLAC_FALSE;
            }
        } break;

        case DRFLAC_SUBFRAME_VERBATIM:
        {
            unsigned int bitsToSeek = frame->header.blockSizeInPCMFrames * subframeBitsPerSample;
            if (!drflac__seek_bits(bs, bitsToSeek)) {
                return DRFLAC_FALSE;
            }
        } break;

        case DRFLAC_SUBFRAME_FIXED:
        {
            unsigned int bitsToSeek = pSubframe->lpcOrder * subframeBitsPerSample;
            if (!drflac__seek_bits(bs, bitsToSeek)) {
                return DRFLAC_FALSE;
            }

            if (!drflac__read_and_seek_residual(bs, frame->header.blockSizeInPCMFrames, pSubframe->lpcOrder)) {
                return DRFLAC_FALSE;
            }
        } break;

        case DRFLAC_SUBFRAME_LPC:
        {
            drflac_uint8 lpcPrecision;

            unsigned int bitsToSeek = pSubframe->lpcOrder * subframeBitsPerSample;
            if (!drflac__seek_bits(bs, bitsToSeek)) {
                return DRFLAC_FALSE;
            }

            if (!drflac__read_uint8(bs, 4, &lpcPrecision)) {
                return DRFLAC_FALSE;
            }
            if (lpcPrecision == 15) {
                return DRFLAC_FALSE;    /* Invalid. */
            }
            lpcPrecision += 1;


            bitsToSeek = (pSubframe->lpcOrder * lpcPrecision) + 5;    /* +5 for shift. */
            if (!drflac__seek_bits(bs, bitsToSeek)) {
                return DRFLAC_FALSE;
            }

            if (!drflac__read_and_seek_residual(bs, frame->header.blockSizeInPCMFrames, pSubframe->lpcOrder)) {
                return DRFLAC_FALSE;
            }
        } break;

        default: return DRFLAC_FALSE;
    }

    return DRFLAC_TRUE;
}


static DRFLAC_INLINE drflac_uint8 drflac__get_channel_count_from_channel_assignment(drflac_int8 channelAssignment)
{
    drflac_uint8 lookup[] = {1, 2, 3, 4, 5, 6, 7, 8, 2, 2, 2};

    DRFLAC_ASSERT(channelAssignment <= 10);
    return lookup[channelAssignment];
}

static drflac_result drflac__decode_flac_frame(drflac* pFlac)
{
    int channelCount;
    int i;
    drflac_uint8 paddingSizeInBits;
    drflac_uint16 desiredCRC16;
#ifndef DR_FLAC_NO_CRC
    drflac_uint16 actualCRC16;
#endif

    /* This function should be called while the stream is sitting on the first byte after the frame header. */
    DRFLAC_ZERO_MEMORY(pFlac->currentFLACFrame.subframes, sizeof(pFlac->currentFLACFrame.subframes));

    /* The frame block size must never be larger than the maximum block size defined by the FLAC stream. */
    if (pFlac->currentFLACFrame.header.blockSizeInPCMFrames > pFlac->maxBlockSizeInPCMFrames) {
        return DRFLAC_ERROR;
    }

    /* The number of channels in the frame must match the channel count from the STREAMINFO block. */
    channelCount = drflac__get_channel_count_from_channel_assignment(pFlac->currentFLACFrame.header.channelAssignment);
    if (channelCount != (int)pFlac->channels) {
        return DRFLAC_ERROR;
    }

    for (i = 0; i < channelCount; ++i) {
        if (!drflac__decode_subframe(&pFlac->bs, &pFlac->currentFLACFrame, i, pFlac->pDecodedSamples + (pFlac->currentFLACFrame.header.blockSizeInPCMFrames * i))) {
            return DRFLAC_ERROR;
        }
    }

    paddingSizeInBits = (drflac_uint8)(DRFLAC_CACHE_L1_BITS_REMAINING(&pFlac->bs) & 7);
    if (paddingSizeInBits > 0) {
        drflac_uint8 padding = 0;
        if (!drflac__read_uint8(&pFlac->bs, paddingSizeInBits, &padding)) {
            return DRFLAC_AT_END;
        }
    }

#ifndef DR_FLAC_NO_CRC
    actualCRC16 = drflac__flush_crc16(&pFlac->bs);
#endif
    if (!drflac__read_uint16(&pFlac->bs, 16, &desiredCRC16)) {
        return DRFLAC_AT_END;
    }

#ifndef DR_FLAC_NO_CRC
    if (actualCRC16 != desiredCRC16) {
        return DRFLAC_CRC_MISMATCH;    /* CRC mismatch. */
    }
#endif

    pFlac->currentFLACFrame.pcmFramesRemaining = pFlac->currentFLACFrame.header.blockSizeInPCMFrames;

    return DRFLAC_SUCCESS;
}

static drflac_result drflac__seek_flac_frame(drflac* pFlac)
{
    int channelCount;
    int i;
    drflac_uint16 desiredCRC16;
#ifndef DR_FLAC_NO_CRC
    drflac_uint16 actualCRC16;
#endif

    channelCount = drflac__get_channel_count_from_channel_assignment(pFlac->currentFLACFrame.header.channelAssignment);
    for (i = 0; i < channelCount; ++i) {
        if (!drflac__seek_subframe(&pFlac->bs, &pFlac->currentFLACFrame, i)) {
            return DRFLAC_ERROR;
        }
    }

    /* Padding. */
    if (!drflac__seek_bits(&pFlac->bs, DRFLAC_CACHE_L1_BITS_REMAINING(&pFlac->bs) & 7)) {
        return DRFLAC_ERROR;
    }

    /* CRC. */
#ifndef DR_FLAC_NO_CRC
    actualCRC16 = drflac__flush_crc16(&pFlac->bs);
#endif
    if (!drflac__read_uint16(&pFlac->bs, 16, &desiredCRC16)) {
        return DRFLAC_AT_END;
    }

#ifndef DR_FLAC_NO_CRC
    if (actualCRC16 != desiredCRC16) {
        return DRFLAC_CRC_MISMATCH;    /* CRC mismatch. */
    }
#endif

    return DRFLAC_SUCCESS;
}

static drflac_bool32 drflac__read_and_decode_next_flac_frame(drflac* pFlac)
{
    DRFLAC_ASSERT(pFlac != NULL);

    for (;;) {
        drflac_result result;

        if (!drflac__read_next_flac_frame_header(&pFlac->bs, pFlac->bitsPerSample, &pFlac->currentFLACFrame.header)) {
            return DRFLAC_FALSE;
        }

        result = drflac__decode_flac_frame(pFlac);
        if (result != DRFLAC_SUCCESS) {
            if (result == DRFLAC_CRC_MISMATCH) {
                continue;   /* CRC mismatch. Skip to the next frame. */
            } else {
                return DRFLAC_FALSE;
            }
        }

        return DRFLAC_TRUE;
    }
}

static void drflac__get_pcm_frame_range_of_current_flac_frame(drflac* pFlac, drflac_uint64* pFirstPCMFrame, drflac_uint64* pLastPCMFrame)
{
    drflac_uint64 firstPCMFrame;
    drflac_uint64 lastPCMFrame;

    DRFLAC_ASSERT(pFlac != NULL);

    firstPCMFrame = pFlac->currentFLACFrame.header.pcmFrameNumber;
    if (firstPCMFrame == 0) {
        firstPCMFrame = ((drflac_uint64)pFlac->currentFLACFrame.header.flacFrameNumber) * pFlac->maxBlockSizeInPCMFrames;
    }

    lastPCMFrame = firstPCMFrame + pFlac->currentFLACFrame.header.blockSizeInPCMFrames;
    if (lastPCMFrame > 0) {
        lastPCMFrame -= 1; /* Needs to be zero based. */
    }

    if (pFirstPCMFrame) {
        *pFirstPCMFrame = firstPCMFrame;
    }
    if (pLastPCMFrame) {
        *pLastPCMFrame = lastPCMFrame;
    }
}

static drflac_bool32 drflac__seek_to_first_frame(drflac* pFlac)
{
    drflac_bool32 result;

    DRFLAC_ASSERT(pFlac != NULL);

    result = drflac__seek_to_byte(&pFlac->bs, pFlac->firstFLACFramePosInBytes);

    DRFLAC_ZERO_MEMORY(&pFlac->currentFLACFrame, sizeof(pFlac->currentFLACFrame));
    pFlac->currentPCMFrame = 0;

    return result;
}

static DRFLAC_INLINE drflac_result drflac__seek_to_next_flac_frame(drflac* pFlac)
{
    /* This function should only ever be called while the decoder is sitting on the first byte past the FRAME_HEADER section. */
    DRFLAC_ASSERT(pFlac != NULL);
    return drflac__seek_flac_frame(pFlac);
}


static drflac_uint64 drflac__seek_forward_by_pcm_frames(drflac* pFlac, drflac_uint64 pcmFramesToSeek)
{
    drflac_uint64 pcmFramesRead = 0;
    while (pcmFramesToSeek > 0) {
        if (pFlac->currentFLACFrame.pcmFramesRemaining == 0) {
            if (!drflac__read_and_decode_next_flac_frame(pFlac)) {
                break;  /* Couldn't read the next frame, so just break from the loop and return. */
            }
        } else {
            if (pFlac->currentFLACFrame.pcmFramesRemaining > pcmFramesToSeek) {
                pcmFramesRead   += pcmFramesToSeek;
                pFlac->currentFLACFrame.pcmFramesRemaining -= (drflac_uint32)pcmFramesToSeek;   /* <-- Safe cast. Will always be < currentFrame.pcmFramesRemaining < 65536. */
                pcmFramesToSeek  = 0;
            } else {
                pcmFramesRead   += pFlac->currentFLACFrame.pcmFramesRemaining;
                pcmFramesToSeek -= pFlac->currentFLACFrame.pcmFramesRemaining;
                pFlac->currentFLACFrame.pcmFramesRemaining = 0;
            }
        }
    }

    pFlac->currentPCMFrame += pcmFramesRead;
    return pcmFramesRead;
}


static drflac_bool32 drflac__seek_to_pcm_frame__brute_force(drflac* pFlac, drflac_uint64 pcmFrameIndex)
{
    drflac_bool32 isMidFrame = DRFLAC_FALSE;
    drflac_uint64 runningPCMFrameCount;

    DRFLAC_ASSERT(pFlac != NULL);

    /* If we are seeking forward we start from the current position. Otherwise we need to start all the way from the start of the file. */
    if (pcmFrameIndex >= pFlac->currentPCMFrame) {
        /* Seeking forward. Need to seek from the current position. */
        runningPCMFrameCount = pFlac->currentPCMFrame;

        /* The frame header for the first frame may not yet have been read. We need to do that if necessary. */
        if (pFlac->currentPCMFrame == 0 && pFlac->currentFLACFrame.pcmFramesRemaining == 0) {
            if (!drflac__read_next_flac_frame_header(&pFlac->bs, pFlac->bitsPerSample, &pFlac->currentFLACFrame.header)) {
                return DRFLAC_FALSE;
            }
        } else {
            isMidFrame = DRFLAC_TRUE;
        }
    } else {
        /* Seeking backwards. Need to seek from the start of the file. */
        runningPCMFrameCount = 0;

        /* Move back to the start. */
        if (!drflac__seek_to_first_frame(pFlac)) {
            return DRFLAC_FALSE;
        }

        /* Decode the first frame in preparation for sample-exact seeking below. */
        if (!drflac__read_next_flac_frame_header(&pFlac->bs, pFlac->bitsPerSample, &pFlac->currentFLACFrame.header)) {
            return DRFLAC_FALSE;
        }
    }

    /*
    We need to as quickly as possible find the frame that contains the target sample. To do this, we iterate over each frame and inspect its
    header. If based on the header we can determine that the frame contains the sample, we do a full decode of that frame.
    */
    for (;;) {
        drflac_uint64 pcmFrameCountInThisFLACFrame;
        drflac_uint64 firstPCMFrameInFLACFrame = 0;
        drflac_uint64 lastPCMFrameInFLACFrame = 0;

        drflac__get_pcm_frame_range_of_current_flac_frame(pFlac, &firstPCMFrameInFLACFrame, &lastPCMFrameInFLACFrame);

        pcmFrameCountInThisFLACFrame = (lastPCMFrameInFLACFrame - firstPCMFrameInFLACFrame) + 1;
        if (pcmFrameIndex < (runningPCMFrameCount + pcmFrameCountInThisFLACFrame)) {
            /*
            The sample should be in this frame. We need to fully decode it, however if it's an invalid frame (a CRC mismatch), we need to pretend
            it never existed and keep iterating.
            */
            drflac_uint64 pcmFramesToDecode = pcmFrameIndex - runningPCMFrameCount;

            if (!isMidFrame) {
                drflac_result result = drflac__decode_flac_frame(pFlac);
                if (result == DRFLAC_SUCCESS) {
                    /* The frame is valid. We just need to skip over some samples to ensure it's sample-exact. */
                    return drflac__seek_forward_by_pcm_frames(pFlac, pcmFramesToDecode) == pcmFramesToDecode;  /* <-- If this fails, something bad has happened (it should never fail). */
                } else {
                    if (result == DRFLAC_CRC_MISMATCH) {
                        goto next_iteration;   /* CRC mismatch. Pretend this frame never existed. */
                    } else {
                        return DRFLAC_FALSE;
                    }
                }
            } else {
                /* We started seeking mid-frame which means we need to skip the frame decoding part. */
                return drflac__seek_forward_by_pcm_frames(pFlac, pcmFramesToDecode) == pcmFramesToDecode;
            }
        } else {
            /*
            It's not in this frame. We need to seek past the frame, but check if there was a CRC mismatch. If so, we pretend this
            frame never existed and leave the running sample count untouched.
            */
            if (!isMidFrame) {
                drflac_result result = drflac__seek_to_next_flac_frame(pFlac);
                if (result == DRFLAC_SUCCESS) {
                    runningPCMFrameCount += pcmFrameCountInThisFLACFrame;
                } else {
                    if (result == DRFLAC_CRC_MISMATCH) {
                        goto next_iteration;   /* CRC mismatch. Pretend this frame never existed. */
                    } else {
                        return DRFLAC_FALSE;
                    }
                }
            } else {
                /*
                We started seeking mid-frame which means we need to seek by reading to the end of the frame instead of with
                drflac__seek_to_next_flac_frame() which only works if the decoder is sitting on the byte just after the frame header.
                */
                runningPCMFrameCount += pFlac->currentFLACFrame.pcmFramesRemaining;
                pFlac->currentFLACFrame.pcmFramesRemaining = 0;
                isMidFrame = DRFLAC_FALSE;
            }

            /* If we are seeking to the end of the file and we've just hit it, we're done. */
            if (pcmFrameIndex == pFlac->totalPCMFrameCount && runningPCMFrameCount == pFlac->totalPCMFrameCount) {
                return DRFLAC_TRUE;
            }
        }

    next_iteration:
        /* Grab the next frame in preparation for the next iteration. */
        if (!drflac__read_next_flac_frame_header(&pFlac->bs, pFlac->bitsPerSample, &pFlac->currentFLACFrame.header)) {
            return DRFLAC_FALSE;
        }
    }
}


#if !defined(DR_FLAC_NO_CRC)
/*
We use an average compression ratio to determine our approximate start location. FLAC files are generally about 50%-70% the size of their
uncompressed counterparts so we'll use this as a basis. I'm going to split the middle and use a factor of 0.6 to determine the starting
location.
*/
#define DRFLAC_BINARY_SEARCH_APPROX_COMPRESSION_RATIO 0.6f

static drflac_bool32 drflac__seek_to_approximate_flac_frame_to_byte(drflac* pFlac, drflac_uint64 targetByte, drflac_uint64 rangeLo, drflac_uint64 rangeHi, drflac_uint64* pLastSuccessfulSeekOffset)
{
    DRFLAC_ASSERT(pFlac != NULL);
    DRFLAC_ASSERT(pLastSuccessfulSeekOffset != NULL);
    DRFLAC_ASSERT(targetByte >= rangeLo);
    DRFLAC_ASSERT(targetByte <= rangeHi);

    *pLastSuccessfulSeekOffset = pFlac->firstFLACFramePosInBytes;

    for (;;) {
        /* After rangeLo == rangeHi == targetByte fails, we need to break out. */
        drflac_uint64 lastTargetByte = targetByte;

        /* When seeking to a byte, failure probably means we've attempted to seek beyond the end of the stream. To counter this we just halve it each attempt. */
        if (!drflac__seek_to_byte(&pFlac->bs, targetByte)) {
            /* If we couldn't even seek to the first byte in the stream we have a problem. Just abandon the whole thing. */
            if (targetByte == 0) {
                drflac__seek_to_first_frame(pFlac); /* Try to recover. */
                return DRFLAC_FALSE;
            }

            /* Halve the byte location and continue. */
            targetByte = rangeLo + ((rangeHi - rangeLo)/2);
            rangeHi = targetByte;
        } else {
            /* Getting here should mean that we have seeked to an appropriate byte. */

            /* Clear the details of the FLAC frame so we don't misreport data. */
            DRFLAC_ZERO_MEMORY(&pFlac->currentFLACFrame, sizeof(pFlac->currentFLACFrame));

            /*
            Now seek to the next FLAC frame. We need to decode the entire frame (not just the header) because it's possible for the header to incorrectly pass the
            CRC check and return bad data. We need to decode the entire frame to be more certain. Although this seems unlikely, this has happened to me in testing
            so it needs to stay this way for now.
            */
#if 1
            if (!drflac__read_and_decode_next_flac_frame(pFlac)) {
                /* Halve the byte location and continue. */
                targetByte = rangeLo + ((rangeHi - rangeLo)/2);
                rangeHi = targetByte;
            } else {
                break;
            }
#else
            if (!drflac__read_next_flac_frame_header(&pFlac->bs, pFlac->bitsPerSample, &pFlac->currentFLACFrame.header)) {
                /* Halve the byte location and continue. */
                targetByte = rangeLo + ((rangeHi - rangeLo)/2);
                rangeHi = targetByte;
            } else {
                break;
            }
#endif
        }

        /* We already tried this byte and there are no more to try, break out. */
        if(targetByte == lastTargetByte) {
            return DRFLAC_FALSE;
        }
    }

    /* The current PCM frame needs to be updated based on the frame we just seeked to. */
    drflac__get_pcm_frame_range_of_current_flac_frame(pFlac, &pFlac->currentPCMFrame, NULL);

    DRFLAC_ASSERT(targetByte <= rangeHi);

    *pLastSuccessfulSeekOffset = targetByte;
    return DRFLAC_TRUE;
}

static drflac_bool32 drflac__decode_flac_frame_and_seek_forward_by_pcm_frames(drflac* pFlac, drflac_uint64 offset)
{
    /* This section of code would be used if we were only decoding the FLAC frame header when calling drflac__seek_to_approximate_flac_frame_to_byte(). */
#if 0
    if (drflac__decode_flac_frame(pFlac) != DRFLAC_SUCCESS) {
        /* We failed to decode this frame which may be due to it being corrupt. We'll just use the next valid FLAC frame. */
        if (drflac__read_and_decode_next_flac_frame(pFlac) == DRFLAC_FALSE) {
            return DRFLAC_FALSE;
        }
    }
#endif

    return drflac__seek_forward_by_pcm_frames(pFlac, offset) == offset;
}


static drflac_bool32 drflac__seek_to_pcm_frame__binary_search_internal(drflac* pFlac, drflac_uint64 pcmFrameIndex, drflac_uint64 byteRangeLo, drflac_uint64 byteRangeHi)
{
    /* This assumes pFlac->currentPCMFrame is sitting on byteRangeLo upon entry. */

    drflac_uint64 targetByte;
    drflac_uint64 pcmRangeLo = pFlac->totalPCMFrameCount;
    drflac_uint64 pcmRangeHi = 0;
    drflac_uint64 lastSuccessfulSeekOffset = (drflac_uint64)-1;
    drflac_uint64 closestSeekOffsetBeforeTargetPCMFrame = byteRangeLo;
    drflac_uint32 seekForwardThreshold = (pFlac->maxBlockSizeInPCMFrames != 0) ? pFlac->maxBlockSizeInPCMFrames*2 : 4096;

    targetByte = byteRangeLo + (drflac_uint64)(((drflac_int64)((pcmFrameIndex - pFlac->currentPCMFrame) * pFlac->channels * pFlac->bitsPerSample)/8.0f) * DRFLAC_BINARY_SEARCH_APPROX_COMPRESSION_RATIO);
    if (targetByte > byteRangeHi) {
        targetByte = byteRangeHi;
    }

    for (;;) {
        if (drflac__seek_to_approximate_flac_frame_to_byte(pFlac, targetByte, byteRangeLo, byteRangeHi, &lastSuccessfulSeekOffset)) {
            /* We found a FLAC frame. We need to check if it contains the sample we're looking for. */
            drflac_uint64 newPCMRangeLo;
            drflac_uint64 newPCMRangeHi;
            drflac__get_pcm_frame_range_of_current_flac_frame(pFlac, &newPCMRangeLo, &newPCMRangeHi);

            /* If we selected the same frame, it means we should be pretty close. Just decode the rest. */
            if (pcmRangeLo == newPCMRangeLo) {
                if (!drflac__seek_to_approximate_flac_frame_to_byte(pFlac, closestSeekOffsetBeforeTargetPCMFrame, closestSeekOffsetBeforeTargetPCMFrame, byteRangeHi, &lastSuccessfulSeekOffset)) {
                    break;  /* Failed to seek to closest frame. */
                }

                if (drflac__decode_flac_frame_and_seek_forward_by_pcm_frames(pFlac, pcmFrameIndex - pFlac->currentPCMFrame)) {
                    return DRFLAC_TRUE;
                } else {
                    break;  /* Failed to seek forward. */
                }
            }

            pcmRangeLo = newPCMRangeLo;
            pcmRangeHi = newPCMRangeHi;

            if (pcmRangeLo <= pcmFrameIndex && pcmRangeHi >= pcmFrameIndex) {
                /* The target PCM frame is in this FLAC frame. */
                if (drflac__decode_flac_frame_and_seek_forward_by_pcm_frames(pFlac, pcmFrameIndex - pFlac->currentPCMFrame) ) {
                    return DRFLAC_TRUE;
                } else {
                    break;  /* Failed to seek to FLAC frame. */
                }
            } else {
                const float approxCompressionRatio = (drflac_int64)(lastSuccessfulSeekOffset - pFlac->firstFLACFramePosInBytes) / ((drflac_int64)(pcmRangeLo * pFlac->channels * pFlac->bitsPerSample)/8.0f);

                if (pcmRangeLo > pcmFrameIndex) {
                    /* We seeked too far forward. We need to move our target byte backward and try again. */
                    byteRangeHi = lastSuccessfulSeekOffset;
                    if (byteRangeLo > byteRangeHi) {
                        byteRangeLo = byteRangeHi;
                    }

                    targetByte = byteRangeLo + ((byteRangeHi - byteRangeLo) / 2);
                    if (targetByte < byteRangeLo) {
                        targetByte = byteRangeLo;
                    }
                } else /*if (pcmRangeHi < pcmFrameIndex)*/ {
                    /* We didn't seek far enough. We need to move our target byte forward and try again. */

                    /* If we're close enough we can just seek forward. */
                    if ((pcmFrameIndex - pcmRangeLo) < seekForwardThreshold) {
                        if (drflac__decode_flac_frame_and_seek_forward_by_pcm_frames(pFlac, pcmFrameIndex - pFlac->currentPCMFrame)) {
                            return DRFLAC_TRUE;
                        } else {
                            break;  /* Failed to seek to FLAC frame. */
                        }
                    } else {
                        byteRangeLo = lastSuccessfulSeekOffset;
                        if (byteRangeHi < byteRangeLo) {
                            byteRangeHi = byteRangeLo;
                        }

                        targetByte = lastSuccessfulSeekOffset + (drflac_uint64)(((drflac_int64)((pcmFrameIndex-pcmRangeLo) * pFlac->channels * pFlac->bitsPerSample)/8.0f) * approxCompressionRatio);
                        if (targetByte > byteRangeHi) {
                            targetByte = byteRangeHi;
                        }

                        if (closestSeekOffsetBeforeTargetPCMFrame < lastSuccessfulSeekOffset) {
                            closestSeekOffsetBeforeTargetPCMFrame = lastSuccessfulSeekOffset;
                        }
                    }
                }
            }
        } else {
            /* Getting here is really bad. We just recover as best we can, but moving to the first frame in the stream, and then abort. */
            break;
        }
    }

    drflac__seek_to_first_frame(pFlac); /* <-- Try to recover. */
    return DRFLAC_FALSE;
}

static drflac_bool32 drflac__seek_to_pcm_frame__binary_search(drflac* pFlac, drflac_uint64 pcmFrameIndex)
{
    drflac_uint64 byteRangeLo;
    drflac_uint64 byteRangeHi;
    drflac_uint32 seekForwardThreshold = (pFlac->maxBlockSizeInPCMFrames != 0) ? pFlac->maxBlockSizeInPCMFrames*2 : 4096;

    /* Our algorithm currently assumes the FLAC stream is currently sitting at the start. */
    if (drflac__seek_to_first_frame(pFlac) == DRFLAC_FALSE) {
        return DRFLAC_FALSE;
    }

    /* If we're close enough to the start, just move to the start and seek forward. */
    if (pcmFrameIndex < seekForwardThreshold) {
        return drflac__seek_forward_by_pcm_frames(pFlac, pcmFrameIndex) == pcmFrameIndex;
    }

    /*
    Our starting byte range is the byte position of the first FLAC frame and the approximate end of the file as if it were completely uncompressed. This ensures
    the entire file is included, even though most of the time it'll exceed the end of the actual stream. This is OK as the frame searching logic will handle it.
    */
    byteRangeLo = pFlac->firstFLACFramePosInBytes;
    byteRangeHi = pFlac->firstFLACFramePosInBytes + (drflac_uint64)((drflac_int64)(pFlac->totalPCMFrameCount * pFlac->channels * pFlac->bitsPerSample)/8.0f);

    return drflac__seek_to_pcm_frame__binary_search_internal(pFlac, pcmFrameIndex, byteRangeLo, byteRangeHi);
}
#endif  /* !DR_FLAC_NO_CRC */

static drflac_bool32 drflac__seek_to_pcm_frame__seek_table(drflac* pFlac, drflac_uint64 pcmFrameIndex)
{
    drflac_uint32 iClosestSeekpoint = 0;
    drflac_bool32 isMidFrame = DRFLAC_FALSE;
    drflac_uint64 runningPCMFrameCount;
    drflac_uint32 iSeekpoint;


    DRFLAC_ASSERT(pFlac != NULL);

    if (pFlac->pSeekpoints == NULL || pFlac->seekpointCount == 0) {
        return DRFLAC_FALSE;
    }

    /* Do not use the seektable if pcmFramIndex is not coverd by it. */
    if (pFlac->pSeekpoints[0].firstPCMFrame > pcmFrameIndex) {
        return DRFLAC_FALSE;
    }

    for (iSeekpoint = 0; iSeekpoint < pFlac->seekpointCount; ++iSeekpoint) {
        if (pFlac->pSeekpoints[iSeekpoint].firstPCMFrame >= pcmFrameIndex) {
            break;
        }

        iClosestSeekpoint = iSeekpoint;
    }

    /* There's been cases where the seek table contains only zeros. We need to do some basic validation on the closest seekpoint. */
    if (pFlac->pSeekpoints[iClosestSeekpoint].pcmFrameCount == 0 || pFlac->pSeekpoints[iClosestSeekpoint].pcmFrameCount > pFlac->maxBlockSizeInPCMFrames) {
        return DRFLAC_FALSE;
    }
    if (pFlac->pSeekpoints[iClosestSeekpoint].firstPCMFrame > pFlac->totalPCMFrameCount && pFlac->totalPCMFrameCount > 0) {
        return DRFLAC_FALSE;
    }

#if !defined(DR_FLAC_NO_CRC)
    /* At this point we should know the closest seek point. We can use a binary search for this. We need to know the total sample count for this. */
    if (pFlac->totalPCMFrameCount > 0) {
        drflac_uint64 byteRangeLo;
        drflac_uint64 byteRangeHi;

        byteRangeHi = pFlac->firstFLACFramePosInBytes + (drflac_uint64)((drflac_int64)(pFlac->totalPCMFrameCount * pFlac->channels * pFlac->bitsPerSample)/8.0f);
        byteRangeLo = pFlac->firstFLACFramePosInBytes + pFlac->pSeekpoints[iClosestSeekpoint].flacFrameOffset;

        /*
        If our closest seek point is not the last one, we only need to search between it and the next one. The section below calculates an appropriate starting
        value for byteRangeHi which will clamp it appropriately.

        Note that the next seekpoint must have an offset greater than the closest seekpoint because otherwise our binary search algorithm will break down. There
        have been cases where a seektable consists of seek points where every byte offset is set to 0 which causes problems. If this happens we need to abort.
        */
        if (iClosestSeekpoint < pFlac->seekpointCount-1) {
            drflac_uint32 iNextSeekpoint = iClosestSeekpoint + 1;

            /* Basic validation on the seekpoints to ensure they're usable. */
            if (pFlac->pSeekpoints[iClosestSeekpoint].flacFrameOffset >= pFlac->pSeekpoints[iNextSeekpoint].flacFrameOffset || pFlac->pSeekpoints[iNextSeekpoint].pcmFrameCount == 0) {
                return DRFLAC_FALSE;    /* The next seekpoint doesn't look right. The seek table cannot be trusted from here. Abort. */
            }

            if (pFlac->pSeekpoints[iNextSeekpoint].firstPCMFrame != (((drflac_uint64)0xFFFFFFFF << 32) | 0xFFFFFFFF)) { /* Make sure it's not a placeholder seekpoint. */
                byteRangeHi = pFlac->firstFLACFramePosInBytes + pFlac->pSeekpoints[iNextSeekpoint].flacFrameOffset - 1; /* byteRangeHi must be zero based. */
            }
        }

        if (drflac__seek_to_byte(&pFlac->bs, pFlac->firstFLACFramePosInBytes + pFlac->pSeekpoints[iClosestSeekpoint].flacFrameOffset)) {
            if (drflac__read_next_flac_frame_header(&pFlac->bs, pFlac->bitsPerSample, &pFlac->currentFLACFrame.header)) {
                drflac__get_pcm_frame_range_of_current_flac_frame(pFlac, &pFlac->currentPCMFrame, NULL);

                if (drflac__seek_to_pcm_frame__binary_search_internal(pFlac, pcmFrameIndex, byteRangeLo, byteRangeHi)) {
                    return DRFLAC_TRUE;
                }
            }
        }
    }
#endif  /* !DR_FLAC_NO_CRC */

    /* Getting here means we need to use a slower algorithm because the binary search method failed or cannot be used. */

    /*
    If we are seeking forward and the closest seekpoint is _before_ the current sample, we just seek forward from where we are. Otherwise we start seeking
    from the seekpoint's first sample.
    */
    if (pcmFrameIndex >= pFlac->currentPCMFrame && pFlac->pSeekpoints[iClosestSeekpoint].firstPCMFrame <= pFlac->currentPCMFrame) {
        /* Optimized case. Just seek forward from where we are. */
        runningPCMFrameCount = pFlac->currentPCMFrame;

        /* The frame header for the first frame may not yet have been read. We need to do that if necessary. */
        if (pFlac->currentPCMFrame == 0 && pFlac->currentFLACFrame.pcmFramesRemaining == 0) {
            if (!drflac__read_next_flac_frame_header(&pFlac->bs, pFlac->bitsPerSample, &pFlac->currentFLACFrame.header)) {
                return DRFLAC_FALSE;
            }
        } else {
            isMidFrame = DRFLAC_TRUE;
        }
    } else {
        /* Slower case. Seek to the start of the seekpoint and then seek forward from there. */
        runningPCMFrameCount = pFlac->pSeekpoints[iClosestSeekpoint].firstPCMFrame;

        if (!drflac__seek_to_byte(&pFlac->bs, pFlac->firstFLACFramePosInBytes + pFlac->pSeekpoints[iClosestSeekpoint].flacFrameOffset)) {
            return DRFLAC_FALSE;
        }

        /* Grab the frame the seekpoint is sitting on in preparation for the sample-exact seeking below. */
        if (!drflac__read_next_flac_frame_header(&pFlac->bs, pFlac->bitsPerSample, &pFlac->currentFLACFrame.header)) {
            return DRFLAC_FALSE;
        }
    }

    for (;;) {
        drflac_uint64 pcmFrameCountInThisFLACFrame;
        drflac_uint64 firstPCMFrameInFLACFrame = 0;
        drflac_uint64 lastPCMFrameInFLACFrame = 0;

        drflac__get_pcm_frame_range_of_current_flac_frame(pFlac, &firstPCMFrameInFLACFrame, &lastPCMFrameInFLACFrame);

        pcmFrameCountInThisFLACFrame = (lastPCMFrameInFLACFrame - firstPCMFrameInFLACFrame) + 1;
        if (pcmFrameIndex < (runningPCMFrameCount + pcmFrameCountInThisFLACFrame)) {
            /*
            The sample should be in this frame. We need to fully decode it, but if it's an invalid frame (a CRC mismatch) we need to pretend
            it never existed and keep iterating.
            */
            drflac_uint64 pcmFramesToDecode = pcmFrameIndex - runningPCMFrameCount;

            if (!isMidFrame) {
                drflac_result result = drflac__decode_flac_frame(pFlac);
                if (result == DRFLAC_SUCCESS) {
                    /* The frame is valid. We just need to skip over some samples to ensure it's sample-exact. */
                    return drflac__seek_forward_by_pcm_frames(pFlac, pcmFramesToDecode) == pcmFramesToDecode;  /* <-- If this fails, something bad has happened (it should never fail). */
                } else {
                    if (result == DRFLAC_CRC_MISMATCH) {
                        goto next_iteration;   /* CRC mismatch. Pretend this frame never existed. */
                    } else {
                        return DRFLAC_FALSE;
                    }
                }
            } else {
                /* We started seeking mid-frame which means we need to skip the frame decoding part. */
                return drflac__seek_forward_by_pcm_frames(pFlac, pcmFramesToDecode) == pcmFramesToDecode;
            }
        } else {
            /*
            It's not in this frame. We need to seek past the frame, but check if there was a CRC mismatch. If so, we pretend this
            frame never existed and leave the running sample count untouched.
            */
            if (!isMidFrame) {
                drflac_result result = drflac__seek_to_next_flac_frame(pFlac);
                if (result == DRFLAC_SUCCESS) {
                    runningPCMFrameCount += pcmFrameCountInThisFLACFrame;
                } else {
                    if (result == DRFLAC_CRC_MISMATCH) {
                        goto next_iteration;   /* CRC mismatch. Pretend this frame never existed. */
                    } else {
                        return DRFLAC_FALSE;
                    }
                }
            } else {
                /*
                We started seeking mid-frame which means we need to seek by reading to the end of the frame instead of with
                drflac__seek_to_next_flac_frame() which only works if the decoder is sitting on the byte just after the frame header.
                */
                runningPCMFrameCount += pFlac->currentFLACFrame.pcmFramesRemaining;
                pFlac->currentFLACFrame.pcmFramesRemaining = 0;
                isMidFrame = DRFLAC_FALSE;
            }

            /* If we are seeking to the end of the file and we've just hit it, we're done. */
            if (pcmFrameIndex == pFlac->totalPCMFrameCount && runningPCMFrameCount == pFlac->totalPCMFrameCount) {
                return DRFLAC_TRUE;
            }
        }

    next_iteration:
        /* Grab the next frame in preparation for the next iteration. */
        if (!drflac__read_next_flac_frame_header(&pFlac->bs, pFlac->bitsPerSample, &pFlac->currentFLACFrame.header)) {
            return DRFLAC_FALSE;
        }
    }
}


#ifndef DR_FLAC_NO_OGG
typedef struct
{
    drflac_uint8 capturePattern[4];  /* Should be "OggS" */
    drflac_uint8 structureVersion;   /* Always 0. */
    drflac_uint8 headerType;
    drflac_uint64 granulePosition;
    drflac_uint32 serialNumber;
    drflac_uint32 sequenceNumber;
    drflac_uint32 checksum;
    drflac_uint8 segmentCount;
    drflac_uint8 segmentTable[255];
} drflac_ogg_page_header;
#endif

typedef struct
{
    drflac_read_proc onRead;
    drflac_seek_proc onSeek;
    drflac_meta_proc onMeta;
    drflac_container container;
    void* pUserData;
    void* pUserDataMD;
    drflac_uint32 sampleRate;
    drflac_uint8  channels;
    drflac_uint8  bitsPerSample;
    drflac_uint64 totalPCMFrameCount;
    drflac_uint16 maxBlockSizeInPCMFrames;
    drflac_uint64 runningFilePos;
    drflac_bool32 hasStreamInfoBlock;
    drflac_bool32 hasMetadataBlocks;
    drflac_bs bs;                           /* <-- A bit streamer is required for loading data during initialization. */
    drflac_frame_header firstFrameHeader;   /* <-- The header of the first frame that was read during relaxed initalization. Only set if there is no STREAMINFO block. */

#ifndef DR_FLAC_NO_OGG
    drflac_uint32 oggSerial;
    drflac_uint64 oggFirstBytePos;
    drflac_ogg_page_header oggBosHeader;
#endif
} drflac_init_info;

static DRFLAC_INLINE void drflac__decode_block_header(drflac_uint32 blockHeader, drflac_uint8* isLastBlock, drflac_uint8* blockType, drflac_uint32* blockSize)
{
    blockHeader = drflac__be2host_32(blockHeader);
    *isLastBlock = (drflac_uint8)((blockHeader & 0x80000000UL) >> 31);
    *blockType   = (drflac_uint8)((blockHeader & 0x7F000000UL) >> 24);
    *blockSize   =                (blockHeader & 0x00FFFFFFUL);
}

static DRFLAC_INLINE drflac_bool32 drflac__read_and_decode_block_header(drflac_read_proc onRead, void* pUserData, drflac_uint8* isLastBlock, drflac_uint8* blockType, drflac_uint32* blockSize)
{
    drflac_uint32 blockHeader;

    *blockSize = 0;
    if (onRead(pUserData, &blockHeader, 4) != 4) {
        return DRFLAC_FALSE;
    }

    drflac__decode_block_header(blockHeader, isLastBlock, blockType, blockSize);
    return DRFLAC_TRUE;
}

static drflac_bool32 drflac__read_streaminfo(drflac_read_proc onRead, void* pUserData, drflac_streaminfo* pStreamInfo)
{
    drflac_uint32 blockSizes;
    drflac_uint64 frameSizes = 0;
    drflac_uint64 importantProps;
    drflac_uint8 md5[16];

    /* min/max block size. */
    if (onRead(pUserData, &blockSizes, 4) != 4) {
        return DRFLAC_FALSE;
    }

    /* min/max frame size. */
    if (onRead(pUserData, &frameSizes, 6) != 6) {
        return DRFLAC_FALSE;
    }

    /* Sample rate, channels, bits per sample and total sample count. */
    if (onRead(pUserData, &importantProps, 8) != 8) {
        return DRFLAC_FALSE;
    }

    /* MD5 */
    if (onRead(pUserData, md5, sizeof(md5)) != sizeof(md5)) {
        return DRFLAC_FALSE;
    }

    blockSizes     = drflac__be2host_32(blockSizes);
    frameSizes     = drflac__be2host_64(frameSizes);
    importantProps = drflac__be2host_64(importantProps);

    pStreamInfo->minBlockSizeInPCMFrames = (drflac_uint16)((blockSizes & 0xFFFF0000) >> 16);
    pStreamInfo->maxBlockSizeInPCMFrames = (drflac_uint16) (blockSizes & 0x0000FFFF);
    pStreamInfo->minFrameSizeInPCMFrames = (drflac_uint32)((frameSizes     &  (((drflac_uint64)0x00FFFFFF << 16) << 24)) >> 40);
    pStreamInfo->maxFrameSizeInPCMFrames = (drflac_uint32)((frameSizes     &  (((drflac_uint64)0x00FFFFFF << 16) <<  0)) >> 16);
    pStreamInfo->sampleRate              = (drflac_uint32)((importantProps &  (((drflac_uint64)0x000FFFFF << 16) << 28)) >> 44);
    pStreamInfo->channels                = (drflac_uint8 )((importantProps &  (((drflac_uint64)0x0000000E << 16) << 24)) >> 41) + 1;
    pStreamInfo->bitsPerSample           = (drflac_uint8 )((importantProps &  (((drflac_uint64)0x0000001F << 16) << 20)) >> 36) + 1;
    pStreamInfo->totalPCMFrameCount      =                ((importantProps & ((((drflac_uint64)0x0000000F << 16) << 16) | 0xFFFFFFFF)));
    DRFLAC_COPY_MEMORY(pStreamInfo->md5, md5, sizeof(md5));

    return DRFLAC_TRUE;
}


static void* drflac__malloc_default(size_t sz, void* pUserData)
{
    (void)pUserData;
    return DRFLAC_MALLOC(sz);
}

static void* drflac__realloc_default(void* p, size_t sz, void* pUserData)
{
    (void)pUserData;
    return DRFLAC_REALLOC(p, sz);
}

static void drflac__free_default(void* p, void* pUserData)
{
    (void)pUserData;
    DRFLAC_FREE(p);
}


static void* drflac__malloc_from_callbacks(size_t sz, const drflac_allocation_callbacks* pAllocationCallbacks)
{
    if (pAllocationCallbacks == NULL) {
        return NULL;
    }

    if (pAllocationCallbacks->onMalloc != NULL) {
        return pAllocationCallbacks->onMalloc(sz, pAllocationCallbacks->pUserData);
    }

    /* Try using realloc(). */
    if (pAllocationCallbacks->onRealloc != NULL) {
        return pAllocationCallbacks->onRealloc(NULL, sz, pAllocationCallbacks->pUserData);
    }

    return NULL;
}

static void* drflac__realloc_from_callbacks(void* p, size_t szNew, size_t szOld, const drflac_allocation_callbacks* pAllocationCallbacks)
{
    if (pAllocationCallbacks == NULL) {
        return NULL;
    }

    if (pAllocationCallbacks->onRealloc != NULL) {
        return pAllocationCallbacks->onRealloc(p, szNew, pAllocationCallbacks->pUserData);
    }

    /* Try emulating realloc() in terms of malloc()/free(). */
    if (pAllocationCallbacks->onMalloc != NULL && pAllocationCallbacks->onFree != NULL) {
        void* p2;

        p2 = pAllocationCallbacks->onMalloc(szNew, pAllocationCallbacks->pUserData);
        if (p2 == NULL) {
            return NULL;
        }

        if (p != NULL) {
            DRFLAC_COPY_MEMORY(p2, p, szOld);
            pAllocationCallbacks->onFree(p, pAllocationCallbacks->pUserData);
        }

        return p2;
    }

    return NULL;
}

static void drflac__free_from_callbacks(void* p, const drflac_allocation_callbacks* pAllocationCallbacks)
{
    if (p == NULL || pAllocationCallbacks == NULL) {
        return;
    }

    if (pAllocationCallbacks->onFree != NULL) {
        pAllocationCallbacks->onFree(p, pAllocationCallbacks->pUserData);
    }
}


static drflac_bool32 drflac__read_and_decode_metadata(drflac_read_proc onRead, drflac_seek_proc onSeek, drflac_meta_proc onMeta, void* pUserData, void* pUserDataMD, drflac_uint64* pFirstFramePos, drflac_uint64* pSeektablePos, drflac_uint32* pSeekpointCount, drflac_allocation_callbacks* pAllocationCallbacks)
{
    /*
    We want to keep track of the byte position in the stream of the seektable. At the time of calling this function we know that
    we'll be sitting on byte 42.
    */
    drflac_uint64 runningFilePos = 42;
    drflac_uint64 seektablePos   = 0;
    drflac_uint32 seektableSize  = 0;

    for (;;) {
        drflac_metadata metadata;
        drflac_uint8 isLastBlock = 0;
        drflac_uint8 blockType;
        drflac_uint32 blockSize;
        if (drflac__read_and_decode_block_header(onRead, pUserData, &isLastBlock, &blockType, &blockSize) == DRFLAC_FALSE) {
            return DRFLAC_FALSE;
        }
        runningFilePos += 4;

        metadata.type = blockType;
        metadata.pRawData = NULL;
        metadata.rawDataSize = 0;

        switch (blockType)
        {
            case DRFLAC_METADATA_BLOCK_TYPE_APPLICATION:
            {
                if (blockSize < 4) {
                    return DRFLAC_FALSE;
                }

                if (onMeta) {
                    void* pRawData = drflac__malloc_from_callbacks(blockSize, pAllocationCallbacks);
                    if (pRawData == NULL) {
                        return DRFLAC_FALSE;
                    }

                    if (onRead(pUserData, pRawData, blockSize) != blockSize) {
                        drflac__free_from_callbacks(pRawData, pAllocationCallbacks);
                        return DRFLAC_FALSE;
                    }

                    metadata.pRawData = pRawData;
                    metadata.rawDataSize = blockSize;
                    metadata.data.application.id       = drflac__be2host_32(*(drflac_uint32*)pRawData);
                    metadata.data.application.pData    = (const void*)((drflac_uint8*)pRawData + sizeof(drflac_uint32));
                    metadata.data.application.dataSize = blockSize - sizeof(drflac_uint32);
                    onMeta(pUserDataMD, &metadata);

                    drflac__free_from_callbacks(pRawData, pAllocationCallbacks);
                }
            } break;

            case DRFLAC_METADATA_BLOCK_TYPE_SEEKTABLE:
            {
                seektablePos  = runningFilePos;
                seektableSize = blockSize;

                if (onMeta) {
                    drflac_uint32 seekpointCount;
                    drflac_uint32 iSeekpoint;
                    void* pRawData;

                    seekpointCount = blockSize/DRFLAC_SEEKPOINT_SIZE_IN_BYTES;

                    pRawData = drflac__malloc_from_callbacks(seekpointCount * sizeof(drflac_seekpoint), pAllocationCallbacks);
                    if (pRawData == NULL) {
                        return DRFLAC_FALSE;
                    }

                    /* We need to read seekpoint by seekpoint and do some processing. */
                    for (iSeekpoint = 0; iSeekpoint < seekpointCount; ++iSeekpoint) {
                        drflac_seekpoint* pSeekpoint = (drflac_seekpoint*)pRawData + iSeekpoint;

                        if (onRead(pUserData, pSeekpoint, DRFLAC_SEEKPOINT_SIZE_IN_BYTES) != DRFLAC_SEEKPOINT_SIZE_IN_BYTES) {
                            drflac__free_from_callbacks(pRawData, pAllocationCallbacks);
                            return DRFLAC_FALSE;
                        }

                        /* Endian swap. */
                        pSeekpoint->firstPCMFrame   = drflac__be2host_64(pSeekpoint->firstPCMFrame);
                        pSeekpoint->flacFrameOffset = drflac__be2host_64(pSeekpoint->flacFrameOffset);
                        pSeekpoint->pcmFrameCount   = drflac__be2host_16(pSeekpoint->pcmFrameCount);
                    }

                    metadata.pRawData = pRawData;
                    metadata.rawDataSize = blockSize;
                    metadata.data.seektable.seekpointCount = seekpointCount;
                    metadata.data.seektable.pSeekpoints = (const drflac_seekpoint*)pRawData;

                    onMeta(pUserDataMD, &metadata);

                    drflac__free_from_callbacks(pRawData, pAllocationCallbacks);
                }
            } break;

            case DRFLAC_METADATA_BLOCK_TYPE_VORBIS_COMMENT:
            {
                if (blockSize < 8) {
                    return DRFLAC_FALSE;
                }

                if (onMeta) {
                    void* pRawData;
                    const char* pRunningData;
                    const char* pRunningDataEnd;
                    drflac_uint32 i;

                    pRawData = drflac__malloc_from_callbacks(blockSize, pAllocationCallbacks);
                    if (pRawData == NULL) {
                        return DRFLAC_FALSE;
                    }

                    if (onRead(pUserData, pRawData, blockSize) != blockSize) {
                        drflac__free_from_callbacks(pRawData, pAllocationCallbacks);
                        return DRFLAC_FALSE;
                    }

                    metadata.pRawData = pRawData;
                    metadata.rawDataSize = blockSize;

                    pRunningData    = (const char*)pRawData;
                    pRunningDataEnd = (const char*)pRawData + blockSize;

                    metadata.data.vorbis_comment.vendorLength = drflac__le2host_32_ptr_unaligned(pRunningData); pRunningData += 4;

                    /* Need space for the rest of the block */
                    if ((pRunningDataEnd - pRunningData) - 4 < (drflac_int64)metadata.data.vorbis_comment.vendorLength) { /* <-- Note the order of operations to avoid overflow to a valid value */
                        drflac__free_from_callbacks(pRawData, pAllocationCallbacks);
                        return DRFLAC_FALSE;
                    }
                    metadata.data.vorbis_comment.vendor       = pRunningData;                                            pRunningData += metadata.data.vorbis_comment.vendorLength;
                    metadata.data.vorbis_comment.commentCount = drflac__le2host_32_ptr_unaligned(pRunningData); pRunningData += 4;

                    /* Need space for 'commentCount' comments after the block, which at minimum is a drflac_uint32 per comment */
                    if ((pRunningDataEnd - pRunningData) / sizeof(drflac_uint32) < metadata.data.vorbis_comment.commentCount) { /* <-- Note the order of operations to avoid overflow to a valid value */
                        drflac__free_from_callbacks(pRawData, pAllocationCallbacks);
                        return DRFLAC_FALSE;
                    }
                    metadata.data.vorbis_comment.pComments    = pRunningData;

                    /* Check that the comments section is valid before passing it to the callback */
                    for (i = 0; i < metadata.data.vorbis_comment.commentCount; ++i) {
                        drflac_uint32 commentLength;

                        if (pRunningDataEnd - pRunningData < 4) {
                            drflac__free_from_callbacks(pRawData, pAllocationCallbacks);
                            return DRFLAC_FALSE;
                        }

                        commentLength = drflac__le2host_32_ptr_unaligned(pRunningData); pRunningData += 4;
                        if (pRunningDataEnd - pRunningData < (drflac_int64)commentLength) { /* <-- Note the order of operations to avoid overflow to a valid value */
                            drflac__free_from_callbacks(pRawData, pAllocationCallbacks);
                            return DRFLAC_FALSE;
                        }
                        pRunningData += commentLength;
                    }

                    onMeta(pUserDataMD, &metadata);

                    drflac__free_from_callbacks(pRawData, pAllocationCallbacks);
                }
            } break;

            case DRFLAC_METADATA_BLOCK_TYPE_CUESHEET:
            {
                if (blockSize < 396) {
                    return DRFLAC_FALSE;
                }

                if (onMeta) {
                    void* pRawData;
                    const char* pRunningData;
                    const char* pRunningDataEnd;
                    size_t bufferSize;
                    drflac_uint8 iTrack;
                    drflac_uint8 iIndex;
                    void* pTrackData;

                    /*
                    This needs to be loaded in two passes. The first pass is used to calculate the size of the memory allocation
                    we need for storing the necessary data. The second pass will fill that buffer with usable data.
                    */
                    pRawData = drflac__malloc_from_callbacks(blockSize, pAllocationCallbacks);
                    if (pRawData == NULL) {
                        return DRFLAC_FALSE;
                    }

                    if (onRead(pUserData, pRawData, blockSize) != blockSize) {
                        drflac__free_from_callbacks(pRawData, pAllocationCallbacks);
                        return DRFLAC_FALSE;
                    }

                    metadata.pRawData = pRawData;
                    metadata.rawDataSize = blockSize;

                    pRunningData    = (const char*)pRawData;
                    pRunningDataEnd = (const char*)pRawData + blockSize;

                    DRFLAC_COPY_MEMORY(metadata.data.cuesheet.catalog, pRunningData, 128);                              pRunningData += 128;
                    metadata.data.cuesheet.leadInSampleCount = drflac__be2host_64(*(const drflac_uint64*)pRunningData); pRunningData += 8;
                    metadata.data.cuesheet.isCD              = (pRunningData[0] & 0x80) != 0;                           pRunningData += 259;
                    metadata.data.cuesheet.trackCount        = pRunningData[0];                                         pRunningData += 1;
                    metadata.data.cuesheet.pTrackData        = NULL;    /* Will be filled later. */

                    /* Pass 1: Calculate the size of the buffer for the track data. */
                    {
                        const char* pRunningDataSaved = pRunningData;   /* Will be restored at the end in preparation for the second pass. */

                        bufferSize = metadata.data.cuesheet.trackCount * DRFLAC_CUESHEET_TRACK_SIZE_IN_BYTES;

                        for (iTrack = 0; iTrack < metadata.data.cuesheet.trackCount; ++iTrack) {
                            drflac_uint8 indexCount;
                            drflac_uint32 indexPointSize;

                            if (pRunningDataEnd - pRunningData < DRFLAC_CUESHEET_TRACK_SIZE_IN_BYTES) {
                                drflac__free_from_callbacks(pRawData, pAllocationCallbacks);
                                return DRFLAC_FALSE;
                            }

                            /* Skip to the index point count */
                            pRunningData += 35;
                            
                            indexCount = pRunningData[0];
                            pRunningData += 1;
                            
                            bufferSize += indexCount * sizeof(drflac_cuesheet_track_index);

                            /* Quick validation check. */
                            indexPointSize = indexCount * DRFLAC_CUESHEET_TRACK_INDEX_SIZE_IN_BYTES;
                            if (pRunningDataEnd - pRunningData < (drflac_int64)indexPointSize) {
                                drflac__free_from_callbacks(pRawData, pAllocationCallbacks);
                                return DRFLAC_FALSE;
                            }

                            pRunningData += indexPointSize;
                        }

                        pRunningData = pRunningDataSaved;
                    }

                    /* Pass 2: Allocate a buffer and fill the data. Validation was done in the step above so can be skipped. */
                    {
                        char* pRunningTrackData;

                        pTrackData = drflac__malloc_from_callbacks(bufferSize, pAllocationCallbacks);
                        if (pTrackData == NULL) {
                            drflac__free_from_callbacks(pRawData, pAllocationCallbacks);
                            return DRFLAC_FALSE;
                        }

                        pRunningTrackData = (char*)pTrackData;

                        for (iTrack = 0; iTrack < metadata.data.cuesheet.trackCount; ++iTrack) {
                            drflac_uint8 indexCount;

                            DRFLAC_COPY_MEMORY(pRunningTrackData, pRunningData, DRFLAC_CUESHEET_TRACK_SIZE_IN_BYTES);
                            pRunningData      += DRFLAC_CUESHEET_TRACK_SIZE_IN_BYTES-1; /* Skip forward, but not beyond the last byte in the CUESHEET_TRACK block which is the index count. */
                            pRunningTrackData += DRFLAC_CUESHEET_TRACK_SIZE_IN_BYTES-1;

                            /* Grab the index count for the next part. */
                            indexCount = pRunningData[0];
                            pRunningData      += 1;
                            pRunningTrackData += 1;

                            /* Extract each track index. */
                            for (iIndex = 0; iIndex < indexCount; ++iIndex) {
                                drflac_cuesheet_track_index* pTrackIndex = (drflac_cuesheet_track_index*)pRunningTrackData;

                                DRFLAC_COPY_MEMORY(pRunningTrackData, pRunningData, DRFLAC_CUESHEET_TRACK_INDEX_SIZE_IN_BYTES);
                                pRunningData      += DRFLAC_CUESHEET_TRACK_INDEX_SIZE_IN_BYTES;
                                pRunningTrackData += sizeof(drflac_cuesheet_track_index);

                                pTrackIndex->offset = drflac__be2host_64(pTrackIndex->offset);
                            }
                        }

                        metadata.data.cuesheet.pTrackData = pTrackData;
                    }

                    /* The original data is no longer needed. */
                    drflac__free_from_callbacks(pRawData, pAllocationCallbacks);
                    pRawData = NULL;

                    onMeta(pUserDataMD, &metadata);

                    drflac__free_from_callbacks(pTrackData, pAllocationCallbacks);
                    pTrackData = NULL;
                }
            } break;

            case DRFLAC_METADATA_BLOCK_TYPE_PICTURE:
            {
                if (blockSize < 32) {
                    return DRFLAC_FALSE;
                }

                if (onMeta) {
                    void* pRawData;
                    const char* pRunningData;
                    const char* pRunningDataEnd;

                    pRawData = drflac__malloc_from_callbacks(blockSize, pAllocationCallbacks);
                    if (pRawData == NULL) {
                        return DRFLAC_FALSE;
                    }

                    if (onRead(pUserData, pRawData, blockSize) != blockSize) {
                        drflac__free_from_callbacks(pRawData, pAllocationCallbacks);
                        return DRFLAC_FALSE;
                    }

                    metadata.pRawData = pRawData;
                    metadata.rawDataSize = blockSize;

                    pRunningData    = (const char*)pRawData;
                    pRunningDataEnd = (const char*)pRawData + blockSize;

                    metadata.data.picture.type       = drflac__be2host_32_ptr_unaligned(pRunningData); pRunningData += 4;
                    metadata.data.picture.mimeLength = drflac__be2host_32_ptr_unaligned(pRunningData); pRunningData += 4;

                    /* Need space for the rest of the block */
                    if ((pRunningDataEnd - pRunningData) - 24 < (drflac_int64)metadata.data.picture.mimeLength) { /* <-- Note the order of operations to avoid overflow to a valid value */
                        drflac__free_from_callbacks(pRawData, pAllocationCallbacks);
                        return DRFLAC_FALSE;
                    }
                    metadata.data.picture.mime              = pRunningData;                                   pRunningData += metadata.data.picture.mimeLength;
                    metadata.data.picture.descriptionLength = drflac__be2host_32_ptr_unaligned(pRunningData); pRunningData += 4;

                    /* Need space for the rest of the block */
                    if ((pRunningDataEnd - pRunningData) - 20 < (drflac_int64)metadata.data.picture.descriptionLength) { /* <-- Note the order of operations to avoid overflow to a valid value */
                        drflac__free_from_callbacks(pRawData, pAllocationCallbacks);
                        return DRFLAC_FALSE;
                    }
                    metadata.data.picture.description     = pRunningData;                                   pRunningData += metadata.data.picture.descriptionLength;
                    metadata.data.picture.width           = drflac__be2host_32_ptr_unaligned(pRunningData); pRunningData += 4;
                    metadata.data.picture.height          = drflac__be2host_32_ptr_unaligned(pRunningData); pRunningData += 4;
                    metadata.data.picture.colorDepth      = drflac__be2host_32_ptr_unaligned(pRunningData); pRunningData += 4;
                    metadata.data.picture.indexColorCount = drflac__be2host_32_ptr_unaligned(pRunningData); pRunningData += 4;
                    metadata.data.picture.pictureDataSize = drflac__be2host_32_ptr_unaligned(pRunningData); pRunningData += 4;
                    metadata.data.picture.pPictureData    = (const drflac_uint8*)pRunningData;

                    /* Need space for the picture after the block */
                    if (pRunningDataEnd - pRunningData < (drflac_int64)metadata.data.picture.pictureDataSize) { /* <-- Note the order of operations to avoid overflow to a valid value */
                        drflac__free_from_callbacks(pRawData, pAllocationCallbacks);
                        return DRFLAC_FALSE;
                    }

                    onMeta(pUserDataMD, &metadata);

                    drflac__free_from_callbacks(pRawData, pAllocationCallbacks);
                }
            } break;

            case DRFLAC_METADATA_BLOCK_TYPE_PADDING:
            {
                if (onMeta) {
                    metadata.data.padding.unused = 0;

                    /* Padding doesn't have anything meaningful in it, so just skip over it, but make sure the caller is aware of it by firing the callback. */
                    if (!onSeek(pUserData, blockSize, drflac_seek_origin_current)) {
                        isLastBlock = DRFLAC_TRUE;  /* An error occurred while seeking. Attempt to recover by treating this as the last block which will in turn terminate the loop. */
                    } else {
                        onMeta(pUserDataMD, &metadata);
                    }
                }
            } break;

            case DRFLAC_METADATA_BLOCK_TYPE_INVALID:
            {
                /* Invalid chunk. Just skip over this one. */
                if (onMeta) {
                    if (!onSeek(pUserData, blockSize, drflac_seek_origin_current)) {
                        isLastBlock = DRFLAC_TRUE;  /* An error occurred while seeking. Attempt to recover by treating this as the last block which will in turn terminate the loop. */
                    }
                }
            } break;

            default:
            {
                /*
                It's an unknown chunk, but not necessarily invalid. There's a chance more metadata blocks might be defined later on, so we
                can at the very least report the chunk to the application and let it look at the raw data.
                */
                if (onMeta) {
                    void* pRawData = drflac__malloc_from_callbacks(blockSize, pAllocationCallbacks);
                    if (pRawData == NULL) {
                        return DRFLAC_FALSE;
                    }

                    if (onRead(pUserData, pRawData, blockSize) != blockSize) {
                        drflac__free_from_callbacks(pRawData, pAllocationCallbacks);
                        return DRFLAC_FALSE;
                    }

                    metadata.pRawData = pRawData;
                    metadata.rawDataSize = blockSize;
                    onMeta(pUserDataMD, &metadata);

                    drflac__free_from_callbacks(pRawData, pAllocationCallbacks);
                }
            } break;
        }

        /* If we're not handling metadata, just skip over the block. If we are, it will have been handled earlier in the switch statement above. */
        if (onMeta == NULL && blockSize > 0) {
            if (!onSeek(pUserData, blockSize, drflac_seek_origin_current)) {
                isLastBlock = DRFLAC_TRUE;
            }
        }

        runningFilePos += blockSize;
        if (isLastBlock) {
            break;
        }
    }

    *pSeektablePos   = seektablePos;
    *pSeekpointCount = seektableSize / DRFLAC_SEEKPOINT_SIZE_IN_BYTES;
    *pFirstFramePos  = runningFilePos;

    return DRFLAC_TRUE;
}

static drflac_bool32 drflac__init_private__native(drflac_init_info* pInit, drflac_read_proc onRead, drflac_seek_proc onSeek, drflac_meta_proc onMeta, void* pUserData, void* pUserDataMD, drflac_bool32 relaxed)
{
    /* Pre Condition: The bit stream should be sitting just past the 4-byte id header. */

    drflac_uint8 isLastBlock;
    drflac_uint8 blockType;
    drflac_uint32 blockSize;

    (void)onSeek;

    pInit->container = drflac_container_native;

    /* The first metadata block should be the STREAMINFO block. */
    if (!drflac__read_and_decode_block_header(onRead, pUserData, &isLastBlock, &blockType, &blockSize)) {
        return DRFLAC_FALSE;
    }

    if (blockType != DRFLAC_METADATA_BLOCK_TYPE_STREAMINFO || blockSize != 34) {
        if (!relaxed) {
            /* We're opening in strict mode and the first block is not the STREAMINFO block. Error. */
            return DRFLAC_FALSE;
        } else {
            /*
            Relaxed mode. To open from here we need to just find the first frame and set the sample rate, etc. to whatever is defined
            for that frame.
            */
            pInit->hasStreamInfoBlock = DRFLAC_FALSE;
            pInit->hasMetadataBlocks  = DRFLAC_FALSE;

            if (!drflac__read_next_flac_frame_header(&pInit->bs, 0, &pInit->firstFrameHeader)) {
                return DRFLAC_FALSE;    /* Couldn't find a frame. */
            }

            if (pInit->firstFrameHeader.bitsPerSample == 0) {
                return DRFLAC_FALSE;    /* Failed to initialize because the first frame depends on the STREAMINFO block, which does not exist. */
            }

            pInit->sampleRate              = pInit->firstFrameHeader.sampleRate;
            pInit->channels                = drflac__get_channel_count_from_channel_assignment(pInit->firstFrameHeader.channelAssignment);
            pInit->bitsPerSample           = pInit->firstFrameHeader.bitsPerSample;
            pInit->maxBlockSizeInPCMFrames = 65535;   /* <-- See notes here: https://xiph.org/flac/format.html#metadata_block_streaminfo */
            return DRFLAC_TRUE;
        }
    } else {
        drflac_streaminfo streaminfo;
        if (!drflac__read_streaminfo(onRead, pUserData, &streaminfo)) {
            return DRFLAC_FALSE;
        }

        pInit->hasStreamInfoBlock      = DRFLAC_TRUE;
        pInit->sampleRate              = streaminfo.sampleRate;
        pInit->channels                = streaminfo.channels;
        pInit->bitsPerSample           = streaminfo.bitsPerSample;
        pInit->totalPCMFrameCount      = streaminfo.totalPCMFrameCount;
        pInit->maxBlockSizeInPCMFrames = streaminfo.maxBlockSizeInPCMFrames;    /* Don't care about the min block size - only the max (used for determining the size of the memory allocation). */
        pInit->hasMetadataBlocks       = !isLastBlock;

        if (onMeta) {
            drflac_metadata metadata;
            metadata.type = DRFLAC_METADATA_BLOCK_TYPE_STREAMINFO;
            metadata.pRawData = NULL;
            metadata.rawDataSize = 0;
            metadata.data.streaminfo = streaminfo;
            onMeta(pUserDataMD, &metadata);
        }

        return DRFLAC_TRUE;
    }
}

#ifndef DR_FLAC_NO_OGG
#define DRFLAC_OGG_MAX_PAGE_SIZE            65307
#define DRFLAC_OGG_CAPTURE_PATTERN_CRC32    1605413199  /* CRC-32 of "OggS". */

typedef enum
{
    drflac_ogg_recover_on_crc_mismatch,
    drflac_ogg_fail_on_crc_mismatch
} drflac_ogg_crc_mismatch_recovery;

#ifndef DR_FLAC_NO_CRC
static drflac_uint32 drflac__crc32_table[] = {
    0x00000000L, 0x04C11DB7L, 0x09823B6EL, 0x0D4326D9L,
    0x130476DCL, 0x17C56B6BL, 0x1A864DB2L, 0x1E475005L,
    0x2608EDB8L, 0x22C9F00FL, 0x2F8AD6D6L, 0x2B4BCB61L,
    0x350C9B64L, 0x31CD86D3L, 0x3C8EA00AL, 0x384FBDBDL,
    0x4C11DB70L, 0x48D0C6C7L, 0x4593E01EL, 0x4152FDA9L,
    0x5F15ADACL, 0x5BD4B01BL, 0x569796C2L, 0x52568B75L,
    0x6A1936C8L, 0x6ED82B7FL, 0x639B0DA6L, 0x675A1011L,
    0x791D4014L, 0x7DDC5DA3L, 0x709F7B7AL, 0x745E66CDL,
    0x9823B6E0L, 0x9CE2AB57L, 0x91A18D8EL, 0x95609039L,
    0x8B27C03CL, 0x8FE6DD8BL, 0x82A5FB52L, 0x8664E6E5L,
    0xBE2B5B58L, 0xBAEA46EFL, 0xB7A96036L, 0xB3687D81L,
    0xAD2F2D84L, 0xA9EE3033L, 0xA4AD16EAL, 0xA06C0B5DL,
    0xD4326D90L, 0xD0F37027L, 0xDDB056FEL, 0xD9714B49L,
    0xC7361B4CL, 0xC3F706FBL, 0xCEB42022L, 0xCA753D95L,
    0xF23A8028L, 0xF6FB9D9FL, 0xFBB8BB46L, 0xFF79A6F1L,
    0xE13EF6F4L, 0xE5FFEB43L, 0xE8BCCD9AL, 0xEC7DD02DL,
    0x34867077L, 0x30476DC0L, 0x3D044B19L, 0x39C556AEL,
    0x278206ABL, 0x23431B1CL, 0x2E003DC5L, 0x2AC12072L,
    0x128E9DCFL, 0x164F8078L, 0x1B0CA6A1L, 0x1FCDBB16L,
    0x018AEB13L, 0x054BF6A4L, 0x0808D07DL, 0x0CC9CDCAL,
    0x7897AB07L, 0x7C56B6B0L, 0x71159069L, 0x75D48DDEL,
    0x6B93DDDBL, 0x6F52C06CL, 0x6211E6B5L, 0x66D0FB02L,
    0x5E9F46BFL, 0x5A5E5B08L, 0x571D7DD1L, 0x53DC6066L,
    0x4D9B3063L, 0x495A2DD4L, 0x44190B0DL, 0x40D816BAL,
    0xACA5C697L, 0xA864DB20L, 0xA527FDF9L, 0xA1E6E04EL,
    0xBFA1B04BL, 0xBB60ADFCL, 0xB6238B25L, 0xB2E29692L,
    0x8AAD2B2FL, 0x8E6C3698L, 0x832F1041L, 0x87EE0DF6L,
    0x99A95DF3L, 0x9D684044L, 0x902B669DL, 0x94EA7B2AL,
    0xE0B41DE7L, 0xE4750050L, 0xE9362689L, 0xEDF73B3EL,
    0xF3B06B3BL, 0xF771768CL, 0xFA325055L, 0xFEF34DE2L,
    0xC6BCF05FL, 0xC27DEDE8L, 0xCF3ECB31L, 0xCBFFD686L,
    0xD5B88683L, 0xD1799B34L, 0xDC3ABDEDL, 0xD8FBA05AL,
    0x690CE0EEL, 0x6DCDFD59L, 0x608EDB80L, 0x644FC637L,
    0x7A089632L, 0x7EC98B85L, 0x738AAD5CL, 0x774BB0EBL,
    0x4F040D56L, 0x4BC510E1L, 0x46863638L, 0x42472B8FL,
    0x5C007B8AL, 0x58C1663DL, 0x558240E4L, 0x51435D53L,
    0x251D3B9EL, 0x21DC2629L, 0x2C9F00F0L, 0x285E1D47L,
    0x36194D42L, 0x32D850F5L, 0x3F9B762CL, 0x3B5A6B9BL,
    0x0315D626L, 0x07D4CB91L, 0x0A97ED48L, 0x0E56F0FFL,
    0x1011A0FAL, 0x14D0BD4DL, 0x19939B94L, 0x1D528623L,
    0xF12F560EL, 0xF5EE4BB9L, 0xF8AD6D60L, 0xFC6C70D7L,
    0xE22B20D2L, 0xE6EA3D65L, 0xEBA91BBCL, 0xEF68060BL,
    0xD727BBB6L, 0xD3E6A601L, 0xDEA580D8L, 0xDA649D6FL,
    0xC423CD6AL, 0xC0E2D0DDL, 0xCDA1F604L, 0xC960EBB3L,
    0xBD3E8D7EL, 0xB9FF90C9L, 0xB4BCB610L, 0xB07DABA7L,
    0xAE3AFBA2L, 0xAAFBE615L, 0xA7B8C0CCL, 0xA379DD7BL,
    0x9B3660C6L, 0x9FF77D71L, 0x92B45BA8L, 0x9675461FL,
    0x8832161AL, 0x8CF30BADL, 0x81B02D74L, 0x857130C3L,
    0x5D8A9099L, 0x594B8D2EL, 0x5408ABF7L, 0x50C9B640L,
    0x4E8EE645L, 0x4A4FFBF2L, 0x470CDD2BL, 0x43CDC09CL,
    0x7B827D21L, 0x7F436096L, 0x7200464FL, 0x76C15BF8L,
    0x68860BFDL, 0x6C47164AL, 0x61043093L, 0x65C52D24L,
    0x119B4BE9L, 0x155A565EL, 0x18197087L, 0x1CD86D30L,
    0x029F3D35L, 0x065E2082L, 0x0B1D065BL, 0x0FDC1BECL,
    0x3793A651L, 0x3352BBE6L, 0x3E119D3FL, 0x3AD08088L,
    0x2497D08DL, 0x2056CD3AL, 0x2D15EBE3L, 0x29D4F654L,
    0xC5A92679L, 0xC1683BCEL, 0xCC2B1D17L, 0xC8EA00A0L,
    0xD6AD50A5L, 0xD26C4D12L, 0xDF2F6BCBL, 0xDBEE767CL,
    0xE3A1CBC1L, 0xE760D676L, 0xEA23F0AFL, 0xEEE2ED18L,
    0xF0A5BD1DL, 0xF464A0AAL, 0xF9278673L, 0xFDE69BC4L,
    0x89B8FD09L, 0x8D79E0BEL, 0x803AC667L, 0x84FBDBD0L,
    0x9ABC8BD5L, 0x9E7D9662L, 0x933EB0BBL, 0x97FFAD0CL,
    0xAFB010B1L, 0xAB710D06L, 0xA6322BDFL, 0xA2F33668L,
    0xBCB4666DL, 0xB8757BDAL, 0xB5365D03L, 0xB1F740B4L
};
#endif

static DRFLAC_INLINE drflac_uint32 drflac_crc32_byte(drflac_uint32 crc32, drflac_uint8 data)
{
#ifndef DR_FLAC_NO_CRC
    return (crc32 << 8) ^ drflac__crc32_table[(drflac_uint8)((crc32 >> 24) & 0xFF) ^ data];
#else
    (void)data;
    return crc32;
#endif
}

#if 0
static DRFLAC_INLINE drflac_uint32 drflac_crc32_uint32(drflac_uint32 crc32, drflac_uint32 data)
{
    crc32 = drflac_crc32_byte(crc32, (drflac_uint8)((data >> 24) & 0xFF));
    crc32 = drflac_crc32_byte(crc32, (drflac_uint8)((data >> 16) & 0xFF));
    crc32 = drflac_crc32_byte(crc32, (drflac_uint8)((data >>  8) & 0xFF));
    crc32 = drflac_crc32_byte(crc32, (drflac_uint8)((data >>  0) & 0xFF));
    return crc32;
}

static DRFLAC_INLINE drflac_uint32 drflac_crc32_uint64(drflac_uint32 crc32, drflac_uint64 data)
{
    crc32 = drflac_crc32_uint32(crc32, (drflac_uint32)((data >> 32) & 0xFFFFFFFF));
    crc32 = drflac_crc32_uint32(crc32, (drflac_uint32)((data >>  0) & 0xFFFFFFFF));
    return crc32;
}
#endif

static DRFLAC_INLINE drflac_uint32 drflac_crc32_buffer(drflac_uint32 crc32, drflac_uint8* pData, drflac_uint32 dataSize)
{
    /* This can be optimized. */
    drflac_uint32 i;
    for (i = 0; i < dataSize; ++i) {
        crc32 = drflac_crc32_byte(crc32, pData[i]);
    }
    return crc32;
}


static DRFLAC_INLINE drflac_bool32 drflac_ogg__is_capture_pattern(drflac_uint8 pattern[4])
{
    return pattern[0] == 'O' && pattern[1] == 'g' && pattern[2] == 'g' && pattern[3] == 'S';
}

static DRFLAC_INLINE drflac_uint32 drflac_ogg__get_page_header_size(drflac_ogg_page_header* pHeader)
{
    return 27 + pHeader->segmentCount;
}

static DRFLAC_INLINE drflac_uint32 drflac_ogg__get_page_body_size(drflac_ogg_page_header* pHeader)
{
    drflac_uint32 pageBodySize = 0;
    int i;

    for (i = 0; i < pHeader->segmentCount; ++i) {
        pageBodySize += pHeader->segmentTable[i];
    }

    return pageBodySize;
}

static drflac_result drflac_ogg__read_page_header_after_capture_pattern(drflac_read_proc onRead, void* pUserData, drflac_ogg_page_header* pHeader, drflac_uint32* pBytesRead, drflac_uint32* pCRC32)
{
    drflac_uint8 data[23];
    drflac_uint32 i;

    DRFLAC_ASSERT(*pCRC32 == DRFLAC_OGG_CAPTURE_PATTERN_CRC32);

    if (onRead(pUserData, data, 23) != 23) {
        return DRFLAC_AT_END;
    }
    *pBytesRead += 23;

    /*
    It's not actually used, but set the capture pattern to 'OggS' for completeness. Not doing this will cause static analysers to complain about
    us trying to access uninitialized data. We could alternatively just comment out this member of the drflac_ogg_page_header structure, but I
    like to have it map to the structure of the underlying data.
    */
    pHeader->capturePattern[0] = 'O';
    pHeader->capturePattern[1] = 'g';
    pHeader->capturePattern[2] = 'g';
    pHeader->capturePattern[3] = 'S';

    pHeader->structureVersion = data[0];
    pHeader->headerType       = data[1];
    DRFLAC_COPY_MEMORY(&pHeader->granulePosition, &data[ 2], 8);
    DRFLAC_COPY_MEMORY(&pHeader->serialNumber,    &data[10], 4);
    DRFLAC_COPY_MEMORY(&pHeader->sequenceNumber,  &data[14], 4);
    DRFLAC_COPY_MEMORY(&pHeader->checksum,        &data[18], 4);
    pHeader->segmentCount     = data[22];

    /* Calculate the CRC. Note that for the calculation the checksum part of the page needs to be set to 0. */
    data[18] = 0;
    data[19] = 0;
    data[20] = 0;
    data[21] = 0;

    for (i = 0; i < 23; ++i) {
        *pCRC32 = drflac_crc32_byte(*pCRC32, data[i]);
    }


    if (onRead(pUserData, pHeader->segmentTable, pHeader->segmentCount) != pHeader->segmentCount) {
        return DRFLAC_AT_END;
    }
    *pBytesRead += pHeader->segmentCount;

    for (i = 0; i < pHeader->segmentCount; ++i) {
        *pCRC32 = drflac_crc32_byte(*pCRC32, pHeader->segmentTable[i]);
    }

    return DRFLAC_SUCCESS;
}

static drflac_result drflac_ogg__read_page_header(drflac_read_proc onRead, void* pUserData, drflac_ogg_page_header* pHeader, drflac_uint32* pBytesRead, drflac_uint32* pCRC32)
{
    drflac_uint8 id[4];

    *pBytesRead = 0;

    if (onRead(pUserData, id, 4) != 4) {
        return DRFLAC_AT_END;
    }
    *pBytesRead += 4;

    /* We need to read byte-by-byte until we find the OggS capture pattern. */
    for (;;) {
        if (drflac_ogg__is_capture_pattern(id)) {
            drflac_result result;

            *pCRC32 = DRFLAC_OGG_CAPTURE_PATTERN_CRC32;

            result = drflac_ogg__read_page_header_after_capture_pattern(onRead, pUserData, pHeader, pBytesRead, pCRC32);
            if (result == DRFLAC_SUCCESS) {
                return DRFLAC_SUCCESS;
            } else {
                if (result == DRFLAC_CRC_MISMATCH) {
                    continue;
                } else {
                    return result;
                }
            }
        } else {
            /* The first 4 bytes did not equal the capture pattern. Read the next byte and try again. */
            id[0] = id[1];
            id[1] = id[2];
            id[2] = id[3];
            if (onRead(pUserData, &id[3], 1) != 1) {
                return DRFLAC_AT_END;
            }
            *pBytesRead += 1;
        }
    }
}


/*
The main part of the Ogg encapsulation is the conversion from the physical Ogg bitstream to the native FLAC bitstream. It works
in three general stages: Ogg Physical Bitstream -> Ogg/FLAC Logical Bitstream -> FLAC Native Bitstream. dr_flac is designed
in such a way that the core sections assume everything is delivered in native format. Therefore, for each encapsulation type
dr_flac is supporting there needs to be a layer sitting on top of the onRead and onSeek callbacks that ensures the bits read from
the physical Ogg bitstream are converted and delivered in native FLAC format.
*/
typedef struct
{
    drflac_read_proc onRead;                /* The original onRead callback from drflac_open() and family. */
    drflac_seek_proc onSeek;                /* The original onSeek callback from drflac_open() and family. */
    void* pUserData;                        /* The user data passed on onRead and onSeek. This is the user data that was passed on drflac_open() and family. */
    drflac_uint64 currentBytePos;           /* The position of the byte we are sitting on in the physical byte stream. Used for efficient seeking. */
    drflac_uint64 firstBytePos;             /* The position of the first byte in the physical bitstream. Points to the start of the "OggS" identifier of the FLAC bos page. */
    drflac_uint32 serialNumber;             /* The serial number of the FLAC audio pages. This is determined by the initial header page that was read during initialization. */
    drflac_ogg_page_header bosPageHeader;   /* Used for seeking. */
    drflac_ogg_page_header currentPageHeader;
    drflac_uint32 bytesRemainingInPage;
    drflac_uint32 pageDataSize;
    drflac_uint8 pageData[DRFLAC_OGG_MAX_PAGE_SIZE];
} drflac_oggbs; /* oggbs = Ogg Bitstream */

static size_t drflac_oggbs__read_physical(drflac_oggbs* oggbs, void* bufferOut, size_t bytesToRead)
{
    size_t bytesActuallyRead = oggbs->onRead(oggbs->pUserData, bufferOut, bytesToRead);
    oggbs->currentBytePos += bytesActuallyRead;

    return bytesActuallyRead;
}

static drflac_bool32 drflac_oggbs__seek_physical(drflac_oggbs* oggbs, drflac_uint64 offset, drflac_seek_origin origin)
{
    if (origin == drflac_seek_origin_start) {
        if (offset <= 0x7FFFFFFF) {
            if (!oggbs->onSeek(oggbs->pUserData, (int)offset, drflac_seek_origin_start)) {
                return DRFLAC_FALSE;
            }
            oggbs->currentBytePos = offset;

            return DRFLAC_TRUE;
        } else {
            if (!oggbs->onSeek(oggbs->pUserData, 0x7FFFFFFF, drflac_seek_origin_start)) {
                return DRFLAC_FALSE;
            }
            oggbs->currentBytePos = offset;

            return drflac_oggbs__seek_physical(oggbs, offset - 0x7FFFFFFF, drflac_seek_origin_current);
        }
    } else {
        while (offset > 0x7FFFFFFF) {
            if (!oggbs->onSeek(oggbs->pUserData, 0x7FFFFFFF, drflac_seek_origin_current)) {
                return DRFLAC_FALSE;
            }
            oggbs->currentBytePos += 0x7FFFFFFF;
            offset -= 0x7FFFFFFF;
        }

        if (!oggbs->onSeek(oggbs->pUserData, (int)offset, drflac_seek_origin_current)) {    /* <-- Safe cast thanks to the loop above. */
            return DRFLAC_FALSE;
        }
        oggbs->currentBytePos += offset;

        return DRFLAC_TRUE;
    }
}

static drflac_bool32 drflac_oggbs__goto_next_page(drflac_oggbs* oggbs, drflac_ogg_crc_mismatch_recovery recoveryMethod)
{
    drflac_ogg_page_header header;
    for (;;) {
        drflac_uint32 crc32 = 0;
        drflac_uint32 bytesRead;
        drflac_uint32 pageBodySize;
#ifndef DR_FLAC_NO_CRC
        drflac_uint32 actualCRC32;
#endif

        if (drflac_ogg__read_page_header(oggbs->onRead, oggbs->pUserData, &header, &bytesRead, &crc32) != DRFLAC_SUCCESS) {
            return DRFLAC_FALSE;
        }
        oggbs->currentBytePos += bytesRead;

        pageBodySize = drflac_ogg__get_page_body_size(&header);
        if (pageBodySize > DRFLAC_OGG_MAX_PAGE_SIZE) {
            continue;   /* Invalid page size. Assume it's corrupted and just move to the next page. */
        }

        if (header.serialNumber != oggbs->serialNumber) {
            /* It's not a FLAC page. Skip it. */
            if (pageBodySize > 0 && !drflac_oggbs__seek_physical(oggbs, pageBodySize, drflac_seek_origin_current)) {
                return DRFLAC_FALSE;
            }
            continue;
        }


        /* We need to read the entire page and then do a CRC check on it. If there's a CRC mismatch we need to skip this page. */
        if (drflac_oggbs__read_physical(oggbs, oggbs->pageData, pageBodySize) != pageBodySize) {
            return DRFLAC_FALSE;
        }
        oggbs->pageDataSize = pageBodySize;

#ifndef DR_FLAC_NO_CRC
        actualCRC32 = drflac_crc32_buffer(crc32, oggbs->pageData, oggbs->pageDataSize);
        if (actualCRC32 != header.checksum) {
            if (recoveryMethod == drflac_ogg_recover_on_crc_mismatch) {
                continue;   /* CRC mismatch. Skip this page. */
            } else {
                /*
                Even though we are failing on a CRC mismatch, we still want our stream to be in a good state. Therefore we
                go to the next valid page to ensure we're in a good state, but return false to let the caller know that the
                seek did not fully complete.
                */
                drflac_oggbs__goto_next_page(oggbs, drflac_ogg_recover_on_crc_mismatch);
                return DRFLAC_FALSE;
            }
        }
#else
        (void)recoveryMethod;   /* <-- Silence a warning. */
#endif

        oggbs->currentPageHeader = header;
        oggbs->bytesRemainingInPage = pageBodySize;
        return DRFLAC_TRUE;
    }
}

/* Function below is unused at the moment, but I might be re-adding it later. */
#if 0
static drflac_uint8 drflac_oggbs__get_current_segment_index(drflac_oggbs* oggbs, drflac_uint8* pBytesRemainingInSeg)
{
    drflac_uint32 bytesConsumedInPage = drflac_ogg__get_page_body_size(&oggbs->currentPageHeader) - oggbs->bytesRemainingInPage;
    drflac_uint8 iSeg = 0;
    drflac_uint32 iByte = 0;
    while (iByte < bytesConsumedInPage) {
        drflac_uint8 segmentSize = oggbs->currentPageHeader.segmentTable[iSeg];
        if (iByte + segmentSize > bytesConsumedInPage) {
            break;
        } else {
            iSeg += 1;
            iByte += segmentSize;
        }
    }

    *pBytesRemainingInSeg = oggbs->currentPageHeader.segmentTable[iSeg] - (drflac_uint8)(bytesConsumedInPage - iByte);
    return iSeg;
}

static drflac_bool32 drflac_oggbs__seek_to_next_packet(drflac_oggbs* oggbs)
{
    /* The current packet ends when we get to the segment with a lacing value of < 255 which is not at the end of a page. */
    for (;;) {
        drflac_bool32 atEndOfPage = DRFLAC_FALSE;

        drflac_uint8 bytesRemainingInSeg;
        drflac_uint8 iFirstSeg = drflac_oggbs__get_current_segment_index(oggbs, &bytesRemainingInSeg);

        drflac_uint32 bytesToEndOfPacketOrPage = bytesRemainingInSeg;
        for (drflac_uint8 iSeg = iFirstSeg; iSeg < oggbs->currentPageHeader.segmentCount; ++iSeg) {
            drflac_uint8 segmentSize = oggbs->currentPageHeader.segmentTable[iSeg];
            if (segmentSize < 255) {
                if (iSeg == oggbs->currentPageHeader.segmentCount-1) {
                    atEndOfPage = DRFLAC_TRUE;
                }

                break;
            }

            bytesToEndOfPacketOrPage += segmentSize;
        }

        /*
        At this point we will have found either the packet or the end of the page. If were at the end of the page we'll
        want to load the next page and keep searching for the end of the packet.
        */
        drflac_oggbs__seek_physical(oggbs, bytesToEndOfPacketOrPage, drflac_seek_origin_current);
        oggbs->bytesRemainingInPage -= bytesToEndOfPacketOrPage;

        if (atEndOfPage) {
            /*
            We're potentially at the next packet, but we need to check the next page first to be sure because the packet may
            straddle pages.
            */
            if (!drflac_oggbs__goto_next_page(oggbs)) {
                return DRFLAC_FALSE;
            }

            /* If it's a fresh packet it most likely means we're at the next packet. */
            if ((oggbs->currentPageHeader.headerType & 0x01) == 0) {
                return DRFLAC_TRUE;
            }
        } else {
            /* We're at the next packet. */
            return DRFLAC_TRUE;
        }
    }
}

static drflac_bool32 drflac_oggbs__seek_to_next_frame(drflac_oggbs* oggbs)
{
    /* The bitstream should be sitting on the first byte just after the header of the frame. */

    /* What we're actually doing here is seeking to the start of the next packet. */
    return drflac_oggbs__seek_to_next_packet(oggbs);
}
#endif

static size_t drflac__on_read_ogg(void* pUserData, void* bufferOut, size_t bytesToRead)
{
    drflac_oggbs* oggbs = (drflac_oggbs*)pUserData;
    drflac_uint8* pRunningBufferOut = (drflac_uint8*)bufferOut;
    size_t bytesRead = 0;

    DRFLAC_ASSERT(oggbs != NULL);
    DRFLAC_ASSERT(pRunningBufferOut != NULL);

    /* Reading is done page-by-page. If we've run out of bytes in the page we need to move to the next one. */
    while (bytesRead < bytesToRead) {
        size_t bytesRemainingToRead = bytesToRead - bytesRead;

        if (oggbs->bytesRemainingInPage >= bytesRemainingToRead) {
            DRFLAC_COPY_MEMORY(pRunningBufferOut, oggbs->pageData + (oggbs->pageDataSize - oggbs->bytesRemainingInPage), bytesRemainingToRead);
            bytesRead += bytesRemainingToRead;
            oggbs->bytesRemainingInPage -= (drflac_uint32)bytesRemainingToRead;
            break;
        }

        /* If we get here it means some of the requested data is contained in the next pages. */
        if (oggbs->bytesRemainingInPage > 0) {
            DRFLAC_COPY_MEMORY(pRunningBufferOut, oggbs->pageData + (oggbs->pageDataSize - oggbs->bytesRemainingInPage), oggbs->bytesRemainingInPage);
            bytesRead += oggbs->bytesRemainingInPage;
            pRunningBufferOut += oggbs->bytesRemainingInPage;
            oggbs->bytesRemainingInPage = 0;
        }

        DRFLAC_ASSERT(bytesRemainingToRead > 0);
        if (!drflac_oggbs__goto_next_page(oggbs, drflac_ogg_recover_on_crc_mismatch)) {
            break;  /* Failed to go to the next page. Might have simply hit the end of the stream. */
        }
    }

    return bytesRead;
}

static drflac_bool32 drflac__on_seek_ogg(void* pUserData, int offset, drflac_seek_origin origin)
{
    drflac_oggbs* oggbs = (drflac_oggbs*)pUserData;
    int bytesSeeked = 0;

    DRFLAC_ASSERT(oggbs != NULL);
    DRFLAC_ASSERT(offset >= 0);  /* <-- Never seek backwards. */

    /* Seeking is always forward which makes things a lot simpler. */
    if (origin == drflac_seek_origin_start) {
        if (!drflac_oggbs__seek_physical(oggbs, (int)oggbs->firstBytePos, drflac_seek_origin_start)) {
            return DRFLAC_FALSE;
        }

        if (!drflac_oggbs__goto_next_page(oggbs, drflac_ogg_fail_on_crc_mismatch)) {
            return DRFLAC_FALSE;
        }

        return drflac__on_seek_ogg(pUserData, offset, drflac_seek_origin_current);
    }

    DRFLAC_ASSERT(origin == drflac_seek_origin_current);

    while (bytesSeeked < offset) {
        int bytesRemainingToSeek = offset - bytesSeeked;
        DRFLAC_ASSERT(bytesRemainingToSeek >= 0);

        if (oggbs->bytesRemainingInPage >= (size_t)bytesRemainingToSeek) {
            bytesSeeked += bytesRemainingToSeek;
            (void)bytesSeeked;  /* <-- Silence a dead store warning emitted by Clang Static Analyzer. */
            oggbs->bytesRemainingInPage -= bytesRemainingToSeek;
            break;
        }

        /* If we get here it means some of the requested data is contained in the next pages. */
        if (oggbs->bytesRemainingInPage > 0) {
            bytesSeeked += (int)oggbs->bytesRemainingInPage;
            oggbs->bytesRemainingInPage = 0;
        }

        DRFLAC_ASSERT(bytesRemainingToSeek > 0);
        if (!drflac_oggbs__goto_next_page(oggbs, drflac_ogg_fail_on_crc_mismatch)) {
            /* Failed to go to the next page. We either hit the end of the stream or had a CRC mismatch. */
            return DRFLAC_FALSE;
        }
    }

    return DRFLAC_TRUE;
}


static drflac_bool32 drflac_ogg__seek_to_pcm_frame(drflac* pFlac, drflac_uint64 pcmFrameIndex)
{
    drflac_oggbs* oggbs = (drflac_oggbs*)pFlac->_oggbs;
    drflac_uint64 originalBytePos;
    drflac_uint64 runningGranulePosition;
    drflac_uint64 runningFrameBytePos;
    drflac_uint64 runningPCMFrameCount;

    DRFLAC_ASSERT(oggbs != NULL);

    originalBytePos = oggbs->currentBytePos;   /* For recovery. Points to the OggS identifier. */

    /* First seek to the first frame. */
    if (!drflac__seek_to_byte(&pFlac->bs, pFlac->firstFLACFramePosInBytes)) {
        return DRFLAC_FALSE;
    }
    oggbs->bytesRemainingInPage = 0;

    runningGranulePosition = 0;
    for (;;) {
        if (!drflac_oggbs__goto_next_page(oggbs, drflac_ogg_recover_on_crc_mismatch)) {
            drflac_oggbs__seek_physical(oggbs, originalBytePos, drflac_seek_origin_start);
            return DRFLAC_FALSE;   /* Never did find that sample... */
        }

        runningFrameBytePos = oggbs->currentBytePos - drflac_ogg__get_page_header_size(&oggbs->currentPageHeader) - oggbs->pageDataSize;
        if (oggbs->currentPageHeader.granulePosition >= pcmFrameIndex) {
            break; /* The sample is somewhere in the previous page. */
        }

        /*
        At this point we know the sample is not in the previous page. It could possibly be in this page. For simplicity we
        disregard any pages that do not begin a fresh packet.
        */
        if ((oggbs->currentPageHeader.headerType & 0x01) == 0) {    /* <-- Is it a fresh page? */
            if (oggbs->currentPageHeader.segmentTable[0] >= 2) {
                drflac_uint8 firstBytesInPage[2];
                firstBytesInPage[0] = oggbs->pageData[0];
                firstBytesInPage[1] = oggbs->pageData[1];

                if ((firstBytesInPage[0] == 0xFF) && (firstBytesInPage[1] & 0xFC) == 0xF8) {    /* <-- Does the page begin with a frame's sync code? */
                    runningGranulePosition = oggbs->currentPageHeader.granulePosition;
                }

                continue;
            }
        }
    }

    /*
    We found the page that that is closest to the sample, so now we need to find it. The first thing to do is seek to the
    start of that page. In the loop above we checked that it was a fresh page which means this page is also the start of
    a new frame. This property means that after we've seeked to the page we can immediately start looping over frames until
    we find the one containing the target sample.
    */
    if (!drflac_oggbs__seek_physical(oggbs, runningFrameBytePos, drflac_seek_origin_start)) {
        return DRFLAC_FALSE;
    }
    if (!drflac_oggbs__goto_next_page(oggbs, drflac_ogg_recover_on_crc_mismatch)) {
        return DRFLAC_FALSE;
    }

    /*
    At this point we'll be sitting on the first byte of the frame header of the first frame in the page. We just keep
    looping over these frames until we find the one containing the sample we're after.
    */
    runningPCMFrameCount = runningGranulePosition;
    for (;;) {
        /*
        There are two ways to find the sample and seek past irrelevant frames:
          1) Use the native FLAC decoder.
          2) Use Ogg's framing system.

        Both of these options have their own pros and cons. Using the native FLAC decoder is slower because it needs to
        do a full decode of the frame. Using Ogg's framing system is faster, but more complicated and involves some code
        duplication for the decoding of frame headers.

        Another thing to consider is that using the Ogg framing system will perform direct seeking of the physical Ogg
        bitstream. This is important to consider because it means we cannot read data from the drflac_bs object using the
        standard drflac__*() APIs because that will read in extra data for its own internal caching which in turn breaks
        the positioning of the read pointer of the physical Ogg bitstream. Therefore, anything that would normally be read
        using the native FLAC decoding APIs, such as drflac__read_next_flac_frame_header(), need to be re-implemented so as to
        avoid the use of the drflac_bs object.

        Considering these issues, I have decided to use the slower native FLAC decoding method for the following reasons:
          1) Seeking is already partially accelerated using Ogg's paging system in the code block above.
          2) Seeking in an Ogg encapsulated FLAC stream is probably quite uncommon.
          3) Simplicity.
        */
        drflac_uint64 firstPCMFrameInFLACFrame = 0;
        drflac_uint64 lastPCMFrameInFLACFrame = 0;
        drflac_uint64 pcmFrameCountInThisFrame;

        if (!drflac__read_next_flac_frame_header(&pFlac->bs, pFlac->bitsPerSample, &pFlac->currentFLACFrame.header)) {
            return DRFLAC_FALSE;
        }

        drflac__get_pcm_frame_range_of_current_flac_frame(pFlac, &firstPCMFrameInFLACFrame, &lastPCMFrameInFLACFrame);

        pcmFrameCountInThisFrame = (lastPCMFrameInFLACFrame - firstPCMFrameInFLACFrame) + 1;

        /* If we are seeking to the end of the file and we've just hit it, we're done. */
        if (pcmFrameIndex == pFlac->totalPCMFrameCount && (runningPCMFrameCount + pcmFrameCountInThisFrame) == pFlac->totalPCMFrameCount) {
            drflac_result result = drflac__decode_flac_frame(pFlac);
            if (result == DRFLAC_SUCCESS) {
                pFlac->currentPCMFrame = pcmFrameIndex;
                pFlac->currentFLACFrame.pcmFramesRemaining = 0;
                return DRFLAC_TRUE;
            } else {
                return DRFLAC_FALSE;
            }
        }

        if (pcmFrameIndex < (runningPCMFrameCount + pcmFrameCountInThisFrame)) {
            /*
            The sample should be in this FLAC frame. We need to fully decode it, however if it's an invalid frame (a CRC mismatch), we need to pretend
            it never existed and keep iterating.
            */
            drflac_result result = drflac__decode_flac_frame(pFlac);
            if (result == DRFLAC_SUCCESS) {
                /* The frame is valid. We just need to skip over some samples to ensure it's sample-exact. */
                drflac_uint64 pcmFramesToDecode = (size_t)(pcmFrameIndex - runningPCMFrameCount);    /* <-- Safe cast because the maximum number of samples in a frame is 65535. */
                if (pcmFramesToDecode == 0) {
                    return DRFLAC_TRUE;
                }

                pFlac->currentPCMFrame = runningPCMFrameCount;

                return drflac__seek_forward_by_pcm_frames(pFlac, pcmFramesToDecode) == pcmFramesToDecode;  /* <-- If this fails, something bad has happened (it should never fail). */
            } else {
                if (result == DRFLAC_CRC_MISMATCH) {
                    continue;   /* CRC mismatch. Pretend this frame never existed. */
                } else {
                    return DRFLAC_FALSE;
                }
            }
        } else {
            /*
            It's not in this frame. We need to seek past the frame, but check if there was a CRC mismatch. If so, we pretend this
            frame never existed and leave the running sample count untouched.
            */
            drflac_result result = drflac__seek_to_next_flac_frame(pFlac);
            if (result == DRFLAC_SUCCESS) {
                runningPCMFrameCount += pcmFrameCountInThisFrame;
            } else {
                if (result == DRFLAC_CRC_MISMATCH) {
                    continue;   /* CRC mismatch. Pretend this frame never existed. */
                } else {
                    return DRFLAC_FALSE;
                }
            }
        }
    }
}



static drflac_bool32 drflac__init_private__ogg(drflac_init_info* pInit, drflac_read_proc onRead, drflac_seek_proc onSeek, drflac_meta_proc onMeta, void* pUserData, void* pUserDataMD, drflac_bool32 relaxed)
{
    drflac_ogg_page_header header;
    drflac_uint32 crc32 = DRFLAC_OGG_CAPTURE_PATTERN_CRC32;
    drflac_uint32 bytesRead = 0;

    /* Pre Condition: The bit stream should be sitting just past the 4-byte OggS capture pattern. */
    (void)relaxed;

    pInit->container = drflac_container_ogg;
    pInit->oggFirstBytePos = 0;

    /*
    We'll get here if the first 4 bytes of the stream were the OggS capture pattern, however it doesn't necessarily mean the
    stream includes FLAC encoded audio. To check for this we need to scan the beginning-of-stream page markers and check if
    any match the FLAC specification. Important to keep in mind that the stream may be multiplexed.
    */
    if (drflac_ogg__read_page_header_after_capture_pattern(onRead, pUserData, &header, &bytesRead, &crc32) != DRFLAC_SUCCESS) {
        return DRFLAC_FALSE;
    }
    pInit->runningFilePos += bytesRead;

    for (;;) {
        int pageBodySize;

        /* Break if we're past the beginning of stream page. */
        if ((header.headerType & 0x02) == 0) {
            return DRFLAC_FALSE;
        }

        /* Check if it's a FLAC header. */
        pageBodySize = drflac_ogg__get_page_body_size(&header);
        if (pageBodySize == 51) {   /* 51 = the lacing value of the FLAC header packet. */
            /* It could be a FLAC page... */
            drflac_uint32 bytesRemainingInPage = pageBodySize;
            drflac_uint8 packetType;

            if (onRead(pUserData, &packetType, 1) != 1) {
                return DRFLAC_FALSE;
            }

            bytesRemainingInPage -= 1;
            if (packetType == 0x7F) {
                /* Increasingly more likely to be a FLAC page... */
                drflac_uint8 sig[4];
                if (onRead(pUserData, sig, 4) != 4) {
                    return DRFLAC_FALSE;
                }

                bytesRemainingInPage -= 4;
                if (sig[0] == 'F' && sig[1] == 'L' && sig[2] == 'A' && sig[3] == 'C') {
                    /* Almost certainly a FLAC page... */
                    drflac_uint8 mappingVersion[2];
                    if (onRead(pUserData, mappingVersion, 2) != 2) {
                        return DRFLAC_FALSE;
                    }

                    if (mappingVersion[0] != 1) {
                        return DRFLAC_FALSE;   /* Only supporting version 1.x of the Ogg mapping. */
                    }

                    /*
                    The next 2 bytes are the non-audio packets, not including this one. We don't care about this because we're going to
                    be handling it in a generic way based on the serial number and packet types.
                    */
                    if (!onSeek(pUserData, 2, drflac_seek_origin_current)) {
                        return DRFLAC_FALSE;
                    }

                    /* Expecting the native FLAC signature "fLaC". */
                    if (onRead(pUserData, sig, 4) != 4) {
                        return DRFLAC_FALSE;
                    }

                    if (sig[0] == 'f' && sig[1] == 'L' && sig[2] == 'a' && sig[3] == 'C') {
                        /* The remaining data in the page should be the STREAMINFO block. */
                        drflac_streaminfo streaminfo;
                        drflac_uint8 isLastBlock;
                        drflac_uint8 blockType;
                        drflac_uint32 blockSize;
                        if (!drflac__read_and_decode_block_header(onRead, pUserData, &isLastBlock, &blockType, &blockSize)) {
                            return DRFLAC_FALSE;
                        }

                        if (blockType != DRFLAC_METADATA_BLOCK_TYPE_STREAMINFO || blockSize != 34) {
                            return DRFLAC_FALSE;    /* Invalid block type. First block must be the STREAMINFO block. */
                        }

                        if (drflac__read_streaminfo(onRead, pUserData, &streaminfo)) {
                            /* Success! */
                            pInit->hasStreamInfoBlock      = DRFLAC_TRUE;
                            pInit->sampleRate              = streaminfo.sampleRate;
                            pInit->channels                = streaminfo.channels;
                            pInit->bitsPerSample           = streaminfo.bitsPerSample;
                            pInit->totalPCMFrameCount      = streaminfo.totalPCMFrameCount;
                            pInit->maxBlockSizeInPCMFrames = streaminfo.maxBlockSizeInPCMFrames;
                            pInit->hasMetadataBlocks       = !isLastBlock;

                            if (onMeta) {
                                drflac_metadata metadata;
                                metadata.type = DRFLAC_METADATA_BLOCK_TYPE_STREAMINFO;
                                metadata.pRawData = NULL;
                                metadata.rawDataSize = 0;
                                metadata.data.streaminfo = streaminfo;
                                onMeta(pUserDataMD, &metadata);
                            }

                            pInit->runningFilePos  += pageBodySize;
                            pInit->oggFirstBytePos  = pInit->runningFilePos - 79;   /* Subtracting 79 will place us right on top of the "OggS" identifier of the FLAC bos page. */
                            pInit->oggSerial        = header.serialNumber;
                            pInit->oggBosHeader     = header;
                            break;
                        } else {
                            /* Failed to read STREAMINFO block. Aww, so close... */
                            return DRFLAC_FALSE;
                        }
                    } else {
                        /* Invalid file. */
                        return DRFLAC_FALSE;
                    }
                } else {
                    /* Not a FLAC header. Skip it. */
                    if (!onSeek(pUserData, bytesRemainingInPage, drflac_seek_origin_current)) {
                        return DRFLAC_FALSE;
                    }
                }
            } else {
                /* Not a FLAC header. Seek past the entire page and move on to the next. */
                if (!onSeek(pUserData, bytesRemainingInPage, drflac_seek_origin_current)) {
                    return DRFLAC_FALSE;
                }
            }
        } else {
            if (!onSeek(pUserData, pageBodySize, drflac_seek_origin_current)) {
                return DRFLAC_FALSE;
            }
        }

        pInit->runningFilePos += pageBodySize;


        /* Read the header of the next page. */
        if (drflac_ogg__read_page_header(onRead, pUserData, &header, &bytesRead, &crc32) != DRFLAC_SUCCESS) {
            return DRFLAC_FALSE;
        }
        pInit->runningFilePos += bytesRead;
    }

    /*
    If we get here it means we found a FLAC audio stream. We should be sitting on the first byte of the header of the next page. The next
    packets in the FLAC logical stream contain the metadata. The only thing left to do in the initialization phase for Ogg is to create the
    Ogg bistream object.
    */
    pInit->hasMetadataBlocks = DRFLAC_TRUE;    /* <-- Always have at least VORBIS_COMMENT metadata block. */
    return DRFLAC_TRUE;
}
#endif

static drflac_bool32 drflac__init_private(drflac_init_info* pInit, drflac_read_proc onRead, drflac_seek_proc onSeek, drflac_meta_proc onMeta, drflac_container container, void* pUserData, void* pUserDataMD)
{
    drflac_bool32 relaxed;
    drflac_uint8 id[4];

    if (pInit == NULL || onRead == NULL || onSeek == NULL) {
        return DRFLAC_FALSE;
    }

    DRFLAC_ZERO_MEMORY(pInit, sizeof(*pInit));
    pInit->onRead       = onRead;
    pInit->onSeek       = onSeek;
    pInit->onMeta       = onMeta;
    pInit->container    = container;
    pInit->pUserData    = pUserData;
    pInit->pUserDataMD  = pUserDataMD;

    pInit->bs.onRead    = onRead;
    pInit->bs.onSeek    = onSeek;
    pInit->bs.pUserData = pUserData;
    drflac__reset_cache(&pInit->bs);


    /* If the container is explicitly defined then we can try opening in relaxed mode. */
    relaxed = container != drflac_container_unknown;

    /* Skip over any ID3 tags. */
    for (;;) {
        if (onRead(pUserData, id, 4) != 4) {
            return DRFLAC_FALSE;    /* Ran out of data. */
        }
        pInit->runningFilePos += 4;

        if (id[0] == 'I' && id[1] == 'D' && id[2] == '3') {
            drflac_uint8 header[6];
            drflac_uint8 flags;
            drflac_uint32 headerSize;

            if (onRead(pUserData, header, 6) != 6) {
                return DRFLAC_FALSE;    /* Ran out of data. */
            }
            pInit->runningFilePos += 6;

            flags = header[1];

            DRFLAC_COPY_MEMORY(&headerSize, header+2, 4);
            headerSize = drflac__unsynchsafe_32(drflac__be2host_32(headerSize));
            if (flags & 0x10) {
                headerSize += 10;
            }

            if (!onSeek(pUserData, headerSize, drflac_seek_origin_current)) {
                return DRFLAC_FALSE;    /* Failed to seek past the tag. */
            }
            pInit->runningFilePos += headerSize;
        } else {
            break;
        }
    }

    if (id[0] == 'f' && id[1] == 'L' && id[2] == 'a' && id[3] == 'C') {
        return drflac__init_private__native(pInit, onRead, onSeek, onMeta, pUserData, pUserDataMD, relaxed);
    }
#ifndef DR_FLAC_NO_OGG
    if (id[0] == 'O' && id[1] == 'g' && id[2] == 'g' && id[3] == 'S') {
        return drflac__init_private__ogg(pInit, onRead, onSeek, onMeta, pUserData, pUserDataMD, relaxed);
    }
#endif

    /* If we get here it means we likely don't have a header. Try opening in relaxed mode, if applicable. */
    if (relaxed) {
        if (container == drflac_container_native) {
            return drflac__init_private__native(pInit, onRead, onSeek, onMeta, pUserData, pUserDataMD, relaxed);
        }
#ifndef DR_FLAC_NO_OGG
        if (container == drflac_container_ogg) {
            return drflac__init_private__ogg(pInit, onRead, onSeek, onMeta, pUserData, pUserDataMD, relaxed);
        }
#endif
    }

    /* Unsupported container. */
    return DRFLAC_FALSE;
}

static void drflac__init_from_info(drflac* pFlac, const drflac_init_info* pInit)
{
    DRFLAC_ASSERT(pFlac != NULL);
    DRFLAC_ASSERT(pInit != NULL);

    DRFLAC_ZERO_MEMORY(pFlac, sizeof(*pFlac));
    pFlac->bs                      = pInit->bs;
    pFlac->onMeta                  = pInit->onMeta;
    pFlac->pUserDataMD             = pInit->pUserDataMD;
    pFlac->maxBlockSizeInPCMFrames = pInit->maxBlockSizeInPCMFrames;
    pFlac->sampleRate              = pInit->sampleRate;
    pFlac->channels                = (drflac_uint8)pInit->channels;
    pFlac->bitsPerSample           = (drflac_uint8)pInit->bitsPerSample;
    pFlac->totalPCMFrameCount      = pInit->totalPCMFrameCount;
    pFlac->container               = pInit->container;
}


static drflac* drflac_open_with_metadata_private(drflac_read_proc onRead, drflac_seek_proc onSeek, drflac_meta_proc onMeta, drflac_container container, void* pUserData, void* pUserDataMD, const drflac_allocation_callbacks* pAllocationCallbacks)
{
    drflac_init_info init;
    drflac_uint32 allocationSize;
    drflac_uint32 wholeSIMDVectorCountPerChannel;
    drflac_uint32 decodedSamplesAllocationSize;
#ifndef DR_FLAC_NO_OGG
    drflac_oggbs* pOggbs = NULL;
#endif
    drflac_uint64 firstFramePos;
    drflac_uint64 seektablePos;
    drflac_uint32 seekpointCount;
    drflac_allocation_callbacks allocationCallbacks;
    drflac* pFlac;

    /* CPU support first. */
    drflac__init_cpu_caps();

    if (!drflac__init_private(&init, onRead, onSeek, onMeta, container, pUserData, pUserDataMD)) {
        return NULL;
    }

    if (pAllocationCallbacks != NULL) {
        allocationCallbacks = *pAllocationCallbacks;
        if (allocationCallbacks.onFree == NULL || (allocationCallbacks.onMalloc == NULL && allocationCallbacks.onRealloc == NULL)) {
            return NULL;    /* Invalid allocation callbacks. */
        }
    } else {
        allocationCallbacks.pUserData = NULL;
        allocationCallbacks.onMalloc  = drflac__malloc_default;
        allocationCallbacks.onRealloc = drflac__realloc_default;
        allocationCallbacks.onFree    = drflac__free_default;
    }


    /*
    The size of the allocation for the drflac object needs to be large enough to fit the following:
      1) The main members of the drflac structure
      2) A block of memory large enough to store the decoded samples of the largest frame in the stream
      3) If the container is Ogg, a drflac_oggbs object

    The complicated part of the allocation is making sure there's enough room the decoded samples, taking into consideration
    the different SIMD instruction sets.
    */
    allocationSize = sizeof(drflac);

    /*
    The allocation size for decoded frames depends on the number of 32-bit integers that fit inside the largest SIMD vector
    we are supporting.
    */
    if ((init.maxBlockSizeInPCMFrames % (DRFLAC_MAX_SIMD_VECTOR_SIZE / sizeof(drflac_int32))) == 0) {
        wholeSIMDVectorCountPerChannel = (init.maxBlockSizeInPCMFrames / (DRFLAC_MAX_SIMD_VECTOR_SIZE / sizeof(drflac_int32)));
    } else {
        wholeSIMDVectorCountPerChannel = (init.maxBlockSizeInPCMFrames / (DRFLAC_MAX_SIMD_VECTOR_SIZE / sizeof(drflac_int32))) + 1;
    }

    decodedSamplesAllocationSize = wholeSIMDVectorCountPerChannel * DRFLAC_MAX_SIMD_VECTOR_SIZE * init.channels;

    allocationSize += decodedSamplesAllocationSize;
    allocationSize += DRFLAC_MAX_SIMD_VECTOR_SIZE;  /* Allocate extra bytes to ensure we have enough for alignment. */

#ifndef DR_FLAC_NO_OGG
    /* There's additional data required for Ogg streams. */
    if (init.container == drflac_container_ogg) {
        allocationSize += sizeof(drflac_oggbs);

        pOggbs = (drflac_oggbs*)drflac__malloc_from_callbacks(sizeof(*pOggbs), &allocationCallbacks);
        if (pOggbs == NULL) {
            return NULL; /*DRFLAC_OUT_OF_MEMORY;*/
        }

        DRFLAC_ZERO_MEMORY(pOggbs, sizeof(*pOggbs));
        pOggbs->onRead = onRead;
        pOggbs->onSeek = onSeek;
        pOggbs->pUserData = pUserData;
        pOggbs->currentBytePos = init.oggFirstBytePos;
        pOggbs->firstBytePos = init.oggFirstBytePos;
        pOggbs->serialNumber = init.oggSerial;
        pOggbs->bosPageHeader = init.oggBosHeader;
        pOggbs->bytesRemainingInPage = 0;
    }
#endif

    /*
    This part is a bit awkward. We need to load the seektable so that it can be referenced in-memory, but I want the drflac object to
    consist of only a single heap allocation. To this, the size of the seek table needs to be known, which we determine when reading
    and decoding the metadata.
    */
    firstFramePos  = 42;   /* <-- We know we are at byte 42 at this point. */
    seektablePos   = 0;
    seekpointCount = 0;
    if (init.hasMetadataBlocks) {
        drflac_read_proc onReadOverride = onRead;
        drflac_seek_proc onSeekOverride = onSeek;
        void* pUserDataOverride = pUserData;

#ifndef DR_FLAC_NO_OGG
        if (init.container == drflac_container_ogg) {
            onReadOverride = drflac__on_read_ogg;
            onSeekOverride = drflac__on_seek_ogg;
            pUserDataOverride = (void*)pOggbs;
        }
#endif

        if (!drflac__read_and_decode_metadata(onReadOverride, onSeekOverride, onMeta, pUserDataOverride, pUserDataMD, &firstFramePos, &seektablePos, &seekpointCount, &allocationCallbacks)) {
        #ifndef DR_FLAC_NO_OGG
            drflac__free_from_callbacks(pOggbs, &allocationCallbacks);
        #endif
            return NULL;
        }

        allocationSize += seekpointCount * sizeof(drflac_seekpoint);
    }


    pFlac = (drflac*)drflac__malloc_from_callbacks(allocationSize, &allocationCallbacks);
    if (pFlac == NULL) {
    #ifndef DR_FLAC_NO_OGG
        drflac__free_from_callbacks(pOggbs, &allocationCallbacks);
    #endif
        return NULL;
    }

    drflac__init_from_info(pFlac, &init);
    pFlac->allocationCallbacks = allocationCallbacks;
    pFlac->pDecodedSamples = (drflac_int32*)drflac_align((size_t)pFlac->pExtraData, DRFLAC_MAX_SIMD_VECTOR_SIZE);

#ifndef DR_FLAC_NO_OGG
    if (init.container == drflac_container_ogg) {
        drflac_oggbs* pInternalOggbs = (drflac_oggbs*)((drflac_uint8*)pFlac->pDecodedSamples + decodedSamplesAllocationSize + (seekpointCount * sizeof(drflac_seekpoint)));
        DRFLAC_COPY_MEMORY(pInternalOggbs, pOggbs, sizeof(*pOggbs));

        /* At this point the pOggbs object has been handed over to pInternalOggbs and can be freed. */
        drflac__free_from_callbacks(pOggbs, &allocationCallbacks);
        pOggbs = NULL;

        /* The Ogg bistream needs to be layered on top of the original bitstream. */
        pFlac->bs.onRead = drflac__on_read_ogg;
        pFlac->bs.onSeek = drflac__on_seek_ogg;
        pFlac->bs.pUserData = (void*)pInternalOggbs;
        pFlac->_oggbs = (void*)pInternalOggbs;
    }
#endif

    pFlac->firstFLACFramePosInBytes = firstFramePos;

    /* NOTE: Seektables are not currently compatible with Ogg encapsulation (Ogg has its own accelerated seeking system). I may change this later, so I'm leaving this here for now. */
#ifndef DR_FLAC_NO_OGG
    if (init.container == drflac_container_ogg)
    {
        pFlac->pSeekpoints = NULL;
        pFlac->seekpointCount = 0;
    }
    else
#endif
    {
        /* If we have a seektable we need to load it now, making sure we move back to where we were previously. */
        if (seektablePos != 0) {
            pFlac->seekpointCount = seekpointCount;
            pFlac->pSeekpoints = (drflac_seekpoint*)((drflac_uint8*)pFlac->pDecodedSamples + decodedSamplesAllocationSize);

            DRFLAC_ASSERT(pFlac->bs.onSeek != NULL);
            DRFLAC_ASSERT(pFlac->bs.onRead != NULL);

            /* Seek to the seektable, then just read directly into our seektable buffer. */
            if (pFlac->bs.onSeek(pFlac->bs.pUserData, (int)seektablePos, drflac_seek_origin_start)) {
                drflac_uint32 iSeekpoint;

                for (iSeekpoint = 0; iSeekpoint < seekpointCount; iSeekpoint += 1) {
                    if (pFlac->bs.onRead(pFlac->bs.pUserData, pFlac->pSeekpoints + iSeekpoint, DRFLAC_SEEKPOINT_SIZE_IN_BYTES) == DRFLAC_SEEKPOINT_SIZE_IN_BYTES) {
                        /* Endian swap. */
                        pFlac->pSeekpoints[iSeekpoint].firstPCMFrame   = drflac__be2host_64(pFlac->pSeekpoints[iSeekpoint].firstPCMFrame);
                        pFlac->pSeekpoints[iSeekpoint].flacFrameOffset = drflac__be2host_64(pFlac->pSeekpoints[iSeekpoint].flacFrameOffset);
                        pFlac->pSeekpoints[iSeekpoint].pcmFrameCount   = drflac__be2host_16(pFlac->pSeekpoints[iSeekpoint].pcmFrameCount);
                    } else {
                        /* Failed to read the seektable. Pretend we don't have one. */
                        pFlac->pSeekpoints = NULL;
                        pFlac->seekpointCount = 0;
                        break;
                    }
                }

                /* We need to seek back to where we were. If this fails it's a critical error. */
                if (!pFlac->bs.onSeek(pFlac->bs.pUserData, (int)pFlac->firstFLACFramePosInBytes, drflac_seek_origin_start)) {
                    drflac__free_from_callbacks(pFlac, &allocationCallbacks);
                    return NULL;
                }
            } else {
                /* Failed to seek to the seektable. Ominous sign, but for now we can just pretend we don't have one. */
                pFlac->pSeekpoints = NULL;
                pFlac->seekpointCount = 0;
            }
        }
    }


    /*
    If we get here, but don't have a STREAMINFO block, it means we've opened the stream in relaxed mode and need to decode
    the first frame.
    */
    if (!init.hasStreamInfoBlock) {
        pFlac->currentFLACFrame.header = init.firstFrameHeader;
        for (;;) {
            drflac_result result = drflac__decode_flac_frame(pFlac);
            if (result == DRFLAC_SUCCESS) {
                break;
            } else {
                if (result == DRFLAC_CRC_MISMATCH) {
                    if (!drflac__read_next_flac_frame_header(&pFlac->bs, pFlac->bitsPerSample, &pFlac->currentFLACFrame.header)) {
                        drflac__free_from_callbacks(pFlac, &allocationCallbacks);
                        return NULL;
                    }
                    continue;
                } else {
                    drflac__free_from_callbacks(pFlac, &allocationCallbacks);
                    return NULL;
                }
            }
        }
    }

    return pFlac;
}



#ifndef DR_FLAC_NO_STDIO
#include <stdio.h>
#ifndef DR_FLAC_NO_WCHAR
#include <wchar.h>      /* For wcslen(), wcsrtombs() */
#endif

/* Errno */
/* drflac_result_from_errno() is only used for fopen() and wfopen() so putting it inside DR_WAV_NO_STDIO for now. If something else needs this later we can move it out. */
#include <errno.h>
static drflac_result drflac_result_from_errno(int e)
{
    switch (e)
    {
        case 0: return DRFLAC_SUCCESS;
    #ifdef EPERM
        case EPERM: return DRFLAC_INVALID_OPERATION;
    #endif
    #ifdef ENOENT
        case ENOENT: return DRFLAC_DOES_NOT_EXIST;
    #endif
    #ifdef ESRCH
        case ESRCH: return DRFLAC_DOES_NOT_EXIST;
    #endif
    #ifdef EINTR
        case EINTR: return DRFLAC_INTERRUPT;
    #endif
    #ifdef EIO
        case EIO: return DRFLAC_IO_ERROR;
    #endif
    #ifdef ENXIO
        case ENXIO: return DRFLAC_DOES_NOT_EXIST;
    #endif
    #ifdef E2BIG
        case E2BIG: return DRFLAC_INVALID_ARGS;
    #endif
    #ifdef ENOEXEC
        case ENOEXEC: return DRFLAC_INVALID_FILE;
    #endif
    #ifdef EBADF
        case EBADF: return DRFLAC_INVALID_FILE;
    #endif
    #ifdef ECHILD
        case ECHILD: return DRFLAC_ERROR;
    #endif
    #ifdef EAGAIN
        case EAGAIN: return DRFLAC_UNAVAILABLE;
    #endif
    #ifdef ENOMEM
        case ENOMEM: return DRFLAC_OUT_OF_MEMORY;
    #endif
    #ifdef EACCES
        case EACCES: return DRFLAC_ACCESS_DENIED;
    #endif
    #ifdef EFAULT
        case EFAULT: return DRFLAC_BAD_ADDRESS;
    #endif
    #ifdef ENOTBLK
        case ENOTBLK: return DRFLAC_ERROR;
    #endif
    #ifdef EBUSY
        case EBUSY: return DRFLAC_BUSY;
    #endif
    #ifdef EEXIST
        case EEXIST: return DRFLAC_ALREADY_EXISTS;
    #endif
    #ifdef EXDEV
        case EXDEV: return DRFLAC_ERROR;
    #endif
    #ifdef ENODEV
        case ENODEV: return DRFLAC_DOES_NOT_EXIST;
    #endif
    #ifdef ENOTDIR
        case ENOTDIR: return DRFLAC_NOT_DIRECTORY;
    #endif
    #ifdef EISDIR
        case EISDIR: return DRFLAC_IS_DIRECTORY;
    #endif
    #ifdef EINVAL
        case EINVAL: return DRFLAC_INVALID_ARGS;
    #endif
    #ifdef ENFILE
        case ENFILE: return DRFLAC_TOO_MANY_OPEN_FILES;
    #endif
    #ifdef EMFILE
        case EMFILE: return DRFLAC_TOO_MANY_OPEN_FILES;
    #endif
    #ifdef ENOTTY
        case ENOTTY: return DRFLAC_INVALID_OPERATION;
    #endif
    #ifdef ETXTBSY
        case ETXTBSY: return DRFLAC_BUSY;
    #endif
    #ifdef EFBIG
        case EFBIG: return DRFLAC_TOO_BIG;
    #endif
    #ifdef ENOSPC
        case ENOSPC: return DRFLAC_NO_SPACE;
    #endif
    #ifdef ESPIPE
        case ESPIPE: return DRFLAC_BAD_SEEK;
    #endif
    #ifdef EROFS
        case EROFS: return DRFLAC_ACCESS_DENIED;
    #endif
    #ifdef EMLINK
        case EMLINK: return DRFLAC_TOO_MANY_LINKS;
    #endif
    #ifdef EPIPE
        case EPIPE: return DRFLAC_BAD_PIPE;
    #endif
    #ifdef EDOM
        case EDOM: return DRFLAC_OUT_OF_RANGE;
    #endif
    #ifdef ERANGE
        case ERANGE: return DRFLAC_OUT_OF_RANGE;
    #endif
    #ifdef EDEADLK
        case EDEADLK: return DRFLAC_DEADLOCK;
    #endif
    #ifdef ENAMETOOLONG
        case ENAMETOOLONG: return DRFLAC_PATH_TOO_LONG;
    #endif
    #ifdef ENOLCK
        case ENOLCK: return DRFLAC_ERROR;
    #endif
    #ifdef ENOSYS
        case ENOSYS: return DRFLAC_NOT_IMPLEMENTED;
    #endif
    #ifdef ENOTEMPTY
        case ENOTEMPTY: return DRFLAC_DIRECTORY_NOT_EMPTY;
    #endif
    #ifdef ELOOP
        case ELOOP: return DRFLAC_TOO_MANY_LINKS;
    #endif
    #ifdef ENOMSG
        case ENOMSG: return DRFLAC_NO_MESSAGE;
    #endif
    #ifdef EIDRM
        case EIDRM: return DRFLAC_ERROR;
    #endif
    #ifdef ECHRNG
        case ECHRNG: return DRFLAC_ERROR;
    #endif
    #ifdef EL2NSYNC
        case EL2NSYNC: return DRFLAC_ERROR;
    #endif
    #ifdef EL3HLT
        case EL3HLT: return DRFLAC_ERROR;
    #endif
    #ifdef EL3RST
        case EL3RST: return DRFLAC_ERROR;
    #endif
    #ifdef ELNRNG
        case ELNRNG: return DRFLAC_OUT_OF_RANGE;
    #endif
    #ifdef EUNATCH
        case EUNATCH: return DRFLAC_ERROR;
    #endif
    #ifdef ENOCSI
        case ENOCSI: return DRFLAC_ERROR;
    #endif
    #ifdef EL2HLT
        case EL2HLT: return DRFLAC_ERROR;
    #endif
    #ifdef EBADE
        case EBADE: return DRFLAC_ERROR;
    #endif
    #ifdef EBADR
        case EBADR: return DRFLAC_ERROR;
    #endif
    #ifdef EXFULL
        case EXFULL: return DRFLAC_ERROR;
    #endif
    #ifdef ENOANO
        case ENOANO: return DRFLAC_ERROR;
    #endif
    #ifdef EBADRQC
        case EBADRQC: return DRFLAC_ERROR;
    #endif
    #ifdef EBADSLT
        case EBADSLT: return DRFLAC_ERROR;
    #endif
    #ifdef EBFONT
        case EBFONT: return DRFLAC_INVALID_FILE;
    #endif
    #ifdef ENOSTR
        case ENOSTR: return DRFLAC_ERROR;
    #endif
    #ifdef ENODATA
        case ENODATA: return DRFLAC_NO_DATA_AVAILABLE;
    #endif
    #ifdef ETIME
        case ETIME: return DRFLAC_TIMEOUT;
    #endif
    #ifdef ENOSR
        case ENOSR: return DRFLAC_NO_DATA_AVAILABLE;
    #endif
    #ifdef ENONET
        case ENONET: return DRFLAC_NO_NETWORK;
    #endif
    #ifdef ENOPKG
        case ENOPKG: return DRFLAC_ERROR;
    #endif
    #ifdef EREMOTE
        case EREMOTE: return DRFLAC_ERROR;
    #endif
    #ifdef ENOLINK
        case ENOLINK: return DRFLAC_ERROR;
    #endif
    #ifdef EADV
        case EADV: return DRFLAC_ERROR;
    #endif
    #ifdef ESRMNT
        case ESRMNT: return DRFLAC_ERROR;
    #endif
    #ifdef ECOMM
        case ECOMM: return DRFLAC_ERROR;
    #endif
    #ifdef EPROTO
        case EPROTO: return DRFLAC_ERROR;
    #endif
    #ifdef EMULTIHOP
        case EMULTIHOP: return DRFLAC_ERROR;
    #endif
    #ifdef EDOTDOT
        case EDOTDOT: return DRFLAC_ERROR;
    #endif
    #ifdef EBADMSG
        case EBADMSG: return DRFLAC_BAD_MESSAGE;
    #endif
    #ifdef EOVERFLOW
        case EOVERFLOW: return DRFLAC_TOO_BIG;
    #endif
    #ifdef ENOTUNIQ
        case ENOTUNIQ: return DRFLAC_NOT_UNIQUE;
    #endif
    #ifdef EBADFD
        case EBADFD: return DRFLAC_ERROR;
    #endif
    #ifdef EREMCHG
        case EREMCHG: return DRFLAC_ERROR;
    #endif
    #ifdef ELIBACC
        case ELIBACC: return DRFLAC_ACCESS_DENIED;
    #endif
    #ifdef ELIBBAD
        case ELIBBAD: return DRFLAC_INVALID_FILE;
    #endif
    #ifdef ELIBSCN
        case ELIBSCN: return DRFLAC_INVALID_FILE;
    #endif
    #ifdef ELIBMAX
        case ELIBMAX: return DRFLAC_ERROR;
    #endif
    #ifdef ELIBEXEC
        case ELIBEXEC: return DRFLAC_ERROR;
    #endif
    #ifdef EILSEQ
        case EILSEQ: return DRFLAC_INVALID_DATA;
    #endif
    #ifdef ERESTART
        case ERESTART: return DRFLAC_ERROR;
    #endif
    #ifdef ESTRPIPE
        case ESTRPIPE: return DRFLAC_ERROR;
    #endif
    #ifdef EUSERS
        case EUSERS: return DRFLAC_ERROR;
    #endif
    #ifdef ENOTSOCK
        case ENOTSOCK: return DRFLAC_NOT_SOCKET;
    #endif
    #ifdef EDESTADDRREQ
        case EDESTADDRREQ: return DRFLAC_NO_ADDRESS;
    #endif
    #ifdef EMSGSIZE
        case EMSGSIZE: return DRFLAC_TOO_BIG;
    #endif
    #ifdef EPROTOTYPE
        case EPROTOTYPE: return DRFLAC_BAD_PROTOCOL;
    #endif
    #ifdef ENOPROTOOPT
        case ENOPROTOOPT: return DRFLAC_PROTOCOL_UNAVAILABLE;
    #endif
    #ifdef EPROTONOSUPPORT
        case EPROTONOSUPPORT: return DRFLAC_PROTOCOL_NOT_SUPPORTED;
    #endif
    #ifdef ESOCKTNOSUPPORT
        case ESOCKTNOSUPPORT: return DRFLAC_SOCKET_NOT_SUPPORTED;
    #endif
    #ifdef EOPNOTSUPP
        case EOPNOTSUPP: return DRFLAC_INVALID_OPERATION;
    #endif
    #ifdef EPFNOSUPPORT
        case EPFNOSUPPORT: return DRFLAC_PROTOCOL_FAMILY_NOT_SUPPORTED;
    #endif
    #ifdef EAFNOSUPPORT
        case EAFNOSUPPORT: return DRFLAC_ADDRESS_FAMILY_NOT_SUPPORTED;
    #endif
    #ifdef EADDRINUSE
        case EADDRINUSE: return DRFLAC_ALREADY_IN_USE;
    #endif
    #ifdef EADDRNOTAVAIL
        case EADDRNOTAVAIL: return DRFLAC_ERROR;
    #endif
    #ifdef ENETDOWN
        case ENETDOWN: return DRFLAC_NO_NETWORK;
    #endif
    #ifdef ENETUNREACH
        case ENETUNREACH: return DRFLAC_NO_NETWORK;
    #endif
    #ifdef ENETRESET
        case ENETRESET: return DRFLAC_NO_NETWORK;
    #endif
    #ifdef ECONNABORTED
        case ECONNABORTED: return DRFLAC_NO_NETWORK;
    #endif
    #ifdef ECONNRESET
        case ECONNRESET: return DRFLAC_CONNECTION_RESET;
    #endif
    #ifdef ENOBUFS
        case ENOBUFS: return DRFLAC_NO_SPACE;
    #endif
    #ifdef EISCONN
        case EISCONN: return DRFLAC_ALREADY_CONNECTED;
    #endif
    #ifdef ENOTCONN
        case ENOTCONN: return DRFLAC_NOT_CONNECTED;
    #endif
    #ifdef ESHUTDOWN
        case ESHUTDOWN: return DRFLAC_ERROR;
    #endif
    #ifdef ETOOMANYREFS
        case ETOOMANYREFS: return DRFLAC_ERROR;
    #endif
    #ifdef ETIMEDOUT
        case ETIMEDOUT: return DRFLAC_TIMEOUT;
    #endif
    #ifdef ECONNREFUSED
        case ECONNREFUSED: return DRFLAC_CONNECTION_REFUSED;
    #endif
    #ifdef EHOSTDOWN
        case EHOSTDOWN: return DRFLAC_NO_HOST;
    #endif
    #ifdef EHOSTUNREACH
        case EHOSTUNREACH: return DRFLAC_NO_HOST;
    #endif
    #ifdef EALREADY
        case EALREADY: return DRFLAC_IN_PROGRESS;
    #endif
    #ifdef EINPROGRESS
        case EINPROGRESS: return DRFLAC_IN_PROGRESS;
    #endif
    #ifdef ESTALE
        case ESTALE: return DRFLAC_INVALID_FILE;
    #endif
    #ifdef EUCLEAN
        case EUCLEAN: return DRFLAC_ERROR;
    #endif
    #ifdef ENOTNAM
        case ENOTNAM: return DRFLAC_ERROR;
    #endif
    #ifdef ENAVAIL
        case ENAVAIL: return DRFLAC_ERROR;
    #endif
    #ifdef EISNAM
        case EISNAM: return DRFLAC_ERROR;
    #endif
    #ifdef EREMOTEIO
        case EREMOTEIO: return DRFLAC_IO_ERROR;
    #endif
    #ifdef EDQUOT
        case EDQUOT: return DRFLAC_NO_SPACE;
    #endif
    #ifdef ENOMEDIUM
        case ENOMEDIUM: return DRFLAC_DOES_NOT_EXIST;
    #endif
    #ifdef EMEDIUMTYPE
        case EMEDIUMTYPE: return DRFLAC_ERROR;
    #endif
    #ifdef ECANCELED
        case ECANCELED: return DRFLAC_CANCELLED;
    #endif
    #ifdef ENOKEY
        case ENOKEY: return DRFLAC_ERROR;
    #endif
    #ifdef EKEYEXPIRED
        case EKEYEXPIRED: return DRFLAC_ERROR;
    #endif
    #ifdef EKEYREVOKED
        case EKEYREVOKED: return DRFLAC_ERROR;
    #endif
    #ifdef EKEYREJECTED
        case EKEYREJECTED: return DRFLAC_ERROR;
    #endif
    #ifdef EOWNERDEAD
        case EOWNERDEAD: return DRFLAC_ERROR;
    #endif
    #ifdef ENOTRECOVERABLE
        case ENOTRECOVERABLE: return DRFLAC_ERROR;
    #endif
    #ifdef ERFKILL
        case ERFKILL: return DRFLAC_ERROR;
    #endif
    #ifdef EHWPOISON
        case EHWPOISON: return DRFLAC_ERROR;
    #endif
        default: return DRFLAC_ERROR;
    }
}
/* End Errno */

/* fopen */
static drflac_result drflac_fopen(FILE** ppFile, const char* pFilePath, const char* pOpenMode)
{
#if defined(_MSC_VER) && _MSC_VER >= 1400
    errno_t err;
#endif

    if (ppFile != NULL) {
        *ppFile = NULL;  /* Safety. */
    }

    if (pFilePath == NULL || pOpenMode == NULL || ppFile == NULL) {
        return DRFLAC_INVALID_ARGS;
    }

#if defined(_MSC_VER) && _MSC_VER >= 1400
    err = fopen_s(ppFile, pFilePath, pOpenMode);
    if (err != 0) {
        return drflac_result_from_errno(err);
    }
#else
#if defined(_WIN32) || defined(__APPLE__)
    *ppFile = fopen(pFilePath, pOpenMode);
#else
    #if defined(_FILE_OFFSET_BITS) && _FILE_OFFSET_BITS == 64 && defined(_LARGEFILE64_SOURCE)
        *ppFile = fopen64(pFilePath, pOpenMode);
    #else
        *ppFile = fopen(pFilePath, pOpenMode);
    #endif
#endif
    if (*ppFile == NULL) {
        drflac_result result = drflac_result_from_errno(errno);
        if (result == DRFLAC_SUCCESS) {
            result = DRFLAC_ERROR;   /* Just a safety check to make sure we never ever return success when pFile == NULL. */
        }

        return result;
    }
#endif

    return DRFLAC_SUCCESS;
}

/*
_wfopen() isn't always available in all compilation environments.

    * Windows only.
    * MSVC seems to support it universally as far back as VC6 from what I can tell (haven't checked further back).
    * MinGW-64 (both 32- and 64-bit) seems to support it.
    * MinGW wraps it in !defined(__STRICT_ANSI__).
    * OpenWatcom wraps it in !defined(_NO_EXT_KEYS).

This can be reviewed as compatibility issues arise. The preference is to use _wfopen_s() and _wfopen() as opposed to the wcsrtombs()
fallback, so if you notice your compiler not detecting this properly I'm happy to look at adding support.
*/
#if defined(_WIN32)
    #if defined(_MSC_VER) || defined(__MINGW64__) || (!defined(__STRICT_ANSI__) && !defined(_NO_EXT_KEYS))
        #define DRFLAC_HAS_WFOPEN
    #endif
#endif

#ifndef DR_FLAC_NO_WCHAR
static drflac_result drflac_wfopen(FILE** ppFile, const wchar_t* pFilePath, const wchar_t* pOpenMode, const drflac_allocation_callbacks* pAllocationCallbacks)
{
    if (ppFile != NULL) {
        *ppFile = NULL;  /* Safety. */
    }

    if (pFilePath == NULL || pOpenMode == NULL || ppFile == NULL) {
        return DRFLAC_INVALID_ARGS;
    }

#if defined(DRFLAC_HAS_WFOPEN)
    {
        /* Use _wfopen() on Windows. */
    #if defined(_MSC_VER) && _MSC_VER >= 1400
        errno_t err = _wfopen_s(ppFile, pFilePath, pOpenMode);
        if (err != 0) {
            return drflac_result_from_errno(err);
        }
    #else
        *ppFile = _wfopen(pFilePath, pOpenMode);
        if (*ppFile == NULL) {
            return drflac_result_from_errno(errno);
        }
    #endif
        (void)pAllocationCallbacks;
    }
#else
    /*
    Use fopen() on anything other than Windows. Requires a conversion. This is annoying because
	fopen() is locale specific. The only real way I can think of to do this is with wcsrtombs(). Note
	that wcstombs() is apparently not thread-safe because it uses a static global mbstate_t object for
    maintaining state. I've checked this with -std=c89 and it works, but if somebody get's a compiler
	error I'll look into improving compatibility.
    */

	/*
	Some compilers don't support wchar_t or wcsrtombs() which we're using below. In this case we just
	need to abort with an error. If you encounter a compiler lacking such support, add it to this list
	and submit a bug report and it'll be added to the library upstream.
	*/
	#if defined(__DJGPP__)
	{
		/* Nothing to do here. This will fall through to the error check below. */
	}
	#else
    {
        mbstate_t mbs;
        size_t lenMB;
        const wchar_t* pFilePathTemp = pFilePath;
        char* pFilePathMB = NULL;
        char pOpenModeMB[32] = {0};

        /* Get the length first. */
        DRFLAC_ZERO_OBJECT(&mbs);
        lenMB = wcsrtombs(NULL, &pFilePathTemp, 0, &mbs);
        if (lenMB == (size_t)-1) {
            return drflac_result_from_errno(errno);
        }

        pFilePathMB = (char*)drflac__malloc_from_callbacks(lenMB + 1, pAllocationCallbacks);
        if (pFilePathMB == NULL) {
            return DRFLAC_OUT_OF_MEMORY;
        }

        pFilePathTemp = pFilePath;
        DRFLAC_ZERO_OBJECT(&mbs);
        wcsrtombs(pFilePathMB, &pFilePathTemp, lenMB + 1, &mbs);

        /* The open mode should always consist of ASCII characters so we should be able to do a trivial conversion. */
        {
            size_t i = 0;
            for (;;) {
                if (pOpenMode[i] == 0) {
                    pOpenModeMB[i] = '\0';
                    break;
                }

                pOpenModeMB[i] = (char)pOpenMode[i];
                i += 1;
            }
        }

        *ppFile = fopen(pFilePathMB, pOpenModeMB);

        drflac__free_from_callbacks(pFilePathMB, pAllocationCallbacks);
    }
	#endif

    if (*ppFile == NULL) {
        return DRFLAC_ERROR;
    }
#endif

    return DRFLAC_SUCCESS;
}
#endif
/* End fopen */

static size_t drflac__on_read_stdio(void* pUserData, void* bufferOut, size_t bytesToRead)
{
    return fread(bufferOut, 1, bytesToRead, (FILE*)pUserData);
}

static drflac_bool32 drflac__on_seek_stdio(void* pUserData, int offset, drflac_seek_origin origin)
{
    DRFLAC_ASSERT(offset >= 0);  /* <-- Never seek backwards. */

    return fseek((FILE*)pUserData, offset, (origin == drflac_seek_origin_current) ? SEEK_CUR : SEEK_SET) == 0;
}


DRFLAC_API drflac* drflac_open_file(const char* pFileName, const drflac_allocation_callbacks* pAllocationCallbacks)
{
    drflac* pFlac;
    FILE* pFile;

    if (drflac_fopen(&pFile, pFileName, "rb") != DRFLAC_SUCCESS) {
        return NULL;
    }

    pFlac = drflac_open(drflac__on_read_stdio, drflac__on_seek_stdio, (void*)pFile, pAllocationCallbacks);
    if (pFlac == NULL) {
        fclose(pFile);
        return NULL;
    }

    return pFlac;
}

#ifndef DR_FLAC_NO_WCHAR
DRFLAC_API drflac* drflac_open_file_w(const wchar_t* pFileName, const drflac_allocation_callbacks* pAllocationCallbacks)
{
    drflac* pFlac;
    FILE* pFile;

    if (drflac_wfopen(&pFile, pFileName, L"rb", pAllocationCallbacks) != DRFLAC_SUCCESS) {
        return NULL;
    }

    pFlac = drflac_open(drflac__on_read_stdio, drflac__on_seek_stdio, (void*)pFile, pAllocationCallbacks);
    if (pFlac == NULL) {
        fclose(pFile);
        return NULL;
    }

    return pFlac;
}
#endif

DRFLAC_API drflac* drflac_open_file_with_metadata(const char* pFileName, drflac_meta_proc onMeta, void* pUserData, const drflac_allocation_callbacks* pAllocationCallbacks)
{
    drflac* pFlac;
    FILE* pFile;

    if (drflac_fopen(&pFile, pFileName, "rb") != DRFLAC_SUCCESS) {
        return NULL;
    }

    pFlac = drflac_open_with_metadata_private(drflac__on_read_stdio, drflac__on_seek_stdio, onMeta, drflac_container_unknown, (void*)pFile, pUserData, pAllocationCallbacks);
    if (pFlac == NULL) {
        fclose(pFile);
        return pFlac;
    }

    return pFlac;
}

#ifndef DR_FLAC_NO_WCHAR
DRFLAC_API drflac* drflac_open_file_with_metadata_w(const wchar_t* pFileName, drflac_meta_proc onMeta, void* pUserData, const drflac_allocation_callbacks* pAllocationCallbacks)
{
    drflac* pFlac;
    FILE* pFile;

    if (drflac_wfopen(&pFile, pFileName, L"rb", pAllocationCallbacks) != DRFLAC_SUCCESS) {
        return NULL;
    }

    pFlac = drflac_open_with_metadata_private(drflac__on_read_stdio, drflac__on_seek_stdio, onMeta, drflac_container_unknown, (void*)pFile, pUserData, pAllocationCallbacks);
    if (pFlac == NULL) {
        fclose(pFile);
        return pFlac;
    }

    return pFlac;
}
#endif
#endif  /* DR_FLAC_NO_STDIO */

static size_t drflac__on_read_memory(void* pUserData, void* bufferOut, size_t bytesToRead)
{
    drflac__memory_stream* memoryStream = (drflac__memory_stream*)pUserData;
    size_t bytesRemaining;

    DRFLAC_ASSERT(memoryStream != NULL);
    DRFLAC_ASSERT(memoryStream->dataSize >= memoryStream->currentReadPos);

    bytesRemaining = memoryStream->dataSize - memoryStream->currentReadPos;
    if (bytesToRead > bytesRemaining) {
        bytesToRead = bytesRemaining;
    }

    if (bytesToRead > 0) {
        DRFLAC_COPY_MEMORY(bufferOut, memoryStream->data + memoryStream->currentReadPos, bytesToRead);
        memoryStream->currentReadPos += bytesToRead;
    }

    return bytesToRead;
}

static drflac_bool32 drflac__on_seek_memory(void* pUserData, int offset, drflac_seek_origin origin)
{
    drflac__memory_stream* memoryStream = (drflac__memory_stream*)pUserData;

    DRFLAC_ASSERT(memoryStream != NULL);
    DRFLAC_ASSERT(offset >= 0); /* <-- Never seek backwards. */

    if (offset > (drflac_int64)memoryStream->dataSize) {
        return DRFLAC_FALSE;
    }

    if (origin == drflac_seek_origin_current) {
        if (memoryStream->currentReadPos + offset <= memoryStream->dataSize) {
            memoryStream->currentReadPos += offset;
        } else {
            return DRFLAC_FALSE;  /* Trying to seek too far forward. */
        }
    } else {
        if ((drflac_uint32)offset <= memoryStream->dataSize) {
            memoryStream->currentReadPos = offset;
        } else {
            return DRFLAC_FALSE;  /* Trying to seek too far forward. */
        }
    }

    return DRFLAC_TRUE;
}

DRFLAC_API drflac* drflac_open_memory(const void* pData, size_t dataSize, const drflac_allocation_callbacks* pAllocationCallbacks)
{
    drflac__memory_stream memoryStream;
    drflac* pFlac;

    memoryStream.data = (const drflac_uint8*)pData;
    memoryStream.dataSize = dataSize;
    memoryStream.currentReadPos = 0;
    pFlac = drflac_open(drflac__on_read_memory, drflac__on_seek_memory, &memoryStream, pAllocationCallbacks);
    if (pFlac == NULL) {
        return NULL;
    }

    pFlac->memoryStream = memoryStream;

    /* This is an awful hack... */
#ifndef DR_FLAC_NO_OGG
    if (pFlac->container == drflac_container_ogg)
    {
        drflac_oggbs* oggbs = (drflac_oggbs*)pFlac->_oggbs;
        oggbs->pUserData = &pFlac->memoryStream;
    }
    else
#endif
    {
        pFlac->bs.pUserData = &pFlac->memoryStream;
    }

    return pFlac;
}

DRFLAC_API drflac* drflac_open_memory_with_metadata(const void* pData, size_t dataSize, drflac_meta_proc onMeta, void* pUserData, const drflac_allocation_callbacks* pAllocationCallbacks)
{
    drflac__memory_stream memoryStream;
    drflac* pFlac;

    memoryStream.data = (const drflac_uint8*)pData;
    memoryStream.dataSize = dataSize;
    memoryStream.currentReadPos = 0;
    pFlac = drflac_open_with_metadata_private(drflac__on_read_memory, drflac__on_seek_memory, onMeta, drflac_container_unknown, &memoryStream, pUserData, pAllocationCallbacks);
    if (pFlac == NULL) {
        return NULL;
    }

    pFlac->memoryStream = memoryStream;

    /* This is an awful hack... */
#ifndef DR_FLAC_NO_OGG
    if (pFlac->container == drflac_container_ogg)
    {
        drflac_oggbs* oggbs = (drflac_oggbs*)pFlac->_oggbs;
        oggbs->pUserData = &pFlac->memoryStream;
    }
    else
#endif
    {
        pFlac->bs.pUserData = &pFlac->memoryStream;
    }

    return pFlac;
}



DRFLAC_API drflac* drflac_open(drflac_read_proc onRead, drflac_seek_proc onSeek, void* pUserData, const drflac_allocation_callbacks* pAllocationCallbacks)
{
    return drflac_open_with_metadata_private(onRead, onSeek, NULL, drflac_container_unknown, pUserData, pUserData, pAllocationCallbacks);
}
DRFLAC_API drflac* drflac_open_relaxed(drflac_read_proc onRead, drflac_seek_proc onSeek, drflac_container container, void* pUserData, const drflac_allocation_callbacks* pAllocationCallbacks)
{
    return drflac_open_with_metadata_private(onRead, onSeek, NULL, container, pUserData, pUserData, pAllocationCallbacks);
}

DRFLAC_API drflac* drflac_open_with_metadata(drflac_read_proc onRead, drflac_seek_proc onSeek, drflac_meta_proc onMeta, void* pUserData, const drflac_allocation_callbacks* pAllocationCallbacks)
{
    return drflac_open_with_metadata_private(onRead, onSeek, onMeta, drflac_container_unknown, pUserData, pUserData, pAllocationCallbacks);
}
DRFLAC_API drflac* drflac_open_with_metadata_relaxed(drflac_read_proc onRead, drflac_seek_proc onSeek, drflac_meta_proc onMeta, drflac_container container, void* pUserData, const drflac_allocation_callbacks* pAllocationCallbacks)
{
    return drflac_open_with_metadata_private(onRead, onSeek, onMeta, container, pUserData, pUserData, pAllocationCallbacks);
}

DRFLAC_API void drflac_close(drflac* pFlac)
{
    if (pFlac == NULL) {
        return;
    }

#ifndef DR_FLAC_NO_STDIO
    /*
    If we opened the file with drflac_open_file() we will want to close the file handle. We can know whether or not drflac_open_file()
    was used by looking at the callbacks.
    */
    if (pFlac->bs.onRead == drflac__on_read_stdio) {
        fclose((FILE*)pFlac->bs.pUserData);
    }

#ifndef DR_FLAC_NO_OGG
    /* Need to clean up Ogg streams a bit differently due to the way the bit streaming is chained. */
    if (pFlac->container == drflac_container_ogg) {
        drflac_oggbs* oggbs = (drflac_oggbs*)pFlac->_oggbs;
        DRFLAC_ASSERT(pFlac->bs.onRead == drflac__on_read_ogg);

        if (oggbs->onRead == drflac__on_read_stdio) {
            fclose((FILE*)oggbs->pUserData);
        }
    }
#endif
#endif

    drflac__free_from_callbacks(pFlac, &pFlac->allocationCallbacks);
}


#if 0
static DRFLAC_INLINE void drflac_read_pcm_frames_s32__decode_left_side__reference(drflac* pFlac, drflac_uint64 frameCount, drflac_uint32 unusedBitsPerSample, const drflac_int32* pInputSamples0, const drflac_int32* pInputSamples1, drflac_int32* pOutputSamples)
{
    drflac_uint64 i;
    for (i = 0; i < frameCount; ++i) {
        drflac_uint32 left  = (drflac_uint32)pInputSamples0[i] << (unusedBitsPerSample + pFlac->currentFLACFrame.subframes[0].wastedBitsPerSample);
        drflac_uint32 side  = (drflac_uint32)pInputSamples1[i] << (unusedBitsPerSample + pFlac->currentFLACFrame.subframes[1].wastedBitsPerSample);
        drflac_uint32 right = left - side;

        pOutputSamples[i*2+0] = (drflac_int32)left;
        pOutputSamples[i*2+1] = (drflac_int32)right;
    }
}
#endif

static DRFLAC_INLINE void drflac_read_pcm_frames_s32__decode_left_side__scalar(drflac* pFlac, drflac_uint64 frameCount, drflac_uint32 unusedBitsPerSample, const drflac_int32* pInputSamples0, const drflac_int32* pInputSamples1, drflac_int32* pOutputSamples)
{
    drflac_uint64 i;
    drflac_uint64 frameCount4 = frameCount >> 2;
    const drflac_uint32* pInputSamples0U32 = (const drflac_uint32*)pInputSamples0;
    const drflac_uint32* pInputSamples1U32 = (const drflac_uint32*)pInputSamples1;
    drflac_uint32 shift0 = unusedBitsPerSample + pFlac->currentFLACFrame.subframes[0].wastedBitsPerSample;
    drflac_uint32 shift1 = unusedBitsPerSample + pFlac->currentFLACFrame.subframes[1].wastedBitsPerSample;

    for (i = 0; i < frameCount4; ++i) {
        drflac_uint32 left0 = pInputSamples0U32[i*4+0] << shift0;
        drflac_uint32 left1 = pInputSamples0U32[i*4+1] << shift0;
        drflac_uint32 left2 = pInputSamples0U32[i*4+2] << shift0;
        drflac_uint32 left3 = pInputSamples0U32[i*4+3] << shift0;

        drflac_uint32 side0 = pInputSamples1U32[i*4+0] << shift1;
        drflac_uint32 side1 = pInputSamples1U32[i*4+1] << shift1;
        drflac_uint32 side2 = pInputSamples1U32[i*4+2] << shift1;
        drflac_uint32 side3 = pInputSamples1U32[i*4+3] << shift1;

        drflac_uint32 right0 = left0 - side0;
        drflac_uint32 right1 = left1 - side1;
        drflac_uint32 right2 = left2 - side2;
        drflac_uint32 right3 = left3 - side3;

        pOutputSamples[i*8+0] = (drflac_int32)left0;
        pOutputSamples[i*8+1] = (drflac_int32)right0;
        pOutputSamples[i*8+2] = (drflac_int32)left1;
        pOutputSamples[i*8+3] = (drflac_int32)right1;
        pOutputSamples[i*8+4] = (drflac_int32)left2;
        pOutputSamples[i*8+5] = (drflac_int32)right2;
        pOutputSamples[i*8+6] = (drflac_int32)left3;
        pOutputSamples[i*8+7] = (drflac_int32)right3;
    }

    for (i = (frameCount4 << 2); i < frameCount; ++i) {
        drflac_uint32 left  = pInputSamples0U32[i] << shift0;
        drflac_uint32 side  = pInputSamples1U32[i] << shift1;
        drflac_uint32 right = left - side;

        pOutputSamples[i*2+0] = (drflac_int32)left;
        pOutputSamples[i*2+1] = (drflac_int32)right;
    }
}

#if defined(DRFLAC_SUPPORT_SSE2)
static DRFLAC_INLINE void drflac_read_pcm_frames_s32__decode_left_side__sse2(drflac* pFlac, drflac_uint64 frameCount, drflac_uint32 unusedBitsPerSample, const drflac_int32* pInputSamples0, const drflac_int32* pInputSamples1, drflac_int32* pOutputSamples)
{
    drflac_uint64 i;
    drflac_uint64 frameCount4 = frameCount >> 2;
    const drflac_uint32* pInputSamples0U32 = (const drflac_uint32*)pInputSamples0;
    const drflac_uint32* pInputSamples1U32 = (const drflac_uint32*)pInputSamples1;
    drflac_uint32 shift0 = unusedBitsPerSample + pFlac->currentFLACFrame.subframes[0].wastedBitsPerSample;
    drflac_uint32 shift1 = unusedBitsPerSample + pFlac->currentFLACFrame.subframes[1].wastedBitsPerSample;

    DRFLAC_ASSERT(pFlac->bitsPerSample <= 24);

    for (i = 0; i < frameCount4; ++i) {
        __m128i left  = _mm_slli_epi32(_mm_loadu_si128((const __m128i*)pInputSamples0 + i), shift0);
        __m128i side  = _mm_slli_epi32(_mm_loadu_si128((const __m128i*)pInputSamples1 + i), shift1);
        __m128i right = _mm_sub_epi32(left, side);

        _mm_storeu_si128((__m128i*)(pOutputSamples + i*8 + 0), _mm_unpacklo_epi32(left, right));
        _mm_storeu_si128((__m128i*)(pOutputSamples + i*8 + 4), _mm_unpackhi_epi32(left, right));
    }

    for (i = (frameCount4 << 2); i < frameCount; ++i) {
        drflac_uint32 left  = pInputSamples0U32[i] << shift0;
        drflac_uint32 side  = pInputSamples1U32[i] << shift1;
        drflac_uint32 right = left - side;

        pOutputSamples[i*2+0] = (drflac_int32)left;
        pOutputSamples[i*2+1] = (drflac_int32)right;
    }
}
#endif

#if defined(DRFLAC_SUPPORT_NEON)
static DRFLAC_INLINE void drflac_read_pcm_frames_s32__decode_left_side__neon(drflac* pFlac, drflac_uint64 frameCount, drflac_uint32 unusedBitsPerSample, const drflac_int32* pInputSamples0, const drflac_int32* pInputSamples1, drflac_int32* pOutputSamples)
{
    drflac_uint64 i;
    drflac_uint64 frameCount4 = frameCount >> 2;
    const drflac_uint32* pInputSamples0U32 = (const drflac_uint32*)pInputSamples0;
    const drflac_uint32* pInputSamples1U32 = (const drflac_uint32*)pInputSamples1;
    drflac_uint32 shift0 = unusedBitsPerSample + pFlac->currentFLACFrame.subframes[0].wastedBitsPerSample;
    drflac_uint32 shift1 = unusedBitsPerSample + pFlac->currentFLACFrame.subframes[1].wastedBitsPerSample;
    int32x4_t shift0_4;
    int32x4_t shift1_4;

    DRFLAC_ASSERT(pFlac->bitsPerSample <= 24);

    shift0_4 = vdupq_n_s32(shift0);
    shift1_4 = vdupq_n_s32(shift1);

    for (i = 0; i < frameCount4; ++i) {
        uint32x4_t left;
        uint32x4_t side;
        uint32x4_t right;

        left  = vshlq_u32(vld1q_u32(pInputSamples0U32 + i*4), shift0_4);
        side  = vshlq_u32(vld1q_u32(pInputSamples1U32 + i*4), shift1_4);
        right = vsubq_u32(left, side);

        drflac__vst2q_u32((drflac_uint32*)pOutputSamples + i*8, vzipq_u32(left, right));
    }

    for (i = (frameCount4 << 2); i < frameCount; ++i) {
        drflac_uint32 left  = pInputSamples0U32[i] << shift0;
        drflac_uint32 side  = pInputSamples1U32[i] << shift1;
        drflac_uint32 right = left - side;

        pOutputSamples[i*2+0] = (drflac_int32)left;
        pOutputSamples[i*2+1] = (drflac_int32)right;
    }
}
#endif

static DRFLAC_INLINE void drflac_read_pcm_frames_s32__decode_left_side(drflac* pFlac, drflac_uint64 frameCount, drflac_uint32 unusedBitsPerSample, const drflac_int32* pInputSamples0, const drflac_int32* pInputSamples1, drflac_int32* pOutputSamples)
{
#if defined(DRFLAC_SUPPORT_SSE2)
    if (drflac__gIsSSE2Supported && pFlac->bitsPerSample <= 24) {
        drflac_read_pcm_frames_s32__decode_left_side__sse2(pFlac, frameCount, unusedBitsPerSample, pInputSamples0, pInputSamples1, pOutputSamples);
    } else
#elif defined(DRFLAC_SUPPORT_NEON)
    if (drflac__gIsNEONSupported && pFlac->bitsPerSample <= 24) {
        drflac_read_pcm_frames_s32__decode_left_side__neon(pFlac, frameCount, unusedBitsPerSample, pInputSamples0, pInputSamples1, pOutputSamples);
    } else
#endif
    {
        /* Scalar fallback. */
#if 0
        drflac_read_pcm_frames_s32__decode_left_side__reference(pFlac, frameCount, unusedBitsPerSample, pInputSamples0, pInputSamples1, pOutputSamples);
#else
        drflac_read_pcm_frames_s32__decode_left_side__scalar(pFlac, frameCount, unusedBitsPerSample, pInputSamples0, pInputSamples1, pOutputSamples);
#endif
    }
}


#if 0
static DRFLAC_INLINE void drflac_read_pcm_frames_s32__decode_right_side__reference(drflac* pFlac, drflac_uint64 frameCount, drflac_uint32 unusedBitsPerSample, const drflac_int32* pInputSamples0, const drflac_int32* pInputSamples1, drflac_int32* pOutputSamples)
{
    drflac_uint64 i;
    for (i = 0; i < frameCount; ++i) {
        drflac_uint32 side  = (drflac_uint32)pInputSamples0[i] << (unusedBitsPerSample + pFlac->currentFLACFrame.subframes[0].wastedBitsPerSample);
        drflac_uint32 right = (drflac_uint32)pInputSamples1[i] << (unusedBitsPerSample + pFlac->currentFLACFrame.subframes[1].wastedBitsPerSample);
        drflac_uint32 left  = right + side;

        pOutputSamples[i*2+0] = (drflac_int32)left;
        pOutputSamples[i*2+1] = (drflac_int32)right;
    }
}
#endif

static DRFLAC_INLINE void drflac_read_pcm_frames_s32__decode_right_side__scalar(drflac* pFlac, drflac_uint64 frameCount, drflac_uint32 unusedBitsPerSample, const drflac_int32* pInputSamples0, const drflac_int32* pInputSamples1, drflac_int32* pOutputSamples)
{
    drflac_uint64 i;
    drflac_uint64 frameCount4 = frameCount >> 2;
    const drflac_uint32* pInputSamples0U32 = (const drflac_uint32*)pInputSamples0;
    const drflac_uint32* pInputSamples1U32 = (const drflac_uint32*)pInputSamples1;
    drflac_uint32 shift0 = unusedBitsPerSample + pFlac->currentFLACFrame.subframes[0].wastedBitsPerSample;
    drflac_uint32 shift1 = unusedBitsPerSample + pFlac->currentFLACFrame.subframes[1].wastedBitsPerSample;

    for (i = 0; i < frameCount4; ++i) {
        drflac_uint32 side0  = pInputSamples0U32[i*4+0] << shift0;
        drflac_uint32 side1  = pInputSamples0U32[i*4+1] << shift0;
        drflac_uint32 side2  = pInputSamples0U32[i*4+2] << shift0;
        drflac_uint32 side3  = pInputSamples0U32[i*4+3] << shift0;

        drflac_uint32 right0 = pInputSamples1U32[i*4+0] << shift1;
        drflac_uint32 right1 = pInputSamples1U32[i*4+1] << shift1;
        drflac_uint32 right2 = pInputSamples1U32[i*4+2] << shift1;
        drflac_uint32 right3 = pInputSamples1U32[i*4+3] << shift1;

        drflac_uint32 left0 = right0 + side0;
        drflac_uint32 left1 = right1 + side1;
        drflac_uint32 left2 = right2 + side2;
        drflac_uint32 left3 = right3 + side3;

        pOutputSamples[i*8+0] = (drflac_int32)left0;
        pOutputSamples[i*8+1] = (drflac_int32)right0;
        pOutputSamples[i*8+2] = (drflac_int32)left1;
        pOutputSamples[i*8+3] = (drflac_int32)right1;
        pOutputSamples[i*8+4] = (drflac_int32)left2;
        pOutputSamples[i*8+5] = (drflac_int32)right2;
        pOutputSamples[i*8+6] = (drflac_int32)left3;
        pOutputSamples[i*8+7] = (drflac_int32)right3;
    }

    for (i = (frameCount4 << 2); i < frameCount; ++i) {
        drflac_uint32 side  = pInputSamples0U32[i] << shift0;
        drflac_uint32 right = pInputSamples1U32[i] << shift1;
        drflac_uint32 left  = right + side;

        pOutputSamples[i*2+0] = (drflac_int32)left;
        pOutputSamples[i*2+1] = (drflac_int32)right;
    }
}

#if defined(DRFLAC_SUPPORT_SSE2)
static DRFLAC_INLINE void drflac_read_pcm_frames_s32__decode_right_side__sse2(drflac* pFlac, drflac_uint64 frameCount, drflac_uint32 unusedBitsPerSample, const drflac_int32* pInputSamples0, const drflac_int32* pInputSamples1, drflac_int32* pOutputSamples)
{
    drflac_uint64 i;
    drflac_uint64 frameCount4 = frameCount >> 2;
    const drflac_uint32* pInputSamples0U32 = (const drflac_uint32*)pInputSamples0;
    const drflac_uint32* pInputSamples1U32 = (const drflac_uint32*)pInputSamples1;
    drflac_uint32 shift0 = unusedBitsPerSample + pFlac->currentFLACFrame.subframes[0].wastedBitsPerSample;
    drflac_uint32 shift1 = unusedBitsPerSample + pFlac->currentFLACFrame.subframes[1].wastedBitsPerSample;

    DRFLAC_ASSERT(pFlac->bitsPerSample <= 24);

    for (i = 0; i < frameCount4; ++i) {
        __m128i side  = _mm_slli_epi32(_mm_loadu_si128((const __m128i*)pInputSamples0 + i), shift0);
        __m128i right = _mm_slli_epi32(_mm_loadu_si128((const __m128i*)pInputSamples1 + i), shift1);
        __m128i left  = _mm_add_epi32(right, side);

        _mm_storeu_si128((__m128i*)(pOutputSamples + i*8 + 0), _mm_unpacklo_epi32(left, right));
        _mm_storeu_si128((__m128i*)(pOutputSamples + i*8 + 4), _mm_unpackhi_epi32(left, right));
    }

    for (i = (frameCount4 << 2); i < frameCount; ++i) {
        drflac_uint32 side  = pInputSamples0U32[i] << shift0;
        drflac_uint32 right = pInputSamples1U32[i] << shift1;
        drflac_uint32 left  = right + side;

        pOutputSamples[i*2+0] = (drflac_int32)left;
        pOutputSamples[i*2+1] = (drflac_int32)right;
    }
}
#endif

#if defined(DRFLAC_SUPPORT_NEON)
static DRFLAC_INLINE void drflac_read_pcm_frames_s32__decode_right_side__neon(drflac* pFlac, drflac_uint64 frameCount, drflac_uint32 unusedBitsPerSample, const drflac_int32* pInputSamples0, const drflac_int32* pInputSamples1, drflac_int32* pOutputSamples)
{
    drflac_uint64 i;
    drflac_uint64 frameCount4 = frameCount >> 2;
    const drflac_uint32* pInputSamples0U32 = (const drflac_uint32*)pInputSamples0;
    const drflac_uint32* pInputSamples1U32 = (const drflac_uint32*)pInputSamples1;
    drflac_uint32 shift0 = unusedBitsPerSample + pFlac->currentFLACFrame.subframes[0].wastedBitsPerSample;
    drflac_uint32 shift1 = unusedBitsPerSample + pFlac->currentFLACFrame.subframes[1].wastedBitsPerSample;
    int32x4_t shift0_4;
    int32x4_t shift1_4;

    DRFLAC_ASSERT(pFlac->bitsPerSample <= 24);

    shift0_4 = vdupq_n_s32(shift0);
    shift1_4 = vdupq_n_s32(shift1);

    for (i = 0; i < frameCount4; ++i) {
        uint32x4_t side;
        uint32x4_t right;
        uint32x4_t left;

        side  = vshlq_u32(vld1q_u32(pInputSamples0U32 + i*4), shift0_4);
        right = vshlq_u32(vld1q_u32(pInputSamples1U32 + i*4), shift1_4);
        left  = vaddq_u32(right, side);

        drflac__vst2q_u32((drflac_uint32*)pOutputSamples + i*8, vzipq_u32(left, right));
    }

    for (i = (frameCount4 << 2); i < frameCount; ++i) {
        drflac_uint32 side  = pInputSamples0U32[i] << shift0;
        drflac_uint32 right = pInputSamples1U32[i] << shift1;
        drflac_uint32 left  = right + side;

        pOutputSamples[i*2+0] = (drflac_int32)left;
        pOutputSamples[i*2+1] = (drflac_int32)right;
    }
}
#endif

static DRFLAC_INLINE void drflac_read_pcm_frames_s32__decode_right_side(drflac* pFlac, drflac_uint64 frameCount, drflac_uint32 unusedBitsPerSample, const drflac_int32* pInputSamples0, const drflac_int32* pInputSamples1, drflac_int32* pOutputSamples)
{
#if defined(DRFLAC_SUPPORT_SSE2)
    if (drflac__gIsSSE2Supported && pFlac->bitsPerSample <= 24) {
        drflac_read_pcm_frames_s32__decode_right_side__sse2(pFlac, frameCount, unusedBitsPerSample, pInputSamples0, pInputSamples1, pOutputSamples);
    } else
#elif defined(DRFLAC_SUPPORT_NEON)
    if (drflac__gIsNEONSupported && pFlac->bitsPerSample <= 24) {
        drflac_read_pcm_frames_s32__decode_right_side__neon(pFlac, frameCount, unusedBitsPerSample, pInputSamples0, pInputSamples1, pOutputSamples);
    } else
#endif
    {
        /* Scalar fallback. */
#if 0
        drflac_read_pcm_frames_s32__decode_right_side__reference(pFlac, frameCount, unusedBitsPerSample, pInputSamples0, pInputSamples1, pOutputSamples);
#else
        drflac_read_pcm_frames_s32__decode_right_side__scalar(pFlac, frameCount, unusedBitsPerSample, pInputSamples0, pInputSamples1, pOutputSamples);
#endif
    }
}


#if 0
static DRFLAC_INLINE void drflac_read_pcm_frames_s32__decode_mid_side__reference(drflac* pFlac, drflac_uint64 frameCount, drflac_uint32 unusedBitsPerSample, const drflac_int32* pInputSamples0, const drflac_int32* pInputSamples1, drflac_int32* pOutputSamples)
{
    for (drflac_uint64 i = 0; i < frameCount; ++i) {
        drflac_uint32 mid  = pInputSamples0U32[i] << pFlac->currentFLACFrame.subframes[0].wastedBitsPerSample;
        drflac_uint32 side = pInputSamples1U32[i] << pFlac->currentFLACFrame.subframes[1].wastedBitsPerSample;

        mid = (mid << 1) | (side & 0x01);

        pOutputSamples[i*2+0] = (drflac_int32)((drflac_uint32)((drflac_int32)(mid + side) >> 1) << unusedBitsPerSample);
        pOutputSamples[i*2+1] = (drflac_int32)((drflac_uint32)((drflac_int32)(mid - side) >> 1) << unusedBitsPerSample);
    }
}
#endif

static DRFLAC_INLINE void drflac_read_pcm_frames_s32__decode_mid_side__scalar(drflac* pFlac, drflac_uint64 frameCount, drflac_uint32 unusedBitsPerSample, const drflac_int32* pInputSamples0, const drflac_int32* pInputSamples1, drflac_int32* pOutputSamples)
{
    drflac_uint64 i;
    drflac_uint64 frameCount4 = frameCount >> 2;
    const drflac_uint32* pInputSamples0U32 = (const drflac_uint32*)pInputSamples0;
    const drflac_uint32* pInputSamples1U32 = (const drflac_uint32*)pInputSamples1;
    drflac_int32 shift = unusedBitsPerSample;

    if (shift > 0) {
        shift -= 1;
        for (i = 0; i < frameCount4; ++i) {
            drflac_uint32 temp0L;
            drflac_uint32 temp1L;
            drflac_uint32 temp2L;
            drflac_uint32 temp3L;
            drflac_uint32 temp0R;
            drflac_uint32 temp1R;
            drflac_uint32 temp2R;
            drflac_uint32 temp3R;

            drflac_uint32 mid0  = pInputSamples0U32[i*4+0] << pFlac->currentFLACFrame.subframes[0].wastedBitsPerSample;
            drflac_uint32 mid1  = pInputSamples0U32[i*4+1] << pFlac->currentFLACFrame.subframes[0].wastedBitsPerSample;
            drflac_uint32 mid2  = pInputSamples0U32[i*4+2] << pFlac->currentFLACFrame.subframes[0].wastedBitsPerSample;
            drflac_uint32 mid3  = pInputSamples0U32[i*4+3] << pFlac->currentFLACFrame.subframes[0].wastedBitsPerSample;

            drflac_uint32 side0 = pInputSamples1U32[i*4+0] << pFlac->currentFLACFrame.subframes[1].wastedBitsPerSample;
            drflac_uint32 side1 = pInputSamples1U32[i*4+1] << pFlac->currentFLACFrame.subframes[1].wastedBitsPerSample;
            drflac_uint32 side2 = pInputSamples1U32[i*4+2] << pFlac->currentFLACFrame.subframes[1].wastedBitsPerSample;
            drflac_uint32 side3 = pInputSamples1U32[i*4+3] << pFlac->currentFLACFrame.subframes[1].wastedBitsPerSample;

            mid0 = (mid0 << 1) | (side0 & 0x01);
            mid1 = (mid1 << 1) | (side1 & 0x01);
            mid2 = (mid2 << 1) | (side2 & 0x01);
            mid3 = (mid3 << 1) | (side3 & 0x01);

            temp0L = (mid0 + side0) << shift;
            temp1L = (mid1 + side1) << shift;
            temp2L = (mid2 + side2) << shift;
            temp3L = (mid3 + side3) << shift;

            temp0R = (mid0 - side0) << shift;
            temp1R = (mid1 - side1) << shift;
            temp2R = (mid2 - side2) << shift;
            temp3R = (mid3 - side3) << shift;

            pOutputSamples[i*8+0] = (drflac_int32)temp0L;
            pOutputSamples[i*8+1] = (drflac_int32)temp0R;
            pOutputSamples[i*8+2] = (drflac_int32)temp1L;
            pOutputSamples[i*8+3] = (drflac_int32)temp1R;
            pOutputSamples[i*8+4] = (drflac_int32)temp2L;
            pOutputSamples[i*8+5] = (drflac_int32)temp2R;
            pOutputSamples[i*8+6] = (drflac_int32)temp3L;
            pOutputSamples[i*8+7] = (drflac_int32)temp3R;
        }
    } else {
        for (i = 0; i < frameCount4; ++i) {
            drflac_uint32 temp0L;
            drflac_uint32 temp1L;
            drflac_uint32 temp2L;
            drflac_uint32 temp3L;
            drflac_uint32 temp0R;
            drflac_uint32 temp1R;
            drflac_uint32 temp2R;
            drflac_uint32 temp3R;

            drflac_uint32 mid0  = pInputSamples0U32[i*4+0] << pFlac->currentFLACFrame.subframes[0].wastedBitsPerSample;
            drflac_uint32 mid1  = pInputSamples0U32[i*4+1] << pFlac->currentFLACFrame.subframes[0].wastedBitsPerSample;
            drflac_uint32 mid2  = pInputSamples0U32[i*4+2] << pFlac->currentFLACFrame.subframes[0].wastedBitsPerSample;
            drflac_uint32 mid3  = pInputSamples0U32[i*4+3] << pFlac->currentFLACFrame.subframes[0].wastedBitsPerSample;

            drflac_uint32 side0 = pInputSamples1U32[i*4+0] << pFlac->currentFLACFrame.subframes[1].wastedBitsPerSample;
            drflac_uint32 side1 = pInputSamples1U32[i*4+1] << pFlac->currentFLACFrame.subframes[1].wastedBitsPerSample;
            drflac_uint32 side2 = pInputSamples1U32[i*4+2] << pFlac->currentFLACFrame.subframes[1].wastedBitsPerSample;
            drflac_uint32 side3 = pInputSamples1U32[i*4+3] << pFlac->currentFLACFrame.subframes[1].wastedBitsPerSample;

            mid0 = (mid0 << 1) | (side0 & 0x01);
            mid1 = (mid1 << 1) | (side1 & 0x01);
            mid2 = (mid2 << 1) | (side2 & 0x01);
            mid3 = (mid3 << 1) | (side3 & 0x01);

            temp0L = (drflac_uint32)((drflac_int32)(mid0 + side0) >> 1);
            temp1L = (drflac_uint32)((drflac_int32)(mid1 + side1) >> 1);
            temp2L = (drflac_uint32)((drflac_int32)(mid2 + side2) >> 1);
            temp3L = (drflac_uint32)((drflac_int32)(mid3 + side3) >> 1);

            temp0R = (drflac_uint32)((drflac_int32)(mid0 - side0) >> 1);
            temp1R = (drflac_uint32)((drflac_int32)(mid1 - side1) >> 1);
            temp2R = (drflac_uint32)((drflac_int32)(mid2 - side2) >> 1);
            temp3R = (drflac_uint32)((drflac_int32)(mid3 - side3) >> 1);

            pOutputSamples[i*8+0] = (drflac_int32)temp0L;
            pOutputSamples[i*8+1] = (drflac_int32)temp0R;
            pOutputSamples[i*8+2] = (drflac_int32)temp1L;
            pOutputSamples[i*8+3] = (drflac_int32)temp1R;
            pOutputSamples[i*8+4] = (drflac_int32)temp2L;
            pOutputSamples[i*8+5] = (drflac_int32)temp2R;
            pOutputSamples[i*8+6] = (drflac_int32)temp3L;
            pOutputSamples[i*8+7] = (drflac_int32)temp3R;
        }
    }

    for (i = (frameCount4 << 2); i < frameCount; ++i) {
        drflac_uint32 mid  = pInputSamples0U32[i] << pFlac->currentFLACFrame.subframes[0].wastedBitsPerSample;
        drflac_uint32 side = pInputSamples1U32[i] << pFlac->currentFLACFrame.subframes[1].wastedBitsPerSample;

        mid = (mid << 1) | (side & 0x01);

        pOutputSamples[i*2+0] = (drflac_int32)((drflac_uint32)((drflac_int32)(mid + side) >> 1) << unusedBitsPerSample);
        pOutputSamples[i*2+1] = (drflac_int32)((drflac_uint32)((drflac_int32)(mid - side) >> 1) << unusedBitsPerSample);
    }
}

#if defined(DRFLAC_SUPPORT_SSE2)
static DRFLAC_INLINE void drflac_read_pcm_frames_s32__decode_mid_side__sse2(drflac* pFlac, drflac_uint64 frameCount, drflac_uint32 unusedBitsPerSample, const drflac_int32* pInputSamples0, const drflac_int32* pInputSamples1, drflac_int32* pOutputSamples)
{
    drflac_uint64 i;
    drflac_uint64 frameCount4 = frameCount >> 2;
    const drflac_uint32* pInputSamples0U32 = (const drflac_uint32*)pInputSamples0;
    const drflac_uint32* pInputSamples1U32 = (const drflac_uint32*)pInputSamples1;
    drflac_int32 shift = unusedBitsPerSample;

    DRFLAC_ASSERT(pFlac->bitsPerSample <= 24);

    if (shift == 0) {
        for (i = 0; i < frameCount4; ++i) {
            __m128i mid;
            __m128i side;
            __m128i left;
            __m128i right;

            mid   = _mm_slli_epi32(_mm_loadu_si128((const __m128i*)pInputSamples0 + i), pFlac->currentFLACFrame.subframes[0].wastedBitsPerSample);
            side  = _mm_slli_epi32(_mm_loadu_si128((const __m128i*)pInputSamples1 + i), pFlac->currentFLACFrame.subframes[1].wastedBitsPerSample);

            mid   = _mm_or_si128(_mm_slli_epi32(mid, 1), _mm_and_si128(side, _mm_set1_epi32(0x01)));

            left  = _mm_srai_epi32(_mm_add_epi32(mid, side), 1);
            right = _mm_srai_epi32(_mm_sub_epi32(mid, side), 1);

            _mm_storeu_si128((__m128i*)(pOutputSamples + i*8 + 0), _mm_unpacklo_epi32(left, right));
            _mm_storeu_si128((__m128i*)(pOutputSamples + i*8 + 4), _mm_unpackhi_epi32(left, right));
        }

        for (i = (frameCount4 << 2); i < frameCount; ++i) {
            drflac_uint32 mid  = pInputSamples0U32[i] << pFlac->currentFLACFrame.subframes[0].wastedBitsPerSample;
            drflac_uint32 side = pInputSamples1U32[i] << pFlac->currentFLACFrame.subframes[1].wastedBitsPerSample;

            mid = (mid << 1) | (side & 0x01);

            pOutputSamples[i*2+0] = (drflac_int32)(mid + side) >> 1;
            pOutputSamples[i*2+1] = (drflac_int32)(mid - side) >> 1;
        }
    } else {
        shift -= 1;
        for (i = 0; i < frameCount4; ++i) {
            __m128i mid;
            __m128i side;
            __m128i left;
            __m128i right;

            mid   = _mm_slli_epi32(_mm_loadu_si128((const __m128i*)pInputSamples0 + i), pFlac->currentFLACFrame.subframes[0].wastedBitsPerSample);
            side  = _mm_slli_epi32(_mm_loadu_si128((const __m128i*)pInputSamples1 + i), pFlac->currentFLACFrame.subframes[1].wastedBitsPerSample);

            mid   = _mm_or_si128(_mm_slli_epi32(mid, 1), _mm_and_si128(side, _mm_set1_epi32(0x01)));

            left  = _mm_slli_epi32(_mm_add_epi32(mid, side), shift);
            right = _mm_slli_epi32(_mm_sub_epi32(mid, side), shift);

            _mm_storeu_si128((__m128i*)(pOutputSamples + i*8 + 0), _mm_unpacklo_epi32(left, right));
            _mm_storeu_si128((__m128i*)(pOutputSamples + i*8 + 4), _mm_unpackhi_epi32(left, right));
        }

        for (i = (frameCount4 << 2); i < frameCount; ++i) {
            drflac_uint32 mid  = pInputSamples0U32[i] << pFlac->currentFLACFrame.subframes[0].wastedBitsPerSample;
            drflac_uint32 side = pInputSamples1U32[i] << pFlac->currentFLACFrame.subframes[1].wastedBitsPerSample;

            mid = (mid << 1) | (side & 0x01);

            pOutputSamples[i*2+0] = (drflac_int32)((mid + side) << shift);
            pOutputSamples[i*2+1] = (drflac_int32)((mid - side) << shift);
        }
    }
}
#endif

#if defined(DRFLAC_SUPPORT_NEON)
static DRFLAC_INLINE void drflac_read_pcm_frames_s32__decode_mid_side__neon(drflac* pFlac, drflac_uint64 frameCount, drflac_uint32 unusedBitsPerSample, const drflac_int32* pInputSamples0, const drflac_int32* pInputSamples1, drflac_int32* pOutputSamples)
{
    drflac_uint64 i;
    drflac_uint64 frameCount4 = frameCount >> 2;
    const drflac_uint32* pInputSamples0U32 = (const drflac_uint32*)pInputSamples0;
    const drflac_uint32* pInputSamples1U32 = (const drflac_uint32*)pInputSamples1;
    drflac_int32 shift = unusedBitsPerSample;
    int32x4_t  wbpsShift0_4; /* wbps = Wasted Bits Per Sample */
    int32x4_t  wbpsShift1_4; /* wbps = Wasted Bits Per Sample */
    uint32x4_t one4;

    DRFLAC_ASSERT(pFlac->bitsPerSample <= 24);

    wbpsShift0_4 = vdupq_n_s32(pFlac->currentFLACFrame.subframes[0].wastedBitsPerSample);
    wbpsShift1_4 = vdupq_n_s32(pFlac->currentFLACFrame.subframes[1].wastedBitsPerSample);
    one4         = vdupq_n_u32(1);

    if (shift == 0) {
        for (i = 0; i < frameCount4; ++i) {
            uint32x4_t mid;
            uint32x4_t side;
            int32x4_t left;
            int32x4_t right;

            mid   = vshlq_u32(vld1q_u32(pInputSamples0U32 + i*4), wbpsShift0_4);
            side  = vshlq_u32(vld1q_u32(pInputSamples1U32 + i*4), wbpsShift1_4);

            mid   = vorrq_u32(vshlq_n_u32(mid, 1), vandq_u32(side, one4));

            left  = vshrq_n_s32(vreinterpretq_s32_u32(vaddq_u32(mid, side)), 1);
            right = vshrq_n_s32(vreinterpretq_s32_u32(vsubq_u32(mid, side)), 1);

            drflac__vst2q_s32(pOutputSamples + i*8, vzipq_s32(left, right));
        }

        for (i = (frameCount4 << 2); i < frameCount; ++i) {
            drflac_uint32 mid  = pInputSamples0U32[i] << pFlac->currentFLACFrame.subframes[0].wastedBitsPerSample;
            drflac_uint32 side = pInputSamples1U32[i] << pFlac->currentFLACFrame.subframes[1].wastedBitsPerSample;

            mid = (mid << 1) | (side & 0x01);

            pOutputSamples[i*2+0] = (drflac_int32)(mid + side) >> 1;
            pOutputSamples[i*2+1] = (drflac_int32)(mid - side) >> 1;
        }
    } else {
        int32x4_t shift4;

        shift -= 1;
        shift4 = vdupq_n_s32(shift);

        for (i = 0; i < frameCount4; ++i) {
            uint32x4_t mid;
            uint32x4_t side;
            int32x4_t left;
            int32x4_t right;

            mid   = vshlq_u32(vld1q_u32(pInputSamples0U32 + i*4), wbpsShift0_4);
            side  = vshlq_u32(vld1q_u32(pInputSamples1U32 + i*4), wbpsShift1_4);

            mid   = vorrq_u32(vshlq_n_u32(mid, 1), vandq_u32(side, one4));

            left  = vreinterpretq_s32_u32(vshlq_u32(vaddq_u32(mid, side), shift4));
            right = vreinterpretq_s32_u32(vshlq_u32(vsubq_u32(mid, side), shift4));

            drflac__vst2q_s32(pOutputSamples + i*8, vzipq_s32(left, right));
        }

        for (i = (frameCount4 << 2); i < frameCount; ++i) {
            drflac_uint32 mid  = pInputSamples0U32[i] << pFlac->currentFLACFrame.subframes[0].wastedBitsPerSample;
            drflac_uint32 side = pInputSamples1U32[i] << pFlac->currentFLACFrame.subframes[1].wastedBitsPerSample;

            mid = (mid << 1) | (side & 0x01);

            pOutputSamples[i*2+0] = (drflac_int32)((mid + side) << shift);
            pOutputSamples[i*2+1] = (drflac_int32)((mid - side) << shift);
        }
    }
}
#endif

static DRFLAC_INLINE void drflac_read_pcm_frames_s32__decode_mid_side(drflac* pFlac, drflac_uint64 frameCount, drflac_uint32 unusedBitsPerSample, const drflac_int32* pInputSamples0, const drflac_int32* pInputSamples1, drflac_int32* pOutputSamples)
{
#if defined(DRFLAC_SUPPORT_SSE2)
    if (drflac__gIsSSE2Supported && pFlac->bitsPerSample <= 24) {
        drflac_read_pcm_frames_s32__decode_mid_side__sse2(pFlac, frameCount, unusedBitsPerSample, pInputSamples0, pInputSamples1, pOutputSamples);
    } else
#elif defined(DRFLAC_SUPPORT_NEON)
    if (drflac__gIsNEONSupported && pFlac->bitsPerSample <= 24) {
        drflac_read_pcm_frames_s32__decode_mid_side__neon(pFlac, frameCount, unusedBitsPerSample, pInputSamples0, pInputSamples1, pOutputSamples);
    } else
#endif
    {
        /* Scalar fallback. */
#if 0
        drflac_read_pcm_frames_s32__decode_mid_side__reference(pFlac, frameCount, unusedBitsPerSample, pInputSamples0, pInputSamples1, pOutputSamples);
#else
        drflac_read_pcm_frames_s32__decode_mid_side__scalar(pFlac, frameCount, unusedBitsPerSample, pInputSamples0, pInputSamples1, pOutputSamples);
#endif
    }
}


#if 0
static DRFLAC_INLINE void drflac_read_pcm_frames_s32__decode_independent_stereo__reference(drflac* pFlac, drflac_uint64 frameCount, drflac_uint32 unusedBitsPerSample, const drflac_int32* pInputSamples0, const drflac_int32* pInputSamples1, drflac_int32* pOutputSamples)
{
    for (drflac_uint64 i = 0; i < frameCount; ++i) {
        pOutputSamples[i*2+0] = (drflac_int32)((drflac_uint32)pInputSamples0[i] << (unusedBitsPerSample + pFlac->currentFLACFrame.subframes[0].wastedBitsPerSample));
        pOutputSamples[i*2+1] = (drflac_int32)((drflac_uint32)pInputSamples1[i] << (unusedBitsPerSample + pFlac->currentFLACFrame.subframes[1].wastedBitsPerSample));
    }
}
#endif

static DRFLAC_INLINE void drflac_read_pcm_frames_s32__decode_independent_stereo__scalar(drflac* pFlac, drflac_uint64 frameCount, drflac_uint32 unusedBitsPerSample, const drflac_int32* pInputSamples0, const drflac_int32* pInputSamples1, drflac_int32* pOutputSamples)
{
    drflac_uint64 i;
    drflac_uint64 frameCount4 = frameCount >> 2;
    const drflac_uint32* pInputSamples0U32 = (const drflac_uint32*)pInputSamples0;
    const drflac_uint32* pInputSamples1U32 = (const drflac_uint32*)pInputSamples1;
    drflac_uint32 shift0 = unusedBitsPerSample + pFlac->currentFLACFrame.subframes[0].wastedBitsPerSample;
    drflac_uint32 shift1 = unusedBitsPerSample + pFlac->currentFLACFrame.subframes[1].wastedBitsPerSample;

    for (i = 0; i < frameCount4; ++i) {
        drflac_uint32 tempL0 = pInputSamples0U32[i*4+0] << shift0;
        drflac_uint32 tempL1 = pInputSamples0U32[i*4+1] << shift0;
        drflac_uint32 tempL2 = pInputSamples0U32[i*4+2] << shift0;
        drflac_uint32 tempL3 = pInputSamples0U32[i*4+3] << shift0;

        drflac_uint32 tempR0 = pInputSamples1U32[i*4+0] << shift1;
        drflac_uint32 tempR1 = pInputSamples1U32[i*4+1] << shift1;
        drflac_uint32 tempR2 = pInputSamples1U32[i*4+2] << shift1;
        drflac_uint32 tempR3 = pInputSamples1U32[i*4+3] << shift1;

        pOutputSamples[i*8+0] = (drflac_int32)tempL0;
        pOutputSamples[i*8+1] = (drflac_int32)tempR0;
        pOutputSamples[i*8+2] = (drflac_int32)tempL1;
        pOutputSamples[i*8+3] = (drflac_int32)tempR1;
        pOutputSamples[i*8+4] = (drflac_int32)tempL2;
        pOutputSamples[i*8+5] = (drflac_int32)tempR2;
        pOutputSamples[i*8+6] = (drflac_int32)tempL3;
        pOutputSamples[i*8+7] = (drflac_int32)tempR3;
    }

    for (i = (frameCount4 << 2); i < frameCount; ++i) {
        pOutputSamples[i*2+0] = (drflac_int32)(pInputSamples0U32[i] << shift0);
        pOutputSamples[i*2+1] = (drflac_int32)(pInputSamples1U32[i] << shift1);
    }
}

#if defined(DRFLAC_SUPPORT_SSE2)
static DRFLAC_INLINE void drflac_read_pcm_frames_s32__decode_independent_stereo__sse2(drflac* pFlac, drflac_uint64 frameCount, drflac_uint32 unusedBitsPerSample, const drflac_int32* pInputSamples0, const drflac_int32* pInputSamples1, drflac_int32* pOutputSamples)
{
    drflac_uint64 i;
    drflac_uint64 frameCount4 = frameCount >> 2;
    const drflac_uint32* pInputSamples0U32 = (const drflac_uint32*)pInputSamples0;
    const drflac_uint32* pInputSamples1U32 = (const drflac_uint32*)pInputSamples1;
    drflac_uint32 shift0 = unusedBitsPerSample + pFlac->currentFLACFrame.subframes[0].wastedBitsPerSample;
    drflac_uint32 shift1 = unusedBitsPerSample + pFlac->currentFLACFrame.subframes[1].wastedBitsPerSample;

    for (i = 0; i < frameCount4; ++i) {
        __m128i left  = _mm_slli_epi32(_mm_loadu_si128((const __m128i*)pInputSamples0 + i), shift0);
        __m128i right = _mm_slli_epi32(_mm_loadu_si128((const __m128i*)pInputSamples1 + i), shift1);

        _mm_storeu_si128((__m128i*)(pOutputSamples + i*8 + 0), _mm_unpacklo_epi32(left, right));
        _mm_storeu_si128((__m128i*)(pOutputSamples + i*8 + 4), _mm_unpackhi_epi32(left, right));
    }

    for (i = (frameCount4 << 2); i < frameCount; ++i) {
        pOutputSamples[i*2+0] = (drflac_int32)(pInputSamples0U32[i] << shift0);
        pOutputSamples[i*2+1] = (drflac_int32)(pInputSamples1U32[i] << shift1);
    }
}
#endif

#if defined(DRFLAC_SUPPORT_NEON)
static DRFLAC_INLINE void drflac_read_pcm_frames_s32__decode_independent_stereo__neon(drflac* pFlac, drflac_uint64 frameCount, drflac_uint32 unusedBitsPerSample, const drflac_int32* pInputSamples0, const drflac_int32* pInputSamples1, drflac_int32* pOutputSamples)
{
    drflac_uint64 i;
    drflac_uint64 frameCount4 = frameCount >> 2;
    const drflac_uint32* pInputSamples0U32 = (const drflac_uint32*)pInputSamples0;
    const drflac_uint32* pInputSamples1U32 = (const drflac_uint32*)pInputSamples1;
    drflac_uint32 shift0 = unusedBitsPerSample + pFlac->currentFLACFrame.subframes[0].wastedBitsPerSample;
    drflac_uint32 shift1 = unusedBitsPerSample + pFlac->currentFLACFrame.subframes[1].wastedBitsPerSample;

    int32x4_t shift4_0 = vdupq_n_s32(shift0);
    int32x4_t shift4_1 = vdupq_n_s32(shift1);

    for (i = 0; i < frameCount4; ++i) {
        int32x4_t left;
        int32x4_t right;

        left  = vreinterpretq_s32_u32(vshlq_u32(vld1q_u32(pInputSamples0U32 + i*4), shift4_0));
        right = vreinterpretq_s32_u32(vshlq_u32(vld1q_u32(pInputSamples1U32 + i*4), shift4_1));

        drflac__vst2q_s32(pOutputSamples + i*8, vzipq_s32(left, right));
    }

    for (i = (frameCount4 << 2); i < frameCount; ++i) {
        pOutputSamples[i*2+0] = (drflac_int32)(pInputSamples0U32[i] << shift0);
        pOutputSamples[i*2+1] = (drflac_int32)(pInputSamples1U32[i] << shift1);
    }
}
#endif

static DRFLAC_INLINE void drflac_read_pcm_frames_s32__decode_independent_stereo(drflac* pFlac, drflac_uint64 frameCount, drflac_uint32 unusedBitsPerSample, const drflac_int32* pInputSamples0, const drflac_int32* pInputSamples1, drflac_int32* pOutputSamples)
{
#if defined(DRFLAC_SUPPORT_SSE2)
    if (drflac__gIsSSE2Supported && pFlac->bitsPerSample <= 24) {
        drflac_read_pcm_frames_s32__decode_independent_stereo__sse2(pFlac, frameCount, unusedBitsPerSample, pInputSamples0, pInputSamples1, pOutputSamples);
    } else
#elif defined(DRFLAC_SUPPORT_NEON)
    if (drflac__gIsNEONSupported && pFlac->bitsPerSample <= 24) {
        drflac_read_pcm_frames_s32__decode_independent_stereo__neon(pFlac, frameCount, unusedBitsPerSample, pInputSamples0, pInputSamples1, pOutputSamples);
    } else
#endif
    {
        /* Scalar fallback. */
#if 0
        drflac_read_pcm_frames_s32__decode_independent_stereo__reference(pFlac, frameCount, unusedBitsPerSample, pInputSamples0, pInputSamples1, pOutputSamples);
#else
        drflac_read_pcm_frames_s32__decode_independent_stereo__scalar(pFlac, frameCount, unusedBitsPerSample, pInputSamples0, pInputSamples1, pOutputSamples);
#endif
    }
}


DRFLAC_API drflac_uint64 drflac_read_pcm_frames_s32(drflac* pFlac, drflac_uint64 framesToRead, drflac_int32* pBufferOut)
{
    drflac_uint64 framesRead;
    drflac_uint32 unusedBitsPerSample;

    if (pFlac == NULL || framesToRead == 0) {
        return 0;
    }

    if (pBufferOut == NULL) {
        return drflac__seek_forward_by_pcm_frames(pFlac, framesToRead);
    }

    DRFLAC_ASSERT(pFlac->bitsPerSample <= 32);
    unusedBitsPerSample = 32 - pFlac->bitsPerSample;

    framesRead = 0;
    while (framesToRead > 0) {
        /* If we've run out of samples in this frame, go to the next. */
        if (pFlac->currentFLACFrame.pcmFramesRemaining == 0) {
            if (!drflac__read_and_decode_next_flac_frame(pFlac)) {
                break;  /* Couldn't read the next frame, so just break from the loop and return. */
            }
        } else {
            unsigned int channelCount = drflac__get_channel_count_from_channel_assignment(pFlac->currentFLACFrame.header.channelAssignment);
            drflac_uint64 iFirstPCMFrame = pFlac->currentFLACFrame.header.blockSizeInPCMFrames - pFlac->currentFLACFrame.pcmFramesRemaining;
            drflac_uint64 frameCountThisIteration = framesToRead;

            if (frameCountThisIteration > pFlac->currentFLACFrame.pcmFramesRemaining) {
                frameCountThisIteration = pFlac->currentFLACFrame.pcmFramesRemaining;
            }

            if (channelCount == 2) {
                const drflac_int32* pDecodedSamples0 = pFlac->currentFLACFrame.subframes[0].pSamplesS32 + iFirstPCMFrame;
                const drflac_int32* pDecodedSamples1 = pFlac->currentFLACFrame.subframes[1].pSamplesS32 + iFirstPCMFrame;

                switch (pFlac->currentFLACFrame.header.channelAssignment)
                {
                    case DRFLAC_CHANNEL_ASSIGNMENT_LEFT_SIDE:
                    {
                        drflac_read_pcm_frames_s32__decode_left_side(pFlac, frameCountThisIteration, unusedBitsPerSample, pDecodedSamples0, pDecodedSamples1, pBufferOut);
                    } break;

                    case DRFLAC_CHANNEL_ASSIGNMENT_RIGHT_SIDE:
                    {
                        drflac_read_pcm_frames_s32__decode_right_side(pFlac, frameCountThisIteration, unusedBitsPerSample, pDecodedSamples0, pDecodedSamples1, pBufferOut);
                    } break;

                    case DRFLAC_CHANNEL_ASSIGNMENT_MID_SIDE:
                    {
                        drflac_read_pcm_frames_s32__decode_mid_side(pFlac, frameCountThisIteration, unusedBitsPerSample, pDecodedSamples0, pDecodedSamples1, pBufferOut);
                    } break;

                    case DRFLAC_CHANNEL_ASSIGNMENT_INDEPENDENT:
                    default:
                    {
                        drflac_read_pcm_frames_s32__decode_independent_stereo(pFlac, frameCountThisIteration, unusedBitsPerSample, pDecodedSamples0, pDecodedSamples1, pBufferOut);
                    } break;
                }
            } else {
                /* Generic interleaving. */
                drflac_uint64 i;
                for (i = 0; i < frameCountThisIteration; ++i) {
                    unsigned int j;
                    for (j = 0; j < channelCount; ++j) {
                        pBufferOut[(i*channelCount)+j] = (drflac_int32)((drflac_uint32)(pFlac->currentFLACFrame.subframes[j].pSamplesS32[iFirstPCMFrame + i]) << (unusedBitsPerSample + pFlac->currentFLACFrame.subframes[j].wastedBitsPerSample));
                    }
                }
            }

            framesRead                += frameCountThisIteration;
            pBufferOut                += frameCountThisIteration * channelCount;
            framesToRead              -= frameCountThisIteration;
            pFlac->currentPCMFrame    += frameCountThisIteration;
            pFlac->currentFLACFrame.pcmFramesRemaining -= (drflac_uint32)frameCountThisIteration;
        }
    }

    return framesRead;
}


#if 0
static DRFLAC_INLINE void drflac_read_pcm_frames_s16__decode_left_side__reference(drflac* pFlac, drflac_uint64 frameCount, drflac_uint32 unusedBitsPerSample, const drflac_int32* pInputSamples0, const drflac_int32* pInputSamples1, drflac_int16* pOutputSamples)
{
    drflac_uint64 i;
    for (i = 0; i < frameCount; ++i) {
        drflac_uint32 left  = (drflac_uint32)pInputSamples0[i] << (unusedBitsPerSample + pFlac->currentFLACFrame.subframes[0].wastedBitsPerSample);
        drflac_uint32 side  = (drflac_uint32)pInputSamples1[i] << (unusedBitsPerSample + pFlac->currentFLACFrame.subframes[1].wastedBitsPerSample);
        drflac_uint32 right = left - side;

        left  >>= 16;
        right >>= 16;

        pOutputSamples[i*2+0] = (drflac_int16)left;
        pOutputSamples[i*2+1] = (drflac_int16)right;
    }
}
#endif

static DRFLAC_INLINE void drflac_read_pcm_frames_s16__decode_left_side__scalar(drflac* pFlac, drflac_uint64 frameCount, drflac_uint32 unusedBitsPerSample, const drflac_int32* pInputSamples0, const drflac_int32* pInputSamples1, drflac_int16* pOutputSamples)
{
    drflac_uint64 i;
    drflac_uint64 frameCount4 = frameCount >> 2;
    const drflac_uint32* pInputSamples0U32 = (const drflac_uint32*)pInputSamples0;
    const drflac_uint32* pInputSamples1U32 = (const drflac_uint32*)pInputSamples1;
    drflac_uint32 shift0 = unusedBitsPerSample + pFlac->currentFLACFrame.subframes[0].wastedBitsPerSample;
    drflac_uint32 shift1 = unusedBitsPerSample + pFlac->currentFLACFrame.subframes[1].wastedBitsPerSample;

    for (i = 0; i < frameCount4; ++i) {
        drflac_uint32 left0 = pInputSamples0U32[i*4+0] << shift0;
        drflac_uint32 left1 = pInputSamples0U32[i*4+1] << shift0;
        drflac_uint32 left2 = pInputSamples0U32[i*4+2] << shift0;
        drflac_uint32 left3 = pInputSamples0U32[i*4+3] << shift0;

        drflac_uint32 side0 = pInputSamples1U32[i*4+0] << shift1;
        drflac_uint32 side1 = pInputSamples1U32[i*4+1] << shift1;
        drflac_uint32 side2 = pInputSamples1U32[i*4+2] << shift1;
        drflac_uint32 side3 = pInputSamples1U32[i*4+3] << shift1;

        drflac_uint32 right0 = left0 - side0;
        drflac_uint32 right1 = left1 - side1;
        drflac_uint32 right2 = left2 - side2;
        drflac_uint32 right3 = left3 - side3;

        left0  >>= 16;
        left1  >>= 16;
        left2  >>= 16;
        left3  >>= 16;

        right0 >>= 16;
        right1 >>= 16;
        right2 >>= 16;
        right3 >>= 16;

        pOutputSamples[i*8+0] = (drflac_int16)left0;
        pOutputSamples[i*8+1] = (drflac_int16)right0;
        pOutputSamples[i*8+2] = (drflac_int16)left1;
        pOutputSamples[i*8+3] = (drflac_int16)right1;
        pOutputSamples[i*8+4] = (drflac_int16)left2;
        pOutputSamples[i*8+5] = (drflac_int16)right2;
        pOutputSamples[i*8+6] = (drflac_int16)left3;
        pOutputSamples[i*8+7] = (drflac_int16)right3;
    }

    for (i = (frameCount4 << 2); i < frameCount; ++i) {
        drflac_uint32 left  = pInputSamples0U32[i] << shift0;
        drflac_uint32 side  = pInputSamples1U32[i] << shift1;
        drflac_uint32 right = left - side;

        left  >>= 16;
        right >>= 16;

        pOutputSamples[i*2+0] = (drflac_int16)left;
        pOutputSamples[i*2+1] = (drflac_int16)right;
    }
}

#if defined(DRFLAC_SUPPORT_SSE2)
static DRFLAC_INLINE void drflac_read_pcm_frames_s16__decode_left_side__sse2(drflac* pFlac, drflac_uint64 frameCount, drflac_uint32 unusedBitsPerSample, const drflac_int32* pInputSamples0, const drflac_int32* pInputSamples1, drflac_int16* pOutputSamples)
{
    drflac_uint64 i;
    drflac_uint64 frameCount4 = frameCount >> 2;
    const drflac_uint32* pInputSamples0U32 = (const drflac_uint32*)pInputSamples0;
    const drflac_uint32* pInputSamples1U32 = (const drflac_uint32*)pInputSamples1;
    drflac_uint32 shift0 = unusedBitsPerSample + pFlac->currentFLACFrame.subframes[0].wastedBitsPerSample;
    drflac_uint32 shift1 = unusedBitsPerSample + pFlac->currentFLACFrame.subframes[1].wastedBitsPerSample;

    DRFLAC_ASSERT(pFlac->bitsPerSample <= 24);

    for (i = 0; i < frameCount4; ++i) {
        __m128i left  = _mm_slli_epi32(_mm_loadu_si128((const __m128i*)pInputSamples0 + i), shift0);
        __m128i side  = _mm_slli_epi32(_mm_loadu_si128((const __m128i*)pInputSamples1 + i), shift1);
        __m128i right = _mm_sub_epi32(left, side);

        left  = _mm_srai_epi32(left,  16);
        right = _mm_srai_epi32(right, 16);

        _mm_storeu_si128((__m128i*)(pOutputSamples + i*8), drflac__mm_packs_interleaved_epi32(left, right));
    }

    for (i = (frameCount4 << 2); i < frameCount; ++i) {
        drflac_uint32 left  = pInputSamples0U32[i] << shift0;
        drflac_uint32 side  = pInputSamples1U32[i] << shift1;
        drflac_uint32 right = left - side;

        left  >>= 16;
        right >>= 16;

        pOutputSamples[i*2+0] = (drflac_int16)left;
        pOutputSamples[i*2+1] = (drflac_int16)right;
    }
}
#endif

#if defined(DRFLAC_SUPPORT_NEON)
static DRFLAC_INLINE void drflac_read_pcm_frames_s16__decode_left_side__neon(drflac* pFlac, drflac_uint64 frameCount, drflac_uint32 unusedBitsPerSample, const drflac_int32* pInputSamples0, const drflac_int32* pInputSamples1, drflac_int16* pOutputSamples)
{
    drflac_uint64 i;
    drflac_uint64 frameCount4 = frameCount >> 2;
    const drflac_uint32* pInputSamples0U32 = (const drflac_uint32*)pInputSamples0;
    const drflac_uint32* pInputSamples1U32 = (const drflac_uint32*)pInputSamples1;
    drflac_uint32 shift0 = unusedBitsPerSample + pFlac->currentFLACFrame.subframes[0].wastedBitsPerSample;
    drflac_uint32 shift1 = unusedBitsPerSample + pFlac->currentFLACFrame.subframes[1].wastedBitsPerSample;
    int32x4_t shift0_4;
    int32x4_t shift1_4;

    DRFLAC_ASSERT(pFlac->bitsPerSample <= 24);

    shift0_4 = vdupq_n_s32(shift0);
    shift1_4 = vdupq_n_s32(shift1);

    for (i = 0; i < frameCount4; ++i) {
        uint32x4_t left;
        uint32x4_t side;
        uint32x4_t right;

        left  = vshlq_u32(vld1q_u32(pInputSamples0U32 + i*4), shift0_4);
        side  = vshlq_u32(vld1q_u32(pInputSamples1U32 + i*4), shift1_4);
        right = vsubq_u32(left, side);

        left  = vshrq_n_u32(left,  16);
        right = vshrq_n_u32(right, 16);

        drflac__vst2q_u16((drflac_uint16*)pOutputSamples + i*8, vzip_u16(vmovn_u32(left), vmovn_u32(right)));
    }

    for (i = (frameCount4 << 2); i < frameCount; ++i) {
        drflac_uint32 left  = pInputSamples0U32[i] << shift0;
        drflac_uint32 side  = pInputSamples1U32[i] << shift1;
        drflac_uint32 right = left - side;

        left  >>= 16;
        right >>= 16;

        pOutputSamples[i*2+0] = (drflac_int16)left;
        pOutputSamples[i*2+1] = (drflac_int16)right;
    }
}
#endif

static DRFLAC_INLINE void drflac_read_pcm_frames_s16__decode_left_side(drflac* pFlac, drflac_uint64 frameCount, drflac_uint32 unusedBitsPerSample, const drflac_int32* pInputSamples0, const drflac_int32* pInputSamples1, drflac_int16* pOutputSamples)
{
#if defined(DRFLAC_SUPPORT_SSE2)
    if (drflac__gIsSSE2Supported && pFlac->bitsPerSample <= 24) {
        drflac_read_pcm_frames_s16__decode_left_side__sse2(pFlac, frameCount, unusedBitsPerSample, pInputSamples0, pInputSamples1, pOutputSamples);
    } else
#elif defined(DRFLAC_SUPPORT_NEON)
    if (drflac__gIsNEONSupported && pFlac->bitsPerSample <= 24) {
        drflac_read_pcm_frames_s16__decode_left_side__neon(pFlac, frameCount, unusedBitsPerSample, pInputSamples0, pInputSamples1, pOutputSamples);
    } else
#endif
    {
        /* Scalar fallback. */
#if 0
        drflac_read_pcm_frames_s16__decode_left_side__reference(pFlac, frameCount, unusedBitsPerSample, pInputSamples0, pInputSamples1, pOutputSamples);
#else
        drflac_read_pcm_frames_s16__decode_left_side__scalar(pFlac, frameCount, unusedBitsPerSample, pInputSamples0, pInputSamples1, pOutputSamples);
#endif
    }
}


#if 0
static DRFLAC_INLINE void drflac_read_pcm_frames_s16__decode_right_side__reference(drflac* pFlac, drflac_uint64 frameCount, drflac_uint32 unusedBitsPerSample, const drflac_int32* pInputSamples0, const drflac_int32* pInputSamples1, drflac_int16* pOutputSamples)
{
    drflac_uint64 i;
    for (i = 0; i < frameCount; ++i) {
        drflac_uint32 side  = (drflac_uint32)pInputSamples0[i] << (unusedBitsPerSample + pFlac->currentFLACFrame.subframes[0].wastedBitsPerSample);
        drflac_uint32 right = (drflac_uint32)pInputSamples1[i] << (unusedBitsPerSample + pFlac->currentFLACFrame.subframes[1].wastedBitsPerSample);
        drflac_uint32 left  = right + side;

        left  >>= 16;
        right >>= 16;

        pOutputSamples[i*2+0] = (drflac_int16)left;
        pOutputSamples[i*2+1] = (drflac_int16)right;
    }
}
#endif

static DRFLAC_INLINE void drflac_read_pcm_frames_s16__decode_right_side__scalar(drflac* pFlac, drflac_uint64 frameCount, drflac_uint32 unusedBitsPerSample, const drflac_int32* pInputSamples0, const drflac_int32* pInputSamples1, drflac_int16* pOutputSamples)
{
    drflac_uint64 i;
    drflac_uint64 frameCount4 = frameCount >> 2;
    const drflac_uint32* pInputSamples0U32 = (const drflac_uint32*)pInputSamples0;
    const drflac_uint32* pInputSamples1U32 = (const drflac_uint32*)pInputSamples1;
    drflac_uint32 shift0 = unusedBitsPerSample + pFlac->currentFLACFrame.subframes[0].wastedBitsPerSample;
    drflac_uint32 shift1 = unusedBitsPerSample + pFlac->currentFLACFrame.subframes[1].wastedBitsPerSample;

    for (i = 0; i < frameCount4; ++i) {
        drflac_uint32 side0  = pInputSamples0U32[i*4+0] << shift0;
        drflac_uint32 side1  = pInputSamples0U32[i*4+1] << shift0;
        drflac_uint32 side2  = pInputSamples0U32[i*4+2] << shift0;
        drflac_uint32 side3  = pInputSamples0U32[i*4+3] << shift0;

        drflac_uint32 right0 = pInputSamples1U32[i*4+0] << shift1;
        drflac_uint32 right1 = pInputSamples1U32[i*4+1] << shift1;
        drflac_uint32 right2 = pInputSamples1U32[i*4+2] << shift1;
        drflac_uint32 right3 = pInputSamples1U32[i*4+3] << shift1;

        drflac_uint32 left0 = right0 + side0;
        drflac_uint32 left1 = right1 + side1;
        drflac_uint32 left2 = right2 + side2;
        drflac_uint32 left3 = right3 + side3;

        left0  >>= 16;
        left1  >>= 16;
        left2  >>= 16;
        left3  >>= 16;

        right0 >>= 16;
        right1 >>= 16;
        right2 >>= 16;
        right3 >>= 16;

        pOutputSamples[i*8+0] = (drflac_int16)left0;
        pOutputSamples[i*8+1] = (drflac_int16)right0;
        pOutputSamples[i*8+2] = (drflac_int16)left1;
        pOutputSamples[i*8+3] = (drflac_int16)right1;
        pOutputSamples[i*8+4] = (drflac_int16)left2;
        pOutputSamples[i*8+5] = (drflac_int16)right2;
        pOutputSamples[i*8+6] = (drflac_int16)left3;
        pOutputSamples[i*8+7] = (drflac_int16)right3;
    }

    for (i = (frameCount4 << 2); i < frameCount; ++i) {
        drflac_uint32 side  = pInputSamples0U32[i] << shift0;
        drflac_uint32 right = pInputSamples1U32[i] << shift1;
        drflac_uint32 left  = right + side;

        left  >>= 16;
        right >>= 16;

        pOutputSamples[i*2+0] = (drflac_int16)left;
        pOutputSamples[i*2+1] = (drflac_int16)right;
    }
}

#if defined(DRFLAC_SUPPORT_SSE2)
static DRFLAC_INLINE void drflac_read_pcm_frames_s16__decode_right_side__sse2(drflac* pFlac, drflac_uint64 frameCount, drflac_uint32 unusedBitsPerSample, const drflac_int32* pInputSamples0, const drflac_int32* pInputSamples1, drflac_int16* pOutputSamples)
{
    drflac_uint64 i;
    drflac_uint64 frameCount4 = frameCount >> 2;
    const drflac_uint32* pInputSamples0U32 = (const drflac_uint32*)pInputSamples0;
    const drflac_uint32* pInputSamples1U32 = (const drflac_uint32*)pInputSamples1;
    drflac_uint32 shift0 = unusedBitsPerSample + pFlac->currentFLACFrame.subframes[0].wastedBitsPerSample;
    drflac_uint32 shift1 = unusedBitsPerSample + pFlac->currentFLACFrame.subframes[1].wastedBitsPerSample;

    DRFLAC_ASSERT(pFlac->bitsPerSample <= 24);

    for (i = 0; i < frameCount4; ++i) {
        __m128i side  = _mm_slli_epi32(_mm_loadu_si128((const __m128i*)pInputSamples0 + i), shift0);
        __m128i right = _mm_slli_epi32(_mm_loadu_si128((const __m128i*)pInputSamples1 + i), shift1);
        __m128i left  = _mm_add_epi32(right, side);

        left  = _mm_srai_epi32(left,  16);
        right = _mm_srai_epi32(right, 16);

        _mm_storeu_si128((__m128i*)(pOutputSamples + i*8), drflac__mm_packs_interleaved_epi32(left, right));
    }

    for (i = (frameCount4 << 2); i < frameCount; ++i) {
        drflac_uint32 side  = pInputSamples0U32[i] << shift0;
        drflac_uint32 right = pInputSamples1U32[i] << shift1;
        drflac_uint32 left  = right + side;

        left  >>= 16;
        right >>= 16;

        pOutputSamples[i*2+0] = (drflac_int16)left;
        pOutputSamples[i*2+1] = (drflac_int16)right;
    }
}
#endif

#if defined(DRFLAC_SUPPORT_NEON)
static DRFLAC_INLINE void drflac_read_pcm_frames_s16__decode_right_side__neon(drflac* pFlac, drflac_uint64 frameCount, drflac_uint32 unusedBitsPerSample, const drflac_int32* pInputSamples0, const drflac_int32* pInputSamples1, drflac_int16* pOutputSamples)
{
    drflac_uint64 i;
    drflac_uint64 frameCount4 = frameCount >> 2;
    const drflac_uint32* pInputSamples0U32 = (const drflac_uint32*)pInputSamples0;
    const drflac_uint32* pInputSamples1U32 = (const drflac_uint32*)pInputSamples1;
    drflac_uint32 shift0 = unusedBitsPerSample + pFlac->currentFLACFrame.subframes[0].wastedBitsPerSample;
    drflac_uint32 shift1 = unusedBitsPerSample + pFlac->currentFLACFrame.subframes[1].wastedBitsPerSample;
    int32x4_t shift0_4;
    int32x4_t shift1_4;

    DRFLAC_ASSERT(pFlac->bitsPerSample <= 24);

    shift0_4 = vdupq_n_s32(shift0);
    shift1_4 = vdupq_n_s32(shift1);

    for (i = 0; i < frameCount4; ++i) {
        uint32x4_t side;
        uint32x4_t right;
        uint32x4_t left;

        side  = vshlq_u32(vld1q_u32(pInputSamples0U32 + i*4), shift0_4);
        right = vshlq_u32(vld1q_u32(pInputSamples1U32 + i*4), shift1_4);
        left  = vaddq_u32(right, side);

        left  = vshrq_n_u32(left,  16);
        right = vshrq_n_u32(right, 16);

        drflac__vst2q_u16((drflac_uint16*)pOutputSamples + i*8, vzip_u16(vmovn_u32(left), vmovn_u32(right)));
    }

    for (i = (frameCount4 << 2); i < frameCount; ++i) {
        drflac_uint32 side  = pInputSamples0U32[i] << shift0;
        drflac_uint32 right = pInputSamples1U32[i] << shift1;
        drflac_uint32 left  = right + side;

        left  >>= 16;
        right >>= 16;

        pOutputSamples[i*2+0] = (drflac_int16)left;
        pOutputSamples[i*2+1] = (drflac_int16)right;
    }
}
#endif

static DRFLAC_INLINE void drflac_read_pcm_frames_s16__decode_right_side(drflac* pFlac, drflac_uint64 frameCount, drflac_uint32 unusedBitsPerSample, const drflac_int32* pInputSamples0, const drflac_int32* pInputSamples1, drflac_int16* pOutputSamples)
{
#if defined(DRFLAC_SUPPORT_SSE2)
    if (drflac__gIsSSE2Supported && pFlac->bitsPerSample <= 24) {
        drflac_read_pcm_frames_s16__decode_right_side__sse2(pFlac, frameCount, unusedBitsPerSample, pInputSamples0, pInputSamples1, pOutputSamples);
    } else
#elif defined(DRFLAC_SUPPORT_NEON)
    if (drflac__gIsNEONSupported && pFlac->bitsPerSample <= 24) {
        drflac_read_pcm_frames_s16__decode_right_side__neon(pFlac, frameCount, unusedBitsPerSample, pInputSamples0, pInputSamples1, pOutputSamples);
    } else
#endif
    {
        /* Scalar fallback. */
#if 0
        drflac_read_pcm_frames_s16__decode_right_side__reference(pFlac, frameCount, unusedBitsPerSample, pInputSamples0, pInputSamples1, pOutputSamples);
#else
        drflac_read_pcm_frames_s16__decode_right_side__scalar(pFlac, frameCount, unusedBitsPerSample, pInputSamples0, pInputSamples1, pOutputSamples);
#endif
    }
}


#if 0
static DRFLAC_INLINE void drflac_read_pcm_frames_s16__decode_mid_side__reference(drflac* pFlac, drflac_uint64 frameCount, drflac_uint32 unusedBitsPerSample, const drflac_int32* pInputSamples0, const drflac_int32* pInputSamples1, drflac_int16* pOutputSamples)
{
    for (drflac_uint64 i = 0; i < frameCount; ++i) {
        drflac_uint32 mid  = (drflac_uint32)pInputSamples0[i] << pFlac->currentFLACFrame.subframes[0].wastedBitsPerSample;
        drflac_uint32 side = (drflac_uint32)pInputSamples1[i] << pFlac->currentFLACFrame.subframes[1].wastedBitsPerSample;

        mid = (mid << 1) | (side & 0x01);

        pOutputSamples[i*2+0] = (drflac_int16)(((drflac_uint32)((drflac_int32)(mid + side) >> 1) << unusedBitsPerSample) >> 16);
        pOutputSamples[i*2+1] = (drflac_int16)(((drflac_uint32)((drflac_int32)(mid - side) >> 1) << unusedBitsPerSample) >> 16);
    }
}
#endif

static DRFLAC_INLINE void drflac_read_pcm_frames_s16__decode_mid_side__scalar(drflac* pFlac, drflac_uint64 frameCount, drflac_uint32 unusedBitsPerSample, const drflac_int32* pInputSamples0, const drflac_int32* pInputSamples1, drflac_int16* pOutputSamples)
{
    drflac_uint64 i;
    drflac_uint64 frameCount4 = frameCount >> 2;
    const drflac_uint32* pInputSamples0U32 = (const drflac_uint32*)pInputSamples0;
    const drflac_uint32* pInputSamples1U32 = (const drflac_uint32*)pInputSamples1;
    drflac_uint32 shift = unusedBitsPerSample;

    if (shift > 0) {
        shift -= 1;
        for (i = 0; i < frameCount4; ++i) {
            drflac_uint32 temp0L;
            drflac_uint32 temp1L;
            drflac_uint32 temp2L;
            drflac_uint32 temp3L;
            drflac_uint32 temp0R;
            drflac_uint32 temp1R;
            drflac_uint32 temp2R;
            drflac_uint32 temp3R;

            drflac_uint32 mid0  = pInputSamples0U32[i*4+0] << pFlac->currentFLACFrame.subframes[0].wastedBitsPerSample;
            drflac_uint32 mid1  = pInputSamples0U32[i*4+1] << pFlac->currentFLACFrame.subframes[0].wastedBitsPerSample;
            drflac_uint32 mid2  = pInputSamples0U32[i*4+2] << pFlac->currentFLACFrame.subframes[0].wastedBitsPerSample;
            drflac_uint32 mid3  = pInputSamples0U32[i*4+3] << pFlac->currentFLACFrame.subframes[0].wastedBitsPerSample;

            drflac_uint32 side0 = pInputSamples1U32[i*4+0] << pFlac->currentFLACFrame.subframes[1].wastedBitsPerSample;
            drflac_uint32 side1 = pInputSamples1U32[i*4+1] << pFlac->currentFLACFrame.subframes[1].wastedBitsPerSample;
            drflac_uint32 side2 = pInputSamples1U32[i*4+2] << pFlac->currentFLACFrame.subframes[1].wastedBitsPerSample;
            drflac_uint32 side3 = pInputSamples1U32[i*4+3] << pFlac->currentFLACFrame.subframes[1].wastedBitsPerSample;

            mid0 = (mid0 << 1) | (side0 & 0x01);
            mid1 = (mid1 << 1) | (side1 & 0x01);
            mid2 = (mid2 << 1) | (side2 & 0x01);
            mid3 = (mid3 << 1) | (side3 & 0x01);

            temp0L = (mid0 + side0) << shift;
            temp1L = (mid1 + side1) << shift;
            temp2L = (mid2 + side2) << shift;
            temp3L = (mid3 + side3) << shift;

            temp0R = (mid0 - side0) << shift;
            temp1R = (mid1 - side1) << shift;
            temp2R = (mid2 - side2) << shift;
            temp3R = (mid3 - side3) << shift;

            temp0L >>= 16;
            temp1L >>= 16;
            temp2L >>= 16;
            temp3L >>= 16;

            temp0R >>= 16;
            temp1R >>= 16;
            temp2R >>= 16;
            temp3R >>= 16;

            pOutputSamples[i*8+0] = (drflac_int16)temp0L;
            pOutputSamples[i*8+1] = (drflac_int16)temp0R;
            pOutputSamples[i*8+2] = (drflac_int16)temp1L;
            pOutputSamples[i*8+3] = (drflac_int16)temp1R;
            pOutputSamples[i*8+4] = (drflac_int16)temp2L;
            pOutputSamples[i*8+5] = (drflac_int16)temp2R;
            pOutputSamples[i*8+6] = (drflac_int16)temp3L;
            pOutputSamples[i*8+7] = (drflac_int16)temp3R;
        }
    } else {
        for (i = 0; i < frameCount4; ++i) {
            drflac_uint32 temp0L;
            drflac_uint32 temp1L;
            drflac_uint32 temp2L;
            drflac_uint32 temp3L;
            drflac_uint32 temp0R;
            drflac_uint32 temp1R;
            drflac_uint32 temp2R;
            drflac_uint32 temp3R;

            drflac_uint32 mid0  = pInputSamples0U32[i*4+0] << pFlac->currentFLACFrame.subframes[0].wastedBitsPerSample;
            drflac_uint32 mid1  = pInputSamples0U32[i*4+1] << pFlac->currentFLACFrame.subframes[0].wastedBitsPerSample;
            drflac_uint32 mid2  = pInputSamples0U32[i*4+2] << pFlac->currentFLACFrame.subframes[0].wastedBitsPerSample;
            drflac_uint32 mid3  = pInputSamples0U32[i*4+3] << pFlac->currentFLACFrame.subframes[0].wastedBitsPerSample;

            drflac_uint32 side0 = pInputSamples1U32[i*4+0] << pFlac->currentFLACFrame.subframes[1].wastedBitsPerSample;
            drflac_uint32 side1 = pInputSamples1U32[i*4+1] << pFlac->currentFLACFrame.subframes[1].wastedBitsPerSample;
            drflac_uint32 side2 = pInputSamples1U32[i*4+2] << pFlac->currentFLACFrame.subframes[1].wastedBitsPerSample;
            drflac_uint32 side3 = pInputSamples1U32[i*4+3] << pFlac->currentFLACFrame.subframes[1].wastedBitsPerSample;

            mid0 = (mid0 << 1) | (side0 & 0x01);
            mid1 = (mid1 << 1) | (side1 & 0x01);
            mid2 = (mid2 << 1) | (side2 & 0x01);
            mid3 = (mid3 << 1) | (side3 & 0x01);

            temp0L = ((drflac_int32)(mid0 + side0) >> 1);
            temp1L = ((drflac_int32)(mid1 + side1) >> 1);
            temp2L = ((drflac_int32)(mid2 + side2) >> 1);
            temp3L = ((drflac_int32)(mid3 + side3) >> 1);

            temp0R = ((drflac_int32)(mid0 - side0) >> 1);
            temp1R = ((drflac_int32)(mid1 - side1) >> 1);
            temp2R = ((drflac_int32)(mid2 - side2) >> 1);
            temp3R = ((drflac_int32)(mid3 - side3) >> 1);

            temp0L >>= 16;
            temp1L >>= 16;
            temp2L >>= 16;
            temp3L >>= 16;

            temp0R >>= 16;
            temp1R >>= 16;
            temp2R >>= 16;
            temp3R >>= 16;

            pOutputSamples[i*8+0] = (drflac_int16)temp0L;
            pOutputSamples[i*8+1] = (drflac_int16)temp0R;
            pOutputSamples[i*8+2] = (drflac_int16)temp1L;
            pOutputSamples[i*8+3] = (drflac_int16)temp1R;
            pOutputSamples[i*8+4] = (drflac_int16)temp2L;
            pOutputSamples[i*8+5] = (drflac_int16)temp2R;
            pOutputSamples[i*8+6] = (drflac_int16)temp3L;
            pOutputSamples[i*8+7] = (drflac_int16)temp3R;
        }
    }

    for (i = (frameCount4 << 2); i < frameCount; ++i) {
        drflac_uint32 mid  = pInputSamples0U32[i] << pFlac->currentFLACFrame.subframes[0].wastedBitsPerSample;
        drflac_uint32 side = pInputSamples1U32[i] << pFlac->currentFLACFrame.subframes[1].wastedBitsPerSample;

        mid = (mid << 1) | (side & 0x01);

        pOutputSamples[i*2+0] = (drflac_int16)(((drflac_uint32)((drflac_int32)(mid + side) >> 1) << unusedBitsPerSample) >> 16);
        pOutputSamples[i*2+1] = (drflac_int16)(((drflac_uint32)((drflac_int32)(mid - side) >> 1) << unusedBitsPerSample) >> 16);
    }
}

#if defined(DRFLAC_SUPPORT_SSE2)
static DRFLAC_INLINE void drflac_read_pcm_frames_s16__decode_mid_side__sse2(drflac* pFlac, drflac_uint64 frameCount, drflac_uint32 unusedBitsPerSample, const drflac_int32* pInputSamples0, const drflac_int32* pInputSamples1, drflac_int16* pOutputSamples)
{
    drflac_uint64 i;
    drflac_uint64 frameCount4 = frameCount >> 2;
    const drflac_uint32* pInputSamples0U32 = (const drflac_uint32*)pInputSamples0;
    const drflac_uint32* pInputSamples1U32 = (const drflac_uint32*)pInputSamples1;
    drflac_uint32 shift = unusedBitsPerSample;

    DRFLAC_ASSERT(pFlac->bitsPerSample <= 24);

    if (shift == 0) {
        for (i = 0; i < frameCount4; ++i) {
            __m128i mid;
            __m128i side;
            __m128i left;
            __m128i right;

            mid   = _mm_slli_epi32(_mm_loadu_si128((const __m128i*)pInputSamples0 + i), pFlac->currentFLACFrame.subframes[0].wastedBitsPerSample);
            side  = _mm_slli_epi32(_mm_loadu_si128((const __m128i*)pInputSamples1 + i), pFlac->currentFLACFrame.subframes[1].wastedBitsPerSample);

            mid   = _mm_or_si128(_mm_slli_epi32(mid, 1), _mm_and_si128(side, _mm_set1_epi32(0x01)));

            left  = _mm_srai_epi32(_mm_add_epi32(mid, side), 1);
            right = _mm_srai_epi32(_mm_sub_epi32(mid, side), 1);

            left  = _mm_srai_epi32(left,  16);
            right = _mm_srai_epi32(right, 16);

            _mm_storeu_si128((__m128i*)(pOutputSamples + i*8), drflac__mm_packs_interleaved_epi32(left, right));
        }

        for (i = (frameCount4 << 2); i < frameCount; ++i) {
            drflac_uint32 mid  = pInputSamples0U32[i] << pFlac->currentFLACFrame.subframes[0].wastedBitsPerSample;
            drflac_uint32 side = pInputSamples1U32[i] << pFlac->currentFLACFrame.subframes[1].wastedBitsPerSample;

            mid = (mid << 1) | (side & 0x01);

            pOutputSamples[i*2+0] = (drflac_int16)(((drflac_int32)(mid + side) >> 1) >> 16);
            pOutputSamples[i*2+1] = (drflac_int16)(((drflac_int32)(mid - side) >> 1) >> 16);
        }
    } else {
        shift -= 1;
        for (i = 0; i < frameCount4; ++i) {
            __m128i mid;
            __m128i side;
            __m128i left;
            __m128i right;

            mid   = _mm_slli_epi32(_mm_loadu_si128((const __m128i*)pInputSamples0 + i), pFlac->currentFLACFrame.subframes[0].wastedBitsPerSample);
            side  = _mm_slli_epi32(_mm_loadu_si128((const __m128i*)pInputSamples1 + i), pFlac->currentFLACFrame.subframes[1].wastedBitsPerSample);

            mid   = _mm_or_si128(_mm_slli_epi32(mid, 1), _mm_and_si128(side, _mm_set1_epi32(0x01)));

            left  = _mm_slli_epi32(_mm_add_epi32(mid, side), shift);
            right = _mm_slli_epi32(_mm_sub_epi32(mid, side), shift);

            left  = _mm_srai_epi32(left,  16);
            right = _mm_srai_epi32(right, 16);

            _mm_storeu_si128((__m128i*)(pOutputSamples + i*8), drflac__mm_packs_interleaved_epi32(left, right));
        }

        for (i = (frameCount4 << 2); i < frameCount; ++i) {
            drflac_uint32 mid  = pInputSamples0U32[i] << pFlac->currentFLACFrame.subframes[0].wastedBitsPerSample;
            drflac_uint32 side = pInputSamples1U32[i] << pFlac->currentFLACFrame.subframes[1].wastedBitsPerSample;

            mid = (mid << 1) | (side & 0x01);

            pOutputSamples[i*2+0] = (drflac_int16)(((mid + side) << shift) >> 16);
            pOutputSamples[i*2+1] = (drflac_int16)(((mid - side) << shift) >> 16);
        }
    }
}
#endif

#if defined(DRFLAC_SUPPORT_NEON)
static DRFLAC_INLINE void drflac_read_pcm_frames_s16__decode_mid_side__neon(drflac* pFlac, drflac_uint64 frameCount, drflac_uint32 unusedBitsPerSample, const drflac_int32* pInputSamples0, const drflac_int32* pInputSamples1, drflac_int16* pOutputSamples)
{
    drflac_uint64 i;
    drflac_uint64 frameCount4 = frameCount >> 2;
    const drflac_uint32* pInputSamples0U32 = (const drflac_uint32*)pInputSamples0;
    const drflac_uint32* pInputSamples1U32 = (const drflac_uint32*)pInputSamples1;
    drflac_uint32 shift = unusedBitsPerSample;
    int32x4_t wbpsShift0_4; /* wbps = Wasted Bits Per Sample */
    int32x4_t wbpsShift1_4; /* wbps = Wasted Bits Per Sample */

    DRFLAC_ASSERT(pFlac->bitsPerSample <= 24);

    wbpsShift0_4 = vdupq_n_s32(pFlac->currentFLACFrame.subframes[0].wastedBitsPerSample);
    wbpsShift1_4 = vdupq_n_s32(pFlac->currentFLACFrame.subframes[1].wastedBitsPerSample);

    if (shift == 0) {
        for (i = 0; i < frameCount4; ++i) {
            uint32x4_t mid;
            uint32x4_t side;
            int32x4_t left;
            int32x4_t right;

            mid   = vshlq_u32(vld1q_u32(pInputSamples0U32 + i*4), wbpsShift0_4);
            side  = vshlq_u32(vld1q_u32(pInputSamples1U32 + i*4), wbpsShift1_4);

            mid   = vorrq_u32(vshlq_n_u32(mid, 1), vandq_u32(side, vdupq_n_u32(1)));

            left  = vshrq_n_s32(vreinterpretq_s32_u32(vaddq_u32(mid, side)), 1);
            right = vshrq_n_s32(vreinterpretq_s32_u32(vsubq_u32(mid, side)), 1);

            left  = vshrq_n_s32(left,  16);
            right = vshrq_n_s32(right, 16);

            drflac__vst2q_s16(pOutputSamples + i*8, vzip_s16(vmovn_s32(left), vmovn_s32(right)));
        }

        for (i = (frameCount4 << 2); i < frameCount; ++i) {
            drflac_uint32 mid  = pInputSamples0U32[i] << pFlac->currentFLACFrame.subframes[0].wastedBitsPerSample;
            drflac_uint32 side = pInputSamples1U32[i] << pFlac->currentFLACFrame.subframes[1].wastedBitsPerSample;

            mid = (mid << 1) | (side & 0x01);

            pOutputSamples[i*2+0] = (drflac_int16)(((drflac_int32)(mid + side) >> 1) >> 16);
            pOutputSamples[i*2+1] = (drflac_int16)(((drflac_int32)(mid - side) >> 1) >> 16);
        }
    } else {
        int32x4_t shift4;

        shift -= 1;
        shift4 = vdupq_n_s32(shift);

        for (i = 0; i < frameCount4; ++i) {
            uint32x4_t mid;
            uint32x4_t side;
            int32x4_t left;
            int32x4_t right;

            mid   = vshlq_u32(vld1q_u32(pInputSamples0U32 + i*4), wbpsShift0_4);
            side  = vshlq_u32(vld1q_u32(pInputSamples1U32 + i*4), wbpsShift1_4);

            mid   = vorrq_u32(vshlq_n_u32(mid, 1), vandq_u32(side, vdupq_n_u32(1)));

            left  = vreinterpretq_s32_u32(vshlq_u32(vaddq_u32(mid, side), shift4));
            right = vreinterpretq_s32_u32(vshlq_u32(vsubq_u32(mid, side), shift4));

            left  = vshrq_n_s32(left,  16);
            right = vshrq_n_s32(right, 16);

            drflac__vst2q_s16(pOutputSamples + i*8, vzip_s16(vmovn_s32(left), vmovn_s32(right)));
        }

        for (i = (frameCount4 << 2); i < frameCount; ++i) {
            drflac_uint32 mid  = pInputSamples0U32[i] << pFlac->currentFLACFrame.subframes[0].wastedBitsPerSample;
            drflac_uint32 side = pInputSamples1U32[i] << pFlac->currentFLACFrame.subframes[1].wastedBitsPerSample;

            mid = (mid << 1) | (side & 0x01);

            pOutputSamples[i*2+0] = (drflac_int16)(((mid + side) << shift) >> 16);
            pOutputSamples[i*2+1] = (drflac_int16)(((mid - side) << shift) >> 16);
        }
    }
}
#endif

static DRFLAC_INLINE void drflac_read_pcm_frames_s16__decode_mid_side(drflac* pFlac, drflac_uint64 frameCount, drflac_uint32 unusedBitsPerSample, const drflac_int32* pInputSamples0, const drflac_int32* pInputSamples1, drflac_int16* pOutputSamples)
{
#if defined(DRFLAC_SUPPORT_SSE2)
    if (drflac__gIsSSE2Supported && pFlac->bitsPerSample <= 24) {
        drflac_read_pcm_frames_s16__decode_mid_side__sse2(pFlac, frameCount, unusedBitsPerSample, pInputSamples0, pInputSamples1, pOutputSamples);
    } else
#elif defined(DRFLAC_SUPPORT_NEON)
    if (drflac__gIsNEONSupported && pFlac->bitsPerSample <= 24) {
        drflac_read_pcm_frames_s16__decode_mid_side__neon(pFlac, frameCount, unusedBitsPerSample, pInputSamples0, pInputSamples1, pOutputSamples);
    } else
#endif
    {
        /* Scalar fallback. */
#if 0
        drflac_read_pcm_frames_s16__decode_mid_side__reference(pFlac, frameCount, unusedBitsPerSample, pInputSamples0, pInputSamples1, pOutputSamples);
#else
        drflac_read_pcm_frames_s16__decode_mid_side__scalar(pFlac, frameCount, unusedBitsPerSample, pInputSamples0, pInputSamples1, pOutputSamples);
#endif
    }
}


#if 0
static DRFLAC_INLINE void drflac_read_pcm_frames_s16__decode_independent_stereo__reference(drflac* pFlac, drflac_uint64 frameCount, drflac_uint32 unusedBitsPerSample, const drflac_int32* pInputSamples0, const drflac_int32* pInputSamples1, drflac_int16* pOutputSamples)
{
    for (drflac_uint64 i = 0; i < frameCount; ++i) {
        pOutputSamples[i*2+0] = (drflac_int16)((drflac_int32)((drflac_uint32)pInputSamples0[i] << (unusedBitsPerSample + pFlac->currentFLACFrame.subframes[0].wastedBitsPerSample)) >> 16);
        pOutputSamples[i*2+1] = (drflac_int16)((drflac_int32)((drflac_uint32)pInputSamples1[i] << (unusedBitsPerSample + pFlac->currentFLACFrame.subframes[1].wastedBitsPerSample)) >> 16);
    }
}
#endif

static DRFLAC_INLINE void drflac_read_pcm_frames_s16__decode_independent_stereo__scalar(drflac* pFlac, drflac_uint64 frameCount, drflac_uint32 unusedBitsPerSample, const drflac_int32* pInputSamples0, const drflac_int32* pInputSamples1, drflac_int16* pOutputSamples)
{
    drflac_uint64 i;
    drflac_uint64 frameCount4 = frameCount >> 2;
    const drflac_uint32* pInputSamples0U32 = (const drflac_uint32*)pInputSamples0;
    const drflac_uint32* pInputSamples1U32 = (const drflac_uint32*)pInputSamples1;
    drflac_uint32 shift0 = unusedBitsPerSample + pFlac->currentFLACFrame.subframes[0].wastedBitsPerSample;
    drflac_uint32 shift1 = unusedBitsPerSample + pFlac->currentFLACFrame.subframes[1].wastedBitsPerSample;

    for (i = 0; i < frameCount4; ++i) {
        drflac_uint32 tempL0 = pInputSamples0U32[i*4+0] << shift0;
        drflac_uint32 tempL1 = pInputSamples0U32[i*4+1] << shift0;
        drflac_uint32 tempL2 = pInputSamples0U32[i*4+2] << shift0;
        drflac_uint32 tempL3 = pInputSamples0U32[i*4+3] << shift0;

        drflac_uint32 tempR0 = pInputSamples1U32[i*4+0] << shift1;
        drflac_uint32 tempR1 = pInputSamples1U32[i*4+1] << shift1;
        drflac_uint32 tempR2 = pInputSamples1U32[i*4+2] << shift1;
        drflac_uint32 tempR3 = pInputSamples1U32[i*4+3] << shift1;

        tempL0 >>= 16;
        tempL1 >>= 16;
        tempL2 >>= 16;
        tempL3 >>= 16;

        tempR0 >>= 16;
        tempR1 >>= 16;
        tempR2 >>= 16;
        tempR3 >>= 16;

        pOutputSamples[i*8+0] = (drflac_int16)tempL0;
        pOutputSamples[i*8+1] = (drflac_int16)tempR0;
        pOutputSamples[i*8+2] = (drflac_int16)tempL1;
        pOutputSamples[i*8+3] = (drflac_int16)tempR1;
        pOutputSamples[i*8+4] = (drflac_int16)tempL2;
        pOutputSamples[i*8+5] = (drflac_int16)tempR2;
        pOutputSamples[i*8+6] = (drflac_int16)tempL3;
        pOutputSamples[i*8+7] = (drflac_int16)tempR3;
    }

    for (i = (frameCount4 << 2); i < frameCount; ++i) {
        pOutputSamples[i*2+0] = (drflac_int16)((pInputSamples0U32[i] << shift0) >> 16);
        pOutputSamples[i*2+1] = (drflac_int16)((pInputSamples1U32[i] << shift1) >> 16);
    }
}

#if defined(DRFLAC_SUPPORT_SSE2)
static DRFLAC_INLINE void drflac_read_pcm_frames_s16__decode_independent_stereo__sse2(drflac* pFlac, drflac_uint64 frameCount, drflac_uint32 unusedBitsPerSample, const drflac_int32* pInputSamples0, const drflac_int32* pInputSamples1, drflac_int16* pOutputSamples)
{
    drflac_uint64 i;
    drflac_uint64 frameCount4 = frameCount >> 2;
    const drflac_uint32* pInputSamples0U32 = (const drflac_uint32*)pInputSamples0;
    const drflac_uint32* pInputSamples1U32 = (const drflac_uint32*)pInputSamples1;
    drflac_uint32 shift0 = unusedBitsPerSample + pFlac->currentFLACFrame.subframes[0].wastedBitsPerSample;
    drflac_uint32 shift1 = unusedBitsPerSample + pFlac->currentFLACFrame.subframes[1].wastedBitsPerSample;

    for (i = 0; i < frameCount4; ++i) {
        __m128i left  = _mm_slli_epi32(_mm_loadu_si128((const __m128i*)pInputSamples0 + i), shift0);
        __m128i right = _mm_slli_epi32(_mm_loadu_si128((const __m128i*)pInputSamples1 + i), shift1);

        left  = _mm_srai_epi32(left,  16);
        right = _mm_srai_epi32(right, 16);

        /* At this point we have results. We can now pack and interleave these into a single __m128i object and then store the in the output buffer. */
        _mm_storeu_si128((__m128i*)(pOutputSamples + i*8), drflac__mm_packs_interleaved_epi32(left, right));
    }

    for (i = (frameCount4 << 2); i < frameCount; ++i) {
        pOutputSamples[i*2+0] = (drflac_int16)((pInputSamples0U32[i] << shift0) >> 16);
        pOutputSamples[i*2+1] = (drflac_int16)((pInputSamples1U32[i] << shift1) >> 16);
    }
}
#endif

#if defined(DRFLAC_SUPPORT_NEON)
static DRFLAC_INLINE void drflac_read_pcm_frames_s16__decode_independent_stereo__neon(drflac* pFlac, drflac_uint64 frameCount, drflac_uint32 unusedBitsPerSample, const drflac_int32* pInputSamples0, const drflac_int32* pInputSamples1, drflac_int16* pOutputSamples)
{
    drflac_uint64 i;
    drflac_uint64 frameCount4 = frameCount >> 2;
    const drflac_uint32* pInputSamples0U32 = (const drflac_uint32*)pInputSamples0;
    const drflac_uint32* pInputSamples1U32 = (const drflac_uint32*)pInputSamples1;
    drflac_uint32 shift0 = unusedBitsPerSample + pFlac->currentFLACFrame.subframes[0].wastedBitsPerSample;
    drflac_uint32 shift1 = unusedBitsPerSample + pFlac->currentFLACFrame.subframes[1].wastedBitsPerSample;

    int32x4_t shift0_4 = vdupq_n_s32(shift0);
    int32x4_t shift1_4 = vdupq_n_s32(shift1);

    for (i = 0; i < frameCount4; ++i) {
        int32x4_t left;
        int32x4_t right;

        left  = vreinterpretq_s32_u32(vshlq_u32(vld1q_u32(pInputSamples0U32 + i*4), shift0_4));
        right = vreinterpretq_s32_u32(vshlq_u32(vld1q_u32(pInputSamples1U32 + i*4), shift1_4));

        left  = vshrq_n_s32(left,  16);
        right = vshrq_n_s32(right, 16);

        drflac__vst2q_s16(pOutputSamples + i*8, vzip_s16(vmovn_s32(left), vmovn_s32(right)));
    }

    for (i = (frameCount4 << 2); i < frameCount; ++i) {
        pOutputSamples[i*2+0] = (drflac_int16)((pInputSamples0U32[i] << shift0) >> 16);
        pOutputSamples[i*2+1] = (drflac_int16)((pInputSamples1U32[i] << shift1) >> 16);
    }
}
#endif

static DRFLAC_INLINE void drflac_read_pcm_frames_s16__decode_independent_stereo(drflac* pFlac, drflac_uint64 frameCount, drflac_uint32 unusedBitsPerSample, const drflac_int32* pInputSamples0, const drflac_int32* pInputSamples1, drflac_int16* pOutputSamples)
{
#if defined(DRFLAC_SUPPORT_SSE2)
    if (drflac__gIsSSE2Supported && pFlac->bitsPerSample <= 24) {
        drflac_read_pcm_frames_s16__decode_independent_stereo__sse2(pFlac, frameCount, unusedBitsPerSample, pInputSamples0, pInputSamples1, pOutputSamples);
    } else
#elif defined(DRFLAC_SUPPORT_NEON)
    if (drflac__gIsNEONSupported && pFlac->bitsPerSample <= 24) {
        drflac_read_pcm_frames_s16__decode_independent_stereo__neon(pFlac, frameCount, unusedBitsPerSample, pInputSamples0, pInputSamples1, pOutputSamples);
    } else
#endif
    {
        /* Scalar fallback. */
#if 0
        drflac_read_pcm_frames_s16__decode_independent_stereo__reference(pFlac, frameCount, unusedBitsPerSample, pInputSamples0, pInputSamples1, pOutputSamples);
#else
        drflac_read_pcm_frames_s16__decode_independent_stereo__scalar(pFlac, frameCount, unusedBitsPerSample, pInputSamples0, pInputSamples1, pOutputSamples);
#endif
    }
}

DRFLAC_API drflac_uint64 drflac_read_pcm_frames_s16(drflac* pFlac, drflac_uint64 framesToRead, drflac_int16* pBufferOut)
{
    drflac_uint64 framesRead;
    drflac_uint32 unusedBitsPerSample;

    if (pFlac == NULL || framesToRead == 0) {
        return 0;
    }

    if (pBufferOut == NULL) {
        return drflac__seek_forward_by_pcm_frames(pFlac, framesToRead);
    }

    DRFLAC_ASSERT(pFlac->bitsPerSample <= 32);
    unusedBitsPerSample = 32 - pFlac->bitsPerSample;

    framesRead = 0;
    while (framesToRead > 0) {
        /* If we've run out of samples in this frame, go to the next. */
        if (pFlac->currentFLACFrame.pcmFramesRemaining == 0) {
            if (!drflac__read_and_decode_next_flac_frame(pFlac)) {
                break;  /* Couldn't read the next frame, so just break from the loop and return. */
            }
        } else {
            unsigned int channelCount = drflac__get_channel_count_from_channel_assignment(pFlac->currentFLACFrame.header.channelAssignment);
            drflac_uint64 iFirstPCMFrame = pFlac->currentFLACFrame.header.blockSizeInPCMFrames - pFlac->currentFLACFrame.pcmFramesRemaining;
            drflac_uint64 frameCountThisIteration = framesToRead;

            if (frameCountThisIteration > pFlac->currentFLACFrame.pcmFramesRemaining) {
                frameCountThisIteration = pFlac->currentFLACFrame.pcmFramesRemaining;
            }

            if (channelCount == 2) {
                const drflac_int32* pDecodedSamples0 = pFlac->currentFLACFrame.subframes[0].pSamplesS32 + iFirstPCMFrame;
                const drflac_int32* pDecodedSamples1 = pFlac->currentFLACFrame.subframes[1].pSamplesS32 + iFirstPCMFrame;

                switch (pFlac->currentFLACFrame.header.channelAssignment)
                {
                    case DRFLAC_CHANNEL_ASSIGNMENT_LEFT_SIDE:
                    {
                        drflac_read_pcm_frames_s16__decode_left_side(pFlac, frameCountThisIteration, unusedBitsPerSample, pDecodedSamples0, pDecodedSamples1, pBufferOut);
                    } break;

                    case DRFLAC_CHANNEL_ASSIGNMENT_RIGHT_SIDE:
                    {
                        drflac_read_pcm_frames_s16__decode_right_side(pFlac, frameCountThisIteration, unusedBitsPerSample, pDecodedSamples0, pDecodedSamples1, pBufferOut);
                    } break;

                    case DRFLAC_CHANNEL_ASSIGNMENT_MID_SIDE:
                    {
                        drflac_read_pcm_frames_s16__decode_mid_side(pFlac, frameCountThisIteration, unusedBitsPerSample, pDecodedSamples0, pDecodedSamples1, pBufferOut);
                    } break;

                    case DRFLAC_CHANNEL_ASSIGNMENT_INDEPENDENT:
                    default:
                    {
                        drflac_read_pcm_frames_s16__decode_independent_stereo(pFlac, frameCountThisIteration, unusedBitsPerSample, pDecodedSamples0, pDecodedSamples1, pBufferOut);
                    } break;
                }
            } else {
                /* Generic interleaving. */
                drflac_uint64 i;
                for (i = 0; i < frameCountThisIteration; ++i) {
                    unsigned int j;
                    for (j = 0; j < channelCount; ++j) {
                        drflac_int32 sampleS32 = (drflac_int32)((drflac_uint32)(pFlac->currentFLACFrame.subframes[j].pSamplesS32[iFirstPCMFrame + i]) << (unusedBitsPerSample + pFlac->currentFLACFrame.subframes[j].wastedBitsPerSample));
                        pBufferOut[(i*channelCount)+j] = (drflac_int16)(sampleS32 >> 16);
                    }
                }
            }

            framesRead                += frameCountThisIteration;
            pBufferOut                += frameCountThisIteration * channelCount;
            framesToRead              -= frameCountThisIteration;
            pFlac->currentPCMFrame    += frameCountThisIteration;
            pFlac->currentFLACFrame.pcmFramesRemaining -= (drflac_uint32)frameCountThisIteration;
        }
    }

    return framesRead;
}


#if 0
static DRFLAC_INLINE void drflac_read_pcm_frames_f32__decode_left_side__reference(drflac* pFlac, drflac_uint64 frameCount, drflac_uint32 unusedBitsPerSample, const drflac_int32* pInputSamples0, const drflac_int32* pInputSamples1, float* pOutputSamples)
{
    drflac_uint64 i;
    for (i = 0; i < frameCount; ++i) {
        drflac_uint32 left  = (drflac_uint32)pInputSamples0[i] << (unusedBitsPerSample + pFlac->currentFLACFrame.subframes[0].wastedBitsPerSample);
        drflac_uint32 side  = (drflac_uint32)pInputSamples1[i] << (unusedBitsPerSample + pFlac->currentFLACFrame.subframes[1].wastedBitsPerSample);
        drflac_uint32 right = left - side;

        pOutputSamples[i*2+0] = (float)((drflac_int32)left  / 2147483648.0);
        pOutputSamples[i*2+1] = (float)((drflac_int32)right / 2147483648.0);
    }
}
#endif

static DRFLAC_INLINE void drflac_read_pcm_frames_f32__decode_left_side__scalar(drflac* pFlac, drflac_uint64 frameCount, drflac_uint32 unusedBitsPerSample, const drflac_int32* pInputSamples0, const drflac_int32* pInputSamples1, float* pOutputSamples)
{
    drflac_uint64 i;
    drflac_uint64 frameCount4 = frameCount >> 2;
    const drflac_uint32* pInputSamples0U32 = (const drflac_uint32*)pInputSamples0;
    const drflac_uint32* pInputSamples1U32 = (const drflac_uint32*)pInputSamples1;
    drflac_uint32 shift0 = unusedBitsPerSample + pFlac->currentFLACFrame.subframes[0].wastedBitsPerSample;
    drflac_uint32 shift1 = unusedBitsPerSample + pFlac->currentFLACFrame.subframes[1].wastedBitsPerSample;

    float factor = 1 / 2147483648.0;

    for (i = 0; i < frameCount4; ++i) {
        drflac_uint32 left0 = pInputSamples0U32[i*4+0] << shift0;
        drflac_uint32 left1 = pInputSamples0U32[i*4+1] << shift0;
        drflac_uint32 left2 = pInputSamples0U32[i*4+2] << shift0;
        drflac_uint32 left3 = pInputSamples0U32[i*4+3] << shift0;

        drflac_uint32 side0 = pInputSamples1U32[i*4+0] << shift1;
        drflac_uint32 side1 = pInputSamples1U32[i*4+1] << shift1;
        drflac_uint32 side2 = pInputSamples1U32[i*4+2] << shift1;
        drflac_uint32 side3 = pInputSamples1U32[i*4+3] << shift1;

        drflac_uint32 right0 = left0 - side0;
        drflac_uint32 right1 = left1 - side1;
        drflac_uint32 right2 = left2 - side2;
        drflac_uint32 right3 = left3 - side3;

        pOutputSamples[i*8+0] = (drflac_int32)left0  * factor;
        pOutputSamples[i*8+1] = (drflac_int32)right0 * factor;
        pOutputSamples[i*8+2] = (drflac_int32)left1  * factor;
        pOutputSamples[i*8+3] = (drflac_int32)right1 * factor;
        pOutputSamples[i*8+4] = (drflac_int32)left2  * factor;
        pOutputSamples[i*8+5] = (drflac_int32)right2 * factor;
        pOutputSamples[i*8+6] = (drflac_int32)left3  * factor;
        pOutputSamples[i*8+7] = (drflac_int32)right3 * factor;
    }

    for (i = (frameCount4 << 2); i < frameCount; ++i) {
        drflac_uint32 left  = pInputSamples0U32[i] << shift0;
        drflac_uint32 side  = pInputSamples1U32[i] << shift1;
        drflac_uint32 right = left - side;

        pOutputSamples[i*2+0] = (drflac_int32)left  * factor;
        pOutputSamples[i*2+1] = (drflac_int32)right * factor;
    }
}

#if defined(DRFLAC_SUPPORT_SSE2)
static DRFLAC_INLINE void drflac_read_pcm_frames_f32__decode_left_side__sse2(drflac* pFlac, drflac_uint64 frameCount, drflac_uint32 unusedBitsPerSample, const drflac_int32* pInputSamples0, const drflac_int32* pInputSamples1, float* pOutputSamples)
{
    drflac_uint64 i;
    drflac_uint64 frameCount4 = frameCount >> 2;
    const drflac_uint32* pInputSamples0U32 = (const drflac_uint32*)pInputSamples0;
    const drflac_uint32* pInputSamples1U32 = (const drflac_uint32*)pInputSamples1;
    drflac_uint32 shift0 = (unusedBitsPerSample + pFlac->currentFLACFrame.subframes[0].wastedBitsPerSample) - 8;
    drflac_uint32 shift1 = (unusedBitsPerSample + pFlac->currentFLACFrame.subframes[1].wastedBitsPerSample) - 8;
    __m128 factor;

    DRFLAC_ASSERT(pFlac->bitsPerSample <= 24);

    factor = _mm_set1_ps(1.0f / 8388608.0f);

    for (i = 0; i < frameCount4; ++i) {
        __m128i left  = _mm_slli_epi32(_mm_loadu_si128((const __m128i*)pInputSamples0 + i), shift0);
        __m128i side  = _mm_slli_epi32(_mm_loadu_si128((const __m128i*)pInputSamples1 + i), shift1);
        __m128i right = _mm_sub_epi32(left, side);
        __m128 leftf  = _mm_mul_ps(_mm_cvtepi32_ps(left),  factor);
        __m128 rightf = _mm_mul_ps(_mm_cvtepi32_ps(right), factor);

        _mm_storeu_ps(pOutputSamples + i*8 + 0, _mm_unpacklo_ps(leftf, rightf));
        _mm_storeu_ps(pOutputSamples + i*8 + 4, _mm_unpackhi_ps(leftf, rightf));
    }

    for (i = (frameCount4 << 2); i < frameCount; ++i) {
        drflac_uint32 left  = pInputSamples0U32[i] << shift0;
        drflac_uint32 side  = pInputSamples1U32[i] << shift1;
        drflac_uint32 right = left - side;

        pOutputSamples[i*2+0] = (drflac_int32)left  / 8388608.0f;
        pOutputSamples[i*2+1] = (drflac_int32)right / 8388608.0f;
    }
}
#endif

#if defined(DRFLAC_SUPPORT_NEON)
static DRFLAC_INLINE void drflac_read_pcm_frames_f32__decode_left_side__neon(drflac* pFlac, drflac_uint64 frameCount, drflac_uint32 unusedBitsPerSample, const drflac_int32* pInputSamples0, const drflac_int32* pInputSamples1, float* pOutputSamples)
{
    drflac_uint64 i;
    drflac_uint64 frameCount4 = frameCount >> 2;
    const drflac_uint32* pInputSamples0U32 = (const drflac_uint32*)pInputSamples0;
    const drflac_uint32* pInputSamples1U32 = (const drflac_uint32*)pInputSamples1;
    drflac_uint32 shift0 = (unusedBitsPerSample + pFlac->currentFLACFrame.subframes[0].wastedBitsPerSample) - 8;
    drflac_uint32 shift1 = (unusedBitsPerSample + pFlac->currentFLACFrame.subframes[1].wastedBitsPerSample) - 8;
    float32x4_t factor4;
    int32x4_t shift0_4;
    int32x4_t shift1_4;

    DRFLAC_ASSERT(pFlac->bitsPerSample <= 24);

    factor4  = vdupq_n_f32(1.0f / 8388608.0f);
    shift0_4 = vdupq_n_s32(shift0);
    shift1_4 = vdupq_n_s32(shift1);

    for (i = 0; i < frameCount4; ++i) {
        uint32x4_t left;
        uint32x4_t side;
        uint32x4_t right;
        float32x4_t leftf;
        float32x4_t rightf;

        left   = vshlq_u32(vld1q_u32(pInputSamples0U32 + i*4), shift0_4);
        side   = vshlq_u32(vld1q_u32(pInputSamples1U32 + i*4), shift1_4);
        right  = vsubq_u32(left, side);
        leftf  = vmulq_f32(vcvtq_f32_s32(vreinterpretq_s32_u32(left)),  factor4);
        rightf = vmulq_f32(vcvtq_f32_s32(vreinterpretq_s32_u32(right)), factor4);

        drflac__vst2q_f32(pOutputSamples + i*8, vzipq_f32(leftf, rightf));
    }

    for (i = (frameCount4 << 2); i < frameCount; ++i) {
        drflac_uint32 left  = pInputSamples0U32[i] << shift0;
        drflac_uint32 side  = pInputSamples1U32[i] << shift1;
        drflac_uint32 right = left - side;

        pOutputSamples[i*2+0] = (drflac_int32)left  / 8388608.0f;
        pOutputSamples[i*2+1] = (drflac_int32)right / 8388608.0f;
    }
}
#endif

static DRFLAC_INLINE void drflac_read_pcm_frames_f32__decode_left_side(drflac* pFlac, drflac_uint64 frameCount, drflac_uint32 unusedBitsPerSample, const drflac_int32* pInputSamples0, const drflac_int32* pInputSamples1, float* pOutputSamples)
{
#if defined(DRFLAC_SUPPORT_SSE2)
    if (drflac__gIsSSE2Supported && pFlac->bitsPerSample <= 24) {
        drflac_read_pcm_frames_f32__decode_left_side__sse2(pFlac, frameCount, unusedBitsPerSample, pInputSamples0, pInputSamples1, pOutputSamples);
    } else
#elif defined(DRFLAC_SUPPORT_NEON)
    if (drflac__gIsNEONSupported && pFlac->bitsPerSample <= 24) {
        drflac_read_pcm_frames_f32__decode_left_side__neon(pFlac, frameCount, unusedBitsPerSample, pInputSamples0, pInputSamples1, pOutputSamples);
    } else
#endif
    {
        /* Scalar fallback. */
#if 0
        drflac_read_pcm_frames_f32__decode_left_side__reference(pFlac, frameCount, unusedBitsPerSample, pInputSamples0, pInputSamples1, pOutputSamples);
#else
        drflac_read_pcm_frames_f32__decode_left_side__scalar(pFlac, frameCount, unusedBitsPerSample, pInputSamples0, pInputSamples1, pOutputSamples);
#endif
    }
}


#if 0
static DRFLAC_INLINE void drflac_read_pcm_frames_f32__decode_right_side__reference(drflac* pFlac, drflac_uint64 frameCount, drflac_uint32 unusedBitsPerSample, const drflac_int32* pInputSamples0, const drflac_int32* pInputSamples1, float* pOutputSamples)
{
    drflac_uint64 i;
    for (i = 0; i < frameCount; ++i) {
        drflac_uint32 side  = (drflac_uint32)pInputSamples0[i] << (unusedBitsPerSample + pFlac->currentFLACFrame.subframes[0].wastedBitsPerSample);
        drflac_uint32 right = (drflac_uint32)pInputSamples1[i] << (unusedBitsPerSample + pFlac->currentFLACFrame.subframes[1].wastedBitsPerSample);
        drflac_uint32 left  = right + side;

        pOutputSamples[i*2+0] = (float)((drflac_int32)left  / 2147483648.0);
        pOutputSamples[i*2+1] = (float)((drflac_int32)right / 2147483648.0);
    }
}
#endif

static DRFLAC_INLINE void drflac_read_pcm_frames_f32__decode_right_side__scalar(drflac* pFlac, drflac_uint64 frameCount, drflac_uint32 unusedBitsPerSample, const drflac_int32* pInputSamples0, const drflac_int32* pInputSamples1, float* pOutputSamples)
{
    drflac_uint64 i;
    drflac_uint64 frameCount4 = frameCount >> 2;
    const drflac_uint32* pInputSamples0U32 = (const drflac_uint32*)pInputSamples0;
    const drflac_uint32* pInputSamples1U32 = (const drflac_uint32*)pInputSamples1;
    drflac_uint32 shift0 = unusedBitsPerSample + pFlac->currentFLACFrame.subframes[0].wastedBitsPerSample;
    drflac_uint32 shift1 = unusedBitsPerSample + pFlac->currentFLACFrame.subframes[1].wastedBitsPerSample;
    float factor = 1 / 2147483648.0;

    for (i = 0; i < frameCount4; ++i) {
        drflac_uint32 side0  = pInputSamples0U32[i*4+0] << shift0;
        drflac_uint32 side1  = pInputSamples0U32[i*4+1] << shift0;
        drflac_uint32 side2  = pInputSamples0U32[i*4+2] << shift0;
        drflac_uint32 side3  = pInputSamples0U32[i*4+3] << shift0;

        drflac_uint32 right0 = pInputSamples1U32[i*4+0] << shift1;
        drflac_uint32 right1 = pInputSamples1U32[i*4+1] << shift1;
        drflac_uint32 right2 = pInputSamples1U32[i*4+2] << shift1;
        drflac_uint32 right3 = pInputSamples1U32[i*4+3] << shift1;

        drflac_uint32 left0 = right0 + side0;
        drflac_uint32 left1 = right1 + side1;
        drflac_uint32 left2 = right2 + side2;
        drflac_uint32 left3 = right3 + side3;

        pOutputSamples[i*8+0] = (drflac_int32)left0  * factor;
        pOutputSamples[i*8+1] = (drflac_int32)right0 * factor;
        pOutputSamples[i*8+2] = (drflac_int32)left1  * factor;
        pOutputSamples[i*8+3] = (drflac_int32)right1 * factor;
        pOutputSamples[i*8+4] = (drflac_int32)left2  * factor;
        pOutputSamples[i*8+5] = (drflac_int32)right2 * factor;
        pOutputSamples[i*8+6] = (drflac_int32)left3  * factor;
        pOutputSamples[i*8+7] = (drflac_int32)right3 * factor;
    }

    for (i = (frameCount4 << 2); i < frameCount; ++i) {
        drflac_uint32 side  = pInputSamples0U32[i] << shift0;
        drflac_uint32 right = pInputSamples1U32[i] << shift1;
        drflac_uint32 left  = right + side;

        pOutputSamples[i*2+0] = (drflac_int32)left  * factor;
        pOutputSamples[i*2+1] = (drflac_int32)right * factor;
    }
}

#if defined(DRFLAC_SUPPORT_SSE2)
static DRFLAC_INLINE void drflac_read_pcm_frames_f32__decode_right_side__sse2(drflac* pFlac, drflac_uint64 frameCount, drflac_uint32 unusedBitsPerSample, const drflac_int32* pInputSamples0, const drflac_int32* pInputSamples1, float* pOutputSamples)
{
    drflac_uint64 i;
    drflac_uint64 frameCount4 = frameCount >> 2;
    const drflac_uint32* pInputSamples0U32 = (const drflac_uint32*)pInputSamples0;
    const drflac_uint32* pInputSamples1U32 = (const drflac_uint32*)pInputSamples1;
    drflac_uint32 shift0 = (unusedBitsPerSample + pFlac->currentFLACFrame.subframes[0].wastedBitsPerSample) - 8;
    drflac_uint32 shift1 = (unusedBitsPerSample + pFlac->currentFLACFrame.subframes[1].wastedBitsPerSample) - 8;
    __m128 factor;

    DRFLAC_ASSERT(pFlac->bitsPerSample <= 24);

    factor = _mm_set1_ps(1.0f / 8388608.0f);

    for (i = 0; i < frameCount4; ++i) {
        __m128i side  = _mm_slli_epi32(_mm_loadu_si128((const __m128i*)pInputSamples0 + i), shift0);
        __m128i right = _mm_slli_epi32(_mm_loadu_si128((const __m128i*)pInputSamples1 + i), shift1);
        __m128i left  = _mm_add_epi32(right, side);
        __m128 leftf  = _mm_mul_ps(_mm_cvtepi32_ps(left),  factor);
        __m128 rightf = _mm_mul_ps(_mm_cvtepi32_ps(right), factor);

        _mm_storeu_ps(pOutputSamples + i*8 + 0, _mm_unpacklo_ps(leftf, rightf));
        _mm_storeu_ps(pOutputSamples + i*8 + 4, _mm_unpackhi_ps(leftf, rightf));
    }

    for (i = (frameCount4 << 2); i < frameCount; ++i) {
        drflac_uint32 side  = pInputSamples0U32[i] << shift0;
        drflac_uint32 right = pInputSamples1U32[i] << shift1;
        drflac_uint32 left  = right + side;

        pOutputSamples[i*2+0] = (drflac_int32)left  / 8388608.0f;
        pOutputSamples[i*2+1] = (drflac_int32)right / 8388608.0f;
    }
}
#endif

#if defined(DRFLAC_SUPPORT_NEON)
static DRFLAC_INLINE void drflac_read_pcm_frames_f32__decode_right_side__neon(drflac* pFlac, drflac_uint64 frameCount, drflac_uint32 unusedBitsPerSample, const drflac_int32* pInputSamples0, const drflac_int32* pInputSamples1, float* pOutputSamples)
{
    drflac_uint64 i;
    drflac_uint64 frameCount4 = frameCount >> 2;
    const drflac_uint32* pInputSamples0U32 = (const drflac_uint32*)pInputSamples0;
    const drflac_uint32* pInputSamples1U32 = (const drflac_uint32*)pInputSamples1;
    drflac_uint32 shift0 = (unusedBitsPerSample + pFlac->currentFLACFrame.subframes[0].wastedBitsPerSample) - 8;
    drflac_uint32 shift1 = (unusedBitsPerSample + pFlac->currentFLACFrame.subframes[1].wastedBitsPerSample) - 8;
    float32x4_t factor4;
    int32x4_t shift0_4;
    int32x4_t shift1_4;

    DRFLAC_ASSERT(pFlac->bitsPerSample <= 24);

    factor4  = vdupq_n_f32(1.0f / 8388608.0f);
    shift0_4 = vdupq_n_s32(shift0);
    shift1_4 = vdupq_n_s32(shift1);

    for (i = 0; i < frameCount4; ++i) {
        uint32x4_t side;
        uint32x4_t right;
        uint32x4_t left;
        float32x4_t leftf;
        float32x4_t rightf;

        side   = vshlq_u32(vld1q_u32(pInputSamples0U32 + i*4), shift0_4);
        right  = vshlq_u32(vld1q_u32(pInputSamples1U32 + i*4), shift1_4);
        left   = vaddq_u32(right, side);
        leftf  = vmulq_f32(vcvtq_f32_s32(vreinterpretq_s32_u32(left)),  factor4);
        rightf = vmulq_f32(vcvtq_f32_s32(vreinterpretq_s32_u32(right)), factor4);

        drflac__vst2q_f32(pOutputSamples + i*8, vzipq_f32(leftf, rightf));
    }

    for (i = (frameCount4 << 2); i < frameCount; ++i) {
        drflac_uint32 side  = pInputSamples0U32[i] << shift0;
        drflac_uint32 right = pInputSamples1U32[i] << shift1;
        drflac_uint32 left  = right + side;

        pOutputSamples[i*2+0] = (drflac_int32)left  / 8388608.0f;
        pOutputSamples[i*2+1] = (drflac_int32)right / 8388608.0f;
    }
}
#endif

static DRFLAC_INLINE void drflac_read_pcm_frames_f32__decode_right_side(drflac* pFlac, drflac_uint64 frameCount, drflac_uint32 unusedBitsPerSample, const drflac_int32* pInputSamples0, const drflac_int32* pInputSamples1, float* pOutputSamples)
{
#if defined(DRFLAC_SUPPORT_SSE2)
    if (drflac__gIsSSE2Supported && pFlac->bitsPerSample <= 24) {
        drflac_read_pcm_frames_f32__decode_right_side__sse2(pFlac, frameCount, unusedBitsPerSample, pInputSamples0, pInputSamples1, pOutputSamples);
    } else
#elif defined(DRFLAC_SUPPORT_NEON)
    if (drflac__gIsNEONSupported && pFlac->bitsPerSample <= 24) {
        drflac_read_pcm_frames_f32__decode_right_side__neon(pFlac, frameCount, unusedBitsPerSample, pInputSamples0, pInputSamples1, pOutputSamples);
    } else
#endif
    {
        /* Scalar fallback. */
#if 0
        drflac_read_pcm_frames_f32__decode_right_side__reference(pFlac, frameCount, unusedBitsPerSample, pInputSamples0, pInputSamples1, pOutputSamples);
#else
        drflac_read_pcm_frames_f32__decode_right_side__scalar(pFlac, frameCount, unusedBitsPerSample, pInputSamples0, pInputSamples1, pOutputSamples);
#endif
    }
}


#if 0
static DRFLAC_INLINE void drflac_read_pcm_frames_f32__decode_mid_side__reference(drflac* pFlac, drflac_uint64 frameCount, drflac_uint32 unusedBitsPerSample, const drflac_int32* pInputSamples0, const drflac_int32* pInputSamples1, float* pOutputSamples)
{
    for (drflac_uint64 i = 0; i < frameCount; ++i) {
        drflac_uint32 mid  = (drflac_uint32)pInputSamples0[i] << pFlac->currentFLACFrame.subframes[0].wastedBitsPerSample;
        drflac_uint32 side = (drflac_uint32)pInputSamples1[i] << pFlac->currentFLACFrame.subframes[1].wastedBitsPerSample;

        mid = (mid << 1) | (side & 0x01);

        pOutputSamples[i*2+0] = (float)((((drflac_int32)(mid + side) >> 1) << (unusedBitsPerSample)) / 2147483648.0);
        pOutputSamples[i*2+1] = (float)((((drflac_int32)(mid - side) >> 1) << (unusedBitsPerSample)) / 2147483648.0);
    }
}
#endif

static DRFLAC_INLINE void drflac_read_pcm_frames_f32__decode_mid_side__scalar(drflac* pFlac, drflac_uint64 frameCount, drflac_uint32 unusedBitsPerSample, const drflac_int32* pInputSamples0, const drflac_int32* pInputSamples1, float* pOutputSamples)
{
    drflac_uint64 i;
    drflac_uint64 frameCount4 = frameCount >> 2;
    const drflac_uint32* pInputSamples0U32 = (const drflac_uint32*)pInputSamples0;
    const drflac_uint32* pInputSamples1U32 = (const drflac_uint32*)pInputSamples1;
    drflac_uint32 shift = unusedBitsPerSample;
    float factor = 1 / 2147483648.0;

    if (shift > 0) {
        shift -= 1;
        for (i = 0; i < frameCount4; ++i) {
            drflac_uint32 temp0L;
            drflac_uint32 temp1L;
            drflac_uint32 temp2L;
            drflac_uint32 temp3L;
            drflac_uint32 temp0R;
            drflac_uint32 temp1R;
            drflac_uint32 temp2R;
            drflac_uint32 temp3R;

            drflac_uint32 mid0  = pInputSamples0U32[i*4+0] << pFlac->currentFLACFrame.subframes[0].wastedBitsPerSample;
            drflac_uint32 mid1  = pInputSamples0U32[i*4+1] << pFlac->currentFLACFrame.subframes[0].wastedBitsPerSample;
            drflac_uint32 mid2  = pInputSamples0U32[i*4+2] << pFlac->currentFLACFrame.subframes[0].wastedBitsPerSample;
            drflac_uint32 mid3  = pInputSamples0U32[i*4+3] << pFlac->currentFLACFrame.subframes[0].wastedBitsPerSample;

            drflac_uint32 side0 = pInputSamples1U32[i*4+0] << pFlac->currentFLACFrame.subframes[1].wastedBitsPerSample;
            drflac_uint32 side1 = pInputSamples1U32[i*4+1] << pFlac->currentFLACFrame.subframes[1].wastedBitsPerSample;
            drflac_uint32 side2 = pInputSamples1U32[i*4+2] << pFlac->currentFLACFrame.subframes[1].wastedBitsPerSample;
            drflac_uint32 side3 = pInputSamples1U32[i*4+3] << pFlac->currentFLACFrame.subframes[1].wastedBitsPerSample;

            mid0 = (mid0 << 1) | (side0 & 0x01);
            mid1 = (mid1 << 1) | (side1 & 0x01);
            mid2 = (mid2 << 1) | (side2 & 0x01);
            mid3 = (mid3 << 1) | (side3 & 0x01);

            temp0L = (mid0 + side0) << shift;
            temp1L = (mid1 + side1) << shift;
            temp2L = (mid2 + side2) << shift;
            temp3L = (mid3 + side3) << shift;

            temp0R = (mid0 - side0) << shift;
            temp1R = (mid1 - side1) << shift;
            temp2R = (mid2 - side2) << shift;
            temp3R = (mid3 - side3) << shift;

            pOutputSamples[i*8+0] = (drflac_int32)temp0L * factor;
            pOutputSamples[i*8+1] = (drflac_int32)temp0R * factor;
            pOutputSamples[i*8+2] = (drflac_int32)temp1L * factor;
            pOutputSamples[i*8+3] = (drflac_int32)temp1R * factor;
            pOutputSamples[i*8+4] = (drflac_int32)temp2L * factor;
            pOutputSamples[i*8+5] = (drflac_int32)temp2R * factor;
            pOutputSamples[i*8+6] = (drflac_int32)temp3L * factor;
            pOutputSamples[i*8+7] = (drflac_int32)temp3R * factor;
        }
    } else {
        for (i = 0; i < frameCount4; ++i) {
            drflac_uint32 temp0L;
            drflac_uint32 temp1L;
            drflac_uint32 temp2L;
            drflac_uint32 temp3L;
            drflac_uint32 temp0R;
            drflac_uint32 temp1R;
            drflac_uint32 temp2R;
            drflac_uint32 temp3R;

            drflac_uint32 mid0  = pInputSamples0U32[i*4+0] << pFlac->currentFLACFrame.subframes[0].wastedBitsPerSample;
            drflac_uint32 mid1  = pInputSamples0U32[i*4+1] << pFlac->currentFLACFrame.subframes[0].wastedBitsPerSample;
            drflac_uint32 mid2  = pInputSamples0U32[i*4+2] << pFlac->currentFLACFrame.subframes[0].wastedBitsPerSample;
            drflac_uint32 mid3  = pInputSamples0U32[i*4+3] << pFlac->currentFLACFrame.subframes[0].wastedBitsPerSample;

            drflac_uint32 side0 = pInputSamples1U32[i*4+0] << pFlac->currentFLACFrame.subframes[1].wastedBitsPerSample;
            drflac_uint32 side1 = pInputSamples1U32[i*4+1] << pFlac->currentFLACFrame.subframes[1].wastedBitsPerSample;
            drflac_uint32 side2 = pInputSamples1U32[i*4+2] << pFlac->currentFLACFrame.subframes[1].wastedBitsPerSample;
            drflac_uint32 side3 = pInputSamples1U32[i*4+3] << pFlac->currentFLACFrame.subframes[1].wastedBitsPerSample;

            mid0 = (mid0 << 1) | (side0 & 0x01);
            mid1 = (mid1 << 1) | (side1 & 0x01);
            mid2 = (mid2 << 1) | (side2 & 0x01);
            mid3 = (mid3 << 1) | (side3 & 0x01);

            temp0L = (drflac_uint32)((drflac_int32)(mid0 + side0) >> 1);
            temp1L = (drflac_uint32)((drflac_int32)(mid1 + side1) >> 1);
            temp2L = (drflac_uint32)((drflac_int32)(mid2 + side2) >> 1);
            temp3L = (drflac_uint32)((drflac_int32)(mid3 + side3) >> 1);

            temp0R = (drflac_uint32)((drflac_int32)(mid0 - side0) >> 1);
            temp1R = (drflac_uint32)((drflac_int32)(mid1 - side1) >> 1);
            temp2R = (drflac_uint32)((drflac_int32)(mid2 - side2) >> 1);
            temp3R = (drflac_uint32)((drflac_int32)(mid3 - side3) >> 1);

            pOutputSamples[i*8+0] = (drflac_int32)temp0L * factor;
            pOutputSamples[i*8+1] = (drflac_int32)temp0R * factor;
            pOutputSamples[i*8+2] = (drflac_int32)temp1L * factor;
            pOutputSamples[i*8+3] = (drflac_int32)temp1R * factor;
            pOutputSamples[i*8+4] = (drflac_int32)temp2L * factor;
            pOutputSamples[i*8+5] = (drflac_int32)temp2R * factor;
            pOutputSamples[i*8+6] = (drflac_int32)temp3L * factor;
            pOutputSamples[i*8+7] = (drflac_int32)temp3R * factor;
        }
    }

    for (i = (frameCount4 << 2); i < frameCount; ++i) {
        drflac_uint32 mid  = pInputSamples0U32[i] << pFlac->currentFLACFrame.subframes[0].wastedBitsPerSample;
        drflac_uint32 side = pInputSamples1U32[i] << pFlac->currentFLACFrame.subframes[1].wastedBitsPerSample;

        mid = (mid << 1) | (side & 0x01);

        pOutputSamples[i*2+0] = (drflac_int32)((drflac_uint32)((drflac_int32)(mid + side) >> 1) << unusedBitsPerSample) * factor;
        pOutputSamples[i*2+1] = (drflac_int32)((drflac_uint32)((drflac_int32)(mid - side) >> 1) << unusedBitsPerSample) * factor;
    }
}

#if defined(DRFLAC_SUPPORT_SSE2)
static DRFLAC_INLINE void drflac_read_pcm_frames_f32__decode_mid_side__sse2(drflac* pFlac, drflac_uint64 frameCount, drflac_uint32 unusedBitsPerSample, const drflac_int32* pInputSamples0, const drflac_int32* pInputSamples1, float* pOutputSamples)
{
    drflac_uint64 i;
    drflac_uint64 frameCount4 = frameCount >> 2;
    const drflac_uint32* pInputSamples0U32 = (const drflac_uint32*)pInputSamples0;
    const drflac_uint32* pInputSamples1U32 = (const drflac_uint32*)pInputSamples1;
    drflac_uint32 shift = unusedBitsPerSample - 8;
    float factor;
    __m128 factor128;

    DRFLAC_ASSERT(pFlac->bitsPerSample <= 24);

    factor = 1.0f / 8388608.0f;
    factor128 = _mm_set1_ps(factor);

    if (shift == 0) {
        for (i = 0; i < frameCount4; ++i) {
            __m128i mid;
            __m128i side;
            __m128i tempL;
            __m128i tempR;
            __m128  leftf;
            __m128  rightf;

            mid    = _mm_slli_epi32(_mm_loadu_si128((const __m128i*)pInputSamples0 + i), pFlac->currentFLACFrame.subframes[0].wastedBitsPerSample);
            side   = _mm_slli_epi32(_mm_loadu_si128((const __m128i*)pInputSamples1 + i), pFlac->currentFLACFrame.subframes[1].wastedBitsPerSample);

            mid    = _mm_or_si128(_mm_slli_epi32(mid, 1), _mm_and_si128(side, _mm_set1_epi32(0x01)));

            tempL  = _mm_srai_epi32(_mm_add_epi32(mid, side), 1);
            tempR  = _mm_srai_epi32(_mm_sub_epi32(mid, side), 1);

            leftf  = _mm_mul_ps(_mm_cvtepi32_ps(tempL), factor128);
            rightf = _mm_mul_ps(_mm_cvtepi32_ps(tempR), factor128);

            _mm_storeu_ps(pOutputSamples + i*8 + 0, _mm_unpacklo_ps(leftf, rightf));
            _mm_storeu_ps(pOutputSamples + i*8 + 4, _mm_unpackhi_ps(leftf, rightf));
        }

        for (i = (frameCount4 << 2); i < frameCount; ++i) {
            drflac_uint32 mid  = pInputSamples0U32[i] << pFlac->currentFLACFrame.subframes[0].wastedBitsPerSample;
            drflac_uint32 side = pInputSamples1U32[i] << pFlac->currentFLACFrame.subframes[1].wastedBitsPerSample;

            mid = (mid << 1) | (side & 0x01);

            pOutputSamples[i*2+0] = ((drflac_int32)(mid + side) >> 1) * factor;
            pOutputSamples[i*2+1] = ((drflac_int32)(mid - side) >> 1) * factor;
        }
    } else {
        shift -= 1;
        for (i = 0; i < frameCount4; ++i) {
            __m128i mid;
            __m128i side;
            __m128i tempL;
            __m128i tempR;
            __m128 leftf;
            __m128 rightf;

            mid    = _mm_slli_epi32(_mm_loadu_si128((const __m128i*)pInputSamples0 + i), pFlac->currentFLACFrame.subframes[0].wastedBitsPerSample);
            side   = _mm_slli_epi32(_mm_loadu_si128((const __m128i*)pInputSamples1 + i), pFlac->currentFLACFrame.subframes[1].wastedBitsPerSample);

            mid    = _mm_or_si128(_mm_slli_epi32(mid, 1), _mm_and_si128(side, _mm_set1_epi32(0x01)));

            tempL  = _mm_slli_epi32(_mm_add_epi32(mid, side), shift);
            tempR  = _mm_slli_epi32(_mm_sub_epi32(mid, side), shift);

            leftf  = _mm_mul_ps(_mm_cvtepi32_ps(tempL), factor128);
            rightf = _mm_mul_ps(_mm_cvtepi32_ps(tempR), factor128);

            _mm_storeu_ps(pOutputSamples + i*8 + 0, _mm_unpacklo_ps(leftf, rightf));
            _mm_storeu_ps(pOutputSamples + i*8 + 4, _mm_unpackhi_ps(leftf, rightf));
        }

        for (i = (frameCount4 << 2); i < frameCount; ++i) {
            drflac_uint32 mid  = pInputSamples0U32[i] << pFlac->currentFLACFrame.subframes[0].wastedBitsPerSample;
            drflac_uint32 side = pInputSamples1U32[i] << pFlac->currentFLACFrame.subframes[1].wastedBitsPerSample;

            mid = (mid << 1) | (side & 0x01);

            pOutputSamples[i*2+0] = (drflac_int32)((mid + side) << shift) * factor;
            pOutputSamples[i*2+1] = (drflac_int32)((mid - side) << shift) * factor;
        }
    }
}
#endif

#if defined(DRFLAC_SUPPORT_NEON)
static DRFLAC_INLINE void drflac_read_pcm_frames_f32__decode_mid_side__neon(drflac* pFlac, drflac_uint64 frameCount, drflac_uint32 unusedBitsPerSample, const drflac_int32* pInputSamples0, const drflac_int32* pInputSamples1, float* pOutputSamples)
{
    drflac_uint64 i;
    drflac_uint64 frameCount4 = frameCount >> 2;
    const drflac_uint32* pInputSamples0U32 = (const drflac_uint32*)pInputSamples0;
    const drflac_uint32* pInputSamples1U32 = (const drflac_uint32*)pInputSamples1;
    drflac_uint32 shift = unusedBitsPerSample - 8;
    float factor;
    float32x4_t factor4;
    int32x4_t shift4;
    int32x4_t wbps0_4;  /* Wasted Bits Per Sample */
    int32x4_t wbps1_4;  /* Wasted Bits Per Sample */

    DRFLAC_ASSERT(pFlac->bitsPerSample <= 24);

    factor  = 1.0f / 8388608.0f;
    factor4 = vdupq_n_f32(factor);
    wbps0_4 = vdupq_n_s32(pFlac->currentFLACFrame.subframes[0].wastedBitsPerSample);
    wbps1_4 = vdupq_n_s32(pFlac->currentFLACFrame.subframes[1].wastedBitsPerSample);

    if (shift == 0) {
        for (i = 0; i < frameCount4; ++i) {
            int32x4_t lefti;
            int32x4_t righti;
            float32x4_t leftf;
            float32x4_t rightf;

            uint32x4_t mid  = vshlq_u32(vld1q_u32(pInputSamples0U32 + i*4), wbps0_4);
            uint32x4_t side = vshlq_u32(vld1q_u32(pInputSamples1U32 + i*4), wbps1_4);

            mid    = vorrq_u32(vshlq_n_u32(mid, 1), vandq_u32(side, vdupq_n_u32(1)));

            lefti  = vshrq_n_s32(vreinterpretq_s32_u32(vaddq_u32(mid, side)), 1);
            righti = vshrq_n_s32(vreinterpretq_s32_u32(vsubq_u32(mid, side)), 1);

            leftf  = vmulq_f32(vcvtq_f32_s32(lefti),  factor4);
            rightf = vmulq_f32(vcvtq_f32_s32(righti), factor4);

            drflac__vst2q_f32(pOutputSamples + i*8, vzipq_f32(leftf, rightf));
        }

        for (i = (frameCount4 << 2); i < frameCount; ++i) {
            drflac_uint32 mid  = pInputSamples0U32[i] << pFlac->currentFLACFrame.subframes[0].wastedBitsPerSample;
            drflac_uint32 side = pInputSamples1U32[i] << pFlac->currentFLACFrame.subframes[1].wastedBitsPerSample;

            mid = (mid << 1) | (side & 0x01);

            pOutputSamples[i*2+0] = ((drflac_int32)(mid + side) >> 1) * factor;
            pOutputSamples[i*2+1] = ((drflac_int32)(mid - side) >> 1) * factor;
        }
    } else {
        shift -= 1;
        shift4 = vdupq_n_s32(shift);
        for (i = 0; i < frameCount4; ++i) {
            uint32x4_t mid;
            uint32x4_t side;
            int32x4_t lefti;
            int32x4_t righti;
            float32x4_t leftf;
            float32x4_t rightf;

            mid    = vshlq_u32(vld1q_u32(pInputSamples0U32 + i*4), wbps0_4);
            side   = vshlq_u32(vld1q_u32(pInputSamples1U32 + i*4), wbps1_4);

            mid    = vorrq_u32(vshlq_n_u32(mid, 1), vandq_u32(side, vdupq_n_u32(1)));

            lefti  = vreinterpretq_s32_u32(vshlq_u32(vaddq_u32(mid, side), shift4));
            righti = vreinterpretq_s32_u32(vshlq_u32(vsubq_u32(mid, side), shift4));

            leftf  = vmulq_f32(vcvtq_f32_s32(lefti),  factor4);
            rightf = vmulq_f32(vcvtq_f32_s32(righti), factor4);

            drflac__vst2q_f32(pOutputSamples + i*8, vzipq_f32(leftf, rightf));
        }

        for (i = (frameCount4 << 2); i < frameCount; ++i) {
            drflac_uint32 mid  = pInputSamples0U32[i] << pFlac->currentFLACFrame.subframes[0].wastedBitsPerSample;
            drflac_uint32 side = pInputSamples1U32[i] << pFlac->currentFLACFrame.subframes[1].wastedBitsPerSample;

            mid = (mid << 1) | (side & 0x01);

            pOutputSamples[i*2+0] = (drflac_int32)((mid + side) << shift) * factor;
            pOutputSamples[i*2+1] = (drflac_int32)((mid - side) << shift) * factor;
        }
    }
}
#endif

static DRFLAC_INLINE void drflac_read_pcm_frames_f32__decode_mid_side(drflac* pFlac, drflac_uint64 frameCount, drflac_uint32 unusedBitsPerSample, const drflac_int32* pInputSamples0, const drflac_int32* pInputSamples1, float* pOutputSamples)
{
#if defined(DRFLAC_SUPPORT_SSE2)
    if (drflac__gIsSSE2Supported && pFlac->bitsPerSample <= 24) {
        drflac_read_pcm_frames_f32__decode_mid_side__sse2(pFlac, frameCount, unusedBitsPerSample, pInputSamples0, pInputSamples1, pOutputSamples);
    } else
#elif defined(DRFLAC_SUPPORT_NEON)
    if (drflac__gIsNEONSupported && pFlac->bitsPerSample <= 24) {
        drflac_read_pcm_frames_f32__decode_mid_side__neon(pFlac, frameCount, unusedBitsPerSample, pInputSamples0, pInputSamples1, pOutputSamples);
    } else
#endif
    {
        /* Scalar fallback. */
#if 0
        drflac_read_pcm_frames_f32__decode_mid_side__reference(pFlac, frameCount, unusedBitsPerSample, pInputSamples0, pInputSamples1, pOutputSamples);
#else
        drflac_read_pcm_frames_f32__decode_mid_side__scalar(pFlac, frameCount, unusedBitsPerSample, pInputSamples0, pInputSamples1, pOutputSamples);
#endif
    }
}

#if 0
static DRFLAC_INLINE void drflac_read_pcm_frames_f32__decode_independent_stereo__reference(drflac* pFlac, drflac_uint64 frameCount, drflac_uint32 unusedBitsPerSample, const drflac_int32* pInputSamples0, const drflac_int32* pInputSamples1, float* pOutputSamples)
{
    for (drflac_uint64 i = 0; i < frameCount; ++i) {
        pOutputSamples[i*2+0] = (float)((drflac_int32)((drflac_uint32)pInputSamples0[i] << (unusedBitsPerSample + pFlac->currentFLACFrame.subframes[0].wastedBitsPerSample)) / 2147483648.0);
        pOutputSamples[i*2+1] = (float)((drflac_int32)((drflac_uint32)pInputSamples1[i] << (unusedBitsPerSample + pFlac->currentFLACFrame.subframes[1].wastedBitsPerSample)) / 2147483648.0);
    }
}
#endif

static DRFLAC_INLINE void drflac_read_pcm_frames_f32__decode_independent_stereo__scalar(drflac* pFlac, drflac_uint64 frameCount, drflac_uint32 unusedBitsPerSample, const drflac_int32* pInputSamples0, const drflac_int32* pInputSamples1, float* pOutputSamples)
{
    drflac_uint64 i;
    drflac_uint64 frameCount4 = frameCount >> 2;
    const drflac_uint32* pInputSamples0U32 = (const drflac_uint32*)pInputSamples0;
    const drflac_uint32* pInputSamples1U32 = (const drflac_uint32*)pInputSamples1;
    drflac_uint32 shift0 = unusedBitsPerSample + pFlac->currentFLACFrame.subframes[0].wastedBitsPerSample;
    drflac_uint32 shift1 = unusedBitsPerSample + pFlac->currentFLACFrame.subframes[1].wastedBitsPerSample;
    float factor = 1 / 2147483648.0;

    for (i = 0; i < frameCount4; ++i) {
        drflac_uint32 tempL0 = pInputSamples0U32[i*4+0] << shift0;
        drflac_uint32 tempL1 = pInputSamples0U32[i*4+1] << shift0;
        drflac_uint32 tempL2 = pInputSamples0U32[i*4+2] << shift0;
        drflac_uint32 tempL3 = pInputSamples0U32[i*4+3] << shift0;

        drflac_uint32 tempR0 = pInputSamples1U32[i*4+0] << shift1;
        drflac_uint32 tempR1 = pInputSamples1U32[i*4+1] << shift1;
        drflac_uint32 tempR2 = pInputSamples1U32[i*4+2] << shift1;
        drflac_uint32 tempR3 = pInputSamples1U32[i*4+3] << shift1;

        pOutputSamples[i*8+0] = (drflac_int32)tempL0 * factor;
        pOutputSamples[i*8+1] = (drflac_int32)tempR0 * factor;
        pOutputSamples[i*8+2] = (drflac_int32)tempL1 * factor;
        pOutputSamples[i*8+3] = (drflac_int32)tempR1 * factor;
        pOutputSamples[i*8+4] = (drflac_int32)tempL2 * factor;
        pOutputSamples[i*8+5] = (drflac_int32)tempR2 * factor;
        pOutputSamples[i*8+6] = (drflac_int32)tempL3 * factor;
        pOutputSamples[i*8+7] = (drflac_int32)tempR3 * factor;
    }

    for (i = (frameCount4 << 2); i < frameCount; ++i) {
        pOutputSamples[i*2+0] = (drflac_int32)(pInputSamples0U32[i] << shift0) * factor;
        pOutputSamples[i*2+1] = (drflac_int32)(pInputSamples1U32[i] << shift1) * factor;
    }
}

#if defined(DRFLAC_SUPPORT_SSE2)
static DRFLAC_INLINE void drflac_read_pcm_frames_f32__decode_independent_stereo__sse2(drflac* pFlac, drflac_uint64 frameCount, drflac_uint32 unusedBitsPerSample, const drflac_int32* pInputSamples0, const drflac_int32* pInputSamples1, float* pOutputSamples)
{
    drflac_uint64 i;
    drflac_uint64 frameCount4 = frameCount >> 2;
    const drflac_uint32* pInputSamples0U32 = (const drflac_uint32*)pInputSamples0;
    const drflac_uint32* pInputSamples1U32 = (const drflac_uint32*)pInputSamples1;
    drflac_uint32 shift0 = (unusedBitsPerSample + pFlac->currentFLACFrame.subframes[0].wastedBitsPerSample) - 8;
    drflac_uint32 shift1 = (unusedBitsPerSample + pFlac->currentFLACFrame.subframes[1].wastedBitsPerSample) - 8;

    float factor = 1.0f / 8388608.0f;
    __m128 factor128 = _mm_set1_ps(factor);

    for (i = 0; i < frameCount4; ++i) {
        __m128i lefti;
        __m128i righti;
        __m128 leftf;
        __m128 rightf;

        lefti  = _mm_slli_epi32(_mm_loadu_si128((const __m128i*)pInputSamples0 + i), shift0);
        righti = _mm_slli_epi32(_mm_loadu_si128((const __m128i*)pInputSamples1 + i), shift1);

        leftf  = _mm_mul_ps(_mm_cvtepi32_ps(lefti),  factor128);
        rightf = _mm_mul_ps(_mm_cvtepi32_ps(righti), factor128);

        _mm_storeu_ps(pOutputSamples + i*8 + 0, _mm_unpacklo_ps(leftf, rightf));
        _mm_storeu_ps(pOutputSamples + i*8 + 4, _mm_unpackhi_ps(leftf, rightf));
    }

    for (i = (frameCount4 << 2); i < frameCount; ++i) {
        pOutputSamples[i*2+0] = (drflac_int32)(pInputSamples0U32[i] << shift0) * factor;
        pOutputSamples[i*2+1] = (drflac_int32)(pInputSamples1U32[i] << shift1) * factor;
    }
}
#endif

#if defined(DRFLAC_SUPPORT_NEON)
static DRFLAC_INLINE void drflac_read_pcm_frames_f32__decode_independent_stereo__neon(drflac* pFlac, drflac_uint64 frameCount, drflac_uint32 unusedBitsPerSample, const drflac_int32* pInputSamples0, const drflac_int32* pInputSamples1, float* pOutputSamples)
{
    drflac_uint64 i;
    drflac_uint64 frameCount4 = frameCount >> 2;
    const drflac_uint32* pInputSamples0U32 = (const drflac_uint32*)pInputSamples0;
    const drflac_uint32* pInputSamples1U32 = (const drflac_uint32*)pInputSamples1;
    drflac_uint32 shift0 = (unusedBitsPerSample + pFlac->currentFLACFrame.subframes[0].wastedBitsPerSample) - 8;
    drflac_uint32 shift1 = (unusedBitsPerSample + pFlac->currentFLACFrame.subframes[1].wastedBitsPerSample) - 8;

    float factor = 1.0f / 8388608.0f;
    float32x4_t factor4 = vdupq_n_f32(factor);
    int32x4_t shift0_4  = vdupq_n_s32(shift0);
    int32x4_t shift1_4  = vdupq_n_s32(shift1);

    for (i = 0; i < frameCount4; ++i) {
        int32x4_t lefti;
        int32x4_t righti;
        float32x4_t leftf;
        float32x4_t rightf;

        lefti  = vreinterpretq_s32_u32(vshlq_u32(vld1q_u32(pInputSamples0U32 + i*4), shift0_4));
        righti = vreinterpretq_s32_u32(vshlq_u32(vld1q_u32(pInputSamples1U32 + i*4), shift1_4));

        leftf  = vmulq_f32(vcvtq_f32_s32(lefti),  factor4);
        rightf = vmulq_f32(vcvtq_f32_s32(righti), factor4);

        drflac__vst2q_f32(pOutputSamples + i*8, vzipq_f32(leftf, rightf));
    }

    for (i = (frameCount4 << 2); i < frameCount; ++i) {
        pOutputSamples[i*2+0] = (drflac_int32)(pInputSamples0U32[i] << shift0) * factor;
        pOutputSamples[i*2+1] = (drflac_int32)(pInputSamples1U32[i] << shift1) * factor;
    }
}
#endif

static DRFLAC_INLINE void drflac_read_pcm_frames_f32__decode_independent_stereo(drflac* pFlac, drflac_uint64 frameCount, drflac_uint32 unusedBitsPerSample, const drflac_int32* pInputSamples0, const drflac_int32* pInputSamples1, float* pOutputSamples)
{
#if defined(DRFLAC_SUPPORT_SSE2)
    if (drflac__gIsSSE2Supported && pFlac->bitsPerSample <= 24) {
        drflac_read_pcm_frames_f32__decode_independent_stereo__sse2(pFlac, frameCount, unusedBitsPerSample, pInputSamples0, pInputSamples1, pOutputSamples);
    } else
#elif defined(DRFLAC_SUPPORT_NEON)
    if (drflac__gIsNEONSupported && pFlac->bitsPerSample <= 24) {
        drflac_read_pcm_frames_f32__decode_independent_stereo__neon(pFlac, frameCount, unusedBitsPerSample, pInputSamples0, pInputSamples1, pOutputSamples);
    } else
#endif
    {
        /* Scalar fallback. */
#if 0
        drflac_read_pcm_frames_f32__decode_independent_stereo__reference(pFlac, frameCount, unusedBitsPerSample, pInputSamples0, pInputSamples1, pOutputSamples);
#else
        drflac_read_pcm_frames_f32__decode_independent_stereo__scalar(pFlac, frameCount, unusedBitsPerSample, pInputSamples0, pInputSamples1, pOutputSamples);
#endif
    }
}

DRFLAC_API drflac_uint64 drflac_read_pcm_frames_f32(drflac* pFlac, drflac_uint64 framesToRead, float* pBufferOut)
{
    drflac_uint64 framesRead;
    drflac_uint32 unusedBitsPerSample;

    if (pFlac == NULL || framesToRead == 0) {
        return 0;
    }

    if (pBufferOut == NULL) {
        return drflac__seek_forward_by_pcm_frames(pFlac, framesToRead);
    }

    DRFLAC_ASSERT(pFlac->bitsPerSample <= 32);
    unusedBitsPerSample = 32 - pFlac->bitsPerSample;

    framesRead = 0;
    while (framesToRead > 0) {
        /* If we've run out of samples in this frame, go to the next. */
        if (pFlac->currentFLACFrame.pcmFramesRemaining == 0) {
            if (!drflac__read_and_decode_next_flac_frame(pFlac)) {
                break;  /* Couldn't read the next frame, so just break from the loop and return. */
            }
        } else {
            unsigned int channelCount = drflac__get_channel_count_from_channel_assignment(pFlac->currentFLACFrame.header.channelAssignment);
            drflac_uint64 iFirstPCMFrame = pFlac->currentFLACFrame.header.blockSizeInPCMFrames - pFlac->currentFLACFrame.pcmFramesRemaining;
            drflac_uint64 frameCountThisIteration = framesToRead;

            if (frameCountThisIteration > pFlac->currentFLACFrame.pcmFramesRemaining) {
                frameCountThisIteration = pFlac->currentFLACFrame.pcmFramesRemaining;
            }

            if (channelCount == 2) {
                const drflac_int32* pDecodedSamples0 = pFlac->currentFLACFrame.subframes[0].pSamplesS32 + iFirstPCMFrame;
                const drflac_int32* pDecodedSamples1 = pFlac->currentFLACFrame.subframes[1].pSamplesS32 + iFirstPCMFrame;

                switch (pFlac->currentFLACFrame.header.channelAssignment)
                {
                    case DRFLAC_CHANNEL_ASSIGNMENT_LEFT_SIDE:
                    {
                        drflac_read_pcm_frames_f32__decode_left_side(pFlac, frameCountThisIteration, unusedBitsPerSample, pDecodedSamples0, pDecodedSamples1, pBufferOut);
                    } break;

                    case DRFLAC_CHANNEL_ASSIGNMENT_RIGHT_SIDE:
                    {
                        drflac_read_pcm_frames_f32__decode_right_side(pFlac, frameCountThisIteration, unusedBitsPerSample, pDecodedSamples0, pDecodedSamples1, pBufferOut);
                    } break;

                    case DRFLAC_CHANNEL_ASSIGNMENT_MID_SIDE:
                    {
                        drflac_read_pcm_frames_f32__decode_mid_side(pFlac, frameCountThisIteration, unusedBitsPerSample, pDecodedSamples0, pDecodedSamples1, pBufferOut);
                    } break;

                    case DRFLAC_CHANNEL_ASSIGNMENT_INDEPENDENT:
                    default:
                    {
                        drflac_read_pcm_frames_f32__decode_independent_stereo(pFlac, frameCountThisIteration, unusedBitsPerSample, pDecodedSamples0, pDecodedSamples1, pBufferOut);
                    } break;
                }
            } else {
                /* Generic interleaving. */
                drflac_uint64 i;
                for (i = 0; i < frameCountThisIteration; ++i) {
                    unsigned int j;
                    for (j = 0; j < channelCount; ++j) {
                        drflac_int32 sampleS32 = (drflac_int32)((drflac_uint32)(pFlac->currentFLACFrame.subframes[j].pSamplesS32[iFirstPCMFrame + i]) << (unusedBitsPerSample + pFlac->currentFLACFrame.subframes[j].wastedBitsPerSample));
                        pBufferOut[(i*channelCount)+j] = (float)(sampleS32 / 2147483648.0);
                    }
                }
            }

            framesRead                += frameCountThisIteration;
            pBufferOut                += frameCountThisIteration * channelCount;
            framesToRead              -= frameCountThisIteration;
            pFlac->currentPCMFrame    += frameCountThisIteration;
            pFlac->currentFLACFrame.pcmFramesRemaining -= (unsigned int)frameCountThisIteration;
        }
    }

    return framesRead;
}


DRFLAC_API drflac_bool32 drflac_seek_to_pcm_frame(drflac* pFlac, drflac_uint64 pcmFrameIndex)
{
    if (pFlac == NULL) {
        return DRFLAC_FALSE;
    }

    /* Don't do anything if we're already on the seek point. */
    if (pFlac->currentPCMFrame == pcmFrameIndex) {
        return DRFLAC_TRUE;
    }

    /*
    If we don't know where the first frame begins then we can't seek. This will happen when the STREAMINFO block was not present
    when the decoder was opened.
    */
    if (pFlac->firstFLACFramePosInBytes == 0) {
        return DRFLAC_FALSE;
    }

    if (pcmFrameIndex == 0) {
        pFlac->currentPCMFrame = 0;
        return drflac__seek_to_first_frame(pFlac);
    } else {
        drflac_bool32 wasSuccessful = DRFLAC_FALSE;
        drflac_uint64 originalPCMFrame = pFlac->currentPCMFrame;

        /* Clamp the sample to the end. */
        if (pcmFrameIndex > pFlac->totalPCMFrameCount) {
            pcmFrameIndex = pFlac->totalPCMFrameCount;
        }

        /* If the target sample and the current sample are in the same frame we just move the position forward. */
        if (pcmFrameIndex > pFlac->currentPCMFrame) {
            /* Forward. */
            drflac_uint32 offset = (drflac_uint32)(pcmFrameIndex - pFlac->currentPCMFrame);
            if (pFlac->currentFLACFrame.pcmFramesRemaining >  offset) {
                pFlac->currentFLACFrame.pcmFramesRemaining -= offset;
                pFlac->currentPCMFrame = pcmFrameIndex;
                return DRFLAC_TRUE;
            }
        } else {
            /* Backward. */
            drflac_uint32 offsetAbs = (drflac_uint32)(pFlac->currentPCMFrame - pcmFrameIndex);
            drflac_uint32 currentFLACFramePCMFrameCount = pFlac->currentFLACFrame.header.blockSizeInPCMFrames;
            drflac_uint32 currentFLACFramePCMFramesConsumed = currentFLACFramePCMFrameCount - pFlac->currentFLACFrame.pcmFramesRemaining;
            if (currentFLACFramePCMFramesConsumed > offsetAbs) {
                pFlac->currentFLACFrame.pcmFramesRemaining += offsetAbs;
                pFlac->currentPCMFrame = pcmFrameIndex;
                return DRFLAC_TRUE;
            }
        }

        /*
        Different techniques depending on encapsulation. Using the native FLAC seektable with Ogg encapsulation is a bit awkward so
        we'll instead use Ogg's natural seeking facility.
        */
#ifndef DR_FLAC_NO_OGG
        if (pFlac->container == drflac_container_ogg)
        {
            wasSuccessful = drflac_ogg__seek_to_pcm_frame(pFlac, pcmFrameIndex);
        }
        else
#endif
        {
            /* First try seeking via the seek table. If this fails, fall back to a brute force seek which is much slower. */
            if (/*!wasSuccessful && */!pFlac->_noSeekTableSeek) {
                wasSuccessful = drflac__seek_to_pcm_frame__seek_table(pFlac, pcmFrameIndex);
            }

#if !defined(DR_FLAC_NO_CRC)
            /* Fall back to binary search if seek table seeking fails. This requires the length of the stream to be known. */
            if (!wasSuccessful && !pFlac->_noBinarySearchSeek && pFlac->totalPCMFrameCount > 0) {
                wasSuccessful = drflac__seek_to_pcm_frame__binary_search(pFlac, pcmFrameIndex);
            }
#endif

            /* Fall back to brute force if all else fails. */
            if (!wasSuccessful && !pFlac->_noBruteForceSeek) {
                wasSuccessful = drflac__seek_to_pcm_frame__brute_force(pFlac, pcmFrameIndex);
            }
        }

        if (wasSuccessful) {
            pFlac->currentPCMFrame = pcmFrameIndex;
        } else {
            /* Seek failed. Try putting the decoder back to it's original state. */
            if (drflac_seek_to_pcm_frame(pFlac, originalPCMFrame) == DRFLAC_FALSE) {
                /* Failed to seek back to the original PCM frame. Fall back to 0. */
                drflac_seek_to_pcm_frame(pFlac, 0);
            }
        }

        return wasSuccessful;
    }
}



/* High Level APIs */

/* SIZE_MAX */
#if defined(SIZE_MAX)
    #define DRFLAC_SIZE_MAX  SIZE_MAX
#else
    #if defined(DRFLAC_64BIT)
        #define DRFLAC_SIZE_MAX  ((drflac_uint64)0xFFFFFFFFFFFFFFFF)
    #else
        #define DRFLAC_SIZE_MAX  0xFFFFFFFF
    #endif
#endif
/* End SIZE_MAX */


/* Using a macro as the definition of the drflac__full_decode_and_close_*() API family. Sue me. */
#define DRFLAC_DEFINE_FULL_READ_AND_CLOSE(extension, type) \
static type* drflac__full_read_and_close_ ## extension (drflac* pFlac, unsigned int* channelsOut, unsigned int* sampleRateOut, drflac_uint64* totalPCMFrameCountOut)\
{                                                                                                                                                                   \
    type* pSampleData = NULL;                                                                                                                                       \
    drflac_uint64 totalPCMFrameCount;                                                                                                                               \
                                                                                                                                                                    \
    DRFLAC_ASSERT(pFlac != NULL);                                                                                                                                   \
                                                                                                                                                                    \
    totalPCMFrameCount = pFlac->totalPCMFrameCount;                                                                                                                 \
                                                                                                                                                                    \
    if (totalPCMFrameCount == 0) {                                                                                                                                  \
        type buffer[4096];                                                                                                                                          \
        drflac_uint64 pcmFramesRead;                                                                                                                                \
        size_t sampleDataBufferSize = sizeof(buffer);                                                                                                               \
                                                                                                                                                                    \
        pSampleData = (type*)drflac__malloc_from_callbacks(sampleDataBufferSize, &pFlac->allocationCallbacks);                                                      \
        if (pSampleData == NULL) {                                                                                                                                  \
            goto on_error;                                                                                                                                          \
        }                                                                                                                                                           \
                                                                                                                                                                    \
        while ((pcmFramesRead = (drflac_uint64)drflac_read_pcm_frames_##extension(pFlac, sizeof(buffer)/sizeof(buffer[0])/pFlac->channels, buffer)) > 0) {          \
            if (((totalPCMFrameCount + pcmFramesRead) * pFlac->channels * sizeof(type)) > sampleDataBufferSize) {                                                   \
                type* pNewSampleData;                                                                                                                               \
                size_t newSampleDataBufferSize;                                                                                                                     \
                                                                                                                                                                    \
                newSampleDataBufferSize = sampleDataBufferSize * 2;                                                                                                 \
                pNewSampleData = (type*)drflac__realloc_from_callbacks(pSampleData, newSampleDataBufferSize, sampleDataBufferSize, &pFlac->allocationCallbacks);    \
                if (pNewSampleData == NULL) {                                                                                                                       \
                    drflac__free_from_callbacks(pSampleData, &pFlac->allocationCallbacks);                                                                          \
                    goto on_error;                                                                                                                                  \
                }                                                                                                                                                   \
                                                                                                                                                                    \
                sampleDataBufferSize = newSampleDataBufferSize;                                                                                                     \
                pSampleData = pNewSampleData;                                                                                                                       \
            }                                                                                                                                                       \
                                                                                                                                                                    \
            DRFLAC_COPY_MEMORY(pSampleData + (totalPCMFrameCount*pFlac->channels), buffer, (size_t)(pcmFramesRead*pFlac->channels*sizeof(type)));                   \
            totalPCMFrameCount += pcmFramesRead;                                                                                                                    \
        }                                                                                                                                                           \
                                                                                                                                                                    \
        /* At this point everything should be decoded, but we just want to fill the unused part buffer with silence - need to                                       \
           protect those ears from random noise! */                                                                                                                 \
        DRFLAC_ZERO_MEMORY(pSampleData + (totalPCMFrameCount*pFlac->channels), (size_t)(sampleDataBufferSize - totalPCMFrameCount*pFlac->channels*sizeof(type)));   \
    } else {                                                                                                                                                        \
        drflac_uint64 dataSize = totalPCMFrameCount*pFlac->channels*sizeof(type);                                                                                   \
        if (dataSize > (drflac_uint64)DRFLAC_SIZE_MAX) {                                                                                                            \
            goto on_error;  /* The decoded data is too big. */                                                                                                      \
        }                                                                                                                                                           \
                                                                                                                                                                    \
        pSampleData = (type*)drflac__malloc_from_callbacks((size_t)dataSize, &pFlac->allocationCallbacks);    /* <-- Safe cast as per the check above. */           \
        if (pSampleData == NULL) {                                                                                                                                  \
            goto on_error;                                                                                                                                          \
        }                                                                                                                                                           \
                                                                                                                                                                    \
        totalPCMFrameCount = drflac_read_pcm_frames_##extension(pFlac, pFlac->totalPCMFrameCount, pSampleData);                                                     \
    }                                                                                                                                                               \
                                                                                                                                                                    \
    if (sampleRateOut) *sampleRateOut = pFlac->sampleRate;                                                                                                          \
    if (channelsOut) *channelsOut = pFlac->channels;                                                                                                                \
    if (totalPCMFrameCountOut) *totalPCMFrameCountOut = totalPCMFrameCount;                                                                                         \
                                                                                                                                                                    \
    drflac_close(pFlac);                                                                                                                                            \
    return pSampleData;                                                                                                                                             \
                                                                                                                                                                    \
on_error:                                                                                                                                                           \
    drflac_close(pFlac);                                                                                                                                            \
    return NULL;                                                                                                                                                    \
}

DRFLAC_DEFINE_FULL_READ_AND_CLOSE(s32, drflac_int32)
DRFLAC_DEFINE_FULL_READ_AND_CLOSE(s16, drflac_int16)
DRFLAC_DEFINE_FULL_READ_AND_CLOSE(f32, float)

DRFLAC_API drflac_int32* drflac_open_and_read_pcm_frames_s32(drflac_read_proc onRead, drflac_seek_proc onSeek, void* pUserData, unsigned int* channelsOut, unsigned int* sampleRateOut, drflac_uint64* totalPCMFrameCountOut, const drflac_allocation_callbacks* pAllocationCallbacks)
{
    drflac* pFlac;

    if (channelsOut) {
        *channelsOut = 0;
    }
    if (sampleRateOut) {
        *sampleRateOut = 0;
    }
    if (totalPCMFrameCountOut) {
        *totalPCMFrameCountOut = 0;
    }

    pFlac = drflac_open(onRead, onSeek, pUserData, pAllocationCallbacks);
    if (pFlac == NULL) {
        return NULL;
    }

    return drflac__full_read_and_close_s32(pFlac, channelsOut, sampleRateOut, totalPCMFrameCountOut);
}

DRFLAC_API drflac_int16* drflac_open_and_read_pcm_frames_s16(drflac_read_proc onRead, drflac_seek_proc onSeek, void* pUserData, unsigned int* channelsOut, unsigned int* sampleRateOut, drflac_uint64* totalPCMFrameCountOut, const drflac_allocation_callbacks* pAllocationCallbacks)
{
    drflac* pFlac;

    if (channelsOut) {
        *channelsOut = 0;
    }
    if (sampleRateOut) {
        *sampleRateOut = 0;
    }
    if (totalPCMFrameCountOut) {
        *totalPCMFrameCountOut = 0;
    }

    pFlac = drflac_open(onRead, onSeek, pUserData, pAllocationCallbacks);
    if (pFlac == NULL) {
        return NULL;
    }

    return drflac__full_read_and_close_s16(pFlac, channelsOut, sampleRateOut, totalPCMFrameCountOut);
}

DRFLAC_API float* drflac_open_and_read_pcm_frames_f32(drflac_read_proc onRead, drflac_seek_proc onSeek, void* pUserData, unsigned int* channelsOut, unsigned int* sampleRateOut, drflac_uint64* totalPCMFrameCountOut, const drflac_allocation_callbacks* pAllocationCallbacks)
{
    drflac* pFlac;

    if (channelsOut) {
        *channelsOut = 0;
    }
    if (sampleRateOut) {
        *sampleRateOut = 0;
    }
    if (totalPCMFrameCountOut) {
        *totalPCMFrameCountOut = 0;
    }

    pFlac = drflac_open(onRead, onSeek, pUserData, pAllocationCallbacks);
    if (pFlac == NULL) {
        return NULL;
    }

    return drflac__full_read_and_close_f32(pFlac, channelsOut, sampleRateOut, totalPCMFrameCountOut);
}

#ifndef DR_FLAC_NO_STDIO
DRFLAC_API drflac_int32* drflac_open_file_and_read_pcm_frames_s32(const char* filename, unsigned int* channels, unsigned int* sampleRate, drflac_uint64* totalPCMFrameCount, const drflac_allocation_callbacks* pAllocationCallbacks)
{
    drflac* pFlac;

    if (sampleRate) {
        *sampleRate = 0;
    }
    if (channels) {
        *channels = 0;
    }
    if (totalPCMFrameCount) {
        *totalPCMFrameCount = 0;
    }

    pFlac = drflac_open_file(filename, pAllocationCallbacks);
    if (pFlac == NULL) {
        return NULL;
    }

    return drflac__full_read_and_close_s32(pFlac, channels, sampleRate, totalPCMFrameCount);
}

DRFLAC_API drflac_int16* drflac_open_file_and_read_pcm_frames_s16(const char* filename, unsigned int* channels, unsigned int* sampleRate, drflac_uint64* totalPCMFrameCount, const drflac_allocation_callbacks* pAllocationCallbacks)
{
    drflac* pFlac;

    if (sampleRate) {
        *sampleRate = 0;
    }
    if (channels) {
        *channels = 0;
    }
    if (totalPCMFrameCount) {
        *totalPCMFrameCount = 0;
    }

    pFlac = drflac_open_file(filename, pAllocationCallbacks);
    if (pFlac == NULL) {
        return NULL;
    }

    return drflac__full_read_and_close_s16(pFlac, channels, sampleRate, totalPCMFrameCount);
}

DRFLAC_API float* drflac_open_file_and_read_pcm_frames_f32(const char* filename, unsigned int* channels, unsigned int* sampleRate, drflac_uint64* totalPCMFrameCount, const drflac_allocation_callbacks* pAllocationCallbacks)
{
    drflac* pFlac;

    if (sampleRate) {
        *sampleRate = 0;
    }
    if (channels) {
        *channels = 0;
    }
    if (totalPCMFrameCount) {
        *totalPCMFrameCount = 0;
    }

    pFlac = drflac_open_file(filename, pAllocationCallbacks);
    if (pFlac == NULL) {
        return NULL;
    }

    return drflac__full_read_and_close_f32(pFlac, channels, sampleRate, totalPCMFrameCount);
}
#endif

DRFLAC_API drflac_int32* drflac_open_memory_and_read_pcm_frames_s32(const void* data, size_t dataSize, unsigned int* channels, unsigned int* sampleRate, drflac_uint64* totalPCMFrameCount, const drflac_allocation_callbacks* pAllocationCallbacks)
{
    drflac* pFlac;

    if (sampleRate) {
        *sampleRate = 0;
    }
    if (channels) {
        *channels = 0;
    }
    if (totalPCMFrameCount) {
        *totalPCMFrameCount = 0;
    }

    pFlac = drflac_open_memory(data, dataSize, pAllocationCallbacks);
    if (pFlac == NULL) {
        return NULL;
    }

    return drflac__full_read_and_close_s32(pFlac, channels, sampleRate, totalPCMFrameCount);
}

DRFLAC_API drflac_int16* drflac_open_memory_and_read_pcm_frames_s16(const void* data, size_t dataSize, unsigned int* channels, unsigned int* sampleRate, drflac_uint64* totalPCMFrameCount, const drflac_allocation_callbacks* pAllocationCallbacks)
{
    drflac* pFlac;

    if (sampleRate) {
        *sampleRate = 0;
    }
    if (channels) {
        *channels = 0;
    }
    if (totalPCMFrameCount) {
        *totalPCMFrameCount = 0;
    }

    pFlac = drflac_open_memory(data, dataSize, pAllocationCallbacks);
    if (pFlac == NULL) {
        return NULL;
    }

    return drflac__full_read_and_close_s16(pFlac, channels, sampleRate, totalPCMFrameCount);
}

DRFLAC_API float* drflac_open_memory_and_read_pcm_frames_f32(const void* data, size_t dataSize, unsigned int* channels, unsigned int* sampleRate, drflac_uint64* totalPCMFrameCount, const drflac_allocation_callbacks* pAllocationCallbacks)
{
    drflac* pFlac;

    if (sampleRate) {
        *sampleRate = 0;
    }
    if (channels) {
        *channels = 0;
    }
    if (totalPCMFrameCount) {
        *totalPCMFrameCount = 0;
    }

    pFlac = drflac_open_memory(data, dataSize, pAllocationCallbacks);
    if (pFlac == NULL) {
        return NULL;
    }

    return drflac__full_read_and_close_f32(pFlac, channels, sampleRate, totalPCMFrameCount);
}


DRFLAC_API void drflac_free(void* p, const drflac_allocation_callbacks* pAllocationCallbacks)
{
    if (pAllocationCallbacks != NULL) {
        drflac__free_from_callbacks(p, pAllocationCallbacks);
    } else {
        drflac__free_default(p, NULL);
    }
}




DRFLAC_API void drflac_init_vorbis_comment_iterator(drflac_vorbis_comment_iterator* pIter, drflac_uint32 commentCount, const void* pComments)
{
    if (pIter == NULL) {
        return;
    }

    pIter->countRemaining = commentCount;
    pIter->pRunningData   = (const char*)pComments;
}

DRFLAC_API const char* drflac_next_vorbis_comment(drflac_vorbis_comment_iterator* pIter, drflac_uint32* pCommentLengthOut)
{
    drflac_int32 length;
    const char* pComment;

    /* Safety. */
    if (pCommentLengthOut) {
        *pCommentLengthOut = 0;
    }

    if (pIter == NULL || pIter->countRemaining == 0 || pIter->pRunningData == NULL) {
        return NULL;
    }

    length = drflac__le2host_32_ptr_unaligned(pIter->pRunningData);
    pIter->pRunningData += 4;

    pComment = pIter->pRunningData;
    pIter->pRunningData += length;
    pIter->countRemaining -= 1;

    if (pCommentLengthOut) {
        *pCommentLengthOut = length;
    }

    return pComment;
}




DRFLAC_API void drflac_init_cuesheet_track_iterator(drflac_cuesheet_track_iterator* pIter, drflac_uint32 trackCount, const void* pTrackData)
{
    if (pIter == NULL) {
        return;
    }

    pIter->countRemaining = trackCount;
    pIter->pRunningData   = (const char*)pTrackData;
}

DRFLAC_API drflac_bool32 drflac_next_cuesheet_track(drflac_cuesheet_track_iterator* pIter, drflac_cuesheet_track* pCuesheetTrack)
{
    drflac_cuesheet_track cuesheetTrack;
    const char* pRunningData;
    drflac_uint64 offsetHi;
    drflac_uint64 offsetLo;

    if (pIter == NULL || pIter->countRemaining == 0 || pIter->pRunningData == NULL) {
        return DRFLAC_FALSE;
    }

    pRunningData = pIter->pRunningData;

    offsetHi                   = drflac__be2host_32(*(const drflac_uint32*)pRunningData); pRunningData += 4;
    offsetLo                   = drflac__be2host_32(*(const drflac_uint32*)pRunningData); pRunningData += 4;
    cuesheetTrack.offset       = offsetLo | (offsetHi << 32);
    cuesheetTrack.trackNumber  = pRunningData[0];                                         pRunningData += 1;
    DRFLAC_COPY_MEMORY(cuesheetTrack.ISRC, pRunningData, sizeof(cuesheetTrack.ISRC));     pRunningData += 12;
    cuesheetTrack.isAudio      = (pRunningData[0] & 0x80) != 0;
    cuesheetTrack.preEmphasis  = (pRunningData[0] & 0x40) != 0;                           pRunningData += 14;
    cuesheetTrack.indexCount   = pRunningData[0];                                         pRunningData += 1;
    cuesheetTrack.pIndexPoints = (const drflac_cuesheet_track_index*)pRunningData;        pRunningData += cuesheetTrack.indexCount * sizeof(drflac_cuesheet_track_index);

    pIter->pRunningData = pRunningData;
    pIter->countRemaining -= 1;

    if (pCuesheetTrack) {
        *pCuesheetTrack = cuesheetTrack;
    }

    return DRFLAC_TRUE;
}

#if defined(__clang__) || (defined(__GNUC__) && (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 6)))
    #pragma GCC diagnostic pop
#endif
#endif  /* dr_flac_c */
#endif  /* DR_FLAC_IMPLEMENTATION */


/*
REVISION HISTORY
================
v0.12.41 - 2023-06-17
  - Fix an incorrect date in revision history. No functional change.

v0.12.40 - 2023-05-22
  - Minor code restructure. No functional change.

v0.12.39 - 2022-09-17
  - Fix compilation with DJGPP.
  - Fix compilation error with Visual Studio 2019 and the ARM build.
  - Fix an error with SSE 4.1 detection.
  - Add support for disabling wchar_t with DR_WAV_NO_WCHAR.
  - Improve compatibility with compilers which lack support for explicit struct packing.
  - Improve compatibility with low-end and embedded hardware by reducing the amount of stack
    allocation when loading an Ogg encapsulated file.

v0.12.38 - 2022-04-10
  - Fix compilation error on older versions of GCC.

v0.12.37 - 2022-02-12
  - Improve ARM detection.

v0.12.36 - 2022-02-07
  - Fix a compilation error with the ARM build.

v0.12.35 - 2022-02-06
  - Fix a bug due to underestimating the amount of precision required for the prediction stage.
  - Fix some bugs found from fuzz testing.

v0.12.34 - 2022-01-07
  - Fix some misalignment bugs when reading metadata.

v0.12.33 - 2021-12-22
  - Fix a bug with seeking when the seek table does not start at PCM frame 0.

v0.12.32 - 2021-12-11
  - Fix a warning with Clang.

v0.12.31 - 2021-08-16
  - Silence some warnings.

v0.12.30 - 2021-07-31
  - Fix platform detection for ARM64.

v0.12.29 - 2021-04-02
  - Fix a bug where the running PCM frame index is set to an invalid value when over-seeking.
  - Fix a decoding error due to an incorrect validation check.

v0.12.28 - 2021-02-21
  - Fix a warning due to referencing _MSC_VER when it is undefined.

v0.12.27 - 2021-01-31
  - Fix a static analysis warning.

v0.12.26 - 2021-01-17
  - Fix a compilation warning due to _BSD_SOURCE being deprecated.

v0.12.25 - 2020-12-26
  - Update documentation.

v0.12.24 - 2020-11-29
  - Fix ARM64/NEON detection when compiling with MSVC.

v0.12.23 - 2020-11-21
  - Fix compilation with OpenWatcom.

v0.12.22 - 2020-11-01
  - Fix an error with the previous release.

v0.12.21 - 2020-11-01
  - Fix a possible deadlock when seeking.
  - Improve compiler support for older versions of GCC.

v0.12.20 - 2020-09-08
  - Fix a compilation error on older compilers.

v0.12.19 - 2020-08-30
  - Fix a bug due to an undefined 32-bit shift.

v0.12.18 - 2020-08-14
  - Fix a crash when compiling with clang-cl.

v0.12.17 - 2020-08-02
  - Simplify sized types.

v0.12.16 - 2020-07-25
  - Fix a compilation warning.

v0.12.15 - 2020-07-06
  - Check for negative LPC shifts and return an error.

v0.12.14 - 2020-06-23
  - Add include guard for the implementation section.

v0.12.13 - 2020-05-16
  - Add compile-time and run-time version querying.
    - DRFLAC_VERSION_MINOR
    - DRFLAC_VERSION_MAJOR
    - DRFLAC_VERSION_REVISION
    - DRFLAC_VERSION_STRING
    - drflac_version()
    - drflac_version_string()

v0.12.12 - 2020-04-30
  - Fix compilation errors with VC6.

v0.12.11 - 2020-04-19
  - Fix some pedantic warnings.
  - Fix some undefined behaviour warnings.

v0.12.10 - 2020-04-10
  - Fix some bugs when trying to seek with an invalid seek table.

v0.12.9 - 2020-04-05
  - Fix warnings.

v0.12.8 - 2020-04-04
  - Add drflac_open_file_w() and drflac_open_file_with_metadata_w().
  - Fix some static analysis warnings.
  - Minor documentation updates.

v0.12.7 - 2020-03-14
  - Fix compilation errors with VC6.

v0.12.6 - 2020-03-07
  - Fix compilation error with Visual Studio .NET 2003.

v0.12.5 - 2020-01-30
  - Silence some static analysis warnings.

v0.12.4 - 2020-01-29
  - Silence some static analysis warnings.

v0.12.3 - 2019-12-02
  - Fix some warnings when compiling with GCC and the -Og flag.
  - Fix a crash in out-of-memory situations.
  - Fix potential integer overflow bug.
  - Fix some static analysis warnings.
  - Fix a possible crash when using custom memory allocators without a custom realloc() implementation.
  - Fix a bug with binary search seeking where the bits per sample is not a multiple of 8.

v0.12.2 - 2019-10-07
  - Internal code clean up.

v0.12.1 - 2019-09-29
  - Fix some Clang Static Analyzer warnings.
  - Fix an unused variable warning.

v0.12.0 - 2019-09-23
  - API CHANGE: Add support for user defined memory allocation routines. This system allows the program to specify their own memory allocation
    routines with a user data pointer for client-specific contextual data. This adds an extra parameter to the end of the following APIs:
    - drflac_open()
    - drflac_open_relaxed()
    - drflac_open_with_metadata()
    - drflac_open_with_metadata_relaxed()
    - drflac_open_file()
    - drflac_open_file_with_metadata()
    - drflac_open_memory()
    - drflac_open_memory_with_metadata()
    - drflac_open_and_read_pcm_frames_s32()
    - drflac_open_and_read_pcm_frames_s16()
    - drflac_open_and_read_pcm_frames_f32()
    - drflac_open_file_and_read_pcm_frames_s32()
    - drflac_open_file_and_read_pcm_frames_s16()
    - drflac_open_file_and_read_pcm_frames_f32()
    - drflac_open_memory_and_read_pcm_frames_s32()
    - drflac_open_memory_and_read_pcm_frames_s16()
    - drflac_open_memory_and_read_pcm_frames_f32()
    Set this extra parameter to NULL to use defaults which is the same as the previous behaviour. Setting this NULL will use
    DRFLAC_MALLOC, DRFLAC_REALLOC and DRFLAC_FREE.
  - Remove deprecated APIs:
    - drflac_read_s32()
    - drflac_read_s16()
    - drflac_read_f32()
    - drflac_seek_to_sample()
    - drflac_open_and_decode_s32()
    - drflac_open_and_decode_s16()
    - drflac_open_and_decode_f32()
    - drflac_open_and_decode_file_s32()
    - drflac_open_and_decode_file_s16()
    - drflac_open_and_decode_file_f32()
    - drflac_open_and_decode_memory_s32()
    - drflac_open_and_decode_memory_s16()
    - drflac_open_and_decode_memory_f32()
  - Remove drflac.totalSampleCount which is now replaced with drflac.totalPCMFrameCount. You can emulate drflac.totalSampleCount
    by doing pFlac->totalPCMFrameCount*pFlac->channels.
  - Rename drflac.currentFrame to drflac.currentFLACFrame to remove ambiguity with PCM frames.
  - Fix errors when seeking to the end of a stream.
  - Optimizations to seeking.
  - SSE improvements and optimizations.
  - ARM NEON optimizations.
  - Optimizations to drflac_read_pcm_frames_s16().
  - Optimizations to drflac_read_pcm_frames_s32().

v0.11.10 - 2019-06-26
  - Fix a compiler error.

v0.11.9 - 2019-06-16
  - Silence some ThreadSanitizer warnings.

v0.11.8 - 2019-05-21
  - Fix warnings.

v0.11.7 - 2019-05-06
  - C89 fixes.

v0.11.6 - 2019-05-05
  - Add support for C89.
  - Fix a compiler warning when CRC is disabled.
  - Change license to choice of public domain or MIT-0.

v0.11.5 - 2019-04-19
  - Fix a compiler error with GCC.

v0.11.4 - 2019-04-17
  - Fix some warnings with GCC when compiling with -std=c99.

v0.11.3 - 2019-04-07
  - Silence warnings with GCC.

v0.11.2 - 2019-03-10
  - Fix a warning.

v0.11.1 - 2019-02-17
  - Fix a potential bug with seeking.

v0.11.0 - 2018-12-16
  - API CHANGE: Deprecated drflac_read_s32(), drflac_read_s16() and drflac_read_f32() and replaced them with
    drflac_read_pcm_frames_s32(), drflac_read_pcm_frames_s16() and drflac_read_pcm_frames_f32(). The new APIs take
    and return PCM frame counts instead of sample counts. To upgrade you will need to change the input count by
    dividing it by the channel count, and then do the same with the return value.
  - API_CHANGE: Deprecated drflac_seek_to_sample() and replaced with drflac_seek_to_pcm_frame(). Same rules as
    the changes to drflac_read_*() apply.
  - API CHANGE: Deprecated drflac_open_and_decode_*() and replaced with drflac_open_*_and_read_*(). Same rules as
    the changes to drflac_read_*() apply.
  - Optimizations.

v0.10.0 - 2018-09-11
  - Remove the DR_FLAC_NO_WIN32_IO option and the Win32 file IO functionality. If you need to use Win32 file IO you
    need to do it yourself via the callback API.
  - Fix the clang build.
  - Fix undefined behavior.
  - Fix errors with CUESHEET metdata blocks.
  - Add an API for iterating over each cuesheet track in the CUESHEET metadata block. This works the same way as the
    Vorbis comment API.
  - Other miscellaneous bug fixes, mostly relating to invalid FLAC streams.
  - Minor optimizations.

v0.9.11 - 2018-08-29
  - Fix a bug with sample reconstruction.

v0.9.10 - 2018-08-07
  - Improve 64-bit detection.

v0.9.9 - 2018-08-05
  - Fix C++ build on older versions of GCC.

v0.9.8 - 2018-07-24
  - Fix compilation errors.

v0.9.7 - 2018-07-05
  - Fix a warning.

v0.9.6 - 2018-06-29
  - Fix some typos.

v0.9.5 - 2018-06-23
  - Fix some warnings.

v0.9.4 - 2018-06-14
  - Optimizations to seeking.
  - Clean up.

v0.9.3 - 2018-05-22
  - Bug fix.

v0.9.2 - 2018-05-12
  - Fix a compilation error due to a missing break statement.

v0.9.1 - 2018-04-29
  - Fix compilation error with Clang.

v0.9 - 2018-04-24
  - Fix Clang build.
  - Start using major.minor.revision versioning.

v0.8g - 2018-04-19
  - Fix build on non-x86/x64 architectures.

v0.8f - 2018-02-02
  - Stop pretending to support changing rate/channels mid stream.

v0.8e - 2018-02-01
  - Fix a crash when the block size of a frame is larger than the maximum block size defined by the FLAC stream.
  - Fix a crash the the Rice partition order is invalid.

v0.8d - 2017-09-22
  - Add support for decoding streams with ID3 tags. ID3 tags are just skipped.

v0.8c - 2017-09-07
  - Fix warning on non-x86/x64 architectures.

v0.8b - 2017-08-19
  - Fix build on non-x86/x64 architectures.

v0.8a - 2017-08-13
  - A small optimization for the Clang build.

v0.8 - 2017-08-12
  - API CHANGE: Rename dr_* types to drflac_*.
  - Optimizations. This brings dr_flac back to about the same class of efficiency as the reference implementation.
  - Add support for custom implementations of malloc(), realloc(), etc.
  - Add CRC checking to Ogg encapsulated streams.
  - Fix VC++ 6 build. This is only for the C++ compiler. The C compiler is not currently supported.
  - Bug fixes.

v0.7 - 2017-07-23
  - Add support for opening a stream without a header block. To do this, use drflac_open_relaxed() / drflac_open_with_metadata_relaxed().

v0.6 - 2017-07-22
  - Add support for recovering from invalid frames. With this change, dr_flac will simply skip over invalid frames as if they
    never existed. Frames are checked against their sync code, the CRC-8 of the frame header and the CRC-16 of the whole frame.

v0.5 - 2017-07-16
  - Fix typos.
  - Change drflac_bool* types to unsigned.
  - Add CRC checking. This makes dr_flac slower, but can be disabled with #define DR_FLAC_NO_CRC.

v0.4f - 2017-03-10
  - Fix a couple of bugs with the bitstreaming code.

v0.4e - 2017-02-17
  - Fix some warnings.

v0.4d - 2016-12-26
  - Add support for 32-bit floating-point PCM decoding.
  - Use drflac_int* and drflac_uint* sized types to improve compiler support.
  - Minor improvements to documentation.

v0.4c - 2016-12-26
  - Add support for signed 16-bit integer PCM decoding.

v0.4b - 2016-10-23
  - A minor change to drflac_bool8 and drflac_bool32 types.

v0.4a - 2016-10-11
  - Rename drBool32 to drflac_bool32 for styling consistency.

v0.4 - 2016-09-29
  - API/ABI CHANGE: Use fixed size 32-bit booleans instead of the built-in bool type.
  - API CHANGE: Rename drflac_open_and_decode*() to drflac_open_and_decode*_s32().
  - API CHANGE: Swap the order of "channels" and "sampleRate" parameters in drflac_open_and_decode*(). Rationale for this is to
    keep it consistent with drflac_audio.

v0.3f - 2016-09-21
  - Fix a warning with GCC.

v0.3e - 2016-09-18
  - Fixed a bug where GCC 4.3+ was not getting properly identified.
  - Fixed a few typos.
  - Changed date formats to ISO 8601 (YYYY-MM-DD).

v0.3d - 2016-06-11
  - Minor clean up.

v0.3c - 2016-05-28
  - Fixed compilation error.

v0.3b - 2016-05-16
  - Fixed Linux/GCC build.
  - Updated documentation.

v0.3a - 2016-05-15
  - Minor fixes to documentation.

v0.3 - 2016-05-11
  - Optimizations. Now at about parity with the reference implementation on 32-bit builds.
  - Lots of clean up.

v0.2b - 2016-05-10
  - Bug fixes.

v0.2a - 2016-05-10
  - Made drflac_open_and_decode() more robust.
  - Removed an unused debugging variable

v0.2 - 2016-05-09
  - Added support for Ogg encapsulation.
  - API CHANGE. Have the onSeek callback take a third argument which specifies whether or not the seek
    should be relative to the start or the current position. Also changes the seeking rules such that
    seeking offsets will never be negative.
  - Have drflac_open_and_decode() fail gracefully if the stream has an unknown total sample count.

v0.1b - 2016-05-07
  - Properly close the file handle in drflac_open_file() and family when the decoder fails to initialize.
  - Removed a stale comment.

v0.1a - 2016-05-05
  - Minor formatting changes.
  - Fixed a warning on the GCC build.

v0.1 - 2016-05-03
  - Initial versioned release.
*/

/*
This software is available as a choice of the following licenses. Choose
whichever you prefer.

===============================================================================
ALTERNATIVE 1 - Public Domain (www.unlicense.org)
===============================================================================
This is free and unencumbered software released into the public domain.

Anyone is free to copy, modify, publish, use, compile, sell, or distribute this
software, either in source code form or as a compiled binary, for any purpose,
commercial or non-commercial, and by any means.

In jurisdictions that recognize copyright laws, the author or authors of this
software dedicate any and all copyright interest in the software to the public
domain. We make this dedication for the benefit of the public at large and to
the detriment of our heirs and successors. We intend this dedication to be an
overt act of relinquishment in perpetuity of all present and future rights to
this software under copyright law.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

For more information, please refer to <http://unlicense.org/>

===============================================================================
ALTERNATIVE 2 - MIT No Attribution
===============================================================================
Copyright 2023 David Reid

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
