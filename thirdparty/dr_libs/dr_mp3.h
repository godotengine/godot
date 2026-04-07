/*
MP3 audio decoder. Choice of public domain or MIT-0. See license statements at the end of this file.
dr_mp3 - v0.7.2 - 2025-12-02

David Reid - mackron@gmail.com

GitHub: https://github.com/mackron/dr_libs

Based on minimp3 (https://github.com/lieff/minimp3) which is where the real work was done. See the bottom of this file for differences between minimp3 and dr_mp3.
*/

/*
Introduction
=============
dr_mp3 is a single file library. To use it, do something like the following in one .c file.

    ```c
    #define DR_MP3_IMPLEMENTATION
    #include "dr_mp3.h"
    ```

You can then #include this file in other parts of the program as you would with any other header file. To decode audio data, do something like the following:

    ```c
    drmp3 mp3;
    if (!drmp3_init_file(&mp3, "MySong.mp3", NULL)) {
        // Failed to open file
    }

    ...

    drmp3_uint64 framesRead = drmp3_read_pcm_frames_f32(pMP3, framesToRead, pFrames);
    ```

The drmp3 object is transparent so you can get access to the channel count and sample rate like so:

    ```
    drmp3_uint32 channels = mp3.channels;
    drmp3_uint32 sampleRate = mp3.sampleRate;
    ```

The example above initializes a decoder from a file, but you can also initialize it from a block of memory and read and seek callbacks with
`drmp3_init_memory()` and `drmp3_init()` respectively.

You do not need to do any annoying memory management when reading PCM frames - this is all managed internally. You can request any number of PCM frames in each
call to `drmp3_read_pcm_frames_f32()` and it will return as many PCM frames as it can, up to the requested amount.

You can also decode an entire file in one go with `drmp3_open_and_read_pcm_frames_f32()`, `drmp3_open_memory_and_read_pcm_frames_f32()` and
`drmp3_open_file_and_read_pcm_frames_f32()`.


Build Options
=============
#define these options before including this file.

#define DR_MP3_NO_STDIO
  Disable drmp3_init_file(), etc.

#define DR_MP3_NO_SIMD
  Disable SIMD optimizations.
*/

#ifndef dr_mp3_h
#define dr_mp3_h

#ifdef __cplusplus
extern "C" {
#endif

#define DRMP3_STRINGIFY(x)      #x
#define DRMP3_XSTRINGIFY(x)     DRMP3_STRINGIFY(x)

#define DRMP3_VERSION_MAJOR     0
#define DRMP3_VERSION_MINOR     7
#define DRMP3_VERSION_REVISION  2
#define DRMP3_VERSION_STRING    DRMP3_XSTRINGIFY(DRMP3_VERSION_MAJOR) "." DRMP3_XSTRINGIFY(DRMP3_VERSION_MINOR) "." DRMP3_XSTRINGIFY(DRMP3_VERSION_REVISION)

#include <stddef.h> /* For size_t. */

/* Sized Types */
typedef   signed char           drmp3_int8;
typedef unsigned char           drmp3_uint8;
typedef   signed short          drmp3_int16;
typedef unsigned short          drmp3_uint16;
typedef   signed int            drmp3_int32;
typedef unsigned int            drmp3_uint32;
#if defined(_MSC_VER) && !defined(__clang__)
    typedef   signed __int64    drmp3_int64;
    typedef unsigned __int64    drmp3_uint64;
#else
    #if defined(__clang__) || (defined(__GNUC__) && (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 6)))
        #pragma GCC diagnostic push
        #pragma GCC diagnostic ignored "-Wlong-long"
        #if defined(__clang__)
            #pragma GCC diagnostic ignored "-Wc++11-long-long"
        #endif
    #endif
    typedef   signed long long  drmp3_int64;
    typedef unsigned long long  drmp3_uint64;
    #if defined(__clang__) || (defined(__GNUC__) && (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 6)))
        #pragma GCC diagnostic pop
    #endif
#endif
#if defined(__LP64__) || defined(_WIN64) || (defined(__x86_64__) && !defined(__ILP32__)) || defined(_M_X64) || defined(__ia64) || defined (_M_IA64) || defined(__aarch64__) || defined(_M_ARM64) || defined(_M_ARM64EC) || defined(__powerpc64__)
    typedef drmp3_uint64        drmp3_uintptr;
#else
    typedef drmp3_uint32        drmp3_uintptr;
#endif
typedef drmp3_uint8             drmp3_bool8;
typedef drmp3_uint32            drmp3_bool32;
#define DRMP3_TRUE              1
#define DRMP3_FALSE             0

/* Weird shifting syntax is for VC6 compatibility. */
#define DRMP3_UINT64_MAX        (((drmp3_uint64)0xFFFFFFFF << 32) | (drmp3_uint64)0xFFFFFFFF)
/* End Sized Types */

/* Decorations */
#if !defined(DRMP3_API)
    #if defined(DRMP3_DLL)
        #if defined(_WIN32)
            #define DRMP3_DLL_IMPORT  __declspec(dllimport)
            #define DRMP3_DLL_EXPORT  __declspec(dllexport)
            #define DRMP3_DLL_PRIVATE static
        #else
            #if defined(__GNUC__) && __GNUC__ >= 4
                #define DRMP3_DLL_IMPORT  __attribute__((visibility("default")))
                #define DRMP3_DLL_EXPORT  __attribute__((visibility("default")))
                #define DRMP3_DLL_PRIVATE __attribute__((visibility("hidden")))
            #else
                #define DRMP3_DLL_IMPORT
                #define DRMP3_DLL_EXPORT
                #define DRMP3_DLL_PRIVATE static
            #endif
        #endif

        #if defined(DR_MP3_IMPLEMENTATION)
            #define DRMP3_API  DRMP3_DLL_EXPORT
        #else
            #define DRMP3_API  DRMP3_DLL_IMPORT
        #endif
        #define DRMP3_PRIVATE DRMP3_DLL_PRIVATE
    #else
        #define DRMP3_API extern
        #define DRMP3_PRIVATE static
    #endif
#endif
/* End Decorations */

/* Result Codes */
typedef drmp3_int32 drmp3_result;
#define DRMP3_SUCCESS                        0
#define DRMP3_ERROR                         -1   /* A generic error. */
#define DRMP3_INVALID_ARGS                  -2
#define DRMP3_INVALID_OPERATION             -3
#define DRMP3_OUT_OF_MEMORY                 -4
#define DRMP3_OUT_OF_RANGE                  -5
#define DRMP3_ACCESS_DENIED                 -6
#define DRMP3_DOES_NOT_EXIST                -7
#define DRMP3_ALREADY_EXISTS                -8
#define DRMP3_TOO_MANY_OPEN_FILES           -9
#define DRMP3_INVALID_FILE                  -10
#define DRMP3_TOO_BIG                       -11
#define DRMP3_PATH_TOO_LONG                 -12
#define DRMP3_NAME_TOO_LONG                 -13
#define DRMP3_NOT_DIRECTORY                 -14
#define DRMP3_IS_DIRECTORY                  -15
#define DRMP3_DIRECTORY_NOT_EMPTY           -16
#define DRMP3_END_OF_FILE                   -17
#define DRMP3_NO_SPACE                      -18
#define DRMP3_BUSY                          -19
#define DRMP3_IO_ERROR                      -20
#define DRMP3_INTERRUPT                     -21
#define DRMP3_UNAVAILABLE                   -22
#define DRMP3_ALREADY_IN_USE                -23
#define DRMP3_BAD_ADDRESS                   -24
#define DRMP3_BAD_SEEK                      -25
#define DRMP3_BAD_PIPE                      -26
#define DRMP3_DEADLOCK                      -27
#define DRMP3_TOO_MANY_LINKS                -28
#define DRMP3_NOT_IMPLEMENTED               -29
#define DRMP3_NO_MESSAGE                    -30
#define DRMP3_BAD_MESSAGE                   -31
#define DRMP3_NO_DATA_AVAILABLE             -32
#define DRMP3_INVALID_DATA                  -33
#define DRMP3_TIMEOUT                       -34
#define DRMP3_NO_NETWORK                    -35
#define DRMP3_NOT_UNIQUE                    -36
#define DRMP3_NOT_SOCKET                    -37
#define DRMP3_NO_ADDRESS                    -38
#define DRMP3_BAD_PROTOCOL                  -39
#define DRMP3_PROTOCOL_UNAVAILABLE          -40
#define DRMP3_PROTOCOL_NOT_SUPPORTED        -41
#define DRMP3_PROTOCOL_FAMILY_NOT_SUPPORTED -42
#define DRMP3_ADDRESS_FAMILY_NOT_SUPPORTED  -43
#define DRMP3_SOCKET_NOT_SUPPORTED          -44
#define DRMP3_CONNECTION_RESET              -45
#define DRMP3_ALREADY_CONNECTED             -46
#define DRMP3_NOT_CONNECTED                 -47
#define DRMP3_CONNECTION_REFUSED            -48
#define DRMP3_NO_HOST                       -49
#define DRMP3_IN_PROGRESS                   -50
#define DRMP3_CANCELLED                     -51
#define DRMP3_MEMORY_ALREADY_MAPPED         -52
#define DRMP3_AT_END                        -53
/* End Result Codes */

#define DRMP3_MAX_PCM_FRAMES_PER_MP3_FRAME  1152
#define DRMP3_MAX_SAMPLES_PER_FRAME         (DRMP3_MAX_PCM_FRAMES_PER_MP3_FRAME*2)

/* Inline */
#ifdef _MSC_VER
    #define DRMP3_INLINE __forceinline
#elif defined(__GNUC__)
    /*
    I've had a bug report where GCC is emitting warnings about functions possibly not being inlineable. This warning happens when
    the __attribute__((always_inline)) attribute is defined without an "inline" statement. I think therefore there must be some
    case where "__inline__" is not always defined, thus the compiler emitting these warnings. When using -std=c89 or -ansi on the
    command line, we cannot use the "inline" keyword and instead need to use "__inline__". In an attempt to work around this issue
    I am using "__inline__" only when we're compiling in strict ANSI mode.
    */
    #if defined(__STRICT_ANSI__)
        #define DRMP3_GNUC_INLINE_HINT __inline__
    #else
        #define DRMP3_GNUC_INLINE_HINT inline
    #endif

    #if (__GNUC__ > 3 || (__GNUC__ == 3 && __GNUC_MINOR__ >= 2)) || defined(__clang__)
        #define DRMP3_INLINE DRMP3_GNUC_INLINE_HINT __attribute__((always_inline))
    #else
        #define DRMP3_INLINE DRMP3_GNUC_INLINE_HINT
    #endif
#elif defined(__WATCOMC__)
    #define DRMP3_INLINE __inline
#else
    #define DRMP3_INLINE
#endif
/* End Inline */


DRMP3_API void drmp3_version(drmp3_uint32* pMajor, drmp3_uint32* pMinor, drmp3_uint32* pRevision);
DRMP3_API const char* drmp3_version_string(void);


/* Allocation Callbacks */
typedef struct
{
    void* pUserData;
    void* (* onMalloc)(size_t sz, void* pUserData);
    void* (* onRealloc)(void* p, size_t sz, void* pUserData);
    void  (* onFree)(void* p, void* pUserData);
} drmp3_allocation_callbacks;
/* End Allocation Callbacks */


/*
Low Level Push API
==================
*/
#define DRMP3_MAX_BITRESERVOIR_BYTES      511
#define DRMP3_MAX_FREE_FORMAT_FRAME_SIZE  2304    /* more than ISO spec's */
#define DRMP3_MAX_L3_FRAME_PAYLOAD_BYTES  DRMP3_MAX_FREE_FORMAT_FRAME_SIZE /* MUST be >= 320000/8/32000*1152 = 1440 */

typedef struct
{
    int frame_bytes, channels, sample_rate, layer, bitrate_kbps;
} drmp3dec_frame_info;

typedef struct
{
    const drmp3_uint8 *buf;
    int pos, limit;
} drmp3_bs;

typedef struct
{
    const drmp3_uint8 *sfbtab;
    drmp3_uint16 part_23_length, big_values, scalefac_compress;
    drmp3_uint8 global_gain, block_type, mixed_block_flag, n_long_sfb, n_short_sfb;
    drmp3_uint8 table_select[3], region_count[3], subblock_gain[3];
    drmp3_uint8 preflag, scalefac_scale, count1_table, scfsi;
} drmp3_L3_gr_info;

typedef struct
{
    drmp3_bs bs;
    drmp3_uint8 maindata[DRMP3_MAX_BITRESERVOIR_BYTES + DRMP3_MAX_L3_FRAME_PAYLOAD_BYTES];
    drmp3_L3_gr_info gr_info[4];
    float grbuf[2][576], scf[40], syn[18 + 15][2*32];
    drmp3_uint8 ist_pos[2][39];
} drmp3dec_scratch;

typedef struct
{
    float mdct_overlap[2][9*32], qmf_state[15*2*32];
    int reserv, free_format_bytes;
    drmp3_uint8 header[4], reserv_buf[511];
    drmp3dec_scratch scratch;
} drmp3dec;

/* Initializes a low level decoder. */
DRMP3_API void drmp3dec_init(drmp3dec *dec);

/* Reads a frame from a low level decoder. */
DRMP3_API int drmp3dec_decode_frame(drmp3dec *dec, const drmp3_uint8 *mp3, int mp3_bytes, void *pcm, drmp3dec_frame_info *info);

/* Helper for converting between f32 and s16. */
DRMP3_API void drmp3dec_f32_to_s16(const float *in, drmp3_int16 *out, size_t num_samples);



/*
Main API (Pull API)
===================
*/
typedef enum
{
    DRMP3_SEEK_SET,
    DRMP3_SEEK_CUR,
    DRMP3_SEEK_END
} drmp3_seek_origin;

typedef struct
{
    drmp3_uint64 seekPosInBytes;        /* Points to the first byte of an MP3 frame. */
    drmp3_uint64 pcmFrameIndex;         /* The index of the PCM frame this seek point targets. */
    drmp3_uint16 mp3FramesToDiscard;    /* The number of whole MP3 frames to be discarded before pcmFramesToDiscard. */
    drmp3_uint16 pcmFramesToDiscard;    /* The number of leading samples to read and discard. These are discarded after mp3FramesToDiscard. */
} drmp3_seek_point;

typedef enum
{
    DRMP3_METADATA_TYPE_ID3V1,
    DRMP3_METADATA_TYPE_ID3V2,
    DRMP3_METADATA_TYPE_APE,
    DRMP3_METADATA_TYPE_XING,
    DRMP3_METADATA_TYPE_VBRI
} drmp3_metadata_type;

typedef struct
{
    drmp3_metadata_type type;
    const void* pRawData;               /* A pointer to the raw data. */
    size_t rawDataSize;
} drmp3_metadata;


/*
Callback for when data is read. Return value is the number of bytes actually read.

pUserData   [in]  The user data that was passed to drmp3_init(), and family.
pBufferOut  [out] The output buffer.
bytesToRead [in]  The number of bytes to read.

Returns the number of bytes actually read.

A return value of less than bytesToRead indicates the end of the stream. Do _not_ return from this callback until
either the entire bytesToRead is filled or you have reached the end of the stream.
*/
typedef size_t (* drmp3_read_proc)(void* pUserData, void* pBufferOut, size_t bytesToRead);

/*
Callback for when data needs to be seeked.

pUserData [in] The user data that was passed to drmp3_init(), and family.
offset    [in] The number of bytes to move, relative to the origin. Can be negative.
origin    [in] The origin of the seek.

Returns whether or not the seek was successful.
*/
typedef drmp3_bool32 (* drmp3_seek_proc)(void* pUserData, int offset, drmp3_seek_origin origin);

/*
Callback for retrieving the current cursor position.

pUserData [in]  The user data that was passed to drmp3_init(), and family.
pCursor   [out] The cursor position in bytes from the start of the stream.

Returns whether or not the cursor position was successfully retrieved.
*/
typedef drmp3_bool32 (* drmp3_tell_proc)(void* pUserData, drmp3_int64* pCursor);


/*
Callback for when metadata is read.

Only the raw data is provided. The client is responsible for parsing the contents of the data themsevles.
*/
typedef void (* drmp3_meta_proc)(void* pUserData, const drmp3_metadata* pMetadata);


typedef struct
{
    drmp3_uint32 channels;
    drmp3_uint32 sampleRate;
} drmp3_config;

typedef struct
{
    drmp3dec decoder;
    drmp3_uint32 channels;
    drmp3_uint32 sampleRate;
    drmp3_read_proc onRead;
    drmp3_seek_proc onSeek;
    drmp3_meta_proc onMeta;
    void* pUserData;
    void* pUserDataMeta;
    drmp3_allocation_callbacks allocationCallbacks;
    drmp3_uint32 mp3FrameChannels;      /* The number of channels in the currently loaded MP3 frame. Internal use only. */
    drmp3_uint32 mp3FrameSampleRate;    /* The sample rate of the currently loaded MP3 frame. Internal use only. */
    drmp3_uint32 pcmFramesConsumedInMP3Frame;
    drmp3_uint32 pcmFramesRemainingInMP3Frame;
    drmp3_uint8 pcmFrames[sizeof(float)*DRMP3_MAX_SAMPLES_PER_FRAME];  /* <-- Multipled by sizeof(float) to ensure there's enough room for DR_MP3_FLOAT_OUTPUT. */
    drmp3_uint64 currentPCMFrame;       /* The current PCM frame, globally. */
    drmp3_uint64 streamCursor;          /* The current byte the decoder is sitting on in the raw stream. */
    drmp3_uint64 streamLength;          /* The length of the stream in bytes. dr_mp3 will not read beyond this. If a ID3v1 or APE tag is present, this will be set to the first byte of the tag. */
    drmp3_uint64 streamStartOffset;     /* The offset of the start of the MP3 data. This is used for skipping ID3v2 and VBR tags. */
    drmp3_seek_point* pSeekPoints;      /* NULL by default. Set with drmp3_bind_seek_table(). Memory is owned by the client. dr_mp3 will never attempt to free this pointer. */
    drmp3_uint32 seekPointCount;        /* The number of items in pSeekPoints. When set to 0 assumes to no seek table. Defaults to zero. */
    drmp3_uint32 delayInPCMFrames;
    drmp3_uint32 paddingInPCMFrames;
    drmp3_uint64 totalPCMFrameCount;    /* Set to DRMP3_UINT64_MAX if the length is unknown. Includes delay and padding. */
    drmp3_bool32 isVBR;
    drmp3_bool32 isCBR;
    size_t dataSize;
    size_t dataCapacity;
    size_t dataConsumed;
    drmp3_uint8* pData;
    drmp3_bool32 atEnd;
    struct
    {
        const drmp3_uint8* pData;
        size_t dataSize;
        size_t currentReadPos;
    } memory;   /* Only used for decoders that were opened against a block of memory. */
} drmp3;

/*
Initializes an MP3 decoder.

onRead    [in]           The function to call when data needs to be read from the client.
onSeek    [in]           The function to call when the read position of the client data needs to move.
onTell    [in]           The function to call when the read position of the client data needs to be retrieved.
pUserData [in, optional] A pointer to application defined data that will be passed to onRead and onSeek.

Returns true if successful; false otherwise.

Close the loader with drmp3_uninit().

See also: drmp3_init_file(), drmp3_init_memory(), drmp3_uninit()
*/
DRMP3_API drmp3_bool32 drmp3_init(drmp3* pMP3, drmp3_read_proc onRead, drmp3_seek_proc onSeek, drmp3_tell_proc onTell, drmp3_meta_proc onMeta, void* pUserData, const drmp3_allocation_callbacks* pAllocationCallbacks);

/*
Initializes an MP3 decoder from a block of memory.

This does not create a copy of the data. It is up to the application to ensure the buffer remains valid for
the lifetime of the drmp3 object.

The buffer should contain the contents of the entire MP3 file.
*/
DRMP3_API drmp3_bool32 drmp3_init_memory_with_metadata(drmp3* pMP3, const void* pData, size_t dataSize, drmp3_meta_proc onMeta, void* pUserDataMeta, const drmp3_allocation_callbacks* pAllocationCallbacks);
DRMP3_API drmp3_bool32 drmp3_init_memory(drmp3* pMP3, const void* pData, size_t dataSize, const drmp3_allocation_callbacks* pAllocationCallbacks);

#ifndef DR_MP3_NO_STDIO
/*
Initializes an MP3 decoder from a file.

This holds the internal FILE object until drmp3_uninit() is called. Keep this in mind if you're caching drmp3
objects because the operating system may restrict the number of file handles an application can have open at
any given time.
*/
DRMP3_API drmp3_bool32 drmp3_init_file_with_metadata(drmp3* pMP3, const char* pFilePath, drmp3_meta_proc onMeta, void* pUserDataMeta, const drmp3_allocation_callbacks* pAllocationCallbacks);
DRMP3_API drmp3_bool32 drmp3_init_file_with_metadata_w(drmp3* pMP3, const wchar_t* pFilePath, drmp3_meta_proc onMeta, void* pUserDataMeta, const drmp3_allocation_callbacks* pAllocationCallbacks);

DRMP3_API drmp3_bool32 drmp3_init_file(drmp3* pMP3, const char* pFilePath, const drmp3_allocation_callbacks* pAllocationCallbacks);
DRMP3_API drmp3_bool32 drmp3_init_file_w(drmp3* pMP3, const wchar_t* pFilePath, const drmp3_allocation_callbacks* pAllocationCallbacks);
#endif

/*
Uninitializes an MP3 decoder.
*/
DRMP3_API void drmp3_uninit(drmp3* pMP3);

/*
Reads PCM frames as interleaved 32-bit IEEE floating point PCM.

Note that framesToRead specifies the number of PCM frames to read, _not_ the number of MP3 frames.
*/
DRMP3_API drmp3_uint64 drmp3_read_pcm_frames_f32(drmp3* pMP3, drmp3_uint64 framesToRead, float* pBufferOut);

/*
Reads PCM frames as interleaved signed 16-bit integer PCM.

Note that framesToRead specifies the number of PCM frames to read, _not_ the number of MP3 frames.
*/
DRMP3_API drmp3_uint64 drmp3_read_pcm_frames_s16(drmp3* pMP3, drmp3_uint64 framesToRead, drmp3_int16* pBufferOut);

/*
Seeks to a specific frame.

Note that this is _not_ an MP3 frame, but rather a PCM frame.
*/
DRMP3_API drmp3_bool32 drmp3_seek_to_pcm_frame(drmp3* pMP3, drmp3_uint64 frameIndex);

/*
Calculates the total number of PCM frames in the MP3 stream. Cannot be used for infinite streams such as internet
radio. Runs in linear time. Returns 0 on error.
*/
DRMP3_API drmp3_uint64 drmp3_get_pcm_frame_count(drmp3* pMP3);

/*
Calculates the total number of MP3 frames in the MP3 stream. Cannot be used for infinite streams such as internet
radio. Runs in linear time. Returns 0 on error.
*/
DRMP3_API drmp3_uint64 drmp3_get_mp3_frame_count(drmp3* pMP3);

/*
Calculates the total number of MP3 and PCM frames in the MP3 stream. Cannot be used for infinite streams such as internet
radio. Runs in linear time. Returns 0 on error.

This is equivalent to calling drmp3_get_mp3_frame_count() and drmp3_get_pcm_frame_count() except that it's more efficient.
*/
DRMP3_API drmp3_bool32 drmp3_get_mp3_and_pcm_frame_count(drmp3* pMP3, drmp3_uint64* pMP3FrameCount, drmp3_uint64* pPCMFrameCount);

/*
Calculates the seekpoints based on PCM frames. This is slow.

pSeekpoint count is a pointer to a uint32 containing the seekpoint count. On input it contains the desired count.
On output it contains the actual count. The reason for this design is that the client may request too many
seekpoints, in which case dr_mp3 will return a corrected count.

Note that seektable seeking is not quite sample exact when the MP3 stream contains inconsistent sample rates.
*/
DRMP3_API drmp3_bool32 drmp3_calculate_seek_points(drmp3* pMP3, drmp3_uint32* pSeekPointCount, drmp3_seek_point* pSeekPoints);

/*
Binds a seek table to the decoder.

This does _not_ make a copy of pSeekPoints - it only references it. It is up to the application to ensure this
remains valid while it is bound to the decoder.

Use drmp3_calculate_seek_points() to calculate the seek points.
*/
DRMP3_API drmp3_bool32 drmp3_bind_seek_table(drmp3* pMP3, drmp3_uint32 seekPointCount, drmp3_seek_point* pSeekPoints);


/*
Opens an decodes an entire MP3 stream as a single operation.

On output pConfig will receive the channel count and sample rate of the stream.

Free the returned pointer with drmp3_free().
*/
DRMP3_API float* drmp3_open_and_read_pcm_frames_f32(drmp3_read_proc onRead, drmp3_seek_proc onSeek, drmp3_tell_proc onTell, void* pUserData, drmp3_config* pConfig, drmp3_uint64* pTotalFrameCount, const drmp3_allocation_callbacks* pAllocationCallbacks);
DRMP3_API drmp3_int16* drmp3_open_and_read_pcm_frames_s16(drmp3_read_proc onRead, drmp3_seek_proc onSeek, drmp3_tell_proc onTell, void* pUserData, drmp3_config* pConfig, drmp3_uint64* pTotalFrameCount, const drmp3_allocation_callbacks* pAllocationCallbacks);

DRMP3_API float* drmp3_open_memory_and_read_pcm_frames_f32(const void* pData, size_t dataSize, drmp3_config* pConfig, drmp3_uint64* pTotalFrameCount, const drmp3_allocation_callbacks* pAllocationCallbacks);
DRMP3_API drmp3_int16* drmp3_open_memory_and_read_pcm_frames_s16(const void* pData, size_t dataSize, drmp3_config* pConfig, drmp3_uint64* pTotalFrameCount, const drmp3_allocation_callbacks* pAllocationCallbacks);

#ifndef DR_MP3_NO_STDIO
DRMP3_API float* drmp3_open_file_and_read_pcm_frames_f32(const char* filePath, drmp3_config* pConfig, drmp3_uint64* pTotalFrameCount, const drmp3_allocation_callbacks* pAllocationCallbacks);
DRMP3_API drmp3_int16* drmp3_open_file_and_read_pcm_frames_s16(const char* filePath, drmp3_config* pConfig, drmp3_uint64* pTotalFrameCount, const drmp3_allocation_callbacks* pAllocationCallbacks);
#endif

/*
Allocates a block of memory on the heap.
*/
DRMP3_API void* drmp3_malloc(size_t sz, const drmp3_allocation_callbacks* pAllocationCallbacks);

/*
Frees any memory that was allocated by a public drmp3 API.
*/
DRMP3_API void drmp3_free(void* p, const drmp3_allocation_callbacks* pAllocationCallbacks);

#ifdef __cplusplus
}
#endif
#endif  /* dr_mp3_h */


/************************************************************************************************************************************************************
 ************************************************************************************************************************************************************

 IMPLEMENTATION

 ************************************************************************************************************************************************************
 ************************************************************************************************************************************************************/
#if defined(DR_MP3_IMPLEMENTATION)
#ifndef dr_mp3_c
#define dr_mp3_c

#include <stdlib.h>
#include <string.h>
#include <limits.h> /* For INT_MAX */

DRMP3_API void drmp3_version(drmp3_uint32* pMajor, drmp3_uint32* pMinor, drmp3_uint32* pRevision)
{
    if (pMajor) {
        *pMajor = DRMP3_VERSION_MAJOR;
    }

    if (pMinor) {
        *pMinor = DRMP3_VERSION_MINOR;
    }

    if (pRevision) {
        *pRevision = DRMP3_VERSION_REVISION;
    }
}

DRMP3_API const char* drmp3_version_string(void)
{
    return DRMP3_VERSION_STRING;
}

/* Disable SIMD when compiling with TCC for now. */
#if defined(__TINYC__)
#define DR_MP3_NO_SIMD
#endif

#define DRMP3_OFFSET_PTR(p, offset) ((void*)((drmp3_uint8*)(p) + (offset)))

#ifndef DRMP3_MAX_FRAME_SYNC_MATCHES
#define DRMP3_MAX_FRAME_SYNC_MATCHES      10
#endif

#define DRMP3_SHORT_BLOCK_TYPE            2
#define DRMP3_STOP_BLOCK_TYPE             3
#define DRMP3_MODE_MONO                   3
#define DRMP3_MODE_JOINT_STEREO           1
#define DRMP3_HDR_SIZE                    4
#define DRMP3_HDR_IS_MONO(h)              (((h[3]) & 0xC0) == 0xC0)
#define DRMP3_HDR_IS_MS_STEREO(h)         (((h[3]) & 0xE0) == 0x60)
#define DRMP3_HDR_IS_FREE_FORMAT(h)       (((h[2]) & 0xF0) == 0)
#define DRMP3_HDR_IS_CRC(h)               (!((h[1]) & 1))
#define DRMP3_HDR_TEST_PADDING(h)         ((h[2]) & 0x2)
#define DRMP3_HDR_TEST_MPEG1(h)           ((h[1]) & 0x8)
#define DRMP3_HDR_TEST_NOT_MPEG25(h)      ((h[1]) & 0x10)
#define DRMP3_HDR_TEST_I_STEREO(h)        ((h[3]) & 0x10)
#define DRMP3_HDR_TEST_MS_STEREO(h)       ((h[3]) & 0x20)
#define DRMP3_HDR_GET_STEREO_MODE(h)      (((h[3]) >> 6) & 3)
#define DRMP3_HDR_GET_STEREO_MODE_EXT(h)  (((h[3]) >> 4) & 3)
#define DRMP3_HDR_GET_LAYER(h)            (((h[1]) >> 1) & 3)
#define DRMP3_HDR_GET_BITRATE(h)          ((h[2]) >> 4)
#define DRMP3_HDR_GET_SAMPLE_RATE(h)      (((h[2]) >> 2) & 3)
#define DRMP3_HDR_GET_MY_SAMPLE_RATE(h)   (DRMP3_HDR_GET_SAMPLE_RATE(h) + (((h[1] >> 3) & 1) + ((h[1] >> 4) & 1))*3)
#define DRMP3_HDR_IS_FRAME_576(h)         ((h[1] & 14) == 2)
#define DRMP3_HDR_IS_LAYER_1(h)           ((h[1] & 6) == 6)

#define DRMP3_BITS_DEQUANTIZER_OUT        -1
#define DRMP3_MAX_SCF                     (255 + DRMP3_BITS_DEQUANTIZER_OUT*4 - 210)
#define DRMP3_MAX_SCFI                    ((DRMP3_MAX_SCF + 3) & ~3)

#define DRMP3_MIN(a, b)           ((a) > (b) ? (b) : (a))
#define DRMP3_MAX(a, b)           ((a) < (b) ? (b) : (a))

#if !defined(DR_MP3_NO_SIMD)

#if !defined(DR_MP3_ONLY_SIMD) && (defined(_M_X64) || defined(__x86_64__) || defined(__aarch64__) || defined(_M_ARM64) || defined(_M_ARM64EC))
/* x64 always have SSE2, arm64 always have neon, no need for generic code */
#define DR_MP3_ONLY_SIMD
#endif

#if ((defined(_MSC_VER) && _MSC_VER >= 1400) && defined(_M_X64)) || ((defined(__i386) || defined(_M_IX86) || defined(__i386__) || defined(__x86_64__)) && ((defined(_M_IX86_FP) && _M_IX86_FP == 2) || defined(__SSE2__)))
#if defined(_MSC_VER)
#include <intrin.h>
#endif
#include <emmintrin.h>
#define DRMP3_HAVE_SSE 1
#define DRMP3_HAVE_SIMD 1
#define DRMP3_VSTORE _mm_storeu_ps
#define DRMP3_VLD _mm_loadu_ps
#define DRMP3_VSET _mm_set1_ps
#define DRMP3_VADD _mm_add_ps
#define DRMP3_VSUB _mm_sub_ps
#define DRMP3_VMUL _mm_mul_ps
#define DRMP3_VMAC(a, x, y) _mm_add_ps(a, _mm_mul_ps(x, y))
#define DRMP3_VMSB(a, x, y) _mm_sub_ps(a, _mm_mul_ps(x, y))
#define DRMP3_VMUL_S(x, s)  _mm_mul_ps(x, _mm_set1_ps(s))
#define DRMP3_VREV(x) _mm_shuffle_ps(x, x, _MM_SHUFFLE(0, 1, 2, 3))
typedef __m128 drmp3_f4;
#if (defined(_MSC_VER) || defined(DR_MP3_ONLY_SIMD)) && !defined(__clang__)
#define drmp3_cpuid __cpuid
#else
static __inline__ __attribute__((always_inline)) void drmp3_cpuid(int CPUInfo[], const int InfoType)
{
#if defined(__PIC__)
    __asm__ __volatile__(
#if defined(__x86_64__)
        "push %%rbx\n"
        "cpuid\n"
        "xchgl %%ebx, %1\n"
        "pop  %%rbx\n"
#else
        "xchgl %%ebx, %1\n"
        "cpuid\n"
        "xchgl %%ebx, %1\n"
#endif
        : "=a" (CPUInfo[0]), "=r" (CPUInfo[1]), "=c" (CPUInfo[2]), "=d" (CPUInfo[3])
        : "a" (InfoType));
#else
    __asm__ __volatile__(
        "cpuid"
        : "=a" (CPUInfo[0]), "=b" (CPUInfo[1]), "=c" (CPUInfo[2]), "=d" (CPUInfo[3])
        : "a" (InfoType));
#endif
}
#endif
static int drmp3_have_simd(void)
{
#ifdef DR_MP3_ONLY_SIMD
    return 1;
#else
    static int g_have_simd;
    int CPUInfo[4];
#ifdef MINIMP3_TEST
    static int g_counter;
    if (g_counter++ > 100)
        return 0;
#endif
    if (g_have_simd)
        goto end;
    drmp3_cpuid(CPUInfo, 0);
    if (CPUInfo[0] > 0)
    {
        drmp3_cpuid(CPUInfo, 1);
        g_have_simd = (CPUInfo[3] & (1 << 26)) + 1; /* SSE2 */
        return g_have_simd - 1;
    }

end:
    return g_have_simd - 1;
#endif
}
#elif defined(__ARM_NEON) || defined(__aarch64__) || defined(_M_ARM64) || defined(_M_ARM64EC)
#include <arm_neon.h>
#define DRMP3_HAVE_SSE 0
#define DRMP3_HAVE_SIMD 1
#define DRMP3_VSTORE vst1q_f32
#define DRMP3_VLD vld1q_f32
#define DRMP3_VSET vmovq_n_f32
#define DRMP3_VADD vaddq_f32
#define DRMP3_VSUB vsubq_f32
#define DRMP3_VMUL vmulq_f32
#define DRMP3_VMAC(a, x, y) vmlaq_f32(a, x, y)
#define DRMP3_VMSB(a, x, y) vmlsq_f32(a, x, y)
#define DRMP3_VMUL_S(x, s)  vmulq_f32(x, vmovq_n_f32(s))
#define DRMP3_VREV(x) vcombine_f32(vget_high_f32(vrev64q_f32(x)), vget_low_f32(vrev64q_f32(x)))
typedef float32x4_t drmp3_f4;
static int drmp3_have_simd(void)
{   /* TODO: detect neon for !DR_MP3_ONLY_SIMD */
    return 1;
}
#else
#define DRMP3_HAVE_SSE 0
#define DRMP3_HAVE_SIMD 0
#ifdef DR_MP3_ONLY_SIMD
#error DR_MP3_ONLY_SIMD used, but SSE/NEON not enabled
#endif
#endif

#else

#define DRMP3_HAVE_SIMD 0

#endif

#if defined(__ARM_ARCH) && (__ARM_ARCH >= 6) && !defined(__aarch64__) && !defined(_M_ARM64) && !defined(_M_ARM64EC) && !defined(__ARM_ARCH_6M__)
#define DRMP3_HAVE_ARMV6 1
static __inline__ __attribute__((always_inline)) drmp3_int32 drmp3_clip_int16_arm(drmp3_int32 a)
{
    drmp3_int32 x = 0;
    __asm__ ("ssat %0, #16, %1" : "=r"(x) : "r"(a));
    return x;
}
#else
#define DRMP3_HAVE_ARMV6 0
#endif


/* Standard library stuff. */
#ifndef DRMP3_ASSERT
#include <assert.h>
#define DRMP3_ASSERT(expression) assert(expression)
#endif
#ifndef DRMP3_COPY_MEMORY
#define DRMP3_COPY_MEMORY(dst, src, sz) memcpy((dst), (src), (sz))
#endif
#ifndef DRMP3_MOVE_MEMORY
#define DRMP3_MOVE_MEMORY(dst, src, sz) memmove((dst), (src), (sz))
#endif
#ifndef DRMP3_ZERO_MEMORY
#define DRMP3_ZERO_MEMORY(p, sz) memset((p), 0, (sz))
#endif
#define DRMP3_ZERO_OBJECT(p) DRMP3_ZERO_MEMORY((p), sizeof(*(p)))
#ifndef DRMP3_MALLOC
#define DRMP3_MALLOC(sz) malloc((sz))
#endif
#ifndef DRMP3_REALLOC
#define DRMP3_REALLOC(p, sz) realloc((p), (sz))
#endif
#ifndef DRMP3_FREE
#define DRMP3_FREE(p) free((p))
#endif



typedef struct
{
    float scf[3*64];
    drmp3_uint8 total_bands, stereo_bands, bitalloc[64], scfcod[64];
} drmp3_L12_scale_info;

typedef struct
{
    drmp3_uint8 tab_offset, code_tab_width, band_count;
} drmp3_L12_subband_alloc;

static void drmp3_bs_init(drmp3_bs *bs, const drmp3_uint8 *data, int bytes)
{
    bs->buf   = data;
    bs->pos   = 0;
    bs->limit = bytes*8;
}

static drmp3_uint32 drmp3_bs_get_bits(drmp3_bs *bs, int n)
{
    drmp3_uint32 next, cache = 0, s = bs->pos & 7;
    int shl = n + s;
    const drmp3_uint8 *p = bs->buf + (bs->pos >> 3);
    if ((bs->pos += n) > bs->limit)
        return 0;
    next = *p++ & (255 >> s);
    while ((shl -= 8) > 0)
    {
        cache |= next << shl;
        next = *p++;
    }
    return cache | (next >> -shl);
}

static int drmp3_hdr_valid(const drmp3_uint8 *h)
{
    return h[0] == 0xff &&
        ((h[1] & 0xF0) == 0xf0 || (h[1] & 0xFE) == 0xe2) &&
        (DRMP3_HDR_GET_LAYER(h) != 0) &&
        (DRMP3_HDR_GET_BITRATE(h) != 15) &&
        (DRMP3_HDR_GET_SAMPLE_RATE(h) != 3);
}

static int drmp3_hdr_compare(const drmp3_uint8 *h1, const drmp3_uint8 *h2)
{
    return drmp3_hdr_valid(h2) &&
        ((h1[1] ^ h2[1]) & 0xFE) == 0 &&
        ((h1[2] ^ h2[2]) & 0x0C) == 0 &&
        !(DRMP3_HDR_IS_FREE_FORMAT(h1) ^ DRMP3_HDR_IS_FREE_FORMAT(h2));
}

static unsigned drmp3_hdr_bitrate_kbps(const drmp3_uint8 *h)
{
    static const drmp3_uint8 halfrate[2][3][15] = {
        { { 0,4,8,12,16,20,24,28,32,40,48,56,64,72,80 }, { 0,4,8,12,16,20,24,28,32,40,48,56,64,72,80 }, { 0,16,24,28,32,40,48,56,64,72,80,88,96,112,128 } },
        { { 0,16,20,24,28,32,40,48,56,64,80,96,112,128,160 }, { 0,16,24,28,32,40,48,56,64,80,96,112,128,160,192 }, { 0,16,32,48,64,80,96,112,128,144,160,176,192,208,224 } },
    };
    return 2*halfrate[!!DRMP3_HDR_TEST_MPEG1(h)][DRMP3_HDR_GET_LAYER(h) - 1][DRMP3_HDR_GET_BITRATE(h)];
}

static unsigned drmp3_hdr_sample_rate_hz(const drmp3_uint8 *h)
{
    static const unsigned g_hz[3] = { 44100, 48000, 32000 };
    return g_hz[DRMP3_HDR_GET_SAMPLE_RATE(h)] >> (int)!DRMP3_HDR_TEST_MPEG1(h) >> (int)!DRMP3_HDR_TEST_NOT_MPEG25(h);
}

static unsigned drmp3_hdr_frame_samples(const drmp3_uint8 *h)
{
    return DRMP3_HDR_IS_LAYER_1(h) ? 384 : (1152 >> (int)DRMP3_HDR_IS_FRAME_576(h));
}

static int drmp3_hdr_frame_bytes(const drmp3_uint8 *h, int free_format_size)
{
    int frame_bytes = drmp3_hdr_frame_samples(h)*drmp3_hdr_bitrate_kbps(h)*125/drmp3_hdr_sample_rate_hz(h);
    if (DRMP3_HDR_IS_LAYER_1(h))
    {
        frame_bytes &= ~3; /* slot align */
    }
    return frame_bytes ? frame_bytes : free_format_size;
}

static int drmp3_hdr_padding(const drmp3_uint8 *h)
{
    return DRMP3_HDR_TEST_PADDING(h) ? (DRMP3_HDR_IS_LAYER_1(h) ? 4 : 1) : 0;
}

#ifndef DR_MP3_ONLY_MP3
static const drmp3_L12_subband_alloc *drmp3_L12_subband_alloc_table(const drmp3_uint8 *hdr, drmp3_L12_scale_info *sci)
{
    const drmp3_L12_subband_alloc *alloc;
    int mode = DRMP3_HDR_GET_STEREO_MODE(hdr);
    int nbands, stereo_bands = (mode == DRMP3_MODE_MONO) ? 0 : (mode == DRMP3_MODE_JOINT_STEREO) ? (DRMP3_HDR_GET_STEREO_MODE_EXT(hdr) << 2) + 4 : 32;

    if (DRMP3_HDR_IS_LAYER_1(hdr))
    {
        static const drmp3_L12_subband_alloc g_alloc_L1[] = { { 76, 4, 32 } };
        alloc = g_alloc_L1;
        nbands = 32;
    } else if (!DRMP3_HDR_TEST_MPEG1(hdr))
    {
        static const drmp3_L12_subband_alloc g_alloc_L2M2[] = { { 60, 4, 4 }, { 44, 3, 7 }, { 44, 2, 19 } };
        alloc = g_alloc_L2M2;
        nbands = 30;
    } else
    {
        static const drmp3_L12_subband_alloc g_alloc_L2M1[] = { { 0, 4, 3 }, { 16, 4, 8 }, { 32, 3, 12 }, { 40, 2, 7 } };
        int sample_rate_idx = DRMP3_HDR_GET_SAMPLE_RATE(hdr);
        unsigned kbps = drmp3_hdr_bitrate_kbps(hdr) >> (int)(mode != DRMP3_MODE_MONO);
        if (!kbps) /* free-format */
        {
            kbps = 192;
        }

        alloc = g_alloc_L2M1;
        nbands = 27;
        if (kbps < 56)
        {
            static const drmp3_L12_subband_alloc g_alloc_L2M1_lowrate[] = { { 44, 4, 2 }, { 44, 3, 10 } };
            alloc = g_alloc_L2M1_lowrate;
            nbands = sample_rate_idx == 2 ? 12 : 8;
        } else if (kbps >= 96 && sample_rate_idx != 1)
        {
            nbands = 30;
        }
    }

    sci->total_bands = (drmp3_uint8)nbands;
    sci->stereo_bands = (drmp3_uint8)DRMP3_MIN(stereo_bands, nbands);

    return alloc;
}

static void drmp3_L12_read_scalefactors(drmp3_bs *bs, drmp3_uint8 *pba, drmp3_uint8 *scfcod, int bands, float *scf)
{
    static const float g_deq_L12[18*3] = {
#define DRMP3_DQ(x) 9.53674316e-07f/x, 7.56931807e-07f/x, 6.00777173e-07f/x
        DRMP3_DQ(3),DRMP3_DQ(7),DRMP3_DQ(15),DRMP3_DQ(31),DRMP3_DQ(63),DRMP3_DQ(127),DRMP3_DQ(255),DRMP3_DQ(511),DRMP3_DQ(1023),DRMP3_DQ(2047),DRMP3_DQ(4095),DRMP3_DQ(8191),DRMP3_DQ(16383),DRMP3_DQ(32767),DRMP3_DQ(65535),DRMP3_DQ(3),DRMP3_DQ(5),DRMP3_DQ(9)
    };
    int i, m;
    for (i = 0; i < bands; i++)
    {
        float s = 0;
        int ba = *pba++;
        int mask = ba ? 4 + ((19 >> scfcod[i]) & 3) : 0;
        for (m = 4; m; m >>= 1)
        {
            if (mask & m)
            {
                int b = drmp3_bs_get_bits(bs, 6);
                s = g_deq_L12[ba*3 - 6 + b % 3]*(int)(1 << 21 >> b/3);
            }
            *scf++ = s;
        }
    }
}

static void drmp3_L12_read_scale_info(const drmp3_uint8 *hdr, drmp3_bs *bs, drmp3_L12_scale_info *sci)
{
    static const drmp3_uint8 g_bitalloc_code_tab[] = {
        0,17, 3, 4, 5,6,7, 8,9,10,11,12,13,14,15,16,
        0,17,18, 3,19,4,5, 6,7, 8, 9,10,11,12,13,16,
        0,17,18, 3,19,4,5,16,
        0,17,18,16,
        0,17,18,19, 4,5,6, 7,8, 9,10,11,12,13,14,15,
        0,17,18, 3,19,4,5, 6,7, 8, 9,10,11,12,13,14,
        0, 2, 3, 4, 5,6,7, 8,9,10,11,12,13,14,15,16
    };
    const drmp3_L12_subband_alloc *subband_alloc = drmp3_L12_subband_alloc_table(hdr, sci);

    int i, k = 0, ba_bits = 0;
    const drmp3_uint8 *ba_code_tab = g_bitalloc_code_tab;

    for (i = 0; i < sci->total_bands; i++)
    {
        drmp3_uint8 ba;
        if (i == k)
        {
            k += subband_alloc->band_count;
            ba_bits = subband_alloc->code_tab_width;
            ba_code_tab = g_bitalloc_code_tab + subband_alloc->tab_offset;
            subband_alloc++;
        }
        ba = ba_code_tab[drmp3_bs_get_bits(bs, ba_bits)];
        sci->bitalloc[2*i] = ba;
        if (i < sci->stereo_bands)
        {
            ba = ba_code_tab[drmp3_bs_get_bits(bs, ba_bits)];
        }
        sci->bitalloc[2*i + 1] = sci->stereo_bands ? ba : 0;
    }

    for (i = 0; i < 2*sci->total_bands; i++)
    {
        sci->scfcod[i] = (drmp3_uint8)(sci->bitalloc[i] ? DRMP3_HDR_IS_LAYER_1(hdr) ? 2 : drmp3_bs_get_bits(bs, 2) : 6);
    }

    drmp3_L12_read_scalefactors(bs, sci->bitalloc, sci->scfcod, sci->total_bands*2, sci->scf);

    for (i = sci->stereo_bands; i < sci->total_bands; i++)
    {
        sci->bitalloc[2*i + 1] = 0;
    }
}

static int drmp3_L12_dequantize_granule(float *grbuf, drmp3_bs *bs, drmp3_L12_scale_info *sci, int group_size)
{
    int i, j, k, choff = 576;
    for (j = 0; j < 4; j++)
    {
        float *dst = grbuf + group_size*j;
        for (i = 0; i < 2*sci->total_bands; i++)
        {
            int ba = sci->bitalloc[i];
            if (ba != 0)
            {
                if (ba < 17)
                {
                    int half = (1 << (ba - 1)) - 1;
                    for (k = 0; k < group_size; k++)
                    {
                        dst[k] = (float)((int)drmp3_bs_get_bits(bs, ba) - half);
                    }
                } else
                {
                    unsigned mod = (2 << (ba - 17)) + 1;    /* 3, 5, 9 */
                    unsigned code = drmp3_bs_get_bits(bs, mod + 2 - (mod >> 3));  /* 5, 7, 10 */
                    for (k = 0; k < group_size; k++, code /= mod)
                    {
                        dst[k] = (float)((int)(code % mod - mod/2));
                    }
                }
            }
            dst += choff;
            choff = 18 - choff;
        }
    }
    return group_size*4;
}

static void drmp3_L12_apply_scf_384(drmp3_L12_scale_info *sci, const float *scf, float *dst)
{
    int i, k;
    DRMP3_COPY_MEMORY(dst + 576 + sci->stereo_bands*18, dst + sci->stereo_bands*18, (sci->total_bands - sci->stereo_bands)*18*sizeof(float));
    for (i = 0; i < sci->total_bands; i++, dst += 18, scf += 6)
    {
        for (k = 0; k < 12; k++)
        {
            dst[k + 0]   *= scf[0];
            dst[k + 576] *= scf[3];
        }
    }
}
#endif

static int drmp3_L3_read_side_info(drmp3_bs *bs, drmp3_L3_gr_info *gr, const drmp3_uint8 *hdr)
{
    static const drmp3_uint8 g_scf_long[8][23] = {
        { 6,6,6,6,6,6,8,10,12,14,16,20,24,28,32,38,46,52,60,68,58,54,0 },
        { 12,12,12,12,12,12,16,20,24,28,32,40,48,56,64,76,90,2,2,2,2,2,0 },
        { 6,6,6,6,6,6,8,10,12,14,16,20,24,28,32,38,46,52,60,68,58,54,0 },
        { 6,6,6,6,6,6,8,10,12,14,16,18,22,26,32,38,46,54,62,70,76,36,0 },
        { 6,6,6,6,6,6,8,10,12,14,16,20,24,28,32,38,46,52,60,68,58,54,0 },
        { 4,4,4,4,4,4,6,6,8,8,10,12,16,20,24,28,34,42,50,54,76,158,0 },
        { 4,4,4,4,4,4,6,6,6,8,10,12,16,18,22,28,34,40,46,54,54,192,0 },
        { 4,4,4,4,4,4,6,6,8,10,12,16,20,24,30,38,46,56,68,84,102,26,0 }
    };
    static const drmp3_uint8 g_scf_short[8][40] = {
        { 4,4,4,4,4,4,4,4,4,6,6,6,8,8,8,10,10,10,12,12,12,14,14,14,18,18,18,24,24,24,30,30,30,40,40,40,18,18,18,0 },
        { 8,8,8,8,8,8,8,8,8,12,12,12,16,16,16,20,20,20,24,24,24,28,28,28,36,36,36,2,2,2,2,2,2,2,2,2,26,26,26,0 },
        { 4,4,4,4,4,4,4,4,4,6,6,6,6,6,6,8,8,8,10,10,10,14,14,14,18,18,18,26,26,26,32,32,32,42,42,42,18,18,18,0 },
        { 4,4,4,4,4,4,4,4,4,6,6,6,8,8,8,10,10,10,12,12,12,14,14,14,18,18,18,24,24,24,32,32,32,44,44,44,12,12,12,0 },
        { 4,4,4,4,4,4,4,4,4,6,6,6,8,8,8,10,10,10,12,12,12,14,14,14,18,18,18,24,24,24,30,30,30,40,40,40,18,18,18,0 },
        { 4,4,4,4,4,4,4,4,4,4,4,4,6,6,6,8,8,8,10,10,10,12,12,12,14,14,14,18,18,18,22,22,22,30,30,30,56,56,56,0 },
        { 4,4,4,4,4,4,4,4,4,4,4,4,6,6,6,6,6,6,10,10,10,12,12,12,14,14,14,16,16,16,20,20,20,26,26,26,66,66,66,0 },
        { 4,4,4,4,4,4,4,4,4,4,4,4,6,6,6,8,8,8,12,12,12,16,16,16,20,20,20,26,26,26,34,34,34,42,42,42,12,12,12,0 }
    };
    static const drmp3_uint8 g_scf_mixed[8][40] = {
        { 6,6,6,6,6,6,6,6,6,8,8,8,10,10,10,12,12,12,14,14,14,18,18,18,24,24,24,30,30,30,40,40,40,18,18,18,0 },
        { 12,12,12,4,4,4,8,8,8,12,12,12,16,16,16,20,20,20,24,24,24,28,28,28,36,36,36,2,2,2,2,2,2,2,2,2,26,26,26,0 },
        { 6,6,6,6,6,6,6,6,6,6,6,6,8,8,8,10,10,10,14,14,14,18,18,18,26,26,26,32,32,32,42,42,42,18,18,18,0 },
        { 6,6,6,6,6,6,6,6,6,8,8,8,10,10,10,12,12,12,14,14,14,18,18,18,24,24,24,32,32,32,44,44,44,12,12,12,0 },
        { 6,6,6,6,6,6,6,6,6,8,8,8,10,10,10,12,12,12,14,14,14,18,18,18,24,24,24,30,30,30,40,40,40,18,18,18,0 },
        { 4,4,4,4,4,4,6,6,4,4,4,6,6,6,8,8,8,10,10,10,12,12,12,14,14,14,18,18,18,22,22,22,30,30,30,56,56,56,0 },
        { 4,4,4,4,4,4,6,6,4,4,4,6,6,6,6,6,6,10,10,10,12,12,12,14,14,14,16,16,16,20,20,20,26,26,26,66,66,66,0 },
        { 4,4,4,4,4,4,6,6,4,4,4,6,6,6,8,8,8,12,12,12,16,16,16,20,20,20,26,26,26,34,34,34,42,42,42,12,12,12,0 }
    };

    unsigned tables, scfsi = 0;
    int main_data_begin, part_23_sum = 0;
    int gr_count = DRMP3_HDR_IS_MONO(hdr) ? 1 : 2;
    int sr_idx = DRMP3_HDR_GET_MY_SAMPLE_RATE(hdr); sr_idx -= (sr_idx != 0);

    if (DRMP3_HDR_TEST_MPEG1(hdr))
    {
        gr_count *= 2;
        main_data_begin = drmp3_bs_get_bits(bs, 9);
        scfsi = drmp3_bs_get_bits(bs, 7 + gr_count);
    } else
    {
        main_data_begin = drmp3_bs_get_bits(bs, 8 + gr_count) >> gr_count;
    }

    do
    {
        if (DRMP3_HDR_IS_MONO(hdr))
        {
            scfsi <<= 4;
        }
        gr->part_23_length = (drmp3_uint16)drmp3_bs_get_bits(bs, 12);
        part_23_sum += gr->part_23_length;
        gr->big_values = (drmp3_uint16)drmp3_bs_get_bits(bs,  9);
        if (gr->big_values > 288)
        {
            return -1;
        }
        gr->global_gain = (drmp3_uint8)drmp3_bs_get_bits(bs, 8);
        gr->scalefac_compress = (drmp3_uint16)drmp3_bs_get_bits(bs, DRMP3_HDR_TEST_MPEG1(hdr) ? 4 : 9);
        gr->sfbtab = g_scf_long[sr_idx];
        gr->n_long_sfb  = 22;
        gr->n_short_sfb = 0;
        if (drmp3_bs_get_bits(bs, 1))
        {
            gr->block_type = (drmp3_uint8)drmp3_bs_get_bits(bs, 2);
            if (!gr->block_type)
            {
                return -1;
            }
            gr->mixed_block_flag = (drmp3_uint8)drmp3_bs_get_bits(bs, 1);
            gr->region_count[0] = 7;
            gr->region_count[1] = 255;
            if (gr->block_type == DRMP3_SHORT_BLOCK_TYPE)
            {
                scfsi &= 0x0F0F;
                if (!gr->mixed_block_flag)
                {
                    gr->region_count[0] = 8;
                    gr->sfbtab = g_scf_short[sr_idx];
                    gr->n_long_sfb = 0;
                    gr->n_short_sfb = 39;
                } else
                {
                    gr->sfbtab = g_scf_mixed[sr_idx];
                    gr->n_long_sfb = DRMP3_HDR_TEST_MPEG1(hdr) ? 8 : 6;
                    gr->n_short_sfb = 30;
                }
            }
            tables = drmp3_bs_get_bits(bs, 10);
            tables <<= 5;
            gr->subblock_gain[0] = (drmp3_uint8)drmp3_bs_get_bits(bs, 3);
            gr->subblock_gain[1] = (drmp3_uint8)drmp3_bs_get_bits(bs, 3);
            gr->subblock_gain[2] = (drmp3_uint8)drmp3_bs_get_bits(bs, 3);
        } else
        {
            gr->block_type = 0;
            gr->mixed_block_flag = 0;
            tables = drmp3_bs_get_bits(bs, 15);
            gr->region_count[0] = (drmp3_uint8)drmp3_bs_get_bits(bs, 4);
            gr->region_count[1] = (drmp3_uint8)drmp3_bs_get_bits(bs, 3);
            gr->region_count[2] = 255;
        }
        gr->table_select[0] = (drmp3_uint8)(tables >> 10);
        gr->table_select[1] = (drmp3_uint8)((tables >> 5) & 31);
        gr->table_select[2] = (drmp3_uint8)((tables) & 31);
        gr->preflag = (drmp3_uint8)(DRMP3_HDR_TEST_MPEG1(hdr) ? drmp3_bs_get_bits(bs, 1) : (gr->scalefac_compress >= 500));
        gr->scalefac_scale = (drmp3_uint8)drmp3_bs_get_bits(bs, 1);
        gr->count1_table = (drmp3_uint8)drmp3_bs_get_bits(bs, 1);
        gr->scfsi = (drmp3_uint8)((scfsi >> 12) & 15);
        scfsi <<= 4;
        gr++;
    } while(--gr_count);

    if (part_23_sum + bs->pos > bs->limit + main_data_begin*8)
    {
        return -1;
    }

    return main_data_begin;
}

static void drmp3_L3_read_scalefactors(drmp3_uint8 *scf, drmp3_uint8 *ist_pos, const drmp3_uint8 *scf_size, const drmp3_uint8 *scf_count, drmp3_bs *bitbuf, int scfsi)
{
    int i, k;
    for (i = 0; i < 4 && scf_count[i]; i++, scfsi *= 2)
    {
        int cnt = scf_count[i];
        if (scfsi & 8)
        {
            DRMP3_COPY_MEMORY(scf, ist_pos, cnt);
        } else
        {
            int bits = scf_size[i];
            if (!bits)
            {
                DRMP3_ZERO_MEMORY(scf, cnt);
                DRMP3_ZERO_MEMORY(ist_pos, cnt);
            } else
            {
                int max_scf = (scfsi < 0) ? (1 << bits) - 1 : -1;
                for (k = 0; k < cnt; k++)
                {
                    int s = drmp3_bs_get_bits(bitbuf, bits);
                    ist_pos[k] = (drmp3_uint8)(s == max_scf ? -1 : s);
                    scf[k] = (drmp3_uint8)s;
                }
            }
        }
        ist_pos += cnt;
        scf += cnt;
    }
    scf[0] = scf[1] = scf[2] = 0;
}

static float drmp3_L3_ldexp_q2(float y, int exp_q2)
{
    static const float g_expfrac[4] = { 9.31322575e-10f,7.83145814e-10f,6.58544508e-10f,5.53767716e-10f };
    int e;
    do
    {
        e = DRMP3_MIN(30*4, exp_q2);
        y *= g_expfrac[e & 3]*(1 << 30 >> (e >> 2));
    } while ((exp_q2 -= e) > 0);
    return y;
}

/*
I've had reports of GCC 14 throwing an incorrect -Wstringop-overflow warning here. This is an attempt
to silence this warning.
*/
#if (defined(__GNUC__) && (__GNUC__ >= 13)) && !defined(__clang__)
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wstringop-overflow"
#endif
static void drmp3_L3_decode_scalefactors(const drmp3_uint8 *hdr, drmp3_uint8 *ist_pos, drmp3_bs *bs, const drmp3_L3_gr_info *gr, float *scf, int ch)
{
    static const drmp3_uint8 g_scf_partitions[3][28] = {
        { 6,5,5, 5,6,5,5,5,6,5, 7,3,11,10,0,0, 7, 7, 7,0, 6, 6,6,3, 8, 8,5,0 },
        { 8,9,6,12,6,9,9,9,6,9,12,6,15,18,0,0, 6,15,12,0, 6,12,9,6, 6,18,9,0 },
        { 9,9,6,12,9,9,9,9,9,9,12,6,18,18,0,0,12,12,12,0,12, 9,9,6,15,12,9,0 }
    };
    const drmp3_uint8 *scf_partition = g_scf_partitions[!!gr->n_short_sfb + !gr->n_long_sfb];
    drmp3_uint8 scf_size[4], iscf[40];
    int i, scf_shift = gr->scalefac_scale + 1, gain_exp, scfsi = gr->scfsi;
    float gain;

    if (DRMP3_HDR_TEST_MPEG1(hdr))
    {
        static const drmp3_uint8 g_scfc_decode[16] = { 0,1,2,3, 12,5,6,7, 9,10,11,13, 14,15,18,19 };
        int part = g_scfc_decode[gr->scalefac_compress];
        scf_size[1] = scf_size[0] = (drmp3_uint8)(part >> 2);
        scf_size[3] = scf_size[2] = (drmp3_uint8)(part & 3);
    } else
    {
        static const drmp3_uint8 g_mod[6*4] = { 5,5,4,4,5,5,4,1,4,3,1,1,5,6,6,1,4,4,4,1,4,3,1,1 };
        int k, modprod, sfc, ist = DRMP3_HDR_TEST_I_STEREO(hdr) && ch;
        sfc = gr->scalefac_compress >> ist;
        for (k = ist*3*4; sfc >= 0; sfc -= modprod, k += 4)
        {
            for (modprod = 1, i = 3; i >= 0; i--)
            {
                scf_size[i] = (drmp3_uint8)(sfc / modprod % g_mod[k + i]);
                modprod *= g_mod[k + i];
            }
        }
        scf_partition += k;
        scfsi = -16;
    }
    drmp3_L3_read_scalefactors(iscf, ist_pos, scf_size, scf_partition, bs, scfsi);

    if (gr->n_short_sfb)
    {
        int sh = 3 - scf_shift;
        for (i = 0; i < gr->n_short_sfb; i += 3)
        {
            iscf[gr->n_long_sfb + i + 0] = (drmp3_uint8)(iscf[gr->n_long_sfb + i + 0] + (gr->subblock_gain[0] << sh));
            iscf[gr->n_long_sfb + i + 1] = (drmp3_uint8)(iscf[gr->n_long_sfb + i + 1] + (gr->subblock_gain[1] << sh));
            iscf[gr->n_long_sfb + i + 2] = (drmp3_uint8)(iscf[gr->n_long_sfb + i + 2] + (gr->subblock_gain[2] << sh));
        }
    } else if (gr->preflag)
    {
        static const drmp3_uint8 g_preamp[10] = { 1,1,1,1,2,2,3,3,3,2 };
        for (i = 0; i < 10; i++)
        {
            iscf[11 + i] = (drmp3_uint8)(iscf[11 + i] + g_preamp[i]);
        }
    }

    gain_exp = gr->global_gain + DRMP3_BITS_DEQUANTIZER_OUT*4 - 210 - (DRMP3_HDR_IS_MS_STEREO(hdr) ? 2 : 0);
    gain = drmp3_L3_ldexp_q2(1 << (DRMP3_MAX_SCFI/4),  DRMP3_MAX_SCFI - gain_exp);
    for (i = 0; i < (int)(gr->n_long_sfb + gr->n_short_sfb); i++)
    {
        scf[i] = drmp3_L3_ldexp_q2(gain, iscf[i] << scf_shift);
    }
}
#if (defined(__GNUC__) && (__GNUC__ >= 13)) && !defined(__clang__)
    #pragma GCC diagnostic pop
#endif

static const float g_drmp3_pow43[129 + 16] = {
    0,-1,-2.519842f,-4.326749f,-6.349604f,-8.549880f,-10.902724f,-13.390518f,-16.000000f,-18.720754f,-21.544347f,-24.463781f,-27.473142f,-30.567351f,-33.741992f,-36.993181f,
    0,1,2.519842f,4.326749f,6.349604f,8.549880f,10.902724f,13.390518f,16.000000f,18.720754f,21.544347f,24.463781f,27.473142f,30.567351f,33.741992f,36.993181f,40.317474f,43.711787f,47.173345f,50.699631f,54.288352f,57.937408f,61.644865f,65.408941f,69.227979f,73.100443f,77.024898f,81.000000f,85.024491f,89.097188f,93.216975f,97.382800f,101.593667f,105.848633f,110.146801f,114.487321f,118.869381f,123.292209f,127.755065f,132.257246f,136.798076f,141.376907f,145.993119f,150.646117f,155.335327f,160.060199f,164.820202f,169.614826f,174.443577f,179.305980f,184.201575f,189.129918f,194.090580f,199.083145f,204.107210f,209.162385f,214.248292f,219.364564f,224.510845f,229.686789f,234.892058f,240.126328f,245.389280f,250.680604f,256.000000f,261.347174f,266.721841f,272.123723f,277.552547f,283.008049f,288.489971f,293.998060f,299.532071f,305.091761f,310.676898f,316.287249f,321.922592f,327.582707f,333.267377f,338.976394f,344.709550f,350.466646f,356.247482f,362.051866f,367.879608f,373.730522f,379.604427f,385.501143f,391.420496f,397.362314f,403.326427f,409.312672f,415.320884f,421.350905f,427.402579f,433.475750f,439.570269f,445.685987f,451.822757f,457.980436f,464.158883f,470.357960f,476.577530f,482.817459f,489.077615f,495.357868f,501.658090f,507.978156f,514.317941f,520.677324f,527.056184f,533.454404f,539.871867f,546.308458f,552.764065f,559.238575f,565.731879f,572.243870f,578.774440f,585.323483f,591.890898f,598.476581f,605.080431f,611.702349f,618.342238f,625.000000f,631.675540f,638.368763f,645.079578f
};

static float drmp3_L3_pow_43(int x)
{
    float frac;
    int sign, mult = 256;

    if (x < 129)
    {
        return g_drmp3_pow43[16 + x];
    }

    if (x < 1024)
    {
        mult = 16;
        x <<= 3;
    }

    sign = 2*x & 64;
    frac = (float)((x & 63) - sign) / ((x & ~63) + sign);
    return g_drmp3_pow43[16 + ((x + sign) >> 6)]*(1.f + frac*((4.f/3) + frac*(2.f/9)))*mult;
}

static void drmp3_L3_huffman(float *dst, drmp3_bs *bs, const drmp3_L3_gr_info *gr_info, const float *scf, int layer3gr_limit)
{
    static const drmp3_int16 tabs[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        785,785,785,785,784,784,784,784,513,513,513,513,513,513,513,513,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,
        -255,1313,1298,1282,785,785,785,785,784,784,784,784,769,769,769,769,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,290,288,
        -255,1313,1298,1282,769,769,769,769,529,529,529,529,529,529,529,529,528,528,528,528,528,528,528,528,512,512,512,512,512,512,512,512,290,288,
        -253,-318,-351,-367,785,785,785,785,784,784,784,784,769,769,769,769,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,819,818,547,547,275,275,275,275,561,560,515,546,289,274,288,258,
        -254,-287,1329,1299,1314,1312,1057,1057,1042,1042,1026,1026,784,784,784,784,529,529,529,529,529,529,529,529,769,769,769,769,768,768,768,768,563,560,306,306,291,259,
        -252,-413,-477,-542,1298,-575,1041,1041,784,784,784,784,769,769,769,769,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,-383,-399,1107,1092,1106,1061,849,849,789,789,1104,1091,773,773,1076,1075,341,340,325,309,834,804,577,577,532,532,516,516,832,818,803,816,561,561,531,531,515,546,289,289,288,258,
        -252,-429,-493,-559,1057,1057,1042,1042,529,529,529,529,529,529,529,529,784,784,784,784,769,769,769,769,512,512,512,512,512,512,512,512,-382,1077,-415,1106,1061,1104,849,849,789,789,1091,1076,1029,1075,834,834,597,581,340,340,339,324,804,833,532,532,832,772,818,803,817,787,816,771,290,290,290,290,288,258,
        -253,-349,-414,-447,-463,1329,1299,-479,1314,1312,1057,1057,1042,1042,1026,1026,785,785,785,785,784,784,784,784,769,769,769,769,768,768,768,768,-319,851,821,-335,836,850,805,849,341,340,325,336,533,533,579,579,564,564,773,832,578,548,563,516,321,276,306,291,304,259,
        -251,-572,-733,-830,-863,-879,1041,1041,784,784,784,784,769,769,769,769,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,-511,-527,-543,1396,1351,1381,1366,1395,1335,1380,-559,1334,1138,1138,1063,1063,1350,1392,1031,1031,1062,1062,1364,1363,1120,1120,1333,1348,881,881,881,881,375,374,359,373,343,358,341,325,791,791,1123,1122,-703,1105,1045,-719,865,865,790,790,774,774,1104,1029,338,293,323,308,-799,-815,833,788,772,818,803,816,322,292,307,320,561,531,515,546,289,274,288,258,
        -251,-525,-605,-685,-765,-831,-846,1298,1057,1057,1312,1282,785,785,785,785,784,784,784,784,769,769,769,769,512,512,512,512,512,512,512,512,1399,1398,1383,1367,1382,1396,1351,-511,1381,1366,1139,1139,1079,1079,1124,1124,1364,1349,1363,1333,882,882,882,882,807,807,807,807,1094,1094,1136,1136,373,341,535,535,881,775,867,822,774,-591,324,338,-671,849,550,550,866,864,609,609,293,336,534,534,789,835,773,-751,834,804,308,307,833,788,832,772,562,562,547,547,305,275,560,515,290,290,
        -252,-397,-477,-557,-622,-653,-719,-735,-750,1329,1299,1314,1057,1057,1042,1042,1312,1282,1024,1024,785,785,785,785,784,784,784,784,769,769,769,769,-383,1127,1141,1111,1126,1140,1095,1110,869,869,883,883,1079,1109,882,882,375,374,807,868,838,881,791,-463,867,822,368,263,852,837,836,-543,610,610,550,550,352,336,534,534,865,774,851,821,850,805,593,533,579,564,773,832,578,578,548,548,577,577,307,276,306,291,516,560,259,259,
        -250,-2107,-2507,-2764,-2909,-2974,-3007,-3023,1041,1041,1040,1040,769,769,769,769,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,-767,-1052,-1213,-1277,-1358,-1405,-1469,-1535,-1550,-1582,-1614,-1647,-1662,-1694,-1726,-1759,-1774,-1807,-1822,-1854,-1886,1565,-1919,-1935,-1951,-1967,1731,1730,1580,1717,-1983,1729,1564,-1999,1548,-2015,-2031,1715,1595,-2047,1714,-2063,1610,-2079,1609,-2095,1323,1323,1457,1457,1307,1307,1712,1547,1641,1700,1699,1594,1685,1625,1442,1442,1322,1322,-780,-973,-910,1279,1278,1277,1262,1276,1261,1275,1215,1260,1229,-959,974,974,989,989,-943,735,478,478,495,463,506,414,-1039,1003,958,1017,927,942,987,957,431,476,1272,1167,1228,-1183,1256,-1199,895,895,941,941,1242,1227,1212,1135,1014,1014,490,489,503,487,910,1013,985,925,863,894,970,955,1012,847,-1343,831,755,755,984,909,428,366,754,559,-1391,752,486,457,924,997,698,698,983,893,740,740,908,877,739,739,667,667,953,938,497,287,271,271,683,606,590,712,726,574,302,302,738,736,481,286,526,725,605,711,636,724,696,651,589,681,666,710,364,467,573,695,466,466,301,465,379,379,709,604,665,679,316,316,634,633,436,436,464,269,424,394,452,332,438,363,347,408,393,448,331,422,362,407,392,421,346,406,391,376,375,359,1441,1306,-2367,1290,-2383,1337,-2399,-2415,1426,1321,-2431,1411,1336,-2447,-2463,-2479,1169,1169,1049,1049,1424,1289,1412,1352,1319,-2495,1154,1154,1064,1064,1153,1153,416,390,360,404,403,389,344,374,373,343,358,372,327,357,342,311,356,326,1395,1394,1137,1137,1047,1047,1365,1392,1287,1379,1334,1364,1349,1378,1318,1363,792,792,792,792,1152,1152,1032,1032,1121,1121,1046,1046,1120,1120,1030,1030,-2895,1106,1061,1104,849,849,789,789,1091,1076,1029,1090,1060,1075,833,833,309,324,532,532,832,772,818,803,561,561,531,560,515,546,289,274,288,258,
        -250,-1179,-1579,-1836,-1996,-2124,-2253,-2333,-2413,-2477,-2542,-2574,-2607,-2622,-2655,1314,1313,1298,1312,1282,785,785,785,785,1040,1040,1025,1025,768,768,768,768,-766,-798,-830,-862,-895,-911,-927,-943,-959,-975,-991,-1007,-1023,-1039,-1055,-1070,1724,1647,-1103,-1119,1631,1767,1662,1738,1708,1723,-1135,1780,1615,1779,1599,1677,1646,1778,1583,-1151,1777,1567,1737,1692,1765,1722,1707,1630,1751,1661,1764,1614,1736,1676,1763,1750,1645,1598,1721,1691,1762,1706,1582,1761,1566,-1167,1749,1629,767,766,751,765,494,494,735,764,719,749,734,763,447,447,748,718,477,506,431,491,446,476,461,505,415,430,475,445,504,399,460,489,414,503,383,474,429,459,502,502,746,752,488,398,501,473,413,472,486,271,480,270,-1439,-1455,1357,-1471,-1487,-1503,1341,1325,-1519,1489,1463,1403,1309,-1535,1372,1448,1418,1476,1356,1462,1387,-1551,1475,1340,1447,1402,1386,-1567,1068,1068,1474,1461,455,380,468,440,395,425,410,454,364,467,466,464,453,269,409,448,268,432,1371,1473,1432,1417,1308,1460,1355,1446,1459,1431,1083,1083,1401,1416,1458,1445,1067,1067,1370,1457,1051,1051,1291,1430,1385,1444,1354,1415,1400,1443,1082,1082,1173,1113,1186,1066,1185,1050,-1967,1158,1128,1172,1097,1171,1081,-1983,1157,1112,416,266,375,400,1170,1142,1127,1065,793,793,1169,1033,1156,1096,1141,1111,1155,1080,1126,1140,898,898,808,808,897,897,792,792,1095,1152,1032,1125,1110,1139,1079,1124,882,807,838,881,853,791,-2319,867,368,263,822,852,837,866,806,865,-2399,851,352,262,534,534,821,836,594,594,549,549,593,593,533,533,848,773,579,579,564,578,548,563,276,276,577,576,306,291,516,560,305,305,275,259,
        -251,-892,-2058,-2620,-2828,-2957,-3023,-3039,1041,1041,1040,1040,769,769,769,769,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,-511,-527,-543,-559,1530,-575,-591,1528,1527,1407,1526,1391,1023,1023,1023,1023,1525,1375,1268,1268,1103,1103,1087,1087,1039,1039,1523,-604,815,815,815,815,510,495,509,479,508,463,507,447,431,505,415,399,-734,-782,1262,-815,1259,1244,-831,1258,1228,-847,-863,1196,-879,1253,987,987,748,-767,493,493,462,477,414,414,686,669,478,446,461,445,474,429,487,458,412,471,1266,1264,1009,1009,799,799,-1019,-1276,-1452,-1581,-1677,-1757,-1821,-1886,-1933,-1997,1257,1257,1483,1468,1512,1422,1497,1406,1467,1496,1421,1510,1134,1134,1225,1225,1466,1451,1374,1405,1252,1252,1358,1480,1164,1164,1251,1251,1238,1238,1389,1465,-1407,1054,1101,-1423,1207,-1439,830,830,1248,1038,1237,1117,1223,1148,1236,1208,411,426,395,410,379,269,1193,1222,1132,1235,1221,1116,976,976,1192,1162,1177,1220,1131,1191,963,963,-1647,961,780,-1663,558,558,994,993,437,408,393,407,829,978,813,797,947,-1743,721,721,377,392,844,950,828,890,706,706,812,859,796,960,948,843,934,874,571,571,-1919,690,555,689,421,346,539,539,944,779,918,873,932,842,903,888,570,570,931,917,674,674,-2575,1562,-2591,1609,-2607,1654,1322,1322,1441,1441,1696,1546,1683,1593,1669,1624,1426,1426,1321,1321,1639,1680,1425,1425,1305,1305,1545,1668,1608,1623,1667,1592,1638,1666,1320,1320,1652,1607,1409,1409,1304,1304,1288,1288,1664,1637,1395,1395,1335,1335,1622,1636,1394,1394,1319,1319,1606,1621,1392,1392,1137,1137,1137,1137,345,390,360,375,404,373,1047,-2751,-2767,-2783,1062,1121,1046,-2799,1077,-2815,1106,1061,789,789,1105,1104,263,355,310,340,325,354,352,262,339,324,1091,1076,1029,1090,1060,1075,833,833,788,788,1088,1028,818,818,803,803,561,561,531,531,816,771,546,546,289,274,288,258,
        -253,-317,-381,-446,-478,-509,1279,1279,-811,-1179,-1451,-1756,-1900,-2028,-2189,-2253,-2333,-2414,-2445,-2511,-2526,1313,1298,-2559,1041,1041,1040,1040,1025,1025,1024,1024,1022,1007,1021,991,1020,975,1019,959,687,687,1018,1017,671,671,655,655,1016,1015,639,639,758,758,623,623,757,607,756,591,755,575,754,559,543,543,1009,783,-575,-621,-685,-749,496,-590,750,749,734,748,974,989,1003,958,988,973,1002,942,987,957,972,1001,926,986,941,971,956,1000,910,985,925,999,894,970,-1071,-1087,-1102,1390,-1135,1436,1509,1451,1374,-1151,1405,1358,1480,1420,-1167,1507,1494,1389,1342,1465,1435,1450,1326,1505,1310,1493,1373,1479,1404,1492,1464,1419,428,443,472,397,736,526,464,464,486,457,442,471,484,482,1357,1449,1434,1478,1388,1491,1341,1490,1325,1489,1463,1403,1309,1477,1372,1448,1418,1433,1476,1356,1462,1387,-1439,1475,1340,1447,1402,1474,1324,1461,1371,1473,269,448,1432,1417,1308,1460,-1711,1459,-1727,1441,1099,1099,1446,1386,1431,1401,-1743,1289,1083,1083,1160,1160,1458,1445,1067,1067,1370,1457,1307,1430,1129,1129,1098,1098,268,432,267,416,266,400,-1887,1144,1187,1082,1173,1113,1186,1066,1050,1158,1128,1143,1172,1097,1171,1081,420,391,1157,1112,1170,1142,1127,1065,1169,1049,1156,1096,1141,1111,1155,1080,1126,1154,1064,1153,1140,1095,1048,-2159,1125,1110,1137,-2175,823,823,1139,1138,807,807,384,264,368,263,868,838,853,791,867,822,852,837,866,806,865,790,-2319,851,821,836,352,262,850,805,849,-2399,533,533,835,820,336,261,578,548,563,577,532,532,832,772,562,562,547,547,305,275,560,515,290,290,288,258 };
    static const drmp3_uint8 tab32[] = { 130,162,193,209,44,28,76,140,9,9,9,9,9,9,9,9,190,254,222,238,126,94,157,157,109,61,173,205};
    static const drmp3_uint8 tab33[] = { 252,236,220,204,188,172,156,140,124,108,92,76,60,44,28,12 };
    static const drmp3_int16 tabindex[2*16] = { 0,32,64,98,0,132,180,218,292,364,426,538,648,746,0,1126,1460,1460,1460,1460,1460,1460,1460,1460,1842,1842,1842,1842,1842,1842,1842,1842 };
    static const drmp3_uint8 g_linbits[] =  { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,2,3,4,6,8,10,13,4,5,6,7,8,9,11,13 };

#define DRMP3_PEEK_BITS(n)    (bs_cache >> (32 - (n)))
#define DRMP3_FLUSH_BITS(n)   { bs_cache <<= (n); bs_sh += (n); }
#define DRMP3_CHECK_BITS      while (bs_sh >= 0) { bs_cache |= (drmp3_uint32)*bs_next_ptr++ << bs_sh; bs_sh -= 8; }
#define DRMP3_BSPOS           ((bs_next_ptr - bs->buf)*8 - 24 + bs_sh)

    float one = 0.0f;
    int ireg = 0, big_val_cnt = gr_info->big_values;
    const drmp3_uint8 *sfb = gr_info->sfbtab;
    const drmp3_uint8 *bs_next_ptr = bs->buf + bs->pos/8;
    drmp3_uint32 bs_cache = (((bs_next_ptr[0]*256u + bs_next_ptr[1])*256u + bs_next_ptr[2])*256u + bs_next_ptr[3]) << (bs->pos & 7);
    int pairs_to_decode, np, bs_sh = (bs->pos & 7) - 8;
    bs_next_ptr += 4;

    while (big_val_cnt > 0)
    {
        int tab_num = gr_info->table_select[ireg];
        int sfb_cnt = gr_info->region_count[ireg++];
        const drmp3_int16 *codebook = tabs + tabindex[tab_num];
        int linbits = g_linbits[tab_num];
        if (linbits)
        {
            do
            {
                np = *sfb++ / 2;
                pairs_to_decode = DRMP3_MIN(big_val_cnt, np);
                one = *scf++;
                do
                {
                    int j, w = 5;
                    int leaf = codebook[DRMP3_PEEK_BITS(w)];
                    while (leaf < 0)
                    {
                        DRMP3_FLUSH_BITS(w);
                        w = leaf & 7;
                        leaf = codebook[DRMP3_PEEK_BITS(w) - (leaf >> 3)];
                    }
                    DRMP3_FLUSH_BITS(leaf >> 8);

                    for (j = 0; j < 2; j++, dst++, leaf >>= 4)
                    {
                        int lsb = leaf & 0x0F;
                        if (lsb == 15)
                        {
                            lsb += DRMP3_PEEK_BITS(linbits);
                            DRMP3_FLUSH_BITS(linbits);
                            DRMP3_CHECK_BITS;
                            *dst = one*drmp3_L3_pow_43(lsb)*((drmp3_int32)bs_cache < 0 ? -1: 1);
                        } else
                        {
                            *dst = g_drmp3_pow43[16 + lsb - 16*(bs_cache >> 31)]*one;
                        }
                        DRMP3_FLUSH_BITS(lsb ? 1 : 0);
                    }
                    DRMP3_CHECK_BITS;
                } while (--pairs_to_decode);
            } while ((big_val_cnt -= np) > 0 && --sfb_cnt >= 0);
        } else
        {
            do
            {
                np = *sfb++ / 2;
                pairs_to_decode = DRMP3_MIN(big_val_cnt, np);
                one = *scf++;
                do
                {
                    int j, w = 5;
                    int leaf = codebook[DRMP3_PEEK_BITS(w)];
                    while (leaf < 0)
                    {
                        DRMP3_FLUSH_BITS(w);
                        w = leaf & 7;
                        leaf = codebook[DRMP3_PEEK_BITS(w) - (leaf >> 3)];
                    }
                    DRMP3_FLUSH_BITS(leaf >> 8);

                    for (j = 0; j < 2; j++, dst++, leaf >>= 4)
                    {
                        int lsb = leaf & 0x0F;
                        *dst = g_drmp3_pow43[16 + lsb - 16*(bs_cache >> 31)]*one;
                        DRMP3_FLUSH_BITS(lsb ? 1 : 0);
                    }
                    DRMP3_CHECK_BITS;
                } while (--pairs_to_decode);
            } while ((big_val_cnt -= np) > 0 && --sfb_cnt >= 0);
        }
    }

    for (np = 1 - big_val_cnt;; dst += 4)
    {
        const drmp3_uint8 *codebook_count1 = (gr_info->count1_table) ? tab33 : tab32;
        int leaf = codebook_count1[DRMP3_PEEK_BITS(4)];
        if (!(leaf & 8))
        {
            leaf = codebook_count1[(leaf >> 3) + (bs_cache << 4 >> (32 - (leaf & 3)))];
        }
        DRMP3_FLUSH_BITS(leaf & 7);
        if (DRMP3_BSPOS > layer3gr_limit)
        {
            break;
        }
#define DRMP3_RELOAD_SCALEFACTOR  if (!--np) { np = *sfb++/2; if (!np) break; one = *scf++; }
#define DRMP3_DEQ_COUNT1(s) if (leaf & (128 >> s)) { dst[s] = ((drmp3_int32)bs_cache < 0) ? -one : one; DRMP3_FLUSH_BITS(1) }
        DRMP3_RELOAD_SCALEFACTOR;
        DRMP3_DEQ_COUNT1(0);
        DRMP3_DEQ_COUNT1(1);
        DRMP3_RELOAD_SCALEFACTOR;
        DRMP3_DEQ_COUNT1(2);
        DRMP3_DEQ_COUNT1(3);
        DRMP3_CHECK_BITS;
    }

    bs->pos = layer3gr_limit;
}

static void drmp3_L3_midside_stereo(float *left, int n)
{
    int i = 0;
    float *right = left + 576;
#if DRMP3_HAVE_SIMD
    if (drmp3_have_simd())
    {
        for (; i < n - 3; i += 4)
        {
            drmp3_f4 vl = DRMP3_VLD(left + i);
            drmp3_f4 vr = DRMP3_VLD(right + i);
            DRMP3_VSTORE(left + i, DRMP3_VADD(vl, vr));
            DRMP3_VSTORE(right + i, DRMP3_VSUB(vl, vr));
        }
#ifdef __GNUC__
        /* Workaround for spurious -Waggressive-loop-optimizations warning from gcc.
         * For more info see: https://github.com/lieff/minimp3/issues/88
         */
        if (__builtin_constant_p(n % 4 == 0) && n % 4 == 0)
            return;
#endif
    }
#endif
    for (; i < n; i++)
    {
        float a = left[i];
        float b = right[i];
        left[i] = a + b;
        right[i] = a - b;
    }
}

static void drmp3_L3_intensity_stereo_band(float *left, int n, float kl, float kr)
{
    int i;
    for (i = 0; i < n; i++)
    {
        left[i + 576] = left[i]*kr;
        left[i] = left[i]*kl;
    }
}

static void drmp3_L3_stereo_top_band(const float *right, const drmp3_uint8 *sfb, int nbands, int max_band[3])
{
    int i, k;

    max_band[0] = max_band[1] = max_band[2] = -1;

    for (i = 0; i < nbands; i++)
    {
        for (k = 0; k < sfb[i]; k += 2)
        {
            if (right[k] != 0 || right[k + 1] != 0)
            {
                max_band[i % 3] = i;
                break;
            }
        }
        right += sfb[i];
    }
}

static void drmp3_L3_stereo_process(float *left, const drmp3_uint8 *ist_pos, const drmp3_uint8 *sfb, const drmp3_uint8 *hdr, int max_band[3], int mpeg2_sh)
{
    static const float g_pan[7*2] = { 0,1,0.21132487f,0.78867513f,0.36602540f,0.63397460f,0.5f,0.5f,0.63397460f,0.36602540f,0.78867513f,0.21132487f,1,0 };
    unsigned i, max_pos = DRMP3_HDR_TEST_MPEG1(hdr) ? 7 : 64;

    for (i = 0; sfb[i]; i++)
    {
        unsigned ipos = ist_pos[i];
        if ((int)i > max_band[i % 3] && ipos < max_pos)
        {
            float kl, kr, s = DRMP3_HDR_TEST_MS_STEREO(hdr) ? 1.41421356f : 1;
            if (DRMP3_HDR_TEST_MPEG1(hdr))
            {
                kl = g_pan[2*ipos];
                kr = g_pan[2*ipos + 1];
            } else
            {
                kl = 1;
                kr = drmp3_L3_ldexp_q2(1, (ipos + 1) >> 1 << mpeg2_sh);
                if (ipos & 1)
                {
                    kl = kr;
                    kr = 1;
                }
            }
            drmp3_L3_intensity_stereo_band(left, sfb[i], kl*s, kr*s);
        } else if (DRMP3_HDR_TEST_MS_STEREO(hdr))
        {
            drmp3_L3_midside_stereo(left, sfb[i]);
        }
        left += sfb[i];
    }
}

static void drmp3_L3_intensity_stereo(float *left, drmp3_uint8 *ist_pos, const drmp3_L3_gr_info *gr, const drmp3_uint8 *hdr)
{
    int max_band[3], n_sfb = gr->n_long_sfb + gr->n_short_sfb;
    int i, max_blocks = gr->n_short_sfb ? 3 : 1;

    drmp3_L3_stereo_top_band(left + 576, gr->sfbtab, n_sfb, max_band);
    if (gr->n_long_sfb)
    {
        max_band[0] = max_band[1] = max_band[2] = DRMP3_MAX(DRMP3_MAX(max_band[0], max_band[1]), max_band[2]);
    }
    for (i = 0; i < max_blocks; i++)
    {
        int default_pos = DRMP3_HDR_TEST_MPEG1(hdr) ? 3 : 0;
        int itop = n_sfb - max_blocks + i;
        int prev = itop - max_blocks;
        ist_pos[itop] = (drmp3_uint8)(max_band[i] >= prev ? default_pos : ist_pos[prev]);
    }
    drmp3_L3_stereo_process(left, ist_pos, gr->sfbtab, hdr, max_band, gr[1].scalefac_compress & 1);
}

static void drmp3_L3_reorder(float *grbuf, float *scratch, const drmp3_uint8 *sfb)
{
    int i, len;
    float *src = grbuf, *dst = scratch;

    for (;0 != (len = *sfb); sfb += 3, src += 2*len)
    {
        for (i = 0; i < len; i++, src++)
        {
            *dst++ = src[0*len];
            *dst++ = src[1*len];
            *dst++ = src[2*len];
        }
    }
    DRMP3_COPY_MEMORY(grbuf, scratch, (dst - scratch)*sizeof(float));
}

static void drmp3_L3_antialias(float *grbuf, int nbands)
{
    static const float g_aa[2][8] = {
        {0.85749293f,0.88174200f,0.94962865f,0.98331459f,0.99551782f,0.99916056f,0.99989920f,0.99999316f},
        {0.51449576f,0.47173197f,0.31337745f,0.18191320f,0.09457419f,0.04096558f,0.01419856f,0.00369997f}
    };

    for (; nbands > 0; nbands--, grbuf += 18)
    {
        int i = 0;
#if DRMP3_HAVE_SIMD
        if (drmp3_have_simd()) for (; i < 8; i += 4)
        {
            drmp3_f4 vu = DRMP3_VLD(grbuf + 18 + i);
            drmp3_f4 vd = DRMP3_VLD(grbuf + 14 - i);
            drmp3_f4 vc0 = DRMP3_VLD(g_aa[0] + i);
            drmp3_f4 vc1 = DRMP3_VLD(g_aa[1] + i);
            vd = DRMP3_VREV(vd);
            DRMP3_VSTORE(grbuf + 18 + i, DRMP3_VSUB(DRMP3_VMUL(vu, vc0), DRMP3_VMUL(vd, vc1)));
            vd = DRMP3_VADD(DRMP3_VMUL(vu, vc1), DRMP3_VMUL(vd, vc0));
            DRMP3_VSTORE(grbuf + 14 - i, DRMP3_VREV(vd));
        }
#endif
#ifndef DR_MP3_ONLY_SIMD
        for(; i < 8; i++)
        {
            float u = grbuf[18 + i];
            float d = grbuf[17 - i];
            grbuf[18 + i] = u*g_aa[0][i] - d*g_aa[1][i];
            grbuf[17 - i] = u*g_aa[1][i] + d*g_aa[0][i];
        }
#endif
    }
}

static void drmp3_L3_dct3_9(float *y)
{
    float s0, s1, s2, s3, s4, s5, s6, s7, s8, t0, t2, t4;

    s0 = y[0]; s2 = y[2]; s4 = y[4]; s6 = y[6]; s8 = y[8];
    t0 = s0 + s6*0.5f;
    s0 -= s6;
    t4 = (s4 + s2)*0.93969262f;
    t2 = (s8 + s2)*0.76604444f;
    s6 = (s4 - s8)*0.17364818f;
    s4 += s8 - s2;

    s2 = s0 - s4*0.5f;
    y[4] = s4 + s0;
    s8 = t0 - t2 + s6;
    s0 = t0 - t4 + t2;
    s4 = t0 + t4 - s6;

    s1 = y[1]; s3 = y[3]; s5 = y[5]; s7 = y[7];

    s3 *= 0.86602540f;
    t0 = (s5 + s1)*0.98480775f;
    t4 = (s5 - s7)*0.34202014f;
    t2 = (s1 + s7)*0.64278761f;
    s1 = (s1 - s5 - s7)*0.86602540f;

    s5 = t0 - s3 - t2;
    s7 = t4 - s3 - t0;
    s3 = t4 + s3 - t2;

    y[0] = s4 - s7;
    y[1] = s2 + s1;
    y[2] = s0 - s3;
    y[3] = s8 + s5;
    y[5] = s8 - s5;
    y[6] = s0 + s3;
    y[7] = s2 - s1;
    y[8] = s4 + s7;
}

static void drmp3_L3_imdct36(float *grbuf, float *overlap, const float *window, int nbands)
{
    int i, j;
    static const float g_twid9[18] = {
        0.73727734f,0.79335334f,0.84339145f,0.88701083f,0.92387953f,0.95371695f,0.97629601f,0.99144486f,0.99904822f,0.67559021f,0.60876143f,0.53729961f,0.46174861f,0.38268343f,0.30070580f,0.21643961f,0.13052619f,0.04361938f
    };

    for (j = 0; j < nbands; j++, grbuf += 18, overlap += 9)
    {
        float co[9], si[9];
        co[0] = -grbuf[0];
        si[0] = grbuf[17];
        for (i = 0; i < 4; i++)
        {
            si[8 - 2*i] =   grbuf[4*i + 1] - grbuf[4*i + 2];
            co[1 + 2*i] =   grbuf[4*i + 1] + grbuf[4*i + 2];
            si[7 - 2*i] =   grbuf[4*i + 4] - grbuf[4*i + 3];
            co[2 + 2*i] = -(grbuf[4*i + 3] + grbuf[4*i + 4]);
        }
        drmp3_L3_dct3_9(co);
        drmp3_L3_dct3_9(si);

        si[1] = -si[1];
        si[3] = -si[3];
        si[5] = -si[5];
        si[7] = -si[7];

        i = 0;

#if DRMP3_HAVE_SIMD
        if (drmp3_have_simd()) for (; i < 8; i += 4)
        {
            drmp3_f4 vovl = DRMP3_VLD(overlap + i);
            drmp3_f4 vc = DRMP3_VLD(co + i);
            drmp3_f4 vs = DRMP3_VLD(si + i);
            drmp3_f4 vr0 = DRMP3_VLD(g_twid9 + i);
            drmp3_f4 vr1 = DRMP3_VLD(g_twid9 + 9 + i);
            drmp3_f4 vw0 = DRMP3_VLD(window + i);
            drmp3_f4 vw1 = DRMP3_VLD(window + 9 + i);
            drmp3_f4 vsum = DRMP3_VADD(DRMP3_VMUL(vc, vr1), DRMP3_VMUL(vs, vr0));
            DRMP3_VSTORE(overlap + i, DRMP3_VSUB(DRMP3_VMUL(vc, vr0), DRMP3_VMUL(vs, vr1)));
            DRMP3_VSTORE(grbuf + i, DRMP3_VSUB(DRMP3_VMUL(vovl, vw0), DRMP3_VMUL(vsum, vw1)));
            vsum = DRMP3_VADD(DRMP3_VMUL(vovl, vw1), DRMP3_VMUL(vsum, vw0));
            DRMP3_VSTORE(grbuf + 14 - i, DRMP3_VREV(vsum));
        }
#endif
        for (; i < 9; i++)
        {
            float ovl  = overlap[i];
            float sum  = co[i]*g_twid9[9 + i] + si[i]*g_twid9[0 + i];
            overlap[i] = co[i]*g_twid9[0 + i] - si[i]*g_twid9[9 + i];
            grbuf[i]      = ovl*window[0 + i] - sum*window[9 + i];
            grbuf[17 - i] = ovl*window[9 + i] + sum*window[0 + i];
        }
    }
}

static void drmp3_L3_idct3(float x0, float x1, float x2, float *dst)
{
    float m1 = x1*0.86602540f;
    float a1 = x0 - x2*0.5f;
    dst[1] = x0 + x2;
    dst[0] = a1 + m1;
    dst[2] = a1 - m1;
}

static void drmp3_L3_imdct12(float *x, float *dst, float *overlap)
{
    static const float g_twid3[6] = { 0.79335334f,0.92387953f,0.99144486f, 0.60876143f,0.38268343f,0.13052619f };
    float co[3], si[3];
    int i;

    drmp3_L3_idct3(-x[0], x[6] + x[3], x[12] + x[9], co);
    drmp3_L3_idct3(x[15], x[12] - x[9], x[6] - x[3], si);
    si[1] = -si[1];

    for (i = 0; i < 3; i++)
    {
        float ovl  = overlap[i];
        float sum  = co[i]*g_twid3[3 + i] + si[i]*g_twid3[0 + i];
        overlap[i] = co[i]*g_twid3[0 + i] - si[i]*g_twid3[3 + i];
        dst[i]     = ovl*g_twid3[2 - i] - sum*g_twid3[5 - i];
        dst[5 - i] = ovl*g_twid3[5 - i] + sum*g_twid3[2 - i];
    }
}

static void drmp3_L3_imdct_short(float *grbuf, float *overlap, int nbands)
{
    for (;nbands > 0; nbands--, overlap += 9, grbuf += 18)
    {
        float tmp[18];
        DRMP3_COPY_MEMORY(tmp, grbuf, sizeof(tmp));
        DRMP3_COPY_MEMORY(grbuf, overlap, 6*sizeof(float));
        drmp3_L3_imdct12(tmp, grbuf + 6, overlap + 6);
        drmp3_L3_imdct12(tmp + 1, grbuf + 12, overlap + 6);
        drmp3_L3_imdct12(tmp + 2, overlap, overlap + 6);
    }
}

static void drmp3_L3_change_sign(float *grbuf)
{
    int b, i;
    for (b = 0, grbuf += 18; b < 32; b += 2, grbuf += 36)
        for (i = 1; i < 18; i += 2)
            grbuf[i] = -grbuf[i];
}

static void drmp3_L3_imdct_gr(float *grbuf, float *overlap, unsigned block_type, unsigned n_long_bands)
{
    static const float g_mdct_window[2][18] = {
        { 0.99904822f,0.99144486f,0.97629601f,0.95371695f,0.92387953f,0.88701083f,0.84339145f,0.79335334f,0.73727734f,0.04361938f,0.13052619f,0.21643961f,0.30070580f,0.38268343f,0.46174861f,0.53729961f,0.60876143f,0.67559021f },
        { 1,1,1,1,1,1,0.99144486f,0.92387953f,0.79335334f,0,0,0,0,0,0,0.13052619f,0.38268343f,0.60876143f }
    };
    if (n_long_bands)
    {
        drmp3_L3_imdct36(grbuf, overlap, g_mdct_window[0], n_long_bands);
        grbuf += 18*n_long_bands;
        overlap += 9*n_long_bands;
    }
    if (block_type == DRMP3_SHORT_BLOCK_TYPE)
        drmp3_L3_imdct_short(grbuf, overlap, 32 - n_long_bands);
    else
        drmp3_L3_imdct36(grbuf, overlap, g_mdct_window[block_type == DRMP3_STOP_BLOCK_TYPE], 32 - n_long_bands);
}

static void drmp3_L3_save_reservoir(drmp3dec *h, drmp3dec_scratch *s)
{
    int pos = (s->bs.pos + 7)/8u;
    int remains = s->bs.limit/8u - pos;
    if (remains > DRMP3_MAX_BITRESERVOIR_BYTES)
    {
        pos += remains - DRMP3_MAX_BITRESERVOIR_BYTES;
        remains = DRMP3_MAX_BITRESERVOIR_BYTES;
    }
    if (remains > 0)
    {
        DRMP3_MOVE_MEMORY(h->reserv_buf, s->maindata + pos, remains);
    }
    h->reserv = remains;
}

static int drmp3_L3_restore_reservoir(drmp3dec *h, drmp3_bs *bs, drmp3dec_scratch *s, int main_data_begin)
{
    int frame_bytes = (bs->limit - bs->pos)/8;
    int bytes_have = DRMP3_MIN(h->reserv, main_data_begin);
    DRMP3_COPY_MEMORY(s->maindata, h->reserv_buf + DRMP3_MAX(0, h->reserv - main_data_begin), DRMP3_MIN(h->reserv, main_data_begin));
    DRMP3_COPY_MEMORY(s->maindata + bytes_have, bs->buf + bs->pos/8, frame_bytes);
    drmp3_bs_init(&s->bs, s->maindata, bytes_have + frame_bytes);
    return h->reserv >= main_data_begin;
}

static void drmp3_L3_decode(drmp3dec *h, drmp3dec_scratch *s, drmp3_L3_gr_info *gr_info, int nch)
{
    int ch;

    for (ch = 0; ch < nch; ch++)
    {
        int layer3gr_limit = s->bs.pos + gr_info[ch].part_23_length;
        drmp3_L3_decode_scalefactors(h->header, s->ist_pos[ch], &s->bs, gr_info + ch, s->scf, ch);
        drmp3_L3_huffman(s->grbuf[ch], &s->bs, gr_info + ch, s->scf, layer3gr_limit);
    }

    if (DRMP3_HDR_TEST_I_STEREO(h->header))
    {
        drmp3_L3_intensity_stereo(s->grbuf[0], s->ist_pos[1], gr_info, h->header);
    } else if (DRMP3_HDR_IS_MS_STEREO(h->header))
    {
        drmp3_L3_midside_stereo(s->grbuf[0], 576);
    }

    for (ch = 0; ch < nch; ch++, gr_info++)
    {
        int aa_bands = 31;
        int n_long_bands = (gr_info->mixed_block_flag ? 2 : 0) << (int)(DRMP3_HDR_GET_MY_SAMPLE_RATE(h->header) == 2);

        if (gr_info->n_short_sfb)
        {
            aa_bands = n_long_bands - 1;
            drmp3_L3_reorder(s->grbuf[ch] + n_long_bands*18, s->syn[0], gr_info->sfbtab + gr_info->n_long_sfb);
        }

        drmp3_L3_antialias(s->grbuf[ch], aa_bands);
        drmp3_L3_imdct_gr(s->grbuf[ch], h->mdct_overlap[ch], gr_info->block_type, n_long_bands);
        drmp3_L3_change_sign(s->grbuf[ch]);
    }
}

static void drmp3d_DCT_II(float *grbuf, int n)
{
    static const float g_sec[24] = {
        10.19000816f,0.50060302f,0.50241929f,3.40760851f,0.50547093f,0.52249861f,2.05778098f,0.51544732f,0.56694406f,1.48416460f,0.53104258f,0.64682180f,1.16943991f,0.55310392f,0.78815460f,0.97256821f,0.58293498f,1.06067765f,0.83934963f,0.62250412f,1.72244716f,0.74453628f,0.67480832f,5.10114861f
    };
    int i, k = 0;
#if DRMP3_HAVE_SIMD
    if (drmp3_have_simd()) for (; k < n; k += 4)
    {
        drmp3_f4 t[4][8], *x;
        float *y = grbuf + k;

        for (x = t[0], i = 0; i < 8; i++, x++)
        {
            drmp3_f4 x0 = DRMP3_VLD(&y[i*18]);
            drmp3_f4 x1 = DRMP3_VLD(&y[(15 - i)*18]);
            drmp3_f4 x2 = DRMP3_VLD(&y[(16 + i)*18]);
            drmp3_f4 x3 = DRMP3_VLD(&y[(31 - i)*18]);
            drmp3_f4 t0 = DRMP3_VADD(x0, x3);
            drmp3_f4 t1 = DRMP3_VADD(x1, x2);
            drmp3_f4 t2 = DRMP3_VMUL_S(DRMP3_VSUB(x1, x2), g_sec[3*i + 0]);
            drmp3_f4 t3 = DRMP3_VMUL_S(DRMP3_VSUB(x0, x3), g_sec[3*i + 1]);
            x[0] = DRMP3_VADD(t0, t1);
            x[8] = DRMP3_VMUL_S(DRMP3_VSUB(t0, t1), g_sec[3*i + 2]);
            x[16] = DRMP3_VADD(t3, t2);
            x[24] = DRMP3_VMUL_S(DRMP3_VSUB(t3, t2), g_sec[3*i + 2]);
        }
        for (x = t[0], i = 0; i < 4; i++, x += 8)
        {
            drmp3_f4 x0 = x[0], x1 = x[1], x2 = x[2], x3 = x[3], x4 = x[4], x5 = x[5], x6 = x[6], x7 = x[7], xt;
            xt = DRMP3_VSUB(x0, x7); x0 = DRMP3_VADD(x0, x7);
            x7 = DRMP3_VSUB(x1, x6); x1 = DRMP3_VADD(x1, x6);
            x6 = DRMP3_VSUB(x2, x5); x2 = DRMP3_VADD(x2, x5);
            x5 = DRMP3_VSUB(x3, x4); x3 = DRMP3_VADD(x3, x4);
            x4 = DRMP3_VSUB(x0, x3); x0 = DRMP3_VADD(x0, x3);
            x3 = DRMP3_VSUB(x1, x2); x1 = DRMP3_VADD(x1, x2);
            x[0] = DRMP3_VADD(x0, x1);
            x[4] = DRMP3_VMUL_S(DRMP3_VSUB(x0, x1), 0.70710677f);
            x5 = DRMP3_VADD(x5, x6);
            x6 = DRMP3_VMUL_S(DRMP3_VADD(x6, x7), 0.70710677f);
            x7 = DRMP3_VADD(x7, xt);
            x3 = DRMP3_VMUL_S(DRMP3_VADD(x3, x4), 0.70710677f);
            x5 = DRMP3_VSUB(x5, DRMP3_VMUL_S(x7, 0.198912367f)); /* rotate by PI/8 */
            x7 = DRMP3_VADD(x7, DRMP3_VMUL_S(x5, 0.382683432f));
            x5 = DRMP3_VSUB(x5, DRMP3_VMUL_S(x7, 0.198912367f));
            x0 = DRMP3_VSUB(xt, x6); xt = DRMP3_VADD(xt, x6);
            x[1] = DRMP3_VMUL_S(DRMP3_VADD(xt, x7), 0.50979561f);
            x[2] = DRMP3_VMUL_S(DRMP3_VADD(x4, x3), 0.54119611f);
            x[3] = DRMP3_VMUL_S(DRMP3_VSUB(x0, x5), 0.60134488f);
            x[5] = DRMP3_VMUL_S(DRMP3_VADD(x0, x5), 0.89997619f);
            x[6] = DRMP3_VMUL_S(DRMP3_VSUB(x4, x3), 1.30656302f);
            x[7] = DRMP3_VMUL_S(DRMP3_VSUB(xt, x7), 2.56291556f);
        }

        if (k > n - 3)
        {
#if DRMP3_HAVE_SSE
#define DRMP3_VSAVE2(i, v) _mm_storel_pi((__m64 *)(void*)&y[i*18], v)
#else
#define DRMP3_VSAVE2(i, v) vst1_f32((float32_t *)&y[(i)*18],  vget_low_f32(v))
#endif
            for (i = 0; i < 7; i++, y += 4*18)
            {
                drmp3_f4 s = DRMP3_VADD(t[3][i], t[3][i + 1]);
                DRMP3_VSAVE2(0, t[0][i]);
                DRMP3_VSAVE2(1, DRMP3_VADD(t[2][i], s));
                DRMP3_VSAVE2(2, DRMP3_VADD(t[1][i], t[1][i + 1]));
                DRMP3_VSAVE2(3, DRMP3_VADD(t[2][1 + i], s));
            }
            DRMP3_VSAVE2(0, t[0][7]);
            DRMP3_VSAVE2(1, DRMP3_VADD(t[2][7], t[3][7]));
            DRMP3_VSAVE2(2, t[1][7]);
            DRMP3_VSAVE2(3, t[3][7]);
        } else
        {
#define DRMP3_VSAVE4(i, v) DRMP3_VSTORE(&y[(i)*18], v)
            for (i = 0; i < 7; i++, y += 4*18)
            {
                drmp3_f4 s = DRMP3_VADD(t[3][i], t[3][i + 1]);
                DRMP3_VSAVE4(0, t[0][i]);
                DRMP3_VSAVE4(1, DRMP3_VADD(t[2][i], s));
                DRMP3_VSAVE4(2, DRMP3_VADD(t[1][i], t[1][i + 1]));
                DRMP3_VSAVE4(3, DRMP3_VADD(t[2][1 + i], s));
            }
            DRMP3_VSAVE4(0, t[0][7]);
            DRMP3_VSAVE4(1, DRMP3_VADD(t[2][7], t[3][7]));
            DRMP3_VSAVE4(2, t[1][7]);
            DRMP3_VSAVE4(3, t[3][7]);
        }
    } else
#endif
#ifdef DR_MP3_ONLY_SIMD
    {} /* for HAVE_SIMD=1, MINIMP3_ONLY_SIMD=1 case we do not need non-intrinsic "else" branch */
#else
    for (; k < n; k++)
    {
        float t[4][8], *x, *y = grbuf + k;

        for (x = t[0], i = 0; i < 8; i++, x++)
        {
            float x0 = y[i*18];
            float x1 = y[(15 - i)*18];
            float x2 = y[(16 + i)*18];
            float x3 = y[(31 - i)*18];
            float t0 = x0 + x3;
            float t1 = x1 + x2;
            float t2 = (x1 - x2)*g_sec[3*i + 0];
            float t3 = (x0 - x3)*g_sec[3*i + 1];
            x[0] = t0 + t1;
            x[8] = (t0 - t1)*g_sec[3*i + 2];
            x[16] = t3 + t2;
            x[24] = (t3 - t2)*g_sec[3*i + 2];
        }
        for (x = t[0], i = 0; i < 4; i++, x += 8)
        {
            float x0 = x[0], x1 = x[1], x2 = x[2], x3 = x[3], x4 = x[4], x5 = x[5], x6 = x[6], x7 = x[7], xt;
            xt = x0 - x7; x0 += x7;
            x7 = x1 - x6; x1 += x6;
            x6 = x2 - x5; x2 += x5;
            x5 = x3 - x4; x3 += x4;
            x4 = x0 - x3; x0 += x3;
            x3 = x1 - x2; x1 += x2;
            x[0] = x0 + x1;
            x[4] = (x0 - x1)*0.70710677f;
            x5 =  x5 + x6;
            x6 = (x6 + x7)*0.70710677f;
            x7 =  x7 + xt;
            x3 = (x3 + x4)*0.70710677f;
            x5 -= x7*0.198912367f;  /* rotate by PI/8 */
            x7 += x5*0.382683432f;
            x5 -= x7*0.198912367f;
            x0 = xt - x6; xt += x6;
            x[1] = (xt + x7)*0.50979561f;
            x[2] = (x4 + x3)*0.54119611f;
            x[3] = (x0 - x5)*0.60134488f;
            x[5] = (x0 + x5)*0.89997619f;
            x[6] = (x4 - x3)*1.30656302f;
            x[7] = (xt - x7)*2.56291556f;

        }
        for (i = 0; i < 7; i++, y += 4*18)
        {
            y[0*18] = t[0][i];
            y[1*18] = t[2][i] + t[3][i] + t[3][i + 1];
            y[2*18] = t[1][i] + t[1][i + 1];
            y[3*18] = t[2][i + 1] + t[3][i] + t[3][i + 1];
        }
        y[0*18] = t[0][7];
        y[1*18] = t[2][7] + t[3][7];
        y[2*18] = t[1][7];
        y[3*18] = t[3][7];
    }
#endif
}

#ifndef DR_MP3_FLOAT_OUTPUT
typedef drmp3_int16 drmp3d_sample_t;

static drmp3_int16 drmp3d_scale_pcm(float sample)
{
    drmp3_int16 s;
#if DRMP3_HAVE_ARMV6
    drmp3_int32 s32 = (drmp3_int32)(sample + .5f);
    s32 -= (s32 < 0);
    s = (drmp3_int16)drmp3_clip_int16_arm(s32);
#else
    if (sample >=  32766.5f) return (drmp3_int16) 32767;
    if (sample <= -32767.5f) return (drmp3_int16)-32768;
    s = (drmp3_int16)(sample + .5f);
    s -= (s < 0);   /* away from zero, to be compliant */
#endif
    return s;
}
#else
typedef float drmp3d_sample_t;

static float drmp3d_scale_pcm(float sample)
{
    return sample*(1.f/32768.f);
}
#endif

static void drmp3d_synth_pair(drmp3d_sample_t *pcm, int nch, const float *z)
{
    float a;
    a  = (z[14*64] - z[    0]) * 29;
    a += (z[ 1*64] + z[13*64]) * 213;
    a += (z[12*64] - z[ 2*64]) * 459;
    a += (z[ 3*64] + z[11*64]) * 2037;
    a += (z[10*64] - z[ 4*64]) * 5153;
    a += (z[ 5*64] + z[ 9*64]) * 6574;
    a += (z[ 8*64] - z[ 6*64]) * 37489;
    a +=  z[ 7*64]             * 75038;
    pcm[0] = drmp3d_scale_pcm(a);

    z += 2;
    a  = z[14*64] * 104;
    a += z[12*64] * 1567;
    a += z[10*64] * 9727;
    a += z[ 8*64] * 64019;
    a += z[ 6*64] * -9975;
    a += z[ 4*64] * -45;
    a += z[ 2*64] * 146;
    a += z[ 0*64] * -5;
    pcm[16*nch] = drmp3d_scale_pcm(a);
}

static void drmp3d_synth(float *xl, drmp3d_sample_t *dstl, int nch, float *lins)
{
    int i;
    float *xr = xl + 576*(nch - 1);
    drmp3d_sample_t *dstr = dstl + (nch - 1);

    static const float g_win[] = {
        -1,26,-31,208,218,401,-519,2063,2000,4788,-5517,7134,5959,35640,-39336,74992,
        -1,24,-35,202,222,347,-581,2080,1952,4425,-5879,7640,5288,33791,-41176,74856,
        -1,21,-38,196,225,294,-645,2087,1893,4063,-6237,8092,4561,31947,-43006,74630,
        -1,19,-41,190,227,244,-711,2085,1822,3705,-6589,8492,3776,30112,-44821,74313,
        -1,17,-45,183,228,197,-779,2075,1739,3351,-6935,8840,2935,28289,-46617,73908,
        -1,16,-49,176,228,153,-848,2057,1644,3004,-7271,9139,2037,26482,-48390,73415,
        -2,14,-53,169,227,111,-919,2032,1535,2663,-7597,9389,1082,24694,-50137,72835,
        -2,13,-58,161,224,72,-991,2001,1414,2330,-7910,9592,70,22929,-51853,72169,
        -2,11,-63,154,221,36,-1064,1962,1280,2006,-8209,9750,-998,21189,-53534,71420,
        -2,10,-68,147,215,2,-1137,1919,1131,1692,-8491,9863,-2122,19478,-55178,70590,
        -3,9,-73,139,208,-29,-1210,1870,970,1388,-8755,9935,-3300,17799,-56778,69679,
        -3,8,-79,132,200,-57,-1283,1817,794,1095,-8998,9966,-4533,16155,-58333,68692,
        -4,7,-85,125,189,-83,-1356,1759,605,814,-9219,9959,-5818,14548,-59838,67629,
        -4,7,-91,117,177,-106,-1428,1698,402,545,-9416,9916,-7154,12980,-61289,66494,
        -5,6,-97,111,163,-127,-1498,1634,185,288,-9585,9838,-8540,11455,-62684,65290
    };
    float *zlin = lins + 15*64;
    const float *w = g_win;

    zlin[4*15]     = xl[18*16];
    zlin[4*15 + 1] = xr[18*16];
    zlin[4*15 + 2] = xl[0];
    zlin[4*15 + 3] = xr[0];

    zlin[4*31]     = xl[1 + 18*16];
    zlin[4*31 + 1] = xr[1 + 18*16];
    zlin[4*31 + 2] = xl[1];
    zlin[4*31 + 3] = xr[1];

    drmp3d_synth_pair(dstr, nch, lins + 4*15 + 1);
    drmp3d_synth_pair(dstr + 32*nch, nch, lins + 4*15 + 64 + 1);
    drmp3d_synth_pair(dstl, nch, lins + 4*15);
    drmp3d_synth_pair(dstl + 32*nch, nch, lins + 4*15 + 64);

#if DRMP3_HAVE_SIMD
    if (drmp3_have_simd()) for (i = 14; i >= 0; i--)
    {
#define DRMP3_VLOAD(k) drmp3_f4 w0 = DRMP3_VSET(*w++); drmp3_f4 w1 = DRMP3_VSET(*w++); drmp3_f4 vz = DRMP3_VLD(&zlin[4*i - 64*k]); drmp3_f4 vy = DRMP3_VLD(&zlin[4*i - 64*(15 - k)]);
#define DRMP3_V0(k) { DRMP3_VLOAD(k) b =               DRMP3_VADD(DRMP3_VMUL(vz, w1), DRMP3_VMUL(vy, w0)) ; a =               DRMP3_VSUB(DRMP3_VMUL(vz, w0), DRMP3_VMUL(vy, w1));  }
#define DRMP3_V1(k) { DRMP3_VLOAD(k) b = DRMP3_VADD(b, DRMP3_VADD(DRMP3_VMUL(vz, w1), DRMP3_VMUL(vy, w0))); a = DRMP3_VADD(a, DRMP3_VSUB(DRMP3_VMUL(vz, w0), DRMP3_VMUL(vy, w1))); }
#define DRMP3_V2(k) { DRMP3_VLOAD(k) b = DRMP3_VADD(b, DRMP3_VADD(DRMP3_VMUL(vz, w1), DRMP3_VMUL(vy, w0))); a = DRMP3_VADD(a, DRMP3_VSUB(DRMP3_VMUL(vy, w1), DRMP3_VMUL(vz, w0))); }
        drmp3_f4 a, b;
        zlin[4*i]     = xl[18*(31 - i)];
        zlin[4*i + 1] = xr[18*(31 - i)];
        zlin[4*i + 2] = xl[1 + 18*(31 - i)];
        zlin[4*i + 3] = xr[1 + 18*(31 - i)];
        zlin[4*i + 64] = xl[1 + 18*(1 + i)];
        zlin[4*i + 64 + 1] = xr[1 + 18*(1 + i)];
        zlin[4*i - 64 + 2] = xl[18*(1 + i)];
        zlin[4*i - 64 + 3] = xr[18*(1 + i)];

        DRMP3_V0(0) DRMP3_V2(1) DRMP3_V1(2) DRMP3_V2(3) DRMP3_V1(4) DRMP3_V2(5) DRMP3_V1(6) DRMP3_V2(7)

        {
#ifndef DR_MP3_FLOAT_OUTPUT
#if DRMP3_HAVE_SSE
            static const drmp3_f4 g_max = { 32767.0f, 32767.0f, 32767.0f, 32767.0f };
            static const drmp3_f4 g_min = { -32768.0f, -32768.0f, -32768.0f, -32768.0f };
            __m128i pcm8 = _mm_packs_epi32(_mm_cvtps_epi32(_mm_max_ps(_mm_min_ps(a, g_max), g_min)),
                                           _mm_cvtps_epi32(_mm_max_ps(_mm_min_ps(b, g_max), g_min)));
            dstr[(15 - i)*nch] = (drmp3_int16)_mm_extract_epi16(pcm8, 1);
            dstr[(17 + i)*nch] = (drmp3_int16)_mm_extract_epi16(pcm8, 5);
            dstl[(15 - i)*nch] = (drmp3_int16)_mm_extract_epi16(pcm8, 0);
            dstl[(17 + i)*nch] = (drmp3_int16)_mm_extract_epi16(pcm8, 4);
            dstr[(47 - i)*nch] = (drmp3_int16)_mm_extract_epi16(pcm8, 3);
            dstr[(49 + i)*nch] = (drmp3_int16)_mm_extract_epi16(pcm8, 7);
            dstl[(47 - i)*nch] = (drmp3_int16)_mm_extract_epi16(pcm8, 2);
            dstl[(49 + i)*nch] = (drmp3_int16)_mm_extract_epi16(pcm8, 6);
#else
            int16x4_t pcma, pcmb;
            a = DRMP3_VADD(a, DRMP3_VSET(0.5f));
            b = DRMP3_VADD(b, DRMP3_VSET(0.5f));
            pcma = vqmovn_s32(vqaddq_s32(vcvtq_s32_f32(a), vreinterpretq_s32_u32(vcltq_f32(a, DRMP3_VSET(0)))));
            pcmb = vqmovn_s32(vqaddq_s32(vcvtq_s32_f32(b), vreinterpretq_s32_u32(vcltq_f32(b, DRMP3_VSET(0)))));
            vst1_lane_s16(dstr + (15 - i)*nch, pcma, 1);
            vst1_lane_s16(dstr + (17 + i)*nch, pcmb, 1);
            vst1_lane_s16(dstl + (15 - i)*nch, pcma, 0);
            vst1_lane_s16(dstl + (17 + i)*nch, pcmb, 0);
            vst1_lane_s16(dstr + (47 - i)*nch, pcma, 3);
            vst1_lane_s16(dstr + (49 + i)*nch, pcmb, 3);
            vst1_lane_s16(dstl + (47 - i)*nch, pcma, 2);
            vst1_lane_s16(dstl + (49 + i)*nch, pcmb, 2);
#endif
#else
        #if DRMP3_HAVE_SSE
            static const drmp3_f4 g_scale = { 1.0f/32768.0f, 1.0f/32768.0f, 1.0f/32768.0f, 1.0f/32768.0f };
        #else
            const drmp3_f4 g_scale = vdupq_n_f32(1.0f/32768.0f);
        #endif
            a = DRMP3_VMUL(a, g_scale);
            b = DRMP3_VMUL(b, g_scale);
#if DRMP3_HAVE_SSE
            _mm_store_ss(dstr + (15 - i)*nch, _mm_shuffle_ps(a, a, _MM_SHUFFLE(1, 1, 1, 1)));
            _mm_store_ss(dstr + (17 + i)*nch, _mm_shuffle_ps(b, b, _MM_SHUFFLE(1, 1, 1, 1)));
            _mm_store_ss(dstl + (15 - i)*nch, _mm_shuffle_ps(a, a, _MM_SHUFFLE(0, 0, 0, 0)));
            _mm_store_ss(dstl + (17 + i)*nch, _mm_shuffle_ps(b, b, _MM_SHUFFLE(0, 0, 0, 0)));
            _mm_store_ss(dstr + (47 - i)*nch, _mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 3, 3, 3)));
            _mm_store_ss(dstr + (49 + i)*nch, _mm_shuffle_ps(b, b, _MM_SHUFFLE(3, 3, 3, 3)));
            _mm_store_ss(dstl + (47 - i)*nch, _mm_shuffle_ps(a, a, _MM_SHUFFLE(2, 2, 2, 2)));
            _mm_store_ss(dstl + (49 + i)*nch, _mm_shuffle_ps(b, b, _MM_SHUFFLE(2, 2, 2, 2)));
#else
            vst1q_lane_f32(dstr + (15 - i)*nch, a, 1);
            vst1q_lane_f32(dstr + (17 + i)*nch, b, 1);
            vst1q_lane_f32(dstl + (15 - i)*nch, a, 0);
            vst1q_lane_f32(dstl + (17 + i)*nch, b, 0);
            vst1q_lane_f32(dstr + (47 - i)*nch, a, 3);
            vst1q_lane_f32(dstr + (49 + i)*nch, b, 3);
            vst1q_lane_f32(dstl + (47 - i)*nch, a, 2);
            vst1q_lane_f32(dstl + (49 + i)*nch, b, 2);
#endif
#endif /* DR_MP3_FLOAT_OUTPUT */
        }
    } else
#endif
#ifdef DR_MP3_ONLY_SIMD
    {} /* for HAVE_SIMD=1, MINIMP3_ONLY_SIMD=1 case we do not need non-intrinsic "else" branch */
#else
    for (i = 14; i >= 0; i--)
    {
#define DRMP3_LOAD(k) float w0 = *w++; float w1 = *w++; float *vz = &zlin[4*i - k*64]; float *vy = &zlin[4*i - (15 - k)*64];
#define DRMP3_S0(k) { int j; DRMP3_LOAD(k); for (j = 0; j < 4; j++) b[j]  = vz[j]*w1 + vy[j]*w0, a[j]  = vz[j]*w0 - vy[j]*w1; }
#define DRMP3_S1(k) { int j; DRMP3_LOAD(k); for (j = 0; j < 4; j++) b[j] += vz[j]*w1 + vy[j]*w0, a[j] += vz[j]*w0 - vy[j]*w1; }
#define DRMP3_S2(k) { int j; DRMP3_LOAD(k); for (j = 0; j < 4; j++) b[j] += vz[j]*w1 + vy[j]*w0, a[j] += vy[j]*w1 - vz[j]*w0; }
        float a[4], b[4];

        zlin[4*i]     = xl[18*(31 - i)];
        zlin[4*i + 1] = xr[18*(31 - i)];
        zlin[4*i + 2] = xl[1 + 18*(31 - i)];
        zlin[4*i + 3] = xr[1 + 18*(31 - i)];
        zlin[4*(i + 16)]   = xl[1 + 18*(1 + i)];
        zlin[4*(i + 16) + 1] = xr[1 + 18*(1 + i)];
        zlin[4*(i - 16) + 2] = xl[18*(1 + i)];
        zlin[4*(i - 16) + 3] = xr[18*(1 + i)];

        DRMP3_S0(0) DRMP3_S2(1) DRMP3_S1(2) DRMP3_S2(3) DRMP3_S1(4) DRMP3_S2(5) DRMP3_S1(6) DRMP3_S2(7)

        dstr[(15 - i)*nch] = drmp3d_scale_pcm(a[1]);
        dstr[(17 + i)*nch] = drmp3d_scale_pcm(b[1]);
        dstl[(15 - i)*nch] = drmp3d_scale_pcm(a[0]);
        dstl[(17 + i)*nch] = drmp3d_scale_pcm(b[0]);
        dstr[(47 - i)*nch] = drmp3d_scale_pcm(a[3]);
        dstr[(49 + i)*nch] = drmp3d_scale_pcm(b[3]);
        dstl[(47 - i)*nch] = drmp3d_scale_pcm(a[2]);
        dstl[(49 + i)*nch] = drmp3d_scale_pcm(b[2]);
    }
#endif
}

static void drmp3d_synth_granule(float *qmf_state, float *grbuf, int nbands, int nch, drmp3d_sample_t *pcm, float *lins)
{
    int i;
    for (i = 0; i < nch; i++)
    {
        drmp3d_DCT_II(grbuf + 576*i, nbands);
    }

    DRMP3_COPY_MEMORY(lins, qmf_state, sizeof(float)*15*64);

    for (i = 0; i < nbands; i += 2)
    {
        drmp3d_synth(grbuf + i, pcm + 32*nch*i, nch, lins + i*64);
    }
#ifndef DR_MP3_NONSTANDARD_BUT_LOGICAL
    if (nch == 1)
    {
        for (i = 0; i < 15*64; i += 2)
        {
            qmf_state[i] = lins[nbands*64 + i];
        }
    } else
#endif
    {
        DRMP3_COPY_MEMORY(qmf_state, lins + nbands*64, sizeof(float)*15*64);
    }
}

static int drmp3d_match_frame(const drmp3_uint8 *hdr, int mp3_bytes, int frame_bytes)
{
    int i, nmatch;
    for (i = 0, nmatch = 0; nmatch < DRMP3_MAX_FRAME_SYNC_MATCHES; nmatch++)
    {
        i += drmp3_hdr_frame_bytes(hdr + i, frame_bytes) + drmp3_hdr_padding(hdr + i);
        if (i + DRMP3_HDR_SIZE > mp3_bytes)
            return nmatch > 0;
        if (!drmp3_hdr_compare(hdr, hdr + i))
            return 0;
    }
    return 1;
}

static int drmp3d_find_frame(const drmp3_uint8 *mp3, int mp3_bytes, int *free_format_bytes, int *ptr_frame_bytes)
{
    int i, k;
    for (i = 0; i < mp3_bytes - DRMP3_HDR_SIZE; i++, mp3++)
    {
        if (drmp3_hdr_valid(mp3))
        {
            int frame_bytes = drmp3_hdr_frame_bytes(mp3, *free_format_bytes);
            int frame_and_padding = frame_bytes + drmp3_hdr_padding(mp3);

            for (k = DRMP3_HDR_SIZE; !frame_bytes && k < DRMP3_MAX_FREE_FORMAT_FRAME_SIZE && i + 2*k < mp3_bytes - DRMP3_HDR_SIZE; k++)
            {
                if (drmp3_hdr_compare(mp3, mp3 + k))
                {
                    int fb = k - drmp3_hdr_padding(mp3);
                    int nextfb = fb + drmp3_hdr_padding(mp3 + k);
                    if (i + k + nextfb + DRMP3_HDR_SIZE > mp3_bytes || !drmp3_hdr_compare(mp3, mp3 + k + nextfb))
                        continue;
                    frame_and_padding = k;
                    frame_bytes = fb;
                    *free_format_bytes = fb;
                }
            }

            if ((frame_bytes && i + frame_and_padding <= mp3_bytes &&
                drmp3d_match_frame(mp3, mp3_bytes - i, frame_bytes)) ||
                (!i && frame_and_padding == mp3_bytes))
            {
                *ptr_frame_bytes = frame_and_padding;
                return i;
            }
            *free_format_bytes = 0;
        }
    }
    *ptr_frame_bytes = 0;
    return mp3_bytes;
}

DRMP3_API void drmp3dec_init(drmp3dec *dec)
{
    dec->header[0] = 0;
}

DRMP3_API int drmp3dec_decode_frame(drmp3dec *dec, const drmp3_uint8 *mp3, int mp3_bytes, void *pcm, drmp3dec_frame_info *info)
{
    int i = 0, igr, frame_size = 0, success = 1;
    const drmp3_uint8 *hdr;
    drmp3_bs bs_frame[1];

    if (mp3_bytes > 4 && dec->header[0] == 0xff && drmp3_hdr_compare(dec->header, mp3))
    {
        frame_size = drmp3_hdr_frame_bytes(mp3, dec->free_format_bytes) + drmp3_hdr_padding(mp3);
        if (frame_size != mp3_bytes && (frame_size + DRMP3_HDR_SIZE > mp3_bytes || !drmp3_hdr_compare(mp3, mp3 + frame_size)))
        {
            frame_size = 0;
        }
    }
    if (!frame_size)
    {
        DRMP3_ZERO_MEMORY(dec, sizeof(drmp3dec));
        i = drmp3d_find_frame(mp3, mp3_bytes, &dec->free_format_bytes, &frame_size);
        if (!frame_size || i + frame_size > mp3_bytes)
        {
            info->frame_bytes = i;
            return 0;
        }
    }

    hdr = mp3 + i;
    DRMP3_COPY_MEMORY(dec->header, hdr, DRMP3_HDR_SIZE);
    info->frame_bytes = i + frame_size;
    info->channels = DRMP3_HDR_IS_MONO(hdr) ? 1 : 2;
    info->sample_rate = drmp3_hdr_sample_rate_hz(hdr);
    info->layer = 4 - DRMP3_HDR_GET_LAYER(hdr);
    info->bitrate_kbps = drmp3_hdr_bitrate_kbps(hdr);

    drmp3_bs_init(bs_frame, hdr + DRMP3_HDR_SIZE, frame_size - DRMP3_HDR_SIZE);
    if (DRMP3_HDR_IS_CRC(hdr))
    {
        drmp3_bs_get_bits(bs_frame, 16);
    }

    if (info->layer == 3)
    {
        int main_data_begin = drmp3_L3_read_side_info(bs_frame, dec->scratch.gr_info, hdr);
        if (main_data_begin < 0 || bs_frame->pos > bs_frame->limit)
        {
            drmp3dec_init(dec);
            return 0;
        }
        success = drmp3_L3_restore_reservoir(dec, bs_frame, &dec->scratch, main_data_begin);
        if (success && pcm != NULL)
        {
            for (igr = 0; igr < (DRMP3_HDR_TEST_MPEG1(hdr) ? 2 : 1); igr++, pcm = DRMP3_OFFSET_PTR(pcm, sizeof(drmp3d_sample_t)*576*info->channels))
            {
                DRMP3_ZERO_MEMORY(dec->scratch.grbuf[0], 576*2*sizeof(float));
                drmp3_L3_decode(dec, &dec->scratch, dec->scratch.gr_info + igr*info->channels, info->channels);
                drmp3d_synth_granule(dec->qmf_state, dec->scratch.grbuf[0], 18, info->channels, (drmp3d_sample_t*)pcm, dec->scratch.syn[0]);
            }
        }
        drmp3_L3_save_reservoir(dec, &dec->scratch);
    } else
    {
#ifdef DR_MP3_ONLY_MP3
        return 0;
#else
        drmp3_L12_scale_info sci[1];

        if (pcm == NULL) {
            return drmp3_hdr_frame_samples(hdr);
        }

        drmp3_L12_read_scale_info(hdr, bs_frame, sci);

        DRMP3_ZERO_MEMORY(dec->scratch.grbuf[0], 576*2*sizeof(float));
        for (i = 0, igr = 0; igr < 3; igr++)
        {
            if (12 == (i += drmp3_L12_dequantize_granule(dec->scratch.grbuf[0] + i, bs_frame, sci, info->layer | 1)))
            {
                i = 0;
                drmp3_L12_apply_scf_384(sci, sci->scf + igr, dec->scratch.grbuf[0]);
                drmp3d_synth_granule(dec->qmf_state, dec->scratch.grbuf[0], 12, info->channels, (drmp3d_sample_t*)pcm, dec->scratch.syn[0]);
                DRMP3_ZERO_MEMORY(dec->scratch.grbuf[0], 576*2*sizeof(float));
                pcm = DRMP3_OFFSET_PTR(pcm, sizeof(drmp3d_sample_t)*384*info->channels);
            }
            if (bs_frame->pos > bs_frame->limit)
            {
                drmp3dec_init(dec);
                return 0;
            }
        }
#endif
    }

    return success*drmp3_hdr_frame_samples(dec->header);
}

DRMP3_API void drmp3dec_f32_to_s16(const float *in, drmp3_int16 *out, size_t num_samples)
{
    size_t i = 0;
#if DRMP3_HAVE_SIMD
    size_t aligned_count = num_samples & ~7;
    for(; i < aligned_count; i+=8)
    {
        drmp3_f4 scale = DRMP3_VSET(32768.0f);
        drmp3_f4 a = DRMP3_VMUL(DRMP3_VLD(&in[i  ]), scale);
        drmp3_f4 b = DRMP3_VMUL(DRMP3_VLD(&in[i+4]), scale);
#if DRMP3_HAVE_SSE
        drmp3_f4 s16max = DRMP3_VSET( 32767.0f);
        drmp3_f4 s16min = DRMP3_VSET(-32768.0f);
        __m128i pcm8 = _mm_packs_epi32(_mm_cvtps_epi32(_mm_max_ps(_mm_min_ps(a, s16max), s16min)),
                                        _mm_cvtps_epi32(_mm_max_ps(_mm_min_ps(b, s16max), s16min)));
        out[i  ] = (drmp3_int16)_mm_extract_epi16(pcm8, 0);
        out[i+1] = (drmp3_int16)_mm_extract_epi16(pcm8, 1);
        out[i+2] = (drmp3_int16)_mm_extract_epi16(pcm8, 2);
        out[i+3] = (drmp3_int16)_mm_extract_epi16(pcm8, 3);
        out[i+4] = (drmp3_int16)_mm_extract_epi16(pcm8, 4);
        out[i+5] = (drmp3_int16)_mm_extract_epi16(pcm8, 5);
        out[i+6] = (drmp3_int16)_mm_extract_epi16(pcm8, 6);
        out[i+7] = (drmp3_int16)_mm_extract_epi16(pcm8, 7);
#else
        int16x4_t pcma, pcmb;
        a = DRMP3_VADD(a, DRMP3_VSET(0.5f));
        b = DRMP3_VADD(b, DRMP3_VSET(0.5f));
        pcma = vqmovn_s32(vqaddq_s32(vcvtq_s32_f32(a), vreinterpretq_s32_u32(vcltq_f32(a, DRMP3_VSET(0)))));
        pcmb = vqmovn_s32(vqaddq_s32(vcvtq_s32_f32(b), vreinterpretq_s32_u32(vcltq_f32(b, DRMP3_VSET(0)))));
        vst1_lane_s16(out+i  , pcma, 0);
        vst1_lane_s16(out+i+1, pcma, 1);
        vst1_lane_s16(out+i+2, pcma, 2);
        vst1_lane_s16(out+i+3, pcma, 3);
        vst1_lane_s16(out+i+4, pcmb, 0);
        vst1_lane_s16(out+i+5, pcmb, 1);
        vst1_lane_s16(out+i+6, pcmb, 2);
        vst1_lane_s16(out+i+7, pcmb, 3);
#endif
    }
#endif
    for(; i < num_samples; i++)
    {
        float sample = in[i] * 32768.0f;
        if (sample >=  32766.5f)
            out[i] = (drmp3_int16) 32767;
        else if (sample <= -32767.5f)
            out[i] = (drmp3_int16)-32768;
        else
        {
            short s = (drmp3_int16)(sample + .5f);
            s -= (s < 0);   /* away from zero, to be compliant */
            out[i] = s;
        }
    }
}



/************************************************************************************************************************************************************

 Main Public API

 ************************************************************************************************************************************************************/
/* SIZE_MAX */
#if defined(SIZE_MAX)
    #define DRMP3_SIZE_MAX  SIZE_MAX
#else
    #if defined(_WIN64) || defined(_LP64) || defined(__LP64__)
        #define DRMP3_SIZE_MAX  ((drmp3_uint64)0xFFFFFFFFFFFFFFFF)
    #else
        #define DRMP3_SIZE_MAX  0xFFFFFFFF
    #endif
#endif
/* End SIZE_MAX */

/* Options. */
#ifndef DRMP3_SEEK_LEADING_MP3_FRAMES
#define DRMP3_SEEK_LEADING_MP3_FRAMES   2
#endif

#define DRMP3_MIN_DATA_CHUNK_SIZE   16384

/* The size in bytes of each chunk of data to read from the MP3 stream. minimp3 recommends at least 16K, but in an attempt to reduce data movement I'm making this slightly larger. */
#ifndef DRMP3_DATA_CHUNK_SIZE
#define DRMP3_DATA_CHUNK_SIZE  (DRMP3_MIN_DATA_CHUNK_SIZE*4)
#endif


#define DRMP3_COUNTOF(x)        (sizeof(x) / sizeof(x[0]))
#define DRMP3_CLAMP(x, lo, hi)  (DRMP3_MAX(lo, DRMP3_MIN(x, hi)))

#ifndef DRMP3_PI_D
#define DRMP3_PI_D    3.14159265358979323846264
#endif

#define DRMP3_DEFAULT_RESAMPLER_LPF_ORDER   2

static DRMP3_INLINE float drmp3_mix_f32(float x, float y, float a)
{
    return x*(1-a) + y*a;
}
static DRMP3_INLINE float drmp3_mix_f32_fast(float x, float y, float a)
{
    float r0 = (y - x);
    float r1 = r0*a;
    return x + r1;
    /*return x + (y - x)*a;*/
}


/*
Greatest common factor using Euclid's algorithm iteratively.
*/
static DRMP3_INLINE drmp3_uint32 drmp3_gcf_u32(drmp3_uint32 a, drmp3_uint32 b)
{
    for (;;) {
        if (b == 0) {
            break;
        } else {
            drmp3_uint32 t = a;
            a = b;
            b = t % a;
        }
    }

    return a;
}


static void* drmp3__malloc_default(size_t sz, void* pUserData)
{
    (void)pUserData;
    return DRMP3_MALLOC(sz);
}

static void* drmp3__realloc_default(void* p, size_t sz, void* pUserData)
{
    (void)pUserData;
    return DRMP3_REALLOC(p, sz);
}

static void drmp3__free_default(void* p, void* pUserData)
{
    (void)pUserData;
    DRMP3_FREE(p);
}


static void* drmp3__malloc_from_callbacks(size_t sz, const drmp3_allocation_callbacks* pAllocationCallbacks)
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

static void* drmp3__realloc_from_callbacks(void* p, size_t szNew, size_t szOld, const drmp3_allocation_callbacks* pAllocationCallbacks)
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
            DRMP3_COPY_MEMORY(p2, p, szOld);
            pAllocationCallbacks->onFree(p, pAllocationCallbacks->pUserData);
        }

        return p2;
    }

    return NULL;
}

static void drmp3__free_from_callbacks(void* p, const drmp3_allocation_callbacks* pAllocationCallbacks)
{
    if (p == NULL || pAllocationCallbacks == NULL) {
        return;
    }

    if (pAllocationCallbacks->onFree != NULL) {
        pAllocationCallbacks->onFree(p, pAllocationCallbacks->pUserData);
    }
}


static drmp3_allocation_callbacks drmp3_copy_allocation_callbacks_or_defaults(const drmp3_allocation_callbacks* pAllocationCallbacks)
{
    if (pAllocationCallbacks != NULL) {
        /* Copy. */
        return *pAllocationCallbacks;
    } else {
        /* Defaults. */
        drmp3_allocation_callbacks allocationCallbacks;
        allocationCallbacks.pUserData = NULL;
        allocationCallbacks.onMalloc  = drmp3__malloc_default;
        allocationCallbacks.onRealloc = drmp3__realloc_default;
        allocationCallbacks.onFree    = drmp3__free_default;
        return allocationCallbacks;
    }
}



static size_t drmp3__on_read(drmp3* pMP3, void* pBufferOut, size_t bytesToRead)
{
    size_t bytesRead;

    DRMP3_ASSERT(pMP3         != NULL);
    DRMP3_ASSERT(pMP3->onRead != NULL);

    /*
    Don't try reading 0 bytes from the callback. This can happen when the stream is clamped against
    ID3v1 or APE tags at the end of the stream.
    */
    if (bytesToRead == 0) {
        return 0;
    }

    bytesRead = pMP3->onRead(pMP3->pUserData, pBufferOut, bytesToRead);
    pMP3->streamCursor += bytesRead;

    return bytesRead;
}

static size_t drmp3__on_read_clamped(drmp3* pMP3, void* pBufferOut, size_t bytesToRead)
{
    DRMP3_ASSERT(pMP3         != NULL);
    DRMP3_ASSERT(pMP3->onRead != NULL);

    if (pMP3->streamLength == DRMP3_UINT64_MAX) {
        return drmp3__on_read(pMP3, pBufferOut, bytesToRead);
    } else {
        drmp3_uint64 bytesRemaining;

        bytesRemaining = (pMP3->streamLength - pMP3->streamCursor);
        if (bytesToRead >         bytesRemaining) {
            bytesToRead = (size_t)bytesRemaining;
        }
    
        return drmp3__on_read(pMP3, pBufferOut, bytesToRead);
    }
}

static drmp3_bool32 drmp3__on_seek(drmp3* pMP3, int offset, drmp3_seek_origin origin)
{
    DRMP3_ASSERT(offset >= 0);
    DRMP3_ASSERT(origin == DRMP3_SEEK_SET || origin == DRMP3_SEEK_CUR);

    if (!pMP3->onSeek(pMP3->pUserData, offset, origin)) {
        return DRMP3_FALSE;
    }

    if (origin == DRMP3_SEEK_SET) {
        pMP3->streamCursor = (drmp3_uint64)offset;
    } else{
        pMP3->streamCursor += offset;
    }

    return DRMP3_TRUE;
}

static drmp3_bool32 drmp3__on_seek_64(drmp3* pMP3, drmp3_uint64 offset, drmp3_seek_origin origin)
{
    if (offset <= 0x7FFFFFFF) {
        return drmp3__on_seek(pMP3, (int)offset, origin);
    }

    /* Getting here "offset" is too large for a 32-bit integer. We just keep seeking forward until we hit the offset. */
    if (!drmp3__on_seek(pMP3, 0x7FFFFFFF, DRMP3_SEEK_SET)) {
        return DRMP3_FALSE;
    }

    offset -= 0x7FFFFFFF;
    while (offset > 0) {
        if (offset <= 0x7FFFFFFF) {
            if (!drmp3__on_seek(pMP3, (int)offset, DRMP3_SEEK_CUR)) {
                return DRMP3_FALSE;
            }
            offset = 0;
        } else {
            if (!drmp3__on_seek(pMP3, 0x7FFFFFFF, DRMP3_SEEK_CUR)) {
                return DRMP3_FALSE;
            }
            offset -= 0x7FFFFFFF;
        }
    }

    return DRMP3_TRUE;
}

static void drmp3__on_meta(drmp3* pMP3, drmp3_metadata_type type, const void* pRawData, size_t rawDataSize)
{
    if (pMP3->onMeta) {
        drmp3_metadata metadata;

        DRMP3_ZERO_OBJECT(&metadata);
        metadata.type        = type;
        metadata.pRawData    = pRawData;
        metadata.rawDataSize = rawDataSize;

        pMP3->onMeta(pMP3->pUserDataMeta, &metadata);
    }
}


static drmp3_uint32 drmp3_decode_next_frame_ex__callbacks(drmp3* pMP3, drmp3d_sample_t* pPCMFrames, drmp3dec_frame_info* pMP3FrameInfo, const drmp3_uint8** ppMP3FrameData)
{
    drmp3_uint32 pcmFramesRead = 0;

    DRMP3_ASSERT(pMP3 != NULL);
    DRMP3_ASSERT(pMP3->onRead != NULL);

    if (pMP3->atEnd) {
        return 0;
    }

    for (;;) {
        drmp3dec_frame_info info;

        /* minimp3 recommends doing data submission in chunks of at least 16K. If we don't have at least 16K bytes available, get more. */
        if (pMP3->dataSize < DRMP3_MIN_DATA_CHUNK_SIZE) {
            size_t bytesRead;

            /* First we need to move the data down. */
            if (pMP3->pData != NULL) {
                DRMP3_MOVE_MEMORY(pMP3->pData, pMP3->pData + pMP3->dataConsumed, pMP3->dataSize);
            }

            pMP3->dataConsumed = 0;

            if (pMP3->dataCapacity < DRMP3_DATA_CHUNK_SIZE) {
                drmp3_uint8* pNewData;
                size_t newDataCap;

                newDataCap = DRMP3_DATA_CHUNK_SIZE;

                pNewData = (drmp3_uint8*)drmp3__realloc_from_callbacks(pMP3->pData, newDataCap, pMP3->dataCapacity, &pMP3->allocationCallbacks);
                if (pNewData == NULL) {
                    return 0; /* Out of memory. */
                }

                pMP3->pData = pNewData;
                pMP3->dataCapacity = newDataCap;
            }

            bytesRead = drmp3__on_read_clamped(pMP3, pMP3->pData + pMP3->dataSize, (pMP3->dataCapacity - pMP3->dataSize));
            if (bytesRead == 0) {
                if (pMP3->dataSize == 0) {
                    pMP3->atEnd = DRMP3_TRUE;
                    return 0; /* No data. */
                }
            }

            pMP3->dataSize += bytesRead;
        }

        if (pMP3->dataSize > INT_MAX) {
            pMP3->atEnd = DRMP3_TRUE;
            return 0; /* File too big. */
        }

        DRMP3_ASSERT(pMP3->pData != NULL);
        DRMP3_ASSERT(pMP3->dataCapacity > 0);

        /* Do a runtime check here to try silencing a false-positive from clang-analyzer. */
        if (pMP3->pData == NULL) {
            return 0;
        }

        pcmFramesRead = drmp3dec_decode_frame(&pMP3->decoder, pMP3->pData + pMP3->dataConsumed, (int)pMP3->dataSize, pPCMFrames, &info);    /* <-- Safe size_t -> int conversion thanks to the check above. */

        /* Consume the data. */
        pMP3->dataConsumed += (size_t)info.frame_bytes;
        pMP3->dataSize     -= (size_t)info.frame_bytes;

        /* pcmFramesRead will be equal to 0 if decoding failed. If it is zero and info.frame_bytes > 0 then we have successfully decoded the frame. */
        if (pcmFramesRead > 0) {
            pcmFramesRead = drmp3_hdr_frame_samples(pMP3->decoder.header);
            pMP3->pcmFramesConsumedInMP3Frame = 0;
            pMP3->pcmFramesRemainingInMP3Frame = pcmFramesRead;
            pMP3->mp3FrameChannels = info.channels;
            pMP3->mp3FrameSampleRate = info.sample_rate;

            if (pMP3FrameInfo != NULL) {
                *pMP3FrameInfo = info;
            }

            if (ppMP3FrameData != NULL) {
                *ppMP3FrameData = pMP3->pData + pMP3->dataConsumed - (size_t)info.frame_bytes;
            }

            break;
        } else if (info.frame_bytes == 0) {
            /* Need more data. minimp3 recommends doing data submission in 16K chunks. */
            size_t bytesRead;

            /* First we need to move the data down. */
            DRMP3_MOVE_MEMORY(pMP3->pData, pMP3->pData + pMP3->dataConsumed, pMP3->dataSize);
            pMP3->dataConsumed = 0;

            if (pMP3->dataCapacity == pMP3->dataSize) {
                /* No room. Expand. */
                drmp3_uint8* pNewData;
                size_t newDataCap;

                newDataCap = pMP3->dataCapacity + DRMP3_DATA_CHUNK_SIZE;

                pNewData = (drmp3_uint8*)drmp3__realloc_from_callbacks(pMP3->pData, newDataCap, pMP3->dataCapacity, &pMP3->allocationCallbacks);
                if (pNewData == NULL) {
                    return 0; /* Out of memory. */
                }

                pMP3->pData = pNewData;
                pMP3->dataCapacity = newDataCap;
            }

            /* Fill in a chunk. */
            bytesRead = drmp3__on_read_clamped(pMP3, pMP3->pData + pMP3->dataSize, (pMP3->dataCapacity - pMP3->dataSize));
            if (bytesRead == 0) {
                pMP3->atEnd = DRMP3_TRUE;
                return 0; /* Error reading more data. */
            }

            pMP3->dataSize += bytesRead;
        }
    };

    return pcmFramesRead;
}

static drmp3_uint32 drmp3_decode_next_frame_ex__memory(drmp3* pMP3, drmp3d_sample_t* pPCMFrames, drmp3dec_frame_info* pMP3FrameInfo, const drmp3_uint8** ppMP3FrameData)
{
    drmp3_uint32 pcmFramesRead = 0;
    drmp3dec_frame_info info;

    DRMP3_ASSERT(pMP3 != NULL);
    DRMP3_ASSERT(pMP3->memory.pData != NULL);

    if (pMP3->atEnd) {
        return 0;
    }

    for (;;) {
        pcmFramesRead = drmp3dec_decode_frame(&pMP3->decoder, pMP3->memory.pData + pMP3->memory.currentReadPos, (int)(pMP3->memory.dataSize - pMP3->memory.currentReadPos), pPCMFrames, &info);
        if (pcmFramesRead > 0) {
            pcmFramesRead = drmp3_hdr_frame_samples(pMP3->decoder.header);
            pMP3->pcmFramesConsumedInMP3Frame  = 0;
            pMP3->pcmFramesRemainingInMP3Frame = pcmFramesRead;
            pMP3->mp3FrameChannels             = info.channels;
            pMP3->mp3FrameSampleRate           = info.sample_rate;

            if (pMP3FrameInfo != NULL) {
                *pMP3FrameInfo = info;
            }

            if (ppMP3FrameData != NULL) {
                *ppMP3FrameData = pMP3->memory.pData + pMP3->memory.currentReadPos;
            }

            break;
        } else if (info.frame_bytes > 0) {
            /* No frames were read, but it looks like we skipped past one. Read the next MP3 frame. */
            pMP3->memory.currentReadPos += (size_t)info.frame_bytes;
            pMP3->streamCursor          += (size_t)info.frame_bytes;
        } else {
            /* Nothing at all was read. Abort. */
            break;
        }
    }

    /* Consume the data. */
    pMP3->memory.currentReadPos += (size_t)info.frame_bytes;
    pMP3->streamCursor          += (size_t)info.frame_bytes;

    return pcmFramesRead;
}

static drmp3_uint32 drmp3_decode_next_frame_ex(drmp3* pMP3, drmp3d_sample_t* pPCMFrames, drmp3dec_frame_info* pMP3FrameInfo, const drmp3_uint8** ppMP3FrameData)
{
    if (pMP3->memory.pData != NULL && pMP3->memory.dataSize > 0) {
        return drmp3_decode_next_frame_ex__memory(pMP3, pPCMFrames, pMP3FrameInfo, ppMP3FrameData);
    } else {
        return drmp3_decode_next_frame_ex__callbacks(pMP3, pPCMFrames, pMP3FrameInfo, ppMP3FrameData);
    }
}

static drmp3_uint32 drmp3_decode_next_frame(drmp3* pMP3)
{
    DRMP3_ASSERT(pMP3 != NULL);
    return drmp3_decode_next_frame_ex(pMP3, (drmp3d_sample_t*)pMP3->pcmFrames, NULL, NULL);
}

#if 0
static drmp3_uint32 drmp3_seek_next_frame(drmp3* pMP3)
{
    drmp3_uint32 pcmFrameCount;

    DRMP3_ASSERT(pMP3 != NULL);

    pcmFrameCount = drmp3_decode_next_frame_ex(pMP3, NULL, NULL, NULL);
    if (pcmFrameCount == 0) {
        return 0;
    }

    /* We have essentially just skipped past the frame, so just set the remaining samples to 0. */
    pMP3->currentPCMFrame             += pcmFrameCount;
    pMP3->pcmFramesConsumedInMP3Frame  = pcmFrameCount;
    pMP3->pcmFramesRemainingInMP3Frame = 0;

    return pcmFrameCount;
}
#endif

static drmp3_bool32 drmp3_init_internal(drmp3* pMP3, drmp3_read_proc onRead, drmp3_seek_proc onSeek, drmp3_tell_proc onTell, drmp3_meta_proc onMeta, void* pUserData, void* pUserDataMeta, const drmp3_allocation_callbacks* pAllocationCallbacks)
{
    drmp3dec_frame_info firstFrameInfo;
    const drmp3_uint8* pFirstFrameData;
    drmp3_uint32 firstFramePCMFrameCount;
    drmp3_uint32 detectedMP3FrameCount = 0xFFFFFFFF;

    DRMP3_ASSERT(pMP3 != NULL);
    DRMP3_ASSERT(onRead != NULL);

    /* This function assumes the output object has already been reset to 0. Do not do that here, otherwise things will break. */
    drmp3dec_init(&pMP3->decoder);

    pMP3->onRead = onRead;
    pMP3->onSeek = onSeek;
    pMP3->onMeta = onMeta;
    pMP3->pUserData = pUserData;
    pMP3->pUserDataMeta = pUserDataMeta;
    pMP3->allocationCallbacks = drmp3_copy_allocation_callbacks_or_defaults(pAllocationCallbacks);

    if (pMP3->allocationCallbacks.onFree == NULL || (pMP3->allocationCallbacks.onMalloc == NULL && pMP3->allocationCallbacks.onRealloc == NULL)) {
        return DRMP3_FALSE;    /* Invalid allocation callbacks. */
    }

    pMP3->streamCursor       = 0;
    pMP3->streamLength       = DRMP3_UINT64_MAX;
    pMP3->streamStartOffset  = 0;
    pMP3->delayInPCMFrames   = 0;
    pMP3->paddingInPCMFrames = 0;
    pMP3->totalPCMFrameCount = DRMP3_UINT64_MAX;

    /* We'll first check for any ID3v1 or APE tags. */
    #if 1
    if (onSeek != NULL && onTell != NULL) {
        if (onSeek(pUserData, 0, DRMP3_SEEK_END)) {
            drmp3_int64 streamLen;
            int streamEndOffset = 0;
        
            /* First get the length of the stream. We need this so we can ensure the stream is big enough to store the tags. */
            if (onTell(pUserData, &streamLen)) {
                /* ID3v1 */
                if (streamLen > 128) {
                    char id3[3];
                    if (onSeek(pUserData, streamEndOffset - 128, DRMP3_SEEK_END)) {
                        if (onRead(pUserData, id3, 3) == 3 && id3[0] == 'T' && id3[1] == 'A' && id3[2] == 'G') {
                            /* We have an ID3v1 tag. */
                            streamEndOffset -= 128;
                            streamLen       -= 128;

                            /* Fire a metadata callback for the TAG data. */
                            if (onMeta != NULL) {
                                drmp3_uint8 tag[128];
                                tag[0] = 'T'; tag[1] = 'A'; tag[2] = 'G';

                                if (onRead(pUserData, tag + 3, 125) == 125) {
                                    drmp3__on_meta(pMP3, DRMP3_METADATA_TYPE_ID3V1, tag, 128);
                                }
                            }
                        } else {
                            /* No ID3v1 tag. */
                        }
                    } else {
                        /* Failed to seek to the ID3v1 tag. */
                    }
                } else {
                    /* Stream too short. No ID3v1 tag. */
                }

                /* APE */
                if (streamLen > 32) {
                    char ape[32];   /* The footer. */
                    if (onSeek(pUserData, streamEndOffset - 32, DRMP3_SEEK_END)) {
                        if (onRead(pUserData, ape, 32) == 32 && ape[0] == 'A' && ape[1] == 'P' && ape[2] == 'E' && ape[3] == 'T' && ape[4] == 'A' && ape[5] == 'G' && ape[6] == 'E' && ape[7] == 'X') {
                            /* We have an APE tag. */
                            drmp3_uint32 tagSize =
                                ((drmp3_uint32)ape[24] << 0)  |
                                ((drmp3_uint32)ape[25] << 8)  |
                                ((drmp3_uint32)ape[26] << 16) |
                                ((drmp3_uint32)ape[27] << 24);

                            if (32 + tagSize < streamLen) {
                                streamEndOffset -= 32 + tagSize;
                                streamLen       -= 32 + tagSize;
                                
                                /* Fire a metadata callback for the APE data. Must include both the main content and footer. */
                                if (onMeta != NULL) {
                                    /* We first need to seek to the start of the APE tag. */
                                    if (onSeek(pUserData, streamEndOffset, DRMP3_SEEK_END)) {
                                        size_t apeTagSize = (size_t)tagSize + 32;
                                        drmp3_uint8* pTagData = (drmp3_uint8*)drmp3_malloc(apeTagSize, pAllocationCallbacks);
                                        if (pTagData != NULL) {
                                            if (onRead(pUserData, pTagData, apeTagSize) == apeTagSize) {
                                                drmp3__on_meta(pMP3, DRMP3_METADATA_TYPE_APE, pTagData, apeTagSize);
                                            }

                                            drmp3_free(pTagData, pAllocationCallbacks);
                                        }
                                    }
                                }
                            } else {
                                /* The tag size is larger than the stream. Invalid APE tag. */
                            }
                        }
                    }
                } else {
                    /* Stream too short. No APE tag. */
                }

                /* Seek back to the start. */
                if (!onSeek(pUserData, 0, DRMP3_SEEK_SET)) {
                    return DRMP3_FALSE; /* Failed to seek back to the start. */
                }

                pMP3->streamLength = (drmp3_uint64)streamLen;

                if (pMP3->memory.pData != NULL) {
                    pMP3->memory.dataSize = (size_t)pMP3->streamLength;
                }
            } else {
                /* Failed to get the length of the stream. ID3v1 and APE tags cannot be skipped. */
                if (!onSeek(pUserData, 0, DRMP3_SEEK_SET)) {
                    return DRMP3_FALSE; /* Failed to seek back to the start. */
                }
            }
        } else {
            /* Failed to seek to the end. Cannot skip ID3v1 or APE tags. */
        }
    } else {
        /* No onSeek or onTell callback. Cannot skip ID3v1 or APE tags. */
    }
    #endif


    /* ID3v2 tags */
    #if 1
    {
        char header[10];
        if (onRead(pUserData, header, 10) == 10) {
            if (header[0] == 'I' && header[1] == 'D' && header[2] == '3') {
                drmp3_uint32 tagSize =
                    (((drmp3_uint32)header[6] & 0x7F) << 21) |
                    (((drmp3_uint32)header[7] & 0x7F) << 14) |
                    (((drmp3_uint32)header[8] & 0x7F) << 7)  |
                    (((drmp3_uint32)header[9] & 0x7F) << 0);

                /* Account for the footer. */
                if (header[5] & 0x10) {
                    tagSize += 10;
                }

                /* Read the tag content and fire a metadata callback. */
                if (onMeta != NULL) {
                    size_t tagSizeWithHeader = 10 + tagSize;
                    drmp3_uint8* pTagData = (drmp3_uint8*)drmp3_malloc(tagSizeWithHeader, pAllocationCallbacks);
                    if (pTagData != NULL) {
                        DRMP3_COPY_MEMORY(pTagData, header, 10);

                        if (onRead(pUserData, pTagData + 10, tagSize) == tagSize) {
                            drmp3__on_meta(pMP3, DRMP3_METADATA_TYPE_ID3V2, pTagData, tagSizeWithHeader);
                        }

                        drmp3_free(pTagData, pAllocationCallbacks);
                    }
                } else {
                    /* Don't have a metadata callback, so just skip the tag. */
                    if (onSeek != NULL) {
                        if (!onSeek(pUserData, tagSize, DRMP3_SEEK_CUR)) {
                            return DRMP3_FALSE; /* Failed to seek past the ID3v2 tag. */
                        }
                    } else {
                        /* Don't have a seek callback. Read and discard. */
                        char discard[1024];

                        while (tagSize > 0) {
                            size_t bytesToRead = tagSize;
                            if (bytesToRead > sizeof(discard)) {
                                bytesToRead = sizeof(discard);
                            }

                            if (onRead(pUserData, discard, bytesToRead) != bytesToRead) {
                                return DRMP3_FALSE; /* Failed to read data. */
                            }

                            tagSize -= (drmp3_uint32)bytesToRead;
                        }
                    }
                }

                pMP3->streamStartOffset += 10 + tagSize;    /* +10 for the header. */
                pMP3->streamCursor = pMP3->streamStartOffset;
            } else {
                /* Not an ID3v2 tag. Seek back to the start. */
                if (onSeek != NULL) {
                    if (!onSeek(pUserData, 0, DRMP3_SEEK_SET)) {
                        return DRMP3_FALSE; /* Failed to seek back to the start. */
                    }
                } else {
                    /* Don't have a seek callback to move backwards. We'll just fall through and let the decoding process re-sync. The ideal solution here would be to read into the cache. */

                    /*
                    TODO: Copy the header into the cache. Will need to allocate space. See drmp3_decode_next_frame_ex__callbacks. There is not need
                    to handle the memory case because that will always have a seek implementation and will never hit this code path.
                    */
                }
            }
        } else {
            /* Failed to read the header. We can return false here. If we couldn't read 10 bytes there's no way we'll have a valid MP3 stream. */
            return DRMP3_FALSE;
        }
    }
    #endif

    /*
    Decode the first frame to confirm that it is indeed a valid MP3 stream. Note that it's possible the first frame
    is actually a Xing/LAME/VBRI header. If this is the case we need to skip over it.
    */
    firstFramePCMFrameCount = drmp3_decode_next_frame_ex(pMP3, (drmp3d_sample_t*)pMP3->pcmFrames, &firstFrameInfo, &pFirstFrameData);
    if (firstFramePCMFrameCount > 0) {
        DRMP3_ASSERT(pFirstFrameData != NULL);

        /*
        It might be a header. If so, we need to clear out the cached PCM frames in order to trigger a reload of fresh
        data when decoding starts. We can assume all validation has already been performed to check if this is a valid
        MP3 frame and that there is more than 0 bytes making up the frame.

        We're going to be basing this parsing code off the minimp3_ex implementation.
        */
        #if 1
        DRMP3_ASSERT(firstFrameInfo.frame_bytes > 0);
        {
            drmp3_bs bs;
            drmp3_L3_gr_info grInfo[4];
            const drmp3_uint8* pTagData = pFirstFrameData;

            drmp3_bs_init(&bs, pFirstFrameData + DRMP3_HDR_SIZE, firstFrameInfo.frame_bytes - DRMP3_HDR_SIZE);

            if (DRMP3_HDR_IS_CRC(pFirstFrameData)) {
                drmp3_bs_get_bits(&bs, 16); /* CRC. */
            }

            if (drmp3_L3_read_side_info(&bs, grInfo, pFirstFrameData) >= 0) {
                drmp3_bool32 isXing = DRMP3_FALSE;
                drmp3_bool32 isInfo = DRMP3_FALSE;
                const drmp3_uint8* pTagDataBeg;

                pTagDataBeg = pFirstFrameData + DRMP3_HDR_SIZE + (bs.pos/8);
                pTagData    = pTagDataBeg;

                /* Check for both "Xing" and "Info" identifiers. */
                isXing = (pTagData[0] == 'X' && pTagData[1] == 'i' && pTagData[2] == 'n' && pTagData[3] == 'g');
                isInfo = (pTagData[0] == 'I' && pTagData[1] == 'n' && pTagData[2] == 'f' && pTagData[3] == 'o');

                if (isXing || isInfo) {
                    drmp3_uint32 bytes = 0;
                    drmp3_uint32 flags = pTagData[7];

                    pTagData += 8;  /* Skip past the ID and flags. */

                    if (flags & 0x01) { /* FRAMES flag. */
                        detectedMP3FrameCount = (drmp3_uint32)pTagData[0] << 24 | (drmp3_uint32)pTagData[1] << 16 | (drmp3_uint32)pTagData[2] << 8 | (drmp3_uint32)pTagData[3];
                        pTagData += 4;
                    }

                    if (flags & 0x02) { /* BYTES flag. */
                        bytes  = (drmp3_uint32)pTagData[0] << 24 | (drmp3_uint32)pTagData[1] << 16 | (drmp3_uint32)pTagData[2] << 8 | (drmp3_uint32)pTagData[3];
                        (void)bytes;    /* <-- Just to silence a warning about `bytes` being assigned but unused. Want to leave this here in case I want to make use of it later. */
                        pTagData += 4;
                    }

                    if (flags & 0x04) { /* TOC flag. */
                        /* TODO: Extract and bind seek points. */
                        pTagData += 100;
                    }

                    if (flags & 0x08) { /* SCALE flag. */
                        pTagData += 4;
                    }

                    /* At this point we're done with the Xing/Info header. Now we can look at the LAME data. */
                    if (pTagData[0]) {
                        pTagData += 21;

                        if (pTagData - pFirstFrameData + 14 < firstFrameInfo.frame_bytes) {
                            int delayInPCMFrames;
                            int paddingInPCMFrames;

                            delayInPCMFrames   = (( (drmp3_uint32)pTagData[0]        << 4) | ((drmp3_uint32)pTagData[1] >> 4)) + (528 + 1);
                            paddingInPCMFrames = ((((drmp3_uint32)pTagData[1] & 0xF) << 8) | ((drmp3_uint32)pTagData[2]     )) - (528 + 1);
                            if (paddingInPCMFrames < 0) {
                                paddingInPCMFrames = 0; /* Padding cannot be negative. Probably a malformed file. Ignore. */
                            }
                            
                            pMP3->delayInPCMFrames   = (drmp3_uint32)delayInPCMFrames;
                            pMP3->paddingInPCMFrames = (drmp3_uint32)paddingInPCMFrames;
                        }
                    }

                    /*
                    My understanding is that if the "Xing" header is present we can consider this to be a VBR stream and if the "Info" header is
                    present it's a CBR stream. If this is not the case let me know! I'm just tracking this for the time being in case I want to
                    look at doing some CBR optimizations later on, such as faster seeking.
                    */
                    if (isXing) {
                        pMP3->isVBR = DRMP3_TRUE;
                    } else if (isInfo) {
                        pMP3->isCBR = DRMP3_TRUE;
                    }

                    /* Post the raw data of the tag to the metadata callback. */
                    if (onMeta != NULL) {
                        drmp3_metadata_type metadataType = isXing ? DRMP3_METADATA_TYPE_XING : DRMP3_METADATA_TYPE_VBRI;
                        size_t tagDataSize;
                    
                        tagDataSize  = (size_t)firstFrameInfo.frame_bytes;
                        tagDataSize -= (size_t)(pTagDataBeg - pFirstFrameData);

                        drmp3__on_meta(pMP3, metadataType, pTagDataBeg, tagDataSize);
                    }

                    /* Since this was identified as a tag, we don't want to treat it as audio. We need to clear out the PCM cache. */
                    pMP3->pcmFramesRemainingInMP3Frame = 0;

                    /* The start offset needs to be moved to the end of this frame so it's not included in any audio processing after seeking. */
                    pMP3->streamStartOffset += (drmp3_uint32)(firstFrameInfo.frame_bytes);
                    pMP3->streamCursor = pMP3->streamStartOffset;

                    /*
                    The internal decoder needs to be reset to clear out any state. If we don't reset this state, it's possible for
                    there to be inconsistencies in the number of samples read when reading to the end of the stream depending on
                    whether or not the caller seeks to the start of the stream.
                    */
                    drmp3dec_init(&pMP3->decoder);
                }
            } else {
                /* Failed to read the side info. */
            }
        }
        #endif
    } else {
        /* Not a valid MP3 stream. */
        drmp3__free_from_callbacks(pMP3->pData, &pMP3->allocationCallbacks);    /* The call above may have allocated memory. Need to make sure it's freed before aborting. */
        return DRMP3_FALSE;
    }

    if (detectedMP3FrameCount != 0xFFFFFFFF) {
        pMP3->totalPCMFrameCount = detectedMP3FrameCount * firstFramePCMFrameCount;
    }

    pMP3->channels   = pMP3->mp3FrameChannels;
    pMP3->sampleRate = pMP3->mp3FrameSampleRate;

    return DRMP3_TRUE;
}

DRMP3_API drmp3_bool32 drmp3_init(drmp3* pMP3, drmp3_read_proc onRead, drmp3_seek_proc onSeek, drmp3_tell_proc onTell, drmp3_meta_proc onMeta, void* pUserData, const drmp3_allocation_callbacks* pAllocationCallbacks)
{
    if (pMP3 == NULL || onRead == NULL) {
        return DRMP3_FALSE;
    }

    DRMP3_ZERO_OBJECT(pMP3);
    return drmp3_init_internal(pMP3, onRead, onSeek, onTell, onMeta, pUserData, pUserData, pAllocationCallbacks);
}


static size_t drmp3__on_read_memory(void* pUserData, void* pBufferOut, size_t bytesToRead)
{
    drmp3* pMP3 = (drmp3*)pUserData;
    size_t bytesRemaining;

    DRMP3_ASSERT(pMP3 != NULL);
    DRMP3_ASSERT(pMP3->memory.dataSize >= pMP3->memory.currentReadPos);

    bytesRemaining = pMP3->memory.dataSize - pMP3->memory.currentReadPos;
    if (bytesToRead > bytesRemaining) {
        bytesToRead = bytesRemaining;
    }

    if (bytesToRead > 0) {
        DRMP3_COPY_MEMORY(pBufferOut, pMP3->memory.pData + pMP3->memory.currentReadPos, bytesToRead);
        pMP3->memory.currentReadPos += bytesToRead;
    }

    return bytesToRead;
}

static drmp3_bool32 drmp3__on_seek_memory(void* pUserData, int byteOffset, drmp3_seek_origin origin)
{
    drmp3* pMP3 = (drmp3*)pUserData;
    drmp3_int64 newCursor;

    DRMP3_ASSERT(pMP3 != NULL);

    newCursor = pMP3->memory.currentReadPos;

    if (origin == DRMP3_SEEK_SET) {
        newCursor = 0;
    } else if (origin == DRMP3_SEEK_CUR) {
        newCursor = (drmp3_int64)pMP3->memory.currentReadPos;
    } else if (origin == DRMP3_SEEK_END) {
        newCursor = (drmp3_int64)pMP3->memory.dataSize;
    } else {
        DRMP3_ASSERT(!"Invalid seek origin");
        return DRMP3_FALSE;
    }

    newCursor += byteOffset;

    if (newCursor < 0) {
        return DRMP3_FALSE;  /* Trying to seek prior to the start of the buffer. */
    }
    if ((size_t)newCursor > pMP3->memory.dataSize) {
        return DRMP3_FALSE;  /* Trying to seek beyond the end of the buffer. */
    }

    pMP3->memory.currentReadPos = (size_t)newCursor;

    return DRMP3_TRUE;
}

static drmp3_bool32 drmp3__on_tell_memory(void* pUserData, drmp3_int64* pCursor)
{
    drmp3* pMP3 = (drmp3*)pUserData;

    DRMP3_ASSERT(pMP3 != NULL);
    DRMP3_ASSERT(pCursor != NULL);

    *pCursor = (drmp3_int64)pMP3->memory.currentReadPos;
    return DRMP3_TRUE;
}

DRMP3_API drmp3_bool32 drmp3_init_memory_with_metadata(drmp3* pMP3, const void* pData, size_t dataSize, drmp3_meta_proc onMeta, void* pUserDataMeta, const drmp3_allocation_callbacks* pAllocationCallbacks)
{
    drmp3_bool32 result;

    if (pMP3 == NULL) {
        return DRMP3_FALSE;
    }

    DRMP3_ZERO_OBJECT(pMP3);

    if (pData == NULL || dataSize == 0) {
        return DRMP3_FALSE;
    }

    pMP3->memory.pData = (const drmp3_uint8*)pData;
    pMP3->memory.dataSize = dataSize;
    pMP3->memory.currentReadPos = 0;

    result = drmp3_init_internal(pMP3, drmp3__on_read_memory, drmp3__on_seek_memory, drmp3__on_tell_memory, onMeta, pMP3, pUserDataMeta, pAllocationCallbacks);
    if (result == DRMP3_FALSE) {
        return DRMP3_FALSE;
    }

    /* Adjust the length of the memory stream to account for ID3v1 and APE tags. */
    if (pMP3->streamLength <= (drmp3_uint64)DRMP3_SIZE_MAX) {
        pMP3->memory.dataSize = (size_t)pMP3->streamLength; /* Safe cast. */
    }

    if (pMP3->streamStartOffset > (drmp3_uint64)DRMP3_SIZE_MAX) {
        return DRMP3_FALSE; /* Tags too big. */
    }

    return DRMP3_TRUE;
}

DRMP3_API drmp3_bool32 drmp3_init_memory(drmp3* pMP3, const void* pData, size_t dataSize, const drmp3_allocation_callbacks* pAllocationCallbacks)
{
    return drmp3_init_memory_with_metadata(pMP3, pData, dataSize, NULL, NULL, pAllocationCallbacks);
}


#ifndef DR_MP3_NO_STDIO
#include <stdio.h>
#include <wchar.h>      /* For wcslen(), wcsrtombs() */

/* Errno */
/* drmp3_result_from_errno() is only used inside DR_MP3_NO_STDIO for now. Move this out if it's ever used elsewhere. */
#include <errno.h>
static drmp3_result drmp3_result_from_errno(int e)
{
    switch (e)
    {
        case 0: return DRMP3_SUCCESS;
    #ifdef EPERM
        case EPERM: return DRMP3_INVALID_OPERATION;
    #endif
    #ifdef ENOENT
        case ENOENT: return DRMP3_DOES_NOT_EXIST;
    #endif
    #ifdef ESRCH
        case ESRCH: return DRMP3_DOES_NOT_EXIST;
    #endif
    #ifdef EINTR
        case EINTR: return DRMP3_INTERRUPT;
    #endif
    #ifdef EIO
        case EIO: return DRMP3_IO_ERROR;
    #endif
    #ifdef ENXIO
        case ENXIO: return DRMP3_DOES_NOT_EXIST;
    #endif
    #ifdef E2BIG
        case E2BIG: return DRMP3_INVALID_ARGS;
    #endif
    #ifdef ENOEXEC
        case ENOEXEC: return DRMP3_INVALID_FILE;
    #endif
    #ifdef EBADF
        case EBADF: return DRMP3_INVALID_FILE;
    #endif
    #ifdef ECHILD
        case ECHILD: return DRMP3_ERROR;
    #endif
    #ifdef EAGAIN
        case EAGAIN: return DRMP3_UNAVAILABLE;
    #endif
    #ifdef ENOMEM
        case ENOMEM: return DRMP3_OUT_OF_MEMORY;
    #endif
    #ifdef EACCES
        case EACCES: return DRMP3_ACCESS_DENIED;
    #endif
    #ifdef EFAULT
        case EFAULT: return DRMP3_BAD_ADDRESS;
    #endif
    #ifdef ENOTBLK
        case ENOTBLK: return DRMP3_ERROR;
    #endif
    #ifdef EBUSY
        case EBUSY: return DRMP3_BUSY;
    #endif
    #ifdef EEXIST
        case EEXIST: return DRMP3_ALREADY_EXISTS;
    #endif
    #ifdef EXDEV
        case EXDEV: return DRMP3_ERROR;
    #endif
    #ifdef ENODEV
        case ENODEV: return DRMP3_DOES_NOT_EXIST;
    #endif
    #ifdef ENOTDIR
        case ENOTDIR: return DRMP3_NOT_DIRECTORY;
    #endif
    #ifdef EISDIR
        case EISDIR: return DRMP3_IS_DIRECTORY;
    #endif
    #ifdef EINVAL
        case EINVAL: return DRMP3_INVALID_ARGS;
    #endif
    #ifdef ENFILE
        case ENFILE: return DRMP3_TOO_MANY_OPEN_FILES;
    #endif
    #ifdef EMFILE
        case EMFILE: return DRMP3_TOO_MANY_OPEN_FILES;
    #endif
    #ifdef ENOTTY
        case ENOTTY: return DRMP3_INVALID_OPERATION;
    #endif
    #ifdef ETXTBSY
        case ETXTBSY: return DRMP3_BUSY;
    #endif
    #ifdef EFBIG
        case EFBIG: return DRMP3_TOO_BIG;
    #endif
    #ifdef ENOSPC
        case ENOSPC: return DRMP3_NO_SPACE;
    #endif
    #ifdef ESPIPE
        case ESPIPE: return DRMP3_BAD_SEEK;
    #endif
    #ifdef EROFS
        case EROFS: return DRMP3_ACCESS_DENIED;
    #endif
    #ifdef EMLINK
        case EMLINK: return DRMP3_TOO_MANY_LINKS;
    #endif
    #ifdef EPIPE
        case EPIPE: return DRMP3_BAD_PIPE;
    #endif
    #ifdef EDOM
        case EDOM: return DRMP3_OUT_OF_RANGE;
    #endif
    #ifdef ERANGE
        case ERANGE: return DRMP3_OUT_OF_RANGE;
    #endif
    #ifdef EDEADLK
        case EDEADLK: return DRMP3_DEADLOCK;
    #endif
    #ifdef ENAMETOOLONG
        case ENAMETOOLONG: return DRMP3_PATH_TOO_LONG;
    #endif
    #ifdef ENOLCK
        case ENOLCK: return DRMP3_ERROR;
    #endif
    #ifdef ENOSYS
        case ENOSYS: return DRMP3_NOT_IMPLEMENTED;
    #endif
    #if defined(ENOTEMPTY) && ENOTEMPTY != EEXIST   /* In AIX, ENOTEMPTY and EEXIST use the same value. */
        case ENOTEMPTY: return DRMP3_DIRECTORY_NOT_EMPTY;
    #endif
    #ifdef ELOOP
        case ELOOP: return DRMP3_TOO_MANY_LINKS;
    #endif
    #ifdef ENOMSG
        case ENOMSG: return DRMP3_NO_MESSAGE;
    #endif
    #ifdef EIDRM
        case EIDRM: return DRMP3_ERROR;
    #endif
    #ifdef ECHRNG
        case ECHRNG: return DRMP3_ERROR;
    #endif
    #ifdef EL2NSYNC
        case EL2NSYNC: return DRMP3_ERROR;
    #endif
    #ifdef EL3HLT
        case EL3HLT: return DRMP3_ERROR;
    #endif
    #ifdef EL3RST
        case EL3RST: return DRMP3_ERROR;
    #endif
    #ifdef ELNRNG
        case ELNRNG: return DRMP3_OUT_OF_RANGE;
    #endif
    #ifdef EUNATCH
        case EUNATCH: return DRMP3_ERROR;
    #endif
    #ifdef ENOCSI
        case ENOCSI: return DRMP3_ERROR;
    #endif
    #ifdef EL2HLT
        case EL2HLT: return DRMP3_ERROR;
    #endif
    #ifdef EBADE
        case EBADE: return DRMP3_ERROR;
    #endif
    #ifdef EBADR
        case EBADR: return DRMP3_ERROR;
    #endif
    #ifdef EXFULL
        case EXFULL: return DRMP3_ERROR;
    #endif
    #ifdef ENOANO
        case ENOANO: return DRMP3_ERROR;
    #endif
    #ifdef EBADRQC
        case EBADRQC: return DRMP3_ERROR;
    #endif
    #ifdef EBADSLT
        case EBADSLT: return DRMP3_ERROR;
    #endif
    #ifdef EBFONT
        case EBFONT: return DRMP3_INVALID_FILE;
    #endif
    #ifdef ENOSTR
        case ENOSTR: return DRMP3_ERROR;
    #endif
    #ifdef ENODATA
        case ENODATA: return DRMP3_NO_DATA_AVAILABLE;
    #endif
    #ifdef ETIME
        case ETIME: return DRMP3_TIMEOUT;
    #endif
    #ifdef ENOSR
        case ENOSR: return DRMP3_NO_DATA_AVAILABLE;
    #endif
    #ifdef ENONET
        case ENONET: return DRMP3_NO_NETWORK;
    #endif
    #ifdef ENOPKG
        case ENOPKG: return DRMP3_ERROR;
    #endif
    #ifdef EREMOTE
        case EREMOTE: return DRMP3_ERROR;
    #endif
    #ifdef ENOLINK
        case ENOLINK: return DRMP3_ERROR;
    #endif
    #ifdef EADV
        case EADV: return DRMP3_ERROR;
    #endif
    #ifdef ESRMNT
        case ESRMNT: return DRMP3_ERROR;
    #endif
    #ifdef ECOMM
        case ECOMM: return DRMP3_ERROR;
    #endif
    #ifdef EPROTO
        case EPROTO: return DRMP3_ERROR;
    #endif
    #ifdef EMULTIHOP
        case EMULTIHOP: return DRMP3_ERROR;
    #endif
    #ifdef EDOTDOT
        case EDOTDOT: return DRMP3_ERROR;
    #endif
    #ifdef EBADMSG
        case EBADMSG: return DRMP3_BAD_MESSAGE;
    #endif
    #ifdef EOVERFLOW
        case EOVERFLOW: return DRMP3_TOO_BIG;
    #endif
    #ifdef ENOTUNIQ
        case ENOTUNIQ: return DRMP3_NOT_UNIQUE;
    #endif
    #ifdef EBADFD
        case EBADFD: return DRMP3_ERROR;
    #endif
    #ifdef EREMCHG
        case EREMCHG: return DRMP3_ERROR;
    #endif
    #ifdef ELIBACC
        case ELIBACC: return DRMP3_ACCESS_DENIED;
    #endif
    #ifdef ELIBBAD
        case ELIBBAD: return DRMP3_INVALID_FILE;
    #endif
    #ifdef ELIBSCN
        case ELIBSCN: return DRMP3_INVALID_FILE;
    #endif
    #ifdef ELIBMAX
        case ELIBMAX: return DRMP3_ERROR;
    #endif
    #ifdef ELIBEXEC
        case ELIBEXEC: return DRMP3_ERROR;
    #endif
    #ifdef EILSEQ
        case EILSEQ: return DRMP3_INVALID_DATA;
    #endif
    #ifdef ERESTART
        case ERESTART: return DRMP3_ERROR;
    #endif
    #ifdef ESTRPIPE
        case ESTRPIPE: return DRMP3_ERROR;
    #endif
    #ifdef EUSERS
        case EUSERS: return DRMP3_ERROR;
    #endif
    #ifdef ENOTSOCK
        case ENOTSOCK: return DRMP3_NOT_SOCKET;
    #endif
    #ifdef EDESTADDRREQ
        case EDESTADDRREQ: return DRMP3_NO_ADDRESS;
    #endif
    #ifdef EMSGSIZE
        case EMSGSIZE: return DRMP3_TOO_BIG;
    #endif
    #ifdef EPROTOTYPE
        case EPROTOTYPE: return DRMP3_BAD_PROTOCOL;
    #endif
    #ifdef ENOPROTOOPT
        case ENOPROTOOPT: return DRMP3_PROTOCOL_UNAVAILABLE;
    #endif
    #ifdef EPROTONOSUPPORT
        case EPROTONOSUPPORT: return DRMP3_PROTOCOL_NOT_SUPPORTED;
    #endif
    #ifdef ESOCKTNOSUPPORT
        case ESOCKTNOSUPPORT: return DRMP3_SOCKET_NOT_SUPPORTED;
    #endif
    #ifdef EOPNOTSUPP
        case EOPNOTSUPP: return DRMP3_INVALID_OPERATION;
    #endif
    #ifdef EPFNOSUPPORT
        case EPFNOSUPPORT: return DRMP3_PROTOCOL_FAMILY_NOT_SUPPORTED;
    #endif
    #ifdef EAFNOSUPPORT
        case EAFNOSUPPORT: return DRMP3_ADDRESS_FAMILY_NOT_SUPPORTED;
    #endif
    #ifdef EADDRINUSE
        case EADDRINUSE: return DRMP3_ALREADY_IN_USE;
    #endif
    #ifdef EADDRNOTAVAIL
        case EADDRNOTAVAIL: return DRMP3_ERROR;
    #endif
    #ifdef ENETDOWN
        case ENETDOWN: return DRMP3_NO_NETWORK;
    #endif
    #ifdef ENETUNREACH
        case ENETUNREACH: return DRMP3_NO_NETWORK;
    #endif
    #ifdef ENETRESET
        case ENETRESET: return DRMP3_NO_NETWORK;
    #endif
    #ifdef ECONNABORTED
        case ECONNABORTED: return DRMP3_NO_NETWORK;
    #endif
    #ifdef ECONNRESET
        case ECONNRESET: return DRMP3_CONNECTION_RESET;
    #endif
    #ifdef ENOBUFS
        case ENOBUFS: return DRMP3_NO_SPACE;
    #endif
    #ifdef EISCONN
        case EISCONN: return DRMP3_ALREADY_CONNECTED;
    #endif
    #ifdef ENOTCONN
        case ENOTCONN: return DRMP3_NOT_CONNECTED;
    #endif
    #ifdef ESHUTDOWN
        case ESHUTDOWN: return DRMP3_ERROR;
    #endif
    #ifdef ETOOMANYREFS
        case ETOOMANYREFS: return DRMP3_ERROR;
    #endif
    #ifdef ETIMEDOUT
        case ETIMEDOUT: return DRMP3_TIMEOUT;
    #endif
    #ifdef ECONNREFUSED
        case ECONNREFUSED: return DRMP3_CONNECTION_REFUSED;
    #endif
    #ifdef EHOSTDOWN
        case EHOSTDOWN: return DRMP3_NO_HOST;
    #endif
    #ifdef EHOSTUNREACH
        case EHOSTUNREACH: return DRMP3_NO_HOST;
    #endif
    #ifdef EALREADY
        case EALREADY: return DRMP3_IN_PROGRESS;
    #endif
    #ifdef EINPROGRESS
        case EINPROGRESS: return DRMP3_IN_PROGRESS;
    #endif
    #ifdef ESTALE
        case ESTALE: return DRMP3_INVALID_FILE;
    #endif
    #ifdef EUCLEAN
        case EUCLEAN: return DRMP3_ERROR;
    #endif
    #ifdef ENOTNAM
        case ENOTNAM: return DRMP3_ERROR;
    #endif
    #ifdef ENAVAIL
        case ENAVAIL: return DRMP3_ERROR;
    #endif
    #ifdef EISNAM
        case EISNAM: return DRMP3_ERROR;
    #endif
    #ifdef EREMOTEIO
        case EREMOTEIO: return DRMP3_IO_ERROR;
    #endif
    #ifdef EDQUOT
        case EDQUOT: return DRMP3_NO_SPACE;
    #endif
    #ifdef ENOMEDIUM
        case ENOMEDIUM: return DRMP3_DOES_NOT_EXIST;
    #endif
    #ifdef EMEDIUMTYPE
        case EMEDIUMTYPE: return DRMP3_ERROR;
    #endif
    #ifdef ECANCELED
        case ECANCELED: return DRMP3_CANCELLED;
    #endif
    #ifdef ENOKEY
        case ENOKEY: return DRMP3_ERROR;
    #endif
    #ifdef EKEYEXPIRED
        case EKEYEXPIRED: return DRMP3_ERROR;
    #endif
    #ifdef EKEYREVOKED
        case EKEYREVOKED: return DRMP3_ERROR;
    #endif
    #ifdef EKEYREJECTED
        case EKEYREJECTED: return DRMP3_ERROR;
    #endif
    #ifdef EOWNERDEAD
        case EOWNERDEAD: return DRMP3_ERROR;
    #endif
    #ifdef ENOTRECOVERABLE
        case ENOTRECOVERABLE: return DRMP3_ERROR;
    #endif
    #ifdef ERFKILL
        case ERFKILL: return DRMP3_ERROR;
    #endif
    #ifdef EHWPOISON
        case EHWPOISON: return DRMP3_ERROR;
    #endif
        default: return DRMP3_ERROR;
    }
}
/* End Errno */

/* fopen */
static drmp3_result drmp3_fopen(FILE** ppFile, const char* pFilePath, const char* pOpenMode)
{
#if defined(_MSC_VER) && _MSC_VER >= 1400
    errno_t err;
#endif

    if (ppFile != NULL) {
        *ppFile = NULL;  /* Safety. */
    }

    if (pFilePath == NULL || pOpenMode == NULL || ppFile == NULL) {
        return DRMP3_INVALID_ARGS;
    }

#if defined(_MSC_VER) && _MSC_VER >= 1400
    err = fopen_s(ppFile, pFilePath, pOpenMode);
    if (err != 0) {
        return drmp3_result_from_errno(err);
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
        drmp3_result result = drmp3_result_from_errno(errno);
        if (result == DRMP3_SUCCESS) {
            result = DRMP3_ERROR;   /* Just a safety check to make sure we never ever return success when pFile == NULL. */
        }

        return result;
    }
#endif

    return DRMP3_SUCCESS;
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
        #define DRMP3_HAS_WFOPEN
    #endif
#endif

static drmp3_result drmp3_wfopen(FILE** ppFile, const wchar_t* pFilePath, const wchar_t* pOpenMode, const drmp3_allocation_callbacks* pAllocationCallbacks)
{
    if (ppFile != NULL) {
        *ppFile = NULL;  /* Safety. */
    }

    if (pFilePath == NULL || pOpenMode == NULL || ppFile == NULL) {
        return DRMP3_INVALID_ARGS;
    }

#if defined(DRMP3_HAS_WFOPEN)
    {
        /* Use _wfopen() on Windows. */
    #if defined(_MSC_VER) && _MSC_VER >= 1400
        errno_t err = _wfopen_s(ppFile, pFilePath, pOpenMode);
        if (err != 0) {
            return drmp3_result_from_errno(err);
        }
    #else
        *ppFile = _wfopen(pFilePath, pOpenMode);
        if (*ppFile == NULL) {
            return drmp3_result_from_errno(errno);
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
        DRMP3_ZERO_OBJECT(&mbs);
        lenMB = wcsrtombs(NULL, &pFilePathTemp, 0, &mbs);
        if (lenMB == (size_t)-1) {
            return drmp3_result_from_errno(errno);
        }

        pFilePathMB = (char*)drmp3__malloc_from_callbacks(lenMB + 1, pAllocationCallbacks);
        if (pFilePathMB == NULL) {
            return DRMP3_OUT_OF_MEMORY;
        }

        pFilePathTemp = pFilePath;
        DRMP3_ZERO_OBJECT(&mbs);
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

        drmp3__free_from_callbacks(pFilePathMB, pAllocationCallbacks);
    }
	#endif

    if (*ppFile == NULL) {
        return DRMP3_ERROR;
    }
#endif

    return DRMP3_SUCCESS;
}
/* End fopen */


static size_t drmp3__on_read_stdio(void* pUserData, void* pBufferOut, size_t bytesToRead)
{
    return fread(pBufferOut, 1, bytesToRead, (FILE*)pUserData);
}

static drmp3_bool32 drmp3__on_seek_stdio(void* pUserData, int offset, drmp3_seek_origin origin)
{
    int whence = SEEK_SET;
    if (origin == DRMP3_SEEK_CUR) {
        whence = SEEK_CUR;
    } else if (origin == DRMP3_SEEK_END) {
        whence = SEEK_END;
    }

    return fseek((FILE*)pUserData, offset, whence) == 0;
}

static drmp3_bool32 drmp3__on_tell_stdio(void* pUserData, drmp3_int64* pCursor)
{
    FILE* pFileStdio = (FILE*)pUserData;
    drmp3_int64 result;

    /* These were all validated at a higher level. */
    DRMP3_ASSERT(pFileStdio != NULL);
    DRMP3_ASSERT(pCursor    != NULL);

#if defined(_WIN32) && !defined(NXDK)
    #if defined(_MSC_VER) && _MSC_VER > 1200
        result = _ftelli64(pFileStdio);
    #else
        result = ftell(pFileStdio);
    #endif
#else
    result = ftell(pFileStdio);
#endif

    *pCursor = result;

    return DRMP3_TRUE;
}

DRMP3_API drmp3_bool32 drmp3_init_file_with_metadata(drmp3* pMP3, const char* pFilePath, drmp3_meta_proc onMeta, void* pUserDataMeta, const drmp3_allocation_callbacks* pAllocationCallbacks)
{
    drmp3_bool32 result;
    FILE* pFile;

    if (pMP3 == NULL) {
        return DRMP3_FALSE;
    }

    DRMP3_ZERO_OBJECT(pMP3);

    if (drmp3_fopen(&pFile, pFilePath, "rb") != DRMP3_SUCCESS) {
        return DRMP3_FALSE;
    }

    result = drmp3_init_internal(pMP3, drmp3__on_read_stdio, drmp3__on_seek_stdio, drmp3__on_tell_stdio, onMeta, (void*)pFile, pUserDataMeta, pAllocationCallbacks);
    if (result != DRMP3_TRUE) {
        fclose(pFile);
        return result;
    }

    return DRMP3_TRUE;
}

DRMP3_API drmp3_bool32 drmp3_init_file_with_metadata_w(drmp3* pMP3, const wchar_t* pFilePath, drmp3_meta_proc onMeta, void* pUserDataMeta, const drmp3_allocation_callbacks* pAllocationCallbacks)
{
    drmp3_bool32 result;
    FILE* pFile;

    if (pMP3 == NULL) {
        return DRMP3_FALSE;
    }

    DRMP3_ZERO_OBJECT(pMP3);

    if (drmp3_wfopen(&pFile, pFilePath, L"rb", pAllocationCallbacks) != DRMP3_SUCCESS) {
        return DRMP3_FALSE;
    }

    result = drmp3_init_internal(pMP3, drmp3__on_read_stdio, drmp3__on_seek_stdio, drmp3__on_tell_stdio, onMeta, (void*)pFile, pUserDataMeta, pAllocationCallbacks);
    if (result != DRMP3_TRUE) {
        fclose(pFile);
        return result;
    }

    return DRMP3_TRUE;
}

DRMP3_API drmp3_bool32 drmp3_init_file(drmp3* pMP3, const char* pFilePath, const drmp3_allocation_callbacks* pAllocationCallbacks)
{
    return drmp3_init_file_with_metadata(pMP3, pFilePath, NULL, NULL, pAllocationCallbacks);
}

DRMP3_API drmp3_bool32 drmp3_init_file_w(drmp3* pMP3, const wchar_t* pFilePath, const drmp3_allocation_callbacks* pAllocationCallbacks)
{
    return drmp3_init_file_with_metadata_w(pMP3, pFilePath, NULL, NULL, pAllocationCallbacks);
}
#endif

DRMP3_API void drmp3_uninit(drmp3* pMP3)
{
    if (pMP3 == NULL) {
        return;
    }

#ifndef DR_MP3_NO_STDIO
    if (pMP3->onRead == drmp3__on_read_stdio) {
        FILE* pFile = (FILE*)pMP3->pUserData;
        if (pFile != NULL) {
            fclose(pFile);
            pMP3->pUserData = NULL; /* Make sure the file handle is cleared to NULL to we don't attempt to close it a second time. */
        }
    }
#endif

    drmp3__free_from_callbacks(pMP3->pData, &pMP3->allocationCallbacks);
}

#if defined(DR_MP3_FLOAT_OUTPUT)
static void drmp3_f32_to_s16(drmp3_int16* dst, const float* src, drmp3_uint64 sampleCount)
{
    drmp3_uint64 i;
    drmp3_uint64 i4;
    drmp3_uint64 sampleCount4;

    /* Unrolled. */
    i = 0;
    sampleCount4 = sampleCount >> 2;
    for (i4 = 0; i4 < sampleCount4; i4 += 1) {
        float x0 = src[i+0];
        float x1 = src[i+1];
        float x2 = src[i+2];
        float x3 = src[i+3];

        x0 = ((x0 < -1) ? -1 : ((x0 > 1) ? 1 : x0));
        x1 = ((x1 < -1) ? -1 : ((x1 > 1) ? 1 : x1));
        x2 = ((x2 < -1) ? -1 : ((x2 > 1) ? 1 : x2));
        x3 = ((x3 < -1) ? -1 : ((x3 > 1) ? 1 : x3));

        x0 = x0 * 32767.0f;
        x1 = x1 * 32767.0f;
        x2 = x2 * 32767.0f;
        x3 = x3 * 32767.0f;

        dst[i+0] = (drmp3_int16)x0;
        dst[i+1] = (drmp3_int16)x1;
        dst[i+2] = (drmp3_int16)x2;
        dst[i+3] = (drmp3_int16)x3;

        i += 4;
    }

    /* Leftover. */
    for (; i < sampleCount; i += 1) {
        float x = src[i];
        x = ((x < -1) ? -1 : ((x > 1) ? 1 : x));    /* clip */
        x = x * 32767.0f;                           /* -1..1 to -32767..32767 */

        dst[i] = (drmp3_int16)x;
    }
}
#endif

#if !defined(DR_MP3_FLOAT_OUTPUT)
static void drmp3_s16_to_f32(float* dst, const drmp3_int16* src, drmp3_uint64 sampleCount)
{
    drmp3_uint64 i;
    for (i = 0; i < sampleCount; i += 1) {
        float x = (float)src[i];
        x = x * 0.000030517578125f;         /* -32768..32767 to -1..0.999969482421875 */
        dst[i] = x;
    }
}
#endif


static drmp3_uint64 drmp3_read_pcm_frames_raw(drmp3* pMP3, drmp3_uint64 framesToRead, void* pBufferOut)
{
    drmp3_uint64 totalFramesRead = 0;

    DRMP3_ASSERT(pMP3 != NULL);
    DRMP3_ASSERT(pMP3->onRead != NULL);

    while (framesToRead > 0) {
        drmp3_uint32 framesToConsume;

        /* Skip frames if necessary. */
        if (pMP3->currentPCMFrame < pMP3->delayInPCMFrames) {
            drmp3_uint32 framesToSkip = (drmp3_uint32)DRMP3_MIN(pMP3->pcmFramesRemainingInMP3Frame, pMP3->delayInPCMFrames - pMP3->currentPCMFrame);

            pMP3->currentPCMFrame              += framesToSkip;
            pMP3->pcmFramesConsumedInMP3Frame  += framesToSkip;
            pMP3->pcmFramesRemainingInMP3Frame -= framesToSkip;
        }

        framesToConsume = (drmp3_uint32)DRMP3_MIN(pMP3->pcmFramesRemainingInMP3Frame, framesToRead);

        /* Clamp the number of frames to read to the padding. */
        if (pMP3->totalPCMFrameCount != DRMP3_UINT64_MAX && pMP3->totalPCMFrameCount > pMP3->paddingInPCMFrames) {
            if (pMP3->currentPCMFrame < (pMP3->totalPCMFrameCount - pMP3->paddingInPCMFrames)) {
                drmp3_uint64 framesRemainigToPadding = (pMP3->totalPCMFrameCount - pMP3->paddingInPCMFrames) - pMP3->currentPCMFrame;
                if (framesToConsume >               framesRemainigToPadding) {
                    framesToConsume = (drmp3_uint32)framesRemainigToPadding;
                }
            } else {
                /* We're into the padding. Abort. */
                break;
            }
        }

        if (pBufferOut != NULL) {
            #if defined(DR_MP3_FLOAT_OUTPUT)
            {
                /* f32 */
                float* pFramesOutF32 = (float*)DRMP3_OFFSET_PTR(pBufferOut,          sizeof(float) * totalFramesRead                   * pMP3->channels);
                float* pFramesInF32  = (float*)DRMP3_OFFSET_PTR(&pMP3->pcmFrames[0], sizeof(float) * pMP3->pcmFramesConsumedInMP3Frame * pMP3->mp3FrameChannels);
                DRMP3_COPY_MEMORY(pFramesOutF32, pFramesInF32, sizeof(float) * framesToConsume * pMP3->channels);
            }
            #else
            {
                /* s16 */
                drmp3_int16* pFramesOutS16 = (drmp3_int16*)DRMP3_OFFSET_PTR(pBufferOut,          sizeof(drmp3_int16) * totalFramesRead                   * pMP3->channels);
                drmp3_int16* pFramesInS16  = (drmp3_int16*)DRMP3_OFFSET_PTR(&pMP3->pcmFrames[0], sizeof(drmp3_int16) * pMP3->pcmFramesConsumedInMP3Frame * pMP3->mp3FrameChannels);
                DRMP3_COPY_MEMORY(pFramesOutS16, pFramesInS16, sizeof(drmp3_int16) * framesToConsume * pMP3->channels);
            }
            #endif
        }

        pMP3->currentPCMFrame              += framesToConsume;
        pMP3->pcmFramesConsumedInMP3Frame  += framesToConsume;
        pMP3->pcmFramesRemainingInMP3Frame -= framesToConsume;
        totalFramesRead                    += framesToConsume;
        framesToRead                       -= framesToConsume;

        if (framesToRead == 0) {
            break;
        }

        /* If the cursor is already at the padding we need to abort. */
        if (pMP3->totalPCMFrameCount != DRMP3_UINT64_MAX && pMP3->totalPCMFrameCount > pMP3->paddingInPCMFrames && pMP3->currentPCMFrame >= (pMP3->totalPCMFrameCount - pMP3->paddingInPCMFrames)) {
            break;
        }

        DRMP3_ASSERT(pMP3->pcmFramesRemainingInMP3Frame == 0);

        /* At this point we have exhausted our in-memory buffer so we need to re-fill. */
        if (drmp3_decode_next_frame(pMP3) == 0) {
            break;
        }
    }

    return totalFramesRead;
}


DRMP3_API drmp3_uint64 drmp3_read_pcm_frames_f32(drmp3* pMP3, drmp3_uint64 framesToRead, float* pBufferOut)
{
    if (pMP3 == NULL || pMP3->onRead == NULL) {
        return 0;
    }

#if defined(DR_MP3_FLOAT_OUTPUT)
    /* Fast path. No conversion required. */
    return drmp3_read_pcm_frames_raw(pMP3, framesToRead, pBufferOut);
#else
    /* Slow path. Convert from s16 to f32. */
    {
        drmp3_int16 pTempS16[8192];
        drmp3_uint64 totalPCMFramesRead = 0;

        while (totalPCMFramesRead < framesToRead) {
            drmp3_uint64 framesJustRead;
            drmp3_uint64 framesRemaining = framesToRead - totalPCMFramesRead;
            drmp3_uint64 framesToReadNow = DRMP3_COUNTOF(pTempS16) / pMP3->channels;
            if (framesToReadNow > framesRemaining) {
                framesToReadNow = framesRemaining;
            }

            framesJustRead = drmp3_read_pcm_frames_raw(pMP3, framesToReadNow, pTempS16);
            if (framesJustRead == 0) {
                break;
            }

            drmp3_s16_to_f32((float*)DRMP3_OFFSET_PTR(pBufferOut, sizeof(float) * totalPCMFramesRead * pMP3->channels), pTempS16, framesJustRead * pMP3->channels);
            totalPCMFramesRead += framesJustRead;
        }

        return totalPCMFramesRead;
    }
#endif
}

DRMP3_API drmp3_uint64 drmp3_read_pcm_frames_s16(drmp3* pMP3, drmp3_uint64 framesToRead, drmp3_int16* pBufferOut)
{
    if (pMP3 == NULL || pMP3->onRead == NULL) {
        return 0;
    }

#if !defined(DR_MP3_FLOAT_OUTPUT)
    /* Fast path. No conversion required. */
    return drmp3_read_pcm_frames_raw(pMP3, framesToRead, pBufferOut);
#else
    /* Slow path. Convert from f32 to s16. */
    {
        float pTempF32[4096];
        drmp3_uint64 totalPCMFramesRead = 0;

        while (totalPCMFramesRead < framesToRead) {
            drmp3_uint64 framesJustRead;
            drmp3_uint64 framesRemaining = framesToRead - totalPCMFramesRead;
            drmp3_uint64 framesToReadNow = DRMP3_COUNTOF(pTempF32) / pMP3->channels;
            if (framesToReadNow > framesRemaining) {
                framesToReadNow = framesRemaining;
            }

            framesJustRead = drmp3_read_pcm_frames_raw(pMP3, framesToReadNow, pTempF32);
            if (framesJustRead == 0) {
                break;
            }

            drmp3_f32_to_s16((drmp3_int16*)DRMP3_OFFSET_PTR(pBufferOut, sizeof(drmp3_int16) * totalPCMFramesRead * pMP3->channels), pTempF32, framesJustRead * pMP3->channels);
            totalPCMFramesRead += framesJustRead;
        }

        return totalPCMFramesRead;
    }
#endif
}

static void drmp3_reset(drmp3* pMP3)
{
    DRMP3_ASSERT(pMP3 != NULL);

    pMP3->pcmFramesConsumedInMP3Frame = 0;
    pMP3->pcmFramesRemainingInMP3Frame = 0;
    pMP3->currentPCMFrame = 0;
    pMP3->dataSize = 0;
    pMP3->atEnd = DRMP3_FALSE;
    drmp3dec_init(&pMP3->decoder);
}

static drmp3_bool32 drmp3_seek_to_start_of_stream(drmp3* pMP3)
{
    DRMP3_ASSERT(pMP3 != NULL);
    DRMP3_ASSERT(pMP3->onSeek != NULL);

    /* Seek to the start of the stream to begin with. */
    if (!drmp3__on_seek_64(pMP3, pMP3->streamStartOffset, DRMP3_SEEK_SET)) {
        return DRMP3_FALSE;
    }

    /* Clear any cached data. */
    drmp3_reset(pMP3);
    return DRMP3_TRUE;
}


static drmp3_bool32 drmp3_seek_forward_by_pcm_frames__brute_force(drmp3* pMP3, drmp3_uint64 frameOffset)
{
    drmp3_uint64 framesRead;

    /*
    Just using a dumb read-and-discard for now. What would be nice is to parse only the header of the MP3 frame, and then skip over leading
    frames without spending the time doing a full decode. I cannot see an easy way to do this in minimp3, however, so it may involve some
    kind of manual processing.
    */
#if defined(DR_MP3_FLOAT_OUTPUT)
    framesRead = drmp3_read_pcm_frames_f32(pMP3, frameOffset, NULL);
#else
    framesRead = drmp3_read_pcm_frames_s16(pMP3, frameOffset, NULL);
#endif
    if (framesRead != frameOffset) {
        return DRMP3_FALSE;
    }

    return DRMP3_TRUE;
}

static drmp3_bool32 drmp3_seek_to_pcm_frame__brute_force(drmp3* pMP3, drmp3_uint64 frameIndex)
{
    DRMP3_ASSERT(pMP3 != NULL);

    if (frameIndex == pMP3->currentPCMFrame) {
        return DRMP3_TRUE;
    }

    /*
    If we're moving foward we just read from where we're at. Otherwise we need to move back to the start of
    the stream and read from the beginning.
    */
    if (frameIndex < pMP3->currentPCMFrame) {
        /* Moving backward. Move to the start of the stream and then move forward. */
        if (!drmp3_seek_to_start_of_stream(pMP3)) {
            return DRMP3_FALSE;
        }
    }

    DRMP3_ASSERT(frameIndex >= pMP3->currentPCMFrame);
    return drmp3_seek_forward_by_pcm_frames__brute_force(pMP3, (frameIndex - pMP3->currentPCMFrame));
}

static drmp3_bool32 drmp3_find_closest_seek_point(drmp3* pMP3, drmp3_uint64 frameIndex, drmp3_uint32* pSeekPointIndex)
{
    drmp3_uint32 iSeekPoint;

    DRMP3_ASSERT(pSeekPointIndex != NULL);

    *pSeekPointIndex = 0;

    if (frameIndex < pMP3->pSeekPoints[0].pcmFrameIndex) {
        return DRMP3_FALSE;
    }

    /* Linear search for simplicity to begin with while I'm getting this thing working. Once it's all working change this to a binary search. */
    for (iSeekPoint = 0; iSeekPoint < pMP3->seekPointCount; ++iSeekPoint) {
        if (pMP3->pSeekPoints[iSeekPoint].pcmFrameIndex > frameIndex) {
            break;  /* Found it. */
        }

        *pSeekPointIndex = iSeekPoint;
    }

    return DRMP3_TRUE;
}

static drmp3_bool32 drmp3_seek_to_pcm_frame__seek_table(drmp3* pMP3, drmp3_uint64 frameIndex)
{
    drmp3_seek_point seekPoint;
    drmp3_uint32 priorSeekPointIndex;
    drmp3_uint16 iMP3Frame;
    drmp3_uint64 leftoverFrames;

    DRMP3_ASSERT(pMP3 != NULL);
    DRMP3_ASSERT(pMP3->pSeekPoints != NULL);
    DRMP3_ASSERT(pMP3->seekPointCount > 0);

    /* If there is no prior seekpoint it means the target PCM frame comes before the first seek point. Just assume a seekpoint at the start of the file in this case. */
    if (drmp3_find_closest_seek_point(pMP3, frameIndex, &priorSeekPointIndex)) {
        seekPoint = pMP3->pSeekPoints[priorSeekPointIndex];
    } else {
        seekPoint.seekPosInBytes     = 0;
        seekPoint.pcmFrameIndex      = 0;
        seekPoint.mp3FramesToDiscard = 0;
        seekPoint.pcmFramesToDiscard = 0;
    }

    /* First thing to do is seek to the first byte of the relevant MP3 frame. */
    if (!drmp3__on_seek_64(pMP3, seekPoint.seekPosInBytes, DRMP3_SEEK_SET)) {
        return DRMP3_FALSE; /* Failed to seek. */
    }

    /* Clear any cached data. */
    drmp3_reset(pMP3);

    /* Whole MP3 frames need to be discarded first. */
    for (iMP3Frame = 0; iMP3Frame < seekPoint.mp3FramesToDiscard; ++iMP3Frame) {
        drmp3_uint32 pcmFramesRead;
        drmp3d_sample_t* pPCMFrames;

        /* Pass in non-null for the last frame because we want to ensure the sample rate converter is preloaded correctly. */
        pPCMFrames = NULL;
        if (iMP3Frame == seekPoint.mp3FramesToDiscard-1) {
            pPCMFrames = (drmp3d_sample_t*)pMP3->pcmFrames;
        }

        /* We first need to decode the next frame. */
        pcmFramesRead = drmp3_decode_next_frame_ex(pMP3, pPCMFrames, NULL, NULL);
        if (pcmFramesRead == 0) {
            return DRMP3_FALSE;
        }
    }

    /* We seeked to an MP3 frame in the raw stream so we need to make sure the current PCM frame is set correctly. */
    pMP3->currentPCMFrame = seekPoint.pcmFrameIndex - seekPoint.pcmFramesToDiscard;

    /*
    Now at this point we can follow the same process as the brute force technique where we just skip over unnecessary MP3 frames and then
    read-and-discard at least 2 whole MP3 frames.
    */
    leftoverFrames = frameIndex - pMP3->currentPCMFrame;
    return drmp3_seek_forward_by_pcm_frames__brute_force(pMP3, leftoverFrames);
}

DRMP3_API drmp3_bool32 drmp3_seek_to_pcm_frame(drmp3* pMP3, drmp3_uint64 frameIndex)
{
    if (pMP3 == NULL || pMP3->onSeek == NULL) {
        return DRMP3_FALSE;
    }

    if (frameIndex == 0) {
        return drmp3_seek_to_start_of_stream(pMP3);
    }

    /* Use the seek table if we have one. */
    if (pMP3->pSeekPoints != NULL && pMP3->seekPointCount > 0) {
        return drmp3_seek_to_pcm_frame__seek_table(pMP3, frameIndex);
    } else {
        return drmp3_seek_to_pcm_frame__brute_force(pMP3, frameIndex);
    }
}

DRMP3_API drmp3_bool32 drmp3_get_mp3_and_pcm_frame_count(drmp3* pMP3, drmp3_uint64* pMP3FrameCount, drmp3_uint64* pPCMFrameCount)
{
    drmp3_uint64 currentPCMFrame;
    drmp3_uint64 totalPCMFrameCount;
    drmp3_uint64 totalMP3FrameCount;

    if (pMP3 == NULL) {
        return DRMP3_FALSE;
    }

    /*
    The way this works is we move back to the start of the stream, iterate over each MP3 frame and calculate the frame count based
    on our output sample rate, the seek back to the PCM frame we were sitting on before calling this function.
    */

    /* The stream must support seeking for this to work. */
    if (pMP3->onSeek == NULL) {
        return DRMP3_FALSE;
    }

    /* We'll need to seek back to where we were, so grab the PCM frame we're currently sitting on so we can restore later. */
    currentPCMFrame = pMP3->currentPCMFrame;

    if (!drmp3_seek_to_start_of_stream(pMP3)) {
        return DRMP3_FALSE;
    }

    totalPCMFrameCount = 0;
    totalMP3FrameCount = 0;

    for (;;) {
        drmp3_uint32 pcmFramesInCurrentMP3Frame;

        pcmFramesInCurrentMP3Frame = drmp3_decode_next_frame_ex(pMP3, NULL, NULL, NULL);
        if (pcmFramesInCurrentMP3Frame == 0) {
            break;
        }

        totalPCMFrameCount += pcmFramesInCurrentMP3Frame;
        totalMP3FrameCount += 1;
    }

    /* Finally, we need to seek back to where we were. */
    if (!drmp3_seek_to_start_of_stream(pMP3)) {
        return DRMP3_FALSE;
    }

    if (!drmp3_seek_to_pcm_frame(pMP3, currentPCMFrame)) {
        return DRMP3_FALSE;
    }

    if (pMP3FrameCount != NULL) {
        *pMP3FrameCount = totalMP3FrameCount;
    }
    if (pPCMFrameCount != NULL) {
        *pPCMFrameCount = totalPCMFrameCount;
    }

    return DRMP3_TRUE;
}

DRMP3_API drmp3_uint64 drmp3_get_pcm_frame_count(drmp3* pMP3)
{
    drmp3_uint64 totalPCMFrameCount;

    if (pMP3 == NULL) {
        return 0;
    }

    if (pMP3->totalPCMFrameCount != DRMP3_UINT64_MAX) {
        totalPCMFrameCount = pMP3->totalPCMFrameCount;

        if (totalPCMFrameCount >= pMP3->delayInPCMFrames) {
            totalPCMFrameCount -= pMP3->delayInPCMFrames;
        } else {
            /* The delay is greater than the frame count reported by the Xing/Info tag. Assume it's invalid and ignore. */
        }

        if (totalPCMFrameCount >= pMP3->paddingInPCMFrames) {
            totalPCMFrameCount -= pMP3->paddingInPCMFrames;
        } else {
            /* The padding is greater than the frame count reported by the Xing/Info tag. Assume it's invalid and ignore. */
        }

        return totalPCMFrameCount;
    } else {
        /* Unknown frame count. Need to calculate it. */
        if (!drmp3_get_mp3_and_pcm_frame_count(pMP3, NULL, &totalPCMFrameCount)) {
            return 0;
        }

        return totalPCMFrameCount;
    }
}

DRMP3_API drmp3_uint64 drmp3_get_mp3_frame_count(drmp3* pMP3)
{
    drmp3_uint64 totalMP3FrameCount;
    if (!drmp3_get_mp3_and_pcm_frame_count(pMP3, &totalMP3FrameCount, NULL)) {
        return 0;
    }

    return totalMP3FrameCount;
}

static void drmp3__accumulate_running_pcm_frame_count(drmp3* pMP3, drmp3_uint32 pcmFrameCountIn, drmp3_uint64* pRunningPCMFrameCount, float* pRunningPCMFrameCountFractionalPart)
{
    float srcRatio;
    float pcmFrameCountOutF;
    drmp3_uint32 pcmFrameCountOut;

    srcRatio = (float)pMP3->mp3FrameSampleRate / (float)pMP3->sampleRate;
    DRMP3_ASSERT(srcRatio > 0);

    pcmFrameCountOutF = *pRunningPCMFrameCountFractionalPart + (pcmFrameCountIn / srcRatio);
    pcmFrameCountOut  = (drmp3_uint32)pcmFrameCountOutF;
    *pRunningPCMFrameCountFractionalPart = pcmFrameCountOutF - pcmFrameCountOut;
    *pRunningPCMFrameCount += pcmFrameCountOut;
}

typedef struct
{
    drmp3_uint64 bytePos;
    drmp3_uint64 pcmFrameIndex; /* <-- After sample rate conversion. */
} drmp3__seeking_mp3_frame_info;

DRMP3_API drmp3_bool32 drmp3_calculate_seek_points(drmp3* pMP3, drmp3_uint32* pSeekPointCount, drmp3_seek_point* pSeekPoints)
{
    drmp3_uint32 seekPointCount;
    drmp3_uint64 currentPCMFrame;
    drmp3_uint64 totalMP3FrameCount;
    drmp3_uint64 totalPCMFrameCount;

    if (pMP3 == NULL || pSeekPointCount == NULL || pSeekPoints == NULL) {
        return DRMP3_FALSE; /* Invalid args. */
    }

    seekPointCount = *pSeekPointCount;
    if (seekPointCount == 0) {
        return DRMP3_FALSE;  /* The client has requested no seek points. Consider this to be invalid arguments since the client has probably not intended this. */
    }

    /* We'll need to seek back to the current sample after calculating the seekpoints so we need to go ahead and grab the current location at the top. */
    currentPCMFrame = pMP3->currentPCMFrame;

    /* We never do more than the total number of MP3 frames and we limit it to 32-bits. */
    if (!drmp3_get_mp3_and_pcm_frame_count(pMP3, &totalMP3FrameCount, &totalPCMFrameCount)) {
        return DRMP3_FALSE;
    }

    /* If there's less than DRMP3_SEEK_LEADING_MP3_FRAMES+1 frames we just report 1 seek point which will be the very start of the stream. */
    if (totalMP3FrameCount < DRMP3_SEEK_LEADING_MP3_FRAMES+1) {
        seekPointCount = 1;
        pSeekPoints[0].seekPosInBytes     = 0;
        pSeekPoints[0].pcmFrameIndex      = 0;
        pSeekPoints[0].mp3FramesToDiscard = 0;
        pSeekPoints[0].pcmFramesToDiscard = 0;
    } else {
        drmp3_uint64 pcmFramesBetweenSeekPoints;
        drmp3__seeking_mp3_frame_info mp3FrameInfo[DRMP3_SEEK_LEADING_MP3_FRAMES+1];
        drmp3_uint64 runningPCMFrameCount = 0;
        float runningPCMFrameCountFractionalPart = 0;
        drmp3_uint64 nextTargetPCMFrame;
        drmp3_uint32 iMP3Frame;
        drmp3_uint32 iSeekPoint;

        if (seekPointCount > totalMP3FrameCount-1) {
            seekPointCount = (drmp3_uint32)totalMP3FrameCount-1;
        }

        pcmFramesBetweenSeekPoints = totalPCMFrameCount / (seekPointCount+1);

        /*
        Here is where we actually calculate the seek points. We need to start by moving the start of the stream. We then enumerate over each
        MP3 frame.
        */
        if (!drmp3_seek_to_start_of_stream(pMP3)) {
            return DRMP3_FALSE;
        }

        /*
        We need to cache the byte positions of the previous MP3 frames. As a new MP3 frame is iterated, we cycle the byte positions in this
        array. The value in the first item in this array is the byte position that will be reported in the next seek point.
        */

        /* We need to initialize the array of MP3 byte positions for the leading MP3 frames. */
        for (iMP3Frame = 0; iMP3Frame < DRMP3_SEEK_LEADING_MP3_FRAMES+1; ++iMP3Frame) {
            drmp3_uint32 pcmFramesInCurrentMP3FrameIn;

            /* The byte position of the next frame will be the stream's cursor position, minus whatever is sitting in the buffer. */
            DRMP3_ASSERT(pMP3->streamCursor >= pMP3->dataSize);
            mp3FrameInfo[iMP3Frame].bytePos       = pMP3->streamCursor - pMP3->dataSize;
            mp3FrameInfo[iMP3Frame].pcmFrameIndex = runningPCMFrameCount;

            /* We need to get information about this frame so we can know how many samples it contained. */
            pcmFramesInCurrentMP3FrameIn = drmp3_decode_next_frame_ex(pMP3, NULL, NULL, NULL);
            if (pcmFramesInCurrentMP3FrameIn == 0) {
                return DRMP3_FALSE; /* This should never happen. */
            }

            drmp3__accumulate_running_pcm_frame_count(pMP3, pcmFramesInCurrentMP3FrameIn, &runningPCMFrameCount, &runningPCMFrameCountFractionalPart);
        }

        /*
        At this point we will have extracted the byte positions of the leading MP3 frames. We can now start iterating over each seek point and
        calculate them.
        */
        nextTargetPCMFrame = 0;
        for (iSeekPoint = 0; iSeekPoint < seekPointCount; ++iSeekPoint) {
            nextTargetPCMFrame += pcmFramesBetweenSeekPoints;

            for (;;) {
                if (nextTargetPCMFrame < runningPCMFrameCount) {
                    /* The next seek point is in the current MP3 frame. */
                    pSeekPoints[iSeekPoint].seekPosInBytes     = mp3FrameInfo[0].bytePos;
                    pSeekPoints[iSeekPoint].pcmFrameIndex      = nextTargetPCMFrame;
                    pSeekPoints[iSeekPoint].mp3FramesToDiscard = DRMP3_SEEK_LEADING_MP3_FRAMES;
                    pSeekPoints[iSeekPoint].pcmFramesToDiscard = (drmp3_uint16)(nextTargetPCMFrame - mp3FrameInfo[DRMP3_SEEK_LEADING_MP3_FRAMES-1].pcmFrameIndex);
                    break;
                } else {
                    size_t i;
                    drmp3_uint32 pcmFramesInCurrentMP3FrameIn;

                    /*
                    The next seek point is not in the current MP3 frame, so continue on to the next one. The first thing to do is cycle the cached
                    MP3 frame info.
                    */
                    for (i = 0; i < DRMP3_COUNTOF(mp3FrameInfo)-1; ++i) {
                        mp3FrameInfo[i] = mp3FrameInfo[i+1];
                    }

                    /* Cache previous MP3 frame info. */
                    mp3FrameInfo[DRMP3_COUNTOF(mp3FrameInfo)-1].bytePos       = pMP3->streamCursor - pMP3->dataSize;
                    mp3FrameInfo[DRMP3_COUNTOF(mp3FrameInfo)-1].pcmFrameIndex = runningPCMFrameCount;

                    /*
                    Go to the next MP3 frame. This shouldn't ever fail, but just in case it does we just set the seek point and break. If it happens, it
                    should only ever do it for the last seek point.
                    */
                    pcmFramesInCurrentMP3FrameIn = drmp3_decode_next_frame_ex(pMP3, NULL, NULL, NULL);
                    if (pcmFramesInCurrentMP3FrameIn == 0) {
                        pSeekPoints[iSeekPoint].seekPosInBytes     = mp3FrameInfo[0].bytePos;
                        pSeekPoints[iSeekPoint].pcmFrameIndex      = nextTargetPCMFrame;
                        pSeekPoints[iSeekPoint].mp3FramesToDiscard = DRMP3_SEEK_LEADING_MP3_FRAMES;
                        pSeekPoints[iSeekPoint].pcmFramesToDiscard = (drmp3_uint16)(nextTargetPCMFrame - mp3FrameInfo[DRMP3_SEEK_LEADING_MP3_FRAMES-1].pcmFrameIndex);
                        break;
                    }

                    drmp3__accumulate_running_pcm_frame_count(pMP3, pcmFramesInCurrentMP3FrameIn, &runningPCMFrameCount, &runningPCMFrameCountFractionalPart);
                }
            }
        }

        /* Finally, we need to seek back to where we were. */
        if (!drmp3_seek_to_start_of_stream(pMP3)) {
            return DRMP3_FALSE;
        }
        if (!drmp3_seek_to_pcm_frame(pMP3, currentPCMFrame)) {
            return DRMP3_FALSE;
        }
    }

    *pSeekPointCount = seekPointCount;
    return DRMP3_TRUE;
}

DRMP3_API drmp3_bool32 drmp3_bind_seek_table(drmp3* pMP3, drmp3_uint32 seekPointCount, drmp3_seek_point* pSeekPoints)
{
    if (pMP3 == NULL) {
        return DRMP3_FALSE;
    }

    if (seekPointCount == 0 || pSeekPoints == NULL) {
        /* Unbinding. */
        pMP3->seekPointCount = 0;
        pMP3->pSeekPoints = NULL;
    } else {
        /* Binding. */
        pMP3->seekPointCount = seekPointCount;
        pMP3->pSeekPoints = pSeekPoints;
    }

    return DRMP3_TRUE;
}


static float* drmp3__full_read_and_close_f32(drmp3* pMP3, drmp3_config* pConfig, drmp3_uint64* pTotalFrameCount)
{
    drmp3_uint64 totalFramesRead = 0;
    drmp3_uint64 framesCapacity = 0;
    float* pFrames = NULL;
    float temp[4096];

    DRMP3_ASSERT(pMP3 != NULL);

    for (;;) {
        drmp3_uint64 framesToReadRightNow = DRMP3_COUNTOF(temp) / pMP3->channels;
        drmp3_uint64 framesJustRead = drmp3_read_pcm_frames_f32(pMP3, framesToReadRightNow, temp);
        if (framesJustRead == 0) {
            break;
        }

        /* Reallocate the output buffer if there's not enough room. */
        if (framesCapacity < totalFramesRead + framesJustRead) {
            drmp3_uint64 oldFramesBufferSize;
            drmp3_uint64 newFramesBufferSize;
            drmp3_uint64 newFramesCap;
            float* pNewFrames;

            newFramesCap = framesCapacity * 2;
            if (newFramesCap < totalFramesRead + framesJustRead) {
                newFramesCap = totalFramesRead + framesJustRead;
            }

            oldFramesBufferSize = framesCapacity * pMP3->channels * sizeof(float);
            newFramesBufferSize = newFramesCap   * pMP3->channels * sizeof(float);
            if (newFramesBufferSize > (drmp3_uint64)DRMP3_SIZE_MAX) {
                break;
            }

            pNewFrames = (float*)drmp3__realloc_from_callbacks(pFrames, (size_t)newFramesBufferSize, (size_t)oldFramesBufferSize, &pMP3->allocationCallbacks);
            if (pNewFrames == NULL) {
                drmp3__free_from_callbacks(pFrames, &pMP3->allocationCallbacks);
                break;
            }

            pFrames = pNewFrames;
            framesCapacity = newFramesCap;
        }

        DRMP3_COPY_MEMORY(pFrames + totalFramesRead*pMP3->channels, temp, (size_t)(framesJustRead*pMP3->channels*sizeof(float)));
        totalFramesRead += framesJustRead;

        /* If the number of frames we asked for is less that what we actually read it means we've reached the end. */
        if (framesJustRead != framesToReadRightNow) {
            break;
        }
    }

    if (pConfig != NULL) {
        pConfig->channels   = pMP3->channels;
        pConfig->sampleRate = pMP3->sampleRate;
    }

    drmp3_uninit(pMP3);

    if (pTotalFrameCount) {
        *pTotalFrameCount = totalFramesRead;
    }

    return pFrames;
}

static drmp3_int16* drmp3__full_read_and_close_s16(drmp3* pMP3, drmp3_config* pConfig, drmp3_uint64* pTotalFrameCount)
{
    drmp3_uint64 totalFramesRead = 0;
    drmp3_uint64 framesCapacity = 0;
    drmp3_int16* pFrames = NULL;
    drmp3_int16 temp[4096];

    DRMP3_ASSERT(pMP3 != NULL);

    for (;;) {
        drmp3_uint64 framesToReadRightNow = DRMP3_COUNTOF(temp) / pMP3->channels;
        drmp3_uint64 framesJustRead = drmp3_read_pcm_frames_s16(pMP3, framesToReadRightNow, temp);
        if (framesJustRead == 0) {
            break;
        }

        /* Reallocate the output buffer if there's not enough room. */
        if (framesCapacity < totalFramesRead + framesJustRead) {
            drmp3_uint64 newFramesBufferSize;
            drmp3_uint64 oldFramesBufferSize;
            drmp3_uint64 newFramesCap;
            drmp3_int16* pNewFrames;

            newFramesCap = framesCapacity * 2;
            if (newFramesCap < totalFramesRead + framesJustRead) {
                newFramesCap = totalFramesRead + framesJustRead;
            }

            oldFramesBufferSize = framesCapacity * pMP3->channels * sizeof(drmp3_int16);
            newFramesBufferSize = newFramesCap   * pMP3->channels * sizeof(drmp3_int16);
            if (newFramesBufferSize > (drmp3_uint64)DRMP3_SIZE_MAX) {
                break;
            }

            pNewFrames = (drmp3_int16*)drmp3__realloc_from_callbacks(pFrames, (size_t)newFramesBufferSize, (size_t)oldFramesBufferSize, &pMP3->allocationCallbacks);
            if (pNewFrames == NULL) {
                drmp3__free_from_callbacks(pFrames, &pMP3->allocationCallbacks);
                break;
            }

            pFrames = pNewFrames;
            framesCapacity = newFramesCap;
        }

        DRMP3_COPY_MEMORY(pFrames + totalFramesRead*pMP3->channels, temp, (size_t)(framesJustRead*pMP3->channels*sizeof(drmp3_int16)));
        totalFramesRead += framesJustRead;

        /* If the number of frames we asked for is less that what we actually read it means we've reached the end. */
        if (framesJustRead != framesToReadRightNow) {
            break;
        }
    }

    if (pConfig != NULL) {
        pConfig->channels   = pMP3->channels;
        pConfig->sampleRate = pMP3->sampleRate;
    }

    drmp3_uninit(pMP3);

    if (pTotalFrameCount) {
        *pTotalFrameCount = totalFramesRead;
    }

    return pFrames;
}


DRMP3_API float* drmp3_open_and_read_pcm_frames_f32(drmp3_read_proc onRead, drmp3_seek_proc onSeek, drmp3_tell_proc onTell, void* pUserData, drmp3_config* pConfig, drmp3_uint64* pTotalFrameCount, const drmp3_allocation_callbacks* pAllocationCallbacks)
{
    drmp3 mp3;
    if (!drmp3_init(&mp3, onRead, onSeek, onTell, NULL, pUserData, pAllocationCallbacks)) {
        return NULL;
    }

    return drmp3__full_read_and_close_f32(&mp3, pConfig, pTotalFrameCount);
}

DRMP3_API drmp3_int16* drmp3_open_and_read_pcm_frames_s16(drmp3_read_proc onRead, drmp3_seek_proc onSeek, drmp3_tell_proc onTell, void* pUserData, drmp3_config* pConfig, drmp3_uint64* pTotalFrameCount, const drmp3_allocation_callbacks* pAllocationCallbacks)
{
    drmp3 mp3;
    if (!drmp3_init(&mp3, onRead, onSeek, onTell, NULL, pUserData, pAllocationCallbacks)) {
        return NULL;
    }

    return drmp3__full_read_and_close_s16(&mp3, pConfig, pTotalFrameCount);
}


DRMP3_API float* drmp3_open_memory_and_read_pcm_frames_f32(const void* pData, size_t dataSize, drmp3_config* pConfig, drmp3_uint64* pTotalFrameCount, const drmp3_allocation_callbacks* pAllocationCallbacks)
{
    drmp3 mp3;
    if (!drmp3_init_memory(&mp3, pData, dataSize, pAllocationCallbacks)) {
        return NULL;
    }

    return drmp3__full_read_and_close_f32(&mp3, pConfig, pTotalFrameCount);
}

DRMP3_API drmp3_int16* drmp3_open_memory_and_read_pcm_frames_s16(const void* pData, size_t dataSize, drmp3_config* pConfig, drmp3_uint64* pTotalFrameCount, const drmp3_allocation_callbacks* pAllocationCallbacks)
{
    drmp3 mp3;
    if (!drmp3_init_memory(&mp3, pData, dataSize, pAllocationCallbacks)) {
        return NULL;
    }

    return drmp3__full_read_and_close_s16(&mp3, pConfig, pTotalFrameCount);
}


#ifndef DR_MP3_NO_STDIO
DRMP3_API float* drmp3_open_file_and_read_pcm_frames_f32(const char* filePath, drmp3_config* pConfig, drmp3_uint64* pTotalFrameCount, const drmp3_allocation_callbacks* pAllocationCallbacks)
{
    drmp3 mp3;
    if (!drmp3_init_file(&mp3, filePath, pAllocationCallbacks)) {
        return NULL;
    }

    return drmp3__full_read_and_close_f32(&mp3, pConfig, pTotalFrameCount);
}

DRMP3_API drmp3_int16* drmp3_open_file_and_read_pcm_frames_s16(const char* filePath, drmp3_config* pConfig, drmp3_uint64* pTotalFrameCount, const drmp3_allocation_callbacks* pAllocationCallbacks)
{
    drmp3 mp3;
    if (!drmp3_init_file(&mp3, filePath, pAllocationCallbacks)) {
        return NULL;
    }

    return drmp3__full_read_and_close_s16(&mp3, pConfig, pTotalFrameCount);
}
#endif

DRMP3_API void* drmp3_malloc(size_t sz, const drmp3_allocation_callbacks* pAllocationCallbacks)
{
    if (pAllocationCallbacks != NULL) {
        return drmp3__malloc_from_callbacks(sz, pAllocationCallbacks);
    } else {
        return drmp3__malloc_default(sz, NULL);
    }
}

DRMP3_API void drmp3_free(void* p, const drmp3_allocation_callbacks* pAllocationCallbacks)
{
    if (pAllocationCallbacks != NULL) {
        drmp3__free_from_callbacks(p, pAllocationCallbacks);
    } else {
        drmp3__free_default(p, NULL);
    }
}

#endif  /* dr_mp3_c */
#endif  /*DR_MP3_IMPLEMENTATION*/

/*
DIFFERENCES BETWEEN minimp3 AND dr_mp3
======================================
- First, keep in mind that minimp3 (https://github.com/lieff/minimp3) is where all the real work was done. All of the
  code relating to the actual decoding remains mostly unmodified, apart from some namespacing changes.
- dr_mp3 adds a pulling style API which allows you to deliver raw data via callbacks. So, rather than pushing data
  to the decoder, the decoder _pulls_ data from your callbacks.
- In addition to callbacks, a decoder can be initialized from a block of memory and a file.
- The dr_mp3 pull API reads PCM frames rather than whole MP3 frames.
- dr_mp3 adds convenience APIs for opening and decoding entire files in one go.
- dr_mp3 is fully namespaced, including the implementation section, which is more suitable when compiling projects
  as a single translation unit (aka unity builds). At the time of writing this, a unity build is not possible when
  using minimp3 in conjunction with stb_vorbis. dr_mp3 addresses this.
*/

/*
REVISION HISTORY
================
v0.7.2 - 2025-12-02
  - Reduce stack space to improve robustness on embedded systems.
  - Fix a compilation error with MSVC Clang toolset relating to cpuid.
  - Fix an error with APE tag parsing.

v0.7.1 - 2025-09-10
  - Silence a warning with GCC.
  - Fix an error with the NXDK build.
  - Fix a decoding inconsistency when seeking. Prior to this change, reading to the end of the stream immediately after initializing will result in a different number of samples read than if the stream is seeked to the start and read to the end.

v0.7.0 - 2025-07-23
  - The old `DRMP3_IMPLEMENTATION` has been removed. Use `DR_MP3_IMPLEMENTATION` instead. The reason for this change is that in the future everything will eventually be using the underscored naming convention in the future, so `drmp3` will become `dr_mp3`.
  - API CHANGE: Seek origins have been renamed to match the naming convention used by dr_wav and my other libraries.
    - drmp3_seek_origin_start   -> DRMP3_SEEK_SET
    - drmp3_seek_origin_current -> DRMP3_SEEK_CUR
    - DRMP3_SEEK_END (new)
  - API CHANGE: Add DRMP3_SEEK_END as a seek origin for the seek callback. This is required for detection of ID3v1 and APE tags.
  - API CHANGE: Add onTell callback to `drmp3_init()`. This is needed in order to track the location of ID3v1 and APE tags.
  - API CHANGE: Add onMeta callback to `drmp3_init()`. This is used for reporting tag data back to the caller. Currently this only reports the raw tag data which means applications need to parse the data themselves.
  - API CHANGE: Rename `drmp3dec_frame_info.hz` to `drmp3dec_frame_info.sample_rate`.
  - Add detection of ID3v2, ID3v1, APE and Xing/VBRI tags. This should fix errors with some files where the decoder was reading tags as audio data.
  - Delay and padding samples from LAME tags are now handled.
  - Fix compilation for AIX OS.

v0.6.40 - 2024-12-17
  - Improve detection of ARM64EC

v0.6.39 - 2024-02-27
  - Fix a Wdouble-promotion warning.

v0.6.38 - 2023-11-02
  - Fix build for ARMv6-M.

v0.6.37 - 2023-07-07
  - Silence a static analysis warning.

v0.6.36 - 2023-06-17
  - Fix an incorrect date in revision history. No functional change.

v0.6.35 - 2023-05-22
  - Minor code restructure. No functional change.

v0.6.34 - 2022-09-17
  - Fix compilation with DJGPP.
  - Fix compilation when compiling with x86 with no SSE2.
  - Remove an unnecessary variable from the drmp3 structure.

v0.6.33 - 2022-04-10
  - Fix compilation error with the MSVC ARM64 build.
  - Fix compilation error on older versions of GCC.
  - Remove some unused functions.

v0.6.32 - 2021-12-11
  - Fix a warning with Clang.

v0.6.31 - 2021-08-22
  - Fix a bug when loading from memory.

v0.6.30 - 2021-08-16
  - Silence some warnings.
  - Replace memory operations with DRMP3_* macros.

v0.6.29 - 2021-08-08
  - Bring up to date with minimp3.

v0.6.28 - 2021-07-31
  - Fix platform detection for ARM64.
  - Fix a compilation error with C89.

v0.6.27 - 2021-02-21
  - Fix a warning due to referencing _MSC_VER when it is undefined.

v0.6.26 - 2021-01-31
  - Bring up to date with minimp3.

v0.6.25 - 2020-12-26
  - Remove DRMP3_DEFAULT_CHANNELS and DRMP3_DEFAULT_SAMPLE_RATE which are leftovers from some removed APIs.

v0.6.24 - 2020-12-07
  - Fix a typo in version date for 0.6.23.

v0.6.23 - 2020-12-03
  - Fix an error where a file can be closed twice when initialization of the decoder fails.

v0.6.22 - 2020-12-02
  - Fix an error where it's possible for a file handle to be left open when initialization of the decoder fails.

v0.6.21 - 2020-11-28
  - Bring up to date with minimp3.

v0.6.20 - 2020-11-21
  - Fix compilation with OpenWatcom.

v0.6.19 - 2020-11-13
  - Minor code clean up.

v0.6.18 - 2020-11-01
  - Improve compiler support for older versions of GCC.

v0.6.17 - 2020-09-28
  - Bring up to date with minimp3.

v0.6.16 - 2020-08-02
  - Simplify sized types.

v0.6.15 - 2020-07-25
  - Fix a compilation warning.

v0.6.14 - 2020-07-23
  - Fix undefined behaviour with memmove().

v0.6.13 - 2020-07-06
  - Fix a bug when converting from s16 to f32 in drmp3_read_pcm_frames_f32().

v0.6.12 - 2020-06-23
  - Add include guard for the implementation section.

v0.6.11 - 2020-05-26
  - Fix use of uninitialized variable error.

v0.6.10 - 2020-05-16
  - Add compile-time and run-time version querying.
    - DRMP3_VERSION_MINOR
    - DRMP3_VERSION_MAJOR
    - DRMP3_VERSION_REVISION
    - DRMP3_VERSION_STRING
    - drmp3_version()
    - drmp3_version_string()

v0.6.9 - 2020-04-30
  - Change the `pcm` parameter of drmp3dec_decode_frame() to a `const drmp3_uint8*` for consistency with internal APIs.

v0.6.8 - 2020-04-26
  - Optimizations to decoding when initializing from memory.

v0.6.7 - 2020-04-25
  - Fix a compilation error with DR_MP3_NO_STDIO
  - Optimization to decoding by reducing some data movement.

v0.6.6 - 2020-04-23
  - Fix a minor bug with the running PCM frame counter.

v0.6.5 - 2020-04-19
  - Fix compilation error on ARM builds.

v0.6.4 - 2020-04-19
  - Bring up to date with changes to minimp3.

v0.6.3 - 2020-04-13
  - Fix some pedantic warnings.

v0.6.2 - 2020-04-10
  - Fix a crash in drmp3_open_*_and_read_pcm_frames_*() if the output config object is NULL.

v0.6.1 - 2020-04-05
  - Fix warnings.

v0.6.0 - 2020-04-04
  - API CHANGE: Remove the pConfig parameter from the following APIs:
    - drmp3_init()
    - drmp3_init_memory()
    - drmp3_init_file()
  - Add drmp3_init_file_w() for opening a file from a wchar_t encoded path.

v0.5.6 - 2020-02-12
  - Bring up to date with minimp3.

v0.5.5 - 2020-01-29
  - Fix a memory allocation bug in high level s16 decoding APIs.

v0.5.4 - 2019-12-02
  - Fix a possible null pointer dereference when using custom memory allocators for realloc().

v0.5.3 - 2019-11-14
  - Fix typos in documentation.

v0.5.2 - 2019-11-02
  - Bring up to date with minimp3.

v0.5.1 - 2019-10-08
  - Fix a warning with GCC.

v0.5.0 - 2019-10-07
  - API CHANGE: Add support for user defined memory allocation routines. This system allows the program to specify their own memory allocation
    routines with a user data pointer for client-specific contextual data. This adds an extra parameter to the end of the following APIs:
    - drmp3_init()
    - drmp3_init_file()
    - drmp3_init_memory()
    - drmp3_open_and_read_pcm_frames_f32()
    - drmp3_open_and_read_pcm_frames_s16()
    - drmp3_open_memory_and_read_pcm_frames_f32()
    - drmp3_open_memory_and_read_pcm_frames_s16()
    - drmp3_open_file_and_read_pcm_frames_f32()
    - drmp3_open_file_and_read_pcm_frames_s16()
  - API CHANGE: Renamed the following APIs:
    - drmp3_open_and_read_f32()        -> drmp3_open_and_read_pcm_frames_f32()
    - drmp3_open_and_read_s16()        -> drmp3_open_and_read_pcm_frames_s16()
    - drmp3_open_memory_and_read_f32() -> drmp3_open_memory_and_read_pcm_frames_f32()
    - drmp3_open_memory_and_read_s16() -> drmp3_open_memory_and_read_pcm_frames_s16()
    - drmp3_open_file_and_read_f32()   -> drmp3_open_file_and_read_pcm_frames_f32()
    - drmp3_open_file_and_read_s16()   -> drmp3_open_file_and_read_pcm_frames_s16()

v0.4.7 - 2019-07-28
  - Fix a compiler error.

v0.4.6 - 2019-06-14
  - Fix a compiler error.

v0.4.5 - 2019-06-06
  - Bring up to date with minimp3.

v0.4.4 - 2019-05-06
  - Fixes to the VC6 build.

v0.4.3 - 2019-05-05
  - Use the channel count and/or sample rate of the first MP3 frame instead of DRMP3_DEFAULT_CHANNELS and
    DRMP3_DEFAULT_SAMPLE_RATE when they are set to 0. To use the old behaviour, just set the relevant property to
    DRMP3_DEFAULT_CHANNELS or DRMP3_DEFAULT_SAMPLE_RATE.
  - Add s16 reading APIs
    - drmp3_read_pcm_frames_s16
    - drmp3_open_memory_and_read_pcm_frames_s16
    - drmp3_open_and_read_pcm_frames_s16
    - drmp3_open_file_and_read_pcm_frames_s16
  - Add drmp3_get_mp3_and_pcm_frame_count() to the public header section.
  - Add support for C89.
  - Change license to choice of public domain or MIT-0.

v0.4.2 - 2019-02-21
  - Fix a warning.

v0.4.1 - 2018-12-30
  - Fix a warning.

v0.4.0 - 2018-12-16
  - API CHANGE: Rename some APIs:
    - drmp3_read_f32 -> to drmp3_read_pcm_frames_f32
    - drmp3_seek_to_frame -> drmp3_seek_to_pcm_frame
    - drmp3_open_and_decode_f32 -> drmp3_open_and_read_pcm_frames_f32
    - drmp3_open_and_decode_memory_f32 -> drmp3_open_memory_and_read_pcm_frames_f32
    - drmp3_open_and_decode_file_f32 -> drmp3_open_file_and_read_pcm_frames_f32
  - Add drmp3_get_pcm_frame_count().
  - Add drmp3_get_mp3_frame_count().
  - Improve seeking performance.

v0.3.2 - 2018-09-11
  - Fix a couple of memory leaks.
  - Bring up to date with minimp3.

v0.3.1 - 2018-08-25
  - Fix C++ build.

v0.3.0 - 2018-08-25
  - Bring up to date with minimp3. This has a minor API change: the "pcm" parameter of drmp3dec_decode_frame() has
    been changed from short* to void* because it can now output both s16 and f32 samples, depending on whether or
    not the DR_MP3_FLOAT_OUTPUT option is set.

v0.2.11 - 2018-08-08
  - Fix a bug where the last part of a file is not read.

v0.2.10 - 2018-08-07
  - Improve 64-bit detection.

v0.2.9 - 2018-08-05
  - Fix C++ build on older versions of GCC.
  - Bring up to date with minimp3.

v0.2.8 - 2018-08-02
  - Fix compilation errors with older versions of GCC.

v0.2.7 - 2018-07-13
  - Bring up to date with minimp3.

v0.2.6 - 2018-07-12
  - Bring up to date with minimp3.

v0.2.5 - 2018-06-22
  - Bring up to date with minimp3.

v0.2.4 - 2018-05-12
  - Bring up to date with minimp3.

v0.2.3 - 2018-04-29
  - Fix TCC build.

v0.2.2 - 2018-04-28
  - Fix bug when opening a decoder from memory.

v0.2.1 - 2018-04-27
  - Efficiency improvements when the decoder reaches the end of the stream.

v0.2 - 2018-04-21
  - Bring up to date with minimp3.
  - Start using major.minor.revision versioning.

v0.1d - 2018-03-30
  - Bring up to date with minimp3.

v0.1c - 2018-03-11
  - Fix C++ build error.

v0.1b - 2018-03-07
  - Bring up to date with minimp3.

v0.1a - 2018-02-28
  - Fix compilation error on GCC/Clang.
  - Fix some warnings.

v0.1 - 2018-02-xx
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

/*
    https://github.com/lieff/minimp3
    To the extent possible under law, the author(s) have dedicated all copyright and related and neighboring rights to this software to the public domain worldwide.
    This software is distributed without any warranty.
    See <http://creativecommons.org/publicdomain/zero/1.0/>.
*/
