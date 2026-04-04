/********************************************************************
 *                                                                  *
 * THIS FILE IS PART OF THE libopusfile SOFTWARE CODEC SOURCE CODE. *
 * USE, DISTRIBUTION AND REPRODUCTION OF THIS LIBRARY SOURCE IS     *
 * GOVERNED BY A BSD-STYLE SOURCE LICENSE INCLUDED WITH THIS SOURCE *
 * IN 'COPYING'. PLEASE READ THESE TERMS BEFORE DISTRIBUTING.       *
 *                                                                  *
 * THE libopusfile SOURCE CODE IS (C) COPYRIGHT 2012-2020           *
 * by the Xiph.Org Foundation and contributors https://xiph.org/    *
 *                                                                  *
 ********************************************************************/
#if !defined(_opusfile_internal_h)
# define _opusfile_internal_h (1)

# if !defined(_REENTRANT)
#  define _REENTRANT
# endif
# if !defined(_GNU_SOURCE)
#  define _GNU_SOURCE
# endif
# if !defined(_LARGEFILE_SOURCE)
#  define _LARGEFILE_SOURCE
# endif
# if !defined(_LARGEFILE64_SOURCE)
#  define _LARGEFILE64_SOURCE
# endif
# if !defined(_FILE_OFFSET_BITS)
#  define _FILE_OFFSET_BITS 64
# endif

# include <stdlib.h>
# include <opusfile.h>

typedef struct OggOpusLink OggOpusLink;

# if defined(OP_FIXED_POINT)

typedef opus_int16 op_sample;

# else

typedef float      op_sample;

/*We're using this define to test for libopus 1.1 or later until libopus
   provides a better mechanism.*/
#  if defined(OPUS_GET_EXPERT_FRAME_DURATION_REQUEST)
/*Enable soft clipping prevention in 16-bit decodes.*/
#   define OP_SOFT_CLIP (1)
#  endif

# endif

# if OP_GNUC_PREREQ(4,2)
/*Disable excessive warnings about the order of operations.*/
#  pragma GCC diagnostic ignored "-Wparentheses"
# elif defined(_MSC_VER)
/*Disable excessive warnings about the order of operations.*/
#  pragma warning(disable:4554)
/*Disable warnings about "deprecated" POSIX functions.*/
#  pragma warning(disable:4996)
# endif

# if OP_GNUC_PREREQ(3,0)
/*Another alternative is
    (__builtin_constant_p(_x)?!!(_x):__builtin_expect(!!(_x),1))
   but that evaluates _x multiple times, which may be bad.*/
#  define OP_LIKELY(_x) (__builtin_expect(!!(_x),1))
#  define OP_UNLIKELY(_x) (__builtin_expect(!!(_x),0))
# else
#  define OP_LIKELY(_x)   (!!(_x))
#  define OP_UNLIKELY(_x) (!!(_x))
# endif

# if defined(OP_ENABLE_ASSERTIONS)
#  if OP_GNUC_PREREQ(2,5)||__SUNPRO_C>=0x590
__attribute__((noreturn))
#  endif
void op_fatal_impl(const char *_str,const char *_file,int _line);

#  define OP_FATAL(_str) (op_fatal_impl(_str,__FILE__,__LINE__))

#  define OP_ASSERT(_cond) \
  do{ \
    if(OP_UNLIKELY(!(_cond)))OP_FATAL("assertion failed: " #_cond); \
  } \
  while(0)
#  define OP_ALWAYS_TRUE(_cond) OP_ASSERT(_cond)

# else
#  define OP_FATAL(_str) abort()
#  define OP_ASSERT(_cond)
#  define OP_ALWAYS_TRUE(_cond) ((void)(_cond))
# endif

# define OP_INT64_MAX (2*(((ogg_int64_t)1<<62)-1)|1)
# define OP_INT64_MIN (-OP_INT64_MAX-1)
# define OP_INT32_MAX (2*(((ogg_int32_t)1<<30)-1)|1)
# define OP_INT32_MIN (-OP_INT32_MAX-1)

# define OP_MIN(_a,_b)        ((_a)<(_b)?(_a):(_b))
# define OP_MAX(_a,_b)        ((_a)>(_b)?(_a):(_b))
# define OP_CLAMP(_lo,_x,_hi) (OP_MAX(_lo,OP_MIN(_x,_hi)))

/*Advance a file offset by the given amount, clamping against OP_INT64_MAX.
  This is used to advance a known offset by things like OP_CHUNK_SIZE or
   OP_PAGE_SIZE_MAX, while making sure to avoid signed overflow.
  It assumes that both _offset and _amount are non-negative.*/
#define OP_ADV_OFFSET(_offset,_amount) \
 (OP_MIN(_offset,OP_INT64_MAX-(_amount))+(_amount))

/*The maximum channel count for any mapping we'll actually decode.*/
# define OP_NCHANNELS_MAX (8)

/*Initial state.*/
# define  OP_NOTOPEN   (0)
/*We've found the first Opus stream in the first link.*/
# define  OP_PARTOPEN  (1)
# define  OP_OPENED    (2)
/*We've found the first Opus stream in the current link.*/
# define  OP_STREAMSET (3)
/*We've initialized the decoder for the chosen Opus stream in the current
   link.*/
# define  OP_INITSET   (4)

/*Information cached for a single link in a chained Ogg Opus file.
  We choose the first Opus stream encountered in each link to play back (and
   require at least one).*/
struct OggOpusLink{
  /*The byte offset of the first header page in this link.*/
  opus_int64   offset;
  /*The byte offset of the first data page from the chosen Opus stream in this
     link (after the headers).*/
  opus_int64   data_offset;
  /*The byte offset of the last page from the chosen Opus stream in this link.
    This is used when seeking to ensure we find a page before the last one, so
     that end-trimming calculations work properly.
    This is only valid for seekable sources.*/
  opus_int64   end_offset;
  /*The total duration of all prior links.
    This is always zero for non-seekable sources.*/
  ogg_int64_t  pcm_file_offset;
  /*The granule position of the last sample.
    This is only valid for seekable sources.*/
  ogg_int64_t  pcm_end;
  /*The granule position before the first sample.*/
  ogg_int64_t  pcm_start;
  /*The serial number.*/
  ogg_uint32_t serialno;
  /*The contents of the info header.*/
  OpusHead     head;
  /*The contents of the comment header.*/
  OpusTags     tags;
};

struct OggOpusFile{
  /*The callbacks used to access the stream.*/
  OpusFileCallbacks  callbacks;
  /*A FILE *, memory buffer, etc.*/
  void              *stream;
  /*Whether or not we can seek with this stream.*/
  int                seekable;
  /*The number of links in this chained Ogg Opus file.*/
  int                nlinks;
  /*The cached information from each link in a chained Ogg Opus file.
    If stream isn't seekable (e.g., it's a pipe), only the current link
     appears.*/
  OggOpusLink       *links;
  /*The number of serial numbers from a single link.*/
  int                nserialnos;
  /*The capacity of the list of serial numbers from a single link.*/
  int                cserialnos;
  /*Storage for the list of serial numbers from a single link.
    This is a scratch buffer used when scanning the BOS pages at the start of
     each link.*/
  ogg_uint32_t      *serialnos;
  /*This is the current offset of the data processed by the ogg_sync_state.
    After a seek, this should be set to the target offset so that we can track
     the byte offsets of subsequent pages.
    After a call to op_get_next_page(), this will point to the first byte after
     that page.*/
  opus_int64         offset;
  /*The total size of this stream, or -1 if it's unseekable.*/
  opus_int64         end;
  /*Used to locate pages in the stream.*/
  ogg_sync_state     oy;
  /*One of OP_NOTOPEN, OP_PARTOPEN, OP_OPENED, OP_STREAMSET, OP_INITSET.*/
  int                ready_state;
  /*The current link being played back.*/
  int                cur_link;
  /*The number of decoded samples to discard from the start of decoding.*/
  opus_int32         cur_discard_count;
  /*The granule position of the previous packet (current packet start time).*/
  ogg_int64_t        prev_packet_gp;
  /*The stream offset of the most recent page with completed packets, or -1.
    This is only needed to recover continued packet data in the seeking logic,
     when we use the current position as one of our bounds, only to later
     discover it was the correct starting point.*/
  opus_int64         prev_page_offset;
  /*The number of bytes read since the last bitrate query, including framing.*/
  opus_int64         bytes_tracked;
  /*The number of samples decoded since the last bitrate query.*/
  ogg_int64_t        samples_tracked;
  /*Takes physical pages and welds them into a logical stream of packets.*/
  ogg_stream_state   os;
  /*Re-timestamped packets from a single page.
    Buffering these relies on the undocumented libogg behavior that ogg_packet
     pointers remain valid until the next page is submitted to the
     ogg_stream_state they came from.*/
  ogg_packet         op[255];
  /*The index of the next packet to return.*/
  int                op_pos;
  /*The total number of packets available.*/
  int                op_count;
  /*Central working state for the packet-to-PCM decoder.*/
  OpusMSDecoder     *od;
  /*The application-provided packet decode callback.*/
  op_decode_cb_func  decode_cb;
  /*The application-provided packet decode callback context.*/
  void              *decode_cb_ctx;
  /*The stream count used to initialize the decoder.*/
  int                od_stream_count;
  /*The coupled stream count used to initialize the decoder.*/
  int                od_coupled_count;
  /*The channel count used to initialize the decoder.*/
  int                od_channel_count;
  /*The channel mapping used to initialize the decoder.*/
  unsigned char      od_mapping[OP_NCHANNELS_MAX];
  /*The buffered data for one decoded packet.*/
  op_sample         *od_buffer;
  /*The current position in the decoded buffer.*/
  int                od_buffer_pos;
  /*The number of valid samples in the decoded buffer.*/
  int                od_buffer_size;
  /*The type of gain offset to apply.
    One of OP_HEADER_GAIN, OP_ALBUM_GAIN, OP_TRACK_GAIN, or OP_ABSOLUTE_GAIN.*/
  int                gain_type;
  /*The offset to apply to the gain.*/
  opus_int32         gain_offset_q8;
  /*Internal state for soft clipping and dithering float->short output.*/
#if !defined(OP_FIXED_POINT)
# if defined(OP_SOFT_CLIP)
  float              clip_state[OP_NCHANNELS_MAX];
# endif
  float              dither_a[OP_NCHANNELS_MAX*4];
  float              dither_b[OP_NCHANNELS_MAX*4];
  opus_uint32        dither_seed;
  int                dither_mute;
  int                dither_disabled;
  /*The number of channels represented by the internal state.
    This gets set to 0 whenever anything that would prevent state propagation
     occurs (switching between the float/short APIs, or between the
     stereo/multistream APIs).*/
  int                state_channel_count;
#endif
};

int op_strncasecmp(const char *_a,const char *_b,int _n);

#endif
