/********************************************************************
 *                                                                  *
 * THIS FILE IS PART OF THE libopusfile SOFTWARE CODEC SOURCE CODE. *
 * USE, DISTRIBUTION AND REPRODUCTION OF THIS LIBRARY SOURCE IS     *
 * GOVERNED BY A BSD-STYLE SOURCE LICENSE INCLUDED WITH THIS SOURCE *
 * IN 'COPYING'. PLEASE READ THESE TERMS BEFORE DISTRIBUTING.       *
 *                                                                  *
 * THE libopusfile SOURCE CODE IS (C) COPYRIGHT 1994-2012           *
 * by the Xiph.Org Foundation and contributors https://xiph.org/    *
 *                                                                  *
 ********************************************************************

 function: stdio-based convenience library for opening/seeking/decoding
 last mod: $Id: vorbisfile.h 17182 2010-04-29 03:48:32Z xiphmont $

 ********************************************************************/
#if !defined(_opusfile_h)
# define _opusfile_h (1)

/**\mainpage
   \section Introduction

   This is the documentation for the <tt>libopusfile</tt> C API.

   The <tt>libopusfile</tt> package provides a convenient high-level API for
    decoding and basic manipulation of all Ogg Opus audio streams.
   <tt>libopusfile</tt> is implemented as a layer on top of Xiph.Org's
    reference
    <tt><a href="https://www.xiph.org/ogg/doc/libogg/reference.html">libogg</a></tt>
    and
    <tt><a href="https://opus-codec.org/docs/opus_api-1.3.1/">libopus</a></tt>
    libraries.

   <tt>libopusfile</tt> provides several sets of built-in routines for
    file/stream access, and may also use custom stream I/O routines provided by
    the embedded environment.
   There are built-in I/O routines provided for ANSI-compliant
    <code>stdio</code> (<code>FILE *</code>), memory buffers, and URLs
    (including <file:> URLs, plus optionally <http:> and <https:> URLs).

   \section Organization

   The main API is divided into several sections:
   - \ref stream_open_close
   - \ref stream_info
   - \ref stream_decoding
   - \ref stream_seeking

   Several additional sections are not tied to the main API.
   - \ref stream_callbacks
   - \ref header_info
   - \ref error_codes

   \section Overview

   The <tt>libopusfile</tt> API always decodes files to 48&nbsp;kHz.
   The original sample rate is not preserved by the lossy compression, though
    it is stored in the header to allow you to resample to it after decoding
    (the <tt>libopusfile</tt> API does not currently provide a resampler,
    but the
    <a href="https://www.speex.org/docs/manual/speex-manual/node7.html#SECTION00760000000000000000">the
    Speex resampler</a> is a good choice if you need one).
   In general, if you are playing back the audio, you should leave it at
    48&nbsp;kHz, provided your audio hardware supports it.
   When decoding to a file, it may be worth resampling back to the original
    sample rate, so as not to surprise users who might not expect the sample
    rate to change after encoding to Opus and decoding.

   Opus files can contain anywhere from 1 to 255 channels of audio.
   The channel mappings for up to 8 channels are the same as the
    <a href="https://www.xiph.org/vorbis/doc/Vorbis_I_spec.html#x1-810004.3.9">Vorbis
    mappings</a>.
   A special stereo API can convert everything to 2 channels, making it simple
    to support multichannel files in an application which only has stereo
    output.
   Although the <tt>libopusfile</tt> ABI provides support for the theoretical
    maximum number of channels, the current implementation does not support
    files with more than 8 channels, as they do not have well-defined channel
    mappings.

   Like all Ogg files, Opus files may be "chained".
   That is, multiple Opus files may be combined into a single, longer file just
    by concatenating the original files.
   This is commonly done in internet radio streaming, as it allows the title
    and artist to be updated each time the song changes, since each link in the
    chain includes its own set of metadata.

   <tt>libopusfile</tt> fully supports chained files.
   It will decode the first Opus stream found in each link of a chained file
    (ignoring any other streams that might be concurrently multiplexed with it,
    such as a video stream).

   The channel count can also change between links.
   If your application is not prepared to deal with this, it can use the stereo
    API to ensure the audio from all links will always get decoded into a
    common format.
   Since <tt>libopusfile</tt> always decodes to 48&nbsp;kHz, you do not have to
    worry about the sample rate changing between links (as was possible with
    Vorbis).
   This makes application support for chained files with <tt>libopusfile</tt>
    very easy.*/

# if defined(__cplusplus)
extern "C" {
# endif

# include <stdarg.h>
# include <stdio.h>
# include <ogg/ogg.h>
# include <opus_multistream.h>

/**@cond PRIVATE*/

/*Enable special features for gcc and gcc-compatible compilers.*/
# if !defined(OP_GNUC_PREREQ)
#  if defined(__GNUC__)&&defined(__GNUC_MINOR__)
#   define OP_GNUC_PREREQ(_maj,_min) \
 ((__GNUC__<<16)+__GNUC_MINOR__>=((_maj)<<16)+(_min))
#  else
#   define OP_GNUC_PREREQ(_maj,_min) 0
#  endif
# endif

# if OP_GNUC_PREREQ(4,0)
#  pragma GCC visibility push(default)
# endif

typedef struct OpusHead          OpusHead;
typedef struct OpusTags          OpusTags;
typedef struct OpusPictureTag    OpusPictureTag;
typedef struct OpusServerInfo    OpusServerInfo;
typedef struct OpusFileCallbacks OpusFileCallbacks;
typedef struct OggOpusFile       OggOpusFile;

/*Warning attributes for libopusfile functions.*/
# if OP_GNUC_PREREQ(3,4)
#  define OP_WARN_UNUSED_RESULT __attribute__((__warn_unused_result__))
# else
#  define OP_WARN_UNUSED_RESULT
# endif
# if OP_GNUC_PREREQ(3,4)
#  define OP_ARG_NONNULL(_x) __attribute__((__nonnull__(_x)))
# else
#  define OP_ARG_NONNULL(_x)
# endif

/**@endcond*/

/**\defgroup error_codes Error Codes*/
/*@{*/
/**\name List of possible error codes
   Many of the functions in this library return a negative error code when a
    function fails.
   This list provides a brief explanation of the common errors.
   See each individual function for more details on what a specific error code
    means in that context.*/
/*@{*/

/**A request did not succeed.*/
#define OP_FALSE         (-1)
/*Currently not used externally.*/
#define OP_EOF           (-2)
/**There was a hole in the page sequence numbers (e.g., a page was corrupt or
    missing).*/
#define OP_HOLE          (-3)
/**An underlying read, seek, or tell operation failed when it should have
    succeeded.*/
#define OP_EREAD         (-128)
/**A <code>NULL</code> pointer was passed where one was unexpected, or an
    internal memory allocation failed, or an internal library error was
    encountered.*/
#define OP_EFAULT        (-129)
/**The stream used a feature that is not implemented, such as an unsupported
    channel family.*/
#define OP_EIMPL         (-130)
/**One or more parameters to a function were invalid.*/
#define OP_EINVAL        (-131)
/**A purported Ogg Opus stream did not begin with an Ogg page, a purported
    header packet did not start with one of the required strings, "OpusHead" or
    "OpusTags", or a link in a chained file was encountered that did not
    contain any logical Opus streams.*/
#define OP_ENOTFORMAT    (-132)
/**A required header packet was not properly formatted, contained illegal
    values, or was missing altogether.*/
#define OP_EBADHEADER    (-133)
/**The ID header contained an unrecognized version number.*/
#define OP_EVERSION      (-134)
/*Currently not used at all.*/
#define OP_ENOTAUDIO     (-135)
/**An audio packet failed to decode properly.
   This is usually caused by a multistream Ogg packet where the durations of
    the individual Opus packets contained in it are not all the same.*/
#define OP_EBADPACKET    (-136)
/**We failed to find data we had seen before, or the bitstream structure was
    sufficiently malformed that seeking to the target destination was
    impossible.*/
#define OP_EBADLINK      (-137)
/**An operation that requires seeking was requested on an unseekable stream.*/
#define OP_ENOSEEK       (-138)
/**The first or last granule position of a link failed basic validity checks.*/
#define OP_EBADTIMESTAMP (-139)

/*@}*/
/*@}*/

/**\defgroup header_info Header Information*/
/*@{*/

/**The maximum number of channels in an Ogg Opus stream.*/
#define OPUS_CHANNEL_COUNT_MAX (255)

/**Ogg Opus bitstream information.
   This contains the basic playback parameters for a stream, and corresponds to
    the initial ID header packet of an Ogg Opus stream.*/
struct OpusHead{
  /**The Ogg Opus format version, in the range 0...255.
     The top 4 bits represent a "major" version, and the bottom four bits
      represent backwards-compatible "minor" revisions.
     The current specification describes version 1.
     This library will recognize versions up through 15 as backwards compatible
      with the current specification.
     An earlier draft of the specification described a version 0, but the only
      difference between version 1 and version 0 is that version 0 did
      not specify the semantics for handling the version field.*/
  int           version;
  /**The number of channels, in the range 1...255.*/
  int           channel_count;
  /**The number of samples that should be discarded from the beginning of the
      stream.*/
  unsigned      pre_skip;
  /**The sampling rate of the original input.
     All Opus audio is coded at 48 kHz, and should also be decoded at 48 kHz
      for playback (unless the target hardware does not support this sampling
      rate).
     However, this field may be used to resample the audio back to the original
      sampling rate, for example, when saving the output to a file.*/
  opus_uint32   input_sample_rate;
  /**The gain to apply to the decoded output, in dB, as a Q8 value in the range
      -32768...32767.
     The <tt>libopusfile</tt> API will automatically apply this gain to the
      decoded output before returning it, scaling it by
      <code>pow(10,output_gain/(20.0*256))</code>.
     You can adjust this behavior with op_set_gain_offset().*/
  int           output_gain;
  /**The channel mapping family, in the range 0...255.
     Channel mapping family 0 covers mono or stereo in a single stream.
     Channel mapping family 1 covers 1 to 8 channels in one or more streams,
      using the Vorbis speaker assignments.
     Channel mapping family 255 covers 1 to 255 channels in one or more
      streams, but without any defined speaker assignment.*/
  int           mapping_family;
  /**The number of Opus streams in each Ogg packet, in the range 1...255.*/
  int           stream_count;
  /**The number of coupled Opus streams in each Ogg packet, in the range
      0...127.
     This must satisfy <code>0 <= coupled_count <= stream_count</code> and
      <code>coupled_count + stream_count <= 255</code>.
     The coupled streams appear first, before all uncoupled streams, in an Ogg
      Opus packet.*/
  int           coupled_count;
  /**The mapping from coded stream channels to output channels.
     Let <code>index=mapping[k]</code> be the value for channel <code>k</code>.
     If <code>index<2*coupled_count</code>, then it refers to the left channel
      from stream <code>(index/2)</code> if even, and the right channel from
      stream <code>(index/2)</code> if odd.
     Otherwise, it refers to the output of the uncoupled stream
      <code>(index-coupled_count)</code>.*/
  unsigned char mapping[OPUS_CHANNEL_COUNT_MAX];
};

/**The metadata from an Ogg Opus stream.

   This structure holds the in-stream metadata corresponding to the 'comment'
    header packet of an Ogg Opus stream.
   The comment header is meant to be used much like someone jotting a quick
    note on the label of a CD.
   It should be a short, to the point text note that can be more than a couple
    words, but not more than a short paragraph.

   The metadata is stored as a series of (tag, value) pairs, in length-encoded
    string vectors, using the same format as Vorbis (without the final "framing
    bit"), Theora, and Speex, except for the packet header.
   The first occurrence of the '=' character delimits the tag and value.
   A particular tag may occur more than once, and order is significant.
   The character set encoding for the strings is always UTF-8, but the tag
    names are limited to ASCII, and treated as case-insensitive.
   See <a href="https://www.xiph.org/vorbis/doc/v-comment.html">the Vorbis
    comment header specification</a> for details.

   In filling in this structure, <tt>libopusfile</tt> will null-terminate the
    #user_comments strings for safety.
   However, the bitstream format itself treats them as 8-bit clean vectors,
    possibly containing NUL characters, so the #comment_lengths array should be
    treated as their authoritative length.

   This structure is binary and source-compatible with a
    <code>vorbis_comment</code>, and pointers to it may be freely cast to
    <code>vorbis_comment</code> pointers, and vice versa.
   It is provided as a separate type to avoid introducing a compile-time
    dependency on the libvorbis headers.*/
struct OpusTags{
  /**The array of comment string vectors.*/
  char **user_comments;
  /**An array of the corresponding length of each vector, in bytes.*/
  int   *comment_lengths;
  /**The total number of comment streams.*/
  int    comments;
  /**The null-terminated vendor string.
     This identifies the software used to encode the stream.*/
  char  *vendor;
};

/**\name Picture tag image formats*/
/*@{*/

/**The MIME type was not recognized, or the image data did not match the
    declared MIME type.*/
#define OP_PIC_FORMAT_UNKNOWN (-1)
/**The MIME type indicates the image data is really a URL.*/
#define OP_PIC_FORMAT_URL     (0)
/**The image is a JPEG.*/
#define OP_PIC_FORMAT_JPEG    (1)
/**The image is a PNG.*/
#define OP_PIC_FORMAT_PNG     (2)
/**The image is a GIF.*/
#define OP_PIC_FORMAT_GIF     (3)

/*@}*/

/**The contents of a METADATA_BLOCK_PICTURE tag.*/
struct OpusPictureTag{
  /**The picture type according to the ID3v2 APIC frame:
     <ol start="0">
     <li>Other</li>
     <li>32x32 pixels 'file icon' (PNG only)</li>
     <li>Other file icon</li>
     <li>Cover (front)</li>
     <li>Cover (back)</li>
     <li>Leaflet page</li>
     <li>Media (e.g. label side of CD)</li>
     <li>Lead artist/lead performer/soloist</li>
     <li>Artist/performer</li>
     <li>Conductor</li>
     <li>Band/Orchestra</li>
     <li>Composer</li>
     <li>Lyricist/text writer</li>
     <li>Recording Location</li>
     <li>During recording</li>
     <li>During performance</li>
     <li>Movie/video screen capture</li>
     <li>A bright colored fish</li>
     <li>Illustration</li>
     <li>Band/artist logotype</li>
     <li>Publisher/Studio logotype</li>
     </ol>
     Others are reserved and should not be used.
     There may only be one each of picture type 1 and 2 in a file.*/
  opus_int32     type;
  /**The MIME type of the picture, in printable ASCII characters 0x20-0x7E.
     The MIME type may also be <code>"-->"</code> to signify that the data part
      is a URL pointing to the picture instead of the picture data itself.
     In this case, a terminating NUL is appended to the URL string in #data,
      but #data_length is set to the length of the string excluding that
      terminating NUL.*/
  char          *mime_type;
  /**The description of the picture, in UTF-8.*/
  char          *description;
  /**The width of the picture in pixels.*/
  opus_uint32    width;
  /**The height of the picture in pixels.*/
  opus_uint32    height;
  /**The color depth of the picture in bits-per-pixel (<em>not</em>
      bits-per-channel).*/
  opus_uint32    depth;
  /**For indexed-color pictures (e.g., GIF), the number of colors used, or 0
      for non-indexed pictures.*/
  opus_uint32    colors;
  /**The length of the picture data in bytes.*/
  opus_uint32    data_length;
  /**The binary picture data.*/
  unsigned char *data;
  /**The format of the picture data, if known.
     One of
     <ul>
     <li>#OP_PIC_FORMAT_UNKNOWN,</li>
     <li>#OP_PIC_FORMAT_URL,</li>
     <li>#OP_PIC_FORMAT_JPEG,</li>
     <li>#OP_PIC_FORMAT_PNG, or</li>
     <li>#OP_PIC_FORMAT_GIF.</li>
     </ul>*/
  int            format;
};

/**\name Functions for manipulating header data

   These functions manipulate the #OpusHead and #OpusTags structures,
    which describe the audio parameters and tag-value metadata, respectively.
   These can be used to query the headers returned by <tt>libopusfile</tt>, or
    to parse Opus headers from sources other than an Ogg Opus stream, provided
    they use the same format.*/
/*@{*/

/**Parses the contents of the ID header packet of an Ogg Opus stream.
   \param[out] _head Returns the contents of the parsed packet.
                     The contents of this structure are untouched on error.
                     This may be <code>NULL</code> to merely test the header
                      for validity.
   \param[in]  _data The contents of the ID header packet.
   \param      _len  The number of bytes of data in the ID header packet.
   \return 0 on success or a negative value on error.
   \retval #OP_ENOTFORMAT If the data does not start with the "OpusHead"
                           string.
   \retval #OP_EVERSION   If the version field signaled a version this library
                           does not know how to parse.
   \retval #OP_EIMPL      If the channel mapping family was 255, which general
                           purpose players should not attempt to play.
   \retval #OP_EBADHEADER If the contents of the packet otherwise violate the
                           Ogg Opus specification:
                          <ul>
                           <li>Insufficient data,</li>
                           <li>Too much data for the known minor versions,</li>
                           <li>An unrecognized channel mapping family,</li>
                           <li>Zero channels or too many channels,</li>
                           <li>Zero coded streams,</li>
                           <li>Too many coupled streams, or</li>
                           <li>An invalid channel mapping index.</li>
                          </ul>*/
OP_WARN_UNUSED_RESULT int opus_head_parse(OpusHead *_head,
 const unsigned char *_data,size_t _len) OP_ARG_NONNULL(2);

/**Converts a granule position to a sample offset for a given Ogg Opus stream.
   The sample offset is simply <code>_gp-_head->pre_skip</code>.
   Granule position values smaller than OpusHead#pre_skip correspond to audio
    that should never be played, and thus have no associated sample offset.
   This function returns -1 for such values.
   This function also correctly handles extremely large granule positions,
    which may have wrapped around to a negative number when stored in a signed
    ogg_int64_t value.
   \param _head The #OpusHead information from the ID header of the stream.
   \param _gp   The granule position to convert.
   \return The sample offset associated with the given granule position
            (counting at a 48 kHz sampling rate), or the special value -1 on
            error (i.e., the granule position was smaller than the pre-skip
            amount).*/
ogg_int64_t opus_granule_sample(const OpusHead *_head,ogg_int64_t _gp)
 OP_ARG_NONNULL(1);

/**Parses the contents of the 'comment' header packet of an Ogg Opus stream.
   \param[out] _tags An uninitialized #OpusTags structure.
                     This returns the contents of the parsed packet.
                     The contents of this structure are untouched on error.
                     This may be <code>NULL</code> to merely test the header
                      for validity.
   \param[in]  _data The contents of the 'comment' header packet.
   \param      _len  The number of bytes of data in the 'info' header packet.
   \retval 0              Success.
   \retval #OP_ENOTFORMAT If the data does not start with the "OpusTags"
                           string.
   \retval #OP_EBADHEADER If the contents of the packet otherwise violate the
                           Ogg Opus specification.
   \retval #OP_EFAULT     If there wasn't enough memory to store the tags.*/
OP_WARN_UNUSED_RESULT int opus_tags_parse(OpusTags *_tags,
 const unsigned char *_data,size_t _len) OP_ARG_NONNULL(2);

/**Performs a deep copy of an #OpusTags structure.
   \param _dst The #OpusTags structure to copy into.
               If this function fails, the contents of this structure remain
                untouched.
   \param _src The #OpusTags structure to copy from.
   \retval 0          Success.
   \retval #OP_EFAULT If there wasn't enough memory to copy the tags.*/
int opus_tags_copy(OpusTags *_dst,const OpusTags *_src) OP_ARG_NONNULL(1);

/**Initializes an #OpusTags structure.
   This should be called on a freshly allocated #OpusTags structure before
    attempting to use it.
   \param _tags The #OpusTags structure to initialize.*/
void opus_tags_init(OpusTags *_tags) OP_ARG_NONNULL(1);

/**Add a (tag, value) pair to an initialized #OpusTags structure.
   \note Neither opus_tags_add() nor opus_tags_add_comment() support values
    containing embedded NULs, although the bitstream format does support them.
   To add such tags, you will need to manipulate the #OpusTags structure
    directly.
   \param _tags  The #OpusTags structure to add the (tag, value) pair to.
   \param _tag   A NUL-terminated, case-insensitive, ASCII string containing
                  the tag to add (without an '=' character).
   \param _value A NUL-terminated UTF-8 containing the corresponding value.
   \return 0 on success, or a negative value on failure.
   \retval #OP_EFAULT An internal memory allocation failed.*/
int opus_tags_add(OpusTags *_tags,const char *_tag,const char *_value)
 OP_ARG_NONNULL(1) OP_ARG_NONNULL(2) OP_ARG_NONNULL(3);

/**Add a comment to an initialized #OpusTags structure.
   \note Neither opus_tags_add_comment() nor opus_tags_add() support comments
    containing embedded NULs, although the bitstream format does support them.
   To add such tags, you will need to manipulate the #OpusTags structure
    directly.
   \param _tags    The #OpusTags structure to add the comment to.
   \param _comment A NUL-terminated UTF-8 string containing the comment in
                    "TAG=value" form.
   \return 0 on success, or a negative value on failure.
   \retval #OP_EFAULT An internal memory allocation failed.*/
int opus_tags_add_comment(OpusTags *_tags,const char *_comment)
 OP_ARG_NONNULL(1) OP_ARG_NONNULL(2);

/**Replace the binary suffix data at the end of the packet (if any).
   \param _tags An initialized #OpusTags structure.
   \param _data A buffer of binary data to append after the encoded user
                 comments.
                The least significant bit of the first byte of this data must
                 be set (to ensure the data is preserved by other editors).
   \param _len  The number of bytes of binary data to append.
                This may be zero to remove any existing binary suffix data.
   \return 0 on success, or a negative value on error.
   \retval #OP_EINVAL \a _len was negative, or \a _len was positive but
                       \a _data was <code>NULL</code> or the least significant
                       bit of the first byte was not set.
   \retval #OP_EFAULT An internal memory allocation failed.*/
int opus_tags_set_binary_suffix(OpusTags *_tags,
 const unsigned char *_data,int _len) OP_ARG_NONNULL(1);

/**Look up a comment value by its tag.
   \param _tags  An initialized #OpusTags structure.
   \param _tag   The tag to look up.
   \param _count The instance of the tag.
                 The same tag can appear multiple times, each with a distinct
                  value, so an index is required to retrieve them all.
                 The order in which these values appear is significant and
                  should be preserved.
                 Use opus_tags_query_count() to get the legal range for the
                  \a _count parameter.
   \return A pointer to the queried tag's value.
           This points directly to data in the #OpusTags structure.
           It should not be modified or freed by the application, and
            modifications to the structure may invalidate the pointer.
   \retval NULL If no matching tag is found.*/
const char *opus_tags_query(const OpusTags *_tags,const char *_tag,int _count)
 OP_ARG_NONNULL(1) OP_ARG_NONNULL(2);

/**Look up the number of instances of a tag.
   Call this first when querying for a specific tag and then iterate over the
    number of instances with separate calls to opus_tags_query() to retrieve
    all the values for that tag in order.
   \param _tags An initialized #OpusTags structure.
   \param _tag  The tag to look up.
   \return The number of instances of this particular tag.*/
int opus_tags_query_count(const OpusTags *_tags,const char *_tag)
 OP_ARG_NONNULL(1) OP_ARG_NONNULL(2);

/**Retrieve the binary suffix data at the end of the packet (if any).
   \param      _tags An initialized #OpusTags structure.
   \param[out] _len  Returns the number of bytes of binary suffix data returned.
   \return A pointer to the binary suffix data, or <code>NULL</code> if none
            was present.*/
const unsigned char *opus_tags_get_binary_suffix(const OpusTags *_tags,
 int *_len) OP_ARG_NONNULL(1) OP_ARG_NONNULL(2);

/**Get the album gain from an R128_ALBUM_GAIN tag, if one was specified.
   This searches for the first R128_ALBUM_GAIN tag with a valid signed,
    16-bit decimal integer value and returns the value.
   This routine is exposed merely for convenience for applications which wish
    to do something special with the album gain (i.e., display it).
   If you simply wish to apply the album gain instead of the header gain, you
    can use op_set_gain_offset() with an #OP_ALBUM_GAIN type and no offset.
   \param      _tags    An initialized #OpusTags structure.
   \param[out] _gain_q8 The album gain, in 1/256ths of a dB.
                        This will lie in the range [-32768,32767], and should
                         be applied in <em>addition</em> to the header gain.
                        On error, no value is returned, and the previous
                         contents remain unchanged.
   \return 0 on success, or a negative value on error.
   \retval #OP_FALSE There was no album gain available in the given tags.*/
int opus_tags_get_album_gain(const OpusTags *_tags,int *_gain_q8)
 OP_ARG_NONNULL(1) OP_ARG_NONNULL(2);

/**Get the track gain from an R128_TRACK_GAIN tag, if one was specified.
   This searches for the first R128_TRACK_GAIN tag with a valid signed,
    16-bit decimal integer value and returns the value.
   This routine is exposed merely for convenience for applications which wish
    to do something special with the track gain (i.e., display it).
   If you simply wish to apply the track gain instead of the header gain, you
    can use op_set_gain_offset() with an #OP_TRACK_GAIN type and no offset.
   \param      _tags    An initialized #OpusTags structure.
   \param[out] _gain_q8 The track gain, in 1/256ths of a dB.
                        This will lie in the range [-32768,32767], and should
                         be applied in <em>addition</em> to the header gain.
                        On error, no value is returned, and the previous
                         contents remain unchanged.
   \return 0 on success, or a negative value on error.
   \retval #OP_FALSE There was no track gain available in the given tags.*/
int opus_tags_get_track_gain(const OpusTags *_tags,int *_gain_q8)
 OP_ARG_NONNULL(1) OP_ARG_NONNULL(2);

/**Clears the #OpusTags structure.
   This should be called on an #OpusTags structure after it is no longer
    needed.
   It will free all memory used by the structure members.
   \param _tags The #OpusTags structure to clear.*/
void opus_tags_clear(OpusTags *_tags) OP_ARG_NONNULL(1);

/**Check if \a _comment is an instance of a \a _tag_name tag.
   \see opus_tagncompare
   \param _tag_name A NUL-terminated, case-insensitive, ASCII string containing
                     the name of the tag to check for (without the terminating
                     '=' character).
   \param _comment  The comment string to check.
   \return An integer less than, equal to, or greater than zero if \a _comment
            is found respectively, to be less than, to match, or be greater
            than a "tag=value" string whose tag matches \a _tag_name.*/
int opus_tagcompare(const char *_tag_name,const char *_comment);

/**Check if \a _comment is an instance of a \a _tag_name tag.
   This version is slightly more efficient than opus_tagcompare() if the length
    of the tag name is already known (e.g., because it is a constant).
   \see opus_tagcompare
   \param _tag_name A case-insensitive ASCII string containing the name of the
                     tag to check for (without the terminating '=' character).
   \param _tag_len  The number of characters in the tag name.
                    This must be non-negative.
   \param _comment  The comment string to check.
   \return An integer less than, equal to, or greater than zero if \a _comment
            is found respectively, to be less than, to match, or be greater
            than a "tag=value" string whose tag matches the first \a _tag_len
            characters of \a _tag_name.*/
int opus_tagncompare(const char *_tag_name,int _tag_len,const char *_comment);

/**Parse a single METADATA_BLOCK_PICTURE tag.
   This decodes the BASE64-encoded content of the tag and returns a structure
    with the MIME type, description, image parameters (if known), and the
    compressed image data.
   If the MIME type indicates the presence of an image format we recognize
    (JPEG, PNG, or GIF) and the actual image data contains the magic signature
    associated with that format, then the OpusPictureTag::format field will be
    set to the corresponding format.
   This is provided as a convenience to avoid requiring applications to parse
    the MIME type and/or do their own format detection for the commonly used
    formats.
   In this case, we also attempt to extract the image parameters directly from
    the image data (overriding any that were present in the tag, which the
    specification says applications are not meant to rely on).
   The application must still provide its own support for actually decoding the
    image data and, if applicable, retrieving that data from URLs.
   \param[out] _pic Returns the parsed picture data.
                    No sanitation is done on the type, MIME type, or
                     description fields, so these might return invalid values.
                    The contents of this structure are left unmodified on
                     failure.
   \param      _tag The METADATA_BLOCK_PICTURE tag contents.
                    The leading "METADATA_BLOCK_PICTURE=" portion is optional,
                     to allow the function to be used on either directly on the
                     values in OpusTags::user_comments or on the return value
                     of opus_tags_query().
   \return 0 on success or a negative value on error.
   \retval #OP_ENOTFORMAT The METADATA_BLOCK_PICTURE contents were not valid.
   \retval #OP_EFAULT     There was not enough memory to store the picture tag
                           contents.*/
OP_WARN_UNUSED_RESULT int opus_picture_tag_parse(OpusPictureTag *_pic,
 const char *_tag) OP_ARG_NONNULL(1) OP_ARG_NONNULL(2);

/**Initializes an #OpusPictureTag structure.
   This should be called on a freshly allocated #OpusPictureTag structure
    before attempting to use it.
   \param _pic The #OpusPictureTag structure to initialize.*/
void opus_picture_tag_init(OpusPictureTag *_pic) OP_ARG_NONNULL(1);

/**Clears the #OpusPictureTag structure.
   This should be called on an #OpusPictureTag structure after it is no longer
    needed.
   It will free all memory used by the structure members.
   \param _pic The #OpusPictureTag structure to clear.*/
void opus_picture_tag_clear(OpusPictureTag *_pic) OP_ARG_NONNULL(1);

/*@}*/

/*@}*/

/**\defgroup url_options URL Reading Options*/
/*@{*/
/**\name URL reading options
   Options for op_url_stream_create() and associated functions.
   These allow you to provide proxy configuration parameters, skip SSL
    certificate checks, etc.
   Options are processed in order, and if the same option is passed multiple
    times, only the value specified by the last occurrence has an effect
    (unless otherwise specified).
   They may be expanded in the future.*/
/*@{*/

/**@cond PRIVATE*/

/*These are the raw numbers used to define the request codes.
  They should not be used directly.*/
#define OP_SSL_SKIP_CERTIFICATE_CHECK_REQUEST (6464)
#define OP_HTTP_PROXY_HOST_REQUEST            (6528)
#define OP_HTTP_PROXY_PORT_REQUEST            (6592)
#define OP_HTTP_PROXY_USER_REQUEST            (6656)
#define OP_HTTP_PROXY_PASS_REQUEST            (6720)
#define OP_GET_SERVER_INFO_REQUEST            (6784)

#define OP_URL_OPT(_request) ((char *)(_request))

/*These macros trigger compilation errors or warnings if the wrong types are
   provided to one of the URL options.*/
#define OP_CHECK_INT(_x) ((void)((_x)==(opus_int32)0),(opus_int32)(_x))
#define OP_CHECK_CONST_CHAR_PTR(_x) ((_x)+((_x)-(const char *)(_x)))
#define OP_CHECK_SERVER_INFO_PTR(_x) ((_x)+((_x)-(OpusServerInfo *)(_x)))

/**@endcond*/

/**HTTP/Shoutcast/Icecast server information associated with a URL.*/
struct OpusServerInfo{
  /**The name of the server (icy-name/ice-name).
     This is <code>NULL</code> if there was no <code>icy-name</code> or
      <code>ice-name</code> header.*/
  char        *name;
  /**A short description of the server (icy-description/ice-description).
     This is <code>NULL</code> if there was no <code>icy-description</code> or
      <code>ice-description</code> header.*/
  char        *description;
  /**The genre the server falls under (icy-genre/ice-genre).
     This is <code>NULL</code> if there was no <code>icy-genre</code> or
      <code>ice-genre</code> header.*/
  char        *genre;
  /**The homepage for the server (icy-url/ice-url).
     This is <code>NULL</code> if there was no <code>icy-url</code> or
      <code>ice-url</code> header.*/
  char        *url;
  /**The software used by the origin server (Server).
     This is <code>NULL</code> if there was no <code>Server</code> header.*/
  char        *server;
  /**The media type of the entity sent to the recepient (Content-Type).
     This is <code>NULL</code> if there was no <code>Content-Type</code>
      header.*/
  char        *content_type;
  /**The nominal stream bitrate in kbps (icy-br/ice-bitrate).
     This is <code>-1</code> if there was no <code>icy-br</code> or
      <code>ice-bitrate</code> header.*/
  opus_int32   bitrate_kbps;
  /**Flag indicating whether the server is public (<code>1</code>) or not
      (<code>0</code>) (icy-pub/ice-public).
     This is <code>-1</code> if there was no <code>icy-pub</code> or
      <code>ice-public</code> header.*/
  int          is_public;
  /**Flag indicating whether the server is using HTTPS instead of HTTP.
     This is <code>0</code> unless HTTPS is being used.
     This may not match the protocol used in the original URL if there were
      redirections.*/
  int          is_ssl;
};

/**Initializes an #OpusServerInfo structure.
   All fields are set as if the corresponding header was not available.
   \param _info The #OpusServerInfo structure to initialize.
   \note If you use this function, you must link against <tt>libopusurl</tt>.*/
void opus_server_info_init(OpusServerInfo *_info) OP_ARG_NONNULL(1);

/**Clears the #OpusServerInfo structure.
   This should be called on an #OpusServerInfo structure after it is no longer
    needed.
   It will free all memory used by the structure members.
   \param _info The #OpusServerInfo structure to clear.
   \note If you use this function, you must link against <tt>libopusurl</tt>.*/
void opus_server_info_clear(OpusServerInfo *_info) OP_ARG_NONNULL(1);

/**Skip the certificate check when connecting via TLS/SSL (https).
   \param _b <code>opus_int32</code>: Whether or not to skip the certificate
              check.
             The check will be skipped if \a _b is non-zero, and will not be
              skipped if \a _b is zero.
   \hideinitializer*/
#define OP_SSL_SKIP_CERTIFICATE_CHECK(_b) \
 OP_URL_OPT(OP_SSL_SKIP_CERTIFICATE_CHECK_REQUEST),OP_CHECK_INT(_b)

/**Proxy connections through the given host.
   If no port is specified via #OP_HTTP_PROXY_PORT, the port number defaults
    to 8080 (http-alt).
   All proxy parameters are ignored for non-http and non-https URLs.
   \param _host <code>const char *</code>: The proxy server hostname.
                This may be <code>NULL</code> to disable the use of a proxy
                 server.
   \hideinitializer*/
#define OP_HTTP_PROXY_HOST(_host) \
 OP_URL_OPT(OP_HTTP_PROXY_HOST_REQUEST),OP_CHECK_CONST_CHAR_PTR(_host)

/**Use the given port when proxying connections.
   This option only has an effect if #OP_HTTP_PROXY_HOST is specified with a
    non-<code>NULL</code> \a _host.
   If this option is not provided, the proxy port number defaults to 8080
    (http-alt).
   All proxy parameters are ignored for non-http and non-https URLs.
   \param _port <code>opus_int32</code>: The proxy server port.
                This must be in the range 0...65535 (inclusive), or the
                 URL function this is passed to will fail.
   \hideinitializer*/
#define OP_HTTP_PROXY_PORT(_port) \
 OP_URL_OPT(OP_HTTP_PROXY_PORT_REQUEST),OP_CHECK_INT(_port)

/**Use the given user name for authentication when proxying connections.
   All proxy parameters are ignored for non-http and non-https URLs.
   \param _user const char *: The proxy server user name.
                              This may be <code>NULL</code> to disable proxy
                               authentication.
                              A non-<code>NULL</code> value only has an effect
                               if #OP_HTTP_PROXY_HOST and #OP_HTTP_PROXY_PASS
                               are also specified with non-<code>NULL</code>
                               arguments.
   \hideinitializer*/
#define OP_HTTP_PROXY_USER(_user) \
 OP_URL_OPT(OP_HTTP_PROXY_USER_REQUEST),OP_CHECK_CONST_CHAR_PTR(_user)

/**Use the given password for authentication when proxying connections.
   All proxy parameters are ignored for non-http and non-https URLs.
   \param _pass const char *: The proxy server password.
                              This may be <code>NULL</code> to disable proxy
                               authentication.
                              A non-<code>NULL</code> value only has an effect
                               if #OP_HTTP_PROXY_HOST and #OP_HTTP_PROXY_USER
                               are also specified with non-<code>NULL</code>
                               arguments.
   \hideinitializer*/
#define OP_HTTP_PROXY_PASS(_pass) \
 OP_URL_OPT(OP_HTTP_PROXY_PASS_REQUEST),OP_CHECK_CONST_CHAR_PTR(_pass)

/**Parse information about the streaming server (if any) and return it.
   Very little validation is done.
   In particular, OpusServerInfo::url may not be a valid URL,
    OpusServerInfo::bitrate_kbps may not really be in kbps, and
    OpusServerInfo::content_type may not be a valid MIME type.
   The character set of the string fields is not specified anywhere, and should
    not be assumed to be valid UTF-8.
   \param _info OpusServerInfo *: Returns information about the server.
                                  If there is any error opening the stream, the
                                   contents of this structure remain
                                   unmodified.
                                  On success, fills in the structure with the
                                   server information that was available, if
                                   any.
                                  After a successful return, the contents of
                                   this structure should be freed by calling
                                   opus_server_info_clear().
   \hideinitializer*/
#define OP_GET_SERVER_INFO(_info) \
 OP_URL_OPT(OP_GET_SERVER_INFO_REQUEST),OP_CHECK_SERVER_INFO_PTR(_info)

/*@}*/
/*@}*/

/**\defgroup stream_callbacks Abstract Stream Reading Interface*/
/*@{*/
/**\name Functions for reading from streams
   These functions define the interface used to read from and seek in a stream
    of data.
   A stream does not need to implement seeking, but the decoder will not be
    able to seek if it does not do so.
   These functions also include some convenience routines for working with
    standard <code>FILE</code> pointers, complete streams stored in a single
    block of memory, or URLs.*/
/*@{*/

/**Reads up to \a _nbytes bytes of data from \a _stream.
   \param      _stream The stream to read from.
   \param[out] _ptr    The buffer to store the data in.
   \param      _nbytes The maximum number of bytes to read.
                       This function may return fewer, though it will not
                        return zero unless it reaches end-of-file.
   \return The number of bytes successfully read, or a negative value on
            error.*/
typedef int (*op_read_func)(void *_stream,unsigned char *_ptr,int _nbytes);

/**Sets the position indicator for \a _stream.
   The new position, measured in bytes, is obtained by adding \a _offset
    bytes to the position specified by \a _whence.
   If \a _whence is set to <code>SEEK_SET</code>, <code>SEEK_CUR</code>, or
    <code>SEEK_END</code>, the offset is relative to the start of the stream,
    the current position indicator, or end-of-file, respectively.
   \retval 0  Success.
   \retval -1 Seeking is not supported or an error occurred.
              <code>errno</code> need not be set.*/
typedef int (*op_seek_func)(void *_stream,opus_int64 _offset,int _whence);

/**Obtains the current value of the position indicator for \a _stream.
   \return The current position indicator.*/
typedef opus_int64 (*op_tell_func)(void *_stream);

/**Closes the underlying stream.
   \retval 0   Success.
   \retval EOF An error occurred.
               <code>errno</code> need not be set.*/
typedef int (*op_close_func)(void *_stream);

/**The callbacks used to access non-<code>FILE</code> stream resources.
   The function prototypes are basically the same as for the stdio functions
    <code>fread()</code>, <code>fseek()</code>, <code>ftell()</code>, and
    <code>fclose()</code>.
   The differences are that the <code>FILE *</code> arguments have been
    replaced with a <code>void *</code>, which is to be used as a pointer to
    whatever internal data these functions might need, that #seek and #tell
    take and return 64-bit offsets, and that #seek <em>must</em> return -1 if
    the stream is unseekable.*/
struct OpusFileCallbacks{
  /**Used to read data from the stream.
     This must not be <code>NULL</code>.*/
  op_read_func  read;
  /**Used to seek in the stream.
     This may be <code>NULL</code> if seeking is not implemented.*/
  op_seek_func  seek;
  /**Used to return the current read position in the stream.
     This may be <code>NULL</code> if seeking is not implemented.*/
  op_tell_func  tell;
  /**Used to close the stream when the decoder is freed.
     This may be <code>NULL</code> to leave the stream open.*/
  op_close_func close;
};

/**Opens a stream with <code>fopen()</code> and fills in a set of callbacks
    that can be used to access it.
   This is useful to avoid writing your own portable 64-bit seeking wrappers,
    and also avoids cross-module linking issues on Windows, where a
    <code>FILE *</code> must be accessed by routines defined in the same module
    that opened it.
   \param[out] _cb   The callbacks to use for this file.
                     If there is an error opening the file, nothing will be
                      filled in here.
   \param      _path The path to the file to open.
                     On Windows, this string must be UTF-8 (to allow access to
                      files whose names cannot be represented in the current
                      MBCS code page).
                     All other systems use the native character encoding.
   \param      _mode The mode to open the file in.
   \return A stream handle to use with the callbacks, or <code>NULL</code> on
            error.*/
OP_WARN_UNUSED_RESULT void *op_fopen(OpusFileCallbacks *_cb,
 const char *_path,const char *_mode) OP_ARG_NONNULL(1) OP_ARG_NONNULL(2)
 OP_ARG_NONNULL(3);

/**Opens a stream with <code>fdopen()</code> and fills in a set of callbacks
    that can be used to access it.
   This is useful to avoid writing your own portable 64-bit seeking wrappers,
    and also avoids cross-module linking issues on Windows, where a
    <code>FILE *</code> must be accessed by routines defined in the same module
    that opened it.
   \param[out] _cb   The callbacks to use for this file.
                     If there is an error opening the file, nothing will be
                      filled in here.
   \param      _fd   The file descriptor to open.
   \param      _mode The mode to open the file in.
   \return A stream handle to use with the callbacks, or <code>NULL</code> on
            error.*/
OP_WARN_UNUSED_RESULT void *op_fdopen(OpusFileCallbacks *_cb,
 int _fd,const char *_mode) OP_ARG_NONNULL(1) OP_ARG_NONNULL(3);

/**Opens a stream with <code>freopen()</code> and fills in a set of callbacks
    that can be used to access it.
   This is useful to avoid writing your own portable 64-bit seeking wrappers,
    and also avoids cross-module linking issues on Windows, where a
    <code>FILE *</code> must be accessed by routines defined in the same module
    that opened it.
   \param[out] _cb     The callbacks to use for this file.
                       If there is an error opening the file, nothing will be
                        filled in here.
   \param      _path   The path to the file to open.
                       On Windows, this string must be UTF-8 (to allow access
                        to files whose names cannot be represented in the
                        current MBCS code page).
                       All other systems use the native character encoding.
   \param      _mode   The mode to open the file in.
   \param      _stream A stream previously returned by op_fopen(), op_fdopen(),
                        or op_freopen().
   \return A stream handle to use with the callbacks, or <code>NULL</code> on
            error.*/
OP_WARN_UNUSED_RESULT void *op_freopen(OpusFileCallbacks *_cb,
 const char *_path,const char *_mode,void *_stream) OP_ARG_NONNULL(1)
 OP_ARG_NONNULL(2) OP_ARG_NONNULL(3) OP_ARG_NONNULL(4);

/**Creates a stream that reads from the given block of memory.
   This block of memory must contain the complete stream to decode.
   This is useful for caching small streams (e.g., sound effects) in RAM.
   \param[out] _cb   The callbacks to use for this stream.
                     If there is an error creating the stream, nothing will be
                      filled in here.
   \param      _data The block of memory to read from.
   \param      _size The size of the block of memory.
   \return A stream handle to use with the callbacks, or <code>NULL</code> on
            error.*/
OP_WARN_UNUSED_RESULT void *op_mem_stream_create(OpusFileCallbacks *_cb,
 const unsigned char *_data,size_t _size) OP_ARG_NONNULL(1);

/**Creates a stream that reads from the given URL.
   This function behaves identically to op_url_stream_create(), except that it
    takes a va_list instead of a variable number of arguments.
   It does not call the <code>va_end</code> macro, and because it invokes the
    <code>va_arg</code> macro, the value of \a _ap is undefined after the call.
   \note If you use this function, you must link against <tt>libopusurl</tt>.
   \param[out]    _cb  The callbacks to use for this stream.
                       If there is an error creating the stream, nothing will
                        be filled in here.
   \param         _url The URL to read from.
                       Currently only the <file:>, <http:>, and <https:>
                        schemes are supported.
                       Both <http:> and <https:> may be disabled at compile
                        time, in which case opening such URLs will always fail.
                       Currently this only supports URIs.
                       IRIs should be converted to UTF-8 and URL-escaped, with
                        internationalized domain names encoded in punycode,
                        before passing them to this function.
   \param[in,out] _ap  A list of the \ref url_options "optional flags" to use.
                       This is a variable-length list of options terminated
                        with <code>NULL</code>.
   \return A stream handle to use with the callbacks, or <code>NULL</code> on
            error.*/
OP_WARN_UNUSED_RESULT void *op_url_stream_vcreate(OpusFileCallbacks *_cb,
 const char *_url,va_list _ap) OP_ARG_NONNULL(1) OP_ARG_NONNULL(2);

/**Creates a stream that reads from the given URL.
   \note If you use this function, you must link against <tt>libopusurl</tt>.
   \param[out] _cb  The callbacks to use for this stream.
                    If there is an error creating the stream, nothing will be
                     filled in here.
   \param      _url The URL to read from.
                    Currently only the <file:>, <http:>, and <https:> schemes
                     are supported.
                    Both <http:> and <https:> may be disabled at compile time,
                     in which case opening such URLs will always fail.
                    Currently this only supports URIs.
                    IRIs should be converted to UTF-8 and URL-escaped, with
                     internationalized domain names encoded in punycode, before
                     passing them to this function.
   \param      ...  The \ref url_options "optional flags" to use.
                    This is a variable-length list of options terminated with
                     <code>NULL</code>.
   \return A stream handle to use with the callbacks, or <code>NULL</code> on
            error.*/
OP_WARN_UNUSED_RESULT void *op_url_stream_create(OpusFileCallbacks *_cb,
 const char *_url,...) OP_ARG_NONNULL(1) OP_ARG_NONNULL(2);

/*@}*/
/*@}*/

/**\defgroup stream_open_close Opening and Closing*/
/*@{*/
/**\name Functions for opening and closing streams

   These functions allow you to test a stream to see if it is Opus, open it,
    and close it.
   Several flavors are provided for each of the built-in stream types, plus a
    more general version which takes a set of application-provided callbacks.*/
/*@{*/

/**Test to see if this is an Opus stream.
   For good results, you will need at least 57 bytes (for a pure Opus-only
    stream).
   Something like 512 bytes will give more reliable results for multiplexed
    streams.
   This function is meant to be a quick-rejection filter.
   Its purpose is not to guarantee that a stream is a valid Opus stream, but to
    ensure that it looks enough like Opus that it isn't going to be recognized
    as some other format (except possibly an Opus stream that is also
    multiplexed with other codecs, such as video).
   \param[out] _head     The parsed ID header contents.
                         You may pass <code>NULL</code> if you do not need
                          this information.
                         If the function fails, the contents of this structure
                          remain untouched.
   \param _initial_data  An initial buffer of data from the start of the
                          stream.
   \param _initial_bytes The number of bytes in \a _initial_data.
   \return 0 if the data appears to be Opus, or a negative value on error.
   \retval #OP_FALSE      There was not enough data to tell if this was an Opus
                           stream or not.
   \retval #OP_EFAULT     An internal memory allocation failed.
   \retval #OP_EIMPL      The stream used a feature that is not implemented,
                           such as an unsupported channel family.
   \retval #OP_ENOTFORMAT If the data did not contain a recognizable ID
                           header for an Opus stream.
   \retval #OP_EVERSION   If the version field signaled a version this library
                           does not know how to parse.
   \retval #OP_EBADHEADER The ID header was not properly formatted or contained
                           illegal values.*/
int op_test(OpusHead *_head,
 const unsigned char *_initial_data,size_t _initial_bytes);

/**Open a stream from the given file path.
   \param      _path  The path to the file to open.
   \param[out] _error Returns 0 on success, or a failure code on error.
                      You may pass in <code>NULL</code> if you don't want the
                       failure code.
                      The failure code will be #OP_EFAULT if the file could not
                       be opened, or one of the other failure codes from
                       op_open_callbacks() otherwise.
   \return A freshly opened \c OggOpusFile, or <code>NULL</code> on error.*/
OP_WARN_UNUSED_RESULT OggOpusFile *op_open_file(const char *_path,int *_error)
 OP_ARG_NONNULL(1);

/**Open a stream from a memory buffer.
   \param      _data  The memory buffer to open.
   \param      _size  The number of bytes in the buffer.
   \param[out] _error Returns 0 on success, or a failure code on error.
                      You may pass in <code>NULL</code> if you don't want the
                       failure code.
                      See op_open_callbacks() for a full list of failure codes.
   \return A freshly opened \c OggOpusFile, or <code>NULL</code> on error.*/
OP_WARN_UNUSED_RESULT OggOpusFile *op_open_memory(const unsigned char *_data,
 size_t _size,int *_error);

/**Open a stream from a URL.
   This function behaves identically to op_open_url(), except that it
    takes a va_list instead of a variable number of arguments.
   It does not call the <code>va_end</code> macro, and because it invokes the
    <code>va_arg</code> macro, the value of \a _ap is undefined after the call.
   \note If you use this function, you must link against <tt>libopusurl</tt>.
   \param         _url   The URL to open.
                         Currently only the <file:>, <http:>, and <https:>
                          schemes are supported.
                         Both <http:> and <https:> may be disabled at compile
                          time, in which case opening such URLs will always
                          fail.
                         Currently this only supports URIs.
                         IRIs should be converted to UTF-8 and URL-escaped,
                          with internationalized domain names encoded in
                          punycode, before passing them to this function.
   \param[out]    _error Returns 0 on success, or a failure code on error.
                         You may pass in <code>NULL</code> if you don't want
                          the failure code.
                         See op_open_callbacks() for a full list of failure
                          codes.
   \param[in,out] _ap    A list of the \ref url_options "optional flags" to
                          use.
                         This is a variable-length list of options terminated
                          with <code>NULL</code>.
   \return A freshly opened \c OggOpusFile, or <code>NULL</code> on error.*/
OP_WARN_UNUSED_RESULT OggOpusFile *op_vopen_url(const char *_url,
 int *_error,va_list _ap) OP_ARG_NONNULL(1);

/**Open a stream from a URL.
   \note If you use this function, you must link against <tt>libopusurl</tt>.
   \param      _url   The URL to open.
                      Currently only the <file:>, <http:>, and <https:> schemes
                       are supported.
                      Both <http:> and <https:> may be disabled at compile
                       time, in which case opening such URLs will always fail.
                      Currently this only supports URIs.
                      IRIs should be converted to UTF-8 and URL-escaped, with
                       internationalized domain names encoded in punycode,
                       before passing them to this function.
   \param[out] _error Returns 0 on success, or a failure code on error.
                      You may pass in <code>NULL</code> if you don't want the
                       failure code.
                      See op_open_callbacks() for a full list of failure codes.
   \param      ...    The \ref url_options "optional flags" to use.
                      This is a variable-length list of options terminated with
                       <code>NULL</code>.
   \return A freshly opened \c OggOpusFile, or <code>NULL</code> on error.*/
OP_WARN_UNUSED_RESULT OggOpusFile *op_open_url(const char *_url,
 int *_error,...) OP_ARG_NONNULL(1);

/**Open a stream using the given set of callbacks to access it.
   \param _stream        The stream to read from (e.g., a <code>FILE *</code>).
                         This value will be passed verbatim as the first
                          argument to all of the callbacks.
   \param _cb            The callbacks with which to access the stream.
                         <code><a href="#op_read_func">read()</a></code> must
                          be implemented.
                         <code><a href="#op_seek_func">seek()</a></code> and
                          <code><a href="#op_tell_func">tell()</a></code> may
                          be <code>NULL</code>, or may always return -1 to
                          indicate a stream is unseekable, but if
                          <code><a href="#op_seek_func">seek()</a></code> is
                          implemented and succeeds on a particular stream, then
                          <code><a href="#op_tell_func">tell()</a></code> must
                          also.
                         <code><a href="#op_close_func">close()</a></code> may
                          be <code>NULL</code>, but if it is not, it will be
                          called when the \c OggOpusFile is destroyed by
                          op_free().
                         It will not be called if op_open_callbacks() fails
                          with an error.
   \param _initial_data  An initial buffer of data from the start of the
                          stream.
                         Applications can read some number of bytes from the
                          start of the stream to help identify this as an Opus
                          stream, and then provide them here to allow the
                          stream to be opened, even if it is unseekable.
   \param _initial_bytes The number of bytes in \a _initial_data.
                         If the stream is seekable, its current position (as
                          reported by
                          <code><a href="#opus_tell_func">tell()</a></code>
                          at the start of this function) must be equal to
                          \a _initial_bytes.
                         Otherwise, seeking to absolute positions will
                          generate inconsistent results.
   \param[out] _error    Returns 0 on success, or a failure code on error.
                         You may pass in <code>NULL</code> if you don't want
                          the failure code.
                         The failure code will be one of
                         <dl>
                           <dt>#OP_EREAD</dt>
                           <dd>An underlying read, seek, or tell operation
                            failed when it should have succeeded, or we failed
                            to find data in the stream we had seen before.</dd>
                           <dt>#OP_EFAULT</dt>
                           <dd>There was a memory allocation failure, or an
                            internal library error.</dd>
                           <dt>#OP_EIMPL</dt>
                           <dd>The stream used a feature that is not
                            implemented, such as an unsupported channel
                            family.</dd>
                           <dt>#OP_EINVAL</dt>
                           <dd><code><a href="#op_seek_func">seek()</a></code>
                            was implemented and succeeded on this source, but
                            <code><a href="#op_tell_func">tell()</a></code>
                            did not, or the starting position indicator was
                            not equal to \a _initial_bytes.</dd>
                           <dt>#OP_ENOTFORMAT</dt>
                           <dd>The stream contained a link that did not have
                            any logical Opus streams in it.</dd>
                           <dt>#OP_EBADHEADER</dt>
                           <dd>A required header packet was not properly
                            formatted, contained illegal values, or was missing
                            altogether.</dd>
                           <dt>#OP_EVERSION</dt>
                           <dd>An ID header contained an unrecognized version
                            number.</dd>
                           <dt>#OP_EBADLINK</dt>
                           <dd>We failed to find data we had seen before after
                            seeking.</dd>
                           <dt>#OP_EBADTIMESTAMP</dt>
                           <dd>The first or last timestamp in a link failed
                            basic validity checks.</dd>
                         </dl>
   \return A freshly opened \c OggOpusFile, or <code>NULL</code> on error.
           <tt>libopusfile</tt> does <em>not</em> take ownership of the stream
            if the call fails.
           The calling application is responsible for closing the stream if
            this call returns an error.*/
OP_WARN_UNUSED_RESULT OggOpusFile *op_open_callbacks(void *_stream,
 const OpusFileCallbacks *_cb,const unsigned char *_initial_data,
 size_t _initial_bytes,int *_error) OP_ARG_NONNULL(2);

/**Partially open a stream from the given file path.
   \see op_test_callbacks
   \param      _path  The path to the file to open.
   \param[out] _error Returns 0 on success, or a failure code on error.
                      You may pass in <code>NULL</code> if you don't want the
                       failure code.
                      The failure code will be #OP_EFAULT if the file could not
                       be opened, or one of the other failure codes from
                       op_open_callbacks() otherwise.
   \return A partially opened \c OggOpusFile, or <code>NULL</code> on error.*/
OP_WARN_UNUSED_RESULT OggOpusFile *op_test_file(const char *_path,int *_error)
 OP_ARG_NONNULL(1);

/**Partially open a stream from a memory buffer.
   \see op_test_callbacks
   \param      _data  The memory buffer to open.
   \param      _size  The number of bytes in the buffer.
   \param[out] _error Returns 0 on success, or a failure code on error.
                      You may pass in <code>NULL</code> if you don't want the
                       failure code.
                      See op_open_callbacks() for a full list of failure codes.
   \return A partially opened \c OggOpusFile, or <code>NULL</code> on error.*/
OP_WARN_UNUSED_RESULT OggOpusFile *op_test_memory(const unsigned char *_data,
 size_t _size,int *_error);

/**Partially open a stream from a URL.
   This function behaves identically to op_test_url(), except that it
    takes a va_list instead of a variable number of arguments.
   It does not call the <code>va_end</code> macro, and because it invokes the
    <code>va_arg</code> macro, the value of \a _ap is undefined after the call.
   \note If you use this function, you must link against <tt>libopusurl</tt>.
   \see op_test_url
   \see op_test_callbacks
   \param         _url    The URL to open.
                          Currently only the <file:>, <http:>, and <https:>
                           schemes are supported.
                          Both <http:> and <https:> may be disabled at compile
                           time, in which case opening such URLs will always
                           fail.
                          Currently this only supports URIs.
                          IRIs should be converted to UTF-8 and URL-escaped,
                           with internationalized domain names encoded in
                           punycode, before passing them to this function.
   \param[out]    _error  Returns 0 on success, or a failure code on error.
                          You may pass in <code>NULL</code> if you don't want
                           the failure code.
                          See op_open_callbacks() for a full list of failure
                           codes.
   \param[in,out] _ap     A list of the \ref url_options "optional flags" to
                           use.
                          This is a variable-length list of options terminated
                           with <code>NULL</code>.
   \return A partially opened \c OggOpusFile, or <code>NULL</code> on error.*/
OP_WARN_UNUSED_RESULT OggOpusFile *op_vtest_url(const char *_url,
 int *_error,va_list _ap) OP_ARG_NONNULL(1);

/**Partially open a stream from a URL.
   \note If you use this function, you must link against <tt>libopusurl</tt>.
   \see op_test_callbacks
   \param      _url    The URL to open.
                       Currently only the <file:>, <http:>, and <https:>
                        schemes are supported.
                       Both <http:> and <https:> may be disabled at compile
                        time, in which case opening such URLs will always fail.
                       Currently this only supports URIs.
                       IRIs should be converted to UTF-8 and URL-escaped, with
                        internationalized domain names encoded in punycode,
                        before passing them to this function.
   \param[out] _error  Returns 0 on success, or a failure code on error.
                       You may pass in <code>NULL</code> if you don't want the
                        failure code.
                       See op_open_callbacks() for a full list of failure
                        codes.
   \param      ...     The \ref url_options "optional flags" to use.
                       This is a variable-length list of options terminated
                        with <code>NULL</code>.
   \return A partially opened \c OggOpusFile, or <code>NULL</code> on error.*/
OP_WARN_UNUSED_RESULT OggOpusFile *op_test_url(const char *_url,
 int *_error,...) OP_ARG_NONNULL(1);

/**Partially open a stream using the given set of callbacks to access it.
   This tests for Opusness and loads the headers for the first link.
   It does not seek (although it tests for seekability).
   You can query a partially open stream for the few pieces of basic
    information returned by op_serialno(), op_channel_count(), op_head(), and
    op_tags() (but only for the first link).
   You may also determine if it is seekable via a call to op_seekable().
   You cannot read audio from the stream, seek, get the size or duration,
    get information from links other than the first one, or even get the total
    number of links until you finish opening the stream with op_test_open().
   If you do not need to do any of these things, you can dispose of it with
    op_free() instead.

   This function is provided mostly to simplify porting existing code that used
    <tt>libvorbisfile</tt>.
   For new code, you are likely better off using op_test() instead, which
    is less resource-intensive, requires less data to succeed, and imposes a
    hard limit on the amount of data it examines (important for unseekable
    streams, where all such data must be buffered until you are sure of the
    stream type).
   \param _stream        The stream to read from (e.g., a <code>FILE *</code>).
                         This value will be passed verbatim as the first
                          argument to all of the callbacks.
   \param _cb            The callbacks with which to access the stream.
                         <code><a href="#op_read_func">read()</a></code> must
                          be implemented.
                         <code><a href="#op_seek_func">seek()</a></code> and
                          <code><a href="#op_tell_func">tell()</a></code> may
                          be <code>NULL</code>, or may always return -1 to
                          indicate a stream is unseekable, but if
                          <code><a href="#op_seek_func">seek()</a></code> is
                          implemented and succeeds on a particular stream, then
                          <code><a href="#op_tell_func">tell()</a></code> must
                          also.
                         <code><a href="#op_close_func">close()</a></code> may
                          be <code>NULL</code>, but if it is not, it will be
                          called when the \c OggOpusFile is destroyed by
                          op_free().
                         It will not be called if op_open_callbacks() fails
                          with an error.
   \param _initial_data  An initial buffer of data from the start of the
                          stream.
                         Applications can read some number of bytes from the
                          start of the stream to help identify this as an Opus
                          stream, and then provide them here to allow the
                          stream to be tested more thoroughly, even if it is
                          unseekable.
   \param _initial_bytes The number of bytes in \a _initial_data.
                         If the stream is seekable, its current position (as
                          reported by
                          <code><a href="#opus_tell_func">tell()</a></code>
                          at the start of this function) must be equal to
                          \a _initial_bytes.
                         Otherwise, seeking to absolute positions will
                          generate inconsistent results.
   \param[out] _error    Returns 0 on success, or a failure code on error.
                         You may pass in <code>NULL</code> if you don't want
                          the failure code.
                         See op_open_callbacks() for a full list of failure
                          codes.
   \return A partially opened \c OggOpusFile, or <code>NULL</code> on error.
           <tt>libopusfile</tt> does <em>not</em> take ownership of the stream
            if the call fails.
           The calling application is responsible for closing the stream if
            this call returns an error.*/
OP_WARN_UNUSED_RESULT OggOpusFile *op_test_callbacks(void *_stream,
 const OpusFileCallbacks *_cb,const unsigned char *_initial_data,
 size_t _initial_bytes,int *_error) OP_ARG_NONNULL(2);

/**Finish opening a stream partially opened with op_test_callbacks() or one of
    the associated convenience functions.
   If this function fails, you are still responsible for freeing the
    \c OggOpusFile with op_free().
   \param _of The \c OggOpusFile to finish opening.
   \return 0 on success, or a negative value on error.
   \retval #OP_EREAD         An underlying read, seek, or tell operation failed
                              when it should have succeeded.
   \retval #OP_EFAULT        There was a memory allocation failure, or an
                              internal library error.
   \retval #OP_EIMPL         The stream used a feature that is not implemented,
                              such as an unsupported channel family.
   \retval #OP_EINVAL        The stream was not partially opened with
                              op_test_callbacks() or one of the associated
                              convenience functions.
   \retval #OP_ENOTFORMAT    The stream contained a link that did not have any
                              logical Opus streams in it.
   \retval #OP_EBADHEADER    A required header packet was not properly
                              formatted, contained illegal values, or was
                              missing altogether.
   \retval #OP_EVERSION      An ID header contained an unrecognized version
                              number.
   \retval #OP_EBADLINK      We failed to find data we had seen before after
                              seeking.
   \retval #OP_EBADTIMESTAMP The first or last timestamp in a link failed basic
                              validity checks.*/
int op_test_open(OggOpusFile *_of) OP_ARG_NONNULL(1);

/**Release all memory used by an \c OggOpusFile.
   \param _of The \c OggOpusFile to free.*/
void op_free(OggOpusFile *_of);

/*@}*/
/*@}*/

/**\defgroup stream_info Stream Information*/
/*@{*/
/**\name Functions for obtaining information about streams

   These functions allow you to get basic information about a stream, including
    seekability, the number of links (for chained streams), plus the size,
    duration, bitrate, header parameters, and meta information for each link
    (or, where available, the stream as a whole).
   Some of these (size, duration) are only available for seekable streams.
   You can also query the current stream position, link, and playback time,
    and instantaneous bitrate during playback.

   Some of these functions may be used successfully on the partially open
    streams returned by op_test_callbacks() or one of the associated
    convenience functions.
   Their documention will indicate so explicitly.*/
/*@{*/

/**Returns whether or not the stream being read is seekable.
   This is true if
   <ol>
   <li>The <code><a href="#op_seek_func">seek()</a></code> and
    <code><a href="#op_tell_func">tell()</a></code> callbacks are both
    non-<code>NULL</code>,</li>
   <li>The <code><a href="#op_seek_func">seek()</a></code> callback was
    successfully executed at least once, and</li>
   <li>The <code><a href="#op_tell_func">tell()</a></code> callback was
    successfully able to report the position indicator afterwards.</li>
   </ol>
   This function may be called on partially-opened streams.
   \param _of The \c OggOpusFile whose seekable status is to be returned.
   \return A non-zero value if seekable, and 0 if unseekable.*/
int op_seekable(const OggOpusFile *_of) OP_ARG_NONNULL(1);

/**Returns the number of links in this chained stream.
   This function may be called on partially-opened streams, but it will always
    return 1.
   The actual number of links is not known until the stream is fully opened.
   \param _of The \c OggOpusFile from which to retrieve the link count.
   \return For fully-open seekable streams, this returns the total number of
            links in the whole stream, which will be at least 1.
           For partially-open or unseekable streams, this always returns 1.*/
int op_link_count(const OggOpusFile *_of) OP_ARG_NONNULL(1);

/**Get the serial number of the given link in a (possibly-chained) Ogg Opus
    stream.
   This function may be called on partially-opened streams, but it will always
    return the serial number of the Opus stream in the first link.
   \param _of The \c OggOpusFile from which to retrieve the serial number.
   \param _li The index of the link whose serial number should be retrieved.
              Use a negative number to get the serial number of the current
               link.
   \return The serial number of the given link.
           If \a _li is greater than the total number of links, this returns
            the serial number of the last link.
           If the stream is not seekable, this always returns the serial number
            of the current link.*/
opus_uint32 op_serialno(const OggOpusFile *_of,int _li) OP_ARG_NONNULL(1);

/**Get the channel count of the given link in a (possibly-chained) Ogg Opus
    stream.
   This is equivalent to <code>op_head(_of,_li)->channel_count</code>, but
    is provided for convenience.
   This function may be called on partially-opened streams, but it will always
    return the channel count of the Opus stream in the first link.
   \param _of The \c OggOpusFile from which to retrieve the channel count.
   \param _li The index of the link whose channel count should be retrieved.
              Use a negative number to get the channel count of the current
               link.
   \return The channel count of the given link.
           If \a _li is greater than the total number of links, this returns
            the channel count of the last link.
           If the stream is not seekable, this always returns the channel count
            of the current link.*/
int op_channel_count(const OggOpusFile *_of,int _li) OP_ARG_NONNULL(1);

/**Get the total (compressed) size of the stream, or of an individual link in
    a (possibly-chained) Ogg Opus stream, including all headers and Ogg muxing
    overhead.
   \warning If the Opus stream (or link) is concurrently multiplexed with other
    logical streams (e.g., video), this returns the size of the entire stream
    (or link), not just the number of bytes in the first logical Opus stream.
   Returning the latter would require scanning the entire file.
   \param _of The \c OggOpusFile from which to retrieve the compressed size.
   \param _li The index of the link whose compressed size should be computed.
              Use a negative number to get the compressed size of the entire
               stream.
   \return The compressed size of the entire stream if \a _li is negative, the
            compressed size of link \a _li if it is non-negative, or a negative
            value on error.
           The compressed size of the entire stream may be smaller than that
            of the underlying stream if trailing garbage was detected in the
            file.
   \retval #OP_EINVAL The stream is not seekable (so we can't know the length),
                       \a _li wasn't less than the total number of links in
                       the stream, or the stream was only partially open.*/
opus_int64 op_raw_total(const OggOpusFile *_of,int _li) OP_ARG_NONNULL(1);

/**Get the total PCM length (number of samples at 48 kHz) of the stream, or of
    an individual link in a (possibly-chained) Ogg Opus stream.
   Users looking for <code>op_time_total()</code> should use op_pcm_total()
    instead.
   Because timestamps in Opus are fixed at 48 kHz, there is no need for a
    separate function to convert this to seconds (and leaving it out avoids
    introducing floating point to the API, for those that wish to avoid it).
   \param _of The \c OggOpusFile from which to retrieve the PCM offset.
   \param _li The index of the link whose PCM length should be computed.
              Use a negative number to get the PCM length of the entire stream.
   \return The PCM length of the entire stream if \a _li is negative, the PCM
            length of link \a _li if it is non-negative, or a negative value on
            error.
   \retval #OP_EINVAL The stream is not seekable (so we can't know the length),
                       \a _li wasn't less than the total number of links in
                       the stream, or the stream was only partially open.*/
ogg_int64_t op_pcm_total(const OggOpusFile *_of,int _li) OP_ARG_NONNULL(1);

/**Get the ID header information for the given link in a (possibly chained) Ogg
    Opus stream.
   This function may be called on partially-opened streams, but it will always
    return the ID header information of the Opus stream in the first link.
   \param _of The \c OggOpusFile from which to retrieve the ID header
               information.
   \param _li The index of the link whose ID header information should be
               retrieved.
              Use a negative number to get the ID header information of the
               current link.
              For an unseekable stream, \a _li is ignored, and the ID header
               information for the current link is always returned, if
               available.
   \return The contents of the ID header for the given link.*/
const OpusHead *op_head(const OggOpusFile *_of,int _li) OP_ARG_NONNULL(1);

/**Get the comment header information for the given link in a (possibly
    chained) Ogg Opus stream.
   This function may be called on partially-opened streams, but it will always
    return the tags from the Opus stream in the first link.
   \param _of The \c OggOpusFile from which to retrieve the comment header
               information.
   \param _li The index of the link whose comment header information should be
               retrieved.
              Use a negative number to get the comment header information of
               the current link.
              For an unseekable stream, \a _li is ignored, and the comment
               header information for the current link is always returned, if
               available.
   \return The contents of the comment header for the given link, or
            <code>NULL</code> if this is an unseekable stream that encountered
            an invalid link.*/
const OpusTags *op_tags(const OggOpusFile *_of,int _li) OP_ARG_NONNULL(1);

/**Retrieve the index of the current link.
   This is the link that produced the data most recently read by
    op_read_float() or its associated functions, or, after a seek, the link
    that the seek target landed in.
   Reading more data may advance the link index (even on the first read after a
    seek).
   \param _of The \c OggOpusFile from which to retrieve the current link index.
   \return The index of the current link on success, or a negative value on
            failure.
           For seekable streams, this is a number between 0 (inclusive) and the
            value returned by op_link_count() (exclusive).
           For unseekable streams, this value starts at 0 and increments by one
            each time a new link is encountered (even though op_link_count()
            always returns 1).
   \retval #OP_EINVAL The stream was only partially open.*/
int op_current_link(const OggOpusFile *_of) OP_ARG_NONNULL(1);

/**Computes the bitrate of the stream, or of an individual link in a
    (possibly-chained) Ogg Opus stream.
   The stream must be seekable to compute the bitrate.
   For unseekable streams, use op_bitrate_instant() to get periodic estimates.
   \warning If the Opus stream (or link) is concurrently multiplexed with other
    logical streams (e.g., video), this uses the size of the entire stream (or
    link) to compute the bitrate, not just the number of bytes in the first
    logical Opus stream.
   Returning the latter requires scanning the entire file, but this may be done
    by decoding the whole file and calling op_bitrate_instant() once at the
    end.
   Install a trivial decoding callback with op_set_decode_callback() if you
    wish to skip actual decoding during this process.
   \param _of The \c OggOpusFile from which to retrieve the bitrate.
   \param _li The index of the link whose bitrate should be computed.
              Use a negative number to get the bitrate of the whole stream.
   \return The bitrate on success, or a negative value on error.
   \retval #OP_EINVAL The stream was only partially open, the stream was not
                       seekable, or \a _li was larger than the number of
                       links.*/
opus_int32 op_bitrate(const OggOpusFile *_of,int _li) OP_ARG_NONNULL(1);

/**Compute the instantaneous bitrate, measured as the ratio of bits to playable
    samples decoded since a) the last call to op_bitrate_instant(), b) the last
    seek, or c) the start of playback, whichever was most recent.
   This will spike somewhat after a seek or at the start/end of a chain
    boundary, as pre-skip, pre-roll, and end-trimming causes samples to be
    decoded but not played.
   \param _of The \c OggOpusFile from which to retrieve the bitrate.
   \return The bitrate, in bits per second, or a negative value on error.
   \retval #OP_FALSE  No data has been decoded since any of the events
                       described above.
   \retval #OP_EINVAL The stream was only partially open.*/
opus_int32 op_bitrate_instant(OggOpusFile *_of) OP_ARG_NONNULL(1);

/**Obtain the current value of the position indicator for \a _of.
   \param _of The \c OggOpusFile from which to retrieve the position indicator.
   \return The byte position that is currently being read from.
   \retval #OP_EINVAL The stream was only partially open.*/
opus_int64 op_raw_tell(const OggOpusFile *_of) OP_ARG_NONNULL(1);

/**Obtain the PCM offset of the next sample to be read.
   If the stream is not properly timestamped, this might not increment by the
    proper amount between reads, or even return monotonically increasing
    values.
   \param _of The \c OggOpusFile from which to retrieve the PCM offset.
   \return The PCM offset of the next sample to be read.
   \retval #OP_EINVAL The stream was only partially open.*/
ogg_int64_t op_pcm_tell(const OggOpusFile *_of) OP_ARG_NONNULL(1);

/*@}*/
/*@}*/

/**\defgroup stream_seeking Seeking*/
/*@{*/
/**\name Functions for seeking in Opus streams

   These functions let you seek in Opus streams, if the underlying stream
    support it.
   Seeking is implemented for all built-in stream I/O routines, though some
    individual streams may not be seekable (pipes, live HTTP streams, or HTTP
    streams from a server that does not support <code>Range</code> requests).

   op_raw_seek() is the fastest: it is guaranteed to perform at most one
    physical seek, but, since the target is a byte position, makes no guarantee
    how close to a given time it will come.
   op_pcm_seek() provides sample-accurate seeking.
   The number of physical seeks it requires is still quite small (often 1 or
    2, even in highly variable bitrate streams).

   Seeking in Opus requires decoding some pre-roll amount before playback to
    allow the internal state to converge (as if recovering from packet loss).
   This is handled internally by <tt>libopusfile</tt>, but means there is
    little extra overhead for decoding up to the exact position requested
    (since it must decode some amount of audio anyway).
   It also means that decoding after seeking may not return exactly the same
    values as would be obtained by decoding the stream straight through.
   However, such differences are expected to be smaller than the loss
    introduced by Opus's lossy compression.*/
/*@{*/

/**Seek to a byte offset relative to the <b>compressed</b> data.
   This also scans packets to update the PCM cursor.
   It will cross a logical bitstream boundary, but only if it can't get any
    packets out of the tail of the link to which it seeks.
   \param _of          The \c OggOpusFile in which to seek.
   \param _byte_offset The byte position to seek to.
                       This must be between 0 and #op_raw_total(\a _of,\c -1)
                        (inclusive).
   \return 0 on success, or a negative error code on failure.
   \retval #OP_EREAD    The underlying seek operation failed.
   \retval #OP_EINVAL   The stream was only partially open, or the target was
                         outside the valid range for the stream.
   \retval #OP_ENOSEEK  This stream is not seekable.
   \retval #OP_EBADLINK Failed to initialize a decoder for a stream for an
                         unknown reason.*/
int op_raw_seek(OggOpusFile *_of,opus_int64 _byte_offset) OP_ARG_NONNULL(1);

/**Seek to the specified PCM offset, such that decoding will begin at exactly
    the requested position.
   \param _of         The \c OggOpusFile in which to seek.
   \param _pcm_offset The PCM offset to seek to.
                      This is in samples at 48 kHz relative to the start of the
                       stream.
   \return 0 on success, or a negative value on error.
   \retval #OP_EREAD    An underlying read or seek operation failed.
   \retval #OP_EINVAL   The stream was only partially open, or the target was
                         outside the valid range for the stream.
   \retval #OP_ENOSEEK  This stream is not seekable.
   \retval #OP_EBADLINK We failed to find data we had seen before, or the
                         bitstream structure was sufficiently malformed that
                         seeking to the target destination was impossible.*/
int op_pcm_seek(OggOpusFile *_of,ogg_int64_t _pcm_offset) OP_ARG_NONNULL(1);

/*@}*/
/*@}*/

/**\defgroup stream_decoding Decoding*/
/*@{*/
/**\name Functions for decoding audio data

   These functions retrieve actual decoded audio data from the stream.
   The general functions, op_read() and op_read_float() return 16-bit or
    floating-point output, both using native endian ordering.
   The number of channels returned can change from link to link in a chained
    stream.
   There are special functions, op_read_stereo() and op_read_float_stereo(),
    which always output two channels, to simplify applications which do not
    wish to handle multichannel audio.
   These downmix multichannel files to two channels, so they can always return
    samples in the same format for every link in a chained file.

   If the rest of your audio processing chain can handle floating point, the
    floating-point routines should be preferred, as they prevent clipping and
    other issues which might be avoided entirely if, e.g., you scale down the
    volume at some other stage.
   However, if you intend to consume 16-bit samples directly, the conversion in
    <tt>libopusfile</tt> provides noise-shaping dithering and, if compiled
    against <tt>libopus</tt>&nbsp;1.1 or later, soft-clipping prevention.

   <tt>libopusfile</tt> can also be configured at compile time to use the
    fixed-point <tt>libopus</tt> API.
   If so, <tt>libopusfile</tt>'s floating-point API may also be disabled.
   In that configuration, nothing in <tt>libopusfile</tt> will use any
    floating-point operations, to simplify support on devices without an
    adequate FPU.

   \warning HTTPS streams may be be vulnerable to truncation attacks if you do
    not check the error return code from op_read_float() or its associated
    functions.
   If the remote peer does not close the connection gracefully (with a TLS
    "close notify" message), these functions will return #OP_EREAD instead of 0
    when they reach the end of the file.
   If you are reading from an <https:> URL (particularly if seeking is not
    supported), you should make sure to check for this error and warn the user
    appropriately.*/
/*@{*/

/**Indicates that the decoding callback should produce signed 16-bit
    native-endian output samples.*/
#define OP_DEC_FORMAT_SHORT (7008)
/**Indicates that the decoding callback should produce 32-bit native-endian
    float samples.*/
#define OP_DEC_FORMAT_FLOAT (7040)

/**Indicates that the decoding callback did not decode anything, and that
    <tt>libopusfile</tt> should decode normally instead.*/
#define OP_DEC_USE_DEFAULT  (6720)

/**Called to decode an Opus packet.
   This should invoke the functional equivalent of opus_multistream_decode() or
    opus_multistream_decode_float(), except that it returns 0 on success
    instead of the number of decoded samples (which is known a priori).
   \param _ctx       The application-provided callback context.
   \param _decoder   The decoder to use to decode the packet.
   \param[out] _pcm  The buffer to decode into.
                     This will always have enough room for \a _nchannels of
                      \a _nsamples samples, which should be placed into this
                      buffer interleaved.
   \param _op        The packet to decode.
                     This will always have its granule position set to a valid
                      value.
   \param _nsamples  The number of samples expected from the packet.
   \param _nchannels The number of channels expected from the packet.
   \param _format    The desired sample output format.
                     This is either #OP_DEC_FORMAT_SHORT or
                      #OP_DEC_FORMAT_FLOAT.
   \param _li        The index of the link from which this packet was decoded.
   \return A non-negative value on success, or a negative value on error.
           Any error codes should be the same as those returned by
            opus_multistream_decode() or opus_multistream_decode_float().
           Success codes are as follows:
   \retval 0                   Decoding was successful.
                               The application has filled the buffer with
                                exactly <code>\a _nsamples*\a
                                _nchannels</code> samples in the requested
                                format.
   \retval #OP_DEC_USE_DEFAULT No decoding was done.
                               <tt>libopusfile</tt> should do the decoding
                                by itself instead.*/
typedef int (*op_decode_cb_func)(void *_ctx,OpusMSDecoder *_decoder,void *_pcm,
 const ogg_packet *_op,int _nsamples,int _nchannels,int _format,int _li);

/**Sets the packet decode callback function.
   If set, this is called once for each packet that needs to be decoded.
   This can be used by advanced applications to do additional processing on the
    compressed or uncompressed data.
   For example, an application might save the final entropy coder state for
    debugging and testing purposes, or it might apply additional filters
    before the downmixing, dithering, or soft-clipping performed by
    <tt>libopusfile</tt>, so long as these filters do not introduce any
    latency.

   A call to this function is no guarantee that the audio will eventually be
    delivered to the application.
   <tt>libopusfile</tt> may discard some or all of the decoded audio data
    (i.e., at the beginning or end of a link, or after a seek), however the
    callback is still required to provide all of it.
   \param _of        The \c OggOpusFile on which to set the decode callback.
   \param _decode_cb The callback function to call.
                     This may be <code>NULL</code> to disable calling the
                      callback.
   \param _ctx       The application-provided context pointer to pass to the
                      callback on each call.*/
void op_set_decode_callback(OggOpusFile *_of,
 op_decode_cb_func _decode_cb,void *_ctx) OP_ARG_NONNULL(1);

/**Gain offset type that indicates that the provided offset is relative to the
    header gain.
   This is the default.*/
#define OP_HEADER_GAIN   (0)

/**Gain offset type that indicates that the provided offset is relative to the
    R128_ALBUM_GAIN value (if any), in addition to the header gain.*/
#define OP_ALBUM_GAIN    (3007)

/**Gain offset type that indicates that the provided offset is relative to the
    R128_TRACK_GAIN value (if any), in addition to the header gain.*/
#define OP_TRACK_GAIN    (3008)

/**Gain offset type that indicates that the provided offset should be used as
    the gain directly, without applying any the header or track gains.*/
#define OP_ABSOLUTE_GAIN (3009)

/**Sets the gain to be used for decoded output.
   By default, the gain in the header is applied with no additional offset.
   The total gain (including header gain and/or track gain, if applicable, and
    this offset), will be clamped to [-32768,32767]/256 dB.
   This is more than enough to saturate or underflow 16-bit PCM.
   \note The new gain will not be applied to any already buffered, decoded
    output.
   This means you cannot change it sample-by-sample, as at best it will be
    updated packet-by-packet.
   It is meant for setting a target volume level, rather than applying smooth
    fades, etc.
   \param _of             The \c OggOpusFile on which to set the gain offset.
   \param _gain_type      One of #OP_HEADER_GAIN, #OP_ALBUM_GAIN,
                           #OP_TRACK_GAIN, or #OP_ABSOLUTE_GAIN.
   \param _gain_offset_q8 The gain offset to apply, in 1/256ths of a dB.
   \return 0 on success or a negative value on error.
   \retval #OP_EINVAL The \a _gain_type was unrecognized.*/
int op_set_gain_offset(OggOpusFile *_of,
 int _gain_type,opus_int32 _gain_offset_q8) OP_ARG_NONNULL(1);

/**Sets whether or not dithering is enabled for 16-bit decoding.
   By default, when <tt>libopusfile</tt> is compiled to use floating-point
    internally, calling op_read() or op_read_stereo() will first decode to
    float, and then convert to fixed-point using noise-shaping dithering.
   This flag can be used to disable that dithering.
   When the application uses op_read_float() or op_read_float_stereo(), or when
    the library has been compiled to decode directly to fixed point, this flag
    has no effect.
   \param _of      The \c OggOpusFile on which to enable or disable dithering.
   \param _enabled A non-zero value to enable dithering, or 0 to disable it.*/
void op_set_dither_enabled(OggOpusFile *_of,int _enabled) OP_ARG_NONNULL(1);

/**Reads more samples from the stream.
   \note Although \a _buf_size must indicate the total number of values that
    can be stored in \a _pcm, the return value is the number of samples
    <em>per channel</em>.
   This is done because
   <ol>
   <li>The channel count cannot be known a priori (reading more samples might
        advance us into the next link, with a different channel count), so
        \a _buf_size cannot also be in units of samples per channel,</li>
   <li>Returning the samples per channel matches the <code>libopus</code> API
        as closely as we're able,</li>
   <li>Returning the total number of values instead of samples per channel
        would mean the caller would need a division to compute the samples per
        channel, and might worry about the possibility of getting back samples
        for some channels and not others, and</li>
   <li>This approach is relatively fool-proof: if an application passes too
        small a value to \a _buf_size, they will simply get fewer samples back,
        and if they assume the return value is the total number of values, then
        they will simply read too few (rather than reading too many and going
        off the end of the buffer).</li>
   </ol>
   \param      _of       The \c OggOpusFile from which to read.
   \param[out] _pcm      A buffer in which to store the output PCM samples, as
                          signed native-endian 16-bit values at 48&nbsp;kHz
                          with a nominal range of <code>[-32768,32767)</code>.
                         Multiple channels are interleaved using the
                          <a href="https://www.xiph.org/vorbis/doc/Vorbis_I_spec.html#x1-810004.3.9">Vorbis
                          channel ordering</a>.
                         This must have room for at least \a _buf_size values.
   \param      _buf_size The number of values that can be stored in \a _pcm.
                         It is recommended that this be large enough for at
                          least 120 ms of data at 48 kHz per channel (5760
                          values per channel).
                         Smaller buffers will simply return less data, possibly
                          consuming more memory to buffer the data internally.
                         <tt>libopusfile</tt> may return less data than
                          requested.
                         If so, there is no guarantee that the remaining data
                          in \a _pcm will be unmodified.
   \param[out] _li       The index of the link this data was decoded from.
                         You may pass <code>NULL</code> if you do not need this
                          information.
                         If this function fails (returning a negative value),
                          this parameter is left unset.
   \return The number of samples read per channel on success, or a negative
            value on failure.
           The channel count can be retrieved on success by calling
            <code>op_head(_of,*_li)</code>.
           The number of samples returned may be 0 if the buffer was too small
            to store even a single sample for all channels, or if end-of-file
            was reached.
           The list of possible failure codes follows.
           Most of them can only be returned by unseekable, chained streams
            that encounter a new link.
   \retval #OP_HOLE          There was a hole in the data, and some samples
                              may have been skipped.
                             Call this function again to continue decoding
                              past the hole.
   \retval #OP_EREAD         An underlying read operation failed.
                             This may signal a truncation attack from an
                              <https:> source.
   \retval #OP_EFAULT        An internal memory allocation failed.
   \retval #OP_EIMPL         An unseekable stream encountered a new link that
                              used a feature that is not implemented, such as
                              an unsupported channel family.
   \retval #OP_EINVAL        The stream was only partially open.
   \retval #OP_ENOTFORMAT    An unseekable stream encountered a new link that
                              did not have any logical Opus streams in it.
   \retval #OP_EBADHEADER    An unseekable stream encountered a new link with a
                              required header packet that was not properly
                              formatted, contained illegal values, or was
                              missing altogether.
   \retval #OP_EVERSION      An unseekable stream encountered a new link with
                              an ID header that contained an unrecognized
                              version number.
   \retval #OP_EBADPACKET    Failed to properly decode the next packet.
   \retval #OP_EBADLINK      We failed to find data we had seen before.
   \retval #OP_EBADTIMESTAMP An unseekable stream encountered a new link with
                              a starting timestamp that failed basic validity
                              checks.*/
OP_WARN_UNUSED_RESULT int op_read(OggOpusFile *_of,
 opus_int16 *_pcm,int _buf_size,int *_li) OP_ARG_NONNULL(1);

/**Reads more samples from the stream.
   \note Although \a _buf_size must indicate the total number of values that
    can be stored in \a _pcm, the return value is the number of samples
    <em>per channel</em>.
   <ol>
   <li>The channel count cannot be known a priori (reading more samples might
        advance us into the next link, with a different channel count), so
        \a _buf_size cannot also be in units of samples per channel,</li>
   <li>Returning the samples per channel matches the <code>libopus</code> API
        as closely as we're able,</li>
   <li>Returning the total number of values instead of samples per channel
        would mean the caller would need a division to compute the samples per
        channel, and might worry about the possibility of getting back samples
        for some channels and not others, and</li>
   <li>This approach is relatively fool-proof: if an application passes too
        small a value to \a _buf_size, they will simply get fewer samples back,
        and if they assume the return value is the total number of values, then
        they will simply read too few (rather than reading too many and going
        off the end of the buffer).</li>
   </ol>
   \param      _of       The \c OggOpusFile from which to read.
   \param[out] _pcm      A buffer in which to store the output PCM samples as
                          signed floats at 48&nbsp;kHz with a nominal range of
                          <code>[-1.0,1.0]</code>.
                         Multiple channels are interleaved using the
                          <a href="https://www.xiph.org/vorbis/doc/Vorbis_I_spec.html#x1-810004.3.9">Vorbis
                          channel ordering</a>.
                         This must have room for at least \a _buf_size floats.
   \param      _buf_size The number of floats that can be stored in \a _pcm.
                         It is recommended that this be large enough for at
                          least 120 ms of data at 48 kHz per channel (5760
                          samples per channel).
                         Smaller buffers will simply return less data, possibly
                          consuming more memory to buffer the data internally.
                         If less than \a _buf_size values are returned,
                          <tt>libopusfile</tt> makes no guarantee that the
                          remaining data in \a _pcm will be unmodified.
   \param[out] _li       The index of the link this data was decoded from.
                         You may pass <code>NULL</code> if you do not need this
                          information.
                         If this function fails (returning a negative value),
                          this parameter is left unset.
   \return The number of samples read per channel on success, or a negative
            value on failure.
           The channel count can be retrieved on success by calling
            <code>op_head(_of,*_li)</code>.
           The number of samples returned may be 0 if the buffer was too small
            to store even a single sample for all channels, or if end-of-file
            was reached.
           The list of possible failure codes follows.
           Most of them can only be returned by unseekable, chained streams
            that encounter a new link.
   \retval #OP_HOLE          There was a hole in the data, and some samples
                              may have been skipped.
                             Call this function again to continue decoding
                              past the hole.
   \retval #OP_EREAD         An underlying read operation failed.
                             This may signal a truncation attack from an
                              <https:> source.
   \retval #OP_EFAULT        An internal memory allocation failed.
   \retval #OP_EIMPL         An unseekable stream encountered a new link that
                              used a feature that is not implemented, such as
                              an unsupported channel family.
   \retval #OP_EINVAL        The stream was only partially open.
   \retval #OP_ENOTFORMAT    An unseekable stream encountered a new link that
                              did not have any logical Opus streams in it.
   \retval #OP_EBADHEADER    An unseekable stream encountered a new link with a
                              required header packet that was not properly
                              formatted, contained illegal values, or was
                              missing altogether.
   \retval #OP_EVERSION      An unseekable stream encountered a new link with
                              an ID header that contained an unrecognized
                              version number.
   \retval #OP_EBADPACKET    Failed to properly decode the next packet.
   \retval #OP_EBADLINK      We failed to find data we had seen before.
   \retval #OP_EBADTIMESTAMP An unseekable stream encountered a new link with
                              a starting timestamp that failed basic validity
                              checks.*/
OP_WARN_UNUSED_RESULT int op_read_float(OggOpusFile *_of,
 float *_pcm,int _buf_size,int *_li) OP_ARG_NONNULL(1);

/**Reads more samples from the stream and downmixes to stereo, if necessary.
   This function is intended for simple players that want a uniform output
    format, even if the channel count changes between links in a chained
    stream.
   \note \a _buf_size indicates the total number of values that can be stored
    in \a _pcm, while the return value is the number of samples <em>per
    channel</em>, even though the channel count is known, for consistency with
    op_read().
   \param      _of       The \c OggOpusFile from which to read.
   \param[out] _pcm      A buffer in which to store the output PCM samples, as
                          signed native-endian 16-bit values at 48&nbsp;kHz
                          with a nominal range of <code>[-32768,32767)</code>.
                         The left and right channels are interleaved in the
                          buffer.
                         This must have room for at least \a _buf_size values.
   \param      _buf_size The number of values that can be stored in \a _pcm.
                         It is recommended that this be large enough for at
                          least 120 ms of data at 48 kHz per channel (11520
                          values total).
                         Smaller buffers will simply return less data, possibly
                          consuming more memory to buffer the data internally.
                         If less than \a _buf_size values are returned,
                          <tt>libopusfile</tt> makes no guarantee that the
                          remaining data in \a _pcm will be unmodified.
   \return The number of samples read per channel on success, or a negative
            value on failure.
           The number of samples returned may be 0 if the buffer was too small
            to store even a single sample for both channels, or if end-of-file
            was reached.
           The list of possible failure codes follows.
           Most of them can only be returned by unseekable, chained streams
            that encounter a new link.
   \retval #OP_HOLE          There was a hole in the data, and some samples
                              may have been skipped.
                             Call this function again to continue decoding
                              past the hole.
   \retval #OP_EREAD         An underlying read operation failed.
                             This may signal a truncation attack from an
                              <https:> source.
   \retval #OP_EFAULT        An internal memory allocation failed.
   \retval #OP_EIMPL         An unseekable stream encountered a new link that
                              used a feature that is not implemented, such as
                              an unsupported channel family.
   \retval #OP_EINVAL        The stream was only partially open.
   \retval #OP_ENOTFORMAT    An unseekable stream encountered a new link that
                              did not have any logical Opus streams in it.
   \retval #OP_EBADHEADER    An unseekable stream encountered a new link with a
                              required header packet that was not properly
                              formatted, contained illegal values, or was
                              missing altogether.
   \retval #OP_EVERSION      An unseekable stream encountered a new link with
                              an ID header that contained an unrecognized
                              version number.
   \retval #OP_EBADPACKET    Failed to properly decode the next packet.
   \retval #OP_EBADLINK      We failed to find data we had seen before.
   \retval #OP_EBADTIMESTAMP An unseekable stream encountered a new link with
                              a starting timestamp that failed basic validity
                              checks.*/
OP_WARN_UNUSED_RESULT int op_read_stereo(OggOpusFile *_of,
 opus_int16 *_pcm,int _buf_size) OP_ARG_NONNULL(1);

/**Reads more samples from the stream and downmixes to stereo, if necessary.
   This function is intended for simple players that want a uniform output
    format, even if the channel count changes between links in a chained
    stream.
   \note \a _buf_size indicates the total number of values that can be stored
    in \a _pcm, while the return value is the number of samples <em>per
    channel</em>, even though the channel count is known, for consistency with
    op_read_float().
   \param      _of       The \c OggOpusFile from which to read.
   \param[out] _pcm      A buffer in which to store the output PCM samples, as
                          signed floats at 48&nbsp;kHz with a nominal range of
                          <code>[-1.0,1.0]</code>.
                         The left and right channels are interleaved in the
                          buffer.
                         This must have room for at least \a _buf_size values.
   \param      _buf_size The number of values that can be stored in \a _pcm.
                         It is recommended that this be large enough for at
                          least 120 ms of data at 48 kHz per channel (11520
                          values total).
                         Smaller buffers will simply return less data, possibly
                          consuming more memory to buffer the data internally.
                         If less than \a _buf_size values are returned,
                          <tt>libopusfile</tt> makes no guarantee that the
                          remaining data in \a _pcm will be unmodified.
   \return The number of samples read per channel on success, or a negative
            value on failure.
           The number of samples returned may be 0 if the buffer was too small
            to store even a single sample for both channels, or if end-of-file
            was reached.
           The list of possible failure codes follows.
           Most of them can only be returned by unseekable, chained streams
            that encounter a new link.
   \retval #OP_HOLE          There was a hole in the data, and some samples
                              may have been skipped.
                             Call this function again to continue decoding
                              past the hole.
   \retval #OP_EREAD         An underlying read operation failed.
                             This may signal a truncation attack from an
                              <https:> source.
   \retval #OP_EFAULT        An internal memory allocation failed.
   \retval #OP_EIMPL         An unseekable stream encountered a new link that
                              used a feature that is not implemented, such as
                              an unsupported channel family.
   \retval #OP_EINVAL        The stream was only partially open.
   \retval #OP_ENOTFORMAT    An unseekable stream encountered a new link that
                              that did not have any logical Opus streams in it.
   \retval #OP_EBADHEADER    An unseekable stream encountered a new link with a
                              required header packet that was not properly
                              formatted, contained illegal values, or was
                              missing altogether.
   \retval #OP_EVERSION      An unseekable stream encountered a new link with
                              an ID header that contained an unrecognized
                              version number.
   \retval #OP_EBADPACKET    Failed to properly decode the next packet.
   \retval #OP_EBADLINK      We failed to find data we had seen before.
   \retval #OP_EBADTIMESTAMP An unseekable stream encountered a new link with
                              a starting timestamp that failed basic validity
                              checks.*/
OP_WARN_UNUSED_RESULT int op_read_float_stereo(OggOpusFile *_of,
 float *_pcm,int _buf_size) OP_ARG_NONNULL(1);

/*@}*/
/*@}*/

# if OP_GNUC_PREREQ(4,0)
#  pragma GCC visibility pop
# endif

# if defined(__cplusplus)
}
# endif

#endif
