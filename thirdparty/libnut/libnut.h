// (C) 2005-2006 Oded Shimon
// This file is available under the MIT/X license, see COPYING

#ifndef LIBNUT_NUT_H
#define LIBNUT_NUT_H

#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>

/// \defgroup common  common defines and enums
/// \defgroup muxer   libnut muxer
/// \defgroup demuxer libnut demuxer

/// \addtogroup common
/// @{
#define NUT_VERSION 2 ///< Version of NUT specification this library implements.

#define NUT_MAX_STREAMS 16

/// Stream class values, only one can be set. Higher values are not legal.
enum nut_stream_class_tt {
	NUT_VIDEO_CLASS    = 0, ///< = 0
	NUT_AUDIO_CLASS    = 1, ///< = 1
	NUT_SUBTITLE_CLASS = 2, ///< = 2
	NUT_USERDATA_CLASS = 3, ///< = 3
};

/// frame flags bitfield (several flags may be set at once)
enum nut_frame_flags_tt {
	NUT_FLAG_KEY = 1,  ///< Marks a frame as keyframe.
	NUT_FLAG_EOR = 2,  ///< Marks end of relevance for stream. #NUT_FLAG_KEY \b must be set together with this flag.
};
/// @}

typedef struct nut_context_s nut_context_tt;

/// memory allocation function pointers \ingroup demuxer muxer
typedef struct {
	void * (*malloc)(size_t size);             ///< memory allocation malloc function pointer
	void * (*realloc)(void *ptr, size_t size); ///< memory allocation realloc function pointer
	void (*free)(void *ptr);                   ///< memory allocation free function pointer
} nut_alloc_tt;

/// timebase struct \ingroup demuxer muxer
typedef struct {
	int num; ///< Example: 1001
	int den; ///< Example: 24000
} nut_timebase_tt;

/// stream header struct \ingroup demuxer muxer
typedef struct {
	int type;                 ///< Possible values are enum ::nut_stream_class_t. A value of -1 terminates a stream header array.
	int fourcc_len;           ///< fourcc length
	uint8_t * fourcc;         ///< fourcc in big-endian format
	nut_timebase_tt time_base; ///< stream timebase
	int fixed_fps;            ///< denote stream as fixed or variable fps
	int decode_delay;         ///< decode delay of codec in this stream
	int codec_specific_len;   ///< length of codec-specific data
	uint8_t * codec_specific; ///< Codec specific data, may be NULL if #codec_specific_len is zero.
	uint64_t max_pts;         ///< Only used in demuxer. If non-zero, then it is the highest value in the stream.

	/// \name Video
	/// Only used if type is #NUT_VIDEO_CLASS @{
	int width;                ///< width of video in pixels
	int height;               ///< height of video in pixels
	int sample_width;         ///< Ratio to stretch the video, may only be zero if #sample_height is zero.
	int sample_height;        ///< Ratio to stretch the video, may only be zero if #sample_width is zero.
	int colorspace_type;

	/// \name Audio
	/// Only used if type is #NUT_AUDIO_CLASS @{
	int samplerate_num;       ///< audio sample rate, example: 44100
	int samplerate_denom;     ///< audio sample rate denominator, example: 1
	int channel_count;        ///< number of audio channels
} nut_stream_header_tt;

/// single info field struct \ingroup demuxer muxer
typedef struct {
	char type[7];      ///< NULL-terminated string
	char name[65];     ///< NULL-terminated string, name of info field
	int64_t val;       ///< meaning of value defined by #type
	int den;           ///< Used if #type is "r", #val is the numerator.
	nut_timebase_tt tb; ///< Used if #type is "t".
	uint8_t * data;    ///< Used if #type is non-numeric.
} nut_info_field_tt;

/// single info packet struct \ingroup demuxer muxer
typedef struct {
	int count;                 ///< number of info fields in #fields.
	int stream_id_plus1;       ///< Zero indicates non-stream-specific info packet.
	int chapter_id;            ///< subsection of the file this info packet applies to
	nut_timebase_tt chapter_tb; ///< timebase of #chapter_start and #chapter_len
	uint64_t chapter_start;    ///< start of chapter or complete file
	uint64_t chapter_len;      ///< length of chapter or complete file
	nut_info_field_tt * fields; ///< Info fields, has #count elements.
} nut_info_packet_tt;

/// single frame packet struct \ingroup demuxer muxer
typedef struct {
	int len;          ///< Length of frame in bytes, \b must be zero if #NUT_FLAG_EOR is set.
	int stream;       ///< stream index of frame
	uint64_t pts;     ///< presentation timestamp of frame
	int flags;        ///< frame flags from #nut_frame_flags_t
	int64_t next_pts; ///< Only used in muxer. Only necessary if nut_write_frame_reorder() is used.
} nut_packet_tt;



/*****************************************
 * muxer                                 *
 *****************************************/

/// \addtogroup muxer
/// @{

/// output stream struct
typedef struct {
	void * priv;                                                ///< opaque priv pointer to be passed to function calls
	int (*write)(void * priv, size_t len, const uint8_t * buf); ///< If NULL, nut_output_stream_tt::priv is used as FILE*.
} nut_output_stream_tt;

/// NUT framecode table input
typedef struct {
	int flag;   ///< framecode entry flags
	int pts;    ///< pts delta from previous frame
	int stream; ///< stream_id of frame
	int mul;    ///< multiplier for coded frame size
	int size;   ///< LSB for coded frame size
	int count;  ///< Explicit count of framecode entry, \b should be (mul-size) in almost all cases.
} nut_frame_table_input_tt;

/// muxer options struct
typedef struct {
	nut_output_stream_tt output;    ///< output stream function pointers
	nut_alloc_tt alloc;             ///< memory allocation function pointers
	int write_index;               ///< whether or not to write an index
	int realtime_stream;           ///< Implies no write_index.
	int max_distance;              ///< Valid values are 32-65536, the recommended value is 32768. Lower values give better seekability and error detection and recovery but cause higher overhead.
	nut_frame_table_input_tt * fti; ///< Framecode table, may be NULL.
} nut_muxer_opts_tt;

/// Allocates NUT muxer context and writes headers to file.
nut_context_tt * nut_muxer_init(const nut_muxer_opts_tt * mopts, const nut_stream_header_tt s[], const nut_info_packet_tt info[]);

/// Deallocates NUT muxer context.
void nut_muxer_uninit(nut_context_tt * nut);

/// Writes a single frame to a NUT file.
void nut_write_frame(nut_context_tt * nut, const nut_packet_tt * p, const uint8_t * buf);

/// Writes a single info packet to a NUT file.
void nut_write_info(nut_context_tt * nut, const nut_info_packet_tt * info);

/// Buffers and sorts a single frame to be written to a NUT file.
void nut_write_frame_reorder(nut_context_tt * nut, const nut_packet_tt * p, const uint8_t * buf);

/// Flushes reorder buffer and deallocates NUT muxer context.
void nut_muxer_uninit_reorder(nut_context_tt * nut);

/// Creates an optimized framecode table for the NUT main header based on stream info.
void nut_framecode_generate(const nut_stream_header_tt s[], nut_frame_table_input_tt fti[256]);
/// @}



/*****************************************
 * demuxer                               *
 *****************************************/

/// \addtogroup demuxer
/// @{

/// input stream struct
typedef struct {
	void * priv;                                            ///< opaque priv pointer to be passed to function calls
	size_t (*read)(void * priv, size_t len, uint8_t * buf); ///< Input stream read function, must return amount of bytes actually read.
	off_t (*seek)(void * priv, long long pos, int whence);  ///< Input stream seek function, must return position in file after seek.
	int (*eof)(void * priv);                                ///< Returns if EOF has been met in stream in case of read error.
	off_t file_pos;                                         ///< file position at beginning of read
} nut_input_stream_tt;

/// demuxer options struct
typedef struct {
	nut_input_stream_tt input;  ///< input stream function pointers
	nut_alloc_tt alloc;         ///< memory allocation function pointers
	int read_index;            ///< Seeks to end-of-file at beginning of playback to search for index. Implies cache_syncpoints.
	int cache_syncpoints;      ///< Improves seekability and error recovery greatly, but costs some memory (0.5MB for very large files).
	void * info_priv;          ///< opaque priv pointer to be passed to #new_info
	void (*new_info)(void * priv, nut_info_packet_tt * info); ///< Function to be called when info is found mid-stream. May be NULL.
} nut_demuxer_opts_tt;

/// Possible errors given from demuxer functions. Only the first 4 errors should ever be returned, the rest are internal.
enum nut_errors {
	NUT_ERR_NO_ERROR      = 0,    ///< = 0
	NUT_ERR_EOF           = 1,    ///< = 1
	NUT_ERR_EAGAIN        = 2,    ///< = 2
	NUT_ERR_OUT_OF_MEM    = 3,    ///< = 3
	NUT_ERR_NOT_SEEKABLE,         ///< Can only be returned by nut_seek(). Indicates that the seek was unsuccessful.
	NUT_ERR_GENERAL_ERROR,
	NUT_ERR_BAD_VERSION,
	NUT_ERR_NOT_FRAME_NOT_N,
	NUT_ERR_BAD_CHECKSUM,
	NUT_ERR_MAX_SYNCPOINT_DISTANCE,
	NUT_ERR_MAX_DISTANCE,
	NUT_ERR_NO_HEADERS,
	NUT_ERR_OUT_OF_ORDER,
	NUT_ERR_MAX_PTS_DISTANCE,
	NUT_ERR_VLC_TOO_LONG,
	NUT_ERR_BAD_STREAM_ORDER,
	NUT_ERR_NOSTREAM_STARTCODE,
	NUT_ERR_BAD_EOF,
};

/// Creates a NUT demuxer context. Does not read any information from file.
nut_context_tt * nut_demuxer_init(nut_demuxer_opts_tt * dopts);

/// Frees a NUT demuxer context. No other functions can be called after this.
void nut_demuxer_uninit(nut_context_tt * nut);

/// Reads headers and index, \b must be called at init.
int nut_read_headers(nut_context_tt * nut, nut_stream_header_tt * s [], nut_info_packet_tt * info []);

/// Gets frame header, must be called before each packet.
int nut_read_next_packet(nut_context_tt * nut, nut_packet_tt * pd);

/// Reads just the frame \b data, not the header.
int nut_read_frame(nut_context_tt * nut, int * len, uint8_t * buf);

/// Gives human readable description of the error return code of any demuxing function.
const char * nut_error(int error);

/// Seeks to the requested position in seconds.
int nut_seek(nut_context_tt * nut, double time_pos, int flags, const int * active_streams);
/// @}



/*****************************************
 * Extended Doxygen Documentation        *
 *****************************************/

/*! \mainpage libnut Documentation
 * \author Oded Shimon <ods15@ods15.dyndns.org>
 * \date 2005-2006
 *
 * Reference implementation for the NUT open container format.
 *
 * libnut source code is at svn://svn.mplayerhq.hu/nut/src/libnut/
 *
 * This library is covered by the MIT/X license. For more details, see
 * \ref License.
 *
 * For more information on the format, please visit http://www.nut-container.org/
 */

/*! \page License
 * \verbinclude COPYING
 */

/*! \struct nut_alloc_t
 * libc semantics are assumed for all functions (realloc must work with NULL or zero size).
 *
 * #malloc function pointer may be NULL. This indicates using libc malloc, realloc and free functions.
 *
 * If #malloc is not NULL, #realloc and #free \b must \b not be NULL.
 */

/*! \struct nut_timebase_t
 * The example shown is if fps is 23.976 (24000/1001). Timebase is the opposite of fps.
 */

/*! \struct nut_info_field_t
 * \par Example:
 * \code
 * char * text = "The Foobar Adventure";
 * nut_info_field_tt title = {
 * 	.type = "UTF-8",
 * 	.name = "Title",
 * 	.val = strlen(text),
 * 	.data = text,
 * };
 *
 * nut_info_field_tt rational = {
 * 	.type = "r",
 * 	.name = "X-Value close to Pi",
 * 	.val = 355,
 * 	.den = 113,
 * };
 * \endcode
 */

/*! \var char nut_info_field_tt::type[7]
 * Type of field value
 * - "v" - integer value
 * - "s" - signed integer value
 * - "r" - fraction rational
 * - "t" - timestamp
 * - "UTF-8" - UTF-8 coded string
 * - Other - binary data
 */

/*! \var int64_t nut_info_field_tt::val
 * - "v" - integer value
 * - "s" - integer value
 * - "r" - numerator of fraction
 * - "t" - integer timestamp in timebase #tb
 * - Other - length of #data in bytes
 *
 * In the case an of UTF-8 string, length of #data \b must \b not contain the
 * terminating NUL (U+0000).
 */

/*! \var nut_timebase_tt nut_info_field_tt::tb
 * In the muxer case, values of #tb \b must be identical to the
 * timebase of one of the streams.
 */

/*! \var uint8_t * nut_info_field_tt::data
 * Even in the case of an UTF-8 string, this data is \b not NULL terminated.
 *
 * For the muxer, this value \b must be NULL if info field carries no binary
 * data.
 */

/*! \var int nut_info_packet_tt::count
 * For arrays of #nut_info_packet_t, the packet with a #count of \a -1
 * terminates the array.
 */

/*! \var nut_info_packet_tt::chapter_id
 * A value of 0 indicates that the info packet applies to the complete file.
 *
 * Positive values are real chapters. Real chapters must not overlap. The
 * #chapter_id of a real chapter must not be higher than the total amount
 * of real chapters in the file.
 *
 * Negative values indicate a file subsection and may overlap.
 *
 * If #chapter_id is 0, #chapter_start and #chapter_len provide the length
 * of the entire file.
 */

/*! \var nut_info_packet_tt::chapter_tb
 * When muxing, values #chapter_tb \b must be identical to the timebase of
 * one of the streams.
 */

/*!
 * \var int nut_frame_table_input_tt::flag
 *
 * This variable is a bitfield. Valid flags are:
 * -    1  FLAG_KEYFRAME
 * -    2  FLAG_EOR
 * -    8  FLAG_CODED_PTS
 * -   16  FLAG_CODED_STREAM_ID
 * -   32  FLAG_SIZE_MSB
 * -   64  FLAG_CHECKSUM
 * -  128  FLAG_RESERVED
 * - 4096  FLAG_CODED
 * - 8192  FLAG_INVALID
 *
 * The last entry of nut_frame_table_input_tt \b must have flag == -1.
 */

/*! \fn nut_context_tt * nut_muxer_init(const nut_muxer_opts_tt * mopts, const nut_stream_header_tt s[], const nut_info_packet_tt info[])
 * \param mopts muxer options
 * \param s     Stream header data, terminated by \a type = -1.
 * \param info  Info packet data, terminated by \a count = -1. May be \a NULL.
 * \return NUT muxer context
 *
 * In case nut_muxer_opts_tt::realtime_stream is set, the first packet passed
 * to the nut_output_stream_tt::write() function will be all of the main
 * headers. This packet must be passed along exactly once with the
 * beginning of any stream forwarded out.
 */

/*! \fn void nut_muxer_uninit(nut_context_tt *nut)
 * \param nut NUT muxer context
 *
 * Optionaly writes index if nut_muxer_opts_tt::write_index is set and
 * nut_muxer_opts_tt::realtime_stream is unset.
 *
 * \warning Must \b not be used directly if nut_write_frame_reorder() was used.
 * \see nut_muxer_uninit_reorder()
 */

/*! \fn void nut_write_frame(nut_context_tt * nut, const nut_packet_tt * p, const uint8_t * buf)
 * \param nut NUT muxer context
 * \param p   information about the frame
 * \param buf actual frame data
 *
 * If nut_muxer_opts_tt::realtime_stream realtime_stream is unset, repeated
 * headers will be written at some positions. Syncpoints will be written in
 * accordance to the NUT spec. If nut_muxer_opts_tt::realtime_stream is set,
 * calling this function will result in a single nut_output_stream_tt::write()
 * call, which will be the full frame NUT packet. If the packet starts with
 * a syncpoint startcode, it may be used as a start point after giving the
 * main headers to a new client.
 *
 * \warning
 * The use of this function is discouraged if more than a single stream is
 * used, as frames must meet the NUT specification ordering rule.
 * nut_write_frame_reorder() should be used instead.
 * \sa nut_write_frame_reorder()
 */

/*! \fn void nut_write_info(nut_context_tt * nut, const nut_info_packet_tt * info)
 * \param nut NUT muxer context
 * \param info a single info packet
 *
 * The use of this function is \b illegal in non-realtime streams, and will
 * do nothing if nut_muxer_opts_tt::realtime_stream is not set. The result
 * is a single call to nut_output_stream_tt::write() with the NUT info packet.
 */

/*! \fn void nut_write_frame_reorder(nut_context_tt * nut, const nut_packet_tt * p, const uint8_t * buf)
 * \param nut NUT muxer context
 * \param p   information about the frame
 * \param buf actual frame data
 *
 * Uses an internal buffer and sorts the frames to meet NUT's ordering rule.
 * Calls to this function \b must \b not be mixed with calls to
 * nut_write_frame().
 *
 * If this function is used, nut_muxer_uninit_reorder() \b must be used.
 * \sa nut_muxer_uninit_reorder()
 */

/*! \fn void nut_muxer_uninit_reorder(nut_context_tt * nut)
 * \param nut NUT muxer context
 *
 * Must be used if nut_write_frame_reorder() was used.
 * \sa nut_muxer_uninit()
 */

/*! \fn void nut_framecode_generate(const nut_stream_header_tt s[], nut_frame_table_input_tt fti[256])
 * \param s   Stream header data, terminated by \a type = -1.
 * \param fti Output framecode table data, must be preallocated to 256 entries.
 *
 * Creates an optimized framecode table for the NUT main header based on stream
 * types and codecs. Currently recognized fourcc values are "mp4v", "h264",
 * "mp3 ", and "vrbs".
 *
 * This function is used directly by nut_muxer_init() if
 * nut_muxer_opts_tt::fti is \a NULL.
 */

/*! \addtogroup demuxer
 * All of the demuxer related functions return an integer value
 * representing one of the following return codes:
 * - 0 (=#NUT_ERR_NO_ERROR) success
 * - #NUT_ERR_EOF           unexpected or expected EOF
 * - #NUT_ERR_EAGAIN        insufficient data for demuxing
 * - #NUT_ERR_NOT_SEEKABLE  unsuccessful seek in nut_seek()
 *
 * \par NUT_ERR_EOF
 * If any function returns EOF, it can be recovered by using nut_seek().
 * The exception to this is nut_read_headers(), then it is a fatal
 * error and only nut_demuxer_uninit() can be called.
 *
 * \par NUT_ERR_EAGAIN
 * Indicates that not enough data was returned by nut_input_stream_tt::read(),
 * but EOF was not reached. Whichever function returned this should be called
 * again with the exact same parameters when more data is available.
 * The exception to this is nut_read_frame(), which should be called with
 * an updated \a buf parameter.
 *
 * \par NUT_ERR_NOT_SEEKABLE
 * Can only be returned from nut_seek(). Indicates that no seek has been
 * performed and that the stream is in the exact same position it was in
 * before the seek.
 *
 * \par other errors
 * All errors except the list above are fatal.
 * The only legal operation after a fatal error is nut_demuxer_uninit().
 */

/*! \var size_t (*nut_input_stream_tt::read)(void * priv, size_t len, uint8_t * buf)
 * If NULL, nut_input_stream_tt::priv is used as FILE*, and
 * nut_input_stream_tt::seek and nut_input_stream_tt::eof are ignored.
 */

/*! \var off_t (*nut_input_stream_tt::seek)(void * priv, long long pos, int whence)
 * If NULL (and nut_input_stream_tt::read is non-NULL), indicates stream is
 * unseekable. This implies nut_demuxer_opts_tt::read_index and
 * nut_demuxer_opts_tt::cache_syncpoints to be unset. Any call to
 * nut_seek() with an unseekable stream will yield #NUT_ERR_NOT_SEEKABLE
 *
 * Parameter whence must support the following libc defines:
 * - SEEK_SET - pos is used as offset from beginning of file
 * - SEEK_CUR - pos is used as offset from current position
 * - SEEK_END - pos is used as offset from end of file
 */

/*! \var int (*nut_input_stream_tt::eof)(void * priv)
 * Only necessary if stream supports non-blocking mode.
 * Returns non-zero if stream is at EOF, 0 otherwise.
 *
 * This function is called if nut_input_stream_tt::read returns less data
 * read than requested. If it returns zero, then the stream is assumed to
 * be lacking data and #NUT_ERR_EAGAIN is raised. Otherwise, #NUT_ERR_EOF
 * is raised.
 *
 * If NULL, then any read error is assumed to be caused by EOF.
 */

/*! \var off_t nut_input_stream_tt::file_pos
 * Must contain the position in the file at which demuxing begins. Should
 * usually be zero.
 */

/*! \fn int nut_read_headers(nut_context_tt * nut, nut_stream_header_tt * s [], nut_info_packet_tt * info [])
 * \param nut  NUT demuxer context
 * \param s    Pointer to stream header variable to be set to an array
 * \param info Pointer to info header variable, may be NULL.
 *
 * Both `s' and `info' are handled by context and are illegal pointers
 * after nut_demuxer_uninit().
 *
 * If main NUT headers are not found at beginning of file and the input
 * stream is seekable, several parts of the file are searched for the main
 * headers before giving up in accordance to NUT spec (EOF and 2^n
 * locations).
 *
 * Any error returned from this function except #NUT_ERR_EAGAIN is fatal.
 * No other functions may be called before a call to
 * nut_read_headers() succeeds.
 */

/*! \fn int nut_read_next_packet(nut_context_tt * nut, nut_packet_tt * pd)
 * \param nut NUT demuxer context
 * \param pd  Pointer to frame header struct to be filled with next frame
 *            information.
 *
 * After a successful call to nut_read_next_packet(), nut_read_frame()
 * \b must be called.
 *
 * If nut_demuxer_opts_tt::new_info is non-NULL, a new info packet may be
 * seen before decoding the frame header and this function pointer will be
 * called (possibly even several times).
 *
 * If a stream error is detected during frame header decoding,
 * nut_read_next_packet() will attempt to recover from the error and return
 * the next undamaged frame in the stream.
 */

/*! \fn int nut_read_frame(nut_context_tt * nut, int * len, uint8_t * buf)
 * \param nut NUT demuxer context
 * \param len length of data left to be read
 * \param buf buffer to write the frame data
 *
 * This function must be called \b after nut_read_next_packet().
 *
 * If the function returns #NUT_ERR_EAGAIN or #NUT_ERR_EOF, \a len will
 * return the amount of bytes actually read.
 *
 * Data is always written to the \b beginning of the buffer \a buf, so it
 * must be moved accordingly in case of being called again for
 * #NUT_ERR_EAGAIN.
 *
 * \par Example:
 * \code
 * nut_packet_tt pd;
 * int len = pd.len;
 * while((err = nut_read_frame(nut, &len, buf+pd.len-len)) == NUT_ERR_EAGAIN)
 * 	sleep(1);
 * \endcode
 */

/*! \fn int nut_seek(nut_context_tt * nut, double time_pos, int flags, const int * active_streams)
 * \param nut            NUT demuxer context
 * \param time_pos       position to seek to in seconds
 * \param flags          bitfield with seek options
 * \param active_streams List of all active streams terminated by -1,
 *                       may be NULL indicating that all streams are active.
 *
 * Flag bitfield options:
 * - 1 RELATIVE: If set, \a time_pos is relative to current position,
 *               otherwise it is absolute.
 * - 2 FORWARD:  If set, the seek should find the nearest keyframe
 *               after the target position. If unset, the nearest
 *               keyframe before the target position is picked.
 *
 * Active streams decide which keyframe the function will seek to. In
 * backwards seeking, a keyframe for all active streams is garunteed to be
 * found before the target presentation timestamp.
 *
 * In forward seeking, no precise seeking is performed. The function seeks
 * to the first keyframe for an active stream after the target timestamp.
 *
 * If the function returns #NUT_ERR_NOT_SEEKABLE, no seek was performed.
 * If #NUT_ERR_EOF is returned, the requested position is outside the
 * bounds of the file.
 *
 * After nut_seek, nut_read_next_packet should be called to get the next frame.
 */

#endif // LIBNUT_NUT_H
