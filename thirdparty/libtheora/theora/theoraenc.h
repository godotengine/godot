/********************************************************************
 *                                                                  *
 * THIS FILE IS PART OF THE OggTheora SOFTWARE CODEC SOURCE CODE.   *
 * USE, DISTRIBUTION AND REPRODUCTION OF THIS LIBRARY SOURCE IS     *
 * GOVERNED BY A BSD-STYLE SOURCE LICENSE INCLUDED WITH THIS SOURCE *
 * IN 'COPYING'. PLEASE READ THESE TERMS BEFORE DISTRIBUTING.       *
 *                                                                  *
 * THE Theora SOURCE CODE IS COPYRIGHT (C) 2002-2009                *
 * by the Xiph.Org Foundation https://www.xiph.org/                 *
 *                                                                  *
 ********************************************************************

  function:

 ********************************************************************/

/**\file
 * The <tt>libtheoraenc</tt> C encoding API.*/

#if !defined(OGG_THEORA_THEORAENC_HEADER)
# define OGG_THEORA_THEORAENC_HEADER (1)
# include <stddef.h>
# include <ogg/ogg.h>
# include "codec.h"

#if defined(__cplusplus)
extern "C" {
#endif



/**\name th_encode_ctl() codes
 * \anchor encctlcodes
 * These are the available request codes for th_encode_ctl().
 * By convention, these are even, to distinguish them from the
 *  \ref decctlcodes "decoder control codes".
 * Keep any experimental or vendor-specific values above \c 0x8000.*/
/*@{*/
/**Sets the Huffman tables to use.
 * The tables are copied, not stored by reference, so they can be freed after
 *  this call.
 * <tt>NULL</tt> may be specified to revert to the default tables.
 *
 * \param[in] _buf <tt>#th_huff_code[#TH_NHUFFMAN_TABLES][#TH_NDCT_TOKENS]</tt>
 * \retval TH_EFAULT \a _enc is <tt>NULL</tt>.
 * \retval TH_EINVAL Encoding has already begun or one or more of the given
 *                     tables is not full or prefix-free, \a _buf is
 *                     <tt>NULL</tt> and \a _buf_sz is not zero, or \a _buf is
 *                     non-<tt>NULL</tt> and \a _buf_sz is not
 *                     <tt>sizeof(#th_huff_code)*#TH_NHUFFMAN_TABLES*#TH_NDCT_TOKENS</tt>.
 * \retval TH_EIMPL   Not supported by this implementation.*/
#define TH_ENCCTL_SET_HUFFMAN_CODES (0)
/**Sets the quantization parameters to use.
 * The parameters are copied, not stored by reference, so they can be freed
 *  after this call.
 * <tt>NULL</tt> may be specified to revert to the default parameters.
 *
 * \param[in] _buf #th_quant_info
 * \retval TH_EFAULT \a _enc is <tt>NULL</tt>.
 * \retval TH_EINVAL Encoding has already begun, \a _buf is
 *                    <tt>NULL</tt> and \a _buf_sz is not zero,
 *                    or \a _buf is non-<tt>NULL</tt> and
 *                    \a _buf_sz is not <tt>sizeof(#th_quant_info)</tt>.
 * \retval TH_EIMPL   Not supported by this implementation.*/
#define TH_ENCCTL_SET_QUANT_PARAMS (2)
/**Sets the maximum distance between key frames.
 * This can be changed during an encode, but will be bounded by
 *  <tt>1<<th_info#keyframe_granule_shift</tt>.
 * If it is set before encoding begins, th_info#keyframe_granule_shift will
 *  be enlarged appropriately.
 *
 * \param[in]  _buf <tt>ogg_uint32_t</tt>: The maximum distance between key
 *                   frames.
 * \param[out] _buf <tt>ogg_uint32_t</tt>: The actual maximum distance set.
 * \retval TH_EFAULT \a _enc or \a _buf is <tt>NULL</tt>.
 * \retval TH_EINVAL \a _buf_sz is not <tt>sizeof(ogg_uint32_t)</tt>.
 * \retval TH_EIMPL   Not supported by this implementation.*/
#define TH_ENCCTL_SET_KEYFRAME_FREQUENCY_FORCE (4)
/**Disables any encoder features that would prevent lossless transcoding back
 *  to VP3.
 * This primarily means disabling block-adaptive quantization and always coding
 *  all four luma blocks in a macro block when 4MV is used.
 * It also includes using the VP3 quantization tables and Huffman codes; if you
 *  set them explicitly after calling this function, the resulting stream will
 *  not be VP3-compatible.
 * If you enable VP3-compatibility when encoding 4:2:2 or 4:4:4 source
 *  material, or when using a picture region smaller than the full frame (e.g.
 *  a non-multiple-of-16 width or height), then non-VP3 bitstream features will
 *  still be disabled, but the stream will still not be VP3-compatible, as VP3
 *  was not capable of encoding such formats.
 * If you call this after encoding has already begun, then the quantization
 *  tables and codebooks cannot be changed, but the frame-level features will
 *  be enabled or disabled as requested.
 *
 * \param[in]  _buf <tt>int</tt>: a non-zero value to enable VP3 compatibility,
 *                   or 0 to disable it (the default).
 * \param[out] _buf <tt>int</tt>: 1 if all bitstream features required for
 *                   VP3-compatibility could be set, and 0 otherwise.
 *                  The latter will be returned if the pixel format is not
 *                   4:2:0, the picture region is smaller than the full frame,
 *                   or if encoding has begun, preventing the quantization
 *                   tables and codebooks from being set.
 * \retval TH_EFAULT \a _enc or \a _buf is <tt>NULL</tt>.
 * \retval TH_EINVAL \a _buf_sz is not <tt>sizeof(int)</tt>.
 * \retval TH_EIMPL   Not supported by this implementation.*/
#define TH_ENCCTL_SET_VP3_COMPATIBLE (10)
/**Gets the maximum speed level.
 * Higher speed levels favor quicker encoding over better quality per bit.
 * Depending on the encoding mode, and the internal algorithms used, quality
 *  may actually improve, but in this case bitrate will also likely increase.
 * In any case, overall rate/distortion performance will probably decrease.
 * The maximum value, and the meaning of each value, may change depending on
 *  the current encoding mode (VBR vs. constant quality, etc.).
 *
 * \param[out] _buf <tt>int</tt>: The maximum encoding speed level.
 * \retval TH_EFAULT \a _enc or \a _buf is <tt>NULL</tt>.
 * \retval TH_EINVAL \a _buf_sz is not <tt>sizeof(int)</tt>.
 * \retval TH_EIMPL   Not supported by this implementation in the current
 *                    encoding mode.*/
#define TH_ENCCTL_GET_SPLEVEL_MAX (12)
/**Sets the speed level.
 * The current speed level may be retrieved using #TH_ENCCTL_GET_SPLEVEL.
 *
 * \param[in] _buf <tt>int</tt>: The new encoding speed level.
 *                 0 is slowest, larger values use less CPU.
 * \retval TH_EFAULT \a _enc or \a _buf is <tt>NULL</tt>.
 * \retval TH_EINVAL \a _buf_sz is not <tt>sizeof(int)</tt>, or the
 *                    encoding speed level is out of bounds.
 *                   The maximum encoding speed level may be
 *                    implementation- and encoding mode-specific, and can be
 *                    obtained via #TH_ENCCTL_GET_SPLEVEL_MAX.
 * \retval TH_EIMPL   Not supported by this implementation in the current
 *                    encoding mode.*/
#define TH_ENCCTL_SET_SPLEVEL (14)
/**Gets the current speed level.
 * The default speed level may vary according to encoder implementation, but if
 *  this control code is not supported (it returns #TH_EIMPL), the default may
 *  be assumed to be the slowest available speed (0).
 * The maximum encoding speed level may be implementation- and encoding
 *  mode-specific, and can be obtained via #TH_ENCCTL_GET_SPLEVEL_MAX.
 *
 * \param[out] _buf <tt>int</tt>: The current encoding speed level.
 *                  0 is slowest, larger values use less CPU.
 * \retval TH_EFAULT \a _enc or \a _buf is <tt>NULL</tt>.
 * \retval TH_EINVAL \a _buf_sz is not <tt>sizeof(int)</tt>.
 * \retval TH_EIMPL   Not supported by this implementation in the current
 *                    encoding mode.*/
#define TH_ENCCTL_GET_SPLEVEL (16)
/**Sets the number of duplicates of the next frame to produce.
 * Although libtheora can encode duplicate frames very cheaply, it costs some
 *  amount of CPU to detect them, and a run of duplicates cannot span a
 *  keyframe boundary.
 * This control code tells the encoder to produce the specified number of extra
 *  duplicates of the next frame.
 * This allows the encoder to make smarter keyframe placement decisions and
 *  rate control decisions, and reduces CPU usage as well, when compared to
 *  just submitting the same frame for encoding multiple times.
 * This setting only applies to the next frame submitted for encoding.
 * You MUST call th_encode_packetout() repeatedly until it returns 0, or the
 *  extra duplicate frames will be lost.
 *
 * \param[in] _buf <tt>int</tt>: The number of duplicates to produce.
 *                 If this is negative or zero, no duplicates will be produced.
 * \retval TH_EFAULT \a _enc or \a _buf is <tt>NULL</tt>.
 * \retval TH_EINVAL \a _buf_sz is not <tt>sizeof(int)</tt>, or the
 *                    number of duplicates is greater than or equal to the
 *                    maximum keyframe interval.
 *                   In the latter case, NO duplicate frames will be produced.
 *                   You must ensure that the maximum keyframe interval is set
 *                    larger than the maximum number of duplicates you will
 *                    ever wish to insert prior to encoding.
 * \retval TH_EIMPL   Not supported by this implementation in the current
 *                    encoding mode.*/
#define TH_ENCCTL_SET_DUP_COUNT (18)
/**Modifies the default bitrate management behavior.
 * Use to allow or disallow frame dropping, and to enable or disable capping
 *  bit reservoir overflows and underflows.
 * See \ref encctlcodes "the list of available flags".
 * The flags are set by default to
 *  <tt>#TH_RATECTL_DROP_FRAMES|#TH_RATECTL_CAP_OVERFLOW</tt>.
 *
 * \param[in] _buf <tt>int</tt>: Any combination of
 *                  \ref ratectlflags "the available flags":
 *                 - #TH_RATECTL_DROP_FRAMES: Enable frame dropping.
 *                 - #TH_RATECTL_CAP_OVERFLOW: Don't bank excess bits for later
 *                    use.
 *                 - #TH_RATECTL_CAP_UNDERFLOW: Don't try to make up shortfalls
 *                    later.
 * \retval TH_EFAULT \a _enc or \a _buf is <tt>NULL</tt>.
 * \retval TH_EINVAL \a _buf_sz is not <tt>sizeof(int)</tt> or rate control
 *                    is not enabled.
 * \retval TH_EIMPL   Not supported by this implementation in the current
 *                    encoding mode.*/
#define TH_ENCCTL_SET_RATE_FLAGS (20)
/**Sets the size of the bitrate management bit reservoir as a function
 *  of number of frames.
 * The reservoir size affects how quickly bitrate management reacts to
 *  instantaneous changes in the video complexity.
 * Larger reservoirs react more slowly, and provide better overall quality, but
 *  require more buffering by a client, adding more latency to live streams.
 * By default, libtheora sets the reservoir to the maximum distance between
 *  keyframes, subject to a minimum and maximum limit.
 * This call may be used to increase or decrease the reservoir, increasing or
 *  decreasing the allowed temporary variance in bitrate.
 * An implementation may impose some limits on the size of a reservoir it can
 *  handle, in which case the actual reservoir size may not be exactly what was
 *  requested.
 * The actual value set will be returned.
 *
 * \param[in]  _buf <tt>int</tt>: Requested size of the reservoir measured in
 *                   frames.
 * \param[out] _buf <tt>int</tt>: The actual size of the reservoir set.
 * \retval TH_EFAULT \a _enc or \a _buf is <tt>NULL</tt>.
 * \retval TH_EINVAL \a _buf_sz is not <tt>sizeof(int)</tt>, or rate control
 *                    is not enabled.  The buffer has an implementation
 *                    defined minimum and maximum size and the value in _buf
 *                    will be adjusted to match the actual value set.
 * \retval TH_EIMPL   Not supported by this implementation in the current
 *                    encoding mode.*/
#define TH_ENCCTL_SET_RATE_BUFFER (22)
/**Enable pass 1 of two-pass encoding mode and retrieve the first pass metrics.
 * Pass 1 mode must be enabled before the first frame is encoded, and a target
 *  bitrate must have already been specified to the encoder.
 * Although this does not have to be the exact rate that will be used in the
 *  second pass, closer values may produce better results.
 * The first call returns the size of the two-pass header data, along with some
 *  placeholder content, and sets the encoder into pass 1 mode implicitly.
 * This call sets the encoder to pass 1 mode implicitly.
 * Then, a subsequent call must be made after each call to
 *  th_encode_ycbcr_in() to retrieve the metrics for that frame.
 * An additional, final call must be made to retrieve the summary data,
 *  containing such information as the total number of frames, etc.
 * This must be stored in place of the placeholder data that was returned
 *  in the first call, before the frame metrics data.
 * All of this data must be presented back to the encoder during pass 2 using
 *  #TH_ENCCTL_2PASS_IN.
 *
 * \param[out] <tt>char *</tt>_buf: Returns a pointer to internal storage
 *              containing the two pass metrics data.
 *             This storage is only valid until the next call, or until the
 *              encoder context is freed, and must be copied by the
 *              application.
 * \retval >=0       The number of bytes of metric data available in the
 *                    returned buffer.
 * \retval TH_EFAULT \a _enc or \a _buf is <tt>NULL</tt>.
 * \retval TH_EINVAL \a _buf_sz is not <tt>sizeof(char *)</tt>, no target
 *                    bitrate has been set, or the first call was made after
 *                    the first frame was submitted for encoding.
 * \retval TH_EIMPL   Not supported by this implementation.*/
#define TH_ENCCTL_2PASS_OUT (24)
/**Submits two-pass encoding metric data collected the first encoding pass to
 *  the second pass.
 * The first call must be made before the first frame is encoded, and a target
 *  bitrate must have already been specified to the encoder.
 * It sets the encoder to pass 2 mode implicitly; this cannot be disabled.
 * The encoder may require reading data from some or all of the frames in
 *  advance, depending on, e.g., the reservoir size used in the second pass.
 * You must call this function repeatedly before each frame to provide data
 *  until either a) it fails to consume all of the data presented or b) all of
 *  the pass 1 data has been consumed.
 * In the first case, you must save the remaining data to be presented after
 *  the next frame.
 * You can call this function with a NULL argument to get an upper bound on
 *  the number of bytes that will be required before the next frame.
 *
 * When pass 2 is first enabled, the default bit reservoir is set to the entire
 *  file; this gives maximum flexibility but can lead to very high peak rates.
 * You can subsequently set it to another value with #TH_ENCCTL_SET_RATE_BUFFER
 *  (e.g., to set it to the keyframe interval for non-live streaming), however,
 *  you may then need to provide more data before the next frame.
 *
 * \param[in] _buf <tt>char[]</tt>: A buffer containing the data returned by
 *                  #TH_ENCCTL_2PASS_OUT in pass 1.
 *                 You may pass <tt>NULL</tt> for \a _buf to return an upper
 *                  bound on the number of additional bytes needed before the
 *                  next frame.
 *                 The summary data returned at the end of pass 1 must be at
 *                  the head of the buffer on the first call with a
 *                  non-<tt>NULL</tt> \a _buf, and the placeholder data
 *                  returned at the start of pass 1 should be omitted.
 *                 After each call you should advance this buffer by the number
 *                  of bytes consumed.
 * \retval >0            The number of bytes of metric data required/consumed.
 * \retval 0             No more data is required before the next frame.
 * \retval TH_EFAULT     \a _enc is <tt>NULL</tt>.
 * \retval TH_EINVAL     No target bitrate has been set, or the first call was
 *                        made after the first frame was submitted for
 *                        encoding.
 * \retval TH_ENOTFORMAT The data did not appear to be pass 1 from a compatible
 *                        implementation of this library.
 * \retval TH_EBADHEADER The data was invalid; this may be returned when
 *                        attempting to read an aborted pass 1 file that still
 *                        has the placeholder data in place of the summary
 *                        data.
 * \retval TH_EIMPL       Not supported by this implementation.*/
#define TH_ENCCTL_2PASS_IN (26)
/**Sets the current encoding quality.
 * This is only valid so long as no bitrate has been specified, either through
 *  the #th_info struct used to initialize the encoder or through
 *  #TH_ENCCTL_SET_BITRATE (this restriction may be relaxed in a future
 *  version).
 * If it is set before the headers are emitted, the target quality encoded in
 *  them will be updated.
 *
 * \param[in] _buf <tt>int</tt>: The new target quality, in the range 0...63,
 *                  inclusive.
 * \retval 0             Success.
 * \retval TH_EFAULT     \a _enc or \a _buf is <tt>NULL</tt>.
 * \retval TH_EINVAL     A target bitrate has already been specified, or the
 *                        quality index was not in the range 0...63.
 * \retval TH_EIMPL       Not supported by this implementation.*/
#define TH_ENCCTL_SET_QUALITY (28)
/**Sets the current encoding bitrate.
 * Once a bitrate is set, the encoder must use a rate-controlled mode for all
 *  future frames (this restriction may be relaxed in a future version).
 * If it is set before the headers are emitted, the target bitrate encoded in
 *  them will be updated.
 * Due to the buffer delay, the exact bitrate of each section of the encode is
 *  not guaranteed.
 * The encoder may have already used more bits than allowed for the frames it
 *  has encoded, expecting to make them up in future frames, or it may have
 *  used fewer, holding the excess in reserve.
 * The exact transition between the two bitrates is not well-defined by this
 *  API, but may be affected by flags set with #TH_ENCCTL_SET_RATE_FLAGS.
 * After a number of frames equal to the buffer delay, one may expect further
 *  output to average at the target bitrate.
 *
 * \param[in] _buf <tt>long</tt>: The new target bitrate, in bits per second.
 * \retval 0             Success.
 * \retval TH_EFAULT     \a _enc or \a _buf is <tt>NULL</tt>.
 * \retval TH_EINVAL     The target bitrate was not positive.
 *                       A future version of this library may allow passing 0
 *                        to disabled rate-controlled mode and return to a
 *                        quality-based mode, in which case this function will
 *                        not return an error for that value.
 * \retval TH_EIMPL      Not supported by this implementation.*/
#define TH_ENCCTL_SET_BITRATE (30)
/**Sets the configuration to be compatible with that from the given setup
 *  header.
 * This sets the Huffman codebooks and quantization parameters to match those
 *  found in the given setup header.
 * This guarantees that packets encoded by this encoder will be decodable using
 *  a decoder configured with the passed-in setup header.
 * It does <em>not</em> guarantee that th_encode_flushheader() will produce a
 *  bit-identical setup header, only that they will be compatible.
 * If you need a bit-identical setup header, then use the one you passed into
 *  this command, and not the one returned by th_encode_flushheader().
 *
 * This also does <em>not</em> enable or disable VP3 compatibility; that is not
 *  signaled in the setup header (or anywhere else in the encoded stream), and
 *  is controlled independently by the #TH_ENCCTL_SET_VP3_COMPATIBLE function.
 * If you wish to enable VP3 compatibility mode <em>and</em> want the codebooks
 *  and quantization parameters to match the given setup header, you should
 *  enable VP3 compatibility before invoking this command, otherwise the
 *  codebooks and quantization parameters will be reset to the VP3 defaults.
 *
 * The current encoder does not support Huffman codebooks which do not contain
 *  codewords for all 32 tokens.
 * Such codebooks are legal, according to the specification, but cannot be
 *  configured with this function.
 *
 * \param[in] _buf <tt>unsigned char[]</tt>: The encoded setup header to copy
 *                                            the configuration from.
 *                                           This should be the original,
 *                                            undecoded setup header packet,
 *                                            and <em>not</em> a #th_setup_info
 *                                            structure filled in by
 *                                            th_decode_headerin().
 * \retval TH_EFAULT     \a _enc or \a _buf is <tt>NULL</tt>.
 * \retval TH_EINVAL     Encoding has already begun, so the codebooks and
 *                        quantization parameters cannot be changed, or the
 *                        data in the setup header was not supported by this
 *                        encoder.
 * \retval TH_EBADHEADER \a _buf did not contain a valid setup header packet.
 * \retval TH_ENOTFORMAT \a _buf did not contain a Theora header at all.
 * \retval TH_EIMPL   Not supported by this implementation.*/
#define TH_ENCCTL_SET_COMPAT_CONFIG (32)

/*@}*/


/**\name TH_ENCCTL_SET_RATE_FLAGS flags
 * \anchor ratectlflags
 * These are the flags available for use with #TH_ENCCTL_SET_RATE_FLAGS.*/
/*@{*/
/**Drop frames to keep within bitrate buffer constraints.
 * This can have a severe impact on quality, but is the only way to ensure that
 *  bitrate targets are met at low rates during sudden bursts of activity.
 * It is enabled by default.*/
#define TH_RATECTL_DROP_FRAMES   (0x1)
/**Ignore bitrate buffer overflows.
 * If the encoder uses so few bits that the reservoir of available bits
 *  overflows, ignore the excess.
 * The encoder will not try to use these extra bits in future frames.
 * At high rates this may cause the result to be undersized, but allows a
 *  client to play the stream using a finite buffer; it should normally be
 *  enabled, which is the default.*/
#define TH_RATECTL_CAP_OVERFLOW  (0x2)
/**Ignore bitrate buffer underflows.
 * If the encoder uses so many bits that the reservoir of available bits
 *  underflows, ignore the deficit.
 * The encoder will not try to make up these extra bits in future frames.
 * At low rates this may cause the result to be oversized; it should normally
 *  be disabled, which is the default.*/
#define TH_RATECTL_CAP_UNDERFLOW (0x4)
/*@}*/



/**The quantization parameters used by VP3.*/
extern const th_quant_info TH_VP31_QUANT_INFO;

/**The Huffman tables used by VP3.*/
extern const th_huff_code
 TH_VP31_HUFF_CODES[TH_NHUFFMAN_TABLES][TH_NDCT_TOKENS];



/**\name Encoder state
   The following data structure is opaque, and its contents are not publicly
    defined by this API.
   Referring to its internals directly is unsupported, and may break without
    warning.*/
/*@{*/
/**The encoder context.*/
typedef struct th_enc_ctx    th_enc_ctx;
/*@}*/



/**\defgroup encfuncs Functions for Encoding*/
/*@{*/
/**\name Functions for encoding
 * You must link to <tt>libtheoraenc</tt> and <tt>libtheoradec</tt>
 *  if you use any of the functions in this section.
 *
 * The functions are listed in the order they are used in a typical encode.
 * The basic steps are:
 * - Fill in a #th_info structure with details on the format of the video you
 *    wish to encode.
 * - Allocate a #th_enc_ctx handle with th_encode_alloc().
 * - Perform any additional encoder configuration required with
 *    th_encode_ctl().
 * - Repeatedly call th_encode_flushheader() to retrieve all the header
 *    packets.
 * - For each uncompressed frame:
 *   - Submit the uncompressed frame via th_encode_ycbcr_in()
 *   - Repeatedly call th_encode_packetout() to retrieve any video
 *      data packets that are ready.
 * - Call th_encode_free() to release all encoder memory.*/
/*@{*/
/**Allocates an encoder instance.
 * \param _info A #th_info struct filled with the desired encoding parameters.
 * \return The initialized #th_enc_ctx handle.
 * \retval NULL If the encoding parameters were invalid.*/
extern th_enc_ctx *th_encode_alloc(const th_info *_info);
/**Encoder control function.
 * This is used to provide advanced control the encoding process.
 * \param _enc    A #th_enc_ctx handle.
 * \param _req    The control code to process.
 *                See \ref encctlcodes "the list of available control codes"
 *                 for details.
 * \param _buf    The parameters for this control code.
 * \param _buf_sz The size of the parameter buffer.
 * \return Possible return values depend on the control code used.
 *          See \ref encctlcodes "the list of control codes" for
 *          specific values. Generally 0 indicates success.*/
extern int th_encode_ctl(th_enc_ctx *_enc,int _req,void *_buf,size_t _buf_sz);
/**Outputs the next header packet.
 * This should be called repeatedly after encoder initialization until it
 *  returns 0 in order to get all of the header packets, in order, before
 *  encoding actual video data.
 * \param _enc      A #th_enc_ctx handle.
 * \param _comments The metadata to place in the comment header, when it is
 *                   encoded.
 * \param _op       An <tt>ogg_packet</tt> structure to fill.
 *                  All of the elements of this structure will be set,
 *                   including a pointer to the header data.
 *                  The memory for the header data is owned by
 *                   <tt>libtheoraenc</tt>, and may be invalidated when the
 *                   next encoder function is called.
 * \return A positive value indicates that a header packet was successfully
 *          produced.
 * \retval 0         No packet was produced, and no more header packets remain.
 * \retval TH_EFAULT \a _enc, \a _comments, or \a _op was <tt>NULL</tt>.*/
extern int th_encode_flushheader(th_enc_ctx *_enc,
 th_comment *_comments,ogg_packet *_op);
/**Submits an uncompressed frame to the encoder.
 * \param _enc   A #th_enc_ctx handle.
 * \param _ycbcr A buffer of Y'CbCr data to encode.
 *               If the width and height of the buffer matches the frame size
 *                the encoder was initialized with, the encoder will only
 *                reference the portion inside the picture region.
 *               Any data outside this region will be ignored, and need not map
 *                to a valid address.
 *               Alternatively, you can pass a buffer equal to the size of the
 *                picture region, if this is less than the full frame size.
 *               When using subsampled chroma planes, odd picture sizes or odd
 *                picture offsets may require an unexpected chroma plane size,
 *                and their use is generally discouraged, as they will not be
 *                well-supported by players and other media frameworks.
 *               See Section 4.4 of
 *                <a href="https://www.theora.org/doc/Theora.pdf">the Theora
 *                specification</a> for details if you wish to use them anyway.
 * \retval 0         Success.
 * \retval TH_EFAULT \a _enc or \a _ycbcr is <tt>NULL</tt>.
 * \retval TH_EINVAL The buffer size matches neither the frame size nor the
 *                    picture size the encoder was initialized with, or
 *                    encoding has already completed.*/
extern int th_encode_ycbcr_in(th_enc_ctx *_enc,th_ycbcr_buffer _ycbcr);
/**Retrieves encoded video data packets.
 * This should be called repeatedly after each frame is submitted to flush any
 *  encoded packets, until it returns 0.
 * The encoder will not buffer these packets as subsequent frames are
 *  compressed, so a failure to do so will result in lost video data.
 * \note Currently the encoder operates in a one-frame-in, one-packet-out
 *        manner.
 *       However, this may be changed in the future.
 * \param _enc  A #th_enc_ctx handle.
 * \param _last Set this flag to a non-zero value if no more uncompressed
 *               frames will be submitted.
 *              This ensures that a proper EOS flag is set on the last packet.
 * \param _op   An <tt>ogg_packet</tt> structure to fill.
 *              All of the elements of this structure will be set, including a
 *               pointer to the video data.
 *              The memory for the video data is owned by
 *               <tt>libtheoraenc</tt>, and may be invalidated when the next
 *               encoder function is called.
 * \return A positive value indicates that a video data packet was successfully
 *          produced.
 * \retval 0         No packet was produced, and no more encoded video data
 *                    remains.
 * \retval TH_EFAULT \a _enc or \a _op was <tt>NULL</tt>.*/
extern int th_encode_packetout(th_enc_ctx *_enc,int _last,ogg_packet *_op);
/**Frees an allocated encoder instance.
 * \param _enc A #th_enc_ctx handle.*/
extern void th_encode_free(th_enc_ctx *_enc);
/*@}*/
/*@}*/



#if defined(__cplusplus)
}
#endif

#endif /* OGG_THEORA_THEORAENC_HEADER */
