/********************************************************************
 *                                                                  *
 * THIS FILE IS PART OF THE OggTheora SOFTWARE CODEC SOURCE CODE.   *
 * USE, DISTRIBUTION AND REPRODUCTION OF THIS LIBRARY SOURCE IS     *
 * GOVERNED BY A BSD-STYLE SOURCE LICENSE INCLUDED WITH THIS SOURCE *
 * IN 'COPYING'. PLEASE READ THESE TERMS BEFORE DISTRIBUTING.       *
 *                                                                  *
 * THE Theora SOURCE CODE IS COPYRIGHT (C) 2002-2009                *
 * by the Xiph.Org Foundation http://www.xiph.org/                  *
 *                                                                  *
 ********************************************************************

  function:
  last mod: $Id: theora.h,v 1.17 2003/12/06 18:06:19 arc Exp $

 ********************************************************************/

#ifndef _O_THEORA_H_
#define _O_THEORA_H_

#ifdef __cplusplus
extern "C"
{
#endif /* __cplusplus */

#include <stddef.h>	/* for size_t */

#include <ogg/ogg.h>

/** \file
 * The libtheora pre-1.0 legacy C API.
 *
 * \ingroup oldfuncs
 *
 * \section intro Introduction
 *
 * This is the documentation for the libtheora legacy C API, declared in 
 * the theora.h header, which describes the old interface used before
 * the 1.0 release. This API was widely deployed for several years and
 * remains supported, but for new code we recommend the cleaner API 
 * declared in theoradec.h and theoraenc.h.
 *
 * libtheora is the reference implementation for
 * <a href="http://www.theora.org/">Theora</a>, a free video codec.
 * Theora is derived from On2's VP3 codec with improved integration with
 * Ogg multimedia formats by <a href="http://www.xiph.org/">Xiph.Org</a>.
 * 
 * \section overview Overview
 *
 * This library will both decode and encode theora packets to/from raw YUV 
 * frames.  In either case, the packets will most likely either come from or
 * need to be embedded in an Ogg stream.  Use 
 * <a href="http://xiph.org/ogg/">libogg</a> or 
 * <a href="http://www.annodex.net/software/liboggz/index.html">liboggz</a>
 * to extract/package these packets.
 *
 * \section decoding Decoding Process
 *
 * Decoding can be separated into the following steps:
 * -# initialise theora_info and theora_comment structures using 
 *    theora_info_init() and theora_comment_init():
 \verbatim
 theora_info     info;
 theora_comment  comment;
   
 theora_info_init(&info);
 theora_comment_init(&comment);
 \endverbatim
 * -# retrieve header packets from Ogg stream (there should be 3) and decode 
 *    into theora_info and theora_comment structures using 
 *    theora_decode_header().  See \ref identification for more information on 
 *    identifying which packets are theora packets.
 \verbatim
 int i;
 for (i = 0; i < 3; i++)
 {
   (get a theora packet "op" from the Ogg stream)
   theora_decode_header(&info, &comment, op);
 }
 \endverbatim
 * -# initialise the decoder based on the information retrieved into the
 *    theora_info struct by theora_decode_header().  You will need a 
 *    theora_state struct.
 \verbatim
 theora_state state;
 
 theora_decode_init(&state, &info);
 \endverbatim
 * -# pass in packets and retrieve decoded frames!  See the yuv_buffer 
 *    documentation for information on how to retrieve raw YUV data.
 \verbatim
 yuf_buffer buffer;
 while (last packet was not e_o_s) {
   (get a theora packet "op" from the Ogg stream)
   theora_decode_packetin(&state, op);
   theora_decode_YUVout(&state, &buffer);
 }
 \endverbatim
 *  
 *
 * \subsection identification Identifying Theora Packets
 *
 * All streams inside an Ogg file have a unique serial_no attached to the 
 * stream.  Typically, you will want to 
 *  - retrieve the serial_no for each b_o_s (beginning of stream) page 
 *    encountered within the Ogg file; 
 *  - test the first (only) packet on that page to determine if it is a theora 
 *    packet;
 *  - once you have found a theora b_o_s page then use the retrieved serial_no 
 *    to identify future packets belonging to the same theora stream.
 * 
 * Note that you \e cannot use theora_packet_isheader() to determine if a 
 * packet is a theora packet or not, as this function does not perform any
 * checking beyond whether a header bit is present.  Instead, use the
 * theora_decode_header() function and check the return value; or examine the
 * header bytes at the beginning of the Ogg page.
 */


/** \defgroup oldfuncs Legacy pre-1.0 C API */
/*  @{ */

/**
 * A YUV buffer for passing uncompressed frames to and from the codec.
 * This holds a Y'CbCr frame in planar format. The CbCr planes can be
 * subsampled and have their own separate dimensions and row stride
 * offsets. Note that the strides may be negative in some 
 * configurations. For theora the width and height of the largest plane
 * must be a multiple of 16. The actual meaningful picture size and 
 * offset are stored in the theora_info structure; frames returned by
 * the decoder may need to be cropped for display.
 *
 * All samples are 8 bits. Within each plane samples are ordered by
 * row from the top of the frame to the bottom. Within each row samples
 * are ordered from left to right.
 *
 * During decode, the yuv_buffer struct is allocated by the user, but all
 * fields (including luma and chroma pointers) are filled by the library.  
 * These pointers address library-internal memory and their contents should 
 * not be modified.
 *
 * Conversely, during encode the user allocates the struct and fills out all
 * fields.  The user also manages the data addressed by the luma and chroma
 * pointers.  See the encoder_example.c and dump_video.c example files in
 * theora/examples/ for more information.
 */
typedef struct {
    int   y_width;      /**< Width of the Y' luminance plane */
    int   y_height;     /**< Height of the luminance plane */
    int   y_stride;     /**< Offset in bytes between successive rows */

    int   uv_width;     /**< Width of the Cb and Cr chroma planes */
    int   uv_height;    /**< Height of the chroma planes */
    int   uv_stride;    /**< Offset between successive chroma rows */
    unsigned char *y;   /**< Pointer to start of luminance data */
    unsigned char *u;   /**< Pointer to start of Cb data */
    unsigned char *v;   /**< Pointer to start of Cr data */

} yuv_buffer;

/**
 * A Colorspace.
 */
typedef enum {
  OC_CS_UNSPECIFIED,    /**< The colorspace is unknown or unspecified */
  OC_CS_ITU_REC_470M,   /**< This is the best option for 'NTSC' content */
  OC_CS_ITU_REC_470BG,  /**< This is the best option for 'PAL' content */
  OC_CS_NSPACES         /**< This marks the end of the defined colorspaces */
} theora_colorspace;

/**
 * A Chroma subsampling
 *
 * These enumerate the available chroma subsampling options supported
 * by the theora format. See Section 4.4 of the specification for
 * exact definitions.
 */
typedef enum {
  OC_PF_420,    /**< Chroma subsampling by 2 in each direction (4:2:0) */
  OC_PF_RSVD,   /**< Reserved value */
  OC_PF_422,    /**< Horizonatal chroma subsampling by 2 (4:2:2) */
  OC_PF_444,    /**< No chroma subsampling at all (4:4:4) */
} theora_pixelformat;

/**
 * Theora bitstream info.
 * Contains the basic playback parameters for a stream,
 * corresponding to the initial 'info' header packet.
 * 
 * Encoded theora frames must be a multiple of 16 in width and height.
 * To handle other frame sizes, a crop rectangle is specified in
 * frame_height and frame_width, offset_x and * offset_y. The offset
 * and size should still be a multiple of 2 to avoid chroma sampling
 * shifts. Offset values in this structure are measured from the
 * upper left of the image.
 *
 * Frame rate, in frames per second, is stored as a rational
 * fraction. Aspect ratio is also stored as a rational fraction, and
 * refers to the aspect ratio of the frame pixels, not of the
 * overall frame itself.
 * 
 * See <a href="http://svn.xiph.org/trunk/theora/examples/encoder_example.c">
 * examples/encoder_example.c</a> for usage examples of the
 * other paramters and good default settings for the encoder parameters.
 */
typedef struct {
  ogg_uint32_t  width;		/**< encoded frame width  */
  ogg_uint32_t  height;		/**< encoded frame height */
  ogg_uint32_t  frame_width;	/**< display frame width  */
  ogg_uint32_t  frame_height;	/**< display frame height */
  ogg_uint32_t  offset_x;	/**< horizontal offset of the displayed frame */
  ogg_uint32_t  offset_y;	/**< vertical offset of the displayed frame */
  ogg_uint32_t  fps_numerator;	    /**< frame rate numerator **/
  ogg_uint32_t  fps_denominator;    /**< frame rate denominator **/
  ogg_uint32_t  aspect_numerator;   /**< pixel aspect ratio numerator */
  ogg_uint32_t  aspect_denominator; /**< pixel aspect ratio denominator */
  theora_colorspace colorspace;	    /**< colorspace */
  int           target_bitrate;	    /**< nominal bitrate in bits per second */
  int           quality;  /**< Nominal quality setting, 0-63 */
  int           quick_p;  /**< Quick encode/decode */

  /* decode only */
  unsigned char version_major;
  unsigned char version_minor;
  unsigned char version_subminor;

  void *codec_setup;

  /* encode only */
  int           dropframes_p;
  int           keyframe_auto_p;
  ogg_uint32_t  keyframe_frequency;
  ogg_uint32_t  keyframe_frequency_force;  /* also used for decode init to
                                              get granpos shift correct */
  ogg_uint32_t  keyframe_data_target_bitrate;
  ogg_int32_t   keyframe_auto_threshold;
  ogg_uint32_t  keyframe_mindistance;
  ogg_int32_t   noise_sensitivity;
  ogg_int32_t   sharpness;

  theora_pixelformat pixelformat;	/**< chroma subsampling mode to expect */

} theora_info;

/** Codec internal state and context.
 */
typedef struct{
  theora_info *i;
  ogg_int64_t granulepos;

  void *internal_encode;
  void *internal_decode;

} theora_state;

/** 
 * Comment header metadata.
 *
 * This structure holds the in-stream metadata corresponding to
 * the 'comment' header packet.
 *
 * Meta data is stored as a series of (tag, value) pairs, in
 * length-encoded string vectors. The first occurence of the 
 * '=' character delimits the tag and value. A particular tag
 * may occur more than once. The character set encoding for
 * the strings is always UTF-8, but the tag names are limited
 * to case-insensitive ASCII. See the spec for details.
 *
 * In filling in this structure, theora_decode_header() will
 * null-terminate the user_comment strings for safety. However,
 * the bitstream format itself treats them as 8-bit clean,
 * and so the length array should be treated as authoritative
 * for their length.
 */
typedef struct theora_comment{
  char **user_comments;         /**< An array of comment string vectors */
  int   *comment_lengths;       /**< An array of corresponding string vector lengths in bytes */
  int    comments;              /**< The total number of comment string vectors */
  char  *vendor;                /**< The vendor string identifying the encoder, null terminated */

} theora_comment;


/**\name theora_control() codes */
/* \anchor decctlcodes_old
 * These are the available request codes for theora_control()
 * when called with a decoder instance.
 * By convention decoder control codes are odd, to distinguish 
 * them from \ref encctlcodes_old "encoder control codes" which
 * are even.
 *
 * Note that since the 1.0 release, both the legacy and the final
 * implementation accept all the same control codes, but only the
 * final API declares the newer codes.
 *
 * Keep any experimental or vendor-specific values above \c 0x8000.*/

/*@{*/

/**Get the maximum post-processing level.
 * The decoder supports a post-processing filter that can improve
 * the appearance of the decoded images. This returns the highest
 * level setting for this post-processor, corresponding to maximum
 * improvement and computational expense.
 */
#define TH_DECCTL_GET_PPLEVEL_MAX (1)

/**Set the post-processing level.
 * Sets the level of post-processing to use when decoding the 
 * compressed stream. This must be a value between zero (off)
 * and the maximum returned by TH_DECCTL_GET_PPLEVEL_MAX.
 */
#define TH_DECCTL_SET_PPLEVEL (3)

/**Sets the maximum distance between key frames.
 * This can be changed during an encode, but will be bounded by
 *  <tt>1<<th_info#keyframe_granule_shift</tt>.
 * If it is set before encoding begins, th_info#keyframe_granule_shift will
 *  be enlarged appropriately.
 *
 * \param[in]  buf <tt>ogg_uint32_t</tt>: The maximum distance between key
 *                   frames.
 * \param[out] buf <tt>ogg_uint32_t</tt>: The actual maximum distance set.
 * \retval OC_FAULT  \a theora_state or \a buf is <tt>NULL</tt>.
 * \retval OC_EINVAL \a buf_sz is not <tt>sizeof(ogg_uint32_t)</tt>.
 * \retval OC_IMPL   Not supported by this implementation.*/
#define TH_ENCCTL_SET_KEYFRAME_FREQUENCY_FORCE (4)

/**Set the granule position.
 * Call this after a seek, to update the internal granulepos
 * in the decoder, to insure that subsequent frames are marked
 * properly. If you track timestamps yourself and do not use
 * the granule postion returned by the decoder, then you do
 * not need to use this control.
 */
#define TH_DECCTL_SET_GRANPOS (5)

/**\anchor encctlcodes_old */

/**Sets the quantization parameters to use.
 * The parameters are copied, not stored by reference, so they can be freed
 *  after this call.
 * <tt>NULL</tt> may be specified to revert to the default parameters.
 *
 * \param[in] buf #th_quant_info
 * \retval OC_FAULT  \a theora_state is <tt>NULL</tt>.
 * \retval OC_EINVAL Encoding has already begun, the quantization parameters
 *                    are not acceptable to this version of the encoder, 
 *                    \a buf is <tt>NULL</tt> and \a buf_sz is not zero, 
 *                    or \a buf is non-<tt>NULL</tt> and \a buf_sz is 
 *                    not <tt>sizeof(#th_quant_info)</tt>.
 * \retval OC_IMPL   Not supported by this implementation.*/
#define TH_ENCCTL_SET_QUANT_PARAMS (2)

/**Disables any encoder features that would prevent lossless transcoding back
 *  to VP3.
 * This primarily means disabling block-level QI values and not using 4MV mode
 *  when any of the luma blocks in a macro block are not coded.
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
 * \param[in]  buf <tt>int</tt>: a non-zero value to enable VP3 compatibility,
 *                   or 0 to disable it (the default).
 * \param[out] buf <tt>int</tt>: 1 if all bitstream features required for
 *                   VP3-compatibility could be set, and 0 otherwise.
 *                  The latter will be returned if the pixel format is not
 *                   4:2:0, the picture region is smaller than the full frame,
 *                   or if encoding has begun, preventing the quantization
 *                   tables and codebooks from being set.
 * \retval OC_FAULT  \a theora_state or \a buf is <tt>NULL</tt>.
 * \retval OC_EINVAL \a buf_sz is not <tt>sizeof(int)</tt>.
 * \retval OC_IMPL   Not supported by this implementation.*/
#define TH_ENCCTL_SET_VP3_COMPATIBLE (10)

/**Gets the maximum speed level.
 * Higher speed levels favor quicker encoding over better quality per bit.
 * Depending on the encoding mode, and the internal algorithms used, quality
 *  may actually improve, but in this case bitrate will also likely increase.
 * In any case, overall rate/distortion performance will probably decrease.
 * The maximum value, and the meaning of each value, may change depending on
 *  the current encoding mode (VBR vs. CQI, etc.).
 *
 * \param[out] buf int: The maximum encoding speed level.
 * \retval OC_FAULT  \a theora_state or \a buf is <tt>NULL</tt>.
 * \retval OC_EINVAL \a buf_sz is not <tt>sizeof(int)</tt>.
 * \retval OC_IMPL   Not supported by this implementation in the current
 *                    encoding mode.*/
#define TH_ENCCTL_GET_SPLEVEL_MAX (12)

/**Sets the speed level.
 * By default a speed value of 1 is used.
 *
 * \param[in] buf int: The new encoding speed level.
 *                      0 is slowest, larger values use less CPU.
 * \retval OC_FAULT  \a theora_state or \a buf is <tt>NULL</tt>.
 * \retval OC_EINVAL \a buf_sz is not <tt>sizeof(int)</tt>, or the
 *                    encoding speed level is out of bounds.
 *                   The maximum encoding speed level may be
 *                    implementation- and encoding mode-specific, and can be
 *                    obtained via #TH_ENCCTL_GET_SPLEVEL_MAX.
 * \retval OC_IMPL   Not supported by this implementation in the current
 *                    encoding mode.*/
#define TH_ENCCTL_SET_SPLEVEL (14)

/*@}*/

#define OC_FAULT       -1       /**< General failure */
#define OC_EINVAL      -10      /**< Library encountered invalid internal data */
#define OC_DISABLED    -11      /**< Requested action is disabled */
#define OC_BADHEADER   -20      /**< Header packet was corrupt/invalid */
#define OC_NOTFORMAT   -21      /**< Packet is not a theora packet */
#define OC_VERSION     -22      /**< Bitstream version is not handled */
#define OC_IMPL        -23      /**< Feature or action not implemented */
#define OC_BADPACKET   -24      /**< Packet is corrupt */
#define OC_NEWPACKET   -25      /**< Packet is an (ignorable) unhandled extension */
#define OC_DUPFRAME    1        /**< Packet is a dropped frame */

/** 
 * Retrieve a human-readable string to identify the encoder vendor and version.
 * \returns A version string.
 */
extern const char *theora_version_string(void);

/**
 * Retrieve a 32-bit version number.
 * This number is composed of a 16-bit major version, 8-bit minor version
 * and 8 bit sub-version, composed as follows:
<pre>
   (VERSION_MAJOR<<16) + (VERSION_MINOR<<8) + (VERSION_SUB)
</pre>
* \returns The version number.
*/
extern ogg_uint32_t theora_version_number(void);

/**
 * Initialize the theora encoder.
 * \param th The theora_state handle to initialize for encoding.
 * \param ti A theora_info struct filled with the desired encoding parameters.
 * \retval 0 Success
 */
extern int theora_encode_init(theora_state *th, theora_info *ti);

/**
 * Submit a YUV buffer to the theora encoder.
 * \param t A theora_state handle previously initialized for encoding.
 * \param yuv A buffer of YUV data to encode.  Note that both the yuv_buffer
 *            struct and the luma/chroma buffers within should be allocated by
 *            the user.
 * \retval OC_EINVAL Encoder is not ready, or is finished.
 * \retval -1 The size of the given frame differs from those previously input
 * \retval 0 Success
 */
extern int theora_encode_YUVin(theora_state *t, yuv_buffer *yuv);

/**
 * Request the next packet of encoded video. 
 * The encoded data is placed in a user-provided ogg_packet structure.
 * \param t A theora_state handle previously initialized for encoding.
 * \param last_p whether this is the last packet the encoder should produce.
 * \param op An ogg_packet structure to fill. libtheora will set all
 *           elements of this structure, including a pointer to encoded
 *           data. The memory for the encoded data is owned by libtheora.
 * \retval 0 No internal storage exists OR no packet is ready
 * \retval -1 The encoding process has completed
 * \retval 1 Success
 */
extern int theora_encode_packetout( theora_state *t, int last_p,
                                    ogg_packet *op);

/**
 * Request a packet containing the initial header.
 * A pointer to the header data is placed in a user-provided ogg_packet
 * structure.
 * \param t A theora_state handle previously initialized for encoding.
 * \param op An ogg_packet structure to fill. libtheora will set all
 *           elements of this structure, including a pointer to the header
 *           data. The memory for the header data is owned by libtheora.
 * \retval 0 Success
 */
extern int theora_encode_header(theora_state *t, ogg_packet *op);

/**
 * Request a comment header packet from provided metadata.
 * A pointer to the comment data is placed in a user-provided ogg_packet
 * structure.
 * \param tc A theora_comment structure filled with the desired metadata
 * \param op An ogg_packet structure to fill. libtheora will set all
 *           elements of this structure, including a pointer to the encoded
 *           comment data. The memory for the comment data is owned by
 *           libtheora.
 * \retval 0 Success
 */
extern int theora_encode_comment(theora_comment *tc, ogg_packet *op);

/**
 * Request a packet containing the codebook tables for the stream.
 * A pointer to the codebook data is placed in a user-provided ogg_packet
 * structure.
 * \param t A theora_state handle previously initialized for encoding.
 * \param op An ogg_packet structure to fill. libtheora will set all
 *           elements of this structure, including a pointer to the codebook
 *           data. The memory for the header data is owned by libtheora.
 * \retval 0 Success
 */
extern int theora_encode_tables(theora_state *t, ogg_packet *op);

/**
 * Decode an Ogg packet, with the expectation that the packet contains
 * an initial header, comment data or codebook tables.
 *
 * \param ci A theora_info structure to fill. This must have been previously
 *           initialized with theora_info_init(). If \a op contains an initial
 *           header, theora_decode_header() will fill \a ci with the
 *           parsed header values. If \a op contains codebook tables,
 *           theora_decode_header() will parse these and attach an internal
 *           representation to \a ci->codec_setup.
 * \param cc A theora_comment structure to fill. If \a op contains comment
 *           data, theora_decode_header() will fill \a cc with the parsed
 *           comments.
 * \param op An ogg_packet structure which you expect contains an initial
 *           header, comment data or codebook tables.
 *
 * \retval OC_BADHEADER \a op is NULL; OR the first byte of \a op->packet
 *                      has the signature of an initial packet, but op is
 *                      not a b_o_s packet; OR this packet has the signature
 *                      of an initial header packet, but an initial header
 *                      packet has already been seen; OR this packet has the
 *                      signature of a comment packet, but the initial header
 *                      has not yet been seen; OR this packet has the signature
 *                      of a comment packet, but contains invalid data; OR
 *                      this packet has the signature of codebook tables,
 *                      but the initial header or comments have not yet
 *                      been seen; OR this packet has the signature of codebook
 *                      tables, but contains invalid data;
 *                      OR the stream being decoded has a compatible version
 *                      but this packet does not have the signature of a
 *                      theora initial header, comments, or codebook packet
 * \retval OC_VERSION   The packet data of \a op is an initial header with
 *                      a version which is incompatible with this version of
 *                      libtheora.
 * \retval OC_NEWPACKET the stream being decoded has an incompatible (future)
 *                      version and contains an unknown signature.
 * \retval 0            Success
 *
 * \note The normal usage is that theora_decode_header() be called on the
 *       first three packets of a theora logical bitstream in succession.
 */
extern int theora_decode_header(theora_info *ci, theora_comment *cc,
                                ogg_packet *op);

/**
 * Initialize a theora_state handle for decoding.
 * \param th The theora_state handle to initialize.
 * \param c  A theora_info struct filled with the desired decoding parameters.
 *           This is of course usually obtained from a previous call to
 *           theora_decode_header().
 * \retval 0 Success
 */
extern int theora_decode_init(theora_state *th, theora_info *c);

/**
 * Input a packet containing encoded data into the theora decoder.
 * \param th A theora_state handle previously initialized for decoding.
 * \param op An ogg_packet containing encoded theora data.
 * \retval 0 Success
 * \retval OC_BADPACKET \a op does not contain encoded video data
 */
extern int theora_decode_packetin(theora_state *th,ogg_packet *op);

/**
 * Output the next available frame of decoded YUV data.
 * \param th A theora_state handle previously initialized for decoding.
 * \param yuv A yuv_buffer in which libtheora should place the decoded data.
 *            Note that the buffer struct itself is allocated by the user, but
 *            that the luma and chroma pointers will be filled in by the 
 *            library.  Also note that these luma and chroma regions should be 
 *            considered read-only by the user.
 * \retval 0 Success
 */
extern int theora_decode_YUVout(theora_state *th,yuv_buffer *yuv);

/**
 * Report whether a theora packet is a header or not
 * This function does no verification beyond checking the header
 * flag bit so it should not be used for bitstream identification;
 * use theora_decode_header() for that.
 *
 * \param op An ogg_packet containing encoded theora data.
 * \retval 1 The packet is a header packet
 * \retval 0 The packet is not a header packet (and so contains frame data)
 *
 * Thus function was added in the 1.0alpha4 release.
 */
extern int theora_packet_isheader(ogg_packet *op);

/**
 * Report whether a theora packet is a keyframe or not
 *
 * \param op An ogg_packet containing encoded theora data.
 * \retval 1 The packet contains a keyframe image
 * \retval 0 The packet is contains an interframe delta
 * \retval -1 The packet is not an image data packet at all
 *
 * Thus function was added in the 1.0alpha4 release.
 */
extern int theora_packet_iskeyframe(ogg_packet *op);

/**
 * Report the granulepos shift radix
 *
 * When embedded in Ogg, Theora uses a two-part granulepos, 
 * splitting the 64-bit field into two pieces. The more-significant
 * section represents the frame count at the last keyframe,
 * and the less-significant section represents the count of
 * frames since the last keyframe. In this way the overall
 * field is still non-decreasing with time, but usefully encodes
 * a pointer to the last keyframe, which is necessary for
 * correctly restarting decode after a seek. 
 *
 * This function reports the number of bits used to represent
 * the distance to the last keyframe, and thus how the granulepos
 * field must be shifted or masked to obtain the two parts.
 * 
 * Since libtheora returns compressed data in an ogg_packet
 * structure, this may be generally useful even if the Theora
 * packets are not being used in an Ogg container. 
 *
 * \param ti A previously initialized theora_info struct
 * \returns The bit shift dividing the two granulepos fields
 *
 * This function was added in the 1.0alpha5 release.
 */
int theora_granule_shift(theora_info *ti);

/**
 * Convert a granulepos to an absolute frame index, starting at 0.
 * The granulepos is interpreted in the context of a given theora_state handle.
 * 
 * Note that while the granulepos encodes the frame count (i.e. starting
 * from 1) this call returns the frame index, starting from zero. Thus
 * One can calculate the presentation time by multiplying the index by
 * the rate.
 *
 * \param th A previously initialized theora_state handle (encode or decode)
 * \param granulepos The granulepos to convert.
 * \returns The frame index corresponding to \a granulepos.
 * \retval -1 The given granulepos is undefined (i.e. negative)
 *
 * Thus function was added in the 1.0alpha4 release.
 */
extern ogg_int64_t theora_granule_frame(theora_state *th,ogg_int64_t granulepos);

/**
 * Convert a granulepos to absolute time in seconds. The granulepos is
 * interpreted in the context of a given theora_state handle, and gives
 * the end time of a frame's presentation as used in Ogg mux ordering.
 *
 * \param th A previously initialized theora_state handle (encode or decode)
 * \param granulepos The granulepos to convert.
 * \returns The absolute time in seconds corresponding to \a granulepos.
 *          This is the "end time" for the frame, or the latest time it should
 *           be displayed.
 *          It is not the presentation time.
 * \retval -1. The given granulepos is undefined (i.e. negative), or
 * \retval -1. The function has been disabled because floating 
 *              point support is not available.
 */
extern double theora_granule_time(theora_state *th,ogg_int64_t granulepos);

/**
 * Initialize a theora_info structure. All values within the given theora_info
 * structure are initialized, and space is allocated within libtheora for
 * internal codec setup data.
 * \param c A theora_info struct to initialize.
 */
extern void theora_info_init(theora_info *c);

/**
 * Clear a theora_info structure. All values within the given theora_info
 * structure are cleared, and associated internal codec setup data is freed.
 * \param c A theora_info struct to initialize.
 */
extern void theora_info_clear(theora_info *c);

/**
 * Free all internal data associated with a theora_state handle.
 * \param t A theora_state handle.
 */
extern void theora_clear(theora_state *t);

/**
 * Initialize an allocated theora_comment structure
 * \param tc An allocated theora_comment structure 
 **/
extern void theora_comment_init(theora_comment *tc);

/**
 * Add a comment to an initialized theora_comment structure
 * \param tc A previously initialized theora comment structure
 * \param comment A null-terminated string encoding the comment in the form
 *                "TAG=the value"
 *
 * Neither theora_comment_add() nor theora_comment_add_tag() support
 * comments containing null values, although the bitstream format
 * supports this. To add such comments you will need to manipulate
 * the theora_comment structure directly.
 **/

extern void theora_comment_add(theora_comment *tc, char *comment);

/**
 * Add a comment to an initialized theora_comment structure.
 * \param tc A previously initialized theora comment structure
 * \param tag A null-terminated string containing the tag 
 *            associated with the comment.
 * \param value The corresponding value as a null-terminated string
 *
 * Neither theora_comment_add() nor theora_comment_add_tag() support
 * comments containing null values, although the bitstream format
 * supports this. To add such comments you will need to manipulate
 * the theora_comment structure directly.
 **/
extern void theora_comment_add_tag(theora_comment *tc,
                                       char *tag, char *value);

/**
 * Look up a comment value by tag.
 * \param tc Tn initialized theora_comment structure
 * \param tag The tag to look up
 * \param count The instance of the tag. The same tag can appear multiple
 *              times, each with a distinct and ordered value, so an index
 *              is required to retrieve them all.
 * \returns A pointer to the queried tag's value
 * \retval NULL No matching tag is found
 *
 * \note Use theora_comment_query_count() to get the legal range for the
 * count parameter.
 **/

extern char *theora_comment_query(theora_comment *tc, char *tag, int count);

/** Look up the number of instances of a tag.
 *  \param tc An initialized theora_comment structure
 *  \param tag The tag to look up
 *  \returns The number on instances of a particular tag.
 * 
 *  Call this first when querying for a specific tag and then interate
 *  over the number of instances with separate calls to 
 *  theora_comment_query() to retrieve all instances in order.
 **/
extern int   theora_comment_query_count(theora_comment *tc, char *tag);

/**
 * Clear an allocated theora_comment struct so that it can be freed.
 * \param tc An allocated theora_comment structure.
 **/
extern void  theora_comment_clear(theora_comment *tc);

/**Encoder control function.
 * This is used to provide advanced control the encoding process.
 * \param th     A #theora_state handle.
 * \param req    The control code to process.
 *                See \ref encctlcodes_old "the list of available 
 *			control codes" for details.
 * \param buf    The parameters for this control code.
 * \param buf_sz The size of the parameter buffer.*/
extern int theora_control(theora_state *th,int req,void *buf,size_t buf_sz);

/* @} */ /* end oldfuncs doxygen group */

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* _O_THEORA_H_ */
