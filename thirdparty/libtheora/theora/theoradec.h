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
 * The <tt>libtheoradec</tt> C decoding API.*/

#if !defined(OGG_THEORA_THEORADEC_HEADER)
# define OGG_THEORA_THEORADEC_HEADER (1)
# include <stddef.h>
# include <ogg/ogg.h>
# include "codec.h"

#if defined(__cplusplus)
extern "C" {
#endif



/**\name th_decode_ctl() codes
 * \anchor decctlcodes
 * These are the available request codes for th_decode_ctl().
 * By convention, these are odd, to distinguish them from the
 *  \ref encctlcodes "encoder control codes".
 * Keep any experimental or vendor-specific values above \c 0x8000.*/
/*@{*/
/**Gets the maximum post-processing level.
 * The decoder supports a post-processing filter that can improve
 * the appearance of the decoded images. This returns the highest
 * level setting for this post-processor, corresponding to maximum
 * improvement and computational expense.
 *
 * \param[out] _buf int: The maximum post-processing level.
 * \retval TH_EFAULT  \a _dec_ctx or \a _buf is <tt>NULL</tt>.
 * \retval TH_EINVAL  \a _buf_sz is not <tt>sizeof(int)</tt>.
 * \retval TH_EIMPL   Not supported by this implementation.*/
#define TH_DECCTL_GET_PPLEVEL_MAX (1)
/**Sets the post-processing level.
 * By default, post-processing is disabled.
 *
 * Sets the level of post-processing to use when decoding the
 * compressed stream. This must be a value between zero (off)
 * and the maximum returned by TH_DECCTL_GET_PPLEVEL_MAX.
 *
 * \param[in] _buf int: The new post-processing level.
 *                      0 to disable; larger values use more CPU.
 * \retval TH_EFAULT  \a _dec_ctx or \a _buf is <tt>NULL</tt>.
 * \retval TH_EINVAL  \a _buf_sz is not <tt>sizeof(int)</tt>, or the
 *                     post-processing level is out of bounds.
 *                    The maximum post-processing level may be
 *                     implementation-specific, and can be obtained via
 *                     #TH_DECCTL_GET_PPLEVEL_MAX.
 * \retval TH_EIMPL   Not supported by this implementation.*/
#define TH_DECCTL_SET_PPLEVEL (3)
/**Sets the granule position.
 * Call this after a seek, before decoding the first frame, to ensure that the
 *  proper granule position is returned for all subsequent frames.
 * If you track timestamps yourself and do not use the granule position
 *  returned by the decoder, then you need not call this function.
 *
 * \param[in] _buf <tt>ogg_int64_t</tt>: The granule position of the next
 *                  frame.
 * \retval TH_EFAULT  \a _dec_ctx or \a _buf is <tt>NULL</tt>.
 * \retval TH_EINVAL  \a _buf_sz is not <tt>sizeof(ogg_int64_t)</tt>, or the
 *                     granule position is negative.*/
#define TH_DECCTL_SET_GRANPOS (5)
/**Sets the striped decode callback function.
 * If set, this function will be called as each piece of a frame is fully
 *  decoded in th_decode_packetin().
 * You can pass in a #th_stripe_callback with
 *  th_stripe_callback#stripe_decoded set to <tt>NULL</tt> to disable the
 *  callbacks at any point.
 * Enabling striped decode does not prevent you from calling
 *  th_decode_ycbcr_out() after the frame is fully decoded.
 *
 * \param[in]  _buf #th_stripe_callback: The callback parameters.
 * \retval TH_EFAULT  \a _dec_ctx or \a _buf is <tt>NULL</tt>.
 * \retval TH_EINVAL  \a _buf_sz is not
 *                     <tt>sizeof(th_stripe_callback)</tt>.*/
#define TH_DECCTL_SET_STRIPE_CB (7)

/**Sets the macroblock display mode. Set to 0 to disable displaying
 * macroblocks.*/
#define TH_DECCTL_SET_TELEMETRY_MBMODE (9)
/**Sets the motion vector display mode. Set to 0 to disable displaying motion
 * vectors.*/
#define TH_DECCTL_SET_TELEMETRY_MV (11)
/**Sets the adaptive quantization display mode. Set to 0 to disable displaying
 * adaptive quantization. */
#define TH_DECCTL_SET_TELEMETRY_QI (13)
/**Sets the bitstream breakdown visualization mode. Set to 0 to disable
 * displaying bitstream breakdown.*/
#define TH_DECCTL_SET_TELEMETRY_BITS (15)
/*@}*/



/**A callback function for striped decode.
 * This is a function pointer to an application-provided function that will be
 *  called each time a section of the image is fully decoded in
 *  th_decode_packetin().
 * This allows the application to process the section immediately, while it is
 *  still in cache.
 * Note that the frame is decoded bottom to top, so \a _yfrag0 will steadily
 *  decrease with each call until it reaches 0, at which point the full frame
 *  is decoded.
 * The number of fragment rows made available in each call depends on the pixel
 *  format and the number of post-processing filters enabled, and may not even
 *  be constant for the entire frame.
 * If a non-<tt>NULL</tt> \a _granpos pointer is passed to
 *  th_decode_packetin(), the granule position for the frame will be stored
 *  in it before the first callback is made.
 * If an entire frame is dropped (a 0-byte packet), then no callbacks will be
 *  made at all for that frame.
 * \param _ctx       An application-provided context pointer.
 * \param _buf       The image buffer for the decoded frame.
 * \param _yfrag0    The Y coordinate of the first row of 8x8 fragments
 *                    decoded.
 *                   Multiply this by 8 to obtain the pixel row number in the
 *                    luma plane.
 *                   If the chroma planes are subsampled in the Y direction,
 *                    this will always be divisible by two.
 * \param _yfrag_end The Y coordinate of the first row of 8x8 fragments past
 *                    the newly decoded section.
 *                   If the chroma planes are subsampled in the Y direction,
 *                    this will always be divisible by two.
 *                   I.e., this section contains fragment rows
 *                    <tt>\a _yfrag0 ...\a _yfrag_end -1</tt>.*/
typedef void (*th_stripe_decoded_func)(void *_ctx,th_ycbcr_buffer _buf,
 int _yfrag0,int _yfrag_end);

/**The striped decode callback data to pass to #TH_DECCTL_SET_STRIPE_CB.*/
typedef struct{
  /**An application-provided context pointer.
   * This will be passed back verbatim to the application.*/
  void                   *ctx;
  /**The callback function pointer.*/
  th_stripe_decoded_func  stripe_decoded;
}th_stripe_callback;



/**\name Decoder state
   The following data structures are opaque, and their contents are not
    publicly defined by this API.
   Referring to their internals directly is unsupported, and may break without
    warning.*/
/*@{*/
/**The decoder context.*/
typedef struct th_dec_ctx    th_dec_ctx;
/**Setup information.
   This contains auxiliary information (Huffman tables and quantization
    parameters) decoded from the setup header by th_decode_headerin() to be
    passed to th_decode_alloc().
   It can be re-used to initialize any number of decoders, and can be freed
    via th_setup_free() at any time.*/
typedef struct th_setup_info th_setup_info;
/*@}*/



/**\defgroup decfuncs Functions for Decoding*/
/*@{*/
/**\name Functions for decoding
 * You must link to <tt>libtheoradec</tt> if you use any of the
 * functions in this section.
 *
 * The functions are listed in the order they are used in a typical decode.
 * The basic steps are:
 * - Parse the header packets by repeatedly calling th_decode_headerin().
 * - Allocate a #th_dec_ctx handle with th_decode_alloc().
 * - Call th_setup_free() to free any memory used for codec setup
 *    information.
 * - Perform any additional decoder configuration with th_decode_ctl().
 * - For each video data packet:
 *   - Submit the packet to the decoder via th_decode_packetin().
 *   - Retrieve the uncompressed video data via th_decode_ycbcr_out().
 * - Call th_decode_free() to release all decoder memory.*/
/*@{*/
/**Decodes the header packets of a Theora stream.
 * This should be called on the initial packets of the stream, in succession,
 *  until it returns <tt>0</tt>, indicating that all headers have been
 *  processed, or an error is encountered.
 * At least three header packets are required, and additional optional header
 *  packets may follow.
 * This can be used on the first packet of any logical stream to determine if
 *  that stream is a Theora stream.
 * \param _info  A #th_info structure to fill in.
 *               This must have been previously initialized with
 *                th_info_init().
 *               The application may immediately begin using the contents of
 *                this structure after the first header is decoded, though it
 *                must continue to be passed in on all subsequent calls.
 * \param _tc    A #th_comment structure to fill in.
 *               The application may immediately begin using the contents of
 *                this structure after the second header is decoded, though it
 *                must continue to be passed in on all subsequent calls.
 * \param _setup Returns a pointer to additional, private setup information
 *                needed by the decoder.
 *               The contents of this pointer must be initialized to
 *                <tt>NULL</tt> on the first call, and the returned value must
 *                continue to be passed in on all subsequent calls.
 * \param _op    An <tt>ogg_packet</tt> structure which contains one of the
 *                initial packets of an Ogg logical stream.
 * \return A positive value indicates that a Theora header was successfully
 *          processed.
 * \retval 0             The first video data packet was encountered after all
 *                        required header packets were parsed.
 *                       The packet just passed in on this call should be saved
 *                        and fed to th_decode_packetin() to begin decoding
 *                        video data.
 * \retval TH_EFAULT     One of \a _info, \a _tc, or \a _setup was
 *                        <tt>NULL</tt>.
 * \retval TH_EBADHEADER \a _op was <tt>NULL</tt>, the packet was not the next
 *                        header packet in the expected sequence, or the format
 *                        of the header data was invalid.
 * \retval TH_EVERSION   The packet data was a Theora info header, but for a
 *                        bitstream version not decodable with this version of
 *                        <tt>libtheoradec</tt>.
 * \retval TH_ENOTFORMAT The packet was not a Theora header.
 */
extern int th_decode_headerin(th_info *_info,th_comment *_tc,
 th_setup_info **_setup,ogg_packet *_op);
/**Allocates a decoder instance.
 *
 * <b>Security Warning:</b> The Theora format supports very large frame sizes,
 *  potentially even larger than the address space of a 32-bit machine, and
 *  creating a decoder context allocates the space for several frames of data.
 * If the allocation fails here, your program will crash, possibly at some
 *  future point because the OS kernel returned a valid memory range and will
 *  only fail when it tries to map the pages in it the first time they are
 *  used.
 * Even if it succeeds, you may experience a denial of service if the frame
 *  size is large enough to cause excessive paging.
 * If you are integrating libtheora in a larger application where such things
 *  are undesirable, it is highly recommended that you check the frame size in
 *  \a _info before calling this function and refuse to decode streams where it
 *  is larger than some reasonable maximum.
 * libtheora will not check this for you, because there may be machines that
 *  can handle such streams and applications that wish to.
 * \param _info  A #th_info struct filled via th_decode_headerin().
 * \param _setup A #th_setup_info handle returned via
 *                th_decode_headerin().
 * \return The initialized #th_dec_ctx handle.
 * \retval NULL If the decoding parameters were invalid.*/
extern th_dec_ctx *th_decode_alloc(const th_info *_info,
 const th_setup_info *_setup);
/**Releases all storage used for the decoder setup information.
 * This should be called after you no longer want to create any decoders for
 *  a stream whose headers you have parsed with th_decode_headerin().
 * \param _setup The setup information to free.
 *               This can safely be <tt>NULL</tt>.*/
extern void th_setup_free(th_setup_info *_setup);
/**Decoder control function.
 * This is used to provide advanced control of the decoding process.
 * \param _dec    A #th_dec_ctx handle.
 * \param _req    The control code to process.
 *                See \ref decctlcodes "the list of available control codes"
 *                 for details.
 * \param _buf    The parameters for this control code.
 * \param _buf_sz The size of the parameter buffer.
 * \return Possible return values depend on the control code used.
 *          See \ref decctlcodes "the list of control codes" for
 *          specific values. Generally 0 indicates success.*/
extern int th_decode_ctl(th_dec_ctx *_dec,int _req,void *_buf,
 size_t _buf_sz);
/**Submits a packet containing encoded video data to the decoder.
 * \param _dec     A #th_dec_ctx handle.
 * \param _op      An <tt>ogg_packet</tt> containing encoded video data.
 * \param _granpos Returns the granule position of the decoded packet.
 *                 If non-<tt>NULL</tt>, the granule position for this specific
 *                  packet is stored in this location.
 *                 This is computed incrementally from previously decoded
 *                  packets.
 *                 After a seek, the correct granule position must be set via
 *                  #TH_DECCTL_SET_GRANPOS for this to work properly.
 * \retval 0             Success.
 *                       A new decoded frame can be retrieved by calling
 *                        th_decode_ycbcr_out().
 * \retval TH_DUPFRAME   The packet represented a dropped frame (either a
 *                        0-byte frame or an INTER frame with no coded blocks).
 *                       The player can skip the call to th_decode_ycbcr_out(),
 *                        as the contents of the decoded frame buffer have not
 *                        changed.
 * \retval TH_EFAULT     \a _dec or \a _op was <tt>NULL</tt>.
 * \retval TH_EBADPACKET \a _op does not contain encoded video data.
 * \retval TH_EIMPL      The video data uses bitstream features which this
 *                        library does not support.*/
extern int th_decode_packetin(th_dec_ctx *_dec,const ogg_packet *_op,
 ogg_int64_t *_granpos);
/**Outputs the next available frame of decoded Y'CbCr data.
 * If a striped decode callback has been set with #TH_DECCTL_SET_STRIPE_CB,
 *  then the application does not need to call this function.
 * \param _dec   A #th_dec_ctx handle.
 * \param _ycbcr A video buffer structure to fill in.
 *               <tt>libtheoradec</tt> will fill in all the members of this
 *                structure, including the pointers to the uncompressed video
 *                data.
 *               The memory for this video data is owned by
 *                <tt>libtheoradec</tt>.
 *               It may be freed or overwritten without notification when
 *                subsequent frames are decoded.
 * \retval 0 Success
 * \retval TH_EFAULT     \a _dec or \a _ycbcr was <tt>NULL</tt>.
 */
extern int th_decode_ycbcr_out(th_dec_ctx *_dec,
 th_ycbcr_buffer _ycbcr);
/**Frees an allocated decoder instance.
 * \param _dec A #th_dec_ctx handle.*/
extern void th_decode_free(th_dec_ctx *_dec);
/*@}*/
/*@}*/



#if defined(__cplusplus)
}
#endif

#endif /* OGG_THEORA_THEORADEC_HEADER */
