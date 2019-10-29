/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */
#ifndef VPX_VPX_ENCODER_H_
#define VPX_VPX_ENCODER_H_

/*!\defgroup encoder Encoder Algorithm Interface
 * \ingroup codec
 * This abstraction allows applications using this encoder to easily support
 * multiple video formats with minimal code duplication. This section describes
 * the interface common to all encoders.
 * @{
 */

/*!\file
 * \brief Describes the encoder algorithm interface to applications.
 *
 * This file describes the interface between an application and a
 * video encoder algorithm.
 *
 */
#ifdef __cplusplus
extern "C" {
#endif

#include "./vpx_codec.h"

  /*! Temporal Scalability: Maximum length of the sequence defining frame
   * layer membership
   */
#define VPX_TS_MAX_PERIODICITY 16

  /*! Temporal Scalability: Maximum number of coding layers */
#define VPX_TS_MAX_LAYERS       5

  /*!\deprecated Use #VPX_TS_MAX_PERIODICITY instead. */
#define MAX_PERIODICITY VPX_TS_MAX_PERIODICITY

/*! Temporal+Spatial Scalability: Maximum number of coding layers */
#define VPX_MAX_LAYERS  12  // 3 temporal + 4 spatial layers are allowed.

/*!\deprecated Use #VPX_MAX_LAYERS instead. */
#define MAX_LAYERS    VPX_MAX_LAYERS  // 3 temporal + 4 spatial layers allowed.

/*! Spatial Scalability: Maximum number of coding layers */
#define VPX_SS_MAX_LAYERS       5

/*! Spatial Scalability: Default number of coding layers */
#define VPX_SS_DEFAULT_LAYERS       1

  /*!\brief Current ABI version number
   *
   * \internal
   * If this file is altered in any way that changes the ABI, this value
   * must be bumped.  Examples include, but are not limited to, changing
   * types, removing or reassigning enums, adding/removing/rearranging
   * fields to structures
   */
#define VPX_ENCODER_ABI_VERSION (5 + VPX_CODEC_ABI_VERSION) /**<\hideinitializer*/


  /*! \brief Encoder capabilities bitfield
   *
   *  Each encoder advertises the capabilities it supports as part of its
   *  ::vpx_codec_iface_t interface structure. Capabilities are extra
   *  interfaces or functionality, and are not required to be supported
   *  by an encoder.
   *
   *  The available flags are specified by VPX_CODEC_CAP_* defines.
   */
#define VPX_CODEC_CAP_PSNR  0x10000 /**< Can issue PSNR packets */

  /*! Can output one partition at a time. Each partition is returned in its
   *  own VPX_CODEC_CX_FRAME_PKT, with the FRAME_IS_FRAGMENT flag set for
   *  every partition but the last. In this mode all frames are always
   *  returned partition by partition.
   */
#define VPX_CODEC_CAP_OUTPUT_PARTITION  0x20000

/*! Can support input images at greater than 8 bitdepth.
 */
#define VPX_CODEC_CAP_HIGHBITDEPTH  0x40000

  /*! \brief Initialization-time Feature Enabling
   *
   *  Certain codec features must be known at initialization time, to allow
   *  for proper memory allocation.
   *
   *  The available flags are specified by VPX_CODEC_USE_* defines.
   */
#define VPX_CODEC_USE_PSNR  0x10000 /**< Calculate PSNR on each frame */
#define VPX_CODEC_USE_OUTPUT_PARTITION  0x20000 /**< Make the encoder output one
  partition at a time. */
#define VPX_CODEC_USE_HIGHBITDEPTH 0x40000 /**< Use high bitdepth */


  /*!\brief Generic fixed size buffer structure
   *
   * This structure is able to hold a reference to any fixed size buffer.
   */
  typedef struct vpx_fixed_buf {
    void          *buf; /**< Pointer to the data */
    size_t         sz;  /**< Length of the buffer, in chars */
  } vpx_fixed_buf_t; /**< alias for struct vpx_fixed_buf */


  /*!\brief Time Stamp Type
   *
   * An integer, which when multiplied by the stream's time base, provides
   * the absolute time of a sample.
   */
  typedef int64_t vpx_codec_pts_t;


  /*!\brief Compressed Frame Flags
   *
   * This type represents a bitfield containing information about a compressed
   * frame that may be useful to an application. The most significant 16 bits
   * can be used by an algorithm to provide additional detail, for example to
   * support frame types that are codec specific (MPEG-1 D-frames for example)
   */
  typedef uint32_t vpx_codec_frame_flags_t;
#define VPX_FRAME_IS_KEY       0x1 /**< frame is the start of a GOP */
#define VPX_FRAME_IS_DROPPABLE 0x2 /**< frame can be dropped without affecting
  the stream (no future frame depends on
              this one) */
#define VPX_FRAME_IS_INVISIBLE 0x4 /**< frame should be decoded but will not
  be shown */
#define VPX_FRAME_IS_FRAGMENT  0x8 /**< this is a fragment of the encoded
  frame */

  /*!\brief Error Resilient flags
   *
   * These flags define which error resilient features to enable in the
   * encoder. The flags are specified through the
   * vpx_codec_enc_cfg::g_error_resilient variable.
   */
  typedef uint32_t vpx_codec_er_flags_t;
#define VPX_ERROR_RESILIENT_DEFAULT     0x1 /**< Improve resiliency against
  losses of whole frames */
#define VPX_ERROR_RESILIENT_PARTITIONS  0x2 /**< The frame partitions are
  independently decodable by the
  bool decoder, meaning that
  partitions can be decoded even
  though earlier partitions have
  been lost. Note that intra
  prediction is still done over
  the partition boundary. */

  /*!\brief Encoder output packet variants
   *
   * This enumeration lists the different kinds of data packets that can be
   * returned by calls to vpx_codec_get_cx_data(). Algorithms \ref MAY
   * extend this list to provide additional functionality.
   */
  enum vpx_codec_cx_pkt_kind {
    VPX_CODEC_CX_FRAME_PKT,    /**< Compressed video frame */
    VPX_CODEC_STATS_PKT,       /**< Two-pass statistics for this frame */
    VPX_CODEC_FPMB_STATS_PKT,  /**< first pass mb statistics for this frame */
    VPX_CODEC_PSNR_PKT,        /**< PSNR statistics for this frame */
    // Spatial SVC is still experimental and may be removed before the next ABI
    // bump.
#if VPX_ENCODER_ABI_VERSION > (5 + VPX_CODEC_ABI_VERSION)
    VPX_CODEC_SPATIAL_SVC_LAYER_SIZES, /**< Sizes for each layer in this frame*/
    VPX_CODEC_SPATIAL_SVC_LAYER_PSNR, /**< PSNR for each layer in this frame*/
#endif
    VPX_CODEC_CUSTOM_PKT = 256 /**< Algorithm extensions  */
  };


  /*!\brief Encoder output packet
   *
   * This structure contains the different kinds of output data the encoder
   * may produce while compressing a frame.
   */
  typedef struct vpx_codec_cx_pkt {
    enum vpx_codec_cx_pkt_kind  kind; /**< packet variant */
    union {
      struct {
        void                    *buf;      /**< compressed data buffer */
        size_t                   sz;       /**< length of compressed data */
        vpx_codec_pts_t          pts;      /**< time stamp to show frame
                                                    (in timebase units) */
        unsigned long            duration; /**< duration to show frame
                                                    (in timebase units) */
        vpx_codec_frame_flags_t  flags;    /**< flags for this frame */
        int                      partition_id; /**< the partition id
                                              defines the decoding order
                                              of the partitions. Only
                                              applicable when "output partition"
                                              mode is enabled. First partition
                                              has id 0.*/

      } frame;  /**< data for compressed frame packet */
      vpx_fixed_buf_t twopass_stats;  /**< data for two-pass packet */
      vpx_fixed_buf_t firstpass_mb_stats; /**< first pass mb packet */
      struct vpx_psnr_pkt {
        unsigned int samples[4];  /**< Number of samples, total/y/u/v */
        uint64_t     sse[4];      /**< sum squared error, total/y/u/v */
        double       psnr[4];     /**< PSNR, total/y/u/v */
      } psnr;                       /**< data for PSNR packet */
      vpx_fixed_buf_t raw;     /**< data for arbitrary packets */
      // Spatial SVC is still experimental and may be removed before the next
      // ABI bump.
#if VPX_ENCODER_ABI_VERSION > (5 + VPX_CODEC_ABI_VERSION)
      size_t layer_sizes[VPX_SS_MAX_LAYERS];
      struct vpx_psnr_pkt layer_psnr[VPX_SS_MAX_LAYERS];
#endif

      /* This packet size is fixed to allow codecs to extend this
       * interface without having to manage storage for raw packets,
       * i.e., if it's smaller than 128 bytes, you can store in the
       * packet list directly.
       */
      char pad[128 - sizeof(enum vpx_codec_cx_pkt_kind)]; /**< fixed sz */
    } data; /**< packet data */
  } vpx_codec_cx_pkt_t; /**< alias for struct vpx_codec_cx_pkt */


  /*!\brief Encoder return output buffer callback
   *
   * This callback function, when registered, returns with packets when each
   * spatial layer is encoded.
   */
  // putting the definitions here for now. (agrange: find if there
  // is a better place for this)
  typedef void (* vpx_codec_enc_output_cx_pkt_cb_fn_t)(vpx_codec_cx_pkt_t *pkt,
                                                       void *user_data);

  /*!\brief Callback function pointer / user data pair storage */
  typedef struct vpx_codec_enc_output_cx_cb_pair {
    vpx_codec_enc_output_cx_pkt_cb_fn_t output_cx_pkt; /**< Callback function */
    void                            *user_priv; /**< Pointer to private data */
  } vpx_codec_priv_output_cx_pkt_cb_pair_t;

  /*!\brief Rational Number
   *
   * This structure holds a fractional value.
   */
  typedef struct vpx_rational {
    int num; /**< fraction numerator */
    int den; /**< fraction denominator */
  } vpx_rational_t; /**< alias for struct vpx_rational */


  /*!\brief Multi-pass Encoding Pass */
  enum vpx_enc_pass {
    VPX_RC_ONE_PASS,   /**< Single pass mode */
    VPX_RC_FIRST_PASS, /**< First pass of multi-pass mode */
    VPX_RC_LAST_PASS   /**< Final pass of multi-pass mode */
  };


  /*!\brief Rate control mode */
  enum vpx_rc_mode {
    VPX_VBR,  /**< Variable Bit Rate (VBR) mode */
    VPX_CBR,  /**< Constant Bit Rate (CBR) mode */
    VPX_CQ,   /**< Constrained Quality (CQ)  mode */
    VPX_Q,    /**< Constant Quality (Q) mode */
  };


  /*!\brief Keyframe placement mode.
   *
   * This enumeration determines whether keyframes are placed automatically by
   * the encoder or whether this behavior is disabled. Older releases of this
   * SDK were implemented such that VPX_KF_FIXED meant keyframes were disabled.
   * This name is confusing for this behavior, so the new symbols to be used
   * are VPX_KF_AUTO and VPX_KF_DISABLED.
   */
  enum vpx_kf_mode {
    VPX_KF_FIXED, /**< deprecated, implies VPX_KF_DISABLED */
    VPX_KF_AUTO,  /**< Encoder determines optimal placement automatically */
    VPX_KF_DISABLED = 0 /**< Encoder does not place keyframes. */
  };


  /*!\brief Encoded Frame Flags
   *
   * This type indicates a bitfield to be passed to vpx_codec_encode(), defining
   * per-frame boolean values. By convention, bits common to all codecs will be
   * named VPX_EFLAG_*, and bits specific to an algorithm will be named
   * /algo/_eflag_*. The lower order 16 bits are reserved for common use.
   */
  typedef long vpx_enc_frame_flags_t;
#define VPX_EFLAG_FORCE_KF (1<<0)  /**< Force this frame to be a keyframe */


  /*!\brief Encoder configuration structure
   *
   * This structure contains the encoder settings that have common representations
   * across all codecs. This doesn't imply that all codecs support all features,
   * however.
   */
  typedef struct vpx_codec_enc_cfg {
    /*
     * generic settings (g)
     */

    /*!\brief Algorithm specific "usage" value
     *
     * Algorithms may define multiple values for usage, which may convey the
     * intent of how the application intends to use the stream. If this value
     * is non-zero, consult the documentation for the codec to determine its
     * meaning.
     */
    unsigned int           g_usage;


    /*!\brief Maximum number of threads to use
     *
     * For multi-threaded implementations, use no more than this number of
     * threads. The codec may use fewer threads than allowed. The value
     * 0 is equivalent to the value 1.
     */
    unsigned int           g_threads;


    /*!\brief Bitstream profile to use
     *
     * Some codecs support a notion of multiple bitstream profiles. Typically
     * this maps to a set of features that are turned on or off. Often the
     * profile to use is determined by the features of the intended decoder.
     * Consult the documentation for the codec to determine the valid values
     * for this parameter, or set to zero for a sane default.
     */
    unsigned int           g_profile;  /**< profile of bitstream to use */



    /*!\brief Width of the frame
     *
     * This value identifies the presentation resolution of the frame,
     * in pixels. Note that the frames passed as input to the encoder must
     * have this resolution. Frames will be presented by the decoder in this
     * resolution, independent of any spatial resampling the encoder may do.
     */
    unsigned int           g_w;


    /*!\brief Height of the frame
     *
     * This value identifies the presentation resolution of the frame,
     * in pixels. Note that the frames passed as input to the encoder must
     * have this resolution. Frames will be presented by the decoder in this
     * resolution, independent of any spatial resampling the encoder may do.
     */
    unsigned int           g_h;

    /*!\brief Bit-depth of the codec
     *
     * This value identifies the bit_depth of the codec,
     * Only certain bit-depths are supported as identified in the
     * vpx_bit_depth_t enum.
     */
    vpx_bit_depth_t        g_bit_depth;

    /*!\brief Bit-depth of the input frames
     *
     * This value identifies the bit_depth of the input frames in bits.
     * Note that the frames passed as input to the encoder must have
     * this bit-depth.
     */
    unsigned int           g_input_bit_depth;

    /*!\brief Stream timebase units
     *
     * Indicates the smallest interval of time, in seconds, used by the stream.
     * For fixed frame rate material, or variable frame rate material where
     * frames are timed at a multiple of a given clock (ex: video capture),
     * the \ref RECOMMENDED method is to set the timebase to the reciprocal
     * of the frame rate (ex: 1001/30000 for 29.970 Hz NTSC). This allows the
     * pts to correspond to the frame number, which can be handy. For
     * re-encoding video from containers with absolute time timestamps, the
     * \ref RECOMMENDED method is to set the timebase to that of the parent
     * container or multimedia framework (ex: 1/1000 for ms, as in FLV).
     */
    struct vpx_rational    g_timebase;


    /*!\brief Enable error resilient modes.
     *
     * The error resilient bitfield indicates to the encoder which features
     * it should enable to take measures for streaming over lossy or noisy
     * links.
     */
    vpx_codec_er_flags_t   g_error_resilient;


    /*!\brief Multi-pass Encoding Mode
     *
     * This value should be set to the current phase for multi-pass encoding.
     * For single pass, set to #VPX_RC_ONE_PASS.
     */
    enum vpx_enc_pass      g_pass;


    /*!\brief Allow lagged encoding
     *
     * If set, this value allows the encoder to consume a number of input
     * frames before producing output frames. This allows the encoder to
     * base decisions for the current frame on future frames. This does
     * increase the latency of the encoding pipeline, so it is not appropriate
     * in all situations (ex: realtime encoding).
     *
     * Note that this is a maximum value -- the encoder may produce frames
     * sooner than the given limit. Set this value to 0 to disable this
     * feature.
     */
    unsigned int           g_lag_in_frames;


    /*
     * rate control settings (rc)
     */

    /*!\brief Temporal resampling configuration, if supported by the codec.
     *
     * Temporal resampling allows the codec to "drop" frames as a strategy to
     * meet its target data rate. This can cause temporal discontinuities in
     * the encoded video, which may appear as stuttering during playback. This
     * trade-off is often acceptable, but for many applications is not. It can
     * be disabled in these cases.
     *
     * Note that not all codecs support this feature. All vpx VPx codecs do.
     * For other codecs, consult the documentation for that algorithm.
     *
     * This threshold is described as a percentage of the target data buffer.
     * When the data buffer falls below this percentage of fullness, a
     * dropped frame is indicated. Set the threshold to zero (0) to disable
     * this feature.
     */
    unsigned int           rc_dropframe_thresh;


    /*!\brief Enable/disable spatial resampling, if supported by the codec.
     *
     * Spatial resampling allows the codec to compress a lower resolution
     * version of the frame, which is then upscaled by the encoder to the
     * correct presentation resolution. This increases visual quality at
     * low data rates, at the expense of CPU time on the encoder/decoder.
     */
    unsigned int           rc_resize_allowed;

    /*!\brief Internal coded frame width.
     *
     * If spatial resampling is enabled this specifies the width of the
     * encoded frame.
     */
    unsigned int           rc_scaled_width;

    /*!\brief Internal coded frame height.
     *
     * If spatial resampling is enabled this specifies the height of the
     * encoded frame.
     */
    unsigned int           rc_scaled_height;

    /*!\brief Spatial resampling up watermark.
     *
     * This threshold is described as a percentage of the target data buffer.
     * When the data buffer rises above this percentage of fullness, the
     * encoder will step up to a higher resolution version of the frame.
     */
    unsigned int           rc_resize_up_thresh;


    /*!\brief Spatial resampling down watermark.
     *
     * This threshold is described as a percentage of the target data buffer.
     * When the data buffer falls below this percentage of fullness, the
     * encoder will step down to a lower resolution version of the frame.
     */
    unsigned int           rc_resize_down_thresh;


    /*!\brief Rate control algorithm to use.
     *
     * Indicates whether the end usage of this stream is to be streamed over
     * a bandwidth constrained link, indicating that Constant Bit Rate (CBR)
     * mode should be used, or whether it will be played back on a high
     * bandwidth link, as from a local disk, where higher variations in
     * bitrate are acceptable.
     */
    enum vpx_rc_mode       rc_end_usage;


    /*!\brief Two-pass stats buffer.
     *
     * A buffer containing all of the stats packets produced in the first
     * pass, concatenated.
     */
    vpx_fixed_buf_t   rc_twopass_stats_in;

    /*!\brief first pass mb stats buffer.
     *
     * A buffer containing all of the first pass mb stats packets produced
     * in the first pass, concatenated.
     */
    vpx_fixed_buf_t   rc_firstpass_mb_stats_in;

    /*!\brief Target data rate
     *
     * Target bandwidth to use for this stream, in kilobits per second.
     */
    unsigned int           rc_target_bitrate;


    /*
     * quantizer settings
     */


    /*!\brief Minimum (Best Quality) Quantizer
     *
     * The quantizer is the most direct control over the quality of the
     * encoded image. The range of valid values for the quantizer is codec
     * specific. Consult the documentation for the codec to determine the
     * values to use. To determine the range programmatically, call
     * vpx_codec_enc_config_default() with a usage value of 0.
     */
    unsigned int           rc_min_quantizer;


    /*!\brief Maximum (Worst Quality) Quantizer
     *
     * The quantizer is the most direct control over the quality of the
     * encoded image. The range of valid values for the quantizer is codec
     * specific. Consult the documentation for the codec to determine the
     * values to use. To determine the range programmatically, call
     * vpx_codec_enc_config_default() with a usage value of 0.
     */
    unsigned int           rc_max_quantizer;


    /*
     * bitrate tolerance
     */


    /*!\brief Rate control adaptation undershoot control
     *
     * This value, expressed as a percentage of the target bitrate,
     * controls the maximum allowed adaptation speed of the codec.
     * This factor controls the maximum amount of bits that can
     * be subtracted from the target bitrate in order to compensate
     * for prior overshoot.
     *
     * Valid values in the range 0-1000.
     */
    unsigned int           rc_undershoot_pct;


    /*!\brief Rate control adaptation overshoot control
     *
     * This value, expressed as a percentage of the target bitrate,
     * controls the maximum allowed adaptation speed of the codec.
     * This factor controls the maximum amount of bits that can
     * be added to the target bitrate in order to compensate for
     * prior undershoot.
     *
     * Valid values in the range 0-1000.
     */
    unsigned int           rc_overshoot_pct;


    /*
     * decoder buffer model parameters
     */


    /*!\brief Decoder Buffer Size
     *
     * This value indicates the amount of data that may be buffered by the
     * decoding application. Note that this value is expressed in units of
     * time (milliseconds). For example, a value of 5000 indicates that the
     * client will buffer (at least) 5000ms worth of encoded data. Use the
     * target bitrate (#rc_target_bitrate) to convert to bits/bytes, if
     * necessary.
     */
    unsigned int           rc_buf_sz;


    /*!\brief Decoder Buffer Initial Size
     *
     * This value indicates the amount of data that will be buffered by the
     * decoding application prior to beginning playback. This value is
     * expressed in units of time (milliseconds). Use the target bitrate
     * (#rc_target_bitrate) to convert to bits/bytes, if necessary.
     */
    unsigned int           rc_buf_initial_sz;


    /*!\brief Decoder Buffer Optimal Size
     *
     * This value indicates the amount of data that the encoder should try
     * to maintain in the decoder's buffer. This value is expressed in units
     * of time (milliseconds). Use the target bitrate (#rc_target_bitrate)
     * to convert to bits/bytes, if necessary.
     */
    unsigned int           rc_buf_optimal_sz;


    /*
     * 2 pass rate control parameters
     */


    /*!\brief Two-pass mode CBR/VBR bias
     *
     * Bias, expressed on a scale of 0 to 100, for determining target size
     * for the current frame. The value 0 indicates the optimal CBR mode
     * value should be used. The value 100 indicates the optimal VBR mode
     * value should be used. Values in between indicate which way the
     * encoder should "lean."
     */
    unsigned int           rc_2pass_vbr_bias_pct;       /**< RC mode bias between CBR and VBR(0-100: 0->CBR, 100->VBR)   */


    /*!\brief Two-pass mode per-GOP minimum bitrate
     *
     * This value, expressed as a percentage of the target bitrate, indicates
     * the minimum bitrate to be used for a single GOP (aka "section")
     */
    unsigned int           rc_2pass_vbr_minsection_pct;


    /*!\brief Two-pass mode per-GOP maximum bitrate
     *
     * This value, expressed as a percentage of the target bitrate, indicates
     * the maximum bitrate to be used for a single GOP (aka "section")
     */
    unsigned int           rc_2pass_vbr_maxsection_pct;


    /*
     * keyframing settings (kf)
     */

    /*!\brief Keyframe placement mode
     *
     * This value indicates whether the encoder should place keyframes at a
     * fixed interval, or determine the optimal placement automatically
     * (as governed by the #kf_min_dist and #kf_max_dist parameters)
     */
    enum vpx_kf_mode       kf_mode;


    /*!\brief Keyframe minimum interval
     *
     * This value, expressed as a number of frames, prevents the encoder from
     * placing a keyframe nearer than kf_min_dist to the previous keyframe. At
     * least kf_min_dist frames non-keyframes will be coded before the next
     * keyframe. Set kf_min_dist equal to kf_max_dist for a fixed interval.
     */
    unsigned int           kf_min_dist;


    /*!\brief Keyframe maximum interval
     *
     * This value, expressed as a number of frames, forces the encoder to code
     * a keyframe if one has not been coded in the last kf_max_dist frames.
     * A value of 0 implies all frames will be keyframes. Set kf_min_dist
     * equal to kf_max_dist for a fixed interval.
     */
    unsigned int           kf_max_dist;

    /*
     * Spatial scalability settings (ss)
     */

    /*!\brief Number of spatial coding layers.
     *
     * This value specifies the number of spatial coding layers to be used.
     */
    unsigned int           ss_number_layers;

    /*!\brief Enable auto alt reference flags for each spatial layer.
     *
     * These values specify if auto alt reference frame is enabled for each
     * spatial layer.
     */
    int                    ss_enable_auto_alt_ref[VPX_SS_MAX_LAYERS];

    /*!\brief Target bitrate for each spatial layer.
     *
     * These values specify the target coding bitrate to be used for each
     * spatial layer.
     */
    unsigned int           ss_target_bitrate[VPX_SS_MAX_LAYERS];

    /*!\brief Number of temporal coding layers.
     *
     * This value specifies the number of temporal layers to be used.
     */
    unsigned int           ts_number_layers;

    /*!\brief Target bitrate for each temporal layer.
     *
     * These values specify the target coding bitrate to be used for each
     * temporal layer.
     */
    unsigned int           ts_target_bitrate[VPX_TS_MAX_LAYERS];

    /*!\brief Frame rate decimation factor for each temporal layer.
     *
     * These values specify the frame rate decimation factors to apply
     * to each temporal layer.
     */
    unsigned int           ts_rate_decimator[VPX_TS_MAX_LAYERS];

    /*!\brief Length of the sequence defining frame temporal layer membership.
     *
     * This value specifies the length of the sequence that defines the
     * membership of frames to temporal layers. For example, if the
     * ts_periodicity = 8, then the frames are assigned to coding layers with a
     * repeated sequence of length 8.
    */
    unsigned int           ts_periodicity;

    /*!\brief Template defining the membership of frames to temporal layers.
     *
     * This array defines the membership of frames to temporal coding layers.
     * For a 2-layer encoding that assigns even numbered frames to one temporal
     * layer (0) and odd numbered frames to a second temporal layer (1) with
     * ts_periodicity=8, then ts_layer_id = (0,1,0,1,0,1,0,1).
    */
    unsigned int           ts_layer_id[VPX_TS_MAX_PERIODICITY];

    /*!\brief Target bitrate for each spatial/temporal layer.
     *
     * These values specify the target coding bitrate to be used for each
     * spatial/temporal layer.
     *
     */
    unsigned int           layer_target_bitrate[VPX_MAX_LAYERS];

    /*!\brief Temporal layering mode indicating which temporal layering scheme to use.
     *
     * The value (refer to VP9E_TEMPORAL_LAYERING_MODE) specifies the
     * temporal layering mode to use.
     *
     */
    int                    temporal_layering_mode;
  } vpx_codec_enc_cfg_t; /**< alias for struct vpx_codec_enc_cfg */

  /*!\brief  vp9 svc extra configure parameters
   *
   * This defines max/min quantizers and scale factors for each layer
   *
   */
  typedef struct vpx_svc_parameters {
    int max_quantizers[VPX_MAX_LAYERS]; /**< Max Q for each layer */
    int min_quantizers[VPX_MAX_LAYERS]; /**< Min Q for each layer */
    int scaling_factor_num[VPX_MAX_LAYERS]; /**< Scaling factor-numerator */
    int scaling_factor_den[VPX_MAX_LAYERS]; /**< Scaling factor-denominator */
    int temporal_layering_mode; /**< Temporal layering mode */
  } vpx_svc_extra_cfg_t;


  /*!\brief Initialize an encoder instance
   *
   * Initializes a encoder context using the given interface. Applications
   * should call the vpx_codec_enc_init convenience macro instead of this
   * function directly, to ensure that the ABI version number parameter
   * is properly initialized.
   *
   * If the library was configured with --disable-multithread, this call
   * is not thread safe and should be guarded with a lock if being used
   * in a multithreaded context.
   *
   * \param[in]    ctx     Pointer to this instance's context.
   * \param[in]    iface   Pointer to the algorithm interface to use.
   * \param[in]    cfg     Configuration to use, if known. May be NULL.
   * \param[in]    flags   Bitfield of VPX_CODEC_USE_* flags
   * \param[in]    ver     ABI version number. Must be set to
   *                       VPX_ENCODER_ABI_VERSION
   * \retval #VPX_CODEC_OK
   *     The decoder algorithm initialized.
   * \retval #VPX_CODEC_MEM_ERROR
   *     Memory allocation failed.
   */
  vpx_codec_err_t vpx_codec_enc_init_ver(vpx_codec_ctx_t      *ctx,
                                         vpx_codec_iface_t    *iface,
                                         const vpx_codec_enc_cfg_t *cfg,
                                         vpx_codec_flags_t     flags,
                                         int                   ver);


  /*!\brief Convenience macro for vpx_codec_enc_init_ver()
   *
   * Ensures the ABI version parameter is properly set.
   */
#define vpx_codec_enc_init(ctx, iface, cfg, flags) \
  vpx_codec_enc_init_ver(ctx, iface, cfg, flags, VPX_ENCODER_ABI_VERSION)


  /*!\brief Initialize multi-encoder instance
   *
   * Initializes multi-encoder context using the given interface.
   * Applications should call the vpx_codec_enc_init_multi convenience macro
   * instead of this function directly, to ensure that the ABI version number
   * parameter is properly initialized.
   *
   * \param[in]    ctx     Pointer to this instance's context.
   * \param[in]    iface   Pointer to the algorithm interface to use.
   * \param[in]    cfg     Configuration to use, if known. May be NULL.
   * \param[in]    num_enc Total number of encoders.
   * \param[in]    flags   Bitfield of VPX_CODEC_USE_* flags
   * \param[in]    dsf     Pointer to down-sampling factors.
   * \param[in]    ver     ABI version number. Must be set to
   *                       VPX_ENCODER_ABI_VERSION
   * \retval #VPX_CODEC_OK
   *     The decoder algorithm initialized.
   * \retval #VPX_CODEC_MEM_ERROR
   *     Memory allocation failed.
   */
  vpx_codec_err_t vpx_codec_enc_init_multi_ver(vpx_codec_ctx_t      *ctx,
                                               vpx_codec_iface_t    *iface,
                                               vpx_codec_enc_cfg_t  *cfg,
                                               int                   num_enc,
                                               vpx_codec_flags_t     flags,
                                               vpx_rational_t       *dsf,
                                               int                   ver);


  /*!\brief Convenience macro for vpx_codec_enc_init_multi_ver()
   *
   * Ensures the ABI version parameter is properly set.
   */
#define vpx_codec_enc_init_multi(ctx, iface, cfg, num_enc, flags, dsf) \
  vpx_codec_enc_init_multi_ver(ctx, iface, cfg, num_enc, flags, dsf, \
                               VPX_ENCODER_ABI_VERSION)


  /*!\brief Get a default configuration
   *
   * Initializes a encoder configuration structure with default values. Supports
   * the notion of "usages" so that an algorithm may offer different default
   * settings depending on the user's intended goal. This function \ref SHOULD
   * be called by all applications to initialize the configuration structure
   * before specializing the configuration with application specific values.
   *
   * \param[in]    iface     Pointer to the algorithm interface to use.
   * \param[out]   cfg       Configuration buffer to populate.
   * \param[in]    reserved  Must set to 0 for VP8 and VP9.
   *
   * \retval #VPX_CODEC_OK
   *     The configuration was populated.
   * \retval #VPX_CODEC_INCAPABLE
   *     Interface is not an encoder interface.
   * \retval #VPX_CODEC_INVALID_PARAM
   *     A parameter was NULL, or the usage value was not recognized.
   */
  vpx_codec_err_t  vpx_codec_enc_config_default(vpx_codec_iface_t    *iface,
                                                vpx_codec_enc_cfg_t  *cfg,
                                                unsigned int          reserved);


  /*!\brief Set or change configuration
   *
   * Reconfigures an encoder instance according to the given configuration.
   *
   * \param[in]    ctx     Pointer to this instance's context
   * \param[in]    cfg     Configuration buffer to use
   *
   * \retval #VPX_CODEC_OK
   *     The configuration was populated.
   * \retval #VPX_CODEC_INCAPABLE
   *     Interface is not an encoder interface.
   * \retval #VPX_CODEC_INVALID_PARAM
   *     A parameter was NULL, or the usage value was not recognized.
   */
  vpx_codec_err_t  vpx_codec_enc_config_set(vpx_codec_ctx_t            *ctx,
                                            const vpx_codec_enc_cfg_t  *cfg);


  /*!\brief Get global stream headers
   *
   * Retrieves a stream level global header packet, if supported by the codec.
   *
   * \param[in]    ctx     Pointer to this instance's context
   *
   * \retval NULL
   *     Encoder does not support global header
   * \retval Non-NULL
   *     Pointer to buffer containing global header packet
   */
  vpx_fixed_buf_t *vpx_codec_get_global_headers(vpx_codec_ctx_t   *ctx);


#define VPX_DL_REALTIME     (1)        /**< deadline parameter analogous to
  *   VPx REALTIME mode. */
#define VPX_DL_GOOD_QUALITY (1000000)  /**< deadline parameter analogous to
  *   VPx GOOD QUALITY mode. */
#define VPX_DL_BEST_QUALITY (0)        /**< deadline parameter analogous to
  *   VPx BEST QUALITY mode. */
  /*!\brief Encode a frame
   *
   * Encodes a video frame at the given "presentation time." The presentation
   * time stamp (PTS) \ref MUST be strictly increasing.
   *
   * The encoder supports the notion of a soft real-time deadline. Given a
   * non-zero value to the deadline parameter, the encoder will make a "best
   * effort" guarantee to  return before the given time slice expires. It is
   * implicit that limiting the available time to encode will degrade the
   * output quality. The encoder can be given an unlimited time to produce the
   * best possible frame by specifying a deadline of '0'. This deadline
   * supercedes the VPx notion of "best quality, good quality, realtime".
   * Applications that wish to map these former settings to the new deadline
   * based system can use the symbols #VPX_DL_REALTIME, #VPX_DL_GOOD_QUALITY,
   * and #VPX_DL_BEST_QUALITY.
   *
   * When the last frame has been passed to the encoder, this function should
   * continue to be called, with the img parameter set to NULL. This will
   * signal the end-of-stream condition to the encoder and allow it to encode
   * any held buffers. Encoding is complete when vpx_codec_encode() is called
   * and vpx_codec_get_cx_data() returns no data.
   *
   * \param[in]    ctx       Pointer to this instance's context
   * \param[in]    img       Image data to encode, NULL to flush.
   * \param[in]    pts       Presentation time stamp, in timebase units.
   * \param[in]    duration  Duration to show frame, in timebase units.
   * \param[in]    flags     Flags to use for encoding this frame.
   * \param[in]    deadline  Time to spend encoding, in microseconds. (0=infinite)
   *
   * \retval #VPX_CODEC_OK
   *     The configuration was populated.
   * \retval #VPX_CODEC_INCAPABLE
   *     Interface is not an encoder interface.
   * \retval #VPX_CODEC_INVALID_PARAM
   *     A parameter was NULL, the image format is unsupported, etc.
   */
  vpx_codec_err_t  vpx_codec_encode(vpx_codec_ctx_t            *ctx,
                                    const vpx_image_t          *img,
                                    vpx_codec_pts_t             pts,
                                    unsigned long               duration,
                                    vpx_enc_frame_flags_t       flags,
                                    unsigned long               deadline);

  /*!\brief Set compressed data output buffer
   *
   * Sets the buffer that the codec should output the compressed data
   * into. This call effectively sets the buffer pointer returned in the
   * next VPX_CODEC_CX_FRAME_PKT packet. Subsequent packets will be
   * appended into this buffer. The buffer is preserved across frames,
   * so applications must periodically call this function after flushing
   * the accumulated compressed data to disk or to the network to reset
   * the pointer to the buffer's head.
   *
   * `pad_before` bytes will be skipped before writing the compressed
   * data, and `pad_after` bytes will be appended to the packet. The size
   * of the packet will be the sum of the size of the actual compressed
   * data, pad_before, and pad_after. The padding bytes will be preserved
   * (not overwritten).
   *
   * Note that calling this function does not guarantee that the returned
   * compressed data will be placed into the specified buffer. In the
   * event that the encoded data will not fit into the buffer provided,
   * the returned packet \ref MAY point to an internal buffer, as it would
   * if this call were never used. In this event, the output packet will
   * NOT have any padding, and the application must free space and copy it
   * to the proper place. This is of particular note in configurations
   * that may output multiple packets for a single encoded frame (e.g., lagged
   * encoding) or if the application does not reset the buffer periodically.
   *
   * Applications may restore the default behavior of the codec providing
   * the compressed data buffer by calling this function with a NULL
   * buffer.
   *
   * Applications \ref MUSTNOT call this function during iteration of
   * vpx_codec_get_cx_data().
   *
   * \param[in]    ctx         Pointer to this instance's context
   * \param[in]    buf         Buffer to store compressed data into
   * \param[in]    pad_before  Bytes to skip before writing compressed data
   * \param[in]    pad_after   Bytes to skip after writing compressed data
   *
   * \retval #VPX_CODEC_OK
   *     The buffer was set successfully.
   * \retval #VPX_CODEC_INVALID_PARAM
   *     A parameter was NULL, the image format is unsupported, etc.
   */
  vpx_codec_err_t vpx_codec_set_cx_data_buf(vpx_codec_ctx_t       *ctx,
                                            const vpx_fixed_buf_t *buf,
                                            unsigned int           pad_before,
                                            unsigned int           pad_after);


  /*!\brief Encoded data iterator
   *
   * Iterates over a list of data packets to be passed from the encoder to the
   * application. The different kinds of packets available are enumerated in
   * #vpx_codec_cx_pkt_kind.
   *
   * #VPX_CODEC_CX_FRAME_PKT packets should be passed to the application's
   * muxer. Multiple compressed frames may be in the list.
   * #VPX_CODEC_STATS_PKT packets should be appended to a global buffer.
   *
   * The application \ref MUST silently ignore any packet kinds that it does
   * not recognize or support.
   *
   * The data buffers returned from this function are only guaranteed to be
   * valid until the application makes another call to any vpx_codec_* function.
   *
   * \param[in]     ctx      Pointer to this instance's context
   * \param[in,out] iter     Iterator storage, initialized to NULL
   *
   * \return Returns a pointer to an output data packet (compressed frame data,
   *         two-pass statistics, etc.) or NULL to signal end-of-list.
   *
   */
  const vpx_codec_cx_pkt_t *vpx_codec_get_cx_data(vpx_codec_ctx_t   *ctx,
                                                  vpx_codec_iter_t  *iter);


  /*!\brief Get Preview Frame
   *
   * Returns an image that can be used as a preview. Shows the image as it would
   * exist at the decompressor. The application \ref MUST NOT write into this
   * image buffer.
   *
   * \param[in]     ctx      Pointer to this instance's context
   *
   * \return Returns a pointer to a preview image, or NULL if no image is
   *         available.
   *
   */
  const vpx_image_t *vpx_codec_get_preview_frame(vpx_codec_ctx_t   *ctx);


  /*!@} - end defgroup encoder*/
#ifdef __cplusplus
}
#endif
#endif  // VPX_VPX_ENCODER_H_

