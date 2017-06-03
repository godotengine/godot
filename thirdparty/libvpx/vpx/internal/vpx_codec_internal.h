/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */


/*!\file
 * \brief Describes the decoder algorithm interface for algorithm
 *        implementations.
 *
 * This file defines the private structures and data types that are only
 * relevant to implementing an algorithm, as opposed to using it.
 *
 * To create a decoder algorithm class, an interface structure is put
 * into the global namespace:
 *     <pre>
 *     my_codec.c:
 *       vpx_codec_iface_t my_codec = {
 *           "My Codec v1.0",
 *           VPX_CODEC_ALG_ABI_VERSION,
 *           ...
 *       };
 *     </pre>
 *
 * An application instantiates a specific decoder instance by using
 * vpx_codec_init() and a pointer to the algorithm's interface structure:
 *     <pre>
 *     my_app.c:
 *       extern vpx_codec_iface_t my_codec;
 *       {
 *           vpx_codec_ctx_t algo;
 *           res = vpx_codec_init(&algo, &my_codec);
 *       }
 *     </pre>
 *
 * Once initialized, the instance is manged using other functions from
 * the vpx_codec_* family.
 */
#ifndef VPX_INTERNAL_VPX_CODEC_INTERNAL_H_
#define VPX_INTERNAL_VPX_CODEC_INTERNAL_H_
#include "../vpx_decoder.h"
#include "../vpx_encoder.h"
#include <stdarg.h>

#ifdef __cplusplus
extern "C" {
#endif

/*!\brief Current ABI version number
 *
 * \internal
 * If this file is altered in any way that changes the ABI, this value
 * must be bumped.  Examples include, but are not limited to, changing
 * types, removing or reassigning enums, adding/removing/rearranging
 * fields to structures
 */
#define VPX_CODEC_INTERNAL_ABI_VERSION (5) /**<\hideinitializer*/

typedef struct vpx_codec_alg_priv  vpx_codec_alg_priv_t;
typedef struct vpx_codec_priv_enc_mr_cfg vpx_codec_priv_enc_mr_cfg_t;

/*!\brief init function pointer prototype
 *
 * Performs algorithm-specific initialization of the decoder context. This
 * function is called by the generic vpx_codec_init() wrapper function, so
 * plugins implementing this interface may trust the input parameters to be
 * properly initialized.
 *
 * \param[in] ctx   Pointer to this instance's context
 * \retval #VPX_CODEC_OK
 *     The input stream was recognized and decoder initialized.
 * \retval #VPX_CODEC_MEM_ERROR
 *     Memory operation failed.
 */
typedef vpx_codec_err_t (*vpx_codec_init_fn_t)(vpx_codec_ctx_t *ctx,
                                               vpx_codec_priv_enc_mr_cfg_t *data);

/*!\brief destroy function pointer prototype
 *
 * Performs algorithm-specific destruction of the decoder context. This
 * function is called by the generic vpx_codec_destroy() wrapper function,
 * so plugins implementing this interface may trust the input parameters
 * to be properly initialized.
 *
 * \param[in] ctx   Pointer to this instance's context
 * \retval #VPX_CODEC_OK
 *     The input stream was recognized and decoder initialized.
 * \retval #VPX_CODEC_MEM_ERROR
 *     Memory operation failed.
 */
typedef vpx_codec_err_t (*vpx_codec_destroy_fn_t)(vpx_codec_alg_priv_t *ctx);

/*!\brief parse stream info function pointer prototype
 *
 * Performs high level parsing of the bitstream. This function is called by the
 * generic vpx_codec_peek_stream_info() wrapper function, so plugins
 * implementing this interface may trust the input parameters to be properly
 * initialized.
 *
 * \param[in]      data    Pointer to a block of data to parse
 * \param[in]      data_sz Size of the data buffer
 * \param[in,out]  si      Pointer to stream info to update. The size member
 *                         \ref MUST be properly initialized, but \ref MAY be
 *                         clobbered by the algorithm. This parameter \ref MAY
 *                         be NULL.
 *
 * \retval #VPX_CODEC_OK
 *     Bitstream is parsable and stream information updated
 */
typedef vpx_codec_err_t (*vpx_codec_peek_si_fn_t)(const uint8_t         *data,
                                                  unsigned int           data_sz,
                                                  vpx_codec_stream_info_t *si);

/*!\brief Return information about the current stream.
 *
 * Returns information about the stream that has been parsed during decoding.
 *
 * \param[in]      ctx     Pointer to this instance's context
 * \param[in,out]  si      Pointer to stream info to update. The size member
 *                         \ref MUST be properly initialized, but \ref MAY be
 *                         clobbered by the algorithm. This parameter \ref MAY
 *                         be NULL.
 *
 * \retval #VPX_CODEC_OK
 *     Bitstream is parsable and stream information updated
 */
typedef vpx_codec_err_t (*vpx_codec_get_si_fn_t)(vpx_codec_alg_priv_t    *ctx,
                                                 vpx_codec_stream_info_t *si);

/*!\brief control function pointer prototype
 *
 * This function is used to exchange algorithm specific data with the decoder
 * instance. This can be used to implement features specific to a particular
 * algorithm.
 *
 * This function is called by the generic vpx_codec_control() wrapper
 * function, so plugins implementing this interface may trust the input
 * parameters to be properly initialized. However,  this interface does not
 * provide type safety for the exchanged data or assign meanings to the
 * control codes. Those details should be specified in the algorithm's
 * header file. In particular, the ctrl_id parameter is guaranteed to exist
 * in the algorithm's control mapping table, and the data parameter may be NULL.
 *
 *
 * \param[in]     ctx              Pointer to this instance's context
 * \param[in]     ctrl_id          Algorithm specific control identifier
 * \param[in,out] data             Data to exchange with algorithm instance.
 *
 * \retval #VPX_CODEC_OK
 *     The internal state data was deserialized.
 */
typedef vpx_codec_err_t (*vpx_codec_control_fn_t)(vpx_codec_alg_priv_t *ctx,
                                                  va_list ap);

/*!\brief control function pointer mapping
 *
 * This structure stores the mapping between control identifiers and
 * implementing functions. Each algorithm provides a list of these
 * mappings. This list is searched by the vpx_codec_control() wrapper
 * function to determine which function to invoke. The special
 * value {0, NULL} is used to indicate end-of-list, and must be
 * present. The special value {0, <non-null>} can be used as a catch-all
 * mapping. This implies that ctrl_id values chosen by the algorithm
 * \ref MUST be non-zero.
 */
typedef const struct vpx_codec_ctrl_fn_map {
  int ctrl_id;
  vpx_codec_control_fn_t fn;
} vpx_codec_ctrl_fn_map_t;

/*!\brief decode data function pointer prototype
 *
 * Processes a buffer of coded data. If the processing results in a new
 * decoded frame becoming available, #VPX_CODEC_CB_PUT_SLICE and
 * #VPX_CODEC_CB_PUT_FRAME events are generated as appropriate. This
 * function is called by the generic vpx_codec_decode() wrapper function,
 * so plugins implementing this interface may trust the input parameters
 * to be properly initialized.
 *
 * \param[in] ctx          Pointer to this instance's context
 * \param[in] data         Pointer to this block of new coded data. If
 *                         NULL, a #VPX_CODEC_CB_PUT_FRAME event is posted
 *                         for the previously decoded frame.
 * \param[in] data_sz      Size of the coded data, in bytes.
 *
 * \return Returns #VPX_CODEC_OK if the coded data was processed completely
 *         and future pictures can be decoded without error. Otherwise,
 *         see the descriptions of the other error codes in ::vpx_codec_err_t
 *         for recoverability capabilities.
 */
typedef vpx_codec_err_t (*vpx_codec_decode_fn_t)(vpx_codec_alg_priv_t  *ctx,
                                                 const uint8_t         *data,
                                                 unsigned int     data_sz,
                                                 void        *user_priv,
                                                 long         deadline);

/*!\brief Decoded frames iterator
 *
 * Iterates over a list of the frames available for display. The iterator
 * storage should be initialized to NULL to start the iteration. Iteration is
 * complete when this function returns NULL.
 *
 * The list of available frames becomes valid upon completion of the
 * vpx_codec_decode call, and remains valid until the next call to vpx_codec_decode.
 *
 * \param[in]     ctx      Pointer to this instance's context
 * \param[in out] iter     Iterator storage, initialized to NULL
 *
 * \return Returns a pointer to an image, if one is ready for display. Frames
 *         produced will always be in PTS (presentation time stamp) order.
 */
typedef vpx_image_t *(*vpx_codec_get_frame_fn_t)(vpx_codec_alg_priv_t *ctx,
                                                 vpx_codec_iter_t     *iter);

/*!\brief Pass in external frame buffers for the decoder to use.
 *
 * Registers functions to be called when libvpx needs a frame buffer
 * to decode the current frame and a function to be called when libvpx does
 * not internally reference the frame buffer. This set function must
 * be called before the first call to decode or libvpx will assume the
 * default behavior of allocating frame buffers internally.
 *
 * \param[in] ctx          Pointer to this instance's context
 * \param[in] cb_get       Pointer to the get callback function
 * \param[in] cb_release   Pointer to the release callback function
 * \param[in] cb_priv      Callback's private data
 *
 * \retval #VPX_CODEC_OK
 *     External frame buffers will be used by libvpx.
 * \retval #VPX_CODEC_INVALID_PARAM
 *     One or more of the callbacks were NULL.
 * \retval #VPX_CODEC_ERROR
 *     Decoder context not initialized, or algorithm not capable of
 *     using external frame buffers.
 *
 * \note
 * When decoding VP9, the application may be required to pass in at least
 * #VP9_MAXIMUM_REF_BUFFERS + #VPX_MAXIMUM_WORK_BUFFERS external frame
 * buffers.
 */
typedef vpx_codec_err_t (*vpx_codec_set_fb_fn_t)(
    vpx_codec_alg_priv_t *ctx,
    vpx_get_frame_buffer_cb_fn_t cb_get,
    vpx_release_frame_buffer_cb_fn_t cb_release, void *cb_priv);


typedef vpx_codec_err_t (*vpx_codec_encode_fn_t)(vpx_codec_alg_priv_t  *ctx,
                                                 const vpx_image_t     *img,
                                                 vpx_codec_pts_t        pts,
                                                 unsigned long          duration,
                                                 vpx_enc_frame_flags_t  flags,
                                                 unsigned long          deadline);
typedef const vpx_codec_cx_pkt_t *(*vpx_codec_get_cx_data_fn_t)(vpx_codec_alg_priv_t *ctx,
                                                                vpx_codec_iter_t     *iter);

typedef vpx_codec_err_t
(*vpx_codec_enc_config_set_fn_t)(vpx_codec_alg_priv_t       *ctx,
                                 const vpx_codec_enc_cfg_t  *cfg);
typedef vpx_fixed_buf_t *
(*vpx_codec_get_global_headers_fn_t)(vpx_codec_alg_priv_t   *ctx);

typedef vpx_image_t *
(*vpx_codec_get_preview_frame_fn_t)(vpx_codec_alg_priv_t   *ctx);

typedef vpx_codec_err_t
(*vpx_codec_enc_mr_get_mem_loc_fn_t)(const vpx_codec_enc_cfg_t     *cfg,
                                     void **mem_loc);

/*!\brief usage configuration mapping
 *
 * This structure stores the mapping between usage identifiers and
 * configuration structures. Each algorithm provides a list of these
 * mappings. This list is searched by the vpx_codec_enc_config_default()
 * wrapper function to determine which config to return. The special value
 * {-1, {0}} is used to indicate end-of-list, and must be present. At least
 * one mapping must be present, in addition to the end-of-list.
 *
 */
typedef const struct vpx_codec_enc_cfg_map {
  int                 usage;
  vpx_codec_enc_cfg_t cfg;
} vpx_codec_enc_cfg_map_t;

/*!\brief Decoder algorithm interface interface
 *
 * All decoders \ref MUST expose a variable of this type.
 */
struct vpx_codec_iface {
  const char               *name;        /**< Identification String  */
  int                       abi_version; /**< Implemented ABI version */
  vpx_codec_caps_t          caps;    /**< Decoder capabilities */
  vpx_codec_init_fn_t       init;    /**< \copydoc ::vpx_codec_init_fn_t */
  vpx_codec_destroy_fn_t    destroy;     /**< \copydoc ::vpx_codec_destroy_fn_t */
  vpx_codec_ctrl_fn_map_t  *ctrl_maps;   /**< \copydoc ::vpx_codec_ctrl_fn_map_t */
  struct vpx_codec_dec_iface {
    vpx_codec_peek_si_fn_t    peek_si;     /**< \copydoc ::vpx_codec_peek_si_fn_t */
    vpx_codec_get_si_fn_t     get_si;      /**< \copydoc ::vpx_codec_get_si_fn_t */
    vpx_codec_decode_fn_t     decode;      /**< \copydoc ::vpx_codec_decode_fn_t */
    vpx_codec_get_frame_fn_t  get_frame;   /**< \copydoc ::vpx_codec_get_frame_fn_t */
    vpx_codec_set_fb_fn_t     set_fb_fn;   /**< \copydoc ::vpx_codec_set_fb_fn_t */
  } dec;
  struct vpx_codec_enc_iface {
    int                                cfg_map_count;
    vpx_codec_enc_cfg_map_t           *cfg_maps;      /**< \copydoc ::vpx_codec_enc_cfg_map_t */
    vpx_codec_encode_fn_t              encode;        /**< \copydoc ::vpx_codec_encode_fn_t */
    vpx_codec_get_cx_data_fn_t         get_cx_data;   /**< \copydoc ::vpx_codec_get_cx_data_fn_t */
    vpx_codec_enc_config_set_fn_t      cfg_set;       /**< \copydoc ::vpx_codec_enc_config_set_fn_t */
    vpx_codec_get_global_headers_fn_t  get_glob_hdrs; /**< \copydoc ::vpx_codec_get_global_headers_fn_t */
    vpx_codec_get_preview_frame_fn_t   get_preview;   /**< \copydoc ::vpx_codec_get_preview_frame_fn_t */
    vpx_codec_enc_mr_get_mem_loc_fn_t  mr_get_mem_loc;   /**< \copydoc ::vpx_codec_enc_mr_get_mem_loc_fn_t */
  } enc;
};

/*!\brief Callback function pointer / user data pair storage */
typedef struct vpx_codec_priv_cb_pair {
  union {
    vpx_codec_put_frame_cb_fn_t    put_frame;
    vpx_codec_put_slice_cb_fn_t    put_slice;
  } u;
  void                            *user_priv;
} vpx_codec_priv_cb_pair_t;


/*!\brief Instance private storage
 *
 * This structure is allocated by the algorithm's init function. It can be
 * extended in one of two ways. First, a second, algorithm specific structure
 * can be allocated and the priv member pointed to it. Alternatively, this
 * structure can be made the first member of the algorithm specific structure,
 * and the pointer cast to the proper type.
 */
struct vpx_codec_priv {
  const char                     *err_detail;
  vpx_codec_flags_t               init_flags;
  struct {
    vpx_codec_priv_cb_pair_t    put_frame_cb;
    vpx_codec_priv_cb_pair_t    put_slice_cb;
  } dec;
  struct {
    vpx_fixed_buf_t             cx_data_dst_buf;
    unsigned int                cx_data_pad_before;
    unsigned int                cx_data_pad_after;
    vpx_codec_cx_pkt_t          cx_data_pkt;
    unsigned int                total_encoders;
  } enc;
};

/*
 * Multi-resolution encoding internal configuration
 */
struct vpx_codec_priv_enc_mr_cfg
{
    unsigned int           mr_total_resolutions;
    unsigned int           mr_encoder_id;
    struct vpx_rational    mr_down_sampling_factor;
    void*                  mr_low_res_mode_info;
};

#undef VPX_CTRL_USE_TYPE
#define VPX_CTRL_USE_TYPE(id, typ) \
  static VPX_INLINE typ id##__value(va_list args) {return va_arg(args, typ);}

#undef VPX_CTRL_USE_TYPE_DEPRECATED
#define VPX_CTRL_USE_TYPE_DEPRECATED(id, typ) \
  static VPX_INLINE typ id##__value(va_list args) {return va_arg(args, typ);}

#define CAST(id, arg) id##__value(arg)

/* CODEC_INTERFACE convenience macro
 *
 * By convention, each codec interface is a struct with extern linkage, where
 * the symbol is suffixed with _algo. A getter function is also defined to
 * return a pointer to the struct, since in some cases it's easier to work
 * with text symbols than data symbols (see issue #169). This function has
 * the same name as the struct, less the _algo suffix. The CODEC_INTERFACE
 * macro is provided to define this getter function automatically.
 */
#define CODEC_INTERFACE(id)\
  vpx_codec_iface_t* id(void) { return &id##_algo; }\
  vpx_codec_iface_t  id##_algo


/* Internal Utility Functions
 *
 * The following functions are intended to be used inside algorithms as
 * utilities for manipulating vpx_codec_* data structures.
 */
struct vpx_codec_pkt_list {
  unsigned int            cnt;
  unsigned int            max;
  struct vpx_codec_cx_pkt pkts[1];
};

#define vpx_codec_pkt_list_decl(n)\
  union {struct vpx_codec_pkt_list head;\
    struct {struct vpx_codec_pkt_list head;\
      struct vpx_codec_cx_pkt    pkts[n];} alloc;}

#define vpx_codec_pkt_list_init(m)\
  (m)->alloc.head.cnt = 0,\
                        (m)->alloc.head.max = sizeof((m)->alloc.pkts) / sizeof((m)->alloc.pkts[0])

int
vpx_codec_pkt_list_add(struct vpx_codec_pkt_list *,
                       const struct vpx_codec_cx_pkt *);

const vpx_codec_cx_pkt_t *
vpx_codec_pkt_list_get(struct vpx_codec_pkt_list *list,
                       vpx_codec_iter_t           *iter);


#include <stdio.h>
#include <setjmp.h>

struct vpx_internal_error_info {
  vpx_codec_err_t  error_code;
  int              has_detail;
  char             detail[80];
  int              setjmp;
  jmp_buf          jmp;
};

#define CLANG_ANALYZER_NORETURN
#if defined(__has_feature)
#if __has_feature(attribute_analyzer_noreturn)
#undef CLANG_ANALYZER_NORETURN
#define CLANG_ANALYZER_NORETURN __attribute__((analyzer_noreturn))
#endif
#endif

void vpx_internal_error(struct vpx_internal_error_info *info,
                        vpx_codec_err_t                 error,
                        const char                     *fmt,
                        ...) CLANG_ANALYZER_NORETURN;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_INTERNAL_VPX_CODEC_INTERNAL_H_
