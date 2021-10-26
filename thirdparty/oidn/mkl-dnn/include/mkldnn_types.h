/*******************************************************************************
* Copyright 2016-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef MKLDNN_TYPES_H
#define MKLDNN_TYPES_H

#ifdef __cplusplus
extern "C" {
#endif

#ifndef DOXYGEN_SHOULD_SKIP_THIS
#include <stddef.h>
#include <stdint.h>
#endif

/** @addtogroup c_api C API
 *  @{
 *
 *  @addtogroup c_api_types Types
 *  @{
 *
 *  @addtogroup c_api_types_generic Generic
 *  @{ */

/** Intel(R) MKL-DNN Version type */
typedef struct {
    int    major;
    int    minor;
    int    patch;
    const char *hash;
} mkldnn_version_t;

/** Status values returned by Intel(R) MKL-DNN functions. */
typedef enum {
    /** The operation was successful */
    mkldnn_success = 0,
    /** The operation failed due to an out-of-memory condition */
    mkldnn_out_of_memory = 1,
    /** The operation failed and should be retried */
    mkldnn_try_again = 2,
    /** The operation failed because of incorrect function arguments  */
    mkldnn_invalid_arguments = 3,
    /** The operation failed because a primitive was not ready for execution */
    mkldnn_not_ready = 4,
    /** The operation failed because requested functionality is not implemented
     */
    mkldnn_unimplemented = 5,
    /** Primitive iterator passed over last primitive descriptor */
    mkldnn_iterator_ends = 6,
    /** Primitive or engine failed on execution */
    mkldnn_runtime_error = 7,
    /** Queried element is not required for given primitive */
    mkldnn_not_required = 8,
} mkldnn_status_t;

/** Data type specification */
typedef enum {
    /** Undefined data type, used for empty memory descriptors. */
    mkldnn_data_type_undef = 0,
    /** 32-bit/single-precision floating point. */
    mkldnn_f32 = 1,
    /** 32-bit signed integer. */
    mkldnn_s32 = 2,
    /** 8-bit signed integer. */
    mkldnn_s8 = 3,
    /** 8-bit unsigned integer. */
    mkldnn_u8 = 4,
} mkldnn_data_type_t;

/** Memory format kind */
typedef enum {
    /** Undefined memory format, used for empty memory descriptors. */
    mkldnn_format_kind_undef = 0,
    /** Unspecified format. The primitive selects a format automatically. */
    mkldnn_format_kind_any,
    /** A tensor in a generic format described by the stride and blocking
     * values in each dimension. See #mkldnn_blocking_desc_t for more
     * information. */
    mkldnn_blocked,
    /** Weights format used in 8bit Winograd convolution */
    mkldnn_format_kind_wino,
    /** Packed weights format used in RNN */
    mkldnn_format_kind_rnn_packed,
} mkldnn_format_kind_t;

/** Memory format tag specification.
 *
 * Intel MKL-DNN formats describe physical data layout. The physical layout
 * is described as a sequence of the dimensions as they are laid out in the
 * memory (from the outer-most to the inner-most). Note that this order
 * doesn't affect the logical order of the dimensions that is kept in the
 * `dims` field of the mkldnn_memory_desc_t structure. The logical order of the
 * dimensions is specified by the type of tensor.
 *
 * For example, CNN 5D tensor always has its logical dimensions in the order
 * `(batch, channels, depth, height, width)`, while the physical layout might be
 * #mkldnn_ncdhw or #mkldnn_ndhwc:
 *
 * ~~~cpp
 * int batch = 2, channels = 16, depth = 13, height = 13, width = 13;
 *
 * int ndims = 5; // 5D tensor
 * mkldnn_dims_t dims = {batch, channels, depth, height, width};
 * mkldnn_memory_desc_t data_in_ncdhw;
 * mkldnn_memory_desc_init_by_tag(
 *      &data_in_ncdhw, 5, dims, mkldnn_f32, mkldnn_ncdhw);
 *
 * // note that in both cases dims passed are the same
 * mkldnn_memory_desc_t data_in_ndhwc;
 * mkldnn_memory_desc_init_by_tag(
 *      &data_in_ndhwc, 5, dims, mkldnn_f32, mkldnn_ndhwc);
 * ~~~
 *
 * The following notation applies to memory format names:
 *  - @c 'n' denotes the mini-batch dimension
 *  - @c 'c' denotes a channels dimension
 *  - When there are multiple channel dimensions (for example, in convolution
 *    weights tensor), @c 'i' and @c 'o' denote dimensions of input and output
 *    channels
 *  - @c 'd', @c 'h', and @c 'w' denote spatial depth, height, and width
 *    respectively
 *  - Upper-case letters indicate that the data is laid out in blocks
 *    for a particular dimension. In such cases, the format name contains both
 *    upper- and lower-case letters for that dimension with a lower-case letter
 *    preceded by the block size. For example: @c 'mkldnn_nChw8c' describes a
 *    format where the outermost dimension is mini-batch, followed by the
 *    channel block number, followed by the spatial height and width, and
 *    finally followed by 8-element channel blocks.
 *
 * @note
 *    Channel designations can be different. For example, both the @c
 *    'mkldnn_nc' and @c 'mkldnn_io' formats can be used to describe a 2D
 *    tensor.
 *
 * @sa @ref understanding_memory_formats
 */
typedef enum {
    /** Undefined memory format tag */
    mkldnn_format_tag_undef = 0,
    /** Undefined memory format tag.
     * The primitive selects a format automatically. */
    mkldnn_format_tag_any,

    /* Semantic agnostic section */
    /* The physical order of dimensions is defined by the permutation of the
     * characters, assuming that ab..z defines the natural order.
     */

    /* Plain formats */

    mkldnn_a,
    mkldnn_ab,
    mkldnn_abc,
    mkldnn_abcd,
    mkldnn_abcde,
    mkldnn_abcdef,
    mkldnn_abdec,
    mkldnn_acb,
    mkldnn_acbde,
    mkldnn_acdb,
    mkldnn_acdeb,
    mkldnn_ba,
    mkldnn_bac,
    mkldnn_bacd,
    mkldnn_bcda,
    mkldnn_cba,
    mkldnn_cdba,
    mkldnn_cdeba,
    mkldnn_decab,

    /* Opaque blocked formats */

    mkldnn_Abc16a,
    mkldnn_ABc16a16b,
    mkldnn_aBc16b,
    mkldnn_ABc16b16a,
    mkldnn_Abc4a,
    mkldnn_aBc4b,
    mkldnn_ABc4b16a4b,
    mkldnn_ABc4b4a,
    mkldnn_ABc8a16b2a,
    mkldnn_ABc8a8b,
    mkldnn_aBc8b,
    mkldnn_ABc8b16a2b,
    mkldnn_ABc8b8a,
    mkldnn_Abcd16a,
    mkldnn_ABcd16a16b,
    mkldnn_aBcd16b,
    mkldnn_ABcd16b16a,
    mkldnn_aBCd16b16c,
    mkldnn_aBCd16c16b,
    mkldnn_Abcd4a,
    mkldnn_aBcd4b,
    mkldnn_ABcd4b16a4b,
    mkldnn_ABcd4b4a,
    mkldnn_aBCd4c16b4c,
    mkldnn_aBCd4c4b,
    mkldnn_ABcd8a16b2a,
    mkldnn_ABcd8a8b,
    mkldnn_aBcd8b,
    mkldnn_ABcd8b16a2b,
    mkldnn_aBCd8b16c2b,
    mkldnn_ABcd8b8a,
    mkldnn_aBCd8b8c,
    mkldnn_aBCd8c16b2c,
    mkldnn_aBCd8c8b,
    mkldnn_Abcde16a,
    mkldnn_ABcde16a16b,
    mkldnn_aBcde16b,
    mkldnn_ABcde16b16a,
    mkldnn_aBCde16b16c,
    mkldnn_aBCde16c16b,
    mkldnn_aBCde2c8b4c,
    mkldnn_Abcde4a,
    mkldnn_aBcde4b,
    mkldnn_ABcde4b4a,
    mkldnn_aBCde4b4c,
    mkldnn_aBCde4c16b4c,
    mkldnn_aBCde4c4b,
    mkldnn_Abcde8a,
    mkldnn_ABcde8a8b,
    mkldnn_aBcde8b,
    mkldnn_ABcde8b16a2b,
    mkldnn_aBCde8b16c2b,
    mkldnn_ABcde8b8a,
    mkldnn_aBCde8b8c,
    mkldnn_aBCde8c16b2c,
    mkldnn_aBCde8c8b,
    mkldnn_aBcdef16b,
    mkldnn_aBCdef16b16c,
    mkldnn_aBCdef16c16b,
    mkldnn_aBcdef4b,
    mkldnn_aBCdef4c4b,
    mkldnn_aBCdef8b8c,
    mkldnn_aBCdef8c16b2c,
    mkldnn_aBCdef8c8b,
    mkldnn_aBdc16b,
    mkldnn_aBdc4b,
    mkldnn_aBdc8b,
    mkldnn_aBdec16b,
    mkldnn_aBdec4b,
    mkldnn_aBdec8b,
    mkldnn_aBdefc16b,
    mkldnn_aBdefc4b,
    mkldnn_aBdefc8b,
    mkldnn_Acb16a,
    mkldnn_Acb4a,
    mkldnn_Acb8a,
    mkldnn_aCBd16b16c,
    mkldnn_aCBde16b16c,
    mkldnn_Acdb16a,
    mkldnn_Acdb4a,
    mkldnn_Acdb8a,
    mkldnn_Acdeb16a,
    mkldnn_Acdeb4a,
    mkldnn_Acdeb8a,
    mkldnn_BAc16a16b,
    mkldnn_BAcd16a16b,

    /** Just a sentinel, not real memory format tag. Must be changed after new
     * format tag is added. */
    mkldnn_format_tag_last,

    /* Aliases */

    mkldnn_x = mkldnn_a,
    mkldnn_nc = mkldnn_ab,
    mkldnn_cn = mkldnn_ba,
    mkldnn_ncw = mkldnn_abc,
    mkldnn_nwc = mkldnn_acb,
    mkldnn_nchw = mkldnn_abcd,
    mkldnn_nhwc = mkldnn_acdb,
    mkldnn_chwn = mkldnn_bcda,
    mkldnn_ncdhw = mkldnn_abcde,
    mkldnn_ndhwc = mkldnn_acdeb,

    mkldnn_oi = mkldnn_ab,
    mkldnn_io = mkldnn_ba,
    mkldnn_oiw = mkldnn_abc,
    mkldnn_wio = mkldnn_cba,
    mkldnn_oihw = mkldnn_abcd,
    mkldnn_hwio = mkldnn_cdba,
    mkldnn_ihwo = mkldnn_bcda,
    mkldnn_iohw = mkldnn_bacd,
    mkldnn_oidhw = mkldnn_abcde,
    mkldnn_dhwio = mkldnn_cdeba,
    mkldnn_goiw = mkldnn_abcd,
    mkldnn_goihw = mkldnn_abcde,
    mkldnn_hwigo = mkldnn_decab,
    mkldnn_giohw = mkldnn_acbde,
    mkldnn_goidhw = mkldnn_abcdef,

    /** 3D RNN data tensor in the format (seq_length, batch, input channels). */
    mkldnn_tnc = mkldnn_abc,
    /** 3D RNN data tensor in the format (batch, seq_length, input channels). */
    mkldnn_ntc = mkldnn_bac,
    /** 5D RNN states tensor in the format (num_layers, num_directions,
     * num_states, batch, state channels). */
    mkldnn_ldsnc = mkldnn_abcde,
    /** 5D RNN weights tensor in the format (num_layers, num_directions,
     *  input_channels, num_gates, output_channels).
     *
     *  - For LSTM cells, the gates order is input, forget, candidate
     *    and output gate.
     *  - For GRU cells, the gates order is update, reset and output gate. */
    mkldnn_ldigo = mkldnn_abcde,
    /** 5D RNN weights tensor in the format (num_layers, num_directions,
     * num_gates, output_channels, input_channels).
     *
     *  - For LSTM cells, the gates order is input, forget, candidate
     *    and output gate.
     *  - For GRU cells, the gates order is update, reset and output gate. */
    mkldnn_ldgoi = mkldnn_abdec,
    /** 4D RNN bias tensor in the format (num_layers, num_directions,
     * num_gates, output_channels).
     *
     *  - For LSTM cells, the gates order is input, forget, candidate
     *    and output gate.
     *  - For GRU cells, the gates order is update, reset and output gate. */
    mkldnn_ldgo = mkldnn_abcd,

    /* Opaque data types, are not to be used explicitly */

    /* data */
    mkldnn_nCdhw16c = mkldnn_aBcde16b,
    mkldnn_nCdhw4c = mkldnn_aBcde4b,
    mkldnn_nCdhw8c = mkldnn_aBcde8b,
    mkldnn_nChw16c = mkldnn_aBcd16b,
    mkldnn_nChw4c = mkldnn_aBcd4b,
    mkldnn_nChw8c = mkldnn_aBcd8b,
    mkldnn_nCw16c = mkldnn_aBc16b,
    mkldnn_nCw4c = mkldnn_aBc4b,
    mkldnn_nCw8c = mkldnn_aBc8b,

    /* weights, 3D */
    mkldnn_IOw16o16i = mkldnn_BAc16a16b,
    mkldnn_OIw16i16o = mkldnn_ABc16b16a,
    mkldnn_OIw16o16i = mkldnn_ABc16a16b,
    mkldnn_Oiw16o = mkldnn_Abc16a,
    mkldnn_OIw4i16o4i = mkldnn_ABc4b16a4b,
    mkldnn_OIw4i4o = mkldnn_ABc4b4a,
    mkldnn_Oiw4o = mkldnn_Abc4a,
    mkldnn_OIw8i16o2i = mkldnn_ABc8b16a2b,
    mkldnn_OIw8i8o = mkldnn_ABc8b8a,
    mkldnn_OIw8o16i2o = mkldnn_ABc8a16b2a,
    mkldnn_OIw8o8i = mkldnn_ABc8a8b,
    mkldnn_Owi16o = mkldnn_Acb16a,
    mkldnn_Owi4o = mkldnn_Acb4a,
    mkldnn_Owi8o = mkldnn_Acb8a,

    /* weights, 4D */
    mkldnn_IOhw16o16i = mkldnn_BAcd16a16b,
    mkldnn_Ohwi16o = mkldnn_Acdb16a,
    mkldnn_Ohwi4o = mkldnn_Acdb4a,
    mkldnn_Ohwi8o = mkldnn_Acdb8a,
    mkldnn_OIhw16i16o = mkldnn_ABcd16b16a,
    mkldnn_OIhw16o16i = mkldnn_ABcd16a16b,
    mkldnn_Oihw16o = mkldnn_Abcd16a,
    mkldnn_OIhw4i16o4i = mkldnn_ABcd4b16a4b,
    mkldnn_OIhw4i4o = mkldnn_ABcd4b4a,
    mkldnn_Oihw4o = mkldnn_Abcd4a,
    mkldnn_OIhw8i16o2i = mkldnn_ABcd8b16a2b,
    mkldnn_OIhw8i8o = mkldnn_ABcd8b8a,
    mkldnn_OIhw8o16i2o = mkldnn_ABcd8a16b2a,
    mkldnn_OIhw8o8i = mkldnn_ABcd8a8b,

    /* weights, 5D */
    mkldnn_Odhwi16o = mkldnn_Acdeb16a,
    mkldnn_Odhwi4o = mkldnn_Acdeb4a,
    mkldnn_Odhwi8o = mkldnn_Acdeb8a,
    mkldnn_OIdhw16i16o = mkldnn_ABcde16b16a,
    mkldnn_OIdhw16o16i = mkldnn_ABcde16a16b,
    mkldnn_Oidhw16o = mkldnn_Abcde16a,
    mkldnn_OIdhw4i4o = mkldnn_ABcde4b4a,
    mkldnn_Oidhw4o = mkldnn_Abcde4a,
    mkldnn_OIdhw8i16o2i = mkldnn_ABcde8b16a2b,
    mkldnn_OIdhw8i8o = mkldnn_ABcde8b8a,
    mkldnn_OIdhw8o8i = mkldnn_ABcde8a8b,

    /* weights w/ groups, 3D */
    mkldnn_Goiw16g = mkldnn_Abcd16a,
    mkldnn_gIOw16o16i = mkldnn_aCBd16b16c,
    mkldnn_gOIw16i16o = mkldnn_aBCd16c16b,
    mkldnn_gOIw16o16i = mkldnn_aBCd16b16c,
    mkldnn_gOiw16o = mkldnn_aBcd16b,
    mkldnn_gOIw4i16o4i = mkldnn_aBCd4c16b4c,
    mkldnn_gOIw4i4o = mkldnn_aBCd4c4b,
    mkldnn_gOiw4o = mkldnn_aBcd4b,
    mkldnn_gOIw8i16o2i = mkldnn_aBCd8c16b2c,
    mkldnn_gOIw8i8o = mkldnn_aBCd8c8b,
    mkldnn_gOIw8o16i2o = mkldnn_aBCd8b16c2b,
    mkldnn_gOIw8o8i = mkldnn_aBCd8b8c,
    mkldnn_gOwi16o = mkldnn_aBdc16b,
    mkldnn_gOwi4o = mkldnn_aBdc4b,
    mkldnn_gOwi8o = mkldnn_aBdc8b,

    /* weights w/ groups, 4D */
    mkldnn_gIOhw16o16i = mkldnn_aCBde16b16c,
    mkldnn_gOhwi16o = mkldnn_aBdec16b,
    mkldnn_gOhwi4o = mkldnn_aBdec4b,
    mkldnn_gOhwi8o = mkldnn_aBdec8b,
    mkldnn_Goihw16g = mkldnn_Abcde16a,
    mkldnn_gOIhw16i16o = mkldnn_aBCde16c16b,
    mkldnn_gOIhw16o16i = mkldnn_aBCde16b16c,
    mkldnn_gOihw16o = mkldnn_aBcde16b,
    mkldnn_gOIhw2i8o4i = mkldnn_aBCde2c8b4c,
    mkldnn_gOIhw4i16o4i = mkldnn_aBCde4c16b4c,
    mkldnn_gOIhw4i4o = mkldnn_aBCde4c4b,
    mkldnn_gOIhw4o4i = mkldnn_aBCde4b4c,
    mkldnn_gOihw4o = mkldnn_aBcde4b,
    mkldnn_Goihw8g = mkldnn_Abcde8a,
    mkldnn_gOIhw8i16o2i = mkldnn_aBCde8c16b2c,
    mkldnn_gOIhw8i8o = mkldnn_aBCde8c8b,
    mkldnn_gOIhw8o16i2o = mkldnn_aBCde8b16c2b,
    mkldnn_gOIhw8o8i = mkldnn_aBCde8b8c,

    /* weights w/ groups, 6D */
    mkldnn_gOdhwi16o = mkldnn_aBdefc16b,
    mkldnn_gOdhwi4o = mkldnn_aBdefc4b,
    mkldnn_gOdhwi8o = mkldnn_aBdefc8b,
    mkldnn_gOIdhw16i16o = mkldnn_aBCdef16c16b,
    mkldnn_gOIdhw16o16i = mkldnn_aBCdef16b16c,
    mkldnn_gOidhw16o = mkldnn_aBcdef16b,
    mkldnn_gOIdhw4i4o = mkldnn_aBCdef4c4b,
    mkldnn_gOidhw4o = mkldnn_aBcdef4b,
    mkldnn_gOIdhw8i16o2i = mkldnn_aBCdef8c16b2c,
    mkldnn_gOIdhw8i8o = mkldnn_aBCdef8c8b,
    mkldnn_gOIdhw8o8i = mkldnn_aBCdef8b8c,
} mkldnn_format_tag_t;

/** Kinds of padding. Define how to interpret the data in padding regions. */
typedef enum {
    /** The data in padding regions is zero. */
    mkldnn_padding_zero,
} mkldnn_padding_kind_t;

/** Kinds of propagation. */
typedef enum {
    /* TODO: suggest renames */
    /** Undefined propagation type. */
    mkldnn_prop_kind_undef = 0,
    /** Forward data propagation (training mode). In this mode primitives
     * perform computations necessary for subsequent backward propagation. */
    mkldnn_forward_training = 64,
    /** Forward data propagation (inference mode). In this mode primitives
     * perform only computations that are necessary for inference and omit
     * computations that are necessary only for backward propagation. */
    mkldnn_forward_inference = 96,
    /** Forward data propagation (alias for @c mkldnn_forward_inference) */
    mkldnn_forward_scoring = mkldnn_forward_inference,
   /** Forward data propagation (alias for @c mkldnn_forward_training) */
    mkldnn_forward = mkldnn_forward_training,
    /** Backward propagation (with respect to all parameters */
    mkldnn_backward = 128,
    /** Backward data propagation */
    mkldnn_backward_data = 160,
    /** Backward weights propagation */
    mkldnn_backward_weights = 192,
    /** Backward bias propagation */
    mkldnn_backward_bias = 193,
} mkldnn_prop_kind_t;

/** Kinds of primitives. Used to implement a way to extend the library with new
 * primitives without changing the ABI. */
typedef enum {
    /** Undefined primitive (XXX: why do we have it?). */
    mkldnn_undefined_primitive,
    /** A reorder primitive.*/
    mkldnn_reorder,
    /** A shuffle primitive.*/
    mkldnn_shuffle,
    /** A (out-of-place) concat primitive. */
    mkldnn_concat,
    /** A sum primitive. */
    mkldnn_sum,
    /** A convolution primitive. */
    mkldnn_convolution,
    /** A deconvolution primitive. */
    mkldnn_deconvolution,
    /** An element-wise primitive. */
    mkldnn_eltwise,
    /** A Softmax primitive. */
    mkldnn_softmax,
    /** A pooling primitive. */
    mkldnn_pooling,
    /** An LRN primitive. */
    mkldnn_lrn,
    /** An batch normalization primitive. */
    mkldnn_batch_normalization,
    /** An inner product primitive. */
    mkldnn_inner_product,
    /** A rnn primitive. */
    mkldnn_rnn,
} mkldnn_primitive_kind_t;

/** Kinds of algorithms. */
typedef enum {
    mkldnn_alg_kind_undef,
    /** Direct convolution */
    mkldnn_convolution_direct = 0x1,
    /** Winograd convolution */
    mkldnn_convolution_winograd = 0x2,
    /** Convolution algorithm(either direct or Winograd) is chosen just in time **/
    mkldnn_convolution_auto = 0x3,
    /** Direct deconvolution */
    mkldnn_deconvolution_direct = 0xa,
    /** Winograd deconvolution */
    mkldnn_deconvolution_winograd = 0xb,
    /** Eltwise: ReLU */
    mkldnn_eltwise_relu = 0x1f,
    /** Eltwise: hyperbolic tangent non-linearity (tanh) */
    mkldnn_eltwise_tanh = 0x2f,
    /** Eltwise: parametric exponential linear unit (elu) */
    mkldnn_eltwise_elu = 0x3f,
    /** Eltwise: square */
    mkldnn_eltwise_square = 0x4f,
    /** Eltwise: abs */
    mkldnn_eltwise_abs = 0x5f,
    /** Eltwise: square root */
    mkldnn_eltwise_sqrt = 0x6f,
    /** Eltwise: linear */
    mkldnn_eltwise_linear = 0x7f,
    /** Eltwise: bounded_relu */
    mkldnn_eltwise_bounded_relu = 0x8f,
    /** Eltwise: soft_relu */
    mkldnn_eltwise_soft_relu = 0x9f,
    /** Eltwise: logistic */
    mkldnn_eltwise_logistic = 0xaf,
    /** Max pooling */
    mkldnn_pooling_max = 0x1ff,
    /** Average pooling include padding */
    mkldnn_pooling_avg_include_padding = 0x2ff,
    /** Average pooling exclude padding */
    mkldnn_pooling_avg_exclude_padding = 0x3ff,
    mkldnn_pooling_avg = mkldnn_pooling_avg_exclude_padding,
    /** Local response normalization (LRN) across multiple channels */
    mkldnn_lrn_across_channels = 0xaff,
    /** LRN within a single channel */
    mkldnn_lrn_within_channel = 0xbff,
    /** RNN cell */
    mkldnn_vanilla_rnn = 0x1fff,
    /** LSTM cell */
    mkldnn_vanilla_lstm = 0x2fff,
    /** GRU cell */
    mkldnn_vanilla_gru = 0x3fff,
    /** GRU cell with linear before reset
     *
     * Modification of original GRU cell. Differs from #mkldnn_vanilla_gru
     * in how the new memory gate is calculated:
     * \f[ c_t = tanh(W_c*x_t + b_{c_x} + r_t*(U_c*h_{t-1}+b_{c_h})) \f]
     * Primitive expects 4 biases on input:
     * \f$[b_{u}, b_{r}, b_{c_x}, b_{c_h}]\f$
     * */
    mkldnn_gru_linear_before_reset = 0x4fff,
} mkldnn_alg_kind_t;

/** Flags for batch-normalization primititve. */
typedef enum {
    /** Use global statistics
     *
     * If specified
     *  - on forward propagation use mean and variance provided by user (input)
     *  - on backward propagation reduces the amount of computations, since
     *    mean and variance are considered as constants
     *
     *  If not specified:
     *   - on forward propagation mean and variance are computed and stored in
     *     output
     *   - on backward propagation compute full derivative wrt to data
     */
    mkldnn_use_global_stats = 0x1U,
    /** Use scale and shift parameters
     *
     * If specified:
     *  - on forward propagation use scale and shift (aka scale and bias) for
     *    the batch normalization results
     *  - on backward propagation (for prop_kind == #mkldnn_backward) compute
     *    diff wrt to scale and shift (hence one extra output used)
     *
     * If no specified:
     *  - on backward propagation prop_kind == #mkldnn_backward_data has the
     *    same behavior as prop_kind == #mkldnn_backward
     */
    mkldnn_use_scaleshift = 0x2U,
    /** Fuse with ReLU
     *
     * If specified:
     *  - on inference this option behaves the same as if the primitive were
     *    fused with ReLU via post ops API
     *  - on training primitive requires workspace (required to be able to
     *    perform backward pass)
     */
    mkldnn_fuse_bn_relu = 0x4U,
} mkldnn_batch_normalization_flag_t;

/** @} */

/** @addtogroup c_api_types_memory Memory
 *  @{ */

/** Maximum number of dimensions a tensor can have. Only restricts the amount
 * of space used for the tensor description. Individual computational
 * primitives may support only tensors of certain dimensions. */
#define MKLDNN_MAX_NDIMS 12

/** A type to describe tensor dimension. */
typedef int64_t mkldnn_dim_t;

/** A type to describe tensor dimensions. */
typedef mkldnn_dim_t mkldnn_dims_t[MKLDNN_MAX_NDIMS];

/** A type to describe strides within a tensor. */
typedef mkldnn_dim_t mkldnn_strides_t[MKLDNN_MAX_NDIMS];

/** Generic description of blocked data layout for most memory formats.
 *
 * @sa @ref understanding_memory_formats */
typedef struct {
    /** The strides between the outermost blocks.
     * In case of plain (non-blocked) formats the strides between dimensions. */
    mkldnn_dims_t strides;
    /* Innermost section
     * ASSUMPTION: the innermost blocks are always dense */
    /** The number of innermost blocks, e.g. 3 in case of `OIhw_4i16o4i_` */
    int inner_nblks;
    /** The size of the blocks, e.g. `{4, 16, 4}` in case of `OIhw_4i16o4i` */
    mkldnn_dims_t inner_blks;
    /** The logical indices of the blocks, e.g. `{1, 0, 1}` in case of
     * `4i16o4i`, because `i` is the 1st dim and `o` is the 0st dim */
    mkldnn_dims_t inner_idxs;
} mkldnn_blocking_desc_t;

typedef enum {
    /** Undefined memory format, used for empty memory descriptors. */
    mkldnn_wino_undef = 0,
    /** Tensors of weights for 2x3 winograd convolutions. */
    mkldnn_wino_wei_aaOIoi,
    mkldnn_wino_wei_aaOio,
    mkldnn_wino_wei_aaOBiOo,
    /** Tensor of weights for 4x3 convolution. */
    mkldnn_wino_wei_OBaaIBOIio
} mkldnn_wino_memory_format_t;

/** Description of tensor of weights for winograd 2x3 convolution. */
typedef struct {
    mkldnn_wino_memory_format_t wino_format;
    int r;
    int alpha;
    int ic;
    int oc;
    int ic_block;
    int oc_block;
    int ic2_block;
    int oc2_block;
    float adj_scale;
    size_t size;
} mkldnn_wino_desc_t;

typedef enum {
    mkldnn_packed_format_undef = 0,
    mkldnn_ldigo_p,
    mkldnn_ldgoi_p
} mkldnn_rnn_packed_memory_format_t;

/* Maximum number of parts of RNN weights tensor that require separate
 * computation. */
#define MKLDNN_RNN_MAX_N_PARTS 4

/** Description of tensor of packed weights for rnn. */
typedef struct {
    mkldnn_rnn_packed_memory_format_t format;
    int n_parts;
    int n;
    int parts[MKLDNN_RNN_MAX_N_PARTS];
    size_t part_pack_size[MKLDNN_RNN_MAX_N_PARTS];
    size_t offset_compensation;
    size_t size;
} mkldnn_rnn_packed_desc_t;

typedef enum {
    mkldnn_memory_extra_flag_none = 0x0U,
    /** Indicates the weights have an additional buffer, that depends on the
     * @p compensation_mask.
     *
     * For instance, in 4D case with the compensation mask equals (1 << 0)
     * the additional buffer would consist of OC values:
     * O[oc : 0,OC] =
     *  -128 * SUM(ic : 0,IC; kh : 0,KH; kw : 0,KW){ weights(oc, ic, kh, kw) }
     */
    mkldnn_memory_extra_flag_compensation_conv_s8s8 = 0x1U,
    mkldnn_memory_extra_flag_scale_adjust = 0x2U,
} mkldnn_memory_extra_flags_t;

/** Description of extra information stored in memory */
typedef struct {
    /** The flags contain arbitrary extra information, such as compensation.
     * @sa mkldnn_memory_extra_flags_t */
    uint64_t flags;
    /** Compensation mask */
    int compensation_mask;
    /** Scale applied to the data */
    float scale_adjust;
    /** For future backwards compatibility */
    char reserved[64];
} mkldnn_memory_extra_desc_t;

/** Memory descriptor. The description is based on a number of dimensions,
 * dimensions themselves, plus information about elements type and memory
 * format. Additionally, contains format-specific descriptions of the data
 * layout. */
typedef struct {
    /** Number of dimensions */
    int ndims;
    /** Dimensions in the following order:
     * - CNN data tensors: mini-batch, channel, spatial
     *   (<code>{N, C, [[D,] H,] W}</code>)
     * - CNN weight tensors: group (optional), output channel, input channel,
     *   spatial (<code>{[G,] O, I, [[D,] H,] W}</code>)
     * - RNN data tensors: time, mini-batch, channels (<code>{T, N, C}</code>)
     *   or layers, directions, states, mini-batch, channels (<code>{L, D, S, N, C}</code>)
     * - RNN weight tensor: layers, directions, input channel, gates, output channels
     *   (<code>{L, D, I, G, O}</code>).
     *
     * @note
     *    The order of dimensions does not depend on the memory format, so
     *    whether the data is laid out in #mkldnn_nchw or #mkldnn_nhwc
     *    the dims for 4D CN data tensor would be <code>{N, C, H, W}</code>.
     */
    mkldnn_dims_t dims;
    /** Data type of the tensor elements. */
    mkldnn_data_type_t data_type;

    /** Size of the data including padding in each dimension. */
    mkldnn_dims_t padded_dims;
    /** Per-dimension offset from the padding to actual data, the top-level
     * tensor with offsets applied must lie within the padding area. */
    mkldnn_dims_t padded_offsets;

    /** Offset from memory origin to the current block, non-zero only in
     * a description of a memory sub-block. */
    mkldnn_dim_t offset0;

    /** Memory format kind. */
    mkldnn_format_kind_t format_kind;
    union {
        /** Description of the data layout for memory formats that use
         * blocking. */
        mkldnn_blocking_desc_t blocking;
        /** Tensor of weights for integer 8bit winograd convolution. */
        mkldnn_wino_desc_t wino_desc;
        /** Tensor of packed weights for RNN. */
        mkldnn_rnn_packed_desc_t rnn_packed_desc;
        /* ... other descriptions possible */
    } format_desc;

    mkldnn_memory_extra_desc_t extra;
} mkldnn_memory_desc_t;

/** @struct mkldnn_memory
 * An opaque structure to describe a memory. */
struct mkldnn_memory;

/** A memory handle. */
typedef struct mkldnn_memory *mkldnn_memory_t;

/** A constant memory handle. */
typedef const struct mkldnn_memory *const_mkldnn_memory_t;

#define MKLDNN_NATIVE_HANDLE_NONE (NULL)
#define MKLDNN_NATIVE_HANDLE_ALLOCATE ((void *)(size_t)-1)

/** @} */

/** @addtogroup c_api_types_op_descs Operation descriptors
 *  @{*/

/** A pointer to any of the operation descriptors. */
typedef void *mkldnn_op_desc_t;
/** A pointer to any of the operation descriptors (constant variant). */
typedef const void *const_mkldnn_op_desc_t;

/** A descriptor of a convolution operation. */
typedef struct {
    /** The kind of primitive. Used for self-identifying the primitive
     * descriptor. Must be #mkldnn_convolution. */
    mkldnn_primitive_kind_t primitive_kind;
    /** The kind of propagation. Possible values: #mkldnn_forward_training,
     * #mkldnn_forward_inference, #mkldnn_backward_data,
     * #mkldnn_backward_weights, and #mkldnn_backward_bias. */
    mkldnn_prop_kind_t prop_kind;
    /** The kind of the convolution algorithm. Possible values:
     * #mkldnn_convolution_direct. */
    mkldnn_alg_kind_t alg_kind;
    /** Source memory descriptor. */
    mkldnn_memory_desc_t src_desc;
    /** Source gradient memory descriptor. */
    mkldnn_memory_desc_t diff_src_desc;
    /** Weights memory descriptor. */
    mkldnn_memory_desc_t weights_desc;
    /** Weights gradient memory descriptor. */
    mkldnn_memory_desc_t diff_weights_desc;
    /** Bias memory descriptor. */
    mkldnn_memory_desc_t bias_desc;
    /** Bias gradient memory descriptor. */
    mkldnn_memory_desc_t diff_bias_desc;
    /** Destination memory descriptor. */
    mkldnn_memory_desc_t dst_desc;
    /** Destination gradient memory descriptor. */
    mkldnn_memory_desc_t diff_dst_desc;
    /** Convolution strides in each spatial dimension. */
    mkldnn_dims_t strides;
    /** Convolution dilates in each spatial dimension. */
    mkldnn_dims_t dilates;
    /** Padding in each spatial dimension. padding[0] is a padding in the
     * beginning (@p padding_l), padding[1] is a padding in the end (@p
     * padding_r). */
    mkldnn_dims_t padding[2];
    /** The kind of padding to use. */
    mkldnn_padding_kind_t padding_kind;
    /** The accumulator data type. Initialized automatically. */
    mkldnn_data_type_t accum_data_type;
} mkldnn_convolution_desc_t;

/** A descriptor of a deconvolution operation. */
typedef mkldnn_convolution_desc_t mkldnn_deconvolution_desc_t;

/** A descriptor of a shuffle operation. */
typedef struct {
    /** The kind of primitive. Used for self-identifying the primitive
     * descriptor. Must be #mkldnn_convolution. */
    mkldnn_primitive_kind_t primitive_kind;
    /** The kind of propagation. Possible values: #mkldnn_forward_training,
     * #mkldnn_forward_inference, and #mkldnn_backward_data. */
    mkldnn_prop_kind_t prop_kind;
    /** Source and destination memory descriptor,
     *  and source and destination gradient memory descriptor. */
    mkldnn_memory_desc_t data_desc;
    /** axis for shuffling. */
    int axis;
    /** number of groups in group convolution */
    mkldnn_dim_t group_size;
} mkldnn_shuffle_desc_t;

/** A descriptor of a element-wise operation. */
typedef struct {
    /** The kind of primitive. Used for self-identifying the primitive
     * descriptor. Must be #mkldnn_eltwise. */
    mkldnn_primitive_kind_t primitive_kind;
    /** The kind of propagation. Possible values: #mkldnn_forward_training,
     * #mkldnn_forward_inference, #mkldnn_backward, and #mkldnn_backward_data.
     */
    mkldnn_prop_kind_t prop_kind;
    /** The kind of eltwise algorithm. Possible values: #mkldnn_eltwise_relu,
     * #mkldnn_eltwise_tanh, #mkldnn_eltwise_elu, #mkldnn_eltwise_square,
     * #mkldnn_eltwise_abs, #mkldnn_eltwise_sqrt, #mkldnn_eltwise_linear,
     * #mkldnn_eltwise_bounded_relu, #mkldnn_eltwise_soft_relu, and
     * #mkldnn_eltwise_logistic. */
    mkldnn_alg_kind_t alg_kind;
    /** Source and destination memory descriptor. */
    mkldnn_memory_desc_t data_desc;
    /** Source and destination gradient memory descriptor. */
    mkldnn_memory_desc_t diff_data_desc;
    /** Algorithm specific parameter.
     * Accordance table:
     *  - #mkldnn_eltwise_relu: @p alpha -- negative slope, @p beta ignored
     *  - #mkldnn_eltwise_tanh: @p alpha and @p beta ignored
     *  - #mkldnn_eltwise_elu: @p alpha -- negative slope, @p beta ignored
     *  - #mkldnn_eltwise_square: @p alpha and @p beta ignored
     *  - #mkldnn_eltwise_abs: @p alpha and @p beta ignored
     *  - #mkldnn_eltwise_sqrt: @p alpha and @p beta ignored
     *  - #mkldnn_eltwise_linear: @p alpha -- scale, @p beta -- shift
     *  - #mkldnn_eltwise_bounded_relu: @p alpha -- upper bound, @p beta ignored
     *  - #mkldnn_eltwise_soft_relu: @p alpha and @p beta ignored
     *  - #mkldnn_eltwise_logistic: @p alpha and @p beta ignored
     */
    float alpha, beta;
} mkldnn_eltwise_desc_t;

/** A descriptor of a Softmax operation. */
typedef struct {
    /** The kind of primitive. Used for self-identifying the primitive
    * descriptor. Must be #mkldnn_softmax. */
    mkldnn_primitive_kind_t primitive_kind;
    /** The kind of propagation. Possible values: #mkldnn_forward_training and
     * #mkldnn_forward_inference. */
    mkldnn_prop_kind_t prop_kind;
    /** Source and destination memory descriptor. */
    mkldnn_memory_desc_t data_desc;
    /** Source and Destination of gradient memory descriptor. */
    mkldnn_memory_desc_t diff_desc;
    /** The axis along which to perform the softmax. */
    int softmax_axis;
} mkldnn_softmax_desc_t;

/** A descriptor of a pooling operation. */
typedef struct {
    /** The kind of primitive. Used for self-identifying the primitive
     * descriptor. Must be #mkldnn_pooling. */
    mkldnn_primitive_kind_t primitive_kind;
    /** The kind of propagation. Possible values: #mkldnn_forward_training,
     * #mkldnn_forward_inference, #mkldnn_backward, and #mkldnn_backward_data.
     */
    mkldnn_prop_kind_t prop_kind;
    /** The kind of pooling algorithm. Possible values: #mkldnn_pooling_max and
     * #mkldnn_pooling_avg. */
    mkldnn_alg_kind_t alg_kind;
    /** Source memory descriptor. */
    mkldnn_memory_desc_t src_desc;
    /** Source gradient memory descriptor. */
    mkldnn_memory_desc_t diff_src_desc;
    /** Destination memory descriptor. */
    mkldnn_memory_desc_t dst_desc;
    /** Destination gradient memory descriptor. */
    mkldnn_memory_desc_t diff_dst_desc;
    /** Pooling kernel strides for spatial dimensions. */
    mkldnn_dims_t strides;
    /** Pooling kernel spatial dimensions. */
    mkldnn_dims_t kernel;
    /** Padding in each spatial dimension. padding[0] is a padding in the
     * beginning (@p padding_l), padding[1] is a padding in the end (@p
     * padding_r). */
    mkldnn_dims_t padding[2];
    /** The kind of padding to use. */
    mkldnn_padding_kind_t padding_kind;
    /** The accumulator data type. Initialized automatically. */
    mkldnn_data_type_t accum_data_type;
} mkldnn_pooling_desc_t;

/** A descriptor of a Local Response Normalization (LRN) operation. */
typedef struct {
    /** The kind of primitive. Used for self-identifying the primitive
     * descriptor. Must be #mkldnn_lrn. */
    mkldnn_primitive_kind_t primitive_kind;
    /** The kind of propagation. Possible values: #mkldnn_forward_training,
     * #mkldnn_forward_inference, #mkldnn_backward, and #mkldnn_backward_data.
     */
    mkldnn_prop_kind_t prop_kind;
    /** LRN algorithm. Possible values: #mkldnn_lrn_within_channel and
     * #mkldnn_lrn_across_channels. */
    mkldnn_alg_kind_t alg_kind;
    /** Source and destination memory descriptor. */
    mkldnn_memory_desc_t data_desc;
    /** Source and destination gradient memory descriptor. */
    mkldnn_memory_desc_t diff_data_desc;
    /** The number of channels to sum over (for cross-channel LRN) or the side
     * length of the square region to sum over (for within-channel LRN). */
    mkldnn_dim_t local_size;
    /** LRN alpha parameter. */
    float lrn_alpha;
    /** LRN beta parameter. */
    float lrn_beta;
    /** LRN k parameter. */
    float lrn_k;
} mkldnn_lrn_desc_t;

/** A descriptor of a Batch Normalization operation. */
typedef struct {
    /** The kind of primitive. Used for self-identifying the primitive
     * descriptor. Must be #mkldnn_batch_normalization. */
    mkldnn_primitive_kind_t primitive_kind;
    /** The kind of propagation. Possible values: #mkldnn_forward_training,
     * #mkldnn_forward_inference, #mkldnn_backward, and #mkldnn_backward_data.
     */
    mkldnn_prop_kind_t prop_kind;
    /** Source and destination memory descriptor. */
    mkldnn_memory_desc_t data_desc;
    /** Source and destination gradient memory descriptor. */
    mkldnn_memory_desc_t diff_data_desc;
    /** Scale and shift data and gradient memory descriptors.
     *
     * Scaleshift memory descriptor uses 2D #mkldnn_nc format[2,Channels]. 1-st
     * dimension contains gamma parameter, 2-nd dimension contains beta
     * parameter. */
    mkldnn_memory_desc_t data_scaleshift_desc;
    mkldnn_memory_desc_t diff_data_scaleshift_desc;
    /** Mean and variance data memory descriptors.
     *
     * Mean and variance memory descriptors use 1D #mkldnn_x format[Channels].
     */
    mkldnn_memory_desc_t mean_desc;
    mkldnn_memory_desc_t variance_desc;
    /** Batch normalization epsilon parameter. */
    float batch_norm_epsilon;
    unsigned flags;
} mkldnn_batch_normalization_desc_t;

/** A descriptor of an inner product operation. */
typedef struct {
    /** The kind of primitive. Used for self-identifying the primitive
     * descriptor. Must be #mkldnn_inner_product. */
    mkldnn_primitive_kind_t primitive_kind;
    /** The kind of propagation. Possible values: #mkldnn_forward_training,
     * #mkldnn_forward_inference, #mkldnn_backward_data,
     * #mkldnn_backward_weights, and #mkldnn_backward_bias. */
    mkldnn_prop_kind_t prop_kind;
    /** Source memory descriptor. */
    mkldnn_memory_desc_t src_desc;
    /** Source gradient memory descriptor. */
    mkldnn_memory_desc_t diff_src_desc;
    /** Weights memory descriptor. */
    mkldnn_memory_desc_t weights_desc;
    /** Weights gradient memory descriptor. */
    mkldnn_memory_desc_t diff_weights_desc;
    /** Bias memory descriptor. */
    mkldnn_memory_desc_t bias_desc;
    /** Bias gradient memory descriptor. */
    mkldnn_memory_desc_t diff_bias_desc;
    /** Destination memory descriptor. */
    mkldnn_memory_desc_t dst_desc;
    /** Destination gradient memory descriptor. */
    mkldnn_memory_desc_t diff_dst_desc;
    /** The accumulator data type. Initialized automatically. */
    mkldnn_data_type_t accum_data_type;
} mkldnn_inner_product_desc_t;

/** Flags for RNN cell. */
typedef enum {
    mkldnn_rnn_cell_with_relu = 0x1U,
    mkldnn_rnn_cell_with_clipping = 0x2U,
} mkldnn_rnn_cell_flags_t;

typedef struct {
    /** RNN cell kind. Must be one of #mkldnn_vanilla_rnn,
     * #mkldnn_vanilla_lstm, #mkldnn_vanilla_gru,
     * or #mkldnn_gru_linear_before_reset. */
    mkldnn_alg_kind_t cell_kind;
    /** Activation function used. Must be either #mkldnn_eltwise_relu or
     * #mkldnn_eltwise_tanh. */
    mkldnn_alg_kind_t activation_kind;
    /** RNN cell flags */
    unsigned int flags;
    /** @c alpha is a negative slope parameter (used only if
     * `(flags & #mkldnn_rnn_cell_with_relu) != 0`) */
    float alpha;
    /** clipping parameter (used only if
     * `(flags & #mkldnn_rnn_cell_with_clipping) != 0`) */
    float clipping;
} mkldnn_rnn_cell_desc_t;

/** A direction of RNN primitive execution. */
typedef enum {
    /* Unidirectional execution of RNN primitive from left to right. */
    mkldnn_unidirectional_left2right,
    /* Unidirectional execution of RNN primitive from right to left. */
    mkldnn_unidirectional_right2left,
    /* Bidirectional execution of RNN primitive with concatenation of the
     * results. */
    mkldnn_bidirectional_concat,
    /* Bidirectional execution of RNN primitive with summation of the
     * results. */
    mkldnn_bidirectional_sum,
    mkldnn_unidirectional = mkldnn_unidirectional_left2right,
} mkldnn_rnn_direction_t;

/** A descriptor for an RNN operation. */
typedef struct {
    /** The kind of primitive. Used for self-identifying the primitive
     * descriptor. Must be #mkldnn_rnn. */
    mkldnn_primitive_kind_t primitive_kind;
    /** The kind of propagation. Possible values: #mkldnn_forward_training,
     * #mkldnn_forward_inference, and #mkldnn_backward. */
    mkldnn_prop_kind_t prop_kind;
    /** The RNN cell desc. */
    mkldnn_rnn_cell_desc_t cell_desc;
    /** The direction of RNN primitive execution. */
    mkldnn_rnn_direction_t direction;
    /** Source layer memory descriptor. */
    mkldnn_memory_desc_t src_layer_desc;
    /** Source iteration memory descriptor. */
    mkldnn_memory_desc_t src_iter_desc;
    /** Weights layer memory descriptor. */
    mkldnn_memory_desc_t weights_layer_desc;
    /** Weights iteration memory descriptor. */
    mkldnn_memory_desc_t weights_iter_desc;
    /** Bias memory descriptor. */
    mkldnn_memory_desc_t bias_desc;
    /** Destination layer memory descriptor. */
    mkldnn_memory_desc_t dst_layer_desc;
    /** Destination iter memory descriptor. */
    mkldnn_memory_desc_t dst_iter_desc;
    /** Source gradient layer memory descriptor. */
    mkldnn_memory_desc_t diff_src_layer_desc;
    /** Source gradient iter memory descriptor. */
    mkldnn_memory_desc_t diff_src_iter_desc;
    /** Weights gradient layer memory descriptor. */
    mkldnn_memory_desc_t diff_weights_layer_desc;
    /** Weights gradient iter memory descriptor. */
    mkldnn_memory_desc_t diff_weights_iter_desc;
    /** Bias gradient memory descriptor. */
    mkldnn_memory_desc_t diff_bias_desc;
    /** Destination gradient layer memory descriptor. */
    mkldnn_memory_desc_t diff_dst_layer_desc;
    /** Destination gradient iteration memory descriptor. */
    mkldnn_memory_desc_t diff_dst_iter_desc;
} mkldnn_rnn_desc_t;

/** @} */

/** @addtogroup c_api_engine_types Engine
 * @{ */

/** @brief Kinds of engines. */
typedef enum {
    /** An unspecified engine. */
    mkldnn_any_engine,
    /** CPU engine. */
    mkldnn_cpu,
} mkldnn_engine_kind_t;

/** @struct mkldnn_engine
 * @brief An opaque structure to describe an engine. */
struct mkldnn_engine;
/** @brief An engine handle. */
typedef struct mkldnn_engine *mkldnn_engine_t;
#if 0
/* FIXME: looks like this never happens */
/** @brief A constant engine handle. */
typedef const struct mkldnn_engine *const_mkldnn_engine_t;
#endif

/** @} */

/** @addtogroup c_api_primitive_desc_iterators Primitive descriptor iterators
 * @{ */

/** @struct mkldnn_primitive_desc_iterator
 * @brief An opaque structure to describe a primitive descriptor iterator. */
struct mkldnn_primitive_desc_iterator;

/** @brief A primitive descriptor iterator handle. */
typedef struct mkldnn_primitive_desc_iterator
    *mkldnn_primitive_desc_iterator_t;

/** @brief A constant primitive descriptor iterator handle. */
typedef const struct mkldnn_primitive_desc_iterator
    *const_mkldnn_primitive_desc_iterator_t;

/** @} */

/** @addtogroup c_api_primitive_descs Primitive descriptors
 * @{ */

/** @struct mkldnn_primitive_desc
 * @brief An opaque structure to describe a primitive descriptor. */
struct mkldnn_primitive_desc;

/** @brief A primitive descriptor handle. */
typedef struct mkldnn_primitive_desc *mkldnn_primitive_desc_t;

/** @brief A constant primitive descriptor handle. */
typedef const struct mkldnn_primitive_desc *const_mkldnn_primitive_desc_t;

/** @} */

/** @addtogroup c_api_primitive_attr Primitive descriptor attributes
 * @{ */

/** Scratchpad mode */
typedef enum {
    /** The library manages scratchpad (default) */
    mkldnn_scratchpad_mode_library,
    /** A user shall query and provide the scratchpad memory to primitives */
    mkldnn_scratchpad_mode_user,
} mkldnn_scratchpad_mode_t;

/** @struct mkldnn_primitive_attr
 * @brief An opaque structure for primitive descriptor attributes.
 *
 * Attributes may contain:
 *  - output scales (to scale the result prior to storing it to the memory)
 */
struct mkldnn_primitive_attr;

/** @brief A primitive descriptor attributes handle that controls primitive
 * behavior. */
typedef struct mkldnn_primitive_attr *mkldnn_primitive_attr_t;

/** @brief A constant primitive descriptor attributes handle. */
typedef const struct mkldnn_primitive_attr *const_mkldnn_primitive_attr_t;

/** @struct mkldnn_post_ops
 * @brief An opaque structure for a chain of post operations.
 *
 * mkldnn_post_ops can be used to perform some (trivial) operations like
 * accumulation or eltwise after certain primitives like convolution.
 *
 * Post operations might be combined together, making a chain of post
 * operations. For instance one can configure convolution followed by
 * accumulation followed by eltwise. This might be especially beneficial
 * for residual learning blocks.
 *
 * @warning
 *      Of course not all combinations are supported, so the user should handle
 *      errors accordingly.
 *
 * Supported post operations:
 *  - accumulation (base primitive: convolution)
 *  - eltwise (base primitive: convolution)
 */
struct mkldnn_post_ops;

/** @brief A post operation chain handle. */
typedef struct mkldnn_post_ops *mkldnn_post_ops_t;

/** @brief A constant post operation chain handle. */
typedef const struct mkldnn_post_ops *const_mkldnn_post_ops_t;

/** @} */

/** @addtogroup c_api_types_primitive Primitive
 * @{ */

/** @struct mkldnn_primitive
 * An opaque structure to describe a primitive. */
struct mkldnn_primitive;
/** A primitive handle. */
typedef struct mkldnn_primitive *mkldnn_primitive_t;
/** A constant primitive handle. */
typedef const struct mkldnn_primitive *const_mkldnn_primitive_t;

/** @addtogroup c_api_types_arguments Argument indices
 * @{ */

#define MKLDNN_ARG_SRC_0                1
#define MKLDNN_ARG_SRC                  MKLDNN_ARG_SRC_0
#define MKLDNN_ARG_SRC_LAYER            MKLDNN_ARG_SRC_0
#define MKLDNN_ARG_FROM                 MKLDNN_ARG_SRC_0

#define MKLDNN_ARG_SRC_1                2
#define MKLDNN_ARG_SRC_ITER             MKLDNN_ARG_SRC_1

#define MKLDNN_ARG_DST_0                17
#define MKLDNN_ARG_DST                  MKLDNN_ARG_DST_0
#define MKLDNN_ARG_TO                   MKLDNN_ARG_DST_0
#define MKLDNN_ARG_DST_LAYER            MKLDNN_ARG_DST_0

#define MKLDNN_ARG_DST_1                18
#define MKLDNN_ARG_DST_ITER             MKLDNN_ARG_DST_1

#define MKLDNN_ARG_WEIGHTS_0            33
#define MKLDNN_ARG_WEIGHTS              MKLDNN_ARG_WEIGHTS_0
#define MKLDNN_ARG_SCALE_SHIFT          MKLDNN_ARG_WEIGHTS_0
#define MKLDNN_ARG_WEIGHTS_LAYER        MKLDNN_ARG_WEIGHTS_0

#define MKLDNN_ARG_WEIGHTS_1            34
#define MKLDNN_ARG_WEIGHTS_ITER         MKLDNN_ARG_WEIGHTS_1

#define MKLDNN_ARG_BIAS                 41

#define MKLDNN_ARG_MEAN                 49
#define MKLDNN_ARG_VARIANCE             50

#define MKLDNN_ARG_WORKSPACE            64
#define MKLDNN_ARG_SCRATCHPAD           80

#define MKLDNN_ARG_DIFF_SRC_0           129
#define MKLDNN_ARG_DIFF_SRC             MKLDNN_ARG_DIFF_SRC_0
#define MKLDNN_ARG_DIFF_SRC_LAYER       MKLDNN_ARG_DIFF_SRC_0

#define MKLDNN_ARG_DIFF_SRC_1           130
#define MKLDNN_ARG_DIFF_SRC_ITER        MKLDNN_ARG_DIFF_SRC_1

#define MKLDNN_ARG_DIFF_DST_0           145
#define MKLDNN_ARG_DIFF_DST             MKLDNN_ARG_DIFF_DST_0
#define MKLDNN_ARG_DIFF_DST_LAYER       MKLDNN_ARG_DIFF_DST_0

#define MKLDNN_ARG_DIFF_DST_1           146
#define MKLDNN_ARG_DIFF_DST_ITER        MKLDNN_ARG_DIFF_DST_1

#define MKLDNN_ARG_DIFF_WEIGHTS_0       161
#define MKLDNN_ARG_DIFF_WEIGHTS         MKLDNN_ARG_DIFF_WEIGHTS_0
#define MKLDNN_ARG_DIFF_SCALE_SHIFT     MKLDNN_ARG_DIFF_WEIGHTS_0
#define MKLDNN_ARG_DIFF_WEIGHTS_LAYER   MKLDNN_ARG_DIFF_WEIGHTS_0

#define MKLDNN_ARG_DIFF_WEIGHTS_1       162
#define MKLDNN_ARG_DIFF_WEIGHTS_ITER    MKLDNN_ARG_DIFF_WEIGHTS_1

#define MKLDNN_ARG_DIFF_BIAS            169

#define MKLDNN_ARG_MULTIPLE_SRC         1024
#define MKLDNN_ARG_MULTIPLE_DST         2048

/** @} */

/** An auxiliary structure to specify primitive's inputs/outputs at execution
 *
 * @warning
 *      With this API it's impossible to preserve constness of memory, so all
 *      memories are passed w/o const qualifier. However only memories with
 *      output semantics might be changed during the execution */
typedef struct {
    int arg; /**< An argument index, e.g. MKLDNN_ARG_SRC */
    mkldnn_memory_t memory; /**< Input/output memory */
} mkldnn_exec_arg_t;

/** @} */

/** @addtogroup c_api_types_query Queries
 * @{ */

/** Primitive descriptor query specification
 *
 * For generic function mkldnn_primitive_desc_query(), the type of result must
 * agree with the queried argument. The correspondence table:
 *      Query                           | type of result
 *      --------------------------------------------------------------
 *      #mkldnn_query_engine            | mkldnn_engine_t *
 *      #mkldnn_query_scratchpad_engine | mkldnn_engine_t *
 *      #mkldnn_query_primitive_kind    | mkldnn_primitive_kind_t *
 *      *_s32                           | int *
 *      *_s64                           | mkldnn_dim_t * (same as int64_t *)
 *      *_f64                           | double *
 *      *_str                           | const char **
 *      #mkldnn_query_op_d              | const_mkldnn_op_desc_t *
 *      *_md                            | const mkldnn_memory_desc_t **
 *      *_${op}_d                       | const mkldnn_${op}_desc_t **
 *      *_pd                            | const_mkldnn_primitive_desc_t *
 *
 * @note
 *     Rule of thumb: all opaque types and structures are returned by
 *     reference. All numbers are returned by value.
 *
 * @warning
 *     All returned references point to constant objects and are valid only
 *     during the lifetime of the queried primitive descriptor. Returned objects
 *     must not be destroyed by the user. If you need to keep the object longer
 *     than the lifetime of the queried primitive descriptor, use
 *     mkldnn_primitive_desc_clone() to make a copy. */
typedef enum {
    mkldnn_query_undef = 0,  /**< no query */

    mkldnn_query_engine, /**< execution engine */
    mkldnn_query_primitive_kind, /**< primitive kind */

    mkldnn_query_num_of_inputs_s32, /**< number of inputs expected */
    mkldnn_query_num_of_outputs_s32, /**< number of outputs expected */

    mkldnn_query_time_estimate_f64, /**< runtime estimation (seconds) */
    mkldnn_query_memory_consumption_s64, /**< memory consumption -- extra
                                           (scratch) memory, additional to all
                                           inputs and outputs memory (bytes) */

    mkldnn_query_scratchpad_engine, /**< scratchpad engine -- engine to be used
                                       for creating scratchpad memory */

    mkldnn_query_impl_info_str, /**< implementation name */

    /* memory and op descriptor section */
    mkldnn_query_some_d = 64, /**< stub */
    mkldnn_query_op_d, /**< op descriptor */
    mkldnn_query_convolution_d, /**< convolution descriptor */
    mkldnn_query_deconvolution_d, /**< deconvolution descriptor */
    mkldnn_query_shuffle_d, /**< shuffle descriptor */
    mkldnn_query_eltwise_d, /**< eltwise descriptor */
    mkldnn_query_softmax_d, /**< softmax descriptor */
    mkldnn_query_pooling_d, /**< pooling descriptor */
    mkldnn_query_lrn_d, /**< lrn descriptor */
    mkldnn_query_batch_normalization_d, /**< batch normalization descriptor */
    mkldnn_query_inner_product_d, /**< inner product descriptor */
    mkldnn_query_rnn_d, /**< rnn descriptor */

    /* memory descriptor section */
    mkldnn_query_some_md = 128, /**< stub */
    mkldnn_query_src_md, /**< source memory desc */
    mkldnn_query_diff_src_md, /**< source gradient memory desc */
    mkldnn_query_weights_md, /**< weights memory descriptor desc */
    mkldnn_query_diff_weights_md, /**< weights grad. memory desc */
    mkldnn_query_dst_md, /**< destination memory desc */
    mkldnn_query_diff_dst_md, /**< destination grad. memory desc */
    mkldnn_query_workspace_md, /**< workspace memory desc */
    mkldnn_query_scratchpad_md, /**< scratchpad memory desc */
} mkldnn_query_t;

/** @} */

/** @addtogroup c_api_types_stream Execution stream
 * @{ */

/** @brief Stream flags. */
typedef enum {
    /** A default stream configuration. */
    mkldnn_stream_default_flags = 0x0U,
} mkldnn_stream_flags_t;

/** @struct mkldnn_stream
 * An opaque structure to describe an execution stream. */
struct mkldnn_stream;
/** An execution stream handle. */
typedef struct mkldnn_stream *mkldnn_stream_t;
/** A constant execution stream handle. */
typedef const struct mkldnn_stream *const_mkldnn_stream_t;

/** @} */
/** @} */
/** @} */

#ifdef __cplusplus
}
#endif


#endif
