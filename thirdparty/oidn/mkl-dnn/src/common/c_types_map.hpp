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

#ifndef TYPE_MAPPING_HPP
#define TYPE_MAPPING_HPP

#include "mkldnn_types.h"

namespace mkldnn {
namespace impl {

// TODO: autogenerate this

using dim_t = mkldnn_dim_t;
using dims_t = mkldnn_dims_t;
using stride_t = mkldnn_dim_t;
using strides_t = mkldnn_strides_t;

using status_t = mkldnn_status_t;
namespace status {
    const status_t success = mkldnn_success;
    const status_t out_of_memory = mkldnn_out_of_memory;
    const status_t try_again = mkldnn_try_again;
    const status_t invalid_arguments = mkldnn_invalid_arguments;
    const status_t not_ready = mkldnn_not_ready;
    const status_t unimplemented = mkldnn_unimplemented;
    const status_t iterator_ends = mkldnn_iterator_ends;
    const status_t runtime_error = mkldnn_runtime_error;
    const status_t not_required = mkldnn_not_required;
}

using prop_kind_t = mkldnn_prop_kind_t;
namespace prop_kind {
    const prop_kind_t undef = mkldnn_prop_kind_undef;
    const prop_kind_t forward_training = mkldnn_forward_training;
    const prop_kind_t forward_inference = mkldnn_forward_inference;
    const prop_kind_t forward_scoring = mkldnn_forward_scoring;
    const prop_kind_t forward = mkldnn_forward;
    const prop_kind_t backward = mkldnn_backward;
    const prop_kind_t backward_data = mkldnn_backward_data;
    const prop_kind_t backward_weights = mkldnn_backward_weights;
    const prop_kind_t backward_bias = mkldnn_backward_bias;
}

using alg_kind_t = mkldnn_alg_kind_t;
namespace alg_kind {
    const alg_kind_t undef = mkldnn_alg_kind_undef;
    const alg_kind_t convolution_auto = mkldnn_convolution_auto;
    const alg_kind_t convolution_direct = mkldnn_convolution_direct;
    const alg_kind_t convolution_winograd = mkldnn_convolution_winograd;
    const alg_kind_t deconvolution_direct = mkldnn_deconvolution_direct;
    const alg_kind_t deconvolution_winograd = mkldnn_deconvolution_winograd;
    const alg_kind_t eltwise_relu = mkldnn_eltwise_relu;
    const alg_kind_t eltwise_tanh = mkldnn_eltwise_tanh;
    const alg_kind_t eltwise_elu = mkldnn_eltwise_elu;
    const alg_kind_t eltwise_square = mkldnn_eltwise_square;
    const alg_kind_t eltwise_abs = mkldnn_eltwise_abs;
    const alg_kind_t eltwise_sqrt = mkldnn_eltwise_sqrt;
    const alg_kind_t eltwise_linear = mkldnn_eltwise_linear;
    const alg_kind_t eltwise_bounded_relu = mkldnn_eltwise_bounded_relu;
    const alg_kind_t eltwise_soft_relu = mkldnn_eltwise_soft_relu;
    const alg_kind_t eltwise_logistic = mkldnn_eltwise_logistic;
    const alg_kind_t pooling_max = mkldnn_pooling_max;
    const alg_kind_t pooling_avg = mkldnn_pooling_avg;
    const alg_kind_t pooling_avg_include_padding = mkldnn_pooling_avg_include_padding;
    const alg_kind_t pooling_avg_exclude_padding = mkldnn_pooling_avg_exclude_padding;
    const alg_kind_t lrn_across_channels = mkldnn_lrn_across_channels;
    const alg_kind_t lrn_within_channel = mkldnn_lrn_within_channel;
    const alg_kind_t vanilla_rnn = mkldnn_vanilla_rnn;
    const alg_kind_t vanilla_lstm = mkldnn_vanilla_lstm;
    const alg_kind_t vanilla_gru = mkldnn_vanilla_gru;
    const alg_kind_t gru_linear_before_reset = mkldnn_gru_linear_before_reset;
}

using data_type_t = mkldnn_data_type_t;
namespace data_type {
    const data_type_t undef = mkldnn_data_type_undef;
    const data_type_t f32 = mkldnn_f32;
    const data_type_t s32 = mkldnn_s32;
    const data_type_t s8 = mkldnn_s8;
    const data_type_t u8 = mkldnn_u8;
}

using scratchpad_mode_t = mkldnn_scratchpad_mode_t;
namespace scratchpad_mode {
    const scratchpad_mode_t library = mkldnn_scratchpad_mode_library;
    const scratchpad_mode_t user = mkldnn_scratchpad_mode_user;
}

using rnn_packed_format_t = mkldnn_rnn_packed_memory_format_t;
namespace rnn_packed_format {
    const rnn_packed_format_t undef = mkldnn_packed_format_undef;
    const rnn_packed_format_t ldigo_p = mkldnn_ldigo_p;
    const rnn_packed_format_t ldgoi_p = mkldnn_ldgoi_p;
}

using format_kind_t = mkldnn_format_kind_t;
namespace format_kind {
    const format_kind_t undef = mkldnn_format_kind_undef;
    const format_kind_t any = mkldnn_format_kind_any;
    const format_kind_t blocked = mkldnn_blocked;
    const format_kind_t wino = mkldnn_format_kind_wino;
    const format_kind_t rnn_packed = mkldnn_format_kind_rnn_packed;
}

using format_tag_t = mkldnn_format_tag_t;
namespace format_tag {
    const format_tag_t undef = mkldnn_format_tag_undef;
    const format_tag_t any = mkldnn_format_tag_any;
    const format_tag_t a = mkldnn_a;
    const format_tag_t ab = mkldnn_ab;
    const format_tag_t abc = mkldnn_abc;
    const format_tag_t abcd = mkldnn_abcd;
    const format_tag_t abcde = mkldnn_abcde;
    const format_tag_t abcdef = mkldnn_abcdef;
    const format_tag_t abdec = mkldnn_abdec;
    const format_tag_t acb = mkldnn_acb;
    const format_tag_t acbde = mkldnn_acbde;
    const format_tag_t acdb = mkldnn_acdb;
    const format_tag_t acdeb = mkldnn_acdeb;
    const format_tag_t ba = mkldnn_ba;
    const format_tag_t bac = mkldnn_bac;
    const format_tag_t bacd = mkldnn_bacd;
    const format_tag_t bcda = mkldnn_bcda;
    const format_tag_t cba = mkldnn_cba;
    const format_tag_t cdba = mkldnn_cdba;
    const format_tag_t cdeba = mkldnn_cdeba;
    const format_tag_t decab = mkldnn_decab;
    const format_tag_t Abc16a = mkldnn_Abc16a;
    const format_tag_t ABc16a16b = mkldnn_ABc16a16b;
    const format_tag_t aBc16b = mkldnn_aBc16b;
    const format_tag_t ABc16b16a = mkldnn_ABc16b16a;
    const format_tag_t Abc4a = mkldnn_Abc4a;
    const format_tag_t aBc4b = mkldnn_aBc4b;
    const format_tag_t ABc4b16a4b = mkldnn_ABc4b16a4b;
    const format_tag_t ABc4b4a = mkldnn_ABc4b4a;
    const format_tag_t ABc8a16b2a = mkldnn_ABc8a16b2a;
    const format_tag_t ABc8a8b = mkldnn_ABc8a8b;
    const format_tag_t aBc8b = mkldnn_aBc8b;
    const format_tag_t ABc8b16a2b = mkldnn_ABc8b16a2b;
    const format_tag_t ABc8b8a = mkldnn_ABc8b8a;
    const format_tag_t Abcd16a = mkldnn_Abcd16a;
    const format_tag_t ABcd16a16b = mkldnn_ABcd16a16b;
    const format_tag_t aBcd16b = mkldnn_aBcd16b;
    const format_tag_t ABcd16b16a = mkldnn_ABcd16b16a;
    const format_tag_t aBCd16b16c = mkldnn_aBCd16b16c;
    const format_tag_t aBCd16c16b = mkldnn_aBCd16c16b;
    const format_tag_t Abcd4a = mkldnn_Abcd4a;
    const format_tag_t aBcd4b = mkldnn_aBcd4b;
    const format_tag_t ABcd4b16a4b = mkldnn_ABcd4b16a4b;
    const format_tag_t ABcd4b4a = mkldnn_ABcd4b4a;
    const format_tag_t aBCd4c16b4c = mkldnn_aBCd4c16b4c;
    const format_tag_t aBCd4c4b = mkldnn_aBCd4c4b;
    const format_tag_t ABcd8a16b2a = mkldnn_ABcd8a16b2a;
    const format_tag_t ABcd8a8b = mkldnn_ABcd8a8b;
    const format_tag_t aBcd8b = mkldnn_aBcd8b;
    const format_tag_t ABcd8b16a2b = mkldnn_ABcd8b16a2b;
    const format_tag_t aBCd8b16c2b = mkldnn_aBCd8b16c2b;
    const format_tag_t ABcd8b8a = mkldnn_ABcd8b8a;
    const format_tag_t aBCd8b8c = mkldnn_aBCd8b8c;
    const format_tag_t aBCd8c16b2c = mkldnn_aBCd8c16b2c;
    const format_tag_t aBCd8c8b = mkldnn_aBCd8c8b;
    const format_tag_t Abcde16a = mkldnn_Abcde16a;
    const format_tag_t ABcde16a16b = mkldnn_ABcde16a16b;
    const format_tag_t aBcde16b = mkldnn_aBcde16b;
    const format_tag_t ABcde16b16a = mkldnn_ABcde16b16a;
    const format_tag_t aBCde16b16c = mkldnn_aBCde16b16c;
    const format_tag_t aBCde16c16b = mkldnn_aBCde16c16b;
    const format_tag_t aBCde2c8b4c = mkldnn_aBCde2c8b4c;
    const format_tag_t Abcde4a = mkldnn_Abcde4a;
    const format_tag_t aBcde4b = mkldnn_aBcde4b;
    const format_tag_t ABcde4b4a = mkldnn_ABcde4b4a;
    const format_tag_t aBCde4b4c = mkldnn_aBCde4b4c;
    const format_tag_t aBCde4c16b4c = mkldnn_aBCde4c16b4c;
    const format_tag_t aBCde4c4b = mkldnn_aBCde4c4b;
    const format_tag_t Abcde8a = mkldnn_Abcde8a;
    const format_tag_t ABcde8a8b = mkldnn_ABcde8a8b;
    const format_tag_t aBcde8b = mkldnn_aBcde8b;
    const format_tag_t ABcde8b16a2b = mkldnn_ABcde8b16a2b;
    const format_tag_t aBCde8b16c2b = mkldnn_aBCde8b16c2b;
    const format_tag_t ABcde8b8a = mkldnn_ABcde8b8a;
    const format_tag_t aBCde8b8c = mkldnn_aBCde8b8c;
    const format_tag_t aBCde8c16b2c = mkldnn_aBCde8c16b2c;
    const format_tag_t aBCde8c8b = mkldnn_aBCde8c8b;
    const format_tag_t aBcdef16b = mkldnn_aBcdef16b;
    const format_tag_t aBCdef16b16c = mkldnn_aBCdef16b16c;
    const format_tag_t aBCdef16c16b = mkldnn_aBCdef16c16b;
    const format_tag_t aBcdef4b = mkldnn_aBcdef4b;
    const format_tag_t aBCdef4c4b = mkldnn_aBCdef4c4b;
    const format_tag_t aBCdef8b8c = mkldnn_aBCdef8b8c;
    const format_tag_t aBCdef8c16b2c = mkldnn_aBCdef8c16b2c;
    const format_tag_t aBCdef8c8b = mkldnn_aBCdef8c8b;
    const format_tag_t aBdc16b = mkldnn_aBdc16b;
    const format_tag_t aBdc4b = mkldnn_aBdc4b;
    const format_tag_t aBdc8b = mkldnn_aBdc8b;
    const format_tag_t aBdec16b = mkldnn_aBdec16b;
    const format_tag_t aBdec4b = mkldnn_aBdec4b;
    const format_tag_t aBdec8b = mkldnn_aBdec8b;
    const format_tag_t aBdefc16b = mkldnn_aBdefc16b;
    const format_tag_t aBdefc4b = mkldnn_aBdefc4b;
    const format_tag_t aBdefc8b = mkldnn_aBdefc8b;
    const format_tag_t Acb16a = mkldnn_Acb16a;
    const format_tag_t Acb4a = mkldnn_Acb4a;
    const format_tag_t Acb8a = mkldnn_Acb8a;
    const format_tag_t aCBd16b16c = mkldnn_aCBd16b16c;
    const format_tag_t aCBde16b16c = mkldnn_aCBde16b16c;
    const format_tag_t Acdb16a = mkldnn_Acdb16a;
    const format_tag_t Acdb4a = mkldnn_Acdb4a;
    const format_tag_t Acdb8a = mkldnn_Acdb8a;
    const format_tag_t Acdeb16a = mkldnn_Acdeb16a;
    const format_tag_t Acdeb4a = mkldnn_Acdeb4a;
    const format_tag_t Acdeb8a = mkldnn_Acdeb8a;
    const format_tag_t BAc16a16b = mkldnn_BAc16a16b;
    const format_tag_t BAcd16a16b = mkldnn_BAcd16a16b;
    const format_tag_t last = mkldnn_format_tag_last;

    const format_tag_t x = mkldnn_x;
    const format_tag_t nc = mkldnn_nc;
    const format_tag_t cn = mkldnn_cn;
    const format_tag_t ncw = mkldnn_ncw;
    const format_tag_t nwc = mkldnn_nwc;
    const format_tag_t nchw = mkldnn_nchw;
    const format_tag_t nhwc = mkldnn_nhwc;
    const format_tag_t chwn = mkldnn_chwn;
    const format_tag_t ncdhw = mkldnn_ncdhw;
    const format_tag_t ndhwc = mkldnn_ndhwc;
    const format_tag_t oi = mkldnn_oi;
    const format_tag_t io = mkldnn_io;
    const format_tag_t oiw = mkldnn_oiw;
    const format_tag_t wio = mkldnn_wio;
    const format_tag_t oihw = mkldnn_oihw;
    const format_tag_t hwio = mkldnn_hwio;
    const format_tag_t ihwo = mkldnn_ihwo;
    const format_tag_t iohw = mkldnn_iohw;
    const format_tag_t oidhw = mkldnn_oidhw;
    const format_tag_t dhwio = mkldnn_dhwio;
    const format_tag_t goiw = mkldnn_goiw;
    const format_tag_t goihw = mkldnn_goihw;
    const format_tag_t hwigo = mkldnn_hwigo;
    const format_tag_t giohw = mkldnn_giohw;
    const format_tag_t goidhw = mkldnn_goidhw;
    const format_tag_t tnc = mkldnn_tnc;
    const format_tag_t ntc = mkldnn_ntc;
    const format_tag_t ldsnc = mkldnn_ldsnc;
    const format_tag_t ldigo = mkldnn_ldigo;
    const format_tag_t ldgoi = mkldnn_ldgoi;
    const format_tag_t ldgo = mkldnn_ldgo;
    const format_tag_t nCdhw16c = mkldnn_nCdhw16c;
    const format_tag_t nCdhw4c = mkldnn_nCdhw4c;
    const format_tag_t nCdhw8c = mkldnn_nCdhw8c;
    const format_tag_t nChw16c = mkldnn_nChw16c;
    const format_tag_t nChw4c = mkldnn_nChw4c;
    const format_tag_t nChw8c = mkldnn_nChw8c;
    const format_tag_t nCw16c = mkldnn_nCw16c;
    const format_tag_t nCw4c = mkldnn_nCw4c;
    const format_tag_t nCw8c = mkldnn_nCw8c;
    const format_tag_t IOw16o16i = mkldnn_IOw16o16i;
    const format_tag_t OIw16i16o = mkldnn_OIw16i16o;
    const format_tag_t OIw16o16i = mkldnn_OIw16o16i;
    const format_tag_t Oiw16o = mkldnn_Oiw16o;
    const format_tag_t OIw4i16o4i = mkldnn_OIw4i16o4i;
    const format_tag_t OIw4i4o = mkldnn_OIw4i4o;
    const format_tag_t Oiw4o = mkldnn_Oiw4o;
    const format_tag_t OIw8i16o2i = mkldnn_OIw8i16o2i;
    const format_tag_t OIw8i8o = mkldnn_OIw8i8o;
    const format_tag_t OIw8o16i2o = mkldnn_OIw8o16i2o;
    const format_tag_t OIw8o8i = mkldnn_OIw8o8i;
    const format_tag_t Owi16o = mkldnn_Owi16o;
    const format_tag_t Owi4o = mkldnn_Owi4o;
    const format_tag_t Owi8o = mkldnn_Owi8o;
    const format_tag_t IOhw16o16i = mkldnn_IOhw16o16i;
    const format_tag_t Ohwi16o = mkldnn_Ohwi16o;
    const format_tag_t Ohwi4o = mkldnn_Ohwi4o;
    const format_tag_t Ohwi8o = mkldnn_Ohwi8o;
    const format_tag_t OIhw16i16o = mkldnn_OIhw16i16o;
    const format_tag_t OIhw16o16i = mkldnn_OIhw16o16i;
    const format_tag_t Oihw16o = mkldnn_Oihw16o;
    const format_tag_t OIhw4i16o4i = mkldnn_OIhw4i16o4i;
    const format_tag_t OIhw4i4o = mkldnn_OIhw4i4o;
    const format_tag_t Oihw4o = mkldnn_Oihw4o;
    const format_tag_t OIhw8i16o2i = mkldnn_OIhw8i16o2i;
    const format_tag_t OIhw8i8o = mkldnn_OIhw8i8o;
    const format_tag_t OIhw8o16i2o = mkldnn_OIhw8o16i2o;
    const format_tag_t OIhw8o8i = mkldnn_OIhw8o8i;
    const format_tag_t Odhwi16o = mkldnn_Odhwi16o;
    const format_tag_t Odhwi4o = mkldnn_Odhwi4o;
    const format_tag_t Odhwi8o = mkldnn_Odhwi8o;
    const format_tag_t OIdhw16i16o = mkldnn_OIdhw16i16o;
    const format_tag_t OIdhw16o16i = mkldnn_OIdhw16o16i;
    const format_tag_t Oidhw16o = mkldnn_Oidhw16o;
    const format_tag_t OIdhw4i4o = mkldnn_OIdhw4i4o;
    const format_tag_t Oidhw4o = mkldnn_Oidhw4o;
    const format_tag_t OIdhw8i16o2i = mkldnn_OIdhw8i16o2i;
    const format_tag_t OIdhw8i8o = mkldnn_OIdhw8i8o;
    const format_tag_t OIdhw8o8i = mkldnn_OIdhw8o8i;
    const format_tag_t gIOw16o16i = mkldnn_gIOw16o16i;
    const format_tag_t Goiw16g = mkldnn_Goiw16g;
    const format_tag_t gOIw16i16o = mkldnn_gOIw16i16o;
    const format_tag_t gOIw16o16i = mkldnn_gOIw16o16i;
    const format_tag_t gOiw16o = mkldnn_gOiw16o;
    const format_tag_t gOIw4i16o4i = mkldnn_gOIw4i16o4i;
    const format_tag_t gOIw4i4o = mkldnn_gOIw4i4o;
    const format_tag_t gOiw4o = mkldnn_gOiw4o;
    const format_tag_t gOIw8i16o2i = mkldnn_gOIw8i16o2i;
    const format_tag_t gOIw8i8o = mkldnn_gOIw8i8o;
    const format_tag_t gOIw8o16i2o = mkldnn_gOIw8o16i2o;
    const format_tag_t gOIw8o8i = mkldnn_gOIw8o8i;
    const format_tag_t gOwi16o = mkldnn_gOwi16o;
    const format_tag_t gOwi4o = mkldnn_gOwi4o;
    const format_tag_t gOwi8o = mkldnn_gOwi8o;
    const format_tag_t gIOhw16o16i = mkldnn_gIOhw16o16i;
    const format_tag_t gOhwi16o = mkldnn_gOhwi16o;
    const format_tag_t gOhwi4o = mkldnn_gOhwi4o;
    const format_tag_t gOhwi8o = mkldnn_gOhwi8o;
    const format_tag_t Goihw16g = mkldnn_Goihw16g;
    const format_tag_t gOIhw16i16o = mkldnn_gOIhw16i16o;
    const format_tag_t gOIhw16o16i = mkldnn_gOIhw16o16i;
    const format_tag_t gOihw16o = mkldnn_gOihw16o;
    const format_tag_t gOIhw2i8o4i = mkldnn_gOIhw2i8o4i;
    const format_tag_t gOIhw4i16o4i = mkldnn_gOIhw4i16o4i;
    const format_tag_t gOIhw4i4o = mkldnn_gOIhw4i4o;
    const format_tag_t gOIhw4o4i = mkldnn_gOIhw4o4i;
    const format_tag_t gOihw4o = mkldnn_gOihw4o;
    const format_tag_t Goihw8g = mkldnn_Goihw8g;
    const format_tag_t gOIhw8i16o2i = mkldnn_gOIhw8i16o2i;
    const format_tag_t gOIhw8i8o = mkldnn_gOIhw8i8o;
    const format_tag_t gOIhw8o16i2o = mkldnn_gOIhw8o16i2o;
    const format_tag_t gOIhw8o8i = mkldnn_gOIhw8o8i;
    const format_tag_t gOdhwi16o = mkldnn_gOdhwi16o;
    const format_tag_t gOdhwi4o = mkldnn_gOdhwi4o;
    const format_tag_t gOdhwi8o = mkldnn_gOdhwi8o;
    const format_tag_t gOIdhw16i16o = mkldnn_gOIdhw16i16o;
    const format_tag_t gOIdhw16o16i = mkldnn_gOIdhw16o16i;
    const format_tag_t gOidhw16o = mkldnn_gOidhw16o;
    const format_tag_t gOIdhw4i4o = mkldnn_gOIdhw4i4o;
    const format_tag_t gOidhw4o = mkldnn_gOidhw4o;
    const format_tag_t gOIdhw8i16o2i = mkldnn_gOIdhw8i16o2i;
    const format_tag_t gOIdhw8i8o = mkldnn_gOIdhw8i8o;
    const format_tag_t gOIdhw8o8i = mkldnn_gOIdhw8o8i;
}

using memory_extra_flags_t = mkldnn_memory_extra_flags_t;
namespace memory_extra_flags {
    const memory_extra_flags_t none = mkldnn_memory_extra_flag_none;
    const memory_extra_flags_t compensation_conv_s8s8 = mkldnn_memory_extra_flag_compensation_conv_s8s8;
    const memory_extra_flags_t scale_adjust = mkldnn_memory_extra_flag_scale_adjust;
}

using padding_kind_t = mkldnn_padding_kind_t;
namespace padding_kind {
    const padding_kind_t padding_zero = mkldnn_padding_zero;
}

using engine_kind_t = mkldnn_engine_kind_t;
namespace engine_kind {
    const engine_kind_t any_engine = mkldnn_any_engine;
    const engine_kind_t cpu = mkldnn_cpu;
}

using primitive_kind_t = mkldnn_primitive_kind_t;
namespace primitive_kind {
    const primitive_kind_t undefined = mkldnn_undefined_primitive;
    const primitive_kind_t reorder = mkldnn_reorder;
    const primitive_kind_t concat = mkldnn_concat;
    const primitive_kind_t sum = mkldnn_sum;
    const primitive_kind_t convolution = mkldnn_convolution;
    const primitive_kind_t deconvolution = mkldnn_deconvolution;
    const primitive_kind_t shuffle = mkldnn_shuffle;
    const primitive_kind_t eltwise = mkldnn_eltwise;
    const primitive_kind_t softmax = mkldnn_softmax;
    const primitive_kind_t pooling = mkldnn_pooling;
    const primitive_kind_t lrn = mkldnn_lrn;
    const primitive_kind_t batch_normalization = mkldnn_batch_normalization;
    const primitive_kind_t inner_product = mkldnn_inner_product;
    const primitive_kind_t rnn = mkldnn_rnn;
}

using query_t = mkldnn_query_t;
namespace query {
    const query_t undef = mkldnn_query_undef;

    const query_t engine = mkldnn_query_engine;
    const query_t primitive_kind = mkldnn_query_primitive_kind;

    const query_t num_of_inputs_s32 = mkldnn_query_num_of_inputs_s32;
    const query_t num_of_outputs_s32 = mkldnn_query_num_of_outputs_s32;

    const query_t time_estimate_f64 = mkldnn_query_time_estimate_f64;
    const query_t memory_consumption_s64 = mkldnn_query_memory_consumption_s64;

    const query_t scratchpad_engine = mkldnn_query_scratchpad_engine;

    const query_t impl_info_str = mkldnn_query_impl_info_str;

    const query_t some_d = mkldnn_query_some_d;
    const query_t op_d = mkldnn_query_op_d;
    const query_t convolution_d = mkldnn_query_convolution_d;
    const query_t deconvolution_d = mkldnn_query_deconvolution_d;
    const query_t shuffle_d = mkldnn_query_shuffle_d;
    const query_t eltwise_d = mkldnn_query_eltwise_d;
    const query_t softmax_d = mkldnn_query_softmax_d;
    const query_t pooling_d = mkldnn_query_pooling_d;
    const query_t lrn_d = mkldnn_query_lrn_d;
    const query_t batch_normalization_d = mkldnn_query_batch_normalization_d;
    const query_t inner_product_d = mkldnn_query_inner_product_d;
    const query_t rnn_d = mkldnn_query_rnn_d;

    const query_t some_md = mkldnn_query_some_md;
    const query_t src_md = mkldnn_query_src_md;
    const query_t diff_src_md = mkldnn_query_diff_src_md;
    const query_t weights_md = mkldnn_query_weights_md;
    const query_t diff_weights_md = mkldnn_query_diff_weights_md;
    const query_t dst_md = mkldnn_query_dst_md;
    const query_t diff_dst_md = mkldnn_query_diff_dst_md;

    const query_t workspace_md = mkldnn_query_workspace_md;
    const query_t scratchpad_md = mkldnn_query_scratchpad_md;
}

using blocking_desc_t = mkldnn_blocking_desc_t;
using rnn_packed_desc_t = mkldnn_rnn_packed_desc_t;
using wino_desc_t = mkldnn_wino_desc_t;
using memory_extra_desc_t = mkldnn_memory_extra_desc_t;
using memory_desc_t = mkldnn_memory_desc_t;
using convolution_desc_t = mkldnn_convolution_desc_t;
using deconvolution_desc_t = mkldnn_deconvolution_desc_t;
using shuffle_desc_t = mkldnn_shuffle_desc_t;
using pooling_desc_t = mkldnn_pooling_desc_t;
using eltwise_desc_t = mkldnn_eltwise_desc_t;
using softmax_desc_t = mkldnn_softmax_desc_t;
using lrn_desc_t = mkldnn_lrn_desc_t;
using batch_normalization_desc_t = mkldnn_batch_normalization_desc_t;
using inner_product_desc_t = mkldnn_inner_product_desc_t;

using rnn_direction_t = mkldnn_rnn_direction_t;
using rnn_cell_desc_t = mkldnn_rnn_cell_desc_t;
using rnn_desc_t = mkldnn_rnn_desc_t;

/* C op_desc_t, which eventually are just (void*) */
using c_op_desc_t = mkldnn_op_desc_t;
using const_c_op_desc_t = const_mkldnn_op_desc_t;

struct op_desc_t {
    union {
        primitive_kind_t kind;
        convolution_desc_t convolution;
        deconvolution_desc_t deconvolution;
        shuffle_desc_t shuffle;
        pooling_desc_t pooling;
        eltwise_desc_t eltwise;
        softmax_desc_t softmax;
        lrn_desc_t lrn;
        batch_normalization_desc_t batch_normalization;
        inner_product_desc_t inner_product;
        rnn_desc_t rnn;
    };

    op_desc_t(const primitive_kind_t &_): kind(_) {}

#   define DECL_CTOR_AND_CONVERTERS(c_type, name) \
    op_desc_t(const c_type &_): name(_) {} \
    static op_desc_t *convert_from_c(c_type *_) \
    { return reinterpret_cast<op_desc_t*>(_); } \
    static const op_desc_t *convert_from_c(const c_type *_) \
    { return reinterpret_cast<const op_desc_t*>(_); }

    DECL_CTOR_AND_CONVERTERS(convolution_desc_t, convolution);
    DECL_CTOR_AND_CONVERTERS(shuffle_desc_t, shuffle);
    DECL_CTOR_AND_CONVERTERS(pooling_desc_t, pooling);
    DECL_CTOR_AND_CONVERTERS(eltwise_desc_t, eltwise);
    DECL_CTOR_AND_CONVERTERS(softmax_desc_t, softmax);
    DECL_CTOR_AND_CONVERTERS(lrn_desc_t, lrn);
    DECL_CTOR_AND_CONVERTERS(batch_normalization_desc_t, batch_normalization);
    DECL_CTOR_AND_CONVERTERS(inner_product_desc_t, inner_product);
    DECL_CTOR_AND_CONVERTERS(rnn_desc_t, rnn);

#   undef DECL_CTOR_AND_CONVERTERS
};

using engine_t = mkldnn_engine;
using primitive_desc_iterator_t = mkldnn_primitive_desc_iterator;
using primitive_desc_t = mkldnn_primitive_desc;
using primitive_attr_t = mkldnn_primitive_attr;
using post_ops_t = mkldnn_post_ops;
using memory_t = mkldnn_memory;
using primitive_t = mkldnn_primitive;

using primitive_arg_index_t = int;

using stream_flags_t = mkldnn_stream_flags_t;
namespace stream_flags {
    const stream_flags_t default_flags = mkldnn_stream_default_flags;
}
using stream_t = mkldnn_stream;

/* forward declaration of the internal primitive_desc types */
struct batch_normalization_bwd_pd_t;
struct batch_normalization_fwd_pd_t;
struct batch_normalization_pd_t;
struct concat_pd_t;
struct convolution_bwd_data_pd_t;
struct convolution_bwd_weights_pd_t;
struct convolution_fwd_pd_t;
struct convolution_pd_t;
struct deconvolution_bwd_data_pd_t;
struct deconvolution_bwd_weights_pd_t;
struct deconvolution_fwd_pd_t;
struct deconvolution_pd_t;
struct eltwise_bwd_pd_t;
struct eltwise_fwd_pd_t;
struct eltwise_pd_t;
struct inner_product_bwd_data_pd_t;
struct inner_product_bwd_weights_pd_t;
struct inner_product_fwd_pd_t;
struct inner_product_pd_t;
struct lrn_bwd_pd_t;
struct lrn_fwd_pd_t;
struct lrn_pd_t;
struct pooling_bwd_pd_t;
struct pooling_fwd_pd_t;
struct pooling_pd_t;
struct reorder_pd_t;
struct rnn_bwd_pd_t;
struct rnn_fwd_pd_t;
struct rnn_pd_t;
struct shuffle_pd_t;
struct softmax_bwd_pd_t;
struct softmax_fwd_pd_t;
struct softmax_pd_t;
struct sum_pd_t;

}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
