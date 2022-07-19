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

#ifndef MKLDNN_TRAITS_HPP
#define MKLDNN_TRAITS_HPP

#include <assert.h>
#include <stdint.h>

#include "mkldnn.h"
#include "c_types_map.hpp"
#include "nstl.hpp"
#include "utils.hpp"
#include "z_magic.hpp"

namespace mkldnn {
namespace impl {

template <data_type_t> struct prec_traits {}; /* ::type -> float */
template <typename> struct data_traits {}; /* ::data_type -> f32 */
template <int> struct typesize_traits {}; /* ::data_type_size -> f32 */
template <primitive_kind_t> struct pkind_traits {}; /* ::desc_type, ::query_d */

template <> struct prec_traits<data_type::f32> { typedef float type; };
template <> struct prec_traits<data_type::s32> { typedef int32_t type; };
template <> struct prec_traits<data_type::s8> { typedef int8_t type; };
template <> struct prec_traits<data_type::u8> { typedef uint8_t type; };

template <> struct data_traits<float>
{ static constexpr data_type_t data_type = data_type::f32; };
template <> struct data_traits<int32_t>
{ static constexpr data_type_t data_type = data_type::s32; };
template <> struct data_traits<int8_t>
{ static constexpr data_type_t data_type = data_type::s8; };
template <> struct data_traits<uint8_t>
{ static constexpr data_type_t data_type = data_type::u8; };

template <> struct typesize_traits<4> { typedef float type; };
template <> struct typesize_traits<2> { typedef int16_t type; };
template <> struct typesize_traits<1> { typedef uint8_t type; };

#define PKIND_TRAITS_INST(op) \
template <> struct pkind_traits<primitive_kind::op> { \
    typedef CONCAT2(op, _desc_t) desc_type; \
    static constexpr query_t query_d = query::CONCAT2(op, _d); \
}
PKIND_TRAITS_INST(convolution);
PKIND_TRAITS_INST(deconvolution);
PKIND_TRAITS_INST(shuffle);
PKIND_TRAITS_INST(eltwise);
PKIND_TRAITS_INST(softmax);
PKIND_TRAITS_INST(pooling);
PKIND_TRAITS_INST(lrn);
PKIND_TRAITS_INST(batch_normalization);
PKIND_TRAITS_INST(inner_product);
PKIND_TRAITS_INST(rnn);
#undef PKIND_TRAITS_INST

}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
