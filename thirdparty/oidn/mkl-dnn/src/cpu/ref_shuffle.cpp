/*******************************************************************************
* Copyright 2018 Intel Corporation
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

#include <assert.h>
#include <math.h>

#include "c_types_map.hpp"
#include "mkldnn_thread.hpp"
#include "type_helpers.hpp"

#include "ref_shuffle.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace format_tag;

template <int data_type_size>
template <mkldnn_format_tag_t tag>
void ref_shuffle_t<data_type_size>::execute_(const exec_ctx_t &ctx) const {
    using namespace prop_kind;
    using namespace utils;

    const memory_desc_wrapper data_d(pd()->data_md());

    auto i_arg = pd()->is_fwd() ? MKLDNN_ARG_SRC : MKLDNN_ARG_DIFF_DST;
    auto o_arg = pd()->is_fwd() ? MKLDNN_ARG_DST : MKLDNN_ARG_DIFF_SRC;
    auto input = CTX_IN_MEM(const data_t *, i_arg);
    auto output = CTX_OUT_MEM(data_t *, o_arg);

    const int axis = pd()->axis();
    const int axis_size = pd()->axis_size();

    const int MB = pd()->MB();
    const int C = pd()->C();
    int H = 1, W = 1, D = 1, HW = 1, SP = 1;
    const bool has_spatial = utils::one_of(data_d.ndims(), 3, 4 ,5);
    if (has_spatial)
    {
        D = pd()->D();
        H = pd()->H();
        W = pd()->W();
        HW = H * W;
        SP = D * HW;
    }
    const size_t stride_mb = data_d.blocking_desc().strides[0];
    constexpr int blksize = one_of(tag, nChw16c, nCdhw16c) ? 16 : 8;

    if (axis == 1 && one_of(tag, nChw16c, nChw8c, nCdhw16c, nCdhw16c)) {
#if MKLDNN_THR == MKLDNN_THR_OMP
#       pragma omp parallel for collapse(3) schedule(static)
        for (int mb = 0; mb < MB; ++mb)
        for (int cb = 0; cb < C; cb += blksize)
        for (int sp = 0; sp < SP; ++sp) {
            const size_t off = mb * stride_mb + sp * blksize;
            const size_t output_off = off + cb * SP;
            PRAGMA_OMP_SIMD()
            for (int cc = 0; cc < nstl::min(blksize, C - cb); ++cc)
            {
                int input_c = rev_transposed_[cb + cc];
                const size_t input_off = off + input_c / blksize * SP * blksize
                                           + input_c % blksize;
                output[output_off + cc] = input[input_off];
            }
        }
#else
        parallel_nd(MB, utils::div_up(C, blksize), SP, [&](int mb, int c,
                  int sp) {
            const size_t off = mb * stride_mb + sp * blksize;
            const int cb = c * blksize;
            const size_t output_off = off + cb * SP;
            for (int cc = 0; cc < nstl::min(blksize, C - cb); ++cc)
            {
                int input_c = rev_transposed_[cb + cc];
                const size_t input_off = off + input_c / blksize * SP * blksize
                                           + input_c % blksize;
                output[output_off + cc] = input[input_off];
            }
        });
#endif
    } else if (axis == 1 && one_of(tag, nhwc, ndhwc)) {
        parallel_nd(MB, SP, [&](int mb, int sp) {
            const size_t off = mb * stride_mb + sp * C;
            PRAGMA_OMP_SIMD()
            for (int c = 0; c < C; ++c)
                output[off + c] = input[off + rev_transposed_[c]];
        });
    } else if (axis == 1 && one_of(tag, nchw, ncdhw)) {
        parallel_nd(MB, C, [&](int mb, int c) {
            const size_t output_off = mb * stride_mb + c * SP;
            const size_t input_off = mb * stride_mb + rev_transposed_[c] * SP;
            PRAGMA_OMP_SIMD()
            for (int sp = 0; sp < SP; ++sp) {
                output[output_off + sp] = input[input_off + sp];
            }
        });
    } else {
        auto dims = pd()->desc()->data_desc.dims;
        auto ndims = pd()->desc()->data_desc.ndims;
        const size_t outer_size = utils::array_product(dims, axis);
        const size_t inner_size = utils::array_product(dims + axis + 1,
                                         ndims - axis - 1);
        const size_t dim = axis_size * inner_size;

        parallel_nd(outer_size, axis_size, inner_size, [&](size_t ou, int a,
               size_t in)
        {
            const size_t off = ou * dim + in;
            auto &o = output[data_d.off_l(off + a * inner_size)];
            o = input[data_d.off_l(off + rev_transposed_[a] * inner_size)];
        });
    }
}

template void ref_shuffle_t<4>::execute_<nCdhw16c>(const exec_ctx_t &ctx) const;
template void ref_shuffle_t<4>::execute_<nChw16c>(const exec_ctx_t &ctx) const;
template void ref_shuffle_t<4>::execute_<nCdhw8c>(const exec_ctx_t &ctx) const;
template void ref_shuffle_t<4>::execute_<nChw8c>(const exec_ctx_t &ctx) const;
template void ref_shuffle_t<4>::execute_<ncdhw>(const exec_ctx_t &ctx) const;
template void ref_shuffle_t<4>::execute_<nchw>(const exec_ctx_t &ctx) const;
template void ref_shuffle_t<4>::execute_<ndhwc>(const exec_ctx_t &ctx) const;
template void ref_shuffle_t<4>::execute_<nhwc>(const exec_ctx_t &ctx) const;
template void ref_shuffle_t<4>::execute_<any>(const exec_ctx_t &ctx) const;

template void ref_shuffle_t<1>::execute_<nCdhw16c>(const exec_ctx_t &ctx) const;
template void ref_shuffle_t<1>::execute_<nChw16c>(const exec_ctx_t &ctx) const;
template void ref_shuffle_t<1>::execute_<nCdhw8c>(const exec_ctx_t &ctx) const;
template void ref_shuffle_t<1>::execute_<nChw8c>(const exec_ctx_t &ctx) const;
template void ref_shuffle_t<1>::execute_<ncdhw>(const exec_ctx_t &ctx) const;
template void ref_shuffle_t<1>::execute_<nchw>(const exec_ctx_t &ctx) const;
template void ref_shuffle_t<1>::execute_<ndhwc>(const exec_ctx_t &ctx) const;
template void ref_shuffle_t<1>::execute_<nhwc>(const exec_ctx_t &ctx) const;
template void ref_shuffle_t<1>::execute_<any>(const exec_ctx_t &ctx) const;

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
