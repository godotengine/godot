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

#include "c_types_map.hpp"
#include "type_helpers.hpp"
#include "mkldnn_thread.hpp"
#include "mkldnn_traits.hpp"
#include "math_utils.hpp"

#include "ref_deconvolution.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

void ref_deconvolution_fwd_t::compute_fwd_bias(const data_t *bias,
        data_t *dst) const {
    const memory_desc_wrapper dst_d(pd()->dst_md());

    const int G = pd()->G();
    const int MB = pd()->MB();
    const int OH = pd()->OH();
    const int OW = pd()->OW();
    const int OD = pd()->OD();
    const int OC = pd()->OC() / G;
    const int ndims = pd()->desc()->src_desc.ndims;

    parallel_nd(MB, G, OC, OD, OH, OW,
        [&](int mb, int g, int oc, int od, int oh, int ow) {
            auto b = bias[g * OC + oc];
            switch (ndims) {
            case 5: dst[dst_d.off(mb, g * OC + oc, od, oh, ow)] += b; break;
            case 4: dst[dst_d.off(mb, g * OC + oc, oh, ow)] += b; break;
            case 3: dst[dst_d.off(mb, g * OC + oc, ow)] += b; break;
            default: assert(!"invalid dimension size");
            }
    });
}

void ref_deconvolution_fwd_t::compute_fwd_bias_ncdhw(const data_t *bias,
        data_t *dst) const {
    const memory_desc_wrapper dst_d(pd()->dst_md());

    const int MB = pd()->MB();
    const int OC = pd()->OC();
    const int SP = pd()->OW()*pd()->OH()*pd()->OD();

    parallel_nd(MB, OC, [&](int mb, int oc) {
        PRAGMA_OMP_SIMD()
        for (int sp = 0; sp < SP; ++sp) {
            auto offset = (size_t)(mb * OC + oc) * SP + sp;
            dst[offset] += bias[oc];
        }
    });
}

template <int blksize>
void ref_deconvolution_fwd_t::compute_fwd_bias_nCdhwXc(const data_t *bias,
        data_t *dst) const {
    const memory_desc_wrapper dst_d(pd()->dst_md());

    const int MB = pd()->MB();
    const int OC = pd()->OC();
    const int SP = pd()->OW() * pd()->OH() * pd()->OD();

    const ptrdiff_t stride_mb = dst_d.blocking_desc().strides[0];

    parallel_nd(MB, utils::div_up(OC, blksize), SP,
        [&](int mb, int oc_blk, int sp) {
        int oc = oc_blk * blksize;
        auto offset = mb * stride_mb + oc * SP + sp * blksize;
        const int blk = nstl::min(blksize, OC - oc);

        PRAGMA_OMP_SIMD()
        for (int i = 0; i < blk; ++i)
            dst[offset + i] += bias[oc + i];
    });
}

void ref_deconvolution_bwd_weights_t::compute_bwd_bias(const data_t *diff_dst,
        data_t *diff_bias) const {
    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());

    const int G = pd()->G();
    const int MB = pd()->MB();
    const int OH = pd()->OH();
    const int OW = pd()->OW();
    const int OC = pd()->OC() / G;
    const int OD = pd()->OD();
    const int ndims = pd()->desc()->src_desc.ndims;

    parallel_nd(G, OC, [&](int g, int oc) {
        data_t db = 0;
        for (int mb = 0; mb < MB; ++mb) {
            for (int od = 0; od < OD; ++od) {
                for (int oh = 0; oh < OH; ++oh) {
                    for (int ow = 0; ow < OW; ++ow) {
                        switch (ndims) {
                        case 5:
                            db += diff_dst[diff_dst_d.off(
                                    mb, g * OC + oc, od, oh, ow)];
                            break;
                        case 4:
                            db += diff_dst[diff_dst_d.off(
                                    mb, g * OC + oc, oh, ow)];
                            break;
                        case 3:
                            db += diff_dst[diff_dst_d.off(mb, g * OC + oc, ow)];
                            break;
                        default: assert(!"invalid dimension size");
                        }
                    }
                }
            }
        }
        diff_bias[g * OC + oc] = db;
    });
}

void ref_deconvolution_bwd_weights_t::compute_bwd_bias_ncdhw(
        const data_t *diff_dst, data_t *diff_bias) const {
    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());

    const int OC = pd()->OC();
    const int MB = pd()->MB();
    const int SP = pd()->OH()*pd()->OW()*pd()->OD();

    parallel_nd(OC, [&](int oc) {
        data_t db = 0;
        for (int mb = 0; mb < MB; ++mb) {
            PRAGMA_OMP_SIMD()
            for (int sp = 0; sp < SP; ++sp) {
                auto offset = (size_t)(mb * OC + oc) * SP + sp;
                db += diff_dst[offset];
            }
        }
        diff_bias[oc] = db;
    });
}

template <int blksize>
void ref_deconvolution_bwd_weights_t::compute_bwd_bias_nCdhwXc(
        const data_t *diff_dst, data_t *diff_bias) const {
    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());

    const int OC = pd()->OC();
    const int MB = pd()->MB();
    const int SP = pd()->OH() * pd()->OW() * pd()->OD();

    const ptrdiff_t stride_mb = diff_dst_d.blocking_desc().strides[0];

    parallel_nd(utils::div_up(OC, blksize), [&](int ocb) {
        data_t db[blksize] = {0};

        for (int mb = 0; mb < MB; ++mb) {
            for (int sp = 0; sp < SP; ++sp) {
                auto offset = mb * stride_mb + (ocb * SP + sp) * blksize;

                PRAGMA_OMP_SIMD()
                for (int i = 0; i < blksize; ++i)
                    db[i] += diff_dst[offset+i];
            }
        }

        const int blk = nstl::min(blksize, OC - ocb * blksize);

        PRAGMA_OMP_SIMD()
        for (int i = 0; i < blk; ++i)
            diff_bias[ocb * blksize + i] = db[i];
    });
}

template void ref_deconvolution_fwd_t::compute_fwd_bias_nCdhwXc<8>(
        const data_t *diff_dst, data_t *diff_bias) const;
template void ref_deconvolution_fwd_t::compute_fwd_bias_nCdhwXc<16>(
        const data_t *diff_dst, data_t *diff_bias) const;
template void ref_deconvolution_bwd_weights_t::compute_bwd_bias_nCdhwXc<8>(
        const data_t *diff_dst, data_t *diff_bias) const;
template void ref_deconvolution_bwd_weights_t::compute_bwd_bias_nCdhwXc<16>(
        const data_t *diff_dst, data_t *diff_bias) const;

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
