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

#include "c_types_map.hpp"
#include "memory_desc_wrapper.hpp"
#include "mkldnn_debug.h"
#include "nstl.hpp"
#include "type_helpers.hpp"

#include "cpu_primitive.hpp"
#include "cpu_reorder_pd.hpp"
#include "jit_uni_reorder.hpp"

#include "jit_generator.hpp"

// #define TR_DEBUG
#if defined(TR_DEBUG)
#define DEBUg(...) do { __VA_ARGS__ } while (0)
#else
#define DEBUg(...)
#endif
#define DEBUG(...) DEBUg(__VA_ARGS__)

#ifdef _WIN32
/* seems like s_addr is a reserved macro on Windows */
#undef s_addr
#endif

using namespace Xbyak;
using namespace mkldnn::impl::types;

namespace mkldnn {
namespace impl {
namespace cpu {

namespace tr {

/** Minimal reasonable/desirable kernel size.
 * The constant might be used to determine how a problem should be split
 * between kernel and threading driver. */
const size_t ker_prb_size_min = 64;

/* kernel */
struct jit_uni_reorder_kernel_f32: public kernel_t, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_reorder_kernel_f32)

    enum {
        len_unroll_max = 256,
        ndims_jit_loop_max = 3,
    };

    struct simple_impl_desc_t {
        int ndims_full_unroll;
        int len_last_dim_unroll;
        int len_unroll;
    };

    static bool simple_impl_desc_init(const prb_t &prb,
            simple_impl_desc_t *desc) {
        const int ndims = prb.ndims;

        int ndims_full_unroll = 0;
        int len_last_dim_unroll = 1;
        int len_unroll = 1;

        for (int d = 0; d < ndims; ++d) {
            auto &node = prb.nodes[d];
            if (len_unroll * node.n <= len_unroll_max) {
                ndims_full_unroll++;
                len_unroll *= node.n;
            } else {
                len_last_dim_unroll = len_unroll_max / len_unroll;
                while (node.n % len_last_dim_unroll)
                    --len_last_dim_unroll;
                len_unroll *= len_last_dim_unroll;
                break;
            }
        }

        if (prb.ndims - ndims_full_unroll > ndims_jit_loop_max)
            return false;

        if (desc) {
            desc->ndims_full_unroll = ndims_full_unroll;
            desc->len_last_dim_unroll = len_last_dim_unroll;
            desc->len_unroll = len_unroll;
        }

        return true;
    }

    static bool applicable(const prb_t &p) {
        using namespace data_type;

        bool ok = true
            && p.ndims > 0
            && utils::one_of(p.itype, f32, s32, s8, u8)
            && utils::one_of(p.otype, f32, s32, s8, u8)
            && utils::everyone_is(0, p.ioff, p.ooff) /* do we need this? */
            && utils::one_of(p.beta, 0.f, 1.f) /* anything else? */
            && simple_impl_desc_init(p, nullptr)
            && mayiuse(sse42)
            && IMPLICATION(!utils::everyone_is(f32, p.itype, p.otype),
                    mayiuse(avx));
        if (!ok) return false;

        const ptrdiff_t max_stride = (1LL<<31) - 1;
        for (int d = 0; d < p.ndims; ++d) {
            const ptrdiff_t cms = max_stride / p.nodes[d].n;
            bool strides_ok = true
                && p.nodes[d].is < cms / (int)data_type_size(p.itype)
                && p.nodes[d].os < cms / (int)data_type_size(p.otype);
            if (!strides_ok) return false;
        }

        return true;
    }

    int n(int d) { assert(d < prb_.ndims); return (int)prb_.nodes[d].n; }
    int is(int d) { assert(d < prb_.ndims); return (int)prb_.nodes[d].is; }
    int os(int d) { assert(d < prb_.ndims); return (int)prb_.nodes[d].os; }
    int ss(int d) { assert(d < prb_.ndims); return (int)prb_.nodes[d].ss; }

    Address i_addr(int i_off)
    { return ptr[reg_ptr_in + reg_off_in + i_off * itype_sz]; }

    Address o_addr(int o_off)
    { return ptr[reg_ptr_out + reg_off_out + o_off * otype_sz]; }

    Address s_addr(int s_off)
    { return ptr[reg_ptr_scale + reg_off_scale + s_off * stype_sz]; }

    void step(int off, int prev_i_off, int prev_o_off, int prev_s_off,
            int &i_off, int &o_off, int &s_off, int step_size = 1) {
        i_off = prev_i_off;
        o_off = prev_o_off;
        s_off = prev_s_off;

        if (off == 0) return;

        int start_dim = 0, dims_prod = 1;
        for (; start_dim < prb_.ndims && dims_prod != step_size; ++start_dim)
            dims_prod *= n(start_dim);
        assert(start_dim < prb_.ndims);
        off /= step_size;

        for (int d = start_dim; d < prb_.ndims; ++d) {
            i_off += is(d);
            o_off += os(d);
            s_off += ss(d);

            if (off % n(d)) break;

            i_off += - n(d) * is(d);
            o_off += - n(d) * os(d);
            s_off += - n(d) * ss(d);
            off /= n(d);

            if (off == 0) break; /* FIXME: is it really required? */
        }
    }

    void step(int off, int prev_i_off, int prev_o_off, int &i_off, int &o_off,
            int step_size = 1) {
        int dummy = 0;
        step(off, prev_i_off, prev_o_off, dummy, i_off, o_off, dummy,
                step_size);
    }

    void tr8x8_avx2(int i_off, int o_off) {
        for (int i = 0; i < 8; i++)
            vmovups(Ymm(i), i_addr(i_off + i * 8));

        for (int i = 0; i < 8 / 2; i++) {
            vunpcklps(Ymm(8 + i), Ymm(2 * i), Ymm(2 * i + 1));
            vunpckhps(Ymm(i), Ymm(2 * i), Ymm(2 * i + 1));
        }

        const unsigned int lfloat = 0x44;
        const unsigned int ufloat = 0xee;
        for (int i = 0; i < 8 / 2; i++) {
            int j = i % 2 == 0 ? 8 + i : i - 1;
            vshufps(Ymm(8 / 2 + 2 * i), Ymm(j), Ymm(j + 1), lfloat);
            vshufps(Ymm(8 / 2 + 2 * i + 1), Ymm(j), Ymm(j + 1), ufloat);
        }

        const unsigned int lquad = 0x20;
        for (int i = 0; i < 8 / 2; i++)
            vperm2f128(Ymm(i), Ymm(8 / 2 + i), Ymm(8 + i), lquad);

        const unsigned int uquad = 0x31;
        for (int i = 8 / 2; i < 8; i++)
            vperm2f128(Ymm(i), Ymm(i), Ymm(8 / 2 + i), uquad);

        for (int i = 0; i < 8; i++)
            vmovups(o_addr(o_off + i * 8), Ymm(i));
    }

    bool process_unroll_tr8x8(int len) {
        bool can_do = true
            && mayiuse(avx2)
            && prb_.ndims >= 2
            && utils::everyone_is(4, itype_sz, otype_sz)
            && utils::everyone_is(8, n(0), n(1))
            && utils::everyone_is(1, os(0), is(1))
            && utils::everyone_is(8, os(1), is(0))
            && prb_.scale_type == scale_type_t::NONE
            && prb_.beta == 0.f;
        if (!can_do) return false;

        const int step_size = n(0) * n(1);
        int i_off = 0, o_off = 0;
        for (int off = 0; off < len; off += step_size) {
            step(off, i_off, o_off, i_off, o_off, step_size);
            tr8x8_avx2(i_off, o_off);
        }

        return true;
    }

    template <cpu_isa_t isa>
    bool process_direct_copy(int len) {
        using namespace data_type;

        using Vmm = typename cpu_isa_traits<isa>::Vmm;
        const int simd_w = cpu_isa_traits<isa>::vlen / itype_sz;

        bool can_do = true
            && mayiuse(isa)
            && utils::everyone_is(1, os(0), is(0))
            && (false
                    || prb_.itype == prb_.otype
                    || (prb_.itype == s32 && prb_.otype == f32)
                    || (prb_.itype == f32 && prb_.otype == s32)
                    )
            && len % simd_w == 0
            && n(0) % len == 0
            && prb_.scale_type == scale_type_t::NONE
            && prb_.beta == 0.f;
        if (!can_do) return false;

        for (int off = 0; off < len;) {
            const int unroll = nstl::min(16, (len - off) / simd_w);

            for (int ur = 0; ur < unroll; ++ur)
                uni_vmovups(Vmm(ur), i_addr(off + ur * simd_w));

            if (prb_.itype != prb_.otype) {
                for (int ur = 0; ur < unroll; ++ur) {
                    if (prb_.itype == s32 && prb_.otype == f32)
                        uni_vcvtdq2ps(Vmm(ur), Vmm(ur));
                    else if (prb_.itype == f32 && prb_.otype == s32)
                        uni_vcvtps2dq(Vmm(ur), Vmm(ur));
                    else assert(!"unreachable");
                }
            }

            for (int ur = 0; ur < unroll; ++ur)
                uni_vmovups(o_addr(off + ur * simd_w), Vmm(ur));

            off += unroll * simd_w;
        }

        return true;
    }

    void process_unroll_generic_step(int reg_unroll, const int *i_off,
            const int *o_off, const int *s_off) {
        using namespace data_type;

        auto cvt2ps = [=](const Xmm &dst, const Operand &src, data_type_t idt) {
            Xmm dst_pure = Xmm(dst.getIdx());
            switch (idt) {
            case f32:
                if (src.isMEM() || src.getIdx() != dst.getIdx())
                    vmovups(dst, src);
                break;
            case s32: vcvtdq2ps(dst, src); break;
            case s8: vpmovsxbd(dst, src); vcvtdq2ps(dst_pure, dst); break;
            case u8: vpmovzxbd(dst, src); vcvtdq2ps(dst_pure, dst); break;
            default: assert(!"unreachable");
            }
        };

        auto cvt2int = [=](const Xmm &xmm, data_type_t odt, data_type_t idt) {
            switch (odt) {
            case s32:
                if (idt == f32) vcvtps2dq(xmm, xmm);
                else if (idt == s8) vpmovsxbd(xmm, xmm);
                else if (idt == u8) vpmovzxbd(xmm, xmm);
                break;
            case s8:
                if (idt == f32) vcvtps2dq(xmm, xmm);
                if (idt == f32 || idt == s32) {
                    if (mayiuse(avx512_core)) {
                        vpmovsdb(xmm, xmm);
                    } else {
                        vpackssdw(xmm, xmm, xmm_zero);
                        vpacksswb(xmm, xmm, xmm_zero);
                    }
                }
                if (idt == u8) vpminub(xmm, xmm, xmm_4x127b);
                break;
            case u8:
                if (idt == f32) vcvtps2dq(xmm, xmm);
                if (idt == f32 || idt == s32) {
                    if (mayiuse(avx512_core)) {
                        vpmaxsd(xmm, xmm, xmm_zero);
                        vpmovusdb(xmm, xmm);
                    } else {
                        vpackssdw(xmm, xmm, xmm_zero);
                        vpackuswb(xmm, xmm, xmm_zero);
                    }
                }
                if (idt == s8) vpmaxsb(xmm, xmm, xmm_zero);
                break;
            default: assert(!"unreachable");
            }
        };

        auto load = [=](const Xmm &xmm, const Address &addr, int size) {
            switch (size) {
            case 16: movups(xmm, addr); break;
            case 4: movss(xmm, addr); break;
            case 1: pinsrb(xmm, addr, 0x0); break;
            default: assert(!"unreachable");
            }
        };

        auto store = [=](const Address &addr, const Xmm &xmm, int size) {
            switch (size) {
            case 16: movups(addr, xmm); break;
            case 4: movss(addr, xmm); break;
            case 1: pextrb(addr, xmm, 0x0); break;
            default: assert(!"unreachable");
            }
        };

        /* check whether loading 4 values at once is possible */
        bool can_load_xmm = mayiuse(avx) && reg_unroll % 4 == 0;
        for (int ur = 1; ur < reg_unroll; ++ur)
            if (i_off[ur] != i_off[ur - 1] + 1)
                can_load_xmm = false;
        const int load_step = can_load_xmm ? 4 : 1;

        /* check whether storing 4 values at once is possible */
        bool can_store_xmm = reg_unroll % 4 == 0;
        for (int ur = 1; ur < reg_unroll; ++ur)
            if (o_off[ur] != o_off[ur - 1] + 1)
                can_store_xmm = false;
        const int ur_step = can_store_xmm ? 4 : 1;

        const bool interim_f32 = false
            || utils::one_of(f32, prb_.itype, prb_.otype)
            || prb_.scale_type != scale_type_t::NONE
            || prb_.beta != 0.f;

        if (!can_load_xmm && can_store_xmm) {
            assert(ur_step == 4);
            /* load with stride */
            for (int ur = 0; ur < reg_unroll; ur += ur_step) {
                for (int r = 0; r < ur_step; ++r) {
                    if (itype_sz == 4)
                        pinsrd(Xmm(ur), i_addr(i_off[ur + r]), r);
                    else
                        pinsrb(Xmm(ur), i_addr(i_off[ur + r]), r);
                }
            }
        } else {
            for (int ur = 0; ur < reg_unroll; ur += load_step)
                load(Xmm(ur), i_addr(i_off[ur]), load_step * itype_sz);
        }

        /* xmm[:] <-- (f32)xmm[:] */
        if (interim_f32) {
            const int cvt_step = nstl::max(load_step, ur_step);
            for (int ur = 0; ur < reg_unroll; ur += cvt_step)
                cvt2ps(Xmm(ur), Xmm(ur), prb_.itype);
        }

        if (can_load_xmm && !can_store_xmm) {
            const bool fast_return = true // transposition on the fly
                && prb_.scale_type != scale_type_t::MANY
                && prb_.beta == 0.f;
            if (fast_return) {
                for (int ur = 0; ur < reg_unroll; ur += load_step) {
                    if (prb_.scale_type == scale_type_t::COMMON)
                        mulps(Xmm(ur), xmm_scale);
                    if (prb_.otype != f32)
                        cvt2int(Xmm(ur), prb_.otype,
                                interim_f32 ? f32 : prb_.itype);
                    for (int r = 0; r < load_step; ++r) {
                        if (otype_sz == 4)
                            pextrd(o_addr(o_off[ur + r]), Xmm(ur), r);
                        else
                            pextrb(o_addr(o_off[ur + r]), Xmm(ur), r);
                    }
                }
                return;
            }

            /* scatter elements of xmm into 4 xmms */
            if (itype_sz == 4 || interim_f32) {
                for (int ur = 0; ur < reg_unroll; ur += load_step)
                    for (int r = 1; r < load_step; ++r)
                        vshufps(Xmm(ur + r), Xmm(ur), Xmm(ur), r);
            } else {
                for (int ur = 0; ur < reg_unroll; ur += load_step)
                    for (int r = 1; r < load_step; ++r)
                        vpalignr(Xmm(ur + r), Xmm(ur), Xmm(ur), r);
            }
        }

        /* scale and beta processing */
        if (can_store_xmm) {
            /* xmm <-- scale * xmm[:] */
            if (prb_.scale_type == scale_type_t::COMMON) {
                for (int ur = 0; ur < reg_unroll; ur += ur_step)
                    mulps(Xmm(ur), xmm_scale);
            } else if (prb_.scale_type == scale_type_t::MANY) {
                enum class scale_load_type_t { bcast, load, gather };

                for (int ur = 0; ur < reg_unroll; ur += ur_step) {
                    scale_load_type_t scale_load_type =
                        scale_load_type_t::bcast; // the best case

                    for (int r = ur + 1; r < ur + ur_step; ++r)
                        if (s_off[r] != s_off[r - 1] + 0)
                            scale_load_type = scale_load_type_t::load;

                    if (scale_load_type == scale_load_type_t::bcast) {
                        movss(xmm_scale, s_addr(s_off[ur]));
                        shufps(xmm_scale, xmm_scale, 0x0);
                        mulps(Xmm(ur), xmm_scale);
                        continue;
                    }

                    // bcast doesn't work, the next try -- load
                    for (int r = ur + 1; r < ur + ur_step; ++r)
                        if (s_off[r] != s_off[r - 1] + 1)
                            scale_load_type = scale_load_type_t::gather;

                    if (scale_load_type == scale_load_type_t::load) {
                        movups(xmm_scale, s_addr(s_off[ur]));
                        mulps(Xmm(ur), xmm_scale);
                        continue;
                    }

                    // load doesn't work as well
                    // so gather the scale factors one by one
                    for (int r = ur; r < ur + ur_step; ++r)
                        pinsrd(xmm_scale, s_addr(s_off[r]), r - ur);
                    mulps(Xmm(ur), xmm_scale);
                }
            }

            /* dst <-- beta * dst + xmm[:] */
            assert(prb_.beta == 0.f || prb_.beta == 1.f);
            if (prb_.beta == 1.f) {
                for (int ur = 0; ur < reg_unroll; ur += ur_step) {
                    if (prb_.otype == f32) {
                        /* non VEX instructions do not support unaligned
                         * memory for instructions other than movups. */
                        if (mayiuse(avx)) {
                            vaddps(Xmm(ur), o_addr(o_off[ur]));
                        } else {
                            /* register xmm(1) is unused */
                            movups(Xmm(1), o_addr(o_off[ur]));
                            addps(Xmm(ur), Xmm(1));
                        }
                    } else {
                        cvt2ps(Xmm(1), o_addr(o_off[ur]), prb_.otype);
                        vaddps(Xmm(ur), Xmm(1));
                    }
                }
            }
        } else {
            /* xmm[0] <-- scale * xmm[0] */
            if (prb_.scale_type == scale_type_t::COMMON) {
                for (int ur = 0; ur < reg_unroll; ur += ur_step)
                    mulss(Xmm(ur), xmm_scale);
            } else if (prb_.scale_type == scale_type_t::MANY) {
                for (int ur = 0; ur < reg_unroll; ur += ur_step) {
                    mulss(Xmm(ur), s_addr(s_off[ur]));
                }
            }

            /* dst <-- beta * dst + xmm[0] */
            assert(prb_.beta == 0.f || prb_.beta == 1.f);
            if (prb_.beta == 1.f) {
                for (int ur = 0; ur < reg_unroll; ur += ur_step) {
                    if (prb_.otype == f32) {
                        addss(Xmm(ur), o_addr(o_off[ur]));
                    } else {
                        if (prb_.otype == s32) {
                            vmovss(xmm_tmp, o_addr(o_off[ur]));
                        } else if (utils::one_of(prb_.otype, s8, u8)) {
                            pinsrb(xmm_tmp, o_addr(o_off[ur]), 0x0);
                        } else {
                            assert(!"unsupported o_type");
                        }
                        cvt2ps(xmm_tmp, xmm_tmp, prb_.otype);
                        addps(Xmm(ur), xmm_tmp);
                    }
                }
            }
        }

        for (int ur = 0; ur < reg_unroll; ur += ur_step) {
            if (prb_.otype != f32)
                cvt2int(Xmm(ur), prb_.otype, interim_f32 ? f32 : prb_.itype);
            store(o_addr(o_off[ur]), Xmm(ur), ur_step * otype_sz);
        }
    }

    void process_unroll_generic(int len) {
        const int blk = 8;

        int i_off[2 * blk] = {0};
        int o_off[2 * blk] = {0};
        int s_off[2 * blk] = {0};

        int curr = 0; // will switch between 0 and 1

        for (int off = 0; off < len; off += blk) {
            const int reg_unroll = nstl::min(off + blk, len) - off;

            /* compute offsets */
            for (int ur = off != 0 ? 0 : 1; ur < reg_unroll; ++ur) {
                const int ur_c = curr * blk + ur;
                const int ur_p = (ur_c - 1 + 2 * blk) % (2 * blk); // prev ur
                step(off + ur,
                        i_off[ur_p], o_off[ur_p], s_off[ur_p],
                        i_off[ur_c], o_off[ur_c], s_off[ur_c]);
            }

            process_unroll_generic_step(reg_unroll, i_off + curr * blk,
                    o_off + curr * blk, s_off + curr * blk);

            curr = 1 - curr;
        }
    }

    void loop_begin(Label &l, Reg64 reg_cnt, int len) {
        mov(reg_cnt, len);
        L(l);
    }

    void loop_end(Label &l, Reg64 reg_cnt, int len,
            int i_step, int o_step, int s_step) {
        add(reg_off_in, i_step * itype_sz);
        add(reg_off_out, o_step * otype_sz);
        if (prb_.scale_type == scale_type_t::MANY)
            add(reg_off_scale, s_step * stype_sz);
        dec(reg_cnt);
        jnz(l);

        sub(reg_off_in, len * i_step * itype_sz);
        sub(reg_off_out, len * o_step * otype_sz);
        if (prb_.scale_type == scale_type_t::MANY)
            sub(reg_off_scale, len * s_step * stype_sz);
    }

    bool simple_impl() {
        simple_impl_desc_t d;
        if (!simple_impl_desc_init(prb_, &d)) return false;

        const int nfu = d.ndims_full_unroll;
        const int ldu = d.len_last_dim_unroll;
        const int n_jit_loops = prb_.ndims - d.ndims_full_unroll;
        assert(n_jit_loops <= ndims_jit_loop_max);

        xor_(reg_off_in, reg_off_in);
        xor_(reg_off_out, reg_off_out);
        if (prb_.scale_type == scale_type_t::MANY)
            xor_(reg_off_scale, reg_off_scale);

        Label l_loop[3];
        Reg64 reg_cnt[3] = {r15, r14, r13};

        if (n_jit_loops > 2)
            loop_begin(l_loop[2], reg_cnt[2], n(nfu + 2));

        if (n_jit_loops > 1)
            loop_begin(l_loop[1], reg_cnt[1], n(nfu + 1));

        if (n_jit_loops > 0)
            loop_begin(l_loop[0], reg_cnt[0], n(nfu + 0) / ldu);

        const bool optimized = false
            || process_direct_copy<avx>(d.len_unroll)
            || process_direct_copy<sse42>(d.len_unroll)
            || process_unroll_tr8x8(d.len_unroll);
        if (!optimized)
            process_unroll_generic(d.len_unroll);

        if (n_jit_loops > 0)
            loop_end(l_loop[0], reg_cnt[0],
                    n(nfu + 0) / ldu, is(nfu + 0) * ldu, os(nfu + 0) * ldu,
                    ss(nfu + 0) * ldu);

        if (n_jit_loops > 1)
            loop_end(l_loop[1], reg_cnt[1],
                    n(nfu + 1), is(nfu + 1), os(nfu + 1), ss(nfu + 1));

        if (n_jit_loops > 2)
            loop_end(l_loop[2], reg_cnt[2],
                    n(nfu + 2), is(nfu + 2), os(nfu + 2), ss(nfu + 2));

        return true;
    }

    void impl() {
        if (simple_impl()) return;
        assert(!"no implementation available");
    }

    jit_uni_reorder_kernel_f32(const desc_t &desc)
        : kernel_t(desc), jit_generator() {
        itype_sz = data_type_size(prb_.itype);
        otype_sz = data_type_size(prb_.otype);
        stype_sz = sizeof(float);

        preamble();
#       define PARAM(x) ptr[abi_param1 + offsetof(call_param_t, x)]
        if (prb_.scale_type == scale_type_t::COMMON) {
            auto reg_ptr_scale_tmp = reg_ptr_in;
            mov(reg_ptr_scale_tmp, PARAM(scale));
            movups(xmm_scale, ptr[reg_ptr_scale_tmp]);
        } else if (prb_.scale_type == scale_type_t::MANY) {
            mov(reg_ptr_scale, PARAM(scale));
        }
        mov(reg_ptr_in, PARAM(in));
        mov(reg_ptr_out, PARAM(out));
#       undef PARAM

        if (mayiuse(avx)) {
            vxorps(xmm_zero, xmm_zero, xmm_zero);

            if (prb_.itype == data_type::u8 && prb_.otype == data_type::s8) {
                mov(reg_tmp.cvt32(), 0x7f7f7f7f);
                movd(xmm_4x127b, reg_tmp.cvt32());
            }
        }

        impl();
        postamble();
        ker_ = (void (*)(const call_param_t *))getCode();
    }

private:
    int itype_sz;
    int otype_sz;
    int stype_sz;

    Reg64 reg_ptr_in = rsi;
    Reg64 reg_ptr_out = rdx;
    Reg64 reg_ptr_scale = abi_not_param1;

    Reg64 reg_off_in = r8;
    Reg64 reg_off_out = r9;
    Reg64 reg_off_scale = r10;

    Reg64 reg_tmp = rax;

    Xmm xmm_scale = xmm15;
    Xmm xmm_zero = xmm14;
    Xmm xmm_4x127b = xmm13; // TODO: unite with xmm_zero
    Xmm xmm_tmp = xmm12;
};

status_t kernel_t::desc_init(kernel_t::desc_t &desc, const prb_t &prb,
        int ndims_ker_max) {
    desc.prb = prb;
    desc.prb.ioff = desc.prb.ooff = 0;

    if (ndims_ker_max > prb.ndims)
        return status::invalid_arguments;

    auto ndims_ker_max_f = [&]() {
        size_t cur_size = 1;
        for (int d = 0; d < prb.ndims; cur_size *= prb.nodes[d++].n)
            if (cur_size >= ker_prb_size_min) return d;
        return prb.ndims;
    };

    if (ndims_ker_max <= 0)
        ndims_ker_max = ndims_ker_max_f();

    /* traverse through kernel implementations */
    /* TODO: find a better way to do that... */
    desc.id = 0;
    for (int ndims_ker = ndims_ker_max; ndims_ker > 0; --ndims_ker) {
        desc.prb.ndims = ndims_ker;
        if (jit_uni_reorder_kernel_f32::applicable(desc.prb))
            return status::success;
    }

    return status::unimplemented;
}

kernel_t *kernel_t::create(const kernel_t::desc_t &desc) {
    switch (desc.id) {
    case 0: return new jit_uni_reorder_kernel_f32(desc);
    default: assert(!"unknown kernel id"); return nullptr;
    }

    return nullptr;
}

}

static void prb_block_for_cache(tr::prb_t &prb) {
    if (prb.nodes[0].is % 64 == 0 && prb.nodes[0].n > 16) {
        /** an attempt to use caches more efficient and
         * address the 4K-aliasing issue */
        /* TODO: improve the logic around here */
        int j = 1;
        for (; j < prb.ndims && prb.nodes[j].is != 1; ++j);
        if (j == prb.ndims) return;

        /* it makes sense to re-prioritize sequential read over
         * sequential write if the former would not trash the
         * cache, i.e. is == 1 and os % 2^smth != 0. Smth is
         * set to 2 at the moment */
        const int move_to = prb.nodes[j].os % 4 != 0 ? 0 : 1;
        if (j == move_to) return;

        if (prb.nodes[j].n > 16 && prb.nodes[j].n % 16 == 0)
            prb_node_split(prb, j, 16);

        prb_node_move(prb, j, move_to);
        DEBUG({ printf("cache: "); prb_dump(prb); });
    }
}

/** finds the maximum number of dimension the kernel should process and
 * optionally splits one of the dimension to achieve better balance between
 * parallel driver and the kernel. */
static void prb_thread_kernel_balance(tr::prb_t &prb, int &ndims_ker_max) {
    size_t sz_total = 1;
    for (int d = 0; d < prb.ndims; ++d)
        sz_total *= prb.nodes[d].n;

    /* sz_drv_min is the minimal size for the parallel
     * driver required for good parallelization */
    const size_t sz_drv_min = nstl::min<size_t>(
            16 * mkldnn_get_max_threads(),
            utils::div_up(sz_total, 1024));

    /* kdims -- # of dimensions processed by a kernel
     * sz_ker_cur -- product of the dimension processed by a kernel
     * sz_drv_cur -- product of the dimension processed by a driver */

    int kdims = prb.ndims;
    size_t sz_drv_cur = 1;
    for (; kdims > 1 && sz_drv_cur < sz_drv_min; --kdims)
        sz_drv_cur *= prb.nodes[kdims - 1].n;

    size_t sz_ker_cur = 1;
    for (int d = 0; d < kdims; ++d)
        sz_ker_cur *= prb.nodes[d].n;

    /* Initially kdims is chosen so that sz_drv_cur >= sz_drv_min.
     *
     * It might happen that for chosen kdims the sz_ker_cur is too small
     * (less than tr::ker_prb_size_min). In that case try to split the
     * innermost driver dimension into two, to increase sz_ker_cur. */
    bool want_borrow_ker_from_drv = true
        && kdims < prb.ndims
        && sz_ker_cur < tr::ker_prb_size_min
        && sz_drv_cur > sz_drv_min;
    if (want_borrow_ker_from_drv) {
        /* sz_want_borrow is the minimal sz, so that:
         *  o) sz_ker_cur * sz_want_borrow >= tr::ker_prb_size_min
         *  o) current innermost driver dimension is divisible by
         *     sz_want_borrow (so that we can evenly split that
         *     dimension into two)
         *
         *  In the worst case the minimal sz_want_borrow is equal
         *  to the innermost driver dimension itself. In that case
         *  we will sacrifice it in favor of kernel (is it fine?). */
        size_t sz_want_borrow
            = utils::div_up(tr::ker_prb_size_min, sz_ker_cur);
        for (; prb.nodes[kdims].n % sz_want_borrow; ++sz_want_borrow);
        if (sz_want_borrow != prb.nodes[kdims].n)
            prb_node_split(prb, kdims, sz_want_borrow);
        kdims += 1;
    }

    /* On the other hand it might happen that for chosen kdims
     * the sz_drv_cur is too small (less than sz_drv_min). In that case
     * try to split the outermost kernel dimension into two, to increase
     * sz_drv_cur. */
    bool want_borrow_drv_from_ker = true
        && sz_ker_cur > tr::ker_prb_size_min
        && sz_drv_cur < sz_drv_min;
    if (want_borrow_drv_from_ker) {
        size_t sz_want_borrow = utils::div_up(sz_drv_min, sz_drv_cur);
        for (; prb.nodes[kdims - 1].n % sz_want_borrow; ++sz_want_borrow);
        if (sz_want_borrow != prb.nodes[kdims - 1].n)
            prb_node_split(prb, kdims - 1,
                    prb.nodes[kdims - 1].n / sz_want_borrow);
    }

    ndims_ker_max = kdims;

    if (want_borrow_ker_from_drv || want_borrow_drv_from_ker) {
        DEBUG({ printf("split: "); prb_dump(prb);
                printf("ndims_ker_max = %d\n", ndims_ker_max); });
    }
}

struct jit_uni_reorder_t : public cpu_primitive_t {
    struct pd_t : public cpu_reorder_pd_t {
        using cpu_reorder_pd_t::cpu_reorder_pd_t;

        DECLARE_COMMON_PD_T("jit:uni", jit_uni_reorder_t);

        static status_t create(reorder_pd_t **reorder_pd,
                engine_t *engine, const primitive_attr_t *attr,
                engine_t *src_engine, const memory_desc_t *src_md,
                engine_t *dst_engine, const memory_desc_t *dst_md) {
            auto prb = tr::prb_t();

            status_t prb_init_status = prb_init(prb, *src_md, *dst_md, attr);
            if (prb_init_status != status::success) return prb_init_status;

            DEBUG({ printf("init : "); prb_dump(prb); });
            prb_normalize(prb);
            DEBUG({ printf("norm : "); prb_dump(prb); });
            prb_simplify(prb);
            DEBUG({ printf("smpl : "); prb_dump(prb); });

            prb_block_for_cache(prb);

            int ndims_ker_max;
            prb_thread_kernel_balance(prb, ndims_ker_max);

            tr::kernel_t::desc_t ker_desc;
            status_t ker_init_status
                = tr::kernel_t::desc_init(ker_desc, prb, ndims_ker_max);
            if (ker_init_status != status::success) return ker_init_status;

            const int ndims_driver = prb.ndims - ker_desc.prb.ndims;
            if (ndims_driver > jit_uni_reorder_t::ndims_driver_max)
                return status::unimplemented;

            DEBUG({ printf("ker  : "); prb_dump(ker_desc.prb); });

            auto _pd = new pd_t(engine, attr, src_engine, src_md, dst_engine,
                    dst_md);
            if (_pd == nullptr) return status::out_of_memory;
            if (_pd->init() != status::success) {
                delete _pd;
                return status::unimplemented;
            }
            _pd->prb_ = prb;
            _pd->ker_desc_ = ker_desc;
            return safe_ptr_assign<reorder_pd_t>(*reorder_pd, _pd);
        }

        tr::prb_t prb_;
        tr::kernel_t::desc_t ker_desc_;
    };

    jit_uni_reorder_t(const pd_t *apd): cpu_primitive_t(apd) {
        kernel_ = tr::kernel_t::create(pd()->ker_desc_);
        assert(kernel_);
    }
    ~jit_uni_reorder_t() { delete kernel_; }

    void omp_driver_0d(int off, const char *in, char *out,
            const float *scale) const {
        tr::call_param_t c{in, out, scale};
        (*kernel_)(&c);
    }

    void omp_driver_1d(int ithr, int nthr, int off, const char *in, char *out,
            const float *scale) const {
        const tr::node_t *ns = pd()->prb_.nodes + off;
        for_nd(ithr, nthr, (ptrdiff_t)ns[0].n, [&](ptrdiff_t d0) {
            auto c = tr::call_param_t();
            c.in = in + d0 * ns[0].is * data_type_size(pd()->prb_.itype);
            c.out = out + d0 * ns[0].os * data_type_size(pd()->prb_.otype);
            c.scale = scale + d0 * ns[0].ss;
            (*kernel_)(&c);
        });
    }

    void omp_driver_2d(int ithr, int nthr, int off, const char *in, char *out,
            const float *scale) const {
        const tr::node_t *ns = pd()->prb_.nodes + off;
        for_nd(ithr, nthr, (ptrdiff_t)ns[1].n, (ptrdiff_t)ns[0].n,
                [&](ptrdiff_t d1, ptrdiff_t d0) {
            auto c = tr::call_param_t();
            c.in = in + (d0 * ns[0].is + d1 * ns[1].is)
                * data_type_size(pd()->prb_.itype);
            c.out = out + (d0 * ns[0].os + d1 * ns[1].os)
                * data_type_size(pd()->prb_.otype);
            c.scale = scale + d0 * ns[0].ss + d1 * ns[1].ss;
            (*kernel_)(&c);
        });
    }

    void omp_driver_3d(int ithr, int nthr, int off, const char *in, char *out,
            const float *scale) const {
        const tr::node_t *ns = pd()->prb_.nodes + off;
        for_nd(ithr, nthr, (ptrdiff_t)ns[2].n, (ptrdiff_t)ns[1].n,
                (ptrdiff_t)ns[0].n,
                [&](ptrdiff_t d2, ptrdiff_t d1, ptrdiff_t d0) {
            auto c = tr::call_param_t();
            c.in = in + (d0 * ns[0].is + d1 * ns[1].is + d2 * ns[2].is)
                * data_type_size(pd()->prb_.itype);
            c.out = out + (d0 * ns[0].os + d1 * ns[1].os + d2 * ns[2].os)
                * data_type_size(pd()->prb_.otype);
            c.scale = scale + d0 * ns[0].ss + d1 * ns[1].ss + d2 * ns[2].ss;
            (*kernel_)(&c);
        });
    }

    void omp_driver_4d(int ithr, int nthr, int off, const char *in, char *out,
            const float *scale) const {
        const tr::node_t *ns = pd()->prb_.nodes + off;
        for_nd(ithr, nthr, (ptrdiff_t)ns[3].n, (ptrdiff_t)ns[2].n,
                (ptrdiff_t)ns[1].n, (ptrdiff_t)ns[0].n,
                [&](ptrdiff_t d3, ptrdiff_t d2, ptrdiff_t d1, ptrdiff_t d0) {
            auto c = tr::call_param_t();
            c.in = in + (d0 * ns[0].is + d1 * ns[1].is + d2 * ns[2].is
                    + d3 * ns[3].is) * data_type_size(pd()->prb_.itype);
            c.out = out + (d0 * ns[0].os + d1 * ns[1].os + d2 * ns[2].os
                    + d3 * ns[3].os) * data_type_size(pd()->prb_.otype);
            c.scale = scale + d0 * ns[0].ss + d1 * ns[1].ss + d2 * ns[2].ss
                + d3 * ns[3].ss;
            (*kernel_)(&c);
        });
    }

    void omp_driver(const char *in, char *out, const float *scale) const {
        in += pd()->prb_.ioff * data_type_size(pd()->prb_.itype);
        out += pd()->prb_.ooff * data_type_size(pd()->prb_.otype);

        DEBUG({ printf("prb : "); tr::prb_dump(pd()->prb_); });
        DEBUG({ printf("ker : "); tr::prb_dump(pd()->ker_desc_.prb); });

        int ndims = pd()->prb_.ndims;
        int ndims_ker = pd()->ker_desc_.prb.ndims;
        assert(ndims - ndims_ker <= ndims_driver_max);

        if (ndims - ndims_ker == 0) {
            omp_driver_0d(ndims_ker, in, out, scale);
        } else {
            parallel(0, [&](const int ithr, const int nthr) {
                switch (ndims - ndims_ker) {
                case 1: omp_driver_1d(ithr, nthr, ndims_ker, in, out, scale); break;
                case 2: omp_driver_2d(ithr, nthr, ndims_ker, in, out, scale); break;
                case 3: omp_driver_3d(ithr, nthr, ndims_ker, in, out, scale); break;
                case 4: omp_driver_4d(ithr, nthr, ndims_ker, in, out, scale); break;
                default: assert(!"unimplemented");
                }
            });
        }
    }

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        auto in = CTX_IN_MEM(const char *, MKLDNN_ARG_FROM);
        auto out = CTX_OUT_MEM(char *, MKLDNN_ARG_TO);

        omp_driver(in, out, pd()->attr()->output_scales_.scales_);

        return status::success;
    }

    enum { ndims_driver_max = 4 };

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
    tr::kernel_t *kernel_;
};

status_t jit_uni_reorder_create(reorder_pd_t **reorder_pd,
        engine_t *engine, const primitive_attr_t *attr,
        engine_t *src_engine, const memory_desc_t *src_md,
        engine_t *dst_engine, const memory_desc_t *dst_md) {
    return jit_uni_reorder_t::pd_t::create(reorder_pd, engine, attr,
            src_engine, src_md, dst_engine, dst_md);
}

}
}
}
