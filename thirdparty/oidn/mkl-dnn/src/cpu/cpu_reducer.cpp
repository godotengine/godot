/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
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

#include "mkldnn_thread.hpp"
#include "mkldnn_types.h"
#include "nstl.hpp"
#include "utils.hpp"

#include "cpu_reducer.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace memory_tracking::names;

void reduce_balancer_t::balance() {
    using namespace nstl;
    using namespace utils;

    assert(nthr_ > 0 && job_size_ > 0 && njobs_ > 0 && reduction_size_ > 0);

    const int job_complexity = 1;

    const int min_njobs_per_group = max(1, njobs_ / nthr_);
    const int max_njobs_per_group = max(1,
            static_cast<int>(max_buffer_size_ / (nthr_ * job_size_)));

    /* initial guess */
    int ngroups = min(njobs_ / min_njobs_per_group, nthr_);
    int nthr_per_group = syncable_ ? min(nthr_ / ngroups, reduction_size_) : 1;
    int njobs_per_group_ub = div_up(njobs_, ngroups);

    /* rough upper-bound estimation, will be fixed during brute force */
    size_t thread_complexity_ub = njobs_ * job_size_ * reduction_size_;

    /* brute force parameters for the best balance... */
    for (int c_njobs_per_group = min_njobs_per_group;
            c_njobs_per_group < njobs_; ++c_njobs_per_group) {
        /* current assumption */
        int c_ngroups = min(njobs_ / c_njobs_per_group, nthr_);
        int c_nthr_per_group = syncable_
            ? min(nthr_ / c_ngroups, reduction_size_) : 1;
        int c_njobs_per_group_ub = div_up(njobs_, c_ngroups);

        if (c_nthr_per_group > 1 && c_njobs_per_group_ub > max_njobs_per_group)
            continue;

        int c_thread_reduction_ub = div_up(reduction_size_, c_nthr_per_group);
        size_t c_group_size_ub = job_size_ * c_njobs_per_group_ub;
        size_t c_thread_complexity_ub = c_group_size_ub * (
                job_complexity * c_thread_reduction_ub
                + (c_nthr_per_group != 1));

        if (c_thread_complexity_ub < thread_complexity_ub) {
            ngroups = c_ngroups;
            nthr_per_group = c_nthr_per_group;
            njobs_per_group_ub = c_njobs_per_group_ub;
            thread_complexity_ub = c_thread_complexity_ub;
        }
    }

    assert(njobs_per_group_ub <= max_njobs_per_group || nthr_per_group == 1);
    assert(ngroups * nthr_per_group <= nthr_);
    assert((size_t)njobs_per_group_ub * job_size_ * nthr_ <= max_buffer_size_
            || nthr_per_group == 1); /* no reduction buffer overflow */
    assert(IMPLICATION(!syncable_, nthr_per_group == 1));

    ngroups_ = ngroups;
    nthr_per_group_ = nthr_per_group;
    njobs_per_group_ub_ = njobs_per_group_ub;
}

/* reducer jit-ted driver */

using namespace Xbyak;

template <impl::data_type_t data_type>
struct reducer_2d_driver_t: public c_compatible {
    typedef typename prec_traits<data_type>::type data_t;

    reducer_2d_driver_t(int n_src, size_t src_ld,
            size_t src_step, size_t dst_step, bool nullify_dst)
        : n_src_(n_src), src_ld_(src_ld), src_step_(src_step)
        , dst_step_(dst_step), nullify_dst_(nullify_dst), ker_(nullptr) {}
    virtual ~reducer_2d_driver_t() {}
    void operator()(data_t *dst, const data_t *srcs, size_t ny, size_t nx)
    { assert(ker_); ker_(dst, srcs, ny, nx); }

protected:
    int n_src_;
    size_t src_ld_, src_step_, dst_step_;
    bool nullify_dst_;
    void (*ker_)(data_t *dst, const data_t *srcs, size_t ny, size_t nx);
};

template <impl::data_type_t data_type, cpu_isa_t isa>
struct reducer_2d_driver_f_s_32_t: public reducer_2d_driver_t<data_type>,
    public jit_generator
{
    DECLARE_CPU_JIT_AUX_FUNCTIONS(reducer_2d_driver_f_s_32_t)

    /* cpu specific part */
    using Vmm = typename utils::conditional<isa == avx2, Ymm, Zmm>::type;
    const AddressFrame &vmmword = (isa == avx2) ? yword : zword;
    void uni_vadd(const Xmm& x1, const Xmm& x2, const Operand& op)
    { if (data_type == data_type::f32) vaddps(x1, x2, op);
      else vpaddd(x1, x2, op); }
    void uni_add(const Xmm& x1, const Operand& op)
    { if (data_type == data_type::f32) addss(x1, op); else paddd(x1, op); }

    const int vlen = cpu_isa_traits<isa>::vlen;
    const int typesize
        = sizeof(typename mkldnn::impl::prec_traits<data_type>::type);
    Xbyak::Reg64 reg_dst = abi_param1;
    Xbyak::Reg64 reg_src = abi_param2;
    Xbyak::Reg64 reg_ny = abi_param3;
    Xbyak::Reg64 reg_nx = abi_param4;

    Xbyak::Reg64 reg_x = rax;
    Xbyak::Reg64 reg_src_id = r10;

    reducer_2d_driver_f_s_32_t(int n_src, size_t src_ld, size_t src_step,
            size_t dst_step, bool nullify_dst)
        : reducer_2d_driver_t<data_type>(n_src, src_ld, src_step,
                dst_step, nullify_dst)
    { generate(); }

    void nullify_dst(int nloads, int load_len) {
        UNUSED(load_len);
        for (int i = 0; i < nloads; ++i)
            uni_vpxor(Vmm(i), Vmm(i), Vmm(i));
        /* prefetches[dst] ? */
    }

    void load_dst(int nloads, int load_len) {
        for (int i = 0; i < nloads; ++i) {
            if (load_len == typesize)
                movd(Xmm(i), ptr[reg_dst + i * load_len]);
            else if (load_len == vlen)
                vmovups(Vmm(i), ptr[reg_dst + i * load_len]);
            else
                assert(!"unsupported");
        }
    }

    void store_dst(int nloads, int load_len) {
        for (int i = 0; i < nloads; ++i) {
            if (load_len == typesize)
                movd(ptr[reg_dst + i * load_len], Xmm(i));
            else if (load_len == vlen)
                vmovups(ptr[reg_dst + i * load_len], Vmm(i));
            else
                assert(!"unsupported");
        }
    }

    void accumulate(int nloads, int load_len, size_t base_off) {
        for (int i = 0; i < nloads; ++i) {
            size_t off = base_off + i * load_len;

            if (load_len == typesize)
                uni_add(Xmm(i), ptr[reg_src + off]);
            else if (load_len == vlen)
                uni_vadd(Vmm(i), Vmm(i), vmmword[reg_src + off]);
            else
                assert(!"unsupported");
        }
    }

    void loop_x() {
        const int nloads[] = {cpu_isa_traits<isa>::n_vregs, 1, 1};
        const int nbranches = sizeof(nloads) / sizeof(nloads[0]);

        const int load_len[nbranches] = {vlen, vlen, typesize};
        Label loop_x_label[nbranches + 1];

        mov(reg_x, reg_nx);

        for (int id = 0; id < nbranches; ++id) {
            L(loop_x_label[id]);

            cmp(reg_x, nloads[id] * load_len[id]);
            jl(loop_x_label[id + 1], T_NEAR);

            if (this->nullify_dst_)
                nullify_dst(nloads[id], load_len[id]);
            else
                load_dst(nloads[id], load_len[id]);

            if (nloads[id] > 1) {
                Label loop_srcs;
                mov(reg_src_id, this->n_src_);
                L(loop_srcs);

                accumulate(nloads[id], load_len[id], 0);
                add(reg_src, this->src_ld_ * typesize);

                dec(reg_src_id);
                jnz(loop_srcs, T_NEAR);

                sub(reg_src, this->n_src_ * this->src_ld_ * typesize);
            } else {
                for (int src_id = 0; src_id < this->n_src_; ++src_id) {
                    const size_t base_off = src_id * this->src_ld_ * typesize;
                    accumulate(nloads[id], load_len[id], base_off);
                }
            }

            store_dst(nloads[id], load_len[id]);

            add(reg_src, nloads[id] * load_len[id]);
            add(reg_dst, nloads[id] * load_len[id]);

            sub(reg_x, nloads[id] * load_len[id]);

            jmp(loop_x_label[id], T_NEAR);
        }

        L(loop_x_label[nbranches]);

        /* restore address registers */
        sub(reg_src, reg_nx);
        sub(reg_dst, reg_nx);
    }

    void generate() {
        assert(isa == avx2 || isa == avx512_common || isa == avx512_mic);

        preamble();

        shl(reg_nx, 2);

        Label ny_loop;
        L(ny_loop);

        loop_x();

        add(reg_dst, this->dst_step_ * typesize);
        add(reg_src, this->src_step_ * typesize);

        dec(reg_ny);
        jnz(ny_loop, T_NEAR);

        postamble();
        this->ker_ = reinterpret_cast<decltype(this->ker_)>(
            const_cast<uint8_t*>(this->getCode()));
    }
};

template <impl::data_type_t data_type>
inline reducer_2d_driver_t<data_type> *create_reduce_2d_drv(int n_src,
        size_t src_ld, size_t src_step, size_t dst_step, bool nullify_dst) {
    if (mayiuse(avx512_common))
        return new reducer_2d_driver_f_s_32_t<data_type, avx512_common>(n_src,
            src_ld, src_step, dst_step, nullify_dst);
    else if (mayiuse(avx2))
        return new reducer_2d_driver_f_s_32_t<data_type, avx2>(n_src, src_ld,
            src_step, dst_step, nullify_dst);
    assert(!"unimplemented");
    return nullptr;
}

/* cpu_reducer_t */

template <impl::data_type_t data_type>
void cpu_reducer_t<data_type>::conf_t::init_scratchpad(
        memory_tracking::registrar_t &scratchpad) const {
    if (balancer_.nthr_per_group_ == 1) return;

    const size_t space_size = balancer_.ngroups_
        * (balancer_.nthr_per_group_ - 1)
        * cpu_reducer_t<data_type>::space_per_thread(balancer_);
    scratchpad.book(key_reducer_space, sizeof(data_t) * space_size, PAGE_4K);
    scratchpad.book(key_reducer_space_bctx,
            sizeof(simple_barrier::ctx_t) * balancer_.ngroups_);
}

template <impl::data_type_t data_type>
cpu_reducer_t<data_type>::cpu_reducer_t(const conf_t &conf)
    : conf_(conf), drv_(nullptr)
{
    if (balancer().nthr_per_group_ == 1) return;

    drv_ = create_reduce_2d_drv<data_type>(balancer().nthr_per_group_ - 1,
            space_per_thread(balancer()), 0, 0, false);
}

template <impl::data_type_t data_type>
cpu_reducer_t<data_type>::~cpu_reducer_t() { delete drv_; }

template <impl::data_type_t data_type>
typename cpu_reducer_t<data_type>::data_t *
cpu_reducer_t<data_type>::get_local_ptr(int ithr, data_t *dst,
        const memory_tracking::grantor_t &scratchpad) const {
    const int id_in_grp = balancer().id_in_group(ithr);

    /* threads 0 from each group writes directly to the destination */
    if (id_in_grp == 0)
        return dst + balancer().ithr_job_off(ithr) * balancer().job_size_;

    const int grp_id = balancer().group_id(ithr);
    const int offset_factor = grp_id * (balancer().nthr_per_group_ - 1)
        + (id_in_grp - 1);

    auto space = scratchpad.template get<data_t>(key_reducer_space);
    return space + offset_factor * space_per_thread(balancer());
}

template <impl::data_type_t data_type>
void cpu_reducer_t<data_type>::reduce_nolock(int ithr, data_t *dst,
        const memory_tracking::grantor_t &scratchpad) const {
    bool redundant_reduction = balancer().nthr_per_group_ == 1
        || balancer().idle(ithr);
    if (redundant_reduction) return;

#ifdef SIMPLE_IMPL
    if (balancer().id_in_group(ithr) != 0)
        return; /* only threads 0 do the reduction */

    const int njobs_in_grp = balancer().ithr_njobs(ithr);
    data_t *d = get_local_ptr(ithr, dst, scratchpad);
    for (int id_in_grp = 1; id_in_grp < balancer_.nthr_per_group_; ++id_in_grp)
    {
        const data_t *space = get_local_ptr(ithr + id_in_grp, dst, scratchpad);
        for (size_t i = 0; i < (size_t)njobs_in_grp * balancer().job_size_; ++i)
            d[i] += space[i];
    }
#else
    using namespace utils;

    const int id_in_grp = balancer().id_in_group(ithr);
    const int njobs_in_grp = balancer().ithr_njobs(ithr);
    const size_t cl = 64 / sizeof(data_t);

    const size_t reduction_size = njobs_in_grp * balancer().job_size_;
    size_t start{0}, end{0};
    balance211(div_up(reduction_size, cl), balancer().nthr_per_group_,
            id_in_grp, start, end);

    if (start == end) return;

    data_t *d = get_local_ptr(ithr - id_in_grp, dst, scratchpad) + start * cl;
    const data_t *space = get_local_ptr(ithr - id_in_grp + 1, dst, scratchpad)
        + start * cl;
    const size_t len = nstl::min(end * cl, reduction_size) - start * cl;

    (*drv_)(d, space, 1, len);
#endif
}

template struct cpu_reducer_t<data_type::f32>;
template struct cpu_reducer_t<data_type::s32>;

/* cpu_reducer_2d_t */

template <impl::data_type_t data_type>
void cpu_reducer_2d_t<data_type>::conf_t::init_scratchpad(
        memory_tracking::registrar_t &scratchpad) const {
    if (balancer_.nthr_per_group_ == 1) return;

    const size_t space_size = balancer_.ngroups_ * balancer_.nthr_per_group_
        * cpu_reducer_2d_t<data_type>::space_per_thread(balancer_);
    scratchpad.book(key_reducer_space, sizeof(data_t) * space_size);
    scratchpad.book(key_reducer_space_bctx,
            sizeof(simple_barrier::ctx_t) * balancer_.ngroups_);
}

template <impl::data_type_t data_type>
cpu_reducer_2d_t<data_type>::cpu_reducer_2d_t(const conf_t &conf)
    : conf_(conf), drv_(nullptr)
{
    if (balancer().nthr_per_group_ == 1) return;

    drv_ = create_reduce_2d_drv<data_type>(balancer().nthr_per_group_,
            space_per_thread(balancer()), conf_.job_size_x_, conf_.dst_x_,
            true);
}

template <impl::data_type_t data_type>
cpu_reducer_2d_t<data_type>::~cpu_reducer_2d_t() { delete drv_; }

template <impl::data_type_t data_type>
typename cpu_reducer_2d_t<data_type>::data_t *cpu_reducer_2d_t<data_type>::
get_local_ptr(int ithr, const memory_tracking::grantor_t &scratchpad) const {
    const int id_in_grp = balancer().id_in_group(ithr);
    const int grp_id = balancer().group_id(ithr);
    const int offset_factor = grp_id * balancer().nthr_per_group_ + id_in_grp;
    auto space = scratchpad.template get<data_t>(key_reducer_space);
    return space + offset_factor * space_per_thread(balancer());
}

template <impl::data_type_t data_type>
int cpu_reducer_2d_t<data_type>::choose_x_blocking(int nx, int ny,
        int nthr_per_grp) const {
    // find x_blocking for better balance reducing work between threads
    assert(conf_.x_block_ > 0 && nx > conf_.x_block_
            && nx % conf_.x_block_ == 0);
    int x_blocking = nx / conf_.x_block_;
    int min_x_blocking =
            utils::div_up(x_blocking, nstl::max(1, nthr_per_grp / ny));
    while (true) {
        if (x_blocking % 2 == 0 && x_blocking >= min_x_blocking * 2)
            x_blocking /= 2;
        else if (x_blocking % 3 == 0 && x_blocking >= min_x_blocking * 3)
            x_blocking /= 3;
        else
            break;
    }
    if (x_blocking >= min_x_blocking * 4) x_blocking = 1;
    x_blocking *= conf_.x_block_;
    return x_blocking;
}

template <impl::data_type_t data_type>
void cpu_reducer_2d_t<data_type>::reduce_block(const data_t* space_base,
        data_t *dst, int job, int start_y, int start_x,
        int ny_start, int nx_start, int ny_step, int nx_step) const {
    data_t *d = dst + (start_y + ny_start) * conf_.dst_x_
                    + start_x + nx_start;
    const data_t *space = space_base + job * balancer().job_size_
                            + ny_start * conf_.job_size_x_ + nx_start;
#ifdef SIMPLE_IMPL
    for (int idg = 0; idg < balancer().nthr_per_group_; ++idg) {
        const data_t *w = &space[idg * space_per_thread(balancer())];
        for (int y = 0; y < ny_step; ++y)
            for (int x = 0; x < nx_step; ++x) {
                d[y * conf_.dst_x_ + x]
                    = (idg == 0 ? 0 : d[y * conf_.dst_x_ + x])
                    + w[y * conf_.job_size_x_ + x];
            }
    }
#else
    (*drv_)(d, space, ny_step, nx_step);
#endif
}

template <impl::data_type_t data_type>
void cpu_reducer_2d_t<data_type>::reduce_nolock(int ithr, data_t *dst,
        const memory_tracking::grantor_t &scratchpad) const {
    bool redundant_reduction = balancer().nthr_per_group_ == 1
        || balancer().idle(ithr);
    if (redundant_reduction) return;

    const int id_in_grp = balancer().id_in_group(ithr);
    const int njobs_in_grp = balancer().ithr_njobs(ithr);
    const int njobs_x = utils::div_up(conf_.dst_x_, conf_.job_size_x_);
    const int global_job_start = balancer().ithr_job_off(ithr);

    const data_t *space_base = get_local_ptr(ithr - id_in_grp, scratchpad);

    const int pr_grps = nstl::min(njobs_in_grp, balancer().nthr_per_group_);
    const int pr_nthr_per_grp = balancer().nthr_per_group_ / pr_grps;

    if (id_in_grp >= pr_grps * pr_nthr_per_grp)
        return; /* idle */

    const int pr_my_grp = id_in_grp / pr_nthr_per_grp;
    const int pr_my_id = id_in_grp % pr_nthr_per_grp;

    int pr_job_start{0}, pr_job_end{0};
    balance211(njobs_in_grp, pr_grps, pr_my_grp, pr_job_start, pr_job_end);

    for (int j = pr_job_start; j < pr_job_end; ++j) {
        const int global_job = global_job_start + j;
        const int j_y = global_job / njobs_x;
        const int j_x = global_job % njobs_x;
        const int start_y = j_y * conf_.job_size_y_;
        const int start_x = j_x * conf_.job_size_x_;
        const int ny = nstl::min(conf_.dst_y_ - start_y, conf_.job_size_y_);
        const int nx = nstl::min(conf_.dst_x_ - start_x, conf_.job_size_x_);
        int x_blocking = choose_x_blocking(nx, ny, pr_nthr_per_grp);

        int nxy_start{0}, nxy_end{0};
        balance211(ny * nx / x_blocking, pr_nthr_per_grp, pr_my_id,
                    nxy_start, nxy_end);
        if (nxy_start == nxy_end) continue;
        nxy_start *= x_blocking;
        nxy_end *= x_blocking;

        int nxy = nxy_start;
        if (nxy % nx != 0) {
            int nx_step = nstl::min(nx - nxy % nx, nxy_end - nxy);
            reduce_block(space_base, dst, j, start_y, start_x,
                        nxy / nx, nxy % nx, 1, nx_step);
            nxy += nx_step;
        }
        if ((nxy_end - nxy) > nx) {
            int ny_step = (nxy_end - nxy) / nx;
            reduce_block(space_base, dst, j, start_y, start_x,
                        nxy / nx, nxy % nx, ny_step, nx);
            nxy += nx * ny_step;
        }
        if ((nxy_end - nxy) > 0) {
            reduce_block(space_base, dst, j, start_y, start_x,
                        nxy / nx, nxy % nx, 1, nxy_end - nxy);
        }
    }
}

template struct cpu_reducer_2d_t<data_type::f32>;
template struct cpu_reducer_2d_t<data_type::s32>;

/* accumulator section */

template <impl::data_type_t data_type>
cpu_accumulator_1d_t<data_type>::cpu_accumulator_1d_t(): drv_(nullptr) {
    drv_ = create_reduce_2d_drv<data_type>(1, 0, 0, 0, false);
}

template <impl::data_type_t data_type>
cpu_accumulator_1d_t<data_type>::~cpu_accumulator_1d_t() {
    delete drv_;
}

template <impl::data_type_t data_type>
void cpu_accumulator_1d_t<data_type>::accumulate(data_t *dst,
        const data_t *src, size_t size) {
    (*drv_)(dst, src, 1, size);
}

template struct cpu_accumulator_1d_t<data_type::f32>;
template struct cpu_accumulator_1d_t<data_type::s32>;

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
