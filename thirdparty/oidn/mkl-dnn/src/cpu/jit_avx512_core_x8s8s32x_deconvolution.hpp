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

#ifndef CPU_JIT_AVX512_CORE_U8S8S32X_DECONVOLUTION_HPP
#define CPU_JIT_AVX512_CORE_U8S8S32X_DECONVOLUTION_HPP

#include "c_types_map.hpp"
#include "cpu_primitive.hpp"
#include "cpu_memory.hpp"
#include "mkldnn_thread.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"
#include "nstl.hpp"

#include "cpu_deconvolution_pd.hpp"
#include "jit_generator.hpp"
#include "jit_primitive_conf.hpp"
#include "jit_uni_eltwise.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

typedef enum {
    no_last_block = 0x1U,
    last_ic_block = 0x2U,
    last_sp_block = 0x4U,
} ker_block_t;

struct jit_avx512_core_x8s8s32x_deconv_fwd_kernel : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_core_x8s8s32x_deconv_fwd_ker_t);

    jit_avx512_core_x8s8s32x_deconv_fwd_kernel(
            const jit_conv_conf_t &ajcp, const primitive_attr_t &attr)
        : jcp(ajcp), attr_(attr), eltwise_injector_(nullptr) {
        if (jcp.with_eltwise)
            eltwise_injector_ = new jit_uni_eltwise_injector_f32<avx512_common>(
                    this, jcp.eltwise);
        generate();
        jit_ker = (void (*)(jit_deconv_call_s *))getCode();
    }

    ~jit_avx512_core_x8s8s32x_deconv_fwd_kernel() {
            delete eltwise_injector_;
    }

    static bool post_ops_ok(jit_conv_conf_t &jcp,
            const primitive_attr_t &attr);

    static status_t init_conf(jit_conv_conf_t &jcp,
            const deconvolution_desc_t &cd,
            memory_desc_t &src_md,
            memory_desc_t &weights_md,
            memory_desc_t &dst_md,
            const bool with_bias,
            memory_desc_t &bias_md,
            const primitive_attr_t &attr);

    static void init_scratchpad(memory_tracking::registrar_t &scratchpad,
            const jit_conv_conf_t &jcp, const primitive_attr_t &attr);

    const jit_conv_conf_t &jcp;
    const primitive_attr_t &attr_;
    void (*jit_ker)(jit_deconv_call_s *);
private:
    jit_uni_eltwise_injector_f32<avx512_common> *eltwise_injector_;
    using reg64_t = const Xbyak::Reg64;
    using zmm_t = const Xbyak::Zmm;
    using xmm_t = const Xbyak::Xmm;

    reg64_t reg_src = r8;
    reg64_t reg_filt = r9;
    reg64_t reg_dst = r10;
    reg64_t param1 = abi_param1;
    reg64_t reg_kh = abi_not_param1;
    reg64_t reg_nur_w = rbx;
    reg64_t reg_bias = rdx;
    reg64_t reg_icb = reg_bias;
    reg64_t reg_ptr_scales = rax;
    reg64_t reg_oc_blocks = rsi;

    reg64_t aux_reg_src = r11;
    reg64_t aux_reg_filt = r12;

    reg64_t reg_compensation = r14;
    reg64_t reg_scratch = r14;
    reg64_t reg_ptr_sum_scale = r11;
    reg64_t reg_bias_alpha = abi_not_param1;
    reg64_t reg_overflow = rax;
    reg64_t reg_comp_strides = reg_overflow;

    Xbyak::Opmask ktail_mask = Xbyak::Opmask(2);
    zmm_t zmm_tmp = zmm_t(28);
    zmm_t zmm_one = zmm_t(29);
    /* used during write-out section of store_output */
    zmm_t zmm_zero = zmm_t(31);
    zmm_t zmm_wei = zmm_t(31);

    /* signed input */
    zmm_t zmm_shift = zmm_t(30);
    zmm_t zmm_comp = zmm_t(30);
    zmm_t zmm_bias = zmm_t(31);
    zmm_t zmm_prev_dst = zmm_t(31);

    zmm_t zmm_out(int i_ur, int i_oc) {
        int idx = i_ur * jcp.nb_oc_blocking + i_oc;
        assert(idx < 31);
        return zmm_t(idx);
    }
    zmm_t zmm_inp(int i_ic, int nb_x_blocking) {
        int idx = i_ic + nb_x_blocking * jcp.ur_w;
        assert(idx < 31);
        return zmm_t(idx);
    }
    zmm_t zmm_bias_alpha() {
        return zmm_t(jcp.nb_oc_blocking * jcp.ur_w);
    }
    xmm_t xmm_bias_alpha() {
        return xmm_t(jcp.nb_oc_blocking * jcp.ur_w);
    }

    int get_ow_start(int ki, int l_overflow) {
        int res = (jcp.ow - 1 + jcp.r_pad) % jcp.stride_w
                + l_overflow * jcp.stride_w
                - (jcp.kw - 1 - ki) * (jcp.dilate_w + 1);
        while (res < 0)
            res += jcp.stride_w;
        return res;
    }

    int get_ow_end(int ur_w, int ki, int r_overflow) {
        if (utils::one_of(ur_w, jcp.ow, jcp.ur_w_tail))
                ur_w += nstl::min(0, jcp.r_pad); // remove negative padding
        int res = (ur_w - 1 + jcp.l_pad) % jcp.stride_w
            + r_overflow * jcp.stride_w - ki * (jcp.dilate_w + 1);
        while (res < 0)
            res += jcp.stride_w;
        return ur_w - res;
    }
    bool maybe_eltwise(int position);
    void compute_eltwise(int ur_w);
    void prepare_output(int ur_w);
    void store_output(int ur_w, bool last_oc_block);
    void compute_ker(int ur_w, int l_overflow, int r_overflow,
             ker_block_t last_ic_block_flag, bool h_padded = false);
    void kh_loop(int ur_w, int pad_l, int pad_r, ker_block_t last_ker_block);
    void icb_loop(int ur_w, int pad_l, int pad_r, bool last_block);
    void generate();
    void cvt2ps(data_type_t type_in, zmm_t zmm_in, const Xbyak::Operand &op,
        bool mask_flag);
};

template <impl::data_type_t src_type, impl::data_type_t dst_type>
struct _jit_avx512_core_x8s8s32x_deconvolution_fwd_t : public cpu_primitive_t {
    struct pd_t : public cpu_deconvolution_fwd_pd_t {
        using cpu_deconvolution_fwd_pd_t::cpu_deconvolution_fwd_pd_t;

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("jit_deconvolution:", avx512_core, ""),
                _jit_avx512_core_x8s8s32x_deconvolution_fwd_t<src_type, dst_type>);

        status_t init() {
            bool ok = true
                && is_fwd()
                && (desc()->alg_kind & alg_kind::deconvolution_direct)
                && desc()->src_desc.data_type == src_type
                && desc()->dst_desc.data_type == dst_type
                && IMPLICATION(with_bias(), utils::one_of(
                            desc()->bias_desc.data_type, data_type::f32,
                            data_type::s32, data_type::s8, data_type::u8))
                && desc()->accum_data_type == data_type::s32;
            if (!ok) return status::unimplemented;

            status_t status = jit_avx512_core_x8s8s32x_deconv_fwd_kernel::
                init_conf(jcp_, *desc(), src_md_, weights_md_, dst_md_,
                        with_bias(), bias_md_, *attr());

            if (status != status::success) return status;

            auto scratchpad = scratchpad_registry().registrar();
            jit_avx512_core_x8s8s32x_deconv_fwd_kernel::init_scratchpad(scratchpad,
                    jcp_, *attr());

            return status::success;
        }

        jit_conv_conf_t jcp_;
    };

    _jit_avx512_core_x8s8s32x_deconvolution_fwd_t(const pd_t *apd)
        : cpu_primitive_t(apd)
    {
        kernel_ = new jit_avx512_core_x8s8s32x_deconv_fwd_kernel(pd()->jcp_,
                *pd()->attr());
    }

    ~_jit_avx512_core_x8s8s32x_deconvolution_fwd_t() { delete kernel_; }

    typedef typename prec_traits<src_type>::type src_data_t;
    typedef typename prec_traits<data_type::s8>::type wei_data_t;
    typedef typename prec_traits<dst_type>::type dst_data_t;

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        if(pd()->ndims() == 3)
            execute_forward_1d(ctx);
        else
            execute_forward_2d(ctx);
        return status::success;
    }

private:
    void execute_forward_1d(const exec_ctx_t &ctx) const;
    void execute_forward_2d(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
    jit_avx512_core_x8s8s32x_deconv_fwd_kernel *kernel_;
};

}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
