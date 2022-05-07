/*******************************************************************************
* Copyright 2019 Intel Corporation
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

/*
 * Cell execution LSTM
 */

#include "rnn_utils.hpp"
#include "../jit_generator.hpp"
#include "../jit_uni_eltwise.hpp"
#include "c_types_map.hpp"
#include "utils.hpp"

#include "mkldnn_thread.hpp"


namespace mkldnn {
namespace impl {
namespace cpu {

struct jit_uni_rnn_postgemm_kernel : public jit_generator {

    typedef void (*kernel_t)(void *gates_, const void *bias, void *states_t_l_,
                     void *c_states_t_l_, void *c_states_tm1_l_);

    jit_uni_rnn_postgemm_kernel(const rnn_utils::rnn_conf_t &rnn, const primitive_attr_t *attr): rnn_(rnn), attr_(attr){}

    virtual void init() = 0;

template <typename src_data_t, typename acc_data_t>
    rnn_elemwise_sig(execute) {
        rnn_utils::ws_gates_aoc<acc_data_t> ws_gates(rnn, ws_gates_);
        rnn_utils::bias_aoc_t bias(rnn, bias_);
        rnn_utils::ws_states_aoc<src_data_t> states_t_l(rnn, states_t_l_);
        rnn_utils::ws_states_aoc_t c_states_t_l(rnn, c_states_t_l_);
        rnn_utils::ws_states_aoc_t c_states_tm1_l(rnn, c_states_tm1_l_);

        // Todo: add parallelization on dic for the batch 1 case
        // Assumption: the kernel runs a loop on dic elements
        parallel_nd(rnn.mb, [&](int i) {
                auto b_ = &bias(0, 0);
                auto g_ = &ws_gates(i, 0, 0);
                auto s_tl_ = &states_t_l(i, 0);
                auto c_tl_ = &c_states_t_l(i, 0);
                auto c_tm1l_ = &c_states_tm1_l(i, 0);
                kernel_(g_, b_, s_tl_, c_tm1l_, c_tl_);
            });
    }

protected:
    kernel_t kernel_;
    const rnn_utils::rnn_conf_t &rnn_;
    const primitive_attr_t *attr_;
};

template <cpu_isa_t isa, impl::data_type_t src_data_t>
struct jit_uni_lstm_postgemm_kernel_fwd: public jit_uni_rnn_postgemm_kernel
{
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_lstm_postgemm_kernel_fwd)

    typedef typename utils::conditional<src_data_t == data_type::u8, int32_t,
            float>::type acc_data_t;
    typedef typename utils::conditional<isa == avx512_core,
            jit_uni_eltwise_injector_f32<avx512_common>,
            jit_uni_eltwise_injector_f32<isa>>::type injector_t;

    jit_uni_lstm_postgemm_kernel_fwd(const rnn_utils::rnn_conf_t &rnn, const primitive_attr_t *attr)
    : jit_uni_rnn_postgemm_kernel(rnn, attr){}

    void init() override {
        // we use rax for both constant tables as they use the same table
        sigmoid_injector_ = new injector_t(this,
                alg_kind::eltwise_logistic, 0.0f, 0.0f, true, rax);
        tanh_injector_ = new injector_t(this,
                alg_kind::eltwise_tanh, 0.0f, 0.0f, true, rax);
        generate();
        kernel_ = (kernel_t) this->getCode();
    }

protected:
    injector_t *sigmoid_injector_;
    injector_t *tanh_injector_;

    // register size in bytes
    using Vmm = typename jit_uni_eltwise_injector_f32<isa>::Vmm;
    size_t vlen = cpu_isa_traits<isa>::vlen;
    size_t vlen_dst = (src_data_t == data_type::u8) ? vlen/4 : vlen;
    size_t cstate_dt_size = sizeof(float);
    size_t hstate_dt_size = (src_data_t == data_type::u8) ? sizeof(uint8_t) : sizeof(float);
    size_t gate_dt_size = (src_data_t == data_type::u8) ? sizeof(uint32_t) : sizeof(float);
    size_t qscale_dt_size = sizeof(float);
    size_t bias_dt_size = sizeof(float);

    void generate() {
        using namespace Xbyak;

        int mask = attr_->rnn_weights_qparams_.mask_;
        float *weights_scales = attr_->rnn_weights_qparams_.scales_;
        float data_scale = attr_->rnn_data_qparams_.scale_;
        float data_shift = attr_->rnn_data_qparams_.shift_;

        // Labels declaration
        Label vector_loop_start_label, vector_loop_end_label;
        Label rem_loop_start_label, rem_loop_end_label;
        Label table_label;

        // Register map
        Reg64 loop_cnt(r11);  // loop counter
        Reg64 table_reg(rbx); // table is used for data scale and shifts
        Reg64 weights_scales_reg(r13);
        // We skip vmm0 as it can be used by the injector for masks on sse4.2
        Vmm G0(1), G1(2), G2(3), G3(4), tmp1_vmm(5), tmp2_vmm(6), zero_vmm(7);

        // constant table map
        Address dscale_off_addr = ptr[table_reg];
        Address dshift_off_addr = ptr[table_reg + vlen];
        Address ymm_perm_mask_addr = ptr[table_reg + 2*vlen];
        Address zmm_perm_mask_addr = ptr[table_reg + 2*vlen + cpu_isa_traits<avx>::vlen];

        // quantize from float to u8
        auto q_d = [&](Vmm f, Vmm tmp_vmm) {
            uni_vpxor(tmp_vmm, tmp_vmm, tmp_vmm);
            uni_vmulps(f, f, dscale_off_addr); // apply scale
            uni_vaddps(f, f, dshift_off_addr); // apply shift
            uni_vcvtps2dq(f, f); // convert to int32
            uni_vpackssdw(f, f, tmp_vmm); // convert from s32 to s16
            uni_vpackuswb(f, f, tmp_vmm); // convert from s16 to u8 with saturation
            // Note that the results are interleaved by 128 bit chunks, so we need to merge them together
            switch (vlen) {
            case 64:  { //avx512
                Zmm fz(f.getIdx()), tmpz(tmp_vmm.getIdx());
                uni_vmovups(tmpz, zmm_perm_mask_addr);
                vpermd(fz, tmpz, fz);
                break; }
            case 32: { //avx
                Ymm fy(f.getIdx()), tmpy(tmp_vmm.getIdx());
                uni_vmovups(tmpy, ymm_perm_mask_addr);
                vpermd(fy, tmpy, fy);
                break; }
            case 16: // sse: nothing to do
                break;
            default: assert(!"Unsupported case");
            };
        };

        auto fast_recip =[&](Vmm s, Vmm tmp, bool packed) {
            if (packed)
                uni_vrcpps(tmp, s);
            else
                uni_vrcpss(tmp, s); // prevent divide by zero
            // we add one Newton iteration
            uni_vmulps(s, s, tmp);
            uni_vmulps(s, s, tmp); // s <- s * tmp^2
            uni_vaddps(tmp, tmp, tmp);
            uni_vsubps(tmp, tmp, s);
            uni_vmovups(s, tmp); // s <- 2 * tmp - s * tmp^2
        };

        // dequantize from s32 to float
        auto deq_w = [&](Vmm s, Vmm tmp1, Vmm tmp2, int gate, bool packed) {
            // TODO: if mask is 0 precompute mul and inverse
            if (mask == 0)
                uni_vbroadcastss(tmp1, ptr[weights_scales_reg]);
            else
                uni_vmovups(tmp1, ptr[weights_scales_reg + gate * rnn_.dic * qscale_dt_size]);
            uni_vcvtdq2ps(s, s);
            uni_vmulps(tmp1, tmp1, dscale_off_addr);
            fast_recip(tmp1, tmp2, packed);
            uni_vmulps(s, s, tmp1);
        };

        // We start code generations here
        preamble();

        // extract addresses passed as parameter
#ifdef _WIN32
        auto addr_ws_gates_reg = abi_param1;
        auto addr_bias_reg = abi_param2;
        auto addr_states_t_l_reg = abi_param3;
        auto addr_c_states_tm1_l_reg = abi_param4;
        auto addr_c_states_t_l_reg = r10;
        // Here we cannot use rbp to have initial stack pointer so we
        // use rsp and offset it with the size of pushed registers in
        // preamble
        mov(addr_c_states_t_l_reg, ptr[rsp + get_size_of_abi_save_regs() + 40]);
#else
        auto addr_ws_gates_reg = abi_param1;
        auto addr_bias_reg = abi_param2;
        auto addr_states_t_l_reg = abi_param3;
        auto addr_c_states_tm1_l_reg = abi_param4;
        auto addr_c_states_t_l_reg = abi_param5;
#endif

        // initialize registers with addresses and constants
        mov(table_reg, table_label);
        mov(weights_scales_reg, size_t(weights_scales));
        // both sigmoid and tanh use the same table so load address just once in rax
        sigmoid_injector_->load_table_addr();

        mov(loop_cnt, rnn_.dic * gate_dt_size);
        cmp(loop_cnt, vlen);
        jl(vector_loop_end_label, Xbyak::CodeGenerator::T_NEAR);

        L(vector_loop_start_label);
        {
            // load G0 G1 G2 G3
            uni_vmovups(G0, ptr[addr_ws_gates_reg + 0 * rnn_.dic * gate_dt_size]);
            uni_vmovups(G1, ptr[addr_ws_gates_reg + 1 * rnn_.dic * gate_dt_size]);
            uni_vmovups(G2, ptr[addr_ws_gates_reg + 2 * rnn_.dic * gate_dt_size]);
            uni_vmovups(G3, ptr[addr_ws_gates_reg + 3 * rnn_.dic * gate_dt_size]);

            // dequantize the gates from s32 to f32 if needed
            if (src_data_t == data_type::u8){
                deq_w(G0, tmp1_vmm, tmp2_vmm, 0, true);
                deq_w(G1, tmp1_vmm, tmp2_vmm, 1, true);
                deq_w(G2, tmp1_vmm, tmp2_vmm, 2, true);
                deq_w(G3, tmp1_vmm, tmp2_vmm, 3, true);
            }

            // add biases
            uni_vaddps(G0, G0, ptr[addr_bias_reg + 0 * rnn_.dic * bias_dt_size]);
            uni_vaddps(G1, G1, ptr[addr_bias_reg + 1 * rnn_.dic * bias_dt_size]);
            uni_vaddps(G2, G2, ptr[addr_bias_reg + 2 * rnn_.dic * bias_dt_size]);
            uni_vaddps(G3, G3, ptr[addr_bias_reg + 3 * rnn_.dic * bias_dt_size]);

            // inject eltwise code
            sigmoid_injector_->compute_vector(G0.getIdx());
            sigmoid_injector_->compute_vector(G1.getIdx());
            tanh_injector_->compute_vector(G2.getIdx());
            sigmoid_injector_->compute_vector(G3.getIdx());

            // compute c_states_t_l = G1 * c_tm1_l + G0 * G2
            uni_vmovups(tmp1_vmm, ptr[addr_c_states_tm1_l_reg]);
            uni_vmulps(tmp1_vmm, tmp1_vmm, G1);
            uni_vfmadd231ps(tmp1_vmm, G0, G2);
            uni_vmovups(ptr[addr_c_states_t_l_reg], tmp1_vmm);

            // states_t_l = G3 * tanh(c_states_t_l)
            tanh_injector_->compute_vector(tmp1_vmm.getIdx());
            uni_vmulps(tmp1_vmm, tmp1_vmm, G3);

            // if int8, we quantize the resulting state
            if (src_data_t == data_type::u8)
                q_d(tmp1_vmm, tmp2_vmm);

            // write back the result
            if(vlen_dst == vlen)
                uni_vmovups(ptr[addr_states_t_l_reg], tmp1_vmm);
            else
                // we write only 1/4 of the register
                switch(vlen_dst){
                case 16: uni_vmovups(ptr[addr_states_t_l_reg], Xmm(tmp1_vmm.getIdx())); break;
                case 8: uni_vmovsd(ptr[addr_states_t_l_reg], Xmm(tmp1_vmm.getIdx())); break;
                case 4: uni_vmovss(ptr[addr_states_t_l_reg], Xmm(tmp1_vmm.getIdx())); break;
                default:
                    assert(!"Unsuported vector length for quantization");
                }

            // increment address pointers
            add(addr_ws_gates_reg, vlen);
            add(addr_bias_reg, vlen);
            add(addr_states_t_l_reg, vlen_dst);
            add(addr_c_states_tm1_l_reg, vlen);
            add(addr_c_states_t_l_reg, vlen);
            if (mask != 0)
                add(weights_scales_reg, vlen);

            // increment loop counter
            sub(loop_cnt, vlen);
            cmp(loop_cnt, vlen);
            jge(vector_loop_start_label);
        }
        L(vector_loop_end_label);

        cmp(loop_cnt, 0);
        je(rem_loop_end_label, Xbyak::CodeGenerator::T_NEAR);
        // Same code as above, we just use movuss for accessing inputs
        // TODO: smarter handling of tails with Zmm -> Ymm -> Xmm -> scalar
        L(rem_loop_start_label);
        {
            // remaping registers to Xmms
            Xmm G0s(G0.getIdx()), G1s(G1.getIdx()), G2s(G2.getIdx()), G3s(G3.getIdx());
            Xmm tmp1s_vmm(tmp1_vmm.getIdx());

            // load G0 G1 G2 G3
            uni_vmovss(G0s, ptr[addr_ws_gates_reg + 0 * rnn_.dic * gate_dt_size]);
            uni_vmovss(G1s, ptr[addr_ws_gates_reg + 1 * rnn_.dic * gate_dt_size]);
            uni_vmovss(G2s, ptr[addr_ws_gates_reg + 2 * rnn_.dic * gate_dt_size]);
            uni_vmovss(G3s, ptr[addr_ws_gates_reg + 3 * rnn_.dic * gate_dt_size]);

            // dequantize the gates from s32 to f32 if needed
            if (src_data_t == data_type::u8){
                deq_w(G0, tmp1_vmm, tmp2_vmm, 0, false);
                deq_w(G1, tmp1_vmm, tmp2_vmm, 1, false);
                deq_w(G2, tmp1_vmm, tmp2_vmm, 2, false);
                deq_w(G3, tmp1_vmm, tmp2_vmm, 3, false);
            }

            // add biases
            uni_vmovss(tmp1s_vmm, ptr[addr_bias_reg + 0 * rnn_.dic * bias_dt_size]);
            uni_vaddps(G0s, G0s, tmp1s_vmm);
            uni_vmovss(tmp1s_vmm, ptr[addr_bias_reg + 1 * rnn_.dic * bias_dt_size]);
            uni_vaddps(G1s, G1s, tmp1s_vmm);
            uni_vmovss(tmp1s_vmm, ptr[addr_bias_reg + 2 * rnn_.dic * bias_dt_size]);
            uni_vaddps(G2s, G2s, tmp1s_vmm);
            uni_vmovss(tmp1s_vmm, ptr[addr_bias_reg + 3 * rnn_.dic * bias_dt_size]);
            uni_vaddps(G3s, G3s, tmp1s_vmm);

            // inject eltwise code
            sigmoid_injector_->compute_vector(G0s.getIdx());
            sigmoid_injector_->compute_vector(G1s.getIdx());
            tanh_injector_->compute_vector(G2s.getIdx());
            sigmoid_injector_->compute_vector(G3s.getIdx());

            // compute c_states_t_l = G1 * c_tm1_l + G0s * G2
            uni_vmovups(tmp1s_vmm, ptr[addr_c_states_tm1_l_reg]);
            uni_vmulps(tmp1s_vmm, tmp1s_vmm, G1s);
            uni_vfmadd231ps(tmp1s_vmm, G0s, G2s);
            uni_vmovss(ptr[addr_c_states_t_l_reg], tmp1s_vmm);

            // states_t_l = G3 * tanh(c_states_t_l)
            tanh_injector_->compute_vector(tmp1s_vmm.getIdx());
            uni_vmulps(tmp1s_vmm, tmp1s_vmm, G3s);

            // if int8, we quantize the resulting state
            if (src_data_t == data_type::u8)
                q_d(tmp1_vmm, tmp2_vmm);

            // write back the result
            if(vlen_dst == vlen)
                uni_vmovups(ptr[addr_states_t_l_reg], tmp1s_vmm);
            else
                // we write only 1/4 of the register
                switch(vlen_dst){
                case 16: uni_vmovups(ptr[addr_states_t_l_reg], Xmm(tmp1s_vmm.getIdx())); break;
                case 8: uni_vmovsd(ptr[addr_states_t_l_reg], Xmm(tmp1s_vmm.getIdx())); break;
                case 4: uni_vmovss(ptr[addr_states_t_l_reg], Xmm(tmp1s_vmm.getIdx())); break;
                default:
                    assert(!"Unsuported vector length for quantization");
                }

            // increment address pointers
            add(addr_ws_gates_reg, gate_dt_size);
            add(addr_bias_reg, bias_dt_size);
            add(addr_states_t_l_reg, hstate_dt_size);
            add(addr_c_states_tm1_l_reg, cstate_dt_size);
            add(addr_c_states_t_l_reg, cstate_dt_size);
            if (mask != 0)
                add(weights_scales_reg, qscale_dt_size);

            // increment loop counter
            sub(loop_cnt, gate_dt_size);
            cmp(loop_cnt, 0);
            jg(rem_loop_start_label);

        }
        L(rem_loop_end_label);

        postamble();

        // Again, only one table is needed and shared between sigmoid and tanh
        sigmoid_injector_->prepare_table(false);
        tanh_injector_->prepare_table(true);

        L(table_label);
        {
            for (size_t i = 0; i < vlen / sizeof(float); i++) dd(float2int(data_scale));
            for (size_t i = 0; i < vlen / sizeof(float); i++) dd(float2int(data_shift));
            // perm mask for ymm
            dd(0); dd(4); dd(2); dd(3); dd(1); dd(5); dd(6); dd(7);
            // perm mask for zmm
            dd(0); dd(4); dd(8); dd(12); dd(1); dd(5); dd(6); dd(7);
            dd(2); dd(9); dd(10); dd(11); dd(3); dd(12); dd(13); dd(14);
        }
    }

};

template struct jit_uni_lstm_postgemm_kernel_fwd<sse42, data_type::f32>;
template struct jit_uni_lstm_postgemm_kernel_fwd<avx2, data_type::f32>;
template struct jit_uni_lstm_postgemm_kernel_fwd<avx512_core, data_type::f32>;

template struct jit_uni_lstm_postgemm_kernel_fwd<sse42, data_type::u8>;
template struct jit_uni_lstm_postgemm_kernel_fwd<avx2, data_type::u8>;
template struct jit_uni_lstm_postgemm_kernel_fwd<avx512_core, data_type::u8>;
}
}
}
