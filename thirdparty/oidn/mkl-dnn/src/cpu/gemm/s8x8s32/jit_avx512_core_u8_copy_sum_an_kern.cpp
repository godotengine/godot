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

#include "jit_generator.hpp"
#include "common.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

jit_avx512_core_u8_copy_sum_an_kern::jit_avx512_core_u8_copy_sum_an_kern(): jit_generator(nullptr, GEMM_CODE_SIZE)
{

#ifndef _WIN32
#define M	rdi
#define N	rsi
#define A	rdx
#define LDA	rcx
#define ALPHA	r8
#define B	r9

#define I	rax
#define A1	r10
#define A2	r8
#define LDA3	r11

#define ARG_BIAS	24+stacksize+rsp

#else

#define M	rcx
#define N	rdx
#define A	r8
#define LDA	r9
#define ALPHA	rax
#define B	rdi

#define I	rax
#define A1	rsi
#define A2	r10
#define LDA3	r11

#define ARG_ALPHA	40+stacksize+rsp
#define ARG_B		48+stacksize+rsp
#define ARG_BIAS	72+stacksize+rsp

#endif

inLocalLabel();
{

Xbyak::Label l1024;
Xbyak::Label l1090;
Xbyak::Label l10d4;
Xbyak::Label l10fc;
Xbyak::Label l111a;
Xbyak::Label l1124;
Xbyak::Label l113c;
Xbyak::Label l11d4;
Xbyak::Label l1234;
Xbyak::Label l1278;
Xbyak::Label l129c;
Xbyak::Label l12bc;
Xbyak::Label l20;
Xbyak::Label l2a0;
Xbyak::Label l3c0;
Xbyak::Label l438;
Xbyak::Label l480;
Xbyak::Label l48c;
Xbyak::Label l4c8;
Xbyak::Label l5c;
Xbyak::Label l6a8;
Xbyak::Label l7b4;
Xbyak::Label l850;
Xbyak::Label l89c;
Xbyak::Label l8a8;
Xbyak::Label l8d0;
Xbyak::Label l9d0;
Xbyak::Label la64;
Xbyak::Label lab8;
Xbyak::Label lae8;
Xbyak::Label laf4;
Xbyak::Label lb14;
Xbyak::Label lc30;
Xbyak::Label lcc8;
Xbyak::Label ld1c;
Xbyak::Label ld54;
Xbyak::Label ld78;
Xbyak::Label ld84;
Xbyak::Label ld9c;
Xbyak::Label le58;
Xbyak::Label lebc;
Xbyak::Label lef8;
Xbyak::Label lf1c;
Xbyak::Label lf3c;
Xbyak::Label lf48;
Xbyak::Label lf60;

	preamble();
	auto stacksize = get_size_of_abi_save_regs();
#ifdef _WIN32
	mov(ALPHA, ptr[ARG_ALPHA]);
	mov(B, ptr[ARG_B]);
#endif

	mov(M, qword[M]);
	mov(N, qword[N]);
	mov(LDA, qword[LDA]);
	lea(LDA3, ptr[LDA+LDA*2]);
	sub(A, -128);
	sub(B, -128);
	cmp(N, 0x30);
	jl(l480, T_NEAR);
	align(4);

L(l20);
	mov(A1, A);
	add(A, 0x30);
	vxorps(ymm8, ymm8, ymm8);
	vxorps(ymm9, ymm9, ymm9);
	vxorps(ymm10, ymm10, ymm10);
	vxorps(ymm11, ymm11, ymm11);
	vxorps(ymm12, ymm12, ymm12);
	vxorps(ymm13, ymm13, ymm13);
	vxorps(ymm14, ymm14, ymm14);
	vxorps(ymm15, ymm15, ymm15);
	mov(I, M);
	sar(I, 0x2);
	jle(l2a0, T_NEAR);
	align(4);

L(l5c);
	vmovdqu(xmm0, xword[A1-0x80]);
	vmovdqu(xmm1, xword[A1+LDA*1-0x80]);
	vmovdqu(xmm2, xword[A1+LDA*2-0x80]);
	vmovdqu(xmm3, xword[A1+LDA3*1-0x80]);
	vpunpcklbw(xmm4, xmm0, xmm1);
	vpunpckhbw(xmm5, xmm0, xmm1);
	vpunpcklbw(xmm6, xmm2, xmm3);
	vpunpckhbw(xmm7, xmm2, xmm3);
	vpunpcklwd(xmm0, xmm4, xmm6);
	vpunpckhwd(xmm1, xmm4, xmm6);
	vpunpcklwd(xmm2, xmm5, xmm7);
	vpunpckhwd(xmm3, xmm5, xmm7);
	vpmovsxbw(ymm5, xmm0);
	vmovhlps(xmm6, xmm0, xmm0);
	vpmovsxbw(ymm6, xmm6);
	vphaddw(ymm5, ymm5, ymm6);
	vpmovsxbw(ymm6, xmm1);
	vmovhlps(xmm7, xmm1, xmm1);
	vpmovsxbw(ymm7, xmm7);
	vphaddw(ymm6, ymm6, ymm7);
	vphaddw(ymm5, ymm5, ymm6);
	vpmovsxwd(ymm5, xmm5);
	vpaddd(ymm8, ymm8, ymm5);
	vmovdqu(xword[B-0x80], xmm0);
	vmovdqu(xword[B-0x70], xmm1);
	vpmovsxbw(ymm5, xmm2);
	vmovhlps(xmm6, xmm2, xmm2);
	vpmovsxbw(ymm6, xmm6);
	vphaddw(ymm5, ymm5, ymm6);
	vpmovsxbw(ymm6, xmm3);
	vmovhlps(xmm7, xmm3, xmm3);
	vpmovsxbw(ymm7, xmm7);
	vphaddw(ymm6, ymm6, ymm7);
	vphaddw(ymm5, ymm5, ymm6);
	vpmovsxwd(ymm5, xmm5);
	vpaddd(ymm9, ymm9, ymm5);
	vmovdqu(xword[B-0x60], xmm2);
	vmovdqu(xword[B-0x50], xmm3);
	vmovdqu(xmm0, xword[A1-0x70]);
	vmovdqu(xmm1, xword[A1+LDA*1-0x70]);
	vmovdqu(xmm2, xword[A1+LDA*2-0x70]);
	vmovdqu(xmm3, xword[A1+LDA3*1-0x70]);
	vpunpcklbw(xmm4, xmm0, xmm1);
	vpunpckhbw(xmm5, xmm0, xmm1);
	vpunpcklbw(xmm6, xmm2, xmm3);
	vpunpckhbw(xmm7, xmm2, xmm3);
	vpunpcklwd(xmm0, xmm4, xmm6);
	vpunpckhwd(xmm1, xmm4, xmm6);
	vpunpcklwd(xmm2, xmm5, xmm7);
	vpunpckhwd(xmm3, xmm5, xmm7);
	vpmovsxbw(ymm5, xmm0);
	vmovhlps(xmm6, xmm0, xmm0);
	vpmovsxbw(ymm6, xmm6);
	vphaddw(ymm5, ymm5, ymm6);
	vpmovsxbw(ymm6, xmm1);
	vmovhlps(xmm7, xmm1, xmm1);
	vpmovsxbw(ymm7, xmm7);
	vphaddw(ymm6, ymm6, ymm7);
	vphaddw(ymm5, ymm5, ymm6);
	vpmovsxwd(ymm5, xmm5);
	vpaddd(ymm10, ymm10, ymm5);
	vmovdqu(xword[B-0x40], xmm0);
	vmovdqu(xword[B-0x30], xmm1);
	vpmovsxbw(ymm5, xmm2);
	vmovhlps(xmm6, xmm2, xmm2);
	vpmovsxbw(ymm6, xmm6);
	vphaddw(ymm5, ymm5, ymm6);
	vpmovsxbw(ymm6, xmm3);
	vmovhlps(xmm7, xmm3, xmm3);
	vpmovsxbw(ymm7, xmm7);
	vphaddw(ymm6, ymm6, ymm7);
	vphaddw(ymm5, ymm5, ymm6);
	vpmovsxwd(ymm5, xmm5);
	vpaddd(ymm11, ymm11, ymm5);
	vmovdqu(xword[B-0x20], xmm2);
	vmovdqu(xword[B-0x10], xmm3);
	vmovdqu(xmm0, xword[A1-0x60]);
	vmovdqu(xmm1, xword[A1+LDA*1-0x60]);
	vmovdqu(xmm2, xword[A1+LDA*2-0x60]);
	vmovdqu(xmm3, xword[A1+LDA3*1-0x60]);
	lea(A1, ptr[A1+LDA*4]);
	vpunpcklbw(xmm4, xmm0, xmm1);
	vpunpckhbw(xmm5, xmm0, xmm1);
	vpunpcklbw(xmm6, xmm2, xmm3);
	vpunpckhbw(xmm7, xmm2, xmm3);
	vpunpcklwd(xmm0, xmm4, xmm6);
	vpunpckhwd(xmm1, xmm4, xmm6);
	vpunpcklwd(xmm2, xmm5, xmm7);
	vpunpckhwd(xmm3, xmm5, xmm7);
	vpmovsxbw(ymm5, xmm0);
	vmovhlps(xmm6, xmm0, xmm0);
	vpmovsxbw(ymm6, xmm6);
	vphaddw(ymm5, ymm5, ymm6);
	vpmovsxbw(ymm6, xmm1);
	vmovhlps(xmm7, xmm1, xmm1);
	vpmovsxbw(ymm7, xmm7);
	vphaddw(ymm6, ymm6, ymm7);
	vphaddw(ymm5, ymm5, ymm6);
	vpmovsxwd(ymm5, xmm5);
	vpaddd(ymm12, ymm12, ymm5);
	vmovdqu(xword[B], xmm0);
	vmovdqu(xword[B+0x10], xmm1);
	vpmovsxbw(ymm5, xmm2);
	vmovhlps(xmm6, xmm2, xmm2);
	vpmovsxbw(ymm6, xmm6);
	vphaddw(ymm5, ymm5, ymm6);
	vpmovsxbw(ymm6, xmm3);
	vmovhlps(xmm7, xmm3, xmm3);
	vpmovsxbw(ymm7, xmm7);
	vphaddw(ymm6, ymm6, ymm7);
	vphaddw(ymm5, ymm5, ymm6);
	vpmovsxwd(ymm5, xmm5);
	vpaddd(ymm13, ymm13, ymm5);
	vmovdqu(xword[B+0x20], xmm2);
	vmovdqu(xword[B+0x30], xmm3);
	sub(B, -192);
	dec(I);
	jg(l5c, T_NEAR);
	align(4);

L(l2a0);
	test(M, 0x2);
	jle(l3c0, T_NEAR);
	vmovdqu(xmm0, xword[A1-0x80]);
	vmovdqu(xmm1, xword[A1-0x70]);
	vmovdqu(xmm2, xword[A1-0x60]);
	add(A1, LDA);
	vmovdqu(xmm6, xword[A1-0x80]);
	vmovdqu(xmm4, xword[A1-0x70]);
	vmovdqu(xmm5, xword[A1-0x60]);
	add(A1, LDA);
	vpunpcklbw(xmm3, xmm0, xmm6);
	vpunpckhbw(xmm0, xmm0, xmm6);
	vpmovsxbw(ymm7, xmm3);
	vmovhlps(xmm6, xmm3, xmm3);
	vpmovsxbw(ymm6, xmm6);
	vphaddw(ymm7, ymm7, ymm6);
	vpmovsxwd(ymm7, xmm7);
	vpaddd(ymm8, ymm8, ymm7);
	vmovdqu(xword[B-0x80], xmm3);
	vpmovsxbw(ymm7, xmm0);
	vmovhlps(xmm6, xmm0, xmm0);
	vpmovsxbw(ymm6, xmm6);
	vphaddw(ymm7, ymm7, ymm6);
	vpmovsxwd(ymm7, xmm7);
	vpaddd(ymm9, ymm9, ymm7);
	vmovdqu(xword[B-0x70], xmm0);
	vpunpcklbw(xmm3, xmm1, xmm4);
	vpunpckhbw(xmm0, xmm1, xmm4);
	vpmovsxbw(ymm7, xmm3);
	vmovhlps(xmm6, xmm3, xmm3);
	vpmovsxbw(ymm6, xmm6);
	vphaddw(ymm7, ymm7, ymm6);
	vpmovsxwd(ymm7, xmm7);
	vpaddd(ymm10, ymm10, ymm7);
	vmovdqu(xword[B-0x60], xmm3);
	vpmovsxbw(ymm7, xmm0);
	vmovhlps(xmm6, xmm0, xmm0);
	vpmovsxbw(ymm6, xmm6);
	vphaddw(ymm7, ymm7, ymm6);
	vpmovsxwd(ymm7, xmm7);
	vpaddd(ymm11, ymm11, ymm7);
	vmovdqu(xword[B-0x50], xmm0);
	vpunpcklbw(xmm3, xmm2, xmm5);
	vpunpckhbw(xmm0, xmm2, xmm5);
	vpmovsxbw(ymm7, xmm3);
	vmovhlps(xmm6, xmm3, xmm3);
	vpmovsxbw(ymm6, xmm6);
	vphaddw(ymm7, ymm7, ymm6);
	vpmovsxwd(ymm7, xmm7);
	vpaddd(ymm12, ymm12, ymm7);
	vmovdqu(xword[B-0x40], xmm3);
	vpmovsxbw(ymm7, xmm0);
	vmovhlps(xmm6, xmm0, xmm0);
	vpmovsxbw(ymm6, xmm6);
	vphaddw(ymm7, ymm7, ymm6);
	vpmovsxwd(ymm7, xmm7);
	vpaddd(ymm13, ymm13, ymm7);
	vmovdqu(xword[B-0x30], xmm0);
	sub(B, -96);
	align(4);

L(l3c0);
	test(M, 0x1);
	jle(l438, T_NEAR);
	vmovdqu(xmm0, xword[A1-0x80]);
	vmovdqu(xmm1, xword[A1-0x70]);
	vmovdqu(xmm2, xword[A1-0x60]);
	add(A1, LDA);
	vpmovsxbd(ymm7, xmm0);
	vpaddd(ymm8, ymm8, ymm7);
	vmovhlps(xmm7, xmm0, xmm0);
	vpmovsxbd(ymm7, xmm7);
	vpaddd(ymm9, ymm9, ymm7);
	vmovdqu(xword[B-0x80], xmm0);
	vpmovsxbd(ymm7, xmm1);
	vpaddd(ymm10, ymm10, ymm7);
	vmovhlps(xmm7, xmm1, xmm1);
	vpmovsxbd(ymm7, xmm7);
	vpaddd(ymm11, ymm11, ymm7);
	vmovdqu(xword[B-0x70], xmm1);
	vpmovsxbd(ymm7, xmm2);
	vpaddd(ymm12, ymm12, ymm7);
	vmovhlps(xmm7, xmm2, xmm2);
	vpmovsxbd(ymm7, xmm7);
	vpaddd(ymm13, ymm13, ymm7);
	vmovdqu(xword[B-0x60], xmm2);
	sub(B, -48);
	align(4);

L(l438);
	mov(A1, qword[ARG_BIAS]);
	vmovdqu(yword[A1], ymm8);
	vmovdqu(yword[A1+0x20], ymm9);
	vmovdqu(yword[A1+0x40], ymm10);
	vmovdqu(yword[A1+0x60], ymm11);
	vmovdqu(yword[A1+0x80], ymm12);
	vmovdqu(yword[A1+0xa0], ymm13);
	add(qword[ARG_BIAS], 0xc0);
	sub(N, 0x30);
	cmp(N, 0x30);
	jge(l20, T_NEAR);
	vzeroupper();
	align(4);

L(l480);
	cmp(N, 0x20);
	jl(l89c, T_NEAR);
	align(4);

L(l48c);
	mov(A1, A);
	add(A, 0x20);
	pxor(xmm8, xmm8);
	pxor(xmm9, xmm9);
	pxor(xmm10, xmm10);
	pxor(xmm11, xmm11);
	pxor(xmm12, xmm12);
	pxor(xmm13, xmm13);
	pxor(xmm14, xmm14);
	pxor(xmm15, xmm15);
	mov(I, M);
	sar(I, 0x2);
	jle(l6a8, T_NEAR);
	align(4);

L(l4c8);
	movdqu(xmm0, xword[A1-0x80]);
	movdqu(xmm1, xword[A1+LDA*1-0x80]);
	movdqu(xmm2, xword[A1+LDA*2-0x80]);
	movdqu(xmm3, xword[A1+LDA3*1-0x80]);
	movdqa(xmm4, xmm0);
	punpcklbw(xmm0, xmm1);
	punpckhbw(xmm4, xmm1);
	movdqa(xmm5, xmm2);
	punpcklbw(xmm2, xmm3);
	punpckhbw(xmm5, xmm3);
	movdqa(xmm1, xmm0);
	punpcklwd(xmm0, xmm2);
	punpckhwd(xmm1, xmm2);
	movdqa(xmm2, xmm4);
	punpcklwd(xmm4, xmm5);
	punpckhwd(xmm2, xmm5);
	pmovsxbw(xmm5, xmm0);
	movhlps(xmm6, xmm0);
	pmovsxbw(xmm6, xmm6);
	phaddw(xmm5, xmm6);
	phaddw(xmm5, xmm5);
	pmovsxwd(xmm5, xmm5);
	paddd(xmm8, xmm5);
	movdqu(xword[B-0x80], xmm0);
	pmovsxbw(xmm5, xmm1);
	movhlps(xmm6, xmm1);
	pmovsxbw(xmm6, xmm6);
	phaddw(xmm5, xmm6);
	phaddw(xmm5, xmm5);
	pmovsxwd(xmm5, xmm5);
	paddd(xmm9, xmm5);
	movdqu(xword[B-0x70], xmm1);
	pmovsxbw(xmm5, xmm4);
	movhlps(xmm6, xmm4);
	pmovsxbw(xmm6, xmm6);
	phaddw(xmm5, xmm6);
	phaddw(xmm5, xmm5);
	pmovsxwd(xmm5, xmm5);
	paddd(xmm10, xmm5);
	movdqu(xword[B-0x60], xmm4);
	pmovsxbw(xmm5, xmm2);
	movhlps(xmm6, xmm2);
	pmovsxbw(xmm6, xmm6);
	phaddw(xmm5, xmm6);
	phaddw(xmm5, xmm5);
	pmovsxwd(xmm5, xmm5);
	paddd(xmm11, xmm5);
	movdqu(xword[B-0x50], xmm2);
	movdqu(xmm0, xword[A1-0x70]);
	movdqu(xmm1, xword[A1+LDA*1-0x70]);
	movdqu(xmm2, xword[A1+LDA*2-0x70]);
	movdqu(xmm3, xword[A1+LDA3*1-0x70]);
	lea(A1, ptr[A1+LDA*4]);
	movdqa(xmm4, xmm0);
	punpcklbw(xmm0, xmm1);
	punpckhbw(xmm4, xmm1);
	movdqa(xmm5, xmm2);
	punpcklbw(xmm2, xmm3);
	punpckhbw(xmm5, xmm3);
	movdqa(xmm1, xmm0);
	punpcklwd(xmm0, xmm2);
	punpckhwd(xmm1, xmm2);
	movdqa(xmm2, xmm4);
	punpcklwd(xmm4, xmm5);
	punpckhwd(xmm2, xmm5);
	pmovsxbw(xmm5, xmm0);
	movhlps(xmm6, xmm0);
	pmovsxbw(xmm6, xmm6);
	phaddw(xmm5, xmm6);
	phaddw(xmm5, xmm5);
	pmovsxwd(xmm5, xmm5);
	paddd(xmm12, xmm5);
	movdqu(xword[B-0x40], xmm0);
	pmovsxbw(xmm5, xmm1);
	movhlps(xmm6, xmm1);
	pmovsxbw(xmm6, xmm6);
	phaddw(xmm5, xmm6);
	phaddw(xmm5, xmm5);
	pmovsxwd(xmm5, xmm5);
	paddd(xmm13, xmm5);
	movdqu(xword[B-0x30], xmm1);
	pmovsxbw(xmm5, xmm4);
	movhlps(xmm6, xmm4);
	pmovsxbw(xmm6, xmm6);
	phaddw(xmm5, xmm6);
	phaddw(xmm5, xmm5);
	pmovsxwd(xmm5, xmm5);
	paddd(xmm14, xmm5);
	movdqu(xword[B-0x20], xmm4);
	pmovsxbw(xmm5, xmm2);
	movhlps(xmm6, xmm2);
	pmovsxbw(xmm6, xmm6);
	phaddw(xmm5, xmm6);
	phaddw(xmm5, xmm5);
	pmovsxwd(xmm5, xmm5);
	paddd(xmm15, xmm5);
	movdqu(xword[B-0x10], xmm2);
	sub(B, -128);
	dec(I);
	jg(l4c8, T_NEAR);
	align(4);

L(l6a8);
	test(M, 0x2);
	jle(l7b4, T_NEAR);
	movdqu(xmm0, xword[A1-0x80]);
	movdqu(xmm1, xword[A1-0x70]);
	add(A1, LDA);
	movdqu(xmm2, xword[A1-0x80]);
	movdqu(xmm3, xword[A1-0x70]);
	add(A1, LDA);
	movdqa(xmm4, xmm0);
	punpcklbw(xmm0, xmm2);
	punpckhbw(xmm4, xmm2);
	pmovsxbw(xmm5, xmm0);
	phaddw(xmm5, xmm5);
	pmovsxwd(xmm5, xmm5);
	paddd(xmm8, xmm5);
	movhlps(xmm6, xmm0);
	pmovsxbw(xmm6, xmm6);
	phaddw(xmm6, xmm6);
	pmovsxwd(xmm6, xmm6);
	paddd(xmm9, xmm6);
	movdqu(xword[B-0x80], xmm0);
	pmovsxbw(xmm5, xmm4);
	phaddw(xmm5, xmm5);
	pmovsxwd(xmm5, xmm5);
	paddd(xmm10, xmm5);
	movhlps(xmm6, xmm4);
	pmovsxbw(xmm6, xmm6);
	phaddw(xmm6, xmm6);
	pmovsxwd(xmm6, xmm6);
	paddd(xmm11, xmm6);
	movdqu(xword[B-0x70], xmm4);
	movdqa(xmm4, xmm1);
	punpcklbw(xmm1, xmm3);
	punpckhbw(xmm4, xmm3);
	pmovsxbw(xmm5, xmm1);
	phaddw(xmm5, xmm5);
	pmovsxwd(xmm5, xmm5);
	paddd(xmm12, xmm5);
	movhlps(xmm6, xmm1);
	pmovsxbw(xmm6, xmm6);
	phaddw(xmm6, xmm6);
	pmovsxwd(xmm6, xmm6);
	paddd(xmm13, xmm6);
	movdqu(xword[B-0x60], xmm1);
	pmovsxbw(xmm5, xmm4);
	phaddw(xmm5, xmm5);
	pmovsxwd(xmm5, xmm5);
	paddd(xmm14, xmm5);
	movhlps(xmm6, xmm4);
	pmovsxbw(xmm6, xmm6);
	phaddw(xmm6, xmm6);
	pmovsxwd(xmm6, xmm6);
	paddd(xmm15, xmm6);
	movdqu(xword[B-0x50], xmm4);
	sub(B, -64);
	align(4);

L(l7b4);
	test(M, 0x1);
	jle(l850, T_NEAR);
	movdqu(xmm0, xword[A1-0x80]);
	movdqu(xmm1, xword[A1-0x70]);
	add(A1, LDA);
	pmovsxbd(xmm5, xmm0);
	paddd(xmm8, xmm5);
	pshufd(xmm6, xmm0, 0x55);
	pmovsxbd(xmm6, xmm6);
	paddd(xmm9, xmm6);
	pshufd(xmm5, xmm0, 0xaa);
	pmovsxbd(xmm5, xmm5);
	paddd(xmm10, xmm5);
	pshufd(xmm6, xmm0, 0xff);
	pmovsxbd(xmm6, xmm6);
	paddd(xmm11, xmm6);
	movdqu(xword[B-0x80], xmm0);
	pmovsxbd(xmm5, xmm1);
	paddd(xmm12, xmm5);
	pshufd(xmm6, xmm1, 0x55);
	pmovsxbd(xmm6, xmm6);
	paddd(xmm13, xmm6);
	pshufd(xmm5, xmm1, 0xaa);
	pmovsxbd(xmm5, xmm5);
	paddd(xmm14, xmm5);
	pshufd(xmm6, xmm1, 0xff);
	pmovsxbd(xmm6, xmm6);
	paddd(xmm15, xmm6);
	movdqu(xword[B-0x70], xmm1);
	sub(B, -32);
	align(4);

L(l850);
	mov(A1, qword[ARG_BIAS]);
	movdqu(xword[A1], xmm8);
	movdqu(xword[A1+0x10], xmm9);
	movdqu(xword[A1+0x20], xmm10);
	movdqu(xword[A1+0x30], xmm11);
	movdqu(xword[A1+0x40], xmm12);
	movdqu(xword[A1+0x50], xmm13);
	movdqu(xword[A1+0x60], xmm14);
	movdqu(xword[A1+0x70], xmm15);
	add(qword[ARG_BIAS], 0x80);
	sub(N, 0x20);
	cmp(N, 0x20);
	jge(l48c, T_NEAR);
	align(4);

L(l89c);
	cmp(N, 0x10);
	jl(lae8, T_NEAR);
	align(4);

L(l8a8);
	mov(A1, A);
	add(A, 0x10);
	pxor(xmm8, xmm8);
	pxor(xmm9, xmm9);
	pxor(xmm10, xmm10);
	pxor(xmm11, xmm11);
	mov(I, M);
	sar(I, 0x2);
	jle(l9d0, T_NEAR);
	align(4);

L(l8d0);
	movdqu(xmm0, xword[A1-0x80]);
	add(A1, LDA);
	movdqu(xmm1, xword[A1-0x80]);
	add(A1, LDA);
	movdqu(xmm2, xword[A1-0x80]);
	add(A1, LDA);
	movdqu(xmm3, xword[A1-0x80]);
	add(A1, LDA);
	movdqa(xmm4, xmm0);
	punpcklbw(xmm0, xmm1);
	punpckhbw(xmm4, xmm1);
	movdqa(xmm1, xmm2);
	punpcklbw(xmm2, xmm3);
	punpckhbw(xmm1, xmm3);
	movdqa(xmm3, xmm0);
	punpcklwd(xmm0, xmm2);
	punpckhwd(xmm3, xmm2);
	movdqa(xmm2, xmm4);
	punpcklwd(xmm4, xmm1);
	punpckhwd(xmm2, xmm1);
	pmovsxbw(xmm5, xmm0);
	movhlps(xmm6, xmm0);
	pmovsxbw(xmm6, xmm6);
	phaddw(xmm5, xmm6);
	phaddw(xmm5, xmm5);
	pmovsxwd(xmm5, xmm5);
	paddd(xmm8, xmm5);
	pmovsxbw(xmm5, xmm3);
	movhlps(xmm6, xmm3);
	pmovsxbw(xmm6, xmm6);
	phaddw(xmm5, xmm6);
	phaddw(xmm5, xmm5);
	pmovsxwd(xmm5, xmm5);
	paddd(xmm9, xmm5);
	movdqu(xword[B-0x80], xmm0);
	movdqu(xword[B-0x70], xmm3);
	pmovsxbw(xmm5, xmm4);
	movhlps(xmm6, xmm4);
	pmovsxbw(xmm6, xmm6);
	phaddw(xmm5, xmm6);
	phaddw(xmm5, xmm5);
	pmovsxwd(xmm5, xmm5);
	paddd(xmm10, xmm5);
	pmovsxbw(xmm5, xmm2);
	movhlps(xmm6, xmm2);
	pmovsxbw(xmm6, xmm6);
	phaddw(xmm5, xmm6);
	phaddw(xmm5, xmm5);
	pmovsxwd(xmm5, xmm5);
	paddd(xmm11, xmm5);
	movdqu(xword[B-0x60], xmm4);
	movdqu(xword[B-0x50], xmm2);
	sub(B, -64);
	dec(I);
	jg(l8d0, T_NEAR);
	align(4);

L(l9d0);
	test(M, 0x2);
	jle(la64, T_NEAR);
	movdqu(xmm0, xword[A1-0x80]);
	add(A1, LDA);
	movdqu(xmm1, xword[A1-0x80]);
	add(A1, LDA);
	movdqa(xmm2, xmm0);
	punpcklbw(xmm0, xmm1);
	punpckhbw(xmm2, xmm1);
	pmovsxbw(xmm5, xmm0);
	phaddw(xmm5, xmm5);
	pmovsxwd(xmm5, xmm5);
	paddd(xmm8, xmm5);
	movhlps(xmm6, xmm0);
	pmovsxbw(xmm6, xmm6);
	phaddw(xmm6, xmm6);
	pmovsxwd(xmm6, xmm6);
	paddd(xmm9, xmm6);
	pmovsxbw(xmm5, xmm2);
	phaddw(xmm5, xmm5);
	pmovsxwd(xmm5, xmm5);
	paddd(xmm10, xmm5);
	movhlps(xmm6, xmm2);
	pmovsxbw(xmm6, xmm6);
	phaddw(xmm6, xmm6);
	pmovsxwd(xmm6, xmm6);
	paddd(xmm11, xmm6);
	movdqu(xword[B-0x80], xmm0);
	movdqu(xword[B-0x70], xmm2);
	sub(B, -32);
	align(4);

L(la64);
	test(M, 0x1);
	jle(lab8, T_NEAR);
	movdqu(xmm0, xword[A1-0x80]);
	add(A1, LDA);
	pmovsxbd(xmm5, xmm0);
	paddd(xmm8, xmm5);
	pshufd(xmm6, xmm0, 0x55);
	pmovsxbd(xmm6, xmm6);
	paddd(xmm9, xmm6);
	pshufd(xmm5, xmm0, 0xaa);
	pmovsxbd(xmm5, xmm5);
	paddd(xmm10, xmm5);
	pshufd(xmm6, xmm0, 0xff);
	pmovsxbd(xmm6, xmm6);
	paddd(xmm11, xmm6);
	movdqu(xword[B-0x80], xmm0);
	sub(B, -16);
	align(4);

L(lab8);
	mov(A1, qword[ARG_BIAS]);
	movdqu(xword[A1], xmm8);
	movdqu(xword[A1+0x10], xmm9);
	movdqu(xword[A1+0x20], xmm10);
	movdqu(xword[A1+0x30], xmm11);
	add(qword[ARG_BIAS], 0x40);
	sub(N, 0x10);
	cmp(N, 0x10);
	jge(l8a8, T_NEAR);
	align(4);

L(lae8);
	cmp(N, 0x8);
	jl(ld78, T_NEAR);
	align(4);

L(laf4);
	mov(A1, A);
	add(A, 0x8);
	pxor(xmm8, xmm8);
	pxor(xmm9, xmm9);
	mov(I, M);
	sar(I, 0x3);
	jle(lc30, T_NEAR);
	align(4);

L(lb14);
	movq(xmm0, qword[A1-0x80]);
	add(A1, LDA);
	movq(xmm1, qword[A1-0x80]);
	add(A1, LDA);
	movq(xmm2, qword[A1-0x80]);
	add(A1, LDA);
	movq(xmm3, qword[A1-0x80]);
	add(A1, LDA);
	punpcklbw(xmm0, xmm1);
	punpcklbw(xmm2, xmm3);
	movdqa(xmm1, xmm0);
	punpcklwd(xmm0, xmm2);
	punpckhwd(xmm1, xmm2);
	pmovsxbw(xmm5, xmm0);
	movhlps(xmm6, xmm0);
	pmovsxbw(xmm6, xmm6);
	phaddw(xmm5, xmm6);
	phaddw(xmm5, xmm5);
	pmovsxwd(xmm5, xmm5);
	paddd(xmm8, xmm5);
	pmovsxbw(xmm5, xmm1);
	movhlps(xmm6, xmm1);
	pmovsxbw(xmm6, xmm6);
	phaddw(xmm5, xmm6);
	phaddw(xmm5, xmm5);
	pmovsxwd(xmm5, xmm5);
	paddd(xmm9, xmm5);
	movdqu(xword[B-0x80], xmm0);
	movdqu(xword[B-0x70], xmm1);
	movq(xmm0, qword[A1-0x80]);
	add(A1, LDA);
	movq(xmm1, qword[A1-0x80]);
	add(A1, LDA);
	movq(xmm2, qword[A1-0x80]);
	add(A1, LDA);
	movq(xmm3, qword[A1-0x80]);
	add(A1, LDA);
	punpcklbw(xmm0, xmm1);
	punpcklbw(xmm2, xmm3);
	movdqa(xmm1, xmm0);
	punpcklwd(xmm0, xmm2);
	punpckhwd(xmm1, xmm2);
	pmovsxbw(xmm5, xmm0);
	movhlps(xmm6, xmm0);
	pmovsxbw(xmm6, xmm6);
	phaddw(xmm5, xmm6);
	phaddw(xmm5, xmm5);
	pmovsxwd(xmm5, xmm5);
	paddd(xmm8, xmm5);
	pmovsxbw(xmm5, xmm1);
	movhlps(xmm6, xmm1);
	pmovsxbw(xmm6, xmm6);
	phaddw(xmm5, xmm6);
	phaddw(xmm5, xmm5);
	pmovsxwd(xmm5, xmm5);
	paddd(xmm9, xmm5);
	movdqu(xword[B-0x60], xmm0);
	movdqu(xword[B-0x50], xmm1);
	sub(B, -64);
	dec(I);
	jg(lb14, T_NEAR);
	align(4);

L(lc30);
	test(M, 0x4);
	jle(lcc8, T_NEAR);
	movq(xmm0, qword[A1-0x80]);
	add(A1, LDA);
	movq(xmm1, qword[A1-0x80]);
	add(A1, LDA);
	movq(xmm2, qword[A1-0x80]);
	add(A1, LDA);
	movq(xmm3, qword[A1-0x80]);
	add(A1, LDA);
	punpcklbw(xmm0, xmm1);
	punpcklbw(xmm2, xmm3);
	movdqa(xmm1, xmm0);
	punpcklwd(xmm0, xmm2);
	punpckhwd(xmm1, xmm2);
	pmovsxbw(xmm5, xmm0);
	movhlps(xmm6, xmm0);
	pmovsxbw(xmm6, xmm6);
	phaddw(xmm5, xmm6);
	phaddw(xmm5, xmm5);
	pmovsxwd(xmm5, xmm5);
	paddd(xmm8, xmm5);
	pmovsxbw(xmm5, xmm1);
	movhlps(xmm6, xmm1);
	pmovsxbw(xmm6, xmm6);
	phaddw(xmm5, xmm6);
	phaddw(xmm5, xmm5);
	pmovsxwd(xmm5, xmm5);
	paddd(xmm9, xmm5);
	movdqu(xword[B-0x80], xmm0);
	movdqu(xword[B-0x70], xmm1);
	sub(B, -32);
	align(4);

L(lcc8);
	test(M, 0x2);
	jle(ld1c, T_NEAR);
	movq(xmm0, qword[A1-0x80]);
	add(A1, LDA);
	movq(xmm1, qword[A1-0x80]);
	add(A1, LDA);
	punpcklbw(xmm0, xmm1);
	pmovsxbw(xmm5, xmm0);
	phaddw(xmm5, xmm5);
	pmovsxwd(xmm5, xmm5);
	paddd(xmm8, xmm5);
	movhlps(xmm6, xmm0);
	pmovsxbw(xmm6, xmm6);
	phaddw(xmm6, xmm6);
	pmovsxwd(xmm6, xmm6);
	paddd(xmm9, xmm6);
	movdqu(xword[B-0x80], xmm0);
	sub(B, -16);
	align(4);

L(ld1c);
	test(M, 0x1);
	jle(ld54, T_NEAR);
	movq(xmm0, qword[A1-0x80]);
	add(A1, LDA);
	pmovsxbd(xmm5, xmm0);
	pshufd(xmm6, xmm0, 0x55);
	pmovsxbd(xmm6, xmm6);
	paddd(xmm8, xmm5);
	paddd(xmm9, xmm6);
	movq(qword[B-0x80], xmm0);
	sub(B, -8);
	align(4);

L(ld54);
	mov(A1, qword[ARG_BIAS]);
	movdqu(xword[A1], xmm8);
	movdqu(xword[A1+0x10], xmm9);
	add(qword[ARG_BIAS], 0x20);
	sub(N, 0x8);
	cmp(N, 0x8);
	jge(laf4, T_NEAR);
	align(4);

L(ld78);
	cmp(N, 0x4);
	jl(lf3c, T_NEAR);
	align(4);

L(ld84);
	mov(A1, A);
	add(A, 0x4);
	pxor(xmm7, xmm7);
	mov(I, M);
	sar(I, 0x3);
	jle(le58, T_NEAR);
	align(4);

L(ld9c);
	movd(xmm0, dword[A1-0x80]);
	add(A1, LDA);
	movd(xmm1, dword[A1-0x80]);
	add(A1, LDA);
	movd(xmm2, dword[A1-0x80]);
	add(A1, LDA);
	movd(xmm3, dword[A1-0x80]);
	add(A1, LDA);
	punpcklbw(xmm0, xmm1);
	punpcklbw(xmm2, xmm3);
	punpcklwd(xmm0, xmm2);
	pmovsxbw(xmm5, xmm0);
	movhlps(xmm6, xmm0);
	pmovsxbw(xmm6, xmm6);
	phaddw(xmm5, xmm6);
	phaddw(xmm5, xmm5);
	pmovsxwd(xmm5, xmm5);
	paddd(xmm7, xmm5);
	movdqu(xword[B-0x80], xmm0);
	movd(xmm0, dword[A1-0x80]);
	add(A1, LDA);
	movd(xmm1, dword[A1-0x80]);
	add(A1, LDA);
	movd(xmm2, dword[A1-0x80]);
	add(A1, LDA);
	movd(xmm3, dword[A1-0x80]);
	add(A1, LDA);
	punpcklbw(xmm0, xmm1);
	punpcklbw(xmm2, xmm3);
	punpcklwd(xmm0, xmm2);
	pmovsxbw(xmm5, xmm0);
	movhlps(xmm6, xmm0);
	pmovsxbw(xmm6, xmm6);
	phaddw(xmm5, xmm6);
	phaddw(xmm5, xmm5);
	pmovsxwd(xmm5, xmm5);
	paddd(xmm7, xmm5);
	movdqu(xword[B-0x70], xmm0);
	sub(B, -32);
	dec(I);
	jg(ld9c, T_NEAR);
	align(4);

L(le58);
	test(M, 0x4);
	jle(lebc, T_NEAR);
	movd(xmm0, dword[A1-0x80]);
	add(A1, LDA);
	movd(xmm1, dword[A1-0x80]);
	add(A1, LDA);
	movd(xmm2, dword[A1-0x80]);
	add(A1, LDA);
	movd(xmm3, dword[A1-0x80]);
	add(A1, LDA);
	punpcklbw(xmm0, xmm1);
	punpcklbw(xmm2, xmm3);
	punpcklwd(xmm0, xmm2);
	pmovsxbw(xmm5, xmm0);
	movhlps(xmm6, xmm0);
	pmovsxbw(xmm6, xmm6);
	phaddw(xmm5, xmm6);
	phaddw(xmm5, xmm5);
	pmovsxwd(xmm5, xmm5);
	paddd(xmm7, xmm5);
	movdqu(xword[B-0x80], xmm0);
	sub(B, -16);
	align(4);

L(lebc);
	test(M, 0x2);
	jle(lef8, T_NEAR);
	movd(xmm0, dword[A1-0x80]);
	add(A1, LDA);
	movd(xmm1, dword[A1-0x80]);
	add(A1, LDA);
	punpcklbw(xmm0, xmm1);
	pmovsxbw(xmm5, xmm0);
	phaddw(xmm5, xmm5);
	pmovsxwd(xmm5, xmm5);
	paddd(xmm7, xmm5);
	movq(qword[B-0x80], xmm0);
	sub(B, -8);
	align(4);

L(lef8);
	test(M, 0x1);
	jle(lf1c, T_NEAR);
	movd(xmm0, dword[A1-0x80]);
	pmovsxbd(xmm5, xmm0);
	paddd(xmm7, xmm5);
	movd(dword[B-0x80], xmm0);
	sub(B, -4);
	align(4);

L(lf1c);
	mov(A1, qword[ARG_BIAS]);
	movdqu(xword[A1], xmm7);
	add(qword[ARG_BIAS], 0x10);
	sub(N, 0x4);
	cmp(N, 0x4);
	jge(ld84, T_NEAR);
	align(4);

L(lf3c);
	cmp(N, 0x2);
	jl(l111a, T_NEAR);
	align(4);

L(lf48);
	mov(A1, A);
	add(A, 0x2);
	pxor(xmm7, xmm7);
	mov(LDA3, M);
	sar(LDA3, 0x3);
	jle(l1024, T_NEAR);
	align(4);

L(lf60);
	mov(ax, word[A1-0x80]);
	add(A1, LDA);
	pinsrw(xmm0, eax, 0x0);
	mov(ax, word[A1-0x80]);
	add(A1, LDA);
	pinsrw(xmm1, eax, 0x0);
	mov(ax, word[A1-0x80]);
	add(A1, LDA);
	pinsrw(xmm2, eax, 0x0);
	mov(ax, word[A1-0x80]);
	add(A1, LDA);
	pinsrw(xmm3, eax, 0x0);
	punpcklbw(xmm0, xmm1);
	punpcklbw(xmm2, xmm3);
	punpcklwd(xmm0, xmm2);
	mov(ax, word[A1-0x80]);
	add(A1, LDA);
	pinsrw(xmm1, eax, 0x0);
	mov(ax, word[A1-0x80]);
	add(A1, LDA);
	pinsrw(xmm2, eax, 0x0);
	mov(ax, word[A1-0x80]);
	add(A1, LDA);
	pinsrw(xmm3, eax, 0x0);
	mov(ax, word[A1-0x80]);
	add(A1, LDA);
	pinsrw(xmm4, eax, 0x0);
	punpcklbw(xmm1, xmm2);
	punpcklbw(xmm3, xmm4);
	punpcklwd(xmm1, xmm3);
	punpcklqdq(xmm0, xmm1);
	pshufd(xmm6, xmm0, 0xd8);
	pmovsxbw(xmm5, xmm6);
	movhlps(xmm6, xmm6);
	pmovsxbw(xmm6, xmm6);
	phaddw(xmm5, xmm6);
	phaddw(xmm5, xmm5);
	phaddw(xmm5, xmm5);
	pmovsxwd(xmm5, xmm5);
	paddd(xmm7, xmm5);
	movdqu(xword[B-0x80], xmm0);
	sub(B, -16);
	dec(LDA3);
	jg(lf60, T_NEAR);
	align(4);

L(l1024);
	test(M, 0x4);
	jle(l1090, T_NEAR);
	mov(ax, word[A1-0x80]);
	add(A1, LDA);
	pinsrw(xmm0, eax, 0x0);
	mov(ax, word[A1-0x80]);
	add(A1, LDA);
	pinsrw(xmm1, eax, 0x0);
	mov(ax, word[A1-0x80]);
	add(A1, LDA);
	pinsrw(xmm2, eax, 0x0);
	mov(ax, word[A1-0x80]);
	add(A1, LDA);
	pinsrw(xmm3, eax, 0x0);
	punpcklbw(xmm0, xmm1);
	punpcklbw(xmm2, xmm3);
	punpcklwd(xmm0, xmm2);
	pmovsxbw(xmm5, xmm0);
	phaddw(xmm5, xmm5);
	phaddw(xmm5, xmm5);
	pmovsxwd(xmm5, xmm5);
	paddd(xmm7, xmm5);
	movq(qword[B-0x80], xmm0);
	sub(B, -8);
	align(4);

L(l1090);
	test(M, 0x2);
	jle(l10d4, T_NEAR);
	mov(ax, word[A1-0x80]);
	add(A1, LDA);
	pinsrw(xmm0, eax, 0x0);
	mov(ax, word[A1-0x80]);
	add(A1, LDA);
	pinsrw(xmm1, eax, 0x0);
	punpcklbw(xmm0, xmm1);
	pmovsxbw(xmm5, xmm0);
	phaddw(xmm5, xmm5);
	pmovsxwd(xmm5, xmm5);
	paddd(xmm7, xmm5);
	movd(dword[B-0x80], xmm0);
	sub(B, -4);
	align(4);

L(l10d4);
	test(M, 0x1);
	jle(l10fc, T_NEAR);
	mov(ax, word[A1-0x80]);
	pinsrw(xmm0, eax, 0x0);
	pmovsxbd(xmm5, xmm0);
	paddd(xmm7, xmm5);
	mov(word[B-0x80], ax);
	sub(B, -2);
	align(4);

L(l10fc);
	mov(A1, qword[ARG_BIAS]);
	movq(qword[A1], xmm7);
	add(qword[ARG_BIAS], 0x8);
	sub(N, 0x2);
	cmp(N, 0x2);
	jge(lf48, T_NEAR);
	align(4);

L(l111a);
	cmp(N, 0x1);
	jl(l12bc, T_NEAR);
	align(4);

L(l1124);
	mov(A1, A);
	add(A, 0x1);
	pxor(xmm7, xmm7);
	mov(LDA3, M);
	sar(LDA3, 0x3);
	jle(l11d4, T_NEAR);
	align(4);

L(l113c);
	mov(al, byte[A1-0x80]);
	add(A1, LDA);
	pinsrb(xmm0, eax, 0x0);
	mov(al, byte[A1-0x80]);
	add(A1, LDA);
	pinsrb(xmm0, eax, 0x1);
	mov(al, byte[A1-0x80]);
	add(A1, LDA);
	pinsrb(xmm0, eax, 0x2);
	mov(al, byte[A1-0x80]);
	add(A1, LDA);
	pinsrb(xmm0, eax, 0x3);
	mov(al, byte[A1-0x80]);
	add(A1, LDA);
	pinsrb(xmm0, eax, 0x4);
	mov(al, byte[A1-0x80]);
	add(A1, LDA);
	pinsrb(xmm0, eax, 0x5);
	mov(al, byte[A1-0x80]);
	add(A1, LDA);
	pinsrb(xmm0, eax, 0x6);
	mov(al, byte[A1-0x80]);
	add(A1, LDA);
	pinsrb(xmm0, eax, 0x7);
	pmovsxbw(xmm5, xmm0);
	phaddw(xmm5, xmm6);
	phaddw(xmm5, xmm5);
	phaddw(xmm5, xmm5);
	pmovsxwd(xmm5, xmm5);
	paddd(xmm7, xmm5);
	movq(qword[B-0x80], xmm0);
	sub(B, -8);
	dec(LDA3);
	jg(l113c, T_NEAR);
	align(4);

L(l11d4);
	test(M, 0x4);
	jle(l1234, T_NEAR);
	mov(al, byte[A1-0x80]);
	add(A1, LDA);
	pinsrb(xmm0, eax, 0x0);
	mov(al, byte[A1-0x80]);
	add(A1, LDA);
	pinsrb(xmm0, eax, 0x1);
	mov(al, byte[A1-0x80]);
	add(A1, LDA);
	pinsrb(xmm0, eax, 0x2);
	mov(al, byte[A1-0x80]);
	add(A1, LDA);
	pinsrb(xmm0, eax, 0x3);
	pmovsxbw(xmm5, xmm0);
	phaddw(xmm5, xmm5);
	phaddw(xmm5, xmm5);
	pmovsxwd(xmm5, xmm5);
	paddd(xmm7, xmm5);
	movd(dword[B-0x80], xmm0);
	sub(B, -4);
	align(4);

L(l1234);
	test(M, 0x2);
	jle(l1278, T_NEAR);
	mov(al, byte[A1-0x80]);
	add(A1, LDA);
	pinsrb(xmm0, eax, 0x0);
	mov(byte[B-0x80], al);
	mov(al, byte[A1-0x80]);
	add(A1, LDA);
	pinsrb(xmm0, eax, 0x1);
	pmovsxbw(xmm5, xmm0);
	phaddw(xmm5, xmm5);
	pmovsxwd(xmm5, xmm5);
	paddd(xmm7, xmm5);
	mov(byte[B-0x7f], al);
	sub(B, -2);
	align(4);

L(l1278);
	test(M, 0x1);
	jle(l129c, T_NEAR);
	mov(al, byte[A1-0x80]);
	pinsrw(xmm0, eax, 0x0);
	pmovsxbd(xmm5, xmm0);
	paddd(xmm7, xmm5);
	mov(byte[B-0x80], al);
	sub(B, -1);
	align(4);

L(l129c);
	mov(A1, qword[ARG_BIAS]);
	movd(dword[A1], xmm7);
	add(qword[ARG_BIAS], 0x4);
	sub(N, 0x1);
	cmp(N, 0x1);
	jge(l1124, T_NEAR);
	align(4);

L(l12bc);

	postamble();
}
outLocalLabel();

#undef M
#undef N
#undef A
#undef LDA
#undef ALPHA
#undef B
#undef I
#undef A1
#undef A2
#undef LDA3
#ifdef _WIN32
#undef ARG_ALPHA
#undef ARG_B
#endif
#undef ARG_BIAS
}

}
}
}
