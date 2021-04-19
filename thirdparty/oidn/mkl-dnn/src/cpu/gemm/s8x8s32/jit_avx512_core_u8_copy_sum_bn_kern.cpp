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

jit_avx512_core_u8_copy_sum_bn_kern::jit_avx512_core_u8_copy_sum_bn_kern(): jit_generator(nullptr, GEMM_CODE_SIZE)
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

Xbyak::Label l20;
Xbyak::Label l22c;
Xbyak::Label l340;
Xbyak::Label l3f8;
Xbyak::Label l48;
Xbyak::Label l498;
Xbyak::Label l51c;
Xbyak::Label l540;
Xbyak::Label l54c;
Xbyak::Label l56c;
Xbyak::Label l664;
Xbyak::Label l6f8;
Xbyak::Label l75c;
Xbyak::Label l7b4;
Xbyak::Label l7fc;
Xbyak::Label l81c;
Xbyak::Label l828;
Xbyak::Label l848;
Xbyak::Label l8d8;
Xbyak::Label l930;
Xbyak::Label l974;
Xbyak::Label l9b8;
Xbyak::Label l9ec;
Xbyak::Label la0a;
Xbyak::Label la14;
Xbyak::Label la28;
Xbyak::Label la6c;
Xbyak::Label laa8;
Xbyak::Label lae0;
Xbyak::Label lb14;
Xbyak::Label lb38;
Xbyak::Label lb58;

	preamble();
	auto stacksize = get_size_of_abi_save_regs();
#ifdef _WIN32
	mov(ALPHA, ptr[ARG_ALPHA]);
	mov(B, ptr[ARG_B]);
#endif

	mov(N, qword[N]);
	mov(M, qword[M]);
	mov(LDA, qword[LDA]);
	sub(A, -128);
	sub(B, -128);
	lea(LDA3, ptr[LDA+LDA*2]);
	cmp(N, 0x8);
	jl(l540, T_NEAR);
	align(4);

L(l20);
	mov(A1, A);
	lea(A2, ptr[A1+LDA*4]);
	lea(I, ptr[A1+LDA*8]);
	mov(A, I);
	pxor(xmm8, xmm8);
	pxor(xmm9, xmm9);
	mov(I, M);
	sar(I, 0x4);
	jle(l22c, T_NEAR);
	align(4);

L(l48);
	movdqu(xmm0, xword[A1-0x80]);
	movdqu(xmm1, xword[A1+LDA*1-0x80]);
	movdqu(xmm2, xword[A1+LDA*2-0x80]);
	movdqu(xmm3, xword[A1+LDA3*1-0x80]);
	sub(A1, -16);
	movdqa(xmm4, xmm0);
	punpckldq(xmm0, xmm1);
	punpckhdq(xmm4, xmm1);
	movdqa(xmm5, xmm2);
	punpckldq(xmm2, xmm3);
	punpckhdq(xmm5, xmm3);
	movdqa(xmm1, xmm0);
	punpcklqdq(xmm0, xmm2);
	punpckhqdq(xmm1, xmm2);
	movdqa(xmm3, xmm4);
	punpcklqdq(xmm4, xmm5);
	punpckhqdq(xmm3, xmm5);
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
	paddd(xmm8, xmm5);
	movdqu(xword[B-0x60], xmm1);
	pmovsxbw(xmm5, xmm4);
	movhlps(xmm6, xmm4);
	pmovsxbw(xmm6, xmm6);
	phaddw(xmm5, xmm6);
	phaddw(xmm5, xmm5);
	pmovsxwd(xmm5, xmm5);
	paddd(xmm8, xmm5);
	movdqu(xword[B-0x40], xmm4);
	pmovsxbw(xmm5, xmm3);
	movhlps(xmm6, xmm3);
	pmovsxbw(xmm6, xmm6);
	phaddw(xmm5, xmm6);
	phaddw(xmm5, xmm5);
	pmovsxwd(xmm5, xmm5);
	paddd(xmm8, xmm5);
	movdqu(xword[B-0x20], xmm3);
	movdqu(xmm0, xword[A2-0x80]);
	movdqu(xmm1, xword[A2+LDA*1-0x80]);
	movdqu(xmm2, xword[A2+LDA*2-0x80]);
	movdqu(xmm3, xword[A2+LDA3*1-0x80]);
	sub(A2, -16);
	movdqa(xmm4, xmm0);
	punpckldq(xmm0, xmm1);
	punpckhdq(xmm4, xmm1);
	movdqa(xmm5, xmm2);
	punpckldq(xmm2, xmm3);
	punpckhdq(xmm5, xmm3);
	movdqa(xmm1, xmm0);
	punpcklqdq(xmm0, xmm2);
	punpckhqdq(xmm1, xmm2);
	movdqa(xmm3, xmm4);
	punpcklqdq(xmm4, xmm5);
	punpckhqdq(xmm3, xmm5);
	pmovsxbw(xmm5, xmm0);
	movhlps(xmm6, xmm0);
	pmovsxbw(xmm6, xmm6);
	phaddw(xmm5, xmm6);
	phaddw(xmm5, xmm5);
	pmovsxwd(xmm5, xmm5);
	paddd(xmm9, xmm5);
	movdqu(xword[B-0x70], xmm0);
	pmovsxbw(xmm5, xmm1);
	movhlps(xmm6, xmm1);
	pmovsxbw(xmm6, xmm6);
	phaddw(xmm5, xmm6);
	phaddw(xmm5, xmm5);
	pmovsxwd(xmm5, xmm5);
	paddd(xmm9, xmm5);
	movdqu(xword[B-0x50], xmm1);
	pmovsxbw(xmm5, xmm4);
	movhlps(xmm6, xmm4);
	pmovsxbw(xmm6, xmm6);
	phaddw(xmm5, xmm6);
	phaddw(xmm5, xmm5);
	pmovsxwd(xmm5, xmm5);
	paddd(xmm9, xmm5);
	movdqu(xword[B-0x30], xmm4);
	pmovsxbw(xmm5, xmm3);
	movhlps(xmm6, xmm3);
	pmovsxbw(xmm6, xmm6);
	phaddw(xmm5, xmm6);
	phaddw(xmm5, xmm5);
	pmovsxwd(xmm5, xmm5);
	paddd(xmm9, xmm5);
	movdqu(xword[B-0x10], xmm3);
	sub(B, -128);
	dec(I);
	jg(l48, T_NEAR);
	align(4);

L(l22c);
	test(M, 0x8);
	jle(l340, T_NEAR);
	movq(xmm0, qword[A1-0x80]);
	movq(xmm1, qword[A1+LDA*1-0x80]);
	movq(xmm2, qword[A1+LDA*2-0x80]);
	movq(xmm3, qword[A1+LDA3*1-0x80]);
	sub(A1, -8);
	punpckldq(xmm0, xmm1);
	punpckldq(xmm2, xmm3);
	movdqa(xmm1, xmm0);
	punpcklqdq(xmm0, xmm2);
	punpckhqdq(xmm1, xmm2);
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
	paddd(xmm8, xmm5);
	movdqu(xword[B-0x60], xmm1);
	movq(xmm0, qword[A2-0x80]);
	movq(xmm1, qword[A2+LDA*1-0x80]);
	movq(xmm2, qword[A2+LDA*2-0x80]);
	movq(xmm3, qword[A2+LDA3*1-0x80]);
	sub(A2, -8);
	punpckldq(xmm0, xmm1);
	punpckldq(xmm2, xmm3);
	movdqa(xmm1, xmm0);
	punpcklqdq(xmm0, xmm2);
	punpckhqdq(xmm1, xmm2);
	pmovsxbw(xmm5, xmm0);
	movhlps(xmm6, xmm0);
	pmovsxbw(xmm6, xmm6);
	phaddw(xmm5, xmm6);
	phaddw(xmm5, xmm5);
	pmovsxwd(xmm5, xmm5);
	paddd(xmm9, xmm5);
	movdqu(xword[B-0x70], xmm0);
	pmovsxbw(xmm5, xmm1);
	movhlps(xmm6, xmm1);
	pmovsxbw(xmm6, xmm6);
	phaddw(xmm5, xmm6);
	phaddw(xmm5, xmm5);
	pmovsxwd(xmm5, xmm5);
	paddd(xmm9, xmm5);
	movdqu(xword[B-0x50], xmm1);
	sub(B, -64);
	align(4);

L(l340);
	test(M, 0x4);
	jle(l3f8, T_NEAR);
	movd(xmm0, dword[A1-0x80]);
	movd(xmm1, dword[A1+LDA*1-0x80]);
	movd(xmm2, dword[A1+LDA*2-0x80]);
	movd(xmm3, dword[A1+LDA3*1-0x80]);
	sub(A1, -4);
	punpckldq(xmm0, xmm1);
	punpckldq(xmm2, xmm3);
	punpcklqdq(xmm0, xmm2);
	pmovsxbw(xmm5, xmm0);
	movhlps(xmm6, xmm0);
	pmovsxbw(xmm6, xmm6);
	phaddw(xmm5, xmm6);
	phaddw(xmm5, xmm5);
	pmovsxwd(xmm5, xmm5);
	paddd(xmm8, xmm5);
	movdqu(xword[B-0x80], xmm0);
	movd(xmm0, dword[A2-0x80]);
	movd(xmm1, dword[A2+LDA*1-0x80]);
	movd(xmm2, dword[A2+LDA*2-0x80]);
	movd(xmm3, dword[A2+LDA3*1-0x80]);
	sub(A2, -4);
	punpckldq(xmm0, xmm1);
	punpckldq(xmm2, xmm3);
	punpcklqdq(xmm0, xmm2);
	pmovsxbw(xmm5, xmm0);
	movhlps(xmm6, xmm0);
	pmovsxbw(xmm6, xmm6);
	phaddw(xmm5, xmm6);
	phaddw(xmm5, xmm5);
	pmovsxwd(xmm5, xmm5);
	paddd(xmm9, xmm5);
	movdqu(xword[B-0x70], xmm0);
	sub(B, -32);
	align(4);

L(l3f8);
	test(M, 0x2);
	jle(l498, T_NEAR);
	mov(ax, word[A1-0x80]);
	pinsrw(xmm0, eax, 0x0);
	mov(ax, word[A1+LDA*1-0x80]);
	pinsrw(xmm0, eax, 0x1);
	mov(ax, word[A1+LDA*2-0x80]);
	pinsrw(xmm0, eax, 0x2);
	mov(ax, word[A1+LDA3*1-0x80]);
	sub(A1, -2);
	pinsrw(xmm0, eax, 0x3);
	mov(ax, word[A2-0x80]);
	pinsrw(xmm0, eax, 0x4);
	mov(ax, word[A2+LDA*1-0x80]);
	pinsrw(xmm0, eax, 0x5);
	mov(ax, word[A2+LDA*2-0x80]);
	pinsrw(xmm0, eax, 0x6);
	mov(ax, word[A2+LDA3*1-0x80]);
	sub(A2, -2);
	pinsrw(xmm0, eax, 0x7);
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

L(l498);
	test(M, 0x1);
	jle(l51c, T_NEAR);
	mov(al, byte[A1-0x80]);
	pinsrb(xmm0, eax, 0x0);
	mov(al, byte[A1+LDA*1-0x80]);
	pinsrb(xmm0, eax, 0x1);
	mov(al, byte[A1+LDA*2-0x80]);
	pinsrb(xmm0, eax, 0x2);
	mov(al, byte[A1+LDA3*1-0x80]);
	pinsrb(xmm0, eax, 0x3);
	mov(al, byte[A2-0x80]);
	pinsrb(xmm0, eax, 0x4);
	mov(al, byte[A2+LDA*1-0x80]);
	pinsrb(xmm0, eax, 0x5);
	mov(al, byte[A2+LDA*2-0x80]);
	pinsrb(xmm0, eax, 0x6);
	mov(al, byte[A2+LDA3*1-0x80]);
	pinsrb(xmm0, eax, 0x7);
	pmovsxbd(xmm5, xmm0);
	pshufd(xmm6, xmm0, 0x55);
	pmovsxbd(xmm6, xmm6);
	paddd(xmm8, xmm5);
	paddd(xmm9, xmm6);
	movq(qword[B-0x80], xmm0);
	sub(B, -8);
	align(4);

L(l51c);
	mov(A1, qword[ARG_BIAS]);
	movdqu(xword[A1], xmm8);
	movdqu(xword[A1+0x10], xmm9);
	add(qword[ARG_BIAS], 0x20);
	sub(N, 0x8);
	cmp(N, 0x8);
	jge(l20, T_NEAR);
	align(4);

L(l540);
	cmp(N, 0x4);
	jl(l81c, T_NEAR);
	align(4);

L(l54c);
	mov(A1, A);
	lea(A2, ptr[A1+LDA*2]);
	lea(I, ptr[A1+LDA*4]);
	mov(A, I);
	pxor(xmm7, xmm7);
	mov(I, M);
	sar(I, 0x4);
	jle(l664, T_NEAR);
	align(4);

L(l56c);
	movdqu(xmm0, xword[A1-0x80]);
	movdqu(xmm1, xword[A1+LDA*1-0x80]);
	sub(A1, -16);
	movdqu(xmm2, xword[A2-0x80]);
	movdqu(xmm3, xword[A2+LDA*1-0x80]);
	sub(A2, -16);
	movdqa(xmm4, xmm0);
	punpckldq(xmm0, xmm1);
	punpckhdq(xmm4, xmm1);
	movdqa(xmm5, xmm2);
	punpckldq(xmm2, xmm3);
	punpckhdq(xmm5, xmm3);
	movdqa(xmm1, xmm0);
	punpcklqdq(xmm0, xmm2);
	punpckhqdq(xmm1, xmm2);
	movdqa(xmm3, xmm4);
	punpcklqdq(xmm4, xmm5);
	punpckhqdq(xmm3, xmm5);
	pmovsxbw(xmm5, xmm0);
	movhlps(xmm6, xmm0);
	pmovsxbw(xmm6, xmm6);
	phaddw(xmm5, xmm6);
	phaddw(xmm5, xmm5);
	pmovsxwd(xmm5, xmm5);
	paddd(xmm7, xmm5);
	movdqu(xword[B-0x80], xmm0);
	pmovsxbw(xmm5, xmm1);
	movhlps(xmm6, xmm1);
	pmovsxbw(xmm6, xmm6);
	phaddw(xmm5, xmm6);
	phaddw(xmm5, xmm5);
	pmovsxwd(xmm5, xmm5);
	paddd(xmm7, xmm5);
	movdqu(xword[B-0x70], xmm1);
	pmovsxbw(xmm5, xmm4);
	movhlps(xmm6, xmm4);
	pmovsxbw(xmm6, xmm6);
	phaddw(xmm5, xmm6);
	phaddw(xmm5, xmm5);
	pmovsxwd(xmm5, xmm5);
	paddd(xmm7, xmm5);
	movdqu(xword[B-0x60], xmm4);
	pmovsxbw(xmm5, xmm3);
	movhlps(xmm6, xmm3);
	pmovsxbw(xmm6, xmm6);
	phaddw(xmm5, xmm6);
	phaddw(xmm5, xmm5);
	pmovsxwd(xmm5, xmm5);
	paddd(xmm7, xmm5);
	movdqu(xword[B-0x50], xmm3);
	sub(B, -64);
	dec(I);
	jg(l56c, T_NEAR);
	align(4);

L(l664);
	test(M, 0x8);
	jle(l6f8, T_NEAR);
	movq(xmm0, qword[A1-0x80]);
	movq(xmm1, qword[A1+LDA*1-0x80]);
	sub(A1, -8);
	movq(xmm2, qword[A2-0x80]);
	movq(xmm3, qword[A2+LDA*1-0x80]);
	sub(A2, -8);
	punpckldq(xmm0, xmm1);
	punpckldq(xmm2, xmm3);
	movdqa(xmm1, xmm0);
	punpcklqdq(xmm0, xmm2);
	punpckhqdq(xmm1, xmm2);
	pmovsxbw(xmm5, xmm0);
	movhlps(xmm6, xmm0);
	pmovsxbw(xmm6, xmm6);
	phaddw(xmm5, xmm6);
	phaddw(xmm5, xmm5);
	pmovsxwd(xmm5, xmm5);
	paddd(xmm7, xmm5);
	movdqu(xword[B-0x80], xmm0);
	pmovsxbw(xmm5, xmm1);
	movhlps(xmm6, xmm1);
	pmovsxbw(xmm6, xmm6);
	phaddw(xmm5, xmm6);
	phaddw(xmm5, xmm5);
	pmovsxwd(xmm5, xmm5);
	paddd(xmm7, xmm5);
	movdqu(xword[B-0x70], xmm1);
	sub(B, -32);
	align(4);

L(l6f8);
	test(M, 0x4);
	jle(l75c, T_NEAR);
	movd(xmm0, dword[A1-0x80]);
	movd(xmm1, dword[A1+LDA*1-0x80]);
	sub(A1, -4);
	movd(xmm2, dword[A2-0x80]);
	movd(xmm3, dword[A2+LDA*1-0x80]);
	sub(A2, -4);
	punpckldq(xmm0, xmm1);
	punpckldq(xmm2, xmm3);
	punpcklqdq(xmm0, xmm2);
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

L(l75c);
	test(M, 0x2);
	jle(l7b4, T_NEAR);
	mov(ax, word[A1-0x80]);
	pinsrw(xmm0, eax, 0x0);
	mov(ax, word[A1+LDA*1-0x80]);
	sub(A1, -2);
	pinsrw(xmm0, eax, 0x1);
	mov(ax, word[A2-0x80]);
	pinsrw(xmm0, eax, 0x2);
	mov(ax, word[A2+LDA*1-0x80]);
	sub(A2, -2);
	pinsrw(xmm0, eax, 0x3);
	pmovsxbw(xmm5, xmm0);
	phaddw(xmm5, xmm5);
	pmovsxwd(xmm5, xmm5);
	paddd(xmm7, xmm5);
	movq(qword[B-0x80], xmm0);
	sub(B, -8);
	align(4);

L(l7b4);
	test(M, 0x1);
	jle(l7fc, T_NEAR);
	mov(al, byte[A1-0x80]);
	pinsrb(xmm0, eax, 0x0);
	mov(al, byte[A1+LDA*1-0x80]);
	pinsrb(xmm0, eax, 0x1);
	mov(al, byte[A2-0x80]);
	pinsrb(xmm0, eax, 0x2);
	mov(al, byte[A2+LDA*1-0x80]);
	pinsrb(xmm0, eax, 0x3);
	pmovsxbd(xmm5, xmm0);
	paddd(xmm7, xmm5);
	movd(dword[B-0x80], xmm0);
	sub(B, -4);
	align(4);

L(l7fc);
	mov(A1, qword[ARG_BIAS]);
	movdqu(xword[A1], xmm7);
	add(qword[ARG_BIAS], 0x10);
	sub(N, 0x4);
	cmp(N, 0x4);
	jge(l54c, T_NEAR);
	align(4);

L(l81c);
	cmp(N, 0x2);
	jl(la0a, T_NEAR);
	align(4);

L(l828);
	mov(A1, A);
	lea(A2, ptr[A1+LDA*1]);
	lea(I, ptr[A1+LDA*2]);
	mov(A, I);
	pxor(xmm7, xmm7);
	mov(I, M);
	sar(I, 0x4);
	jle(l8d8, T_NEAR);
	align(4);

L(l848);
	movdqu(xmm0, xword[A1-0x80]);
	sub(A1, -16);
	movdqu(xmm1, xword[A2-0x80]);
	sub(A2, -16);
	movdqa(xmm2, xmm0);
	punpckldq(xmm0, xmm1);
	punpckhdq(xmm2, xmm1);
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
	pshufd(xmm6, xmm2, 0xd8);
	pmovsxbw(xmm5, xmm6);
	movhlps(xmm6, xmm6);
	pmovsxbw(xmm6, xmm6);
	phaddw(xmm5, xmm6);
	phaddw(xmm5, xmm5);
	phaddw(xmm5, xmm5);
	pmovsxwd(xmm5, xmm5);
	paddd(xmm7, xmm5);
	movdqu(xword[B-0x70], xmm2);
	sub(B, -32);
	dec(I);
	jg(l848, T_NEAR);
	align(4);

L(l8d8);
	test(M, 0x8);
	jle(l930, T_NEAR);
	movq(xmm0, qword[A1-0x80]);
	sub(A1, -8);
	movq(xmm1, qword[A2-0x80]);
	sub(A2, -8);
	punpckldq(xmm0, xmm1);
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
	align(4);

L(l930);
	test(M, 0x4);
	jle(l974, T_NEAR);
	movd(xmm0, dword[A1-0x80]);
	sub(A1, -4);
	movd(xmm1, dword[A2-0x80]);
	sub(A2, -4);
	punpckldq(xmm0, xmm1);
	pmovsxbw(xmm5, xmm0);
	phaddw(xmm5, xmm5);
	phaddw(xmm5, xmm5);
	pmovsxwd(xmm5, xmm5);
	paddd(xmm7, xmm5);
	movq(qword[B-0x80], xmm0);
	sub(B, -8);
	align(4);

L(l974);
	test(M, 0x2);
	jle(l9b8, T_NEAR);
	mov(ax, word[A1-0x80]);
	sub(A1, -2);
	pinsrw(xmm0, eax, 0x0);
	mov(ax, word[A2-0x80]);
	sub(A2, -2);
	pinsrw(xmm0, eax, 0x1);
	pmovsxbw(xmm5, xmm0);
	phaddw(xmm5, xmm5);
	pmovsxwd(xmm5, xmm5);
	paddd(xmm7, xmm5);
	movd(dword[B-0x80], xmm0);
	sub(B, -4);
	align(4);

L(l9b8);
	test(M, 0x1);
	jle(l9ec, T_NEAR);
	mov(al, byte[A1-0x80]);
	pinsrb(xmm0, eax, 0x0);
	mov(byte[B-0x80], al);
	mov(al, byte[A2-0x80]);
	pinsrb(xmm0, eax, 0x1);
	mov(byte[B-0x7f], al);
	sub(B, -2);
	pmovsxbd(xmm5, xmm0);
	paddd(xmm7, xmm5);
	align(4);

L(l9ec);
	mov(A1, qword[ARG_BIAS]);
	movq(qword[A1], xmm7);
	add(qword[ARG_BIAS], 0x8);
	sub(N, 0x2);
	cmp(N, 0x2);
	jge(l828, T_NEAR);
	align(4);

L(la0a);
	cmp(N, 0x1);
	jl(lb58, T_NEAR);
	align(4);

L(la14);
	mov(A1, A);
	add(A, LDA);
	pxor(xmm7, xmm7);
	mov(I, M);
	sar(I, 0x4);
	jle(la6c, T_NEAR);
	align(4);

L(la28);
	movdqu(xmm0, xword[A1-0x80]);
	sub(A1, -16);
	pmovsxbw(xmm5, xmm0);
	movhlps(xmm6, xmm0);
	pmovsxbw(xmm6, xmm6);
	phaddw(xmm5, xmm6);
	phaddw(xmm5, xmm5);
	phaddw(xmm5, xmm5);
	phaddw(xmm5, xmm5);
	pmovsxwd(xmm5, xmm5);
	paddd(xmm7, xmm5);
	movdqu(xword[B-0x80], xmm0);
	sub(B, -16);
	dec(I);
	jg(la28, T_NEAR);
	align(4);

L(la6c);
	test(M, 0x8);
	jle(laa8, T_NEAR);
	movq(xmm0, qword[A1-0x80]);
	sub(A1, -8);
	pmovsxbw(xmm5, xmm0);
	phaddw(xmm5, xmm6);
	phaddw(xmm5, xmm5);
	phaddw(xmm5, xmm5);
	pmovsxwd(xmm5, xmm5);
	paddd(xmm7, xmm5);
	movq(qword[B-0x80], xmm0);
	sub(B, -8);
	align(4);

L(laa8);
	test(M, 0x4);
	jle(lae0, T_NEAR);
	movd(xmm0, dword[A1-0x80]);
	sub(A1, -4);
	pmovsxbw(xmm5, xmm0);
	phaddw(xmm5, xmm5);
	phaddw(xmm5, xmm5);
	pmovsxwd(xmm5, xmm5);
	paddd(xmm7, xmm5);
	movd(dword[B-0x80], xmm0);
	sub(B, -4);
	align(4);

L(lae0);
	test(M, 0x2);
	jle(lb14, T_NEAR);
	mov(ax, word[A1-0x80]);
	pinsrw(xmm0, eax, 0x0);
	pmovsxbw(xmm5, xmm0);
	phaddw(xmm5, xmm5);
	pmovsxwd(xmm5, xmm5);
	paddd(xmm7, xmm5);
	mov(word[B-0x80], ax);
	sub(A1, -2);
	sub(B, -2);
	align(4);

L(lb14);
	test(M, 0x1);
	jle(lb38, T_NEAR);
	mov(al, byte[A1-0x80]);
	pinsrb(xmm0, eax, 0x0);
	pmovsxbd(xmm5, xmm0);
	paddd(xmm7, xmm5);
	mov(byte[B-0x80], al);
	sub(B, -1);
	align(4);

L(lb38);
	mov(A1, qword[ARG_BIAS]);
	movd(dword[A1], xmm7);
	add(qword[ARG_BIAS], 0x4);
	sub(N, 0x1);
	cmp(N, 0x1);
	jge(la14, T_NEAR);
	align(4);

L(lb58);

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
