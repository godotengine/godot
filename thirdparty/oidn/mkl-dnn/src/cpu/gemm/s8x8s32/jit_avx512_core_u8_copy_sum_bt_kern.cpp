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

jit_avx512_core_u8_copy_sum_bt_kern::jit_avx512_core_u8_copy_sum_bt_kern(): jit_generator(nullptr, GEMM_CODE_SIZE)
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

Xbyak::Label l15c;
Xbyak::Label l1f4;
Xbyak::Label l20;
Xbyak::Label l248;
Xbyak::Label l280;
Xbyak::Label l2a4;
Xbyak::Label l2b0;
Xbyak::Label l2c8;
Xbyak::Label l384;
Xbyak::Label l3e8;
Xbyak::Label l40;
Xbyak::Label l424;
Xbyak::Label l448;
Xbyak::Label l468;
Xbyak::Label l474;
Xbyak::Label l48c;
Xbyak::Label l550;
Xbyak::Label l5bc;
Xbyak::Label l600;
Xbyak::Label l628;
Xbyak::Label l646;
Xbyak::Label l650;
Xbyak::Label l668;
Xbyak::Label l700;
Xbyak::Label l760;
Xbyak::Label l7a4;
Xbyak::Label l7c8;
Xbyak::Label l7e8;

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
	cmp(N, 0x8);
	jl(l2a4, T_NEAR);
	align(4);

L(l20);
	mov(A1, A);
	add(A, 0x8);
	pxor(xmm8, xmm8);
	pxor(xmm9, xmm9);
	mov(I, M);
	sar(I, 0x3);
	jle(l15c, T_NEAR);
	align(4);

L(l40);
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
	jg(l40, T_NEAR);
	align(4);

L(l15c);
	test(M, 0x4);
	jle(l1f4, T_NEAR);
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

L(l1f4);
	test(M, 0x2);
	jle(l248, T_NEAR);
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

L(l248);
	test(M, 0x1);
	jle(l280, T_NEAR);
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

L(l280);
	mov(A1, qword[ARG_BIAS]);
	movdqu(xword[A1], xmm8);
	movdqu(xword[A1+0x10], xmm9);
	add(qword[ARG_BIAS], 0x20);
	sub(N, 0x8);
	cmp(N, 0x8);
	jge(l20, T_NEAR);
	align(4);

L(l2a4);
	cmp(N, 0x4);
	jl(l468, T_NEAR);
	align(4);

L(l2b0);
	mov(A1, A);
	add(A, 0x4);
	pxor(xmm7, xmm7);
	mov(I, M);
	sar(I, 0x3);
	jle(l384, T_NEAR);
	align(4);

L(l2c8);
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
	jg(l2c8, T_NEAR);
	align(4);

L(l384);
	test(M, 0x4);
	jle(l3e8, T_NEAR);
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

L(l3e8);
	test(M, 0x2);
	jle(l424, T_NEAR);
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

L(l424);
	test(M, 0x1);
	jle(l448, T_NEAR);
	movd(xmm0, dword[A1-0x80]);
	pmovsxbd(xmm5, xmm0);
	paddd(xmm7, xmm5);
	movd(dword[B-0x80], xmm0);
	sub(B, -4);
	align(4);

L(l448);
	mov(A1, qword[ARG_BIAS]);
	movdqu(xword[A1], xmm7);
	add(qword[ARG_BIAS], 0x10);
	sub(N, 0x4);
	cmp(N, 0x4);
	jge(l2b0, T_NEAR);
	align(4);

L(l468);
	cmp(N, 0x2);
	jl(l646, T_NEAR);
	align(4);

L(l474);
	mov(A1, A);
	add(A, 0x2);
	pxor(xmm7, xmm7);
	mov(LDA3, M);
	sar(LDA3, 0x3);
	jle(l550, T_NEAR);
	align(4);

L(l48c);
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
	jg(l48c, T_NEAR);
	align(4);

L(l550);
	test(M, 0x4);
	jle(l5bc, T_NEAR);
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

L(l5bc);
	test(M, 0x2);
	jle(l600, T_NEAR);
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

L(l600);
	test(M, 0x1);
	jle(l628, T_NEAR);
	mov(ax, word[A1-0x80]);
	pinsrw(xmm0, eax, 0x0);
	pmovsxbd(xmm5, xmm0);
	paddd(xmm7, xmm5);
	mov(word[B-0x80], ax);
	sub(B, -2);
	align(4);

L(l628);
	mov(A1, qword[ARG_BIAS]);
	movq(qword[A1], xmm7);
	add(qword[ARG_BIAS], 0x8);
	sub(N, 0x2);
	cmp(N, 0x2);
	jge(l474, T_NEAR);
	align(4);

L(l646);
	cmp(N, 0x1);
	jl(l7e8, T_NEAR);
	align(4);

L(l650);
	mov(A1, A);
	add(A, 0x1);
	pxor(xmm7, xmm7);
	mov(LDA3, M);
	sar(LDA3, 0x3);
	jle(l700, T_NEAR);
	align(4);

L(l668);
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
	jg(l668, T_NEAR);
	align(4);

L(l700);
	test(M, 0x4);
	jle(l760, T_NEAR);
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

L(l760);
	test(M, 0x2);
	jle(l7a4, T_NEAR);
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

L(l7a4);
	test(M, 0x1);
	jle(l7c8, T_NEAR);
	mov(al, byte[A1-0x80]);
	pinsrw(xmm0, eax, 0x0);
	pmovsxbd(xmm5, xmm0);
	paddd(xmm7, xmm5);
	mov(byte[B-0x80], al);
	sub(B, -1);
	align(4);

L(l7c8);
	mov(A1, qword[ARG_BIAS]);
	movd(dword[A1], xmm7);
	add(qword[ARG_BIAS], 0x4);
	sub(N, 0x1);
	cmp(N, 0x1);
	jge(l650, T_NEAR);
	align(4);

L(l7e8);

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
