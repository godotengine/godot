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

jit_avx512_core_u8_copy_bn_kern::jit_avx512_core_u8_copy_bn_kern(): jit_generator(nullptr, GEMM_CODE_SIZE)
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

#endif

inLocalLabel();
{

Xbyak::Label l118;
Xbyak::Label l1a8;
Xbyak::Label l20;
Xbyak::Label l218;
Xbyak::Label l28c;
Xbyak::Label l2f8;
Xbyak::Label l308;
Xbyak::Label l314;
Xbyak::Label l32c;
Xbyak::Label l3a0;
Xbyak::Label l3c;
Xbyak::Label l3f0;
Xbyak::Label l434;
Xbyak::Label l47c;
Xbyak::Label l4bc;
Xbyak::Label l4cc;
Xbyak::Label l4d8;
Xbyak::Label l4f0;
Xbyak::Label l528;
Xbyak::Label l554;
Xbyak::Label l580;
Xbyak::Label l5b0;
Xbyak::Label l5d0;
Xbyak::Label l5de;
Xbyak::Label l5e8;
Xbyak::Label l5f8;
Xbyak::Label l614;
Xbyak::Label l634;
Xbyak::Label l654;
Xbyak::Label l670;
Xbyak::Label l688;
Xbyak::Label l698;

	preamble();
#ifdef _WIN32
	auto stacksize = get_size_of_abi_save_regs();
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
	jl(l308, T_NEAR);
	align(4);

L(l20);
	mov(A1, A);
	lea(A2, ptr[A1+LDA*4]);
	lea(I, ptr[A1+LDA*8]);
	mov(A, I);
	mov(I, M);
	sar(I, 0x4);
	jle(l118, T_NEAR);
	align(4);

L(l3c);
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
	movdqu(xword[B-0x80], xmm0);
	movdqu(xword[B-0x60], xmm1);
	movdqu(xword[B-0x40], xmm4);
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
	movdqu(xword[B-0x70], xmm0);
	movdqu(xword[B-0x50], xmm1);
	movdqu(xword[B-0x30], xmm4);
	movdqu(xword[B-0x10], xmm3);
	sub(B, -128);
	dec(I);
	jg(l3c, T_NEAR);
	align(4);

L(l118);
	test(M, 0x8);
	jle(l1a8, T_NEAR);
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
	movdqu(xword[B-0x80], xmm0);
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
	movdqu(xword[B-0x70], xmm0);
	movdqu(xword[B-0x50], xmm1);
	sub(B, -64);
	align(4);

L(l1a8);
	test(M, 0x4);
	jle(l218, T_NEAR);
	movd(xmm0, dword[A1-0x80]);
	movd(xmm1, dword[A1+LDA*1-0x80]);
	movd(xmm2, dword[A1+LDA*2-0x80]);
	movd(xmm3, dword[A1+LDA3*1-0x80]);
	sub(A1, -4);
	punpckldq(xmm0, xmm1);
	punpckldq(xmm2, xmm3);
	punpcklqdq(xmm0, xmm2);
	movdqu(xword[B-0x80], xmm0);
	movd(xmm0, dword[A2-0x80]);
	movd(xmm1, dword[A2+LDA*1-0x80]);
	movd(xmm2, dword[A2+LDA*2-0x80]);
	movd(xmm3, dword[A2+LDA3*1-0x80]);
	sub(A2, -4);
	punpckldq(xmm0, xmm1);
	punpckldq(xmm2, xmm3);
	punpcklqdq(xmm0, xmm2);
	movdqu(xword[B-0x70], xmm0);
	sub(B, -32);
	align(4);

L(l218);
	test(M, 0x2);
	jle(l28c, T_NEAR);
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
	movdqu(xword[B-0x80], xmm0);
	sub(B, -16);
	align(4);

L(l28c);
	test(M, 0x1);
	jle(l2f8, T_NEAR);
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
	movq(qword[B-0x80], xmm0);
	sub(B, -8);
	align(4);

L(l2f8);
	sub(N, 0x8);
	cmp(N, 0x8);
	jge(l20, T_NEAR);
	align(4);

L(l308);
	cmp(N, 0x4);
	jl(l4cc, T_NEAR);
	align(4);

L(l314);
	mov(A1, A);
	lea(A2, ptr[A1+LDA*2]);
	lea(I, ptr[A1+LDA*4]);
	mov(A, I);
	mov(I, M);
	sar(I, 0x4);
	jle(l3a0, T_NEAR);
	align(4);

L(l32c);
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
	movdqu(xword[B-0x80], xmm0);
	movdqu(xword[B-0x70], xmm1);
	movdqu(xword[B-0x60], xmm4);
	movdqu(xword[B-0x50], xmm3);
	sub(B, -64);
	dec(I);
	jg(l32c, T_NEAR);
	align(4);

L(l3a0);
	test(M, 0x8);
	jle(l3f0, T_NEAR);
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
	movdqu(xword[B-0x80], xmm0);
	movdqu(xword[B-0x70], xmm1);
	sub(B, -32);
	align(4);

L(l3f0);
	test(M, 0x4);
	jle(l434, T_NEAR);
	movd(xmm0, dword[A1-0x80]);
	movd(xmm1, dword[A1+LDA*1-0x80]);
	sub(A1, -4);
	movd(xmm2, dword[A2-0x80]);
	movd(xmm3, dword[A2+LDA*1-0x80]);
	sub(A2, -4);
	punpckldq(xmm0, xmm1);
	punpckldq(xmm2, xmm3);
	punpcklqdq(xmm0, xmm2);
	movdqu(xword[B-0x80], xmm0);
	sub(B, -16);
	align(4);

L(l434);
	test(M, 0x2);
	jle(l47c, T_NEAR);
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
	movq(qword[B-0x80], xmm0);
	sub(B, -8);
	align(4);

L(l47c);
	test(M, 0x1);
	jle(l4bc, T_NEAR);
	mov(al, byte[A1-0x80]);
	pinsrb(xmm0, eax, 0x0);
	mov(al, byte[A1+LDA*1-0x80]);
	pinsrb(xmm0, eax, 0x1);
	mov(al, byte[A2-0x80]);
	pinsrb(xmm0, eax, 0x2);
	mov(al, byte[A2+LDA*1-0x80]);
	pinsrb(xmm0, eax, 0x3);
	movd(dword[B-0x80], xmm0);
	sub(B, -4);
	align(4);

L(l4bc);
	sub(N, 0x4);
	cmp(N, 0x4);
	jge(l314, T_NEAR);
	align(4);

L(l4cc);
	cmp(N, 0x2);
	jl(l5de, T_NEAR);
	align(4);

L(l4d8);
	mov(A1, A);
	lea(A2, ptr[A1+LDA*1]);
	lea(I, ptr[A1+LDA*2]);
	mov(A, I);
	mov(I, M);
	sar(I, 0x4);
	jle(l528, T_NEAR);
	align(4);

L(l4f0);
	movdqu(xmm0, xword[A1-0x80]);
	sub(A1, -16);
	movdqu(xmm1, xword[A2-0x80]);
	sub(A2, -16);
	movdqa(xmm2, xmm0);
	punpckldq(xmm0, xmm1);
	punpckhdq(xmm2, xmm1);
	movdqu(xword[B-0x80], xmm0);
	movdqu(xword[B-0x70], xmm2);
	sub(B, -32);
	dec(I);
	jg(l4f0, T_NEAR);
	align(4);

L(l528);
	test(M, 0x8);
	jle(l554, T_NEAR);
	movq(xmm0, qword[A1-0x80]);
	sub(A1, -8);
	movq(xmm1, qword[A2-0x80]);
	sub(A2, -8);
	punpckldq(xmm0, xmm1);
	movdqu(xword[B-0x80], xmm0);
	sub(B, -16);
	align(4);

L(l554);
	test(M, 0x4);
	jle(l580, T_NEAR);
	movd(xmm0, dword[A1-0x80]);
	sub(A1, -4);
	movd(xmm1, dword[A2-0x80]);
	sub(A2, -4);
	punpckldq(xmm0, xmm1);
	movq(qword[B-0x80], xmm0);
	sub(B, -8);
	align(4);

L(l580);
	test(M, 0x2);
	jle(l5b0, T_NEAR);
	mov(ax, word[A1-0x80]);
	sub(A1, -2);
	pinsrw(xmm0, eax, 0x0);
	mov(ax, word[A2-0x80]);
	sub(A2, -2);
	pinsrw(xmm0, eax, 0x1);
	movd(dword[B-0x80], xmm0);
	sub(B, -4);
	align(4);

L(l5b0);
	test(M, 0x1);
	jle(l5d0, T_NEAR);
	mov(al, byte[A1-0x80]);
	mov(byte[B-0x80], al);
	mov(al, byte[A2-0x80]);
	mov(byte[B-0x7f], al);
	sub(B, -2);
	align(4);

L(l5d0);
	sub(N, 0x2);
	cmp(N, 0x2);
	jge(l4d8, T_NEAR);
	align(4);

L(l5de);
	cmp(N, 0x1);
	jl(l698, T_NEAR);
	align(4);

L(l5e8);
	mov(A1, A);
	add(A, LDA);
	mov(I, M);
	sar(I, 0x4);
	jle(l614, T_NEAR);
	align(4);

L(l5f8);
	movdqu(xmm0, xword[A1-0x80]);
	sub(A1, -16);
	movdqu(xword[B-0x80], xmm0);
	sub(B, -16);
	dec(I);
	jg(l5f8, T_NEAR);
	align(4);

L(l614);
	test(M, 0x8);
	jle(l634, T_NEAR);
	movq(xmm0, qword[A1-0x80]);
	sub(A1, -8);
	movq(qword[B-0x80], xmm0);
	sub(B, -8);
	align(4);

L(l634);
	test(M, 0x4);
	jle(l654, T_NEAR);
	movd(xmm0, dword[A1-0x80]);
	sub(A1, -4);
	movd(dword[B-0x80], xmm0);
	sub(B, -4);
	align(4);

L(l654);
	test(M, 0x2);
	jle(l670, T_NEAR);
	mov(ax, word[A1-0x80]);
	mov(word[B-0x80], ax);
	sub(A1, -2);
	sub(B, -2);
	align(4);

L(l670);
	test(M, 0x1);
	jle(l688, T_NEAR);
	mov(al, byte[A1-0x80]);
	mov(byte[B-0x80], al);
	sub(B, -1);
	align(4);

L(l688);
	sub(N, 0x1);
	cmp(N, 0x1);
	jge(l5e8, T_NEAR);
	align(4);

L(l698);

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
}

}
}
}
