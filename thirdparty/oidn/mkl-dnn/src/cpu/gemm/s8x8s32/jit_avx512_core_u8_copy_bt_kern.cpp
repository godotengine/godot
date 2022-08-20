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

jit_avx512_core_u8_copy_bt_kern::jit_avx512_core_u8_copy_bt_kern(): jit_generator(nullptr, GEMM_CODE_SIZE)
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

Xbyak::Label l120;
Xbyak::Label l14c;
Xbyak::Label l168;
Xbyak::Label l178;
Xbyak::Label l184;
Xbyak::Label l194;
Xbyak::Label l20;
Xbyak::Label l20c;
Xbyak::Label l250;
Xbyak::Label l27c;
Xbyak::Label l298;
Xbyak::Label l2a8;
Xbyak::Label l2b4;
Xbyak::Label l2c8;
Xbyak::Label l34;
Xbyak::Label l360;
Xbyak::Label l3b4;
Xbyak::Label l3e8;
Xbyak::Label l400;
Xbyak::Label l40e;
Xbyak::Label l418;
Xbyak::Label l428;
Xbyak::Label l4a0;
Xbyak::Label l4e8;
Xbyak::Label l50c;
Xbyak::Label l524;
Xbyak::Label l534;
Xbyak::Label lcc;

	preamble();
#ifdef _WIN32
	auto stacksize = get_size_of_abi_save_regs();
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
	jl(l178, T_NEAR);
	align(4);

L(l20);
	mov(A1, A);
	add(A, 0x8);
	mov(I, M);
	sar(I, 0x3);
	jle(lcc, T_NEAR);
	align(4);

L(l34);
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
	movdqu(xword[B-0x60], xmm0);
	movdqu(xword[B-0x50], xmm1);
	sub(B, -64);
	dec(I);
	jg(l34, T_NEAR);
	align(4);

L(lcc);
	test(M, 0x4);
	jle(l120, T_NEAR);
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
	movdqu(xword[B-0x80], xmm0);
	movdqu(xword[B-0x70], xmm1);
	sub(B, -32);
	align(4);

L(l120);
	test(M, 0x2);
	jle(l14c, T_NEAR);
	movq(xmm0, qword[A1-0x80]);
	add(A1, LDA);
	movq(xmm1, qword[A1-0x80]);
	add(A1, LDA);
	punpcklbw(xmm0, xmm1);
	movdqu(xword[B-0x80], xmm0);
	sub(B, -16);
	align(4);

L(l14c);
	test(M, 0x1);
	jle(l168, T_NEAR);
	movq(xmm0, qword[A1-0x80]);
	add(A1, LDA);
	movq(qword[B-0x80], xmm0);
	sub(B, -8);
	align(4);

L(l168);
	sub(N, 0x8);
	cmp(N, 0x8);
	jge(l20, T_NEAR);
	align(4);

L(l178);
	cmp(N, 0x4);
	jl(l2a8, T_NEAR);
	align(4);

L(l184);
	mov(A1, A);
	add(A, 0x4);
	mov(I, M);
	sar(I, 0x3);
	jle(l20c, T_NEAR);
	align(4);

L(l194);
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
	movdqu(xword[B-0x70], xmm0);
	sub(B, -32);
	dec(I);
	jg(l194, T_NEAR);
	align(4);

L(l20c);
	test(M, 0x4);
	jle(l250, T_NEAR);
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
	movdqu(xword[B-0x80], xmm0);
	sub(B, -16);
	align(4);

L(l250);
	test(M, 0x2);
	jle(l27c, T_NEAR);
	movd(xmm0, dword[A1-0x80]);
	add(A1, LDA);
	movd(xmm1, dword[A1-0x80]);
	add(A1, LDA);
	punpcklbw(xmm0, xmm1);
	movq(qword[B-0x80], xmm0);
	sub(B, -8);
	align(4);

L(l27c);
	test(M, 0x1);
	jle(l298, T_NEAR);
	movd(xmm0, dword[A1-0x80]);
	movd(dword[B-0x80], xmm0);
	sub(B, -4);
	align(4);

L(l298);
	sub(N, 0x4);
	cmp(N, 0x4);
	jge(l184, T_NEAR);
	align(4);

L(l2a8);
	cmp(N, 0x2);
	jl(l40e, T_NEAR);
	align(4);

L(l2b4);
	mov(A1, A);
	add(A, 0x2);
	mov(LDA3, M);
	sar(LDA3, 0x3);
	jle(l360, T_NEAR);
	align(4);

L(l2c8);
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
	movdqu(xword[B-0x80], xmm0);
	sub(B, -16);
	dec(LDA3);
	jg(l2c8, T_NEAR);
	align(4);

L(l360);
	test(M, 0x4);
	jle(l3b4, T_NEAR);
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
	movq(qword[B-0x80], xmm0);
	sub(B, -8);
	align(4);

L(l3b4);
	test(M, 0x2);
	jle(l3e8, T_NEAR);
	mov(ax, word[A1-0x80]);
	add(A1, LDA);
	pinsrw(xmm0, eax, 0x0);
	mov(ax, word[A1-0x80]);
	add(A1, LDA);
	pinsrw(xmm1, eax, 0x0);
	punpcklbw(xmm0, xmm1);
	movd(dword[B-0x80], xmm0);
	sub(B, -4);
	align(4);

L(l3e8);
	test(M, 0x1);
	jle(l400, T_NEAR);
	mov(ax, word[A1-0x80]);
	mov(word[B-0x80], ax);
	sub(B, -2);
	align(4);

L(l400);
	sub(N, 0x2);
	cmp(N, 0x2);
	jge(l2b4, T_NEAR);
	align(4);

L(l40e);
	cmp(N, 0x1);
	jl(l534, T_NEAR);
	align(4);

L(l418);
	mov(A1, A);
	add(A, 0x1);
	mov(LDA3, M);
	sar(LDA3, 0x3);
	jle(l4a0, T_NEAR);
	align(4);

L(l428);
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
	movq(qword[B-0x80], xmm0);
	sub(B, -8);
	dec(LDA3);
	jg(l428, T_NEAR);
	align(4);

L(l4a0);
	test(M, 0x4);
	jle(l4e8, T_NEAR);
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
	movd(dword[B-0x80], xmm0);
	sub(B, -4);
	align(4);

L(l4e8);
	test(M, 0x2);
	jle(l50c, T_NEAR);
	mov(al, byte[A1-0x80]);
	add(A1, LDA);
	mov(byte[B-0x80], al);
	mov(al, byte[A1-0x80]);
	add(A1, LDA);
	mov(byte[B-0x7f], al);
	sub(B, -2);
	align(4);

L(l50c);
	test(M, 0x1);
	jle(l524, T_NEAR);
	mov(al, byte[A1-0x80]);
	mov(byte[B-0x80], al);
	sub(B, -1);
	align(4);

L(l524);
	sub(N, 0x1);
	cmp(N, 0x1);
	jge(l418, T_NEAR);
	align(4);

L(l534);

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
