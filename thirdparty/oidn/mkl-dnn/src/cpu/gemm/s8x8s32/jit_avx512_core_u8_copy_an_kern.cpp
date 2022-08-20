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

jit_avx512_core_u8_copy_an_kern::jit_avx512_core_u8_copy_an_kern(): jit_generator(nullptr, GEMM_CODE_SIZE)
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

Xbyak::Label l170;
Xbyak::Label l1f0;
Xbyak::Label l20;
Xbyak::Label l224;
Xbyak::Label l234;
Xbyak::Label l240;
Xbyak::Label l254;
Xbyak::Label l32c;
Xbyak::Label l34;
Xbyak::Label l388;
Xbyak::Label l3b0;
Xbyak::Label l3c0;
Xbyak::Label l3cc;
Xbyak::Label l3dc;
Xbyak::Label l454;
Xbyak::Label l48c;
Xbyak::Label l4a8;
Xbyak::Label l4b8;
Xbyak::Label l4c4;
Xbyak::Label l4d8;
Xbyak::Label l570;
Xbyak::Label l5c4;
Xbyak::Label l5f0;
Xbyak::Label l60c;
Xbyak::Label l61c;
Xbyak::Label l628;
Xbyak::Label l638;
Xbyak::Label l6b0;
Xbyak::Label l6f4;
Xbyak::Label l720;
Xbyak::Label l73c;
Xbyak::Label l74c;
Xbyak::Label l758;
Xbyak::Label l76c;
Xbyak::Label l804;
Xbyak::Label l858;
Xbyak::Label l88c;
Xbyak::Label l8a4;
Xbyak::Label l8b2;
Xbyak::Label l8bc;
Xbyak::Label l8cc;
Xbyak::Label l944;
Xbyak::Label l98c;
Xbyak::Label l9b0;
Xbyak::Label l9c8;
Xbyak::Label l9d8;

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
	cmp(N, 0x30);
	jl(l234, T_NEAR);
	align(4);

L(l20);
	mov(A1, A);
	add(A, 0x30);
	mov(I, M);
	sar(I, 0x2);
	jle(l170, T_NEAR);
	align(4);

L(l34);
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
	movdqu(xword[B-0x80], xmm0);
	movdqu(xword[B-0x70], xmm1);
	movdqu(xword[B-0x60], xmm4);
	movdqu(xword[B-0x50], xmm2);
	movdqu(xmm0, xword[A1-0x70]);
	movdqu(xmm1, xword[A1+LDA*1-0x70]);
	movdqu(xmm2, xword[A1+LDA*2-0x70]);
	movdqu(xmm3, xword[A1+LDA3*1-0x70]);
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
	movdqu(xword[B-0x40], xmm0);
	movdqu(xword[B-0x30], xmm1);
	movdqu(xword[B-0x20], xmm4);
	movdqu(xword[B-0x10], xmm2);
	movdqu(xmm0, xword[A1-0x60]);
	movdqu(xmm1, xword[A1+LDA*1-0x60]);
	movdqu(xmm2, xword[A1+LDA*2-0x60]);
	movdqu(xmm3, xword[A1+LDA3*1-0x60]);
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
	movdqu(xword[B], xmm0);
	movdqu(xword[B+0x10], xmm1);
	movdqu(xword[B+0x20], xmm4);
	movdqu(xword[B+0x30], xmm2);
	sub(B, -192);
	dec(I);
	jg(l34, T_NEAR);
	align(4);

L(l170);
	test(M, 0x2);
	jle(l1f0, T_NEAR);
	movdqu(xmm0, xword[A1-0x80]);
	movdqu(xmm1, xword[A1-0x70]);
	movdqu(xmm2, xword[A1-0x60]);
	add(A1, LDA);
	movdqu(xmm3, xword[A1-0x80]);
	movdqu(xmm4, xword[A1-0x70]);
	movdqu(xmm5, xword[A1-0x60]);
	add(A1, LDA);
	movdqa(xmm6, xmm0);
	punpcklbw(xmm0, xmm3);
	punpckhbw(xmm6, xmm3);
	movdqu(xword[B-0x80], xmm0);
	movdqu(xword[B-0x70], xmm6);
	movdqa(xmm6, xmm1);
	punpcklbw(xmm1, xmm4);
	punpckhbw(xmm6, xmm4);
	movdqu(xword[B-0x60], xmm1);
	movdqu(xword[B-0x50], xmm6);
	movdqa(xmm6, xmm2);
	punpcklbw(xmm2, xmm5);
	punpckhbw(xmm6, xmm5);
	movdqu(xword[B-0x40], xmm2);
	movdqu(xword[B-0x30], xmm6);
	sub(B, -96);
	align(4);

L(l1f0);
	test(M, 0x1);
	jle(l224, T_NEAR);
	movdqu(xmm0, xword[A1-0x80]);
	movdqu(xmm1, xword[A1-0x70]);
	movdqu(xmm2, xword[A1-0x60]);
	add(A1, LDA);
	movdqu(xword[B-0x80], xmm0);
	movdqu(xword[B-0x70], xmm1);
	movdqu(xword[B-0x60], xmm2);
	sub(B, -48);
	align(4);

L(l224);
	sub(N, 0x30);
	cmp(N, 0x30);
	jge(l20, T_NEAR);
	align(4);

L(l234);
	cmp(N, 0x20);
	jl(l3c0, T_NEAR);
	align(4);

L(l240);
	mov(A1, A);
	add(A, 0x20);
	mov(I, M);
	sar(I, 0x2);
	jle(l32c, T_NEAR);
	align(4);

L(l254);
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
	movdqu(xword[B-0x80], xmm0);
	movdqu(xword[B-0x70], xmm1);
	movdqu(xword[B-0x60], xmm4);
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
	movdqu(xword[B-0x40], xmm0);
	movdqu(xword[B-0x30], xmm1);
	movdqu(xword[B-0x20], xmm4);
	movdqu(xword[B-0x10], xmm2);
	sub(B, -128);
	dec(I);
	jg(l254, T_NEAR);
	align(4);

L(l32c);
	test(M, 0x2);
	jle(l388, T_NEAR);
	movdqu(xmm0, xword[A1-0x80]);
	movdqu(xmm1, xword[A1-0x70]);
	add(A1, LDA);
	movdqu(xmm2, xword[A1-0x80]);
	movdqu(xmm3, xword[A1-0x70]);
	add(A1, LDA);
	movdqa(xmm4, xmm0);
	punpcklbw(xmm0, xmm2);
	punpckhbw(xmm4, xmm2);
	movdqu(xword[B-0x80], xmm0);
	movdqu(xword[B-0x70], xmm4);
	movdqa(xmm4, xmm1);
	punpcklbw(xmm1, xmm3);
	punpckhbw(xmm4, xmm3);
	movdqu(xword[B-0x60], xmm1);
	movdqu(xword[B-0x50], xmm4);
	sub(B, -64);
	align(4);

L(l388);
	test(M, 0x1);
	jle(l3b0, T_NEAR);
	movdqu(xmm0, xword[A1-0x80]);
	movdqu(xmm1, xword[A1-0x70]);
	add(A1, LDA);
	movdqu(xword[B-0x80], xmm0);
	movdqu(xword[B-0x70], xmm1);
	sub(B, -32);
	align(4);

L(l3b0);
	sub(N, 0x20);
	cmp(N, 0x20);
	jge(l240, T_NEAR);
	align(4);

L(l3c0);
	cmp(N, 0x10);
	jl(l4b8, T_NEAR);
	align(4);

L(l3cc);
	mov(A1, A);
	add(A, 0x10);
	mov(I, M);
	sar(I, 0x2);
	jle(l454, T_NEAR);
	align(4);

L(l3dc);
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
	movdqu(xword[B-0x80], xmm0);
	movdqu(xword[B-0x70], xmm3);
	movdqu(xword[B-0x60], xmm4);
	movdqu(xword[B-0x50], xmm2);
	sub(B, -64);
	dec(I);
	jg(l3dc, T_NEAR);
	align(4);

L(l454);
	test(M, 0x2);
	jle(l48c, T_NEAR);
	movdqu(xmm0, xword[A1-0x80]);
	add(A1, LDA);
	movdqu(xmm1, xword[A1-0x80]);
	add(A1, LDA);
	movdqa(xmm2, xmm0);
	punpcklbw(xmm0, xmm1);
	punpckhbw(xmm2, xmm1);
	movdqu(xword[B-0x80], xmm0);
	movdqu(xword[B-0x70], xmm2);
	sub(B, -32);
	align(4);

L(l48c);
	test(M, 0x1);
	jle(l4a8, T_NEAR);
	movdqu(xmm0, xword[A1-0x80]);
	add(A1, LDA);
	movdqu(xword[B-0x80], xmm0);
	sub(B, -16);
	align(4);

L(l4a8);
	sub(N, 0x10);
	cmp(N, 0x10);
	jge(l3cc, T_NEAR);
	align(4);

L(l4b8);
	cmp(N, 0x8);
	jl(l61c, T_NEAR);
	align(4);

L(l4c4);
	mov(A1, A);
	add(A, 0x8);
	mov(I, M);
	sar(I, 0x3);
	jle(l570, T_NEAR);
	align(4);

L(l4d8);
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
	jg(l4d8, T_NEAR);
	align(4);

L(l570);
	test(M, 0x4);
	jle(l5c4, T_NEAR);
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

L(l5c4);
	test(M, 0x2);
	jle(l5f0, T_NEAR);
	movq(xmm0, qword[A1-0x80]);
	add(A1, LDA);
	movq(xmm1, qword[A1-0x80]);
	add(A1, LDA);
	punpcklbw(xmm0, xmm1);
	movdqu(xword[B-0x80], xmm0);
	sub(B, -16);
	align(4);

L(l5f0);
	test(M, 0x1);
	jle(l60c, T_NEAR);
	movq(xmm0, qword[A1-0x80]);
	add(A1, LDA);
	movq(qword[B-0x80], xmm0);
	sub(B, -8);
	align(4);

L(l60c);
	sub(N, 0x8);
	cmp(N, 0x8);
	jge(l4c4, T_NEAR);
	align(4);

L(l61c);
	cmp(N, 0x4);
	jl(l74c, T_NEAR);
	align(4);

L(l628);
	mov(A1, A);
	add(A, 0x4);
	mov(I, M);
	sar(I, 0x3);
	jle(l6b0, T_NEAR);
	align(4);

L(l638);
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
	jg(l638, T_NEAR);
	align(4);

L(l6b0);
	test(M, 0x4);
	jle(l6f4, T_NEAR);
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

L(l6f4);
	test(M, 0x2);
	jle(l720, T_NEAR);
	movd(xmm0, dword[A1-0x80]);
	add(A1, LDA);
	movd(xmm1, dword[A1-0x80]);
	add(A1, LDA);
	punpcklbw(xmm0, xmm1);
	movq(qword[B-0x80], xmm0);
	sub(B, -8);
	align(4);

L(l720);
	test(M, 0x1);
	jle(l73c, T_NEAR);
	movd(xmm0, dword[A1-0x80]);
	movd(dword[B-0x80], xmm0);
	sub(B, -4);
	align(4);

L(l73c);
	sub(N, 0x4);
	cmp(N, 0x4);
	jge(l628, T_NEAR);
	align(4);

L(l74c);
	cmp(N, 0x2);
	jl(l8b2, T_NEAR);
	align(4);

L(l758);
	mov(A1, A);
	add(A, 0x2);
	mov(LDA3, M);
	sar(LDA3, 0x3);
	jle(l804, T_NEAR);
	align(4);

L(l76c);
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
	jg(l76c, T_NEAR);
	align(4);

L(l804);
	test(M, 0x4);
	jle(l858, T_NEAR);
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

L(l858);
	test(M, 0x2);
	jle(l88c, T_NEAR);
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

L(l88c);
	test(M, 0x1);
	jle(l8a4, T_NEAR);
	mov(ax, word[A1-0x80]);
	mov(word[B-0x80], ax);
	sub(B, -2);
	align(4);

L(l8a4);
	sub(N, 0x2);
	cmp(N, 0x2);
	jge(l758, T_NEAR);
	align(4);

L(l8b2);
	cmp(N, 0x1);
	jl(l9d8, T_NEAR);
	align(4);

L(l8bc);
	mov(A1, A);
	add(A, 0x1);
	mov(LDA3, M);
	sar(LDA3, 0x3);
	jle(l944, T_NEAR);
	align(4);

L(l8cc);
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
	jg(l8cc, T_NEAR);
	align(4);

L(l944);
	test(M, 0x4);
	jle(l98c, T_NEAR);
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

L(l98c);
	test(M, 0x2);
	jle(l9b0, T_NEAR);
	mov(al, byte[A1-0x80]);
	add(A1, LDA);
	mov(byte[B-0x80], al);
	mov(al, byte[A1-0x80]);
	add(A1, LDA);
	mov(byte[B-0x7f], al);
	sub(B, -2);
	align(4);

L(l9b0);
	test(M, 0x1);
	jle(l9c8, T_NEAR);
	mov(al, byte[A1-0x80]);
	mov(byte[B-0x80], al);
	sub(B, -1);
	align(4);

L(l9c8);
	sub(N, 0x1);
	cmp(N, 0x1);
	jge(l8bc, T_NEAR);
	align(4);

L(l9d8);

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
