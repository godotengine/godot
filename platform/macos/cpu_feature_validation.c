/**************************************************************************/
/*  cpu_feature_validation.c                                              */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#if defined(__x86_64) || defined(__x86_64__)
void __cpuid(int *r_cpuinfo, int p_info) {
	// Note: Some compilers have a buggy `__cpuid` intrinsic, using inline assembly (based on LLVM-20 implementation) instead.
	__asm__ __volatile__(
			"xchgq %%rbx, %q1;"
			"cpuid;"
			"xchgq %%rbx, %q1;"
			: "=a"(r_cpuinfo[0]), "=r"(r_cpuinfo[1]), "=c"(r_cpuinfo[2]), "=d"(r_cpuinfo[3])
			: "0"(p_info));
}

void __cpuidex(int *r_cpuinfo, int p_info, int p_count) {
	// Note: Some compilers have a buggy `__cpuidex` intrinsic, using inline assembly (based on LLVM-20 implementation) instead.
	__asm__ __volatile__(
			"xchgq %%rbx, %q1;"
			"cpuid;"
			"xchgq %%rbx, %q1;"
			: "=a"(r_cpuinfo[0]), "=r"(r_cpuinfo[1]), "=c"(r_cpuinfo[2]), "=d"(r_cpuinfo[3])
			: "0"(p_info), "2"(p_count));
}

#ifndef _STR
#define _STR(m_x) #m_x
#define _MKSTR(m_x) _STR(m_x)
#endif

#define ARCH_EXT_ERROR(m_ext) \
	printf("A CPU with " _STR(m_ext) " instruction set support is required.\n"); \
	system("osascript -e \"display alert \\\"Godot Engine\\\" message \\\"A CPU with " _STR(m_ext) " instruction set support is required.\\\"\"");

__attribute__((constructor)) void cpu_validation_shim() {
	bool cpuid_supported = false;

#if defined(GODOT_SSE42)
	int cpuinfo[4];
	__cpuid(cpuinfo, 0x01);
	cpuid_supported = (cpuinfo[2] & (1 << 20)) && (cpuinfo[2] & (1 << 23)); // SSE4.2 + POPCNT
#elif defined(GODOT_AVX)
	int cpuinfo[4];
	__cpuid(cpuinfo, 0x01);
	cpuid_supported = cpuinfo[2] & (1 << 28); // AVX
#elif defined(GODOT_AVX2)
	int cpuinfo[4];
	__cpuid(cpuinfo, 0x01);
	cpuid_supported = (cpuinfo[2] & (1 << 12)) && (cpuinfo[2] & (1 << 29)); // FMA + F16C
	__cpuidex(cpuinfo, 0x07, 0x00);
	cpuid_supported = cpuid_supported && (cpuinfo[1] & (1 << 5)) && (cpuinfo[1] & (1 << 3)) && (cpuinfo[1] & (1 << 8)); // AVX2 + BMI + BMI2
#elif defined(GODOT_AVX512)
	int cpuinfo[4];
	__cpuid(cpuinfo, 0x01);
	cpuid_supported = (cpuinfo[2] & (1 << 12)) && (cpuinfo[2] & (1 << 29)); // FMA + F16C
	__cpuidex(cpuinfo, 0x07, 0x00);
	cpuid_supported = cpuid_supported && (cpuinfo[1] & (1 << 16)) && (cpuinfo[1] & (1 << 17)) && (cpuinfo[1] & (1 << 31)) && (cpuinfo[1] & (1 << 3)) && (cpuinfo[1] & (1 << 8)); // AVX512-F + AVX512-DQ + AVX512-VL + BMI + BMI2
#else
	cpuid_supported = true;
#endif

	if (!cpuid_supported) {
#if defined(GODOT_SSE42)
		ARCH_EXT_ERROR("SSE4.2");
#elif defined(GODOT_AVX)
		ARCH_EXT_ERROR("AVX");
#elif defined(GODOT_AVX2)
		ARCH_EXT_ERROR("AVX2");
#elif defined(GODOT_AVX512)
		ARCH_EXT_ERROR("AVX512");
#endif
		abort();
	}
}
#endif
