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

void __cpuid(int *r_cpuinfo, int p_info) {
	// Note: Some compilers have a buggy `__cpuid` intrinsic, using inline assembly (based on LLVM-20 implementation) instead.
	__asm__ __volatile__(
			"xchgq %%rbx, %q1;"
			"cpuid;"
			"xchgq %%rbx, %q1;"
			: "=a"(r_cpuinfo[0]), "=r"(r_cpuinfo[1]), "=c"(r_cpuinfo[2]), "=d"(r_cpuinfo[3])
			: "0"(p_info));
}

__attribute__((constructor)) void cpu_validation_shim() {
	bool cpuid_supported = false;

	int cpuinfo[4];
	__cpuid(cpuinfo, 0x01);
	cpuid_supported = (cpuinfo[2] & (1 << 20)) && (cpuinfo[2] & (1 << 23)); // SSE4.2 + POPCNT

	if (!cpuid_supported) {
		printf("A CPU with SSE4.2 instruction set support is required.\n");
		int ret = system("zenity --warning --title \"Godot Engine\" --text \"A CPU with SSE4.2 instruction set support is required.\" 2> /dev/null");
		if (ret != 0) {
			ret = system("kdialog --title \"Godot Engine\" --sorry \"A CPU with SSE4.2 instruction set support is required.\" 2> /dev/null");
		}
		if (ret != 0) {
			ret = system("Xdialog --title \"Godot Engine\" --msgbox \"A CPU with SSE4.2 instruction set support is required.\" 0 0 2> /dev/null");
		}
		if (ret != 0) {
			ret = system("xmessage -center -title \"Godot Engine\" \"A CPU with SSE4.2 instruction set support is required.\" 2> /dev/null");
		}
		abort();
	}
}
