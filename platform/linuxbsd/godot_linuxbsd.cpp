/**************************************************************************/
/*  godot_linuxbsd.cpp                                                    */
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

#include "os_linuxbsd.h"

#include "core/profiling/profiling.h"
#include "main/main.h"

#include <unistd.h>
#include <climits>
#include <clocale>
#include <cstdlib>

#if defined(ASAN_ENABLED)
#include <sys/resource.h>
#endif // defined(ASAN_ENABLED)

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
#endif

// For export templates, add a section; the exporter will patch it to enclose
// the data appended to the executable (bundled PCK).
#if !defined(TOOLS_ENABLED) && defined(__GNUC__)
static const char dummy[8] __attribute__((section("pck"), used)) = { 0 };

// Dummy function to prevent LTO from discarding "pck" section.
extern "C" const char *pck_section_dummy_call() __attribute__((used));
extern "C" const char *pck_section_dummy_call() {
	return &dummy[0];
}
#endif

int main(int argc, char *argv[]) {
#if defined(__x86_64) || defined(__x86_64__)
	int cpuinfo[4];
	__cpuid(cpuinfo, 0x01);

	if (!(cpuinfo[2] & (1 << 20))) {
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
#endif

#if defined(ASAN_ENABLED)
	// Note: Set stack size to be at least 30 MB (vs 8 MB default) to avoid overflow, address sanitizer can increase stack usage up to 3 times.
	struct rlimit stack_lim = { 0x1E00000, 0x1E00000 };
	setrlimit(RLIMIT_STACK, &stack_lim);
#endif // defined(ASAN_ENABLED)

	godot_init_profiler();

	OS_LinuxBSD os;

	setlocale(LC_CTYPE, "");

	// We must override main when testing is enabled
	TEST_MAIN_OVERRIDE

	char *cwd = (char *)malloc(PATH_MAX);
	ERR_FAIL_NULL_V(cwd, ERR_OUT_OF_MEMORY);
	char *ret = getcwd(cwd, PATH_MAX);

	Error err = Main::setup(argv[0], argc - 1, &argv[1]);

	if (err != OK) {
		free(cwd);
		if (err == ERR_HELP) { // Returned by --help and --version, so success.
			return EXIT_SUCCESS;
		}
		return EXIT_FAILURE;
	}

	if (Main::start() == EXIT_SUCCESS) {
		os.run();
	} else {
		os.set_exit_code(EXIT_FAILURE);
	}
	Main::cleanup();

	if (ret) { // Previous getcwd was successful
		if (chdir(cwd) != 0) {
			ERR_PRINT("Couldn't return to previous working directory.");
		}
	}
	free(cwd);

	godot_cleanup_profiler();
	return os.get_exit_code();
}
