/**************************************************************************/
/*  fuzz_spirv_to_wgsl.cpp                                                */
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

// Fuzz the complete SPIR-V -> WGSL conversion pipeline.
//
// Exercises spirv_preprocess::run_pipeline() followed by Tint parsing,
// validation, and WGSL code generation.
//
// Tint can abort() on malformed SPIR-V (TINT_UNIMPLEMENTED, TINT_ICE).
// Fork isolation around the Tint call prevents one crash from killing the
// fuzzer.
// Crashes in the preprocessing passes (our code) are NOT fork-isolated
// and will correctly terminate the fuzzer as bugs.

#include "drivers/webgpu/spirv_preprocess.h"
#include "drivers/webgpu/tint_wrapper.h"

#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

static bool initialized = false;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
	if (!initialized) {
		tint_wrapper_initialize();
		initialized = true;
	}

	// Skip inputs that are obviously not SPIR-V.
	if (size < 20 || (size % 4) != 0) {
		return 0;
	}

	Vector<uint8_t> spv;
	spv.resize((int64_t)size);
	memcpy(spv.ptrw(), data, size);

	// The pipeline runs in-process (crashes here ARE bugs).
	spv = spirv_preprocess::run_pipeline(spv);

	// Tint conversion in a forked child — Tint may abort() on malformed SPIR-V.
	size_t word_count = (size_t)spv.size() / 4;
	const uint32_t *words = reinterpret_cast<const uint32_t *>(spv.ptr());

	pid_t pid = fork();
	if (pid == 0) {
		// Child: suppress stdout/stderr, try conversion, exit.
		int devnull = open("/dev/null", O_WRONLY);
		if (devnull >= 0) {
			dup2(devnull, STDOUT_FILENO);
			dup2(devnull, STDERR_FILENO);
			close(devnull);
		}
		char *error_msg = nullptr;
		char *wgsl = tint_wrapper_spirv_to_wgsl(words, word_count, &error_msg);
		free(wgsl);
		free(error_msg);
		_exit(0);
	} else if (pid > 0) {
		int status;
		waitpid(pid, &status, 0);
		// Parent continues regardless — Tint crashes are expected on random data.
	}
	// If fork() failed, skip Tint conversion (preprocessing was still tested).

	return 0;
}
