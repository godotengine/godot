/**************************************************************************/
/*  main.cpp                                                              */
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

// tint_convert_cli — host SPIR-V → WGSL converter for the compiler tests.
//
// Runs spirv_preprocess::run_pipeline() and Tint exactly as the WebGPU driver does at runtime.
//
// Usage: tint_convert_cli <file.spv>   (WGSL on stdout, errors on stderr)

#include "drivers/webgpu/spirv_preprocess.h"
#include "drivers/webgpu/tint_wrapper.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <vector>

int main(int argc, char *argv[]) {
	if (argc != 2) {
		std::fprintf(stderr, "Usage: tint_convert_cli <file.spv>\n");
		return 1;
	}
	std::ifstream file(argv[1], std::ios::binary);
	if (!file) {
		std::fprintf(stderr, "Error: Failed to read '%s'\n", argv[1]);
		return 1;
	}
	const std::vector<char> bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
	if (bytes.size() < 20 || bytes.size() % 4 != 0) {
		std::fprintf(stderr, "Error: Invalid SPIR-V: too small or not aligned to 4 bytes\n");
		return 1;
	}

	Vector<uint8_t> spv;
	spv.resize((int64_t)bytes.size());
	memcpy(spv.ptrw(), bytes.data(), bytes.size());
	spv = spirv_preprocess::run_pipeline(spv);

	tint_wrapper_initialize();
	char *error = nullptr;
	char *wgsl = tint_wrapper_spirv_to_wgsl(reinterpret_cast<const uint32_t *>(spv.ptr()), (size_t)spv.size() / 4, &error);
	if (!wgsl) {
		std::fprintf(stderr, "Error: %s\n", error ? error : "Tint conversion failed (unknown error)");
		std::free(error);
		return 1;
	}
	std::fputs(wgsl, stdout);
	std::free(wgsl);
	return 0;
}
