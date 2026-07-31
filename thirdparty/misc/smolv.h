// smol-v - public domain - https://github.com/aras-p/smol-v
// authored 2016-2024 by Aras Pranckevicius
// no warranty implied; use at your own risk
// See end of file for license information.
//
//
// ### OVERVIEW:
//
// SMOL-V encodes Vulkan/Khronos SPIR-V format programs into a form that is smaller, and is more
// compressible. Normally no changes to the programs are done; they decode
// into exactly same program as what was encoded. Optionally, debug information
// can be removed too.
//
// SPIR-V is a very verbose format, several times larger than same programs expressed in other
// shader formats (e.g. DX11 bytecode, GLSL, DX9 bytecode etc.). The SSA-form with ever increasing
// IDs is not very appreciated by regular data compressors either. SMOL-V does several things
// to improve this:
// - Many words, especially ones that most often have small values, are encoded using
//   "varint" scheme (1-5 bytes per word, with just one byte for values in 0..127 range).
//   See https://developers.google.com/protocol-buffers/docs/encoding
// - Some IDs used in the program are delta-encoded, relative to previously seen IDs (e.g. Result
//   IDs). Often instructions reference things that were computed just before, so this results in
//   small deltas. These values are also encoded using "varint" scheme.
// - Reordering instruction opcodes so that the most common ones are the smallest values, for smaller
//  varint encoding.
// - Encoding several instructions in a more compact form, e.g. the "typical <=4 component swizzle"
//  shape of a VectorShuffle instruction, or sequences of MemberDecorate instructions.
//
// A somewhat similar utility is spirv-remap from glslang, see
// https://github.com/KhronosGroup/glslang/blob/master/README-spirv-remap.txt
//
//
// ### USAGE:
//
// Add source/smolv.h and source/smolv.cpp to your C++ project build.
// Currently it might require C++11 or somesuch; I only tested with Visual Studio 2017/2019, Mac Xcode 11 and Gcc 5.4.
//
// smolv::Encode and smolv::Decode is the basic functionality.
//
// Other functions are for development/statistics purposes, to figure out frequencies and
// distributions of the instructions.
//
// There's a test + compression benchmarking suite in testing/testmain.cpp; using that needs adding
// other files under testing/external to the build too (3rd party code: glslang remapper, Zstd, LZ4).
//
//
// ### LIMITATIONS / TODO:
//
// - SPIR-V where the words got stored in big-endian layout is not supported yet.
// - The whole thing might not work on Big-Endian CPUs. It might, but I'm not 100% sure.
// - Not much prevention is done against malformed/corrupted inputs, TODO.
// - Out of memory cases are not handled. The code will either throw exception
//   or crash, depending on your compilation flags.

#pragma once

#include <stdint.h>
#include <vector>
#include <cstddef>

namespace smolv
{
	typedef std::vector<uint8_t> ByteArray;

	enum EncodeFlags
	{
		kEncodeFlagNone = 0,
		kEncodeFlagStripDebugInfo = (1<<0), // Strip all optional SPIR-V instructions (debug names etc.)
	};
	enum DecodeFlags
	{
		kDecodeFlagNone = 0,
		kDecodeFlagUse20160831AsZeroVersion = (1 << 0), // For "version zero" of SMOL-V encoding, use 2016 08 31 code path (this is what happens to be used by Unity 2017-2020)
	};

	// Preserve *some* OpName debug names.
	// Return true to preserve, false to strip.
	// This is really only used to implement a workaround for problems with some Vulkan drivers.
	typedef bool(*StripOpNameFilterFunc)(const char* name);

	// -------------------------------------------------------------------
	// Encoding / Decoding

	// Encode SPIR-V into SMOL-V.
	//
	// Resulting data is appended to outSmolv array (the array is not cleared).
	//
	// flags is bitset of EncodeFlags values.
	//
	// Returns false on malformed SPIR-V input; if that happens the output array might get
	// partial/broken SMOL-V program.
	bool Encode(const void* spirvData, size_t spirvSize, ByteArray& outSmolv, uint32_t flags = kEncodeFlagNone, StripOpNameFilterFunc stripFilter = 0);


	// Decode SMOL-V into SPIR-V.
	//
	// Resulting data is written into the passed buffer. Get required buffer space with
	// GetDecodeBufferSize; this is the size of decoded SPIR-V program.
	//
	// flags is bitset of DecodeFlags values.

	// Decoding does no memory allocations.
	//
	// Returns false on malformed input; if that happens the output buffer might be only partially
	// written to.
	bool Decode(const void* smolvData, size_t smolvSize, void* spirvOutputBuffer, size_t spirvOutputBufferSize, uint32_t flags = kDecodeFlagNone);


	// Given a SMOL-V program, get size of the decoded SPIR-V program.
	// This is the buffer size that Decode expects.
	//
	// Returns zero on malformed input (just checks the header, not the full input).
	size_t GetDecodedBufferSize(const void* smolvData, size_t smolvSize);


	// -------------------------------------------------------------------
	// Computing instruction statistics on SPIR-V/SMOL-V programs

	struct Stats;

	Stats* StatsCreate();
	void StatsDelete(Stats* s);

	bool StatsCalculate(Stats* stats, const void* spirvData, size_t spirvSize);
	bool StatsCalculateSmol(Stats* stats, const void* smolvData, size_t smolvSize);
	void StatsPrint(const Stats* stats);

} // namespace smolv


// ------------------------------------------------------------------------------
// This software is available under 2 licenses -- choose whichever you prefer.
// ------------------------------------------------------------------------------
// ALTERNATIVE A - MIT License
// Copyright (c) 2016-2024 Aras Pranckevicius
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
// of the Software, and to permit persons to whom the Software is furnished to do
// so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
// ------------------------------------------------------------------------------
// ALTERNATIVE B - Public Domain (www.unlicense.org)
// This is free and unencumbered software released into the public domain.
// Anyone is free to copy, modify, publish, use, compile, sell, or distribute this
// software, either in source code form or as a compiled binary, for any purpose,
// commercial or non-commercial, and by any means.
// In jurisdictions that recognize copyright laws, the author or authors of this
// software dedicate any and all copyright interest in the software to the public
// domain. We make this dedication for the benefit of the public at large and to
// the detriment of our heirs and successors. We intend this dedication to be an
// overt act of relinquishment in perpetuity of all present and future rights to
// this software under copyright law.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
// ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
// ------------------------------------------------------------------------------
