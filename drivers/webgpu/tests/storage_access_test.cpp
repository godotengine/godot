/**************************************************************************/
/*  storage_access_test.cpp                                               */
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

#include "spirv-tools/libspirv.h"
#include "spirv_preprocess.h"

#include "drivers/webgpu/tests/check.h"

#include <cstdio>
#include <cstring>

static Vector<uint8_t> assemble(const char *p_source) {
	spv_context context = spvContextCreate(SPV_ENV_UNIVERSAL_1_3);
	spv_binary binary = nullptr;
	spv_diagnostic diagnostic = nullptr;
	const spv_result_t assembled = spvTextToBinaryWithOptions(context, p_source, std::strlen(p_source), SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS, &binary, &diagnostic);
	if (diagnostic) {
		std::fprintf(stderr, "%s\n", diagnostic->error);
	}
	CHECK(assembled == SPV_SUCCESS);
	Vector<uint8_t> bytes;
	bytes.resize(binary->wordCount * 4);
	std::memcpy(bytes.ptrw(), binary->code, bytes.size());
	spvBinaryDestroy(binary);
	spvDiagnosticDestroy(diagnostic);
	spvContextDestroy(context);
	return bytes;
}

// Word index of the first instruction with p_opcode (and p_operand as its first operand, if nonzero), or 0.
static size_t find_instruction(const Vector<uint8_t> &p_bytes, uint32_t p_opcode, uint32_t p_operand = 0) {
	const auto *words = reinterpret_cast<const uint32_t *>(p_bytes.ptr());
	for (size_t pos = 5; pos < size_t(p_bytes.size() / 4); pos += words[pos] >> 16) {
		if ((words[pos] & 0xffff) == p_opcode && (!p_operand || words[pos + 1] == p_operand)) {
			return pos;
		}
	}
	return 0;
}

static bool is_non_writable(const Vector<uint8_t> &p_bytes, uint32_t p_var) {
	const auto *words = reinterpret_cast<const uint32_t *>(p_bytes.ptr());
	for (size_t pos = 5; pos < size_t(p_bytes.size() / 4); pos += words[pos] >> 16) {
		if (words[pos] == ((3u << 16) | 71) && words[pos + 1] == p_var && words[pos + 2] == 24) {
			return true;
		}
	}
	return false;
}

int main() {
	// Both variable and member Volatile decorations must be removed; keep layout offsets.
	const uint32_t volatile_words[] = {
		0x07230203,
		0x00010300,
		0,
		101,
		0,
		(3u << 16) | 71,
		20,
		21,
		(4u << 16) | 72,
		10,
		0,
		21,
		(5u << 16) | 72,
		10,
		0,
		35,
		0,
	};
	Vector<uint8_t> volatile_bytes;
	volatile_bytes.resize(sizeof(volatile_words));
	std::memcpy(volatile_bytes.ptrw(), volatile_words, sizeof(volatile_words));
	const auto stripped = spirv_preprocess::strip_restrict_decoration(volatile_bytes);
	CHECK(stripped.size() == 10 * 4);
	CHECK(!find_instruction(stripped, 71));
	CHECK(find_instruction(stripped, 72, 10) == 5);

	// Storage-buffer access qualifiers change; image and unrelated members stay.
	const uint32_t words[] = {
		0x07230203,
		0x00010300,
		0,
		101,
		0,
		(3u << 16) | 71,
		20,
		25,
		(4u << 16) | 72,
		10,
		0,
		25,
		(3u << 16) | 71,
		30,
		25,
		(4u << 16) | 72,
		11,
		0,
		25,
		(3u << 16) | 71,
		20,
		24,
		(4u << 16) | 32,
		100,
		12,
		10,
		(4u << 16) | 59,
		100,
		20,
		12,
	};
	Vector<uint8_t> bytes;
	bytes.resize(sizeof(words));
	std::memcpy(bytes.ptrw(), words, sizeof(words));
	auto result = spirv_preprocess::normalize_storage_buffer_access(bytes);
	CHECK(result.size() == sizeof(words) - 7 * sizeof(uint32_t));
	const auto *output = reinterpret_cast<const uint32_t *>(result.ptr());
	CHECK(output[6] == 30 && output[7] == 25); // Image NonReadable retained.
	CHECK(output[9] == 11 && output[11] == 25); // Unrelated member retained.
	CHECK(output[13] == 20 && output[14] == 24); // NonWritable retained.
	auto repeated = spirv_preprocess::normalize_storage_buffer_access(result);
	CHECK(repeated.size() == result.size());
	CHECK(std::memcmp(repeated.ptr(), result.ptr(), result.size()) == 0);
	bytes.resize(21);
	CHECK(spirv_preprocess::normalize_storage_buffer_access(bytes).size() == 21);

	// Writes through copied, selected and phi pointers and float atomics keep buffers writable. %21 is
	// only read. The phi deliberately refers to a pointer defined after it.
	const Vector<uint8_t> writes = assemble(R"(
OpCapability Shader
OpCapability AtomicFloat32AddEXT
OpExtension "SPV_EXT_shader_atomic_float_add"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %1 "main"
OpExecutionMode %1 LocalSize 1 1 1
OpDecorate %2 Block
OpMemberDecorate %2 0 Offset 0
%3 = OpTypeVoid
%4 = OpTypeFunction %3
%5 = OpTypeFloat 32
%6 = OpTypeInt 32 0
%7 = OpTypeBool
%2 = OpTypeStruct %5
%8 = OpTypePointer StorageBuffer %2
%9 = OpTypePointer StorageBuffer %5
%21 = OpVariable %8 StorageBuffer
%22 = OpVariable %8 StorageBuffer
%23 = OpVariable %8 StorageBuffer
%24 = OpVariable %8 StorageBuffer
%25 = OpVariable %8 StorageBuffer
%26 = OpVariable %8 StorageBuffer
%10 = OpConstant %6 0
%11 = OpConstant %5 1
%12 = OpConstantTrue %7
%13 = OpConstant %6 1
%1 = OpFunction %3 None %4
%30 = OpLabel
%31 = OpAccessChain %9 %21 %10
%32 = OpLoad %5 %31
%33 = OpAccessChain %9 %22 %10
%34 = OpCopyObject %9 %33
OpStore %34 %32
%35 = OpAccessChain %9 %23 %10
%36 = OpAccessChain %9 %24 %10
%37 = OpSelect %9 %12 %35 %36
OpStore %37 %11
%38 = OpAccessChain %9 %25 %10
%39 = OpAtomicFAddEXT %5 %38 %13 %10 %11
OpBranch %40
%40 = OpLabel
%41 = OpPhi %9 %43 %42
OpStore %41 %11
OpBranch %42
%42 = OpLabel
%43 = OpAccessChain %9 %26 %10
OpReturn
OpFunctionEnd
)");
	const Vector<uint8_t> inferred = spirv_preprocess::infer_readonly_storage(writes);
	CHECK(is_non_writable(inferred, 21));
	for (uint32_t var = 22; var <= 26; var++) {
		CHECK(!is_non_writable(inferred, var));
	}

	// Without other annotations the decoration still goes before the type declarations.
	const Vector<uint8_t> bare = assemble(R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %1 "main"
OpExecutionMode %1 LocalSize 1 1 1
%3 = OpTypeVoid
%4 = OpTypeFunction %3
%5 = OpTypeFloat 32
%2 = OpTypeStruct %5
%8 = OpTypePointer StorageBuffer %2
%21 = OpVariable %8 StorageBuffer
%1 = OpFunction %3 None %4
%30 = OpLabel
OpReturn
OpFunctionEnd
)");
	const Vector<uint8_t> decorated = spirv_preprocess::infer_readonly_storage(bare);
	const size_t decoration = find_instruction(decorated, 71, 21);
	CHECK(decoration && decoration < find_instruction(decorated, 19));
	std::puts("PASS: storage access normalization and read-only inference");
}
