/**************************************************************************/
/*  binding_array_literals_test.cpp                                       */
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

int main() {
	// Array type ID 112 deliberately collides with offsets, bindings and constants.
	const char *source = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %color
OpExecutionMode %main OriginUpperLeft
OpDecorate %color Location 0
OpDecorate %resources DescriptorSet 0
OpDecorate %resources Binding 112
OpDecorate %texture DescriptorSet 0
OpDecorate %texture Binding 5
OpDecorate %block Block
OpMemberDecorate %block 0 Offset 0
OpMemberDecorate %block 1 Offset 112
OpDecorate %data DescriptorSet 0
OpDecorate %data Binding 3
%void = OpTypeVoid
%float = OpTypeFloat 32
%uint = OpTypeInt 32 0
%v4 = OpTypeVector %float 4
%v2 = OpTypeVector %float 2
%half = OpConstant %float 0.5
%uv = OpConstantComposite %v2 %half %half
%image = OpTypeImage %float 2D 0 0 0 1 Unknown
%pimage = OpTypePointer UniformConstant %image
%texture = OpVariable %pimage UniformConstant
%sampled_type = OpTypeSampledImage %image
%zero = OpConstant %uint 0
%one = OpConstant %uint 1
%two = OpConstant %uint 2
%literal = OpConstant %uint 112
%sampler = OpTypeSampler
%112 = OpTypeArray %sampler %two
%parray = OpTypePointer UniformConstant %112
%psampler = OpTypePointer UniformConstant %sampler
%resources = OpVariable %parray UniformConstant
%block = OpTypeStruct %v4 %v4
%pblock = OpTypePointer Uniform %block
%pvec = OpTypePointer Uniform %v4
%data = OpVariable %pblock Uniform
%pout = OpTypePointer Output %v4
%color = OpVariable %pout Output
%main_type = OpTypeFunction %void
%main = OpFunction %void None %main_type
%label = OpLabel
%element = OpAccessChain %psampler %resources %zero
%loaded = OpLoad %sampler %element
%field = OpAccessChain %pvec %data %one
%value = OpLoad %v4 %field
%image_value = OpLoad %image %texture
%combined = OpSampledImage %sampled_type %image_value %loaded
%pixel = OpImageSampleImplicitLod %v4 %combined %uv
%weighted = OpFMul %v4 %pixel %value
%literal_float = OpConvertUToF %float %literal
%result = OpVectorTimesScalar %v4 %weighted %literal_float
OpStore %color %result
OpReturn
OpFunctionEnd
)";
	auto context = spvContextCreate(SPV_ENV_UNIVERSAL_1_3);
	spv_binary binary = nullptr;
	spv_diagnostic diagnostic = nullptr;
	const spv_result_t assembled = spvTextToBinaryWithOptions(context, source, std::strlen(source), SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS, &binary, &diagnostic);
	CHECK(assembled == SPV_SUCCESS);
	Vector<uint8_t> bytes;
	bytes.resize(binary->wordCount * 4);
	std::memcpy(bytes.ptrw(), binary->code, bytes.size());
	auto result = spirv_preprocess::flatten_binding_arrays(bytes);
	const auto *words = reinterpret_cast<const uint32_t *>(result.ptr());
	bool offset = false, binding = false, constant = false;
	for (size_t pos = 5; pos < size_t(result.size() / 4); pos += words[pos] >> 16) {
		uint32_t opcode = words[pos] & 0xffff;
		if (opcode == 72 && words[pos + 3] == 35 && words[pos + 2] == 1) {
			CHECK(words[pos + 4] == 112);
			offset = true;
		}
		if (opcode == 71 && words[pos + 2] == 33 && words[pos + 3] == 112) {
			binding = true;
		}
		if (opcode == 43 && words[pos + 3] == 112) {
			constant = true;
		}
		CHECK(opcode != 28); // Handle array type removed.
	}
	CHECK(offset && binding && constant);
	const spv_result_t validated = spvValidateBinary(context, words, result.size() / 4, &diagnostic);
	CHECK(validated == SPV_SUCCESS);
	spvBinaryDestroy(binary);
	spvDiagnosticDestroy(diagnostic);
	spvContextDestroy(context);
	std::puts("PASS: handle array flattening preserves colliding literal offsets, bindings and constants");
}
