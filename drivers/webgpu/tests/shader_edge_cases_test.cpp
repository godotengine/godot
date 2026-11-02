/**************************************************************************/
/*  shader_edge_cases_test.cpp                                            */
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

#include "shader_metadata_webgpu.h"
#include "spirv-tools/libspirv.h"
#include "spirv_preprocess.h"
#include "tint_wrapper.h"

#include "drivers/webgpu/tests/check.h"

#include <cstdlib>
#include <cstring>
#include <string>

static std::string translate(const char *source, bool success = true) {
	auto context = spvContextCreate(SPV_ENV_VULKAN_1_1);
	spv_binary binary = nullptr;
	spv_diagnostic diagnostic = nullptr;
	CHECK(spvTextToBinary(context, source, std::strlen(source), &binary, &diagnostic) == SPV_SUCCESS);
	const auto validated = spvValidateBinary(context, binary->code, binary->wordCount, &diagnostic);
	if (diagnostic) {
		std::fprintf(stderr, "%s\n", diagnostic->error);
	}
	CHECK(validated == SPV_SUCCESS);
	Vector<uint8_t> bytes;
	bytes.resize(binary->wordCount * 4);
	std::memcpy(bytes.ptrw(), binary->code, bytes.size());
	spvBinaryDestroy(binary);
	spvDiagnosticDestroy(diagnostic);
	spvContextDestroy(context);
	bytes = spirv_preprocess::run_pipeline(bytes);
	char *error = nullptr;
	char *wgsl = tint_wrapper_spirv_to_wgsl(reinterpret_cast<const uint32_t *>(bytes.ptr()), bytes.size() / 4, &error);
	if (success && error) {
		std::fprintf(stderr, "%s\n", error);
	}
	CHECK((wgsl != nullptr) == success);
	std::string result = success ? wgsl : (error ? error : "");
	std::free(wgsl);
	std::free(error);
	return result;
}

int main() {
	const char *point = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %coord %color
OpExecutionMode %main OriginUpperLeft
OpDecorate %coord BuiltIn PointCoord
OpDecorate %color Location 0
%void = OpTypeVoid
%float = OpTypeFloat 32
%v2 = OpTypeVector %float 2
%v4 = OpTypeVector %float 4
%pin = OpTypePointer Input %v2
%pout = OpTypePointer Output %v4
%coord = OpVariable %pin Input
%color = OpVariable %pout Output
%zero = OpConstant %float 0
%one = OpConstant %float 1
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%uv = OpLoad %v2 %coord
%x = OpCompositeExtract %float %uv 0
%y = OpCompositeExtract %float %uv 1
%out = OpCompositeConstruct %v4 %x %y %zero %one
OpStore %color %out
OpReturn
OpFunctionEnd
)";
	const char *loop = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %color
OpExecutionMode %main OriginUpperLeft
OpDecorate %color Location 0
%void = OpTypeVoid
%bool = OpTypeBool
%int = OpTypeInt 32 1
%float = OpTypeFloat 32
%v4 = OpTypeVector %float 4
%pout = OpTypePointer Output %v4
%color = OpVariable %pout Output
%zero = OpConstant %int 0
%one = OpConstant %int 1
%ten = OpConstant %int 10
%f1 = OpConstant %float 1
%white = OpConstantComposite %v4 %f1 %f1 %f1 %f1
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
OpBranch %loop
%loop = OpLabel
%i = OpPhi %int %zero %entry %next %continue
%cond = OpSLessThan %bool %i %ten
OpLoopMerge %merge %continue None
OpBranchConditional %cond %body %merge
%body = OpLabel
OpSelectionMerge %switch_merge None
OpSwitch %i %default 0 %case
%case = OpLabel
OpBranch %merge
%default = OpLabel
OpBranch %switch_merge
%switch_merge = OpLabel
OpBranch %continue
%continue = OpLabel
%next = OpIAdd %int %i %one
OpBranch %loop
%merge = OpLabel
OpStore %color %white
OpReturn
OpFunctionEnd
)";
	const char *depth = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %color
OpExecutionMode %main OriginUpperLeft
OpDecorate %color Location 0
OpDecorate %image DescriptorSet 0
OpDecorate %image Binding 0
OpDecorate %sampler DescriptorSet 0
OpDecorate %sampler Binding 1
OpDecorate %unused DescriptorSet 0
OpDecorate %unused Binding 2
OpDecorate %unused_volume DescriptorSet 0
OpDecorate %unused_volume Binding 3
%void = OpTypeVoid
%float = OpTypeFloat 32
%v2 = OpTypeVector %float 2
%v4 = OpTypeVector %float 4
%img = OpTypeImage %float 2D 1 0 0 1 Unknown
%samp = OpTypeSampler
%sampled = OpTypeSampledImage %img
%pimg = OpTypePointer UniformConstant %img
%volume = OpTypeImage %float 3D 0 0 0 1 Unknown
%pvolume = OpTypePointer UniformConstant %volume
%psamp = OpTypePointer UniformConstant %samp
%pout = OpTypePointer Output %v4
%image = OpVariable %pimg UniformConstant
%sampler = OpVariable %psamp UniformConstant
%unused = OpVariable %psamp UniformConstant
%unused_volume = OpVariable %pvolume UniformConstant
%color = OpVariable %pout Output
%half = OpConstant %float 0.5
%uv = OpConstantComposite %v2 %half %half
%helper_type = OpTypeFunction %v4 %pimg %psamp
%main_type = OpTypeFunction %void
%inner = OpFunction %v4 None %helper_type
%inner_image = OpFunctionParameter %pimg
%inner_sampler = OpFunctionParameter %psamp
%inner_label = OpLabel
%loaded_image = OpLoad %img %inner_image
%loaded_sampler = OpLoad %samp %inner_sampler
%combined = OpSampledImage %sampled %loaded_image %loaded_sampler
%pixel = OpImageSampleImplicitLod %v4 %combined %uv
OpReturnValue %pixel
OpFunctionEnd
%outer = OpFunction %v4 None %helper_type
%outer_image = OpFunctionParameter %pimg
%outer_sampler = OpFunctionParameter %psamp
%outer_label = OpLabel
%nested = OpFunctionCall %v4 %inner %outer_image %outer_sampler
OpReturnValue %nested
OpFunctionEnd
%main = OpFunction %void None %main_type
%main_label = OpLabel
%result = OpFunctionCall %v4 %outer %image %sampler
OpStore %color %result
OpReturn
OpFunctionEnd
)";
	const char *specialization = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %color
OpExecutionMode %main OriginUpperLeft
OpDecorate %value SpecId 0
OpDecorate %color Location 0
%void = OpTypeVoid
%float = OpTypeFloat 32
%v2 = OpTypeVector %float 2
%v4 = OpTypeVector %float 4
%pin = OpTypePointer Input %v2
%pout = OpTypePointer Output %v4
%uint = OpTypeInt 32 0
%value = OpSpecConstant %uint 7
%color = OpVariable %pout Output
%zero = OpConstant %float 0
%one = OpConstant %float 1
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%x = OpConvertUToF %float %value
%y = OpCopyObject %float %x
%out = OpCompositeConstruct %v4 %x %y %zero %one
OpStore %color %out
OpReturn
OpFunctionEnd
)";
	const char *unsupported = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %coord %color
OpExecutionMode %main OriginUpperLeft
OpDecorate %coord BuiltIn HelperInvocation
OpDecorate %color Location 0
%void = OpTypeVoid
%float = OpTypeFloat 32
%v2 = OpTypeVector %float 2
%v4 = OpTypeVector %float 4
%bool = OpTypeBool
%pin = OpTypePointer Input %bool
%pout = OpTypePointer Output %v4
%coord = OpVariable %pin Input
%color = OpVariable %pout Output
%zero = OpConstant %float 0
%one = OpConstant %float 1
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%helper = OpLoad %bool %coord
%x = OpSelect %float %helper %zero %one
%y = OpCopyObject %float %x
%out = OpCompositeConstruct %v4 %x %y %zero %one
OpStore %color %out
OpReturn
OpFunctionEnd
)";
	CHECK(translate(point).find("0.5") != std::string::npos);
	std::string point_component(point);
	auto at = point_component.find("%pin = OpTypePointer Input %v2");
	point_component.insert(at, "%uint = OpTypeInt 32 0\n%index = OpConstant %uint 0\n%pfloat = OpTypePointer Input %float\n");
	at = point_component.find("%x = OpCompositeExtract %float %uv 0");
	point_component.replace(at, std::strlen("%x = OpCompositeExtract %float %uv 0"),
			"%component = OpAccessChain %pfloat %coord %index\n%x = OpLoad %float %component");
	CHECK(!translate(point_component.c_str()).empty());
	std::string unused_point(point);
	at = unused_point.find("%uv = OpLoad %v2 %coord");
	auto end = unused_point.find("%out = OpCompositeConstruct", at);
	unused_point.replace(at, end - at, "%x = OpCopyObject %float %one\n%y = OpCopyObject %float %one\n");
	CHECK(!translate(unused_point.c_str()).empty());
	CHECK(!translate(loop).empty());
	auto depth_wgsl = translate(depth);
	CHECK(depth_wgsl.find("texture_depth_2d") != std::string::npos);
	CHECK(depth_wgsl.find("textureLoad") != std::string::npos);
	CHECK(depth_wgsl.find("textureSample") == std::string::npos);
	// A depth-copy source is declared as a non-depth GLSL texture2D.
	// The reserved renderer name must restore depth without changing other images.
	std::string hinted_depth(depth);
	auto at_depth = hinted_depth.find("OpDecorate");
	hinted_depth.insert(at_depth, "OpName %image \"webgpu_depth_source\"\n");
	at_depth = hinted_depth.find("OpTypeImage %float 2D 1");
	hinted_depth.replace(at_depth, std::strlen("OpTypeImage %float 2D 1"), "OpTypeImage %float 2D 0");
	auto hinted_wgsl = translate(hinted_depth.c_str());
	CHECK(hinted_wgsl.find("texture_depth_2d") != std::string::npos);
	CHECK(hinted_wgsl.find("textureSample") == std::string::npos);
	std::string nearest_depth(hinted_depth);
	at_depth = nearest_depth.find("OpName %image");
	nearest_depth.insert(at_depth, "OpName %sampler \"SAMPLER_NEAREST_CLAMP\"\n");
	auto nearest_wgsl = translate(nearest_depth.c_str());
	CHECK(nearest_wgsl.find("textureLoad") != std::string::npos);
	CHECK(nearest_wgsl.find("mix(") == std::string::npos);
	auto specialized = translate(specialization);
	webgpu_metadata::Shader metadata;
	CHECK(webgpu_metadata::parse(specialized.c_str(), metadata));
	CHECK(metadata.has_overrides && metadata.overrides.size() == 1 && metadata.overrides[0] == 0);
	CHECK(specialized.find("override") != std::string::npos);
	for (int i = 0; i < 2; i++) {
		CHECK(translate(unsupported, false).find("HelperInvocation") != std::string::npos);
		CHECK(!translate(point).empty());
	}
	std::puts("PASS: PointCoord, loop exits, depth loads, scalar overrides and compiler-error recovery");
}
