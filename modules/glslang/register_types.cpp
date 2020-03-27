/*************************************************************************/
/*  register_types.cpp                                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#include "register_types.h"

#include "servers/rendering/rendering_device.h"

#include <SPIRV/GlslangToSpv.h>
#include <glslang/Include/Types.h>
#include <glslang/Public/ShaderLang.h>

static const TBuiltInResource default_builtin_resource = {
	/*maxLights*/ 32,
	/*maxClipPlanes*/ 6,
	/*maxTextureUnits*/ 32,
	/*maxTextureCoords*/ 32,
	/*maxVertexAttribs*/ 64,
	/*maxVertexUniformComponents*/ 4096,
	/*maxVaryingFloats*/ 64,
	/*maxVertexTextureImageUnits*/ 32,
	/*maxCombinedTextureImageUnits*/ 80,
	/*maxTextureImageUnits*/ 32,
	/*maxFragmentUniformComponents*/ 4096,
	/*maxDrawBuffers*/ 32,
	/*maxVertexUniformVectors*/ 128,
	/*maxVaryingVectors*/ 8,
	/*maxFragmentUniformVectors*/ 16,
	/*maxVertexOutputVectors*/ 16,
	/*maxFragmentInputVectors*/ 15,
	/*minProgramTexelOffset*/ -8,
	/*maxProgramTexelOffset*/ 7,
	/*maxClipDistances*/ 8,
	/*maxComputeWorkGroupCountX*/ 65535,
	/*maxComputeWorkGroupCountY*/ 65535,
	/*maxComputeWorkGroupCountZ*/ 65535,
	/*maxComputeWorkGroupSizeX*/ 1024,
	/*maxComputeWorkGroupSizeY*/ 1024,
	/*maxComputeWorkGroupSizeZ*/ 64,
	/*maxComputeUniformComponents*/ 1024,
	/*maxComputeTextureImageUnits*/ 16,
	/*maxComputeImageUniforms*/ 8,
	/*maxComputeAtomicCounters*/ 8,
	/*maxComputeAtomicCounterBuffers*/ 1,
	/*maxVaryingComponents*/ 60,
	/*maxVertexOutputComponents*/ 64,
	/*maxGeometryInputComponents*/ 64,
	/*maxGeometryOutputComponents*/ 128,
	/*maxFragmentInputComponents*/ 128,
	/*maxImageUnits*/ 8,
	/*maxCombinedImageUnitsAndFragmentOutputs*/ 8,
	/*maxCombinedShaderOutputResources*/ 8,
	/*maxImageSamples*/ 0,
	/*maxVertexImageUniforms*/ 0,
	/*maxTessControlImageUniforms*/ 0,
	/*maxTessEvaluationImageUniforms*/ 0,
	/*maxGeometryImageUniforms*/ 0,
	/*maxFragmentImageUniforms*/ 8,
	/*maxCombinedImageUniforms*/ 8,
	/*maxGeometryTextureImageUnits*/ 16,
	/*maxGeometryOutputVertices*/ 256,
	/*maxGeometryTotalOutputComponents*/ 1024,
	/*maxGeometryUniformComponents*/ 1024,
	/*maxGeometryVaryingComponents*/ 64,
	/*maxTessControlInputComponents*/ 128,
	/*maxTessControlOutputComponents*/ 128,
	/*maxTessControlTextureImageUnits*/ 16,
	/*maxTessControlUniformComponents*/ 1024,
	/*maxTessControlTotalOutputComponents*/ 4096,
	/*maxTessEvaluationInputComponents*/ 128,
	/*maxTessEvaluationOutputComponents*/ 128,
	/*maxTessEvaluationTextureImageUnits*/ 16,
	/*maxTessEvaluationUniformComponents*/ 1024,
	/*maxTessPatchComponents*/ 120,
	/*maxPatchVertices*/ 32,
	/*maxTessGenLevel*/ 64,
	/*maxViewports*/ 16,
	/*maxVertexAtomicCounters*/ 0,
	/*maxTessControlAtomicCounters*/ 0,
	/*maxTessEvaluationAtomicCounters*/ 0,
	/*maxGeometryAtomicCounters*/ 0,
	/*maxFragmentAtomicCounters*/ 8,
	/*maxCombinedAtomicCounters*/ 8,
	/*maxAtomicCounterBindings*/ 1,
	/*maxVertexAtomicCounterBuffers*/ 0,
	/*maxTessControlAtomicCounterBuffers*/ 0,
	/*maxTessEvaluationAtomicCounterBuffers*/ 0,
	/*maxGeometryAtomicCounterBuffers*/ 0,
	/*maxFragmentAtomicCounterBuffers*/ 1,
	/*maxCombinedAtomicCounterBuffers*/ 1,
	/*maxAtomicCounterBufferSize*/ 16384,
	/*maxTransformFeedbackBuffers*/ 4,
	/*maxTransformFeedbackInterleavedComponents*/ 64,
	/*maxCullDistances*/ 8,
	/*maxCombinedClipAndCullDistances*/ 8,
	/*maxSamples*/ 4,
	/*maxMeshOutputVerticesNV*/ 0,
	/*maxMeshOutputPrimitivesNV*/ 0,
	/*maxMeshWorkGroupSizeX_NV*/ 0,
	/*maxMeshWorkGroupSizeY_NV*/ 0,
	/*maxMeshWorkGroupSizeZ_NV*/ 0,
	/*maxTaskWorkGroupSizeX_NV*/ 0,
	/*maxTaskWorkGroupSizeY_NV*/ 0,
	/*maxTaskWorkGroupSizeZ_NV*/ 0,
	/*maxMeshViewCountNV*/ 0,
	/*limits*/ {
			/*nonInductiveForLoops*/ 1,
			/*whileLoops*/ 1,
			/*doWhileLoops*/ 1,
			/*generalUniformIndexing*/ 1,
			/*generalAttributeMatrixVectorIndexing*/ 1,
			/*generalVaryingIndexing*/ 1,
			/*generalSamplerIndexing*/ 1,
			/*generalVariableIndexing*/ 1,
			/*generalConstantMatrixVectorIndexing*/ 1,
	}
};

static Vector<uint8_t> _compile_shader_glsl(RenderingDevice::ShaderStage p_stage, const String &p_source_code, RenderingDevice::ShaderLanguage p_language, String *r_error) {

	Vector<uint8_t> ret;

	ERR_FAIL_COND_V(p_language == RenderingDevice::SHADER_LANGUAGE_HLSL, ret);

	EShLanguage stages[RenderingDevice::SHADER_STAGE_MAX] = {
		EShLangVertex,
		EShLangFragment,
		EShLangTessControl,
		EShLangTessEvaluation,
		EShLangCompute
	};

	int ClientInputSemanticsVersion = 100; // maps to, say, #define VULKAN 100

	glslang::EShTargetClientVersion VulkanClientVersion = glslang::EShTargetVulkan_1_0;
	glslang::EShTargetLanguageVersion TargetVersion = glslang::EShTargetSpv_1_0;
	glslang::TShader::ForbidIncluder includer;

	glslang::TShader shader(stages[p_stage]);
	CharString cs = p_source_code.ascii();
	const char *cs_strings = cs.get_data();

	shader.setStrings(&cs_strings, 1);
	shader.setEnvInput(glslang::EShSourceGlsl, stages[p_stage], glslang::EShClientVulkan, ClientInputSemanticsVersion);
	shader.setEnvClient(glslang::EShClientVulkan, VulkanClientVersion);
	shader.setEnvTarget(glslang::EShTargetSpv, TargetVersion);

	EShMessages messages = (EShMessages)(EShMsgSpvRules | EShMsgVulkanRules);
	const int DefaultVersion = 100;
	std::string pre_processed_code;

	//preprocess
	if (!shader.preprocess(&default_builtin_resource, DefaultVersion, ENoProfile, false, false, messages, &pre_processed_code, includer)) {

		if (r_error) {
			(*r_error) = "Failed pre-process:\n";
			(*r_error) += shader.getInfoLog();
			(*r_error) += "\n";
			(*r_error) += shader.getInfoDebugLog();
		}

		return ret;
	}
	//set back..
	cs_strings = pre_processed_code.c_str();
	shader.setStrings(&cs_strings, 1);

	//parse
	if (!shader.parse(&default_builtin_resource, DefaultVersion, false, messages)) {
		if (r_error) {
			(*r_error) = "Failed parse:\n";
			(*r_error) += shader.getInfoLog();
			(*r_error) += "\n";
			(*r_error) += shader.getInfoDebugLog();
		}
		return ret;
	}

	//link
	glslang::TProgram program;
	program.addShader(&shader);

	if (!program.link(messages)) {
		if (r_error) {
			(*r_error) = "Failed link:\n";
			(*r_error) += program.getInfoLog();
			(*r_error) += "\n";
			(*r_error) += program.getInfoDebugLog();
		}

		return ret;
	}

	std::vector<uint32_t> SpirV;
	spv::SpvBuildLogger logger;
	glslang::SpvOptions spvOptions;
	glslang::GlslangToSpv(*program.getIntermediate(stages[p_stage]), SpirV, &logger, &spvOptions);

	ret.resize(SpirV.size() * sizeof(uint32_t));
	{
		uint8_t *w = ret.ptrw();
		copymem(w, &SpirV[0], SpirV.size() * sizeof(uint32_t));
	}

	return ret;
}

void preregister_glslang_types() {
	// initialize in case it's not initialized. This is done once per thread
	// and it's safe to call multiple times
	glslang::InitializeProcess();
	RenderingDevice::shader_set_compile_function(_compile_shader_glsl);
}

void register_glslang_types() {
}
void unregister_glslang_types() {

	glslang::FinalizeProcess();
}
