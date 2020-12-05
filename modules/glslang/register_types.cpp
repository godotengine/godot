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
#include <StandAlone/ResourceLimits.h>
#include <glslang/Include/Types.h>
#include <glslang/Public/ShaderLang.h>

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
	if (!shader.preprocess(&glslang::DefaultTBuiltInResource, DefaultVersion, ENoProfile, false, false, messages, &pre_processed_code, includer)) {
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
	if (!shader.parse(&glslang::DefaultTBuiltInResource, DefaultVersion, false, messages)) {
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
