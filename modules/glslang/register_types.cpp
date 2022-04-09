/*************************************************************************/
/*  register_types.cpp                                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "glslang_resource_limits.h"

#include <glslang/Include/Types.h>
#include <glslang/Public/ShaderLang.h>
#include <glslang/SPIRV/GlslangToSpv.h>

static Vector<uint8_t> _compile_shader_glsl(RenderingDevice::ShaderStage p_stage, const String &p_source_code, RenderingDevice::ShaderLanguage p_language, String *r_error, const RenderingDevice::Capabilities *p_capabilities) {
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
	bool check_subgroup_support = true; // assume we support subgroups

	glslang::EShTargetClientVersion ClientVersion = glslang::EShTargetVulkan_1_2;
	glslang::EShTargetLanguageVersion TargetVersion = glslang::EShTargetSpv_1_5;
	glslang::TShader::ForbidIncluder includer;

	if (p_capabilities->device_family == RenderingDevice::DeviceFamily::DEVICE_VULKAN) {
		if (p_capabilities->version_major == 1 && p_capabilities->version_minor == 0) {
			ClientVersion = glslang::EShTargetVulkan_1_0;
			TargetVersion = glslang::EShTargetSpv_1_0;
			check_subgroup_support = false; // subgroups are not supported in Vulkan 1.0
		} else if (p_capabilities->version_major == 1 && p_capabilities->version_minor == 1) {
			ClientVersion = glslang::EShTargetVulkan_1_1;
			TargetVersion = glslang::EShTargetSpv_1_3;
		} else {
			// use defaults
		}
	} else {
		// once we support other backends we'll need to do something here
		if (r_error) {
			(*r_error) = "GLSLANG - Unsupported device family";
		}
		return ret;
	}

	glslang::TShader shader(stages[p_stage]);
	CharString cs = p_source_code.ascii();
	const char *cs_strings = cs.get_data();
	std::string preamble = "";

	shader.setStrings(&cs_strings, 1);
	shader.setEnvInput(glslang::EShSourceGlsl, stages[p_stage], glslang::EShClientVulkan, ClientInputSemanticsVersion);
	shader.setEnvClient(glslang::EShClientVulkan, ClientVersion);
	shader.setEnvTarget(glslang::EShTargetSpv, TargetVersion);

	if (check_subgroup_support) {
		uint32_t stage_bit = 1 << p_stage;

		if ((p_capabilities->subgroup_in_shaders & stage_bit) == stage_bit) {
			// stage supports subgroups
			preamble += "#define has_GL_KHR_shader_subgroup_basic 1\n";
			if (p_capabilities->subgroup_operations & RenderingDevice::SUBGROUP_VOTE_BIT) {
				preamble += "#define has_GL_KHR_shader_subgroup_vote 1\n";
			}
			if (p_capabilities->subgroup_operations & RenderingDevice::SUBGROUP_ARITHMETIC_BIT) {
				preamble += "#define has_GL_KHR_shader_subgroup_arithmetic 1\n";
			}
			if (p_capabilities->subgroup_operations & RenderingDevice::SUBGROUP_BALLOT_BIT) {
				preamble += "#define has_GL_KHR_shader_subgroup_ballot 1\n";
			}
			if (p_capabilities->subgroup_operations & RenderingDevice::SUBGROUP_SHUFFLE_BIT) {
				preamble += "#define has_GL_KHR_shader_subgroup_shuffle 1\n";
			}
			if (p_capabilities->subgroup_operations & RenderingDevice::SUBGROUP_SHUFFLE_RELATIVE_BIT) {
				preamble += "#define has_GL_KHR_shader_subgroup_shuffle_relative 1\n";
			}
			if (p_capabilities->subgroup_operations & RenderingDevice::SUBGROUP_CLUSTERED_BIT) {
				preamble += "#define has_GL_KHR_shader_subgroup_clustered 1\n";
			}
			if (p_capabilities->subgroup_operations & RenderingDevice::SUBGROUP_QUAD_BIT) {
				preamble += "#define has_GL_KHR_shader_subgroup_quad 1\n";
			}
		}
	}

	if (p_capabilities->supports_multiview) {
		preamble += "#define has_VK_KHR_multiview 1\n";
	}

	if (!preamble.empty()) {
		shader.setPreamble(preamble.c_str());
	}

	EShMessages messages = (EShMessages)(EShMsgSpvRules | EShMsgVulkanRules);
	const int DefaultVersion = 100;
	std::string pre_processed_code;

	//preprocess
	if (!shader.preprocess(&DefaultTBuiltInResource, DefaultVersion, ENoProfile, false, false, messages, &pre_processed_code, includer)) {
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
	if (!shader.parse(&DefaultTBuiltInResource, DefaultVersion, false, messages)) {
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
		memcpy(w, &SpirV[0], SpirV.size() * sizeof(uint32_t));
	}

	return ret;
}

static String _get_cache_key_function_glsl(const RenderingDevice::Capabilities *p_capabilities) {
	String version;
	version = "SpirVGen=" + itos(glslang::GetSpirvGeneratorVersion()) + ", major=" + itos(p_capabilities->version_major) + ", minor=" + itos(p_capabilities->version_minor) + " , subgroup_size=" + itos(p_capabilities->subgroup_operations) + " , subgroup_ops=" + itos(p_capabilities->subgroup_operations) + " , subgroup_in_shaders=" + itos(p_capabilities->subgroup_in_shaders);
	return version;
}

void preregister_glslang_types() {
	// Initialize in case it's not initialized. This is done once per thread
	// and it's safe to call multiple times.
	glslang::InitializeProcess();
	RenderingDevice::shader_set_compile_to_spirv_function(_compile_shader_glsl);
	RenderingDevice::shader_set_get_cache_key_function(_get_cache_key_function_glsl);
}

void register_glslang_types() {
}

void unregister_glslang_types() {
	glslang::FinalizeProcess();
}
