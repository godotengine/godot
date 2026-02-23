/**************************************************************************/
/*  register_types.cpp                                                    */
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

#include "register_types.h"

#include "core/config/engine.h"
#include "core/io/file_access.h"
#include "core/os/os.h"
#include "shader_compile.h"

GODOT_GCC_WARNING_PUSH_AND_IGNORE("-Wshadow")

#include <glslang/Public/ResourceLimits.h>
#include <glslang/Public/ShaderLang.h>
#include <glslang/SPIRV/GlslangToSpv.h>

GODOT_GCC_WARNING_POP

class GlslIncluder : public glslang::TShader::Includer {
	static const int max_depth = 1000;

	int num_base_include_dirs = 0;
	Vector<String> dirs;

public:
	void add_include_dirs(const Vector<String> &p_include_dirs) {
		dirs.append_array(p_include_dirs);
		num_base_include_dirs += p_include_dirs.size();
	}

	virtual IncludeResult *includeSystem(const char *p_header_name, const char *p_includer_name, size_t p_inclusion_depth) override {
		return nullptr;
	}

	virtual IncludeResult *includeLocal(const char *p_header_name, const char *p_includer_name, size_t p_inclusion_depth) override {
		if (p_inclusion_depth > max_depth) {
			return nullptr;
		}

		dirs.resize(num_base_include_dirs + p_inclusion_depth - 1);

		String include_path = p_header_name;
		if (include_path.is_relative_path()) {
			for (int i = dirs.size() - 1; i >= 0; --i) {
				String tmp_include = dirs[i].path_join(include_path);
				if (FileAccess::exists(tmp_include)) {
					include_path = tmp_include;
					break;
				}
			}
		}

		Error err;
		Ref<FileAccess> file_inc = FileAccess::open(include_path, FileAccess::READ, &err);
		if (err == OK) {
			dirs.append(include_path.get_base_dir());
			CharString cs_include_path = include_path.utf8();

			CharString cs_include_content = file_inc->get_as_utf8_string().utf8();
			char *cs_include_content_ptr = (char *)memalloc(cs_include_content.length() + 1);
			memcpy(cs_include_content_ptr, cs_include_content.get_data(), cs_include_content.length() + 1);

			return memnew(IncludeResult(cs_include_path.get_data(), cs_include_content_ptr, cs_include_content.length(), cs_include_content_ptr));
		}

		return nullptr;
	}

	virtual void releaseInclude(IncludeResult *p_include_result) override {
		if (p_include_result) {
			if (p_include_result->userData) {
				memfree(p_include_result->userData);
			}
			memdelete(p_include_result);
		}
	}

	virtual ~GlslIncluder() override {}
};

Vector<uint8_t> compile_glslang_shader(RenderingDeviceCommons::ShaderStage p_stage, const String &p_source_code, RenderingDeviceCommons::ShaderLanguageVersion p_language_version, RenderingDeviceCommons::ShaderSpirvVersion p_spirv_version, String *r_error, const String &p_source_file_path, const Vector<String> &p_include_dirs) {
	Vector<uint8_t> ret;
	EShLanguage stages[RenderingDeviceCommons::SHADER_STAGE_MAX] = {
		EShLangVertex,
		EShLangFragment,
		EShLangTessControl,
		EShLangTessEvaluation,
		EShLangCompute,
		EShLangRayGen,
		EShLangAnyHit,
		EShLangClosestHit,
		EShLangMiss,
		EShLangIntersect,
	};

	int ClientInputSemanticsVersion = 100; // maps to, say, #define VULKAN 100

	// The enum values can be converted directly.
	glslang::EShTargetClientVersion ClientVersion = (glslang::EShTargetClientVersion)p_language_version;
	glslang::EShTargetLanguageVersion TargetVersion = (glslang::EShTargetLanguageVersion)p_spirv_version;

	glslang::TShader shader(stages[p_stage]);
	CharString cs = p_source_code.utf8();
	const char *cs_strings = cs.get_data();
	int source_code_length = cs.length();
	std::string preamble = "";

	CharString cs_source_file_path = p_source_file_path.utf8();
	const char *cs_names = cs_source_file_path.get_data();
	shader.setStringsWithLengthsAndNames(&cs_strings, &source_code_length, &cs_names, 1);
	shader.setEnvInput(glslang::EShSourceGlsl, stages[p_stage], glslang::EShClientVulkan, ClientInputSemanticsVersion);
	shader.setEnvClient(glslang::EShClientVulkan, ClientVersion);
	shader.setEnvTarget(glslang::EShTargetSpv, TargetVersion);

	if (!preamble.empty()) {
		shader.setPreamble(preamble.c_str());
	}

	bool generate_spirv_debug_info = Engine::get_singleton()->is_generate_spirv_debug_info_enabled();
#ifdef D3D12_ENABLED
	if (OS::get_singleton()->get_current_rendering_driver_name() == "d3d12") {
		// SPIRV to DXIL conversion does not support debug info.
		generate_spirv_debug_info = false;
	}
#endif

	EShMessages messages = (EShMessages)(EShMsgSpvRules | EShMsgVulkanRules);
	if (generate_spirv_debug_info) {
		messages = (EShMessages)(messages | EShMsgDebugInfo);
	}
	const int DefaultVersion = 100;

	GlslIncluder includer = {};

	includer.add_include_dirs(p_include_dirs);
	if (!p_source_file_path.is_empty()) {
		includer.add_include_dirs({ p_source_file_path.get_base_dir() });
	}

	//parse
	if (!shader.parse(GetDefaultResources(), DefaultVersion, false, messages, includer)) {
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

	if (generate_spirv_debug_info) {
		spvOptions.generateDebugInfo = true;
		spvOptions.emitNonSemanticShaderDebugInfo = true;
		spvOptions.emitNonSemanticShaderDebugSource = true;
	}

	glslang::GlslangToSpv(*program.getIntermediate(stages[p_stage]), SpirV, &logger, &spvOptions);

	ret.resize(SpirV.size() * sizeof(uint32_t));
	{
		uint8_t *w = ret.ptrw();
		memcpy(w, &SpirV[0], SpirV.size() * sizeof(uint32_t));
	}

	return ret;
}

void initialize_glslang_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_CORE) {
		return;
	}

	// Initialize in case it's not initialized. This is done once per thread
	// and it's safe to call multiple times.
	glslang::InitializeProcess();
}

void uninitialize_glslang_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_CORE) {
		return;
	}

	glslang::FinalizeProcess();
}
