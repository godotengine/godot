/**
    This code is based on the glslang_c_interface implementation by Viktor Latypov
**/

/**
BSD 2-Clause License

Copyright (c) 2019, Viktor Latypov
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**/

#include "glslang/Include/glslang_c_interface.h"

#include "StandAlone/DirStackFileIncluder.h"
#include "StandAlone/ResourceLimits.h"
#include "glslang/Include/ShHandle.h"

#include "glslang/Include/ResourceLimits.h"
#include "glslang/MachineIndependent/Versions.h"

static_assert(int(GLSLANG_STAGE_COUNT) == EShLangCount, "");
static_assert(int(GLSLANG_STAGE_MASK_COUNT) == EShLanguageMaskCount, "");
static_assert(int(GLSLANG_SOURCE_COUNT) == glslang::EShSourceCount, "");
static_assert(int(GLSLANG_CLIENT_COUNT) == glslang::EShClientCount, "");
static_assert(int(GLSLANG_TARGET_COUNT) == glslang::EShTargetCount, "");
static_assert(int(GLSLANG_TARGET_CLIENT_VERSION_COUNT) == glslang::EShTargetClientVersionCount, "");
static_assert(int(GLSLANG_TARGET_LANGUAGE_VERSION_COUNT) == glslang::EShTargetLanguageVersionCount, "");
static_assert(int(GLSLANG_OPT_LEVEL_COUNT) == EshOptLevelCount, "");
static_assert(int(GLSLANG_TEX_SAMP_TRANS_COUNT) == EShTexSampTransCount, "");
static_assert(int(GLSLANG_MSG_COUNT) == EShMsgCount, "");
static_assert(int(GLSLANG_REFLECTION_COUNT) == EShReflectionCount, "");
static_assert(int(GLSLANG_PROFILE_COUNT) == EProfileCount, "");
static_assert(sizeof(glslang_limits_t) == sizeof(TLimits), "");
static_assert(sizeof(glslang_resource_t) == sizeof(TBuiltInResource), "");

typedef struct glslang_shader_s {
    glslang::TShader* shader;
    std::string preprocessedGLSL;
} glslang_shader_t;

typedef struct glslang_program_s {
    glslang::TProgram* program;
    std::vector<unsigned int> spirv;
    std::string loggerMessages;
} glslang_program_t;

/* Wrapper/Adapter for C glsl_include_callbacks_t functions

   This class contains a 'glsl_include_callbacks_t' structure
   with C include_local/include_system callback pointers.

   This class implement TShader::Includer interface
   by redirecting C++ virtual methods to C callbacks.

   The 'IncludeResult' instances produced by this Includer
   contain a reference to glsl_include_result_t C structure
   to allow its lifetime management by another C callback
   (CallbackIncluder::callbacks::free_include_result)
*/
class CallbackIncluder : public glslang::TShader::Includer {
public:
    /* Wrapper of IncludeResult which stores a glsl_include_result object internally */
    class CallbackIncludeResult : public glslang::TShader::Includer::IncludeResult {
    public:
        CallbackIncludeResult(const std::string& headerName, const char* const headerData, const size_t headerLength,
                              void* userData, glsl_include_result_t* includeResult)
            : glslang::TShader::Includer::IncludeResult(headerName, headerData, headerLength, userData),
              includeResult(includeResult)
        {
        }

        virtual ~CallbackIncludeResult() {}

    protected:
        friend class CallbackIncluder;

        glsl_include_result_t* includeResult;
    };

public:
    CallbackIncluder(glsl_include_callbacks_t _callbacks, void* _context) : callbacks(_callbacks), context(_context) {}

    virtual ~CallbackIncluder() {}

    virtual IncludeResult* includeSystem(const char* headerName, const char* includerName,
                                         size_t inclusionDepth) override
    {
        if (this->callbacks.include_system) {
            glsl_include_result_t* result =
                this->callbacks.include_system(this->context, headerName, includerName, inclusionDepth);

            return new CallbackIncludeResult(std::string(headerName), result->header_data, result->header_length,
                                             nullptr, result);
        }

        return glslang::TShader::Includer::includeSystem(headerName, includerName, inclusionDepth);
    }

    virtual IncludeResult* includeLocal(const char* headerName, const char* includerName,
                                        size_t inclusionDepth) override
    {
        if (this->callbacks.include_local) {
            glsl_include_result_t* result =
                this->callbacks.include_local(this->context, headerName, includerName, inclusionDepth);

            return new CallbackIncludeResult(std::string(headerName), result->header_data, result->header_length,
                                             nullptr, result);
        }

        return glslang::TShader::Includer::includeLocal(headerName, includerName, inclusionDepth);
    }

    /* This function only calls free_include_result callback
       when the IncludeResult instance is allocated by a C function */
    virtual void releaseInclude(IncludeResult* result) override
    {
        if (result == nullptr)
            return;

        if (this->callbacks.free_include_result && (result->userData == nullptr)) {
            CallbackIncludeResult* innerResult = static_cast<CallbackIncludeResult*>(result);
            /* use internal free() function */
            this->callbacks.free_include_result(this->context, innerResult->includeResult);
            /* ignore internal fields of TShader::Includer::IncludeResult */
            delete result;
            return;
        }

        delete[] static_cast<char*>(result->userData);
        delete result;
    }

private:
    CallbackIncluder() {}

    /* C callback pointers */
    glsl_include_callbacks_t callbacks;
    /* User-defined context */
    void* context;
};

GLSLANG_EXPORT int glslang_initialize_process() { return static_cast<int>(glslang::InitializeProcess()); }

GLSLANG_EXPORT void glslang_finalize_process() { glslang::FinalizeProcess(); }

static EShLanguage c_shader_stage(glslang_stage_t stage)
{
    switch (stage) {
    case GLSLANG_STAGE_VERTEX:
        return EShLangVertex;
    case GLSLANG_STAGE_TESSCONTROL:
        return EShLangTessControl;
    case GLSLANG_STAGE_TESSEVALUATION:
        return EShLangTessEvaluation;
    case GLSLANG_STAGE_GEOMETRY:
        return EShLangGeometry;
    case GLSLANG_STAGE_FRAGMENT:
        return EShLangFragment;
    case GLSLANG_STAGE_COMPUTE:
        return EShLangCompute;
    case GLSLANG_STAGE_RAYGEN_NV:
        return EShLangRayGen;
    case GLSLANG_STAGE_INTERSECT_NV:
        return EShLangIntersect;
    case GLSLANG_STAGE_ANYHIT_NV:
        return EShLangAnyHit;
    case GLSLANG_STAGE_CLOSESTHIT_NV:
        return EShLangClosestHit;
    case GLSLANG_STAGE_MISS_NV:
        return EShLangMiss;
    case GLSLANG_STAGE_CALLABLE_NV:
        return EShLangCallable;
    case GLSLANG_STAGE_TASK_NV:
        return EShLangTaskNV;
    case GLSLANG_STAGE_MESH_NV:
        return EShLangMeshNV;
    default:
        break;
    }
    return EShLangCount;
}

static int c_shader_messages(glslang_messages_t messages)
{
#define CONVERT_MSG(in, out)                                                                                           \
    if ((messages & in) == in)                                                                                         \
        res |= out;

    int res = 0;

    CONVERT_MSG(GLSLANG_MSG_RELAXED_ERRORS_BIT, EShMsgRelaxedErrors);
    CONVERT_MSG(GLSLANG_MSG_SUPPRESS_WARNINGS_BIT, EShMsgSuppressWarnings);
    CONVERT_MSG(GLSLANG_MSG_AST_BIT, EShMsgAST);
    CONVERT_MSG(GLSLANG_MSG_SPV_RULES_BIT, EShMsgSpvRules);
    CONVERT_MSG(GLSLANG_MSG_VULKAN_RULES_BIT, EShMsgVulkanRules);
    CONVERT_MSG(GLSLANG_MSG_ONLY_PREPROCESSOR_BIT, EShMsgOnlyPreprocessor);
    CONVERT_MSG(GLSLANG_MSG_READ_HLSL_BIT, EShMsgReadHlsl);
    CONVERT_MSG(GLSLANG_MSG_CASCADING_ERRORS_BIT, EShMsgCascadingErrors);
    CONVERT_MSG(GLSLANG_MSG_KEEP_UNCALLED_BIT, EShMsgKeepUncalled);
    CONVERT_MSG(GLSLANG_MSG_HLSL_OFFSETS_BIT, EShMsgHlslOffsets);
    CONVERT_MSG(GLSLANG_MSG_DEBUG_INFO_BIT, EShMsgDebugInfo);
    CONVERT_MSG(GLSLANG_MSG_HLSL_ENABLE_16BIT_TYPES_BIT, EShMsgHlslEnable16BitTypes);
    CONVERT_MSG(GLSLANG_MSG_HLSL_LEGALIZATION_BIT, EShMsgHlslLegalization);
    CONVERT_MSG(GLSLANG_MSG_HLSL_DX9_COMPATIBLE_BIT, EShMsgHlslDX9Compatible);
    CONVERT_MSG(GLSLANG_MSG_BUILTIN_SYMBOL_TABLE_BIT, EShMsgBuiltinSymbolTable);
    return res;
#undef CONVERT_MSG
}

static glslang::EShTargetLanguageVersion
c_shader_target_language_version(glslang_target_language_version_t target_language_version)
{
    switch (target_language_version) {
    case GLSLANG_TARGET_SPV_1_0:
        return glslang::EShTargetSpv_1_0;
    case GLSLANG_TARGET_SPV_1_1:
        return glslang::EShTargetSpv_1_1;
    case GLSLANG_TARGET_SPV_1_2:
        return glslang::EShTargetSpv_1_2;
    case GLSLANG_TARGET_SPV_1_3:
        return glslang::EShTargetSpv_1_3;
    case GLSLANG_TARGET_SPV_1_4:
        return glslang::EShTargetSpv_1_4;
    case GLSLANG_TARGET_SPV_1_5:
        return glslang::EShTargetSpv_1_5;
    default:
        break;
    }
    return glslang::EShTargetSpv_1_0;
}

static glslang::EShClient c_shader_client(glslang_client_t client)
{
    switch (client) {
    case GLSLANG_CLIENT_VULKAN:
        return glslang::EShClientVulkan;
    case GLSLANG_CLIENT_OPENGL:
        return glslang::EShClientOpenGL;
    default:
        break;
    }

    return glslang::EShClientNone;
}

static glslang::EShTargetClientVersion c_shader_client_version(glslang_target_client_version_t client_version)
{
    switch (client_version) {
    case GLSLANG_TARGET_VULKAN_1_1:
        return glslang::EShTargetVulkan_1_1;
    case GLSLANG_TARGET_OPENGL_450:
        return glslang::EShTargetOpenGL_450;
    default:
        break;
    }

    return glslang::EShTargetVulkan_1_0;
}

static glslang::EShTargetLanguage c_shader_target_language(glslang_target_language_t target_language)
{
    if (target_language == GLSLANG_TARGET_NONE)
        return glslang::EShTargetNone;

    return glslang::EShTargetSpv;
}

static glslang::EShSource c_shader_source(glslang_source_t source)
{
    switch (source) {
    case GLSLANG_SOURCE_GLSL:
        return glslang::EShSourceGlsl;
    case GLSLANG_SOURCE_HLSL:
        return glslang::EShSourceHlsl;
    default:
        break;
    }

    return glslang::EShSourceNone;
}

static EProfile c_shader_profile(glslang_profile_t profile)
{
    switch (profile) {
    case GLSLANG_BAD_PROFILE:
        return EBadProfile;
    case GLSLANG_NO_PROFILE:
        return ENoProfile;
    case GLSLANG_CORE_PROFILE:
        return ECoreProfile;
    case GLSLANG_COMPATIBILITY_PROFILE:
        return ECompatibilityProfile;
    case GLSLANG_ES_PROFILE:
        return EEsProfile;
    case GLSLANG_PROFILE_COUNT: // Should not use this
        break;
    }

    return EProfile();
}

GLSLANG_EXPORT glslang_shader_t* glslang_shader_create(const glslang_input_t* input)
{
    if (!input || !input->code) {
        printf("Error creating shader: null input(%p)/input->code\n", input);

        if (input)
            printf("input->code = %p\n", input->code);

        return nullptr;
    }

    glslang_shader_t* shader = new glslang_shader_t();

    shader->shader = new glslang::TShader(c_shader_stage(input->stage));
    shader->shader->setStrings(&input->code, 1);
    shader->shader->setEnvInput(c_shader_source(input->language), c_shader_stage(input->stage),
                                c_shader_client(input->client), input->default_version);
    shader->shader->setEnvClient(c_shader_client(input->client), c_shader_client_version(input->client_version));
    shader->shader->setEnvTarget(c_shader_target_language(input->target_language),
                                 c_shader_target_language_version(input->target_language_version));

    return shader;
}

GLSLANG_EXPORT const char* glslang_shader_get_preprocessed_code(glslang_shader_t* shader)
{
    return shader->preprocessedGLSL.c_str();
}

GLSLANG_EXPORT int glslang_shader_preprocess(glslang_shader_t* shader, const glslang_input_t* input)
{
    DirStackFileIncluder Includer;
    /* TODO: use custom callbacks if they are available in 'i->callbacks' */
    return shader->shader->preprocess(
        reinterpret_cast<const TBuiltInResource*>(input->resource),
        input->default_version,
        c_shader_profile(input->default_profile),
        input->force_default_version_and_profile != 0,
        input->forward_compatible != 0,
        (EShMessages)c_shader_messages(input->messages),
        &shader->preprocessedGLSL,
        Includer
    );
}

GLSLANG_EXPORT int glslang_shader_parse(glslang_shader_t* shader, const glslang_input_t* input)
{
    const char* preprocessedCStr = shader->preprocessedGLSL.c_str();
    shader->shader->setStrings(&preprocessedCStr, 1);

    return shader->shader->parse(
        reinterpret_cast<const TBuiltInResource*>(input->resource),
        input->default_version,
        input->forward_compatible != 0,
        (EShMessages)c_shader_messages(input->messages)
    );
}

GLSLANG_EXPORT const char* glslang_shader_get_info_log(glslang_shader_t* shader) { return shader->shader->getInfoLog(); }

GLSLANG_EXPORT const char* glslang_shader_get_info_debug_log(glslang_shader_t* shader) { return shader->shader->getInfoDebugLog(); }

GLSLANG_EXPORT void glslang_shader_delete(glslang_shader_t* shader)
{
    if (!shader)
        return;

    delete (shader->shader);
    delete (shader);
}

GLSLANG_EXPORT glslang_program_t* glslang_program_create()
{
    glslang_program_t* p = new glslang_program_t();
    p->program = new glslang::TProgram();
    return p;
}

GLSLANG_EXPORT void glslang_program_delete(glslang_program_t* program)
{
    if (!program)
        return;

    delete (program->program);
    delete (program);
}

GLSLANG_EXPORT void glslang_program_add_shader(glslang_program_t* program, glslang_shader_t* shader)
{
    program->program->addShader(shader->shader);
}

GLSLANG_EXPORT int glslang_program_link(glslang_program_t* program, int messages)
{
    return (int)program->program->link((EShMessages)messages);
}

GLSLANG_EXPORT const char* glslang_program_get_info_log(glslang_program_t* program)
{
    return program->program->getInfoLog();
}

GLSLANG_EXPORT const char* glslang_program_get_info_debug_log(glslang_program_t* program)
{
    return program->program->getInfoDebugLog();
}
