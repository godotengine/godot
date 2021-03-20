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

#include "SPIRV/GlslangToSpv.h"
#include "SPIRV/Logger.h"
#include "SPIRV/SpvTools.h"

typedef struct glslang_program_s {
    glslang::TProgram* program;
    std::vector<unsigned int> spirv;
    std::string loggerMessages;
} glslang_program_t;

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

GLSLANG_EXPORT void glslang_program_SPIRV_generate(glslang_program_t* program, glslang_stage_t stage)
{
    spv::SpvBuildLogger logger;
    glslang::SpvOptions spvOptions;
    spvOptions.validate = true;

    const glslang::TIntermediate* intermediate = program->program->getIntermediate(c_shader_stage(stage));

    glslang::GlslangToSpv(*intermediate, program->spirv, &logger, &spvOptions);

    program->loggerMessages = logger.getAllMessages();
}

GLSLANG_EXPORT size_t glslang_program_SPIRV_get_size(glslang_program_t* program) { return program->spirv.size(); }

GLSLANG_EXPORT void glslang_program_SPIRV_get(glslang_program_t* program, unsigned int* out)
{
    memcpy(out, program->spirv.data(), program->spirv.size() * sizeof(unsigned int));
}

GLSLANG_EXPORT unsigned int* glslang_program_SPIRV_get_ptr(glslang_program_t* program)
{
    return program->spirv.data();
}

GLSLANG_EXPORT const char* glslang_program_SPIRV_get_messages(glslang_program_t* program)
{
    return program->loggerMessages.empty() ? nullptr : program->loggerMessages.c_str();
}
