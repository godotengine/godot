//
// Copyright (C) 2014-2016 LunarG, Inc.
// Copyright (C) 2018-2020 Google, Inc.
//
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
//    Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//
//    Redistributions in binary form must reproduce the above
//    copyright notice, this list of conditions and the following
//    disclaimer in the documentation and/or other materials provided
//    with the distribution.
//
//    Neither the name of 3Dlabs Inc. Ltd. nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
// COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

//
// Call into SPIRV-Tools to disassemble, validate, and optimize.
//

#if ENABLE_OPT

#include <cstdio>
#include <iostream>

#include "SpvTools.h"
#include "spirv-tools/optimizer.hpp"

namespace glslang {

// Translate glslang's view of target versioning to what SPIRV-Tools uses.
spv_target_env MapToSpirvToolsEnv(const SpvVersion& spvVersion, spv::SpvBuildLogger* logger)
{
    switch (spvVersion.vulkan) {
    case glslang::EShTargetVulkan_1_0:
        return spv_target_env::SPV_ENV_VULKAN_1_0;
    case glslang::EShTargetVulkan_1_1:
        switch (spvVersion.spv) {
        case EShTargetSpv_1_0:
        case EShTargetSpv_1_1:
        case EShTargetSpv_1_2:
        case EShTargetSpv_1_3:
            return spv_target_env::SPV_ENV_VULKAN_1_1;
        case EShTargetSpv_1_4:
            return spv_target_env::SPV_ENV_VULKAN_1_1_SPIRV_1_4;
        default:
            logger->missingFunctionality("Target version for SPIRV-Tools validator");
            return spv_target_env::SPV_ENV_VULKAN_1_1;
        }
    case glslang::EShTargetVulkan_1_2:
        return spv_target_env::SPV_ENV_VULKAN_1_2;
    default:
        break;
    }

    if (spvVersion.openGl > 0)
        return spv_target_env::SPV_ENV_OPENGL_4_5;

    logger->missingFunctionality("Target version for SPIRV-Tools validator");
    return spv_target_env::SPV_ENV_UNIVERSAL_1_0;
}

// Callback passed to spvtools::Optimizer::SetMessageConsumer
void OptimizerMesssageConsumer(spv_message_level_t level, const char *source,
        const spv_position_t &position, const char *message)
{
    auto &out = std::cerr;
    switch (level)
    {
    case SPV_MSG_FATAL:
    case SPV_MSG_INTERNAL_ERROR:
    case SPV_MSG_ERROR:
        out << "error: ";
        break;
    case SPV_MSG_WARNING:
        out << "warning: ";
        break;
    case SPV_MSG_INFO:
    case SPV_MSG_DEBUG:
        out << "info: ";
        break;
    default:
        break;
    }
    if (source)
    {
        out << source << ":";
    }
    out << position.line << ":" << position.column << ":" << position.index << ":";
    if (message)
    {
        out << " " << message;
    }
    out << std::endl;
}

// Use the SPIRV-Tools disassembler to print SPIR-V using a SPV_ENV_UNIVERSAL_1_3 environment.
void SpirvToolsDisassemble(std::ostream& out, const std::vector<unsigned int>& spirv)
{
    SpirvToolsDisassemble(out, spirv, spv_target_env::SPV_ENV_UNIVERSAL_1_3);
}

// Use the SPIRV-Tools disassembler to print SPIR-V with a provided SPIR-V environment.
void SpirvToolsDisassemble(std::ostream& out, const std::vector<unsigned int>& spirv,
                           spv_target_env requested_context)
{
    // disassemble
    spv_context context = spvContextCreate(requested_context);
    spv_text text;
    spv_diagnostic diagnostic = nullptr;
    spvBinaryToText(context, spirv.data(), spirv.size(),
        SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES | SPV_BINARY_TO_TEXT_OPTION_INDENT,
        &text, &diagnostic);

    // dump
    if (diagnostic == nullptr)
        out << text->str;
    else
        spvDiagnosticPrint(diagnostic);

    // teardown
    spvDiagnosticDestroy(diagnostic);
    spvContextDestroy(context);
}

// Apply the SPIRV-Tools validator to generated SPIR-V.
void SpirvToolsValidate(const glslang::TIntermediate& intermediate, std::vector<unsigned int>& spirv,
                        spv::SpvBuildLogger* logger, bool prelegalization)
{
    // validate
    spv_context context = spvContextCreate(MapToSpirvToolsEnv(intermediate.getSpv(), logger));
    spv_const_binary_t binary = { spirv.data(), spirv.size() };
    spv_diagnostic diagnostic = nullptr;
    spv_validator_options options = spvValidatorOptionsCreate();
    spvValidatorOptionsSetRelaxBlockLayout(options, intermediate.usingHlslOffsets());
    spvValidatorOptionsSetBeforeHlslLegalization(options, prelegalization);
    spvValidatorOptionsSetScalarBlockLayout(options, intermediate.usingScalarBlockLayout());
    spvValidatorOptionsSetWorkgroupScalarBlockLayout(options, intermediate.usingScalarBlockLayout());
    spvValidateWithOptions(context, options, &binary, &diagnostic);

    // report
    if (diagnostic != nullptr) {
        logger->error("SPIRV-Tools Validation Errors");
        logger->error(diagnostic->error);
    }

    // tear down
    spvValidatorOptionsDestroy(options);
    spvDiagnosticDestroy(diagnostic);
    spvContextDestroy(context);
}

// Apply the SPIRV-Tools optimizer to generated SPIR-V.  HLSL SPIR-V is legalized in the process.
void SpirvToolsTransform(const glslang::TIntermediate& intermediate, std::vector<unsigned int>& spirv,
                         spv::SpvBuildLogger* logger, const SpvOptions* options)
{
    spv_target_env target_env = MapToSpirvToolsEnv(intermediate.getSpv(), logger);

    spvtools::Optimizer optimizer(target_env);
    optimizer.SetMessageConsumer(OptimizerMesssageConsumer);

    // If debug (specifically source line info) is being generated, propagate
    // line information into all SPIR-V instructions. This avoids loss of
    // information when instructions are deleted or moved. Later, remove
    // redundant information to minimize final SPRIR-V size.
    if (options->stripDebugInfo) {
        optimizer.RegisterPass(spvtools::CreateStripDebugInfoPass());
    }
    optimizer.RegisterPass(spvtools::CreateWrapOpKillPass());
    optimizer.RegisterPass(spvtools::CreateDeadBranchElimPass());
    optimizer.RegisterPass(spvtools::CreateMergeReturnPass());
    optimizer.RegisterPass(spvtools::CreateInlineExhaustivePass());
    optimizer.RegisterPass(spvtools::CreateEliminateDeadFunctionsPass());
    optimizer.RegisterPass(spvtools::CreateScalarReplacementPass());
    optimizer.RegisterPass(spvtools::CreateLocalAccessChainConvertPass());
    optimizer.RegisterPass(spvtools::CreateLocalSingleBlockLoadStoreElimPass());
    optimizer.RegisterPass(spvtools::CreateLocalSingleStoreElimPass());
    optimizer.RegisterPass(spvtools::CreateSimplificationPass());
    optimizer.RegisterPass(spvtools::CreateAggressiveDCEPass());
    optimizer.RegisterPass(spvtools::CreateVectorDCEPass());
    optimizer.RegisterPass(spvtools::CreateDeadInsertElimPass());
    optimizer.RegisterPass(spvtools::CreateAggressiveDCEPass());
    optimizer.RegisterPass(spvtools::CreateDeadBranchElimPass());
    optimizer.RegisterPass(spvtools::CreateBlockMergePass());
    optimizer.RegisterPass(spvtools::CreateLocalMultiStoreElimPass());
    optimizer.RegisterPass(spvtools::CreateIfConversionPass());
    optimizer.RegisterPass(spvtools::CreateSimplificationPass());
    optimizer.RegisterPass(spvtools::CreateAggressiveDCEPass());
    optimizer.RegisterPass(spvtools::CreateVectorDCEPass());
    optimizer.RegisterPass(spvtools::CreateDeadInsertElimPass());
    optimizer.RegisterPass(spvtools::CreateInterpolateFixupPass());
    if (options->optimizeSize) {
        optimizer.RegisterPass(spvtools::CreateRedundancyEliminationPass());
    }
    optimizer.RegisterPass(spvtools::CreateAggressiveDCEPass());
    optimizer.RegisterPass(spvtools::CreateCFGCleanupPass());

    spvtools::OptimizerOptions spvOptOptions;
    optimizer.SetTargetEnv(MapToSpirvToolsEnv(intermediate.getSpv(), logger));
    spvOptOptions.set_run_validator(false); // The validator may run as a separate step later on
    optimizer.Run(spirv.data(), spirv.size(), &spirv, spvOptOptions);
}

// Apply the SPIRV-Tools optimizer to strip debug info from SPIR-V.  This is implicitly done by
// SpirvToolsTransform if spvOptions->stripDebugInfo is set, but can be called separately if
// optimization is disabled.
void SpirvToolsStripDebugInfo(const glslang::TIntermediate& intermediate,
        std::vector<unsigned int>& spirv, spv::SpvBuildLogger* logger)
{
    spv_target_env target_env = MapToSpirvToolsEnv(intermediate.getSpv(), logger);

    spvtools::Optimizer optimizer(target_env);
    optimizer.SetMessageConsumer(OptimizerMesssageConsumer);

    optimizer.RegisterPass(spvtools::CreateStripDebugInfoPass());

    spvtools::OptimizerOptions spvOptOptions;
    optimizer.SetTargetEnv(MapToSpirvToolsEnv(intermediate.getSpv(), logger));
    spvOptOptions.set_run_validator(false); // The validator may run as a separate step later on
    optimizer.Run(spirv.data(), spirv.size(), &spirv, spvOptOptions);
}

}; // end namespace glslang

#endif
