//
// Copyright (C) 2014-2016 LunarG, Inc.
// Copyright (C) 2018 Google, Inc.
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

#pragma once
#ifndef GLSLANG_SPV_TOOLS_H
#define GLSLANG_SPV_TOOLS_H

#if ENABLE_OPT
#include <vector>
#include <ostream>
#include "spirv-tools/libspirv.h"
#endif

#include "glslang/MachineIndependent/localintermediate.h"
#include "Logger.h"

namespace glslang {

struct SpvOptions {
    SpvOptions() : generateDebugInfo(false), stripDebugInfo(false), disableOptimizer(true),
        optimizeSize(false), disassemble(false), validate(false) { }
    bool generateDebugInfo;
    bool stripDebugInfo;
    bool disableOptimizer;
    bool optimizeSize;
    bool disassemble;
    bool validate;
};

#if ENABLE_OPT

// Use the SPIRV-Tools disassembler to print SPIR-V using a SPV_ENV_UNIVERSAL_1_3 environment.
void SpirvToolsDisassemble(std::ostream& out, const std::vector<unsigned int>& spirv);

// Use the SPIRV-Tools disassembler to print SPIR-V with a provided SPIR-V environment.
void SpirvToolsDisassemble(std::ostream& out, const std::vector<unsigned int>& spirv,
                           spv_target_env requested_context);

// Apply the SPIRV-Tools validator to generated SPIR-V.
void SpirvToolsValidate(const glslang::TIntermediate& intermediate, std::vector<unsigned int>& spirv,
                        spv::SpvBuildLogger*, bool prelegalization);

// Apply the SPIRV-Tools optimizer to generated SPIR-V.  HLSL SPIR-V is legalized in the process.
void SpirvToolsTransform(const glslang::TIntermediate& intermediate, std::vector<unsigned int>& spirv,
                         spv::SpvBuildLogger*, const SpvOptions*);

// Apply the SPIRV-Tools optimizer to strip debug info from SPIR-V.  This is implicitly done by
// SpirvToolsTransform if spvOptions->stripDebugInfo is set, but can be called separately if
// optimization is disabled.
void SpirvToolsStripDebugInfo(const glslang::TIntermediate& intermediate,
        std::vector<unsigned int>& spirv, spv::SpvBuildLogger*);

#endif

} // end namespace glslang

#endif // GLSLANG_SPV_TOOLS_H
