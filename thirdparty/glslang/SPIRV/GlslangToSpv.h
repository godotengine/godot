//
// Copyright (C) 2014 LunarG, Inc.
// Copyright (C) 2015-2018 Google, Inc.
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

#pragma once

#if defined(_MSC_VER) && _MSC_VER >= 1900
    #pragma warning(disable : 4464) // relative include path contains '..'
#endif

#include "SpvTools.h"
#include "glslang/Include/intermediate.h"

#include <string>
#include <vector>

#include "Logger.h"

namespace glslang {

void GetSpirvVersion(std::string&);
int GetSpirvGeneratorVersion();
void GlslangToSpv(const glslang::TIntermediate& intermediate, std::vector<unsigned int>& spirv,
                  SpvOptions* options = nullptr);
void GlslangToSpv(const glslang::TIntermediate& intermediate, std::vector<unsigned int>& spirv,
                  spv::SpvBuildLogger* logger, SpvOptions* options = nullptr);
bool OutputSpvBin(const std::vector<unsigned int>& spirv, const char* baseName);
bool OutputSpvHex(const std::vector<unsigned int>& spirv, const char* baseName, const char* varName);

}
