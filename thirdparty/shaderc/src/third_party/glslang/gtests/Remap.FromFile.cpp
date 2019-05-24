//
// Copyright (C) 2016 LunarG, Inc.
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
//    Neither the name of Google Inc. nor the names of its
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

#include <gtest/gtest.h>

#include "TestFixture.h"

namespace glslangtest {
namespace {

struct RemapTestArgs {
    const char*  fileName;
    const char*  entryPoint;
    Source       sourceLanguage;
    unsigned int remapOpts;
};

// We are using FileNameEntryPointPair objects as parameters for instantiating
// the template, so the global FileNameAsCustomTestSuffix() won't work since
// it assumes std::string as parameters. Thus, an overriding one here.
std::string FileNameAsCustomTestSuffix(
    const ::testing::TestParamInfo<RemapTestArgs>& info) {
    std::string name = info.param.fileName;
    // A valid test case suffix cannot have '.' and '-' inside.
    std::replace(name.begin(), name.end(), '.', '_');
    std::replace(name.begin(), name.end(), '-', '_');
    return name;
}

using RemapTest = GlslangTest<::testing::TestWithParam<RemapTestArgs>>;

// Remapping SPIR-V modules.
TEST_P(RemapTest, FromFile)
{
    if (GetSuffix(GetParam().fileName) == "spv") {
        loadFileRemapAndCheck(GlobalTestSettings.testRoot, GetParam().fileName,
                              GetParam().sourceLanguage,
                              Semantics::Vulkan,
                              Target::Spv,
                              GetParam().remapOpts);
    } else {
        loadFileCompileRemapAndCheck(GlobalTestSettings.testRoot, GetParam().fileName,
                                     GetParam().sourceLanguage,
                                     Semantics::Vulkan,
                                     Target::Spv,
                                     GetParam().entryPoint,
                                     GetParam().remapOpts);
    }
}

// clang-format off
INSTANTIATE_TEST_CASE_P(
    ToSpirv, RemapTest,
    ::testing::ValuesIn(std::vector<RemapTestArgs>{
            // GLSL remapper tests
            // testname                                   entry   language      remapper_options
            { "remap.basic.none.frag",                    "main", Source::GLSL, spv::spirvbin_t::NONE },
            { "remap.basic.everything.frag",              "main", Source::GLSL, spv::spirvbin_t::DO_EVERYTHING },
            { "remap.basic.dcefunc.frag",                 "main", Source::GLSL, spv::spirvbin_t::DCE_FUNCS },
            { "remap.basic.strip.frag",                   "main", Source::GLSL, spv::spirvbin_t::STRIP },
            { "remap.specconst.comp",                     "main", Source::GLSL, spv::spirvbin_t::DO_EVERYTHING },
            { "remap.switch.none.frag",                   "main", Source::GLSL, spv::spirvbin_t::NONE },
            { "remap.switch.everything.frag",             "main", Source::GLSL, spv::spirvbin_t::DO_EVERYTHING },
            { "remap.literal64.none.spv",                 "main", Source::GLSL, spv::spirvbin_t::NONE },
            { "remap.literal64.everything.spv",           "main", Source::GLSL, spv::spirvbin_t::DO_EVERYTHING },
            { "remap.if.none.frag",                       "main", Source::GLSL, spv::spirvbin_t::NONE },
            { "remap.if.everything.frag",                 "main", Source::GLSL, spv::spirvbin_t::DO_EVERYTHING },
            { "remap.similar_1a.none.frag",               "main", Source::GLSL, spv::spirvbin_t::NONE },
            { "remap.similar_1b.none.frag",               "main", Source::GLSL, spv::spirvbin_t::NONE },
            { "remap.similar_1a.everything.frag",         "main", Source::GLSL, spv::spirvbin_t::DO_EVERYTHING },
            { "remap.similar_1b.everything.frag",         "main", Source::GLSL, spv::spirvbin_t::DO_EVERYTHING },
            { "remap.uniformarray.none.frag",             "main", Source::GLSL, spv::spirvbin_t::NONE },
            { "remap.uniformarray.everything.frag",       "main", Source::GLSL, spv::spirvbin_t::DO_EVERYTHING },

            // HLSL remapper tests
            { "remap.hlsl.sample.basic.strip.frag",       "main", Source::HLSL, spv::spirvbin_t::STRIP },
            { "remap.hlsl.sample.basic.everything.frag",  "main", Source::HLSL, spv::spirvbin_t::DO_EVERYTHING },
            { "remap.hlsl.sample.basic.none.frag",        "main", Source::HLSL, spv::spirvbin_t::NONE },
            { "remap.hlsl.templatetypes.none.frag",       "main", Source::HLSL, spv::spirvbin_t::NONE },
            { "remap.hlsl.templatetypes.everything.frag", "main", Source::HLSL, spv::spirvbin_t::DO_EVERYTHING },
            }),
    FileNameAsCustomTestSuffix
);
// clang-format on

}  // anonymous namespace
}  // namespace glslangtest
