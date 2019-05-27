//
// Copyright (C) 2016-2017 Google, Inc.
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

#include <memory>

#include <gtest/gtest.h>

#include "TestFixture.h"

namespace glslangtest {
namespace {

using LinkTestVulkan = GlslangTest<
    ::testing::TestWithParam<std::vector<std::string>>>;

TEST_P(LinkTestVulkan, FromFile)
{
    const auto& fileNames = GetParam();
    const size_t fileCount = fileNames.size();
    const EShMessages controls = DeriveOptions(Source::GLSL, Semantics::Vulkan, Target::AST);
    GlslangResult result;

    // Compile each input shader file.
    bool success = true;
    std::vector<std::unique_ptr<glslang::TShader>> shaders;
    for (size_t i = 0; i < fileCount; ++i) {
        std::string contents;
        tryLoadFile(GlobalTestSettings.testRoot + "/" + fileNames[i],
                    "input", &contents);
        shaders.emplace_back(
                new glslang::TShader(GetShaderStage(GetSuffix(fileNames[i]))));
        auto* shader = shaders.back().get();
        shader->setAutoMapLocations(true);
        success &= compile(shader, contents, "", controls);
        result.shaderResults.push_back(
            {fileNames[i], shader->getInfoLog(), shader->getInfoDebugLog()});
    }

    // Link all of them.
    glslang::TProgram program;
    for (const auto& shader : shaders) program.addShader(shader.get());
    success &= program.link(controls);
    result.linkingOutput = program.getInfoLog();
    result.linkingError = program.getInfoDebugLog();

    if (success && (controls & EShMsgSpvRules)) {
        spv::SpvBuildLogger logger;
        std::vector<uint32_t> spirv_binary;
        options().disableOptimizer = true;
        glslang::GlslangToSpv(*program.getIntermediate(shaders.front()->getStage()),
                                spirv_binary, &logger, &options());

        std::ostringstream disassembly_stream;
        spv::Parameterize();
        spv::Disassemble(disassembly_stream, spirv_binary);
        result.spirvWarningsErrors = logger.getAllMessages();
        result.spirv = disassembly_stream.str();
        result.validationResult = !options().validate || logger.getAllMessages().empty();
    }

    std::ostringstream stream;
    outputResultToStream(&stream, result, controls);

    // Check with expected results.
    const std::string expectedOutputFname =
        GlobalTestSettings.testRoot + "/baseResults/" + fileNames.front() + ".out";
    std::string expectedOutput;
    tryLoadFile(expectedOutputFname, "expected output", &expectedOutput);

    checkEqAndUpdateIfRequested(expectedOutput, stream.str(), expectedOutputFname,
                                result.spirvWarningsErrors);
}

// clang-format off
INSTANTIATE_TEST_CASE_P(
    Glsl, LinkTestVulkan,
    ::testing::ValuesIn(std::vector<std::vector<std::string>>({
        {"link1.vk.frag", "link2.vk.frag"},
        {"spv.unit1.frag", "spv.unit2.frag", "spv.unit3.frag"},
    }))
);
// clang-format on

}  // anonymous namespace
}  // namespace glslangtest
