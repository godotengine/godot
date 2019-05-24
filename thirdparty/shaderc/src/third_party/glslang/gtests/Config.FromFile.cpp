//
// Copyright (C) 2016 Google, Inc.
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

#include "StandAlone/ResourceLimits.h"
#include "TestFixture.h"

namespace glslangtest {
namespace {

struct TestCaseSpec {
    std::string input;
    std::string config;
    std::string output;
    EShMessages controls;
};

using ConfigTest = GlslangTest<::testing::TestWithParam<TestCaseSpec>>;

TEST_P(ConfigTest, FromFile)
{
    TestCaseSpec testCase = GetParam();
    GlslangResult result;
    result.validationResult = true;

    // Get the contents for input shader and limit configurations.
    std::string shaderContents, configContents;
    tryLoadFile(GlobalTestSettings.testRoot + "/" + testCase.input, "input", &shaderContents);
    tryLoadFile(GlobalTestSettings.testRoot + "/" + testCase.config, "limits config", &configContents);

    // Decode limit configurations.
    TBuiltInResource resources = {};
    {
        const size_t len = configContents.size();
        char* configChars = new char[len + 1];
        memcpy(configChars, configContents.data(), len);
        configChars[len] = 0;
        glslang::DecodeResourceLimits(&resources, configChars);
        delete[] configChars;
    }

    // Compile the shader.
    glslang::TShader shader(GetShaderStage(GetSuffix(testCase.input)));
    compile(&shader, shaderContents, "", testCase.controls, &resources);
    result.shaderResults.push_back(
        {testCase.input, shader.getInfoLog(), shader.getInfoDebugLog()});

    // Link the shader.
    glslang::TProgram program;
    program.addShader(&shader);
    program.link(testCase.controls);
    result.linkingOutput = program.getInfoLog();
    result.linkingError = program.getInfoDebugLog();

    std::ostringstream stream;
    outputResultToStream(&stream, result, testCase.controls);

    // Check with expected results.
    const std::string expectedOutputFname =
        GlobalTestSettings.testRoot + "/baseResults/" + testCase.output;
    std::string expectedOutput;
    tryLoadFile(expectedOutputFname, "expected output", &expectedOutput);

    checkEqAndUpdateIfRequested(expectedOutput, stream.str(), expectedOutputFname);
}

// clang-format off
INSTANTIATE_TEST_CASE_P(
    Glsl, ConfigTest,
    ::testing::ValuesIn(std::vector<TestCaseSpec>({
        {"specExamples.vert", "baseResults/test.conf", "specExamplesConf.vert.out", (EShMessages)(EShMsgAST | EShMsgCascadingErrors)},
        {"100Limits.vert", "100.conf", "100LimitsConf.vert.out", EShMsgCascadingErrors},
    }))
);
// clang-format on

}  // anonymous namespace
}  // namespace glslangtest
