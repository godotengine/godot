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

#include "TestFixture.h"

namespace glslangtest {

std::string FileNameAsCustomTestSuffix(
    const ::testing::TestParamInfo<std::string>& info)
{
    std::string name = info.param;
    // A valid test case suffix cannot have '.' and '-' inside.
    std::replace(name.begin(), name.end(), '.', '_');
    std::replace(name.begin(), name.end(), '-', '_');
    return name;
}

EShLanguage GetShaderStage(const std::string& stage)
{
    if (stage == "vert") {
        return EShLangVertex;
    } else if (stage == "tesc") {
        return EShLangTessControl;
    } else if (stage == "tese") {
        return EShLangTessEvaluation;
    } else if (stage == "geom") {
        return EShLangGeometry;
    } else if (stage == "frag") {
        return EShLangFragment;
    } else if (stage == "comp") {
        return EShLangCompute;
#ifdef NV_EXTENSIONS
    } else if (stage == "rgen") {
        return EShLangRayGenNV;
    } else if (stage == "rint") {
        return EShLangIntersectNV;
    } else if (stage == "rahit") {
        return EShLangAnyHitNV;
    } else if (stage == "rchit") {
        return EShLangClosestHitNV;
    } else if (stage == "rmiss") {
        return EShLangMissNV;
    } else if (stage == "rcall") {
        return EShLangCallableNV;
    } else if (stage == "task") {
        return EShLangTaskNV;
    } else if (stage == "mesh") {
        return EShLangMeshNV;
#endif
    } else {
        assert(0 && "Unknown shader stage");
        return EShLangCount;
    }
}

EShMessages DeriveOptions(Source source, Semantics semantics, Target target)
{
    EShMessages result = EShMsgCascadingErrors;

    switch (source) {
        case Source::GLSL:
            break;
        case Source::HLSL:
            result = static_cast<EShMessages>(result | EShMsgReadHlsl);
            break;
    }

    switch (target) {
        case Target::AST:
            result = static_cast<EShMessages>(result | EShMsgAST);
            break;
        case Target::Spv:
            result = static_cast<EShMessages>(result | EShMsgSpvRules);
            result = static_cast<EShMessages>(result | EShMsgKeepUncalled);
            break;
        case Target::BothASTAndSpv:
            result = static_cast<EShMessages>(result | EShMsgSpvRules | EShMsgAST);
            result = static_cast<EShMessages>(result | EShMsgKeepUncalled);
            break;
    };

    switch (semantics) {
        case Semantics::OpenGL:
            break;
        case Semantics::Vulkan:
            result = static_cast<EShMessages>(result | EShMsgVulkanRules | EShMsgSpvRules);
            break;
    }

    result = static_cast<EShMessages>(result | EShMsgHlslLegalization);

    return result;
}

std::pair<bool, std::string> ReadFile(const std::string& path)
{
    std::ifstream fstream(path, std::ios::in);
    if (fstream) {
        std::string contents;
        fstream.seekg(0, std::ios::end);
        contents.reserve((std::string::size_type)fstream.tellg());
        fstream.seekg(0, std::ios::beg);
        contents.assign((std::istreambuf_iterator<char>(fstream)),
                        std::istreambuf_iterator<char>());
        return std::make_pair(true, contents);
    }
    return std::make_pair(false, "");
}

std::pair<bool, std::vector<std::uint32_t> > ReadSpvBinaryFile(const std::string& path)
{
    std::ifstream fstream(path, std::fstream::in | std::fstream::binary);

    if (!fstream)
        return std::make_pair(false, std::vector<std::uint32_t>());

    std::vector<std::uint32_t> contents;

    // Reserve space (for efficiency, not for correctness)
    fstream.seekg(0, fstream.end);
    contents.reserve(size_t(fstream.tellg()) / sizeof(std::uint32_t));
    fstream.seekg(0, fstream.beg);

    // There is no istream iterator traversing by uint32_t, so we must loop.
    while (!fstream.eof()) {
        std::uint32_t inWord;
        fstream.read((char *)&inWord, sizeof(inWord));

        if (!fstream.eof())
            contents.push_back(inWord);
    }

    return std::make_pair(true, contents); // hopefully, c++11 move semantics optimizes the copy away.
}

bool WriteFile(const std::string& path, const std::string& contents)
{
    std::ofstream fstream(path, std::ios::out);
    if (!fstream) return false;
    fstream << contents;
    fstream.flush();
    return true;
}

std::string GetSuffix(const std::string& name)
{
    const size_t pos = name.rfind('.');
    return (pos == std::string::npos) ? "" : name.substr(name.rfind('.') + 1);
}

}  // namespace glslangtest
