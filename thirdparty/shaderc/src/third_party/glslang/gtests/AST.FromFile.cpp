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

#include <gtest/gtest.h>

#include "TestFixture.h"

namespace glslangtest {
namespace {

using CompileToAstTest = GlslangTest<::testing::TestWithParam<std::string>>;

#ifdef NV_EXTENSIONS
using CompileToAstTestNV = GlslangTest<::testing::TestWithParam<std::string>>;
#endif

TEST_P(CompileToAstTest, FromFile)
{
    loadFileCompileAndCheck(GlobalTestSettings.testRoot, GetParam(),
                            Source::GLSL, Semantics::OpenGL, glslang::EShTargetVulkan_1_0,
                            Target::AST);
}

#ifdef NV_EXTENSIONS
// Compiling GLSL to SPIR-V under OpenGL semantics (NV extensions enabled).
TEST_P(CompileToAstTestNV, FromFile)
{
    loadFileCompileAndCheck(GlobalTestSettings.testRoot, GetParam(),
                            Source::GLSL, Semantics::OpenGL, glslang::EShTargetVulkan_1_0,
                            Target::AST);
}
#endif

// clang-format off
INSTANTIATE_TEST_CASE_P(
    Glsl, CompileToAstTest,
    ::testing::ValuesIn(std::vector<std::string>({
        "sample.frag",
        "sample.vert",
        "decls.frag",
        "specExamples.frag",
        "specExamples.vert",
        "versionsClean.frag",
        "versionsClean.vert",
        "versionsErrors.frag",
        "versionsErrors.vert",
        "100.frag",
        "100samplerExternal.frag",
        "120.vert",
        "120.frag",
        "130.vert",
        "130.frag",
        "140.vert",
        "140.frag",
        "150.vert",
        "150.geom",
        "150.frag",
        "precision.frag",
        "precision.vert",
        "nonSquare.vert",
        "matrixError.vert",
        "cppSimple.vert",
        "cppIndent.vert",
        "cppIntMinOverNegativeOne.frag",
        "cppMerge.frag",
        "cppNest.vert",
        "cppBad.vert",
        "cppBad2.vert",
        "cppBad3.vert",
        "cppBad4.vert",
        "cppBad5.vert",
        "cppComplexExpr.vert",
        "cppDeepNest.frag",
        "cppPassMacroName.frag",
        "cppRelaxSkipTokensErrors.vert",
        "badChars.frag",
        "pointCoord.frag",
        "array.frag",
        "array100.frag",
        "comment.frag",
        "300.vert",
        "300.frag",
        "300BuiltIns.frag",
        "300layout.vert",
        "300layout.frag",
        "300operations.frag",
        "300block.frag",
        "300samplerExternal.frag",
        "300samplerExternalYUV.frag",
        "310.comp",
        "310.vert",
        "310.geom",
        "310.frag",
        "310.tesc",
        "310.tese",
        "310implicitSizeArrayError.vert",
        "310AofA.vert",
        "310runtimeArray.vert",
        "320.comp",
        "320.vert",
        "320.geom",
        "320.frag",
        "320.tesc",
        "320.tese",
        "330.frag",
        "330comp.frag",
        "constErrors.frag",
        "constFold.frag",
        "constFoldIntMin.frag",
        "errors.frag",
        "forwardRef.frag",
        "uint.frag",
        "switch.frag",
        "tokenLength.vert",
        "100Limits.vert",
        "100scope.vert",
        "110scope.vert",
        "300scope.vert",
        "400.frag",
        "400.vert",
        "410.vert",
        "420.comp",
        "420.frag",
        "420.vert",
        "420.geom",
        "420_size_gl_in.geom",
        "430scope.vert",
        "lineContinuation100.vert",
        "lineContinuation.vert",
        "numeral.frag",
        "400.geom",
        "400.tesc",
        "400.tese",
        "410.tesc",
        "420.tesc",
        "420.tese",
        "410.geom",
        "430.vert",
        "430.comp",
        "430AofA.frag",
        "435.vert",
        "440.vert",
        "440.frag",
        "450.vert",
        "450.geom",
        "450.tesc",
        "450.tese",
        "450.frag",
        "450.comp",
        "460.frag",
        "460.vert",
        "dce.frag",
        "atomic_uint.frag",
        "implicitInnerAtomicUint.frag",
        "aggOps.frag",
        "always-discard.frag",
        "always-discard2.frag",
        "conditionalDiscard.frag",
        "conversion.frag",
        "dataOut.frag",
        "dataOutIndirect.frag",
        "deepRvalue.frag",
        "depthOut.frag",
        "discard-dce.frag",
        "doWhileLoop.frag",
        "earlyReturnDiscard.frag",
        "flowControl.frag",
        "forLoop.frag",
        "functionCall.frag",
        "functionSemantics.frag",
        "length.frag",
        "localAggregates.frag",
        "loops.frag",
        "loopsArtificial.frag",
        "matrix.frag",
        "matrix2.frag",
        "mixedArrayDecls.frag",
        "nonuniform.frag",
        "newTexture.frag",
        "Operations.frag",
        "overlongLiteral.frag",
        "prepost.frag",
        "runtimeArray.vert",
        "simpleFunctionCall.frag",
        "stringToDouble.vert",
        "structAssignment.frag",
        "structDeref.frag",
        "structure.frag",
        "swizzle.frag",
        "invalidSwizzle.vert",
        "syntaxError.frag",
        "test.frag",
        "texture.frag",
        "tokenPaste.vert",
        "types.frag",
        "uniformArray.frag",
        "variableArrayIndex.frag",
        "varyingArray.frag",
        "varyingArrayIndirect.frag",
        "voidFunction.frag",
        "whileLoop.frag",
        "nonVulkan.frag",
        "negativeArraySize.comp",
        "precise.tesc",
        "precise_struct_block.vert",
        "maxClipDistances.vert",
        "findFunction.frag",
        "constantUnaryConversion.comp",
        "glsl.450.subgroup.frag",
        "glsl.450.subgroup.geom",
        "glsl.450.subgroup.tesc",
        "glsl.450.subgroup.tese",
        "glsl.450.subgroup.vert",
        "glsl.450.subgroupArithmetic.comp",
        "glsl.450.subgroupBasic.comp",
        "glsl.450.subgroupBallot.comp",
        "glsl.450.subgroupBallotNeg.comp",
        "glsl.450.subgroupClustered.comp",
        "glsl.450.subgroupClusteredNeg.comp",
        "glsl.450.subgroupPartitioned.comp",
        "glsl.450.subgroupShuffle.comp",
        "glsl.450.subgroupShuffleRelative.comp",
        "glsl.450.subgroupQuad.comp",
        "glsl.450.subgroupVote.comp",
        "glsl.es320.subgroup.frag",
        "glsl.es320.subgroup.geom",
        "glsl.es320.subgroup.tesc",
        "glsl.es320.subgroup.tese",
        "glsl.es320.subgroup.vert",
        "glsl.es320.subgroupArithmetic.comp",
        "glsl.es320.subgroupBasic.comp",
        "glsl.es320.subgroupBallot.comp",
        "glsl.es320.subgroupBallotNeg.comp",
        "glsl.es320.subgroupClustered.comp",
        "glsl.es320.subgroupClusteredNeg.comp",
        "glsl.es320.subgroupPartitioned.comp",
        "glsl.es320.subgroupShuffle.comp",
        "glsl.es320.subgroupShuffleRelative.comp",
        "glsl.es320.subgroupQuad.comp",
        "glsl.es320.subgroupVote.comp",
    })),
    FileNameAsCustomTestSuffix
);

#ifdef NV_EXTENSIONS
INSTANTIATE_TEST_CASE_P(
    Glsl, CompileToAstTestNV,
    ::testing::ValuesIn(std::vector<std::string>({
        "nvShaderNoperspectiveInterpolation.frag",
    })),
    FileNameAsCustomTestSuffix
);
#endif
// clang-format on

}  // anonymous namespace
}  // namespace glslangtest
