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

#include <algorithm>

#include <gtest/gtest.h>

#include "TestFixture.h"

namespace glslangtest {
namespace {

struct IoMapData {
    const char* fileName;
    const char* entryPoint;
    int baseSamplerBinding;
    int baseTextureBinding;
    int baseImageBinding;
    int baseUboBinding;
    int baseSsboBinding;
    bool autoMapBindings;
    bool flattenUniforms;
};

std::string FileNameAsCustomTestSuffixIoMap(
    const ::testing::TestParamInfo<IoMapData>& info) {
    std::string name = info.param.fileName;
    // A valid test case suffix cannot have '.' and '-' inside.
    std::replace(name.begin(), name.end(), '.', '_');
    std::replace(name.begin(), name.end(), '-', '_');
    return name;
}

using CompileVulkanToSpirvTest = GlslangTest<::testing::TestWithParam<std::string>>;
using CompileVulkanToDebugSpirvTest = GlslangTest<::testing::TestWithParam<std::string>>;
using CompileVulkan1_1ToSpirvTest = GlslangTest<::testing::TestWithParam<std::string>>;
using CompileOpenGLToSpirvTest = GlslangTest<::testing::TestWithParam<std::string>>;
using VulkanSemantics = GlslangTest<::testing::TestWithParam<std::string>>;
using OpenGLSemantics = GlslangTest<::testing::TestWithParam<std::string>>;
using VulkanAstSemantics = GlslangTest<::testing::TestWithParam<std::string>>;
using HlslIoMap = GlslangTest<::testing::TestWithParam<IoMapData>>;
using GlslIoMap = GlslangTest<::testing::TestWithParam<IoMapData>>;
#ifdef AMD_EXTENSIONS
using CompileVulkanToSpirvTestAMD = GlslangTest<::testing::TestWithParam<std::string>>;
#endif
#ifdef NV_EXTENSIONS
using CompileVulkanToSpirvTestNV = GlslangTest<::testing::TestWithParam<std::string>>;
#endif
using CompileUpgradeTextureToSampledTextureAndDropSamplersTest = GlslangTest<::testing::TestWithParam<std::string>>;

// Compiling GLSL to SPIR-V under Vulkan semantics. Expected to successfully
// generate SPIR-V.
TEST_P(CompileVulkanToSpirvTest, FromFile)
{
    loadFileCompileAndCheck(GlobalTestSettings.testRoot, GetParam(),
                            Source::GLSL, Semantics::Vulkan, glslang::EShTargetVulkan_1_0,
                            Target::Spv);
}

// Compiling GLSL to SPIR-V with debug info under Vulkan semantics. Expected
// to successfully generate SPIR-V.
TEST_P(CompileVulkanToDebugSpirvTest, FromFile)
{
    loadFileCompileAndCheck(GlobalTestSettings.testRoot, GetParam(),
                            Source::GLSL, Semantics::Vulkan,
                            glslang::EShTargetVulkan_1_0,
                            Target::Spv, true, "",
                            "/baseResults/", false, true);
}

TEST_P(CompileVulkan1_1ToSpirvTest, FromFile)
{
    loadFileCompileAndCheck(GlobalTestSettings.testRoot, GetParam(),
                            Source::GLSL, Semantics::Vulkan, glslang::EShTargetVulkan_1_1,
                            Target::Spv);
}

// Compiling GLSL to SPIR-V under OpenGL semantics. Expected to successfully
// generate SPIR-V.
TEST_P(CompileOpenGLToSpirvTest, FromFile)
{
    loadFileCompileAndCheck(GlobalTestSettings.testRoot, GetParam(),
                            Source::GLSL, Semantics::OpenGL, glslang::EShTargetVulkan_1_0,
                            Target::Spv);
}

// GLSL-level Vulkan semantics test. Expected to error out before generating
// SPIR-V.
TEST_P(VulkanSemantics, FromFile)
{
    loadFileCompileAndCheck(GlobalTestSettings.testRoot, GetParam(),
                            Source::GLSL, Semantics::Vulkan, glslang::EShTargetVulkan_1_0,
                            Target::Spv, false);
}

// GLSL-level Vulkan semantics test. Expected to error out before generating
// SPIR-V.
TEST_P(OpenGLSemantics, FromFile)
{
    loadFileCompileAndCheck(GlobalTestSettings.testRoot, GetParam(),
                            Source::GLSL, Semantics::OpenGL, glslang::EShTargetVulkan_1_0,
                            Target::Spv, false);
}

// GLSL-level Vulkan semantics test that need to see the AST for validation.
TEST_P(VulkanAstSemantics, FromFile)
{
    loadFileCompileAndCheck(GlobalTestSettings.testRoot, GetParam(),
                            Source::GLSL, Semantics::Vulkan, glslang::EShTargetVulkan_1_0,
                            Target::AST);
}

// HLSL-level Vulkan semantics tests.
TEST_P(HlslIoMap, FromFile)
{
    loadFileCompileIoMapAndCheck(GlobalTestSettings.testRoot, GetParam().fileName,
                                 Source::HLSL, Semantics::Vulkan,
                                 Target::Spv, GetParam().entryPoint,
                                 GetParam().baseSamplerBinding,
                                 GetParam().baseTextureBinding,
                                 GetParam().baseImageBinding,
                                 GetParam().baseUboBinding,
                                 GetParam().baseSsboBinding,
                                 GetParam().autoMapBindings,
                                 GetParam().flattenUniforms);
}

// GLSL-level Vulkan semantics tests.
TEST_P(GlslIoMap, FromFile)
{
    loadFileCompileIoMapAndCheck(GlobalTestSettings.testRoot, GetParam().fileName,
                                 Source::GLSL, Semantics::Vulkan,
                                 Target::Spv, GetParam().entryPoint,
                                 GetParam().baseSamplerBinding,
                                 GetParam().baseTextureBinding,
                                 GetParam().baseImageBinding,
                                 GetParam().baseUboBinding,
                                 GetParam().baseSsboBinding,
                                 GetParam().autoMapBindings,
                                 GetParam().flattenUniforms);
}

#ifdef AMD_EXTENSIONS
// Compiling GLSL to SPIR-V under Vulkan semantics (AMD extensions enabled).
// Expected to successfully generate SPIR-V.
TEST_P(CompileVulkanToSpirvTestAMD, FromFile)
{
    loadFileCompileAndCheck(GlobalTestSettings.testRoot, GetParam(),
                            Source::GLSL, Semantics::Vulkan, glslang::EShTargetVulkan_1_0,
                            Target::Spv);
}
#endif

#ifdef NV_EXTENSIONS
// Compiling GLSL to SPIR-V under Vulkan semantics (NV extensions enabled).
// Expected to successfully generate SPIR-V.
TEST_P(CompileVulkanToSpirvTestNV, FromFile)
{
    loadFileCompileAndCheck(GlobalTestSettings.testRoot, GetParam(),
                            Source::GLSL, Semantics::Vulkan, glslang::EShTargetVulkan_1_0,
                            Target::Spv);
}
#endif

TEST_P(CompileUpgradeTextureToSampledTextureAndDropSamplersTest, FromFile)
{
    loadCompileUpgradeTextureToSampledTextureAndDropSamplersAndCheck(GlobalTestSettings.testRoot,
                                                                     GetParam(),
                                                                     Source::GLSL,
                                                                     Semantics::Vulkan,
                                                                     Target::Spv);
}

// clang-format off
INSTANTIATE_TEST_CASE_P(
    Glsl, CompileVulkanToSpirvTest,
    ::testing::ValuesIn(std::vector<std::string>({
        // Test looping constructs.
        // No tests yet for making sure break and continue from a nested loop
        // goes to the innermost target.
        "spv.barrier.vert",
        "spv.do-simple.vert",
        "spv.do-while-continue-break.vert",
        "spv.for-complex-condition.vert",
        "spv.for-continue-break.vert",
        "spv.for-simple.vert",
        "spv.for-notest.vert",
        "spv.for-nobody.vert",
        "spv.while-continue-break.vert",
        "spv.while-simple.vert",
        // vulkan-specific tests
        "spv.set.vert",
        "spv.double.comp",
        "spv.100ops.frag",
        "spv.130.frag",
        "spv.140.frag",
        "spv.150.geom",
        "spv.150.vert",
        "spv.16bitstorage.frag",
        "spv.16bitstorage_Error.frag",
        "spv.16bitstorage-int.frag",
        "spv.16bitstorage_Error-int.frag",
        "spv.16bitstorage-uint.frag",
        "spv.16bitstorage_Error-uint.frag",
        "spv.300BuiltIns.vert",
        "spv.300layout.frag",
        "spv.300layout.vert",
        "spv.300layoutp.vert",
        "spv.310.comp",
        "spv.310.bitcast.frag",
        "spv.330.geom",
        "spv.400.frag",
        "spv.400.tesc",
        "spv.400.tese",
        "spv.420.geom",
        "spv.430.frag",
        "spv.430.vert",
        "spv.450.tesc",
        "spv.450.geom",
        "spv.450.noRedecl.tesc",
        "spv.8bitstorage-int.frag",
        "spv.8bitstorage_Error-int.frag",
        "spv.8bitstorage-uint.frag",
        "spv.8bitstorage_Error-uint.frag",
        "spv.8bitstorage-ubo.vert",
        "spv.8bitstorage-ssbo.vert",
        "spv.accessChain.frag",
        "spv.aggOps.frag",
        "spv.always-discard.frag",
        "spv.always-discard2.frag",
        "spv.arbPostDepthCoverage.frag",
        "spv.arbPostDepthCoverage_Error.frag",
        "spv.bitCast.frag",
        "spv.bool.vert",
        "spv.boolInBlock.frag",
        "spv.branch-return.vert",
        "spv.bufferhandle1.frag",
        "spv.bufferhandle10.frag",
        "spv.bufferhandle11.frag",
        "spv.bufferhandle12.frag",
        "spv.bufferhandle13.frag",
        "spv.bufferhandle14.frag",
        "spv.bufferhandle15.frag",
        "spv.bufferhandle16.frag",
        "spv.bufferhandle17_Errors.frag",
        "spv.bufferhandle2.frag",
        "spv.bufferhandle3.frag",
        "spv.bufferhandle4.frag",
        "spv.bufferhandle5.frag",
        "spv.bufferhandle6.frag",
        "spv.bufferhandle7.frag",
        "spv.bufferhandle8.frag",
        "spv.bufferhandle9.frag",
        "spv.bufferhandle_Error.frag",
        "spv.builtInXFB.vert",
        "spv.conditionalDiscard.frag",
        "spv.constStruct.vert",
        "spv.controlFlowAttributes.frag",
        "spv.conversion.frag",
        "spv.coopmat.comp",
        "spv.coopmat_Error.comp",
        "spv.dataOut.frag",
        "spv.dataOutIndirect.frag",
        "spv.dataOutIndirect.vert",
        "spv.deepRvalue.frag",
        "spv.depthOut.frag",
        "spv.discard-dce.frag",
        "spv.doWhileLoop.frag",
        "spv.earlyReturnDiscard.frag",
        "spv.extPostDepthCoverage.frag",
        "spv.extPostDepthCoverage_Error.frag",
        "spv.flowControl.frag",
        "spv.forLoop.frag",
        "spv.forwardFun.frag",
        "spv.fragmentDensity.frag",
        "spv.fragmentDensity.vert",
        "spv.fragmentDensity-es.frag",
        "spv.fragmentDensity-neg.frag",
        "spv.fullyCovered.frag",
        "spv.functionCall.frag",
        "spv.functionNestedOpaque.vert",
        "spv.functionSemantics.frag",
        "spv.GeometryShaderPassthrough.geom",
        "spv.interpOps.frag",
        "spv.int64.frag",
        "spv.intOps.vert",
        "spv.layoutNested.vert",
        "spv.length.frag",
        "spv.localAggregates.frag",
        "spv.loops.frag",
        "spv.loopsArtificial.frag",
        "spv.matFun.vert",
        "spv.matrix.frag",
        "spv.matrix2.frag",
        "spv.memoryQualifier.frag",
        "spv.merge-unreachable.frag",
        "spv.multiStruct.comp",
        "spv.multiStructFuncall.frag",
        "spv.newTexture.frag",
        "spv.noDeadDecorations.vert",
        "spv.nonSquare.vert",
        "spv.nonuniform.frag",
        "spv.noWorkgroup.comp",
        "spv.offsets.frag",
        "spv.Operations.frag",
        "spv.paramMemory.frag",
        "spv.precision.frag",
        "spv.precisionNonESSamp.frag",
        "spv.prepost.frag",
        "spv.qualifiers.vert",
        "spv.sample.frag",
        "spv.sampleId.frag",
        "spv.samplePosition.frag",
        "spv.sampleMaskOverrideCoverage.frag",
        "spv.scalarlayout.frag",
        "spv.scalarlayoutfloat16.frag",
        "spv.shaderBallot.comp",
        "spv.shaderDrawParams.vert",
        "spv.shaderGroupVote.comp",
        "spv.shaderStencilExport.frag",
        "spv.shiftOps.frag",
        "spv.simpleFunctionCall.frag",
        "spv.simpleMat.vert",
        "spv.sparseTexture.frag",
        "spv.sparseTextureClamp.frag",
        "spv.structAssignment.frag",
        "spv.structDeref.frag",
        "spv.structure.frag",
        "spv.switch.frag",
        "spv.swizzle.frag",
        "spv.swizzleInversion.frag",
        "spv.test.frag",
        "spv.test.vert",
        "spv.texture.frag",
        "spv.texture.vert",
        "spv.textureBuffer.vert",
        "spv.image.frag",
        "spv.types.frag",
        "spv.uint.frag",
        "spv.uniformArray.frag",
        "spv.variableArrayIndex.frag",
        "spv.varyingArray.frag",
        "spv.varyingArrayIndirect.frag",
        "spv.vecMatConstruct.frag",
        "spv.voidFunction.frag",
        "spv.whileLoop.frag",
        "spv.AofA.frag",
        "spv.queryL.frag",
        "spv.separate.frag",
        "spv.shortCircuit.frag",
        "spv.pushConstant.vert",
        "spv.pushConstantAnon.vert",
        "spv.subpass.frag",
        "spv.specConstant.vert",
        "spv.specConstant.comp",
        "spv.specConstantComposite.vert",
        "spv.specConstantOperations.vert",
        "spv.storageBuffer.vert",
        "spv.precise.tese",
        "spv.precise.tesc",
        "spv.vulkan100.subgroupArithmetic.comp",
        "spv.vulkan100.subgroupPartitioned.comp",
        "spv.xfb.vert",
        "spv.xfb2.vert",
        "spv.xfb3.vert",
        "spv.samplerlessTextureFunctions.frag",
    })),
    FileNameAsCustomTestSuffix
);

// clang-format off
INSTANTIATE_TEST_CASE_P(
    Glsl, CompileVulkanToDebugSpirvTest,
    ::testing::ValuesIn(std::vector<std::string>({
        "spv.pp.line.frag",
    })),
    FileNameAsCustomTestSuffix
);

// clang-format off
INSTANTIATE_TEST_CASE_P(
    Glsl, CompileVulkan1_1ToSpirvTest,
    ::testing::ValuesIn(std::vector<std::string>({
        "spv.1.3.8bitstorage-ubo.vert",
        "spv.1.3.8bitstorage-ssbo.vert",
        "spv.1.3.coopmat.comp",
        "spv.deviceGroup.frag",
        "spv.drawParams.vert",
        "spv.int8.frag",
        "spv.vulkan110.int16.frag",
        "spv.int32.frag",
        "spv.explicittypes.frag",
        "spv.float32.frag",
        "spv.float64.frag",
        "spv.memoryScopeSemantics.comp",
        "spv.memoryScopeSemantics_Error.comp",
        "spv.multiView.frag",
        "spv.RayGenShader11.rgen",
        "spv.subgroup.frag",
        "spv.subgroup.geom",
        "spv.subgroup.tesc",
        "spv.subgroup.tese",
        "spv.subgroup.vert",
        "spv.subgroupArithmetic.comp",
        "spv.subgroupBasic.comp",
        "spv.subgroupBallot.comp",
        "spv.subgroupBallotNeg.comp",
        "spv.subgroupClustered.comp",
        "spv.subgroupClusteredNeg.comp",
        "spv.subgroupPartitioned.comp",
        "spv.subgroupShuffle.comp",
        "spv.subgroupShuffleRelative.comp",
        "spv.subgroupQuad.comp",
        "spv.subgroupVote.comp",
        "spv.vulkan110.storageBuffer.vert",
    })),
    FileNameAsCustomTestSuffix
);

// clang-format off
INSTANTIATE_TEST_CASE_P(
    Hlsl, HlslIoMap,
    ::testing::ValuesIn(std::vector<IoMapData>{
        { "spv.register.autoassign.frag", "main_ep", 5, 10, 0, 20, 30, true, false },
        { "spv.register.noautoassign.frag", "main_ep", 5, 10, 0, 15, 30, false, false },
        { "spv.register.autoassign-2.frag", "main", 5, 10, 0, 15, 30, true, true },
        { "spv.register.subpass.frag", "main", 0, 20, 0, 0, 0, true, true },
        { "spv.buffer.autoassign.frag", "main", 5, 10, 0, 15, 30, true, true },
        { "spv.ssbo.autoassign.frag", "main", 5, 10, 0, 15, 30, true, true },
        { "spv.ssboAlias.frag", "main", 0, 0, 0, 0, 83, true, false },
        { "spv.rw.autoassign.frag", "main", 5, 10, 20, 15, 30, true, true },
        { "spv.register.autoassign.rangetest.frag", "main",
                glslang::TQualifier::layoutBindingEnd-2,
                glslang::TQualifier::layoutBindingEnd+5,
                20, 30, true, false },
    }),
    FileNameAsCustomTestSuffixIoMap
);

// clang-format off
INSTANTIATE_TEST_CASE_P(
    Hlsl, GlslIoMap,
    ::testing::ValuesIn(std::vector<IoMapData>{
        { "spv.glsl.register.autoassign.frag", "main", 5, 10, 0, 20, 30, true, false },
        { "spv.glsl.register.noautoassign.frag", "main", 5, 10, 0, 15, 30, false, false },
    }),
    FileNameAsCustomTestSuffixIoMap
);

// clang-format off
INSTANTIATE_TEST_CASE_P(
    Glsl, CompileOpenGLToSpirvTest,
    ::testing::ValuesIn(std::vector<std::string>({
        "spv.460.frag",
        "spv.460.vert",
        "spv.460.comp",
        "spv.atomic.comp",
        "spv.glFragColor.frag",
        "spv.rankShift.comp",
        "spv.specConst.vert",
        "spv.OVR_multiview.vert",
        "spv.xfbOffsetOnBlockMembersAssignment.vert",
        "spv.xfbOffsetOnStructMembersAssignment.vert",
        "spv.xfbOverlapOffsetCheckWithBlockAndMember.vert",
        "spv.xfbStrideJustOnce.vert",
    })),
    FileNameAsCustomTestSuffix
);

INSTANTIATE_TEST_CASE_P(
    Glsl, VulkanSemantics,
    ::testing::ValuesIn(std::vector<std::string>({
        "vulkan.frag",
        "vulkan.vert",
        "vulkan.comp",
        "samplerlessTextureFunctions.frag",
    })),
    FileNameAsCustomTestSuffix
);

INSTANTIATE_TEST_CASE_P(
    Glsl, OpenGLSemantics,
    ::testing::ValuesIn(std::vector<std::string>({
        "glspv.esversion.vert",
        "glspv.version.frag",
        "glspv.version.vert",
        "glspv.frag",
        "glspv.vert",
    })),
    FileNameAsCustomTestSuffix
);

INSTANTIATE_TEST_CASE_P(
    Glsl, VulkanAstSemantics,
    ::testing::ValuesIn(std::vector<std::string>({
        "vulkan.ast.vert",
    })),
    FileNameAsCustomTestSuffix
);

#ifdef AMD_EXTENSIONS
INSTANTIATE_TEST_CASE_P(
    Glsl, CompileVulkanToSpirvTestAMD,
    ::testing::ValuesIn(std::vector<std::string>({
        "spv.16bitxfb.vert",
        "spv.float16.frag",
        "spv.float16Fetch.frag",
        "spv.imageLoadStoreLod.frag",
        "spv.int16.frag",
        "spv.int16.amd.frag",
        "spv.shaderBallotAMD.comp",
        "spv.shaderFragMaskAMD.frag",
        "spv.textureGatherBiasLod.frag",
    })),
    FileNameAsCustomTestSuffix
);
#endif

#ifdef NV_EXTENSIONS
INSTANTIATE_TEST_CASE_P(
    Glsl, CompileVulkanToSpirvTestNV,
    ::testing::ValuesIn(std::vector<std::string>({
    "spv.sampleMaskOverrideCoverage.frag",
    "spv.GeometryShaderPassthrough.geom",
    "spv.viewportArray2.vert",
    "spv.viewportArray2.tesc",
    "spv.stereoViewRendering.vert",
    "spv.stereoViewRendering.tesc",
    "spv.multiviewPerViewAttributes.vert",
    "spv.multiviewPerViewAttributes.tesc",
    "spv.atomicInt64.comp",
    "spv.shadingRate.frag",
    "spv.RayGenShader.rgen",
    "spv.RayGenShader_Errors.rgen",
    "spv.RayConstants.rgen",
    "spv.IntersectShader.rint",
    "spv.IntersectShader_Errors.rint",
    "spv.AnyHitShader.rahit",
    "spv.AnyHitShader_Errors.rahit",
    "spv.ClosestHitShader.rchit",
    "spv.ClosestHitShader_Errors.rchit",
    "spv.MissShader.rmiss",
    "spv.MissShader_Errors.rmiss",
    "spv.RayCallable.rcall",
    "spv.RayCallable_Errors.rcall",
    "spv.fragmentShaderBarycentric.frag",
    "spv.fragmentShaderBarycentric2.frag",
    "spv.computeShaderDerivatives.comp",
    "spv.computeShaderDerivatives2.comp",
    "spv.shaderImageFootprint.frag",
    "spv.meshShaderBuiltins.mesh",
    "spv.meshShaderUserDefined.mesh",
    "spv.meshShaderPerViewBuiltins.mesh",
    "spv.meshShaderPerViewUserDefined.mesh",
    "spv.meshShaderSharedMem.mesh",
    "spv.meshShaderTaskMem.mesh",
    "spv.320.meshShaderUserDefined.mesh",
    "spv.meshShaderRedeclBuiltins.mesh",
    "spv.meshShaderRedeclPerViewBuiltins.mesh",
    "spv.meshTaskShader.task",
    "spv.perprimitiveNV.frag",
})),
FileNameAsCustomTestSuffix
);
#endif

INSTANTIATE_TEST_CASE_P(
    Glsl, CompileUpgradeTextureToSampledTextureAndDropSamplersTest,
    ::testing::ValuesIn(std::vector<std::string>({
      "spv.texture.sampler.transform.frag",
    })),
    FileNameAsCustomTestSuffix
);
// clang-format on

}  // anonymous namespace
}  // namespace glslangtest
