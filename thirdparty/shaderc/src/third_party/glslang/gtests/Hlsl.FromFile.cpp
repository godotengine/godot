//
// Copyright (C) 2016 Google, Inc.
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

struct FileNameEntryPointPair {
  const char* fileName;
  const char* entryPoint;
};

// We are using FileNameEntryPointPair objects as parameters for instantiating
// the template, so the global FileNameAsCustomTestSuffix() won't work since
// it assumes std::string as parameters. Thus, an overriding one here.
std::string FileNameAsCustomTestSuffix(
    const ::testing::TestParamInfo<FileNameEntryPointPair>& info) {
    std::string name = info.param.fileName;
    // A valid test case suffix cannot have '.' and '-' inside.
    std::replace(name.begin(), name.end(), '.', '_');
    std::replace(name.begin(), name.end(), '-', '_');
    return name;
}

using HlslCompileTest = GlslangTest<::testing::TestWithParam<FileNameEntryPointPair>>;
using HlslVulkan1_1CompileTest = GlslangTest<::testing::TestWithParam<FileNameEntryPointPair>>;
using HlslCompileAndFlattenTest = GlslangTest<::testing::TestWithParam<FileNameEntryPointPair>>;
using HlslLegalizeTest = GlslangTest<::testing::TestWithParam<FileNameEntryPointPair>>;
using HlslDebugTest = GlslangTest<::testing::TestWithParam<FileNameEntryPointPair>>;
using HlslDX9CompatibleTest = GlslangTest<::testing::TestWithParam<FileNameEntryPointPair>>;
using HlslLegalDebugTest = GlslangTest<::testing::TestWithParam<FileNameEntryPointPair>>;

// Compiling HLSL to pre-legalized SPIR-V under Vulkan semantics. Expected
// to successfully generate both AST and SPIR-V.
TEST_P(HlslCompileTest, FromFile)
{
    loadFileCompileAndCheck(GlobalTestSettings.testRoot, GetParam().fileName,
                            Source::HLSL, Semantics::Vulkan, glslang::EShTargetVulkan_1_0,
                            Target::BothASTAndSpv, true, GetParam().entryPoint);
}

TEST_P(HlslVulkan1_1CompileTest, FromFile)
{
    loadFileCompileAndCheck(GlobalTestSettings.testRoot, GetParam().fileName,
                            Source::HLSL, Semantics::Vulkan, glslang::EShTargetVulkan_1_1,
                            Target::BothASTAndSpv, true, GetParam().entryPoint);
}

TEST_P(HlslCompileAndFlattenTest, FromFile)
{
    loadFileCompileFlattenUniformsAndCheck(GlobalTestSettings.testRoot, GetParam().fileName,
                                           Source::HLSL, Semantics::Vulkan,
                                           Target::BothASTAndSpv, GetParam().entryPoint);
}

// Compiling HLSL to legal SPIR-V under Vulkan semantics. Expected to
// successfully generate SPIR-V.
TEST_P(HlslLegalizeTest, FromFile)
{
    loadFileCompileAndCheck(GlobalTestSettings.testRoot, GetParam().fileName,
                            Source::HLSL, Semantics::Vulkan, glslang::EShTargetVulkan_1_0,
                            Target::Spv, true, GetParam().entryPoint,
                            "/baseLegalResults/", true);
}

// Compiling HLSL to pre-legalized SPIR-V. Expected to successfully generate
// SPIR-V with debug instructions, particularly line info.
TEST_P(HlslDebugTest, FromFile)
{
    loadFileCompileAndCheck(GlobalTestSettings.testRoot, GetParam().fileName,
                            Source::HLSL, Semantics::Vulkan, glslang::EShTargetVulkan_1_0,
                            Target::Spv, true, GetParam().entryPoint,
                            "/baseResults/", false, true);
}

TEST_P(HlslDX9CompatibleTest, FromFile)
{
    loadFileCompileAndCheckWithOptions(GlobalTestSettings.testRoot, GetParam().fileName, Source::HLSL,
                                       Semantics::Vulkan, glslang::EShTargetVulkan_1_0, Target::BothASTAndSpv, true,
                                       GetParam().entryPoint, "/baseResults/",
                                       EShMessages::EShMsgHlslDX9Compatible);
}

// Compiling HLSL to legalized SPIR-V with debug instructions. Expected to
// successfully generate SPIR-V with debug instructions preserved through
// legalization, particularly line info.
TEST_P(HlslLegalDebugTest, FromFile)
{
    loadFileCompileAndCheck(GlobalTestSettings.testRoot, GetParam().fileName,
                            Source::HLSL, Semantics::Vulkan, glslang::EShTargetVulkan_1_0,
                            Target::Spv, true, GetParam().entryPoint,
                            "/baseResults/", true, true);
}

// clang-format off
INSTANTIATE_TEST_CASE_P(
    ToSpirv, HlslCompileTest,
    ::testing::ValuesIn(std::vector<FileNameEntryPointPair>{
        {"hlsl.amend.frag", "f1"},
        {"hlsl.aliasOpaque.frag", "main"},
        {"hlsl.array.frag", "PixelShaderFunction"},
        {"hlsl.array.implicit-size.frag", "PixelShaderFunction"},
        {"hlsl.array.multidim.frag", "main"},
        {"hlsl.assoc.frag", "PixelShaderFunction"},
        {"hlsl.attribute.frag", "PixelShaderFunction"},
        {"hlsl.attribute.expression.comp", "main"},
        {"hlsl.attributeC11.frag", "main"},
        {"hlsl.attributeGlobalBuffer.frag", "main"},
        {"hlsl.basic.comp", "main"},
        {"hlsl.basic.geom", "main"},
        {"hlsl.boolConv.vert", "main"},
        {"hlsl.buffer.frag", "PixelShaderFunction"},
        {"hlsl.calculatelod.dx10.frag", "main"},
        {"hlsl.calculatelodunclamped.dx10.frag", "main"},
        {"hlsl.cast.frag", "PixelShaderFunction"},
        {"hlsl.cbuffer-identifier.vert", "main"},
        {"hlsl.charLit.vert", "main"},
        {"hlsl.clip.frag", "main"},
        {"hlsl.clipdistance-1.frag", "main"},
        {"hlsl.clipdistance-1.geom", "main"},
        {"hlsl.clipdistance-1.vert", "main"},
        {"hlsl.clipdistance-2.frag", "main"},
        {"hlsl.clipdistance-2.geom", "main"},
        {"hlsl.clipdistance-2.vert", "main"},
        {"hlsl.clipdistance-3.frag", "main"},
        {"hlsl.clipdistance-3.geom", "main"},
        {"hlsl.clipdistance-3.vert", "main"},
        {"hlsl.clipdistance-4.frag", "main"},
        {"hlsl.clipdistance-4.geom", "main"},
        {"hlsl.clipdistance-4.vert", "main"},
        {"hlsl.clipdistance-5.frag", "main"},
        {"hlsl.clipdistance-5.vert", "main"},
        {"hlsl.clipdistance-6.frag", "main"},
        {"hlsl.clipdistance-6.vert", "main"},
        {"hlsl.clipdistance-7.frag", "main"},
        {"hlsl.clipdistance-7.vert", "main"},
        {"hlsl.clipdistance-8.frag", "main"},
        {"hlsl.clipdistance-8.vert", "main"},
        {"hlsl.clipdistance-9.frag", "main"},
        {"hlsl.clipdistance-9.vert", "main"},
        {"hlsl.color.hull.tesc", "main"},
        {"hlsl.comparison.vec.frag", "main"},
        {"hlsl.conditional.frag", "PixelShaderFunction"},
        {"hlsl.constantbuffer.frag", "main"},
        {"hlsl.constructArray.vert", "main"},
        {"hlsl.constructexpr.frag", "main"},
        {"hlsl.constructimat.frag", "main"},
        {"hlsl.coverage.frag", "main"},
        {"hlsl.depthGreater.frag", "PixelShaderFunction"},
        {"hlsl.depthLess.frag", "PixelShaderFunction"},
        {"hlsl.discard.frag", "PixelShaderFunction"},
        {"hlsl.doLoop.frag", "PixelShaderFunction"},
        {"hlsl.earlydepthstencil.frag", "main"},
        {"hlsl.emptystructreturn.frag", "main"},
        {"hlsl.emptystructreturn.vert", "main"},
        {"hlsl.emptystruct.init.vert", "main"},
        {"hlsl.entry-in.frag", "PixelShaderFunction"},
        {"hlsl.entry-out.frag", "PixelShaderFunction"},
        {"hlsl.fraggeom.frag", "main"},
        {"hlsl.float1.frag", "PixelShaderFunction"},
        {"hlsl.float4.frag", "PixelShaderFunction"},
        {"hlsl.flatten.return.frag", "main"},
        {"hlsl.flattenOpaque.frag", "main"},
        {"hlsl.flattenOpaqueInit.vert", "main"},
        {"hlsl.flattenOpaqueInitMix.vert", "main"},
        {"hlsl.flattenSubset.frag", "main"},
        {"hlsl.flattenSubset2.frag", "main"},
        {"hlsl.forLoop.frag", "PixelShaderFunction"},
        {"hlsl.gather.array.dx10.frag", "main"},
        {"hlsl.gather.basic.dx10.frag", "main"},
        {"hlsl.gather.basic.dx10.vert", "main"},
        {"hlsl.gather.offset.dx10.frag", "main"},
        {"hlsl.gather.offsetarray.dx10.frag", "main"},
        {"hlsl.gathercmpRGBA.offset.dx10.frag", "main"},
        {"hlsl.gatherRGBA.array.dx10.frag", "main"},
        {"hlsl.gatherRGBA.basic.dx10.frag", "main"},
        {"hlsl.gatherRGBA.offset.dx10.frag", "main"},
        {"hlsl.gatherRGBA.offsetarray.dx10.frag", "main"},
        {"hlsl.getdimensions.dx10.frag", "main"},
        {"hlsl.getdimensions.rw.dx10.frag", "main"},
        {"hlsl.getdimensions.dx10.vert", "main"},
        {"hlsl.getsampleposition.dx10.frag", "main"},
        {"hlsl.global-const-init.frag", "main"},
        {"hlsl.gs-hs-mix.tesc", "HSMain"},
        {"hlsl.domain.1.tese", "main"},
        {"hlsl.domain.2.tese", "main"},
        {"hlsl.domain.3.tese", "main"},
        {"hlsl.function.frag", "main"},
        {"hlsl.hull.1.tesc", "main"},
        {"hlsl.hull.2.tesc", "main"},
        {"hlsl.hull.3.tesc", "main"},
        {"hlsl.hull.4.tesc", "main"},
        {"hlsl.hull.5.tesc", "main"},
        {"hlsl.hull.void.tesc", "main"},
        {"hlsl.hull.ctrlpt-1.tesc", "main"},
        {"hlsl.hull.ctrlpt-2.tesc", "main"},
        {"hlsl.groupid.comp", "main"},
        {"hlsl.identifier.sample.frag", "main"},
        {"hlsl.if.frag", "PixelShaderFunction"},
        {"hlsl.imagefetch-subvec4.comp", "main"},
        {"hlsl.implicitBool.frag", "main"},
        {"hlsl.inf.vert", "main"},
        {"hlsl.inoutquals.frag", "main"},
        {"hlsl.init.frag", "ShaderFunction"},
        {"hlsl.init2.frag", "main"},
        {"hlsl.isfinite.frag", "main"},
        {"hlsl.intrinsics.barriers.comp", "ComputeShaderFunction"},
        {"hlsl.intrinsics.comp", "ComputeShaderFunction"},
        {"hlsl.intrinsics.evalfns.frag", "main"},
        {"hlsl.intrinsics.d3dcolortoubyte4.frag", "main"},
        {"hlsl.intrinsics.double.frag", "PixelShaderFunction"},
        {"hlsl.intrinsics.f1632.frag", "main"},
        {"hlsl.intrinsics.f3216.frag", "main"},
        {"hlsl.intrinsics.frag", "main"},
        {"hlsl.intrinsic.frexp.frag", "main"},
        {"hlsl.intrinsics.lit.frag", "PixelShaderFunction"},
        {"hlsl.intrinsics.negative.comp", "ComputeShaderFunction"},
        {"hlsl.intrinsics.negative.frag", "PixelShaderFunction"},
        {"hlsl.intrinsics.negative.vert", "VertexShaderFunction"},
        {"hlsl.intrinsics.promote.frag", "main"},
        {"hlsl.intrinsics.promote.down.frag", "main"},
        {"hlsl.intrinsics.promote.outputs.frag", "main"},
        {"hlsl.layout.frag", "main"},
        {"hlsl.layoutOverride.vert", "main"},
        {"hlsl.load.2dms.dx10.frag", "main"},
        {"hlsl.load.array.dx10.frag", "main"},
        {"hlsl.load.basic.dx10.frag", "main"},
        {"hlsl.load.basic.dx10.vert", "main"},
        {"hlsl.load.buffer.dx10.frag", "main"},
        {"hlsl.load.buffer.float.dx10.frag", "main"},
        {"hlsl.load.rwbuffer.dx10.frag", "main"},
        {"hlsl.load.rwtexture.dx10.frag", "main"},
        {"hlsl.load.rwtexture.array.dx10.frag", "main"},
        {"hlsl.load.offset.dx10.frag", "main"},
        {"hlsl.load.offsetarray.dx10.frag", "main"},
        {"hlsl.localStructuredBuffer.comp", "main"},
        {"hlsl.logical.binary.frag", "main"},
        {"hlsl.logical.binary.vec.frag", "main"},
        {"hlsl.logicalConvert.frag", "main"},
        {"hlsl.logical.unary.frag", "main"},
        {"hlsl.loopattr.frag", "main"},
        {"hlsl.matpack-pragma.frag", "main"},
        {"hlsl.mip.operator.frag", "main"},
        {"hlsl.mip.negative.frag", "main"},
        {"hlsl.mip.negative2.frag", "main"},
        {"hlsl.namespace.frag", "main"},
        {"hlsl.nonint-index.frag", "main"},
        {"hlsl.matNx1.frag", "main"},
        {"hlsl.matpack-1.frag", "main"},
        {"hlsl.matrixSwizzle.vert", "ShaderFunction"},
        {"hlsl.memberFunCall.frag", "main"},
        {"hlsl.mintypes.frag", "main"},
        {"hlsl.mul-truncate.frag", "main"},
        {"hlsl.multiEntry.vert", "RealEntrypoint"},
        {"hlsl.multiReturn.frag", "main"},
        {"hlsl.matrixindex.frag", "main"},
        {"hlsl.nonstaticMemberFunction.frag", "main"},
        {"hlsl.numericsuffixes.frag", "main"},
        {"hlsl.numthreads.comp", "main_aux2"},
        {"hlsl.overload.frag", "PixelShaderFunction"},
        {"hlsl.opaque-type-bug.frag", "main"},
        {"hlsl.params.default.frag", "main"},
        {"hlsl.params.default.negative.frag", "main"},
        {"hlsl.partialInit.frag", "PixelShaderFunction"},
        {"hlsl.partialFlattenLocal.vert", "main"},
        {"hlsl.PointSize.geom", "main"},
        {"hlsl.PointSize.vert", "main"},
        {"hlsl.pp.vert", "main"},
        {"hlsl.pp.line.frag", "main"},
        {"hlsl.precise.frag", "main"},
        {"hlsl.promote.atomic.frag", "main"},
        {"hlsl.promote.binary.frag", "main"},
        {"hlsl.promote.vec1.frag", "main"},
        {"hlsl.promotions.frag", "main"},
        {"hlsl.rw.atomics.frag", "main"},
        {"hlsl.rw.bracket.frag", "main"},
        {"hlsl.rw.register.frag", "main"},
        {"hlsl.rw.scalar.bracket.frag", "main"},
        {"hlsl.rw.swizzle.frag", "main"},
        {"hlsl.rw.vec2.bracket.frag", "main"},
        {"hlsl.sample.array.dx10.frag", "main"},
        {"hlsl.sample.basic.dx10.frag", "main"},
        {"hlsl.sample.offset.dx10.frag", "main"},
        {"hlsl.sample.offsetarray.dx10.frag", "main"},
        {"hlsl.samplebias.array.dx10.frag", "main"},
        {"hlsl.samplebias.basic.dx10.frag", "main"},
        {"hlsl.samplebias.offset.dx10.frag", "main"},
        {"hlsl.samplebias.offsetarray.dx10.frag", "main"},
        {"hlsl.samplecmp.array.dx10.frag", "main"},
        {"hlsl.samplecmp.basic.dx10.frag", "main"},
        {"hlsl.samplecmp.dualmode.frag", "main"},
        {"hlsl.samplecmp.offset.dx10.frag", "main"},
        {"hlsl.samplecmp.offsetarray.dx10.frag", "main"},
        {"hlsl.samplecmp.negative.frag", "main"},
        {"hlsl.samplecmp.negative2.frag", "main"},
        {"hlsl.samplecmplevelzero.array.dx10.frag", "main"},
        {"hlsl.samplecmplevelzero.basic.dx10.frag", "main"},
        {"hlsl.samplecmplevelzero.offset.dx10.frag", "main"},
        {"hlsl.samplecmplevelzero.offsetarray.dx10.frag", "main"},
        {"hlsl.samplegrad.array.dx10.frag", "main"},
        {"hlsl.samplegrad.basic.dx10.frag", "main"},
        {"hlsl.samplegrad.basic.dx10.vert", "main"},
        {"hlsl.samplegrad.offset.dx10.frag", "main"},
        {"hlsl.samplegrad.offsetarray.dx10.frag", "main"},
        {"hlsl.samplelevel.array.dx10.frag", "main"},
        {"hlsl.samplelevel.basic.dx10.frag", "main"},
        {"hlsl.samplelevel.basic.dx10.vert", "main"},
        {"hlsl.samplelevel.offset.dx10.frag", "main"},
        {"hlsl.samplelevel.offsetarray.dx10.frag", "main"},
        {"hlsl.sample.sub-vec4.dx10.frag", "main"},
        {"hlsl.scalar-length.frag", "main"},
        {"hlsl.scalarCast.vert", "main"},
        {"hlsl.semicolons.frag", "main"},
        {"hlsl.shapeConv.frag", "main"},
        {"hlsl.shapeConvRet.frag", "main"},
        {"hlsl.self_cast.frag", "main"},
        {"hlsl.snorm.uav.comp", "main"},
        {"hlsl.staticMemberFunction.frag", "main"},
        {"hlsl.staticFuncInit.frag", "main"},
        {"hlsl.store.rwbyteaddressbuffer.type.comp", "main"},
        {"hlsl.stringtoken.frag", "main"},
        {"hlsl.string.frag", "main"},
        {"hlsl.struct.split-1.vert", "main"},
        {"hlsl.struct.split.array.geom", "main"},
        {"hlsl.struct.split.assign.frag", "main"},
        {"hlsl.struct.split.call.vert", "main"},
        {"hlsl.struct.split.nested.geom", "main"},
        {"hlsl.struct.split.trivial.geom", "main"},
        {"hlsl.struct.split.trivial.vert", "main"},
        {"hlsl.structarray.flatten.frag", "main"},
        {"hlsl.structarray.flatten.geom", "main"},
        {"hlsl.structbuffer.frag", "main"},
        {"hlsl.structbuffer.append.frag", "main"},
        {"hlsl.structbuffer.append.fn.frag", "main"},
        {"hlsl.structbuffer.atomics.frag", "main"},
        {"hlsl.structbuffer.byte.frag", "main"},
        {"hlsl.structbuffer.coherent.frag", "main"},
        {"hlsl.structbuffer.floatidx.comp", "main"},
        {"hlsl.structbuffer.incdec.frag", "main"},
        {"hlsl.structbuffer.fn.frag", "main"},
        {"hlsl.structbuffer.fn2.comp", "main"},
        {"hlsl.structbuffer.rw.frag", "main"},
        {"hlsl.structbuffer.rwbyte.frag", "main"},
        {"hlsl.structin.vert", "main"},
        {"hlsl.structIoFourWay.frag", "main"},
        {"hlsl.structStructName.frag", "main"},
        {"hlsl.subpass.frag", "main"},
        {"hlsl.synthesizeInput.frag", "main"},
        {"hlsl.texturebuffer.frag", "main"},
        {"hlsl.texture.struct.frag", "main"},
        {"hlsl.texture.subvec4.frag", "main"},
        {"hlsl.this.frag", "main"},
        {"hlsl.intrinsics.vert", "VertexShaderFunction"},
        {"hlsl.intrinsic.frexp.vert", "VertexShaderFunction"},
        {"hlsl.matType.frag", "PixelShaderFunction"},
        {"hlsl.matType.bool.frag", "main"},
        {"hlsl.matType.int.frag", "main"},
        {"hlsl.max.frag", "PixelShaderFunction"},
        {"hlsl.preprocessor.frag", "main"},
        {"hlsl.precedence.frag", "PixelShaderFunction"},
        {"hlsl.precedence2.frag", "PixelShaderFunction"},
        {"hlsl.scalar2matrix.frag", "main"},
        {"hlsl.semantic.geom", "main"},
        {"hlsl.semantic.vert", "main"},
        {"hlsl.semantic-1.vert", "main"},
        {"hlsl.scope.frag", "PixelShaderFunction"},
        {"hlsl.sin.frag", "PixelShaderFunction"},
        {"hlsl.struct.frag", "PixelShaderFunction"},
        {"hlsl.switch.frag", "PixelShaderFunction"},
        {"hlsl.swizzle.frag", "PixelShaderFunction"},
        {"hlsl.target.frag", "main"},
        {"hlsl.targetStruct1.frag", "main"},
        {"hlsl.targetStruct2.frag", "main"},
        {"hlsl.templatetypes.frag", "PixelShaderFunction"},
        {"hlsl.tristream-append.geom", "main"},
        {"hlsl.tx.bracket.frag", "main"},
        {"hlsl.tx.overload.frag", "main"},
        {"hlsl.type.half.frag", "main"},
        {"hlsl.type.identifier.frag", "main"},
        {"hlsl.typeGraphCopy.vert", "main"},
        {"hlsl.typedef.frag", "PixelShaderFunction"},
        {"hlsl.whileLoop.frag", "PixelShaderFunction"},
        {"hlsl.void.frag", "PixelShaderFunction"},
        {"hlsl.type.type.conversion.all.frag", "main"}
    }),
    FileNameAsCustomTestSuffix
);
// clang-format on

// clang-format off
INSTANTIATE_TEST_CASE_P(
    ToSpirv, HlslVulkan1_1CompileTest,
    ::testing::ValuesIn(std::vector<FileNameEntryPointPair>{
        {"hlsl.wavebroadcast.comp", "CSMain"},
        {"hlsl.waveprefix.comp", "CSMain"},
        {"hlsl.wavequad.comp", "CSMain"},
        {"hlsl.wavequery.comp", "CSMain"},
        {"hlsl.wavequery.frag", "PixelShaderFunction"},
        {"hlsl.wavereduction.comp", "CSMain"},
        {"hlsl.wavevote.comp", "CSMain"},
        { "hlsl.type.type.conversion.valid.frag", "main" },
        {"hlsl.int.dot.frag", "main"}
    }),
    FileNameAsCustomTestSuffix
);
// clang-format on

// clang-format off
INSTANTIATE_TEST_CASE_P(
    ToSpirv, HlslCompileAndFlattenTest,
    ::testing::ValuesIn(std::vector<FileNameEntryPointPair>{
        {"hlsl.array.flatten.frag", "main"},
        {"hlsl.partialFlattenMixed.vert", "main"},
    }),
    FileNameAsCustomTestSuffix
);
// clang-format on

#if ENABLE_OPT
// clang-format off
INSTANTIATE_TEST_CASE_P(
    ToSpirv, HlslLegalizeTest,
    ::testing::ValuesIn(std::vector<FileNameEntryPointPair>{
        {"hlsl.aliasOpaque.frag", "main"},
        {"hlsl.flattenOpaque.frag", "main"},
        {"hlsl.flattenOpaqueInit.vert", "main"},
        {"hlsl.flattenOpaqueInitMix.vert", "main"},
        {"hlsl.flattenSubset.frag", "main"},
        {"hlsl.flattenSubset2.frag", "main"},
        {"hlsl.partialFlattenLocal.vert", "main"},
        {"hlsl.partialFlattenMixed.vert", "main"}
    }),
    FileNameAsCustomTestSuffix
);
// clang-format on
#endif

// clang-format off
INSTANTIATE_TEST_CASE_P(
    ToSpirv, HlslDebugTest,
    ::testing::ValuesIn(std::vector<FileNameEntryPointPair>{
        {"hlsl.pp.line2.frag", "MainPs"}
    }),
    FileNameAsCustomTestSuffix
);

INSTANTIATE_TEST_CASE_P(
    ToSpirv, HlslDX9CompatibleTest,
    ::testing::ValuesIn(std::vector<FileNameEntryPointPair>{
        {"hlsl.sample.dx9.frag", "main"},
        {"hlsl.sample.dx9.vert", "main"},
    }),
    FileNameAsCustomTestSuffix
);

// clang-format off
INSTANTIATE_TEST_CASE_P(
    ToSpirv, HlslLegalDebugTest,
    ::testing::ValuesIn(std::vector<FileNameEntryPointPair>{
        {"hlsl.pp.line4.frag", "MainPs"}
    }),
    FileNameAsCustomTestSuffix
);

// clang-format on

}  // anonymous namespace
}  // namespace glslangtest
