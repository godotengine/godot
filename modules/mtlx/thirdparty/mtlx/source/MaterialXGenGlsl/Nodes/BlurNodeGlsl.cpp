//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXGenGlsl/Nodes/BlurNodeGlsl.h>

#include <MaterialXGenShader/GenContext.h>
#include <MaterialXGenShader/ShaderNode.h>
#include <MaterialXGenShader/ShaderStage.h>
#include <MaterialXGenShader/ShaderGenerator.h>

MATERIALX_NAMESPACE_BEGIN

ShaderNodeImplPtr BlurNodeGlsl::create()
{
    return std::make_shared<BlurNodeGlsl>();
}

void BlurNodeGlsl::emitSamplingFunctionDefinition(const ShaderNode& /*node*/, GenContext& context, ShaderStage& stage) const
{
    const ShaderGenerator& shadergen = context.getShaderGenerator();
    shadergen.emitLibraryInclude("stdlib/genglsl/lib/mx_sampling.glsl", context, stage);
    shadergen.emitLineBreak(stage);
}

MATERIALX_NAMESPACE_END
