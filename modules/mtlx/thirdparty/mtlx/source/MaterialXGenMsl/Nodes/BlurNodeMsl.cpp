//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXGenMsl/Nodes/BlurNodeMsl.h>

#include <MaterialXGenShader/GenContext.h>
#include <MaterialXGenShader/ShaderNode.h>
#include <MaterialXGenShader/ShaderStage.h>
#include <MaterialXGenShader/ShaderGenerator.h>

MATERIALX_NAMESPACE_BEGIN

ShaderNodeImplPtr BlurNodeMsl::create()
{
    return std::make_shared<BlurNodeMsl>();
}

void BlurNodeMsl::emitSamplingFunctionDefinition(const ShaderNode& /*node*/, GenContext& context, ShaderStage& stage) const
{
    const ShaderGenerator& shadergen = context.getShaderGenerator();
    shadergen.emitLibraryInclude("stdlib/genmsl/lib/mx_sampling.metal", context, stage);
    shadergen.emitLineBreak(stage);
}

MATERIALX_NAMESPACE_END
