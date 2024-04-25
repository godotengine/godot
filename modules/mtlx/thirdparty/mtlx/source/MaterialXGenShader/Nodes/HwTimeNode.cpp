//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXGenShader/Nodes/HwTimeNode.h>

#include <MaterialXGenShader/Shader.h>

MATERIALX_NAMESPACE_BEGIN

ShaderNodeImplPtr HwTimeNode::create()
{
    return std::make_shared<HwTimeNode>();
}

void HwTimeNode::createVariables(const ShaderNode&, GenContext&, Shader& shader) const
{
    ShaderStage& ps = shader.getStage(Stage::PIXEL);
    addStageUniform(HW::PRIVATE_UNIFORMS, Type::FLOAT, HW::T_FRAME, ps);
}

void HwTimeNode::emitFunctionCall(const ShaderNode& node, GenContext& context, ShaderStage& stage) const
{
    DEFINE_SHADER_STAGE(stage, Stage::PIXEL)
    {
        const ShaderGenerator& shadergen = context.getShaderGenerator();
        shadergen.emitLineBegin(stage);
        shadergen.emitOutput(node.getOutput(), true, false, context, stage);
        shadergen.emitString(" = " + HW::T_FRAME + " / ", stage);
        const ShaderInput* fpsInput = node.getInput("fps");
        const string fps = fpsInput->getValue()->getValueString();
        shadergen.emitString(fps, stage);
        shadergen.emitLineEnd(stage);
    }
}

MATERIALX_NAMESPACE_END
