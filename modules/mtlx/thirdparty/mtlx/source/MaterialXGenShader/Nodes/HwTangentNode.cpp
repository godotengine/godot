//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXGenShader/Nodes/HwTangentNode.h>

#include <MaterialXGenShader/Shader.h>

MATERIALX_NAMESPACE_BEGIN

ShaderNodeImplPtr HwTangentNode::create()
{
    return std::make_shared<HwTangentNode>();
}

void HwTangentNode::createVariables(const ShaderNode& node, GenContext&, Shader& shader) const
{
    ShaderStage& vs = shader.getStage(Stage::VERTEX);
    ShaderStage& ps = shader.getStage(Stage::PIXEL);

    addStageInput(HW::VERTEX_INPUTS, Type::VECTOR3, HW::T_IN_TANGENT, vs);

    const ShaderInput* spaceInput = node.getInput(SPACE);
    const int space = spaceInput ? spaceInput->getValue()->asA<int>() : OBJECT_SPACE;
    if (space == WORLD_SPACE)
    {
        addStageUniform(HW::PRIVATE_UNIFORMS, Type::MATRIX44, HW::T_WORLD_MATRIX, vs);
        addStageConnector(HW::VERTEX_DATA, Type::VECTOR3, HW::T_TANGENT_WORLD, vs, ps);
    }
    else
    {
        addStageConnector(HW::VERTEX_DATA, Type::VECTOR3, HW::T_TANGENT_OBJECT, vs, ps);
    }
}

void HwTangentNode::emitFunctionCall(const ShaderNode& node, GenContext& context, ShaderStage& stage) const
{
    const HwShaderGenerator& shadergen = static_cast<const HwShaderGenerator&>(context.getShaderGenerator());

    const ShaderInput* spaceInput = node.getInput(SPACE);
    const int space = spaceInput ? spaceInput->getValue()->asA<int>() : OBJECT_SPACE;

    DEFINE_SHADER_STAGE(stage, Stage::VERTEX)
    {
        VariableBlock& vertexData = stage.getOutputBlock(HW::VERTEX_DATA);
        const string prefix = shadergen.getVertexDataPrefix(vertexData);
        if (space == WORLD_SPACE)
        {
            ShaderPort* tangent = vertexData[HW::T_TANGENT_WORLD];
            if (!tangent->isEmitted())
            {
                tangent->setEmitted();
                shadergen.emitLine(prefix + tangent->getVariable() + " = normalize((" + HW::T_WORLD_MATRIX + " * vec4(" + HW::T_IN_TANGENT + ", 0.0)).xyz)", stage);
            }
        }
        else
        {
            ShaderPort* tangent = vertexData[HW::T_TANGENT_OBJECT];
            if (!tangent->isEmitted())
            {
                tangent->setEmitted();
                shadergen.emitLine(prefix + tangent->getVariable() + " = " + HW::T_IN_TANGENT, stage);
            }
        }
    }

    DEFINE_SHADER_STAGE(stage, Stage::PIXEL)
    {
        VariableBlock& vertexData = stage.getInputBlock(HW::VERTEX_DATA);
        const string prefix = shadergen.getVertexDataPrefix(vertexData);
        shadergen.emitLineBegin(stage);
        shadergen.emitOutput(node.getOutput(), true, false, context, stage);
        if (space == WORLD_SPACE)
        {
            const ShaderPort* tangent = vertexData[HW::T_TANGENT_WORLD];
            shadergen.emitString(" = normalize(" + prefix + tangent->getVariable() + ")", stage);
        }
        else
        {
            const ShaderPort* tangent = vertexData[HW::T_TANGENT_OBJECT];
            shadergen.emitString(" = normalize(" + prefix + tangent->getVariable() + ")", stage);
        }
        shadergen.emitLineEnd(stage);
    }
}

MATERIALX_NAMESPACE_END
