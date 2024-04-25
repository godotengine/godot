//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXGenShader/Nodes/HwPositionNode.h>

#include <MaterialXGenShader/Shader.h>

MATERIALX_NAMESPACE_BEGIN

ShaderNodeImplPtr HwPositionNode::create()
{
    return std::make_shared<HwPositionNode>();
}

void HwPositionNode::createVariables(const ShaderNode& node, GenContext&, Shader& shader) const
{
    ShaderStage vs = shader.getStage(Stage::VERTEX);
    ShaderStage ps = shader.getStage(Stage::PIXEL);

    addStageInput(HW::VERTEX_INPUTS, Type::VECTOR3, HW::T_IN_POSITION, vs);

    const ShaderInput* spaceInput = node.getInput(SPACE);
    const int space = spaceInput ? spaceInput->getValue()->asA<int>() : OBJECT_SPACE;
    if (space == WORLD_SPACE)
    {
        addStageConnector(HW::VERTEX_DATA, Type::VECTOR3, HW::T_POSITION_WORLD, vs, ps);
    }
    else
    {
        addStageConnector(HW::VERTEX_DATA, Type::VECTOR3, HW::T_POSITION_OBJECT, vs, ps);
    }
}

void HwPositionNode::emitFunctionCall(const ShaderNode& node, GenContext& context, ShaderStage& stage) const
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
            ShaderPort* position = vertexData[HW::T_POSITION_WORLD];
            if (!position->isEmitted())
            {
                position->setEmitted();
                shadergen.emitLine(prefix + position->getVariable() + " = hPositionWorld.xyz", stage);
            }
        }
        else
        {
            ShaderPort* position = vertexData[HW::T_POSITION_OBJECT];
            if (!position->isEmitted())
            {
                position->setEmitted();
                shadergen.emitLine(prefix + position->getVariable() + " = " + HW::T_IN_POSITION, stage);
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
            const ShaderPort* position = vertexData[HW::T_POSITION_WORLD];
            shadergen.emitString(" = " + prefix + position->getVariable(), stage);
        }
        else
        {
            const ShaderPort* position = vertexData[HW::T_POSITION_OBJECT];
            shadergen.emitString(" = " + prefix + position->getVariable(), stage);
        }
        shadergen.emitLineEnd(stage);
    }
}

MATERIALX_NAMESPACE_END
