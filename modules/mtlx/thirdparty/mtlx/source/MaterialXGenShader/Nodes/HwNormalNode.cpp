//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXGenShader/Nodes/HwNormalNode.h>

#include <MaterialXGenShader/Shader.h>

MATERIALX_NAMESPACE_BEGIN

ShaderNodeImplPtr HwNormalNode::create()
{
    return std::make_shared<HwNormalNode>();
}

void HwNormalNode::createVariables(const ShaderNode& node, GenContext&, Shader& shader) const
{
    ShaderStage& vs = shader.getStage(Stage::VERTEX);
    ShaderStage& ps = shader.getStage(Stage::PIXEL);

    addStageInput(HW::VERTEX_INPUTS, Type::VECTOR3, HW::T_IN_NORMAL, vs);

    const ShaderInput* spaceInput = node.getInput(SPACE);
    const int space = spaceInput ? spaceInput->getValue()->asA<int>() : OBJECT_SPACE;
    if (space == WORLD_SPACE)
    {
        addStageUniform(HW::PRIVATE_UNIFORMS, Type::MATRIX44, HW::T_WORLD_INVERSE_TRANSPOSE_MATRIX, vs);
        addStageConnector(HW::VERTEX_DATA, Type::VECTOR3, HW::T_NORMAL_WORLD, vs, ps);
    }
    else
    {
        addStageConnector(HW::VERTEX_DATA, Type::VECTOR3, HW::T_NORMAL_OBJECT, vs, ps);
    }
}

void HwNormalNode::emitFunctionCall(const ShaderNode& node, GenContext& context, ShaderStage& stage) const
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
            ShaderPort* normal = vertexData[HW::T_NORMAL_WORLD];
            if (!normal->isEmitted())
            {
                normal->setEmitted();
                shadergen.emitLine(prefix + normal->getVariable() + " = normalize((" + HW::T_WORLD_INVERSE_TRANSPOSE_MATRIX + " * vec4(" + HW::T_IN_NORMAL + ", 0.0)).xyz)", stage);
            }
        }
        else
        {
            ShaderPort* normal = vertexData[HW::T_NORMAL_OBJECT];
            if (!normal->isEmitted())
            {
                normal->setEmitted();
                shadergen.emitLine(prefix + normal->getVariable() + " = " + HW::T_IN_NORMAL, stage);
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
            const ShaderPort* normal = vertexData[HW::T_NORMAL_WORLD];
            shadergen.emitString(" = normalize(" + prefix + normal->getVariable() + ")", stage);
        }
        else
        {
            const ShaderPort* normal = vertexData[HW::T_NORMAL_OBJECT];
            shadergen.emitString(" = normalize(" + prefix + normal->getVariable() + ")", stage);
        }
        shadergen.emitLineEnd(stage);
    }
}

MATERIALX_NAMESPACE_END
