//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXGenShader/Nodes/HwTexCoordNode.h>
#include <MaterialXGenShader/HwShaderGenerator.h>
#include <MaterialXGenShader/Shader.h>

MATERIALX_NAMESPACE_BEGIN

string HwTexCoordNode::INDEX = "index";

ShaderNodeImplPtr HwTexCoordNode::create()
{
    return std::make_shared<HwTexCoordNode>();
}

void HwTexCoordNode::createVariables(const ShaderNode& node, GenContext&, Shader& shader) const
{
    const ShaderOutput* output = node.getOutput();
    const string index = getIndex(node);

    ShaderStage& vs = shader.getStage(Stage::VERTEX);
    ShaderStage& ps = shader.getStage(Stage::PIXEL);

    addStageInput(HW::VERTEX_INPUTS, output->getType(), HW::T_IN_TEXCOORD + "_" + index, vs, true);
    addStageConnector(HW::VERTEX_DATA, output->getType(), HW::T_TEXCOORD + "_" + index, vs, ps, true);
}

void HwTexCoordNode::emitFunctionCall(const ShaderNode& node, GenContext& context, ShaderStage& stage) const
{
    const HwShaderGenerator& shadergen = static_cast<const HwShaderGenerator&>(context.getShaderGenerator());

    const string index = getIndex(node);
    const string variable = HW::T_TEXCOORD + "_" + index;
    const ShaderOutput* output = node.getOutput();

    DEFINE_SHADER_STAGE(stage, Stage::VERTEX)
    {
        VariableBlock& vertexData = stage.getOutputBlock(HW::VERTEX_DATA);
        const string prefix = shadergen.getVertexDataPrefix(vertexData);
        ShaderPort* texcoord = vertexData[variable];
        if (!texcoord->isEmitted())
        {
            shadergen.emitLine(prefix + texcoord->getVariable() + " = " + HW::T_IN_TEXCOORD + "_" + index, stage);
            texcoord->setEmitted();
        }
    }

    DEFINE_SHADER_STAGE(stage, Stage::PIXEL)
    {
        VariableBlock& vertexData = stage.getInputBlock(HW::VERTEX_DATA);
        const string prefix = shadergen.getVertexDataPrefix(vertexData);
        ShaderPort* texcoord = vertexData[variable];
        shadergen.emitLineBegin(stage);
        shadergen.emitOutput(output, true, false, context, stage);

        // Extract the requested number of components from the texture coordinates (which may be a
        // larger datatype than the requested number of texture coordinates, if several texture
        // coordinate nodes with different width coexist).
        string suffix = EMPTY_STRING;
        if (*output->getType() == *Type::VECTOR2)
        {
            suffix = ".xy";
        }
        else if (*output->getType() == *Type::VECTOR3)
        {
            suffix = ".xyz";
        }

        shadergen.emitString(" = " + prefix + texcoord->getVariable() + suffix, stage);
        shadergen.emitLineEnd(stage);
    }
}

string HwTexCoordNode::getIndex(const ShaderNode& node) const
{
    const ShaderInput* input = node.getInput(INDEX);
    return input ? input->getValue()->getValueString() : "0";
}

MATERIALX_NAMESPACE_END
