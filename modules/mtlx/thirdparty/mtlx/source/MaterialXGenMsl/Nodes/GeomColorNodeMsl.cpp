//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXGenMsl/Nodes/GeomColorNodeMsl.h>

#include <MaterialXGenShader/Shader.h>

MATERIALX_NAMESPACE_BEGIN

ShaderNodeImplPtr GeomColorNodeMsl::create()
{
    return std::make_shared<GeomColorNodeMsl>();
}

void GeomColorNodeMsl::createVariables(const ShaderNode& node, GenContext&, Shader& shader) const
{
    const ShaderInput* indexInput = node.getInput(INDEX);
    const string index = indexInput ? indexInput->getValue()->getValueString() : "0";

    ShaderStage& vs = shader.getStage(Stage::VERTEX);
    ShaderStage& ps = shader.getStage(Stage::PIXEL);
    addStageInput(HW::VERTEX_INPUTS, Type::COLOR4, HW::T_IN_COLOR + "_" + index, vs);
    addStageConnector(HW::VERTEX_DATA, Type::COLOR4, HW::T_COLOR + "_" + index, vs, ps);
}

void GeomColorNodeMsl::emitFunctionCall(const ShaderNode& node, GenContext& context, ShaderStage& stage) const
{
    const MslShaderGenerator& shadergen = static_cast<const MslShaderGenerator&>(context.getShaderGenerator());

    const ShaderOutput* output = node.getOutput();
    const ShaderInput* indexInput = node.getInput(INDEX);
    string index = indexInput ? indexInput->getValue()->getValueString() : "0";
    string variable = HW::T_COLOR + "_" + index;

    DEFINE_SHADER_STAGE(stage, Stage::VERTEX)
    {
        VariableBlock& vertexData = stage.getOutputBlock(HW::VERTEX_DATA);
        const string prefix = shadergen.getVertexDataPrefix(vertexData);
        ShaderPort* color = vertexData[variable];
        if (!color->isEmitted())
        {
            color->setEmitted();
            shadergen.emitLine(prefix + color->getVariable() + " = " + HW::T_IN_COLOR + "_" + index, stage);
        }
    }

    DEFINE_SHADER_STAGE(stage, Stage::PIXEL)
    {
        string suffix = "";
        if (*output->getType() == *Type::FLOAT)
        {
            suffix = ".r";
        }
        else if (*output->getType() == *Type::COLOR3)
        {
            suffix = ".rgb";
        }
        VariableBlock& vertexData = stage.getInputBlock(HW::VERTEX_DATA);
        const string prefix = shadergen.getVertexDataPrefix(vertexData);
        ShaderPort* color = vertexData[variable];
        shadergen.emitLineBegin(stage);
        shadergen.emitOutput(node.getOutput(), true, false, context, stage);
        shadergen.emitString(" = " + prefix + color->getVariable() + suffix, stage);
        shadergen.emitLineEnd(stage);
    }
}

MATERIALX_NAMESPACE_END
