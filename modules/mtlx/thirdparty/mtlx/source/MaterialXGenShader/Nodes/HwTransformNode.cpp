//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXGenShader/Nodes/HwTransformNode.h>
#include <MaterialXGenShader/HwShaderGenerator.h>
#include <MaterialXGenShader/Shader.h>

MATERIALX_NAMESPACE_BEGIN

const string HwTransformNode::FROM_SPACE = "fromspace";
const string HwTransformNode::TO_SPACE = "tospace";
const string HwTransformNode::MODEL = "model";
const string HwTransformNode::OBJECT = "object";
const string HwTransformNode::WORLD = "world";

void HwTransformNode::createVariables(const ShaderNode& node, GenContext&, Shader& shader) const
{
    const string toSpace = getToSpace(node);
    const string fromSpace = getFromSpace(node);
    const string& matrix = getMatrix(fromSpace, toSpace);
    if (!matrix.empty())
    {
        ShaderStage& ps = shader.getStage(Stage::PIXEL);
        addStageUniform(HW::PRIVATE_UNIFORMS, Type::MATRIX44, matrix, ps);
    }
}

void HwTransformNode::emitFunctionCall(const ShaderNode& node, GenContext& context, ShaderStage& stage) const
{
    DEFINE_SHADER_STAGE(stage, Stage::PIXEL)
    {
        const ShaderGenerator& shadergen = context.getShaderGenerator();

        const ShaderOutput* output = node.getOutput();
        const ShaderInput* inInput = node.getInput("in");
        if (inInput->getType() != Type::VECTOR3 && inInput->getType() != Type::VECTOR4)
        {
            throw ExceptionShaderGenError("Transform node must have 'in' type of vector3 or vector4.");
        }

        shadergen.emitLineBegin(stage);
        shadergen.emitOutput(output, true, false, context, stage);
        shadergen.emitString(" = (", stage);

        const string toSpace = getToSpace(node);
        const string fromSpace = getFromSpace(node);
        const string& matrix = getMatrix(fromSpace, toSpace);
        if (!matrix.empty())
        {
            shadergen.emitString(matrix + " * ", stage);
        }

        const string type = shadergen.getSyntax().getTypeName(Type::VECTOR4);
        const string input = shadergen.getUpstreamResult(inInput, context);
        shadergen.emitString(type + "(" + input + ", " + getHomogeneousCoordinate() + ")).xyz", stage);
        shadergen.emitLineEnd(stage);

        if (shouldNormalize())
        {
            shadergen.emitLineBegin(stage);
            shadergen.emitOutput(output, false, false, context, stage);
            shadergen.emitString(" = normalize(" + output->getVariable() + ")", stage);
            shadergen.emitLineEnd(stage);
        }
    }
}

string HwTransformNode::getFromSpace(const ShaderNode& node) const
{
    const ShaderInput* input = node.getInput(FROM_SPACE);
    return input ? input->getValueString() : EMPTY_STRING;
}

string HwTransformNode::getToSpace(const ShaderNode& node) const
{
    const ShaderInput* input = node.getInput(TO_SPACE);
    return input ? input->getValueString() : EMPTY_STRING;
}

const string& HwTransformNode::getMatrix(const string& fromSpace, const string& toSpace) const
{
    if ((fromSpace == MODEL || fromSpace == OBJECT) && toSpace == WORLD)
    {
        return getModelToWorldMatrix();
    }
    else if (fromSpace == WORLD && (toSpace == MODEL || toSpace == OBJECT))
    {
        return getWorldToModelMatrix();
    }
    return EMPTY_STRING;
}

ShaderNodeImplPtr HwTransformVectorNode::create()
{
    return std::make_shared<HwTransformVectorNode>();
}

ShaderNodeImplPtr HwTransformPointNode::create()
{
    return std::make_shared<HwTransformPointNode>();
}

ShaderNodeImplPtr HwTransformNormalNode::create()
{
    return std::make_shared<HwTransformNormalNode>();
}

MATERIALX_NAMESPACE_END
