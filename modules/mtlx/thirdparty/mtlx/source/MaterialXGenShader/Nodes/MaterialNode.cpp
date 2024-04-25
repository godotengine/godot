//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXGenShader/Nodes/MaterialNode.h>
#include <MaterialXGenShader/ShaderGenerator.h>
#include <MaterialXGenShader/Shader.h>
#include <MaterialXGenShader/GenContext.h>

MATERIALX_NAMESPACE_BEGIN

ShaderNodeImplPtr MaterialNode::create()
{
    return std::make_shared<MaterialNode>();
}

void MaterialNode::addClassification(ShaderNode& node) const
{
    const ShaderInput* surfaceshaderInput = node.getInput(ShaderNode::SURFACESHADER);
    if (surfaceshaderInput && surfaceshaderInput->getConnection())
    {
        // This is a material node with a surfaceshader connected.
        // Add the classification from this shader.
        const ShaderNode* surfaceshaderNode = surfaceshaderInput->getConnection()->getNode();
        node.addClassification(surfaceshaderNode->getClassification());
    }
}

void MaterialNode::emitFunctionCall(const ShaderNode& _node, GenContext& context, ShaderStage& stage) const
{
    DEFINE_SHADER_STAGE(stage, Stage::PIXEL)
    {
        ShaderNode& node = const_cast<ShaderNode&>(_node);
        ShaderInput* surfaceshaderInput = node.getInput(ShaderNode::SURFACESHADER);

        if (!surfaceshaderInput->getConnection())
        {
            // Just declare the output variable with default value.
            emitOutputVariables(node, context, stage);
            return;
        }

        const ShaderGenerator& shadergen = context.getShaderGenerator();
        const Syntax& syntax = shadergen.getSyntax();

        // Emit the function call for upstream surface shader.
        const ShaderNode* surfaceshaderNode = surfaceshaderInput->getConnection()->getNode();
        shadergen.emitFunctionCall(*surfaceshaderNode, context, stage);

        // Assing this result to the material output variable.
        const ShaderOutput* output = node.getOutput();
        shadergen.emitLine(syntax.getTypeName(output->getType()) + " " + output->getVariable() + " = " + surfaceshaderInput->getConnection()->getVariable(), stage);
    }
}

MATERIALX_NAMESPACE_END
