//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXGenShader/Nodes/CompoundNode.h>
#include <MaterialXGenShader/ShaderGenerator.h>
#include <MaterialXGenShader/HwShaderGenerator.h>
#include <MaterialXGenShader/Util.h>

#include <MaterialXCore/Library.h>
#include <MaterialXCore/Definition.h>
#include <MaterialXCore/Document.h>

MATERIALX_NAMESPACE_BEGIN

ShaderNodeImplPtr CompoundNode::create()
{
    return std::make_shared<CompoundNode>();
}

void CompoundNode::initialize(const InterfaceElement& element, GenContext& context)
{
    ShaderNodeImpl::initialize(element, context);

    if (!element.isA<NodeGraph>())
    {
        throw ExceptionShaderGenError("Element '" + element.getName() + "' is not a node graph implementation");
    }

    const NodeGraph& graph = static_cast<const NodeGraph&>(element);

    _functionName = graph.getName();
    context.getShaderGenerator().getSyntax().makeValidName(_functionName);

    // For compounds we do not want to publish all internal inputs
    // so always use the reduced interface for this graph.
    const ShaderInterfaceType oldShaderInterfaceType = context.getOptions().shaderInterfaceType;
    context.getOptions().shaderInterfaceType = SHADER_INTERFACE_REDUCED;
    _rootGraph = ShaderGraph::create(nullptr, graph, context);
    context.getOptions().shaderInterfaceType = oldShaderInterfaceType;

    // Set hash using the function name.
    // TODO: Could be improved to include the full function signature.
    _hash = std::hash<string>{}(_functionName);
}

void CompoundNode::createVariables(const ShaderNode&, GenContext& context, Shader& shader) const
{
    // Gather shader inputs from all child nodes
    for (const ShaderNode* childNode : _rootGraph->getNodes())
    {
        childNode->getImplementation().createVariables(*childNode, context, shader);
    }
}

void CompoundNode::emitFunctionDefinition(const ShaderNode&, GenContext& context, ShaderStage& stage) const
{
    DEFINE_SHADER_STAGE(stage, Stage::PIXEL)
    {
        const ShaderGenerator& shadergen = context.getShaderGenerator();
        const Syntax& syntax = shadergen.getSyntax();

        // Emit functions for all child nodes
        shadergen.emitFunctionDefinitions(*_rootGraph, context, stage);

        // Begin function signature.
        shadergen.emitLineBegin(stage);
        shadergen.emitString("void " + _functionName + +"(", stage);

        string delim = "";

        // Add all inputs
        for (ShaderGraphInputSocket* inputSocket : _rootGraph->getInputSockets())
        {
            shadergen.emitString(delim + syntax.getTypeName(inputSocket->getType()) + " " + inputSocket->getVariable(), stage);
            delim = ", ";
        }

        // Add all outputs
        for (ShaderGraphOutputSocket* outputSocket : _rootGraph->getOutputSockets())
        {
            shadergen.emitString(delim + syntax.getOutputTypeName(outputSocket->getType()) + " " + outputSocket->getVariable(), stage);
            delim = ", ";
        }

        // End function signature.
        shadergen.emitString(")", stage);
        shadergen.emitLineEnd(stage, false);

        // Begin function body.
        shadergen.emitFunctionBodyBegin(*_rootGraph, context, stage);
        shadergen.emitFunctionCalls(*_rootGraph, context, stage);

        // Emit final results
        for (ShaderGraphOutputSocket* outputSocket : _rootGraph->getOutputSockets())
        {
            const string result = shadergen.getUpstreamResult(outputSocket, context);
            shadergen.emitLine(outputSocket->getVariable() + " = " + result, stage);
        }

        // End function body.
        shadergen.emitFunctionBodyEnd(*_rootGraph, context, stage);
    }
}

void CompoundNode::emitFunctionCall(const ShaderNode& node, GenContext& context, ShaderStage& stage) const
{
    const ShaderGenerator& shadergen = context.getShaderGenerator();

    DEFINE_SHADER_STAGE(stage, Stage::VERTEX)
    {
        // Emit function calls for all child nodes to the vertex shader stage
        shadergen.emitFunctionCalls(*_rootGraph, context, stage);
    }

    DEFINE_SHADER_STAGE(stage, Stage::PIXEL)
    {
        // Declare the output variables.
        emitOutputVariables(node, context, stage);

        // Begin function call.
        shadergen.emitLineBegin(stage);
        shadergen.emitString(_functionName + "(", stage);

        string delim = "";

        // Emit inputs.
        for (ShaderInput* input : node.getInputs())
        {
            shadergen.emitString(delim, stage);
            shadergen.emitInput(input, context, stage);
            delim = ", ";
        }

        // Emit outputs.
        for (size_t i = 0; i < node.numOutputs(); ++i)
        {
            shadergen.emitString(delim, stage);
            shadergen.emitOutput(node.getOutput(i), false, false, context, stage);
            delim = ", ";
        }

        // End function call
        shadergen.emitString(")", stage);
        shadergen.emitLineEnd(stage);
    }
}

MATERIALX_NAMESPACE_END
