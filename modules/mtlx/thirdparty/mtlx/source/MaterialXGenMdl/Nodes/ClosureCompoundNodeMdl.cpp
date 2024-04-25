//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXGenMdl/Nodes/ClosureCompoundNodeMdl.h>

#include <MaterialXGenShader/HwShaderGenerator.h>
#include <MaterialXGenShader/ShaderGenerator.h>
#include <MaterialXGenShader/Util.h>

#include <MaterialXCore/Definition.h>

MATERIALX_NAMESPACE_BEGIN

ShaderNodeImplPtr ClosureCompoundNodeMdl::create()
{
    return std::make_shared<ClosureCompoundNodeMdl>();
}

void ClosureCompoundNodeMdl::addClassification(ShaderNode& node) const
{
    // Add classification from the graph implementation.
    node.addClassification(_rootGraph->getClassification());
}

void ClosureCompoundNodeMdl::emitFunctionDefinition(const ShaderNode& node, GenContext& context, ShaderStage& stage) const
{
    DEFINE_SHADER_STAGE(stage, Stage::PIXEL)
    {
        const ShaderGenerator& shadergen = context.getShaderGenerator();

        // Emit functions for all child nodes
        shadergen.emitFunctionDefinitions(*_rootGraph, context, stage);

        // split all fields into separate functions
        if (!_returnStruct.empty() && _unrollReturnStructMembers)
        {
            // make sure the upstream definitions are known
            for (const ShaderGraphOutputSocket* outputSocket : _rootGraph->getOutputSockets())
            {
                if (!outputSocket->getConnection())
                    continue;

                const ShaderNode* upstream = outputSocket->getConnection()->getNode();
                const bool isMaterialExpr = (upstream->hasClassification(ShaderNode::Classification::CLOSURE) ||
                                             upstream->hasClassification(ShaderNode::Classification::SHADER));

                // since the emit fuctions are const, the field name to generate a function for is passed via context
                const std::string& fieldName = outputSocket->getName();
                GenUserDataStringPtr fieldNamePtr = std::make_shared<GenUserDataString>(fieldName);
                context.pushUserData(CompoundNodeMdl::GEN_USER_DATA_RETURN_STRUCT_FIELD_NAME, fieldNamePtr);

                // Emit function signature.
                shadergen.emitComment("unrolled structure field: " + _returnStruct + "." + fieldName + " (name=\"" + node.getName() + "\")", stage);
                emitFunctionSignature(node, context, stage);

                // Special case for material expresions.
                if (isMaterialExpr)
                {
                    shadergen.emitLine(" = let", stage, false);
                }

                // Function body.
                shadergen.emitScopeBegin(stage);

                // Emit all texturing nodes. These are inputs to the
                // closure nodes and need to be emitted first.
                shadergen.emitFunctionCalls(*_rootGraph, context, stage, ShaderNode::Classification::TEXTURE);

                // Emit function calls for internal closures nodes connected to the graph sockets.
                // These will in turn emit function calls for any dependent closure nodes upstream.
                if (upstream->getParent() == _rootGraph.get() &&
                    (upstream->hasClassification(ShaderNode::Classification::CLOSURE) || upstream->hasClassification(ShaderNode::Classification::SHADER)))
                {
                    shadergen.emitFunctionCall(*upstream, context, stage);
                }

                // Emit final results
                if (isMaterialExpr)
                {
                    shadergen.emitScopeEnd(stage);
                    const string result = shadergen.getUpstreamResult(outputSocket, context);
                    shadergen.emitLine("in material(" + result + ")", stage);
                }
                else
                {
                    const string result = shadergen.getUpstreamResult(outputSocket, context);
                    shadergen.emitLine("return " + result, stage);
                }
                shadergen.emitLineBreak(stage);

                context.popUserData(CompoundNodeMdl::GEN_USER_DATA_RETURN_STRUCT_FIELD_NAME);
            }
            return;
        }

        const bool isMaterialExpr = (_rootGraph->hasClassification(ShaderNode::Classification::CLOSURE) ||
                                     _rootGraph->hasClassification(ShaderNode::Classification::SHADER));

        // Emit function signature.
        emitFunctionSignature(node, context, stage);

        // Special case for material expresions.
        if (isMaterialExpr)
        {
            shadergen.emitLine(" = let", stage, false);
        }

        // Function body.
        shadergen.emitScopeBegin(stage);

        // Emit all texturing nodes. These are inputs to the
        // closure nodes and need to be emitted first.
        shadergen.emitFunctionCalls(*_rootGraph, context, stage, ShaderNode::Classification::TEXTURE);

        // Emit function calls for internal closures nodes connected to the graph sockets.
        // These will in turn emit function calls for any dependent closure nodes upstream.
        for (ShaderGraphOutputSocket* outputSocket : _rootGraph->getOutputSockets())
        {
            if (outputSocket->getConnection())
            {
                const ShaderNode* upstream = outputSocket->getConnection()->getNode();
                if (upstream->getParent() == _rootGraph.get() &&
                    (upstream->hasClassification(ShaderNode::Classification::CLOSURE) || upstream->hasClassification(ShaderNode::Classification::SHADER)))
                {
                    shadergen.emitFunctionCall(*upstream, context, stage);
                }
            }
        }

        // Emit final results
        if (isMaterialExpr)
        {
            shadergen.emitScopeEnd(stage);
            const ShaderGraphOutputSocket* outputSocket = _rootGraph->getOutputSocket();
            const string result = shadergen.getUpstreamResult(outputSocket, context);
            shadergen.emitLine("in material(" + result + ")", stage);
        }
        else
        {
            if (!_returnStruct.empty())
            {
                const string resultVariableName = "result__";
                shadergen.emitLine(_returnStruct + " " + resultVariableName, stage);
                for (const ShaderGraphOutputSocket* output : _rootGraph->getOutputSockets())
                {
                    const string result = shadergen.getUpstreamResult(output, context);
                    shadergen.emitLine(resultVariableName + ".mxp_" + output->getName() + " = " + result, stage);
                }
                shadergen.emitLine("return " + resultVariableName, stage);
            }
            else
            {
                const ShaderGraphOutputSocket* outputSocket = _rootGraph->getOutputSocket();
                const string result = shadergen.getUpstreamResult(outputSocket, context);
                shadergen.emitLine("return " + result, stage);
            }
            shadergen.emitScopeEnd(stage);
        }

        shadergen.emitLineBreak(stage);
    }
}

void ClosureCompoundNodeMdl::emitFunctionCall(const ShaderNode& node, GenContext& context, ShaderStage& stage) const
{
    DEFINE_SHADER_STAGE(stage, Stage::PIXEL)
    {
        const ShaderGenerator& shadergen = context.getShaderGenerator();

        // First emit calls for any closure dependencies upstream from this node.
        shadergen.emitDependentFunctionCalls(node, context, stage, ShaderNode::Classification::CLOSURE);

        // Then emit this nodes function call.
        CompoundNodeMdl::emitFunctionCall(node, context, stage);
    }
}

MATERIALX_NAMESPACE_END
