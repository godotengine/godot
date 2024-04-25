//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXGenShader/Nodes/ClosureAddNode.h>
#include <MaterialXGenShader/ShaderGenerator.h>
#include <MaterialXGenShader/GenContext.h>
#include <MaterialXGenShader/TypeDesc.h>

MATERIALX_NAMESPACE_BEGIN

const string ClosureAddNode::IN1 = "in1";
const string ClosureAddNode::IN2 = "in2";

ShaderNodeImplPtr ClosureAddNode::create()
{
    return std::make_shared<ClosureAddNode>();
}

void ClosureAddNode::emitFunctionCall(const ShaderNode& _node, GenContext& context, ShaderStage& stage) const
{
    DEFINE_SHADER_STAGE(stage, Stage::PIXEL)
    {
        const ShaderGenerator& shadergen = context.getShaderGenerator();
        ClosureContext* cct = context.getClosureContext();

        ShaderNode& node = const_cast<ShaderNode&>(_node);

        ShaderInput* in1 = node.getInput(IN1);
        ShaderInput* in2 = node.getInput(IN2);

        // If the add node has closure parameters set,
        // we pass this on to both connected components.

        if (in1->getConnection())
        {
            // Make sure it's a connection to a sibling and not the graph interface.
            ShaderNode* in1Node = in1->getConnection()->getNode();
            if (in1Node->getParent() == node.getParent())
            {
                ScopedSetClosureParams setParams(&node, in1Node, cct);
                shadergen.emitFunctionCall(*in1Node, context, stage);
            }
        }
        if (in2->getConnection())
        {
            // Make sure it's a connection to a sibling and not the graph interface.
            ShaderNode* in2Node = in2->getConnection()->getNode();
            if (in2Node->getParent() == node.getParent())
            {
                ScopedSetClosureParams setParams(&node, in2Node, cct);
                shadergen.emitFunctionCall(*in2Node, context, stage);
            }
        }

        // Get their results.
        const string in1Result = shadergen.getUpstreamResult(in1, context);
        const string in2Result = shadergen.getUpstreamResult(in2, context);

        ShaderOutput* output = node.getOutput();
        if (*output->getType() == *Type::BSDF)
        {
            emitOutputVariables(node, context, stage);
            shadergen.emitLine(output->getVariable() + ".response = " + in1Result + ".response + " + in2Result + ".response", stage);
            shadergen.emitLine(output->getVariable() + ".throughput = " + in1Result + ".throughput * " + in2Result + ".throughput", stage);
        }
        else if (*output->getType() == *Type::EDF)
        {
            shadergen.emitLine(shadergen.getSyntax().getTypeName(Type::EDF) + " " + output->getVariable() + " = " + in1Result + " + " + in2Result, stage);
        }
    }
}

MATERIALX_NAMESPACE_END
