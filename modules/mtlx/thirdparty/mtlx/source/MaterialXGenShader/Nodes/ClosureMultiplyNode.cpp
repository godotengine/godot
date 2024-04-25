//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXGenShader/Nodes/ClosureMultiplyNode.h>
#include <MaterialXGenShader/ShaderGenerator.h>
#include <MaterialXGenShader/GenContext.h>
#include <MaterialXGenShader/TypeDesc.h>

MATERIALX_NAMESPACE_BEGIN

const string ClosureMultiplyNode::IN1 = "in1";
const string ClosureMultiplyNode::IN2 = "in2";

ShaderNodeImplPtr ClosureMultiplyNode::create()
{
    return std::make_shared<ClosureMultiplyNode>();
}

void ClosureMultiplyNode::emitFunctionCall(const ShaderNode& _node, GenContext& context, ShaderStage& stage) const
{
    DEFINE_SHADER_STAGE(stage, Stage::PIXEL)
    {
        const ShaderGenerator& shadergen = context.getShaderGenerator();
        const Syntax& syntax = shadergen.getSyntax();
        ClosureContext* cct = context.getClosureContext();

        ShaderNode& node = const_cast<ShaderNode&>(_node);

        ShaderInput* in1 = node.getInput(IN1);
        ShaderInput* in2 = node.getInput(IN2);

        // If the multiply node has closure parameters set,
        // we pass this on to the in1 closure component.

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

        // Get their results.
        const string in1Result = shadergen.getUpstreamResult(in1, context);
        const string in2Result = shadergen.getUpstreamResult(in2, context);

        ShaderOutput* output = node.getOutput();
        if (*output->getType() == *Type::BSDF)
        {
            const string in2clamped = output->getVariable() + "_in2_clamped";
            shadergen.emitLine(syntax.getTypeName(in2->getType()) + " " + in2clamped + " = clamp(" + in2Result + ", 0.0, 1.0)", stage);

            emitOutputVariables(node, context, stage);
            shadergen.emitLine(output->getVariable() + ".response = " + in1Result + ".response * " + in2clamped, stage);
            shadergen.emitLine(output->getVariable() + ".throughput = " + in1Result + ".throughput * " + in2clamped, stage);
        }
        else if (*output->getType() == *Type::EDF)
        {
            shadergen.emitLine(shadergen.getSyntax().getTypeName(Type::EDF) + " " + output->getVariable() + " = " + in1Result + " * " + in2Result, stage);
        }
    }
}

MATERIALX_NAMESPACE_END
