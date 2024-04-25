//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXGenShader/Nodes/ClosureSourceCodeNode.h>
#include <MaterialXGenShader/ShaderGenerator.h>
#include <MaterialXGenShader/GenContext.h>

MATERIALX_NAMESPACE_BEGIN

ShaderNodeImplPtr ClosureSourceCodeNode::create()
{
    return std::make_shared<ClosureSourceCodeNode>();
}

void ClosureSourceCodeNode::emitFunctionCall(const ShaderNode& node, GenContext& context, ShaderStage& stage) const
{
    DEFINE_SHADER_STAGE(stage, Stage::PIXEL)
    {
        const ShaderGenerator& shadergen = context.getShaderGenerator();

        // Emit calls for any closure dependencies upstream from this node.
        shadergen.emitDependentFunctionCalls(node, context, stage, ShaderNode::Classification::CLOSURE);

        if (_inlined)
        {
            SourceCodeNode::emitFunctionCall(node, context, stage);
        }
        else
        {
            const ShaderOutput* output = node.getOutput();
            string delim = "";

            // Declare the output variable.
            emitOutputVariables(node, context, stage);

            // Check if we have a closure context to modify the function call.
            ClosureContext* cct = context.getClosureContext();
            if (cct)
            {
                // Check if extra parameters has been added for this node.
                const TypeDesc* closureType = output->getType();
                const ClosureContext::ClosureParams* params = cct->getClosureParams(&node);
                if (*closureType == *Type::BSDF && params)
                {
                    // Assign the parameters to the BSDF.
                    for (auto it : *params)
                    {
                        shadergen.emitLine(output->getVariable() + "." + it.first + " = " + shadergen.getUpstreamResult(it.second, context), stage);
                    }
                }

                // Emit function name.
                shadergen.emitLineBegin(stage);
                shadergen.emitString(_functionName + cct->getSuffix(closureType) + "(", stage);

                // Emit extra argument.
                for (const ClosureContext::Argument& arg : cct->getArguments(closureType))
                {
                    shadergen.emitString(delim + arg.second, stage);
                    delim = ", ";
                }
            }
            else
            {
                // Emit function name.
                shadergen.emitLineBegin(stage);
                shadergen.emitString(_functionName + "(", stage);
            }

            // Emit all inputs.
            for (ShaderInput* input : node.getInputs())
            {
                shadergen.emitString(delim, stage);
                shadergen.emitInput(input, context, stage);
                delim = ", ";
            }

            // Emit the output.
            shadergen.emitString(delim + node.getOutput()->getVariable() + ")", stage);

            // End function call
            shadergen.emitLineEnd(stage);
        }
    }
}

MATERIALX_NAMESPACE_END
