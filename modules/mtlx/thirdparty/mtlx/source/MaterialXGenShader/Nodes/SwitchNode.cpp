//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXGenShader/Nodes/SwitchNode.h>
#include <MaterialXGenShader/GenContext.h>
#include <MaterialXGenShader/ShaderNode.h>
#include <MaterialXGenShader/ShaderStage.h>
#include <MaterialXGenShader/ShaderGenerator.h>

MATERIALX_NAMESPACE_BEGIN

const StringVec SwitchNode::INPUT_NAMES = { "in1", "in2", "in3", "in4", "in5", "which" };

ShaderNodeImplPtr SwitchNode::create()
{
    return std::make_shared<SwitchNode>();
}

void SwitchNode::emitFunctionCall(const ShaderNode& node, GenContext& context, ShaderStage& stage) const
{
    DEFINE_SHADER_STAGE(stage, Stage::PIXEL)
    {
        const ShaderGenerator& shadergen = context.getShaderGenerator();

        // Declare the output variable
        shadergen.emitLineBegin(stage);
        shadergen.emitOutput(node.getOutput(), true, true, context, stage);
        shadergen.emitLineEnd(stage);

        const ShaderInput* which = node.getInput(INPUT_NAMES[5]);

        // Process the branches of the switch node
        for (int branch = 0; branch < 5; ++branch)
        {
            const ShaderInput* input = node.getInput(INPUT_NAMES[branch]);
            if (!input)
            {
                // The boolean version only has two inputs
                // so break if the input doesn't exist
                break;
            }

            shadergen.emitLineBegin(stage);
            if (branch > 0)
            {
                shadergen.emitString("else ", stage);
            }
            // Convert to float to insure a valid comparison, since the 'which'
            // input may be float, integer or boolean.
            shadergen.emitString("if (float(", stage);
            shadergen.emitInput(which, context, stage);
            shadergen.emitString(") < float(", stage);
            shadergen.emitValue(float(branch + 1), stage);
            shadergen.emitString("))", stage);
            shadergen.emitLineEnd(stage, false);

            shadergen.emitScopeBegin(stage);
            shadergen.emitLineBegin(stage);
            shadergen.emitOutput(node.getOutput(), false, false, context, stage);
            shadergen.emitString(" = ", stage);
            shadergen.emitInput(input, context, stage);
            shadergen.emitLineEnd(stage);
            shadergen.emitScopeEnd(stage);
        }
    }
}

MATERIALX_NAMESPACE_END
