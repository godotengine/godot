//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXGenMdl/Nodes/MaterialNodeMdl.h>
#include <MaterialXGenMdl/MdlShaderGenerator.h>
#include <MaterialXGenShader/ShaderGenerator.h>
#include <MaterialXGenShader/Shader.h>
#include <MaterialXGenShader/GenContext.h>

MATERIALX_NAMESPACE_BEGIN

ShaderNodeImplPtr MaterialNodeMdl::create()
{
    return std::make_shared<MaterialNodeMdl>();
}

void MaterialNodeMdl::emitFunctionCall(const ShaderNode& _node, GenContext& context, ShaderStage& stage) const
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
        const MdlShaderGenerator& shadergenMdl = static_cast<const MdlShaderGenerator&>(shadergen);

        // Emit the function call for upstream surface shader.
        const ShaderNode* surfaceshaderNode = surfaceshaderInput->getConnection()->getNode();
        shadergen.emitFunctionCall(*surfaceshaderNode, context, stage);

        shadergen.emitLineBegin(stage);

        // Emit the output and funtion name.
        shadergen.emitOutput(node.getOutput(), true, false, context, stage);
        shadergen.emitString(" = materialx::stdlib_", stage);
        shadergenMdl.emitMdlVersionFilenameSuffix(context, stage);
        shadergen.emitString("::mx_surfacematerial(", stage);

        // Emit all inputs on the node.
        string delim = "";
        for (ShaderInput* input : node.getInputs())
        {
            shadergen.emitString(delim, stage);
            shadergen.emitString("mxp_", stage);
            shadergen.emitString(input->getName(), stage);
            shadergen.emitString(": ", stage);
            shadergen.emitInput(input, context, stage);
            delim = ", ";
        }

        // End function call
        shadergen.emitString(")", stage);
        shadergen.emitLineEnd(stage);
    }
}

MATERIALX_NAMESPACE_END
