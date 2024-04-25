//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXGenMdl/Nodes/SurfaceNodeMdl.h>

#include <MaterialXGenMdl/MdlShaderGenerator.h>

#include <MaterialXGenShader/GenContext.h>

MATERIALX_NAMESPACE_BEGIN

ShaderNodeImplPtr SurfaceNodeMdl::create()
{
    return std::make_shared<SurfaceNodeMdl>();
}

const ShaderInput* findTransmissionIOR(const ShaderNode& node)
{
    if (node.hasClassification(ShaderNode::Classification::BSDF_T))
    {
        const ShaderInput* ior = node.getInput("ior");
        if (ior)
        {
            bool transparent = true;
            const ShaderInput* scatterMode = node.getInput("scatter_mode");
            if (scatterMode && scatterMode->getValue())
            {
                const string scatterModeValue = scatterMode->getValue()->getValueString();
                transparent = scatterModeValue == "T" || scatterModeValue == "RT";
            }
            if (transparent)
            {
                return ior;
            }
        }
    }
    for (const ShaderInput* input : node.getInputs())
    {
        if (*input->getType() == *Type::BSDF && input->getConnection())
        {
            const ShaderInput* ior = findTransmissionIOR(*input->getConnection()->getNode());
            if (ior)
            {
                return ior;
            }
        }
    }
    return nullptr;
}

void SurfaceNodeMdl::emitFunctionCall(const ShaderNode& node, GenContext& context, ShaderStage& stage) const
{
    DEFINE_SHADER_STAGE(stage, Stage::PIXEL)
    {
        const MdlShaderGenerator& shadergen = static_cast<const MdlShaderGenerator&>(context.getShaderGenerator());

        // Emit calls for the closure dependencies upstream from this node.
        shadergen.emitDependentFunctionCalls(node, context, stage, ShaderNode::Classification::CLOSURE);

        // Check if transmission IOR is used for this shader.
        // MDL only supports a single transmission IOR per material and
        // it is given as an input on the 'material' constructor.
        // So if used we must forward this value/connection to the surface
        // constructor. It's set as an extra input below.
        const ShaderInput* ior = findTransmissionIOR(node);

        shadergen.emitLineBegin(stage);

        // Emit the output and funtion name.
        shadergen.emitOutput(node.getOutput(), true, false, context, stage);
        shadergen.emitString(" = materialx::pbrlib_", stage);
        shadergen.emitMdlVersionFilenameSuffix(context, stage);
        shadergen.emitString("::mx_surface(", stage);

        // Emit all inputs on the node.
        string delim = "";
        for (ShaderInput* input : node.getInputs())
        {
            shadergen.emitString(delim, stage);
            shadergen.emitInput(input, context, stage);
            delim = ", ";
        }

        if (ior)
        {
            // Emit the extra input for transmission IOR.
            shadergen.emitString(delim, stage);
            shadergen.emitInput(ior, context, stage);
        }

        // End function call
        shadergen.emitString(")", stage);
        shadergen.emitLineEnd(stage);
    }
}

MATERIALX_NAMESPACE_END
