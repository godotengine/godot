//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXGenGlsl/Nodes/SurfaceNodeGlsl.h>
#include <MaterialXGenGlsl/GlslShaderGenerator.h>

#include <MaterialXGenShader/Shader.h>
#include <MaterialXGenShader/GenContext.h>

MATERIALX_NAMESPACE_BEGIN

SurfaceNodeGlsl::SurfaceNodeGlsl() :
    _callReflection(HwShaderGenerator::ClosureContextType::REFLECTION),
    _callTransmission(HwShaderGenerator::ClosureContextType::TRANSMISSION),
    _callIndirect(HwShaderGenerator::ClosureContextType::INDIRECT),
    _callEmission(HwShaderGenerator::ClosureContextType::EMISSION)
{
    // Create closure contexts for calling closure functions.
    //
    // Reflection context
    _callReflection.setSuffix(Type::BSDF, HwShaderGenerator::CLOSURE_CONTEXT_SUFFIX_REFLECTION);
    _callReflection.addArgument(Type::BSDF, ClosureContext::Argument(Type::VECTOR3, HW::DIR_L));
    _callReflection.addArgument(Type::BSDF, ClosureContext::Argument(Type::VECTOR3, HW::DIR_V));
    _callReflection.addArgument(Type::BSDF, ClosureContext::Argument(Type::VECTOR3, HW::WORLD_POSITION));
    _callReflection.addArgument(Type::BSDF, ClosureContext::Argument(Type::FLOAT, HW::OCCLUSION));
    // Transmission context
    _callTransmission.setSuffix(Type::BSDF, HwShaderGenerator::CLOSURE_CONTEXT_SUFFIX_TRANSMISSION);
    _callTransmission.addArgument(Type::BSDF, ClosureContext::Argument(Type::VECTOR3, HW::DIR_V));
    // Indirect/Environment context
    _callIndirect.setSuffix(Type::BSDF, HwShaderGenerator::CLOSURE_CONTEXT_SUFFIX_INDIRECT);
    _callIndirect.addArgument(Type::BSDF, ClosureContext::Argument(Type::VECTOR3, HW::DIR_V));
    // Emission context
    _callEmission.addArgument(Type::EDF, ClosureContext::Argument(Type::VECTOR3, HW::DIR_N));
    _callEmission.addArgument(Type::EDF, ClosureContext::Argument(Type::VECTOR3, HW::DIR_V));
}

ShaderNodeImplPtr SurfaceNodeGlsl::create()
{
    return std::make_shared<SurfaceNodeGlsl>();
}

void SurfaceNodeGlsl::createVariables(const ShaderNode&, GenContext& context, Shader& shader) const
{
    // TODO:
    // The surface shader needs position, normal, view position and light sources. We should solve this by adding some
    // dependency mechanism so this implementation can be set to depend on the HwPositionNode, HwNormalNode
    // HwViewDirectionNode and LightNodeGlsl nodes instead? This is where the MaterialX attribute "internalgeomprops"
    // is needed.
    //
    ShaderStage& vs = shader.getStage(Stage::VERTEX);
    ShaderStage& ps = shader.getStage(Stage::PIXEL);

    addStageInput(HW::VERTEX_INPUTS, Type::VECTOR3, HW::T_IN_POSITION, vs);
    addStageInput(HW::VERTEX_INPUTS, Type::VECTOR3, HW::T_IN_NORMAL, vs);
    addStageUniform(HW::PRIVATE_UNIFORMS, Type::MATRIX44, HW::T_WORLD_INVERSE_TRANSPOSE_MATRIX, vs);

    addStageConnector(HW::VERTEX_DATA, Type::VECTOR3, HW::T_POSITION_WORLD, vs, ps);
    addStageConnector(HW::VERTEX_DATA, Type::VECTOR3, HW::T_NORMAL_WORLD, vs, ps);

    addStageUniform(HW::PRIVATE_UNIFORMS, Type::VECTOR3, HW::T_VIEW_POSITION, ps);

    const GlslShaderGenerator& shadergen = static_cast<const GlslShaderGenerator&>(context.getShaderGenerator());
    shadergen.addStageLightingUniforms(context, ps);
}

void SurfaceNodeGlsl::emitFunctionCall(const ShaderNode& node, GenContext& context, ShaderStage& stage) const
{
    const GlslShaderGenerator& shadergen = static_cast<const GlslShaderGenerator&>(context.getShaderGenerator());

    DEFINE_SHADER_STAGE(stage, Stage::VERTEX)
    {
        VariableBlock& vertexData = stage.getOutputBlock(HW::VERTEX_DATA);
        const string prefix = shadergen.getVertexDataPrefix(vertexData);
        ShaderPort* position = vertexData[HW::T_POSITION_WORLD];
        if (!position->isEmitted())
        {
            position->setEmitted();
            shadergen.emitLine(prefix + position->getVariable() + " = hPositionWorld.xyz", stage);
        }
        ShaderPort* normal = vertexData[HW::T_NORMAL_WORLD];
        if (!normal->isEmitted())
        {
            normal->setEmitted();
            shadergen.emitLine(prefix + normal->getVariable() + " = normalize((" + HW::T_WORLD_INVERSE_TRANSPOSE_MATRIX + " * vec4(" + HW::T_IN_NORMAL + ", 0)).xyz)", stage);
        }
        if (context.getOptions().hwAmbientOcclusion)
        {
            ShaderPort* texcoord = vertexData[HW::T_TEXCOORD + "_0"];
            if (!texcoord->isEmitted())
            {
                texcoord->setEmitted();
                shadergen.emitLine(prefix + texcoord->getVariable() + " = " + HW::T_IN_TEXCOORD + "_0", stage);
            }
        }
    }

    DEFINE_SHADER_STAGE(stage, Stage::PIXEL)
    {
        VariableBlock& vertexData = stage.getInputBlock(HW::VERTEX_DATA);
        const string prefix = shadergen.getVertexDataPrefix(vertexData);

        // Declare the output variable
        const ShaderOutput* output = node.getOutput();
        shadergen.emitLineBegin(stage);
        shadergen.emitOutput(output, true, true, context, stage);
        shadergen.emitLineEnd(stage);

        shadergen.emitScopeBegin(stage);

        shadergen.emitLine("vec3 N = normalize(" + prefix + HW::T_NORMAL_WORLD + ")", stage);
        shadergen.emitLine("vec3 V = normalize(" + HW::T_VIEW_POSITION + " - " + prefix + HW::T_POSITION_WORLD + ")", stage);
        shadergen.emitLine("vec3 P = " + prefix + HW::T_POSITION_WORLD, stage);
        shadergen.emitLineBreak(stage);

        const string outColor = output->getVariable() + ".color";
        const string outTransparency = output->getVariable() + ".transparency";

        const ShaderInput* bsdfInput = node.getInput("bsdf");
        const ShaderNode* bsdf = bsdfInput->getConnectedSibling();
        if (bsdf)
        {
            shadergen.emitLineBegin(stage);
            shadergen.emitString("float surfaceOpacity = ", stage);
            shadergen.emitInput(node.getInput("opacity"), context, stage);
            shadergen.emitLineEnd(stage);
            shadergen.emitLineBreak(stage);

            //
            // Handle direct lighting
            //
            shadergen.emitComment("Shadow occlusion", stage);
            if (context.getOptions().hwShadowMap)
            {
                shadergen.emitLine("vec3 shadowCoord = (" + HW::T_SHADOW_MATRIX + " * vec4(" + prefix + HW::T_POSITION_WORLD + ", 1.0)).xyz", stage);
                shadergen.emitLine("shadowCoord = shadowCoord * 0.5 + 0.5", stage);
                shadergen.emitLine("vec2 shadowMoments = texture(" + HW::T_SHADOW_MAP + ", shadowCoord.xy).xy", stage);
                shadergen.emitLine("float occlusion = mx_variance_shadow_occlusion(shadowMoments, shadowCoord.z)", stage);
            }
            else
            {
                shadergen.emitLine("float occlusion = 1.0", stage);
            }
            shadergen.emitLineBreak(stage);

            emitLightLoop(node, context, stage, outColor);

            //
            // Handle indirect lighting.
            //
            shadergen.emitComment("Ambient occlusion", stage);
            if (context.getOptions().hwAmbientOcclusion)
            {
                ShaderPort* texcoord = vertexData[HW::T_TEXCOORD + "_0"];
                shadergen.emitLine("vec2 ambOccUv = mx_transform_uv(" + prefix + texcoord->getVariable() + ", vec2(1.0), vec2(0.0))", stage);
                shadergen.emitLine("occlusion = mix(1.0, texture(" + HW::T_AMB_OCC_MAP + ", ambOccUv).x, " + HW::T_AMB_OCC_GAIN + ")", stage);
            }
            else
            {
                shadergen.emitLine("occlusion = 1.0", stage);
            }
            shadergen.emitLineBreak(stage);

            shadergen.emitComment("Add environment contribution", stage);
            shadergen.emitScopeBegin(stage);

            context.pushClosureContext(&_callIndirect);
            shadergen.emitFunctionCall(*bsdf, context, stage);
            context.popClosureContext();

            shadergen.emitLineBreak(stage);
            shadergen.emitLine(outColor + " += occlusion * " + bsdf->getOutput()->getVariable() + ".response", stage);
            shadergen.emitScopeEnd(stage);
            shadergen.emitLineBreak(stage);
        }

        //
        // Handle surface emission.
        //
        const ShaderInput* edfInput = node.getInput("edf");
        const ShaderNode* edf = edfInput->getConnectedSibling();
        if (edf)
        {
            shadergen.emitComment("Add surface emission", stage);
            shadergen.emitScopeBegin(stage);

            context.pushClosureContext(&_callEmission);
            shadergen.emitFunctionCall(*edf, context, stage);
            context.popClosureContext();

            shadergen.emitLine(outColor + " += " + edf->getOutput()->getVariable(), stage);
            shadergen.emitScopeEnd(stage);
            shadergen.emitLineBreak(stage);
        }

        //
        // Handle surface transmission and opacity.
        //
        if (bsdf)
        {
            shadergen.emitComment("Calculate the BSDF transmission for viewing direction", stage);
            shadergen.emitScopeBegin(stage);
            context.pushClosureContext(&_callTransmission);
            shadergen.emitFunctionCall(*bsdf, context, stage);
            if (context.getOptions().hwTransmissionRenderMethod == TRANSMISSION_REFRACTION)
            {
                shadergen.emitLine(outColor + " += " + bsdf->getOutput()->getVariable() + ".response", stage);
            }
            else
            {
                shadergen.emitLine(outTransparency + " += " + bsdf->getOutput()->getVariable() + ".response", stage);
            }
            shadergen.emitScopeEnd(stage);
            context.popClosureContext();

            shadergen.emitLineBreak(stage);
            shadergen.emitComment("Compute and apply surface opacity", stage);
            shadergen.emitScopeBegin(stage);
            shadergen.emitLine(outColor + " *= surfaceOpacity", stage);
            shadergen.emitLine(outTransparency + " = mix(vec3(1.0), " + outTransparency + ", surfaceOpacity)", stage);
            shadergen.emitScopeEnd(stage);
        }

        shadergen.emitScopeEnd(stage);
        shadergen.emitLineBreak(stage);
    }
}

void SurfaceNodeGlsl::emitLightLoop(const ShaderNode& node, GenContext& context, ShaderStage& stage, const string& outColor) const
{
    //
    // Generate Light loop if requested
    //
    if (context.getOptions().hwMaxActiveLightSources > 0)
    {
        const GlslShaderGenerator& shadergen = static_cast<const GlslShaderGenerator&>(context.getShaderGenerator());
        const VariableBlock& vertexData = stage.getInputBlock(HW::VERTEX_DATA);
        const string prefix = shadergen.getVertexDataPrefix(vertexData);

        const ShaderInput* bsdfInput = node.getInput("bsdf");
        const ShaderNode* bsdf = bsdfInput->getConnectedSibling();

        shadergen.emitComment("Light loop", stage);
        shadergen.emitLine("int numLights = numActiveLightSources()", stage);
        shadergen.emitLine("lightshader lightShader", stage);
        shadergen.emitLine("for (int activeLightIndex = 0; activeLightIndex < numLights; ++activeLightIndex)", stage, false);

        shadergen.emitScopeBegin(stage);

        shadergen.emitLine("sampleLightSource(" + HW::T_LIGHT_DATA_INSTANCE + "[activeLightIndex], " + prefix + HW::T_POSITION_WORLD + ", lightShader)", stage);
        shadergen.emitLine("vec3 L = lightShader.direction", stage);
        shadergen.emitLineBreak(stage);

        shadergen.emitComment("Calculate the BSDF response for this light source", stage);
        context.pushClosureContext(&_callReflection);
        shadergen.emitFunctionCall(*bsdf, context, stage);
        context.popClosureContext();

        shadergen.emitLineBreak(stage);

        shadergen.emitComment("Accumulate the light's contribution", stage);
        shadergen.emitLine(outColor + " += lightShader.intensity * " + bsdf->getOutput()->getVariable() + ".response", stage);

        shadergen.emitScopeEnd(stage);
        shadergen.emitLineBreak(stage);
    }
}

MATERIALX_NAMESPACE_END
