//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXGenMsl/Nodes/LightNodeMsl.h>

#include <MaterialXGenShader/Shader.h>

MATERIALX_NAMESPACE_BEGIN

namespace
{

const string LIGHT_DIRECTION_CALCULATION =
    "float3 L = light.position - position;\n"
    "float distance = length(L);\n"
    "L /= distance;\n"
    "result.direction = L;\n";

}

LightNodeMsl::LightNodeMsl() :
    _callEmission(HwShaderGenerator::ClosureContextType::EMISSION)
{
    // Emission context
    _callEmission.addArgument(Type::EDF, ClosureContext::Argument(Type::VECTOR3, "light.direction"));
    _callEmission.addArgument(Type::EDF, ClosureContext::Argument(Type::VECTOR3, "-L"));
}

ShaderNodeImplPtr LightNodeMsl::create()
{
    return std::make_shared<LightNodeMsl>();
}

void LightNodeMsl::createVariables(const ShaderNode&, GenContext& context, Shader& shader) const
{
    ShaderStage& ps = shader.getStage(Stage::PIXEL);

    // Create uniform for intensity, exposure and direction
    VariableBlock& lightUniforms = ps.getUniformBlock(HW::LIGHT_DATA);
    lightUniforms.add(Type::FLOAT, "intensity", Value::createValue<float>(1.0f));
    lightUniforms.add(Type::FLOAT, "exposure", Value::createValue<float>(0.0f));
    lightUniforms.add(Type::VECTOR3, "direction", Value::createValue<Vector3>(Vector3(0.0f, 1.0f, 0.0f)));

    const MslShaderGenerator& shadergen = static_cast<const MslShaderGenerator&>(context.getShaderGenerator());
    shadergen.addStageLightingUniforms(context, ps);
}

void LightNodeMsl::emitFunctionCall(const ShaderNode& node, GenContext& context, ShaderStage& stage) const
{
    DEFINE_SHADER_STAGE(stage, Stage::PIXEL)
    {
        const MslShaderGenerator& shadergen = static_cast<const MslShaderGenerator&>(context.getShaderGenerator());

        shadergen.emitBlock(LIGHT_DIRECTION_CALCULATION, FilePath(), context, stage);
        shadergen.emitLineBreak(stage);

        const ShaderInput* edfInput = node.getInput("edf");
        const ShaderNode* edf = edfInput->getConnectedSibling();
        if (edf)
        {
            context.pushClosureContext(&_callEmission);
            shadergen.emitFunctionCall(*edf, context, stage);
            context.popClosureContext();

            shadergen.emitLineBreak(stage);

            shadergen.emitComment("Apply quadratic falloff and adjust intensity", stage);
            shadergen.emitLine("result.intensity = " + edf->getOutput()->getVariable() + " / (distance * distance)", stage);

            const ShaderInput* intensity = node.getInput("intensity");
            const ShaderInput* exposure = node.getInput("exposure");

            shadergen.emitLineBegin(stage);
            shadergen.emitString("result.intensity *= ", stage);
            shadergen.emitInput(intensity, context, stage);
            shadergen.emitLineEnd(stage);

            // Emit exposure adjustment only if it matters
            if (exposure->getConnection() || (exposure->getValue() && exposure->getValue()->asA<float>() != 0.0f))
            {
                shadergen.emitLineBegin(stage);
                shadergen.emitString("result.intensity *= pow(2, ", stage);
                shadergen.emitInput(exposure, context, stage);
                shadergen.emitString(")", stage);
                shadergen.emitLineEnd(stage);
            }
        }
        else
        {
            shadergen.emitLine("result.intensity = float3(0.0)", stage);
        }
    }
}

MATERIALX_NAMESPACE_END
