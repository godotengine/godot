//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXGenGlsl/Nodes/LightNodeGlsl.h>

#include <MaterialXGenShader/Shader.h>

MATERIALX_NAMESPACE_BEGIN

namespace
{

const string LIGHT_DIRECTION_CALCULATION =
    "vec3 L = light.position - position;\n"
    "float distance = length(L);\n"
    "L /= distance;\n"
    "result.direction = L;\n";

} // anonymous namespace

LightNodeGlsl::LightNodeGlsl() :
    _callEmission(HwShaderGenerator::ClosureContextType::EMISSION)
{
    // Emission context
    _callEmission.addArgument(Type::EDF, ClosureContext::Argument(Type::VECTOR3, "light.direction"));
    _callEmission.addArgument(Type::EDF, ClosureContext::Argument(Type::VECTOR3, "-L"));
}

ShaderNodeImplPtr LightNodeGlsl::create()
{
    return std::make_shared<LightNodeGlsl>();
}

void LightNodeGlsl::createVariables(const ShaderNode&, GenContext& context, Shader& shader) const
{
    ShaderStage& ps = shader.getStage(Stage::PIXEL);

    // Create uniform for intensity, exposure and direction
    VariableBlock& lightUniforms = ps.getUniformBlock(HW::LIGHT_DATA);
    lightUniforms.add(Type::FLOAT, "intensity", Value::createValue<float>(1.0f));
    lightUniforms.add(Type::FLOAT, "exposure", Value::createValue<float>(0.0f));
    lightUniforms.add(Type::VECTOR3, "direction", Value::createValue<Vector3>(Vector3(0.0f, 1.0f, 0.0f)));

    const GlslShaderGenerator& shadergen = static_cast<const GlslShaderGenerator&>(context.getShaderGenerator());
    shadergen.addStageLightingUniforms(context, ps);
}

void LightNodeGlsl::emitFunctionCall(const ShaderNode& node, GenContext& context, ShaderStage& stage) const
{
    DEFINE_SHADER_STAGE(stage, Stage::PIXEL)
    {
        const GlslShaderGenerator& shadergen = static_cast<const GlslShaderGenerator&>(context.getShaderGenerator());

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
            shadergen.emitLine("result.intensity = vec3(0.0)", stage);
        }
    }
}

MATERIALX_NAMESPACE_END
