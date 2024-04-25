//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXGenGlsl/Nodes/NumLightsNodeGlsl.h>

#include <MaterialXGenShader/Shader.h>

MATERIALX_NAMESPACE_BEGIN

namespace
{

const string NUM_LIGHTS_FUNC_SIGNATURE = "int numActiveLightSources()";

} // anonymous namespace

NumLightsNodeGlsl::NumLightsNodeGlsl()
{
    _hash = std::hash<string>{}(NUM_LIGHTS_FUNC_SIGNATURE);
}

ShaderNodeImplPtr NumLightsNodeGlsl::create()
{
    return std::make_shared<NumLightsNodeGlsl>();
}

void NumLightsNodeGlsl::createVariables(const ShaderNode&, GenContext&, Shader& shader) const
{
    // Create uniform for number of active light sources
    ShaderStage& ps = shader.getStage(Stage::PIXEL);
    ShaderPort* numActiveLights = addStageUniform(HW::PRIVATE_UNIFORMS, Type::INTEGER, HW::T_NUM_ACTIVE_LIGHT_SOURCES, ps);
    numActiveLights->setValue(Value::createValue<int>(0));
}

void NumLightsNodeGlsl::emitFunctionDefinition(const ShaderNode& node, GenContext& context, ShaderStage& stage) const
{
    DEFINE_SHADER_STAGE(stage, Stage::PIXEL)
    {
        const ShaderGenerator& shadergen = context.getShaderGenerator();
        shadergen.emitLine(NUM_LIGHTS_FUNC_SIGNATURE, stage, false);
        shadergen.emitFunctionBodyBegin(node, context, stage);
        shadergen.emitLine("return min(" + HW::T_NUM_ACTIVE_LIGHT_SOURCES + ", " + HW::LIGHT_DATA_MAX_LIGHT_SOURCES + ") ", stage);
        shadergen.emitFunctionBodyEnd(node, context, stage);
    }
}

MATERIALX_NAMESPACE_END
