//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXGenGlsl/Nodes/LightShaderNodeGlsl.h>

#include <MaterialXGenShader/Shader.h>

MATERIALX_NAMESPACE_BEGIN

LightShaderNodeGlsl::LightShaderNodeGlsl() :
    _lightUniforms(HW::LIGHT_DATA, EMPTY_STRING)
{
}

ShaderNodeImplPtr LightShaderNodeGlsl::create()
{
    return std::make_shared<LightShaderNodeGlsl>();
}

const string& LightShaderNodeGlsl::getTarget() const
{
    return GlslShaderGenerator::TARGET;
}

void LightShaderNodeGlsl::initialize(const InterfaceElement& element, GenContext& context)
{
    SourceCodeNode::initialize(element, context);

    if (_inlined)
    {
        throw ExceptionShaderGenError("Light shaders doesn't support inlined implementations'");
    }

    if (!element.isA<Implementation>())
    {
        throw ExceptionShaderGenError("Element '" + element.getName() + "' is not an Implementation element");
    }

    const Implementation& impl = static_cast<const Implementation&>(element);

    // Store light uniforms for all inputs on the interface
    NodeDefPtr nodeDef = impl.getNodeDef();
    for (InputPtr input : nodeDef->getActiveInputs())
    {
        _lightUniforms.add(TypeDesc::get(input->getType()), input->getName(), input->getValue());
    }
}

void LightShaderNodeGlsl::createVariables(const ShaderNode&, GenContext& context, Shader& shader) const
{
    ShaderStage& ps = shader.getStage(Stage::PIXEL);

    // Create all light uniforms
    VariableBlock& lightData = ps.getUniformBlock(HW::LIGHT_DATA);
    for (size_t i = 0; i < _lightUniforms.size(); ++i)
    {
        const ShaderPort* u = _lightUniforms[i];
        lightData.add(u->getType(), u->getName());
    }

    const GlslShaderGenerator& shadergen = static_cast<const GlslShaderGenerator&>(context.getShaderGenerator());
    shadergen.addStageLightingUniforms(context, ps);
}

void LightShaderNodeGlsl::emitFunctionCall(const ShaderNode&, GenContext& context, ShaderStage& stage) const
{
    DEFINE_SHADER_STAGE(stage, Stage::PIXEL)
    {
        const ShaderGenerator& shadergen = context.getShaderGenerator();
        shadergen.emitLine(_functionName + "(light, position, result)", stage);
    }
}

MATERIALX_NAMESPACE_END
