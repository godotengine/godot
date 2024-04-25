//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXGenGlsl/Nodes/SurfaceShaderNodeGlsl.h>
#include <MaterialXGenGlsl/GlslShaderGenerator.h>

#include <MaterialXGenShader/Shader.h>

MATERIALX_NAMESPACE_BEGIN

ShaderNodeImplPtr SurfaceShaderNodeGlsl::create()
{
    return std::make_shared<SurfaceShaderNodeGlsl>();
}

const string& SurfaceShaderNodeGlsl::getTarget() const
{
    return GlslShaderGenerator::TARGET;
}

void SurfaceShaderNodeGlsl::createVariables(const ShaderNode&, GenContext& context, Shader& shader) const
{
    // TODO:
    // The surface shader needs position, view position and light sources. We should solve this by adding some
    // dependency mechanism so this implementation can be set to depend on the HwPositionNode,
    // HwViewDirectionNode and LightNodeGlsl nodes instead? This is where the MaterialX attribute "internalgeomprops"
    // is needed.
    //
    ShaderStage& vs = shader.getStage(Stage::VERTEX);
    ShaderStage& ps = shader.getStage(Stage::PIXEL);

    addStageInput(HW::VERTEX_INPUTS, Type::VECTOR3, HW::T_IN_POSITION, vs);
    addStageConnector(HW::VERTEX_DATA, Type::VECTOR3, HW::T_POSITION_WORLD, vs, ps);

    addStageUniform(HW::PRIVATE_UNIFORMS, Type::VECTOR3, HW::T_VIEW_POSITION, ps);

    const GlslShaderGenerator& shadergen = static_cast<const GlslShaderGenerator&>(context.getShaderGenerator());
    shadergen.addStageLightingUniforms(context, ps);
}

void SurfaceShaderNodeGlsl::emitFunctionCall(const ShaderNode& node, GenContext& context, ShaderStage& stage) const
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
            context.getShaderGenerator().emitLine(prefix + position->getVariable() + " = hPositionWorld.xyz", stage);
        }
    }

    DEFINE_SHADER_STAGE(stage, Stage::PIXEL)
    {
        SourceCodeNode::emitFunctionCall(node, context, stage);
    }
}

MATERIALX_NAMESPACE_END
