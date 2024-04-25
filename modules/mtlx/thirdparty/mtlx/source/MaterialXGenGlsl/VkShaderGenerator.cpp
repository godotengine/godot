//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXGenGlsl/VkShaderGenerator.h>
#include <MaterialXGenGlsl/VkSyntax.h>

MATERIALX_NAMESPACE_BEGIN

const string VkShaderGenerator::TARGET = "genglsl";
const string VkShaderGenerator::VERSION = "450";

VkShaderGenerator::VkShaderGenerator() :
    GlslShaderGenerator()
{
    _syntax = VkSyntax::create();
    // Add in Vulkan specific keywords
    const StringSet reservedWords = { "texture2D", "sampler" };
    _syntax->registerReservedWords(reservedWords);

    // Set binding context to handle resource binding layouts
    _resourceBindingCtx = std::make_shared<MaterialX::VkResourceBindingContext>(0);
}

void VkShaderGenerator::emitDirectives(GenContext&, ShaderStage& stage) const
{
    emitLine("#version " + getVersion(), stage, false);
    emitLineBreak(stage);
}

void VkShaderGenerator::emitInputs(GenContext& context, ShaderStage& stage) const
{
    DEFINE_SHADER_STAGE(stage, Stage::VERTEX)
    {
        const VariableBlock& vertexInputs = stage.getInputBlock(HW::VERTEX_INPUTS);
        if (!vertexInputs.empty())
        {
            emitComment("Inputs block: " + vertexInputs.getName(), stage);
            for (size_t i = 0; i < vertexInputs.size(); ++i)
            {
                emitLineBegin(stage);
                emitString("layout (location = " + std::to_string(i) + ") ", stage);
                emitVariableDeclaration(vertexInputs[i], _syntax->getInputQualifier(), context, stage, false);
                emitString(Syntax::SEMICOLON, stage);
                emitLineEnd(stage, false);
            }
            emitLineBreak(stage);
        }
    }

    DEFINE_SHADER_STAGE(stage, Stage::PIXEL)
    {
        const VariableBlock& vertexData = stage.getInputBlock(HW::VERTEX_DATA);
        if (!vertexData.empty())
        {
            emitComment("Inputs: " + vertexData.getName(), stage);
            for (size_t i = 0; i < vertexData.size(); ++i)
            {

                emitLineBegin(stage);
                emitString("layout (location = " + std::to_string(i) + ") ", stage);
                emitVariableDeclaration(vertexData[i], _syntax->getInputQualifier(), context, stage, false);
                emitString(Syntax::SEMICOLON, stage);
                emitLineEnd(stage, false);
            }
            emitLineBreak(stage);
        }
    }
}

string VkShaderGenerator::getVertexDataPrefix(const VariableBlock&) const
{
    return EMPTY_STRING;
}

void VkShaderGenerator::emitOutputs(GenContext& context, ShaderStage& stage) const
{
    DEFINE_SHADER_STAGE(stage, Stage::VERTEX)
    {
        const VariableBlock& vertexData = stage.getOutputBlock(HW::VERTEX_DATA);
        if (!vertexData.empty())
        {
            for (size_t i = 0; i < vertexData.size(); ++i)
            {
                emitLineBegin(stage);
                emitString("layout (location = " + std::to_string(i) + ") ", stage);
                emitVariableDeclaration(vertexData[i], _syntax->getOutputQualifier(), context, stage, false);
                emitString(Syntax::SEMICOLON, stage);
                emitLineEnd(stage, false);
            }
            emitLineBreak(stage);
        }
    }

    DEFINE_SHADER_STAGE(stage, Stage::PIXEL)
    {
        const VariableBlock& outputs = stage.getOutputBlock(HW::PIXEL_OUTPUTS);

        emitComment("Pixel shader outputs", stage);
        for (size_t i = 0; i < outputs.size(); ++i)
        {
            emitLineBegin(stage);
            emitString("layout (location = " + std::to_string(i) + ") ", stage);
            emitVariableDeclaration(outputs[i], _syntax->getOutputQualifier(), context, stage, false);
            emitString(Syntax::SEMICOLON, stage);
            emitLineEnd(stage, false);
        }
        emitLineBreak(stage);
    }
}

HwResourceBindingContextPtr VkShaderGenerator::getResourceBindingContext(GenContext& /*context*/) const
{
    return _resourceBindingCtx;
}

MATERIALX_NAMESPACE_END
