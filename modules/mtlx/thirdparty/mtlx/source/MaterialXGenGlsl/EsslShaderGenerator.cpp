//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXGenGlsl/EsslShaderGenerator.h>
#include <MaterialXGenGlsl/EsslSyntax.h>

#include <MaterialXGenShader/Nodes/HwImageNode.h>

MATERIALX_NAMESPACE_BEGIN

const string EsslShaderGenerator::TARGET = "essl";
const string EsslShaderGenerator::VERSION = "300 es"; // Current target is WebGL 2.0

EsslShaderGenerator::EsslShaderGenerator() :
    GlslShaderGenerator()
{
    _syntax = EsslSyntax::create();
    // Add in ESSL specific keywords
    const StringSet reservedWords = { "precision", "highp", "mediump", "lowp" };
    _syntax->registerReservedWords(reservedWords);
}

void EsslShaderGenerator::emitDirectives(GenContext&, ShaderStage& stage) const
{
    emitLine("#version " + getVersion(), stage, false);
    emitLineBreak(stage);
    emitLine("precision mediump float", stage);
}

void EsslShaderGenerator::emitUniforms(GenContext& context, ShaderStage& stage) const
{
    for (const auto& it : stage.getUniformBlocks())
    {
        const VariableBlock& uniforms = *it.second;

        // Skip light uniforms as they are handled separately
        if (!uniforms.empty() && uniforms.getName() != HW::LIGHT_DATA)
        {
            emitComment("Uniform block: " + uniforms.getName(), stage);
            emitVariableDeclarations(uniforms, _syntax->getUniformQualifier(), Syntax::SEMICOLON, context, stage, false);
            emitLineBreak(stage);
        }
    }
}

void EsslShaderGenerator::emitInputs(GenContext& context, ShaderStage& stage) const
{
    DEFINE_SHADER_STAGE(stage, Stage::VERTEX)
    {
        const VariableBlock& vertexInputs = stage.getInputBlock(HW::VERTEX_INPUTS);
        if (!vertexInputs.empty())
        {
            emitComment("Inputs block: " + vertexInputs.getName(), stage);
            emitVariableDeclarations(vertexInputs, _syntax->getInputQualifier(), Syntax::SEMICOLON, context, stage, false);
            emitLineBreak(stage);
        }
    }

    DEFINE_SHADER_STAGE(stage, Stage::PIXEL)
    {
        const VariableBlock& vertexData = stage.getInputBlock(HW::VERTEX_DATA);
        if (!vertexData.empty())
        {
            emitVariableDeclarations(vertexData, _syntax->getInputQualifier(), Syntax::SEMICOLON, context, stage, false);
            emitLineBreak(stage);
        }
    }
}

void EsslShaderGenerator::emitOutputs(GenContext& context, ShaderStage& stage) const
{
    DEFINE_SHADER_STAGE(stage, Stage::VERTEX)
    {
        const VariableBlock& vertexData = stage.getOutputBlock(HW::VERTEX_DATA);
        if (!vertexData.empty())
        {
            emitVariableDeclarations(vertexData, _syntax->getOutputQualifier(), Syntax::SEMICOLON, context, stage, false);
            emitLineBreak(stage);
        }
    }

    DEFINE_SHADER_STAGE(stage, Stage::PIXEL)
    {
        emitComment("Pixel shader outputs", stage);
        const VariableBlock& outputs = stage.getOutputBlock(HW::PIXEL_OUTPUTS);
        emitVariableDeclarations(outputs, _syntax->getOutputQualifier(), Syntax::SEMICOLON, context, stage, false);
        emitLineBreak(stage);
    }
}

string EsslShaderGenerator::getVertexDataPrefix(const VariableBlock&) const
{
    return EMPTY_STRING;
}

HwResourceBindingContextPtr EsslShaderGenerator::getResourceBindingContext(GenContext& context) const
{
    HwResourceBindingContextPtr resoureBindingCtx = GlslShaderGenerator::getResourceBindingContext(context);
    if (resoureBindingCtx)
    {
        throw ExceptionShaderGenError("The EsslShaderGenerator does not support resource binding.");
    }
    return nullptr;
}

MATERIALX_NAMESPACE_END
