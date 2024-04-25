//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXGenGlsl/GlslShaderGenerator.h>

#include <MaterialXGenGlsl/GlslSyntax.h>
#include <MaterialXGenGlsl/Nodes/GeomColorNodeGlsl.h>
#include <MaterialXGenGlsl/Nodes/GeomPropValueNodeGlsl.h>
#include <MaterialXGenGlsl/Nodes/SurfaceNodeGlsl.h>
#include <MaterialXGenGlsl/Nodes/UnlitSurfaceNodeGlsl.h>
#include <MaterialXGenGlsl/Nodes/LightNodeGlsl.h>
#include <MaterialXGenGlsl/Nodes/LightCompoundNodeGlsl.h>
#include <MaterialXGenGlsl/Nodes/LightShaderNodeGlsl.h>
#include <MaterialXGenGlsl/Nodes/HeightToNormalNodeGlsl.h>
#include <MaterialXGenGlsl/Nodes/LightSamplerNodeGlsl.h>
#include <MaterialXGenGlsl/Nodes/NumLightsNodeGlsl.h>
#include <MaterialXGenGlsl/Nodes/BlurNodeGlsl.h>

#include <MaterialXGenShader/Nodes/MaterialNode.h>
#include <MaterialXGenShader/Nodes/SwizzleNode.h>
#include <MaterialXGenShader/Nodes/ConvertNode.h>
#include <MaterialXGenShader/Nodes/CombineNode.h>
#include <MaterialXGenShader/Nodes/SwitchNode.h>
#include <MaterialXGenShader/Nodes/HwImageNode.h>
#include <MaterialXGenShader/Nodes/HwTexCoordNode.h>
#include <MaterialXGenShader/Nodes/HwTransformNode.h>
#include <MaterialXGenShader/Nodes/HwPositionNode.h>
#include <MaterialXGenShader/Nodes/HwNormalNode.h>
#include <MaterialXGenShader/Nodes/HwTangentNode.h>
#include <MaterialXGenShader/Nodes/HwBitangentNode.h>
#include <MaterialXGenShader/Nodes/HwFrameNode.h>
#include <MaterialXGenShader/Nodes/HwTimeNode.h>
#include <MaterialXGenShader/Nodes/HwViewDirectionNode.h>
#include <MaterialXGenShader/Nodes/ClosureSourceCodeNode.h>
#include <MaterialXGenShader/Nodes/ClosureCompoundNode.h>
#include <MaterialXGenShader/Nodes/ClosureLayerNode.h>
#include <MaterialXGenShader/Nodes/ClosureMixNode.h>
#include <MaterialXGenShader/Nodes/ClosureAddNode.h>
#include <MaterialXGenShader/Nodes/ClosureMultiplyNode.h>

MATERIALX_NAMESPACE_BEGIN

const string GlslShaderGenerator::TARGET = "genglsl";
const string GlslShaderGenerator::VERSION = "400";

//
// GlslShaderGenerator methods
//

GlslShaderGenerator::GlslShaderGenerator() :
    HwShaderGenerator(GlslSyntax::create())
{
    //
    // Register all custom node implementation classes
    //

    StringVec elementNames;

    // <!-- <switch> -->
    elementNames = {
        // <!-- 'which' type : float -->
        "IM_switch_float_" + GlslShaderGenerator::TARGET,
        "IM_switch_color3_" + GlslShaderGenerator::TARGET,
        "IM_switch_color4_" + GlslShaderGenerator::TARGET,
        "IM_switch_vector2_" + GlslShaderGenerator::TARGET,
        "IM_switch_vector3_" + GlslShaderGenerator::TARGET,
        "IM_switch_vector4_" + GlslShaderGenerator::TARGET,

        // <!-- 'which' type : integer -->
        "IM_switch_floatI_" + GlslShaderGenerator::TARGET,
        "IM_switch_color3I_" + GlslShaderGenerator::TARGET,
        "IM_switch_color4I_" + GlslShaderGenerator::TARGET,
        "IM_switch_vector2I_" + GlslShaderGenerator::TARGET,
        "IM_switch_vector3I_" + GlslShaderGenerator::TARGET,
        "IM_switch_vector4I_" + GlslShaderGenerator::TARGET,
    };
    registerImplementation(elementNames, SwitchNode::create);

    // <!-- <swizzle> -->
    elementNames = {
        // <!-- from type : float -->
        "IM_swizzle_float_color3_" + GlslShaderGenerator::TARGET,
        "IM_swizzle_float_color4_" + GlslShaderGenerator::TARGET,
        "IM_swizzle_float_vector2_" + GlslShaderGenerator::TARGET,
        "IM_swizzle_float_vector3_" + GlslShaderGenerator::TARGET,
        "IM_swizzle_float_vector4_" + GlslShaderGenerator::TARGET,

        // <!-- from type : color3 -->
        "IM_swizzle_color3_float_" + GlslShaderGenerator::TARGET,
        "IM_swizzle_color3_color3_" + GlslShaderGenerator::TARGET,
        "IM_swizzle_color3_color4_" + GlslShaderGenerator::TARGET,
        "IM_swizzle_color3_vector2_" + GlslShaderGenerator::TARGET,
        "IM_swizzle_color3_vector3_" + GlslShaderGenerator::TARGET,
        "IM_swizzle_color3_vector4_" + GlslShaderGenerator::TARGET,

        // <!-- from type : color4 -->
        "IM_swizzle_color4_float_" + GlslShaderGenerator::TARGET,
        "IM_swizzle_color4_color3_" + GlslShaderGenerator::TARGET,
        "IM_swizzle_color4_color4_" + GlslShaderGenerator::TARGET,
        "IM_swizzle_color4_vector2_" + GlslShaderGenerator::TARGET,
        "IM_swizzle_color4_vector3_" + GlslShaderGenerator::TARGET,
        "IM_swizzle_color4_vector4_" + GlslShaderGenerator::TARGET,

        // <!-- from type : vector2 -->
        "IM_swizzle_vector2_float_" + GlslShaderGenerator::TARGET,
        "IM_swizzle_vector2_color3_" + GlslShaderGenerator::TARGET,
        "IM_swizzle_vector2_color4_" + GlslShaderGenerator::TARGET,
        "IM_swizzle_vector2_vector2_" + GlslShaderGenerator::TARGET,
        "IM_swizzle_vector2_vector3_" + GlslShaderGenerator::TARGET,
        "IM_swizzle_vector2_vector4_" + GlslShaderGenerator::TARGET,

        // <!-- from type : vector3 -->
        "IM_swizzle_vector3_float_" + GlslShaderGenerator::TARGET,
        "IM_swizzle_vector3_color3_" + GlslShaderGenerator::TARGET,
        "IM_swizzle_vector3_color4_" + GlslShaderGenerator::TARGET,
        "IM_swizzle_vector3_vector2_" + GlslShaderGenerator::TARGET,
        "IM_swizzle_vector3_vector3_" + GlslShaderGenerator::TARGET,
        "IM_swizzle_vector3_vector4_" + GlslShaderGenerator::TARGET,

        // <!-- from type : vector4 -->
        "IM_swizzle_vector4_float_" + GlslShaderGenerator::TARGET,
        "IM_swizzle_vector4_color3_" + GlslShaderGenerator::TARGET,
        "IM_swizzle_vector4_color4_" + GlslShaderGenerator::TARGET,
        "IM_swizzle_vector4_vector2_" + GlslShaderGenerator::TARGET,
        "IM_swizzle_vector4_vector3_" + GlslShaderGenerator::TARGET,
        "IM_swizzle_vector4_vector4_" + GlslShaderGenerator::TARGET,
    };
    registerImplementation(elementNames, SwizzleNode::create);

    // <!-- <convert> -->
    elementNames = {
        "IM_convert_float_color3_" + GlslShaderGenerator::TARGET,
        "IM_convert_float_color4_" + GlslShaderGenerator::TARGET,
        "IM_convert_float_vector2_" + GlslShaderGenerator::TARGET,
        "IM_convert_float_vector3_" + GlslShaderGenerator::TARGET,
        "IM_convert_float_vector4_" + GlslShaderGenerator::TARGET,
        "IM_convert_vector2_vector3_" + GlslShaderGenerator::TARGET,
        "IM_convert_vector3_vector2_" + GlslShaderGenerator::TARGET,
        "IM_convert_vector3_color3_" + GlslShaderGenerator::TARGET,
        "IM_convert_vector3_vector4_" + GlslShaderGenerator::TARGET,
        "IM_convert_vector4_vector3_" + GlslShaderGenerator::TARGET,
        "IM_convert_vector4_color4_" + GlslShaderGenerator::TARGET,
        "IM_convert_color3_vector3_" + GlslShaderGenerator::TARGET,
        "IM_convert_color4_vector4_" + GlslShaderGenerator::TARGET,
        "IM_convert_color3_color4_" + GlslShaderGenerator::TARGET,
        "IM_convert_color4_color3_" + GlslShaderGenerator::TARGET,
        "IM_convert_boolean_float_" + GlslShaderGenerator::TARGET,
        "IM_convert_integer_float_" + GlslShaderGenerator::TARGET,
    };
    registerImplementation(elementNames, ConvertNode::create);

    // <!-- <combine> -->
    elementNames = {
        "IM_combine2_vector2_" + GlslShaderGenerator::TARGET,
        "IM_combine2_color4CF_" + GlslShaderGenerator::TARGET,
        "IM_combine2_vector4VF_" + GlslShaderGenerator::TARGET,
        "IM_combine2_vector4VV_" + GlslShaderGenerator::TARGET,
        "IM_combine3_color3_" + GlslShaderGenerator::TARGET,
        "IM_combine3_vector3_" + GlslShaderGenerator::TARGET,
        "IM_combine4_color4_" + GlslShaderGenerator::TARGET,
        "IM_combine4_vector4_" + GlslShaderGenerator::TARGET,
    };
    registerImplementation(elementNames, CombineNode::create);

    // <!-- <position> -->
    registerImplementation("IM_position_vector3_" + GlslShaderGenerator::TARGET, HwPositionNode::create);
    // <!-- <normal> -->
    registerImplementation("IM_normal_vector3_" + GlslShaderGenerator::TARGET, HwNormalNode::create);
    // <!-- <tangent> -->
    registerImplementation("IM_tangent_vector3_" + GlslShaderGenerator::TARGET, HwTangentNode::create);
    // <!-- <bitangent> -->
    registerImplementation("IM_bitangent_vector3_" + GlslShaderGenerator::TARGET, HwBitangentNode::create);
    // <!-- <texcoord> -->
    registerImplementation("IM_texcoord_vector2_" + GlslShaderGenerator::TARGET, HwTexCoordNode::create);
    registerImplementation("IM_texcoord_vector3_" + GlslShaderGenerator::TARGET, HwTexCoordNode::create);
    // <!-- <geomcolor> -->
    registerImplementation("IM_geomcolor_float_" + GlslShaderGenerator::TARGET, GeomColorNodeGlsl::create);
    registerImplementation("IM_geomcolor_color3_" + GlslShaderGenerator::TARGET, GeomColorNodeGlsl::create);
    registerImplementation("IM_geomcolor_color4_" + GlslShaderGenerator::TARGET, GeomColorNodeGlsl::create);
    // <!-- <geompropvalue> -->
    elementNames = {
        "IM_geompropvalue_integer_" + GlslShaderGenerator::TARGET,
        "IM_geompropvalue_float_" + GlslShaderGenerator::TARGET,
        "IM_geompropvalue_color3_" + GlslShaderGenerator::TARGET,
        "IM_geompropvalue_color4_" + GlslShaderGenerator::TARGET,
        "IM_geompropvalue_vector2_" + GlslShaderGenerator::TARGET,
        "IM_geompropvalue_vector3_" + GlslShaderGenerator::TARGET,
        "IM_geompropvalue_vector4_" + GlslShaderGenerator::TARGET,
    };
    registerImplementation(elementNames, GeomPropValueNodeGlsl::create);
    registerImplementation("IM_geompropvalue_boolean_" + GlslShaderGenerator::TARGET, GeomPropValueNodeGlslAsUniform::create);
    registerImplementation("IM_geompropvalue_string_" + GlslShaderGenerator::TARGET, GeomPropValueNodeGlslAsUniform::create);

    // <!-- <frame> -->
    registerImplementation("IM_frame_float_" + GlslShaderGenerator::TARGET, HwFrameNode::create);
    // <!-- <time> -->
    registerImplementation("IM_time_float_" + GlslShaderGenerator::TARGET, HwTimeNode::create);
    // <!-- <viewdirection> -->
    registerImplementation("IM_viewdirection_vector3_" + GlslShaderGenerator::TARGET, HwViewDirectionNode::create);

    // <!-- <surface> -->
    registerImplementation("IM_surface_" + GlslShaderGenerator::TARGET, SurfaceNodeGlsl::create);
    registerImplementation("IM_surface_unlit_" + GlslShaderGenerator::TARGET, UnlitSurfaceNodeGlsl::create);

    // <!-- <light> -->
    registerImplementation("IM_light_" + GlslShaderGenerator::TARGET, LightNodeGlsl::create);

    // <!-- <point_light> -->
    registerImplementation("IM_point_light_" + GlslShaderGenerator::TARGET, LightShaderNodeGlsl::create);
    // <!-- <directional_light> -->
    registerImplementation("IM_directional_light_" + GlslShaderGenerator::TARGET, LightShaderNodeGlsl::create);
    // <!-- <spot_light> -->
    registerImplementation("IM_spot_light_" + GlslShaderGenerator::TARGET, LightShaderNodeGlsl::create);

    // <!-- <heighttonormal> -->
    registerImplementation("IM_heighttonormal_vector3_" + GlslShaderGenerator::TARGET, HeightToNormalNodeGlsl::create);

    // <!-- <blur> -->
    elementNames = {
        "IM_blur_float_" + GlslShaderGenerator::TARGET,
        "IM_blur_color3_" + GlslShaderGenerator::TARGET,
        "IM_blur_color4_" + GlslShaderGenerator::TARGET,
        "IM_blur_vector2_" + GlslShaderGenerator::TARGET,
        "IM_blur_vector3_" + GlslShaderGenerator::TARGET,
        "IM_blur_vector4_" + GlslShaderGenerator::TARGET,
    };
    registerImplementation(elementNames, BlurNodeGlsl::create);

    // <!-- <ND_transformpoint> ->
    registerImplementation("IM_transformpoint_vector3_" + GlslShaderGenerator::TARGET, HwTransformPointNode::create);

    // <!-- <ND_transformvector> ->
    registerImplementation("IM_transformvector_vector3_" + GlslShaderGenerator::TARGET, HwTransformVectorNode::create);

    // <!-- <ND_transformnormal> ->
    registerImplementation("IM_transformnormal_vector3_" + GlslShaderGenerator::TARGET, HwTransformNormalNode::create);

    // <!-- <image> -->
    elementNames = {
        "IM_image_float_" + GlslShaderGenerator::TARGET,
        "IM_image_color3_" + GlslShaderGenerator::TARGET,
        "IM_image_color4_" + GlslShaderGenerator::TARGET,
        "IM_image_vector2_" + GlslShaderGenerator::TARGET,
        "IM_image_vector3_" + GlslShaderGenerator::TARGET,
        "IM_image_vector4_" + GlslShaderGenerator::TARGET,
    };
    registerImplementation(elementNames, HwImageNode::create);

    // <!-- <layer> -->
    registerImplementation("IM_layer_bsdf_" + GlslShaderGenerator::TARGET, ClosureLayerNode::create);
    registerImplementation("IM_layer_vdf_" + GlslShaderGenerator::TARGET, ClosureLayerNode::create);
    // <!-- <mix> -->
    registerImplementation("IM_mix_bsdf_" + GlslShaderGenerator::TARGET, ClosureMixNode::create);
    registerImplementation("IM_mix_edf_" + GlslShaderGenerator::TARGET, ClosureMixNode::create);
    // <!-- <add> -->
    registerImplementation("IM_add_bsdf_" + GlslShaderGenerator::TARGET, ClosureAddNode::create);
    registerImplementation("IM_add_edf_" + GlslShaderGenerator::TARGET, ClosureAddNode::create);
    // <!-- <multiply> -->
    elementNames = {
        "IM_multiply_bsdfC_" + GlslShaderGenerator::TARGET,
        "IM_multiply_bsdfF_" + GlslShaderGenerator::TARGET,
        "IM_multiply_edfC_" + GlslShaderGenerator::TARGET,
        "IM_multiply_edfF_" + GlslShaderGenerator::TARGET,
    };
    registerImplementation(elementNames, ClosureMultiplyNode::create);

    // <!-- <thin_film> -->
    registerImplementation("IM_thin_film_bsdf_" + GlslShaderGenerator::TARGET, NopNode::create);

    // <!-- <surfacematerial> -->
    registerImplementation("IM_surfacematerial_" + GlslShaderGenerator::TARGET, MaterialNode::create);

    _lightSamplingNodes.push_back(ShaderNode::create(nullptr, "numActiveLightSources", NumLightsNodeGlsl::create()));
    _lightSamplingNodes.push_back(ShaderNode::create(nullptr, "sampleLightSource", LightSamplerNodeGlsl::create()));
}

ShaderPtr GlslShaderGenerator::generate(const string& name, ElementPtr element, GenContext& context) const
{
    ShaderPtr shader = createShader(name, element, context);

    // Request fixed floating-point notation for consistency across targets.
    ScopedFloatFormatting fmt(Value::FloatFormatFixed);

    // Make sure we initialize/reset the binding context before generation.
    HwResourceBindingContextPtr resourceBindingCtx = getResourceBindingContext(context);
    if (resourceBindingCtx)
    {
        resourceBindingCtx->initialize();
    }

    // Emit code for vertex shader stage
    ShaderStage& vs = shader->getStage(Stage::VERTEX);
    emitVertexStage(shader->getGraph(), context, vs);
    replaceTokens(_tokenSubstitutions, vs);

    // Emit code for pixel shader stage
    ShaderStage& ps = shader->getStage(Stage::PIXEL);
    emitPixelStage(shader->getGraph(), context, ps);
    replaceTokens(_tokenSubstitutions, ps);

    return shader;
}

void GlslShaderGenerator::emitVertexStage(const ShaderGraph& graph, GenContext& context, ShaderStage& stage) const
{
    HwResourceBindingContextPtr resourceBindingCtx = getResourceBindingContext(context);

    emitDirectives(context, stage);
    if (resourceBindingCtx)
    {
        resourceBindingCtx->emitDirectives(context, stage);
    }
    emitLineBreak(stage);

    // Add all constants
    emitConstants(context, stage);

    // Add all uniforms
    emitUniforms(context, stage);

    // Add vertex inputs
    emitInputs(context, stage);

    // Add vertex data outputs block
    emitOutputs(context, stage);

    emitFunctionDefinitions(graph, context, stage);

    // Add main function
    setFunctionName("main", stage);
    emitLine("void main()", stage, false);
    emitFunctionBodyBegin(graph, context, stage);
    emitLine("vec4 hPositionWorld = " + HW::T_WORLD_MATRIX + " * vec4(" + HW::T_IN_POSITION + ", 1.0)", stage);
    emitLine("gl_Position = " + HW::T_VIEW_PROJECTION_MATRIX + " * hPositionWorld", stage);

    // Emit all function calls in order
    for (const ShaderNode* node : graph.getNodes())
    {
        emitFunctionCall(*node, context, stage);
    }

    emitFunctionBodyEnd(graph, context, stage);
}

void GlslShaderGenerator::emitSpecularEnvironment(GenContext& context, ShaderStage& stage) const
{
    int specularMethod = context.getOptions().hwSpecularEnvironmentMethod;
    if (specularMethod == SPECULAR_ENVIRONMENT_FIS)
    {
        emitLibraryInclude("pbrlib/genglsl/lib/mx_environment_fis.glsl", context, stage);
    }
    else if (specularMethod == SPECULAR_ENVIRONMENT_PREFILTER)
    {
        emitLibraryInclude("pbrlib/genglsl/lib/mx_environment_prefilter.glsl", context, stage);
    }
    else if (specularMethod == SPECULAR_ENVIRONMENT_NONE)
    {
        emitLibraryInclude("pbrlib/genglsl/lib/mx_environment_none.glsl", context, stage);
    }
    else
    {
        throw ExceptionShaderGenError("Invalid hardware specular environment method specified: '" + std::to_string(specularMethod) + "'");
    }
    emitLineBreak(stage);
}

void GlslShaderGenerator::emitTransmissionRender(GenContext& context, ShaderStage& stage) const
{
    int transmissionMethod = context.getOptions().hwTransmissionRenderMethod;
    if (transmissionMethod == TRANSMISSION_REFRACTION)
    {
        emitLibraryInclude("pbrlib/genglsl/lib/mx_transmission_refract.glsl", context, stage);
    }
    else if (transmissionMethod == TRANSMISSION_OPACITY)
    {
        emitLibraryInclude("pbrlib/genglsl/lib/mx_transmission_opacity.glsl", context, stage);
    }
    else
    {
        throw ExceptionShaderGenError("Invalid transmission render specified: '" + std::to_string(transmissionMethod) + "'");
    }
    emitLineBreak(stage);
}

void GlslShaderGenerator::emitDirectives(GenContext&, ShaderStage& stage) const
{
    emitLine("#version " + getVersion(), stage, false);
}

void GlslShaderGenerator::emitConstants(GenContext& context, ShaderStage& stage) const
{
    const VariableBlock& constants = stage.getConstantBlock();
    if (!constants.empty())
    {
        emitVariableDeclarations(constants, _syntax->getConstantQualifier(), Syntax::SEMICOLON, context, stage);
        emitLineBreak(stage);
    }
}

void GlslShaderGenerator::emitUniforms(GenContext& context, ShaderStage& stage) const
{
    for (const auto& it : stage.getUniformBlocks())
    {
        const VariableBlock& uniforms = *it.second;

        // Skip light uniforms as they are handled separately
        if (!uniforms.empty() && uniforms.getName() != HW::LIGHT_DATA)
        {
            emitComment("Uniform block: " + uniforms.getName(), stage);
            HwResourceBindingContextPtr resourceBindingCtx = getResourceBindingContext(context);
            if (resourceBindingCtx)
            {
                resourceBindingCtx->emitResourceBindings(context, uniforms, stage);
            }
            else
            {
                emitVariableDeclarations(uniforms, _syntax->getUniformQualifier(), Syntax::SEMICOLON, context, stage);
                emitLineBreak(stage);
            }
        }
    }
}

void GlslShaderGenerator::emitLightData(GenContext& context, ShaderStage& stage) const
{
    const VariableBlock& lightData = stage.getUniformBlock(HW::LIGHT_DATA);
    const string structArraySuffix = "[" + HW::LIGHT_DATA_MAX_LIGHT_SOURCES + "]";
    const string structName = lightData.getInstance();
    HwResourceBindingContextPtr resourceBindingCtx = getResourceBindingContext(context);
    if (resourceBindingCtx)
    {
        resourceBindingCtx->emitStructuredResourceBindings(
            context, lightData, stage, structName, structArraySuffix);
    }
    else
    {
        emitLine("struct " + lightData.getName(), stage, false);
        emitScopeBegin(stage);
        emitVariableDeclarations(lightData, EMPTY_STRING, Syntax::SEMICOLON, context, stage, false);
        emitScopeEnd(stage, true);
        emitLineBreak(stage);
        emitLine("uniform " + lightData.getName() + " " + structName + structArraySuffix, stage);
    }
    emitLineBreak(stage);
}

void GlslShaderGenerator::emitInputs(GenContext& context, ShaderStage& stage) const
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
            emitLine("in " + vertexData.getName(), stage, false);
            emitScopeBegin(stage);
            emitVariableDeclarations(vertexData, EMPTY_STRING, Syntax::SEMICOLON, context, stage, false);
            emitScopeEnd(stage, false, false);
            emitString(" " + vertexData.getInstance() + Syntax::SEMICOLON, stage);
            emitLineBreak(stage);
            emitLineBreak(stage);
        }
    }
}

void GlslShaderGenerator::emitOutputs(GenContext& context, ShaderStage& stage) const
{
    DEFINE_SHADER_STAGE(stage, Stage::VERTEX)
    {
        const VariableBlock& vertexData = stage.getOutputBlock(HW::VERTEX_DATA);
        if (!vertexData.empty())
        {
            emitLine("out " + vertexData.getName(), stage, false);
            emitScopeBegin(stage);
            emitVariableDeclarations(vertexData, EMPTY_STRING, Syntax::SEMICOLON, context, stage, false);
            emitScopeEnd(stage, false, false);
            emitString(" " + vertexData.getInstance() + Syntax::SEMICOLON, stage);
            emitLineBreak(stage);
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

HwResourceBindingContextPtr GlslShaderGenerator::getResourceBindingContext(GenContext& context) const
{
    return context.getUserData<HwResourceBindingContext>(HW::USER_DATA_BINDING_CONTEXT);
}

string GlslShaderGenerator::getVertexDataPrefix(const VariableBlock& vertexData) const
{
    return vertexData.getInstance() + ".";
}

bool GlslShaderGenerator::requiresLighting(const ShaderGraph& graph) const
{
    const bool isBsdf = graph.hasClassification(ShaderNode::Classification::BSDF);
    const bool isLitSurfaceShader = graph.hasClassification(ShaderNode::Classification::SHADER) &&
                                    graph.hasClassification(ShaderNode::Classification::SURFACE) &&
                                    !graph.hasClassification(ShaderNode::Classification::UNLIT);
    return isBsdf || isLitSurfaceShader;
}

void GlslShaderGenerator::emitPixelStage(const ShaderGraph& graph, GenContext& context, ShaderStage& stage) const
{
    HwResourceBindingContextPtr resourceBindingCtx = getResourceBindingContext(context);

    // Add directives
    emitDirectives(context, stage);
    if (resourceBindingCtx)
    {
        resourceBindingCtx->emitDirectives(context, stage);
    }
    emitLineBreak(stage);

    // Add type definitions
    emitTypeDefinitions(context, stage);

    // Add all constants
    emitConstants(context, stage);

    // Add all uniforms
    emitUniforms(context, stage);

    // Add vertex data inputs block
    emitInputs(context, stage);

    // Add the pixel shader output. This needs to be a vec4 for rendering
    // and upstream connection will be converted to vec4 if needed in emitFinalOutput()
    emitOutputs(context, stage);

    // Add common math functions
    emitLibraryInclude("stdlib/genglsl/lib/mx_math.glsl", context, stage);
    emitLineBreak(stage);

    // Determine whether lighting is required
    bool lighting = requiresLighting(graph);

    // Define directional albedo approach
    if (lighting || context.getOptions().hwWriteAlbedoTable || context.getOptions().hwWriteEnvPrefilter)
    {
        emitLine("#define DIRECTIONAL_ALBEDO_METHOD " + std::to_string(int(context.getOptions().hwDirectionalAlbedoMethod)), stage, false);
        emitLineBreak(stage);
    }

    // Add lighting support
    if (lighting)
    {
        if (context.getOptions().hwMaxActiveLightSources > 0)
        {
            const unsigned int maxLights = std::max(1u, context.getOptions().hwMaxActiveLightSources);
            emitLine("#define " + HW::LIGHT_DATA_MAX_LIGHT_SOURCES + " " + std::to_string(maxLights), stage, false);
        }
        emitSpecularEnvironment(context, stage);
        emitTransmissionRender(context, stage);

        if (context.getOptions().hwMaxActiveLightSources > 0)
        {
            emitLightData(context, stage);
        }
    }

    // Add shadowing support
    bool shadowing = (lighting && context.getOptions().hwShadowMap) ||
                     context.getOptions().hwWriteDepthMoments;
    if (shadowing)
    {
        emitLibraryInclude("pbrlib/genglsl/lib/mx_shadow.glsl", context, stage);
    }

    // Emit directional albedo table code.
    if (context.getOptions().hwWriteAlbedoTable)
    {
        emitLibraryInclude("pbrlib/genglsl/lib/mx_generate_albedo_table.glsl", context, stage);
        emitLineBreak(stage);
    }

    // Emit environment prefiltering code
    if (context.getOptions().hwWriteEnvPrefilter)
    {
        emitLibraryInclude("pbrlib/genglsl/lib/mx_generate_prefilter_env.glsl", context, stage);
        emitLineBreak(stage);
    }

    // Set the include file to use for uv transformations,
    // depending on the vertical flip flag.
    if (context.getOptions().fileTextureVerticalFlip)
    {
        _tokenSubstitutions[ShaderGenerator::T_FILE_TRANSFORM_UV] = "mx_transform_uv_vflip.glsl";
    }
    else
    {
        _tokenSubstitutions[ShaderGenerator::T_FILE_TRANSFORM_UV] = "mx_transform_uv.glsl";
    }

    // Emit uv transform code globally if needed.
    if (context.getOptions().hwAmbientOcclusion)
    {
        emitLibraryInclude("stdlib/genglsl/lib/" + _tokenSubstitutions[ShaderGenerator::T_FILE_TRANSFORM_UV], context, stage);
    }

    emitLightFunctionDefinitions(graph, context, stage);

    // Emit function definitions for all nodes in the graph.
    emitFunctionDefinitions(graph, context, stage);

    const ShaderGraphOutputSocket* outputSocket = graph.getOutputSocket();

    // Add main function
    setFunctionName("main", stage);
    emitLine("void main()", stage, false);
    emitFunctionBodyBegin(graph, context, stage);

    if (graph.hasClassification(ShaderNode::Classification::CLOSURE) &&
        !graph.hasClassification(ShaderNode::Classification::SHADER))
    {
        // Handle the case where the graph is a direct closure.
        // We don't support rendering closures without attaching
        // to a surface shader, so just output black.
        emitLine(outputSocket->getVariable() + " = vec4(0.0, 0.0, 0.0, 1.0)", stage);
    }
    else if (context.getOptions().hwWriteDepthMoments)
    {
        emitLine(outputSocket->getVariable() + " = vec4(mx_compute_depth_moments(), 0.0, 1.0)", stage);
    }
    else if (context.getOptions().hwWriteAlbedoTable)
    {
        emitLine(outputSocket->getVariable() + " = vec4(mx_generate_dir_albedo_table(), 1.0)", stage);
    }
    else if (context.getOptions().hwWriteEnvPrefilter)
    {
        emitLine(outputSocket->getVariable() + " = vec4(mx_generate_prefilter_env(), 1.0)", stage);
    }
    else
    {
        // Add all function calls.
        //
        // Surface shaders need special handling.
        if (graph.hasClassification(ShaderNode::Classification::SHADER | ShaderNode::Classification::SURFACE))
        {
            // Emit all texturing nodes. These are inputs to any
            // closure/shader nodes and need to be emitted first.
            emitFunctionCalls(graph, context, stage, ShaderNode::Classification::TEXTURE);

            // Emit function calls for "root" closure/shader nodes.
            // These will internally emit function calls for any dependent closure nodes upstream.
            for (ShaderGraphOutputSocket* socket : graph.getOutputSockets())
            {
                if (socket->getConnection())
                {
                    const ShaderNode* upstream = socket->getConnection()->getNode();
                    if (upstream->getParent() == &graph &&
                        (upstream->hasClassification(ShaderNode::Classification::CLOSURE) ||
                         upstream->hasClassification(ShaderNode::Classification::SHADER)))
                    {
                        emitFunctionCall(*upstream, context, stage);
                    }
                }
            }
        }
        else
        {
            // No surface shader graph so just generate all
            // function calls in order.
            emitFunctionCalls(graph, context, stage);
        }

        // Emit final output
        const ShaderOutput* outputConnection = outputSocket->getConnection();
        if (outputConnection)
        {
            string finalOutput = outputConnection->getVariable();
            const string& channels = outputSocket->getChannels();
            if (!channels.empty())
            {
                finalOutput = _syntax->getSwizzledVariable(finalOutput, outputConnection->getType(), channels, outputSocket->getType());
            }

            if (graph.hasClassification(ShaderNode::Classification::SURFACE))
            {
                if (context.getOptions().hwTransparency)
                {
                    emitLine("float outAlpha = clamp(1.0 - dot(" + finalOutput + ".transparency, vec3(0.3333)), 0.0, 1.0)", stage);
                    emitLine(outputSocket->getVariable() + " = vec4(" + finalOutput + ".color, outAlpha)", stage);
                    emitLine("if (outAlpha < " + HW::T_ALPHA_THRESHOLD + ")", stage, false);
                    emitScopeBegin(stage);
                    emitLine("discard", stage);
                    emitScopeEnd(stage);
                }
                else
                {
                    emitLine(outputSocket->getVariable() + " = vec4(" + finalOutput + ".color, 1.0)", stage);
                }
            }
            else
            {
                if (!outputSocket->getType()->isFloat4())
                {
                    toVec4(outputSocket->getType(), finalOutput);
                }
                emitLine(outputSocket->getVariable() + " = " + finalOutput, stage);
            }
        }
        else
        {
            string outputValue = outputSocket->getValue() ?
                                 _syntax->getValue(outputSocket->getType(), *outputSocket->getValue()) :
                                 _syntax->getDefaultValue(outputSocket->getType());
            if (!outputSocket->getType()->isFloat4())
            {
                string finalOutput = outputSocket->getVariable() + "_tmp";
                emitLine(_syntax->getTypeName(outputSocket->getType()) + " " + finalOutput + " = " + outputValue, stage);
                toVec4(outputSocket->getType(), finalOutput);
                emitLine(outputSocket->getVariable() + " = " + finalOutput, stage);
            }
            else
            {
                emitLine(outputSocket->getVariable() + " = " + outputValue, stage);
            }
        }
    }

    // End main function
    emitFunctionBodyEnd(graph, context, stage);
}

void GlslShaderGenerator::emitLightFunctionDefinitions(const ShaderGraph& graph, GenContext& context, ShaderStage& stage) const
{
    DEFINE_SHADER_STAGE(stage, Stage::PIXEL)
    {
        // Emit Light functions if requested
        if (requiresLighting(graph) && context.getOptions().hwMaxActiveLightSources > 0)
        {
            // For surface shaders we need light shaders
            if (graph.hasClassification(ShaderNode::Classification::SHADER | ShaderNode::Classification::SURFACE))
            {
                // Emit functions for all bound light shaders
                HwLightShadersPtr lightShaders = context.getUserData<HwLightShaders>(HW::USER_DATA_LIGHT_SHADERS);
                if (lightShaders)
                {
                    for (const auto& it : lightShaders->get())
                    {
                        emitFunctionDefinition(*it.second, context, stage);
                    }
                }
                // Emit functions for light sampling
                for (const auto& it : _lightSamplingNodes)
                {
                    emitFunctionDefinition(*it, context, stage);
                }
            }
        }
    }
}

void GlslShaderGenerator::toVec4(const TypeDesc* type, string& variable)
{
    if (type->isFloat3())
    {
        variable = "vec4(" + variable + ", 1.0)";
    }
    else if (type->isFloat2())
    {
        variable = "vec4(" + variable + ", 0.0, 1.0)";
    }
    else if (*type == *Type::FLOAT || *type == *Type::INTEGER)
    {
        variable = "vec4(" + variable + ", " + variable + ", " + variable + ", 1.0)";
    }
    else if (*type == *Type::BSDF || *type == *Type::EDF)
    {
        variable = "vec4(" + variable + ", 1.0)";
    }
    else
    {
        // Can't understand other types. Just return black.
        variable = "vec4(0.0, 0.0, 0.0, 1.0)";
    }
}

void GlslShaderGenerator::emitVariableDeclaration(const ShaderPort* variable, const string& qualifier,
                                                  GenContext&, ShaderStage& stage,
                                                  bool assignValue) const
{
    // A file texture input needs special handling on GLSL
    if (*variable->getType() == *Type::FILENAME)
    {
        // Samplers must always be uniforms
        string str = qualifier.empty() ? EMPTY_STRING : qualifier + " ";
        emitString(str + "sampler2D " + variable->getVariable(), stage);
    }
    else
    {
        string str = qualifier.empty() ? EMPTY_STRING : qualifier + " ";
        // Varying parameters of type int must be flat qualified on output from vertex stage and
        // input to pixel stage. The only way to get these is with geompropvalue_integer nodes.
        if (qualifier.empty() && *variable->getType() == *Type::INTEGER && !assignValue && variable->getName().rfind(HW::T_IN_GEOMPROP, 0) == 0)
        {
            str += GlslSyntax::FLAT_QUALIFIER + " ";
        }
        str += _syntax->getTypeName(variable->getType()) + " " + variable->getVariable();

        // If an array we need an array qualifier (suffix) for the variable name
        if (variable->getType()->isArray() && variable->getValue())
        {
            str += _syntax->getArrayVariableSuffix(variable->getType(), *variable->getValue());
        }

        if (!variable->getSemantic().empty())
        {
            str += " : " + variable->getSemantic();
        }

        if (assignValue)
        {
            const string valueStr = (variable->getValue() ?
                                    _syntax->getValue(variable->getType(), *variable->getValue(), true) :
                                    _syntax->getDefaultValue(variable->getType(), true));
            str += valueStr.empty() ? EMPTY_STRING : " = " + valueStr;
        }

        emitString(str, stage);
    }
}

ShaderNodeImplPtr GlslShaderGenerator::getImplementation(const NodeDef& nodedef, GenContext& context) const
{
    InterfaceElementPtr implElement = nodedef.getImplementation(getTarget());
    if (!implElement)
    {
        return nullptr;
    }

    const string& name = implElement->getName();

    // Check if it's created and cached already.
    ShaderNodeImplPtr impl = context.findNodeImplementation(name);
    if (impl)
    {
        return impl;
    }

    vector<OutputPtr> outputs = nodedef.getActiveOutputs();
    if (outputs.empty())
    {
        throw ExceptionShaderGenError("NodeDef '" + nodedef.getName() + "' has no outputs defined");
    }

    const TypeDesc* outputType = TypeDesc::get(outputs[0]->getType());

    if (implElement->isA<NodeGraph>())
    {
        // Use a compound implementation.
        if (*outputType == *Type::LIGHTSHADER)
        {
            impl = LightCompoundNodeGlsl::create();
        }
        else if (outputType->isClosure())
        {
            impl = ClosureCompoundNode::create();
        }
        else
        {
            impl = CompoundNode::create();
        }
    }
    else if (implElement->isA<Implementation>())
    {
        // Try creating a new in the factory.
        impl = _implFactory.create(name);
        if (!impl)
        {
            // Fall back to source code implementation.
            if (outputType->isClosure())
            {
                impl = ClosureSourceCodeNode::create();
            }
            else
            {
                impl = SourceCodeNode::create();
            }
        }
    }
    if (!impl)
    {
        return nullptr;
    }

    impl->initialize(*implElement, context);

    // Cache it.
    context.addNodeImplementation(name, impl);

    return impl;
}

const string& GlslImplementation::getTarget() const
{
    return GlslShaderGenerator::TARGET;
}

MATERIALX_NAMESPACE_END
