//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXGenOsl/OslShaderGenerator.h>
#include <MaterialXGenOsl/OslSyntax.h>

#include <MaterialXGenShader/GenContext.h>
#include <MaterialXGenShader/Shader.h>
#include <MaterialXGenShader/ShaderStage.h>
#include <MaterialXGenShader/Nodes/SwizzleNode.h>
#include <MaterialXGenShader/Nodes/ConvertNode.h>
#include <MaterialXGenShader/Nodes/CombineNode.h>
#include <MaterialXGenShader/Nodes/SwitchNode.h>
#include <MaterialXGenShader/Nodes/SourceCodeNode.h>
#include <MaterialXGenShader/Nodes/ClosureAddNode.h>
#include <MaterialXGenShader/Nodes/ClosureMixNode.h>
#include <MaterialXGenShader/Nodes/ClosureMultiplyNode.h>

#include <MaterialXGenOsl/Nodes/BlurNodeOsl.h>
#include <MaterialXGenOsl/Nodes/SurfaceNodeOsl.h>
#include <MaterialXGenOsl/Nodes/ClosureLayerNodeOsl.h>
#include <MaterialXGenOsl/Nodes/MaterialNodeOsl.h>

MATERIALX_NAMESPACE_BEGIN

const string OslShaderGenerator::TARGET = "genosl";

//
// OslShaderGenerator methods
//

OslShaderGenerator::OslShaderGenerator() :
    ShaderGenerator(OslSyntax::create())
{
    // Register build-in implementations

    // <!-- <switch> -->
    // <!-- 'which' type : float -->
    registerImplementation("IM_switch_float_" + OslShaderGenerator::TARGET, SwitchNode::create);
    registerImplementation("IM_switch_color3_" + OslShaderGenerator::TARGET, SwitchNode::create);
    registerImplementation("IM_switch_color4_" + OslShaderGenerator::TARGET, SwitchNode::create);
    registerImplementation("IM_switch_vector2_" + OslShaderGenerator::TARGET, SwitchNode::create);
    registerImplementation("IM_switch_vector3_" + OslShaderGenerator::TARGET, SwitchNode::create);
    registerImplementation("IM_switch_vector4_" + OslShaderGenerator::TARGET, SwitchNode::create);
    // <!-- 'which' type : integer -->
    registerImplementation("IM_switch_floatI_" + OslShaderGenerator::TARGET, SwitchNode::create);
    registerImplementation("IM_switch_color3I_" + OslShaderGenerator::TARGET, SwitchNode::create);
    registerImplementation("IM_switch_color4I_" + OslShaderGenerator::TARGET, SwitchNode::create);
    registerImplementation("IM_switch_vector2I_" + OslShaderGenerator::TARGET, SwitchNode::create);
    registerImplementation("IM_switch_vector3I_" + OslShaderGenerator::TARGET, SwitchNode::create);
    registerImplementation("IM_switch_vector4I_" + OslShaderGenerator::TARGET, SwitchNode::create);

    // <!-- <swizzle> -->
    // <!-- from type : float -->
    registerImplementation("IM_swizzle_float_color3_" + OslShaderGenerator::TARGET, SwizzleNode::create);
    registerImplementation("IM_swizzle_float_color4_" + OslShaderGenerator::TARGET, SwizzleNode::create);
    registerImplementation("IM_swizzle_float_vector2_" + OslShaderGenerator::TARGET, SwizzleNode::create);
    registerImplementation("IM_swizzle_float_vector3_" + OslShaderGenerator::TARGET, SwizzleNode::create);
    registerImplementation("IM_swizzle_float_vector4_" + OslShaderGenerator::TARGET, SwizzleNode::create);
    // <!-- from type : color3 -->
    registerImplementation("IM_swizzle_color3_float_" + OslShaderGenerator::TARGET, SwizzleNode::create);
    registerImplementation("IM_swizzle_color3_color3_" + OslShaderGenerator::TARGET, SwizzleNode::create);
    registerImplementation("IM_swizzle_color3_color4_" + OslShaderGenerator::TARGET, SwizzleNode::create);
    registerImplementation("IM_swizzle_color3_vector2_" + OslShaderGenerator::TARGET, SwizzleNode::create);
    registerImplementation("IM_swizzle_color3_vector3_" + OslShaderGenerator::TARGET, SwizzleNode::create);
    registerImplementation("IM_swizzle_color3_vector4_" + OslShaderGenerator::TARGET, SwizzleNode::create);
    // <!-- from type : color4 -->
    registerImplementation("IM_swizzle_color4_float_" + OslShaderGenerator::TARGET, SwizzleNode::create);
    registerImplementation("IM_swizzle_color4_color3_" + OslShaderGenerator::TARGET, SwizzleNode::create);
    registerImplementation("IM_swizzle_color4_color4_" + OslShaderGenerator::TARGET, SwizzleNode::create);
    registerImplementation("IM_swizzle_color4_vector2_" + OslShaderGenerator::TARGET, SwizzleNode::create);
    registerImplementation("IM_swizzle_color4_vector3_" + OslShaderGenerator::TARGET, SwizzleNode::create);
    registerImplementation("IM_swizzle_color4_vector4_" + OslShaderGenerator::TARGET, SwizzleNode::create);
    // <!-- from type : vector2 -->
    registerImplementation("IM_swizzle_vector2_float_" + OslShaderGenerator::TARGET, SwizzleNode::create);
    registerImplementation("IM_swizzle_vector2_color3_" + OslShaderGenerator::TARGET, SwizzleNode::create);
    registerImplementation("IM_swizzle_vector2_color4_" + OslShaderGenerator::TARGET, SwizzleNode::create);
    registerImplementation("IM_swizzle_vector2_vector2_" + OslShaderGenerator::TARGET, SwizzleNode::create);
    registerImplementation("IM_swizzle_vector2_vector3_" + OslShaderGenerator::TARGET, SwizzleNode::create);
    registerImplementation("IM_swizzle_vector2_vector4_" + OslShaderGenerator::TARGET, SwizzleNode::create);
    // <!-- from type : vector3 -->
    registerImplementation("IM_swizzle_vector3_float_" + OslShaderGenerator::TARGET, SwizzleNode::create);
    registerImplementation("IM_swizzle_vector3_color3_" + OslShaderGenerator::TARGET, SwizzleNode::create);
    registerImplementation("IM_swizzle_vector3_color4_" + OslShaderGenerator::TARGET, SwizzleNode::create);
    registerImplementation("IM_swizzle_vector3_vector2_" + OslShaderGenerator::TARGET, SwizzleNode::create);
    registerImplementation("IM_swizzle_vector3_vector3_" + OslShaderGenerator::TARGET, SwizzleNode::create);
    registerImplementation("IM_swizzle_vector3_vector4_" + OslShaderGenerator::TARGET, SwizzleNode::create);
    // <!-- from type : vector4 -->
    registerImplementation("IM_swizzle_vector4_float_" + OslShaderGenerator::TARGET, SwizzleNode::create);
    registerImplementation("IM_swizzle_vector4_color3_" + OslShaderGenerator::TARGET, SwizzleNode::create);
    registerImplementation("IM_swizzle_vector4_color4_" + OslShaderGenerator::TARGET, SwizzleNode::create);
    registerImplementation("IM_swizzle_vector4_vector2_" + OslShaderGenerator::TARGET, SwizzleNode::create);
    registerImplementation("IM_swizzle_vector4_vector3_" + OslShaderGenerator::TARGET, SwizzleNode::create);
    registerImplementation("IM_swizzle_vector4_vector4_" + OslShaderGenerator::TARGET, SwizzleNode::create);

    // <!-- <convert> -->
    registerImplementation("IM_convert_float_color3_" + OslShaderGenerator::TARGET, ConvertNode::create);
    registerImplementation("IM_convert_float_color4_" + OslShaderGenerator::TARGET, ConvertNode::create);
    registerImplementation("IM_convert_float_vector2_" + OslShaderGenerator::TARGET, ConvertNode::create);
    registerImplementation("IM_convert_float_vector3_" + OslShaderGenerator::TARGET, ConvertNode::create);
    registerImplementation("IM_convert_float_vector4_" + OslShaderGenerator::TARGET, ConvertNode::create);
    registerImplementation("IM_convert_vector2_vector3_" + OslShaderGenerator::TARGET, ConvertNode::create);
    registerImplementation("IM_convert_vector3_vector2_" + OslShaderGenerator::TARGET, ConvertNode::create);
    registerImplementation("IM_convert_vector3_color3_" + OslShaderGenerator::TARGET, ConvertNode::create);
    registerImplementation("IM_convert_vector3_vector4_" + OslShaderGenerator::TARGET, ConvertNode::create);
    registerImplementation("IM_convert_vector4_vector3_" + OslShaderGenerator::TARGET, ConvertNode::create);
    registerImplementation("IM_convert_vector4_color4_" + OslShaderGenerator::TARGET, ConvertNode::create);
    registerImplementation("IM_convert_color3_vector3_" + OslShaderGenerator::TARGET, ConvertNode::create);
    registerImplementation("IM_convert_color4_vector4_" + OslShaderGenerator::TARGET, ConvertNode::create);
    registerImplementation("IM_convert_color3_color4_" + OslShaderGenerator::TARGET, ConvertNode::create);
    registerImplementation("IM_convert_color4_color3_" + OslShaderGenerator::TARGET, ConvertNode::create);
    registerImplementation("IM_convert_boolean_float_" + OslShaderGenerator::TARGET, ConvertNode::create);
    registerImplementation("IM_convert_integer_float_" + OslShaderGenerator::TARGET, ConvertNode::create);

    // <!-- <combine> -->
    registerImplementation("IM_combine2_vector2_" + OslShaderGenerator::TARGET, CombineNode::create);
    registerImplementation("IM_combine2_color4CF_" + OslShaderGenerator::TARGET, CombineNode::create);
    registerImplementation("IM_combine2_vector4VF_" + OslShaderGenerator::TARGET, CombineNode::create);
    registerImplementation("IM_combine2_vector4VV_" + OslShaderGenerator::TARGET, CombineNode::create);
    registerImplementation("IM_combine3_color3_" + OslShaderGenerator::TARGET, CombineNode::create);
    registerImplementation("IM_combine3_vector3_" + OslShaderGenerator::TARGET, CombineNode::create);
    registerImplementation("IM_combine4_color4_" + OslShaderGenerator::TARGET, CombineNode::create);
    registerImplementation("IM_combine4_vector4_" + OslShaderGenerator::TARGET, CombineNode::create);

    // <!-- <blur> -->
    registerImplementation("IM_blur_float_" + OslShaderGenerator::TARGET, BlurNodeOsl::create);
    registerImplementation("IM_blur_color3_" + OslShaderGenerator::TARGET, BlurNodeOsl::create);
    registerImplementation("IM_blur_color4_" + OslShaderGenerator::TARGET, BlurNodeOsl::create);
    registerImplementation("IM_blur_vector2_" + OslShaderGenerator::TARGET, BlurNodeOsl::create);
    registerImplementation("IM_blur_vector3_" + OslShaderGenerator::TARGET, BlurNodeOsl::create);
    registerImplementation("IM_blur_vector4_" + OslShaderGenerator::TARGET, BlurNodeOsl::create);

    // <!-- <layer> -->
    registerImplementation("IM_layer_bsdf_" + OslShaderGenerator::TARGET, ClosureLayerNodeOsl::create);
    registerImplementation("IM_layer_vdf_" + OslShaderGenerator::TARGET, ClosureLayerNodeOsl::create);

#ifdef MATERIALX_OSL_LEGACY_CLOSURES

    // <!-- <mix> -->
    registerImplementation("IM_mix_bsdf_" + OslShaderGenerator::TARGET, ClosureMixNode::create);
    registerImplementation("IM_mix_edf_" + OslShaderGenerator::TARGET, ClosureMixNode::create);
    // <!-- <add> -->
    registerImplementation("IM_add_bsdf_" + OslShaderGenerator::TARGET, ClosureAddNode::create);
    registerImplementation("IM_add_edf_" + OslShaderGenerator::TARGET, ClosureAddNode::create);
    // <!-- <multiply> -->
    registerImplementation("IM_multiply_bsdfC_" + OslShaderGenerator::TARGET, ClosureMultiplyNode::create);
    registerImplementation("IM_multiply_bsdfF_" + OslShaderGenerator::TARGET, ClosureMultiplyNode::create);
    registerImplementation("IM_multiply_edfC_" + OslShaderGenerator::TARGET, ClosureMultiplyNode::create);
    registerImplementation("IM_multiply_edfF_" + OslShaderGenerator::TARGET, ClosureMultiplyNode::create);

#endif // MATERIALX_OSL_LEGACY_CLOSURES

    // <!-- <thin_film> -->
    registerImplementation("IM_thin_film_bsdf_" + OslShaderGenerator::TARGET, NopNode::create);

    // <!-- <surface> -->
    registerImplementation("IM_surface_" + OslShaderGenerator::TARGET, SurfaceNodeOsl::create);

    // <!-- <surfacematerial> -->
    registerImplementation("IM_surfacematerial_" + OslShaderGenerator::TARGET, MaterialNodeOsl::create);
}

ShaderPtr OslShaderGenerator::generate(const string& name, ElementPtr element, GenContext& context) const
{
    ShaderPtr shader = createShader(name, element, context);

    // Request fixed floating-point notation for consistency across targets.
    ScopedFloatFormatting fmt(Value::FloatFormatFixed);

    ShaderGraph& graph = shader->getGraph();
    ShaderStage& stage = shader->getStage(Stage::PIXEL);

    emitLibraryIncludes(stage, context);

    // Add global constants and type definitions
    emitTypeDefinitions(context, stage);
    emitLine("#define M_FLOAT_EPS 1e-8", stage, false);
    emitLineBreak(stage);

    // Set the include file to use for uv transformations,
    // depending on the vertical flip flag.
    if (context.getOptions().fileTextureVerticalFlip)
    {
        _tokenSubstitutions[ShaderGenerator::T_FILE_TRANSFORM_UV] = "mx_transform_uv_vflip.osl";
    }
    else
    {
        _tokenSubstitutions[ShaderGenerator::T_FILE_TRANSFORM_UV] = "mx_transform_uv.osl";
    }

    // Emit function definitions for all nodes
    emitFunctionDefinitions(graph, context, stage);

    // Emit shader type, determined from the first
    // output if there are multiple outputs.
    const ShaderGraphOutputSocket* outputSocket0 = graph.getOutputSocket(0);
    if (*outputSocket0->getType() == *Type::SURFACESHADER)
    {
        emitString("surface ", stage);
    }
    else if (*outputSocket0->getType() == *Type::VOLUMESHADER)
    {
        emitString("volume ", stage);
    }
    else
    {
        emitString("shader ", stage);
    }

    // Begin shader signature. Note that makeIdentifier() will sanitize the name.
    string functionName = shader->getName();
    _syntax->makeIdentifier(functionName, graph.getIdentifierMap());
    setFunctionName(functionName, stage);
    emitLine(functionName, stage, false);

    const ShaderMetadataVecPtr& metadata = graph.getMetadata();
    bool haveShaderMetaData = metadata && metadata->size();

    // Always emit node information
    emitScopeBegin(stage, Syntax::DOUBLE_SQUARE_BRACKETS);
    emitLine("string mtlx_category = \"" + element->getCategory() + "\"" + Syntax::COMMA, stage, false);
    emitLine("string mtlx_name = \"" + element->getQualifiedName(element->getName()) + "\"" +
                 (haveShaderMetaData ? Syntax::COMMA : EMPTY_STRING),
             stage, false);

    // Add any metadata if set on the graph.
    if (haveShaderMetaData)
    {
        for (size_t j = 0; j < metadata->size(); ++j)
        {
            const ShaderMetadata& data = metadata->at(j);
            const string& delim = (j == metadata->size() - 1) ? EMPTY_STRING : Syntax::COMMA;
            const string& dataType = _syntax->getTypeName(data.type);
            const string dataValue = _syntax->getValue(data.type, *data.value, true);
            emitLine(dataType + " " + data.name + " = " + dataValue + delim, stage, false);
        }
    }
    emitScopeEnd(stage, false, false);
    emitLineEnd(stage, false);

    emitScopeBegin(stage, Syntax::PARENTHESES);

    // Emit shader inputs
    emitShaderInputs(stage.getInputBlock(OSL::INPUTS), stage);
    emitShaderInputs(stage.getUniformBlock(OSL::UNIFORMS), stage);

    // Emit shader output
    const VariableBlock& outputs = stage.getOutputBlock(OSL::OUTPUTS);
    const ShaderPort* singleOutput = outputs.size() == 1 ? outputs[0] : NULL;

    const bool isSurfaceShaderOutput = singleOutput && *singleOutput->getType() == *Type::SURFACESHADER;

#ifdef MATERIALX_OSL_LEGACY_CLOSURES
    const bool isBsdfOutput = singleOutput && *singleOutput->getType() == *Type::BSDF;
#endif

    if (isSurfaceShaderOutput)
    {
        // Special case for having 'surfaceshader' as final output type.
        // This type is a struct internally (BSDF, EDF, opacity) so we must
        // declare this as a single closure color type in order for renderers
        // to understand this output.
        emitLine("output closure color " + singleOutput->getVariable() + " = 0", stage, false);
    }
#ifdef MATERIALX_OSL_LEGACY_CLOSURES
    else if (isBsdfOutput)
    {
        // Special case for having 'BSDF' as final output type.
        // For legacy closures this type is a struct internally (response, throughput, thickness, ior)
        // so we must declare this as a single closure color type in order for renderers
        // to understand this output.
        emitLine("output closure color " + singleOutput->getVariable() + " = 0", stage, false);
    }
#endif
    else
    {
        // Just emit all outputs the way they are declared.
        emitShaderOutputs(outputs, stage);
    }

    // End shader signature
    emitScopeEnd(stage);

    // Begin shader body
    emitFunctionBodyBegin(graph, context, stage);

    // Emit constants
    const VariableBlock& constants = stage.getConstantBlock();
    if (constants.size())
    {
        emitVariableDeclarations(constants, _syntax->getConstantQualifier(), Syntax::SEMICOLON, context, stage);
        emitLineBreak(stage);
    }

    // Inputs of type 'filename' has been generated into two shader inputs.
    // So here we construct a single 'textureresource' from these inputs,
    // to be used further downstream. See emitShaderInputs() for details.
    VariableBlock& inputs = stage.getUniformBlock(OSL::UNIFORMS);
    for (size_t i = 0; i < inputs.size(); ++i)
    {
        ShaderPort* input = inputs[i];
        if (*input->getType() == *Type::FILENAME)
        {
            // Construct the textureresource variable.
            const string newVariableName = input->getVariable() + "_";
            const string& type = _syntax->getTypeName(input->getType());
            emitLine(type + newVariableName + " = {" + input->getVariable() + ", " + input->getVariable() + "_colorspace}", stage);

            // Update the variable name to be used downstream.
            input->setVariable(newVariableName);
        }
    }

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

    // Emit final outputs
    if (isSurfaceShaderOutput)
    {
        // Special case for having 'surfaceshader' as final output type.
        // This type is a struct internally (BSDF, EDF, opacity) so we must
        // comvert this to a single closure color type in order for renderers
        // to understand this output.
        const ShaderGraphOutputSocket* socket = graph.getOutputSocket(0);
        const string result = getUpstreamResult(socket, context);
        emitScopeBegin(stage);
        emitLine("float opacity_weight = clamp(" + result + ".opacity, 0.0, 1.0)", stage);
        emitLine(singleOutput->getVariable() + " = (" + result + ".bsdf + " + result + ".edf) * opacity_weight + transparent() * (1.0 - opacity_weight)", stage);
        emitScopeEnd(stage);
    }
#ifdef MATERIALX_OSL_LEGACY_CLOSURES
    else if (isBsdfOutput)
    {
        // Special case for having 'BSDF' as final output type.
        // For legacy closures this type is a struct internally (response, throughput, thickness, ior)
        // so we must declare this as a single closure color type in order for renderers
        // to understand this output.
        const ShaderGraphOutputSocket* socket = graph.getOutputSocket(0);
        const string result = getUpstreamResult(socket, context);
        emitLine(singleOutput->getVariable() + " = " + result + ".response", stage);
    }
#endif
    else
    {
        // Assign results to final outputs.
        for (size_t i = 0; i < outputs.size(); ++i)
        {
            const ShaderGraphOutputSocket* outputSocket = graph.getOutputSocket(i);
            const string result = getUpstreamResult(outputSocket, context);
            emitLine(outputSocket->getVariable() + " = " + result, stage);
        }
    }

    // End shader body
    emitFunctionBodyEnd(graph, context, stage);

    // Perform token substitution
    replaceTokens(_tokenSubstitutions, stage);

    return shader;
}

void OslShaderGenerator::registerShaderMetadata(const DocumentPtr& doc, GenContext& context) const
{
    // Register all standard metadata.
    ShaderGenerator::registerShaderMetadata(doc, context);

    ShaderMetadataRegistryPtr registry = context.getUserData<ShaderMetadataRegistry>(ShaderMetadataRegistry::USER_DATA_NAME);
    if (!registry)
    {
        throw ExceptionShaderGenError("Registration of metadata faild");
    }

    // Rename the standard metadata names to corresponding OSL metadata names.
    const StringMap nameRemapping =
    {
        { ValueElement::UI_NAME_ATTRIBUTE, "label" },
        { ValueElement::UI_FOLDER_ATTRIBUTE, "page" },
        { ValueElement::UI_MIN_ATTRIBUTE, "min" },
        { ValueElement::UI_MAX_ATTRIBUTE, "max" },
        { ValueElement::UI_SOFT_MIN_ATTRIBUTE, "slidermin" },
        { ValueElement::UI_SOFT_MAX_ATTRIBUTE, "slidermax" },
        { ValueElement::UI_STEP_ATTRIBUTE, "sensitivity" },
        { ValueElement::DOC_ATTRIBUTE, "help" }
    };
    for (auto it : nameRemapping)
    {
        ShaderMetadata* data = registry->findMetadata(it.first);
        if (data)
        {
            data->name = it.second;
        }
    }
}

ShaderPtr OslShaderGenerator::createShader(const string& name, ElementPtr element, GenContext& context) const
{
    // Create the root shader graph
    ShaderGraphPtr graph = ShaderGraph::create(nullptr, name, element, context);
    ShaderPtr shader = std::make_shared<Shader>(name, graph);

    // Create our stage.
    ShaderStagePtr stage = createStage(Stage::PIXEL, *shader);
    stage->createUniformBlock(OSL::UNIFORMS);
    stage->createInputBlock(OSL::INPUTS);
    stage->createOutputBlock(OSL::OUTPUTS);

    // Create shader variables for all nodes that need this.
    createVariables(graph, context, *shader);

    // Create uniforms for the published graph interface.
    VariableBlock& uniforms = stage->getUniformBlock(OSL::UNIFORMS);
    for (ShaderGraphInputSocket* inputSocket : graph->getInputSockets())
    {
        // Only for inputs that are connected/used internally,
        // and are editable by users.
        if (inputSocket->getConnections().size() && graph->isEditable(*inputSocket))
        {
            uniforms.add(inputSocket->getSelf());
        }
    }

    // Create outputs from the graph interface.
    VariableBlock& outputs = stage->getOutputBlock(OSL::OUTPUTS);
    for (ShaderGraphOutputSocket* outputSocket : graph->getOutputSockets())
    {
        outputs.add(outputSocket->getSelf());
    }

    return shader;
}

void OslShaderGenerator::emitFunctionCalls(const ShaderGraph& graph, GenContext& context, ShaderStage& stage, uint32_t classification) const
{
    // Special handling for closures functions.
    if ((classification & ShaderNode::Classification::CLOSURE) != 0)
    {
        // Emit function calls for closures connected to the outputs.
        // These will internally emit other closure function calls
        // for upstream nodes if needed.
        for (ShaderGraphOutputSocket* outputSocket : graph.getOutputSockets())
        {
            const ShaderNode* upstream = outputSocket->getConnection() ? outputSocket->getConnection()->getNode() : nullptr;
            if (upstream && upstream->hasClassification(classification))
            {
                emitFunctionCall(*upstream, context, stage);
            }
        }
    }
    else
    {
        // Not a closures graph so just generate all
        // function calls in order.
        ShaderGenerator::emitFunctionCalls(graph, context, stage, classification);
    }
}

void OslShaderGenerator::emitFunctionBodyBegin(const ShaderNode& node, GenContext&, ShaderStage& stage, Syntax::Punctuation punc) const
{
    emitScopeBegin(stage, punc);

    if (node.hasClassification(ShaderNode::Classification::SHADER) || node.hasClassification(ShaderNode::Classification::CLOSURE))
    {
        emitLine("closure color null_closure = 0", stage);
    }
}

void OslShaderGenerator::emitLibraryIncludes(ShaderStage& stage, GenContext& context) const
{
    static const string INCLUDE_PREFIX = "#include \"";
    static const string INCLUDE_SUFFIX = "\"";
    static const StringVec INCLUDE_FILES =
    {
        "mx_funcs.h"
    };

    for (const string& file : INCLUDE_FILES)
    {
        FilePath path = context.resolveSourceFile(file, FilePath());

        // Force path to use slash since backslash even if escaped
        // gives problems when saving the source code to file.
        string pathStr = path.asString();
        std::replace(pathStr.begin(), pathStr.end(), '\\', '/');

        emitLine(INCLUDE_PREFIX + pathStr + INCLUDE_SUFFIX, stage, false);
    }

    emitLineBreak(stage);
}

void OslShaderGenerator::emitShaderInputs(const VariableBlock& inputs, ShaderStage& stage) const
{
    static const std::unordered_map<string, string> GEOMPROP_DEFINITIONS =
    {
        { "Pobject", "transform(\"object\", P)" },
        { "Pworld", "P" },
        { "Nobject", "transform(\"object\", N)" },
        { "Nworld", "N" },
        { "Tobject", "transform(\"object\", dPdu)" },
        { "Tworld", "dPdu" },
        { "Bobject", "transform(\"object\", dPdv)" },
        { "Bworld", "dPdv" },
        { "UV0", "{u,v}" },
        { "Vworld", "I" }
    };

    for (size_t i = 0; i < inputs.size(); ++i)
    {
        const ShaderPort* input = inputs[i];
        const string& type = _syntax->getTypeName(input->getType());

        if (*input->getType() == *Type::FILENAME)
        {
            // Shader inputs of type 'filename' (textures) need special handling.
            // In OSL codegen a 'filename' is translated to the custom type 'textureresource',
            // which is a struct containing a file string and a colorspace string.
            // For the published shader interface we here split this into two separate inputs,
            // which gives a nicer shader interface with widget metadata on each input.

            ValuePtr value = input->getValue();
            const string valueStr = value ? value->getValueString() : EMPTY_STRING;

            // Add the file string input
            emitLineBegin(stage);
            emitString("string " + input->getVariable() + " = \"" + valueStr + "\"", stage);
            emitMetadata(input, stage);
            emitString(",", stage);
            emitLineEnd(stage, false);

            // Add the colorspace string input
            emitLineBegin(stage);
            emitString("string " + input->getVariable() + "_colorspace = \"" + input->getColorSpace() + "\"", stage);
            emitLineEnd(stage, false);
            emitScopeBegin(stage, Syntax::DOUBLE_SQUARE_BRACKETS);
            emitLine("string widget = \"colorspace\"", stage, false);
            emitScopeEnd(stage, false, false);
        }
        else
        {
            emitLineBegin(stage);
            emitString(type + " " + input->getVariable(), stage);

            string value = _syntax->getValue(input, true);
            const string& geomprop = input->getGeomProp();
            if (!geomprop.empty())
            {
                auto it = GEOMPROP_DEFINITIONS.find(geomprop);
                if (it != GEOMPROP_DEFINITIONS.end())
                {
                    value = it->second;
                }
            }
            if (value.empty())
            {
                value = _syntax->getDefaultValue(input->getType());
            }

            emitString(" = " + value, stage);
            emitMetadata(input, stage);
        }

        if (i < inputs.size())
        {
            emitString(",", stage);
        }

        emitLineEnd(stage, false);
    }
}

void OslShaderGenerator::emitShaderOutputs(const VariableBlock& outputs, ShaderStage& stage) const
{
    for (size_t i = 0; i < outputs.size(); ++i)
    {
        const ShaderPort* output = outputs[i];
        const TypeDesc* outputType = output->getType();
        const string type = _syntax->getOutputTypeName(outputType);
        const string value = _syntax->getDefaultValue(outputType, true);
        const string& delim = (i == outputs.size() - 1) ? EMPTY_STRING : Syntax::COMMA;
        emitLine(type + " " + output->getVariable() + " = " + value + delim, stage, false);
    }
}

void OslShaderGenerator::emitMetadata(const ShaderPort* port, ShaderStage& stage) const
{
    static const std::unordered_map<const TypeDesc*, ShaderMetadata> UI_WIDGET_METADATA =
    {
        { Type::FLOAT, ShaderMetadata("widget", Type::STRING, Value::createValueFromStrings("number", Type::STRING->getName())) },
        { Type::INTEGER, ShaderMetadata("widget", Type::STRING, Value::createValueFromStrings("number", Type::STRING->getName())) },
        { Type::FILENAME, ShaderMetadata("widget", Type::STRING, Value::createValueFromStrings("filename", Type::STRING->getName())) },
        { Type::BOOLEAN, ShaderMetadata("widget", Type::STRING, Value::createValueFromStrings("checkBox", Type::STRING->getName())) }
    };

    static const std::set<const TypeDesc*> METADATA_TYPE_BLACKLIST =
    {
        Type::VECTOR2,  // Custom struct types doesn't support metadata declarations.
        Type::VECTOR4,  //
        Type::COLOR4,   //
        Type::FILENAME, //
        Type::BSDF      //
    };

    auto widgetMetadataIt = UI_WIDGET_METADATA.find(port->getType());
    const ShaderMetadata* widgetMetadata = widgetMetadataIt != UI_WIDGET_METADATA.end() ? &widgetMetadataIt->second : nullptr;
    const ShaderMetadataVecPtr& metadata = port->getMetadata();

    if (widgetMetadata || (metadata && metadata->size()))
    {
        StringVec metadataLines;
        if (metadata)
        {
            for (size_t j = 0; j < metadata->size(); ++j)
            {
                const ShaderMetadata& data = metadata->at(j);
                if (METADATA_TYPE_BLACKLIST.count(data.type) == 0)
                {
                    const string& delim = (widgetMetadata || j < metadata->size() - 1) ? Syntax::COMMA : EMPTY_STRING;
                    const string& dataType = _syntax->getTypeName(data.type);
                    const string dataValue = _syntax->getValue(data.type, *data.value, true);
                    metadataLines.push_back(dataType + " " + data.name + " = " + dataValue + delim);
                }
            }
        }
        if (widgetMetadata)
        {
            const string& dataType = _syntax->getTypeName(widgetMetadata->type);
            const string dataValue = _syntax->getValue(widgetMetadata->type, *widgetMetadata->value, true);
            metadataLines.push_back(dataType + " " + widgetMetadata->name + " = " + dataValue);
        }
        if (metadataLines.size())
        {
            emitLineEnd(stage, false);
            emitScopeBegin(stage, Syntax::DOUBLE_SQUARE_BRACKETS);
            for (const auto& line : metadataLines)
            {
                emitLine(line, stage, false);
            }
            emitScopeEnd(stage, false, false);
        }
    }
}

namespace OSL
{

// Identifiers for OSL variable blocks
const string UNIFORMS = "u";
const string INPUTS = "i";
const string OUTPUTS = "o";

} // namespace OSL

MATERIALX_NAMESPACE_END
