//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXGenShader/HwShaderGenerator.h>
#include <MaterialXGenShader/GenContext.h>
#include <MaterialXGenShader/Shader.h>

#include <MaterialXCore/Document.h>
#include <MaterialXCore/Definition.h>

MATERIALX_NAMESPACE_BEGIN

const string HwImplementation::SPACE = "space";
const string HwImplementation::INDEX = "index";
const string HwImplementation::GEOMPROP = "geomprop";

namespace
{

// When node inputs with these names are modified, we assume the
// associated HW shader must be recompiled.
const StringSet IMMUTABLE_INPUTS =
{
    "index",
    "space",
    "attrname"
};

} // anonymous namespace

namespace HW
{

const string T_IN_POSITION                    = "$inPosition";
const string T_IN_NORMAL                      = "$inNormal";
const string T_IN_TANGENT                     = "$inTangent";
const string T_IN_BITANGENT                   = "$inBitangent";
const string T_IN_TEXCOORD                    = "$inTexcoord";
const string T_IN_GEOMPROP                    = "$inGeomprop";
const string T_IN_COLOR                       = "$inColor";
const string T_POSITION_WORLD                 = "$positionWorld";
const string T_NORMAL_WORLD                   = "$normalWorld";
const string T_TANGENT_WORLD                  = "$tangentWorld";
const string T_BITANGENT_WORLD                = "$bitangentWorld";
const string T_POSITION_OBJECT                = "$positionObject";
const string T_NORMAL_OBJECT                  = "$normalObject";
const string T_TANGENT_OBJECT                 = "$tangentObject";
const string T_BITANGENT_OBJECT               = "$bitangentObject";
const string T_TEXCOORD                       = "$texcoord";
const string T_COLOR                          = "$color";
const string T_WORLD_MATRIX                   = "$worldMatrix";
const string T_WORLD_INVERSE_MATRIX           = "$worldInverseMatrix";
const string T_WORLD_TRANSPOSE_MATRIX         = "$worldTransposeMatrix";
const string T_WORLD_INVERSE_TRANSPOSE_MATRIX = "$worldInverseTransposeMatrix";
const string T_VIEW_MATRIX                    = "$viewMatrix";
const string T_VIEW_INVERSE_MATRIX            = "$viewInverseMatrix";
const string T_VIEW_TRANSPOSE_MATRIX          = "$viewTransposeMatrix";
const string T_VIEW_INVERSE_TRANSPOSE_MATRIX  = "$viewInverseTransposeMatrix";
const string T_PROJ_MATRIX                    = "$projectionMatrix";
const string T_PROJ_INVERSE_MATRIX            = "$projectionInverseMatrix";
const string T_PROJ_TRANSPOSE_MATRIX          = "$projectionTransposeMatrix";
const string T_PROJ_INVERSE_TRANSPOSE_MATRIX  = "$projectionInverseTransposeMatrix";
const string T_WORLD_VIEW_MATRIX              = "$worldViewMatrix";
const string T_VIEW_PROJECTION_MATRIX         = "$viewProjectionMatrix";
const string T_WORLD_VIEW_PROJECTION_MATRIX   = "$worldViewProjectionMatrix";
const string T_VIEW_POSITION                  = "$viewPosition";
const string T_VIEW_DIRECTION                 = "$viewDirection";
const string T_FRAME                          = "$frame";
const string T_TIME                           = "$time";
const string T_GEOMPROP                       = "$geomprop";
const string T_ALPHA_THRESHOLD                = "$alphaThreshold";
const string T_NUM_ACTIVE_LIGHT_SOURCES       = "$numActiveLightSources";
const string T_ENV_MATRIX                     = "$envMatrix";
const string T_ENV_RADIANCE                   = "$envRadiance";
const string T_ENV_RADIANCE_MIPS              = "$envRadianceMips";
const string T_ENV_RADIANCE_SAMPLES           = "$envRadianceSamples";
const string T_ENV_IRRADIANCE                 = "$envIrradiance";
const string T_ENV_LIGHT_INTENSITY            = "$envLightIntensity";
const string T_ENV_PREFILTER_MIP              = "$envPrefilterMip";
const string T_REFRACTION_TWO_SIDED           = "$refractionTwoSided";
const string T_ALBEDO_TABLE                   = "$albedoTable";
const string T_ALBEDO_TABLE_SIZE              = "$albedoTableSize";
const string T_AMB_OCC_MAP                    = "$ambOccMap";
const string T_AMB_OCC_GAIN                   = "$ambOccGain";
const string T_SHADOW_MAP                     = "$shadowMap";
const string T_SHADOW_MATRIX                  = "$shadowMatrix";
const string T_VERTEX_DATA_INSTANCE           = "$vd";
const string T_LIGHT_DATA_INSTANCE            = "$lightData";

const string IN_POSITION                      = "i_position";
const string IN_NORMAL                        = "i_normal";
const string IN_TANGENT                       = "i_tangent";
const string IN_BITANGENT                     = "i_bitangent";
const string IN_TEXCOORD                      = "i_texcoord";
const string IN_GEOMPROP                      = "i_geomprop";
const string IN_COLOR                         = "i_color";
const string POSITION_WORLD                   = "positionWorld";
const string NORMAL_WORLD                     = "normalWorld";
const string TANGENT_WORLD                    = "tangentWorld";
const string BITANGENT_WORLD                  = "bitangentWorld";
const string POSITION_OBJECT                  = "positionObject";
const string NORMAL_OBJECT                    = "normalObject";
const string TANGENT_OBJECT                   = "tangentObject";
const string BITANGENT_OBJECT                 = "bitangentObject";
const string TEXCOORD                         = "texcoord";
const string COLOR                            = "color";
const string WORLD_MATRIX                     = "u_worldMatrix";
const string WORLD_INVERSE_MATRIX             = "u_worldInverseMatrix";
const string WORLD_TRANSPOSE_MATRIX           = "u_worldTransposeMatrix";
const string WORLD_INVERSE_TRANSPOSE_MATRIX   = "u_worldInverseTransposeMatrix";
const string VIEW_MATRIX                      = "u_viewMatrix";
const string VIEW_INVERSE_MATRIX              = "u_viewInverseMatrix";
const string VIEW_TRANSPOSE_MATRIX            = "u_viewTransposeMatrix";
const string VIEW_INVERSE_TRANSPOSE_MATRIX    = "u_viewInverseTransposeMatrix";
const string PROJ_MATRIX                      = "u_projectionMatrix";
const string PROJ_INVERSE_MATRIX              = "u_projectionInverseMatrix";
const string PROJ_TRANSPOSE_MATRIX            = "u_projectionTransposeMatrix";
const string PROJ_INVERSE_TRANSPOSE_MATRIX    = "u_projectionInverseTransposeMatrix";
const string WORLD_VIEW_MATRIX                = "u_worldViewMatrix";
const string VIEW_PROJECTION_MATRIX           = "u_viewProjectionMatrix";
const string WORLD_VIEW_PROJECTION_MATRIX     = "u_worldViewProjectionMatrix";
const string VIEW_POSITION                    = "u_viewPosition";
const string VIEW_DIRECTION                   = "u_viewDirection";
const string FRAME                            = "u_frame";
const string TIME                             = "u_time";
const string GEOMPROP                         = "u_geomprop";
const string ALPHA_THRESHOLD                  = "u_alphaThreshold";
const string NUM_ACTIVE_LIGHT_SOURCES         = "u_numActiveLightSources";
const string ENV_MATRIX                       = "u_envMatrix";
const string ENV_RADIANCE                     = "u_envRadiance";
const string ENV_RADIANCE_MIPS                = "u_envRadianceMips";
const string ENV_RADIANCE_SAMPLES             = "u_envRadianceSamples";
const string ENV_IRRADIANCE                   = "u_envIrradiance";
const string ENV_LIGHT_INTENSITY              = "u_envLightIntensity";
const string ENV_PREFILTER_MIP                = "u_envPrefilterMip";
const string REFRACTION_TWO_SIDED             = "u_refractionTwoSided";
const string ALBEDO_TABLE                     = "u_albedoTable";
const string ALBEDO_TABLE_SIZE                = "u_albedoTableSize";
const string AMB_OCC_MAP                      = "u_ambOccMap";
const string AMB_OCC_GAIN                     = "u_ambOccGain";
const string SHADOW_MAP                       = "u_shadowMap";
const string SHADOW_MATRIX                    = "u_shadowMatrix";
const string VERTEX_DATA_INSTANCE             = "vd";
const string LIGHT_DATA_INSTANCE              = "u_lightData";
const string LIGHT_DATA_MAX_LIGHT_SOURCES     = "MAX_LIGHT_SOURCES";

const string VERTEX_INPUTS                    = "VertexInputs";
const string VERTEX_DATA                      = "VertexData";
const string PRIVATE_UNIFORMS                 = "PrivateUniforms";
const string PUBLIC_UNIFORMS                  = "PublicUniforms";
const string LIGHT_DATA                       = "LightData";
const string PIXEL_OUTPUTS                    = "PixelOutputs";
const string DIR_N                            = "N";
const string DIR_L                            = "L";
const string DIR_V                            = "V";
const string WORLD_POSITION                   = "P";
const string OCCLUSION                        = "occlusion";
const string ATTR_TRANSPARENT                 = "transparent";
const string USER_DATA_CLOSURE_CONTEXT        = "udcc";
const string USER_DATA_LIGHT_SHADERS          = "udls";
const string USER_DATA_BINDING_CONTEXT        = "udbinding";

} // namespace HW

namespace Stage
{

const string VERTEX = "vertex";

} // namespace Stage

const ClosureContext::Arguments ClosureContext::EMPTY_ARGUMENTS;

//
// HwShaderGenerator methods
//

const string HwShaderGenerator::CLOSURE_CONTEXT_SUFFIX_REFLECTION("_reflection");
const string HwShaderGenerator::CLOSURE_CONTEXT_SUFFIX_TRANSMISSION("_transmission");
const string HwShaderGenerator::CLOSURE_CONTEXT_SUFFIX_INDIRECT("_indirect");

HwShaderGenerator::HwShaderGenerator(SyntaxPtr syntax) :
    ShaderGenerator(syntax),
    _defDefault(HwShaderGenerator::ClosureContextType::DEFAULT),
    _defReflection(HwShaderGenerator::ClosureContextType::REFLECTION),
    _defTransmission(HwShaderGenerator::ClosureContextType::TRANSMISSION),
    _defIndirect(HwShaderGenerator::ClosureContextType::INDIRECT),
    _defEmission(HwShaderGenerator::ClosureContextType::EMISSION)
{
    // Assign default identifiers names for all tokens.
    // Derived generators can override these names.
    _tokenSubstitutions[HW::T_IN_POSITION] = HW::IN_POSITION;
    _tokenSubstitutions[HW::T_IN_NORMAL] = HW::IN_NORMAL;
    _tokenSubstitutions[HW::T_IN_TANGENT] = HW::IN_TANGENT;
    _tokenSubstitutions[HW::T_IN_BITANGENT] = HW::IN_BITANGENT;
    _tokenSubstitutions[HW::T_IN_TEXCOORD] = HW::IN_TEXCOORD;
    _tokenSubstitutions[HW::T_IN_GEOMPROP] = HW::IN_GEOMPROP;
    _tokenSubstitutions[HW::T_IN_COLOR] = HW::IN_COLOR;
    _tokenSubstitutions[HW::T_POSITION_WORLD] = HW::POSITION_WORLD;
    _tokenSubstitutions[HW::T_NORMAL_WORLD] = HW::NORMAL_WORLD;
    _tokenSubstitutions[HW::T_TANGENT_WORLD] = HW::TANGENT_WORLD;
    _tokenSubstitutions[HW::T_BITANGENT_WORLD] = HW::BITANGENT_WORLD;
    _tokenSubstitutions[HW::T_POSITION_OBJECT] = HW::POSITION_OBJECT;
    _tokenSubstitutions[HW::T_NORMAL_OBJECT] = HW::NORMAL_OBJECT;
    _tokenSubstitutions[HW::T_TANGENT_OBJECT] = HW::TANGENT_OBJECT;
    _tokenSubstitutions[HW::T_BITANGENT_OBJECT] = HW::BITANGENT_OBJECT;
    _tokenSubstitutions[HW::T_TEXCOORD] = HW::TEXCOORD;
    _tokenSubstitutions[HW::T_COLOR] = HW::COLOR;
    _tokenSubstitutions[HW::T_WORLD_MATRIX] = HW::WORLD_MATRIX;
    _tokenSubstitutions[HW::T_WORLD_INVERSE_MATRIX] = HW::WORLD_INVERSE_MATRIX;
    _tokenSubstitutions[HW::T_WORLD_TRANSPOSE_MATRIX] = HW::WORLD_TRANSPOSE_MATRIX;
    _tokenSubstitutions[HW::T_WORLD_INVERSE_TRANSPOSE_MATRIX] = HW::WORLD_INVERSE_TRANSPOSE_MATRIX;
    _tokenSubstitutions[HW::T_VIEW_MATRIX] = HW::VIEW_MATRIX;
    _tokenSubstitutions[HW::T_VIEW_INVERSE_MATRIX] = HW::VIEW_INVERSE_MATRIX;
    _tokenSubstitutions[HW::T_VIEW_TRANSPOSE_MATRIX] = HW::VIEW_TRANSPOSE_MATRIX;
    _tokenSubstitutions[HW::T_VIEW_INVERSE_TRANSPOSE_MATRIX] = HW::VIEW_INVERSE_TRANSPOSE_MATRIX;
    _tokenSubstitutions[HW::T_PROJ_MATRIX] = HW::PROJ_MATRIX;
    _tokenSubstitutions[HW::T_PROJ_INVERSE_MATRIX] = HW::PROJ_INVERSE_MATRIX;
    _tokenSubstitutions[HW::T_PROJ_TRANSPOSE_MATRIX] = HW::PROJ_TRANSPOSE_MATRIX;
    _tokenSubstitutions[HW::T_PROJ_INVERSE_TRANSPOSE_MATRIX] = HW::PROJ_INVERSE_TRANSPOSE_MATRIX;
    _tokenSubstitutions[HW::T_WORLD_VIEW_MATRIX] = HW::WORLD_VIEW_MATRIX;
    _tokenSubstitutions[HW::T_VIEW_PROJECTION_MATRIX] = HW::VIEW_PROJECTION_MATRIX;
    _tokenSubstitutions[HW::T_WORLD_VIEW_PROJECTION_MATRIX] = HW::WORLD_VIEW_PROJECTION_MATRIX;
    _tokenSubstitutions[HW::T_VIEW_POSITION] = HW::VIEW_POSITION;
    _tokenSubstitutions[HW::T_VIEW_DIRECTION] = HW::VIEW_DIRECTION;
    _tokenSubstitutions[HW::T_FRAME] = HW::FRAME;
    _tokenSubstitutions[HW::T_TIME] = HW::TIME;
    _tokenSubstitutions[HW::T_GEOMPROP] = HW::GEOMPROP;
    _tokenSubstitutions[HW::T_ALPHA_THRESHOLD] = HW::ALPHA_THRESHOLD;
    _tokenSubstitutions[HW::T_NUM_ACTIVE_LIGHT_SOURCES] = HW::NUM_ACTIVE_LIGHT_SOURCES;
    _tokenSubstitutions[HW::T_ENV_MATRIX] = HW::ENV_MATRIX;
    _tokenSubstitutions[HW::T_ENV_RADIANCE] = HW::ENV_RADIANCE;
    _tokenSubstitutions[HW::T_ENV_RADIANCE_MIPS] = HW::ENV_RADIANCE_MIPS;
    _tokenSubstitutions[HW::T_ENV_RADIANCE_SAMPLES] = HW::ENV_RADIANCE_SAMPLES;
    _tokenSubstitutions[HW::T_ENV_IRRADIANCE] = HW::ENV_IRRADIANCE;
    _tokenSubstitutions[HW::T_ENV_LIGHT_INTENSITY] = HW::ENV_LIGHT_INTENSITY;
    _tokenSubstitutions[HW::T_REFRACTION_TWO_SIDED] = HW::REFRACTION_TWO_SIDED;
    _tokenSubstitutions[HW::T_ALBEDO_TABLE] = HW::ALBEDO_TABLE;
    _tokenSubstitutions[HW::T_ALBEDO_TABLE_SIZE] = HW::ALBEDO_TABLE_SIZE;
    _tokenSubstitutions[HW::T_SHADOW_MAP] = HW::SHADOW_MAP;
    _tokenSubstitutions[HW::T_SHADOW_MATRIX] = HW::SHADOW_MATRIX;
    _tokenSubstitutions[HW::T_AMB_OCC_MAP] = HW::AMB_OCC_MAP;
    _tokenSubstitutions[HW::T_AMB_OCC_GAIN] = HW::AMB_OCC_GAIN;
    _tokenSubstitutions[HW::T_VERTEX_DATA_INSTANCE] = HW::VERTEX_DATA_INSTANCE;
    _tokenSubstitutions[HW::T_LIGHT_DATA_INSTANCE] = HW::LIGHT_DATA_INSTANCE;
    _tokenSubstitutions[HW::T_ENV_PREFILTER_MIP] = HW::ENV_PREFILTER_MIP;

    // Setup closure contexts for defining closure functions
    //
    // Reflection context
    _defReflection.setSuffix(Type::BSDF, CLOSURE_CONTEXT_SUFFIX_REFLECTION);
    _defReflection.addArgument(Type::BSDF, ClosureContext::Argument(Type::VECTOR3, HW::DIR_L));
    _defReflection.addArgument(Type::BSDF, ClosureContext::Argument(Type::VECTOR3, HW::DIR_V));
    _defReflection.addArgument(Type::BSDF, ClosureContext::Argument(Type::VECTOR3, HW::WORLD_POSITION));
    _defReflection.addArgument(Type::BSDF, ClosureContext::Argument(Type::FLOAT, HW::OCCLUSION));
    // Transmission context
    _defTransmission.setSuffix(Type::BSDF, CLOSURE_CONTEXT_SUFFIX_TRANSMISSION);
    _defTransmission.addArgument(Type::BSDF, ClosureContext::Argument(Type::VECTOR3, HW::DIR_V));
    // Environment context
    _defIndirect.setSuffix(Type::BSDF, CLOSURE_CONTEXT_SUFFIX_INDIRECT);
    _defIndirect.addArgument(Type::BSDF, ClosureContext::Argument(Type::VECTOR3, HW::DIR_V));
    // Emission context
    _defEmission.addArgument(Type::EDF, ClosureContext::Argument(Type::VECTOR3, HW::DIR_N));
    _defEmission.addArgument(Type::EDF, ClosureContext::Argument(Type::VECTOR3, HW::DIR_L));
}

ShaderPtr HwShaderGenerator::createShader(const string& name, ElementPtr element, GenContext& context) const
{
    // Create the root shader graph
    ShaderGraphPtr graph = ShaderGraph::create(nullptr, name, element, context);
    ShaderPtr shader = std::make_shared<Shader>(name, graph);

    // Check if there are inputs with default geomprops assigned. In order to bind the
    // corresponding data to these inputs we insert geomprop nodes in the graph.
    bool geomNodeAdded = false;
    for (ShaderGraphInputSocket* socket : graph->getInputSockets())
    {
        if (!socket->getGeomProp().empty())
        {
            ConstDocumentPtr doc = element->getDocument();
            GeomPropDefPtr geomprop = doc->getGeomPropDef(socket->getGeomProp());
            if (geomprop)
            {
                // A default geomprop was assigned to this graph input.
                // For all internal connections to this input, break the connection
                // and assign a geomprop node that generates this data.
                // Note: If a geomprop node exists already it is reused,
                // so only a single node per geometry type is created.
                ShaderInputVec connections = socket->getConnections();
                for (auto connection : connections)
                {
                    connection->breakConnection();
                    graph->addDefaultGeomNode(connection, *geomprop, context);
                    geomNodeAdded = true;
                }
            }
        }
    }
    // If nodes were added we need to re-sort the nodes in topological order.
    if (geomNodeAdded)
    {
        graph->topologicalSort();
    }

    // Create vertex stage.
    ShaderStagePtr vs = createStage(Stage::VERTEX, *shader);
    vs->createInputBlock(HW::VERTEX_INPUTS, "i_vs");

    // Each Stage must have three types of uniform blocks:
    // Private, Public and Sampler blocks
    // Public uniforms are inputs that should be published in a user interface for user interaction,
    // while private uniforms are internal variables needed by the system which should not be exposed in UI.
    // So when creating these uniforms for a shader node, if the variable is user-facing it should go into the public block,
    // and otherwise the private block.
    // All texture based objects should be added to Sampler block

    vs->createUniformBlock(HW::PRIVATE_UNIFORMS, "u_prv");
    vs->createUniformBlock(HW::PUBLIC_UNIFORMS, "u_pub");

    // Create required variables for vertex stage
    VariableBlock& vsInputs = vs->getInputBlock(HW::VERTEX_INPUTS);
    vsInputs.add(Type::VECTOR3, HW::T_IN_POSITION);
    VariableBlock& vsPrivateUniforms = vs->getUniformBlock(HW::PRIVATE_UNIFORMS);
    vsPrivateUniforms.add(Type::MATRIX44, HW::T_WORLD_MATRIX);
    vsPrivateUniforms.add(Type::MATRIX44, HW::T_VIEW_PROJECTION_MATRIX);

    // Create pixel stage.
    ShaderStagePtr ps = createStage(Stage::PIXEL, *shader);
    VariableBlockPtr psOutputs = ps->createOutputBlock(HW::PIXEL_OUTPUTS, "o_ps");

    // Create required Uniform blocks and any additonal blocks if needed.
    VariableBlockPtr psPrivateUniforms = ps->createUniformBlock(HW::PRIVATE_UNIFORMS, "u_prv");
    VariableBlockPtr psPublicUniforms = ps->createUniformBlock(HW::PUBLIC_UNIFORMS, "u_pub");
    VariableBlockPtr lightData = ps->createUniformBlock(HW::LIGHT_DATA, HW::T_LIGHT_DATA_INSTANCE);
    lightData->add(Type::INTEGER, "type");

    // Add a block for data from vertex to pixel shader.
    addStageConnectorBlock(HW::VERTEX_DATA, HW::T_VERTEX_DATA_INSTANCE, *vs, *ps);

    // Add uniforms for transparent rendering.
    if (context.getOptions().hwTransparency)
    {
        psPrivateUniforms->add(Type::FLOAT, HW::T_ALPHA_THRESHOLD, Value::createValue(0.001f));
    }

    // Add uniforms for shadow map rendering.
    if (context.getOptions().hwShadowMap)
    {
        psPrivateUniforms->add(Type::FILENAME, HW::T_SHADOW_MAP);
        psPrivateUniforms->add(Type::MATRIX44, HW::T_SHADOW_MATRIX, Value::createValue(Matrix44::IDENTITY));
    }

    // Add inputs and uniforms for ambient occlusion.
    if (context.getOptions().hwAmbientOcclusion)
    {
        addStageInput(HW::VERTEX_INPUTS, Type::VECTOR2, HW::T_IN_TEXCOORD + "_0", *vs);
        addStageConnector(HW::VERTEX_DATA, Type::VECTOR2, HW::T_TEXCOORD + "_0", *vs, *ps);
        psPrivateUniforms->add(Type::FILENAME, HW::T_AMB_OCC_MAP);
        psPrivateUniforms->add(Type::FLOAT, HW::T_AMB_OCC_GAIN, Value::createValue(1.0f));
    }

    // Add uniforms for environment lighting.
    bool lighting = graph->hasClassification(ShaderNode::Classification::SHADER | ShaderNode::Classification::SURFACE) ||
                    graph->hasClassification(ShaderNode::Classification::BSDF);
    if (lighting && context.getOptions().hwSpecularEnvironmentMethod != SPECULAR_ENVIRONMENT_NONE)
    {
        const Matrix44 yRotationPI = Matrix44::createScale(Vector3(-1, 1, -1));
        psPrivateUniforms->add(Type::MATRIX44, HW::T_ENV_MATRIX, Value::createValue(yRotationPI));
        psPrivateUniforms->add(Type::FILENAME, HW::T_ENV_RADIANCE);
        psPrivateUniforms->add(Type::FLOAT, HW::T_ENV_LIGHT_INTENSITY, Value::createValue(1.0f));
        psPrivateUniforms->add(Type::INTEGER, HW::T_ENV_RADIANCE_MIPS, Value::createValue<int>(1));
        psPrivateUniforms->add(Type::INTEGER, HW::T_ENV_RADIANCE_SAMPLES, Value::createValue<int>(16));
        psPrivateUniforms->add(Type::FILENAME, HW::T_ENV_IRRADIANCE);
        psPrivateUniforms->add(Type::BOOLEAN, HW::T_REFRACTION_TWO_SIDED);
    }

    // Add uniforms for the directional albedo table.
    if (context.getOptions().hwDirectionalAlbedoMethod == DIRECTIONAL_ALBEDO_TABLE ||
        context.getOptions().hwWriteAlbedoTable)
    {
        psPrivateUniforms->add(Type::FILENAME, HW::T_ALBEDO_TABLE);
        psPrivateUniforms->add(Type::INTEGER, HW::T_ALBEDO_TABLE_SIZE, Value::createValue<int>(64));
    }

    // Add uniforms for environment prefiltering.
    if (context.getOptions().hwWriteEnvPrefilter)
    {
        psPrivateUniforms->add(Type::FILENAME, HW::T_ENV_RADIANCE);
        psPrivateUniforms->add(Type::FLOAT, HW::T_ENV_LIGHT_INTENSITY, Value::createValue(1.0f));
        psPrivateUniforms->add(Type::INTEGER, HW::T_ENV_PREFILTER_MIP, Value::createValue<int>(1));
        const Matrix44 yRotationPI = Matrix44::createScale(Vector3(-1, 1, -1));
        psPrivateUniforms->add(Type::MATRIX44, HW::T_ENV_MATRIX, Value::createValue(yRotationPI));
        psPrivateUniforms->add(Type::INTEGER, HW::T_ENV_RADIANCE_MIPS, Value::createValue<int>(1));
    }

    // Create uniforms for the published graph interface
    for (ShaderGraphInputSocket* inputSocket : graph->getInputSockets())
    {
        // Only for inputs that are connected/used internally,
        // and are editable by users.
        if (!inputSocket->getConnections().empty() && graph->isEditable(*inputSocket))
        {
            psPublicUniforms->add(inputSocket->getSelf());
        }
    }

    // Add the pixel stage output. This needs to be a color4 for rendering,
    // so copy name and variable from the graph output but set type to color4.
    // TODO: Improve this to support multiple outputs and other data types.
    ShaderGraphOutputSocket* outputSocket = graph->getOutputSocket();
    ShaderPort* output = psOutputs->add(Type::COLOR4, outputSocket->getName());
    output->setVariable(outputSocket->getVariable());
    output->setPath(outputSocket->getPath());

    // Create shader variables for all nodes that need this.
    createVariables(graph, context, *shader);

    HwLightShadersPtr lightShaders = context.getUserData<HwLightShaders>(HW::USER_DATA_LIGHT_SHADERS);

    // For surface shaders we need light shaders
    if (lightShaders && graph->hasClassification(ShaderNode::Classification::SHADER | ShaderNode::Classification::SURFACE))
    {
        // Create shader variables for all bound light shaders
        for (const auto& it : lightShaders->get())
        {
            ShaderNode* node = it.second.get();
            node->getImplementation().createVariables(*node, context, *shader);
        }
    }

    //
    // For image textures we need to convert filenames into uniforms (texture samplers).
    // Any unconnected filename input on file texture nodes needs to have a corresponding
    // uniform.
    //

    // Start with top level graphs.
    vector<ShaderGraph*> graphStack = { graph.get() };
    if (lightShaders)
    {
        for (const auto& it : lightShaders->get())
        {
            ShaderNode* node = it.second.get();
            ShaderGraph* lightGraph = node->getImplementation().getGraph();
            if (lightGraph)
            {
                graphStack.push_back(lightGraph);
            }
        }
    }

    while (!graphStack.empty())
    {
        ShaderGraph* g = graphStack.back();
        graphStack.pop_back();

        for (ShaderNode* node : g->getNodes())
        {
            if (node->hasClassification(ShaderNode::Classification::FILETEXTURE))
            {
                for (ShaderInput* input : node->getInputs())
                {
                    if (!input->getConnection() && *input->getType() == *Type::FILENAME)
                    {
                        // Create the uniform using the filename type to make this uniform into a texture sampler.
                        ShaderPort* filename = psPublicUniforms->add(Type::FILENAME, input->getVariable(), input->getValue());
                        filename->setPath(input->getPath());

                        // Assing the uniform name to the input value
                        // so we can reference it during code generation.
                        input->setValue(Value::createValue(input->getVariable()));
                    }
                }
            }
            // Push subgraphs on the stack to process these as well.
            ShaderGraph* subgraph = node->getImplementation().getGraph();
            if (subgraph)
            {
                graphStack.push_back(subgraph);
            }
        }
    }

    if (context.getOptions().hwTransparency)
    {
        // Flag the shader as being transparent.
        shader->setAttribute(HW::ATTR_TRANSPARENT);
    }

    return shader;
}

void HwShaderGenerator::emitFunctionCall(const ShaderNode& node, GenContext& context, ShaderStage& stage) const
{
    // Check if it's emitted already.
    if (stage.isEmitted(node, context))
    {
        emitComment("Omitted node '" + node.getName() + "'. Function already called in this scope.", stage);
        return;
    }

    bool match = true;

    if (node.hasClassification(ShaderNode::Classification::CLOSURE) && !node.hasClassification(ShaderNode::Classification::SHADER))
    {
        // Check if we have a closure context to modify the function call.
        ClosureContext* cct = context.getClosureContext();
        if (cct)
        {
            match =
                // For reflection and environment we support reflective closures.
                ((cct->getType() == ClosureContextType::REFLECTION || cct->getType() == ClosureContextType::INDIRECT) &&
                 node.hasClassification(ShaderNode::Classification::BSDF_R)) ||
                // For transmissive we support transmissive closures.
                ((cct->getType() == ClosureContextType::TRANSMISSION) &&
                 (node.hasClassification(ShaderNode::Classification::BSDF_T) || node.hasClassification(ShaderNode::Classification::VDF))) ||
                // For emission we only support emission closures.
                ((cct->getType() == ClosureContextType::EMISSION) &&
                 (node.hasClassification(ShaderNode::Classification::EDF)));
        }
    }

    if (match)
    {
        // A match between closure context and node classification was found.
        // So add the function call in this context.
        stage.addFunctionCall(node, context);
    }
    else
    {
        // Context and node classification doesn't match so just
        // emit the output variable set to default value, in case
        // it is referenced by another nodes in this context.
        emitLineBegin(stage);
        emitOutput(node.getOutput(), true, true, context, stage);
        emitLineEnd(stage);

        // Register the node as emitted, but omit the function call.
        stage.addFunctionCall(node, context, false);
    }
}

void HwShaderGenerator::bindLightShader(const NodeDef& nodeDef, unsigned int lightTypeId, GenContext& context)
{
    if (TypeDesc::get(nodeDef.getType()) != Type::LIGHTSHADER)
    {
        throw ExceptionShaderGenError("Error binding light shader. Given nodedef '" + nodeDef.getName() + "' is not of lightshader type");
    }

    HwLightShadersPtr lightShaders = context.getUserData<HwLightShaders>(HW::USER_DATA_LIGHT_SHADERS);
    if (!lightShaders)
    {
        lightShaders = HwLightShaders::create();
        context.pushUserData(HW::USER_DATA_LIGHT_SHADERS, lightShaders);
    }

    if (lightShaders->get(lightTypeId))
    {
        throw ExceptionShaderGenError("Error binding light shader. Light type id '" + std::to_string(lightTypeId) +
                                      "' has already been bound");
    }

    ShaderNodePtr shader = ShaderNode::create(nullptr, nodeDef.getNodeString(), nodeDef, context);

    // Check if this is a graph implementation.
    // If so prepend the light struct instance name on all input socket variables,
    // since in generated code these inputs will be members of the light struct.
    ShaderGraph* graph = shader->getImplementation().getGraph();
    if (graph)
    {
        for (ShaderGraphInputSocket* inputSockets : graph->getInputSockets())
        {
            inputSockets->setVariable("light." + inputSockets->getName());
        }
    }

    lightShaders->bind(lightTypeId, shader);
}

void HwShaderGenerator::unbindLightShader(unsigned int lightTypeId, GenContext& context)
{
    HwLightShadersPtr lightShaders = context.getUserData<HwLightShaders>(HW::USER_DATA_LIGHT_SHADERS);
    if (lightShaders)
    {
        lightShaders->unbind(lightTypeId);
    }
}

void HwShaderGenerator::unbindLightShaders(GenContext& context)
{
    HwLightShadersPtr lightShaders = context.getUserData<HwLightShaders>(HW::USER_DATA_LIGHT_SHADERS);
    if (lightShaders)
    {
        lightShaders->clear();
    }
}

void HwShaderGenerator::getClosureContexts(const ShaderNode& node, vector<ClosureContext*>& ccts) const
{
    if (node.hasClassification(ShaderNode::Classification::BSDF))
    {
        if (node.hasClassification(ShaderNode::Classification::BSDF_R | ShaderNode::Classification::BSDF_T))
        {
            // A general BSDF handling both reflection and transmission
            ccts.push_back(&_defReflection);
            ccts.push_back(&_defTransmission);
            ccts.push_back(&_defIndirect);
        }
        else if (node.hasClassification(ShaderNode::Classification::BSDF_R))
        {
            // A BSDF for reflection only
            ccts.push_back(&_defReflection);
            ccts.push_back(&_defIndirect);
        }
        else if (node.hasClassification(ShaderNode::Classification::BSDF_T))
        {
            // A BSDF for transmission only
            ccts.push_back(&_defTransmission);
        }
    }
    else if (node.hasClassification(ShaderNode::Classification::EDF))
    {
        // An EDF
        ccts.push_back(&_defEmission);
    }
    else if (node.hasClassification(ShaderNode::Classification::SHADER))
    {
        // A shader
        ccts.push_back(&_defDefault);
    }
}

void HwShaderGenerator::addStageLightingUniforms(GenContext& context, ShaderStage& stage) const
{
    // Create uniform for number of active light sources
    if (context.getOptions().hwMaxActiveLightSources > 0)
    {
        ShaderPort* numActiveLights = addStageUniform(HW::PRIVATE_UNIFORMS, Type::INTEGER, HW::T_NUM_ACTIVE_LIGHT_SOURCES, stage);
        numActiveLights->setValue(Value::createValue<int>(0));
    }
}

bool HwImplementation::isEditable(const ShaderInput& input) const
{
    return IMMUTABLE_INPUTS.count(input.getName()) == 0;
}

MATERIALX_NAMESPACE_END
