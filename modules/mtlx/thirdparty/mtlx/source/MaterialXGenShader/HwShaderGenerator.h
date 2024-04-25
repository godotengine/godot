//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_HWSHADERGENERATOR_H
#define MATERIALX_HWSHADERGENERATOR_H

/// @file
/// Hardware shader generator base class

#include <MaterialXGenShader/Export.h>

#include <MaterialXGenShader/GenContext.h>
#include <MaterialXGenShader/ShaderGenerator.h>

MATERIALX_NAMESPACE_BEGIN

/*
The HW shader generators have a number of predefined variables (inputs and uniforms) with binding rules.
When these are used by a shader the application must bind them to the expected data. The following table is
a listing of the variables with a description of what data they should be bound to.

However, different renderers can have different requirements on naming conventions for these variables.
In order to facilitate this the generators will use token substitution for naming the variables. The
first colum below shows the token names that should be used in source code before the token substitution
is done. The second row shows the real identifier names that will be used by default after substitution.
An generator can override these identifier names in order to use a custom naming convention for these.
Overriding identifier names is done by changing the entries in the identifiers map given to the function
replaceIdentifiers(), which is handling the token substitution on a shader stage.

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    TOKEN NAME                          DEFAULT IDENTIFIER NAME             TYPE       BINDING
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Vertex input variables :
    $inPosition                         i_position                          vec3       Vertex position in object space
    $inNormal                           i_normal                            vec3       Vertex normal in object space
    $inTangent                          i_tangent                           vec3       Vertex tangent in object space
    $inBitangent                        i_bitangent                         vec3       Vertex bitangent in object space
    $inTexcoord_N                       i_texcoord_N                        vec2       Vertex texture coordinate for the N:th uv set
    $inColor_N                          i_color_N                           vec4       Vertex color for the N:th color set (RGBA)

Uniform variables :
    $worldMatrix                        u_worldMatrix                       mat4       World transformation
    $worldInverseMatrix                 u_worldInverseMatrix                mat4       World transformation, inverted
    $worldTransposeMatrix               u_worldTransposeMatrix              mat4       World transformation, transposed
    $worldInverseTransposeMatrix        u_worldInverseTransposeMatrix       mat4       World transformation, inverted and transposed
    $viewMatrix                         u_viewMatrix                        mat4       View transformation
    $viewInverseMatrix                  u_viewInverseMatrix                 mat4       View transformation, inverted
    $viewTransposeMatrix                u_viewTransposeMatrix               mat4       View transformation, transposed
    $viewInverseTransposeMatrix         u_viewInverseTransposeMatrix        mat4       View transformation, inverted and transposed
    $projectionMatrix                   u_projectionMatrix                  mat4       Projection transformation
    $projectionInverseMatrix            u_projectionInverseMatrix           mat4       Projection transformation, inverted
    $projectionTransposeMatrix          u_projectionTransposeMatrix         mat4       Projection transformation, transposed
    $projectionInverseTransposeMatrix   u_projectionInverseTransposeMatrix  mat4       Projection transformation, inverted and transposed
    $worldViewMatrix                    u_worldViewMatrix                   mat4       World-view transformation
    $viewProjectionMatrix               u_viewProjectionMatrix              mat4       View-projection transformation
    $worldViewProjectionMatrix          u_worldViewProjectionMatrix         mat4       World-view-projection transformation
    $viewPosition                       u_viewPosition                      vec3       World-space position of the view (camera)
    $viewDirection                      u_viewDirection                     vec3       World-space direction of the view (camera)
    $frame                              u_frame                             float      The current frame number as defined by the host application
    $time                               u_time                              float      The current time in seconds
    $geomprop_<name>                    u_geomprop_<name>                   <type>     A named property of given <type> where <name> is the name of the variable on the geometry
    $numActiveLightSources              u_numActiveLightSources             int        The number of currently active light sources. Note that in shader this is clamped against
                                                                                       the maximum allowed number of lights sources. The maximum number is set by the generation
                                                                                       option GenOptions.hwMaxActiveLightSources.
    $lightData[]                        u_lightData[]                       struct     Array of struct LightData holding parameters for active light sources.
                                                                                       The LightData struct is built dynamically depending on requirements for
                                                                                       bound light shaders.
    $envMatrix                          u_envMatrix                         mat4       Rotation matrix for the environment.
    $envIrradiance                      u_envIrradiance                     sampler2D  Sampler for the texture used for diffuse environment lighting.
    $envRadiance                        u_envRadiance                       sampler2D  Sampler for the texture used for specular environment lighting.
    $envRadianceMips                    u_envRadianceMips                   int        Number of mipmaps used on the specular environment texture.
    $envRadianceSamples                 u_envRadianceSamples                int        Samples to use if Filtered Importance Sampling is used for specular environment lighting.

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
*/

/// HW specific identifiers.
namespace HW
{
/// Token identifiers
extern MX_GENSHADER_API const string T_IN_POSITION;
extern MX_GENSHADER_API const string T_IN_NORMAL;
extern MX_GENSHADER_API const string T_IN_TANGENT;
extern MX_GENSHADER_API const string T_IN_BITANGENT;
extern MX_GENSHADER_API const string T_IN_TEXCOORD;
extern MX_GENSHADER_API const string T_IN_GEOMPROP;
extern MX_GENSHADER_API const string T_IN_COLOR;
extern MX_GENSHADER_API const string T_POSITION_WORLD;
extern MX_GENSHADER_API const string T_NORMAL_WORLD;
extern MX_GENSHADER_API const string T_TANGENT_WORLD;
extern MX_GENSHADER_API const string T_BITANGENT_WORLD;
extern MX_GENSHADER_API const string T_POSITION_OBJECT;
extern MX_GENSHADER_API const string T_NORMAL_OBJECT;
extern MX_GENSHADER_API const string T_TANGENT_OBJECT;
extern MX_GENSHADER_API const string T_BITANGENT_OBJECT;
extern MX_GENSHADER_API const string T_TEXCOORD;
extern MX_GENSHADER_API const string T_COLOR;
extern MX_GENSHADER_API const string T_WORLD_MATRIX;
extern MX_GENSHADER_API const string T_WORLD_INVERSE_MATRIX;
extern MX_GENSHADER_API const string T_WORLD_TRANSPOSE_MATRIX;
extern MX_GENSHADER_API const string T_WORLD_INVERSE_TRANSPOSE_MATRIX;
extern MX_GENSHADER_API const string T_VIEW_MATRIX;
extern MX_GENSHADER_API const string T_VIEW_INVERSE_MATRIX;
extern MX_GENSHADER_API const string T_VIEW_TRANSPOSE_MATRIX;
extern MX_GENSHADER_API const string T_VIEW_INVERSE_TRANSPOSE_MATRIX;
extern MX_GENSHADER_API const string T_PROJ_MATRIX;
extern MX_GENSHADER_API const string T_PROJ_INVERSE_MATRIX;
extern MX_GENSHADER_API const string T_PROJ_TRANSPOSE_MATRIX;
extern MX_GENSHADER_API const string T_PROJ_INVERSE_TRANSPOSE_MATRIX;
extern MX_GENSHADER_API const string T_WORLD_VIEW_MATRIX;
extern MX_GENSHADER_API const string T_VIEW_PROJECTION_MATRIX;
extern MX_GENSHADER_API const string T_WORLD_VIEW_PROJECTION_MATRIX;
extern MX_GENSHADER_API const string T_VIEW_POSITION;
extern MX_GENSHADER_API const string T_VIEW_DIRECTION;
extern MX_GENSHADER_API const string T_FRAME;
extern MX_GENSHADER_API const string T_TIME;
extern MX_GENSHADER_API const string T_GEOMPROP;
extern MX_GENSHADER_API const string T_ALPHA_THRESHOLD;
extern MX_GENSHADER_API const string T_NUM_ACTIVE_LIGHT_SOURCES;
extern MX_GENSHADER_API const string T_ENV_MATRIX;
extern MX_GENSHADER_API const string T_ENV_RADIANCE;
extern MX_GENSHADER_API const string T_ENV_RADIANCE_MIPS;
extern MX_GENSHADER_API const string T_ENV_RADIANCE_SAMPLES;
extern MX_GENSHADER_API const string T_ENV_IRRADIANCE;
extern MX_GENSHADER_API const string T_ENV_PREFILTER_MIP;
extern MX_GENSHADER_API const string T_REFRACTION_TWO_SIDED;
extern MX_GENSHADER_API const string T_ALBEDO_TABLE;
extern MX_GENSHADER_API const string T_ALBEDO_TABLE_SIZE;
extern MX_GENSHADER_API const string T_AMB_OCC_MAP;
extern MX_GENSHADER_API const string T_AMB_OCC_GAIN;
extern MX_GENSHADER_API const string T_SHADOW_MAP;
extern MX_GENSHADER_API const string T_SHADOW_MATRIX;
extern MX_GENSHADER_API const string T_VERTEX_DATA_INSTANCE;
extern MX_GENSHADER_API const string T_LIGHT_DATA_INSTANCE;

/// Default names for identifiers.
/// Replacing above tokens in final code.
extern MX_GENSHADER_API const string IN_POSITION;
extern MX_GENSHADER_API const string IN_NORMAL;
extern MX_GENSHADER_API const string IN_TANGENT;
extern MX_GENSHADER_API const string IN_BITANGENT;
extern MX_GENSHADER_API const string IN_TEXCOORD;
extern MX_GENSHADER_API const string IN_GEOMPROP;
extern MX_GENSHADER_API const string IN_COLOR;
extern MX_GENSHADER_API const string POSITION_WORLD;
extern MX_GENSHADER_API const string NORMAL_WORLD;
extern MX_GENSHADER_API const string TANGENT_WORLD;
extern MX_GENSHADER_API const string BITANGENT_WORLD;
extern MX_GENSHADER_API const string POSITION_OBJECT;
extern MX_GENSHADER_API const string NORMAL_OBJECT;
extern MX_GENSHADER_API const string TANGENT_OBJECT;
extern MX_GENSHADER_API const string BITANGENT_OBJECT;
extern MX_GENSHADER_API const string TEXCOORD;
extern MX_GENSHADER_API const string COLOR;
extern MX_GENSHADER_API const string WORLD_MATRIX;
extern MX_GENSHADER_API const string WORLD_INVERSE_MATRIX;
extern MX_GENSHADER_API const string WORLD_TRANSPOSE_MATRIX;
extern MX_GENSHADER_API const string WORLD_INVERSE_TRANSPOSE_MATRIX;
extern MX_GENSHADER_API const string VIEW_MATRIX;
extern MX_GENSHADER_API const string VIEW_INVERSE_MATRIX;
extern MX_GENSHADER_API const string VIEW_TRANSPOSE_MATRIX;
extern MX_GENSHADER_API const string VIEW_INVERSE_TRANSPOSE_MATRIX;
extern MX_GENSHADER_API const string PROJ_MATRIX;
extern MX_GENSHADER_API const string PROJ_INVERSE_MATRIX;
extern MX_GENSHADER_API const string PROJ_TRANSPOSE_MATRIX;
extern MX_GENSHADER_API const string PROJ_INVERSE_TRANSPOSE_MATRIX;
extern MX_GENSHADER_API const string WORLD_VIEW_MATRIX;
extern MX_GENSHADER_API const string VIEW_PROJECTION_MATRIX;
extern MX_GENSHADER_API const string WORLD_VIEW_PROJECTION_MATRIX;
extern MX_GENSHADER_API const string VIEW_POSITION;
extern MX_GENSHADER_API const string VIEW_DIRECTION;
extern MX_GENSHADER_API const string FRAME;
extern MX_GENSHADER_API const string TIME;
extern MX_GENSHADER_API const string GEOMPROP;
extern MX_GENSHADER_API const string ALPHA_THRESHOLD;
extern MX_GENSHADER_API const string NUM_ACTIVE_LIGHT_SOURCES;
extern MX_GENSHADER_API const string ENV_MATRIX;
extern MX_GENSHADER_API const string ENV_RADIANCE;
extern MX_GENSHADER_API const string ENV_RADIANCE_MIPS;
extern MX_GENSHADER_API const string ENV_RADIANCE_SAMPLES;
extern MX_GENSHADER_API const string ENV_IRRADIANCE;
extern MX_GENSHADER_API const string ENV_PREFILTER_MIP;
extern MX_GENSHADER_API const string REFRACTION_TWO_SIDED;
extern MX_GENSHADER_API const string ALBEDO_TABLE;
extern MX_GENSHADER_API const string ALBEDO_TABLE_SIZE;
extern MX_GENSHADER_API const string AMB_OCC_MAP;
extern MX_GENSHADER_API const string AMB_OCC_GAIN;
extern MX_GENSHADER_API const string SHADOW_MAP;
extern MX_GENSHADER_API const string SHADOW_MATRIX;
extern MX_GENSHADER_API const string VERTEX_DATA_INSTANCE;
extern MX_GENSHADER_API const string LIGHT_DATA_INSTANCE;
extern MX_GENSHADER_API const string LIGHT_DATA_MAX_LIGHT_SOURCES;

/// Variable blocks names.
extern MX_GENSHADER_API const string VERTEX_INPUTS;    // Geometric inputs for vertex stage.
extern MX_GENSHADER_API const string VERTEX_DATA;      // Connector block for data transfer from vertex stage to pixel stage.
extern MX_GENSHADER_API const string PRIVATE_UNIFORMS; // Uniform inputs set privately by application.
extern MX_GENSHADER_API const string PUBLIC_UNIFORMS;  // Uniform inputs visible in UI and set by user.
extern MX_GENSHADER_API const string LIGHT_DATA;       // Uniform inputs for light sources.
extern MX_GENSHADER_API const string PIXEL_OUTPUTS;    // Outputs from the main/pixel stage.

/// Variable names for lighting parameters.
extern MX_GENSHADER_API const string DIR_N;
extern MX_GENSHADER_API const string DIR_L;
extern MX_GENSHADER_API const string DIR_V;
extern MX_GENSHADER_API const string WORLD_POSITION;
extern MX_GENSHADER_API const string OCCLUSION;

/// Attribute names.
extern MX_GENSHADER_API const string ATTR_TRANSPARENT;

/// User data names.
extern MX_GENSHADER_API const string USER_DATA_LIGHT_SHADERS;
extern MX_GENSHADER_API const string USER_DATA_BINDING_CONTEXT;
} // namespace HW

namespace Stage
{
/// Identifier for vertex stage.
extern MX_GENSHADER_API const string VERTEX;
} // namespace Stage

class HwLightShaders;
class HwShaderGenerator;
class HwResourceBindingContext;

/// Shared pointer to a HwLightShaders
using HwLightShadersPtr = shared_ptr<class HwLightShaders>;
/// Shared pointer to a HwShaderGenerator
using HwShaderGeneratorPtr = shared_ptr<class HwShaderGenerator>;
/// Shared pointer to a HwResourceBindingContext
using HwResourceBindingContextPtr = shared_ptr<class HwResourceBindingContext>;

/// @class HwLightShaders
/// Hardware light shader user data
class MX_GENSHADER_API HwLightShaders : public GenUserData
{
  public:
    /// Create and return a new instance.
    static HwLightShadersPtr create()
    {
        return std::make_shared<HwLightShaders>();
    }

    /// Bind a light shader to a light type id.
    void bind(unsigned int type, ShaderNodePtr shader)
    {
        _shaders[type] = shader;
    }

    /// Unbind a light shader previously bound to a light type id.
    void unbind(unsigned int type)
    {
        _shaders.erase(type);
    }

    /// Clear all light shaders previously bound.
    void clear()
    {
        _shaders.clear();
    }

    /// Return the light shader bound to the given light type,
    /// or nullptr if not light shader is bound to this type.
    const ShaderNode* get(unsigned int type) const
    {
        auto it = _shaders.find(type);
        return it != _shaders.end() ? it->second.get() : nullptr;
    }

    /// Return the map of bound light shaders.
    const std::unordered_map<unsigned int, ShaderNodePtr>& get() const
    {
        return _shaders;
    }

  protected:
    std::unordered_map<unsigned int, ShaderNodePtr> _shaders;
};

/// @class HwShaderGenerator
/// Base class for shader generators targeting HW rendering.
class MX_GENSHADER_API HwShaderGenerator : public ShaderGenerator
{
  public:
    /// Add the function call for a single node.
    void emitFunctionCall(const ShaderNode& node, GenContext& context, ShaderStage& stage) const override;

    /// Emit code for active light count definitions and uniforms
    virtual void addStageLightingUniforms(GenContext& context, ShaderStage& stage) const;

    /// Return the closure contexts defined for the given node.
    void getClosureContexts(const ShaderNode& node, vector<ClosureContext*>& cct) const override;

    /// Bind a light shader to a light type id, for usage in surface shaders created
    /// by the generator. The lightTypeId should be a unique identifier for the light
    /// type (node definition) and the same id should be used when setting light parameters on a
    /// generated surface shader.
    static void bindLightShader(const NodeDef& nodeDef, unsigned int lightTypeId, GenContext& context);

    /// Unbind a light shader previously bound to the given light type id.
    static void unbindLightShader(unsigned int lightTypeId, GenContext& context);

    /// Unbind all light shaders previously bound.
    static void unbindLightShaders(GenContext& context);

    /// Determine the prefix of vertex data variables.
    virtual string getVertexDataPrefix(const VariableBlock& vertexData) const = 0;

    /// Types of closure contexts for HW.
    enum ClosureContextType
    {
        DEFAULT,
        REFLECTION,
        TRANSMISSION,
        INDIRECT,
        EMISSION
    };

    /// String constants for closure context suffixes.
    static const string CLOSURE_CONTEXT_SUFFIX_REFLECTION;
    static const string CLOSURE_CONTEXT_SUFFIX_TRANSMISSION;
    static const string CLOSURE_CONTEXT_SUFFIX_INDIRECT;

  protected:
    HwShaderGenerator(SyntaxPtr syntax);

    /// Create and initialize a new HW shader for shader generation.
    virtual ShaderPtr createShader(const string& name, ElementPtr element, GenContext& context) const;

    /// Closure contexts for defining closure functions.
    mutable ClosureContext _defDefault;
    mutable ClosureContext _defReflection;
    mutable ClosureContext _defTransmission;
    mutable ClosureContext _defIndirect;
    mutable ClosureContext _defEmission;
};

/// @class HwShaderGenerator
/// Base class for HW node implementations.
class MX_GENSHADER_API HwImplementation : public ShaderNodeImpl
{
  public:
    bool isEditable(const ShaderInput& input) const override;

  protected:
    HwImplementation() { }

    // Integer identifiers for coordinate spaces.
    // The order must match the order given for the space enum string in stdlib.
    enum Space
    {
        MODEL_SPACE = 0,
        OBJECT_SPACE = 1,
        WORLD_SPACE = 2
    };

    /// Internal string constants
    static const string SPACE;
    static const string INDEX;
    static const string GEOMPROP;
};

/// @class HwResourceBindingContext
/// Class representing a context for resource binding for hardware resources.
class MX_GENSHADER_API HwResourceBindingContext : public GenUserData
{
  public:
    virtual ~HwResourceBindingContext() { }

    // Initialize the context before generation starts.
    virtual void initialize() = 0;

    // Emit directives required for binding support
    virtual void emitDirectives(GenContext& context, ShaderStage& stage) = 0;

    // Emit uniforms with binding information
    virtual void emitResourceBindings(GenContext& context, const VariableBlock& uniforms, ShaderStage& stage) = 0;

    // Emit struct uniforms with binding information
    virtual void emitStructuredResourceBindings(GenContext& context, const VariableBlock& uniforms,
                                                ShaderStage& stage, const std::string& structInstanceName,
                                                const std::string& arraySuffix = EMPTY_STRING) = 0;
};

MATERIALX_NAMESPACE_END

#endif
