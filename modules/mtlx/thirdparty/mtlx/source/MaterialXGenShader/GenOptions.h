//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_GENOPTIONS_H
#define MATERIALX_GENOPTIONS_H

/// @file
/// Shader generation options class

#include <MaterialXGenShader/Export.h>

#include <MaterialXFormat/File.h>

MATERIALX_NAMESPACE_BEGIN

/// Type of shader interface to be generated
enum ShaderInterfaceType
{
    /// Create a complete interface with uniforms for all
    /// editable inputs on all nodes used by the shader.
    /// This interface makes the shader fully editable by
    /// value without requiring any rebuilds.
    /// This is the default interface type.
    SHADER_INTERFACE_COMPLETE,

    /// Create a reduced interface with uniforms only for
    /// the inputs that has been declared in the shaders
    /// nodedef interface. If values on other inputs are
    /// changed the shader needs to be rebuilt.
    SHADER_INTERFACE_REDUCED
};

/// Method to use for specular environment lighting
enum HwSpecularEnvironmentMethod
{
    /// Do not use specular environment maps
    SPECULAR_ENVIRONMENT_NONE,

    /// Use Filtered Importance Sampling for
    /// specular environment/indirect lighting.
    SPECULAR_ENVIRONMENT_FIS,

    /// Use pre-filtered environment maps for
    /// specular environment/indirect lighting.
    SPECULAR_ENVIRONMENT_PREFILTER
};

/// Method to use for directional albedo evaluation
enum HwDirectionalAlbedoMethod
{
    /// Use an analytic approximation for directional albedo.
    DIRECTIONAL_ALBEDO_ANALYTIC,

    /// Use a table look-up for directional albedo.
    DIRECTIONAL_ALBEDO_TABLE,

    /// Use Monte Carlo integration for directional albedo.
    DIRECTIONAL_ALBEDO_MONTE_CARLO
};

/// Method to use for transmission rendering
enum HwTransmissionRenderMethod
{
    /// Use a refraction approximation for transmission rendering
    TRANSMISSION_REFRACTION,

    /// Use opacity for transmission rendering
    TRANSMISSION_OPACITY,
};

/// @class GenOptions
/// Class holding options to configure shader generation.
class MX_GENSHADER_API GenOptions
{
  public:
    GenOptions() :
        shaderInterfaceType(SHADER_INTERFACE_COMPLETE),
        fileTextureVerticalFlip(false),
        addUpstreamDependencies(true),
        libraryPrefix("libraries"),
        hwTransparency(false),
        hwSpecularEnvironmentMethod(SPECULAR_ENVIRONMENT_FIS),
        hwDirectionalAlbedoMethod(DIRECTIONAL_ALBEDO_ANALYTIC),
        hwTransmissionRenderMethod(TRANSMISSION_REFRACTION),
        hwWriteDepthMoments(false),
        hwShadowMap(false),
        hwAmbientOcclusion(false),
        hwMaxActiveLightSources(3),
        hwNormalizeUdimTexCoords(false),
        hwWriteAlbedoTable(false),
        hwWriteEnvPrefilter(false),
        hwImplicitBitangents(true),
        emitColorTransforms(true)
    {
    }
    virtual ~GenOptions() { }

    // TODO: Add options for:
    //  - shader gen optimization level
    //  - graph flattening or not

    /// Sets the type of shader interface to be generated
    ShaderInterfaceType shaderInterfaceType;

    /// If true the y-component of texture coordinates used for sampling
    /// file textures will be flipped before sampling. This can be used if
    /// file textures need to be flipped vertically to match the target's
    /// texture space convention. By default this option is false.
    bool fileTextureVerticalFlip;

    /// An optional override for the target color space.
    /// Shader fragments will be generated to transform
    /// input values and textures into this color space.
    string targetColorSpaceOverride;

    /// Define the target distance unit.
    /// Shader fragments will be generated to transform
    /// input distance values to the given unit.
    string targetDistanceUnit;

    /// Sets whether to include upstream dependencies
    /// for the element to generate a shader for.
    bool addUpstreamDependencies;

    /// The standard library prefix, which will be applied to
    /// calls to emitLibraryInclude during code generation.
    /// Defaults to "libraries".
    FilePath libraryPrefix;

    /// Sets if transparency is needed or not for HW shaders.
    /// If a surface shader has potential of being transparent
    /// this must be set to true, otherwise no transparency
    /// code fragments will be generated for the shader and
    /// the surface will be fully opaque.
    bool hwTransparency;

    /// Sets the method to use for specular environment lighting
    /// for HW shader targets.
    HwSpecularEnvironmentMethod hwSpecularEnvironmentMethod;

    /// Sets the method to use for directional albedo evaluation
    /// for HW shader targets.
    HwDirectionalAlbedoMethod hwDirectionalAlbedoMethod;

    /// Sets the method to use for transmission rendering
    /// for HW shader targets.
    HwTransmissionRenderMethod hwTransmissionRenderMethod;

    /// Enables the writing of depth moments for HW shader targets.
    /// Defaults to false.
    bool hwWriteDepthMoments;

    /// Enables shadow mapping for HW shader targets.
    /// Defaults to false.
    bool hwShadowMap;

    /// Enables ambient occlusion rendering for HW shader targets.
    /// Defaults to false.
    bool hwAmbientOcclusion;

    /// Sets the maximum number of light sources that can
    /// be active at once.
    unsigned int hwMaxActiveLightSources;

    /// Sets whether to transform texture coordinates to normalize
    /// uv space when UDIMs images are bound to an image. Can be
    /// enabled for when texture atlas generation is performed to
    /// compress a set of UDIMs into a single normalized image for
    /// hardware rendering.
    bool hwNormalizeUdimTexCoords;

    /// Enables the writing of a directional albedo table.
    /// Defaults to false.
    bool hwWriteAlbedoTable;

    /// Enables the generation of a prefiltered environment map.
    /// Defaults to false.
    bool hwWriteEnvPrefilter;

    /// Calculate fallback bitangents from existing normals and tangents
    /// inside the bitangent node.
    bool hwImplicitBitangents;

    /// Enable emitting colorspace transform code if a color management
    /// system is defined. Defaults to true.
    bool emitColorTransforms;
};

MATERIALX_NAMESPACE_END

#endif
