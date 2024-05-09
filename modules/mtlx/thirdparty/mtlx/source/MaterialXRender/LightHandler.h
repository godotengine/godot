//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_LIGHTHANDLER_H
#define MATERIALX_LIGHTHANDLER_H

/// @file
/// Handler for hardware lights

#include <MaterialXRender/Export.h>
#include <MaterialXRender/Image.h>
#include <MaterialXRender/Util.h>

#include <MaterialXCore/Document.h>

MATERIALX_NAMESPACE_BEGIN

extern MX_RENDER_API const int DEFAULT_ENV_SAMPLE_COUNT;

class GenContext;

/// Shared pointer to a LightHandler
using LightHandlerPtr = std::shared_ptr<class LightHandler>;

/// An unordered map from light names to light indices.
using LightIdMap = std::unordered_map<string, unsigned int>;

/// @class LightHandler
/// Utility light handler for creating and providing
/// light data for shader binding.
class MX_RENDER_API LightHandler
{
  public:
    LightHandler() :
        _lightTransform(Matrix44::IDENTITY),
        _directLighting(true),
        _indirectLighting(true),
        _usePrefilteredMap(false),
        _envLightIntensity(1.0f),
        _envSampleCount(DEFAULT_ENV_SAMPLE_COUNT),
        _refractionTwoSided(false)
    {
    }
    virtual ~LightHandler() { }

    /// Create a new light handler
    static LightHandlerPtr create() { return std::make_shared<LightHandler>(); }

    /// @name Global State
    /// @{

    /// Set the light transform.
    void setLightTransform(const Matrix44& mat)
    {
        _lightTransform = mat;
    }

    /// Return the light transform.
    Matrix44 getLightTransform() const
    {
        return _lightTransform;
    }

    /// Set whether direct lighting is enabled.
    void setDirectLighting(bool enable)
    {
        _directLighting = enable;
    }

    /// Return whether direct lighting is enabled.
    bool getDirectLighting() const
    {
        return _directLighting;
    }

    /// Set whether indirect lighting is enabled.
    void setIndirectLighting(bool enable)
    {
        _indirectLighting = enable;
    }

    /// Return whether indirect lighting is enabled.
    bool getIndirectLighting() const
    {
        return _indirectLighting;
    }

    /// @}
    /// @name Environment Lighting
    /// @{

    /// Set the environment radiance map
    void setEnvRadianceMap(ImagePtr map)
    {
        _envRadianceMap = map;
    }

    /// Return the environment radiance map
    ImagePtr getEnvRadianceMap() const
    {
        return _envRadianceMap;
    }

    /// Set the environment radiance map for the prefiltered environment lighting model.
    void setEnvPrefilteredMap(ImagePtr map)
    {
        _envPrefilteredMap = map;
    }

    /// Return the environment radiance map for the prefiltered environment lighting model.
    ImagePtr getEnvPrefilteredMap() const
    {
        return _envPrefilteredMap;
    }

    /// Set whether to use the prefiltered environment lighting model.
    void setUsePrefilteredMap(bool val)
    {
        _usePrefilteredMap = val;
    }

    /// Return whether to use the prefiltered environment lighting model.
    bool getUsePrefilteredMap()
    {
        return _usePrefilteredMap;
    }

    /// Set the environment irradiance map
    void setEnvIrradianceMap(ImagePtr map)
    {
        _envIrradianceMap = map;
    }

    /// Return the environment irradiance map
    ImagePtr getEnvIrradianceMap() const
    {
        return _envIrradianceMap;
    }

    /// Set the environment lighting sample count.
    void setEnvSampleCount(int count)
    {
        _envSampleCount = count;
    }

    /// Return the environment lighting sample count.
    int getEnvSampleCount() const
    {
        return _envSampleCount;
    }

    /// Set the environment light intensity.
    void setEnvLightIntensity(const float intensity)
    {
        _envLightIntensity = intensity;
    }

    /// Return the environment light intensity.
    float getEnvLightIntensity()
    {
        return _envLightIntensity;
    }

    /// Set the two-sided refraction property.
    void setRefractionTwoSided(bool enable)
    {
        _refractionTwoSided = enable;
    }

    /// Return the two-sided refraction property.
    int getRefractionTwoSided() const
    {
        return _refractionTwoSided;
    }

    /// @}
    /// @name Albedo Table
    /// @{

    /// Set the directional albedo table
    void setAlbedoTable(ImagePtr table)
    {
        _albedoTable = table;
    }

    /// Return the directional albedo table
    ImagePtr getAlbedoTable() const
    {
        return _albedoTable;
    }

    /// @}
    /// @name Light Sources
    /// @{

    /// Add a light source.
    void addLightSource(NodePtr node);

    /// Set the vector of light sources.
    void setLightSources(const vector<NodePtr>& lights)
    {
        _lightSources = lights;
    }

    /// Return the vector of light sources.
    const vector<NodePtr>& getLightSources() const
    {
        return _lightSources;
    }

    /// Return the first light source, if any, of the given category.
    NodePtr getFirstLightOfCategory(const string& category)
    {
        for (NodePtr light : _lightSources)
        {
            if (light->getCategory() == category)
            {
                return light;
            }
        }
        return nullptr;
    }

    /// @}
    /// @name Light IDs
    /// @{

    /// Get a list of identifiers associated with a given light nodedef
    const std::unordered_map<string, unsigned int>& getLightIdMap() const
    {
        return _lightIdMap;
    }

    /// From a set of nodes, create a mapping of corresponding
    /// nodedef identifiers to numbers
    LightIdMap computeLightIdMap(const vector<NodePtr>& nodes);

    /// Find lights to use based on an input document
    /// @param doc Document to scan for lights
    /// @param lights List of lights found in document
    void findLights(DocumentPtr doc, vector<NodePtr>& lights);

    /// Register light node definitions and light count with a given generation context
    /// @param doc Document containing light nodes and definitions
    /// @param lights Lights to register
    /// @param context Context to update
    void registerLights(DocumentPtr doc, const vector<NodePtr>& lights, GenContext& context);

    /// @}

  protected:
    Matrix44 _lightTransform;
    bool _directLighting;
    bool _indirectLighting;
    bool _usePrefilteredMap;

    ImagePtr _envRadianceMap;
    ImagePtr _envPrefilteredMap;
    ImagePtr _envIrradianceMap;
    float _envLightIntensity;
    int _envSampleCount;

    bool _refractionTwoSided;

    ImagePtr _albedoTable;

    vector<NodePtr> _lightSources;
    std::unordered_map<string, unsigned int> _lightIdMap;
};

MATERIALX_NAMESPACE_END

#endif
