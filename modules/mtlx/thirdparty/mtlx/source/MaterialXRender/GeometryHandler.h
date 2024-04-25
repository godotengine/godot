//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_GEOMETRYHANDLER_H
#define MATERIALX_GEOMETRYHANDLER_H

/// @file
/// Geometry loader interfaces

#include <MaterialXRender/Export.h>
#include <MaterialXRender/Mesh.h>

#include <MaterialXFormat/File.h>

#include <map>

MATERIALX_NAMESPACE_BEGIN

/// Shared pointer to a GeometryLoader
using GeometryLoaderPtr = std::shared_ptr<class GeometryLoader>;

/// @class GeometryLoader
/// Base class representing a geometry loader. A loader can be
/// associated with one or more file extensions.
class MX_RENDER_API GeometryLoader
{
  public:
    GeometryLoader()
    {
    }
    virtual ~GeometryLoader() { }

    /// Returns a list of supported extensions
    /// @return List of support extensions
    const StringSet& supportedExtensions() const
    {
        return _extensions;
    }

    /// Load geometry from disk. Must be implemented by derived classes.
    /// @param filePath Path to file to load
    /// @param meshList List of meshes to update
    /// @param texcoordVerticalFlip Flip texture coordinates in V when loading
    /// @return True if load was successful
    virtual bool load(const FilePath& filePath, MeshList& meshList, bool texcoordVerticalFlip = false) = 0;

  protected:
    // List of supported string extensions
    StringSet _extensions;
};

/// Shared pointer to an GeometryHandler
using GeometryHandlerPtr = std::shared_ptr<class GeometryHandler>;

/// Map of extensions to image loaders
using GeometryLoaderMap = std::multimap<string, GeometryLoaderPtr>;

/// @class GeometryHandler
/// Class which holds a set of geometry loaders. Each loader is associated with
/// a given set of file extensions.
class MX_RENDER_API GeometryHandler
{
  public:
    GeometryHandler()
    {
    }
    virtual ~GeometryHandler() { }

    /// Create a new geometry handler
    static GeometryHandlerPtr create()
    {
        return std::make_shared<GeometryHandler>();
    }

    /// Add a geometry loader
    /// @param loader Loader to add to list of available loaders.
    void addLoader(GeometryLoaderPtr loader);

    /// Get a list of extensions supported by the handler
    void supportedExtensions(StringSet& extensions);

    /// Clear all loaded geometry
    void clearGeometry();

    // Determine if any meshes have been loaded from a given location
    bool hasGeometry(const string& location);

    // Find all meshes loaded from a given location
    void getGeometry(MeshList& meshes, const string& location);

    /// Load geometry from a given location
    /// @param filePath Path to geometry
    /// @param texcoordVerticalFlip Flip texture coordinates in V. Default is to not flip.
    bool loadGeometry(const FilePath& filePath, bool texcoordVerticalFlip = false);

    /// Get list of meshes
    const MeshList& getMeshes() const
    {
        return _meshes;
    }

    /// Return the first mesh in our list containing the given partition.
    /// If no matching mesh is found, then nullptr is returned.
    MeshPtr findParentMesh(MeshPartitionPtr part);

    /// Return the minimum bounds for all meshes
    const Vector3& getMinimumBounds() const
    {
        return _minimumBounds;
    }

    /// Return the minimum bounds for all meshes
    const Vector3& getMaximumBounds() const
    {
        return _maximumBounds;
    }

    /// Utility to create a quad mesh
    static MeshPtr createQuadMesh(const Vector2& uvMin = Vector2(0.0f, 0.0f),
                                  const Vector2& uvMax = Vector2(1.0f, 1.0f),
                                  bool flipTexCoordsHorizontally = false);

  protected:
    // Recompute bounds for all stored geometry
    void computeBounds();

  protected:
    GeometryLoaderMap _geometryLoaders;
    MeshList _meshes;
    Vector3 _minimumBounds;
    Vector3 _maximumBounds;
};

MATERIALX_NAMESPACE_END

#endif
