//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_CGLTFLOADER_H
#define MATERIALX_CGLTFLOADER_H

/// @file
/// GLTF format loader using the Cgltf library

#include <MaterialXRender/GeometryHandler.h>

MATERIALX_NAMESPACE_BEGIN

/// Shared pointer to a GLTFLoader
using CgltfLoaderPtr = std::shared_ptr<class CgltfLoader>;

/// @class CgltfLoader
/// Wrapper for loader to read in GLTF files using the Cgltf library.
class MX_RENDER_API CgltfLoader : public GeometryLoader
{
  public:
    CgltfLoader() :
        _debugLevel(0)
    {
        _extensions = { "glb", "GLB", "gltf", "GLTF" };
    }
    virtual ~CgltfLoader() { }

    /// Create a new loader
    static CgltfLoaderPtr create() { return std::make_shared<CgltfLoader>(); }

    /// Load geometry from file path
    bool load(const FilePath& filePath, MeshList& meshList, bool texcoordVerticalFlip = false) override;

  private:
    unsigned int _debugLevel;
};

MATERIALX_NAMESPACE_END

#endif
