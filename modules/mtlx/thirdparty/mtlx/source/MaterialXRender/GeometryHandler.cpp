//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXRender/GeometryHandler.h>

#include <MaterialXGenShader/HwShaderGenerator.h>
#include <MaterialXGenShader/Util.h>

#include <limits>

MATERIALX_NAMESPACE_BEGIN

void GeometryHandler::addLoader(GeometryLoaderPtr loader)
{
    const StringSet& extensions = loader->supportedExtensions();
    for (const auto& extension : extensions)
    {
        _geometryLoaders.emplace(extension, loader);
    }
}

void GeometryHandler::supportedExtensions(StringSet& extensions)
{
    extensions.clear();
    for (const auto& loader : _geometryLoaders)
    {
        const StringSet& loaderExtensions = loader.second->supportedExtensions();
        extensions.insert(loaderExtensions.begin(), loaderExtensions.end());
    }
}

void GeometryHandler::clearGeometry()
{
    _meshes.clear();
    computeBounds();
}

bool GeometryHandler::hasGeometry(const string& location)
{
    for (const auto& mesh : _meshes)
    {
        if (mesh->getSourceUri() == location)
        {
            return true;
        }
    }
    return false;
}

void GeometryHandler::getGeometry(MeshList& meshes, const string& location)
{
    meshes.clear();
    for (const auto& mesh : _meshes)
    {
        if (mesh->getSourceUri() == location)
        {
            meshes.push_back(mesh);
        }
    }
}

void GeometryHandler::computeBounds()
{
    const float MAX_FLOAT = std::numeric_limits<float>::max();
    _minimumBounds = { MAX_FLOAT, MAX_FLOAT, MAX_FLOAT };
    _maximumBounds = { -MAX_FLOAT, -MAX_FLOAT, -MAX_FLOAT };
    for (const auto& mesh : _meshes)
    {
        const Vector3& minMesh = mesh->getMinimumBounds();
        _minimumBounds[0] = std::min(minMesh[0], _minimumBounds[0]);
        _minimumBounds[1] = std::min(minMesh[1], _minimumBounds[1]);
        _minimumBounds[2] = std::min(minMesh[2], _minimumBounds[2]);
        const Vector3& maxMesh = mesh->getMaximumBounds();
        _maximumBounds[0] = std::max(maxMesh[0], _maximumBounds[0]);
        _maximumBounds[1] = std::max(maxMesh[1], _maximumBounds[1]);
        _maximumBounds[2] = std::max(maxMesh[2], _maximumBounds[2]);
    }
}

bool GeometryHandler::loadGeometry(const FilePath& filePath, bool texcoordVerticalFlip)
{
    // Early return if already loaded
    if (hasGeometry(filePath))
    {
        return true;
    }

    bool loaded = false;

    std::pair<GeometryLoaderMap::iterator, GeometryLoaderMap::iterator> range;
    string extension = filePath.getExtension();
    range = _geometryLoaders.equal_range(extension);
    GeometryLoaderMap::iterator first = --range.second;
    GeometryLoaderMap::iterator last = --range.first;
    for (auto it = first; it != last; --it)
    {
        loaded = it->second->load(filePath, _meshes, texcoordVerticalFlip);
        if (loaded)
        {
            break;
        }
    }

    // Recompute bounds if load was successful
    if (loaded)
    {
        computeBounds();
    }

    return loaded;
}

MeshPtr GeometryHandler::findParentMesh(MeshPartitionPtr part)
{
    for (MeshPtr mesh : getMeshes())
    {
        for (size_t i = 0; i < mesh->getPartitionCount(); i++)
        {
            if (mesh->getPartition(i) == part)
            {
                return mesh;
            }
        }
    }

    return nullptr;
}

MeshPtr GeometryHandler::createQuadMesh(const Vector2& uvMin, const Vector2& uvMax, bool flipTexCoordsHorizontally)
{
    MeshStreamPtr quadPositions = MeshStream::create(HW::IN_POSITION, MeshStream::POSITION_ATTRIBUTE, 0);
    quadPositions->setStride(MeshStream::STRIDE_3D);
    quadPositions->getData().assign({  1.0f,  1.0f, 0.0f,
                                       1.0f, -1.0f, 0.0f,
                                      -1.0f, -1.0f, 0.0f,
                                      -1.0f,  1.0f, 0.0f });
    MeshStreamPtr quadTexCoords = MeshStream::create(HW::IN_TEXCOORD + "_0", MeshStream::TEXCOORD_ATTRIBUTE, 0);
    quadTexCoords->setStride(MeshStream::STRIDE_2D);
    if (!flipTexCoordsHorizontally)
    {
        quadTexCoords->getData().assign({ uvMax[0], uvMax[1],
                                          uvMax[0], uvMin[1],
                                          uvMin[0], uvMin[1],
                                          uvMin[0], uvMax[1] });
    }
    else
    {
        quadTexCoords->getData().assign({ uvMax[0], uvMin[1],
                                          uvMax[0], uvMax[1],
                                          uvMin[0], uvMax[1],
                                          uvMin[0], uvMin[1] });
    }
    MeshPartitionPtr quadIndices = MeshPartition::create();
    quadIndices->getIndices().assign({ 0, 1, 3, 1, 2, 3 });
    quadIndices->setFaceCount(6);
    MeshPtr quadMesh = Mesh::create("ScreenSpaceQuad");
    quadMesh->addStream(quadPositions);
    quadMesh->addStream(quadTexCoords);
    quadMesh->addPartition(quadIndices);

    return quadMesh;
}

MATERIALX_NAMESPACE_END
