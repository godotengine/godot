//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXRender/Mesh.h>

#include <limits>
#include <map>

MATERIALX_NAMESPACE_BEGIN

const string MeshStream::POSITION_ATTRIBUTE("position");
const string MeshStream::NORMAL_ATTRIBUTE("normal");
const string MeshStream::TEXCOORD_ATTRIBUTE("texcoord");
const string MeshStream::TANGENT_ATTRIBUTE("tangent");
const string MeshStream::BITANGENT_ATTRIBUTE("bitangent");
const string MeshStream::COLOR_ATTRIBUTE("color");
const string MeshStream::GEOMETRY_PROPERTY_ATTRIBUTE("geomprop");

namespace
{

const float MAX_FLOAT = std::numeric_limits<float>::max();
const size_t FACE_VERTEX_COUNT = 3;

} // anonymous namespace

//
// Mesh methods
//

Mesh::Mesh(const string& name) :
    _name(name),
    _minimumBounds(MAX_FLOAT, MAX_FLOAT, MAX_FLOAT),
    _maximumBounds(-MAX_FLOAT, -MAX_FLOAT, -MAX_FLOAT),
    _sphereCenter(0.0f, 0.0f, 0.0f),
    _sphereRadius(0.0f),
    _vertexCount(0)
{
}

MeshStreamPtr Mesh::generateNormals(MeshStreamPtr positionStream)
{
    // Create the normal stream.
    MeshStreamPtr normalStream = MeshStream::create("i_" + MeshStream::NORMAL_ATTRIBUTE, MeshStream::NORMAL_ATTRIBUTE, 0);
    normalStream->resize(positionStream->getSize());

    // Iterate through partitions.
    for (size_t i = 0; i < getPartitionCount(); i++)
    {
        MeshPartitionPtr part = getPartition(i);

        // Iterate through faces.
        for (size_t faceIndex = 0; faceIndex < part->getFaceCount(); faceIndex++)
        {
            uint32_t i0 = part->getIndices()[faceIndex * FACE_VERTEX_COUNT + 0];
            uint32_t i1 = part->getIndices()[faceIndex * FACE_VERTEX_COUNT + 1];
            uint32_t i2 = part->getIndices()[faceIndex * FACE_VERTEX_COUNT + 2];

            const Vector3& p0 = positionStream->getElement<Vector3>(i0);
            const Vector3& p1 = positionStream->getElement<Vector3>(i1);
            const Vector3& p2 = positionStream->getElement<Vector3>(i2);

            Vector3& n0 = normalStream->getElement<Vector3>(i0);
            Vector3& n1 = normalStream->getElement<Vector3>(i1);
            Vector3& n2 = normalStream->getElement<Vector3>(i2);

            Vector3 faceNormal = (p1 - p0).cross(p2 - p0).getNormalized();
            n0 = faceNormal;
            n1 = faceNormal;
            n2 = faceNormal;
        }
    }

    return normalStream;
}

MeshStreamPtr Mesh::generateTextureCoordinates(MeshStreamPtr positionStream)
{
    size_t vertexCount = positionStream->getData().size() / MeshStream::STRIDE_3D;
    MeshStreamPtr texcoordStream = MeshStream::create("i_" + MeshStream::TEXCOORD_ATTRIBUTE + "_0", MeshStream::TEXCOORD_ATTRIBUTE, 0);
    texcoordStream->setStride(MeshStream::STRIDE_2D);
    texcoordStream->resize(vertexCount);
    std::fill(texcoordStream->getData().begin(), texcoordStream->getData().end(), 0.0f);

    return texcoordStream;
}

MeshStreamPtr Mesh::generateTangents(MeshStreamPtr positionStream, MeshStreamPtr normalStream, MeshStreamPtr texcoordStream)
{
    size_t vertexCount = positionStream->getData().size() / positionStream->getStride();
    size_t normalCount = normalStream->getData().size() / normalStream->getStride();
    size_t texcoordCount = texcoordStream->getData().size() / texcoordStream->getStride();
    if (vertexCount != normalCount || vertexCount != texcoordCount)
    {
        return nullptr;
    }

    // Create the tangent stream.
    MeshStreamPtr tangentStream = MeshStream::create("i_" + MeshStream::TANGENT_ATTRIBUTE, MeshStream::TANGENT_ATTRIBUTE, 0);
    tangentStream->resize(positionStream->getSize());
    std::fill(tangentStream->getData().begin(), tangentStream->getData().end(), 0.0f);

    // Iterate through partitions.
    for (size_t i = 0; i < getPartitionCount(); i++)
    {
        MeshPartitionPtr part = getPartition(i);

        // Iterate through faces.
        for (size_t faceIndex = 0; faceIndex < part->getFaceCount(); faceIndex++)
        {
            uint32_t i0 = part->getIndices()[faceIndex * FACE_VERTEX_COUNT + 0];
            uint32_t i1 = part->getIndices()[faceIndex * FACE_VERTEX_COUNT + 1];
            uint32_t i2 = part->getIndices()[faceIndex * FACE_VERTEX_COUNT + 2];

            const Vector3& p0 = positionStream->getElement<Vector3>(i0);
            const Vector3& p1 = positionStream->getElement<Vector3>(i1);
            const Vector3& p2 = positionStream->getElement<Vector3>(i2);

            const Vector2& w0 = texcoordStream->getElement<Vector2>(i0);
            const Vector2& w1 = texcoordStream->getElement<Vector2>(i1);
            const Vector2& w2 = texcoordStream->getElement<Vector2>(i2);

            Vector3& t0 = tangentStream->getElement<Vector3>(i0);
            Vector3& t1 = tangentStream->getElement<Vector3>(i1);
            Vector3& t2 = tangentStream->getElement<Vector3>(i2);

            // Based on Eric Lengyel at http://www.terathon.com/code/tangent.html

            Vector3 e1 = p1 - p0;
            Vector3 e2 = p2 - p0;

            float x1 = w1[0] - w0[0];
            float x2 = w2[0] - w0[0];
            float y1 = w1[1] - w0[1];
            float y2 = w2[1] - w0[1];

            float denom = x1 * y2 - x2 * y1;
            float r = denom ? (1.0f / denom) : 0.0f;
            Vector3 t = (e1 * y2 - e2 * y1) * r;

            t0 += t;
            t1 += t;
            t2 += t;
        }
    }

    // Iterate through vertices.
    for (size_t v = 0; v < vertexCount; v++)
    {
        Vector3& n = normalStream->getElement<Vector3>(v);
        Vector3& t = tangentStream->getElement<Vector3>(v);

        if (t != Vector3(0.0f))
        {
            // Gram-Schmidt orthogonalize.
            t = (t - n * n.dot(t)).getNormalized();
        }
        else
        {
            // Generate an arbitrary tangent.
            // https://graphics.pixar.com/library/OrthonormalB/paper.pdf
            float sign = (n[2] < 0.0f) ? -1.0f : 1.0f;
            float a = -1.0f / (sign + n[2]);
            float b = n[0] * n[1] * a;
            t = Vector3(1.0f + sign * n[0] * n[0] * a, sign * b, -sign * n[0]);
        }
    }

    return tangentStream;
}

MeshStreamPtr Mesh::generateBitangents(MeshStreamPtr normalStream, MeshStreamPtr tangentStream)
{
    if (normalStream->getSize() != tangentStream->getSize())
    {
        return nullptr;
    }

    MeshStreamPtr bitangentStream = MeshStream::create("i_" + MeshStream::BITANGENT_ATTRIBUTE, MeshStream::BITANGENT_ATTRIBUTE, 0);
    bitangentStream->resize(normalStream->getSize());

    for (size_t i = 0; i < normalStream->getSize(); i++)
    {
        const Vector3& normal = normalStream->getElement<Vector3>(i);
        const Vector3& tangent = tangentStream->getElement<Vector3>(i);

        Vector3& bitangent = bitangentStream->getElement<Vector3>(i);
        bitangent = normal.cross(tangent);
    }

    return bitangentStream;
}

void Mesh::mergePartitions()
{
    if (getPartitionCount() <= 1)
    {
        return;
    }

    MeshPartitionPtr merged = MeshPartition::create();
    merged->setName("merged");
    for (size_t p = 0; p < getPartitionCount(); p++)
    {
        MeshPartitionPtr part = getPartition(p);
        merged->getIndices().insert(merged->getIndices().end(),
                                    part->getIndices().begin(),
                                    part->getIndices().end());
        merged->setFaceCount(merged->getFaceCount() + part->getFaceCount());
        merged->addSourceName(part->getName());
    }

    _partitions.clear();
    addPartition(merged);
}

void Mesh::splitByUdims()
{
    MeshStreamPtr texcoords = getStream(MeshStream::TEXCOORD_ATTRIBUTE, 0);
    if (!texcoords)
    {
        return;
    }

    std::map<uint32_t, MeshPartitionPtr> udimMap;
    for (size_t p = 0; p < getPartitionCount(); p++)
    {
        MeshPartitionPtr part = getPartition(p);
        for (size_t f = 0; f < part->getFaceCount(); f++)
        {
            uint32_t i0 = part->getIndices()[f * FACE_VERTEX_COUNT + 0];
            uint32_t i1 = part->getIndices()[f * FACE_VERTEX_COUNT + 1];
            uint32_t i2 = part->getIndices()[f * FACE_VERTEX_COUNT + 2];

            const Vector2& uv0 = texcoords->getElement<Vector2>(i0);
            uint32_t udimU = (uint32_t) uv0[0];
            uint32_t udimV = (uint32_t) uv0[1];
            uint32_t udim = 1001 + udimU + (10 * udimV);
            if (!udimMap.count(udim))
            {
                udimMap[udim] = MeshPartition::create();
                udimMap[udim]->setName(std::to_string(udim));
            }

            MeshPartitionPtr udimPart = udimMap[udim];
            udimPart->getIndices().push_back(i0);
            udimPart->getIndices().push_back(i1);
            udimPart->getIndices().push_back(i2);
            udimPart->setFaceCount(udimPart->getFaceCount() + 1);
            udimPart->addSourceName(part->getName());
        }
    }

    if (udimMap.size() >= 2)
    {
        _partitions.clear();
        for (const auto& pair : udimMap)
        {
            addPartition(pair.second);
        }
    }
}

//
// MeshStream methods
//

void MeshStream::transform(const Matrix44& matrix)
{
    unsigned int stride = getStride();
    size_t numElements = _data.size() / getStride();
    if (getType() == MeshStream::POSITION_ATTRIBUTE ||
        getType() == MeshStream::TEXCOORD_ATTRIBUTE ||
        getType() == MeshStream::GEOMETRY_PROPERTY_ATTRIBUTE)
    {
        for (size_t i = 0; i < numElements; i++)
        {
            Vector4 vec(0.0, 0.0, 0.0, 1.0);
            for (size_t j = 0; j < stride; j++)
            {
                vec[j] = _data[i * stride + j];
            }
            vec = matrix.multiply(vec);
            for (size_t k = 0; k < stride; k++)
            {
                _data[i * stride + k] = vec[k];
            }
        }
    }
    else if (getType() == MeshStream::NORMAL_ATTRIBUTE ||
             getType() == MeshStream::TANGENT_ATTRIBUTE ||
             getType() == MeshStream::BITANGENT_ATTRIBUTE)
    {
        bool isNormalStream = (getType() == MeshStream::NORMAL_ATTRIBUTE);
        Matrix44 transformMatrix = isNormalStream ? matrix.getInverse().getTranspose() : matrix;

        for (size_t i = 0; i < numElements; i++)
        {
            Vector3 vec(0.0, 0.0, 0.0);
            for (size_t j = 0; j < stride; j++)
            {
                vec[j] = _data[i * stride + j];
            }
            vec = transformMatrix.transformVector(vec).getNormalized();
            for (size_t k = 0; k < stride; k++)
            {
                _data[i * stride + k] = vec[k];
            }
        }
    }
}

MATERIALX_NAMESPACE_END
