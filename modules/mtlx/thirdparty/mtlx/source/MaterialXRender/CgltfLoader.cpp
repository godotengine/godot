//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXRender/CgltfLoader.h>

#if defined(__GNUC__)
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wswitch"
#endif

#if defined(_MSC_VER)
    #pragma warning(push)
    #pragma warning(disable : 4996)
#endif

#define CGLTF_IMPLEMENTATION
#include <MaterialXRender/External/Cgltf/cgltf.h>
#undef CGLTF_IMPLEMENTATION

#if defined(_MSC_VER)
    #pragma warning(pop)
#endif

#if defined(__GNUC__)
    #pragma GCC diagnostic pop
#endif

#include <cstring>
#include <iostream>
#include <limits>

MATERIALX_NAMESPACE_BEGIN

namespace
{

const float MAX_FLOAT = std::numeric_limits<float>::max();
const size_t FACE_VERTEX_COUNT = 3;

// List of transforms which match to meshes
using GLTFMeshMatrixList = std::unordered_map<cgltf_mesh*, std::vector<Matrix44>>;

// Compute matrices for each mesh. Appends a transform for each transform instance
void computeMeshMatrices(GLTFMeshMatrixList& meshMatrices, cgltf_node* cnode)
{
    cgltf_mesh* cmesh = cnode->mesh;
    if (cmesh)
    {
        float t[16];
        cgltf_node_transform_world(cnode, t);
        Matrix44 positionMatrix = Matrix44(
            (float) t[0], (float) t[1], (float) t[2], (float) t[3],
            (float) t[4], (float) t[5], (float) t[6], (float) t[7],
            (float) t[8], (float) t[9], (float) t[10], (float) t[11],
            (float) t[12], (float) t[13], (float) t[14], (float) t[15]);
        meshMatrices[cmesh].push_back(positionMatrix);
    }

    // Iterate over all children. Note that the existence of a mesh
    // does not imply that this is a leaf node so traversal should
    // continue even when a mesh is encountered.
    for (cgltf_size i = 0; i < cnode->children_count; i++)
    {
        computeMeshMatrices(meshMatrices, cnode->children[i]);
    }
}

const std::string DEFAULT_NODE_PREFIX = "NODE_";
const std::string DEFAULT_MESH_PREFIX = "MESH_";

// List of path names which match to meshes
using GLTFMeshPathList = std::unordered_map<cgltf_mesh*, StringVec>;

void computeMeshPaths(GLTFMeshPathList& meshPaths, cgltf_node* cnode, FilePath path, size_t nodeCount, size_t meshCount)
{
    string cnodeName = cnode->name ? string(cnode->name) : DEFAULT_NODE_PREFIX + std::to_string(nodeCount++);
    path = path / (createValidName(cnodeName) + "/");

    cgltf_mesh* cmesh = cnode->mesh;
    if (cmesh)
    {
        // Set path to mesh if no transform path found
        if (path.isEmpty())
        {
            string meshName = cmesh->name ? string(cmesh->name) : DEFAULT_MESH_PREFIX + std::to_string(meshCount++);
            path = createValidName(meshName);
        }

        meshPaths[cmesh].push_back(path.asString(FilePath::FormatPosix));
    }

    // Iterate over all children. Note that the existence of a mesh
    // does not imply that this is a leaf node so traversal should
    // continue even when a mesh is encountered.
    for (cgltf_size i = 0; i < cnode->children_count; i++)
    {
        computeMeshPaths(meshPaths, cnode->children[i], path, nodeCount, meshCount);
    }
}

void decodeVec4Tangents(MeshStreamPtr vec4TangentStream, MeshStreamPtr normalStream, MeshStreamPtr& tangentStream, MeshStreamPtr& bitangentStream)
{
    if (vec4TangentStream->getSize() != normalStream->getSize())
    {
        return;
    }

    tangentStream = MeshStream::create("i_" + MeshStream::TANGENT_ATTRIBUTE, MeshStream::TANGENT_ATTRIBUTE, 0);
    bitangentStream = MeshStream::create("i_" + MeshStream::BITANGENT_ATTRIBUTE, MeshStream::BITANGENT_ATTRIBUTE, 0);

    tangentStream->resize(vec4TangentStream->getSize());
    bitangentStream->resize(vec4TangentStream->getSize());

    for (size_t i = 0; i < vec4TangentStream->getSize(); i++)
    {
        const Vector4& vec4Tangent = vec4TangentStream->getElement<Vector4>(i);
        const Vector3& normal = normalStream->getElement<Vector3>(i);

        Vector3& tangent = tangentStream->getElement<Vector3>(i);
        Vector3& bitangent = bitangentStream->getElement<Vector3>(i);

        tangent = Vector3(vec4Tangent[0], vec4Tangent[1], vec4Tangent[2]);
        bitangent = normal.cross(tangent) * vec4Tangent[3];
    }
}

} // anonymous namespace

bool CgltfLoader::load(const FilePath& filePath, MeshList& meshList, bool texcoordVerticalFlip)
{
    const string input_filename = filePath.asString();
    const string ext = stringToLower(filePath.getExtension());
    const string BINARY_EXTENSION = "glb";
    const string ASCII_EXTENSION = "gltf";
    if (ext != BINARY_EXTENSION && ext != ASCII_EXTENSION)
    {
        return false;
    }

    cgltf_options options;
    std::memset(&options, 0, sizeof(options));
    cgltf_data* data = nullptr;

    // Read file
    cgltf_result result = cgltf_parse_file(&options, input_filename.c_str(), &data);
    if (result != cgltf_result_success)
    {
        return false;
    }
    if (cgltf_load_buffers(&options, data, input_filename.c_str()) != cgltf_result_success)
    {
        return false;
    }

    // Precompute mesh / matrix associations starting from the root
    // of the scene.
    GLTFMeshMatrixList gltfMeshMatrixList;
    for (cgltf_size sceneIndex = 0; sceneIndex < data->scenes_count; ++sceneIndex)
    {
        cgltf_scene* scene = &data->scenes[sceneIndex];
        for (cgltf_size nodeIndex = 0; nodeIndex < scene->nodes_count; ++nodeIndex)
        {
            cgltf_node* cnode = scene->nodes[nodeIndex];
            if (!cnode)
            {
                continue;
            }
            computeMeshMatrices(gltfMeshMatrixList, cnode);
        }
    }

    GLTFMeshPathList gltfMeshPathList;
    unsigned int nodeCount = 0;
    unsigned int meshCount = 0;
    FilePath path;
    for (cgltf_size sceneIndex = 0; sceneIndex < data->scenes_count; ++sceneIndex)
    {
        cgltf_scene* scene = &data->scenes[sceneIndex];
        for (cgltf_size nodeIndex = 0; nodeIndex < scene->nodes_count; ++nodeIndex)
        {
            cgltf_node* cnode = scene->nodes[nodeIndex];
            if (!cnode)
            {
                continue;
            }
            computeMeshPaths(gltfMeshPathList, cnode, path, nodeCount, meshCount);
        }
    }

    // Read in all meshes
    StringSet meshNames;
    for (size_t m = 0; m < data->meshes_count; m++)
    {
        cgltf_mesh* cmesh = &(data->meshes[m]);
        if (!cmesh)
        {
            continue;
        }
        std::vector<Matrix44> positionMatrices;
        if (gltfMeshMatrixList.find(cmesh) != gltfMeshMatrixList.end())
        {
            positionMatrices = gltfMeshMatrixList[cmesh];
        }
        if (positionMatrices.empty())
        {
            positionMatrices.push_back(Matrix44::IDENTITY);
        }

        StringVec paths;
        if (gltfMeshPathList.find(cmesh) != gltfMeshPathList.end())
        {
            paths = gltfMeshPathList[cmesh];
        }
        if (paths.empty())
        {
            string meshName = cmesh->name ? string(cmesh->name) : DEFAULT_MESH_PREFIX + std::to_string(meshCount++);
            paths.push_back(meshName);
        }

        // Iterate through all parent transform
        for (size_t mtx = 0; mtx < positionMatrices.size(); mtx++)
        {
            const Matrix44& positionMatrix = positionMatrices[mtx];
            const Matrix44 normalMatrix = positionMatrix.getInverse().getTranspose();

            for (cgltf_size primitiveIndex = 0; primitiveIndex < cmesh->primitives_count; ++primitiveIndex)
            {
                cgltf_primitive* primitive = &cmesh->primitives[primitiveIndex];
                if (!primitive)
                {
                    continue;
                }

                if (primitive->type != cgltf_primitive_type_triangles)
                {
                    if (_debugLevel > 0)
                    {
                        std::cout << "Skip non-triangle indexed mesh: " << cmesh->name << std::endl;
                    }
                    continue;
                }

                Vector3 boxMin = { MAX_FLOAT, MAX_FLOAT, MAX_FLOAT };
                Vector3 boxMax = { -MAX_FLOAT, -MAX_FLOAT, -MAX_FLOAT };

                // Create a unique path for the mesh.
                string meshName = paths[mtx];
                while (meshNames.count(meshName))
                {
                    meshName = incrementName(meshName);
                }
                meshNames.insert(meshName);

                MeshPtr mesh = Mesh::create(meshName);
                if (_debugLevel > 0)
                {
                    std::cout << "Translate mesh: " << meshName << std::endl;
                }
                meshList.push_back(mesh);
                mesh->setSourceUri(filePath);

                MeshStreamPtr positionStream = nullptr;
                MeshStreamPtr normalStream = nullptr;
                MeshStreamPtr colorStream = nullptr;
                MeshStreamPtr texcoordStream = nullptr;
                MeshStreamPtr vec4TangentStream = nullptr;
                int colorAttrIndex = 0;

                // Read in vertex streams
                for (cgltf_size prim = 0; prim < primitive->attributes_count; prim++)
                {
                    cgltf_attribute* attribute = &primitive->attributes[prim];
                    cgltf_accessor* accessor = attribute->data;
                    if (!accessor)
                    {
                        continue;
                    }
                    // Only load one stream of each type for now.
                    cgltf_int streamIndex = attribute->index;
                    if (streamIndex != 0)
                    {
                        continue;
                    }

                    // Get data as floats
                    cgltf_size floatCount = cgltf_accessor_unpack_floats(accessor, NULL, 0);
                    std::vector<float> attributeData;
                    attributeData.resize(floatCount);
                    floatCount = cgltf_accessor_unpack_floats(accessor, &attributeData[0], floatCount);

                    cgltf_size vectorSize = cgltf_num_components(accessor->type);
                    size_t desiredVectorSize = 3;

                    MeshStreamPtr geomStream = nullptr;

                    bool isPositionStream = (attribute->type == cgltf_attribute_type_position);
                    bool isNormalStream = (attribute->type == cgltf_attribute_type_normal);
                    bool isTangentStream = (attribute->type == cgltf_attribute_type_tangent);
                    bool isColorStream = (attribute->type == cgltf_attribute_type_color);
                    bool isTexCoordStream = (attribute->type == cgltf_attribute_type_texcoord);

                    if (isPositionStream)
                    {
                        // Create position stream
                        positionStream = MeshStream::create("i_" + MeshStream::POSITION_ATTRIBUTE, MeshStream::POSITION_ATTRIBUTE, streamIndex);
                        mesh->addStream(positionStream);
                        geomStream = positionStream;
                    }
                    else if (isNormalStream)
                    {
                        normalStream = MeshStream::create("i_" + MeshStream::NORMAL_ATTRIBUTE, MeshStream::NORMAL_ATTRIBUTE, streamIndex);
                        mesh->addStream(normalStream);
                        geomStream = normalStream;
                    }
                    else if (isTangentStream)
                    {
                        vec4TangentStream = MeshStream::create("i_" + MeshStream::TANGENT_ATTRIBUTE + "4", MeshStream::TANGENT_ATTRIBUTE, streamIndex);
                        vec4TangentStream->setStride(MeshStream::STRIDE_4D); // glTF stores the bitangent sign in the 4th component
                        geomStream = vec4TangentStream;
                        desiredVectorSize = 4;
                    }
                    else if (isColorStream)
                    {
                        colorStream = MeshStream::create("i_" + MeshStream::COLOR_ATTRIBUTE + "_" + std::to_string(colorAttrIndex), MeshStream::COLOR_ATTRIBUTE, streamIndex);
                        mesh->addStream(colorStream);
                        geomStream = colorStream;
                        if (vectorSize == 4)
                        {
                            colorStream->setStride(MeshStream::STRIDE_4D);
                            desiredVectorSize = 4;
                        }
                        colorAttrIndex++;
                    }
                    else if (isTexCoordStream)
                    {
                        texcoordStream = MeshStream::create("i_" + MeshStream::TEXCOORD_ATTRIBUTE + "_0", MeshStream::TEXCOORD_ATTRIBUTE, 0);
                        mesh->addStream(texcoordStream);
                        if (vectorSize == 2)
                        {
                            texcoordStream->setStride(MeshStream::STRIDE_2D);
                            desiredVectorSize = 2;
                        }
                        geomStream = texcoordStream;
                    }
                    else
                    {
                        if (_debugLevel > 0)
                            std::cout << "Unknown stream type: " << std::to_string(attribute->type) << std::endl;
                    }

                    // Fill in stream
                    if (geomStream)
                    {
                        MeshFloatBuffer& buffer = geomStream->getData();
                        cgltf_size vertexCount = accessor->count;
                        geomStream->reserve(vertexCount);

                        if (_debugLevel > 0)
                        {
                            std::cout << "** Read stream: " << geomStream->getName() << std::endl;
                            std::cout << " - vertex count: " << std::to_string(vertexCount) << std::endl;
                            std::cout << " - vector size: " << std::to_string(vectorSize) << std::endl;
                        }

                        for (cgltf_size i = 0; i < vertexCount; i++)
                        {
                            const float* input = &attributeData[vectorSize * i];
                            if (isPositionStream)
                            {
                                Vector3 position;
                                for (cgltf_size v = 0; v < desiredVectorSize; v++)
                                {
                                    // Update bounding box
                                    float floatValue = (v < vectorSize) ? input[v] : 0.0f;
                                    position[v] = floatValue;
                                }
                                position = positionMatrix.transformPoint(position);
                                for (cgltf_size v = 0; v < desiredVectorSize; v++)
                                {
                                    buffer.push_back(position[v]);
                                    boxMin[v] = std::min(position[v], boxMin[v]);
                                    boxMax[v] = std::max(position[v], boxMax[v]);
                                }
                            }
                            else if (isNormalStream)
                            {
                                Vector3 normal;
                                for (cgltf_size v = 0; v < desiredVectorSize; v++)
                                {
                                    float floatValue = (v < vectorSize) ? input[v] : 0.0f;
                                    normal[v] = floatValue;
                                }
                                normal = normalMatrix.transformVector(normal).getNormalized();
                                for (cgltf_size v = 0; v < desiredVectorSize; v++)
                                {
                                    buffer.push_back(normal[v]);
                                }
                            }
                            else
                            {
                                for (cgltf_size v = 0; v < desiredVectorSize; v++)
                                {
                                    float floatValue = (v < vectorSize) ? input[v] : 0.0f;
                                    // Perform v-flip
                                    if (isTexCoordStream && v == 1)
                                    {
                                        if (!texcoordVerticalFlip)
                                        {
                                            floatValue = 1.0f - floatValue;
                                        }
                                    }
                                    buffer.push_back(floatValue);
                                }
                            }
                        }
                    }
                }

                if (!positionStream)
                {
                    continue;
                }

                // Read indexing
                MeshPartitionPtr part = MeshPartition::create();
                size_t indexCount = 0;
                cgltf_accessor* indexAccessor = primitive->indices;
                if (indexAccessor)
                {
                    indexCount = indexAccessor->count;
                }
                else
                {
                    indexCount = positionStream->getData().size();
                }
                size_t faceCount = indexCount / FACE_VERTEX_COUNT;
                part->setFaceCount(faceCount);
                part->setName(meshName);

                MeshIndexBuffer& indices = part->getIndices();
                if (_debugLevel > 0)
                {
                    std::cout << "** Read indexing: Count = " << std::to_string(indexCount) << std::endl;
                }
                if (indexAccessor)
                {
                    for (cgltf_size i = 0; i < indexCount; i++)
                    {
                        uint32_t vertexIndex = static_cast<uint32_t>(cgltf_accessor_read_index(indexAccessor, i));
                        indices.push_back(vertexIndex);
                    }
                }
                else
                {
                    for (cgltf_size i = 0; i < indexCount; i++)
                    {
                        indices.push_back(static_cast<uint32_t>(i));
                    }
                }
                mesh->addPartition(part);

                // Update positional information.
                mesh->setVertexCount(positionStream->getData().size() / MeshStream::STRIDE_3D);
                mesh->setMinimumBounds(boxMin);
                mesh->setMaximumBounds(boxMax);
                Vector3 sphereCenter = (boxMax + boxMin) * 0.5;
                mesh->setSphereCenter(sphereCenter);
                mesh->setSphereRadius((sphereCenter - boxMin).getMagnitude());

                // According to glTF spec. 3.7.2.1, tangents must be ignored when normals are missing
                if (vec4TangentStream && normalStream)
                {
                    // Decode glTF vec4 tangents to MaterialX vec3 tangents and bitangents
                    MeshStreamPtr tangentStream;
                    MeshStreamPtr bitangentStream;
                    decodeVec4Tangents(vec4TangentStream, normalStream, tangentStream, bitangentStream);

                    if (tangentStream)
                    {
                        mesh->addStream(tangentStream);
                    }
                    if (bitangentStream)
                    {
                        mesh->addStream(bitangentStream);
                    }
                }

                // Generate tangents, normals and texture coordinates if none are provided
                if (!normalStream)
                {
                    normalStream = mesh->generateNormals(positionStream);
                    mesh->addStream(normalStream);
                }
                if (!texcoordStream)
                {
                    texcoordStream = mesh->generateTextureCoordinates(positionStream);
                    mesh->addStream(texcoordStream);
                }
                if (!vec4TangentStream)
                {
                    MeshStreamPtr tangentStream = mesh->generateTangents(positionStream, normalStream, texcoordStream);
                    if (tangentStream)
                    {
                        mesh->addStream(tangentStream);
                    }
                    MeshStreamPtr bitangentStream = mesh->generateBitangents(normalStream, tangentStream);
                    if (bitangentStream)
                    {
                        mesh->addStream(bitangentStream);
                    }
                }
            }
        }
    }

    cgltf_free(data);

    return true;
}

MATERIALX_NAMESPACE_END
