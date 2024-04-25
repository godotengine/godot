//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <PyMaterialX/PyMaterialX.h>

#include <MaterialXRender/Mesh.h>

namespace py = pybind11;
namespace mx = MaterialX;

void bindPyMesh(py::module& mod)
{
    py::class_<mx::MeshStream, mx::MeshStreamPtr>(mod, "MeshStream")
        .def_readonly_static("POSITION_ATTRIBUTE", &mx::MeshStream::POSITION_ATTRIBUTE)
        .def_readonly_static("NORMAL_ATTRIBUTE", &mx::MeshStream::NORMAL_ATTRIBUTE)
        .def_readonly_static("TEXCOORD_ATTRIBUTE", &mx::MeshStream::TEXCOORD_ATTRIBUTE)
        .def_readonly_static("TANGENT_ATTRIBUTE", &mx::MeshStream::TANGENT_ATTRIBUTE)
        .def_readonly_static("BITANGENT_ATTRIBUTE", &mx::MeshStream::BITANGENT_ATTRIBUTE)
        .def_readonly_static("COLOR_ATTRIBUTE", &mx::MeshStream::COLOR_ATTRIBUTE)
        .def_readonly_static("GEOMETRY_PROPERTY_ATTRIBUTE", &mx::MeshStream::GEOMETRY_PROPERTY_ATTRIBUTE)
        .def_static("create", &mx::MeshStream::create)
        .def(py::init<const std::string&, const std::string&, unsigned int>())
        .def("reserve", &mx::MeshStream::reserve)
        .def("resize", &mx::MeshStream::resize)
        .def("getName", &mx::MeshStream::getName)
        .def("getType", &mx::MeshStream::getType)
        .def("getIndex", &mx::MeshStream::getIndex)
        .def("getData", static_cast<mx::MeshFloatBuffer& (mx::MeshStream::*)()>(&mx::MeshStream::getData), py::return_value_policy::reference)
        .def("getStride", &mx::MeshStream::getStride)
        .def("setStride", &mx::MeshStream::setStride)
        .def("getSize", &mx::MeshStream::getSize)
        .def("transform", &mx::MeshStream::transform);

    py::class_<mx::MeshPartition, mx::MeshPartitionPtr>(mod, "MeshPartition")
        .def_static("create", &mx::MeshPartition::create)
        .def(py::init<>())
        .def("resize", &mx::MeshPartition::resize)
        .def("setName", &mx::MeshPartition::setName)
        .def("getName", &mx::MeshPartition::getName)
        .def("addSourceName", &mx::MeshPartition::addSourceName)
        .def("getSourceNames", &mx::MeshPartition::getSourceNames)
        .def("getIndices", static_cast<mx::MeshIndexBuffer& (mx::MeshPartition::*)()>(&mx::MeshPartition::getIndices), py::return_value_policy::reference)
        .def("getFaceCount", &mx::MeshPartition::getFaceCount)
        .def("setFaceCount", &mx::MeshPartition::setFaceCount);

    py::class_<mx::Mesh, mx::MeshPtr>(mod, "Mesh")
        .def_static("create", &mx::Mesh::create)
        .def(py::init<const std::string&>())
        .def("getName", &mx::Mesh::getName)
        .def("setSourceUri", &mx::Mesh::setSourceUri)
        .def("hasSourceUri", &mx::Mesh::hasSourceUri)
        .def("getSourceUri", &mx::Mesh::getSourceUri)
        .def("getStream", static_cast<mx::MeshStreamPtr (mx::Mesh::*)(const std::string&) const>(&mx::Mesh::getStream))
        .def("getStream", static_cast<mx::MeshStreamPtr (mx::Mesh::*)(const std::string&, unsigned int) const> (&mx::Mesh::getStream))
        .def("addStream", &mx::Mesh::addStream)
        .def("setVertexCount", &mx::Mesh::setVertexCount)
        .def("getVertexCount", &mx::Mesh::getVertexCount)
        .def("setMinimumBounds", &mx::Mesh::setMinimumBounds)
        .def("getMinimumBounds", &mx::Mesh::getMinimumBounds)
        .def("setMaximumBounds", &mx::Mesh::setMaximumBounds)
        .def("getMaximumBounds", &mx::Mesh::getMaximumBounds)
        .def("setSphereCenter", &mx::Mesh::setSphereCenter)
        .def("getSphereCenter", &mx::Mesh::getSphereCenter)
        .def("setSphereRadius", &mx::Mesh::setSphereRadius)
        .def("getSphereRadius", &mx::Mesh::getSphereRadius)
        .def("getPartitionCount", &mx::Mesh::getPartitionCount)
        .def("addPartition", &mx::Mesh::addPartition)
        .def("getPartition", &mx::Mesh::getPartition)
        .def("generateTextureCoordinates", &mx::Mesh::generateTextureCoordinates)
        .def("generateNormals", &mx::Mesh::generateNormals)
        .def("generateTangents", &mx::Mesh::generateTangents)
        .def("generateBitangents", &mx::Mesh::generateBitangents)
        .def("mergePartitions", &mx::Mesh::mergePartitions)
        .def("splitByUdims", &mx::Mesh::splitByUdims);
}
