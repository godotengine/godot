//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <PyMaterialX/PyMaterialX.h>

#include <MaterialXCore/Types.h>

#include <MaterialXCore/Value.h>

#include <sstream>

namespace py = pybind11;
namespace mx = MaterialX;

using IndexPair = std::pair<size_t, size_t>;

#define BIND_VECTOR_SUBCLASS(V, N)                      \
.def(py::init<>())                                      \
.def(py::init<float>())                                 \
.def(py::init<const std::array<float, N>&>())           \
.def(py::init<const std::vector<float>&>())             \
.def(py::self == py::self)                              \
.def(py::self != py::self)                              \
.def(py::self + py::self)                               \
.def(py::self - py::self)                               \
.def(py::self * py::self)                               \
.def(py::self / py::self)                               \
.def(py::self * float())                                \
.def(py::self / float())                                \
.def("getMagnitude", &V::getMagnitude)                  \
.def("getNormalized", &V::getNormalized)                \
.def("dot", &V::dot)                                    \
.def("__getitem__", [](const V& v, size_t i)            \
    { return v[i]; } )                                  \
.def("__setitem__", [](V& v, size_t i, float f)         \
    { v[i] = f; } )                                     \
.def("__str__", [](const V& v)                          \
    { return mx::toValueString(v); })                   \
.def("copy", [](const V& v) { return V(v); })           \
.def_static("__len__", &V::numElements)

#define BIND_MATRIX_SUBCLASS(M, N)                      \
.def(py::init<>())                                      \
.def(py::init<float>())                                 \
.def(py::self == py::self)                              \
.def(py::self != py::self)                              \
.def(py::self + py::self)                               \
.def(py::self - py::self)                               \
.def(py::self * py::self)                               \
.def(py::self / py::self)                               \
.def(py::self * float())                                \
.def(py::self / float())                                \
.def("__getitem__", [](const M& m, IndexPair i)         \
    { return m[i.first][i.second]; } )                  \
.def("__setitem__", [](M& m, IndexPair i, float f)      \
    { m[i.first][i.second] = f; })                      \
.def("__str__", [](const M& m)                          \
    { return mx::toValueString(m); })                   \
.def("copy", [](const M& m) { return M(m); })           \
.def("isEquivalent", &M::isEquivalent)                  \
.def("getTranspose", &M::getTranspose)                  \
.def("getDeterminant", &M::getDeterminant)              \
.def("getAdjugate", &M::getAdjugate)                    \
.def("getInverse", &M::getInverse)                      \
.def_static("createScale", &M::createScale)             \
.def_static("createTranslation", &M::createTranslation) \
.def_static("numRows", &M::numRows)                     \
.def_static("numColumns", &M::numColumns)               \
.def_static("__len__", &M::numRows)

void bindPyTypes(py::module& mod)
{
    py::class_<mx::VectorBase>(mod, "VectorBase");
    py::class_<mx::MatrixBase>(mod, "MatrixBase");

    py::class_<mx::Vector2, mx::VectorBase>(mod, "Vector2")
        BIND_VECTOR_SUBCLASS(mx::Vector2, 2)
        .def(py::init<float, float>())
        .def("cross", &mx::Vector2::cross)
        .def("asTuple", [](const mx::Vector2& v) { return std::make_tuple(v[0], v[1]); });

    py::class_<mx::Vector3, mx::VectorBase>(mod, "Vector3")
        BIND_VECTOR_SUBCLASS(mx::Vector3, 3)
        .def(py::init<float, float, float>())
        .def("cross", &mx::Vector3::cross)
        .def("asTuple", [](const mx::Vector3& v) { return std::make_tuple(v[0], v[1], v[2]); });

    py::class_<mx::Vector4, mx::VectorBase>(mod, "Vector4")
        BIND_VECTOR_SUBCLASS(mx::Vector4, 4)
        .def(py::init<float, float, float, float>())
        .def("asTuple", [](const mx::Vector4& v) { return std::make_tuple(v[0], v[1], v[2], v[3]); });

    py::class_<mx::Color3, mx::VectorBase>(mod, "Color3")
        BIND_VECTOR_SUBCLASS(mx::Color3, 3)
        .def(py::init<float, float, float>())
        .def("linearToSrgb", &mx::Color3::linearToSrgb)
        .def("srgbToLinear", &mx::Color3::srgbToLinear)
        .def("asTuple", [](const mx::Color3& v) { return std::make_tuple(v[0], v[1], v[2]); });

    py::class_<mx::Color4, mx::VectorBase>(mod, "Color4")
        BIND_VECTOR_SUBCLASS(mx::Color4, 4)
        .def(py::init<float, float, float, float>())
        .def("asTuple", [](const mx::Color4& v) { return std::make_tuple(v[0], v[1], v[2], v[3]); });

    py::class_<mx::Matrix33, mx::MatrixBase>(mod, "Matrix33")
        BIND_MATRIX_SUBCLASS(mx::Matrix33, 3)
        .def(py::init<float, float, float,
                      float, float, float,
                      float, float, float>())
        .def("multiply", &mx::Matrix33::multiply)
        .def("transformPoint", &mx::Matrix33::transformPoint)
        .def("transformVector", &mx::Matrix33::transformVector)
        .def("transformNormal", &mx::Matrix33::transformNormal)
        .def_static("createRotation", &mx::Matrix33::createRotation)
        .def_readonly_static("IDENTITY", &mx::Matrix33::IDENTITY);

    py::class_<mx::Matrix44, mx::MatrixBase>(mod, "Matrix44")
        BIND_MATRIX_SUBCLASS(mx::Matrix44, 4)
        .def(py::init<float, float, float, float,
                      float, float, float, float,
                      float, float, float, float,
                      float, float, float, float>())
        .def("multiply", &mx::Matrix44::multiply)
        .def("transformPoint", &mx::Matrix44::transformPoint)
        .def("transformVector", &mx::Matrix44::transformVector)
        .def("transformNormal", &mx::Matrix44::transformNormal)
        .def_static("createRotationX", &mx::Matrix44::createRotationX)
        .def_static("createRotationY", &mx::Matrix44::createRotationY)
        .def_static("createRotationZ", &mx::Matrix44::createRotationZ)
        .def_readonly_static("IDENTITY", &mx::Matrix44::IDENTITY);

    mod.attr("DEFAULT_TYPE_STRING") = mx::DEFAULT_TYPE_STRING;
    mod.attr("FILENAME_TYPE_STRING") = mx::FILENAME_TYPE_STRING;
    mod.attr("GEOMNAME_TYPE_STRING") = mx::GEOMNAME_TYPE_STRING;
    mod.attr("STRING_TYPE_STRING") = mx::STRING_TYPE_STRING;
    mod.attr("BSDF_TYPE_STRING") = mx::BSDF_TYPE_STRING;
    mod.attr("EDF_TYPE_STRING") = mx::EDF_TYPE_STRING;
    mod.attr("VDF_TYPE_STRING") = mx::VDF_TYPE_STRING;
    mod.attr("SURFACE_SHADER_TYPE_STRING") = mx::SURFACE_SHADER_TYPE_STRING;
    mod.attr("DISPLACEMENT_SHADER_TYPE_STRING") = mx::DISPLACEMENT_SHADER_TYPE_STRING;
    mod.attr("VOLUME_SHADER_TYPE_STRING") = mx::VOLUME_SHADER_TYPE_STRING;
    mod.attr("LIGHT_SHADER_TYPE_STRING") = mx::LIGHT_SHADER_TYPE_STRING;
    mod.attr("MATERIAL_TYPE_STRING") = mx::MATERIAL_TYPE_STRING;
    mod.attr("SURFACE_MATERIAL_NODE_STRING") = mx::SURFACE_MATERIAL_NODE_STRING;
    mod.attr("VOLUME_MATERIAL_NODE_STRING") = mx::VOLUME_MATERIAL_NODE_STRING;
    mod.attr("MULTI_OUTPUT_TYPE_STRING") = mx::MULTI_OUTPUT_TYPE_STRING;
    mod.attr("NONE_TYPE_STRING") = mx::NONE_TYPE_STRING;
    mod.attr("VALUE_STRING_TRUE") = mx::VALUE_STRING_TRUE;
    mod.attr("VALUE_STRING_FALSE") = mx::VALUE_STRING_FALSE;
    mod.attr("NAME_PREFIX_SEPARATOR") = mx::NAME_PREFIX_SEPARATOR;
    mod.attr("NAME_PATH_SEPARATOR") = mx::NAME_PATH_SEPARATOR;
    mod.attr("ARRAY_VALID_SEPARATORS") = mx::ARRAY_VALID_SEPARATORS;
    mod.attr("ARRAY_PREFERRED_SEPARATOR") = mx::ARRAY_PREFERRED_SEPARATOR;
}
