//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <JsMaterialX/VectorHelper.h>
#include <JsMaterialX/Helpers.h>

#include <MaterialXCore/Types.h>
#include <MaterialXCore/Value.h>

#include <emscripten/bind.h>

namespace ems = emscripten;
namespace mx = MaterialX;

#define BIND_VECTOR_SUBCLASS(V)                                                                           \
    .function("equals", ems::optional_override([](V &self, const V &rhs) { return self == rhs; }))        \
    .function("notEquals", ems::optional_override([](V &self, const V &rhs) { return self != rhs; }))     \
    .function("add", ems::optional_override([](V &self, const V &rhs) { return self + rhs; }))            \
    .function("sub", ems::optional_override([](V &self, const V &rhs) { return self - rhs; }))            \
    .function("multiply", ems::optional_override([](V &self, const V &rhs) { return self * rhs; }))       \
    .function("multiplyScalar", ems::optional_override([](V &self, float s) { return self * s; }))        \
    .function("divide", ems::optional_override([](V &self, const V &rhs) { return self / rhs; }))         \
    .function("divideScalar", ems::optional_override([](V &self, float s) { return self / s; }))          \
    .function("negate", ems::optional_override([](V &self) { return -self; }))                            \
    .function("getMagnitude", ems::optional_override([](V &self) { return self.V::getMagnitude(); }))     \
    .function("getNormalized", ems::optional_override([](V &self) { return self.V::getNormalized(); }))   \
    .function("dot", ems::optional_override([](V &self, const V &rhs) { return self.V::dot(rhs); }))      \
    .function("getItem", ems::optional_override([](V &self, size_t i) { return self[i]; }))               \
    .function("setItem", ems::optional_override([](V &self, size_t i, float f) { self[i] = f; }))         \
    .function("toString", ems::optional_override([](const V &self) { return toValueString(self); }))      \
    .function("copy", ems::optional_override([](const V &self) { return V(self); }))                      \
    .function("length", ems::optional_override([](const V &self) { return self.V::numElements(); }))      \
    .function("data", ems::optional_override([](V& self) { return ems::val(ems::typed_memory_view(self.numElements(), self.data())); }))

#define BIND_MATRIX_SUBCLASS(M)                                                                                                                          \
    .function("equals", ems::optional_override([](M &self, const M &rhs) { return self == rhs; }))                                                       \
    .function("notEquals", ems::optional_override([](M &self, const M &rhs) { return self != rhs; }))                                                   \
    .function("add", ems::optional_override([](M &self, const M &rhs) { return self + rhs; }))                                                           \
    .function("sub", ems::optional_override([](M &self, const M &rhs) { return self - rhs; }))                                                           \
    .function("multiply", ems::optional_override([](M &self, const M &rhs) { return self * rhs; }))                                                      \
    .function("multiplyScalar", ems::optional_override([](M &self, float s) { return self * s; }))                                                       \
    .function("divide", ems::optional_override([](M &self, const M &rhs) { return self / rhs; }))                                                        \
    .function("divideScalar", ems::optional_override([](M &self, float s) { return self / s; }))                                                         \
    .function("getItem", ems::optional_override([](M &self, size_t row, size_t col) { return self[row][col]; }))                                         \
    .function("setItem", ems::optional_override([](M &self, size_t row, size_t col, float f) { self[row][col] = f; }))                                   \
    .function("toString", ems::optional_override([](const M &self) { return toValueString(self); }))                                                     \
    .function("copy", ems::optional_override([](const M &self) { return M(self); }))                                                                     \
    .function("isEquivalent", ems::optional_override([](const M &self, const M &rhs, float tolerance) { return self.M::isEquivalent(rhs, tolerance); })) \
    .function("getTranspose", ems::optional_override([](const M &self) { return self.M::getTranspose(); }))                                              \
    .function("getDeterminant", ems::optional_override([](const M &self) { return self.M::getDeterminant(); }))                                          \
    .function("getAdjugate", ems::optional_override([](const M &self) { return self.M::getAdjugate(); }))                                                \
    .function("getInverse", ems::optional_override([](const M &self) { return self.M::getInverse(); }))                                                  \
    .function("numRows", ems::optional_override([](const M &self) { return self.M::numRows(); }))                                                        \
    .function("numColumns", ems::optional_override([](const M &self) { return self.M::numColumns(); }))                                                  \
    .function("length", ems::optional_override([](const M &self) { return self.M::numRows(); }))

EMSCRIPTEN_BINDINGS(types)
{
    ems::class_<mx::VectorBase>("VectorBase");
    ems::class_<mx::MatrixBase>("MatrixBase");

    ems::class_<mx::Vector2, ems::base<mx::VectorBase>>("Vector2")
        .constructor<>()
        .constructor<float, float>()
            BIND_VECTOR_SUBCLASS(mx::Vector2)
        .function("cross", &mx::Vector2::cross);

    ems::class_<mx::Vector3, ems::base<mx::VectorBase>>("Vector3")
        .constructor<>()
        .constructor<float, float, float>()
            BIND_VECTOR_SUBCLASS(mx::Vector3)
        .function("cross", &mx::Vector3::cross);

    ems::class_<mx::Vector4, ems::base<mx::VectorBase>>("Vector4")
        .constructor<>()
        .constructor<float, float, float, float>()
            BIND_VECTOR_SUBCLASS(mx::Vector4);                

    ems::class_<mx::Color3, ems::base<mx::VectorBase>>("Color3")
        .constructor<>()
        .constructor<float, float, float>()
            BIND_VECTOR_SUBCLASS(mx::Color3);

    ems::class_<mx::Color4, ems::base<mx::VectorBase>>("Color4")
        .constructor<>()
        .constructor<float, float, float, float>()
            BIND_VECTOR_SUBCLASS(mx::Color4);

    ems::class_<mx::Matrix33, ems::base<mx::MatrixBase>>("Matrix33")
        .constructor<>()
        .constructor<float, float, float, float, float, float, float, float, float>()
            BIND_MATRIX_SUBCLASS(mx::Matrix33)
        .function("multiplyVector3", ems::optional_override([](const mx::Matrix33 &self, const mx::Vector3 &v) { return self.mx::Matrix33::multiply(v); }))
        .function("transformPoint", ems::optional_override([](const mx::Matrix33 &self, const mx::Vector2 &v) { return self.mx::Matrix33::transformPoint(v); }))
        .function("transformVector", ems::optional_override([](const mx::Matrix33 &self, const mx::Vector2 &v) { return self.mx::Matrix33::transformVector(v); }))
        .function("transformNormal", ems::optional_override([](const mx::Matrix33 &self, const mx::Vector3 &v) { return self.mx::Matrix33::transformNormal(v); }))
        .class_function("createTranslation", &mx::Matrix33::createTranslation)
        .class_function("createScale", &mx::Matrix33::createScale)
        .class_function("createRotation", &mx::Matrix33::createRotation)
        .class_property("IDENTITY", &mx::Matrix33::IDENTITY);

    ems::class_<mx::Matrix44, ems::base<mx::MatrixBase>>("Matrix44")
        .constructor<>()
        .constructor<float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float>()
            BIND_MATRIX_SUBCLASS(mx::Matrix44)
        .function("multiplyVector4", ems::optional_override([](const mx::Matrix44 &self, const mx::Vector4 &v) { return self.mx::Matrix44::multiply(v); }))                
        .function("transformPoint", ems::optional_override([](const mx::Matrix44 &self, const mx::Vector3 &v) { return self.mx::Matrix44::transformPoint(v); }))
        .function("transformVector", ems::optional_override([](const mx::Matrix44 &self, const mx::Vector3 &v) { return self.mx::Matrix44::transformVector(v); }))
        .function("transformNormal", ems::optional_override([](const mx::Matrix44 &self, const mx::Vector3 &v) { return self.mx::Matrix44::transformNormal(v); }))
        .class_function("createTranslation", &mx::Matrix44::createTranslation)
        .class_function("createScale", &mx::Matrix44::createScale)
        .class_function("createRotationX", &mx::Matrix44::createRotationX)
        .class_function("createRotationY", &mx::Matrix44::createRotationY)
        .class_function("createRotationZ", &mx::Matrix44::createRotationZ)
        .class_property("IDENTITY", &mx::Matrix44::IDENTITY);

    ems::constant("DEFAULT_TYPE_STRING", mx::DEFAULT_TYPE_STRING);
    ems::constant("FILENAME_TYPE_STRING", mx::FILENAME_TYPE_STRING);
    ems::constant("GEOMNAME_TYPE_STRING", mx::GEOMNAME_TYPE_STRING);
    ems::constant("STRING_TYPE_STRING", mx::STRING_TYPE_STRING);
    ems::constant("SURFACE_SHADER_TYPE_STRING", mx::SURFACE_SHADER_TYPE_STRING);
    ems::constant("DISPLACEMENT_SHADER_TYPE_STRING", mx::DISPLACEMENT_SHADER_TYPE_STRING);
    ems::constant("VOLUME_SHADER_TYPE_STRING", mx::VOLUME_SHADER_TYPE_STRING);
    ems::constant("LIGHT_SHADER_TYPE_STRING", mx::LIGHT_SHADER_TYPE_STRING);
    ems::constant("MATERIAL_TYPE_STRING", mx::MATERIAL_TYPE_STRING);
    ems::constant("SURFACE_MATERIAL_NODE_STRING", mx::SURFACE_MATERIAL_NODE_STRING);
    ems::constant("VOLUME_MATERIAL_NODE_STRING", mx::VOLUME_MATERIAL_NODE_STRING);
    ems::constant("MULTI_OUTPUT_TYPE_STRING", mx::MULTI_OUTPUT_TYPE_STRING);
    ems::constant("NONE_TYPE_STRING", mx::NONE_TYPE_STRING);
    ems::constant("VALUE_STRING_TRUE", mx::VALUE_STRING_TRUE);
    ems::constant("VALUE_STRING_FALSE", mx::VALUE_STRING_FALSE);
    ems::constant("NAME_PREFIX_SEPARATOR", mx::NAME_PREFIX_SEPARATOR);
    ems::constant("NAME_PATH_SEPARATOR", mx::NAME_PATH_SEPARATOR);
    ems::constant("ARRAY_VALID_SEPARATORS", mx::ARRAY_VALID_SEPARATORS);
    ems::constant("ARRAY_PREFERRED_SEPARATOR", mx::ARRAY_PREFERRED_SEPARATOR);
}
