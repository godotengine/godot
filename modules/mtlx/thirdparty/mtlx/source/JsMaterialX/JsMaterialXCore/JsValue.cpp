//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <JsMaterialX/VectorHelper.h>
#include <JsMaterialX/Helpers.h>

#include <MaterialXCore/Value.h>

#include <emscripten/bind.h>

namespace ems = emscripten;
namespace mx = MaterialX;

#define BIND_TYPE_INSTANCE(NAME, T)                                           \
    ems::class_<mx::TypedValue<T>, ems::base<mx::Value>>("TypedValue" #NAME) \
        .smart_ptr<std::shared_ptr<mx::TypedValue<T>>>("TypedValue<T>")       \
        .function("copy", &mx::TypedValue<T>::copy)                     \
        .function("setData", ems::select_overload<void(const T& value)>(&mx::TypedValue<T>::setData)) \
        .function("setDataTypedValue", ems::select_overload<void(const mx::TypedValue<T>& value)>(&mx::TypedValue<T>::setData)) \
        .function("getData", &mx::TypedValue<T>::getData)                     \
        .function("getTypeString", &mx::TypedValue<T>::getTypeString)         \
        .function("getValueString", &mx::TypedValue<T>::getValueString)       \
        .class_function("createFromString", &mx::TypedValue<T>::createFromString);

#define BIND_TYPE_SPECIFIC_VALUE_FUNCS(NAME, T)                           \
        .class_function("createValue" #NAME, &mx::Value::createValue<T>)  \
        .function("isA" #NAME, &mx::Value::isA<T>)                        \
        .function("asA" #NAME, &mx::Value::asA<T>)

#define BIND_GLOBAL_FUNCS(NAME, T)   \
        ems::function("getTypeString", &mx::getTypeString<T>); \
        ems::function("toValueString", &mx::toValueString<T>); \
        ems::function("fromValueString", &mx::fromValueString<T>);

EMSCRIPTEN_BINDINGS(value)
{
    ems::enum_<mx::Value::FloatFormat>("FloatFormat")
        .value("FloatFormatDefault", mx::Value::FloatFormat::FloatFormatDefault)
        .value("FloatFormatFixed", mx::Value::FloatFormat::FloatFormatFixed)
        .value("FloatFormatScientific", mx::Value::FloatFormat::FloatFormatScientific);

    ems::class_<mx::Value>("Value")
        .smart_ptr<std::shared_ptr<mx::Value>>("Value")
        .smart_ptr<std::shared_ptr<const mx::Value>>("Value")
        .function("copy", &mx::Value::copy, ems::pure_virtual()) 
        .function("getTypeString", &mx::Value::getTypeString)
        .function("getValueString", &mx::Value::getValueString)
        .class_function("createValueFromStrings", &mx::Value::createValueFromStrings)
        .class_function("setFloatFormat", &mx::Value::setFloatFormat)
        .class_function("setFloatPrecision", &mx::Value::setFloatPrecision)
        .class_function("getFloatFormat", &mx::Value::getFloatFormat)
        .class_function("getFloatPrecision", &mx::Value::getFloatPrecision)
        BIND_TYPE_SPECIFIC_VALUE_FUNCS(Integer, int)
        BIND_TYPE_SPECIFIC_VALUE_FUNCS(Boolean, bool)
        BIND_TYPE_SPECIFIC_VALUE_FUNCS(Float, float)
        BIND_TYPE_SPECIFIC_VALUE_FUNCS(Color3, mx::Color3)
        BIND_TYPE_SPECIFIC_VALUE_FUNCS(Color4, mx::Color4)
        BIND_TYPE_SPECIFIC_VALUE_FUNCS(Vector2, mx::Vector2)
        BIND_TYPE_SPECIFIC_VALUE_FUNCS(Vector3, mx::Vector3)
        BIND_TYPE_SPECIFIC_VALUE_FUNCS(Vector4, mx::Vector4)
        BIND_TYPE_SPECIFIC_VALUE_FUNCS(Matrix33, mx::Matrix33)
        BIND_TYPE_SPECIFIC_VALUE_FUNCS(Matrix44, mx::Matrix44)
        BIND_TYPE_SPECIFIC_VALUE_FUNCS(String, std::string)
        BIND_TYPE_SPECIFIC_VALUE_FUNCS(IntegerArray, mx::IntVec)
        BIND_TYPE_SPECIFIC_VALUE_FUNCS(BooleanArray, mx::BoolVec)
        BIND_TYPE_SPECIFIC_VALUE_FUNCS(FloatArray, mx::FloatVec)
        BIND_TYPE_SPECIFIC_VALUE_FUNCS(StringArray, mx::StringVec);

    BIND_TYPE_INSTANCE(Integer, int)
    BIND_TYPE_INSTANCE(Boolean, bool)
    BIND_TYPE_INSTANCE(Float, float)
    BIND_TYPE_INSTANCE(Color3, mx::Color3)
    BIND_TYPE_INSTANCE(Color4, mx::Color4)
    BIND_TYPE_INSTANCE(Vector2, mx::Vector2)
    BIND_TYPE_INSTANCE(Vector3, mx::Vector3)
    BIND_TYPE_INSTANCE(Vector4, mx::Vector4)
    BIND_TYPE_INSTANCE(Matrix33, mx::Matrix33)
    BIND_TYPE_INSTANCE(Matrix44, mx::Matrix44)
    BIND_TYPE_INSTANCE(String, std::string)
    BIND_TYPE_INSTANCE(IntegerArray, mx::IntVec)
    BIND_TYPE_INSTANCE(BooleanArray, mx::BoolVec)
    BIND_TYPE_INSTANCE(FloatArray, mx::FloatVec)
    BIND_TYPE_INSTANCE(StringArray, mx::StringVec)

    BIND_GLOBAL_FUNCS(Integer, int)
}
