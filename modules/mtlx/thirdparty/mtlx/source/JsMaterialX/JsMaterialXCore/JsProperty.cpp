//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <JsMaterialX/VectorHelper.h>
#include <JsMaterialX/Helpers.h>

#include <MaterialXCore/Property.h>

#include <emscripten/bind.h>

namespace ems = emscripten;
namespace mx = MaterialX;

#define BIND_PROPERTYSET_TYPE_INSTANCE(NAME, T) \
    BIND_MEMBER_FUNC("setPropertyValue" #NAME, mx::PropertySet, setPropertyValue<T>, 2, 3, const std::string&, const T&, const std::string&)

EMSCRIPTEN_BINDINGS(property)
{

    ems::class_<mx::Property, ems::base<mx::ValueElement>>("Property")
        .smart_ptr_constructor("Property", &std::make_shared<mx::Property, mx::ElementPtr, const std::string &>)
        .smart_ptr<std::shared_ptr<const mx::Property>>("Property")
        .class_property("CATEGORY", &mx::Property::CATEGORY);

    ems::class_<mx::PropertyAssign, ems::base<mx::ValueElement>>("PropertyAssign")
        .smart_ptr_constructor("PropertyAssign", &std::make_shared<mx::PropertyAssign, mx::ElementPtr, const std::string &>)
        .smart_ptr<std::shared_ptr<const mx::PropertyAssign>>("PropertyAssign")
        .function("setProperty", &mx::PropertyAssign::setProperty)
        .function("hasProperty", &mx::PropertyAssign::hasProperty)
        .function("getProperty", &mx::PropertyAssign::getProperty)
        .function("setGeom", &mx::PropertyAssign::setGeom)
        .function("hasGeom", &mx::PropertyAssign::hasGeom)
        .function("getGeom", &mx::PropertyAssign::getGeom)
        .function("setCollectionString", &mx::PropertyAssign::setCollectionString)
        .function("hasCollectionString", &mx::PropertyAssign::hasCollectionString)
        .function("getCollectionString", &mx::PropertyAssign::getCollectionString)
        .function("setCollection", &mx::PropertyAssign::setCollection)
        .function("getCollection", &mx::PropertyAssign::getCollection)
        .class_property("CATEGORY", &mx::PropertyAssign::CATEGORY)
        .class_property("PROPERTY_ATTRIBUTE", &mx::PropertyAssign::PROPERTY_ATTRIBUTE)
        .class_property("GEOM_ATTRIBUTE", &mx::PropertyAssign::GEOM_ATTRIBUTE)
        .class_property("COLLECTION_ATTRIBUTE", &mx::PropertyAssign::COLLECTION_ATTRIBUTE);

    ems::class_<mx::PropertySet, ems::base<mx::Element>>("PropertySet")
        .smart_ptr_constructor("PropertySet", &std::make_shared<mx::PropertySet, mx::ElementPtr, const std::string &>)
        .smart_ptr<std::shared_ptr<const mx::PropertySet>>("PropertySet")
        .function("addProperty", &mx::PropertySet::addProperty)
        .function("getProperty", &mx::PropertySet::getProperty)
        .function("getProperties", &mx::PropertySet::getProperties)
        .function("removeProperty", &mx::PropertySet::removeProperty)
        BIND_PROPERTYSET_TYPE_INSTANCE(Integer, int)
        BIND_PROPERTYSET_TYPE_INSTANCE(Boolean, bool)
        BIND_PROPERTYSET_TYPE_INSTANCE(Float, float)
        BIND_PROPERTYSET_TYPE_INSTANCE(Color3, mx::Color3)
        BIND_PROPERTYSET_TYPE_INSTANCE(Color4, mx::Color4)
        BIND_PROPERTYSET_TYPE_INSTANCE(Vector2, mx::Vector2)
        BIND_PROPERTYSET_TYPE_INSTANCE(Vector3, mx::Vector3)
        BIND_PROPERTYSET_TYPE_INSTANCE(Vector4, mx::Vector4)
        BIND_PROPERTYSET_TYPE_INSTANCE(Matrix33, mx::Matrix33)
        BIND_PROPERTYSET_TYPE_INSTANCE(Matrix44, mx::Matrix44)
        BIND_PROPERTYSET_TYPE_INSTANCE(String, std::string)
        BIND_PROPERTYSET_TYPE_INSTANCE(IntegerArray, mx::IntVec)
        BIND_PROPERTYSET_TYPE_INSTANCE(BooleanArray, mx::BoolVec)
        BIND_PROPERTYSET_TYPE_INSTANCE(FloatArray, mx::FloatVec)
        BIND_PROPERTYSET_TYPE_INSTANCE(StringArray, mx::StringVec)
        .function("getPropertyValue", &mx::PropertySet::getPropertyValue)
        .class_property("CATEGORY", &mx::Property::CATEGORY);
        
    ems::class_<mx::PropertySetAssign, ems::base<mx::GeomElement>>("PropertySetAssign")
        .smart_ptr_constructor("PropertySetAssign", &std::make_shared<mx::PropertySetAssign, mx::ElementPtr, const std::string &>)
        .smart_ptr<std::shared_ptr<const mx::PropertySetAssign>>("PropertySetAssign")
        .function("setPropertySetString", &mx::PropertySetAssign::setPropertySetString)
        .function("hasPropertySetString", &mx::PropertySetAssign::hasPropertySetString)
        .function("getPropertySetString", &mx::PropertySetAssign::getPropertySetString)
        .function("setPropertySet", &mx::PropertySetAssign::setPropertySet)
        .function("getPropertySet", &mx::PropertySetAssign::getPropertySet)
        .class_property("CATEGORY", &mx::PropertySetAssign::CATEGORY)
        .class_property("PROPERTY_SET_ATTRIBUTE", &mx::PropertySetAssign::PROPERTY_SET_ATTRIBUTE);
}
