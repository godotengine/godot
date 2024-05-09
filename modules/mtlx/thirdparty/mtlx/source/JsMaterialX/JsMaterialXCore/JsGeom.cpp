//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <JsMaterialX/VectorHelper.h>
#include <JsMaterialX/Helpers.h>

#include <MaterialXCore/Geom.h>

#include <emscripten/bind.h>

namespace ems = emscripten;
namespace mx = MaterialX;

#define BIND_GEOMINFO_FUNC_INSTANCE(NAME, T) \
    BIND_MEMBER_FUNC("setGeomPropValue" #NAME, mx::GeomInfo, setGeomPropValue<T>, 2, 3, stRef, const T&, stRef)

EMSCRIPTEN_BINDINGS(geom)
{
    ems::constant("GEOM_PATH_SEPARATOR", mx::GEOM_PATH_SEPARATOR);
    ems::constant("UNIVERSAL_GEOM_NAME", mx::UNIVERSAL_GEOM_NAME);
    ems::constant("UDIM_TOKEN", mx::UDIM_TOKEN);
    ems::constant("UV_TILE_TOKEN", mx::UV_TILE_TOKEN);       
    ems::constant("UDIM_SET_PROPERTY", mx::UDIM_SET_PROPERTY);

    ems::class_<mx::GeomElement, ems::base<mx::Element>>("GeomElement")
        .smart_ptr<std::shared_ptr<mx::GeomElement>>("GeomElement")
        .smart_ptr<std::shared_ptr<const mx::GeomElement>>("GeomElement")
        .function("setGeom", &mx::GeomElement::setGeom)
        .function("hasGeom", &mx::GeomElement::hasGeom)
        .function("getGeom", &mx::GeomElement::getGeom)
        .function("getActiveGeom", &mx::GeomElement::getActiveGeom)
        .function("setCollectionString", &mx::GeomElement::setCollectionString)
        .function("hasCollectionString", &mx::GeomElement::hasCollectionString)
        .function("getCollectionString", &mx::GeomElement::getCollectionString)
        .function("setCollection", &mx::GeomElement::setCollection)
        .function("getCollection", &mx::GeomElement::getCollection)
        .class_property("GEOM_ATTRIBUTE", &mx::GeomElement::GEOM_ATTRIBUTE)
        .class_property("COLLECTION_ATTRIBUTE", &mx::GeomElement::COLLECTION_ATTRIBUTE);

    ems::class_<mx::GeomInfo, ems::base<mx::GeomElement>>("GeomInfo")
        .smart_ptr_constructor("GeomInfo", &std::make_shared<mx::GeomInfo, mx::ElementPtr, const std::string &>)
        .smart_ptr<std::shared_ptr<const mx::GeomInfo>>("GeomInfo")
        BIND_MEMBER_FUNC("addGeomProp", mx::GeomInfo, addGeomProp, 0, 1, stRef)
        .function("getGeomProp", &mx::GeomInfo::getGeomProp)
        .function("getGeomProps", &mx::GeomInfo::getGeomProps)
        .function("removeGeomProp", &mx::GeomInfo::removeGeomProp)
        .function("getToken", &mx::GeomInfo::getToken)
        .function("getTokens", &mx::GeomInfo::getTokens)
        .function("removeToken", &mx::GeomInfo::removeToken)
        BIND_GEOMINFO_FUNC_INSTANCE(Integer, int)
        BIND_GEOMINFO_FUNC_INSTANCE(Boolean, bool)
        BIND_GEOMINFO_FUNC_INSTANCE(Float, float)
        BIND_GEOMINFO_FUNC_INSTANCE(Color3, mx::Color3)
        BIND_GEOMINFO_FUNC_INSTANCE(Color4, mx::Color4)
        BIND_GEOMINFO_FUNC_INSTANCE(Vector2, mx::Vector2)
        BIND_GEOMINFO_FUNC_INSTANCE(Vector3, mx::Vector3)
        BIND_GEOMINFO_FUNC_INSTANCE(Vector4, mx::Vector4)
        BIND_GEOMINFO_FUNC_INSTANCE(Matrix33, mx::Matrix33)
        BIND_GEOMINFO_FUNC_INSTANCE(Matrix44, mx::Matrix44)
        BIND_GEOMINFO_FUNC_INSTANCE(String, std::string)
        BIND_GEOMINFO_FUNC_INSTANCE(IntegerArray, mx::IntVec)
        BIND_GEOMINFO_FUNC_INSTANCE(BooleanArray, mx::BoolVec)
        BIND_GEOMINFO_FUNC_INSTANCE(FloatArray, mx::FloatVec)
        BIND_GEOMINFO_FUNC_INSTANCE(StringArray, mx::StringVec)
        .function("setTokenValue", &mx::GeomInfo::setTokenValue)
        .class_property("CATEGORY", &mx::GeomInfo::CATEGORY);

    ems::class_<mx::GeomProp, ems::base<mx::ValueElement>>("GeomProp")
        .smart_ptr_constructor("GeomProp", &std::make_shared<mx::GeomProp, mx::ElementPtr, const std::string &>)
        .smart_ptr<std::shared_ptr<const mx::GeomProp>>("GeomProp")
        .class_property("CATEGORY", &mx::GeomProp::CATEGORY);

    ems::class_<mx::GeomPropDef, ems::base<mx::Element>>("GeomPropDef")
        .smart_ptr_constructor("GeomPropDef", &std::make_shared<mx::GeomPropDef, mx::ElementPtr, const std::string &>)
        .smart_ptr<std::shared_ptr<const mx::GeomPropDef>>("GeomPropDef")
        .function("setGeomProp", &mx::GeomPropDef::setGeomProp)
        .function("hasGeomProp", &mx::GeomPropDef::hasGeomProp)
        .function("getGeomProp", &mx::GeomPropDef::getGeomProp)
        .function("setSpace", &mx::GeomPropDef::setSpace)
        .function("hasSpace", &mx::GeomPropDef::hasSpace)
        .function("getSpace", &mx::GeomPropDef::getSpace)
        .function("setIndex", &mx::GeomPropDef::setIndex)
        .function("hasIndex", &mx::GeomPropDef::hasIndex)
        .function("getIndex", &mx::GeomPropDef::getIndex)
        .class_property("CATEGORY", &mx::GeomPropDef::CATEGORY)
        .class_property("GEOM_PROP_ATTRIBUTE", &mx::GeomPropDef::GEOM_PROP_ATTRIBUTE)
        .class_property("SPACE_ATTRIBUTE", &mx::GeomPropDef::SPACE_ATTRIBUTE)
        .class_property("INDEX_ATTRIBUTE", &mx::GeomPropDef::INDEX_ATTRIBUTE);

    ems::class_<mx::Collection, ems::base<mx::Element>>("Collection")
        .smart_ptr_constructor("Collection", &std::make_shared<mx::Collection, mx::ElementPtr, const std::string &>)
        .smart_ptr<std::shared_ptr<const mx::Collection>>("Collection")
        .function("setIncludeGeom", &mx::Collection::setIncludeGeom)
        .function("hasIncludeGeom", &mx::Collection::hasIncludeGeom)
        .function("getIncludeGeom", &mx::Collection::getIncludeGeom)
        .function("getActiveIncludeGeom", &mx::Collection::getActiveIncludeGeom)
        .function("setExcludeGeom", &mx::Collection::setExcludeGeom)
        .function("hasExcludeGeom", &mx::Collection::hasExcludeGeom)
        .function("getExcludeGeom", &mx::Collection::getExcludeGeom)
        .function("getActiveExcludeGeom", &mx::Collection::getActiveExcludeGeom)
        .function("setIncludeCollectionString", &mx::Collection::setIncludeCollectionString)
        .function("hasIncludeCollectionString", &mx::Collection::hasIncludeCollectionString)
        .function("getIncludeCollectionString", &mx::Collection::getIncludeCollectionString)
        .function("setIncludeCollection", &mx::Collection::setIncludeCollection)
        .function("setIncludeCollections", &mx::Collection::setIncludeCollections)
        .function("getIncludeCollections", &mx::Collection::getIncludeCollections)
        .function("hasIncludeCycle", &mx::Collection::hasIncludeCycle)
        .function("matchesGeomString", &mx::Collection::matchesGeomString)
        .class_property("CATEGORY", &mx::Collection::CATEGORY)
        .class_property("INCLUDE_GEOM_ATTRIBUTE", &mx::Collection::INCLUDE_GEOM_ATTRIBUTE)
        .class_property("EXCLUDE_GEOM_ATTRIBUTE", &mx::Collection::EXCLUDE_GEOM_ATTRIBUTE)
        .class_property("INCLUDE_COLLECTION_ATTRIBUTE", &mx::Collection::INCLUDE_COLLECTION_ATTRIBUTE);

    ems::function("geomStringsMatch", &mx::geomStringsMatch); 
}
