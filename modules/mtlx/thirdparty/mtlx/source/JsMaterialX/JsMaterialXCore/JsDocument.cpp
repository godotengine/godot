//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <JsMaterialX/VectorHelper.h>
#include <JsMaterialX/Helpers.h>

#include <MaterialXCore/Document.h>

#include <emscripten/bind.h>

namespace ems = emscripten;
namespace mx = MaterialX;

EMSCRIPTEN_BINDINGS(document)
{
    ems::function("createDocument", &mx::createDocument);
    ems::class_<mx::Document, ems::base<mx::GraphElement>>("Document")
        .smart_ptr_constructor("Document", &std::make_shared<mx::Document, mx::ElementPtr, const std::string &>)
        .smart_ptr<std::shared_ptr<const mx::Document>>("Document")
            // At the moment only the Document type is used. Once more types are added this binding needs to be adapted accordingly.
        .class_function("createDocument", &mx::Document::createDocument<mx::Document>)
        .function("initialize", &mx::Document::initialize)
        .function("copy", &mx::Document::copy)
        .function("importLibrary", &mx::Document::importLibrary)
        .function("getReferencedSourceUris", ems::optional_override([](mx::Document &self) {
            mx::StringSet set = self.getReferencedSourceUris();
            return ems::val::array(set.begin(), set.end());
        }))
        BIND_MEMBER_FUNC("addNodeGraph", mx::Document, addNodeGraph, 0, 1, stRef)
        .function("getNodeGraph", &mx::Document::getNodeGraph)
        .function("getNodeGraphs", &mx::Document::getNodeGraphs)
        .function("removeNodeGraph", &mx::Document::removeNodeGraph)
        .function("getMatchingPorts", &mx::Document::getMatchingPorts)
        BIND_MEMBER_FUNC("addGeomInfo", mx::Document, addGeomInfo, 0, 2, stRef, stRef)
        .function("getGeomInfo", &mx::Document::getGeomInfo)
        .function("getGeomInfos", &mx::Document::getGeomInfos)
        .function("removeGeomInfo", &mx::Document::removeGeomInfo)
        BIND_MEMBER_FUNC("getGeomPropValue", mx::Document, getGeomPropValue, 1, 2, stRef, stRef)
        .function("addGeomPropDef", &mx::Document::addGeomPropDef)
        .function("getGeomPropDef", &mx::Document::getGeomPropDef)
        .function("getGeomPropDefs", &mx::Document::getGeomPropDefs)
        .function("removeGeomPropDef", &mx::Document::removeGeomPropDef)
        BIND_MEMBER_FUNC("addLook", mx::Document, addLook, 0, 1, stRef)
        .function("getLook", &mx::Document::getLook)
        .function("getLooks", &mx::Document::getLooks)
        .function("removeLook", &mx::Document::removeLook)
        BIND_MEMBER_FUNC("addLookGroup", mx::Document, addLookGroup, 0, 1, stRef)
        .function("getLookGroup", &mx::Document::getLookGroup)
        .function("getLookGroups", &mx::Document::getLookGroups)
        .function("removeLookGroup", &mx::Document::removeLookGroup)
        BIND_MEMBER_FUNC("addCollection", mx::Document, addCollection, 0, 1, stRef)
        .function("getCollection", &mx::Document::getCollection)
        .function("getCollections", &mx::Document::getCollections)
        .function("removeCollection", &mx::Document::removeCollection)
        .function("addTypeDef", &mx::Document::addTypeDef)
        .function("getTypeDef", &mx::Document::getTypeDef)
        .function("getTypeDefs", &mx::Document::getTypeDefs)
        .function("removeTypeDef", &mx::Document::removeTypeDef)
        BIND_MEMBER_FUNC("addNodeDef", mx::Document, addNodeDef, 0, 3, stRef, stRef, stRef)
        BIND_MEMBER_FUNC("addNodeDefFromGraph", mx::Document, addNodeDefFromGraph, 7, 7, const mx::NodeGraphPtr, 
            stRef, stRef, stRef, bool, stRef, std::string)
        .function("getNodeDef", &mx::Document::getNodeDef)
        .function("getNodeDefs", &mx::Document::getNodeDefs)
        .function("removeNodeDef", &mx::Document::removeNodeDef)
        .function("getMatchingNodeDefs", &mx::Document::getMatchingNodeDefs)
        BIND_MEMBER_FUNC("addAttributeDef", mx::Document, addAttributeDef, 0, 1, stRef)
        .function("getAttributeDef", &mx::Document::getAttributeDef)
        .function("getAttributeDefs", &mx::Document::getAttributeDefs)
        .function("removeAttributeDef", &mx::Document::removeAttributeDef)
        BIND_MEMBER_FUNC("addTargetDef", mx::Document, addTargetDef, 0, 1, stRef)
        .function("getTargetDef", &mx::Document::getTargetDef)
        .function("getTargetDefs", &mx::Document::getTargetDefs)
        .function("removeTargetDef", &mx::Document::removeTargetDef)
        BIND_MEMBER_FUNC("addPropertySet", mx::Document, addPropertySet, 0, 1, stRef)
        .function("getPropertySet", &mx::Document::getPropertySet)
        .function("getPropertySets", &mx::Document::getPropertySets)
        .function("removePropertySet", &mx::Document::removePropertySet)
        BIND_MEMBER_FUNC("addVariantSet", mx::Document, addVariantSet, 0, 1, stRef)
        .function("getVariantSet", &mx::Document::getVariantSet)
        .function("getVariantSets", &mx::Document::getVariantSets)
        .function("removeVariantSet", &mx::Document::removeVariantSet)
        BIND_MEMBER_FUNC("addImplementation", mx::Document, addImplementation, 0, 1, stRef)
        .function("getImplementation", &mx::Document::getImplementation)
        .function("getImplementations", &mx::Document::getImplementations)
        .function("removeImplementation", &mx::Document::removeImplementation)
        .function("getMatchingImplementations", &mx::Document::getMatchingImplementations)
        .function("addUnitDef", &mx::Document::addUnitDef)
        .function("getUnitDef", &mx::Document::getUnitDef)
        .function("getUnitDefs", &mx::Document::getUnitDefs)
        .function("removeUnitDef", &mx::Document::removeUnitDef)
        .function("addUnitTypeDef", &mx::Document::addUnitTypeDef)
        .function("getUnitTypeDef", &mx::Document::getUnitTypeDef)
        .function("getUnitTypeDefs", &mx::Document::getUnitTypeDefs)
        .function("removeUnitTypeDef", &mx::Document::removeUnitTypeDef)
        .function("getVersionIntegers", &mx::Document::getVersionIntegers)
        .function("upgradeVersion", &mx::Document::upgradeVersion)
        .function("setColorManagementSystem", &mx::Document::setColorManagementSystem)
        .function("hasColorManagementSystem", &mx::Document::hasColorManagementSystem)
        .function("getColorManagementSystem", &mx::Document::getColorManagementSystem)
        .function("setColorManagementConfig", &mx::Document::setColorManagementConfig)
        .function("hasColorManagementConfig", &mx::Document::hasColorManagementConfig)
        .function("getColorManagementConfig", &mx::Document::getColorManagementConfig)
        .function("invalidateCache", &mx::Document::invalidateCache)
        .class_property("CATEGORY", &mx::Document::CATEGORY)
        .class_property("CMS_ATTRIBUTE", &mx::Document::CMS_ATTRIBUTE)
        .class_property("CMS_CONFIG_ATTRIBUTE", &mx::Document::CMS_CONFIG_ATTRIBUTE);
}
