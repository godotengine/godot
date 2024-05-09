//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <JsMaterialX/VectorHelper.h>
#include <JsMaterialX/Helpers.h>

#include <MaterialXCore/Look.h>

#include <emscripten/bind.h>

namespace ems = emscripten;
namespace mx = MaterialX;

EMSCRIPTEN_BINDINGS(look)
{
    ems::class_<mx::Look, ems::base<mx::Element>>("Look")
        .smart_ptr_constructor("Look", &std::make_shared<mx::Look, mx::ElementPtr, const std::string &>)
        .smart_ptr<std::shared_ptr<const mx::Look>>("Look")
        BIND_MEMBER_FUNC("addMaterialAssign", mx::Look, addMaterialAssign, 0, 2, stRef, stRef)
        .function("getMaterialAssign", &mx::Look::getMaterialAssign)
        .function("getMaterialAssigns", &mx::Look::getMaterialAssigns)
        .function("getActiveMaterialAssigns", &mx::Look::getActiveMaterialAssigns)
        .function("removeMaterialAssign", &mx::Look::removeMaterialAssign)
        BIND_MEMBER_FUNC("addPropertyAssign", mx::Look, addPropertyAssign, 0, 1, stRef) 
        .function("getPropertyAssign", &mx::Look::getPropertyAssign)
        .function("getPropertyAssigns", &mx::Look::getPropertyAssigns)
        .function("getActivePropertyAssigns", &mx::Look::getActivePropertyAssigns)
        .function("removePropertyAssign", &mx::Look::removePropertyAssign)
        BIND_MEMBER_FUNC("addPropertySetAssign", mx::Look, addPropertySetAssign, 0, 1, stRef)
        .function("getPropertySetAssign", &mx::Look::getPropertySetAssign)
        .function("getPropertySetAssigns", &mx::Look::getPropertySetAssigns)
        .function("getActivePropertySetAssigns", &mx::Look::getActivePropertySetAssigns)
        .function("removePropertySetAssign", &mx::Look::removePropertySetAssign)
        BIND_MEMBER_FUNC("addVariantAssign", mx::Look, addVariantAssign, 0, 1, stRef)
        .function("getVariantAssign", &mx::Look::getVariantAssign)
        .function("getVariantAssigns", &mx::Look::getVariantAssigns)
        .function("getActiveVariantAssigns", &mx::Look::getActiveVariantAssigns)
        .function("removeVariantAssign", &mx::Look::removeVariantAssign)
        BIND_MEMBER_FUNC("addVisibility", mx::Look, addVisibility, 0, 1, stRef)
        .function("getVisibility", &mx::Look::getVisibility)
        .function("getVisibilities", &mx::Look::getVisibilities)
        .function("getActiveVisibilities", &mx::Look::getActiveVisibilities)
        .function("removeVisibility", &mx::Look::removeVisibility)
        .class_property("CATEGORY", &mx::Look::CATEGORY);

    ems::class_<mx::LookGroup, ems::base<mx::Element>>("LookGroup")
        .smart_ptr_constructor("LookGroup", &std::make_shared<mx::LookGroup, mx::ElementPtr, const std::string &>)
        .smart_ptr<std::shared_ptr<const mx::LookGroup>>("LookGroup")
        .function("setLooks", &mx::LookGroup::setLooks)
        .function("getLooks", &mx::LookGroup::getLooks)
        .function("setActiveLook", &mx::LookGroup::setActiveLook)
        .function("getActiveLook", &mx::LookGroup::getActiveLook)
        .class_property("CATEGORY", &mx::LookGroup::CATEGORY)
        .class_property("LOOKS_ATTRIBUTE", &mx::LookGroup::LOOKS_ATTRIBUTE)
        .class_property("ACTIVE_ATTRIBUTE", &mx::LookGroup::ACTIVE_ATTRIBUTE);

    ems::class_<mx::MaterialAssign, ems::base<mx::GeomElement>>("MaterialAssign")
        .smart_ptr_constructor("MaterialAssign", &std::make_shared<mx::MaterialAssign, mx::ElementPtr, const std::string &>)
        .smart_ptr<std::shared_ptr<const mx::MaterialAssign>>("MaterialAssign")
        .function("setMaterial", &mx::MaterialAssign::setMaterial)
        .function("hasMaterial", &mx::MaterialAssign::hasMaterial)
        .function("getMaterial", &mx::MaterialAssign::getMaterial)
        .function("setExclusive", &mx::MaterialAssign::setExclusive)
        .function("getExclusive", &mx::MaterialAssign::getExclusive)
        .function("getReferencedMaterial", &mx::MaterialAssign::getReferencedMaterial)
        BIND_MEMBER_FUNC("addVariantAssign", mx::MaterialAssign, addVariantAssign, 0, 1, stRef)
        .function("getVariantAssign", &mx::MaterialAssign::getVariantAssign)
        .function("getVariantAssigns", &mx::MaterialAssign::getVariantAssigns)
        .function("getActiveVariantAssigns", &mx::MaterialAssign::getActiveVariantAssigns)
        .function("removeVariantAssign", &mx::MaterialAssign::removeVariantAssign)
        .class_property("CATEGORY", &mx::MaterialAssign::CATEGORY)
        .class_property("MATERIAL_ATTRIBUTE", &mx::MaterialAssign::MATERIAL_ATTRIBUTE)
        .class_property("EXCLUSIVE_ATTRIBUTE", &mx::MaterialAssign::EXCLUSIVE_ATTRIBUTE);

    ems::class_<mx::Visibility, ems::base<mx::GeomElement>>("Visibility")
        .smart_ptr_constructor("Visibility", &std::make_shared<mx::Visibility, mx::ElementPtr, const std::string &>)
        .smart_ptr<std::shared_ptr<const mx::Visibility>>("Visibility")
        .function("setViewerGeom", &mx::Visibility::setViewerGeom)
        .function("hasViewerGeom", &mx::Visibility::hasViewerGeom)
        .function("getViewerGeom", &mx::Visibility::getViewerGeom)
        .function("setViewerCollection", &mx::Visibility::setViewerCollection)
        .function("hasViewerCollection", &mx::Visibility::hasViewerCollection)
        .function("getViewerCollection", &mx::Visibility::getViewerCollection)
        .function("setVisibilityType", &mx::Visibility::setVisibilityType)
        .function("hasVisibilityType", &mx::Visibility::hasVisibilityType)
        .function("getVisibilityType", &mx::Visibility::getVisibilityType)
        .function("setVisible", &mx::Visibility::setVisible)
        .function("getVisible", &mx::Visibility::getVisible)
        .class_property("CATEGORY", &mx::Visibility::CATEGORY)
        .class_property("VIEWER_GEOM_ATTRIBUTE", &mx::Visibility::VIEWER_GEOM_ATTRIBUTE)
        .class_property("VIEWER_COLLECTION_ATTRIBUTE", &mx::Visibility::VIEWER_COLLECTION_ATTRIBUTE)
        .class_property("VISIBILITY_TYPE_ATTRIBUTE", &mx::Visibility::VISIBILITY_TYPE_ATTRIBUTE)
        .class_property("VISIBLE_ATTRIBUTE", &mx::Visibility::VISIBLE_ATTRIBUTE);

    ems::function("getGeometryBindings", &mx::getGeometryBindings);
}
