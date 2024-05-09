//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <JsMaterialX/VectorHelper.h>
#include <JsMaterialX/Helpers.h>

#include <MaterialXCore/Definition.h>

#include <emscripten/bind.h>

namespace ems = emscripten;
namespace mx = MaterialX;

#define BIND_ATTRIBUTE_DEF_FUNC_INSTANCE(NAME, T)          \
    BIND_MEMBER_FUNC("setValue" #NAME, mx::AttributeDef, setValue<T>, 1, 2, const T&, stRef)

EMSCRIPTEN_BINDINGS(definition)
{
    ems::class_<mx::NodeDef, ems::base<mx::InterfaceElement>>("NodeDef")
        .smart_ptr_constructor("NodeDef", &std::make_shared<mx::NodeDef, mx::ElementPtr, const std::string &>)
        .smart_ptr<std::shared_ptr<const mx::NodeDef>>("NodeDef")
        .function("setNodeString", &mx::NodeDef::setNodeString)
        .function("hasNodeString", &mx::NodeDef::hasNodeString)
        .function("getNodeString", &mx::NodeDef::getNodeString)
        .function("getType", &mx::NodeDef::getType)
        .function("setNodeGroup", &mx::NodeDef::setNodeGroup)
        .function("hasNodeGroup", &mx::NodeDef::hasNodeGroup)
        .function("getNodeGroup", &mx::NodeDef::getNodeGroup)
        BIND_MEMBER_FUNC("getImplementation", mx::NodeDef, getImplementation, 0, 1, stRef)
        .function("isVersionCompatible", &mx::NodeDef::isVersionCompatible)
        .class_property("CATEGORY", &mx::NodeDef::CATEGORY)
        .class_property("NODE_ATTRIBUTE", &mx::NodeDef::NODE_ATTRIBUTE)
        .class_property("NODE_GROUP_ATTRIBUTE", &mx::NodeDef::NODE_GROUP_ATTRIBUTE)
        .class_property("TEXTURE_NODE_GROUP", &mx::NodeDef::TEXTURE_NODE_GROUP)
        .class_property("PROCEDURAL_NODE_GROUP", &mx::NodeDef::PROCEDURAL_NODE_GROUP)
        .class_property("GEOMETRIC_NODE_GROUP", &mx::NodeDef::GEOMETRIC_NODE_GROUP)
        .class_property("ADJUSTMENT_NODE_GROUP", &mx::NodeDef::ADJUSTMENT_NODE_GROUP)
        .class_property("CONDITIONAL_NODE_GROUP", &mx::NodeDef::CONDITIONAL_NODE_GROUP)
        .class_property("ORGANIZATION_NODE_GROUP", &mx::NodeDef::ORGANIZATION_NODE_GROUP)
        .class_property("TRANSLATION_NODE_GROUP", &mx::NodeDef::TRANSLATION_NODE_GROUP);

    ems::class_<mx::Implementation, ems::base<mx::InterfaceElement>>("Implementation")
        .smart_ptr_constructor("Implementation", &std::make_shared<mx::Implementation, mx::ElementPtr, const std::string &>)
        .smart_ptr<std::shared_ptr<const mx::Implementation>>("Implementation")
        .function("setFile", &mx::Implementation::setFile)
        .function("hasFile", &mx::Implementation::hasFile)
        .function("getFile", &mx::Implementation::getFile)
        .function("setFunction", &mx::Implementation::setFunction)
        .function("hasFunction", &mx::Implementation::hasFunction)
        .function("getFunction", &mx::Implementation::getFunction)
        .function("setNodeDef", &mx::Implementation::setNodeDef)
        .function("getNodeDef", &mx::Implementation::getNodeDef)
        .class_property("CATEGORY", &mx::Implementation::CATEGORY)
        .class_property("FILE_ATTRIBUTE", &mx::Implementation::FILE_ATTRIBUTE)
        .class_property("FUNCTION_ATTRIBUTE", &mx::Implementation::FUNCTION_ATTRIBUTE);

    ems::class_<mx::TypeDef, ems::base<mx::Element>>("TypeDef")
        .smart_ptr_constructor("TypeDef", &std::make_shared<mx::TypeDef, mx::ElementPtr, const std::string &>)
        .smart_ptr<std::shared_ptr<const mx::TypeDef>>("TypeDef")
        .function("setSemantic", &mx::TypeDef::setSemantic)
        .function("hasSemantic", &mx::TypeDef::hasSemantic)
        .function("getSemantic", &mx::TypeDef::getSemantic)
        .function("setContext", &mx::TypeDef::setContext)
        .function("hasContext", &mx::TypeDef::hasContext)
        .function("getContext", &mx::TypeDef::getContext)
        BIND_MEMBER_FUNC("addMember", mx::TypeDef, addMember, 0, 1, stRef)
        .function("getMember", &mx::TypeDef::getMember)
        .function("getMembers", &mx::TypeDef::getMembers)
        .function("removeMember", &mx::TypeDef::removeMember)
        .class_property("CATEGORY", &mx::TypeDef::CATEGORY)
        .class_property("SEMANTIC_ATTRIBUTE", &mx::TypeDef::SEMANTIC_ATTRIBUTE)
        .class_property("CONTEXT_ATTRIBUTE", &mx::TypeDef::CONTEXT_ATTRIBUTE);

    ems::class_<mx::TargetDef, ems::base<mx::TypedElement>>("TargetDef")
        .smart_ptr_constructor("TargetDef", &std::make_shared<mx::TargetDef, mx::ElementPtr, const std::string &>)
        .smart_ptr<std::shared_ptr<const mx::TargetDef>>("TargetDef")
        .function("getMatchingTargets", &mx::TargetDef::getMatchingTargets)
        .class_property("CATEGORY", &mx::TargetDef::CATEGORY);

    ems::class_<mx::Member, ems::base<mx::TypedElement>>("Member")
        .smart_ptr_constructor("Member", &std::make_shared<mx::Member, mx::ElementPtr, const std::string &>)
        .smart_ptr<std::shared_ptr<const mx::Member>>("Member")
        .class_property("CATEGORY", &mx::Member::CATEGORY);

    ems::class_<mx::Unit, ems::base<mx::Element>>("Unit")
        .smart_ptr_constructor("Unit", &std::make_shared<mx::Unit, mx::ElementPtr, const std::string &>)
        .smart_ptr<std::shared_ptr<const mx::Unit>>("Unit")
        .class_property("CATEGORY", &mx::Unit::CATEGORY);

    ems::class_<mx::UnitDef, ems::base<mx::Element>>("UnitDef")
        .smart_ptr_constructor("UnitDef", &std::make_shared<mx::UnitDef, mx::ElementPtr, const std::string &>)
        .smart_ptr<std::shared_ptr<const mx::UnitDef>>("UnitDef")
        .function("setUnitType", &mx::UnitDef::setUnitType)
        .function("hasUnitType", &mx::UnitDef::hasUnitType)
        .function("getUnitType", &mx::UnitDef::getUnitType)
        .function("addUnit", &mx::UnitDef::addUnit)
        .function("getUnit", &mx::UnitDef::getUnit)
        .function("getUnits", &mx::UnitDef::getUnits)
        .function("removeUnit", &mx::UnitDef::removeUnit)
        .class_property("CATEGORY", &mx::UnitDef::CATEGORY)
        .class_property("UNITTYPE_ATTRIBUTE", &mx::UnitDef::UNITTYPE_ATTRIBUTE);

    ems::class_<mx::UnitTypeDef, ems::base<mx::Element>>("UnitTypeDef")
        .smart_ptr_constructor("UnitTypeDef", &std::make_shared<mx::UnitTypeDef, mx::ElementPtr, const std::string &>)
        .smart_ptr<std::shared_ptr<const mx::UnitTypeDef>>("UnitTypeDef")
        .function("getUnitDefs", &mx::UnitTypeDef::getUnitDefs)
        .class_property("CATEGORY", &mx::UnitTypeDef::CATEGORY);

    ems::class_<mx::AttributeDef, ems::base<mx::TypedElement>>("AttributeDef")
        .smart_ptr_constructor("AttributeDef", &std::make_shared<mx::AttributeDef, mx::ElementPtr, const std::string &>)
        .smart_ptr<std::shared_ptr<const mx::AttributeDef>>("AttributeDef")
        .function("setAttrName", &mx::AttributeDef::setAttrName)
        .function("hasAttrName", &mx::AttributeDef::hasAttrName)
        .function("getAttrName", &mx::AttributeDef::getAttrName)
        .function("setValueString", &mx::AttributeDef::setValueString)
        .function("hasValueString", &mx::AttributeDef::hasValueString)
        .function("getValueString", &mx::AttributeDef::getValueString)
        BIND_ATTRIBUTE_DEF_FUNC_INSTANCE(Integer, int)
        BIND_ATTRIBUTE_DEF_FUNC_INSTANCE(Boolean, bool)
        BIND_ATTRIBUTE_DEF_FUNC_INSTANCE(Float, float)
        BIND_ATTRIBUTE_DEF_FUNC_INSTANCE(Color3, mx::Color3)
        BIND_ATTRIBUTE_DEF_FUNC_INSTANCE(Color4, mx::Color4)
        BIND_ATTRIBUTE_DEF_FUNC_INSTANCE(Vector2, mx::Vector2)
        BIND_ATTRIBUTE_DEF_FUNC_INSTANCE(Vector3, mx::Vector3)
        BIND_ATTRIBUTE_DEF_FUNC_INSTANCE(Vector4, mx::Vector4)
        BIND_ATTRIBUTE_DEF_FUNC_INSTANCE(Matrix33, mx::Matrix33)
        BIND_ATTRIBUTE_DEF_FUNC_INSTANCE(Matrix44, mx::Matrix44)
        BIND_ATTRIBUTE_DEF_FUNC_INSTANCE(String, std::string)
        BIND_ATTRIBUTE_DEF_FUNC_INSTANCE(IntegerArray, mx::IntVec)
        BIND_ATTRIBUTE_DEF_FUNC_INSTANCE(BooleanArray, mx::BoolVec)
        BIND_ATTRIBUTE_DEF_FUNC_INSTANCE(FloatArray, mx::FloatVec)
        BIND_ATTRIBUTE_DEF_FUNC_INSTANCE(StringArray, mx::StringVec)
        .function("hasValue", &mx::AttributeDef::hasValue)
        .function("getValue", &mx::AttributeDef::getValue)
        .function("setElements", &mx::AttributeDef::setElements)
        .function("hasElements", &mx::AttributeDef::hasElements)
        .function("getElements", &mx::AttributeDef::getElements)
        .function("setExportable", &mx::AttributeDef::setExportable)
        .function("getExportable", &mx::AttributeDef::getExportable)
        .class_property("CATEGORY", &mx::AttributeDef::CATEGORY)
        .class_property("ATTRNAME_ATTRIBUTE", &mx::AttributeDef::ATTRNAME_ATTRIBUTE)
        .class_property("VALUE_ATTRIBUTE", &mx::AttributeDef::VALUE_ATTRIBUTE)
        .class_property("ELEMENTS_ATTRIBUTE", &mx::AttributeDef::ELEMENTS_ATTRIBUTE)
        .class_property("EXPORTABLE_ATTRIBUTE", &mx::AttributeDef::EXPORTABLE_ATTRIBUTE);
}
