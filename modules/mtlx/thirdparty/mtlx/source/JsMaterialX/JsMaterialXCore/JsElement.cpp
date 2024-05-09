//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <JsMaterialX/VectorHelper.h>
#include <JsMaterialX/Helpers.h>

#include <MaterialXCore/Document.h>
#include <MaterialXCore/Geom.h>
#include <MaterialXCore/Look.h>
#include <MaterialXCore/Material.h>
#include <MaterialXCore/Node.h>
#include <MaterialXCore/Traversal.h>

#include <emscripten/bind.h>

namespace ems = emscripten;
namespace mx = MaterialX;

#define BIND_VALUE_ELEMENT_FUNC_INSTANCE(NAME, T)          \
    BIND_MEMBER_FUNC("setValue" #NAME, mx::ValueElement, setValue<T>, 1, 2, const T&, stRef)

#define BIND_ELEMENT_CHILD_FUNC_INSTANCE(NAME, T)                                                 \
    BIND_MEMBER_FUNC("addChild" #NAME, mx::Element, addChild<T>, 0, 1, stRef)                     \
    .function("getChildOfType" #NAME, &mx::Element::getChildOfType<T>)                            \
    BIND_MEMBER_FUNC("getChildrenOfType" #NAME, mx::Element, getChildrenOfType<T>, 0, 1, stRef)   \
    .function("removeChildOfType" #NAME, &mx::Element::removeChildOfType<T>)                      \
    .function("getAncestorOfType" #NAME, &mx::Element::getAncestorOfType<T>)                      \
    BIND_MEMBER_FUNC("isA" #NAME, mx::Element, isA<T>, 0, 1, stRef)                               \
    .function("asA" #NAME, ems::select_overload<std::shared_ptr<T>()>(&mx::Element::asA<T>))
  
#define BIND_ELEMENT_FUNC_INSTANCE(NAME, T)                                   \
    .function("setTypedAttribute" #NAME, &mx::Element::setTypedAttribute<T>)  \
    .function("getTypedAttribute" #NAME, &mx::Element::getTypedAttribute<T>)

EMSCRIPTEN_BINDINGS(element)
{
    ems::class_<mx::Element>("Element")
        .smart_ptr<std::shared_ptr<mx::Element>>("Element")
        .smart_ptr<std::shared_ptr<const mx::Element>>("Element") // mx::ConstElementPtr
        .function("equals", ems::optional_override([](mx::Element& self, const mx::Element& rhs) { return self == rhs; }))
        .function("notEquals", ems::optional_override([](mx::Element& self, const mx::Element& rhs) { return self != rhs; }))
        .function("setCategory", &mx::Element::setCategory)
        .function("getCategory", &mx::Element::getCategory)
        .function("setName", &mx::Element::setName)
        .function("getName", &mx::Element::getName)
        BIND_MEMBER_FUNC("getNamePath", mx::Element, getNamePath, 0, 1, mx::ConstElementPtr)
        .function("getDescendant", &mx::Element::getDescendant)
        .function("setFilePrefix", &mx::Element::setFilePrefix)
        .function("hasFilePrefix", &mx::Element::hasFilePrefix)
        .function("getFilePrefix", &mx::Element::getFilePrefix)
        .function("getActiveFilePrefix", &mx::Element::getActiveFilePrefix)
        .function("setGeomPrefix", &mx::Element::setGeomPrefix)
        .function("hasGeomPrefix", &mx::Element::hasGeomPrefix)
        .function("getGeomPrefix", &mx::Element::getGeomPrefix)
        .function("getActiveGeomPrefix", &mx::Element::getActiveGeomPrefix)
        .function("setColorSpace", &mx::Element::setColorSpace)
        .function("hasColorSpace", &mx::Element::hasColorSpace)
        .function("getColorSpace", &mx::Element::getColorSpace)
        .function("getActiveColorSpace", &mx::Element::getActiveColorSpace)
        .function("setInheritString", &mx::Element::setInheritString)
        .function("hasInheritString", &mx::Element::hasInheritString)
        .function("getInheritString", &mx::Element::getInheritString)
        .function("setInheritsFrom", &mx::Element::setInheritsFrom)
        .function("getInheritsFrom", &mx::Element::getInheritsFrom)
        .function("hasInheritedBase", &mx::Element::hasInheritedBase)
        .function("hasInheritanceCycle", &mx::Element::hasInheritanceCycle)
        .function("setNamespace", &mx::Element::setNamespace)
        .function("hasNamespace", &mx::Element::hasNamespace)
        .function("getNamespace", &mx::Element::getNamespace)
        .function("getQualifiedName", &mx::Element::getQualifiedName)
        .function("setDocString", &mx::Element::setDocString)
        .function("getDocString", &mx::Element::getDocString)
        BIND_MEMBER_FUNC("addChildOfCategory", mx::Element, addChildOfCategory, 1, 2, stRef, std::string)
        .function("changeChildCategory", &mx::Element::changeChildCategory)
        .function("getChild", &mx::Element::getChild)
        .function("getChildren", &mx::Element::getChildren)
        .function("setChildIndex", &mx::Element::setChildIndex)
        .function("getChildIndex", &mx::Element::getChildIndex)
        .function("removeChild", &mx::Element::removeChild)
        BIND_ELEMENT_CHILD_FUNC_INSTANCE(Backdrop, mx::Backdrop)
        BIND_ELEMENT_CHILD_FUNC_INSTANCE(Collection, mx::Collection)
        BIND_ELEMENT_CHILD_FUNC_INSTANCE(Comment, mx::CommentElement)
        BIND_ELEMENT_CHILD_FUNC_INSTANCE(Generic, mx::GenericElement)
        BIND_ELEMENT_CHILD_FUNC_INSTANCE(GeomInfo, mx::GeomInfo)
        BIND_ELEMENT_CHILD_FUNC_INSTANCE(MaterialAssign, mx::MaterialAssign)
        BIND_ELEMENT_CHILD_FUNC_INSTANCE(PropertySetAssign, mx::PropertySetAssign)
        BIND_ELEMENT_CHILD_FUNC_INSTANCE(Visibility, mx::Visibility)
        BIND_ELEMENT_CHILD_FUNC_INSTANCE(GeomPropDef, mx::GeomPropDef)
        BIND_ELEMENT_CHILD_FUNC_INSTANCE(Implementation, mx::Implementation)
        BIND_ELEMENT_CHILD_FUNC_INSTANCE(Look, mx::Look)
        BIND_ELEMENT_CHILD_FUNC_INSTANCE(LookGroup, mx::LookGroup)
        BIND_ELEMENT_CHILD_FUNC_INSTANCE(PropertySet, mx::PropertySet)
        BIND_ELEMENT_CHILD_FUNC_INSTANCE(TypeDef, mx::TypeDef)
        BIND_ELEMENT_CHILD_FUNC_INSTANCE(AttributeDef, mx::AttributeDef)
        BIND_ELEMENT_CHILD_FUNC_INSTANCE(MaterialX, mx::Document)
        BIND_ELEMENT_CHILD_FUNC_INSTANCE(Node, mx::Node)
        BIND_ELEMENT_CHILD_FUNC_INSTANCE(NodeDef, mx::NodeDef)
        BIND_ELEMENT_CHILD_FUNC_INSTANCE(NodeGraph, mx::NodeGraph)
        BIND_ELEMENT_CHILD_FUNC_INSTANCE(Member, mx::Member)
        BIND_ELEMENT_CHILD_FUNC_INSTANCE(TargetDef, mx::TargetDef)
        BIND_ELEMENT_CHILD_FUNC_INSTANCE(Token, mx::Token)
        BIND_ELEMENT_CHILD_FUNC_INSTANCE(GeomProp, mx::GeomProp)
        BIND_ELEMENT_CHILD_FUNC_INSTANCE(Input, mx::Input)
        BIND_ELEMENT_CHILD_FUNC_INSTANCE(Output, mx::Output)
        BIND_ELEMENT_CHILD_FUNC_INSTANCE(Property, mx::Property)
        BIND_ELEMENT_CHILD_FUNC_INSTANCE(PropertyAssign, mx::PropertyAssign)
        BIND_ELEMENT_CHILD_FUNC_INSTANCE(Unit, mx::Unit)
        BIND_ELEMENT_CHILD_FUNC_INSTANCE(UnitDef, mx::UnitDef)
        BIND_ELEMENT_CHILD_FUNC_INSTANCE(UnitTypeDef, mx::UnitTypeDef)
        BIND_ELEMENT_CHILD_FUNC_INSTANCE(VariantAssign, mx::VariantAssign)
        BIND_ELEMENT_CHILD_FUNC_INSTANCE(VariantSet, mx::VariantSet)
        BIND_ELEMENT_CHILD_FUNC_INSTANCE(Variant, mx::Variant)
        .function("setAttribute", &mx::Element::setAttribute)
        .function("hasAttribute", &mx::Element::hasAttribute)
        .function("getAttribute", &mx::Element::getAttribute)
        .function("getAttributeNames", &mx::Element::getAttributeNames)
        .function("removeAttribute", &mx::Element::removeAttribute)
        .function("getSelf", ems::select_overload<mx::ConstElementPtr()const>(&mx::Element::getSelf))
        .function("getParent", ems::select_overload<mx::ConstElementPtr()const>(&mx::Element::getParent))
        .function("getRoot", ems::select_overload<mx::ConstElementPtr()const>(&mx::Element::getRoot))
        .function("getDocument", ems::select_overload<mx::ConstDocumentPtr()const>(&mx::Element::getDocument))
        BIND_ELEMENT_FUNC_INSTANCE(Integer, int)
        BIND_ELEMENT_FUNC_INSTANCE(Boolean, bool)
        BIND_ELEMENT_FUNC_INSTANCE(Float, float)
        BIND_ELEMENT_FUNC_INSTANCE(Color3, mx::Color3)
        BIND_ELEMENT_FUNC_INSTANCE(Color4, mx::Color4)
        BIND_ELEMENT_FUNC_INSTANCE(Vector2, mx::Vector2)
        BIND_ELEMENT_FUNC_INSTANCE(Vector3, mx::Vector3)
        BIND_ELEMENT_FUNC_INSTANCE(Vector4, mx::Vector4)
        BIND_ELEMENT_FUNC_INSTANCE(Matrix33, mx::Matrix33)
        BIND_ELEMENT_FUNC_INSTANCE(Matrix44, mx::Matrix44)
        BIND_ELEMENT_FUNC_INSTANCE(String, std::string)
        BIND_ELEMENT_FUNC_INSTANCE(IntegerArray, mx::IntVec)
        BIND_ELEMENT_FUNC_INSTANCE(BooleanArray, mx::BoolVec)
        BIND_ELEMENT_FUNC_INSTANCE(FloatArray, mx::FloatVec)
        BIND_ELEMENT_FUNC_INSTANCE(StringArray, mx::StringVec)
        .function("traverseTree", &mx::Element::traverseTree)
        .function("traverseGraph", &mx::Element::traverseGraph)
        BIND_MEMBER_FUNC("getUpstreamEdge", mx::Element, getUpstreamEdge, 0, 1, std::size_t)
        .function("getUpstreamEdgeCount", &mx::Element::getUpstreamEdgeCount)
        BIND_MEMBER_FUNC("getUpstreamElement", mx::Element, getUpstreamElement, 0, 1, std::size_t)
        .function("traverseInheritance", &mx::Element::traverseInheritance)
        .function("setSourceUri", &mx::Element::setSourceUri)
        .function("hasSourceUri", &mx::Element::hasSourceUri)
        .function("getSourceUri", &mx::Element::getSourceUri)
        .function("getActiveSourceUri", &mx::Element::getActiveSourceUri)
        .function("validate", ems::optional_override([](mx::Element &self) {
            return self.validate();
        }))
        .function("validate", ems::optional_override([](mx::Element &self, ems::val message) {
            std::string nativeMessage;
            bool handleMessage = message.typeOf().as<std::string>() == "object";
            bool res = self.validate(handleMessage ? &nativeMessage : nullptr);
            if (!res && handleMessage)
                message.set("message", nativeMessage);
            return res;
        }))
        .function("copyContentFrom", &mx::Element::copyContentFrom)
        .function("clearContent", &mx::Element::clearContent)
        .function("createValidChildName", &mx::Element::createValidChildName)
        BIND_MEMBER_FUNC("createStringResolver", mx::Element, createStringResolver, 0, 1, stRef)
        .function("asString", &mx::Element::asString)
        .class_property("NAME_ATTRIBUTE", &mx::Element::NAME_ATTRIBUTE)
        .class_property("FILE_PREFIX_ATTRIBUTE", &mx::Element::FILE_PREFIX_ATTRIBUTE)
        .class_property("GEOM_PREFIX_ATTRIBUTE", &mx::Element::GEOM_PREFIX_ATTRIBUTE)
        .class_property("COLOR_SPACE_ATTRIBUTE", &mx::Element::COLOR_SPACE_ATTRIBUTE)
        .class_property("INHERIT_ATTRIBUTE", &mx::Element::INHERIT_ATTRIBUTE)
        .class_property("NAMESPACE_ATTRIBUTE", &mx::Element::NAMESPACE_ATTRIBUTE)
        .class_property("DOC_ATTRIBUTE", &mx::Element::DOC_ATTRIBUTE);

    ems::class_<mx::TypedElement, ems::base<mx::Element>>("TypedElement")
        .smart_ptr<std::shared_ptr<mx::TypedElement>>("TypedElement")
        .smart_ptr<std::shared_ptr<const mx::TypedElement>>("TypedElement")
        .function("setType", &mx::TypedElement::setType)
        .function("hasType", &mx::TypedElement::hasType)
        .function("getType", &mx::TypedElement::getType)
        .function("isMultiOutputType", &mx::TypedElement::isMultiOutputType)
        .function("getTypeDef", &mx::TypedElement::getTypeDef)
        .class_property("TYPE_ATTRIBUTE", &mx::TypedElement::TYPE_ATTRIBUTE);

    ems::class_<mx::ValueElement, ems::base<mx::TypedElement>>("ValueElement")
        .smart_ptr<std::shared_ptr<mx::ValueElement>>("ValueElement")
        .smart_ptr<std::shared_ptr<const mx::ValueElement>>("ValueElement")
        .function("setValueString", &mx::ValueElement::setValueString)
        .function("hasValueString", &mx::ValueElement::hasValueString)
        .function("getValueString", &mx::ValueElement::getValueString)
        BIND_MEMBER_FUNC("getResolvedValueString", mx::ValueElement, getResolvedValueString, 0, 1, mx::StringResolverPtr)
        .function("setInterfaceName", &mx::ValueElement::setInterfaceName)
        .function("hasInterfaceName", &mx::ValueElement::hasInterfaceName)
        .function("getInterfaceName", &mx::ValueElement::getInterfaceName)
        .function("setImplementationName", &mx::ValueElement::setImplementationName)
        .function("hasImplementationName", &mx::ValueElement::hasImplementationName)
        .function("getImplementationName", &mx::ValueElement::getImplementationName)
        BIND_VALUE_ELEMENT_FUNC_INSTANCE(Integer, int)
        BIND_VALUE_ELEMENT_FUNC_INSTANCE(Boolean, bool)
        BIND_VALUE_ELEMENT_FUNC_INSTANCE(Float, float)
        BIND_VALUE_ELEMENT_FUNC_INSTANCE(Color3, mx::Color3)
        BIND_VALUE_ELEMENT_FUNC_INSTANCE(Color4, mx::Color4)
        BIND_VALUE_ELEMENT_FUNC_INSTANCE(Vector2, mx::Vector2)
        BIND_VALUE_ELEMENT_FUNC_INSTANCE(Vector3, mx::Vector3)
        BIND_VALUE_ELEMENT_FUNC_INSTANCE(Vector4, mx::Vector4)
        BIND_VALUE_ELEMENT_FUNC_INSTANCE(Matrix33, mx::Matrix33)
        BIND_VALUE_ELEMENT_FUNC_INSTANCE(Matrix44, mx::Matrix44)
        BIND_VALUE_ELEMENT_FUNC_INSTANCE(String, std::string)
        BIND_VALUE_ELEMENT_FUNC_INSTANCE(IntegerArray, mx::IntVec)
        BIND_VALUE_ELEMENT_FUNC_INSTANCE(BooleanArray, mx::BoolVec)
        BIND_VALUE_ELEMENT_FUNC_INSTANCE(FloatArray, mx::FloatVec)
        BIND_VALUE_ELEMENT_FUNC_INSTANCE(StringArray, mx::StringVec)
        .function("hasValue", &mx::ValueElement::hasValue)
        .function("getValue", &mx::ValueElement::getValue)
        BIND_MEMBER_FUNC("getResolvedValue", mx::ValueElement, getResolvedValue, 0, 1, mx::StringResolverPtr)
        .function("getDefaultValue", &mx::ValueElement::getDefaultValue)
        .function("setUnit", &mx::ValueElement::setUnit)
        .function("hasUnit", &mx::ValueElement::hasUnit)
        .function("getUnit", &mx::ValueElement::getUnit)
        .function("getActiveUnit", &mx::ValueElement::getActiveUnit)
        .function("setUnitType", &mx::ValueElement::setUnitType)
        .function("hasUnitType", &mx::ValueElement::hasUnitType)
        .function("getUnitType", &mx::ValueElement::getUnitType)
        .function("setIsUniform", &mx::ValueElement::setIsUniform)
        .function("getIsUniform", &mx::ValueElement::getIsUniform)
        .class_property("VALUE_ATTRIBUTE", &mx::ValueElement::VALUE_ATTRIBUTE)
        .class_property("INTERFACE_NAME_ATTRIBUTE", &mx::ValueElement::INTERFACE_NAME_ATTRIBUTE)
        .class_property("IMPLEMENTATION_NAME_ATTRIBUTE", &mx::ValueElement::IMPLEMENTATION_NAME_ATTRIBUTE)
        .class_property("IMPLEMENTATION_TYPE_ATTRIBUTE", &mx::ValueElement::IMPLEMENTATION_TYPE_ATTRIBUTE)
        .class_property("ENUM_ATTRIBUTE", &mx::ValueElement::ENUM_ATTRIBUTE)
        .class_property("ENUM_VALUES_ATTRIBUTE", &mx::ValueElement::ENUM_VALUES_ATTRIBUTE)
        .class_property("UI_NAME_ATTRIBUTE", &mx::ValueElement::UI_NAME_ATTRIBUTE)
        .class_property("UI_FOLDER_ATTRIBUTE", &mx::ValueElement::UI_FOLDER_ATTRIBUTE)
        .class_property("UI_MIN_ATTRIBUTE", &mx::ValueElement::UI_MIN_ATTRIBUTE)
        .class_property("UI_MAX_ATTRIBUTE", &mx::ValueElement::UI_MAX_ATTRIBUTE)
        .class_property("UI_SOFT_MIN_ATTRIBUTE", &mx::ValueElement::UI_SOFT_MIN_ATTRIBUTE)
        .class_property("UI_SOFT_MAX_ATTRIBUTE", &mx::ValueElement::UI_SOFT_MAX_ATTRIBUTE)
        .class_property("UI_STEP_ATTRIBUTE", &mx::ValueElement::UI_STEP_ATTRIBUTE)
        .class_property("UI_ADVANCED_ATTRIBUTE", &mx::ValueElement::UI_ADVANCED_ATTRIBUTE)
        .class_property("UNIT_ATTRIBUTE", &mx::ValueElement::UNIT_ATTRIBUTE)
        .class_property("UNITTYPE_ATTRIBUTE", &mx::ValueElement::UNITTYPE_ATTRIBUTE)
        .class_property("UNIFORM_ATTRIBUTE", &mx::ValueElement::UNIFORM_ATTRIBUTE);

    ems::class_<mx::Token, ems::base<mx::ValueElement>>("Token")
        .smart_ptr_constructor("Token", &std::make_shared<mx::Token, mx::ElementPtr, const std::string &>)
        .class_property("CATEGORY", &mx::Token::CATEGORY);

    ems::class_<mx::CommentElement, ems::base<mx::Element>>("CommentElement")
        .smart_ptr_constructor("CommentElement", &std::make_shared<mx::CommentElement, mx::ElementPtr, const std::string &>)
        .class_property("CATEGORY", &mx::CommentElement::CATEGORY);

    ems::class_<mx::GenericElement, ems::base<mx::Element>>("GenericElement")
        .smart_ptr_constructor("GenericElement", &std::make_shared<mx::GenericElement, mx::ElementPtr, const std::string &>)
        .class_property("CATEGORY", &mx::GenericElement::CATEGORY);

    ems::class_<mx::StringResolver>("StringResolver")
        .smart_ptr<std::shared_ptr<mx::StringResolver>>("StringResolver")
        .class_function("create", &mx::StringResolver::create) // Static function for creating a mx::StringResolver instance
        .function("setFilePrefix", &mx::StringResolver::setFilePrefix)
        .function("getFilePrefix", &mx::StringResolver::getFilePrefix)
        .function("setGeomPrefix", &mx::StringResolver::setGeomPrefix)
        .function("getGeomPrefix", &mx::StringResolver::getGeomPrefix)
        .function("setUdimString", &mx::StringResolver::setUdimString)
        .function("setUvTileString", &mx::StringResolver::setUvTileString)
        .function("setFilenameSubstitution", &mx::StringResolver::setFilenameSubstitution)
        .function("getFilenameSubstitutions", ems::optional_override([](mx::StringResolver &self) {
            std::unordered_map<std::string, std::string> res = self.mx::StringResolver::getFilenameSubstitutions();
            ems::val obj = ems::val::object();
            for (std::pair<std::string, std::string> element : res)
            {
                obj.set(element.first, element.second);
            }

            return obj;
        }))
        .function("setGeomNameSubstitution", &mx::StringResolver::setGeomNameSubstitution)
        .function("getGeomNameSubstitutions", ems::optional_override([](mx::StringResolver &self) {
            std::unordered_map<std::string, std::string> res = self.mx::StringResolver::getGeomNameSubstitutions();
            ems::val obj = ems::val::object();
            for (std::pair<std::string, std::string> element : res)
            {
                obj.set(element.first, element.second);
            }

            return obj;
        }))
        .function("resolve", &mx::StringResolver::resolve)
        .class_function("isResolvedType", &mx::StringResolver::isResolvedType);

    ems::class_<mx::ElementPredicate>("ElementPredicate");

    ems::function("targetStringsMatch", &mx::targetStringsMatch);
    ems::function("prettyPrint", &mx::prettyPrint);
}
