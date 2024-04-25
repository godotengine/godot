//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <PyMaterialX/PyMaterialX.h>

#include <MaterialXCore/Document.h>
#include <MaterialXCore/Geom.h>
#include <MaterialXCore/Look.h>
#include <MaterialXCore/Material.h>
#include <MaterialXCore/Node.h>
#include <MaterialXCore/Traversal.h>

#define BIND_ELEMENT_FUNC_INSTANCE(T)                                                                           \
.def("_addChild" #T, &mx::Element::addChild<mx::T>)                                                             \
.def("_getChildOfType" #T, &mx::Element::getChildOfType<mx::T>)                                                 \
.def("_getChildrenOfType" #T, &mx::Element::getChildrenOfType<mx::T>, py::arg("category") = mx::EMPTY_STRING)   \
.def("_removeChildOfType" #T, &mx::Element::removeChildOfType<mx::T>)

#define BIND_VALUE_ELEMENT_FUNC_INSTANCE(NAME, T)                                                               \
.def("_setValue" #NAME, &mx::ValueElement::setValue<T>, py::arg("value"), py::arg("type") = mx::EMPTY_STRING)

namespace py = pybind11;
namespace mx = MaterialX;

void bindPyElement(py::module& mod)
{
    py::class_<mx::Element, mx::ElementPtr>(mod, "Element")
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def("setCategory", &mx::Element::setCategory)
        .def("getCategory", &mx::Element::getCategory)
        .def("setName", &mx::Element::setName)
        .def("getName", &mx::Element::getName)
        .def("getNamePath", &mx::Element::getNamePath,
            py::arg("relativeTo") = nullptr)
        .def("getDescendant", &mx::Element::getDescendant)
        .def("setFilePrefix", &mx::Element::setFilePrefix)
        .def("hasFilePrefix", &mx::Element::hasFilePrefix)
        .def("getFilePrefix", &mx::Element::getFilePrefix)
        .def("getActiveFilePrefix", &mx::Element::getActiveFilePrefix)
        .def("setGeomPrefix", &mx::Element::setGeomPrefix)
        .def("hasGeomPrefix", &mx::Element::hasGeomPrefix)
        .def("getGeomPrefix", &mx::Element::getGeomPrefix)
        .def("getActiveGeomPrefix", &mx::Element::getActiveGeomPrefix)
        .def("setColorSpace", &mx::Element::setColorSpace)
        .def("hasColorSpace", &mx::Element::hasColorSpace)
        .def("getColorSpace", &mx::Element::getColorSpace)
        .def("getActiveColorSpace", &mx::Element::getActiveColorSpace)
        .def("setInheritString", &mx::Element::setInheritString)
        .def("hasInheritString", &mx::Element::hasInheritString)
        .def("getInheritString", &mx::Element::getInheritString)
        .def("setInheritsFrom", &mx::Element::setInheritsFrom)
        .def("getInheritsFrom", &mx::Element::getInheritsFrom)
        .def("hasInheritedBase", &mx::Element::hasInheritedBase)
        .def("hasInheritanceCycle", &mx::Element::hasInheritanceCycle)
        .def("setNamespace", &mx::Element::setNamespace)
        .def("hasNamespace", &mx::Element::hasNamespace)
        .def("getNamespace", &mx::Element::getNamespace)
        .def("getQualifiedName", &mx::Element::getQualifiedName)
        .def("setDocString", &mx::Element::setDocString)
        .def("getDocString", &mx::Element::getDocString)
        .def("addChildOfCategory", &mx::Element::addChildOfCategory,
            py::arg("category"), py::arg("name") = mx::EMPTY_STRING)
        .def("changeChildCategory", &mx::Element::changeChildCategory)
        .def("_getChild", &mx::Element::getChild)
        .def("getChildren", &mx::Element::getChildren)
        .def("setChildIndex", &mx::Element::setChildIndex)
        .def("getChildIndex", &mx::Element::getChildIndex)
        .def("removeChild", &mx::Element::removeChild)
        .def("setAttribute", &mx::Element::setAttribute)
        .def("hasAttribute", &mx::Element::hasAttribute)
        .def("getAttribute", &mx::Element::getAttribute)
        .def("getAttributeNames", &mx::Element::getAttributeNames)
        .def("removeAttribute", &mx::Element::removeAttribute)
        .def("getSelf", static_cast<mx::ElementPtr (mx::Element::*)()>(&mx::Element::getSelf))
        .def("getParent", static_cast<mx::ElementPtr(mx::Element::*)()>(&mx::Element::getParent))
        .def("getRoot", static_cast<mx::ElementPtr(mx::Element::*)()>(&mx::Element::getRoot))
        .def("getDocument", static_cast<mx::DocumentPtr(mx::Element::*)()>(&mx::Element::getDocument))
        .def("traverseTree", &mx::Element::traverseTree)
        .def("traverseGraph", &mx::Element::traverseGraph)
        .def("getUpstreamEdge", &mx::Element::getUpstreamEdge,
            py::arg("index") = 0)
        .def("getUpstreamEdgeCount", &mx::Element::getUpstreamEdgeCount)
        .def("getUpstreamElement", &mx::Element::getUpstreamElement,
            py::arg("index") = 0)
        .def("traverseInheritance", &mx::Element::traverseInheritance)
        .def("setSourceUri", &mx::Element::setSourceUri)
        .def("hasSourceUri", &mx::Element::hasSourceUri)
        .def("getSourceUri", &mx::Element::getSourceUri)
        .def("getActiveSourceUri", &mx::Element::getActiveSourceUri)
        .def("validate", [](const mx::Element& elem)
            {
                std::string message;
                bool res = elem.validate(&message);
                return std::pair<bool, std::string>(res, message);
            })
        .def("copyContentFrom", &mx::Element::copyContentFrom)
        .def("clearContent", &mx::Element::clearContent)
        .def("createValidChildName", &mx::Element::createValidChildName)
        .def("createStringResolver", &mx::Element::createStringResolver,
             py::arg("geom") = mx::EMPTY_STRING)
        .def("asString", &mx::Element::asString)
        .def("__str__", &mx::Element::asString)
        BIND_ELEMENT_FUNC_INSTANCE(Collection)
        BIND_ELEMENT_FUNC_INSTANCE(Document)
        BIND_ELEMENT_FUNC_INSTANCE(GeomInfo)
        BIND_ELEMENT_FUNC_INSTANCE(GeomProp)
        BIND_ELEMENT_FUNC_INSTANCE(Implementation)
        BIND_ELEMENT_FUNC_INSTANCE(Look)
        BIND_ELEMENT_FUNC_INSTANCE(MaterialAssign)
        BIND_ELEMENT_FUNC_INSTANCE(Node)
        BIND_ELEMENT_FUNC_INSTANCE(NodeDef)
        BIND_ELEMENT_FUNC_INSTANCE(NodeGraph)
        BIND_ELEMENT_FUNC_INSTANCE(Property)
        BIND_ELEMENT_FUNC_INSTANCE(PropertySet)
        BIND_ELEMENT_FUNC_INSTANCE(PropertySetAssign)
        BIND_ELEMENT_FUNC_INSTANCE(Token)
        BIND_ELEMENT_FUNC_INSTANCE(TypeDef)
        BIND_ELEMENT_FUNC_INSTANCE(Visibility);

    py::class_<mx::TypedElement, mx::TypedElementPtr, mx::Element>(mod, "TypedElement")
        .def("setType", &mx::TypedElement::setType)
        .def("hasType", &mx::TypedElement::hasType)
        .def("getType", &mx::TypedElement::getType)
        .def("isColorType", &mx::TypedElement::isColorType)
        .def("isMultiOutputType", &mx::TypedElement::isMultiOutputType)
        .def("getTypeDef", &mx::TypedElement::getTypeDef)
        .def_readonly_static("TYPE_ATTRIBUTE", &mx::TypedElement::TYPE_ATTRIBUTE);

    py::class_<mx::ValueElement, mx::ValueElementPtr, mx::TypedElement>(mod, "ValueElement")
        .def("setValueString", &mx::ValueElement::setValueString)
        .def("hasValueString", &mx::ValueElement::hasValueString)
        .def("getValueString", &mx::ValueElement::getValueString)
        .def("getResolvedValueString", &mx::ValueElement::getResolvedValueString,
            py::arg("resolver") = nullptr)
        .def("setInterfaceName", &mx::ValueElement::setInterfaceName)
        .def("hasInterfaceName", &mx::ValueElement::hasInterfaceName)
        .def("getInterfaceName", &mx::ValueElement::getInterfaceName)
        .def("setImplementationName", &mx::ValueElement::setImplementationName)
        .def("hasImplementationName", &mx::ValueElement::hasImplementationName)
        .def("getImplementationName", &mx::ValueElement::getImplementationName)
        .def("_getValue", &mx::ValueElement::getValue)
        .def("_getDefaultValue", &mx::ValueElement::getDefaultValue)
        .def("setUnit", &mx::ValueElement::setUnit)
        .def("hasUnit", &mx::ValueElement::hasUnit)
        .def("getUnit", &mx::ValueElement::getUnit)
        .def("getActiveUnit", &mx::ValueElement::getActiveUnit)
        .def("setUnitType", &mx::ValueElement::setUnitType)
        .def("hasUnitType", &mx::ValueElement::hasUnitType)
        .def("getUnitType", &mx::ValueElement::getUnitType)
        .def("getIsUniform", &mx::ValueElement::getIsUniform)
        .def("setIsUniform", &mx::ValueElement::setIsUniform)
        .def_readonly_static("VALUE_ATTRIBUTE", &mx::ValueElement::VALUE_ATTRIBUTE)
        .def_readonly_static("INTERFACE_NAME_ATTRIBUTE", &mx::ValueElement::INTERFACE_NAME_ATTRIBUTE)
        .def_readonly_static("IMPLEMENTATION_NAME_ATTRIBUTE", &mx::ValueElement::IMPLEMENTATION_NAME_ATTRIBUTE)
        .def_readonly_static("IMPLEMENTATION_TYPE_ATTRIBUTE", &mx::ValueElement::IMPLEMENTATION_TYPE_ATTRIBUTE)
        .def_readonly_static("ENUM_ATTRIBUTE", &mx::ValueElement::ENUM_ATTRIBUTE)
        .def_readonly_static("ENUM_VALUES_ATTRIBUTE", &mx::ValueElement::ENUM_VALUES_ATTRIBUTE)
        .def_readonly_static("UNIT_ATTRIBUTE", &mx::ValueElement::UNIT_ATTRIBUTE)
        .def_readonly_static("UI_NAME_ATTRIBUTE", &mx::ValueElement::UI_NAME_ATTRIBUTE)
        .def_readonly_static("UI_FOLDER_ATTRIBUTE", &mx::ValueElement::UI_FOLDER_ATTRIBUTE)
        .def_readonly_static("UI_MIN_ATTRIBUTE", &mx::ValueElement::UI_MIN_ATTRIBUTE)
        .def_readonly_static("UI_MAX_ATTRIBUTE", &mx::ValueElement::UI_MAX_ATTRIBUTE)
        .def_readonly_static("UI_SOFT_MIN_ATTRIBUTE", &mx::ValueElement::UI_SOFT_MIN_ATTRIBUTE)
        .def_readonly_static("UI_SOFT_MAX_ATTRIBUTE", &mx::ValueElement::UI_SOFT_MAX_ATTRIBUTE)
        .def_readonly_static("UI_STEP_ATTRIBUTE", &mx::ValueElement::UI_STEP_ATTRIBUTE)
        .def_readonly_static("UI_ADVANCED_ATTRIBUTE", &mx::ValueElement::UI_ADVANCED_ATTRIBUTE)

        BIND_VALUE_ELEMENT_FUNC_INSTANCE(integer, int)
        BIND_VALUE_ELEMENT_FUNC_INSTANCE(boolean, bool)
        BIND_VALUE_ELEMENT_FUNC_INSTANCE(float, float)
        BIND_VALUE_ELEMENT_FUNC_INSTANCE(color3, mx::Color3)
        BIND_VALUE_ELEMENT_FUNC_INSTANCE(color4, mx::Color4)
        BIND_VALUE_ELEMENT_FUNC_INSTANCE(vector2, mx::Vector2)
        BIND_VALUE_ELEMENT_FUNC_INSTANCE(vector3, mx::Vector3)
        BIND_VALUE_ELEMENT_FUNC_INSTANCE(vector4, mx::Vector4)
        BIND_VALUE_ELEMENT_FUNC_INSTANCE(matrix33, mx::Matrix33)
        BIND_VALUE_ELEMENT_FUNC_INSTANCE(matrix44, mx::Matrix44)
        BIND_VALUE_ELEMENT_FUNC_INSTANCE(string, std::string)
        BIND_VALUE_ELEMENT_FUNC_INSTANCE(integerarray, mx::IntVec)
        BIND_VALUE_ELEMENT_FUNC_INSTANCE(booleanarray, mx::BoolVec)
        BIND_VALUE_ELEMENT_FUNC_INSTANCE(floatarray, mx::FloatVec)
        BIND_VALUE_ELEMENT_FUNC_INSTANCE(stringarray, mx::StringVec);

    py::class_<mx::Token, mx::TokenPtr, mx::ValueElement>(mod, "Token")
        .def_readonly_static("CATEGORY", &mx::Token::CATEGORY);

    py::class_<mx::CommentElement, mx::CommentElementPtr, mx::Element>(mod, "CommentElement")
        .def_readonly_static("CATEGORY", &mx::CommentElement::CATEGORY);

    py::class_<mx::NewlineElement, mx::NewlineElementPtr, mx::Element>(mod, "NewlineElement")
        .def_readonly_static("CATEGORY", &mx::NewlineElement::CATEGORY);

    py::class_<mx::GenericElement, mx::GenericElementPtr, mx::Element>(mod, "GenericElement")
        .def_readonly_static("CATEGORY", &mx::GenericElement::CATEGORY);

    py::class_<mx::StringResolver, mx::StringResolverPtr>(mod, "StringResolver")
        .def("setFilePrefix", &mx::StringResolver::setFilePrefix)
        .def("getFilePrefix", &mx::StringResolver::getFilePrefix)
        .def("setGeomPrefix", &mx::StringResolver::setGeomPrefix)
        .def("getGeomPrefix", &mx::StringResolver::getGeomPrefix)
        .def("setUdimString", &mx::StringResolver::setUdimString)
        .def("setUvTileString", &mx::StringResolver::setUvTileString)
        .def("setFilenameSubstitution", &mx::StringResolver::setFilenameSubstitution)
        .def("getFilenameSubstitutions", &mx::StringResolver::getFilenameSubstitutions)
        .def("setGeomNameSubstitution", &mx::StringResolver::setGeomNameSubstitution)
        .def("getGeomNameSubstitutions", &mx::StringResolver::getGeomNameSubstitutions)
        .def("resolve", &mx::StringResolver::resolve);

    py::class_<mx::ElementPredicate>(mod, "ElementPredicate");

    py::register_exception<mx::ExceptionOrphanedElement>(mod, "ExceptionOrphanedElement");

    mod.def("targetStringsMatch", &mx::targetStringsMatch);
    mod.def("prettyPrint", &mx::prettyPrint);
}
