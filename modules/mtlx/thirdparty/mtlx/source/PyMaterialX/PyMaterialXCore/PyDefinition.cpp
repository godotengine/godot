//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <PyMaterialX/PyMaterialX.h>

#include <MaterialXCore/Definition.h>

#include <MaterialXCore/Material.h>

namespace py = pybind11;
namespace mx = MaterialX;

void bindPyDefinition(py::module& mod)
{
    py::class_<mx::NodeDef, mx::NodeDefPtr, mx::InterfaceElement>(mod, "NodeDef")
        .def("setNodeString", &mx::NodeDef::setNodeString)
        .def("hasNodeString", &mx::NodeDef::hasNodeString)
        .def("getNodeString", &mx::NodeDef::getNodeString)
        .def("setNodeGroup", &mx::NodeDef::setNodeGroup)
        .def("hasNodeGroup", &mx::NodeDef::hasNodeGroup)
        .def("getNodeGroup", &mx::NodeDef::getNodeGroup)
        .def("getImplementation", &mx::NodeDef::getImplementation)
        .def("getImplementation", &mx::NodeDef::getImplementation,
            py::arg("target") = mx::EMPTY_STRING)
        .def("isVersionCompatible", &mx::NodeDef::isVersionCompatible)
        .def_readonly_static("CATEGORY", &mx::NodeDef::CATEGORY)
        .def_readonly_static("NODE_ATTRIBUTE", &mx::NodeDef::NODE_ATTRIBUTE)
        .def_readonly_static("TEXTURE_NODE_GROUP", &mx::NodeDef::TEXTURE_NODE_GROUP)
        .def_readonly_static("PROCEDURAL_NODE_GROUP", &mx::NodeDef::PROCEDURAL_NODE_GROUP)
        .def_readonly_static("GEOMETRIC_NODE_GROUP", &mx::NodeDef::GEOMETRIC_NODE_GROUP)
        .def_readonly_static("ADJUSTMENT_NODE_GROUP", &mx::NodeDef::ADJUSTMENT_NODE_GROUP)
        .def_readonly_static("CONDITIONAL_NODE_GROUP", &mx::NodeDef::CONDITIONAL_NODE_GROUP)
        .def_readonly_static("ORGANIZATION_NODE_GROUP", &mx::NodeDef::ORGANIZATION_NODE_GROUP);

    py::class_<mx::Implementation, mx::ImplementationPtr, mx::InterfaceElement>(mod, "Implementation")
        .def("setFile", &mx::Implementation::setFile)
        .def("hasFile", &mx::Implementation::hasFile)
        .def("getFile", &mx::Implementation::getFile)
        .def("setFunction", &mx::Implementation::setFunction)
        .def("hasFunction", &mx::Implementation::hasFunction)
        .def("getFunction", &mx::Implementation::getFunction)
        .def("setNodeDef", &mx::Implementation::setNodeDef)
        .def("getNodeDef", &mx::Implementation::getNodeDef)
        .def("setNodeGraph", &mx::Implementation::setNodeGraph)
        .def("hasNodeGraph", &mx::Implementation::hasNodeGraph)        
        .def("getNodeGraph", &mx::Implementation::getNodeGraph)
        .def_readonly_static("CATEGORY", &mx::Implementation::CATEGORY)
        .def_readonly_static("FILE_ATTRIBUTE", &mx::Implementation::FILE_ATTRIBUTE)
        .def_readonly_static("FUNCTION_ATTRIBUTE", &mx::Implementation::FUNCTION_ATTRIBUTE);

    py::class_<mx::TypeDef, mx::TypeDefPtr, mx::Element>(mod, "TypeDef")
        .def("setSemantic", &mx::TypeDef::setSemantic)
        .def("hasSemantic", &mx::TypeDef::hasSemantic)
        .def("getSemantic", &mx::TypeDef::getSemantic)
        .def("setContext", &mx::TypeDef::setContext)
        .def("hasContext", &mx::TypeDef::hasContext)
        .def("getContext", &mx::TypeDef::getContext)
        .def("addMember", &mx::TypeDef::addMember,
            py::arg("name") = mx::EMPTY_STRING)
        .def("getMember", &mx::TypeDef::getMember)
        .def("getMembers", &mx::TypeDef::getMembers)
        .def("removeMember", &mx::TypeDef::removeMember)
        .def_readonly_static("CATEGORY", &mx::TypeDef::CATEGORY)
        .def_readonly_static("SEMANTIC_ATTRIBUTE", &mx::TypeDef::SEMANTIC_ATTRIBUTE)
        .def_readonly_static("CONTEXT_ATTRIBUTE", &mx::TypeDef::CONTEXT_ATTRIBUTE);

    py::class_<mx::Member, mx::MemberPtr, mx::TypedElement>(mod, "Member")
        .def_readonly_static("CATEGORY", &mx::TypeDef::CATEGORY);

    py::class_<mx::Unit, mx::UnitPtr, mx::Element>(mod, "Unit")
        .def_readonly_static("CATEGORY", &mx::Unit::CATEGORY);

    py::class_<mx::UnitDef, mx::UnitDefPtr, mx::Element>(mod, "UnitDef")
        .def("setUnitType", &mx::UnitDef::hasUnitType)
        .def("hasUnitType", &mx::UnitDef::hasUnitType)
        .def("getUnitType", &mx::UnitDef::getUnitType)
        .def("addUnit", &mx::UnitDef::addUnit)
        .def("getUnit", &mx::UnitDef::getUnit)
        .def("getUnits", &mx::UnitDef::getUnits)
        .def_readonly_static("CATEGORY", &mx::UnitDef::CATEGORY)
        .def_readonly_static("UNITTYPE_ATTRIBUTE", &mx::UnitDef::UNITTYPE_ATTRIBUTE);

    py::class_<mx::UnitTypeDef, mx::UnitTypeDefPtr, mx::Element>(mod, "UnitTypeDef")
        .def("getUnitDefs", &mx::UnitTypeDef::getUnitDefs)
        .def_readonly_static("CATEGORY", &mx::UnitTypeDef::CATEGORY);

    py::class_<mx::AttributeDef, mx::AttributeDefPtr, mx::TypedElement>(mod, "AttributeDef")
        .def("setAttrName", &mx::AttributeDef::setAttrName)
        .def("hasAttrName", &mx::AttributeDef::hasAttrName)
        .def("getAttrName", &mx::AttributeDef::getAttrName)
        .def("setValueString", &mx::AttributeDef::setValueString)
        .def("hasValueString", &mx::AttributeDef::hasValueString)
        .def("getValueString", &mx::AttributeDef::getValueString) 
        .def("setExportable", &mx::AttributeDef::setExportable)         
        .def("getExportable", &mx::AttributeDef::getExportable)         
        .def_readonly_static("CATEGORY", &mx::AttributeDef::CATEGORY);

    py::class_<mx::TargetDef, mx::TargetDefPtr, mx::TypedElement>(mod, "TargetDef")
        .def("getMatchingTargets", &mx::TargetDef::getMatchingTargets)
        .def_readonly_static("CATEGORY", &mx::TargetDef::CATEGORY);
}
