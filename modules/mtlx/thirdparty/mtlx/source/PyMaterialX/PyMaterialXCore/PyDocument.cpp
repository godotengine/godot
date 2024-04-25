//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <PyMaterialX/PyMaterialX.h>

#include <MaterialXCore/Document.h>

namespace py = pybind11;
namespace mx = MaterialX;

void bindPyDocument(py::module& mod)
{
    mod.def("createDocument", &mx::createDocument);

    py::class_<mx::Document, mx::DocumentPtr, mx::GraphElement>(mod, "Document")
        .def("initialize", &mx::Document::initialize)
        .def("copy", &mx::Document::copy)
        .def("importLibrary", &mx::Document::importLibrary)
        .def("getReferencedSourceUris", &mx::Document::getReferencedSourceUris)
        .def("addNodeGraph", &mx::Document::addNodeGraph,
            py::arg("name") = mx::EMPTY_STRING)
        .def("getNodeGraph", &mx::Document::getNodeGraph)
        .def("getNodeGraphs", &mx::Document::getNodeGraphs)
        .def("removeNodeGraph", &mx::Document::removeNodeGraph)
        .def("getMatchingPorts", &mx::Document::getMatchingPorts)
        .def("addGeomInfo", &mx::Document::addGeomInfo,
            py::arg("name") = mx::EMPTY_STRING, py::arg("geom") = mx::UNIVERSAL_GEOM_NAME)
        .def("getGeomInfo", &mx::Document::getGeomInfo)
        .def("getGeomInfos", &mx::Document::getGeomInfos)
        .def("removeGeomInfo", &mx::Document::removeGeomInfo)
        .def("getGeomPropValue", &mx::Document::getGeomPropValue,
            py::arg("geomPropName"), py::arg("geom") = mx::UNIVERSAL_GEOM_NAME)
        .def("addGeomPropDef", &mx::Document::addGeomPropDef)
        .def("getGeomPropDef", &mx::Document::getGeomPropDef)
        .def("getGeomPropDefs", &mx::Document::getGeomPropDefs)
        .def("removeGeomPropDef", &mx::Document::removeGeomPropDef)
        .def("getMaterialOutputs", &mx::Document::getMaterialOutputs)        
        .def("addLook", &mx::Document::addLook,
            py::arg("name") = mx::EMPTY_STRING)
        .def("getLook", &mx::Document::getLook)
        .def("getLooks", &mx::Document::getLooks)
        .def("removeLook", &mx::Document::removeLook)
        .def("addLookGroup", &mx::Document::addLookGroup,
            py::arg("name") = mx::EMPTY_STRING)
        .def("getLookGroup", &mx::Document::getLookGroup)
        .def("getLookGroups", &mx::Document::getLookGroups)
        .def("removeLookGroup", &mx::Document::removeLookGroup)
        .def("addCollection", &mx::Document::addCollection,
            py::arg("name") = mx::EMPTY_STRING)
        .def("getCollection", &mx::Document::getCollection)
        .def("getCollections", &mx::Document::getCollections)
        .def("removeCollection", &mx::Document::removeCollection)
        .def("addTypeDef", &mx::Document::addTypeDef,
            py::arg("name") = mx::EMPTY_STRING)
        .def("getTypeDef", &mx::Document::getTypeDef)
        .def("getTypeDefs", &mx::Document::getTypeDefs)
        .def("removeTypeDef", &mx::Document::removeTypeDef)
        .def("addNodeDef", &mx::Document::addNodeDef,
            py::arg("name") = mx::EMPTY_STRING, py::arg("type") = mx::DEFAULT_TYPE_STRING, py::arg("node") = mx::EMPTY_STRING)
        .def("addNodeDefFromGraph", &mx::Document::addNodeDefFromGraph)
        .def("getNodeDef", &mx::Document::getNodeDef)
        .def("getNodeDefs", &mx::Document::getNodeDefs)
        .def("removeNodeDef", &mx::Document::removeNodeDef)
        .def("getMatchingNodeDefs", &mx::Document::getMatchingNodeDefs)
        .def("addAttributeDef", &mx::Document::addAttributeDef)
        .def("getAttributeDef", &mx::Document::getAttributeDef)
        .def("getAttributeDefs", &mx::Document::getAttributeDefs)
        .def("removeAttributeDef", &mx::Document::removeAttributeDef)
        .def("addTargetDef", &mx::Document::addTargetDef)
        .def("getTargetDef", &mx::Document::getTargetDef)
        .def("getTargetDefs", &mx::Document::getTargetDefs)
        .def("removeTargetDef", &mx::Document::removeTargetDef)
        .def("addPropertySet", &mx::Document::addPropertySet,
            py::arg("name") = mx::EMPTY_STRING)
        .def("getPropertySet", &mx::Document::getPropertySet)
        .def("getPropertySets", &mx::Document::getPropertySets)
        .def("removePropertySet", &mx::Document::removePropertySet)
        .def("addVariantSet", &mx::Document::addVariantSet,
            py::arg("name") = mx::EMPTY_STRING)
        .def("getVariantSet", &mx::Document::getVariantSet)
        .def("getVariantSets", &mx::Document::getVariantSets)
        .def("removeVariantSet", &mx::Document::removeVariantSet)
        .def("addImplementation", &mx::Document::addImplementation,
            py::arg("name") = mx::EMPTY_STRING)
        .def("getImplementation", &mx::Document::getImplementation)
        .def("getImplementations", &mx::Document::getImplementations)
        .def("removeImplementation", &mx::Document::removeImplementation)
        .def("getMatchingImplementations", &mx::Document::getMatchingImplementations)
        .def("addUnitDef", &mx::Document::addUnitDef)
        .def("getUnitDef", &mx::Document::getUnitDef)
        .def("getUnitDefs", &mx::Document::getUnitDefs)
        .def("removeUnitDef", &mx::Document::removeUnitDef)
        .def("addUnitTypeDef", &mx::Document::addUnitTypeDef)
        .def("getUnitTypeDef", &mx::Document::getUnitTypeDef)
        .def("getUnitTypeDefs", &mx::Document::getUnitTypeDefs)
        .def("removeUnitTypeDef", &mx::Document::removeUnitTypeDef)
        .def("upgradeVersion", &mx::Document::upgradeVersion)
        .def("setColorManagementSystem", &mx::Document::setColorManagementSystem)
        .def("hasColorManagementSystem", &mx::Document::hasColorManagementSystem)
        .def("getColorManagementSystem", &mx::Document::getColorManagementSystem)
        .def("setColorManagementConfig", &mx::Document::setColorManagementConfig)
        .def("hasColorManagementConfig", &mx::Document::hasColorManagementConfig)
        .def("getColorManagementConfig", &mx::Document::getColorManagementConfig);
}
