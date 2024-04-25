//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <PyMaterialX/PyMaterialX.h>

#include <MaterialXCore/Node.h>

namespace py = pybind11;
namespace mx = MaterialX;

void bindPyNode(py::module& mod)
{
    py::class_<mx::NodePredicate>(mod, "NodePredicate");

    py::class_<mx::Node, mx::NodePtr, mx::InterfaceElement>(mod, "Node")
        .def("setConnectedNode", &mx::Node::setConnectedNode)
        .def("getConnectedNode", &mx::Node::getConnectedNode)
        .def("setConnectedNodeName", &mx::Node::setConnectedNodeName)
        .def("getConnectedNodeName", &mx::Node::getConnectedNodeName)
        .def("getNodeDef", &mx::Node::getNodeDef,
            py::arg("target") = mx::EMPTY_STRING, py::arg("allowRoughMatch") = false)
        .def("getImplementation", &mx::Node::getImplementation,
            py::arg("target") = mx::EMPTY_STRING)
        .def("getDownstreamPorts", &mx::Node::getDownstreamPorts)
        .def("addInputFromNodeDef", &mx::Node::addInputFromNodeDef)
        .def("addInputsFromNodeDef", &mx::Node::addInputsFromNodeDef)
        .def_readonly_static("CATEGORY", &mx::Node::CATEGORY);

    py::class_<mx::GraphElement, mx::GraphElementPtr, mx::InterfaceElement>(mod, "GraphElement")
        .def("addNode", &mx::GraphElement::addNode,
            py::arg("category"), py::arg("name") = mx::EMPTY_STRING, py::arg("type") = mx::DEFAULT_TYPE_STRING)
        .def("addNodeInstance", &mx::GraphElement::addNodeInstance,
            py::arg("nodeDef"), py::arg("name") = mx::EMPTY_STRING)
        .def("getNode", &mx::GraphElement::getNode)
        .def("getNodes", &mx::GraphElement::getNodes,
            py::arg("category") = mx::EMPTY_STRING)
        .def("removeNode", &mx::GraphElement::removeNode)
        .def("addMaterialNode", &mx::GraphElement::addMaterialNode,
            py::arg("name") = mx::EMPTY_STRING, py::arg("shaderNode") = nullptr)
        .def("getMaterialNodes", &mx::GraphElement::getMaterialNodes)
        .def("addBackdrop", &mx::GraphElement::addBackdrop,
            py::arg("name") = mx::EMPTY_STRING)
        .def("getBackdrop", &mx::GraphElement::getBackdrop)
        .def("getBackdrops", &mx::GraphElement::getBackdrops)
        .def("removeBackdrop", &mx::GraphElement::removeBackdrop)
        .def("flattenSubgraphs", &mx::GraphElement::flattenSubgraphs,
            py::arg("target") = mx::EMPTY_STRING, py::arg("filter") = nullptr)
        .def("topologicalSort", &mx::GraphElement::topologicalSort)
        .def("addGeomNode", &mx::GraphElement::addGeomNode)
        .def("asStringDot", &mx::GraphElement::asStringDot);

    py::class_<mx::NodeGraph, mx::NodeGraphPtr, mx::GraphElement>(mod, "NodeGraph")
        .def("getMaterialOutputs", &mx::NodeGraph::getMaterialOutputs)        
        .def("setNodeDef", &mx::NodeGraph::setNodeDef)
        .def("getNodeDef", &mx::NodeGraph::getNodeDef)
        .def("getDeclaration", &mx::NodeGraph::getDeclaration)
        .def("addInterfaceName", &mx::NodeGraph::addInterfaceName)
        .def("removeInterfaceName", &mx::NodeGraph::removeInterfaceName)
        .def("modifyInterfaceName", &mx::NodeGraph::modifyInterfaceName)
        .def("getDownstreamPorts", &mx::NodeGraph::getDownstreamPorts)
        .def_readonly_static("CATEGORY", &mx::NodeGraph::CATEGORY);

    py::class_<mx::Backdrop, mx::BackdropPtr, mx::Element>(mod, "Backdrop")
        .def("setContainsString", &mx::Backdrop::setContainsString)
        .def("hasContainsString", &mx::Backdrop::hasContainsString)
        .def("getContainsString", &mx::Backdrop::getContainsString)
        .def("setWidth", &mx::Backdrop::setWidth)
        .def("hasWidth", &mx::Backdrop::hasWidth)
        .def("getWidth", &mx::Backdrop::getWidth)
        .def("setHeight", &mx::Backdrop::setHeight)
        .def("hasHeight", &mx::Backdrop::hasHeight)
        .def("getHeight", &mx::Backdrop::getHeight)
        .def("setContainsElements", &mx::Backdrop::setContainsElements)
        .def("getContainsElements", &mx::Backdrop::getContainsElements)
        .def_readonly_static("CATEGORY", &mx::Backdrop::CATEGORY)
        .def_readonly_static("CONTAINS_ATTRIBUTE", &mx::Backdrop::CONTAINS_ATTRIBUTE)
        .def_readonly_static("WIDTH_ATTRIBUTE", &mx::Backdrop::WIDTH_ATTRIBUTE)
        .def_readonly_static("HEIGHT_ATTRIBUTE", &mx::Backdrop::HEIGHT_ATTRIBUTE);
}
