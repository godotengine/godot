//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <JsMaterialX/VectorHelper.h>
#include <JsMaterialX/Helpers.h>

#include <MaterialXCore/Node.h>

#include <emscripten/bind.h>

namespace ems = emscripten;
namespace mx = MaterialX;

EMSCRIPTEN_BINDINGS(node)
{
    ems::class_<mx::Node, ems::base<mx::InterfaceElement>>("Node")
        .smart_ptr_constructor("Node", &std::make_shared<mx::Node, mx::ElementPtr, const std::string &>)
        .smart_ptr<std::shared_ptr<const mx::Node>>("Node")
        .function("setConnectedNode", &mx::Node::setConnectedNode)
        .function("getConnectedNode", &mx::Node::getConnectedNode)
        .function("setConnectedNodeName", &mx::Node::setConnectedNodeName)
        .function("getConnectedNodeName", &mx::Node::getConnectedNodeName)
        .function("setConnectedOutput", &mx::Node::setConnectedOutput)
        .function("getConnectedOutput", &mx::Node::getConnectedOutput)
        BIND_MEMBER_FUNC("getNodeDef", mx::Node, getNodeDef, 0, 1, stRef)
        BIND_MEMBER_FUNC("getImplementation", mx::Node, getImplementation, 0, 1, stRef)
        .function("getUpstreamEdgeCount", &mx::Node::getUpstreamEdgeCount)
        .function("getNodeDefOutput", &mx::Node::getNodeDefOutput)
        .function("getDownstreamPorts", &mx::Node::getDownstreamPorts)
        .function("addInputFromNodeDef", &mx::Node::addInputFromNodeDef)
        .class_property("CATEGORY", &mx::Node::CATEGORY);

    ems::class_<mx::GraphElement, ems::base<mx::InterfaceElement>>("GraphElement")
        .smart_ptr<std::shared_ptr<mx::GraphElement>>("GraphElement")
        .smart_ptr<std::shared_ptr<const mx::GraphElement>>("GraphElement")
        BIND_MEMBER_FUNC("addNode", mx::GraphElement, addNode, 1, 3, stRef, stRef, stRef)
        BIND_MEMBER_FUNC("addNodeInstance", mx::GraphElement, addNodeInstance, 1, 2, mx::ConstNodeDefPtr, stRef)
        .function("getNode", &mx::GraphElement::getNode)
        BIND_MEMBER_FUNC("getNodes", mx::GraphElement, getNodes, 0, 1, stRef)
        .function("getNodesOfType", &mx::GraphElement::getNodesOfType)
        .function("removeNode", &mx::GraphElement::removeNode)
        BIND_MEMBER_FUNC("addMaterialNode", mx::GraphElement, addMaterialNode, 0, 2, stRef, mx::ConstNodePtr)
        .function("getMaterialNodes", &mx::GraphElement::getMaterialNodes)
        BIND_MEMBER_FUNC("addBackdrop", mx::GraphElement, addBackdrop, 0, 1, stRef)
        .function("getBackdrop", &mx::GraphElement::getBackdrop)
        .function("getBackdrops", &mx::GraphElement::getBackdrops)
        .function("removeBackdrop", &mx::GraphElement::removeBackdrop)
        BIND_MEMBER_FUNC("flattenSubgraphs", mx::GraphElement, flattenSubgraphs, 0, 2, stRef, mx::NodePredicate)
        .function("topologicalSort", &mx::GraphElement::topologicalSort)
        .function("asStringDot", &mx::GraphElement::asStringDot);

    ems::class_<mx::NodeGraph, ems::base<mx::GraphElement>>("NodeGraph")
        .smart_ptr_constructor("NodeGraph", &std::make_shared<mx::NodeGraph, mx::ElementPtr, const std::string &>)
        .smart_ptr<std::shared_ptr<const mx::NodeGraph>>("NodeGraph")
        .function("setNodeDef", &mx::NodeGraph::setNodeDef)
        .function("getNodeDef", &mx::NodeGraph::getNodeDef)
        .function("getImplementation", &mx::NodeGraph::getImplementation)
        .function("getDownstreamPorts", &mx::NodeGraph::getDownstreamPorts)
        .function("addInterfaceName", &mx::NodeGraph::addInterfaceName)
        .function("removeInterfaceName", &mx::NodeGraph::removeInterfaceName)
        .function("modifyInterfaceName", &mx::NodeGraph::modifyInterfaceName)
        .class_property("CATEGORY", &mx::NodeGraph::CATEGORY);

    ems::class_<mx::Backdrop, ems::base<mx::Element>>("Backdrop")
        .smart_ptr_constructor("Backdrop", &std::make_shared<mx::Backdrop, mx::ElementPtr, const std::string &>)
        .smart_ptr<std::shared_ptr<const mx::Backdrop>>("Backdrop")
        .function("setContainsString", &mx::Backdrop::setContainsString)
        .function("hasContainsString", &mx::Backdrop::hasContainsString)
        .function("getContainsString", &mx::Backdrop::getContainsString)
        .function("setWidth", &mx::Backdrop::setWidth)
        .function("hasWidth", &mx::Backdrop::hasWidth)
        .function("getWidth", &mx::Backdrop::getWidth)
        .function("setHeight", &mx::Backdrop::setHeight)
        .function("hasHeight", &mx::Backdrop::hasHeight)
        .function("getHeight", &mx::Backdrop::getHeight)
        .function("setContainsElements", &mx::Backdrop::setContainsElements)
        .function("getContainsElements", &mx::Backdrop::getContainsElements)
        .class_property("CATEGORY", &mx::Backdrop::CATEGORY)
        .class_property("CONTAINS_ATTRIBUTE", &mx::Backdrop::CONTAINS_ATTRIBUTE)
        .class_property("WIDTH_ATTRIBUTE", &mx::Backdrop::WIDTH_ATTRIBUTE)
        .class_property("HEIGHT_ATTRIBUTE", &mx::Backdrop::HEIGHT_ATTRIBUTE);
}
