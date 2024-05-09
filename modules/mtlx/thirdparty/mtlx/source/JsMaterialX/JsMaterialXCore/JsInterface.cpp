//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <JsMaterialX/VectorHelper.h>
#include <JsMaterialX/Helpers.h>

#include <MaterialXCore/Interface.h>
#include <MaterialXCore/Node.h>

#include <emscripten/bind.h>

namespace ems = emscripten;
namespace mx = MaterialX;

#define BIND_INTERFACE_TYPE_INSTANCE(NAME, T)                                 \
    BIND_MEMBER_FUNC("setInputValue" #NAME, mx::InterfaceElement, setInputValue<T>, 2, 3, stRef, const T&, stRef)

EMSCRIPTEN_BINDINGS(interface)
{
    ems::class_<mx::PortElement, ems::base<mx::ValueElement>>("PortElement")
        .smart_ptr<std::shared_ptr<mx::PortElement>>("PortElement")
        .smart_ptr<std::shared_ptr<const mx::PortElement>>("PortElement")
        .function("setNodeName", &mx::PortElement::setNodeName)
        .function("hasNodeName", &mx::PortElement::hasNodeName)
        .function("getNodeName", &mx::PortElement::getNodeName)
        .function("setNodeGraphString", &mx::PortElement::setNodeGraphString)
        .function("hasNodeGraphString", &mx::PortElement::hasNodeGraphString)
        .function("getNodeGraphString", &mx::PortElement::getNodeGraphString)
        .function("setOutputString", &mx::PortElement::setOutputString)
        .function("hasOutputString", &mx::PortElement::hasOutputString)
        .function("getOutputString", &mx::PortElement::getOutputString)
        .function("setChannels", &mx::PortElement::setChannels)
        .function("hasChannels", &mx::PortElement::hasChannels)
        .function("getChannels", &mx::PortElement::getChannels)
        .function("validChannelsCharacters", &mx::PortElement::validChannelsCharacters)
        .function("validChannelsString", &mx::PortElement::validChannelsString)
        .function("setConnectedNode", &mx::PortElement::setConnectedNode)
        .function("getConnectedNode", &mx::PortElement::getConnectedNode)
        .class_property("NODE_NAME_ATTRIBUTE", &mx::PortElement::NODE_NAME_ATTRIBUTE)
        .class_property("NODE_GRAPH_ATTRIBUTE", &mx::PortElement::NODE_GRAPH_ATTRIBUTE)
        .class_property("OUTPUT_ATTRIBUTE", &mx::PortElement::OUTPUT_ATTRIBUTE)
        .class_property("CHANNELS_ATTRIBUTE", &mx::PortElement::CHANNELS_ATTRIBUTE);

    ems::class_<mx::Input, ems::base<mx::PortElement>>("Input")
        .smart_ptr_constructor("Input", &std::make_shared<mx::Input, mx::ElementPtr, const std::string &>)
        .smart_ptr<std::shared_ptr<const mx::Input>>("Input")
        .function("setDefaultGeomPropString", &mx::Input::setDefaultGeomPropString)
        .function("hasDefaultGeomPropString", &mx::Input::hasDefaultGeomPropString)
        .function("getDefaultGeomPropString", &mx::Input::getDefaultGeomPropString)
        .function("getDefaultGeomProp", &mx::Input::getDefaultGeomProp)
        .function("getConnectedNode", &mx::Input::getConnectedNode)
        .function("setConnectedOutput", &mx::Input::setConnectedOutput)
        .function("getConnectedOutput", &mx::Input::getConnectedOutput)
        .function("getInterfaceInput", &mx::Input::getInterfaceInput)
        .class_property("CATEGORY", &mx::Input::CATEGORY)
        .class_property("DEFAULT_GEOM_PROP_ATTRIBUTE", &mx::Input::DEFAULT_GEOM_PROP_ATTRIBUTE);

    ems::class_<mx::Output, ems::base<mx::PortElement>>("Output")
        .smart_ptr_constructor("Output", &std::make_shared<mx::Output, mx::ElementPtr, const std::string &>)
        .smart_ptr<std::shared_ptr<const mx::Output>>("Output")
        .function("getUpstreamEdgeCount", &mx::Output::getUpstreamEdgeCount)
        .function("hasUpstreamCycle", &mx::Output::hasUpstreamCycle)
        .class_property("CATEGORY", &mx::Output::CATEGORY)
        .class_property("DEFAULT_INPUT_ATTRIBUTE", &mx::Output::DEFAULT_INPUT_ATTRIBUTE);

    ems::class_<mx::InterfaceElement, ems::base<mx::TypedElement>>("InterfaceElement")
        .smart_ptr<std::shared_ptr<mx::InterfaceElement>>("InterfaceElement")
        .smart_ptr<std::shared_ptr<const mx::InterfaceElement>>("InterfaceElement")
        .function("setNodeDefString", &mx::InterfaceElement::setNodeDefString)
        .function("hasNodeDefString", &mx::InterfaceElement::hasNodeDefString)
        .function("getNodeDefString", &mx::InterfaceElement::getNodeDefString)
        BIND_MEMBER_FUNC("addInput", mx::InterfaceElement, addInput, 0, 2, stRef, stRef)
        .function("getInput", &mx::InterfaceElement::getInput)
        .function("getInputs", &mx::InterfaceElement::getInputs)
        .function("getInputCount", &mx::InterfaceElement::getInputCount)
        .function("removeInput", &mx::InterfaceElement::removeInput)
        .function("getActiveInput", &mx::InterfaceElement::getActiveInput)
        .function("getActiveInputs", &mx::InterfaceElement::getActiveInputs)
        BIND_MEMBER_FUNC("addOutput", mx::InterfaceElement, addOutput, 0, 2, stRef, stRef)
        .function("getOutput", &mx::InterfaceElement::getOutput)
        .function("getOutputs", &mx::InterfaceElement::getOutputs)
        .function("getOutputCount", &mx::InterfaceElement::getOutputCount)
        .function("removeOutput", &mx::InterfaceElement::removeOutput)
        .function("getActiveOutput", &mx::InterfaceElement::getActiveOutput)
        .function("getActiveOutputs", &mx::InterfaceElement::getActiveOutputs)
        .function("setConnectedOutput", &mx::InterfaceElement::setConnectedOutput)
        .function("getConnectedOutput", &mx::InterfaceElement::getConnectedOutput)
        .function("getToken", &mx::InterfaceElement::getToken)
        .function("getTokens", &mx::InterfaceElement::getTokens)
        .function("removeToken", &mx::InterfaceElement::removeToken)
        .function("getActiveToken", &mx::InterfaceElement::getActiveToken)
        .function("getActiveTokens", &mx::InterfaceElement::getActiveTokens)
        .function("getValueElement", &mx::InterfaceElement::getValueElement)
        .function("getActiveValueElement", &mx::InterfaceElement::getActiveValueElement)
        .function("getActiveValueElements", &mx::InterfaceElement::getActiveValueElements)
        BIND_INTERFACE_TYPE_INSTANCE(Integer, int)
        BIND_INTERFACE_TYPE_INSTANCE(Boolean, bool)
        BIND_INTERFACE_TYPE_INSTANCE(Float, float)
        BIND_INTERFACE_TYPE_INSTANCE(Color3, mx::Color3)
        BIND_INTERFACE_TYPE_INSTANCE(Color4, mx::Color4)
        BIND_INTERFACE_TYPE_INSTANCE(Vector2, mx::Vector2)
        BIND_INTERFACE_TYPE_INSTANCE(Vector3, mx::Vector3)
        BIND_INTERFACE_TYPE_INSTANCE(Vector4, mx::Vector4)
        BIND_INTERFACE_TYPE_INSTANCE(Matrix33, mx::Matrix33)
        BIND_INTERFACE_TYPE_INSTANCE(Matrix44, mx::Matrix44)
        BIND_INTERFACE_TYPE_INSTANCE(String, std::string)
        BIND_INTERFACE_TYPE_INSTANCE(IntegerArray, mx::IntVec)
        BIND_INTERFACE_TYPE_INSTANCE(BooleanArray, mx::BoolVec)
        BIND_INTERFACE_TYPE_INSTANCE(FloatArray, mx::FloatVec)
        BIND_INTERFACE_TYPE_INSTANCE(StringArray, mx::StringVec)
        BIND_MEMBER_FUNC("getInputValue", mx::InterfaceElement, getInputValue, 1, 2, stRef, stRef)
        .function("setTokenValue", &mx::InterfaceElement::setTokenValue)
        .function("getTokenValue", &mx::InterfaceElement::getTokenValue)
        .function("setTarget", &mx::InterfaceElement::setTarget)
        .function("hasTarget", &mx::InterfaceElement::hasTarget)
        .function("getTarget", &mx::InterfaceElement::getTarget)
        .function("setVersionString", &mx::InterfaceElement::setVersionString)
        .function("hasVersionString", &mx::InterfaceElement::hasVersionString)
        .function("getVersionString", &mx::InterfaceElement::getVersionString)
        .function("setVersionIntegers", &mx::InterfaceElement::setVersionIntegers)
        .function("getVersionIntegers", &mx::InterfaceElement::getVersionIntegers)
        .function("setDefaultVersion", &mx::InterfaceElement::setDefaultVersion)
        .function("getDefaultVersion", &mx::InterfaceElement::getDefaultVersion)
        BIND_MEMBER_FUNC("getDeclaration", mx::InterfaceElement, getDeclaration, 0, 1, stRef)
        .class_property("NODE_DEF_ATTRIBUTE", &mx::InterfaceElement::NODE_DEF_ATTRIBUTE)
        .class_property("TARGET_ATTRIBUTE", &mx::InterfaceElement::TARGET_ATTRIBUTE)
        .class_property("VERSION_ATTRIBUTE", &mx::InterfaceElement::VERSION_ATTRIBUTE)
        .class_property("DEFAULT_VERSION_ATTRIBUTE", &mx::InterfaceElement::DEFAULT_VERSION_ATTRIBUTE);
}
