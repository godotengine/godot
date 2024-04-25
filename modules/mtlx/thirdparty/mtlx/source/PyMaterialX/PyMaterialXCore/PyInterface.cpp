//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <PyMaterialX/PyMaterialX.h>

#include <MaterialXCore/Interface.h>

#include <MaterialXCore/Node.h>

namespace py = pybind11;
namespace mx = MaterialX;

#define BIND_INTERFACE_TYPE_INSTANCE(NAME, T)                                                                                                           \
.def("_setInputValue" #NAME, &mx::InterfaceElement::setInputValue<T>, py::arg("name"), py::arg("value"), py::arg("type") = mx::EMPTY_STRING)

void bindPyInterface(py::module& mod)
{
    py::class_<mx::PortElement, mx::PortElementPtr, mx::ValueElement>(mod, "PortElement")
        .def("setNodeName", &mx::PortElement::setNodeName)
        .def("getNodeName", &mx::PortElement::getNodeName)
        .def("setNodeGraphString", &mx::PortElement::setNodeGraphString)
        .def("hasNodeGraphString", &mx::PortElement::hasNodeGraphString)
        .def("getNodeGraphString", &mx::PortElement::getNodeGraphString)
        .def("setOutputString", &mx::PortElement::setOutputString)
        .def("hasOutputString", &mx::PortElement::hasOutputString)
        .def("getOutputString", &mx::PortElement::getOutputString)
        .def("setChannels", &mx::PortElement::setChannels)
        .def("getChannels", &mx::PortElement::getChannels)
        .def("setConnectedNode", &mx::PortElement::setConnectedNode)
        .def("getConnectedNode", &mx::PortElement::getConnectedNode)
        .def("setConnectedOutput", &mx::PortElement::setConnectedOutput)
        .def("getConnectedOutput", &mx::PortElement::getConnectedOutput);

    py::class_<mx::Input, mx::InputPtr, mx::PortElement>(mod, "Input")
        .def("setDefaultGeomPropString", &mx::Input::setDefaultGeomPropString)
        .def("hasDefaultGeomPropString", &mx::Input::hasDefaultGeomPropString)
        .def("getDefaultGeomPropString", &mx::Input::getDefaultGeomPropString)
        .def("getDefaultGeomProp", &mx::Input::getDefaultGeomProp)
        .def("getConnectedNode", &mx::Input::getConnectedNode)
        .def("getInterfaceInput", &mx::Input::getInterfaceInput)
        .def_readonly_static("CATEGORY", &mx::Input::CATEGORY);

    py::class_<mx::Output, mx::OutputPtr, mx::PortElement>(mod, "Output")
        .def("hasUpstreamCycle", &mx::Output::hasUpstreamCycle)
        .def_readonly_static("CATEGORY", &mx::Output::CATEGORY)
        .def_readonly_static("DEFAULT_INPUT_ATTRIBUTE", &mx::Output::DEFAULT_INPUT_ATTRIBUTE);

    py::class_<mx::InterfaceElement, mx::InterfaceElementPtr, mx::TypedElement>(mod, "InterfaceElement")
        .def("setNodeDefString", &mx::InterfaceElement::setNodeDefString)
        .def("hasNodeDefString", &mx::InterfaceElement::hasNodeDefString)
        .def("getNodeDefString", &mx::InterfaceElement::getNodeDefString)
        .def("addInput", &mx::InterfaceElement::addInput,
            py::arg("name") = mx::EMPTY_STRING, py::arg("type") = mx::DEFAULT_TYPE_STRING)
        .def("getInput", &mx::InterfaceElement::getInput)
        .def("getInputs", &mx::InterfaceElement::getInputs)
        .def("getInputCount", &mx::InterfaceElement::getInputCount)
        .def("removeInput", &mx::InterfaceElement::removeInput)
        .def("getActiveInput", &mx::InterfaceElement::getActiveInput)
        .def("getActiveInputs", &mx::InterfaceElement::getActiveInputs)
        .def("addOutput", &mx::InterfaceElement::addOutput,
            py::arg("name") = mx::EMPTY_STRING, py::arg("type") = mx::DEFAULT_TYPE_STRING)
        .def("getOutput", &mx::InterfaceElement::getOutput)
        .def("getOutputs", &mx::InterfaceElement::getOutputs)
        .def("getOutputCount", &mx::InterfaceElement::getOutputCount)
        .def("removeOutput", &mx::InterfaceElement::removeOutput)
        .def("getActiveOutput", &mx::InterfaceElement::getActiveOutput)
        .def("getActiveOutputs", &mx::InterfaceElement::getActiveOutputs)
        .def("setConnectedOutput", &mx::InterfaceElement::setConnectedOutput)
        .def("getConnectedOutput", &mx::InterfaceElement::getConnectedOutput)
        .def("addToken", &mx::InterfaceElement::addToken,
            py::arg("name") = mx::DEFAULT_TYPE_STRING)
        .def("getToken", &mx::InterfaceElement::getToken)
        .def("getTokens", &mx::InterfaceElement::getTokens)
        .def("removeToken", &mx::InterfaceElement::removeToken)
        .def("getActiveToken", &mx::InterfaceElement::getActiveToken)
        .def("getActiveTokens", &mx::InterfaceElement::getActiveTokens)
        .def("getActiveValueElement", &mx::InterfaceElement::getActiveValueElement)
        .def("getActiveValueElements", &mx::InterfaceElement::getActiveValueElements)
        .def("_getInputValue", &mx::InterfaceElement::getInputValue)
        .def("setTokenValue", &mx::InterfaceElement::setTokenValue)
        .def("getTokenValue", &mx::InterfaceElement::getTokenValue)
        .def("setTarget", &mx::InterfaceElement::setTarget)
        .def("hasTarget", &mx::InterfaceElement::hasTarget)
        .def("getTarget", &mx::InterfaceElement::getTarget)
        .def("setVersionString", &mx::InterfaceElement::setVersionString)
        .def("hasVersionString", &mx::InterfaceElement::hasVersionString)
        .def("getVersionString", &mx::InterfaceElement::getVersionString)
        .def("setVersionIntegers", &mx::InterfaceElement::setVersionIntegers)
        .def("getVersionIntegers", &mx::InterfaceElement::getVersionIntegers)
        .def("setDefaultVersion", &mx::InterfaceElement::setDefaultVersion)
        .def("getDefaultVersion", &mx::InterfaceElement::getDefaultVersion)
        .def("getDeclaration", &mx::InterfaceElement::getDeclaration,
            py::arg("target") = mx::EMPTY_STRING)
        .def("clearContent", &mx::InterfaceElement::clearContent)
        .def("hasExactInputMatch", &mx::InterfaceElement::hasExactInputMatch,
            py::arg("declaration"), py::arg("message") = nullptr)
        BIND_INTERFACE_TYPE_INSTANCE(integer, int)
        BIND_INTERFACE_TYPE_INSTANCE(boolean, bool)
        BIND_INTERFACE_TYPE_INSTANCE(float, float)
        BIND_INTERFACE_TYPE_INSTANCE(color3, mx::Color3)
        BIND_INTERFACE_TYPE_INSTANCE(color4, mx::Color4)
        BIND_INTERFACE_TYPE_INSTANCE(vector2, mx::Vector2)
        BIND_INTERFACE_TYPE_INSTANCE(vector3, mx::Vector3)
        BIND_INTERFACE_TYPE_INSTANCE(vector4, mx::Vector4)
        BIND_INTERFACE_TYPE_INSTANCE(matrix33, mx::Matrix33)
        BIND_INTERFACE_TYPE_INSTANCE(matrix44, mx::Matrix44)
        BIND_INTERFACE_TYPE_INSTANCE(string, std::string)
        BIND_INTERFACE_TYPE_INSTANCE(integerarray, mx::IntVec)
        BIND_INTERFACE_TYPE_INSTANCE(booleanarray, mx::BoolVec)
        BIND_INTERFACE_TYPE_INSTANCE(floatarray, mx::FloatVec)
        BIND_INTERFACE_TYPE_INSTANCE(stringarray, mx::StringVec)
        .def_readonly_static("NODE_DEF_ATTRIBUTE", &mx::InterfaceElement::NODE_DEF_ATTRIBUTE);
}
