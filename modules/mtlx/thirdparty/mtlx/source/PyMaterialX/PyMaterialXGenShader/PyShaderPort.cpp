//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <PyMaterialX/PyMaterialX.h>

#include <MaterialXGenShader/ShaderNode.h>

namespace py = pybind11;
namespace mx = MaterialX;

void bindPyShaderPort(py::module& mod)
{
    py::class_<mx::ShaderPort, mx::ShaderPortPtr>(mod, "ShaderPort")
        .def("setType", &mx::ShaderPort::setType)
        .def("getType", &mx::ShaderPort::getType)
        .def("setName", &mx::ShaderPort::setName)
        .def("getName", &mx::ShaderPort::getName)
        .def("getFullName", &mx::ShaderPort::getFullName)
        .def("setVariable", &mx::ShaderPort::setVariable)
        .def("getVariable", &mx::ShaderPort::getVariable)
        .def("setSemantic", &mx::ShaderPort::setSemantic)
        .def("getSemantic", &mx::ShaderPort::getSemantic)
        .def("setValue", &mx::ShaderPort::setValue)
        .def("getValue", &mx::ShaderPort::getValue)
        .def("getValueString", &mx::ShaderPort::getValueString)
        .def("setGeomProp", &mx::ShaderPort::setGeomProp)
        .def("getGeomProp", &mx::ShaderPort::getGeomProp)
        .def("setPath", &mx::ShaderPort::setPath)
        .def("getPath", &mx::ShaderPort::getPath)
        .def("setUnit", &mx::ShaderPort::setUnit)
        .def("getUnit", &mx::ShaderPort::getUnit)
        .def("setColorSpace", &mx::ShaderPort::setColorSpace)
        .def("getColorSpace", &mx::ShaderPort::getColorSpace)
        .def("isUniform", &mx::ShaderPort::isUniform)
        .def("isEmitted", &mx::ShaderPort::isEmitted);
}
