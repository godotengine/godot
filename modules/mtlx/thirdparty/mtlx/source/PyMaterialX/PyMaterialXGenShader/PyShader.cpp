//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <PyMaterialX/PyMaterialX.h>

#include <MaterialXGenShader/Shader.h>

#include <string>

namespace py = pybind11;
namespace mx = MaterialX;

void bindPyShader(py::module& mod)
{
    // Note: py::return_value_policy::reference was needed because getStage returns a
    // ShaderStage& and without this parameter it would return a copy and not a
    // reference
    py::class_<mx::Shader, mx::ShaderPtr>(mod, "Shader")
        .def(py::init<const std::string&, mx::ShaderGraphPtr>())
        .def("getName", &mx::Shader::getName)
        .def("hasStage", &mx::Shader::hasStage)
        .def("numStages", &mx::Shader::numStages)
        .def("getStage", static_cast<mx::ShaderStage& (mx::Shader::*)(size_t)>(&mx::Shader::getStage), py::return_value_policy::reference)
        .def("getStage", static_cast<mx::ShaderStage& (mx::Shader::*)(const std::string&)>(&mx::Shader::getStage), py::return_value_policy::reference)
        .def("getSourceCode", &mx::Shader::getSourceCode)
        .def("hasAttribute", &mx::Shader::hasAttribute)
        .def("getAttribute", &mx::Shader::getAttribute)
        .def("setAttribute", static_cast<void (mx::Shader::*)(const std::string&)>(&mx::Shader::setAttribute))
        .def("setAttribute", static_cast<void (mx::Shader::*)(const std::string&, mx::ValuePtr)>(&mx::Shader::setAttribute));
}
