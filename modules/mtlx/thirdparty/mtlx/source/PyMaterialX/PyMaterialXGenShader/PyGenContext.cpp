//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <PyMaterialX/PyMaterialX.h>

#include <MaterialXGenShader/GenContext.h>
#include <MaterialXGenShader/HwShaderGenerator.h>

namespace py = pybind11;
namespace mx = MaterialX;

void bindPyGenContext(py::module& mod)
{
    py::class_<mx::ApplicationVariableHandler>(mod, "ApplicationVariableHandler");

    py::class_<mx::GenContext, mx::GenContextPtr>(mod, "GenContext")
        .def(py::init<mx::ShaderGeneratorPtr>())
        .def("getShaderGenerator", &mx::GenContext::getShaderGenerator)
        .def("getOptions", static_cast<mx::GenOptions & (mx::GenContext::*)()>(&mx::GenContext::getOptions), py::return_value_policy::reference)
        .def("registerSourceCodeSearchPath", static_cast<void (mx::GenContext::*)(const mx::FilePath&)>(&mx::GenContext::registerSourceCodeSearchPath))
        .def("registerSourceCodeSearchPath", static_cast<void (mx::GenContext::*)(const mx::FileSearchPath&)>(&mx::GenContext::registerSourceCodeSearchPath))
        .def("resolveSourceFile", &mx::GenContext::resolveSourceFile)
        .def("pushUserData", &mx::GenContext::pushUserData)
        .def("setApplicationVariableHandler", &mx::GenContext::setApplicationVariableHandler)
        .def("getApplicationVariableHandler", &mx::GenContext::getApplicationVariableHandler);
}

void bindPyGenUserData(py::module& mod)
{
    py::class_<mx::GenUserData, mx::GenUserDataPtr>(mod, "GenUserData")
        .def("getSelf", static_cast<mx::GenUserDataPtr(mx::GenUserData::*)()>(&mx::GenUserData::getSelf));
}