//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <PyMaterialX/PyMaterialX.h>

#include <MaterialXRenderMsl/MslPipelineStateObject.h>
#include <MaterialXRenderMsl/MetalFramebuffer.h>

namespace py = pybind11;
namespace mx = MaterialX;

void bindPyMslProgram(py::module& mod)
{
    py::class_<mx::MslProgram, mx::MslProgramPtr>(mod, "MslProgram")
        .def_static("create", &mx::MslProgram::create)
        .def("setStages", &mx::MslProgram::setStages)
        .def("addStage", &mx::MslProgram::addStage)
        .def("getStageSourceCode", &mx::MslProgram::getStageSourceCode)
        .def("getShader", &mx::MslProgram::getShader)
        .def("build", &mx::MslProgram::build)
        .def("prepareUsedResources", &mx::MslProgram::prepareUsedResources)
        .def("getUniformsList", &mx::MslProgram::getUniformsList)
        .def("getAttributesList", &mx::MslProgram::getAttributesList)
        .def("findInputs", &mx::MslProgram::findInputs)
        .def("bind", &mx::MslProgram::bind)
        .def("bindUniform", &mx::MslProgram::bindUniform)
        .def("bindAttribute", &mx::MslProgram::bindAttribute)
        .def("bindPartition", &mx::MslProgram::bindPartition)
        .def("bindMesh", &mx::MslProgram::bindMesh)
        .def("unbindGeometry", &mx::MslProgram::unbindGeometry)
        .def("bindTextures", &mx::MslProgram::bindTextures)
        .def("bindLighting", &mx::MslProgram::bindLighting)
        .def("bindViewInformation", &mx::MslProgram::bindViewInformation)
        .def("bindTimeAndFrame", &mx::MslProgram::bindTimeAndFrame,
             py::arg("time") = 1.0f, py::arg("frame") = 1.0f);

    py::class_<mx::MslProgram::Input>(mod, "Input")
        .def_readwrite("location", &mx::MslProgram::Input::location)
        .def_readwrite("size", &mx::MslProgram::Input::size)
        .def_readwrite("typeString", &mx::MslProgram::Input::typeString)
        .def_readwrite("value", &mx::MslProgram::Input::value)
        .def_readwrite("isConstant", &mx::MslProgram::Input::isConstant)
        .def_readwrite("path", &mx::MslProgram::Input::path)
        .def(py::init<int, int, int, std::string>());
}
