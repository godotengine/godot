//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <PyMaterialX/PyMaterialX.h>

#include <MaterialXRenderOsl/OslRenderer.h>

namespace py = pybind11;
namespace mx = MaterialX;

void bindPyOslRenderer(py::module& mod)
{
    py::class_<mx::OslRenderer, mx::ShaderRenderer, mx::OslRendererPtr>(mod, "OslRenderer")
        .def_static("create", &mx::OslRenderer::create)
        .def_readwrite_static("OSL_CLOSURE_COLOR_STRING", &mx::OslRenderer::OSL_CLOSURE_COLOR_STRING)
        .def("initialize", &mx::OslRenderer::initialize, py::arg("renderContextHandle") = nullptr)
        .def("createProgram", static_cast<void (mx::OslRenderer::*)(const mx::ShaderPtr)>(&mx::OslRenderer::createProgram))
        .def("createProgram", static_cast<void (mx::OslRenderer::*)(const mx::OslRenderer::StageMap&)>(&mx::OslRenderer::createProgram))
        .def("validateInputs", &mx::OslRenderer::validateInputs)
        .def("render", &mx::OslRenderer::render)
        .def("captureImage", &mx::OslRenderer::captureImage)
        .def("setOslCompilerExecutable", &mx::OslRenderer::setOslCompilerExecutable)
        .def("setOslIncludePath", &mx::OslRenderer::setOslIncludePath)
        .def("setOslOutputFilePath", &mx::OslRenderer::setOslOutputFilePath)
        .def("setShaderParameterOverrides", &mx::OslRenderer::setShaderParameterOverrides)
        .def("setOslShaderOutput", &mx::OslRenderer::setOslShaderOutput)
        .def("setOslTestShadeExecutable", &mx::OslRenderer::setOslTestShadeExecutable)
        .def("setOslTestRenderExecutable", &mx::OslRenderer::setOslTestRenderExecutable)
        .def("setOslTestRenderSceneTemplateFile", &mx::OslRenderer::setOslTestRenderSceneTemplateFile)
        .def("setOslShaderName", &mx::OslRenderer::setOslShaderName)
        .def("setOslUtilityOSOPath", &mx::OslRenderer::setOslUtilityOSOPath)
        .def("useTestRender", &mx::OslRenderer::useTestRender)
        .def("compileOSL", &mx::OslRenderer::compileOSL);
}
