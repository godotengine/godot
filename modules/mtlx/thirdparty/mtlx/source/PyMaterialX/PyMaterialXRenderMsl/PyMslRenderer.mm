//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <PyMaterialX/PyMaterialX.h>

#include <MaterialXRenderMsl/MslRenderer.h>
#include <MaterialXRenderMsl/MetalFramebuffer.h>

namespace py = pybind11;
namespace mx = MaterialX;

void bindPyMslRenderer(py::module& mod)
{
    py::class_<mx::MslRenderer, mx::ShaderRenderer, mx::MslRendererPtr>(mod, "MslRenderer")
        .def_static("create", &mx::MslRenderer::create)
        .def("initialize", &mx::MslRenderer::initialize, py::arg("renderContextHandle") = nullptr)
        .def("createProgram", static_cast<void (mx::MslRenderer::*)(const mx::ShaderPtr)>(&mx::MslRenderer::createProgram))
        .def("createProgram", static_cast<void (mx::MslRenderer::*)(const mx::MslRenderer::StageMap&)>(&mx::MslRenderer::createProgram))
        .def("validateInputs", &mx::MslRenderer::validateInputs)
        .def("render", &mx::MslRenderer::render)
        .def("renderTextureSpace", &mx::MslRenderer::renderTextureSpace)
        .def("captureImage", &mx::MslRenderer::captureImage)
        .def("getProgram", &mx::MslRenderer::getProgram);
}
