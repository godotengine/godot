//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <PyMaterialX/PyMaterialX.h>

#include <MaterialXRenderGlsl/GlslRenderer.h>

namespace py = pybind11;
namespace mx = MaterialX;

void bindPyGlslRenderer(py::module& mod)
{
    py::class_<mx::GlslRenderer, mx::ShaderRenderer, mx::GlslRendererPtr>(mod, "GlslRenderer")
        .def_static("create", &mx::GlslRenderer::create)
        .def("initialize", &mx::GlslRenderer::initialize, py::arg("renderContextHandle") = nullptr)
        .def("createProgram", static_cast<void (mx::GlslRenderer::*)(const mx::ShaderPtr)>(&mx::GlslRenderer::createProgram))
        .def("createProgram", static_cast<void (mx::GlslRenderer::*)(const mx::GlslRenderer::StageMap&)>(&mx::GlslRenderer::createProgram))
        .def("validateInputs", &mx::GlslRenderer::validateInputs)
        .def("render", &mx::GlslRenderer::render)
        .def("renderTextureSpace", &mx::GlslRenderer::renderTextureSpace)
        .def("captureImage", &mx::GlslRenderer::captureImage)
        .def("getProgram", &mx::GlslRenderer::getProgram);
}
