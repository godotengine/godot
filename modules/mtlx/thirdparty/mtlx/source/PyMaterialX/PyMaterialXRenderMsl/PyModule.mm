//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <PyMaterialX/PyMaterialX.h>

namespace py = pybind11;

void bindPyMslProgram(py::module& mod);
void bindPyMslRenderer(py::module& mod);
void bindPyMetalTextureHandler(py::module& mod);
void bindPyTextureBaker(py::module& mod);

PYBIND11_MODULE(PyMaterialXRenderMsl, mod)
{
    mod.doc() = "Module containing Python bindings for the MaterialXRenderMsl library";

    // PyMaterialXRenderMsl depends on types defined in PyMaterialXRender
    PYMATERIALX_IMPORT_MODULE(PyMaterialXRender);

    bindPyMslProgram(mod);
    bindPyMslRenderer(mod);
    bindPyMetalTextureHandler(mod);
    bindPyTextureBaker(mod);
}
