//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <PyMaterialX/PyMaterialX.h>

namespace py = pybind11;

void bindPyOslRenderer(py::module& mod);

PYBIND11_MODULE(PyMaterialXRenderOsl, mod)
{
    mod.doc() = "Module containing Python bindings for the MaterialXRenderOsl library";

    // PyMaterialXRenderOsl depends on types defined in PyMaterialXRender
    PYMATERIALX_IMPORT_MODULE(PyMaterialXRender);

    bindPyOslRenderer(mod);
}
