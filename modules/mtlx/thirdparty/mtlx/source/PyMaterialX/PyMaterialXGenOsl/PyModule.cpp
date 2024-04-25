//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <PyMaterialX/PyMaterialX.h>

namespace py = pybind11;

void bindPyOslShaderGenerator(py::module& mod);

PYBIND11_MODULE(PyMaterialXGenOsl, mod)
{
    mod.doc() = "Module containing Python bindings for the MaterialXGenOsl library";

    // PyMaterialXGenOsl depends on types defined in PyMaterialXGenShader
    PYMATERIALX_IMPORT_MODULE(PyMaterialXGenShader);

    bindPyOslShaderGenerator(mod);
}
