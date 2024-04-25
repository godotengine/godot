//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <PyMaterialX/PyMaterialX.h>

namespace py = pybind11;

void bindPyGlslShaderGenerator(py::module& mod);
void bindPyGlslResourceBindingContext(py::module &mod);
void bindPyEsslShaderGenerator(py::module& mod);
void bindPyVkShaderGenerator(py::module& mod);

PYBIND11_MODULE(PyMaterialXGenGlsl, mod)
{
    mod.doc() = "Module containing Python bindings for the MaterialXGenGlsl library";

    // PyMaterialXGenGlsl depends on types defined in PyMaterialXGenShader
    PYMATERIALX_IMPORT_MODULE(PyMaterialXGenShader);

    bindPyGlslShaderGenerator(mod);
    bindPyGlslResourceBindingContext(mod);

    bindPyEsslShaderGenerator(mod);
    bindPyVkShaderGenerator(mod);
}
