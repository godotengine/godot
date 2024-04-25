//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <PyMaterialX/PyMaterialX.h>

namespace py = pybind11;

void bindPyFile(py::module& mod);
void bindPyXmlIo(py::module& mod);
void bindPyUtil(py::module& mod);

PYBIND11_MODULE(PyMaterialXFormat, mod)
{
    mod.doc() = "Module containing Python bindings for the MaterialXFormat library";

    // PyMaterialXFormat depends on types defined in PyMaterialXCore
    PYMATERIALX_IMPORT_MODULE(PyMaterialXCore);

    bindPyFile(mod);
    bindPyXmlIo(mod);
    bindPyUtil(mod);
}
