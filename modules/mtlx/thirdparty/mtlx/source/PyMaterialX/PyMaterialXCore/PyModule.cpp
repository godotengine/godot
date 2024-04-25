//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <PyMaterialX/PyMaterialX.h>

namespace py = pybind11;

void bindPyDefinition(py::module& mod);
void bindPyDocument(py::module& mod);
void bindPyElement(py::module& mod);
void bindPyException(py::module& mod);
void bindPyGeom(py::module& mod);
void bindPyInterface(py::module& mod);
void bindPyLook(py::module& mod);
void bindPyMaterial(py::module& mod);
void bindPyNode(py::module& mod);
void bindPyProperty(py::module& mod);
void bindPyTraversal(py::module& mod);
void bindPyTypes(py::module& mod);
void bindPyUnitConverters(py::module& mod);
void bindPyUtil(py::module& mod);
void bindPyValue(py::module& mod);
void bindPyVariant(py::module& mod);

PYBIND11_MODULE(PyMaterialXCore, mod)
{
    mod.doc() = "Module containing Python bindings for the MaterialXCore library";

    bindPyElement(mod);
    bindPyTraversal(mod);
    bindPyInterface(mod);
    bindPyValue(mod);
    bindPyGeom(mod);
    bindPyProperty(mod);
    bindPyLook(mod);
    bindPyDefinition(mod);
    bindPyNode(mod);
    bindPyMaterial(mod);
    bindPyVariant(mod);
    bindPyDocument(mod);
    bindPyTypes(mod);
    bindPyUnitConverters(mod);
    bindPyUtil(mod);
    bindPyException(mod);
}
