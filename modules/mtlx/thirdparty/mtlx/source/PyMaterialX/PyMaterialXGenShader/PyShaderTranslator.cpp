//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <PyMaterialX/PyMaterialX.h>

#include <MaterialXGenShader/ShaderTranslator.h>

namespace py = pybind11;
namespace mx = MaterialX;

void bindPyShaderTranslator(py::module& mod)
{
    py::class_<mx::ShaderTranslator, mx::ShaderTranslatorPtr>(mod, "ShaderTranslator")
        .def_static("create", &mx::ShaderTranslator::create)
        .def("translateShader", &mx::ShaderTranslator::translateShader)
        .def("translateAllMaterials", &mx::ShaderTranslator::translateAllMaterials);
}
