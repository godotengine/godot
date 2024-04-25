//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <PyMaterialX/PyMaterialX.h>

namespace py = pybind11;

void bindPyMesh(py::module& mod);
void bindPyGeometryHandler(py::module& mod);
void bindPyLightHandler(py::module& mod);
void bindPyImage(py::module& mod);
void bindPyImageHandler(py::module& mod);
void bindPyStbImageLoader(py::module& mod);
#ifdef MATERIALX_BUILD_OIIO
void bindPyOiioImageLoader(py::module& mod);
#endif
void bindPyTinyObjLoader(py::module& mod);
void bindPyCamera(py::module& mod);
void bindPyShaderRenderer(py::module& mod);
void bindPyCgltfLoader(py::module& mod);

PYBIND11_MODULE(PyMaterialXRender, mod)
{
    mod.doc() = "Module containing Python bindings for the MaterialXRender library";

    // PyMaterialXRender depends on types defined in PyMaterialXCore
    PYMATERIALX_IMPORT_MODULE(PyMaterialXCore);

    bindPyMesh(mod);
    bindPyGeometryHandler(mod);
    bindPyLightHandler(mod);
    bindPyImage(mod);
    bindPyImageHandler(mod);
    bindPyStbImageLoader(mod);
#ifdef MATERIALX_BUILD_OIIO
    bindPyOiioImageLoader(mod);
#endif
    bindPyTinyObjLoader(mod);
    bindPyCamera(mod);
    bindPyShaderRenderer(mod);
    bindPyCgltfLoader(mod);
}
