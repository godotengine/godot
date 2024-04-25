//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <PyMaterialX/PyMaterialX.h>

#include <MaterialXRender/TinyObjLoader.h>

namespace py = pybind11;
namespace mx = MaterialX;

void bindPyTinyObjLoader(py::module& mod)
{
    py::class_<mx::TinyObjLoader, mx::TinyObjLoaderPtr, mx::GeometryLoader>(mod, "TinyObjLoader")
        .def_static("create", &mx::TinyObjLoader::create)
        .def(py::init<>())
        .def("load", &mx::TinyObjLoader::load);
}
