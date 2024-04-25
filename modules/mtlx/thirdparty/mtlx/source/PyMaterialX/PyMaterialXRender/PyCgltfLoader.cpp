//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <PyMaterialX/PyMaterialX.h>
#include <MaterialXRender/CgltfLoader.h>

namespace py = pybind11;
namespace mx = MaterialX;

void bindPyCgltfLoader(py::module& mod)
{
    py::class_<mx::CgltfLoader, mx::CgltfLoaderPtr, mx::GeometryLoader>(mod, "CgltfLoader")
        .def_static("create", &mx::CgltfLoader::create)
        .def(py::init<>())
        .def("load", &mx::CgltfLoader::load);
}
