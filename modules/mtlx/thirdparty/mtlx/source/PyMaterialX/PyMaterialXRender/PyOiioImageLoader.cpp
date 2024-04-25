//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <PyMaterialX/PyMaterialX.h>

#include <MaterialXRender/OiioImageLoader.h>

namespace py = pybind11;
namespace mx = MaterialX;

void bindPyOiioImageLoader(py::module& mod)
{
    py::class_<mx::OiioImageLoader, mx::ImageLoader, mx::OiioImageLoaderPtr>(mod, "OiioImageLoader")
        .def_static("create", &mx::OiioImageLoader::create)
        .def(py::init<>())
        .def("saveImage", &mx::OiioImageLoader::saveImage)
        .def("loadImage", &mx::OiioImageLoader::loadImage);
}
