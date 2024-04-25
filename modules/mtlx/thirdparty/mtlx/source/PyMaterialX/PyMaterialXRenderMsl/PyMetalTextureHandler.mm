//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <PyMaterialX/PyMaterialX.h>

#include <MaterialXRenderMsl/MetalTextureHandler.h>

namespace py = pybind11;
namespace mx = MaterialX;

void bindPyMetalTextureHandler(py::module& mod)
{
    py::class_<mx::MetalTextureHandler, mx::ImageHandler, mx::MetalTextureHandlerPtr>(mod, "MetalTextureHandler")
        .def_static("create", &mx::MetalTextureHandler::create)
        .def("bindImage", &mx::MetalTextureHandler::unbindImage)
        .def("unbindImage", &mx::MetalTextureHandler::unbindImage)
        .def("createRenderResources", &mx::MetalTextureHandler::createRenderResources)
        .def("releaseRenderResources", &mx::MetalTextureHandler::releaseRenderResources,
            py::arg("image") = nullptr);
}
