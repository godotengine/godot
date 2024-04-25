//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <PyMaterialX/PyMaterialX.h>

#include <MaterialXRender/Camera.h>

namespace py = pybind11;
namespace mx = MaterialX;

void bindPyCamera(py::module& mod)
{
    py::class_<mx::Camera, mx::CameraPtr>(mod, "Camera")
        .def_static("create", &mx::Camera::create)
        .def("setWorldMatrix", &mx::Camera::setWorldMatrix)
        .def("getWorldMatrix", &mx::Camera::getWorldMatrix)
        .def("setViewMatrix", &mx::Camera::setViewMatrix)
        .def("getViewMatrix", &mx::Camera::getViewMatrix)
        .def("setProjectionMatrix", &mx::Camera::setProjectionMatrix)
        .def("getProjectionMatrix", &mx::Camera::getProjectionMatrix)
        .def("getWorldViewProjMatrix", &mx::Camera::getWorldViewProjMatrix)
        .def("getViewPosition", &mx::Camera::getViewPosition)
        .def("getViewDirection", &mx::Camera::getViewDirection)
        .def("setViewportSize", &mx::Camera::setViewportSize)
        .def("getViewportSize", &mx::Camera::getViewportSize)
        .def("projectToViewport", &mx::Camera::projectToViewport)
        .def("unprojectFromViewport", &mx::Camera::unprojectFromViewport)
        .def_static("createViewMatrix", &mx::Camera::createViewMatrix)
        .def_static("createPerspectiveMatrix", &mx::Camera::createPerspectiveMatrix)
        .def_static("createOrthographicMatrix", &mx::Camera::createOrthographicMatrix)
        .def_static("transformPointPerspective", &mx::Camera::transformPointPerspective);
}
