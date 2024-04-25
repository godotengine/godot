//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <PyMaterialX/PyMaterialX.h>

#include <MaterialXRender/ShaderRenderer.h>

namespace py = pybind11;
namespace mx = MaterialX;

void bindPyShaderRenderer(py::module& mod)
{
    py::class_<mx::ShaderRenderer, mx::ShaderRendererPtr>(mod, "ShaderRenderer")
        .def("initialize", &mx::ShaderRenderer::initialize, py::arg("renderContextHandle") = nullptr)
        .def("setCamera", &mx::ShaderRenderer::setCamera)
        .def("getCamera", &mx::ShaderRenderer::getCamera)
        .def("setImageHandler", &mx::ShaderRenderer::setImageHandler)
        .def("getImageHandler", &mx::ShaderRenderer::getImageHandler)
        .def("setLightHandler", &mx::ShaderRenderer::setLightHandler)
        .def("getLightHandler", &mx::ShaderRenderer::getLightHandler)
        .def("setGeometryHandler", &mx::ShaderRenderer::setGeometryHandler)
        .def("getGeometryHandler", &mx::ShaderRenderer::getGeometryHandler)
        .def("createProgram", static_cast<void (mx::ShaderRenderer::*)(mx::ShaderPtr)>(&mx::ShaderRenderer::createProgram))
        .def("createProgram", static_cast<void (mx::ShaderRenderer::*)(const mx::ShaderRenderer::StageMap&)>(&mx::ShaderRenderer::createProgram))
        .def("validateInputs", &mx::ShaderRenderer::validateInputs)
        .def("updateUniform", &mx::ShaderRenderer::updateUniform)
        .def("setSize", &mx::ShaderRenderer::setSize)
        .def("render", &mx::ShaderRenderer::render);

    static py::exception<mx::ExceptionRenderError> pyExceptionRenderError(mod, "ExceptionRenderError");

    py::register_exception_translator(
        [](std::exception_ptr errPtr)
        {
            try
            {
                if (errPtr != NULL)
                    std::rethrow_exception(errPtr);
            }
            catch (const mx::ExceptionRenderError& err)
            {
                std::string errorMsg = err.what();
                for (std::string error : err.errorLog())
                {
                    errorMsg += "\n" + error;
                }
                PyErr_SetString(PyExc_LookupError, errorMsg.c_str());
            }
        }
    );
}
