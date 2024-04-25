//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <PyMaterialX/PyMaterialX.h>

#include <MaterialXGenShader/GenContext.h>
#include <MaterialXGenShader/GenUserData.h>
#include <MaterialXGenShader/HwShaderGenerator.h>

namespace py = pybind11;
namespace mx = MaterialX;

void bindPyHwShaderGenerator(py::module& mod)
{
    mod.attr("VERTEX_STAGE") = mx::Stage::VERTEX;

    mod.attr("HW_VERTEX_INPUTS") = mx::HW::VERTEX_INPUTS;
    mod.attr("HW_VERTEX_DATA") = mx::HW::VERTEX_DATA;
    mod.attr("HW_PRIVATE_UNIFORMS") = mx::HW::PRIVATE_UNIFORMS;
    mod.attr("HW_PUBLIC_UNIFORMS") = mx::HW::PUBLIC_UNIFORMS;
    mod.attr("HW_LIGHT_DATA") = mx::HW::LIGHT_DATA;
    mod.attr("HW_PIXEL_OUTPUTS") = mx::HW::PIXEL_OUTPUTS;
    mod.attr("HW_ATTR_TRANSPARENT") =  mx::HW::ATTR_TRANSPARENT;

    py::class_<mx::HwShaderGenerator, mx::ShaderGenerator, mx::HwShaderGeneratorPtr>(mod, "HwShaderGenerator")
        .def("getClosureContexts", &mx::HwShaderGenerator::getClosureContexts)
        .def("bindLightShader", &mx::HwShaderGenerator::bindLightShader)
        .def("unbindLightShader", &mx::HwShaderGenerator::unbindLightShader)
        .def("unbindLightShaders", &mx::HwShaderGenerator::unbindLightShaders);
}

void bindPyHwResourceBindingContext(py::module& mod)
{
    py::class_<mx::HwResourceBindingContext, mx::GenUserData, mx::HwResourceBindingContextPtr>(mod, "HwResourceBindingContext")
        .def("emitDirectives", &mx::HwResourceBindingContext::emitDirectives)
        .def("emitResourceBindings", &mx::HwResourceBindingContext::emitResourceBindings);
}
