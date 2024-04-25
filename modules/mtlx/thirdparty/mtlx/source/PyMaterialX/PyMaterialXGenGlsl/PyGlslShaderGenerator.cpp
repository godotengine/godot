//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <PyMaterialX/PyMaterialX.h>

#include <MaterialXGenGlsl/GlslShaderGenerator.h>
#include <MaterialXGenGlsl/GlslResourceBindingContext.h>
#include <MaterialXGenGlsl/EsslShaderGenerator.h>
#include <MaterialXGenGlsl/VkShaderGenerator.h>
#include <MaterialXGenShader/Shader.h>
#include <MaterialXGenShader/ShaderGenerator.h>

namespace py = pybind11;
namespace mx = MaterialX;

// GLSL shader generator bindings

void bindPyGlslShaderGenerator(py::module& mod)
{
    py::class_<mx::GlslShaderGenerator, mx::HwShaderGenerator, mx::GlslShaderGeneratorPtr>(mod, "GlslShaderGenerator")
        .def_static("create", &mx::GlslShaderGenerator::create)
        .def(py::init<>())
        .def("generate", &mx::GlslShaderGenerator::generate)
        .def("getTarget", &mx::GlslShaderGenerator::getTarget)
        .def("getVersion", &mx::GlslShaderGenerator::getVersion);
}

void bindPyGlslResourceBindingContext(py::module &mod)
{
    py::class_<mx::GlslResourceBindingContext, mx::HwResourceBindingContext, mx::GlslResourceBindingContextPtr>(mod, "GlslResourceBindingContext")
        .def_static("create", &mx::GlslResourceBindingContext::create)
        .def(py::init<size_t, size_t>())
        .def("emitDirectives", &mx::GlslResourceBindingContext::emitDirectives)
        .def("emitResourceBindings", &mx::GlslResourceBindingContext::emitResourceBindings);
}

// Essl shader generator bindings

void bindPyEsslShaderGenerator(py::module& mod)
{
    py::class_<mx::EsslShaderGenerator, mx::GlslShaderGenerator, mx::EsslShaderGeneratorPtr>(mod, "EsslShaderGenerator")
        .def_static("create", &mx::EsslShaderGenerator::create)
        .def(py::init<>())
        .def("generate", &mx::EsslShaderGenerator::generate)
        .def("getTarget", &mx::EsslShaderGenerator::getTarget)
        .def("getVersion", &mx::EsslShaderGenerator::getVersion);
}

// Glsl Vulkan shader generator bindings

void bindPyVkShaderGenerator(py::module& mod)
{
    py::class_<mx::VkShaderGenerator, mx::GlslShaderGenerator, mx::VkShaderGeneratorPtr>(mod, "VkShaderGenerator")
        .def_static("create", &mx::VkShaderGenerator::create)
        .def(py::init<>())
        .def("generate", &mx::VkShaderGenerator::generate)
        .def("getTarget", &mx::VkShaderGenerator::getTarget)
        .def("getVersion", &mx::VkShaderGenerator::getVersion);
}
