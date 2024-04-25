//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <PyMaterialX/PyMaterialX.h>

#include <MaterialXGenShader/UnitSystem.h>
#include <MaterialXGenShader/GenContext.h>
#include <MaterialXGenShader/ShaderGraph.h>

namespace py = pybind11;
namespace mx = MaterialX;

void bindPyUnitSystem(py::module& mod)
{
    py::class_<mx::UnitTransform>(mod, "UnitTransform")
        .def(py::init<const std::string&, const std::string&, const mx::TypeDesc*, const std::string&>())
        .def_readwrite("sourceUnit", &mx::UnitTransform::sourceUnit)
        .def_readwrite("targetUnit", &mx::UnitTransform::targetUnit)
        .def_readwrite("type", &mx::UnitTransform::type)
        .def_readwrite("unitType", &mx::UnitTransform::type);

    py::class_<mx::UnitSystem, mx::UnitSystemPtr>(mod, "UnitSystem")
        .def_static("create", &mx::UnitSystem::create)
        .def("getName", &mx::UnitSystem::getName)
        .def("loadLibrary", &mx::UnitSystem::loadLibrary)
        .def("supportsTransform", &mx::UnitSystem::supportsTransform)
        .def("setUnitConverterRegistry", &mx::UnitSystem::setUnitConverterRegistry)
        .def("getUnitConverterRegistry", &mx::UnitSystem::getUnitConverterRegistry);
}
