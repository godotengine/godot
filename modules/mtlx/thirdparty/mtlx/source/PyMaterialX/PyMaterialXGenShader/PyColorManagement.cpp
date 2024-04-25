//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <PyMaterialX/PyMaterialX.h>

#include <MaterialXGenShader/ColorManagementSystem.h>
#include <MaterialXGenShader/DefaultColorManagementSystem.h>
#include <MaterialXGenShader/GenContext.h>
#include <MaterialXGenShader/ShaderGraph.h>

namespace py = pybind11;
namespace mx = MaterialX;

class PyColorManagementSystem : public mx::ColorManagementSystem
{
  public:
    PyColorManagementSystem()
    {
    }

    const std::string& getName() const override
    {
        PYBIND11_OVERLOAD_PURE(
            const std::string&,
            mx::ColorManagementSystem,
            getName
        );
    }

  protected:
    mx::NodeDefPtr getNodeDef(const mx::ColorSpaceTransform& transform) const override
    {
        PYBIND11_OVERLOAD_PURE(
            mx::NodeDefPtr,
            mx::ColorManagementSystem,
            getNodeDef,
            transform
        );
    }
};

void bindPyColorManagement(py::module& mod)
{
    py::class_<mx::ColorSpaceTransform>(mod, "ColorSpaceTransform")
        .def(py::init<const std::string&, const std::string&, const mx::TypeDesc*>())
        .def_readwrite("sourceSpace", &mx::ColorSpaceTransform::sourceSpace)
        .def_readwrite("targetSpace", &mx::ColorSpaceTransform::targetSpace)
        .def_readwrite("type", &mx::ColorSpaceTransform::type);

    py::class_<mx::ColorManagementSystem, PyColorManagementSystem, mx::ColorManagementSystemPtr>(mod, "ColorManagementSystem")
        .def(py::init<>())
        .def("getName", &mx::ColorManagementSystem::getName)
        .def("loadLibrary", &mx::ColorManagementSystem::loadLibrary)
        .def("supportsTransform", &mx::ColorManagementSystem::supportsTransform);

    py::class_<mx::DefaultColorManagementSystem, mx::DefaultColorManagementSystemPtr, mx::ColorManagementSystem>(mod, "DefaultColorManagementSystem")
        .def_static("create", &mx::DefaultColorManagementSystem::create)
        .def("getName", &mx::DefaultColorManagementSystem::getName);
}
