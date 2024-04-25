//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <PyMaterialX/PyMaterialX.h>

#include <MaterialXGenShader/TypeDesc.h>

namespace py = pybind11;
namespace mx = MaterialX;

void bindPyTypeDesc(py::module& mod)
{
    // Set nodelete as destructor on returned TypeDescs since they are owned
    // by the container they are stored in and should not be destroyed when 
    // garbage collected by the python interpreter
    py::class_<mx::TypeDesc, std::unique_ptr<MaterialX::TypeDesc, py::nodelete>>(mod, "TypeDesc")
        .def_static("get", &mx::TypeDesc::get)
        .def("getName", &mx::TypeDesc::getName)
        .def("getBaseType", &mx::TypeDesc::getBaseType)
        .def("getChannelIndex", &mx::TypeDesc::getChannelIndex)
        .def("getSemantic", &mx::TypeDesc::getSemantic)
        .def("getSize", &mx::TypeDesc::getSize)
        .def("isEditable", &mx::TypeDesc::isEditable)
        .def("isScalar", &mx::TypeDesc::isScalar)
        .def("isAggregate", &mx::TypeDesc::isAggregate)
        .def("isArray", &mx::TypeDesc::isArray)
        .def("isFloat2", &mx::TypeDesc::isFloat2)
        .def("isFloat3", &mx::TypeDesc::isFloat3)
        .def("isFloat4", &mx::TypeDesc::isFloat4);
}
