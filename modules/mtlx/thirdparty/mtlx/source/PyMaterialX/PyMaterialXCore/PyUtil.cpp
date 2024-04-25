//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <PyMaterialX/PyMaterialX.h>

#include <MaterialXCore/Node.h>
#include <MaterialXCore/Util.h>

#include <MaterialXFormat/File.h>

namespace py = pybind11;
namespace mx = MaterialX;

void bindPyUtil(py::module& mod)
{
    mod.def("getVersionString", &mx::getVersionString);
    mod.def("getVersionIntegers", &mx::getVersionIntegers);
    mod.def("createValidName", &mx::createValidName, py::arg("name"), py::arg("replaceChar") = '_');
    mod.def("isValidName", &mx::isValidName);
    mod.def("incrementName", &mx::incrementName);
    mod.def("splitString", &mx::splitString);
    mod.def("joinStrings", &mx::joinStrings);
    mod.def("replaceSubstrings", &mx::replaceSubstrings);
    mod.def("stringStartsWith", &mx::stringStartsWith);
    mod.def("stringEndsWith", &mx::stringEndsWith);
    mod.def("splitNamePath", &mx::splitNamePath);
    mod.def("createNamePath", &mx::createNamePath);
    mod.def("parentNamePath", &mx::parentNamePath);
}
