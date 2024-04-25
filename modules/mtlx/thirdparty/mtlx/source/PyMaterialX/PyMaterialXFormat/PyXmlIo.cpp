//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <PyMaterialX/PyMaterialX.h>

#include <MaterialXFormat/XmlIo.h>
#include <MaterialXCore/Document.h>

namespace py = pybind11;
namespace mx = MaterialX;

void bindPyXmlIo(py::module& mod)
{
    py::class_<mx::XmlReadOptions>(mod, "XmlReadOptions")
        .def(py::init())
        .def_readwrite("readXIncludeFunction", &mx::XmlReadOptions::readXIncludeFunction)
        .def_readwrite("readComments", &mx::XmlReadOptions::readComments)
        .def_readwrite("readNewlines", &mx::XmlReadOptions::readNewlines)
        .def_readwrite("upgradeVersion", &mx::XmlReadOptions::upgradeVersion)        
        .def_readwrite("parentXIncludes", &mx::XmlReadOptions::parentXIncludes);

    py::class_<mx::XmlWriteOptions>(mod, "XmlWriteOptions")
        .def(py::init())
        .def_readwrite("writeXIncludeEnable", &mx::XmlWriteOptions::writeXIncludeEnable)
        .def_readwrite("elementPredicate", &mx::XmlWriteOptions::elementPredicate);

    mod.def("readFromXmlFileBase", &mx::readFromXmlFile,
        py::arg("doc"), py::arg("filename"), py::arg("searchPath") = mx::FileSearchPath(), py::arg("readOptions") = (mx::XmlReadOptions*) nullptr);
    mod.def("readFromXmlString", &mx::readFromXmlString,
        py::arg("doc"), py::arg("str"), py::arg("searchPath") = mx::FileSearchPath(), py::arg("readOptions") = (mx::XmlReadOptions*) nullptr);
    mod.def("writeToXmlFile", mx::writeToXmlFile,
        py::arg("doc"), py::arg("filename"), py::arg("writeOptions") = (mx::XmlWriteOptions*) nullptr);
    mod.def("writeToXmlString", mx::writeToXmlString,
        py::arg("doc"), py::arg("writeOptions") = nullptr);
    mod.def("prependXInclude", mx::prependXInclude);

    mod.def("getEnvironmentPath", &mx::getEnvironmentPath,
        py::arg("sep") = mx::PATH_LIST_SEPARATOR);

    mod.attr("PATH_LIST_SEPARATOR") = mx::PATH_LIST_SEPARATOR;
    mod.attr("MATERIALX_SEARCH_PATH_ENV_VAR") = mx::MATERIALX_SEARCH_PATH_ENV_VAR;

    py::register_exception<mx::ExceptionParseError>(mod, "ExceptionParseError");
    py::register_exception<mx::ExceptionFileMissing>(mod, "ExceptionFileMissing");
}
